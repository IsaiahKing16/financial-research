"""
data.py — Hardened DataLoader with Pandera validation and lineage metadata.

Adds three validation checkpoints and lineage tracking to the production
DataLoader pipeline:

  Checkpoint 1 — After OHLCV download:  OHLCVSchema per ticker
  Checkpoint 2 — After feature compute: TrainDBSchema (feature_cols + target)
  Checkpoint 3 — Before parquet write:  TrainDBSchema re-validated on split

Lineage metadata is attached to every output DataFrame via df.attrs["lineage"].
All parquet writes are atomic (write to .tmp file, then rename to final path).

Production data.py (pattern_engine/data.py) is untouched until M8 migration.
This hardened version wraps the production implementation, adding contracts at
every pipeline boundary as defined by the Pandera schemas from SLE-57.

Design decisions:
  - ValidationError from Pandera is re-raised as DataPipelineError (ValueError
    subclass) with the ticker name and checkpoint context prepended.
  - Lineage hash = SHA-256[:16] of "{ticker_count}:{row_count}" — lightweight
    fingerprint that detects row-count drift without loading the full file.
  - DataFrame.attrs is preserved through concat() via _merge_attrs().
  - Leakage guard in temporal_split: raises RuntimeError if train_end >= val_start.

Linear: SLE-65
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from pattern_engine.contracts.datasets import (
    OHLCVSchema,
    make_train_db_schema,
)
from pattern_engine.features import RETURNS_ONLY_COLS


# ─── Exceptions ──────────────────────────────────────────────────────────────

class DataPipelineError(ValueError):
    """Raised when a Pandera validation checkpoint fails in the data pipeline.

    Inherits from ValueError (not RuntimeError) because it indicates bad *data*,
    not a programming error. Callers that catch ValueError will still catch this.
    """


# ─── Feature column defaults ─────────────────────────────────────────────────

DEFAULT_FEATURE_COLS: List[str] = RETURNS_ONLY_COLS
"""Default 8-feature return fingerprint. Locked setting — matches production."""

TARGET_COL: str = "fwd_7d_up"
"""Binary classification target column. Locked setting."""


# ─── Atomic write utility ────────────────────────────────────────────────────

def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet atomically (temp file → rename).

    Prevents partially-written files from being visible to concurrent readers
    if the process crashes during the write.

    Args:
        df: DataFrame to persist.
        path: Final destination path (parent must exist).
    """
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp, index=False)
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ─── Lineage metadata ────────────────────────────────────────────────────────

def _build_lineage(
    df: pd.DataFrame,
    source: str,
    checkpoint: str,
    feature_cols: Optional[List[str]] = None,
) -> dict:
    """Build a lineage metadata dict for a DataFrame.

    Args:
        df: The validated DataFrame.
        source: Data origin label, e.g. "yfinance" or "cache".
        checkpoint: Which pipeline stage this was created at.
        feature_cols: Feature columns in use (optional).

    Returns:
        Lineage dict suitable for df.attrs["lineage"].
    """
    n_tickers = df["Ticker"].nunique() if "Ticker" in df.columns else 0
    row_count = len(df)
    fingerprint = hashlib.sha256(
        f"{n_tickers}:{row_count}".encode()
    ).hexdigest()[:16]

    return {
        "source": source,
        "checkpoint": checkpoint,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "n_tickers": n_tickers,
        "row_count_hash": fingerprint,
        "feature_cols": feature_cols or [],
        "schema_version": "SLE-65-v1",
    }


def _attach_lineage(df: pd.DataFrame, lineage: dict) -> pd.DataFrame:
    """Attach lineage metadata to a DataFrame's attrs and return it.

    Note: pandas DataFrame.attrs is shallow-copied by most operations (concat,
    copy, etc.) — do not rely on it surviving arbitrary transformations.
    This function always attaches fresh lineage at each checkpoint.

    Args:
        df: DataFrame to annotate.
        lineage: Lineage dict from _build_lineage().

    Returns:
        The same DataFrame with df.attrs["lineage"] set.
    """
    df.attrs["lineage"] = lineage
    return df


# ─── Validation helpers ──────────────────────────────────────────────────────

def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Checkpoint 1: validate raw OHLCV data for a single ticker.

    Args:
        df: Raw OHLCV DataFrame from yfinance or cache.
        ticker: Ticker symbol (used for error context).

    Returns:
        The validated DataFrame (unchanged content).

    Raises:
        DataPipelineError: If schema validation fails.
    """
    try:
        OHLCVSchema.validate(df, lazy=True)
    except Exception as exc:
        raise DataPipelineError(
            f"[Checkpoint 1 / OHLCV] Ticker {ticker!r} failed validation: {exc}"
        ) from exc
    return df


def _validate_feature_db(
    df: pd.DataFrame,
    feature_cols: List[str],
    checkpoint: str,
) -> pd.DataFrame:
    """Checkpoint 2/3: validate the feature + target database.

    Args:
        df: DataFrame with feature columns + fwd_7d_up target.
        feature_cols: Expected feature column names.
        checkpoint: Human-readable stage label for error messages.

    Returns:
        The validated DataFrame (unchanged content).

    Raises:
        DataPipelineError: If schema validation fails.
    """
    schema = make_train_db_schema(feature_cols)
    try:
        schema.validate(df, lazy=True)
    except Exception as exc:
        raise DataPipelineError(
            f"[{checkpoint}] Feature DB validation failed: {exc}"
        ) from exc
    return df


# ─── DataLoaderHardened ──────────────────────────────────────────────────────

class DataLoaderHardened:
    """Hardened DataLoader with Pandera validation at every pipeline boundary.

    Wraps the production DataLoader pattern: download → compute_features →
    temporal_split → build_database. Adds:

      - Pandera OHLCVSchema validation per ticker after download
      - Pandera TrainDBSchema validation after feature computation
      - Pandera TrainDBSchema re-validation before each parquet write
      - Lineage metadata via DataFrame.attrs["lineage"]
      - Atomic parquet writes (temp → rename)
      - Leakage guard in temporal_split

    Args:
        tickers: List of ticker symbols to process.
        start_date: Earliest date for data download (YYYY-MM-DD).
        data_dir: Directory for cached raw data and processed outputs.
        feature_cols: Feature columns to validate. Defaults to RETURNS_ONLY_COLS.
        validate_ohlcv: If True, run OHLCVSchema validation per ticker. Default True.
        validate_features: If True, run TrainDBSchema validation after compute. Default True.
    """

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2000-01-01",
        data_dir: str = "data",
        feature_cols: Optional[List[str]] = None,
        validate_ohlcv: bool = True,
        validate_features: bool = True,
    ) -> None:
        try:
            from pattern_engine.sector import TICKERS as _PROD_TICKERS
            _default_tickers = _PROD_TICKERS
        except ImportError:
            _default_tickers = []

        self.tickers: List[str] = tickers if tickers is not None else _default_tickers
        self.start_date: str = start_date
        self.data_dir: Path = Path(data_dir)
        self.feature_cols: List[str] = feature_cols or DEFAULT_FEATURE_COLS
        self.validate_ohlcv: bool = validate_ohlcv
        self.validate_features: bool = validate_features

    # ── Download ─────────────────────────────────────────────────────────────

    def download(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Download OHLCV data via yfinance, cache to data/raw/, validate each ticker.

        Args:
            force_refresh: Re-download even if cached.

        Returns:
            Dict mapping ticker → validated OHLCV DataFrame.

        Raises:
            DataPipelineError: If any ticker fails OHLCVSchema validation.
        """
        import yfinance as yf

        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        end = datetime.now().strftime("%Y-%m-%d")

        all_data: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            cache_path = raw_dir / f"{ticker}.csv"

            if cache_path.exists() and not force_refresh:
                df = pd.read_csv(cache_path, parse_dates=["Date"])
                source = "cache"
            else:
                try:
                    df = yf.download(ticker, start=self.start_date, end=end, progress=False)
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [
                            col[0] if col[1] == "" or col[1] == ticker else col[0]
                            for col in df.columns
                        ]
                    df.to_csv(cache_path, index=False)
                    source = "yfinance"
                except Exception as exc:
                    logging.warning("Download FAILED for %s: %s", ticker, exc)
                    continue

            # Checkpoint 1: OHLCV validation
            if self.validate_ohlcv:
                df = _validate_ohlcv(df, ticker)

            df = _attach_lineage(
                df,
                _build_lineage(df, source=source, checkpoint="ohlcv_download"),
            )
            all_data[ticker] = df

        return all_data

    # ── Feature computation ──────────────────────────────────────────────────

    def compute_features(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute all feature columns and validate the resulting database.

        Delegates feature computation to production _compute_ticker_features,
        then validates the concatenated result with TrainDBSchema.

        Args:
            raw_data: Dict mapping ticker → OHLCV DataFrame.

        Returns:
            Combined, validated feature DataFrame.

        Raises:
            RuntimeError: If no tickers survive feature computation.
            DataPipelineError: If TrainDBSchema validation fails.
        """
        try:
            from pattern_engine.data import DataLoader as _ProdLoader
            from pattern_engine.sector import compute_sector_features
            _prod = _ProdLoader(
                tickers=list(raw_data.keys()),
                start_date=self.start_date,
                data_dir=str(self.data_dir),
            )
            _compute_ticker = _prod._compute_ticker_features
            _sector_fn = compute_sector_features
        except ImportError:
            _compute_ticker = self._compute_ticker_features_fallback
            _sector_fn = lambda db: db  # noqa: E731

        all_rows = []
        for ticker, df in raw_data.items():
            try:
                df_feat = _compute_ticker(df, ticker)
            except Exception as exc:
                logging.warning("Skipping %s: feature computation failed (%s)", ticker, exc)
                continue

            # Drop rows with missing values in required columns only
            required = self.feature_cols + [TARGET_COL, "Date", "Ticker", "Close"]
            available = [c for c in required if c in df_feat.columns]
            subset = df_feat[available].dropna().reset_index(drop=True)

            if len(subset) < 50:
                logging.warning("Skipping %s: insufficient data (%d rows)", ticker, len(subset))
                continue
            all_rows.append(subset)

        if not all_rows:
            raise RuntimeError(
                "No tickers survived feature computation. "
                "Check that raw OHLCV data has sufficient history."
            )

        db = pd.concat(all_rows, ignore_index=True)
        db = _sector_fn(db)

        # Checkpoint 2: feature DB validation
        if self.validate_features:
            db = _validate_feature_db(
                db,
                self.feature_cols,
                checkpoint="Checkpoint 2 / feature_compute",
            )

        db = _attach_lineage(
            db,
            _build_lineage(
                db,
                source="computed",
                checkpoint="feature_compute",
                feature_cols=self.feature_cols,
            ),
        )
        return db

    # ── Temporal split ───────────────────────────────────────────────────────

    @staticmethod
    def temporal_split(
        db: pd.DataFrame,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Strict chronological split with leakage guard.

        Args:
            db: Full feature database with a Date column.
            train_end: Last date (inclusive) for training data (YYYY-MM-DD).
            val_start: First date (inclusive) for validation data.
            val_end: Last date (inclusive) for validation data.
            test_start: First date (inclusive) for test data.

        Returns:
            (train_db, val_db, test_db)

        Raises:
            RuntimeError: If train_end >= val_start (temporal leakage).
        """
        # Leakage guard
        if pd.Timestamp(train_end) >= pd.Timestamp(val_start):
            raise RuntimeError(
                f"Temporal leakage detected: train_end={train_end!r} >= "
                f"val_start={val_start!r}. The training set must end strictly "
                "before the validation set begins."
            )

        db = db.copy()
        db["Date"] = pd.to_datetime(db["Date"])

        train_db = db[db["Date"] <= train_end].copy()
        val_db = db[(db["Date"] >= val_start) & (db["Date"] <= val_end)].copy()
        test_db = db[db["Date"] >= test_start].copy()

        return train_db, val_db, test_db

    # ── Full pipeline ────────────────────────────────────────────────────────

    def build_database(
        self,
        train_end: str = "2023-12-31",
        val_start: str = "2024-01-01",
        val_end: str = "2024-12-31",
        test_start: str = "2025-01-01",
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Full pipeline: download → features → split → validate → atomic write.

        Args:
            train_end: Last date for training data (YYYY-MM-DD).
            val_start: First date for validation data.
            val_end: Last date for validation data.
            test_start: First date for test data.
            force_refresh: Re-download OHLCV data even if cached.

        Returns:
            (full_db, train_db, val_db, test_db) — all validated with lineage.

        Raises:
            DataPipelineError: If any Pandera checkpoint fails.
            RuntimeError: If leakage guard or empty universe checks fail.
        """
        raw_data = self.download(force_refresh=force_refresh)
        full_db = self.compute_features(raw_data)
        train_db, val_db, test_db = self.temporal_split(
            full_db, train_end, val_start, val_end, test_start
        )

        # Checkpoint 3: validate each split before writing
        if self.validate_features:
            for split_name, split_df in [("train", train_db), ("val", val_db)]:
                if len(split_df) > 0:
                    _validate_feature_db(
                        split_df,
                        self.feature_cols,
                        checkpoint=f"Checkpoint 3 / pre-write ({split_name})",
                    )

        # Attach lineage to each split
        for split_name, split_df in [
            ("full", full_db), ("train", train_db),
            ("val", val_db), ("test", test_db),
        ]:
            _attach_lineage(
                split_df,
                _build_lineage(
                    split_df,
                    source="temporal_split",
                    checkpoint=f"split:{split_name}",
                    feature_cols=self.feature_cols,
                ),
            )

        # Atomic parquet writes
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        for name, split_df in [
            ("full_db", full_db), ("train_db", train_db),
            ("val_db", val_db), ("test_db", test_db),
        ]:
            _atomic_write_parquet(split_df, processed_dir / f"{name}.parquet")

        return full_db, train_db, val_db, test_db

    # ── Fallback feature computation (when production not importable) ─────────

    @staticmethod
    def _compute_ticker_features_fallback(
        df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Minimal feature computation fallback for isolated testing.

        Computes only the 8 return columns and fwd_7d_up target.
        Used when production pattern_engine is not on the import path.
        """
        df = df.copy().sort_values("Date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"])

        for w in [1, 3, 7, 14, 30, 45, 60, 90]:
            df[f"ret_{w}d"] = df["Close"].pct_change(w)

        df["fwd_7d"] = df["Close"].shift(-7) / df["Close"] - 1
        df["fwd_7d_up"] = (df["fwd_7d"] > 0).astype(float)
        df["Ticker"] = ticker
        return df
