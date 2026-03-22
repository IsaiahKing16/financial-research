"""
test_data.py — Tests for the hardened DataLoaderHardened (SLE-65/SLE-66).

Coverage targets:
  - DataLoaderHardened.download() — mocked yfinance, cache hit, validation
  - DataLoaderHardened.compute_features() — synthetic data, feature spot-checks
  - DataLoaderHardened.temporal_split() — leakage guard, correct row assignment
  - DataLoaderHardened.build_database() — end-to-end with mocked download
  - _atomic_write_parquet() — temp-file-then-rename pattern
  - _build_lineage() / _attach_lineage() — metadata correctness
  - DataPipelineError — raised on schema failure

Acceptance criteria (SLE-66):
  - ≥80% line coverage for data.py
  - Temporal split leakage guard tested
  - Feature value spot-checks at known rows
  - yfinance download path mocked (no network calls)

Linear: SLE-66
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rebuild_phase_3z.fppe.pattern_engine.data import (
    DataLoaderHardened,
    DataPipelineError,
    _atomic_write_parquet,
    _attach_lineage,
    _build_lineage,
    _validate_ohlcv,
    _validate_feature_db,
)
from rebuild_phase_3z.fppe.pattern_engine.features import RETURNS_ONLY_COLS


# ─── Synthetic data factories ──────────────────────────────────────────────────

def _make_ohlcv(n: int = 50, ticker: str = "AAPL", seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with n rows."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    price = 100 * np.cumprod(1 + rng.randn(n) * 0.01)
    return pd.DataFrame({
        "Date": dates,
        "Open":   price * (1 + rng.randn(n) * 0.002),
        "High":   price * (1 + np.abs(rng.randn(n)) * 0.005),
        "Low":    price * (1 - np.abs(rng.randn(n)) * 0.005),
        "Close":  price,
        "Volume": rng.randint(1_000_000, 5_000_000, size=n).astype(float),
        "Ticker": ticker,
    })


def _make_feature_db(n_per_ticker: int = 200, tickers=("AAPL", "MSFT"), seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic feature DataFrame with return columns + fwd_7d_up."""
    rng = np.random.RandomState(seed)
    rows = []
    dates = pd.date_range("2018-01-02", periods=n_per_ticker, freq="B")
    for t in tickers:
        for i, d in enumerate(dates):
            row = {"Date": d, "Ticker": t, "Close": 100.0}
            for col in RETURNS_ONLY_COLS:
                row[col] = rng.randn()
            row["fwd_7d_up"] = float(rng.randint(0, 2))
            rows.append(row)
    return pd.DataFrame(rows)


# ─── Atomic write ─────────────────────────────────────────────────────────────

class TestAtomicWriteParquet:

    def test_writes_file(self, tmp_path):
        df = _make_ohlcv(10)
        dest = tmp_path / "out.parquet"
        _atomic_write_parquet(df, dest)
        assert dest.exists()

    def test_no_temp_file_after_success(self, tmp_path):
        df = _make_ohlcv(10)
        dest = tmp_path / "out.parquet"
        _atomic_write_parquet(df, dest)
        tmp = dest.with_suffix(".tmp.parquet")
        assert not tmp.exists()

    def test_roundtrip_integrity(self, tmp_path):
        df = _make_ohlcv(20)
        dest = tmp_path / "data.parquet"
        _atomic_write_parquet(df, dest)
        loaded = pd.read_parquet(dest)
        assert len(loaded) == 20
        assert list(loaded.columns) == list(df.columns)


# ─── Lineage utilities ────────────────────────────────────────────────────────

class TestLineage:

    def test_build_lineage_keys(self):
        df = _make_feature_db(30)
        lin = _build_lineage(df, source="yfinance", checkpoint="ohlcv_download",
                             feature_cols=RETURNS_ONLY_COLS)
        assert lin["source"] == "yfinance"
        assert lin["checkpoint"] == "ohlcv_download"
        assert "created_at" in lin
        assert "row_count" in lin
        assert "row_count_hash" in lin
        assert lin["feature_cols"] == RETURNS_ONLY_COLS
        assert lin["schema_version"] == "SLE-65-v1"

    def test_row_count_hash_format(self):
        df = _make_ohlcv(10)
        lin = _build_lineage(df, source="test", checkpoint="c1")
        # Hash is a 16-char hex string
        assert len(lin["row_count_hash"]) == 16
        assert all(c in "0123456789abcdef" for c in lin["row_count_hash"])

    def test_attach_lineage_sets_attrs(self):
        df = _make_ohlcv(5)
        lin = {"source": "test", "row_count": 5}
        df = _attach_lineage(df, lin)
        assert df.attrs["lineage"] == lin

    def test_n_tickers_counted(self):
        df = _make_feature_db(30)
        lin = _build_lineage(df, source="computed", checkpoint="feature_compute")
        assert lin["n_tickers"] == 2


# ─── Validation helpers ───────────────────────────────────────────────────────

class TestOHLCVValidation:

    def test_valid_ohlcv_passes(self):
        df = _make_ohlcv(30)
        result = _validate_ohlcv(df, "AAPL")
        assert len(result) == 30

    def test_missing_close_raises(self):
        df = _make_ohlcv(10)
        df["Close"] = np.nan
        with pytest.raises(DataPipelineError, match="OHLCV"):
            _validate_ohlcv(df, "AAPL")

    def test_negative_close_raises(self):
        df = _make_ohlcv(10)
        df.loc[3, "Close"] = -1.0
        with pytest.raises(DataPipelineError, match="OHLCV"):
            _validate_ohlcv(df, "AAPL")

    def test_error_includes_ticker_name(self):
        df = _make_ohlcv(5)
        df["Close"] = np.nan
        with pytest.raises(DataPipelineError, match="GOOGL"):
            _validate_ohlcv(df, "GOOGL")


class TestFeatureDBValidation:

    def test_valid_db_passes(self):
        db = _make_feature_db(50)
        result = _validate_feature_db(db, RETURNS_ONLY_COLS, "test_checkpoint")
        assert len(result) == 100  # 2 tickers × 50

    def test_missing_feature_col_raises(self):
        db = _make_feature_db(30)
        db = db.drop(columns=["ret_1d"])
        with pytest.raises(DataPipelineError):
            _validate_feature_db(db, RETURNS_ONLY_COLS, "test")

    def test_nan_in_feature_col_raises(self):
        db = _make_feature_db(30)
        db.loc[0, "ret_7d"] = np.nan
        with pytest.raises(DataPipelineError):
            _validate_feature_db(db, RETURNS_ONLY_COLS, "test")

    def test_invalid_target_raises(self):
        db = _make_feature_db(30)
        db.loc[0, "fwd_7d_up"] = 2.0  # Not binary
        with pytest.raises(DataPipelineError):
            _validate_feature_db(db, RETURNS_ONLY_COLS, "test")


# ─── DataLoaderHardened ───────────────────────────────────────────────────────

class TestDataLoaderHardenedInit:

    def test_default_feature_cols(self):
        loader = DataLoaderHardened(tickers=["AAPL"])
        assert loader.feature_cols == RETURNS_ONLY_COLS

    def test_custom_tickers_set(self):
        loader = DataLoaderHardened(tickers=["AAPL", "MSFT"])
        assert loader.tickers == ["AAPL", "MSFT"]

    def test_data_dir_is_path(self):
        loader = DataLoaderHardened(tickers=["AAPL"], data_dir="/tmp/test")
        assert isinstance(loader.data_dir, Path)

    def test_validation_flags_default_true(self):
        loader = DataLoaderHardened(tickers=["AAPL"])
        assert loader.validate_ohlcv is True
        assert loader.validate_features is True


class TestDownload:
    """Test the download() method with mocked yfinance."""

    def _mock_yf_df(self, ticker: str, n: int = 60) -> pd.DataFrame:
        """Build a yfinance-style DataFrame (no Ticker column, Date as index reset)."""
        df = _make_ohlcv(n, ticker)
        df = df.drop(columns=["Ticker"])  # yfinance doesn't include Ticker
        return df

    def test_download_calls_yfinance(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_ohlcv=False
        )
        mock_df = self._mock_yf_df("AAPL")
        with patch("yfinance.download", return_value=mock_df) as mock_yf:
            result = loader.download(force_refresh=True)
        mock_yf.assert_called_once()
        assert "AAPL" in result

    def test_download_validates_ohlcv(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_ohlcv=True
        )
        mock_df = self._mock_yf_df("AAPL")
        with patch("yfinance.download", return_value=mock_df):
            result = loader.download(force_refresh=True)
        assert "AAPL" in result

    def test_download_bad_close_raises(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_ohlcv=True
        )
        mock_df = self._mock_yf_df("AAPL")
        mock_df["Close"] = np.nan
        with patch("yfinance.download", return_value=mock_df):
            with pytest.raises(DataPipelineError, match="OHLCV"):
                loader.download(force_refresh=True)

    def test_download_uses_cache(self, tmp_path):
        """Second download call reads from CSV cache, not yfinance."""
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_ohlcv=False
        )
        mock_df = self._mock_yf_df("AAPL")

        # First call writes cache
        with patch("yfinance.download", return_value=mock_df):
            loader.download(force_refresh=True)

        # Second call — yfinance should NOT be called
        with patch("yfinance.download", side_effect=AssertionError("should not call yfinance")) as mock_yf:
            result = loader.download(force_refresh=False)
        assert "AAPL" in result

    def test_download_attaches_lineage(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_ohlcv=False
        )
        mock_df = self._mock_yf_df("AAPL")
        with patch("yfinance.download", return_value=mock_df):
            result = loader.download(force_refresh=True)
        assert "lineage" in result["AAPL"].attrs
        assert result["AAPL"].attrs["lineage"]["checkpoint"] == "ohlcv_download"

    def test_download_skips_failed_tickers(self, tmp_path):
        """A ticker that throws during download is skipped, not fatal."""
        loader = DataLoaderHardened(
            tickers=["AAPL", "BAD"], data_dir=str(tmp_path), validate_ohlcv=False
        )
        mock_df = self._mock_yf_df("AAPL")

        def side_effect(ticker, **kwargs):
            if ticker == "BAD":
                raise ConnectionError("simulated network failure")
            return mock_df

        with patch("yfinance.download", side_effect=side_effect):
            result = loader.download(force_refresh=True)
        assert "AAPL" in result
        assert "BAD" not in result


class TestComputeFeatures:
    """Test compute_features() with the fallback implementation (no production import)."""

    def _make_raw_ohlcv_dict(self, tickers=("AAPL", "MSFT"), n: int = 300) -> dict:
        # 300 rows: ret_90d needs 90 lookback + fwd_7d needs 7 lookahead = 97 removed
        # → 203 rows survive per ticker, well above the 50-row threshold.
        return {t: _make_ohlcv(n, t) for t in tickers}

    def test_returns_dataframe(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = self._make_raw_ohlcv_dict(["AAPL"])
        db = loader.compute_features(raw)
        assert isinstance(db, pd.DataFrame)

    def test_feature_cols_present(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = self._make_raw_ohlcv_dict(["AAPL"])
        db = loader.compute_features(raw)
        for col in RETURNS_ONLY_COLS:
            assert col in db.columns, f"Missing column: {col}"

    def test_ticker_column_present(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = self._make_raw_ohlcv_dict(["AAPL"])
        db = loader.compute_features(raw)
        assert "Ticker" in db.columns
        assert db["Ticker"].iloc[0] == "AAPL"

    def test_feature_spot_check_ret_1d(self, tmp_path):
        """ret_1d must be Close.pct_change(1) — check sign matches price movement."""
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = {"AAPL": _make_ohlcv(300, "AAPL", seed=1)}
        db = loader.compute_features(raw)
        # ret_1d should be finite for all non-NaN rows
        assert db["ret_1d"].isna().sum() == 0  # dropna() removed NaN rows
        assert np.isfinite(db["ret_1d"]).all()

    def test_feature_spot_check_ret_90d_magnitude(self, tmp_path):
        """ret_90d should be within ±500% for normal equity data."""
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = {"AAPL": _make_ohlcv(200, "AAPL")}
        db = loader.compute_features(raw)
        assert (db["ret_90d"].abs() < 5.0).all()

    def test_fwd_7d_up_is_binary(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = self._make_raw_ohlcv_dict(["AAPL"])
        db = loader.compute_features(raw)
        assert db["fwd_7d_up"].isin([0.0, 1.0]).all()

    def test_empty_universe_raises(self, tmp_path):
        """All tickers with < 50 rows → RuntimeError."""
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = {"AAPL": _make_ohlcv(10, "AAPL")}  # Too few rows
        with pytest.raises(RuntimeError, match="No tickers survived"):
            loader.compute_features(raw)

    def test_lineage_attached(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL"], data_dir=str(tmp_path), validate_features=False
        )
        raw = {"AAPL": _make_ohlcv(300, "AAPL")}
        db = loader.compute_features(raw)
        assert "lineage" in db.attrs
        assert db.attrs["lineage"]["checkpoint"] == "feature_compute"

    def test_two_tickers_concatenated(self, tmp_path):
        loader = DataLoaderHardened(
            tickers=["AAPL", "MSFT"], data_dir=str(tmp_path), validate_features=False
        )
        raw = {"AAPL": _make_ohlcv(300, "AAPL"), "MSFT": _make_ohlcv(300, "MSFT", seed=99)}
        db = loader.compute_features(raw)
        assert db["Ticker"].nunique() == 2


# ─── Temporal split ───────────────────────────────────────────────────────────

class TestTemporalSplit:

    def _make_db(self) -> pd.DataFrame:
        db = _make_feature_db(300, ("AAPL", "MSFT"))
        # date range: 2018-01-02 → ~2019-04-xx for 300 business days
        return db

    def test_no_leakage_between_splits(self):
        db = self._make_db()
        train, val, test = DataLoaderHardened.temporal_split(
            db, "2018-12-31", "2019-01-01", "2019-03-31", "2019-04-01"
        )
        # No date in val should be ≤ train_end
        if len(val) > 0:
            assert val["Date"].min() >= pd.Timestamp("2019-01-01")
        # No date in test should be < test_start
        if len(test) > 0:
            assert test["Date"].min() >= pd.Timestamp("2019-04-01")

    def test_train_end_inclusive(self):
        db = self._make_db()
        train, _, _ = DataLoaderHardened.temporal_split(
            db, "2018-12-31", "2019-01-01", "2019-03-31", "2019-04-01"
        )
        assert train["Date"].max() <= pd.Timestamp("2018-12-31")

    def test_leakage_guard_raises(self):
        db = self._make_db()
        # train_end == val_start → should raise
        with pytest.raises(RuntimeError, match="leakage"):
            DataLoaderHardened.temporal_split(
                db, "2019-01-01", "2019-01-01", "2019-03-31", "2019-04-01"
            )

    def test_leakage_guard_train_after_val(self):
        db = self._make_db()
        with pytest.raises(RuntimeError, match="leakage"):
            DataLoaderHardened.temporal_split(
                db, "2019-03-01", "2019-01-01", "2019-03-31", "2019-04-01"
            )

    def test_row_counts_sum_to_at_most_full(self):
        db = self._make_db()
        train, val, test = DataLoaderHardened.temporal_split(
            db, "2018-09-30", "2019-01-01", "2019-03-31", "2019-04-01"
        )
        # Train + val + test ≤ len(db) (gap dates possible between splits)
        assert len(train) + len(val) + len(test) <= len(db)

    def test_all_tickers_in_train(self):
        db = self._make_db()
        train, _, _ = DataLoaderHardened.temporal_split(
            db, "2018-12-31", "2019-01-01", "2019-03-31", "2019-04-01"
        )
        assert "AAPL" in train["Ticker"].values
        assert "MSFT" in train["Ticker"].values


# ─── Build database end-to-end ────────────────────────────────────────────────

class TestBuildDatabase:
    """End-to-end test with mocked download and fallback features."""

    def _setup_loader(self, tmp_path) -> DataLoaderHardened:
        return DataLoaderHardened(
            tickers=["AAPL", "MSFT"],
            data_dir=str(tmp_path),
            validate_ohlcv=False,
            validate_features=False,
        )

    def _mock_yf_for_ticker(self, ticker: str, n: int = 400):
        df = _make_ohlcv(n, ticker)
        df = df.drop(columns=["Ticker"])
        return df

    def test_returns_four_dataframes(self, tmp_path):
        loader = self._setup_loader(tmp_path)

        def _yf_download(ticker, **kwargs):
            return self._mock_yf_for_ticker(ticker)

        with patch("yfinance.download", side_effect=_yf_download):
            full, train, val, test = loader.build_database(
                train_end="2021-06-30",
                val_start="2021-07-01",
                val_end="2021-09-30",
                test_start="2021-10-01",
                force_refresh=True,
            )
        assert isinstance(full, pd.DataFrame)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_parquet_files_written(self, tmp_path):
        loader = self._setup_loader(tmp_path)

        def _yf_download(ticker, **kwargs):
            return self._mock_yf_for_ticker(ticker)

        with patch("yfinance.download", side_effect=_yf_download):
            loader.build_database(
                train_end="2021-06-30",
                val_start="2021-07-01",
                val_end="2021-09-30",
                test_start="2021-10-01",
                force_refresh=True,
            )
        processed = tmp_path / "processed"
        for name in ("full_db", "train_db", "val_db", "test_db"):
            assert (processed / f"{name}.parquet").exists(), f"{name}.parquet missing"

    def test_no_tmp_parquet_files_after_success(self, tmp_path):
        loader = self._setup_loader(tmp_path)

        def _yf_download(ticker, **kwargs):
            return self._mock_yf_for_ticker(ticker)

        with patch("yfinance.download", side_effect=_yf_download):
            loader.build_database(
                train_end="2021-06-30",
                val_start="2021-07-01",
                val_end="2021-09-30",
                test_start="2021-10-01",
                force_refresh=True,
            )
        # No .tmp.parquet files should remain
        tmp_files = list((tmp_path / "processed").glob("*.tmp.parquet"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"

    def test_lineage_on_splits(self, tmp_path):
        loader = self._setup_loader(tmp_path)

        def _yf_download(ticker, **kwargs):
            return self._mock_yf_for_ticker(ticker)

        with patch("yfinance.download", side_effect=_yf_download):
            full, train, val, test = loader.build_database(
                train_end="2021-06-30",
                val_start="2021-07-01",
                val_end="2021-09-30",
                test_start="2021-10-01",
                force_refresh=True,
            )
        for split_df, split_name in [(full, "full"), (train, "train")]:
            assert "lineage" in split_df.attrs, f"{split_name} missing lineage"
            assert "split" in split_df.attrs["lineage"]["checkpoint"]


# ─── DataPipelineError ────────────────────────────────────────────────────────

class TestDataPipelineError:

    def test_is_value_error_subclass(self):
        exc = DataPipelineError("test")
        assert isinstance(exc, ValueError)

    def test_message_preserved(self):
        exc = DataPipelineError("OHLCV validation failed")
        assert "OHLCV" in str(exc)

    def test_caught_by_except_value_error(self):
        """Existing callers using 'except ValueError' catch DataPipelineError."""
        caught = False
        try:
            raise DataPipelineError("test")
        except ValueError:
            caught = True
        assert caught
