"""
data.py — Data pipeline for the Pattern Engine.

Clean-break replacement for prepare.py. Downloads OHLCV data via
yfinance, computes all feature columns (returns, candlestick,
supplementary, sector), and performs strict temporal splitting.

Does NOT depend on prepare.py — fully self-contained.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

from pattern_engine.features import (
    RETURN_WINDOWS, RETURN_COLS, SUPPLEMENT_COLS,
    FORWARD_WINDOWS, FORWARD_RETURN_COLS, FORWARD_BINARY_COLS,
)
from pattern_engine.candlestick import add_candlestick_features
from pattern_engine.sector import TICKERS, SECTOR_MAP, compute_sector_features


class DataLoader:
    """Downloads market data and builds the analogue database.

    Args:
        tickers: list of ticker symbols (default: 52-ticker universe)
        start_date: earliest date for data download
        data_dir: directory for cached data and outputs
    """

    def __init__(self, tickers: list[str] = None,
                 start_date: str = "2000-01-01",
                 data_dir: str = "data"):
        self.tickers = tickers or TICKERS
        self.start_date = start_date
        self.data_dir = Path(data_dir)

    def download(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Download OHLCV data via yfinance, cache to data/raw/.

        Args:
            force_refresh: re-download even if cached

        Returns:
            dict mapping ticker → DataFrame
        """
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        end = datetime.now().strftime("%Y-%m-%d")

        all_data = {}
        for ticker in self.tickers:
            cache_path = raw_dir / f"{ticker}.csv"

            if cache_path.exists() and not force_refresh:
                df = pd.read_csv(cache_path, parse_dates=["Date"])
                print(f"  [cached] {ticker}: {len(df)} rows")
            else:
                print(f"  [downloading] {ticker}...", end=" ")
                try:
                    df = yf.download(ticker, start=self.start_date, end=end,
                                     progress=False)
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] if col[1] == '' or col[1] == ticker
                                      else col[0] for col in df.columns]
                    df.to_csv(cache_path, index=False)
                    print(f"{len(df)} rows")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

            all_data[ticker] = df

        return all_data

    def compute_features(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute all feature columns for each ticker.

        Computes: return vectors, supplementary features, candlestick
        encoding (1d/3d/5d), and forward targets.

        Args:
            raw_data: dict mapping ticker → OHLCV DataFrame

        Returns:
            Combined DataFrame with all features
        """
        all_rows = []

        for ticker, df in raw_data.items():
            print(f"  Processing {ticker}...")
            df = self._compute_ticker_features(df, ticker)

            # Drop rows with incomplete data
            required_cols = (RETURN_COLS + SUPPLEMENT_COLS +
                             FORWARD_RETURN_COLS + FORWARD_BINARY_COLS +
                             ["Date", "Ticker", "Open", "High", "Low", "Close"])
            available = [c for c in required_cols if c in df.columns]
            subset = df[available].dropna().reset_index(drop=True)

            if len(subset) < 100:
                print(f"    Skipping {ticker}: insufficient data ({len(subset)} rows)")
                continue

            all_rows.append(subset)

        db = pd.concat(all_rows, ignore_index=True)
        print(f"\n  Database built: {len(db):,} rows, "
              f"{db['Ticker'].nunique()} tickers")

        # Add sector features (requires full database)
        db = compute_sector_features(db)

        return db

    def build_database(self, train_end: str = "2023-12-31",
                       val_start: str = "2024-01-01",
                       val_end: str = "2024-12-31",
                       test_start: str = "2025-01-01",
                       force_refresh: bool = False
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Full pipeline: download → features → temporal split.

        Returns:
            (full_db, train_db, val_db, test_db)
        """
        print("\n[1/3] Downloading market data...")
        raw_data = self.download(force_refresh=force_refresh)

        print("\n[2/3] Computing features...")
        full_db = self.compute_features(raw_data)

        print("\n[3/3] Temporal split...")
        train_db, val_db, test_db = self.temporal_split(
            full_db, train_end, val_start, val_end, test_start
        )

        # Save to parquet
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        full_db.to_parquet(processed_dir / "full_db.parquet", index=False)
        train_db.to_parquet(processed_dir / "train_db.parquet", index=False)
        val_db.to_parquet(processed_dir / "val_db.parquet", index=False)
        test_db.to_parquet(processed_dir / "test_db.parquet", index=False)

        print(f"\n  Saved to {processed_dir}/")
        return full_db, train_db, val_db, test_db

    @staticmethod
    def temporal_split(db: pd.DataFrame, train_end: str,
                       val_start: str, val_end: str,
                       test_start: str
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Strict chronological split — no data leakage.

        Returns:
            (train_db, val_db, test_db)
        """
        db = db.copy()
        db["Date"] = pd.to_datetime(db["Date"])

        train_db = db[db["Date"] <= train_end].copy()
        val_db = db[(db["Date"] >= val_start) & (db["Date"] <= val_end)].copy()
        test_db = db[db["Date"] >= test_start].copy()

        for name, split in [("train", train_db), ("val", val_db), ("test", test_db)]:
            print(f"  {name:5s}: {len(split):>8,} rows | "
                  f"{split['Ticker'].nunique()} tickers | "
                  f"{split['Date'].min().strftime('%Y-%m-%d')} → "
                  f"{split['Date'].max().strftime('%Y-%m-%d')}")

        return train_db, val_db, test_db

    def _compute_ticker_features(self, df: pd.DataFrame,
                                 ticker: str) -> pd.DataFrame:
        """Compute all features for a single ticker."""
        df = df.copy()
        df = df.sort_values("Date").reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"])

        # Trailing returns
        for w in RETURN_WINDOWS:
            df[f"ret_{w}d"] = df["Close"].pct_change(w)

        # Forward returns and binary targets
        for w in FORWARD_WINDOWS:
            df[f"fwd_{w}d"] = df["Close"].shift(-w) / df["Close"] - 1
            df[f"fwd_{w}d_up"] = (df[f"fwd_{w}d"] > 0).astype(int)

        # Supplementary features
        df = self._compute_supplementary(df)

        # Candlestick features (1d/3d/5d)
        df = add_candlestick_features(df)

        df["Ticker"] = ticker
        return df

    @staticmethod
    def _compute_supplementary(df: pd.DataFrame) -> pd.DataFrame:
        """Compute supplementary features (volatility, volume, momentum)."""
        from ta.momentum import RSIIndicator
        from ta.volatility import AverageTrueRange
        from ta.trend import SMAIndicator

        df = df.copy()

        # Volatility
        df["vol_10d"] = df["Close"].pct_change().rolling(10).std()
        df["vol_30d"] = df["Close"].pct_change().rolling(30).std()
        df["vol_ratio"] = df["vol_10d"] / (df["vol_30d"] + 1e-8)

        # Volume profile
        vol_mean_20 = df["Volume"].rolling(20).mean()
        df["vol_abnormal"] = (df["Volume"] - vol_mean_20) / (vol_mean_20 + 1)

        # Momentum
        df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
        df["atr_14"] = AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=14
        ).average_true_range()

        # Price vs moving averages
        df["price_vs_sma20"] = (
            df["Close"] / SMAIndicator(df["Close"], window=20).sma_indicator() - 1
        )
        df["price_vs_sma50"] = (
            df["Close"] / SMAIndicator(df["Close"], window=50).sma_indicator() - 1
        )

        return df
