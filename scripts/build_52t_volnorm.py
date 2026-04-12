"""
scripts/build_52t_volnorm.py — Build 52T dataset with VOL_NORM_COLS.

Option A from Session 2026-04-02 escalation:
  The 52T dataset in data/processed/ was built pre-M9 with RETURNS_ONLY_COLS.
  The baseline Fold6 BSS=+0.00103 could not be reproduced, and the VOL_NORM_COLS
  feature set (M9) has not been tested at 52T scale.

Strategy:
  - 48/52 tickers: filter directly from data/train_db.parquet (585T, already has VOL_NORM_COLS)
  - 4 missing tickers (ABBV, META, PYPL, TSLA): rebuild from data/<T>.csv using
    prepare.py feature pipeline (compute_return_vector + compute_vol_normalized_features)
  - Save to data/52t_volnorm/ (separate from data/processed/ to avoid overwriting legacy)

Output:
  data/52t_volnorm/train_db.parquet   — 52T, VOL_NORM_COLS, dates <= 2023-12-31
  data/52t_volnorm/val_db.parquet     — 52T, VOL_NORM_COLS, dates 2024-01-01 to 2024-12-31
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import feature pipeline directly from prepare.py (pure functions, safe to import)
from prepare import (
    compute_return_vector,
    compute_supplementary_features,
    compute_vol_normalized_features,
    RETURN_WINDOWS,
    FORWARD_WINDOWS,
)

DATA_DIR       = project_root / "data"
OUTPUT_DIR     = project_root / "data" / "52t_volnorm"
TRAIN_END      = "2023-12-31"
VAL_START      = "2024-01-01"
VAL_END        = "2024-12-31"

TICKERS_52 = [
    'AAPL', 'ABBV', 'ADBE', 'AMD',  'AMGN', 'AMZN', 'AVGO', 'AXP',  'BA',   'BAC',
    'BRK-B','CAT',  'COST', 'CRM',  'CSCO', 'CVX',  'DIS',  'GE',   'GILD', 'GOOGL',
    'GS',   'HD',   'INTC', 'ISRG', 'JNJ',  'JPM',  'KO',   'LLY',  'MA',   'META',
    'MRK',  'MS',   'MSFT', 'MU',   'NFLX', 'NVDA', 'ORCL', 'PEP',  'PFE',  'PG',
    'PYPL', 'QCOM', 'QQQ',  'SPY',  'TMO',  'TSLA', 'TXN',  'UNH',  'V',    'WFC',
    'WMT',  'XOM',
]

# These 4 are absent from the 585T parquet — must be built from CSV caches
MISSING_FROM_585T = ['ABBV', 'META', 'PYPL', 'TSLA']

# Columns required in the output (mirrors 585T parquet schema)
REQUIRED_COLS = (
    [f"ret_{w}d" for w in RETURN_WINDOWS]
    + [f"ret_{w}d_norm" for w in RETURN_WINDOWS]
    + ["vol_10d", "vol_30d", "vol_ratio", "vol_abnormal",
       "rsi_14", "atr_14", "price_vs_sma20", "price_vs_sma50"]
    + [f"fwd_{w}d" for w in FORWARD_WINDOWS]
    + [f"fwd_{w}d_up" for w in FORWARD_WINDOWS]
    + ["Date", "Ticker", "Open", "High", "Low", "Close"]
)


def _build_from_csv(ticker: str) -> pd.DataFrame:
    """Build VOL_NORM features for a single ticker from its CSV cache."""
    csv_path = DATA_DIR / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV cache not found: {csv_path}\n"
            "Download with: yf.download('{ticker}', start='2000-01-01')"
        )

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # Handle yfinance MultiIndex columns (Price/Ticker level)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Some yfinance CSVs have Price/Ticker rows as header artifacts — clean them
    df = df[pd.to_datetime(df["Date"], errors="coerce").notna()].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"]).copy()

    df = compute_return_vector(df, ticker)
    df = compute_supplementary_features(df)
    df = compute_vol_normalized_features(df)

    # Drop rows with any NaN in required columns
    available = [c for c in REQUIRED_COLS if c in df.columns]
    df = df[available].dropna().reset_index(drop=True)

    print(f"    {ticker}: {len(df):,} rows from CSV "
          f"({df['Date'].min().date()} → {df['Date'].max().date()})")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  BUILD 52T DATASET WITH VOL_NORM_COLS")
    print("=" * 65)
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Train:  dates <= {TRAIN_END}")
    print(f"  Val:    {VAL_START} to {VAL_END}")

    # ── Part 1: Load 48 tickers from 585T parquet ─────────────────────────
    tickers_from_parquet = [t for t in TICKERS_52 if t not in MISSING_FROM_585T]
    print(f"\n[1/3] Filtering 585T parquet → {len(tickers_from_parquet)} tickers...")

    train_585 = pd.read_parquet(DATA_DIR / "train_db.parquet")
    val_585   = pd.read_parquet(DATA_DIR / "val_db.parquet")

    # Combine, filter to 52T tickers, then re-split
    combined_585 = pd.concat([train_585, val_585], ignore_index=True)
    combined_585["Date"] = pd.to_datetime(combined_585["Date"])
    subset_585 = combined_585[combined_585["Ticker"].isin(tickers_from_parquet)].copy()

    # Keep only required columns that exist
    avail_cols = [c for c in REQUIRED_COLS if c in subset_585.columns]
    missing_cols = [c for c in REQUIRED_COLS if c not in subset_585.columns]
    if missing_cols:
        print(f"  WARNING: These required cols absent from 585T parquet: {missing_cols}")
    subset_585 = subset_585[avail_cols].copy()

    n_tickers = subset_585["Ticker"].nunique()
    print(f"  Got {len(subset_585):,} rows, {n_tickers} tickers "
          f"({subset_585['Date'].min().date()} → {subset_585['Date'].max().date()})")

    # ── Part 2: Build 4 missing tickers from CSV ──────────────────────────
    print(f"\n[2/3] Building {len(MISSING_FROM_585T)} missing tickers from CSV caches...")
    csv_frames: list[pd.DataFrame] = []
    for ticker in MISSING_FROM_585T:
        try:
            df = _build_from_csv(ticker)
            # Align columns to same schema
            avail = [c for c in avail_cols if c in df.columns]
            csv_frames.append(df[avail])
        except FileNotFoundError as e:
            print(f"  SKIPPING {ticker}: {e}")

    # ── Part 3: Combine and split ─────────────────────────────────────────
    print(f"\n[3/3] Combining and splitting...")
    all_frames = [subset_585] + csv_frames
    full = pd.concat(all_frames, ignore_index=True)
    full["Date"] = pd.to_datetime(full["Date"])
    full = full.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # Temporal split
    train_df = full[full["Date"] <= TRAIN_END].copy()
    val_df   = full[(full["Date"] >= VAL_START) & (full["Date"] <= VAL_END)].copy()

    print(f"\n  Full:  {len(full):,} rows, {full['Ticker'].nunique()} tickers")
    print(f"  Train: {len(train_df):,} rows  ({train_df['Date'].min().date()} → {train_df['Date'].max().date()})")
    print(f"  Val:   {len(val_df):,} rows  ({val_df['Date'].min().date()} → {val_df['Date'].max().date()})")

    # Sanity check: confirm VOL_NORM_COLS are present
    norm_cols = [f"ret_{w}d_norm" for w in RETURN_WINDOWS]
    if all(c in train_df.columns for c in norm_cols):
        nan_frac = train_df[norm_cols].isna().mean().mean()
        print(f"  VOL_NORM_COLS: present, NaN fraction = {nan_frac:.4f}")
    else:
        missing_norm = [c for c in norm_cols if c not in train_df.columns]
        print(f"  ERROR: VOL_NORM_COLS missing: {missing_norm}")
        sys.exit(1)

    # Save
    out_train = OUTPUT_DIR / "train_db.parquet"
    out_val   = OUTPUT_DIR / "val_db.parquet"
    train_df.to_parquet(out_train, index=False)
    val_df.to_parquet(out_val, index=False)

    print(f"\n  Saved: {out_train}")
    print(f"  Saved: {out_val}")

    # ── Verify tickers ────────────────────────────────────────────────────
    final_tickers = sorted(train_df["Ticker"].unique().tolist())
    missing_final = [t for t in TICKERS_52 if t not in final_tickers]
    print(f"\n  Final ticker count: {len(final_tickers)}")
    if missing_final:
        print(f"  WARNING — tickers not in output: {missing_final}")
    else:
        print(f"  All 52 tickers present.")

    print("\n" + "=" * 65)
    print("  BUILD COMPLETE")
    print(f"  Next: re-run validate_52t_best_config.py with data_dir=52t_volnorm")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
