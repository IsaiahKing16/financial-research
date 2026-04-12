"""
scripts/build_52t_features.py — Add cross-sectional features to 52T VOL_NORM dataset.

Enriches data/52t_volnorm/ with three cross-sectional features:
  sector_relative_return_7d  — ticker's ret_7d minus mean ret_7d of its sector peers on that date
  sector_rank_30d            — percentile rank (0–1) of ticker's ret_30d within its sector on that date
  spy_correlation_30d        — rolling 30-day correlation of ticker's ret_1d with SPY's ret_1d

Motivation (2026-04-02 Phase 1 escalation):
  The 8D VOL_NORM fingerprint captures absolute momentum shape but not cross-sectional
  relative performance. Bear market failures may stem from spurious matches: in a downturn,
  all tickers share similar absolute return trajectories, creating high KNN similarity
  that doesn't predict forward returns. Sector-relative features distinguish tickers that
  outperformed their peers from those that merely fell with the market.

No temporal leakage:
  All three features are computed cross-sectionally or from lagged data only.
  sector_relative_return_7d uses same-date peers (available at time T).
  spy_correlation_30d uses SPY ret_1d shifted by rolling window (available at time T).

Output: data/52t_features/ (adds 3 columns on top of 52t_volnorm schema)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.sector import SECTOR_MAP

INPUT_DIR  = project_root / "data" / "52t_volnorm"
OUTPUT_DIR = project_root / "data" / "52t_features"

# Fallback sectors for tickers missing from SECTOR_MAP (excluded from 585T expansion)
_SECTOR_FALLBACK = {
    "ABBV": "Health",
    "META": "Tech",
    "PYPL": "Finance",
    "TSLA": "Consumer",
}


def _get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker) or _SECTOR_FALLBACK.get(ticker, "Unknown")


def compute_sector_relative_return_7d(df: pd.DataFrame) -> pd.Series:
    """Ticker's ret_7d minus sector-mean ret_7d on the same date.

    Cross-sectional: uses all available tickers on each date.
    No look-ahead bias — sector peers' returns on date T are observable at T.
    """
    df = df.copy()
    df["_sector"] = df["Ticker"].map(lambda t: _get_sector(t))
    sector_mean = df.groupby(["Date", "_sector"])["ret_7d"].transform("mean")
    return (df["ret_7d"] - sector_mean).rename("sector_relative_return_7d")


def compute_sector_rank_30d(df: pd.DataFrame) -> pd.Series:
    """Percentile rank (0–1) of ticker's ret_30d within its sector on each date.

    Rank 1.0 = best performer in sector, 0.0 = worst.
    Cross-sectional: same no-look-ahead guarantee as above.
    """
    df = df.copy()
    df["_sector"] = df["Ticker"].map(lambda t: _get_sector(t))

    def _rank_within_group(x: pd.Series) -> pd.Series:
        # pct=True gives fractional rank in [0,1]
        return x.rank(pct=True, method="average")

    return (
        df.groupby(["Date", "_sector"])["ret_30d"]
        .transform(_rank_within_group)
        .rename("sector_rank_30d")
    )


def compute_spy_correlation_30d(df: pd.DataFrame) -> pd.Series:
    """Rolling 30-day Pearson correlation of ticker's daily returns vs SPY.

    Uses ret_1d (daily return already in dataset). SPY must be in the ticker list.
    Window min_periods=10 to avoid NaN at the start of each ticker's history.
    """
    if "SPY" not in df["Ticker"].values:
        raise RuntimeError("SPY is not in the dataset — required for spy_correlation_30d.")

    # Build SPY daily return series indexed by Date
    spy_returns = (
        df[df["Ticker"] == "SPY"]
        .set_index("Date")["ret_1d"]
        .rename("spy_ret_1d")
    )

    df = df.copy()
    df = df.join(spy_returns, on="Date")

    results = []
    for ticker, grp in df.groupby("Ticker"):
        grp = grp.sort_values("Date").copy()
        corr = grp["ret_1d"].rolling(30, min_periods=10).corr(grp["spy_ret_1d"])
        grp["spy_correlation_30d"] = corr
        results.append(grp[["Date", "Ticker", "spy_correlation_30d"]])

    corr_df = pd.concat(results, ignore_index=True)
    merged = df[["Date", "Ticker"]].merge(corr_df, on=["Date", "Ticker"], how="left")
    return merged["spy_correlation_30d"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  BUILD 52T FEATURE-ENRICHED DATASET")
    print("=" * 65)
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Adding: sector_relative_return_7d, sector_rank_30d, spy_correlation_30d")

    # Load combined dataset for cross-sectional computation
    train = pd.read_parquet(INPUT_DIR / "train_db.parquet")
    val   = pd.read_parquet(INPUT_DIR / "val_db.parquet")
    full  = pd.concat([train, val], ignore_index=True)
    full["Date"] = pd.to_datetime(full["Date"])
    print(f"\n  Loaded: {len(full):,} rows, {full['Ticker'].nunique()} tickers")

    # ── Compute features ──────────────────────────────────────────────────
    print("\n  Computing sector_relative_return_7d...", end="", flush=True)
    full["sector_relative_return_7d"] = compute_sector_relative_return_7d(full)
    nan_pct = full["sector_relative_return_7d"].isna().mean()
    print(f" done (NaN={nan_pct:.2%})")

    print("  Computing sector_rank_30d...", end="", flush=True)
    full["sector_rank_30d"] = compute_sector_rank_30d(full)
    nan_pct = full["sector_rank_30d"].isna().mean()
    print(f" done (NaN={nan_pct:.2%})")

    print("  Computing spy_correlation_30d...", end="", flush=True)
    full["spy_correlation_30d"] = compute_spy_correlation_30d(full)
    nan_pct = full["spy_correlation_30d"].isna().mean()
    print(f" done (NaN={nan_pct:.2%})")

    # ── Validation ────────────────────────────────────────────────────────
    new_cols = ["sector_relative_return_7d", "sector_rank_30d", "spy_correlation_30d"]
    print("\n  Feature stats:")
    for col in new_cols:
        s = full[col].dropna()
        print(f"    {col:<35s}  mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"min={s.min():.4f}  max={s.max():.4f}")

    # ── Re-split and save ─────────────────────────────────────────────────
    train_df = full[full["Date"] <= "2023-12-31"].copy()
    val_df   = full[(full["Date"] >= "2024-01-01") & (full["Date"] <= "2024-12-31")].copy()

    # Drop rows where any new feature is NaN (to keep dataset clean)
    n_before = len(train_df)
    train_df = train_df.dropna(subset=new_cols).reset_index(drop=True)
    val_df   = val_df.dropna(subset=new_cols).reset_index(drop=True)
    n_after = len(train_df)
    if n_before > n_after:
        print(f"\n  Dropped {n_before - n_after:,} train rows with NaN in new features.")

    train_df.to_parquet(OUTPUT_DIR / "train_db.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "val_db.parquet", index=False)

    print(f"\n  Saved train: {len(train_df):,} rows")
    print(f"  Saved val:   {len(val_df):,} rows")
    print(f"\n  Columns: {len(train_df.columns)} total")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
