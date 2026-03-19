"""
diagnose_distances.py — Feature space geometry diagnostic

QUESTION: Why does AvgK ≈ 50 even at MAX_DISTANCE = 0.2?

This script answers that by measuring the actual cosine distance distribution
between val queries and training analogues, then checks whether cross-sectional
(market-relative) encoding would produce more discriminating distances.

WHAT IT PRODUCES:
  1. Distance percentile table — what fraction of training pairs are within
     each distance threshold. If 95%+ are within 0.2, the space is collapsed.
  2. Comparison: raw cosine vs Euclidean vs cross-sectional cosine
  3. A recommendation on which encoding to try next

USAGE:
  python diagnose_distances.py

REQUIRES: data/train_db.parquet, data/val_db.parquet, models/analogue_scaler.pkl
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

from prepare import FEATURE_COLS


def sample_distances(X_train, X_val, metric="cosine", n_val_samples=500, n_neighbors=200):
    """
    Sample pairwise distances between a subset of val queries and training set.
    Returns flat array of all distances found.
    """
    idx = np.random.choice(len(X_val), min(n_val_samples, len(X_val)), replace=False)
    X_val_sample = X_val[idx]

    nn = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(X_train)),
        metric=metric,
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(X_train)
    distances, _ = nn.kneighbors(X_val_sample)
    return distances.flatten()


def cross_sectional_encode(db, feature_cols):
    """
    Replace raw feature values with their deviation from the daily cross-sectional median.
    For each day, subtract the median value of that feature across all tickers.

    This removes market-wide drift — instead of "AAPL returned +3% over 7 days,"
    the encoded value becomes "+3% minus the median 7d return across all stocks that day."
    Stocks that moved like the crowd get values near zero; outliers get large values.

    The distance metric then finds stocks that behaved unusually in the SAME WAY,
    rather than stocks that all happened to drift upward together.
    """
    db = db.copy()
    db["Date"] = pd.to_datetime(db["Date"])

    for col in feature_cols:
        daily_median = db.groupby("Date")[col].transform("median")
        db[col] = db[col] - daily_median

    return db


def percentile_table(distances, thresholds=(0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)):
    """Print what fraction of distances fall within each threshold."""
    print(f"\n  {'Threshold':>10} {'% within':>10} {'Cumulative':>12}")
    print(f"  {'─'*35}")
    for t in thresholds:
        within = (distances <= t).mean() * 100
        print(f"  {t:>10.2f} {within:>9.1f}% {'← filter barely binds' if within > 95 else ''}")


def main():
    print("\n" + "="*65)
    print("  DISTANCE DISTRIBUTION DIAGNOSTIC")
    print("="*65)

    print("\n  Loading data...")
    train_db = pd.read_parquet(DATA_DIR / "train_db.parquet")
    val_db   = pd.read_parquet(DATA_DIR / "val_db.parquet")
    scaler   = joblib.load(MODEL_DIR / "analogue_scaler.pkl")

    print(f"  Train: {len(train_db):,} rows | Val: {len(val_db):,} rows")
    print(f"  Features: {len(FEATURE_COLS)}")

    # ── 1. Raw scaled cosine distances (current approach) ──────────────
    print("\n[1/3] Current approach: StandardScaler + cosine distance")
    X_train_scaled = scaler.transform(train_db[FEATURE_COLS].values)
    X_val_scaled   = scaler.transform(val_db[FEATURE_COLS].values)

    np.random.seed(42)
    dist_cosine_raw = sample_distances(X_train_scaled, X_val_scaled, metric="cosine")
    print(f"  Sampled {len(dist_cosine_raw):,} val→train distances")
    print(f"  Mean distance: {dist_cosine_raw.mean():.4f}")
    print(f"  Median distance: {np.median(dist_cosine_raw):.4f}")
    print(f"  10th pct: {np.percentile(dist_cosine_raw, 10):.4f}")
    print(f"  90th pct: {np.percentile(dist_cosine_raw, 90):.4f}")
    percentile_table(dist_cosine_raw)

    # ── 2. Euclidean distance on same scaled features ──────────────────
    print("\n[2/3] Alternative: StandardScaler + Euclidean distance")
    dist_euclidean = sample_distances(X_train_scaled, X_val_scaled, metric="euclidean")
    max_e = np.percentile(dist_euclidean, 99)  # Use 99th pct as effective max
    dist_euclidean_norm = dist_euclidean / max_e
    print(f"  Mean distance: {dist_euclidean.mean():.4f}  (raw Euclidean)")
    print(f"  Std distance:  {dist_euclidean.std():.4f}")
    print(f"  Normalized to [0,1] using 99th percentile = {max_e:.3f}")
    print(f"  Spread ratio (std/mean): {dist_euclidean.std()/dist_euclidean.mean():.3f}")
    print(f"  (higher spread = better discrimination)")
    print(f"  Spread ratio cosine:     {dist_cosine_raw.std()/dist_cosine_raw.mean():.3f}")

    # Quantile-based threshold table — the principled way to set MAX_DISTANCE
    # for Euclidean. Rather than manually deriving thresholds from the 99th pct
    # normalisation, compute the raw Euclidean distance at each desired AvgK target.
    # These values can be copied directly into the sweep config.
    print(f"\n  Quantile-based thresholds for Euclidean MAX_DISTANCE:")
    print(f"  (sets how many of the 200 sampled neighbours fall within threshold)")
    print(f"  {'Target AvgK':>12} {'Quantile':>10} {'Raw threshold':>15} {'Use in sweep as'}")
    print(f"  {'─'*58}")
    for target_k in [5, 10, 15, 20, 30, 40, 50]:
        quantile = target_k / 200.0          # fraction of top-200 neighbours
        raw_threshold = np.percentile(dist_euclidean, quantile * 100)
        note = " ← recommended (tight)" if target_k == 10 else (" ← recommended (medium)" if target_k == 20 else "")
        print(f"  {target_k:>12}     {quantile*100:>7.1f}%   {raw_threshold:>12.4f}    MAX_DISTANCE={raw_threshold:.4f}{note}")
    print(f"  Note: these are raw Euclidean units — not comparable to cosine thresholds.")

    # ── 3. Cross-sectional encoding + cosine distance ──────────────────
    print("\n[3/3] Cross-sectional encoding: market-relative features + cosine")
    print("  Encoding features as deviation from daily cross-sectional median...")

    full_db = pd.concat([train_db, val_db]).copy()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    full_db_cs = cross_sectional_encode(full_db, FEATURE_COLS)

    train_cs = full_db_cs[full_db_cs["Date"] <= pd.Timestamp("2023-12-31")].copy()
    val_cs   = full_db_cs[full_db_cs["Date"] >= pd.Timestamp("2024-01-01")].copy()

    scaler_cs = StandardScaler()
    X_train_cs = scaler_cs.fit_transform(train_cs[FEATURE_COLS].values)
    X_val_cs   = scaler_cs.transform(val_cs[FEATURE_COLS].values)

    dist_cosine_cs = sample_distances(X_train_cs, X_val_cs, metric="cosine")
    print(f"  Mean distance: {dist_cosine_cs.mean():.4f}")
    print(f"  Median distance: {np.median(dist_cosine_cs):.4f}")
    print(f"  Spread ratio: {dist_cosine_cs.std()/dist_cosine_cs.mean():.3f}")
    percentile_table(dist_cosine_cs)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  SUMMARY — distance discrimination comparison")
    print("="*65)
    print(f"\n  {'Method':<40} {'Mean dist':>10} {'Spread':>8} {'%<0.2':>8}")
    print(f"  {'─'*68}")

    methods = [
        ("Current: StandardScaler + cosine",     dist_cosine_raw,  True),
        ("Alternative: StandardScaler + Euclid (norm)", dist_euclidean_norm, False),
        ("Proposed: Cross-sectional + cosine",    dist_cosine_cs,   True),
    ]
    for name, dist, is_cosine in methods:
        pct_under = (dist <= 0.2).mean() * 100 if is_cosine else (dist <= 0.2).mean() * 100
        spread    = dist.std() / dist.mean()
        print(f"  {name:<40} {dist.mean():>10.4f} {spread:>8.3f} {pct_under:>7.1f}%")

    print()
    cosine_pct = (dist_cosine_raw <= 0.2).mean() * 100
    cs_pct     = (dist_cosine_cs <= 0.2).mean() * 100
    euc_pct    = (dist_euclidean_norm <= 0.2).mean() * 100

    print("  Interpretation:")
    if cosine_pct > 90:
        print(f"  ✗ Current approach: {cosine_pct:.1f}% of distances < 0.2")
        print(f"    The distance filter is geometrically collapsed.")
        print(f"    Most stocks look like twins to cosine distance on these features.")

    if cs_pct < cosine_pct - 10:
        print(f"  ✓ Cross-sectional encoding: {cs_pct:.1f}% within 0.2")
        print(f"    Market-relative features spread the distance distribution.")
        print(f"    Recommended: try this encoding next.")
    else:
        print(f"  ~ Cross-sectional: {cs_pct:.1f}% — marginal improvement only.")

    if euc_pct < cosine_pct - 10:
        print(f"  ✓ Euclidean distance: {euc_pct:.1f}% within 0.2 (normalised)")
        print(f"    May discriminate better — worth testing.")

    print()
    print("  Next step: update prepare.py cross_sectional_encode flag OR")
    print("  add DISTANCE_METRIC = 'euclidean' to strategy.py and rerun sweep.")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
