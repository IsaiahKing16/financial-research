"""
validate_emd_bma.py — Standalone fold comparison: EMDDistance + BMACalibrator vs baseline.

Loads the full analogue DB via DataLoader (same path as bss_regression_test.py),
splits into 2024 train/val fold, then runs:
  - EMDDistance (time_penalty=0.5, price_penalty=1.0) for neighbour retrieval
  - BMACalibrator (df=3, n_iter=30) for probability calibration

Calibration leakage prevention:
  - CAL_FRAC = 0.76 chronological split within X_train
  - X_fit  = X_train[:cal_split]  → KNN index built on this portion
  - X_cal  = X_train[cal_split:]  → calibration samples, each queries X_fit only
  This mirrors the temporal ordering constraint from production Platt setup.

Logs result to data/results/experiments_emd_bma.tsv for provenance.
Every BSS number reported here MUST trace to this file or terminal output.

Usage:
    python scripts/validate_emd_bma.py
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss

# Project root on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.data import DataLoader
from pattern_engine.features import RETURN_COLS  # [ret_1d, ret_3d, ..., ret_90d]
from research.emd_distance import EMDDistance
from research.bma_calibrator import BMACalibrator


# ---------------------------------------------------------------------------
# Config — mirrors locked settings from CLAUDE.md
# ---------------------------------------------------------------------------
FOLD_TRAIN_END = "2023-12-31"
FOLD_VAL_START = "2024-01-01"
FOLD_VAL_END   = "2024-12-31"
TOP_K          = 50
MAX_DISTANCE   = 1.1019
FEATURE_COLS   = RETURN_COLS           # ret_1d, ret_3d, ret_7d, ..., ret_90d (8 cols)
TARGET_COL     = "fwd_7d_up"
CAL_FRAC       = 0.76                  # chronological split within training data
BASELINE_BSS   = 0.00103              # fold 6 (2024), walkforward terminal output — locked

# ---------------------------------------------------------------------------
# Load full DB via DataLoader (same pattern as bss_regression_test.py)
# ---------------------------------------------------------------------------
data_dir = project_root / "data"
raw_dir = data_dir / "raw"

# Main repo stores CSVs directly in data/ (not data/raw/)
if not raw_dir.exists() and list(data_dir.glob("*.csv")):
    import shutil
    raw_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in data_dir.glob("*.csv"):
        dest = raw_dir / csv_file.name
        if not dest.exists():
            shutil.copy2(csv_file, dest)
    print(f"  Copied {len(list(raw_dir.glob('*.csv')))} CSVs to data/raw/")

print("Loading data...")
loader = DataLoader(data_dir=str(data_dir))
raw_data = loader.download(force_refresh=False)
print(f"  Loaded {len(raw_data)} tickers")

full_db = loader.compute_features(raw_data)
print(f"  Full DB: {len(full_db)} rows")

# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
full_db["Date"] = full_db["Date"].astype("datetime64[ns]")
train_db = full_db[full_db["Date"] <= FOLD_TRAIN_END]
val_db   = full_db[
    (full_db["Date"] >= FOLD_VAL_START) & (full_db["Date"] <= FOLD_VAL_END)
]

print(f"\nFold split: train={len(train_db)} rows, val={len(val_db)} rows")

X_train = train_db[FEATURE_COLS].values.astype(float)  # (N_train, 8)
X_val   = val_db[FEATURE_COLS].values.astype(float)    # (N_val, 8)
y_train = train_db[TARGET_COL].values.astype(float)
y_val   = val_db[TARGET_COL].values.astype(float)

# ---------------------------------------------------------------------------
# EMDDistance: fit on training set
# ---------------------------------------------------------------------------
print("\nFitting EMDDistance...")
t0 = time.time()
metric = EMDDistance(time_penalty=0.5, price_penalty=1.0)
metric.fit(X_train)

# Retrieve top-K raw analogue outcomes for each validation sample
print(f"Running EMD on {len(X_val)} validation samples (this may take several minutes)...")
raw_probs = np.full((len(X_val), TOP_K), np.nan)
skipped = 0

for i, query in enumerate(X_val):
    distances = metric.compute(query, X_train)          # (N_train,)
    within_max = distances <= MAX_DISTANCE
    n_within = int(within_max.sum())
    if n_within < TOP_K:
        skipped += 1
        continue
    # Sort only the within-max distances; take first TOP_K
    within_idx = np.where(within_max)[0]
    top_idx = np.argsort(distances[within_idx])[:TOP_K]
    analogue_idx = within_idx[top_idx]
    raw_probs[i] = y_train[analogue_idx]

    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        print(f"  {i + 1}/{len(X_val)} done ({elapsed:.0f}s, {skipped} skipped)")

elapsed_emd = time.time() - t0
print(f"\nEMD complete in {elapsed_emd:.0f}s. Skipped {skipped}/{len(X_val)} samples (< {TOP_K} neighbours).")

# Drop rows with insufficient neighbours
valid_mask = ~np.isnan(raw_probs[:, 0])
raw_probs_valid = raw_probs[valid_mask]   # (N_valid, TOP_K)
y_val_valid     = y_val[valid_mask]       # (N_valid,)
print(f"Valid samples for calibration: {valid_mask.sum()}/{len(X_val)}")

# ---------------------------------------------------------------------------
# BMACalibrator — fit on CAL_FRAC chronological split of training data
# CRITICAL: Cal samples must query X_fit only (not X_train) — prevents leakage
# ---------------------------------------------------------------------------
print(f"\nFitting BMACalibrator (CAL_FRAC={CAL_FRAC})...")
cal_split = int(len(X_train) * CAL_FRAC)
X_fit = X_train[:cal_split]   # (N_fit, 8)
X_cal = X_train[cal_split:]   # (N_cal, 8)
y_fit = y_train[:cal_split]
y_cal = y_train[cal_split:]

print(f"  Fit portion: {len(X_fit)} rows, Cal portion: {len(X_cal)} rows")

# Build cal raw_probs: each cal sample queries X_fit only
cal_raw = np.full((len(X_cal), TOP_K), np.nan)
for i, query in enumerate(X_cal):
    distances = metric.compute(query, X_fit)    # query against fit portion only
    top_idx = np.argsort(distances)[:TOP_K]
    cal_raw[i] = y_fit[top_idx]

# Drop cal rows with any NaN (shouldn't happen since no MAX_DISTANCE filter here)
cal_valid = ~np.isnan(cal_raw[:, 0])
cal_raw_valid = cal_raw[cal_valid]
y_cal_valid   = y_cal[cal_valid]
print(f"  Cal rows for BMA fit: {cal_valid.sum()}/{len(X_cal)}")

bma = BMACalibrator(n_iter=30, df=3.0)
bma.fit(cal_raw_valid, y_cal_valid)
print(f"  BMA fitted. Top 5 analogue weights: {bma.weights[:5].round(4)}")

# ---------------------------------------------------------------------------
# Calibrated probabilities on validation set
# ---------------------------------------------------------------------------
cal_probs = np.array([float(bma.transform(row)) for row in raw_probs_valid])

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
brier_emd_bma = brier_score_loss(y_val_valid, cal_probs)
brier_ref     = brier_score_loss(y_val_valid, np.full(len(y_val_valid), y_val_valid.mean()))
bss_emd_bma   = 1.0 - (brier_emd_bma / brier_ref)
delta         = bss_emd_bma - BASELINE_BSS
gate_cleared  = delta >= 0.02

print("\n" + "=" * 60)
print("  EMD + BMA vs Baseline (2024 fold)")
print("=" * 60)
print(f"  BSS (EMD+BMA):  {bss_emd_bma:+.5f}")
print(f"  BSS (baseline): {BASELINE_BSS:+.5f}")
print(f"  Delta:          {delta:+.5f}")
print(f"  Gate cleared:   {gate_cleared}  (need delta >= +0.02000)")
print("=" * 60)

if gate_cleared:
    print("\n  PROMOTION GATE CLEARED — Phase C work may proceed.")
else:
    print("\n  Gate NOT cleared. Try tuning time_penalty (0.0-2.0) or price_penalty.")

# ---------------------------------------------------------------------------
# Provenance log → data/results/experiments_emd_bma.tsv
# ---------------------------------------------------------------------------
results_dir = data_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
tsv_path = results_dir / "experiments_emd_bma.tsv"

row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "experiment_name": "validate_emd_bma_2024fold",
    "fold": "2024",
    "n_valid": int(valid_mask.sum()),
    "n_skipped": skipped,
    "bss_emd_bma": round(float(bss_emd_bma), 7),
    "bss_baseline": BASELINE_BSS,
    "delta": round(float(delta), 7),
    "gate_cleared": gate_cleared,
    "notes": "EMDDistance(tp=0.5,pp=1.0)+BMA(df=3,n_iter=30) CAL_FRAC=0.76 TOP_K=50 MAX_DIST=1.1019",
}

write_header = not tsv_path.exists()
with open(tsv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f"\n  Provenance logged → {tsv_path}")
