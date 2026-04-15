"""P8-PRE-4: Feature Standardization Verification Experiment.

Runs 52T walk-forward under two conditions:
  - standardize_features=True  (CURRENT production behavior)
  - standardize_features=False (no-scaler baseline, new flag)

Compares per-fold BSS and Murphy B3 decomposition.
Output: results/phase8_pre/standardization_experiment.tsv

Expected outcome:
  If True produces higher BSS -> standardization is confirmed beneficial (ADR-007).
  If BSS is identical -> investigate; scaler may not be applied in distance space.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_standardization_experiment.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_engine.features import get_feature_cols
from pattern_engine.walkforward import run_walkforward, load_and_augment_db
from pattern_engine.config import WALKFORWARD_FOLDS

OUTPUT_PATH = Path("results/phase8_pre/standardization_experiment.tsv")
FEATURE_COLS = get_feature_cols("returns_candle")  # 23 features


def run_condition(full_db, label: str, standardize: bool) -> list[dict]:
    """Run all 6 folds for one standardization condition.

    run_walkforward returns a dict with keys: mean_bss, trimmed_mean_bss,
    positive_folds, fold_results (list of per-fold dicts), wilcoxon_p.
    We extract fold_results for the per-fold comparison.
    """
    print(f"\n=== Condition: standardize_features={standardize} ===")
    cfg_overrides = {"standardize_features": standardize}
    wf_result = run_walkforward(
        full_db,
        feature_cols=FEATURE_COLS,
        cfg_overrides=cfg_overrides,
        folds=WALKFORWARD_FOLDS,
    )
    fold_rows = wf_result["fold_results"]   # list[dict] — one entry per fold
    for r in fold_rows:
        r["condition"] = "scaled" if standardize else "raw"
    return fold_rows


def main():
    print("Loading 52T database...")
    full_db = load_and_augment_db()
    print(f"  {len(full_db):,} rows loaded.")

    all_results = []
    all_results.extend(run_condition(full_db, "scaled", standardize=True))
    all_results.extend(run_condition(full_db, "raw", standardize=False))

    # Write TSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "fold", "bss", "n_scored", "n_total",
                  "base_rate", "mean_prob", "reliability", "resolution", "uncertainty"]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults written to {OUTPUT_PATH}")

    # Print comparison summary
    print("\n=== BSS Comparison ===")
    print(f"{'Fold':<12} {'Scaled':>10} {'Raw':>10} {'Delta':>10}")
    print("-" * 45)

    scaled = {r["fold"]: r["bss"] for r in all_results if r["condition"] == "scaled"}
    raw    = {r["fold"]: r["bss"] for r in all_results if r["condition"] == "raw"}

    n_scaled_wins = 0
    for fold in sorted(scaled):
        s = scaled[fold]
        r = raw.get(fold, float("nan"))
        delta = s - r if not (np.isnan(s) or np.isnan(r)) else float("nan")
        win = "+" if delta > 0 else "-" if delta < 0 else "="
        print(f"  {fold:<10} {s:>10.6f} {r:>10.6f} {delta:>+10.6f} {win}")
        if not np.isnan(delta) and delta > 0:
            n_scaled_wins += 1

    print(f"\nScaled wins: {n_scaled_wins}/6 folds")
    mean_scaled = np.nanmean([r["bss"] for r in all_results if r["condition"] == "scaled"])
    mean_raw    = np.nanmean([r["bss"] for r in all_results if r["condition"] == "raw"])
    print(f"Mean BSS (scaled): {mean_scaled:+.6f}")
    print(f"Mean BSS (raw):    {mean_raw:+.6f}")
    print(f"Mean delta:        {mean_scaled - mean_raw:+.6f}")

    # Gate check
    gate_pass = n_scaled_wins >= 4 or (mean_scaled - mean_raw) >= 0.005
    print(f"\nGate (>=4/6 wins OR mean delta >=+0.005): {'PASS' if gate_pass else 'FAIL'}")
    if gate_pass:
        print("ACTION: Standardization confirmed. Write ADR-007 CONFIRMED.")
    else:
        print("ACTION: Investigate why scaling shows no benefit. Check _prepare_features path.")


if __name__ == "__main__":
    main()
