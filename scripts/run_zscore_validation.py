"""scripts/run_zscore_validation.py — T7.5-1 Phase 7.5 gate validation.

Runs three conditions over all 6 walk-forward folds on 52T beta_abm data:
  1. z_score_on    : standardize_features=True,  uniform weights  (production baseline)
  2. z_score_off   : standardize_features=False, uniform weights  (raw comparison)
  3. group_balanced: standardize_features=True, group-equalized weights (fallback)

Output: results/phase7_5/zscore_bss_comparison.tsv

Gate criterion (G7.5-1):
  z_score_on BSS >= z_score_off BSS on >= 4/6 folds  -> normalization CONFIRMED
  mean(z_score_on BSS) >= 0.00033                     -> absolute threshold

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_zscore_validation.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_engine.candlestick import CANDLE_COLS
from pattern_engine.config import WALKFORWARD_FOLDS
from pattern_engine.features import VOL_NORM_COLS, get_feature_cols, group_balanced_weights
from pattern_engine.walkforward import load_and_augment_db, run_walkforward

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "phase7_5" / "zscore_bss_comparison.tsv"
FEATURE_COLS = get_feature_cols("returns_candle")  # 23 features


def run_condition(full_db, label: str, cfg_overrides: dict) -> list[dict]:
    """Run all 6 folds for one condition. Returns list of per-fold result dicts."""
    print(f"\n=== Condition: {label} ===")
    result = run_walkforward(
        full_db,
        feature_cols=FEATURE_COLS,
        cfg_overrides=cfg_overrides,
        folds=WALKFORWARD_FOLDS,
    )
    rows = result["fold_results"]
    for r in rows:
        r["condition"] = label
    print(
        f"  mean_bss={result['mean_bss']:.6f}  "
        f"trimmed_mean={result['trimmed_mean_bss']:.6f}  "
        f"positive_folds={result['positive_folds']}/6"
    )
    return rows


def print_comparison(all_rows: list[dict]) -> None:
    """Print a side-by-side BSS comparison table to stdout."""
    by_cond: dict[str, dict[str, float]] = {}
    for r in all_rows:
        by_cond.setdefault(r["condition"], {})[r["fold"]] = r["bss"]

    fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    conditions = list(by_cond.keys())

    header = f"{'Fold':<16}" + "".join(f"{c:>16}" for c in conditions)
    print("\n=== BSS Comparison ===")
    print(header)
    print("-" * len(header))
    for fold in fold_labels:
        row = f"{fold:<16}"
        for c in conditions:
            bss = by_cond[c].get(fold, float("nan"))
            row += f"{bss:>16.6f}"
        print(row)

    baseline = by_cond.get("z_score_on", {})
    print("\n=== Deltas vs z_score_on ===")
    for cond in conditions:
        if cond == "z_score_on":
            continue
        wins = sum(
            1 for f in fold_labels
            if (by_cond[cond].get(f, float("nan")) > baseline.get(f, float("nan")))
        )
        print(f"  {cond}: {wins}/6 folds better than z_score_on")

    z_on_rows = [r for r in all_rows if r["condition"] == "z_score_on"]
    z_off_rows = [r for r in all_rows if r["condition"] == "z_score_off"]

    wins_vs_off = sum(
        1 for fl in fold_labels
        if (
            next((r["bss"] for r in z_on_rows if r["fold"] == fl), float("nan"))
            > next((r["bss"] for r in z_off_rows if r["fold"] == fl), float("nan"))
        )
    )
    mean_bss_on = float(np.mean([r["bss"] for r in z_on_rows]))

    cond_a = "PASS" if wins_vs_off >= 4 else "FAIL"
    cond_b = "PASS" if mean_bss_on >= 0.00033 else "FAIL"

    print("\n=== G7.5-1 Gate Check ===")
    print(f"  Condition A (z_score_on wins >= 4/6 vs z_score_off): {wins_vs_off}/6 -> {cond_a}")
    print(f"  Condition B (mean BSS z_score_on >= 0.00033):        {mean_bss_on:.6f} -> {cond_b}")
    if cond_a == "PASS" and cond_b == "PASS":
        print("  G7.5-1: PASS")
    else:
        print("  G7.5-1: FAIL — see ADR-015 for follow-up actions")


def main() -> None:
    print("Loading 52T database...")
    full_db = load_and_augment_db()
    print(f"  {len(full_db):,} rows loaded.")

    returns_cols = list(VOL_NORM_COLS)
    candle_cols = list(CANDLE_COLS)
    gb_weights = group_balanced_weights(returns_cols, candle_cols)

    conditions = [
        ("z_score_on",    {"standardize_features": True}),
        ("z_score_off",   {"standardize_features": False}),
        ("group_balanced", {"standardize_features": True, "feature_weights": gb_weights}),
    ]

    all_rows: list[dict] = []
    for label, overrides in conditions:
        all_rows.extend(run_condition(full_db, label, overrides))

    print_comparison(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition", "fold", "bss", "n_scored", "n_total",
        "base_rate", "mean_prob", "reliability", "resolution", "uncertainty",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults written to {OUTPUT_PATH}")
    print("\nG7.5-1 gate: z_score_on >= z_score_off on >= 4/6 folds AND mean_bss >= 0.00033?")


if __name__ == "__main__":
    main()
