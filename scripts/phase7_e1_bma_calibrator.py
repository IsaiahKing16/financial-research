"""
scripts/phase7_e1_bma_calibrator.py — E1: BMA vs beta_abm BSS comparison.
Gate: BMA_BSS - baseline_BSS >= +0.001 on >= 3/6 folds.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase7_e1_bma_calibrator.py

Provenance: Phase 7 implementation plan, Task E1 (2026-04-09)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root resolution (works in worktrees and main repo) ───────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.config import WALKFORWARD_FOLDS
from pattern_engine.features import FeatureRegistry

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase7_baseline import (
    run_fold_with_config,
    _BetaCalibrator,
    _augment_with_candlestick,
    DATA_DIR,
    FEATURE_COLS,
    HORIZON,
)

RESULTS_DIR  = REPO_ROOT / "results" / "phase7"
BASELINE_TSV = RESULTS_DIR / "baseline_23d.tsv"
OUTPUT_TSV   = RESULTS_DIR / "e1_bma_vs_beta_abm.tsv"
SUMMARY_TSV  = RESULTS_DIR / "enhancement_summary.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GATE_DELTA     = 0.001
GATE_MIN_FOLDS = 3


def main():
    print("=" * 72)
    print("  Task E1 — BMA Calibrator vs beta_abm BSS comparison")
    print("=" * 72)
    print(f"  Gate: BMA_BSS - baseline_BSS >= +{GATE_DELTA} on >= {GATE_MIN_FOLDS}/6 folds")
    print()

    # Load baseline per-fold BSS
    if not BASELINE_TSV.exists():
        print(f"ERROR: Baseline TSV not found: {BASELINE_TSV}")
        print("       Run scripts/phase7_baseline.py first.")
        sys.exit(1)

    baseline_df = pd.read_csv(BASELINE_TSV, sep="\t")
    baseline_bss = baseline_df.set_index("fold")["bss"].to_dict()
    print(f"Baseline BSS loaded from: {BASELINE_TSV}")
    for fold_label, bss in baseline_bss.items():
        print(f"  {fold_label:<16}: {bss:+.5f}")
    print()

    # Load full DB (same logic as phase7_baseline.py)
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"

    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        print("       Run scripts/build_52t_features.py first.")
        sys.exit(1)

    print(f"Loading 52T features from {DATA_DIR} ...", flush=True)
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"date range {full_db['Date'].min().date()} – {full_db['Date'].max().date()}")

    # Augment with candlestick features
    full_db = _augment_with_candlestick(full_db)

    print(f"\n{'─'*72}")
    print(f"  Running 6-fold walk-forward with use_bma=True")
    print(f"{'─'*72}\n")

    t_total = time.time()
    rows = []
    for fi, fold in enumerate(WALKFORWARD_FOLDS):
        label = fold["label"]
        print(f"  [{fi+1}/6] BMA fold {label}...", end=" ", flush=True)
        t0 = time.time()
        result = run_fold_with_config(
            fold, full_db, list(FEATURE_COLS),
            cfg_overrides={"use_bma": True},
        )
        elapsed = time.time() - t0
        bma_bss = result["bss"]
        base_bss = baseline_bss.get(label, float("nan"))
        delta = bma_bss - base_bss if not (np.isnan(bma_bss) or np.isnan(base_bss)) else float("nan")
        improved = (delta >= GATE_DELTA) if not np.isnan(delta) else False
        print(
            f"BMA={bma_bss:+.5f}  base={base_bss:+.5f}  "
            f"delta={delta:+.5f}  ({elapsed:.1f}s)"
        )
        rows.append({
            "fold":         label,
            "baseline_bss": base_bss,
            "bma_bss":      bma_bss,
            "delta":        delta,
            "improved":     improved,
        })

    total_elapsed = time.time() - t_total

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.6f")

    n_improved = int(df["improved"].sum())
    gate = "PASS" if n_improved >= GATE_MIN_FOLDS else "FAIL"

    print()
    print("=" * 72)
    print("  E1 BMA SUMMARY")
    print("=" * 72)
    print(f"  {'Fold':<16}  {'Baseline':>9}  {'BMA':>9}  {'Delta':>9}  {'Improved':>8}")
    print(f"  {'─'*16}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
    for r in rows:
        imp_s = "YES" if r["improved"] else "no"
        print(
            f"  {r['fold']:<16}  {r['baseline_bss']:>+9.5f}  "
            f"{r['bma_bss']:>+9.5f}  {r['delta']:>+9.5f}  {imp_s:>8}"
        )
    print(f"  {'─'*16}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
    print(f"\n  Gate: {n_improved}/{len(df)} folds improved by >= +{GATE_DELTA} → {gate}")
    print(f"  Total runtime: {total_elapsed:.0f}s")
    print(f"  Output: {OUTPUT_TSV}")
    print()

    # Append to enhancement_summary.tsv
    summary_row = {
        "enhancement":           "E1_BMA",
        "flag_value":            str(gate == "PASS"),
        "baseline_bss_per_fold": str(list(df["baseline_bss"].round(5))),
        "enhanced_bss_per_fold": str(list(df["bma_bss"].round(5))),
        "delta_per_fold":        str(list(df["delta"].round(5))),
        "folds_improved":        n_improved,
        "gate_metric":           "bss_delta",
        "gate_threshold":        GATE_DELTA,
        "gate_result":           gate,
        "provenance_file":       str(OUTPUT_TSV),
        "runtime_seconds":       round(total_elapsed, 1),
    }
    summary_df = pd.DataFrame([summary_row])
    if SUMMARY_TSV.exists():
        summary_df.to_csv(SUMMARY_TSV, sep="\t", index=False, mode="a", header=False)
    else:
        summary_df.to_csv(SUMMARY_TSV, sep="\t", index=False)
    print(f"  Summary appended to: {SUMMARY_TSV}")

    return gate


if __name__ == "__main__":
    gate = main()
    sys.exit(0 if gate == "PASS" else 1)
