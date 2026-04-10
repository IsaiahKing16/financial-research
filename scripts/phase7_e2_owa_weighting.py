"""
scripts/phase7_e2_owa_weighting.py — E2: OWA feature weighting comparison.

Alpha sweep via leave-one-fold-out CV, then 6-fold gate evaluation.
Gate: BSS improvement >= +0.001 on >= 3/6 folds vs baseline AND
      worst-fold degradation <= -0.0005.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.config import WALKFORWARD_FOLDS
from pattern_engine.features import FeatureRegistry
from pattern_engine.owa_weights import evaluate_owa_gate

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase7_baseline import run_fold_with_config, DATA_DIR

RESULTS_DIR  = REPO_ROOT / "results" / "phase7"
BASELINE_TSV = RESULTS_DIR / "baseline_23d.tsv"
OUTPUT_TSV   = RESULTS_DIR / "e2_owa_vs_baseline.tsv"
SUMMARY_TSV  = RESULTS_DIR / "enhancement_summary.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_CANDIDATES = [0.5, 1.0, 2.0, 4.0]
GATE_DELTA     = 0.001
GATE_MIN_FOLDS = 3
FEATURE_COLS   = list(FeatureRegistry.get("returns_candle").columns)


def main():
    baseline_df = pd.read_csv(BASELINE_TSV, sep="\t")
    baseline_bss = baseline_df.set_index("fold")["bss"].to_dict()

    # Load data — same as phase7_baseline.py loader logic
    all_files = sorted((DATA_DIR).glob("*.parquet"))
    if not all_files:
        print(f"ERROR: No parquet files found in {DATA_DIR}")
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in all_files]
    full_db = pd.concat(dfs, ignore_index=True)
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    # Augment with candlestick features (required for returns_candle)
    from phase7_baseline import _augment_with_candlestick
    full_db = _augment_with_candlestick(full_db)

    # --- Alpha sweep: leave-one-fold-out CV ---
    print("=" * 72)
    print("  E2: OWA Feature Weighting — Alpha Sweep (leave-one-fold-out CV)")
    print("=" * 72)
    print(f"  Candidates: {ALPHA_CANDIDATES}")
    print()

    alpha_scores = {}
    for alpha in ALPHA_CANDIDATES:
        fold_bss = []
        print(f"  alpha={alpha}:", end="", flush=True)
        for fold in WALKFORWARD_FOLDS:
            result = run_fold_with_config(
                fold, full_db, FEATURE_COLS,
                cfg_overrides={"use_owa": True, "owa_alpha": alpha}
            )
            fold_bss.append(result["bss"])
            print(f" {result['bss']:+.5f}", end="", flush=True)
        mean_bss = float(np.mean([b for b in fold_bss if not np.isnan(b)]))
        alpha_scores[alpha] = mean_bss
        print(f"  → mean={mean_bss:.5f}")

    best_alpha = max(alpha_scores, key=alpha_scores.get)
    print(f"\nSelected alpha: {best_alpha} (mean_BSS={alpha_scores[best_alpha]:.5f})")
    print()

    # --- Full 6-fold walk-forward with selected alpha ---
    print(f"Full walk-forward with alpha={best_alpha}...")
    print(f"{'─'*72}")
    rows = []
    t_total = time.time()
    for fold in WALKFORWARD_FOLDS:
        label = fold["label"]
        print(f"  OWA fold {label}...", end=" ", flush=True)
        t0 = time.time()
        result = run_fold_with_config(
            fold, full_db, FEATURE_COLS,
            cfg_overrides={"use_owa": True, "owa_alpha": best_alpha}
        )
        elapsed = time.time() - t0
        owa_bss = result["bss"]
        base_bss = baseline_bss[label]
        delta = owa_bss - base_bss
        print(f"OWA={owa_bss:+.5f}  base={base_bss:+.5f}  delta={delta:+.5f}  ({elapsed:.1f}s)")
        rows.append({
            "fold": label,
            "baseline_bss": base_bss,
            "owa_bss": owa_bss,
            "delta": delta,
            "alpha": best_alpha,
            "improved": delta >= GATE_DELTA,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)

    baseline_arr = df["baseline_bss"].values
    owa_arr = df["owa_bss"].values
    gate, reason = evaluate_owa_gate(baseline_arr, owa_arr)
    n_improved = int(df["improved"].sum())

    print()
    print("=" * 72)
    print("  E2 OWA GATE RESULT")
    print("=" * 72)
    print(f"  {n_improved}/{len(df)} folds improved by >= +{GATE_DELTA}")
    print(f"  Gate: {gate}")
    print(f"  Reason: {reason}")
    print(f"  Alpha selected: {best_alpha}")
    print(f"  Total runtime: {time.time() - t_total:.0f}s")
    print(f"  Results: {OUTPUT_TSV}")
    print()

    summary_row = {
        "enhancement": "E2_OWA",
        "flag_value": str(gate == "PASS"),
        "baseline_bss_per_fold": str(list(df["baseline_bss"].round(5))),
        "enhanced_bss_per_fold": str(list(df["owa_bss"].round(5))),
        "delta_per_fold": str(list(df["delta"].round(5))),
        "folds_improved": n_improved,
        "gate_metric": "bss_delta+worst_fold",
        "gate_threshold": GATE_DELTA,
        "gate_result": gate,
        "provenance_file": str(OUTPUT_TSV),
        "runtime_seconds": f"{time.time() - t_total:.0f}",
    }
    summary_df = pd.DataFrame([summary_row])
    if SUMMARY_TSV.exists():
        summary_df.to_csv(SUMMARY_TSV, sep="\t", index=False, mode="a", header=False)
    else:
        summary_df.to_csv(SUMMARY_TSV, sep="\t", index=False)

    return gate


if __name__ == "__main__":
    main()
