"""
scripts/phase7_e3_dtw_reranker.py — E3: DTW Reranker BSS walk-forward.

Gate: BSS delta >= +0.001 on >= 3/6 folds.
Spearman fast-fail: if mean rho > 0.95 on 100 queries, FAIL immediately.

Output: results/phase7/e3_dtw_vs_baseline.tsv
Appends to: results/phase7/enhancement_summary.tsv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pattern_engine.config import WALKFORWARD_FOLDS
from pattern_engine.features import FeatureRegistry
from phase7_baseline import (
    run_fold_with_config,
    DATA_DIR,
    _augment_with_candlestick,
    _BetaCalibrator,
)
import pattern_engine.matcher as _matcher_module

RESULTS_DIR  = REPO_ROOT / "results" / "phase7"
BASELINE_TSV = RESULTS_DIR / "baseline_23d.tsv"
OUTPUT_TSV   = RESULTS_DIR / "e3_dtw_vs_baseline.tsv"
SUMMARY_TSV  = RESULTS_DIR / "enhancement_summary.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GATE_DELTA     = 0.001
GATE_MIN_FOLDS = 3
FEATURE_COLS   = list(FeatureRegistry.get("returns_candle").columns)


def spearman_fast_fail_check(
    full_db: pd.DataFrame,
    n_queries: int = 100,
    rho_threshold: float = 0.95,
) -> tuple[bool, float]:
    """Run Spearman fast-fail check on fold 2019 (first fold)."""
    from scipy.stats import spearmanr
    from pattern_engine.config import EngineConfig
    from pattern_engine.matcher import PatternMatcher
    from phase7_baseline import _build_cfg

    fold = WALKFORWARD_FOLDS[0]
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].dropna(subset=["fwd_7d_up"]).copy()
    val_db   = full_db[
        (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
    ].dropna(subset=["fwd_7d_up"]).copy()

    # Fit matcher (no DTW) using locked config
    cfg = _build_cfg({"use_dtw_reranker": False})
    original_calibrator = _matcher_module._PlattCalibrator
    try:
        _matcher_module._PlattCalibrator = _BetaCalibrator
        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, FEATURE_COLS)
    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    # Sample queries
    rng = np.random.RandomState(0)
    X_raw = val_db[FEATURE_COLS].values
    X_scaled = matcher._prepare_features(X_raw, fit_scaler=False)
    sample_size = min(n_queries, len(X_scaled))
    sample_idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
    X_sample = X_scaled[sample_idx]

    from research.wfa_reranker import dtw_rerank
    rhos = []
    for i in range(len(X_sample)):
        distances_b, indices_b = matcher._query_batch(X_sample[i:i+1])
        euclidean_order = np.argsort(distances_b[0])

        nbrs_scaled = matcher._X_train_weighted[indices_b[0]]
        _, dtw_dists = dtw_rerank(
            X_sample[i], nbrs_scaled, indices_b[0], k=len(indices_b[0])
        )
        dtw_order = np.argsort(dtw_dists)

        n_compare = min(50, len(euclidean_order), len(dtw_order))
        rho, _ = spearmanr(euclidean_order[:n_compare], dtw_order[:n_compare])
        rhos.append(rho)

    mean_rho = float(np.mean(rhos))
    return mean_rho > rho_threshold, mean_rho


def append_to_summary(row: dict) -> None:
    if SUMMARY_TSV.exists():
        summary = pd.read_csv(SUMMARY_TSV, sep="\t")
    else:
        summary = pd.DataFrame(columns=["enhancement", "gate_result", "folds_improved",
                                         "mean_delta", "best_delta", "worst_delta", "notes"])
    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    summary.to_csv(SUMMARY_TSV, sep="\t", index=False)


def main():
    print("=== E3: DTW Reranker Walk-forward ===\n")

    # Load baseline
    if not BASELINE_TSV.exists():
        raise RuntimeError(f"Baseline TSV not found: {BASELINE_TSV}")
    baseline_df = pd.read_csv(BASELINE_TSV, sep="\t")
    baseline_bss = dict(zip(baseline_df["fold"], baseline_df["bss"]))

    # Load and augment data
    print("Loading data...")
    all_files = sorted(DATA_DIR.glob("*.parquet"))
    if not all_files:
        print(f"ERROR: No parquet files found in {DATA_DIR}")
        sys.exit(1)
    dfs = [pd.read_parquet(f) for f in all_files]
    full_db = pd.concat(dfs, ignore_index=True)
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    full_db = _augment_with_candlestick(full_db)
    print(f"  {len(full_db):,} rows loaded.\n")

    # Spearman fast-fail check
    print("Running Spearman fast-fail check on fold 2019 (100 queries)...")
    t_ff = time.time()
    should_fail, mean_rho = spearman_fast_fail_check(full_db)
    print(f"  Mean Spearman rho = {mean_rho:.4f} ({time.time()-t_ff:.1f}s)")

    fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    base_bss_list = [baseline_bss[lbl] for lbl in fold_labels]

    if should_fail:
        print(f"\nFAST-FAIL: Mean Spearman rho={mean_rho:.4f} > 0.95 — DTW ≈ Euclidean ordering.")
        print("DTW reranking provides no meaningful reordering. Writing FAIL.\n")
        gate_result = "FAIL"
        rows = [
            {
                "fold": lbl,
                "baseline_bss": b,
                "dtw_bss": float("nan"),
                "delta": float("nan"),
                "improved": False,
            }
            for lbl, b in zip(fold_labels, base_bss_list)
        ]
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_TSV, sep="\t", index=False)
        append_to_summary({
            "enhancement": "E3_DTW",
            "gate_result": gate_result,
            "folds_improved": 0,
            "mean_delta": float("nan"),
            "best_delta": float("nan"),
            "worst_delta": float("nan"),
            "notes": f"Spearman fast-fail: mean_rho={mean_rho:.4f}",
        })
        print(f"Result: {gate_result}")
        return

    # Full 6-fold walk-forward
    print("Spearman check passed. Running 6-fold walk-forward with DTW reranker...\n")
    rows = []
    for i, fold in enumerate(WALKFORWARD_FOLDS):
        label = fold["label"]
        print(f"  [{i+1}/6] {label}...", end=" ", flush=True)
        t0 = time.time()
        result = run_fold_with_config(
            fold, full_db, FEATURE_COLS,
            cfg_overrides={"use_dtw_reranker": True, "dtw_rerank_k": 20}
        )
        elapsed = time.time() - t0
        base_bss = baseline_bss[label]
        delta = result["bss"] - base_bss
        print(f"BSS={result['bss']:.5f}  delta={delta:+.5f}  ({elapsed:.1f}s)")
        rows.append({
            "fold": label,
            "baseline_bss": base_bss,
            "dtw_bss": result["bss"],
            "delta": delta,
            "improved": delta >= GATE_DELTA,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)

    # Gate evaluation
    deltas = df["delta"].values
    folds_improved = int((deltas >= GATE_DELTA).sum())
    mean_delta = float(deltas.mean())
    best_delta = float(deltas.max())
    worst_delta = float(deltas.min())

    gate_result = "PASS" if folds_improved >= GATE_MIN_FOLDS else "FAIL"

    print(f"\n{'='*50}")
    print(f"E3 DTW Gate: {folds_improved}/{len(WALKFORWARD_FOLDS)} folds improved by >=+{GATE_DELTA}")
    print(f"Mean delta: {mean_delta:+.5f}, Best: {best_delta:+.5f}, Worst: {worst_delta:+.5f}")
    print(f"Gate result: {gate_result}")

    append_to_summary({
        "enhancement": "E3_DTW",
        "gate_result": gate_result,
        "folds_improved": folds_improved,
        "mean_delta": round(mean_delta, 6),
        "best_delta": round(best_delta, 6),
        "worst_delta": round(worst_delta, 6),
        "notes": f"Spearman rho={mean_rho:.4f}; {folds_improved}/6 folds >= +{GATE_DELTA}",
    })

    print(f"\nSaved: {OUTPUT_TSV}")
    print(df[["fold", "baseline_bss", "dtw_bss", "delta", "improved"]].to_string(index=False))


if __name__ == "__main__":
    main()
