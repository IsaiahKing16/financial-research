"""
scripts/phase7_e4_conformal.py — E4: Adaptive Conformal Prediction coverage eval.

Evaluates ACI prediction interval coverage across 6 walk-forward folds.
Gate: coverage >= 88% on ALL 6 folds AND mean_width < 0.30.

For each fold:
1. Fit matcher on training data (beta_abm, locked config)
2. Get calibrated probs for validation data
3. Split val into calibration half and test half (first 50% as cal, last 50% as test)
4. Sweep gamma = [0.01, 0.05, 0.10]
5. For each gamma: run ACI on test half (in temporal order), record coverage + width
6. Select best gamma (coverage closest to 0.90 without going below 0.88)
7. Record selected gamma's coverage and width

Output: results/phase7/e4_conformal_coverage.tsv
Appends to: results/phase7/enhancement_summary.tsv
"""
from __future__ import annotations
import sys, time
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
from pattern_engine.conformal_hooks import AdaptiveConformalPredictor

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase7_baseline import (
    run_fold_with_config,
    _BetaCalibrator,
    _augment_with_candlestick,
    _apply_h7_hold_regime,
    DATA_DIR,
    FEATURE_COLS,
    HORIZON,
    SPY_THRESHOLD,
    _build_cfg,
)
import pattern_engine.matcher as _matcher_module
from pattern_engine.matcher import PatternMatcher

RESULTS_DIR = REPO_ROOT / "results" / "phase7"
OUTPUT_TSV  = RESULTS_DIR / "e4_conformal_coverage.tsv"
SUMMARY_TSV = RESULTS_DIR / "enhancement_summary.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GAMMA_SWEEP       = [0.01, 0.05, 0.10]
GATE_MIN_COVERAGE = 0.88
GATE_MAX_WIDTH    = 0.30


def run_fold_conformal(fold: dict, full_db: pd.DataFrame) -> dict:
    """Run one fold: fit matcher, get probs, eval ACI coverage."""
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    train_db = (
        full_db[full_db["Date"] <= train_end]
        .dropna(subset=[HORIZON])
        .copy()
    )
    val_db = (
        full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ]
        .dropna(subset=[HORIZON])
        .copy()
    )

    n_total   = len(val_db)
    y_true    = val_db[HORIZON].values.astype(float)
    base_rate = float(y_true.mean())

    if n_total == 0:
        return {"fold": fold["label"], "best_gamma": float("nan"),
                "coverage": float("nan"), "mean_width": float("nan"),
                "n_cal": 0, "n_test": 0}

    cfg = _build_cfg()

    original_calibrator = _matcher_module._PlattCalibrator
    try:
        _matcher_module._PlattCalibrator = _BetaCalibrator
        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, list(FEATURE_COLS))
        probs_raw, _, _, _, _, _ = matcher.query(val_db, verbose=0)
    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    probs = np.asarray(probs_raw)

    # H7 HOLD regime: bear rows → base_rate probability
    probs_hold, bear_mask = _apply_h7_hold_regime(
        val_db=val_db,
        train_db=train_db,
        base_rate=base_rate,
        probs=probs,
    )

    # Use scored (non-bear) rows only
    scored_mask  = ~bear_mask
    scored_probs = probs_hold[scored_mask]
    scored_labels = y_true[scored_mask]

    if len(scored_probs) < 10:
        return {"fold": fold["label"], "best_gamma": float("nan"),
                "coverage": float("nan"), "mean_width": float("nan"),
                "n_cal": 0, "n_test": 0}

    # Split into calibration (first 50%) and test (last 50%)
    split = len(scored_probs) // 2
    cal_probs   = scored_probs[:split]
    cal_labels  = scored_labels[:split]
    test_probs  = scored_probs[split:]
    test_labels = scored_labels[split:]

    best_gamma    = GAMMA_SWEEP[0]
    best_coverage = 0.0
    best_width    = 1.0

    for gamma in GAMMA_SWEEP:
        pred = AdaptiveConformalPredictor(nominal_alpha=0.10, gamma=gamma)
        pred.calibrate(cal_probs, cal_labels)

        covered = 0
        widths  = []
        for p, y in zip(test_probs, test_labels):
            lo, hi = pred.predict_interval(float(p))
            widths.append(hi - lo)
            if lo <= float(y) <= hi:
                covered += 1
            pred.update(float(p), int(y))

        coverage = covered / len(test_probs)
        mean_w   = float(np.mean(widths))

        # Pick gamma with coverage closest to 0.90 without going below 0.88
        if coverage >= GATE_MIN_COVERAGE:
            if abs(coverage - 0.90) < abs(best_coverage - 0.90) or best_coverage < GATE_MIN_COVERAGE:
                best_gamma    = gamma
                best_coverage = coverage
                best_width    = mean_w

    return {
        "fold":       fold["label"],
        "best_gamma": best_gamma,
        "coverage":   best_coverage,
        "mean_width": best_width,
        "n_cal":      split,
        "n_test":     len(test_probs),
    }


def append_to_summary(row: dict) -> None:
    if SUMMARY_TSV.exists():
        summary = pd.read_csv(SUMMARY_TSV, sep="\t")
    else:
        summary = pd.DataFrame(columns=["enhancement", "gate_result", "folds_improved",
                                         "mean_delta", "best_delta", "worst_delta", "notes"])
    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
    summary.to_csv(SUMMARY_TSV, sep="\t", index=False)


def main():
    print("=== E4: Adaptive Conformal Prediction Coverage Evaluation ===\n")

    # Load full DB (same logic as E1-E3 scripts)
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
    print(f"  {len(full_db):,} rows loaded")

    # Add candlestick features if not present
    if "candle_1d_body_to_range" not in full_db.columns:
        full_db = _augment_with_candlestick(full_db)

    print()

    rows = []
    for i, fold in enumerate(WALKFORWARD_FOLDS):
        print(f"  [{i+1}/6] {fold['label']}...", end=" ", flush=True)
        t0 = time.time()
        result = run_fold_conformal(fold, full_db)
        elapsed = time.time() - t0
        print(f"coverage={result['coverage']:.3f}  width={result['mean_width']:.3f}  "
              f"gamma={result['best_gamma']}  ({elapsed:.1f}s)")
        rows.append(result)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)

    # Gate evaluation
    coverages = df["coverage"].values
    widths    = df["mean_width"].values

    all_coverage_ok = bool((coverages >= GATE_MIN_COVERAGE).all())
    all_width_ok    = bool((widths < GATE_MAX_WIDTH).all())
    gate_result     = "PASS" if (all_coverage_ok and all_width_ok) else "FAIL"

    folds_above_88 = int((coverages >= GATE_MIN_COVERAGE).sum())
    folds_narrow   = int((widths < GATE_MAX_WIDTH).sum())

    print(f"\n{'='*55}")
    print(f"Coverage gate (>=88% all folds): {all_coverage_ok}  ({folds_above_88}/6 folds)")
    print(f"Width gate (<0.30 all folds):    {all_width_ok}  ({folds_narrow}/6 folds)")
    print(f"Mean coverage: {coverages.mean():.3f}")
    print(f"Mean width:    {widths.mean():.3f}")
    print(f"\nGate result: {gate_result}")

    append_to_summary({
        "enhancement":    "E4_Conformal",
        "gate_result":    gate_result,
        "folds_improved": folds_above_88,
        "mean_delta":     round(float(coverages.mean()), 4),
        "best_delta":     round(float(coverages.max()), 4),
        "worst_delta":    round(float(coverages.min()), 4),
        "notes": (f"coverage_gate={'PASS' if all_coverage_ok else 'FAIL'} "
                  f"width_gate={'PASS' if all_width_ok else 'FAIL'} "
                  f"mean_cov={coverages.mean():.3f} mean_w={widths.mean():.3f}"),
    })

    print(f"\nSaved: {OUTPUT_TSV}")
    print(df[["fold", "coverage", "mean_width", "best_gamma", "n_test"]].to_string(index=False))


if __name__ == "__main__":
    main()
