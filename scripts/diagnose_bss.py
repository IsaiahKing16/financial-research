"""
diagnose_bss.py — Root cause investigation for BSS < 0 across all 6 walk-forward folds.

Experiments run on fold 6 (2024-Val: train_end=2023-12-31, val_start=2024-01-01,
val_end=2024-12-31).

Usage:
    PYTHONUTF8=1 py -3.12 scripts/diagnose_bss.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import VOL_NORM_COLS, RETURNS_ONLY_COLS
from scripts.run_walkforward import WalkForwardConfig, load_full_db, FOLDS

# ── Fold 6 (index 5) ─────────────────────────────────────────────────────────
FOLD = FOLDS[5]  # 2024-Val


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_bss_components(probs: np.ndarray, y_true: np.ndarray) -> dict:
    """Return brier, brier_clim, bss, prob_range, and distribution stats."""
    brier = float(np.mean((probs - y_true) ** 2))
    brier_clim = float(np.var(y_true))
    bss_val = (1.0 - brier / brier_clim) if brier_clim > 0 else 0.0
    prob_range = float(np.max(probs) - np.min(probs))
    return {
        "brier": brier,
        "brier_clim": brier_clim,
        "bss": bss_val,
        "prob_range": prob_range,
        "prob_mean": float(np.mean(probs)),
        "prob_std": float(np.std(probs)),
        "prob_p5": float(np.percentile(probs, 5)),
        "prob_p95": float(np.percentile(probs, 95)),
        "base_rate": float(np.mean(y_true)),
        "n": len(probs),
    }


def print_metrics(label: str, m: dict) -> None:
    print(f"\n  {'─'*50}")
    print(f"  {label}")
    print(f"  {'─'*50}")
    print(f"  BSS          = {m['bss']:+.6f}")
    print(f"  Brier        = {m['brier']:.6f}")
    print(f"  Brier_clim   = {m['brier_clim']:.6f}")
    print(f"  Base rate    = {m['base_rate']:.4f}  (P(up==1))")
    print(f"  N samples    = {m['n']:,}")
    print(f"  Prob range   = {m['prob_range']:.6f}  (max - min)")
    print(f"  Prob mean    = {m['prob_mean']:.6f}")
    print(f"  Prob std     = {m['prob_std']:.6f}")
    print(f"  Prob p5/p95  = [{m['prob_p5']:.6f}, {m['prob_p95']:.6f}]")


def print_reliability_diagram(probs: np.ndarray, y_true: np.ndarray,
                               n_bins: int = 10, label: str = "") -> None:
    """Print a text-based reliability diagram with 10 equal-width bins."""
    if label:
        print(f"\n  Reliability Diagram — {label}")
    else:
        print("\n  Reliability Diagram")
    print(f"  {'Bin':>18}  {'Mean_pred':>10}  {'Mean_actual':>12}  {'Count':>8}  {'Diff':>8}")
    print(f"  {'-'*62}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            mean_pred = float("nan")
            mean_act = float("nan")
            diff = float("nan")
            diff_str = "  N/A"
        else:
            mean_pred = float(np.mean(probs[mask]))
            mean_act = float(np.mean(y_true[mask]))
            diff = mean_pred - mean_act
            diff_str = f"{diff:+.4f}"
        bin_label = f"[{lo:.1f}, {hi:.1f})"
        if count == 0:
            print(f"  {bin_label:>18}  {'N/A':>10}  {'N/A':>12}  {count:>8}  {diff_str:>8}")
        else:
            print(f"  {bin_label:>18}  {mean_pred:>10.4f}  {mean_act:>12.4f}  {count:>8}  {diff_str:>8}")


def run_one_fold(
    full_db: pd.DataFrame,
    feature_cols: list[str],
    calibration_method: str = "platt",
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit PatternMatcher on fold 6 train data and query val data.

    Returns (probs, y_true, metrics_dict).
    """
    train_end = pd.Timestamp(FOLD["train_end"])
    val_start = pd.Timestamp(FOLD["val_start"])
    val_end = pd.Timestamp(FOLD["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].copy()
    val_db = full_db[
        (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
    ].copy()

    cfg = WalkForwardConfig()
    # Override calibration method for this experiment
    cfg.calibration_method = calibration_method
    # nn_jobs=1: Windows/Py3.12 deadlock guard (CLAUDE.md rule 5)
    cfg.nn_jobs = 1

    train_db = train_db.dropna(subset=[cfg.projection_horizon])
    val_db = val_db.dropna(subset=[cfg.projection_horizon])

    print(f"    Train rows: {len(train_db):,}  |  Val rows: {len(val_db):,}")

    t0 = time.time()
    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, feature_cols)
    print(f"    fit() done in {time.time()-t0:.1f}s")

    t1 = time.time()
    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
    print(f"    query() done in {time.time()-t1:.1f}s")

    probs = np.asarray(probs, dtype=np.float64)
    y_true = val_db[cfg.projection_horizon].values.astype(np.float64)

    m = compute_bss_components(probs, y_true)
    return probs, y_true, m


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 68

    print(sep)
    print("  BSS Diagnostic — Fold 6 (2024-Val)")
    print(f"  Fold: {FOLD['label']}  train_end={FOLD['train_end']}")
    print(f"        val_start={FOLD['val_start']}  val_end={FOLD['val_end']}")
    print(sep)

    print("\nLoading analogue database...")
    full_db = load_full_db()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"{full_db['Date'].min().date()} to {full_db['Date'].max().date()}")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 1: Platt calibration vs raw K-NN frequencies
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  EXPERIMENT 1: Platt calibrated vs raw K-NN probabilities")
    print(f"  Features: VOL_NORM_COLS")
    print(sep)

    print("\n  [1a] Running with calibration_method='platt' ...")
    probs_platt, y_true, m_platt = run_one_fold(
        full_db, VOL_NORM_COLS, calibration_method="platt", label="Platt"
    )
    print_metrics("Platt calibrated", m_platt)

    print("\n  [1b] Running with calibration_method='none' (raw K-NN freq) ...")
    probs_raw, _, m_raw = run_one_fold(
        full_db, VOL_NORM_COLS, calibration_method="none", label="Raw"
    )
    print_metrics("Raw K-NN (no calibration)", m_raw)

    print(f"\n  BSS delta (Platt - Raw): {m_platt['bss'] - m_raw['bss']:+.6f}")
    if m_raw["bss"] > m_platt["bss"]:
        print("  => Platt calibration is HURTING BSS (raw is better)")
    elif m_platt["bss"] > m_raw["bss"]:
        print("  => Platt calibration is HELPING BSS (Platt is better)")
    else:
        print("  => Platt calibration makes no difference")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 2: Reliability diagram (Platt-calibrated)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  EXPERIMENT 2: Reliability diagram (Platt-calibrated, VOL_NORM)")
    print(sep)
    print_reliability_diagram(probs_platt, y_true, n_bins=10, label="Platt / VOL_NORM")

    # Narrow-range reliability diagram: 20 bins in [0.4, 0.65] to reveal
    # clustering structure when prob_range is small
    if m_platt["prob_range"] < 0.3:
        print("\n  [Note: prob_range is small — printing fine-grained bins in [0.3, 0.7]]")
        lo_clip = max(0.0, m_platt["prob_p5"] - 0.02)
        hi_clip = min(1.0, m_platt["prob_p95"] + 0.02)
        print(f"\n  Fine-grained reliability diagram  [{lo_clip:.3f}, {hi_clip:.3f}]  (20 bins)")
        print(f"  {'Bin':>22}  {'Mean_pred':>10}  {'Mean_actual':>12}  {'Count':>8}  {'Diff':>8}")
        print(f"  {'-'*66}")
        edges_fine = np.linspace(lo_clip, hi_clip, 21)
        for i in range(20):
            lo, hi = edges_fine[i], edges_fine[i + 1]
            if i == 19:
                mask = (probs_platt >= lo) & (probs_platt <= hi)
            else:
                mask = (probs_platt >= lo) & (probs_platt < hi)
            count = int(mask.sum())
            if count == 0:
                print(f"  [{lo:.4f}, {hi:.4f})  {'N/A':>10}  {'N/A':>12}  {count:>8}  {'N/A':>8}")
                continue
            mean_pred = float(np.mean(probs_platt[mask]))
            mean_act = float(np.mean(y_true[mask]))
            diff = mean_pred - mean_act
            print(f"  [{lo:.4f}, {hi:.4f})  {mean_pred:>10.4f}  {mean_act:>12.4f}  {count:>8}  {diff:>+8.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # EXPERIMENT 3: VOL_NORM_COLS vs RETURNS_ONLY_COLS (both Platt)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  EXPERIMENT 3: VOL_NORM_COLS vs RETURNS_ONLY_COLS (both Platt)")
    print(sep)

    # VOL_NORM already run above — reuse m_platt
    print("\n  [3a] VOL_NORM_COLS + Platt (already computed above)")
    print_metrics("VOL_NORM + Platt", m_platt)

    print("\n  [3b] Running RETURNS_ONLY_COLS + Platt ...")
    probs_ret, _, m_ret = run_one_fold(
        full_db, RETURNS_ONLY_COLS, calibration_method="platt", label="RETURNS_ONLY+Platt"
    )
    print_metrics("RETURNS_ONLY + Platt", m_ret)

    print(f"\n  BSS delta (VOL_NORM - RETURNS_ONLY): {m_platt['bss'] - m_ret['bss']:+.6f}")
    if m_ret["bss"] > m_platt["bss"]:
        print("  => VOL_NORM features are HURTING BSS vs raw returns")
    elif m_platt["bss"] > m_ret["bss"]:
        print("  => VOL_NORM features are HELPING BSS vs raw returns")
    else:
        print("  => Feature set makes no difference")

    # ──────────────────────────────────────────────────────────────────────────
    # Summary table
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  SUMMARY TABLE")
    print(sep)
    print(f"\n  {'Config':<36}  {'BSS':>10}  {'Brier':>10}  {'ProbRange':>10}  {'ProbStd':>8}")
    print(f"  {'-'*78}")
    rows = [
        ("VOL_NORM + Platt (current config)", m_platt),
        ("VOL_NORM + Raw (no calibration)",   m_raw),
        ("RETURNS_ONLY + Platt",              m_ret),
    ]
    for name, m in rows:
        print(f"  {name:<36}  {m['bss']:>+10.6f}  {m['brier']:>10.6f}  "
              f"{m['prob_range']:>10.6f}  {m['prob_std']:>8.6f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Root cause diagnosis
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  ROOT CAUSE DIAGNOSIS")
    print(sep)

    # Determine the dominant driver
    all_bss_negative = all(m["bss"] < 0 for _, m in rows)
    platt_hurts = m_raw["bss"] > m_platt["bss"]
    vol_norm_hurts = m_ret["bss"] > m_platt["bss"]
    low_discrimination = m_platt["prob_range"] < 0.20

    print(f"\n  Base rate (P(up)):       {m_platt['base_rate']:.4f}")
    print(f"  All configs BSS < 0:     {all_bss_negative}")
    print(f"  Platt hurts vs raw:      {platt_hurts}")
    print(f"  VOL_NORM hurts vs ret:   {vol_norm_hurts}")
    print(f"  Low discrimination       {low_discrimination}  "
          f"(prob_range={m_platt['prob_range']:.4f})")

    print("\n  Diagnosis:")

    if all_bss_negative:
        print("  [!] ALL configs produce negative BSS. This is not a calibration-only")
        print("      issue — the K-NN matches carry no predictive signal for fwd_7d_up.")

    if low_discrimination:
        print(f"  [!] Predicted probabilities span only {m_platt['prob_range']:.4f} (p5={m_platt['prob_p5']:.4f},")
        print(f"      p95={m_platt['prob_p95']:.4f}). The model has almost no discrimination —")
        print("      all queries return near-base-rate probabilities. This collapses BSS")
        print("      because Brier ≈ Brier_clim when prob ≈ constant ≈ base_rate.")

    if platt_hurts:
        print("  [!] Platt calibration degrades BSS. The calibrator is overfitting the")
        print("      training-set self-query distribution and compressing probs further.")
    else:
        print("  [ok] Platt calibration is not making BSS worse (or helps marginally).")

    if vol_norm_hurts:
        print("  [!] VOL_NORM features produce worse BSS than RETURNS_ONLY. Vol-normalization")
        print("      may be mixing magnitude signal with noise or the norm columns have NaNs")
        print("      / outliers that collapse K-NN distance geometry.")
    else:
        print("  [ok] VOL_NORM features are not worse than RETURNS_ONLY (or are better).")

    print("\n  Likely root causes (ranked by signal strength above):")
    causes = []
    if low_discrimination:
        causes.append("1. Low discrimination — K-NN freq ≈ base rate for every query. "
                      "The 585-ticker universe may have diluted meaningful analogues "
                      "so that all 50 neighbours are near base-rate, leaving no "
                      "discriminative signal for Platt or raw probs to exploit.")
    if vol_norm_hurts:
        causes.append("2. VOL_NORM features hurt — raw returns carry more discriminative "
                      "signal than vol-normalised ratios for this universe size / horizon.")
    if platt_hurts:
        causes.append("3. Platt calibration overcorrects — the double-pass self-query "
                      "on training data compresses the already-narrow prob distribution "
                      "further toward 0.5, amplifying Brier score relative to the "
                      "climatological baseline.")
    if not causes:
        causes.append("1. Marginal negative BSS close to zero — may be statistical noise "
                      "or slight feature mismatch rather than a systemic failure.")

    for c in causes:
        print(f"\n  {c}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
