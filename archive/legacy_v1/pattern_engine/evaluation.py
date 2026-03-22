"""
evaluation.py — Evaluation metrics for analogue matching predictions.

Combines signal-aligned classification metrics with proper scoring rules
(Brier Score, BSS, CRPS) for comprehensive probabilistic evaluation.
"""

import numpy as np
from pattern_engine.scoring import brier_score, brier_skill_score, compute_crps, compute_calibration


def evaluate_from_signals(y_true_binary: np.ndarray, probabilities: np.ndarray,
                          signals: np.ndarray) -> dict:
    """Evaluate using actual signals from generate_signal().

    Counts as "confident" only trades that pass ALL three filters:
    MIN_MATCHES, AGREEMENT_SPREAD, and CONFIDENCE_THRESHOLD.

    Args:
        y_true_binary: array of 0/1 ground truth
        probabilities: array of P(up) from analogue matching
        signals: array of "BUY"/"SELL"/"HOLD" strings

    Returns:
        dict of classification metrics aligned with actual signal logic
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred_all = (probabilities >= 0.5).astype(int)
    acc_all = accuracy_score(y_true_binary, y_pred_all)

    confident_mask = (signals == "BUY") | (signals == "SELL")
    n_confident = confident_mask.sum()

    if n_confident == 0:
        return {
            "total_samples": len(y_true_binary),
            "accuracy_all": float(acc_all),
            "confident_trades": 0,
            "confident_pct": 0.0,
            "accuracy_confident": 0.0,
            "precision_confident": 0.0,
            "recall_confident": 0.0,
            "f1_confident": 0.0,
        }

    y_true_conf = y_true_binary[confident_mask]
    y_pred_conf = (probabilities[confident_mask] >= 0.5).astype(int)

    return {
        "total_samples": len(y_true_binary),
        "accuracy_all": float(acc_all),
        "confident_trades": int(n_confident),
        "confident_pct": float(n_confident / len(y_true_binary)),
        "accuracy_confident": float(accuracy_score(y_true_conf, y_pred_conf)),
        "precision_confident": float(precision_score(y_true_conf, y_pred_conf, zero_division=0)),
        "recall_confident": float(recall_score(y_true_conf, y_pred_conf, zero_division=0)),
        "f1_confident": float(f1_score(y_true_conf, y_pred_conf, zero_division=0)),
    }


def evaluate_probabilistic(y_true_binary: np.ndarray, y_true_returns: np.ndarray,
                            probabilities: np.ndarray, ensemble_returns_list: list,
                            signals: np.ndarray, horizon_label: str = "") -> dict:
    """Full probabilistic evaluation suite.

    Combines signal-aligned classification metrics with proper scoring rules.

    Args:
        y_true_binary: array of 0/1 (did price go up?)
        y_true_returns: array of actual continuous returns
        probabilities: array of predicted P(up)
        ensemble_returns_list: list of arrays, raw analogue returns per query
        signals: array of "BUY"/"SELL"/"HOLD"
        horizon_label: e.g. "fwd_7d" for reporting

    Returns:
        dict with all metrics
    """
    class_metrics = evaluate_from_signals(y_true_binary, probabilities, signals)

    bs = brier_score(y_true_binary, probabilities)
    bss = brier_skill_score(y_true_binary, probabilities)
    crps = compute_crps(y_true_returns, ensemble_returns_list)
    calibration = compute_calibration(y_true_binary, probabilities)

    metrics = {
        **class_metrics,
        "brier_score": round(bs, 5),
        "brier_skill_score": round(bss, 5),
        "crps": round(crps, 5) if crps is not None else None,
        "calibration_buckets": calibration,
        "horizon": horizon_label,
    }

    return metrics


def print_metrics(metrics: dict, label: str = "") -> None:
    """Print full probabilistic evaluation in a clean format."""
    h = metrics.get("horizon", "")
    tag = f" — {label}" if label else ""
    tag += f" | {h}" if h else ""

    print(f"\n{'=' * 60}")
    print(f"  PROBABILISTIC EVALUATION{tag}")
    print(f"{'=' * 60}")
    print(f"  Total samples:        {metrics['total_samples']:,}")
    print(f"  Accuracy (all):       {metrics['accuracy_all']:.1%}")
    print(f"  Confident trades:     {metrics['confident_trades']:,} ({metrics['confident_pct']:.1%})")
    print(f"  Accuracy (confident): {metrics['accuracy_confident']:.1%}")
    print(f"  Precision (conf):     {metrics['precision_confident']:.1%}")
    print(f"  F1 (confident):       {metrics['f1_confident']:.1%}")
    print(f"  ---")
    print(f"  Brier Score:          {metrics['brier_score']:.5f}  (lower=better, naive=0.25)")
    print(f"  Brier Skill Score:    {metrics['brier_skill_score']:+.5f} (positive=beats base rate)")
    if metrics.get("crps") is not None:
        print(f"  CRPS:                 {metrics['crps']:.5f}  (lower=better)")
    else:
        print(f"  CRPS:                 N/A (pip install scoringrules)")

    if metrics.get("calibration_buckets"):
        print(f"\n  Calibration (predicted prob vs. actual UP rate):")
        print(f"  {'Range':<12} {'N':>6} {'Pred':>6} {'Actual':>8} {'Gap':>8}")
        print(f"  {'~' * 44}")
        for b in metrics["calibration_buckets"]:
            flag = "  <- well calibrated" if abs(b["gap"]) < 0.03 else ""
            print(f"  {b['pred_range']:<12} {b['n']:>6} {b['pred_prob']:>6.2f} "
                  f"{b['actual_rate']:>8.4f} {b['gap']:>+8.4f}{flag}")

    print(f"{'=' * 60}\n")
