# ==========================================================================
# DEPRECATED — DO NOT MODIFY
# This file is superseded by pattern_engine/ (v2.1). It remains in the repo
# root ONLY because trading_system/signal_adapter.py::simulate_signals_from_val_db()
# still imports from it.  That dependency will be eliminated in Phase 2 when
# simulate_signals_from_val_db() is migrated to use PatternEngine directly.
# Once the Phase 2 migration is complete, move this file to archive/.
# See TECH_DEBT_AUDIT.md item A1 for details.
# ==========================================================================
"""
strategy.py — System A: Historical Analogue Matching, Calibration & Live Signals

VISION:
  Take end-of-day updated financial data across 52 US tickers.
  For each ticker, compute its current return fingerprint and find the
  historical moments across all 52 stocks and 25 years that looked
  most like today. Project forward: what did those historical twins do
  over the next 7 days? Rank the universe by match quality and signal
  strength to identify the highest-confidence opportunities.

THE 5-STEP PIPELINE:
  1. Profiling:      Compute 8-feature return vector for target stock/date
  2. Analogue Search: K-NN search (Euclidean, ball_tree) for nearest twins
  3. Regime Filter:  Restrict to same macro regime (bull/bear via SPY ret_90d)
  4. Calibration:    Platt or isotonic regression maps raw P(up) → calibrated P
  5. Signal:         Three-filter rule (MIN_MATCHES + AGREEMENT + THRESHOLD)

PRODUCTION MODE:
  run_live_signals() — fetches today's end-of-day data, scores all 52 tickers,
  returns ranked BUY/SELL signals with match count, calibrated probability,
  and top-5 historical analogues per ticker.

RESEARCH MODES:
  run_analogue_sweep()     — parameter search, results → results_analogue.tsv
  run_platt_sweep()        — calibration fraction optimisation
  run_platt_walkforward()  — walk-forward with train-set Platt calibration
  run_regime_walkforward() — walk-forward with regime filter + Platt/isotonic

LOCKED SETTINGS (evidence in PROJECT_GUIDE.md):
  Distance metric:   Euclidean (cosine collapsed — 93.3% within dist 0.20)
  MAX_DISTANCE:      1.1019 (quantile-calibrated, AvgK ~42)
  DISTANCE_WEIGHTING: "uniform" (beats inverse — sweep 1)
  Feature set:       return-only 8 cols (supplements 3.5x worse — sweeps 2/3)
  CONFIDENCE_THRESHOLD: 0.65 (best binary accuracy — sweep 1)
  SAME_SECTOR_ONLY:  False (sector-only worst BSS — sweep 1)
  Calibration:       Platt cal_frac=0.76 on training set (generalises across regimes)
  Regime filter:     SPY ret_90d > 0 = bull, <= 0 = bear (fixes 2022 Bear fold)

CURRENT STATUS (2026-03-16):
  Positive BSS:    +0.00103 Platt / +0.00074 Isotonic (fold 2024, v4 walk-forward)
  Walk-forward v4: Regime labels CONFIRMED CORRECT across all 6 folds.
                   Zero trades at threshold=0.65 — regime filter compresses P(up).
                   Overnight testing threshold=0.60/0.55 to restore trade coverage.
  Regime labels verified:
    2019: 11024 bull / 2080 bear  |  2020: 8788 bull / 4368 bear (COVID mixed)
    2021: 13104 bull / 0 bear     |  2022: 2756 bull / 10296 bear (correctly bear)
    2023: 11596 bull / 1404 bear  |  2024: 13052 bull / 52 bear
  Test suite: 52 tests passing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# Graceful import for scoringrules — install with: pip install scoringrules
try:
    import scoringrules
    SCORINGRULES_AVAILABLE = True
except ImportError:
    SCORINGRULES_AVAILABLE = False
    print("[WARNING] scoringrules not installed. Run: pip install scoringrules")
    print("          CRPS scoring will be skipped until installed.")


# ============================================================
# HYPERPARAMETERS — autoresearch modifies these
# ============================================================

# --- Matching Algorithm ---
TOP_K = 50                       # Neighbours retrieved per query
MAX_DISTANCE = 1.1019            # Euclidean distance ceiling — quantile-calibrated (AvgK ~42)
                                 # Locked from Platt sweep 2. Do not change without re-running sweep.
DISTANCE_WEIGHTING = "uniform"   # "uniform" beats "inverse" — locked from sweep 1
MIN_MATCHES = 10                 # Minimum valid matches required to generate a signal

# --- Cohort Filtering ---
SAME_SECTOR_ONLY = False       # If True, only match within same sector (try: True/False)
EXCLUDE_SAME_TICKER = True     # Don't match a stock against its own history (prevents autocorrelation)

# --- Forward Projection ---
PROJECTION_HORIZON = "fwd_7d_up"  # Which forward window to predict (try: fwd_1d_up, fwd_3d_up, fwd_7d_up, fwd_14d_up, fwd_30d_up)

# --- Signal Generation ---
CONFIDENCE_THRESHOLD = 0.65    # Locked from sweep 1 — best binary accuracy trade-off
                               # 0.65 = at least 65% of historical twins went UP for BUY signal
AGREEMENT_SPREAD = 0.10        # Inert at AvgK ~42 but kept for signal quality filtering

# --- Feature Weights (which parts of the return vector matter most) ---
# These multiply each feature before distance calculation.
# Higher weight = that feature influences matching more.
FEATURE_WEIGHTS = {
    "ret_1d": 1.0,
    "ret_3d": 1.0,
    "ret_7d": 1.5,      # Medium-term trend matters more
    "ret_14d": 1.5,
    "ret_30d": 1.0,
    "ret_45d": 0.8,
    "ret_60d": 0.8,
    "ret_90d": 0.5,      # Very long-term trend matters less
    "vol_10d": 1.2,      # Volatility regime is important context
    "vol_30d": 1.0,
    "vol_ratio": 1.0,
    "vol_abnormal": 0.8,
    "rsi_14": 1.0,
    "atr_14": 0.8,
    "price_vs_sma20": 1.2,  # Mean reversion signal
    "price_vs_sma50": 1.0,
}

# --- Walk-Forward Validation ---
# Expanding windows: each fold adds one more year of training data.
# Fold structure: train through year X, validate on year X+1.
# 2020 is the most important fold — COVID crash tests regime robustness.
WALKFORWARD_FOLDS = [
    {"train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31", "label": "2019"},
    {"train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31", "label": "2020 (COVID)"},
    {"train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31", "label": "2021"},
    {"train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31", "label": "2022 (Bear)"},
    {"train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31", "label": "2023"},
    {"train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-12-31", "label": "2024 (Standard Val)"},
]

# --- Paths ---
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Analogue runs write to a separate TSV from the neural-net experiments.
# The old results.tsv mixed two incompatible schemas (21 neural-net columns vs
# 30 analogue columns), causing pandas ParserErrors and misaligned column reads.
# All analogue experiments now go to results_analogue.tsv.
ANALOGUE_RESULTS_FILE = "results_analogue.tsv"

# --- Runtime performance (added after Windows/Python 3.12 joblib deadlock) ---
# The original n_jobs=-1 + brute algorithm caused repeated sklearn parallel
# warnings and a hard hang on Euclidean queries on Windows + Python 3.12.
# Root cause: joblib threading backend incompatibility with sklearn's chunked
# pairwise distance path for Euclidean distance.
#
# NN_JOBS = 1:       removes the joblib threading path entirely — safe on all platforms.
# BATCH_SIZE = 256:  pre-transforms all val rows once, then calls kneighbors() in
#                    chunks rather than one row at a time. Reduces Python overhead
#                    and joblib dispatch count by ~10-50x.
NN_JOBS = 1
BATCH_SIZE = 256


# ============================================================
# SECTOR MAP (for cohort filtering)
# ============================================================

SECTOR_MAP = {
    "SPY": "Index", "QQQ": "Index",
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMZN": "Tech",
    "GOOGL": "Tech", "META": "Tech", "TSLA": "Tech", "AVGO": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "AMD": "Tech",
    "NFLX": "Tech", "INTC": "Tech", "CSCO": "Tech", "QCOM": "Tech",
    "TXN": "Tech", "MU": "Tech", "PYPL": "Tech",
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance",
    "MS": "Finance", "V": "Finance", "MA": "Finance", "AXP": "Finance",
    "BRK-B": "Finance",
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health",
    "MRK": "Health", "PFE": "Health", "TMO": "Health", "ISRG": "Health",
    "AMGN": "Health", "GILD": "Health",
    "WMT": "Consumer", "COST": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "PEP": "Consumer", "HD": "Consumer",
    "DIS": "Industrial", "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial",
    "XOM": "Energy", "CVX": "Energy",
}


# ============================================================
# ANALOGUE MATCHING ENGINE
# ============================================================

def apply_feature_weights(X, feature_cols, weights=FEATURE_WEIGHTS):
    """
    Apply feature weights before distance calculation.
    Higher weight = that feature influences matching more.
    """
    X_weighted = X.copy()
    for i, col in enumerate(feature_cols):
        w = weights.get(col, 1.0)
        X_weighted[:, i] *= w
    return X_weighted


def project_forward(matches, horizon=PROJECTION_HORIZON, weighting=DISTANCE_WEIGHTING):
    """
    Given a set of historical analogues, compute the forward projection.

    Returns:
        dict with:
          - probability_up: weighted % of analogues that went UP
          - mean_return: weighted average forward return
          - median_return: median forward return
          - agreement: how strongly the analogues agree (0.5 = no signal, 1.0 = unanimous)
          - n_matches: number of valid matches used
          - ensemble_returns: raw array of analogue forward returns (for CRPS scoring)
    """
    if len(matches) == 0:
        return {
            "probability_up": 0.5, "mean_return": 0.0, "median_return": 0.0,
            "agreement": 0.0, "n_matches": 0, "ensemble_returns": np.array([]),
        }

    return_col = horizon.replace("_up", "")

    binary_outcomes = matches[horizon].values
    return_outcomes = matches[return_col].values
    distances = matches["distance"].values

    if weighting == "inverse":
        weights = 1.0 / (distances + 0.01)
        weights = weights / weights.sum()
        probability_up = np.average(binary_outcomes, weights=weights)
        mean_return = np.average(return_outcomes, weights=weights)
    else:
        probability_up = binary_outcomes.mean()
        mean_return = return_outcomes.mean()

    median_return = np.median(return_outcomes)
    agreement = abs(probability_up - 0.5) * 2

    return {
        "probability_up": float(probability_up),
        "mean_return": float(mean_return),
        "median_return": float(median_return),
        "agreement": float(agreement),
        "n_matches": len(matches),
        "ensemble_returns": return_outcomes,  # Raw ensemble for CRPS
    }


# ============================================================
# PROBABILISTIC EVALUATION (NEW — 2026-03-16)
# ============================================================

def brier_score(y_true, y_pred_proba):
    """
    Brier score: mean squared error between predicted probability and binary outcome.
    Range: 0 (perfect) to 1 (worst). A naive 0.5 forecast scores 0.25.
    Lower is better.
    """
    return float(np.mean((y_pred_proba - y_true) ** 2))


def brier_skill_score(y_true, y_pred_proba):
    """
    Brier Skill Score: how much better is the model vs. always predicting the base rate?
    BSS = 1 - (BS_model / BS_climatology)
    Range: -inf to 1.0. Positive = better than climatology. 0 = no skill.
    """
    bs_model = brier_score(y_true, y_pred_proba)
    # Climatology forecast = base rate for every prediction
    base_rate = y_true.mean()
    bs_clim = brier_score(y_true, np.full_like(y_pred_proba, base_rate))
    if bs_clim == 0:
        return 0.0
    return float(1.0 - (bs_model / bs_clim))


def compute_crps(y_true_returns, ensemble_returns_list):
    """
    CRPS (Continuous Ranked Probability Score) using the analogue ensemble.

    For each query, we have a set of K analogue forward returns (the ensemble).
    CRPS measures the distance between the predicted CDF and the actual outcome.
    Lower is better. A perfect point forecast scores 0.

    Why this matters over binary accuracy:
      - A model that says "60% up" when it goes up is better than one that
        says "51% up" when it goes up — but binary accuracy treats them equal.
      - CRPS rewards confident correct predictions and penalises confident wrong ones.

    Args:
        y_true_returns: array of actual forward returns (continuous, not binary)
        ensemble_returns_list: list of arrays, each array = analogue returns for that query
    """
    if not SCORINGRULES_AVAILABLE:
        return None

    crps_scores = []
    for obs, ensemble in zip(y_true_returns, ensemble_returns_list):
        if len(ensemble) < 2:
            continue
        try:
            score = scoringrules.crps_ensemble(float(obs), ensemble.astype(float))
            crps_scores.append(score)
        except Exception:
            continue

    if not crps_scores:
        return None
    return float(np.mean(crps_scores))


def evaluate_from_signals(y_true_binary, probabilities, signals):
    """
    Evaluate using the ACTUAL signals array produced by generate_signal().

    This is the correct evaluator. It counts as "confident" only the trades
    that generate_signal() would actually take — i.e. those that pass ALL
    three filters: MIN_MATCHES, AGREEMENT_SPREAD, and CONFIDENCE_THRESHOLD.

    Previously, prepare.evaluate_predictions() scored "confident trades" using
    only the probability threshold, which produces a larger (optimistic) set
    than the signal logic would ever trade in practice.

    Args:
        y_true_binary: array of 0/1 ground truth
        probabilities: array of P(up) from analogue matching
        signals:       array of "BUY"/"SELL"/"HOLD" strings from generate_signal()

    Returns:
        dict of classification metrics aligned with actual signal logic
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred_all = (probabilities >= 0.5).astype(int)
    acc_all = accuracy_score(y_true_binary, y_pred_all)

    # "Confident" = any trade generate_signal() actually made (BUY or SELL)
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


def evaluate_probabilistic(y_true_binary, y_true_returns, probabilities,
                            ensemble_returns_list, signals, horizon_label=""):
    """
    Full probabilistic evaluation suite.
    Combines signal-aligned classification metrics with proper scoring rules.

    Args:
        y_true_binary:       array of 0/1 (did price go up?)
        y_true_returns:      array of actual continuous returns
        probabilities:       array of predicted P(up) from analogue matching
        ensemble_returns_list: list of arrays (one per query), raw analogue returns
        signals:             array of "BUY"/"SELL"/"HOLD" from generate_signal()
        horizon_label:       e.g. "fwd_7d" for reporting

    Returns:
        dict with all metrics
    """
    # Signal-aligned classification metrics (fixes Gemini's evaluation mismatch bug)
    class_metrics = evaluate_from_signals(y_true_binary, probabilities, signals)

    # Brier score (proper scoring rule for binary probabilities)
    bs = brier_score(y_true_binary, probabilities)
    bss = brier_skill_score(y_true_binary, probabilities)

    # CRPS (proper scoring rule for full distributions)
    crps = compute_crps(y_true_returns, ensemble_returns_list)

    # Calibration: bucket predictions and check actual rates
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


def compute_calibration(y_true, y_pred_proba, n_buckets=5):
    """
    Reliability / calibration check.
    Splits predictions into n_buckets probability ranges and checks whether
    the actual UP rate matches the predicted probability.

    Perfect calibration: if model says 60%, 60% of those actually went up.
    Returns a list of dicts — one per bucket — for inspection.
    """
    buckets = []
    edges = np.linspace(0, 1, n_buckets + 1)

    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        # Last bucket uses <= hi to capture probability exactly 1.0 (edge case fix)
        if i == n_buckets - 1:
            mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
        else:
            mask = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if mask.sum() == 0:
            continue
        actual_rate = y_true[mask].mean()
        pred_center = (lo + hi) / 2
        buckets.append({
            "pred_range": f"{lo:.1f}-{hi:.1f}",
            "n": int(mask.sum()),
            "pred_prob": round(float(pred_center), 2),
            "actual_rate": round(float(actual_rate), 4),
            "gap": round(float(actual_rate - pred_center), 4),
        })

    return buckets


def print_probabilistic_metrics(metrics, label=""):
    """Print full probabilistic evaluation in a clean format."""
    h = metrics.get("horizon", "")
    tag = f" — {label}" if label else ""
    tag += f" | {h}" if h else ""

    print(f"\n{'='*60}")
    print(f"  PROBABILISTIC EVALUATION{tag}")
    print(f"{'='*60}")
    print(f"  Total samples:        {metrics['total_samples']:,}")
    print(f"  Accuracy (all):       {metrics['accuracy_all']:.1%}")
    print(f"  Confident trades:     {metrics['confident_trades']:,} ({metrics['confident_pct']:.1%})")
    print(f"  Accuracy (confident): {metrics['accuracy_confident']:.1%}  ← primary metric")
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
        print(f"  {'~'*44}")
        for b in metrics["calibration_buckets"]:
            flag = "  ← well calibrated" if abs(b["gap"]) < 0.03 else ""
            print(f"  {b['pred_range']:<12} {b['n']:>6} {b['pred_prob']:>6.2f} "
                  f"{b['actual_rate']:>8.4f} {b['gap']:>+8.4f}{flag}")

    print(f"{'='*60}\n")


# ============================================================
# SIGNAL GENERATION
# ============================================================

def generate_signal(projection, threshold=CONFIDENCE_THRESHOLD,
                    min_agreement=AGREEMENT_SPREAD, min_matches=MIN_MATCHES):
    """
    Convert a forward projection into a BUY/SELL/HOLD signal.

    Rules (learned from Phase 1):
      - Need at least min_matches valid analogues (prevents thin-data trades)
      - Probability must exceed threshold (0.55 = 55% of twins went up)
      - Agreement must exceed min_agreement (prevents 51/49 split trades)
    """
    prob = projection["probability_up"]
    agree = projection["agreement"]
    n = projection["n_matches"]

    if n < min_matches:
        return "HOLD", "insufficient_matches"
    if agree < min_agreement:
        return "HOLD", "low_agreement"
    if prob >= threshold:
        return "BUY", f"prob={prob:.3f}_agree={agree:.3f}"
    elif prob <= (1 - threshold):
        return "SELL", f"prob={prob:.3f}_agree={agree:.3f}"
    else:
        return "HOLD", f"prob={prob:.3f}_below_threshold"


# ============================================================
# CORE MATCHING LOOP (shared by run_experiment and run_walkforward)
# ============================================================

def _run_matching_loop(train_db, val_db, scaler, feature_cols, verbose=1,
                       top_k=None, max_distance=None, distance_weighting=None,
                       projection_horizon=None, same_sector_only=None,
                       confidence_threshold=None, agreement_spread=None,
                       min_matches=None, metric_override=None,
                       feature_cols_override=None, exclude_same_ticker=None,
                       regime_filter=False):
    """
    Inner matching loop — runs analogue search on val_db using train_db.
    Shared between run_experiment(), run_walkforward(), and run_analogue_sweep().

    Explicit config parameters override module-level globals when provided.
    This is the correct pattern for the sweep: pass values directly rather than
    mutating globals, which avoids the __main__ vs imported-module split and
    the Python default-argument-binding issue.

    metric_override:        "euclidean" to test Euclidean distance vs default cosine
    feature_cols_override:  list of column names to use instead of full feature_cols
                            (used for return-only ablation)
    exclude_same_ticker:    if True, remove same-ticker rows from candidate set

    SCALER NOTE: When feature_cols_override is provided, the saved scaler (fitted
    on the full 16-feature set) cannot be used — shape mismatch will crash.
    A local scaler is refitted on the training subset of the active feature set.
    This is safe as long as we fit only on train_db (no leakage).

    Returns:
        probabilities, signals, reasons, n_matches, mean_returns, ensemble_list
        where ensemble_list is a list of arrays (one per val query) of raw returns.
    """
    # Resolve effective config — explicit args take priority over module globals
    _top_k              = top_k              if top_k              is not None else TOP_K
    _max_distance       = max_distance       if max_distance       is not None else MAX_DISTANCE
    _distance_weighting = distance_weighting if distance_weighting is not None else DISTANCE_WEIGHTING
    _projection_horizon = projection_horizon if projection_horizon is not None else PROJECTION_HORIZON
    _same_sector_only   = same_sector_only   if same_sector_only   is not None else SAME_SECTOR_ONLY
    _threshold          = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    _agreement_spread   = agreement_spread   if agreement_spread   is not None else AGREEMENT_SPREAD
    _min_matches        = min_matches        if min_matches        is not None else MIN_MATCHES
    _metric             = metric_override    if metric_override    is not None else "cosine"
    _fcols              = feature_cols_override if feature_cols_override is not None else feature_cols
    _exclude_ticker     = exclude_same_ticker if exclude_same_ticker is not None else EXCLUDE_SAME_TICKER

    # Scaler selection — fatal bug fix (Gemini, 2026-03-16):
    # The saved scaler was fitted on the full 16-feature set during prepare.py.
    # If feature_cols_override reduces the feature count (e.g. return-only = 8 cols),
    # calling scaler.transform() on the smaller matrix causes a shape mismatch crash.
    # Solution: refit a local StandardScaler on the training data for this feature set.
    # This is safe — we fit only on train_db (no val/test data contamination).
    if feature_cols_override is not None:
        active_scaler = StandardScaler()
        active_scaler.fit(train_db[_fcols].values)
    else:
        active_scaler = scaler   # Use the pre-fitted scaler for the standard 16-feature case

    X_train_scaled   = active_scaler.transform(train_db[_fcols].values)
    X_train_weighted = apply_feature_weights(X_train_scaled, _fcols)

    # Metric-aware algorithm selection.
    # cosine must stay "brute" (not supported by tree indexes).
    # euclidean can use "ball_tree" which avoids the joblib/threading path that
    # caused the Windows + Python 3.12 parallel warning flood and hang.
    if _metric == "euclidean":
        nn_algorithm = "ball_tree"
    else:
        nn_algorithm = "brute"

    nn_index = NearestNeighbors(
        n_neighbors=min(_top_k * 3, len(train_db)),
        metric=_metric,
        algorithm=nn_algorithm,
        n_jobs=NN_JOBS,   # 1 = no threading, avoids joblib deadlock on Windows
    )
    nn_index.fit(X_train_weighted)

    probabilities = []
    all_signals = []
    all_reasons = []
    all_n_matches = []
    all_mean_returns = []
    all_ensembles = []

    # Pre-transform the entire validation set once (not per row).
    # Then query kneighbors() in chunks of BATCH_SIZE.
    # This reduces Python overhead and joblib dispatch count by ~10-50x vs
    # the original per-row loop.
    X_val_scaled   = active_scaler.transform(val_db[_fcols].values)
    X_val_weighted = apply_feature_weights(X_val_scaled, _fcols)

    import time
    start_time = time.time()
    n_val = len(val_db)

    # ── Regime conditioning (Gemini recommendation, 2026-03-16) ────────────
    # Pre-compute SPY regime label for every row in train_db and val_db.
    # Regime = bull (SPY ret_90d > 0) or bear (SPY ret_90d <= 0).
    # Applied as a post-search filter: after distance search, discard candidates
    # whose market regime differs from the query date's regime.
    # This directly fixes the 2022 Bear walk-forward failure where bull-market
    # analogues systematically signal UP in a -20% year.
    _regime_labels_train = None
    _regime_labels_val   = None
    if regime_filter:
        # BUG FIX (v3 → v4, 2026-03-16):
        # v3 used train_db SPY rows to label BOTH train AND val queries.
        # When train ends in a bear year (e.g. 2022-12-31), searchsorted
        # clips all 2023 val dates to the last train SPY row → all labeled bear.
        # But 2023 was a +24% bull year — regime labels were systematically wrong.
        #
        # Fix: use val_db SPY rows to label val queries. No leakage — we only
        # read SPY ret_90d (a lagging indicator), not any forward returns.
        # Platt calibrator still trains exclusively on train_db. ✓
        #
        # Evidence: fold 2023 showed "val=0 bull / 13000 bear" (wrong).
        # fold 2024 showed "val=13104 bull / 0 bear" (correct by coincidence —
        # 2023 also ended bull, so projection happened to match 2024 reality).

        # Train analogue regime labels — use train_db SPY only (strict, no leakage)
        spy_train = train_db[train_db["Ticker"] == "SPY"][["Date", "ret_90d"]].copy()
        spy_train = spy_train.sort_values("Date").reset_index(drop=True)

        # Val query regime labels — use val_db SPY rows for accurate classification
        # If val_db has no SPY (e.g. when val_db=train_db in calibration pass),
        # fall back to extending from train_db SPY.
        spy_val = val_db[val_db["Ticker"] == "SPY"][["Date", "ret_90d"]].copy()
        if len(spy_val) > 0:
            spy_val = spy_val.sort_values("Date").reset_index(drop=True)
        else:
            spy_val = spy_train  # fallback: same-source SPY (e.g. train calibration pass)

        if len(spy_train) >= 10:
            # Label train analogues using train SPY
            spy_dates_train = np.array(pd.to_datetime(spy_train["Date"].values), dtype="datetime64[ns]")
            spy_ret90_train = spy_train["ret_90d"].values
            train_dates_arr = np.array(pd.to_datetime(train_db["Date"].values), dtype="datetime64[ns]")
            t_idx = np.searchsorted(spy_dates_train, train_dates_arr, side="right") - 1
            t_idx = np.clip(t_idx, 0, len(spy_ret90_train) - 1)
            _regime_labels_train = (spy_ret90_train[t_idx] > 0).astype(np.int8)

            # Label val queries using val SPY (accurate current-year regime)
            spy_dates_val = np.array(pd.to_datetime(spy_val["Date"].values), dtype="datetime64[ns]")
            spy_ret90_val = spy_val["ret_90d"].values
            val_dates_arr = np.array(pd.to_datetime(val_db["Date"].values), dtype="datetime64[ns]")
            v_idx = np.searchsorted(spy_dates_val, val_dates_arr, side="right") - 1
            v_idx = np.clip(v_idx, 0, len(spy_ret90_val) - 1)
            _regime_labels_val = (spy_ret90_val[v_idx] > 0).astype(np.int8)

            if verbose:
                bull_q = (_regime_labels_val == 1).sum()
                bear_q = (_regime_labels_val == 0).sum()
                bull_t = (_regime_labels_train == 1).sum()
                bear_t = (_regime_labels_train == 0).sum()
                print(f"    Regime filter: val={bull_q} bull / {bear_q} bear queries | "
                      f"train pool={bull_t} bull / {bear_t} bear analogues")
        else:
            if verbose:
                print("    Regime filter: SPY data insufficient in train_db — filter disabled")
    # ────────────────────────────────────────────────────────────────────────

    for batch_start in range(0, n_val, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_val)
        q_batch   = X_val_weighted[batch_start:batch_end]

        distances_batch, indices_batch = nn_index.kneighbors(q_batch)

        for local_i in range(batch_end - batch_start):
            idx     = batch_start + local_i
            row     = val_db.iloc[idx]
            ticker  = row["Ticker"]
            sector  = SECTOR_MAP.get(ticker, None)

            distances = distances_batch[local_i]
            indices   = indices_batch[local_i]

            matches = train_db.iloc[indices].copy()
            matches["distance"] = distances

            # Apply filters using resolved config (not bare globals)
            matches = matches[matches["distance"] <= _max_distance]
            if _exclude_ticker:                                         # parameterised (Gemini fix)
                matches = matches[matches["Ticker"] != ticker]
            if _same_sector_only and sector:
                matches = matches[matches["Ticker"].map(SECTOR_MAP) == sector]
            # Regime filter: keep only analogues from same SPY macro regime
            if _regime_labels_train is not None and _regime_labels_val is not None:
                query_regime = _regime_labels_val[idx]
                match_regimes = _regime_labels_train[indices]
                regime_mask = (match_regimes == query_regime)
                matches = matches[regime_mask[:len(matches)]]
            matches = matches.head(_top_k)

            projection = project_forward(matches, horizon=_projection_horizon,
                                         weighting=_distance_weighting)

            # Pass threshold params explicitly — fixes Python default-arg binding issue.
            signal, reason = generate_signal(
                projection,
                threshold=_threshold,
                min_agreement=_agreement_spread,
                min_matches=_min_matches,
            )

            probabilities.append(projection["probability_up"])
            all_signals.append(signal)
            all_reasons.append(reason)
            all_n_matches.append(projection["n_matches"])
            all_mean_returns.append(projection["mean_return"])
            all_ensembles.append(projection["ensemble_returns"])

        if verbose and batch_end % 2000 < BATCH_SIZE:
            elapsed  = time.time() - start_time
            rate     = batch_end / elapsed if elapsed > 0 else 0
            remaining = (n_val - batch_end) / rate if rate > 0 else 0
            print(f"    {batch_end:,}/{n_val:,} "
                  f"({batch_end/n_val*100:.0f}%) | "
                  f"{rate:.0f} queries/sec | "
                  f"ETA: {remaining/60:.1f} min")

    return (np.array(probabilities), np.array(all_signals), all_reasons,
            all_n_matches, all_mean_returns, all_ensembles)


# ============================================================
# EXPERIMENT RUNNER (original — unchanged interface)
# ============================================================

def run_experiment(experiment_name=None, verbose=1):
    """
    Run a complete analogue matching experiment on the validation set.

    Pipeline:
      1. Load training database + validation database
      2. For each day in validation, find K nearest historical analogues
      3. Project forward to get probability of UP
      4. Generate BUY/SELL/HOLD signal
      5. Evaluate accuracy on confident trades
      6. Log results to results.tsv
    """
    if experiment_name is None:
        experiment_name = f"analogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {experiment_name}")
        print(f"  Method: Historical Analogue Matching")
        print(f"  K={TOP_K} | Max Dist={MAX_DISTANCE} | Weighting={DISTANCE_WEIGHTING}")
        print(f"  Horizon={PROJECTION_HORIZON} | Threshold={CONFIDENCE_THRESHOLD}")
        print(f"  Agreement={AGREEMENT_SPREAD} | Min Matches={MIN_MATCHES}")
        print(f"{'='*60}")

    # --- Load data ---
    if verbose: print("\n[1/4] Loading data...")
    train_db = pd.read_parquet(DATA_DIR / "train_db.parquet")
    val_db = pd.read_parquet(DATA_DIR / "val_db.parquet")

    # NOTE: This uses the scaler saved by prepare.py (fitted on training data only).
    # run_walkforward() refits a fresh scaler per fold from that fold's training slice.
    # The two modes are conceptually consistent (both fit on training data only),
    # but the scaler objects are different instances. Documented for reproducibility.
    scaler = joblib.load(MODEL_DIR / "analogue_scaler.pkl")

    from prepare import FEATURE_COLS

    if verbose:
        print(f"  Training analogues: {len(train_db):,}")
        print(f"  Validation queries: {len(val_db):,}")

    # --- Build index and run matching ---
    if verbose: print("\n[2/4] Building weighted nearest-neighbor index...")
    if verbose: print("\n[3/4] Matching analogues for validation set...")

    (probabilities, all_signals, all_reasons,
     all_n_matches, all_mean_returns, all_ensembles) = _run_matching_loop(
        train_db, val_db, scaler, FEATURE_COLS, verbose=verbose
    )

    buy_count  = (all_signals == "BUY").sum()
    sell_count = (all_signals == "SELL").sum()
    hold_count = (all_signals == "HOLD").sum()

    if verbose:
        print(f"  Signals: BUY={buy_count:,}, SELL={sell_count:,}, HOLD={hold_count:,}")
        print(f"  Avg matches per query: {np.mean(all_n_matches):.1f}")
        print(f"  Avg projected return: {np.mean(all_mean_returns):.4f}")

    # --- Evaluate ---
    if verbose: print("\n[4/4] Evaluating...")

    from prepare import save_results

    horizon_return_col = PROJECTION_HORIZON.replace("_up", "")
    y_true_binary  = val_db[PROJECTION_HORIZON].values
    y_true_returns = val_db[horizon_return_col].values

    metrics = evaluate_probabilistic(
        y_true_binary, y_true_returns, probabilities,
        all_ensembles, signals=all_signals, horizon_label=PROJECTION_HORIZON
    )

    if verbose:
        print_probabilistic_metrics(metrics, label=f"K={TOP_K} | {PROJECTION_HORIZON}")

    from prepare import FEATURE_COLS as _FC
    # Enrich with experiment metadata.
    # COLUMN ORDER NOTE: new fields must be appended AFTER the original 30-column
    # schema, not inserted in the middle. The TSV header is written from the first
    # row ever saved. Inserting new fields before existing ones shifts all subsequent
    # columns right, causing misaligned reads (e.g. avg_matches reads as min_matches).
    metrics.update({
        "experiment_name": experiment_name,
        "method": "analogue_matching",
        "top_k": TOP_K,
        "max_distance": MAX_DISTANCE,
        "distance_weighting": DISTANCE_WEIGHTING,
        "projection_horizon": PROJECTION_HORIZON,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "agreement_spread": AGREEMENT_SPREAD,
        "min_matches": MIN_MATCHES,
        "same_sector_only": SAME_SECTOR_ONLY,
        "exclude_same_ticker": EXCLUDE_SAME_TICKER,
        "avg_matches": float(np.mean(all_n_matches)),
        "avg_projected_return": float(np.mean(all_mean_returns)),
        "buy_signals": int(buy_count),
        "sell_signals": int(sell_count),
        "hold_signals": int(hold_count),
        # New fields appended at end — preserves backward compatibility with existing TSV rows
        "distance_metric": "cosine",
        "feature_set_name": "full",
        "n_features": len(_FC),
    })

    # Strip calibration_buckets before saving to TSV (not TSV-friendly)
    metrics_for_tsv = {k: v for k, v in metrics.items() if k != "calibration_buckets"}
    save_results(metrics_for_tsv, experiment_name,
                 filepath=RESULTS_DIR / ANALOGUE_RESULTS_FILE)

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / f"{experiment_name}_config.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


# ============================================================
# WALK-FORWARD VALIDATION (NEW — 2026-03-16)
# ============================================================

def run_walkforward(experiment_name=None, verbose=1):
    """
    Expanding-window walk-forward validation across 6 market regimes.

    Each fold trains on all data up to year X and validates on year X+1.
    This answers: "Is the analogue signal stable, or did it only work in 2024?"

    Key folds to watch:
      - 2020 (COVID): Does the signal survive a regime-breaking crash?
      - 2022 (Bear):  Does it work in a sustained downtrend?
      - 2024:         Matches run_experiment() — the standard val split.

    All fold results are logged to results.tsv and a summary table is printed.
    A stable strategy should have consistent CRPS and Brier Skill Score
    across folds — not just good numbers on one lucky year.
    """
    if experiment_name is None:
        experiment_name = f"walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD VALIDATION: {experiment_name}")
    print(f"  {len(WALKFORWARD_FOLDS)} folds | K={TOP_K} | Horizon={PROJECTION_HORIZON}")
    print(f"  Threshold={CONFIDENCE_THRESHOLD} | Max Dist={MAX_DISTANCE}")
    print(f"{'='*65}")

    # Load the full database once — folds are carved out per iteration
    print("\n  Loading full analogue database...")
    full_db_path = DATA_DIR / "full_analogue_db.parquet"
    if not full_db_path.exists():
        raise FileNotFoundError(
            "full_analogue_db.parquet not found. Run prepare.py first.\n"
            "It should be saved alongside train_db.parquet and val_db.parquet."
        )

    full_db = pd.read_parquet(full_db_path)
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    from prepare import FEATURE_COLS, save_results

    horizon_return_col = PROJECTION_HORIZON.replace("_up", "")
    fold_results = []

    for fold_idx, fold in enumerate(WALKFORWARD_FOLDS):
        label     = fold["label"]
        train_end = fold["train_end"]
        val_start = fold["val_start"]
        val_end   = fold["val_end"]

        print(f"\n  ── Fold {fold_idx+1}/{len(WALKFORWARD_FOLDS)}: {label} ──")
        print(f"     Train: start → {train_end} | Val: {val_start} → {val_end}")

        train_db = full_db[full_db["Date"] <= train_end].copy().reset_index(drop=True)
        val_db   = full_db[(full_db["Date"] >= val_start) &
                           (full_db["Date"] <= val_end)].copy().reset_index(drop=True)

        if len(train_db) < 1000:
            print(f"     SKIP — insufficient training data ({len(train_db)} rows)")
            continue
        if len(val_db) < 100:
            print(f"     SKIP — insufficient validation data ({len(val_db)} rows)")
            continue

        print(f"     Train: {len(train_db):,} rows | Val: {len(val_db):,} rows")

        # Refit scaler on this fold's training data (no future leakage)
        scaler = StandardScaler()
        scaler.fit(train_db[FEATURE_COLS].values)

        # Run matching
        verbose_inner = 1 if verbose >= 2 else 0
        (probabilities, all_signals, _, all_n_matches,
         all_mean_returns, all_ensembles) = _run_matching_loop(
            train_db, val_db, scaler, FEATURE_COLS, verbose=verbose_inner
        )

        # Evaluate
        y_true_binary  = val_db[PROJECTION_HORIZON].values
        y_true_returns = val_db[horizon_return_col].values

        metrics = evaluate_probabilistic(
            y_true_binary, y_true_returns, probabilities,
            all_ensembles, signals=all_signals, horizon_label=PROJECTION_HORIZON
        )

        buy_count  = (all_signals == "BUY").sum()
        sell_count = (all_signals == "SELL").sum()
        hold_count = (all_signals == "HOLD").sum()

        # Print inline summary
        crps_str = f"{metrics['crps']:.5f}" if metrics.get("crps") is not None else "N/A"
        print(f"     Trades={metrics['confident_trades']:,} | "
              f"Acc(conf)={metrics['accuracy_confident']:.1%} | "
              f"BSS={metrics['brier_skill_score']:+.4f} | "
              f"CRPS={crps_str}")

        # Calibration summary (worst gap across buckets)
        if metrics.get("calibration_buckets"):
            max_gap = max(abs(b["gap"]) for b in metrics["calibration_buckets"])
            print(f"     Calibration max gap: {max_gap:.4f} "
                  f"({'OK' if max_gap < 0.05 else 'POOR — predictions poorly calibrated'})")

        # Log to results.tsv
        fold_name = f"{experiment_name}_fold_{label.replace(' ', '_').replace('(', '').replace(')', '')}"
        row = {k: v for k, v in metrics.items() if k != "calibration_buckets"}
        row.update({
            "experiment_name": fold_name,
            "method": "walkforward_analogue",
            "fold_label": label,
            "train_end": train_end,
            "val_year": val_start[:4],
            # Full experiment definition — matches run_experiment() logging (Gemini fix)
            "top_k": TOP_K,
            "max_distance": MAX_DISTANCE,
            "distance_weighting": DISTANCE_WEIGHTING,
            "projection_horizon": PROJECTION_HORIZON,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "agreement_spread": AGREEMENT_SPREAD,
            "min_matches": MIN_MATCHES,
            "same_sector_only": SAME_SECTOR_ONLY,
            "exclude_same_ticker": EXCLUDE_SAME_TICKER,
            "avg_matches": float(np.mean(all_n_matches)),
            "avg_projected_return": float(np.mean(all_mean_returns)),
            "buy_signals": int(buy_count),
            "sell_signals": int(sell_count),
            "hold_signals": int(hold_count),
        })
        save_results(row, fold_name,
                     filepath=RESULTS_DIR / ANALOGUE_RESULTS_FILE)
        fold_results.append({"label": label, **metrics,
                              "buy_signals": int(buy_count),
                              "avg_matches": float(np.mean(all_n_matches))})

    # ── Summary Table ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Fold':<20} {'Trades':>7} {'Acc(c)':>7} {'BSS':>8} {'CRPS':>10} {'Avg K':>7}")
    print(f"  {'~'*55}")

    acc_list, bss_list, crps_list = [], [], []

    for r in fold_results:
        crps_str = f"{r['crps']:.5f}" if r.get("crps") is not None else "     N/A"
        bss_val  = r.get("brier_skill_score")
        bss_str  = f"{bss_val:>+8.4f}" if bss_val is not None else "     N/A"
        print(f"  {r['label']:<20} {r['confident_trades']:>7} "
              f"{r['accuracy_confident']:>7.1%} "
              f"{bss_str} "
              f"{crps_str:>10} "
              f"{r['avg_matches']:>7.1f}")

        # ACCURACY: exclude zero-trade folds only from accuracy statistics.
        # A fold with 0 trades has no meaningful accuracy — the model never
        # committed, so counting 0.0% would corrupt the mean.
        if r["confident_trades"] > 0:
            acc_list.append(r["accuracy_confident"])
        else:
            print(f"    (accuracy excluded from mean — 0 trades in this fold)")

        # BSS and CRPS: unconditional — include even for zero-trade folds.
        # These metrics score the full probability distribution across all 13K+
        # validation queries, not just the traded subset. A fold that generates
        # 0 trades still produced probability outputs that can be scored.
        # NaN guard required: np.nan passes `is not None`, so check explicitly.
        if bss_val is not None and not np.isnan(bss_val):
            bss_list.append(bss_val)

        crps_val = r.get("crps")
        if crps_val is not None and not np.isnan(crps_val):
            crps_list.append(crps_val)

    if acc_list:
        print(f"  {'~'*55}")
        crps_mean_str = f"{np.mean(crps_list):.5f}" if crps_list else "N/A"
        bss_mean_str  = f"{np.mean(bss_list):>+8.4f}" if bss_list else "     N/A"
        bss_std_str   = f"{np.std(bss_list):>8.4f}"  if bss_list else "     N/A"
        n_included    = len(acc_list)
        n_total       = len(fold_results)
        excluded_note = f" ({n_total - n_included} fold(s) excluded from Acc)" if n_included < n_total else ""
        print(f"  {'MEAN':<20} {'':>7} "
              f"{np.mean(acc_list):>7.1%} "
              f"{bss_mean_str} "
              f"{crps_mean_str:>10}"
              f"{excluded_note}")
        print(f"  {'STD':<20} {'':>7} "
              f"{np.std(acc_list):>7.1%} "
              f"{bss_std_str}")
        print(f"\n  Stability check (std of Acc across folds):")
        std_acc = np.std(acc_list)
        if std_acc < 0.02:
            print(f"  ✓ STABLE — std={std_acc:.3f}. Signal is consistent across regimes.")
        elif std_acc < 0.05:
            print(f"  ~ MODERATE — std={std_acc:.3f}. Some regime sensitivity, watch 2020/2022.")
        else:
            print(f"  ✗ UNSTABLE — std={std_acc:.3f}. Results vary too much across regimes.")

    print(f"{'='*65}\n")
    return fold_results


# ============================================================
# PARAMETER SWEEP (v2 — micro-distance + Euclidean, 2026-03-16)
# ============================================================

def run_analogue_sweep(budget_minutes=60, verbose=1):
    """
    Time-budgeted parameter sweep for the analogue matching engine.

    SWEEP 1 LEARNINGS (results_analogue.tsv):
      - All 9 configs produced negative BSS — parameter tuning is not the fix
      - AvgK = 50.0 at dist=0.35/0.3 and 49.3 at dist=0.2: filter never binding
        The cosine distance distribution on StandardScaler features is tightly
        clustered near zero — most stock vectors look like twins to cosine distance
      - uniform > inverse weighting (best BSS = -0.0286 vs -0.031 baseline)
      - smaller K (20/25) made BSS materially WORSE — fewer analogues = worse calibration
      - sector-only was the worst performer — sector labels are too coarse
      - threshold tuning (0.55→0.65) improves binary accuracy but doesn't move BSS

    SWEEP 2 FOCUS — answer three questions in order:
      Q1: Is the distance filter EVER binding? (micro-distance sweep)
          If AvgK still = 50 at dist=0.05, the representation itself is the problem.
          If AvgK drops to 10-30 at dist=0.05, there is signal in tight neighborhoods.
      Q2: Does Euclidean distance discriminate better than cosine?
          Cosine ignores magnitude — a 5% move and a 0.5% move look identical in shape.
          Euclidean penalises both direction and magnitude differences.
      Q3: Does removing supplementary features (return-only) change anything?
          The 8 return windows may be cleaner signal than the mixed 16-feature vector.

    LOCKED FROM SWEEP 1 (best settings):
      - DISTANCE_WEIGHTING = "uniform"  (beat inverse on BSS)
      - SAME_SECTOR_ONLY = False        (sector-only was worst)
      - CONFIDENCE_THRESHOLD = 0.65    (best binary accuracy trade-off)

    PRIMARY RANKING METRIC: Brier Skill Score
      BSS > 0 = beats base rate | BSS > 0.05 = materially useful
      If no config reaches BSS > 0, the representation needs changing before more sweeps.

    Results → results_analogue.tsv
    """
    import time

    # ── Sweep 2 configs — informed by diagnose_distances.py ────────────
    #
    # DIAGNOSTIC FINDINGS (2026-03-16):
    #   Cosine distance distribution on StandardScaler features:
    #     7.1% of distances < 0.05  → AvgK ~14 at dist=0.05  ← BINDS
    #    44.2% of distances < 0.10  → AvgK ~50 at dist=0.10  ← still saturated
    #    93.3% of distances < 0.20  → filter never meaningful above 0.05
    #
    #   Euclidean distance (raw, mean=1.84, 99th pct=5.54 for normalisation):
    #    ~5% within raw 0.28  → AvgK ~10
    #   ~10% within raw 0.55  → AvgK ~20
    #   ~15% within raw 0.83  → AvgK ~30
    #   Spread ratio 0.515 vs cosine 0.469 — better discrimination
    #   Only 15.4% within normalised 0.2 vs 93.3% for cosine
    #
    #   Cross-sectional encoding: NOT the fix (92.4% ≈ 93.3% — marginal).
    #   Do not spend experiments on it.
    #
    # LOCKED FROM SWEEP 1:
    #   DISTANCE_WEIGHTING = "uniform"  (beat inverse)
    #   SAME_SECTOR_ONLY = False        (sector-only was worst)

    RETURN_ONLY_COLS = [
        "ret_1d","ret_3d","ret_7d","ret_14d",
        "ret_30d","ret_45d","ret_60d","ret_90d"
    ]

    # ── Sweep 3 configs — focused on winner region (2026-03-16) ───────────
    #
    # SWEEP 2 LEARNINGS:
    #   - Euclidean + return-only is the leading config: BSS=-0.0498, Acc=58.5%
    #   - Full features are worse than return-only (supp features add noise)
    #   - Cosine micro-distance (AvgK=5-14): WORSE BSS despite tighter selection
    #     Reason: small ensembles → extreme P(up) → poor calibration
    #   - BSS correlates with AvgK: larger neighborhoods = better-calibrated probs
    #     AvgK=50→BSS=-0.029, AvgK=41→BSS=-0.050, AvgK=11→BSS=-0.236
    #   - CRPS inverts: tight Euclidean (AvgK=5-10) has best CRPS but worst BSS
    #     This means tight neighborhoods produce sharper but miscalibrated distributions
    #
    # SWEEP 3 GOALS:
    #   G1: Find BSS peak for Euclidean+retonly by scanning around the winner (1.0115)
    #   G2: Test whether probability calibration (Platt scaling) fixes the BSS gap
    #   G3: Confirm whether AvgK~40 is genuinely optimal or just less-wrong
    #
    # NAMING CONVENTION (Gemini fix, sweep 2):
    #   euc_r_dX.XXXX = Euclidean, return-only, raw threshold X.XXXX
    #   euc_f_dX.XXXX = Euclidean, full features, raw threshold X.XXXX
    #   Threshold suffix encodes MAX_DISTANCE directly — no implied K value.
    #
    # EUCLIDEAN THRESHOLDS (from diagnose_distances.py quantile table):
    #   Raw 0.8974 → AvgK ~5-8   (5th pct)
    #   Raw 0.9586 → AvgK ~15    (7.5th pct)
    #   Raw 1.0115 → AvgK ~20-40 (10th pct) ← WINNER from sweep 2
    #   Raw 1.0655 → AvgK ~25-35 (12.5th pct, between 1.0115 and 1.1019)
    #   Raw 1.1019 → AvgK ~30    (15th pct)
    #   Raw 1.1794 → AvgK ~35-45 (20th pct)
    #   Raw 1.2457 → AvgK ~50    (25th pct, saturated)

    configs = [
        # ── Tier A: Fine-grained Euclidean+retonly scan around winner ─────
        # Winner was euc_d1.0115_retonly (BSS=-0.0498, AvgK=41.2).
        # Test tighter (0.9586) and looser (1.0655, 1.1019) to find the BSS peak.
        # All configs: retonly features, uniform weighting, threshold=0.65.
        {
            "name": "euc_r_d0.9586",
            "MAX_DISTANCE": 0.9586, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },
        {
            "name": "euc_r_d1.0115",   # Sweep 2 winner — baseline for this scan
            "MAX_DISTANCE": 1.0115, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },
        {
            "name": "euc_r_d1.0655",   # Between winner and 15th pct
            "MAX_DISTANCE": 1.0655, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },
        {
            "name": "euc_r_d1.1019",   # 15th pct — AvgK ~30
            "MAX_DISTANCE": 1.1019, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },
        {
            "name": "euc_r_d1.2457",   # 25th pct — AvgK ~50, saturated baseline
            "MAX_DISTANCE": 1.2457, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },

        # ── Tier B: Threshold sensitivity on winner config ────────────────
        # Binary accuracy improved with higher threshold (55.9% at thr=0.65).
        # Test whether threshold affects BSS at all for Euclidean+retonly.
        {
            "name": "euc_r_d1.0115_t55",
            "MAX_DISTANCE": 1.0115, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.55, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },
        {
            "name": "euc_r_d1.0115_t60",
            "MAX_DISTANCE": 1.0115, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.60, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },

        # ── Tier C: Agreement spread sensitivity ──────────────────────────
        # AGREEMENT_SPREAD has been fixed at 0.10 across all sweeps.
        # Looser (0.05) = more trades, tighter (0.15) = fewer but more confident.
        {
            "name": "euc_r_d1.0115_ag05",
            "MAX_DISTANCE": 1.0115, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.05,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },
        {
            "name": "euc_r_d1.0115_ag15",
            "MAX_DISTANCE": 1.0115, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.15,
            "_metric_override": "euclidean",
            "_feature_cols_override": RETURN_ONLY_COLS,
            "_feature_set_name": "returns_only",
        },

        # ── Tier D: Full-feature Euclidean control ────────────────────────
        # Sweep 2 showed full features worse than retonly on Euclidean.
        # Confirm by running the winner distance with full features.
        # If retonly is consistently better, supplementary features are ruled out.
        {
            "name": "euc_f_d1.0115",
            "MAX_DISTANCE": 1.0115, "TOP_K": 50,
            "SAME_SECTOR_ONLY": False, "DISTANCE_WEIGHTING": "uniform",
            "CONFIDENCE_THRESHOLD": 0.65, "AGREEMENT_SPREAD": 0.10,
            "_metric_override": "euclidean",
            "_feature_set_name": "full",
        },
    ]

    print(f"\n{'='*65}")
    print(f"  ANALOGUE PARAMETER SWEEP")
    print(f"  {len(configs)} experiments | budget={budget_minutes} min")
    print(f"  Primary metric: Brier Skill Score (BSS > 0 = beats base rate)")
    print(f"  Results → results/{ANALOGUE_RESULTS_FILE}")
    print(f"{'='*65}")

    # Load data once
    print("\n  Loading data...")
    train_db = pd.read_parquet(DATA_DIR / "train_db.parquet")
    val_db   = pd.read_parquet(DATA_DIR / "val_db.parquet")
    scaler   = joblib.load(MODEL_DIR / "analogue_scaler.pkl")
    from prepare import FEATURE_COLS, save_results

    session_start = time.time()
    deadline = session_start + (budget_minutes * 60)
    all_results = []

    for i, cfg in enumerate(configs):
        elapsed  = time.time() - session_start
        remaining = deadline - time.time()

        if remaining < 60:
            print(f"\n  Time budget reached ({elapsed/60:.1f} min). Ran {i} experiments.")
            break

        print(f"\n  [{i+1}/{len(configs)}] {cfg['name']} | "
              f"dist={cfg['MAX_DISTANCE']} K={cfg['TOP_K']} "
              f"sector={cfg['SAME_SECTOR_ONLY']} "
              f"thr={cfg['CONFIDENCE_THRESHOLD']} | "
              f"remaining={remaining/60:.0f}min")

        # Pass config explicitly — no global mutation needed.
        try:
            proj_horizon = cfg.get("PROJECTION_HORIZON", PROJECTION_HORIZON)

            (probabilities, all_signals, _, all_n_matches,
             all_mean_returns, all_ensembles) = _run_matching_loop(
                train_db, val_db, scaler, FEATURE_COLS, verbose=0,
                top_k=cfg["TOP_K"],
                max_distance=cfg["MAX_DISTANCE"],
                distance_weighting=cfg["DISTANCE_WEIGHTING"],
                projection_horizon=proj_horizon,
                same_sector_only=cfg["SAME_SECTOR_ONLY"],
                confidence_threshold=cfg["CONFIDENCE_THRESHOLD"],
                agreement_spread=cfg["AGREEMENT_SPREAD"],
                min_matches=MIN_MATCHES,
                metric_override=cfg.get("_metric_override"),
                feature_cols_override=cfg.get("_feature_cols_override"),
            )

            avg_k = float(np.mean(all_n_matches))

            # Early diagnostic: if the distance filter still isn't binding even
            # at micro-distance levels, flag it immediately.
            # From diagnose_distances.py: dist=0.05 cosine → expected AvgK ~14.
            # If AvgK ≥ 47 at dist ≤ 0.05, the representation is collapsed.
            if cfg["MAX_DISTANCE"] <= 0.05 and cfg.get("_metric_override") != "euclidean" and avg_k >= cfg["TOP_K"] * 0.95:
                print(f"    ⚠  AvgK={avg_k:.1f} at cosine dist={cfg['MAX_DISTANCE']} — "
                      f"still not binding. Expected ~14 from diagnostic. "
                      f"Check scaler or feature pipeline.")
            y_true_binary  = val_db[proj_horizon].values
            y_true_returns = val_db[proj_horizon.replace("_up", "")].values

            metrics = evaluate_probabilistic(
                y_true_binary, y_true_returns, probabilities,
                all_ensembles, signals=all_signals, horizon_label=proj_horizon
            )

            buy_c  = (all_signals == "BUY").sum()
            sell_c = (all_signals == "SELL").sum()
            hold_c = (all_signals == "HOLD").sum()

            active_fcols = cfg.get("_feature_cols_override") or FEATURE_COLS
            feature_set_name = cfg.get("_feature_set_name",
                                       "returns_only" if len(active_fcols) == 8 else "full")

            metrics.update({
                "experiment_name": cfg["name"],
                "method": "analogue_sweep",
                "top_k": cfg["TOP_K"],
                "max_distance": cfg["MAX_DISTANCE"],
                "distance_weighting": cfg["DISTANCE_WEIGHTING"],
                "projection_horizon": proj_horizon,
                "confidence_threshold": cfg["CONFIDENCE_THRESHOLD"],
                "agreement_spread": cfg["AGREEMENT_SPREAD"],
                "min_matches": MIN_MATCHES,
                "same_sector_only": cfg["SAME_SECTOR_ONLY"],
                "exclude_same_ticker": EXCLUDE_SAME_TICKER,
                "avg_matches": float(np.mean(all_n_matches)),
                "avg_projected_return": float(np.mean(all_mean_returns)),
                "buy_signals": int(buy_c),
                "sell_signals": int(sell_c),
                "hold_signals": int(hold_c),
                # New fields at end — preserves TSV column alignment with earlier rows
                "distance_metric": cfg.get("_metric_override") or "cosine",
                "feature_set_name": feature_set_name,
                "n_features": len(active_fcols),
            })

            row = {k: v for k, v in metrics.items() if k != "calibration_buckets"}
            save_results(row, cfg["name"],
                         filepath=RESULTS_DIR / ANALOGUE_RESULTS_FILE)
            all_results.append(metrics)

            bss_val = metrics.get("brier_skill_score")
            bss_str = f"{bss_val:+.4f}" if bss_val is not None else "N/A"
            bss_col = "✓" if (bss_val is not None and bss_val > 0) else "✗"
            print(f"    {bss_col} Trades={metrics['confident_trades']:,} "
                  f"Acc={metrics['accuracy_confident']:.1%} "
                  f"BSS={bss_str} "
                  f"AvgK={metrics['avg_matches']:.1f}")

        except Exception as e:
            print(f"    FAILED: {e}")

    # ── Leaderboard ──────────────────────────────────────────
    total_time = time.time() - session_start
    print(f"\n{'='*65}")
    print(f"  SWEEP LEADERBOARD — {len(all_results)} runs in {total_time/60:.1f} min")
    print(f"  Ranked by: BSS (primary) → accuracy (secondary)")
    print(f"{'='*65}\n")

    ranked = sorted(all_results, key=lambda r: (
        r["brier_skill_score"] if r.get("brier_skill_score") is not None else -99,
        r["accuracy_confident"],
    ), reverse=True)

    print(f"  {'Name':<26} {'Trades':>7} {'Acc(c)':>7} {'BSS':>9} {'CRPS':>9} {'AvgK':>6}")
    print(f"  {'─'*65}")
    for j, r in enumerate(ranked):
        bss = r.get('brier_skill_score')
        bss = bss if bss is not None else 0.0
        crps = r.get('crps') or 0
        marker = " ← BEST" if j == 0 else ""
        bss_flag = "✓" if bss > 0 else "✗"
        print(f"  {r['experiment_name']:<26} {r['confident_trades']:>7,} "
              f"{r['accuracy_confident']:>7.1%} "
              f"{bss_flag}{bss:>+8.4f} "
              f"{crps:>9.5f} "
              f"{r.get('avg_matches', 0):>6.1f}"
              f"{marker}")

    if ranked:
        best = ranked[0]
        print(f"\n  WINNER: {best['experiment_name']}")
        print(f"  BSS={best.get('brier_skill_score', 0):+.4f} | "
              f"Acc={best['accuracy_confident']:.1%} | "
              f"Trades={best['confident_trades']:,} | "
              f"AvgK={best.get('avg_matches',0):.1f}")
        print(f"\n  To apply to strategy.py, set:")
        print(f"    MAX_DISTANCE = {best.get('max_distance')}")
        print(f"    TOP_K = {best.get('top_k')}")
        print(f"    SAME_SECTOR_ONLY = {best.get('same_sector_only')}")
        print(f"    DISTANCE_WEIGHTING = \"{best.get('distance_weighting')}\"")
        print(f"    CONFIDENCE_THRESHOLD = {best.get('confidence_threshold')}")
        print(f"    AGREEMENT_SPREAD = {best.get('agreement_spread')}")
        # Critical: also print the fields that distinguish the winning config.
        # Without these, copying the block above would not reproduce the winner.
        # (Gemini finding, sweep 2 — 2026-03-16)
        dm = best.get("distance_metric", "cosine")
        fs = best.get("feature_set_name", "full")
        nf = best.get("n_features", 16)
        if dm != "cosine":
            print(f"    # Also set: metric_override=\"{dm}\" in _run_matching_loop call")
        if fs != "full":
            print(f"    # Also set: feature_cols_override=RETURN_COLS  "
                  f"(feature_set={fs}, n_features={nf})")
        print(f"    distance_metric = \"{dm}\"  # {fs}, {nf} features")

    print(f"{'='*65}\n")
    return ranked


def run_live_signals(tickers=None, top_n=10, verbose=1):
    """
    PRODUCTION MODE — End-of-day signal generation across the full universe.

    This is the entry point for live operation. It:
      1. Fetches the last 120 days of OHLCV data for all 52 tickers via yfinance
      2. Computes the 8-feature return fingerprint for today's close
      3. Runs Euclidean analogue search against the full training database
      4. Applies regime conditioning (bull/bear via SPY ret_90d)
      5. Applies the fitted Platt calibrator to raw probabilities
      6. Ranks all tickers by calibrated P(up) and signal strength
      7. Returns the top_n BUY and SELL candidates with full analogue detail

    No validation or historical evaluation is done here — this is purely
    forward-looking: "given today's fingerprint, what does history suggest?"

    Args:
        tickers:  list of tickers to scan (default: all 52 from SECTOR_MAP)
        top_n:    number of top BUY and SELL candidates to return (default: 10)
        verbose:  1 = progress, 0 = silent

    Returns:
        dict with keys:
          "date":     today's date string
          "signals":  list of dicts ranked by calibrated_prob, each containing:
                        ticker, signal, calibrated_prob, raw_prob,
                        n_matches, avg_distance, top_analogues (list of 5),
                        sector, regime
          "summary":  printable leaderboard string
          "regime":   current SPY macro regime ("bull" / "bear" / "unknown")

    Usage:
        results = run_live_signals()
        for s in results["signals"]:
            print(s["ticker"], s["signal"], f"{s['calibrated_prob']:.1%}")
    """
    import yfinance as yf
    import time

    if tickers is None:
        tickers = list(SECTOR_MAP.keys())

    print(f"\n{'='*65}")
    print(f"  SYSTEM A — LIVE SIGNAL SCAN")
    print(f"  Universe: {len(tickers)} tickers | Horizon: {PROJECTION_HORIZON}")
    print(f"  Distance: Euclidean | Max dist: {MAX_DISTANCE} | Top K: {TOP_K}")
    print(f"{'='*65}\n")

    # ── Step 1: Load training database and fitted calibrator ──────────────
    if verbose:
        print("  [1/5] Loading analogue database and calibrator...")

    train_db = pd.read_parquet(DATA_DIR / "train_db.parquet")

    from prepare import FEATURE_COLS
    RETURN_ONLY = [c for c in FEATURE_COLS if c.startswith("ret_")]

    # Refit scaler on training data (return-only features)
    scaler = StandardScaler()
    scaler.fit(train_db[RETURN_ONLY].values)

    # Fit Platt calibrator on training set probabilities
    # This mirrors run_platt_walkforward v2 — calibrator sees all 14 years
    if verbose:
        print("  [2/5] Fitting Platt calibrator on training set...")
    train_probs, _, _, _, _, _ = _run_matching_loop(
        train_db, train_db, scaler, FEATURE_COLS, verbose=0,
        max_distance=MAX_DISTANCE,
        metric_override="euclidean",
        feature_cols_override=RETURN_ONLY,
        top_k=TOP_K, distance_weighting=DISTANCE_WEIGHTING,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        agreement_spread=AGREEMENT_SPREAD,
        exclude_same_ticker=True,
        regime_filter=True,
    )
    train_y_bin = train_db[PROJECTION_HORIZON].values
    calibrator = fit_platt_scaling(train_probs, train_y_bin)
    if verbose:
        print(f"     Calibrator fitted on {len(train_probs):,} training samples")

    # ── Step 2: Fetch today's end-of-day data for all tickers ─────────────
    if verbose:
        print(f"  [3/5] Fetching end-of-day data for {len(tickers)} tickers...")

    today_str = datetime.now().strftime("%Y-%m-%d")
    lookback_days = 130  # need 90 days for ret_90d + buffer for weekends/holidays

    all_today_data = {}
    failed_tickers = []
    for i, ticker in enumerate(tickers):
        if verbose and i % 10 == 0:
            print(f"     Downloading {i+1}/{len(tickers)}...", end="\r")
        try:
            df = yf.download(ticker, period=f"{lookback_days}d",
                             progress=False, auto_adjust=True)
            if len(df) >= 95:  # need at least 95 trading days for ret_90d
                df = df.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df["Ticker"] = ticker
                all_today_data[ticker] = df
        except Exception as e:
            failed_tickers.append((ticker, str(e)))

    if verbose:
        print(f"     Downloaded {len(all_today_data)}/{len(tickers)} tickers "
              f"({len(failed_tickers)} failed)    ")

    # ── Step 3: Compute today's return fingerprint for each ticker ─────────
    if verbose:
        print(f"  [4/5] Computing return fingerprints...")

    from prepare import compute_return_vector

    query_rows = []
    for ticker, df in all_today_data.items():
        df = compute_return_vector(df, ticker)
        df = df.dropna(subset=RETURN_ONLY)
        if len(df) == 0:
            continue
        latest = df.iloc[-1]  # most recent trading day = today's fingerprint
        row = {"Ticker": ticker, "Date": latest["Date"]}
        row.update({c: latest[c] for c in RETURN_ONLY})
        query_rows.append(row)

    if not query_rows:
        print("  ERROR: No valid fingerprints computed. Check yfinance connection.")
        return {"date": today_str, "signals": [], "summary": "No data", "regime": "unknown"}

    query_db = pd.DataFrame(query_rows).reset_index(drop=True)

    # Determine current market regime from SPY
    spy_row = query_db[query_db["Ticker"] == "SPY"]
    if len(spy_row) > 0:
        spy_ret90 = float(spy_row.iloc[-1]["ret_90d"])
        current_regime = "bull" if spy_ret90 > 0 else "bear"
    else:
        current_regime = "unknown"
    if verbose:
        print(f"     Current regime: {current_regime.upper()} "
              f"(SPY ret_90d = {spy_ret90:+.1%})" if current_regime != "unknown"
              else "     Current regime: unknown (SPY data unavailable)")

    # ── Step 4: Run analogue matching on today's fingerprints ─────────────
    if verbose:
        print(f"  [5/5] Matching {len(query_db)} tickers against {len(train_db):,} historical rows...")

    raw_probs, signals, reasons, n_matches, mean_rets, ensembles = _run_matching_loop(
        train_db, query_db, scaler, FEATURE_COLS, verbose=0,
        max_distance=MAX_DISTANCE,
        metric_override="euclidean",
        feature_cols_override=RETURN_ONLY,
        top_k=TOP_K, distance_weighting=DISTANCE_WEIGHTING,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        agreement_spread=AGREEMENT_SPREAD,
        regime_filter=True,
    )

    # ── Step 5: Apply calibration and build signal records ─────────────────
    cal_probs = calibrate_probabilities(calibrator, raw_probs)

    # Re-generate signals with calibrated probabilities
    final_signals = []
    for i in range(len(query_db)):
        ticker  = query_db.iloc[i]["Ticker"]
        raw_p   = float(raw_probs[i])
        cal_p   = float(cal_probs[i])
        nm      = int(n_matches[i])
        mr      = float(mean_rets[i])
        sector  = SECTOR_MAP.get(ticker, "Unknown")

        # Generate signal from calibrated probability
        proj_mock = {
            "probability_up": cal_p,
            "agreement": abs(cal_p - 0.5) * 2,
            "n_matches": nm,
            "mean_return": mr,
            "median_return": mr,
            "ensemble_returns": ensembles[i],
        }
        signal, reason = generate_signal(
            proj_mock,
            threshold=CONFIDENCE_THRESHOLD,
            min_agreement=AGREEMENT_SPREAD,
            min_matches=MIN_MATCHES,
        )

        # Find top-5 nearest analogues for this ticker
        top_analogues = []
        if nm > 0:
            # Re-run a single query to get actual neighbour rows with dates
            X_q = scaler.transform(
                query_db.iloc[[i]][RETURN_ONLY].values
            )
            nn_idx = NearestNeighbors(
                n_neighbors=min(5, len(train_db)),
                metric="euclidean", algorithm="ball_tree", n_jobs=1
            )
            nn_idx.fit(scaler.transform(train_db[RETURN_ONLY].values))
            dists, idxs = nn_idx.kneighbors(X_q)
            for dist, idx in zip(dists[0], idxs[0]):
                if dist <= MAX_DISTANCE:
                    analogue_row = train_db.iloc[idx]
                    top_analogues.append({
                        "date":    str(analogue_row["Date"])[:10],
                        "ticker":  analogue_row["Ticker"],
                        "distance": round(float(dist), 4),
                        "fwd_7d":  round(float(analogue_row.get("fwd_7d", 0)) * 100, 2),
                        "went_up": bool(analogue_row.get("fwd_7d_up", 0)),
                    })

        final_signals.append({
            "ticker":          ticker,
            "signal":          signal,
            "calibrated_prob": round(cal_p, 4),
            "raw_prob":        round(raw_p, 4),
            "n_matches":       nm,
            "mean_7d_return":  round(mr * 100, 2),
            "sector":          sector,
            "regime":          current_regime,
            "reason":          reason,
            "top_analogues":   top_analogues,
        })

    # Rank: BUY by highest calibrated_prob, SELL by lowest, HOLD filtered out
    buys  = sorted([s for s in final_signals if s["signal"] == "BUY"],
                   key=lambda x: x["calibrated_prob"], reverse=True)
    sells = sorted([s for s in final_signals if s["signal"] == "SELL"],
                   key=lambda x: x["calibrated_prob"])
    holds = [s for s in final_signals if s["signal"] == "HOLD"]

    top_buys  = buys[:top_n]
    top_sells = sells[:top_n]

    # ── Print leaderboard ──────────────────────────────────────────────────
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  LIVE SIGNAL LEADERBOARD — {today_str}")
    lines.append(f"  Regime: {current_regime.upper()} | "
                 f"Universe: {len(final_signals)} tickers | "
                 f"BUY: {len(buys)} | SELL: {len(sells)} | HOLD: {len(holds)}")
    lines.append(f"{'='*65}")

    if top_buys:
        lines.append(f"\n  TOP BUY SIGNALS")
        lines.append(f"  {'Ticker':<8} {'P(up)':>7} {'Raw P':>7} "
                     f"{'Matches':>8} {'AvgRet':>8} {'Sector':<12} Top Analogues")
        lines.append(f"  {'─'*75}")
        for s in top_buys:
            analogue_str = ", ".join(
                f"{a['ticker']}@{a['date'][:7]}" for a in s["top_analogues"][:3]
            )
            lines.append(f"  {s['ticker']:<8} {s['calibrated_prob']:>7.1%} "
                         f"{s['raw_prob']:>7.1%} {s['n_matches']:>8} "
                         f"{s['mean_7d_return']:>+7.1f}% {s['sector']:<12} {analogue_str}")

    if top_sells:
        lines.append(f"\n  TOP SELL SIGNALS")
        lines.append(f"  {'Ticker':<8} {'P(up)':>7} {'Raw P':>7} "
                     f"{'Matches':>8} {'AvgRet':>8} {'Sector':<12} Top Analogues")
        lines.append(f"  {'─'*75}")
        for s in top_sells:
            analogue_str = ", ".join(
                f"{a['ticker']}@{a['date'][:7]}" for a in s["top_analogues"][:3]
            )
            lines.append(f"  {s['ticker']:<8} {s['calibrated_prob']:>7.1%} "
                         f"{s['raw_prob']:>7.1%} {s['n_matches']:>8} "
                         f"{s['mean_7d_return']:>+7.1f}% {s['sector']:<12} {analogue_str}")

    lines.append(f"\n  Regime: {current_regime.upper()} — "
                 + ("Bear-market analogues only. " if current_regime == "bear" else
                    "Bull-market analogues only. ")
                 + f"HOLD: {len(holds)} tickers filtered.")
    lines.append(f"{'='*65}\n")

    summary = "\n".join(lines)
    print(summary)

    return {
        "date":    today_str,
        "signals": top_buys + top_sells,
        "all_signals": final_signals,
        "summary": summary,
        "regime":  current_regime,
        "n_buys":  len(buys),
        "n_sells": len(sells),
        "n_holds": len(holds),
    }


# ============================================================
# PLATT SCALING CALIBRATION
# ============================================================

def fit_platt_scaling(probabilities, y_true):
    """
    Fit a logistic regression to map raw P(up) outputs to calibrated
    probabilities. This is the standard post-hoc calibration fix when
    a model has directional skill but miscalibrated probability outputs.

    The raw P(up) from analogue matching is the single input feature.
    The fitted model learns a monotonic mapping to calibrated probabilities.

    Args:
        probabilities: array of raw P(up) from _run_matching_loop
        y_true:        array of 0/1 ground truth outcomes

    Returns:
        fitted LogisticRegression calibrator
    """
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    cal.fit(probabilities.reshape(-1, 1), y_true)
    return cal


def fit_isotonic_scaling(probabilities, y_true):
    """
    Non-parametric calibration via isotonic regression.

    Isotonic regression makes no assumption about the shape of the
    P(raw) → P(calibrated) mapping — it fits a monotone step function.
    This generalises better than Platt (logistic) when the miscalibration
    is non-linear or regime-dependent.

    Source: Gemini deep research — "Isotonic regression outperforms Platt
    scaling when the underlying calibration curve is non-parametric."
    Niculescu-Mizil & Caruana (2005): isotonic beats Platt on datasets where
    the probability outputs are not sigmoid-shaped.

    Args:
        probabilities: 1-D array of raw P(up) from _run_matching_loop
        y_true:        1-D array of 0/1 ground truth outcomes

    Returns:
        fitted IsotonicRegression calibrator
    """
    from sklearn.isotonic import IsotonicRegression
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(probabilities, y_true)
    return cal


def calibrate_probabilities(calibrator, probabilities):
    """
    Apply a fitted calibrator to raw probabilities. Abstracts Platt vs isotonic
    so callers don't need to know which was used.

    Platt (LogisticRegression): needs reshape(-1,1) + predict_proba[:,1]
    Isotonic (IsotonicRegression): takes 1-D array + predict()
    """
    if hasattr(calibrator, "predict_proba"):
        # Platt scaling (LogisticRegression)
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    else:
        # Isotonic regression (IsotonicRegression)
        return calibrator.predict(probabilities)


def evaluate_with_calibration(probabilities, y_true_binary, y_true_returns,
                               all_signals, all_ensembles, horizon_label,
                               cal_frac=0.5, experiment_name=""):
    """
    Fit Platt scaling on the first cal_frac of validation, evaluate on the rest.

    Temporal split: first half = earlier 2024 (calibration), second = later 2024
    (evaluation). This avoids using the same data to fit and evaluate the calibrator.

    OVERNIGHT RESULTS (cal_frac=0.50):
      platt_euc_r_d1.0115: pre=-0.03807 → post=-0.00192  delta=+0.036  ← best ever
      platt_euc_r_d1.1019: pre=-0.03429 → post=-0.00228  delta=+0.032
      platt_euc_r_d1.2457: pre=-0.03351 → post=-0.00277  delta=+0.031
      platt_cos_uniform:   pre=-0.04008 → post=-0.00491  delta=+0.035
    Gap to BSS=0: 0.002. Hypothesis: higher cal_frac closes the gap.

    Args:
        cal_frac: fraction of val set used for calibration (try 0.50–0.80)

    Returns:
        dict with pre_bss, post_bss, bss_delta, pre_acc, post_acc, etc.
    """
    from prepare import save_results

    n     = len(probabilities)
    split = int(n * cal_frac)

    # Fit calibrator on first cal_frac of validation (temporal order)
    cal_probs = probabilities[:split]
    cal_y     = y_true_binary[:split]
    calibrator = fit_platt_scaling(cal_probs, cal_y)

    # Evaluate on second half only
    eval_probs  = probabilities[split:]
    eval_y_bin  = y_true_binary[split:]
    eval_y_ret  = y_true_returns[split:]
    eval_sigs   = all_signals[split:]
    eval_ens    = all_ensembles[split:]

    # Apply calibrated probabilities to eval set (Platt or isotonic)
    cal_probs_eval = calibrate_probabilities(calibrator, eval_probs)

    # Re-generate signals with calibrated probabilities
    recal_signals = []
    for p in cal_probs_eval:
        proj_mock = {
            "probability_up": float(p),
            "agreement": abs(p - 0.5) * 2,
            "n_matches": MIN_MATCHES + 1,
            "mean_return": 0.0,
            "median_return": 0.0,
            "ensemble_returns": np.array([]),
        }
        sig, _ = generate_signal(proj_mock,
                                  threshold=CONFIDENCE_THRESHOLD,
                                  min_agreement=AGREEMENT_SPREAD,
                                  min_matches=1)
        recal_signals.append(sig)
    recal_signals = np.array(recal_signals)

    # Score uncalibrated on eval set
    pre_metrics = evaluate_probabilistic(
        eval_y_bin, eval_y_ret, eval_probs,
        eval_ens, signals=eval_sigs, horizon_label=horizon_label
    )

    # Score calibrated on eval set
    post_metrics = evaluate_probabilistic(
        eval_y_bin, eval_y_ret, cal_probs_eval,
        eval_ens, signals=recal_signals, horizon_label=horizon_label
    )

    pre_bss  = pre_metrics.get("brier_skill_score")
    post_bss = post_metrics.get("brier_skill_score")
    bss_delta = (post_bss or 0) - (pre_bss or 0)

    result = {
        "experiment_name": experiment_name,
        "method": "platt_calibration",
        "cal_frac": cal_frac,
        "cal_samples": split,
        "eval_samples": n - split,
        "pre_bss":    pre_bss,
        "pre_acc":    pre_metrics.get("accuracy_confident"),
        "pre_trades": pre_metrics.get("confident_trades"),
        "pre_crps":   pre_metrics.get("crps"),
        "post_bss":    post_bss,
        "post_acc":    post_metrics.get("accuracy_confident"),
        "post_trades": post_metrics.get("confident_trades"),
        "post_crps":   post_metrics.get("crps"),
        "bss_delta":   bss_delta,
    }

    save_results(result, experiment_name,
                 filepath=RESULTS_DIR / ANALOGUE_RESULTS_FILE)
    return result


def run_platt_sweep(cal_fracs=None, verbose=1):
    """
    Targeted Platt calibration fraction sweep.

    Tests multiple cal_frac values (0.50–0.80) on the best-performing
    analogue configs. The gap to BSS=0 is 0.002 as of 2026-03-16 —
    this sweep is designed to close it.

    EVIDENCE:
      - Best raw BSS:   -0.038 (euc_r_d1.2457, sweep 3)
      - Best Platt BSS: -0.00192 (platt_euc_r_d1.0115, cal_frac=0.50)
      - Platt lifts BSS by +0.031 to +0.036 across all configs tested
      - More calibration data (higher cal_frac) should narrow the gap further

    Results → results_analogue.tsv. Prints leaderboard ranked by post_bss.

    Args:
        cal_fracs: list of fractions to test. Default: [0.50, 0.65, 0.70, 0.75, 0.80]
    """
    import time

    if cal_fracs is None:
        cal_fracs = [0.50, 0.65, 0.70, 0.75, 0.80]

    # Load data once
    print(f"\n{'='*65}")
    print(f"  PLATT CALIBRATION FRACTION SWEEP")
    print(f"  cal_frac values: {cal_fracs}")
    print(f"  Goal: post_bss > 0  (gap = 0.002 as of 2026-03-16)")
    print(f"{'='*65}\n")

    train_db = pd.read_parquet(DATA_DIR / "train_db.parquet")
    val_db   = pd.read_parquet(DATA_DIR / "val_db.parquet")
    scaler   = joblib.load(MODEL_DIR / "analogue_scaler.pkl")
    from prepare import FEATURE_COLS, save_results

    RETURN_ONLY = [c for c in FEATURE_COLS if c.startswith("ret_")]

    # Configs — top 2 from Platt sweep 1 (both peaked at cf=0.75)
    # d1.1019: best post_bss=-0.00048 at cf=0.75
    # d1.2457: best post_bss=-0.00078 at cf=0.75
    configs = [
        {"name_base": "platt_euc_r_d1.1019", "MAX_DISTANCE": 1.1019,
         "metric": "euclidean", "feat": RETURN_ONLY},
        {"name_base": "platt_euc_r_d1.2457", "MAX_DISTANCE": 1.2457,
         "metric": "euclidean", "feat": RETURN_ONLY},
        {"name_base": "platt_euc_r_d1.0115", "MAX_DISTANCE": 1.0115,
         "metric": "euclidean", "feat": RETURN_ONLY},
    ]

    all_results = []
    session_start = time.time()

    # Pre-compute probabilities for each config once (reused across all cal_fracs)
    config_probs = {}
    for cfg in configs:
        print(f"  Computing analogues: {cfg['name_base']}...")
        probs, sigs, _, n_matches, mean_rets, ensembles = _run_matching_loop(
            train_db, val_db, scaler, FEATURE_COLS, verbose=0,
            max_distance=cfg["MAX_DISTANCE"],
            metric_override=cfg["metric"],
            feature_cols_override=cfg["feat"],
            top_k=50, distance_weighting="uniform",
            confidence_threshold=CONFIDENCE_THRESHOLD,
            agreement_spread=AGREEMENT_SPREAD,
        )
        config_probs[cfg["name_base"]] = (probs, sigs, ensembles,
                                           float(np.mean(n_matches)))
        print(f"    AvgK={np.mean(n_matches):.1f} | raw P(up) range: "
              f"{probs.min():.3f}–{probs.max():.3f}")

    y_bin = val_db[PROJECTION_HORIZON].values
    y_ret = val_db[PROJECTION_HORIZON.replace("_up", "")].values

    print(f"\n  Running Platt sweep across {len(cal_fracs)} cal_frac values...\n")

    # Now test all cal_fracs for each config
    for cfg in configs:
        name_base = cfg["name_base"]
        probs, sigs, ensembles, avgk = config_probs[name_base]

        for cal_frac in cal_fracs:
            exp_name = f"{name_base}_cf{int(cal_frac*100)}"

            r = evaluate_with_calibration(
                probs, y_bin, y_ret, sigs, ensembles,
                horizon_label=PROJECTION_HORIZON,
                experiment_name=exp_name,
                cal_frac=cal_frac,
            )
            all_results.append(r)

            pre  = r.get("pre_bss") or 0
            post = r.get("post_bss") or 0
            delta = r.get("bss_delta") or 0
            flag  = "✓ POSITIVE BSS" if post > 0 else ("↑ close" if post > -0.01 else "✗")
            if verbose:
                print(f"  {flag} {exp_name:<40} "
                      f"pre={pre:+.5f} → post={post:+.5f}  Δ={delta:+.5f}  "
                      f"AvgK={avgk:.1f}")

    # Leaderboard
    total_time = time.time() - session_start
    ranked = sorted(all_results,
                    key=lambda r: r.get("post_bss") or -99, reverse=True)

    print(f"\n{'='*65}")
    print(f"  PLATT SWEEP LEADERBOARD — {len(all_results)} configs in {total_time/60:.1f} min")
    print(f"  Ranked by post-calibration BSS")
    print(f"{'='*65}\n")
    print(f"  {'Experiment':<42} {'cal_frac':>8} {'pre_BSS':>9} {'post_BSS':>9} {'delta':>8}")
    print(f"  {'─'*75}")

    for i, r in enumerate(ranked):
        post = r.get("post_bss") or 0
        pre  = r.get("pre_bss") or 0
        flag = "✓" if post > 0 else " "
        marker = " ← BEST" if i == 0 else ""
        print(f"  {flag}{r['experiment_name']:<41} "
              f"{r.get('cal_frac', 0):>8.2f} "
              f"{pre:>+9.5f} "
              f"{post:>+9.5f} "
              f"{r.get('bss_delta', 0):>+8.5f}"
              f"{marker}")

    if ranked and (ranked[0].get("post_bss") or 0) > 0:
        best = ranked[0]
        print(f"\n  ✓ POSITIVE BSS ACHIEVED: {best['experiment_name']}")
        print(f"  post_bss={best['post_bss']:+.5f} at cal_frac={best['cal_frac']}")
        print(f"\n  NEXT STEP: Run walk-forward with this config to confirm regime stability.")
    else:
        best = ranked[0] if ranked else {}
        post = best.get("post_bss") or 0
        print(f"\n  Best post_bss: {post:+.5f} ({best.get('experiment_name','?')})")
        print(f"  Still negative. Gap to 0: {abs(post):.5f}")
        if abs(post) < 0.01:
            print(f"  Very close — try a wider cal_frac range or isotonic regression.")

    print(f"{'='*65}\n")
    return ranked


def run_platt_walkforward(experiment_name="platt_walkforward",
                          max_distance=1.1019):
    """
    Walk-forward validation using Platt calibration fitted on TRAINING data.

    Previous version (v1) fitted Platt on the first 76% of each fold's val set.
    That caused two failures:
      1. 2022 Bear: calibrator fitted on Jan-Sep 2022 (all-bear) mapped all
         probabilities toward 0.47 → zero trades generated
      2. 1/6 positive BSS — calibration overfit to each fold's own regime

    This version (v2) fits Platt on the training set probabilities:
      - Training set spans all regimes (2010–fold train_end)
      - Calibrator learns a stable, regime-agnostic P(raw) → P(calibrated) mapping
      - Full validation set is evaluated (no holdout split needed)
      - Strictly no leakage: calibrator uses only train_db rows

    Config: Euclidean distance | return-only (8 features) | dist={max_distance}
    """
    print(f"\n{'='*65}")
    print(f"  PLATT WALK-FORWARD v2: {experiment_name}")
    print(f"  Config: Euclidean + return-only | dist={max_distance}")
    print(f"  Calibration: fitted on TRAINING SET (regime-agnostic)")
    print(f"  {len(WALKFORWARD_FOLDS)} folds × (match train + calibrate on train + eval full val)")
    print(f"{'='*65}")

    full_db_path = DATA_DIR / "full_analogue_db.parquet"
    if not full_db_path.exists():
        raise FileNotFoundError("full_analogue_db.parquet not found. Run prepare.py first.")

    full_db = pd.read_parquet(full_db_path)
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    from prepare import FEATURE_COLS, save_results
    RETURN_ONLY = [c for c in FEATURE_COLS if c.startswith("ret_")]

    horizon_return_col = PROJECTION_HORIZON.replace("_up", "")
    fold_results = []

    for fold_idx, fold in enumerate(WALKFORWARD_FOLDS):
        label     = fold["label"]
        train_end = fold["train_end"]
        val_start = fold["val_start"]
        val_end   = fold["val_end"]

        print(f"\n  ── Fold {fold_idx+1}/{len(WALKFORWARD_FOLDS)}: {label} ──")
        print(f"     Train: start → {train_end} | Val: {val_start} → {val_end}")

        train_db = full_db[full_db["Date"] <= train_end].copy().reset_index(drop=True)
        val_db   = full_db[(full_db["Date"] >= val_start) &
                           (full_db["Date"] <= val_end)].copy().reset_index(drop=True)

        if len(train_db) < 1000 or len(val_db) < 100:
            print(f"     SKIP — insufficient data")
            continue

        print(f"     Train: {len(train_db):,} rows | Val: {len(val_db):,} rows")

        # Refit scaler on this fold's training data only (no leakage)
        scaler = StandardScaler()
        scaler.fit(train_db[RETURN_ONLY].values)

        # Step 1: Run matching on TRAINING set to get train probabilities
        # These are used to fit the Platt calibrator — all 14 years of regimes
        print(f"     Fitting Platt calibrator on training set...")
        train_probs, _, _, _, _, _ = _run_matching_loop(
            train_db, train_db, scaler, FEATURE_COLS, verbose=0,
            max_distance=max_distance,
            metric_override="euclidean",
            feature_cols_override=RETURN_ONLY,
            top_k=50, distance_weighting="uniform",
            confidence_threshold=CONFIDENCE_THRESHOLD,
            agreement_spread=AGREEMENT_SPREAD,
            exclude_same_ticker=True,  # must exclude self-matches on train set
        )
        train_y_bin = train_db[PROJECTION_HORIZON].values

        # Fit Platt calibrator on full training set probabilities
        calibrator = fit_platt_scaling(train_probs, train_y_bin)
        print(f"     Calibrator fitted on {len(train_probs):,} training samples")

        # Step 2: Run matching on VALIDATION set
        val_probs, all_signals, _, all_n_matches, all_mean_returns, all_ensembles = \
            _run_matching_loop(
                train_db, val_db, scaler, FEATURE_COLS, verbose=0,
                max_distance=max_distance,
                metric_override="euclidean",
                feature_cols_override=RETURN_ONLY,
                top_k=50, distance_weighting="uniform",
                confidence_threshold=CONFIDENCE_THRESHOLD,
                agreement_spread=AGREEMENT_SPREAD,
            )

        y_bin = val_db[PROJECTION_HORIZON].values
        y_ret = val_db[horizon_return_col].values

        # Step 3: Apply training-fitted calibrator to validation probabilities
        cal_probs = calibrator.predict_proba(val_probs.reshape(-1, 1))[:, 1]

        # Re-generate signals with calibrated probabilities (full val set)
        recal_signals = []
        for p in cal_probs:
            proj_mock = {"probability_up": float(p), "agreement": abs(p - 0.5) * 2,
                         "n_matches": MIN_MATCHES + 1, "mean_return": 0.0,
                         "median_return": 0.0, "ensemble_returns": np.array([])}
            sig, _ = generate_signal(proj_mock, threshold=CONFIDENCE_THRESHOLD,
                                      min_agreement=AGREEMENT_SPREAD, min_matches=1)
            recal_signals.append(sig)
        recal_signals = np.array(recal_signals)

        # Evaluate on FULL validation set
        metrics = evaluate_probabilistic(
            y_bin, y_ret, cal_probs,
            all_ensembles, signals=recal_signals,
            horizon_label=PROJECTION_HORIZON
        )

        bss  = metrics.get("brier_skill_score") or 0
        crps = metrics.get("crps")
        crps_str = f"{crps:.5f}" if crps is not None else "N/A"
        bss_flag = "✓" if bss > 0 else "✗"
        print(f"     {bss_flag} Trades={metrics['confident_trades']:,} | "
              f"Acc={metrics['accuracy_confident']:.1%} | "
              f"BSS={bss:+.5f} | CRPS={crps_str} | AvgK={np.mean(all_n_matches):.1f}")

        # Log to TSV
        fold_name = f"{experiment_name}_fold_{label.replace(' ', '_').replace('(','').replace(')','')}"
        row = {k: v for k, v in metrics.items() if k != "calibration_buckets"}
        row.update({
            "experiment_name": fold_name,
            "method": "platt_walkforward_v2",
            "fold_label": label,
            "train_end": train_end,
            "val_year": val_start[:4],
            "top_k": 50,
            "max_distance": max_distance,
            "distance_weighting": "uniform",
            "projection_horizon": PROJECTION_HORIZON,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "agreement_spread": AGREEMENT_SPREAD,
            "min_matches": MIN_MATCHES,
            "same_sector_only": False,
            "exclude_same_ticker": True,
            "cal_source": "training_set",
            "distance_metric": "euclidean",
            "feature_set_name": "returns_only",
            "n_features": len(RETURN_ONLY),
            "avg_matches": float(np.mean(all_n_matches)),
            "avg_projected_return": float(np.mean(all_mean_returns)),
            "buy_signals": int((all_signals == "BUY").sum()),
            "sell_signals": int((all_signals == "SELL").sum()),
            "hold_signals": int((all_signals == "HOLD").sum()),
        })
        save_results(row, fold_name, filepath=RESULTS_DIR / ANALOGUE_RESULTS_FILE)
        fold_results.append({"label": label, **metrics,
                              "avg_matches": float(np.mean(all_n_matches))})

    # ── Summary ──────────────────────────────────────────────
    acc_list, bss_list, crps_list = [], [], []
    print(f"\n{'='*65}")
    print(f"  PLATT WALK-FORWARD SUMMARY — {experiment_name}")
    print(f"{'='*65}")
    print(f"  {'Fold':<22} {'Trades':>7} {'Acc(c)':>7} {'BSS':>9} {'CRPS':>10} {'AvgK':>6}")
    print(f"  {'~'*57}")

    for r in fold_results:
        bss_val = r.get("brier_skill_score")
        bss_str = f"{bss_val:>+9.5f}" if bss_val is not None else "      N/A"
        crps_str = f"{r['crps']:.5f}" if r.get("crps") is not None else "       N/A"
        bss_flag = "✓" if (bss_val or 0) > 0 else " "
        print(f"  {bss_flag}{r['label']:<21} {r['confident_trades']:>7} "
              f"{r['accuracy_confident']:>7.1%} {bss_str} {crps_str:>10} "
              f"{r['avg_matches']:>6.1f}")

        if r["confident_trades"] > 0:
            acc_list.append(r["accuracy_confident"])
        bss_v = r.get("brier_skill_score")
        if bss_v is not None and not np.isnan(bss_v):
            bss_list.append(bss_v)
        crps_v = r.get("crps")
        if crps_v is not None and not np.isnan(crps_v):
            crps_list.append(crps_v)

    if acc_list:
        n_pos = sum(1 for b in bss_list if b > 0)
        bss_mean = np.mean(bss_list) if bss_list else float("nan")
        print(f"  {'~'*57}")
        print(f"  {'MEAN':<22} {'':>7} {np.mean(acc_list):>7.1%} "
              f"{bss_mean:>+9.5f} "
              f"{np.mean(crps_list):>10.5f}" if crps_list else "")
        print(f"  {'STD':<22} {'':>7} {np.std(acc_list):>7.1%} "
              f"{np.std(bss_list):>9.5f}")
        print(f"\n  Positive BSS folds: {n_pos}/{len(bss_list)}")
        std_acc = np.std(acc_list)
        if std_acc < 0.02:
            print(f"  ✓ STABLE — std={std_acc:.3f}. Signal consistent across regimes.")
        elif std_acc < 0.05:
            print(f"  ~ MODERATE — std={std_acc:.3f}. Some regime sensitivity.")
        else:
            print(f"  ✗ UNSTABLE — std={std_acc:.3f}. Results vary too much.")

    print(f"{'='*65}\n")
    return fold_results


def run_regime_walkforward(experiment_name="regime_walkforward",
                            max_distance=1.1019,
                            calibration_method="both",
                            confidence_threshold=None):
    """
    Walk-forward validation with macro-regime conditioning (v4).

    v4 confirms regime labels correct: 2022 Bear = 2756 bull / 10296 bear ✓
    Positive BSS: +0.00103 Platt / +0.00074 Isotonic on fold 2024 (v4 run).
    Zero trades at threshold=0.65 — regime filter compresses P(up) toward 0.5.
    Use confidence_threshold=0.60 or 0.55 to restore coverage. Overnight testing.

    Args:
        calibration_method:   "platt", "isotonic", or "both"
        confidence_threshold: override threshold (default CONFIDENCE_THRESHOLD=0.65)
    """
    print(f"\n{'='*65}")
    print(f"  REGIME WALK-FORWARD v3: {experiment_name}")
    print(f"  Config: Euclidean + return-only | dist={max_distance}")
    print(f"  Regime filter: ON (SPY ret_90d bull/bear conditioning)")
    print(f"  Calibration: {calibration_method}")
    print(f"  {len(WALKFORWARD_FOLDS)} folds × (regime-match + calibrate on train + eval full val)")
    print(f"{'='*65}")

    full_db_path = DATA_DIR / "full_analogue_db.parquet"
    if not full_db_path.exists():
        raise FileNotFoundError("full_analogue_db.parquet not found. Run prepare.py first.")

    full_db = pd.read_parquet(full_db_path)
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    from prepare import FEATURE_COLS, save_results
    RETURN_ONLY = [c for c in FEATURE_COLS if c.startswith("ret_")]

    horizon_return_col = PROJECTION_HORIZON.replace("_up", "")

    _conf_thresh = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
    methods = []
    if calibration_method in ("platt", "both"):
        methods.append(("platt", fit_platt_scaling))
    if calibration_method in ("isotonic", "both"):
        methods.append(("isotonic", fit_isotonic_scaling))

    all_fold_results = {m: [] for m, _ in methods}

    for fold_idx, fold in enumerate(WALKFORWARD_FOLDS):
        label     = fold["label"]
        train_end = fold["train_end"]
        val_start = fold["val_start"]
        val_end   = fold["val_end"]

        print(f"\n  ── Fold {fold_idx+1}/{len(WALKFORWARD_FOLDS)}: {label} ──")
        print(f"     Train: start → {train_end} | Val: {val_start} → {val_end}")

        train_db = full_db[full_db["Date"] <= train_end].copy().reset_index(drop=True)
        val_db   = full_db[(full_db["Date"] >= val_start) &
                           (full_db["Date"] <= val_end)].copy().reset_index(drop=True)

        if len(train_db) < 1000 or len(val_db) < 100:
            print(f"     SKIP — insufficient data")
            continue

        print(f"     Train: {len(train_db):,} rows | Val: {len(val_db):,} rows")

        scaler = StandardScaler()
        scaler.fit(train_db[RETURN_ONLY].values)

        # Step 1: Run matching on TRAINING set (for calibration), with regime filter
        print(f"     Fitting calibrators on training set (regime-conditioned)...")
        train_probs, _, _, _, _, _ = _run_matching_loop(
            train_db, train_db, scaler, FEATURE_COLS, verbose=0,
            max_distance=max_distance,
            metric_override="euclidean",
            feature_cols_override=RETURN_ONLY,
            top_k=50, distance_weighting="uniform",
            confidence_threshold=_conf_thresh,
            agreement_spread=AGREEMENT_SPREAD,
            exclude_same_ticker=True,
            regime_filter=True,
        )
        train_y_bin = train_db[PROJECTION_HORIZON].values

        # Step 2: Run matching on VALIDATION set, with regime filter
        val_probs, all_signals, _, all_n_matches, all_mean_returns, all_ensembles =             _run_matching_loop(
                train_db, val_db, scaler, FEATURE_COLS, verbose=1,
                max_distance=max_distance,
                metric_override="euclidean",
                feature_cols_override=RETURN_ONLY,
                top_k=50, distance_weighting="uniform",
                confidence_threshold=_conf_thresh,
                agreement_spread=AGREEMENT_SPREAD,
                regime_filter=True,
            )

        y_bin = val_db[PROJECTION_HORIZON].values
        y_ret = val_db[horizon_return_col].values

        # Step 3: Evaluate each calibration method
        for method_name, fit_fn in methods:
            calibrator  = fit_fn(train_probs, train_y_bin)
            cal_probs   = calibrate_probabilities(calibrator, val_probs)

            recal_signals = []
            for p in cal_probs:
                proj_mock = {"probability_up": float(p), "agreement": abs(p-0.5)*2,
                             "n_matches": MIN_MATCHES+1, "mean_return": 0.0,
                             "median_return": 0.0, "ensemble_returns": np.array([])}
                sig, _ = generate_signal(proj_mock, threshold=_conf_thresh,
                                          min_agreement=AGREEMENT_SPREAD, min_matches=1)
                recal_signals.append(sig)
            recal_signals = np.array(recal_signals)

            metrics = evaluate_probabilistic(
                y_bin, y_ret, cal_probs,
                all_ensembles, signals=recal_signals,
                horizon_label=PROJECTION_HORIZON
            )

            bss  = metrics.get("brier_skill_score") or 0
            crps = metrics.get("crps")
            crps_str = f"{crps:.5f}" if crps is not None else "N/A"
            bss_flag = "✓" if bss > 0 else "✗"
            print(f"     [{method_name:8s}] {bss_flag} "
                  f"Trades={metrics['confident_trades']:,} | "
                  f"Acc={metrics['accuracy_confident']:.1%} | "
                  f"BSS={bss:+.5f} | CRPS={crps_str} | AvgK={np.mean(all_n_matches):.1f}")

            fold_name = (f"{experiment_name}_{method_name}_fold_"
                        f"{label.replace(' ','_').replace('(','').replace(')','')}")
            row = {k: v for k, v in metrics.items() if k != "calibration_buckets"}
            row.update({
                "experiment_name": fold_name,
                "method": f"regime_walkforward_v4_{method_name}",
                "confidence_threshold_used": _conf_thresh,
                "fold_label": label,
                "train_end": train_end,
                "val_year": val_start[:4],
                "top_k": 50,
                "max_distance": max_distance,
                "distance_weighting": "uniform",
                "projection_horizon": PROJECTION_HORIZON,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "agreement_spread": AGREEMENT_SPREAD,
                "min_matches": MIN_MATCHES,
                "same_sector_only": False,
                "exclude_same_ticker": True,
                "regime_filter": True,
                "calibration": method_name,
                "cal_source": "training_set",
                "distance_metric": "euclidean",
                "feature_set_name": "returns_only",
                "n_features": len(RETURN_ONLY),
                "avg_matches": float(np.mean(all_n_matches)),
                "avg_projected_return": float(np.mean(all_mean_returns)),
                "buy_signals": int((all_signals == "BUY").sum()),
                "sell_signals": int((all_signals == "SELL").sum()),
                "hold_signals": int((all_signals == "HOLD").sum()),
            })
            save_results(row, fold_name, filepath=RESULTS_DIR / ANALOGUE_RESULTS_FILE)
            all_fold_results[method_name].append({"label": label, **metrics,
                                                  "avg_matches": float(np.mean(all_n_matches))})

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  REGIME WALK-FORWARD v3 SUMMARY — {experiment_name}")
    print(f"{'='*65}")

    for method_name, fold_results in all_fold_results.items():
        if not fold_results:
            continue
        bss_list  = [r.get("brier_skill_score") or 0 for r in fold_results]
        acc_list  = [r.get("accuracy_confident") or 0 for r in fold_results
                     if (r.get("confident_trades") or 0) > 0]
        n_pos     = sum(1 for b in bss_list if b > 0)

        print(f"\n  [{method_name.upper()}]")
        print(f"  {'Fold':<22} {'Trades':>7} {'Acc(c)':>7} {'BSS':>9} {'AvgK':>6}")
        print(f"  {'~'*50}")
        for r in fold_results:
            bss_val = r.get("brier_skill_score") or 0
            flag = "✓" if bss_val > 0 else " "
            print(f"  {flag}{r['label']:<21} {r['confident_trades']:>7} "
                  f"{r['accuracy_confident']:>7.1%} {bss_val:>+9.5f} "
                  f"{r['avg_matches']:>6.1f}")
        if bss_list:
            print(f"  {'~'*50}")
            print(f"  {'MEAN':<22} {'':>7} {np.mean(acc_list):>7.1%} "
                  f"{np.mean(bss_list):>+9.5f}")
            print(f"  Positive BSS folds: {n_pos}/{len(bss_list)}")

    print(f"{'='*65}\n")
    return all_fold_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ── MODE 1: Baseline experiment (single val split, 2024) ──
    # run_experiment("analogue_v1_baseline")

    # ── MODE 2: Standard walk-forward (default config) ──
    # run_walkforward("analogue_v2_walkforward")

    # ── MODE 3: Parameter sweep (Euclidean + retonly fine grid) ──
    # Sweep 3 winner: euc_r_d1.2457, BSS=-0.038, Acc=58.6%
    # run_analogue_sweep(budget_minutes=60)

    # ── MODE 4: Platt calibration fraction sweep ──
    # Platt sweep 2 result: POSITIVE BSS at cal_frac=0.76
    #   platt_euc_r_d1.1019_cf76: post_bss=+0.00033  ← winner
    #   platt_euc_r_d1.2457_cf76: post_bss=+0.00003
    # run_platt_sweep(cal_fracs=[0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])

    # ── MODE 0: Live end-of-day signal scan ── (production use)
    # Fetches today's data, scores all 52 tickers, prints ranked BUY/SELL signals.
    # Run this after market close each day once walk-forward is validated.
    # results = run_live_signals()
    # run_live_signals(top_n=5)  # top 5 buys + sells only

    # ── MODE 1: Baseline experiment ──
    # run_experiment("analogue_v1_baseline")

    # ── MODE 2: Standard walk-forward (default config) ──
    # run_walkforward("analogue_v2_walkforward")

    # ── MODE 3: Parameter sweep ──
    # run_analogue_sweep(budget_minutes=60)

    # ── MODE 4: Platt calibration fraction sweep ──
    # run_platt_sweep(cal_fracs=[0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79])

    # ── MODE 5: Platt walk-forward v2 (train-set calibration, no regime filter) ──
    # v2 result: 1/6 positive BSS. Mean BSS=-0.00797.
    # 2022 Bear: Acc=48.7%, BSS=-0.031 — still fails without regime filter.
    # run_platt_walkforward(experiment_name="platt_wf_v2_d1.1019_train_cal", max_distance=1.1019)

    # ── MODE 6: Regime walk-forward ──
    # v4 result: regime labels confirmed correct. BSS=+0.00103 Platt on 2024.
    # BUT 0 trades at thresh=0.65 — regime filter compresses P(up) toward 0.5.
    # Overnight is testing thresh=0.60/0.55 across distance configs.
    # run_regime_walkforward(
    #     experiment_name="regime_wf_v4_d1.1019",
    #     max_distance=1.1019,
    #     calibration_method="both",
    # )

    # ── MODE 6b: Regime walk-forward with relaxed threshold ── ACTIVE ──
    # Gemini (2026-03-16): regime filter compresses P(up) → 0 trades at 0.65.
    # 0.60 is the first test — if trades appear with positive BSS → lock it.
    run_regime_walkforward(
        experiment_name="regime_wf_v4_d1.1019_t60",
        max_distance=1.1019,
        calibration_method="both",
        confidence_threshold=0.60,
    )
