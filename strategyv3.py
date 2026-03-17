"""
strategy.py — System A: Historical Analogue Matching & Forward Projection
THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.

APPROACH:
  Instead of training a neural network to predict up/down from indicators,
  we find historical moments that looked EXACTLY like today and see what
  happened next. This is a fundamentally different bet:

  Old approach: "Given these 32 indicators, will price go up?"
  New approach: "The last 50 times a stock looked like this, what happened?"

THE 4-STEP PIPELINE:
  1. Target Profiling: Compute the current return vector for a target stock
  2. Analogue Search: Find the K nearest historical matches using cosine distance
  3. Forward Projection: Look at what happened after each match (1-30 days out)
  4. Signal Generation: If historical analogues overwhelmingly went up -> BUY

WHAT THE AUTORESEARCH AGENT ITERATES ON:
  - Number of neighbours (K)
  - Distance weighting (uniform vs inverse-distance)
  - Cohort filtering (same sector, similar market cap, all stocks)
  - Confidence threshold and agreement requirement
  - Forward projection horizon (1d, 3d, 7d, 14d, 30d)
  - Feature weighting (which return windows matter most)
  - Minimum match quality (maximum cosine distance to accept)

LEARNINGS FROM PHASE 1 CARRIED FORWARD:
  - Threshold 0.53 was the sweet spot for confidence filtering
  - Uncertainty filtering (agreement among predictions) prevents bad trades
  - 52 tickers across 6 sectors is the right diversity level
  - Results.tsv logging is essential for tracking progress
  - Quick experiments (15-30 min) are more productive than long runs

EVALUATION UPGRADE (added 2026-03-16):
  - CRPS (Continuous Ranked Probability Score) via scoringrules
  - Brier score per horizon
  - Both now reported alongside legacy classification metrics
  - Walk-forward validation across 6 expanding windows (2019-2024)
  - This catches regime-dependent overfitting that a single val split misses
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
TOP_K = 50                     # Number of nearest neighbours (try: 20, 50, 100, 200)
MAX_DISTANCE = 0.5             # Reject matches with cosine distance above this (try: 0.3-0.7)
DISTANCE_WEIGHTING = "inverse" # "uniform" = all matches equal, "inverse" = closer matches count more
MIN_MATCHES = 10               # Minimum valid matches required to generate a signal

# --- Cohort Filtering ---
SAME_SECTOR_ONLY = False       # If True, only match within same sector (try: True/False)
EXCLUDE_SAME_TICKER = True     # Don't match a stock against its own history (prevents autocorrelation)

# --- Forward Projection ---
PROJECTION_HORIZON = "fwd_7d_up"  # Which forward window to predict (try: fwd_1d_up, fwd_3d_up, fwd_7d_up, fwd_14d_up, fwd_30d_up)

# --- Signal Generation ---
CONFIDENCE_THRESHOLD = 0.55    # Minimum agreement among analogues to trade (try: 0.50-0.65)
                               # 0.55 = at least 55% of historical twins went UP
AGREEMENT_SPREAD = 0.10        # Also require the UP% to be this far from 50% (try: 0.05-0.15)

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

def _run_matching_loop(train_db, val_db, scaler, feature_cols, verbose=1):
    """
    Inner matching loop — runs analogue search on val_db using train_db.
    Shared between run_experiment() and run_walkforward() to avoid duplication.

    Returns:
        probabilities, signals, reasons, n_matches, mean_returns, ensemble_list
        where ensemble_list is a list of arrays (one per val query) of raw returns.
    """
    X_train_scaled = scaler.transform(train_db[feature_cols].values)
    X_train_weighted = apply_feature_weights(X_train_scaled, feature_cols)

    nn_index = NearestNeighbors(
        n_neighbors=min(TOP_K * 3, len(train_db)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn_index.fit(X_train_weighted)

    probabilities = []
    all_signals = []
    all_reasons = []
    all_n_matches = []
    all_mean_returns = []
    all_ensembles = []      # Raw analogue return arrays — needed for CRPS

    import time
    start_time = time.time()

    for idx in range(len(val_db)):
        row = val_db.iloc[idx]
        query = row[feature_cols].values.astype(float)
        ticker = row["Ticker"]
        sector = SECTOR_MAP.get(ticker, None)

        query_scaled = scaler.transform(query.reshape(1, -1))
        query_weighted = apply_feature_weights(query_scaled, feature_cols)

        distances, indices = nn_index.kneighbors(query_weighted)
        matches = train_db.iloc[indices[0]].copy()
        matches["distance"] = distances[0]

        # Apply filters
        matches = matches[matches["distance"] <= MAX_DISTANCE]
        if EXCLUDE_SAME_TICKER:
            matches = matches[matches["Ticker"] != ticker]
        if SAME_SECTOR_ONLY and sector:
            matches = matches[matches["Ticker"].map(SECTOR_MAP) == sector]
        matches = matches.head(TOP_K)

        projection = project_forward(matches, horizon=PROJECTION_HORIZON,
                                     weighting=DISTANCE_WEIGHTING)
        signal, reason = generate_signal(projection)

        probabilities.append(projection["probability_up"])
        all_signals.append(signal)
        all_reasons.append(reason)
        all_n_matches.append(projection["n_matches"])
        all_mean_returns.append(projection["mean_return"])
        all_ensembles.append(projection["ensemble_returns"])

        if verbose and (idx + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(val_db) - idx - 1) / rate
            print(f"    {idx+1:,}/{len(val_db):,} "
                  f"({(idx+1)/len(val_db)*100:.0f}%) | "
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

    # Enrich with experiment metadata
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
    })

    # Strip calibration_buckets before saving to TSV (not TSV-friendly)
    metrics_for_tsv = {k: v for k, v in metrics.items() if k != "calibration_buckets"}
    save_results(metrics_for_tsv, experiment_name)

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
        save_results(row, fold_name)
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
        print(f"  {r['label']:<20} {r['confident_trades']:>7} "
              f"{r['accuracy_confident']:>7.1%} "
              f"{r['brier_skill_score']:>+8.4f} "
              f"{crps_str:>10} "
              f"{r['avg_matches']:>7.1f}")
        acc_list.append(r["accuracy_confident"])
        bss_list.append(r["brier_skill_score"])
        if r.get("crps") is not None:
            crps_list.append(r["crps"])

    if acc_list:
        print(f"  {'~'*55}")
        crps_mean_str = f"{np.mean(crps_list):.5f}" if crps_list else "N/A"
        print(f"  {'MEAN':<20} {'':>7} "
              f"{np.mean(acc_list):>7.1%} "
              f"{np.mean(bss_list):>+8.4f} "
              f"{crps_mean_str:>10}")
        print(f"  {'STD':<20} {'':>7} "
              f"{np.std(acc_list):>7.1%} "
              f"{np.std(bss_list):>8.4f}")
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
# MAIN
# ============================================================

if __name__ == "__main__":

    # ── MODE 1: Baseline experiment (single val split, 2024) ──
    # Fastest way to get first numbers. Run this first.
    run_experiment("analogue_v1_baseline")

    # ── MODE 2: Walk-forward validation across all regimes ──
    # Run after baseline to check if the signal is stable or lucky.
    # Uncomment when ready (takes ~6x longer than a single run_experiment).
    # run_walkforward("analogue_v1_walkforward")

    # ── MODE 3: Both in sequence ──
    # run_experiment("analogue_v1_baseline")
    # run_walkforward("analogue_v1_walkforward")
