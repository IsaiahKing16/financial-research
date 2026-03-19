# SUPERSEDED by pattern_engine/ (v2.1) — kept for reference
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
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
import joblib
import json

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
    """
    if len(matches) == 0:
        return {"probability_up": 0.5, "mean_return": 0.0, "median_return": 0.0,
                "agreement": 0.0, "n_matches": 0}

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
    }


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
# EXPERIMENT RUNNER
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

    scaler = joblib.load(MODEL_DIR / "analogue_scaler.pkl")

    from prepare import FEATURE_COLS

    if verbose:
        print(f"  Training analogues: {len(train_db):,}")
        print(f"  Validation queries: {len(val_db):,}")

    # --- Build weighted NN index ---
    if verbose: print("\n[2/4] Building weighted nearest-neighbor index...")
    X_train_scaled = scaler.transform(train_db[FEATURE_COLS].values)
    X_train_weighted = apply_feature_weights(X_train_scaled, FEATURE_COLS)

    nn_index = NearestNeighbors(
        n_neighbors=min(TOP_K * 3, len(train_db)),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn_index.fit(X_train_weighted)

    # --- Run matching on validation set ---
    if verbose: print("\n[3/4] Matching analogues for validation set...")
    probabilities = []
    all_signals = []
    all_reasons = []
    all_n_matches = []
    all_mean_returns = []

    import time
    start_time = time.time()

    for idx in range(len(val_db)):
        row = val_db.iloc[idx]
        query = row[FEATURE_COLS].values.astype(float)
        ticker = row["Ticker"]
        sector = SECTOR_MAP.get(ticker, None)

        # Scale and weight query
        query_scaled = scaler.transform(query.reshape(1, -1))
        query_weighted = apply_feature_weights(query_scaled, FEATURE_COLS)

        # Find matches
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

        # Project forward
        projection = project_forward(matches, horizon=PROJECTION_HORIZON, weighting=DISTANCE_WEIGHTING)
        signal, reason = generate_signal(projection)

        probabilities.append(projection["probability_up"])
        all_signals.append(signal)
        all_reasons.append(reason)
        all_n_matches.append(projection["n_matches"])
        all_mean_returns.append(projection["mean_return"])

        if verbose and (idx + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(val_db) - idx - 1) / rate
            print(f"    {idx+1:,}/{len(val_db):,} "
                  f"({(idx+1)/len(val_db)*100:.0f}%) | "
                  f"{rate:.0f} queries/sec | "
                  f"ETA: {remaining/60:.1f} min")

    elapsed_total = time.time() - start_time
    probabilities = np.array(probabilities)
    all_signals = np.array(all_signals)

    # Signal distribution
    buy_count = (all_signals == "BUY").sum()
    sell_count = (all_signals == "SELL").sum()
    hold_count = (all_signals == "HOLD").sum()

    if verbose:
        print(f"\n  Completed in {elapsed_total/60:.1f} min ({len(val_db)/elapsed_total:.0f} queries/sec)")
        print(f"  Signals: BUY={buy_count:,}, SELL={sell_count:,}, HOLD={hold_count:,}")
        print(f"  Avg matches per query: {np.mean(all_n_matches):.1f}")
        print(f"  Avg projected return: {np.mean(all_mean_returns):.4f}")

    # --- Evaluate ---
    if verbose: print("\n[4/4] Evaluating...")

    from prepare import evaluate_predictions, print_metrics, save_results

    y_true = val_db[PROJECTION_HORIZON].values
    metrics = evaluate_predictions(y_true, probabilities, threshold=CONFIDENCE_THRESHOLD)

    if verbose:
        print_metrics(metrics, label=f"Analogue Matching | {PROJECTION_HORIZON} | K={TOP_K}")

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
        "runtime_sec": round(elapsed_total, 1),
    })

    save_results(metrics, experiment_name)

    # Save config
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / f"{experiment_name}_config.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_experiment("analogue_v1_baseline")
