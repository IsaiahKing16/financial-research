"""
prepare.py — System A: Quant Engine Data Pipeline
This file is NOT modified by the autoresearch agent.
Human edits only for adding new data sources or tickers.

PURPOSE:
  Downloads 15-25 years of stock data, computes multi-timeframe return vectors
  for every trading day, and builds the database that the matching algorithm
  searches against. This replaces indicator-based feature engineering with
  a pure price-action approach: "Find me historical moments that looked
  exactly like today."

PIPELINE:
  1. Download OHLCV data for 52 tickers (yfinance, no API key)
  2. Compute trailing return vectors: 1, 3, 7, 14, 30, 45, 60, 90-day returns
  3. Compute supplementary features (volatility, volume profile, momentum)
  4. Build the historical analogue database (every day = one searchable vector)
  5. Temporal split for validation (no data leakage)

LEARNINGS CARRIED FORWARD:
  - 52 tickers across 6 sectors is the right diversity level
  - Temporal split is sacred — train on past, test on future
  - StandardScaler per-ticker for normalization
  - Results logging to results.tsv for experiment tracking
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import warnings
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Load from sector.py (800 tickers, Russell 1000 scope)
from pattern_engine.sector import TICKERS


# Multi-timeframe return windows (the core of the analogue matching)
RETURN_WINDOWS = [1, 3, 7, 14, 30, 45, 60, 90]

# Forward projection windows (what happened AFTER the match?)
FORWARD_WINDOWS = [1, 3, 7, 14, 30]

# Temporal split
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"

# Matching algorithm
MAX_LOOKBACK_DAYS = 90   # Need 90 days of history to compute all windows
TOP_K_MATCHES = 50       # Number of nearest neighbours to retrieve per query


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(tickers=TICKERS, start="2000-01-01", end=None, force_refresh=False):
    """Download historical stock data via yfinance. No API key needed."""
    DATA_DIR.mkdir(exist_ok=True)

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    all_data = {}

    for ticker in tickers:
        cache_path = DATA_DIR / f"{ticker}.csv"

        if cache_path.exists() and not force_refresh:
            df = pd.read_csv(cache_path, parse_dates=["Date"])
            print(f"  [cached] {ticker}: {len(df)} rows")
        else:
            print(f"  [downloading] {ticker}...", end=" ")
            try:
                df = yf.download(ticker, start=start, end=end, progress=False)
                df = df.reset_index()

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0]
                                  for col in df.columns]

                df.to_csv(cache_path, index=False)
                print(f"{len(df)} rows")
            except Exception as e:
                print(f"FAILED: {e}")
                continue

        all_data[ticker] = df

    return all_data


# ============================================================
# MULTI-TIMEFRAME RETURN VECTORS
# ============================================================

def compute_return_vector(df, ticker=""):
    """
    For every trading day, compute the trailing return over each window.
    This creates the "fingerprint" of that moment in time.

    Example: On 2023-06-15 for AAPL, the vector might be:
      [1d: -0.8%, 3d: +1.2%, 7d: -2.1%, 14d: +0.5%, 30d: +4.3%, ...]

    Two stocks with similar vectors experienced similar price trajectories
    leading up to that point — they are "twins."
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # Trailing returns for each window
    for w in RETURN_WINDOWS:
        df[f"ret_{w}d"] = df["Close"].pct_change(w)

    # Forward returns for each window (what happened NEXT — this is our target)
    for w in FORWARD_WINDOWS:
        df[f"fwd_{w}d"] = df["Close"].shift(-w) / df["Close"] - 1

    # Forward binary targets (did price go UP in next N days?)
    for w in FORWARD_WINDOWS:
        df[f"fwd_{w}d_up"] = (df[f"fwd_{w}d"] > 0).astype(int)

    df["Ticker"] = ticker
    return df


def compute_supplementary_features(df):
    """
    Additional features that enrich the return vector for better matching.
    These capture market regime context beyond pure returns.

    Kept from Phase 1 learnings: RSI, volatility, volume profile, and
    ATR proved useful in the indicator-based model.
    """
    df = df.copy()

    # Volatility regime (realized vol over different windows)
    df["vol_10d"] = df["Close"].pct_change().rolling(10).std()
    df["vol_30d"] = df["Close"].pct_change().rolling(30).std()
    df["vol_ratio"] = df["vol_10d"] / (df["vol_30d"] + 1e-8)  # Short/long vol ratio

    # Volume profile
    vol_mean_20 = df["Volume"].rolling(20).mean()
    df["vol_abnormal"] = (df["Volume"] - vol_mean_20) / (vol_mean_20 + 1)

    # Momentum indicators
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    df["atr_14"] = AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()

    # Price relative to moving averages (mean reversion signal)
    df["price_vs_sma20"] = df["Close"] / SMAIndicator(df["Close"], window=20).sma_indicator() - 1
    df["price_vs_sma50"] = df["Close"] / SMAIndicator(df["Close"], window=50).sma_indicator() - 1

    return df


def compute_vol_normalized_features(df):
    """
    Volatility-normalized return fingerprint (M9 feature set).

    Divides each ret_Xd by its own-window realized daily volatility,
    producing a dimensionless Sharpe-like ratio that is directly
    comparable across tickers with different volatility regimes.

    Formula: ret_Xd_norm = ret_Xd / (rolling_std(daily_returns, window=X) + 1e-8)

    Motivation: raw ret_Xd = +3% means "normal day" for NVDA (high-vol) but
    "exceptional event" for JNJ (low-vol). K-NN matching on raw magnitudes
    produces spurious cross-ticker analogues. The normalized version ensures
    two tickers with the same fingerprint truly experienced analogous momentum
    profiles relative to their own volatility baseline.

    Diagnostic provenance: diagnose_signal.py 2026-03-24 — all conditions
    (platt/none × 585T/52T) showed negative BSS; root cause identified as
    raw magnitude features not normalizing across heterogeneous market caps.
    """
    df = df.copy()
    daily_returns = df["Close"].pct_change()

    for w in RETURN_WINDOWS:
        # Use max(w, 5) as minimum window: rolling(1).std(ddof=1) is always NaN
        # (std undefined for n=1). A 5-day minimum gives a stable vol estimate
        # for the short windows (ret_1d, ret_3d) while keeping per-window
        # normalization for longer windows (ret_7d and above).
        vol_window = max(w, 5)
        rolling_vol = daily_returns.rolling(vol_window, min_periods=2).std()
        raw_norm = df[f"ret_{w}d"] / (rolling_vol + 1e-8)
        df[f"ret_{w}d_norm"] = raw_norm.clip(-10.0, 10.0)

    return df


def compute_candlestick_features(df):
    """
    Continuous proportional candlestick microstructure encoding.

    Translates raw OHLC data into normalised ratio vectors that capture
    intra-bar momentum, rejection patterns, and directional pressure.
    Unlike absolute prices or returns, these ratios are scale-invariant:
    a penny stock and a mega-cap can be compared symmetrically.

    Features (all bounded [0,1] except direction [-1,1]):
      body_to_range    = |Close - Open| / (High - Low)      — body dominance
      upper_wick_ratio = (High - max(O,C)) / (High - Low)   — buying rejection
      lower_wick_ratio = (min(O,C) - Low) / (High - Low)    — selling rejection
      body_upper_ratio = |Close - Open| / (High - max(O,C) + ε)  — body vs upper wick
      body_lower_ratio = |Close - Open| / (min(O,C) - Low + ε)   — body vs lower wick
      direction        = +1.0 if Close > Open, -1.0 if Close <= Open

    Source: Gemini deep research — continuous proportional scaling for KNN
    (Enhancing_Stock_Prediction_with_Candlesticks.docx, 2026-03-16).
    Lin et al. (2021): continuous encoding outperforms 13-class discrete labels
    for distance-based algorithms like KNN.

    NOTE: These features require Open/High/Low columns in the dataframe.
    They are NOT added to FEATURE_COLS by default — our evidence shows
    supplementary features can hurt BSS. Test via feature_cols_override
    in strategy.py after regime conditioning is validated.
    """
    df = df.copy()
    eps = 1e-8  # prevent division by zero

    high  = df["High"].values
    low   = df["Low"].values
    open_ = df["Open"].values
    close = df["Close"].values

    total_range = high - low + eps
    body        = np.abs(close - open_)
    upper_body  = np.maximum(open_, close)
    lower_body  = np.minimum(open_, close)
    upper_wick  = high - upper_body
    lower_wick  = lower_body - low

    df["candle_body_to_range"]    = body / total_range
    df["candle_upper_wick_ratio"] = upper_wick / total_range
    df["candle_lower_wick_ratio"] = lower_wick / total_range
    df["candle_body_upper_ratio"] = body / (upper_wick + eps)
    df["candle_body_lower_ratio"] = body / (lower_wick + eps)
    df["candle_direction"]        = np.where(close > open_, 1.0, -1.0)

    return df


# The full feature vector used for matching
RETURN_COLS = [f"ret_{w}d" for w in RETURN_WINDOWS]
VOL_NORM_COLS = [f"ret_{w}d_norm" for w in RETURN_WINDOWS]  # M9: vol-normalized fingerprint
SUPPLEMENT_COLS = ["vol_10d", "vol_30d", "vol_ratio", "vol_abnormal",
                   "rsi_14", "atr_14", "price_vs_sma20", "price_vs_sma50"]
CANDLE_COLS = ["candle_body_to_range", "candle_upper_wick_ratio",
               "candle_lower_wick_ratio", "candle_body_upper_ratio",
               "candle_body_lower_ratio", "candle_direction"]
FEATURE_COLS = RETURN_COLS + SUPPLEMENT_COLS
# CANDLE_COLS are available but NOT in FEATURE_COLS by default.
# VOL_NORM_COLS stored in parquet for PatternMatcher to use with VOL_NORM_COLS feature set.
# To test: set FEATURE_COLS = VOL_NORM_COLS in scripts/run_walkforward.py.
NUM_FEATURES = len(FEATURE_COLS)

# Forward target columns
FORWARD_RETURN_COLS = [f"fwd_{w}d" for w in FORWARD_WINDOWS]
FORWARD_BINARY_COLS = [f"fwd_{w}d_up" for w in FORWARD_WINDOWS]


# ============================================================
# BUILD ANALOGUE DATABASE
# ============================================================

def build_analogue_database(all_data):
    """
    Build the master database of historical moments.
    Each row is one ticker on one day, with:
      - The trailing return vector (fingerprint)
      - Supplementary features (regime context)
      - Forward returns (what actually happened next)

    This is the database the matching algorithm searches against.
    """
    all_rows = []

    for ticker, df in all_data.items():
        print(f"  Processing {ticker}...")
        df = compute_return_vector(df, ticker)
        df = compute_supplementary_features(df)
        df = compute_vol_normalized_features(df)   # M9: vol-normalized fingerprint
        df = compute_candlestick_features(df)      # intra-bar microstructure (Gemini rec)

        # Keep only rows with complete data (need 90 days trailing + forward windows)
        # VOL_NORM_COLS and CANDLE_COLS stored for PatternMatcher feature-set switching.
        required_cols = (FEATURE_COLS + VOL_NORM_COLS + CANDLE_COLS + FORWARD_RETURN_COLS +
                         FORWARD_BINARY_COLS + ["Date", "Ticker", "Open", "High", "Low", "Close"])
        subset = df[required_cols].dropna().reset_index(drop=True)

        if len(subset) < 100:
            print(f"    Skipping {ticker}: not enough complete data ({len(subset)} rows)")
            continue

        all_rows.append(subset)

    db = pd.concat(all_rows, ignore_index=True)

    print(f"\n  Analogue database built:")
    print(f"  Total entries: {len(db):,}")
    print(f"  Tickers: {db['Ticker'].nunique()}")
    print(f"  Date range: {db['Date'].min().strftime('%Y-%m-%d')} → {db['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Features per entry: {NUM_FEATURES}")

    return db


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(db, train_end=TRAIN_END, val_start=VAL_START,
                   val_end=VAL_END, test_start=TEST_START):
    """
    Split the analogue database temporally.
    Training analogues come from the past only — no future leakage.

    The matching algorithm searches the TRAINING set to find twins.
    We evaluate whether those twins' forward returns predict correctly
    on the VALIDATION set.
    """
    db["Date"] = pd.to_datetime(db["Date"])

    splits = {
        "train": db[db["Date"] <= train_end].copy(),
        "val": db[(db["Date"] >= val_start) & (db["Date"] <= val_end)].copy(),
        "test": db[db["Date"] >= test_start].copy(),
    }

    for name, split_df in splits.items():
        print(f"  {name:5s}: {len(split_df):>8,} entries | "
              f"{split_df['Ticker'].nunique()} tickers | "
              f"{split_df['Date'].min().strftime('%Y-%m-%d')} → "
              f"{split_df['Date'].max().strftime('%Y-%m-%d')}")

    return splits


# ============================================================
# FIT SCALER AND NEAREST NEIGHBORS INDEX
# ============================================================

def fit_matching_index(train_db, n_neighbors=TOP_K_MATCHES):
    """
    Fit a StandardScaler on training features and build a NearestNeighbors
    index for fast analogue retrieval.

    The scaler ensures all return windows are on the same scale
    (a 1-day return of 5% and a 90-day return of 5% have very different
    significance). Cosine distance is used because we care about the
    SHAPE of the return profile, not the magnitude.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_db[FEATURE_COLS].values)

    # Cosine distance: finds vectors pointing in the same direction
    # regardless of magnitude. A stock down 2% across all windows matches
    # a stock down 5% across all windows — same pattern, different scale.
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",  # Exact search (fast enough for <1M entries)
        n_jobs=-1,           # Use all CPU cores
    )
    nn.fit(X_train)

    # Save for reuse
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / "analogue_scaler.pkl")
    joblib.dump(nn, MODEL_DIR / "analogue_nn_index.pkl")

    print(f"  NearestNeighbors index fitted: {len(X_train):,} entries, {n_neighbors} neighbors")
    return scaler, nn


# ============================================================
# EVALUATION UTILITIES
# ============================================================

def evaluate_predictions(y_true, y_pred_proba, threshold=0.53):
    """
    Evaluate analogue-based predictions with confidence filtering.
    Carried forward from Phase 1 — same interface so results.tsv stays consistent.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred_all = (y_pred_proba >= 0.5).astype(int)

    confident_mask = (y_pred_proba >= threshold) | (y_pred_proba <= (1 - threshold))
    y_true_conf = y_true[confident_mask]
    y_pred_conf = (y_pred_proba[confident_mask] >= 0.5).astype(int)

    metrics = {
        "total_samples": len(y_true),
        "accuracy_all": accuracy_score(y_true, y_pred_all),
        "confident_trades": int(confident_mask.sum()),
        "confident_pct": confident_mask.mean(),
        "accuracy_confident": accuracy_score(y_true_conf, y_pred_conf) if len(y_true_conf) > 0 else 0.0,
        "precision_confident": precision_score(y_true_conf, y_pred_conf, zero_division=0) if len(y_true_conf) > 0 else 0.0,
        "recall_confident": recall_score(y_true_conf, y_pred_conf, zero_division=0) if len(y_true_conf) > 0 else 0.0,
        "f1_confident": f1_score(y_true_conf, y_pred_conf, zero_division=0) if len(y_true_conf) > 0 else 0.0,
    }

    return metrics


def print_metrics(metrics, label=""):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  EVALUATION{f' — {label}' if label else ''}")
    print(f"{'='*50}")
    print(f"  Total samples:      {metrics['total_samples']:,}")
    print(f"  Accuracy (all):     {metrics['accuracy_all']:.1%}")
    print(f"  Confident trades:   {metrics['confident_trades']:,} ({metrics['confident_pct']:.1%} of total)")
    print(f"  Accuracy (conf):    {metrics['accuracy_confident']:.1%}")
    print(f"  Precision (conf):   {metrics['precision_confident']:.1%}")
    print(f"  Recall (conf):      {metrics['recall_confident']:.1%}")
    print(f"  F1 (conf):          {metrics['f1_confident']:.1%}")
    print(f"{'='*50}\n")


def save_results(metrics, experiment_name, filepath=None):
    """Append experiment results to results.tsv for autoresearch tracking."""
    if filepath is None:
        filepath = RESULTS_DIR / "results.tsv"

    RESULTS_DIR.mkdir(exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment_name,
        **metrics,
    }

    df_row = pd.DataFrame([row])

    if filepath.exists():
        df_row.to_csv(filepath, mode="a", header=False, index=False, sep="\t")
    else:
        df_row.to_csv(filepath, index=False, sep="\t")

    print(f"  Results saved to {filepath}")


# ============================================================
# MAIN — run this to build the analogue database
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SYSTEM A: Quant Engine — Building Analogue Database")
    print("=" * 60)

    # Step 1: Download data (25 years)
    print("\n[1/4] Downloading market data (25-year lookback)...")
    all_data = download_data()

    # Step 2: Build analogue database
    print("\n[2/4] Building analogue database (return vectors + supplementary features)...")
    db = build_analogue_database(all_data)

    # Step 3: Temporal split
    print("\n[3/4] Splitting data (temporal — no leakage)...")
    splits = temporal_split(db)

    # Step 4: Fit matching index on training data
    print("\n[4/4] Fitting NearestNeighbors index...")
    scaler, nn = fit_matching_index(splits["train"])

    # Save everything
    DATA_DIR.mkdir(exist_ok=True)
    splits["train"].to_parquet(DATA_DIR / "train_db.parquet", index=False)
    splits["val"].to_parquet(DATA_DIR / "val_db.parquet", index=False)
    splits["test"].to_parquet(DATA_DIR / "test_db.parquet", index=False)
    db.to_parquet(DATA_DIR / "full_analogue_db.parquet", index=False)

    print(f"\n  Saved: train_db.parquet, val_db.parquet, test_db.parquet")

    print("\n" + "=" * 60)
    print("  SETUP COMPLETE — ready to run strategy.py")
    print("=" * 60 + "\n")
