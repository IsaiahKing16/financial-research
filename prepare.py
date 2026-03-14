"""
prepare.py — Fixed data pipeline, feature engineering, and evaluation utilities.
This file is NOT modified by the autoresearch agent.
Human edits only for adding new data sources or indicators.

Phase 1: US Stocks (via yfinance, no API key needed)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import warnings
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS — adjust these for your setup
# ============================================================

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# Tickers to track (S&P 500 representative + high-volume individual stocks)
TICKERS = [
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq 100 ETF
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "AMZN",  # Amazon
    "GOOGL", # Alphabet
    "META",  # Meta
    "TSLA",  # Tesla
    "JPM",   # JPMorgan
    "AVGO",
    "ORCL",
    "ADBE",
    "CRM",
    "AMD",
    "NFLX",
    "INTC",
    "CSCO",
    "QCOM",
    "TXN",
    "MU",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "V",
    "MA",
    "AXP",
    "PYPL",
    "BRK-B",
    "LLY",
    "UNH",
    "JNJ",
    "ABBV",
    "MRK",
    "PFE",
    "TMO",
    "ISRG",
    "AMGN",
    "GILD",
    "WMT",
    "COST",
    "PG",
    "KO",
    "PEP",
    "HD",
    "DIS",
    "CAT",
    "BA",
    "GE",
    "XOM",
    "CVX"
]

# Time split — model learns on the past, tests on the future
TRAIN_END = "2023-12-31"      # Train on data up to this date
VAL_START = "2024-01-01"      # Validate on 2024
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"     # Test on 2025+ (most recent data)

# Feature engineering
WINDOW_SIZE = 60              # Days of history per prediction sample
PREDICTION_HORIZON = 1        # Predict 1 day ahead (binary: up or down)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(tickers=TICKERS, start="2010-01-01", end=None, force_refresh=False):
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
            df = yf.download(ticker, start=start, end=end, progress=False)
            df = df.reset_index()
            
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] 
                              for col in df.columns]
            
            df.to_csv(cache_path, index=False)
            print(f"{len(df)} rows")
        
        all_data[ticker] = df
    
    return all_data


# ============================================================
# FEATURE ENGINEERING — 38 indicators
# ============================================================

def compute_features(df, ticker=""):
    """
    Compute 38 technical indicators from OHLCV data.
    Based on Noisy's prediction market model feature set.
    
    Returns DataFrame with features added as columns.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Ensure we have the required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    
    # --- Price-based features ---
    df["YesterdayClose"] = df["Close"].shift(1)
    df["YesterdayOpenLogR"] = np.log(df["Open"] / df["Open"].shift(1))
    df["YesterdayHighLogR"] = np.log(df["High"] / df["High"].shift(1))
    df["YesterdayLowLogR"] = np.log(df["Low"] / df["Low"].shift(1))
    df["YesterdayVolumeLogR"] = np.log((df["Volume"] + 1) / (df["Volume"].shift(1) + 1))
    df["YesterdayCloseLogR"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # --- Moving Averages ---
    df["MA10"] = SMAIndicator(df["Close"], window=10).sma_indicator()
    df["MA20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["MA30"] = SMAIndicator(df["Close"], window=30).sma_indicator()
    
    # --- Exponential Moving Averages ---
    df["EMA10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["EMA30"] = EMAIndicator(df["Close"], window=30).ema_indicator()
    
    # --- RSI ---
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    
    # --- MACD ---
    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    
    # --- Bollinger Bands ---
    bb = BollingerBands(df["Close"], window=20)
    df["BollingerUpper"] = bb.bollinger_hband()
    df["BollingerLower"] = bb.bollinger_lband()
    
    # --- Volatility ---
    df["Volatility_10"] = df["Close"].pct_change().rolling(10).std()
    df["Volatility_20"] = df["Close"].pct_change().rolling(20).std()
    df["Volatility_30"] = df["Close"].pct_change().rolling(30).std()
    
    # --- On-Balance Volume ---
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    
    # --- Statistical ---
    df["ZScore"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).std()
    
    # --- Temporal features ---
    df["DayOfWeek"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df["DayOfMonth"] = pd.to_datetime(df["Date"]).dt.day
    df["MonthNumber"] = pd.to_datetime(df["Date"]).dt.month
    
    # --- Momentum ---
    df["momentum_5d"] = df["Close"].pct_change(5)
    df["momentum_20d"] = df["Close"].pct_change(20)
    
    # --- Additional volatility ---
    df["volatility_5d"] = df["Close"].pct_change().rolling(5).std()
    df["volatility_20d"] = df["Close"].pct_change().rolling(20).std()
    
    # --- Overnight gap ---
    df["overnight_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    
    # --- Abnormal volume ---
    vol_mean = df["Volume"].rolling(20).mean()
    df["abnormal_vol"] = (df["Volume"] - vol_mean) / (vol_mean + 1)
    
    # --- Skewness (60-day rolling) ---
    df["skew_60"] = df["Close"].pct_change().rolling(60).skew()
    
    # --- Average True Range ---
    df["ATR"] = AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    
    # --- Target variable: will price go UP tomorrow? ---
    df["Target"] = (df["Close"].shift(-PREDICTION_HORIZON) > df["Close"]).astype(int)
    
    # Add ticker label
    df["Ticker"] = ticker
    
    return df


# List of feature columns used by the model
FEATURE_COLS = [
    "YesterdayClose", "YesterdayOpenLogR", "YesterdayHighLogR", 
    "YesterdayLowLogR", "YesterdayVolumeLogR", "YesterdayCloseLogR",
    "MA10", "MA20", "MA30", "EMA10", "EMA30",
    "RSI", "MACD", "MACD_Signal",
    "BollingerUpper", "BollingerLower",
    "Volatility_10", "Volatility_20", "Volatility_30",
    "OBV", "ZScore",
    "DayOfWeek", "DayOfMonth", "MonthNumber",
    "momentum_5d", "momentum_20d",
    "volatility_5d", "volatility_20d",
    "overnight_gap", "abnormal_vol", "skew_60", "ATR",
]

NUM_FEATURES = len(FEATURE_COLS)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(all_data, feature_cols=FEATURE_COLS, window_size=WINDOW_SIZE):
    """
    Prepare windowed dataset from all tickers.
    Returns X (samples, window, features), y (binary targets), dates, tickers.
    """
    all_X, all_y, all_dates, all_tickers = [], [], [], []
    
    for ticker, df in all_data.items():
        print(f"  Processing {ticker}...")
        df = compute_features(df, ticker)
        
        # Drop rows with NaN features
        subset = df[feature_cols + ["Target", "Date", "Ticker"]].dropna().reset_index(drop=True)
        
        if len(subset) < window_size + 1:
            print(f"    Skipping {ticker}: not enough data ({len(subset)} rows)")
            continue
        
        # Scale features per-ticker
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(subset[feature_cols])
        
        # Save scaler for this ticker
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler, MODEL_DIR / f"scaler_{ticker}.pkl")
        
        targets = subset["Target"].values
        dates = subset["Date"].values
        
        # Create sliding windows
        for i in range(window_size, len(subset) - 1):
            X_window = features_scaled[i - window_size:i]
            all_X.append(X_window)
            all_y.append(targets[i])
            all_dates.append(dates[i])
            all_tickers.append(ticker)
    
    X = np.array(all_X)
    y = np.array(all_y)
    dates = np.array(all_dates)
    tickers = np.array(all_tickers)
    
    print(f"\n  Total samples: {len(X)}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Class balance: {y.mean():.1%} positive (up days)")
    
    return X, y, dates, tickers


def temporal_split(X, y, dates, tickers, 
                   train_end=TRAIN_END, val_start=VAL_START, 
                   val_end=VAL_END, test_start=TEST_START):
    """
    Split data temporally — NO data leakage.
    Train on past, validate on recent past, test on present.
    """
    dates_dt = pd.to_datetime(dates)
    
    train_mask = dates_dt <= train_end
    val_mask = (dates_dt >= val_start) & (dates_dt <= val_end)
    test_mask = dates_dt >= test_start
    
    splits = {
        "train": (X[train_mask], y[train_mask], dates[train_mask], tickers[train_mask]),
        "val": (X[val_mask], y[val_mask], dates[val_mask], tickers[val_mask]),
        "test": (X[test_mask], y[test_mask], dates[test_mask], tickers[test_mask]),
    }
    
    for name, (X_s, y_s, d_s, t_s) in splits.items():
        print(f"  {name:5s}: {len(X_s):6d} samples | "
              f"{y_s.mean():.1%} positive | "
              f"{pd.to_datetime(d_s).min().strftime('%Y-%m-%d') if len(d_s) > 0 else 'N/A'} → "
              f"{pd.to_datetime(d_s).max().strftime('%Y-%m-%d') if len(d_s) > 0 else 'N/A'}")
    
    return splits


# ============================================================
# EVALUATION UTILITIES
# ============================================================

def evaluate_predictions(y_true, y_pred_proba, threshold=0.7):
    """
    Evaluate model predictions with confidence filtering.
    
    Args:
        y_true: actual binary labels
        y_pred_proba: predicted probabilities (0 to 1)
        threshold: minimum confidence to trade (default 0.7 from Noisy)
    
    Returns:
        dict of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # All predictions
    y_pred_all = (y_pred_proba >= 0.5).astype(int)
    
    # Filtered predictions (high confidence only)
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
    print(f"  Total samples:      {metrics['total_samples']}")
    print(f"  Accuracy (all):     {metrics['accuracy_all']:.1%}")
    print(f"  Confident trades:   {metrics['confident_trades']} ({metrics['confident_pct']:.1%} of total)")
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
    
    df = pd.DataFrame([row])
    
    if filepath.exists():
        df.to_csv(filepath, mode="a", header=False, index=False, sep="\t")
    else:
        df.to_csv(filepath, index=False, sep="\t")
    
    print(f"  Results saved to {filepath}")


# ============================================================
# MAIN — run this to download data and verify setup
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  FINANCIAL RESEARCH — Phase 1 Setup")
    print("="*60)
    
    # Step 1: Download data
    print("\n[1/3] Downloading market data...")
    all_data = download_data()
    
    # Step 2: Compute features and prepare dataset
    print("\n[2/3] Computing features (38 indicators)...")
    X, y, dates, tickers = prepare_dataset(all_data)
    
    # Step 3: Temporal split
    print("\n[3/3] Splitting data (temporal — no leakage)...")
    splits = temporal_split(X, y, dates, tickers)
    
    # Save prepared data for strategy.py to load
    DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "prepared_data.npz",
        X_train=splits["train"][0], y_train=splits["train"][1],
        X_val=splits["val"][0], y_val=splits["val"][1],
        X_test=splits["test"][0], y_test=splits["test"][1],
        dates_train=splits["train"][2], dates_val=splits["val"][2],
        dates_test=splits["test"][2],
    )
    print(f"\n  Prepared data saved to {DATA_DIR / 'prepared_data.npz'}")
    
    print("\n" + "="*60)
    print("  SETUP COMPLETE — ready to run strategy.py")
    print("="*60 + "\n")
