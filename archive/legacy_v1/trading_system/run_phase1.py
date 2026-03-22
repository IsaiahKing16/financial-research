"""
run_phase1.py — Phase 1 Backtest Runner

Runs the equal-weight backtest on 2024 validation data.
This is a PIPELINE VERIFICATION step — no parameters are optimized.

Usage:
    python -m trading_system.run_phase1
    python -m trading_system.run_phase1 --use-cached-signals
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_system.config import TradingConfig, SECTOR_MAP
from trading_system.backtest_engine import BacktestEngine


def generate_or_load_signals(use_cached: bool = False) -> pd.DataFrame:
    """Generate FPPE signals for 2024 validation data, or load from cache.

    Signal generation requires running the full K-NN matching pipeline,
    which takes several minutes. Caching allows rapid iteration on the
    backtester without re-running matching every time.
    """
    cache_path = REPO_ROOT / "results" / "cached_signals_2024.csv"

    if use_cached and cache_path.exists():
        print(f"\n  Loading cached signals from {cache_path}")
        from trading_system.signal_adapter import load_cached_signals
        return load_cached_signals(str(cache_path))

    print("\n  Generating signals from FPPE (this takes several minutes)...")
    from trading_system.signal_adapter import (
        simulate_signals_from_val_db, save_signals,
    )

    train_db = pd.read_parquet(REPO_ROOT / "data" / "processed" / "train_db.parquet")
    val_db = pd.read_parquet(REPO_ROOT / "data" / "processed" / "val_db.parquet")

    signal_df = simulate_signals_from_val_db(
        val_db=val_db,
        train_db=train_db,
        sector_map=SECTOR_MAP,
    )

    # Cache for next time
    save_signals(signal_df, str(cache_path))

    return signal_df


def build_price_df() -> pd.DataFrame:
    """Load OHLC price data for the validation period from the parquet database."""
    val_db = pd.read_parquet(REPO_ROOT / "data" / "processed" / "val_db.parquet")

    # The val_db already has Date, Ticker, Open, High, Low, Close
    price_df = val_db[["Date", "Ticker", "Open", "High", "Low", "Close"]].copy()

    print(f"  Price data: {len(price_df):,} rows, "
          f"{price_df['Ticker'].nunique()} tickers, "
          f"{price_df['Date'].min().date()} to {price_df['Date'].max().date()}")

    return price_df


def run_baseline_spy(price_df: pd.DataFrame, config: TradingConfig) -> dict:
    """Baseline 1: SPY buy-and-hold.

    Invest 100% of capital in SPY on day 1, hold throughout.
    Apply same friction model for fair comparison.
    """
    spy_prices = price_df[price_df["Ticker"] == "SPY"].sort_values("Date")
    if spy_prices.empty:
        return {"total_return": None, "sharpe": None, "max_dd": None}

    initial = config.capital.initial_capital
    # Buy at first day's open with slippage
    entry_price = spy_prices.iloc[0]["Open"] * (1 + config.costs.total_entry_bps / 10_000)
    shares = initial / entry_price

    # Track daily
    equity_series = shares * spy_prices["Close"].values
    returns = pd.Series(equity_series).pct_change().dropna()

    # Final value with exit friction
    final_raw = spy_prices.iloc[-1]["Close"]
    exit_price = final_raw * (1 - config.costs.total_exit_bps / 10_000)
    final_equity = shares * exit_price

    total_return = (final_equity / initial) - 1
    n_days = len(spy_prices)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # Sharpe
    daily_rf = config.costs.risk_free_annual_rate / 252
    excess = returns - daily_rf
    sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else None

    # Max drawdown
    cummax = pd.Series(equity_series).cummax()
    drawdown = (pd.Series(equity_series) - cummax) / cummax
    max_dd = abs(drawdown.min())

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "final_equity": final_equity,
    }


def run_baseline_raw_signals(signal_df: pd.DataFrame, price_df: pd.DataFrame) -> dict:
    """Baseline 3: FPPE raw signal accuracy.

    What % of BUY signals were followed by a price increase within
    the holding window (7 days, matching FPPE's projection horizon)?
    This isolates signal quality from trading system mechanics.
    """
    buys = signal_df[signal_df["signal"] == "BUY"].copy()
    if buys.empty:
        return {"buy_accuracy": None, "sell_accuracy": None, "total_signals": 0}

    price_dict = {}
    for _, row in price_df.iterrows():
        key = (pd.Timestamp(row["Date"]), row["Ticker"])
        price_dict[key] = float(row["Close"])

    # For each BUY signal, check if price went up in next 7 trading days
    all_dates = sorted(price_df["Date"].unique())
    date_list = [pd.Timestamp(d) for d in all_dates]

    buy_correct = 0
    buy_total = 0
    sell_correct = 0
    sell_total = 0

    for _, sig in signal_df.iterrows():
        sig_date = pd.Timestamp(sig["date"])
        ticker = sig["ticker"]
        signal = sig["signal"]

        if signal not in ("BUY", "SELL"):
            continue

        # Find price at signal date
        price_now = price_dict.get((sig_date, ticker))
        if price_now is None:
            continue

        # Find price 7 trading days later
        idx = None
        for i, d in enumerate(date_list):
            if d == sig_date:
                idx = i
                break
        if idx is None or idx + 7 >= len(date_list):
            continue

        future_date = date_list[idx + 7]
        price_future = price_dict.get((future_date, ticker))
        if price_future is None:
            continue

        if signal == "BUY":
            buy_total += 1
            if price_future > price_now:
                buy_correct += 1
        elif signal == "SELL":
            sell_total += 1
            if price_future < price_now:
                sell_correct += 1

    return {
        "buy_accuracy": buy_correct / buy_total if buy_total > 0 else None,
        "sell_accuracy": sell_correct / sell_total if sell_total > 0 else None,
        "buy_signals": buy_total,
        "sell_signals": sell_total,
    }


def main():
    parser = argparse.ArgumentParser(description="FPPE Trading System — Phase 1 Backtest")
    parser.add_argument("--use-cached-signals", action="store_true",
                        help="Load signals from cache instead of regenerating")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FPPE TRADING SYSTEM — PHASE 1 BACKTEST")
    print("  Pipeline verification on 2024 validation data")
    print("=" * 60)

    # ── Configuration ────────────────────────────────────────────
    config = TradingConfig()
    print(config.summary())

    # ── Load data ────────────────────────────────────────────────
    signal_df = generate_or_load_signals(use_cached=args.use_cached_signals)
    price_df = build_price_df()

    # ── Re-label signals using config threshold ───────────────────
    # Cached signals store calibrated confidence for all rows.
    # We re-apply the configured threshold here so changing
    # config.signals.confidence_threshold takes effect immediately
    # without regenerating signals from the K-NN pipeline.
    if "confidence" in signal_df.columns and "n_matches" in signal_df.columns:
        threshold = config.signals.confidence_threshold
        min_matches = config.signals.min_matches
        min_agreement = config.signals.min_agreement
        signal_df = signal_df.copy()
        signal_df["agreement"] = (signal_df["confidence"] - 0.5).abs() * 2

        buy_mask = (
            (signal_df["confidence"] >= threshold) &
            (signal_df["n_matches"] >= min_matches) &
            (signal_df["agreement"] >= min_agreement)
        )
        sell_mask = (
            (signal_df["confidence"] <= (1.0 - threshold)) &
            (signal_df["n_matches"] >= min_matches) &
            (signal_df["agreement"] >= min_agreement)
        )
        signal_df["signal"] = "HOLD"
        signal_df.loc[buy_mask, "signal"] = "BUY"
        signal_df.loc[sell_mask, "signal"] = "SELL"
        print(f"\n  Re-labeled at threshold={threshold:.2f}: "
              f"{buy_mask.sum()} BUY, {sell_mask.sum()} SELL, "
              f"{(~buy_mask & ~sell_mask).sum()} HOLD")

    # ── Run backtest ─────────────────────────────────────────────
    engine = BacktestEngine(config)
    results = engine.run(signal_df, price_df, equal_weight_pct=0.05)

    # ── Print results ────────────────────────────────────────────
    print(results.summary())

    # ── Run baselines ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISONS")
    print("=" * 60)

    # Baseline 1: SPY buy-and-hold
    spy_results = run_baseline_spy(price_df, config)
    print(f"\n  SPY Buy-and-Hold:")
    print(f"    Total return:     {spy_results['total_return']:>8.1%}" if spy_results['total_return'] else "    No data")
    print(f"    Annualized:       {spy_results['annualized_return']:>8.1%}" if spy_results.get('annualized_return') else "")
    print(f"    Sharpe ratio:     {spy_results['sharpe']:>8.2f}" if spy_results['sharpe'] else "")
    print(f"    Max drawdown:     {spy_results['max_dd']:>8.1%}" if spy_results['max_dd'] else "")

    # Baseline 3: Raw signal accuracy
    raw_results = run_baseline_raw_signals(signal_df, price_df)
    print(f"\n  FPPE Raw Signal Accuracy (7-day horizon):")
    print(f"    BUY accuracy:     {raw_results['buy_accuracy']:>8.1%} ({raw_results['buy_signals']} signals)" if raw_results['buy_accuracy'] else "    No BUY signals")
    print(f"    SELL accuracy:    {raw_results['sell_accuracy']:>8.1%} ({raw_results['sell_signals']} signals)" if raw_results.get('sell_accuracy') else "    No SELL signals")

    # ── v1 Success Criteria Check ────────────────────────────────
    print("\n" + "=" * 60)
    print("  v1 SUCCESS CRITERIA CHECK")
    print("=" * 60)

    ne = results.net_expectancy()
    wr = results.win_rate()
    md = results.max_drawdown()
    tc = results.total_trades()

    criteria = [
        ("Net expectancy > $0", ne is not None and ne > 0, f"${ne:.2f}/trade" if ne else "N/A"),
        ("Drawdown ≤ 25%", md <= 0.25, f"{md:.1%}"),
        ("≥ 50 trades", tc >= 50, f"{tc} trades"),
        ("Beats ≥2 of 3 baselines", "Manual check required", "—"),
        ("No 30-day Sharpe < -1.0", "Manual check required", "—"),
    ]

    for name, passed, value in criteria:
        if isinstance(passed, bool):
            status = "PASS" if passed else "FAIL"
        else:
            status = str(passed)
        print(f"  [{status:^6}] {name:<30} {value}")

    # ── Save results ─────────────────────────────────────────────
    results.save(str(REPO_ROOT / "results"))
    print("\n  Done. Results saved to results/")


if __name__ == "__main__":
    main()
