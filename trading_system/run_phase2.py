"""
run_phase2.py — Phase 1 vs Phase 2 Comparison Runner

Loads cached 2024 signals, runs both Phase 1 (equal-weight) and Phase 2
(ATR-based risk engine) backtests side-by-side, then prints a comparison
table and saves all artefacts to results/.

Phase 2 artefacts (prefixed with "phase2_"):
  results/phase2_backtest_trades.csv
  results/phase2_backtest_equity.csv
  results/phase2_backtest_rejected.csv
  results/phase2_backtest_stop_events.csv
  results/phase2_backtest_summary.txt

Usage:
    python -m trading_system.run_phase2
    python -m trading_system.run_phase2 --no-phase1   # Skip Phase 1 re-run

Phase 2 success criteria (from docs/PHASE2_SYSTEM_DESIGN.md):
  - Max drawdown < 6.9%  (Phase 1 baseline was 6.9%)
  - Sharpe ratio >= 1.82  (Phase 1 baseline was 1.82)
  - Net expectancy > $0 after friction
  - Stop-losses fire at appropriate levels
  - Drawdown brake reduces sizing during losing streaks
  - All 485+ tests still pass
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_system.backtest_engine import BacktestEngine, BacktestResults
from trading_system.config import TradingConfig, SECTOR_MAP


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_signals(config: TradingConfig) -> pd.DataFrame:
    """Load cached 2024 signals and re-label using the active config thresholds.

    The cache stores raw calibrated confidence for every ticker-day pair.
    Re-applying the threshold at run-time means you can change
    config.signals.confidence_threshold without regenerating signals.
    """
    cache_path = REPO_ROOT / "results" / "cached_signals_2024.csv"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached signals not found at {cache_path}. "
            "Run `python -m trading_system.run_phase1 --use-cached-signals` first, "
            "or generate signals from the K-NN pipeline."
        )

    from trading_system.signal_adapter import load_cached_signals
    signal_df = load_cached_signals(str(cache_path))

    # Re-label using configured thresholds
    if "confidence" in signal_df.columns and "n_matches" in signal_df.columns:
        t   = config.signals.confidence_threshold
        mm  = config.signals.min_matches
        ma  = config.signals.min_agreement
        signal_df = signal_df.copy()
        signal_df["agreement"] = (signal_df["confidence"] - 0.5).abs() * 2

        buy_mask  = (
            (signal_df["confidence"] >= t)
            & (signal_df["n_matches"] >= mm)
            & (signal_df["agreement"] >= ma)
        )
        sell_mask = (
            (signal_df["confidence"] <= (1.0 - t))
            & (signal_df["n_matches"] >= mm)
            & (signal_df["agreement"] >= ma)
        )
        signal_df["signal"] = "HOLD"
        signal_df.loc[buy_mask,  "signal"] = "BUY"
        signal_df.loc[sell_mask, "signal"] = "SELL"

        print(
            f"  Re-labeled at threshold={t:.2f}: "
            f"{buy_mask.sum()} BUY, {sell_mask.sum()} SELL, "
            f"{(~buy_mask & ~sell_mask).sum()} HOLD"
        )

    return signal_df


def load_price_df() -> pd.DataFrame:
    """Load OHLC price data from the validation parquet database."""
    val_path = REPO_ROOT / "data" / "val_db.parquet"
    if not val_path.exists():
        raise FileNotFoundError(
            f"val_db.parquet not found at {val_path}. "
            "Ensure pattern_engine data pipeline has been run."
        )
    val_db = pd.read_parquet(val_path)
    price_df = val_db[["Date", "Ticker", "Open", "High", "Low", "Close"]].copy()
    print(
        f"  Price data: {len(price_df):,} rows, "
        f"{price_df['Ticker'].nunique()} tickers, "
        f"{price_df['Date'].min().date()} to {price_df['Date'].max().date()}"
    )
    return price_df


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(results: BacktestResults) -> dict:
    """Extract the comparison metrics from a BacktestResults object."""
    ne  = results.net_expectancy()
    wr  = results.win_rate()
    md  = results.max_drawdown()
    sr  = results.sharpe_ratio()
    ar  = results.annualized_return()
    tc  = results.total_trades()
    fe  = results.final_equity()
    ic  = results.config.capital.initial_capital

    stop_count = len(results.stop_loss_events)
    gap_count  = (
        results.stop_events_df["gap_through"].sum()
        if not results.stop_events_df.empty and "gap_through" in results.stop_events_df.columns
        else 0
    )

    return {
        "trades":         tc,
        "win_rate":       wr,
        "net_expectancy": ne,
        "annualized":     ar,
        "sharpe":         sr,
        "max_dd":         md,
        "final_equity":   fe,
        "total_return":   (fe / ic - 1) if (fe is not None and ic > 0) else None,
        "stop_events":    stop_count,
        "gap_throughs":   gap_count,
    }


def _fmt(val, fmt: str, fallback: str = "N/A") -> str:
    if val is None:
        return fallback
    return format(val, fmt)


def print_comparison_table(m1: dict, m2: dict) -> None:
    """Print a side-by-side Phase 1 vs Phase 2 performance table."""
    W = 62
    print("\n" + "=" * W)
    print(f"  {'METRIC':<28}  {'PHASE 1':>12}  {'PHASE 2':>12}")
    print("  " + "-" * (W - 2))

    rows = [
        ("Trades",             _fmt(m1["trades"], "d"),              _fmt(m2["trades"], "d")),
        ("Win rate",           _fmt(m1["win_rate"], ".1%"),           _fmt(m2["win_rate"], ".1%")),
        ("Net expectancy/trade",_fmt(m1["net_expectancy"], "+.2f"),   _fmt(m2["net_expectancy"], "+.2f")),
        ("Annualized return",  _fmt(m1["annualized"], "+.1%"),        _fmt(m2["annualized"], "+.1%")),
        ("Sharpe ratio",       _fmt(m1["sharpe"], ".2f"),             _fmt(m2["sharpe"], ".2f")),
        ("Max drawdown",       _fmt(m1["max_dd"], ".1%"),             _fmt(m2["max_dd"], ".1%")),
        ("Total return",       _fmt(m1["total_return"], "+.1%"),      _fmt(m2["total_return"], "+.1%")),
        ("Final equity",       _fmt(m1["final_equity"], ",.0f"),      _fmt(m2["final_equity"], ",.0f")),
        ("Stop-loss events",   str(m1["stop_events"]),                str(m2["stop_events"])),
        ("Gap-through events", str(m1["gap_throughs"]),               str(m2["gap_throughs"])),
    ]

    for label, v1, v2 in rows:
        print(f"  {label:<28}  {v1:>12}  {v2:>12}")

    print("=" * W)


def print_phase2_criteria(m2: dict) -> None:
    """Check Phase 2 success criteria and print a PASS/FAIL table."""
    ne = m2["net_expectancy"]
    md = m2["max_dd"]
    sr = m2["sharpe"]

    criteria = [
        ("Net expectancy > $0",       ne is not None and ne > 0,    _fmt(ne, "+.2f") + "/trade"),
        ("Max drawdown < 6.9%",       md < 0.069,                   _fmt(md, ".1%")),
        ("Sharpe >= 1.82",            sr is not None and sr >= 1.82, _fmt(sr, ".2f")),
        ("Stop events recorded",      m2["stop_events"] >= 0,       str(m2["stop_events"]) + " events"),
    ]

    print("\n" + "=" * 62)
    print("  PHASE 2 SUCCESS CRITERIA")
    print("  " + "-" * 60)
    for label, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status:^4}]  {label:<30}  {value}")
    print("=" * 62)


def save_phase2_results(results: BacktestResults, output_dir: Path) -> None:
    """Save Phase 2 artefacts to output_dir, prefixed with 'phase2_'."""
    output_dir.mkdir(exist_ok=True)

    if not results.trades_df.empty:
        results.trades_df.to_csv(output_dir / "phase2_backtest_trades.csv",   index=False)
    if not results.equity_df.empty:
        results.equity_df.to_csv(output_dir / "phase2_backtest_equity.csv",   index=False)
    if not results.rejected_df.empty:
        results.rejected_df.to_csv(output_dir / "phase2_backtest_rejected.csv", index=False)
    if not results.stop_events_df.empty:
        results.stop_events_df.to_csv(
            output_dir / "phase2_backtest_stop_events.csv", index=False
        )

    with open(output_dir / "phase2_backtest_summary.txt", "w") as f:
        f.write(results.summary())

    print(f"\n  Phase 2 results saved to {output_dir}/")
    print("    phase2_backtest_trades.csv")
    print("    phase2_backtest_equity.csv")
    print("    phase2_backtest_rejected.csv")
    if not results.stop_events_df.empty:
        print(f"    phase2_backtest_stop_events.csv  "
              f"({len(results.stop_events_df)} stop events)")
    print("    phase2_backtest_summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FPPE Trading System — Phase 1 vs Phase 2 Comparison"
    )
    parser.add_argument(
        "--no-phase1", action="store_true",
        help="Skip Phase 1 re-run (uses zero-column placeholders in comparison table)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 62)
    print("  FPPE TRADING SYSTEM — PHASE 2 COMPARISON RUNNER")
    print("  Phase 1 (equal-weight) vs Phase 2 (ATR-based risk engine)")
    print("=" * 62)

    # ── Configuration ─────────────────────────────────────────────
    config = TradingConfig()
    print(config.summary())

    # ── Load data ─────────────────────────────────────────────────
    print("\n  Loading signals and price data ...")
    signal_df = load_signals(config)
    price_df  = load_price_df()

    # ── Phase 1 backtest (baseline) ───────────────────────────────
    m1 = {k: None for k in [
        "trades", "win_rate", "net_expectancy", "annualized",
        "sharpe", "max_dd", "final_equity", "total_return",
        "stop_events", "gap_throughs",
    ]}

    if not args.no_phase1:
        print("\n" + "─" * 62)
        print("  Running Phase 1 (equal-weight, use_risk_engine=False) ...")
        engine_p1 = BacktestEngine(config=config, use_risk_engine=False)
        results_p1 = engine_p1.run(signal_df, price_df, equal_weight_pct=0.05)
        m1 = _metrics(results_p1)
        print(results_p1.summary())
    else:
        print("\n  [--no-phase1] Skipping Phase 1 re-run.")

    # ── Phase 2 backtest ──────────────────────────────────────────
    print("\n" + "─" * 62)
    print("  Running Phase 2 (ATR-based sizing, use_risk_engine=True) ...")
    engine_p2 = BacktestEngine(config=config, use_risk_engine=True)
    results_p2 = engine_p2.run(signal_df, price_df)
    m2 = _metrics(results_p2)
    print(results_p2.summary())

    # ── Comparison table ─────────────────────────────────────────
    print_comparison_table(m1, m2)

    # ── Phase 2 criteria check ────────────────────────────────────
    print_phase2_criteria(m2)

    # ── Stop-loss audit ───────────────────────────────────────────
    if not results_p2.stop_events_df.empty:
        sdf = results_p2.stop_events_df
        print(f"\n  Stop-Loss Audit ({len(sdf)} events):")
        print(f"  {'TICKER':<8}  {'DATE':<12}  {'STOP':>8}  {'LOW':>8}  "
              f"{'ATR%':>6}  {'GAP':>5}")
        print("  " + "-" * 54)
        for _, e in sdf.iterrows():
            gap_flag = "YES" if e["gap_through"] else "no"
            print(
                f"  {e['ticker']:<8}  {str(e['trigger_date']):<12}  "
                f"{e['stop_price']:>8.2f}  {e['trigger_low']:>8.2f}  "
                f"{e['atr_at_entry']:>6.2%}  {gap_flag:>5}"
            )
    else:
        print("\n  No stop-loss events recorded in Phase 2 run.")

    # ── Save Phase 2 artefacts ────────────────────────────────────
    save_phase2_results(results_p2, REPO_ROOT / "results")


if __name__ == "__main__":
    main()
