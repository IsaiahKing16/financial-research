"""
run_phase2.py - Phase 2 Comparison Runner

Loads cached 2024 signals, runs:
  - Phase 1 baseline (use_risk_engine=False)
  - Phase 2 risk-managed backtest (use_risk_engine=True)

Then prints a metric comparison table and saves result artifacts to results/.

Usage:
    python -m trading_system.run_phase2
    python -m trading_system.run_phase2 --price-path data/val_db.parquet
"""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_system.backtest_engine import BacktestEngine
from trading_system.config import TradingConfig
from trading_system.signal_adapter import load_cached_signals

DEFAULT_SIGNALS_PATH = REPO_ROOT / "results" / "cached_signals_2024.csv"
DEFAULT_PRICE_PATH = REPO_ROOT / "data" / "val_db.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results"


def _load_price_df(price_path: Path) -> pd.DataFrame:
    """Load OHLC prices and return standardized DataFrame for backtest."""
    if not price_path.exists():
        raise FileNotFoundError(
            f"Price data not found: {price_path}. "
            "Pass --price-path to a parquet/csv file containing "
            "[Date, Ticker, Open, High, Low, Close]."
        )

    if price_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(price_path)
    elif price_path.suffix.lower() == ".csv":
        raw = pd.read_csv(price_path)
    else:
        raise ValueError(
            f"Unsupported price file format: {price_path.suffix}. "
            "Use parquet or csv."
        )

    required = ["Date", "Ticker", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(
            f"Price file missing required columns: {missing}. "
            f"Found: {list(raw.columns)}"
        )

    price_df = raw[required].copy()
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    # Align with signal dates (naive calendar dates): strip tz without shifting the NY session date.
    if getattr(price_df["Date"].dt, "tz", None) is not None:
        price_df["Date"] = (
            price_df["Date"]
            .dt.tz_convert("America/New_York")
            .dt.normalize()
            .dt.tz_localize(None)
        )
    return price_df


def _run_backtest_with_mode(
    engine: BacktestEngine,
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    use_risk_engine: bool,
) -> Any:
    """Run backtest in baseline/risk mode with interface compatibility checks."""
    run_signature = inspect.signature(engine.run)
    has_use_risk_engine = "use_risk_engine" in run_signature.parameters

    if use_risk_engine and not has_use_risk_engine:
        raise RuntimeError(
            "BacktestEngine.run() does not support use_risk_engine yet. "
            "Phase 2 integration (SLE-13) must be merged before run_phase2 "
            "can execute risk-managed mode."
        )

    if has_use_risk_engine:
        return engine.run(
            signal_df=signal_df,
            price_df=price_df,
            use_risk_engine=use_risk_engine,
        )

    # Backward-compatible baseline call path
    return engine.run(signal_df=signal_df, price_df=price_df)


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:,.2%}"


def _fmt_float(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:,.3f}"


def _fmt_currency(value: Optional[float]) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def _fmt_delta(value: Optional[float], formatter: str) -> str:
    if value is None:
        return "N/A"
    if formatter == "pct":
        return f"{value:+,.2%}"
    if formatter == "float":
        return f"{value:+,.3f}"
    return f"{value:+,.2f}"


def _print_comparison_table(phase1_results: Any, phase2_results: Any) -> None:
    """Print core Phase 1 vs Phase 2 metric comparison table."""
    rows = [
        ("Annual return", phase1_results.annualized_return(), phase2_results.annualized_return(), "pct"),
        ("Sharpe", phase1_results.sharpe_ratio(), phase2_results.sharpe_ratio(), "float"),
        ("Max drawdown", phase1_results.max_drawdown(), phase2_results.max_drawdown(), "pct"),
        ("Win rate", phase1_results.win_rate(), phase2_results.win_rate(), "pct"),
        ("Net expectancy", phase1_results.net_expectancy(), phase2_results.net_expectancy(), "currency"),
    ]

    print("\n" + "=" * 84)
    print("  PHASE 1 VS PHASE 2 COMPARISON")
    print("=" * 84)
    print(f"{'Metric':<20} {'Phase 1':>18} {'Phase 2':>18} {'Delta (P2-P1)':>20}")
    print("-" * 84)

    for name, p1, p2, kind in rows:
        if kind == "pct":
            p1_fmt = _fmt_pct(p1)
            p2_fmt = _fmt_pct(p2)
        elif kind == "float":
            p1_fmt = _fmt_float(p1)
            p2_fmt = _fmt_float(p2)
        else:
            p1_fmt = _fmt_currency(p1)
            p2_fmt = _fmt_currency(p2)

        delta = p2 - p1 if p1 is not None and p2 is not None else None
        delta_fmt = _fmt_delta(delta, kind if kind != "currency" else "currency")
        print(f"{name:<20} {p1_fmt:>18} {p2_fmt:>18} {delta_fmt:>20}")

    print("=" * 84)


def _save_results_bundle(results: Any, output_dir: Path, prefix: str) -> None:
    """Save trades, equity, rejected signals, and summary for a phase run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(results, "trades_df", None) is not None:
        results.trades_df.to_csv(output_dir / f"{prefix}_trades.csv", index=False)
    if getattr(results, "equity_df", None) is not None:
        results.equity_df.to_csv(output_dir / f"{prefix}_equity.csv", index=False)
    if getattr(results, "rejected_df", None) is not None:
        results.rejected_df.to_csv(output_dir / f"{prefix}_rejected.csv", index=False)

    with open(output_dir / f"{prefix}_summary.txt", "w", encoding="utf-8") as handle:
        handle.write(results.summary())


def _extract_stop_loss_events_df(results: Any) -> pd.DataFrame:
    """Extract stop-loss events from results in a backward-compatible way."""
    events_df = getattr(results, "stop_loss_events_df", None)
    if isinstance(events_df, pd.DataFrame):
        return events_df.copy()

    events = getattr(results, "stop_loss_events", None)
    if events:
        rows = []
        for event in events:
            if isinstance(event, dict):
                rows.append(event)
            elif is_dataclass(event):
                rows.append(asdict(event))
            else:
                rows.append({"event": str(event)})
        return pd.DataFrame(rows)

    trades_df = getattr(results, "trades_df", None)
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and "exit_reason" in trades_df.columns:
        stop_df = trades_df[trades_df["exit_reason"] == "stop_loss"].copy()
        preferred_cols = [
            "trade_id",
            "ticker",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "net_pnl",
            "holding_days",
            "exit_reason",
        ]
        existing_cols = [col for col in preferred_cols if col in stop_df.columns]
        return stop_df[existing_cols]

    return pd.DataFrame(
        columns=[
            "trade_id",
            "ticker",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "net_pnl",
            "holding_days",
            "exit_reason",
        ]
    )


def main() -> None:
    """Run baseline and risk-managed backtests and print/save comparison output."""
    parser = argparse.ArgumentParser(
        description="FPPE Trading System - Phase 2 comparison runner"
    )
    parser.add_argument(
        "--signals-path",
        type=Path,
        default=DEFAULT_SIGNALS_PATH,
        help="Path to cached signal CSV (default: results/cached_signals_2024.csv)",
    )
    parser.add_argument(
        "--price-path",
        type=Path,
        default=DEFAULT_PRICE_PATH,
        help="Path to OHLC price parquet/csv (default: data/val_db.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output artifacts (default: results/)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 84)
    print("  FPPE TRADING SYSTEM - PHASE 2 COMPARISON RUNNER")
    print("=" * 84)

    if not args.signals_path.exists():
        raise FileNotFoundError(
            f"Signals file not found: {args.signals_path}. "
            "Generate it first or point --signals-path to a valid cache file."
        )

    signal_df = load_cached_signals(str(args.signals_path))
    price_df = _load_price_df(args.price_path)

    print(f"\n  Signals: {len(signal_df):,} rows from {args.signals_path}")
    print(
        "  Prices:  "
        f"{len(price_df):,} rows, "
        f"{price_df['Ticker'].nunique()} tickers, "
        f"{price_df['Date'].min().date()} to {price_df['Date'].max().date()}"
    )

    config = TradingConfig()
    engine = BacktestEngine(config)

    print("\n  Running Phase 1 baseline (use_risk_engine=False)...")
    phase1_results = _run_backtest_with_mode(
        engine, signal_df, price_df, use_risk_engine=False
    )

    print("\n  Running Phase 2 (use_risk_engine=True)...")
    phase2_results = _run_backtest_with_mode(
        engine, signal_df, price_df, use_risk_engine=True
    )

    _print_comparison_table(phase1_results, phase2_results)

    _save_results_bundle(phase1_results, args.output_dir, "phase1")
    _save_results_bundle(phase2_results, args.output_dir, "phase2")

    stop_loss_df = _extract_stop_loss_events_df(phase2_results)
    stop_loss_path = args.output_dir / "phase2_stop_loss_events.csv"
    stop_loss_df.to_csv(stop_loss_path, index=False)

    print("\n  Saved output artifacts:")
    print(f"    - {args.output_dir / 'phase1_trades.csv'}")
    print(f"    - {args.output_dir / 'phase1_equity.csv'}")
    print(f"    - {args.output_dir / 'phase2_trades.csv'}")
    print(f"    - {args.output_dir / 'phase2_equity.csv'}")
    print(f"    - {stop_loss_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
