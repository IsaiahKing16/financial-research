"""
run_phase2.py — Phase 2 Comparison Runner

Loads cached FPPE signals and runs a direct comparison:
  - Phase 1 baseline (use_risk_engine=False)
  - Phase 2 risk-enabled (use_risk_engine=True)

Outputs:
  - Console comparison table (annual return, Sharpe, max DD, win rate, net expectancy)
  - CSV exports under results/ for trades, equity, rejections, and stop-loss events
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trading_system.backtest_engine import BacktestEngine, BacktestResults
from trading_system.config import TradingConfig
from trading_system.signal_adapter import load_cached_signals


def _load_price_df(price_path: Path) -> pd.DataFrame:
    """Load OHLC price data required by BacktestEngine."""
    if not price_path.exists():
        raise RuntimeError(f"Price data file not found: {price_path}")

    val_db = pd.read_parquet(price_path)
    required = ["Date", "Ticker", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in val_db.columns]
    if missing:
        raise RuntimeError(f"Price data missing required columns: {missing}")

    return val_db[required].copy()


def _require_phase2_compatible_engine() -> None:
    """Ensure BacktestEngine.run supports use_risk_engine for Phase 2 comparison."""
    signature = inspect.signature(BacktestEngine.run)
    if "use_risk_engine" not in signature.parameters:
        raise RuntimeError(
            "BacktestEngine.run() does not support 'use_risk_engine' yet. "
            "This runner requires SLE-13 integration to be merged first."
        )


def _compute_metrics(results: BacktestResults) -> Dict[str, Optional[float]]:
    """Extract comparable metrics from BacktestResults."""
    return {
        "annual_return": results.annualized_return(),
        "sharpe": results.sharpe_ratio(),
        "max_dd": results.max_drawdown(),
        "win_rate": results.win_rate(),
        "net_expectancy": results.net_expectancy(),
    }


def _format_metric(name: str, value: Optional[float]) -> str:
    """Format metric values for aligned console output."""
    if value is None:
        return "N/A"
    if name in {"annual_return", "max_dd", "win_rate"}:
        return f"{value:.2%}"
    if name == "net_expectancy":
        return f"${value:,.2f}"
    return f"{value:.3f}"


def _print_comparison(phase1: Dict[str, Optional[float]], phase2: Dict[str, Optional[float]]) -> None:
    """Print Phase 1 vs Phase 2 comparison table."""
    labels = {
        "annual_return": "Annual return",
        "sharpe": "Sharpe",
        "max_dd": "Max drawdown",
        "win_rate": "Win rate",
        "net_expectancy": "Net expectancy",
    }

    print("\n" + "=" * 74)
    print("  PHASE 2 COMPARISON (Phase 1 baseline vs Phase 2 risk engine)")
    print("=" * 74)
    print(f"{'Metric':<18} {'Phase 1':>16} {'Phase 2':>16} {'Delta (P2-P1)':>16}")
    print("-" * 74)

    for key in ("annual_return", "sharpe", "max_dd", "win_rate", "net_expectancy"):
        p1 = phase1.get(key)
        p2 = phase2.get(key)
        delta: Optional[float] = None if p1 is None or p2 is None else (p2 - p1)
        print(
            f"{labels[key]:<18} "
            f"{_format_metric(key, p1):>16} "
            f"{_format_metric(key, p2):>16} "
            f"{_format_metric(key, delta):>16}"
        )


def _save_phase_outputs(results: BacktestResults, output_dir: Path, prefix: str) -> None:
    """Save per-phase backtest artifacts to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results.trades_df.empty:
        results.trades_df.to_csv(output_dir / f"{prefix}_trades.csv", index=False)
    if not results.equity_df.empty:
        results.equity_df.to_csv(output_dir / f"{prefix}_equity.csv", index=False)
    if not results.rejected_df.empty:
        results.rejected_df.to_csv(output_dir / f"{prefix}_rejections.csv", index=False)


def _save_stop_loss_events(phase2_results: BacktestResults, output_dir: Path) -> Path:
    """Save stop-loss event log for Phase 2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase2_stop_loss_events.csv"

    if phase2_results.trades_df.empty:
        pd.DataFrame(columns=["event"]).to_csv(output_path, index=False)
        return output_path

    stop_events = phase2_results.trades_df[
        phase2_results.trades_df["exit_reason"] == "stop_loss"
    ].copy()
    stop_events.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="FPPE Trading System — Phase 2 comparison runner")
    parser.add_argument(
        "--signals-path",
        type=str,
        default=str(REPO_ROOT / "results" / "cached_signals_2024.csv"),
        help="Path to cached signal CSV (default: results/cached_signals_2024.csv)",
    )
    parser.add_argument(
        "--price-path",
        type=str,
        default=str(REPO_ROOT / "data" / "val_db.parquet"),
        help="Path to validation parquet with OHLC columns (default: data/val_db.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results"),
        help="Directory for exported CSV artifacts (default: results/)",
    )
    parser.add_argument(
        "--equal-weight-pct",
        type=float,
        default=0.05,
        help="Phase 1 baseline equal-weight size as decimal fraction (default: 0.05)",
    )
    args = parser.parse_args()

    signals_path = Path(args.signals_path)
    price_path = Path(args.price_path)
    output_dir = Path(args.output_dir)

    print("\n" + "=" * 74)
    print("  FPPE TRADING SYSTEM — PHASE 2 COMPARISON RUNNER")
    print("=" * 74)

    if not signals_path.exists():
        raise RuntimeError(f"Signal cache file not found: {signals_path}")

    _require_phase2_compatible_engine()

    config = TradingConfig()

    print(f"\n  Loading cached signals: {signals_path}")
    signal_df = load_cached_signals(str(signals_path))
    print(
        f"  Signals loaded: {len(signal_df):,} rows | "
        f"{signal_df['ticker'].nunique()} tickers | "
        f"{signal_df['date'].min().date()} to {signal_df['date'].max().date()}"
    )

    print(f"  Loading price data: {price_path}")
    price_df = _load_price_df(price_path)
    print(
        f"  Price rows: {len(price_df):,} | "
        f"{price_df['Ticker'].nunique()} tickers | "
        f"{price_df['Date'].min()} to {price_df['Date'].max()}"
    )

    engine = BacktestEngine(config)

    print("\n  Running Phase 1 baseline (use_risk_engine=False)...")
    phase1_results = engine.run(
        signal_df=signal_df,
        price_df=price_df,
        equal_weight_pct=args.equal_weight_pct,
        use_risk_engine=False,
    )

    print("\n  Running Phase 2 risk engine (use_risk_engine=True)...")
    phase2_results = engine.run(
        signal_df=signal_df,
        price_df=price_df,
        equal_weight_pct=args.equal_weight_pct,
        use_risk_engine=True,
    )

    phase1_metrics = _compute_metrics(phase1_results)
    phase2_metrics = _compute_metrics(phase2_results)
    _print_comparison(phase1_metrics, phase2_metrics)

    _save_phase_outputs(phase1_results, output_dir, "phase1")
    _save_phase_outputs(phase2_results, output_dir, "phase2")
    stop_events_path = _save_stop_loss_events(phase2_results, output_dir)

    print("\n  Exported artifacts:")
    print(f"    - {output_dir / 'phase1_trades.csv'}")
    print(f"    - {output_dir / 'phase1_equity.csv'}")
    print(f"    - {output_dir / 'phase1_rejections.csv'}")
    print(f"    - {output_dir / 'phase2_trades.csv'}")
    print(f"    - {output_dir / 'phase2_equity.csv'}")
    print(f"    - {output_dir / 'phase2_rejections.csv'}")
    print(f"    - {stop_events_path}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
