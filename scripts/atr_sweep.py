"""
atr_sweep.py — ATR stop-loss multiple sweep: 3.0× to 4.0×.

Phase 2 showed Sharpe regression (1.82 → 1.16) with stop_loss_atr_multiple=2.0
(38% premature exit rate on 14-day holds). This script sweeps [3.0, 3.25, 3.5,
3.75, 4.0] and logs Sharpe, MaxDD, and stop-event count to TSV for provenance.

Prerequisites (run once to generate):
    python -m trading_system.run_phase1 --use-cached-signals
    # → produces results/cached_signals_2024.csv

    python -m pattern_engine.data  (or any script that builds val_db.parquet)
    # → produces data/processed/val_db.parquet

Usage:
    python scripts/atr_sweep.py

Decision criterion: choose ATR multiple that maximises Sharpe without MaxDD > 8%.
"""

import csv
import dataclasses
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from trading_system.backtest_engine import BacktestEngine
from trading_system.config import RiskConfig, TradingConfig
from trading_system.signal_adapter import load_cached_signals

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
SIGNALS_PATH = REPO_ROOT / "results" / "cached_signals_2024.csv"
PRICE_PATH   = REPO_ROOT / "data" / "processed" / "val_db.parquet"

if not SIGNALS_PATH.exists():
    print(f"ERROR: {SIGNALS_PATH} not found.")
    print("Run: python -m trading_system.run_phase1 --use-cached-signals")
    sys.exit(1)

if not PRICE_PATH.exists():
    print(f"ERROR: {PRICE_PATH} not found.")
    print("Run the pattern_engine data pipeline to generate val_db.parquet.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading cached signals...")
base_config = TradingConfig()
signal_df = load_cached_signals(str(SIGNALS_PATH))

# Re-label signals at configured thresholds
if "confidence" in signal_df.columns and "n_matches" in signal_df.columns:
    t  = base_config.signals.confidence_threshold
    mm = base_config.signals.min_matches
    ma = base_config.signals.min_agreement
    signal_df = signal_df.copy()
    signal_df["agreement"] = (signal_df["confidence"] - 0.5).abs() * 2
    buy_mask  = (signal_df["confidence"] >= t) & (signal_df["n_matches"] >= mm) & (signal_df["agreement"] >= ma)
    sell_mask = (signal_df["confidence"] <= (1.0 - t)) & (signal_df["n_matches"] >= mm) & (signal_df["agreement"] >= ma)
    signal_df["signal"] = "HOLD"
    signal_df.loc[buy_mask,  "signal"] = "BUY"
    signal_df.loc[sell_mask, "signal"] = "SELL"
    print(f"  Re-labeled: {buy_mask.sum()} BUY, {sell_mask.sum()} SELL")

print("Loading price data...")
val_db = pd.read_parquet(PRICE_PATH)
price_df = val_db[["Date", "Ticker", "Open", "High", "Low", "Close"]].copy()
print(f"  Price data: {len(price_df):,} rows, {price_df['Ticker'].nunique()} tickers")

# ---------------------------------------------------------------------------
# ATR sweep
# ---------------------------------------------------------------------------
ATR_MULTIPLES = [3.0, 3.25, 3.5, 3.75, 4.0]
BASELINE = {"atr": 2.0, "sharpe": 1.16, "max_dd": 0.067, "stops": 73, "trades": 191}

results_dir = REPO_ROOT / "results"
results_dir.mkdir(exist_ok=True)
tsv_path = results_dir / "atr_sweep_results.tsv"

print(f"\n{'=' * 65}")
print(f"  ATR STOP SWEEP — {len(ATR_MULTIPLES)} values: {ATR_MULTIPLES}")
print(f"  Phase 2 baseline: Sharpe={BASELINE['sharpe']:.2f}, MaxDD={BASELINE['max_dd']:.1%}, "
      f"stops={BASELINE['stops']}/{BASELINE['trades']} trades")
print(f"{'=' * 65}")

sweep_rows = []

for atr_mult in ATR_MULTIPLES:
    print(f"\n  ATR={atr_mult}× ...")
    new_risk   = dataclasses.replace(base_config.risk, stop_loss_atr_multiple=atr_mult)
    new_config = dataclasses.replace(base_config, risk=new_risk)

    engine  = BacktestEngine(config=new_config, use_risk_engine=True)
    results = engine.run(signal_df, price_df)

    sharpe    = results.sharpe_ratio()
    max_dd    = results.max_drawdown()
    trades    = results.total_trades()
    stops     = len(results.stop_loss_events)
    win_rate  = results.win_rate()

    gate_ok = (max_dd is not None) and (max_dd < 0.08)
    print(f"    Sharpe={sharpe:.2f}  MaxDD={max_dd:.1%}  stops={stops}/{trades}  "
          f"MaxDD<8%={gate_ok}")

    row = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "atr_multiple":  atr_mult,
        "sharpe":        round(float(sharpe), 4) if sharpe is not None else None,
        "max_dd":        round(float(max_dd), 4) if max_dd is not None else None,
        "trades":        trades,
        "stop_events":   stops,
        "win_rate":      round(float(win_rate), 4) if win_rate is not None else None,
        "max_dd_gate":   gate_ok,
        "notes":         f"stop_loss_atr_multiple={atr_mult}",
    }
    sweep_rows.append(row)

# ---------------------------------------------------------------------------
# Write TSV (provenance log)
# ---------------------------------------------------------------------------
write_header = not tsv_path.exists()
with open(tsv_path, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()), delimiter="\t")
    if write_header:
        writer.writeheader()
    writer.writerows(sweep_rows)

print(f"\n  Provenance logged → {tsv_path}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'=' * 65}")
print(f"  {'ATR×':>5}  {'Sharpe':>8}  {'MaxDD':>7}  {'Stops':>6}  {'MaxDD<8%':>9}")
print(f"  {'-' * 55}")

# Baseline row
print(f"  {BASELINE['atr']:>5.2f}  {BASELINE['sharpe']:>8.2f}  {BASELINE['max_dd']:>7.1%}  "
      f"  {BASELINE['stops']:>4}  {'YES':>9}  ← Phase 2 baseline")

best_row = None
best_sharpe = float("-inf")

for row in sweep_rows:
    sharpe = row["sharpe"]
    max_dd = row["max_dd"]
    stops  = row["stop_events"]
    gate   = "YES" if row["max_dd_gate"] else " NO"
    print(f"  {row['atr_multiple']:>5.2f}  {sharpe:>8.2f}  {max_dd:>7.1%}  "
          f"  {stops:>4}  {gate:>9}")
    if row["max_dd_gate"] and (sharpe or 0) > best_sharpe:
        best_sharpe = sharpe
        best_row = row

print(f"{'=' * 65}")

if best_row:
    print(f"\n  WINNER: ATR={best_row['atr_multiple']}×  "
          f"Sharpe={best_row['sharpe']:.2f}  MaxDD={best_row['max_dd']:.1%}")
    print(f"  Update CLAUDE.md locked setting: stop_loss_atr_multiple = {best_row['atr_multiple']}")
    print(f"  Provenance: {tsv_path} row ATR={best_row['atr_multiple']}")
else:
    print("\n  No ATR multiple passed MaxDD < 8% gate. Widen sweep or revisit MaxDD threshold.")
