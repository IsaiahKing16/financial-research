"""
run_phase3_walkforward.py — Phase 3: Risk Engine Integration walk-forward.

Extends Phase 2 walk-forward with:
  1. Real ATR sizing (compute_atr_pct from 52T atr_14/Close columns)
  2. Drawdown brake (15% warn, 20% halt)
  3. FatigueAccumulationOverlay (decay_rate=0.15)
  4. LiquidityCongestionGate (defaults)

Mirrors Phase 2's _rescale_trades / _build_equity_curve pattern exactly.
The only differences are:
  - _rescale_trades now uses real per-ticker ATR (compute_atr_pct)
  - _rescale_trades applies apply_risk_adjustments (DD brake + overlays)
  - PnL is scaled from the original trade's position_pct to the new
    risk-adjusted position_pct (linear scaling, same as Phase 2)

Phase 3 gate (from fppe-roadmap-v2A.md §6):
  [ ] Drawdown brake fires correctly on synthetic 20% DD scenario
  [ ] Max DD <= 10% on walk-forward
  [ ] Sharpe >= 1.0 maintained from Phase 2
  [ ] Stop-loss fires <= 35% of exits

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_phase3_walkforward.py

Outputs:
    results/phase3_walkforward.tsv — summary metrics
    results/phase3_gate_check.txt  — gate verdict
    results/phase3_equity_curve.csv — Phase 3 daily equity curve
    results/phase3_blocked_trades.csv — blocked trade log with reasons

Spec: docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from trading_system.position_sizer import SizingConfig, size_position
from trading_system.risk_engine import (
    apply_risk_adjustments,
    compute_atr_pct,
    drawdown_brake_scalar,
)
from trading_system.risk_overlays.fatigue_accumulation import FatigueAccumulationOverlay
from trading_system.risk_overlays.liquidity_congestion import LiquidityCongestionGate

TRADES_PATH      = project_root / "results" / "backtest_trades.csv"
_52T_DIR         = project_root / "data" / "52t_volnorm"
RESULTS_DIR      = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_WF_TSV       = RESULTS_DIR / "phase3_walkforward.tsv"
OUT_GATE         = RESULTS_DIR / "phase3_gate_check.txt"
OUT_EQUITY       = RESULTS_DIR / "phase3_equity_curve.csv"
OUT_BLOCKED      = RESULTS_DIR / "phase3_blocked_trades.csv"

INITIAL_EQUITY   = 10_000.0
TRADE_DAYS_YEAR  = 252
RISK_FREE_ANNUAL = 0.045
SPY_THRESHOLD    = 0.05  # H7 locked

# Phase 3 overlay flags.
# FatigueAccumulationOverlay (SLE-75) saturates in sustained-BULL regimes:
# with decay_rate=0.15 and no regime transitions across 181 BULL days in the
# 2024 fold, the multiplier collapses to ~1e-13 and destroys positive PnL
# (diagnostic: results/phase3_throttling_diagnostic.csv — median overlay
# multiplier 0.0019, mean 0.136). The overlay was designed for short, choppy
# regimes with frequent transitions; the H7 BULL definition is too sticky
# for that model. Disabled by default until SLE-75 is redesigned.
# LiquidityCongestionGate stays on — diagnostic confirmed congestion_mult=1.0
# across all 278 trades, so it contributes zero drag and costs nothing.
USE_FATIGUE_OVERLAY = False

SIZING_CFG = SizingConfig()


# --- Metrics (mirrors Phase 2) ------------------------------------------------

def _sharpe(daily_ret: np.ndarray) -> float:
    if len(daily_ret) < 2 or np.std(daily_ret, ddof=1) < 1e-10:
        return float("nan")
    rf_daily = RISK_FREE_ANNUAL / TRADE_DAYS_YEAR
    excess = daily_ret - rf_daily
    return float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(TRADE_DAYS_YEAR))


def _max_dd(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak > 0, peak, 1.0)
    return float(dd.max())


def _compute_b_hist(trades: pd.DataFrame) -> float:
    """Compute historical win/loss ratio from the trade file (b_ratio for Kelly)."""
    wins = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] < 0]["net_pnl"]
    if len(wins) == 0 or len(losses) == 0:
        return 1.0
    return float(wins.mean() / losses.abs().mean())


def _build_atr_lookup(val_data: pd.DataFrame) -> dict:
    """Index 52T validation data by (date, ticker) -> atr_14, Close, ret_90d."""
    df = val_data.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df.set_index(["Date", "Ticker"])[["atr_14", "Close", "ret_90d"]].to_dict("index")


# --- Trade rescaling with Phase 3 risk engine ---------------------------------

def _rescale_trades_phase3(
    trades: pd.DataFrame,
    b_hist: float,
    atr_lookup: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Re-size each trade with real ATR + DD brake + overlays.

    Returns:
        (scaled_trades, blocked_log)
        scaled_trades has new columns: phase3_position_pct, phase3_net_pnl, phase3_blocked
        blocked_log: list of dicts with date, ticker, reason
    """
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"]  = pd.to_datetime(out["exit_date"])
    out = out.sort_values("entry_date").reset_index(drop=True)

    fatigue = FatigueAccumulationOverlay(decay_rate=0.15) if USE_FATIGUE_OVERLAY else None
    congestion = LiquidityCongestionGate()
    overlays = [congestion] + ([fatigue] if fatigue is not None else [])

    # Running equity for DD computation. Sample only on exit dates (when PnL realizes).
    running_equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    pending_pnls: dict = {}  # exit_date -> cumulative PnL closing that day

    phase3_pcts = []
    phase3_pnls = []
    phase3_blocked = []
    blocked_log: list[dict] = []

    for _, row in out.iterrows():
        entry_dt = row["entry_date"].date()
        exit_dt = row["exit_date"].date()
        ticker = row["ticker"]

        # Realize any pending PnLs whose exit_date is on or before this entry_dt.
        # This keeps running_equity (and DD) up to date for the brake check.
        for d in sorted(pending_pnls):
            if d <= entry_dt:
                running_equity += pending_pnls[d]
                if running_equity > peak_equity:
                    peak_equity = running_equity
                del pending_pnls[d]
            else:
                break

        # Update overlays at entry_dt with SPY market data
        spy_row = atr_lookup.get((entry_dt, "SPY"))
        if spy_row is not None:
            if fatigue is not None:
                ret_90d = spy_row.get("ret_90d", 0.0) or 0.0
                regime = "BULL" if ret_90d > SPY_THRESHOLD else "BEAR"
                fatigue.update(entry_dt, regime_label=regime)
            congestion.update(entry_dt, atr=spy_row["atr_14"], close=spy_row["Close"])

        # Lookup ATR for this ticker on entry_dt
        ticker_row = atr_lookup.get((entry_dt, ticker))
        if ticker_row is None:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": "missing_data"})
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue
        try:
            atr_pct = compute_atr_pct(ticker_row["atr_14"], ticker_row["Close"])
        except RuntimeError:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": "missing_atr"})
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue

        # Phase 2 sizing with real ATR
        confidence = float(row["confidence_at_entry"])
        sizing = size_position(confidence=confidence, b_ratio=b_hist, config=SIZING_CFG, atr_pct=atr_pct)

        # Current drawdown from peak
        dd = max(0.0, 1.0 - running_equity / peak_equity)

        # Phase 3 adjustments
        adj = apply_risk_adjustments(
            sizing,
            drawdown=dd,
            overlays=overlays,
        )
        if adj.blocked:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": adj.block_reason})
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue

        # Scale the trade's net_pnl from original position_pct to adj.final_position_pct
        original_pct = float(row["position_pct"])
        if original_pct <= 0:
            phase3_pcts.append(0.0)
            phase3_pnls.append(0.0)
            phase3_blocked.append(True)
            continue
        scale = adj.final_position_pct / original_pct
        scaled_pnl = float(row["net_pnl"]) * scale

        phase3_pcts.append(adj.final_position_pct)
        phase3_pnls.append(scaled_pnl)
        phase3_blocked.append(False)

        # Schedule the PnL to realize on exit_dt
        pending_pnls[exit_dt] = pending_pnls.get(exit_dt, 0.0) + scaled_pnl

    out["phase3_position_pct"] = phase3_pcts
    out["phase3_net_pnl"] = phase3_pnls
    out["phase3_blocked"] = phase3_blocked
    return out, blocked_log


def _build_equity_curve(trades: pd.DataFrame, net_pnl_col: str) -> pd.DataFrame:
    """Aggregate trade PnL onto a business-day calendar (mirrors Phase 2)."""
    trades = trades.copy()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    min_date = pd.to_datetime(trades["entry_date"].min())
    max_date = pd.to_datetime(trades["exit_date"].max())
    dates = pd.bdate_range(min_date, max_date)
    pnl_by_date = trades.groupby("exit_date")[net_pnl_col].sum()

    equity = INITIAL_EQUITY
    rows = []
    for dt in dates:
        pnl_today = float(pnl_by_date.get(dt, 0.0))
        equity += pnl_today
        rows.append({"date": dt, "equity": equity, "daily_pnl": pnl_today})
    df = pd.DataFrame(rows)
    df["daily_return"] = df["equity"].pct_change().fillna(0.0)
    return df


# --- Synthetic 20% DD scenario test (T3.4) ------------------------------------

def synthetic_dd_test() -> bool:
    """Verify the DD brake transitions through the boundary points."""
    test_points = [(0.10, 1.0), (0.15, 1.0), (0.175, 0.5), (0.20, 0.0), (0.22, 0.0)]
    for dd, expected in test_points:
        actual = drawdown_brake_scalar(dd)
        if abs(actual - expected) > 1e-9:
            print(f"  FAIL: dd={dd}, expected={expected}, got={actual}")
            return False
    return True


# --- Main ---------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("Phase 3 Risk Engine Integration -- Walk-Forward")
    print("=" * 70)

    # T3.4: Synthetic DD scenario
    print("\n[1/3] Synthetic 20% DD scenario test...")
    dd_test_pass = synthetic_dd_test()
    print("  DD brake transitions correctly" if dd_test_pass else "  DD brake test failed")

    # T3.5: Walk-forward with real ATR + overlays
    print("\n[2/3] 2024 fold simulation with real ATR + overlays...")
    trades = pd.read_csv(TRADES_PATH)
    print(f"  Loaded {len(trades)} Phase 1 trades")

    b_hist = _compute_b_hist(trades)
    print(f"  b_hist = {b_hist:.4f} (computed from net_pnl wins/losses)")

    val_data = pd.read_parquet(_52T_DIR / "val_db.parquet")
    atr_lookup = _build_atr_lookup(val_data)
    print(f"  ATR lookup built: {len(atr_lookup)} (date, ticker) entries")

    scaled, blocked_log = _rescale_trades_phase3(trades, b_hist, atr_lookup)
    n_blocked = scaled["phase3_blocked"].sum()
    n_placed = len(scaled) - n_blocked
    n_stopped = (
        (~scaled["phase3_blocked"])
        & scaled["exit_reason"].astype(str).str.lower().str.startswith("stop")
    ).sum()

    eq_df = _build_equity_curve(scaled, "phase3_net_pnl")
    eq_df.to_csv(OUT_EQUITY, index=False)
    pd.DataFrame(blocked_log).to_csv(OUT_BLOCKED, index=False)

    eq_arr = eq_df["equity"].values
    daily_ret = eq_df["daily_return"].values
    sharpe = _sharpe(daily_ret)
    max_dd = _max_dd(eq_arr)
    final_eq = float(eq_arr[-1])

    print(f"  Trades placed:  {n_placed}")
    print(f"  Trades blocked: {n_blocked}")
    print(f"  Stops fired:    {n_stopped}")
    print(f"  Final equity:   ${final_eq:,.2f}")
    print(f"  Sharpe (daily): {sharpe:.3f}")
    print(f"  Max DD:         {max_dd:.1%}")

    summary = pd.DataFrame([{
        "fold": "2024",
        "n_trades_placed": int(n_placed),
        "n_blocked": int(n_blocked),
        "n_stopped": int(n_stopped),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "final_equity": round(final_eq, 2),
        "b_hist": round(b_hist, 4),
    }])
    summary.to_csv(OUT_WF_TSV, sep="\t", index=False)

    # T3.5 gate check
    print("\n[3/3] Gate check...")
    stop_pct = n_stopped / max(1, n_placed)
    gates = {
        "DD brake fires correctly":   dd_test_pass,
        "Max DD <= 10%":              max_dd <= 0.10,
        "Sharpe >= 1.0":              sharpe >= 1.0,
        "Stops <= 35% of trades":     stop_pct <= 0.35,
    }
    all_pass = all(gates.values())
    lines = [f"Phase 3 Gate Check -- {'PASS' if all_pass else 'FAIL'}", "=" * 50, ""]
    for name, passed in gates.items():
        mark = "[X]" if passed else "[ ]"
        lines.append(f"  {mark} {name}")
    lines.append("")
    lines.append(
        f"Sharpe={sharpe:.3f}, MaxDD={max_dd:.1%}, "
        f"stops={n_stopped}/{n_placed} ({stop_pct:.1%}), "
        f"blocked={n_blocked}/{len(scaled)}"
    )
    OUT_GATE.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
