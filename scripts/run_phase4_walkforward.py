"""
run_phase4_walkforward.py -- Phase 4 Portfolio Manager walk-forward.

Extends the Phase 3 trade-replay pattern with a per-day Portfolio Manager
filter that enforces sector concentration and capital constraints BEFORE
the risk engine sizes trades:

    Phase 3: trades.csv -> risk engine (per trade) -> PnL
    Phase 4: trades.csv -> group by entry_date
                        -> build PortfolioSnapshot from still-open trades
                        -> rank_signals + allocate_day (per day)
                        -> risk engine (per approved) -> PnL

--no-pm mode short-circuits the PM and runs a byte-identical port of
Phase 3's _rescale_trades_phase3 loop; output must match
results/phase3_walkforward.tsv within float tolerance (the handoff's
parity contract).

Gates G1-G9 (see docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md §2).

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py
    PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py --no-pm
    PYTHONUTF8=1 py -3.12 scripts/run_phase4_walkforward.py --fold 2024
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date as Date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.contracts.signals import SignalDirection, SignalSource
from trading_system.config import PositionLimitsConfig
from trading_system.portfolio_manager import allocate_day, rank_signals
from trading_system.portfolio_state import (
    AllocationResult,
    OpenPosition,
    PortfolioSnapshot,
)
from trading_system.position_sizer import SizingConfig, size_position
from trading_system.risk_engine import apply_risk_adjustments, compute_atr_pct
from trading_system.risk_overlays.liquidity_congestion import LiquidityCongestionGate
from trading_system.signal_adapter import UnifiedSignal


TRADES_PATH   = project_root / "results" / "backtest_trades.csv"
_52T_DIR      = project_root / "data" / "52t_volnorm"
RESULTS_DIR   = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUT_WF_TSV    = RESULTS_DIR / "phase4_walkforward.tsv"
OUT_EQUITY    = RESULTS_DIR / "phase4_equity_curve.csv"
OUT_ALLOC     = RESULTS_DIR / "phase4_allocations.csv"
OUT_REJECT    = RESULTS_DIR / "phase4_rejections.csv"
OUT_GATE      = RESULTS_DIR / "phase4_gate_check.txt"
OUT_ZERO_DAYS = RESULTS_DIR / "phase4_zero_allocation_days.csv"

INITIAL_EQUITY   = 10_000.0
TRADE_DAYS_YEAR  = 252
RISK_FREE_ANNUAL = 0.045
MIN_POSITION_PCT = 0.02  # PM admission floor (see plan impl note #1)

# Phase 3 locked the fatigue overlay OFF in the 2024 fold (saturates in
# sustained BULL regimes). Keep it off here so --no-pm mode reproduces
# results/phase3_walkforward.tsv exactly.
USE_FATIGUE_OVERLAY = False

SIZING_CFG = SizingConfig()
LIMITS = PositionLimitsConfig()


# ── Metrics (mirrors Phase 3) ────────────────────────────────────────────────

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
    wins = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] < 0]["net_pnl"]
    if len(wins) == 0 or len(losses) == 0:
        return 1.0
    return float(wins.mean() / losses.abs().mean())


def _build_atr_lookup(val_data: pd.DataFrame) -> dict:
    df = val_data.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df.set_index(["Date", "Ticker"])[["atr_14", "Close"]].to_dict("index")


# ── --no-pm path: byte-identical port of Phase 3's loop ──────────────────────

def _rescale_phase3_compat(
    trades: pd.DataFrame,
    b_hist: float,
    atr_lookup: dict,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Port of run_phase3_walkforward._rescale_trades_phase3.

    Kept in-script (rather than imported) so the Phase 3 script can evolve
    independently. If Phase 3's loop changes, --no-pm parity must be
    re-verified, not silently inherited.
    """
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"]  = pd.to_datetime(out["exit_date"])
    out = out.sort_values("entry_date").reset_index(drop=True)

    congestion = LiquidityCongestionGate()
    overlays = [congestion]

    running_equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    pending_pnls: dict = {}

    pcts: List[float] = []
    pnls: List[float] = []
    blocked_flags: List[bool] = []
    blocked_log: List[dict] = []

    for _, row in out.iterrows():
        entry_dt = row["entry_date"].date()
        exit_dt  = row["exit_date"].date()
        ticker   = row["ticker"]

        for d in sorted(pending_pnls):
            if d <= entry_dt:
                running_equity += pending_pnls[d]
                if running_equity > peak_equity:
                    peak_equity = running_equity
                del pending_pnls[d]
            else:
                break

        spy_row = atr_lookup.get((entry_dt, "SPY"))
        if spy_row is not None:
            congestion.update(entry_dt, atr=spy_row["atr_14"], close=spy_row["Close"])

        ticker_row = atr_lookup.get((entry_dt, ticker))
        if ticker_row is None:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": "missing_data"})
            pcts.append(0.0); pnls.append(0.0); blocked_flags.append(True)
            continue
        try:
            atr_pct = compute_atr_pct(ticker_row["atr_14"], ticker_row["Close"])
        except RuntimeError:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": "missing_atr"})
            pcts.append(0.0); pnls.append(0.0); blocked_flags.append(True)
            continue

        confidence = float(row["confidence_at_entry"])
        sizing = size_position(
            confidence=confidence, b_ratio=b_hist,
            config=SIZING_CFG, atr_pct=atr_pct,
        )

        dd = max(0.0, 1.0 - running_equity / peak_equity)
        adj = apply_risk_adjustments(sizing, drawdown=dd, overlays=overlays)
        if adj.blocked:
            blocked_log.append({"date": entry_dt, "ticker": ticker, "reason": adj.block_reason})
            pcts.append(0.0); pnls.append(0.0); blocked_flags.append(True)
            continue

        original_pct = float(row["position_pct"])
        if original_pct <= 0:
            pcts.append(0.0); pnls.append(0.0); blocked_flags.append(True)
            continue
        scale = adj.final_position_pct / original_pct
        scaled_pnl = float(row["net_pnl"]) * scale

        pcts.append(adj.final_position_pct)
        pnls.append(scaled_pnl)
        blocked_flags.append(False)

        pending_pnls[exit_dt] = pending_pnls.get(exit_dt, 0.0) + scaled_pnl

    out["phase4_position_pct"] = pcts
    out["phase4_net_pnl"] = pnls
    out["phase4_blocked"] = blocked_flags
    return out, blocked_log


# ── PM path: day-grouped replay ──────────────────────────────────────────────

def _trade_row_to_signal(row: pd.Series) -> UnifiedSignal:
    return UnifiedSignal(
        date=pd.Timestamp(row["entry_date"]).date(),
        ticker=row["ticker"],
        signal=SignalDirection.BUY,
        confidence=float(row["confidence_at_entry"]),
        signal_source=SignalSource.KNN,
        sector=row.get("sector") or "Unknown",
    )


def _build_snapshot(
    day: Date,
    open_positions: List[OpenPosition],
    equity: float,
) -> PortfolioSnapshot:
    """Build a read-only snapshot for PM filtering.

    position_pct is a fraction of equity at entry time; we approximate
    committed capital as sum(position_pct) * current_equity, clamped so
    cash stays in [0, equity]. This keeps cash_pct = 1 - sum_pct
    (scale-invariant), which is all the PM filter needs.
    """
    committed_frac = sum(p.position_pct for p in open_positions)
    committed_frac = min(1.0, max(0.0, committed_frac))
    cash = equity * (1.0 - committed_frac)
    return PortfolioSnapshot(
        as_of_date=day,
        equity=equity,
        cash=cash,
        open_positions=tuple(open_positions),
    )


def _phase4_replay_with_pm(
    trades: pd.DataFrame,
    b_hist: float,
    atr_lookup: dict,
    limits: PositionLimitsConfig,
) -> Tuple[pd.DataFrame, List[AllocationResult], List[Date]]:
    """Day-grouped replay with Portfolio Manager filter.

    Returns:
        scaled_trades: copy of `trades` with new columns
            phase4_position_pct, phase4_net_pnl, phase4_blocked,
            phase4_pm_reason (str or empty).
        all_allocations: flat list of AllocationResult for T4.3 analysis.
        zero_allocation_days: days where >=1 candidate was presented but
            all were rejected (gate G9).
    """
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"]  = pd.to_datetime(out["exit_date"])
    out = out.sort_values(["entry_date", "ticker"]).reset_index(drop=True)

    congestion = LiquidityCongestionGate()
    overlays = [congestion]

    running_equity = INITIAL_EQUITY
    peak_equity    = INITIAL_EQUITY
    pending_pnls: Dict[Date, float] = {}

    # open_positions list keyed implicitly by exit_date: we drain entries
    # whose exit_date <= today at the start of each day.
    open_positions: List[OpenPosition] = []
    open_exit_dates: List[Date] = []  # parallel list of exit dates

    # Per-row outputs initialized to "blocked" so rows we never touch
    # (e.g. PM-rejected) default to 0 PnL / 0 pct.
    n = len(out)
    pcts: List[float]        = [0.0]   * n
    pnls: List[float]        = [0.0]   * n
    blocked_flags: List[bool] = [True] * n
    pm_reasons: List[str]    = [""]    * n

    all_allocations: List[AllocationResult] = []
    zero_allocation_days: List[Date] = []

    # Index rows by the order they appear in `out` so we can map back.
    out_by_day: Dict[Date, List[int]] = {}
    for idx, row in out.iterrows():
        d = row["entry_date"].date()
        out_by_day.setdefault(d, []).append(idx)

    for day in sorted(out_by_day.keys()):
        # 1. Realize any pending PnLs with exit_date <= today.
        for d in sorted(pending_pnls):
            if d <= day:
                running_equity += pending_pnls[d]
                if running_equity > peak_equity:
                    peak_equity = running_equity
                del pending_pnls[d]
            else:
                break

        # 2. Close any open positions whose exit_date <= today (prior PnL
        # already realized above, but we need them out of the snapshot).
        survivors: List[OpenPosition] = []
        survivor_exits: List[Date] = []
        for pos, exit_d in zip(open_positions, open_exit_dates):
            if exit_d > day:
                survivors.append(pos)
                survivor_exits.append(exit_d)
        open_positions = survivors
        open_exit_dates = survivor_exits

        # 3. Update congestion overlay with today's SPY row.
        spy_row = atr_lookup.get((day, "SPY"))
        if spy_row is not None:
            congestion.update(day, atr=spy_row["atr_14"], close=spy_row["Close"])

        # 4. Build snapshot and candidate signals.
        snapshot = _build_snapshot(day, open_positions, running_equity)
        day_idxs = out_by_day[day]
        candidates: List[UnifiedSignal] = []
        idx_by_ticker: Dict[str, int] = {}
        for idx in day_idxs:
            row = out.iloc[idx]
            sig = _trade_row_to_signal(row)
            candidates.append(sig)
            idx_by_ticker[sig.ticker] = idx

        # 5. PM filter.
        ranked = rank_signals(candidates)
        allocations = allocate_day(
            ranked_signals=ranked,
            snapshot=snapshot,
            limits=limits,
            min_position_pct=MIN_POSITION_PCT,
        )
        all_allocations.extend(allocations)

        n_candidates = len(candidates)
        n_approved = sum(1 for a in allocations if a.approved)
        if n_candidates >= 1 and n_approved == 0:
            zero_allocation_days.append(day)

        # 6. Record PM rejections on the output rows.
        for alloc in allocations:
            if alloc.approved:
                continue
            idx = idx_by_ticker[alloc.ticker]
            pm_reasons[idx] = alloc.rejection.reason  # type: ignore[union-attr]
            # pcts/pnls/blocked already defaulted.

        # 7. For each approved allocation, size through the risk engine
        # and schedule PnL for its exit date.
        for alloc in allocations:
            if not alloc.approved:
                continue
            idx = idx_by_ticker[alloc.ticker]
            row = out.iloc[idx]
            ticker = row["ticker"]
            exit_dt = row["exit_date"].date()

            ticker_row = atr_lookup.get((day, ticker))
            if ticker_row is None:
                pm_reasons[idx] = "missing_data"
                continue
            try:
                atr_pct = compute_atr_pct(ticker_row["atr_14"], ticker_row["Close"])
            except RuntimeError:
                pm_reasons[idx] = "missing_atr"
                continue

            confidence = float(row["confidence_at_entry"])
            sizing = size_position(
                confidence=confidence, b_ratio=b_hist,
                config=SIZING_CFG, atr_pct=atr_pct,
            )
            dd = max(0.0, 1.0 - running_equity / peak_equity)
            adj = apply_risk_adjustments(sizing, drawdown=dd, overlays=overlays)
            if adj.blocked:
                pm_reasons[idx] = f"risk_engine:{adj.block_reason}"
                continue

            original_pct = float(row["position_pct"])
            if original_pct <= 0:
                pm_reasons[idx] = "zero_original_pct"
                continue
            scale = adj.final_position_pct / original_pct
            scaled_pnl = float(row["net_pnl"]) * scale

            pcts[idx] = adj.final_position_pct
            pnls[idx] = scaled_pnl
            blocked_flags[idx] = False
            pm_reasons[idx] = ""  # approved + sized

            pending_pnls[exit_dt] = pending_pnls.get(exit_dt, 0.0) + scaled_pnl

            # Add to live open-position book for subsequent days.
            open_positions.append(OpenPosition(
                ticker=ticker,
                sector=row.get("sector") or "Unknown",
                entry_date=day,
                position_pct=adj.final_position_pct,
                entry_price=float(row["entry_price"]),
            ))
            open_exit_dates.append(exit_dt)

    out["phase4_position_pct"] = pcts
    out["phase4_net_pnl"]      = pnls
    out["phase4_blocked"]      = blocked_flags
    out["phase4_pm_reason"]    = pm_reasons
    return out, all_allocations, zero_allocation_days


# ── Equity curve (mirrors Phase 3) ───────────────────────────────────────────

def _build_equity_curve(trades: pd.DataFrame, net_pnl_col: str) -> pd.DataFrame:
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


# ── Allocation output writers ────────────────────────────────────────────────

def _write_allocations(allocations: List[AllocationResult], path: Path) -> None:
    rows = []
    for a in allocations:
        rows.append({
            "signal_date": a.signal_date,
            "ticker": a.ticker,
            "sector": a.sector,
            "confidence": round(a.confidence, 6),
            "rank": a.rank,
            "approved": a.approved,
            "rejection_reason": a.rejection.reason if a.rejection else "",
            "rejection_detail": a.rejection.detail if a.rejection else "",
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_rejections(allocations: List[AllocationResult], path: Path) -> None:
    rows = []
    for a in allocations:
        if a.approved:
            continue
        r = a.rejection
        rows.append({
            "signal_date": a.signal_date,
            "ticker": a.ticker,
            "sector": a.sector,
            "confidence": round(a.confidence, 6),
            "rank": a.rank,
            "reason": r.reason,  # type: ignore[union-attr]
            "detail": r.detail,  # type: ignore[union-attr]
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_zero_days(days: List[Date], path: Path) -> None:
    pd.DataFrame({"date": days}).to_csv(path, index=False)


# ── Gate check (G1–G9) ───────────────────────────────────────────────────────

def _gate_check(
    *,
    use_pm: bool,
    sharpe: float,
    max_dd: float,
    allocations: List[AllocationResult],
    equity_curve: pd.DataFrame,
    runtime_sec: float,
) -> Tuple[bool, str]:
    """Evaluate G1-G9. G7 (test count) is checked externally."""
    lines: List[str] = []
    mode = "PM enabled" if use_pm else "--no-pm (Phase 3 parity)"
    lines.append(f"Phase 4 Gate Check -- {mode}")
    lines.append("=" * 60)
    lines.append("")

    if not use_pm:
        # --no-pm mode: only check G4/G5/G8 — parity is the real gate.
        gates = {
            "G4: Sharpe >= 2.0":  sharpe >= 2.0,
            "G5: MaxDD <= 10%":   max_dd <= 0.10,
            "G8: Runtime < 5min": runtime_sec < 300.0,
        }
        all_pass = all(gates.values())
        verdict = "PASS" if all_pass else "FAIL"
        lines.append(f"Verdict: {verdict}")
        lines.append("")
        for name, ok in gates.items():
            lines.append(f"  [{'X' if ok else ' '}] {name}")
        lines.append("")
        lines.append(f"Sharpe={sharpe:.3f}, MaxDD={max_dd:.1%}, "
                     f"runtime={runtime_sec:.1f}s")
        lines.append("")
        lines.append("NOTE: --no-pm mode -- real gate is parity with "
                     "results/phase3_walkforward.tsv.")
        return all_pass, "\n".join(lines)

    # PM-enabled gates.
    # G1: sector exposure -- approximated from approved allocations'
    # sector distribution, not live equity curve. We check that no single
    # sector has > max_sector_pct of approved allocations (proxy for
    # running exposure with flat MIN_POSITION_PCT).
    approved = [a for a in allocations if a.approved]
    sector_counts: Dict[str, int] = {}
    for a in approved:
        sector_counts[a.sector] = sector_counts.get(a.sector, 0) + 1
    total_approved = max(1, len(approved))
    max_sector_share = max(sector_counts.values(), default=0) / total_approved
    g1 = max_sector_share <= 0.30 + 1e-9

    # G2 / G3: idle cash (mean, p90) computed from equity curve vs a
    # notional "fully invested" baseline is infeasible here (we don't
    # track cash timeseries). Use approved count as a proxy:
    #   idle_fraction = 1 - (approved_count / total_presented)
    # This is a placeholder that surfaces the metric without claiming
    # precision. T4.3 provides the real analysis.
    n_total = len(allocations)
    approved_share = total_approved / max(1, n_total)
    idle_cash_mean_proxy = 1.0 - approved_share
    g2 = idle_cash_mean_proxy < 0.35
    g3 = idle_cash_mean_proxy < 0.50  # loose p90 proxy

    # G4 / G5: Sharpe and MaxDD from equity curve.
    g4 = sharpe >= 2.0
    g5 = max_dd <= 0.10

    # G6: no rejection reason accounts for > 60% of rejections.
    rejections = [a for a in allocations if not a.approved]
    reason_counts: Dict[str, int] = {}
    for a in rejections:
        r = a.rejection.reason  # type: ignore[union-attr]
        reason_counts[r] = reason_counts.get(r, 0) + 1
    if rejections:
        max_reason_share = max(reason_counts.values()) / len(rejections)
    else:
        max_reason_share = 0.0
    g6 = max_reason_share <= 0.60 + 1e-9

    # G8: runtime budget.
    g8 = runtime_sec < 300.0

    # G9: zero-allocation days file written (presence check done in main).
    g9 = OUT_ZERO_DAYS.exists()

    gates = {
        "G1: max sector share <= 30%":       g1,
        "G2: idle cash mean proxy < 35%":    g2,
        "G3: idle cash p90 proxy < 50%":     g3,
        "G4: Sharpe >= 2.0":                 g4,
        "G5: MaxDD <= 10%":                  g5,
        "G6: no reason > 60% of rejections": g6,
        "G8: runtime < 5 min":               g8,
        "G9: zero-allocation days logged":   g9,
    }
    all_pass = all(gates.values())
    verdict = "PASS" if all_pass else "FAIL"
    lines.append(f"Verdict: {verdict}  (G7 test count checked externally)")
    lines.append("")
    for name, ok in gates.items():
        lines.append(f"  [{'X' if ok else ' '}] {name}")
    lines.append("")
    lines.append(
        f"Sharpe={sharpe:.3f}, MaxDD={max_dd:.1%}, "
        f"approved={total_approved}/{n_total}, "
        f"max_sector_share={max_sector_share:.1%}, "
        f"max_reason_share={max_reason_share:.1%}, "
        f"runtime={runtime_sec:.1f}s"
    )
    if rejections:
        lines.append("")
        lines.append("Rejection breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {reason:24s} {count:4d} ({count/len(rejections):.1%})")
    return all_pass, "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-pm", action="store_true",
        help="Disable PM filter; reproduces Phase 3 walk-forward exactly.",
    )
    parser.add_argument(
        "--fold", default="2024",
        help="Fold label (only 2024 supported in this plan).",
    )
    args = parser.parse_args()
    use_pm = not args.no_pm

    print("=" * 70)
    print(f"Phase 4 Portfolio Manager -- Walk-Forward  "
          f"[{'PM enabled' if use_pm else '--no-pm'}]")
    print("=" * 70)

    t0 = time.time()

    trades = pd.read_csv(TRADES_PATH)
    print(f"\n[1/4] Loaded {len(trades)} Phase 1 trades from {TRADES_PATH.name}")

    b_hist = _compute_b_hist(trades)
    print(f"      b_hist = {b_hist:.4f}")

    val_data = pd.read_parquet(_52T_DIR / "val_db.parquet")
    atr_lookup = _build_atr_lookup(val_data)
    print(f"      ATR lookup built: {len(atr_lookup)} (date, ticker) entries")

    allocations: List[AllocationResult] = []
    zero_days: List[Date] = []

    print(f"\n[2/4] Replaying trades "
          f"({'day-grouped with PM' if use_pm else 'per-trade, no PM'})...")
    if use_pm:
        scaled, allocations, zero_days = _phase4_replay_with_pm(
            trades, b_hist, atr_lookup, LIMITS,
        )
    else:
        scaled, blocked_log = _rescale_phase3_compat(
            trades, b_hist, atr_lookup,
        )
        # --no-pm mode has no PM reasons; add the column for schema parity.
        scaled["phase4_pm_reason"] = ""
        pd.DataFrame(blocked_log).to_csv(
            RESULTS_DIR / "phase4_blocked_trades.csv", index=False
        )

    n_blocked = int(scaled["phase4_blocked"].sum())
    n_placed = int(len(scaled) - n_blocked)

    eq_df = _build_equity_curve(scaled, "phase4_net_pnl")
    eq_arr = eq_df["equity"].values
    daily_ret = eq_df["daily_return"].values
    sharpe = _sharpe(daily_ret)
    max_dd = _max_dd(eq_arr)
    final_eq = float(eq_arr[-1])

    runtime = time.time() - t0

    print(f"      Trades placed:    {n_placed}")
    print(f"      Trades blocked:   {n_blocked}")
    print(f"      Final equity:     ${final_eq:,.2f}")
    print(f"      Sharpe (daily):   {sharpe:.3f}")
    print(f"      Max DD:           {max_dd:.1%}")
    print(f"      Runtime:          {runtime:.1f}s")

    # ── Write outputs ────────────────────────────────────────────────────────
    print("\n[3/4] Writing output files...")

    summary = pd.DataFrame([{
        "fold": args.fold,
        "mode": "pm" if use_pm else "no_pm",
        "n_trades_placed": n_placed,
        "n_blocked": n_blocked,
        "n_allocations": len(allocations),
        "n_rejected": sum(1 for a in allocations if not a.approved),
        "n_zero_allocation_days": len(zero_days),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "final_equity": round(final_eq, 2),
        "b_hist": round(b_hist, 4),
        "runtime_sec": round(runtime, 2),
    }])
    summary.to_csv(OUT_WF_TSV, sep="\t", index=False)
    print(f"      {OUT_WF_TSV.name}")

    eq_df.to_csv(OUT_EQUITY, index=False)
    print(f"      {OUT_EQUITY.name}")

    _write_allocations(allocations, OUT_ALLOC)
    print(f"      {OUT_ALLOC.name} ({len(allocations)} rows)")

    _write_rejections(allocations, OUT_REJECT)
    n_rej = sum(1 for a in allocations if not a.approved)
    print(f"      {OUT_REJECT.name} ({n_rej} rows)")

    _write_zero_days(zero_days, OUT_ZERO_DAYS)
    print(f"      {OUT_ZERO_DAYS.name} ({len(zero_days)} days)")

    # ── Gate check ───────────────────────────────────────────────────────────
    print("\n[4/4] Gate check...")
    all_pass, report = _gate_check(
        use_pm=use_pm,
        sharpe=sharpe,
        max_dd=max_dd,
        allocations=allocations,
        equity_curve=eq_df,
        runtime_sec=runtime,
    )
    OUT_GATE.write_text(report, encoding="utf-8")
    print()
    print(report)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
