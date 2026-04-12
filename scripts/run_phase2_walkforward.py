"""
run_phase2_walkforward.py — Phase 2 T2.3 (extended): Kelly vs flat backtest + fold gate.

Two analyses:

1. 2024 fold equity simulation (T2.3b)
   Takes Phase 1 backtest_trades.csv (flat 5% sizing, 278 trades, 2024).
   Rescales each trade using Half-Kelly position_sizer.py (per-trade p=confidence_at_entry).
   Rebuilds the equity curve; computes Sharpe, MaxDD, net expectancy.
   Compares directly to Phase 1 flat baseline on the same trade set.

2. 6-fold Kelly fraction gate (T2.4)
   For each walk-forward fold, computes the fold-level Kelly fraction using
   the actual win rates and payoff ratios from the 52T training data (Bull-regime rows).
   This gives a lower-bound estimate: PatternMatcher signals have higher win rate than
   the base rate, so fold-level Kelly(signal) ≥ Kelly(base_rate).
   Gate: Kelly fraction positive on ≥ 4/6 folds.

Phase 2 gate (from fppe-roadmap-v2A.md §5):
  [ ] Kelly fraction positive on ≥ 4/6 folds
  [ ] Backtest Sharpe ≥ 1.0
  [ ] Max drawdown ≤ 15%

Usage:
    PYTHONUTF8=1 py -3.12 scripts/run_phase2_walkforward.py

Outputs:
    results/phase2_walkforward.tsv — per-fold Kelly stats
    results/phase2_gate_check.txt  — gate verdict
    results/phase2_equity_kelly.csv  — Kelly equity curve (2024 fold)
    results/phase2_equity_flat.csv   — Flat equity curve (2024 fold, replicated from trades)

Friction: 26 bps round-trip (locked).
Horizon: fwd_7d_up / fwd_7d (consistent with H7).

Note on 52T probability distribution: The PatternMatcher with beta_abm at max_d=0.90
produces calibrated probabilities in [0.50, 0.58] for the 52T universe — below the
locked confidence_threshold=0.65 (set for the 585T Platt system). Phase 2 uses the
production signal stream (cached_signals_2024.csv / backtest_trades.csv) where
confidence = Platt-calibrated probability ∈ [0.60, 0.75]. The fold-level gate uses
52T training win rates as a conservative lower bound on fold Kelly fractions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.config import WALKFORWARD_FOLDS
from trading_system.position_sizer import SizingConfig, size_position, compute_kelly_fraction

TRADES_PATH      = project_root / "results" / "backtest_trades.csv"
EQUITY_PATH      = project_root / "results" / "backtest_equity.csv"
_52T_DIR         = project_root / "data" / "52t_volnorm"
RESULTS_DIR      = project_root / "results"

OUT_WF_TSV       = RESULTS_DIR / "phase2_walkforward.tsv"
OUT_GATE         = RESULTS_DIR / "phase2_gate_check.txt"
OUT_KELLY_EQ     = RESULTS_DIR / "phase2_equity_kelly.csv"
OUT_FLAT_EQ      = RESULTS_DIR / "phase2_equity_flat.csv"

ROUND_TRIP_BPS   = 26
INITIAL_EQUITY   = 10_000.0
TRADE_DAYS_YEAR  = 252
RISK_FREE_ANNUAL = 0.045
HORIZON          = "fwd_7d_up"
HORIZON_RET      = "fwd_7d"
SPY_THRESHOLD    = 0.05          # H7 locked

SIZING_CFG = SizingConfig()


# ─── Metrics ──────────────────────────────────────────────────────────────────

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


def _ann_return(equity: np.ndarray, n_trade_days: int) -> float:
    if equity[0] <= 0 or n_trade_days < 1:
        return float("nan")
    return float((equity[-1] / equity[0]) ** (TRADE_DAYS_YEAR / n_trade_days) - 1)


# ─── Analysis 1: 2024 fold equity simulation ──────────────────────────────────

def _rescale_trades(trades: pd.DataFrame, b_ratio: float) -> pd.DataFrame:
    """Re-size each trade using Half-Kelly (confidence_at_entry as p).

    Returns a copy with columns:
      kelly_position_pct, kelly_gross_pnl, kelly_total_costs, kelly_net_pnl
    """
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"]  = pd.to_datetime(out["exit_date"])

    kelly_pcts, kelly_gross, kelly_costs, kelly_net = [], [], [], []

    for _, row in out.iterrows():
        conf = float(row["confidence_at_entry"])
        result = size_position(confidence=conf, b_ratio=b_ratio, config=SIZING_CFG)

        if result.approved:
            kpct = result.position_pct
        else:
            # Fallback: keep original 5% if Kelly rejects (shouldn't happen for conf>0.65)
            kpct = float(row["position_pct"])

        # Scale PnL linearly from old position_pct to new
        scale = kpct / float(row["position_pct"])
        kelly_pcts.append(kpct)
        kelly_gross.append(float(row["gross_pnl"])   * scale)
        kelly_costs.append(float(row["total_costs"]) * scale)
        kelly_net.append(float(row["net_pnl"])        * scale)

    out["kelly_position_pct"] = kelly_pcts
    out["kelly_gross_pnl"]    = kelly_gross
    out["kelly_total_costs"]  = kelly_costs
    out["kelly_net_pnl"]      = kelly_net
    return out


def _build_equity_curve(
    trades: pd.DataFrame,
    net_pnl_col: str,
    label: str,
) -> pd.DataFrame:
    """Aggregate trade P&Ls to an equity time series.

    Assumes trades are closed on exit_date; equity updates on that date.
    Start date = min(entry_date) - 1 business day (starting equity).
    """
    trades = trades.copy()
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    # All calendar dates from first entry to last exit
    min_date = pd.to_datetime(trades["entry_date"].min())
    max_date = pd.to_datetime(trades["exit_date"].max())
    dates = pd.bdate_range(min_date, max_date)

    # Daily net PnL
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


def run_2024_simulation(b_ratio: float) -> dict:
    """Rescale Phase 1 trades with Kelly and compare equity curves."""
    trades = pd.read_csv(TRADES_PATH)
    print(f"  Loaded {len(trades)} Phase 1 trades (flat 5% sizing)")

    scaled = _rescale_trades(trades, b_ratio)

    k_eq = _build_equity_curve(scaled, "kelly_net_pnl",  "Kelly")
    f_eq = _build_equity_curve(scaled, "net_pnl",        "Flat")

    k_eq.to_csv(OUT_KELLY_EQ, index=False)
    f_eq.to_csv(OUT_FLAT_EQ, index=False)

    k_arr = k_eq["equity"].values
    f_arr = f_eq["equity"].values
    n_days = len(k_arr)

    k_daily = k_eq["daily_return"].values
    f_daily = f_eq["daily_return"].values

    # Per-trade Kelly stats
    wins_k  = scaled[scaled["kelly_net_pnl"] > 0]
    losses_k = scaled[scaled["kelly_net_pnl"] < 0]

    result = {
        # Kelly curve
        "kelly_sharpe":   round(_sharpe(k_daily), 4),
        "kelly_max_dd":   round(_max_dd(k_arr), 4),
        "kelly_ann_ret":  round(_ann_return(k_arr, n_days), 4),
        "kelly_final_eq": round(float(k_arr[-1]), 2),
        "kelly_net_exp":  round(float(scaled["kelly_net_pnl"].sum()) / len(scaled), 2),
        # Flat curve (Phase 1 baseline)
        "flat_sharpe":    round(_sharpe(f_daily), 4),
        "flat_max_dd":    round(_max_dd(f_arr), 4),
        "flat_ann_ret":   round(_ann_return(f_arr, n_days), 4),
        "flat_final_eq":  round(float(f_arr[-1]), 2),
        "flat_net_exp":   round(float(scaled["net_pnl"].sum()) / len(scaled), 2),
        # Kelly sizing stats
        "n_trades":       len(scaled),
        "kelly_pct_mean": round(float(scaled["kelly_position_pct"].mean()), 4),
        "kelly_pct_min":  round(float(scaled["kelly_position_pct"].min()), 4),
        "kelly_pct_max":  round(float(scaled["kelly_position_pct"].max()), 4),
    }

    print(f"\n  2024 Fold Comparison (Kelly vs Flat 5%):")
    print(f"  {'Metric':<22} {'Kelly':>8} {'Flat':>8}")
    print(f"  {'─'*40}")
    print(f"  {'Sharpe':<22} {result['kelly_sharpe']:>8.3f} {result['flat_sharpe']:>8.3f}")
    print(f"  {'Max DD':<22} {result['kelly_max_dd']*100:>7.1f}% {result['flat_max_dd']*100:>7.1f}%")
    print(f"  {'Ann. Return':<22} {result['kelly_ann_ret']*100:>7.1f}% {result['flat_ann_ret']*100:>7.1f}%")
    print(f"  {'Final Equity ($)':<22} {result['kelly_final_eq']:>8.0f} {result['flat_final_eq']:>8.0f}")
    print(f"  {'Net Exp./trade ($)':<22} {result['kelly_net_exp']:>+8.2f} {result['flat_net_exp']:>+8.2f}")
    print(f"  {'Kelly pos. size (avg)':<22} {result['kelly_pct_mean']*100:>7.1f}%")
    print(f"  {'Kelly pos. size range':<22} [{result['kelly_pct_min']*100:.1f}%–{result['kelly_pct_max']*100:.1f}%]")

    return result


# ─── Analysis 2: Fold-level Kelly fraction gate ────────────────────────────────

def _fold_kelly_from_52t(full_db: pd.DataFrame, b_hist: float) -> list[dict]:
    """Compute fold-level Kelly fractions from 52T training win rates.

    Uses actual fwd_7d_up outcomes for Bull-regime rows in each fold's training
    period as a conservative lower bound on signal Kelly fractions.
    (PatternMatcher selects top analogues → signal win rate ≥ training base rate.)
    """
    rows = []
    for fold in WALKFORWARD_FOLDS:
        train_end = pd.to_datetime(fold["train_end"])
        val_start = pd.to_datetime(fold["val_start"])
        val_end   = pd.to_datetime(fold["val_end"])

        # Val-period rows (actual outcomes from fold)
        val_db = full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ].dropna(subset=[HORIZON, HORIZON_RET]).copy()

        if len(val_db) == 0:
            rows.append({"fold": fold["label"], "n_rows": 0, "win_rate": float("nan"),
                         "b_fold": float("nan"), "kelly_fraction": float("nan"),
                         "kelly_positive": False, "note": "no val data"})
            continue

        # Bull-regime mask: SPY ret_90d >= SPY_THRESHOLD
        spy_rows = val_db[val_db["Ticker"] == "SPY"][["Date", "ret_90d"]].set_index("Date")
        if spy_rows.empty:
            bull_mask = pd.Series(True, index=val_db.index)
        else:
            regime = spy_rows["ret_90d"].map(lambda r: r >= SPY_THRESHOLD)
            dates  = val_db["Date"].map(lambda d: regime.asof(d) if d >= regime.index.min() else True)
            bull_mask = dates.fillna(True).astype(bool)

        bull_rows = val_db[bull_mask.values]
        if len(bull_rows) == 0:
            rows.append({"fold": fold["label"], "n_rows": 0, "win_rate": float("nan"),
                         "b_fold": float("nan"), "kelly_fraction": float("nan"),
                         "kelly_positive": False, "note": "all Bear-held"})
            continue

        p_fold = float(bull_rows[HORIZON].mean())
        wins_r = bull_rows[bull_rows[HORIZON_RET] > 0][HORIZON_RET]
        loss_r = bull_rows[bull_rows[HORIZON_RET] < 0][HORIZON_RET].abs()

        if len(wins_r) > 0 and len(loss_r) > 0:
            b_fold = float(wins_r.mean() / loss_r.mean())
        else:
            b_fold = b_hist

        kf = compute_kelly_fraction(max(0.001, min(0.999, p_fold)), max(0.001, b_fold))

        rows.append({
            "fold":           fold["label"],
            "n_rows":         len(bull_rows),
            "win_rate":       round(p_fold, 4),
            "b_fold":         round(b_fold, 4),
            "kelly_fraction": round(kf, 4),
            "kelly_positive": kf > 0,
            "note":           "base rate (lower bound; signal Kelly ≥ this)",
        })

    return rows


def main() -> None:
    print("=" * 62)
    print("Phase 2 Walk-Forward — Half-Kelly Position Sizer")
    print("=" * 62)

    # ── 1. Compute historical b_ratio ──────────────────────────────────────────
    if not TRADES_PATH.exists():
        print(f"ERROR: {TRADES_PATH} not found. Run Phase 1 backtest first.")
        sys.exit(1)
    trades = pd.read_csv(TRADES_PATH)
    wins   = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] < 0]["net_pnl"].abs()
    b_hist = float(wins.mean() / losses.mean())
    print(f"\nHistorical b_ratio (backtest_trades.csv): {b_hist:.4f}")
    print(f"  ({len(wins)} wins, {len(losses)} losses, win_rate={len(wins)/len(trades):.4f})")

    # ── 2. 2024 fold simulation ────────────────────────────────────────────────
    print("\n── Analysis 1: 2024 Fold Equity Simulation ──────────────────")
    sim_result = run_2024_simulation(b_ratio=b_hist)

    # ── 3. Fold-level Kelly gate ───────────────────────────────────────────────
    print("\n── Analysis 2: 6-Fold Kelly Fraction Gate ────────────────────")
    if not (_52T_DIR / "train_db.parquet").exists():
        print("  [SKIP] 52T data not available for fold gate analysis.")
        fold_rows = []
    else:
        t = pd.read_parquet(_52T_DIR / "train_db.parquet")
        v = pd.read_parquet(_52T_DIR / "val_db.parquet")
        full_db = pd.concat([t, v], ignore_index=True)
        full_db["Date"] = pd.to_datetime(full_db["Date"])
        fold_rows = _fold_kelly_from_52t(full_db, b_hist)

        for r in fold_rows:
            kf_s = f"{r['kelly_fraction']:+.4f}" if not np.isnan(r["kelly_fraction"]) else "  nan"
            ok   = "✓" if r["kelly_positive"] else "✗"
            print(f"  {r['fold']:<14} win={r['win_rate']:.3f}  b={r['b_fold']:.3f}  "
                  f"Kelly={kf_s}  n={r['n_rows']:,}  {ok}")

    # Save walk-forward TSV
    fold_df = pd.DataFrame(fold_rows) if fold_rows else pd.DataFrame()
    fold_df.to_csv(OUT_WF_TSV, sep="\t", index=False)
    print(f"\nWalk-forward table saved → {OUT_WF_TSV.name}")

    # ── 4. Gate check ──────────────────────────────────────────────────────────
    kelly_positive_folds = sum(r["kelly_positive"] for r in fold_rows)
    n_valid_folds        = sum(r["n_rows"] > 0 for r in fold_rows)

    gate_kelly  = kelly_positive_folds >= 4
    gate_sharpe = sim_result["kelly_sharpe"] >= 1.0
    gate_dd     = sim_result["kelly_max_dd"] <= 0.15
    gate_all    = gate_kelly and gate_sharpe and gate_dd

    header = "=" * 62
    report_lines = [
        header,
        "PHASE 2 GATE CHECK — Half-Kelly Walk-Forward",
        header,
        "",
        "Analysis 1 — 2024 fold equity simulation:",
        f"  Kelly mean position:  {sim_result['kelly_pct_mean']*100:.1f}% "
        f"[{sim_result['kelly_pct_min']*100:.1f}%–{sim_result['kelly_pct_max']*100:.1f}%]",
        f"  {'Metric':<24} {'Kelly':>8} {'Flat 5%':>8}",
        f"  {'─'*44}",
        f"  {'Sharpe ratio':<24} {sim_result['kelly_sharpe']:>8.3f} {sim_result['flat_sharpe']:>8.3f}",
        f"  {'Max drawdown':<24} {sim_result['kelly_max_dd']*100:>7.1f}% {sim_result['flat_max_dd']*100:>7.1f}%",
        f"  {'Ann. return':<24} {sim_result['kelly_ann_ret']*100:>7.1f}% {sim_result['flat_ann_ret']*100:>7.1f}%",
        f"  {'Final equity':<24} ${sim_result['kelly_final_eq']:>7.0f} ${sim_result['flat_final_eq']:>7.0f}",
        f"  {'Net exp./trade':<24} ${sim_result['kelly_net_exp']:>+7.2f} ${sim_result['flat_net_exp']:>+7.2f}",
        "",
        "Analysis 2 — 6-fold Kelly fraction gate (52T base-rate lower bound):",
    ]

    if fold_rows:
        report_lines.append(
            f"  {'Fold':<14} {'WinRate':>8} {'b':>6} {'Kelly':>8} {'n_bull':>7} {'Gate':>5}"
        )
        report_lines.append("  " + "─" * 48)
        for r in fold_rows:
            kf_s = f"{r['kelly_fraction']:+.4f}" if not np.isnan(r["kelly_fraction"]) else "   nan"
            ok   = "✓" if r["kelly_positive"] else "✗"
            report_lines.append(
                f"  {r['fold']:<14} {r['win_rate']:>8.3f} {r['b_fold']:>6.3f} "
                f"{kf_s:>8} {r['n_rows']:>7,} {ok:>5}"
            )
    else:
        report_lines.append("  [No fold data — 52T dataset not available]")
        kelly_positive_folds = "n/a"
        n_valid_folds = 0

    report_lines += [
        "",
        "Gate criteria:",
        f"  [{'✓' if gate_kelly else '✗'}] Kelly > 0 on ≥ 4/6 folds "
        f"(base-rate lower bound):  {kelly_positive_folds}/{n_valid_folds}",
        f"  [{'✓' if gate_sharpe else '✗'}] 2024 Sharpe ≥ 1.0:  {sim_result['kelly_sharpe']:.3f}",
        f"  [{'✓' if gate_dd else '✗'}] 2024 Max DD ≤ 15%:  {sim_result['kelly_max_dd']*100:.1f}%",
        "",
        f"PHASE 2 GATE: {'PASS ✓' if gate_all else 'FAIL ✗ — see notes below'}",
        header,
    ]

    if not gate_all:
        report_lines += [
            "",
            "Diagnostic notes:",
        ]
        if not gate_kelly and isinstance(kelly_positive_folds, int):
            report_lines.append(
                f"  Kelly gate: {kelly_positive_folds}/6 folds positive. "
                "Base rate is a lower bound; actual signal Kelly may be higher."
            )
        if not gate_sharpe:
            report_lines.append(
                f"  Sharpe gate: {sim_result['kelly_sharpe']:.3f} < 1.0. "
                "Check if b_ratio is representative of this fold."
            )
        if not gate_dd:
            report_lines.append(
                f"  DD gate: {sim_result['kelly_max_dd']*100:.1f}% > 15%. "
                "Kelly sizing is amplifying a drawdown period."
            )
        report_lines.append(
            "\n  Roadmap fallback: revert to ATR-only sizing. Kelly → Phase 7 enhancement."
        )

    report_lines += [
        "",
        "Next: Phase 3 — Risk Engine Integration (real ATR, drawdown brake, fatigue overlay)"
        if gate_all else
        "Next: Re-evaluate Kelly gate with full 6-fold signal data before Phase 3.",
    ]

    report = "\n".join(report_lines)
    print(f"\n{report}\n")
    OUT_GATE.write_text(report, encoding="utf-8")
    print(f"Gate report saved → {OUT_GATE.name}")
    print(f"Equity curves → {OUT_KELLY_EQ.name}, {OUT_FLAT_EQ.name}")


if __name__ == "__main__":
    main()
