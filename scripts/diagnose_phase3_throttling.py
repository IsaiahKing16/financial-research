"""
diagnose_phase3_throttling.py — Diagnose Sharpe failure root cause.

Replays the Phase 3 walk-forward pipeline but records per-trade:
  - sizing.position_pct (post-Kelly, pre-overlay)
  - adj.dd_scalar
  - adj.overlay_multiplier
  - adj.final_position_pct
  - expected Phase 1 scaled PnL (if no overlays)
  - actual Phase 3 PnL (with overlays)

Goal: confirm whether FatigueAccumulationOverlay / LiquidityCongestionGate
are silently throttling positions via overlay_multiplier ∈ (0, 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from trading_system.position_sizer import SizingConfig, size_position
from trading_system.risk_engine import apply_risk_adjustments, compute_atr_pct
from trading_system.risk_overlays.fatigue_accumulation import FatigueAccumulationOverlay
from trading_system.risk_overlays.liquidity_congestion import LiquidityCongestionGate

TRADES_PATH = project_root / "results" / "backtest_trades.csv"
VAL_PATH    = project_root / "data" / "52t_volnorm" / "val_db.parquet"
OUT_PATH    = project_root / "results" / "phase3_throttling_diagnostic.csv"

INITIAL_EQUITY = 10_000.0
SPY_THRESHOLD  = 0.05

cfg = SizingConfig()


def compute_b_hist(trades: pd.DataFrame) -> float:
    wins = trades[trades["net_pnl"] > 0]["net_pnl"]
    losses = trades[trades["net_pnl"] < 0]["net_pnl"]
    return float(wins.mean() / losses.abs().mean())


def main() -> None:
    trades = pd.read_csv(TRADES_PATH)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"]  = pd.to_datetime(trades["exit_date"])
    trades = trades.sort_values("entry_date").reset_index(drop=True)

    b_hist = compute_b_hist(trades)
    print(f"b_hist = {b_hist:.4f}, n_trades = {len(trades)}")

    val = pd.read_parquet(VAL_PATH)
    val["Date"] = pd.to_datetime(val["Date"]).dt.date
    atr_lookup = val.set_index(["Date", "Ticker"])[["atr_14", "Close", "ret_90d"]].to_dict("index")

    fatigue = FatigueAccumulationOverlay(decay_rate=0.15)
    congestion = LiquidityCongestionGate()

    running_equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    pending_pnls: dict = {}
    rows = []

    for _, row in trades.iterrows():
        entry_dt = row["entry_date"].date()
        exit_dt = row["exit_date"].date()
        ticker = row["ticker"]

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
            ret_90d = spy_row.get("ret_90d", 0.0) or 0.0
            regime = "BULL" if ret_90d > SPY_THRESHOLD else "BEAR"
            fatigue.update(entry_dt, regime_label=regime)
            congestion.update(entry_dt, atr=spy_row["atr_14"], close=spy_row["Close"])

        fatigue_mult = fatigue.get_signal_multiplier()
        congestion_mult = congestion.get_signal_multiplier()

        ticker_row = atr_lookup.get((entry_dt, ticker))
        if ticker_row is None:
            continue
        try:
            atr_pct = compute_atr_pct(ticker_row["atr_14"], ticker_row["Close"])
        except RuntimeError:
            continue

        sizing = size_position(
            confidence=float(row["confidence_at_entry"]),
            b_ratio=b_hist,
            config=cfg,
            atr_pct=atr_pct,
        )
        dd = max(0.0, 1.0 - running_equity / peak_equity)
        adj = apply_risk_adjustments(sizing, drawdown=dd, overlays=[fatigue, congestion])

        original_pct = float(row["position_pct"])
        scale = adj.final_position_pct / original_pct if original_pct > 0 else 0.0
        scaled_pnl = float(row["net_pnl"]) * scale

        rows.append({
            "entry_date": entry_dt,
            "ticker": ticker,
            "sizing_position_pct": sizing.position_pct,
            "dd_scalar": adj.dd_scalar,
            "overlay_multiplier": adj.overlay_multiplier,
            "fatigue_mult": fatigue_mult,
            "congestion_mult": congestion_mult,
            "final_position_pct": adj.final_position_pct,
            "blocked": adj.blocked,
            "block_reason": adj.block_reason or "",
            "original_pct": original_pct,
            "scale_factor": scale,
            "phase1_net_pnl": float(row["net_pnl"]),
            "phase3_scaled_pnl": scaled_pnl,
        })

        if not adj.blocked and adj.final_position_pct > 0:
            pending_pnls[exit_dt] = pending_pnls.get(exit_dt, 0.0) + scaled_pnl

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n=== Per-Trade Diagnostic ({len(df)} trades) ===\n")

    print("sizing.position_pct (pre-overlay, post-Kelly):")
    print(f"  mean={df['sizing_position_pct'].mean():.4f}  "
          f"median={df['sizing_position_pct'].median():.4f}  "
          f"min={df['sizing_position_pct'].min():.4f}  "
          f"max={df['sizing_position_pct'].max():.4f}")

    print("\ndd_scalar distribution:")
    print(df["dd_scalar"].describe().to_string())
    print(f"  dd_scalar == 1.0: {(df['dd_scalar'] >= 0.999).sum()} / {len(df)}")
    print(f"  dd_scalar <  1.0: {(df['dd_scalar'] < 0.999).sum()}")

    print("\noverlay_multiplier distribution:")
    print(df["overlay_multiplier"].describe().to_string())
    print(f"  overlay == 1.0: {(df['overlay_multiplier'] >= 0.999).sum()} / {len(df)}")
    print(f"  overlay < 1.0:  {(df['overlay_multiplier'] < 0.999).sum()}")
    print(f"  overlay < 0.5:  {(df['overlay_multiplier'] < 0.5).sum()}")
    print(f"  overlay == 0.0: {(df['overlay_multiplier'] <= 0.001).sum()}")

    print("\nfatigue_mult distribution:")
    print(df["fatigue_mult"].describe().to_string())

    print("\ncongestion_mult distribution:")
    print(df["congestion_mult"].describe().to_string())

    print("\nfinal_position_pct distribution:")
    print(df["final_position_pct"].describe().to_string())
    print(f"  final == 0: {(df['final_position_pct'] <= 0.0001).sum()} (blocked)")

    print("\nblocked count:", df["blocked"].sum())
    if df["blocked"].sum() > 0:
        print("block reasons:")
        print(df[df["blocked"]]["block_reason"].value_counts().to_string())

    print("\n=== PnL Comparison ===")
    total_phase1 = df["phase1_net_pnl"].sum()
    total_phase3 = df["phase3_scaled_pnl"].sum()
    print(f"  Sum of Phase 1 net_pnl (flat 5%):    ${total_phase1:,.2f}")
    print(f"  Sum of Phase 3 scaled_pnl (overlays): ${total_phase3:,.2f}")
    print(f"  Ratio: {total_phase3 / total_phase1:.3f}")

    # Hypothetical: what if overlay_multiplier were forced to 1.0?
    df["sizing_only_scale"] = df["sizing_position_pct"] / df["original_pct"].where(df["original_pct"] > 0, 1.0)
    df["sizing_only_pnl"] = df["phase1_net_pnl"] * df["sizing_only_scale"]
    total_sizing_only = df["sizing_only_pnl"].sum()
    print(f"\n  Sizing-only (no overlay, no DD):     ${total_sizing_only:,.2f}")
    print(f"  Overlay+DD drag: ${total_sizing_only - total_phase3:,.2f}")

    print(f"\nDiagnostic saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
