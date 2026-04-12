"""
compare_kelly_vs_flat.py — Phase 2 T2.3: Kelly vs flat sizing comparison.

Loads results/cached_signals_2024.csv and results/backtest_trades.csv,
applies Half-Kelly sizing to each BUY signal, and prints a comparison
table showing position sizes vs the Phase 1 flat 5% baseline.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/compare_kelly_vs_flat.py

Outputs:
    Console: summary table + Kelly diagnostics
    results/phase2_kelly_sizing.csv — per-signal sizing results

Provenance:
    b_ratio = avg_win / avg_loss from results/backtest_trades.csv (2024 fold)
    p = calibrated confidence from each BUY signal
    All Kelly fractions traceable to this script's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_system.position_sizer import SizingConfig, size_position


FLAT_BASELINE_PCT = 0.05          # Phase 1 equal-weight
SIGNALS_PATH = Path("results/cached_signals_2024.csv")
TRADES_PATH  = Path("results/backtest_trades.csv")
OUTPUT_PATH  = Path("results/phase2_kelly_sizing.csv")


def compute_b_ratio(trades_path: Path) -> float:
    """Derive avg_win / avg_loss from historical backtest trades."""
    df = pd.read_csv(trades_path)
    wins   = df[df["net_pnl"] > 0]["net_pnl"]
    losses = df[df["net_pnl"] < 0]["net_pnl"].abs()
    if len(wins) == 0 or len(losses) == 0:
        raise RuntimeError("Cannot compute b_ratio: no wins or no losses in trades file")
    b = wins.mean() / losses.mean()
    print(f"  b_ratio (avg_win/avg_loss): {b:.4f}  "
          f"[{len(wins)} wins, {len(losses)} losses]")
    return float(b)


def apply_kelly_to_signals(signals: pd.DataFrame, b_ratio: float,
                            config: SizingConfig) -> pd.DataFrame:
    """Compute Kelly position sizes for all BUY signals."""
    rows = []
    for _, row in signals.iterrows():
        result = size_position(
            confidence=float(row["confidence"]),
            b_ratio=b_ratio,
            config=config,
        )
        rows.append({
            "date":            row["date"],
            "ticker":          row["ticker"],
            "confidence":      row["confidence"],
            "kelly_fraction":  result.kelly_fraction if result.approved else float("nan"),
            "half_kelly":      result.scaled_kelly   if result.approved else float("nan"),
            "atr_weight":      result.atr_weight     if result.approved else float("nan"),
            "position_pct":    result.position_pct,
            "flat_pct":        FLAT_BASELINE_PCT,
            "delta_vs_flat":   result.position_pct - FLAT_BASELINE_PCT,
            "approved":        result.approved,
            "rejection_reason": result.rejection_reason or "",
        })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, all_buy: pd.DataFrame) -> None:
    approved = df[df["approved"]]
    rejected = df[~df["approved"]]

    print(f"\n{'='*60}")
    print("PHASE 2 — Half-Kelly vs Flat Sizing: 2024 BUY Signals")
    print(f"{'='*60}")
    print(f"  Total BUY signals:         {len(all_buy):>6}")
    print(f"  Approved (Kelly > 0):      {len(approved):>6}")
    print(f"  Rejected (Kelly ≤ 0):      {len(rejected):>6}")

    if len(approved) == 0:
        print("\n  [WARNING] No approved positions — all BUY signals rejected by Kelly gate.")
        return

    print(f"\n  Position size (pct of equity):")
    print(f"    Flat baseline:           {FLAT_BASELINE_PCT*100:>6.1f}%")
    print(f"    Kelly mean:              {approved['position_pct'].mean()*100:>6.2f}%")
    print(f"    Kelly min:               {approved['position_pct'].min()*100:>6.2f}%")
    print(f"    Kelly max:               {approved['position_pct'].max()*100:>6.2f}%")
    print(f"    Kelly std:               {approved['position_pct'].std()*100:>6.2f}%")

    print(f"\n  Kelly fraction (full, before 0.5× multiplier):")
    print(f"    mean:                    {approved['kelly_fraction'].mean():>6.4f}")
    print(f"    min:                     {approved['kelly_fraction'].min():>6.4f}")
    print(f"    max:                     {approved['kelly_fraction'].max():>6.4f}")

    print(f"\n  Delta vs flat baseline:")
    positive_delta = (approved["delta_vs_flat"] > 0).sum()
    negative_delta = (approved["delta_vs_flat"] < 0).sum()
    print(f"    Larger than 5% flat:     {positive_delta:>6}  ({positive_delta/len(approved)*100:.1f}%)")
    print(f"    Smaller than 5% flat:    {negative_delta:>6}  ({negative_delta/len(approved)*100:.1f}%)")
    print(f"    Mean delta:              {approved['delta_vs_flat'].mean()*100:>+6.2f}%")

    # Gate check
    print(f"\n{'─'*60}")
    print("  PHASE 2 GATE CHECK:")

    # Compute Kelly fraction per "fold" using confidence as proxy for p
    # (full fold validation requires walk-forward — see run_walkforward.py)
    positive_kelly = (approved["kelly_fraction"] > 0).sum()
    pct_positive = positive_kelly / len(df)
    print(f"    Kelly positive signals:  {positive_kelly}/{len(df)} ({pct_positive*100:.1f}%)")

    # Max drawdown and Sharpe require full backtest — noted as pending
    print(f"    Sharpe ≥ 1.0:            [Run full backtest to evaluate]")
    print(f"    Max DD ≤ 15%:            [Run full backtest to evaluate]")

    status = "PASS" if pct_positive >= 2/3 else "NEEDS_BACKTEST"
    print(f"\n  Signal-level gate: {status}")
    print(f"  (Full gate requires fold-level Kelly fractions from walk-forward)")
    print(f"{'='*60}\n")


def main() -> None:
    print("Loading data...")
    if not SIGNALS_PATH.exists():
        print(f"ERROR: {SIGNALS_PATH} not found. Run walk-forward first.")
        sys.exit(1)
    if not TRADES_PATH.exists():
        print(f"ERROR: {TRADES_PATH} not found. Run backtest first.")
        sys.exit(1)

    signals = pd.read_csv(SIGNALS_PATH)
    buy_signals = signals[signals["signal"] == "BUY"].copy()
    print(f"  Loaded {len(signals):,} total signals, {len(buy_signals)} BUY signals")

    print("Computing b_ratio from historical trades...")
    b_ratio = compute_b_ratio(TRADES_PATH)

    config = SizingConfig()
    print(f"\nSizingConfig:")
    print(f"  max_loss_per_trade_pct: {config.max_loss_per_trade_pct}")
    print(f"  stop_loss_atr_multiple: {config.stop_loss_atr_multiple}")
    print(f"  flat_atr_pct:           {config.flat_atr_pct} (Phase 2 constant)")
    print(f"  kelly_multiplier:       {config.kelly_multiplier} (half-Kelly)")
    print(f"  position clamp:         [{config.min_position_pct*100:.0f}%, {config.max_position_pct*100:.0f}%]")

    print("\nApplying Kelly sizing to BUY signals...")
    results = apply_kelly_to_signals(buy_signals, b_ratio, config)

    print_summary(results, buy_signals)

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
