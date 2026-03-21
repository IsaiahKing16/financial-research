"""
backtest_engine.py — Layer 1: Trade Simulation with Realistic Friction

Takes FPPE signal output and simulates trade execution at next-day open
with configurable transaction costs. Produces:
  1. Trade log (one row per completed trade)
  2. Daily equity curve
  3. Rejected signal log

This module is PURELY MECHANICAL. It applies rules to data and records
outcomes. It contains zero optimization logic.

Phase 1 uses equal-weight position sizing (5% of equity per trade).
Phases 2-3 will replace this with volatility-based sizing (Layer 2)
and confidence-based ranking (Layer 3).

Design doc reference: FPPE_TRADING_SYSTEM_DESIGN.md v0.3, Section 4.2
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, replace as _dc_replace
from typing import List, Dict, Optional, Tuple
from .config import TradingConfig, DEFAULT_CONFIG, SECTOR_MAP
from .risk_engine import check_stop_loss, size_position
from .risk_state import RiskState, StopLossEvent
from .portfolio_manager import (
    rank_signals as _pm_rank_signals,
    check_allocation as _pm_check_allocation,
)
from .portfolio_state import PortfolioSnapshot
from research.slip_deficit import SlipDeficit as _SlipDeficit


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class OpenPosition:
    """Tracks a single open position."""
    trade_id: int
    ticker: str
    sector: str
    entry_date: pd.Timestamp
    raw_entry_price: float         # Next-day open BEFORE friction; used for gross P&L
    entry_price: float             # raw_entry_price × (1 + entry_bps) — actual cash cost basis
    shares: float
    position_pct: float            # % of equity at entry
    confidence_at_entry: float
    stop_loss_price: float         # Set by risk engine; 0 = no stop in Phase 1
    atr_pct_at_entry: float = 0.0  # ATR% at entry for stop-loss audit records
    days_held: int = 0
    last_close_price: float = 0.0  # Last observed close; MTM fallback for halted days


@dataclass
class CompletedTrade:
    """Record of a completed (closed) trade."""
    trade_id: int
    ticker: str
    sector: str
    direction: str                 # "LONG" for v1
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    position_pct: float
    shares: float
    gross_pnl: float               # (raw_exit - raw_entry) × shares — no friction
    entry_friction_cost: float     # Friction paid at entry (slippage + spread)
    exit_friction_cost: float      # Friction paid at exit (slippage + spread)
    slippage_cost: float           # Exit-side slippage only (for attribution)
    spread_cost: float             # Exit-side spread only (for attribution)
    total_costs: float             # entry_friction_cost + exit_friction_cost
    net_pnl: float                 # gross_pnl - total_costs
    holding_days: int
    exit_reason: str               # "signal", "stop_loss", "max_hold", "drawdown_halt"
    confidence_at_entry: float


@dataclass
class DailyRecord:
    """One row of the daily equity curve."""
    date: pd.Timestamp
    equity: float
    cash: float
    invested_capital: float
    gross_exposure: float          # Sum of position values / equity
    open_positions: int
    daily_return: float
    cumulative_return: float
    drawdown_from_peak: float
    cash_yield_today: float
    strategy_return_excl_cash: float
    strategy_return_incl_cash: float


@dataclass
class RejectedSignal:
    """Record of a signal that was filtered out."""
    date: pd.Timestamp
    ticker: str
    signal: str
    confidence: float
    rejection_reason: str
    rejection_layer: str           # "cooldown", "position_limit", "sector_limit", etc.


# ============================================================
# BACKTEST ENGINE
# ============================================================

class BacktestEngine:
    """Simulates trade execution on historical FPPE signals.

    Usage:
        engine = BacktestEngine(config)
        results = engine.run(signal_df, price_df)
        print(results.summary())
    """

    def __init__(
        self,
        config: TradingConfig = None,
        use_risk_engine: bool = False,
        use_portfolio_manager: bool = False,
    ):
        if use_portfolio_manager and not use_risk_engine:
            raise ValueError(
                "use_portfolio_manager=True requires use_risk_engine=True"
            )
        self.config = config or DEFAULT_CONFIG
        self.use_risk_engine = use_risk_engine
        self.use_portfolio_manager = use_portfolio_manager
        self._slip_deficit = _SlipDeficit()  # stateless; instantiated once per engine
        self._validate_config()

    def _validate_config(self):
        errors = self.config.validate()
        if errors:
            raise ValueError(
                f"Invalid config: {'; '.join(errors)}"
            )

    def run(
        self,
        signal_df: pd.DataFrame,
        price_df: pd.DataFrame,
        equal_weight_pct: float = 0.05,
        use_risk_engine: Optional[bool] = None,
        use_portfolio_manager: Optional[bool] = None,
    ) -> "BacktestResults":
        """Run the full backtest simulation.

        Args:
            signal_df: FPPE signals with columns:
                [date, ticker, signal, confidence, sector]
                One row per ticker per day.

            price_df: Historical OHLC data with columns:
                [Date, Ticker, Open, High, Low, Close]
                Must cover the same date range as signal_df plus one extra
                trading day at the end (for final exits).

            equal_weight_pct: Position size as % of equity for Phase 1.
                Default 5% (max 20 simultaneous positions at 100% exposure).
                Ignored when use_risk_engine=True.

            use_risk_engine: Optional override for this run call. If None,
                uses BacktestEngine.use_risk_engine from construction.

        Returns:
            BacktestResults containing trade log, equity curve, and rejected signals.
        """
        # ── Validate inputs ──────────────────────────────────────────
        self._validate_inputs(signal_df, price_df)
        risk_engine_enabled = (
            self.use_risk_engine if use_risk_engine is None else use_risk_engine
        )
        pm_enabled = (
            self.use_portfolio_manager if use_portfolio_manager is None
            else use_portfolio_manager
        )
        if pm_enabled and not risk_engine_enabled:
            raise ValueError(
                "use_portfolio_manager=True requires use_risk_engine=True"
            )

        # ── Initialize state ─────────────────────────────────────────
        cash = self.config.capital.initial_capital
        equity = cash
        peak_equity = equity
        open_positions: Dict[str, OpenPosition] = {}  # ticker → position
        trade_log: List[CompletedTrade] = []
        daily_records: List[DailyRecord] = []
        rejected_signals: List[RejectedSignal] = []
        stop_loss_events: List[StopLossEvent] = []
        cooldowns: Dict[str, Dict] = {}  # ticker → {until_date, last_confidence}
        trade_counter = 0
        risk_state = (
            RiskState.initial(cash) if risk_engine_enabled else None
        )

        # Build price lookup: (date, ticker) → {open, high, low, close}
        price_lookup = self._build_price_lookup(price_df)
        price_history_by_ticker: Dict[str, pd.DataFrame] = {}
        if risk_engine_enabled:
            history_df = price_df.copy()
            history_df["Date"] = pd.to_datetime(history_df["Date"])
            for ticker, ticker_hist in history_df.groupby("Ticker", sort=False):
                price_history_by_ticker[ticker] = (
                    ticker_hist[["Date", "Open", "High", "Low", "Close"]]
                    .sort_values("Date")
                    .reset_index(drop=True)
                )

        # Get sorted unique trading dates from signals
        signal_dates = sorted(signal_df["date"].unique())
        # Get all trading dates from price data for mark-to-market
        all_price_dates = sorted(price_df["Date"].unique())

        # ── Configuration shortcuts ──────────────────────────────────
        cfg_costs = self.config.costs
        cfg_trade = self.config.trade_management
        cfg_pos = self.config.position_limits
        cfg_risk = self.config.risk

        daily_rf_rate = cfg_costs.risk_free_annual_rate / 252

        # Track cumulative cash yield separately from trading P&L
        cumulative_cash_yield = 0.0
        cumulative_trading_pnl = 0.0

        if pm_enabled:
            engine_mode = "Phase 3 (Portfolio Manager + Risk Engine)"
        elif risk_engine_enabled:
            engine_mode = "Phase 2 (Risk Engine)"
        else:
            engine_mode = "Phase 1 (Equal Weight)"
        print(f"\n{'='*60}")
        print(f"  BACKTEST ENGINE — {engine_mode}")
        if risk_engine_enabled:
            print(
                f"  Capital: ${cash:,.0f} | "
                f"ATR lookback: {cfg_risk.volatility_lookback} | "
                f"Stop: {cfg_risk.stop_loss_atr_multiple:.1f}x ATR"
            )
        else:
            print(f"  Capital: ${cash:,.0f} | Position size: {equal_weight_pct:.0%}")
        print(f"  Friction: {cfg_costs.round_trip_bps:.0f} bps round-trip")
        print(f"  Signals: {len(signal_df):,} rows over {len(signal_dates)} days")
        print(f"{'='*60}\n")

        # ── Main simulation loop ─────────────────────────────────────
        for day_idx, current_date in enumerate(signal_dates):
            current_date = pd.Timestamp(current_date)

            # Find next trading day for execution
            next_date = self._next_trading_day(current_date, all_price_dates)
            if next_date is None:
                # Last day — can't execute, just mark-to-market
                continue

            # ── Step 1: Check exits on open positions ────────────────
            positions_to_close = []
            stop_trigger_details: Dict[str, Dict[str, float]] = {}

            for ticker, pos in open_positions.items():
                pos.days_held += 1

                # Get today's price data
                prices_today = price_lookup.get((current_date, ticker))
                if prices_today is None:
                    continue

                exit_reason = None

                if risk_engine_enabled:
                    # Phase 2: keep stop-loss as highest-priority exit reason
                    # to preserve explicit stop-loss auditability.
                    if pos.days_held >= cfg_trade.max_holding_days:
                        exit_reason = "max_hold"

                    if check_stop_loss(prices_today["low"], pos.stop_loss_price):
                        exit_reason = "stop_loss"
                        stop_trigger_details[ticker] = {
                            "trigger_low": prices_today["low"],
                            "stop_price": pos.stop_loss_price,
                        }

                    # Check for exit signal on held long position.
                    # SELL = explicit exit signal → close position.
                    # HOLD = "don't open new positions" but does NOT close existing ones.
                    # Only SELL triggers an exit from a signal.
                    day_signals = signal_df[
                        (signal_df["date"] == current_date) &
                        (signal_df["ticker"] == ticker)
                    ]
                    if len(day_signals) > 0:
                        sig = day_signals.iloc[0]["signal"]
                        if sig in ("SELL",) and exit_reason != "stop_loss":
                            exit_reason = "signal"

                    # Drawdown halt for forced exits only applies when no
                    # higher-priority reason has already been selected.
                    if exit_reason is None and equity > 0:
                        current_dd = 1.0 - (equity / peak_equity)
                        if current_dd >= cfg_risk.drawdown_halt_threshold:
                            exit_reason = "drawdown_halt"
                else:
                    # Check max holding period
                    if pos.days_held >= cfg_trade.max_holding_days:
                        exit_reason = "max_hold"

                    # Check stop-loss (evaluated on intraday low)
                    if pos.stop_loss_price > 0 and prices_today["low"] <= pos.stop_loss_price:
                        exit_reason = "stop_loss"

                    # Check for exit signal on held long position.
                    # SELL = explicit exit signal → close position.
                    # HOLD = "don't open new positions" but does NOT close existing ones.
                    # Only SELL triggers an exit from a signal. The original bug
                    # treated HOLD as an exit, causing every position to close after
                    # 1 day (since most tickers return to HOLD the next day).
                    day_signals = signal_df[
                        (signal_df["date"] == current_date) &
                        (signal_df["ticker"] == ticker)
                    ]
                    if len(day_signals) > 0:
                        sig = day_signals.iloc[0]["signal"]
                        if sig in ("SELL",):
                            exit_reason = "signal"

                    # Check drawdown halt
                    if equity > 0:
                        current_dd = 1.0 - (equity / peak_equity)
                        if current_dd >= cfg_risk.drawdown_halt_threshold:
                            exit_reason = "drawdown_halt"

                if exit_reason:
                    positions_to_close.append((ticker, exit_reason))

            # ── Step 2: Execute exits at next-day open ───────────────
            for ticker, exit_reason in positions_to_close:
                pos = open_positions[ticker]
                next_prices = price_lookup.get((next_date, ticker))
                if next_prices is None:
                    continue  # Can't exit if no price data

                # Exit at next open with slippage (selling: price goes down)
                raw_exit_price = next_prices["open"]
                slippage_cost_per_share = raw_exit_price * (cfg_costs.slippage_bps / 10_000)
                spread_cost_per_share = raw_exit_price * (cfg_costs.spread_bps / 10_000)
                exit_price = raw_exit_price - slippage_cost_per_share - spread_cost_per_share

                # ── P&L: use raw_entry_price so entry friction is not double-counted ──
                # gross_pnl measures pure price movement from raw open to raw open.
                # Friction at both legs is captured separately in entry/exit_friction_cost.
                gross_pnl = (raw_exit_price - pos.raw_entry_price) * pos.shares

                # Exit friction (paid now at exit)
                exit_slippage = slippage_cost_per_share * pos.shares
                exit_spread = spread_cost_per_share * pos.shares
                exit_friction_cost = exit_slippage + exit_spread

                # Entry friction (already paid at entry; stored for attribution)
                entry_friction_cost = (pos.entry_price - pos.raw_entry_price) * pos.shares

                total_costs = entry_friction_cost + exit_friction_cost
                net_pnl = gross_pnl - total_costs

                # Record completed trade
                trade_log.append(CompletedTrade(
                    trade_id=pos.trade_id,
                    ticker=ticker,
                    sector=pos.sector,
                    direction="LONG",
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=next_date,
                    exit_price=exit_price,
                    position_pct=pos.position_pct,
                    shares=pos.shares,
                    gross_pnl=gross_pnl,
                    entry_friction_cost=entry_friction_cost,
                    exit_friction_cost=exit_friction_cost,
                    slippage_cost=exit_slippage,
                    spread_cost=exit_spread,
                    total_costs=total_costs,
                    net_pnl=net_pnl,
                    holding_days=pos.days_held,
                    exit_reason=exit_reason,
                    confidence_at_entry=pos.confidence_at_entry,
                ))

                # Update cash
                proceeds = pos.shares * exit_price
                cash += proceeds
                cumulative_trading_pnl += net_pnl

                if exit_reason == "stop_loss" and ticker in stop_trigger_details:
                    stop_details = stop_trigger_details[ticker]
                    stop_loss_events.append(
                        StopLossEvent(
                            ticker=ticker,
                            trigger_date=str(current_date.date()),
                            stop_price=float(stop_details["stop_price"]),
                            trigger_low=float(stop_details["trigger_low"]),
                            entry_price=pos.entry_price,
                            exit_price=raw_exit_price,
                            gap_through=float(stop_details["trigger_low"]) < float(stop_details["stop_price"]),
                            atr_at_entry=pos.atr_pct_at_entry,
                        )
                    )

                # Set cooldown if exit was stop-loss or max-hold
                if exit_reason in ("stop_loss", "max_hold"):
                    cooldown_days = (
                        cfg_trade.cooldown_after_stop_days if exit_reason == "stop_loss"
                        else cfg_trade.cooldown_after_maxhold_days
                    )
                    cooldown_until = self._advance_trading_days(
                        next_date, cooldown_days, all_price_dates
                    )
                    cooldowns[ticker] = {
                        "until_date": cooldown_until,
                        "last_confidence": pos.confidence_at_entry,
                    }

                if risk_engine_enabled and risk_state is not None:
                    risk_state.remove_stop(ticker)
                del open_positions[ticker]

            # ── Step 3: Process new BUY signals ──────────────────────
            day_buys = signal_df[
                (signal_df["date"] == current_date) &
                (signal_df["signal"] == "BUY")
            ].sort_values("confidence", ascending=False)

            if pm_enabled:
                # === Phase 3: Unified PM + Risk Engine loop ====================
                # PM and risk engine are interleaved per signal so that a risk
                # engine rejection does NOT burn a PM sector slot.  Running state
                # (sector counts, open tickers) is updated ONLY after BOTH layers
                # approve and the trade is physically executed.

                # 1. Clean expired cooldowns before ranking
                expired_cd = [
                    t for t, cd in cooldowns.items()
                    if current_date >= cd["until_date"]
                ]
                for t in expired_cd:
                    del cooldowns[t]

                # 2. Convert signal rows to dicts for portfolio_manager
                day_buy_signals = []
                for _, row in day_buys.iterrows():
                    _t = row["ticker"]
                    day_buy_signals.append({
                        "ticker": _t,
                        "confidence": row["confidence"],
                        "date": current_date,
                        "sector": row.get("sector") or SECTOR_MAP.get(_t, "Unknown"),
                    })

                # 3. Rank signals (confidence desc, ticker asc tie-break)
                ranked_signals = _pm_rank_signals(
                    day_buy_signals, self.config.sector_map
                )

                # 4. Running state — updated ONLY when BOTH PM + risk engine approve
                running_sector_counts: Dict[str, int] = {}
                for p in open_positions.values():
                    running_sector_counts[p.sector] = (
                        running_sector_counts.get(p.sector, 0) + 1
                    )
                running_open_tickers: set = set(open_positions.keys())

                for signal in ranked_signals:
                    ticker = signal.ticker
                    confidence = signal.confidence
                    sector = signal.sector

                    # 4a. PM gate: count-based constraints against running state
                    running_snapshot = PortfolioSnapshot(
                        open_tickers=frozenset(running_open_tickers),
                        sector_position_counts=running_sector_counts,
                        cooldowns=cooldowns,
                        cooldown_reentry_margin=cfg_trade.reentry_confidence_margin,
                    )
                    pm_dec = _pm_check_allocation(signal, running_snapshot, cfg_pos)
                    if not pm_dec.approved:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=pm_dec.rejection_reason or "Portfolio rejected",
                            rejection_layer="portfolio",
                        ))
                        continue

                    # 4b. Data check
                    next_prices = price_lookup.get((next_date, ticker))
                    if next_prices is None:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason="No price data for next trading day",
                            rejection_layer="data",
                        ))
                        continue

                    raw_entry_price = next_prices["open"]
                    entry_price = raw_entry_price * (1 + cfg_costs.total_entry_bps / 10_000)

                    # 4c. Risk engine: ATR history + sizing
                    # PM checked count-based constraints (holding, cooldown, sector count).
                    # size_position re-checks holding + sector count internally for
                    # defense-in-depth — intentional, not redundant duplication.
                    # Dollar-based constraints (ATR stop, drawdown brake, exposure %)
                    # are ONLY handled here; PM has no visibility into position sizes.
                    history_rows = cfg_risk.volatility_lookback + 1
                    price_history = self._get_ticker_history(
                        price_history_by_ticker=price_history_by_ticker,
                        ticker=ticker,
                        as_of_date=current_date,
                        n_rows=history_rows,
                    )
                    if price_history.empty:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason="Insufficient history: 0 rows",
                            rejection_layer="risk_engine",
                        ))
                        continue

                    # --- SlipDeficit TTF gate (Phase 3.5 research integration) ---
                    # Tighten stop to 1.5×ATR when short-term vol Z-score > 0.8 sigmoid.
                    # SlipDeficit requires 200 rows — fetch a wider window separately.
                    _slip_history = self._get_ticker_history(
                        price_history_by_ticker=price_history_by_ticker,
                        ticker=ticker,
                        as_of_date=current_date,
                        n_rows=200,
                    )
                    try:
                        _slip_df = _slip_history[["Close"]].rename(columns={"Close": "close"})
                        _overlay = self._slip_deficit.compute(_slip_df)
                        _effective_atr_mult = (
                            1.5 if _overlay.ttf_probability > 0.8
                            else cfg_risk.stop_loss_atr_multiple
                        )
                    except ValueError:
                        # Insufficient history (<200 rows) — use configured multiple unchanged
                        _effective_atr_mult = cfg_risk.stop_loss_atr_multiple
                    _cfg_risk_effective = _dc_replace(
                        cfg_risk, stop_loss_atr_multiple=_effective_atr_mult
                    )

                    decision = size_position(
                        ticker=ticker,
                        entry_price=entry_price,
                        current_equity=equity,
                        price_history=price_history,
                        risk_state=risk_state if risk_state is not None else RiskState.initial(equity),
                        config=_cfg_risk_effective,
                        position_limits=cfg_pos,
                        sector_map=self.config.sector_map,
                        open_positions=open_positions,
                        fractional_shares=self.config.capital.fractional_shares,
                    )
                    if not decision.approved:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=decision.rejection_reason or "Risk engine rejected trade",
                            rejection_layer="risk_engine",
                        ))
                        continue

                    # 4d. Post-sizing dollar guards (PM checks counts; these check dollars)
                    sector_exposure = sum(
                        p.position_pct for p in open_positions.values()
                        if p.sector == sector
                    )
                    if sector_exposure + decision.position_pct > cfg_pos.max_sector_pct:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Sector {sector} exposure would exceed {cfg_pos.max_sector_pct:.0%}",
                            rejection_layer="sector_limit",
                        ))
                        continue

                    current_exposure = sum(
                        p.position_pct for p in open_positions.values()
                    )
                    if current_exposure + decision.position_pct > self.config.capital.max_gross_exposure:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Gross exposure would exceed {self.config.capital.max_gross_exposure:.0%}",
                            rejection_layer="exposure_limit",
                        ))
                        continue

                    if cash < decision.dollar_amount:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Insufficient cash (${cash:,.0f} < ${decision.dollar_amount:,.0f})",
                            rejection_layer="capital",
                        ))
                        continue

                    # 4e. Both layers approved — execute and update PM running state
                    cost = decision.shares * entry_price
                    cash -= cost

                    trade_counter += 1
                    open_positions[ticker] = OpenPosition(
                        trade_id=trade_counter,
                        ticker=ticker,
                        sector=sector,
                        entry_date=next_date,
                        raw_entry_price=raw_entry_price,
                        entry_price=entry_price,
                        shares=decision.shares,
                        position_pct=decision.position_pct,
                        confidence_at_entry=confidence,
                        stop_loss_price=decision.stop_price,
                        atr_pct_at_entry=decision.atr_pct,
                    )
                    if risk_state is not None:
                        risk_state.register_stop(ticker, decision.stop_price)

                    # Sector slot committed only here — after full execution
                    running_open_tickers.add(ticker)
                    running_sector_counts[sector] = (
                        running_sector_counts.get(sector, 0) + 1
                    )

            # Phase 1/2 path — skipped when pm_enabled=True
            for _, sig_row in ([] if pm_enabled else day_buys.iterrows()):
                ticker = sig_row["ticker"]
                confidence = sig_row["confidence"]
                sector = sig_row.get("sector", SECTOR_MAP.get(ticker, None))
                if sector is None:
                    # Ticker not in signal_df.sector AND not in SECTOR_MAP.
                    # "Unknown" bypasses sector concentration checks — flag it
                    # visibly rather than silently allowing unlimited exposure.
                    import warnings
                    warnings.warn(
                        f"Ticker '{ticker}' has no sector mapping. "
                        f"Using 'Unknown' — this bypasses sector concentration limits. "
                        f"Add '{ticker}' to SECTOR_MAP in config.py.",
                        stacklevel=2
                    )
                    sector = "Unknown"

                if risk_engine_enabled:
                    # ── Phase 2 rejection checks before sizing ────────────────
                    if ticker in open_positions:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason="Already holding position",
                            rejection_layer="backtest",
                        ))
                        continue

                    if ticker in cooldowns:
                        cd = cooldowns[ticker]
                        if current_date < cd["until_date"]:
                            if confidence < cd["last_confidence"] + cfg_trade.reentry_confidence_margin:
                                rejected_signals.append(RejectedSignal(
                                    date=current_date, ticker=ticker, signal="BUY",
                                    confidence=confidence,
                                    rejection_reason=(
                                        f"In cooldown until {cd['until_date'].date()}; "
                                        f"need confidence ≥ {cd['last_confidence'] + cfg_trade.reentry_confidence_margin:.2f}"
                                    ),
                                    rejection_layer="cooldown",
                                ))
                                continue
                        else:
                            del cooldowns[ticker]

                    sector_positions = sum(
                        1 for p in open_positions.values() if p.sector == sector
                    )
                    if sector_positions >= cfg_pos.max_positions_per_sector:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Sector {sector} at max {cfg_pos.max_positions_per_sector} positions",
                            rejection_layer="sector_limit",
                        ))
                        continue

                    next_prices = price_lookup.get((next_date, ticker))
                    if next_prices is None:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason="No price data for next trading day",
                            rejection_layer="data",
                        ))
                        continue

                    # Buy at next open with slippage (buying: price goes up)
                    raw_entry_price = next_prices["open"]
                    entry_price = raw_entry_price * (1 + cfg_costs.total_entry_bps / 10_000)

                    history_rows = cfg_risk.volatility_lookback + 1
                    price_history = self._get_ticker_history(
                        price_history_by_ticker=price_history_by_ticker,
                        ticker=ticker,
                        as_of_date=current_date,
                        n_rows=history_rows,
                    )
                    if price_history.empty:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason="Insufficient history: 0 rows",
                            rejection_layer="risk_engine",
                        ))
                        continue

                    # --- SlipDeficit TTF gate (Phase 3.5 research integration) ---
                    _slip_history = self._get_ticker_history(
                        price_history_by_ticker=price_history_by_ticker,
                        ticker=ticker,
                        as_of_date=current_date,
                        n_rows=200,
                    )
                    try:
                        _slip_df = _slip_history[["Close"]].rename(columns={"Close": "close"})
                        _overlay = self._slip_deficit.compute(_slip_df)
                        _effective_atr_mult = (
                            1.5 if _overlay.ttf_probability > 0.8
                            else cfg_risk.stop_loss_atr_multiple
                        )
                    except ValueError:
                        _effective_atr_mult = cfg_risk.stop_loss_atr_multiple
                    _cfg_risk_effective = _dc_replace(
                        cfg_risk, stop_loss_atr_multiple=_effective_atr_mult
                    )

                    decision = size_position(
                        ticker=ticker,
                        entry_price=entry_price,
                        current_equity=equity,
                        price_history=price_history,
                        risk_state=risk_state if risk_state is not None else RiskState.initial(equity),
                        config=_cfg_risk_effective,
                        position_limits=cfg_pos,
                        sector_map=self.config.sector_map,
                        open_positions=open_positions,
                        fractional_shares=self.config.capital.fractional_shares,
                    )
                    if not decision.approved:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=decision.rejection_reason or "Risk engine rejected trade",
                            rejection_layer="risk_engine",
                        ))
                        continue

                    # Keep the existing backtest-level exposure/cash constraints.
                    sector_exposure = sum(
                        p.position_pct for p in open_positions.values() if p.sector == sector
                    )
                    if sector_exposure + decision.position_pct > cfg_pos.max_sector_pct:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Sector {sector} exposure would exceed {cfg_pos.max_sector_pct:.0%}",
                            rejection_layer="sector_limit",
                        ))
                        continue

                    current_exposure = sum(p.position_pct for p in open_positions.values())
                    if current_exposure + decision.position_pct > self.config.capital.max_gross_exposure:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Gross exposure would exceed {self.config.capital.max_gross_exposure:.0%}",
                            rejection_layer="exposure_limit",
                        ))
                        continue

                    if cash < decision.dollar_amount:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason=f"Insufficient cash (${cash:,.0f} < ${decision.dollar_amount:,.0f})",
                            rejection_layer="capital",
                        ))
                        continue

                    # Deduct from cash
                    cost = decision.shares * entry_price
                    cash -= cost

                    trade_counter += 1
                    open_positions[ticker] = OpenPosition(
                        trade_id=trade_counter,
                        ticker=ticker,
                        sector=sector,
                        entry_date=next_date,
                        raw_entry_price=raw_entry_price,
                        entry_price=entry_price,
                        shares=decision.shares,
                        position_pct=decision.position_pct,
                        confidence_at_entry=confidence,
                        stop_loss_price=decision.stop_price,
                        atr_pct_at_entry=decision.atr_pct,
                    )
                    if risk_state is not None:
                        risk_state.register_stop(ticker, decision.stop_price)
                    continue

                # ── Rejection checks ─────────────────────────────────

                # Already holding this ticker
                if ticker in open_positions:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason="Already holding position",
                        rejection_layer="backtest",
                    ))
                    continue

                # Cooldown check
                if ticker in cooldowns:
                    cd = cooldowns[ticker]
                    if current_date < cd["until_date"]:
                        # In cooldown — check if re-entry margin is met
                        if confidence < cd["last_confidence"] + cfg_trade.reentry_confidence_margin:
                            rejected_signals.append(RejectedSignal(
                                date=current_date, ticker=ticker, signal="BUY",
                                confidence=confidence,
                                rejection_reason=(
                                    f"In cooldown until {cd['until_date'].date()}; "
                                    f"need confidence ≥ {cd['last_confidence'] + cfg_trade.reentry_confidence_margin:.2f}"
                                ),
                                rejection_layer="cooldown",
                            ))
                            continue
                    else:
                        # Cooldown expired
                        del cooldowns[ticker]

                # Sector limit check
                sector_positions = sum(
                    1 for p in open_positions.values() if p.sector == sector
                )
                if sector_positions >= cfg_pos.max_positions_per_sector:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason=f"Sector {sector} at max {cfg_pos.max_positions_per_sector} positions",
                        rejection_layer="sector_limit",
                    ))
                    continue

                # Sector exposure check
                sector_exposure = sum(
                    p.position_pct for p in open_positions.values() if p.sector == sector
                )
                if sector_exposure + equal_weight_pct > cfg_pos.max_sector_pct:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason=f"Sector {sector} exposure would exceed {cfg_pos.max_sector_pct:.0%}",
                        rejection_layer="sector_limit",
                    ))
                    continue

                # Capital check
                position_value = equity * equal_weight_pct
                if position_value < equity * cfg_pos.min_position_pct:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason="Position below minimum size",
                        rejection_layer="position_limit",
                    ))
                    continue

                if cash < position_value:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason=f"Insufficient cash (${cash:,.0f} < ${position_value:,.0f})",
                        rejection_layer="capital",
                    ))
                    continue

                # Gross exposure check
                current_exposure = sum(
                    p.position_pct for p in open_positions.values()
                )
                if current_exposure + equal_weight_pct > self.config.capital.max_gross_exposure:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason=f"Gross exposure would exceed {self.config.capital.max_gross_exposure:.0%}",
                        rejection_layer="exposure_limit",
                    ))
                    continue

                # Drawdown halt check
                current_dd = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0.0
                if current_dd >= cfg_risk.drawdown_halt_threshold:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason=f"Drawdown halt ({current_dd:.1%} ≥ {cfg_risk.drawdown_halt_threshold:.0%})",
                        rejection_layer="risk",
                    ))
                    continue

                # ── Execute entry at next-day open ───────────────────
                next_prices = price_lookup.get((next_date, ticker))
                if next_prices is None:
                    rejected_signals.append(RejectedSignal(
                        date=current_date, ticker=ticker, signal="BUY",
                        confidence=confidence,
                        rejection_reason="No price data for next trading day",
                        rejection_layer="data",
                    ))
                    continue

                # Buy at next open with slippage (buying: price goes up)
                raw_entry_price = next_prices["open"]
                entry_price = raw_entry_price * (1 + cfg_costs.total_entry_bps / 10_000)

                shares = position_value / entry_price
                if not self.config.capital.fractional_shares:
                    shares = int(shares)
                    if shares == 0:
                        rejected_signals.append(RejectedSignal(
                            date=current_date, ticker=ticker, signal="BUY",
                            confidence=confidence,
                            rejection_reason="Position too small for whole shares",
                            rejection_layer="position_limit",
                        ))
                        continue

                # Deduct from cash
                cost = shares * entry_price
                cash -= cost

                trade_counter += 1
                open_positions[ticker] = OpenPosition(
                    trade_id=trade_counter,
                    ticker=ticker,
                    sector=sector,
                    entry_date=next_date,
                    raw_entry_price=raw_entry_price,   # next-day open before friction
                    entry_price=entry_price,           # friction-adjusted cost basis
                    shares=shares,
                    position_pct=equal_weight_pct,
                    confidence_at_entry=confidence,
                    stop_loss_price=0.0,  # Phase 1: no stop-loss
                )

            # ── Step 4: Mark-to-market and record daily state ────────
            invested_value = 0.0
            for ticker, pos in open_positions.items():
                prices = price_lookup.get((current_date, ticker))
                if prices:
                    pos.last_close_price = prices["close"]
                    invested_value += pos.shares * prices["close"]
                else:
                    # Fall back to last known close, not entry price.
                    # Using entry_price collapses P&L to zero on halted days,
                    # creating phantom drawdown spikes that can trigger the
                    # drawdown brake erroneously.
                    fallback = pos.last_close_price if pos.last_close_price > 0 else pos.entry_price
                    invested_value += pos.shares * fallback

            # Cash yield
            cash_yield_today = cash * daily_rf_rate
            cash += cash_yield_today
            cumulative_cash_yield += cash_yield_today

            equity = cash + invested_value
            peak_equity = max(peak_equity, equity)
            drawdown = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0.0
            if risk_engine_enabled and risk_state is not None:
                risk_state.update(
                    current_equity=equity,
                    brake_threshold=cfg_risk.drawdown_brake_threshold,
                    halt_threshold=cfg_risk.drawdown_halt_threshold,
                )

            cum_return = (equity / self.config.capital.initial_capital) - 1.0
            prev_equity = daily_records[-1].equity if daily_records else self.config.capital.initial_capital
            daily_ret = (equity / prev_equity) - 1.0 if prev_equity > 0 else 0.0

            # Strategy return excluding cash yield
            strategy_equity_excl = equity - cumulative_cash_yield
            strategy_ret_excl = (strategy_equity_excl / self.config.capital.initial_capital) - 1.0

            daily_records.append(DailyRecord(
                date=current_date,
                equity=equity,
                cash=cash,
                invested_capital=invested_value,
                gross_exposure=invested_value / equity if equity > 0 else 0.0,
                open_positions=len(open_positions),
                daily_return=daily_ret,
                cumulative_return=cum_return,
                drawdown_from_peak=drawdown,
                cash_yield_today=cash_yield_today,
                strategy_return_excl_cash=strategy_ret_excl,
                strategy_return_incl_cash=cum_return,
            ))

            # Progress reporting
            if (day_idx + 1) % 50 == 0 or day_idx == len(signal_dates) - 1:
                print(
                    f"  Day {day_idx+1}/{len(signal_dates)} | "
                    f"Equity: ${equity:,.0f} | "
                    f"Positions: {len(open_positions)} | "
                    f"Trades: {len(trade_log)} | "
                    f"DD: {drawdown:.1%}"
                )

        # ── Force-close any remaining positions at end of backtest ────
        # Track total exit friction for D1 fix: final_equity() must subtract this
        # because daily_records[-1].equity marks these positions at close (no friction).
        force_close_exit_friction = 0.0

        if open_positions:
            last_date = signal_dates[-1] if signal_dates else None
            if last_date is not None:
                last_date = pd.Timestamp(last_date)
                for ticker, pos in list(open_positions.items()):
                    prices = price_lookup.get((last_date, ticker))
                    if prices is None:
                        continue
                    raw_exit = prices["close"]
                    slippage_per_share = raw_exit * (cfg_costs.slippage_bps / 10_000)
                    spread_per_share = raw_exit * (cfg_costs.spread_bps / 10_000)
                    exit_price = raw_exit - slippage_per_share - spread_per_share

                    # ── P&L: same fix as regular exits ───────────────────────────
                    gross_pnl = (raw_exit - pos.raw_entry_price) * pos.shares

                    exit_friction = (slippage_per_share + spread_per_share) * pos.shares
                    entry_friction_cost = (pos.entry_price - pos.raw_entry_price) * pos.shares
                    total_costs = entry_friction_cost + exit_friction
                    net_pnl = gross_pnl - total_costs

                    force_close_exit_friction += exit_friction

                    trade_log.append(CompletedTrade(
                        trade_id=pos.trade_id,
                        ticker=ticker,
                        sector=pos.sector,
                        direction="LONG",
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=last_date,
                        exit_price=exit_price,
                        position_pct=pos.position_pct,
                        shares=pos.shares,
                        gross_pnl=gross_pnl,
                        entry_friction_cost=entry_friction_cost,
                        exit_friction_cost=exit_friction,
                        slippage_cost=slippage_per_share * pos.shares,
                        spread_cost=spread_per_share * pos.shares,
                        total_costs=total_costs,
                        net_pnl=net_pnl,
                        holding_days=pos.days_held,
                        exit_reason="backtest_end",
                        confidence_at_entry=pos.confidence_at_entry,
                    ))
                    if risk_engine_enabled and risk_state is not None:
                        risk_state.remove_stop(ticker)

        return BacktestResults(
            trade_log=trade_log,
            daily_records=daily_records,
            rejected_signals=rejected_signals,
            config=self.config,
            force_close_exit_friction=force_close_exit_friction,
            stop_loss_events=stop_loss_events,
        )

    # ── Helper methods ────────────────────────────────────────────

    def _validate_inputs(self, signal_df: pd.DataFrame, price_df: pd.DataFrame):
        """Check that input DataFrames have required columns."""
        sig_required = ["date", "ticker", "signal", "confidence"]
        sig_missing = [c for c in sig_required if c not in signal_df.columns]
        if sig_missing:
            raise ValueError(f"signal_df missing columns: {sig_missing}")

        price_required = ["Date", "Ticker", "Open", "High", "Low", "Close"]
        price_missing = [c for c in price_required if c not in price_df.columns]
        if price_missing:
            raise ValueError(f"price_df missing columns: {price_missing}")

    def _normalize_signal_and_price_dates(
        self, signal_df: pd.DataFrame, price_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return copies with tz-naive calendar dates so signals align with OHLC rows.

        yfinance and some parquet writers emit tz-aware ``Date`` columns; cached
        signals typically use naive dates — mixing them breaks timestamp compares
        in the simulation loop.
        """
        sig = signal_df.copy()
        price = price_df.copy()
        sig["date"] = pd.to_datetime(sig["date"])
        if getattr(sig["date"].dt, "tz", None) is not None:
            sig["date"] = (
                sig["date"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            )
        price["Date"] = pd.to_datetime(price["Date"])
        if getattr(price["Date"].dt, "tz", None) is not None:
            price["Date"] = (
                price["Date"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
            )
        return sig, price

    def _build_price_lookup(
        self, price_df: pd.DataFrame
    ) -> Dict[Tuple[pd.Timestamp, str], Dict[str, float]]:
        """Build a fast (date, ticker) → prices dictionary.

        Uses to_dict('records') instead of iterrows() — approximately 5× faster
        for large DataFrames. Critical for scale: at 10 years of data, price_df
        will have ~130k+ rows. iterrows() would noticeably slow initialization.
        """
        price_df = price_df.copy()
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        records = price_df.to_dict("records")
        lookup = {}
        for row in records:
            key = (row["Date"], row["Ticker"])
            lookup[key] = {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
            }
        return lookup

    def _next_trading_day(
        self, current_date: pd.Timestamp, all_dates: list
    ) -> Optional[pd.Timestamp]:
        """Find the next trading day after current_date."""
        for d in all_dates:
            d = pd.Timestamp(d)
            if d > current_date:
                return d
        return None

    def _get_ticker_history(
        self,
        price_history_by_ticker: Dict[str, pd.DataFrame],
        ticker: str,
        as_of_date: pd.Timestamp,
        n_rows: int,
    ) -> pd.DataFrame:
        """Return up-to-date trailing OHLC history for one ticker.

        Uses binary search (O(log N)) rather than boolean mask (O(N)).
        Safe because price_history_by_ticker DataFrames are pre-sorted by Date.
        """
        ticker_history = price_history_by_ticker.get(ticker)
        if ticker_history is None:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        dates = ticker_history["Date"].values
        idx = dates.searchsorted(np.datetime64(as_of_date, "ns"), side="right")
        if idx == 0:
            return ticker_history.iloc[0:0].copy()
        start_idx = max(0, idx - n_rows)
        return ticker_history.iloc[start_idx:idx].copy()

    def _advance_trading_days(
        self, start_date: pd.Timestamp, n_days: int, all_dates: list
    ) -> pd.Timestamp:
        """Advance n trading days from start_date.

        D2 fix: when the price data ends before the cooldown completes (e.g. year-end
        or multi-year boundary), return a calendar-based estimate instead of silently
        shortening the cooldown to the last available date.  1 trading day ≈ 1.4
        calendar days (accounts for weekends; rounds up slightly so the cooldown is
        never shorter than intended).

        This matters most in multi-year backtests where Phase 2 adds stop-losses,
        creating more cooldown events near year boundaries.
        """
        future_dates = [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d) > start_date]
        if len(future_dates) >= n_days:
            return future_dates[n_days - 1]
        # Insufficient trading days remain in the loaded price data.
        # Estimate the cooldown end-date using calendar days (1 trading day ≈ 1.4 cal days).
        remaining = n_days - len(future_dates)
        estimated_end = (future_dates[-1] if future_dates else start_date) + \
                        pd.Timedelta(days=int(remaining * 1.4) + 1)
        return estimated_end


# ============================================================
# BACKTEST RESULTS
# ============================================================

class BacktestResults:
    """Container for backtest output with analysis methods."""

    def __init__(
        self,
        trade_log: List[CompletedTrade],
        daily_records: List[DailyRecord],
        rejected_signals: List[RejectedSignal],
        config: TradingConfig,
        force_close_exit_friction: float = 0.0,
        stop_loss_events: Optional[List[StopLossEvent]] = None,
    ):
        self.trade_log = trade_log
        self.daily_records = daily_records
        self.rejected_signals = rejected_signals
        self.stop_loss_events = stop_loss_events or []
        self.config = config
        # D1 fix: daily_records[-1].equity marks force-closed positions at close (MTM,
        # no friction).  Store the exit friction so final_equity() returns the correct
        # post-friction value that matches actual cash realised.
        self._force_close_exit_friction = force_close_exit_friction

        # Convert to DataFrames for easy analysis
        self.trades_df = self._trades_to_df()
        self.equity_df = self._equity_to_df()
        self.rejected_df = self._rejected_to_df()
        self.stop_events_df = self._stop_events_to_df()

    def _trades_to_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        records = []
        for t in self.trade_log:
            records.append({
                "trade_id": t.trade_id,
                "ticker": t.ticker,
                "sector": t.sector,
                "direction": t.direction,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "position_pct": t.position_pct,
                "shares": t.shares,
                "gross_pnl": t.gross_pnl,
                "entry_friction_cost": t.entry_friction_cost,
                "exit_friction_cost": t.exit_friction_cost,
                "slippage_cost": t.slippage_cost,
                "spread_cost": t.spread_cost,
                "total_costs": t.total_costs,
                "net_pnl": t.net_pnl,
                "holding_days": t.holding_days,
                "exit_reason": t.exit_reason,
                "confidence_at_entry": t.confidence_at_entry,
            })
        return pd.DataFrame(records)

    def _equity_to_df(self) -> pd.DataFrame:
        if not self.daily_records:
            return pd.DataFrame()
        records = []
        for d in self.daily_records:
            records.append({
                "date": d.date,
                "equity": d.equity,
                "cash": d.cash,
                "invested_capital": d.invested_capital,
                "gross_exposure": d.gross_exposure,
                "open_positions": d.open_positions,
                "daily_return": d.daily_return,
                "cumulative_return": d.cumulative_return,
                "drawdown_from_peak": d.drawdown_from_peak,
                "cash_yield_today": d.cash_yield_today,
                "strategy_return_excl_cash": d.strategy_return_excl_cash,
                "strategy_return_incl_cash": d.strategy_return_incl_cash,
            })
        return pd.DataFrame(records)

    def _rejected_to_df(self) -> pd.DataFrame:
        if not self.rejected_signals:
            return pd.DataFrame()
        records = []
        for r in self.rejected_signals:
            records.append({
                "date": r.date,
                "ticker": r.ticker,
                "signal": r.signal,
                "confidence": r.confidence,
                "rejection_reason": r.rejection_reason,
                "rejection_layer": r.rejection_layer,
            })
        return pd.DataFrame(records)

    def _stop_events_to_df(self) -> pd.DataFrame:
        if not self.stop_loss_events:
            return pd.DataFrame()
        records = []
        for event in self.stop_loss_events:
            records.append({
                "ticker": event.ticker,
                "trigger_date": event.trigger_date,
                "stop_price": event.stop_price,
                "trigger_low": event.trigger_low,
                "entry_price": event.entry_price,
                "exit_price": event.exit_price,
                "gap_through": event.gap_through,
                "atr_at_entry": event.atr_at_entry,
            })
        return pd.DataFrame(records)

    # ── Analysis methods ──────────────────────────────────────────

    def total_trades(self) -> int:
        return len(self.trade_log)

    def win_rate(self) -> Optional[float]:
        """Fraction of trades with positive net P&L."""
        if not self.trade_log:
            return None
        wins = sum(1 for t in self.trade_log if t.net_pnl > 0)
        return wins / len(self.trade_log)

    def net_expectancy(self) -> Optional[float]:
        """Average net P&L per trade (in dollars)."""
        if not self.trade_log:
            return None
        return sum(t.net_pnl for t in self.trade_log) / len(self.trade_log)

    def profit_factor(self) -> Optional[float]:
        """Gross wins / gross losses. >1.0 means profitable."""
        if not self.trade_log:
            return None
        gross_wins = sum(t.net_pnl for t in self.trade_log if t.net_pnl > 0)
        gross_losses = abs(sum(t.net_pnl for t in self.trade_log if t.net_pnl < 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else None
        return gross_wins / gross_losses

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown."""
        if not self.daily_records:
            return 0.0
        return max(d.drawdown_from_peak for d in self.daily_records)

    def sharpe_ratio(self) -> Optional[float]:
        """Annualized Sharpe ratio (excess return over risk-free / annualized vol)."""
        if len(self.daily_records) < 30:
            return None
        returns = [d.daily_return for d in self.daily_records]
        daily_rf = self.config.costs.risk_free_annual_rate / 252
        excess = [r - daily_rf for r in returns]
        if np.std(excess) == 0:
            return None
        return (np.mean(excess) / np.std(excess)) * np.sqrt(252)

    def total_costs(self) -> float:
        """Total transaction costs across all trades."""
        return sum(t.total_costs for t in self.trade_log)

    def total_cash_yield(self) -> float:
        """Total cash yield earned on idle capital."""
        return sum(d.cash_yield_today for d in self.daily_records)

    def avg_holding_days(self) -> Optional[float]:
        """Average holding period in trading days."""
        if not self.trade_log:
            return None
        return np.mean([t.holding_days for t in self.trade_log])

    def avg_idle_cash_pct(self) -> Optional[float]:
        """Average percentage of equity sitting in cash."""
        if not self.daily_records:
            return None
        return np.mean([d.cash / d.equity if d.equity > 0 else 1.0 for d in self.daily_records])

    def final_equity(self) -> float:
        """Final portfolio equity after all costs.

        D1 fix: daily_records[-1].equity marks force-closed positions at the last
        available close price with no exit friction applied (the daily loop ended
        before force-close executed).  Subtract the force-close exit friction so
        the reported final equity matches actual cash that would be realised.
        """
        if not self.daily_records:
            return self.config.capital.initial_capital
        return self.daily_records[-1].equity - self._force_close_exit_friction

    def annualized_return(self) -> Optional[float]:
        """Annualized return based on total period."""
        if len(self.daily_records) < 2:
            return None
        total_days = len(self.daily_records)
        total_return = self.final_equity() / self.config.capital.initial_capital
        return total_return ** (252 / total_days) - 1

    def rejection_summary(self) -> Dict[str, int]:
        """Count rejections by layer."""
        if not self.rejected_signals:
            return {}
        from collections import Counter
        return dict(Counter(r.rejection_layer for r in self.rejected_signals))

    def summary(self) -> str:
        """Human-readable performance summary."""
        lines = [
            "",
            "=" * 60,
            "  BACKTEST RESULTS SUMMARY",
            "=" * 60,
            "",
            f"  Period: {self.daily_records[0].date.date()} to {self.daily_records[-1].date.date()}" if self.daily_records else "  No data",
            f"  Starting capital:    ${self.config.capital.initial_capital:>10,.0f}",
            f"  Final equity:        ${self.final_equity():>10,.0f}",
            "",
            "  ── Performance ──────────────────────────────────────",
        ]

        ann_ret = self.annualized_return()
        lines.append(f"  Annualized return:   {ann_ret:>10.1%}" if ann_ret is not None else "  Annualized return:   N/A")

        sharpe = self.sharpe_ratio()
        lines.append(f"  Sharpe ratio:        {sharpe:>10.2f}" if sharpe is not None else "  Sharpe ratio:        N/A")

        lines.append(f"  Max drawdown:        {self.max_drawdown():>10.1%}")

        wr = self.win_rate()
        lines.append(f"  Win rate:            {wr:>10.1%}" if wr is not None else "  Win rate:            N/A")

        pf = self.profit_factor()
        lines.append(f"  Profit factor:       {pf:>10.2f}" if pf is not None else "  Profit factor:       N/A")

        ne = self.net_expectancy()
        lines.append(f"  Net expectancy:      ${ne:>9.2f}/trade" if ne is not None else "  Net expectancy:      N/A")

        lines.extend([
            "",
            "  ── Trade Statistics ──────────────────────────────────",
            f"  Total trades:        {self.total_trades():>10}",
        ])

        avg_hold = self.avg_holding_days()
        lines.append(f"  Avg holding days:    {avg_hold:>10.1f}" if avg_hold is not None else "  Avg holding days:    N/A")

        lines.append(f"  Total costs:         ${self.total_costs():>10,.2f}")
        lines.append(f"  Total cash yield:    ${self.total_cash_yield():>10,.2f}")

        idle = self.avg_idle_cash_pct()
        lines.append(f"  Avg idle cash:       {idle:>10.1%}" if idle is not None else "  Avg idle cash:       N/A")

        # Rejection summary
        rej = self.rejection_summary()
        if rej:
            lines.extend([
                "",
                "  ── Rejections by Layer ───────────────────────────────",
            ])
            for layer, count in sorted(rej.items(), key=lambda x: -x[1]):
                lines.append(f"  {layer:<25} {count:>6}")

        # Exit reason breakdown
        if self.trade_log:
            from collections import Counter
            exit_reasons = Counter(t.exit_reason for t in self.trade_log)
            lines.extend([
                "",
                "  ── Exit Reasons ─────────────────────────────────────",
            ])
            for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  {reason:<25} {count:>6}")

        if self.stop_loss_events:
            lines.append(f"  Stop-loss events:    {len(self.stop_loss_events):>10}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, output_dir: str = "results"):
        """Save all results to CSV files."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        if not self.trades_df.empty:
            self.trades_df.to_csv(out / "backtest_trades.csv", index=False)
        if not self.equity_df.empty:
            self.equity_df.to_csv(out / "backtest_equity.csv", index=False)
        if not self.rejected_df.empty:
            self.rejected_df.to_csv(out / "backtest_rejected.csv", index=False)
        if not self.stop_events_df.empty:
            self.stop_events_df.to_csv(out / "backtest_stop_events.csv", index=False)

        # Save summary text
        with open(out / "backtest_summary.txt", "w") as f:
            f.write(self.summary())

        print(f"\n  Results saved to {out}/")
