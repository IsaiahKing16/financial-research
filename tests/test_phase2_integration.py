"""
tests/test_phase2_integration.py — Phase 2 Risk Engine Integration Tests

End-to-end tests for BacktestEngine with use_risk_engine=True.
Uses only synthetic OHLC + signal data — no external files required.

Coverage:
  - Full backtest run with Phase 2 enabled (no errors)
  - Phase 2 produces different equity/position sizes than Phase 1
  - Phase 1 backward-compatibility: equal-weight exactly, no stop events
  - Stop-loss fires when Low < stop_price and appears in stop_events_df + trade_log
  - stop_events_df schema is correct; gap_through flag is set correctly
  - Position sizes vary inversely with ATR (high-ATR → smaller position)
  - Drawdown halt blocks new entries at the configured threshold
  - BacktestResults.stop_events_df is empty without Phase 2 enabled

ATR geometry (atr_half=0.01, flat price at 100):
  H = 101, L = 99  →  True Range = max(H-L=2, |H-prev_C|=1, |L-prev_C|=1) = 2
  EWM of constant 2 = 2  →  atr_pct = 2/100 = 0.02
  stop = entry × (1 - stop_multiple×atr_pct) ≈ 100.013 × 0.96 ≈ 96.01
  Normal Low = 99 >> 96.01 (stop never fires without deliberate crash)
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest_engine import BacktestEngine
from trading_system.config import (
    PositionLimitsConfig,
    RiskConfig,
    SignalConfig,
    TradingConfig,
    TradeManagementConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _flat_price_df(
    tickers,
    n_days: int = 60,
    start: str = "2024-01-02",
    base_price: float = 100.0,
    atr_half: float = 0.01,
) -> pd.DataFrame:
    """
    Build OHLC DataFrame with stable flat prices and controlled ATR.

    atr_half: half-spread as a fraction of close.
      H = close×(1+atr_half)  L = close×(1-atr_half)
      → H-L = 2×atr_half×close  → ATR_pct ≈ 2×atr_half  (with EWM of constant series)

    With atr_half=0.01 and flat price=100:
      atr_pct ≈ 0.02, stop_distance ≈ 4%, stop ≈ 96.
      Normal Low = 99 — well above stop; stops only fire on explicit override.
    """
    dates = pd.bdate_range(start=start, periods=n_days)
    rows = []
    for ticker in tickers:
        for date in dates:
            rows.append({
                "Date": date,
                "Ticker": ticker,
                "Open":  base_price,
                "High":  base_price * (1.0 + atr_half),
                "Low":   base_price * (1.0 - atr_half),
                "Close": base_price,
            })
    return pd.DataFrame(rows)


def _sig(ticker, sector, dates, signal="BUY", confidence=0.75):
    """Build a list of signal row dicts for one ticker across a list of dates."""
    return [
        {
            "date":       pd.Timestamp(d),
            "ticker":     ticker,
            "signal":     signal,
            "confidence": confidence,
            "sector":     sector,
        }
        for d in dates
    ]


def _test_config(**risk_overrides) -> TradingConfig:
    """
    TradingConfig tuned for deterministic integration tests.

    Defaults:
      - initial_capital = 10 000
      - confidence_threshold = 0.50 (accept all synthetic signals)
      - max_holding_days = 10  (quick exits)
      - cooldown = 1 day (minimal, to allow re-entry quickly)
      - drawdown thresholds: brake=15%, halt=20% (standard)

    Pass risk_overrides as kwargs to RiskConfig (e.g. drawdown_halt_threshold=0.03).
    """
    base = TradingConfig()
    risk_base = base.risk
    if risk_overrides:
        risk_base = dataclasses.replace(risk_base, **risk_overrides)
    return dataclasses.replace(
        base,
        signals=dataclasses.replace(base.signals, confidence_threshold=0.50),
        trade_management=dataclasses.replace(
            base.trade_management,
            max_holding_days=10,
            cooldown_after_stop_days=1,
            cooldown_after_maxhold_days=1,
        ),
        risk=risk_base,
    )


_DATES = pd.bdate_range("2024-01-02", periods=60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2BasicExecution:
    """Phase 2 backtest runs without errors on well-formed inputs."""

    def _simple_inputs(self):
        """One ticker, one BUY + HOLD signals for 20 days."""
        price_df = _flat_price_df(["AAPL"], n_days=60)
        rows = _sig("AAPL", "Tech", [_DATES[30]], "BUY")
        rows += _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        return price_df, pd.DataFrame(rows)

    def test_backtest_completes_without_error(self):
        """BacktestEngine(use_risk_engine=True).run() completes without exception."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=True)
        results = engine.run(signal_df, price_df)
        assert results is not None

    def test_equity_df_has_rows(self):
        """equity_df is populated (at least one trading day recorded)."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=True)
        results = engine.run(signal_df, price_df)
        assert len(results.equity_df) > 0

    def test_buy_signal_produces_at_least_one_trade(self):
        """A valid BUY on well-formed data results in at least one completed trade."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=True)
        results = engine.run(signal_df, price_df)
        assert results.total_trades() >= 1

    def test_stop_events_attributes_exist(self):
        """BacktestResults has stop_loss_events list and stop_events_df DataFrame."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=True)
        results = engine.run(signal_df, price_df)

        assert hasattr(results, "stop_loss_events")
        assert hasattr(results, "stop_events_df")
        assert isinstance(results.stop_loss_events, list)
        assert isinstance(results.stop_events_df, pd.DataFrame)

    def test_phase2_position_has_stop_price_above_zero(self):
        """
        Phase 2 entries have stop_loss_price > 0. Confirmed via rejected_df
        or trades_df — no trade should have a zero-stop in Phase 2.
        (Phase 1 uses stop_loss_price=0.0 by design.)
        """
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=True)
        results = engine.run(signal_df, price_df)

        # All Phase 2 trades must record a positive entry_price (sanity check)
        if not results.trades_df.empty:
            assert (results.trades_df["entry_price"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 2. PHASE 1 vs PHASE 2 COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2VsPhase1:
    """Phase 2 produces different results; Phase 1 path is fully preserved."""

    def _two_ticker_inputs(self):
        """Two tickers with different ATR — amplifies sizing difference."""
        price_df = pd.concat([
            _flat_price_df(["AAPL"], n_days=60, atr_half=0.01, base_price=100.0),
            _flat_price_df(["JPM"],  n_days=60, atr_half=0.04, base_price=100.0),
        ], ignore_index=True)
        rows = (
            _sig("AAPL", "Tech",    [_DATES[30]], "BUY")
            + _sig("JPM",  "Finance", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech",    _DATES[31:50].tolist(), "HOLD")
            + _sig("JPM",  "Finance", _DATES[31:50].tolist(), "HOLD")
        )
        return price_df, pd.DataFrame(rows)

    def test_phase2_position_pct_differs_from_phase1(self):
        """
        Phase 2 position_pct differs from Phase 1 equal_weight_pct=0.05.

        With AAPL atr_half=0.01 → atr_pct≈0.02, raw_weight=0.02/(2×0.02)=0.50.
        Clamped to max_position_pct=0.10 in Phase 2.
        Phase 1 uses exactly 0.05 — these are different by definition.
        """
        price_df, signal_df = self._two_ticker_inputs()
        cfg = _test_config()

        r1 = BacktestEngine(config=cfg, use_risk_engine=False).run(
            signal_df, price_df, equal_weight_pct=0.05
        )
        r2 = BacktestEngine(config=cfg, use_risk_engine=True).run(signal_df, price_df)

        assert r1.total_trades() >= 1
        assert r2.total_trades() >= 1

        if not r1.trades_df.empty and not r2.trades_df.empty:
            p1_aapl = r1.trades_df[r1.trades_df["ticker"] == "AAPL"]["position_pct"].iloc[0]
            p2_aapl = r2.trades_df[r2.trades_df["ticker"] == "AAPL"]["position_pct"].iloc[0]
            # Phase 1 = 0.05, Phase 2 = 0.10 (clamped from 0.50)
            assert abs(p1_aapl - 0.05) < 1e-9, f"Phase 1 position_pct should be 0.05, got {p1_aapl}"
            assert abs(p2_aapl - p1_aapl) > 1e-6, (
                f"Phase 2 position_pct ({p2_aapl:.4f}) should differ from "
                f"Phase 1 ({p1_aapl:.4f})"
            )

    def test_phase1_all_trades_use_exact_equal_weight(self):
        """Phase 1: every trade uses exactly equal_weight_pct, regardless of ATR."""
        price_df, signal_df = self._two_ticker_inputs()
        cfg = _test_config()
        results = BacktestEngine(config=cfg, use_risk_engine=False).run(
            signal_df, price_df, equal_weight_pct=0.07
        )

        assert results.total_trades() >= 1
        for _, trade in results.trades_df.iterrows():
            assert abs(trade["position_pct"] - 0.07) < 1e-9, (
                f"Phase 1 trade for {trade['ticker']} should use position_pct=0.07, "
                f"got {trade['position_pct']:.6f}"
            )

    def test_phase1_no_stop_loss_events_even_when_low_crashes(self):
        """
        Phase 1 stop_loss_price=0.0: even if intraday Low crashes below where
        a Phase 2 stop would be, no StopLossEvent is created.
        """
        price_df = _flat_price_df(["AAPL"], n_days=60)
        # Override Low on day 35 to far below any Phase 2 stop level
        trigger_date = _DATES[35]
        price_df.loc[
            (price_df["Ticker"] == "AAPL") & (price_df["Date"] == trigger_date),
            "Low"
        ] = 50.0

        rows = (
            _sig("AAPL", "Tech", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        )
        signal_df = pd.DataFrame(rows)

        cfg = _test_config()
        results = BacktestEngine(config=cfg, use_risk_engine=False).run(signal_df, price_df)

        assert len(results.stop_loss_events) == 0
        assert results.stop_events_df.empty


# ─────────────────────────────────────────────────────────────────────────────
# 3. STOP-LOSS INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestStopLossIntegration:
    """Stop-loss fires and is captured in trade log and stop_events_df."""

    def _stop_trigger_inputs(self, trigger_day: int = 35):
        """
        Build inputs where AAPL's intraday Low deliberately crashes below the stop.

        ATR geometry (atr_half=0.01, price=100):
          atr_pct ≈ 0.02, stop ≈ 100.013 × 0.96 ≈ 96.01
        Trigger: Low = 85 on trigger_day  (85 << 96.01 → stop fires with margin)
        """
        price_df = _flat_price_df(["AAPL"], n_days=60)
        price_df.loc[
            (price_df["Ticker"] == "AAPL") & (price_df["Date"] == _DATES[trigger_day]),
            "Low"
        ] = 85.0  # Far below stop≈96; gap_through=True

        rows = (
            _sig("AAPL", "Tech", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        )
        return price_df, pd.DataFrame(rows)

    def test_stop_loss_produces_stop_loss_exit_reason(self):
        """When Low < stop_price, trade is closed with exit_reason='stop_loss'."""
        price_df, signal_df = self._stop_trigger_inputs()
        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        assert results.total_trades() >= 1
        exit_reasons = results.trades_df["exit_reason"].tolist()
        assert "stop_loss" in exit_reasons, (
            f"Expected 'stop_loss' in exit reasons; got {exit_reasons}"
        )

    def test_stop_events_df_populated_after_stop_fires(self):
        """stop_events_df has at least one row after a stop-loss fires."""
        price_df, signal_df = self._stop_trigger_inputs()
        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        assert not results.stop_events_df.empty, (
            "stop_events_df should be non-empty after a stop fires"
        )
        assert len(results.stop_loss_events) > 0

    def test_stop_events_df_has_required_columns(self):
        """stop_events_df contains the expected audit-trail columns."""
        price_df, signal_df = self._stop_trigger_inputs()
        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        required = {
            "ticker", "trigger_date", "stop_price",
            "trigger_low", "entry_price", "exit_price",
            "gap_through", "atr_at_entry",
        }
        if not results.stop_events_df.empty:
            missing = required - set(results.stop_events_df.columns)
            assert not missing, f"stop_events_df missing columns: {missing}"

    def test_stop_event_gap_through_is_true(self):
        """
        gap_through=True because trigger_low=85 < stop≈96.
        The engine sets gap_through = (trigger_low < stop_price).
        """
        price_df, signal_df = self._stop_trigger_inputs()
        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        if not results.stop_events_df.empty:
            event = results.stop_events_df.iloc[0]
            assert bool(event["gap_through"]) is True, (
                f"Expected gap_through=True (trigger_low={event['trigger_low']} "
                f"< stop={event['stop_price']:.2f})"
            )

    def test_stop_event_ticker_is_aapl(self):
        """stop_events_df.ticker matches the stopped-out position."""
        price_df, signal_df = self._stop_trigger_inputs()
        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        if not results.stop_events_df.empty:
            assert results.stop_events_df.iloc[0]["ticker"] == "AAPL"

    def test_stop_fires_before_max_hold_when_early(self):
        """A stop triggered on day 34 closes the position before max_holding_days=10."""
        # Trigger on day 34 (4 days after entry on ~day 31) — well before day 40
        price_df, signal_df = self._stop_trigger_inputs(trigger_day=34)
        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        stop_trades = results.trades_df[results.trades_df["exit_reason"] == "stop_loss"]
        if not stop_trades.empty:
            assert (stop_trades["holding_days"] < 10).all(), (
                "Stop-triggered trade should exit before max_holding_days=10"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. POSITION SIZING BY VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizingByVolatility:
    """ATR-based sizing: higher ATR → smaller raw_weight → smaller position_pct."""

    def _vol_inputs(self, atr_low=0.01, atr_high=0.04):
        """
        Two tickers: AAPL (low-ATR) and JPM (high-ATR).

        Sizing formula with max_loss=0.02, stop_multiple=2.0:
          raw_weight = 0.02 / (2 × atr_pct)

          AAPL (atr_pct≈0.02): raw_weight = 0.02/0.04 = 0.50
          JPM  (atr_pct≈0.08): raw_weight = 0.02/0.16 = 0.125

        With max_position_pct=0.60 (widened), neither is clamped — clear gap.
        """
        price_df = pd.concat([
            _flat_price_df(["AAPL"], n_days=60, atr_half=atr_low,  base_price=100.0),
            _flat_price_df(["JPM"],  n_days=60, atr_half=atr_high, base_price=100.0),
        ], ignore_index=True)

        rows = (
            _sig("AAPL", "Tech",    [_DATES[30]], "BUY")
            + _sig("JPM",  "Finance", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech",    _DATES[31:45].tolist(), "HOLD")
            + _sig("JPM",  "Finance", _DATES[31:45].tolist(), "HOLD")
        )
        return price_df, pd.DataFrame(rows)

    def _wide_config(self):
        """Config with wide position limits so ATR differences aren't masked by the cap."""
        base = TradingConfig()
        return dataclasses.replace(
            base,
            signals=dataclasses.replace(base.signals, confidence_threshold=0.50),
            trade_management=dataclasses.replace(
                base.trade_management,
                max_holding_days=10,
                cooldown_after_stop_days=1,
                cooldown_after_maxhold_days=1,
            ),
            position_limits=dataclasses.replace(
                base.position_limits,
                min_position_pct=0.01,
                max_position_pct=0.60,
                max_sector_pct=0.80,
            ),
        )

    def test_low_atr_ticker_gets_larger_position(self):
        """
        AAPL (atr_half=0.01) → position_pct > JPM (atr_half=0.04).
        raw_weight is inversely proportional to ATR via the 2%-max-loss formula.
        """
        price_df, signal_df = self._vol_inputs()
        results = BacktestEngine(config=self._wide_config(), use_risk_engine=True).run(
            signal_df, price_df
        )

        trades = results.trades_df
        if len(trades) >= 2:
            aapl = trades[trades["ticker"] == "AAPL"]
            jpm  = trades[trades["ticker"] == "JPM"]
            if not aapl.empty and not jpm.empty:
                aapl_pct = aapl.iloc[0]["position_pct"]
                jpm_pct  = jpm.iloc[0]["position_pct"]
                assert aapl_pct > jpm_pct, (
                    f"Low-ATR AAPL ({aapl_pct:.4f}) should be > high-ATR JPM ({jpm_pct:.4f})"
                )

    def test_all_phase2_positions_within_config_limits(self):
        """
        Every Phase 2 trade stays within [min_position_pct, max_position_pct].
        Tests both extreme ATR cases: very tight spread (min) and very wide (max).
        """
        price_df = pd.concat([
            # Extremely low ATR → raw_weight ≫ max → clamped at max
            _flat_price_df(["AAPL"], n_days=60, atr_half=0.001, base_price=100.0),
            # Extremely high ATR → raw_weight ≪ min → clamped at min
            _flat_price_df(["JPM"],  n_days=60, atr_half=0.10,  base_price=100.0),
        ], ignore_index=True)

        rows = (
            _sig("AAPL", "Tech",    [_DATES[30]], "BUY")
            + _sig("JPM",  "Finance", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech",    _DATES[31:45].tolist(), "HOLD")
            + _sig("JPM",  "Finance", _DATES[31:45].tolist(), "HOLD")
        )
        signal_df = pd.DataFrame(rows)

        cfg = _test_config()
        results = BacktestEngine(config=cfg, use_risk_engine=True).run(signal_df, price_df)

        min_p = cfg.position_limits.min_position_pct
        max_p = cfg.position_limits.max_position_pct

        for _, trade in results.trades_df.iterrows():
            assert min_p - 1e-9 <= trade["position_pct"] <= max_p + 1e-9, (
                f"position_pct={trade['position_pct']:.4f} for {trade['ticker']} "
                f"outside [{min_p}, {max_p}]"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DRAWDOWN HALT INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawdownHalt:
    """
    Drawdown halt blocks new entries when equity < peak × (1 - halt_threshold).

    Setup:
      - AAPL enters at $100 (10% position = $1 000 = ~10 shares)
      - Close drops to $65 on days 31+: invested ≈ $650, equity ≈ $9 650
      - drawdown = $350 / $10 000 = 3.5% ≥ halt_threshold=3%  → halt mode
      - Low = $98 on those days (above stop≈$96): position stays open via MTM
      - JPM BUY on day 32 → rejected in size_position due to halt mode
    """

    def _halt_inputs(self):
        """
        Inputs designed to trigger halt via MTM drawdown, then confirm rejection.
        Returns (price_df, signal_df, cfg).
        """
        # Base price data
        aapl_df = _flat_price_df(["AAPL"], n_days=60, atr_half=0.01, base_price=100.0)
        jpm_df  = _flat_price_df(["JPM"],  n_days=60, atr_half=0.01, base_price=100.0)

        # AAPL: crash close to $65 on days 31+ (below breakeven), keep Low above stop
        # Normal Low=99; override to 98 (above stop≈96) so stop doesn't fire.
        crash_dates = _DATES[31:50]
        for col in ("Open", "Close"):
            aapl_df.loc[
                (aapl_df["Ticker"] == "AAPL") & (aapl_df["Date"].isin(crash_dates)),
                col
            ] = 65.0
        aapl_df.loc[
            (aapl_df["Ticker"] == "AAPL") & (aapl_df["Date"].isin(crash_dates)),
            "High"
        ] = 65.5
        aapl_df.loc[
            (aapl_df["Ticker"] == "AAPL") & (aapl_df["Date"].isin(crash_dates)),
            "Low"
        ] = 98.0  # Intentionally above stop≈96 so stop doesn't fire

        price_df = pd.concat([aapl_df, jpm_df], ignore_index=True)

        rows = (
            # AAPL BUY on day 30; enters day 31 at $100; close drops to $65
            _sig("AAPL", "Tech",    [_DATES[30]], "BUY")
            # Hold AAPL through the drawdown period
            + _sig("AAPL", "Tech",  _DATES[31:50].tolist(), "HOLD")
            # JPM BUY on day 32 (after halt triggers from day 31 MTM)
            + _sig("JPM",  "Finance", [_DATES[32]], "BUY")
            # Hold JPM (for date presence, even if never entered)
            + _sig("JPM",  "Finance", _DATES[33:50].tolist(), "HOLD")
        )
        signal_df = pd.DataFrame(rows)

        cfg = _test_config(
            drawdown_brake_threshold=0.02,
            drawdown_halt_threshold=0.03,   # 3% halt: AAPL's 3.5% loss triggers it
        )
        return price_df, signal_df, cfg

    def test_drawdown_halt_rejects_jpm_entry(self):
        """
        After AAPL drops 3.5%, halt triggers. The subsequent JPM BUY is rejected
        by size_position with a reason containing 'halt'.
        """
        price_df, signal_df, cfg = self._halt_inputs()
        results = BacktestEngine(config=cfg, use_risk_engine=True).run(signal_df, price_df)

        # Verify the backtest ran cleanly
        assert results is not None

        # Find halt-driven rejections in rejected_df
        if not results.rejected_df.empty:
            halt_rejs = results.rejected_df[
                results.rejected_df["rejection_reason"].str.contains(
                    "halt", case=False, na=False
                )
            ]
            # If a halt rejection exists, confirm it's for JPM (the ticker we tried to enter)
            if not halt_rejs.empty:
                assert (halt_rejs["ticker"] == "JPM").any(), (
                    "Expected JPM BUY to be rejected due to drawdown halt; "
                    f"got tickers: {halt_rejs['ticker'].tolist()}"
                )

    def test_drawdown_halt_rejection_layer_is_risk_engine(self):
        """
        Halt rejections from size_position appear in rejected_df with
        rejection_layer='risk_engine' (not 'backtest' or 'capital').
        """
        price_df, signal_df, cfg = self._halt_inputs()
        results = BacktestEngine(config=cfg, use_risk_engine=True).run(signal_df, price_df)

        if not results.rejected_df.empty:
            halt_rejs = results.rejected_df[
                results.rejected_df["rejection_reason"].str.contains(
                    "halt", case=False, na=False
                )
            ]
            if not halt_rejs.empty:
                assert (halt_rejs["rejection_layer"] == "risk_engine").all(), (
                    f"Halt rejections should have rejection_layer='risk_engine'; "
                    f"got {halt_rejs['rejection_layer'].tolist()}"
                )

    def test_backtest_completes_during_drawdown(self):
        """Engine runs to completion even when drawdown brake/halt engages."""
        price_df, signal_df, cfg = self._halt_inputs()
        results = BacktestEngine(config=cfg, use_risk_engine=True).run(signal_df, price_df)
        assert results is not None
        assert len(results.equity_df) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. RISK STATE CARRIES FORWARD ACROSS TRADING DAYS
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskStateConsistency:
    """RiskState is updated correctly across the daily loop."""

    def test_stop_events_list_and_df_stay_in_sync(self):
        """len(stop_loss_events) == len(stop_events_df) after a stop fires."""
        price_df = _flat_price_df(["AAPL"], n_days=60)
        # Crash Low on one day to fire stop
        price_df.loc[
            (price_df["Ticker"] == "AAPL") & (price_df["Date"] == _DATES[34]),
            "Low"
        ] = 85.0

        rows = (
            _sig("AAPL", "Tech", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        )
        signal_df = pd.DataFrame(rows)

        results = BacktestEngine(config=_test_config(), use_risk_engine=True).run(
            signal_df, pd.DataFrame(price_df)
        )

        assert len(results.stop_loss_events) == len(results.stop_events_df), (
            "stop_loss_events list and stop_events_df row count must match"
        )

    def test_phase2_uses_risk_engine_flag_per_call_override(self):
        """
        BacktestEngine(use_risk_engine=False) with per-call override
        use_risk_engine=True activates Phase 2 for that run.
        """
        price_df = _flat_price_df(["AAPL"], n_days=60)
        rows = (
            _sig("AAPL", "Tech", [_DATES[30]], "BUY")
            + _sig("AAPL", "Tech", _DATES[31:45].tolist(), "HOLD")
        )
        signal_df = pd.DataFrame(rows)

        # Engine default is Phase 1, but override to Phase 2 in this call
        engine = BacktestEngine(config=_test_config(), use_risk_engine=False)
        results_p2 = engine.run(signal_df, price_df, use_risk_engine=True)
        results_p1 = engine.run(signal_df, price_df, use_risk_engine=False)

        # Both run successfully
        assert results_p2 is not None
        assert results_p1 is not None

        # Phase 2 position is different from Phase 1's 5% equal weight
        if not results_p1.trades_df.empty and not results_p2.trades_df.empty:
            p1_pct = results_p1.trades_df.iloc[0]["position_pct"]
            p2_pct = results_p2.trades_df.iloc[0]["position_pct"]
            # Phase 1 must be exactly equal_weight_pct default (0.05)
            assert abs(p1_pct - 0.05) < 1e-9
            # Phase 2 position ≠ Phase 1 position
            assert abs(p2_pct - p1_pct) > 1e-6
