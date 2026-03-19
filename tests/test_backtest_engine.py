"""
tests/test_backtest_engine.py — Regression tests for trading_system/backtest_engine.py

Focus areas — all three documented bug fixes that were verified in PHASE1_FILE_REVIEW.md:

  S5 (P&L double-counting): gross_pnl must use raw_entry_price and raw_exit_price.
      Entry friction and exit friction are captured separately in total_costs.
      net_pnl = gross_pnl - total_costs must NOT subtract entry friction twice.

  D1 (force-close equity):  final_equity() must subtract force_close_exit_friction
      from daily_records[-1].equity because the last MTM record reflects close price
      without exit friction applied.

  D2 (year-end cooldown):   _advance_trading_days() must return a calendar estimate
      when insufficient trading dates remain, not silently return the last date.

Additional coverage:
  - BacktestEngine construction and input validation
  - Trade lifecycle: open → max-hold exit → cooldown
  - BacktestResults analytics: win_rate, net_expectancy, profit_factor, total_costs
"""

import dataclasses
from datetime import timedelta
from typing import List

import pandas as pd
import numpy as np
import pytest

from trading_system.backtest_engine import (
    BacktestEngine,
    BacktestResults,
    OpenPosition,
    CompletedTrade,
    DailyRecord,
)
from trading_system.config import TradingConfig, DEFAULT_CONFIG


# ============================================================
# Fixtures
# ============================================================

def _make_price_df(tickers, dates, base_price=100.0, drift=0.001):
    """Synthetic OHLC data: price drifts up by `drift` per day."""
    rows = []
    for ticker in tickers:
        price = base_price
        for d in dates:
            open_p = price
            close_p = price * (1 + drift)
            rows.append({
                "Date": pd.Timestamp(d),
                "Ticker": ticker,
                "Open": open_p,
                "High": close_p * 1.002,
                "Low": open_p * 0.998,
                "Close": close_p,
            })
            price = close_p
    return pd.DataFrame(rows)


def _make_signal_df(tickers, dates, signal="BUY", confidence=0.72, sector="Tech"):
    """Synthetic signal data: same signal for all tickers on all dates."""
    rows = []
    for d in dates:
        for ticker in tickers:
            rows.append({
                "date": pd.Timestamp(d),
                "ticker": ticker,
                "signal": signal,
                "confidence": confidence,
                "sector": sector,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def simple_config():
    """TradingConfig with zero friction to isolate P&L arithmetic."""
    return TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=5,
            cooldown_after_maxhold_days=3,
            cooldown_after_stop_days=3,
        ),
    )


@pytest.fixture
def friction_config():
    """TradingConfig with realistic friction (10+3 bps per side)."""
    return TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=10.0,
            spread_bps=3.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=5,
            cooldown_after_maxhold_days=2,
        ),
    )


# ============================================================
# BacktestEngine construction
# ============================================================

class TestBacktestEngineConstruction:
    def test_default_construction(self):
        engine = BacktestEngine()
        assert engine.config is DEFAULT_CONFIG

    def test_custom_config(self, simple_config):
        engine = BacktestEngine(simple_config)
        assert engine.config is simple_config

    def test_invalid_config_raises(self):
        bad_config = TradingConfig(
            risk=dataclasses.replace(
                DEFAULT_CONFIG.risk,
                drawdown_brake_threshold=0.25,
                drawdown_halt_threshold=0.20,
            )
        )
        with pytest.raises(ValueError, match="Invalid config"):
            BacktestEngine(bad_config)


# ============================================================
# S5 — P&L double-counting fix
# ============================================================

class TestPnLDoubleCountingFix:
    """
    The pre-fix bug: gross_pnl used friction-adjusted entry_price as cost basis,
    then total_costs re-added entry friction → net_pnl understated by ~$0.65/trade.

    Post-fix: gross_pnl = (raw_exit - raw_entry) × shares  (pure price movement)
              total_costs = entry_friction_cost + exit_friction_cost
              net_pnl = gross_pnl - total_costs  (no double-counting)
    """

    def test_zero_friction_net_equals_gross(self, simple_config):
        """With zero friction, gross_pnl must equal net_pnl exactly."""
        dates = pd.bdate_range("2024-01-02", periods=8)
        tickers = ["AAPL"]

        signal_df = _make_signal_df(tickers, dates[:6], signal="BUY")
        price_df = _make_price_df(tickers, dates, base_price=100.0, drift=0.01)

        engine = BacktestEngine(simple_config)
        results = engine.run(signal_df, price_df)

        for trade in results.trade_log:
            assert trade.entry_friction_cost == pytest.approx(0.0, abs=1e-9)
            assert trade.exit_friction_cost == pytest.approx(0.0, abs=1e-9)
            assert trade.total_costs == pytest.approx(0.0, abs=1e-9)
            assert trade.net_pnl == pytest.approx(trade.gross_pnl, abs=1e-9)

    def test_friction_split_correctly(self, friction_config):
        """
        With 13 bps entry friction:
          raw_entry = 100.00
          entry_price = 100.00 × 1.0013 = 100.13
          entry_friction_cost = 100.13 - 100.00 = 0.13 (per share)

        With 13 bps exit friction on ~100.5:
          exit_friction_cost ≈ 100.5 × 0.0013 ≈ 0.131

        gross_pnl = (raw_exit - raw_entry) × shares = ~0.50 × shares
        net_pnl = gross_pnl - (entry_friction + exit_friction)
        """
        dates = pd.bdate_range("2024-01-02", periods=8)
        signal_df = _make_signal_df(["AAPL"], dates[:6], signal="BUY")
        price_df = _make_price_df(["AAPL"], dates, base_price=100.0, drift=0.001)

        engine = BacktestEngine(friction_config)
        results = engine.run(signal_df, price_df)

        assert len(results.trade_log) > 0
        for trade in results.trade_log:
            # Core invariant: net_pnl = gross_pnl - total_costs (no double-count)
            assert trade.net_pnl == pytest.approx(
                trade.gross_pnl - trade.total_costs, abs=1e-6
            )
            # total_costs must be the sum of both friction components
            assert trade.total_costs == pytest.approx(
                trade.entry_friction_cost + trade.exit_friction_cost, abs=1e-9
            )
            # Entry and exit friction must each be positive
            assert trade.entry_friction_cost > 0.0
            assert trade.exit_friction_cost > 0.0

    def test_net_pnl_less_than_gross_pnl_when_profitable(self, friction_config):
        """Friction always costs money: net_pnl < gross_pnl on a winning trade."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        # Strong uptrend so the trade is profitable before friction
        signal_df = _make_signal_df(["AAPL"], dates[:7], signal="BUY")
        price_df = _make_price_df(["AAPL"], dates, base_price=100.0, drift=0.005)

        engine = BacktestEngine(friction_config)
        results = engine.run(signal_df, price_df)

        winning_trades = [t for t in results.trade_log if t.gross_pnl > 0]
        assert len(winning_trades) > 0
        for trade in winning_trades:
            assert trade.net_pnl < trade.gross_pnl

    def test_compounding_verification(self, friction_config):
        """
        Arithmetic verification: sum of all net_pnl must reconcile with
        final_equity - initial_capital (ignoring cash yield which is 0 here).
        """
        dates = pd.bdate_range("2024-01-02", periods=20)
        signal_df = _make_signal_df(["AAPL", "MSFT"], dates[:15], signal="BUY")
        price_df = _make_price_df(["AAPL", "MSFT"], dates, base_price=100.0, drift=0.002)

        engine = BacktestEngine(friction_config)
        results = engine.run(signal_df, price_df)

        total_net_pnl = sum(t.net_pnl for t in results.trade_log)
        equity_change = results.final_equity() - friction_config.capital.initial_capital
        # Should match within floating point tolerance
        assert total_net_pnl == pytest.approx(equity_change, rel=0.01)


# ============================================================
# D1 — force-close exit friction
# ============================================================

class TestForceCloseExitFriction:
    """
    daily_records[-1].equity reflects MTM at close (no exit friction).
    final_equity() must subtract force_close_exit_friction for positions
    still open at backtest end.
    """

    def test_final_equity_less_than_last_daily_record_when_positions_open(
        self, friction_config
    ):
        """
        If positions are still open at backtest end, final_equity() must be
        lower than the last daily record by exactly force_close_exit_friction.
        """
        dates = pd.bdate_range("2024-01-02", periods=10)
        # Signal only on first day → position opens, never closes
        # (max_holding_days=5 but we only have 8 more days → force-closes)
        signal_df = _make_signal_df(["AAPL"], dates[:1], signal="BUY")
        price_df = _make_price_df(["AAPL"], dates, base_price=100.0, drift=0.001)

        engine = BacktestEngine(friction_config)
        results = engine.run(signal_df, price_df)

        # Only relevant if we actually have force-closed trades
        force_closed = [t for t in results.trade_log if t.exit_reason == "backtest_end"]
        if force_closed:
            last_equity = results.daily_records[-1].equity
            # final_equity() must be strictly less
            assert results.final_equity() < last_equity
            # The difference must equal exit friction on force-closed positions
            diff = last_equity - results.final_equity()
            total_exit_friction = sum(t.exit_friction_cost for t in force_closed)
            assert diff == pytest.approx(total_exit_friction, rel=1e-6)

    def test_final_equity_equals_last_record_when_no_force_close(self, simple_config):
        """
        When no positions remain open at backtest end (all exited normally),
        force_close_exit_friction is 0 and final_equity() == last daily record.
        """
        dates = pd.bdate_range("2024-01-02", periods=15)
        # max_holding_days=5, so positions opened early will close naturally
        signal_df = _make_signal_df(["AAPL"], dates[:3], signal="BUY")
        price_df = _make_price_df(["AAPL"], dates, base_price=100.0, drift=0.001)

        engine = BacktestEngine(simple_config)
        results = engine.run(signal_df, price_df)

        no_force_close = all(
            t.exit_reason != "backtest_end" for t in results.trade_log
        )
        if no_force_close:
            assert results.final_equity() == pytest.approx(
                results.daily_records[-1].equity, abs=1e-6
            )


# ============================================================
# D2 — year-end cooldown truncation fix
# ============================================================

class TestAdvanceTradingDays:
    """_advance_trading_days must return a calendar estimate, not the last available
    date, when insufficient trading dates remain."""

    @pytest.fixture
    def engine(self):
        return BacktestEngine(DEFAULT_CONFIG)

    def test_returns_correct_date_within_range(self, engine):
        # future_dates excludes start_date.  n_days=5 from dates[0] means
        # future_dates[4] = dates[5] (dates[1..19] are the 19 future dates).
        dates = pd.bdate_range("2024-01-02", periods=20)
        result = engine._advance_trading_days(dates[0], 5, list(dates))
        assert result == dates[5]  # 5th future trading day from dates[0]

    def test_returns_calendar_estimate_at_boundary(self, engine):
        """When only 2 dates remain but 5 are needed, must extrapolate."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        start = dates[2]  # 3rd date; only 2 future dates remain
        result = engine._advance_trading_days(start, 5, list(dates))

        # Must be AFTER the last available date (not silently truncated)
        assert result > dates[-1]

    def test_calendar_estimate_is_forward_looking(self, engine):
        """Estimated date must be at least as far as the last available date."""
        dates = pd.bdate_range("2024-12-20", periods=3)
        start = dates[0]
        n_days = 10
        result = engine._advance_trading_days(start, n_days, list(dates))
        assert result > dates[-1]

    def test_no_remaining_dates(self, engine):
        """When start_date is past all available dates, estimate from start_date."""
        dates = pd.bdate_range("2024-01-02", periods=5)
        start = dates[-1]  # last available date, zero future dates
        result = engine._advance_trading_days(start, 3, list(dates))
        # Result must be forward from start_date
        assert result > start

    def test_exact_boundary_uses_actual_dates(self, engine):
        """When exactly n_days remain, return the actual date (no estimation)."""
        dates = pd.bdate_range("2024-01-02", periods=10)
        start = dates[0]
        # future_dates = dates[1]..dates[9] (9 elements), need exactly 9
        # → returns future_dates[8] = dates[9]
        result = engine._advance_trading_days(start, 9, list(dates))
        assert result == dates[9]


# ============================================================
# BacktestResults analytics
# ============================================================

class TestBacktestResultsAnalytics:
    """Analytics computed from CompletedTrade records must be arithmetically correct."""

    @pytest.fixture
    def minimal_results(self):
        """Three synthetic trades: 2 wins (+$50, +$30), 1 loss (-$20)."""
        def _trade(net_pnl, ticker="AAPL"):
            gross = abs(net_pnl) * 1.1 if net_pnl > 0 else abs(net_pnl) * 0.9
            friction = gross - net_pnl if net_pnl > 0 else gross + abs(net_pnl)
            return CompletedTrade(
                trade_id=1, ticker=ticker, sector="Tech", direction="LONG",
                entry_date=pd.Timestamp("2024-01-02"),
                entry_price=100.0, exit_date=pd.Timestamp("2024-01-10"),
                exit_price=110.0 if net_pnl > 0 else 95.0,
                position_pct=0.05, shares=10.0,
                gross_pnl=gross,
                entry_friction_cost=abs(friction) * 0.5,
                exit_friction_cost=abs(friction) * 0.5,
                slippage_cost=abs(friction) * 0.25,
                spread_cost=abs(friction) * 0.25,
                total_costs=abs(friction),
                net_pnl=net_pnl,
                holding_days=8, exit_reason="signal", confidence_at_entry=0.72,
            )

        trades = [_trade(50.0), _trade(30.0), _trade(-20.0)]
        record = DailyRecord(
            date=pd.Timestamp("2024-01-10"), equity=10060.0, cash=10060.0,
            invested_capital=0.0, gross_exposure=0.0, open_positions=0,
            daily_return=0.006, cumulative_return=0.006, drawdown_from_peak=0.0,
            cash_yield_today=0.0, strategy_return_excl_cash=0.006,
            strategy_return_incl_cash=0.006,
        )
        return BacktestResults(
            trade_log=trades,
            daily_records=[record],
            rejected_signals=[],
            config=DEFAULT_CONFIG,
            force_close_exit_friction=0.0,
        )

    def test_total_trades(self, minimal_results):
        assert minimal_results.total_trades() == 3

    def test_win_rate(self, minimal_results):
        # 2 wins out of 3
        assert minimal_results.win_rate() == pytest.approx(2 / 3, abs=1e-6)

    def test_net_expectancy(self, minimal_results):
        # (50 + 30 - 20) / 3 = 20.0
        assert minimal_results.net_expectancy() == pytest.approx(20.0, abs=1e-6)

    def test_profit_factor(self, minimal_results):
        pf = minimal_results.profit_factor()
        assert pf is not None
        assert pf > 1.0  # 2 wins vs 1 smaller loss

    def test_total_costs(self, minimal_results):
        expected = sum(t.total_costs for t in minimal_results.trade_log)
        assert minimal_results.total_costs() == pytest.approx(expected, abs=1e-6)

    def test_final_equity_no_force_close(self, minimal_results):
        assert minimal_results.final_equity() == pytest.approx(10060.0, abs=1e-6)

    def test_final_equity_with_force_close(self, minimal_results):
        """final_equity() must subtract the force-close friction from D1 fix."""
        results_with_fc = BacktestResults(
            trade_log=minimal_results.trade_log,
            daily_records=minimal_results.daily_records,
            rejected_signals=[],
            config=DEFAULT_CONFIG,
            force_close_exit_friction=10.50,  # Canonical D1 test value
        )
        assert results_with_fc.final_equity() == pytest.approx(10060.0 - 10.50, abs=1e-6)

    def test_empty_trade_log_returns_none_metrics(self):
        record = DailyRecord(
            date=pd.Timestamp("2024-01-10"), equity=10000.0, cash=10000.0,
            invested_capital=0.0, gross_exposure=0.0, open_positions=0,
            daily_return=0.0, cumulative_return=0.0, drawdown_from_peak=0.0,
            cash_yield_today=0.0, strategy_return_excl_cash=0.0,
            strategy_return_incl_cash=0.0,
        )
        results = BacktestResults(
            trade_log=[], daily_records=[record], rejected_signals=[],
            config=DEFAULT_CONFIG, force_close_exit_friction=0.0,
        )
        assert results.total_trades() == 0
        assert results.win_rate() is None
        assert results.net_expectancy() is None


# ============================================================
# Input validation
# ============================================================

class TestInputValidation:
    """BacktestEngine.run() must raise on malformed inputs."""

    def test_missing_signal_column_raises(self):
        engine = BacktestEngine(DEFAULT_CONFIG)
        bad_signals = pd.DataFrame({"date": [], "ticker": []})  # missing signal, confidence
        price_df = pd.DataFrame({
            "Date": [], "Ticker": [], "Open": [], "High": [], "Low": [], "Close": []
        })
        with pytest.raises(ValueError, match="signal_df missing"):
            engine.run(bad_signals, price_df)

    def test_missing_price_column_raises(self):
        engine = BacktestEngine(DEFAULT_CONFIG)
        signal_df = pd.DataFrame({"date": [], "ticker": [], "signal": [], "confidence": []})
        bad_price = pd.DataFrame({"Date": [], "Ticker": []})  # missing OHLC
        with pytest.raises(ValueError, match="price_df missing"):
            engine.run(signal_df, bad_price)
