"""
tests/test_phase3_integration.py — Phase 3 Portfolio Manager Integration Tests

End-to-end tests for BacktestEngine with use_portfolio_manager=True.
Uses synthetic OHLC + signal data only — no external files required.

Coverage:
  - Phase 3 backtest runs without error
  - Phase 3 requires Phase 2 (ValueError at init and run)
  - Single signal: Phase 3 produces identical trade to Phase 2
  - Sector limit: 4th signal in same sector is rejected by portfolio manager
  - Rejection appears in results.rejected_signals with layer="portfolio"
  - Approved signals execute in rank (confidence) order
  - run() override: use_portfolio_manager=True on a Phase-2-only engine
  - run() override: use_portfolio_manager=True without risk engine → ValueError
"""

import dataclasses

import pandas as pd
import pytest

from trading_system.backtest_engine import BacktestEngine
from trading_system.config import (
    PositionLimitsConfig,
    RiskConfig,
    TradingConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (same conventions as test_phase2_integration.py)
# ─────────────────────────────────────────────────────────────────────────────

def _flat_price_df(
    tickers,
    n_days: int = 60,
    start: str = "2024-01-02",
    base_price: float = 100.0,
    atr_half: float = 0.01,
) -> pd.DataFrame:
    """Synthetic OHLC with flat prices and controlled ATR."""
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
    """Build signal rows for one ticker across a list of dates."""
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


def _test_config(
    max_positions_per_sector: int = 3,
    max_sector_pct: float = 0.30,
    **risk_overrides,
) -> TradingConfig:
    """TradingConfig tuned for deterministic integration tests.

    max_sector_pct: Set to 0.99 in sector-limit tests so the PM's count gate
    is the binding constraint, not the dollar-based sector exposure check.
    """
    base = TradingConfig()
    risk_base = base.risk
    if risk_overrides:
        risk_base = dataclasses.replace(risk_base, **risk_overrides)
    pos_limits = dataclasses.replace(
        base.position_limits,
        max_positions_per_sector=max_positions_per_sector,
        max_sector_pct=max_sector_pct,
    )
    return dataclasses.replace(
        base,
        signals=dataclasses.replace(base.signals, confidence_threshold=0.50),
        trade_management=dataclasses.replace(
            base.trade_management,
            max_holding_days=10,
            cooldown_after_stop_days=1,
            cooldown_after_maxhold_days=1,
        ),
        position_limits=pos_limits,
        risk=risk_base,
    )


_DATES = pd.bdate_range("2024-01-02", periods=60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3BasicExecution:
    """Phase 3 backtest runs without errors on well-formed inputs."""

    def _simple_inputs(self):
        price_df = _flat_price_df(["AAPL"], n_days=60)
        rows = _sig("AAPL", "Tech", [_DATES[30]], "BUY")
        rows += _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        return price_df, pd.DataFrame(rows)

    def test_phase3_completes_without_error(self):
        """BacktestEngine(use_risk_engine=True, use_portfolio_manager=True).run() works."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(
            config=_test_config(),
            use_risk_engine=True,
            use_portfolio_manager=True,
        )
        results = engine.run(signal_df, price_df)
        assert results is not None
        assert len(results.equity_df) > 0

    def test_phase3_requires_risk_engine_at_init(self):
        """BacktestEngine(use_portfolio_manager=True) without risk engine raises ValueError."""
        with pytest.raises(ValueError, match="requires use_risk_engine=True"):
            BacktestEngine(
                config=_test_config(),
                use_risk_engine=False,
                use_portfolio_manager=True,
            )

    def test_phase3_requires_risk_engine_at_run_override(self):
        """run(use_portfolio_manager=True, use_risk_engine=False) raises ValueError."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=False)
        with pytest.raises(ValueError, match="requires use_risk_engine=True"):
            engine.run(signal_df, price_df, use_portfolio_manager=True)

    def test_phase3_run_override_on_phase2_engine(self):
        """run(use_portfolio_manager=True) on a use_risk_engine=True engine works."""
        price_df, signal_df = self._simple_inputs()
        engine = BacktestEngine(config=_test_config(), use_risk_engine=True)
        results = engine.run(signal_df, price_df, use_portfolio_manager=True)
        assert results is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. PHASE 3 vs PHASE 2 PARITY (single signal)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3VsPhase2:
    """Single signal should produce identical trade in Phase 2 vs Phase 3."""

    def test_single_signal_same_trade_count(self):
        """Phase 3 and Phase 2 produce the same number of trades for one BUY signal."""
        price_df = _flat_price_df(["AAPL"], n_days=60)
        rows = _sig("AAPL", "Tech", [_DATES[30]], "BUY")
        rows += _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        signal_df = pd.DataFrame(rows)
        cfg = _test_config()

        engine_p2 = BacktestEngine(config=cfg, use_risk_engine=True)
        engine_p3 = BacktestEngine(config=cfg, use_risk_engine=True, use_portfolio_manager=True)

        results_p2 = engine_p2.run(signal_df, price_df)
        results_p3 = engine_p3.run(signal_df, price_df)

        assert results_p3.total_trades() == results_p2.total_trades()

    def test_single_signal_no_portfolio_rejections(self):
        """A single BUY signal should produce zero portfolio-layer rejections."""
        price_df = _flat_price_df(["AAPL"], n_days=60)
        rows = _sig("AAPL", "Tech", [_DATES[30]], "BUY")
        rows += _sig("AAPL", "Tech", _DATES[31:50].tolist(), "HOLD")
        signal_df = pd.DataFrame(rows)

        engine = BacktestEngine(
            config=_test_config(),
            use_risk_engine=True,
            use_portfolio_manager=True,
        )
        results = engine.run(signal_df, price_df)

        portfolio_rejections = [
            r for r in results.rejected_signals
            if r.rejection_layer == "portfolio"
        ]
        assert len(portfolio_rejections) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. SECTOR LIMITS (core PM gate)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3SectorLimits:
    """Portfolio manager enforces max_positions_per_sector on same-day signals."""

    def test_4th_signal_in_same_sector_rejected(self):
        """With max_positions_per_sector=3, a 4th same-sector BUY is rejected by PM.

        max_sector_pct is set to 0.99 so the PM's count gate is the binding
        constraint, not the dollar-based sector exposure check.
        """
        tickers = ["AAPL", "MSFT", "GOOG", "META"]
        price_df = _flat_price_df(tickers, n_days=60)

        # All 4 Tech BUYs on the same date
        signal_date = _DATES[30]
        rows = []
        for t in tickers:
            rows += _sig(t, "Tech", [signal_date], "BUY", confidence=0.75)

        signal_df = pd.DataFrame(rows)

        engine = BacktestEngine(
            config=_test_config(max_positions_per_sector=3, max_sector_pct=0.99),
            use_risk_engine=True,
            use_portfolio_manager=True,
        )
        results = engine.run(signal_df, price_df)

        portfolio_rejections = [
            r for r in results.rejected_signals
            if r.rejection_layer == "portfolio"
        ]
        assert len(portfolio_rejections) >= 1

    def test_sector_rejection_logged_with_portfolio_layer(self):
        """The 4th-sector rejection has rejection_layer == 'portfolio'.

        max_sector_pct is set to 0.99 so PM's count gate fires before the
        dollar-based sector_limit check.
        """
        tickers = ["AAPL", "MSFT", "GOOG", "META"]
        price_df = _flat_price_df(tickers, n_days=60)

        signal_date = _DATES[30]
        rows = []
        for t in tickers:
            rows += _sig(t, "Tech", [signal_date], "BUY", confidence=0.75)

        signal_df = pd.DataFrame(rows)

        engine = BacktestEngine(
            config=_test_config(max_positions_per_sector=3, max_sector_pct=0.99),
            use_risk_engine=True,
            use_portfolio_manager=True,
        )
        results = engine.run(signal_df, price_df)

        rejection_layers = {r.rejection_layer for r in results.rejected_signals}
        assert "portfolio" in rejection_layers

    def test_phase2_does_not_produce_portfolio_rejections(self):
        """Phase 2 (no PM) never logs rejection_layer='portfolio' even at sector limit."""
        tickers = ["AAPL", "MSFT", "GOOG", "META"]
        price_df = _flat_price_df(tickers, n_days=60)

        signal_date = _DATES[30]
        rows = []
        for t in tickers:
            rows += _sig(t, "Tech", [signal_date], "BUY", confidence=0.75)

        signal_df = pd.DataFrame(rows)

        engine = BacktestEngine(
            config=_test_config(max_positions_per_sector=3),
            use_risk_engine=True,
            use_portfolio_manager=False,
        )
        results = engine.run(signal_df, price_df)

        portfolio_rejections = [
            r for r in results.rejected_signals
            if r.rejection_layer == "portfolio"
        ]
        assert len(portfolio_rejections) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. RANKING — highest confidence executes first
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase3Ranking:
    """PM processes signals in rank (confidence descending) order."""

    def test_highest_confidence_signal_enters_trade(self):
        """With sector limit=1, only the highest-confidence signal enters."""
        tickers = ["AAPL", "MSFT"]
        price_df = _flat_price_df(tickers, n_days=60)

        signal_date = _DATES[30]
        rows = _sig("AAPL", "Tech", [signal_date], "BUY", confidence=0.90)
        rows += _sig("MSFT", "Tech", [signal_date], "BUY", confidence=0.70)
        signal_df = pd.DataFrame(rows)

        engine = BacktestEngine(
            config=_test_config(max_positions_per_sector=1),
            use_risk_engine=True,
            use_portfolio_manager=True,
        )
        results = engine.run(signal_df, price_df)

        # Only AAPL (higher confidence) should have traded
        traded_tickers = {t.ticker for t in results.trade_log}
        assert "AAPL" in traded_tickers

        # MSFT should be in portfolio rejections
        portfolio_rejections = [
            r for r in results.rejected_signals
            if r.rejection_layer == "portfolio"
        ]
        rejected_tickers = {r.ticker for r in portfolio_rejections}
        assert "MSFT" in rejected_tickers

    def test_lower_confidence_signal_rejected_at_sector_limit(self):
        """Lower-confidence signal is the one rejected when sector is full."""
        tickers = ["AAPL", "MSFT"]
        price_df = _flat_price_df(tickers, n_days=60)

        signal_date = _DATES[30]
        # MSFT has higher confidence → takes the slot
        rows = _sig("AAPL", "Tech", [signal_date], "BUY", confidence=0.65)
        rows += _sig("MSFT", "Tech", [signal_date], "BUY", confidence=0.85)
        signal_df = pd.DataFrame(rows)

        engine = BacktestEngine(
            config=_test_config(max_positions_per_sector=1),
            use_risk_engine=True,
            use_portfolio_manager=True,
        )
        results = engine.run(signal_df, price_df)

        portfolio_rejections = [
            r for r in results.rejected_signals
            if r.rejection_layer == "portfolio"
        ]
        rejected_tickers = {r.ticker for r in portfolio_rejections}
        # AAPL is lower confidence — should be the one rejected
        assert "AAPL" in rejected_tickers
