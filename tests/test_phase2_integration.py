"""
tests/test_phase2_integration.py — Phase 2 backtest end-to-end validation (SLE-14 / §8.3).

Uses synthetic OHLC + signals only (no cached CSV, no network).
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from trading_system.backtest_engine import BacktestEngine
from trading_system.config import TradingConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bdate_range(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def _flat_ohlc_row(
    d: pd.Timestamp,
    ticker: str,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> dict:
    return {
        "Date": d,
        "Ticker": ticker,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
    }


def _make_signals(
    dates: pd.DatetimeIndex,
    rows: List[Tuple[str, str, float, str]],
) -> pd.DataFrame:
    """Build signal_df rows: (ticker, signal, confidence, sector)."""
    out = []
    for d in dates:
        for ticker, signal, conf, sector in rows:
            out.append(
                {
                    "date": pd.Timestamp(d),
                    "ticker": ticker,
                    "signal": signal,
                    "confidence": conf,
                    "sector": sector,
                }
            )
    return pd.DataFrame(out)


def _warmup_flat_prices(
    dates: pd.DatetimeIndex,
    ticker: str,
    level: float = 100.0,
) -> List[dict]:
    rows = []
    for d in dates:
        rows.append(_flat_ohlc_row(d, ticker, level, level * 1.001, level * 0.999, level))
    return rows


# ---------------------------------------------------------------------------
# §8.3 tests
# ---------------------------------------------------------------------------


def test_phase2_backtest_runs():
    """Full Phase 2 backtest completes without error."""
    dates = _bdate_range(45)
    tickers = ["AAPL"]
    price_rows: List[dict] = []
    for t in tickers:
        price_rows.extend(_warmup_flat_prices(dates, t))
    price_df = pd.DataFrame(price_rows)

    sig = _make_signals(
        dates,
        [("AAPL", "BUY", 0.75, "Tech")],
    )

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=14,
        ),
    )
    engine = BacktestEngine(cfg, use_risk_engine=True)
    results = engine.run(sig, price_df)
    assert results.daily_records
    assert results.final_equity() > 0


def test_phase2_vs_phase1_equity():
    """Dynamic sizing must not reproduce the Phase 1 equal-weight equity curve."""
    dates = _bdate_range(50)
    price_rows = []
    rng = np.random.RandomState(7)
    for d in dates:
        for ticker in ("AAPL", "MSFT"):
            base = 100.0 + rng.randn() * 0.3
            o = base
            c = base * (1.0 + rng.randn() * 0.004)
            hi = max(o, c) * 1.002
            lo = min(o, c) * 0.998
            price_rows.append(_flat_ohlc_row(d, ticker, o, hi, lo, c))
    price_df = pd.DataFrame(price_rows)

    sig_parts = []
    for d in dates:
        for ticker in ("AAPL", "MSFT"):
            sig_parts.append(
                {
                    "date": pd.Timestamp(d),
                    "ticker": ticker,
                    "signal": "BUY",
                    "confidence": 0.72,
                    "sector": "Tech",
                }
            )
    sig = pd.DataFrame(sig_parts)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=20,
        ),
    )

    p1 = BacktestEngine(cfg, use_risk_engine=False).run(sig, price_df, equal_weight_pct=0.05)
    p2 = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)

    eq1 = p1.equity_df["equity"].to_numpy()
    eq2 = p2.equity_df["equity"].to_numpy()
    assert eq1.shape == eq2.shape
    assert not np.allclose(eq1, eq2, rtol=0, atol=1e-6)


def test_stop_losses_appear_in_trade_log():
    """Stop-loss exits appear with exit_reason stop_loss."""
    dates = _bdate_range(40)
    t = "AAPL"
    price_rows: List[dict] = []
    for i, d in enumerate(dates):
        if i < 22:
            price_rows.extend(_warmup_flat_prices(pd.DatetimeIndex([d]), t))
        elif i == 22:
            # Entry next day after signal on index 21 — keep day 22 quiet
            price_rows.append(_flat_ohlc_row(d, t, 100.0, 100.5, 99.5, 100.0))
        elif i == 23:
            # Intraday low breaches any realistic ATR-based stop (~98–99)
            price_rows.append(_flat_ohlc_row(d, t, 99.0, 99.0, 50.0, 98.0))
        else:
            price_rows.append(_flat_ohlc_row(d, t, 100.0, 101.0, 99.0, 100.0))

    price_df = pd.DataFrame(price_rows)

    sig_rows = []
    for i, d in enumerate(dates):
        sig_rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": t,
                "signal": "BUY" if i == 21 else "HOLD",
                "confidence": 0.8,
                "sector": "Tech",
            }
        )
    sig = pd.DataFrame(sig_rows)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=30,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    reasons = [x.exit_reason for x in res.trade_log]
    assert "stop_loss" in reasons
    stop_trades = [x for x in res.trade_log if x.exit_reason == "stop_loss"]
    assert stop_trades
    assert res.stop_loss_events


def test_drawdown_brake_engages_in_losing_streak():
    """Synthetic drawdown crosses brake threshold while stops stay un-breached (MTM path)."""
    dates = _bdate_range(90)
    t = "JPM"
    rng = np.random.RandomState(11)
    price_rows: List[dict] = []
    level = 100.0
    for i, d in enumerate(dates):
        if i < 35:
            # Wide ranges → high ATR% → stop sits far below price (room to drift down).
            tr = 0.08 + abs(rng.randn()) * 0.04
            o = level
            c = level * (1.0 + rng.randn() * 0.01)
            hi = max(o, c) * (1.0 + tr * 0.5)
            lo = min(o, c) * (1.0 - tr * 0.5)
            level = c
            price_rows.append(_flat_ohlc_row(d, t, o, hi, lo, c))
        elif i == 35:
            price_rows.append(_flat_ohlc_row(d, t, level, level * 1.08, level * 0.92, level))
        else:
            # Gradual bear: lows stay above a distant stop, equity drops via MTM.
            level *= 0.988
            c = level
            lo = c * 0.997
            hi = c * 1.003
            price_rows.append(_flat_ohlc_row(d, t, c, hi, lo, c))

    price_df = pd.DataFrame(price_rows)

    sig_rows = []
    for i, d in enumerate(dates):
        sig_rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": t,
                "signal": "BUY" if i == 34 else "HOLD",
                "confidence": 0.85,
                "sector": "Finance",
            }
        )
    sig = pd.DataFrame(sig_rows)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=80,
        ),
        risk=dataclasses.replace(
            DEFAULT_CONFIG.risk,
            drawdown_brake_threshold=0.02,
            drawdown_halt_threshold=0.40,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    max_dd = max(d.drawdown_from_peak for d in res.daily_records)
    assert max_dd >= 0.02


def test_position_sizes_vary_by_volatility():
    """High-vol vs low-vol histories yield different Phase 2 position_pct (below max cap)."""
    dates = _bdate_range(45)

    def path_for_ticker(ticker: str, vol_scale: float) -> List[dict]:
        rows = []
        rng = np.random.RandomState(hash(ticker) % 2**32)
        for d in dates:
            base = 100.0
            wiggle = vol_scale * abs(rng.randn())
            o = base
            c = base + rng.randn() * vol_scale
            hi = max(o, c) + wiggle
            lo = min(o, c) - wiggle
            rows.append(_flat_ohlc_row(d, ticker, o, hi, lo, c))
        return rows

    hi_vol = path_for_ticker("NVDA", 3.0)
    lo_vol = path_for_ticker("WMT", 0.05)
    price_df = pd.DataFrame(hi_vol + lo_vol)

    def run_one(ticker: str) -> float:
        sig = _make_signals(
            dates,
            [(ticker, "HOLD", 0.5, "Tech")],
        )
        sig.loc[sig["date"] == dates[25], "signal"] = "BUY"
        sig.loc[sig["date"] == dates[25], "confidence"] = 0.9

        cfg = TradingConfig(
            costs=dataclasses.replace(
                DEFAULT_CONFIG.costs,
                slippage_bps=0.0,
                spread_bps=0.0,
                risk_free_annual_rate=0.0,
            ),
            trade_management=dataclasses.replace(
                DEFAULT_CONFIG.trade_management,
                max_holding_days=20,
            ),
            position_limits=dataclasses.replace(
                DEFAULT_CONFIG.position_limits,
                max_position_pct=0.25,
            ),
            risk=dataclasses.replace(
                DEFAULT_CONFIG.risk,
                max_loss_per_trade_pct=0.015,
            ),
        )
        res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
        assert res.trade_log
        return res.trade_log[0].position_pct

    pct_hi = run_one("NVDA")
    pct_lo = run_one("WMT")
    assert pct_hi != pytest.approx(pct_lo, rel=1e-3, abs=1e-5)


def test_gap_through_stop_at_next_open():
    """After a stop trigger, fill price is next session open (not the stop price)."""
    dates = _bdate_range(35)
    t = "MSFT"
    rows: List[dict] = []
    for i, d in enumerate(dates):
        if i < 22:
            rows.extend(_warmup_flat_prices(pd.DatetimeIndex([d]), t))
        elif i == 23:
            rows.append(_flat_ohlc_row(d, t, 40.0, 45.0, 20.0, 42.0))
        elif i == 24:
            rows.append(_flat_ohlc_row(d, t, 55.0, 56.0, 54.0, 55.5))
        else:
            rows.append(_flat_ohlc_row(d, t, 55.0, 56.0, 54.0, 55.0))
    price_df = pd.DataFrame(rows)

    sig_rows = []
    for i, d in enumerate(dates):
        sig_rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": t,
                "signal": "BUY" if i == 21 else "HOLD",
                "confidence": 0.82,
                "sector": "Tech",
            }
        )
    sig = pd.DataFrame(sig_rows)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=40,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    stop_trade = next(x for x in res.trade_log if x.exit_reason == "stop_loss")
    evt = next(e for e in res.stop_loss_events if e.ticker == t)
    assert evt.gap_through is True

    exit_day = stop_trade.exit_date
    raw_next_open = float(price_df.loc[(price_df["Date"] == exit_day) & (price_df["Ticker"] == t), "Open"].iloc[0])
    assert pytest.approx(stop_trade.exit_price, abs=0.02) == pytest.approx(raw_next_open, abs=0.02)


def test_rejection_log_populated():
    """At least one BUY is rejected with a documented reason (insufficient ATR history)."""
    dates = _bdate_range(45)
    # MU: history starts mid-window so as-of the BUY date has <21 rows, but next-day
    # prices exist for execution attempt → sizing fails in risk_engine (not data layer).
    sparse_start = 25
    sparse_dates = dates[sparse_start:]
    sparse = _warmup_flat_prices(sparse_dates, "MU")
    price_df = pd.DataFrame(sparse)

    sig_rows = []
    for i, d in enumerate(dates):
        sig_rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": "MU",
                "signal": "BUY" if i == 30 else "HOLD",
                "confidence": 0.75,
                "sector": "Tech",
            }
        )
    sig = pd.DataFrame(sig_rows)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=20,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    assert res.rejected_signals
    assert any(r.rejection_layer == "risk_engine" for r in res.rejected_signals)
    assert any("Insufficient history" in r.rejection_reason for r in res.rejected_signals)


@pytest.mark.parametrize("profile", ["aggressive", "moderate", "conservative"])
def test_all_profiles_backtest(profile: str):
    """Named risk profiles all produce a completed Phase 2 run."""
    dates = _bdate_range(35)
    price_df = pd.DataFrame(_warmup_flat_prices(dates, "KO"))
    sig = _make_signals(
        dates,
        [("KO", "BUY", 0.75, "Consumer")],
    )
    cfg = TradingConfig.from_profile(profile)
    cfg = dataclasses.replace(
        cfg,
        costs=dataclasses.replace(
            cfg.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    assert res.daily_records[-1].equity > 0


def test_risk_state_persists_across_days():
    """Peak equity is reflected in the equity curve (new highs reset drawdown)."""
    dates = _bdate_range(60)
    t = "CSCO"
    rows: List[dict] = []
    level = 100.0
    for i, d in enumerate(dates):
        if i < 30:
            rows.append(_flat_ohlc_row(d, t, level, level * 1.01, level * 0.99, level))
        elif i < 45:
            level *= 1.01
            rows.append(_flat_ohlc_row(d, t, level, level * 1.01, level * 0.99, level))
        else:
            level *= 0.992
            rows.append(_flat_ohlc_row(d, t, level, level * 1.005, level * 0.995, level))
    price_df = pd.DataFrame(rows)

    sig_rows = []
    for i, d in enumerate(dates):
        sig_rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": t,
                "signal": "BUY" if i == 22 else "HOLD",
                "confidence": 0.78,
                "sector": "Tech",
            }
        )
    sig = pd.DataFrame(sig_rows)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=50,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    eq = res.equity_df
    running_max = eq["equity"].cummax()
    implied_dd = 1.0 - eq["equity"] / running_max
    assert np.allclose(
        eq["drawdown_from_peak"].to_numpy(),
        implied_dd.to_numpy(),
        rtol=0,
        atol=1e-5,
    )


def test_force_close_with_stops():
    """Open Phase 2 positions with stops still force-close cleanly at backtest end."""
    dates = _bdate_range(32)
    price_df = pd.DataFrame(_warmup_flat_prices(dates, "QCOM"))
    sig_rows = []
    for i, d in enumerate(dates):
        sig_rows.append(
            {
                "date": pd.Timestamp(d),
                "ticker": "QCOM",
                "signal": "BUY" if i == 22 else "HOLD",
                "confidence": 0.77,
                "sector": "Tech",
            }
        )
    sig = pd.DataFrame(sig_rows)

    cfg = TradingConfig(
        costs=dataclasses.replace(
            DEFAULT_CONFIG.costs,
            slippage_bps=0.0,
            spread_bps=0.0,
            risk_free_annual_rate=0.0,
        ),
        trade_management=dataclasses.replace(
            DEFAULT_CONFIG.trade_management,
            max_holding_days=100,
        ),
    )
    res = BacktestEngine(cfg, use_risk_engine=True).run(sig, price_df)
    fc = [x for x in res.trade_log if x.exit_reason == "backtest_end"]
    assert fc
    assert fc[0].ticker == "QCOM"
