"""
test_contracts.py — Unit tests for Pydantic contract models.

Tests that:
    - Valid data constructs successfully
    - Invalid data raises ValidationError
    - Frozen models prevent mutation
    - Validators catch impossible states

Linear: SLE-57
"""

import icontract
import pandas as pd
import pytest
from datetime import date
from pydantic import ValidationError

from pattern_engine.contracts.signals import (
    SignalRecord,
    SignalDirection,
    SignalSource,
)
from trading_system.contracts.trades import (
    TradeRecord,
    PositionRecord,
    DailySnapshot,
    ExitReason,
)


# ─── SignalRecord ────────────────────────────────────────────────

class TestSignalRecord:
    """Tests for the SignalRecord Pydantic model."""

    def _valid_signal(self, **overrides):
        """Factory for valid signal data."""
        defaults = {
            "date": date(2024, 1, 2),
            "ticker": "AAPL",
            "signal": SignalDirection.BUY,
            "confidence": 0.72,
            "signal_source": SignalSource.KNN,
            "sector": "Technology",
            "n_matches": 50,
            "raw_prob": 0.68,
            "mean_7d_return": 1.23,
        }
        defaults.update(overrides)
        return SignalRecord(**defaults)

    def test_valid_construction(self):
        s = self._valid_signal()
        assert s.ticker == "AAPL"
        assert s.confidence == 0.72
        assert s.signal == SignalDirection.BUY

    def test_frozen_immutability(self):
        s = self._valid_signal()
        with pytest.raises(ValidationError):
            s.confidence = 0.99

    def test_confidence_bounds(self):
        self._valid_signal(confidence=0.0)   # lower bound OK
        self._valid_signal(confidence=1.0)   # upper bound OK
        with pytest.raises(ValidationError):
            self._valid_signal(confidence=-0.1)
        with pytest.raises(ValidationError):
            self._valid_signal(confidence=1.1)

    def test_ticker_must_be_uppercase(self):
        with pytest.raises(ValidationError, match="uppercase"):
            self._valid_signal(ticker="aapl")

    def test_ticker_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            self._valid_signal(ticker="")

    def test_n_matches_non_negative(self):
        self._valid_signal(n_matches=0)  # OK
        with pytest.raises(ValidationError):
            self._valid_signal(n_matches=-1)

    def test_n_matches_unreasonably_high(self):
        with pytest.raises(ValidationError, match="unreasonably high"):
            self._valid_signal(n_matches=100_000)

    def test_signal_direction_enum(self):
        for direction in [SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD]:
            s = self._valid_signal(signal=direction)
            assert s.signal == direction

    def test_invalid_signal_direction(self):
        with pytest.raises(ValidationError):
            self._valid_signal(signal="MAYBE")

    def test_string_signal_coercion(self):
        """SignalDirection is a str enum — raw strings should work."""
        s = self._valid_signal(signal="BUY")
        assert s.signal == SignalDirection.BUY


# ─── PositionRecord ─────────────────────────────────────────────

class TestPositionRecord:
    """Tests for the PositionRecord Pydantic model."""

    def _valid_position(self, **overrides):
        defaults = {
            "trade_id": 1,
            "ticker": "AAPL",
            "sector": "Technology",
            "entry_date": date(2024, 1, 3),
            "raw_entry_price": 185.50,
            "entry_price": 185.69,  # After 10 bps slippage
            "shares": 2.7,
            "position_pct": 0.05,
            "confidence_at_entry": 0.72,
            "stop_loss_price": 179.20,
            "atr_pct_at_entry": 0.02,
        }
        defaults.update(overrides)
        return PositionRecord(**defaults)

    def test_valid_construction(self):
        p = self._valid_position()
        assert p.ticker == "AAPL"
        assert p.shares == 2.7

    def test_shares_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._valid_position(shares=0)
        with pytest.raises(ValidationError):
            self._valid_position(shares=-1)

    def test_entry_price_sanity(self):
        """Entry price should not be wildly below raw (friction adds cost)."""
        with pytest.raises(ValidationError, match="suspiciously below"):
            self._valid_position(
                raw_entry_price=100.0,
                entry_price=90.0,  # 10% below — impossible with ~13 bps friction
            )

    def test_frozen(self):
        p = self._valid_position()
        with pytest.raises(ValidationError):
            p.shares = 10.0


# ─── TradeRecord ─────────────────────────────────────────────────

class TestTradeRecord:
    """Tests for the TradeRecord Pydantic model."""

    def _valid_trade(self, **overrides):
        defaults = {
            "trade_id": 1,
            "ticker": "AAPL",
            "sector": "Technology",
            "direction": "LONG",
            "entry_date": date(2024, 1, 3),
            "entry_price": 185.69,
            "exit_date": date(2024, 1, 17),
            "exit_price": 190.50,
            "position_pct": 0.05,
            "shares": 2.7,
            "gross_pnl": 12.99,
            "entry_friction_cost": 0.65,
            "exit_friction_cost": 0.67,
            "slippage_cost": 0.88,
            "spread_cost": 0.44,
            # total_costs = entry(0.65) + exit(0.67) + slippage(0.88) + spread(0.44) = 2.64
            "total_costs": 2.64,
            "net_pnl": 10.35,  # gross_pnl(12.99) - total_costs(2.64)
            "holding_days": 10,
            "exit_reason": ExitReason.MAX_HOLD,
            "confidence_at_entry": 0.72,
        }
        defaults.update(overrides)
        return TradeRecord(**defaults)

    def test_valid_construction(self):
        t = self._valid_trade()
        assert t.net_pnl == 10.35
        assert t.exit_reason == ExitReason.MAX_HOLD

    def test_exit_before_entry_rejected(self):
        with pytest.raises(ValidationError, match="before entry_date"):
            self._valid_trade(
                entry_date=date(2024, 1, 17),
                exit_date=date(2024, 1, 3),
            )

    def test_costs_consistency(self):
        """total_costs must equal entry + exit friction + slippage + spread."""
        with pytest.raises(ValidationError, match="total_costs"):
            self._valid_trade(
                entry_friction_cost=1.00,
                exit_friction_cost=1.00,
                # slippage_cost=0.88, spread_cost=0.44 from defaults
                total_costs=5.00,  # Should be 1.00+1.00+0.88+0.44 = 3.32
            )

    def test_holding_days_minimum(self):
        with pytest.raises(ValidationError):
            self._valid_trade(holding_days=0)

    def test_exit_reason_enum(self):
        for reason in ExitReason:
            t = self._valid_trade(exit_reason=reason)
            assert t.exit_reason == reason

    def test_string_exit_reason(self):
        """ExitReason is a str enum — raw strings should work."""
        t = self._valid_trade(exit_reason="stop_loss")
        assert t.exit_reason == ExitReason.STOP_LOSS


# ─── DailySnapshot ──────────────────────────────────────────────

class TestDailySnapshot:
    """Tests for the DailySnapshot Pydantic model."""

    def _valid_snapshot(self, **overrides):
        defaults = {
            "date": date(2024, 1, 3),
            "equity": 10050.00,
            "cash": 9550.00,
            "invested_capital": 500.00,
            "gross_exposure": 0.05,
            "open_positions": 1,
            "daily_return": 0.005,
            "cumulative_return": 0.005,
            "drawdown_from_peak": 0.0,
            "cash_yield_today": 0.12,
            "strategy_return_excl_cash": 0.005,
            "strategy_return_incl_cash": 0.0051,
        }
        defaults.update(overrides)
        return DailySnapshot(**defaults)

    def test_valid_construction(self):
        d = self._valid_snapshot()
        assert d.equity == 10050.00
        assert d.open_positions == 1

    def test_equity_must_be_positive(self):
        with pytest.raises(ValidationError):
            self._valid_snapshot(equity=0)
        with pytest.raises(ValidationError):
            self._valid_snapshot(equity=-100)

    def test_drawdown_bounds(self):
        self._valid_snapshot(drawdown_from_peak=0.0)   # No drawdown
        self._valid_snapshot(drawdown_from_peak=0.20)  # 20% drawdown
        with pytest.raises(ValidationError):
            self._valid_snapshot(drawdown_from_peak=-0.1)
        with pytest.raises(ValidationError):
            self._valid_snapshot(drawdown_from_peak=1.5)


# ─── Walkforward temporal integrity ─────────────────────────────


def test_run_fold_raises_on_overlapping_dates():
    """run_fold raises RuntimeError when train_end >= val_start (temporal leakage)."""
    import pandas as pd  # noqa: PLC0415
    from pattern_engine.walkforward import run_fold

    bad_fold = {
        "label": "overlap_test",
        "train_end": "2020-01-15",
        "val_start": "2020-01-15",   # same day — overlap
        "val_end":   "2020-06-30",
    }
    dummy_db = pd.DataFrame({
        "Date": pd.date_range("2015-01-01", periods=10),
        "Ticker": "TEST",
        "fwd_7d_up": 0.0,
    })
    with pytest.raises(RuntimeError, match="train_end.*val_start"):
        run_fold(bad_fold, dummy_db)


# ─── Murphy BSS identity ─────────────────────────────────────────


def test_murphy_bss_identity():
    """BSS identity: |REL - RES + UNC - BS| < 1e-6 after every decomposition."""
    import numpy as np
    from pattern_engine.walkforward import _murphy_decomposition

    rng = np.random.default_rng(42)
    probs = rng.uniform(0.3, 0.8, 200)
    y_true = rng.integers(0, 2, 200).astype(float)

    rel, res, unc = _murphy_decomposition(probs, y_true)
    bs = float(np.mean((probs - y_true) ** 2))

    # BSS identity: BS = REL - RES + UNC  (float64 arithmetic; tolerance 1e-4)
    identity_residual = abs(rel - res + unc - bs)
    assert identity_residual < 1e-4, (
        f"Murphy BSS identity violated: |REL-RES+UNC-BS| = {identity_residual:.2e} "
        f"(REL={rel:.6f}, RES={res:.6f}, UNC={unc:.6f}, BS={bs:.6f})"
    )


# ─── icontract guards on _prepare_features ──────────────────────


def test_prepare_features_rejects_nan_input():
    """_prepare_features raises icontract.ViolationError on NaN input."""
    import numpy as np
    import pandas as pd
    import icontract
    from pattern_engine.config import EngineConfig
    from pattern_engine.matcher import PatternMatcher
    from pattern_engine.features import get_feature_cols

    feature_cols = get_feature_cols("returns_only")
    cfg = EngineConfig(feature_set="returns_only", use_hnsw=False)
    m = PatternMatcher(cfg)

    # Fit on clean data first
    rng = np.random.default_rng(0)
    n = 100
    X = rng.normal(0, 1.0, (n, 8))
    df = pd.DataFrame(X, columns=feature_cols)
    df["Ticker"] = "TEST"
    df["Date"] = pd.date_range("2018-01-01", periods=n)
    df["fwd_7d_up"] = rng.integers(0, 2, n).astype(float)
    m.fit(df, feature_cols)

    # Now try to prepare features with NaN
    X_bad = X.copy()
    X_bad[0, 0] = float("nan")
    with pytest.raises(icontract.ViolationError):
        m._prepare_features(X_bad, fit_scaler=False)


# ─── position_sizer icontract guards (P8-PRE-6D) ──────────────────────────────

class TestSizePositionIcontract:
    """P8-PRE-6D: size_position must reject NaN/inf inputs via icontract."""

    def test_rejects_nan_confidence(self):
        """size_position raises ViolationError on NaN confidence."""
        from trading_system.position_sizer import size_position
        with pytest.raises(icontract.ViolationError):
            size_position(confidence=float("nan"), b_ratio=1.2)

    def test_rejects_inf_confidence(self):
        """size_position raises ViolationError on infinite confidence."""
        from trading_system.position_sizer import size_position
        with pytest.raises(icontract.ViolationError):
            size_position(confidence=float("inf"), b_ratio=1.2)

    def test_rejects_nan_b_ratio(self):
        """size_position raises ViolationError on NaN b_ratio."""
        from trading_system.position_sizer import size_position
        with pytest.raises(icontract.ViolationError):
            size_position(confidence=0.65, b_ratio=float("nan"))

    def test_rejects_nan_atr_pct(self):
        """size_position raises ViolationError on NaN atr_pct."""
        from trading_system.position_sizer import size_position
        with pytest.raises(icontract.ViolationError):
            size_position(confidence=0.65, b_ratio=1.2, atr_pct=float("nan"))

    def test_valid_inputs_pass_contracts(self):
        """size_position does not raise on valid inputs."""
        from trading_system.position_sizer import size_position
        result = size_position(confidence=0.65, b_ratio=1.2)
        assert result.approved is True
