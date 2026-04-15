import math
import pytest
from pydantic import ValidationError


def test_finite_float_rejects_nan():
    """FiniteFloat type alias rejects NaN."""
    from pattern_engine.contracts.finite_types import FiniteFloat
    from pydantic import BaseModel

    class M(BaseModel):
        price: FiniteFloat

    with pytest.raises(ValidationError):
        M(price=float("nan"))

    # String "nan" from JSON deserializers should also be rejected
    with pytest.raises(ValidationError):
        M(price="nan")


def test_finite_float_rejects_inf():
    """FiniteFloat type alias rejects positive and negative infinity."""
    from pattern_engine.contracts.finite_types import FiniteFloat
    from pydantic import BaseModel

    class M(BaseModel):
        price: FiniteFloat

    with pytest.raises(ValidationError):
        M(price=float("inf"))
    with pytest.raises(ValidationError):
        M(price=float("-inf"))


def test_finite_float_accepts_valid():
    """FiniteFloat accepts valid finite floats including edge cases."""
    from pattern_engine.contracts.finite_types import FiniteFloat
    from pydantic import BaseModel

    class M(BaseModel):
        price: FiniteFloat

    m = M(price=0.0)
    assert m.price == 0.0
    m = M(price=-1.5)
    assert m.price == -1.5
    m = M(price=1e-300)
    assert m.price == 1e-300


def test_allocation_decision_rejects_nan_position_pct():
    """AllocationDecision rejects NaN in final_position_pct."""
    import datetime
    from trading_system.contracts.decisions import AllocationDecision, EvaluatorStatus
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AllocationDecision(
            ticker="AAPL",
            signal_date=datetime.date(2024, 1, 2),
            final_position_pct=float("nan"),
            evaluator_status=EvaluatorStatus.GREEN,
            capital_allocated=1000.0,
            rank_in_queue=1,
            sector="Technology",
        )


def test_position_decision_rejects_inf_stop_loss():
    """PositionDecision rejects Inf in stop_loss_price."""
    import datetime
    from trading_system.contracts.decisions import PositionDecision
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PositionDecision(
            ticker="AAPL",
            signal_date=datetime.date(2024, 1, 2),
            approved=True,
            rejection_reason=None,
            position_pct=0.05,
            target_shares=10.0,
            entry_price_estimate=150.0,
            stop_loss_price=float("inf"),
            atr_pct=0.02,
            confidence=0.70,
            sector="Technology",
        )


def test_trade_event_rejects_nan_fill_price():
    """TradeEvent rejects NaN in fill_price (direct order execution field).

    Uses all required fields with valid values; only fill_price is invalid.
    status=PENDING avoids the filled_requires_price validator so the ONLY
    reason for rejection is FiniteFloat catching NaN on fill_price.
    """
    from trading_system.contracts.trades import TradeEvent, OrderSide, OrderStatus
    from pydantic import ValidationError
    from datetime import date

    with pytest.raises(ValidationError):
        TradeEvent(
            trade_event_id=1,
            trade_id=42,
            ticker="AAPL",
            side=OrderSide.BUY,
            order_date=date(2024, 1, 2),
            ordered_quantity=10.0,
            limit_price_estimate=150.0,
            fill_quantity=10.0,
            fill_price=float("nan"),  # <-- sole invalid value
            fill_ratio=1.0,
            status=OrderStatus.PENDING,
            execution_latency_seconds=0.0,
        )


def test_trade_event_valid_construction():
    """Confirm a fully valid TradeEvent can be constructed without error."""
    from trading_system.contracts.trades import TradeEvent, OrderSide, OrderStatus
    from datetime import date

    t = TradeEvent(
        trade_event_id=1,
        trade_id=42,
        ticker="AAPL",
        side=OrderSide.BUY,
        order_date=date(2024, 1, 2),
        ordered_quantity=10.0,
        limit_price_estimate=150.0,
        fill_quantity=10.0,
        fill_price=150.0,
        fill_ratio=1.0,
        status=OrderStatus.FILLED,
        execution_latency_seconds=0.0,
    )
    assert t.fill_price == 150.0


def test_calibrated_probability_rejects_nan():
    """CalibratedProbability rejects NaN — this is the ML/execution boundary."""
    from pattern_engine.contracts.signals import CalibratedProbability
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CalibratedProbability(
            query_ticker="AAPL",
            query_date=__import__("datetime").date(2024, 1, 2),
            raw_prob=float("nan"),
            calibrated_prob=0.70,
            n_neighbors_found=50,
            mean_distance=1.2,
        )


def test_engine_state_rejects_nan_scaler_params():
    """EngineState rejects NaN in scaler_mean (would corrupt distance computation)."""
    import hashlib
    import json
    from pattern_engine.contracts.state import EngineState
    from pydantic import ValidationError

    # Build a valid config_hash from a dummy config
    dummy = json.dumps({"dummy": True}, sort_keys=True)
    valid_hash = hashlib.sha256(dummy.encode()).hexdigest()

    feature_cols = [f"f{i}" for i in range(23)]

    with pytest.raises(ValidationError):
        EngineState(
            feature_cols=feature_cols,
            scaler_mean=[float("nan")] * 23,
            scaler_scale=[1.0] * 23,
            n_samples=100,
            matcher_backend="balltree",
            matcher_params={"backend": "balltree", "n_samples_fitted": 100},
            config_hash=valid_hash,
            fit_timestamp="2024-01-01T00:00:00+00:00",
            feature_set_name="returns_candle",
        )


def test_exception_hierarchy_importable():
    """Error hierarchy is importable and correctly structured."""
    from trading_system.exceptions import (
        TradingSystemError,
        DataError, MarketDataError, StaleDataError,
        ExecutionError, OrderRejectedError, InsufficientFundsError,
        ModelError, CalibrationError,
        RiskLimitError,
    )

    # All are subclasses of TradingSystemError
    for exc_cls in [DataError, ExecutionError, ModelError, RiskLimitError]:
        assert issubclass(exc_cls, TradingSystemError)
    assert issubclass(MarketDataError, DataError)
    assert issubclass(StaleDataError, DataError)
    assert issubclass(OrderRejectedError, ExecutionError)
    assert issubclass(InsufficientFundsError, ExecutionError)
    assert issubclass(CalibrationError, ModelError)
