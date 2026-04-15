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
