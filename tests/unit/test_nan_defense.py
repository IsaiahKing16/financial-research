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
