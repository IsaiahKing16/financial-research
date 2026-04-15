"""
pattern_engine/contracts/finite_types.py — Finite-value type aliases.

FiniteFloat: A float that rejects NaN and ±Inf at Pydantic validation time.
Use on all float fields in the execution layer (orders, allocation, risk).

Why: NaN or Inf in a position size or price field can produce catastrophic
order values that are silently sent to a broker.  This type alias fails fast
at the model boundary — the only place where we can intercept bad data
before it propagates through the pipeline.

Reference: PRD §7A.3 (NaN/Infinity Defense), Knight Capital post-mortem.
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BeforeValidator


def _require_finite(v: object) -> float:
    """Validate that a value is a finite float.  Raises ValueError for NaN/Inf."""
    f = float(v)  # type: ignore[arg-type]
    if not math.isfinite(f):
        raise ValueError(
            f"Expected a finite float (not NaN or ±Inf), got {v!r}. "
            "Check upstream data pipeline for missing or overflow values."
        )
    return f


FiniteFloat = Annotated[float, BeforeValidator(_require_finite)]
"""A float field that rejects NaN and ±Inf at Pydantic validation time.

Usage:
    from pattern_engine.contracts.finite_types import FiniteFloat

    class OrderRequest(BaseModel):
        model_config = ConfigDict(frozen=True, allow_inf_nan=False)
        price: FiniteFloat
        quantity: FiniteFloat
"""
