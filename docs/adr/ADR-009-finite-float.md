# ADR-009: FiniteFloat + allow_inf_nan=False on Execution Layer

**Date:** 2026-04-15
**Status:** ACCEPTED
**Task:** P8-PRE-5A

## Decision

All Pydantic models in the execution layer (trading_system/contracts/ and
pattern_engine/contracts/signals.py) use `ConfigDict(allow_inf_nan=False)`
and `FiniteFloat` on financial float fields.

## Why

NaN or Inf in position_pct, fill_price, or stop_loss_price sent to a broker
can produce catastrophic orders. The Knight Capital $440M loss (2012) traces
to unvalidated values propagating silently through a pipeline. FiniteFloat
fails fast at the model boundary — the only point where external data enters.

## Implementation

- `pattern_engine/contracts/finite_types.py`: `FiniteFloat = Annotated[float, BeforeValidator(_require_finite)]`
- All execution-layer models: `model_config = ConfigDict(frozen=True, allow_inf_nan=False)`
- `EngineState.scaler_mean/scaler_scale`: field_validator rejects NaN in scaler arrays
- `NeighborResult.mean_distance`: sentinel changed from float("inf") to 0.0 to avoid ValidationError at pipeline seam
- Test: `tests/unit/test_nan_defense.py`
