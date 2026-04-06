# Phase 3 — Risk Engine Integration Design

**Date:** 2026-04-06
**Status:** APPROVED
**Author:** Claude Opus 4.6 + Isaia
**Prerequisite:** Phase 2 Half-Kelly GATE PASSED (2026-04-06)
**Roadmap ref:** fppe-roadmap-v2A.md §6

---

## Goal

Activate existing risk engine research pilots into production: real ATR position sizing, drawdown brake, fatigue and liquidity overlays. Replace Phase 2's flat ATR constant with per-ticker ATR computed from the 52T dataset.

## Architecture: Thin Orchestrator

Phase 3 uses a **thin orchestrator** pattern — a stateless function in `risk_engine.py` that composes existing building blocks. No new classes with complex state. The caller (backtest loop) wires the pieces together.

```
Signal → size_position(atr_pct=real) → SizingResult
                                            ↓
                              apply_risk_adjustments(sizing, drawdown, overlays)
                                            ↓
                                      AdjustedSizing
                                   (final_position_pct, blocked, reason)
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Stateless function, not class | YAGNI — 4 fixed adjustments, unlikely to change. Flat code, easy to test. |
| ATR source | `atr_14 / Close` from 52T dataset | Both columns already exist. No new data loading. |
| ATR override | Optional `atr_pct` param on `size_position()` | Backward-compatible. Phase 2 tests unchanged (default still uses `flat_atr_pct`). |
| Drawdown brake | Linear scalar, separate from overlays | Different contract — scalar on position size, not on signal confidence. |
| Overlay activation | Instantiate in walk-forward script | No changes to overlay source code. Feature flags in `ResearchFlagsConfig` remain in place but are bypassed — the script instantiates overlays directly. Already production-ready (SLE-74, SLE-75). |
| Stop-loss model | Backtest script concern, not risk_engine | Stop evaluation is simulation logic, not a risk adjustment. |
| **Overlay multiplier target** | **Position size, not confidence** | **Phase 3 contract evolution from `BaseRiskOverlay` original docstring.** Half-Kelly already incorporates confidence into position size — re-throttling confidence would double-count it. Composing overlays + DD brake as scalars on `position_pct` is mathematically clean and preserves more diagnostic signal. The `BaseRiskOverlay` ABC docstring will be updated to reflect this. |
| Zero ATR handling | `compute_atr_pct` raises `RuntimeError` | Loud failure. The walk-forward script catches and skips the trade with a logged reason. Better than silent rejection cascading through two layers. |

---

## Component Specifications

### 1. `compute_atr_pct(atr_14, close) → float`

Location: `trading_system/risk_engine.py`

Computes ATR as a fraction of price from pre-computed `atr_14` (dollar ATR) and `Close` price, both available in the 52T volnorm dataset.

```
atr_pct = atr_14 / close
```

Guards:
- `close <= 0` → `RuntimeError("close must be > 0")`
- `atr_14 <= 0` → `RuntimeError("atr_14 must be > 0; ticker has no volatility data")`

**Caller contract:** The walk-forward script wraps `compute_atr_pct` in a try/except. On `RuntimeError`, the trade is skipped and logged with reason `"missing_atr"`. This produces a loud failure at the source rather than silent rejection two layers downstream.

### 2. `size_position()` — ATR Override (modification)

Location: `trading_system/position_sizer.py`

Add optional `atr_pct: Optional[float] = None` parameter. When provided, overrides `config.flat_atr_pct` in the stop distance calculation:

```python
effective_atr = atr_pct if atr_pct is not None else config.flat_atr_pct
stop_distance = config.stop_loss_atr_multiple * effective_atr
```

Validation: `atr_pct` must be in (0, 1) when provided, else return rejected `SizingResult`.

All existing Phase 2 tests pass unchanged (they don't provide `atr_pct`, so `flat_atr_pct` is still used).

### 3. `drawdown_brake_scalar(drawdown, warn, halt) → float`

Location: `trading_system/risk_engine.py`

Linear interpolation:
- `drawdown < warn (0.15)` → `1.0` (full sizing)
- `warn ≤ drawdown < halt (0.20)` → linear from `1.0` to `0.0`
- `drawdown ≥ halt (0.20)` → `0.0` (halt all new trades)

Formula: `scalar = 1.0 - (drawdown - warn) / (halt - warn)`, clamped to [0.0, 1.0].

Guards:
- `warn >= halt` → `RuntimeError`
- `drawdown < 0` → treated as 0 (no drawdown)

### 4. `AdjustedSizing` — Result Dataclass

Location: `trading_system/risk_engine.py`

```python
@dataclass(frozen=True)
class AdjustedSizing:
    original: SizingResult        # Pre-adjustment sizing from position_sizer
    final_position_pct: float     # After DD brake + overlays
    dd_scalar: float              # Drawdown brake scalar [0, 1]
    overlay_multiplier: float     # Product of all overlay multipliers [0, 1]
    blocked: bool                 # True if final_position_pct == 0
    block_reason: Optional[str]   # Human-readable reason if blocked
```

### 5. `apply_risk_adjustments()` — Orchestrator

Location: `trading_system/risk_engine.py`

```python
def apply_risk_adjustments(
    sizing: SizingResult,
    drawdown: float,
    overlays: list[BaseRiskOverlay] | None = None,
    dd_warn: float = 0.15,
    dd_halt: float = 0.20,
) -> AdjustedSizing:
```

Logic:
1. Validate `drawdown`: if `drawdown > 1.0` → `RuntimeError` (catches upstream bugs).
2. If `not sizing.approved` → return blocked `AdjustedSizing` with original rejection reason.
3. Compute `dd_scalar = drawdown_brake_scalar(drawdown, dd_warn, dd_halt)`.
4. Compute `overlay_multiplier = product(o.get_signal_multiplier() for o in overlays)` (1.0 if no overlays).
5. `final_position_pct = sizing.position_pct × dd_scalar × overlay_multiplier`.
6. If `final_position_pct == 0` → `blocked=True` with structured reason.

**`block_reason` format:** Structured strings for parseable diagnostics:
- `"sizing_rejected:<original_reason>"` — Phase 2 sizing rejected the signal
- `"dd_brake:<drawdown_pct>"` — drawdown brake fully halted (e.g., `"dd_brake:0.21"`)
- `"overlay:<OverlayClassName>"` — an overlay returned 0.0 (e.g., `"overlay:LiquidityCongestionGate"`)
- If multiple components contributed to a block, the first one in priority order (sizing → dd → overlays) is reported.

### 6. Stop-Loss Execution Model

Location: `scripts/run_phase3_walkforward.py` (backtest simulation only)

Per the roadmap spec:
- Each day, check if `day_low < entry_price × (1 - stop_loss_atr_multiple × atr_pct_at_entry)`
- If triggered: exit at next trading day's open price
- Record as stop-loss exit in trade log

This is simulation logic in the walk-forward script, not in `risk_engine.py`.

---

## Testing Plan

### `tests/test_risk_engine.py`

**Unit tests:**

| Test | Description |
|------|-------------|
| `test_compute_atr_pct_basic` | `atr_14=2.0, close=100 → 0.02` |
| `test_compute_atr_pct_zero_atr_raises` | `atr_14=0 → RuntimeError` |
| `test_compute_atr_pct_negative_atr_raises` | `atr_14=-1 → RuntimeError` |
| `test_compute_atr_pct_zero_close_raises` | `close=0 → RuntimeError` |
| `test_compute_atr_pct_negative_close_raises` | `close=-1 → RuntimeError` |
| `test_dd_brake_no_drawdown` | `dd=0.0 → 1.0` |
| `test_dd_brake_below_warn` | `dd=0.10 → 1.0` |
| `test_dd_brake_at_warn` | `dd=0.15 → 1.0` |
| `test_dd_brake_midpoint` | `dd=0.175 → 0.5` |
| `test_dd_brake_at_halt` | `dd=0.20 → 0.0` |
| `test_dd_brake_above_halt` | `dd=0.25 → 0.0` |
| `test_dd_brake_invalid_thresholds` | `warn >= halt → RuntimeError` |
| `test_apply_no_overlays` | Sizing passes through with dd_scalar only |
| `test_apply_with_fatigue` | Fatigue multiplier reduces position |
| `test_apply_with_congestion` | Congestion gate blocks |
| `test_apply_both_overlays` | Multiplicative composition |
| `test_apply_rejected_sizing` | `approved=False` input → blocked output |
| `test_apply_dd_halt_blocks` | `dd=0.22 → blocked=True, reason starts with "dd_brake:"` |
| `test_apply_invalid_drawdown_raises` | `drawdown=1.5 → RuntimeError` |
| `test_apply_overlay_order_invariant` | Same overlays in different orders → identical result |
| `test_apply_block_reason_format` | Verify structured `block_reason` strings parse correctly |

**Integration tests:**

| Test | Description |
|------|-------------|
| `test_end_to_end_real_atr` | `compute_atr_pct → size_position(atr_pct=) → apply_risk_adjustments → final_position_pct` |
| `test_zero_atr_pipeline_raises` | `compute_atr_pct(atr_14=0, close=100)` raises before reaching `size_position` (loud failure at source) |
| `test_synthetic_20pct_dd_scenario` | Sequence of losses pushing DD from 10% → 22%. Verify brake transitions: 1.0 → partial → 0.0. |

**Phase 2 regression:** All 28 existing `test_position_sizer.py` tests must still pass (they use default `atr_pct=None`).

---

## Walk-Forward Comparison (T3.5)

### Script: `scripts/run_phase3_walkforward.py`

Same 6-fold structure as Phase 2 (`scripts/run_phase2_walkforward.py`):
- Load 52T volnorm dataset
- For each fold:
  - Instantiate `FatigueAccumulationOverlay(decay_rate=0.15)` and `LiquidityCongestionGate()` (defaults)
  - For each trading day:
    ```python
    fatigue.update(current_date, regime_label=regime)
    congestion.update(current_date, atr=row['atr_14'], close=row['Close'])
    ```
  - For each BUY signal that day:
    ```python
    try:
        atr_pct = compute_atr_pct(row['atr_14'], row['Close'])
    except RuntimeError:
        skip_trade(reason="missing_atr"); continue

    sizing = size_position(confidence, b_ratio, atr_pct=atr_pct)
    adjusted = apply_risk_adjustments(
        sizing, drawdown=current_dd,
        overlays=[fatigue, congestion],
    )
    if adjusted.blocked:
        log_block(adjusted.block_reason); continue
    open_position(ticker, adjusted.final_position_pct)
    ```
  - Stop-loss check: `if day_low < entry × (1 - 3.0 × atr_pct_at_entry): exit at next_day_open`
  - Track equity curve, drawdown, stops, blocked-trade reasons
- Output: `results/phase3_walkforward.tsv`, `results/phase3_gate_check.txt`

### Gate Criteria (from roadmap)

```
[ ] Drawdown brake fires correctly on synthetic 20% DD scenario?
[ ] Max DD ≤ 10% on walk-forward?
[ ] Sharpe ≥ 1.0 maintained from Phase 2?
[ ] Stop-loss fires ≤ 35% of exits?
    YES → Proceed to Phase 4
    NO  → Fallback: keep ATR sizing + DD brake, disable fatigue+congestion behind flags.
```

---

## File Change Summary

| File | Type | Description |
|------|------|-------------|
| `trading_system/risk_engine.py` | NEW | `compute_atr_pct`, `drawdown_brake_scalar`, `AdjustedSizing`, `apply_risk_adjustments` |
| `trading_system/position_sizer.py` | MODIFY | Add `atr_pct` optional param to `size_position()` |
| `tests/test_risk_engine.py` | NEW | ~20 unit + integration tests |
| `scripts/run_phase3_walkforward.py` | NEW | 6-fold walk-forward with real ATR + overlays |
| `docs/PHASE2_RISK_ENGINE.md` | UPDATE | Phase 3 progress tracking |

---

## Out-of-Scope Notes

- **`ResearchFlagsConfig` feature flags** (`use_fatigue_accumulation`, `use_liquidity_congestion_gate`) remain in place. The Phase 3 walk-forward script bypasses them by instantiating overlays directly. They will be retired in a later phase when production paths consolidate.
- **`RiskState.current_atr_estimates`** (in `contracts/state.py`) is not populated by Phase 3's stateless orchestrator. It is reserved for Phase 4 (Portfolio Manager) where stateful per-ticker tracking is needed.
- **`BaseRiskOverlay` ABC docstring update** — separate small commit noting the Phase 3 contract evolution (multiplier targets position size, not confidence). This is documentation-only; the overlay implementations themselves don't change.

---

## Risk & Fallback

| Risk | Mitigation |
|------|------------|
| Real ATR creates larger positions than flat 2% | ATR distribution check: if mean ATR% >> 2%, positions will be smaller (inverse relationship). Verify in walk-forward. |
| Fatigue overlay too aggressive → Sharpe drops | Disable fatigue behind flag, keep ATR + DD brake only |
| Congestion gate blocks too many trades | Check ATR/price distribution. Adjust threshold if > 10% of trades blocked. |
| Max DD worsens vs Phase 2 | Compare trade logs. DD brake at 10% as fallback. |
| Phase 2 tests break | `atr_pct=None` default preserves Phase 2 behavior exactly |
