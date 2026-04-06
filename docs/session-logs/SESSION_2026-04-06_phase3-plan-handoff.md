# SESSION 2026-04-06 — Phase 3 Plan Handoff

**AI:** Claude Opus 4.6
**Duration:** ~1 session (continuation across compaction)
**Outcome:** Phase 3 spec + plan written, reviewed, approved, committed. Ready to execute.

## What Happened

1. Phase 1 (BSS Gate H7) and Phase 2 (Half-Kelly) confirmed complete from prior sessions.
2. Brainstormed Phase 3 — Risk Engine Integration. Chose **Approach A: Thin Orchestrator** (stateless functions composing existing building blocks; no new classes with state).
3. Wrote spec → spec reviewer found 2 HIGH issues (overlay semantics ambiguity, silent zero-ATR cascade) → fixed → APPROVED.
4. Wrote plan (7 tasks, TDD) → plan reviewer found HIGH H1 (invented CSV columns), MEDIUM M4 (per-trade vs daily Sharpe), MEDIUM M1 (mid-file imports) → fixed → APPROVED on re-review.

## Key Design Decisions (locked)

- **Phase 3 contract evolution:** overlays multiply **position size**, not confidence. Half-Kelly already incorporates confidence; double-throttling would double-count. `BaseRiskOverlay` docstring update is a separate Out-of-Scope follow-up.
- **`compute_atr_pct` raises `RuntimeError`** on zero/negative inputs. Walk-forward catches and logs `"missing_atr"`. Loud failure at source > silent rejection cascade.
- **Stateless orchestrator** in `trading_system/risk_engine.py`: `compute_atr_pct`, `drawdown_brake_scalar`, `apply_risk_adjustments`, `AdjustedSizing` dataclass. No new classes.
- **`size_position()` gets optional `atr_pct` param** (default `None`) → Phase 2 tests unchanged.
- **Walk-forward mirrors Phase 2 exactly:** `_build_equity_curve` aggregates `net_pnl` by `exit_date` on `pd.bdate_range`; `_sharpe`/`_max_dd` operate on daily returns. `b_hist` computed from `net_pnl` wins/losses (not from a nonexistent `b_ratio` column).

## Artifacts

| File | Commit | Purpose |
|------|--------|---------|
| `docs/superpowers/specs/2026-04-06-phase3-risk-engine-integration-design.md` | `977c8d1` | Approved spec |
| `docs/superpowers/plans/2026-04-06-phase3-risk-engine-integration.md` | `0615c95` | Approved plan, 7 tasks |

## Plan Tasks (summary)

1. Create `trading_system/risk_engine.py` skeleton + `compute_atr_pct` (TDD)
2. Add `drawdown_brake_scalar` (TDD)
3. Modify `trading_system/position_sizer.py` — add `atr_pct` optional param
4. Add `AdjustedSizing` + `apply_risk_adjustments` orchestrator (TDD)
5. Integration tests (real ATR end-to-end, synthetic 20% DD scenario)
6. `scripts/run_phase3_walkforward.py` — 6-fold walk-forward with overlays + DD brake
7. Update `docs/PHASE2_RISK_ENGINE.md` with Phase 3 results, gate check

## Gate Criteria

- [ ] DD brake fires correctly on synthetic 20% scenario
- [ ] Max DD ≤ 10% on walk-forward
- [ ] Sharpe ≥ 1.0 (held from Phase 2's 2.527)
- [ ] Stop-loss exits ≤ 35% of total exits

Fallback: keep ATR + DD brake, disable fatigue/congestion behind flags.

## Next Session

> Execute the Phase 3 plan at `docs/superpowers/plans/2026-04-06-phase3-risk-engine-integration.md` using superpowers:subagent-driven-development. Fresh subagent per task, review between tasks.

All 644 existing tests must continue to pass throughout. Test cmd: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`.
