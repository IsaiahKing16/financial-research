# Phase 4 — Portfolio Manager Activation: Planning Outline

**Status:** PRE-PLAN — decisions resolved 2026-04-06; ready for formal plan authoring.
**Author:** Claude Opus 4.6
**Date:** 2026-04-06
**Prerequisite check:** Phase 3 gate PASSED (Sharpe 2.659, MaxDD 4.3%, merged to main). Hardening PR on `fix/phase3-risk-hardening`, not yet merged. Phase 1 BSS gate PASSED via H7 (mean_BSS=+0.00033, 3/6 positive folds).
**Roadmap source:** `fppe-roadmap-v2A.md` §7 (lines 546–601)

---

## 1. Goal (one sentence)

Wire Layer 3 — a stateless Portfolio Manager — into the production trading stack so that on any given day the system ranks competing BUY signals, enforces sector-diversification limits, and forwards an approved subset to the risk engine for sizing, instead of sizing every signal independently.

## 2. Current State vs Roadmap Claim

| Roadmap claim (v2A §7 line 560) | Reality (2026-04-06) |
|---|---|
| "`trading_system/portfolio_manager.py` exists" | **False.** File exists only in `archive/legacy_v1/` (254 lines) and an unrelated worktree. Production `trading_system/` has no PM module. |
| "Validated on 52-ticker universe (37 rejections)" | Refers to legacy validation; not reproducible against current codebase. |
| "Not yet tested on 585T" | Also true — no PM in prod, so no 585T run. |

**Implication:** T4.1 cannot be "flip a flag". It expands to: port or rewrite `portfolio_manager.py` into `trading_system/`, then wire it in.

## 3. What Already Exists (production, not archive)

- `trading_system/contracts/decisions.py::AllocationDecision` — Pydantic model, already aligned to Phase 4 needs (has `rank_in_queue`, `sector`, `final_position_pct`, `capital_allocated`, `adjusted_for_evaluator`).
- `trading_system/config.py::PositionLimitsConfig` — `max_sector_pct=0.30`, `max_positions_per_sector=3`. Gate-ready defaults.
- Sector map infrastructure: `pattern_engine/sector.py`, `scripts/build_sector_map.py`. Used by signal intelligence layer already.
- Walk-forward scaffold: `scripts/run_phase3_walkforward.py` — replays signals day-by-day, feeds risk engine. Natural extension point for Phase 4.
- `trading_system/risk_engine.py` — Phase 3 orchestrator. Takes one `SizingResult` at a time. PM sits *before* the risk engine (ranks → filters → risk engine sizes approved).

## 4. What's Missing

1. `trading_system/portfolio_manager.py` — the stateless module itself.
2. `trading_system/portfolio_state.py` — `PortfolioSnapshot` (today's open positions, sector breakdown, cash, equity) that the PM consumes. Legacy version exists in `archive/legacy_v1/` but uses frozen dataclass; production stack now uses Pydantic, so needs a port.
3. `scripts/run_phase4_walkforward.py` — multi-position day loop (Phase 3 is single-position-per-step).
4. `scripts/analyze_pm_rejections.py` — T4.3 diagnostic.
5. Tests (`tests/test_portfolio_manager.py`, `tests/test_portfolio_state.py`) — target 30+ new tests to match the Phase 2/Phase 3 rigor.

## 5. Task Breakdown (proposed expansion of roadmap T4.1–T4.4)

| Task | Scope | Est. complexity | Depends on |
|---|---|---|---|
| **T4.0** | **Port / rewrite `portfolio_manager.py` into `trading_system/`**. Decision point #1 below. | M | — |
| **T4.0b** | Port `PortfolioSnapshot` to Pydantic (or reuse existing `EvaluatorSnapshot`?). Decision #2. | S | T4.0 |
| **T4.1a** | Wire PM into Phase 3 walk-forward → new `scripts/run_phase4_walkforward.py`. Signal queue per day → rank → filter → size. | M | T4.0, T4.0b |
| **T4.1b** | Add `use_portfolio_manager` flag to config. Default OFF until validated. | XS | T4.1a |
| **T4.2** | Run 585T 6-fold walk-forward with PM active. Collect gate metrics. | M | T4.1 |
| **T4.3** | `scripts/analyze_pm_rejections.py` — histogram by reason, by sector, by confidence bucket. | S | T4.2 |
| **T4.4** | Walk-forward comparison: Phase 3 vs Phase 4 head-to-head (same signals, same risk engine, PM on/off). | S | T4.2 |
| **T4.5 (new)** | Review + two-stage TDD per task — Phase 3 discipline. | spans all | — |

## 6. Gate Check (from roadmap §7 + local sanity)

Required to close Phase 4:
- [ ] No sector > 30% equity at any point in walk-forward
- [ ] Average idle cash < 50%
- [ ] Sharpe ≥ 1.0 maintained (Phase 3 baseline: 2.659 — plenty of headroom)
- [ ] PM rejection log shows reasonable distribution (no single reason > 60%)
- [ ] All existing 696 tests still pass; 30+ new PM tests added
- [ ] Phase 4 walk-forward runs in < 5 min on the 2024 fold (no regression vs Phase 3's ~9.7s suite run)

## 7. Resolved Decisions (2026-04-06)

### Decision #1 — Rewrite clean ✓

**Choice:** Rewrite the Portfolio Manager against current production contracts (`AllocationDecision`, `EvaluatorSnapshot`, `UnifiedSignal`). Do **not** port the legacy `archive/legacy_v1/trading_system/portfolio_manager.py`.

**Implications:**
- New `trading_system/portfolio_manager.py` with Pydantic inputs/outputs from day one.
- Ranking logic is redesigned, not ported — use the legacy file as a **reference only** for invariants (deterministic tie-break, sector cap enforcement order), not as a starting point.
- Allows leaner API surface: stateless free functions (matching Phase 3 risk_engine style), not a class.
- Projected 30+ new tests (matching Phase 2/3 rigor).

### Decision #2 — `trading_system/portfolio_state.py` ✓

**Choice:** New file `trading_system/portfolio_state.py` hosts `PortfolioSnapshot` as a Pydantic model. Do not nest it under `contracts/state.py` alongside `EvaluatorSnapshot`.

**Implications:**
- `EvaluatorSnapshot` (per-ticker evaluator status) and `PortfolioSnapshot` (portfolio-wide open positions, sector breakdown, cash, equity) are distinct concerns — keeping them in separate files avoids a bloated state module.
- `portfolio_state.py` will also host the `PortfolioSnapshot` constructor/factory used by the walk-forward loop.
- Pydantic style (`model_config = {"frozen": True}`) matching Phase 2/3.

### Decision #3 — Regenerate from scratch ✓

**Choice:** Phase 4 walk-forward **regenerates** signals from the matcher with PM active, rather than replaying Phase 1 trade stream.

**Implications:**
- `scripts/run_phase4_walkforward.py` drives the full pipeline: daily KNN match → signal filter → **PM rank + allocate** → risk engine size → trade log.
- PM can show "unlocked" behavior — surfacing signals that Phase 1 flat-sizing missed because of its single-position-at-a-time assumption.
- Runtime cost: 6-fold walk-forward against 585T with KNN recomputation is minutes, not seconds. Acceptable given Phase 3 suite runs in ~10s and the walk-forward is a CI-offline operation.
- Phase 3 vs Phase 4 comparison (T4.4) uses the **same regenerated signals** through both pipelines — apples-to-apples.
- Cached signals in `results/cached_signals_2024.csv` (13,104 rows, 159 BUY) may still be used as a sanity-check reference.

### Decision #4 — Document, don't bypass ✓

**Choice:** 2022-Bear fold Kelly-negative behavior is **documented** in the Phase 4 gate-check. Do **not** add an explicit "PM bypass when Kelly < 0" fallback in the v1 Portfolio Manager.

**Implications:**
- The Phase 2 position sizer already rejects Kelly ≤ 0 via `SizingResult(approved=False, rejection_reason=...)`. Phase 3 risk engine propagates this as `block_reason="sizing_rejected:..."`. PM simply inherits: if no signals survive sizing, PM allocates nothing that day. Emergent safety, no new code path.
- The Phase 4 gate explicitly calls out the 2022-Bear weakness in the gate doc so it doesn't surprise future sessions.
- If 2022-Bear Sharpe tanks Phase 4 gate, we revisit — but we do NOT pre-build a band-aid.

### Decision #5 — H7 is current; Phase 1 BSS gate PASSED ✓

**Choice:** Phase 1 BSS gate is **PASSED** via H7 (regime HOLD mode, `spy_threshold=+0.05`, mean_BSS=+0.00033, 3/6 positive folds). Phase 4 is unblocked.

**Implications:**
- No `Phase 1 BSS BLOCKING` language in CLAUDE.md — any such paragraph is stale and should be removed on next CLAUDE.md touch.
- The "caution: thin margin" note (mean_BSS is positive but only +0.00033) stays — it affects Phase 4 gate interpretation but does not block entry.
- Phase 4 gate still requires Sharpe ≥ 1.0 maintained; thin BSS margin means Phase 4 cannot expect to gain resolution from sharper sizing alone.

## 8. Not in Scope for Phase 4

- Live broker plumbing (Phase 5)
- Regime-aware re-ranking beyond sector (signal intelligence layer)
- Correlation-based diversification (Phase 4 v2)
- SLE-75 fatigue redesign (deferred separately — see Linear comment on SLE-75)
- Changing any locked settings (`max_distance`, `confidence_threshold`, etc.)
- Risk engine hardening PR — already committed on `fix/phase3-risk-hardening`, independent merge decision.

## 9. Next Steps

All decisions resolved. Ready to move to formal plan authoring.

1. **Claude (next session):** invoke `superpowers:writing-plans` to produce the formal Phase 4 implementation plan at `docs/superpowers/plans/2026-04-XX-phase4-portfolio-manager-plan.md`. The plan must include:
   - Per-task TDD skeletons for T4.0, T4.0b, T4.1a/b, T4.2, T4.3, T4.4
   - API signatures for the stateless PM functions (`rank_signals`, `check_allocation`, `allocate_day`) with Pydantic types
   - `PortfolioSnapshot` schema
   - Walk-forward script outline (`scripts/run_phase4_walkforward.py`) showing the daily pipeline
   - Gate-check script (extending `scripts/run_phase3_walkforward.py`'s gate-check pattern)
2. **Claude:** execute the plan via `superpowers:subagent-driven-development` (one subagent per task, TDD red→green→review).
3. **User:** create Linear issues for T4.0–T4.5 before code lands — either under the existing "FPPE Phase 3Z Rebuild" project, or a new "FPPE Phase 4" project for clean separation. Recommend the latter.

### Implementation sequencing
```
T4.0   (rewrite PM)        ──┐
T4.0b  (PortfolioSnapshot) ──┼──► T4.1a (walk-forward wiring) ──► T4.1b (config flag)
                             │                                         │
                             │                                         ▼
                             │                                       T4.2 (585T run)
                             │                                         │
                             │                                         ▼
                             └──────────────────────────────────►  T4.3 (rejection analysis)
                                                                       │
                                                                       ▼
                                                                     T4.4 (Phase 3 vs 4 diff)
```

---

*This outline captured the preconditions and user-confirmed decisions for Phase 4. The formal execution plan is authored separately next session.*
