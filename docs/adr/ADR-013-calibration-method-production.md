# ADR-013: Production Calibration Method

**Status:** Accepted  
**Date:** 2026-04-16  
**Triggered by:** P8-PRE-1 FAIL — calibration ambiguity resolved as part of closeout

---

## Context

Two calibration methods coexist in the codebase:

- **Platt scaling** (`sklearn.calibration.CalibratedClassifierCV`, `method='sigmoid'`) — implemented natively in `PatternMatcher.fit()`, outputs probabilities via logistic sigmoid over KNN decision scores.
- **beta_abm** (`betacal` library, Beta calibration with `parameters='abm'`) — monkey-patched in `walkforward.run_fold()` via `_apply_beta_abm_calibration()`, overrides Platt post-hoc.

### Observed behavior per universe size

| Universe | Calibration | Probability range | Usable (>= 0.65 threshold)? |
|----------|-------------|-------------------|------------------------------|
| 52T | beta_abm (via walkforward monkey-patch) | [0.50, 0.58] | NO — all below 0.65 |
| 585T | Platt (PatternMatcher native, bypass monkey-patch) | [0.65, 0.75] | YES |

The 52T probabilities cluster below the 0.65 confidence threshold regardless of calibration method. This is a pool-size / Resolution issue, not a calibration issue (confirmed by Murphy B3 decomposition: Resolution = 0.000709 at 585T, 0.007621 at 52T). Calibration cannot compensate for zero discriminative signal.

### The monkey-patch path

`walkforward.run_fold()` calls `_apply_beta_abm_calibration()` which replaces the Platt-fitted calibrator on the PatternMatcher instance. This makes the walkforward module implicitly responsible for calibration method selection — a design smell that caused confusion throughout Phases 1–7 and P8-PRE-1.

---

## Decision

1. **Production calibration (585T): Platt scaling** — native to `PatternMatcher.fit()`. Scripts targeting the 585T production universe call `PatternMatcher` directly and do NOT route through `walkforward.run_fold()`.

2. **Research/legacy calibration (52T): beta_abm** — the `walkforward.run_fold()` monkey-patch applies beta_abm to 52T experiments only. This path is valid for research but is **not** on the deployment path.

3. The `walkforward.py` monkey-patch is retained for backward compatibility with 52T research experiments but must not be used in any 585T production script or campaign script.

4. `CLAUDE.md` locked settings updated: `Calibration=Platt (585T production); beta_abm (52T research only)`

---

## Consequences

- Future 585T scripts (Track A, B, C campaign scripts; future EOD pipeline) call `PatternMatcher` directly — no `walkforward.run_fold()` involvement for calibration.
- beta_abm remains available for 52T research experiments (E5/E6 deferred queue, R1 research).
- Any new calibration comparison must be framed as: "does X outperform Platt on 585T?" not "does X outperform beta_abm?" — beta_abm is 52T-only.
- `scripts/run_585t_full_stack.py` (P8-PRE-1 script) is the reference implementation for the Platt path. See commit 6051142.

---

## Provenance

- P8-PRE-1 gate check: `results/phase8_pre/585t_gate_check.txt`
- P8-PRE-1 walkforward: `results/phase8_pre/585t_walkforward.tsv`
- Phase 1 Murphy decomposition: `SESSION_2026-04-05_phase1-h5-h6-handoff.md`
- HANDOFF: `HANDOFF_P8-PRE-1_EXECUTE-AND-CLOSE.md` §6
