# HANDOFF: Phase 7.5 T7.5-1 Review + Documentation Closure
**Date:** 2026-04-19
**Branch:** main (`029ba8a`)
**Task type:** SR (documentation review and correction)

---

## What Happened This Session

This session had one task: review the master plan v4 against the T7.5-1 work completed
the previous session to verify it was done properly. The review found two documentation
gaps and one false premise. All three were corrected.

### Commits This Session

| Commit | Description |
|--------|-------------|
| `6bde221` | docs(t75-1): add key findings to ADR-015, retract false premise in master plan |
| `029ba8a` | docs(handoff): update T7.5-1-5 handoff with 2026-04-19 documentation closure |

No code changes. 948 tests passing (unchanged).

---

## What Was Found and Fixed

### Fix 1 — ADR-015 missing mechanistic evidence

ADR-015 declared a CONDITIONAL PASS for G7.5-1 but did not explain *why* the verdict is
justified despite Condition B (mean BSS < +0.00033) failing. Two findings from the raw TSV
were not documented:

**Finding 1 — `z_score_off` has `resolution = 0.0` on all 6 folds.**
Without StandardScaler, the model collapses to the base rate on every fold — zero
discriminative power. Normalization is not incrementally better; it is the only condition
under which the model can discriminate at all. This is the strongest argument for keeping
`standardize_features=True` and justifies overriding the absolute Condition B threshold.

**Finding 2 — 2022-Bear fold: `n_scored = 416 / 13,052 (3.2%)`.**
The -0.122 BSS in 2022-Bear that dominates the mean is not a normalization failure — it is a
coverage failure. In a bear regime, max_distance=2.5 in 23D feature space produces almost no
matches. 97% of queries go unscored. A fold with 3.2% coverage produces statistically
meaningless reliability/resolution estimates. This is the mechanistic cause of the mean BSS
regression vs the master plan's +0.00033 baseline, and it is already tracked under the
P8-PRE-1 recovery campaign (Track B/C: pool expansion).

Both findings were added to ADR-015 under a new "Key Findings (post-hoc TSV analysis)" section.

### Fix 2 — Master plan line 25 false premise

The master plan stated: *"~91% of L2 distance variance comes from the 8 return features...
expected BSS lift +0.005 to +0.02."* This premise was written before ADR-007 (2026-04-15)
confirmed StandardScaler was already in place. T7.5-1 could never have delivered the expected
lift because the correction was already baked in to all prior BSS numbers.

The paragraph was corrected to reflect the actual finding: StandardScaler was verified (not
added), raw features have zero resolution, and the remaining BSS constraint is bear-regime
coverage collapse — not a normalization gap.

---

## Current State

| Item | State |
|------|-------|
| T7.5-1 | **FULLY CLOSED** — CONDITIONAL PASS, ADR-015 complete, master plan corrected |
| Tests | **948 passing** (945 baseline + 3 T7.5-1 unit tests) |
| Ruff baseline | ≤ 275 (unchanged) |
| `standardize_features=True` | **Locked** — confirmed only source of resolution |
| G7.5-1 gate | **CONDITIONAL PASS** — Condition A (4/6 fold wins) PASS; Condition B (absolute BSS) FAIL due to coverage collapse, not normalization |

---

## Next Session

Pick up **T7.5-2 through T7.5-5** from the detailed work brief:

**`HANDOFF_T75-1-5_phase75-gates.md`** — full implementation briefs for all four tasks.

### Quick summary of what's next

| Task | What | New file | Notes |
|------|------|----------|-------|
| T7.5-2 | `braess_gate()` | `pattern_engine/diagnostics.py` | Mock walk-forward in tests — no real compute |
| T7.5-3 | `identifiability_gate()` | Same `diagnostics.py` | Write with T7.5-2 in one pass |
| T7.5-4 | HMM look-ahead audit | `tests/unit/test_regime_lookahead.py` | Fast — `hmmlearn` not found in codebase |
| T7.5-5 | `cv_bss_estimator()` | `pattern_engine/scoring.py` | Bootstrap CI; synthetic data in tests |

Recommended order: T7.5-4 first (fast audit), then T7.5-2+3 together, then T7.5-5.
Close with ADR-016 covering all four verdicts.

### Key facts the next session must not re-derive

1. `hmmlearn` is NOT in the codebase. The only `predict_proba` is `matcher.py:80`
   (sklearn LogisticRegression — Platt calibrator, forward-only, correct).
2. `_murphy_decomposition()` already exists at `walkforward.py:91–129` as a private
   function. Do NOT duplicate it in `scoring.py`.
3. `pattern_engine/diagnostics.py` and `pattern_engine/scoring.py` do not exist yet.
4. Ruff baseline is ≤ 275. Zero new findings allowed (R9). Run after every new file.
5. All new public functions need icontract `@require`/`@ensure` (R10).

---

## Files Modified This Session

| File | Change |
|------|--------|
| `docs/adr/ADR-015-g75-zscore-revalidation.md` | Added Key Findings section (resolution=0.0, n_scored=3.2%) |
| `FPPE_MASTER_PLAN_v4.md` | Corrected line 25 — retracted false "91% from returns / +0.02 lift" premise |
| `HANDOFF_T75-1-5_phase75-gates.md` | Updated §0 with 2026-04-19 documentation closure summary |
