# Session Log: 2026-04-10 Phase 7 Cleanup + Handoff
## AI: Claude Code (Opus 4.6)
## Duration: ~20 min
## Campaign: none (Phase 7 closure)

## What Was Accomplished
- Closed Phase 7 Model Enhancements: E1-E4 FAIL (tested), E5-E6 DEFERRED (superseded by new research)
- Fixed CLAUDE.md locked settings divergence: Phase 6 values (`returns_candle(23)`, `max_distance=2.5`) were never committed to main — reconciled during Phase 7 merge
- Merged `phase7-enhancements` branch to main (commit `cc60eff`), deleted worktree + branch
- Updated `enhancement_summary.tsv` to 6 rows (E5/E6 marked DEFERRED)
- 875 tests passing on main (858 committed + 17 untracked Phase 6 test files)

## Decisions Made
- E5/E6 DEFERRED: User chose to defer pending new research direction that will reshape project planning
- CLAUDE.md reconciliation: Phase 6 locked settings carried into Phase 7 merge commit rather than separate commit on main

## What Was Tried But Failed
- E1 BMA: 0/6 folds improved (delta -0.09 to -0.14). Binary labels + Student's t EM = structural mismatch.
- E2 OWA: 0/6 folds met +0.001 gate. MI-ranked features show no differentiation on 23D returns_candle.
- E3 DTW: Spearman rho=1.0000 fast-fail. DTW on equal-length return vectors is mathematically identical to Euclidean.
- E4 Conformal: Width=1.0 (trivial [0,1] intervals). 52T probs in [0.50-0.59] make nonconformity threshold ~0.57.
- **Root cause for all:** 52T probability range [0.50-0.59] is below 0.65 threshold. Any calibration/post-processing on 52T signals is structurally constrained.

## Files Modified
- `CLAUDE.md`: Fixed locked settings (was VOL_NORM_COLS(8)/0.90, now returns_candle(23)/2.5), Phase 7 COMPLETE, test count 858
- `results/phase7/enhancement_summary.tsv`: Added E5/E6 DEFERRED rows (now 6 rows total)

## Metrics Observed (with provenance)
- 875 tests passing: pytest terminal output on main after merge (858 committed + 17 untracked)
- Phase 7 baseline mean BSS = -0.02356: `results/phase7/baseline_23d.tsv`
- 0/4 enhancements passed gates: `results/phase7/enhancement_summary.tsv` (all FAIL)

## Next Session Should
1. Introduce new research that will change overall project planning (user's stated intent)
2. Consider the 52T probability constraint as a structural finding — future work should either operate on 585T production signals or change the signal generation algorithm fundamentally
3. Optionally commit the 17 untracked Phase 6 test files (`tests/unit/test_phase6_bss_comparison.py`, `tests/unit/test_phase6_redundancy_test.py`) and related scripts

## Context for Non-Claude AI
All numbers in this log have provenance. Do not extrapolate or round.
The 52T KNN pipeline produces probabilities only in [0.50, 0.59] — structurally below the 0.65 confidence threshold. This is not a bug; it is a universe-size constraint. Production signals use 585T Platt scaling (probs in [0.65, 0.75]).
Phase 7 tested 4 model enhancements (BMA calibration, OWA feature weighting, DTW reranking, adaptive conformal prediction). All failed. E5 (LOF anomaly) and E6 (STUMPY matrix profile) were deferred, not tested.
Current production pipeline: KNN + returns_candle(23D) + beta_abm calibration + Half-Kelly sizing + risk engine + portfolio manager + mock broker.
