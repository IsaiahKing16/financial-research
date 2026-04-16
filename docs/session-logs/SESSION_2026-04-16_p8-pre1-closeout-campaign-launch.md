# Session Log: P8-PRE-1 Closeout + Recovery Campaign Launch
**Date:** 2026-04-16  
**Branch:** main (committed directly — documentation-only changes)  
**Commit:** d1db9ea  
**Handoff sources:** HANDOFF_P8-PRE-1_EXECUTE-AND-CLOSE.md, P8_RECOVERY_CAMPAIGN.md, P8_RECOVERY_CAMPAIGN_ADDENDUM.md

---

## What Was Done

Executed the P8-PRE-1 closeout per the handoff document, then set up the P8 recovery campaign for Track A execution in the next session.

### Tasks Completed

| Task | Status | Notes |
|------|--------|-------|
| Verify P8-PRE-1 result files exist | DONE | Found in worktree, not main |
| Copy 585t_gate_check.txt to results/phase8_pre/ | DONE | Confirmed OVERALL: FAIL |
| Copy 585t_walkforward.tsv to results/phase8_pre/ | DONE | 6 rows, all BSS negative |
| CLAUDE.md: Current Phase → P8-PRE-1 FAIL, T8.1 BLOCKED | DONE | Campaign ACTIVE added |
| CLAUDE.md: Locked Settings Calibration split | DONE | Platt (585T) / beta_abm (52T) |
| CLAUDE.md: Key Design Docs → campaign + ADR-013 | DONE | |
| PHASE_COMPLETION_LOG.md: P8-PRE-1 FAIL entry | DONE | Full provenance, root cause |
| ADR-013: Calibration method production decision | DONE | docs/adr/ADR-013-calibration-method-production.md |
| Create results/campaign_p8_recovery/{track_a,track_b,track_c}/ | DONE | .gitkeep files for git tracking |
| Create docs/campaigns/P8_RECOVERY_CAMPAIGN.md | DONE | Main + addendum merged into single canonical doc |
| Commit all changes | DONE | commit d1db9ea, 9 files, 860 insertions |

---

## Key Findings and Decisions

### P8-PRE-1 Gate Results (Confirmed)

| Gate | Value | Status |
|------|-------|--------|
| G1 BSS positive folds | 0/6 | FAIL |
| G2 Sharpe | 0.04 | FAIL |
| G3 MaxDD | 4.2% | PASS |
| G4 Trades | 200 | PASS |
| G5 Win rate | 47.5% | FAIL |

Per-fold BSS: 2019: -0.00153, 2020: -0.00130, 2021: -0.00131, 2022-Bear: -0.04417, 2023: -0.02419, 2024-Val: -0.00182

### Root Cause

Phase 1 Murphy decomposition confirmed: Resolution = 0.000709 at 585T vs 0.007621 at 52T. Pool dilution at full universe destroys KNN discriminative power. Calibration method cannot compensate for near-zero Resolution.

### Calibration Ambiguity Resolved (ADR-013)

The `walkforward.py` beta_abm monkey-patch was causing confusion. Clarified:
- **Platt**: production path (585T). PatternMatcher native. Used in run_585t_full_stack.py.
- **beta_abm**: 52T research only. walkforward.py monkey-patch is a research artifact.

This is now in CLAUDE.md locked settings and ADR-013.

### Campaign Document Created

`docs/campaigns/P8_RECOVERY_CAMPAIGN.md` merges the main campaign doc + addendum into one canonical reference. Addendum changes incorporated:
- Track A: equal weights (0.1667 each) instead of varied weights; weight sweep deferred until passing N found
- Track A: mandatory `criterion_diagnostics.tsv` before sweep runs
- Track B: connector similarity sweep S1-S6 (conditional on B.2 > B.1); each metric with 4 adversarial checks

---

## State at Session End

- **Branch:** main, clean (staged + committed)
- **Tests:** 945 pass (not re-run this session — no code changes, docs only)
- **CLAUDE.md:** Updated. T8.1 BLOCKED, campaign ACTIVE.
- **Next action:** Begin Track A per HANDOFF_TRACK-A_universe-sweep.md

---

## Open Issues / Watch Items

1. **Worktree not merged:** `feature/p8-pre-1-585t-revalidation` worktree still exists at `.worktrees/p8-pre-1-585t-revalidation`. Since P8-PRE-1 FAILED, the handoff §5 says do NOT merge. Worktree can be cleaned up after campaign completes.

2. **Many unstaged modified files:** git status shows 40+ ` M` files in pattern_engine/ and trading_system/ that were not staged. These are from prior sessions. Not related to this session's work. Track A work should go in its own worktree.

3. **HANDOFF files in repo root:** Several `HANDOFF_*.md` files are untracked in repo root. These are advisor-written strategic docs, not committed by convention. Leave as-is.
