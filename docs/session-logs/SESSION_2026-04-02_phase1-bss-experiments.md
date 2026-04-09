# Session Log — 2026-04-02 Phase 1 BSS Experiments

**Date:** 2026-04-02
**Phase:** Phase 1 — BSS Diagnosis & Calibration Fix
**Status:** COMPLETE — Gate NOT Met. 3-Strike Rule Triggered. Escalated.
**Duration:** Full day (multiple sub-sessions with background jobs)

---

## Work Completed

### Infrastructure Built

| File | Description |
|------|-------------|
| `pattern_engine/config.py` | New: `EngineConfig` mutable dataclass + `WALKFORWARD_FOLDS`. All Phase 1 scripts import from here. |
| `pattern_engine/matcher.py` | Modified: added `same_sector_boost_factor` support in `_package_results()` (no-op at default 1.0). Bug fix: `val_sectors_b` NameError → derive from `_SECTOR_MAP` via `val_tickers_b`. |
| `scripts/diagnostics/__init__.py` | New: empty |
| `scripts/diagnostics/murphy_gate.py` | New: MurphyGate enforcement class (Phase A → Phase B gate) |
| `scripts/diagnostics/b3_murphy_decomposition.py` | New: Phase A Murphy (1973) BSS decomposition |
| `scripts/experiments/__init__.py` | New: empty |
| `scripts/experiments/h1_max_distance_sweep.py` | New: 12-run grid (max_distance × weighting). Incremental TSV writes + resume logic. |
| `scripts/experiments/h2_sector_filter_sweep.py` | New: 4-run sector filter sweep. Incremental TSV writes + resume logic. |
| `scripts/experiments/h3_topk_sweep.py` | New: 7-run top_k sweep. Incremental TSV writes + resume logic. |
| `scripts/experiments/h4_beta_calibration.py` | New: 2-run beta calibration sweep. Incremental TSV writes + resume logic. |
| `scripts/experiments/phase1_summary.py` | New: ranks all sweep results, records winner or triggers escalation |
| `requirements.txt` | Modified: added `betacal>=0.1.0` |

### Results Produced

| File | Content |
|------|---------|
| `results/benchmarks/b3_murphy_decomposition.tsv` | Per-fold Murphy decomposition (Phase A) |
| `results/murphy_gate.json` | Machine-readable gate state (phase_b_complete=false, winning_config=null) |
| `results/bss_fix_sweep_h1.tsv` | 12 configs: max_distance × weighting sweep |
| `results/bss_fix_sweep_h2.tsv` | 4 configs: sector filter sweep |
| `results/bss_fix_sweep_h3.tsv` | 7 configs: top_k sweep |
| `results/bss_fix_sweep_h4.tsv` | 2 configs: beta calibration sweep |
| `results/phase1_sweep_summary.tsv` | All 25 configs ranked |
| `results/phase1_escalation_log.txt` | 3-strike escalation log with full provenance |

---

## Experiment Results

### Phase A — Murphy Decomposition

```
Mean Resolution:   0.000709  (< 0.001 threshold → NEAR ZERO)
Mean Reliability:  0.001834  (< 0.002 threshold → not dominant)
Dominant failure:  RESOLUTION
Recommended:       signal_quality_first
H4 initially:      BLOCKED (unlocked after H1 completed)
```

Verdict: pool dilution at 585T has destroyed discriminative signal. Calibration cannot help.

### Phase B — All Experiments (25 configs, 150 fold evaluations)

| Experiment | Best config | mean_BSS | pos_folds |
|------------|-------------|----------|-----------|
| H4 | beta_best_h1 (max_d=0.5, inverse, beta_abm) | **-0.00401** | 0/6 |
| H4 | beta_baseline (1.1019, uniform, beta_abm) | -0.00402 | 0/6 |
| H1 | max_d=0.5, uniform | -0.00419 | 0/6 |
| H2 | hard_filter (same_sector_only=True) | -0.00438 | 0/6 |
| H3 | all top_k values (k=10–50) | -0.00435 to -0.00437 | 0/6 |
| Baseline | 1.1019, uniform, Platt | -0.00459 | 0/6 |

Gate required: BSS > 0 on ≥ 3/6 folds. **NOT MET by any config.**

### Key Findings

1. **Beta calibration (+0.00058 BSS)** — real, reproducible. Worth keeping permanently.
2. **Inverse distance weighting** — consistently *worse* than uniform. Amplifies noise when Resolution≈0. Reject.
3. **max_distance=0.5** — marginal improvement (+0.00040). Near neighbors as noisy as far ones.
4. **Sector filtering** — no improvement. Within-sector pool is equally diluted at 585T.
5. **top_k variation** — BSS variance = 0.000022 across k=10–50. Irrelevant. Model is a constant-function approximator.

### Root Cause Confirmed

The KNN model outputs probabilities close to the base rate for essentially every query point at 585T scale. The 8D vol-normalized return fingerprint cannot discriminate 585 heterogeneous tickers in a shared KNN pool. This is an architectural problem, not a parametric one.

---

## Bugs Fixed

- `matcher.py:_package_results()`: `NameError: val_sectors_b not defined` — the sector soft-prior boost block referenced a variable only in scope in `_post_filter()`. Fixed by deriving sectors from `_SECTOR_MAP.get(ticker)` using `val_tickers_b` which is in scope.

---

## Session Infrastructure Lesson

Background bash tasks are tied to the Claude Code session's process group. When a session times out, long-running background processes are killed. Fix applied to all sweep scripts: **incremental TSV writes** (append each completed row immediately) + **resume logic** (read completed rows from existing TSV on restart, skip already-done configs).

---

## Next Session — Recommended Action

**Pursue escalation Path 1 (fastest to live deployment):**

1. Run walk-forward on **52-ticker universe** with `beta calibration + max_d=0.5`
2. If BSS > 0 on ≥ 3/6 folds → Phase 1 gate met at 52T
3. Lock settings, proceed to Phase 2 (Half-Kelly) with 52T universe
4. Plan 585T re-expansion as Phase 6 (after live deployment validated)

The 52T baseline had Fold6 BSS=+0.00103. Signal exists at 52T scale. Beta calibration should improve it further.

**Command to run:**
```bash
# First: confirm 52T data is available or rebuild it
# Then run walk-forward with beta cal + max_d=0.5
PYTHONUTF8=1 py -3.12 scripts/run_walkforward.py
```

Note: `run_walkforward.py` uses `WalkForwardConfig` not `EngineConfig`. To test beta cal + max_d=0.5 at 52T, either:
- Create a dedicated `scripts/experiments/validate_52t_best_config.py` (recommended)
- Or modify `run_walkforward.py` temporarily (not recommended — touches production script)

---

## Test Suite Status

616 passing, 1 skipped — confirmed before and after all code changes.

---

## Locked Settings Update

No locked settings changed. Phase 1 gate was not met.
Beta calibration identified as beneficial but not yet locked (gate not met with 585T universe).
