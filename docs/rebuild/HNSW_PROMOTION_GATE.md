# HNSW Promotion Gate — SLE-64

**Status:** Draft
**Linear:** SLE-64
**Decision authority:** Sleep (Isaia)
**Machine gate script:** `scripts/check_hnsw_promotion_gate.py`

---

## Purpose

This document defines the criteria that must all be satisfied before HNSW is promoted
from opt-in (`EngineConfig(use_hnsw=True)`) to the default backend in production
`pattern_engine/`. Until every gate below passes, production continues using BallTree.

The gate is machine-checkable. Run:

```bash
python scripts/check_hnsw_promotion_gate.py
```

All gates must emit `PASS` for promotion to be approved.

---

## Gate 1: Recall Parity (SLE-62)

**Criterion:** HNSW `kneighbors()` must recover ≥ 99.5% of the exact top-50 BallTree
neighbors on the production training set.

| Metric | Threshold | Source |
|--------|-----------|--------|
| Recall@50 | ≥ 0.995 | SLE-62 parity test |
| Signal agreement | ≥ 99% | `test_matcher_parity_staged.py::TestHNSWApproximateParity` |
| Probability Pearson r | ≥ 0.98 | same test class |
| BSS delta vs exact | ≤ 0.001 | same test class |

**Machine check:** Runs parity tests via pytest; reads recall metric from test output.

**Rationale:** HNSW is approximate ANN. If recall drops below 0.995, the engine would
miss meaningful neighbours, potentially degrading signal quality. The 0.5% tolerance
is derived from the production recall@50 = 0.9996 already observed in M1 (SLE-47).

---

## Gate 2: Latency Budget (SLE-63)

**Criterion:** HNSW p95 query latency < 0.1 ms/query at N=50,000 training fingerprints.

| Metric | Threshold | Source |
|--------|-----------|--------|
| p95 latency | < 0.1 ms/query | SLE-63 benchmark |
| Speedup vs BallTree | > 20× | SLE-63 benchmark |

**Machine check:** Reads most recent `artifacts/benchmarks/hnsw_benchmark_*.json`.
Falls back to running the benchmark if no artifact exists.

**Rationale:** The nightly run must finish within a 2-hour window. At N=50k and
~500 tickers, 500 queries × 0.1ms = 50ms total — well within budget. The 20× speedup
floor ensures the latency gain is substantial enough to justify the recall trade-off.

**Note:** The benchmark is `@pytest.mark.slow` and excluded from standard CI. It must
be run explicitly before promotion vote:

```bash
pytest rebuild_phase_3z/tests/performance/test_hnsw_benchmark.py -m slow -v
```

---

## Gate 3: Signal Parity (Full Walk-Forward)

**Criterion:** Signals produced by `PatternMatcher(use_hnsw=True)` must agree with
`PatternMatcher(use_hnsw=False)` on ≥ 99% of rows across the full historical dataset
(2018-01-01 → 2025-12-31, all tickers in production universe).

| Metric | Threshold |
|--------|-----------|
| Signal agreement (BUY/SELL/HOLD) | ≥ 99% |
| Mean probability abs delta | ≤ 0.01 |
| BSS delta (HNSW vs exact, per fold) | ≤ 0.005 per fold |

**Machine check:** Requires walk-forward results in `artifacts/walkforward/`. If absent,
gate emits `SKIP` (not FAIL) — promotion can proceed for individual fold sign-off if
other gates pass, subject to human review.

**How to run the walk-forward:**

```bash
python scripts/run_walkforward.py --use-hnsw --compare-exact
```

Results saved to `artifacts/walkforward/hnsw_parity_report.json`.

---

## Gate 4: Zero Regressions on Full Test Suite

**Criterion:** All tests pass with `EngineConfig(use_hnsw=True)` default. Zero regressions
vs the BallTree baseline. This includes:

- `rebuild_phase_3z/tests/` — all unit + parity tests
- `tests/` — production suite (596 tests)
- `trading_system/tests/` — 556 tests

| Metric | Threshold |
|--------|-----------|
| Test failures | 0 |
| Test warnings introduced | ≤ 5 (excludes pre-existing) |

**Machine check:**

```bash
python -m pytest tests/ trading_system/tests/ rebuild_phase_3z/tests/ -q --tb=no
```

---

## Gate 5: Human Sign-Off

**Required approvals before merge:**

- [ ] Sleep (Isaia) reviews benchmark artifact JSON and confirms latency acceptable
- [ ] Sleep reviews walk-forward parity report (or approves skip with rationale)
- [ ] Linear issue SLE-64 moved to "Done" state

**This gate cannot be machine-checked.** It is the final human decision.

---

## Promotion Checklist

Before updating `EngineConfig.use_hnsw` default to `True` in production:

- [ ] Gate 1: `pytest rebuild_phase_3z/tests/parity/ -v` → all PASS
- [ ] Gate 2: `pytest rebuild_phase_3z/tests/performance/ -m slow -v` → all PASS + artifact written
- [ ] Gate 3: Walk-forward parity report → PASS (or SKIP with human approval)
- [ ] Gate 4: Full test suite → 0 failures
- [ ] Gate 5: Human sign-off

**Promotion PR must include:** The benchmark artifact JSON as an attachment to the PR
description, and a link to the walk-forward parity report.

---

## What Happens at Promotion

1. `pattern_engine/config.py`: `use_hnsw: bool = True` (was `False`)
2. `research/hnsw_distance.py` → deprecated; callers migrated to `PatternMatcher`
3. Documentation updated: `CLAUDE.md` locked settings note updated
4. `phase3z-hnsw-promoted` git tag applied
5. `docs/campaigns/PHASE_3Z_CAMPAIGN.md` M3 gate updated to PROMOTED

---

## Rollback Plan

If production signals degrade post-promotion:

```python
# Immediate rollback — one-line config change
EngineConfig(use_hnsw=False)   # reverts to BallTree
```

BallTree code is retained in perpetuity. HNSW is a config switch, not a replacement.
A rollback does not require a new deployment — just a config update.

---

*Last updated: 2026-03-21 | SLE-64*
