# FPPE Tech Debt Audit
**Date:** 2026-03-19
**Scope:** `pattern_engine/`, `trading_system/`, root-level files, tests, infrastructure
**Test suite baseline:** 331 tests, 0 failures
**Auditor:** Claude (automated analysis + static inspection)

---

## Prioritization Framework

Each item is scored on three axes (1–5 scale):

| Axis | Definition |
|------|-----------|
| **Impact** | How much does this slow development or degrade output quality? |
| **Risk** | What is the failure mode if left unaddressed? |
| **Effort** | How hard is the fix? (1 = trivial, 5 = major refactor) |

**Priority = (Impact + Risk) × (6 − Effort)**

Higher score = fix sooner. Items scoring ≥ 28 are P0. Items 15–27 are P1. Items < 15 are P2.

---

## Findings — Ranked by Priority

### P0 — Fix before Phase 2 starts

---

#### D1 · SECTOR_MAP is defined in three separate files and has already diverged
**Category:** Code Debt
**Score: 36** — Impact 4, Risk 5, Effort 2

**What it is.** `SECTOR_MAP` exists as an independent dict literal in:
- `pattern_engine/matching.py` (line 26) — the NN matching loop uses this for `same_sector_only` filtering
- `pattern_engine/sector.py` (line 13) — canonical source; also exports `TICKERS`
- `trading_system/config.py` (line 21) — trading system copies it wholesale

**Why it matters now.** These maps have *already diverged*. The C1 fix in the last session corrected DIS from Industrial → Consumer in `sector.py` and `trading_system/config.py`, but **`matching.py` still classifies DIS as Industrial**. Any backtest or live run where `same_sector_only=True` would apply the wrong sector cohort to DIS. The bug is silent — no error, just wrong matches.

**Fix.** Delete the dict literal from `matching.py` and `trading_system/config.py`. Import from `pattern_engine.sector`:
```python
# matching.py — replace lines 26–48 with:
from pattern_engine.sector import SECTOR_MAP
```
```python
# trading_system/config.py — replace the SECTOR_MAP literal with:
from pattern_engine.sector import SECTOR_MAP as _ENGINE_SECTOR_MAP
SECTOR_MAP: Dict[str, str] = dict(_ENGINE_SECTOR_MAP)
```
Add a regression test asserting `matching.SECTOR_MAP["DIS"] == "Consumer"`.

**Effort estimate:** 2–3 hours including test.

---

#### D2 · `assert` used as API guard in `live.py` (silently stripped under `-O`)
**Category:** Code Debt
**Score: 35** — Impact 3, Risk 4, Effort 1

**What it is.** `live.py` lines 54 and 58 use `assert` to guard against missing `train_db` and `query_db`:
```python
assert train_db is not None, "Provide train_db or a pre-fitted engine"
assert query_db is not None, "Provide query_db with today's data"
```
Python strips all `assert` statements when run with `-O` or `-OO` (optimized mode). These guards disappear entirely in optimized execution, causing a `TypeError: 'NoneType' object is not iterable` deep in NumPy instead of the intended clear error message.

This is the *exact same bug* that was fixed in `matching.py` during the last session (assert → RuntimeError). `live.py` was missed.

**Fix.** Two-line change matching the established project pattern:
```python
if train_db is None:
    raise RuntimeError("Provide train_db or a pre-fitted PatternEngine.")
if query_db is None:
    raise RuntimeError("Provide query_db with today's market data.")
```

**Effort estimate:** 15 minutes.

---

#### I2 · Parquet files are not in `.gitignore` — potentially committed to history
**Category:** Infrastructure Debt
**Score: 35** — Impact 3, Risk 4, Effort 1

**What it is.** `.gitignore` excludes `data/*.csv`, `data/*.npz`, `data/*.pkl`, and `models/*.pkl`. It does **not** exclude `data/*.parquet`. The repo currently contains:
```
data/full_analogue_db.parquet
data/train_db.parquet
data/val_db.parquet
data/test_db.parquet
```
These are multi-year OHLCV databases for 52 tickers, likely tens of MB to >1 GB. If committed to git history, they bloat every clone, make `git log` slow, and — critically — expose proprietary market data in the repo.

**Fix.** Add to `.gitignore`:
```
data/*.parquet
results/*.csv
results/*.tsv
results/*.txt
results/*.json
```
Then verify these files are not tracked: `git ls-files data/ results/`. If tracked, run `git rm --cached data/*.parquet` and commit. (You will need to do this from your Windows terminal — the VM cannot access git.)

**Effort estimate:** 30 minutes including verification.

---

#### I1 · No CI pipeline — tests are run manually or not at all
**Category:** Infrastructure Debt
**Score: 28** — Impact 3, Risk 4, Effort 2

**What it is.** There is no `.github/workflows/`, no `Makefile`, no pre-commit hook. The 331-test suite must be run manually. With Phase 2 adding risk_engine.py and Phase 3 adding live execution, the risk of a silent regression shipping to the `main` branch grows significantly.

**Fix.** Create `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest   # Match production OS (Windows/joblib behavior)
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --tb=short
```
Using `windows-latest` is important: `nn_jobs=1` and `tempfile.TemporaryDirectory(ignore_cleanup_errors=True)` are both Windows-specific gotchas that would mask on Linux CI.

**Effort estimate:** 2–3 hours including testing the workflow.

---

### P1 — Address in Phase 2 sprint

---

#### T1 · `data.py` has zero test coverage (269 lines, the entire data pipeline)
**Category:** Test Debt
**Score: 24** — Impact 4, Risk 4, Effort 3

**What it is.** `pattern_engine/data.py` contains `DataLoader` — the only class that downloads market data, computes all feature columns, handles temporal splitting, and writes parquet files. It has **0 tests**. If `DataLoader.compute_features()` silently produces wrong values (e.g., off-by-one on return windows, wrong sign on overnight features), every downstream model trains on corrupt data. There is no automated check.

`prepare.py` serves the same function (it is the "human-only, do not modify" predecessor) with equally zero test coverage. Both are untested entry points to the entire feature computation pipeline.

**Why testing is hard.** `download()` calls `yfinance.download()`. This must be mocked. `compute_features()` is 130 lines of pure pandas that can be tested with synthetic OHLCV data.

**Fix.** Add `tests/test_data.py` targeting:
- `_compute_ticker_features()` with 30 rows of synthetic OHLCV — verify return columns have correct values at spot-check rows
- `_compute_overnight_features()` — verify `ret_overnight = log(Open[t] / Close[t-1])`
- `temporal_split()` — verify no train rows appear in val set (temporal leakage guard)
- `DataLoader.download()` with `unittest.mock.patch("yfinance.download")` returning a canned DataFrame

**Effort estimate:** 4–6 hours.

---

#### C1 · 11,379 lines of superseded scripts pollute the root directory
**Category:** Code Debt
**Score: 24** — Impact 3, Risk 3, Effort 2

**What it is.** The repo root contains eight superseded strategy files totalling 11,379 lines:

| File | Lines | Status |
|------|-------|--------|
| `strategy.py` | 2,499 | Superseded by `pattern_engine/` — but *still imported* by `signal_adapter.py` |
| `strategy_overnight.py` | 2,602 | Superseded by `pattern_engine/overnight.py` |
| `strategyv4.py` | 1,106 | Superseded |
| `strategyv3.py` | 862 | Superseded |
| `strategyv2.py` | 792 | Superseded |
| `strategyv1.py` | 377 | Superseded |
| `oldstrategy.py` | 722 | Superseded |
| `oldstrategy1.py` | 433 | Superseded |
| `test_strategy.py` | 1,352 | Tests for superseded code |
| `dedup.py` | 29 | Utility for old strategy |
| `diagnose_distances.py` | 208 | Debug script |
| `quick_sweep.py` | 397 | Ad-hoc sweep runner |

**Critical note.** `strategy.py` cannot simply be deleted: `signal_adapter.simulate_signals_from_val_db()` imports 10 symbols from it at call time. Moving it to `archive/` before completing the Phase 2 migration will break Phase 1 backtesting. Strategy: move all files *except* `strategy.py` to `archive/`, and leave `strategy.py` in root with a prominent deprecation notice until the Phase 2 PatternEngine migration is done.

Also present are junk files with no code value: `experiment_log.md.txt`, `freshcmdwindow.txt`, `newprogram.md`, `program - Copy.md`.

**Effort estimate:** 2 hours (git mv + verify tests still pass).

---

#### C2 · `returns_hybrid` feature set is a placeholder with no runtime guard
**Category:** Code Debt
**Score: 20** — Impact 2, Risk 3, Effort 2

**What it is.** `features.py` registers a feature set called `"returns_hybrid"` that includes 16 columns named `lstm_latent_0` through `lstm_latent_15`. These columns do not exist in any dataset. If a user accidentally configures `EngineConfig(feature_set="returns_hybrid")`, the engine will call `validate_train_db()`, which will raise a `SchemaError` listing 16 missing columns — but with no explanation that the feature set requires a trained CONV_LSTM encoder.

The CONV_LSTM network does not exist in this codebase. The feature set is a forward declaration for a system that has not been built.

**Fix.** Either add an explicit `NotImplementedError` guard in `FeatureRegistry.get()` for `"returns_hybrid"`, or add a `requires_network: bool = False` flag to `FeatureSet` and raise a clear error:
```python
if feature_set.requires_network:
    raise NotImplementedError(
        f"Feature set '{name}' requires a trained neural network encoder "
        f"that is not yet implemented. See Phase 3 roadmap."
    )
```

**Effort estimate:** 1–2 hours including test.

---

#### D3 · `requirements.txt` uses `>=` floors only — no lock file
**Category:** Dependency Debt
**Score: 20** — Impact 2, Risk 3, Effort 2

**What it is.** Every dependency in `requirements.txt` is a lower bound (`>=`). The installed versions are significantly ahead of those bounds — for example, `pandas==2.3.3` vs `pandas>=2.0.0`, and `yfinance==1.2.0` vs `yfinance>=0.2.36`. The yfinance API has historically had breaking changes between minor versions. An automatic dependency update on a new machine could silently install a version that breaks the data pipeline.

There is also no distinction between runtime and development dependencies (e.g., `pytest`, `pytest-cov` are currently not in `requirements.txt` at all — they must be installed ad hoc).

**Fix.** Add `requirements.lock.txt` generated from `pip freeze` on the verified working environment. Split `requirements.txt` into `requirements.txt` (runtime) and `requirements-dev.txt` (pytest, pytest-cov). Update CI to install from lock file.

**Effort estimate:** 1–2 hours.

---

#### I3 · Results files and scratch files not gitignored
**Category:** Infrastructure Debt
**Score: 20** — Impact 2, Risk 2, Effort 1

**What it is.** `results/` contains `backtest_trades.csv`, `cached_signals_2024.csv`, `results.tsv`, `results_analogue.tsv`, and others that are not gitignored. These are generated outputs that should never be in version control: they are large, machine-generated, and may contain sensitive financial analysis. `overnight_progress.json` (a checkpoint file) is also untracked-but-present.

Additionally, `freshcmdwindow.txt` (empty), `experiment_log.md.txt`, `newprogram.md`, and `program - Copy.md` are personal scratch files in the root with no functional purpose.

**Fix.** Extend `.gitignore` (see I2 fix above). Delete or move scratch files.

**Effort estimate:** 30 minutes.

---

#### T2 · `live.py` has zero test coverage (125 lines, the production signal runner)
**Category:** Test Debt
**Score: 18** — Impact 3, Risk 3, Effort 3

**What it is.** `LiveSignalRunner.run()` is the production end-of-day entry point that generates real trading signals. Its logic includes sorting by signal strength, optional cross-model consensus filtering, and the `assert → RuntimeError` gap noted in D2. None of this is tested. There is no test that verifies a BUY signal appears in the output for a row with probability above threshold.

**Fix.** Add `tests/test_live.py` with:
- A test that `LiveSignalRunner` with a pre-fitted `PatternEngine` (using `conftest.py` fixtures) returns a DataFrame with the correct columns
- A test that signal sorting puts BUY before HOLD
- A test that `run()` raises `RuntimeError` (not `AssertionError`) when `query_db=None`

**Effort estimate:** 3–4 hours.

---

#### A2 · No engine state version migration path — v2.2 upgrade will break all saved engines
**Category:** Architecture Debt
**Score: 18** — Impact 3, Risk 3, Effort 3

**What it is.** `PatternEngine.save()` bakes `"version": "2.1"` into the pickle state. `PatternEngine.load()` accepts versions `"2.0"` and `"2.1"` only. When Phase 2 ships new fields (e.g., `risk_engine` state, stop-loss parameters), the version will bump to `"2.2"`, and every saved `engine_state.pkl` will fail to load with:
```
ValueError: Incompatible engine state version '2.1'
```
There is no upgrade path — users must re-fit the engine from scratch.

**Fix.** Implement a `_migrate_state(state: dict) -> dict` function that upgrades older versions to current format by injecting defaults for new fields. Update `load()` to call it before reconstructing the engine. This is the standard forward-compatibility pattern for serialized ML models.

**Effort estimate:** 3–5 hours (including tests for migration logic).

---

### P2 — Backlog (address when touching the relevant module)

---

#### A1 · `simulate_signals_from_val_db()` depends on `strategy.py` (legacy entrypoint)
**Category:** Architecture Debt
**Score: 14** — Impact 4, Risk 3, Effort 4

**What it is.** The Phase 1 backtesting pipeline depends on `strategy._run_matching_loop()`, `strategy.fit_platt_scaling()`, and 8 other symbols from the superseded `strategy.py`. The dependency is guarded with an `ImportError` with a clear migration note (Phase 2), but it creates a fragile coupling: moving `strategy.py` to `archive/` without completing the migration silently breaks all Phase 1 backtesting.

This is already documented as the primary Phase 2 task. It is listed here for completeness and because its priority score would increase significantly once `strategy.py` is moved.

**Fix (Phase 2).** Replace `simulate_signals_from_val_db()` with a call to `PatternEngine.evaluate()` via the new `pattern_engine` API. The entire body of the function (180 lines) reduces to roughly 12 lines using the established API.

**Effort estimate:** 6–10 hours including validation that Phase 1 backtest results are numerically consistent.

---

#### C3 · Magic number `252` (trading days/year) in `run_phase1.py`
**Category:** Code Debt
**Score: 10** — Impact 1, Risk 1, Effort 1

**What it is.** `252` appears inline in `run_phase1.py` in Sharpe ratio annualization and annualized return calculations. There is no named constant. If this file expands (multi-year backtests, crypto which has 365 trading days), the magic number creates confusion.

**Fix.** Add `TRADING_DAYS_PER_YEAR: int = 252` to `trading_system/config.py` alongside the other constants. Replace inline occurrences.

**Effort estimate:** 20 minutes.

---

#### C4 · Missing docstrings on public properties in `calibration.py`, `engine.py`, `matching.py`, `regime.py`, `reliability.py`
**Category:** Code Debt
**Score: 9** — Impact 2, Risk 1, Effort 3

**What it is.** Several public properties and methods have no docstrings, causing IDE help to show nothing:
- `calibration.py`: `fit()`, `transform()`, `fitted` (all calibrator classes)
- `engine.py`: `buy_count`, `sell_count`, `hold_count`, `avg_matches` properties on `PredictionResult`
- `matching.py`: `scaler`, `fitted` properties
- `regime.py`: `n_regimes`, `fitted` properties
- `reliability.py`: `info()`, `error()`, `warn()`, `phase_start()`, `phase_end()` on `ProgressLog`

**Fix.** Add one-line docstrings consistent with the project style (one-line summary, then Args/Returns). Address module-by-module when touching each file.

**Effort estimate:** 2 hours across all files.

---

#### C5 · `Matcher.query()` has 17 branches — complex and difficult to extend
**Category:** Code Debt
**Score: 10** — Impact 3, Risk 2, Effort 4

**What it is.** The inner loop of `Matcher.query()` handles distance filtering, ticker exclusion, sector filtering, regime filtering, and regime fallback logic — all inline inside a nested `for` loop inside a `for` loop over batches. At 17 branches, it is the most complex function in the codebase outside the superseded scripts. Adding Phase 2 features (e.g., stop-loss date filtering, confidence-based weighting of matches) will require touching an already-dense function.

**Fix.** Extract a `_filter_matches(matches, ticker, sector, regime_labels_train, query_regime, cfg)` method that returns the filtered DataFrame. This isolates the filtering logic for testing and extension. Then `query()` becomes a clean batching loop that delegates filtering.

**Effort estimate:** 4–6 hours including test coverage of the extracted method.

---

#### S1 · `PatternEngine.save/load` uses `pickle` — arbitrary code execution risk
**Category:** Security Debt (low severity for current use)
**Score: 8** — Impact 2, Risk 2, Effort 4

**What it is.** `engine.save()` serializes the full engine state with `pickle.dump()`. A maliciously crafted `engine_state.pkl` file would execute arbitrary code on load. The current `load()` catches `pickle.UnpicklingError` but that is not a security defense — a crafted payload would not raise `UnpicklingError`.

For the current use case (single researcher, files never shared), this risk is acceptable. It becomes material if engine states are ever shared with collaborators, deployed to cloud environments, or loaded from external storage.

**Fix (when risk profile changes).** Migrate to `joblib.dump`/`joblib.load` for the sklearn components (scaler, NN index) and `dataclasses.asdict()` + JSON for the config. Store the combined state as a directory or zip. This eliminates the arbitrary execution surface.

**Effort estimate:** 8–12 hours (significant format change with migration needed for existing .pkl files).

---

## Summary Table

| ID | Description | Category | Score | Priority |
|----|-------------|----------|-------|----------|
| D1 | SECTOR_MAP triplication + DIS divergence bug | Code | **36** | P0 |
| D2 | `assert` in `live.py` (stripped under `-O`) | Code | **35** | P0 |
| I2 | Parquet files not gitignored | Infrastructure | **35** | P0 |
| I1 | No CI/CD pipeline | Infrastructure | **28** | P0 |
| T1 | `data.py` zero test coverage | Test | **24** | P1 |
| C1 | 11,379 lines superseded root scripts | Code | **24** | P1 |
| C2 | `returns_hybrid` placeholder — no runtime guard | Code | **20** | P1 |
| D3 | `requirements.txt` unpinned, no lock file | Dependency | **20** | P1 |
| I3 | Results CSVs and scratch files not gitignored | Infrastructure | **20** | P1 |
| T2 | `live.py` zero test coverage | Test | **18** | P1 |
| A2 | No engine state version migration path | Architecture | **18** | P1 |
| A1 | `simulate_signals_from_val_db` → `strategy.py` | Architecture | **14** | P2 |
| C3 | Magic number `252` | Code | **10** | P2 |
| C5 | `Matcher.query()` complexity (17 branches) | Code | **10** | P2 |
| C4 | Missing docstrings (5 modules) | Code | **9** | P2 |
| S1 | Pickle serialization — code execution risk | Security | **8** | P2 |

**16 items total. 4 P0, 7 P1, 5 P2.**

---

## Phased Remediation Plan

### Phase 2 Sprint — Pre-development (parallel to `risk_engine.py` kickoff)

Fix the P0 items first. They require < 4 hours combined and carry active bug risk.

**Week 1 (pre-Phase 2 code):**
- [ ] **D1** — Consolidate SECTOR_MAP to `sector.py`, fix DIS bug in `matching.py` *(2 hrs)*
- [ ] **D2** — Replace `assert` with `RuntimeError` in `live.py` *(15 min)*
- [ ] **I2 + I3** — Extend `.gitignore` for parquet/results/scratch; verify untracked *(30 min)*
- [ ] **D3** — Generate `requirements.lock.txt` from pip freeze *(1 hr)*
- [ ] **C1** — Move all superseded scripts (except `strategy.py`) to `archive/` *(2 hrs)*

**Week 2–3 (alongside Phase 2 feature development):**
- [ ] **I1** — Add GitHub Actions CI (windows-latest) *(3 hrs)*
- [ ] **C2** — Add NotImplementedError guard to `returns_hybrid` *(1 hr)*
- [ ] **T1** — Add `tests/test_data.py` targeting `DataLoader` *(5 hrs)*
- [ ] **T2** — Add `tests/test_live.py` targeting `LiveSignalRunner` *(4 hrs)*

**End of Phase 2:**
- [ ] **A1** — Migrate `simulate_signals_from_val_db()` to `PatternEngine` API *(8 hrs)*
- [ ] **A2** — Implement `_migrate_state()` for engine version upgrades *(4 hrs)*

### Phase 3 Sprint (address during normal development)
- [ ] **C3** — Named constant for 252 *(20 min)*
- [ ] **C4** — Fill docstring gaps across 5 modules *(2 hrs)*
- [ ] **C5** — Extract `_filter_matches()` from `Matcher.query()` *(5 hrs)*

### Phase 4 or never (reassess when risk profile changes)
- [ ] **S1** — Migrate pickle to joblib+JSON if engine states are ever shared externally

---

## Key Observations

**The codebase quality is high relative to its age and origin.** The frozen dataclass pattern, atomic writes, schema validation at every boundary, and the 331-test suite are all materially better than typical research codebases at this stage. The debt that exists is largely structural (root directory clutter from iterative development) and operational (no CI, not gitignored artifacts) rather than algorithmic.

**The most dangerous single item is D1** — not because it is hard to fix, but because it is *silent*. The DIS sector bug in `matching.py` would produce incorrect cohort filtering in any run where `same_sector_only=True`. The three-way SECTOR_MAP duplication guarantees this will happen again on the next ticker universe update unless consolidated.

**The most consequential architectural gap is A1** — `simulate_signals_from_val_db()` depending on `strategy.py`. It is defended, documented, and deferred correctly. But it is a ticking clock: the moment Phase 2's `pattern_engine` branch lands, the Phase 1 backtesting pipeline breaks unless the migration is completed first.
