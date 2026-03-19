# CLAUDE.md — Claude Code Context for FPPE

## Project
Financial Pattern Prediction Engine (FPPE) v2.2 — K-NN analogue matching for probabilistic equity signals.
See PROJECT_GUIDE.md for full architecture, API, and roadmap.

## Commands
- `python -m pytest tests/ -v` — Run all 388 tests (always run before committing)
- `python -m pattern_engine.live` — Production EOD signals
- `python -m pattern_engine.overnight` — 6-hour overnight runner
- `venv\Scripts\activate` — Windows venv activation

## Code Style
- Frozen dataclasses for configs (`@dataclass(frozen=True)`)
- Type hints on all public functions
- Docstrings: one-line summary, then Args/Returns sections
- Tests in `tests/test_<module>.py` mirroring `pattern_engine/<module>.py`
- Fixtures in `tests/conftest.py` (synthetic_db, train_db, val_db)
- **Never use `assert` for public API validation** — use RuntimeError/ValueError

## Gotchas
- `hash(EngineConfig)` fails — contains dict field (feature_weights). Use `repr()` instead
- `np.issubdtype()` fails on pandas Arrow StringDtype — use `pd.api.types.is_numeric_dtype()`
- `tempfile.TemporaryDirectory()` needs `ignore_cleanup_errors=True` on Windows (SQLite WAL locking)
- `nn_jobs` must be 1 — joblib deadlocks on Windows/Python 3.12 with parallel NN
- Windows bash paths: use single quotes for backslash paths, never end with trailing `\`
- Schema validation is native Python (no pandera) — deliberate choice to avoid heavy deps
- `cal_frac` was removed in v2.2 (was a no-op). Do not add it back without implementing the calibration holdout split

## Key Files
- `PROJECT_GUIDE.md` — Cross-AI context doc (share with Gemini/ChatGPT sessions)
- `pattern_engine/config.py` — All hyperparameters (EngineConfig + WALKFORWARD_FOLDS)
- `pattern_engine/engine.py` — Core fit/predict/evaluate API
- `pattern_engine/sweep.py` — Grid + Bayesian (Optuna) sweep runners
- `pattern_engine/schema.py` — DataFrame validation at engine boundaries
- `pattern_engine/features.py` — Feature sets including `returns_overnight` and `returns_session`
- `pattern_engine/overnight.py` — Checkpoint state machine (pending/running/completed/partial/failed)
- `pattern_engine/manifest.py` — Run manifests, data versioning, prior-run context loading
- `trading_system/backtest_engine.py` — Phase 1 backtester (42KB, all bugs fixed)
- `trading_system/signal_adapter.py` — Normalizes FPPE output to UnifiedSignal
- `trading_system/config.py` — Trading system configuration
- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` — Trading system design doc
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` — Candlestick categorization design
- `docs/PHASE1_FILE_REVIEW.md` — Phase 1 file review
- `docs/PHASE2_SYSTEM_DESIGN.md` — Phase 2 system design
- `docs/research/Gemini FPPE Research/` — Gemini research documents (10 .docx files)
- `results/cached_signals_2024.csv` — CRITICAL: FPPE signal cache for backtests
- `tests/test_review_fixes.py` — 37 tests for code review fixes (P0/P1)
- `tests/test_manifest.py` — 15 tests for manifest system
- `archive/` — Legacy Phase 1 scripts (superseded by pattern_engine/)

## Dependencies
Core: pandas, numpy, scikit-learn, yfinance, ta, pyarrow, optuna
Optional: scoringrules (CRPS), python-docx (reports)
Test: pytest
