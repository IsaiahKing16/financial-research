# CLAUDE.md — FPPE Project Root
# This file is loaded every session. Keep it under 80 lines.
# Detailed protocols live in .claude/skills/ — Claude loads them on demand.

## Project

**FPPE (Financial Pattern Prediction Engine)** — K-nearest-neighbor historical analogue
matching on return fingerprints. Generates probabilistic BUY/SELL/HOLD signals.

## Commands
- `python -m pytest tests/ -v` — Run full suite (always run before committing)
- `python -m pattern_engine.live` — Production EOD signals
- `python -m pattern_engine.overnight` — 6-hour overnight runner
- `venv\Scripts\activate` — Windows venv activation

## Codebases

- `pattern_engine/` — 21 modules, 300 tests (Python package)
- `trading_system/` — 7 modules, 485 tests (Phase 1 & 2 complete)
- `pattern-engine-v2.1.jsx` — React demo (standalone artifact)

## Critical Rules

1. **Run tests first.** `python -m pytest tests/ -v` — all must pass before any commit.
2. **Numbers require provenance.** Any claimed metric must trace to walk-forward results
   or experiment logs. If it cannot be traced, it is fabricated. No exceptions.
3. **Do NOT modify `prepare.py` or this file** unless explicitly asked.
4. **assert → RuntimeError** for all public API guards. `assert` is stripped under `-O`.
5. **nn_jobs=1** always. Prevents Windows/Py3.12 joblib deadlock.
6. **3-strike rule:** If three consecutive attempts at the same fix fail, STOP. Log what
   was tried in the session log and escalate.

## Locked Settings (do not change without new experiment evidence)

Distance=Euclidean, Weighting=uniform, Features=returns_only(8), Calibration=Platt,
cal_frac=0.76, max_distance=1.1019, top_k=50, confidence_threshold=0.65,
regime=binary, horizon=fwd_7d_up

## Key Design Docs (read before modifying related code)

- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.3 — Trading layer architecture
- `docs/PHASE1_FILE_REVIEW.md` — Phase 1 bug audit (all fixed)
- `docs/PHASE2_SYSTEM_DESIGN.md` — Phase 2 risk engine spec
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` v0.2 — Future Phase 6

## Current Phase

**Phase 3: Portfolio Manager** — Signal ranking, sector allocation, capital queue.
Phase 2 (risk engine) is complete. See `docs/PHASE2_SYSTEM_DESIGN.md`.

## Session Protocol

Every session must:
1. Read this file (automatic in Claude Code)
2. Check for active campaign: `docs/campaigns/`
3. Before ending: update session log via `/session-handoff`

## Environment

- Windows 11, Ryzen 9 5900X (12 cores), 32GB RAM, Python 3.12
- venv: `C:\Users\Isaia\.claude\financial-research\venv`
- Activate: `venv\Scripts\activate`

## Skills Available

Claude: check `.claude/skills/` for task-specific protocols. Key skills:
run-walkforward, debug-bss, add-ticker, add-feature-set, run-backtest,
phase2-risk-engine, session-handoff
