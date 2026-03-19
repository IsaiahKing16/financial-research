# Project Reorganization Plan
## FPPE — Eliminating Structural Inefficiency

**Date:** March 19, 2026
**Status:** DESIGN COMPLETE — Ready for execution
**Priority:** HIGH — Must be completed before Phase 2 implementation begins

---

## 0. ROOT CAUSE DIAGNOSIS

The primary inefficiency is not a folder naming problem — it is a **branch fragmentation problem**. The codebase is split across three locations that are not talking to each other:

```
Location A: tech-debt-remediation-p0 branch (CURRENT CHECKOUT)
    ├── Clean pattern_engine/ (21 modules, all tech debt fixed)
    ├── trading_system/config.py ONLY (missing 4 Phase 1 files)
    └── Missing: backtest_engine.py, signal_adapter.py, __init__.py, run_phase1.py

Location B: bold-heisenberg worktree (STRANDED)
    ├── Complete trading_system/ with all 5 Phase 1 files
    ├── All Phase 1 results (backtest_trades.csv, cached_signals_2024.csv, etc.)
    ├── 3 new tests (test_trading_config, test_signal_adapter, test_backtest_engine)
    ├── Newer CLAUDE.md, PROJECT_GUIDE.md (v2.2 references)
    └── Several design docs not in the main branch

Location C: main branch (STALE)
    ├── Legacy files at root (oldstrategy.py, dedup.py, etc.) — not archived yet
    ├── Pre-tech-debt-remediation pattern_engine
    └── Missing all Phase 1 trading system work
```

**The bold-heisenberg worktree contains all of the "missing" Phase 1 files** that PHASE2_SYSTEM_DESIGN.md flagged as Issue 0.1. They exist — they are just stranded on a prunable worktree branch that was never merged.

**Secondary issues:** Duplicate data in `data/results/` vs root `results/`, design docs scattered at root level instead of in `docs/`, and a root-level `__pycache__/` that should not exist.

---

## 1. WHAT EXISTS — COMPLETE INVENTORY

### 1.1 Branches and Their State

| Branch | What It Has | Status |
|--------|------------|--------|
| `main` | Pre-tech-debt code; legacy files at root | STALE — should be updated |
| `tech-debt-remediation-p0` | Clean pattern_engine; trading_system/config.py only | CURRENT — needs Phase 1 merged in |
| `claude/bold-heisenberg` | Full Phase 1 trading system + all results | PRUNABLE — needs to be merged first |
| `claude/exciting-brattain` | Already merged | PRUNABLE — safe to delete |
| `claude/nifty-swirles` | Pre-archive legacy structure | PRUNABLE — safe to delete |
| `feature/research-docs-and-results` | Research docs | Assess before deleting |

### 1.2 Worktrees (on Windows host, visible from Linux mount)

| Worktree | Branch | Contains | Action |
|----------|--------|----------|--------|
| `bold-heisenberg` | `claude/bold-heisenberg` | Complete Phase 1 trading system | **MERGE FIRST, THEN PRUNE** |
| `exciting-brattain` | `claude/exciting-brattain` | Already merged to main | PRUNE |
| `nifty-swirles` | `claude/nifty-swirles` | Old pre-archive files | PRUNE |

### 1.3 Duplicate and Misplaced Files

| File/Folder | Current Location | Problem | Target Location |
|-------------|-----------------|---------|-----------------|
| `backtest_trades.csv`, `cached_signals_2024.csv`, etc. | `bold-heisenberg/results/` ONLY | Stranded on worktree | `results/` (root) |
| `experiments.tsv` | `data/results/experiments.tsv` | Wrong parent folder | `results/experiments.tsv` |
| `CANDLESTICK_CATEGORIZATION_DESIGN.md` | Root of bold-heisenberg | Design doc at wrong level | `docs/` |
| `FPPE_TRADING_SYSTEM_DESIGN.md` | Root of bold-heisenberg | Design doc at wrong level | `docs/` |
| `PHASE1_FILE_REVIEW.md` | Root of bold-heisenberg | Review doc at wrong level | `docs/` |
| `PHASE2_SYSTEM_DESIGN.md` | Root (current branch) | Design doc at root | `docs/` |
| `TECH_DEBT_AUDIT.md` | Root (current branch) | Audit doc at root | `docs/` |
| `Gemini FPPE Research/` | Root | Research folder at root | `docs/research/` |
| `docs/FPPE_v2_briefing.md` | `docs/` | Exists and correct | Keep |
| `data/results/` | `data/` subdirectory | Nested results in data folder | Merge into `results/` |
| `__pycache__/` | **Root of project** | Cache dir should not exist at root | Delete |
| `prepare.py` | Root | Human-only data pipeline | Keep at root (correct) |
| `FPPE_v2.1_Project_Report.docx` | Not found | Referenced in PROJECT_GUIDE §12.5 | Unknown — may be Windows-only |

---

## 2. TARGET STRUCTURE

This is what the project should look like after reorganization. Every directory has a single clear purpose.

```
financial-research/                 ← Project root
│
├── CLAUDE.md                       ← AI context (stays at root — required)
├── PROJECT_GUIDE.md                ← Multi-AI reference (stays at root)
├── README.md                       ← Human-facing overview (stays at root)
├── prepare.py                      ← Data pipeline (HUMAN-ONLY — stays at root)
├── requirements.txt                ← Production dependencies
├── requirements-dev.txt            ← Dev/test dependencies
│
├── pattern_engine/                 ← Core prediction package (21 modules)
│   └── [21 modules — unchanged]
│
├── trading_system/                 ← Trading layers (Phases 1–4)
│   ├── __init__.py                 ← FROM bold-heisenberg
│   ├── config.py                   ← EXISTS — already current
│   ├── signal_adapter.py           ← FROM bold-heisenberg
│   ├── backtest_engine.py          ← FROM bold-heisenberg (42KB, Phase 1 complete)
│   └── run_phase1.py               ← FROM bold-heisenberg
│
├── tests/                          ← All tests (pattern_engine + trading_system)
│   ├── conftest.py
│   ├── [21 pattern_engine test files — unchanged]
│   ├── test_trading_config.py      ← FROM bold-heisenberg
│   ├── test_signal_adapter.py      ← FROM bold-heisenberg
│   └── test_backtest_engine.py     ← FROM bold-heisenberg
│
├── data/                           ← Raw data only (no subfolder for results)
│   ├── .gitkeep
│   ├── [52 ticker CSVs]            ← Ignored by .gitignore
│   ├── full_analogue_db.parquet    ← Tracked
│   ├── train_db.parquet            ← Tracked
│   ├── val_db.parquet              ← Tracked
│   ├── test_db.parquet             ← Tracked
│   └── prepared_data.npz           ← Ignored by .gitignore
│   [NO data/results/ subfolder]
│
├── results/                        ← All generated outputs
│   ├── .gitkeep
│   ├── experiments.tsv             ← MOVED from data/results/
│   ├── results.tsv
│   ├── results_analogue.tsv
│   ├── backtest_trades.csv         ← FROM bold-heisenberg
│   ├── backtest_equity.csv         ← FROM bold-heisenberg
│   ├── backtest_summary.txt        ← FROM bold-heisenberg
│   ├── backtest_rejected.csv       ← FROM bold-heisenberg
│   ├── cached_signals_2024.csv     ← FROM bold-heisenberg (CRITICAL — signal cache)
│   ├── holding_period_sweep.csv    ← FROM bold-heisenberg
│   ├── threshold_sweep.csv         ← FROM bold-heisenberg
│   └── overnight_progress.json
│
├── models/                         ← Trained model artifacts (all gitignored)
│   ├── .gitkeep
│   └── [.keras, .pkl, _config.json files]
│
├── docs/                           ← All design docs, research, and references
│   ├── FPPE_v2_briefing.md         ← Already here
│   ├── FPPE_TRADING_SYSTEM_DESIGN.md  ← MOVE from bold-heisenberg root
│   ├── CANDLESTICK_CATEGORIZATION_DESIGN.md  ← MOVE from bold-heisenberg root
│   ├── PHASE1_FILE_REVIEW.md       ← MOVE from bold-heisenberg root
│   ├── PHASE2_SYSTEM_DESIGN.md     ← MOVE from current root
│   ├── TECH_DEBT_AUDIT.md          ← MOVE from current root
│   └── research/                   ← Research documents subfolder
│       └── Gemini FPPE Research/   ← MOVE from root
│           └── [10 .docx files]
│
├── archive/                        ← Superseded legacy files (reference only)
│   └── [existing archive contents — unchanged]
│
├── scripts/                        ← Utility/one-off scripts
│   └── bss_regression_test.py
│
├── .github/                        ← CI configuration
│   └── workflows/test.yml
│
└── .gitignore                      ← Update: add data/results/ exclusion
```

---

## 3. EXECUTION PLAN

Execute these steps **in order**. Do NOT skip steps. Each step is safe only after the prior step is verified.

### STEP 1: Merge bold-heisenberg into current branch (CRITICAL — Do first)

**Why first:** All subsequent steps depend on having Phase 1 files in the working tree. If you prune the worktree before merging, the files are lost.

**Risk: LOW** — bold-heisenberg is ahead of tech-debt-remediation-p0 on trading_system. The only conflict risk is if config.py differs. Inspect the diff before merging.

```cmd
cd C:\Users\Isaia\.claude\financial-research

REM First, inspect what bold-heisenberg has that current branch doesn't
git diff tech-debt-remediation-p0..claude/bold-heisenberg --name-only

REM Check for trading_system diff specifically
git diff tech-debt-remediation-p0..claude/bold-heisenberg -- trading_system/

REM Merge bold-heisenberg into current branch
git merge claude/bold-heisenberg --no-ff -m "Merge Phase 1 trading system from bold-heisenberg"

REM If conflicts occur in config.py:
REM Keep current branch version of trading_system/config.py (it has the S5/D1/D2 fixes)
REM Accept all new files (backtest_engine.py, signal_adapter.py, __init__.py, run_phase1.py)
```

**Verify after merge:**
```cmd
REM Confirm all 5 trading_system files exist
dir trading_system\
REM Should show: __init__.py, config.py, signal_adapter.py, backtest_engine.py, run_phase1.py

REM Confirm new test files exist
dir tests\test_trading_config.py tests\test_signal_adapter.py tests\test_backtest_engine.py

REM Run full test suite
python -m pytest tests/ -v
REM All tests must pass before proceeding
```

### STEP 2: Move results from bold-heisenberg to main results folder

The Phase 1 backtest outputs exist only in the worktree. They must be copied to the main results folder before the worktree is pruned.

**The most critical file is `cached_signals_2024.csv`** — this is the signal cache that Phase 1 and Phase 2 backtests depend on. Without it, you cannot reproduce the 22.3% annual / Sharpe 1.82 result without re-running FPPE signal generation (which takes hours).

```cmd
REM IMPORTANT: Execute from the financial-research root

REM Copy Phase 1 results from worktree to main results folder
copy ".claude\worktrees\bold-heisenberg\results\cached_signals_2024.csv" "results\"
copy ".claude\worktrees\bold-heisenberg\results\backtest_trades.csv" "results\"
copy ".claude\worktrees\bold-heisenberg\results\backtest_equity.csv" "results\"
copy ".claude\worktrees\bold-heisenberg\results\backtest_summary.txt" "results\"
copy ".claude\worktrees\bold-heisenberg\results\backtest_rejected.csv" "results\"
copy ".claude\worktrees\bold-heisenberg\results\holding_period_sweep.csv" "results\"
copy ".claude\worktrees\bold-heisenberg\results\threshold_sweep.csv" "results\"
```

**Verify:** Confirm file sizes match between source and destination before proceeding.

### STEP 3: Move misplaced docs from worktree root

```cmd
REM Copy design docs that exist in bold-heisenberg but not in current branch
copy ".claude\worktrees\bold-heisenberg\CANDLESTICK_CATEGORIZATION_DESIGN.md" "docs\"
copy ".claude\worktrees\bold-heisenberg\FPPE_TRADING_SYSTEM_DESIGN.md" "docs\"
copy ".claude\worktrees\bold-heisenberg\PHASE1_FILE_REVIEW.md" "docs\"
```

### STEP 4: Reorganize docs at current root into docs/

```cmd
REM Move design docs from root to docs/
move "PHASE2_SYSTEM_DESIGN.md" "docs\"
move "TECH_DEBT_AUDIT.md" "docs\"
```

### STEP 5: Consolidate data/results/ into results/

The `data/results/experiments.tsv` file belongs in the top-level `results/` folder. Having results nested inside data is confusing — data is inputs, results are outputs.

```cmd
REM Move experiments.tsv to root results folder
copy "data\results\experiments.tsv" "results\experiments.tsv"

REM Verify the copy is intact (compare file sizes)
dir "data\results\experiments.tsv"
dir "results\experiments.tsv"

REM Then remove the data/results subfolder
del "data\results\experiments.tsv"
rmdir "data\results"
```

**Update .gitignore** to reflect this change:
```
# Before
results/*.csv
results/*.tsv
results/*.txt
results/*.json

# After (add explicit exclusion for data/results if it ever reappears)
data/results/
```

### STEP 6: Move Gemini Research folder under docs/

```cmd
mkdir "docs\research"
move "Gemini FPPE Research" "docs\research\Gemini FPPE Research"
```

**Update CLAUDE.md and PROJECT_GUIDE.md** to reflect the new path (`docs/research/Gemini FPPE Research/`).

### STEP 7: Delete root-level __pycache__

```cmd
REM This should not exist at the project root
rmdir /s /q __pycache__
```

Add to .gitignore if not already covered (it should be covered by `__pycache__/` already).

### STEP 8: Prune stale worktrees

Only execute this after Steps 1–7 are confirmed complete. Once you prune, the worktree files are gone.

```cmd
REM Prune worktrees that are no longer needed
git worktree prune

REM Verify which worktrees remain (should only be main)
git worktree list

REM Optionally delete the worktree branches that are now merged
git branch -d claude/bold-heisenberg
git branch -d claude/exciting-brattain
REM claude/nifty-swirles: check if it has anything unique before deleting
git log main..claude/nifty-swirles --oneline
REM If output is empty, it's fully merged: git branch -d claude/nifty-swirles
```

### STEP 9: Merge current branch to main

Once the restructuring is stable and all tests pass:

```cmd
git checkout main
git merge tech-debt-remediation-p0 --no-ff -m "Merge tech-debt cleanup and Phase 1 trading system to main"
git push origin main
```

### STEP 10: Update CLAUDE.md

Update the test count and key file references:

```
## Commands
- `python -m pytest tests/ -v` — Run all 331 tests (always run before committing)
                                           ^^^
                                           Update from 294 to actual post-merge count

## Key Files
- `docs/FPPE_TRADING_SYSTEM_DESIGN.md`   ← Updated path
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md`  ← Updated path
- `docs/PHASE1_FILE_REVIEW.md`           ← Updated path
- `docs/PHASE2_SYSTEM_DESIGN.md`         ← New entry
- `results/cached_signals_2024.csv`      ← CRITICAL: signal cache for backtests
```

---

## 4. RISK ASSESSMENT

| Step | Risk | Mitigation |
|------|------|------------|
| Step 1: Merge | config.py conflict | Inspect diff first; keep current branch config.py (has all Phase 1 bug fixes) |
| Step 2: Copy results | File corruption | Verify sizes before deleting source |
| Step 2: cached_signals_2024.csv lost | Cannot reproduce Phase 1 results | Copy this file FIRST; verify before touching anything else |
| Step 5: Delete data/results/ | experiments.tsv lost | Copy to results/ first, verify, then delete |
| Step 8: Prune worktrees | Source files gone forever | Only prune after confirming all files are in main branch |
| Step 9: Merge to main | Regression | Full test suite must pass before this step |

**Absolute no-regret rule: Never delete a worktree before verifying its unique files are in the main branch.**

---

## 5. FILES THAT MUST NOT BE LOST

These files are irreplaceable (cannot be regenerated without significant compute or work):

| File | Where It Lives Now | Why Irreplaceable |
|------|--------------------|-------------------|
| `cached_signals_2024.csv` | bold-heisenberg/results/ | FPPE K-NN signal run for full 2024 year; takes hours to regenerate |
| `backtest_trades.csv` | bold-heisenberg/results/ | Phase 1 trade log; provenance for 22.3% annual result |
| `threshold_sweep.csv` | bold-heisenberg/results/ | Empirical data behind confidence_threshold=0.60 choice |
| `holding_period_sweep.csv` | bold-heisenberg/results/ | Empirical data behind max_holding_days=14 choice |
| `backtest_engine.py` | bold-heisenberg/trading_system/ | 42KB, all Phase 1 bugs fixed; would take days to rewrite |
| `signal_adapter.py` | bold-heisenberg/trading_system/ | Normalizes FPPE output to UnifiedSignal |
| `data/full_analogue_db.parquet` | Main data/ folder | 54MB training database; takes hours to rebuild |
| `data/train_db.parquet` | Main data/ folder | 48MB; training split |
| `experiments.tsv` | data/results/ | Experiment log for all historical sweep runs |

---

## 6. FILES THAT ARE SAFE TO DELETE

| File | Reason Safe |
|------|-------------|
| `archive/*.py` | Superseded by pattern_engine; already archived by tech-debt work |
| `claude/exciting-brattain` worktree | Merged to main already |
| Root-level `__pycache__/` | Regenerated automatically by Python |
| `data/prepared_data.npz` | Regenerated by prepare.py (70MB; also gitignored) |
| Model `.keras` files | Regenerated by training scripts (all gitignored anyway) |
| Old `.pkl` scalers (per-ticker) | Regenerated; gitignored |

---

## 7. WHAT THE CLEAN STRUCTURE SOLVES

| Problem | Solution |
|---------|---------|
| Phase 1 trading system "missing" | Files exist in bold-heisenberg; Step 1 merges them |
| Cannot run 331 tests from main | After Step 1, all tests run from one location |
| Results in two locations (`data/results/` and `results/`) | Step 5 consolidates to `results/` only |
| Design docs scattered at root and in worktrees | Steps 3–4 move all to `docs/` |
| `Gemini FPPE Research` folder at root level | Step 6 moves to `docs/research/` |
| Stale worktrees consuming disk space | Step 8 prunes after content is safely merged |
| Main branch is 2 months stale | Step 9 catches main up |
| Root `__pycache__/` confusing | Step 7 deletes it |

---

## 8. TIMELINE ESTIMATE

This is all manual git operations on Windows. Assuming you follow the steps carefully:

| Step | Time Estimate |
|------|--------------|
| Step 1: Merge (+ conflict resolution if any) | 15–30 min |
| Step 2: Copy results files | 5 min |
| Step 3–4: Move docs | 5 min |
| Step 5: Consolidate data/results | 5 min |
| Step 6: Move Gemini folder | 2 min |
| Step 7: Delete pycache | 1 min |
| Step 8: Prune worktrees | 5 min |
| Step 9: Merge to main + push | 10 min |
| Step 10: Update CLAUDE.md | 10 min |
| Full test suite run to verify | 10–15 min |
| **Total** | **~70–90 min** |

---

*Reorganization Plan v1.0 — March 19, 2026*
*Execute Steps 1–2 immediately to prevent data loss. Steps 3–10 can be done at any time after Step 1 is complete.*
