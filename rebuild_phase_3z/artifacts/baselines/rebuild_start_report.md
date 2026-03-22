# Phase 3Z Rebuild — Start Report

**Date:** 2026-03-21
**Purpose:** Frozen baseline metrics for parity verification throughout Phase 3Z rebuild
**Status:** FROZEN — do not edit after this date

---

## Test Suite Baseline

| Metric | Value |
|--------|-------|
| Total tests | 596 |
| All passing | Yes |
| Framework | pytest 9.0.2 |
| Python version | 3.12.10 |
| Platform | Windows 11 |

## Backtest Performance Baselines

### Phase 3 (Aggressive Profile, 2024 Fold)
Source: `results/backtest_summary.txt`

| Metric | Value |
|--------|-------|
| Annual return | 18.5% |
| Sharpe ratio | 1.16 |
| Max drawdown | 6.7% |
| Total trades | 191 |
| Win rate | 51.8% |
| Hold period | 14 days (optimal) |

### Phase 2 (Risk Engine)
Source: `docs/PHASE2_RESULTS.md`

| Metric | Value |
|--------|-------|
| Annual return | 15.76% |
| Sharpe ratio | 1.376 |
| Max drawdown | 3.81% |

### Phase 1 (Backtest Engine baseline from holding sweep)
Source: `CLAUDE.md` locked settings + results

| Metric | Value |
|--------|-------|
| Annual return | ~22.3% (aggressive, 14d hold) |
| Sharpe ratio | 1.82 |
| Max drawdown | ~6.9% |

### Phase 3 (Portfolio Manager, walk-forward validation)
Source: `results/backtest_summary.txt`, AGENTS.md

| Metric | Value |
|--------|-------|
| Annual return | 15.1% |
| Sharpe ratio | 1.93 |
| Max drawdown | 3.9% |

## Walk-Forward Validation

| Metric | Value |
|--------|-------|
| BSS (2024 fold) | +0.00103 |
| Folds | 6 (expanding window, 2019–2024) |
| Horizon | fwd_7d_up |
| Source | Walk-forward runner experiments.tsv |

## Locked Settings

```
Distance metric:       Euclidean
Distance weighting:    uniform
Feature set:           returns_only (8 features)
Calibration:           Platt
cal_frac:              0.76
max_distance:          1.1019
top_k:                 50
confidence_threshold:  0.65 (pattern engine) / 0.60 (trading system)
regime:                binary
horizon:               fwd_7d_up
stop_loss_atr_multiple: 3.0
max_holding_days:      14
nn_jobs:               1
```

## Dataset Hashes

See: `rebuild_phase_3z/artifacts/manifests/baseline_manifest_20260321.json`

- 52 ticker CSVs (SHA-256 hashed, all verified)
- `results/cached_signals_2024.csv` (SHA-256 hashed)
- Verification script: `python scripts/hash_baseline_inputs.py`

## Parity Tolerances

| Data Type | Tolerance |
|-----------|-----------|
| Float values | rtol=1e-7 |
| Integer counts | exact |
| Categorical labels | exact |
| Signal counts | exact |
| Sharpe ratio | rtol=1e-4 |
| Annual return | rtol=1e-4 |
| BSS per fold | rtol=1e-5 |

---

*This report is the immutable reference for the Phase 3Z rebuild. Any metric deviation beyond the tolerances above is a regression, not progress.*
