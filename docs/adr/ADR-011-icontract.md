# ADR-011: icontract for ML Pipeline Design-by-Contract

**Date:** 2026-04-15
**Status:** ACCEPTED
**Task:** P8-PRE-5E

## Decision

icontract @require and @ensure decorators on critical functions in:
pattern_engine/matcher.py (_prepare_features), pattern_engine/walkforward.py (run_fold temporal assertions + BSS identity guard).

Minimum: 2 contract conditions per decorated function (1 pre + 1 post).
Contracts disabled via PYTHONOPTIMIZE=1 if runtime overhead is a concern
(icontract overhead: ~3–6 μs per condition).

## Why

Microsoft Research (Kudrjavets et al., ISSRE 2006) found assertion density
correlates inversely with defect density. NASA mandates ≥2 per function.
icontract provides ViolationError with readable condition descriptions,
unlike raw assert which is stripped by Python -O.

## Implementation

- `_prepare_features`: @require(finite input) + @ensure(finite output)
- `run_fold`: RuntimeError guards on train_end >= val_start and BSS identity
  (|REL-RES+UNC-BS| > 1e-3 threshold; Murphy binning noise ~3e-5)
