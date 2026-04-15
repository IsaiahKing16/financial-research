# ADR-010: structlog Adoption for Phase 8+ Logging

**Date:** 2026-04-15
**Status:** PENDING (activates at T8.1)
**Task:** P8-PRE-5C (prep), T8.1 (activation)

## Decision

Phase 8+ modules (eod_pipeline.py and anything it imports) use structlog
for structured JSON output with context bindings:
trade_id, strategy_name, session_id, pipeline_run_id.

print() and logging.info() are replaced with structlog equivalents.
Existing modules use stdlib logging until they are touched during Phase 8.

## Why

A 60-day autonomous process needs machine-parseable logs for post-hoc
diagnosis. JSON logs are trivially filterable by trade_id or fold label.

## Current State (P8-PRE-5C)

All print() calls in pattern_engine/ and trading_system/ have been migrated
to stdlib logging. structlog activation is deferred to T8.1.
