# ADR-008: Static Analysis Toolchain Adoption (Power of 10)

**Date:** 2026-04-15
**Status:** ACCEPTED
**Task:** P8-PRE-5D

## Decision

Adopt a 5-layer static analysis toolchain gated in CI:
- **Ruff**: lint + format + complexity (max CC=10 core, 12 ETL)
- **mypy --strict**: type checking (zero new errors policy)
- **Bandit**: security scanning (zero HIGH findings)
- **xenon**: complexity grade enforcement (max grade B)

## Adoption Policy

Existing violations are **baselined** (see results/phase8_pre/*_baseline.txt).
New code must not introduce new violations. Existing violations are fixed
opportunistically when touching a file.

## Configuration

See `pyproject.toml` for full configuration.
