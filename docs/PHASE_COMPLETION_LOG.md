# Phase Completion Log — FPPE

**Purpose:** Historical record of all completed phases with key metrics. Extracted
from CLAUDE.md to keep the root config file concise.

---

## Phase 1 — Backtest Engine (2026-03-17)
- Equal-weight 5% sizing, 278 trades, 2024 fold.
- Ann. 22.3%, Sharpe 1.82, MaxDD 6.9%. Beats SPY risk-adjusted.

## Phase 2 — Risk Engine (2026-03-19)
- Half-Kelly ATR-based sizing. Sharpe=2.527 (after ATR sweep locked 3.0x).
- stop_loss_atr_multiple=3.0 locked. Provenance: results/atr_sweep_results.tsv

## Phase 3 — Portfolio Manager (2026-03-21)
- Risk engine integration. Sharpe=2.659, MaxDD=4.3%.
- Provenance: results/phase3_gate_check.txt

## Phase 3.5 — Research Integration (2026-03-21)
- EMD+BMA: NEGATIVE. ATR sweep: 3.0x LOCKED. SlipDeficit TTF gate wired.
- HNSW (SLE-47): 54.5x speedup, recall@50=0.9996. Vectorized query (SLE-48): 12,508 rows/sec.

## Phase 4 — Strategy Evaluator / PM Filter (2026-04-06)
- PM confidence ranking + sector allocation + capital queue.
- Sharpe=2.649, MaxDD=4.4%. Provenance: results/phase4/ (when backtest re-run)

## Phase 5 — Live Execution Plumbing (2026-04-07)
- OrderManager, MockBroker, LiveRunner, reconciliation. G1-G3 gates passed.
- No backtest metric change (execution plumbing only).

## Phase 6 — Candlestick Features (2026-04-09)
- returns_candle(23D), max_distance=2.5, wins 5/6 folds vs returns_only.
- 846 tests at phase completion.
- Provenance: results/phase6/bss_comparison_candle_vs_baseline.tsv

## Phase 7 — Model Enhancements (2026-04-10)
- E1-E4 ALL FAIL. Root cause: 52T probs cluster [0.50-0.59], below 0.65 threshold.
- E5/E6 deferred. All 6 enhancement flags remain False.
- Provenance: results/phase7/enhancement_summary.tsv

## R3 — Optuna Infrastructure (2026-04-11)
- OptunaSweep (TPE) + GridSweep. 908 tests.
- Research roadmap: docs/research/UNIFIED_RESEARCH_ROADMAP.md

## P8-PRE-4 — Feature Standardization (2026-04-15)
- ADR-007: VOL_NORM standardization for all feature sets.

## P8-PRE-5 — Power of 10 Hardening (2026-04-15)
- FiniteFloat type, MAX_* constants, TradingSystemError hierarchy.
- Static analysis CI: Ruff/mypy/Bandit baselines. icontract guards.
- ADRs: ADR-008, ADR-009, ADR-011. 927 tests at completion.

## P8-PRE-6 — Power of 10 Retroactive Audit (2026-04-15)
- Audited all production files against NASA P10 rules.
- R7 FIXED: features.py silent swallow. R4 improved: CC 35->18, no grade-D/E remain.
- icontract contracts: size_position, risk_engine, matcher.fit/query.
- FiniteFloat: portfolio_state (equity, cash, confidence, position_pct).
- ADR-012: docs/adr/ADR-012-power-of-10-retroactive-audit.md. 945 tests pass.

## P8-PRE-1 — 585T Full-Stack Revalidation (2026-04-16)
- Result: FAIL
- Gate metrics: BSS 0/6 positive folds, Sharpe 0.04, MaxDD 4.2%, Trades 200, WR 47.5%
- Failed gates: G1 (BSS), G2 (Sharpe), G5 (Win rate). Passed: G3 (MaxDD), G4 (Trades).
- Root cause: Resolution ≈ 0 at 585T — pool dilution destroys KNN discriminative power.
  585T Resolution=0.000709 vs 52T Resolution=0.007621 (Phase 1 Murphy decomposition).
- Calibration: Platt (585T production path, bypasses walkforward beta_abm monkey-patch)
- Commit: 6051142 (branch: feature/p8-pre-1-585t-revalidation)
- Script: scripts/run_585t_full_stack.py
- Provenance: results/phase8_pre/585t_gate_check.txt, results/phase8_pre/585t_walkforward.tsv
- T8.1 BLOCKED. Recovery campaign launched: docs/campaigns/P8_RECOVERY_CAMPAIGN.md
