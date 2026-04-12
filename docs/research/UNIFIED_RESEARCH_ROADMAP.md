# FPPE Unified Research Roadmap

**Created:** 2026-04-11 | **Status:** ACTIVE
**Purpose:** Map all research papers to actionable project phases, track what's done, what's next, and what's deferred.

---

## Priority Tiers

Papers are assigned to priority tiers based on impact on the critical path: getting FPPE live with capital.

| Tier | Meaning | Timeline |
|------|---------|----------|
| **DONE** | Research applied, results locked | — |
| **R1** | Critical path — blocks live deployment | Next 2 weeks |
| **R3** | High value — improves edge or infrastructure | Next 4 weeks |
| **R5** | Medium value — deepens understanding | Next 2 months |
| **R7** | Low urgency — nice to have | Backlog |
| **SHELVED** | Not actionable given current evidence | Revisit trigger defined |

---

## Paper Inventory (20 papers)

### DONE — Research Applied

| # | Paper | Artifact | Applied In | Key Outcome |
|---|-------|----------|------------|-------------|
| 1 | **Bayesian optimization for financial HP tuning** | `wf-f4e7ad04` | R3 Optuna Infrastructure | OptunaSweep + GridSweep + ExperimentLogger. TPE sampler, Wilcoxon p-values, trimmed_mean_bss. Implemented 2026-04-11 on `p3-optuna-infrastructure` branch. |
| 2 | **Optuna for walk-forward tuning** | `wf-26b4b173` | R3 Optuna Infrastructure | Confirmed Optuna as best framework. WilcoxonPruner deferred (6 folds too few for pruning benefit). 80-trial budget adopted. |
| 3 | **Distance metrics verdict** | `wf-ba2ede68` | Phase 7 E3 DTW | **DTW FAILED** (Spearman rho=1.0000 vs Euclidean). Euclidean confirmed as optimal for point-in-time return vectors. DTW/EMD inapplicable to this problem class. |
| 4 | **Conformal prediction guide** | `wf-976dc3b2` | Phase 7 E4 Conformal | **Conformal FAILED** on 52T. Root cause: probs cluster [0.50–0.59], coverage trivial. ACI over-tightens on COVID volatility. |
| 5 | **MAPIE for conformal prediction** | `wf-ca449770` | Phase 7 E4 Conformal | Confirmed MAPIE v1 lacks Mondrian support. Binary prediction sets ({up}, {down}, {up,down}) are fundamentally limited. |
| 6 | **PBO tooling + CSCV** | `wf-76c4bbd6` | R3 design (Wilcoxon) | 3-of-6 gate has 65.6% FPR under null → gate used for TPE steering only. Wilcoxon p-value added as proper significance test. CPCV deferred as future path. |
| 7 | **KNN calibration failure under pool dilution** | `wf-6833345c` (root) | Phase 1 BSS diagnosis | Root cause identified: analogue pool dilution at 585T scale. Platt helping (+0.023 BSS vs raw). Led to max_distance tightening (0.90 → 2.5 with 23D features). |

### R1 — Critical Path (Next)

| # | Paper | Artifact | Target Phase | Action Required |
|---|-------|----------|--------------|-----------------|
| 8 | **KNN rarely beats modern alternatives** | `wf-34d1c6c2` | H9 KNN vs LightGBM | **STATUS: READY.** Head-to-head walk-forward: KNN vs LightGBM on identical 52T data + 23D features. Use new OptunaSweep for LightGBM HP tuning. Plan: `docs/research/H9_VARIANT_PLAN_KNN_VS_LIGHTGBM.md`. Critical question: does gradient boosting overcome the [0.50–0.58] probability ceiling that limits KNN on 52T? |

### R3 — High Value

| # | Paper | Artifact | Target Phase | Action Required |
|---|-------|----------|--------------|-----------------|
| 9 | **Position sizing beyond Kelly** | `wf-9df4b680` | Risk Engine refinement | Fractional Kelly at 75–80% when edge uncertainty is ±5%. Current: Half-Kelly (50%). Baker & McHale shrinkage formula: α* = edge²/(edge² + σ²). Could increase position sizes by 50–60% while maintaining drawdown discipline. Requires BSS > 0 with narrow confidence intervals first. |
| 10 | **Feature selection and metric learning** | `wf-290ad335` | Feature engineering | LMNN/NCA metric learning could improve neighbor quality without expanding feature count. ReliefF as fast baseline. Pre-transform data before hnswlib L2 index. Blocked on: need R1 (H9) results first to know if KNN survives as primary model. |

### R5 — Medium Value

| # | Paper | Artifact | Target Phase | Action Required |
|---|-------|----------|--------------|-----------------|
| 11 | **Transaction cost analysis** | `wf-c3800467` | Phase 8 Paper Trading | Round-trip costs 4–19 bps, base case ~10–11 bps. At 5-day holding period: 2–5.3% annual cost drag. Must incorporate into backtest P&L before live deployment. Almgren-Chriss unnecessary at our scale (<0.25% ADV). |
| 12 | **NautilusTrader review** | `wf-dc7a502a` | Phase 10 | Most complete Python-native framework for IBKR execution. IBKR adapter has 15+ bug fixes in recent releases — not yet production-ready for unattended trading. Windows 11 + Python 3.12 supported. Plan: `docs/research/PHASE10_NAUTILUSTRADER_EVALUATION_PLAN.md`. |
| 13 | **Adversarial robustness / failure modes** | `wf-b5d09a49` | Risk assessment | Five compounding failure modes: survivorship bias, distribution shift, deflated Sharpe, data poisoning, look-ahead. 52-ticker universe has inherent survivorship bias (43% winners). Backtest Sharpe ≠ live Sharpe. Informs risk register, not a code task. |
| 14 | **Online ANN index maintenance / drift** | `wf-c9d33a61` | Drift monitoring | HNSW incremental insertion OK for append-only. Deletion churn degrades recall. Hybrid: hnswlib + exponential decay + periodic full rebuilds on ADWIN drift detection. `drift_monitor.py` already exists; this paper informs its evolution. |

### R7 — Backlog

| # | Paper | Artifact | Target Phase | Action Required |
|---|-------|----------|--------------|-----------------|
| 15 | **Mondrian conformal prediction** | `wf-447c8acf` | Post-ticker-expansion | Needs ≥19 calibration samples per regime to avoid trivial sets. 52T too small. Revisit when universe hits 1500+ tickers and probability range widens beyond [0.50–0.58]. |
| 16 | **Venn-ABERS predictors** | `wf-915a7c74` | Future calibration | Distribution-free calibration guarantees but O(k) online cost. No published financial applications. Exchangeability assumption violated by regime changes. Low priority unless beta_abm calibration degrades on expanded universe. |
| 17 | **STUMPY matrix profiles** | `wf-965b0840` | E6 STUMPY (DEFERRED) | Powerful for motif discovery / anomaly detection. No published evidence of actionable trading signals. z-normalization discards magnitude. Phase 7 E6 deferred pending empirical hypothesis. |

### SHELVED — Not Actionable

| # | Paper | Artifact | Reason | Revisit Trigger |
|---|-------|----------|--------|-----------------|
| 18 | **Synthetic data poisoning** | `wf-27f67be1` | Anti-recommendation: synthetic data poisons KNN analogue pools. SMOTE degrades Brier scores. | Never (for KNN). Revisit only if switching to classification model. |
| 19 | **Regime detection beyond thresholds** | `wf-cd046fcb` | H8 HMM SHELVED. H7 HOLD regime passed (mean_BSS=+0.00033). HMM adds complexity without clear BSS benefit at 52T scale. Plan: `docs/research/H8_VARIANT_PLAN_HMM_REGIME_UPGRADE.md`. | BSS regression on expanded universe, or H7 gate fails on 1500T. |
| 20 | **Conformal prediction (E2 plan)** | `wf-976dc3b2` | Phase 7 E2 DEFERRED. 52T probs in [0.50–0.59] make conformal intervals trivially wide. Plan: `docs/research/PHASE7_E2_CONFORMAL_PREDICTION_PLAN.md`. | Probability range widens beyond 0.65 threshold on expanded universe. |

---

## Critical Path Sequence

```
DONE: R3 Optuna Infrastructure (2026-04-11)
  │
  ├─→ R1: H9 KNN vs LightGBM (next — uses OptunaSweep)
  │     Question: Does LightGBM break the [0.50–0.58] probability ceiling?
  │     If YES → LightGBM becomes primary model, KNN becomes ensemble member
  │     If NO  → KNN remains primary, focus shifts to feature engineering (R3 #10)
  │
  ├─→ R3: 75% Kelly sizing (after H9 results)
  │     Requires: BSS > 0 with narrow CI from primary model
  │
  ├─→ R5: Transaction cost model (Phase 8 prep)
  │     10-11 bps round-trip → 2-5% annual drag
  │
  └─→ R5: NautilusTrader integration (Phase 10)
        IBKR adapter stabilizing; revisit in ~3 months
```

## Gemini Research (Supplementary)

The `docs/research/Gemini FPPE Research/` directory contains 20 documents covering architecture, cross-domain applications, candlestick features, psychology, military data strategies, and testing tools. These are background/context documents, not actionable research papers. Key items already absorbed:
- Candlestick features → Phase 6 (COMPLETE)
- Cross-domain architecture → Phase 3Z rebuild (COMPLETE)
- Testing tools → pytest infrastructure (COMPLETE)

---

## Provenance

This roadmap consolidates:
- 18 Claude Research compass artifacts (`docs/research/Claude Research/`)
- 1 root-level compass artifact (`compass_artifact_wf-6833345c`)
- 1 Gemini compass artifact (`docs/research/Gemini FPPE Research/compass_artifact_wf-caccb9e4`)
- 4 existing research plans (`H8`, `H9`, `Phase 7 E2`, `Phase 10`)
- Session log prioritization from 2026-04-10

Paper count: 18 Claude + 1 root + 1 Gemini = 20 total artifacts mapped. The session log's "15" likely excluded the 5 already-applied papers (now in DONE tier).
