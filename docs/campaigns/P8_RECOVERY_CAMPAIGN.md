# CAMPAIGN: P8-PRE-1 Failure Recovery — Three-Track Research
**Created:** 2026-04-16  
**Advisor:** Claude Opus (strategic)  
**Executor:** Claude Code (implementation)  
**Status:** ACTIVE — triggered by P8-PRE-1 FAIL (2026-04-16)  
**Addendum integrated:** 2026-04-16 (equal weights, connector similarity sweep)

---

## §0 CONTEXT & TRIGGER

### P8-PRE-1 Result Summary

585T full-stack walk-forward (Platt calibration, returns_candle 23D features, H7 regime hold):

| Gate | Value | Status |
|------|-------|--------|
| G1 BSS positive folds | 0/6 | FAIL |
| G2 Sharpe | 0.04 | FAIL |
| G3 MaxDD | 4.2% | PASS |
| G4 Trades | 200 | PASS |
| G5 Win rate | 47.5% | FAIL |

**Per-fold BSS:** 2019: -0.00153, 2020: -0.00130, 2021: -0.00131, 2022-Bear: -0.04417, 2023: -0.02419, 2024-Val: -0.00182

### Root Cause (Confirmed by Phase 1 Murphy Decomposition)

- 585T Resolution = 0.000709 (near-zero signal)
- 52T Resolution = 0.007621 (genuine signal)
- Pool dilution at 585T destroys KNN discriminative power
- Calibration method choice (Platt vs beta_abm) cannot fix Resolution ≈ 0

### Phase 2-7 Metrics Provenance Audit

The Sharpe=2.659 / MaxDD=4.3% results throughout Phase 2-7 were computed on:
- 52T walk-forward (Phase 2-4 experiments)
- 585T cached signals for single 2024 fold (Phase 4 gate check)

No full 585T walk-forward was ever executed end-to-end prior to P8-PRE-1.

---

## §1 CAMPAIGN STRATEGY

### 1.1 Three Parallel Tracks

This campaign runs three independent research tracks to identify a viable production architecture. Each track has distinct failure modes and distinct upside; running them in parallel hedges against any single track failing.

| Track | Approach | Architectural Change | Fallback If Fails |
|-------|----------|---------------------|-------------------|
| **A** | Dynamic universe sizing | Reduce universe, keep KNN | Track B or C |
| **B** | Per-sector pools + connectors | Partition search space, keep KNN global universe | Track A or C |
| **C** | LightGBM prediction engine | Replace KNN entirely | Track A or B |

### 1.2 Execution Order

Sequential, not parallel, to preserve compute budget and avoid cross-contamination of findings:

1. **Track A first** — cheapest, maps the KNN capacity curve, informs B and C baselines
2. **Track B second** — uses Track A findings to size per-sector pools correctly
3. **Track C third** — activates only if A and B fail, OR runs regardless for R2-H9 scheduled validation

### 1.3 Common Infrastructure (Build Once, Use Everywhere)

Before any track runs, build shared infrastructure:

1. **Per-fold Murphy B3 decomposition** must be computed for every experiment
2. **Walk-forward harness factored** so algorithm/universe/features swap cleanly
3. **Shared gate metric computation** (BSS, Sharpe, MaxDD, win rate) — single implementation
4. **Experiment logging** — every run writes to `results/campaign_p8_recovery/<track>/<experiment_id>/`

### 1.4 Campaign Gate Logic

```
Track A completes
  ├── PASS at any universe size → Lock winning config, proceed to P8-PRE-1 retest
  └── FAIL at all sizes → Track B activates (still mandatory)

Track B completes  
  ├── PASS → Lock sector-pool architecture, proceed to P8-PRE-1 retest
  └── FAIL → Track C activates

Track C completes (or runs regardless)
  ├── LightGBM BSS > KNN best (A or B) → Pivot prediction engine
  ├── KNN best beats LightGBM → Keep KNN, document capacity finding
  └── Both fail → ESCALATE. Phase 8 fundamentally blocked.
```

### 1.5 Provenance Requirements

Every experiment in this campaign must produce:
- `config.json` — exact EngineConfig used
- `walkforward.tsv` — per-fold BSS, Sharpe, MaxDD, trades, win_rate
- `murphy_b3.tsv` — per-fold Resolution, Reliability, Uncertainty
- `gate_check.txt` — human-readable PASS/FAIL summary
- Git commit SHA in campaign log

No numeric claim in any summary is valid without a file path pointing to one of the above.

---

## §2 TRACK A — DYNAMIC UNIVERSE SIZING

### 2.1 Hypothesis

KNN discriminative power exists at some universe size between 52T and 585T. A universe-size sweep with **intelligently selected tickers** (not just top-N by market cap) will identify the capacity crossover where BSS turns positive.

### 2.2 Key Insight (User Requirement)

The universe selection must be **dynamic and criterion-driven**, not fixed lists. The system must choose N tickers from a master pool using multiple signals, not naive market-cap ranking.

### 2.3 Selection Criteria — Equal Weights (v1)

**All six criteria are equally weighted at 0.1667.** This is a deliberate choice to avoid encoding prior assumptions into the first measurement. Weight adjustment sweep runs only if Track A produces at least one passing universe size.

Each candidate ticker gets a composite score across these dimensions. All criteria computed **point-in-time** using only data available before the fold train period starts.

| Criterion | Weight (v1) | Computation | Rationale |
|-----------|-------------|-------------|-----------|
| Liquidity | 0.1667 | 60-day median ADV in USD, z-scored across universe | Thinly-traded tickers produce noisy features |
| Data completeness | 0.1667 | % trading days with valid OHLCV over lookback window | Gaps inject discontinuities into return fingerprints |
| Volatility regime coverage | 0.1667 | Rolling 30D vol distribution coverage vs universe median | Universe should span volatility spectrum, not cluster |
| Sector representation | 0.1667 | Inverse sector count (prefer tickers from under-represented sectors) | Prevents sector concentration at small universe sizes |
| Signal contribution | 0.1667 | Hold-out BSS contribution from prior fold (0 for first fold) | Directly rewards tickers that produce useful analogues |
| Survival-adjusted inclusion | 0.1667 | Include delisted tickers proportional to historical base rate | Prevents survivorship bias at small universe sizes |

Implementation note: weights must be a FiniteFloat-validated Pydantic field that sums to 1.0 ± 1e-6.

### 2.3a Mandatory Pre-Sweep Diagnostic

Before Track A begins, produce per-criterion ranked TSV for a fixed reference date (2019-01-01 train boundary) to verify:

1. Each criterion produces a non-degenerate distribution (not all same value)
2. Criteria are not near-perfectly correlated (if two are >0.95 correlated, effective dimension is 5, not 6)
3. No criterion produces NaN/Inf for any ticker

Output: `results/campaign_p8_recovery/track_a/criterion_diagnostics.tsv`

If two criteria are >0.95 correlated, drop one before running the sweep and document in ADR-014.

### 2.3b Deferred Weight Sweep (Activates Only If Track A Passes)

After Track A identifies a passing universe size N*, run a secondary Optuna sweep over weights:
- **Budget:** 50 trials at N* only
- **Objective:** mean BSS across 6 walk-forward folds
- **Constraint:** weights sum to 1.0 (Dirichlet-like parametrization: 5 free, 6th = 1 − sum)
- **Gate:** winning vector must improve mean BSS by ≥ +0.002 vs equal weights to be adopted

If no vector improves by ≥ +0.002, equal weights remain default. Document null result in ADR-014.

### 2.4 Universe Size Sweep

Test universe sizes: **N ∈ {52, 100, 150, 200, 300, 400, 585}**

For each N:
1. Run ticker selector to pick top-N by composite score (from master pool of all 585 tickers)
2. Rebuild feature matrix restricted to selected N tickers
3. Run full 6-fold walk-forward with locked config (returns_candle 23D, max_distance=2.5, top_k=50, H7 regime hold)
4. Compute BSS + Murphy B3 per fold
5. Compute trading metrics (Sharpe, MaxDD, trades, win rate)

### 2.5 Re-selection Cadence

Re-run ticker selection at the start of each walk-forward fold (not once per sweep). This mirrors production where universe composition should adapt as liquidity/sector balance shifts over years.

### 2.6 Data Leakage Guards

CRITICAL: The selector runs on data available **before** the fold's train period starts. No future information allowed. Test this explicitly:

- `tests/unit/test_dynamic_selector_no_leakage.py` — Assert selection for fold T uses only data with timestamp < fold T train start

### 2.7 Handoff Spec (Track A)

```xml
<goal>
Build a dynamic ticker selector (composite score across 6 equal-weight criteria) and run a
universe-size sweep across N ∈ {52, 100, 150, 200, 300, 400, 585} on the 585T master pool.
Run criterion_diagnostics.tsv first (§2.3a). For each universe size, run full 6-fold
walk-forward with Platt calibration, returns_candle(23) features, and H7 regime hold.
Produce BSS, Murphy B3, and trading metrics per fold.
</goal>

<success_criteria>
1. At least one universe size N produces BSS > 0 on >= 3/6 folds
2. At that universe size: Sharpe >= 1.0, MaxDD <= 15%, Trades >= 50, Win rate >= 50%
3. Resolution > 0.002 in Murphy decomposition (discriminative signal exists)
4. No data leakage: selector uses only point-in-time data (test passes)
</success_criteria>

<files_to_modify>
- pattern_engine/universe_selector.py (NEW) - composite scoring + selection
- pattern_engine/contracts/selector.py (NEW) - SelectionCriteria Pydantic schema
- scripts/run_track_a_universe_sweep.py (NEW) - sweep driver
- tests/unit/test_universe_selector.py (NEW)
- tests/unit/test_dynamic_selector_no_leakage.py (NEW) - CRITICAL leakage test
- results/campaign_p8_recovery/track_a/ - outputs
- docs/adr/ADR-014-dynamic-universe-selection.md (NEW)
</files_to_modify>

<steps>
1. Create SelectionCriteria Pydantic schema with 6 equal-weight components (FiniteFloat weights, sum=1.0)
2. Run criterion_diagnostics.tsv (§2.3a) FIRST — drop any >0.95 correlated criterion before proceeding
3. Implement UniverseSelector.score(ticker, as_of_date) -> float
4. Implement UniverseSelector.select_top_n(n, as_of_date) -> list[str]
5. Write leakage test FIRST (TDD mandate): assert future-data access raises RuntimeError
6. Write unit tests for each of 6 criteria with known-value fixtures
7. Build sweep driver that iterates N ∈ {52, 100, 150, 200, 300, 400, 585}
8. Per N: select tickers, build feature matrix, run 6-fold walkforward, compute BSS + Murphy
9. Write campaign-level TSV: results/campaign_p8_recovery/track_a/summary.tsv with columns
   (N, mean_bss, pos_folds, mean_sharpe, mean_maxdd, mean_resolution, mean_reliability)
10. Generate gate_check.txt identifying winning N (if any)
11. If passing N found: run deferred weight sweep (§2.3b, 50 Optuna trials at N*)
12. Create ADR-014 documenting selector design, winning N, and weight sweep result
</steps>

<verification>
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_universe_selector.py tests/unit/test_dynamic_selector_no_leakage.py -v
PYTHONUTF8=1 py -3.12 scripts/run_track_a_universe_sweep.py
cat results/campaign_p8_recovery/track_a/gate_check.txt
</verification>

<task_type>SR</task_type>
```

### 2.8 Expected Outcomes (A Priori Reasoning)

Based on Phase 1 Murphy decomposition:
- **N=52 to N=100:** Likely positive BSS (signal exists). But probability compression at N<100 limits tradeable signal strength.
- **N=150 to N=300:** The unknown zone. If Resolution degrades gracefully, this is the sweet spot.
- **N=400+:** Likely still broken (Resolution ≈ 0 regime).

If the crossover is at N<150, capacity is severely limited — prompt consideration of Track B or C even if Track A passes.

---

## §3 TRACK B — PER-SECTOR POOLS WITH CROSS-SECTOR CONNECTORS

### 3.1 Hypothesis

Global 585T KNN search dilutes signal because cross-sector analogues dominate the nearest-neighbor pool. Partitioning into per-sector pools restricts each query to same-sector analogues, preserving effective pool size near 52T while retaining the full 585T universe. **Cross-sector connectors** prevent loss of valid cross-sector analogues for tickers with comparable market-cap/liquidity profiles.

### 3.2 Architecture

```
                                  QUERY: ticker X
                                         |
                          ┌──────────────┼──────────────┐
                          |              |              |
                   Own-sector pool   Connector graph   (Optional) global
                   (top_k=40)         (cross-sector     (top_k=5, penalized)
                                      top_k=10)
                          |              |              |
                          └──────────────┼──────────────┘
                                         |
                                  Merge + rank by distance
                                         |
                                  Calibrate (per-sector Platt)
                                         |
                                  Final probability
```

### 3.3 Sector Pool Construction

- 11 GICS sectors (approx 50-60 tickers each from 585T universe)
- One HNSW index per sector, stored in `indices/sector_<SECTOR>.idx`
- Per-sector Platt calibration (each sector's probabilities fit independently)

### 3.4 Cross-Sector Connector Graph

**Purpose:** Allow cross-sector analogue lookups for tickers whose market-cap and liquidity profile matches peers in other sectors.

**Connector computation (rebuilt quarterly):**

1. For each ticker X, compute two features: `log_market_cap_pct` and `log_adv_pct`
2. Define "magnitude profile" as the 2D point (log_market_cap_pct, log_adv_pct)
3. For each ticker X in sector S, find top-M=10 cross-sector peers in magnitude space
4. Store as `connectors[X] = list of (peer_ticker, peer_sector, magnitude_distance)`
5. Store graph in `data/connectors/connectors_<YYYY-Q>.parquet`

### 3.5 Query-Time Logic

```python
def query(ticker_x, feature_vec, top_k=50):
    own_sector = SECTOR_MAP[ticker_x]
    own_results = sector_indices[own_sector].query(feature_vec, k=40)
    connector_peers = connectors[ticker_x]  # ~10 peers
    cross_results = []
    for peer_ticker, peer_sector, mag_dist in connector_peers:
        peer_index = sector_indices[peer_sector]
        peer_vec_results = peer_index.query(feature_vec, k=2)
        for result in peer_vec_results:
            result.distance *= (1 + CROSS_SECTOR_PENALTY * mag_dist)
        cross_results.extend(peer_vec_results)
    merged = own_results + cross_results
    merged.sort(key=lambda r: r.distance)
    return merged[:top_k]
```

### 3.6 Key Design Parameters

| Parameter | Initial Value | Sweep Range | Rationale |
|-----------|---------------|-------------|-----------|
| `sector_top_k` | 40 | {30, 40, 50} | Primary analogue pool size |
| `connector_top_k` | 10 | {5, 10, 15, 20} | Cross-sector pool size |
| `connector_peers_per_ticker` | 10 | {5, 10, 20} | How many magnitude-peers per ticker |
| `CROSS_SECTOR_PENALTY` | 0.1 | {0.0, 0.05, 0.1, 0.2} | Distance penalty for cross-sector matches |
| `connector_rebuild_freq` | Quarterly | Fixed | Balance freshness vs. stability |

### 3.7 Ablation Tests (Mandatory)

1. **B.1 Per-sector pools only** (no connectors) — baseline for B
2. **B.2 Per-sector pools + connectors** — full architecture
3. **B.3 Global pool + per-sector calibration** — does calibration split alone help?
4. **B.4 Global pool + magnitude filter** — does magnitude filtering alone help?

B.2 must beat all three ablations to justify its complexity.

### 3.8 Connector Similarity Dimension Sweep (Conditional on B.2 > B.1)

**Runs only if B.2 beats B.1 by ≥ +0.001 mean BSS.** Six candidate metrics, each with 4 mandatory adversarial checks.

#### Candidate Metrics (ordered by complexity)

| Metric | Description | Key risk |
|--------|-------------|----------|
| **S1** (Baseline v1) | 2D: log market cap pct + log ADV pct | Fails to capture beta/systematic risk |
| **S2** | S1 + rolling 60/252d beta to SPY | Beta unstable across regimes |
| **S3** | GICS 4-digit industry group + S1 magnitude | Requires extended SECTOR_MAP; small pools at 4-digit level |
| **S4** | Fama-French 4-factor loadings (market, SMB, HML, MOM) | Noisy 252d window; requires FF daily factor data |
| **S5** | LMNN-learned 8D projection from 23D returns_candle | High overfit risk at small pool sizes; reuses R1-E5 infrastructure |
| **S6** | Composite weighted ensemble of S1-S4 | Extra degrees of freedom multiply overfit; nested CV mandatory |

#### Adversarial Checks (Mandatory Per Metric S1–S6)

For EACH metric, before claiming a gate pass:

1. **Recovery check:** `pct_connectors_same_4digit_industry` — if >80%, metric is a GICS proxy, not adding info. Drop.
2. **Degeneracy check:** Count tickers with zero connectors. If >5% of universe, metric too restrictive.
3. **Stability check:** Between consecutive folds, fraction of top-10 connectors that change. If >60% turnover, metric tracks noise.
4. **Null comparison:** Run with random connectors as baseline. If random ≈ metric BSS, the improvement is from pool size, not connector quality.

All four checks write to: `results/campaign_p8_recovery/track_b/metric_<S#>_diagnostics.tsv`

#### Sweep Execution Protocol

```
Stage 1: S1 baseline (B.2 variant already gates this)
  ├── B.2 wins vs B.1 by >= +0.001 → Stage 2
  └── Does not beat B.1 → skip sweep

Stage 2: S2 (3D + beta)
  ├── Beats S1 by >= +0.001 → adopt as baseline, Stage 3
  └── Does not beat → keep S1, Stage 3

Stage 3: S3 (GICS hybrid)
  ├── Beats current baseline by >= +0.001 → adopt, Stage 4
  └── → keep current, Stage 4

Stage 4: S4 (Fama-French factor loadings)
  ├── Beats current baseline by >= +0.001 → adopt, Stage 5
  └── → keep current, Stage 5

Stage 5: S5 (LMNN) — BRANCH, compare independently to best of S1-S4

Stage 6: S6 (composite) — OPTIONAL, only if ≥2 of S1-S4 showed partial signal
  Nested CV mandatory (3-fold inner for weight tuning, 6-fold outer)
  Must beat best individual by >= +0.002 to be adopted
```

### 3.9 Handoff Spec (Track B)

```xml
<goal>
Build per-sector KNN pools with cross-sector connectors indexed by market-cap and ADV
magnitude profile. Run 6-fold walk-forward on 585T universe. Run 4 ablation variants
(B.1-B.4) to isolate the contribution of sector partitioning vs. connectors vs. per-sector
calibration. IF B.2 beats B.1 by >= +0.001 mean BSS, run connector similarity sweep S1-S6
with 4 adversarial checks per metric.
</goal>

<success_criteria>
1. B.2 (full architecture) produces BSS > 0 on >= 3/6 folds on 585T
2. B.2 beats each of B.1, B.3, B.4 by >= +0.001 mean BSS
3. Connector computation has no lookahead (point-in-time test passes)
4. Sharpe >= 1.0, MaxDD <= 15%, Trades >= 50, Win rate >= 50% on B.2
5. Resolution > 0.002 on B.2 in Murphy decomposition
6. (If sweep runs) At least one S1-S5 metric passes all 4 adversarial checks
7. (If sweep runs) Random-connector null shows BSS delta >= -0.002 vs winning metric
</success_criteria>

<files_to_modify>
- pattern_engine/sector_pools.py (NEW) - SectorPoolIndex class
- pattern_engine/connectors.py (NEW) - ConnectorGraph builder + loader
- pattern_engine/connectors/magnitude.py (NEW) - S1 implementation
- pattern_engine/connectors/beta.py (NEW) - S2 implementation
- pattern_engine/connectors/gics_hybrid.py (NEW) - S3 (requires extended SECTOR_MAP)
- pattern_engine/connectors/factor_loadings.py (NEW) - S4 (requires FF factor data)
- pattern_engine/connectors/learned_lmnn.py (NEW) - S5 (requires metric-learn)
- pattern_engine/connectors/composite.py (NEW) - S6
- pattern_engine/connectors/random_baseline.py (NEW) - null baseline
- pattern_engine/matcher.py (MODIFY) - add sector_pool_mode flag
- pattern_engine/sector_map.py (MODIFY) - extend with industry_group (4-digit) field for S3
- data/connectors/ (NEW directory)
- data/fama_french/ (NEW directory) - daily FF factor time series
- scripts/build_connector_graph.py (NEW)
- scripts/run_track_b_sector_pools.py (NEW) - B.1-B.4 driver
- scripts/run_track_b_connector_sweep.py (NEW) - S1-S6 sweep driver
- scripts/load_fama_french_factors.py (NEW) - one-time FF data download
- tests/unit/test_sector_pools.py (NEW)
- tests/unit/test_connectors_no_lookahead.py (NEW) - CRITICAL
- tests/unit/test_connector_magnitude.py (NEW)
- tests/unit/test_connector_beta.py (NEW)
- tests/unit/test_connector_gics_hybrid.py (NEW)
- tests/unit/test_connector_factor_loadings.py (NEW)
- tests/unit/test_connector_learned_lmnn.py (NEW)
- tests/unit/test_connector_adversarial_checks.py (NEW)
- results/campaign_p8_recovery/track_b/ - outputs
- docs/adr/ADR-015-per-sector-pools-connectors.md (NEW)
- requirements.txt (MODIFY) - add metric-learn>=0.7.0 (for S5)
</files_to_modify>

<steps>
1. Write lookahead test FIRST: assert connector_graph.build(as_of_date) cannot access data > as_of_date
2. Extend SECTOR_MAP with GICS industry_group (4-digit) for all 585 tickers (needed for S3)
3. Load Fama-French daily factors into data/fama_french/factors.parquet
4. Build ConnectorGraph class: magnitude profiles, top-M peers per ticker
5. Build SectorPoolIndex: wraps 11 HNSW indices, per-sector Platt calibrators
6. Modify PatternMatcher: accept sector_pool_mode in {None, "pure", "with_connectors"}
7. Build B.1-B.4 sweep driver: runs 4 variants, produces ablation delta table
8. IF B.2 beats B.1 by >= +0.001: build ConnectorMetric ABC (fit/peers interface)
9. Implement S1-S5 sequentially per sweep protocol (§3.8), adversarial checks after each
10. Conditional S6: only if >= 2 of S1-S4 showed partial signal, nested CV mandatory
11. Write campaign TSV: results/campaign_p8_recovery/track_b/connector_sweep_summary.tsv
    columns: (metric, mean_bss, pos_folds, recovery_pct, degeneracy_pct, stability_pct, null_delta, verdict)
12. Create ADR-015 with winning architecture (variant + metric) and ablation summary
</steps>

<verification>
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sector_pools.py tests/unit/test_connectors_no_lookahead.py tests/unit/test_connector_*.py -v
PYTHONUTF8=1 py -3.12 scripts/build_connector_graph.py --as_of 2018-12-31
PYTHONUTF8=1 py -3.12 scripts/run_track_b_sector_pools.py
PYTHONUTF8=1 py -3.12 scripts/run_track_b_connector_sweep.py
cat results/campaign_p8_recovery/track_b/gate_check.txt
cat results/campaign_p8_recovery/track_b/connector_sweep_summary.tsv
</verification>

<task_type>SR</task_type>
```

### 3.10 Compute Budget

| Stage | Per-fold | 6-fold total | Notes |
|-------|----------|--------------|-------|
| B.1-B.4 ablations | ~15 min | ~6 hrs | 4 variants × 6 folds |
| S1 magnitude | <1 min | <6 min | Negligible |
| S2 beta | 2-5 min | 15-30 min | Per-ticker rolling regression |
| S3 GICS hybrid | <1 min | <6 min | Lookup-based |
| S4 factor loadings | 5-10 min | 30-60 min | 4 rolling regressions per ticker |
| S5 LMNN | 30-60 min | 3-6 hours | Dominant cost |
| S6 composite | 10-20 min | 1-2 hours | Weight tuning via nested CV |

Worst case (all metrics): ~12 hours compute. Acceptable for overnight run.

### 3.11 R1 Research Dependencies

| Connector Metric | Depends on | Reuse direction |
|------------------|------------|-----------------|
| S2 (beta) | R1-E4 Mahalanobis — shared rolling regression | Same rolling window machinery |
| S4 (factor loadings) | Post-launch factor model research | Opens factor-neutral sizing |
| S5 (LMNN) | R1-E5 Supervised Metric Learning — full code reuse | Entire LMNN path is shared |

### 3.12 Expected Outcomes

**If Track B passes:** Confirms cross-sector dilution as the core problem, not KNN itself.
**If Track B fails despite B.2 > B.1:** Connectors help but the S1 magnitude metric is too coarse. Secondary sweep on S2-S5 warranted.
**If B.1 already passes at a good margin:** Connectors are unnecessary complexity. Ship pure per-sector pools.

---

## §4 TRACK C — LIGHTGBM PREDICTION ENGINE

### 4.1 Prior Research Synthesis

| Document | Key Finding |
|----------|------------|
| `H9_VARIANT_PLAN_KNN_VS_LIGHTGBM.md` | Full LightGBM spec exists. Walk-forward harness, nested CV, Murphy B3, ensemble design |
| KNN vs ML alternatives review | Gu/Kelly/Xiu (2020): GBT >= KNN on cross-sectional equity prediction |
| Feature selection research | LMNN/NCA could improve KNN, but at 585T overfit risk is lower |

### 4.2 Track C Modifications to H9 Plan

**Modification 1:** Run on 585T (not 52T). LightGBM must be tested in the same regime that broke KNN.

**Modification 2:** Use Track A/B winner as KNN baseline. If both fail, use 52T H7 config (mean BSS = +0.00033) as floor.

**Modification 3:** Feature expansion permitted. LightGBM doesn't suffer from distance concentration. Test 23D base vs 50-100D extended (sector + cross-sectional + technical indicators).

### 4.3 Handoff Spec (Track C)

```xml
<goal>
Execute H9 plan (KNN vs LightGBM head-to-head) on 585T universe with modifications:
(1) 585T not 52T, (2) Track A/B winner as KNN baseline if available,
(3) secondary experiment tests LightGBM with expanded feature set (50-100D).
Stacked ensemble runs IF both algorithms pass individually.
</goal>

<success_criteria>
1. LightGBM walk-forward completes on 585T with fold-boundary parity to KNN baseline
2. Both algorithms use identical calibration (Platt) and BSS evaluation
3. Primary gate: mean BSS across 6 folds (algorithm with higher mean wins)
4. Secondary gates (tiebreakers if |delta| < 0.01): pos_folds, reliability, worst-fold BSS, runtime
5. Murphy B3 decomposition produced for both algorithms
6. IF both pass individually: stacked ensemble evaluated vs both
7. Feature expansion experiment (LightGBM only): BSS on 50-100D vs 23D baseline
</success_criteria>

<files_to_modify>
- pattern_engine/lightgbm_matcher.py (NEW) - LightGBMMatcher implementing BaseMatcher ABC
- pattern_engine/contracts/matchers/lightgbm_matcher.py (NEW)
- pattern_engine/feature_expansion.py (NEW) - extended feature set builder
- scripts/run_track_c_knn_vs_lightgbm.py (NEW)
- scripts/run_track_c_lightgbm_feature_expansion.py (NEW)
- scripts/run_track_c_ensemble.py (NEW) - conditional stacking
- tests/unit/test_lightgbm_matcher.py (NEW)
- tests/unit/test_algorithm_parity.py (NEW)
- results/campaign_p8_recovery/track_c/ - outputs
- docs/adr/ADR-016-prediction-engine-selection.md (NEW)
- requirements.txt (MODIFY) - add lightgbm>=4.0
</files_to_modify>

<steps>
1. Add lightgbm to requirements.txt; verify install
2. Implement LightGBMMatcher conforming to BaseMatcher ABC
3. Write algorithm parity test: identical train/test splits, identical Platt calibration
4. Build nested CV driver: 3-fold inner TPESampler, 50 trials, 6-fold outer
5. Run LightGBM on 585T with 23D features (parity baseline)
6. Compute Murphy B3 on LightGBM outputs
7. Run KNN baseline (Track A/B winner OR 52T H7 config)
8. Run ensemble (conditional): stacked LR meta-learner on out-of-fold predictions
9. Run feature expansion (secondary): 50-100D feature set, LightGBM only
10. Write campaign TSV: results/campaign_p8_recovery/track_c/summary.tsv
    (algorithm, feature_set, mean_bss, pos_folds, mean_sharpe, mean_maxdd,
     mean_resolution, mean_reliability, runtime_min, decision)
11. Create ADR-016 with winning algorithm + rationale
</steps>

<verification>
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_lightgbm_matcher.py tests/unit/test_algorithm_parity.py -v
PYTHONUTF8=1 py -3.12 scripts/run_track_c_knn_vs_lightgbm.py
PYTHONUTF8=1 py -3.12 scripts/run_track_c_lightgbm_feature_expansion.py
cat results/campaign_p8_recovery/track_c/gate_check.txt
</verification>

<task_type>SR</task_type>
```

---

## §5 POST-CAMPAIGN DECISION MATRIX

| Track A | Track B | Track C | Recommended Path |
|---------|---------|---------|------------------|
| PASS | - | - | Lock Track A winner, retest P8-PRE-1 with dynamic universe selection |
| FAIL | PASS | - | Lock Track B architecture, retest P8-PRE-1 with sector pools + connectors |
| FAIL | FAIL | LightGBM wins | Pivot prediction engine to LightGBM, keep all other FPPE infrastructure |
| FAIL | FAIL | KNN wins | ESCALATE. FPPE is fundamentally capacity-limited. Reconsider project scope. |
| PASS | PASS | LightGBM wins | Run all three configs in ensemble (A + B + C stacked) |
| PASS | PASS | KNN wins | Choose between A and B by operational simplicity (A simpler, B more capacity) |

### 5.1 Success = Retesting P8-PRE-1

Any track passing its internal gate is necessary but not sufficient. The winning configuration must then re-run the P8-PRE-1 script (or updated equivalent) and pass **all 5 gates**: BSS positive folds >= 3/6, Sharpe >= 1.0, MaxDD <= 15%, Trades >= 50, Win rate >= 50%.

---

## §6 CAMPAIGN GOVERNANCE

### 6.1 Adversarial Self-Review (Mandatory Per Track)

Every track's closing handoff must include, before any PASS claim:
- 3 ways the result could be wrong
- 3 ways the test setup could be biased toward passing
- 1 alternative explanation for any positive result

### 6.2 Campaign Log

Maintain this document with per-track start/end dates, gate results, commits, key decisions, and open questions.

### 6.3 Feature Flag Discipline

Every architectural change behind a feature flag:
- Track A: `use_dynamic_universe_selection` (default False)
- Track B: `sector_pool_mode` (default None, options: "pure", "with_connectors")
- Track C: `prediction_engine` (default "knn", options: "knn", "lightgbm", "ensemble")

No flag defaults change without the full P8-PRE-1 retest passing.

### 6.4 Test Count Discipline

Current baseline: 945 tests. Track A adds ~15-25. Track B adds ~25-40. Track C adds ~20-35.

### 6.5 3-Strike Rule Per Track

Each track has its own 3-strike budget. Track failure does not exhaust the campaign.

---

## §7 TRACK STATUS LOG

| Track | Status | Started | Completed | Gate Result | Commit |
|-------|--------|---------|-----------|-------------|--------|
| Track A | PENDING | — | — | — | — |
| Track B | PENDING | — | — | — | — |
| Track C | PENDING | — | — | — | — |

---

## §8 EXECUTION ORDER (Final)

1. ~~Close out P8-PRE-1 per HANDOFF_P8-PRE-1_EXECUTE-AND-CLOSE.md~~ ✓ DONE (2026-04-16)
2. ~~Create this campaign document~~ ✓ DONE (2026-04-16)
3. **Track A** — universe sweep with equal weights; deferred weight sweep on success
4. **Track B** — sector pools B.1-B.4; connector sweep S1-S6 if B.2 > B.1
5. **Track C** — LightGBM vs KNN (KNN baseline = best of A/B; or 52T H7 config if both failed)
6. Post-campaign: re-run P8-PRE-1 script with winning config, verify all 5 gates pass

---

## §9 REFERENCES (Project Knowledge)

- `H9_VARIANT_PLAN_KNN_VS_LIGHTGBM.md` — complete LightGBM implementation spec
- `KNN_Analogue_Matching_Versus_Modern_Alternatives_for_Equity_Return_Prediction__A_Critical_Evidence_Review.md` — KNN vs GBT evidence
- `Feature_Selection_and_Metric_Learning_for_KNN_with_hnswlib_at_Scale.md` — LMNN/NCA research
- `FPPE_MASTER_PLAN_v3.md` §6 R1 experiment queue, §7 R2-H9
- `SESSION_2026-04-02_phase1-bss-experiments.md` — 585T failure mode confirmation
- `SESSION_2026-04-05_phase1-h5-h6-handoff.md` — Murphy decomposition per-universe
- `docs/adr/ADR-013-calibration-method-production.md` — Platt vs beta_abm split decision
- `results/phase8_pre/585t_gate_check.txt` — P8-PRE-1 gate results (provenance)
- `results/phase8_pre/585t_walkforward.tsv` — P8-PRE-1 per-fold data (provenance)

---

*Campaign document prepared by Claude Opus (advisor) — 2026-04-16*  
*Addendum integrated by Claude Code (Sonnet) — 2026-04-16*  
*For execution by Claude Code (Sonnet). Task type: SR throughout.*
