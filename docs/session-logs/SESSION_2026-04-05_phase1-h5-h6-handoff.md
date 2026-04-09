# Session Log — 2026-04-05
**AI:** Claude Sonnet 4.6
**Duration:** ~2.5 hours
**Phase:** Phase 1 — BSS Gate Experiments (H5, H6, Murphy @ 52T)
**Status:** COMPLETE — Gate NOT Met. Handing off to H7 (regime filter).

---

## Session Summary

Continued from SESSION_2026-04-02_phase1-52t-validation.md. Three experiments ran:
- **H5**: max_distance recalibration sweep at 52T VOL_NORM (11 values, 0.70–1.50)
- **Murphy decomposition**: on 52T VOL_NORM best config (max_d=0.90) — key diagnostic
- **H6**: cross-sectional feature expansion (6 feature sets, 52t_features dataset)

---

## Files Created / Modified

| File | Type | Description |
|------|------|-------------|
| `scripts/experiments/h5_52t_maxd_volnorm_sweep.py` | New | 11-value max_distance sweep at 52T VOL_NORM, beta_abm |
| `scripts/build_52t_volnorm.py` | New | Builds 52T VOL_NORM dataset from 585T parquet + 4 CSV caches |
| `scripts/build_52t_features.py` | New | Adds sector_relative_return_7d, sector_rank_30d, spy_correlation_30d |
| `scripts/experiments/h6_feature_expansion.py` | New | 6 feature-set sweep at 52T, max_d=0.90, beta_abm |
| `scripts/experiments/validate_52t_best_config.py` | Modified | Updated to use 52t_volnorm + VOL_NORM_COLS |
| `data/52t_volnorm/train_db.parquet` | New | 52T + VOL_NORM_COLS, 175,605 rows, 2010–2023 |
| `data/52t_volnorm/val_db.parquet` | New | 52T + VOL_NORM_COLS, 13,104 rows, 2024 |
| `data/52t_features/train_db.parquet` | New | 52T + VOL_NORM + 3 cross-sectional features |
| `data/52t_features/val_db.parquet` | New | As above, val split |
| `results/bss_fix_sweep_h5.tsv` | New | H5 results: 11 max_d values |
| `results/bss_fix_sweep_h6.tsv` | New | H6 results: 6 feature sets |
| `results/validate_52t_best_config.tsv` | Updated | VOL_NORM results |
| `docs/session-logs/SESSION_2026-04-02_phase1-52t-validation.md` | Updated | H5/H6/Murphy results appended |

---

## Experiment Results

### H5 — max_distance Sweep at 52T VOL_NORM

**11 configs**: max_d=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.1019, 1.20, 1.50]
**Fixed**: beta_abm, uniform, 52t_volnorm dataset

| max_d | AvgK | mean_BSS | pos/6 |
|-------|------|----------|-------|
| **0.90** | **35.9** | **-0.00651** | **2/6** ← best |
| 0.95 | 39.0 | -0.00654 | 1/6 |
| 1.50 | 49.1 | -0.00649 | 1/6 |
| others | … | -0.00664 to -0.00688 | 0-1/6 |

Gate NOT met. Total BSS spread across all 11 values: 0.00039 (noise-level). Parametric tuning exhausted.

Positive folds: **2020-COVID** and sometimes **2021** only.

### Murphy Decomposition — 52T VOL_NORM, max_d=0.90

```
Resolution  = 0.007621   ← 10× higher than 585T (0.000709). SIGNAL EXISTS at 52T.
Reliability = 0.009544   ← dominant. Calibration is the bottleneck.
Verdict:     RELIABILITY-DOMINATED (not resolution-dominated like 585T)
```

Per-fold gaps (Reliability − Resolution):
- 2019: +0.002942
- 2020-COVID: **-0.000112** (negative = positive BSS, model slightly better than calibration)
- 2021: +0.000109
- **2022-Bear: +0.007198** ← structural outlier, drives all negative BSS
- 2023: +0.001046
- 2024-Val: +0.000352

**Key**: 2023 and 2024 have gap < 0.001, nearly positive. Only 2022 Bear is a serious problem.

### H6 — Cross-Sectional Feature Expansion

**6 feature sets**, fixed: max_d=0.90, beta_abm, 52t_features dataset

| Feature Set | nD | mean_BSS | pos/6 | Resolution | Reliability |
|------------|-----|----------|-------|------------|-------------|
| **8D VOL_NORM baseline** | 8 | **-0.00676** | **2/6** | 0.00756 | 0.00922 |
| +sector_rel_7d | 9 | -0.00686 | 1/6 | 0.00756 | 0.00925 |
| +sector_rank_30d | 9 | -0.00692 | 1/6 | 0.00763 | 0.00934 |
| +spy_corr_30d | 9 | -0.00724 | 0/6 | 0.00776 | 0.00954 |
| +rel_7d+rank_30d | 10 | -0.00696 | 0/6 | 0.00617 | 0.00789 |
| +all 3 | 11 | -0.00711 | 0/6 | 0.00183 | 0.00358 |

Gate NOT met. 8D baseline is still best (2/6).

**Why feature expansion failed**: adding dimensions changes the Euclidean distance metric — max_d=0.90 was calibrated for 8D. Each extra dimension cuts AvgK drastically (9D→AvgK≈21, 11D→AvgK≈1.5). The Reliability gap is unchanged across all feature sets — confirmed as a calibration problem, not a signal problem.

---

## Complete Experiment History (Phase 1)

| Experiment | Universe | Features | Config | Best mean_BSS | pos/6 |
|-----------|----------|----------|--------|---------------|-------|
| H1 | 585T | VOL_NORM | max_d=0.5, uniform | -0.00419 | 0/6 |
| H2 | 585T | VOL_NORM | sector filter | -0.00438 | 0/6 |
| H3 | 585T | VOL_NORM | top_k sweep | -0.00435 | 0/6 |
| H4 | 585T | VOL_NORM | beta cal | -0.00401 | 0/6 |
| 52T R1 | 52T | RETURNS_ONLY | max_d=1.1019, platt | -0.00683 | 0/6 |
| 52T R2 | 52T | VOL_NORM | max_d=1.1019, beta | -0.00656 | 1/6 |
| **H5** | **52T** | **VOL_NORM** | **max_d=0.90, beta** | **-0.00651** | **2/6** |
| H6 | 52T | VOL_NORM+3 | max_d=0.90, beta | -0.00676 | 2/6 |

**Best ever: 2/6 positive folds. Gate requires 3/6.**

---

## Diagnostic State

**Confirmed findings (all have experimental evidence):**
1. 585T fails due to Resolution≈0 (analogue pool dilution). Parametric fixes cannot help.
2. 52T has Resolution=0.0076 — genuine signal exists. Problem is Reliability (miscalibration).
3. Beta calibration consistently improves BSS by ~+0.001–0.002 vs Platt. Real and reproducible.
4. max_distance=0.90 is empirically best for 8D VOL_NORM at 52T (vs locked 1.1019).
5. 2022 Bear fold is the structural outlier: Reliability=0.015 (gap=0.007). All other folds near-balanced.
6. Adding cross-sectional features doesn't help at current max_d — dimensionality changes distance structure.

---

## Recommended Next Experiment: H7 — Regime Filter

**Hypothesis**: The 2022 Bear Reliability gap (0.007) is driven by systematic probability bias in bear-market regimes. The model has genuine signal in 2022 (Resolution=0.008) but miscalibrated probabilities. Routing bear-regime queries to base-rate (HOLD) would:
- Replace bad-Reliability predictions with Reliability=0 (base-rate = perfect calibration)
- Expected 2022 fold: BSS → ≈0 (from -0.029)
- Expected total: 3–4/6 positive folds (2020-COVID, 2021, and likely 2023/2024 become positive)

**The math**: BSS(fold) = (Resolution − Reliability) / Uncertainty. For 2022: (0.008 − 0.015)/0.25 = -0.028. If bear queries are HOLDed: Resolution→0, Reliability→0, BSS→0. Net effect for that fold: 0 vs -0.029.

**Existing infrastructure**:
- `pattern_engine/regime.py` — `RegimeLabeler` (SPY drawdown + VIX + yield curve)
- `pattern_engine/config.py` — `regime_filter: bool = False` (locked off)
- `pattern_engine/matcher.py` — regime filter logic already wired in (just disabled)

**H7 experiment design**:
- Test `regime_filter=True` at 52T VOL_NORM, max_d=0.90, beta_abm
- Sweep regime thresholds (SPY drawdown threshold for "bear" classification)
- Track: BSS per fold, n_filtered per fold (how many queries routed to HOLD)
- Gate: BSS > 0 on ≥ 3/6 folds (same as all Phase 1)
- Data: `data/52t_volnorm/` (8D VOL_NORM, 52T)

**Key caution**: BSS is computed on ALL val rows. HOLD rows are assigned base_rate (0.5), contributing 0 to Brier Score improvement. If regime filter HOLDs too many non-bear days, it will degrade 2020/2021 positive folds. Need to verify that 2022 Bear days are well-separated by the RegimeLabeler.

---

## Locked Settings (UNCHANGED — gate not met)

```
max_distance=1.1019, top_k=50, distance_weighting=uniform,
calibration=Platt, features=VOL_NORM_COLS(8), cal_frac=0.76
```

No settings locked from this session. All experiments were exploratory with experimental configs.

---

## Test Suite Status

Not re-run this session (no production code modified).
Prior status: 616 passing, 1 skipped.

---

## Data Assets Created (new, persistent)

| Path | Description |
|------|-------------|
| `data/52t_volnorm/` | 52T + VOL_NORM_COLS. Canonical 52T M9 dataset. |
| `data/52t_features/` | 52T + VOL_NORM + 3 cross-sectional features. |

Both datasets are reproducible from scripts if deleted.
