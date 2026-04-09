# Session Log — 2026-04-02 Phase 1 Escalation Path 1 (52T Validation)
**Date:** 2026-04-02 (continuation of morning BSS experiments session)
**Phase:** Phase 1 — Escalation Path 1: 52T Universe Validation
**Status:** COMPLETE — Gate NOT Met on RETURNS_ONLY or VOL_NORM at 52T. Third Escalation.

---

## Context

Morning session (SESSION_2026-04-02_phase1-bss-experiments.md) concluded:
- All 25 configs on 585T universe failed the BSS gate (BSS > 0 on ≥ 3/6 folds)
- Root cause: analogue pool dilution at 585T. Resolution ≈ 0 — model is a constant-function approximator.
- Escalation Path 1 recommended: validate best Phase 1 config at 52T universe.
- Premise: "52T baseline had Fold6 BSS=+0.00103. Signal exists at 52T scale."

---

## Work Completed

| File | Description |
|------|-------------|
| `scripts/experiments/validate_52t_best_config.py` | New: 4-config sweep at 52T universe. Beta cal + max_d variations. Incremental TSV + resume logic. Updated to use VOL_NORM_COLS + 52t_volnorm dataset. |
| `scripts/build_52t_volnorm.py` | New: Builds 52T dataset with VOL_NORM_COLS (48T filtered from 585T parquet + 4 missing rebuilt from CSV) |
| `data/52t_volnorm/train_db.parquet` | New: 52T training data with VOL_NORM_COLS (175,605 rows, 52 tickers, 2010–2023) |
| `data/52t_volnorm/val_db.parquet` | New: 52T validation data with VOL_NORM_COLS (13,104 rows, 52 tickers, 2024) |
| `results/validate_52t_best_config.tsv` | Updated: VOL_NORM results for all 4 configs |

---

## Experiment Results — 52T Validation

**Dataset:** `data/processed/train_db.parquet + val_db.parquet`
**Tickers:** 52 | **Rows:** 188,709 | **Features:** RETURNS_ONLY_COLS (8 raw return cols)
**Gate:** BSS > 0 on ≥ 3/6 folds

| Config | Cal | max_d | mean_BSS | pos_folds | Gate |
|--------|-----|-------|----------|-----------|------|
| beta_maxd05_52t | beta_abm | 0.5 | -0.00683 | 0/6 | NO |
| platt_maxd05_52t | platt | 0.5 | -0.00689 | 1/6 | NO |
| beta_baseline_52t | beta_abm | 1.1019 | -0.00732 | 1/6 | NO |
| baseline_52t | platt | 1.1019 | -0.00897 | 0/6 | NO |

### Per-Fold Detail (baseline_52t — sanity check)

| Fold | BSS | AvgK | n_val |
|------|-----|------|-------|
| 2019 | -0.014588 | 42.2 | 13,104 |
| 2020-COVID | -0.004576 | 27.1 | 13,156 |
| 2021 | -0.002255 | 42.8 | 13,104 |
| 2022-Bear | -0.031196 | 32.8 | 13,052 |
| 2023 | -0.001085 | 42.0 | 13,000 |
| 2024-Val | **-0.000099** | 42.3 | 13,104 |

---

## Round 2 — VOL_NORM_COLS at 52T (Option A)

**Dataset:** `data/52t_volnorm/` (built by scripts/build_52t_volnorm.py)
**Tickers:** 52 | **Rows:** 188,709 | **Features:** VOL_NORM_COLS (8 vol-normalized cols)
**Method:** 48 tickers filtered from 585T parquet + ABBV/META/PYPL/TSLA rebuilt from CSV caches

| Config | Cal | max_d | mean_BSS | pos_folds | Gate |
|--------|-----|-------|----------|-----------|------|
| beta_baseline_52t | beta_abm | 1.1019 | -0.00656 | 1/6 | NO |
| beta_maxd05_52t | beta_abm | 0.5 | -0.00667 | 1/6 | NO |
| platt_maxd05_52t | platt | 0.5 | -0.00675 | 1/6 | NO |
| baseline_52t | platt | 1.1019 | -0.00808 | 1/6 | NO |

**Gate still not met.** Only Fold2 (COVID 2020) is positive across all configs.

### Key VOL_NORM Findings

1. **max_distance=0.5 is broken with VOL_NORM_COLS**: AvgK drops to 1.2–1.8 (vs 12–14 with
   RETURNS_ONLY). The max_distance threshold was calibrated for raw return space and is not
   portable to normalized-feature space. With AvgK≈1, probability estimates are meaningless.

2. **The positive fold shifted**: RETURNS_ONLY had Fold6 (2024) positive; VOL_NORM has
   Fold2 (COVID) positive. Different feature spaces surface different signal pockets.

3. **VOL_NORM modestly outperforms RETURNS_ONLY** at baseline:
   mean_BSS -0.00808 vs -0.00897 (+0.00089). Real but insufficient for gate.

4. **Beta calibration still helps**: best config is beta+1.1019 (mean_BSS=-0.00656).
   Consistent with 585T finding.

---

## CRITICAL FINDING: Baseline Non-Reproduction

**Expected** (from `run_walkforward.py` docstring): Fold6 BSS = **+0.00103**
**Actual** (this session): Fold6 BSS = **-0.000099**

The sign has flipped and the magnitude is 10× smaller. This is not a numerical precision issue.

### Probable Explanation

When the baseline (+0.00103) was established, `data/train_db.parquet` was the 52T file
(pre-expansion). After the 585T expansion, `data/train_db.parquet` was replaced with
585T data. The 52T data was preserved in `data/processed/` but may have been regenerated
or updated since the baseline was measured.

Candidate causes:
1. **Dataset regeneration**: `data/processed/train_db.parquet` was rebuilt after baseline,
   changing feature values (different rolling window edge behaviour, corporate actions, etc.)
2. **Feature engineering drift**: Some feature column computation was updated between
   baseline measurement and now.
3. **HNSW vs ball_tree**: Baseline may have been established with `use_hnsw=False`.
   HNSW recall@50=0.9996 — unlikely to cause this discrepancy, but possible near zero-BSS.
4. **Note**: `data/processed/full_db.parquet` has 202,645 rows (to 2026-01-28) vs
   train+val = 188,709 rows. If baseline used full_db (no train/test split files at the time),
   the training window for Fold6 might include more data.

### Impact

The premise of Escalation Path 1 was "52T has signal (+0.00103 in Fold6)." This is now
**unconfirmed** — the signal was real at some point but cannot be reproduced with the
current 52T dataset.

---

## Key Observations Across All Configs

1. **Beta calibration still helps** (mean_BSS improves by ~+0.00165 vs platt baseline at 52T)
   — consistent with 585T finding. Real and reproducible.
2. **max_distance=0.5 helps** at 52T (mean_BSS -0.00689 vs -0.00897 for platt/1.1019)
   — AvgK drops to 12–14 (from 42), which means only very close neighbours are used.
3. **Beta + max_d=0.5 combined** is marginally worse than platt + max_d=0.5
   — at low AvgK (5–14), beta cal may overfit the small calibration set. Less data for cal fitting.
4. **Gate NOT met by any config on 52T.** The "safe harbour" at 52T does not exist with current data.

---

## Data Landscape (documented for reference)

| Path | Rows | Tickers | Feature Cols | Date Range |
|------|------|---------|--------------|------------|
| `data/train_db.parquet` | 3,208,668 | 585 | VOL_NORM_COLS (M9) | 2010–2023 |
| `data/val_db.parquet` | ~500K+ | 585 | VOL_NORM_COLS (M9) | 2024 |
| `data/full_analogue_db.parquet` | 3,516,089 | 585 | RETURNS_ONLY_COLS | 2000–2026 |
| `data/processed/train_db.parquet` | 175,605 | 52 | RETURNS_ONLY_COLS | 2010–2023 |
| `data/processed/val_db.parquet` | 13,104 | 52 | RETURNS_ONLY_COLS | 2024 |
| `data/processed/full_db.parquet` | 202,645 | 52 | RETURNS_ONLY_COLS | 2010–2026 |

**Gap closed (Option A)**: `data/52t_volnorm/` now exists with VOL_NORM_COLS (built 2026-04-02).
48T filtered from 585T parquet + ABBV/META/PYPL/TSLA rebuilt from CSV caches via `build_52t_volnorm.py`.

---

## Recommended Next Steps (updated after Option A completion)

### H5: max_distance Sweep Results (Option B — COMPLETE)

**Script:** `scripts/experiments/h5_52t_maxd_volnorm_sweep.py`
**TSV:** `results/bss_fix_sweep_h5.tsv`
**11 configs:** max_d=[0.70…1.50], beta_abm, uniform, VOL_NORM, 52T

| max_d | AvgK | mean_BSS | pos/6 | Gate |
|-------|------|----------|-------|------|
| 1.50 | 49.1 | -0.00649 | 1/6 | no |
| **0.90** | **35.9** | **-0.00651** | **2/6** | no |
| 0.95 | 39.0 | -0.00654 | 1/6 | no |
| 1.1019 | 44.9 | -0.00656 | 1/6 | no |
| … | … | … | 0-1/6 | no |

Best result: **max_d=0.90, 2/6 positive** (Fold2 COVID +0.00022, Fold3 2021 +0.00003).
Gate NOT met. **Total BSS spread across all 11 values: only 0.00039** — noise-level variation.

#### Fold-by-Fold Pattern (all 11 configs)

| Fold | Range | Always positive? |
|------|-------|-----------------|
| 2019 | -0.010 to -0.013 | Never |
| 2020-COVID | -0.00037 to +0.00038 | **Sometimes** (at max_d 0.85–1.5) |
| 2021 | -0.00050 to +0.00003 | Once (max_d=0.90 only) |
| 2022-Bear | -0.027 to -0.029 | Never (catastrophic drag, ~-0.028) |
| 2023 | -0.00017 to -0.00123 | Never |
| 2024-Val | -0.00003 to -0.00088 | Never |

**Dominant pattern**: The 2022 Bear fold alone accounts for most of the negative mean BSS.
The model has genuine signal in high-volatility regimes (COVID 2020) but fails systematically
in bear markets and low-volatility environments.

#### Option B Conclusion

max_distance recalibration is exhausted. Parametric tuning cannot overcome:
- BSS landscape is nearly flat across all 11 values (spread = 0.00039)
- 2022 Bear always -0.028 (structural failure in bear regimes)
- Normal markets (2019, 2023, 2024) always near-zero or negative

---

### Summary of All Experiments (as of 2026-04-02 EOD)

| Attempt | Universe | Features | maxD | Best mean_BSS | pos_folds |
|---------|----------|----------|------|---------------|-----------|
| Phase 1 H1–H4 | 585T | VOL_NORM | 0.5 | -0.00401 | 0/6 |
| 52T Round 1 | 52T | RETURNS_ONLY | 0.5 | -0.00683 | 0/6 |
| 52T Round 2 (VOL_NORM) | 52T | VOL_NORM | 1.1019 | -0.00656 | 1/6 |
| **H5 sweep (Option B)** | 52T | VOL_NORM | **0.9** | **-0.00651** | **2/6** |

Best ever: 2/6 positive folds. Gate requires 3/6.

---

### Remaining Options

#### Option C — Feature expansion (COMPLETE — did not meet gate)

**Scripts:** `scripts/build_52t_features.py`, `scripts/experiments/h6_feature_expansion.py`
**TSV:** `results/bss_fix_sweep_h6.tsv`

**Murphy decomposition (52T VOL_NORM, max_d=0.90) BEFORE H6:**
```
Resolution  = 0.007621  ← 10× higher than 585T (0.000709). Signal EXISTS at 52T.
Reliability = 0.009544  ← dominant (>0.002). Calibration is the bottleneck.
Gap needed: reduce Reliability by ~0.002 to flip BSS positive.
Per-fold: 2022 Bear has Reliability=0.015 vs Resolution=0.008 (gap=0.007 — structural outlier).
```

**H6 results (6 feature sets, fixed max_d=0.90, beta_abm):**

| Feature Set | nD | mean_BSS | pos/6 | Resolution | Reliability |
|------------|-----|----------|-------|------------|-------------|
| base_8d (VOL_NORM) | 8 | -0.00676 | 2/6 | 0.00756 | 0.00922 |
| +sector_rel_7d | 9 | -0.00686 | 1/6 | 0.00756 | 0.00925 |
| +sector_rank_30d | 9 | -0.00692 | 1/6 | 0.00763 | 0.00934 |
| +sector_rel+rank | 10 | -0.00696 | 0/6 | 0.00617 | 0.00789 |
| +all 3 | 11 | -0.00711 | 0/6 | 0.00183 | 0.00358 |
| +spy_corr | 9 | -0.00724 | 0/6 | 0.00776 | 0.00954 |

**Gate NOT met. 8D baseline is still best (2/6 positive folds).**

Key findings:
1. Adding features HURTS at max_d=0.90 — each dimension added reduces AvgK drastically
   (9D → AvgK≈21, 10D → AvgK≈8, 11D → AvgK≈1.5). max_d must be recalibrated per dimensionality.
2. The Reliability gap (Rel-Res ≈ 0.002) is unchanged across feature sets.
   Feature expansion cannot fix a calibration problem.
3. **2023 and 2024 folds are within 0.0003 of positive** (gap < 0.0003). The only
   structural failure is 2022 Bear (gap = 0.007). If 2022 Bear predictions were replaced
   with base-rate (HOLD), we'd likely see 4/6 positive folds.

#### Option C — Feature expansion [REVISED — exhausted at current max_d]

H6 showed feature expansion doesn't help at max_d=0.90. A follow-up could recalibrate
max_d per dimensionality (9D needs max_d≈0.95, 10D≈1.00, 11D≈1.05 per sqrt(d) scaling).
But the Reliability gap is a calibration problem, not a signal problem — more features
are unlikely to close a 0.007 Reliability excess in the 2022 Bear fold.

#### Option D — Regime-conditioned matching (RECOMMENDED NEXT)

**Why this is now the highest-priority path:**

The Murphy decomposition confirmed Resolution=0.0076 (signal exists) but Reliability=0.0095
is the binding constraint. The 2022 Bear fold has Reliability=0.015 — nearly 2× its Resolution.
This means: the model has genuine predictive signal in 2022 but makes systematically
biased probability predictions in that regime.

In a bear market, ALL tickers share similar negative return trajectories. The KNN model
finds "close analogues" (high similarity) but those analogues' forward returns are poorly
predictive because the post-analogue outcomes depend on regime continuation, not pattern
matching. The model's probabilities are systematically too high (or too low) across the board.

**Mechanism**: When `regime_filter=True`, queries in unrecognised regimes are assigned the
base rate (0.5) instead of model predictions. This means Reliability=0 for those queries
(predicting base rate is perfectly calibrated). If the bear-market queries are the ones
driving high Reliability, routing them to HOLD would close the gap.

**The BSS math**: 2023 and 2024 folds have Rel-Res gap < 0.0003 — almost positive now.
2022 Bear gap = 0.007. If those queries are filtered to base-rate, 2022 BSS → near 0
(instead of -0.029). We'd expect 4/6 positive folds with effective regime filtering.

**Files to examine:**
- `pattern_engine/regime.py` — RegimeLabeler (SPY + VIX + yield curve)
- `pattern_engine/config.py` — `regime_filter=False` (off in production)
- H5 best config: max_d=0.90, beta_abm, uniform at 52T VOL_NORM

**Experiment design (H7):**
- Test regime_filter=True at 52T VOL_NORM, max_d=0.90, beta_abm
- Compare BSS with and without regime filter per fold
- Track n_filtered (how many val queries got routed to base-rate)
- If gate met: validate regime filter thresholds before locking

#### Recommended Action

Proceed with Option D (H7: regime filter experiment).
The diagnostic evidence is now clear enough to justify this without further preliminary work.

---

## Test Suite Status

Not re-run this sub-session (no production code was modified).
Prior status: 616 passing, 1 skipped.

---

## Locked Settings

No changes. Gate not met — no settings are updated without gate evidence.
