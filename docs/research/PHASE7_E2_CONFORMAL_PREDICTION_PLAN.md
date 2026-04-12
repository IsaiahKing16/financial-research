# Phase 7 E2: Conformal Prediction Dual-Gating Architecture

**Created:** 2026-04-10  
**Status:** BLOCKED — Phase 7 complete (2026-04-10). E4 tested plain ACI and FAILED due to 52T probability constraint. DtACI (proposed here) faces the same structural limitation at 52T scale. Viable only if probability range widens (585T signals or algorithm change).  
**Prerequisite:** Phase 1 BSS gate passed + Phase 7 E1 (Venn-ABERS) evaluated. **Update:** Phase 7 E1 tested BMA (FAIL), not Venn-ABERS. Venn-ABERS remains untested.  
**Source:** "Conformal Prediction Meets Financial Forecasting: A Practitioner's Guide to Dual-Gating with BSS and Adaptive Coverage Methods" research paper synthesis  

---

## Context

FPPE currently uses a two-component decision system:
1. **Model-level gate:** BSS > 0 on ≥3 of 6 walk-forward folds (pass/fail for entire model)
2. **Instance-level filter:** Confidence threshold (operating at 0.55, locked target 0.65 — provenance gap unresolved)

The confidence threshold is an ad-hoc heuristic with no statistical foundation. Conformal prediction replaces it with a formally guaranteed instance-level filter that produces prediction sets with coverage guarantees. The singleton filter ("trade only when C(x) = {UP}") provides a mathematically grounded abstention mechanism.

The research paper establishes that BSS and conformal coverage measure orthogonal axes of prediction quality — BSS captures probability accuracy and discrimination, conformal prediction captures per-instance uncertainty. Neither subsumes the other. Deploying both together creates a provably sound two-stage filter superior to either alone.

**Phase 7 E4 Result (2026-04-09):** Plain ACI was tested on 52T walk-forward and FAILED. Coverage=0.814 mean, width=1.000 (trivial [0,1] intervals). Root cause: 52T beta_abm probabilities cluster in [0.50-0.59]; |prob-label| nonconformity scores are always >=0.41; the quantile threshold settles at ~0.57, producing near-trivial prediction sets. The DtACI method proposed in this document is theoretically superior (eliminates gamma tuning) but faces the same structural constraint -- the probability compression at 52T scale makes any conformal approach produce trivially wide intervals. This plan becomes viable only when operating on 585T production signals (probs in [0.65-0.75]) or after an algorithm change that widens the probability range.

---

## Dual-Gating Architecture

### Level 1 — Model Quality Gate (BSS) [Existing]

In each walk-forward window, compute BSS on the validation fold. BSS > 0 confirms the model is skillful — resolution exceeds calibration error. If BSS ≤ 0, shut off trading for this window entirely.

**This already exists in FPPE. No changes needed.**

### Level 2 — Instance Quality Gate (Conformal) [New — This Document]

For each prediction where Level 1 has passed:
1. Compute nonconformity score for the test instance
2. Generate conformal prediction set at significance level α (default: 0.10 for 90% coverage)
3. Decision logic:
   - `C(x) = {UP}` → Execute trade (singleton = high confidence)
   - `C(x) = {UP, HOLD}` or `{UP, DOWN}` → Abstain (doubleton = moderate uncertainty)
   - `C(x) = {UP, DOWN, HOLD}` → Abstain (full set = model cannot discriminate)

### What This Replaces

The confidence threshold parameter (0.55/0.65) becomes **obsolete**. Instead of "trade when P(UP) > threshold," the system trades when "C(x) = {UP}" at significance level α. This resolves the provenance gap between the operating threshold (0.55) and the locked target (0.65) — neither value is needed anymore.

The α parameter (significance level) replaces the confidence threshold. Unlike the ad-hoc threshold:
- α has formal statistical meaning (target miscoverage rate)
- Coverage guarantees hold in finite samples under exchangeability
- Adaptive methods (DtACI) maintain guarantees even under distribution shift

---

## Technical Design

### Nonconformity Score

KNN was the original algorithm in Vovk's foundational conformal prediction work. Natural nonconformity measures exist:

**Primary (recommended):** Distance-ratio score  
`α_i = (sum of k nearest same-class distances) / (sum of k nearest different-class distances)`  
High ratios = example sits far from its own class relative to others.

**Alternative:** KNN fraction  
`α_i = 1 − (count of k neighbors sharing predicted class) / k`  
Simpler, but less discriminating.

**Recommendation:** Start with distance-ratio. FPPE's hnswlib index already computes neighbor distances — the nonconformity score requires no additional KNN queries.

### Adaptive Method: DtACI (Not Plain ACI)

Standard ACI (Gibbs & Candès, NeurIPS 2021) requires choosing γ, the learning rate controlling reactivity vs stability. Wrong γ → either too-slow adaptation or noisy oscillation.

DtACI (Gibbs & Candès, JMLR 2024) eliminates this choice by running multiple ACI copies with different γ values and aggregating via exponential reweighting. It achieves strongly adaptive regret bounds across all local time intervals simultaneously.

**Decision: DtACI is LOCKED as the adaptive method. Do not use plain ACI.**

### Mondrian Categorization

Standard conformal prediction guarantees only marginal coverage — averaged across all conditions. Mondrian CP stratifies by category and guarantees coverage within each stratum independently.

**Categorization variable:** Regime label from HMM filtered probabilities (Paper #3 / H8 connection).  
- If H8 has run: Use discretized P(bull) bins (e.g., [0, 0.3) = bear, [0.3, 0.7) = transition, [0.7, 1.0] = bull)
- If H7 threshold labeler is active: Use binary bull/bear label

**Critical constraint:** Each Mondrian category needs ≥ ⌈1/α⌉ − 1 calibration samples (19 for α = 0.05, 9 for α = 0.10). Bear-regime categories with fewer samples degenerate into trivial "include all labels" prediction sets — functionally equivalent to abstaining on all bear-regime predictions. Monitor per-category sample counts and log warnings when categories are undersized.

### Calibration Pipeline Integration

The recommended full stack combining Papers #1 and #2:

```
hnswlib KNN → Venn-ABERS calibration (E1) → Mondrian conformal prediction sets (E2)
                    ↓                                    ↓
            P(UP) for Kelly sizing              {UP}/{UP,HOLD}/{UP,DOWN,HOLD} for trade/abstain
```

Venn-ABERS produces calibrated P(UP) that feeds both:
1. Kelly position sizing (how much to bet)
2. Conformal nonconformity scores (whether to bet at all)

Better calibration → better nonconformity scores → higher singleton rates → more actionable trades. This is why E1 (Venn-ABERS) must precede E2 (conformal).

**Note:** This pipeline assumes E1 = Venn-ABERS. Phase 7 E1 was BMA (FAIL). If this plan is revisited, Venn-ABERS must be tested first, or beta_abm must remain as the calibrator input.

### Alternative: SelfCalibratingConformal (NeurIPS 2024)

Combines Venn-ABERS calibration with conformal prediction in a single framework, producing both calibrated probabilities and valid prediction sets simultaneously. Evaluate as an alternative to the two-step pipeline. If it produces equivalent or better results with simpler implementation, it replaces the E1→E2 two-step approach.

---

## Library Decisions

| Library | Role | Status | Rationale |
|---------|------|--------|-----------|
| **MAPIE** (v1.3.0, ~1,400 stars) | Primary conformal prediction | Recommended | sklearn-compatible, LAC/APS/RAPS/Mondrian, backed by Capgemini + ENS Paris-Saclay |
| **crepes** (v0.9.0, ~540 stars) | Fallback / Mondrian specialist | Fallback | First-class Mondrian support via `MondrianCategorizer`, exchangeability martingales for diagnostics |
| **puncc** (v0.8.0, ~352 stars) | Time series specialist | Monitor | EnbPI/aEnbPI for non-exchangeable data, but classification support less mature |
| **TorchCP** (v1.0.3, ~454 stars) | Not applicable | Skip | PyTorch-native, overkill for sklearn-based FPPE |
| **nonconformist** (v2.1.0) | Deprecated | Skip | Abandoned, superseded by MAPIE |

**Decision: MAPIE primary, crepes fallback. Not locked until Phase 7 starts — library landscape may shift.**

---

## Implementation Spec (Claude Code Handoff)

### Goal

Add conformal prediction as an instance-level quality gate (Level 2) to FPPE's existing BSS model-level gate (Level 1). Replace the ad-hoc confidence threshold with a formally guaranteed singleton filter using DtACI-adapted Mondrian conformal prediction sets.

### Success Criteria

1. Conformal prediction sets generated for every prediction where BSS > 0
2. Singleton rate measured per fold and logged as primary trade-frequency metric
3. DtACI maintains empirical coverage ≥ (1−α) × 0.98 across all folds (allowing 2% degradation tolerance)
4. Mondrian categories receive adequate calibration samples (≥ ⌈1/α⌉ − 1 per category)
5. BSS computed on singleton-filtered predictions is ≥ BSS on unfiltered predictions (filtering should improve, not degrade, quality)
6. Confidence threshold parameter removed from production config

### Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `pattern_engine/conformal.py` | Create | Conformal prediction module: nonconformity scores, DtACI, Mondrian categorization |
| `pattern_engine/config.py` | Modify | Add conformal config (alpha, method, mondrian_categories); deprecate confidence_threshold |
| `scripts/run_walkforward.py` | Modify | Wire conformal prediction sets into post-calibration pipeline |
| `tests/test_conformal.py` | Create | Coverage tests, singleton rate tests, Mondrian category size checks |

### Step-by-Step Implementation Plan

#### Step 1: KNN Distance-Ratio Nonconformity Score

```
class KNNNonconformityScorer:
    """Computes nonconformity scores from KNN neighbor distances."""
    
    score(distances: np.ndarray, neighbor_labels: np.ndarray, predicted_label: int) -> float
        - same_class_dist = sum of distances to neighbors with label == predicted_label
        - diff_class_dist = sum of distances to neighbors with label != predicted_label
        - Returns same_class_dist / (diff_class_dist + epsilon)
        - epsilon = 1e-10 to avoid division by zero
```

No additional KNN queries required — reuse distances already computed by hnswlib during prediction.

#### Step 2: DtACI Adaptive Calibration

```
class DtACICalibrator:
    """Discounted Adaptive Conformal Inference — no γ tuning required."""
    
    __init__(alpha: float = 0.10, gamma_grid: list = [0.001, 0.005, 0.01, 0.05, 0.1])
        - Initializes multiple ACI instances with different γ values
    update(score: float, covered: bool) -> None
        - Updates each ACI instance's α_t
        - Reweights instances via exponential aggregation
    get_threshold() -> float
        - Returns aggregated quantile threshold for current prediction
```

#### Step 3: Mondrian Categorization

```
class RegimeMondrian:
    """Maps regime probabilities to Mondrian categories."""
    
    categorize(regime_prob: float) -> str
        - P(bull) >= 0.7 → "bull"
        - 0.3 <= P(bull) < 0.7 → "transition" 
        - P(bull) < 0.3 → "bear"
    check_category_sizes(cal_set: pd.DataFrame, alpha: float) -> dict
        - Returns {category: n_samples} and warns if any < ceil(1/alpha) - 1
```

#### Step 4: Prediction Set Generation

```
class ConformalPredictor:
    """Generates prediction sets with coverage guarantees."""
    
    __init__(scorer: KNNNonconformityScorer, calibrator: DtACICalibrator, mondrian: RegimeMondrian)
    calibrate(cal_scores: np.ndarray, cal_labels: np.ndarray, cal_regimes: np.ndarray) -> None
        - Computes per-category nonconformity score distributions
    predict_set(test_score: float, test_regime: float) -> set
        - Returns prediction set: {UP}, {UP, HOLD}, {UP, DOWN}, or {UP, DOWN, HOLD}
    singleton_rate(predictions: list) -> float
        - Returns fraction of predictions that are singletons
```

#### Step 5: Walk-Forward Integration

Modify `scripts/run_walkforward.py`:
- After calibration step, compute nonconformity scores for test set
- Generate prediction sets via `ConformalPredictor`
- Log per-fold: singleton rate, doubleton rate, full-set rate, empirical coverage
- Compute BSS on singleton-filtered subset (should be ≥ unfiltered BSS)
- Remove confidence threshold from decision logic

#### Step 6: Diagnostics and Monitoring

Per-fold output:
- `singleton_rate`: fraction of predictions with C(x) = {UP} or {DOWN} (actionable)
- `empirical_coverage`: fraction of true labels contained in prediction sets
- `coverage_by_regime`: per-Mondrian-category coverage (should each be ≥ 1−α)
- `category_sizes`: calibration samples per Mondrian category
- `bss_filtered`: BSS computed only on singleton predictions
- `bss_unfiltered`: BSS computed on all predictions (existing metric)

### Verification Commands

```bash
pytest tests/test_conformal.py -v

# Walk-forward with conformal gating
python scripts/run_walkforward.py --conformal=True --alpha=0.10 --output=results/e2_conformal/

# Coverage check
python scripts/check_coverage.py results/e2_conformal/ --target=0.90

# Compare filtered vs unfiltered BSS
python scripts/compare_bss.py results/e2_conformal/ --metric=singleton_filtered
```

### Task Type

**MIX** — Steps 1-3 are **SR** (nonconformity score design, DtACI aggregation logic, Mondrian category boundaries), Steps 4-6 are **JR** (wiring with clear specs).

---

## Carry-Forward Items

| Item | Status | Blocks | Phase |
|------|--------|--------|-------|
| Dual-gating: BSS (model gate) + conformal (instance gate) | **Design locked** | Phase 7 E2 | Phase 7 |
| KNN distance-ratio nonconformity score | Recommended default | Phase 7 E2 implementation | Phase 7 |
| DtACI over plain ACI | **LOCKED** — eliminates γ tuning | Phase 7 E2 | Phase 7 |
| Mondrian CP with regime as categorization variable | Connects to Paper #3 (HMM filtered probs) | Phase 7 E2 + E4/E5 | Phase 7 |
| Library: MAPIE primary, crepes fallback | Recommended, not locked until Phase 7 | Phase 7 E2 | Phase 7 |
| Confidence threshold (0.55/0.65) obsoleted by conformal singleton filter | **Architecture note** — resolves provenance gap. Remains unresolved -- conformal approach blocked by 52T probability constraint | Phase 7 E2 | Phase 7 |
| `SelfCalibratingConformal` (NeurIPS 2024) | Evaluate as single-framework alternative to E1+E2 two-step | Phase 7 E1/E2 | Phase 7 |
| Full stack: KNN → Venn-ABERS → Mondrian CP with DtACI | **Design reference** | Phase 7 E1+E2 | Phase 7 |
| E1 must precede E2 (calibration quality drives singleton rate) | **Ordering locked**. E1 was BMA (FAIL); Venn-ABERS untested | Phase 7 scheduling | Phase 7 |
| Per-regime calibration sample monitoring (≥ ⌈1/α⌉ − 1 per category) | Implementation requirement | Phase 7 E2 | Phase 7 |
| Singleton rate replaces confidence threshold as trade-frequency control | **Architecture decision** | Phase 7 E2 | Phase 7 |
| Online Platt Scaling with calibeating (Gupta & Ramdas, ICML 2023) | Shelf — streaming/live alternative | Phase 9/10 | Phase 9+ |

---

## Connection Map: How Papers #1, #2, #3 Integrate

```
Paper #3 (Regime Detection)
    │
    ├── H7: Threshold RegimeLabeler (existing, Phase 1)
    ├── H8: HMM 2-state filtered probabilities (if H7 fails)
    │       ↓
    │   Regime probabilities feed into:
    │       ├── KNN feature space (H8a: 9th feature dimension)
    │       └── Mondrian categories (E2: conformal stratification)
    │
Paper #1 (Venn-ABERS Calibration)
    │
    ├── Phase 7 E1: Replace beta_abm with Venn-ABERS
    │       ↓
    │   Calibrated P(UP) feeds into:
    │       ├── Kelly position sizing (how much to bet)
    │       └── Nonconformity scores (E2: better scores → more singletons)
    │
Paper #2 (Conformal Prediction)
    │
    ├── Phase 7 E2: Dual-gating architecture (this document)
    │       ↓
    │   Prediction sets feed into:
    │       ├── Trade/abstain decision (singleton filter)
    │       └── Replaces confidence threshold (0.55/0.65 provenance gap resolved)
    │
    └── Combined stack: KNN → Venn-ABERS → Mondrian CP (DtACI)
```

---

## Post-Phase-7 Status (2026-04-10)

- **Phase 7 E4 tested plain ACI on 52T walk-forward: FAIL.** Coverage=0.814 mean (gate needed >=0.88), width=1.000 (trivial [0,1] intervals), 2020-COVID fold: 0% coverage.
- **Root cause:** 52T beta_abm probability compression [0.50-0.59]. |prob-label| nonconformity scores are always >=0.41; quantile threshold ~0.57 produces near-trivial prediction sets.
- **DtACI (this doc) faces the same constraint.** The structural issue is the input signal quality, not the conformal method choice.
- **Path forward:** Test on 585T production signals where probs reach [0.65-0.75], or after an algorithm change that widens the probability range (e.g., LightGBM from H9).
- **The dual-gating architecture (BSS + conformal) remains sound in principle** -- it is the input signal quality at 52T scale that blocks it, not a flaw in the architecture itself.

---

## Key References

- Vovk, Gammerman, Shafer (2005) — *Algorithmic Learning in a Random World* (Mondrian CP, Section 4.5)
- Gibbs & Candès (NeurIPS 2021) — Adaptive Conformal Inference (ACI)
- Gibbs & Candès (JMLR 2024) — DtACI (eliminates γ choice)
- Romano, Sesia & Candès (NeurIPS 2020) — Adaptive Prediction Sets (APS)
- Angelopoulos et al. (ICLR 2021) — RAPS (regularized, 5-10× smaller sets)
- Wang, Sun & Dobriban (2025) — SOCOP (singleton-optimized CP, +20% singleton rate)
- Ding et al. (NeurIPS 2023) — Class-conditional coverage failures in standard CP
- Barber, Candès, Ramdas & Tibshirani (Annals of Statistics 2023) — Conformal beyond exchangeability
- Schmitt (2025) — Conformal VaR on CRSP: uncalibrated = 431% overshoot, time-weighted = 1.09%
- Dabah & Tirer (ICML 2025) — Calibration-coverage tradeoff (non-monotonic)
- NeurIPS 2024 — SelfCalibratingConformal (Venn-ABERS + CP combined)
