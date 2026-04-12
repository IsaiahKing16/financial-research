# Mondrian conformal prediction could replace your regime filter

**Mondrian conformal prediction (MCP) provides exactly the per-regime calibration guarantee that FPPE needs — and the literature gap means you'd be pioneering its application in finance.** The core mechanism is simple: partition calibration data by regime label and compute conformal p-values within each stratum, yielding finite-sample, distribution-free coverage guarantees *per category*. This replaces blanket Bear-regime signal suppression with appropriately wider prediction sets that still allow selective trading. The `crepes` Python library provides first-class Mondrian support compatible with KNN-based nonconformity scores. However, the critical constraint is calibration set size per regime — Bear categories need at minimum **⌈1/α⌉−1 samples** (19 for α=0.05) to avoid degenerating into the trivial "include all labels" prediction set, which is functionally equivalent to the current regime filter anyway.

## Label-conditional validity solves the right problem

Standard conformal prediction guarantees only **marginal coverage**: averaged across all examples, the error rate is at most ε. This permits severe asymmetry — the minority class or difficult regime can have far higher error rates, compensated by lower rates elsewhere. Mondrian CP, introduced in Section 4.5 of Vovk et al.'s *Algorithmic Learning in a Random World* (2005), strengthens this to **category-wise exact validity**: for any Mondrian taxonomy κ that partitions examples into categories, the p-values within each category are exactly uniformly distributed on [0,1]. This is a finite-sample result, not asymptotic.

The name derives from Piet Mondrian's grid-based paintings — the taxonomy partitions the example space into non-overlapping rectangles. Formally, a Mondrian taxonomy is a function mapping data sequences to category sequences that is **equivariant with respect to permutations** — the category assignment depends on the bag of examples, not their ordering. The p-value for a test example is computed only against calibration examples sharing the same category:

**p^y = |{i : κ_i = κ_test AND α_i ≤ α_test}| / |{i : κ_i = κ_test}|**

The most critical theoretical property for FPPE is that MCP **only requires exchangeability within each category, not across categories**. Bull-regime data need not be exchangeable with Bear-regime data — only within-regime exchangeability is required. This is a significant relaxation that Vovk et al. illustrate with handwritten text recognition: different instances of the same symbol must be exchangeable, but the sequence of symbols need not be.

The "reference class problem" (Balasubramanian et al., 2014) captures the fundamental tradeoff: large categories yield reliable quantile estimates but coarse predictions, while small homogeneous categories give specific predictions but few calibration samples. The recommended adaptive strategy is to start with no conditioning, add label-conditional categories as data grows, then introduce finer taxonomies progressively.

## Three category designs, one clear winner for FPPE

The central architectural question — what to use as the Mondrian category — has three viable answers, each targeting a different failure mode.

**Option A: Regime labels (Bull/Bear/Neutral) as categories.** This directly addresses FPPE's core problem. The theory explicitly supports object-type conditioning (analogous to demographic conditioning in fairness literature). During Bear regimes, the prediction set calibration draws only from Bear-regime calibration data, producing wider sets that reflect higher uncertainty. During Bull regimes, sets are tighter. The per-regime validity guarantee holds exactly in finite samples. This is the recommended starting point.

**Option B: Predicted class labels (Up/Down) as categories.** This is the standard label-conditional conformal prediction, well-studied and implemented via `class_cond=True` in the `crepes` library. It guarantees that error rates are controlled separately for "up" and "down" predictions — important for avoiding asymmetric classification errors in imbalanced markets. However, it does not address regime-dependent calibration degradation, since both classes pool data across regimes.

**Option C: Cross-product (Regime × Class) categories.** This creates six cells (3 regimes × 2 classes), providing both per-regime and per-class validity simultaneously. It is theoretically optimal but practically dangerous: the Berkeley class-conditional CP paper (Ding et al.) shows that for any category with fewer than ⌈1/α⌉−1 calibration instances, the conformal predictor degenerates to including all labels. With α=0.05, Bear×Up and Bear×Down cells each need **19+ samples**, a threshold easily violated in walk-forward validation with short Bear windows.

**The recommendation is Option A, with a fallback strategy.** When Bear calibration data drops below the minimum threshold, temporarily merge Bear and Neutral into a "non-Bull" super-category (inspired by clustered conformal prediction from Ding et al.). This maintains some regime-specificity while avoiding degenerate prediction sets. The `crepes` library's `MondrianCategorizer` directly supports this dynamic recategorization through its `mc` function parameter.

## The practical architecture: global scores, stratified p-values

The implementation follows a straightforward two-step pattern. First, compute nonconformity scores using the global KNN model trained on all regimes. Second, compute conformal p-values by comparing the test point's score only against calibration scores within the matching regime category.

This "global NCM, stratified p-values" architecture preserves the benefits of training on all available data (a better KNN model) while achieving per-regime calibrated statistical inference. The alternative — computing NCMs only within each regime's data — would sacrifice training data and is not required by the theory.

For binary classification, the Mondrian conformal prediction set at confidence level 1−α contains either `{up}`, `{down}`, `{up, down}`, or `∅`. Singleton sets are actionable trading signals. The `{up, down}` set indicates insufficient evidence to distinguish — this is the "abstain" signal, replacing the current regime filter. During Bear regimes, you'd expect more `{up, down}` predictions (reflecting genuine uncertainty), but **some signals would still be singletons with per-regime calibrated confidence**. This is strictly superior to blanket suppression.

Here is a concrete implementation using `crepes`:

```python
from crepes import ConformalClassifier
from crepes.extras import hinge

# Compute nonconformity scores using global KNN model
alphas_cal = hinge(knn_model.predict_proba(X_cal), knn_model.classes_, y_cal)
regime_cal = get_regime_labels(X_cal)  # [0=bear, 1=neutral, 2=bull]

# Fit Mondrian conformal classifier with regime categories
cc = ConformalClassifier()
cc.fit(alphas_cal, bins=regime_cal)

# At prediction time
alphas_test = hinge(knn_model.predict_proba(X_test))
regime_test = get_regime_labels(X_test)
prediction_sets = cc.predict_set(alphas_test, bins=regime_test, confidence=0.90)
p_values = cc.predict_p(alphas_test, bins=regime_test)
```

The computational overhead of Mondrian vs. standard conformal prediction is **negligible** — the only additional cost is grouping calibration scores by category and computing quantiles per group instead of globally. The dominant cost remains the KNN inference itself. Using hnswlib's approximate nearest neighbors does not invalidate the conformal guarantee (which depends on exchangeability, not exact computation), though it may slightly affect efficiency (interval width). The approximation error is controllable via hnswlib's `ef` parameter.

## Python ecosystem: `crepes` is the clear choice

Among available implementations, **`crepes`** (Henrik Boström, KTH Royal Institute of Technology) provides first-class Mondrian support with the most direct API. Version 0.9.0 offers Mondrian conformal classifiers, regressors, and predictive systems through a `MondrianCategorizer` class and an `mc` parameter that accepts any function mapping examples to category labels. It includes built-in KNN difficulty estimation via `DifficultyEstimator`, hinge and margin nonconformity scores, and even exchangeability testing via martingales — useful for monitoring whether within-regime exchangeability holds. The library is BSD-3-licensed, actively maintained with **562 GitHub stars**, and published at COPA/PMLR conferences.

**MAPIE** (scikit-learn-contrib, backed by Capgemini and ENS Paris-Saclay) is the most widely adopted conformal prediction library but temporarily removed its `MondrianCP` class in v1 for API redesign. Manual Mondrian implementation is documented but requires stratifying the conformalization set yourself. For Venn-ABERS specifically, the `venn-abers` package (v1.4.6, by Ivan Petej) provides a scikit-learn-compatible `VennAbersCalibrator`. A notable recent addition is `SelfCalibratingConformal` (NeurIPS 2024), which combines Venn-ABERS calibration with conformal prediction to produce both calibrated probabilities and valid prediction sets — directly relevant to FPPE's dual need for probability estimates (Brier Score) and coverage guarantees.

For FPPE's layered architecture, the recommended stack is: **hnswlib KNN → Venn-ABERS probability calibration → Mondrian conformal prediction sets**, where Venn-ABERS-calibrated scores feed into Mondrian CP as better-quality nonconformity measures.

## The literature gap is real — and an opportunity

**Zero published papers apply Mondrian conformal prediction to financial forecasting, trading systems, or stock direction prediction.** Zero papers apply Mondrian Venn prediction machines to finance. The Mondrian CP literature is concentrated in drug discovery/chemoinformatics (Toccaceli & Gammerman, 2019), industrial manufacturing (Zhang & Zhou, 2025), and general ML methodology.

Standard conformal prediction in finance exists but is thin: **~5-10 identifiable papers** since 2020. The most relevant include Kaya & Nguyen (COPA 2025) on stock selection, Kato (2025) on conformal predictive portfolio selection, and Wisniewski et al. (COPA 2020) on FX market maker positions. The DCP paper by Wüthrich & Chernozhukov (PNAS, 2021) is particularly instructive: **when predicting daily stock returns, standard 90% conformal intervals dropped to ~50% coverage during high-volatility periods**, while distributional conformal prediction maintained ~90% across all volatility levels. This directly mirrors FPPE's Bear-regime calibration failure.

The closest conceptual analog to regime-based Mondrian CP is the **CPTC algorithm** (Sun & Yu, NeurIPS 2025), which integrates a switching dynamical system with online conformal prediction, running state-specific conformal inference within each detected regime. Though the authors don't use Mondrian terminology, the mechanism is nearly identical: per-state calibration with adaptive aggregation during transitions. Lu et al. (2025) similarly couple deep switching state-space models with adaptive conformal inference, achieving near-nominal coverage across regime changes.

The **adaptive conformal inference (ACI)** framework of Gibbs & Candès (NeurIPS 2021, JMLR 2024) addresses within-regime non-stationarity through dynamically-tuned calibration — tested on stock market volatility prediction where it maintained coverage through the 2008 crisis while non-adaptive methods failed catastrophically. Combining ACI's dynamic calibration with Mondrian's regime stratification would address both between-regime and within-regime non-stationarity, though no paper has attempted this synthesis.

## The within-regime exchangeability challenge remains

The principal theoretical risk is **non-stationarity within each regime**. MCP requires exchangeability within categories, but financial data within a single Bull or Bear regime exhibits trending behavior, autocorrelation, and evolving volatility structure. Three mitigation strategies have empirical support.

First, **adaptive conformal inference (ACI)** from Zaffran et al. (ICML 2022) dynamically updates calibration quantiles with an online learning rate. This can be combined with Mondrian categories — run ACI separately within each regime stratum. Second, **rolling calibration windows** use only recent within-regime data, maintaining approximate exchangeability at the cost of reducing effective calibration set size. Third, the `crepes.martingales` module provides exchangeability testing — monitoring the martingale statistic within each regime would signal when the exchangeability assumption breaks down, triggering either window shrinkage or regime reclassification.

The graceful degradation property of Mondrian CP provides a safety net: when Bear calibration data is insufficient (fewer than ⌈1/α⌉−1 samples), the prediction set automatically includes all labels — functionally equivalent to the current regime filter's signal suppression. The system never produces miscalibrated signals; it simply abstains. This means **the worst-case behavior of Mondrian CP exactly matches the current regime filter**, while the best case produces actionable, calibrated signals during Bear regimes that the current approach discards entirely.

## Conclusion

Mondrian conformal prediction with regime labels as categories is a theoretically sound and practically implementable replacement for FPPE's blanket Bear-regime signal suppression. The key insight is that MCP's **finite-sample, per-category validity guarantee requires only within-category exchangeability** — a weaker assumption than standard CP's global exchangeability. The `crepes` library provides a production-ready implementation requiring fewer than 20 lines of code changes. The critical bottleneck is Bear-regime calibration set size: below 19 samples (for α=0.05), the method degenerates to the current approach's behavior — a graceful failure mode. The complete absence of Mondrian CP applications in finance represents both a literature gap and a genuine research opportunity, particularly the unexplored combination of regime-stratified Mondrian categories with adaptive conformal inference for handling within-regime non-stationarity. For FPPE specifically, the recommended next step is an A/B comparison of Mondrian CP (regime categories) against the current regime filter, evaluated via Brier Skill Score in walk-forward validation, with the prediction set singleton rate during Bear regimes as the key diagnostic metric.