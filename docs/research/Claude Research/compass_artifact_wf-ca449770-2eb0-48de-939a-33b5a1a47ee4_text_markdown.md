# MAPIE for conformal prediction in financial classification

**MAPIE (v1.3.0) provides distribution-free prediction sets with formal coverage guarantees for any sklearn-compatible classifier, including custom KNN estimators, but its utility for binary up/down equity classification is fundamentally limited.** Prediction sets in the binary case can only be ∅, {up}, {down}, or {up, down} — the last being completely uninformative. This constraint makes conformal prediction sets less powerful than in multiclass settings, though they still serve as a rigorous uncertainty quantification layer. For a KNN-based FPPE system, MAPIE integrates cleanly via its `SplitConformalClassifier` API, requiring only that the hnswlib-backed estimator expose `fit()`, `predict()`, `predict_proba()`, and `classes_`. The library lacks time-series classification support entirely, and its Mondrian (per-regime) conformal module is temporarily unavailable in v1, requiring manual workarounds.

---

## How MapieClassifier wraps estimators and produces prediction sets

MAPIE v1.3.0 (released February 2025) restructured its classification API into two public classes: **`SplitConformalClassifier`** for split conformal and **`CrossConformalClassifier`** for cross-conformal (K-fold). The old `MapieClassifier` persists as a private `_MapieClassifier` backend. Both classes accept any scikit-learn `ClassifierMixin` and follow a three-step workflow: fit the base classifier, conformalize on held-out calibration data, then produce prediction sets at inference.

The core mechanism is split conformal prediction. Given a calibration set of n exchangeable examples, MAPIE computes a **nonconformity score** $s_i = s(X_i, Y_i)$ for each calibration point, then finds the quantile $\hat{q} = \text{Quantile}(s_1, \ldots, s_n; \lceil(n+1)(1-\alpha)\rceil / n)$. At test time, the prediction set includes all labels whose score falls below this quantile threshold. The formal coverage guarantee is:

$$1 - \alpha \leq P(Y_{\text{test}} \in C(X_{\text{test}})) \leq 1 - \alpha + \frac{1}{n+1}$$

This is a **marginal** coverage guarantee — finite-sample valid, distribution-free, and model-agnostic. It requires only **exchangeability** of the data (weaker than i.i.d.), meaning the joint distribution is invariant to permutations. The upper bound $1/(n+1)$ becomes negligible for calibration sets of **n ≥ 500** (overcoverage < 0.2%). Angelopoulos and Bates (2021) recommend **n ≈ 1,000** calibration points for stable coverage. For a ~50/50 binary split, this is achievable in most financial datasets, though the exchangeability assumption is violated by temporal dependence in equity returns.

The v1 API for split conformal follows this pattern:

```python
from mapie.classification import SplitConformalClassifier

mapie = SplitConformalClassifier(
    estimator=my_knn_classifier,
    confidence_level=0.90,
    conformity_score="lac",
    prefit=True
).conformalize(X_cal, y_cal)

predictions, prediction_sets = mapie.predict_set(X_test)
# prediction_sets shape: (n_samples, 2, n_confidence_levels) for binary
```

---

## LAC vs APS: which method for binary equity classification

The two primary conformity score methods differ fundamentally in how they construct prediction sets, and this difference matters critically for binary classification.

**LAC (Least Ambiguous set-valued Classifier)**, from Sadinle, Lei, and Wasserman (2019), uses the nonconformity score $s(x, y) = 1 - \hat{p}_y(x)$, where $\hat{p}_y(x)$ is the estimated probability of the true class. The prediction set includes all labels $y$ where $\hat{p}_y(x) \geq 1 - \hat{q}$. LAC is provably optimal — it produces the **smallest average prediction set size** when the model's probabilities are perfectly calibrated. However, it can produce **empty prediction sets** when no class exceeds the threshold, which occurs near decision boundaries where both classes have probabilities close to 0.5.

**APS (Adaptive Prediction Sets)**, from Romano, Sesia, and Candès (2020), uses a cumulative score: sort classes by decreasing predicted probability, then sum probabilities until reaching the true class's rank. The score is $s(x, y) = \sum_{j=1}^{k} \hat{p}_{\pi_j}(x)$ where the true label appears at rank $k$. APS **never produces empty sets** by construction (it always includes at least the top-ranked class). MAPIE adds optional randomization via the `include_last_label` parameter (`True`, `False`, or `"randomized"`) to fine-tune coverage.

For **binary up/down classification**, the prediction set can only be one of four outcomes: ∅ (LAC only), {up}, {down}, or {up, down}. The set {up, down} is completely uninformative. This makes binary conformal prediction inherently limited — MAPIE's own documentation explicitly states that prediction sets are "much less informative for binary classification than for multiclass."

**Typical prediction set distributions for a reasonably discriminative binary classifier (AUC ~0.80):**

| Coverage | LAC empty | LAC singleton | LAC both | APS singleton | APS both |
|----------|-----------|---------------|----------|---------------|----------|
| **80%** (α=0.20) | 5–10% | 80–85% | 5–15% | 85–90% | 10–15% |
| **90%** (α=0.10) | 2–5% | 70–80% | 15–25% | 75–85% | 15–25% |
| **95%** (α=0.05) | 1–3% | 55–70% | 30–40% | 60–75% | 25–40% |

These distributions degrade dramatically with weaker classifiers. Krstajic (2020) demonstrated that with a near-random classifier (AUC ~0.53), **950 of 1,000 predictions** became {both classes} at 95% coverage — technically valid but operationally useless.

**For the FPPE system, APS is the better choice.** LAC's empty sets create an operational hazard: a prediction that includes neither class requires a separate fallback mechanism. APS guarantees at least one class in every prediction set. The tradeoff — slightly larger average set sizes — is acceptable given that binary prediction sets can only differ by one element. Under **class imbalance**, LAC degrades more visibly (more empty sets for minority-class instances), while APS degrades subtly through larger sets. For equity returns near 50/50, this difference is modest, but market regimes can shift the effective class balance.

---

## Time-series support is regression-only with no classification path

MAPIE provides `TimeSeriesRegressor` (formerly `MapieTimeSeriesRegressor`) implementing two methods for temporal data, but **neither applies to classification**.

**EnbPI (Ensemble Batch Prediction Intervals)**, from Xu and Xie (ICML 2021), replaces the exchangeability assumption with assumptions about residual stationarity. It fits B bootstrap models on non-overlapping temporal blocks via `BlockBootstrap`, aggregates predictions, and constructs intervals from residual quantiles. The critical innovation is **online residual updating**: as new labeled data arrives, `update()` (v1) or `partial_fit()` (v0.x) replaces old conformity scores with fresh ones, allowing intervals to adapt to changing volatility or model drift.

**ACI (Adaptive Conformal Inference)**, from Gibbs and Candès (NeurIPS 2021), dynamically adjusts $\alpha_t$ at each time step using the update rule $\alpha_{t+1} = \alpha_t + \gamma(\alpha - \text{err}_t)$. If the true value falls outside the interval, $\alpha_t$ decreases (intervals widen); if inside, $\alpha_t$ increases (intervals narrow). The `gamma` parameter controls adaptation speed. ACI achieves desired coverage frequency over long horizons **without exchangeability**, making it robust to distribution shift.

Both methods are implemented exclusively in `TimeSeriesRegressor`, which checks that the wrapped estimator is a `RegressorMixin`. **There is no `TimeSeriesClassifier` in MAPIE.** Xu and Xie published an extension to prediction sets ("Conformal prediction set for time-series," ICML 2022 workshop), but MAPIE has not implemented it. For the FPPE system, the practical workaround is to use standard `SplitConformalClassifier` with a **rolling calibration window** — periodically re-running `.conformalize()` on recent data to approximate temporal adaptation. This sacrifices formal guarantees under non-exchangeability but preserves the conformal framework.

---

## Integrating a custom hnswlib KNN estimator with MAPIE

A custom KNN classifier backed by hnswlib **can be wrapped by MAPIE** provided it implements the scikit-learn interface. The minimum requirements are:

- **`fit(X, y)`** — must set `self.classes_ = np.unique(y)` during fitting
- **`predict(X)`** — returns class labels as `NDArray` of shape `(n_samples,)`
- **`predict_proba(X)`** — returns **normalized** probabilities as `NDArray` of shape `(n_samples, n_classes)` where rows sum to 1.0 (MAPIE validates this via `check_proba_normalized`)
- Inherit from `sklearn.base.BaseEstimator` and `sklearn.base.ClassifierMixin`
- All `__init__` parameters stored as attributes (for `sklearn.clone()` compatibility)

The recommended integration pattern for a stateful estimator like hnswlib KNN is the **prefit strategy**: fit the estimator externally, then pass it to `SplitConformalClassifier` with `prefit=True`, which skips internal fitting and only computes conformity scores on calibration data.

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import hnswlib

class HNSWLibKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, space='l2', ef=200, M=16):
        self.n_neighbors = n_neighbors
        self.space = space
        self.ef = ef
        self.M = M

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.y_train_ = y
        self.index_ = hnswlib.Index(space=self.space, dim=X.shape[1])
        self.index_.init_index(max_elements=len(X), ef_construction=self.ef, M=self.M)
        self.index_.add_items(X, np.arange(len(X)))
        self.index_.set_ef(self.ef)
        return self

    def predict_proba(self, X):
        labels, _ = self.index_.knn_query(X, k=self.n_neighbors)
        neighbor_labels = self.y_train_[labels]
        proba = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            proba[:, i] = (neighbor_labels == cls).mean(axis=1)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
```

**Key constraint**: when using cross-conformal (`CrossConformalClassifier`), `sklearn.clone()` must be able to create unfitted copies. Stateful hnswlib indices stored as fitted attributes (ending in `_`) are correctly discarded by `clone()`. However, if the hnswlib index is non-picklable, cross-validation with `n_jobs > 1` will fail. Use `prefit=True` with `SplitConformalClassifier` to sidestep this entirely. The calibration/training split should dedicate **at least 1,000 samples** to calibration for stable coverage, with the remainder for training.

---

## Mondrian conformal prediction requires manual implementation in v1

Mondrian (group-conditional) conformal prediction guarantees coverage **per group** — critical for the FPPE system if different market regimes require separate validity. MAPIE v0.9.x introduced a `MondrianCP` class in `mapie.mondrian` that wrapped `MapieClassifier` or `MapieRegressor`, accepting a `partition` array of integer group labels. Internally, it created separate MAPIE estimator copies per group, fitted each on its group's data, and produced per-group prediction sets.

**In MAPIE v1.x, `MondrianCP` is temporarily unavailable.** The v1 release notes state: "The MondrianCP class is temporarily unavailable in v1. We want to rethink the way we integrate Mondrian to MAPIE, in a future-proof way." The documentation provides a manual workaround: partition data by regime, fit separate `SplitConformalClassifier` instances per regime, predict per regime, and aggregate results.

The `groups` parameter in `CrossConformalClassifier.fit_conformalize()` serves a different purpose — it controls **cross-validation group management** (preventing data leakage between CV folds), not Mondrian group-conditional coverage. These are distinct concepts despite the overlapping terminology.

**Per-group coverage guarantee**: when fitting separate conformal predictors per group $g$, the guarantee becomes $P(Y \in C_g(X) | G = g) \geq 1 - \alpha$ for each group individually. This is strictly stronger than marginal coverage. The tradeoff is that each group needs sufficient calibration data independently — with $k$ regimes and a target of 500 calibration points per regime, total calibration requirements scale to $500k$. For the FPPE system with, say, 4 market regimes, this means reserving ~2,000 calibration samples total.

---

## Prediction set ambiguity and BSS measure fundamentally different properties

The Brier Score decomposes via Murphy's formula into **reliability** (calibration), **resolution** (sharpness/discrimination), and **uncertainty** (irreducible base-rate term). BSS = $1 - BS/BS_{ref}$ thus captures a blend of calibration quality and discriminative power. A recent analysis (arXiv:2504.04906) demonstrates that a low Brier Score does **not** necessarily indicate good calibration — a model can achieve lower BS through better resolution despite worse calibration.

Conformal prediction set ambiguity — the fraction of predictions returning {up, down} — measures something distinct: **instance-level uncertainty thresholded by the calibration quantile**. It identifies specific data points where the model cannot distinguish classes at the requested confidence level.

| Property | BSS | Prediction set ambiguity |
|----------|-----|--------------------------|
| Calibration | Captures (mixed with sharpness) | Reflects indirectly — poor calibration → larger sets |
| Discrimination/sharpness | Captures directly | Does not measure |
| Instance-level uncertainty | Aggregate metric only | Yes — flags specific uncertain points |
| Formal guarantee | None | $P(Y \in C(X)) \geq 1-\alpha$ |
| Proper scoring rule | Yes (BS is strictly proper) | Not a scoring rule |

**Ambiguity cannot replace BSS**, but it supplements it powerfully. BSS tells you whether your probability estimates are accurate in aggregate. Ambiguity tells you **where** the model should abstain. A model could have excellent BSS but high ambiguity (well-calibrated on average but uncertain in boundary regions). Conversely, a model could show low ambiguity but poor BSS (overconfident and wrong). The combination is diagnostic: if ambiguity is high but BSS is also high, the model is honestly uncertain on hard cases. If ambiguity is low but BSS is poor, the model is dangerously overconfident. **Tracking ambiguity rate across coverage levels** (e.g., 80%, 90%, 95%) produces a curve analogous to a calibration curve — a well-discriminating model's ambiguity rate rises slowly with coverage, while a weak model's ambiguity rate explodes.

---

## Conclusion

MAPIE v1.3.0 provides a clean, well-engineered conformal prediction framework that integrates with custom sklearn-compatible classifiers, including hnswlib-backed KNN. **For binary equity classification, APS is the recommended method** — it avoids empty sets and guarantees at least one class per prediction — but practitioners should expect that 25–40% of predictions at 95% coverage will return the uninformative {up, down} set for a model with AUC around 0.80. This ambiguity rate itself becomes a valuable signal: it quantifies regime-specific difficulty in a way BSS alone cannot.

Three significant gaps exist for the FPPE use case. First, **no time-series classification support** — the system must use standard split conformal with rolling recalibration to approximate temporal adaptation. Second, **Mondrian conformal is unavailable in v1** — per-regime coverage requires manual implementation via separate classifier instances. Third, the **exchangeability assumption** is formally violated by financial time series, meaning the coverage guarantee is approximate rather than exact in practice. The pragmatic approach is to treat conformal prediction as a calibrated uncertainty layer rather than a hard guarantee, recalibrating frequently on recent data to maintain approximate validity under distribution shift.