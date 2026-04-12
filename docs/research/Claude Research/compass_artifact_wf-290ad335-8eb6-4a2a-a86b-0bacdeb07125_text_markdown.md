# Feature selection and metric learning for KNN with hnswlib at scale

**For a financial KNN system with 23 features, 50K+ samples, and hnswlib indexing, the most practical path is LMNN or NCA metric learning to simultaneously select features and learn an optimal distance metric, followed by pre-transforming data into a lower-dimensional space before building the hnswlib L2 index.** ReliefF provides the fastest filter-based baseline, running in minutes on this scale. Feature weighting via gradient optimization offers a middle ground, and sequential feature selection—not traditional RFE—is the correct wrapper approach for KNN. All five methods are computationally feasible at this scale, but they differ dramatically in how well they integrate with approximate nearest neighbor search and handle financial data's low signal-to-noise characteristics.

The key architectural insight is that hnswlib supports only L2, cosine, and inner product distances—so any learned metric must be converted to a feature transformation applied *before* indexing. Both LMNN and NCA learn exactly such transformations, and both can simultaneously reduce dimensionality from 23 to 8 features, which improves hnswlib recall and speed.

---

## ReliefF scores features through nearest-hit/miss contrasts

ReliefF (Kononenko, 1994) extends the original Relief algorithm (Kira & Rendell, 1992) by using **k nearest neighbors** from each class rather than a single nearest hit and miss. For each sampled instance, it finds k nearest same-class neighbors ("hits") and k nearest different-class neighbors ("misses"), then updates per-feature weights: features that differ more between misses than hits receive higher scores. The weight update formula for multi-class problems weights each class's miss contribution by its prior probability P(C)/(1−P(class(Rᵢ))), handling class imbalance naturally.

The algorithm's computational complexity is **O(n²d)** when all n instances are sampled (m=n), dominated by pairwise distance computation. For 23 features and 50,000 samples, this means roughly **57.5 billion operations** for the full distance matrix—feasible in **2–10 minutes** on modern hardware with vectorized NumPy operations. Memory for the 50K×50K float32 distance matrix is ~9.3 GB, significant but manageable. The algorithm is embarrassingly parallelizable since each instance's weight contribution is independent.

**SURF** (Greene et al., 2009) replaces the fixed-k neighborhood with a distance threshold equal to the mean pairwise distance, eliminating k as a hyperparameter. **SURF*** (Greene et al., 2010) additionally uses far instances with inverted scoring, improving interaction detection but sacrificing univariate signal detection. The most recent variant, **MultiSURF** (Urbanowicz et al., 2018), uses per-instance adaptive thresholds based on mean ± standard deviation of distances and was shown to perform most consistently across problem types in comprehensive benchmarking.

A critical limitation: **ReliefF cannot detect redundant features**. Two perfectly correlated features both receive high weights. Post-processing with correlation-based filtering is necessary. ReliefF also degrades when many irrelevant features corrupt the distance metric used for neighbor finding, though with only 23 features this is unlikely to be severe. The iterative TuRF variant (Moore & White, 2007) progressively removes low-scoring features to improve neighbor quality in subsequent iterations.

The primary Python implementation is **scikit-rebate** (skrebate v0.7), which includes ReliefF, SURF, SURF*, MultiSURF, and TuRF. However, the package is **largely unmaintained**—no new releases in over 12 months, with Travis CI badges referencing Python 2.7/3.6. It may require compatibility patches for scikit-learn ≥1.0. The **ReBATE** standalone package uses Cython for better performance. For 23 features, a custom NumPy implementation is also straightforward (<100 lines of code).

**Integration with hnswlib** requires pre-transforming features: given ReliefF weights W, compute `scale = sqrt(clip(W, 0, None))` and multiply each feature by its corresponding scale factor. Standard L2 distance in the transformed space then equals the weighted Euclidean distance. Features with negative weights (deemed harmful) can be zeroed out, effectively performing feature selection. The hnswlib index must be rebuilt whenever weights change.

ReliefF has been applied directly in financial prediction. Miao et al. (2025) combined ReliefF with RFE for stock movement direction prediction on Chinese stock datasets, finding it "effectively identifies discriminative feature subsets" of technical indicators. Bragança et al. (2025) compared Relief and Information Gain for Brazilian stock market prediction. However, ReliefF does **not handle non-stationarity** inherently—rolling-window retraining on recent data windows is necessary.

---

## Learned feature weights turn KNN's distance metric into a tunable parameter

Feature weighting learns a diagonal Mahalanobis matrix M = diag(w₁,...,w_d), equivalent to a weighted Euclidean distance d(x,y) = √(Σ wᵢ(xᵢ−yᵢ)²). With only **23 weight parameters** to optimize, this is a trivially low-dimensional optimization problem even with expensive per-evaluation costs.

Three families of approaches exist. **Filter-based methods** compute weights from statistical measures in a single pass: Fisher's Discriminant Ratio (FDR_j = (μ₁−μ₂)²/(σ₁²+σ₂²)), mutual information MI(X_j, Y), or information gain. These complete in under a second for 50K×23 data but ignore feature interactions entirely. Chen & Hao (2017) used information gain to weight 9 technical indicators for KNN stock prediction on Shanghai/Shenzhen exchanges, demonstrating improved prediction across multiple time horizons.

**Gradient-based optimization** is more principled. The key challenge is that KNN classification involves discrete neighbor selection, making the loss piecewise-constant with respect to weights. Three solutions exist: NCA uses a softmax over distances to create a smooth probability distribution; LMNN uses hinge-loss upper bounds; and **WkNN-FS** (Bugata & Drotár, 2019) uses distance-weighted prediction with an exponential kernel to create a differentiable objective. WkNN-FS is particularly relevant—it directly optimizes per-feature weights via gradient descent on KNN prediction error, uses the square-root parameterization `v_l` where the actual weight is `v_l²` to avoid zero-gradient traps, and has an open-source implementation (github.com/bugatap/WkNN-FS) with TensorFlow/GPU support. Its per-iteration cost is O(n×k×d) where k≪n, making it significantly faster than the O(n²) methods.

**Wrapper-based coordinate descent** is the simplest practical approach: initialize weights to 1.0, iteratively multiply/divide each weight by a step factor (e.g., 2.0), and accept changes that improve cross-validation accuracy. With 23 dimensions, ~100 iterations of coordinate descent means ~2,300 evaluations, each taking ~1 second for 5-fold CV on 50K samples—roughly **40 minutes** total, entirely feasible.

For hnswlib integration, the standard approach applies: transform features as x'ᵢ = √(wᵢ)·xᵢ, then use L2 distance. During weight optimization, use **brute-force KNN** (fast enough at 50K×23 for ~1ms per query) and only build the hnswlib index with final optimized weights. The metric-learn library supports diagonal Mahalanobis learning via its MMC algorithm with `diagonal=True`, directly compatible with scikit-learn pipelines. Bhardwaj et al. (2018) demonstrated using random forest out-of-bag importance scores, normalized via z-scores, as KNN feature weights—a pragmatic hybrid approach.

---

## LMNN learns a discriminative distance metric via large-margin optimization

Large Margin Nearest Neighbor (Weinberger & Saul, 2009, JMLR 10:207–244) is the most well-established metric learning algorithm specifically designed for KNN. It learns a Mahalanobis distance d_M(x,y) = (x−y)ᵀM(x−y) where **M = LᵀL**, making the learned distance equivalent to Euclidean distance after the linear transformation x′ = Lx.

The loss function combines two terms. The **pull term** minimizes distances between each point and its k pre-selected "target neighbors" (same-class nearest neighbors chosen before training). The **push term** uses hinge loss to penalize "impostors"—differently-labeled points closer than target neighbors plus a unit margin: [d_M(xᵢ,x_j) + 1 − d_M(xᵢ,x_l)]₊. The overall objective ε(L) = (1−μ)·ε_pull + μ·ε_push (typically μ=0.5) is **convex** when optimizing over M with the constraint M ≽ 0, guaranteeing a global minimum. This is a significant advantage over NCA's non-convex formulation.

Computational complexity is **O(n²d) per iteration** naively, but LMNN exploits sparsity: most impostor constraints are naturally satisfied, so the active constraint set is small. Modern implementations (metric-learn, pylmnn) solve the unconstrained optimization over L using L-BFGS or gradient descent rather than the full semidefinite program, dramatically improving practical runtime. For **50K samples with 23 features, LMNN is entirely feasible**—the M matrix has only 529 parameters (23×23), and with dimensionality reduction to 8 dimensions, L has just **184 parameters** (8×23). The original paper demonstrated LMNN on MNIST (60K samples, 784 dimensions), which is substantially harder than this problem. Expected training time: **minutes to tens of minutes**.

The **metric-learn** library (v0.7.0, MIT license, ~1,400 GitHub stars) provides the most actively maintained implementation. Key parameters include `n_components` (set to 8 to reduce from 23 to 8 dimensions), `n_neighbors`, `max_iter`, and `init` ('auto', 'pca', 'lda', 'identity', 'random'). The library is scikit-learn compatible, supporting pipelines, GridSearchCV, and cross_val_score. **pylmnn** (v1.6.3, BSD-3) uses L-BFGS via SciPy and may be faster but is potentially unmaintained since 2018, with compatibility issues for recent scikit-learn versions. For maximum speed, mlpack's C++ implementation with Python bindings offers the fastest option.

**Integration with hnswlib is the cleanest of all five methods.** LMNN's transformation directly produces lower-dimensional features suitable for L2 indexing:

```python
from metric_learn import LMNN
lmnn = LMNN(n_neighbors=5, n_components=8)
lmnn.fit(X_train, y_train)
L = lmnn.components_  # shape (8, 23)

X_transformed = lmnn.transform(X_train)  # (n, 8)
index = hnswlib.Index(space='l2', dim=8)
index.init_index(max_elements=len(X_transformed), ef_construction=200, M=16)
index.add_items(X_transformed)

# Query: transform then search
query_transformed = lmnn.transform(query.reshape(1, -1))
labels, distances = index.knn_query(query_transformed, k=5)
```

The dimensionality reduction from 23→8 simultaneously improves hnswlib performance (fewer distance computations per traversal, better recall at lower dimensions) and learns a discriminative metric. The query-time transformation is a single matrix multiply—**184 multiply-adds, taking microseconds**. Scalability studies show that training on just 10% of samples (5K from 50K) achieves nearly identical performance at 8–40% of training time (Fast LMNN, IEEE 2015).

No published papers apply LMNN directly to financial prediction, making this an underexplored area. For non-stationarity, periodic retraining on rolling windows is necessary since LMNN learns a fixed transformation. Deep metric learning alternatives (triplet networks, Proxy-NCA++) are overkill for tabular data with 23 features—LMNN provides a simpler, interpretable, convex solution.

---

## NCA maximizes a differentiable approximation to KNN accuracy

Neighborhood Component Analysis (Goldberger, Hinton, Roweis & Salakhutdinov, NeurIPS 2004) defines a stochastic neighbor assignment where the probability that point i selects point j as its neighbor follows a softmax: **p(i→j) = exp(−‖Axᵢ − Axⱼ‖²) / Σ_{k≠i} exp(−‖Axᵢ − Axₖ‖²)**. The objective maximizes the expected leave-one-out accuracy f(A) = Σᵢ Σ_{j: cⱼ=cᵢ} p(i→j). This is elegant because it eliminates the need to choose k—the effective number of neighbors emerges from the scale of A.

The gradient ∂f/∂A involves all n² pairwise interactions, giving **O(n²d') per iteration** complexity. Scikit-learn's `NeighborhoodComponentsAnalysis` uses full-batch L-BFGS-B optimization via SciPy, meaning every iteration evaluates the objective across all 2.5 billion pairs for 50K samples. This makes **sklearn's implementation impractical at 50K samples**—expect hours per iteration and ~40–50 GB peak memory. The softmax normalization over all pairs is the bottleneck.

Three scalability solutions exist. **Subsampling** to 10–20K points is the simplest approach. **Mini-batch SGD** via Kevin Zakka's torchnca package (pip install torchnca) runs on GPU with batch sizes of 256–1024, though small batches poorly approximate the full-dataset softmax. **Proxy-NCA** (Movshovitz-Attias et al., ICCV 2017) replaces per-sample comparisons with P≪N class proxy vectors, reducing complexity from O(n²) to O(n×P); Proxy-NCA++ (Teh, DeVries & Taylor, ECCV 2020) improved this further with critical temperature scaling.

Compared to LMNN, NCA is **generally slower** and produces initialization-dependent results due to non-convexity. Weinberger & Saul (2009) directly state: "NCA is the slowest, mainly due to the O(n²) normalization of its softmax probability distributions" and "LMNN outperforms these other methods for distance metric learning on the four largest data sets." However, NCA has a more natural dimensionality reduction capability (rectangular A ∈ ℝ^{d'×d} is a first-class feature) and doesn't require pre-specifying k. For the 23→8 reduction, NCA learns just **184 parameters** with 50K training samples, a very favorable ratio that mitigates overfitting risk.

Integration with hnswlib follows the identical pattern as LMNN: X_transformed = X @ A.T, then build an L2 index on 8-dimensional vectors. The query-time overhead is identical—one matrix multiply per query. Regularized NCA (Yang & Laaksonen, 2007) adds a Gaussian prior on A, which is particularly important for financial data's low SNR to prevent the transformation from amplifying noise.

For the user's system, **LMNN is preferable to NCA** given: (a) convexity guarantees more reliable convergence, (b) better scalability at 50K samples, (c) stronger empirical results on larger datasets, and (d) the k for KNN is presumably already known. NCA's advantage—implicit k selection—is less relevant when k is predetermined. If dimensionality reduction to 8 dimensions is the primary goal, try NCA on a 10–20K subsample first, then compare with LMNN on the full dataset.

---

## Sequential backward selection replaces RFE for KNN classifiers

Traditional RFE (Guyon et al., 2002) cannot work with KNN because it requires `coef_` or `feature_importances_` attributes, which KNN lacks. Scikit-learn's `RFE` with `KNeighborsClassifier` raises a `RuntimeError` (scikit-learn GitHub Issue #6920). The correct adaptation uses **Sequential Feature Selection** (SFS), specifically backward elimination, which evaluates KNN performance directly via cross-validation at each step.

The computational cost is tractable. With 23 features, backward elimination from 23 to the optimal subset requires Σ(d from 23 to 1) = 276 elimination steps. With 5-fold cross-validation, that's **1,380 KNN fit-predict cycles**. Each cycle on 50K samples with 23 features takes ~0.5–2 seconds using scikit-learn's brute-force KNN, giving a total time of roughly **23 minutes on a single core** or **5–10 minutes** with parallelization. This is entirely practical.

Scikit-learn's **`SequentialFeatureSelector`** (added v0.24) is the recommended tool—it works with any estimator regardless of whether it exposes importance attributes. The `direction='backward'` mode performs SBS. Alternatively, **mlxtend**'s `SequentialFeatureSelector` (Sebastian Raschka) offers floating variants (SFFS/SBFS) that add backtracking steps, range-based feature count selection (`k_features=(min, max)`), and fixed-feature constraints, providing more flexibility at somewhat higher computational cost.

A hybrid approach using **permutation importance** as a proxy for RFE is also viable. Since scikit-learn v0.24, RFE accepts an `importance_getter` callable parameter. A custom function computing `permutation_importance(estimator, X, y)` at each elimination step makes RFE work with KNN, though this adds ~10× overhead per step (10 permutation repeats × n predictions). The **eli5** library provides a `PermutationImportance` wrapper that directly exposes `feature_importances_`, making it compatible with `SelectFromModel`.

**Random KNN Feature Selection** (RKNN-FS; Li et al., 2011, BMC Bioinformatics) offers an ensemble alternative: build many KNN models on random feature subsets, then score features by their frequency in high-performing subsets. This showed higher stability than random forest feature selection in noisy settings—potentially valuable for financial data.

For financial applications, the wrapper approach has a critical advantage: it directly optimizes the metric that matters (KNN classification accuracy on the actual data), making it robust to the complex feature interactions and low SNR typical of financial data. The disadvantage is that it cannot learn feature weights—only binary include/exclude decisions—limiting its expressiveness compared to metric learning approaches.

---

## How these methods compare for a financial KNN system at scale

| Method | Training time (50K×23) | Parameters | Dimensionality reduction | hnswlib integration | Library maturity | Financial SNR handling |
|--------|----------------------|------------|------------------------|-------------------|-----------------|----------------------|
| **ReliefF/MultiSURF** | 2–10 min | 0 (scores only) | Via thresholding | Feature scaling | skrebate (unmaintained) | Moderate (TuRF helps) |
| **Feature weighting** | 1 sec–40 min | 23 weights | Via zeroing | Feature scaling | metric-learn, custom | Good (adaptive) |
| **LMNN** | 5–30 min | 184 (8×23) | Native (n_components=8) | Pre-transformation | metric-learn (active) | Untested in finance |
| **NCA** | Hours (sklearn), 10–30 min (subsampled) | 184 (8×23) | Native (n_components=8) | Pre-transformation | sklearn (stable) | Regularization needed |
| **SFS/SBS** | 5–23 min | 0 (subset only) | Via elimination | Reduced features | sklearn, mlxtend (mature) | Direct optimization |

**For hnswlib integration, LMNN and NCA are architecturally superior** because they learn a transformation that simultaneously performs dimensionality reduction and metric learning. Reducing from 23 to 8 dimensions before indexing improves HNSW recall and search speed—the `M` parameter in hnswlib is "tightly connected with internal dimensionality of the data" (Malkov & Yashunin, 2018). Lower dimensionality reduces distance concentration effects that degrade ANN quality.

Comparative studies consistently show LMNN as the strongest classical metric learning method for KNN. Weinberger & Saul (2009) demonstrated it outperforms NCA, RCA, LDA, and PCA on larger datasets. However, a study on the AwA2 dataset found Euclidean distance "performs best among simple metrics and even beats 9 out of 10 metric learning algorithms"—only LMNN slightly outperformed it. This suggests that on some problems, the feature scaling matters more than the learned metric.

A 2025 study on ensemble feature selection for fuzzy KNN compared Chi-squared, Fisher Score, Mutual Information, MRMR, Pearson Correlation, and ReliefF across 12 datasets, finding that **ensemble feature selection** (combining multiple selectors via mean-rank) tends to outperform individual methods. This suggests a practical strategy: run ReliefF, mutual information, and Fisher's discriminant ratio independently, then combine their rankings before applying threshold-based selection or weighting.

Recent advances (2020–2025) in metric learning have focused on deep approaches—Proxy-NCA++ (Teh et al., ECCV 2020), self-supervised contrastive methods, and angular margin losses (ArcFace, CosFace). These are designed for high-dimensional embeddings in vision tasks and are **overkill for 23-feature tabular data**. The comprehensive survey by Ghojogh et al. (2022, arXiv:2201.09267) covers both classical and deep methods. For tabular financial data, classical LMNN with periodic retraining remains the most appropriate choice.

Financial applications of KNN-specific metric learning are sparse. Most financial KNN work uses standard Euclidean distance with basic preprocessing (Chen & Hao, 2017; Tang et al., 2018) or PCA dimensionality reduction. The combination of metric learning + ANN indexing + rolling-window retraining for financial prediction appears to be a genuinely novel application area.

---

## Recommended implementation strategy for the financial KNN system

The most practical approach combines multiple methods in a pipeline. First, use **ReliefF or MultiSURF** as a fast diagnostic to identify clearly irrelevant features (those with negative scores), reducing the feature set from 23 to perhaps 15–18. Second, apply **LMNN with n_components=8** on the reduced feature set to learn a discriminative 8-dimensional projection. Third, pre-transform all training data via x′ = Lx and build the hnswlib L2 index on 8-dimensional vectors. At query time, the overhead is a single 8×23 matrix multiply (microseconds) before hnswlib search.

For handling non-stationarity, retrain the full pipeline on a **rolling window** (e.g., trailing 2 years of daily data) at regular intervals—monthly for stable markets, weekly during volatile regimes. Track the LMNN transformation matrix L across retraining periods; if ‖L_new − L_old‖ exceeds a threshold, rebuild the hnswlib index. Otherwise, the existing index remains valid. With training times of 5–30 minutes for LMNN at 50K×23, weekly retraining is computationally trivial.

For validation, use **blocked time-series cross-validation** (not random CV, which would leak future information) to evaluate each feature selection method. Compare the full pipeline against baselines: raw 23-feature KNN, PCA-to-8 + KNN, and ReliefF-weighted KNN. The metric-learn library's scikit-learn compatibility makes this straightforward via `Pipeline` and custom cross-validation splitters.

## Conclusion

The five methods span a spectrum from simple filtering (ReliefF) through feature weighting to full metric learning (LMNN/NCA), with sequential selection as a model-agnostic wrapper. **LMNN emerges as the strongest choice** for this specific system—it is convex, scales to 50K samples in minutes, natively reduces dimensionality from 23 to 8, integrates cleanly with hnswlib via pre-transformation, and has a mature implementation in metric-learn. NCA is a viable alternative when sklearn ecosystem integration matters most, but requires subsampling or GPU acceleration at 50K scale. ReliefF serves best as a fast preprocessing step rather than a standalone solution due to its inability to detect redundancy. The wrapper approach (SFS/SBS) provides a useful validation baseline but cannot learn the continuous feature weights that metric learning offers. The combination of LMNN metric learning + hnswlib approximate search + rolling-window retraining appears to be a novel and promising architecture for financial KNN prediction that bridges a gap in the existing literature.