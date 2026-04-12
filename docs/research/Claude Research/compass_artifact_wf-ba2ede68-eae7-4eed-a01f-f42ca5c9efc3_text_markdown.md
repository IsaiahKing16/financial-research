# Distance metrics for financial KNN: a practical verdict

**For KNN on point-in-time vectors of vol-normalized returns, Euclidean distance with a Mahalanobis whitening transform is the clear winner — and two of the five candidate metrics (DTW and EMD) are fundamentally inapplicable to this problem.** This finding eliminates the two most computationally expensive options and simplifies the engineering decision considerably. At the scale described (625K–6.25M vectors in 8–23 dimensions), brute-force exact search with FAISS takes **2–6 ms per query on CPU**, meaning approximate nearest neighbor indexing may not even be necessary. The critical insight is that DTW's temporal warping provides zero benefit when feature dimensions represent different sectors or factors at a single point in time, not sequential observations of the same variable.

---

## DTW and EMD are conceptually invalid for this problem

The most important finding is negative: **two of the five candidate metrics cannot meaningfully operate on point-in-time feature vectors**, and using them would produce nonsensical results.

**Dynamic Time Warping** solves a sequence alignment problem. It finds the optimal warping path between two temporal sequences, allowing local stretching and compression of the time axis (Sakoe & Chiba, 1978). When a feature vector is `[tech_ret, healthcare_ret, energy_ret, ..., momentum_factor, value_factor]`, these dimensions have **no sequential ordering**. DTW would attempt to "warp" technology returns to align with energy returns — a semantically meaningless operation. With a Sakoe-Chiba band of zero (no warping), DTW degenerates exactly to Euclidean distance; with warping permitted, it produces artificially lower distances by matching unrelated features, creating false similarity. DTW is only valid in this financial context if the problem is reformulated to compare **windows** of consecutive daily vectors (e.g., the last 20 days of cross-sectional snapshots), where the temporal axis within each window provides genuine sequential structure for alignment.

**Earth Mover's Distance (Wasserstein)** is defined between probability distributions, not between individual vectors. To apply EMD, each feature vector would need to be reinterpreted as a discrete distribution over "bins" — but vol-normalized returns are signed (violating non-negativity requirements for probability mass), the features have no natural ordering that defines transport cost, and forcing this interpretation discards the identity of each feature. EMD is well-suited for comparing the *distribution* of returns across a universe of stocks between two dates, as demonstrated by Horvath et al. (2021) for regime clustering, but not for comparing two d-dimensional factor-exposure snapshots.

This leaves three viable candidates: **Euclidean, Mahalanobis, and cosine similarity**.

## Mahalanobis adds genuine value over Euclidean for correlated z-scored features

A common misconception is that Mahalanobis distance collapses to Euclidean when features are z-scored. This is only true when features are **both** standardized **and** uncorrelated (Σ = I). Z-scoring sets the diagonal of the covariance matrix to unity but **preserves off-diagonal correlations**. For financial features — sector returns sharing common market factors, or value and momentum exhibiting negative correlation — the covariance matrix after z-scoring retains meaningful off-diagonal structure.

Euclidean distance implicitly double-weights correlated features. If tech and growth sector returns have ρ = 0.9, a deviation in both contributes nearly twice to the Euclidean distance despite carrying only marginally more information than a deviation in one. Mahalanobis corrects this by applying the inverse covariance matrix, effectively decorrelating features and counting each independent dimension of variation once. The mathematical equivalence is precise: **Mahalanobis distance equals Euclidean distance in the whitened space** obtained by the transform x′ = Σ^(−1/2)x, where Σ^(−1/2) is the inverse square root of the covariance matrix (obtainable via Cholesky decomposition).

The financial literature validates this. Kritzman & Li (2010) introduced the **Mahalanobis-based turbulence index** in the *Financial Analysts Journal*, defined as d_t = (r_t − μ)ᵀ Σ⁻¹ (r_t − μ), which captures both extreme price movements and unusual correlation patterns simultaneously. Kinlaw & Turkington (2013) decomposed this into volatility and correlation components, finding that the correlation surprise component is orthogonal to volatility and carries incremental predictive power. Stöckl & Hanke (2014) provided a comprehensive treatment in *Applied Economics and Finance*, demonstrating Mahalanobis distance's scale-invariance and applicability to elliptically distributed returns. However, Mahalanobis requires a **stable, invertible covariance matrix**. For d = 23 features, the covariance matrix has 276 unique parameters, and robust estimation (Ledoit-Wolf shrinkage) is essential, particularly given the non-stationarity of financial correlations.

After full PCA whitening (where Σ = I by construction), Mahalanobis and Euclidean become identical, and the distinction vanishes.

## Cosine similarity captures pattern shape but discards regime intensity

Cosine similarity measures angular distance between vectors, ignoring magnitude. On L2-normalized vectors, Euclidean and cosine produce identical KNN rankings (since ‖x̂ − ŷ‖² = 2 − 2cos(x̂, ŷ)). But **z-scoring is not L2-normalization**. Z-scored vectors have varying L2 norms across dates — days with extreme cross-sectional dispersion produce longer vectors than calm days. Cosine similarity discards this magnitude information entirely.

This distinction matters practically: cosine similarity finds days with the **same relative pattern** of sector returns (tech up, energy down, financials flat) regardless of whether z-scores were ±0.5 or ±3.0. Euclidean distance treats a mild version and an extreme version of the same pattern as distant. **The right choice depends on whether regime intensity carries predictive signal.** If cross-sectional dispersion magnitude matters for forward returns (plausible given volatility clustering), Euclidean or Mahalanobis is preferable. If only the directional pattern matters, cosine is appropriate.

## The curse of dimensionality at 8–23 dimensions is moderate but real

The landmark paper by Aggarwal, Hinneburg, & Keim (ICDT 2001) proved that the relative contrast (D_max − D_min)/D_min degrades as dimensionality increases, with the rate depending on the Lp norm used. At **d = 20, L1 provides better discrimination than L2 with >97% probability** in their experiments. The relative contrast for L1 scales as √(1/3) ≈ 0.577 versus √(1/5) ≈ 0.447 for L2 — a meaningful but not dramatic difference. Their experiments on real UCI datasets confirmed that nearest-neighbor classification accuracy degrades monotonically from L_0.1 through L_∞, with fractional norms (p < 1) providing the best discrimination.

However, Mirkes, Allohibi, & Gorban (Entropy, 2020) systematically tested fractional norms across 25 real datasets and found that **the practical kNN classification difference between p = 0.5, 1, and 2 is statistically insignificant** (Friedman + Nemenyi tests). Greater relative contrast does not automatically translate to better classification. The "best" norm is data-dependent, and the theoretical advantage of fractional norms is most pronounced when many dimensions contain pure noise — less likely with curated financial factor features.

For financial data specifically, the effective dimensionality is often lower than the ambient dimensionality due to **factor structure**. If 15 sector returns are driven by 3–5 latent factors, the intrinsic dimensionality is 3–5, and curse-of-dimensionality effects are less severe than the nominal d = 15 would suggest. The Hughes phenomenon (1968) suggests an optimal feature count typically in the **5–20 range** for KNN — exactly the 8–23 dimension range under consideration. Gu, Kelly & Xiu (2020, *Review of Financial Studies*) demonstrated that PCA-based factor models improve ML prediction performance, confirming that dimensionality reduction to the intrinsic factor space is beneficial.

## All three ANN libraries support the viable metrics natively or via transform

The engineering implementation is straightforward for the three valid metrics:

| Metric | hnswlib | FAISS | Annoy | Implementation |
|---|---|---|---|---|
| **Euclidean** | Native (`'l2'`) | Native (`METRIC_L2`) | Native (`'euclidean'`) | Direct use |
| **Cosine** | Native (`'cosine'`) | Via L2-normalize + `METRIC_INNER_PRODUCT` | Native (`'angular'`) | Normalize vectors to unit length |
| **Mahalanobis** | Via transform + `'l2'` | Via transform + `METRIC_L2` (documented in FAISS wiki with example notebook) | Via transform + `'euclidean'` | Pre-multiply all vectors by Σ^(−1/2), then use L2 |

The **whitening transform for Mahalanobis** is a one-time O(d³) computation for the matrix decomposition (trivial at d = 23) plus O(N·d²) to transform all vectors — roughly 1.6 billion FLOPs for 6.25M vectors at d = 16, completing in **0.5–2 seconds**. After transformation, all standard ANN indexes work at full speed with no performance penalty. FAISS documents this workflow explicitly in its `mahalanobis_to_L2.ipynb` example notebook.

Neither hnswlib, FAISS, nor Annoy support custom distance functions through their Python APIs. hnswlib allows custom distances in C++ only via the `SpaceInterface<T>` template. FAISS does not expose a custom distance API — its SIMD-optimized, compile-time-dispatched architecture deliberately trades flexibility for speed. Annoy's maintainer explicitly declined to add custom metric support (GitHub issue #553). This means DTW and EMD are not just conceptually inappropriate but also **engineering-incompatible** with the target ANN stack.

## Brute-force search is fast enough at this scale

At **6.25M vectors in 16 dimensions**, this is a small problem by modern ANN standards. FAISS brute-force (IndexFlat) with BLAS-optimized matrix multiplication delivers **~2–6 ms per single-threaded CPU query** and ~0.1–0.3 ms on GPU. Batching 1000 queries exploits cache efficiency and BLAS optimization, pushing amortized per-query cost well below 1 ms. For a daily rebalancing workflow processing 5000 tickers, total query time is on the order of **10–30 seconds** with brute force — entirely adequate.

If sub-millisecond latency is needed (streaming or real-time applications), HNSW indexes achieve **~20,000–50,000 queries per second** at 95%+ recall for d = 16, with build times of 5–15 seconds for 1M vectors. The recall-speed tradeoff is controlled by `ef_search` (hnswlib) or `nprobe` (FAISS IVF): at ef_search = 200, recall exceeds 99% with ~3× slower queries than optimal-speed settings. Memory overhead for HNSW is ~600–1000 bytes per vector (M = 16), adding ~4–6 GB for the largest dataset — manageable on any modern server.

For **financial backtesting** where reproducibility is paramount, exact brute-force search is arguably preferable to approximate methods, eliminating the recall parameter as a confound.

## What the financial literature recommends

The academic literature on distance metrics for financial KNN is surprisingly thin in top-tier finance journals. Most KNN applications in published work default to Euclidean distance without systematic comparison. However, several findings are instructive:

Ding, Trajcevski, Scheuermann, Wang & Keogh (2008, *PVLDB*) compared 9 similarity measures across 38 time-series datasets and found that **Euclidean distance is "very competitive with more complex approaches such as DTW when a database is relatively large."** Constrained DTW (with a Sakoe-Chiba band) often outperformed unconstrained DTW. A 2020 IEEE study comparing DTW vs. Euclidean hierarchical clustering on Indonesian stocks found DTW gave marginally higher silhouette scores but the difference was **"not significantly different from Euclidean-based"** — while being substantially slower.

Dolphin et al. (2021, arXiv:2107.03926) found that "traditional Euclidean and correlation-based approaches are not effective" for financial time series similarity and developed a hybrid metric — but their application was raw price series matching, not pre-normalized factor features. For the Wasserstein metric, Horvath et al. (2021) demonstrated that **Wasserstein k-means for regime clustering outperforms Euclidean k-means on return moments**, particularly for non-Gaussian returns. However, their application compared empirical return *distributions* across time windows, not individual feature vectors.

The practitioner consensus is clear: **most systematic trading firms default to Euclidean distance** for cross-sectional feature matching. DTW appears in the literature primarily for chart pattern recognition and analogue-based forecasting on raw price series (Tsinaslanidis & Kugiumtzis, 2014; Kinlay, 2016). Mahalanobis distance enters finance mainly through risk management and regime detection (the Kritzman-Li turbulence framework), not return prediction KNN systems.

## Practical recommendation matrix

For a KNN system on **point-in-time vectors of vol-normalized returns in 8–23 dimensions**:

- **Best default: Euclidean (L2) on Mahalanobis-whitened features.** Pre-transform all vectors using Σ^(−1/2) estimated with Ledoit-Wolf shrinkage, then use standard L2 in FAISS IndexFlat (brute force) or hnswlib HNSW. This captures correlation structure, is ANN-compatible, and adds negligible preprocessing cost. Update the whitening transform periodically (monthly or quarterly) to reflect evolving correlations.

- **Strong alternative: Manhattan (L1).** Aggarwal et al.'s theoretical results favor L1 over L2 in moderate dimensions, and L1 is supported natively in FAISS (IndexFlat and IndexHNSW). Worth A/B testing against whitened-L2.

- **Conditional use: Cosine similarity.** Appropriate only if cross-sectional dispersion magnitude is noise rather than signal. Easily testable — compare KNN forecast performance with and without L2-normalization of feature vectors before Euclidean search.

- **Do not use: DTW on single feature vectors.** Conceptually meaningless when feature dimensions are categorical (sectors, factors). If temporal pattern matching is desired, reformulate as window-based comparison with constrained DTW (Sakoe-Chiba band = 3–5), accepting ~400× computational overhead and brute-force-only search.

- **Do not use: EMD/Wasserstein on individual feature vectors.** Requires probability distributions, not signed real-valued vectors. Appropriate for comparing cross-sectional return distributions (e.g., histogram of all stock returns on date t vs. date s), not for the described point-vector KNN problem.

## Conclusion

The distance metric decision for financial KNN systems is simpler than it appears. The exotic options (DTW, EMD) are definitively excluded by a conceptual mismatch: point-in-time factor vectors are neither sequences nor distributions. Among the viable metrics, Mahalanobis-whitened Euclidean distance is theoretically optimal for correlated z-scored features and implementable at zero runtime cost through a linear pre-transform — a result that FAISS documents explicitly. The scale of the problem (6.25M vectors, d ≤ 23) is small enough that brute-force exact search is practical at single-digit millisecond latency, making the ANN library choice a convenience rather than a necessity. The marginal research effort is better spent on **feature engineering and covariance estimation** (shrinkage method, estimation window, update frequency) than on distance metric exotica.