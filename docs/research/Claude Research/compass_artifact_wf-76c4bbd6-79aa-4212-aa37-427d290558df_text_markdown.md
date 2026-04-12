# Open-source PBO tooling is sparse, and FPPE's gate structure is dangerously permissive

**Only one Python library directly computes the Probability of Backtest Overfitting (PBO), it hasn't been maintained since ~2018, and FPPE's current 3-of-6 majority-vote gate produces a family-wise false positive rate above 99% across its 7 hypotheses.** This report surveys every notable open-source implementation of PBO, the Deflated Sharpe Ratio (DSR), purged cross-validation, and multiple-testing corrections — then applies this toolkit to assess FPPE's statistical validity. The landscape reveals a striking gap: the theoretical framework is well-developed, but production-ready Python tooling remains fragmented. For FPPE specifically, switching from walk-forward to CPCV and replacing the BSS > 0 gate with proper p-values under Holm-Bonferroni correction would be the minimum viable fix.

---

## The PBO/CSCV ecosystem has exactly one Python library worth using

The CSCV method from Bailey, Borwein, López de Prado & Zhu (2014) partitions a matrix of strategy trial returns into S subsets, then enumerates all C(S, S/2) ways to split those subsets into in-sample and out-of-sample halves. For each combination, the best in-sample strategy is identified, and its out-of-sample rank is recorded. PBO is the proportion of combinations where the best in-sample strategy underperforms the median out-of-sample.

The critical distinction often missed in the literature: **CSCV ≠ CPCV**. Combinatorially Symmetric Cross-Validation computes PBO directly. Combinatorially Purged Cross-Validation (from AFML Chapter 12) provides purged train/test splits but does not compute PBO. Most Python libraries implement CPCV, not CSCV.

| Repository | PBO/CSCV? | Stars | License | PyPI | Python 3.12 | Maintained |
|---|---|---|---|---|---|---|
| `esvhd/pypbo` | ✅ Yes | 128 | AGPL-3.0 | ❌ | ⚠️ Likely with fixes | ❌ |
| quantresearch.org scripts | ✅ Yes | N/A | GPL (academic) | ❌ | ❌ (Python 2) | ❌ |
| `skfolio` CombinatorialPurgedCV | ❌ CPCV only | ~1,900 | BSD-3 | ✅ | ✅ Confirmed | ✅ Active |
| `sam31415/timeseriescv` | ❌ CPCV only | 284 | MIT | ✅ | ✅ Likely | ⚠️ Stable |
| `hudson-and-thames/mlfinlab` | ❌ CPCV only | 4,654 | Proprietary | ❌ Private | ⚠️ Unknown | Commercial only |
| `mrbcuda/pbo` (R) | ✅ Yes | N/A | CRAN | N/A | N/A | ✅ (v1.3.5, 2025) |

**`esvhd/pypbo`** is the only Python library implementing the full PBO computation. Its API is straightforward — `pbo.pbo(rtns_df, S=16, metric_func=metric, threshold=1, n_jobs=4, plot=True)` — and it also includes DSR, MinBTL, and performance degradation analysis. However, it pins `statsmodels==0.8.0`, has no `pyproject.toml`, and hasn't seen meaningful commits since ~2018. Running it on Python 3.12 requires manually upgrading dependencies. The most mature PBO implementation is actually the R `pbo` package on CRAN, which has no Python equivalent of comparable quality. The mlfinlab project has an open issue (#382, since June 2020) requesting PBO via CSCV — still unresolved.

---

## Deflated Sharpe Ratio implementations exist but aren't where you'd expect

The DSR corrects the Sharpe Ratio for two inflation sources: **selection bias from multiple testing** and **non-normal returns**. Mechanically, it computes the Probabilistic Sharpe Ratio (PSR) using the expected maximum SR under the null as the benchmark rather than zero:

**DSR = Φ((SR̂ − SR⁰) / σ(SR̂))**, where SR⁰ ≈ √(V[SR]) × [(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))], γ ≈ 0.5772 is the Euler-Mascheroni constant, and N is the number of independent trials. The SR standard error accounts for skewness and kurtosis: σ(SR) = √((1 + 0.5·SR² − skew·SR + (kurt−3)/4·SR²)/(T−1)). **A DSR above 0.95 indicates significance at the 5% level.**

Neither **quantstats** nor **empyrical** implements DSR. Both provide only the basic Sharpe ratio. The best standalone implementation is `rubenbriones/Probabilistic-Sharpe-Ratio` (127 stars, clean functional API with `deflated_sharpe_ratio()`, `probabilistic_sharpe_ratio()`, `min_track_record_length()`, and `expected_maximum_sr()`). The `esvhd/pypbo` library also bundles DSR alongside PBO. Stefan Jansen's `machine-learning-for-trading` repository includes a DSR script at `08_ml4t_workflow/01_multiple_testing/deflated_sharpe_ratio.py`, based on López de Prado's original code. In R, the `braverock/quantstrat` package provides `deflatedSharpe()`.

---

## Purged k-fold CV is well-served by skfolio and timeseriescv

The purge-and-embargo technique prevents information leakage in financial time-series cross-validation. **Purging** removes training observations whose label windows overlap with test set boundaries. **Embargo** adds a buffer period after each test set to account for serial correlation that purging alone might miss. The combinatorial extension (CPCV) generates C(N,k) train/test combinations from N groups with k test groups, producing φ(N,k) = k·C(N,k)/N distinct backtest paths.

**skfolio** is the clear winner for new projects. It provides `CombinatorialPurgedCV` with full sklearn integration, confirmed Python 3.10–3.13 support, **5,000+ unit tests**, and enterprise backing from Skfolio Labs. Its API is clean: `cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=2, purged_size=0, embargo_size=0)`. It includes an `optimal_folds_number()` utility and path-reconstruction visualization. The library also offers `WalkForward` and `MultipleRandomizedCV` splitters for comparison.

**`sam31415/timeseriescv`** is the best lightweight alternative — MIT-licensed, pip-installable, with `CombPurgedKFoldCV` and `PurgedWalkForwardCV`. Its `split()` method requires `pred_times` and `eval_times` arguments (pandas Series), adding moderate API complexity. For educational purposes, the `BlackArbsCEO/Adv_Fin_ML_Exercises` repository provides direct transcriptions of AFML Chapter 7 snippets. The open-source `mlfinpy` package reimplements mlfinlab's purged CV after Hudson & Thames went proprietary, but has limited community adoption.

---

## CPCV decisively outperforms 6-fold walk-forward in statistical power

Arian, Norouzi, and Seco (2024) provided the most rigorous comparison in "Backtest Overfitting in the Machine Learning Era" (Knowledge-Based Systems, Vol. 305). Using synthetic data generated from Heston, Merton jump-diffusion, and drift-burst models alongside historical S&P 500 data, they found **CPCV's marked superiority in mitigating overfitting risks**, with lower PBO and superior DSR test statistics versus walk-forward, standard k-fold, and purged k-fold.

The fundamental problem with walk-forward is path count. Six chronological folds produce **5 out-of-sample test periods** — a single historical path. This is a point estimate, not a distribution. A December 2025 arXiv paper ("Interpretable Hypothesis-Driven Trading," 2512.12924) demonstrated that even with 34 walk-forward folds on 10 years of data, statistical power was only **12%** at the observed effect sizes. Approximately **540 independent test periods** would be needed for 80% power. Six folds is an order of magnitude too few.

CPCV with the same 6 groups and k=3 test groups generates C(6,3) = 20 combinations and φ(6,3) = **10 backtest paths** — double the walk-forward count. With 10 groups and k=8, CPCV produces **36 paths** from 45 combinations. The key tradeoffs:

- **Walk-forward preserves strict chronological ordering** — no future data ever appears in training sets, making it more realistic for live trading simulation. CPCV partially relaxes this, relying on purging and embargo to prevent leakage.
- **CPCV produces a distribution** of performance metrics rather than a single estimate, enabling proper statistical inference (confidence intervals, hypothesis tests on the distribution of Sharpe ratios).
- **Computational cost scales combinatorially** — CPCV requires C(N,k) model fits versus walk-forward's N−1 fits.

**For FPPE, switching to CPCV is strongly recommended.** The current 6-fold walk-forward provides insufficient statistical power for any of its hypothesis tests. At minimum, FPPE should use CPCV with N≥10 groups to generate 30+ backtest paths. The `skfolio.model_selection.CombinatorialPurgedCV` class with `optimal_folds_number()` can determine the right configuration.

---

## MinBTL sets a hard floor on dataset requirements

The Minimum Backtest Length formula from Bailey et al. (2014, Notices of the AMS) quantifies how long a backtest must be to avoid selecting a strategy with spurious performance from N independent trials. It derives from the expected maximum Sharpe Ratio under the null:

**E[max_N] ≈ √(V[SR]) × [(1−γ)Φ⁻¹(1−1/N) + γΦ⁻¹(1−1/(Ne))]**

**MinBTL (years) ≈ (E[max_N])²**

The practical implications are stark. After just **7 independent strategy configurations, a researcher should expect to find at least one 2-year backtest with annualized SR > 1, even when the true SR is zero**. Five years of data supports roughly 45 independent configurations before a spurious SR=1 appears. For FPPE testing 7+ hypotheses, the dataset should span at least **(E[max_7])² ≈ 4–5 years** of data at minimum, assuming the hypotheses represent independent trials. If FPPE's hypotheses are correlated (likely, given shared data and features), the effective N is lower, relaxing the requirement somewhat — but the effective N should be estimated via PCA or correlation clustering, not assumed.

The `esvhd/pypbo` library includes MinBTL computation. The formula is also simple enough to implement directly in ~10 lines of Python using `scipy.stats.norm.ppf`.

---

## FPPE's 3-of-6 gate is statistically broken — here's how to fix it

This is the most critical finding. Under the null hypothesis of no predictive skill, BSS fluctuates around zero with approximately **50% probability** of being positive on any given fold (BSS > 0 simply means the model beat the reference, which a no-skill model does half the time by chance). Modeling each fold as a Bernoulli trial with p = 0.5:

**P(≥3 of 6 folds positive | null) = Σ C(6,k) × 0.5⁶ for k=3..6 = 42/64 = 65.6%**

This means each hypothesis has a **65.6% false positive rate** — worse than a coin flip as a filter. Across 7 independent hypotheses, the family-wise error rate is:

**FWER = 1 − (1 − 0.656)⁷ = 1 − 0.344⁷ ≈ 99.9%**

Even under a more favorable assumption of p = 0.3 per fold (the model is typically worse than the reference), FWER remains **87.4%**. The sequential gate structure provides essentially no protection against false discoveries.

The appropriate corrections, ranked by power:

- **Holm-Bonferroni** (minimum standard): Controls FWER, uniformly dominates Bonferroni, valid under any dependence structure, trivial to implement via `statsmodels.stats.multitest.multipletests(p_values, method='holm')`. With 7 tests, the most significant test requires p < 0.0071, the next p < 0.0083, scaling to p < 0.05 for the last.
- **Romano-Wolf stepdown** (gold standard): Captures correlation between hypotheses via bootstrap, yielding highest power while maintaining FWER control. Romano & Wolf (2005) specifically designed this for "formalized data snooping" in finance. No mature Python package exists — implementations are available in Stata (`rwolf2`) and R (`wildrwolf`), or must be coded manually.
- **Benjamini-Hochberg FDR**: Appropriate only if FPPE treats hypothesis testing as screening rather than deployment decisions. Controls the expected false discovery proportion rather than the probability of any false discovery.
- **Bonferroni**: Valid but unnecessarily conservative with Holm available.

To apply any correction, FPPE must first produce proper p-values. The fix: replace the BSS > 0 majority-vote with a **one-sample Wilcoxon signed-rank test** on the 6 per-fold BSS values (testing H₀: median BSS ≤ 0), yielding one p-value per hypothesis. Then apply Holm-Bonferroni across all 7 p-values. Harvey, Liu & Zhu (2016) further argue that in finance, t-statistics should exceed **3.0** (not 2.0) given extensive prior testing — corresponding to p < 0.0027 for a single test.

---

## Recent advances push beyond the original PBO framework

Three developments from 2024–2026 extend the Bailey et al. framework significantly:

**Bagged and Adaptive CPCV** (Arian, Norouzi & Seco, 2024, Knowledge-Based Systems). This paper introduced two CPCV variants: Bagged CPCV applies ensemble methods to cross-validation splits for reduced variance, while Adaptive CPCV dynamically adjusts the validation structure based on detected market regimes. Both outperformed standard CPCV in synthetic controlled environments combining Heston stochastic volatility, Merton jump-diffusion, and drift-burst models.

**The GT-Score** (Sheppert, 2026, Journal of Risk and Financial Management). This composite objective function embeds anti-overfitting principles directly into strategy optimization rather than correcting for overfitting post-hoc. It integrates performance, statistical significance, consistency, and downside risk into a single score, achieving **98% higher generalization ratios** in walk-forward validation compared to standard objectives. The GT-Score is complementary to PBO/DSR — it biases the search during optimization while PBO evaluates after.

**Statistical power quantification for walk-forward** (arXiv 2512.12924, December 2025). This paper provided the first rigorous power analysis for walk-forward validation in finance, finding that practical configurations achieve shockingly low power (~12% with 34 folds) and that hundreds of independent test periods are needed for reliable inference. This paper reinforces the case for CPCV.

The skfolio library also introduced `MultipleRandomizedCV`, a Monte Carlo-style approach sampling random asset subsets and time windows with inner walk-forward splits, based on Daniel Palomar's textbook methodology. This provides yet another alternative to pure walk-forward or CPCV for portfolio-level validation.

## Conclusion

The PBO tooling landscape is surprisingly thin: **`pypbo` is the only Python PBO implementation and it's abandoned**; DSR exists in two small repos but not in major quantitative libraries; purged CV is well-served by skfolio alone. FPPE faces three compounding problems: its 6-fold walk-forward generates too few paths for meaningful inference, its BSS > 0 majority-vote gate has a 65.6% per-hypothesis false positive rate, and it applies no multiple-testing correction across 7 hypotheses. The minimum fix is threefold: switch to CPCV via skfolio with ≥10 groups, replace the majority-vote gate with Wilcoxon signed-rank p-values, and apply Holm-Bonferroni correction. The gold standard would add Romano-Wolf stepdown and DSR computation via `pypbo` or a custom implementation, supplemented by MinBTL validation to confirm the dataset is long enough for 7+ trials.