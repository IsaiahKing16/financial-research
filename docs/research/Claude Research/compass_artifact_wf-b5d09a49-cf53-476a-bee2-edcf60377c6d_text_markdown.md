# Adversarial robustness and failure modes of KNN-based financial prediction systems

A KNN-based equity prediction engine using hnswlib, walk-forward validation with BSS gating, Kelly sizing, and regime detection faces five compounding failure modes that can each individually destroy live performance. **The most dangerous interaction is between survivorship bias and KNN analogue matching**: a 52-ticker volume-selected universe guarantees every historical analogue is drawn from the winning 43% of stocks (Bessembinder, 2018), systematically underestimating tail risk. Backtest-to-live performance degradation studies show near-zero correlation between in-sample and out-of-sample Sharpe ratios across hundreds of strategies (Quantopian, 2016), and the Deflated Sharpe Ratio framework demonstrates that even a Sharpe of 2.0 in-sample can be consistent with zero true skill after correcting for multiple testing. This report covers five critical failure domains — data poisoning, distribution shift, deflated Sharpe ratio, survivorship bias, and look-ahead bias — with quantitative estimates, academic foundations, detection tools, and specific recommendations for the system architecture described.

---

## 1. Data poisoning corrupts KNN analogues through compounding adjustment errors

### How bad data enters the analogue database

Corporate actions are the primary poisoning vector for KNN feature databases. A stock split creates a discontinuity that, if unadjusted, propagates through every technical indicator computed across the event boundary. The magnitude is severe: eBay's unadjusted IPO price of $47.38 maps to an adjusted price of **$0.83** after accounting for splits, dividends, and the PayPal spinoff — a 57× difference. When adjustment factor chains break, every feature vector downstream is corrupted.

Yahoo Finance has been documented with multiple systematic errors. Their OHLC data is reported on a split-adjusted basis while dividends are reported unadjusted, creating compounding errors that grow larger going backwards in time (quantmod GitHub Issue #253). For CF Industries' 5:1 split, pre-split dividends were reported at $1.50/share instead of the correct $0.30 split-adjusted value. At one point, Yahoo swapped the "Close" and "Adj Close" columns entirely. Most critically for backtesting, Yahoo's adjusted close is no longer adjusted for dividends — only splits — making dividend-heavy securities effectively useless as data sources. As of 2025, historical data downloads have been restricted to premium subscribers, breaking the yfinance API that many quantitative systems depend on.

Polygon.io sources data directly from exchanges and provides both adjusted and unadjusted series, offering higher reliability. Alpha Vantage, despite being an official NASDAQ vendor, has been reported to show "suspicious gaps" in intraday data. Bloomberg uses its Corporate Action Coefficient methodology with adjustments applied at market close prior to ex-date, representing the institutional gold standard.

### Exchange glitches create persistent outlier analogues

Flash crashes inject extreme feature vectors into the analogue database. Johnson et al. (2013, *Scientific Reports*) detected **18,520 mini flash crashes between 2006 and 2011** — approximately **12 per trading day** — defined as price moves exceeding 0.8% in under 1.5 seconds. Major events like the May 6, 2010 Flash Crash (DJIA dropped 998 points in minutes, with Procter & Gamble briefly trading at one cent) and August 24, 2015 (ETFs diverging from NAV by double digits) leave permanent scars in historical databases.

The SEC's Clearly Erroneous Execution rules allow trade busts for deviations of **3–10%** from reference prices depending on price tier, but busted trades may or may not be removed from consolidated tape feeds depending on timing. For a KNN system, a flash crash feature vector — with extreme return magnitudes, volume spikes, and volatility readings — becomes a poisoned analogue. Because KNN uses distance metrics, this outlier can become the "nearest neighbor" to normal patterns if the distance calculation is dominated by the corrupted dimensions.

### HNSW amplifies data quality problems through graph structure

The hnswlib approximate nearest-neighbor search introduces a structural vulnerability. HNSW builds a navigational graph optimized for the training data's topology. In MATLAB benchmarks, **118 of 1,000 HNSW results differed from exact search** with default parameters — roughly a 12% miss rate. More critically, if a corrupted data point is well-connected in the HNSW graph (added early during construction when `ef_construction` explores more of the graph), it becomes a navigation hub visited during many queries. hnswlib has no built-in quality filtering during `add_items()`, and deleted elements leave structural artifacts in the graph.

Li, Wang, and Wang (2023, ISSTA) demonstrated that KNN data poisoning can have **indirect influence** — modifying training data changes the optimal K parameter, which in turn changes predictions for entirely unrelated test inputs. Jia et al. (2022, AAAI) showed that KNN's majority-vote mechanism provides some certified robustness, requiring an attacker to poison at least ⌊k/2⌋ + 1 neighbors to flip a prediction, but this assumes the adversary must target specific queries rather than degrading overall quality.

### Stale analogues from dead market regimes

Market microstructure has undergone irreversible structural breaks that render old analogues invalid. Decimalization in 2001 caused a **139% increase in trade count** and fundamentally changed spread dynamics. HFT grew from negligible to **61% of market volume** by 2009. The introduction of Limit Up-Limit Down circuit breakers after 2010 changed tail-risk dynamics entirely. Schmitt et al. (2013, arXiv:1304.5130) showed that sample-averaged observables like variances and correlations in financial time series are strongly dependent on the time window, confirming that pre-decimalization patterns have fundamentally different statistical properties from modern patterns.

### Detection and cleaning recommendations

**PyOD** (Zhao et al., 2019, JMLR; 50+ detection algorithms, 26M+ downloads) provides the most comprehensive outlier detection toolkit. For financial OHLCV data specifically, the Brownlees and Gallo (2006) method — comparing each observation against the mean and standard deviation of k neighboring observations with a granularity parameter — is well-suited. Practical best practices include: multi-source cross-validation of OHLCV from at least two vendors; OHLCV consistency checks (High ≥ max(Open, Close), Volume > 0); rolling z-score filtering with |z| > 5 thresholds on daily returns; maintaining an explicit flash crash calendar for exclusion; and converting to returns before feature engineering to eliminate level-dependent split artifacts. For the HNSW index, set `ef` ≥ `ef_construction` and monitor recall rate ≥ 0.9; rebuild the index periodically rather than relying on incremental updates.

---

## 2. Distribution shift makes KNN uniquely vulnerable to regime change

### The probability of backtest overfitting is alarmingly high

Bailey, Borwein, López de Prado, and Zhu (2014, *Journal of Computational Finance*; SSRN: 2326253) introduced a measure-theoretic framework for quantifying backtest overfitting. Their **Combinatorially Symmetric Cross-Validation (CSCV)** algorithm partitions the performance matrix into S equal submatrices, forms all C(S, S/2) combinations of in-sample/out-of-sample splits, and computes the **Probability of Backtest Overfitting (PBO)** as the proportion of combinations where the in-sample-optimal strategy underperforms the out-of-sample median.

The quantitative findings are sobering. On random walk data with seasonal optimization, **PBO = 55%**, with 53% of out-of-sample Sharpe ratios negative despite in-sample Sharpe ratios between 1.0 and 2.2. A genuine seasonal strategy showed PBO = 13%. The most damaging finding: the regression of out-of-sample performance on in-sample performance showed a **negative slope** — higher in-sample Sharpe ratios predicted *lower* out-of-sample Sharpe ratios, because "the model is so fit to past noise that it is rendered unfit for future signal." The authors conclude that no Sharpe ratio threshold or haircut can be considered universally safe.

### Backtest-to-live degradation is severe across all strategy types

A Quantopian study of 888 strategies found that in-sample Sharpe ratio had correlation with out-of-sample Sharpe of effectively **R² < 0.025** — near-zero predictive power. Strategies with more extensive backtesting showed *larger* performance gaps. A Stanford study (2025) found that **58% of retail algorithmic strategies collapse within 3 months** of live deployment. High-turnover strategies face an additional **30–50% return erosion** from execution friction alone.

Arian et al. (2024, *Knowledge-Based Systems*) directly compared walk-forward validation against Combinatorial Purged Cross-Validation (CPCV) using PBO and DSR diagnostics. **CPCV demonstrated marked superiority**, showing lower PBO and superior DSR test statistics. Walk-forward exhibited "notable shortcomings in false discovery prevention, characterized by increased temporal variability and weaker stationarity." For a 6-fold system, CPCV with N=6 groups and k=2 test groups yields C(6,2) = 15 splits and 5 distinct backtest paths — substantially more robust than the single historical path produced by walk-forward.

### KNN fails catastrophically under distribution shift

KNN is strictly interpolative — it has no mechanism to extrapolate beyond the training distribution, and no way to detect when it is doing so. When live data drifts to untrained regions, KNN simply returns the nearest training points regardless of distance, treating them as informative. In high-dimensional feature spaces, the curse of dimensionality compounds this: with d=100 dimensions and n=1,000 training points, finding 10 nearest neighbors requires covering **95% of the feature space** (ℓ ≈ (k/n)^(1/d) = 0.95). At these dimensions, the ratio of nearest-to-average distance approaches 1.0, meaning the "nearest" neighbor has little more predictive value than a random point.

A PMC study (PMID: 5070592) directly tested KNN under covariate shift and found its performance was inferior to both logistic regression and covariate-shift correction methods. Springer (2023) measured degradation rates exceeding **25%** in high-dimensional experiments under covariate shift. The HNSW graph structure compounds the problem: built on the training distribution's topology, the graph's navigational shortcuts route queries through neighborhoods that may not correspond to meaningful regions of the shifted distribution. The Ada-ef paper (arXiv:2512.06636) documented that HNSW suffers from "absence of recall guarantees and inefficient ANNS performance due to over- or under-searching" when data is non-uniform.

### Monitoring distribution shift in production

For KNN-specific drift detection, **track the distribution of distances to k-nearest neighbors over time**. Rising average distance signals that live data is moving to untrained regions. Compute the **Population Stability Index (PSI)** per feature on rolling windows — thresholds of PSI < 0.1 (stable), 0.1–0.25 (moderate shift), and > 0.25 (significant shift) provide actionable alerts. **Alibi-Detect** (Seldon, Apache 2.0) supports KS, MMD, and Chi-Square drift tests with streaming capability. **Evidently AI** provides 100+ built-in metrics with Grafana/Prometheus integration. For the HNSW index specifically, monitor whether the `ef` parameter required to achieve target recall is increasing — if so, the data distribution has shifted relative to the graph structure, and a rebuild is needed.

---

## 3. The Deflated Sharpe Ratio corrects for the multiple testing epidemic

### Mathematical formulation builds on three components

The DSR framework, developed by Bailey and López de Prado (2014, *Journal of Portfolio Management*, 40(5), 94–107; SSRN: 2460551), corrects the observed Sharpe ratio for both non-normal returns and selection bias from multiple testing. It builds on three mathematical components.

**Component 1 — The Probabilistic Sharpe Ratio (PSR):**

PSR(SR\*) = Φ\[(SR̂ − SR\*) / σ(SR̂)\]

where the standard deviation of the estimated Sharpe ratio, incorporating non-normality (Lo, 2002; Mertens, 2002; Opdyke, 2007), is:

σ(SR̂) = √\[(1 − γ₃·SR̂ + ((γ₄ − 1)/4)·SR̂²) / (T − 1)\]

Here Φ(·) is the standard normal CDF, SR̂ is the estimated (non-annualized) Sharpe ratio, SR\* is a benchmark threshold, γ₃ is return skewness, γ₄ is return kurtosis (Fisher=False, i.e., full kurtosis where normal = 3), and T is the number of observations. Under normality (γ₃=0, γ₄=3), this simplifies to Lo's formula, but real equity returns with negative skewness and fat tails substantially widen the confidence bands.

**Component 2 — The Expected Maximum Sharpe Ratio (False Strategy Theorem):**

Given N independent trials with Sharpe ratios drawn from N(0, V\[SR\]), the expected maximum SR under the null hypothesis of zero skill is:

SR₀ = √(V\[SR\]) · \[(1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))\]

where γ ≈ 0.5772 is the Euler-Mascheroni constant. This grows logarithmically with N through extreme value theory — even with zero true skill, testing more strategies produces increasingly impressive-looking Sharpe ratios.

**Component 3 — The DSR itself:**

DSR = PSR\[SR₀\] = Φ\[(SR̂ − SR₀)·√(T − 1) / √(1 − γ₃·SR₀ + ((γ₄−1)/4)·SR₀²)\]

**DSR ≥ 0.95** indicates strong evidence against noise (reject H₀ at 5%). DSR ≤ 0.50 means the observed performance is indistinguishable from luck. The DSR increases with larger observed SR, longer track records, and positive skewness; it decreases with more trials tested, fatter tails, and negative skewness.

### DSR outperforms classical multiple testing corrections

The **Bonferroni correction** divides α by N, controlling family-wise error rate but growing linearly with N — far too conservative for large trial counts. DSR's threshold grows only logarithmically through extreme value theory and additionally accounts for non-normality and sample length. The **Benjamini-Hochberg** procedure controls false discovery rate rather than family-wise error rate, but López de Prado and Fabozzi (2026, SSRN: 6450418) recently proved that FDR cannot be identified from in-sample statistics alone under search-and-selection, and standard FDR estimators necessarily underestimate the true FDR.

**Harvey, Liu, and Zhu (2016, *Review of Financial Studies*, 29(1), 5–68)** showed that after accounting for hundreds of factors tested since 1967, a new factor needs a **t-statistic exceeding 3.0** (not the traditional 2.0) to be credible. White's Reality Check (2000, *Econometrica*) and Hansen's Superior Predictive Ability test (2005, *JBES*) use bootstrap methods that are non-parametric but computationally expensive and require raw return data from all strategies, whereas DSR requires only summary statistics.

### Applying DSR to a 6-fold walk-forward system with BSS gating

A critical subtlety: **the 6 folds are NOT 6 independent trials for DSR purposes**. N is the number of distinct strategy configurations (hyperparameter combinations, feature sets, K values) evaluated during the research process. The folds are components of evaluating each single configuration. If 50 KNN configurations were tested across 6 folds, N = 50 (or N_effective estimated via clustering of strategy returns).

For BSS as a gate metric, the recommended approach is a two-stage validation. Use BSS as a pre-filter across walk-forward folds, then validate surviving strategies with DSR applied to their out-of-sample Sharpe ratios. Alternatively, with 6 partitions, compute PBO using CSCV: C(6,3) = 20 combinatorial splits, measuring what fraction of times the in-sample-best configuration underperforms the out-of-sample median. **Require PBO < 0.05 as the acceptance threshold.** Bailey et al. (2014) showed that with only 5 years of daily data, no more than approximately 45 independent configurations should be tested before one is virtually guaranteed to find an in-sample Sharpe of 1.0 with zero true expected performance.

### Python implementations are readily available

The most complete open-source implementation is **rubenbriones/Probabilistic-Sharpe-Ratio** (GitHub, 127 stars), providing `deflated_sharpe_ratio()`, `probabilistic_sharpe_ratio()`, `expected_maximum_sr()`, and `min_track_record_length()` functions. **esvhd/pypbo** implements PBO via CSCV with parallel computation support. **Hudson & Thames' mlfinlab** (commercial, ~£100/month) provides DSR, PSR, Sharpe Ratio Haircuts (Bonferroni, Holm, BHY), and CPCV, endorsed by López de Prado. A recent comprehensive framework by López de Prado, Lipton, and Zoonekynd (2025, SSRN: 5520741) at **github.com/zoonek/2025-sharpe-ratio** includes Bayesian FDR, FWER, non-normality corrections, and serial correlation adjustments.

---

## 4. Survivorship bias inflates a 52-ticker volume universe by 2–5% annually

### The foundational evidence is unambiguous

Elton, Gruber, and Blake (1996, *Review of Financial Studies*, 9(4), 1097–1120) estimated survivorship bias at **0.9% per annum** across U.S. mutual funds, defined as the difference in risk-adjusted returns between surviving funds and all funds. They found that funds disappear primarily due to poor performance, that bias is larger in the small-fund sector, and that bias **increases with sample period length**. Subsequent work expanded these estimates: Carhart et al. (2002) found bias ranging from **21 to 109 basis points** depending on methodology; Ibbotson and Chen estimated **2.74%** for hedge funds; Bianchi and Koutmos measured **2.1%** during the 2008 crisis specifically.

Shumway (1997, *Journal of Finance*, 52(1), 327–340) documented that correct delisting returns are unavailable for most stocks delisted for negative reasons in the CRSP database. Using OTC data, he showed missing returns are large and negative. Shumway and Warther (1999) estimated the corrected return for missing NASDAQ performance delistings at **−55%**. After correcting for this bias, there was no evidence that a size effect ever existed on NASDAQ. Boynton and Oppenheimer (2006) attributed **over 40%** of the commonly reported size premium to delisting and bid-ask spread bias.

### Bessembinder's findings are devastating for survivor-only KNN training

Bessembinder (2018, *Journal of Financial Economics*, 129(3), 440–457) showed that **4 out of 7 common stocks** (57%) in the CRSP database since 1926 have lifetime buy-and-hold returns less than one-month Treasuries. The best-performing **4% of listed companies** explain the net gain for the entire U.S. stock market. The median stock has a lifespan of only approximately 7 years before delisting. For a KNN system trained on 52 current high-volume tickers, this means every analogue is drawn from the winning minority. Monte Carlo simulations showed a **96% chance of underperforming** a value-weighted index with random stock selection from the full universe — the very stocks that would be missing from a survivorship-biased training set.

### A volume-selected universe creates triple bias

Selecting tickers by current volume creates three compounding biases: **survivorship bias** (only stocks that exist today are candidates), **success bias** (high-volume stocks are the most successful, as volume correlates with market cap and institutional interest), and **look-ahead bias** (current 2026 volume data is used to construct a universe for historical backtesting back to, say, 2015). Michael Harris tested cross-sectional momentum on the S&P 100 and found returns **dropped from 26% to 12.2% CAGR** when survivorship bias was removed. On the NASDAQ 100, returns fell from **46% to 16.4%**, and maximum drawdown increased from 41% to 83%.

### KNN analogue matching systematically misses failure patterns

When trained exclusively on survivors, every analogue in the KNN database comes from a stock that ultimately thrived. Even their distress periods resolved positively. The model never encounters patterns that preceded terminal decline, bankruptcy, or delisting. This creates what might be called the **"bright past" bias**: analogues are systematically skewed toward optimistic resolution scenarios. If a current stock shows a pattern similar to pre-bankruptcy Enron, the KNN model has no such analogue and will match to the nearest surviving pattern — one that had a fundamentally different outcome. Tail risk is underestimated because the left tail of the return distribution is missing entirely from the training data. Brown, Goetzmann, Ibbotson, and Ross found this can inflate **Sharpe ratios by as much as 0.5 points** and underestimate drawdowns by **14 percentage points**.

### Building a survivorship-free KNN training pipeline

The correct approach requires **point-in-time universe construction**: at each historical rebalancing date, select the top 52 stocks by trailing volume using only data available at that date, including stocks that later delist. **Norgate Data** (~$30–50/month) provides survivorship-bias-free data with 50,000+ listed and delisted U.S. securities plus historical index constituents. **CRSP** is the gold-standard academic database with delisting-adjusted returns back to 1926. **Zipline-Reloaded** (community-maintained fork of Quantopian's Zipline) natively supports dynamic universe construction with Norgate data integration. The KNN training universe should be expanded beyond the 52 trading tickers to include all historically-liquid stocks that met volume criteria at any point, with delisting-adjusted returns using the Shumway correction (−30% for NYSE/AMEX, −55% for NASDAQ performance delistings).

---

## 5. Look-ahead bias in the RegimeLabeler demands immediate audit

### Four channels of look-ahead in financial ML

Look-ahead bias enters financial ML pipelines through four primary channels: **feature engineering** (using `.mean()` instead of `.rolling()`, centered instead of trailing windows, full-sample normalization), **hyperparameter tuning** (using test-set information to select parameters without proper purged cross-validation), **universe selection** (selecting tickers based on future volume or performance), and **regime labeling** (using future information to classify current market state). A practitioner reported that correcting a single feature computation — from same-day close to properly lagged close — reduced a strategy's Sharpe from 1.5 to 0.8, a **47% inflation** from a single look-ahead error.

### The RegimeLabeler's SPY 90-day return is the highest-risk component

If the RegimeLabeler computes SPY 90-day return as `(price[t+90] − price[t]) / price[t]`, this is direct look-ahead bias using 90 days of future data. The correct implementation **must** use trailing return: `(price[t] − price[t−90]) / price[t−90]`. Even with trailing returns, the regime label computed at time t should only be applied to decisions at time t+1 or later (requiring a `.shift(1)` in the pipeline). VIX data is contemporaneous and generally safe, but VIX settlement values have specific publication times — using intraday VIX data that wasn't available at the decision timestamp creates bias. Yield curve data from FRED has **publication lag**: monthly macroeconomic indicators may not be published until weeks after the reference period.

### Hamilton filter versus Kim smoother is a critical architectural choice

For any HMM-based regime detection, the distinction between the **Hamilton filter** and **Kim smoother** is make-or-break. The Hamilton filter (Hamilton, 1989) computes filtered probabilities P(S_t | Y_1, ..., Y_t) using only information up to time t — it is causal and safe for live trading. The Kim smoother (Kim, 1994) computes smoothed probabilities P(S_t | Y_1, ..., Y_T) using the **entire dataset including future observations** — it explicitly contains look-ahead bias. In `statsmodels`, the `MarkovSwitching` class exposes both: `.filtered_marginal_probabilities` (safe) and `.smoothed_marginal_probabilities` (contaminated). The RegimeLabeler must use only Hamilton-filtered probabilities, and the model must be fit on an expanding or rolling window of strictly past data.

### Automated detection tools exist but require adaptation

**Freqtrade's `lookahead-analysis`** is the most directly applicable automated tool. It runs a baseline backtest, then reruns for each signal separately, comparing indicator values between full and sliced backtests. Any difference indicates look-ahead — detected empirically without examining source code. For the KNN system's pipeline, **scikit-learn's `permutation_importance`** provides a leakage diagnostic: if a feature has abnormally high importance AND removing it improves out-of-sample performance, it likely contains look-ahead. The **delay/perturbation test** is the simplest audit: add 1-day lag to all features and re-run. If performance collapses with a single additional lag, the strategy was exploiting same-bar information.

### The purging and embargo framework prevents temporal leakage

López de Prado (2018, *Advances in Financial Machine Learning*, Ch. 7) formalized **purging** (removing training observations whose label windows overlap with test labels) and **embargo** (removing a fixed percentage of training observations immediately following each test period to prevent leakage through autocorrelated features). For the KNN system, if predicting 90-day returns, the embargo should be ≥ 90 trading days. The **Combinatorial Purged Cross-Validation (CPCV)** extends this by generating C(N,k) train-test splits from N groups with k test groups, producing multiple backtest paths and enabling PBO computation. Implementations include `timeseriescv` (GitHub: sam31415/timeseriescv) and the open-source `mlfinpy` package.

### Audit checklist for the specific system

The RegimeLabeler audit should verify: (1) SPY 90-day return uses strictly trailing calculation with `.shift()` verification; (2) VIX input uses only data published before the decision timestamp; (3) yield curve data accounts for FRED publication lag; (4) regime labels are applied with at minimum a 1-day shift; (5) any HMM or clustering model is fit only on data up to time t, not on the full dataset; (6) walk-forward validation applies purging with embargo ≥ prediction horizon. The "too good to be true" heuristic provides a quick screen: unleveraged Sharpe > 1.5, annualized return > 12%, very smooth equity curve, or near-perfect accuracy each warrant investigation for look-ahead contamination.

---

## Conclusion: five biases compound multiplicatively

These five failure modes do not operate independently — they compound. Survivorship bias eliminates failure patterns from the KNN training set while look-ahead bias in the RegimeLabeler makes remaining analogues appear more predictive than they are. Distribution shift between backtest and live deployment degrades KNN performance catastrophically because the algorithm has no extrapolation capability, while data poisoning from vendor errors and flash crashes inserts extreme feature vectors that the HNSW graph may route through as navigation hubs. The Deflated Sharpe Ratio reveals that apparent walk-forward performance can be entirely attributable to multiple testing.

The highest-priority interventions are: **(1)** replace the current-volume universe selection with point-in-time construction using Norgate or CRSP data, expanding the KNN training set to include delisted stocks; **(2)** audit the RegimeLabeler for trailing-vs-forward SPY returns and Hamilton-filter-only HMM usage; **(3)** implement DSR with N equal to the total number of configurations tested, requiring DSR ≥ 0.95 for deployment; **(4)** upgrade from walk-forward to CPCV for overfitting assessment, computing PBO across C(6,3) = 20 combinatorial splits; and **(5)** monitor per-feature PSI and KNN neighbor distance distributions in production to detect distribution shift before it destroys live performance. The tools exist — PyOD for data quality, alibi-detect and Evidently for drift monitoring, pypbo for PBO, and the rubenbriones PSR library for DSR — but the architectural changes to eliminate survivorship and look-ahead bias must come first, as no amount of statistical correction can rescue a fundamentally contaminated training set.