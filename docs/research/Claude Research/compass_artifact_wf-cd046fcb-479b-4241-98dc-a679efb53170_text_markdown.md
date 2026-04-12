# Regime detection beyond thresholds: a practitioner's guide

**Hidden Markov Models, Bayesian changepoint detection, and PCA-based fragility measures each offer distinct advantages for equity regime labeling, but all share a single fatal risk: lookahead bias from smoothed probabilities or full-sample statistics.** The distinction between filtered (causal) and smoothed (non-causal) inference is the single most important implementation detail — getting it wrong renders any backtest meaningless. Across the five methods surveyed here, the evidence shows regime detection reliably reduces drawdowns and improves risk-adjusted returns, though it adds modest alpha at best. For a KNN-based system, regime probabilities work best as continuous features or model-selection filters rather than hard categorical labels.

---

## 1. Hidden Markov Models partition returns into discrete volatility states

Hamilton's (1989) regime-switching framework models equity returns as emissions from unobservable discrete states, each with its own Gaussian distribution. A 2-state model typically captures **low-volatility bull** (monthly μ ≈ +0.5–1%, σ ≈ 3–4%) and **high-volatility bear** (μ ≈ −1 to −2%, σ ≈ 7–10%) regimes. Hardy (2001) adapted this for equity-linked insurance pricing, finding BIC favored 2 states for TSE 300 and AIC preferred 3 for S&P 500. The regime-switching lognormal model demonstrated superior left-tail fit compared to GARCH — critical for risk management.

The EM algorithm (Baum-Welch) estimates parameters by alternating between a forward-backward E-step and parameter-update M-step. Convergence is guaranteed to a local maximum, typically in **20–200 iterations**. The algorithm is highly sensitive to initialization, making 10–50 random restarts essential for reliable estimation. Computational cost is **O(K²T)** per EM iteration, where K is the number of states and T the sequence length — trivially fast for financial applications. A 2-state model on 5,000 daily observations runs in under one second.

Three inference modes exist, and confusing them is the most common source of lookahead bias:

- **Viterbi decoding** finds the single most probable state sequence using the entire dataset. It is a full-sample method and introduces lookahead.
- **Smoothed probabilities** (forward-backward) compute P(s_t = j | x₁:T), conditioning on all observations including future ones. The backward pass explicitly uses data from t+1 through T. These are not safe for trading.
- **Filtered probabilities** (forward algorithm only) compute P(s_t = j | x₁:t), conditioning only on data available at time t. These are the only probabilities safe for real-time use.

In stable regimes, filtered and smoothed classifications agree roughly **85–92%** of the time. Near regime transitions, agreement drops to 50–70%, and filtered inference typically detects regime shifts **2–4 weeks later** than smoothed retrospective labeling. Nystrup et al. found that real-time inference produces approximately twice as many regime shifts as in-sample training, due to noisy filtered probabilities.

**Practical performance** is mixed but positive for risk management. Ang & Bekaert (2004) demonstrated out-of-sample outperformance in global equity allocation. Nystrup et al. (2015, 2018) showed Sharpe ratios above 2 for some instruments using "sticky" HMM variants. However, the LSEG study noted post-2022 instability due to shorter-duration volatile episodes. The consensus: HMMs reliably reduce drawdowns but generate modest alpha. Shu et al. (2024) showed that Statistical Jump Models outperform HMMs, particularly maintaining positive Sharpe ratios even with 10-day trading delays where HMMs degrade significantly.

**Known failure modes** include the label-switching problem (states arbitrarily reorder between fits — resolve by sorting states by variance), the geometric sojourn-time assumption (real regimes don't have memoryless durations — Hidden Semi-Markov Models address this), Gaussian emission misspecification (financial returns have fat tails — Student-t emissions help), and parameter instability as market structure evolves. For state selection, **2 states** is the robust default; 3 states capture a crisis regime but require proportionally more data; 4+ states risk overfitting. BIC is standard but theoretically questionable for HMMs.

**Minimum data**: a 2-state model needs ~200 daily observations (1 year); 3 states need ~500 (2 years). The rule of thumb is 30×K² minimum observations. Monthly frequency requires 5–10+ years for reliable estimation.

| Assessment dimension | Rating |
|---|---|
| Lookahead risk | **High** if using smoothed/Viterbi; **low** with filtered-only protocol |
| Computational cost | O(K²T) forward pass — sub-second for daily data |
| Minimum data | ~200 daily obs (2-state), ~500 (3-state) |
| Real-time capable | Yes, forward algorithm processes each new observation in O(K²) |
| Peer-reviewed validation | Extensive — Hamilton (1989), Hardy (2001), Ang & Bekaert (2004), Nystrup et al. (2015–2020) |

---

## 2. Python libraries differ sharply on filtered probability access

### hmmlearn (v0.3.3, October 2024)

The most widely used HMM library (~692K monthly downloads), but explicitly in **limited-maintenance mode** with no new releases in 18+ months. Built on scikit-learn with Cython-accelerated forward-backward and Viterbi passes.

The critical API detail: **`predict_proba()` returns smoothed posteriors**, not filtered probabilities. The source code runs the full forward-backward algorithm, conditioning on the entire observation sequence. This means calling `model.predict_proba(X)` on a backtest dataset introduces lookahead bias. Similarly, `predict()` uses Viterbi decoding on the full sequence.

To obtain filtered probabilities, you must implement the forward algorithm manually using the trained model's parameters (`model.startprob_`, `model.transmat_`, `model.means_`, `model.covars_`). The internal `_do_forward_pass()` method exists but is not part of the public API. No native support for streaming or online parameter updates exists — walk-forward requires repeated batch refitting.

```python
# WRONG for trading — returns smoothed (lookahead) probabilities
probs = model.predict_proba(X_full_history)

# CORRECT — manual forward-only pass for filtered probability at time t
def filtered_prob(model, X_up_to_t):
    """Forward algorithm only, returns P(state|data_1:t)."""
    log_startprob = np.log(model.startprob_)
    log_transmat = np.log(model.transmat_)
    framelogprob = model._compute_log_likelihood(X_up_to_t)
    fwd = np.zeros((len(X_up_to_t), model.n_components))
    fwd[0] = log_startprob + framelogprob[0]
    for t in range(1, len(X_up_to_t)):
        for j in range(model.n_components):
            fwd[t, j] = logsumexp(fwd[t-1] + log_transmat[:, j]) + framelogprob[t, j]
    log_filtered = fwd[-1] - logsumexp(fwd[-1])
    return np.exp(log_filtered)
```

### statsmodels (v0.14.6, December 2025)

The `tsa.regime_switching` module provides `MarkovRegression` and `MarkovAutoregression` with a crucial advantage: **it exposes both filtered and smoothed probabilities as distinct attributes**.

```python
res = sm.tsa.MarkovRegression(returns, k_regimes=2, 
                               switching_variance=True).fit()
# SAFE for trading — uses only data up to time t
filtered = res.filtered_marginal_probabilities
# NOT safe — uses full sample
smoothed = res.smoothed_marginal_probabilities
```

Estimation uses a hybrid EM + MLE approach (EM iterations for initialization, then BFGS optimization via the Hamilton filter). The module supports **time-varying transition probabilities** via `exog_tvtp`, allowing transitions to depend on observable indicators — a significant feature unavailable in hmmlearn. It provides standard errors, AIC/BIC, and expected regime durations. The limitation is that it handles only univariate endogenous variables, whereas hmmlearn supports multivariate observations natively.

### Head-to-head comparison

| Feature | hmmlearn 0.3.3 | statsmodels 0.14.6 |
|---|---|---|
| Filtered probabilities | Not exposed (manual implementation required) | `filtered_marginal_probabilities` |
| Multivariate observations | Native support | Univariate only |
| Switching regression | Not supported | Full support via `exog` |
| Time-varying transitions | Not supported | Supported via `exog_tvtp` |
| Statistical inference (SEs, p-values) | Not available | Full suite |
| Speed (simple 2-state) | Faster (Cython) | Moderate (MLE overhead) |
| Maintenance status | Limited/inactive | Active (~30M downloads/month) |

**Recommendation**: Use statsmodels `MarkovRegression` with `switching_variance=True` as the primary tool for univariate regime detection — it provides filtered probabilities, standard errors, and information criteria out of the box. Use hmmlearn only when multivariate features are needed, and implement a custom forward pass for filtered inference.

### Other libraries worth noting

**pomegranate v1.1.2** was rewritten from scratch on PyTorch, offering GPU acceleration (5–10× speedup) and flexible distribution support. Its `DenseHMM` class follows scikit-learn conventions, but `predict_proba()` returns forward-backward posteriors. The PyTorch dependency is heavy for a production trading system.

**PyMC 5.x** with `pymc-extras` enables fully Bayesian HMM estimation via MCMC, providing posterior distributions over all parameters. This naturally handles uncertainty quantification and can address label switching via ordered priors. However, sampling takes minutes to hours versus seconds for EM — impractical for frequent refitting in live trading.

**ruptures v1.1.10** (active, academically backed) provides offline changepoint detection via PELT, Binary Segmentation, and Dynamic Programming. It identifies where regime changes occurred but doesn't model regimes themselves. Complementary to, not a replacement for, HMMs.

**bayesian_changepoint_detection** (hildensia on GitHub) implements Adams & MacKay (2007) BOCD. It is truly online, processing one observation at a time while maintaining the run-length distribution. Meta's **Kats** library provides a production-grade BOCPD implementation with Normal and NormalGamma priors.

No major Python HMM library supports true online/streaming EM parameter updates. The standard pattern for live systems is expanding-window refitting at regular intervals (weekly or monthly) with cached models for inter-period forward-pass inference.

---

## 3. Online detection requires disciplined walk-forward protocols

The most reliable way to avoid lookahead bias is a strict walk-forward protocol. The protocol begins with an initial training period (minimum 2–5 years of daily data), fits the HMM on training data only, then for each subsequent day computes the filtered probability using only the forward algorithm with current parameters. At periodic intervals (weekly or monthly), the model is refit on an expanding or rolling window. After each refit, label-switching resolution ensures state identities remain consistent (ordering states by emission variance is the standard fix).

**Common mistakes that introduce lookahead** include using `smoothed_marginal_probabilities` instead of `filtered_marginal_probabilities`, fitting the HMM on the entire dataset and backtesting on the same data, applying Viterbi decoding on the full sample, normalizing features with full-sample statistics, and selecting the number of states K using BIC on the entire dataset rather than only training data.

**Expanding-window fitting** uses all data from start to time t, maximizing information but potentially including stale regime dynamics. **Rolling-window fitting** (typically 500–1,260 daily observations) adapts to changing markets but suffers from higher parameter variance and boundary effects — Bulla et al. (2011) showed the first and last observations in each window have the highest estimation error. Nystrup et al. (2018) proposed **exponential forgetting** as a compromise, down-weighting older observations without discarding them.

### Bayesian Online Changepoint Detection offers a fundamentally different approach

Adams & MacKay's (2007) algorithm maintains a posterior distribution over the "run length" — time since the last changepoint. At each new observation, it updates this distribution using only the current observation, making it **inherently causal with zero lookahead risk**. The key recursion computes growth probability (run continues) and changepoint probability (run resets to zero). When posterior mass concentrates on short run lengths, a changepoint is detected.

The hazard rate parameter λ controls sensitivity: **λ = 100–250** works well for daily equity data. Detection latency is inherent at **5–15 trading days**, as sufficient posterior evidence must accumulate. Computational cost is **O(T)** per step with run-length truncation, making it suitable for real-time streaming.

The primary limitation is that BOCPD detects when parameters change but doesn't classify the regime type. It assumes i.i.d. observations within regimes, which is violated by volatility clustering. Recent extensions (Tsaknaki et al., 2024) add autoregressive structure and time-varying within-regime parameters. An effective hybrid strategy combines BOCPD for detecting when to refit an HMM, with the HMM providing probabilistic regime labels.

| Method | Lookahead risk | Latency | Labels regimes? |
|---|---|---|---|
| Sequential HMM (forward only) | Low if protocol followed | 2–4 weeks at transitions | Yes (probabilistic) |
| Expanding-window refit | Low | Depends on refit frequency | Yes |
| Rolling-window refit | Low | Better adaptation | Yes |
| BOCPD (Adams & MacKay) | None by design | 5–15 days | No (changepoint only) |
| BOCPD + HMM hybrid | Low | 5–15 days | Yes |

---

## 4. The turbulence index captures statistical unusualness in real time

Kritzman & Li's (2010) turbulence index computes the **Mahalanobis distance** of current multi-asset returns relative to their historical distribution:

d_t = (r_t − μ)′ Σ⁻¹ (r_t − μ)

Under multivariate normality, this follows a chi-squared distribution with n degrees of freedom (n = number of assets). In practice, fat tails mean theoretical percentiles don't match empirical ones — empirical percentile thresholds are more robust.

**A critical implementation warning**: the original Kritzman & Li paper used **full-sample** mean and covariance, which introduces lookahead bias. The Portfolio Optimizer blog explicitly flags this, recommending rolling-window estimation instead. The correct real-time computation uses μ and Σ estimated from a trailing window only (typically **252 trading days**), comparing the current return vector against these historical estimates. Window lengths from 60 to 252+ days produce similar general patterns, though shorter windows are noisier and longer windows lag.

The standard asset universe is **6–10 major asset classes** or sector indices, though Baitinger & Flegel (2021) tested 10–30 equity indices. More assets capture richer cross-correlation structure but require larger estimation windows and shrinkage estimators (Ledoit-Wolf) when the T/n ratio falls below 5. Computational cost is **O(n³)** for the covariance inversion, trivially fast for n < 50 — under 1ms for 10 assets. Pre-computing the Cholesky decomposition (Σ = LL′) and solving L⁻¹(r_t − μ) avoids explicit matrix inversion.

For regime classification, the standard approach uses **percentile-based thresholds**: the 75th percentile of the turbulence distribution separates "quiet" from "turbulent" markets. Kritzman et al. (2012) advocate feeding the turbulence time series into a 2-state HMM for probabilistic classification rather than using fixed thresholds — this combined approach outperformed static asset allocation.

Kinlaw & Turkington (2013) decomposed turbulence into **correlation surprise** (unusual cross-asset relationships) and **magnitude surprise** (unusually large moves), providing richer regime information. Validation spans U.S. equities, global equities, FX, housing markets, and hedge funds. Salisu et al. (2022) confirmed predictive power across 7 economies using GARCH-MIDAS. The COVID-19 crash provided strong live validation — Windham Capital's turbulence-based index triggered warnings within days of the February 2020 market peak.

The turbulence index is reactive rather than predictive — it detects unusual conditions as they occur, not before. Elevated readings tend to persist for approximately 2 weeks. False positives occur: high turbulence indicates fragility, not guaranteed losses.

---

## 5. Absorption ratio measures systemic fragility before it materializes

The absorption ratio captures the fraction of total asset variance explained by the top eigenvectors from PCA:

AR = Σᵢ₌₁ⁿ σ²(Eᵢ) / Σⱼ₌₁ᴺ σ²(Aⱼ)

where n = N/5 eigenvectors (rounded) and N = total assets. High absorption (>0.8) indicates risk sources are tightly coupled — the market is fragile. Low absorption (<0.6) indicates diversified, resilient risk. The original paper used **51 MSCI USA industry portfolios** with a **500-trading-day rolling window** and exponential weighting (half-life 250 days).

The most actionable signal is the **standardized absorption ratio shift** (ΔAR):

ΔAR = (AR₁₅ₐᵧ − AR₁ᵧₑₐᵣ) / σ(AR₁ᵧₑₐᵣ)

Trading rules use ΔAR > +1σ to reduce equity exposure and ΔAR < −1σ to increase it. In the original paper's backtest (1998–2010), this dynamic strategy earned **9.58% return** versus 5.08% for a static 50/50 portfolio, with only 1.72 average trades per year. The result that **100% of the worst 1% of monthly drawdowns** were preceded by a 1σ ΔAR spike is striking — absorption spikes are a near-necessary condition for significant drawdowns, though not a sufficient one. Many spikes do not lead to crashes, creating a meaningful false positive rate.

The absorption ratio leads the turbulence index by approximately one month, making it more valuable as an early warning system. During COVID-19, Windham Capital's combined index hit 48 on February 19 (market peak), rose to 76 by February 25, and reached 99 by February 27 — providing actionable signal within days.

**Data requirements** are substantial: 500 trading days (~2 years) for initial covariance estimation, plus 252 days for ΔAR standardization, totaling approximately **3 years before the first signal**. The number of eigenvalues to retain is debated — the original N/5 rule is arbitrary, and some researchers (Zheng et al., 2012) advocate using only the first eigenvector. Computational cost is **O(N³)** per eigendecomposition, trivial for N < 100 but requiring randomized PCA for larger universes.

Rolling-window PCA is causal by construction when using only trailing data. The only lookahead risk comes from standardizing ΔAR with full-sample statistics — the correct approach uses expanding or rolling windows for the σ computation. The original paper explicitly reserves 3 years as a burn-in period before generating signals.

| Dimension | Turbulence index | Absorption ratio |
|---|---|---|
| What it detects | Current unusual behavior | Structural fragility buildup |
| Signal timing | Contemporaneous / reactive | Leading (by ~1 month) |
| Best application | Real-time risk scaling, stop-loss triggers | Strategic allocation, early warning |
| Assets needed | 5–10 asset classes | 10–50+ industries or sectors |
| History needed | 252 days (1 year) | 750+ days (3 years) for ΔAR |
| Computation | O(n³), trivially fast | O(N³), fast for N < 100 |

---

## 6. Integration with KNN systems and choosing the right number of regimes

Three approaches exist for incorporating regime information into a KNN-based prediction system. **Using regime probabilities as continuous features** preserves uncertainty and avoids the information loss of hard labels — the filtered probability P(regime_k | data₁:t) for each state serves as a feature vector. **Training separate KNN models per regime** and blending predictions using regime probabilities outperforms hard-switching in practice. **Simple categorical regime labels** are the least informative option. The most effective approach combines the second and third: regime-specific models weighted by filtered probabilities.

Additional engineered features from regime information include regime duration (time since last transition), the rate of change of regime probabilities, and the transition matrix entries themselves. All must use only filtered probabilities — smoothed probabilities contaminate every downstream feature.

For cross-validation with regime features, Lopez de Prado's **Combinatorial Purged Cross-Validation** (CPCV) is the gold standard. Standard k-fold cross-validation fails because regime labels from HMM inference leak information across folds. A purge gap around each fold boundary prevents this leakage.

On the number of regimes: **2 states** (bull/bear or low/high volatility) is the robust default, sufficient for most trading applications, and the choice of most practitioners. **3 states** add a crisis or transition regime and show the biggest log-likelihood improvement over 2 states, but require significantly more data. **4+ states** are rarely justified — studies showing exceptional returns from 4-state models often exhibit overfitting artifacts, and BIC is theoretically questionable for HMM state selection (Gassiat & Rousseau, 2014). For a KNN system, start with 2 states. If the crisis regime is economically meaningful for your universe, test 3 states on a held-out validation period and verify that the third state adds value after transaction costs.

## Conclusion

The five methods form a complementary toolkit rather than competing alternatives. The absorption ratio provides the earliest warning of structural fragility (weeks ahead), the turbulence index captures real-time unusual conditions, and HMMs assign probabilistic regime labels that encode both methods' information. BOCPD serves as a trigger for model refitting rather than a standalone regime classifier. For a production KNN system, the recommended pipeline is: compute turbulence and absorption ratio as features, feed them alongside returns into a 2-state HMM, extract filtered (never smoothed) probabilities via the forward algorithm, and use these as continuous features or model-selection weights. The entire pipeline runs in milliseconds on daily data. The primary risk is not computational but methodological — any use of smoothed probabilities, Viterbi decoding, or full-sample statistics in backtesting will produce illusory performance that vanishes in live trading.