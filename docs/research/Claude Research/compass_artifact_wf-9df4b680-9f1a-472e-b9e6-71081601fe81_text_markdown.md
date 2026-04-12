# Position sizing beyond Kelly: a framework for uncertain edges

**When your edge estimate carries ±5% error, the optimal Kelly fraction drops to roughly 75–80% of full Kelly** — a result derivable from Baker and McHale's (2013) shrinkage formula α* = edge²/(edge² + σ²). This finding anchors a broader landscape of position sizing methods that systematically account for what pure Kelly ignores: parameter uncertainty, asset correlations, drawdown constraints, and the distributional richness of real trading outcomes. For the FPPE system using KNN-derived probabilistic signals evaluated via Brier Skill Score, fractional Kelly emerges as the dominant practical framework, but several complementary and alternative approaches offer important structural advantages depending on the system's risk tolerance and portfolio complexity.

The six methods surveyed below form a coherent toolkit. Fractional Kelly and Bayesian Kelly address the same core problem — estimation uncertainty — from frequentist and Bayesian angles respectively. Multi-asset Kelly and risk parity tackle portfolio construction, with risk parity offering robustness when return estimates are noisy. Drawdown constraints impose hard floors that map directly to Kelly fractions. And Optimal-f extends Kelly to non-binary outcomes, albeit with serious practical limitations.

---

## Fractional Kelly sacrifices little growth for dramatically lower risk

The Kelly criterion, formulated by Kelly (1956) and rigorously proved by Breiman (1961), prescribes betting the fraction f* = (bp − q)/b for a binary wager with win probability p, loss probability q, and payoff odds b. For continuous lognormal assets, Thorp (2006) showed this becomes **f* = μ/σ²**, where μ is excess return and σ² is return variance. The Kelly fraction uniquely maximizes the expected logarithmic growth rate g(f) = p·log(1 + bf) + q·log(1 − f), and Breiman proved it asymptotically minimizes the expected time to reach any wealth target.

The case for betting *less* than Kelly rests on a remarkably elegant result. Thorp (2006, §7) demonstrated that betting fraction c of the Kelly amount achieves a growth rate proportional to **c(2 − c)** of the full Kelly rate. At half-Kelly (c = 0.5), this yields exactly 75% of full-Kelly growth — while variance scales as c², meaning **half-Kelly cuts variance to 25% of full Kelly**. The growth-to-variance ratio improves threefold. MacLean and Ziemba (1999) computed the probability of doubling before halving capital: **67% at full Kelly versus 89% at half-Kelly**. Perhaps most striking, twice Kelly (c = 2) produces zero expected growth, since 2(2 − 2) = 0. Betting more than Kelly is always dominated — it delivers both lower growth and higher risk.

MacLean, Ziemba, and Blazenko (1992) in *Management Science* established the utility-theoretic foundation: fractional Kelly strategies correspond to negative power utility functions U(w) = δw^δ. Half-Kelly maps to δ = −1 (reciprocal utility), quarter-Kelly to δ = −3, and full Kelly to δ → 0 (logarithmic utility). This equivalence means fractional Kelly is not an ad hoc adjustment but the optimal strategy for investors with finite risk aversion. The authors showed these strategies trace out a complete growth–security tradeoff frontier, parametrized by the Kelly fraction.

The theoretical argument for fractional Kelly becomes decisive under parameter uncertainty. Baker and McHale (2013) in *Decision Analysis* proved that **full Kelly is never optimal when there is any estimation uncertainty in p**, regardless of the uncertainty's distribution. The mechanism is the asymmetric concavity of the growth function: the penalty for overbetting (f > f*) far exceeds the cost of underbetting (f < f*). When f exceeds 2f*, growth turns negative and ruin becomes certain. When the estimate p̂ fluctuates around true p, this asymmetry means the expected growth at full Kelly is strictly less than the growth at some k < 1 times Kelly. Baker and McHale's approximate shrinkage factor, derived via second-order Taylor expansion, is:

**α\* = edge² / (edge² + σ²)**

where edge = p̂ − q is the estimated advantage and σ is the standard deviation of the estimation error. Equivalently, α* = SR²/(SR² + 1), where SR = edge/σ is a signal-to-noise ratio. When estimation uncertainty equals the edge (SR = 1), this yields **half-Kelly**. When σ is small relative to the edge, α* approaches 1.

---

## Multi-asset Kelly is theoretically clean but practically fragile

Extending Kelly to portfolios of correlated assets yields the multivariate formula **F\* = Σ⁻¹μ**, where Σ is the return covariance matrix and μ is the vector of expected excess returns. This result follows from maximizing the expected log-growth rate G(f) = f^Tμ − ½f^TΣf, whose first-order condition is Σf = μ. The maximum growth rate equals r + ½μ^TΣ⁻¹μ — the risk-free rate plus half the squared maximum portfolio Sharpe ratio.

Luenberger (1998, §15.5 of *Investment Science*) and Luenberger (1993) in the *Journal of Economic Dynamics and Control* established the connection to mean-variance optimization: **the Kelly portfolio is identical to the Markowitz tangency portfolio** in the continuous-time limit under geometric Brownian motion. Fractional Kelly with parameter κ ≥ 1 (investing 1/κ of the Kelly allocation) corresponds to a Markowitz portfolio with proportionally higher risk aversion. The Kelly portfolio sits on the efficient frontier at the most aggressive point — maximum return per unit of risk, but also maximum risk.

The practical problems are severe. Chopra and Ziemba (1993) in the *Journal of Portfolio Management* demonstrated that **errors in expected return estimates are over 10 times as damaging as errors in variance estimates and over 20 times as damaging as errors in covariance estimates** for optimal portfolio weights. Since the Kelly portfolio corresponds to the lowest risk aversion on the efficient frontier, it is maximally sensitive to mean estimation errors. The covariance matrix inversion amplifies these errors: near-singular matrices produce extreme position sizes, and when the number of assets exceeds the number of observations, Σ is guaranteed singular. Low-correlation assets receive disproportionately large weights, creating dangerous concentration.

Remedies include covariance shrinkage via the Ledoit-Wolf estimator, Bayesian updating following Browne and Whitt (1996), and the Multivariate Volatility Regulated Kelly (MVRK) approach. Carta and Conversano (2020) in *Frontiers in Applied Mathematics and Statistics* found that rolling Kelly portfolios outperform Markowitz tangency portfolios with two-year estimation windows, but remain "more risky and less diversified." For the FPPE system, the multivariate Kelly framework applies when extending from single-signal position sizing to multi-asset allocation, but the covariance estimation challenge argues strongly for either fractional Kelly or alternative approaches.

---

## Risk parity sidesteps the estimation problem Kelly cannot solve

Risk parity, formalized by Qian (2005) and rigorously analyzed by Maillard, Roncalli, and Teiletche (2010), constructs portfolios where each asset contributes equally to total portfolio risk. The equal risk contribution (ERC) portfolio satisfies xᵢ(Σx)ᵢ = xⱼ(Σx)ⱼ for all asset pairs i, j, where xᵢ is the weight and (Σx)ᵢ is the i-th element of the covariance-weighted position vector. Maillard et al. proved existence and uniqueness when Σ is positive definite, and showed that **ERC portfolio volatility lies between the minimum-variance and equal-weight portfolios**.

The simplest variant — inverse-volatility weighting, where wᵢ = (1/σᵢ)/Σⱼ(1/σⱼ) — coincides with the ERC portfolio when all pairwise correlations are equal. It requires only volatility estimates, no return forecasts and no matrix inversion. This structural advantage is decisive: as Rob Carver notes, regressions of monthly volatility on prior-month volatility produce R² ≈ 0.6, compared to **R² ≈ 0.01 for means and Sharpe ratios**. Risk parity exploits the most predictable parameter in finance while Kelly depends on the least predictable one.

Qian (2005) demonstrated that a traditional 60/40 stock/bond portfolio derives roughly **90% of its risk from equities**, making it far less diversified than it appears. Over 32 months where the 60/40 portfolio lost more than 3%, equities contributed 97% of the loss on average. Leveraged risk parity (at approximately 1.8:1) outperformed 60/40 by close to 5% per year with better Sharpe ratios in Qian's analysis. Qian (2006) in the *Journal of Investment Management* proved that risk contributions have genuine financial significance beyond ex-ante decomposition — they predict ex-post loss contributions, especially for large-magnitude drawdowns.

The comparison between Kelly and risk parity reduces to a question about what you know. Kelly is optimal when return estimates are accurate; risk parity is optimal when Sharpe ratios are approximately equal across assets (empirically plausible — US stocks, Treasuries, and commodities all show Sharpe ratios near 0.25–0.30 over long horizons) and return estimation is noisy. Asness, Frazzini, and Pedersen (2012) in the *Financial Analysts Journal* found risk parity achieved a gross Sharpe ratio of **0.75 versus 0.52 for 60/40** over 1947–2015. For the FPPE system, risk parity offers a robust portfolio-level allocation framework that complements signal-level Kelly sizing: use fractional Kelly to size individual positions based on signal confidence, then apply risk parity principles across strategy components or asset classes.

---

## Drawdown constraints create a direct bridge from risk tolerance to Kelly fraction

Grossman and Zhou (1993) in *Mathematical Finance* solved the problem of maximizing long-term growth subject to the constraint that wealth W_t never falls below fraction α of its running maximum M_t. The maximum permissible drawdown is D = 1 − α. Their central result: the optimal strategy invests in the risky asset proportionally to the "cushion" W_t − αM_t, resembling Constant Proportion Portfolio Insurance (CPPI) but with a stochastic floor.

The key formula for practitioners is strikingly simple. At the high-water mark (W_t = M_t), the constrained optimal fraction equals:

**f_constrained = D × f_Kelly**

A 20% maximum drawdown constraint (D = 0.20) implies investing only 20% of the unconstrained Kelly fraction. **Half-Kelly implicitly targets a 50% maximum drawdown; quarter-Kelly targets 25%.** This provides a direct mapping from risk tolerance to position size. Cvitanić and Karatzas (1995) extended this result to multiple assets using martingale methods and proved an elegant equivalence: the drawdown-constrained problem with utility U is identical to an unconstrained problem with a modified utility function where the CRRA parameter γ is replaced by γ(1 − α).

Busseti, Ryu, and Boyd (2016) at Stanford developed Risk-Constrained Kelly (RCK), which formally bounds the probability of breaching a drawdown threshold. Their constraint, E[(r^Tb)^{−λ}] ≤ 1, where λ = log β / log α combines the drawdown floor α and the breach probability β, produces **higher growth rates than fractional Kelly for the same drawdown risk**. In their numerical example, for 10% drawdown risk, RCK achieved growth of ~0.047 versus fractional Kelly's ~0.035. Hsieh and Barmish (2017) contributed a sobering perspective: even with p = 0.99, full Kelly over 252 trading days produces a **92% probability of maximum drawdown exceeding 98%**. This stark statistic alone justifies fractional Kelly for any finite-horizon system.

---

## Optimal-f generalizes Kelly to continuous distributions but at a steep practical cost

Ralph Vince's Optimal-f framework, introduced in *Portfolio Management Formulas* (1990), maximizes the Terminal Wealth Relative (TWR) = ∏ᵢ(1 + f · Trade_i / |BiggestLoss|) over a historical trade sequence. The fraction f ∈ (0, 1] is expressed as a divisor of the largest historical loss, and the optimal f* is found by numerical search. Position size then follows: Units = (Equity × f*) / |BiggestLoss|.

Optimal-f and Kelly coincide exactly for binary outcomes with uniform loss sizes — the Bernoulli special case. For continuous or multi-valued return distributions, they diverge. Vince argued this makes Optimal-f the more general solution, as Kelly's closed-form formula "could not give us the correct optimal f" when "wins are for varying amounts and losses are for varying amounts." Maier-Paape (2016) and Hermes and Maier-Paape (2017) in *Risks* (MDPI) established formal existence and uniqueness results for optimal-f under general distributions, proving the TWR optimization is concave after log transformation.

The criticisms of Optimal-f are substantial and largely justified. First, **f\* depends critically on the largest historical loss**, which is always underestimated relative to future possibilities. Any loss exceeding the historical worst causes ruin at optimal-f. Second, the method produces severe drawdowns — typically 50–95% of equity on the path to terminal wealth maximization, making it psychologically and practically intolerable. Third, it is inherently backward-looking with no mechanism for adapting to regime changes or non-stationarity. Fourth, it over-leverages: in practice, Vince's examples suggest risking 23–34% of capital per trade, far beyond conventional risk management limits.

For the FPPE system, Optimal-f is best understood as a theoretical complement to Kelly rather than a practical replacement. Its value lies in handling the non-binary nature of real trading returns. But the system would be better served by the continuous-time Kelly formula f* = μ/σ² applied to the actual return distribution, with fractional adjustment for estimation uncertainty, rather than Optimal-f's extreme leverage suggestions.

---

## Bayesian Kelly integrates directly over posterior uncertainty in the edge

The Bayesian approach to Kelly criterion, pioneered by Browne and Whitt (1996) in *Advances in Applied Probability*, resolves the estimation uncertainty problem from first principles. When the true probability θ is unknown with prior distribution π(θ), the optimal strategy maximizing E[log W_n] is a state-dependent control: **bet a fraction equal to the posterior mean of the edge**. Formally, f*_n = E[2θ − 1 | Z₁,...,Z_{n−1}] = 2·E[θ | history] − 1. For a Beta(α,β) prior, the posterior after k wins in n trials gives f*_n = (2(α + k) − (α + β + n))/(α + β + n). This strategy automatically incorporates shrinkage: the posterior mean pulls the edge estimate toward the prior, producing smaller bets than naive plug-in Kelly.

Chu, Wu, and Swartz (2018) in the *Journal of Quantitative Analysis in Sports* developed a full Bayesian decision-theoretic framework yielding multiple modified Kelly estimators. Under their "natural loss function" — the ratio of optimal-Kelly growth to actual growth — the Bayes estimator simplifies to f₀ = (p̂θ − 1)/(θ − 1), where p̂ is the posterior mean probability. In their worked example with 100 wins in 180 trials and a Beta(50,50) prior, standard Kelly gives 8.9% while Bayesian modified Kelly gives approximately **4.8% — roughly half-Kelly**. They concluded this "provides a theoretical rationale for the use of half-Kelly," since under realistic uncertainty, the Bayesian posterior naturally produces a fraction near 0.5 of the plug-in Kelly estimate.

For continuous-time assets with uncertain drift μ ~ N(m₀, v₀), Browne and Whitt showed the optimal Bayesian control is f*_t = E[μ | observations up to t] / σ², where the posterior mean is maintained via a Kalman filter. An additional important result for the FPPE system: when volatility σ itself is uncertain with variance σ²_σ, the optimal Kelly leverage becomes **f\* = μ / (E[σ]² + Var(σ))**, as derived by Kurtti (2020) extending Thorp's framework. The volatility-of-volatility term universally reduces optimal leverage below what historical point estimates suggest.

---

## For ±5% estimation error at p̂ = 0.55, the optimal fraction is 75–80% of Kelly

The critical question for the FPPE system has a precise answer grounded in Baker and McHale's (2013) framework. Consider a binary signal with estimated probability p̂ = 0.55 of a favorable outcome against an even-money benchmark (implied q = 0.50). The full Kelly fraction is f* = 2(0.55) − 1 = 0.10, meaning 10% of capital. The edge is 0.05.

The ±5 percentage point estimation error (true p ∈ [0.50, 0.60]) must be translated to a standard deviation σ_p. If the error is modeled as p ~ Uniform(0.50, 0.60), then σ_p = 0.10/(2√3) ≈ 0.029, and **α\* = 0.05²/(0.05² + 0.029²) ≈ 0.75 (75% Kelly)**. If modeled as p ~ Normal(0.55, 0.025²), treating ±5% as a 2σ interval, then σ_p = 0.025 and **α\* = 0.05²/(0.05² + 0.025²) = 0.80 (80% Kelly)**. The recommended bet drops from 10% to 7.5–8.0% of capital.

The complete mapping from estimation error to optimal Kelly fraction, holding edge = 0.05 fixed, is:

| Estimation error σ_p | Signal-to-noise (edge/σ) | Optimal Kelly fraction | Recommended bet size |
|---|---|---|---|
| 0.010 | 5.0 | 96% | 9.6% |
| 0.020 | 2.5 | 86% | 8.6% |
| **0.025** | **2.0** | **80%** | **8.0%** |
| **0.029** | **1.7** | **75%** | **7.5%** |
| 0.050 | 1.0 | 50% | 5.0% |
| 0.075 | 0.67 | 31% | 3.1% |
| 0.100 | 0.50 | 20% | 2.0% |

The BSS connection provides a calibration check. The Brier Score decomposes as BS = Reliability − Resolution + Uncertainty (Murphy, 1973). The **Reliability component directly measures E[(p̂ − p_true)²] ≈ σ²_p**, so extracting the reliability term from a BSS decomposition yields the estimation error variance needed for the shrinkage formula. A system with BSS = 0 (no skill) warrants α* ≈ 0; BSS = 1 (perfect calibration) warrants full Kelly. For intermediate BSS values, the reliability component provides σ_p directly.

**The concrete recommendation for the FPPE system: apply 75% of Kelly sizing when the KNN probability estimates carry ±5% calibration error.** This corresponds to betting 7.5% of capital when the signal indicates p̂ = 0.55 for an even-money proposition. If drawdown constraints are also binding (say, 25% maximum drawdown), the Grossman-Zhou result compounds multiplicatively: the final fraction becomes 0.25 × 0.75 × f_Kelly = 0.1875 × f_Kelly, or about 1.9% of capital. This layered approach — Bayesian shrinkage for estimation uncertainty, drawdown constraint for risk management — provides a principled, academically grounded sizing framework that maps directly from the system's BSS-measured calibration quality to concrete position sizes.

## Conclusion

The landscape of position sizing beyond Kelly reveals a fundamental insight: **full Kelly is a theoretical upper bound that should never be used in practice**. Every realistic consideration — parameter uncertainty (Baker and McHale 2013), drawdown constraints (Grossman and Zhou 1993), Bayesian learning (Browne and Whitt 1996), or finite risk aversion (MacLean, Ziemba, and Blazenko 1992) — produces the same qualitative recommendation: bet less than Kelly. The quantitative recommendation depends on the system's specific characteristics. For the FPPE system, the Baker-McHale formula α* = edge²/(edge² + σ²) provides a closed-form mapping from BSS-derived calibration error to optimal Kelly fraction, yielding **75–80% Kelly for ±5% estimation error**. Risk parity offers a robust portfolio-level complement that avoids the fragility of multi-asset Kelly. And drawdown constraints provide a separate, multiplicative adjustment that can be layered on top. The result is a modular framework: signal-level sizing via fractional Bayesian Kelly, portfolio-level allocation via risk parity, and overall leverage via drawdown-constrained optimal growth.