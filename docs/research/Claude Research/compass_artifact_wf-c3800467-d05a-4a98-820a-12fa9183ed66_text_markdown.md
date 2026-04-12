# Transaction Cost Analysis for Systematic Equity Trading

**For a systematic quantitative system trading 50–500 mid-cap US equities daily through Interactive Brokers with sub-$10M AUM, realistic round-trip transaction costs range from 4 to 19 basis points, with a well-supported base case of approximately 10–11 bps.** At a 5-day average holding period, this translates to roughly 50 round trips per year and an annual cost drag of **2–5.3%** of portfolio value — a significant hurdle requiring meaningful alpha generation. The dominant cost component at this scale is the bid-ask spread, not market impact; the Almgren-Chriss optimal execution framework, while foundational to the field, is largely unnecessary for orders representing less than 0.25% of average daily volume. Below is a comprehensive analysis of every component.

---

## The Almgren-Chriss framework and why it barely matters at small scale

Robert Almgren and Neil Chriss published their landmark paper "Optimal Execution of Portfolio Transactions" in the *Journal of Risk* (Vol. 3, No. 2, Winter 2000/2001), establishing the mathematical foundation for algorithmic execution. The core insight is a **mean-variance tradeoff**: trading fast incurs high market impact but eliminates timing risk (price volatility during execution), while trading slowly minimizes impact but exposes the portfolio to adverse price movements.

The model formalizes this as minimizing **U(x) = E(x) + λ·V(x)**, where E(x) is expected implementation shortfall, V(x) is cost variance, and λ is the risk aversion parameter. The price evolves as arithmetic Brownian motion with two impact components: **permanent impact** g(v) = γ·v, which shifts the equilibrium price proportionally to trading rate (the only functional form free from arbitrage, per Huberman & Stanzl 2004), and **temporary impact** h(v) = ε·sign(v) + η·|v|, representing the immediate liquidity premium that reverts after each trade. The fixed cost ε approximates half the bid-ask spread plus fees, while η captures the variable cost of demanding liquidity.

The closed-form optimal trajectory takes a beautifully simple shape: **x(t) = X₀·sinh(κ(T−t))/sinh(κT)**, where κ = √(λσ²/η) controls curvature. When risk aversion λ → 0, the trajectory becomes linear (TWAP — sell at constant rate). When λ → ∞, the trader liquidates immediately, accepting enormous temporary impact to eliminate all market risk. The intermediate case produces a front-loaded hyperbolic curve, with the "half-life" θ = 1/κ = √(η/(λσ²)) governing the natural decay rate. If the execution deadline T is much shorter than θ, the optimal strategy approximates TWAP; if T is much longer, execution is heavily front-loaded.

**For a sub-$10M system, the full Almgren-Chriss optimization is unnecessary.** A $50K trade in a mid-cap stock with $50M daily dollar volume represents **0.1% of ADV**. At this participation rate, the Almgren et al. (2005) empirical model predicts total market impact of roughly **1–3 bps** — below the bid-ask spread for most mid-cap names. The "optimization" between impact and timing risk is moot when both quantities are negligible. The framework's conceptual lessons still apply: keep individual orders below 1–2% of ADV, front-load execution when signals are decaying, and understand that permanent impact is a fixed cost independent of trajectory. But implementing the full machinery would be over-engineering at this scale.

---

## Implementation shortfall: measuring the gap between theory and reality

André Perold's 1988 *Journal of Portfolio Management* paper "The Implementation Shortfall: Paper versus Reality" defined IS as the difference between the return on a hypothetical paper portfolio (executed frictionlessly at decision prices) and the actual portfolio return. For a complete execution, this simplifies to **IS = (Average Execution Price − Decision Price) × Shares + Fees**, expressed in basis points relative to the intended trade value.

The Wagner-Edwards (1993) expansion decomposed IS into four actionable components. **Delay cost** captures the price drift between the investment decision and order entry — when a portfolio manager decides to buy at $50.00 at 9:00 AM but the trader doesn't submit until 10:30 AM at $50.10, the delay cost is 20 bps. **Market impact** measures the execution price versus the arrival price, reflecting the trade's own pressure on the market. **Opportunity cost** accounts for unfilled shares — if only 75% of the intended order executes and the stock subsequently rises, the missed gain on the remaining 25% is captured here. **Commission cost** covers explicit fees. A worked example from CFA Institute materials: with a $50 decision price, 75,000 of 100,000 shares executed at $50.30, closing at $50.65 with $0.05/share commission, total IS equals **85 bps** (delay 15 bps, impact 30 bps, opportunity 32.5 bps, fees 7.5 bps).

Robert Kissell extended this in his 2006 *Journal of Trading* paper and subsequent textbooks (2003 with Glantz, 2013 solo) to a nine-component framework spanning investment-related, execution-related, and opportunity-related costs. His **I-Star model** estimates instantaneous market impact as **I* = a₁·(Q/ADV)^a₂·σ^a₃**, where published parameters are a₁ ≈ 750, a₂ ≈ 0.2, a₃ ≈ 0.9. When combined with a participation rate adjustment, market impact becomes MI = b₁·I*·POV^a₄ + (1−b₁)·I*, with b₁ ≈ 0.9 splitting temporary from permanent components and a₄ ≈ 0.5. The Kissell Research Group publishes quarterly calibration updates; their **Q1 2024 report** shows mid-cap trading costs of **18.7 bps** for orders at 10% of ADV using VWAP execution, compared to 17.0 bps for large-caps and 28.9 bps for small-caps.

The Frazzini, Israel, and Moskowitz (2012/2017) study — using **$721 billion in live AQR trades** across 5.3 million orders from 1998–2011 — provides the most comprehensive empirical IS data available. They found mean market impact of **12.18 bps** across all trades, with median impact of **7.98 bps**. Large-cap impact averaged 11.21 bps while small-cap averaged 21.27 bps, placing mid-caps in the interpolated range of **15–20 bps** for institutional-scale orders. A critical finding: roughly **85–90% of price impact is permanent**, with only about 2 bps of temporary impact reversing the following day.

---

## Market impact models converge on the square-root law

The most robust empirical regularity in market microstructure is the **square-root law of market impact**: ΔP/P ≈ σ·√(Q/V), where σ is daily volatility, Q is order size, and V is average daily volume. The proportionality constant is empirically close to **1.0** (dimensionless). For a mid-cap stock with 2% daily volatility and a trade representing 5% of daily volume, expected impact is 0.02 × √0.05 ≈ **45 basis points**. This relationship has been confirmed across equities, futures, FX, Bitcoin, and bonds — a near-universal feature of financial markets.

Kyle's foundational 1985 *Econometrica* model predicts **linear** impact: ΔP = λ·y, where λ = √(Σ₀)/(2σ_u) captures the market maker's inference problem between informed and noise trading. While theoretically elegant and applicable to individual transactions, the linear prediction contradicts the concave (square-root) impact observed for large metaorders. Kyle and Obizhaeva (2016–2018) attempted reconciliation through "market microstructure invariance," arguing that a properly specified linear model — accounting for execution duration — generates an apparent square-root pattern when execution time is omitted.

The **Almgren, Thum, Hauptmann, and Li (2005)** empirical study, using roughly 29,500 filtered institutional orders from Citigroup's US equity brokerage, found permanent impact linear in Q/V with coefficient **γ = 0.314**, and temporary impact following a **3/5 power law** (exponent β = 0.600 ± 0.038, rejecting the pure square-root of 0.5 at 95% confidence). Their calibrated model predicts that buying 10% of ADV in a mid-cap stock like Darden Restaurants (σ = 2.26%, ADV = 1.93M shares) costs **23–43 bps** depending on execution speed, with permanent impact of 22 bps and temporary impact ranging from 12 bps (half-day execution) to 32 bps (one-hour execution).

Tóth et al. (2011), publishing in *Physical Review X* from Capital Fund Management, provided theoretical grounding for the square-root law by demonstrating that **latent liquidity is V-shaped and vanishes near the current price**. This local depletion of available orders breaks the linear response assumption, producing concave impact that scales as Q^(1/2). Their analysis of ~500,000 proprietary CFM futures trades confirmed exponents of 0.5 for small-tick and 0.6 for large-tick contracts. Bouchaud, Farmer, and Lillo's "propagator model" further showed that individual trade impact decays as a power law G(τ) ~ τ^(-0.5), with roughly **2/3 of peak impact persisting permanently** — the "fair pricing" condition.

For the target profile of $10K–$100K trades in mid-cap equities, these models converge on practical estimates:

| Trade Size | % of Typical Mid-Cap ADV | Expected Impact |
|---|---|---|
| $10,000 | 0.02% | ~3 bps |
| $25,000 | 0.05% | ~4 bps |
| $50,000 | 0.10% | ~6 bps |
| $100,000 | 0.20% | ~9 bps |

At sub-$10M AUM, **market impact is a secondary cost driver**, typically contributing only 1–5 bps per side versus 3–8 bps for spread crossing.

---

## Interactive Brokers: commissions, algorithms, and execution infrastructure

IBKR Pro's **fixed pricing** charges $0.005 per share with a $1 minimum, capped at 1% of trade value. For a $50 stock, this translates to **1.0 bp per side**. The tiered structure starts at $0.0035/share for monthly volumes under 300,000 shares, declining to $0.0005/share above 100 million shares, plus exchange, clearing, and regulatory pass-throughs. An important detail for API users: **directed orders cannot use tiered pricing** — only SmartRouted orders qualify. Regulatory fees (SEC at $0.0000206 per dollar sold, FINRA TAF at $0.000195 per share sold) add roughly 0.05 bps — negligible. A typical $50K trade of a $50 stock (1,000 shares) incurs approximately $5 in commissions on fixed pricing, or **1 bp per side**.

IBKR offers **12+ algorithms through the TWS API**, each specified via the `algoStrategy` and `algoParams` fields. The **Adaptive algo** is the recommended default for most orders at this scale — it accepts a single priority parameter (`Urgent`, `Normal`, or `Patient`) and uses SmartRouting to work between the bid and ask, typically achieving better prices than naked limit orders. The **VWAP algo** accepts max percentage of volume (1–50%), start/end times, and options to avoid taking liquidity or speed up in favorable conditions; it seeks to match the volume-weighted average price benchmark and works best for orders spanning a significant portion of the trading day. The **TWAP algo** distributes orders uniformly over a specified time window with configurable strategy types (`Marketable`, `MatchMidpoint`, `MatchSameSide`, `MatchLast`). The **Arrival Price algo** is the implementation shortfall minimizer, benchmarking against the mid-quote at submission time with urgency settings from `Passive` to `Get Done`. For larger blocks, the **Dark Ice** algo hides order size across dark pools, and the **Accumulate/Distribute** algo offers granular control over child order size, timing, and randomization.

For the target profile, the optimal algo selection is straightforward. **Use Adaptive "Normal" for 80%+ of orders** — at $10K–$100K size, these are small enough that simple patient execution inside the spread delivers the best cost/fill tradeoff. Reserve VWAP or Arrival Price for occasional larger orders exceeding 2–3% of ADV. Use Dark Ice when trading less liquid mid-caps where showing size might move the quote.

Mid-cap US equities exhibit **quoted spreads of 5–15 bps** (2–6 cents for $30–$100 stocks), with effective half-spreads of **3–8 bps** after accounting for price improvement. IBKR Pro's SmartRouting — which does not accept payment for order flow, unlike IBKR Lite — scans all market centers for best execution and routes to the venue offering the best price including fee/rebate considerations. Testing by BrokerChooser found a **100% price improvement rate** on IBKR Pro marketable orders. Spreads are widest at the open (2–3× normal) and tightest from 10:00 AM to 3:00 PM Eastern.

---

## Consolidated slippage: three scenarios for backtesting and live deployment

Drawing from every source examined — Quantopian's live-trading-calibrated 5 bps/side default, Frazzini et al.'s 12 bps institutional mean, Kissell's 18.7 bps mid-cap figure for 10% ADV orders, Schwarz's 2025 *Journal of Finance* finding of 7.2 bps round-trip at the best retail broker, and practitioner consensus from Ernie Chan and the quantitative trading community — here are the consolidated estimates.

**Per-side costs (basis points)**

| Component | Aggressive | Moderate | Conservative |
|---|---|---|---|
| Spread (half-spread) | 1.5 | 3.0 | 5.0 |
| Market impact | 0.5 | 1.5 | 3.0 |
| Commission (IBKR) | 0.5 | 1.0 | 1.5 |
| **Total per side** | **2.5** | **5.5** | **9.5** |

**Round-trip costs (basis points)**

| Component | Aggressive | Moderate | Conservative |
|---|---|---|---|
| Spread | 3.0 | 6.0 | 10.0 |
| Market impact | 1.0 | 3.0 | 6.0 |
| Commission | 1.0 | 2.0 | 3.0 |
| **Total round-trip** | **5.0** | **11.0** | **19.0** |

The **aggressive** scenario assumes predominantly limit orders with good fill rates, upper-range mid-caps ($5B+ market cap, tighter spreads), and execution during high-liquidity periods. The **moderate** scenario — validated by Quantopian's 5 bps/side live trading data and corroborated by Schwarz (2025) — uses a mix of market and limit orders with IBKR Pro Fixed pricing and typical mid-range mid-caps. The **conservative** scenario assumes market orders, wider-spread stocks near the $2B market cap boundary, and elevated volatility environments. For backtesting, **the moderate case of ~5 bps per side (11 bps round-trip) is the recommended default**, with sensitivity analysis spanning the aggressive and conservative bounds.

Ernie Chan's guidance reinforces this range: he notes SPX stock spreads of 1–2 bps and warns that "if your expected edge per trade is 5 basis points and your transaction cost is 4 basis points, you don't have a strategy." Kissell's institutional mid-cap figure of 18.7 bps applies to orders at 10% of ADV — roughly 50× larger than typical trades in this profile — making it an extreme upper bound for small-AUM traders.

---

## What a 5-day holding period means for your cost budget

A 5-day average holding period with a fully invested $5–10M portfolio implies approximately **50 complete round trips per year** per unit of capital deployed. At the moderate cost estimate, this produces an annual transaction cost drag of **~5.3%** — a substantial hurdle that exceeds many factor premiums. The table below shows how holding period dramatically affects the cost calculus:

| Holding Period | Round Trips/Year | Annual Cost (Moderate) | Break-Even Alpha |
|---|---|---|---|
| 1 day | 252 | 27.7% | Impractical |
| 2 days | 126 | 13.9% | ~14% |
| 5 days | 50 | 5.5% | ~5.5% |
| 10 days | 25 | 2.8% | ~3% |
| 21 days | 12 | 1.3% | ~1.3% |

Transaction costs become the **dominant performance factor below approximately 5-day holding periods** for strategies generating alpha in the 5–10% annual range. The aggressive cost scenario reduces the 5-day break-even to a more manageable 2.5%, while the conservative scenario pushes it to 9.5%. This framework clarifies an essential design constraint: a 5-day systematic strategy through IBKR must generate at least **5–6% gross annual alpha** just to cover friction, before considering implementation risk, data costs, and drawdown management.

---

## Conclusion: the practitioner's cost map

The body of evidence — from Almgren-Chriss theory through Kyle's lambda to Tóth et al.'s universal square-root law — converges on a clear picture for small-AUM systematic equity trading. Market impact, the central concern of institutional execution research, is **nearly irrelevant at sub-$10M scale** because individual orders represent a negligible fraction of daily volume. The bid-ask spread is the dominant cost, contributing roughly 55% of total friction in the moderate scenario. IBKR's Adaptive algorithm with Normal priority is the optimal default execution method, and the full Almgren-Chriss trajectory optimization is unnecessary overkill.

**Use 5 bps per side (10–11 bps round-trip) as the base case for backtesting**, consistent with Quantopian's live-trading calibration and the Schwarz (2025) retail execution study. Run sensitivity at 2.5 bps (aggressive) and 9.5 bps (conservative) per side to bracket outcomes. For a 5-day holding period, budget approximately 5% of portfolio value annually for transaction costs and ensure your strategy's gross alpha clears this bar with margin. The single highest-leverage execution improvement available at this scale is not sophisticated impact modeling — it is disciplined use of limit orders and patient execution during high-liquidity windows between 10 AM and 3 PM Eastern.