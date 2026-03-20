# SLE-26 (REMAKE): Alternative Phase 2 Approaches and Assumption Stress-Test

## Scope covered
- 2x ATR stop-loss multiple
- Linear drawdown brake vs exponential/sigmoid alternatives
- Stateless function API vs class-based risk engine
- Kelly/risk-parity style alternatives
- Rejection conditions in Section 5

## Executive summary
The current Phase 2 design is directionally strong for a first productionable risk layer. The main recommendation is to **keep the current baseline choices for implementation simplicity**, but add a small number of safeguards to reduce tail-risk and edge-case fragility before M3/M4 validation.

---

## 1) Alternative approaches considered (with recommendations)

### A. Stop distance: fixed 2x ATR vs regime-aware ATR multiple
**Alternative:** Use a regime-aware ATR multiplier (for example, 1.5x in stable regimes, 2.5-3.0x in high-vol regimes) instead of fixed 2x.

**Why consider it:**
- Fixed ATR multiples are common, but canonical trend-stop variants (for example Chandelier Exit) often use 3x ATR to avoid noise exits.
- Stop-loss literature suggests value is regime-dependent; static rules can help in momentum regimes and hurt in mean-reverting regimes.

**Recommendation:** **DEFER (keep fixed 2x ATR for Phase 2).**  
Rationale: fixed 2x is a valid baseline for controlled implementation and testability. Add this as a Phase 3 experiment behind config flags:
- `stop_loss_atr_multiple_low_vol`
- `stop_loss_atr_multiple_high_vol`
- regime classifier based on recent realized volatility percentile.

---

### B. Drawdown scaler: linear vs exponential/sigmoid with hysteresis
**Alternative:** Replace strict linear scaling with a smooth non-linear curve (sigmoid/exponential), plus hysteresis around thresholds to avoid mode-chatter.

**Why consider it:**
- Linear is simple and interpretable.
- Non-linear curves can keep more risk budget in mild drawdown while cutting faster near halt levels (better convexity near danger zone).
- Hysteresis prevents repeated flip-flops when drawdown hovers near a boundary.

**Recommendation:** **ADOPT PARTIALLY now.**  
Rationale: keep linear as default, but add a **hysteresis buffer** in mode transitions immediately (small code cost, high robustness).  
Suggested rule:
- Enter brake at 15.0%, return to normal only below 14.5%.
- Enter halt at 20.0%, return to brake only below 19.5%.

Defer sigmoid/exponential curve until there is enough out-of-sample evidence from stress tests.

---

### C. Architecture: stateless function module vs class-based risk engine
**Alternative:** Move to a class-based `RiskEngine` object encapsulating config, caches, and validation lifecycle.

**Why consider it:**
- Class API can improve dependency injection and test ergonomics for multi-step workflows.
- Easier extension path once trailing stops, correlation limits, and cross-position constraints are added.

**Recommendation:** **ADOPT current stateless design for Phase 2; DEFER class wrapper.**  
Rationale: for current scope (ATR sizing + drawdown scalar + stop checks), pure functions are easier to reason about and safer from hidden state bugs.  
Middle-ground path:
- Keep pure core functions.
- Optionally add a thin `RiskEngine` facade later that delegates to pure functions.

---

### D. Position sizing framework: fixed-fractional ATR risk vs fractional Kelly
**Alternative:** Use fractional Kelly sizing from model-estimated edge/probability and payoff ratio.

**Why consider it:**
- Kelly is asymptotically growth-optimal under correctly specified probabilities.
- Fractional Kelly can improve long-run capital growth in principle.

**Recommendation:** **REJECT for Phase 2 baseline; DEFER for research track only.**  
Rationale: Kelly is highly sensitive to estimation error and fat tails; with noisy edge estimates it can materially increase drawdown volatility. Current FPPE confidence outputs are not yet calibrated for safe Kelly deployment across changing regimes.

---

### E. Portfolio construction: single-name caps vs risk parity / equal risk contribution
**Alternative:** Shift from per-position ATR sizing plus hard caps to portfolio-level risk budgeting (risk parity / equal risk contribution).

**Why consider it:**
- Portfolio risk budgeting can better control concentration and volatility clustering across simultaneous positions.

**Recommendation:** **DEFER to Phase 3+ portfolio manager.**  
Rationale: risk parity is a portfolio optimizer, not a single-trade acceptance rule; it requires stable covariance estimation, cross-asset exposure controls, and rebalance policy. That is beyond Phase 2's scoped objective.

---

## 2) Rejection-condition review (Section 5): missing conditions

Current rejection rules are good, but several important production guards are missing or under-specified.

### Missing conditions to add
1. **Invalid or non-finite input prices**
   - Reject if `entry_price <= 0` or any required OHLC value is NaN/inf.

2. **Stop price invalid**
   - Reject if computed `stop_price <= 0` or `stop_price >= entry_price` (for long positions).

3. **Stop distance outside sanity bounds**
   - Reject if `stop_distance_pct <= 0` or exceeds a configured hard cap (for example > 50%).

4. **ATR staleness / missingness rule**
   - Explicitly reject if NaN ratio in ATR lookback exceeds threshold (design notes suggest 20%; formalize this in Section 5 with reason string).

5. **Data freshness / date alignment**
   - Reject if latest history row is stale relative to signal date (prevents sizing on lagged data).

6. **Fractional-share disabled and computed whole shares == 0**
   - Already discussed in algorithm section; should also be a formal rejection condition table row.

7. **Available buying power / cash check**
   - Reject if `dollar_amount` exceeds available deployable capital after existing commitments and fees/slippage reserve.

---

## 3) Blind spots in current design

1. **Correlation shock blind spot**
   - ATR sizing is single-name volatility-aware but not correlation-aware; simultaneous losses can still stack in highly correlated names/sectors.

2. **Gap risk underestimation**
   - Next-open execution is realistic, but the 2% risk target is not a hard cap under overnight jumps. Stress scenarios should explicitly test clustered gap-down events.

3. **Boundary-chatter risk near drawdown thresholds**
   - Without hysteresis, repeated oscillation around 15%/20% may produce unstable allocation behavior.

4. **Regime transition fragility**
   - ATR is backward-looking; abrupt volatility expansions can still lead to temporary oversizing.

5. **Validation concentration risk**
   - Heavy reliance on 2024 (bull regime) can mask brake/halt behavior and stop dynamics; synthetic stress and alternate market folds are mandatory for confidence.

---

## References (for rationale)
- Wilder, J. W. (1978), *New Concepts in Technical Trading Systems* (ATR origin).
- Kaminski, K. M., & Lo, A. W. (2014), "When do stop-loss rules stop losses?", *Journal of Financial Markets*.
- Kelly, J. L. (1956), "A New Interpretation of Information Rate", *Bell System Technical Journal*.
- MacLean, Thorp, Ziemba (various works on Kelly/fractional Kelly and estimation risk).

---

## Final recommendation summary
- **2x ATR stop:** **DEFER change** (keep baseline now; test adaptive multiplier later).
- **Linear drawdown brake:** **ADOPT with hysteresis now**; defer non-linear curve experiments.
- **Stateless risk engine:** **ADOPT** for Phase 2; consider thin class facade later.
- **Kelly sizing:** **REJECT for baseline**, keep for research only.
- **Risk parity/ERC:** **DEFER** to Phase 3+ portfolio layer.
- **Rejection conditions:** **ADOPT additions** listed above before production validation.
