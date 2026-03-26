# FPPE Full Roadmap — Product Requirements & Design Specification

**Version:** 1.0
**Date:** 2026-03-26
**Status:** APPROVED — Ready for implementation planning
**Timeline:** April 2026 → January 2027 (9 months, measured/sequential gates)
**Starting Capital:** $10,000

---

## 1. Project Overview & Objectives

### 1.1 Product

**FPPE (Financial Pattern Prediction Engine)** — K-nearest-neighbor historical analogue matching on return fingerprints. Generates probabilistic BUY/SELL/HOLD signals. Full roadmap from current state (M9, BSS < 0) through live deployment with $10,000 capital by January 2027.

### 1.2 Core Problem

FPPE has a mature, well-tested architecture (616 tests, 5-stage PatternMatcher, 4-layer trading system) but cannot deploy capital because:

1. **Probability calibration is broken** — BSS < 0 on all 6 walk-forward folds (mean -0.00459)
2. **Position sizing lacks compounding logic** — Half-Kelly blocked on BSS gate
3. **Risk engine and portfolio manager** exist as research pilots but aren't integrated into production
4. **No live execution infrastructure** — broker adapters, order management, reconciliation
5. **Universe is limited** to 585 tickers (target: 1500+)

### 1.3 Success Criteria (Live Deployment Gate)

All of the following must hold on out-of-sample data before real capital is deployed:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| BSS | > 0 on ≥ 3/6 walk-forward folds | Model probabilities beat base rate |
| Net expectancy per trade | > $0 after all friction | System makes money per trade on average |
| Confidence calibration | 70% confidence signals win 65-75% | Probabilities are meaningful |
| Drawdown tolerance | Peak-to-trough ≤ 25% | Capital preservation during losing streaks |
| Baseline comparison | Outperforms ≥ 2/3 baselines | System adds value beyond alternatives |
| No performance collapse | No 30-day window with Sharpe < -1.0 | No catastrophic regime failure |
| Minimum trade count | ≥ 50 trades in backtest/paper period | Enough data for statistical significance |
| Paper trading stability | 3 months with all criteria holding | Real-time validation |

**Baselines:** SPY buy-and-hold, equal-weight random selection (100 iterations, median), FPPE raw signals (no trading system).

### 1.4 Starting Capital & Growth Model

- **Live deployment begins with $10,000 starting capital**
- System is percentage-based and scales naturally
- Growth milestones ($25k, $50k, $100k, $500k) represent review checkpoints, not separate configurations

---

## 2. Phase Architecture & Gate Structure

### 2.1 Design Principles

- **9 sequential phases + 1 post-launch side track** (options foundation)
- **Every phase has:** Scope, Gate, Diagnostic Protocol, Fallback
- **3-strike rule:** If three consecutive attempts at the same fix fail, STOP, log, escalate
- **Selective parallelism:** Phases 5 and 6 can run in parallel with Phase 4
- **Feature flags:** All enhancements behind flags; revert if they hurt

### 2.2 Phase Dependency Graph

```
Phase 1 (BSS Fix) ──► Phase 2 (Half-Kelly) ──► Phase 3 (Risk Engine)
                                                      │
                                                      ▼
                                              Phase 4 (Portfolio Mgr)
                                                 │          │
                                                 │    [parallel start]
                                                 ▼          ▼
                                          Phase 6      Phase 5
                                        (Universe)   (Live Plumbing)
                                                 │          │
                                                 ▼          ▼
                                              Phase 7 (Model Enhancements)
                                                      │
                                                      ▼
                                              Phase 8 (Paper Trading 3mo)
                                                      │
                                                      ▼
                                              Phase 9 (Live Deploy)
                                                      │
                                                      ▼
                                           Phase 10 (Options Foundation)
```

### 2.3 Timeline Allocation

| Phase | Duration | Target Complete | Dependencies |
|-------|----------|-----------------|--------------|
| 1 — BSS Fix | 3-4 weeks | Late April 2026 | None |
| 2 — Half-Kelly | 2 weeks | Mid May 2026 | Phase 1 gate |
| 3 — Risk Engine | 3 weeks | Early June 2026 | Phase 2 gate |
| 4 — Portfolio Manager | 3 weeks | Late June 2026 | Phase 3 gate |
| 5 — Live Plumbing | 4 weeks | Late July 2026 | Phase 3 gate (parallel w/ 4, 6) |
| 6 — Universe Expansion | 3 weeks | Mid July 2026 | Phase 1 gate (parallel w/ 4, 5) |
| 7 — Model Enhancements | 4 weeks | Late August 2026 | Phases 4, 5, 6 gates |
| 8 — Paper Trading | 12 weeks | Late November 2026 | Phase 7 gate |
| 9 — Live Deploy | 4 weeks | January 2027 | Phase 8 gate |
| 10 — Options Foundation | 4-6 weeks | Post-launch | Phase 9 stable |

---

## 3. Phase 1 — BSS Diagnosis & Calibration Fix (M9 Completion)

### 3.1 Goal

Achieve BSS > 0 on ≥ 3/6 walk-forward folds. This is the most critical phase — nothing else matters if the model's probability estimates are worse than the base rate.

### 3.2 Current State

- BSS mean: -0.00459, 0/6 positive folds
- Root cause identified (session 2026-03-25): analogue pool dilution at 585-ticker scale
- Platt calibration is actually helping (+0.023 BSS vs raw) — not the culprit
- Threshold mismatch: locked 0.65 never fires (max Platt prob = 0.6195 on 585T), operating at 0.55
- SectorConvictionFilter recalibrated (0.03 → 0.005) but untested post-fix

### 3.3 Diagnostic Protocol (Structured Investigation Order)

1. **Reliability Diagram** — Plot predicted probability vs. actual hit rate in 10 equal-frequency bins across all 6 folds. Shows exactly where calibration is off (overconfident? underconfident? specific probability ranges?).

2. **Base-Rate Decomposition** — Compute per-fold base rates (% of all rows where `fwd_7d_up = 1`). If base rate is ~50% and BSS is slightly negative, the model is barely distinguishable from a coin flip at this universe scale. Compare 585T base rates to the original 52-ticker base rates to quantify dilution.

3. **Distance Distribution Analysis** — Plot the distribution of neighbor distances per fold. If 585T pushes mean distances up (analogues are farther away), the KNN signal weakens. Compare to 52-ticker distance distributions.

4. **Hypothesis Testing (in order):**
   - **H1: Tighten max_distance.** Current 1.1019 was calibrated on 52 tickers. Run sweep [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] on 585T. If BSS improves, the pool was too dilute.
   - **H2: same_sector_only filtering.** Restrict analogues to same sector. Reduces pool size dramatically but increases relevance. Run walk-forward with `same_sector_only=True`.
   - **H3: Reduce top_k.** Current 50 neighbors may average out signal at 585T. Sweep [10, 20, 30, 40, 50].
   - **H4: Refit Platt on 585T distribution.** Current Platt parameters may not match the new probability distribution. Force recalibration with fresh sigmoid fit.

5. **Post-fix validation:** Re-run SectorConviction (E1) and Momentum (E2) experiments with corrected min_sector_lift=0.005 to measure filter impact on the fixed baseline.

### 3.4 Gate

- BSS > 0 on ≥ 3/6 walk-forward folds

### 3.5 Fallback (3-Strike Rule)

If H1-H4 all fail to produce positive BSS:
- Log all results in session log with exact numbers
- Escalate decision: consider reverting to 52-ticker universe for live deployment while investigating 585T separately
- Consider alternative calibration (BMA calibrator from `research/bma_calibrator.py`) as a rescue path

### 3.6 Deliverables

- `scripts/reliability_diagram.py` — visualization tool
- `results/bss_fix_sweep_*.tsv` — experiment results with provenance
- Updated locked settings if max_distance or top_k changes
- Session log documenting which hypothesis resolved it

---

## 4. Phase 2 — Half-Kelly Position Sizer (M10)

### 4.1 Goal

Replace flat dollar sizing with fractional-Kelly compounding so positions scale with portfolio growth.

### 4.2 Prerequisite

Phase 1 gate passed (BSS > 0 on ≥ 3/6 folds).

### 4.3 Design

```
kelly_fraction = (p * b - q) / b
  where p = calibrated win probability (from Platt)
        q = 1 - p
        b = avg_win / avg_loss (from historical walk-forward data)

half_kelly = 0.5 * kelly_fraction  # conservative scaling
position_size = half_kelly * current_equity

# Safety clamps (from existing config):
position_size = clamp(position_size, min=2% equity, max=10% equity)
```

### 4.4 Integration Point

New module `trading_system/position_sizer.py` sits between Layer 2 (risk engine) and Layer 3 (portfolio manager). Receives raw volatility-based size from risk engine, scales by Kelly fraction.

### 4.5 Diagnostic Protocol

1. **If Kelly fraction is consistently tiny (< 0.01):** BSS is positive but edge is thin. Check if win probability from Platt is barely above 50%. May need to revisit Phase 1.
2. **If Kelly fraction is negative for any fold:** That fold's calibration produces p < q — the model predicts losing trades. Isolate that fold, check if it's a specific regime (e.g., 2022-Bear).
3. **If backtest Sharpe drops vs. flat sizing:** Kelly is amplifying variance faster than it compounds. Reduce to quarter-Kelly (0.25×) and retest.

### 4.6 Gate

- Kelly fraction positive on ≥ 4/6 folds
- Backtest Sharpe ≥ 1.0 (full walk-forward, Kelly-sized)
- Max drawdown ≤ 15%

### 4.7 Fallback

If Kelly sizing degrades performance, revert to volatility-only sizing (ATR-based). Kelly becomes a Phase 7 enhancement instead.

### 4.8 Deliverables

- `trading_system/position_sizer.py` — Half-Kelly module with Pydantic config
- Tests: kelly fraction computation, clamp behavior, negative-edge rejection, fold-level validation
- Walk-forward results comparing Kelly vs. flat sizing

---

## 5. Phase 3 — Risk Engine Integration

### 5.1 Goal

Activate existing risk engine research pilots into production: ATR stops, volatility sizing, drawdown brake.

### 5.2 Current State

`trading_system/risk_overlays/` has FatigueAccumulation and LiquidityCongestion behind flags. ATR stop-loss locked at 3.0×. Risk engine design exists in `FPPE_TRADING_SYSTEM_DESIGN.md` Section 4.3 but isn't wired into production.

### 5.3 Scope

1. **Activate ATR position sizing:**
   ```
   stop_distance = 3.0 × ATR%
   raw_weight = max_loss_per_trade_pct (2%) / stop_distance
   final_weight = clamp(raw_weight × drawdown_scalar, 2%, 10%)
   ```
   Replaces flat sizing as base layer; Half-Kelly from Phase 2 applied as multiplier on top.

2. **Wire drawdown brake** — Linear scalar 1.0 → 0.0 as drawdown goes 15% → 20%. At 20%, halt all new trades.

3. **Activate FatigueAccumulation overlay** — Exponential approach model reducing sizing after consecutive losses. Regime-transition reset prevents stale fatigue.

4. **Activate LiquidityCongestion gate** — ATR/price ratio check. High congestion throttles position size.

5. **Stop-loss execution model** — Stops evaluated against intraday low; exit executes at next-day open (not stop price).

### 5.4 Diagnostic Protocol

1. **If max DD worsens:** Compare trade-by-trade logs before/after. Check if drawdown brake threshold (15%) is too loose — try 10%. Check if ATR sizing produces larger positions than flat sizing.
2. **If Sharpe drops > 0.3:** Fatigue overlay may be too aggressive. Test with fatigue disabled, tune decay rate.
3. **If trade count drops > 40%:** Liquidity congestion gate too strict. Plot ATR/price distribution, adjust threshold to block only worst 5-10%.
4. **If stops fire too frequently (> 50% of exits):** 3.0× ATR may be tight for current vol regime. Document finding — locked setting, do NOT change without formal sweep.

### 5.5 Gate

- Drawdown brake fires correctly (verified with synthetic 20% DD scenario)
- Max DD ≤ 10% on walk-forward
- Sharpe ≥ 1.0 maintained from Phase 2
- Stop-loss fires ≤ 35% of exits

### 5.6 Fallback

If overlays degrade performance, keep ATR sizing + drawdown brake (core risk), disable fatigue and congestion (stay behind flags).

### 5.7 Deliverables

- `trading_system/risk_engine.py` — production activation
- Integration tests: drawdown brake boundaries, fatigue + reset, congestion gate
- Walk-forward comparison: Phase 2 vs Phase 3

---

## 6. Phase 4 — Portfolio Manager Activation

### 6.1 Goal

Activate Layer 3 for multi-position capital allocation with sector limits, signal ranking, and capital utilization.

### 6.2 Current State

`trading_system/portfolio_manager.py` exists with `rank_signals`, `check_allocation`, `allocate_day`. Validated on 52-ticker universe (37 rejections). Not yet tested on 585T.

### 6.3 Scope

1. **Activate portfolio manager** — Wire `use_portfolio_manager=True` into default config.
2. **Signal ranking (v1):** Confidence ranking, tie-break on lower correlation, then alphabetical.
3. **Sector limit enforcement:** Max 30% equity per sector, max 3 positions per sector.
4. **Capital utilization tracking:** Target < 50% average idle cash. > 80% = RED alert.
5. **Concurrent position capacity:** At $10k with 2-10% sizes, expect 10-20 concurrent positions.

### 6.4 Diagnostic Protocol

1. **If idle cash > 80%:** Signal throughput too low. Check: confidence threshold, sector limits, cooldown periods. Run attribution to identify binding constraint.
2. **If sector concentration despite limits:** Audit `SECTOR_MAP` against current GICS classifications.
3. **If PM rejections > 60%:** Constraints too tight. Relax `max_positions_per_sector` from 3 → 5.
4. **If total return drops:** PM rejecting good trades. Log every rejection with reason. Check if high-confidence signals blocked by sector limits.

### 6.5 Gate

- Sector limits enforced (no sector > 30% at any point)
- Average idle cash < 50%
- Sharpe ≥ 1.0 maintained
- PM rejection log shows reasonable distribution

### 6.6 Fallback

Simplify to sector-limit-only mode (skip ranking, accept all signals that pass sector check). Full ranking becomes Phase 7 enhancement.

### 6.7 Deliverables

- Production config update: `use_portfolio_manager=True`
- Validation on 585T universe
- PM rejection log analysis script
- Walk-forward comparison: Phase 3 vs Phase 4

---

## 7. Phase 5 — Live Execution Plumbing (Parallel Start During Phase 4)

### 7.1 Goal

Build infrastructure to connect FPPE to a real broker. Can begin in parallel with Phase 4 (no shared code paths).

### 7.2 Scope

1. **BaseBroker ABC** (`trading_system/broker/base.py`):
   ```python
   class BaseBroker(ABC):
       submit_order(order: Order) -> OrderResult
       get_positions() -> list[Position]
       get_account() -> AccountSnapshot
       cancel_order(order_id: str) -> bool
   ```

2. **MockBrokerAdapter enhancement** — Configurable latency (0-500ms), partial fills, rejection scenarios, slippage simulation (10bps model).

3. **IBKR Adapter** (`trading_system/broker/ibkr.py`):
   - TWS API or Client Portal Gateway
   - REST JSON payloads
   - Connection health monitoring with auto-reconnect

4. **Order Execution Manager** (`trading_system/order_manager.py`):
   - Consumes `AllocationDecision` from Layer 3
   - Translates to broker-specific orders
   - Tracks state: PENDING → SUBMITTED → FILLED / REJECTED / CANCELLED
   - Handles partial fills

5. **OOB Reconciliation** (`scripts/reconcile.py`):
   - Daily 09:00 AM ET
   - Polls broker API for actual positions
   - Compares against `PortfolioSnapshot`
   - Mismatch > 0.05% → `SystemError`, blocks 4:00 PM execution

6. **LiveRunner hardening:**
   - Full Pydantic state management
   - Strict test coverage: connection loss, partial fills, timeout, rejection
   - 5-minute hard timeout for 4:00 PM execution window

### 7.3 Diagnostic Protocol

1. **If mock tests pass but IBKR fails:** API authentication or connection issue. Check TWS gateway, API permissions, paper trading account.
2. **If OOB shows persistent mismatches:** Corporate actions (splits, dividends). Add corporate action handler.
3. **If execution latency > 5 minutes:** Profile pipeline. Likely: HNSW index loading (use `load_index`), feature computation (batch optimize), broker API throttling.
4. **If partial fills cause drift:** Track cumulative fill %. If < 90% after 2 minutes, cancel remainder, adjust SharedState.

### 7.4 Gate

- Mock broker round-trip parity: 100 synthetic trades correct
- OOB reconciliation passes 30 consecutive days
- Execution pipeline < 3 minutes for full 585T
- All error scenarios have test coverage

### 7.5 Fallback

If IBKR too complex, deploy with Alpaca API (simpler REST, free paper trading). IBKR becomes Phase 9 upgrade.

### 7.6 Deliverables

- `trading_system/broker/` — BaseBroker ABC, MockBrokerAdapter v2, IBKRAdapter
- `trading_system/order_manager.py` — order lifecycle
- `scripts/reconcile.py` — daily OOB reconciliation
- Enhanced `pattern_engine/live.py` — production LiveRunner
- Full test suite: broker parity, reconciliation, timeout, partial fills

---

## 8. Phase 6 — Universe Expansion (585 → 1500+ Tickers, Parallel with Phase 5)

### 8.1 Goal

Scale from 585 to S&P 500 + Russell 1000 (~1500 unique tickers) while maintaining signal quality and execution speed.

### 8.2 Current State

HNSW disk persistence implemented. 54.5× speedup confirmed. Vol-norm epsilon guard and inf clipping committed. Architecture is ready.

### 8.3 Scope

1. **Ticker sourcing & data pipeline** — Extend `prepare.py` for Russell 1000. Apply 2010 historical depth gate. ~1500 tickers expected to survive.

2. **HNSW index scaling validation** — Build index on ~3.7M rows. Measure:
   - Build time (single-threaded, Py3.12 Windows)
   - Query latency for 1500 queries
   - Recall@50 vs BallTree exact **(must hold ≥ 0.9999)**
   - Disk serialization size and load time

3. **Overnight index build pipeline** — Index built after EOD data pull. LiveRunner loads via `load_index()` for sub-second startup.

4. **Sector mapping expansion** — Extend `SECTOR_MAP` to all 1500 tickers with GICS classifications.

5. **BSS re-validation at scale** — Re-run walk-forward on 1500T. Confirm Phase 1 fix holds at larger scale.

6. **Feature pipeline performance** — Profile `prepare.py` on 1500T. Optimize if > 30 minutes.

### 8.4 Diagnostic Protocol

1. **If BSS degrades on 1500T:** Analogue pool dilution recurring. Apply same Phase 1 fix (likely tighter max_distance or same_sector_only). May need further tightening.
2. **If HNSW build time > 30 min:** Acceptable for overnight. If query latency > 5s, increase `ef_construction` or reduce `M`. Profile memory — 3.7M rows may approach 32GB limit.
3. **If recall drops below 0.9999:** Increase HNSW `ef_search` parameter. If recall < 0.9995 at any reasonable `ef_search`, fall back to BallTree with sector pre-filtering.
4. **If thinly-traded tickers produce NaN/inf:** Vol-norm epsilon should catch. If not, add zero-volume filter (volume < 10,000 shares → drop).

### 8.5 Gate

- HNSW recall@50 ≥ 0.9999 on 1500T
- BSS > 0 on ≥ 3/6 folds at 1500T
- Full pipeline (data → features → index → query → signals) < 2 hours overnight
- LiveRunner 4:00 PM execution < 3 minutes with pre-built index
- No unmapped sectors, no NaN/inf in feature matrix

### 8.6 Fallback

If BSS cannot hold at 1500T, deploy live with 585T universe. Expansion becomes post-launch enhancement.

### 8.7 Deliverables

- Extended data pipeline for Russell 1000
- HNSW scaling benchmark results (`results/hnsw_1500t_benchmark.tsv`)
- `scripts/build_overnight_index.py`
- Expanded `SECTOR_MAP`
- Walk-forward results at 1500T with provenance

---

## 9. Phase 7 — Model Enhancements

### 9.1 Goal

Improve probability quality and signal accuracy. Each enhancement is independently gated — must improve BSS to be kept.

### 9.2 Principle

Every enhancement behind a feature flag. If it hurts, revert. No enhancement is mandatory for live deployment — Phases 1-6 produce a deployable system. Phase 7 makes it better.

### 9.3 Enhancement 1: BMA Calibrator Integration (Highest ROI)

Wire `research/bma_calibrator.py` into prediction pipeline. Student's t mixture EM produces full posterior distributions.

- **Integration:** `EngineConfig(calibration="bma")` alongside `"platt"`
- **Gate:** BSS improvement ≥ +0.005 on ≥ 3/6 folds vs Platt baseline
- **Diagnostic:** If BMA produces very wide posteriors, analogue pool too heterogeneous. Check if same_sector_only tightens the posterior.

### 9.4 Enhancement 2: Conformal Prediction Intervals

Wrap output with statistically guaranteed prediction intervals (e.g., "90% coverage: return between +1.2% and +4.5%").

- **Integration:** Wire `pattern_engine/conformal_hooks.py` into post-calibration. Interval width feeds Kelly sizing — wider → smaller positions.
- **Gate:** Empirical coverage ≥ 88% at nominal 90% level across all 6 folds
- **Diagnostic:** If coverage < 85%, recalibrate non-conformity score function (residual vs normalized).

### 9.5 Enhancement 3: Dynamic Time Warping Distance

Replace/augment Euclidean with DTW for shape-based matching.

- **Integration:** Promote `wfa_reranker.py` to production, or implement DTW as primary `BaseDistanceMetric`.
- **Gate:** BSS improvement ≥ +0.003 on ≥ 3/6 folds. Execution within 5-minute window.
- **Diagnostic:** If too slow at 1500T, use reranker approach (DTW on top-50 only). If BSS doesn't improve, shape-based matching may not add value for 7-day horizon.

### 9.6 Enhancement 4: CPOD/EILOF Anomaly Detection Pre-Filter

Flag extreme outlier patterns, slash confidence before evaluation.

- **Integration:** New `SignalFilterBase` subclass, fires before SectorConviction.
- **Gate:** Reduces false positive rate ≥ 5% without reducing true positive rate > 2%.
- **Diagnostic:** If vetoes > 30% of signals, anomaly threshold too aggressive. Tune contamination parameter.

### 9.7 Enhancement 5: Dynamic Feature Weighting (OWA)

Regime-conditional feature weights via Ordered Weighted Averaging.

- **Integration:** Modify `_prepare_features` to accept weight vectors keyed by regime.
- **Gate:** BSS improvement ≥ +0.003 on ≥ 3/6 folds. Must not degrade worst-fold BSS.
- **Diagnostic:** If regime-conditional weights overfit, use leave-one-fold-out CV for weight selection.

### 9.8 Protocol

Implement one at a time in order: BMA → Conformal → DTW → CPOD → OWA. Each gets a walk-forward run. Keep what passes, revert what doesn't. Cumulative BSS tracked.

### 9.9 Fallback

If no enhancement passes, deploy with Phases 1-6 baseline.

### 9.10 Deliverables (per enhancement)

- Implementation behind feature flag
- Walk-forward results with provenance (`results/enhancement_<name>_*.tsv`)
- Gate pass/fail in session log
- If pass: flag set to `True` in production config

---

## 10. Phase 8 — Paper Trading Validation (3 Months)

### 10.1 Goal

Run FPPE live against real market data for 3 months without capital at risk. Validate all v1 success criteria in real-time.

### 10.2 Daily Execution Cycle

| Time (ET) | Action |
|-----------|--------|
| 09:00 AM | OOB reconciliation |
| Market hours | Price data streaming, stop-loss monitoring |
| 4:00 PM | Signal generation → pipeline → mock/paper orders |
| 4:30 PM | SharedState checkpoint, journal entry |

### 10.3 Metrics Tracked Daily

- P&L (gross, net of friction), equity curve
- Trade count, win rate, average hold time
- BSS on rolling 30/90-day windows
- Calibration curve (predicted vs actual, 10 bins)
- Sector attribution (which sectors profitable?)
- Idle cash %
- Execution latency (signal → order submission)
- Slippage tracking (assumed 10bps vs actual)

### 10.4 Weekly Review Protocol

- Generate automated performance report
- Compare to baselines (SPY, random, raw signals)
- Check StrategyEvaluator status (GREEN/YELLOW/RED)
- Review any alerts with root cause
- Document in weekly session log

### 10.5 Month-End Gate Checks

| Month | Focus |
|-------|-------|
| Month 1 | System stability — no crashes, no missed windows, reconciliation clean |
| Month 2 | Performance trending — metrics converging, no regime collapse |
| Month 3 | Full v1 criteria — all thresholds from Section 1.3 evaluated |

### 10.6 Diagnostic Protocol

1. **System crash:** Root cause analysis. Check memory (32GB tight with 1500T HNSW), broker API timeout, uncaught exceptions. Fix, restart, log incident.
2. **BSS turns negative:** Check regime shift. Compare to walk-forward fold matching current regime. Transient if historically positive on that fold.
3. **Calibration drift (bucket diverges > 15pp):** Trigger `drift_monitor.py` CUSUM. If sustained > 2 weeks, re-run Platt calibration.
4. **Slippage > 10bps consistently:** Adjust parameter upward. If > 25bps, consider VWAP execution window.
5. **Idle cash > 50%:** Signal throughput too low. Lower confidence threshold (requires experiment) or add feature dimensions.

### 10.7 Gate (All Must Hold by Month 3)

- Net expectancy per trade > $0
- 70% confidence signals win 65-75%
- Peak-to-trough drawdown ≤ 25%
- Outperforms ≥ 2/3 baselines
- No 30-day window with Sharpe < -1.0
- ≥ 50 trades in 3-month period
- Zero reconciliation failures in final 30 days
- Zero execution window misses in final 30 days

### 10.8 Fallback

- **Minor (single metric fails):** Extend paper trading 1 month, apply targeted fix, revalidate.
- **Major (multiple metrics fail):** Halt, diagnose root cause, may return to Phase 1 or 7. Document thoroughly.

### 10.9 Deliverables

- `scripts/daily_report.py` — automated daily metrics
- `scripts/weekly_review.py` — weekly performance summary
- `docs/paper_trading/` — weekly logs, month-end reports
- Final paper trading report with go/no-go recommendation

---

## 11. Phase 9 — Live Deployment ($10,000)

### 11.1 Goal

Deploy FPPE with real capital. $10,000 starting balance.

### 11.2 Broker Setup

IBKR (or Alpaca fallback) live account. API credentials stored securely (environment variables or secrets manager, not in repo).

### 11.3 Gradual Ramp-Up

| Period | Capital Deployed | Purpose |
|--------|-----------------|---------|
| Week 1-2 | 25% (max 2-3 positions) | Verify real fills match paper assumptions |
| Week 3-4 | 50% | Validate slippage model |
| Month 2+ | 100% | Full deployment if weeks 1-4 clean |

### 11.4 Real Slippage Tracking

Compare every fill price to assumed entry. Log `actual_slippage_bps` per trade. If mean > 15bps, adjust model.

### 11.5 Capital Growth Milestones

| Milestone | Review Action |
|-----------|--------------|
| $10k → $25k | Maintain current parameters |
| $25k → $50k | Review position size impact on fills (market impact) |
| $50k → $100k | Consider more tickers, reduce max position size % |
| $100k+ | Evaluate institutional execution (IBKR Pro, DMA) |

### 11.6 Kill Switch

- **Automated:** RED status from StrategyEvaluator → halt new trades, hold positions for manual review
- **Manual:** Single command to flatten all positions, go to 100% cash

### 11.7 Tax Tracking & Reporting

**Capital Gains Tax Tracker** (`trading_system/tax_tracker.py`):
- Classifies every closed trade as short-term (≤ 365 days) or long-term (> 365 days)
- Tracks realized gains/losses per tax lot (FIFO default, specific-lot identification option)
- Running YTD tallies: short-term gains, long-term gains, total tax liability estimate (configurable federal + state brackets)
- **Wash sale detection:** flags re-entry into same ticker within 30 days of a loss; adjusts cost basis on replacement lot per IRS rules
- **Tax-loss harvesting signals:** when unrealized loss exceeds configurable threshold and no wash sale conflict, emit YELLOW alert to weekly review
- Trade log export compatible with tax software (CSV matching TurboTax/TaxAct import schema)
- Quarterly estimated tax liability report (for estimated tax payment planning)

### 11.8 Diagnostic Protocol

1. **Real fills consistently worse than paper:** Check order type, execution venue, time of day. Consider limit orders at mid-price with 30-second timeout.
2. **Drawdown > 15% in first month:** Reduce to 25% deployment. Investigate: broad market or model-specific. If model-specific, halt and return to paper.
3. **Broker API down during execution:** OOB reconciliation catches next morning. No new trades until reconciliation passes.

### 11.9 Deliverables

- Production deployment runbook (`docs/DEPLOYMENT_RUNBOOK.md`)
- Kill switch script (`scripts/emergency_halt.py`)
- `trading_system/tax_tracker.py` — capital gains tracking with wash sale detection
- Capital ramp-up checklist
- Ongoing performance dashboard

---

## 12. Phase 10 — Options Trading Foundation (Post-Launch Side Track)

### 12.1 Goal

Build structural scaffolding for options trading. Infrastructure only — no live options trading. Creates contracts, data models, and integration points for future development.

### 12.2 Rationale

FPPE's KNN pattern matching produces probability distributions that map naturally to options strategies:
- "72% chance of +3% in 7 days" → directional calls/puts
- Existing equity positions → covered calls for income
- Defined-risk spreads → capital-efficient directional bets
- IV vs predicted RV divergence → volatility strategies

### 12.3 Scope

#### 12.3.1 Options Contract Model (`trading_system/options/contracts.py`)

```python
class OptionContract(BaseModel, frozen=True):
    underlying: str          # ticker
    expiration: date
    strike: float
    option_type: OptionType  # CALL / PUT
    style: OptionStyle       # AMERICAN / EUROPEAN

class OptionQuote(BaseModel, frozen=True):
    contract: OptionContract
    bid: float
    ask: float
    mid: float
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    open_interest: int
    volume: int

class OptionOrder(BaseModel, frozen=True):
    contract: OptionContract
    direction: Direction     # BUY / SELL
    quantity: int
    order_type: OrderType    # MARKET / LIMIT
    limit_price: Optional[float]

class OptionPosition(BaseModel, frozen=True):
    contract: OptionContract
    quantity: int            # positive = long, negative = short
    avg_cost: float
    current_value: float
    unrealized_pnl: float
```

#### 12.3.2 Strategy Templates (`trading_system/options/strategies.py`)

```python
class BaseOptionStrategy(ABC):
    @abstractmethod
    def evaluate(self, signal: UnifiedSignal, chain: OptionChain) -> list[OptionOrder]: ...
    @abstractmethod
    def max_loss(self) -> float: ...
    @abstractmethod
    def max_profit(self) -> Optional[float]: ...  # None = unlimited

# Concrete stubs:
class CoveredCallStrategy(BaseOptionStrategy): ...
class ProtectivePutStrategy(BaseOptionStrategy): ...
class VerticalSpreadStrategy(BaseOptionStrategy): ...
class IronCondorStrategy(BaseOptionStrategy): ...
```

#### 12.3.3 Options Data Interface (`trading_system/options/data.py`)

```python
class BaseOptionsDataProvider(ABC):
    @abstractmethod
    def get_chain(self, ticker: str, min_dte: int, max_dte: int) -> OptionChain: ...
    @abstractmethod
    def get_quote(self, contract: OptionContract) -> OptionQuote: ...
```

#### 12.3.4 Greeks Calculator (`trading_system/options/greeks.py`)

- Black-Scholes pricing for European options
- Binomial tree for American options
- Implied volatility solver (Newton-Raphson)
- Portfolio-level Greeks aggregation (net delta, gamma, theta, vega)

#### 12.3.5 Signal-to-Options Bridge (`trading_system/options/signal_bridge.py`)

| Signal Condition | Options Strategy |
|-----------------|-----------------|
| High-confidence BUY (> 0.70) + low IV rank | Long call or bull call spread |
| Moderate BUY (0.55-0.70) + existing position | Covered call |
| High-confidence + high IV | Sell put spread (collect premium) |
| HOLD on existing + high IV | Covered call for income |

#### 12.3.6 SharedState Extension

```python
class OptionsState(BaseModel, frozen=True):
    positions: dict[str, list[OptionPosition]]  # ticker → positions
    portfolio_greeks: PortfolioGreeks
    total_options_exposure: float
    margin_used: float
```

#### 12.3.7 Risk Integration Points

- Options positions feed into existing drawdown calculation
- Portfolio-level delta exposure tracked alongside equity exposure
- Options margin requirements subtracted from available capital

### 12.4 Not In Scope (Deferred)

- Live options data feed integration
- Options backtesting engine
- Real options order execution
- Complex multi-leg strategy optimization
- Volatility surface modeling
- Greeks-based hedging automation

### 12.5 Gate

Code quality gate (structural phase, not performance):
- All contracts have Pydantic validation with tests
- ABCs implemented with at least one concrete stub each
- Greeks calculator verified against published pricing examples
- Signal bridge mapping logic has unit tests
- SharedState extension doesn't break existing test suite

### 12.6 Deliverables

- `trading_system/options/` — contracts, strategies, data, greeks, signal_bridge
- `tests/unit/test_options_contracts.py`
- `tests/unit/test_greeks.py`
- `tests/unit/test_signal_bridge.py`
- `docs/OPTIONS_TRADING_DESIGN.md`

---

## 13. Locked Settings Reference

Settings locked by experiment evidence. Do NOT change without new walk-forward results.

| Parameter | Value | Source |
|-----------|-------|--------|
| Distance | Euclidean (L2) | Phase 1 |
| Weighting | Uniform | Phase 1 |
| Features | VOL_NORM_COLS (8) | M9 |
| Calibration | Platt | SLE-89 |
| cal_frac | 0.76 | M1 |
| max_distance | 1.1019 (may change Phase 1) | 2026-03-21 sweep |
| top_k | 50 (may change Phase 1) | Phase 1 |
| confidence_threshold | 0.65 locked / 0.55 operating | 2026-03-26 |
| horizon | fwd_7d_up | Phase 1 |
| stop_loss_atr_multiple | 3.0× | 2026-03-21 ATR sweep |
| min_sector_lift | 0.005 | 2026-03-26 recalibration |
| nn_jobs | 1 | Windows/Py3.12 deadlock prevention |
| HNSW recall@50 | ≥ 0.9999 | Accuracy requirement |

---

## 14. Risk Register

| Risk | Phase | Likelihood | Impact | Mitigation |
|------|-------|-----------|--------|------------|
| BSS stays negative after all H1-H4 | 1 | Medium | Critical | Fall back to 52T; BMA rescue path |
| Kelly fraction too small to matter | 2 | Low | Medium | Revert to ATR-only sizing |
| Analogue dilution recurs at 1500T | 6 | High | High | Same fix as Phase 1; fall back to 585T |
| HNSW recall < 0.9999 at scale | 6 | Low | High | Increase ef_search; BallTree fallback |
| Broker API instability | 5, 9 | Medium | Medium | Alpaca fallback; retry with backoff |
| Regime shift during paper trading | 8 | Medium | Medium | Compare to historical fold; transient check |
| Slippage exceeds model | 8, 9 | Medium | Low | Adjust parameter; VWAP window |
| Memory limit (32GB) at 1500T | 6, 8 | Low | Medium | Sector pre-filtering; chunked processing |
| Tax lot tracking edge cases | 9 | Low | Low | Manual override + tax advisor review |

---

## 15. Environment & Constraints

- **Hardware:** Windows 11, Ryzen 9 5900X (12 cores), 32GB RAM
- **Python:** 3.12, venv at `C:\Users\Isaia\.claude\financial-research\venv`
- **nn_jobs=1:** Always. Prevents Windows/Py3.12 joblib deadlock.
- **Test command:** `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
- **Current test count:** 616 passed, 1 skipped
- **`prepare.py`:** Locked — do not modify without explicit request
- **`assert` → `RuntimeError`:** All public API guards use RuntimeError, not assert

---

## Appendix A: Existing Module Map

### Pattern Engine (`pattern_engine/`)
- `matcher.py` — 5-stage PatternMatcher (857 LOC)
- `contracts/` — Pydantic schemas, BaseMatcher ABC, EngineState
- `features.py`, `data.py`, `schema.py` — feature pipeline
- `regime.py` — RegimeLabeler (SPY + VIX + yield curve)
- `live.py` — LiveRunner + MockBrokerAdapter
- `sector_conviction.py`, `momentum_signal.py`, `sentiment_veto.py` — signal filters
- `signal_pipeline.py` — filter chain orchestrator
- Research pilots: `sax_filter.py`, `wfa_reranker.py`, `ib_compression.py`, `conformal_hooks.py`

### Trading System (`trading_system/`)
- `contracts/state.py` — SharedState (frozen, inter-layer bus)
- `contracts/decisions.py` — EvaluatorStatus, PositionDecision, AllocationDecision
- `contracts/trades.py` — TradeEvent
- `signal_adapter.py` — UnifiedSignal (KNN/DL adapters)
- `strategy_evaluator.py` — rolling metrics, RED/YELLOW/GREEN
- `drift_monitor.py` — CUSUM + EWMA drift detection
- `risk_overlays/` — FatigueAccumulation, LiquidityCongestion
- `shared_state.py` — layer isolation, command queue
- `config.py` — TradingConfig + ResearchFlagsConfig

### Research (`research/`)
- `hnsw_distance.py` — HNSWIndex (54.5× speedup, recall@50=0.9996)
- `bma_calibrator.py` — Bayesian Model Averaging (not integrated)
- `emd_distance.py` — Earth Mover's Distance stub
- `slip_deficit.py` — slippage deficit tracking

### Tests (`tests/`)
- 31 test files, 616 tests
- `unit/` — contracts, matchers, data, features, filters, overlays, drift
- `parity/` — BallTree vs HNSW recall
- `regression/` — Sharpe formula, output invariants, snapshot
- `performance/` — HNSW benchmarks

---

## Appendix B: Key Design Documents

- `docs/FPPE_TRADING_SYSTEM_DESIGN.md` v0.5 — 4-layer architecture
- `docs/campaigns/PHASE_3Z_CAMPAIGN.md` — full rebuild history (SLE-51–89)
- `docs/CANDLESTICK_CATEGORIZATION_DESIGN.md` v0.2 — future Phase 6
- `confidence_improvements.md` — BMA/conformal research outline
- `next_steps_plan.md` — original roadmap to live
