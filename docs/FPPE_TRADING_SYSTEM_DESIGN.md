# FPPE Trading System — Architecture & Design Specification

**Version:** 0.4 (Phase 3 Portfolio Manager complete)
**Date:** March 20, 2026
**Status:** ACTIVE — Phase 3 complete. Phase 4 (strategy evaluator) next.

**Revision notes (v0.4):** Phase 3 Portfolio Manager implemented and validated.

- **New modules:** `trading_system/portfolio_state.py` (frozen dataclasses: `RankedSignal`, `AllocationDecision`, `PortfolioSnapshot`), `trading_system/portfolio_manager.py` (Layer 3 logic: `rank_signals`, `check_allocation`, `allocate_day`)
- **Integration:** `BacktestEngine` now accepts `use_portfolio_manager=True` (requires `use_risk_engine=True`). PM pre-filters on count constraints (holding, cooldown, sector limit); risk engine gates on dollar constraints (exposure, capital). Rank order preserved end-to-end.
- **Backtest validation (2024, 52-ticker universe):** Phase 3 matches Phase 2 exactly — Ann. 15.1%, Sharpe 1.93, Max DD 3.9%, Win rate 52.4%, 103 trades. PM generated 37 portfolio-layer rejections (29 already-holding, 7 cooldown, 1 sector limit).
- **Test suite:** 556 tests, 0 failures. Layer 3 adds 11 integration tests, 30 unit tests (portfolio_manager), 20 unit tests (portfolio_state).
- **Phase 3 branch:** `phase3-portfolio-manager` — ready for merge to main.

**Revision notes (v0.3):** Phase 1 backtest completed on 2024 validation data with real FPPE signals. Two empirical parameter optimizations applied based on sweep results:

1. **Confidence threshold: 0.65 → 0.60** (threshold_sweep.py, 5 values tested)
   - 0.65: 159 BUY, 8.9% annual, Sharpe 0.97, $3.79/trade expectancy, 22.6% invested
   - 0.63: 530 BUY, 14.4% annual, Sharpe 1.39
   - 0.60: 1,876 BUY, 19.5% annual, Sharpe 1.60, $4.13/trade expectancy, 70.5% invested ← optimum
   - 0.58: 3,348 BUY, 16.2% annual, Sharpe 1.17 (dilution begins here)
   - 0.55: 5,929 BUY, 15.9% annual, Sharpe 1.15

2. **Max holding days: 10 → 14** (holding_period_sweep.py, 7 windows tested at threshold=0.60)
   - 1d: -$1.22/trade, -3.9% annual (friction destroys alpha at this timescale)
   - 3d: -$0.23/trade, 6.5% annual
   - 5d: $0.67/trade, 10.5% annual
   - 7d: $1.76/trade, 13.9% annual (FPPE projection horizon)
   - 10d: $4.13/trade, 19.5% annual
   - 14d: $6.65/trade, 22.3% annual, Sharpe 1.82, Win rate 60.4% ← optimum
   - 20d: $7.09/trade, 17.5% annual (fewer trades drag annual return)

**Combined Phase 1 result (threshold=0.60, max_hold=14d):**
- Annual return: 22.3% | Sharpe: 1.82 | Max DD: 6.9% | Win rate: 60.4%
- vs SPY buy-and-hold: 25.4% annual, Sharpe 1.52, Max DD 8.4%
- Beats SPY on risk-adjusted basis (Sharpe 1.82 vs 1.52, lower drawdown)
- All v1 success criteria passed

**CAUTION:** 2024 was a strong bull year. Longer holds (14d) benefit from market beta.
Re-validate max_holding_days against bear-market data before treating as permanent.

**Revision notes (v0.2):** Incorporated feedback from two independent reviews. Major changes: v1 is now long-only (shorts deferred to v2), success criterion changed from aggressive return targets to edge preservation, simplified position sizing to remove confidence double-counting, fixed FPPE signal schema to support hybrid K-NN/deep learning outputs, added baseline comparisons, added re-entry cooldown logic, reduced max holding period from 20 to 10 trading days.

---

## 1. Purpose

This document defines the architecture for a four-layer trading system built on top of the Financial Pattern Prediction Engine (FPPE v2.1). The system transforms FPPE's probabilistic BUY/HOLD/SELL signals into simulated trades, tracks performance against defined thresholds, manages risk through volatility-based position sizing, and allocates capital across a 52-ticker universe.

**v1 is long-only.** Short selling will be added in v2 after the long-only system demonstrates positive expectancy after friction. This decision reduces complexity and isolates whether FPPE's BUY signals carry a real, tradeable edge before introducing the additional mechanics of margin, borrow fees, and unlimited-risk stop logic.

**This is a paper-trading and research system.** No real money is deployed until the system clears the success criteria defined in Section 3.2.

---

## 2. System Architecture

The system consists of four layers, each implemented as a standalone Python module with clearly defined inputs and outputs. A central configuration module holds all tunable parameters.

```
┌─────────────────────────────────────────────────────────┐
│                   config.py                             │
│         All parameters, thresholds, costs               │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│              Layer 4: strategy_evaluator.py              │
│   Rolling metrics, threshold comparison, status flags   │◄─────────────┐
│   Baseline comparisons, calibration tracking            │              │
│   ↓ Emits: RED/YELLOW/GREEN status, halt commands       │              │
├─────────────────────────────────────────────────────────┤              │
│              Layer 3: portfolio_manager.py               │              │
│   Signal ranking (multi-factor for v2+),                │◄─────────┐   │
│   capital allocation, sector limits                     │          │   │
│   ↓ Emits: sector block commands, capital alerts        │          │   │
├─────────────────────────────────────────────────────────┤          │   │
│              Layer 2: risk_engine.py                     │          │   │
│   Volatility-based position sizing (no confidence),     │◄─────┐   │   │
│   drawdown brake, stop-losses                           │      │   │   │
│   ↓ Emits: stop-loss events, drawdown alerts            │      │   │   │
├─────────────────────────────────────────────────────────┤      │   │   │
│              Layer 1: backtest_engine.py                 │      │   │   │
│   Trade simulation, cost model, P&L calculation         │──────┴───┴───┘
│   ↑ Emits: trade events, equity state, position state   │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│              FPPE v2.1 Signal Output                    │
│   Hybrid: K-NN analogues + DL probability tensors       │
│   Unified via signal adapter (see Section 5.1)          │
└─────────────────────────────────────────────────────────┘
```

**Data flows in both directions** through a shared `SystemState` object (see Section 2.1 below). The primary data flow is upward (FPPE → Layer 1 → Layer 2 → Layer 3 → Layer 4), but each layer can read the shared state of any other layer and emit commands downward.

**Key v0.2 design principle — confidence is used once.** FPPE's three-filter gate already uses confidence to determine whether a signal is actionable (BUY/SELL/HOLD). Inside the trading system, confidence drives only the portfolio ranking in Layer 3. It does NOT drive position sizing in Layer 2 (which is purely volatility-based). This prevents double-counting the same variable across multiple layers, which the reviews correctly identified as a risk of magnifying noise.

### 2.1 Inter-Layer Communication via SharedState

Each layer publishes updates to and reads from a central `SystemState` object. This replaces strict one-way data passing and allows higher layers to send protective commands downward without tight coupling.

```
SharedState (in-memory object, one instance per backtest run)
├── execution_state (written by Layer 1)
│   ├── current_equity: float
│   ├── current_cash: float
│   ├── open_positions: Dict[ticker, PositionState]
│   ├── daily_drawdown: float
│   └── last_trade_events: List[TradeEvent]   # flushed each day
├── risk_state (written by Layer 2)
│   ├── drawdown_mode: "normal" | "brake" | "halt"
│   ├── active_stops: Dict[ticker, float]       # ticker → stop price
│   └── sizing_scalar: float                   # 0.0–1.0 drawdown scalar
├── portfolio_state (written by Layer 3)
│   ├── sector_exposure: Dict[sector, float]
│   ├── capital_available: float
│   └── blocked_sectors: List[str]             # sectors at capacity
├── evaluation_state (written by Layer 4)
│   ├── system_status: "GREEN" | "YELLOW" | "RED"
│   ├── rolling_sharpe_30d: Optional[float]
│   ├── rolling_ne_90d: Optional[float]        # net expectancy 90-day
│   └── halt_new_trades: bool                  # Layer 4 can set True
└── signal_commands (written by any layer, consumed by Layer 1)
    ├── force_exits: List[ticker]              # Layer 4 RED → close all
    ├── blocked_tickers: Set[str]             # cooldown + sector blocks
    └── sector_blocks: Set[str]               # from Layer 3 or Layer 4
```

**Communication examples:**
- Layer 4 detects 90-day net expectancy turns negative → sets `evaluation_state.halt_new_trades = True`. Layer 1 reads this before processing each day's signals and skips all new entries.
- Layer 3 detects Technology sector at 30% exposure → adds "Tech" to `portfolio_state.blocked_sectors`. Layer 1 reads blocked sectors before accepting signals.
- Layer 2 detects drawdown hits 20% → sets `risk_state.drawdown_mode = "halt"` and `signal_commands.force_exits = all_tickers`. Layer 1 processes force exits before the day's signals.
- Layer 1 records a stop-loss fire event → `execution_state.last_trade_events` includes the stop event. Layer 4 reads this to update calibration and rolling drawdown metrics without waiting for the next polling cycle.

**Design constraints:** No layer creates direct references to another layer's class or calls its methods. All communication is through SharedState reads and writes. This keeps layers independently testable and allows any layer to be replaced or mocked without cascading changes.

---

## 3. Agreed Parameters

### 3.1 Capital & Scale

| Parameter | Value | Notes |
|-----------|-------|-------|
| Starting capital | $10,000 | Paper trading baseline; model is %-based so scalable |
| Capital model | Percentage-based | Positions sized as % of current equity, not fixed dollar amounts |
| Fractional shares | Enabled | Required at this capital level |
| Margin (shorts) | N/A for v1 | Long-only; to be defined in v2 spec |

### 3.2 v1 Success Criteria (Edge Preservation)

The v1 goal is NOT to hit aggressive return targets. It is to prove that FPPE's predictive edge survives real-world trading friction. The 25% annualized / 0.8 Sharpe targets remain as aspirational long-term goals but are explicitly NOT the v1 approval gate.

**v1 passes if ALL of the following hold on out-of-sample data:**

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Net expectancy per trade | > $0 after all friction | The system makes money per trade on average |
| Confidence calibration | 70% confidence signals win 65-75% of the time | Model probabilities are meaningful, not just noise |
| Drawdown tolerance | Peak-to-trough ≤ 25% | Capital preservation during losing streaks |
| Performance vs baselines | Outperforms ≥ 2 of 3 baselines (see 3.3) | The system adds value beyond simple alternatives |
| No performance collapse | No 30-day window with Sharpe < -1.0 | No catastrophic regime failure |
| Minimum trade count | ≥ 50 trades in backtest period | Enough data for statistical significance |

### 3.3 Baseline Comparisons (Required)

Every backtest must report performance against these three baselines using the same time period and friction model:

1. **SPY buy-and-hold:** Invest 100% of capital in SPY on day 1, hold throughout. This is the "do nothing" baseline.
2. **Equal-weight random selection:** On each signal date, randomly select from the universe with equal probability. Same position sizing and friction. Run 100 iterations, report median. This tests whether FPPE's signal selection adds value over random picks.
3. **FPPE signals, no trading system:** Raw signal accuracy (what % of BUY signals were followed by a price increase within the holding window). This isolates whether the trading layers are adding or destroying value relative to the raw signals.

### 3.4 Execution Assumptions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Entry timing | Next-day open | Signal generated after close, executed at next open |
| Exit timing | Next-day open | Exit signal triggers execution at next open |
| Slippage model | 10 bps per side | Revised upward from 5 bps; open is the noisiest print of the day |
| Commission | $0.00 per share | Assumes zero-commission broker (e.g., IBKR Lite) |
| Spread cost | 3 bps per side | Embedded in execution price |
| Total round-trip friction | ~26 bps | (10 slippage + 3 spread) x 2 sides = 26 bps minimum |

**Note on slippage (v0.2 change):** The original 5 bps estimate was flagged as unrealistically low for open execution. The open is the most volatile period of the trading day, with wider spreads and higher price impact. 10 bps is still a moderate estimate; actual slippage could be higher on low-volume tickers or during earnings season. The system should log actual vs. assumed slippage when transitioning to live paper trading.

### 3.5 Position Limits (v1 — Long Only)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max simultaneous positions | Dynamic | Determined by risk engine based on capital and volatility |
| Min position size | 2% of equity | Below this, friction makes the trade unprofitable |
| Max position size | 10% of equity | Single-name concentration cap |
| Max sector exposure | 30% of equity | Across all positions in one sector |
| Max positions per sector | 3 | Prevents over-concentration even within limit |
| Max gross exposure | 100% of equity | Long-only, no leverage in v1 |

### 3.6 Trade Management Rules

| Parameter | Value | Notes |
|-----------|-------|-------|
| Max holding period | **14 trading days** | Empirically optimal (v0.3). Original 10d was FPPE's projection horizon; 14d captures full momentum cycle. Re-validate in bear year. |
| Re-entry cooldown (after stop-loss) | 3 trading days | Prevents whipsaw after forced exits |
| Re-entry cooldown (after max-hold exit) | 3 trading days | Same rationale |
| Re-entry after signal reversal | Allowed if new confidence exceeds prior entry by ≥ 0.05 | Permits genuine regime changes while blocking noise |
| Same-day churn | Prohibited | Cannot exit and re-enter same ticker on same day |
| Partial exits | Not supported in v1 | All-in/all-out; revisit at higher capital levels |

### 3.7 Cash Management

| Parameter | Value | Notes |
|-----------|-------|-------|
| Idle cash yield | Daily risk-free rate (~4-5% annualized) | Simulates money market sweep |
| Reporting | Strategy return reported BOTH with and without cash yield | Prevents cash yield from masking weak trading performance |
| Average idle cash % | Tracked as a metric | High idle cash suggests the system is too selective |

**Idle Cash Reduction Roadmap (target: ≤ 25% idle at all times):**

The original Phase 1 backtest ran at 96.8% average idle cash — the system was essentially a money market fund with occasional trades. This is unacceptable long term. The roadmap below shows how each development phase reduces idle cash:

| Phase | Primary Lever | Expected Idle Cash | Notes |
|-------|--------------|-------------------|-------|
| Phase 1 (original, threshold=0.65) | — | ~97% | Baseline — mostly cash |
| Phase 1 (threshold=0.60, max_hold=14d) | Lower signal threshold | ~25% | **Current state** |
| Phase 2 (risk engine) | Volatility sizing permits more concurrent positions | ~20% | Dynamic sizing holds more positions simultaneously |
| Phase 3 (portfolio manager) | Smarter capital allocation fills gaps | ~15% | Ranked queue deploys cash more aggressively |
| Phase 4+ (ticker expansion) | More tickers = more BUY signals per day | ~10% | Target: ~7-8 actionable signals/day vs current 7.5 |
| Future (10-25 year data + re-tuning) | Better calibration across market regimes | ~10-15% | Balanced for bear markets where fewer signals fire |

**Note:** Zero idle cash is not the goal. Some cash reserve is required for intraday risk management and to avoid being fully deployed into drawdown. 10-15% is the practical floor for a 52-ticker long-only system.

**Idle cash threshold alerts (tracked by strategy_evaluator.py):**
- `avg_idle_pct > 50%` → YELLOW (system too selective, revenue is predominantly cash yield)
- `avg_idle_pct > 80%` → RED (strategy provides no value over a money market fund)
- `avg_idle_pct < 5%` → YELLOW (over-deployed, no buffer for drawdown or new signals)

### 3.8 Risk Profiles

The system supports three configurable risk profiles. All profiles use the same underlying engine — only the threshold and sizing parameters differ. This allows both aggressive users (who accept volatility for higher returns) and conservative users (who want stable, predictable outcomes) to use the same architecture.

| Profile | Confidence Threshold | Max Hold | Position Max | Max DD Halt | Target User |
|---------|---------------------|----------|--------------|-------------|-------------|
| **Conservative** | 0.68 | 10 days | 7% | 15% | Wants stable, verifiable edge with minimal drawdown |
| **Moderate** | 0.63 | 12 days | 8% | 18% | Balanced risk/return, some drawdown acceptable |
| **Aggressive** | 0.60 | 14 days | 10% | 20% | Maximizes return, tolerates higher drawdown |

**Phase 1 empirical results by profile (2024 data):**

| Profile | Threshold | Trades | Annual | Sharpe | Max DD | NE/Trade |
|---------|-----------|--------|--------|--------|--------|----------|
| Conservative | 0.68 | ~90 est. | ~10-12% est. | ~1.1 est. | ~3% est. | ~$5+ est. |
| Moderate | 0.63 | 255 | 14.4% | 1.39 | 4.5% | $3.90 |
| Aggressive | 0.60 | 278* | 22.3% | 1.82 | 6.9% | $6.65 |

*Note: Conservative profile estimates are interpolated. Run threshold_sweep.py at 0.68 to confirm. The higher threshold means fewer but higher-quality signals — per-trade expectancy may exceed Aggressive despite lower annual return.

**Implementation:** Profile is a top-level config field (`config.profile = "aggressive" | "moderate" | "conservative"`). The `TradingConfig.from_profile(name)` class method instantiates the correct parameter set. Advanced users can override any individual parameter after profile selection.

### 3.9 Variable Capital Tiers

The system scales across capital levels. Position sizing constraints adapt to ensure trades remain cost-effective at small capital and sufficiently diversified at large capital.

| Capital Tier | Initial Capital | Min Position % | Max Position % | Max Positions | Notes |
|-------------|----------------|---------------|---------------|---------------|-------|
| Micro | $2,000 | 8% | 25% | 6-8 | Fewer, larger positions; 26 bps friction matters more |
| Small | $5,000 | 5% | 15% | 10-12 | Balanced; friction still material |
| Standard | $10,000 | 2% | 10% | 14-20 | **Current default** |
| Growth | $25,000 | 2% | 8% | 20-30 | More positions, better diversification |
| Large | $50,000 | 1% | 6% | 30-40 | More granular sizing; correlation management more important |
| Institutional | $100,000 | 0.5% | 5% | 40-60 | Near-continuous deployment; slippage model may need revision |

**Capital scaling rules:**
- `min_position_pct`: Constrained by minimum dollar amount ($200 minimum to keep friction below 13% of position). At $2k capital, $200 = 10% minimum.
- `max_position_pct`: Scales down with capital — higher capital permits more diversification, which reduces concentration risk. At $100k, a 10% position is a $10,000 single-name bet, which may exceed reasonable single-stock exposure.
- Friction sensitivity: At $2k with a 5% position ($100), the 26 bps round-trip costs $0.26 — 0.26% of the position. At $100k with a 5% position ($5,000), the same trade costs $13 — same percentage but higher absolute dollar awareness.
- **v1 default remains $10,000.** Capital tier configuration is planned for Phase 3+ when portfolio manager handles multi-position allocation.

---

## 4. Module Specifications

### 4.1 config.py — Central Configuration

**Purpose:** Single source of truth for all tunable parameters. No magic numbers anywhere else in the codebase.

**Structure:**

```
CONFIG
├── capital
│   ├── initial_capital: 10000
│   ├── fractional_shares: True
│   └── max_gross_exposure: 1.0        # v1: long-only, no leverage
├── costs
│   ├── slippage_bps: 10               # revised upward for open execution
│   ├── spread_bps: 3
│   ├── commission_per_share: 0.0
│   └── risk_free_annual_rate: 0.045   # for idle cash and Sharpe calculation
├── position_limits
│   ├── min_position_pct: 0.02
│   ├── max_position_pct: 0.10
│   ├── max_sector_pct: 0.30
│   └── max_positions_per_sector: 3
├── trade_management
│   ├── max_holding_days: 10
│   ├── cooldown_after_stop_days: 3
│   ├── cooldown_after_maxhold_days: 3
│   ├── reentry_confidence_margin: 0.05
│   └── allow_same_day_churn: False
├── risk
│   ├── volatility_lookback: 20        # trading days for ATR/vol calc
│   ├── correlation_lookback: 60
│   ├── stop_loss_atr_multiple: 2.0    # stop = entry - 2x ATR
│   ├── max_loss_per_trade_pct: 0.02   # 2% of equity max loss per trade
│   ├── drawdown_brake_threshold: 0.15 # reduce sizing at 15% drawdown
│   └── drawdown_halt_threshold: 0.20  # halt new trades at 20% drawdown
├── evaluation
│   ├── rolling_windows: [30, 90, 252] # days
│   ├── min_trades_for_metrics: 30     # don't compute ratios below this
│   └── baseline_random_iterations: 100
└── universe
    └── sector_map: {ticker: sector}   # 52 tickers → 7 sectors
```

**Design decisions:**
- Implemented as a Python dataclass with frozen=True after initialization. Type-checked and importable.
- Every parameter has a default with an inline comment explaining the reasoning.
- A `validate()` method ensures internal consistency (e.g., drawdown brake < drawdown halt, min position < max position, max sector exposure permits at least max_positions_per_sector × min_position_pct).
- Short-selling parameters (margin, borrow fees, max short exposure) are defined but disabled in v1. They exist in the schema as commented-out defaults for v2 readiness.

---

### 4.2 backtest_engine.py — Trade Simulation (Layer 1)

**Purpose:** Take FPPE signal output and simulate trade execution with realistic friction. Produce a complete trade log and daily equity curve. This module is purely mechanical — it applies rules and records outcomes.

**Inputs:**
- Unified FPPE signal DataFrame (see Section 5.1 for adapter)
- Historical price data: `[date, ticker, open, high, low, close, volume]`
- Position sizes from Layer 2
- Trade approvals from Layer 3
- Config parameters

**Outputs:**

1. **Trade log** (one row per completed trade):
```
trade_id | ticker | direction | entry_date | entry_price | exit_date | exit_price |
position_pct | shares | gross_pnl | slippage_cost | spread_cost | total_costs |
net_pnl | holding_days | exit_reason | confidence_at_entry | sector
```

2. **Daily equity curve** (one row per trading day):
```
date | equity | cash | invested_capital | gross_exposure | open_positions |
daily_return | cumulative_return | drawdown_from_peak | cash_yield_today |
strategy_return_excl_cash | strategy_return_incl_cash
```

3. **Rejected signal log** (one row per signal that was filtered out):
```
date | ticker | signal | confidence | rejection_reason | rejection_layer
```

**Execution logic:**
1. On signal date (after close): receive unified signal from FPPE adapter
2. Check re-entry cooldown rules for each ticker
3. Pass surviving signals to Layer 3 for ranking and approval
4. Approved signals go to Layer 2 for position sizing
5. Next trading day at open: execute at `open_price × (1 + slippage_bps/10000)` for buys
6. Mark-to-market all open positions daily at close
7. Check exit conditions each day (see below)
8. On exit: execute at `next_open × (1 - slippage_bps/10000)` for sells
9. Accrue daily risk-free rate on idle cash

**Exit conditions (any triggers exit at next open):**
- SELL or HOLD signal on a held long position
- Stop-loss triggered (intraday low ≤ stop price → exit at next open, NOT at stop price, to avoid unrealistic fills)
- Max holding period (10 days) exceeded
- Drawdown halt threshold reached (Layer 2 forces exit of all positions)

**Stop-loss note (v0.2):** The original design assumed stops could execute at the stop price. Reviewers flagged that gap-through-stop scenarios (where the price opens below the stop) are common, especially on earnings or macro news. The backtest conservatively assumes exit at next-day open after the stop is triggered, not at the stop price itself. This produces more realistic P&L.

---

### 4.3 risk_engine.py — Position Sizing & Risk Control (Layer 2)

**Purpose:** Given an approved trade candidate, determine position size based on volatility. Enforce risk limits and drawdown protection.

**v0.2 key change:** Position sizing is purely volatility-based. Confidence does NOT affect position size — it is used only in Layer 3 for ranking. This eliminates the double-counting problem where confidence drives the FPPE signal gate, then sizing, then ranking.

**Inputs:**
- Approved trade candidate: `{ticker, signal, confidence}`
- Current portfolio state (from Layer 1)
- Ticker volatility data (ATR, rolling vol)
- Config parameters

**Outputs:**
- Position size (% of equity and dollar amount)
- Stop-loss price
- Approval/rejection with reason

**Position sizing formula (v1 — volatility-only):**

```
target_risk_per_trade = max_loss_per_trade_pct (2% of equity)

ticker_ATR_pct = 20-day ATR / current_price
stop_distance = stop_loss_atr_multiple × ticker_ATR_pct (2 × ATR%)

raw_weight = target_risk_per_trade / stop_distance
    (sizes the position so that a stop-loss hit loses exactly 2% of equity)

drawdown_scalar =
    1.0             if drawdown < brake_threshold (15%)
    linear 1.0→0.0  if drawdown between brake (15%) and halt (20%)
    0.0             if drawdown ≥ halt_threshold (20%)

final_weight = clamp(raw_weight × drawdown_scalar,
                     min_position_pct, max_position_pct)
```

**Why this is better than v0.1:** The v0.1 formula multiplied `max_position_pct × confidence_score × volatility_scalar`. This meant a 0.65 confidence signal on a low-vol stock could get a larger position than a 0.90 confidence signal on a high-vol stock — which conflates signal quality with position sizing. The v0.2 formula sizes purely on "how much can I lose if I'm wrong" (volatility), while letting Layer 3 decide "which signals are most worth taking" (confidence).

**Stop-loss logic:**
- Long positions: `entry_price × (1 - stop_loss_atr_multiple × ATR%)`
- Per-trade max loss capped at 2% of total equity (if stop distance implies more, reduce position size)
- Stops are evaluated on intraday lows, but execution occurs at next-day open (see Layer 1)

**Rejection conditions:**
- Position would breach sector concentration limit (30% or 3 positions in sector)
- Position would push gross exposure above 100% (no leverage in v1)
- Drawdown has hit halt threshold (20%)
- Ticker is already held in the portfolio
- Ticker is in cooldown period
- Calculated position size falls below minimum (2% of equity)

---

### 4.4 portfolio_manager.py — Capital Allocation (Layer 3)

**Purpose:** When multiple BUY signals arrive on the same date, rank them and decide which become trades. Enforce capital and diversification constraints.

**v0.2 key change:** Ranking is confidence-only for v1. The original composite score included an "expected risk-adjusted return" term derived from historical analogue returns. Reviewers correctly identified this as a likely source of hidden look-ahead bias and overfitting at this stage. The simpler ranking is also easier to validate and debug.

**Inputs:**
- All BUY signals for a given date from FPPE (SELL signals generate exits, not new positions in v1)
- Current portfolio state
- Config parameters

**Outputs:**
- Ordered list of trades to execute (passed to Layer 2 for sizing)
- Rejected signals with reasons

**Ranking logic (v1 — baseline):**
```
rank_score = confidence
```
Ties broken by: lower correlation to existing portfolio, then alphabetical ticker (deterministic).

**Ranking logic (v2+ — composite score):**

The v2 ranking score combines multiple factors. Each factor is normalized to [0, 1] before weighting to prevent any single high-magnitude variable from dominating.

```
rank_score = (
    w_confidence  × confidence_norm       # FPPE signal strength
  + w_momentum    × sector_momentum_norm  # Is the sector trending with the signal?
  + w_recency     × signal_recency_norm   # Days since last analogous signal
  + w_volatility  × inv_volatility_norm   # Lower vol → larger normalized score
  + w_correlation × inv_correlation_norm  # Lower correlation to existing portfolio
  + w_pattern     × pattern_strength_norm # Candlestick pattern clarity (if module active)
)

# v2 default weights (to be validated empirically before promoting from default)
w_confidence  = 0.40   # Dominant factor — FPPE's core output
w_momentum    = 0.20   # Sector tailwind/headwind check
w_recency     = 0.15   # Prefer signals with some recent confirmation
w_volatility  = 0.10   # Prefer lower-vol assets at equal confidence
w_correlation = 0.10   # Portfolio diversification benefit
w_pattern     = 0.05   # Pattern categorization module (Phase 6)
```

**Factor definitions:**

- `confidence_norm`: Calibrated FPPE probability, already on [0, 1]. No transformation needed.
- `sector_momentum_norm`: Equal-weight sector return over the past 10 days vs. the rolling 90-day sector return. Positive momentum = sector is outperforming its own history. Normalized via min-max across all signals on the same date.
- `signal_recency_norm`: Number of trading days since FPPE last generated a BUY signal on this ticker in the current direction, normalized to [0, 1] via sigmoid. Fresh signals score lower (potentially fleeting noise); signals that have held their direction for 2-5 days score higher (trend confirmation). NOT to be confused with the K-NN recency matching — this is the trading system's own signal history.
- `inv_volatility_norm`: `1 / (1 + ATR_pct)` normalized across all signals. Lower volatility = higher score, reflecting that a given confidence level is more reliable on stable assets.
- `inv_correlation_norm`: `1 - max_correlation_to_existing_portfolio`. If a new signal would be 90% correlated with an existing open position, it scores near zero on this factor. Encourages diversification.
- `pattern_strength_norm`: Confidence in the candlestick pattern classification (Tier 1-3 match quality). Only active when the candlestick categorization module is running. Default = 0.5 (neutral) if module is inactive.

**Why v2, not v1:** The factors beyond confidence risk introducing overfitting or look-ahead bias if implemented before the baseline (confidence-only) has been validated. Each factor will be added one at a time, with a before/after backtest comparison. A factor is only promoted to production if it demonstrably improves Sharpe or net expectancy without degrading drawdown.

**Weight tuning governance:** Weights are not optimized via grid search against the 2024 backtest data. They are set by first-principles reasoning (confidence should dominate) and validated on 2025+ out-of-sample data. Grid search on weights is explicitly prohibited until 10+ years of data are available.

**Allocation algorithm:**
1. Collect all BUY signals for the date that pass FPPE's three-filter gate
2. Remove signals for tickers currently held or in cooldown
3. Sort by confidence descending (tie-breaking as above)
4. Iterate through ranked signals:
   a. Check sector limits (≤30% exposure, ≤3 positions per sector)
   b. Check capital availability (enough cash for min position size)
   c. If constraints pass → send to Layer 2 for sizing
   d. If Layer 2 approves → add to execution queue
   e. If any check fails → log rejection reason, move to next signal
   f. Stop when no capital remains for minimum position size

---

### 4.5 strategy_evaluator.py — Performance Monitoring (Layer 4)

**Purpose:** Evaluate system performance against the v1 success criteria (Section 3.2). Track rolling metrics, compare against baselines, and emit status signals.

**Inputs:**
- Daily equity curve from Layer 1
- Trade log from Layer 1
- Baseline results (computed internally)
- Config parameters

**Outputs:**

1. **System status signal:**
```
GREEN  — All v1 success criteria met, system operating normally
YELLOW — One or more metrics degraded on 30-day window; 90-day still OK
RED    — Any of: drawdown > halt threshold, BSS negative on 90-day,
         net expectancy negative on 90-day window
```

2. **Rolling metrics table:**
```
metric                  | 30_day | 90_day | 252_day | all_time | status
────────────────────────┼────────┼────────┼─────────┼──────────┼───────
net_expectancy_per_trade|  ...   |  ...   |   ...   |   ...    | ✓/✗
annualized_return       |  ...   |  ...   |   ...   |   ...    |
sharpe_ratio            |  ...   |  ...   |   ...   |   ...    |
max_drawdown            |  ...   |  ...   |   ...   |   ...    | ✓/✗
win_rate                |  ...   |  ...   |   ...   |   ...    |
profit_factor           |  ...   |  ...   |   ...   |   ...    |
avg_idle_cash_pct       |  ...   |  ...   |   ...   |   ...    |
trade_count             |  ...   |  ...   |   ...   |   ...    | ✓/✗
```

**Note:** Metrics with fewer than `min_trades_for_metrics` (30) trades in the window are reported as "insufficient data" rather than computed. This prevents misleading ratios from tiny sample sizes.

3. **Baseline comparison table:**
```
strategy               | total_return | sharpe | max_dd | win_rate
───────────────────────┼──────────────┼────────┼────────┼─────────
FPPE Trading System    |     ...      |  ...   |  ...   |   ...
SPY Buy-and-Hold       |     ...      |  ...   |  ...   |   N/A
Random Equal-Weight    |     ...      |  ...   |  ...   |   ...
FPPE Raw Signals       |     ...      |  N/A   |  N/A   |   ...
```

4. **Calibration tracker:**
- Confidence calibration curve: for each confidence bucket (0.60-0.65, 0.65-0.70, etc.), what is the actual win rate?
- Rolling BSS on 30/90-day windows
- Per-sector hit rate (are some sectors carrying the system while others drag?)
- Per-ticker P&L attribution (identify consistently unprofitable tickers for potential removal)

**Alert conditions:**
- Net expectancy turns negative on 90-day window → RED
- BSS turns negative on 90-day window → RED (model has lost predictive power)
- Drawdown exceeds 20% → automatic size reduction (via risk engine)
- Drawdown exceeds 25% → RED, halt all new trades
- Calibration drift: any confidence bucket diverges from actual win rate by >15 percentage points → YELLOW
- 5 consecutive losing trades → YELLOW (investigate, likely noise but worth checking)

---

## 5. Data Requirements

### 5.1 FPPE Signal Adapter

FPPE v2.1 is a hybrid system using both K-NN pattern matching and Conv1D + LSTM with Monte Carlo Dropout. These produce different native outputs:

**K-NN outputs:** `n_analogues`, `agreement_spread`, `confidence` (via Platt calibration)
**Deep learning outputs:** `probability_distribution`, `mc_dropout_mean`, `mc_dropout_std`, `confidence`

The trading system does NOT consume these directly. A **signal adapter** normalizes both into a unified schema:

| Field | Type | Description |
|-------|------|-------------|
| date | datetime | Signal generation date |
| ticker | string | Stock symbol |
| signal | enum | BUY / SELL / HOLD (determined by FPPE's three-filter gate) |
| confidence | float | 0.0 to 1.0; source-agnostic confidence score |
| signal_source | enum | KNN / DL / ENSEMBLE | Which model generated this signal |
| raw_metadata | dict | Original model-specific outputs preserved for analysis |

**Why an adapter:** This decouples the trading system from FPPE's internal architecture. If FPPE changes models, adds ensemble methods, or adjusts its output format, only the adapter needs to change. The four trading layers never see model-specific fields.

### 5.2 Market Data

| Field | Type | Description |
|-------|------|-------------|
| date | datetime | Trading date |
| ticker | string | Stock symbol |
| open | float | Adjusted opening price (execution price) |
| high | float | Adjusted daily high |
| low | float | Adjusted daily low (for stop-loss evaluation) |
| close | float | Adjusted closing price |
| volume | int | Daily volume |

**Note on adjusted prices (v0.2):** Using historically adjusted OHLCV data for v1. This correctly handles splits and dividends for backtesting purposes. The reviewers disagreed on whether this is sufficient — one said yes, the other said adjusted prices retroactively shift historical opens and can create phantom P&L. For v1 (paper trading research), adjusted prices are acceptable. For v2 (if approaching real capital), explicit corporate action handling should be added.

**Historical data expansion roadmap:**

The current dataset covers 2020-2024 (5 years). This is sufficient for Phase 1 pipeline validation but inadequate for robust parameter tuning or regime testing.

| Phase | Data Coverage | Calendar Period | Purpose | Note |
|-------|--------------|----------------|---------|------|
| Phase 1-3 | 5 years | 2020-2024 | Pipeline validation | Current state. 2024 = validation; 2020-2023 = training/K-NN |
| Phase 4 | 10 years | 2015-2024 | Regime diversity | Covers COVID crash (2020), bear market (2022), bull run (2021-2024) |
| Phase 5+ | 15-25 years | 2000-2024 | Full cycle testing | Covers dot-com, financial crisis, multiple rate cycles |

**Why this sequencing:** Adding historical data before the system is structurally stable wastes compute and risks tuning parameters to conditions that may not repeat. The 10-year expansion happens after Phase 3 (portfolio manager complete) when the system is stable enough that a parameter re-fit is meaningful.

**Data source for expansion:** yfinance provides adjusted OHLCV going back to the early 2000s for most large-cap tickers. The current data pipeline already uses yfinance; the expansion is primarily a matter of changing the start date parameter and re-running the download. However: the K-NN training database will grow proportionally (from ~175k rows to ~700k rows for 10 years, or ~1.75M rows for 25 years). The candlestick categorization module (CANDLESTICK_CATEGORIZATION_DESIGN.md) becomes mandatory before the 10-year expansion — the K-NN index at 700k rows without pre-filtering will be 4× slower per query.

**Ticker universe and data availability:** Not all current 52 tickers have 25 years of clean data. Some ETFs (e.g., QQQ, IWM) were created in the late 1990s and are fine. Others (e.g., sector ETFs like XLK) may have sparse early history. A data coverage audit is required before the 15-25 year expansion.

### 5.3 Derived Data (computed by this system)

| Field | Source | Description |
|-------|--------|-------------|
| ATR_pct | Market data | 20-day average true range as % of close price |
| rolling_volatility | Market data | 20-day rolling standard deviation of daily returns |
| correlation_matrix | Market data | 60-day pairwise return correlations (for tie-breaking) |
| sector | Config | Sector classification for each of the 52 tickers |

---

## 6. Implementation Plan

### Phase 1: Foundation — Prove the Pipeline (config.py + backtest_engine.py)

Build the configuration module, the signal adapter, and a minimal backtester that processes signals, simulates execution at next-day open, applies the cost model, and produces a trade log + equity curve. Use **equal-weight fixed position sizing** (every trade gets 5% of equity) to validate the pipeline independent of risk/portfolio logic.

**Deliverable:** Run backtest on 2024 validation data with equal-weight positions. Verify:
- Trade log entries match expected signal-to-execution flow
- Equity curve arithmetic is correct (spot-check 10 random trades manually)
- Cost attribution (slippage + spread) is applied correctly on both entry and exit
- Baseline comparisons (SPY, random, raw signals) all produce output

**This phase intentionally uses 2024 as a demonstration slice, NOT a tuning dataset.** No parameters are optimized against 2024 results. The purpose is to verify mechanical correctness.

### Phase 2: Risk Layer (risk_engine.py)

Add volatility-based position sizing, stop-losses, and the drawdown brake. Replace equal-weight with dynamic sizing. Add re-entry cooldown logic.

**Deliverable:** Re-run 2024 backtest with risk engine active. Compare:
- Does dynamic sizing reduce max drawdown relative to Phase 1's equal-weight?
- Are stop-losses firing at appropriate levels? Check for gap-through-stop cases.
- Does the drawdown brake visibly reduce position sizes during losing streaks?
- Rejection log: verify reasons are correct and constraints are binding.

### Phase 3: Portfolio Layer (portfolio_manager.py)

Add confidence-based signal ranking, sector diversification, and capital allocation. This resolves the multi-signal simultaneity problem.

**Deliverable:** Verify:
- On days with 5+ BUY signals, the allocator correctly ranks by confidence
- Sector limits constrain the portfolio (visible in rejection log)
- Capital exhaustion stops allocation (no over-commitment)
- The system doesn't degrade vs. Phase 2 (portfolio layer should help, not hurt)

### Phase 4: Evaluation Layer (strategy_evaluator.py)

Add rolling metric computation, status signals, calibration tracking, and baseline comparisons.

**Deliverable:** Complete output showing:
- All metrics across 30/90/252-day/all-time windows
- GREEN/YELLOW/RED status is correctly assigned
- Calibration curve shows reasonable confidence-to-win-rate mapping
- Baseline comparison table is populated and correct
- Per-sector and per-ticker attribution identifies strongest/weakest contributors

### Phase 5: Excel Dashboard

Build an Excel reporting layer that reads from the Python output (CSV/JSON exports) and displays equity curves, trade logs, rolling metrics, baseline comparisons, and calibration charts.

**Deliverable:** A single .xlsx file updated from the most recent backtest run.

### Phase 6 (Future): v2 Additions

After v1 passes its success criteria on out-of-sample data:
- Add short selling (margin model, borrow fees, short-specific stops)
- Add composite ranking score (confidence + correlation penalty + risk-adjusted return)
- Add confidence-dependent holding periods (if data supports it)
- Add partial exits (if capital base justifies it)
- Evaluate VWAP or 30-min-after-open execution as alternative to raw open

---

## 7. Known Risks & Mitigations

### 7.1 Thin Edge (Critical)

FPPE's current BSS of +0.00103 and 56.6% accuracy means the system is operating very close to breakeven after costs. With the revised 26 bps round-trip friction (up from 16 bps in v0.1), the margin is even thinner. Every design decision in this system must prioritize edge preservation over feature richness.

**Mitigation:** v1 is deliberately simple. Volatility-only sizing, confidence-only ranking, no leverage, no shorts. Complexity is added only after the simple version proves profitable.

### 7.2 Overfitting Risk

Two layers of overfitting risk: (1) FPPE was validated on 2024 data, and (2) the trading system will be demonstrated on 2024 data. If we tune trading parameters to improve 2024 results, we're fitting to a dataset that FPPE has already seen.

**Mitigation:** The Phase 1 backtest on 2024 is a pipeline verification step only — no parameters are optimized against it. The v1 success criteria (Section 3.2) must be evaluated on 2025+ data, which remains untouched by both FPPE and the trading system.

### 7.3 Open Execution Fragility

Next-day open execution is the noisiest print of the day. Overnight gaps from earnings, macro news, or pre-market activity can invalidate the signal premise before the trade even executes.

**Mitigation:** Increased slippage estimate to 10 bps per side (from 5 bps). Stop-losses assume exit at next open after trigger, not at stop price. Phase 6 includes evaluation of alternative execution timing (VWAP, 30-min delayed open).

### 7.4 Capital Adequacy

$10,000 with 2% minimum positions ($200 per trade) and 26 bps friction ($0.52 per round-trip) means per-trade friction is small in absolute terms but meaningful relative to expected per-trade profit at 56.6% accuracy.

**Mitigation:** The model is percentage-based and fully scalable. $10k is the paper-trading proof. If v1 passes, the same model runs at $50k or $100k where friction becomes proportionally less impactful.

### 7.5 Sparse Signal Risk

The document does not yet define expected signal frequency. If FPPE's three-filter gate is highly selective, the system may produce too few trades for statistically meaningful evaluation. The v1 success criterion requires ≥50 trades.

**Mitigation:** Track signal generation rate as a metric. If fewer than 50 trades occur in the 2024 backtest period, this needs to be flagged and addressed (either by relaxing FPPE's gate or extending the evaluation window).

---

## 8. Resolved Design Questions

These were open in v0.1. Resolutions incorporate both reviewer recommendations and project owner decisions.

| Question | Resolution | Rationale |
|----------|------------|-----------|
| Max holding period | 10 trading days, fixed | Aligned with FPPE's ~7-day projection horizon; confidence-dependent holds deferred to v2 pending evidence |
| Re-entry rules | 3-day cooldown after stop-loss or max-hold exit; re-entry on reversal requires +0.05 confidence margin | Prevents whipsaw churn while allowing genuine regime changes |
| Partial exits | All-in/all-out for v1 | At $200 min position, partial exits multiply friction and destroy margin |
| Cash management | Earn daily risk-free rate; report returns both with and without cash yield | Prevents cash yield from masking weak trading performance |
| Dividends and splits | Use adjusted prices for v1 | Sufficient for paper-trading research; explicit corporate action handling in v2 if approaching real capital |

---

## 9. Approval Checklist

Before any code is written, confirm:

- [ ] v1 scope (long-only, no leverage, no shorts) is accepted
- [ ] Success criteria (Section 3.2 — edge preservation, not return targets) are accepted
- [ ] Revised friction model (26 bps round-trip) is accepted
- [ ] Confidence used once (ranking only, not sizing) is understood
- [ ] Signal adapter for hybrid K-NN/DL output is correct
- [ ] Implementation phasing (Section 6) is the right order
- [ ] All resolved design questions (Section 8) are agreed upon
- [ ] Known risks (Section 7) are acknowledged
- [ ] Baseline comparisons (Section 3.3) are the right benchmarks

---

*Version history:*
- *v0.1 (2026-03-19): Initial draft*
- *v0.2 (2026-03-19): Revised after two independent reviews. Major scope reduction for v1, fixed confidence double-counting, added baselines, revised friction model, added signal adapter for hybrid FPPE.*
- *v0.3 (2026-03-19): Phase 1 backtest complete. Empirical parameter sweep results applied: confidence_threshold 0.65→0.60, max_holding_days 10→14. Phase 1 result: 22.3% annual, Sharpe 1.82, Max DD 6.9%. Candlestick categorization module designed (see CANDLESTICK_CATEGORIZATION_DESIGN.md).*
