# FPPE Trading System — Architecture & Design Specification

**Version:** 0.5 (Phase 3.5 Research Integration complete; SLE-47 HNSW + SLE-48 vectorized query)
**Date:** March 21, 2026
**Status:** ACTIVE — Phase 3 & 3.5 complete. Phase 4 (strategy evaluator) next.

**Revision notes (v0.5):** Phase 3.5 Research Integration complete. ATR sweep locked 3.0×. HNSW index (SLE-47) integrated. Vectorized Matcher.query() (SLE-48) replaces per-row iloc. 596 tests passing.

**Revision notes (v0.4):** Phase 3 Portfolio Manager implemented and validated.

- **New modules:** `trading_system/portfolio_state.py` (frozen dataclasses: `RankedSignal`, `AllocationDecision`, `PortfolioSnapshot`), `trading_system/portfolio_manager.py` (Layer 3 logic: `rank_signals`, `check_allocation`, `allocate_day`)
- **Integration:** `BacktestEngine` now accepts `use_portfolio_manager=True` (requires `use_risk_engine=True`). PM pre-filters on count constraints (holding, cooldown, sector limit); risk engine gates on dollar constraints (exposure, capital). Rank order preserved end-to-end.
- **Backtest validation (2024, 52-ticker universe):** Phase 3 matches Phase 2 exactly — Ann. 15.1%, Sharpe 1.93, Max DD 3.9%, Win rate 52.4%, 103 trades. PM generated 37 portfolio-layer rejections (29 already-holding, 7 cooldown, 1 sector limit).
- **Test suite:** 556 tests, 0 failures. Layer 3 adds 11 integration tests, 30 unit tests (portfolio_manager), 20 unit tests (portfolio_state).
- **Phase 3 branch:** `phase3-portfolio-manager` — ready for merge to main.

**Revision notes (v0.3):** Phase 1 backtest on 2024 data. Empirical sweeps set confidence_threshold=0.60 (5 values tested) and max_holding_days=14 (7 windows tested). Result: 22.3% annual, Sharpe 1.82, Max DD 6.9%, beats SPY risk-adjusted. All v1 success criteria passed. CAUTION: 2024 was a strong bull year — re-validate max_holding_days in bear-market data.

**Revision notes (v0.2):** v1 is long-only (shorts deferred to v2). Position sizing is volatility-only (no confidence double-counting). Slippage revised to 10 bps (open is noisiest print). Max hold reduced from 20d to 10d (later revised to 14d in v0.3).

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

### 3.9 Capital

Current default: **$10,000 Standard tier** (min 2%, max 10% per position, 14-20 max positions). Capital tier scaling (Micro → Institutional) is deferred until Phase 4+ when portfolio manager handles multi-position allocation.

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
│   ├── stop_loss_atr_multiple: 3.0    # LOCKED (swept 2.0–4.0; 3.0× winner, 2026-03-21)
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

**Outputs:** Trade log (trade_id, ticker, entry/exit date+price, gross/net PnL, holding_days, exit_reason), daily equity curve (equity, cash, drawdown, cash_yield), rejected signal log (rejection_reason, rejection_layer).

**Execution:** Signal after close → cooldown check → Layer 3 rank → Layer 2 size → execute at next-day open with slippage. MTM at close daily. Exit triggers: SELL/HOLD signal on held position, stop triggered (exit at next open — not stop price), max hold exceeded, drawdown halt.

---

### 4.3 risk_engine.py — Position Sizing & Risk Control (Layer 2)

**Purpose:** Given an approved trade candidate, determine position size based on volatility. Enforce risk limits and drawdown protection.

**Key design:** Sizing is purely volatility-based; confidence drives Layer 3 ranking only (no double-counting).

**Position sizing:**
```
stop_distance = stop_loss_atr_multiple × ATR%   # LOCKED: 3.0× (swept 2.0–4.0, 2026-03-21)
raw_weight = max_loss_per_trade_pct / stop_distance  # 2% equity risk per trade
final_weight = clamp(raw_weight × drawdown_scalar, min_pct, max_pct)
drawdown_scalar: 1.0 → 0.0 linearly from 15% → 20% drawdown; 0.0 at halt
```

**Rejection conditions:** sector limit exceeded, gross exposure > 100%, drawdown halt, already held, cooldown, position below 2% minimum.

---

### 4.4 portfolio_manager.py — Capital Allocation (Layer 3)

**Purpose:** When multiple BUY signals arrive on the same date, rank them and decide which become trades. Enforce capital and diversification constraints.

**v1 ranking:** `rank_score = confidence` (ties: lower portfolio correlation, then alphabetical). v2 composite ranking (confidence + momentum + recency + volatility + correlation + pattern) deferred until v1 baseline is empirically validated. No grid-search on weights until 10+ years of data available.

**Allocation algorithm:** Collect BUY signals → remove held/cooldown → sort by confidence → iterate: check sector (≤30%, ≤3) + capital → send to Layer 2 → stop when no capital remains.

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

**Historical data:** Current dataset 2010-2024 (~175k training rows). 10-year expansion (2015-2024) planned after Phase 4. HNSW index (SLE-47) makes 700k-row expansion feasible — ball_tree would require candlestick pre-filtering first.

### 5.3 Derived Data (computed by this system)

| Field | Source | Description |
|-------|--------|-------------|
| ATR_pct | Market data | 20-day average true range as % of close price |
| rolling_volatility | Market data | 20-day rolling standard deviation of daily returns |
| correlation_matrix | Market data | 60-day pairwise return correlations (for tie-breaking) |
| sector | Config | Sector classification for each of the 52 tickers |

---

## 6. Implementation Status

| Phase | Module | Status | Key Result |
|-------|--------|--------|------------|
| 1 | backtest_engine.py (equal-weight) | **DONE** | 22.3% annual, Sharpe 1.82, Max DD 6.9% |
| 2 | risk_engine.py (ATR sizing) | **DONE** | $9.31 NE/trade, stops=28 at 3.0× ATR |
| 3 | portfolio_manager.py | **DONE** | 37 PM rejections, matches Phase 2 exactly |
| 3.5 | Research integration | **DONE** | ATR=3.0× locked, HNSW 54.5×, vectorized query |
| **4** | **strategy_evaluator.py** | **NEXT** | Rolling metrics, RED/YELLOW/GREEN, TWRR |
| 5 | Excel dashboard | Future | — |
| v2 | Shorts, composite ranking, partial exits | Future | After v1 proven on OOS data |

---

## 7. Active Risks

| Risk | Mitigation |
|------|-----------|
| Thin edge (BSS +0.00103) | v1 is deliberately simple — volatility sizing, confidence ranking, no leverage. Complexity only after edge is proven. |
| Overfitting to 2024 | 2024 is pipeline verification only. v1 success criteria evaluated on 2025+ held-out data. |
| Open execution noise | 10 bps slippage estimate; stops exit at next open, not stop price. |

---

