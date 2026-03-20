# Phase 2 System Design — risk_engine.py
## FPPE Trading System Layer 2: Position Sizing & Risk Control

**Version:** 1.0
**Date:** March 19, 2026
**Status:** IMPLEMENTED — See `trading_system/risk_engine.py`, `risk_state.py`, and `tests/test_phase2_integration.py`
**Prerequisite:** Phase 1 complete (backtest_engine.py, config.py, signal_adapter.py)
**Design doc parent:** FPPE_TRADING_SYSTEM_DESIGN.md v0.4, Section 4.3

---

## 0. PRE-IMPLEMENTATION ISSUES

These issues **must be resolved before writing any Phase 2 code.** They were discovered during the design audit and represent real discrepancies between documentation and codebase state.

### ISSUE 0.1: Missing Phase 1 Files (BLOCKING)

**Observed state (2026-03-19):** The `trading_system/` directory contains only `config.py` and `__pycache__/`. The following files referenced in PROJECT_GUIDE.md and PHASE1_FILE_REVIEW.md are absent:

| Expected File | Status | Referenced By |
|---------------|--------|---------------|
| `trading_system/__init__.py` | **MISSING** | PROJECT_GUIDE §2, PHASE1_FILE_REVIEW |
| `trading_system/signal_adapter.py` | **MISSING** | PROJECT_GUIDE §2, Design doc §5.1 |
| `trading_system/backtest_engine.py` | **MISSING** | PROJECT_GUIDE §2, Design doc §4.2, PHASE1_FILE_REVIEW |
| `trading_system/run_phase1.py` | **MISSING** | PROJECT_GUIDE §2 |

Additionally, the 88 trading_system tests referenced in PROJECT_GUIDE (test_trading_config, test_signal_adapter, test_backtest_engine) are not present in `tests/`.

**Resolution required:** Either (a) locate and restore these files, or (b) re-implement Phase 1 before beginning Phase 2. Phase 2 cannot proceed without a working backtest_engine.py — the risk engine's entire output is consumed by the backtest loop.

**Confidence level:** High that files are missing, not merely in a different location. The `ls` output of trading_system/ shows only config.py (17KB). The total test count visible in tests/ is consistent with the 243 pattern_engine tests but not the claimed 331 total (243 + 88).

### ISSUE 0.2: Conservative Profile Brake Inconsistency (NON-BLOCKING)

`config.py` line 294-295 sets the conservative profile's `drawdown_brake_threshold=0.10` and `drawdown_halt_threshold=0.15`. The design doc Section 3.8 specifies conservative `Max DD Halt = 15%`, which matches. However, a 10% brake with 15% halt gives only 5 percentage points of linear ramp. The aggressive profile has a 5-point ramp (15% → 20%), and moderate has a 6-point ramp (12% → 18%). The conservative profile's ramp is proportionally correct but worth calling out — it means the conservative profile transitions from full-size to zero-size positions over a very narrow drawdown band.

**Recommendation:** Acceptable as-is. The narrow band is intentional — conservative users want aggressive risk reduction once drawdown starts.

### ISSUE 0.3: SharedState Not Yet Implemented (NON-BLOCKING for Phase 2, BLOCKING for Phase 3+)

The design doc Section 2.1 specifies a `SharedState` in-memory object for inter-layer communication. This does not exist in the codebase. For Phase 2, the risk_engine can operate as a pure function (input: trade candidate + price data + config → output: sized position + stop price). SharedState is needed when Layer 3 and Layer 4 need to read/write state across layers.

**Recommendation:** Phase 2 implements risk_engine as a stateless module with a lightweight `RiskState` dataclass for drawdown tracking. SharedState is deferred to Phase 3.

### ISSUE 0.4: ATR Computation Source Unspecified

The design doc references "20-day ATR" but does not specify whether this uses:
- The `ta` library's ATR indicator (already a project dependency)
- Manual True Range computation
- A pre-computed column from the FPPE data pipeline

**Recommendation:** Use the `ta` library (`ta.volatility.AverageTrueRange`) since it's already in requirements.txt. Compute ATR inside risk_engine from raw OHLCV data rather than expecting a pre-computed column — this keeps the risk engine self-contained and testable without FPPE pipeline dependencies.

---

## 1. PURPOSE AND SCOPE

### 1.1 What Phase 2 Does

Phase 2 replaces the fixed 5% equal-weight position sizing from Phase 1 with three capabilities:

1. **Volatility-based position sizing** — Size each position so that a stop-loss hit loses exactly 2% of current equity. High-volatility stocks get smaller positions; low-volatility stocks get larger positions.

2. **ATR stop-losses** — Set a stop price at `entry_price - (2 × ATR%)` for each long position. Stop is evaluated against intraday lows but executed at next-day open (realistic fill assumption).

3. **Drawdown brake and halt** — A portfolio-level risk overlay that linearly reduces position sizes as drawdown approaches 15%, and halts all new entries at 20% drawdown.

### 1.2 What Phase 2 Does NOT Do

- **No confidence-based sizing.** Confidence drives only signal generation (FPPE's three-filter gate) and, in Phase 3, ranking. Position sizing is purely volatility-based. This is a deliberate design choice to prevent double-counting confidence across multiple layers.
- **No portfolio correlation management.** Pairwise correlations are a Phase 3 concern (portfolio_manager.py).
- **No short selling.** v1 is long-only.
- **No trailing stops.** Fixed ATR stops only. Trailing stops are a Phase 6 consideration.
- **No partial exits.** All-in/all-out. At $10k capital with 2% min positions ($200), partial exits multiply friction.

### 1.3 Success Criteria for Phase 2

Phase 2 passes if ALL of the following hold when re-running the 2024 backtest with dynamic sizing:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Max drawdown reduced | < Phase 1's 6.9% | Dynamic sizing should actively reduce drawdown |
| Sharpe maintained or improved | ≥ 1.82 (Phase 1 baseline) | Risk-adjusted return should improve, not degrade |
| Net expectancy remains positive | > $0 after friction | Edge must survive the risk layer |
| Stop-losses fire at appropriate levels | Manual spot-check of 20 trades | No stops triggered by normal intraday noise |
| Drawdown brake visibly reduces sizing | Inspect sizing log during losing streaks | The mechanism must actually engage |
| No gap-through-stop P&L errors | Exit at next-open, not stop price | Gap-through scenarios handled correctly |
| All existing 331+ tests still pass | Zero failures | No regressions |
| New risk_engine tests pass | ≥ 40 new tests | Comprehensive coverage |

**CAUTION (carried from Phase 1):** 2024 was a strong bull year. The drawdown brake may never engage in a bull backtest. Phase 2 validation should include a synthetic stress test or a manual check against the 2022 bear fold data if available in signal cache.

---

## 2. ARCHITECTURE

### 2.1 Module Placement

```
trading_system/
├── __init__.py              # Updated exports
├── config.py                # Existing — RiskConfig already defined
├── signal_adapter.py        # Phase 1 — unified signal schema
├── backtest_engine.py       # Phase 1 — modified to call risk_engine
├── risk_engine.py           # NEW — this document's primary deliverable
├── risk_state.py            # NEW — lightweight drawdown state tracker
└── run_phase1.py            # Phase 1 entry point (updated to support Phase 2 mode)
```

### 2.2 Data Flow

```
FPPE Signal (UnifiedSignal)
    │
    ▼
backtest_engine.py  ─── "Should I take this trade?" ──►  risk_engine.py
    │                                                         │
    │                    ◄── PositionDecision ─────────────────┘
    │                        (size, stop_price, or rejection)
    │
    ▼
Execute trade at next-day open with sized position
    │
    ▼
Daily mark-to-market loop
    │
    ├── Check stop-loss: low ≤ stop_price?
    │       Yes → flag for exit at next-day open
    │
    ├── Check max-hold exceeded?
    │       Yes → flag for exit at next-day open
    │
    ├── Update RiskState (current drawdown from peak)
    │       risk_engine.update_drawdown(current_equity, peak_equity)
    │
    └── Next day
```

### 2.3 Integration Point with backtest_engine.py

The backtest engine currently uses fixed 5% equal-weight sizing (Phase 1). Phase 2 changes this to call `risk_engine.size_position()` for each approved BUY signal. The modification is localized to the section of backtest_engine that converts a BUY signal into an OpenPosition.

**Phase 1 (current):**
```python
position_pct = 0.05  # Fixed 5%
shares = (current_equity * position_pct) / entry_price
```

**Phase 2 (new):**
```python
decision = risk_engine.size_position(
    ticker=signal.ticker,
    entry_price=next_open_price,
    current_equity=current_equity,
    peak_equity=peak_equity,
    price_history=ticker_price_history,  # last 20+ trading days of OHLCV
    config=config.risk,
    position_limits=config.position_limits,
)
if decision.approved:
    shares = decision.shares
    stop_price = decision.stop_price
else:
    # Log rejection with decision.rejection_reason
    continue
```

---

## 3. DATA STRUCTURES

### 3.1 PositionDecision (output of size_position)

```python
@dataclass(frozen=True)
class PositionDecision:
    """Output of the risk engine's position sizing calculation.

    Immutable — computed once per trade candidate, never modified.
    """
    approved: bool                      # True if trade is permitted
    ticker: str                         # Ticker symbol
    position_pct: float                 # Position as fraction of equity (0.0–1.0)
    shares: float                       # Share count (fractional allowed)
    dollar_amount: float                # Position dollar value
    stop_price: float                   # ATR-based stop-loss price
    stop_distance_pct: float            # Distance from entry to stop as %
    atr_pct: float                      # 20-day ATR as % of current price
    drawdown_scalar: float              # 0.0–1.0 drawdown adjustment applied
    raw_weight: float                   # Pre-clamp, pre-drawdown weight
    rejection_reason: Optional[str]     # None if approved; string if rejected

    def __post_init__(self):
        """Validate decision integrity."""
        if self.approved and self.shares <= 0:
            raise ValueError("Approved decision must have positive shares")
        if self.approved and self.stop_price <= 0:
            raise ValueError("Approved decision must have positive stop_price")
        if not 0.0 <= self.drawdown_scalar <= 1.0:
            raise ValueError(f"drawdown_scalar must be in [0, 1], got {self.drawdown_scalar}")
```

### 3.2 RiskState (drawdown tracker)

```python
@dataclass
class RiskState:
    """Mutable state tracking portfolio-level risk metrics.

    Updated daily by backtest_engine after mark-to-market.
    Read by risk_engine when sizing new positions.

    NOT frozen — this is mutable state that changes daily.
    """
    peak_equity: float                  # High-water mark
    current_equity: float               # As of last MTM
    current_drawdown: float             # (peak - current) / peak, always ≥ 0
    drawdown_mode: str                  # "normal" | "brake" | "halt"
    sizing_scalar: float                # 0.0–1.0 based on drawdown position
    active_stops: Dict[str, float]      # ticker → stop price for open positions
    daily_atr_cache: Dict[str, float]   # ticker → most recent ATR% (refreshed daily)

    def update(self, current_equity: float, config: RiskConfig) -> None:
        """Recompute drawdown state after daily MTM.

        Args:
            current_equity: Portfolio equity after today's MTM.
            config: Risk configuration for brake/halt thresholds.
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.current_equity = current_equity

        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0

        # Compute drawdown mode and sizing scalar
        brake = config.drawdown_brake_threshold
        halt = config.drawdown_halt_threshold

        if self.current_drawdown >= halt:
            self.drawdown_mode = "halt"
            self.sizing_scalar = 0.0
        elif self.current_drawdown >= brake:
            self.drawdown_mode = "brake"
            # Linear interpolation: 1.0 at brake → 0.0 at halt
            self.sizing_scalar = (halt - self.current_drawdown) / (halt - brake)
        else:
            self.drawdown_mode = "normal"
            self.sizing_scalar = 1.0

    def register_stop(self, ticker: str, stop_price: float) -> None:
        """Record a new stop-loss for an open position."""
        self.active_stops[ticker] = stop_price

    def remove_stop(self, ticker: str) -> None:
        """Remove stop when position is closed."""
        self.active_stops.pop(ticker, None)

    @classmethod
    def initial(cls, starting_equity: float) -> "RiskState":
        """Create initial risk state at start of backtest."""
        return cls(
            peak_equity=starting_equity,
            current_equity=starting_equity,
            current_drawdown=0.0,
            drawdown_mode="normal",
            sizing_scalar=1.0,
            active_stops={},
            daily_atr_cache={},
        )
```

### 3.3 StopLossEvent (emitted when stop triggers)

```python
@dataclass(frozen=True)
class StopLossEvent:
    """Record of a stop-loss trigger for audit trail.

    Created when intraday low ≤ stop_price. The actual exit
    occurs at next-day open (not at the stop price) per design
    doc Section 4.2.
    """
    ticker: str
    trigger_date: str                   # Date the low breached the stop
    stop_price: float                   # The stop level that was breached
    trigger_low: float                  # The intraday low that triggered it
    entry_price: float                  # Original entry price
    exit_price: float                   # Next-day open (actual execution)
    gap_through: bool                   # True if trigger_low < stop_price (gap-down)
    atr_at_entry: float                 # ATR% when position was opened
```

---

## 4. CORE ALGORITHM

### 4.1 Position Sizing Formula

```
INPUTS:
    entry_price         — next-day open price (after slippage)
    current_equity      — portfolio equity after last MTM
    peak_equity         — high-water mark equity
    atr_pct             — 20-day ATR / current_price
    config.max_loss_per_trade_pct    — 0.02 (2% of equity)
    config.stop_loss_atr_multiple    — 2.0
    config.drawdown_brake_threshold  — 0.15
    config.drawdown_halt_threshold   — 0.20
    position_limits.min_position_pct — 0.02
    position_limits.max_position_pct — 0.10

STEP 1: Compute stop distance
    stop_distance_pct = stop_loss_atr_multiple × atr_pct
    stop_price = entry_price × (1 - stop_distance_pct)

STEP 2: Compute raw position weight
    raw_weight = max_loss_per_trade_pct / stop_distance_pct

    EXPLANATION: If ATR% is 3% and we use 2× ATR stop, stop_distance = 6%.
    raw_weight = 0.02 / 0.06 = 0.333 (33.3% of equity).
    A stop hit loses 33.3% × 6% = 2% of equity. ✓

    If ATR% is 1% (low vol), stop_distance = 2%.
    raw_weight = 0.02 / 0.02 = 1.0 (100% of equity).
    Clamped to max_position_pct = 10%.

    If ATR% is 5% (high vol), stop_distance = 10%.
    raw_weight = 0.02 / 0.10 = 0.20 (20% of equity).
    Clamped to max_position_pct = 10%.

STEP 3: Apply drawdown scalar
    drawdown = (peak_equity - current_equity) / peak_equity

    IF drawdown < brake_threshold (0.15):
        drawdown_scalar = 1.0
    ELIF drawdown < halt_threshold (0.20):
        drawdown_scalar = (halt - drawdown) / (halt - brake)
        # Linear ramp: 1.0 at 15% DD → 0.0 at 20% DD
    ELSE:
        drawdown_scalar = 0.0  → REJECT (halt mode)

    adjusted_weight = raw_weight × drawdown_scalar

STEP 4: Clamp to position limits
    final_weight = clamp(adjusted_weight, min_position_pct, max_position_pct)

    IF final_weight < min_position_pct AND adjusted_weight > 0:
        → REJECT: "Position size below minimum after drawdown adjustment"

STEP 5: Compute shares
    dollar_amount = current_equity × final_weight
    shares = dollar_amount / entry_price
    IF NOT fractional_shares:
        shares = floor(shares)
        IF shares == 0: → REJECT: "Insufficient capital for whole shares"
```

### 4.2 ATR Computation

```python
def compute_atr_pct(price_history: pd.DataFrame, lookback: int = 20) -> float:
    """Compute Average True Range as percentage of current price.

    Uses the standard ATR definition:
        TR = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = EMA(TR, lookback)
        ATR% = ATR / current_close

    Args:
        price_history: DataFrame with columns [Date, Open, High, Low, Close].
                       Must have at least lookback + 1 rows.
        lookback: Number of periods for ATR (default 20 trading days).

    Returns:
        ATR as fraction of current price (e.g., 0.03 = 3%).

    Raises:
        ValueError: If insufficient data (< lookback + 1 rows).
        ValueError: If current price is zero or negative.
    """
```

**Implementation note:** Use `ta.volatility.AverageTrueRange(high, low, close, window=lookback).average_true_range()` from the `ta` library. Then divide the final ATR value by the current close price to get ATR%.

**Edge cases:**
- If a ticker has fewer than `lookback + 1` rows of history (newly added or IPO), REJECT the trade with reason "Insufficient price history for ATR computation".
- If ATR% computes to 0.0 (price was perfectly flat for 20 days — theoretically possible but practically means bad data), REJECT with reason "Zero ATR — likely data issue".
- If ATR% > 0.20 (20% — extreme volatility, e.g., meme stocks during a squeeze), allow but log a warning. The sizing formula will naturally produce a very small position.

### 4.3 Stop-Loss Evaluation

Stops are evaluated **daily** in the backtest loop against intraday lows:

```
FOR each open position with a stop:
    IF today's Low ≤ position.stop_price:
        → Mark position for exit
        → Exit price = NEXT trading day's Open (NOT the stop price)
        → Record StopLossEvent with gap_through flag

    gap_through = (today's Low < position.stop_price)
    # If the stock gapped below the stop, the actual loss exceeds
    # the intended 2% max. This is expected and realistic.
    # The sizing formula targets 2% loss AT the stop price,
    # but gap-through scenarios can exceed it.
```

**Why next-day open, not stop price:** The design doc (Section 4.2, stop-loss note) explicitly requires this. In real markets, a stop-loss triggered during the session would execute at market price, which could be below the stop. Since we're using daily bars, we can't know the intraday execution price. Using the stop price would understate losses. Using the next-day open is conservative and avoids unrealistic fill assumptions.

### 4.4 Drawdown Brake Mechanics

The drawdown brake is a **continuous** function, not a step function. This prevents the cliff-edge problem where a portfolio at 14.99% drawdown takes full-size positions and at 15.01% suddenly takes half-size.

```
Sizing scalar as a function of drawdown:

    1.0 ┤─────────────────────┐
        │                     │
        │    Normal Mode      │ Brake Mode
        │                     │
    0.5 ┤                     ├─ ─ ─ ─ ─
        │                     │          ╲
        │                     │           ╲
    0.0 ┤─────────────────────┴────────────╳─── Halt Mode
        └──────┬──────────────┬────────────┬───
              0%            15%          20%
                         Drawdown from peak

    Normal: 0% – 14.99% DD → scalar = 1.0 (full size)
    Brake:  15% – 19.99% DD → scalar = linear 1.0 → 0.0
    Halt:   ≥ 20% DD       → scalar = 0.0 (no new trades)
```

**Halt mode behavior:** When drawdown hits 20%, the risk engine rejects ALL new trade requests. Existing positions are NOT force-closed by the risk engine (that's a Layer 4 responsibility in Phase 4). The risk engine only controls entry sizing.

**Recovery from halt:** When equity recovers and drawdown drops below 20%, the risk engine re-enters brake mode automatically. When drawdown drops below 15%, normal sizing resumes. There is no manual intervention required. The peak_equity high-water mark does NOT reset on recovery — it only moves upward.

---

## 5. REJECTION CONDITIONS

The risk engine may reject a trade candidate for any of the following reasons. Each rejection is logged with its reason string for the rejection log.

| Condition | Reason String | Priority |
|-----------|---------------|----------|
| Drawdown ≥ halt threshold | `"Drawdown halt: {dd:.1%} ≥ {halt:.1%}"` | 1 (checked first) |
| Insufficient price history | `"Insufficient history: {n} rows < {required}"` | 2 |
| Zero or negative ATR | `"Zero ATR for {ticker} — likely data issue"` | 3 |
| Position below minimum after DD scalar | `"Below min size: {adj:.1%} < {min:.1%} after DD adjustment"` | 4 |
| Would breach max_position_pct | N/A — clamped, not rejected | — |
| Would breach sector limit | `"Sector {sector} at capacity: {n}/{max} positions"` | 5 |
| Would breach gross exposure limit | `"Gross exposure would exceed {max:.0%}"` | 6 |
| Ticker already held | `"Already holding {ticker}"` | 7 |
| Ticker in cooldown | `"Ticker {ticker} in cooldown until {date}"` | 8 |

**Note on sector and exposure checks:** These are currently enforced in backtest_engine.py (Phase 1). Phase 2 adds them as redundant checks inside risk_engine as well — defense in depth. If backtest_engine rejects before risk_engine is called, the risk_engine check never fires. If backtest_engine's check is accidentally removed in a future refactor, the risk_engine still catches it.

---

## 6. CONFIGURATION MAPPING

Phase 2 uses parameters already defined in `trading_system/config.py`. No new config fields are required.

| Phase 2 Parameter | Config Path | Default | Source |
|-------------------|-------------|---------|--------|
| ATR lookback | `config.risk.volatility_lookback` | 20 | Design doc §4.3 |
| Stop ATR multiple | `config.risk.stop_loss_atr_multiple` | 2.0 | Design doc §4.3 |
| Max loss per trade | `config.risk.max_loss_per_trade_pct` | 0.02 | Design doc §4.3 |
| Drawdown brake | `config.risk.drawdown_brake_threshold` | 0.15 | Design doc §3.8 |
| Drawdown halt | `config.risk.drawdown_halt_threshold` | 0.20 | Design doc §3.8 |
| Min position % | `config.position_limits.min_position_pct` | 0.02 | Design doc §3.5 |
| Max position % | `config.position_limits.max_position_pct` | 0.10 | Design doc §3.5 |
| Max sector % | `config.position_limits.max_sector_pct` | 0.30 | Design doc §3.5 |
| Max per sector | `config.position_limits.max_positions_per_sector` | 3 | Design doc §3.5 |
| Fractional shares | `config.capital.fractional_shares` | True | Design doc §3.1 |

**Profile-specific overrides (from config.py from_profile()):**

| Parameter | Aggressive | Moderate | Conservative |
|-----------|-----------|----------|--------------|
| max_position_pct | 10% | 8% | 7% |
| drawdown_brake | 15% | 12% | 10% |
| drawdown_halt | 20% | 18% | 15% |

---

## 7. PUBLIC API

### 7.1 Module-Level Functions

```python
def size_position(
    ticker: str,
    entry_price: float,
    current_equity: float,
    price_history: pd.DataFrame,
    risk_state: RiskState,
    config: RiskConfig,
    position_limits: PositionLimitsConfig,
    sector_map: Dict[str, str],
    open_positions: Dict[str, Any],
) -> PositionDecision:
    """Compute position size and stop-loss for a trade candidate.

    This is the primary entry point for the risk engine. Called by
    backtest_engine for each BUY signal that passes signal filtering
    and cooldown checks.

    The function is PURE (no side effects) — it reads risk_state but
    does not modify it. The caller (backtest_engine) is responsible
    for updating risk_state after trade execution.

    Args:
        ticker: Stock symbol.
        entry_price: Expected execution price (next-day open with slippage).
        current_equity: Portfolio equity after last MTM.
        price_history: OHLCV DataFrame for this ticker, last N trading days.
                       Must have columns: Date, Open, High, Low, Close, Volume.
                       Must have ≥ volatility_lookback + 1 rows.
        risk_state: Current portfolio risk state (drawdown, active stops).
        config: RiskConfig with all risk parameters.
        position_limits: PositionLimitsConfig with sizing constraints.
        sector_map: Ticker → sector mapping.
        open_positions: Currently held positions {ticker: position_info}.

    Returns:
        PositionDecision with sizing details or rejection reason.
    """


def compute_atr_pct(
    price_history: pd.DataFrame,
    lookback: int = 20,
) -> float:
    """Compute ATR as percentage of current price.

    Args:
        price_history: DataFrame with High, Low, Close columns.
        lookback: EMA window for ATR (default 20).

    Returns:
        ATR / current_close as a float (e.g., 0.03 = 3%).

    Raises:
        ValueError: If insufficient data or zero price.
    """


def check_stop_loss(
    current_low: float,
    stop_price: float,
) -> bool:
    """Check if a stop-loss has been triggered.

    Args:
        current_low: Today's intraday low price.
        stop_price: The stop-loss level.

    Returns:
        True if stop is triggered (low ≤ stop_price).
    """


def compute_drawdown_scalar(
    current_equity: float,
    peak_equity: float,
    brake_threshold: float,
    halt_threshold: float,
) -> tuple[float, str]:
    """Compute the drawdown sizing scalar and mode.

    Args:
        current_equity: Current portfolio value.
        peak_equity: High-water mark.
        brake_threshold: Drawdown level where brake engages (e.g., 0.15).
        halt_threshold: Drawdown level where trading halts (e.g., 0.20).

    Returns:
        Tuple of (scalar: float 0.0–1.0, mode: "normal"|"brake"|"halt").
    """
```

### 7.2 Usage Pattern in backtest_engine

```python
# At start of backtest
risk_state = RiskState.initial(config.capital.initial_capital)

# Daily loop
for date in trading_days:
    # 1. Mark-to-market all positions
    current_equity = compute_mtm(...)

    # 2. Update risk state
    risk_state.update(current_equity, config.risk)

    # 3. Check stop-losses on all open positions
    for ticker, position in open_positions.items():
        if check_stop_loss(today_low[ticker], position.stop_price):
            exits_pending.append((ticker, "stop_loss"))

    # 4. Process exits (stop-loss, max-hold, signal reversal)
    for ticker, reason in exits_pending:
        execute_exit(ticker, reason, next_day_open)
        risk_state.remove_stop(ticker)

    # 5. Process new BUY signals (if not in halt mode)
    for signal in today_buy_signals:
        decision = size_position(
            ticker=signal.ticker,
            entry_price=next_day_open_with_slippage(signal.ticker),
            current_equity=risk_state.current_equity,
            price_history=get_price_history(signal.ticker, lookback=21),
            risk_state=risk_state,
            config=config.risk,
            position_limits=config.position_limits,
            sector_map=config.sector_map,
            open_positions=open_positions,
        )
        if decision.approved:
            execute_entry(signal.ticker, decision)
            risk_state.register_stop(signal.ticker, decision.stop_price)
        else:
            log_rejection(signal, decision.rejection_reason)
```

---

## 8. TEST PLAN

### 8.1 Unit Tests — risk_engine.py (~30 tests)

**ATR computation (6 tests):**
- `test_atr_pct_basic` — known OHLCV sequence, verify ATR% matches manual calculation
- `test_atr_pct_flat_prices` — constant prices → ATR% should be ~0 (reject)
- `test_atr_pct_insufficient_data` — fewer than lookback+1 rows → ValueError
- `test_atr_pct_single_large_move` — one big candle in otherwise flat data
- `test_atr_pct_high_volatility` — volatile data, verify ATR% > 5%
- `test_atr_pct_matches_ta_library` — cross-check against ta.volatility.AverageTrueRange

**Position sizing (10 tests):**
- `test_sizing_low_vol_stock` — low ATR → large raw weight, clamped to max
- `test_sizing_high_vol_stock` — high ATR → small position
- `test_sizing_exact_2pct_loss` — verify stop hit loses exactly 2% of equity
- `test_sizing_minimum_clamp` — raw weight below min → position at min
- `test_sizing_maximum_clamp` — raw weight above max → position at max
- `test_sizing_fractional_shares` — verify fractional shares computed correctly
- `test_sizing_whole_shares_only` — with fractional_shares=False
- `test_sizing_zero_equity` — edge case, should reject
- `test_sizing_drawdown_scalar_applied` — brake mode reduces size
- `test_sizing_halt_mode_rejects` — halt mode → always rejected

**Stop-loss (5 tests):**
- `test_stop_price_computation` — verify stop = entry × (1 - 2 × ATR%)
- `test_stop_triggered_exact` — low == stop_price → triggered
- `test_stop_not_triggered` — low > stop_price → not triggered
- `test_stop_gap_through` — low < stop_price (gap-down)
- `test_stop_negative_price_guard` — stop can't be negative

**Drawdown brake (6 tests):**
- `test_drawdown_normal_mode` — DD < brake → scalar = 1.0
- `test_drawdown_brake_onset` — DD exactly at brake → scalar = 1.0 (inclusive)
- `test_drawdown_brake_midpoint` — DD halfway between brake and halt
- `test_drawdown_halt_mode` — DD ≥ halt → scalar = 0.0
- `test_drawdown_recovery` — DD crosses back below brake → normal
- `test_drawdown_peak_only_increases` — peak never decreases even during drawdown

**Rejection conditions (5 tests):**
- `test_reject_halt_mode` — drawdown halt rejects all trades
- `test_reject_insufficient_history` — not enough price data
- `test_reject_zero_atr` — flat price history
- `test_reject_below_minimum_after_brake` — brake reduces below min
- `test_reject_already_held` — duplicate position blocked

### 8.2 Unit Tests — risk_state.py (~10 tests)

- `test_initial_state` — verify starting values
- `test_update_new_peak` — equity above prior peak updates peak
- `test_update_drawdown` — equity below peak computes correct DD%
- `test_update_mode_transitions` — normal → brake → halt → brake → normal
- `test_register_and_remove_stop` — add/remove stops correctly
- `test_atr_cache` — cache stores and retrieves values
- `test_multiple_stops` — multiple concurrent stops tracked
- `test_zero_peak_equity` — edge case handling
- `test_drawdown_scalar_boundary_values` — exact threshold values
- `test_state_immutability_where_expected` — mutable fields are mutable, right types

### 8.3 Integration Tests (~10 tests)

- `test_phase2_backtest_runs` — full backtest completes without error
- `test_phase2_vs_phase1_equity` — dynamic sizing produces different (not identical) equity curve
- `test_stop_losses_appear_in_trade_log` — stop-loss exits recorded with correct reason
- `test_drawdown_brake_engages_in_losing_streak` — synthetic losing streak triggers brake
- `test_position_sizes_vary_by_volatility` — tech stock vs. utility stock get different sizes
- `test_gap_through_stop_exit_at_next_open` — exit price is next-day open, not stop
- `test_rejection_log_populated` — rejected trades appear with reasons
- `test_all_profiles_backtest` — aggressive, moderate, conservative all run
- `test_risk_state_persists_across_days` — peak equity maintained correctly
- `test_force_close_with_stops` — end-of-backtest force-close still works with stops

### 8.4 Stress Tests (~5 tests)

- `test_synthetic_crash_scenario` — 10% daily drops for 5 consecutive days
- `test_all_stops_trigger_same_day` — every position hits stop simultaneously
- `test_extreme_atr` — ATR% = 50% (penny stock behavior)
- `test_minimal_capital` — $2,000 starting capital, verify min position math
- `test_maximum_positions` — fill portfolio to gross exposure limit

---

## 9. IMPLEMENTATION ORDER

### Step 1: risk_state.py (Day 1)
Create the `RiskState` dataclass and `PositionDecision` dataclass. These are pure data containers with no external dependencies. Write all risk_state unit tests. Run and pass.

### Step 2: risk_engine.py — ATR computation (Day 1)
Implement `compute_atr_pct()` using the `ta` library. Write ATR unit tests. Run and pass.

### Step 3: risk_engine.py — Sizing and stops (Day 1–2)
Implement `size_position()`, `check_stop_loss()`, `compute_drawdown_scalar()`. Write sizing, stop-loss, and drawdown unit tests. Run and pass.

### Step 4: backtest_engine.py integration (Day 2)
Modify backtest_engine to:
- Accept a `use_risk_engine: bool = False` parameter (backward compatible)
- When True: call `size_position()` instead of fixed 5%
- Add stop-loss checking to the daily loop
- Track RiskState across the backtest
- Record stop-loss events in the trade log (new `exit_reason="stop_loss"` value)

Write integration tests. Run full test suite (all existing + new).

### Step 5: run_phase2.py entry point (Day 2)
Create a Phase 2 runner that:
- Loads cached signals (same as Phase 1)
- Runs backtest with `use_risk_engine=True`
- Prints comparison table: Phase 1 vs Phase 2 metrics
- Saves trade log and equity curve

### Step 6: Validation and stress testing (Day 3)
- Run Phase 2 backtest on 2024 data
- Compare all success criteria against Phase 1 baseline
- Run stress tests with synthetic data
- Spot-check 20 random trades for correct sizing
- Verify stop-loss events are realistic
- Document results in PHASE2_RESULTS.md

### Step 7: Update documentation (Day 3)
- Update PROJECT_GUIDE.md with Phase 2 results
- Update FPPE_TRADING_SYSTEM_DESIGN.md to v0.4
- Update CLAUDE.md with new test count and module list
- Update trading_system/__init__.py exports

---

## 10. KNOWN RISKS AND EDGE CASES

### 10.1 Gap-Through Stop Losses

When a stock gaps down through the stop price (e.g., earnings miss), the actual loss exceeds the intended 2% max. The sizing formula assumes loss AT the stop price. A 10% overnight gap below the stop could cause a 4–5% equity loss on a single trade.

**Mitigation:** This is a known and accepted risk. The 2% target is a *typical case* budget, not a guarantee. The max_position_pct clamp (10%) sets an absolute ceiling — even a 100% loss on a 10% position only costs 10% of equity. Gap-through events should be rare in large-cap equities.

### 10.2 ATR Regime Sensitivity

ATR is backward-looking. A low-vol stock that enters an earnings period has artificially low ATR, causing oversized positions right before a volatility event.

**Mitigation:** Not addressed in Phase 2. Phase 6 (candlestick categorization) may add volume/volatility regime awareness. For now, the max_position_pct clamp limits damage.

### 10.3 Drawdown Brake in Bull Markets

In a strong bull market (like 2024), the drawdown brake may never engage, making it impossible to validate. Phase 2 success criteria includes a synthetic stress test specifically to exercise the brake logic.

### 10.4 ATR Computation on Sparse Data

Some tickers may have trading halts or missing data within the lookback window. The `ta` library handles NaN by forward-filling, which could produce stale ATR values.

**Mitigation:** If more than 20% of the lookback window is NaN, reject the trade. This check should be explicit in `compute_atr_pct()`.

### 10.5 Interaction with Cooldown System

Phase 1's cooldown system (3-day cooldown after stop or max-hold exit) remains unchanged. Stop-loss exits in Phase 2 trigger the same `cooldown_after_stop_days` cooldown. The risk engine does NOT modify cooldown logic — it only sizes and sets stops.

---

## 11. TRADE-OFF ANALYSIS

| Decision | Alternative | Why This Choice |
|----------|-------------|-----------------|
| Pure volatility sizing (no confidence) | Confidence × volatility hybrid | Prevents double-counting. Confidence used in signal gate + future ranking. Adding it to sizing would triple-count. |
| Fixed 2× ATR stop | Adaptive stops (1.5× in low vol, 2.5× in high vol) | Simpler to validate. Adaptive stops add a parameter that must be swept. 2× is the industry standard starting point. |
| Linear drawdown brake | Step function (binary on/off at 15%) | Avoids cliff-edge sizing jumps that cause erratic behavior at the threshold boundary. |
| Next-day open for stop exits | Stop price as exit (ideal fill) | Conservative and realistic. Gap-through scenarios are common in equity markets. Using stop price would overstate performance. |
| Stateless risk_engine (pure function) | SharedState object from design doc | Phase 2 only needs drawdown + ATR. SharedState adds complexity for Phase 3+ features. Build the simple version first. |
| ta library for ATR | Manual computation | Already a dependency. Battle-tested. Reduces implementation risk and code to maintain. |
| Redundant sector/exposure checks | Single enforcement point | Defense in depth. If backtest_engine's check is accidentally removed, risk_engine still catches it. |

---

## 12. DEPENDENCY GRAPH

```
risk_engine.py
├── imports: pandas, numpy, ta (existing deps — no new packages)
├── imports: trading_system.config (RiskConfig, PositionLimitsConfig)
├── imports: trading_system.risk_state (RiskState, PositionDecision, StopLossEvent)
└── imported by: trading_system.backtest_engine

risk_state.py
├── imports: dataclasses (stdlib only)
└── imported by: trading_system.risk_engine, trading_system.backtest_engine

backtest_engine.py (modified)
├── existing imports unchanged
├── NEW import: trading_system.risk_engine
├── NEW import: trading_system.risk_state
└── NEW parameter: use_risk_engine: bool = False
```

No new pip packages required. All dependencies (`ta`, `pandas`, `numpy`) are already in requirements.txt.

---

## 13. FILE-BY-FILE CHANGE SUMMARY

| File | Action | Changes |
|------|--------|---------|
| `trading_system/risk_state.py` | **CREATE** | RiskState, PositionDecision, StopLossEvent dataclasses |
| `trading_system/risk_engine.py` | **CREATE** | size_position, compute_atr_pct, check_stop_loss, compute_drawdown_scalar |
| `trading_system/backtest_engine.py` | **MODIFY** | Add use_risk_engine param, stop-loss loop, RiskState tracking |
| `trading_system/__init__.py` | **MODIFY** | Export new symbols |
| `trading_system/run_phase2.py` | **CREATE** | Phase 2 entry point with comparison output |
| `tests/test_risk_engine.py` | **CREATE** | ~30 unit tests |
| `tests/test_risk_state.py` | **CREATE** | ~10 unit tests |
| `tests/test_phase2_integration.py` | **CREATE** | ~15 integration + stress tests |
| `PROJECT_GUIDE.md` | **MODIFY** | Phase 2 results, updated module list |
| `FPPE_TRADING_SYSTEM_DESIGN.md` | **MODIFY** | Bump to v0.4, add Phase 2 completion notes |
| `CLAUDE.md` | **MODIFY** | Updated test count, new module references |

---

*Phase 2 System Design v1.0 — March 19, 2026*
*Designed for implementation by any AI or human developer with access to the FPPE codebase.*
*Resolve ISSUE 0.1 (missing Phase 1 files) before beginning implementation.*
