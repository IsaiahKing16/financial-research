# Phase 10 Infrastructure Decision: NautilusTrader Evaluation Plan

**Created:** 2026-04-10  
**Status:** QUEUED — decision point at Phase 9/10 boundary (target Q1 2027)  
**Prerequisite:** Phase 5 custom execution layer operational ✓ (2026-04-09) + Phase 8 paper trading (pending)  
**Source:** "NautilusTrader: Architectural Review for Systematic Equity Trading with IBKR Execution" research paper synthesis  

---

## Context

FPPE's production infrastructure path involves two sequential decisions:

1. **Phase 5 (COMPLETE, 2026-04-09):** Built a custom execution layer with `BaseBroker` ABC, `MockBroker`, `OrderManager`, and reconciliation. Completed: G1 100/100 fills, G2 30-day recon pass, G3 0.18s. MockBroker, OrderManager, and reconciliation operational.
2. **Phase 10 (Q1 2027):** Before deploying real capital, evaluate whether NautilusTrader should replace the custom execution layer.

NautilusTrader (v1.225.0, April 2026) is the most architecturally aligned Python-native framework for FPPE's specific use case: daily signal-based equity trading with IBKR execution across 50–500 tickers on Python 3.12 / Windows 11. However, its IBKR adapter is beta-quality with 15+ critical bug fixes in just two releases (v1.223.0–v1.224.0). The alternatives — Lean (C# foundation, Python 3.11 ceiling), Zipline (no live trading), VectorBT (research only) — each carry larger structural compromises.

**The decision is timing, not direction.** NautilusTrader is the correct long-term architecture if its IBKR adapter stabilizes. The question is whether it stabilizes before FPPE needs to go live.

---

## Evidence Summary from Research Paper

### What NautilusTrader provides that FPPE needs

| Capability | NautilusTrader | FPPE Custom (Phase 5) |
|------------|---------------|----------------------|
| Backtest-to-live code parity | Genuine — same `Strategy` subclass in `BacktestEngine` and `TradingNode` | Separate backtest and live code paths |
| Custom data injection (KNN signals) | First-class `Data` subclass with `on_data()` handler, chronological merge with price bars | Custom signal pipeline, manual event ordering |
| Order management state machine | Built-in: PENDING → SUBMITTED → FILLED / REJECTED / CANCELLED | Must build: `trading_system/order_manager.py` |
| Position reconciliation | Built-in (buggy for IBKR — active stabilization) | Must build: `scripts/reconcile.py` |
| Risk engine | Built-in: rate limiting, notional limits, trading state controls | Must build: guards in strategy logic |
| Fill simulation (backtest) | `SimulatedExchange` with `OrderMatchingEngine`, `FillModel`, `LatencyModel` | `MockBrokerAdapter` — simpler, less realistic |
| Kelly position sizing | `PositionSizer` ABC — subclass and override `calculate()` | Standalone module (portable) |
| Portfolio-level constraints | NOT built-in — strategy logic required | Same — strategy logic required |
| Windows 11 / Python 3.12 | Officially supported, pre-built 94MB wheel, standard-precision only (64-bit, 9 decimal places) | Native — no platform concerns |
| Performance (daily bars) | Trivially fast — 625K bars in <1 second at 5M rows/sec claimed throughput | Also trivially fast — daily bars are not a bottleneck for any framework |

### What NautilusTrader's IBKR adapter gets wrong (as of v1.225.0)

| Issue | Severity | Reference |
|-------|----------|-----------|
| Position reconciliation fragility — phantom duplicate orders on restart with Redis persistence | Critical | GitHub #3176 |
| Connection instability — bar data resubscription fails with `TypeError` after IB routine disconnects (every 4–30 min) | Critical | GitHub #3001 |
| External order ID collision — TWS assigns `orderId=0` to external orders, causing misattributed fills | High | GitHub #3465 |
| Synthetic `OrderStatusReport` objects with random UUIDs and hardcoded `LIMIT` types | High | GitHub #3176 |
| Flat positions closed externally leaving phantom positions causing "downstream mayhem" | High | GitHub #3023 |

**Fix velocity:** 12+ IBKR fixes in v1.223.0 (Feb 2026), 3+ in v1.224.0 — rapid stabilization in progress. Dedicated contributor @shzhng systematically addressing IBKR issues since late 2025.

### Risk assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| IBKR adapter not production-ready by Q1 2027 | Medium (40% probability) | Retain custom adapter as fallback; migration to NautilusTrader is additive, not required |
| Bus-factor: single primary architect (Chris Sellers, Nautech Systems) | Medium | Modular Rust crate structure, NautilusTrader Pro commercial tier provides sustainability path |
| Breaking API changes between releases | Medium | Pin version; only upgrade during scheduled maintenance windows |
| Windows standard-precision limitation (64-bit, 9 decimal places) | None | Daily equity trading uses 2–4 decimal places maximum |
| Redis dependency for state persistence | Low | Optional — Docker or WSL2 provides Redis on Windows if needed |

---

## Decision Framework: Phase 9/10 Boundary Evaluation

### When to evaluate

At the completion of Phase 8 paper trading (30–60 days of simulated live operation), before committing real capital in Phase 10.

### Evaluation criteria

| Criterion | Adopt NautilusTrader | Keep Custom Adapter |
|-----------|---------------------|-------------------|
| IBKR bug fix frequency (last 3 releases) | ≤ 2 IBKR-specific fixes per release (stabilized) | > 5 IBKR fixes per release (still churning) |
| Position reconciliation | Zero phantom position issues reported in last 6 months | Reconciliation bugs still open |
| Connection stability | Survives IB disconnects without data loss for 30+ consecutive days | Disconnect recovery still unreliable |
| API stability | No breaking changes in last 3 releases | Breaking changes in recent release |
| Version | v1.230+ or v2.x stable milestone reached | Still on v1.22x with "active development" warnings |
| Community IBKR usage | ≥ 3 independent users reporting successful unattended equity trading | Only supervised/paper trading reports |
| Custom adapter maturity | Custom adapter has had significant operational issues during Phase 8 | Custom adapter passed Phase 8 paper trading cleanly |

**Decision rule:** Adopt NautilusTrader if ≥ 5 of 7 criteria favor adoption AND custom adapter had operational issues during Phase 8. Keep custom adapter if it passed Phase 8 cleanly and NautilusTrader stability is uncertain. The bias is toward the working system — switching costs are real.

### Migration cost estimate (if adopting)

| Component | Effort | Risk |
|-----------|--------|------|
| Rewrite `KNNStrategy` as NautilusTrader `Strategy` subclass | 2–3 days | Low — clean mapping from `on_data()` pattern |
| Port Kelly `PositionSizer` module | 1 day | Low — subclass `PositionSizer`, override `calculate()` |
| Port reconciliation logic | 0 days | Zero — NautilusTrader provides built-in reconciliation |
| Port order management state machine | 0 days | Zero — NautilusTrader provides built-in order lifecycle |
| Implement portfolio-level constraints in strategy logic | 1 day | Low — same approach in both architectures |
| Convert data pipeline to NautilusTrader `Data` subclasses + `ParquetDataCatalog` | 2 days | Medium — format conversion, timestamp alignment |
| Integration testing (30 days paper trading through NautilusTrader) | 30 days | Medium — discovering NautilusTrader-specific edge cases |
| **Total** | **~5 days code + 30 days validation** | |

---

## Phase 5 Compatibility Requirements

**Phase 5 Status (2026-04-10):** Phase 5 is COMPLETE. The compatibility constraints below should be verified against the implemented code.

To minimize migration cost if NautilusTrader is adopted at Phase 10, Phase 5's custom execution layer must follow these design constraints:

### Architectural requirement: signal-execution separation

```
Signal Layer (KNN + calibration + conformal)
    │
    ▼
Signal Interface (algorithm-agnostic, execution-agnostic)
    │
    ▼
Execution Layer (BaseBroker ABC — swappable)
    ├── MockBrokerAdapter (Phase 5–8)
    ├── IBKRAdapter via ib_async (Phase 5–9)
    └── [Future] NautilusTrader TradingNode (Phase 10 if adopted)
```

The signal interface must be a clean data contract — not embedded in execution logic. If adopting NautilusTrader, the signal interface maps to a custom `Data` subclass consumed by `on_data()`.

### Specific compatibility constraints for Phase 5

| Constraint | Rationale | Impact |
|------------|-----------|--------|
| Kelly sizing as standalone module with pure-function `calculate(equity, probability, edge) -> quantity` | Portable to NautilusTrader `PositionSizer` subclass | ✓ Implemented in `trading_system/position_sizer.py` |
| Signal output as a dataclass with `timestamp`, `ticker`, `direction`, `probability`, `confidence_set` | Maps to NautilusTrader `Data` subclass with `ts_event`, `ts_init` | Verify against `pattern_engine/contracts/` |
| `BaseBroker` ABC methods: `submit_order()`, `cancel_order()`, `get_positions()`, `get_portfolio_value()` | Minimal interface that both custom adapter and NautilusTrader wrapper can implement | ✓ Implemented in `trading_system/broker/base_broker.py` |
| Order state machine as standalone module (not embedded in broker adapter) | NautilusTrader provides its own state machine — custom one must be bypassable | ✓ Implemented in `trading_system/order_manager.py` |
| Reconciliation as a standalone script comparing expected vs actual positions | NautilusTrader has built-in reconciliation — custom one runs independently | ✓ Implemented in `trading_system/reconciliation.py` |

### Implementation spec for compatibility layer

```xml
<handoff id="phase5-nautilus-compat">
  <goal>
    Ensure Phase 5 execution layer is designed so that migrating to NautilusTrader 
    at Phase 10 requires swapping the broker adapter — not rewriting strategy logic, 
    sizing, or signal generation.
    
    Success criteria:
    - Signal output is a serializable dataclass independent of any execution framework
    - Kelly sizing module has zero imports from trading_system/broker/
    - BaseBroker ABC has ≤ 6 methods (submit, cancel, get_positions, get_portfolio, 
      get_order_status, connect)
    - Order manager is composable (can be bypassed if NautilusTrader provides its own)
    - Reconciliation script reads from BaseBroker.get_positions(), not directly from 
      broker API
  </goal>
  
  <files>
    pattern_engine/signal.py — Signal dataclass (ts, ticker, direction, prob, conf_set)
    trading_system/sizing.py — Kelly module, pure function, no broker imports
    trading_system/broker/base.py — BaseBroker ABC (≤ 6 methods)
    trading_system/broker/mock.py — MockBrokerAdapter implements BaseBroker
    trading_system/broker/ibkr.py — IBKRAdapter implements BaseBroker via ib_async
    trading_system/order_manager.py — Standalone state machine, composable
    scripts/reconcile.py — Reads from BaseBroker.get_positions()
  </files>
  
  <steps>
    <step n="1">
      Define Signal dataclass in pattern_engine/signal.py:
      - Fields: timestamp (datetime), ticker (str), direction (Enum: UP/DOWN/HOLD),
        probability (float), confidence_set (frozenset), regime_prob (float)
      - Must be JSON-serializable (for logging and NautilusTrader Data subclass conversion)
      - No imports from trading_system/ — signal layer is execution-agnostic
    </step>
    
    <step n="2">
      Implement Kelly sizing in trading_system/sizing.py:
      - Pure function: calculate(equity: float, probability: float, 
        win_loss_ratio: float, max_fraction: float = 0.02) -> float
      - Returns dollar position size
      - Zero imports from trading_system/broker/
      - Future NautilusTrader port: wrap in PositionSizer subclass that calls this function
    </step>
    
    <step n="3">
      Define BaseBroker ABC in trading_system/broker/base.py:
      - connect() -> None
      - submit_order(signal: Signal, quantity: float) -> OrderId
      - cancel_order(order_id: OrderId) -> bool
      - get_positions() -> dict[str, Position]
      - get_portfolio_value() -> float
      - get_order_status(order_id: OrderId) -> OrderStatus
      - No additional methods — keep interface minimal for portability
    </step>
    
    <step n="4">
      Implement OrderManager as composable module:
      - Tracks state: PENDING → SUBMITTED → FILLED / REJECTED / CANCELLED
      - Can be instantiated or bypassed (NautilusTrader provides its own)
      - Receives order events from BaseBroker, updates internal state
      - Does NOT call BaseBroker methods directly — receives callbacks
    </step>
    
    <step n="5">
      Implement reconciliation as BaseBroker-agnostic script:
      - Calls BaseBroker.get_positions() for actual state
      - Compares against expected state from SharedState/PortfolioSnapshot
      - Mismatch > 0.05% → blocks execution, logs alert
      - Works identically regardless of which BaseBroker implementation is active
    </step>
  </steps>
  
  <verification>
    pytest tests/test_signal_dataclass.py -v  # Signal is JSON-serializable, no broker imports
    pytest tests/test_sizing_isolation.py -v   # Sizing module has zero broker imports
    pytest tests/test_broker_abc.py -v         # ABC has exactly 6 methods
    pytest tests/test_order_manager_composable.py -v  # Can run with/without OrderManager
    pytest tests/test_reconcile_broker_agnostic.py -v  # Reconciliation works with MockBroker
    
    # Import isolation check (automated)
    python -c "import ast, sys; tree=ast.parse(open('trading_system/sizing.py').read()); \
      imports=[n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]; \
      assert 'trading_system.broker' not in str(imports), 'Sizing has broker dependency'"
  </verification>
  
  <task_type>SR — architectural design with long-term migration implications.
  Wrong interface boundaries here cost weeks of rework at Phase 10.</task_type>
</handoff>
```

---

## Locked Decisions

| Decision | Status | Rationale |
|----------|--------|-----------|
| Phase 5 broker library: `ib_async` | **LOCKED** | Lightweight, async-native, actively maintained successor to `ib_insync` |
| Phase 5 architecture: custom `BaseBroker` ABC | **LOCKED** | Full control during paper trading; NautilusTrader dependency deferred |
| NautilusTrader evaluation: Phase 9/10 boundary | **LOCKED** | IBKR adapter needs 12+ months of stabilization before trust |
| Signal-execution separation: mandatory | **LOCKED** | Enables framework swap without strategy rewrite |
| Kelly sizing: standalone module, no broker imports | **LOCKED** | Portable to any execution framework |
| Alpaca API: Phase 5 fallback if IBKR too complex | **Existing roadmap fallback** | REST-based, simpler than TWS API |
| VectorBT for signal research, NautilusTrader for production | **Design principle** | Different tools for different phases of the pipeline |
| Windows standard-precision (64-bit): confirmed adequate | **No action needed** | Daily equity prices use ≤ 4 decimal places |

---

## Carry-Forward Items

| Item | Status | Blocks | Phase |
|------|--------|--------|-------|
| NautilusTrader adoption decision | **Deferred to Phase 9/10 boundary (Q1 2027)** | Phase 10 live deployment | Phase 10 |
| Phase 5 `BaseBroker` ABC designed for NautilusTrader compatibility | ✓ **IMPLEMENTED** in `trading_system/broker/base_broker.py` | Phase 5 implementation | Phase 5 |
| Monitor NautilusTrader IBKR adapter stability: track release notes for IBKR fix frequency | **Ongoing monitoring task** | Phase 10 decision | Ongoing |
| `ib_async` locked as Phase 5 broker library | **LOCKED** — Phase 5 uses MockBroker; IBKRAdapter is Phase 8+ work | Phase 5 | Phase 5 |
| Alpaca API as fallback | **Existing fallback** | Phase 5 | Phase 5 |
| Kelly sizing as standalone portable module | ✓ **IMPLEMENTED** in `trading_system/position_sizer.py` | Phase 3 (sizing) + Phase 5 (execution) | Phase 3/5 |
| Signal dataclass as framework-agnostic contract | Verify: check `pattern_engine/contracts/` | Phase 5 | Phase 5 |
| Transaction cost baseline: 5 bps/side moderate, sensitivity at 2.5 and 9.5 bps | **From TCA paper** — cross-referenced in NautilusTrader review | Backtesting cost model | Phase 2+ |
| NautilusTrader bus-factor risk: single architect + self-funded | **Risk noted** — mitigated by modular codebase and commercial tier | Phase 10 decision | Phase 10 |
| NautilusTrader `ParquetDataCatalog` for multi-instrument data management | **Evaluate at Phase 10** — may replace custom data pipeline | Phase 10 | Phase 10 |
| NautilusTrader `ExecAlgorithm` for TWAP/iceberg execution | **Evaluate at Phase 10** — relevant for larger position sizes | Phase 10 | Phase 10 |

---

## Monitoring Checklist (Quarterly Through Phase 10)

Track these indicators each quarter to inform the Phase 10 decision:

```
[ ] NautilusTrader version: _______ (current: v1.225.0, April 2026)
[ ] IBKR-specific fixes in last release: _______ (target: ≤ 2)
[ ] Open IBKR-critical issues on GitHub: _______ (target: ≤ 3)
[ ] Last IBKR reconciliation bug report: _______ (target: > 6 months ago)
[ ] v2.x stable milestone reached: Y/N
[ ] Independent user reports of unattended IBKR equity trading: _______
[ ] README still contains "active development / breaking changes" warning: Y/N
[ ] @shzhng (IBKR contributor) still active: Y/N
[ ] NautilusTrader Pro commercial customers reported: Y/N
[ ] Python version support: _______ (must include 3.12+)
```

---

## Alternatives Assessed and Rejected

| Framework | Reason for Rejection | Reassess? |
|-----------|---------------------|-----------|
| **Lean (QuantConnect)** | C# foundation with PythonNet friction; Python 3.11 ceiling (FPPE requires 3.12); paid subscription for local live trading ($28/month minimum). Most mature IBKR integration (20+ brokers), lowest bus-factor (250K+ users, 180+ contributors). | Only if NautilusTrader AND custom adapter both fail at Phase 10. |
| **Zipline-reloaded** (v3.1.1) | No live trading capability. No IBKR integration. Single maintainer (Stefan Jansen). Purpose-built for daily equity backtesting — excellent at that, useless for execution. | No — fundamental capability gap. |
| **VectorBT Pro** ($25–500) | Research-only tool. Zero live trading. No broker connections, order management, or position tracking. Unmatched for parameter sweeps (millions of combinations in seconds). | Not as execution framework. Keep for signal research (complements NautilusTrader). |
| **Backtrader** | Legacy architecture, limited maintenance. No Rust performance layer. Community has largely migrated to NautilusTrader or Lean. | No. |
| **FreqTrade** | Crypto-focused. Equity support is an afterthought. Wrong domain. | No. |

---

## Key References

- NautilusTrader GitHub: 21,700+ stars, 2,600+ forks, 127+ contributors, v1.225.0 (April 6, 2026)
- Chris Sellers (cjdsellers): founder/CEO Nautech Systems, Engineering Manager at Databento
- NautilusTrader IBKR issues: #3176 (reconciliation), #3001 (connection), #3465 (order ID collision), #3023 (phantom positions)
- IBKR third-party listing: NautilusTrader listed on IB's official providers page
- Malkov & Yashunin (IEEE TPAMI 2018): HNSW — relevant for KNN integration with NautilusTrader data pipeline
- Almgren & Chriss (2000): optimal execution framework — NautilusTrader's `ExecAlgorithm` implements variants
- Transaction Cost Analysis paper: 5 bps/side moderate baseline for IBKR Pro fixed pricing
