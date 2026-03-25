# Triggered Ingestion / Streaming Scaffolding Design
# SLE-79 — Linear: https://linear.app/sleepern/issue/SLE-79

## Summary

Architecture design for event-triggered ingestion and streaming/stateful
feature computation.  This document covers trade-offs, interface definitions,
and compatibility with the current overnight batch pipeline.

---

## 1. Current Architecture (Batch, EOD)

```
[yfinance API] → daily_download()
    → validate_ohlcv()   (Pandera)
    → compute_features() (ta library — ATR, RSI, SMA)
    → atomic_write()     (parquet)
    → PatternMatcher.fit() / .query()
    → signals CSV
```

Runs once per night (~18:00 ET).  Total latency: ~8 minutes for 52 tickers.
Data is processed in full-table batch even if only 1 ticker changed.

**Limitation:** No intraday updates.  A news event at 10:00 cannot trigger
a signal until next-day EOD.

---

## 2. Event-Triggered Ingestion Design

### Trigger Types

| Trigger | Source | Latency | Use Case |
|---------|--------|---------|---------|
| EOD close | Market close 16:00 ET | Seconds | Current overnight pipeline |
| Intraday price alert | Broker WebSocket | Milliseconds | Intraday pattern update |
| Earnings release | SEC EDGAR XBRL | Minutes | Feature refresh post-announcement |
| Corporate action | Broker event stream | Minutes | Dividend/split adjustment |

### Interface Definitions

```python
class IngestionEvent(Protocol):
    """Minimal event contract for the triggered ingestion pipeline."""
    ticker: str
    event_type: str          # "eod_close" | "intraday" | "earnings" | "corp_action"
    timestamp: datetime
    payload: dict[str, Any]  # Event-specific data


class IngestionHandler(ABC):
    """Process a single ingestion event and update the feature store."""

    @abstractmethod
    def can_handle(self, event: IngestionEvent) -> bool:
        """True if this handler knows how to process this event type."""

    @abstractmethod
    def handle(self, event: IngestionEvent, feature_store: FeatureStore) -> None:
        """Process event and update feature_store in place (atomic write)."""


class FeatureStore(Protocol):
    """Read/write interface for the FPPE feature database."""

    def get_ticker(self, ticker: str) -> pd.DataFrame: ...
    def update_ticker(self, ticker: str, df: pd.DataFrame) -> None: ...
    def get_all_tickers(self) -> list[str]: ...
```

### Stateful Feature Computation

Rolling features (ATR, RSI, SMA) require history.  In streaming mode:

```
Event arrives for AAPL →
    Load last 200 AAPL OHLCV rows from FeatureStore
    Append new row
    Recompute rolling features for last 1 row only (incremental)
    Validate new row (Pandera)
    Atomic write back
```

**Incremental ATR computation** (avoids full-table recompute):
```python
atr_prev = store.get_ticker(ticker)["ATR"].iloc[-1]
true_range = max(high - low, abs(high - close_prev), abs(low - close_prev))
atr_new = (atr_prev * (n - 1) + true_range) / n   # Wilder's smoothing
```

---

## 3. Compatibility with Overnight Pipeline

The streaming pipeline must be additive — it does NOT replace the overnight
batch.  The overnight batch remains the authoritative rebuild of the full
feature store.  Streaming updates are *incremental patches* valid until the
next overnight refresh.

```
Overnight batch → full feature store rebuild (authoritative)
Streaming triggers → incremental patches during the trading day
                   → discarded at next overnight rebuild
```

**Consistency guarantee:** Any streaming patch must pass the same Pandera
validation as the overnight batch.  Invalid patches are logged and dropped.

---

## 4. Recommended Implementation Sequence

1. **Phase 1 (M8/M9):** Add `IngestionEvent` and `FeatureStore` interfaces.
2. **Phase 2:** Implement `EODCloseHandler` (replaces current overnight polling).
3. **Phase 3:** Add `IntradayHandler` using broker WebSocket data.
4. **Phase 4:** Add `EarningsHandler` using SEC EDGAR XBRL feed.

**Not recommended for FPPE v1:** Full real-time streaming (sub-second updates)
would require replacing the BallTree/HNSW index with an online learning
variant.  The current PatternMatcher is a batch-fit model — full refit takes
~0.4s for N=50k, acceptable at EOD but not intraday.

---

## 5. Trade-off Summary

| Approach | Latency | Complexity | Risk |
|----------|---------|------------|------|
| EOD batch (current) | 8 hours | Low | None |
| Triggered EOD | Minutes | Low | Low |
| Triggered intraday | Minutes | Medium | Medium |
| Real-time streaming | Seconds | High | High |

**Recommendation:** Triggered intraday (Phase 3) is the best near-term target.
Full real-time streaming deferred to a future phase pending PatternMatcher
online-learning capability.

---

## 6. Files

| File | Purpose |
|------|---------|
| `docs/rebuild/STREAMING_INGESTION_DESIGN.md` | This document |
| `rebuild_phase_3z/fppe/pattern_engine/data.py` | Current DataLoaderHardened |
| `.claude/skills/add-ticker.md` | Ticker onboarding protocol |
