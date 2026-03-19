# Candlestick Pattern Categorization Module
## Architecture Design — FPPE Scaling Solution

**Version:** 0.1
**Status:** Design Phase
**Context:** FPPE_TRADING_SYSTEM_DESIGN.md v0.2, Section 6 (Future Scaling)

---

## 1. The Problem

The K-NN analogue matching engine searches the **entire training database** for every
signal query. At 52 tickers × 252 trading days = 13,104 query rows, this is manageable.
But the user's goal is to expand the ticker universe exponentially. At 500 tickers, that's
~126,000 queries per year against the same training database — search time scales linearly.

The deeper problem: most training rows are *structurally dissimilar* to any given query.
Searching all of them wastes computation and dilutes the K-NN quality by including distant
analogues that weaken signal resolution.

---

## 2. The Proposed Solution: Pre-Categorization

Before searching, classify each row (both training and query) into a **candlestick pattern
category**. At query time, only search within the matching category. This:

1. **Reduces search space** proportionally to the number of categories (if 10 categories,
   ~10× speedup at same accuracy)
2. **Improves signal quality** by ensuring K-NN analogues are structurally similar
   (a "doji + gap-up" shouldn't match against a "long bear candle + high volume")
3. **Enables category-specific calibration** — Platt calibrators can be fit per-category
   instead of globally, improving probability accuracy for each pattern type

---

## 3. Architecture

```
                    ┌─────────────────────────────────────┐
                    │        CandlestickClassifier         │
                    │                                      │
                    │  Input: OHLC + Volume (single row)   │
                    │  Output: CategoryID (0-N)            │
                    │          CategoryName (string)       │
                    │          PatternFeatures (dict)      │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┼──────────────────────┐
              │                       │                      │
              ▼                       ▼                      ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │  CategorizedDB   │   │  CategoryIndex   │   │ CategoryCalibra- │
    │  (training rows  │   │  (fast lookup:   │   │ tors (per-       │
    │  tagged with      │   │  category →      │   │ category Platt   │
    │  category ID)    │   │  row subset)     │   │ scalers)         │
    └─────────┬────────┘   └──────────────────┘   └──────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────┐
    │         FPPE K-NN Matching (unchanged)       │
    │                                              │
    │  BEFORE: search all 175k training rows       │
    │  AFTER:  search only matching category rows  │
    │          (estimated 10k-30k per category)    │
    └──────────────────────────────────────────────┘
```

---

## 4. Pattern Categories

Derived from classical technical analysis, adapted for algorithmic use.
Each category maps to a distinct market microstructure state.

### Tier 1: Trend State (3 categories)
The broadest classification — primary market direction.

| Category | Definition | Signal Implication |
|----------|-----------|-------------------|
| `BULL_TREND` | Close > 20d MA, 7d return > 1% | Momentum may continue |
| `BEAR_TREND` | Close < 20d MA, 7d return < -1% | Continuation or reversal setup |
| `SIDEWAYS` | abs(7d return) < 1%, ATR/Price < 1.5% | Range-bound; mean reversion likely |

### Tier 2: Session Structure (6 categories)
The shape of the current candle's body and wicks.

| Category | Body/Range Ratio | Wick Pattern |
|----------|-----------------|--------------|
| `STRONG_BULL` | Body/Range > 0.7, Close near High | Small lower wick |
| `STRONG_BEAR` | Body/Range > 0.7, Close near Low | Small upper wick |
| `DOJI` | Body/Range < 0.1 | Indecision — roughly equal wicks |
| `HAMMER` | Lower wick > 2× body, Close in upper 30% | Potential reversal signal |
| `SHOOTING_STAR` | Upper wick > 2× body, Close in lower 30% | Potential reversal signal |
| `SPINNING_TOP` | Body/Range 0.1-0.4, significant both wicks | High uncertainty |

### Tier 3: Multi-Day Pattern (12 categories)
Requires 3-5 days of prior candles. More computationally intensive but highest signal quality.

| Category | Pattern Description |
|----------|-------------------|
| `ENGULFING_BULL` | Current candle body engulfs prior candle; from downtrend |
| `ENGULFING_BEAR` | Current candle body engulfs prior candle; from uptrend |
| `MORNING_STAR` | 3-candle: bear → doji → bull |
| `EVENING_STAR` | 3-candle: bull → doji → bear |
| `THREE_WHITE` | Three consecutive bull candles, each closing higher |
| `THREE_BLACK` | Three consecutive bear candles, each closing lower |
| `GAP_UP` | Open > prior High; gap not filled intraday |
| `GAP_DOWN` | Open < prior Low; gap not filled intraday |
| `INSIDE_BAR` | Entire candle within prior candle's range (compression) |
| `OUTSIDE_BAR` | Candle range exceeds prior candle on both sides |
| `BREAKOUT` | Close above N-day high with above-average volume |
| `BREAKDOWN` | Close below N-day low with above-average volume |

---

## 5. Implementation Plan

### Phase A: Classifier Module (`pattern_classifier.py`)

```python
# Conceptual interface — not yet implemented
class CandlestickClassifier:
    def classify(self, ohlcv_window: pd.DataFrame) -> PatternResult:
        """
        Args:
            ohlcv_window: DataFrame with last 5 days of OHLCV for one ticker.
                          Columns: Date, Open, High, Low, Close, Volume
        Returns:
            PatternResult(
                tier1_trend: str,      # BULL_TREND / BEAR_TREND / SIDEWAYS
                tier2_session: str,    # STRONG_BULL / DOJI / etc.
                tier3_multi: str,      # ENGULFING_BULL / GAP_UP / etc. / NONE
                composite_key: str,   # e.g. "BULL_TREND|STRONG_BULL|GAP_UP"
                pattern_features: dict # numeric features for debugging
            )
```

### Phase B: Training Database Pre-tagging

Run the classifier over all rows in `train_db.parquet` and save a
`train_db_categorized.parquet` that adds `tier1`, `tier2`, `tier3`, `composite_key` columns.

This runs once. At ~175k rows, estimated runtime: under 60 seconds.

### Phase C: Category Index

Build a dict mapping `composite_key → [row_indices]` for fast subsetting.
Serialize to `train_db_index.pkl` so it loads in milliseconds.

```python
category_index = {
    "BULL_TREND|STRONG_BULL|GAP_UP": [45, 1023, 8901, ...],
    "BEAR_TREND|SHOOTING_STAR|NONE": [23, 456, ...],
    ...
}
```

### Phase D: Modified K-NN Search (FPPE integration)

In `strategy.py`, before calling `_run_matching_loop`:

```python
# NEW: filter training database to matching category
query_pattern = classifier.classify(query_ohlcv_window)
category_rows = category_index.get(query_pattern.composite_key, [])

if len(category_rows) >= MIN_MATCHES:
    train_subset = train_db.iloc[category_rows]
else:
    # Fallback: use tier1 only (broader match)
    tier1_rows = tier1_index.get(query_pattern.tier1_trend, [])
    train_subset = train_db.iloc[tier1_rows]

# K-NN search against train_subset instead of full train_db
```

### Phase E: Per-Category Calibrators

Instead of one global Platt scaler, fit one per `tier1_trend` category
(3 calibrators instead of 1). More granular calibration = better probability accuracy.

---

## 6. Tier 4/5 Analysis: Diminishing Returns and the Sparsity Problem

The natural question is whether adding Tier 4 (Volume/Volatility Regime) and Tier 5
(Market Regime / Macro State) would improve accuracy further.

**Short answer: the information in Tier 4/5 is real and useful, but rigid additional tiers
hit diminishing returns due to data sparsity, not diminishing information.**

**The math:**

| Tier Configuration | Composite Categories | Avg Rows/Category (175k training rows) | Min-Match Risk |
|-------------------|---------------------|----------------------------------------|----------------|
| Tier 1-3 only | ~21 combinations | ~8,330 | Very low |
| + Tier 4 (4 vol/volume buckets) | ~84 combinations | ~2,083 | Low |
| + Tier 5 (3 market regimes) | ~252 combinations | ~694 | **Moderate — some rare combos below 10** |
| + Tier 6 (5 sector regimes) | ~1,260 combinations | ~139 | **High — majority of combos unusable** |

The 10-match minimum is not arbitrary — it's the statistical floor below which K-NN
probabilities become unreliable. When a category has 8 rows, the system falls back to Tier
1 anyway, wasting the computation spent building Tier 4/5.

**The solution: Hierarchical Fallback + Floating Modifiers (replaces fixed Tier 4/5)**

Instead of fixed tiers, the system uses:
1. A cascade from most-specific to least-specific until ≥ MIN_MATCHES are found
2. Volume/Volatility and Market Regime as **floating modifiers** applied within a category
   when enough data exists — skipped silently when data is insufficient

```
Query classification:
  Step 1: Classify query → composite_key (Tier 1 + Tier 2 + Tier 3)
  Step 2: Apply floating modifiers (if available):
            - volume_regime:    "high" / "low" (vs. 20-day avg volume)
            - vol_regime:       "elevated" / "normal" (vs. 20-day ATR)
            - market_regime:    "bull" / "bear" / "neutral" (SPY 90d return)

  K-NN search cascade:
    A. Try: category[composite_key + all_modifiers]  → if ≥ 10 rows: DONE
    B. Try: category[composite_key + volume_regime]  → if ≥ 10 rows: DONE
    C. Try: category[composite_key + market_regime]  → if ≥ 10 rows: DONE
    D. Try: category[composite_key]                  → if ≥ 10 rows: DONE
    E. Try: category[tier1 + tier2]                  → if ≥ 10 rows: DONE
    F. Fallback: category[tier1]                     → always has ≥ 10 rows
```

This gives the information value of Tiers 4/5 when the data supports it, without
locking the system into unusable sparse categories when it doesn't.

**For rare patterns specifically:** A `MORNING_STAR` pattern (Tier 3) is rare by
definition — maybe 500 examples in 175k rows. Subdividing it further by Tier 4
(volume regime) splits those 500 rows into ~125 per subcategory. A `MORNING_STAR`
in high-volume bull market would have perhaps 60 rows — still above MIN_MATCHES.
But a `MORNING_STAR + low volume + bear market` might have 8 rows and trigger
cascade fallback. This is correct behavior, not a failure.

## 6.1 Expected Impact

| Metric | Before | After (Tier 1-3 + Floating Modifiers) |
|--------|--------|--------------------------------------|
| Search space per query | 175k rows | ~8k-30k rows (Tier 3 match) |
| K-NN speed improvement | 1× | 6-20× |
| Scalable ticker limit | ~100 tickers | ~600-2,000 tickers |
| Signal quality | Baseline | Improved (structurally similar analogues) |
| Calibration | 1 global Platt | 3 per-trend Platt scalers |
| Rare pattern handling | Same as common | Graceful cascade fallback, no errors |

---

## 7. Risk and Caveats

**Category sparsity**: Rare patterns (e.g., `MORNING_STAR`) may have too few training
analogues. The fallback to Tier 1 handles this, but monitor minimum match counts.

**Lookback dependency**: Tier 3 patterns require 3-5 candles of history, so they
cannot be computed on the first 4 days of a ticker's data in the database.

**Bull-market bias of pattern library**: Classical candlestick patterns were developed
for commodity and forex markets. Their predictive power in large-cap equities is debated.
We treat them as *structural similarity filters*, not predictive signals on their own.
The K-NN analogues do the prediction — patterns only narrow the search space.

**Do not use patterns as signals**: Pattern → BUY/SELL directly would be a separate,
unvalidated system. The categorization module is purely an efficiency layer for K-NN.

---

## 8. Build Order

1. `pattern_classifier.py` — classification logic (no FPPE integration yet)
2. Unit tests on known candles (verify HAMMER classifier catches hammers, etc.)
3. `pre_tag_database.py` — batch classification of train_db
4. `build_category_index.py` — index construction and serialization
5. Integrate into `signal_adapter.py` / `strategy.py`
6. Benchmark: compare K-NN runtime before/after on a sample of 100 queries
7. Validate: confirm signal quality (win rate, net expectancy) is maintained or improved

---

## 9. Implementation Status

**Current state: Design document only. No code has been written for this module.**

What exists:
- This design document (v0.2) — complete architecture specification
- `FPPE_TRADING_SYSTEM_DESIGN.md` — references this module in the scaling section

What does NOT yet exist:
- `pattern_classifier.py` — the actual classification logic
- `pre_tag_database.py` — batch tagger for train_db
- `build_category_index.py` — index builder
- Any modifications to `signal_adapter.py` or FPPE's `strategy.py`

**This module is scheduled for implementation in Phase 6 (after the 10-year data expansion),
when the ticker universe is large enough that K-NN speed is a real bottleneck.**

At 52 tickers, K-NN completes in 128 seconds per year of signals — acceptable.
At 500 tickers (the expansion target), K-NN would take ~20+ minutes per year without
pre-filtering. That is when this module becomes necessary, not before.

---

*Design doc v0.2 — architecture updated with Tier 4/5 analysis and hierarchical fallback.*
