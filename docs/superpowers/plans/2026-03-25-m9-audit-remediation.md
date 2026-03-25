# M9 Audit Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address two high-priority findings from the M9 code audit and scaling bottleneck reports, then document two deferred items for future phases.

**Architecture:**
- **Task 1** commits pending session changes (already-written `prepare.py` vol-norm feature code that has no unit tests), adds `clip(-10, 10)` as belt-and-suspenders protection against extreme but finite ratios when rolling vol approaches zero, and writes the missing unit test coverage.
- **Task 2** adds a circuit breaker to `SentimentVetoFilter.fetch_sentiment()`: if FMP API failure rate exceeds a configurable threshold during a live run, execution halts rather than silently defaulting all sentiments to neutral. Requires extracting `_fetch_ticker()` for clean testability.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, unittest.mock

---

## Audit Findings Summary

### M9 Phase (Implemented in this plan)

| # | Finding | Source | Severity | Action |
|---|---------|--------|---------|--------|
| 1 | `ret_Xd / rolling_std` can produce extreme finite ratios when rolling_std is near-zero | Bottleneck Report §2 | High | Add `clip(-10, 10)` after vol-norm computation + unit tests for the pending `compute_vol_normalized_features` function |
| 2 | Silent failure in `fetch_sentiment()` when FMP MCP goes down — all tickers default to 0.0 neutral, BUY signals execute unprotected | Audit Report §3 | High | Add circuit breaker with configurable threshold; extract `_fetch_ticker()` for testability |

### Deferred to Future Phases

| # | Finding | Source | Phase | Action |
|---|---------|--------|-------|--------|
| 3 | No upside compounding — `StrategyEvaluator` manages downside only, no Half-Kelly position sizing | Bottleneck Report §1 | M10 | New `trading_system/position_sizer.py` module: ingests BSS + Rolling Sharpe → Target Exposure Fraction |
| 4 | Single-threaded HNSW index rebuild is a bottleneck at 1,500+ tickers | Bottleneck Report §3 | M11 | Persisted index model: `save_index()` overnight + `load_index()` in live runner |

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `prepare.py` | MODIFY | Add `clip(-10, 10)` to `compute_vol_normalized_features()` |
| `tests/unit/test_vol_norm.py` | CREATE | Unit tests for `compute_vol_normalized_features()` |
| `pattern_engine/sentiment_veto.py` | MODIFY | Extract `_fetch_ticker()`, add `circuit_breaker_threshold`, raise `RuntimeError` on overload |
| `tests/unit/test_sentiment_veto.py` | MODIFY | Add 2 tests: circuit breaker triggers, circuit breaker below threshold passes |
| `pattern_engine/features.py` | COMMIT | Pending: `VOL_NORM_COLS` constant (already written) |
| `pattern_engine/sector.py` | COMMIT | Pending: 800→585 ticker trimming (already written) |
| `scripts/build_sector_map.py` | COMMIT | Pending: two-pass 2010 depth gate (already written) |
| `pattern_engine/contracts/matchers/hnsw_matcher.py` | COMMIT | Pending: ef comment fix / k*2 restore (already written) |

---

## Task 1: Test + commit pending vol-norm session changes with clip guard

**Context:** `prepare.py` has uncommitted changes from the prior session that add
`compute_vol_normalized_features()`. The function already uses `1e-8` epsilon (prevents
exact division-by-zero) and the calling code already has `dropna()` (removes NaN rows).
The remaining gap: when `rolling_std` is very small but not zero (e.g., 1e-5 for a
thinly-traded period), the ratio can reach 10,000+. These extreme-but-finite values
would distort K-NN distances for any ticker that briefly traded in a low-volatility
window. Adding `clip(-10, 10)` caps the Sharpe-like ratio at ±10σ, which is:
- Wide enough to preserve meaningful signal (±10σ is a genuine outlier in finance)
- Tight enough to prevent a single illiquid tick dominating all K-NN distances

No tests exist for `compute_vol_normalized_features()`. Write them before committing.

**Files:**
- Create: `tests/unit/test_vol_norm.py`
- Modify: `prepare.py` — add `.clip(-10, 10)` to vol-norm computation (line ~206)
- Commit: `prepare.py`, `pattern_engine/features.py`, `pattern_engine/sector.py`,
  `scripts/build_sector_map.py`, `pattern_engine/contracts/matchers/hnsw_matcher.py`

- [ ] **Step 1: Verify current test suite passes before touching anything**

```bash
cd C:/Users/Isaia/.claude/financial-research
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: `562 passed, 1 skipped`

- [ ] **Step 2: Confirm the function to be tested exists in pending prepare.py**

```bash
PYTHONUTF8=1 py -3.12 -c "
import sys; sys.path.insert(0, '.')
# prepare.py has uncommitted changes — import directly
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('prepare', 'prepare.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(hasattr(mod, 'compute_vol_normalized_features'))
print(hasattr(mod, 'VOL_NORM_COLS'))
" 2>&1
```

Expected: `True` / `True`

- [ ] **Step 3: Write the test file**

Create `tests/unit/test_vol_norm.py`:

```python
"""Tests for prepare.compute_vol_normalized_features().

Covers: normal computation, epsilon guard against zero-vol, clip guard against
extreme ratios, and NaN propagation for rows with insufficient history.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import prepare once at module level — it has a module-level side effect
# (from pattern_engine.sector import TICKERS) that is slow to re-execute.
import prepare  # noqa: E402


def _make_df(n=30, daily_return=0.01, seed=42):
    """Minimal DataFrame with Close and ret_Xd columns."""
    rng = np.random.RandomState(seed)
    close = 100.0 * np.cumprod(1 + rng.randn(n) * daily_return)
    df = pd.DataFrame({"Close": close})
    for w in prepare.RETURN_WINDOWS:
        df[f"ret_{w}d"] = df["Close"].pct_change(w)
    return df


def test_vol_norm_columns_created():
    """compute_vol_normalized_features() produces all 8 ret_Xd_norm columns."""
    df = _make_df(n=60)
    out = prepare.compute_vol_normalized_features(df)
    for w in prepare.RETURN_WINDOWS:
        assert f"ret_{w}d_norm" in out.columns, f"Missing ret_{w}d_norm"


def test_vol_norm_values_finite_for_normal_ticker():
    """All non-NaN vol-norm values are finite after clip guard."""
    df = _make_df(n=60, daily_return=0.01)
    out = prepare.compute_vol_normalized_features(df)
    for w in prepare.RETURN_WINDOWS:
        col = out[f"ret_{w}d_norm"].dropna()
        assert np.all(np.isfinite(col.values)), f"ret_{w}d_norm has non-finite values"


def test_vol_norm_clipped_at_plus_minus_10():
    """Vol-norm values are clipped to [-10, 10] — extreme ratios cannot reach HNSW."""
    df = _make_df(n=60, daily_return=0.01)
    # Inject an abnormally large return to force extreme ratio.
    # Note: compute_vol_normalized_features() reads ret_Xd columns directly;
    # setting ret_1d = 100.0 while rolling_vol stays small forces the clip path.
    df["ret_1d"] = df["ret_1d"].fillna(0.0)
    df.loc[df.index[20], "ret_1d"] = 100.0  # +10,000% outlier
    out = prepare.compute_vol_normalized_features(df)
    col = out["ret_1d_norm"].dropna()
    assert col.max() <= 10.0, f"ret_1d_norm exceeded clip upper bound: {col.max()}"
    assert col.min() >= -10.0, f"ret_1d_norm exceeded clip lower bound: {col.min()}"


def test_epsilon_prevents_division_by_zero():
    """When rolling_std = 0.0 (constant Close prices), result is finite (not inf).

    The epsilon guard prevents division by zero. The function derives rolling_vol
    from df["Close"].pct_change() — setting Close to constant makes rolling_vol = 0.0.
    The pre-computed ret_Xd columns are not used for vol estimation; only Close matters.
    """
    df = _make_df(n=30)
    # Constant Close → daily_returns = 0.0 everywhere → rolling_std = 0.0
    df["Close"] = 100.0
    out = prepare.compute_vol_normalized_features(df)
    for w in prepare.RETURN_WINDOWS:
        col = out[f"ret_{w}d_norm"].dropna()
        assert not col.isin([np.inf, -np.inf]).any(), (
            f"ret_{w}d_norm has inf despite epsilon guard"
        )


def test_early_rows_produce_nan_before_dropna():
    """First rows (< min_periods rolling window) produce NaN vol-norm.

    These NaN rows are caught by build_analogue_database()'s dropna() call
    and never reach PatternMatcher or the HNSW guard.
    """
    df = _make_df(n=15)  # very short history — 90d window cannot fill
    out = prepare.compute_vol_normalized_features(df)
    col = out["ret_90d_norm"]
    assert col.isna().any(), (
        "Expected NaN in ret_90d_norm for short history — these rows "
        "must be caught by prepare.py's dropna() before reaching HNSW"
    )
```

- [ ] **Step 4: Run the tests — verify they fail (function exists but clip is not yet added)**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_vol_norm.py -v 2>&1
```

Expected: `test_vol_norm_clipped_at_plus_minus_10` FAILS (clip not yet applied).
Others may pass if prepare.py already has partial implementation.

- [ ] **Step 5: Add `clip(-10, 10)` to `compute_vol_normalized_features` in `prepare.py`**

Find the loop body in `compute_vol_normalized_features()` (currently ~line 206):

```python
        df[f"ret_{w}d_norm"] = df[f"ret_{w}d"] / (rolling_vol + 1e-8)
```

Replace with:

```python
        raw_norm = df[f"ret_{w}d"] / (rolling_vol + 1e-8)
        df[f"ret_{w}d_norm"] = raw_norm.clip(-10.0, 10.0)
```

- [ ] **Step 6: Run the new tests — verify all 5 pass**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_vol_norm.py -v 2>&1
```

Expected: `5 passed`

- [ ] **Step 7: Run full test suite**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: `567 passed, 1 skipped` (562 + 5 new)

- [ ] **Step 8a: Commit vol-norm changes and tests**

```bash
git add prepare.py pattern_engine/features.py tests/unit/test_vol_norm.py
git commit -m "feat(vol-norm): add compute_vol_normalized_features with clip(-10,10) guard and unit tests"
```

- [ ] **Step 8b: Commit universe / sector changes**

```bash
git add pattern_engine/sector.py scripts/build_sector_map.py
git commit -m "chore(sector): trim universe to 585T, add 2010 historical depth gate to build_sector_map.py"
```

- [ ] **Step 8c: Commit the HNSW comment fix**

```bash
git add pattern_engine/contracts/matchers/hnsw_matcher.py
git commit -m "fix(hnsw): restore ef=k*2 baseline, document M9 ef=k*4 experiment result"
```

---

## Task 2: Circuit breaker in SentimentVetoFilter

**Context:** `fetch_sentiment()` currently catches per-ticker errors and defaults to
`0.0` (neutral). If FMP MCP goes down, all 585 tickers silently return neutral and
the sentiment layer is effectively disabled — but no exception is raised, so the live
runner happily executes BUY signals without any sentiment protection.

The fix: add a `circuit_breaker_threshold` (default 0.30) and raise `RuntimeError`
after the loop if `error_count / len(tickers) > threshold`.

For testability, extract the inner FMP call into `_fetch_ticker()` so tests can
mock it to raise errors without patching module internals. The stub implementation
in `_fetch_ticker()` returns `0.0` (unchanged behavior for backtesting).

**Files:**
- Modify: `pattern_engine/sentiment_veto.py`
  - `__init__`: add `circuit_breaker_threshold: float = 0.30`
  - Extract: `_fetch_ticker(self, ticker, since_date) -> float`
  - Modify: `fetch_sentiment()` — call `_fetch_ticker()`, count errors, raise
- Modify: `tests/unit/test_sentiment_veto.py` — add 2 new tests

- [ ] **Step 1: Read the current `fetch_sentiment` implementation to get exact context**

Open `pattern_engine/sentiment_veto.py` lines 62–103.

Confirm the loop structure:
```python
for ticker in tickers:
    try:
        scores[ticker] = 0.0  # stub
    except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
        logging.warning(...)
        scores[ticker] = 0.0
return scores
```

- [ ] **Step 2: Write the three new failing tests**

Append to `tests/unit/test_sentiment_veto.py`:

```python
def test_fetch_sentiment_stub_returns_neutral():
    """fetch_sentiment() delegates to _fetch_ticker() and returns 0.0 per ticker (stub)."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    from datetime import date
    filt = SentimentVetoFilter()
    scores = filt.fetch_sentiment(["AAPL", "MSFT"], query_date=date(2024, 1, 5))
    assert scores == {"AAPL": 0.0, "MSFT": 0.0}


def test_circuit_breaker_raises_when_error_rate_exceeds_threshold():
    """RuntimeError is raised when FMP failure rate exceeds circuit_breaker_threshold."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    from datetime import date
    from unittest.mock import patch

    filt = SentimentVetoFilter(circuit_breaker_threshold=0.30)

    # Make every ticker fetch fail
    def always_fail(ticker, since_date):
        raise ConnectionError("FMP MCP unreachable")

    with patch.object(filt, "_fetch_ticker", side_effect=always_fail):
        with pytest.raises(RuntimeError, match="Halting execution"):
            filt.fetch_sentiment(
                ["AAPL", "MSFT", "GOOG", "META"],
                query_date=date(2024, 1, 5),
            )


def test_circuit_breaker_passes_when_error_rate_below_threshold():
    """No exception raised when error rate is below threshold."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    from datetime import date
    from unittest.mock import patch

    filt = SentimentVetoFilter(circuit_breaker_threshold=0.30)

    call_count = {"n": 0}

    def fail_first_only(ticker, since_date):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ConnectionError("one transient error")
        return 0.0  # all others succeed

    # 1 failure in 4 tickers = 25% < 30% threshold — should NOT raise
    with patch.object(filt, "_fetch_ticker", side_effect=fail_first_only):
        scores = filt.fetch_sentiment(
            ["AAPL", "MSFT", "GOOG", "META"],
            query_date=date(2024, 1, 5),
        )
    # Failed ticker defaults to 0.0 (neutral)
    assert scores["AAPL"] == 0.0
    assert scores["MSFT"] == 0.0
```

- [ ] **Step 3: Run the new tests — verify they fail**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sentiment_veto.py -v 2>&1 | tail -15
```

Expected: all three new tests FAIL (`_fetch_ticker` not yet extracted, `circuit_breaker_threshold` not yet added).

- [ ] **Step 4: Implement the changes in `sentiment_veto.py`**

**4a: Add `circuit_breaker_threshold` to `__init__`**

Current:
```python
    def __init__(
        self,
        veto_threshold: float = -0.20,
        lookback_days: int = 3,
    ):
        self.veto_threshold = veto_threshold
        self.lookback_days = lookback_days
```

Replace with:
```python
    def __init__(
        self,
        veto_threshold: float = -0.20,
        lookback_days: int = 3,
        circuit_breaker_threshold: float = 0.30,
    ):
        self.veto_threshold = veto_threshold
        self.lookback_days = lookback_days
        self.circuit_breaker_threshold = circuit_breaker_threshold
```

**4b: Extract `_fetch_ticker()` method**

Add this method directly before `fetch_sentiment()`:

```python
    def _fetch_ticker(self, ticker: str, since_date) -> float:
        """Fetch sentiment score for a single ticker.

        Stub implementation returns 0.0 (neutral) until FMP MCP is wired.
        In live usage, replace with:
            news = mcp__fmp__stock_news(symbol=ticker, limit=20)
            recent = [n for n in news if n["date"] >= str(since_date)]
            return mean([n["sentimentScore"] for n in recent]) if recent else 0.0

        Raises:
            ConnectionError, TimeoutError, OSError, ValueError: on fetch failure.
        """
        return 0.0  # neutral default until FMP wired
```

**4c: Rewrite `fetch_sentiment()` to use `_fetch_ticker()` and circuit breaker**

Replace the current `fetch_sentiment()` body:

```python
    def fetch_sentiment(
        self,
        tickers: list[str],
        query_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """Fetch news sentiment scores for a list of tickers.

        Calls _fetch_ticker() for each ticker and aggregates results.
        If the failure rate exceeds circuit_breaker_threshold, raises
        RuntimeError rather than silently returning all-neutral scores.

        Args:
            tickers:    List of ticker symbols (uppercase).
            query_date: Date to fetch sentiment for (defaults to today).

        Returns:
            Dict[str, float]: ticker -> mean sentiment score [-1.0, +1.0].
            Tickers with no news or transient errors return 0.0 (neutral),
            provided the overall error rate stays below circuit_breaker_threshold.

        Raises:
            RuntimeError: If error_count / len(tickers) > circuit_breaker_threshold.
        """
        if query_date is None:
            query_date = date.today()

        since_date = query_date - timedelta(days=self.lookback_days)
        scores: Dict[str, float] = {}
        error_count = 0

        for ticker in tickers:
            try:
                scores[ticker] = self._fetch_ticker(ticker, since_date)
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
                logging.warning("SentimentVetoFilter: failed to fetch %s: %s", ticker, exc)
                scores[ticker] = 0.0  # neutral on individual error
                error_count += 1

        if tickers and error_count / len(tickers) > self.circuit_breaker_threshold:
            raise RuntimeError(
                f"SentimentVetoFilter circuit breaker: {error_count}/{len(tickers)} "
                f"fetch failures ({error_count / len(tickers):.0%}) exceed threshold "
                f"{self.circuit_breaker_threshold:.0%}. Halting execution pipeline."
            )

        return scores
```

- [ ] **Step 5: Run the new tests — verify all 8 pass**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sentiment_veto.py -v 2>&1
```

Expected: `8 passed` (5 existing + 3 new)

- [ ] **Step 6: Run full test suite**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: `570 passed, 1 skipped` (562 baseline + 5 vol-norm + 3 sentiment)

- [ ] **Step 7: Commit**

```bash
git add pattern_engine/sentiment_veto.py tests/unit/test_sentiment_veto.py
git commit -m "feat(sentiment-veto): add circuit breaker and extract _fetch_ticker for testability"
```

---

## Future Phase Documentation

### M10: Half-Kelly Position Sizer Module

**Trigger:** When live capital deployment begins (current BSS must be positive on ≥3/6 folds first).

**Problem:** `StrategyEvaluator` (trading_system/strategy_evaluator.py) manages downside gating (HALT, REDUCE_EXPOSURE) but has no mechanism to compound profits back into larger positions as capital grows.

**Design:**
- New file: `trading_system/position_sizer.py`
- Class: `KellyPositionSizer`
- Inputs: `bss: float`, `rolling_sharpe: float`, `current_equity: float`, `max_position_fraction: float = 0.25`
- Output: `target_position_size: float` (dollar amount, capped at `max_position_fraction * equity`)
- Formula: Half-Kelly = `(E[R] / Variance) * 0.5`, where E[R] is approximated from recent BSS and Sharpe
- Integration point: `StrategyEvaluator.evaluate_signal()` calls `KellyPositionSizer.size()` after a BUY decision

**Prerequisite:** Positive BSS on ≥3/6 walk-forward folds (currently 0/6 positive). Do not implement until the signal has demonstrated edge — Kelly on a zero-edge signal compounds losses.

---

### M11: HNSW Disk Persistence

**Trigger:** Universe expansion to 1,500+ tickers (currently 585T, index build = ~469s/fold).

**Problem:** At 1,500+ tickers over 10 years (~8M rows), single-threaded HNSW index construction during the live 4:00 PM run becomes a blocking bottleneck.

**Design:**
- Add to `research/hnsw_distance.py`:
  - `HNSWIndex.save(path: str) -> None` — wraps `self._index.save_index(path)`
  - `HNSWIndex.load(path: str, n_features: int) -> HNSWIndex` — classmethod, wraps `load_index(path)`
- New script: `scripts/build_hnsw_index.py` — overnight cron job that rebuilds and persists the index to `data/hnsw_index.bin`
- Modify live runner: call `HNSWIndex.load()` instead of `fit()` when index file is fresh (< 24h old)

**Note on joblib constraint (CLAUDE.md):** `nn_jobs=1` (hardcoded) prevents Windows/Py3.12 deadlock in PatternMatcher. HNSW disk persistence is a separate optimization that does not involve joblib parallelism — it is safe to implement independently.

---

## Testing Matrix

| Module | Test File | New Tests |
|--------|-----------|-----------|
| `prepare.compute_vol_normalized_features` | `tests/unit/test_vol_norm.py` | 5 (columns, finite, clip, epsilon, nan-propagation) |
| `SentimentVetoFilter` circuit breaker + delegation | `tests/unit/test_sentiment_veto.py` | 3 (stub-delegation, triggers, below-threshold-passes) |

**Total new tests: 8.** All existing 562 tests must continue to pass. Expected final count: 570 passed, 1 skipped.
