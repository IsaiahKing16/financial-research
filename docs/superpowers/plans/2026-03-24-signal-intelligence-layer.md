# Signal Intelligence Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four signal intelligence layers to FPPE — a Decision Journal (top-5/10/25 analogues), Sector Conviction Score, Momentum Agreement Filter, and Sentiment Veto — without changing the existing `query()` return signature or any locked settings.

**Architecture:** All four features are flag-gated and non-intrusive: zero cost when disabled, no changes to the 6-tuple `query()` return, no locked-setting modifications. The Decision Journal adds a 5th training cache (`_train_dates_arr`) and a side-effect attribute (`matcher.last_journal`) populated after each `query()` call when `journal_top_n > 0`. The three signal filters (Sector Conviction, Momentum Agreement, Sentiment Veto) are post-processing steps that operate on the `query()` output tuple and all inherit from `SignalFilterBase` for a unified polymorphic `apply()` interface. `SignalPipeline` composes the active filters into a single dispatch point in `run_walkforward.py`, satisfying Single Responsibility: the script constructs the pipeline and calls `pipeline.run()`, with no per-filter branching logic.

**Tech Stack:** Python 3.12, pandas, numpy, dataclasses, FMP MCP (sentiment veto only), pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `pattern_engine/signal_filter_base.py` | CREATE | `SignalFilterBase` ABC — unified `apply(probs, signals, val_db, **kwargs)` interface |
| `pattern_engine/signal_pipeline.py` | CREATE | `SignalPipeline` — ordered filter chain runner; single dispatch point in `run_walkforward.py` |
| `pattern_engine/journal.py` | CREATE | `JournalEntry` dataclass, `_normalize_date()` helper, Parquet writer, `build_journal_entries()` |
| `pattern_engine/sector_conviction.py` | CREATE | `SectorConvictionLayer(SignalFilterBase)` — per-sector base rate vs K-NN aggregate score |
| `pattern_engine/momentum_signal.py` | CREATE | `MomentumSignalFilter(SignalFilterBase)` — ticker vs sector rolling return comparison |
| `pattern_engine/sentiment_veto.py` | CREATE | `SentimentVetoFilter(SignalFilterBase)` — FMP news sentiment BUY veto |
| `pattern_engine/matcher.py` | MODIFY | Add `_train_dates_arr` cache in `_rebuild_caches()` + journal accumulation in `query()` loop |
| `scripts/run_walkforward.py` | MODIFY | `WalkForwardConfig` flags + per-fold journal write + `SignalPipeline` dispatch |
| `scripts/query_journal.py` | CREATE | CLI inspection tool — filter by ticker/date/signal, show top-N analogues |
| `tests/unit/test_signal_filter_base.py` | CREATE | Unit tests for `SignalFilterBase` ABC and `SignalPipeline` |
| `tests/unit/test_journal.py` | CREATE | Unit tests for journal entry building, file output |
| `tests/unit/test_sector_conviction.py` | CREATE | Unit tests for sector conviction layer |
| `tests/unit/test_momentum_signal.py` | CREATE | Unit tests for momentum filter |
| `tests/unit/test_sentiment_veto.py` | CREATE | Unit tests for sentiment veto (mock FMP calls) |
| `results/journals/` | CREATE DIR | Output directory for journal Parquet files |

---

## Task 1: Add `_train_dates_arr` cache to PatternMatcher

**Context:** `_rebuild_caches()` at `matcher.py:559` currently caches tickers, sectors, targets, returns — but not dates. The journal needs dates to show WHEN each analogue occurred historically.

**Files:**
- Modify: `pattern_engine/matcher.py:559-588`

- [ ] **Step 1: Read the `_rebuild_caches` method to confirm current state**

Open `pattern_engine/matcher.py`, find `_rebuild_caches` at line 559. Confirm it sets `_train_tickers_arr`, `_train_sector_arr`, `_train_target_arr`, `_train_ret_arr`.

- [ ] **Step 2: Add `_train_dates_arr` to `__init__`**

In `PatternMatcher.__init__` (line ~120), after the four existing cache declarations, add:

```python
self._train_dates_arr: Optional[np.ndarray] = None
```

- [ ] **Step 3: Populate `_train_dates_arr` in `_rebuild_caches()`**

After the existing four cache assignments (line ~587), add:

```python
self._train_dates_arr = (
    train_db["Date"].values
    if "Date" in train_db.columns
    else np.array([None] * len(train_db))
)
```

- [ ] **Step 4: Verify no tests broken**

```bash
cd C:/Users/Isaia/.claude/financial-research
source venv/Scripts/activate
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: `543 passed` (or same count as before this change).

- [ ] **Step 5: Commit**

```bash
git add pattern_engine/matcher.py
git commit -m "feat(journal): cache _train_dates_arr in PatternMatcher._rebuild_caches"
```

---

## Task 2: Create `pattern_engine/journal.py`

**Context:** The journal captures, for each query row that produced a BUY or SELL signal, the top-N closest historical analogues so a human can see exactly which past patterns drove the prediction.

**Files:**
- Create: `pattern_engine/journal.py`
- Create: `results/journals/` (directory)

- [ ] **Step 1: Create the directory**

```bash
mkdir -p "C:/Users/Isaia/.claude/financial-research/results/journals"
```

- [ ] **Step 2: Write failing tests first**

Create `tests/unit/test_journal.py`:

```python
"""Tests for pattern_engine.journal — JournalEntry and build helpers."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import date


def test_journal_entry_fields():
    """JournalEntry dataclass has required fields."""
    from pattern_engine.journal import JournalEntry, AnalogueRecord
    rec = AnalogueRecord(
        rank=1,
        ticker="AAPL",
        date=date(2020, 3, 15),
        distance=0.42,
        label=1,
        fwd_return=0.035,
    )
    entry = JournalEntry(
        query_date=date(2024, 1, 5),
        query_ticker="MSFT",
        raw_prob=0.62,
        calibrated_prob=0.67,
        signal="BUY",
        n_matches=42,
        top_analogues=[rec],
    )
    assert entry.query_ticker == "MSFT"
    assert entry.signal == "BUY"
    assert len(entry.top_analogues) == 1
    assert entry.top_analogues[0].rank == 1


def test_build_journal_entries_basic():
    """build_journal_entries returns one entry per BUY/SELL row."""
    from pattern_engine.journal import build_journal_entries
    train_dates = np.array([date(2020, 1, i+1) for i in range(10)])
    train_tickers = np.array(["AAPL"] * 10, dtype=object)
    train_targets = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 1], dtype=np.float64)
    train_returns = np.array([0.01, 0.03, -0.02, 0.01, 0.04,
                               -0.01, 0.02, 0.01, -0.03, 0.05])

    # Two query rows: one BUY, one HOLD
    top_masks = np.array([
        [True, True, True, False, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False, False, False],
    ])
    distances = np.zeros((2, 10))
    distances[0] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    distances[1] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    indices = np.tile(np.arange(10), (2, 1))

    val_tickers = np.array(["MSFT", "GOOG"], dtype=object)
    val_dates = np.array([date(2024, 1, 5), date(2024, 1, 6)])
    raw_probs = np.array([0.62, 0.51])
    cal_probs = np.array([0.67, 0.53])
    signals = ["BUY", "HOLD"]
    n_matches = [3, 1]
    top_n = 5

    entries = build_journal_entries(
        top_masks=top_masks,
        distances=distances,
        indices=indices,
        val_tickers=val_tickers,
        val_dates=val_dates,
        raw_probs=raw_probs,
        cal_probs=cal_probs,
        signals=signals,
        n_matches=n_matches,
        train_tickers=train_tickers,
        train_dates=train_dates,
        train_targets=train_targets,
        train_returns=train_returns,
        top_n=top_n,
    )
    # Only BUY/SELL rows are journalled (HOLD skipped)
    assert len(entries) == 1
    assert entries[0].signal == "BUY"
    assert entries[0].query_ticker == "MSFT"
    assert len(entries[0].top_analogues) == 3  # only 3 survivors
    assert entries[0].top_analogues[0].rank == 1
    assert entries[0].top_analogues[0].ticker == "AAPL"


def test_build_journal_entries_top_n_cap():
    """top_n caps the number of analogues stored per entry."""
    from pattern_engine.journal import build_journal_entries
    n = 50
    train_dates = np.array([date(2019, 1, 1)] * n)
    train_tickers = np.array(["SPY"] * n, dtype=object)
    train_targets = np.ones(n)
    train_returns = np.ones(n) * 0.01
    top_masks = np.ones((1, n), dtype=bool)
    distances = np.arange(n, dtype=float).reshape(1, n)
    indices = np.arange(n).reshape(1, n)

    entries = build_journal_entries(
        top_masks=top_masks,
        distances=distances,
        indices=indices,
        val_tickers=np.array(["AAPL"], dtype=object),
        val_dates=np.array([date(2024, 6, 1)]),
        raw_probs=np.array([0.70]),
        cal_probs=np.array([0.71]),
        signals=["BUY"],
        n_matches=[50],
        train_tickers=train_tickers,
        train_dates=train_dates,
        train_targets=train_targets,
        train_returns=train_returns,
        top_n=10,
    )
    assert len(entries) == 1
    assert len(entries[0].top_analogues) == 10


def test_write_and_read_journal_parquet(tmp_path):
    """Journal entries can be written to Parquet and read back."""
    from pattern_engine.journal import JournalEntry, AnalogueRecord, write_journal_parquet, read_journal_parquet
    rec = AnalogueRecord(rank=1, ticker="AAPL", date=date(2020, 3, 15),
                          distance=0.42, label=1, fwd_return=0.035)
    entry = JournalEntry(
        query_date=date(2024, 1, 5),
        query_ticker="MSFT",
        raw_prob=0.62,
        calibrated_prob=0.67,
        signal="BUY",
        n_matches=42,
        top_analogues=[rec],
    )
    out_path = tmp_path / "test_journal.parquet"
    write_journal_parquet([entry], out_path)
    assert out_path.exists()

    df = read_journal_parquet(out_path)
    assert len(df) == 1
    assert df["query_ticker"].iloc[0] == "MSFT"
    assert df["signal"].iloc[0] == "BUY"
    assert df["analogue_rank"].iloc[0] == 1
    assert df["analogue_ticker"].iloc[0] == "AAPL"

    # top_n_view accepts any positive int (not restricted to 5/10/25)
    from pattern_engine.journal import top_n_view
    view5 = top_n_view(df, n=5)
    assert len(view5) == 1   # only 1 analogue in this entry
    view3 = top_n_view(df, n=3)
    assert len(view3) == 1   # still 1 (rank=1 <= 3)
    with pytest.raises(ValueError):
        top_n_view(df, n=0)   # 0 is invalid
```

- [ ] **Step 3: Run tests — verify they fail (journal.py doesn't exist yet)**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_journal.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'pattern_engine.journal'`

- [ ] **Step 4: Implement `pattern_engine/journal.py`**

```python
"""
journal.py — Decision Journal for PatternMatcher signal inspection.

Records the top-N historical analogues for each BUY/SELL signal, enabling
human-readable audit trails: which past patterns drove a prediction, when
they occurred, how close they were, and what actually happened.

Usage:
    # After matcher.query(), if journal_top_n > 0 in config:
    journal_entries = matcher.last_journal  # list[JournalEntry]
    write_journal_parquet(journal_entries, "results/journals/fold6.parquet")

    # Inspection:
    df = read_journal_parquet("results/journals/fold6.parquet")
    buy_signals = df[df["signal"] == "BUY"]
    top5 = buy_signals[buy_signals["analogue_rank"] <= 5]

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AnalogueRecord:
    """One historical analogue that contributed to a signal.

    Attributes:
        rank:       1-based rank by distance (1 = closest).
        ticker:     Training ticker symbol.
        date:       Date in training data when this pattern occurred.
        distance:   Euclidean distance from query point (post-scale).
        label:      Binary forward return label (0 = down, 1 = up).
        fwd_return: Actual forward return value (e.g., fwd_14d value).
    """
    rank: int
    ticker: str
    date: date
    distance: float
    label: int
    fwd_return: float


@dataclass
class JournalEntry:
    """Complete decision record for one BUY or SELL signal.

    Attributes:
        query_date:      Date of the query row.
        query_ticker:    Ticker being evaluated.
        raw_prob:        Uncalibrated K-NN vote fraction (0-1).
        calibrated_prob: Platt-calibrated probability (0-1).
        signal:          "BUY" or "SELL" (HOLD rows are not journalled).
        n_matches:       Total accepted analogues (may exceed top_n).
        top_analogues:   Up to top_n closest analogues, sorted by distance.
    """
    query_date: date
    query_ticker: str
    raw_prob: float
    calibrated_prob: float
    signal: str
    n_matches: int
    top_analogues: list[AnalogueRecord] = field(default_factory=list)


def _normalize_date(val) -> date:
    """Normalize any date-like value to datetime.date.

    Handles pd.Timestamp, datetime.datetime, np.datetime64, datetime.date, and None.
    Falls back to date(1900, 1, 1) for None or unrecognised types.
    """
    if hasattr(val, "date"):            # pd.Timestamp or datetime.datetime
        return val.date()
    if isinstance(val, np.datetime64):
        return pd.Timestamp(val).date()
    if val is None:
        return date(1900, 1, 1)
    return val                          # already a datetime.date


def build_journal_entries(
    *,
    top_masks: np.ndarray,        # (B, n_probe) bool — survived filter
    distances: np.ndarray,        # (B, n_probe) float — Euclidean distances
    indices: np.ndarray,          # (B, n_probe) int — training row indices
    val_tickers: np.ndarray,      # (B,) object — query tickers
    val_dates: np.ndarray,        # (B,) — query dates
    raw_probs: np.ndarray,        # (B,) float — raw K-NN probabilities
    cal_probs: np.ndarray,        # (B,) float — calibrated probabilities
    signals: list[str],           # length B
    n_matches: list[int],         # length B
    train_tickers: np.ndarray,    # (N_train,) object
    train_dates: np.ndarray,      # (N_train,) date
    train_targets: np.ndarray,    # (N_train,) float — binary labels
    train_returns: np.ndarray,    # (N_train,) float — forward returns
    top_n: int = 25,              # max analogues per entry (5, 10, or 25)
) -> list[JournalEntry]:
    """Build journal entries for all BUY/SELL rows in a batch.

    HOLD rows are skipped — they have no actionable signal to explain.
    Only the top_n closest accepted analogues are stored per entry.

    Args:
        top_masks:   Boolean survival mask from PatternMatcher._post_filter().
        distances:   Euclidean distances from PatternMatcher._query_batch().
        indices:     Training row indices from PatternMatcher._query_batch().
        val_tickers: Query ticker array for this batch.
        val_dates:   Query date array for this batch.
        raw_probs:   Pre-calibration K-NN frequencies for this batch.
        cal_probs:   Post-calibration probabilities for this batch.
        signals:     Signal strings for this batch.
        n_matches:   Accepted analogue counts for this batch.
        train_*:     Training data arrays (from PatternMatcher caches).
        top_n:       Maximum analogues to store per entry (5, 10, or 25).

    Returns:
        List of JournalEntry objects (one per BUY/SELL row).
    """
    entries: list[JournalEntry] = []
    B = top_masks.shape[0]

    for i in range(B):
        sig = signals[i]
        if sig not in ("BUY", "SELL"):
            continue  # HOLD rows not journalled

        # Get accepted analogue positions in sorted distance order
        accepted_pos = np.where(top_masks[i])[0]   # indices into n_probe axis
        if len(accepted_pos) == 0:
            continue

        # Sort by distance (already sorted by HNSW/BallTree, but be explicit)
        sorted_pos = accepted_pos[np.argsort(distances[i][accepted_pos])]
        # Cap at top_n
        cap = min(top_n, len(sorted_pos))
        sorted_pos = sorted_pos[:cap]

        analogues: list[AnalogueRecord] = []
        for rank, pos in enumerate(sorted_pos, start=1):
            train_idx = int(indices[i][pos])
            analogue_date = _normalize_date(train_dates[train_idx])

            analogues.append(AnalogueRecord(
                rank=rank,
                ticker=str(train_tickers[train_idx]),
                date=analogue_date,
                distance=float(distances[i][pos]),
                label=int(train_targets[train_idx]),
                fwd_return=float(train_returns[train_idx]),
            ))

        qdate = _normalize_date(val_dates[i])

        entries.append(JournalEntry(
            query_date=qdate,
            query_ticker=str(val_tickers[i]),
            raw_prob=float(raw_probs[i]),
            calibrated_prob=float(cal_probs[i]),
            signal=sig,
            n_matches=int(n_matches[i]),
            top_analogues=analogues,
        ))

    return entries


def write_journal_parquet(entries: list[JournalEntry], path: Path | str) -> None:
    """Write journal entries to a flat Parquet file.

    Schema (one row per analogue):
        query_date, query_ticker, raw_prob, calibrated_prob, signal,
        n_matches, analogue_rank, analogue_ticker, analogue_date,
        analogue_distance, analogue_label, analogue_fwd_return

    A single JournalEntry with top_25 analogues produces 25 rows.
    Reconstruct per-signal view by grouping on (query_date, query_ticker).

    Args:
        entries: List of JournalEntry objects from build_journal_entries().
        path:    Output file path (e.g., results/journals/fold6.parquet).
    """
    rows = []
    for e in entries:
        for a in e.top_analogues:
            rows.append({
                "query_date":          e.query_date,
                "query_ticker":        e.query_ticker,
                "raw_prob":            e.raw_prob,
                "calibrated_prob":     e.calibrated_prob,
                "signal":              e.signal,
                "n_matches":           e.n_matches,
                "analogue_rank":       a.rank,
                "analogue_ticker":     a.ticker,
                "analogue_date":       a.date,
                "analogue_distance":   a.distance,
                "analogue_label":      a.label,
                "analogue_fwd_return": a.fwd_return,
            })

    if not rows:
        # Write empty DataFrame with correct schema
        df = pd.DataFrame(columns=[
            "query_date", "query_ticker", "raw_prob", "calibrated_prob",
            "signal", "n_matches", "analogue_rank", "analogue_ticker",
            "analogue_date", "analogue_distance", "analogue_label",
            "analogue_fwd_return",
        ])
    else:
        df = pd.DataFrame(rows)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)


def read_journal_parquet(path: Path | str) -> pd.DataFrame:
    """Read a journal Parquet file back into a DataFrame.

    Returns flat DataFrame (one row per analogue).
    Use .groupby(["query_date", "query_ticker"]) for per-signal views.
    """
    return pd.read_parquet(str(path))


def top_n_view(journal_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Filter journal to top-N analogues per signal.

    Args:
        journal_df: Output of read_journal_parquet().
        n:          Maximum analogue rank to include. Any positive int works.
                    Typical values: 5, 10, 25.

    Returns:
        Filtered DataFrame with analogue_rank <= n.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    return journal_df[journal_df["analogue_rank"] <= n].copy()
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_journal.py -v 2>&1
```

Expected: `4 passed`

- [ ] **Step 6: Run full suite to check no regressions**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: `543 passed` (or 547 with new tests)

- [ ] **Step 7: Commit**

```bash
git add pattern_engine/journal.py tests/unit/test_journal.py results/journals/.gitkeep
git commit -m "feat(journal): add JournalEntry dataclass and Parquet writer"
```

---

## Task 3: Instrument PatternMatcher to populate `last_journal`

**Context:** The journal must capture the **post-calibration** signals (what the caller actually sees), not the pre-calibration signals from `_package_results()`. The Platt calibrator runs AFTER the batch loop and regenerates all signals at lines 722–746. The solution: store batch metadata during the loop, then build journal entries in a single second pass after calibration using `out_probs` and the final `all_signals`.

Additionally, add `and self._fitted` guard so the journal code is skipped during the Platt calibration double-pass inside `fit()` (which calls `self.query(_cal_db)` when `self._fitted` is `True` but `self._calibrator` is `None`). Wait — actually during the double-pass, `self._fitted` IS True (set at line 516 before the calibration block in `fit()`). The correct guard is checking whether `self._calibrator is None` inside `fit()` context, but that's not accessible in `query()`. The simpler guard: since journal entries during the double-pass are harmless (they get overwritten by the outer `query()` call) but wasteful, just accept this minor overhead. The entries will be overwritten anyway.

**Files:**
- Modify: `pattern_engine/matcher.py` — `__init__` and `query()` method

- [ ] **Step 1: Add `last_journal` attribute to `__init__`**

After `self._active_overlays: list = []` (line ~134), add:

```python
# Decision journal (populated after query() when journal_top_n > 0)
self.last_journal: list = []   # list[JournalEntry]
```

- [ ] **Step 2: Add `journal_top_n` to `WalkForwardConfig` in `run_walkforward.py`**

In `WalkForwardConfig` dataclass (after `cal_max_samples`), add:

```python
journal_top_n: int = 0   # 0=disabled, 5/10/25=capture top-N analogues per BUY/SELL
```

- [ ] **Step 3: Store batch metadata during the loop (NOT journal entries yet)**

In `query()`, add a batch metadata accumulator BEFORE the `for batch_start` loop:

```python
        _j_top_n = getattr(cfg, 'journal_top_n', 0)
        _batch_meta: list = []   # stores (top_mask, dist, idx, tickers_b, dates_b, raw_probs_b, start, end)
```

Inside the loop, after `all_ensembles.extend(ens_b)`, add:

```python
            # Store batch metadata for journal (second pass after calibration)
            if _j_top_n > 0:
                _batch_meta.append((
                    top_mask.copy(),
                    distances_b.copy(),
                    indices_b.copy(),
                    val_tickers_arr[batch_start:batch_end].copy(),
                    val_dates_arr[batch_start:batch_end].copy(),
                    np.array(prob_b.tolist()),   # raw probs for this batch
                    batch_start,
                    batch_end,
                ))
```

- [ ] **Step 4: Build journal entries AFTER calibration using final signals**

The calibration block ends at line ~746 and sets `out_probs` and final `all_signals`.
**After** the entire `if self._calibrator is not None: ... else: ...` block and immediately before the `return (out_probs, ...)` statement, add:

```python
        # Build decision journal from post-calibration signals (correct final state)
        self.last_journal = []
        if _j_top_n > 0 and _batch_meta:
            from pattern_engine.journal import build_journal_entries
            for (_tm, _dist, _idx, _tickers_b, _dates_b, _raw_b, _s, _e) in _batch_meta:
                _batch_size_j = _e - _s
                _final_sigs_b = list(all_signals)[_s:_e]   # post-calibration slice
                _cal_b = out_probs[_s:_e]
                _nm_b = all_n_matches[_s:_e]
                _entries = build_journal_entries(
                    top_masks=_tm,
                    distances=_dist,
                    indices=_idx,
                    val_tickers=_tickers_b,
                    val_dates=_dates_b,
                    raw_probs=_raw_b,
                    cal_probs=_cal_b,
                    signals=_final_sigs_b,
                    n_matches=_nm_b,
                    train_tickers=self._train_tickers_arr,
                    train_dates=self._train_dates_arr,
                    train_targets=self._train_target_arr,
                    train_returns=self._train_ret_arr,
                    top_n=_j_top_n,
                )
                self.last_journal.extend(_entries)
```

Note: `all_signals` at this point is a Python list (not np.array) because the calibration block rebuilds it as `all_signals = []` (line ~731) then extends it. Use `list(all_signals)[_s:_e]` to slice correctly.

- [ ] **Step 5: Write integration test**

Add to `tests/unit/test_journal.py`:

```python
def test_matcher_populates_last_journal():
    """PatternMatcher.last_journal is populated when journal_top_n > 0."""
    import numpy as np
    import pandas as pd
    from datetime import date
    from dataclasses import dataclass, field

    @dataclass
    class MockConfig:
        top_k: int = 5
        max_distance: float = 999.0
        distance_weighting: str = "uniform"
        feature_weights: dict = field(default_factory=dict)
        batch_size: int = 256
        confidence_threshold: float = 0.55
        agreement_spread: float = 0.01
        min_matches: int = 1
        exclude_same_ticker: bool = False
        same_sector_only: bool = False
        regime_filter: bool = False
        regime_fallback: bool = False
        projection_horizon: str = "fwd_7d_up"
        calibration_method: str = "none"
        use_hnsw: bool = False
        use_sax_filter: bool = False
        use_wfa_rerank: bool = False
        use_ib_compression: bool = False
        journal_top_n: int = 5   # enable journal

    from pattern_engine.matcher import PatternMatcher
    # Build minimal 20-row training set
    rng = np.random.RandomState(0)
    n = 20
    train_db = pd.DataFrame({
        "Ticker": ["AAPL"] * n,
        "Date": pd.date_range("2020-01-01", periods=n),
        "ret_1d": rng.randn(n) * 0.01,
        "ret_3d": rng.randn(n) * 0.02,
        "fwd_7d_up": (rng.rand(n) > 0.5).astype(int),
        "fwd_7d": rng.randn(n) * 0.03,
    })
    val_db = pd.DataFrame({
        "Ticker": ["MSFT"] * 3,
        "Date": pd.date_range("2024-01-01", periods=3),
        "ret_1d": rng.randn(3) * 0.01,
        "ret_3d": rng.randn(3) * 0.02,
        "fwd_7d_up": [1, 0, 1],
        "fwd_7d": [0.02, -0.01, 0.03],
    })

    cfg = MockConfig()
    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, ["ret_1d", "ret_3d"])
    matcher.query(val_db, verbose=0)

    assert hasattr(matcher, "last_journal")
    # last_journal contains only BUY/SELL entries (HOLDs are skipped)
    for entry in matcher.last_journal:
        assert entry.signal in ("BUY", "SELL")
        assert len(entry.top_analogues) <= 5
        for a in entry.top_analogues:
            assert a.rank >= 1
            assert a.ticker == "AAPL"
```

- [ ] **Step 6: Run all journal tests**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_journal.py -v 2>&1
```

Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add pattern_engine/matcher.py scripts/run_walkforward.py tests/unit/test_journal.py
git commit -m "feat(journal): instrument PatternMatcher.query() to populate last_journal"
```

---

## Task 4: Add journal output to walk-forward folds

**Context:** Enable `journal_top_n=25` in the walk-forward config so each fold writes a Parquet file with the top-25 analogues for every BUY/SELL signal. Files land in `results/journals/`.

**Files:**
- Modify: `scripts/run_walkforward.py`

- [ ] **Step 1: Enable journal in WalkForwardConfig**

Change the newly-added `journal_top_n: int = 0` to `journal_top_n: int = 25`.

- [ ] **Step 2: Write journal file after each fold in `run_fold()`**

The `run_fold()` function currently ends at `return {...}`. After `matcher.query(val_db, verbose=0)`, add:

```python
    # Write decision journal for this fold (BUY/SELL signals only)
    _j_top_n = getattr(cfg, 'journal_top_n', 0)
    if _j_top_n > 0 and matcher.last_journal:
        from pattern_engine.journal import write_journal_parquet
        from pathlib import Path
        _jdir = REPO_ROOT / "results" / "journals"
        _jdir.mkdir(parents=True, exist_ok=True)
        _jpath = _jdir / f"journal_fold_{fold['label'].replace(' ', '_')}.parquet"
        write_journal_parquet(matcher.last_journal, _jpath)
        print(f"  Journal: {len(matcher.last_journal)} BUY/SELL entries -> {_jpath.name}")
```

- [ ] **Step 3: Run Fold 6 only to verify journal output**

Temporarily change `FOLDS` to just the last entry for a fast test:

```bash
# In a Python REPL or quick test:
PYTHONUTF8=1 py -3.12 -c "
import sys; sys.path.insert(0, '.')
# Quick smoke test: fit on 100 rows, query 20
import numpy as np, pandas as pd
from scripts.run_walkforward import WalkForwardConfig
cfg = WalkForwardConfig()
print('journal_top_n:', cfg.journal_top_n)
print('Config OK')
"
```

Expected: `journal_top_n: 25`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_walkforward.py
git commit -m "feat(journal): enable journal_top_n=25 in walk-forward, write per-fold Parquet"
```

---

## Task 5: Create `scripts/query_journal.py` inspection tool

**Files:**
- Create: `scripts/query_journal.py`

- [ ] **Step 1: Write the script**

```python
"""
query_journal.py — CLI inspection tool for FPPE decision journals.

Reads the Parquet journal files written by run_walkforward.py and displays
the top-N analogues for each BUY/SELL signal, optionally filtered by ticker,
date range, or signal type.

Usage:
    # Show all BUY signals for AAPL in fold 6, top-5 analogues each:
    py -3.12 scripts/query_journal.py --fold 2024-Val --ticker AAPL --top 5

    # Show top-10 for all SELL signals in all folds:
    py -3.12 scripts/query_journal.py --signal SELL --top 10

    # Show signals from a specific query date range:
    py -3.12 scripts/query_journal.py --from 2024-03-01 --to 2024-03-31 --top 25

Options:
    --fold      Fold label (e.g. 2024-Val). Omit for all folds.
    --ticker    Query ticker to filter (e.g. AAPL).
    --signal    BUY or SELL. Omit for both.
    --top       Analogues to show per signal: 5, 10, or 25 (default: 5).
    --from      Start date filter (YYYY-MM-DD).
    --to        End date filter (YYYY-MM-DD).
    --out       Optional CSV output path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = REPO_ROOT / "results" / "journals"


def load_all_journals(fold: str | None = None) -> pd.DataFrame:
    """Load all (or one) journal Parquet files from results/journals/."""
    if not JOURNAL_DIR.exists():
        print(f"No journal directory found at {JOURNAL_DIR}")
        print("Run scripts/run_walkforward.py with journal_top_n > 0 first.")
        sys.exit(1)

    files = sorted(JOURNAL_DIR.glob("journal_fold_*.parquet"))
    if not files:
        print(f"No journal files found in {JOURNAL_DIR}")
        sys.exit(1)

    if fold:
        # Match fold label (normalize spaces to underscores)
        target = fold.replace(" ", "_")
        files = [f for f in files if target in f.stem]
        if not files:
            print(f"No journal file matching fold '{fold}'")
            print(f"Available: {[f.stem for f in sorted(JOURNAL_DIR.glob('*.parquet'))]}")
            sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["fold"] = f.stem.replace("journal_fold_", "")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FPPE decision journal")
    parser.add_argument("--fold",   default=None, help="Fold label (e.g. 2024-Val)")
    parser.add_argument("--ticker", default=None, help="Query ticker (e.g. AAPL)")
    parser.add_argument("--signal", default=None, choices=["BUY", "SELL"], help="Signal type")
    parser.add_argument("--top",    default=5, type=int, choices=[5, 10, 25],
                        help="Analogues per signal (default: 5)")
    parser.add_argument("--from",   dest="date_from", default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--to",     dest="date_to", default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--out",    default=None, help="Optional CSV output path")
    args = parser.parse_args()

    df = load_all_journals(args.fold)

    # Apply filters
    if args.ticker:
        df = df[df["query_ticker"] == args.ticker.upper()]
    if args.signal:
        df = df[df["signal"] == args.signal]
    if args.date_from:
        df = df[pd.to_datetime(df["query_date"]) >= pd.Timestamp(args.date_from)]
    if args.date_to:
        df = df[pd.to_datetime(df["query_date"]) <= pd.Timestamp(args.date_to)]

    # Apply top-N filter
    df = df[df["analogue_rank"] <= args.top]

    if df.empty:
        print("No results matching filters.")
        sys.exit(0)

    # Summary
    n_signals = df.groupby(["query_date", "query_ticker"]).ngroups
    print(f"\nFound {n_signals} signals ({args.top} analogues each)")
    print(f"Folds:   {df['fold'].unique().tolist()}")
    print(f"Signals: {df['signal'].value_counts().to_dict()}")
    print(f"Tickers: {df['query_ticker'].nunique()} unique\n")

    # Display
    display_cols = [
        "fold", "query_date", "query_ticker", "calibrated_prob", "signal",
        "n_matches", "analogue_rank", "analogue_ticker", "analogue_date",
        "analogue_distance", "analogue_label", "analogue_fwd_return",
    ]
    print(df[display_cols].to_string(index=False, max_rows=50))

    if args.out:
        df[display_cols].to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs (will show "no journal files" until full walk-forward completes)**

```bash
PYTHONUTF8=1 py -3.12 scripts/query_journal.py --help 2>&1 | head -10
```

Expected: help text printed without import errors.

- [ ] **Step 3: Commit**

```bash
git add scripts/query_journal.py
git commit -m "feat(journal): add query_journal.py CLI inspection tool"
```

---

## Task 6: Create `SignalFilterBase` ABC and `SignalPipeline`

**Context:** All three signal filters must share a common polymorphic interface so `SignalPipeline` can dispatch to any of them without knowing the concrete type (Interface Segregation Principle). `SignalPipeline` encapsulates the filter chain so `run_walkforward.py` only constructs and calls the pipeline rather than managing individual filter dispatches (Single Responsibility Principle).

**Files:**
- Create: `pattern_engine/signal_filter_base.py`
- Create: `pattern_engine/signal_pipeline.py`
- Create: `tests/unit/test_signal_filter_base.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/test_signal_filter_base.py`:

```python
"""Tests for SignalFilterBase ABC and SignalPipeline."""
import numpy as np
import pandas as pd
import pytest


def test_signal_filter_base_is_abstract():
    """SignalFilterBase cannot be instantiated directly."""
    from pattern_engine.signal_filter_base import SignalFilterBase
    with pytest.raises(TypeError):
        SignalFilterBase()  # type: ignore


def test_concrete_filter_without_apply_raises():
    """Subclass that does not implement apply() raises TypeError on instantiation."""
    from pattern_engine.signal_filter_base import SignalFilterBase

    class IncompleteFilter(SignalFilterBase):
        pass

    with pytest.raises(TypeError):
        IncompleteFilter()


def test_signal_pipeline_applies_filters_in_order():
    """SignalPipeline threads signals through each filter sequentially."""
    from pattern_engine.signal_filter_base import SignalFilterBase
    from pattern_engine.signal_pipeline import SignalPipeline

    class NoOpFilter(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            return list(signals), np.zeros(len(signals), dtype=bool)

    class HoldAllFilter(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            mask = np.array([s != "HOLD" for s in signals], dtype=bool)
            return ["HOLD"] * len(signals), mask

    pipeline = SignalPipeline(filters=[NoOpFilter(), HoldAllFilter()])
    probs = np.array([0.70, 0.30])
    signals = ["BUY", "SELL"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "JPM"]})
    filtered, mask = pipeline.run(probs, signals, val_db)
    assert filtered == ["HOLD", "HOLD"]
    assert mask.all()


def test_signal_pipeline_combined_mask_is_union():
    """combined_mask is the union of all per-filter masks."""
    from pattern_engine.signal_filter_base import SignalFilterBase
    from pattern_engine.signal_pipeline import SignalPipeline

    class FilterFirst(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            filtered = list(signals)
            mask = np.zeros(len(signals), dtype=bool)
            filtered[0] = "HOLD"
            mask[0] = True
            return filtered, mask

    class FilterSecond(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            filtered = list(signals)
            mask = np.zeros(len(signals), dtype=bool)
            filtered[1] = "HOLD"
            mask[1] = True
            return filtered, mask

    pipeline = SignalPipeline(filters=[FilterFirst(), FilterSecond()])
    probs = np.array([0.70, 0.68])
    signals = ["BUY", "BUY"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "JPM"]})
    filtered, mask = pipeline.run(probs, signals, val_db)
    assert filtered == ["HOLD", "HOLD"]
    assert mask[0] and mask[1]
```

- [ ] **Step 2: Run tests — confirm failure**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_signal_filter_base.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'pattern_engine.signal_filter_base'`

- [ ] **Step 3: Implement `pattern_engine/signal_filter_base.py`**

```python
"""
signal_filter_base.py — Abstract base class for all FPPE post-query signal filters.

All signal filters (SectorConvictionLayer, MomentumSignalFilter, SentimentVetoFilter)
inherit from SignalFilterBase and implement a unified apply() interface. This enables
SignalPipeline to dispatch polymorphically without knowing the concrete filter type.

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SignalFilterBase(ABC):
    """Abstract base class for post-query signal filters.

    Each filter receives the calibrated probabilities, current signals,
    and the validation DataFrame, then returns a (possibly modified)
    signal list alongside a boolean mask indicating changed positions.
    """

    @abstractmethod
    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Apply the filter to a set of signals.

        Args:
            probs:   (N,) calibrated probabilities from PatternMatcher.query().
            signals: List of N signal strings ("BUY"/"SELL"/"HOLD").
            val_db:  Validation DataFrame for this batch (must have Ticker).
            **kwargs: Filter-specific extras (e.g. ``sentiment={}``).

        Returns:
            (filtered_signals, filter_mask):
              filtered_signals: list[str] — signals after filtering.
              filter_mask:      (N,) bool — True where this filter changed the signal.
        """
        ...
```

- [ ] **Step 4: Implement `pattern_engine/signal_pipeline.py`**

```python
"""
signal_pipeline.py — Ordered filter chain for FPPE post-query signal processing.

SignalPipeline runs a list of SignalFilterBase-compatible filters in sequence,
accumulating a combined change mask. It is the single integration point between
PatternMatcher.query() and the signal intelligence layer in run_walkforward.py.

Usage:
    pipeline = SignalPipeline(filters=[conviction_layer, mom_filter])
    signals, combined_mask = pipeline.run(np.asarray(probs), signals, val_db)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class SignalPipeline:
    """Runs a sequence of SignalFilterBase filters in order.

    Each filter receives the signals as modified by all prior filters.
    The combined mask is the union of all per-filter change masks.

    Args:
        filters: Ordered list of SignalFilterBase instances.
    """

    def __init__(self, filters: list[SignalFilterBase]):
        self.filters = filters

    def run(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Run all filters in sequence.

        Args:
            probs:   (N,) calibrated probabilities.
            signals: Initial signal list.
            val_db:  Validation DataFrame for this batch.
            **kwargs: Forwarded to each filter's apply() call.

        Returns:
            (filtered_signals, combined_mask):
              combined_mask is True wherever ANY filter changed a signal.
        """
        combined_mask = np.zeros(len(signals), dtype=bool)
        for filt in self.filters:
            signals, mask = filt.apply(probs, signals, val_db, **kwargs)
            combined_mask |= mask
        return signals, combined_mask
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_signal_filter_base.py -v 2>&1
```

Expected: `4 passed`

- [ ] **Step 6: Run full suite**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: `543 passed` (or 547 with new tests)

- [ ] **Step 7: Commit**

```bash
git add pattern_engine/signal_filter_base.py pattern_engine/signal_pipeline.py tests/unit/test_signal_filter_base.py
git commit -m "feat(filters): add SignalFilterBase ABC and SignalPipeline"
```

---

## Task 7: Sector Conviction Layer

**Context:** For each K-NN query batch, compute the mean predicted probability per sector and compare to that sector's historical base rate. A sector where the K-NN engine agrees strongly (>2% above base rate) has "conviction". Sectors where K-NN barely beats the base rate can downgrade signal confidence.

**Files:**
- Create: `pattern_engine/sector_conviction.py`
- Create: `tests/unit/test_sector_conviction.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for SectorConvictionLayer."""
import numpy as np
import pandas as pd
import pytest

# pd is used in test_conviction_filter_vetoes_weak_sector (val_db construction)


def _make_train_db(n=100):
    import random
    rng = np.random.RandomState(42)
    sectors = ["Tech", "Finance", "Health"]
    tickers_by_sector = {
        "Tech": ["AAPL", "MSFT"],
        "Finance": ["JPM", "BAC"],
        "Health": ["JNJ", "PFE"],
    }
    rows = []
    for i in range(n):
        s = sectors[i % 3]
        t = tickers_by_sector[s][i % 2]
        rows.append({
            "Ticker": t,
            "Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
            "fwd_7d_up": int(rng.rand() > 0.4),
        })
    return pd.DataFrame(rows)


def test_fit_computes_sector_base_rates():
    """SectorConvictionLayer.fit() computes per-sector base rates."""
    from pattern_engine.sector_conviction import SectorConvictionLayer
    sector_map = {
        "AAPL": "Tech", "MSFT": "Tech",
        "JPM": "Finance", "BAC": "Finance",
        "JNJ": "Health", "PFE": "Health",
    }
    layer = SectorConvictionLayer(sector_map)
    train_db = _make_train_db(120)
    layer.fit(train_db, target_col="fwd_7d_up")

    assert "Tech" in layer.sector_base_rates_
    assert "Finance" in layer.sector_base_rates_
    assert "Health" in layer.sector_base_rates_
    for sector, rate in layer.sector_base_rates_.items():
        assert 0.0 <= rate <= 1.0


def test_sector_scores_returns_per_sector_mean():
    """sector_scores() aggregates probs by sector correctly."""
    from pattern_engine.sector_conviction import SectorConvictionLayer
    sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance"}
    layer = SectorConvictionLayer(sector_map)
    layer.sector_base_rates_ = {"Tech": 0.50, "Finance": 0.50}
    probs = np.array([0.70, 0.80, 0.60])
    tickers = np.array(["AAPL", "MSFT", "JPM"], dtype=object)
    scores = layer.sector_scores(probs, tickers)
    assert abs(scores["Tech"] - 0.75) < 1e-9
    assert abs(scores["Finance"] - 0.60) < 1e-9


def test_conviction_filter_vetoes_weak_sector():
    """Signals in sectors below conviction threshold are downgraded to HOLD."""
    from pattern_engine.sector_conviction import SectorConvictionLayer
    sector_map = {"AAPL": "Tech", "JPM": "Finance"}
    layer = SectorConvictionLayer(sector_map, min_sector_lift=0.05)
    # Tech base_rate=0.55, Finance base_rate=0.55
    layer.sector_base_rates_ = {"Tech": 0.55, "Finance": 0.55}

    # AAPL (Tech): prob=0.70 → lift=0.15 → above threshold → keep BUY
    # JPM (Finance): prob=0.58 → lift=0.03 → below min_sector_lift → veto to HOLD
    probs = np.array([0.70, 0.58])
    signals = ["BUY", "BUY"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "JPM"]})

    filtered_signals, veto_mask = layer.apply(probs, signals, val_db)
    assert filtered_signals[0] == "BUY"
    assert filtered_signals[1] == "HOLD"
    assert veto_mask[1] is True
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sector_conviction.py -v 2>&1 | head -10
```

- [ ] **Step 3: Implement `pattern_engine/sector_conviction.py`**

```python
"""
sector_conviction.py — Sector-level conviction scoring for FPPE signals.

Computes the mean K-NN predicted probability per sector and compares to
that sector's historical base rate. Signals in sectors where K-NN doesn't
beat the base rate by at least min_sector_lift are downgraded to HOLD.

Inspired by ARC Solomon's thematic conviction scoring: rather than acting
on individual ticker signals alone, require that the sector context supports
the direction.

Usage (post-query filter):
    layer = SectorConvictionLayer(SECTOR_MAP, min_sector_lift=0.03)
    layer.fit(train_db, target_col=cfg.projection_horizon)
    filtered_signals, veto_mask = layer.apply(probs, signals, val_tickers)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from pattern_engine.signal_filter_base import SignalFilterBase


class SectorConvictionLayer(SignalFilterBase):
    """Post-query filter that vetoes signals in low-conviction sectors.

    A sector has conviction when the K-NN aggregate predicted probability
    exceeds the sector's historical base rate by at least min_sector_lift.
    Signals in sectors below this bar are downgraded to HOLD.

    Args:
        sector_map:       ticker -> sector name mapping (SECTOR_MAP from sector.py).
        min_sector_lift:  Minimum excess probability over base rate required
                          to retain signals. Default 0.03 (3 percentage points).
    """

    def __init__(self, sector_map: Dict[str, str], min_sector_lift: float = 0.03):
        self.sector_map = sector_map
        self.min_sector_lift = min_sector_lift
        self.sector_base_rates_: Dict[str, float] = {}

    def fit(self, train_db: pd.DataFrame, target_col: str = "fwd_7d_up") -> "SectorConvictionLayer":
        """Compute per-sector historical base rates from training data.

        Args:
            train_db:   Training DataFrame (must have Ticker and target_col).
            target_col: Binary target column name.
        Returns:
            self (for method chaining).
        """
        if target_col not in train_db.columns:
            return self  # no-op if target not present

        df = train_db.copy()
        df["_sector"] = df["Ticker"].map(lambda t: self.sector_map.get(t, "Unknown"))
        grouped = df.groupby("_sector")[target_col].mean()
        self.sector_base_rates_ = grouped.to_dict()
        return self

    def sector_scores(
        self,
        probs: np.ndarray,
        tickers: np.ndarray,
    ) -> Dict[str, float]:
        """Compute mean predicted probability per sector.

        Args:
            probs:   (N,) calibrated probabilities from PatternMatcher.query().
            tickers: (N,) object array of ticker strings.

        Returns:
            Dict mapping sector name -> mean probability for tickers in that sector.
        """
        sectors = pd.Series([self.sector_map.get(str(t), "Unknown") for t in tickers])
        return pd.Series(probs).groupby(sectors).mean().to_dict()

    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> Tuple[list[str], np.ndarray]:
        """Veto BUY/SELL signals where sector conviction is insufficient.

        For each BUY/SELL signal, check whether the sector-level mean
        probability exceeds the sector's base rate by min_sector_lift.
        Signals that fail this check are downgraded to HOLD.

        Args:
            probs:   (N,) calibrated probabilities.
            signals: List of N signal strings ("BUY"/"SELL"/"HOLD").
            val_db:  Validation DataFrame (must have Ticker column).

        Returns:
            (filtered_signals, veto_mask):
              filtered_signals: list[str] with vetoed signals set to "HOLD".
              veto_mask: (N,) bool array, True where signal was vetoed.
        """
        if not self.sector_base_rates_:
            return signals, np.zeros(len(signals), dtype=bool)

        tickers = val_db["Ticker"].values
        sector_scores = self.sector_scores(probs, tickers)
        sectors = pd.Series([self.sector_map.get(str(t), "Unknown") for t in tickers]).values

        filtered = list(signals)
        veto_mask = np.zeros(len(signals), dtype=bool)

        for i, sig in enumerate(signals):
            if sig == "HOLD":
                continue  # already HOLD, skip
            sector = sectors[i]
            base_rate = self.sector_base_rates_.get(sector, 0.5)
            sector_mean_prob = sector_scores.get(sector, 0.5)
            lift = sector_mean_prob - base_rate
            if lift < self.min_sector_lift:
                filtered[i] = "HOLD"
                veto_mask[i] = True

        return filtered, veto_mask
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sector_conviction.py -v 2>&1
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add pattern_engine/sector_conviction.py tests/unit/test_sector_conviction.py
git commit -m "feat(sector-conviction): add SectorConvictionLayer post-query filter"
```

---

## Task 8: Momentum Signal Filter

**Context:** Require that the ticker's recent return is outperforming (for BUY) or underperforming (for SELL) its sector average. This is an independent confirmatory signal alongside K-NN.

**Files:**
- Create: `pattern_engine/momentum_signal.py`
- Create: `tests/unit/test_momentum_signal.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for MomentumSignalFilter."""
import numpy as np
import pandas as pd
import pytest
from datetime import date


def _make_val_db():
    """Val DataFrame with ticker, date, and 7d return columns."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "JPM", "BAC"],
        "Date": [pd.Timestamp("2024-01-05")] * 4,
        "ret_7d": [0.05, 0.02, -0.03, -0.01],
    })


def test_fit_computes_sector_averages():
    """fit() computes rolling sector averages from training data."""
    from pattern_engine.momentum_signal import MomentumSignalFilter
    sector_map = {"AAPL": "Tech", "MSFT": "Tech", "JPM": "Finance", "BAC": "Finance"}
    filt = MomentumSignalFilter(sector_map, lookback_col="ret_7d")
    train_db = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "JPM", "BAC"] * 10,
        "Date": pd.date_range("2023-01-01", periods=40),
        "ret_7d": np.array([0.03, 0.02, -0.01, -0.02] * 10),
    })
    filt.fit(train_db)
    assert "Tech" in filt.sector_mean_returns_
    assert "Finance" in filt.sector_mean_returns_


def test_momentum_agrees_with_buy():
    """BUY signal is kept when ticker outperforms sector by threshold."""
    from pattern_engine.momentum_signal import MomentumSignalFilter
    sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
    filt = MomentumSignalFilter(sector_map, lookback_col="ret_7d", min_outperformance=0.02)
    filt.sector_mean_returns_ = {"Tech": 0.01}

    val_db = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Date": [pd.Timestamp("2024-01-05")],
        "ret_7d": [0.05],  # outperforms Tech avg (0.01) by 0.04 > 0.02 threshold
    })
    probs = np.array([0.70])
    signals = ["BUY"]
    result, agreed = filt.apply(probs, signals, val_db)
    assert result[0] == "BUY"
    assert agreed[0] is True


def test_momentum_vetoes_disagreeing_buy():
    """BUY signal is downgraded when ticker underperforms sector."""
    from pattern_engine.momentum_signal import MomentumSignalFilter
    sector_map = {"AAPL": "Tech"}
    filt = MomentumSignalFilter(sector_map, lookback_col="ret_7d", min_outperformance=0.02)
    filt.sector_mean_returns_ = {"Tech": 0.04}

    val_db = pd.DataFrame({
        "Ticker": ["AAPL"],
        "Date": [pd.Timestamp("2024-01-05")],
        "ret_7d": [0.01],  # underperforms Tech avg (0.04): delta=-0.03
    })
    probs = np.array([0.68])
    signals = ["BUY"]
    result, agreed = filt.apply(probs, signals, val_db)
    assert result[0] == "HOLD"
    assert agreed[0] is False
```

- [ ] **Step 2: Implement `pattern_engine/momentum_signal.py`**

```python
"""
momentum_signal.py — Ticker-vs-sector momentum agreement filter.

Inspired by ARC Solomon's multi-agent agreement requirement: a BUY signal
should only be acted on when the ticker is outperforming its sector average
on the same return window. A SELL should only be acted on when underperforming.

Signals where K-NN and momentum disagree are downgraded to HOLD.

Usage:
    filt = MomentumSignalFilter(SECTOR_MAP, lookback_col="ret_7d",
                                 min_outperformance=0.015)
    filt.fit(train_db)
    filtered_signals, agreed = filt.apply(probs, signals, val_db)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class MomentumSignalFilter(SignalFilterBase):
    """Filters signals where ticker momentum disagrees with the K-NN direction.

    For BUY: require ticker's lookback_col return > sector avg + min_outperformance.
    For SELL: require ticker's lookback_col return < sector avg - min_outperformance.
    Disagreeing signals are downgraded to HOLD.

    Args:
        sector_map:           ticker -> sector name mapping.
        lookback_col:         Return column to use for momentum (e.g. "ret_7d").
        min_outperformance:   Minimum excess return vs sector to retain signal
                              (default 0.015 = 1.5 percentage points).
    """

    def __init__(
        self,
        sector_map: Dict[str, str],
        lookback_col: str = "ret_7d",
        min_outperformance: float = 0.015,
    ):
        self.sector_map = sector_map
        self.lookback_col = lookback_col
        self.min_outperformance = min_outperformance
        self.sector_mean_returns_: Dict[str, float] = {}

    def fit(self, train_db: pd.DataFrame) -> "MomentumSignalFilter":
        """Compute sector mean returns from training data.

        Args:
            train_db: Training DataFrame (must have Ticker and lookback_col).
        Returns:
            self.
        """
        if self.lookback_col not in train_db.columns:
            return self

        df = train_db.copy()
        df["_sector"] = df["Ticker"].map(lambda t: self.sector_map.get(str(t), "Unknown"))
        grouped = df.groupby("_sector")[self.lookback_col].mean()
        self.sector_mean_returns_ = grouped.to_dict()
        return self

    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
    ) -> Tuple[list[str], np.ndarray]:
        """Downgrade signals where ticker momentum disagrees with K-NN direction.

        Args:
            probs:   (N,) calibrated probabilities.
            signals: List of N signal strings.
            val_db:  Validation DataFrame (must have Ticker, lookback_col).

        Returns:
            (filtered_signals, agreed):
              filtered_signals: list[str] with vetoed signals set to HOLD.
              agreed: (N,) bool array, True where signal was kept.
        """
        if not self.sector_mean_returns_ or self.lookback_col not in val_db.columns:
            return signals, np.ones(len(signals), dtype=bool)

        tickers = val_db["Ticker"].values
        returns = val_db[self.lookback_col].values

        filtered = list(signals)
        agreed = np.ones(len(signals), dtype=bool)

        for i, sig in enumerate(signals):
            if sig == "HOLD":
                continue
            ticker = str(tickers[i])
            sector = self.sector_map.get(ticker, "Unknown")
            sector_avg = self.sector_mean_returns_.get(sector, 0.0)
            ticker_ret = float(returns[i]) if not np.isnan(returns[i]) else sector_avg

            if sig == "BUY":
                if ticker_ret - sector_avg < self.min_outperformance:
                    filtered[i] = "HOLD"
                    agreed[i] = False
            elif sig == "SELL":
                if sector_avg - ticker_ret < self.min_outperformance:
                    filtered[i] = "HOLD"
                    agreed[i] = False

        return filtered, agreed
```

- [ ] **Step 3: Run tests**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_momentum_signal.py -v 2>&1
```

Expected: `3 passed`

- [ ] **Step 4: Commit**

```bash
git add pattern_engine/momentum_signal.py tests/unit/test_momentum_signal.py
git commit -m "feat(momentum): add MomentumSignalFilter K-NN agreement check"
```

---

## Task 9: Sentiment Veto Filter

**Context:** Veto BUY signals for tickers with recent negative news sentiment from FMP MCP. For backtesting, accepts pre-fetched sentiment; for live trading, calls FMP MCP directly.

**Files:**
- Create: `pattern_engine/sentiment_veto.py`
- Create: `tests/unit/test_sentiment_veto.py`

- [ ] **Step 1: Write failing tests (using mock sentiment)**

```python
"""Tests for SentimentVetoFilter — uses mock FMP data."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def test_veto_negative_sentiment_buy():
    """BUY is downgraded to HOLD when sentiment score < threshold."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    probs = np.array([0.70, 0.68])
    signals = ["BUY", "BUY"]
    tickers = ["AAPL", "META"]
    # AAPL sentiment is fine; META has bad news
    sentiment = {"AAPL": 0.10, "META": -0.35}
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)
    assert filtered[0] == "BUY"
    assert filtered[1] == "HOLD"


def test_sell_not_vetoed_by_negative_sentiment():
    """SELL signals are NOT vetoed by negative sentiment (confirms bearish thesis)."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    probs = np.array([0.25])
    signals = ["SELL"]
    tickers = ["META"]
    sentiment = {"META": -0.50}
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)
    assert filtered[0] == "SELL"  # Negative sentiment confirms SELL — don't veto


def test_hold_passes_through():
    """HOLD signals are unaffected by sentiment filter."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter()
    filtered = filt.apply_with_sentiment(
        np.array([0.51]), ["HOLD"], ["AAPL"], {"AAPL": -1.0}
    )
    assert filtered[0] == "HOLD"


def test_missing_ticker_sentiment_neutral():
    """Tickers with no sentiment data are treated as neutral (not vetoed)."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    filtered = filt.apply_with_sentiment(
        np.array([0.70]), ["BUY"], ["UNKNOWN_TICKER"], {}
    )
    assert filtered[0] == "BUY"


def test_apply_unified_interface():
    """apply() satisfies SignalFilterBase interface — neutral sentiment means no veto."""
    from pattern_engine.sentiment_veto import SentimentVetoFilter
    import pandas as pd
    filt = SentimentVetoFilter(veto_threshold=-0.20)
    probs = np.array([0.70, 0.68])
    signals = ["BUY", "BUY"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "META"]})
    # No sentiment kwarg provided → empty dict → neutral → no vetos
    filtered, mask = filt.apply(probs, signals, val_db)
    assert filtered == ["BUY", "BUY"]
    assert not mask.any()
```

- [ ] **Step 2: Implement `pattern_engine/sentiment_veto.py`**

```python
"""
sentiment_veto.py — News sentiment veto filter for FPPE BUY signals.

Inspired by ARC Solomon's narrative analyst + risk officer architecture:
a BUY signal should be vetoed if recent news sentiment for the ticker
is strongly negative, since markets increasingly move on narrative.

Two usage modes:
    1. Backtesting: pass pre-fetched sentiment dict (ticker -> score)
    2. Live trading: call fetch_sentiment() which queries FMP MCP

Sentiment score range: [-1.0, +1.0] (FMP normalized sentiment).
Veto threshold: -0.20 by default (veto BUY if score < -0.20).

Usage:
    filt = SentimentVetoFilter(veto_threshold=-0.20, lookback_days=3)

    # Backtesting (pre-fetched):
    sentiment = {"AAPL": 0.15, "META": -0.35}
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)

    # Live (FMP MCP call):
    sentiment = filt.fetch_sentiment(tickers, current_date)
    filtered = filt.apply_with_sentiment(probs, signals, tickers, sentiment)

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pattern_engine.signal_filter_base import SignalFilterBase


class SentimentVetoFilter(SignalFilterBase):
    """Vetoes BUY signals when recent news sentiment is strongly negative.

    Negative sentiment confirms bearish thesis for SELL signals — those
    are NOT vetoed. Only BUY signals are affected.

    Sentiment score interpretation:
        > +0.20:  Positive news — supports BUY
        -0.20 to +0.20: Neutral — no veto
        < -0.20:  Negative news — veto BUY -> HOLD

    Args:
        veto_threshold: Sentiment score below which BUY is vetoed (default -0.20).
        lookback_days:  Days of news to average for sentiment score (default 3).
    """

    def __init__(
        self,
        veto_threshold: float = -0.20,
        lookback_days: int = 3,
    ):
        self.veto_threshold = veto_threshold
        self.lookback_days = lookback_days

    def fetch_sentiment(
        self,
        tickers: list[str],
        query_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """Fetch news sentiment scores from FMP MCP for a list of tickers.

        Queries the FMP stock-news endpoint for each ticker, averaging
        sentiment over the past lookback_days. Returns a dict mapping
        ticker -> mean sentiment score [-1.0, +1.0].

        NOTE: This method requires FMP MCP to be configured. For backtesting,
        use apply_with_sentiment() directly with pre-fetched data instead.

        Args:
            tickers:    List of ticker symbols (uppercase).
            query_date: Date to fetch sentiment for (defaults to today).

        Returns:
            Dict[str, float]: ticker -> mean sentiment score.
            Tickers with no news return 0.0 (neutral).
        """
        if query_date is None:
            query_date = date.today()

        since_date = query_date - timedelta(days=self.lookback_days)
        scores: Dict[str, float] = {}

        for ticker in tickers:
            try:
                # FMP MCP call — stock-news endpoint
                # In live usage: call mcp__fmp__stock_news(symbol=ticker, limit=20)
                # and average the sentimentScore field over lookback_days
                # This is a placeholder — wire to FMP MCP in live runner
                scores[ticker] = 0.0  # neutral default until FMP wired
            except (ConnectionError, TimeoutError, OSError, ValueError) as exc:
                logging.warning("SentimentVetoFilter: failed to fetch %s: %s", ticker, exc)
                scores[ticker] = 0.0  # neutral on error

        return scores

    def apply_with_sentiment(
        self,
        probs: np.ndarray,
        signals: list[str],
        tickers: list[str],
        sentiment: Dict[str, float],
    ) -> list[str]:
        """Apply sentiment veto to signals using pre-fetched sentiment scores.

        BUY signals with sentiment < veto_threshold are downgraded to HOLD.
        SELL signals are never vetoed by negative sentiment (it confirms them).
        HOLD signals pass through unchanged.

        Args:
            probs:     (N,) calibrated probabilities (informational only).
            signals:   List of N signal strings.
            tickers:   List of N ticker symbols.
            sentiment: Dict[str, float] from fetch_sentiment() or pre-fetched.

        Returns:
            List[str]: Filtered signal strings.
        """
        filtered = list(signals)

        for i, sig in enumerate(signals):
            if sig != "BUY":
                continue  # SELL confirms negative news; HOLD unchanged

            ticker = tickers[i] if i < len(tickers) else ""
            score = sentiment.get(ticker, 0.0)  # missing = neutral

            if score < self.veto_threshold:
                filtered[i] = "HOLD"

        return filtered

    def apply(
        self,
        probs: np.ndarray,
        signals: list[str],
        val_db: pd.DataFrame,
        **kwargs,
    ) -> tuple[list[str], np.ndarray]:
        """Unified SignalFilterBase interface.

        Wraps apply_with_sentiment() using sentiment from kwargs.
        Pass sentiment={ticker: score} for live mode; omit for neutral (no veto).

        Args:
            probs:    (N,) calibrated probabilities.
            signals:  List of N signal strings.
            val_db:   Validation DataFrame (must have Ticker).
            **kwargs: Optional ``sentiment`` dict override.

        Returns:
            (filtered_signals, veto_mask).
        """
        tickers = list(val_db["Ticker"].values)
        sentiment: Dict[str, float] = kwargs.get("sentiment", {})
        filtered = self.apply_with_sentiment(probs, signals, tickers, sentiment)
        veto_mask = np.array(
            [filtered[i] != signals[i] for i in range(len(signals))], dtype=bool
        )
        return filtered, veto_mask
```

- [ ] **Step 3: Run tests**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_sentiment_veto.py -v 2>&1
```

Expected: `4 passed`

- [ ] **Step 4: Commit**

```bash
git add pattern_engine/sentiment_veto.py tests/unit/test_sentiment_veto.py
git commit -m "feat(sentiment-veto): add SentimentVetoFilter with FMP MCP stub"
```

---

## Task 10: Integration — wire filters into walk-forward script

**Context:** Add optional filter chain after `matcher.query()` in `run_walkforward.py`. Each filter is off by default (false/0 flags). Enable via config flags to test BSS impact.

**Files:**
- Modify: `scripts/run_walkforward.py`

- [ ] **Step 1: Add filter flags to `WalkForwardConfig`**

Add three new fields after `use_ib_compression`:

```python
# Signal intelligence filters (M9)
use_sector_conviction: bool = False   # SectorConvictionLayer
use_momentum_filter: bool = False     # MomentumSignalFilter
use_sentiment_veto: bool = False      # SentimentVetoFilter (live only)
sector_conviction_lift: float = 0.03  # min sector lift threshold
momentum_min_outperformance: float = 0.015  # min ticker vs sector delta
```

- [ ] **Step 2: Add `SignalPipeline` dispatch in `run_fold()`**

After `probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)`, add:

```python
    # ── Optional signal intelligence filters (via SignalPipeline) ──────────
    signals = list(signals)  # ensure mutable list
    _active_filters = []

    if getattr(cfg, 'use_sector_conviction', False):
        from pattern_engine.sector_conviction import SectorConvictionLayer
        from pattern_engine.sector import SECTOR_MAP
        conviction_layer = SectorConvictionLayer(
            SECTOR_MAP,
            min_sector_lift=getattr(cfg, 'sector_conviction_lift', 0.03),
        )
        conviction_layer.fit(train_db, target_col=cfg.projection_horizon)
        _active_filters.append(conviction_layer)

    if getattr(cfg, 'use_momentum_filter', False):
        from pattern_engine.momentum_signal import MomentumSignalFilter
        from pattern_engine.sector import SECTOR_MAP
        # Use ret_7d if present (raw return), else ret_7d_norm (M9 vol-norm pipeline).
        _mom_col = "ret_7d" if "ret_7d" in val_db.columns else "ret_7d_norm"
        mom_filter = MomentumSignalFilter(
            SECTOR_MAP,
            lookback_col=_mom_col,
            min_outperformance=getattr(cfg, 'momentum_min_outperformance', 0.015),
        )
        mom_filter.fit(train_db)
        _active_filters.append(mom_filter)

    if _active_filters:
        from pattern_engine.signal_pipeline import SignalPipeline
        pipeline = SignalPipeline(filters=_active_filters)
        signals, _ = pipeline.run(np.asarray(probs), signals, val_db)
```

- [ ] **Step 3: Add filter count fields to result dict (keep existing keys)**

In the return dict for `run_fold()`, ADD these keys alongside the existing `buy`, `sell`, `hold` (do NOT rename them — renaming breaks the print statement at line ~209 and the early-return dict at line ~149):

```python
        # existing keys stay: "buy", "sell", "hold"
        # add post-filter counts for analysis
        "buy_after_filter":  int(np.sum(np.array(signals) == "BUY")),
        "sell_after_filter": int(np.sum(np.array(signals) == "SELL")),
        "hold_after_filter": int(np.sum(np.array(signals) == "HOLD")),
```

Also add `"buy_after_filter": 0, "sell_after_filter": 0, "hold_after_filter": 0` to the early-return dict (line ~149) for the `n_val == 0` case.

- [ ] **Step 4: Verify existing walk-forward still works (filters off by default)**

```bash
PYTHONUTF8=1 py -3.12 -c "
from scripts.run_walkforward import WalkForwardConfig
cfg = WalkForwardConfig()
assert cfg.use_sector_conviction == False
assert cfg.use_momentum_filter == False
assert cfg.use_sentiment_veto == False
print('Integration config OK')
"
```

Expected: `Integration config OK`

- [ ] **Step 5: Run full test suite**

```bash
PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow" 2>&1 | tail -5
```

Expected: all tests pass (≥543)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_walkforward.py
git commit -m "feat(filters): wire SectorConviction and MomentumFilter via SignalPipeline"
```

---

## Task 11: Final validation — run complete walk-forward with journal enabled

**Context:** This task validates the full pipeline end-to-end with `journal_top_n=25` enabled and verifies journal files are written correctly.

**Files:**
- No new code — validation only

- [ ] **Step 1: Verify all prior tasks are committed**

Run `git log --oneline -10` and confirm Tasks 2–10 commits are present.

- [ ] **Step 2: Confirm locked horizon setting**

Confirm `WalkForwardConfig.projection_horizon = "fwd_7d_up"` (locked setting per CLAUDE.md).

- [ ] **Step 3: Run Fold 6 only to validate journal output (fast test, ~400s)**

Temporarily modify `FOLDS` in `run_walkforward.py` to just `FOLDS[-1:]` (Fold 6 only) and run:

```bash
PYTHONUTF8=1 py -3.12 scripts/run_walkforward.py 2>&1
```

Expected output includes:
```
Journal: N BUY/SELL entries -> journal_fold_2024-Val.parquet
```

- [ ] **Step 4: Inspect journal output**

```bash
PYTHONUTF8=1 py -3.12 scripts/query_journal.py --fold 2024-Val --top 5 2>&1 | head -40
```

Expected: table showing top-5 analogues per BUY/SELL signal with dates, tickers, distances, labels.

- [ ] **Step 5: Restore full 6-fold FOLDS list and run complete walk-forward**

Revert `FOLDS` to all 6 folds if modified. Run full walk-forward.

- [ ] **Step 6: Final commit**

```bash
git add docs/session-logs/
git commit -m "feat(M9-signal-intelligence): complete journal, SignalFilterBase, SignalPipeline, sector conviction, momentum filter, sentiment veto"
```

---

## Testing Matrix

| Module | Test File | Tests |
|--------|-----------|-------|
| signal_filter_base.py + signal_pipeline.py | tests/unit/test_signal_filter_base.py | 4 (abstract, incomplete, pipeline_order, pipeline_mask) |
| journal.py | tests/unit/test_journal.py | 5 (entry, build_basic, top_n_cap, parquet_io, matcher_integration) |
| sector_conviction.py | tests/unit/test_sector_conviction.py | 3 (fit, scores, filter_veto) |
| momentum_signal.py | tests/unit/test_momentum_signal.py | 3 (fit, agree, disagree) |
| sentiment_veto.py | tests/unit/test_sentiment_veto.py | 5 (negative, sell_pass, hold_pass, missing_neutral, apply_interface) |

Total new tests: **20**. All existing 543 tests must continue to pass.

---

## Enabling Filters for Experiments

To test each filter's BSS impact, enable one at a time in `WalkForwardConfig`:

```python
# In scripts/run_walkforward.py — experiment variants:

# Experiment E1: Sector conviction only
cfg = WalkForwardConfig(use_sector_conviction=True, sector_conviction_lift=0.03)

# Experiment E2: Momentum agreement only
cfg = WalkForwardConfig(use_momentum_filter=True, momentum_min_outperformance=0.015)

# Experiment E3: Both filters combined
cfg = WalkForwardConfig(use_sector_conviction=True, use_momentum_filter=True)
```

Provenance rule (CLAUDE.md): each experiment result must be tied to a terminal output and logged in the session log before updating any locked settings.
