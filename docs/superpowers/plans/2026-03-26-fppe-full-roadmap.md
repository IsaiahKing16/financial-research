# FPPE Full Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take FPPE from BSS < 0 (broken probability calibration) through live deployment with $10,000 capital by January 2027, including options trading foundation.

**Architecture:** 10 sequential phases with selective parallelism. Each phase produces testable, committable software with an explicit quality gate. The pattern engine's 5-stage matcher feeds through signal filters → position sizer → risk engine → portfolio manager → broker adapter. All communication flows through frozen Pydantic SharedState.

**Tech Stack:** Python 3.12, pytest, Pydantic v2, scikit-learn (BallTree), hnswlib, numpy, pandas, scipy (Platt sigmoid), matplotlib (diagnostics). Broker: IBKR TWS API (Alpaca fallback).

**Spec:** `docs/superpowers/specs/2026-03-26-fppe-full-roadmap-design.md`

**Test command:** `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`

**Critical rules:**
- `assert` → `RuntimeError` for all public API guards
- `nn_jobs=1` always (Windows/Py3.12 joblib deadlock)
- Numbers require provenance — any claimed metric must trace to walk-forward results
- 3-strike rule: if 3 consecutive attempts at same fix fail, STOP and escalate

---

## Execution Contract

Before coding starts, all implementers must know these integration points:

**Walk-forward runner entrypoint:**
- `pattern_engine/matcher.py` → `PatternMatcher.walk_forward()` is the canonical runner
- Input: feature DataFrame from `prepare.py` (DO NOT modify `prepare.py`)
- Fold artifacts: `results/fold_<N>/` directories contain per-fold predictions, metrics
- Target column: `fwd_7d_up` (binary, 1 = price up after 7 trading days). **Not** `y_up_7d`.

**Expected input artifacts for Phase 1 diagnostics:**
- Walk-forward fold results: `results/walkforward_results.tsv` (or fold-level CSVs in `results/`)
- Each row has: `ticker`, `date`, `fold`, `predicted_prob`, `actual` (0/1), `distance_to_nearest`
- If fold artifacts don't exist in this format, the first diagnostic task must run a walk-forward to generate them

**Results directory layout:**
- `results/` — all experiment outputs with provenance
- `results/fold_<N>/` — per-fold artifacts
- `results/<experiment_name>_<date>.tsv` — sweep/experiment results

**Locked settings source of truth:** `CLAUDE.md` → "Locked Settings" section. Any change requires new experiment evidence logged in `results/`.

**Memory/infrastructure budgets (hard gates):**
- Peak RAM during HNSW index build: < 24 GB (32 GB machine, 8 GB headroom)
- Overnight build pipeline: < 2 hours wall-clock
- LiveRunner 4:00 PM execution: < 3 minutes with pre-built index

---

## File Structure Map

### New Files (Created Across All Phases)

```
# Phase 1 — BSS Diagnosis
scripts/reliability_diagram.py          # Calibration visualization
scripts/base_rate_analysis.py           # Per-fold base rate decomposition
scripts/distance_distribution.py        # Neighbor distance analysis
scripts/bss_sweep.py                    # Hypothesis testing sweep runner

# Phase 2 — Half-Kelly
trading_system/position_sizer.py        # Kelly fraction computation + sizing
tests/unit/test_position_sizer.py       # Full TDD coverage

# Phase 3 — Risk Engine
trading_system/risk_engine.py           # Production risk engine (wires overlays)
tests/unit/test_risk_engine.py          # Drawdown brake, ATR sizing, overlay integration
tests/integration/test_risk_pipeline.py # End-to-end: signal → risk → sized position

# Phase 4 — Portfolio Manager
tests/integration/test_portfolio_pipeline.py  # PM + risk engine integration on 585T

# Phase 5 — Live Plumbing
trading_system/broker/__init__.py
trading_system/broker/base.py           # BaseBroker ABC (extends existing BaseBrokerAdapter)
trading_system/broker/mock.py           # Enhanced MockBrokerAdapter v2
trading_system/broker/ibkr.py           # IBKR TWS adapter
trading_system/order_manager.py         # Order lifecycle management
scripts/reconcile.py                    # Daily OOB reconciliation
tests/unit/test_broker_mock.py          # Mock broker scenarios
tests/unit/test_order_manager.py        # Order state machine
tests/unit/test_reconciliation.py       # Reconciliation logic

# Phase 6 — Universe Expansion
scripts/build_overnight_index.py        # Overnight HNSW index builder
scripts/validate_sector_map.py          # Sector mapping validation
tests/performance/test_hnsw_1500t.py    # Scaling benchmarks

# Phase 7 — Model Enhancements
pattern_engine/anomaly_filter.py        # CPOD/EILOF pre-filter
tests/unit/test_bma_integration.py      # BMA calibrator in pipeline
tests/unit/test_conformal.py            # Conformal prediction coverage
tests/unit/test_anomaly_filter.py       # Anomaly detection

# Phase 8 — Paper Trading
scripts/daily_report.py                 # Automated daily metrics
scripts/weekly_review.py                # Weekly performance summary

# Phase 9 — Live Deploy
trading_system/tax_tracker.py           # Capital gains + wash sale tracking
scripts/emergency_halt.py               # Kill switch
tests/unit/test_tax_tracker.py          # Tax lot, wash sale, FIFO
docs/DEPLOYMENT_RUNBOOK.md              # Production deployment guide

# Phase 10 — Options Foundation
trading_system/options/__init__.py
trading_system/options/contracts.py     # OptionContract, OptionQuote, OptionOrder, OptionPosition
trading_system/options/strategies.py    # BaseOptionStrategy ABC + stubs
trading_system/options/data.py          # BaseOptionsDataProvider ABC
trading_system/options/greeks.py        # Black-Scholes, binomial tree, IV solver
trading_system/options/signal_bridge.py # Signal-to-options mapping
tests/unit/test_options_contracts.py
tests/unit/test_greeks.py
tests/unit/test_signal_bridge.py
docs/OPTIONS_TRADING_DESIGN.md
```

### Existing Files Modified

```
# Phase 1
pattern_engine/matcher.py:315           # _package_results: Platt refit logic
CLAUDE.md                               # Update locked settings if max_distance/top_k changes

# Phase 2
trading_system/config.py                # Add KellyConfig parameters

# Phase 3
trading_system/config.py                # Add RiskEngineConfig, activate overlay flags
trading_system/risk_overlays/__init__.py # Export production overlay instances

# Phase 4
trading_system/config.py                # Set use_portfolio_manager=True

# Phase 5
pattern_engine/live.py:140              # LiveRunner hardening
trading_system/contracts/state.py       # Add reconciliation fields if needed

# Phase 6
pattern_engine/features.py              # Validate on expanded universe
trading_system/config.py                # Expanded SECTOR_MAP

# Phase 7
pattern_engine/matcher.py:144           # _prepare_features: OWA weight support
pattern_engine/signal_pipeline.py:35    # Register anomaly filter
pattern_engine/conformal_hooks.py       # Wire into post-calibration
trading_system/config.py                # Enhancement feature flags

# Phase 10
trading_system/contracts/state.py       # Add OptionsState
```

---

## Phase 1: BSS Diagnosis & Calibration Fix

**Gate:** BSS > 0 on ≥ 3/6 walk-forward folds
**Duration:** 3-4 weeks

### Task 1.1: Reliability Diagram Diagnostic

**Files:**
- Create: `scripts/reliability_diagram.py`

- [ ] **Step 1: Write reliability diagram script**

```python
"""Reliability diagram: predicted probability vs actual hit rate.

Usage: PYTHONUTF8=1 py -3.12 scripts/reliability_diagram.py
Reads walk-forward results and plots calibration curve per fold.
Output: results/reliability_diagram.png + results/reliability_data.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def compute_reliability(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10):
    """Bin predictions into equal-frequency bins, compute actual hit rate per bin.

    Returns:
        bin_centers: Mean predicted probability per bin.
        bin_actuals: Actual hit rate per bin.
        bin_counts: Number of samples per bin.
    """
    # Equal-frequency binning (quantile-based)
    bin_edges = np.quantile(predicted, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8  # include rightmost edge
    bin_centers, bin_actuals, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(predicted[mask].mean())
        bin_actuals.append(actual[mask].mean())
        bin_counts.append(mask.sum())
    return np.array(bin_centers), np.array(bin_actuals), np.array(bin_counts)


def plot_reliability(fold_data: dict, output_path: Path):
    """Plot reliability diagrams for all folds."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (fold_name, (predicted, actual)) in enumerate(fold_data.items()):
        centers, actuals, counts = compute_reliability(predicted, actual)
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.scatter(centers, actuals, s=counts / 10, alpha=0.7)
        ax.plot(centers, actuals, "b-o", markersize=4)
        ax.set_title(f"Fold: {fold_name}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual hit rate")
        ax.set_xlim(0.45, 0.65)
        ax.set_ylim(0.35, 0.65)
        ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Placeholder — integrate with actual walk-forward runner
    print("Run walk-forward first, then point this at the fold results.")
    print("Expected input: per-fold predicted_prob and y_true arrays.")
```

- [ ] **Step 2: Run to verify script loads cleanly**

Run: `PYTHONUTF8=1 py -3.12 scripts/reliability_diagram.py`
Expected: prints usage instructions without error

- [ ] **Step 3: Commit**

```bash
git add scripts/reliability_diagram.py
git commit -m "feat(diag): add reliability diagram script for BSS calibration analysis"
```

### Task 1.2: Base-Rate Decomposition

**Files:**
- Create: `scripts/base_rate_analysis.py`

- [ ] **Step 1: Write base-rate analysis script**

```python
"""Base-rate decomposition: compare per-fold base rates at 585T vs 52T.

Computes: % of rows where fwd_7d_up=1 per fold, per universe size.
Output: results/base_rate_analysis.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path


def compute_base_rates(val_db: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    """Compute base rate (% positive) per fold.

    Args:
        val_db: Validation DataFrame with 'fwd_7d_up' target column.
        fold_col: Column identifying the fold.

    Returns:
        DataFrame with columns: fold, n_rows, n_positive, base_rate.
    """
    rows = []
    for fold_name, group in val_db.groupby(fold_col):
        n = len(group)
        n_pos = (group["fwd_7d_up"] == 1).sum()
        rows.append({
            "fold": fold_name,
            "n_rows": n,
            "n_positive": n_pos,
            "base_rate": n_pos / n if n > 0 else 0.0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Base rate analysis: integrate with walk-forward data loader.")
    print("Compare 585T base rates to 52T baseline (expected ~50%).")
```

- [ ] **Step 2: Run to verify**

Run: `PYTHONUTF8=1 py -3.12 scripts/base_rate_analysis.py`
Expected: prints usage instructions

- [ ] **Step 3: Commit**

```bash
git add scripts/base_rate_analysis.py
git commit -m "feat(diag): add base-rate decomposition for fold-level analysis"
```

### Task 1.3: Distance Distribution Analysis

**Files:**
- Create: `scripts/distance_distribution.py`

- [ ] **Step 1: Write distance distribution script**

```python
"""Distance distribution analysis: compare neighbor distances at 585T vs 52T.

If 585T pushes mean distances up, KNN signal weakens.
Output: results/distance_distribution.png + results/distance_stats.csv
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_distances(distances: np.ndarray, fold_name: str) -> dict:
    """Compute distance statistics for a fold.

    Args:
        distances: Shape (n_queries, top_k) — neighbor distances.
        fold_name: Fold identifier.

    Returns:
        Dict with mean, median, p25, p75, p95 distance stats.
    """
    flat = distances.flatten()
    return {
        "fold": fold_name,
        "mean_dist": float(np.mean(flat)),
        "median_dist": float(np.median(flat)),
        "p25_dist": float(np.percentile(flat, 25)),
        "p75_dist": float(np.percentile(flat, 75)),
        "p95_dist": float(np.percentile(flat, 95)),
        "pct_above_max_distance": float((flat > 1.1019).mean()),
    }


if __name__ == "__main__":
    print("Distance distribution: integrate with PatternMatcher query output.")
    print("Compare 585T distributions to 52T baseline.")
```

- [ ] **Step 2: Run to verify**

Run: `PYTHONUTF8=1 py -3.12 scripts/distance_distribution.py`
Expected: prints usage instructions

- [ ] **Step 3: Commit**

```bash
git add scripts/distance_distribution.py
git commit -m "feat(diag): add distance distribution analysis for pool dilution"
```

### Task 1.4: BSS Hypothesis Sweep Runner

**Files:**
- Create: `scripts/bss_sweep.py`

- [ ] **Step 1: Write hypothesis sweep script**

This is the core experiment runner. It tests H1 (max_distance), H2 (same_sector_only), H3 (top_k), H4 (Platt refit) sequentially.

```python
"""BSS hypothesis sweep: test H1-H4 to find BSS fix.

H1: Tighten max_distance [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
H2: same_sector_only=True
H3: Reduce top_k [10, 20, 30, 40, 50]
H4: Refit Platt on 585T distribution

Gate: BSS > 0 on >= 3/6 folds.
Output: results/bss_sweep_results.tsv

IMPORTANT: Run hypotheses in order. Stop at first hypothesis that passes gate.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_hypothesis_h1(max_distances: list[float]):
    """H1: Sweep max_distance values."""
    print("H1: Sweeping max_distance...")
    for md in max_distances:
        print(f"  Testing max_distance={md}")
        # Integration point: call walk-forward runner with overridden max_distance
        # Record: fold BSS values, mean BSS, positive fold count
    print("  → Log results to results/bss_sweep_h1.tsv")


def run_hypothesis_h2():
    """H2: Enable same_sector_only filtering."""
    print("H2: Testing same_sector_only=True...")
    # Integration point: call walk-forward with same_sector_only=True
    print("  → Log results to results/bss_sweep_h2.tsv")


def run_hypothesis_h3(top_k_values: list[int]):
    """H3: Sweep top_k values."""
    print("H3: Sweeping top_k...")
    for k in top_k_values:
        print(f"  Testing top_k={k}")
        # Integration point: call walk-forward with overridden top_k
    print("  → Log results to results/bss_sweep_h3.tsv")


def run_hypothesis_h4():
    """H4: Force Platt recalibration on 585T."""
    print("H4: Refitting Platt on 585T distribution...")
    # Integration point: recalibrate Platt, then run walk-forward
    print("  → Log results to results/bss_sweep_h4.tsv")


def check_gate(results_path: Path) -> bool:
    """Check if BSS > 0 on >= 3/6 folds."""
    # Read results TSV, count positive folds
    return False  # placeholder


if __name__ == "__main__":
    print("BSS Hypothesis Sweep")
    print("=" * 50)
    print("Run hypotheses in order. Stop at first gate pass.")
    print()

    # H1
    run_hypothesis_h1([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # H2
    run_hypothesis_h2()

    # H3
    run_hypothesis_h3([10, 20, 30, 40, 50])

    # H4
    run_hypothesis_h4()

    print("\nDone. Check results/ for outputs.")
```

- [ ] **Step 2: Run to verify**

Run: `PYTHONUTF8=1 py -3.12 scripts/bss_sweep.py`
Expected: prints hypothesis names and placeholders

- [ ] **Step 3: Commit**

```bash
git add scripts/bss_sweep.py
git commit -m "feat(diag): add BSS hypothesis sweep runner (H1-H4)"
```

### Task 1.5: Integrate Diagnostics with Walk-Forward Runner

**Files:**
- Modify: `scripts/bss_sweep.py`
- Reference: `pattern_engine/matcher.py` (PatternMatcher), existing walk-forward scripts

- [ ] **Step 1: Wire bss_sweep.py to actual walk-forward runner**

Connect the hypothesis functions to the real `PatternMatcher` and walk-forward data splits. This is the main integration work — read the existing walk-forward script (likely in `scripts/`) and replicate its data loading + fold splitting, then override parameters per hypothesis.

- [ ] **Step 2: Run H1 sweep (max_distance)**

Run: `PYTHONUTF8=1 py -3.12 scripts/bss_sweep.py --hypothesis h1`
Expected: TSV file with 6 rows per max_distance value, BSS per fold

- [ ] **Step 3: Evaluate H1 results — does any max_distance produce BSS > 0 on ≥ 3/6 folds?**

If YES → update `CLAUDE.md` locked settings with new max_distance, commit, proceed to Phase 2.
If NO → continue to H2.

- [ ] **Step 4: Run remaining hypotheses as needed (H2, H3, H4)**

Stop at first hypothesis that passes gate.

- [ ] **Step 5: Commit results with provenance**

```bash
git add results/bss_sweep_*.tsv CLAUDE.md
git commit -m "fix(bss): [HYPOTHESIS] resolves BSS — [NEW_VALUE], [N]/6 positive folds"
```

### Task 1.6: Post-Fix Filter Validation

**Files:**
- Reference: `pattern_engine/sector_conviction.py`, `pattern_engine/momentum_signal.py`

- [ ] **Step 1: Re-run SectorConviction experiment with min_sector_lift=0.005 on fixed baseline**

Verify filter now allows 5-15% signal throughput (was 0% at old 0.03 value).

- [ ] **Step 2: Re-run Momentum experiment on fixed baseline**

Document AccFilt impact (previously mixed: -1.4pp to +2.9pp).

- [ ] **Step 3: Log results in session log**

- [ ] **Step 4: Run full test suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: 616+ tests pass

- [ ] **Step 5: Commit**

```bash
git add results/ docs/session-logs/
git commit -m "docs(bss): post-fix filter validation — sector conviction and momentum"
```

### GATE CHECK: Phase 1

- [ ] **BSS > 0 on ≥ 3/6 walk-forward folds?**
  - YES → Proceed to Phase 2
  - NO (after all H1-H4) → Fallback: consider 52T universe or BMA rescue path. STOP and escalate per 3-strike rule.

---

## Phase 2: Half-Kelly Position Sizer (M10)

**Gate:** Kelly fraction positive on ≥ 4/6 folds, Sharpe ≥ 1.0, Max DD ≤ 15%
**Duration:** 2 weeks
**Prerequisite:** Phase 1 gate passed

### Task 2.1: Kelly Fraction Core — Failing Tests

**Files:**
- Create: `tests/unit/test_position_sizer.py`
- Create: `trading_system/position_sizer.py`

- [ ] **Step 1: Write failing tests for kelly_fraction computation**

```python
"""Tests for trading_system/position_sizer.py — Half-Kelly position sizing."""
import pytest
from trading_system.position_sizer import compute_kelly_fraction, HalfKellySizer


class TestComputeKellyFraction:
    """Test the raw Kelly formula: (p*b - q) / b."""

    def test_positive_edge(self):
        """60% win rate, 1.5 avg_win/avg_loss → positive fraction."""
        fraction = compute_kelly_fraction(p=0.60, b=1.5)
        # (0.6 * 1.5 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.333...
        assert abs(fraction - 1 / 3) < 1e-6

    def test_zero_edge(self):
        """50% win rate, 1.0 payoff → zero fraction (no edge)."""
        fraction = compute_kelly_fraction(p=0.50, b=1.0)
        assert fraction == 0.0

    def test_negative_edge(self):
        """40% win rate, 1.0 payoff → negative fraction."""
        fraction = compute_kelly_fraction(p=0.40, b=1.0)
        assert fraction < 0.0

    def test_p_out_of_range_raises(self):
        """p must be in [0, 1]."""
        with pytest.raises(RuntimeError):
            compute_kelly_fraction(p=1.5, b=1.0)

    def test_b_zero_raises(self):
        """b (avg_win/avg_loss) must be positive."""
        with pytest.raises(RuntimeError):
            compute_kelly_fraction(p=0.6, b=0.0)


class TestHalfKellySizer:
    """Test the full sizing pipeline: atr_weight × half_kelly × equity."""

    def test_basic_sizing(self):
        """Known inputs produce expected position size."""
        sizer = HalfKellySizer(min_position_pct=0.02, max_position_pct=0.10)
        size = sizer.compute_position_size(
            atr_weight=0.05,         # flat constant for Phase 2
            win_prob=0.60,
            avg_win_loss_ratio=1.5,
            current_equity=10_000.0,
        )
        # atr_weight=0.05, half_kelly=0.5*(1/3)=0.1667, equity=10000
        # raw = 0.05 * 0.1667 * 10000 = 83.33
        # as pct = 83.33 / 10000 = 0.83% → clamped to min 2% = 200.0
        assert size == pytest.approx(200.0, rel=0.01)

    def test_clamp_max(self):
        """Position size clamped at max_position_pct."""
        sizer = HalfKellySizer(min_position_pct=0.02, max_position_pct=0.10)
        size = sizer.compute_position_size(
            atr_weight=1.0,           # very large ATR weight
            win_prob=0.80,
            avg_win_loss_ratio=3.0,
            current_equity=10_000.0,
        )
        assert size <= 10_000.0 * 0.10  # max 10%

    def test_negative_kelly_returns_zero(self):
        """Negative Kelly fraction → zero position (no trade)."""
        sizer = HalfKellySizer(min_position_pct=0.02, max_position_pct=0.10)
        size = sizer.compute_position_size(
            atr_weight=0.05,
            win_prob=0.40,
            avg_win_loss_ratio=1.0,
            current_equity=10_000.0,
        )
        assert size == 0.0

    def test_flat_atr_weight_for_phase2(self):
        """Phase 2 uses flat atr_weight before risk engine is wired."""
        sizer = HalfKellySizer(
            min_position_pct=0.02,
            max_position_pct=0.10,
            default_atr_weight=0.05,  # flat constant
        )
        size = sizer.compute_position_size(
            atr_weight=None,  # uses default
            win_prob=0.60,
            avg_win_loss_ratio=1.5,
            current_equity=10_000.0,
        )
        assert size > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_position_sizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading_system.position_sizer'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/unit/test_position_sizer.py
git commit -m "test(kelly): add failing tests for Half-Kelly position sizer"
```

### Task 2.2: Kelly Fraction Core — Implementation

**Files:**
- Create: `trading_system/position_sizer.py`

- [ ] **Step 1: Implement position_sizer.py**

```python
"""position_sizer.py — Half-Kelly position sizing module.

Sits between Layer 2 (risk engine) and Layer 3 (portfolio manager).
Phase 2: atr_weight is a flat constant.
Phase 3: atr_weight comes from real ATR computation in risk engine.

Combined formula:
    position_size = atr_weight × half_kelly × current_equity
    position_size = clamp(position_size, min_pct * equity, max_pct * equity)
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class KellyConfig(BaseModel):
    """Configuration for Half-Kelly position sizer."""
    model_config = {"frozen": True}

    min_position_pct: float = Field(default=0.02, ge=0.0, le=1.0)
    max_position_pct: float = Field(default=0.10, ge=0.0, le=1.0)
    kelly_fraction_multiplier: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="0.5 = half-Kelly, 0.25 = quarter-Kelly",
    )
    default_atr_weight: float = Field(
        default=0.05, ge=0.0,
        description="Flat ATR weight used before risk engine is wired (Phase 2)",
    )


def compute_kelly_fraction(p: float, b: float) -> float:
    """Compute raw Kelly fraction: (p*b - q) / b.

    Args:
        p: Win probability [0, 1].
        b: Average win / average loss ratio (must be > 0).

    Returns:
        Raw Kelly fraction (can be negative if no edge).

    Raises:
        RuntimeError: If p not in [0,1] or b <= 0.
    """
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"Win probability must be in [0, 1], got {p}")
    if b <= 0.0:
        raise RuntimeError(f"Avg win/loss ratio must be > 0, got {b}")
    q = 1.0 - p
    return (p * b - q) / b


class HalfKellySizer:
    """Half-Kelly position sizer with safety clamps.

    Args:
        min_position_pct: Minimum position as fraction of equity.
        max_position_pct: Maximum position as fraction of equity.
        kelly_multiplier: Fraction of full Kelly to use (0.5 = half).
        default_atr_weight: Flat ATR weight for Phase 2 (before risk engine).
    """

    def __init__(
        self,
        min_position_pct: float = 0.02,
        max_position_pct: float = 0.10,
        kelly_multiplier: float = 0.5,
        default_atr_weight: float = 0.05,
    ) -> None:
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct
        self.kelly_multiplier = kelly_multiplier
        self.default_atr_weight = default_atr_weight

    def compute_position_size(
        self,
        win_prob: float,
        avg_win_loss_ratio: float,
        current_equity: float,
        atr_weight: Optional[float] = None,
    ) -> float:
        """Compute position size in dollars.

        Args:
            win_prob: Calibrated win probability from Platt.
            avg_win_loss_ratio: avg_win / avg_loss from walk-forward.
            current_equity: Current portfolio equity in dollars.
            atr_weight: ATR-based weight from risk engine.
                        None → use default_atr_weight (Phase 2 mode).

        Returns:
            Position size in dollars. 0.0 if Kelly fraction is negative.
        """
        kelly = compute_kelly_fraction(win_prob, avg_win_loss_ratio)
        if kelly <= 0.0:
            return 0.0

        half_kelly = self.kelly_multiplier * kelly
        effective_atr_weight = atr_weight if atr_weight is not None else self.default_atr_weight
        raw_size = effective_atr_weight * half_kelly * current_equity

        # Clamp to position limits
        min_size = self.min_position_pct * current_equity
        max_size = self.max_position_pct * current_equity
        if raw_size <= 0:
            return 0.0
        return max(min(raw_size, max_size), min_size)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_position_sizer.py -v`
Expected: all tests PASS

- [ ] **Step 3: Run full suite to verify no regressions**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: 616+ tests pass

- [ ] **Step 4: Commit**

```bash
git add trading_system/position_sizer.py
git commit -m "feat(kelly): implement Half-Kelly position sizer with safety clamps"
```

### Task 2.3: Kelly Config Integration

**Files:**
- Modify: `trading_system/config.py`

- [ ] **Step 1: Add KellyConfig to TradingConfig**

Add `kelly: KellyConfig = KellyConfig()` field to the existing TradingConfig dataclass.

- [ ] **Step 2: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add trading_system/config.py
git commit -m "feat(config): add KellyConfig to TradingConfig"
```

### Task 2.4: Walk-Forward Validation with Kelly Sizing

**Files:**
- Reference: walk-forward runner scripts, `results/`

- [ ] **Step 1: Run walk-forward with Kelly sizing enabled**

Using the BSS-fixed baseline from Phase 1, run full 6-fold walk-forward with Half-Kelly sizing. Record Sharpe, max DD, and Kelly fraction per fold.

- [ ] **Step 2: Compare Kelly vs flat sizing results**

Create `results/kelly_comparison.tsv` with columns: fold, flat_sharpe, kelly_sharpe, flat_maxdd, kelly_maxdd, kelly_fraction.

- [ ] **Step 3: Verify gate criteria**

Check: Kelly fraction positive on ≥ 4/6 folds, Sharpe ≥ 1.0, Max DD ≤ 15%.

- [ ] **Step 4: Commit results**

```bash
git add results/kelly_comparison.tsv
git commit -m "results(kelly): walk-forward comparison — Kelly vs flat sizing"
```

### GATE CHECK: Phase 2

- [ ] **Kelly fraction positive on ≥ 4/6 folds? Sharpe ≥ 1.0? Max DD ≤ 15%?**
  - YES → Proceed to Phase 3
  - NO → Fallback: try quarter-Kelly (0.25×). If still fails, revert to ATR-only sizing and proceed.

---

## Phase 3: Risk Engine Integration

**Gate:** Drawdown brake fires correctly, Max DD ≤ 10%, Sharpe ≥ 1.0, stops ≤ 35%
**Duration:** 3 weeks
**Prerequisite:** Phase 2 gate passed

### Task 3.1: Risk Engine — Failing Tests

**Files:**
- Create: `tests/unit/test_risk_engine.py`
- Create: `trading_system/risk_engine.py`

- [ ] **Step 1: Write failing tests for risk engine**

```python
"""Tests for trading_system/risk_engine.py — production risk engine."""
import pytest
from trading_system.risk_engine import RiskEngine, RiskEngineConfig


class TestATRPositionSizing:
    """ATR-based position weight: max_loss_pct / (atr_multiple * atr_pct)."""

    def test_basic_atr_weight(self):
        """2% risk, 3.0× ATR, 1% ATR → weight = 0.02 / 0.03 = 0.667."""
        engine = RiskEngine(RiskEngineConfig())
        weight = engine.compute_atr_weight(atr_pct=0.01)
        assert abs(weight - 2 / 3) < 0.01

    def test_high_volatility_reduces_weight(self):
        """Higher ATR% → smaller position weight."""
        engine = RiskEngine(RiskEngineConfig())
        w_low = engine.compute_atr_weight(atr_pct=0.01)
        w_high = engine.compute_atr_weight(atr_pct=0.05)
        assert w_high < w_low


class TestDrawdownBrake:
    """Linear scalar: 1.0 at 0% DD → 0.0 at halt threshold."""

    def test_no_drawdown(self):
        engine = RiskEngine(RiskEngineConfig())
        assert engine.drawdown_scalar(drawdown_pct=0.0) == 1.0

    def test_at_brake_threshold(self):
        """At 15% DD (brake threshold), scalar starts reducing."""
        engine = RiskEngine(RiskEngineConfig(
            drawdown_brake_pct=0.15, drawdown_halt_pct=0.20,
        ))
        scalar = engine.drawdown_scalar(drawdown_pct=0.15)
        assert scalar == 1.0  # just at threshold, not yet reduced

    def test_midpoint_drawdown(self):
        """At 17.5% DD (midpoint between 15% and 20%), scalar = 0.5."""
        engine = RiskEngine(RiskEngineConfig(
            drawdown_brake_pct=0.15, drawdown_halt_pct=0.20,
        ))
        scalar = engine.drawdown_scalar(drawdown_pct=0.175)
        assert abs(scalar - 0.5) < 0.01

    def test_at_halt_threshold(self):
        """At 20% DD, scalar = 0.0 (full halt)."""
        engine = RiskEngine(RiskEngineConfig(
            drawdown_brake_pct=0.15, drawdown_halt_pct=0.20,
        ))
        scalar = engine.drawdown_scalar(drawdown_pct=0.20)
        assert scalar == 0.0

    def test_beyond_halt(self):
        """Past 20% DD, scalar stays 0.0."""
        engine = RiskEngine(RiskEngineConfig(
            drawdown_brake_pct=0.15, drawdown_halt_pct=0.20,
        ))
        scalar = engine.drawdown_scalar(drawdown_pct=0.25)
        assert scalar == 0.0


class TestStopLossEvaluation:
    """Stop-loss: breach check against intraday low."""

    def test_stop_not_breached(self):
        engine = RiskEngine(RiskEngineConfig())
        breached = engine.check_stop_breach(
            stop_price=95.0, intraday_low=96.0
        )
        assert breached is False

    def test_stop_breached(self):
        engine = RiskEngine(RiskEngineConfig())
        breached = engine.check_stop_breach(
            stop_price=95.0, intraday_low=94.0
        )
        assert breached is True

    def test_stop_exact_touch(self):
        """Exact touch of stop price = breached."""
        engine = RiskEngine(RiskEngineConfig())
        breached = engine.check_stop_breach(
            stop_price=95.0, intraday_low=95.0
        )
        assert breached is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_risk_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/unit/test_risk_engine.py
git commit -m "test(risk): add failing tests for risk engine — ATR sizing, drawdown brake, stops"
```

### Task 3.2: Risk Engine — Implementation

**Files:**
- Create: `trading_system/risk_engine.py`

- [ ] **Step 1: Implement risk engine**

```python
"""risk_engine.py — Production risk engine (Layer 2).

Computes ATR-based position weights, drawdown brake scaling,
and stop-loss evaluation. Wires FatigueAccumulation and
LiquidityCongestion overlays when enabled.

Integration with position_sizer.py:
    atr_weight = risk_engine.compute_atr_weight(atr_pct)
    position_size = sizer.compute_position_size(
        atr_weight=atr_weight * drawdown_scalar,
        win_prob=..., avg_win_loss_ratio=..., current_equity=...
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field


class RiskEngineConfig(BaseModel):
    """Risk engine configuration."""
    model_config = {"frozen": True}

    stop_loss_atr_multiple: float = Field(default=3.0, description="LOCKED: swept 2.0-4.0, 3.0× won")
    max_loss_per_trade_pct: float = Field(default=0.02, description="2% equity risk per trade")
    drawdown_brake_pct: float = Field(default=0.15, description="Start reducing at 15% DD")
    drawdown_halt_pct: float = Field(default=0.20, description="Full halt at 20% DD")
    min_position_pct: float = Field(default=0.02)
    max_position_pct: float = Field(default=0.10)


class RiskEngine:
    """Production risk engine.

    Responsibilities:
        1. ATR-based position weight computation
        2. Drawdown brake (linear scalar)
        3. Stop-loss breach evaluation
        4. Optional overlay integration (fatigue, congestion)
    """

    def __init__(self, config: RiskEngineConfig) -> None:
        self.config = config
        self._overlays: list = []  # BaseRiskOverlay instances

    def compute_atr_weight(self, atr_pct: float) -> float:
        """Compute position weight from ATR.

        Formula: max_loss_pct / (atr_multiple * atr_pct)

        Args:
            atr_pct: 20-day ATR as fraction of close price.

        Returns:
            Raw position weight (before drawdown scalar and Kelly).
        """
        if atr_pct <= 0:
            raise RuntimeError(f"ATR% must be positive, got {atr_pct}")
        stop_distance = self.config.stop_loss_atr_multiple * atr_pct
        return self.config.max_loss_per_trade_pct / stop_distance

    def drawdown_scalar(self, drawdown_pct: float) -> float:
        """Compute drawdown brake scalar.

        Linear interpolation: 1.0 at brake_pct → 0.0 at halt_pct.
        Below brake_pct: 1.0. Above halt_pct: 0.0.

        Args:
            drawdown_pct: Current peak-to-trough drawdown as fraction.

        Returns:
            Scalar in [0.0, 1.0].
        """
        if drawdown_pct <= self.config.drawdown_brake_pct:
            return 1.0
        if drawdown_pct >= self.config.drawdown_halt_pct:
            return 0.0
        # Linear interpolation
        span = self.config.drawdown_halt_pct - self.config.drawdown_brake_pct
        progress = (drawdown_pct - self.config.drawdown_brake_pct) / span
        return 1.0 - progress

    def check_stop_breach(self, stop_price: float, intraday_low: float) -> bool:
        """Check if stop-loss was breached.

        Args:
            stop_price: Stop-loss trigger price.
            intraday_low: Lowest price during the trading day.

        Returns:
            True if stop was breached (intraday_low <= stop_price).
        """
        return intraday_low <= stop_price

    def register_overlay(self, overlay) -> None:
        """Register a risk overlay (fatigue, congestion, etc.)."""
        self._overlays.append(overlay)

    def get_overlay_multiplier(self) -> float:
        """Get combined multiplier from all registered overlays."""
        multiplier = 1.0
        for overlay in self._overlays:
            multiplier *= overlay.get_signal_multiplier()
        return multiplier
```

- [ ] **Step 2: Run tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_risk_engine.py -v`
Expected: all PASS

- [ ] **Step 3: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add trading_system/risk_engine.py
git commit -m "feat(risk): implement production risk engine — ATR sizing, drawdown brake, stops"
```

### Task 3.3: Overlay Activation & Integration Tests

**Files:**
- Modify: `trading_system/config.py` — activate overlay flags
- Create: `tests/integration/test_risk_pipeline.py`

- [ ] **Step 1: Write integration tests for risk pipeline**

Test the full flow: signal → risk engine (ATR weight + drawdown scalar) → position sizer (Kelly) → final position size. Include overlay integration.

- [ ] **Step 2: Activate overlays in config**

Set `use_fatigue_accumulation=True` and `use_liquidity_congestion_gate=True` in `ResearchFlagsConfig` defaults.

- [ ] **Step 3: Run integration tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/integration/test_risk_pipeline.py -v`
Expected: all PASS

- [ ] **Step 4: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add trading_system/config.py tests/integration/test_risk_pipeline.py
git commit -m "feat(risk): activate fatigue + congestion overlays, add integration tests"
```

### Task 3.4: Walk-Forward with Risk Engine

- [ ] **Step 1: Run walk-forward with full risk engine (ATR + Kelly + overlays)**
- [ ] **Step 2: Compare to Phase 2 results**
- [ ] **Step 3: Verify gate: Max DD ≤ 10%, Sharpe ≥ 1.0, stops ≤ 35% of exits**
- [ ] **Step 4: If overlays degrade: disable and retest (fallback)**
- [ ] **Step 5: Commit results**

```bash
git add results/
git commit -m "results(risk): walk-forward with risk engine — [SHARPE], [MAXDD], [STOP_PCT]"
```

### GATE CHECK: Phase 3

- [ ] **Drawdown brake fires correctly? Max DD ≤ 10%? Sharpe ≥ 1.0? Stops ≤ 35%?**
  - YES → Proceed to Phase 4
  - NO → Disable overlays, keep core ATR + brake. Proceed with known-good primitives.

---

## Phase 4: Portfolio Manager Activation

**Gate:** Sector limits enforced, idle cash < 50%, Sharpe ≥ 1.0
**Duration:** 3 weeks
**Prerequisite:** Phase 3 gate passed

### Task 4.1: Portfolio Manager 585T Validation Tests

**Files:**
- Create: `tests/integration/test_portfolio_pipeline.py`

- [ ] **Step 1: Write integration tests for PM on 585T universe**

Test: sector limits (max 30%, max 3 per sector), signal ranking by confidence, capital utilization tracking, rejection logging.

- [ ] **Step 2: Run to verify they fail (PM not yet activated)**

- [ ] **Step 3: Commit failing tests**

### Task 4.2: Activate Portfolio Manager

**Files:**
- Modify: `trading_system/config.py` — set `use_portfolio_manager=True`

- [ ] **Step 1: Set `use_portfolio_manager=True` in default config**
- [ ] **Step 2: Run integration tests**
- [ ] **Step 3: Run full suite**
- [ ] **Step 4: Commit**

### Task 4.3: PM Rejection Analysis Script

**Files:**
- Create: `scripts/pm_rejection_analysis.py`

- [ ] **Step 1: Write script to analyze PM rejection reasons**

Reads rejection log, outputs breakdown: how many blocked by sector, by capital, by cooldown.

- [ ] **Step 2: Run on walk-forward results**
- [ ] **Step 3: Commit**

### Task 4.4: Walk-Forward with Full Pipeline

- [ ] **Step 1: Run walk-forward: matcher → filters → Kelly → risk → PM**
- [ ] **Step 2: Verify sector limits enforced (no sector > 30%)**
- [ ] **Step 3: Verify idle cash < 50%**
- [ ] **Step 4: Verify Sharpe ≥ 1.0**
- [ ] **Step 5: Commit results**

### GATE CHECK: Phase 4

- [ ] **Sector limits enforced? Idle cash < 50%? Sharpe ≥ 1.0?**
  - YES → Proceed to Phase 5 + 6 (parallel)
  - NO → Simplify to sector-limit-only mode. Proceed.

---

## Phase 5: Live Execution Plumbing (Can Start Parallel with Phase 4)

**Gate:** Mock broker parity for 100 trades, OOB reconciliation 30 days, pipeline < 3 min
**Duration:** 4 weeks
**Prerequisite:** Phase 3 gate passed

### Task 5.1: BaseBroker ABC

**Files:**
- Create: `trading_system/broker/__init__.py`
- Create: `trading_system/broker/base.py`

- [ ] **Step 1: Create broker package with ABC**

```python
"""base.py — BaseBroker ABC for broker adapters.

All broker implementations (mock, IBKR, Alpaca) implement this interface.
Extends the existing BaseBrokerAdapter from pattern_engine/live.py with
additional methods needed for production use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Order(BaseModel):
    """Order to submit to broker."""
    model_config = {"frozen": True}

    ticker: str
    direction: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class OrderResult(BaseModel):
    """Result from broker after order submission."""
    model_config = {"frozen": True}

    order_id: str
    ticker: str
    filled_quantity: float
    fill_price: float
    status: str  # "FILLED", "PARTIAL", "REJECTED", "CANCELLED"
    error: Optional[str] = None
    latency_ms: float = 0.0


class Position(BaseModel):
    """Current position from broker."""
    model_config = {"frozen": True}

    ticker: str
    quantity: float
    avg_cost: float
    current_value: float
    unrealized_pnl: float


class AccountSnapshot(BaseModel):
    """Account state from broker."""
    model_config = {"frozen": True}

    total_value: float
    cash: float
    buying_power: float
    positions: list[Position] = Field(default_factory=list)


class BaseBroker(ABC):
    """Abstract base class for all broker adapters."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult: ...

    @abstractmethod
    def get_positions(self) -> list[Position]: ...

    @abstractmethod
    def get_account(self) -> AccountSnapshot: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def is_connected(self) -> bool: ...
```

- [ ] **Step 2: Create `__init__.py`**
- [ ] **Step 3: Commit**

```bash
git add trading_system/broker/
git commit -m "feat(broker): add BaseBroker ABC with Order/Position/Account models"
```

### Task 5.2: Enhanced MockBrokerAdapter

**Files:**
- Create: `trading_system/broker/mock.py`
- Create: `tests/unit/test_broker_mock.py`

- [ ] **Step 1: Write failing tests for enhanced mock broker**

Test: configurable latency, partial fills, rejections, slippage simulation (10bps), order state tracking.

- [ ] **Step 2: Implement MockBrokerAdapter v2**

Extends `BaseBroker` with all test scenarios. Supports configurable fill fraction, latency, fail tickers, slippage model.

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

### Task 5.3: Order Execution Manager

**Files:**
- Create: `trading_system/order_manager.py`
- Create: `tests/unit/test_order_manager.py`

- [ ] **Step 1: Write failing tests for order manager**

Test: order lifecycle (PENDING → SUBMITTED → FILLED), partial fill handling, timeout cancellation, AllocationDecision → Order translation.

- [ ] **Step 2: Implement order manager**

Consumes `AllocationDecision` from Layer 3 (or test fixtures when PM not yet active). Translates to broker `Order` objects. Tracks order state machine.

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

### Task 5.4: OOB Reconciliation

**Files:**
- Create: `scripts/reconcile.py`
- Create: `tests/unit/test_reconciliation.py`

- [ ] **Step 1: Write failing tests for reconciliation logic**

Test: matching positions pass, mismatch > 0.05% raises SystemError, empty broker positions vs non-empty SharedState.

- [ ] **Step 2: Implement reconciliation script**

Polls broker API, compares to SharedState PortfolioSnapshot, logs discrepancies.

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

### Task 5.5: LiveRunner Hardening

**Files:**
- Modify: `pattern_engine/live.py:140`
- Create/extend: `tests/unit/test_live_runner.py`

- [ ] **Step 1: Write tests for LiveRunner error scenarios**

Test: connection loss, partial fills, timeout (5-min), broker rejection, Pydantic state management.

- [ ] **Step 2: Harden LiveRunner with full error handling**
- [ ] **Step 3: Run full suite**
- [ ] **Step 4: Commit**

### Task 5.6: IBKR Adapter (or Alpaca Fallback)

**Files:**
- Create: `trading_system/broker/ibkr.py`

- [ ] **Step 1: Implement IBKR adapter implementing BaseBroker**

Connection health monitoring, auto-reconnect, REST JSON payloads.

- [ ] **Step 2: Write adapter tests (using mock IBKR responses)**
- [ ] **Step 3: Commit**

### GATE CHECK: Phase 5

- [ ] **Mock broker round-trip parity for 100 trades? OOB passes? Pipeline < 3 min?**
  - YES → Ready for Phase 7 merge
  - NO → Use Alpaca as fallback

---

## Phase 6: Universe Expansion (Can Start After Phase 1 Gate)

**Gate:** HNSW recall@50 ≥ 0.9999, BSS > 0 on ≥ 3/6 folds at 1500T, pipeline < 2hr, peak RAM < 24 GB
**Duration:** 3 weeks
**Prerequisite:** Phase 1 gate passed

### Task 6.1: Russell 1000 Data Pipeline

**Files:**
- Reference: `prepare.py` (LOCKED — do not modify)
- Create: `scripts/build_overnight_index.py`
- Create: `scripts/validate_sector_map.py`

- [ ] **Step 1: Extend ticker list to cover Russell 1000**

Create a ticker sourcing script that fetches S&P 500 + Russell 1000 constituents, applies the 2010 historical depth gate, and outputs the expanded ticker list.

- [ ] **Step 2: Validate sector mapping for all new tickers**

Write `scripts/validate_sector_map.py` that checks every ticker has a GICS sector assignment. Flag unmapped tickers.

- [ ] **Step 3: Expand SECTOR_MAP in config**
- [ ] **Step 4: Run data pipeline on expanded universe**
- [ ] **Step 5: Commit**

### Task 6.2: HNSW Scaling Benchmark

**Files:**
- Create: `tests/performance/test_hnsw_1500t.py`

- [ ] **Step 1: Write scaling benchmark test**

```python
"""Benchmark HNSW at 1500T scale (~3.7M rows).

Measures: build time, query latency, recall@50 vs BallTree, disk size.
Gate: recall@50 >= 0.9999
"""
import pytest


@pytest.mark.slow
class TestHNSW1500TScaling:

    def test_recall_at_50_meets_threshold(self, hnsw_1500t_index, balltree_1500t_index):
        """HNSW recall@50 must be >= 0.9999 vs exact BallTree."""
        # Query both, compare neighbor sets
        # recall = |intersection| / k for each query, average
        assert recall >= 0.9999

    def test_query_latency_within_budget(self, hnsw_1500t_index):
        """Full universe query must complete in < 5 seconds."""
        # Time 1500 queries
        assert total_seconds < 5.0

    def test_disk_serialization_roundtrip(self, hnsw_1500t_index, tmp_path):
        """Save/load preserves index exactly."""
        # save_index, load_index, compare query results
        pass
```

- [ ] **Step 2: Run benchmark**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/performance/test_hnsw_1500t.py -v -m slow`
Expected: recall ≥ 0.9999

- [ ] **Step 3: If recall < 0.9999, increase ef_search and retest**
- [ ] **Step 4: Commit results**

### Task 6.3: Overnight Index Build Script

**Files:**
- Create: `scripts/build_overnight_index.py`

- [ ] **Step 1: Write overnight build script**

Loads full 1500T feature matrix, builds HNSW index, saves to disk. Designed to run as overnight batch (after EOD data pull).

- [ ] **Step 2: Test: build → save → load → query parity**
- [ ] **Step 3: Commit**

### Task 6.4: BSS Re-Validation at 1500T

- [ ] **Step 1: Run walk-forward on 1500T with Phase 1 fix settings**
- [ ] **Step 2: Check BSS > 0 on ≥ 3/6 folds**
- [ ] **Step 3: If BSS degrades → apply same fix as Phase 1 (tighter max_distance, etc.)**
- [ ] **Step 4: Commit results with provenance**

### GATE CHECK: Phase 6

- [ ] **HNSW recall@50 ≥ 0.9999? BSS > 0 on ≥ 3/6? Pipeline < 2hr? No NaN/inf?**
  - YES → Ready for Phase 7 merge
  - NO → Fall back to 585T. Expansion becomes post-launch enhancement.

---

## Phase 7: Model Enhancements

**Gate:** Each enhancement individually gated — keep what passes, revert what doesn't
**Duration:** 4 weeks
**Prerequisite:** Phases 4, 5, 6 gates all passed

### Task 7.1: BMA Calibrator Integration

**Files:**
- Modify: `pattern_engine/matcher.py:315` — add BMA calibration option
- Create: `tests/unit/test_bma_integration.py`
- Reference: `research/bma_calibrator.py:35` — BMACalibrator class

- [ ] **Step 1: Write failing test for BMA in pipeline**

```python
def test_bma_calibration_produces_valid_probabilities():
    """BMA calibrator output is in [0, 1] and different from raw."""
    # Use EngineConfig(calibration="bma")
    # Run matcher with BMA, verify output probabilities
    pass
```

- [ ] **Step 2: Wire BMA into _package_results as calibration option**

Add `if self.config.calibration == "bma"` branch alongside existing Platt.

- [ ] **Step 3: Run walk-forward with BMA**
- [ ] **Step 4: Check gate: BSS improvement ≥ +0.005 on ≥ 3/6 folds**
- [ ] **Step 5: If pass → set flag True. If fail → revert, keep Platt.**
- [ ] **Step 6: Commit**

### Task 7.2: Conformal Prediction Intervals

**Files:**
- Modify: `pattern_engine/conformal_hooks.py` — wire into pipeline
- Create: `tests/unit/test_conformal.py`

- [ ] **Step 1: Write failing test for conformal coverage**

```python
def test_conformal_coverage_at_90_percent():
    """Empirical coverage must be >= 88% at nominal 90% level."""
    # Generate prediction intervals, check coverage on validation set
    pass
```

- [ ] **Step 2: Implement conformal prediction in conformal_hooks.py**
- [ ] **Step 3: Run walk-forward, check coverage per fold**
- [ ] **Step 4: Gate: coverage ≥ 88% on all 6 folds**
- [ ] **Step 5: If pass → wire interval width into Kelly sizing. If fail → revert.**
- [ ] **Step 6: Commit**

### Task 7.3: DTW Distance (WFA Reranker Promotion)

**Files:**
- Modify: `pattern_engine/wfa_reranker.py` — promote from research pilot
- Reference: `pattern_engine/matcher.py` — post-filter reranking

- [ ] **Step 1: Write test for DTW reranker BSS impact**
- [ ] **Step 2: Enable WFA reranker flag, run walk-forward**
- [ ] **Step 3: Gate: BSS improvement ≥ +0.003 on ≥ 3/6 folds, within 5-min window**
- [ ] **Step 4: If pass → activate. If fail → keep behind flag.**
- [ ] **Step 5: Commit**

### Task 7.4: CPOD/EILOF Anomaly Pre-Filter

**Files:**
- Create: `pattern_engine/anomaly_filter.py`
- Create: `tests/unit/test_anomaly_filter.py`

- [ ] **Step 1: Write failing tests for anomaly filter**

```python
def test_anomaly_filter_reduces_false_positives():
    """Filter should reduce FPR by >= 5% without TPR drop > 2%."""
    pass

def test_anomaly_filter_does_not_over_veto():
    """Filter should veto <= 30% of signals."""
    pass
```

- [ ] **Step 2: Implement as SignalFilterBase subclass**

Uses Isolation Forest for local outlier detection. Slashes confidence for extreme outlier patterns.

- [ ] **Step 3: Register in signal_pipeline.py (before SectorConviction)**
- [ ] **Step 4: Run walk-forward, check FPR/TPR gate**
- [ ] **Step 5: If pass → activate. If fail → revert.**
- [ ] **Step 6: Commit**

### Task 7.5: Dynamic Feature Weighting (OWA)

**Files:**
- Modify: `pattern_engine/matcher.py:144` — _prepare_features regime-conditional weights

- [ ] **Step 1: Write test for regime-conditional weights**
- [ ] **Step 2: Add weight profile selection keyed by regime label**
- [ ] **Step 3: Run walk-forward, check BSS gate (≥ +0.003 on ≥ 3/6)**
- [ ] **Step 4: If pass → activate. If fail → revert.**
- [ ] **Step 5: Commit**

### Task 7.6: Cumulative Enhancement Summary

- [ ] **Step 1: Create results/enhancement_summary.tsv with all enhancement results**
- [ ] **Step 2: Document which enhancements are active in production config**
- [ ] **Step 3: Run full test suite to verify all changes are clean**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: 616+ tests pass

- [ ] **Step 4: Commit**

```bash
git add results/enhancement_summary.tsv trading_system/config.py
git commit -m "docs(enhancements): summary of Phase 7 — [N]/5 enhancements activated"
```

---

## Phase 8: Paper Trading Validation (3 Months)

**Gate:** All v1 success criteria by Month 3 (see spec Section 10.7)
**Duration:** 12 weeks
**Prerequisite:** Phase 7 gate passed

### Task 8.1: Daily Report Script

**Files:**
- Create: `scripts/daily_report.py`

- [ ] **Step 1: Write daily metrics script**

Generates: P&L, equity curve, trade count, win rate, BSS (30d/90d rolling), calibration curve, sector attribution, idle cash %, execution latency, slippage.

- [ ] **Step 2: Test with synthetic data**
- [ ] **Step 3: Commit**

### Task 8.2: Weekly Review Script

**Files:**
- Create: `scripts/weekly_review.py`

- [ ] **Step 1: Write weekly performance summary**

Compares to baselines, checks StrategyEvaluator status, generates markdown report.

- [ ] **Step 2: Commit**

### Task 8.3: Paper Trading Infrastructure

- [ ] **Step 1: Configure LiveRunner for paper trading mode**

Wire: OOB reconciliation (09:00 AM), signal generation (4:00 PM), SharedState checkpoint (4:30 PM).

- [ ] **Step 2: Set up scheduled execution (Windows Task Scheduler or cron equivalent)**
- [ ] **Step 3: Run first paper trading day, verify end-to-end**
- [ ] **Step 4: Commit**

### Task 8.4: Month 1 Review — Stability

- [ ] **Step 1: No crashes, no missed execution windows, reconciliation clean?**
- [ ] **Step 2: Document any issues in `docs/paper_trading/month1_review.md`**
- [ ] **Step 3: Fix any issues found**

### Task 8.5: Month 2 Review — Performance Trending

- [ ] **Step 1: Rolling metrics converging? No regime collapse?**
- [ ] **Step 2: Check calibration drift (drift_monitor.py CUSUM alerts)**
- [ ] **Step 3: Document in `docs/paper_trading/month2_review.md`**

### Task 8.6: Month 3 Review — Full Gate Evaluation

- [ ] **Step 1: Evaluate all v1 success criteria (spec Section 1.3)**
- [ ] **Step 2: Generate final paper trading report**
- [ ] **Step 3: Go/no-go recommendation**

### GATE CHECK: Phase 8

- [ ] **All v1 criteria hold? Zero reconciliation failures in final 30 days?**
  - YES → Proceed to Phase 9
  - Minor fail → Extend paper trading 1 month
  - Major fail → Halt, diagnose, may return to earlier phase

---

## Phase 9: Live Deployment ($10,000)

**Gate:** Ongoing monitoring via StrategyEvaluator
**Duration:** 4 weeks initial ramp
**Prerequisite:** Phase 8 gate passed

### Task 9.1: Tax Tracker — Failing Tests

**Files:**
- Create: `tests/unit/test_tax_tracker.py`
- Create: `trading_system/tax_tracker.py`

- [ ] **Step 1: Write failing tests for tax tracker**

```python
"""Tests for trading_system/tax_tracker.py — capital gains + wash sale tracking."""
import pytest
from datetime import date
from trading_system.tax_tracker import TaxTracker, TaxLot, TradeRecord


class TestTaxLotClassification:
    def test_short_term_under_365_days(self):
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2026, 1, 1), qty=10, price=150.0)
        result = tracker.record_sell("AAPL", date(2026, 6, 1), qty=10, price=170.0)
        assert result.tax_type == "SHORT_TERM"
        assert result.realized_gain == pytest.approx(200.0)

    def test_long_term_over_365_days(self):
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2025, 1, 1), qty=10, price=150.0)
        result = tracker.record_sell("AAPL", date(2026, 3, 1), qty=10, price=170.0)
        assert result.tax_type == "LONG_TERM"

    def test_fifo_ordering(self):
        """FIFO: oldest lot sold first."""
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2026, 1, 1), qty=5, price=100.0)
        tracker.record_buy("AAPL", date(2026, 2, 1), qty=5, price=150.0)
        result = tracker.record_sell("AAPL", date(2026, 3, 1), qty=5, price=120.0)
        # FIFO: sells the $100 lot → gain = (120-100)*5 = 100
        assert result.realized_gain == pytest.approx(100.0)


class TestWashSaleDetection:
    def test_wash_sale_within_30_days(self):
        """Re-entry within 30 days of loss = wash sale."""
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2026, 1, 1), qty=10, price=150.0)
        tracker.record_sell("AAPL", date(2026, 1, 15), qty=10, price=140.0)  # loss
        is_wash = tracker.check_wash_sale("AAPL", date(2026, 1, 20))
        assert is_wash is True

    def test_no_wash_sale_after_30_days(self):
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2026, 1, 1), qty=10, price=150.0)
        tracker.record_sell("AAPL", date(2026, 1, 15), qty=10, price=140.0)
        is_wash = tracker.check_wash_sale("AAPL", date(2026, 2, 20))
        assert is_wash is False

    def test_wash_sale_adjusts_cost_basis(self):
        """Wash sale disallowed loss added to replacement lot cost basis."""
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2026, 1, 1), qty=10, price=150.0)
        sell_result = tracker.record_sell("AAPL", date(2026, 1, 15), qty=10, price=140.0)
        # Loss = $100 (disallowed)
        tracker.record_buy("AAPL", date(2026, 1, 20), qty=10, price=145.0)
        # Adjusted cost basis = 145 + (150-140) = 155 per share
        lots = tracker.get_lots("AAPL")
        assert lots[0].cost_basis_per_share == pytest.approx(155.0)


class TestYTDSummary:
    def test_ytd_tallies(self):
        tracker = TaxTracker()
        tracker.record_buy("AAPL", date(2026, 1, 1), qty=10, price=100.0)
        tracker.record_sell("AAPL", date(2026, 3, 1), qty=10, price=120.0)
        summary = tracker.ytd_summary(as_of=date(2026, 3, 1))
        assert summary["short_term_gains"] == pytest.approx(200.0)
        assert summary["long_term_gains"] == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_tax_tracker.py -v`
Expected: FAIL

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/unit/test_tax_tracker.py
git commit -m "test(tax): add failing tests for capital gains tracker + wash sale detection"
```

### Task 9.2: Tax Tracker — Implementation

**Files:**
- Create: `trading_system/tax_tracker.py`

- [ ] **Step 1: Implement TaxTracker**

```python
"""tax_tracker.py — Capital gains tax tracking with wash sale detection.

Tracks realized gains/losses per tax lot (FIFO). Detects wash sales
(re-entry within 30 days of loss). Provides YTD summaries and
quarterly estimated tax liability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


@dataclass
class TaxLot:
    """A single tax lot (one purchase)."""
    ticker: str
    buy_date: date
    quantity: float
    cost_basis_per_share: float
    wash_sale_adjustment: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.quantity * self.cost_basis_per_share


@dataclass
class SaleResult:
    """Result of selling shares."""
    ticker: str
    sell_date: date
    quantity: float
    sell_price: float
    cost_basis_per_share: float
    realized_gain: float
    tax_type: str  # "SHORT_TERM" or "LONG_TERM"
    is_wash_sale: bool = False


class TaxTracker:
    """FIFO-based tax lot tracker with wash sale detection.

    Tracks all buy/sell activity, classifies gains as short-term
    or long-term, detects wash sales per IRS 30-day rule.
    """

    def __init__(self) -> None:
        self._lots: dict[str, list[TaxLot]] = {}  # ticker → lots (FIFO order)
        self._sales: list[SaleResult] = []
        self._recent_losses: dict[str, list[tuple[date, float]]] = {}  # ticker → [(date, loss)]

    def record_buy(self, ticker: str, buy_date: date, qty: float, price: float) -> TaxLot:
        """Record a share purchase. Returns the created tax lot."""
        lot = TaxLot(ticker=ticker, buy_date=buy_date, quantity=qty, cost_basis_per_share=price)

        # Check wash sale: if buying within 30 days of a loss on same ticker
        if ticker in self._recent_losses:
            for loss_date, loss_amount in self._recent_losses[ticker]:
                if (buy_date - loss_date).days <= 30:
                    # Adjust cost basis: add disallowed loss
                    per_share_adjustment = abs(loss_amount) / qty
                    lot.cost_basis_per_share += per_share_adjustment
                    lot.wash_sale_adjustment = per_share_adjustment

        if ticker not in self._lots:
            self._lots[ticker] = []
        self._lots[ticker].append(lot)
        return lot

    def record_sell(self, ticker: str, sell_date: date, qty: float, price: float) -> SaleResult:
        """Record a share sale using FIFO. Returns sale result."""
        if ticker not in self._lots or not self._lots[ticker]:
            raise RuntimeError(f"No lots to sell for {ticker}")

        # FIFO: sell from oldest lot
        lot = self._lots[ticker][0]
        if lot.quantity < qty:
            raise RuntimeError(f"Insufficient shares in oldest lot: {lot.quantity} < {qty}")

        holding_days = (sell_date - lot.buy_date).days
        tax_type = "LONG_TERM" if holding_days > 365 else "SHORT_TERM"
        realized_gain = (price - lot.cost_basis_per_share) * qty

        result = SaleResult(
            ticker=ticker, sell_date=sell_date, quantity=qty,
            sell_price=price, cost_basis_per_share=lot.cost_basis_per_share,
            realized_gain=realized_gain, tax_type=tax_type,
        )
        self._sales.append(result)

        # Update lot quantity
        lot.quantity -= qty
        if lot.quantity <= 0:
            self._lots[ticker].pop(0)

        # Track losses for wash sale detection
        if realized_gain < 0:
            if ticker not in self._recent_losses:
                self._recent_losses[ticker] = []
            self._recent_losses[ticker].append((sell_date, realized_gain))

        return result

    def check_wash_sale(self, ticker: str, proposed_buy_date: date) -> bool:
        """Check if buying ticker on this date would trigger a wash sale."""
        if ticker not in self._recent_losses:
            return False
        for loss_date, _ in self._recent_losses[ticker]:
            if (proposed_buy_date - loss_date).days <= 30:
                return True
        return False

    def get_lots(self, ticker: str) -> list[TaxLot]:
        """Get current open lots for a ticker."""
        return self._lots.get(ticker, [])

    def ytd_summary(self, as_of: date) -> dict:
        """YTD capital gains summary."""
        year_start = date(as_of.year, 1, 1)
        st_gains = sum(
            s.realized_gain for s in self._sales
            if s.sell_date >= year_start and s.tax_type == "SHORT_TERM" and s.realized_gain > 0
        )
        st_losses = sum(
            s.realized_gain for s in self._sales
            if s.sell_date >= year_start and s.tax_type == "SHORT_TERM" and s.realized_gain < 0
        )
        lt_gains = sum(
            s.realized_gain for s in self._sales
            if s.sell_date >= year_start and s.tax_type == "LONG_TERM" and s.realized_gain > 0
        )
        lt_losses = sum(
            s.realized_gain for s in self._sales
            if s.sell_date >= year_start and s.tax_type == "LONG_TERM" and s.realized_gain < 0
        )
        return {
            "short_term_gains": st_gains,
            "short_term_losses": st_losses,
            "long_term_gains": lt_gains,
            "long_term_losses": lt_losses,
            "net_short_term": st_gains + st_losses,
            "net_long_term": lt_gains + lt_losses,
            "total_realized": st_gains + st_losses + lt_gains + lt_losses,
        }

    def export_csv(self, output_path: str) -> None:
        """Export trade log in tax software compatible CSV."""
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Ticker", "Sell Date", "Quantity", "Sell Price",
                "Cost Basis", "Realized Gain/Loss", "Tax Type", "Wash Sale",
            ])
            for s in self._sales:
                writer.writerow([
                    s.ticker, s.sell_date.isoformat(), s.quantity,
                    f"{s.sell_price:.2f}", f"{s.cost_basis_per_share:.2f}",
                    f"{s.realized_gain:.2f}", s.tax_type, s.is_wash_sale,
                ])
```

- [ ] **Step 2: Run tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_tax_tracker.py -v`
Expected: all PASS

- [ ] **Step 3: Run full suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add trading_system/tax_tracker.py
git commit -m "feat(tax): implement capital gains tracker — FIFO lots, wash sale detection, YTD summary"
```

> **Note:** Tasks 9.1-9.2 cover **launch-critical** tax tracking (FIFO lots, short/long-term classification, wash sale detection, CSV export). **Deferred reporting** (tax liability estimates, tax-loss harvesting alerts, quarterly estimated tax reports) should be added post-launch after the trading loop is stable. See spec Section 11.7.

### Task 9.3: Kill Switch

**Files:**
- Create: `scripts/emergency_halt.py`

- [ ] **Step 1: Implement kill switch**

Single command to: set StrategyEvaluator to RED, cancel all pending orders, optionally flatten all positions.

- [ ] **Step 2: Commit**

### Task 9.4: Deployment Runbook

**Files:**
- Create: `docs/DEPLOYMENT_RUNBOOK.md`

- [ ] **Step 1: Write deployment runbook**

Covers: broker account setup, API credential configuration, gradual ramp-up schedule, monitoring checklist, emergency procedures.

- [ ] **Step 2: Commit**

### Task 9.5: Live Deployment — Gradual Ramp

- [ ] **Step 1: Week 1-2: 25% capital (max 2-3 positions). Verify fills.**
- [ ] **Step 2: Week 3-4: 50% capital. Validate slippage model.**
- [ ] **Step 3: Month 2+: 100% capital if clean.**

---

## Phase 10: Options Trading Foundation

**Gate:** Code quality — all contracts validated, Greeks accurate, tests pass
**Duration:** 4-6 weeks
**Prerequisite:** Phase 9 stable

### Task 10.1: Options Contracts — Failing Tests

**Files:**
- Create: `tests/unit/test_options_contracts.py`
- Create: `trading_system/options/__init__.py`
- Create: `trading_system/options/contracts.py`

- [ ] **Step 1: Write failing tests for options contracts**

```python
"""Tests for trading_system/options/contracts.py — Pydantic option models."""
import pytest
from datetime import date
from trading_system.options.contracts import (
    OptionContract, OptionType, OptionStyle, OptionQuote, OptionOrder,
    OptionPosition, Direction, OrderType,
)


class TestOptionContract:
    def test_valid_call_option(self):
        contract = OptionContract(
            underlying="AAPL",
            expiration=date(2026, 6, 20),
            strike=150.0,
            option_type=OptionType.CALL,
            style=OptionStyle.AMERICAN,
        )
        assert contract.underlying == "AAPL"
        assert contract.option_type == OptionType.CALL

    def test_negative_strike_rejected(self):
        with pytest.raises(Exception):
            OptionContract(
                underlying="AAPL",
                expiration=date(2026, 6, 20),
                strike=-10.0,
                option_type=OptionType.CALL,
                style=OptionStyle.AMERICAN,
            )

    def test_frozen_immutability(self):
        contract = OptionContract(
            underlying="AAPL",
            expiration=date(2026, 6, 20),
            strike=150.0,
            option_type=OptionType.CALL,
            style=OptionStyle.AMERICAN,
        )
        with pytest.raises(Exception):
            contract.strike = 200.0


class TestOptionQuote:
    def test_greeks_present(self):
        contract = OptionContract(
            underlying="AAPL", expiration=date(2026, 6, 20),
            strike=150.0, option_type=OptionType.CALL, style=OptionStyle.AMERICAN,
        )
        quote = OptionQuote(
            contract=contract, bid=5.0, ask=5.50, mid=5.25,
            implied_vol=0.25, delta=0.55, gamma=0.03,
            theta=-0.05, vega=0.15, open_interest=1000, volume=500,
        )
        assert quote.delta == 0.55
        assert quote.implied_vol == 0.25


class TestOptionPosition:
    def test_long_position(self):
        contract = OptionContract(
            underlying="AAPL", expiration=date(2026, 6, 20),
            strike=150.0, option_type=OptionType.CALL, style=OptionStyle.AMERICAN,
        )
        pos = OptionPosition(
            contract=contract, quantity=10, avg_cost=5.0,
            current_value=6.0, unrealized_pnl=10.0,
        )
        assert pos.quantity == 10  # positive = long

    def test_short_position(self):
        contract = OptionContract(
            underlying="AAPL", expiration=date(2026, 6, 20),
            strike=150.0, option_type=OptionType.PUT, style=OptionStyle.AMERICAN,
        )
        pos = OptionPosition(
            contract=contract, quantity=-5, avg_cost=3.0,
            current_value=2.0, unrealized_pnl=5.0,
        )
        assert pos.quantity == -5  # negative = short
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_options_contracts.py -v`
Expected: FAIL

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_options_contracts.py
git commit -m "test(options): add failing tests for options contract models"
```

### Task 10.2: Options Contracts — Implementation

**Files:**
- Create: `trading_system/options/__init__.py`
- Create: `trading_system/options/contracts.py`

- [ ] **Step 1: Implement options contract models**

All frozen Pydantic models: OptionContract, OptionType, OptionStyle, OptionQuote, OptionOrder, OptionPosition, Direction, OrderType. Validation: positive strike, valid expiration, quantity sign convention.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

### Task 10.3: Greeks Calculator — Failing Tests

**Files:**
- Create: `tests/unit/test_greeks.py`
- Create: `trading_system/options/greeks.py`

- [ ] **Step 1: Write failing tests for Black-Scholes and binomial tree**

```python
"""Tests for trading_system/options/greeks.py — pricing and Greeks."""
import pytest
from trading_system.options.greeks import black_scholes_call, black_scholes_put, implied_vol


class TestBlackScholes:
    def test_call_price_known_example(self):
        """S=100, K=100, T=1, r=0.05, sigma=0.20 → call ≈ $10.45."""
        price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert abs(price - 10.45) < 0.10

    def test_put_price_known_example(self):
        """S=100, K=100, T=1, r=0.05, sigma=0.20 → put ≈ $5.57."""
        price = black_scholes_put(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert abs(price - 5.57) < 0.10

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT)."""
        import math
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        parity = S - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 0.01


class TestImpliedVol:
    def test_iv_recovery(self):
        """Given a BS price, IV solver recovers the original sigma."""
        price = black_scholes_call(S=100, K=100, T=1.0, r=0.05, sigma=0.25)
        iv = implied_vol(price, S=100, K=100, T=1.0, r=0.05, option_type="call")
        assert abs(iv - 0.25) < 0.001
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Commit**

### Task 10.4: Greeks Calculator — Implementation

**Files:**
- Create: `trading_system/options/greeks.py`

- [ ] **Step 1: Implement Black-Scholes, binomial tree, IV solver**

Black-Scholes for European, binomial tree for American. Newton-Raphson IV solver. Portfolio-level Greeks aggregation.

- [ ] **Step 2: Run tests**
- [ ] **Step 3: Commit**

### Task 10.5: Strategy Templates & Signal Bridge

**Files:**
- Create: `trading_system/options/strategies.py`
- Create: `trading_system/options/data.py`
- Create: `trading_system/options/signal_bridge.py`
- Create: `tests/unit/test_signal_bridge.py`

- [ ] **Step 1: Write failing tests for signal→options mapping**

```python
def test_high_confidence_buy_low_iv_maps_to_long_call():
    """BUY > 0.70 + low IV → long call or bull spread."""
    pass

def test_moderate_buy_with_position_maps_to_covered_call():
    """BUY 0.55-0.70 + existing equity position → covered call."""
    pass

def test_hold_with_high_iv_maps_to_covered_call():
    """HOLD on existing + high IV → covered call for income."""
    pass
```

- [ ] **Step 2: Implement strategy ABCs (BaseOptionStrategy + stubs)**
- [ ] **Step 3: Implement BaseOptionsDataProvider ABC**
- [ ] **Step 4: Implement signal bridge mapping logic**
- [ ] **Step 5: Run tests**
- [ ] **Step 6: Commit**

### Task 10.6: SharedState Extension

**Files:**
- Modify: `trading_system/contracts/state.py`

- [ ] **Step 1: Add OptionsState to SharedState**

```python
class OptionsState(BaseModel):
    """Options positions and portfolio-level Greeks."""
    model_config = {"frozen": True}

    positions: Dict[str, list] = Field(default_factory=dict)
    net_delta: float = Field(default=0.0)
    net_gamma: float = Field(default=0.0)
    net_theta: float = Field(default=0.0)
    net_vega: float = Field(default=0.0)
    total_options_exposure: float = Field(default=0.0)
    margin_used: float = Field(default=0.0)
```

- [ ] **Step 2: Run full test suite to verify nothing breaks**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: all pass

- [ ] **Step 3: Commit**

### Task 10.7: Options Design Document

**Files:**
- Create: `docs/OPTIONS_TRADING_DESIGN.md`

- [ ] **Step 1: Write design doc for future options implementation**

Cover: strategy selection logic, risk management for options (delta limits, margin), backtesting approach, data feed requirements.

- [ ] **Step 2: Commit**

### GATE CHECK: Phase 10

- [ ] **All contracts validated? Greeks accurate? Signal bridge tested? No regressions?**
  - YES → Options foundation complete. Ready for future implementation.

---

## Diagnostic Quick-Reference

When things go wrong, use this lookup table:

| Symptom | Phase | First Check | Second Check | Escalation |
|---------|-------|-------------|--------------|------------|
| BSS stays negative | 1 | Reliability diagram | Distance distribution | BMA rescue or 52T fallback |
| Kelly fraction tiny | 2 | Win probability distribution | avg_win/avg_loss ratio | Quarter-Kelly or revert to flat |
| Max DD increases | 3 | Trade-by-trade before/after | Brake threshold too loose | Disable overlays |
| Sharpe drops > 0.3 | 3 | Fatigue overlay too aggressive | Congestion gate too strict | Keep core, disable extras |
| Idle cash > 80% | 4 | Confidence threshold binding? | Sector limits too tight? | Sector-limit-only mode |
| PM rejects > 60% | 4 | Which constraint is binding? | Relax max_positions_per_sector | Skip ranking |
| Broker connection fails | 5 | TWS gateway running? | API permissions? | Alpaca fallback |
| OOB mismatch | 5 | Corporate actions? | Partial fill tracking | Manual reconciliation |
| HNSW recall < 0.9999 | 6 | Increase ef_search | Increase ef_construction | BallTree fallback |
| BSS degrades at 1500T | 6 | Tighter max_distance | same_sector_only | Stay at 585T |
| Enhancement hurts BSS | 7 | Check on worst fold | Verify flag isolation | Revert, keep behind flag |
| Paper trading crash | 8 | Memory usage (32GB limit) | Uncaught exception | Profile, fix, restart |
| Calibration drift | 8 | CUSUM alert details | Regime shift? | Re-run Platt calibration |
| Real slippage > model | 9 | Order type (market vs limit) | Execution venue/timing | VWAP window |
