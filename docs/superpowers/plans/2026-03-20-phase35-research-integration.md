# Phase 3.5 Research Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three isolated research modules — EMD distance metric, BMA calibrator, and seismic slip-deficit TTF overlay — as ABC-backed, smoke-tested Python packages, without touching any existing production code.

**Architecture:** A new `research/` package at project root defines three ABCs (`BaseDistanceMetric`, `BaseCalibrator`, `BaseRiskOverlay`) in `__init__.py`. Each concrete module subclasses the relevant ABC. Tests live in `tests/test_research/`. No production modules are modified.

**Tech Stack:** Python 3.12, numpy, scipy (already present), `pot` (new — Python Optimal Transport), pytest

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `research/__init__.py` | Create | ABCs + `RiskOverlayResult` dataclass |
| `research/emd_distance.py` | Create | Earth Mover's Distance metric |
| `research/bma_calibrator.py` | Create | Bayesian Model Averaging calibrator |
| `research/slip_deficit.py` | Create | Seismic slip-deficit + TTF overlay |
| `research/phase_c_roadmap.md` | Create | Deferred domain stubs |
| `tests/test_research/__init__.py` | Create | Empty — makes directory a package |
| `tests/test_research/test_emd_distance.py` | Create | 6 EMD smoke tests |
| `tests/test_research/test_bma_calibrator.py` | Create | 5 BMA smoke tests |
| `tests/test_research/test_slip_deficit.py` | Create | 7 slip-deficit smoke tests |

**Production files touched:** None.

---

## Task 1: Environment Setup

**Files:**
- Create: `research/__init__.py` (empty placeholder)
- Create: `tests/test_research/__init__.py` (empty)

- [ ] **Step 1: Create branch**

```bash
cd C:/Users/Isaia/.claude/financial-research
git checkout -b phase35-research
```

Expected: `Switched to a new branch 'phase35-research'`

- [ ] **Step 2: Install `pot` library**

```bash
venv/Scripts/pip install pot
```

Expected: `Successfully installed pot-...` (or already installed). Verify:

```bash
venv/Scripts/python -c "import ot; print('ot version:', ot.__version__)"
```

- [ ] **Step 3: Create package directories and empty `__init__.py` files**

```bash
mkdir -p research tests/test_research
```

Create `research/__init__.py` — leave empty for now:
```python
# Populated in Task 2
```

Create `tests/test_research/__init__.py` — empty file.

- [ ] **Step 4: Verify existing test suite still passes**

```bash
venv/Scripts/python -m pytest tests/ -v --tb=short -q
```

Expected: `556 passed` (or current count). Zero failures. If anything fails, stop — do not proceed.

- [ ] **Step 5: Commit scaffold**

```bash
git add research/__init__.py tests/test_research/__init__.py
git commit -m "chore(phase3.5): create research package scaffold and test directory"
```

---

## Task 2: ABCs and `RiskOverlayResult`

**Files:**
- Modify: `research/__init__.py`

No TDD here — abstract base classes cannot be instantiated and are verified implicitly by the concrete class tests in Tasks 3–5. Implement directly.

- [ ] **Step 1: Write `research/__init__.py`**

```python
"""
research/__init__.py — Abstract base classes for FPPE research modules.

Each research module subclasses the relevant ABC. This enforces the interface
contract so that a validated module can be promoted to production with zero
changes to its callers.

Promotion notes:
  - BaseDistanceMetric → replaces ball_tree in pattern_engine/matching.py
  - BaseCalibrator     → replaces PlattCalibrator in pattern_engine/calibration.py
                         (requires migration of signal_adapter.py callers — Phase C)
  - BaseRiskOverlay    → augments risk_engine.py (wiring is Phase C work)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


class BaseDistanceMetric(ABC):
    """Interface for distance metrics that shadow matching.py's ball_tree computation."""

    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseDistanceMetric":
        """Store training data for optional normalization. May be a no-op."""
        ...

    @abstractmethod
    def compute(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute distance from query (shape (D,)) to each candidate (shape (N, D)).

        Returns: np.ndarray of shape (N,) — one distance per candidate.
        """
        ...


class BaseCalibrator(ABC):
    """Interface matching PlattCalibrator in pattern_engine/calibration.py.

    BMACalibrator.transform() is a drop-in for PlattCalibrator.transform().
    Wiring into signal_adapter.py callers is Phase C promotion work.

    Note: generate_pdf() is NOT declared here — it is BMA-specific and will be
    called through a concrete BMACalibrator reference in Phase C integration.
    If a future calibrator also exposes a PDF method, promote it to this ABC.
    """

    @abstractmethod
    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator":
        """Fit calibrator on training data.

        Args:
            raw_probs: Raw probabilities. Shape depends on subclass:
                       PlattCalibrator expects (N,); BMACalibrator expects (N, K).
            y_true:    Binary labels, shape (N,).
        """
        ...

    @abstractmethod
    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Map raw probabilities to calibrated probabilities in [0, 1]."""
        ...

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """True after fit() has been called successfully."""
        ...


class BaseRiskOverlay(ABC):
    """Interface for additive risk overlays that augment risk_engine.py.

    No fit() method: overlays are stateless and compute entirely from the price
    series passed at call time. If a future overlay requires historical
    calibration, add fit() to this ABC at that point.
    """

    @abstractmethod
    def compute(
        self,
        prices_df: pd.DataFrame,
        positions: Optional[list] = None,
    ) -> "RiskOverlayResult":
        """Compute risk overlay signals from price history.

        Args:
            prices_df: DataFrame with at minimum a 'close' column, sorted ascending.
            positions: Optional list of open positions (ignored by SlipDeficit).
        """
        ...


@dataclass
class RiskOverlayResult:
    """Output of BaseRiskOverlay.compute().

    slip_deficit: Signed normalized divergence from the SMA anchor.
        Formula: (current_price - SMA_N) / SMA_N
        Positive = price overextended above anchor (seismic "loading").
        Negative = price below anchor (oversold).
        The seismic analogy uses non-negative loading, but the signed
        financial interpretation is intentional — negative values are
        valid data, not errors.

    ttf_probability: Continuous probability in [0, 1] of imminent failure
        (volatility expansion / mean-reversion event). Computed as
        sigmoid(vol_zscore).

    tighten_stops: True when the volatility Z-score exceeds ttf_threshold.
        The threshold is always interpreted in Z-score units — not probability.
    """

    slip_deficit: float
    ttf_probability: float
    tighten_stops: bool
```

- [ ] **Step 2: Verify it imports cleanly**

```bash
venv/Scripts/python -c "from research import BaseDistanceMetric, BaseCalibrator, BaseRiskOverlay, RiskOverlayResult; print('ABCs OK')"
```

Expected: `ABCs OK`

- [ ] **Step 3: Commit**

```bash
git add research/__init__.py
git commit -m "feat(research): add ABCs and RiskOverlayResult to research/__init__.py"
```

---

## Task 3: EMD Distance — Tests First

**Files:**
- Create: `tests/test_research/test_emd_distance.py`
- Create: `research/emd_distance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research/test_emd_distance.py`:

```python
"""
Smoke tests for EMDDistance.

Tests verify: initialization, correct shape output, known-answer correctness,
and penalty-axis isolation. Uses synthetic numpy arrays only — no market data.
"""

import numpy as np
import pytest

from research.emd_distance import EMDDistance


def _dummy_fingerprint(value: float = 0.0) -> np.ndarray:
    """Return a uniform 8-feature fingerprint."""
    return np.full(8, value, dtype=np.float64)


def test_init_and_fit_no_error():
    """EMDDistance initializes and fits without raising."""
    metric = EMDDistance()
    X_dummy = np.random.default_rng(0).standard_normal((50, 8))
    metric.fit(X_dummy)  # no-op — should not raise


def test_compute_returns_array_of_distances():
    """compute() returns shape (N,) for N candidates, all finite."""
    metric = EMDDistance().fit(np.zeros((1, 8)))
    query = _dummy_fingerprint(0.01)
    candidates = np.stack([_dummy_fingerprint(0.01), _dummy_fingerprint(0.05)])
    result = metric.compute(query, candidates)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_identical_distributions_zero():
    """EMD between identical fingerprints is 0."""
    metric = EMDDistance().fit(np.zeros((1, 8)))
    fp = np.array([0.01, -0.02, 0.05, 0.03, -0.01, 0.04, 0.02, -0.03])
    result = metric.compute(fp, fp.reshape(1, -1))
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_directional_ordering():
    """EMD of close pair < EMD of far pair."""
    metric = EMDDistance().fit(np.zeros((1, 8)))
    base = np.zeros(8)
    close = base + 0.001
    far = base + 1.0
    dist_close = metric.compute(base, close.reshape(1, -1))[0]
    dist_far = metric.compute(base, far.reshape(1, -1))[0]
    assert dist_close < dist_far


def test_time_penalty_zero_collapses_time_axis():
    """With time_penalty=0, fingerprints with same return VALUES but reversed temporal
    ordering have distance ≈ 0 — the time axis is zeroed so only return magnitude matters.
    With time_penalty=1 (default), the same pair would have nonzero distance.
    """
    metric = EMDDistance(time_penalty=0.0, price_penalty=1.0).fit(np.zeros((1, 8)))
    # fp_a: returns increase with time. fp_b: same values, reversed order.
    # With time_penalty=0 → coords are (0, r_i) for both → same multiset → EMD = 0
    fp_a = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    fp_b = fp_a[::-1].copy()  # same values, different temporal assignment
    result = metric.compute(fp_a, fp_b.reshape(1, -1))
    assert result[0] == pytest.approx(0.0, abs=1e-6)


def test_price_penalty_zero_collapses_price_axis():
    """With price_penalty=0, fingerprints with different returns have distance ≈ 0
    because the return axis is zeroed in the cost matrix."""
    metric = EMDDistance(time_penalty=1.0, price_penalty=0.0).fit(np.zeros((1, 8)))
    # Returns differ dramatically — but price axis is zeroed, so only time-lags matter
    # Both use the same time lags [1,3,7,...,90], so cost matrix rows are identical
    fp_a = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    fp_b = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20])
    result = metric.compute(fp_a, fp_b.reshape(1, -1))
    assert result[0] == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
venv/Scripts/python -m pytest tests/test_research/test_emd_distance.py -v
```

Expected: All 6 tests FAIL with `ModuleNotFoundError: No module named 'research.emd_distance'`

- [ ] **Step 3: Implement `research/emd_distance.py`**

```python
"""
emd_distance.py — Earth Mover's Distance metric for FPPE fingerprints.

Treats each 8-feature return fingerprint as a 2D weighted point set:
  - x-axis: time lag (calendar days) × time_penalty
  - y-axis: return value × price_penalty

Time lags are [1, 3, 7, 14, 30, 45, 60, 90] matching RETURN_WINDOWS in
pattern_engine/features.py.

Two-stage retrieval (Phase C): In production this would be Stage 2 — FAISS
(HNSW) provides top-100 candidates, then EMD re-ranks them. For Phase B,
EMD computes directly on any candidate set.

Fallback: If `pot` is unavailable, uses scipy.stats.wasserstein_distance_nd
(requires SciPy >= 1.13). The fallback is equivalent ONLY for uniform weights;
it does not support non-uniform volume weights (Phase C) and must be replaced
at that time.

Reference: Docs 1 & 2, "Linguistics and Audio Processing" sections.
"""

import numpy as np
from scipy.spatial.distance import cdist

from research import BaseDistanceMetric

# Calendar-day lags matching pattern_engine/features.py RETURN_WINDOWS
RETURN_WINDOWS = [1, 3, 7, 14, 30, 45, 60, 90]


class EMDDistance(BaseDistanceMetric):
    """Earth Mover's Distance between 8-feature return fingerprints.

    Handles temporal phase shifts gracefully: a pattern that took 2 weeks
    in 2008 but 3 weeks in 2020 is penalized less than a structurally
    different pattern, because time_penalty < price_penalty.
    """

    def __init__(self, time_penalty: float = 0.5, price_penalty: float = 1.0):
        """
        Args:
            time_penalty:  Scale applied to the time-lag axis of the cost matrix.
                           Lower values allow the algorithm to match patterns that
                           unfolded at different speeds (market time-warping).
                           Default 0.5 down-weights temporal misalignment vs magnitude.
            price_penalty: Scale applied to the return-value axis. Default 1.0.
        """
        self.time_penalty = time_penalty
        self.price_penalty = price_penalty
        self._X_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "EMDDistance":
        """Store training data reference. No-op for Phase B.

        Future variants may compute normalization statistics here.
        """
        self._X_train = X_train
        return self

    def _construct_coords(self, fingerprint: np.ndarray) -> np.ndarray:
        """Transform 8-feature fingerprint to (8, 2) weighted point set.

        Each row: (time_lag * time_penalty, return_value * price_penalty)
        time_lags are [1, 3, 7, 14, 30, 45, 60, 90] — calendar-day windows.
        """
        time_lags = np.array(RETURN_WINDOWS, dtype=np.float64)
        coords = np.column_stack([
            time_lags * self.time_penalty,
            fingerprint * self.price_penalty,
        ])
        return coords  # shape (8, 2)

    def _emd_scalar(
        self,
        current_coords: np.ndarray,
        hist_coords: np.ndarray,
    ) -> float:
        """Compute EMD between two (8, 2) coordinate arrays."""
        weights_a = np.full(8, 1.0 / 8, dtype=np.float64)
        weights_b = np.full(8, 1.0 / 8, dtype=np.float64)
        cost_matrix = cdist(current_coords, hist_coords, metric="euclidean")
        try:
            import ot
            return float(ot.emd2(weights_a, weights_b, cost_matrix))
        except ImportError:
            return self._scipy_fallback(current_coords, hist_coords)

    def _scipy_fallback(
        self,
        current_coords: np.ndarray,
        hist_coords: np.ndarray,
    ) -> float:
        """Fallback using scipy.stats.wasserstein_distance_nd (SciPy >= 1.13).

        Equivalent to POT for uniform weights only. When Phase C adds
        volume-based non-uniform weights, replace with a pure-numpy
        min-cost-flow implementation.
        """
        import scipy
        from packaging.version import Version

        if Version(scipy.__version__) < Version("1.13"):
            raise ImportError(
                "scipy >= 1.13 is required for the EMD fallback. "
                "Install pot instead: pip install pot"
            )
        from scipy.stats import wasserstein_distance_nd

        return float(wasserstein_distance_nd(current_coords, hist_coords))

    def compute(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Compute EMD from query fingerprint to each candidate fingerprint.

        Args:
            query:      Shape (8,) — current market fingerprint.
            candidates: Shape (N, 8) — historical analogue fingerprints.

        Returns:
            Shape (N,) — EMD distance to each candidate. Lower = more similar.
        """
        query_coords = self._construct_coords(query)
        distances = np.array([
            self._emd_scalar(query_coords, self._construct_coords(c))
            for c in candidates
        ])
        return distances
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
venv/Scripts/python -m pytest tests/test_research/test_emd_distance.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Verify full suite still passes**

```bash
venv/Scripts/python -m pytest tests/ -v --tb=short -q
```

Expected: 562 passed (556 original + 6 new), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add research/emd_distance.py tests/test_research/test_emd_distance.py
git commit -m "feat(research): add EMDDistance with 6 passing smoke tests"
```

---

## Task 4: BMA Calibrator — Tests First

**Files:**
- Create: `tests/test_research/test_bma_calibrator.py`
- Create: `research/bma_calibrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research/test_bma_calibrator.py`:

```python
"""
Smoke tests for BMACalibrator.

Tests verify: initialization state, post-fit invariants, output range,
degenerate EM convergence, and two-cluster weight separation.
Uses synthetic numpy arrays only — no market data.
"""

import numpy as np
import pytest

from research.bma_calibrator import BMACalibrator


def _make_training_data(
    n_samples: int = 100,
    n_analogs: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw_probs (N, K), y_true (N,)) with random synthetic values."""
    rng = np.random.default_rng(seed)
    raw_probs = rng.uniform(0.1, 0.9, (n_samples, n_analogs))
    y_true = rng.integers(0, 2, n_samples).astype(float)
    return raw_probs, y_true


def test_init_not_fitted():
    """BMACalibrator starts unfitted."""
    cal = BMACalibrator(num_analogs=5)
    assert cal.fitted is False


def test_fit_sets_fitted_and_weights_sum_to_one():
    """After fit(), fitted is True and weights sum to 1."""
    cal = BMACalibrator(num_analogs=5)
    raw_probs, y_true = _make_training_data(n_samples=100, n_analogs=5)
    cal.fit(raw_probs, y_true)
    assert cal.fitted is True
    assert cal.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_transform_output_in_unit_interval():
    """transform() output is in [0, 1] for any query."""
    cal = BMACalibrator(num_analogs=5)
    raw_probs, y_true = _make_training_data(n_samples=100, n_analogs=5)
    cal.fit(raw_probs, y_true)
    rng = np.random.default_rng(99)
    query_probs = rng.uniform(0.0, 1.0, 5)
    result = cal.transform(query_probs)
    assert 0.0 <= result <= 1.0


def test_uniform_convergence_degenerate_case():
    """With identical analogs and constant y, weights stay approximately uniform.

    This is the degenerate case: EM has no signal to differentiate components,
    so weights should remain near 1/K. Catches catastrophic weight collapse bugs.
    """
    K = 5
    N = 100
    # All analogs always predict 0.5, y is constant 1.0
    raw_probs = np.full((N, K), 0.5)
    y_true = np.ones(N)
    cal = BMACalibrator(num_analogs=K, max_iter=50)
    cal.fit(raw_probs, y_true)
    expected = 1.0 / K
    assert np.allclose(cal.weights, expected, atol=0.05)


def test_two_cluster_weight_separation():
    """Two distinct analog clusters each absorb ~50% of total weight.

    Analogs 0-2 always predict 0.1 (good for y=0 samples).
    Analogs 3-5 always predict 0.9 (good for y=1 samples).
    With half y=0 and half y=1, each cluster is "responsible" for 50%
    of samples — so EM should converge to each cluster weight ≈ 0.5.
    Catches weight normalization bugs that the degenerate test cannot detect.
    """
    K = 6
    N = 200
    # Cluster 0: analogs 0-2 always predict 0.1
    # Cluster 1: analogs 3-5 always predict 0.9
    raw_probs = np.hstack([
        np.full((N, 3), 0.1),
        np.full((N, 3), 0.9),
    ])
    # Alternating y=0 and y=1 — each cluster handles half the samples
    y_true = np.tile([0.0, 1.0], N // 2)
    cal = BMACalibrator(num_analogs=K, df=3.0, max_iter=100)
    cal.fit(raw_probs, y_true)
    low_cluster_weight = cal.weights[:3].sum()
    high_cluster_weight = cal.weights[3:].sum()
    assert low_cluster_weight == pytest.approx(0.5, abs=0.15)
    assert high_cluster_weight == pytest.approx(0.5, abs=0.15)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
venv/Scripts/python -m pytest tests/test_research/test_bma_calibrator.py -v
```

Expected: All 5 tests FAIL with `ModuleNotFoundError: No module named 'research.bma_calibrator'`

- [ ] **Step 3: Implement `research/bma_calibrator.py`**

```python
"""
bma_calibrator.py — Bayesian Model Averaging calibrator for FPPE signals.

Replaces Platt scaling with a probabilistic ensemble calibration method.
Each historical analogue contributes a Student's t-distribution PDF weighted
by its historical predictive skill, optimized via Expectation-Maximization.

Correctly models heavy-tailed financial returns that Platt's Gaussian
assumption underestimates (df=3.0 default gives fat tails typical of equities).

Input/output semantics:
  fit(raw_probs, y_true):
    raw_probs: shape (N_training_samples, num_analogs)
               raw KNN probability from each of the K nearest neighbours
               across all training samples. Analogue locations are FIXED
               throughout EM — only weights and variances are updated.
    y_true: shape (N_training_samples,) — binary outcome (1=up, 0=not-up)

  transform(raw_probs):
    raw_probs: shape (num_analogs,) — K raw probs for a single query point
    Returns: scalar = np.dot(self.weights, raw_probs)
             Posterior-weighted mean of the K analogue probabilities.
             Guaranteed in [0, 1] as a convex combination of [0, 1] values.
             Matches PlattCalibrator.transform() output type.

  generate_pdf(analogue_probs, return_grid):
    Called through a concrete BMACalibrator reference (NOT through BaseCalibrator).
    Used for confidence interval computation in Phase C three-filter gate integration.

EM Algorithm note:
  The M-step uses a Gaussian variance update rather than the full Student's t
  M-step (which requires auxiliary u-weight precision variables). This is a
  deliberate approximation: the t-distribution is used in the E-step PDF for
  heavy-tail probability mass, while the simpler Gaussian variance update avoids
  the u-weight complexity at the cost of slight variance underestimation in
  extreme tails. The full Student's t M-step is a Phase C upgrade candidate.

Production promotion path:
  Replaces PlattCalibrator in pattern_engine/calibration.py once BSS improvement
  confirmed. Requires Phase C migration of signal_adapter.py legacy callers
  (fit_platt_scaling / calibrate_probabilities functions).

Reference: Doc 1, "Improving Baseline Probabilities" section; Raftery et al. BMA.
"""

import numpy as np
from scipy.stats import t as t_dist

from research import BaseCalibrator


class BMACalibrator(BaseCalibrator):
    """Bayesian Model Averaging calibrator with Student's t components."""

    def __init__(
        self,
        num_analogs: int,
        df: float = 3.0,
        max_iter: int = 50,
    ):
        """
        Args:
            num_analogs: Number of historical analogues (typically top_k=50).
            df:          Student's t degrees of freedom. Lower = heavier tails.
                         Default 3.0 is appropriate for equity return distributions.
            max_iter:    Maximum EM iterations. 50 is sufficient for convergence
                         in most cases; increase to 100 for noisy data.
        """
        self.num_analogs = num_analogs
        self.df = df
        self.max_iter = max_iter
        self.weights = np.ones(num_analogs) / num_analogs
        self.variances = np.ones(num_analogs)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "BMACalibrator":
        """Fit BMA weights and variances via Expectation-Maximization.

        Args:
            raw_probs: shape (N, K) — raw probability from each of the K
                       nearest neighbours for each of the N training samples.
                       Analogue location for sample n, component i = raw_probs[n, i].
                       Locations are FIXED — EM only updates weights and variances.
            y_true:    shape (N,) — binary outcome labels.
        """
        N = raw_probs.shape[0]
        # Re-initialize to uniform for each fit call
        self.weights = np.ones(self.num_analogs) / self.num_analogs
        self.variances = np.ones(self.num_analogs)

        for _ in range(self.max_iter):
            # --- E-step: compute latent posterior probabilities ---
            latent_pdfs = np.zeros((N, self.num_analogs))
            for i in range(self.num_analogs):
                scale = np.sqrt(self.variances[i]) + 1e-8
                latent_pdfs[:, i] = self.weights[i] * t_dist.pdf(
                    y_true,
                    df=self.df,
                    loc=raw_probs[:, i],
                    scale=scale,
                )

            total_pdf = latent_pdfs.sum(axis=1, keepdims=True) + 1e-10
            latent_probs = latent_pdfs / total_pdf  # shape (N, K)

            # --- M-step: update weights and variances ---
            # Weight update: posterior mean responsibility of each component
            self.weights = latent_probs.mean(axis=0)

            # Variance update: Gaussian approximation (deliberate simplification).
            # See module docstring for rationale. Full Student's t M-step is Phase C.
            for i in range(self.num_analogs):
                residuals = y_true - raw_probs[:, i]
                var_num = (latent_probs[:, i] * residuals ** 2).sum()
                var_den = latent_probs[:, i].sum() + 1e-10
                self.variances[i] = var_num / var_den

        self._fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> float:
        """Return BMA point forecast as posterior-weighted mean.

        Args:
            raw_probs: shape (num_analogs,) — raw probs for a single query.

        Returns:
            Scalar in [0, 1] — convex combination of [0, 1] inputs,
            guaranteed in range. Matches PlattCalibrator.transform() type.
        """
        if not self._fitted:
            raise RuntimeError(
                "BMACalibrator.transform() called before fit(). Call fit() first."
            )
        return float(np.dot(self.weights, raw_probs))

    def generate_pdf(
        self,
        analogue_probs: np.ndarray,
        return_grid: np.ndarray,
    ) -> np.ndarray:
        """Generate full BMA mixture PDF over a return grid.

        NOT declared in BaseCalibrator — call through a concrete BMACalibrator
        reference. Used for confidence interval computation in Phase C
        three-filter gate integration.

        Args:
            analogue_probs: shape (num_analogs,) — raw probs for current query,
                            used as the location parameter for each t-component.
            return_grid:    shape (M,) — grid of return values to evaluate PDF over.

        Returns:
            shape (M,) — BMA mixture density at each grid point.
        """
        pdf = np.zeros_like(return_grid, dtype=np.float64)
        for w, loc, var in zip(self.weights, analogue_probs, self.variances):
            scale = np.sqrt(var) + 1e-8
            pdf += w * t_dist.pdf(return_grid, df=self.df, loc=loc, scale=scale)
        return pdf
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
venv/Scripts/python -m pytest tests/test_research/test_bma_calibrator.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Verify full suite still passes**

```bash
venv/Scripts/python -m pytest tests/ -v --tb=short -q
```

Expected: 566 passed, 0 failed.

- [ ] **Step 6: Commit**

```bash
git add research/bma_calibrator.py tests/test_research/test_bma_calibrator.py
git commit -m "feat(research): add BMACalibrator with 5 passing smoke tests"
```

---

## Task 5: Slip-Deficit + TTF Overlay — Tests First

**Files:**
- Create: `tests/test_research/test_slip_deficit.py`
- Create: `research/slip_deficit.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_research/test_slip_deficit.py`:

```python
"""
Smoke tests for SlipDeficit.

Tests verify: initialization, zero deficit on flat series, positive deficit
with high vol, RiskOverlayResult field types, optional positions parameter,
and ValueError on insufficient history.
Uses synthetic price DataFrames only — no real market data.
"""

import numpy as np
import pandas as pd
import pytest

from research import RiskOverlayResult
from research.slip_deficit import SlipDeficit


def _flat_prices(n: int = 250, base: float = 100.0) -> pd.DataFrame:
    """Helper: flat price series of length n with no returns."""
    return pd.DataFrame({"close": np.full(n, base, dtype=np.float64)})


def test_init_no_error():
    """SlipDeficit initializes without raising."""
    SlipDeficit(sma_window=200)


def test_zero_deficit_flat_series():
    """slip_deficit == 0 when price == SMA exactly (perfectly flat series)."""
    sd = SlipDeficit(sma_window=50, vol_lookback=30)
    result = sd.compute(_flat_prices(n=100, base=100.0))
    assert result.slip_deficit == pytest.approx(0.0, abs=1e-6)


def test_positive_deficit_and_tighten_stops():
    """Price above SMA + recent vol spike → slip_deficit > 0, tighten_stops True.

    Strategy: 270 flat prices establish the baseline (vol ≈ 0), then 30 volatile
    prices spike upward. With ttf_threshold=0.0, any positive vol_zscore triggers
    tighten_stops. The price above SMA200 gives positive slip_deficit.
    """
    rng = np.random.default_rng(42)
    n_flat, n_volatile = 270, 30
    flat = np.full(n_flat, 100.0)
    factors = np.concatenate([[1.0], 1 + rng.normal(0.01, 0.12, n_volatile)])
    volatile = 100.0 * np.cumprod(factors)[1:]
    close = np.concatenate([flat, volatile])
    prices = pd.DataFrame({"close": close})

    sd = SlipDeficit(sma_window=200, ttf_threshold=0.0, vol_lookback=90)
    result = sd.compute(prices)
    assert result.slip_deficit > 0
    assert result.tighten_stops is True


def test_result_types_and_ttf_probability_range():
    """RiskOverlayResult fields are present, correctly typed, and ttf in [0, 1]."""
    sd = SlipDeficit(sma_window=50, vol_lookback=30)
    result = sd.compute(_flat_prices(n=100))
    assert isinstance(result, RiskOverlayResult)
    assert isinstance(result.slip_deficit, float)
    assert isinstance(result.ttf_probability, float)
    assert isinstance(result.tighten_stops, bool)
    assert 0.0 <= result.ttf_probability <= 1.0


def test_positions_none_accepted():
    """compute() accepts positions=None without error."""
    sd = SlipDeficit(sma_window=50, vol_lookback=30)
    sd.compute(_flat_prices(n=100), positions=None)  # must not raise


def test_positions_empty_list_accepted():
    """compute() accepts positions=[] without error."""
    sd = SlipDeficit(sma_window=50, vol_lookback=30)
    sd.compute(_flat_prices(n=100), positions=[])  # must not raise


def test_insufficient_history_raises_value_error():
    """Raises ValueError when price series is shorter than max(sma_window, vol_lookback)."""
    sd = SlipDeficit(sma_window=200, vol_lookback=90)
    short_prices = _flat_prices(n=50)  # need 200, have 50
    with pytest.raises(ValueError, match="Price series too short"):
        sd.compute(short_prices)
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
venv/Scripts/python -m pytest tests/test_research/test_slip_deficit.py -v
```

Expected: All 7 tests FAIL with `ModuleNotFoundError: No module named 'research.slip_deficit'`

- [ ] **Step 3: Implement `research/slip_deficit.py`**

```python
"""
slip_deficit.py — Seismic slip-deficit + Time-To-Failure overlay.

Additive risk overlay inspired by geophysical seismic cycle mechanics.
Tracks two signals:

1. Slip-deficit: How far price has drifted from its fundamental anchor (SMA-N).
   Formula: (current_price - SMA_N) / SMA_N
   Signed: positive = overextended above anchor (seismic "loading phase").
           negative = below anchor (oversold). Negative values are valid data.

2. TTF (Time-To-Failure) probability: Whether volatility "acoustic emissions"
   suggest imminent mean-reversion.
   - realized_vol_10d: annualized std of log-returns over the most recent 10
     trading days, computed as a rolling series across the full price history.
   - vol_zscore: Z-score of current vol relative to vol_lookback baseline.
   - ttf_probability: sigmoid(vol_zscore) — continuous [0, 1] signal.
   - tighten_stops: vol_zscore > ttf_threshold — direct Z-score comparison.
     ttf_threshold is ALWAYS interpreted in Z-score units (not probability).

Does NOT replace the existing risk engine — augments it with a forward-looking
overlay. Wiring into backtest_engine.py is Phase C work.

Behavior for insufficient history:
  Raises ValueError if len(prices_df) < max(sma_window, vol_lookback).
  Consistent with compute_atr_pct() in trading_system/risk_engine.py.

Reference: Docs 1 & 2, "Geophysics and Seismology" sections.
"""

from typing import Optional

import numpy as np
import pandas as pd

from research import BaseRiskOverlay, RiskOverlayResult


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    return float(1.0 / (1.0 + np.exp(-float(x))))


class SlipDeficit(BaseRiskOverlay):
    """Seismic slip-deficit and Time-To-Failure overlay."""

    def __init__(
        self,
        sma_window: int = 200,
        ttf_threshold: float = 2.0,
        vol_lookback: int = 90,
    ):
        """
        Args:
            sma_window:    SMA window for fundamental anchor. Default 200 (SMA-200).
            ttf_threshold: Vol Z-score threshold above which tighten_stops=True.
                           Always interpreted in Z-score units. Default 2.0 means
                           "vol is 2 standard deviations above its recent baseline."
            vol_lookback:  Rolling window (in trading days) for the vol Z-score
                           baseline. Default 90.
        """
        self.sma_window = sma_window
        self.ttf_threshold = ttf_threshold
        self.vol_lookback = vol_lookback

    def compute(
        self,
        prices_df: pd.DataFrame,
        positions: Optional[list] = None,
    ) -> RiskOverlayResult:
        """Compute slip-deficit and TTF signals from price history.

        Args:
            prices_df: DataFrame with a 'close' column, sorted ascending by date.
                       Must have at least max(sma_window, vol_lookback) rows.
            positions: Accepted but ignored — SlipDeficit is stateless with
                       respect to open positions. Included for ABC compliance.

        Returns:
            RiskOverlayResult with slip_deficit, ttf_probability, tighten_stops.

        Raises:
            ValueError: If prices_df has fewer than max(sma_window, vol_lookback) rows.
        """
        min_rows = max(self.sma_window, self.vol_lookback)
        if len(prices_df) < min_rows:
            raise ValueError(
                f"Price series too short: need {min_rows} rows "
                f"(max(sma_window={self.sma_window}, vol_lookback={self.vol_lookback})), "
                f"got {len(prices_df)} rows."
            )

        close = prices_df["close"]

        # --- Slip-deficit: signed divergence from SMA anchor ---
        sma = close.rolling(self.sma_window).mean().iloc[-1]
        current_price = float(close.iloc[-1])
        slip_deficit = (current_price - float(sma)) / float(sma)

        # --- TTF: volatility acoustic emissions ---
        # realized_vol_10d: annualized std of log-returns over 10 trading days
        log_returns = np.log(close / close.shift(1)).dropna()
        vol_series = log_returns.rolling(10).std() * np.sqrt(252)
        vol_series = vol_series.dropna()

        recent_vol = float(vol_series.iloc[-1])
        baseline = vol_series.iloc[-self.vol_lookback:]
        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std()) + 1e-8  # guard against zero-variance flat baseline

        vol_zscore = (recent_vol - baseline_mean) / baseline_std
        ttf_probability = _sigmoid(vol_zscore)
        tighten_stops = bool(vol_zscore > self.ttf_threshold)

        return RiskOverlayResult(
            slip_deficit=float(slip_deficit),
            ttf_probability=ttf_probability,
            tighten_stops=tighten_stops,
        )
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
venv/Scripts/python -m pytest tests/test_research/test_slip_deficit.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Verify full suite still passes**

```bash
venv/Scripts/python -m pytest tests/ -v --tb=short -q
```

Expected: 574 passed (556 original + 6 EMD + 5 BMA + 7 slip-deficit), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add research/slip_deficit.py tests/test_research/test_slip_deficit.py
git commit -m "feat(research): add SlipDeficit overlay with 7 passing smoke tests"
```

---

## Task 6: Phase C Roadmap

**Files:**
- Create: `research/phase_c_roadmap.md`

- [ ] **Step 1: Write `research/phase_c_roadmap.md`**

```markdown
# Phase C Research Integration Roadmap

Implement after Phase B modules (EMD, BMA, SlipDeficit) pass walk-forward
validation and the locked settings in CLAUDE.md are updated with experiment
evidence. Each domain has its own research module following the same ABC
pattern established in Phase B.

---

## Domain 1: FAISS/HNSW Approximate Nearest Neighbor

**What it is:**
Replaces the `ball_tree` NearestNeighbors index in `pattern_engine/matching.py`
with a Hierarchical Navigable Small World graph via FAISS. Enables sub-linear
scaling from 52 to 500+ tickers without sacrificing recall accuracy.

**Two-stage retrieval with Phase B EMD:**
- Stage 1 (FAISS): Scan full 25-year database in <10ms, retrieve top-100
  approximate nearest neighbours using L2 distance proxy with feature weights.
- Stage 2 (EMD): Re-rank the top-100 using the exact Earth Mover's Distance
  from `research/emd_distance.py`.

**Production integration point:**
`pattern_engine/matching.py` — `Matcher` class, `fit()` and `kneighbors()`
methods. Implements `BaseDistanceMetric`.

**Key hyperparameters:**
- `dimension=8`, `m_links=32`, `efConstruction=200`, `efSearch=64`
- Feature weights applied before indexing: `[1.0, 1.0, 1.5, 1.5, 1.0, 1.0, 1.0, 0.5]`

**New dependencies:** `faiss-cpu` (`pip install faiss-cpu`)

**Success metric:**
- Query latency < 10ms at 500-ticker scale
- Recall@50 ≥ 0.95 vs exact ball_tree on validation fold
- BSS on held-out fold ≥ Phase B EMD baseline

**Estimated complexity:** L (large) — requires database serialization, dynamic
insertion for new tickers, and integration with existing matching loop.

---

## Domain 2: Hawkes Process + Multiplex Contagion

**What it is:**
Augments `trading_system/risk_engine.py` with a self-exciting contagion model
across the 52-ticker universe. Models how a volatility shock in one ticker
propagates to correlated tickers through three distinct network layers.

**Multiplex network layers:**
1. Fundamental / Supply Chain: direct business relationships, shared regulatory
   jurisdiction (e.g., semiconductor maker → hardware OEM).
2. Financial / Credit Exposure: shared institutional ownership, counterparty risk,
   CDO linkages.
3. Statistical / Sentiment: ETF basket membership, algorithmic trading correlations,
   constructed via Planar Maximally Filtered Graph (PMFG) on tail dependence.

**Hawkes process:**
Continuously estimates the background intensity and self-exciting branching ratio
of recent high-frequency order flow proxies (bid-ask spread variance, volume
spikes). When branching ratio exceeds a threshold, the system flags "contagion
in progress" and restricts cross-sector position sizing.

**Production integration point:**
`trading_system/risk_engine.py` — new `contagion_check()` call before position
sizing. Implements `BaseRiskOverlay`.

**New dependencies:** `tick` or `hawkeslib` for Hawkes process estimation.

**Success metric:**
Detects historical contagion events (2008 GFC, 2020 COVID crash, 2022 rate shock)
with lead-time > 1 trading day on held-out walk-forward fold. False positive rate
< 10% on non-crisis periods.

**Estimated complexity:** L (large) — requires network construction, streaming
parameter updates, and careful handling of sparse LOB proxy data.

---

## Domain 3: OODA Loop + CPOD/EILOF Circuit Breakers

**What it is:**
Augments the execution layer in `trading_system/backtest_engine.py` with
real-time outlier detection algorithms that act as circuit breakers against
bad-tick contamination and flash-crash anomalies.

**Algorithms:**
- **CPOD (Continuous Pattern Outlier Detection):** Uses a "core point" data
  structure and multi-distance indexing. Up to 73× faster than traditional
  LOF. Identifies normal inliers, reducing neighbour search space for outlier
  validation. Flags anomalous price prints before they enter the matching pipeline.
- **EILOF (Extended Incremental Local Outlier Factor):** Incremental update
  variant of LOF. Maintains anomaly scores as the ticker universe grows without
  full recomputation.

**OODA execution loop:**
Observe (ingest OHLCV + LOB proxies) → Orient (FAISS query + EMD re-rank) →
Decide (BMA calibration + TTF threshold) → Act (signal routing or circuit-break).

**Production integration point:**
`trading_system/backtest_engine.py` — pre-signal-processing validation hook.
No new ABC needed — circuit breakers return a boolean `is_anomalous` flag.

**New dependencies:** None (pure numpy implementation viable for both algorithms).

**Success metric:**
Zero bad-tick contamination in backtest on injected synthetic spike test suite.
Detection latency < 1ms on synthetic spike injection. No false positives on
normal market days in 2024 validation fold.

**Estimated complexity:** M (medium) — CPOD/EILOF are well-documented algorithms;
primary complexity is integrating the circuit-breaker hook cleanly into the
existing backtest engine loop.

---

## Domain 4: Case-Based Reasoning + OWA Dynamic Feature Weighting

**What it is:**
Augments `pattern_engine/matching.py` to replace the static feature weights
(`ret_7d=1.5, ret_90d=0.5`) with regime-adaptive Ordered Weighted Averaging
(OWA). The weights recompute per-query based on the current macro regime state,
making the similarity metric context-aware rather than globally fixed.

**OWA operator:**
Applies varying weights to features based on their magnitude, sorted order,
and the current regime's "stress vector." During credit stress regimes,
spread-related features dominate. During momentum regimes, medium-term return
features dominate. The regime classification feeds directly from the existing
`pattern_engine/regime.py` 8-state binary regime filter.

**Case-based reasoning (CBR) lattice:**
Adapts the legal AI concept of a "claim lattice" to analogue ranking: retrieved
neighbours are not just sorted by distance but organized by which feature
dimensions they match on. This allows the engine to distinguish between a
"geometrically close but regime-mismatched" analogue and a "slightly further
but regime-aligned" one, surfacing the latter as more relevant.

**Production integration point:**
`pattern_engine/matching.py` — `apply_feature_weights()` function, called in
`Matcher.fit()` and query transform. Implements `BaseDistanceMetric` (weight
computation as a pre-processing step before the distance call).

**New dependencies:** None — pure numpy.

**Success metric:**
BSS improvement vs locked static weights on held-out 2022 regime-shift fold
(the period of most dramatic macro state transitions in the training data).
Improvement must be confirmed without data leakage (regime labels computed
from information available at query time only).

**Estimated complexity:** M (medium) — OWA computation is straightforward;
the complexity is in regime-conditioning the weight vector without look-ahead
bias.

---

## Promotion Gate (applies to all Phase C modules)

A Phase C module is ready for production promotion when:
1. Walk-forward BSS or Sharpe improvement confirmed on held-out fold
2. All 574 existing tests (556 production + 6 EMD + 5 BMA + 7 slip-deficit) still pass with
   the module wired in
3. Locked settings in `CLAUDE.md` updated with experiment evidence citation
4. `docs/superpowers/specs/` contains a spec for the Phase C module following
   the same review-loop process used for Phase B
```

- [ ] **Step 2: Commit**

```bash
git add research/phase_c_roadmap.md
git commit -m "docs(research): add Phase C roadmap with 4 deferred domain stubs"
```

---

## Task 7: Final Integration Verification

- [ ] **Step 1: Run the complete test suite**

```bash
venv/Scripts/python -m pytest tests/ -v --tb=short
```

Expected output summary:
```
574 passed, 0 failed
```
(556 original + 6 EMD + 5 BMA + 7 slip-deficit = 574)

If any test fails, diagnose before proceeding. Do not commit a broken suite.

- [ ] **Step 2: Verify the research package structure is complete**

```bash
venv/Scripts/python -c "
from research import BaseDistanceMetric, BaseCalibrator, BaseRiskOverlay, RiskOverlayResult
from research.emd_distance import EMDDistance
from research.bma_calibrator import BMACalibrator
from research.slip_deficit import SlipDeficit
print('All modules import cleanly.')

# ABC compliance spot-check
import numpy as np, pandas as pd
emd = EMDDistance().fit(np.zeros((1, 8)))
print('EMDDistance.compute:', emd.compute(np.zeros(8), np.zeros((2, 8))).shape)

bma = BMACalibrator(num_analogs=3)
print('BMACalibrator.fitted before fit:', bma.fitted)

sd = SlipDeficit(sma_window=50, vol_lookback=30)
r = sd.compute(pd.DataFrame({'close': [100.0] * 100}))
print('SlipDeficit result:', r)
"
```

Expected: All imports succeed, shapes and types are correct.

- [ ] **Step 3: Final commit if any uncommitted changes remain**

```bash
git status
# If clean, nothing to do. Otherwise:
git add -A
git commit -m "chore(research): final integration verification — 574 tests passing"
```

- [ ] **Step 4: Push branch**

Ask the user before pushing — this is a shared-state action.

```bash
# Only run with user approval:
git push -u origin phase35-research
```

---

## Success Checklist

Verify all items before declaring Phase B complete:

- [ ] `research/` package exists with `__init__.py`, `emd_distance.py`, `bma_calibrator.py`, `slip_deficit.py`, `phase_c_roadmap.md`
- [ ] `tests/test_research/` exists with `__init__.py`, `test_emd_distance.py`, `test_bma_calibrator.py`, `test_slip_deficit.py`
- [ ] All 574 tests pass: `venv/Scripts/python -m pytest tests/ -v -q`
- [ ] `pot` is installed and importable: `venv/Scripts/python -c "import ot; print(ot.__version__)"`
- [ ] All research modules import without error
- [ ] Branch is `phase35-research`
- [ ] Phase C roadmap document written with all 4 domains (description, integration point, deps, success metric, complexity)
