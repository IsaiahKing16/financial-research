# Phase 3.5 — Research Integration Design
**Date:** 2026-03-20
**Status:** Approved, pending implementation
**Branch:** `phase35-research` (create from `phase3-portfolio-manager`)

---

## Overview

Phase 3.5 integrates methodologies from four Gemini research papers into the FPPE codebase as
isolated, experimentally-validated research modules. The goal is to build validated alternatives
to three production components — the distance metric, probability calibrator, and risk overlay —
without touching any existing production code. Validated modules promote to production only when
walk-forward experiment evidence justifies unlocking the currently locked settings.

**Source papers:**
- `docs/research/Gemini FPPE Research/FPPE_ Cross-Domain Architecture Upgrade.docx`
- `docs/research/Gemini FPPE Research/Cross-Domain Financial Analogue Engine.docx`
- `docs/research/Gemini FPPE Research/Applying Military and Finance Data Strategies.docx`
- `docs/research/Gemini FPPE Research/Finance Psychology and FPPE Project.docx`

---

## Scope

### Phase B (this spec — implement now)
Three high-value research modules with correctness smoke tests:
1. Earth Mover's Distance (`emd_distance.py`)
2. Bayesian Model Averaging calibrator (`bma_calibrator.py`)
3. Seismic Slip-Deficit + Time-To-Failure overlay (`slip_deficit.py`)

### Phase C (roadmap stubs — implement after Phase B validated)
Four remaining domains deferred until Phase B modules pass walk-forward validation:
1. FAISS/HNSW Approximate Nearest Neighbor
2. Hawkes Process + Multiplex Contagion
3. OODA Loop + CPOD/EILOF Circuit Breakers
4. Case-Based Reasoning + OWA Dynamic Feature Weighting

See `research/phase_c_roadmap.md` for structured stubs.

---

## Constraints

- **No production code modified.** `matching.py`, `calibration.py`, `risk_engine.py`,
  `risk_state.py`, and all existing modules are read-only for this phase.
- **Locked settings unchanged.** Distance=Euclidean, Calibration=Platt, Features=returns_only(8)
  remain locked until walk-forward evidence from Phase C validates promotion.
- **All 556 existing tests must continue to pass** after each module is added.
- **New dependencies** are permitted: `pot` (Python Optimal Transport), `scipy` (already present),
  `numpy` (already present). FAISS is deferred to Phase C.
- **nn_jobs=1** preserved. No parallel execution introduced.

---

## Architecture

### Approach: Formal ABC Hierarchy

Abstract base classes defined in `research/__init__.py` enforce the interface contract each
research module must satisfy. When a module is ready for production promotion, it already
implements the exact interface the production module it replaces exposes. Note: callers in
`signal_adapter.py` currently invoke calibration via a legacy function pair
(`fit_platt_scaling` / `calibrate_probabilities`) rather than through any ABC seam.
Migration of those callers is Phase C promotion work, not Phase B.

### Package Structure

```
research/
├── __init__.py              # ABCs + RiskOverlayResult dataclass
├── emd_distance.py          # EMD distance metric (shadows matching.py)
├── bma_calibrator.py        # BMA calibrator (shadows calibration.py)
├── slip_deficit.py          # Slip-deficit + TTF overlay (augments risk_engine.py)
└── phase_c_roadmap.md       # Structured stubs for 4 deferred domains

tests/
└── test_research/
    ├── __init__.py
    ├── test_emd_distance.py
    ├── test_bma_calibrator.py
    └── test_slip_deficit.py
```

### Abstract Base Classes (`research/__init__.py`)

```python
class BaseDistanceMetric(ABC):
    @abstractmethod
    def fit(self, X_train: np.ndarray) -> "BaseDistanceMetric": ...
    @abstractmethod
    def compute(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray: ...

class BaseCalibrator(ABC):
    @abstractmethod
    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "BaseCalibrator": ...
    @abstractmethod
    def transform(self, raw_probs: np.ndarray) -> np.ndarray: ...
    @property
    @abstractmethod
    def fitted(self) -> bool: ...
    # generate_pdf() is NOT declared in BaseCalibrator. It is BMA-specific and will be
    # called through a concrete BMACalibrator reference in Phase C integration code.
    # If a future calibrator also exposes a PDF method, promote it to the ABC at that time.

class BaseRiskOverlay(ABC):
    # No fit() method: overlays are stateless and compute entirely from the price series
    # passed at call time. If a future overlay requires historical calibration, add fit()
    # to the ABC at that point (known asymmetry with BaseDistanceMetric / BaseCalibrator).
    @abstractmethod
    def compute(self, prices_df: pd.DataFrame,
                positions: Optional[list] = None) -> "RiskOverlayResult": ...

@dataclass
class RiskOverlayResult:
    slip_deficit: float   # Signed: (price - SMA200) / SMA200. Positive = above anchor
                          # (overextended/loaded); negative = below anchor (oversold).
                          # Seismic analogy maps to strictly non-negative loading, but the
                          # signed financial interpretation is intentional and correct.
    ttf_probability: float     # [0, 1] — probability of imminent failure
    tighten_stops: bool        # True when vol Z-score exceeds ttf_threshold
```

---

## Module Designs

### 1. `research/emd_distance.py` — Earth Mover's Distance

**What it is:** Optimal transport distance between two financial return fingerprints, treating
each fingerprint as a 2D weighted point set (time-lag × return-value). Handles temporal phase
shifts gracefully — a pattern that took 2 weeks in 2008 but 3 weeks in 2020 is penalized less
than a pattern with different structural shape.

**Research source:** Audio/NLP domain — melodic similarity via Wasserstein metric
(Docs 1 & 2, "Linguistics and Audio Processing" sections).

**ABC implemented:** `BaseDistanceMetric`

**Key parameters (constructor):**
- `time_penalty: float = 0.5` — down-weights temporal misalignment vs magnitude misalignment
- `price_penalty: float = 1.0` — weight on return-axis distance

**Data flow:**
```
8-feature fingerprint (ret_1d..ret_90d)
  → construct_weighted_point_set()
        time_lags = [1, 3, 7, 14, 30, 45, 60, 90]   # calendar-day lag per feature,
                                                      # matching RETURN_WINDOWS in features.py
        current_coords: shape (8, 2)  # each row: (time_lag * time_penalty,
                                      #            return_value * price_penalty)
        hist_coords:    shape (8, 2)
        weights_a = weights_b = np.full(8, 1/8)   # uniform; volume weights Phase C
  → cost_matrix = cdist(current_coords, hist_coords, metric='euclidean')
                                      # shape (8, 8) — required by ot.emd2
  → ot.emd2(weights_a, weights_b, cost_matrix)   # POT network simplex solver
  → scalar distance
```

**`fit()` semantics:** EMD has no training phase. `fit(X_train)` stores `X_train` for
optional normalization and returns `self`. For Phase B it is a no-op beyond storing the
reference. Future variants may use it to compute normalization statistics.

**Fallback:** If `pot` is unavailable, the fallback applies penalty weights by pre-scaling
the coordinate arrays before calling `scipy.stats.wasserstein_distance_nd(current_coords,
hist_coords)`. This is numerically equivalent to POT for the **uniform-weight case only**,
because `wasserstein_distance_nd` accepts no weights parameter. When Phase C adds
volume-based non-uniform weights, this fallback must be replaced with a pure-numpy
min-cost-flow implementation. Minimum SciPy version required: **1.13** (when
`wasserstein_distance_nd` was added). If SciPy < 1.13, raise `ImportError` with a clear
message directing the user to install `pot` instead.

**Production promotion path:** Replaces `ball_tree` distance computation in `matching.py`
`Matcher` class once walk-forward BSS improvement is confirmed.

**Design note:** Volume-weighted points (where liquidity = "mass") are the paper's recommended
enhancement. Deferred to Phase C when LOB/volume data is integrated into fingerprints.

---

### 2. `research/bma_calibrator.py` — Bayesian Model Averaging Calibrator

**What it is:** Replaces Platt scaling with a probabilistic ensemble calibration method.
Each historical analogue contributes a Student's t-distribution PDF weighted by its
historical predictive skill, optimized via Expectation-Maximization. Correctly models
heavy-tailed financial returns that Platt's Gaussian assumption underestimates.

**Research source:** Meteorological forecast post-processing — AnEn-CNN-BMA hybridization
(Doc 1, "Improving Baseline Probabilities" section; references Raftery et al. BMA paper).

**ABC implemented:** `BaseCalibrator` — `transform()` matches `PlattCalibrator` interface
exactly. Drop-in compatibility with production callers requires Phase C migration of
`signal_adapter.py`'s legacy `fit_platt_scaling`/`calibrate_probabilities` functions to
call through the ABC. This wiring is deferred to Phase C.

**Key parameters (constructor):**
- `num_analogs: int` — number of historical analogues (typically top_k=50)
- `df: float = 3.0` — Student's t degrees of freedom (lower = heavier tails)
- `max_iter: int = 50` — EM convergence iterations

**Input/output semantics:**
- `fit(raw_probs, y_true)`:
  - `raw_probs`: shape `(N_training_samples, num_analogs)` — the raw KNN probability from
    each of the K nearest neighbours across all training samples. `analog[i]` for sample n
    is `raw_probs[n, i]`. Analogue locations are **fixed** throughout EM; only weights and
    variances are updated.
  - `y_true`: shape `(N_training_samples,)` — binary outcome (1=up, 0=not-up)
- `transform(raw_probs)`:
  - `raw_probs`: shape `(num_analogs,)` — the K raw probs for a single query point
  - Returns: scalar = `np.dot(self.weights, raw_probs)` — posterior-weighted mean of the
    K analogue probabilities. This is the BMA point forecast, guaranteed in [0, 1] as a
    convex combination of [0, 1] values. Matches `PlattCalibrator.transform()` output type.

**EM Algorithm:**

The implementation uses a Gaussian variance M-step as a deliberate approximation of the
full Student's t EM (which requires an auxiliary u-weight precision variable). This is
acceptable for a research module: the t-distribution is used in the E-step PDF (providing
heavy-tail probability mass), while the simpler Gaussian variance update avoids the
u-weight complexity at the cost of slight variance underestimation in extreme tails. A
comment in the code must document this tradeoff explicitly. The full Student's t M-step
(with u-weights) is a Phase C upgrade candidate.

```
E-step: latent_pdfs[n, i] = w[i] * t_pdf(y[n] | loc=raw_probs[n,i], scale=sqrt(var[i]), df=df)
        latent_probs[n, i] = latent_pdfs[n, i] / sum(latent_pdfs[n, :])   # normalize

M-step: w[i]   = mean(latent_probs[:, i])                                 # weight update
        var[i] = sum(latent_probs[:, i] * (y - raw_probs[:,i])^2)         # Gaussian approx
                 / sum(latent_probs[:, i])                                 # (see note above)
```

**Additional method:** `generate_pdf(analogue_probs: np.ndarray, return_grid: np.ndarray) -> np.ndarray`
- `analogue_probs`: shape `(num_analogs,)` — raw probs for the current query
- `return_grid`: shape `(M,)` — grid of return values at which to evaluate the PDF
- Returns: shape `(M,)` — BMA mixture density at each grid point
Used for confidence interval computation in the three-filter gate (Phase C integration).
Called through a concrete `BMACalibrator` reference, not through `BaseCalibrator`.

**Production promotion path:** Replaces `PlattCalibrator` in `calibration.py` once
BSS improvement confirmed. Requires Phase C migration of `signal_adapter.py` callers.

---

### 3. `research/slip_deficit.py` — Seismic Slip-Deficit + Time-To-Failure

**What it is:** Additive risk overlay inspired by geophysical seismic cycle mechanics.
Tracks two signals: (1) how far price has drifted from its fundamental anchor (slip-deficit),
and (2) whether volatility "acoustic emissions" suggest imminent mean-reversion (TTF).
Does not replace the existing risk engine — augments it with a forward-looking overlay.

**Research source:** Geophysics/seismology domain — seismic loading cycles and laboratory
earthquake forecasting (Docs 1 & 2, "Geophysics and Seismology" sections).

**ABC implemented:** `BaseRiskOverlay`

**Key parameters (constructor):**
- `sma_window: int = 200` — fundamental anchor (SMA-200)
- `ttf_threshold: float = 2.0` — vol Z-score threshold above which `tighten_stops=True`
- `vol_lookback: int = 90` — rolling window for volatility baseline

**Behavior for insufficient history:** If `len(prices_df) < max(sma_window, vol_lookback)`,
raise `ValueError` with message `"Price series too short: need {max(sma_window,
vol_lookback)} rows, got {len} rows"`. Both `sma_window` and `vol_lookback` are independent
parameters, so the guard uses their maximum to avoid a degenerate vol Z-score when
`vol_lookback > sma_window`. Consistent with `compute_atr_pct()` in `risk_engine.py`.

**Computation:**
```
slip_deficit     = (current_price - SMA(sma_window)) / SMA(sma_window)
                   # signed: positive = overextended above anchor (loaded)
                   # negative = below anchor (oversold)

# realized_vol_10d: annualized std of log-returns over the most recent 10 trading days
#   = std(log(close[t]/close[t-1]) for t in last 10 days) * sqrt(252)
# This is computed as a rolling series across the full price history.
# The Z-score baseline is the mean and std of that rolling series over vol_lookback periods.
vol_series       = rolling_annualized_vol(prices_df['close'], window=10)  # shape (N,)
vol_zscore       = (vol_series.iloc[-1] - vol_series.iloc[-vol_lookback:].mean())
                   / vol_series.iloc[-vol_lookback:].std()

ttf_probability  = sigmoid(vol_zscore)       # [0,1] continuous probability
tighten_stops    = vol_zscore > ttf_threshold # direct Z-score comparison
                                              # semantically clear: ttf_threshold is
                                              # always interpreted as a Z-score
```

**Integration intent:** `backtest_engine.py` can optionally call `SlipDeficit.compute()`
per trading day. When `tighten_stops=True`, the ATR multiplier in `risk_engine.py` is
reduced by a configurable scalar. This wiring is Phase C work — for now the module
computes and returns results independently.

**Data requirements:** Standard OHLCV price series already available in the system.
No new data sources required.

---

## Test Design (`tests/test_research/`)

All tests use synthetic numpy arrays. No real market data, no file I/O, no fixtures.
All run within the existing `pytest` suite without special configuration.

### `test_emd_distance.py`
| Test | Input | Expected |
|------|-------|----------|
| Init + shape | dummy (8,) arrays | no error, returns scalar |
| Identical distributions | same fingerprint twice | distance ≈ 0.0 |
| Directional ordering | close vs far pairs | EMD(close) < EMD(far) |
| time_penalty=0.0 | sets differing only in time axis | distance ≈ 0.0 |
| price_penalty=0.0 | sets differing only in return axis | distance ≈ 0.0 |

### `test_bma_calibrator.py`
| Test | Input | Expected |
|------|-------|----------|
| Init | num_analogs=5 | `fitted == False` |
| Post-fit state | fit on synthetic data | `fitted == True`, `weights.sum() ≈ 1.0` |
| Output range | transform() | all outputs in [0, 1] |
| Uniform convergence | identical analogs, same y | weights ≈ equal |
| Two-cluster convergence | half analogs at 0.0, half at 1.0 | each cluster weight ≈ 0.5 (catches weight collapse bugs) |

### `test_slip_deficit.py`
| Test | Input | Expected |
|------|-------|----------|
| Init | sma_window=200 | no error |
| Zero deficit | price == SMA200 exactly | `slip_deficit == 0.0` |
| Positive deficit + stops | price >> SMA200, high vol | `slip_deficit > 0`, `tighten_stops == True` |
| Result types | any valid input | all `RiskOverlayResult` fields present and typed correctly |
| positions=None | passed to compute() | no error (positions is optional and unused) |
| positions=[] | passed to compute() | no error — implement as two separate test functions or one `@pytest.mark.parametrize` |
| Insufficient history | len(prices_df) < max(sma_window, vol_lookback) | raises `ValueError` with message citing required row count |

---

## Dependencies

| Library | Purpose | Install |
|---------|---------|---------|
| `pot` | EMD exact solver (network simplex) | `pip install pot` |
| `scipy` | Wasserstein fallback, t-distribution | already present |
| `numpy` | Array operations throughout | already present |

FAISS (`faiss-cpu`) deferred to Phase C.

---

## Phase C Roadmap

See `research/phase_c_roadmap.md` for structured stubs covering:
1. FAISS/HNSW Approximate Nearest Neighbor
2. Hawkes Process + Multiplex Contagion
3. OODA Loop + CPOD/EILOF Circuit Breakers
4. Case-Based Reasoning + OWA Dynamic Feature Weighting

---

## Success Criteria for Phase B

- [ ] `research/` package created with ABCs and all 3 modules
- [ ] All 3 modules pass ABC interface compliance (instantiation + method signatures)
- [ ] All smoke tests pass (17 tests total: 5 EMD + 5 BMA + 7 slip-deficit)
- [ ] All 556 existing tests continue to pass
- [ ] `pot` added to project dependencies
- [ ] Phase C roadmap document written at `research/phase_c_roadmap.md` with all 4 domains,
      each containing: description, production integration point, new dependencies, success
      metric, and estimated complexity rating (S/M/L)
- [ ] Implementation committed on branch `phase35-research`

## Promotion Criteria (Phase C gate)

A module is ready for production promotion when:
1. Walk-forward BSS or Sharpe improvement confirmed on held-out fold
2. All existing tests still pass with module wired in
3. Locked settings updated in `CLAUDE.md` with experiment evidence citation
