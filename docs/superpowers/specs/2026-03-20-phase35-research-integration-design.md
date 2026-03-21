# Phase 3.5 — Research Integration Design
**Date:** 2026-03-20
**Status:** Approved, pending implementation
**Branch:** phase3-portfolio-manager (to be implemented on new branch)

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
implements the exact interface the production caller expects — zero changes to callers required.

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

class BaseRiskOverlay(ABC):
    @abstractmethod
    def compute(self, prices_df: pd.DataFrame, positions: list) -> "RiskOverlayResult": ...

@dataclass
class RiskOverlayResult:
    slip_deficit: float        # (price - SMA200) / SMA200
    ttf_probability: float     # [0, 1] — probability of imminent failure
    tighten_stops: bool        # True when TTF probability exceeds threshold
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
  → construct_weighted_point_set()     # shape: (8, 2) — (time_lag, return_value)
  → weights = uniform (1/8 each)      # volume weights deferred to Phase C (LOB data)
  → ot.emd2(current_weights,          # exact EMD via POT network simplex solver
             hist_weights,
             cost_matrix)             # pairwise euclidean on (time, return) coords
  → scalar distance
```

**Fallback:** If `pot` unavailable, falls back to `scipy.stats.wasserstein_distance_nd`.

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
exactly for drop-in compatibility with the three-filter gate in `signal_adapter.py`.

**Key parameters (constructor):**
- `num_analogs: int` — number of historical analogues (typically top_k=50)
- `df: float = 3.0` — Student's t degrees of freedom (lower = heavier tails)
- `max_iter: int = 50` — EM convergence iterations

**EM Algorithm:**
```
E-step: latent_probs[i] = w[i] * t_pdf(realized_return | analog[i], var[i])
        normalized across all analogs
M-step: w[i]   = mean(latent_probs[:, i])
        var[i] = sum(latent_probs[:, i] * (y - analog[i])^2) / sum(latent_probs[:, i])
```

**Additional method:** `generate_pdf(analogue_returns, return_grid) -> np.ndarray`
Returns the full BMA probability density function — used for confidence interval
computation in the three-filter gate (Phase C integration).

**Production promotion path:** Replaces `PlattCalibrator` in `calibration.py` once
BSS improvement confirmed. Zero changes to `signal_adapter.py` callers required.

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
- `ttf_threshold: float = 2.0` — Z-score threshold above which `tighten_stops=True`
- `vol_lookback: int = 90` — rolling window for volatility baseline

**Computation:**
```
slip_deficit     = (current_price - SMA(sma_window)) / SMA(sma_window)
vol_zscore       = (realized_vol_10d - mean(realized_vol, vol_lookback))
                   / std(realized_vol, vol_lookback)
ttf_probability  = sigmoid(vol_zscore)
tighten_stops    = ttf_probability > sigmoid(ttf_threshold)
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

### `test_bma_calibrator.py`
| Test | Input | Expected |
|------|-------|----------|
| Init | num_analogs=5 | `fitted == False` |
| Post-fit state | fit on synthetic data | `fitted == True`, `weights.sum() ≈ 1.0` |
| Output range | transform() | all outputs in [0, 1] |
| Uniform convergence | identical analogs, same y | weights ≈ equal |

### `test_slip_deficit.py`
| Test | Input | Expected |
|------|-------|----------|
| Init | sma_window=200 | no error |
| Zero deficit | price == SMA200 exactly | `slip_deficit == 0.0` |
| Positive deficit + stops | price >> SMA200, high vol | `slip_deficit > 0`, `tighten_stops == True` |
| Result types | any valid input | all `RiskOverlayResult` fields present and typed correctly |

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
- [ ] All smoke tests pass (12 tests total across 3 files)
- [ ] All 556 existing tests continue to pass
- [ ] `pot` added to project dependencies
- [ ] Phase C roadmap document written
- [ ] Implementation committed on dedicated branch

## Promotion Criteria (Phase C gate)

A module is ready for production promotion when:
1. Walk-forward BSS or Sharpe improvement confirmed on held-out fold
2. All existing tests still pass with module wired in
3. Locked settings updated in `CLAUDE.md` with experiment evidence citation
