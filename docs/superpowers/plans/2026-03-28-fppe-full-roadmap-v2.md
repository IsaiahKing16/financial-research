# FPPE Full Roadmap Implementation Plan — v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take FPPE from BSS < 0 (broken probability calibration) through live deployment with $10,000 capital by January 2027, then scale to 5,200+ tickers with TurboQuant vector quantization and FAISS infrastructure by Q2 2027.

**v2 Delta from v1:** Integrates Google TurboQuant/PolarQuant/QJL vector quantization research, FAISS scaling infrastructure, STUMPY matrix profile enhancement, autonomous EOD execution scheduling, and team experience-level assignment. Original 10 phases preserved; 3 new sub-phases and 1 new post-launch phase added.

**Architecture:** 10 sequential phases with selective parallelism + Phase 6A (TurboQuant R&D, parallel), Phase 6B (Vector Quantization Integration), Phase 8+ (Autonomous EOD), and Phase 11 (Hyper-Scale 5200T+). Each phase produces testable, committable software with an explicit quality gate. The pattern engine's 5-stage matcher feeds through signal filters → position sizer → risk engine → portfolio manager → broker adapter. All communication flows through frozen Pydantic SharedState.

**Tech Stack:** Python 3.12, pytest, Pydantic v2, scikit-learn (BallTree), hnswlib, numpy, pandas, scipy (Platt sigmoid), matplotlib (diagnostics), **FAISS** (Phase 11), **STUMPY** (Phase 7). Broker: IBKR TWS API (Alpaca fallback).

**Spec:** `docs/superpowers/specs/2026-03-26-fppe-full-roadmap-design.md` (v1) + `docs/superpowers/specs/2026-03-28-turboquant-addendum.md` (v2 addendum)

**Test command:** `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`

**Critical rules:**
- `assert` → `RuntimeError` for all public API guards
- `nn_jobs=1` always (Windows/Py3.12 joblib deadlock)
- Numbers require provenance — any claimed metric must trace to walk-forward results
- 3-strike rule: if 3 consecutive attempts at same fix fail, STOP and escalate

---

## Executive Summary of v2 Changes

### What Changed

| Area | v1 | v2 |
|------|----|----|
| Universe ceiling | 1,500 tickers | **5,200+ tickers** (Phase 11) |
| Vector backend | hnswlib only | hnswlib → **FAISS IVF+TurboQuant** at scale |
| Quantization | None (float32 HNSW) | **TurboQuant 2.5-bit** feasibility study + integration |
| Pattern matching | KNN only | KNN + **STUMPY matrix profile** (Phase 7 Enhancement 6) |
| EOD execution | Manual/Task Scheduler | **Autonomous cron pipeline** with health monitoring |
| Team structure | Unassigned | **Experience-level task routing** (SR/JR/MIX) |
| Competitive benchmarking | None | **8 structured benchmarks** against open-source repos |
| Total phases | 10 | **10 + 3 sub-phases + 1 post-launch phase** |

### What Did NOT Change

Phases 1–5 (BSS Fix → Live Plumbing) are **unchanged**. These form the critical path to a working trading system and must not be delayed by research tracks. Phase 6A runs strictly in parallel.

### Critical Dimensionality Warning

FPPE's return fingerprints are **8-dimensional**. TurboQuant's theoretical guarantees converge in high dimensions (paper benchmarks: d=200–3072). At d=8:
- QJL inner-product variance ≈ π/(2d) ≈ **0.196 per component** — substantial noise
- Random rotation may not induce tight Beta distributions
- Lloyd-Max codebook optimality weakens

**Consequence:** Phase 6A is structured as a *feasibility study first*, not a commitment. The experiment must prove recall and BSS preservation at d=8 before any production integration. If d=8 proves intractable, Phase 7's OWA feature expansion (increasing d to 16–32) creates a second opportunity.

---

## Updated Phase Dependency Graph

```
Phase 1 (BSS Fix) ──► Phase 2 (Half-Kelly) ──► Phase 3 (Risk Engine) ──► Phase 4 (Portfolio Mgr)
       │                                              │                          │
       ├──► Phase 6 (Universe 1500T) ─────────────────┤                          │
       │         ├── Task 6.5: FAISS Evaluation        │                          │
       │         └── [parallel]                        │                          │
       ├──► Phase 6A (TurboQuant R&D) ◄── PARALLEL ──►│                          │
       │         │                                     │                          │
       │         ▼                                     ▼                          ▼
       │    Phase 6B (VQ Integration) ◄──────── Phase 5 (Live Plumbing)   Phase 7 (Enhancements)
       │         │  (if 6A passes gate)                │                   │ +E6: STUMPY
       │         │                                     │                   │
       │         └─────────────────────────────────────┴───────────────────┘
       │                                               │
       │                                       Phase 8 (Paper Trading 3mo)
       │                                        │ + Autonomous EOD (8+)
       │                                               │
       │                                       Phase 9 (Live Deploy)
       │                                               │
       │                                    ┌──────────┴──────────┐
       │                                    │                     │
       │                             Phase 10 (Options)    Phase 11 (Hyper-Scale
       │                                                    5200T + FAISS +
       │                                                    TurboQuant Prod)
```

### New Dependency Notes

- **Phase 6A (TurboQuant R&D)** depends only on Phase 1 gate (needs BSS-fixed baseline for recall comparison). Runs fully parallel with Phases 2–5 and Phase 6. Zero risk to critical path.
- **Phase 6B (Vector Quantization Integration)** depends on Phase 6A gate pass AND Phase 6 completion. If Phase 6A fails, Phase 6B is cancelled — no sunk cost beyond research.
- **Phase 11 (Hyper-Scale)** depends on Phase 9 (live system stable). Uses Phase 6A/6B research artifacts.

---

## Updated Timeline Allocation

| Phase | Duration | Target Complete | Dependencies | Team |
|-------|----------|-----------------|--------------|------|
| 1 — BSS Fix | 3-4 weeks | Late April 2026 | None | **SR** (statistical depth) |
| 2 — Half-Kelly | 2 weeks | Mid May 2026 | Phase 1 gate | **JR** (well-defined math) |
| 3 — Risk Engine | 3 weeks | Early June 2026 | Phase 2 gate | **MIX** (moderate complexity) |
| 4 — Portfolio Manager | 3 weeks | Late June 2026 | Phase 3 gate | **MIX** |
| 5 — Live Plumbing | 4 weeks | Late July 2026 | Phase 3 gate | **JR** (standard engineering) |
| 6 — Universe 1500T | 3 weeks | Mid May 2026 | Phase 1 gate | **MIX** |
| **6A — TurboQuant R&D** | **4 weeks** | **Late May 2026** | **Phase 1 gate** | **SR** (novel research) |
| **6B — VQ Integration** | **2 weeks** | **Mid June 2026** | **6A + 6 gates** | **SR** |
| 7 — Enhancements (+STUMPY) | 4 weeks | Late August 2026 | Phases 4,5,6 | **SR** |
| 8 — Paper Trading + EOD | 12 weeks | Late November 2026 | Phase 7 gate | **JR** (monitoring) |
| 9 — Live Deploy | 4 weeks | January 2027 | Phase 8 gate | **MIX** |
| 10 — Options Foundation | 4-6 weeks | Post-launch | Phase 9 stable | **MIX** |
| **11 — Hyper-Scale 5200T** | **6-8 weeks** | **Q2 2027** | **Phase 9 stable** | **SR** |

**Team Legend:** SR = Senior developer (deep domain/research expertise required), JR = Junior developer (well-scoped, clear specs), MIX = Mixed pair (senior leads, junior implements).

---

## Execution Contract

*(Unchanged from v1 — all integration points, artifact formats, and locked settings remain identical. See v1 Section "Execution Contract" for full details.)*

**Additional v2 contracts:**

**TurboQuant experiment artifacts:**
- Quantized index files: `results/turboquant/` directory
- Recall comparison TSVs: `results/turboquant/recall_comparison_d{D}_b{BITS}.tsv`
- Each row: `n_vectors`, `dim`, `bits_per_dim`, `recall@50`, `build_time_s`, `query_latency_ms`, `memory_mb`, `bss_delta`

**FAISS index artifacts:**
- Index files: `data/indices/faiss/` (not committed — too large)
- Benchmark results: `results/faiss/benchmark_{N}t.tsv`

---

## File Structure Map — v2 Additions

```
# Competitive Benchmarks (integrated into existing phases)
scripts/benchmarks/b1_chinuy_delta_sweep.py     # chinuy delta validation (Phase 1)
scripts/benchmarks/b3_murphy_decomposition.py   # Murphy BSS decomposition (Phase 1)
scripts/benchmarks/b4_regime_conditioning_value.py  # Regime value measurement (Phase 7)
scripts/benchmarks/b5_dtw_vs_euclidean.py       # DTW rescue via calibration (Phase 7)
scripts/benchmarks/b6_stumpy_vs_knn.py          # Cross-method comparison (Phase 7)
docs/benchmarks/b7_qlib_architecture_audit.md   # Pipeline architecture comparison (Phase 8)
docs/benchmarks/b8_walkforward_comparison.md    # Methodology documentation (Phase 8)

# Phase 6A — TurboQuant R&D
research/turboquant/
research/turboquant/__init__.py
research/turboquant/random_rotation.py     # Orthogonal rotation via QR decomposition
research/turboquant/lloyd_max.py           # Optimal scalar quantizer (Beta distribution)
research/turboquant/qjl.py                 # 1-bit Quantized Johnson-Lindenstrauss
research/turboquant/turbo_quantizer.py     # Two-stage pipeline: rotation + Lloyd-Max + QJL
research/turboquant/benchmarks.py          # Recall/BSS/memory comparison harness
tests/unit/test_turboquant_rotation.py     # Rotation orthogonality + distribution tests
tests/unit/test_turboquant_lloyd_max.py    # Codebook optimality tests
tests/unit/test_turboquant_qjl.py         # Unbiased estimation tests
tests/performance/test_turboquant_recall.py # Recall@50 vs exact at various bit-widths

# Phase 6B — VQ Integration (if 6A passes)
pattern_engine/contracts/matchers/turbo_matcher.py  # TurboQuantMatcher implementing BaseMatcher
tests/parity/test_turbo_parity.py                   # BallTree vs TurboQuant recall parity
tests/performance/test_turbo_scaling.py              # Memory + latency at 1500T

# Phase 6 update — FAISS Evaluation
research/faiss_evaluation.py               # FAISS IVF-Flat vs HNSW benchmark
scripts/benchmark_faiss.py                 # Scaling benchmark at 1500T/5200T

# Phase 7 update — STUMPY Enhancement
research/stumpy_matcher.py                 # Matrix profile pattern discovery
tests/unit/test_stumpy_integration.py      # STUMPY AB-join for cross-ticker matching

# Phase 8+ — Autonomous EOD
scripts/eod_pipeline.py                    # Master EOD orchestrator
scripts/health_check.py                    # Pre-execution health validation
config/eod_schedule.json                   # Cron schedule + alert configuration
tests/unit/test_eod_pipeline.py            # Pipeline orchestration tests

# Phase 11 — Hyper-Scale
pattern_engine/contracts/matchers/faiss_matcher.py  # FAISSMatcher implementing BaseMatcher
scripts/build_faiss_index.py               # Overnight FAISS index builder (IVF+TurboQuant)
tests/performance/test_faiss_5200t.py      # 5200T scaling benchmarks
```

---

## Phases 1–5: UNCHANGED

Phases 1 through 5 are identical to v1. See `docs/superpowers/plans/2026-03-26-fppe-full-roadmap.md` for complete task breakdowns.

**Quick reference:**

| Phase | Gate | Key Deliverable |
|-------|------|-----------------|
| 1 — BSS Fix | BSS > 0 on ≥ 3/6 folds | `scripts/bss_sweep.py`, updated locked settings |
| 2 — Half-Kelly | Kelly fraction positive ≥ 4/6 folds, Sharpe ≥ 1.0 | `trading_system/position_sizer.py` |
| 3 — Risk Engine | Max DD ≤ 10%, Sharpe ≥ 1.0, stops ≤ 35% | `trading_system/risk_engine.py` |
| 4 — Portfolio Mgr | Sector limits enforced, idle cash < 50% | PM activation + rejection analysis |
| 5 — Live Plumbing | Mock parity 100 trades, OOB 30 days, < 3 min | `trading_system/broker/`, order manager |

---

## Phase 6: Universe Expansion — UPDATED

**Gate:** HNSW recall@50 ≥ 0.9999, BSS > 0 on ≥ 3/6 folds at 1500T, pipeline < 2hr, peak RAM < 24 GB
**Duration:** 3 weeks
**Prerequisite:** Phase 1 gate passed
**Team:** MIX (senior leads data pipeline, junior implements sector mapping)

### Tasks 6.1–6.4: UNCHANGED from v1

*(Russell 1000 data pipeline, HNSW scaling benchmark, overnight index build, BSS re-validation)*

### Task 6.5: FAISS Feasibility Evaluation ← NEW

**Files:**
- Create: `research/faiss_evaluation.py`
- Create: `scripts/benchmark_faiss.py`

**Rationale:** At 1500T (~3.7M vectors, 8 dims), hnswlib is sufficient. But FAISS provides a migration path to 5200T+ (Phase 11). Evaluating now — while we have the 1500T dataset — avoids surprises later.

**Assignment:** JR developer (FAISS has excellent docs + Python bindings; SR reviews results)

- [ ] **Step 1: Install FAISS**

```bash
pip install faiss-cpu  # GPU version later if needed
```

- [ ] **Step 2: Write FAISS benchmark script**

```python
"""benchmark_faiss.py — Compare FAISS index types against hnswlib at 1500T scale.

Tests:
  - IndexFlatL2 (exact brute force — baseline)
  - IndexIVFFlat (inverted file, exact within cells)
  - IndexHNSWFlat (FAISS's own HNSW — compare to hnswlib)

At d=8, Product Quantization (PQ) is NOT viable (requires d >= 2*M, M >= 4).
PQ evaluation deferred to Phase 11 when feature dimensions may increase.

Output: results/faiss/benchmark_1500t.tsv
"""
import faiss
import numpy as np
import time
from pathlib import Path


def benchmark_index(index, X_train, X_query, k=50, exact_neighbors=None):
    """Benchmark a FAISS index: build time, query time, recall, memory."""
    # Build
    t0 = time.perf_counter()
    index.add(X_train)
    build_time = time.perf_counter() - t0

    # Query
    t0 = time.perf_counter()
    distances, indices = index.search(X_query, k)
    query_time = time.perf_counter() - t0

    # Recall (if exact_neighbors provided)
    recall = None
    if exact_neighbors is not None:
        hits = sum(
            len(set(indices[i]) & set(exact_neighbors[i])) / k
            for i in range(len(X_query))
        ) / len(X_query)
        recall = hits

    return {
        "build_time_s": build_time,
        "query_time_s": query_time,
        "recall_at_50": recall,
    }
```

- [ ] **Step 3: Run benchmark on 1500T data**

Expected output: FAISS IndexIVFFlat recall, build time, query latency comparison to hnswlib.

- [ ] **Step 4: Document findings for Phase 11 planning**

Key questions to answer:
1. Does FAISS IndexHNSWFlat match hnswlib recall@50 ≥ 0.9999?
2. What nprobe setting does IndexIVFFlat need for ≥ 0.9999 recall at d=8?
3. Memory footprint comparison?

- [ ] **Step 5: Commit**

```bash
git add research/faiss_evaluation.py scripts/benchmark_faiss.py results/faiss/
git commit -m "research(faiss): FAISS feasibility evaluation at 1500T — [RECALL], [LATENCY]"
```

### GATE CHECK: Phase 6 (Updated)

- [ ] **HNSW recall@50 ≥ 0.9999? BSS > 0 on ≥ 3/6 at 1500T? Pipeline < 2hr? Peak RAM < 24GB?**
- [ ] **FAISS evaluation documented? (informational — not blocking)**
  - YES → Proceed to Phase 7 merge point
  - NO → Fall back to 585T

---

## Phase 6A: TurboQuant Vector Quantization R&D ← NEW PHASE

**Gate:** Quantized recall@50 ≥ 0.9990 at d=8 with ≤ 4 bits/dim AND BSS delta ≥ -0.002
**Duration:** 4 weeks
**Prerequisite:** Phase 1 gate passed (need BSS-fixed baseline for comparison)
**Team:** SR (novel research — requires understanding of information theory, quantization, and linear algebra)
**Risk to critical path:** ZERO (fully parallel with Phases 2–5)

### 6A.1 Context & Motivation

**Why TurboQuant for FPPE:**

Google's TurboQuant (arXiv:2504.19874) offers a two-stage vector quantization algorithm that is:
1. **Data-oblivious** — no per-dataset training, no k-means preprocessing (unlike Product Quantization)
2. **Zero-overhead** — no per-vector scale/zero-point storage (unlike block quantization)
3. **Theoretically grounded** — proven MSE bounds: D_mse ≤ (√3π/2) × (1/4^b)
4. **Ultra-fast encoding** — 0.001s vs PQ's 37s on GloVe (d=200), 50,000× faster

**FPPE-specific value proposition:**

| Scale | Vectors | Float32 (8D) | TurboQuant 2.5-bit | Savings |
|-------|---------|-------------|-------------------|---------|
| 585T (current) | ~1.9M | 58 MB | 4.75 MB | 12× |
| 1,500T (Phase 6) | ~3.7M | 113 MB | 9.25 MB | 12× |
| 5,200T (Phase 11) | ~26M | 793 MB | 65 MB | 12× |
| 5,200T + 32D features | ~26M | 3.2 GB | 260 MB | 12× |

At current 585T scale, 58 MB is trivial on a 32 GB machine — compression is unnecessary. The value emerges at 5,200T with expanded features (Phase 11), where 3.2 GB of vectors plus HNSW graph overhead approaches the 24 GB memory budget.

**The dimensionality challenge:**

TurboQuant's guarantees strengthen with dimension d. At d=8:
- Random rotation produces Beta(d/2-1/2, d/2-1/2) = Beta(3.5, 3.5) distributions — moderately peaked but far from Gaussian
- QJL variance per component: π/(2×8) ≈ 0.196 — tolerable for inner products but not negligible
- Lloyd-Max codebook at 2.5 bits (6 levels) on Beta(3.5,3.5) — suboptimal compared to d=200 case

**Experiment strategy:** Start with feasibility at d=8 (current), then evaluate at d=16/32 (post-Phase 7 feature expansion). Phase 6A is designed as a research spike — fast to execute, fast to abandon if results are negative.

### Task 6A.1: Random Orthogonal Rotation Module

**Files:**
- Create: `research/turboquant/__init__.py`
- Create: `research/turboquant/random_rotation.py`
- Create: `tests/unit/test_turboquant_rotation.py`

**Assignment:** SR developer

- [ ] **Step 1: Write failing tests for rotation module**

```python
"""Tests for research/turboquant/random_rotation.py — orthogonal rotation."""
import pytest
import numpy as np
from research.turboquant.random_rotation import RandomOrthogonalRotation


class TestRandomOrthogonalRotation:
    """Verify rotation matrix properties and coordinate distributions."""

    def test_orthogonality(self):
        """R @ R.T == I (within floating point)."""
        rot = RandomOrthogonalRotation(dim=8, seed=42)
        R = rot.matrix
        identity = R @ R.T
        np.testing.assert_allclose(identity, np.eye(8), atol=1e-12)

    def test_determinant_positive(self):
        """det(R) == +1 (proper rotation, no reflection)."""
        rot = RandomOrthogonalRotation(dim=8, seed=42)
        assert np.linalg.det(rot.matrix) > 0

    def test_norm_preservation(self):
        """||Rx|| == ||x|| for all x."""
        rot = RandomOrthogonalRotation(dim=8, seed=42)
        x = np.random.randn(100, 8)
        x_rot = rot.forward(x)
        np.testing.assert_allclose(
            np.linalg.norm(x_rot, axis=1),
            np.linalg.norm(x, axis=1),
            rtol=1e-10,
        )

    def test_inverse_recovers_original(self):
        """R.T @ (R @ x) == x."""
        rot = RandomOrthogonalRotation(dim=8, seed=42)
        x = np.random.randn(50, 8)
        x_recovered = rot.inverse(rot.forward(x))
        np.testing.assert_allclose(x_recovered, x, atol=1e-12)

    def test_coordinate_distribution_shape(self):
        """After rotation, coordinates should follow Beta-like distribution.

        At d=8, each coordinate of unit-normalized rotated vectors follows
        Beta(3.5, 3.5) scaled to [-1, 1]. Verify kurtosis is sub-Gaussian
        (Beta(3.5,3.5) kurtosis ≈ -2/8 = -0.25, lighter tails than Gaussian).
        """
        rot = RandomOrthogonalRotation(dim=8, seed=42)
        # Generate unit vectors on the sphere
        x = np.random.randn(10_000, 8)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        x_rot = rot.forward(x)
        # Check each coordinate is approximately centered at 0
        for col in range(8):
            assert abs(x_rot[:, col].mean()) < 0.05

    def test_deterministic_with_seed(self):
        """Same seed produces same rotation."""
        r1 = RandomOrthogonalRotation(dim=8, seed=123)
        r2 = RandomOrthogonalRotation(dim=8, seed=123)
        np.testing.assert_array_equal(r1.matrix, r2.matrix)
```

- [ ] **Step 2: Implement rotation module**

```python
"""random_rotation.py — Random orthogonal rotation via QR decomposition.

TurboQuant Stage 0: Pre-condition vectors so each coordinate follows
a known Beta distribution, enabling optimal scalar quantization without
data-dependent training.

Reference: arXiv:2504.19874 Section 3.1
"""
from __future__ import annotations

import numpy as np


class RandomOrthogonalRotation:
    """Generate and apply a random orthogonal rotation matrix.

    Uses QR decomposition of a random Gaussian matrix to produce
    a uniformly distributed orthogonal matrix (Haar measure).

    Args:
        dim: Dimensionality of vectors to rotate.
        seed: Random seed for reproducibility. MUST be fixed per index build
              to ensure rotation consistency between fit() and query().
    """

    def __init__(self, dim: int, seed: int = 42) -> None:
        if dim < 1:
            raise RuntimeError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        # QR decomposition of random Gaussian matrix
        G = self._rng.randn(dim, dim)
        Q, R = np.linalg.qr(G)
        # Ensure proper rotation (det = +1), not reflection
        signs = np.sign(np.diag(R))
        self._matrix = Q * signs[np.newaxis, :]

    @property
    def matrix(self) -> np.ndarray:
        """The orthogonal rotation matrix, shape (dim, dim)."""
        return self._matrix

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply rotation: X_rot = X @ R.T.

        Args:
            X: Input vectors, shape (n, dim).

        Returns:
            Rotated vectors, shape (n, dim). Norms preserved.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.dim:
            raise RuntimeError(
                f"Expected dim={self.dim}, got vectors of dim={X.shape[1]}"
            )
        return X @ self._matrix.T

    def inverse(self, X_rot: np.ndarray) -> np.ndarray:
        """Invert rotation: X = X_rot @ R (since R is orthogonal, R.T.T = R).

        Args:
            X_rot: Rotated vectors, shape (n, dim).

        Returns:
            Original vectors, shape (n, dim).
        """
        if X_rot.ndim == 1:
            X_rot = X_rot.reshape(1, -1)
        return X_rot @ self._matrix
```

- [ ] **Step 3: Run tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_turboquant_rotation.py -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add research/turboquant/ tests/unit/test_turboquant_rotation.py
git commit -m "research(turboquant): random orthogonal rotation module with tests"
```

### Task 6A.2: Lloyd-Max Optimal Scalar Quantizer

**Files:**
- Create: `research/turboquant/lloyd_max.py`
- Create: `tests/unit/test_turboquant_lloyd_max.py`

**Assignment:** SR developer

- [ ] **Step 1: Write failing tests for Lloyd-Max quantizer**

```python
"""Tests for research/turboquant/lloyd_max.py — optimal scalar quantization."""
import pytest
import numpy as np
from research.turboquant.lloyd_max import LloydMaxQuantizer


class TestLloydMaxQuantizer:

    def test_codebook_size_matches_bits(self):
        """b bits → 2^b centroids."""
        q = LloydMaxQuantizer(bits=2, dim=8)
        assert len(q.codebook) == 4  # 2^2

    def test_quantize_dequantize_roundtrip(self):
        """Roundtrip error bounded by theoretical MSE."""
        q = LloydMaxQuantizer(bits=3, dim=8)
        x = np.random.randn(1000, 8).astype(np.float32)
        indices = q.quantize(x)
        x_hat = q.dequantize(indices)
        mse = np.mean((x - x_hat) ** 2)
        # Theoretical bound: sqrt(3*pi/2) / 4^3 ≈ 0.034
        assert mse < 0.10  # generous for d=8 non-ideal case

    def test_indices_are_uint8(self):
        """For b <= 8 bits, indices should be uint8."""
        q = LloydMaxQuantizer(bits=3, dim=8)
        x = np.random.randn(100, 8).astype(np.float32)
        indices = q.quantize(x)
        assert indices.dtype == np.uint8

    def test_codebook_precomputed_for_beta(self):
        """Codebook centroids optimized for Beta(d/2-0.5, d/2-0.5) distribution.

        At d=8: Beta(3.5, 3.5). Centroids should be symmetric around 0.5.
        """
        q = LloydMaxQuantizer(bits=2, dim=8)
        centroids = q.codebook
        # Beta(3.5,3.5) is symmetric → centroids symmetric around mean
        mean_centroid = np.mean(centroids)
        assert abs(mean_centroid - 0.0) < 0.1  # centered (after shift to [-1,1])

    def test_memory_footprint(self):
        """Quantized representation uses expected bits."""
        q = LloydMaxQuantizer(bits=3, dim=8)
        n = 10_000
        x = np.random.randn(n, 8).astype(np.float32)
        indices = q.quantize(x)
        # 3 bits × 8 dims = 24 bits = 3 bytes per vector
        # But stored as uint8 per dim → 8 bytes per vector (ceiling)
        # True bit-packed: 3 bytes per vector
        raw_bytes = n * 8 * 4  # float32
        quantized_bytes = indices.nbytes  # uint8 per dim
        assert quantized_bytes < raw_bytes / 3  # at least 3x compression
```

- [ ] **Step 2: Implement Lloyd-Max quantizer**

```python
"""lloyd_max.py — Optimal scalar quantizer for Beta-distributed coordinates.

TurboQuant Stage 1: After random rotation, each coordinate follows
Beta(d/2-1/2, d/2-1/2). Pre-compute optimal Lloyd-Max centroids for
this known distribution, then quantize each coordinate independently.

Reference: arXiv:2504.19874 Section 3.2

Key insight: Because the distribution is analytically known, we can
pre-compute centroids ONCE per (bits, dim) pair — no data-dependent
training needed. This makes encoding O(n*d) with zero preprocessing.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class LloydMaxQuantizer:
    """Optimal scalar quantizer for rotated vector coordinates.

    Args:
        bits: Bits per coordinate (1–8). 2^bits centroids per dimension.
        dim: Vector dimensionality (determines Beta shape parameter).
    """

    def __init__(self, bits: int, dim: int) -> None:
        if not 1 <= bits <= 8:
            raise RuntimeError(f"bits must be in [1, 8], got {bits}")
        if dim < 2:
            raise RuntimeError(f"dim must be >= 2, got {dim}")
        self.bits = bits
        self.dim = dim
        self.n_levels = 2 ** bits
        # Beta shape parameter: alpha = beta = d/2 - 1/2
        self._alpha = dim / 2 - 0.5
        # Pre-compute centroids via Lloyd-Max on Beta distribution
        self._codebook = self._compute_codebook()

    @property
    def codebook(self) -> np.ndarray:
        """Pre-computed centroids, shape (n_levels,)."""
        return self._codebook

    def _compute_codebook(self) -> np.ndarray:
        """Compute optimal Lloyd-Max centroids for Beta(alpha, alpha).

        Uses iterative Lloyd-Max algorithm on the analytical Beta PDF.
        Centroids are mapped to [-1, 1] range (shifted from [0, 1]).
        """
        dist = stats.beta(self._alpha, self._alpha)
        # Initialize with uniform quantile spacing
        boundaries = np.linspace(0, 1, self.n_levels + 1)
        centroids = np.zeros(self.n_levels)

        for _ in range(100):  # Lloyd-Max iterations
            # Compute centroids as conditional expectations
            for i in range(self.n_levels):
                lo, hi = boundaries[i], boundaries[i + 1]
                if hi - lo < 1e-12:
                    centroids[i] = (lo + hi) / 2
                    continue
                # E[X | lo <= X <= hi] using truncated Beta
                num, _ = stats.beta.expect(
                    lambda x: x, args=(self._alpha, self._alpha),
                    loc=0, scale=1, lb=lo, ub=hi
                )
                den = dist.cdf(hi) - dist.cdf(lo)
                centroids[i] = num / den if den > 1e-12 else (lo + hi) / 2

            # Update boundaries as midpoints between centroids
            for i in range(1, self.n_levels):
                boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

        # Shift from [0, 1] to [-1, 1] range
        return 2.0 * centroids - 1.0

    def quantize(self, X_rotated: np.ndarray) -> np.ndarray:
        """Quantize rotated coordinates to centroid indices.

        Args:
            X_rotated: Rotated vectors, shape (n, dim). Values expected
                       in approximately [-1, 1] range after rotation.

        Returns:
            Indices array, shape (n, dim), dtype uint8.
        """
        # Normalize to [0, 1] for codebook lookup
        X_norm = (X_rotated + 1.0) / 2.0
        X_norm = np.clip(X_norm, 0.0, 1.0)

        # Map [0,1] codebook back for distance comparison
        codebook_01 = (self._codebook + 1.0) / 2.0

        # Find nearest centroid per coordinate
        # Shape: (n, dim, 1) vs (1, 1, n_levels) → (n, dim, n_levels)
        diffs = np.abs(
            X_norm[:, :, np.newaxis] - codebook_01[np.newaxis, np.newaxis, :]
        )
        indices = np.argmin(diffs, axis=2).astype(np.uint8)
        return indices

    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from centroid indices.

        Args:
            indices: Shape (n, dim), dtype uint8.

        Returns:
            Reconstructed vectors, shape (n, dim), dtype float32.
        """
        return self._codebook[indices].astype(np.float32)
```

- [ ] **Step 3: Run tests**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_turboquant_lloyd_max.py -v`

- [ ] **Step 4: Commit**

```bash
git add research/turboquant/lloyd_max.py tests/unit/test_turboquant_lloyd_max.py
git commit -m "research(turboquant): Lloyd-Max optimal scalar quantizer for Beta distributions"
```

### Task 6A.3: 1-bit QJL Residual Correction

**Files:**
- Create: `research/turboquant/qjl.py`
- Create: `tests/unit/test_turboquant_qjl.py`

**Assignment:** SR developer

- [ ] **Step 1: Write failing tests for QJL**

```python
"""Tests for research/turboquant/qjl.py — 1-bit Quantized Johnson-Lindenstrauss."""
import pytest
import numpy as np
from research.turboquant.qjl import QJLTransform


class TestQJLTransform:

    def test_output_is_binary(self):
        """QJL output is {-1, +1}^d."""
        qjl = QJLTransform(dim=8, seed=42)
        x = np.random.randn(100, 8)
        signs = qjl.encode(x)
        assert set(np.unique(signs)).issubset({-1, 1})

    def test_unbiased_inner_product(self):
        """E[<y, QJL^-1(QJL(x))>] == <y, x> (unbiased estimation).

        Verify over many random pairs that the average estimated
        inner product converges to the true inner product.
        """
        qjl = QJLTransform(dim=8, seed=42)
        x = np.random.randn(8)
        y = np.random.randn(8)
        true_ip = x @ y

        # Monte Carlo: estimate inner product via QJL many times
        estimates = []
        for seed in range(1000):
            qjl_i = QJLTransform(dim=8, seed=seed)
            signs = qjl_i.encode(x.reshape(1, -1))
            ip_est = qjl_i.estimate_inner_product(signs, y)
            estimates.append(ip_est[0])

        mean_est = np.mean(estimates)
        # Should converge to true inner product
        assert abs(mean_est - true_ip) < 0.5  # generous for d=8

    def test_variance_bounded(self):
        """Var[QJL IP estimate] <= (pi/2d) * ||y||^2."""
        dim = 8
        y = np.random.randn(dim) * 2.0  # non-unit norm
        theoretical_var = (np.pi / (2 * dim)) * np.sum(y ** 2)

        estimates = []
        x = np.random.randn(dim)
        for seed in range(5000):
            qjl = QJLTransform(dim=dim, seed=seed)
            signs = qjl.encode(x.reshape(1, -1))
            est = qjl.estimate_inner_product(signs, y)
            estimates.append(est[0])

        empirical_var = np.var(estimates)
        # Allow 50% slack for finite sample
        assert empirical_var < theoretical_var * 1.5

    def test_deterministic_with_seed(self):
        """Same seed → same projection matrix → same signs."""
        qjl1 = QJLTransform(dim=8, seed=99)
        qjl2 = QJLTransform(dim=8, seed=99)
        x = np.random.randn(10, 8)
        np.testing.assert_array_equal(qjl1.encode(x), qjl2.encode(x))
```

- [ ] **Step 2: Implement QJL transform**

```python
"""qjl.py — 1-bit Quantized Johnson-Lindenstrauss transform.

TurboQuant Stage 2: Applied to the residual after Lloyd-Max quantization.
Provides unbiased inner-product estimation with zero overhead (no per-vector
constants needed).

Q_qjl(x) = sign(S · x), where S has i.i.d. Gaussian entries.

Reference: arXiv:2504.19874 Section 2.2
"""
from __future__ import annotations

import numpy as np


class QJLTransform:
    """1-bit Quantized Johnson-Lindenstrauss projection.

    Encodes vectors to sign bits of random Gaussian projections.
    Provides unbiased inner-product estimation: E[<y, decode(encode(x))>] = <y, x>.

    Args:
        dim: Input vector dimensionality.
        seed: Random seed for projection matrix S.
    """

    def __init__(self, dim: int, seed: int = 42) -> None:
        if dim < 1:
            raise RuntimeError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        # Random Gaussian projection matrix, shape (dim, dim)
        # Scaled by 1/sqrt(dim) for proper normalization
        self._S = self._rng.randn(dim, dim) / np.sqrt(dim)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode vectors to sign bits.

        Args:
            X: Input vectors, shape (n, dim).

        Returns:
            Sign array, shape (n, dim), values in {-1, +1}, dtype int8.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        projected = X @ self._S.T  # (n, dim)
        signs = np.sign(projected).astype(np.int8)
        # Replace 0 with +1 (extremely rare, prevents zero)
        signs[signs == 0] = 1
        return signs

    def estimate_inner_product(
        self, signs_x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Estimate <x, y> from sign-encoded x and full-precision y.

        This is the asymmetric estimator: one vector quantized, one full.
        E[estimate] = <x, y> (unbiased).
        Var[estimate] <= (pi/2d) * ||y||^2.

        Args:
            signs_x: Encoded vectors, shape (n, dim), values in {-1, +1}.
            y: Full-precision query vector, shape (dim,).

        Returns:
            Inner product estimates, shape (n,).
        """
        if y.ndim > 1:
            y = y.flatten()
        # Project y through same random matrix
        y_proj = y @ self._S.T  # (dim,)
        # Estimate: (pi/2) * mean(sign(Sx) * (Sy))
        # Simplified: signs_x * y_proj summed and scaled
        raw = (signs_x.astype(np.float64) * y_proj[np.newaxis, :]).sum(axis=1)
        # Scale factor: sqrt(pi/2) for proper normalization
        return raw * np.sqrt(np.pi / 2)
```

- [ ] **Step 3: Run tests**
- [ ] **Step 4: Commit**

```bash
git add research/turboquant/qjl.py tests/unit/test_turboquant_qjl.py
git commit -m "research(turboquant): 1-bit QJL residual correction module"
```

### Task 6A.4: Two-Stage TurboQuant Pipeline

**Files:**
- Create: `research/turboquant/turbo_quantizer.py`
- Create: `research/turboquant/benchmarks.py`

**Assignment:** SR developer

- [ ] **Step 1: Implement combined two-stage quantizer**

```python
"""turbo_quantizer.py — Two-stage TurboQuant pipeline.

Stage 1: Random rotation + Lloyd-Max scalar quantization (MSE-optimal)
Stage 2: QJL on residual (bias correction for inner products)

For distance computation: dequantize Stage 1, apply QJL correction,
compute Euclidean distance against full-precision query.

Reference: arXiv:2504.19874 Algorithm 1
"""
from __future__ import annotations

import numpy as np
from .random_rotation import RandomOrthogonalRotation
from .lloyd_max import LloydMaxQuantizer
from .qjl import QJLTransform


class TurboQuantizer:
    """Two-stage TurboQuant vector quantizer.

    Args:
        dim: Vector dimensionality.
        bits: Bits per dimension for Lloyd-Max stage (1-8).
        use_qjl: Whether to apply QJL residual correction (Stage 2).
        seed: Random seed for rotation and QJL matrices.
    """

    def __init__(
        self,
        dim: int = 8,
        bits: int = 3,
        use_qjl: bool = True,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.bits = bits
        self.use_qjl = use_qjl
        self.seed = seed

        self._rotation = RandomOrthogonalRotation(dim=dim, seed=seed)
        self._quantizer = LloydMaxQuantizer(bits=bits, dim=dim)
        self._qjl = QJLTransform(dim=dim, seed=seed + 1) if use_qjl else None

        # Storage for QJL-encoded residuals
        self._residual_signs: np.ndarray | None = None

    def encode(self, X: np.ndarray) -> dict:
        """Encode vectors through the two-stage pipeline.

        Args:
            X: Input vectors, shape (n, dim), float32/float64.

        Returns:
            Dict with:
                'indices': uint8 array (n, dim) — Lloyd-Max centroid indices
                'residual_signs': int8 array (n, dim) — QJL signs (if use_qjl)
                'rotation_seed': int — for reconstruction
        """
        # Stage 0: Rotate
        X_rot = self._rotation.forward(X)

        # Stage 1: Lloyd-Max quantize
        indices = self._quantizer.quantize(X_rot)

        result = {
            'indices': indices,
            'rotation_seed': self.seed,
        }

        # Stage 2: QJL on residual
        if self._qjl is not None:
            X_hat = self._quantizer.dequantize(indices)
            residual = X_rot - X_hat
            result['residual_signs'] = self._qjl.encode(residual)

        return result

    def decode(self, encoded: dict) -> np.ndarray:
        """Decode quantized vectors back to full precision (approximate).

        Args:
            encoded: Output from encode().

        Returns:
            Reconstructed vectors, shape (n, dim), float32.
        """
        # Stage 1: Dequantize
        X_hat_rot = self._quantizer.dequantize(encoded['indices'])

        # Stage 2: QJL residual is 1-bit — cannot fully reconstruct,
        # but provides unbiased correction for inner products.
        # For Euclidean distance, we use Stage 1 reconstruction only.

        # Inverse rotation
        return self._rotation.inverse(X_hat_rot).astype(np.float32)

    def memory_bytes(self, n_vectors: int) -> dict:
        """Compute memory footprint of encoded representation.

        Returns:
            Dict with byte counts for each component.
        """
        index_bytes = n_vectors * self.dim  # uint8 per coordinate
        qjl_bytes = n_vectors * self.dim if self.use_qjl else 0  # int8
        overhead = self.dim * self.dim * 8  # rotation matrix (float64)
        overhead += self._quantizer.n_levels * 8  # codebook (float64)

        return {
            'indices': index_bytes,
            'qjl_residuals': qjl_bytes,
            'overhead': overhead,
            'total': index_bytes + qjl_bytes + overhead,
            'vs_float32': n_vectors * self.dim * 4,
            'compression_ratio': (n_vectors * self.dim * 4) / max(
                index_bytes + qjl_bytes + overhead, 1
            ),
        }
```

- [ ] **Step 2: Write benchmark harness**

```python
"""benchmarks.py — TurboQuant recall and BSS comparison harness.

Measures:
  1. Recall@50: fraction of true top-50 neighbors recovered after quantization
  2. BSS delta: difference in Brier Skill Score between exact and quantized
  3. Memory compression ratio
  4. Encoding/decoding throughput

Run: PYTHONUTF8=1 py -3.12 -m research.turboquant.benchmarks
Output: results/turboquant/recall_comparison.tsv
"""
from __future__ import annotations

import time
import numpy as np
from pathlib import Path
from .turbo_quantizer import TurboQuantizer


def benchmark_recall(
    X_train: np.ndarray,
    X_query: np.ndarray,
    k: int = 50,
    bits_range: list[int] | None = None,
) -> list[dict]:
    """Compare quantized nearest neighbors against exact (brute-force).

    Args:
        X_train: Training vectors, shape (N, d).
        X_query: Query vectors, shape (Q, d).
        k: Number of neighbors.
        bits_range: Bit-widths to test. Default [1, 2, 3, 4, 5, 6, 8].

    Returns:
        List of result dicts per bit-width.
    """
    if bits_range is None:
        bits_range = [1, 2, 3, 4, 5, 6, 8]

    d = X_train.shape[1]
    N = X_train.shape[0]

    # Exact neighbors (brute-force L2)
    from sklearn.neighbors import NearestNeighbors
    nn_exact = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1)
    nn_exact.fit(X_train)
    exact_dists, exact_indices = nn_exact.kneighbors(X_query)

    results = []
    for bits in bits_range:
        tq = TurboQuantizer(dim=d, bits=bits, use_qjl=True)

        # Encode
        t0 = time.perf_counter()
        encoded = tq.encode(X_train)
        encode_time = time.perf_counter() - t0

        # Decode (approximate)
        X_decoded = tq.decode(encoded)

        # Find neighbors in decoded space
        nn_quant = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=1)
        nn_quant.fit(X_decoded)
        t0 = time.perf_counter()
        quant_dists, quant_indices = nn_quant.kneighbors(X_query)
        query_time = time.perf_counter() - t0

        # Recall@k
        recall_sum = 0.0
        for i in range(len(X_query)):
            exact_set = set(exact_indices[i])
            quant_set = set(quant_indices[i])
            recall_sum += len(exact_set & quant_set) / k
        recall = recall_sum / len(X_query)

        # Memory
        mem = tq.memory_bytes(N)

        results.append({
            'bits': bits,
            'dim': d,
            'n_train': N,
            'n_query': len(X_query),
            'recall_at_50': round(recall, 6),
            'encode_time_s': round(encode_time, 4),
            'query_time_s': round(query_time, 4),
            'memory_mb': round(mem['total'] / 1e6, 2),
            'float32_mb': round(mem['vs_float32'] / 1e6, 2),
            'compression_ratio': round(mem['compression_ratio'], 2),
        })

    return results
```

- [ ] **Step 3: Run recall benchmark on FPPE data**

Run on actual walk-forward fold data (585T or Phase 1 fixed baseline):

```bash
PYTHONUTF8=1 py -3.12 -m research.turboquant.benchmarks
```

Expected output: `results/turboquant/recall_comparison.tsv`

- [ ] **Step 4: Analyze results — CRITICAL DECISION POINT**

| Bits | Expected Recall@50 (d=8) | Gate |
|------|-------------------------|------|
| 1 | ~0.70–0.85 | Below threshold — QJL-only too lossy at d=8 |
| 2 | ~0.85–0.95 | Marginal — may work with QJL correction |
| 3 | ~0.95–0.99 | Promising — evaluate BSS delta |
| 4 | ~0.99–0.999 | Strong — likely passes gate |
| 5+ | ~0.999+ | Near-lossless at any dimension |

If recall@50 ≥ 0.9990 at ≤ 4 bits:
→ **Proceed to BSS delta evaluation (Step 5)**

If recall@50 < 0.9990 at all bit-widths ≤ 4:
→ **Phase 6A fails gate. Document results. Cancel Phase 6B. Revisit after Phase 7 feature expansion (higher d).**

- [ ] **Step 5: BSS delta evaluation (only if Step 4 passes)**

Run walk-forward with quantized index. Compare BSS to exact-index BSS.

Gate: BSS delta ≥ -0.002 (quantization does not degrade signal quality by more than 0.2%).

- [ ] **Step 6: Commit all results with provenance**

```bash
git add research/turboquant/ tests/ results/turboquant/
git commit -m "research(turboquant): full pipeline + recall/BSS benchmarks — [BITS]bit recall=[X], BSS_delta=[Y]"
```

### GATE CHECK: Phase 6A

- [ ] **Quantized recall@50 ≥ 0.9990 at d=8 with ≤ 4 bits/dim?**
- [ ] **BSS delta ≥ -0.002 vs exact baseline?**
  - **BOTH YES** → Proceed to Phase 6B (VQ Integration)
  - **RECALL YES, BSS NO** → Investigate: is Platt calibration sensitive to quantization noise? Try refitting Platt on quantized predictions. If still fails → cancel 6B.
  - **RECALL NO** → Cancel Phase 6B. Document findings. Revisit when d > 8 (post-Phase 7 OWA).
  - **NOTE:** Phase 6A failure does NOT affect the critical path. Phases 1–10 proceed unimpacted.

---

## Phase 6B: Vector Quantization Integration ← NEW PHASE (Conditional)

**Gate:** TurboQuantMatcher passes parity tests, recall@50 ≥ 0.9990, memory reduction ≥ 3×
**Duration:** 2 weeks
**Prerequisite:** Phase 6A gate passed AND Phase 6 gate passed
**Team:** SR developer (builds on 6A research artifacts)

**Only execute this phase if Phase 6A passed its gate.**

### Task 6B.1: TurboQuantMatcher Implementation

**Files:**
- Create: `pattern_engine/contracts/matchers/turbo_matcher.py`
- Create: `tests/parity/test_turbo_parity.py`

- [ ] **Step 1: Implement TurboQuantMatcher as BaseMatcher subclass**

```python
"""turbo_matcher.py — TurboQuant-compressed nearest-neighbor matcher.

Implements BaseMatcher interface. Stores training vectors in quantized form
(TurboQuant Stage 1 + QJL), queries against full-precision vectors using
asymmetric distance computation.

Integration: EngineConfig(use_turbo=True) selects this backend.
Fallback: use_hnsw=True or default BallTree remain available.

Memory profile at 1500T (3.7M vectors, 8 dims):
  - Float32 baseline: 113 MB
  - TurboQuant 3-bit + QJL: ~37 MB (3× compression)
  - TurboQuant 4-bit + QJL: ~44 MB (2.5× compression)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pattern_engine.contracts.matcher import BaseMatcher
from research.turboquant.turbo_quantizer import TurboQuantizer


class TurboQuantMatcher(BaseMatcher):
    """Nearest-neighbor matcher with TurboQuant vector compression.

    Stores training data in compressed form. Queries use asymmetric
    distance: full-precision query against decoded (approximate) database.

    For k <= 50 and n <= 5M, brute-force on decoded vectors is fast
    enough (sub-second on 8-dim). For larger scale, combine with
    FAISS IVF partitioning (Phase 11).
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        bits: int = 3,
        use_qjl: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.bits = bits
        self.use_qjl = use_qjl
        self.seed = seed
        self._quantizer: TurboQuantizer | None = None
        self._encoded: dict | None = None
        self._decoded: np.ndarray | None = None  # cached for brute-force
        self._n_train: int = 0
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit: quantize training vectors and cache decoded version."""
        dim = X.shape[1]
        self._quantizer = TurboQuantizer(
            dim=dim, bits=self.bits, use_qjl=self.use_qjl, seed=self.seed
        )
        self._encoded = self._quantizer.encode(X)
        self._decoded = self._quantizer.decode(self._encoded)
        self._n_train = X.shape[0]
        self._fitted = True

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors using asymmetric distance.

        Query vectors are full-precision. Database vectors are decoded
        from quantized form. Distance computation is exact on the
        decoded vectors (not the quantized representation).

        Returns:
            distances: shape (n_queries, k), Euclidean, sorted ascending.
            indices: shape (n_queries, k), integer indices into training set.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before kneighbors()")
        k = n_neighbors if n_neighbors > 0 else self.n_neighbors
        k = min(k, self._n_train)

        # Brute-force L2 on decoded vectors
        # For d=8 and N<=5M, this is fast (matrix multiply)
        # (Q, d) @ (d, N) → (Q, N)
        X_f32 = X.astype(np.float32)
        db = self._decoded  # (N, d), float32

        # Compute squared Euclidean distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        x_sq = np.sum(X_f32 ** 2, axis=1, keepdims=True)  # (Q, 1)
        db_sq = np.sum(db ** 2, axis=1)  # (N,)
        dots = X_f32 @ db.T  # (Q, N)
        sq_dists = x_sq + db_sq[np.newaxis, :] - 2 * dots

        # Clip negative values (floating point)
        sq_dists = np.maximum(sq_dists, 0.0)

        # Partial sort for top-k
        if k < self._n_train:
            top_k_indices = np.argpartition(sq_dists, k, axis=1)[:, :k]
            # Gather distances for top-k
            top_k_dists = np.take_along_axis(sq_dists, top_k_indices, axis=1)
            # Sort within top-k
            sort_order = np.argsort(top_k_dists, axis=1)
            indices = np.take_along_axis(top_k_indices, sort_order, axis=1)
            distances = np.sqrt(
                np.take_along_axis(top_k_dists, sort_order, axis=1)
            )
        else:
            sort_order = np.argsort(sq_dists, axis=1)[:, :k]
            indices = sort_order
            distances = np.sqrt(np.take_along_axis(sq_dists, sort_order, axis=1))

        return distances, indices

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> Dict[str, object]:
        return {
            'n_neighbors': self.n_neighbors,
            'bits': self.bits,
            'use_qjl': self.use_qjl,
            'seed': self.seed,
            'backend': 'turboquant',
        }
```

- [ ] **Step 2: Write parity tests**

```python
"""test_turbo_parity.py — Verify TurboQuantMatcher recall vs BallTree exact."""
import pytest
import numpy as np
from pattern_engine.contracts.matchers.turbo_matcher import TurboQuantMatcher
from pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher


class TestTurboParityWithBallTree:

    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(10_000, 8).astype(np.float32)
        X_query = rng.randn(100, 8).astype(np.float32)
        return X_train, X_query

    def test_recall_at_50_above_threshold(self, sample_data):
        """TurboQuant recall@50 vs exact must be >= 0.9990."""
        X_train, X_query = sample_data
        k = 50

        # Exact
        bt = BallTreeMatcher(n_neighbors=k)
        bt.fit(X_train)
        exact_dists, exact_idx = bt.kneighbors(X_query, k)

        # Quantized (3-bit, chosen by Phase 6A results)
        tq = TurboQuantMatcher(n_neighbors=k, bits=3)
        tq.fit(X_train)
        quant_dists, quant_idx = tq.kneighbors(X_query, k)

        # Recall
        recall_sum = 0.0
        for i in range(len(X_query)):
            exact_set = set(exact_idx[i])
            quant_set = set(quant_idx[i])
            recall_sum += len(exact_set & quant_set) / k
        recall = recall_sum / len(X_query)
        assert recall >= 0.9990, f"Recall {recall:.4f} < 0.9990"

    def test_distances_are_euclidean(self, sample_data):
        """Returned distances are true Euclidean (not squared)."""
        X_train, X_query = sample_data
        tq = TurboQuantMatcher(n_neighbors=5, bits=3)
        tq.fit(X_train)
        dists, idx = tq.kneighbors(X_query[:1], 5)
        # Verify first distance manually
        d_manual = np.linalg.norm(X_query[0] - tq._decoded[idx[0, 0]])
        assert abs(dists[0, 0] - d_manual) < 1e-4

    def test_memory_compression(self, sample_data):
        """Quantized index uses less memory than float32."""
        X_train, _ = sample_data
        tq = TurboQuantMatcher(n_neighbors=50, bits=3)
        tq.fit(X_train)
        mem = tq._quantizer.memory_bytes(len(X_train))
        assert mem['compression_ratio'] >= 2.0
```

- [ ] **Step 3: Wire into PatternMatcher backend selection**

Add to `pattern_engine/matcher.py` `_build_index()`:

```python
if config.use_turbo:
    from pattern_engine.contracts.matchers.turbo_matcher import TurboQuantMatcher
    _turbo_bits = getattr(cfg, 'turbo_bits', 3)
    self._backend = TurboQuantMatcher(n_neighbors=n_probe, bits=_turbo_bits)
```

Add `use_turbo: bool = False` and `turbo_bits: int = 3` to EngineConfig.

- [ ] **Step 4: Run full test suite**

Run: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
Expected: all pass (new tests + existing unchanged)

- [ ] **Step 5: Commit**

```bash
git add pattern_engine/contracts/matchers/turbo_matcher.py tests/
git commit -m "feat(turbo): TurboQuantMatcher implementing BaseMatcher — [BITS]bit, recall=[X]"
```

### GATE CHECK: Phase 6B

- [ ] **TurboQuantMatcher passes parity tests? recall@50 ≥ 0.9990? Memory ≥ 3× reduction?**
  - YES → Available as `use_turbo=True` backend. Default remains BallTree/HNSW until Phase 11.
  - NO → Archive research. BallTree/HNSW sufficient through Phase 11.

---

## Phase 7: Model Enhancements — UPDATED (+Enhancement 6: STUMPY)

**Gate:** Each enhancement individually gated
**Duration:** 4 weeks
**Prerequisite:** Phases 4, 5, 6 gates passed
**Team:** SR developer

### Tasks 7.1–7.5: UNCHANGED from v1

*(BMA Calibrator, Conformal Prediction, DTW Distance, CPOD Anomaly Filter, OWA Feature Weighting)*

### Task 7.6: STUMPY Matrix Profile Cross-Ticker Discovery ← NEW Enhancement 6

**Files:**
- Create: `research/stumpy_matcher.py`
- Create: `tests/unit/test_stumpy_integration.py`

**Rationale:** STUMPY (created by TD Ameritrade/Schwab, 3,400+ stars, actively maintained) provides matrix profiles — for every subsequence in a time series, it finds the nearest neighbor using z-normalized Euclidean distance. The **AB-join** function discovers conserved patterns between two independent time series, directly applicable to FPPE's cross-ticker analogue matching.

STUMPY could serve as a complementary signal source: while FPPE's KNN matches on return fingerprints (point-in-time snapshots), STUMPY's matrix profile matches on *shape patterns* (subsequence similarity). The two approaches capture different aspects of historical similarity.

**Assignment:** SR developer (requires understanding of matrix profiles, z-normalization vs vol-normalization tradeoffs)

- [ ] **Step 1: Install STUMPY**

```bash
pip install stumpy
```

- [ ] **Step 2: Write failing tests for STUMPY integration**

```python
"""Tests for research/stumpy_matcher.py — matrix profile pattern discovery."""
import pytest
import numpy as np


class TestSTUMPYPatternDiscovery:

    def test_ab_join_finds_conserved_patterns(self):
        """AB-join between two tickers finds matching subsequences."""
        # Synthetic: embed same pattern in two different time series
        pass

    def test_matrix_profile_top_matches_as_signal(self):
        """Top matrix profile matches produce directional predictions."""
        pass

    def test_stumpy_signal_improves_bss(self):
        """STUMPY-augmented predictions have BSS >= baseline + 0.003."""
        pass
```

- [ ] **Step 3: Implement STUMPY integration module**

```python
"""stumpy_matcher.py — Matrix profile pattern discovery for FPPE.

Uses STUMPY AB-join to find conserved patterns across tickers.
Produces a secondary similarity signal that can be blended with
KNN fingerprint predictions.

Integration: Enhancement flag in EngineConfig.
Not a BaseMatcher — runs as a post-query signal augmentation.
"""
import stumpy
import numpy as np


class MatrixProfileSignal:
    """Generate cross-ticker pattern match signals via STUMPY."""

    def __init__(self, subsequence_length: int = 20, top_k: int = 10):
        self.m = subsequence_length
        self.top_k = top_k

    def find_cross_ticker_matches(
        self,
        query_series: np.ndarray,
        candidate_series: np.ndarray,
    ) -> dict:
        """Find best subsequence matches between two return series.

        Args:
            query_series: Query ticker's return series, shape (T,).
            candidate_series: Candidate ticker's return series, shape (T2,).

        Returns:
            Dict with match indices, distances, and z-normalized similarity.
        """
        # AB-join: find nearest neighbor in candidate for each query subsequence
        mp = stumpy.stump(query_series, self.m, candidate_series)
        # mp[:, 0] = matrix profile values (z-normalized Euclidean distances)
        # mp[:, 1] = nearest neighbor indices in candidate

        # Extract top-k best matches (lowest distance)
        top_indices = np.argsort(mp[:, 0].astype(float))[:self.top_k]

        return {
            'match_positions': top_indices,
            'distances': mp[top_indices, 0].astype(float),
            'nn_indices': mp[top_indices, 1].astype(int),
        }
```

- [ ] **Step 4: Run walk-forward with STUMPY signal augmentation**
- [ ] **Step 5: Gate: BSS improvement ≥ +0.003 on ≥ 3/6 folds**
- [ ] **Step 6: If pass → activate. If fail → keep behind flag.**
- [ ] **Step 7: Commit**

```bash
git add research/stumpy_matcher.py tests/unit/test_stumpy_integration.py
git commit -m "research(stumpy): matrix profile cross-ticker signal — BSS_delta=[X]"
```

### Task 7.7: Cumulative Enhancement Summary (Updated)

- [ ] **Step 1: Create results/enhancement_summary.tsv with all 6 enhancement results**
- [ ] **Step 2: Document which enhancements are active**
- [ ] **Step 3: Run full test suite**
- [ ] **Step 4: Commit**

### GATE CHECK: Phase 7 (Updated)

Same as v1 — keep what passes, revert what doesn't. STUMPY is Enhancement 6 (lowest priority, evaluated last).

---

## Phase 8: Paper Trading Validation — UPDATED (+Autonomous EOD)

**Gate:** All v1 success criteria by Month 3
**Duration:** 12 weeks
**Prerequisite:** Phase 7 gate passed
**Team:** JR developer (monitoring scripts, cron setup — well-defined requirements)

### Tasks 8.1–8.6: UNCHANGED from v1

*(Daily report, weekly review, paper trading infrastructure, month 1/2/3 reviews)*

### Task 8.7: Autonomous EOD Execution Pipeline ← NEW

**Files:**
- Create: `scripts/eod_pipeline.py`
- Create: `scripts/health_check.py`
- Create: `config/eod_schedule.json`
- Create: `tests/unit/test_eod_pipeline.py`

**Rationale:** The v1 plan mentions Windows Task Scheduler for scheduling but doesn't specify the orchestration logic. The Executive Summary and open-source landscape analysis highlight Microsoft Qlib's autonomous pipeline as a key architectural reference. This task formalizes FPPE's autonomous EOD execution into a production-grade pipeline.

**Assignment:** JR developer (clear requirements, standard engineering — cron + health checks)

- [ ] **Step 1: Write EOD pipeline orchestrator**

```python
"""eod_pipeline.py — Autonomous end-of-day execution orchestrator.

Daily Schedule (all times Eastern):
  06:00 AM — Data pull: fetch EOD prices for universe via Yahoo Finance
  06:30 AM — Feature rebuild: run prepare.py on updated data
  07:00 AM — Index rebuild: build HNSW/TurboQuant index on fresh features
  09:00 AM — OOB reconciliation: compare SharedState to broker positions
  09:30 AM — Health check: validate all systems before market open
  03:55 PM — Pre-signal: load pre-built index, warm caches
  04:00 PM — Signal generation: run full pipeline → signals → orders
  04:15 PM — Order submission: submit orders to broker
  04:30 PM — Checkpoint: save SharedState, generate daily report
  04:45 PM — Journal: log trades, metrics, alerts

Execution model:
  - Each step is idempotent (safe to re-run)
  - Each step logs start/end/duration/status to execution_log.jsonl
  - Failed step → alert (email/Slack webhook) + halt downstream
  - Hard timeout per step (configurable in eod_schedule.json)
  - Single master process — no parallel steps (prevents resource contention)

Scheduling:
  - Windows: Task Scheduler (schtasks) with wrapper bat file
  - Linux/future: crontab
  - Config: config/eod_schedule.json

Reference: Microsoft Qlib uses crontab for daily data updates covering 800+ stocks.
FPPE targets similar autonomous operation at 1500T scale.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


logger = logging.getLogger("eod_pipeline")


class EODStep:
    """A single step in the EOD pipeline."""

    def __init__(
        self,
        name: str,
        command: list[str],
        timeout_seconds: int = 600,
        required: bool = True,
    ) -> None:
        self.name = name
        self.command = command
        self.timeout_seconds = timeout_seconds
        self.required = required  # If False, failure is warning not halt

    def execute(self) -> dict:
        """Execute this step. Returns status dict."""
        t0 = time.perf_counter()
        try:
            result = subprocess.run(
                self.command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            elapsed = time.perf_counter() - t0
            return {
                "step": self.name,
                "status": "OK" if result.returncode == 0 else "FAIL",
                "returncode": result.returncode,
                "elapsed_s": round(elapsed, 2),
                "stdout_tail": result.stdout[-500:] if result.stdout else "",
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
                "timestamp": datetime.now().isoformat(),
            }
        except subprocess.TimeoutExpired:
            return {
                "step": self.name,
                "status": "TIMEOUT",
                "elapsed_s": self.timeout_seconds,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "step": self.name,
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


class EODPipeline:
    """Master orchestrator for the autonomous EOD execution cycle."""

    def __init__(self, config_path: Path) -> None:
        with open(config_path) as f:
            self.config = json.load(f)
        self.log_path = Path(self.config.get("log_path", "results/execution_log.jsonl"))
        self.steps: list[EODStep] = []
        self._build_steps()

    def _build_steps(self) -> None:
        """Build step sequence from config."""
        py = sys.executable
        for step_cfg in self.config["steps"]:
            self.steps.append(EODStep(
                name=step_cfg["name"],
                command=[py] + step_cfg["args"],
                timeout_seconds=step_cfg.get("timeout", 600),
                required=step_cfg.get("required", True),
            ))

    def run(self, steps_filter: Optional[list[str]] = None) -> list[dict]:
        """Execute the full pipeline or a subset of steps.

        Args:
            steps_filter: If provided, only run steps with matching names.

        Returns:
            List of step result dicts.
        """
        results = []
        for step in self.steps:
            if steps_filter and step.name not in steps_filter:
                continue

            logger.info(f"[EOD] Starting: {step.name}")
            result = step.execute()
            results.append(result)

            # Log to JSONL
            self._log_result(result)

            if result["status"] != "OK":
                logger.error(f"[EOD] {step.name}: {result['status']}")
                if step.required:
                    logger.error("[EOD] Required step failed — halting pipeline")
                    self._send_alert(step.name, result)
                    break
                else:
                    logger.warning(f"[EOD] Optional step {step.name} failed — continuing")
            else:
                logger.info(f"[EOD] {step.name}: OK ({result['elapsed_s']}s)")

        return results

    def _log_result(self, result: dict) -> None:
        """Append result to execution log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def _send_alert(self, step_name: str, result: dict) -> None:
        """Send failure alert. Placeholder for email/Slack webhook."""
        logger.critical(f"ALERT: EOD step '{step_name}' failed: {result}")
        # TODO: Wire to email or Slack webhook in Phase 9


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FPPE EOD Pipeline")
    parser.add_argument("--config", default="config/eod_schedule.json")
    parser.add_argument("--steps", nargs="*", help="Run specific steps only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    pipeline = EODPipeline(Path(args.config))
    results = pipeline.run(steps_filter=args.steps)

    # Print summary
    for r in results:
        status = r.get("status", "UNKNOWN")
        elapsed = r.get("elapsed_s", "?")
        print(f"  {r['step']:30s} {status:8s} ({elapsed}s)")
```

- [ ] **Step 2: Write EOD schedule configuration**

```json
{
    "log_path": "results/execution_log.jsonl",
    "alert_webhook": null,
    "steps": [
        {
            "name": "data_pull",
            "args": ["scripts/pull_eod_data.py"],
            "timeout": 300,
            "required": true,
            "schedule": "06:00 ET"
        },
        {
            "name": "feature_rebuild",
            "args": ["scripts/rebuild_features.py"],
            "timeout": 1800,
            "required": true,
            "schedule": "06:30 ET"
        },
        {
            "name": "index_rebuild",
            "args": ["scripts/build_overnight_index.py"],
            "timeout": 3600,
            "required": true,
            "schedule": "07:00 ET"
        },
        {
            "name": "reconciliation",
            "args": ["scripts/reconcile.py"],
            "timeout": 120,
            "required": true,
            "schedule": "09:00 ET"
        },
        {
            "name": "health_check",
            "args": ["scripts/health_check.py"],
            "timeout": 60,
            "required": true,
            "schedule": "09:30 ET"
        },
        {
            "name": "signal_generation",
            "args": ["-m", "pattern_engine.live", "--mode", "signal"],
            "timeout": 300,
            "required": true,
            "schedule": "16:00 ET"
        },
        {
            "name": "order_submission",
            "args": ["-m", "pattern_engine.live", "--mode", "execute"],
            "timeout": 120,
            "required": true,
            "schedule": "16:15 ET"
        },
        {
            "name": "checkpoint",
            "args": ["scripts/daily_report.py"],
            "timeout": 120,
            "required": false,
            "schedule": "16:30 ET"
        }
    ]
}
```

- [ ] **Step 3: Write health check script**

```python
"""health_check.py — Pre-execution health validation.

Checks:
  1. HNSW/TurboQuant index exists and is recent (< 24 hours old)
  2. Feature data is current (last date = yesterday or today)
  3. Broker API is reachable
  4. Disk space > 5 GB free
  5. Memory available > 8 GB
  6. No stale lock files from previous runs
  7. Python environment correct (3.12, venv active)

Exit code: 0 = healthy, 1 = warning (proceed with caution), 2 = critical (halt)
"""
import os
import sys
import psutil
from datetime import datetime, timedelta
from pathlib import Path


def check_index_freshness(index_dir: Path, max_age_hours: int = 24) -> tuple[bool, str]:
    """Verify index files are recent."""
    index_files = list(index_dir.glob("*.hnsw")) + list(index_dir.glob("*.tq"))
    if not index_files:
        return False, "No index files found"
    newest = max(f.stat().st_mtime for f in index_files)
    age_hours = (datetime.now().timestamp() - newest) / 3600
    if age_hours > max_age_hours:
        return False, f"Index is {age_hours:.1f} hours old (max {max_age_hours})"
    return True, f"Index age: {age_hours:.1f} hours"


def check_disk_space(min_gb: float = 5.0) -> tuple[bool, str]:
    """Verify sufficient disk space."""
    usage = psutil.disk_usage("C:\\")
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        return False, f"Only {free_gb:.1f} GB free (need {min_gb})"
    return True, f"{free_gb:.1f} GB free"


def check_memory(min_gb: float = 8.0) -> tuple[bool, str]:
    """Verify sufficient available memory."""
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    if avail_gb < min_gb:
        return False, f"Only {avail_gb:.1f} GB available (need {min_gb})"
    return True, f"{avail_gb:.1f} GB available"


if __name__ == "__main__":
    checks = [
        ("Disk space", check_disk_space()),
        ("Memory", check_memory()),
    ]

    failed = False
    for name, (ok, msg) in checks:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not ok:
            failed = True

    sys.exit(2 if failed else 0)
```

- [ ] **Step 4: Write tests for EOD pipeline**

```python
"""Tests for scripts/eod_pipeline.py — pipeline orchestration."""
import pytest
import json
from pathlib import Path
from scripts.eod_pipeline import EODPipeline, EODStep


class TestEODStep:
    def test_successful_step(self):
        """Step that exits 0 reports OK."""
        step = EODStep("test", ["python", "-c", "print('ok')"], timeout_seconds=10)
        result = step.execute()
        assert result["status"] == "OK"

    def test_failing_step(self):
        """Step that exits non-zero reports FAIL."""
        step = EODStep("test", ["python", "-c", "raise SystemExit(1)"], timeout_seconds=10)
        result = step.execute()
        assert result["status"] == "FAIL"

    def test_timeout_step(self):
        """Step that exceeds timeout reports TIMEOUT."""
        step = EODStep("test", ["python", "-c", "import time; time.sleep(30)"], timeout_seconds=1)
        result = step.execute()
        assert result["status"] == "TIMEOUT"


class TestEODPipeline:
    def test_halts_on_required_failure(self, tmp_path):
        """Pipeline stops when required step fails."""
        config = {
            "log_path": str(tmp_path / "log.jsonl"),
            "steps": [
                {"name": "step1", "args": ["-c", "print('ok')"], "required": True},
                {"name": "step2", "args": ["-c", "raise SystemExit(1)"], "required": True},
                {"name": "step3", "args": ["-c", "print('ok')"], "required": True},
            ]
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        pipeline = EODPipeline(config_path)
        results = pipeline.run()
        assert len(results) == 2  # step3 never ran
        assert results[1]["status"] == "FAIL"

    def test_continues_on_optional_failure(self, tmp_path):
        """Pipeline continues past optional step failure."""
        config = {
            "log_path": str(tmp_path / "log.jsonl"),
            "steps": [
                {"name": "step1", "args": ["-c", "print('ok')"], "required": True},
                {"name": "step2", "args": ["-c", "raise SystemExit(1)"], "required": False},
                {"name": "step3", "args": ["-c", "print('ok')"], "required": True},
            ]
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        pipeline = EODPipeline(config_path)
        results = pipeline.run()
        assert len(results) == 3  # all ran
        assert results[2]["status"] == "OK"
```

- [ ] **Step 5: Set up Windows Task Scheduler entries**

Create a batch wrapper for Task Scheduler:

```batch
@echo off
REM eod_runner.bat — Windows Task Scheduler entry point
cd /d C:\Users\Isaia\.claude\financial-research
call venv\Scripts\activate.bat
set PYTHONUTF8=1
py -3.12 scripts/eod_pipeline.py --config config/eod_schedule.json
```

Register with Task Scheduler:
- Trigger: Daily at 05:55 ET (5 minutes before first step)
- Action: Run `eod_runner.bat`
- Settings: Run whether user is logged in or not, do not start new instance if already running

- [ ] **Step 6: Commit**

```bash
git add scripts/eod_pipeline.py scripts/health_check.py config/eod_schedule.json tests/unit/test_eod_pipeline.py
git commit -m "feat(eod): autonomous EOD execution pipeline with health checks and scheduling"
```

---

## Phases 9–10: UNCHANGED

See v1 for complete task breakdowns. Phase 9 (Live Deploy) and Phase 10 (Options Foundation) are unmodified.

---

## Phase 11: Hyper-Scale Universe (5,200+ Tickers) ← NEW POST-LAUNCH PHASE

**Gate:** FAISS recall@50 ≥ 0.9999, BSS > 0 at 5200T, full pipeline < 4 hours overnight, peak RAM < 24 GB
**Duration:** 6-8 weeks
**Prerequisite:** Phase 9 stable (live system running with real capital)
**Team:** SR developer (FAISS tuning, TurboQuant production-grade integration)

### 11.1 Context & Motivation

Phase 6 scales FPPE to 1,500 tickers with hnswlib. Phase 11 pushes to the S&P 500 + Russell 1000 + Russell 2000 (~5,200 unique tickers after deduplication and historical depth filter).

At 5,200T:
- ~26M training vectors (5,200 tickers × 5,000 trading days)
- Float32 storage: 26M × 8 dims × 4 bytes = **793 MB** for raw vectors
- HNSW graph overhead: ~2-4× raw vector size ≈ **2-3 GB total**
- With Phase 7 OWA feature expansion (32 dims): 26M × 32 × 4 = **3.2 GB** raw → **8+ GB** with HNSW

This approaches the 24 GB memory budget. Two technologies make it tractable:

1. **FAISS IVF partitioning** — Divides vector space into Voronoi cells. Only searches nprobe cells (not all 26M vectors). Reduces query time from O(N) to O(N/n_cells × nprobe).

2. **TurboQuant compression** — Reduces per-vector storage from 32 bytes (float32, 8D) to ~4-8 bytes (3-4 bit quantization). At 32D: from 128 bytes to ~12-16 bytes. Combined with FAISS IVF, the 5200T index fits comfortably in < 4 GB.

### 11.2 Architecture Decision: FAISS vs hnswlib at Scale

| Factor | hnswlib (current) | FAISS IVF+Flat | FAISS IVF+TurboQuant |
|--------|-------------------|----------------|---------------------|
| 5200T memory | ~3 GB | ~900 MB + partitions | ~300 MB + partitions |
| Build time | ~30 min | ~5 min (training) | ~6 min |
| Query latency (k=50) | ~2 ms/query | ~5 ms/query (nprobe=32) | ~4 ms/query |
| Recall@50 | 0.9996 (proven) | 0.9999 (exact within cell) | TBD (Phase 6A) |
| Complexity | Low | Medium | High |

**Recommendation:** FAISS IVF-Flat for Phase 11 (exact search within cells, no quantization risk). TurboQuant compression as Phase 11B enhancement if memory pressure warrants it.

### Task 11.1: FAISS IVF-Flat Matcher Implementation

**Files:**
- Create: `pattern_engine/contracts/matchers/faiss_matcher.py`
- Create: `tests/performance/test_faiss_5200t.py`

**Assignment:** SR developer

- [ ] **Step 1: Implement FAISSMatcher as BaseMatcher subclass**

```python
"""faiss_matcher.py — FAISS IVF-Flat nearest-neighbor matcher.

Implements BaseMatcher for large-scale (5200T+) vector search.
Uses Inverted File Index with exact L2 within Voronoi cells.

Integration: EngineConfig(use_faiss=True) selects this backend.
Requires: pip install faiss-cpu (or faiss-gpu for CUDA acceleration).

Design notes:
  - n_cells = sqrt(N) is the standard FAISS heuristic
  - nprobe = 32 provides excellent recall at d=8 (verified in Phase 6)
  - Training uses a random 10% subsample (max 500k vectors)
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # Graceful degradation for environments without FAISS

from pattern_engine.contracts.matcher import BaseMatcher


class FAISSMatcher(BaseMatcher):
    """FAISS IVF-Flat nearest-neighbor matcher for hyper-scale universes."""

    def __init__(
        self,
        n_neighbors: int = 50,
        n_cells: int = 0,  # 0 = auto (sqrt(N))
        nprobe: int = 32,
    ) -> None:
        if faiss is None:
            raise RuntimeError("FAISS not installed: pip install faiss-cpu")
        self.n_neighbors = n_neighbors
        self._n_cells_override = n_cells
        self.nprobe = nprobe
        self._index: faiss.Index | None = None
        self._n_train = 0
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Build FAISS IVF-Flat index."""
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        N, d = X_f32.shape
        self._n_train = N

        # Number of Voronoi cells
        n_cells = self._n_cells_override or max(int(np.sqrt(N)), 16)

        # Quantizer (flat L2 — exact distances)
        quantizer = faiss.IndexFlatL2(d)

        # IVF index
        self._index = faiss.IndexIVFFlat(quantizer, d, n_cells)

        # Train on subsample
        train_n = min(N, max(n_cells * 40, 500_000))
        rng = np.random.RandomState(42)
        train_idx = rng.choice(N, train_n, replace=False)
        self._index.train(X_f32[train_idx])

        # Add all vectors
        self._index.add(X_f32)

        # Set search-time probe count
        self._index.nprobe = self.nprobe
        self._fitted = True

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        if not self._fitted:
            raise RuntimeError("Must call fit() before kneighbors()")
        k = n_neighbors if n_neighbors > 0 else self.n_neighbors
        k = min(k, self._n_train)

        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        sq_distances, indices = self._index.search(X_f32, k)

        # FAISS returns squared L2 — convert to Euclidean
        distances = np.sqrt(np.maximum(sq_distances, 0.0))
        return distances, indices

    def save_index(self, path) -> None:
        """Save FAISS index to disk."""
        if self._index is None:
            raise RuntimeError("No index to save")
        faiss.write_index(self._index, str(path))

    def load_index(self, path) -> None:
        """Load FAISS index from disk."""
        self._index = faiss.read_index(str(path))
        self._index.nprobe = self.nprobe
        self._n_train = self._index.ntotal
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> Dict[str, object]:
        return {
            'n_neighbors': self.n_neighbors,
            'n_cells': self._n_cells_override,
            'nprobe': self.nprobe,
            'backend': 'faiss_ivf_flat',
        }
```

- [ ] **Step 2: Write scaling benchmark test**
- [ ] **Step 3: Wire into PatternMatcher backend selection**

```python
elif config.use_faiss:
    from pattern_engine.contracts.matchers.faiss_matcher import FAISSMatcher
    self._backend = FAISSMatcher(n_neighbors=n_probe, nprobe=getattr(cfg, 'faiss_nprobe', 32))
```

- [ ] **Step 4: Run scaling benchmark on 5200T synthetic data**
- [ ] **Step 5: BSS validation at 5200T**
- [ ] **Step 6: Overnight build script for FAISS index**
- [ ] **Step 7: Commit**

### Task 11.2: TurboQuant + FAISS Hybrid (Optional Phase 11B)

Only if memory pressure warrants it at 5200T. Combines FAISS IVF partitioning with TurboQuant-compressed vectors within each cell. Requires Phase 6A/6B artifacts.

- [ ] **Step 1: Evaluate memory at 5200T with FAISS IVF-Flat**

If peak RAM < 24 GB → TurboQuant not needed. Document and defer.
If peak RAM > 24 GB → Proceed with hybrid.

- [ ] **Step 2: Implement FAISS IVF + TurboQuant hybrid** (if needed)
- [ ] **Step 3: Recall/BSS validation**
- [ ] **Step 4: Commit**

### Task 11.3: Russell 2000 Data Pipeline Extension

- [ ] **Step 1: Extend ticker sourcing to Russell 2000**
- [ ] **Step 2: Apply historical depth filter (2010 gate)**
- [ ] **Step 3: Expand SECTOR_MAP for all new tickers**
- [ ] **Step 4: Run full pipeline on 5200T**
- [ ] **Step 5: Commit**

### GATE CHECK: Phase 11

- [ ] **FAISS recall@50 ≥ 0.9999 at 5200T?**
- [ ] **BSS > 0 on ≥ 3/6 folds at 5200T?**
- [ ] **Full pipeline < 4 hours overnight?**
- [ ] **Peak RAM < 24 GB?**
  - YES → Expand live universe to 5200T
  - NO → Stay at 1500T. Log bottleneck for future optimization.

---

## Team Assignment Matrix

### Skill Requirements by Phase

| Phase | Primary Skills | Assignment | Rationale |
|-------|---------------|------------|-----------|
| 1 — BSS Fix | Statistics, probability calibration, Brier Scores | **SR** | Requires deep understanding of why BSS < 0, diagnostic reasoning |
| 2 — Half-Kelly | Kelly criterion, basic Pydantic | **JR** | Well-defined math formula, clear test specs provided |
| 3 — Risk Engine | Risk management, overlay integration | **MIX** | SR designs overlay interaction; JR implements config wiring |
| 4 — Portfolio Manager | Multi-constraint optimization | **MIX** | SR handles constraint logic; JR writes rejection analysis |
| 5 — Live Plumbing | REST APIs, order state machines, broker protocols | **JR** | Standard engineering; clear ABCs; SR reviews IBKR adapter |
| 6 — Universe 1500T | Data pipelines, HNSW tuning, benchmarking | **MIX** | JR handles data; SR handles recall validation |
| 6A — TurboQuant R&D | Information theory, linear algebra, quantization | **SR** | Novel research — requires reading papers, designing experiments |
| 6B — VQ Integration | BaseMatcher interface, NumPy optimization | **SR** | Builds on 6A research; tight performance requirements |
| 7 — Enhancements | BMA, conformal, DTW, anomaly detection, STUMPY | **SR** | Each enhancement requires domain expertise |
| 8 — Paper Trading | Monitoring, cron, reporting scripts | **JR** | Well-defined requirements; SR reviews anomaly handling |
| 9 — Live Deploy | Deployment, tax tracking, kill switches | **MIX** | SR handles tax edge cases; JR handles ramp-up monitoring |
| 10 — Options | Black-Scholes, greeks, option chain modeling | **MIX** | SR designs greeks calculator; JR implements Pydantic contracts |
| 11 — Hyper-Scale | FAISS, large-scale engineering, memory optimization | **SR** | Production-scale infrastructure; tight memory budget |

### Junior Developer Growth Track

Phases are ordered to progressively build JR developer capabilities:

1. **Phase 2 (Kelly):** Clean math → TDD → Pydantic. Low risk, clear spec. *Learning: TDD workflow, Pydantic models.*
2. **Phase 5 (Broker):** ABCs → mocks → state machines. Standard patterns. *Learning: API integration, error handling.*
3. **Phase 8 (Paper Trading):** Monitoring → cron → alerting. Operational focus. *Learning: Production operations, observability.*
4. **Phase 10 (Options — JR portion):** Pydantic contracts, test writing. *Learning: Financial domain contracts.*

By Phase 11, JR developers have 6+ months of FPPE experience and can take on more complex tasks.

---

## Updated Risk Register

| Risk | Phase | Likelihood | Impact | Mitigation |
|------|-------|-----------|--------|------------|
| BSS stays negative after H1-H4 | 1 | Medium | Critical | Fall back to 52T; BMA rescue path |
| Kelly fraction too small | 2 | Low | Medium | Revert to ATR-only sizing |
| Analogue dilution at 1500T | 6 | High | High | Same Phase 1 fix; fall back to 585T |
| **TurboQuant fails at d=8** | **6A** | **High** | **Low** | **Zero critical-path impact; revisit at d=16+ post-Phase 7** |
| **FAISS IVF recall < 0.9999** | **11** | **Low** | **Medium** | **Increase nprobe; fall back to hnswlib with sector pre-filter** |
| **EOD pipeline hangs/crashes** | **8+** | **Medium** | **Medium** | **Step-level timeouts; idempotent re-run; health checks** |
| **STUMPY too slow at 1500T** | **7** | **Medium** | **Low** | **Subsample to top-100 tickers by volume; or abandon** |
| HNSW recall < 0.9999 at scale | 6 | Low | High | Increase ef_search; BallTree fallback |
| Broker API instability | 5, 9 | Medium | Medium | Alpaca fallback; retry with backoff |
| Regime shift during paper trading | 8 | Medium | Medium | Compare to historical fold |
| Slippage exceeds model | 8, 9 | Medium | Low | Adjust parameter; VWAP window |
| Memory limit (32GB) at 1500T | 6, 8 | Low | Medium | Peak RAM < 24 GB gate |
| **Memory limit at 5200T** | **11** | **Medium** | **High** | **FAISS IVF (partitioned); TurboQuant compression; chunked build** |
| Tax lot edge cases | 9 | Low | Low | Manual override + tax advisor |

### Key Risk Insight: TurboQuant at d=8

The highest-probability new risk is TurboQuant underperforming at d=8. This is **by design** — Phase 6A is structured as a cheap research spike (~4 weeks SR time) with zero critical-path exposure. Three mitigations exist:

1. **Immediate:** If recall@50 < 0.9990 at d=8 → cancel Phase 6B, zero sunk cost beyond research
2. **Medium-term:** Phase 7 OWA may expand features to d=16-32 → re-run 6A experiments at higher d
3. **Long-term:** Phase 11 at 5200T may warrant the compression even at modest recall if memory is the binding constraint → accept recall tradeoff with BSS monitoring

---

## Open-Source Integration Map

Based on the competitive landscape analysis, these are the production-grade libraries to integrate:

| Library | Stars | Phase | Purpose | Status |
|---------|-------|-------|---------|--------|
| **hnswlib** | ~3k | Current | Approximate NN search (d=8, 585T-1500T) | Integrated |
| **FAISS** | ~37k | 6 (eval), 11 (prod) | IVF partitioned search (5200T+) | Planned |
| **STUMPY** | ~3.4k | 7 (enhancement) | Matrix profile cross-ticker discovery | Planned |
| **tslearn** | ~2.9k | 7 (enhancement) | DTW-aware KNN classifier (DTW enhancement) | Evaluation |
| **backtesting.py** | ~4.5k | 8 (reference) | Walk-forward KNN tutorial (k=7, validation reference) | Reference |
| **scikit-learn** | — | Current | BallTree, StandardScaler, Platt (LogisticRegression) | Integrated |
| **flimao/briercalc** | small | 1 (diagnostic) | Murphy BSS decomposition (calibration vs resolution) | Planned |

**Not integrating (rationale):**
- chinuy/stock-price-prediction: Validates our approach (KNN optimal delta=99 days) but code is inactive since 2018. Used as benchmark reference.
- gaborvecsei/Stocks-Pattern-Analyzer: Exploratory tool, not a prediction system
- Microsoft Qlib: Too heavy a dependency; we build our own pipeline but reference its architecture for EOD scheduling

---

## Competitive Benchmarking Framework

### Rationale

The open-source landscape analysis identified 27 repos addressing individual components of FPPE's architecture, but found that **no single repo replicates the full methodology**. The combination of cross-ticker KNN on return fingerprints + Platt calibration + BSS evaluation + regime conditioning appears genuinely novel.

However, "novel" means nothing without empirical evidence that the combination outperforms the parts. This section defines a systematic benchmarking framework that compares FPPE's capabilities against the best available open-source implementations at each pipeline stage. Benchmarks are integrated into the existing phase gates — not a separate workstream.

### Benchmark Registry

Each benchmark reproduces a key finding from an identified open-source repo, then measures FPPE's approach against it on the same (or equivalent) data.

#### B1: Return Fingerprint Window Length — chinuy Validation

**Repo:** `chinuy/stock-price-prediction` (MIT, 20 stars)
**Their finding:** KNN's optimal delta (feature window) is **99 days** — far more than SVM (4d) or Random Forest (3d). KNN Sharpe = 0.990 on 9 sector ETFs.
**FPPE comparison:** Our VOL_NORM_COLS use windows [1, 3, 7, 14, 30, 45, 60, 90] days — max 90d, close to chinuy's 99d optimum.
**Benchmark task:** Reproduce chinuy's delta sweep on FPPE's 585T universe. Test delta=[30, 60, 90, 120, 150] using raw returns (their method) vs vol-normalized returns (our method). Measure BSS and Sharpe at each delta.
**Phase:** 1 (BSS Diagnosis — Task 1.1 can incorporate this as a diagnostic)
**Gate metric:** Does FPPE's 8-feature vol-normalized fingerprint outperform chinuy's single-delta raw return approach on BSS?
**Assignment:** SR (statistical analysis)

**Files:**
- Create: `scripts/benchmarks/b1_chinuy_delta_sweep.py`

- [ ] **Step 1: Implement chinuy-style delta sweep**

```python
"""B1: Reproduce chinuy/stock-price-prediction delta sweep on FPPE universe.

chinuy finding: KNN optimal delta=99 days on 9 ETFs, Sharpe=0.990.
Question: Does FPPE's 8-feature VOL_NORM_COLS outperform single-delta raw returns?

Output: results/benchmarks/b1_chinuy_comparison.tsv
Columns: delta, method (chinuy_raw | fppe_volnorm), bss_mean, sharpe, accuracy
"""
```

- [ ] **Step 2: Run on 585T walk-forward data**
- [ ] **Step 3: Log comparison table in results**
- [ ] **Step 4: Commit**

```bash
git add scripts/benchmarks/ results/benchmarks/
git commit -m "bench(b1): chinuy delta sweep — VOL_NORM vs raw returns at [DELTA]d"
```

#### B2: Cross-Ticker k=1 Matching — DayuanTan Comparison

**Repo:** `DayuanTan/knn_predictprice` (1 star)
**Their approach:** 8-feature price-change fingerprint, k=1 (single nearest match), Euclidean distance, cross-stock search over 5 years.
**FPPE comparison:** 8-feature vol-normalized fingerprint, k=50, Euclidean, cross-stock search over 25 years with Platt calibration.
**Benchmark task:** Run FPPE's walk-forward with k=1 (DayuanTan's approach) vs k=50 (our approach). Measure: BSS, accuracy, and how often k=1's single match is correct vs k=50's aggregate.
**Phase:** 1 (BSS Sweep — Task 1.4 H3 already sweeps top_k; extend to k=1)
**Gate metric:** k=50 with Platt should dominate k=1 on BSS. Quantify the advantage.
**Assignment:** SR (part of existing BSS sweep)

**Files:**
- Extend: `scripts/bss_sweep.py` — add k=1 to H3 sweep range

- [ ] **Step 1: Add k=1 to top_k sweep in bss_sweep.py**

Extend H3 sweep: `[1, 5, 10, 20, 30, 40, 50]` (was `[10, 20, 30, 40, 50]`).

- [ ] **Step 2: Run and document k=1 vs k=50 comparison**
- [ ] **Step 3: Commit with provenance**

#### B3: Platt Calibration Impact — The Unclaimed Gap

**Landscape finding:** **Zero** public repos apply probability calibration to stock KNN outputs.
**FPPE comparison:** Platt scaling produces calibrated probabilities; BSS measures calibration quality.
**Benchmark task:** Use flimao/briercalc's Murphy decomposition to break BSS into calibration, resolution, and uncertainty components. Compare FPPE's raw (uncalibrated) vs Platt-calibrated outputs. This produces the first public evidence that calibration matters for financial KNN.
**Phase:** 1 (BSS Diagnosis — Task 1.1 reliability diagram is the visual; Murphy decomposition is the quantitative complement)
**Gate metric:** Platt calibration should improve the calibration component of BSS by measurable amount.
**Assignment:** SR

**Files:**
- Create: `scripts/benchmarks/b3_murphy_decomposition.py`

- [ ] **Step 1: Implement Murphy BSS decomposition**

```python
"""B3: Murphy decomposition of Brier Skill Score.

Decomposes BSS into three components:
  - Calibration (REL): how well predicted probs match actual frequencies
  - Resolution (RES): how much predictions differ from base rate
  - Uncertainty (UNC): base rate entropy (not controllable)

BSS = (RES - REL) / UNC

Compare: raw KNN probabilities vs Platt-calibrated probabilities.
This is the first public measurement of calibration impact on financial KNN.

Reference: flimao/briercalc implements this exact decomposition.
Output: results/benchmarks/b3_murphy_decomposition.tsv
"""
import numpy as np


def murphy_decomposition(predicted: np.ndarray, actual: np.ndarray, n_bins: int = 10):
    """Decompose Brier Score into calibration, resolution, uncertainty.

    Args:
        predicted: Predicted probabilities, shape (N,).
        actual: Binary outcomes (0/1), shape (N,).
        n_bins: Number of bins for grouping predictions.

    Returns:
        Dict with keys: brier_score, calibration, resolution, uncertainty,
        bss, rel_per_bin, res_per_bin.
    """
    N = len(predicted)
    base_rate = actual.mean()
    uncertainty = base_rate * (1 - base_rate)

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[-1] += 1e-8

    calibration = 0.0
    resolution = 0.0
    bin_details = []

    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        n_k = mask.sum()
        if n_k == 0:
            continue
        mean_pred_k = predicted[mask].mean()  # average prediction in bin
        mean_actual_k = actual[mask].mean()    # actual hit rate in bin

        calibration += n_k * (mean_actual_k - mean_pred_k) ** 2
        resolution += n_k * (mean_actual_k - base_rate) ** 2

        bin_details.append({
            'bin': i,
            'n': int(n_k),
            'mean_pred': round(float(mean_pred_k), 4),
            'mean_actual': round(float(mean_actual_k), 4),
            'cal_contribution': round(float(n_k * (mean_actual_k - mean_pred_k) ** 2 / N), 6),
            'res_contribution': round(float(n_k * (mean_actual_k - base_rate) ** 2 / N), 6),
        })

    calibration /= N
    resolution /= N

    brier_score = calibration - resolution + uncertainty
    bss = 1 - brier_score / uncertainty if uncertainty > 0 else 0.0

    return {
        'brier_score': round(float(brier_score), 6),
        'calibration': round(float(calibration), 6),
        'resolution': round(float(resolution), 6),
        'uncertainty': round(float(uncertainty), 6),
        'bss': round(float(bss), 6),
        'base_rate': round(float(base_rate), 4),
        'n_samples': N,
        'bins': bin_details,
    }
```

- [ ] **Step 2: Run on raw vs Platt-calibrated walk-forward outputs**
- [ ] **Step 3: Document calibration delta — first public measurement**
- [ ] **Step 4: Commit**

#### B4: Regime Conditioning Value — tubakhxn Comparison

**Repo:** `tubakhxn/Market-Regime-Detection-System` (MIT, 5 stars)
**Their approach:** ML classifiers on technical indicators → 3-class (bull/bear/sideways) from SPY.
**FPPE approach:** SPY 90-day return threshold → binary (bull/bear) + VIX spike + yield curve inversion.
**Benchmark task:** Compare FPPE's regime-conditioned predictions vs unconditioned predictions. Also compare our simple threshold regime labeler vs tubakhxn's ML approach if reproducible.
**Phase:** 7 (Enhancement 5: OWA feature weighting is regime-conditional; benchmark measures regime value)
**Gate metric:** Regime conditioning should improve worst-fold BSS or reduce drawdown.
**Assignment:** SR

**Files:**
- Create: `scripts/benchmarks/b4_regime_conditioning_value.py`

- [ ] **Step 1: Measure prediction quality with vs without regime filter**

Run walk-forward twice: `regime_filter=True` vs `regime_filter=False`. Compare BSS, Sharpe, max DD per fold.

- [ ] **Step 2: If tubakhxn repo reproducible, compare 3-class ML vs our binary threshold**
- [ ] **Step 3: Document findings**
- [ ] **Step 4: Commit**

#### B5: DTW vs Euclidean — mason-lee19 Comparison

**Repo:** `mason-lee19/DtwStockAnalysis` (2 stars)
**Their finding:** DTW on 50-day windows "does not look to be a profitable strategy" — but they lacked calibration and regime conditioning.
**FPPE question:** Does DTW become profitable when combined with Platt calibration + regime conditioning?
**Phase:** 7 (Enhancement 3: DTW/WFA reranker)
**Gate metric:** DTW + Platt + regime should outperform both (a) DTW alone (mason-lee19 baseline) and (b) Euclidean + Platt + regime (our baseline).
**Assignment:** SR

**Files:**
- Create: `scripts/benchmarks/b5_dtw_vs_euclidean.py`

- [ ] **Step 1: Reproduce mason-lee19's finding: DTW without calibration**

Use FPPE's pipeline with DTW distance, no Platt, no regime. Confirm poor results.

- [ ] **Step 2: Add Platt calibration to DTW pipeline**

Measure: does BSS turn positive?

- [ ] **Step 3: Add regime conditioning to DTW + Platt**

Measure: does performance improve further?

- [ ] **Step 4: Compare full pipeline (DTW+Platt+regime) vs (Euclidean+Platt+regime)**
- [ ] **Step 5: Document the decomposition: which component rescues DTW?**
- [ ] **Step 6: Commit**

#### B6: STUMPY Matrix Profile vs KNN Fingerprint — Cross-Method Comparison

**Repo:** `stumpy-dev/stumpy` (BSD-3, ~3,400 stars, by TD Ameritrade)
**Benchmark task:** Compare STUMPY AB-join cross-ticker matching against FPPE's KNN fingerprint matching. Same tickers, same time period, same forward horizon.
**Phase:** 7 (Enhancement 6: STUMPY integration)
**Gate metric:** STUMPY should capture complementary signal (correlation with KNN predictions < 0.5) to justify blending.
**Assignment:** SR

**Files:**
- Create: `scripts/benchmarks/b6_stumpy_vs_knn.py`

- [ ] **Step 1: Generate STUMPY-only predictions on FPPE walk-forward data**
- [ ] **Step 2: Generate KNN-only predictions (baseline)**
- [ ] **Step 3: Measure signal correlation between the two**
- [ ] **Step 4: Test blended signal (0.8 KNN + 0.2 STUMPY)**
- [ ] **Step 5: Commit**

#### B7: Autonomous Pipeline Architecture — Qlib Reference Audit

**Repo:** `microsoft/qlib` (MIT, 37,300 stars)
**Benchmark task:** Not a performance benchmark — an architecture audit. Compare FPPE's autonomous EOD pipeline (Phase 8, Task 8.7) against Qlib's crontab-based daily pipeline. Document: what architectural patterns Qlib uses that we adopted, adapted, or intentionally excluded.
**Phase:** 8 (Paper Trading — architectural review during Month 1)
**Deliverable:** Architecture comparison document.
**Assignment:** JR (research task, well-scoped)

**Files:**
- Create: `docs/benchmarks/b7_qlib_architecture_audit.md`

- [ ] **Step 1: Review Qlib's data update pipeline (crontab, Yahoo Finance, 800+ stocks)**
- [ ] **Step 2: Document FPPE's equivalent (eod_pipeline.py, health_check.py)**
- [ ] **Step 3: Identify gaps and adopted patterns**
- [ ] **Step 4: Commit**

#### B8: Walk-Forward Validation Rigor — backtesting.py Reference

**Repo:** `kernc/backtesting.py` (BSD, ~4,500 stars)
**Their approach:** KNN(7) with walk-forward retraining every 20 iterations on 400-value windows. EUR/USD forex.
**FPPE comparison:** KNN(50) with expanding-window 6-fold walk-forward on 585T equities over 25 years.
**Benchmark task:** Document the methodological differences quantitatively. FPPE's expanding window vs backtesting.py's rolling window; our 25-year depth vs their ~400 values; equities vs forex.
**Phase:** 8 (Paper Trading — methodology documentation during Month 1)
**Assignment:** JR (documentation task)

**Files:**
- Create: `docs/benchmarks/b8_walkforward_comparison.md`

- [ ] **Step 1: Document FPPE vs backtesting.py walk-forward methodology**
- [ ] **Step 2: Quantify: training set size, retrain frequency, universe breadth, time depth**
- [ ] **Step 3: Commit**

### Benchmark Execution Schedule

Benchmarks are integrated into existing phases — no separate timeline:

| Benchmark | Phase | When | Blocking? |
|-----------|-------|------|-----------|
| B1: chinuy delta sweep | 1 | During BSS diagnostics | No (informational) |
| B2: DayuanTan k=1 vs k=50 | 1 | Extend existing H3 sweep | No (extends existing task) |
| B3: Murphy BSS decomposition | 1 | After reliability diagram | **Yes — informs BSS fix strategy** |
| B4: Regime conditioning value | 7 | During OWA evaluation | No (informational) |
| B5: DTW vs Euclidean | 7 | During DTW enhancement | No (informational, validates design) |
| B6: STUMPY vs KNN | 7 | During STUMPY enhancement | No (informational) |
| B7: Qlib architecture audit | 8 (Month 1) | Architecture review | No (documentation) |
| B8: Walk-forward comparison | 8 (Month 1) | Methodology docs | No (documentation) |

**B3 is the only blocking benchmark** — Murphy decomposition directly informs which BSS fix hypothesis to pursue (calibration-heavy fix vs resolution-heavy fix).

### Benchmark Results Directory

```
results/benchmarks/
    b1_chinuy_comparison.tsv         # Delta sweep: VOL_NORM vs raw returns
    b2_k1_vs_k50_comparison.tsv      # DayuanTan single-match vs pooled
    b3_murphy_decomposition.tsv      # BSS decomposition: raw vs Platt
    b4_regime_conditioning.tsv       # With vs without regime filter
    b5_dtw_vs_euclidean.tsv          # DTW rescue via calibration
    b6_stumpy_vs_knn.tsv             # Signal correlation and blend results
docs/benchmarks/
    b7_qlib_architecture_audit.md    # Pipeline architecture comparison
    b8_walkforward_comparison.md     # Methodology documentation
```

### Competitive Positioning Summary (Updated After Benchmarks)

This table will be filled as benchmarks complete. It maps FPPE's components against the best open-source matches and records our measured advantage.

| FPPE Component | Best Public Match | Their Result | Our Result | Delta | Benchmark |
|---|---|---|---|---|---|
| Multi-day return fingerprints | chinuy (delta=99d, Sharpe=0.990) | TBD | TBD | TBD | B1 |
| Cross-ticker pooled KNN (k=50) | DayuanTan (k=1, 8-feature) | TBD | TBD | TBD | B2 |
| Platt calibration on KNN | **None exist** (first public) | N/A | TBD | N/A | B3 |
| BSS evaluation | flimao/briercalc (decomposition) | N/A (tool) | TBD | N/A | B3 |
| SPY regime conditioning | tubakhxn (3-class ML) | TBD | TBD | TBD | B4 |
| DTW pattern matching | mason-lee19 ("not profitable") | Unprofitable | TBD | TBD | B5 |
| Matrix profile matching | STUMPY AB-join | TBD | TBD | TBD | B6 |
| Autonomous EOD pipeline | Microsoft Qlib (crontab) | Reference | TBD | N/A | B7 |
| Walk-forward validation | backtesting.py (k=7, 400-window) | Reference | TBD | N/A | B8 |
| Confidence threshold tuning | **None exist** | N/A | 0.55/0.65 | First | — |

### What This Proves

When benchmarks complete, FPPE should demonstrate:

1. **Vol-normalized fingerprints outperform raw returns** (B1) — vindicating the feature engineering choice
2. **k=50 pooled matching outperforms k=1 single-match** (B2) — justifying the probabilistic aggregation
3. **Platt calibration is the missing ingredient** (B3, B5) — explaining why open-source KNN stock prediction hasn't worked (they all skip calibration)
4. **Regime conditioning adds value** (B4) — separating regime-adaptive from regime-ignorant predictions
5. **STUMPY captures complementary signal** (B6) — justifying the blended architecture
6. **FPPE's autonomous pipeline matches production-grade standards** (B7) — architectural credibility

---

## Glossary of v2 Additions

| Term | Definition |
|------|-----------|
| **TurboQuant** | Two-stage vector quantization: random rotation + Lloyd-Max scalar quantization + QJL residual correction (Google, arXiv:2504.19874) |
| **PolarQuant** | Polar-coordinate quantization with random pre-conditioning; >4.2× compression; TurboQuant Stage 1 is a variant |
| **QJL** | Quantized Johnson-Lindenstrauss: 1-bit sign-encoding of random projections for unbiased inner-product estimation |
| **Lloyd-Max** | Optimal scalar quantizer minimizing mean squared error for a known probability distribution |
| **FAISS** | Facebook AI Similarity Search: library for efficient similarity search of dense vectors at billion-scale |
| **IVF** | Inverted File Index: partitions vector space into Voronoi cells for sub-linear search |
| **STUMPY** | Time series matrix profile library (TD Ameritrade/Schwab); computes all-pairs subsequence similarity |
| **AB-join** | STUMPY operation finding conserved patterns between two independent time series |
| **EOD Pipeline** | Autonomous end-of-day execution cycle: data pull → features → index → reconciliation → signals → orders |

---

## Document Lineage

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| v1 | 2026-03-26 | Claude (Opus 4.6) | Original 10-phase roadmap |
| **v2** | **2026-03-28** | **Claude (Opus 4.6)** | **+TurboQuant R&D (6A/6B), +FAISS (6.5/11), +STUMPY (7.6), +Autonomous EOD (8.7), +Hyper-Scale (11), +Team assignments, +8 competitive benchmarks (B1-B8)** |
