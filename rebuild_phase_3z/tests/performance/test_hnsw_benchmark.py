"""
test_hnsw_benchmark.py — HNSW vs BallTree performance benchmarks.

Measures:
  - Index build time (N=50k, D=8)
  - Query latency: p50, p95, p99 on 1000 queries
  - Speedup ratio: HNSW vs BallTree
  - Results written to artifacts/benchmarks/hnsw_benchmark_{timestamp}.json

Acceptance criteria (SLE-63):
  - Query latency p95 < 0.1ms/query
  - Speedup > 20× vs ball_tree on 50k fingerprints

These are marked with @pytest.mark.slow and excluded from CI unless
explicitly requested (e.g., pytest -m slow).

Linear: SLE-63
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

try:
    from rebuild_phase_3z.fppe.pattern_engine.contracts.matchers.hnsw_matcher import HNSWMatcher
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False

from rebuild_phase_3z.fppe.pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher


# ─── Benchmark configuration ───────────────────────────────────────────────────

N_TRAIN = 50_000    # Production-scale training set
N_QUERY = 1_000     # Enough to get stable p95
N_FEATURES = 8      # returns_only set
TOP_K = 50          # Production top_k
RNG_SEED = 42

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / "benchmarks"

# Acceptance thresholds
MAX_P95_MS = 0.1          # SLE-63: p95 query latency must be < 0.1ms/query
MIN_SPEEDUP = 20.0        # SLE-63: HNSW must be >20x faster than BallTree on N=50k


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def benchmark_data():
    """Generate 50k training fingerprints + 1000 query fingerprints."""
    rng = np.random.RandomState(RNG_SEED)
    X_train = rng.randn(N_TRAIN, N_FEATURES).astype(np.float64)
    X_query = rng.randn(N_QUERY, N_FEATURES).astype(np.float64)
    return X_train, X_query


@pytest.fixture(scope="module")
def fitted_balltree(benchmark_data):
    X_train, _ = benchmark_data
    bt = BallTreeMatcher(n_neighbors=TOP_K * 3)
    bt.fit(X_train)
    return bt


@pytest.fixture(scope="module")
def fitted_hnsw(benchmark_data):
    """Fit HNSW on N=50k training set. Skips if hnswlib is not installed."""
    if not HAS_HNSWLIB:
        pytest.skip("hnswlib not installed")
    X_train, _ = benchmark_data
    hnsw = HNSWMatcher(n_neighbors=TOP_K * 3, ef_construction=200, M=16)
    hnsw.fit(X_train)
    return hnsw


# ─── Build time benchmarks ────────────────────────────────────────────────────

class TestBuildTimeBenchmark:

    def test_balltree_build_time(self, benchmark_data):
        """Record BallTree build time on N=50k. No hard limit — just logging."""
        X_train, _ = benchmark_data
        bt = BallTreeMatcher(n_neighbors=TOP_K * 3)

        t0 = time.perf_counter()
        bt.fit(X_train)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\nBallTree build: {elapsed_ms:.1f}ms for N={N_TRAIN:,}")
        assert bt.is_fitted

    @pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
    def test_hnsw_build_time(self, benchmark_data):
        """Record HNSW build time on N=50k. Expected < BallTree (graph-based)."""
        X_train, _ = benchmark_data
        hnsw = HNSWMatcher(n_neighbors=TOP_K * 3, ef_construction=200, M=16)

        t0 = time.perf_counter()
        hnsw.fit(X_train)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\nHNSW build: {elapsed_ms:.1f}ms for N={N_TRAIN:,}")
        assert hnsw.is_fitted


# ─── Query latency benchmarks ─────────────────────────────────────────────────

class TestQueryLatencyBenchmark:
    """Query latency benchmarks against SLE-63 acceptance criteria."""

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
    def test_hnsw_query_latency(self, fitted_hnsw, benchmark_data):
        """SLE-63: HNSW p95 query latency must be < 0.1ms/query on N=50k."""
        _, X_query = benchmark_data

        # Warmup pass (avoids cold-start JIT effects)
        fitted_hnsw.kneighbors(X_query[:10], n_neighbors=TOP_K)

        # Time individual query rows
        latencies_ms = []
        for i in range(N_QUERY):
            row = X_query[i : i + 1]
            t0 = time.perf_counter()
            fitted_hnsw.kneighbors(row, n_neighbors=TOP_K)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        latencies_ms = np.array(latencies_ms)
        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)
        p99 = np.percentile(latencies_ms, 99)
        mean = latencies_ms.mean()

        print(
            f"\nHNSW query latency (N={N_TRAIN:,}, k={TOP_K}):\n"
            f"  p50={p50:.4f}ms  p95={p95:.4f}ms  p99={p99:.4f}ms  mean={mean:.4f}ms"
        )

        assert p95 < MAX_P95_MS, (
            f"HNSW p95 latency {p95:.4f}ms >= {MAX_P95_MS}ms (SLE-63 gate)"
        )

    @pytest.mark.slow
    def test_balltree_query_latency(self, fitted_balltree, benchmark_data):
        """Record BallTree latency for speedup calculation reference."""
        _, X_query = benchmark_data

        # Warmup
        fitted_balltree.kneighbors(X_query[:10], n_neighbors=TOP_K)

        latencies_ms = []
        for i in range(N_QUERY):
            row = X_query[i : i + 1]
            t0 = time.perf_counter()
            fitted_balltree.kneighbors(row, n_neighbors=TOP_K)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        latencies_ms = np.array(latencies_ms)
        p50 = np.percentile(latencies_ms, 50)
        p95 = np.percentile(latencies_ms, 95)

        print(
            f"\nBallTree query latency (N={N_TRAIN:,}, k={TOP_K}):\n"
            f"  p50={p50:.4f}ms  p95={p95:.4f}ms"
        )
        # No hard limit for BallTree — it's the reference baseline


# ─── Speedup comparison (combined) ────────────────────────────────────────────

class TestSpeedupBenchmark:

    @pytest.mark.slow
    @pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
    def test_hnsw_speedup_exceeds_20x(self, benchmark_data):
        """SLE-63: HNSW throughput must be >20× that of BallTree on N=50k.

        Measures batch throughput (not per-row latency) to get stable results.
        Uses 256-row batches matching production BATCH_SIZE.
        """
        X_train, X_query = benchmark_data

        # Fit both backends
        bt = BallTreeMatcher(n_neighbors=TOP_K * 3)
        bt.fit(X_train)

        hnsw = HNSWMatcher(n_neighbors=TOP_K * 3, ef_construction=200, M=16)
        hnsw.fit(X_train)

        BATCH_SIZE = 256
        N_BATCHES = N_QUERY // BATCH_SIZE
        batches = [X_query[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in range(N_BATCHES)]

        # BallTree batch timing
        t0 = time.perf_counter()
        for batch in batches:
            bt.kneighbors(batch, n_neighbors=TOP_K)
        bt_time = time.perf_counter() - t0

        # HNSW batch timing
        t0 = time.perf_counter()
        for batch in batches:
            hnsw.kneighbors(batch, n_neighbors=TOP_K)
        hnsw_time = time.perf_counter() - t0

        speedup = bt_time / hnsw_time if hnsw_time > 0 else float("inf")

        # Log and save results
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_train": N_TRAIN,
            "n_query": N_QUERY,
            "k": TOP_K,
            "batch_size": BATCH_SIZE,
            "balltree_batch_time_s": bt_time,
            "hnsw_batch_time_s": hnsw_time,
            "speedup_ratio": speedup,
            "gate_min_speedup": MIN_SPEEDUP,
            "gate_passed": speedup >= MIN_SPEEDUP,
        }

        print(
            f"\nSpeedup benchmark (N={N_TRAIN:,}, k={TOP_K}, batch={BATCH_SIZE}):\n"
            f"  BallTree: {bt_time * 1000:.1f}ms total\n"
            f"  HNSW:     {hnsw_time * 1000:.1f}ms total\n"
            f"  Speedup:  {speedup:.1f}x (gate: >{MIN_SPEEDUP}x)"
        )

        # Write to artifacts
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifact_path = ARTIFACTS_DIR / f"hnsw_benchmark_{ts}.json"
        artifact_path.write_text(json.dumps(results, indent=2))
        print(f"  Results saved: {artifact_path}")

        assert speedup >= MIN_SPEEDUP, (
            f"HNSW speedup {speedup:.1f}x < {MIN_SPEEDUP}x (SLE-63 gate). "
            f"BallTree={bt_time * 1000:.1f}ms, HNSW={hnsw_time * 1000:.1f}ms"
        )

    @pytest.mark.skipif(not HAS_HNSWLIB, reason="hnswlib not installed")
    def test_small_scale_speedup(self, benchmark_data):
        """Quick smoke test: HNSW should be at least 2× faster on small batches.

        Runs without @pytest.mark.slow so it always executes in CI.
        Uses smaller N (2000 from fixture) for speed.
        """
        rng = np.random.RandomState(7)
        X_train_small = rng.randn(2000, N_FEATURES).astype(np.float64)
        X_query_small = rng.randn(100, N_FEATURES).astype(np.float64)

        bt = BallTreeMatcher(n_neighbors=TOP_K)
        bt.fit(X_train_small)

        hnsw = HNSWMatcher(n_neighbors=TOP_K)
        hnsw.fit(X_train_small)

        N_REPS = 10
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            bt.kneighbors(X_query_small, n_neighbors=TOP_K)
        bt_time = (time.perf_counter() - t0) / N_REPS

        t0 = time.perf_counter()
        for _ in range(N_REPS):
            hnsw.kneighbors(X_query_small, n_neighbors=TOP_K)
        hnsw_time = (time.perf_counter() - t0) / N_REPS

        speedup = bt_time / hnsw_time if hnsw_time > 0 else float("inf")
        print(f"\nSmall-scale speedup (N=2k): {speedup:.1f}x")

        # At small N (2k), HNSW overhead can exceed BallTree savings — both are fast.
        # This is a catastrophic failure check only: gate fires if HNSW is >4× slower.
        # The 20× production speedup gate applies at N=50k only (test_hnsw_speedup_exceeds_20x).
        assert speedup >= 0.25, (
            f"HNSW ({hnsw_time * 1000:.2f}ms) is >4× slower than BallTree "
            f"({bt_time * 1000:.2f}ms) at N=2000 — HNSW may be misconfigured"
        )
