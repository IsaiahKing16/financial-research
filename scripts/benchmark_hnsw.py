"""
benchmark_hnsw.py — Latency and recall benchmark for HNSWIndex vs ball_tree.

Validates two SLE-47 success criteria:
    1. recall@50 ≥ 0.95 vs exact ball_tree on held-out set
    2. Query latency < 10 ms/query at 50k training fingerprints

Uses data from data/processed/full_db.parquet (52 tickers, 202k rows).
Falls back to synthetic data if parquet not found.

Usage:
    python scripts/benchmark_hnsw.py
"""

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from research.hnsw_distance import HNSWIndex
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

try:
    import hnswlib  # noqa: F401
except ImportError:
    print("ERROR: hnswlib not installed.")
    print("Run: python -m pip install hnswlib")
    sys.exit(1)

from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Load or generate fingerprint data
# ---------------------------------------------------------------------------

def load_fingerprints():
    parquet_path = REPO_ROOT / "data" / "processed" / "full_db.parquet"
    feature_cols = [f"ret_{w}d" for w in [1, 3, 7, 14, 30, 45, 60, 90]]

    if parquet_path.exists():
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        db = pd.read_parquet(parquet_path)
        X = db[feature_cols].dropna().values.astype(np.float32)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        source = f"real data ({parquet_path.name}, {len(X):,} rows)"
    else:
        print("  [WARNING] data/processed/full_db.parquet not found — using synthetic data")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50_000, 8)).astype(np.float32)
        source = "synthetic (50k × 8-dim)"

    return X, source


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark_recall(X_train, X_query, k=50):
    """Compute recall@k: fraction of exact top-k neighbours found by HNSW."""
    print(f"\n--- recall@{k} benchmark ---")
    print(f"  N_train={len(X_train):,}  N_query={len(X_query):,}  k={k}")

    # Exact ground truth
    t0 = time.perf_counter()
    exact = NearestNeighbors(n_neighbors=k, metric="euclidean",
                             algorithm="ball_tree", n_jobs=1)
    exact.fit(X_train)
    exact_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, exact_idx = exact.kneighbors(X_query)
    exact_query = time.perf_counter() - t0

    print(f"  ball_tree  build: {exact_build*1000:.0f} ms  "
          f"query ({len(X_query)} rows): {exact_query*1000:.0f} ms  "
          f"({exact_query/len(X_query)*1000:.2f} ms/query)")

    # HNSW approximate
    t0 = time.perf_counter()
    hnsw = HNSWIndex(n_neighbors=k, dim=X_train.shape[1])
    hnsw.fit(X_train)
    hnsw_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, hnsw_idx = hnsw.kneighbors(X_query, n_neighbors=k)
    hnsw_query = time.perf_counter() - t0

    ms_per_query = hnsw_query / len(X_query) * 1000
    print(f"  HNSWIndex  build: {hnsw_build*1000:.0f} ms  "
          f"query ({len(X_query)} rows): {hnsw_query*1000:.0f} ms  "
          f"({ms_per_query:.2f} ms/query)")

    # Recall
    recalls = []
    for i in range(len(X_query)):
        true_set = set(exact_idx[i])
        found_set = set(hnsw_idx[i])
        recalls.append(len(true_set & found_set) / k)
    mean_recall = float(np.mean(recalls))

    recall_pass = mean_recall >= 0.95
    latency_pass = ms_per_query < 10.0
    speedup = exact_query / hnsw_query if hnsw_query > 0 else float("inf")

    print(f"\n  recall@{k}:      {mean_recall:.4f}  {'✓ PASS' if recall_pass else '✗ FAIL'} (threshold: 0.95)")
    print(f"  ms/query:       {ms_per_query:.2f}    {'✓ PASS' if latency_pass else '✗ FAIL'} (threshold: 10 ms)")
    print(f"  speedup:        {speedup:.1f}×  vs ball_tree")

    return mean_recall, ms_per_query, recall_pass, latency_pass


def benchmark_latency_at_scale(k=50):
    """Extrapolate query latency to 50k training fingerprints."""
    print("\n--- latency-at-scale benchmark (50k fingerprints) ---")
    sizes = [10_000, 25_000, 50_000]
    rng = np.random.default_rng(99)

    for N in sizes:
        X_train = rng.standard_normal((N, 8)).astype(np.float32)
        X_query = rng.standard_normal((100, 8)).astype(np.float32)

        hnsw = HNSWIndex(n_neighbors=k, dim=8).fit(X_train)

        t0 = time.perf_counter()
        hnsw.kneighbors(X_query, n_neighbors=k)
        elapsed = time.perf_counter() - t0
        ms_per_q = elapsed / len(X_query) * 1000

        gate = "✓" if ms_per_q < 10.0 else "✗"
        print(f"  N={N:>6,}: {ms_per_q:.2f} ms/query {gate}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  HNSW Benchmark — recall@50 + query latency")
    print("=" * 60)

    X, source = load_fingerprints()
    print(f"\n  Data: {source}")

    # Split: use last 500 rows as query, rest as train
    N_query = min(500, len(X) // 10)
    X_train = X[:-N_query]
    X_query = X[-N_query:]

    mean_recall, ms_per_query, recall_pass, latency_pass = benchmark_recall(
        X_train, X_query, k=50
    )

    benchmark_latency_at_scale(k=50)

    # Final verdict
    all_pass = recall_pass and latency_pass
    print("\n" + "=" * 60)
    print(f"  SLE-47 GATE: {'✓ CLEARED' if all_pass else '✗ NOT CLEARED'}")
    if recall_pass:
        print(f"    recall@50 = {mean_recall:.4f}  ≥ 0.95  ✓")
    else:
        print(f"    recall@50 = {mean_recall:.4f}  < 0.95  ✗  — increase ef_construction or M")
    if latency_pass:
        print(f"    latency   = {ms_per_query:.2f} ms/q  < 10 ms  ✓")
    else:
        print(f"    latency   = {ms_per_query:.2f} ms/q  ≥ 10 ms  ✗  — tune ef at query time")
    print("=" * 60)


if __name__ == "__main__":
    main()
