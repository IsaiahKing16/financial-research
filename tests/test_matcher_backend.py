"""Tests for pattern_engine.matcher_backend."""

import numpy as np
import pytest

from pattern_engine.matcher_backend import (
    HNSWMatcherBackend,
    SklearnNNBackend,
    build_matcher_backend,
)


def test_sklearn_backend_matches_requested_shapes():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    backend = SklearnNNBackend(
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
        n_neighbors=3,
    )
    backend.fit(X)

    distances, indices = backend.kneighbors(np.array([[0.0, 0.0]], dtype=float), n_neighbors=2)

    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert indices[0, 0] == 0
    assert pytest.approx(distances[0, 0]) == 0.0


def test_sklearn_backend_requires_fit_before_query():
    backend = SklearnNNBackend(
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
        n_neighbors=3,
    )
    with pytest.raises(RuntimeError, match="fit"):
        backend.kneighbors(np.array([[0.0, 0.0]], dtype=float))


def test_factory_returns_sklearn_backend_by_default():
    backend = build_matcher_backend(
        use_hnsw=False,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=1,
        n_neighbors=10,
    )
    assert isinstance(backend, SklearnNNBackend)


def test_factory_rejects_non_euclidean_hnsw_initially():
    with pytest.raises(ValueError, match="euclidean"):
        build_matcher_backend(
            use_hnsw=True,
            metric="cosine",
            algorithm="brute",
            n_jobs=1,
            n_neighbors=10,
        )


def test_hnsw_backend_dimension_guard():
    backend = HNSWMatcherBackend(n_neighbors=2)
    backend._index = object()
    backend._dim = 3
    backend._row_count = 5

    with pytest.raises(ValueError, match="dimension mismatch"):
        backend.kneighbors(np.ones((1, 2), dtype=np.float32))
