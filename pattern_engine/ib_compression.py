"""
ib_compression.py — Information Bottleneck compression pilot (SLE-78).

Compresses the 8-dimensional FPPE return fingerprint to a lower-dimensional
representation that retains only information predictive of the target
(fwd_7d_up), discarding noise dimensions that inflate nearest-neighbour
distance without improving matching quality.

The Information Bottleneck (IB) principle (Tishby et al. 1999):
    min I(X; Z) − β·I(Z; Y)
    subject to: Z is a compressed representation of X.

For FPPE, X = scaled return features, Y = fwd_7d_up.  The IB objective
finds the best trade-off between compression (low I(X; Z)) and retention
of predictive information (high I(Z; Y)), controlled by β.

Practical implementation:
    We use a linear approximation (sufficient statistics approach) equivalent
    to IB for Gaussian features.  The algorithm finds the d_out-dimensional
    linear subspace of X that maximises I(Z; Y) via the Kernel IB bound.

    For d_out < D (e.g., 4 out of 8 dimensions), this is equivalent to
    selecting the principal directions most correlated with Y — a supervised
    PCA.  Full nonlinear IB requires iterative algorithms (Blahut-Arimoto);
    that is left as a future enhancement.

When IB helps vs hurts:
    HELPS: When some return windows are noisy for the current regime (e.g.,
           30-day returns are noisy in a sideways market).  IB compression
           can suppress these dimensions.
    HURTS: When all features contribute independent predictive signal.
           Compression then discards useful information.
    VERDICT: Run BSS comparison with and without compression to determine.

Feature flag:
    PatternMatcher reads `getattr(config, 'use_ib_compression', False)`.
    When True, X_train is passed through IBCompressor.fit_transform() before
    fitting the BallTree/HNSW index.

Linear: SLE-78
"""

from __future__ import annotations

import numpy as np

# ─── IBCompressor ─────────────────────────────────────────────────────────────

class IBCompressor:
    """Linear IB compressor via supervised PCA (Gaussian features).

    Finds the linear projection of X that maximises correlation with Y
    by computing the covariance between each feature and the label, then
    selecting the dimensions with highest predictive variance.

    This is the Gaussian IB lower bound — equivalent to IB for Gaussian
    (X, Y) pairs.  It is computationally O(D²) and requires no iteration.

    Args:
        d_out:     Output dimensionality (compressed). Must be < D.
        method:    Compression method. Currently only "supervised_pca".
    """

    def __init__(
        self,
        d_out: int = 4,
        method: str = "supervised_pca",
    ) -> None:
        if d_out < 1:
            raise ValueError(f"d_out must be >= 1, got {d_out}")
        if method not in ("supervised_pca",):
            raise ValueError(f"Unsupported method: {method!r}")
        self.d_out = d_out
        self.method = method
        self._projection: np.ndarray | None = None   # (D, d_out)
        self._feature_importance: np.ndarray | None = None  # (D,) scores

    def fit(self, X: np.ndarray, y: np.ndarray) -> IBCompressor:
        """Fit the compression projection on training data.

        Args:
            X: (N, D) scaled training features.
            y: (N,) binary labels (fwd_7d_up).

        Returns:
            self (for chaining).
        """
        N, D = X.shape
        if self.d_out >= D:
            raise ValueError(
                f"d_out={self.d_out} must be < feature_dim={D}; "
                "no compression needed."
            )

        # Supervised PCA: find directions that explain covariance with y.
        # Step 1: correlation-weighted PCA.
        # Compute correlation of each feature with y.
        y_mean = y.mean()
        y_centered = y - y_mean

        # Covariance matrix of X weighted by |corr(X_i, y)|
        correlations = np.abs(
            np.dot(X.T, y_centered) / (N * X.std(axis=0).clip(1e-10))
        )  # (D,)
        self._feature_importance = correlations

        # Weight the feature space: X_weighted = X * sqrt(corr)
        weights = np.sqrt(correlations + 1e-8)   # add epsilon for stability
        X_w = X * weights[np.newaxis, :]

        # SVD of weighted X for the top d_out directions
        U, S, Vt = np.linalg.svd(X_w, full_matrices=False)
        # Vt is (min(N,D), D); take first d_out rows → (d_out, D).
        # The feature weighting was already applied to X_w above (X_w = X * weights);
        # do NOT re-apply weights here — that would double-weight the features.
        self._projection = Vt[:self.d_out].T  # (D, d_out)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X into the compressed representation.

        Args:
            X: (N, D) scaled features (or (D,) for a single row).

        Returns:
            (N, d_out) or (d_out,) compressed features.
        """
        if self._projection is None:
            raise RuntimeError("IBCompressor.fit() must be called before transform()")
        return X @ self._projection

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one call.

        Args:
            X: (N, D) scaled training features.
            y: (N,) binary labels.

        Returns:
            (N, d_out) compressed training features.
        """
        return self.fit(X, y).transform(X)

    @property
    def feature_importance(self) -> np.ndarray | None:
        """Per-feature predictive importance scores (correlations with y)."""
        return self._feature_importance

    def top_features(self, feature_names: list[str]) -> list[str]:
        """Return feature names ranked by predictive importance (descending).

        Args:
            feature_names: List of D feature names.

        Returns:
            List of D feature names sorted most-important first.
        """
        if self._feature_importance is None:
            raise RuntimeError("IBCompressor.fit() must be called first")
        ranked = np.argsort(self._feature_importance)[::-1]
        return [feature_names[i] for i in ranked]


# ─── BSS comparison utility ───────────────────────────────────────────────────

def compare_bss_with_ib(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    d_out: int = 4,
) -> dict:
    """Compare KNN matching quality with and without IB compression.

    This is the acceptance-criterion comparison for SLE-78.  Both regimes
    use the same KNN logic (top_k=50, uniform weighting) with sklearn's
    BallTree; only the feature space differs.

    Args:
        X_train:  (N_train, D) scaled training features.
        y_train:  (N_train,) binary labels.
        X_val:    (N_val, D) scaled validation features.
        y_val:    (N_val,) binary labels.
        d_out:    IB output dimensionality.

    Returns:
        Dict with keys:
            bss_baseline (float): BSS without IB compression (all D features).
            bss_ib (float):       BSS with IB compression (d_out features).
            delta_bss (float):    bss_ib - bss_baseline (positive = IB helps).
            d_in (int):           Input feature dimensionality.
            d_out (int):          IB output dimensionality.
    """
    from sklearn.neighbors import BallTree

    top_k = min(50, len(X_train) - 1)

    def _knn_bss(X_tr, X_vl, y_tr, y_vl, k):
        tree = BallTree(X_tr, metric="euclidean")
        dists, idx = tree.query(X_vl, k=k)
        probs = y_tr[idx].mean(axis=1)
        brier = float(np.mean((probs - y_vl) ** 2))
        brier_clim = float(np.var(y_vl))
        return 1.0 - brier / brier_clim if brier_clim > 0 else 0.0

    bss_baseline = _knn_bss(X_train, X_val, y_train, y_val, top_k)

    compressor = IBCompressor(d_out=d_out)
    X_train_compressed = compressor.fit_transform(X_train, y_train)
    X_val_compressed = compressor.transform(X_val)
    bss_ib = _knn_bss(X_train_compressed, X_val_compressed, y_train, y_val, top_k)

    # Round baseline values first, then compute delta from rounded values.
    # This ensures delta_bss == round(bss_ib - bss_baseline, 6) at the
    # precision level callers observe, avoiding 1-ULP mismatch when both
    # sides are independently rounded.
    bss_baseline_r = round(bss_baseline, 6)
    bss_ib_r = round(bss_ib, 6)
    return {
        "bss_baseline": bss_baseline_r,
        "bss_ib": bss_ib_r,
        "delta_bss": round(bss_ib_r - bss_baseline_r, 6),
        "d_in": X_train.shape[1],
        "d_out": d_out,
    }
