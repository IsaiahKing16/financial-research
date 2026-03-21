"""
bma_calibrator.py — Bayesian Model Averaging calibrator for FPPE analogue ensembles.

Fits a K-component Student's t mixture via EM.  Each component i represents
one analogue slot; its location is the raw probability from analogue i and its
scale is estimated from training residuals.

Math summary
------------
E-step:
    log_pdf[n, i] = log t_pdf(y[n] | loc=raw_probs[n,i], scale=sqrt(var[i]), df=3)
    latent_probs[n, i] ∝ w[i] * exp(log_pdf[n, i])   # row-normalised

M-step (Gaussian variance approximation — deliberate simplification):
    w[i]   = mean_n(latent_probs[n, i])
    var[i] = sum_n(latent_probs[n,i] * (y[n] - raw_probs[n,i])^2) / sum_n(latent_probs[n,i])

transform(raw_probs):
    out = sum_i(w[i] * raw_probs[i])   # posterior-weighted mean, guaranteed in [0,1]

generate_pdf(analogue_probs, return_grid):
    mixture PDF evaluated on return_grid — useful for uncertainty quantification.
"""

import numpy as np
from scipy.stats import t as t_dist

from research import BaseCalibrator

_DF = 3.0          # Student's t degrees of freedom
_N_ITER = 30       # EM iterations (empirically sufficient for convergence)
_MIN_VAR = 1e-6    # Variance floor to prevent numerical collapse


class BMACalibrator(BaseCalibrator):
    """Bayesian Model Averaging calibrator using a Student's t mixture via EM.

    Args:
        n_iter: Number of EM iterations (default 30).
        df:     Degrees of freedom for the Student's t components (default 3.0).
    """

    def __init__(self, n_iter: int = _N_ITER, df: float = _DF) -> None:
        self.n_iter = n_iter
        self.df = df
        self._fitted = False
        self.weights: np.ndarray | None = None
        self.variances: np.ndarray | None = None

    # ------------------------------------------------------------------
    # BaseCalibrator interface
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "BMACalibrator":
        """Fit BMA via EM on training data.

        Args:
            raw_probs: Shape (N, K) — K raw probabilities per training sample.
            y_true:    Shape (N,)   — binary labels {0, 1}.

        Returns:
            self
        """
        raw_probs = np.asarray(raw_probs, dtype=float)  # (N, K)
        y_true = np.asarray(y_true, dtype=float)        # (N,)
        N, K = raw_probs.shape

        # Initialise with uniform weights and small variance
        weights = np.full(K, 1.0 / K)
        variances = np.full(K, 0.1)

        for _ in range(self.n_iter):
            # --- E-step: compute un-normalised log responsibilities ---
            log_resps = np.zeros((N, K))
            for i in range(K):
                scale_i = float(np.sqrt(max(variances[i], _MIN_VAR)))
                log_resps[:, i] = (
                    np.log(weights[i] + 1e-300)
                    + t_dist.logpdf(y_true, df=self.df, loc=raw_probs[:, i], scale=scale_i)
                )

            # Numerically stable row-wise softmax
            log_resps -= log_resps.max(axis=1, keepdims=True)
            resps = np.exp(log_resps)
            resps /= resps.sum(axis=1, keepdims=True) + 1e-300  # (N, K)

            # --- M-step: update weights and variances ---
            resp_sums = resps.sum(axis=0) + 1e-300  # (K,)
            weights = resp_sums / resp_sums.sum()

            for i in range(K):
                residuals_sq = (y_true - raw_probs[:, i]) ** 2
                variances[i] = float(
                    np.dot(resps[:, i], residuals_sq) / resp_sums[i]
                )
                variances[i] = max(variances[i], _MIN_VAR)

        self.weights = weights
        self.variances = variances
        self._fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        """Compute posterior-weighted calibrated probability.

        Args:
            raw_probs: Shape (K,) — one raw probability per analogue for a
                       single query point.

        Returns:
            Scalar float in [0, 1] — posterior-weighted mean.
        """
        if not self._fitted:
            raise RuntimeError("BMACalibrator.fit() must be called before transform().")
        raw_probs = np.asarray(raw_probs, dtype=float)
        return np.dot(self.weights, raw_probs)

    # ------------------------------------------------------------------
    # BMA-specific (not in ABC)
    # ------------------------------------------------------------------

    def generate_pdf(
        self, analogue_probs: np.ndarray, return_grid: np.ndarray
    ) -> np.ndarray:
        """Evaluate the BMA mixture PDF on a grid.

        Args:
            analogue_probs: Shape (K,) — raw probs from K analogues.
            return_grid:    Shape (M,) — grid of probability values in [0, 1].

        Returns:
            np.ndarray of shape (M,) — mixture PDF values (non-negative).
        """
        if not self._fitted:
            raise RuntimeError("BMACalibrator.fit() must be called before generate_pdf().")
        analogue_probs = np.asarray(analogue_probs, dtype=float)
        return_grid = np.asarray(return_grid, dtype=float)

        pdf = np.zeros(len(return_grid))
        for i, (w, var) in enumerate(zip(self.weights, self.variances)):
            scale_i = float(np.sqrt(max(var, _MIN_VAR)))
            pdf += w * t_dist.pdf(
                return_grid, df=self.df, loc=analogue_probs[i], scale=scale_i
            )
        return pdf
