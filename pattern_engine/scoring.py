"""pattern_engine/scoring.py — Phase 7.5 statistical scoring functions.

Public API:
    cv_bss_estimator(predictions, actuals, n_bootstrap, ci_level, rng_seed) -> dict

G7.5-5: Control-variate BSS estimator with bootstrap confidence interval.
Formula: BS_CV(model) = BS(model) - beta * (BS(clim) - E[BS(clim)])
         beta = cov(BS_model, BS_clim) / var(BS_clim)
         95% CI via bootstrap (n_bootstrap=1000)

Note: _murphy_decomposition() already exists at walkforward.py:91-129.
      Do NOT duplicate it here. This module uses Brier Score directly.
"""
from __future__ import annotations

import icontract
import numpy as np


def _brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean squared error of probability predictions vs binary actuals."""
    return float(np.mean((predictions - actuals) ** 2))


def _climatological_bs(actuals: np.ndarray) -> float:
    """Brier Score of the climatological (base-rate) forecast."""
    base_rate = float(actuals.mean())
    return base_rate * (1.0 - base_rate)


def _bss_point(bs_model: float, bs_clim: float) -> float:
    """Standard BSS point estimate. Returns 0.0 when bs_clim < 1e-10."""
    if bs_clim < 1e-10:
        return 0.0
    return 1.0 - bs_model / bs_clim


def _compute_cv_beta(
    bs_model_samples: np.ndarray,
    bs_clim_samples: np.ndarray,
) -> tuple[float, float]:
    """Compute CV beta and E[BS(clim)] from bootstrap samples.

    Returns (beta, expected_bs_clim).
    beta = cov(BS_model, BS_clim) / var(BS_clim); 0.0 if var(BS_clim) < 1e-15.
    """
    var_clim = float(np.var(bs_clim_samples, ddof=1))
    if var_clim < 1e-15:
        return 0.0, float(np.mean(bs_clim_samples))
    cov = float(np.cov(bs_model_samples, bs_clim_samples)[0, 1])
    beta = cov / var_clim
    expected_bs_clim = float(np.mean(bs_clim_samples))
    return beta, expected_bs_clim


def _bootstrap_cv_bss(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Bootstrap the CV-adjusted BSS. Returns (bss_cv_samples, beta, variance_reduction)."""
    n = len(predictions)
    bs_model_samples = np.empty(n_bootstrap)
    bs_clim_samples = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs_model_samples[i] = _brier_score(predictions[idx], actuals[idx])
        bs_clim_samples[i] = _climatological_bs(actuals[idx])

    beta, expected_bs_clim = _compute_cv_beta(bs_model_samples, bs_clim_samples)

    bss_cv_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        bs_cv_i = bs_model_samples[i] - beta * (bs_clim_samples[i] - expected_bs_clim)
        bs_clim_i = bs_clim_samples[i]
        bss_cv_samples[i] = _bss_point(bs_cv_i, bs_clim_i)

    rho = float(np.corrcoef(bs_model_samples, bs_clim_samples)[0, 1])
    variance_reduction = max(0.0, 1.0 - rho ** 2)

    return bss_cv_samples, beta, variance_reduction


@icontract.require(lambda predictions: len(predictions) > 0)
@icontract.require(lambda actuals: len(actuals) > 0)
@icontract.require(
    lambda predictions, actuals: len(predictions) == len(actuals),
    "predictions and actuals must be same length",
)
@icontract.require(lambda n_bootstrap: n_bootstrap >= 100)
@icontract.require(lambda ci_level: 0.0 < ci_level < 1.0)
@icontract.ensure(lambda result: "bss_point" in result and "bss_cv" in result)
@icontract.ensure(lambda result: "ci_lower" in result and "ci_upper" in result)
@icontract.ensure(lambda result: 0.0 <= result["variance_reduction"] <= 1.0)
def cv_bss_estimator(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 42,
) -> dict:
    """Control-variate BSS estimator with bootstrap confidence interval.

    BS_CV(model) = BS(model) - beta * (BS(clim) - E[BS(clim)])
    beta = cov(BS_model, BS_clim) / var(BS_clim)
    Variance reduction factor = (1 - rho^2) where rho = corr(BS_model, BS_clim).
    95% CI via bootstrap (n_bootstrap=1000 default).

    Decision criterion: ci_lower > 0 means the edge is statistically significant.

    Returns:
        {
          "bss_point": float,          # standard point estimate
          "bss_cv": float,             # control-variate adjusted estimate
          "ci_lower": float,           # CI lower bound
          "ci_upper": float,           # CI upper bound
          "variance_reduction": float, # 1 - rho^2
          "beta": float,               # CV regression coefficient
          "n": int,                    # sample size
        }
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    bs_model = _brier_score(predictions, actuals)
    bs_clim = _climatological_bs(actuals)
    bss_point = _bss_point(bs_model, bs_clim)

    rng = np.random.default_rng(rng_seed)
    bss_cv_samples, beta, variance_reduction = _bootstrap_cv_bss(
        predictions, actuals, n_bootstrap, rng
    )

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bss_cv_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(bss_cv_samples, 100 * (1.0 - alpha / 2)))
    bss_cv = float(np.mean(bss_cv_samples))

    return {
        "bss_point": bss_point,
        "bss_cv": bss_cv,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "variance_reduction": variance_reduction,
        "beta": beta,
        "n": len(predictions),
    }
