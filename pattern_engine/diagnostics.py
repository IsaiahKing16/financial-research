"""pattern_engine/diagnostics.py — Phase 7.5 diagnostic gate functions.

Public API:
    braess_gate(full_db, feature_cols_with, feature_cols_without, folds, cfg_overrides) -> dict
    identifiability_gate(training_n, k, min_ratio) -> dict

G7.5-2: braess_gate — verifies adding a feature group does not degrade BSS.
G7.5-3: identifiability_gate — confirms effective parameters << training samples.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import icontract
import numpy as np

from pattern_engine.walkforward import run_walkforward

if TYPE_CHECKING:
    import pandas as pd

_BRAESS_WIN_THRESHOLD = 4  # wins_with >= 4 of 6 folds → PASS


def _compare_fold_bss(
    folds_with: list[dict],
    folds_without: list[dict],
) -> tuple[list[float], int]:
    """Return (fold_deltas, wins_with) comparing per-fold BSS for two feature sets."""
    if len(folds_with) != len(folds_without):
        raise RuntimeError(
            f"braess_gate: fold count mismatch {len(folds_with)} vs {len(folds_without)}"
        )
    fold_deltas: list[float] = []
    wins_with = 0
    for fw, fwo in zip(folds_with, folds_without, strict=True):
        bss_w = fw["bss"]
        bss_wo = fwo["bss"]
        if math.isnan(bss_w) or math.isnan(bss_wo):
            raise RuntimeError(
                f"braess_gate: NaN BSS in fold — bss_with={bss_w}, bss_without={bss_wo}. "
                "Fold has insufficient data to determine winner."
            )
        delta = bss_w - bss_wo
        fold_deltas.append(delta)
        if delta > 0.0:
            wins_with += 1
    return fold_deltas, wins_with


@icontract.require(lambda feature_cols_with: len(feature_cols_with) > 0)
@icontract.require(lambda feature_cols_without: len(feature_cols_without) > 0)
@icontract.ensure(lambda result: result["verdict"] in ("PASS", "FAIL"))
@icontract.ensure(lambda result: "wins_with" in result and "fold_deltas" in result)
def braess_gate(
    full_db: pd.DataFrame,
    feature_cols_with: list[str],
    feature_cols_without: list[str],
    folds: list[dict] | None = None,
    cfg_overrides: dict | None = None,
) -> dict:
    """Braess gate: verify adding features does not degrade per-fold BSS.

    PASS = feature_cols_with wins >= 4/6 folds vs feature_cols_without.
    Named after Braess's paradox: adding a resource can worsen system performance.

    Returns:
        {
          "verdict": "PASS" | "FAIL",
          "wins_with": int,           # folds where BSS(with) > BSS(without)
          "fold_deltas": list[float], # per-fold BSS(with) - BSS(without)
          "mean_bss_with": float,
          "mean_bss_without": float,
          "n_folds": int,
        }
    """
    result_with = run_walkforward(
        full_db=full_db,
        feature_cols=feature_cols_with,
        cfg_overrides=cfg_overrides,
        folds=folds,
    )
    result_without = run_walkforward(
        full_db=full_db,
        feature_cols=feature_cols_without,
        cfg_overrides=cfg_overrides,
        folds=folds,
    )

    folds_with = result_with["fold_results"]
    folds_without = result_without["fold_results"]
    fold_deltas, wins_with = _compare_fold_bss(folds_with, folds_without)

    bss_vals_with = [fw["bss"] for fw in folds_with]
    bss_vals_without = [fw["bss"] for fw in folds_without]
    verdict = "PASS" if wins_with >= _BRAESS_WIN_THRESHOLD else "FAIL"

    return {
        "verdict": verdict,
        "wins_with": wins_with,
        "fold_deltas": fold_deltas,
        "mean_bss_with": float(np.mean(bss_vals_with)),
        "mean_bss_without": float(np.mean(bss_vals_without)),
        "n_folds": len(fold_deltas),
    }


@icontract.require(lambda training_n: training_n > 0)
@icontract.require(lambda k: k > 0)
@icontract.require(lambda min_ratio: min_ratio > 0.0)
@icontract.ensure(lambda result: result["verdict"] in ("PASS", "FAIL"))
@icontract.ensure(lambda result: result["ratio"] > 0.0)
def identifiability_gate(
    training_n: int,
    k: int,
    min_ratio: float = 20.0,
) -> dict:
    """Identifiability gate: confirm effective parameters << training samples.

    Uses Hastie et al. (2009) effective df for local methods: df ≈ N/k.
    PASS = training_n / k >= min_ratio (threshold 20:1 conservative for KNN).

    Returns:
        {
          "verdict": "PASS" | "FAIL",
          "ratio": float,           # training_n / effective_params
          "training_n": int,
          "effective_params": int,  # k (Hastie effective df for local methods)
          "min_ratio": float,
        }
    """
    effective_params = k
    ratio = training_n / effective_params
    verdict = "PASS" if ratio >= min_ratio else "FAIL"

    return {
        "verdict": verdict,
        "ratio": ratio,
        "training_n": training_n,
        "effective_params": effective_params,
        "min_ratio": min_ratio,
    }
