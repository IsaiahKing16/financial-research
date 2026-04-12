"""
scripts/diagnostics/murphy_gate.py

Machine-readable gate between Phase A (Murphy decomposition) and Phase B (experiments).

Purpose
-------
The Murphy (1973) BSS decomposition determines the dominant failure mode:
  - Resolution ≈ 0  → model has no discriminative signal at 585T scale.
                       Pool dilution destroyed the KNN signal entirely.
                       H4 (calibration) CANNOT fix this. H1/H3 are essential first.
  - Reliability >> 0 → probabilities are systematically biased relative to outcomes.
                       Calibration fixes (H4) are the primary lever.
  - Mixed            → signal quality and calibration both need work.

This class writes a JSON gate file after decomposition and provides a classmethod
that Phase B scripts call at startup to enforce authorized experiment ordering.

Usage (in b3_murphy_decomposition.py after computing decomposition):
    MurphyGate.write(
        mean_resolution=0.000123,
        mean_reliability=0.004567,
        fold_results=fold_results_list,
    )

Usage (in each Phase B experiment script, at the top of main()):
    gate = MurphyGate.load_and_enforce(caller="H4")
    # Raises RuntimeError if H4 is not yet authorized.
    # Returns gate dict for logging/provenance if authorized.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


# Path is relative to project root. Adjust if scripts/ is not at root.
_GATE_FILE = Path("results/murphy_gate.json")

# Thresholds derived from the literature (KNN_Calibration_Phase_1.pdf, §4):
#   Resolution < 0.001 → near-zero signal (pool dilution dominant)
#   Reliability > 0.002 → systematic calibration bias present
_RESOLUTION_NEAR_ZERO_THRESHOLD = 0.001
_RELIABILITY_DOMINANT_THRESHOLD = 0.002


class MurphyGate:
    """Enforcement gate between Murphy decomposition and Phase B experiments."""

    # Maps each experiment to which failure mode authorizes it.
    # "always" = run regardless of Murphy result.
    # "signal_quality" = only authorized when Resolution is the dominant issue.
    # "calibration" = only authorized when Reliability is the dominant issue.
    # "any" = authorized under both failure modes.
    _EXPERIMENT_AUTHORIZATION: dict[str, str] = {
        "H1":  "any",          # distance threshold + weighting: fixes signal quality
        "H2":  "any",          # sector filtering: fixes signal quality
        "H3":  "any",          # top_k reduction: fixes signal quality
        "H4":  "any",          # beta calibration: fixes calibration bias
        # NOTE: H4 is authorized "any" because beta calibration does no harm
        # when run after H1/H2/H3. What this gate PREVENTS is running H4 *first*
        # when Resolution ≈ 0, which would falsely attribute zero BSS to calibration.
        # The ordering enforcement is handled by _H4_REQUIRES_SIGNAL_FIX_FIRST below.
    }

    # H4 is blocked from running FIRST when Resolution is near zero.
    # It may only run after at least one signal-quality experiment has been attempted.
    _H4_REQUIRES_SIGNAL_FIX_FIRST = True

    @classmethod
    def write(
        cls,
        mean_resolution: float,
        mean_reliability: float,
        fold_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compute gate state and write results/murphy_gate.json.

        Called by b3_murphy_decomposition.py after computing decomposition.

        Args:
            mean_resolution: Mean Resolution component across all folds.
            mean_reliability: Mean Reliability component across all folds.
            fold_results: List of per-fold dicts with keys:
                fold, bs, bss, uncertainty, resolution, reliability, n_queries, base_rate

        Returns:
            The gate dict (same content as written to JSON).

        Raises:
            RuntimeError: If results/ directory does not exist and cannot be created.
        """
        resolution_near_zero = mean_resolution < _RESOLUTION_NEAR_ZERO_THRESHOLD
        reliability_dominant = mean_reliability > _RELIABILITY_DOMINANT_THRESHOLD

        if resolution_near_zero and not reliability_dominant:
            dominant_failure_mode = "resolution"
            recommended_branch = "signal_quality_first"
            h4_blocked_reason = (
                f"Resolution={mean_resolution:.6f} is near zero — the KNN model has no "
                f"discriminative signal at 585T scale. Beta calibration cannot fix a "
                f"model with zero resolution. Run H1/H3 (signal quality) first. "
                f"Threshold: resolution < {_RESOLUTION_NEAR_ZERO_THRESHOLD}"
            )
        elif reliability_dominant and not resolution_near_zero:
            dominant_failure_mode = "reliability"
            recommended_branch = "calibration_first"
            h4_blocked_reason = None
        else:
            dominant_failure_mode = "mixed"
            recommended_branch = "combined"
            h4_blocked_reason = None

        gate = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "config_at_generation": {
                "max_distance": 1.1019,
                "top_k": 50,
                "distance_weighting": "uniform",
                "calibration": "platt",
                "n_tickers": 585,
            },
            "decomposition": {
                "mean_resolution": mean_resolution,
                "mean_reliability": mean_reliability,
                "fold_results": fold_results,
            },
            "diagnosis": {
                "resolution_near_zero": resolution_near_zero,
                "reliability_dominant": reliability_dominant,
                "dominant_failure_mode": dominant_failure_mode,
                "recommended_branch": recommended_branch,
            },
            "experiment_authorization": {
                "H1": True,
                "H2": True,
                "H3": True,
                "H4": h4_blocked_reason is None,
                "H4_blocked_reason": h4_blocked_reason,
                "signal_fix_attempted": False,  # Updated by each H1/H2/H3 script on success
            },
            "phase_b_complete": False,
            "winning_config": None,
        }

        _GATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_GATE_FILE, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)

        print(f"\n[MurphyGate] Gate file written: {_GATE_FILE}")
        print(f"[MurphyGate] dominant_failure_mode = {dominant_failure_mode}")
        print(f"[MurphyGate] recommended_branch    = {recommended_branch}")
        if h4_blocked_reason:
            print(f"[MurphyGate] H4 BLOCKED: {h4_blocked_reason}")

        return gate

    @classmethod
    def load(cls) -> dict[str, Any]:
        """Load the gate file. Raises RuntimeError if not found.

        Returns:
            Gate dict from results/murphy_gate.json.

        Raises:
            RuntimeError: If gate file does not exist (Phase A not complete).
        """
        if not _GATE_FILE.exists():
            raise RuntimeError(
                f"Murphy gate file not found at {_GATE_FILE}. "
                f"Phase A (Murphy decomposition) must run before any Phase B experiment. "
                f"Run: PYTHONUTF8=1 py -3.12 scripts/diagnostics/b3_murphy_decomposition.py"
            )
        with open(_GATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_and_enforce(cls, caller: str) -> dict[str, Any]:
        """Load gate and raise RuntimeError if caller is not authorized.

        Call this at the top of main() in every Phase B experiment script.

        Args:
            caller: One of "H1", "H2", "H3", "H4".

        Returns:
            Gate dict if authorized.

        Raises:
            RuntimeError: If Phase A not complete OR if this experiment is blocked.
        """
        if caller not in ("H1", "H2", "H3", "H4"):
            raise RuntimeError(
                f"Unknown caller '{caller}'. Must be one of: H1, H2, H3, H4."
            )

        gate = cls.load()
        auth = gate["experiment_authorization"]

        if not auth.get(caller, False):
            reason = auth.get(f"{caller}_blocked_reason", "No reason recorded.")
            raise RuntimeError(
                f"Experiment {caller} is NOT authorized by the Murphy gate.\n"
                f"Reason: {reason}\n"
                f"Recommended branch: {gate['diagnosis']['recommended_branch']}\n"
                f"Re-read the gate file at {_GATE_FILE} and follow the recommended "
                f"branch before running {caller}."
            )

        diag = gate["diagnosis"]
        print(f"\n[MurphyGate] {caller} authorized.")
        print(f"[MurphyGate] dominant_failure_mode = {diag['dominant_failure_mode']}")
        print(f"[MurphyGate] recommended_branch    = {diag['recommended_branch']}")
        if diag["resolution_near_zero"]:
            print(
                f"[MurphyGate] WARNING: Resolution is near zero. "
                f"Signal quality is the primary issue. "
                f"Calibration-only fixes will not rescue BSS."
            )

        return gate

    @classmethod
    def mark_signal_fix_attempted(cls) -> None:
        """Called by H1/H2/H3 scripts when they complete a run.

        Unlocks H4 if it was blocked due to Resolution-near-zero.
        H4 is only meaningful after at least one signal-quality intervention
        has been tested — combined signal+calibration is valid; calibration-first
        when Resolution=0 is not.
        """
        gate = cls.load()
        gate["experiment_authorization"]["signal_fix_attempted"] = True
        # Unblock H4 now that signal fix has been attempted
        gate["experiment_authorization"]["H4"] = True
        gate["experiment_authorization"]["H4_blocked_reason"] = None
        with open(_GATE_FILE, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
        print(f"[MurphyGate] signal_fix_attempted=True. H4 is now unlocked.")

    @classmethod
    def record_winning_config(
        cls,
        experiment: str,
        params: dict[str, Any],
        mean_bss: float,
        positive_folds: int,
    ) -> None:
        """Record the winning parameter combination when gate is met.

        Called by phase1_summary.py once a combination achieves BSS > 0 on ≥ 3/6 folds.

        Args:
            experiment: e.g. "H1", "H4", "H1+H4"
            params: dict of winning parameter values
            mean_bss: mean BSS across 6 folds
            positive_folds: count of folds with BSS > 0
        """
        if positive_folds < 3:
            raise RuntimeError(
                f"record_winning_config called with positive_folds={positive_folds}. "
                f"Gate requires ≥ 3. Do not record a winner that has not met the gate."
            )
        gate = cls.load()
        gate["phase_b_complete"] = True
        gate["winning_config"] = {
            "recorded_at": datetime.utcnow().isoformat() + "Z",
            "experiment": experiment,
            "params": params,
            "mean_bss": mean_bss,
            "positive_folds": positive_folds,
        }
        with open(_GATE_FILE, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
        print(f"\n[MurphyGate] Winning config recorded: {experiment} "
              f"mean_BSS={mean_bss:+.5f} positive_folds={positive_folds}/6")
