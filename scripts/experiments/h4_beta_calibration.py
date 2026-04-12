"""
scripts/experiments/h4_beta_calibration.py

Hypothesis H4: Replace Platt calibration with beta calibration.

Platt scaling assumes class-conditional score distributions are Gaussian with equal
variance (Böken 2021). KNN probability outputs violate this assumption: discrete
values at multiples of 1/k, compressed toward 0.5 by distant neighbours under pool
dilution. Platt can actively degrade calibration under these conditions (Caruana 2005).

Beta calibration (Kull et al. 2017, AISTATS ★★★) uses 3 parameters and includes the
Platt sigmoid as a special case, so it cannot harm already-calibrated scores.

Run after any H1/H2/H3 attempt (gate enforces this via signal_fix_attempted).
Also tests best H1 config + beta calibration combined.

Dependency: pip install betacal>=0.1.0
  Add 'betacal>=0.1.0' to requirements.txt before running.

Implementation: _PlattCalibrator is monkey-patched → _BetaCalibrator for each fold.
Production code (matcher.py) is NOT modified — patch is scoped to run_walkforward_beta().

Output: results/bss_fix_sweep_h4.tsv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.features import VOL_NORM_COLS
from scripts.diagnostics.murphy_gate import MurphyGate
from scripts.experiments.h1_max_distance_sweep import (
    _load_full_db,
    _get_best_h1_max_distance,
    GATE_THRESHOLD_POSITIVE_FOLDS,
)

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError(
        "betacal is not installed.\n"
        "Run: py -3.12 -m pip install betacal\n"
        "Then add 'betacal>=0.1.0' to requirements.txt."
    )


# ---------------------------------------------------------------------------
# Drop-in calibrator that mirrors _PlattCalibrator's interface
# ---------------------------------------------------------------------------

class _BetaCalibrator:
    """Beta calibration drop-in for PatternMatcher._calibrator.

    Kull et al. (2017, AISTATS) — handles non-sigmoidal KNN score distributions.
    parameters='abm': 3-parameter model; includes Platt sigmoid as special case.

    Interface is identical to PatternMatcher._PlattCalibrator so it can be
    injected without modifying matcher.py.
    """

    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Walk-forward runner that injects _BetaCalibrator
# ---------------------------------------------------------------------------

def run_walkforward_beta(config: EngineConfig, full_db: pd.DataFrame) -> list[dict]:
    """Run 6-fold walk-forward with beta calibration injected."""
    import pattern_engine.matcher as _matcher_module

    full_db = full_db.copy()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    fold_results = []
    horizon = config.projection_horizon

    for fold in WALKFORWARD_FOLDS:
        train_end = pd.to_datetime(fold["train_end"])
        val_start = pd.to_datetime(fold["val_start"])
        val_end   = pd.to_datetime(fold["val_end"])

        train_db = full_db[full_db["Date"] <= train_end].dropna(subset=[horizon]).copy()
        val_db   = full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ].dropna(subset=[horizon]).copy()

        # Monkey-patch _PlattCalibrator → _BetaCalibrator for this fold only.
        # This patches the class used inside fit()'s double-pass so no
        # production code is modified.
        original_calibrator = _matcher_module._PlattCalibrator
        try:
            _matcher_module._PlattCalibrator = _BetaCalibrator
            from pattern_engine.matcher import PatternMatcher
            matcher = PatternMatcher(config)
            matcher.fit(train_db, VOL_NORM_COLS)
            probs, _, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
        finally:
            # Always restore — even if an exception occurs
            _matcher_module._PlattCalibrator = original_calibrator

        y_true    = val_db[horizon].values.astype(float)
        base_rate = y_true.mean()
        bs_ref    = base_rate * (1 - base_rate)
        bs_model  = float(np.mean((np.asarray(probs) - y_true) ** 2))
        bss       = 1.0 - bs_model / bs_ref if bs_ref > 1e-10 else 0.0

        fold_results.append({
            "fold":    fold["label"],
            "bss":     round(bss, 6),
            "bs":      round(bs_model, 6),
            "avg_k":   round(float(np.mean(n_matches)), 2),
            "n_val":   len(val_db),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gate = MurphyGate.load_and_enforce(caller="H4")

    start = time.time()
    print("\n" + "=" * 70)
    print("  H4: BETA CALIBRATION SWEEP")
    print("=" * 70)
    print("\n  Tests beta calibration as drop-in Platt replacement.")
    print("  Also tests best H1 config + beta calibration combined.")

    best_max_d = _get_best_h1_max_distance()

    print(f"\n  Loading data...", end="", flush=True)
    full_db = _load_full_db()
    print(f" {len(full_db):,} rows")

    _fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    _tsv_cols = (
        ["config_label", "calibrator", "max_distance", "distance_weighting",
         "mean_bss", "positive_folds", "gate_met", "mean_avg_k"]
        + [f"bss_{lbl}" for lbl in _fold_labels]
    )
    results_path = project_root / "results" / "bss_fix_sweep_h4.tsv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "# bss_fix_sweep_h4.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        "# Hypothesis H4: beta calibration (Kull et al. 2017) vs Platt",
        "# betacal parameters='abm' (3-param; includes Platt sigmoid as special case)",
        f"# Gate: positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        f"# Rows appended incrementally — file is valid even if run is interrupted.",
        "#",
    ]

    # Resume: detect already-completed configs
    all_results = []
    _completed_labels: set[str] = set()
    if results_path.exists():
        try:
            _existing = pd.read_csv(results_path, comment="#", sep="\t")
            if "config_label" in _existing.columns and len(_existing) > 0:
                _completed_labels = set(_existing["config_label"].tolist())
                for _, _row in _existing.iterrows():
                    all_results.append(_row.to_dict())
                print(f"\n  [RESUME] Skipping completed configs: {sorted(_completed_labels)}")
        except Exception:
            pass

    if not _completed_labels:
        with open(results_path, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(_tsv_cols) + "\n")

    configs = [
        # Baseline config + beta cal
        {"label": "beta_baseline", "max_d": 1.1019, "weighting": "uniform"},
        # Best H1 config + beta cal
        {"label": "beta_best_h1",  "max_d": best_max_d, "weighting": "inverse"},
    ]

    for spec in configs:
        if spec["label"] in _completed_labels:
            print(f"\n  Skipping: {spec['label']} (already in TSV)")
            continue

        print(f"\n  Running: {spec['label']} ...", end="", flush=True)
        cfg = EngineConfig()
        cfg.max_distance       = spec["max_d"]
        cfg.distance_weighting = spec["weighting"]

        fold_metrics   = run_walkforward_beta(cfg, full_db)
        bss_values     = [m["bss"] for m in fold_metrics]
        mean_bss       = float(np.mean(bss_values))
        positive_folds = sum(1 for b in bss_values if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS

        row = {
            "config_label":       spec["label"],
            "calibrator":         "beta_abm",
            "max_distance":       spec["max_d"],
            "distance_weighting": spec["weighting"],
            "mean_bss":           round(mean_bss, 6),
            "positive_folds":     positive_folds,
            "gate_met":           gate_met,
            "mean_avg_k":         round(float(np.mean([m["avg_k"] for m in fold_metrics])), 2),
        }
        for m in fold_metrics:
            row[f"bss_{m['fold']}"] = m["bss"]
        all_results.append(row)

        # Incremental append
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in _tsv_cols) + "\n")

        print(f" mean_BSS={mean_bss:+.5f} pos_folds={positive_folds}/6"
              f"{'  *** GATE ***' if gate_met else ''}")

    # Final sorted rewrite
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_bss", ascending=False)
    with open(results_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    print(f"\n  TSV written: {results_path}")
    elapsed = (time.time() - start) / 60
    print(f"  Total time: {elapsed:.1f} min")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
