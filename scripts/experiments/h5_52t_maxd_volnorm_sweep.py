"""
scripts/experiments/h5_52t_maxd_volnorm_sweep.py

Hypothesis H5: max_distance recalibration for VOL_NORM_COLS at 52T universe.

Context (2026-04-02 session):
  Phase 1 H1 sweep found max_d=0.5 marginally best on 585T with RETURNS_ONLY_COLS.
  At 52T with VOL_NORM_COLS, max_d=0.5 is catastrophically wrong: AvgK drops to 1.2.
  The threshold is not portable across feature transformations — it must be recalibrated.

Distance profile (Fold1, 52T VOL_NORM_COLS, 110K train rows):
  max_d=0.60 → AvgK≈4.9   (too few neighbors)
  max_d=0.70 → AvgK≈14.0  (borderline)
  max_d=0.80 → AvgK≈25.4
  max_d=0.90 → AvgK≈34.5
  max_d=1.00 → AvgK≈40.4
  max_d=1.10 → AvgK≈44.2  (current locked value ≈ 1.1019)
  max_d=1.20 → AvgK≈46.4
  max_d=1.50 → AvgK≈48.9  (near cap)
  max_d=2.00 → AvgK≈49.8  (effectively uncapped)

Sweep grid: max_distance = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.1019, 1.20, 1.50]
Calibration: beta_abm only (consistently better than Platt in all prior experiments)
Weighting: uniform (inverse consistently worse)

Gate: BSS > 0 on ≥ 3/6 folds

Data: data/52t_volnorm/ (52T, VOL_NORM_COLS, M9 feature set)
Output: results/bss_fix_sweep_h5.tsv
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
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import VOL_NORM_COLS

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError(
        "betacal is not installed. Run: py -3.12 -m pip install betacal"
    )

GATE_THRESHOLD_POSITIVE_FOLDS = 3
RESULTS_PATH = project_root / "results" / "bss_fix_sweep_h5.tsv"
_52T_DIR = project_root / "data" / "52t_volnorm"

MAX_DISTANCE_VALUES = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.1019, 1.20, 1.50]


# ---------------------------------------------------------------------------
# Beta calibrator (identical interface to _PlattCalibrator)
# ---------------------------------------------------------------------------

class _BetaCalibrator:
    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def _load_52t_volnorm() -> pd.DataFrame:
    t = _52T_DIR / "train_db.parquet"
    v = _52T_DIR / "val_db.parquet"
    if not (t.exists() and v.exists()):
        raise FileNotFoundError(
            f"52T VOL_NORM dataset not found at {_52T_DIR}.\n"
            "Run: py -3.12 scripts/build_52t_volnorm.py"
        )
    combined = pd.concat([pd.read_parquet(t), pd.read_parquet(v)], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    print(f"  Loaded: {len(combined):,} rows, {combined['Ticker'].nunique()} tickers "
          f"from {_52T_DIR.name}/")
    return combined


# ---------------------------------------------------------------------------
# Walk-forward runner with beta calibration injection
# ---------------------------------------------------------------------------

def _run_walkforward(config: EngineConfig, full_db: pd.DataFrame) -> list[dict]:
    """Run 6-fold walk-forward with beta calibration patched in."""
    import pattern_engine.matcher as _matcher_module

    horizon = config.projection_horizon
    fold_results = []
    original_calibrator = _matcher_module._PlattCalibrator

    for fold in WALKFORWARD_FOLDS:
        train_end = pd.to_datetime(fold["train_end"])
        val_start = pd.to_datetime(fold["val_start"])
        val_end   = pd.to_datetime(fold["val_end"])

        train_db = full_db[full_db["Date"] <= train_end].dropna(subset=[horizon]).copy()
        val_db   = full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ].dropna(subset=[horizon]).copy()

        if len(val_db) == 0:
            fold_results.append({
                "fold": fold["label"], "bss": float("nan"), "avg_k": 0.0, "n_val": 0,
            })
            continue

        try:
            _matcher_module._PlattCalibrator = _BetaCalibrator
            matcher = PatternMatcher(config)
            matcher.fit(train_db, VOL_NORM_COLS)
            probs, _, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
        finally:
            _matcher_module._PlattCalibrator = original_calibrator

        y_true  = val_db[horizon].values.astype(float)
        bs_ref  = y_true.mean() * (1 - y_true.mean())
        bs_model = float(np.mean((np.asarray(probs) - y_true) ** 2))
        bss_val = 1.0 - bs_model / bs_ref if bs_ref > 1e-10 else 0.0

        fold_results.append({
            "fold":  fold["label"],
            "bss":   round(bss_val, 6),
            "avg_k": round(float(np.mean(n_matches)), 2),
            "n_val": len(val_db),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    tsv_cols = (
        ["config_label", "max_distance", "mean_bss", "positive_folds",
         "gate_met", "mean_avg_k"]
        + [f"bss_{lbl}" for lbl in fold_labels]
        + [f"avgk_{lbl}" for lbl in fold_labels]
    )
    header_lines = [
        "# bss_fix_sweep_h5.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        "# H5: max_distance recalibration for VOL_NORM_COLS at 52T",
        "# Calibrator: beta_abm (consistently best in H1-H4 + 52T validation)",
        "# Weighting: uniform (inverse consistently worse)",
        "# Data: data/52t_volnorm/ (M9 VOL_NORM feature set, 52 tickers)",
        f"# Gate: positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        "# Rows appended incrementally — file valid if run is interrupted.",
        "#",
    ]

    # Resume logic
    all_results: list[dict] = []
    completed_labels: set[str] = set()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        try:
            existing = pd.read_csv(RESULTS_PATH, comment="#", sep="\t")
            if "config_label" in existing.columns and len(existing) > 0:
                completed_labels = set(existing["config_label"].tolist())
                for _, row in existing.iterrows():
                    all_results.append(row.to_dict())
                print(f"\n  [RESUME] Skipping completed: {sorted(completed_labels)}")
        except Exception:
            pass

    if not completed_labels:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(tsv_cols) + "\n")

    start = time.time()
    print("\n" + "=" * 70)
    print("  H5: MAX_DISTANCE RECALIBRATION — VOL_NORM at 52T")
    print("=" * 70)
    print(f"\n  {len(MAX_DISTANCE_VALUES)} values: {MAX_DISTANCE_VALUES}")
    print(f"  Calibrator: beta_abm | Weighting: uniform | Data: 52t_volnorm")

    print(f"\n  Loading 52T VOL_NORM data...", end="", flush=True)
    full_db = _load_52t_volnorm()
    print()

    print(f"\n{'':4}{'maxD':>8} {'AvgK':>6} {'meanBSS':>10} {'pos/6':>6}  Fold BSS values")
    print("  " + "-" * 68)

    for max_d in MAX_DISTANCE_VALUES:
        label = f"maxd_{max_d:.4f}"
        if label in completed_labels:
            print(f"    {max_d:8.4f}  [skip — already done]")
            continue

        cfg = EngineConfig()
        cfg.max_distance       = max_d
        cfg.distance_weighting = "uniform"

        t0 = time.time()
        fold_metrics   = _run_walkforward(cfg, full_db)
        elapsed        = time.time() - t0

        bss_values     = [m["bss"] for m in fold_metrics if not np.isnan(m["bss"])]
        mean_bss       = float(np.mean(bss_values)) if bss_values else float("nan")
        positive_folds = sum(1 for b in bss_values if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS
        mean_avg_k     = float(np.mean([m["avg_k"] for m in fold_metrics if m["avg_k"] > 0]))

        row: dict = {
            "config_label":   label,
            "max_distance":   max_d,
            "mean_bss":       round(mean_bss, 6),
            "positive_folds": positive_folds,
            "gate_met":       gate_met,
            "mean_avg_k":     round(mean_avg_k, 2),
        }
        for m in fold_metrics:
            row[f"bss_{m['fold']}"]  = m["bss"]
            row[f"avgk_{m['fold']}"] = m["avg_k"]
        all_results.append(row)

        # Incremental append
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in tsv_cols) + "\n")

        bss_strs = "  ".join(
            f"{m['bss']:+.5f}{'*' if m['bss'] > 0 else ' '}"
            for m in fold_metrics
        )
        gate_str = "  *** GATE ***" if gate_met else ""
        print(f"    {max_d:8.4f}  {mean_avg_k:6.1f}  {mean_bss:+10.5f}  {positive_folds:3}/6"
              f"  [{bss_strs}]{gate_str}  ({elapsed:.0f}s)")

    # Final sorted rewrite
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_bss", ascending=False)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY — sorted by mean_BSS")
    print("=" * 70)
    print(f"\n  {'maxD':>8}  {'AvgK':>6}  {'meanBSS':>10}  {'pos/6':>5}  Gate")
    print("  " + "-" * 50)
    for _, r in df.iterrows():
        gate_str = "YES ***" if r["gate_met"] else "no"
        print(f"  {r['max_distance']:8.4f}  {r['mean_avg_k']:6.1f}  "
              f"{r['mean_bss']:+10.5f}  {int(r['positive_folds']):3}/6  {gate_str}")

    gate_winners = df[df["gate_met"] == True]
    if len(gate_winners) > 0:
        best = gate_winners.iloc[0]
        print(f"\n  GATE MET at max_distance={best['max_distance']:.4f} — "
              f"mean_BSS={best['mean_bss']:+.5f}, {int(best['positive_folds'])}/6 positive folds")
        print("  NEXT: Lock max_distance, proceed to Phase 2 (Half-Kelly) at 52T.")
    else:
        best = df.iloc[0]
        print(f"\n  Gate NOT met. Best: max_d={best['max_distance']:.4f} "
              f"(mean_BSS={best['mean_bss']:+.5f}, {int(best['positive_folds'])}/6)")
        print("  ESCALATE: max_distance recalibration insufficient.")
        print("  Next candidates: Option C (feature expansion) or Murphy decomposition at 52T VOL_NORM.")

    print(f"\n  TSV: {RESULTS_PATH}")
    print(f"  Total time: {(time.time() - start) / 60:.1f} min")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
