"""
scripts/experiments/h6_feature_expansion.py

Hypothesis H6: Cross-sectional feature expansion at 52T VOL_NORM.

Murphy decomposition (2026-04-02) on 52T VOL_NORM (max_d=0.90):
  Resolution  = 0.007621  — 10× higher than 585T (0.000709). Signal EXISTS.
  Reliability = 0.009544  — slightly larger, hence negative BSS.
  Gap needed  = ~0.002 reduction in Reliability to flip BSS positive.

  Per-fold: 2022 Bear is the outlier (Reliability=0.015, gap=0.007).
  COVID fold near-balanced (Resolution≈Reliability → positive BSS).

Hypothesis: the 8D VOL_NORM fingerprint captures absolute momentum shape but not
cross-sectional relative performance. In bear markets, all tickers share similar
absolute return trajectories (spurious KNN similarity), but sector-relative performance
is a genuinely discriminative signal even in downturns.

Test matrix — 6 feature sets, fixed: max_d=0.90, beta_abm, uniform, 52T:
  0. VOL_NORM_8D            (baseline — 8 vol-normalized returns)
  1. +sector_rel_7d         (8D + sector_relative_return_7d → 9D)
  2. +sector_rank_30d       (8D + sector_rank_30d → 9D)
  3. +spy_correlation       (8D + spy_correlation_30d → 9D)
  4. +rel_7d+rank_30d       (8D + 2 sector features → 10D)
  5. +all_3                 (8D + all 3 cross-sectional → 11D)

Gate: BSS > 0 on ≥ 3/6 folds

Data: data/52t_features/ (52T + VOL_NORM_COLS + 3 cross-sectional features)
Output: results/bss_fix_sweep_h6.tsv
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
    raise RuntimeError("betacal not installed. Run: py -3.12 -m pip install betacal")

GATE_THRESHOLD_POSITIVE_FOLDS = 3
RESULTS_PATH = project_root / "results" / "bss_fix_sweep_h6.tsv"
_DATA_DIR = project_root / "data" / "52t_features"

# Cross-sectional features added by build_52t_features.py
SECTOR_REL_7D  = "sector_relative_return_7d"
SECTOR_RANK_30 = "sector_rank_30d"
SPY_CORR_30    = "spy_correlation_30d"

# Fixed config from H5: max_d=0.90 best, beta_abm, uniform
FIXED_MAX_DISTANCE = 0.90


# ---------------------------------------------------------------------------
# Feature sets to test
# ---------------------------------------------------------------------------

FEATURE_SETS = [
    {
        "label": "base_8d",
        "cols":  VOL_NORM_COLS,
        "desc":  "8D VOL_NORM baseline (same as H5 best)",
    },
    {
        "label": "9d_sector_rel_7d",
        "cols":  VOL_NORM_COLS + [SECTOR_REL_7D],
        "desc":  "+sector_relative_return_7d",
    },
    {
        "label": "9d_sector_rank_30d",
        "cols":  VOL_NORM_COLS + [SECTOR_RANK_30],
        "desc":  "+sector_rank_30d",
    },
    {
        "label": "9d_spy_corr_30d",
        "cols":  VOL_NORM_COLS + [SPY_CORR_30],
        "desc":  "+spy_correlation_30d",
    },
    {
        "label": "10d_rel7d_rank30d",
        "cols":  VOL_NORM_COLS + [SECTOR_REL_7D, SECTOR_RANK_30],
        "desc":  "+sector_rel_7d + sector_rank_30d",
    },
    {
        "label": "11d_all_cross_sect",
        "cols":  VOL_NORM_COLS + [SECTOR_REL_7D, SECTOR_RANK_30, SPY_CORR_30],
        "desc":  "+all 3 cross-sectional features",
    },
]


# ---------------------------------------------------------------------------
# Beta calibrator
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

def _load_data() -> pd.DataFrame:
    t = _DATA_DIR / "train_db.parquet"
    v = _DATA_DIR / "val_db.parquet"
    if not (t.exists() and v.exists()):
        raise FileNotFoundError(
            f"Feature-enriched dataset not found at {_DATA_DIR}.\n"
            "Run: py -3.12 scripts/build_52t_features.py"
        )
    combined = pd.concat([pd.read_parquet(t), pd.read_parquet(v)], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    print(f"  Loaded: {len(combined):,} rows, {combined['Ticker'].nunique()} tickers "
          f"from {_DATA_DIR.name}/")
    return combined


# ---------------------------------------------------------------------------
# Walk-forward runner
# ---------------------------------------------------------------------------

def _run_walkforward(feature_cols: list[str], full_db: pd.DataFrame) -> list[dict]:
    """Run 6-fold walk-forward with beta calibration and specified feature columns."""
    import pattern_engine.matcher as _matcher_module

    cfg = EngineConfig()
    cfg.max_distance       = FIXED_MAX_DISTANCE
    cfg.distance_weighting = "uniform"
    horizon = cfg.projection_horizon

    original_calibrator = _matcher_module._PlattCalibrator
    fold_results = []

    for fold in WALKFORWARD_FOLDS:
        train_end = pd.to_datetime(fold["train_end"])
        val_start = pd.to_datetime(fold["val_start"])
        val_end   = pd.to_datetime(fold["val_end"])

        # Only keep rows that have all required feature columns non-NaN
        train_db = full_db[full_db["Date"] <= train_end].dropna(
            subset=[horizon] + feature_cols
        ).copy()
        val_db = full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ].dropna(subset=[horizon] + feature_cols).copy()

        if len(val_db) == 0:
            fold_results.append({
                "fold": fold["label"], "bss": float("nan"), "avg_k": 0.0,
                "n_val": 0, "resolution": float("nan"), "reliability": float("nan"),
            })
            continue

        try:
            _matcher_module._PlattCalibrator = _BetaCalibrator
            matcher = PatternMatcher(cfg)
            matcher.fit(train_db, feature_cols)
            probs, _, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
        finally:
            _matcher_module._PlattCalibrator = original_calibrator

        y_true = val_db[horizon].values.astype(float)
        probs_arr = np.asarray(probs, dtype=np.float64)

        # BSS
        bs_ref   = y_true.mean() * (1 - y_true.mean())
        bs_model = float(np.mean((probs_arr - y_true) ** 2))
        bss_val  = 1.0 - bs_model / bs_ref if bs_ref > 1e-10 else 0.0

        # Murphy decomposition (to track resolution/reliability shifts)
        base_rate = y_true.mean()
        resolution = reliability = 0.0
        for p_val in np.unique(probs_arr):
            mask = probs_arr == p_val
            n_k = mask.sum(); o_k = y_true[mask].mean()
            resolution  += n_k * (o_k - base_rate) ** 2
            reliability += n_k * (p_val - o_k) ** 2
        n = len(y_true)
        resolution /= n; reliability /= n

        fold_results.append({
            "fold":        fold["label"],
            "bss":         round(bss_val, 6),
            "avg_k":       round(float(np.mean(n_matches)), 2),
            "n_val":       len(val_db),
            "resolution":  round(resolution, 6),
            "reliability": round(reliability, 6),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    tsv_cols = (
        ["config_label", "n_features", "mean_bss", "positive_folds", "gate_met",
         "mean_avg_k", "mean_resolution", "mean_reliability"]
        + [f"bss_{lbl}" for lbl in fold_labels]
        + [f"res_{lbl}" for lbl in fold_labels]
        + [f"rel_{lbl}" for lbl in fold_labels]
    )
    header_lines = [
        "# bss_fix_sweep_h6.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        "# H6: cross-sectional feature expansion at 52T VOL_NORM",
        f"# Fixed: max_d={FIXED_MAX_DISTANCE}, beta_abm, uniform, data=52t_features/",
        "# Murphy baseline: Resolution=0.007621, Reliability=0.009544",
        f"# Gate: positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        "# Rows appended incrementally.",
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
    print("\n" + "=" * 72)
    print("  H6: CROSS-SECTIONAL FEATURE EXPANSION — 52T VOL_NORM")
    print("=" * 72)
    print(f"\n  Murphy baseline: Resolution=0.0076, Reliability=0.0095")
    print(f"  Need: Reliability gap -0.002 to flip BSS positive")
    print(f"  Fixed: max_d={FIXED_MAX_DISTANCE}, beta_abm, uniform")

    print(f"\n  Loading data...", end="", flush=True)
    full_db = _load_data()
    print()

    for spec in FEATURE_SETS:
        label = spec["label"]
        if label in completed_labels:
            print(f"\n  Skipping: {label} (already done)")
            continue

        n_feat = len(spec["cols"])
        print(f"\n  [{label}] {n_feat}D — {spec['desc']}")
        print(f"  Features: {spec['cols']}")

        t0 = time.time()
        fold_metrics   = _run_walkforward(spec["cols"], full_db)
        elapsed        = time.time() - t0

        bss_vals    = [m["bss"]         for m in fold_metrics if not np.isnan(m["bss"])]
        res_vals    = [m["resolution"]  for m in fold_metrics if not np.isnan(m.get("resolution", float("nan")))]
        rel_vals    = [m["reliability"] for m in fold_metrics if not np.isnan(m.get("reliability", float("nan")))]

        mean_bss       = float(np.mean(bss_vals))       if bss_vals else float("nan")
        mean_res       = float(np.mean(res_vals))       if res_vals else float("nan")
        mean_rel       = float(np.mean(rel_vals))       if rel_vals else float("nan")
        positive_folds = sum(1 for b in bss_vals if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS
        mean_avg_k     = float(np.mean([m["avg_k"] for m in fold_metrics if m["avg_k"] > 0]))

        row: dict = {
            "config_label":    label,
            "n_features":      n_feat,
            "mean_bss":        round(mean_bss, 6),
            "positive_folds":  positive_folds,
            "gate_met":        gate_met,
            "mean_avg_k":      round(mean_avg_k, 2),
            "mean_resolution": round(mean_res, 6),
            "mean_reliability":round(mean_rel, 6),
        }
        for m in fold_metrics:
            row[f"bss_{m['fold']}"] = m["bss"]
            row[f"res_{m['fold']}"] = m.get("resolution", float("nan"))
            row[f"rel_{m['fold']}"] = m.get("reliability", float("nan"))
        all_results.append(row)

        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in tsv_cols) + "\n")

        gate_str = "  *** GATE ***" if gate_met else ""
        print(f"  → mean_BSS={mean_bss:+.5f}  pos_folds={positive_folds}/6  "
              f"Res={mean_res:.5f}  Rel={mean_rel:.5f}  ({elapsed:.0f}s){gate_str}")

        print(f"  {'Fold':<14} {'BSS':>9} {'AvgK':>5} {'Resolution':>11} {'Reliability':>12}")
        for m in fold_metrics:
            marker = " *" if m["bss"] > 0 else ""
            print(f"  {m['fold']:<14} {m['bss']:>+9.5f} {m['avg_k']:>5.1f} "
                  f"{m.get('resolution', float('nan')):>11.6f} "
                  f"{m.get('reliability', float('nan')):>12.6f}{marker}")

    # Final sorted rewrite
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_bss", ascending=False)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY — sorted by mean_BSS")
    print("=" * 72)
    print(f"\n  {'Label':<22} {'nD':>3}  {'meanBSS':>10}  {'pos/6':>5}  "
          f"{'Res':>8}  {'Rel':>8}  Gate")
    print("  " + "-" * 70)
    for _, r in df.iterrows():
        gate_str = "YES ***" if r["gate_met"] else "no"
        print(f"  {r['config_label']:<22} {int(r['n_features']):3}  "
              f"{r['mean_bss']:+10.5f}  {int(r['positive_folds']):3}/6  "
              f"{r['mean_resolution']:8.5f}  {r['mean_reliability']:8.5f}  {gate_str}")

    gate_winners = df[df["gate_met"] == True]
    if len(gate_winners) > 0:
        best = gate_winners.iloc[0]
        print(f"\n  GATE MET: {best['config_label']} ({int(best['n_features'])}D) — "
              f"mean_BSS={best['mean_bss']:+.5f}, {int(best['positive_folds'])}/6 folds")
        print("  NEXT: Lock feature set, proceed to Phase 2 (Half-Kelly) at 52T.")
    else:
        best = df.iloc[0]
        print(f"\n  Gate NOT met. Best: {best['config_label']} "
              f"(mean_BSS={best['mean_bss']:+.5f}, {int(best['positive_folds'])}/6)")
        print("  Resolution exists (0.0076) but Reliability gap persists.")
        print("  Consider: regime-aware calibration (Option D) or deeper feature engineering.")

    print(f"\n  TSV: {RESULTS_PATH}")
    print(f"  Total time: {(time.time() - start) / 60:.1f} min")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
