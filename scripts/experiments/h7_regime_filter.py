"""
scripts/experiments/h7_regime_filter.py

Hypothesis H7: Routing Bear-regime val queries to base_rate (HOLD) eliminates the
Reliability gap in 2022-Bear and lifts total positive-fold count to ≥ 3/6.

Context (2026-04-05):
  H5 best: max_d=0.90, beta_abm → 2/6 positive folds. Gate requires 3/6.
  Murphy decomposition on best config (52T VOL_NORM, max_d=0.90, beta_abm):
    Resolution  = 0.007621  (genuine signal exists at 52T)
    Reliability = 0.009544  (dominant — calibration is the bottleneck)
    2022-Bear Reliability gap = 0.007198  (all other folds < 0.003)

Mechanism (HOLD mode):
  Bear-regime val rows receive p = base_rate (y_val.mean()) instead of KNN prob.
  BSS impact for Bear rows: (p_i - y_i)^2 = (y_bar - y_i)^2 = BS_ref contribution
  Net BSS effect of HOLD rows = 0 (neither helps nor hurts vs climatology).
  Only Bull-regime rows contribute to BSS improvement or degradation.

  If 2022-Bear year is mostly Bear-regime: BSS(2022) → ≈ 0 (from -0.029).
  Expected total: 3-4/6 positive folds (2020-COVID, 2021, and 2022/2023 near zero).

Two modes compared:
  1. hold   — Bear val rows → base_rate. KNN still runs on all rows; Bear probs replaced.
  2. filter — regime_filter=True in matcher. Bear queries match only Bear training analogues.
  3. none   — baseline: no regime modification (reproduces H5 best config for reference).

Sweep: spy_threshold — SPY ret_90d < threshold → Bear regime
  Values: [-0.15, -0.10, -0.05, 0.0, 0.05]
  threshold=0.0 matches RegimeLabeler default (negative trailing return = Bear).
  threshold=-0.15 is conservative (only deep bear markets qualify).
  threshold=+0.05 is aggressive (even mild dips = Bear).

Fixed: max_distance=0.90, beta_abm, uniform, VOL_NORM_COLS(8), 52t_volnorm dataset.
Gate: BSS > 0 on ≥ 3/6 folds.
Output: results/bss_fix_sweep_h7.tsv

Regime labels use SPY ret_90d extracted directly from the dataset being labeled
(both train and val periods). SPY is included in the 52T universe. Using in-sample
SPY ret_90d for regime labeling does NOT introduce look-ahead bias for the HOLD mode
because we are using SPY's own observable return to classify SPY's market regime
(not using any target/forward returns). For the FILTER mode, we use the same labeling
to ensure the two modes are directly comparable.
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
    raise RuntimeError("betacal is not installed. Run: py -3.12 -m pip install betacal")

GATE_THRESHOLD_POSITIVE_FOLDS = 3
RESULTS_PATH = project_root / "results" / "bss_fix_sweep_h7.tsv"
_52T_DIR = project_root / "data" / "52t_volnorm"

MAX_DISTANCE = 0.90
SPY_THRESHOLDS = [-0.15, -0.10, -0.05, 0.0, 0.05]
MODES = ["none", "hold", "filter"]


# ---------------------------------------------------------------------------
# Beta calibrator (identical to H5 — consistently best vs Platt in H1-H5)
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
# Regime labeler compatible with matcher interface
# ---------------------------------------------------------------------------

class _ThresholdRegimeLabeler:
    """SPY-based regime labeler with configurable bear threshold.

    Uses SPY ret_90d extracted directly from the database being labeled.
    Requires "SPY" ticker rows to be present in both train and val databases.

    Interface contract matches RegimeLabeler:
        labeler.fitted   bool
        labeler.mode     str
        labeler.label(db, reference_db=None) → np.ndarray[int]
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.fitted: bool = False
        self.mode: str = "binary"
        self._spy_ret90_train: pd.Series | None = None  # Fallback for val dates

    def fit(self, reference_db: pd.DataFrame) -> "_ThresholdRegimeLabeler":
        spy = reference_db[reference_db["Ticker"] == "SPY"].copy()
        if spy.empty:
            raise RuntimeError(
                "_ThresholdRegimeLabeler.fit(): SPY not found in reference_db. "
                "Confirm SPY is in the 52T ticker universe."
            )
        spy["Date"] = pd.to_datetime(spy["Date"])
        spy = spy.set_index("Date").sort_index()
        self._spy_ret90_train = spy["ret_90d"]
        self.fitted = True
        return self

    def label(self, db: pd.DataFrame, reference_db=None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(
                "_ThresholdRegimeLabeler.label() called before fit()."
            )
        # Prefer SPY rows from the db being labeled (accurate for all dates)
        spy_rows = db[db["Ticker"] == "SPY"].copy()
        if not spy_rows.empty:
            spy_rows["Date"] = pd.to_datetime(spy_rows["Date"])
            spy_rows = spy_rows.set_index("Date").sort_index()
            spy_ret90 = spy_rows["ret_90d"]
        else:
            # Fallback: use training SPY series with nearest-fill
            spy_ret90 = self._spy_ret90_train

        spy_regime = spy_ret90.map(lambda r: 0 if r < self.threshold else 1)

        dates = pd.to_datetime(db["Date"])
        mapped = spy_regime.reindex(dates.values, method="nearest")
        labels = pd.Series(mapped.values, index=db.index).fillna(1).astype(int)
        return labels.values.astype(int)


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
# BSS helpers
# ---------------------------------------------------------------------------

def _compute_bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    base_rate = y_true.mean()
    bs_ref = base_rate * (1.0 - base_rate)
    if bs_ref < 1e-10:
        return 0.0
    bs_model = float(np.mean((probs - y_true) ** 2))
    return 1.0 - bs_model / bs_ref


# ---------------------------------------------------------------------------
# Walk-forward runner — three modes
# ---------------------------------------------------------------------------

def _run_walkforward(
    mode: str,
    spy_threshold: float,
    full_db: pd.DataFrame,
) -> list[dict]:
    """Run 6-fold walk-forward for a given mode and regime threshold.

    Args:
        mode: "none" | "hold" | "filter"
        spy_threshold: SPY ret_90d < threshold → Bear (0), else Bull (1).
        full_db: Full combined dataset (train + val years).

    Returns:
        List of per-fold result dicts with keys:
          fold, bss, avg_k, n_val, n_bear, n_bull, pct_bear
    """
    import pattern_engine.matcher as _matcher_module

    horizon = "fwd_7d_up"
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
                "fold": fold["label"], "bss": float("nan"),
                "avg_k": 0.0, "n_val": 0,
                "n_bear": 0, "n_bull": 0, "pct_bear": float("nan"),
            })
            continue

        # Regime labels for this fold's val period
        regime_labeler = _ThresholdRegimeLabeler(threshold=spy_threshold)
        regime_labeler.fit(train_db)

        cfg = EngineConfig()
        cfg.max_distance       = MAX_DISTANCE
        cfg.distance_weighting = "uniform"

        try:
            if mode == "filter":
                # Regime filter: Bear queries → Bear analogues only
                cfg.regime_filter = True
                _matcher_module._PlattCalibrator = _BetaCalibrator
                matcher = PatternMatcher(cfg)
                matcher.fit(train_db, VOL_NORM_COLS, regime_labeler=regime_labeler)
                probs, _, _, n_matches, _, _ = matcher.query(
                    val_db, regime_labeler=regime_labeler, verbose=0
                )
                probs = np.asarray(probs)

                # Count regime labels for diagnostics
                val_labels = regime_labeler.label(val_db)
                n_bear = int((val_labels == 0).sum())
                n_bull = int((val_labels == 1).sum())

            else:
                # none and hold: run matcher without regime filter
                _matcher_module._PlattCalibrator = _BetaCalibrator
                matcher = PatternMatcher(cfg)
                matcher.fit(train_db, VOL_NORM_COLS)
                probs, _, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
                probs = np.asarray(probs)

                val_labels = regime_labeler.label(val_db)
                bear_mask  = (val_labels == 0)
                n_bear = int(bear_mask.sum())
                n_bull = int((~bear_mask).sum())

                if mode == "hold":
                    # Replace Bear row probs with val base_rate (climatological)
                    base_rate = val_db[horizon].values.astype(float).mean()
                    probs_out = probs.copy()
                    probs_out[bear_mask] = base_rate
                    probs = probs_out

        finally:
            _matcher_module._PlattCalibrator = original_calibrator
            if mode == "filter":
                cfg.regime_filter = False  # Reset for safety

        y_true = val_db[horizon].values.astype(float)
        bss_val = _compute_bss(probs, y_true)

        fold_results.append({
            "fold":     fold["label"],
            "bss":      round(bss_val, 6),
            "avg_k":    round(float(np.mean(n_matches)), 2),
            "n_val":    len(val_db),
            "n_bear":   n_bear,
            "n_bull":   n_bull,
            "pct_bear": round(n_bear / len(val_db), 3) if len(val_db) > 0 else float("nan"),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    fold_key = lambda prefix, lbl: f"{prefix}_{lbl}"

    tsv_cols = (
        ["config_label", "mode", "spy_threshold",
         "mean_bss", "positive_folds", "gate_met", "mean_avg_k", "mean_pct_bear"]
        + [fold_key("bss", lbl) for lbl in fold_labels]
        + [fold_key("avgk", lbl) for lbl in fold_labels]
        + [fold_key("n_bear", lbl) for lbl in fold_labels]
        + [fold_key("pct_bear", lbl) for lbl in fold_labels]
    )

    header_lines = [
        "# bss_fix_sweep_h7.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        "# H7: Regime filter / HOLD sweep at 52T VOL_NORM",
        "# Fixed: max_distance=0.90, beta_abm, uniform, VOL_NORM_COLS(8)",
        "# Data: data/52t_volnorm/ (52 tickers including SPY)",
        "# Regime signal: SPY ret_90d < spy_threshold → Bear (0), else Bull (1)",
        "# mode=none: baseline (no regime modification, reproduces H5 best)",
        "# mode=hold: Bear val rows → base_rate prob (HOLD)",
        "# mode=filter: regime_filter=True in matcher (Bear→Bear analogues)",
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
                print(f"\n  [RESUME] Skipping completed configs: {sorted(completed_labels)}")
        except Exception:
            pass

    if not completed_labels:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(tsv_cols) + "\n")

    start = time.time()
    print("\n" + "=" * 75)
    print("  H7: REGIME FILTER / HOLD — VOL_NORM at 52T")
    print("=" * 75)

    print(f"\n  Modes:      {MODES}")
    print(f"  Thresholds: {SPY_THRESHOLDS}")
    print(f"  Fixed:      max_d={MAX_DISTANCE}, beta_abm, uniform")
    print(f"  Gate:       positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}")

    print(f"\n  Loading 52T VOL_NORM data...", end="", flush=True)
    full_db = _load_52t_volnorm()
    print()

    # Build config sweep: (mode, threshold) pairs
    # "none" mode uses threshold=0.0 (irrelevant — no regime logic runs)
    configs: list[tuple[str, float]] = [("none", 0.0)]
    for mode in ["hold", "filter"]:
        for thr in SPY_THRESHOLDS:
            configs.append((mode, thr))

    print(f"\n  Total configs: {len(configs)}, folds: 6, runs: {len(configs) * 6}")
    print(f"\n{'':4}{'Mode':>8} {'Thr':>6} {'AvgK':>6} {'meanBSS':>10} "
          f"{'pos/6':>6} {'pctBear':>8}  Fold BSS values")
    print("  " + "-" * 80)

    for mode, thr in configs:
        thr_str = f"{thr:+.2f}" if mode != "none" else "n/a"
        label = f"{mode}_thr{thr:+.2f}" if mode != "none" else "baseline_none"

        if label in completed_labels:
            print(f"    {mode:>8} {thr_str:>6}  [skip — already done]")
            continue

        t0 = time.time()
        fold_metrics = _run_walkforward(mode, thr, full_db)
        elapsed = time.time() - t0

        bss_values     = [m["bss"] for m in fold_metrics if not np.isnan(m["bss"])]
        mean_bss       = float(np.mean(bss_values)) if bss_values else float("nan")
        positive_folds = sum(1 for b in bss_values if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS
        mean_avg_k     = float(np.mean([m["avg_k"] for m in fold_metrics if m["avg_k"] > 0]))
        pct_bears      = [m["pct_bear"] for m in fold_metrics if not np.isnan(m.get("pct_bear", float("nan")))]
        mean_pct_bear  = float(np.mean(pct_bears)) if pct_bears else float("nan")

        row: dict = {
            "config_label":  label,
            "mode":          mode,
            "spy_threshold": thr if mode != "none" else float("nan"),
            "mean_bss":      round(mean_bss, 6),
            "positive_folds": positive_folds,
            "gate_met":      gate_met,
            "mean_avg_k":    round(mean_avg_k, 2),
            "mean_pct_bear": round(mean_pct_bear, 3) if not np.isnan(mean_pct_bear) else float("nan"),
        }
        for m in fold_metrics:
            lbl = m["fold"]
            row[f"bss_{lbl}"]     = m["bss"]
            row[f"avgk_{lbl}"]    = m["avg_k"]
            row[f"n_bear_{lbl}"]  = m["n_bear"]
            row[f"pct_bear_{lbl}"] = m["pct_bear"]
        all_results.append(row)

        # Incremental append
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in tsv_cols) + "\n")

        bss_strs = "  ".join(
            f"{m['bss']:+.5f}{'*' if m['bss'] > 0 else ' '}"
            for m in fold_metrics
        )
        bear_str = f"{mean_pct_bear:.1%}" if not np.isnan(mean_pct_bear) else "n/a"
        gate_str = "  *** GATE ***" if gate_met else ""
        print(
            f"    {mode:>8} {thr_str:>6}  {mean_avg_k:6.1f}  {mean_bss:+10.5f}  "
            f"{positive_folds:3}/6  {bear_str:>8}  [{bss_strs}]{gate_str}  ({elapsed:.0f}s)"
        )

    # Final sorted rewrite
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_bss", ascending=False)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    # Summary
    print("\n" + "=" * 75)
    print("  SUMMARY — sorted by mean_BSS")
    print("=" * 75)
    print(f"\n  {'Mode':>8}  {'Thr':>6}  {'AvgK':>6}  {'meanBSS':>10}  "
          f"{'pos/6':>5}  {'pctBear':>8}  Gate")
    print("  " + "-" * 65)
    for _, r in df.iterrows():
        gate_str = "YES ***" if r["gate_met"] else "no"
        thr_str  = f"{r['spy_threshold']:+.2f}" if not np.isnan(r.get("spy_threshold", float("nan"))) else "n/a"
        bear_str = f"{r['mean_pct_bear']:.1%}" if not np.isnan(r.get("mean_pct_bear", float("nan"))) else "n/a"
        print(
            f"  {r['mode']:>8}  {thr_str:>6}  {r['mean_avg_k']:6.1f}  "
            f"{r['mean_bss']:+10.5f}  {int(r['positive_folds']):3}/6  "
            f"{bear_str:>8}  {gate_str}"
        )

    gate_winners = df[df["gate_met"] == True]
    print()
    if len(gate_winners) > 0:
        best = gate_winners.iloc[0]
        print(f"  GATE MET — mode={best['mode']}, threshold={best['spy_threshold']:+.2f}")
        print(f"  mean_BSS={best['mean_bss']:+.5f}, {int(best['positive_folds'])}/6 positive folds")
        print(f"  NEXT: lock regime config, proceed to Phase 2 (Half-Kelly at 52T).")
    else:
        best = df.iloc[0]
        print(f"  Gate NOT met. Best: mode={best['mode']}, thr={best.get('spy_threshold', 'n/a')} "
              f"(mean_BSS={best['mean_bss']:+.5f}, {int(best['positive_folds'])}/6)")
        print()

        # Per-fold Bear breakdown for 2022-Bear (key diagnostic)
        bear_col = "pct_bear_2022-Bear"
        if bear_col in df.columns:
            print("  2022-Bear HOLD fraction by config:")
            for _, r in df.iterrows():
                pct = r.get(bear_col, float("nan"))
                bss_val = r.get("bss_2022-Bear", float("nan"))
                if not np.isnan(pct):
                    print(f"    {r['mode']:>8} thr={r.get('spy_threshold', 'n/a'):>6}  "
                          f"pct_bear={pct:.1%}  bss_2022={bss_val:+.5f}")
        print()
        print("  ESCALATE: regime filter did not meet gate at any threshold.")
        print("  Consider: H8 — per-fold calibration, or H9 — restricted-universe sweep.")

    print(f"\n  TSV: {RESULTS_PATH}")
    print(f"  Total time: {(time.time() - start) / 60:.1f} min")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
