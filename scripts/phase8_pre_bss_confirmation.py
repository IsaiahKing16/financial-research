"""
scripts/phase8_pre_bss_confirmation.py — P8-PRE-7 BSS confirmation.

Runs a clean 6-fold walk-forward with max_distance=2.5 (the P8-PRE-7 sweep winner)
and compares per-fold BSS against the Phase 6 historical baseline stored in
results/phase6/bss_comparison_candle_vs_baseline.tsv.

Gate: new BSS within ±0.001 of Phase 6 baseline on ≥ 4/6 folds.
  PASS → max_distance=2.5 is confirmed. Lock it.
  FAIL → report per-fold deltas, escalate.

Rationale for using Phase 6 as baseline: ADR-007 found that the old geometry
(pre-P8-PRE-4) cannot be reproduced separately because standardization was
ALWAYS active. The Phase 6 results were produced with the same geometry as the
current codebase. The ±0.001 tolerance accounts for any minor code changes
between Phase 6 and P8-PRE-7.

Full clean fit+query per fold (not the sweep shortcut). Calibrator is fitted
on each fold's training data at exactly max_distance=2.5. This matches the
Phase 6 methodology exactly.

Locked settings applied (CLAUDE.md):
    feature_set=returns_candle(23), calibration=beta_abm, max_distance=2.5,
    top_k=50, regime=hold_spy_threshold+0.05 (H7 HOLD mode), nn_jobs=1

Output:
    results/phase8_pre/bss_confirmation_standardized.tsv

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase8_pre_bss_confirmation.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pattern_engine.matcher as _matcher_module
from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import FeatureRegistry
from pattern_engine.candlestick import compute_candlestick_features, CANDLE_COLS

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError("betacal is not installed. Run: py -3.12 -m pip install betacal")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = REPO_ROOT / "data" / "52t_features"
RESULTS_DIR = REPO_ROOT / "results" / "phase8_pre"
BASELINE_PATH = REPO_ROOT / "results" / "phase6" / "bss_comparison_candle_vs_baseline.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Locked settings ───────────────────────────────────────────────────────────
RETURNS_CANDLE_COLS: list[str] = FeatureRegistry.get("returns_candle").columns
MAX_DISTANCE    = 2.5   # P8-PRE-7 winner
SPY_THRESHOLD   = 0.05  # H7 locked
HORIZON         = "fwd_7d_up"
BSS_TOLERANCE   = 0.001  # ±0.001 from Phase 6 baseline
BSS_MIN_FOLDS   = 4      # pass if ≥4/6 folds within tolerance


# ── Beta calibrator ───────────────────────────────────────────────────────────

class _BetaCalibrator:
    """Drop-in replacement for _PlattCalibrator using betacal BetaCalibration(abm)."""

    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ── Regime labeler ────────────────────────────────────────────────────────────

class _ThresholdRegimeLabeler:
    """SPY-based binary regime labeler: bear (0) = SPY ret_90d < threshold."""

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.fitted: bool = False
        self.mode: str = "binary"
        self._spy_ret90_train: pd.Series | None = None

    def fit(self, reference_db: pd.DataFrame) -> "_ThresholdRegimeLabeler":
        spy = reference_db[reference_db["Ticker"] == "SPY"].copy()
        if spy.empty:
            raise RuntimeError(
                "_ThresholdRegimeLabeler.fit(): SPY not found in reference_db."
            )
        spy["Date"] = pd.to_datetime(spy["Date"])
        spy = spy.set_index("Date").sort_index()
        self._spy_ret90_train = spy["ret_90d"]
        self.fitted = True
        return self

    def label(self, db: pd.DataFrame, reference_db=None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("label() called before fit()")
        spy_rows = db[db["Ticker"] == "SPY"].copy()
        if not spy_rows.empty:
            spy_rows["Date"] = pd.to_datetime(spy_rows["Date"])
            spy_rows = spy_rows.set_index("Date").sort_index()
            spy_ret90 = spy_rows["ret_90d"]
        else:
            spy_ret90 = self._spy_ret90_train
        spy_regime = spy_ret90.map(lambda r: 0 if r < self.threshold else 1)
        dates = pd.to_datetime(db["Date"])
        mapped = spy_regime.reindex(dates.values, method="nearest")
        labels = pd.Series(mapped.values, index=db.index).fillna(1).astype(int)
        return labels.values.astype(int)


# ── BSS ───────────────────────────────────────────────────────────────────────

def _bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    base_rate = y_true.mean()
    bs_ref = base_rate * (1.0 - base_rate)
    if bs_ref < 1e-10:
        return 0.0
    return 1.0 - float(np.mean((probs - y_true) ** 2)) / bs_ref


# ── Candlestick augmentation ──────────────────────────────────────────────────

def _impute_candle_nans(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prop_fill = {c: 0.0 for c in CANDLE_COLS if "direction" not in c}
    dir_fill  = {c: 1.0 for c in CANDLE_COLS if "direction" in c}
    df.fillna({**prop_fill, **dir_fill}, inplace=True)
    return df


def _augment_with_candlestick(full_db: pd.DataFrame) -> pd.DataFrame:
    print("Computing candlestick features...", flush=True)
    t0 = time.time()
    candle_df = compute_candlestick_features(full_db)
    print(f"  Computed {len(CANDLE_COLS)} columns in {time.time() - t0:.1f}s")
    augmented = pd.concat(
        [full_db.reset_index(drop=True), candle_df.reset_index(drop=True)], axis=1
    )
    augmented = _impute_candle_nans(augmented)
    remaining = int(augmented[CANDLE_COLS].isna().values.sum())
    if remaining > 0:
        raise RuntimeError(
            f"Imputation failed: {remaining} NaNs remain in candlestick cols."
        )
    return augmented


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    print("=" * 72)
    print("  P8-PRE-7 BSS Confirmation — max_distance=2.5 vs Phase 6 baseline")
    print("=" * 72)
    print(f"  max_distance : {MAX_DISTANCE} (P8-PRE-7 winner)")
    print(f"  Baseline     : results/phase6/bss_comparison_candle_vs_baseline.tsv")
    print(f"  Gate         : new BSS within ±{BSS_TOLERANCE} of baseline on ≥{BSS_MIN_FOLDS}/6 folds")
    print(f"  Calibration  : beta_abm (H5 locked)")
    print(f"  nn_jobs      : 1 (Windows/Py3.12 deadlock prevention)")
    print()

    # ── Load Phase 6 baseline ─────────────────────────────────────────────────
    if not BASELINE_PATH.exists():
        raise RuntimeError(
            f"Phase 6 baseline not found: {BASELINE_PATH}\n"
            "Cannot run BSS confirmation without historical baseline."
        )
    baseline_df = pd.read_csv(BASELINE_PATH, sep="\t")
    baseline_candle = (
        baseline_df[baseline_df["config"] == "returns_candle"]
        .sort_values("fold")
        .reset_index(drop=True)
    )
    if len(baseline_candle) != 6:
        raise RuntimeError(
            f"Expected 6 returns_candle baseline rows, got {len(baseline_candle)}."
        )
    print("Phase 6 baseline (returns_candle, max_distance=2.5):")
    for _, row in baseline_candle.iterrows():
        print(f"  fold {int(row['fold'])} ({row['fold_label']}): BSS={row['BSS']:+.6f}, AvgK={row['AvgK']:.1f}")
    print()

    # ── Load + augment data ───────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"
    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        sys.exit(1)

    print(f"Loading 52T features data from {DATA_DIR}...")
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers")
    full_db = _augment_with_candlestick(full_db)
    print()

    # ── Clean 6-fold walk-forward at max_distance=2.5 ─────────────────────────
    original_calibrator = _matcher_module._PlattCalibrator
    _matcher_module._PlattCalibrator = _BetaCalibrator

    rows: list[dict] = []

    try:
        for fi, fold in enumerate(WALKFORWARD_FOLDS):
            train_end = pd.Timestamp(fold["train_end"])
            val_start = pd.Timestamp(fold["val_start"])
            val_end   = pd.Timestamp(fold["val_end"])

            train_db = (
                full_db[full_db["Date"] <= train_end]
                .dropna(subset=[HORIZON])
                .copy()
            )
            val_db = (
                full_db[
                    (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
                ]
                .dropna(subset=[HORIZON])
                .copy()
            )

            regime_labeler = _ThresholdRegimeLabeler(threshold=SPY_THRESHOLD)
            regime_labeler.fit(train_db)
            val_regime = regime_labeler.label(val_db)
            bear_mask  = (val_regime == 0)
            y_true     = val_db[HORIZON].values.astype(float)
            base_rate  = float(y_true.mean())

            cfg = EngineConfig()
            cfg.max_distance         = MAX_DISTANCE
            cfg.top_k                = 50
            cfg.distance_weighting   = "uniform"
            cfg.confidence_threshold = 0.65
            cfg.agreement_spread     = 0.05
            cfg.min_matches          = 5
            cfg.exclude_same_ticker  = True
            cfg.same_sector_only     = False
            cfg.regime_filter        = False
            cfg.regime_fallback      = False
            cfg.projection_horizon   = HORIZON
            cfg.cal_max_samples      = 100_000
            cfg.use_hnsw             = True
            cfg.nn_jobs              = 1
            cfg.feature_set          = "returns_candle"
            cfg.use_sax_filter       = False
            cfg.use_wfa_rerank       = False
            cfg.use_ib_compression   = False
            cfg.use_sector_conviction = False
            cfg.use_momentum_filter  = False
            cfg.use_sentiment_veto   = False

            print(f"[{fi+1}/6] fold={fold['label']}  "
                  f"train={len(train_db):,}  val={len(val_db):,}", flush=True)
            t0 = time.time()
            matcher = PatternMatcher(cfg)
            matcher.fit(train_db, RETURNS_CANDLE_COLS)

            probs_raw, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
            elapsed = time.time() - t0

            probs  = np.asarray(probs_raw)
            avg_k  = float(np.mean(n_matches))

            probs_hold = probs.copy()
            probs_hold[bear_mask] = base_rate
            bss_val = _bss(probs_hold, y_true)

            baseline_bss = float(baseline_candle.iloc[fi]["BSS"])
            delta = bss_val - baseline_bss
            within = abs(delta) <= BSS_TOLERANCE

            print(f"  BSS={bss_val:+.6f}  baseline={baseline_bss:+.6f}  "
                  f"delta={delta:+.6f}  AvgK={avg_k:.1f}  "
                  f"{'WITHIN' if within else 'OUTSIDE'} ±{BSS_TOLERANCE}  "
                  f"({elapsed:.0f}s)")

            rows.append({
                "config":       "returns_candle_p8pre7",
                "fold":         fi,
                "fold_label":   fold["label"],
                "BSS_new":      round(bss_val, 6),
                "BSS_baseline": round(baseline_bss, 6),
                "delta":        round(delta, 6),
                "within_gate":  within,
                "AvgK":         round(avg_k, 3),
                "runtime_s":    round(elapsed, 2),
            })

    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    # ── Write TSV ──────────────────────────────────────────────────────────────
    tsv_path = RESULTS_DIR / "bss_confirmation_standardized.tsv"
    df_out = pd.DataFrame(rows)
    df_out.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nSaved: {tsv_path}")

    # ── Gate evaluation ────────────────────────────────────────────────────────
    folds_within = int(df_out["within_gate"].sum())
    mean_delta   = float(df_out["delta"].mean())

    print()
    print("=" * 72)
    print("  BSS CONFIRMATION SUMMARY")
    print("=" * 72)
    print(f"  Folds within ±{BSS_TOLERANCE}: {folds_within}/6  (need ≥{BSS_MIN_FOLDS})")
    print(f"  Mean BSS delta: {mean_delta:+.6f}")
    print()

    for _, row in df_out.iterrows():
        marker = "✓" if row["within_gate"] else "✗"
        print(f"  {marker} fold {int(row['fold'])} ({row['fold_label']}): "
              f"new={row['BSS_new']:+.6f}  base={row['BSS_baseline']:+.6f}  "
              f"delta={row['delta']:+.6f}")

    print()
    if folds_within >= BSS_MIN_FOLDS:
        print(f"  ★ PASS — {folds_within}/6 folds within ±{BSS_TOLERANCE}.")
        print(f"  max_distance=2.5 confirmed. Proceed to lock.")
    else:
        print(f"  ✗ FAIL — only {folds_within}/6 folds within ±{BSS_TOLERANCE}.")
        print(f"  Investigate per-fold deltas. Escalate before locking.")

    print(f"\nTotal runtime: {time.time() - t_total:.0f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()
