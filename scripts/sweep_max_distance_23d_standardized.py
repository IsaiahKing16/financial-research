"""
scripts/sweep_max_distance_23d_standardized.py — P8-PRE-7: max_distance re-sweep.

Re-calibrates max_distance after ADR-007 confirmed StandardScaler is applied to
ALL 23 returns_candle features. Previous sweep (Task 6.1) found max_distance=2.5
on the pre-standardization geometry. This sweep re-validates (or updates) that
value on the confirmed post-standardization geometry.

ADR-007 finding: standardization was ALREADY active when 2.5 was locked. This
sweep formally documents re-validation so the provenance record is unambiguous.

Sweep range: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
  - Drop sub-1.0 values (expected AvgK≈0 in standardized 23D space).
  - Extend to 5.0 for upward coverage.
  - Retain 2.0-3.0 overlap with Task 6.1 for comparison.

Strategy: fit matcher ONCE per fold at max_distance=5.0 (the largest sweep value),
then mutate cfg.max_distance for each sweep value. This produces 6 fits instead
of 48. BSS at non-fitted thresholds carries slight calibration error — acceptable
because the primary gate criterion is AvgK, not BSS.

Locked settings applied (CLAUDE.md):
    feature_set=returns_candle(23), calibration=beta_abm, top_k=50,
    regime=hold_spy_threshold+0.05 (H7 HOLD mode), nn_jobs=1

Provenance: HANDOFF_P8-PRE-7_max-distance-resweep.md

Usage:
    PYTHONUTF8=1 py -3.12 scripts/sweep_max_distance_23d_standardized.py

Output:
    results/phase8_pre/sweep_max_distance_23d_standardized.tsv
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature set (23 cols = 8 VOL_NORM + 15 candlestick) ──────────────────────
RETURNS_CANDLE_COLS: list[str] = FeatureRegistry.get("returns_candle").columns

# ── P8-PRE-7 sweep range ──────────────────────────────────────────────────────
# Rationale: drop sub-1.0 (certain AvgK≈0 in standardized 23D space).
# Extend to 5.0 for upward coverage. Retain 2.0-3.0 overlap with Task 6.1.
# Largest value (5.0) is used as the fit threshold; others sweep by mutation.
MAX_DISTANCE_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
FIT_DISTANCE = 5.0  # fit at largest value, then sweep by mutation

# ── Locked settings (CLAUDE.md) ───────────────────────────────────────────────
SPY_THRESHOLD = 0.05    # H7: bear = SPY ret_90d < +0.05
HORIZON       = "fwd_7d_up"
AVGK_GATE     = 20      # SUCCESS: all 6 folds must exceed this


# ── Beta calibrator (beta_abm — locked since H5) ──────────────────────────────

class _BetaCalibrator:
    """Drop-in replacement for _PlattCalibrator using betacal BetaCalibration(abm).

    Injected via monkey-patch before each fold fit; restored in finally block.
    Same mechanism used in scripts/experiments/h7_regime_filter.py and Task 6.1.
    """

    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ── Regime labeler — H7 HOLD mode ─────────────────────────────────────────────

class _ThresholdRegimeLabeler:
    """SPY-based binary regime labeler: bear (0) = SPY ret_90d < threshold.

    Interface contract matches RegimeLabeler: fitted / mode / label().
    Copied from Task 6.1 template (authoritative for H7 regime logic).
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold
        self.fitted: bool = False
        self.mode: str = "binary"
        self._spy_ret90_train: pd.Series | None = None

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
    """Impute NaN in candlestick cols. Proportions → 0.0; direction → 1.0."""
    df = df.copy()
    prop_fill = {c: 0.0 for c in CANDLE_COLS if "direction" not in c}
    dir_fill  = {c: 1.0 for c in CANDLE_COLS if "direction" in c}
    df.fillna({**prop_fill, **dir_fill}, inplace=True)
    return df


def _augment_with_candlestick(full_db: pd.DataFrame) -> pd.DataFrame:
    """Compute, audit, and impute candlestick features; return augmented DataFrame."""
    print("Computing candlestick features...", flush=True)
    t0 = time.time()
    candle_df = compute_candlestick_features(full_db)
    print(f"  Computed {len(CANDLE_COLS)} columns in {time.time() - t0:.1f}s")

    nan_rates = {c: float(candle_df[c].isna().mean()) for c in CANDLE_COLS}
    max_col   = max(nan_rates, key=nan_rates.get)
    print(f"  NaN pre-imputation: max={nan_rates[max_col]:.2%} [{max_col}]")
    if nan_rates[max_col] > 0.05:
        print(f"  [WARNING] NaN rate > 5% — investigate data gaps before BSS experiment.")

    augmented = pd.concat(
        [full_db.reset_index(drop=True), candle_df.reset_index(drop=True)], axis=1
    )
    augmented = _impute_candle_nans(augmented)
    remaining = int(augmented[CANDLE_COLS].isna().values.sum())
    if remaining > 0:
        raise RuntimeError(
            f"Imputation failed: {remaining} NaNs remain in candlestick cols after fillna."
        )
    print(f"  Imputation complete. 0 NaNs remaining.")
    return augmented


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    print("=" * 72)
    print("  P8-PRE-7 — max_distance re-sweep, returns_candle (23D), 52T universe")
    print("  Re-validates max_distance after ADR-007 standardization confirmation")
    print("=" * 72)
    print(f"  Feature cols : {len(RETURNS_CANDLE_COLS)} ({RETURNS_CANDLE_COLS[:3]}... etc.)")
    print(f"  Sweep values : {MAX_DISTANCE_VALUES}")
    print(f"  Fit at       : max_distance={FIT_DISTANCE} (largest sweep value)")
    print(f"  Gate         : AvgK ≥ {AVGK_GATE} on ALL 6 folds")
    print(f"  Calibration  : beta_abm (H5 locked)")
    print(f"  Regime       : HOLD mode, SPY threshold={SPY_THRESHOLD:+.2f} (H7 locked)")
    print(f"  nn_jobs      : 1 (Windows/Py3.12 deadlock prevention)")
    print()

    # ── Load + augment data ────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"
    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        print("       Run scripts/build_52t_features.py first.")
        sys.exit(1)

    print(f"Loading 52T features data from {DATA_DIR}...")
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"{full_db['Date'].min().date()} → {full_db['Date'].max().date()}")

    missing_ohlc = [c for c in ("Ticker", "Open", "High", "Low", "Close")
                    if c not in full_db.columns]
    if missing_ohlc:
        print(f"ERROR: Missing OHLC columns: {missing_ohlc}")
        sys.exit(1)

    full_db = _augment_with_candlestick(full_db)
    print()

    # ── Walk-forward sweep ────────────────────────────────────────────────────
    # Strategy: fit matcher ONCE per fold at FIT_DISTANCE=5.0, then mutate the
    # threshold for each sweep value. This avoids 8× redundant fits.
    # Calibrator is fitted at 5.0; BSS at other thresholds carries slight
    # miscalibration — acceptable because the gate criterion is AvgK, not BSS.
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

            if len(val_db) == 0:
                for d in MAX_DISTANCE_VALUES:
                    rows.append({
                        "max_distance": d, "fold": fi,
                        "fold_label": fold["label"],
                        "AvgK": 0.0, "BSS": float("nan"),
                        "n_scored": 0, "n_buy": 0, "n_sell": 0, "n_hold": 0,
                        "runtime_s": 0.0,
                    })
                continue

            # Regime HOLD — compute bear_mask for this fold's val period
            regime_labeler = _ThresholdRegimeLabeler(threshold=SPY_THRESHOLD)
            regime_labeler.fit(train_db)
            val_regime = regime_labeler.label(val_db)
            bear_mask  = (val_regime == 0)
            y_true     = val_db[HORIZON].values.astype(float)
            base_rate  = float(y_true.mean())

            # Fit at FIT_DISTANCE (calibrator sees maximum analogue pool)
            cfg = EngineConfig()
            cfg.max_distance        = FIT_DISTANCE
            cfg.top_k               = 50
            cfg.distance_weighting  = "uniform"
            cfg.confidence_threshold = 0.65
            cfg.agreement_spread    = 0.05
            cfg.min_matches         = 5
            cfg.exclude_same_ticker = True
            cfg.same_sector_only    = False
            cfg.regime_filter       = False   # HOLD applied post-hoc
            cfg.regime_fallback     = False
            cfg.projection_horizon  = HORIZON
            cfg.cal_max_samples     = 100_000
            cfg.use_hnsw            = True
            cfg.nn_jobs             = 1        # MUST stay 1 — Windows/Py3.12 deadlock
            cfg.feature_set         = "returns_candle"
            cfg.use_sax_filter      = False
            cfg.use_wfa_rerank      = False
            cfg.use_ib_compression  = False
            cfg.use_sector_conviction = False
            cfg.use_momentum_filter   = False
            cfg.use_sentiment_veto    = False

            print(f"[{fi+1}/6] fold={fold['label']}  "
                  f"train={len(train_db):,}  val={len(val_db):,}  "
                  f"bear={bear_mask.sum()}/{len(val_db)} ({bear_mask.mean():.1%})")
            t_fit = time.time()
            matcher = PatternMatcher(cfg)
            matcher.fit(train_db, RETURNS_CANDLE_COLS)
            print(f"  Fit done in {time.time() - t_fit:.0f}s. "
                  f"Sweeping {len(MAX_DISTANCE_VALUES)} distances...", flush=True)

            print(f"  {'d':>5}  {'AvgK':>6}  {'BSS':>9}  "
                  f"{'n_scored':>8}  {'n_buy':>6}  {'n_sell':>6}  {'n_hold':>6}  {'t(s)':>5}")
            print(f"  {'-'*70}")

            for d in MAX_DISTANCE_VALUES:
                cfg.max_distance = d
                t0 = time.time()
                probs_raw, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
                elapsed = time.time() - t0

                probs    = np.asarray(probs_raw)
                sigs     = np.array(signals)
                avg_k    = float(np.mean(n_matches))

                # Degenerate fold detection
                if avg_k < 1.0 and d >= FIT_DISTANCE:
                    print(f"  [WARNING] AvgK={avg_k:.3f} at max_distance={FIT_DISTANCE} on fold "
                          f"{fold['label']}. Possible data loading or feature computation "
                          f"bug — investigate before proceeding.")

                # Apply H7 HOLD: bear rows → base_rate (net BSS effect ≈ 0)
                probs_hold = probs.copy()
                probs_hold[bear_mask] = base_rate
                bss_val = _bss(probs_hold, y_true)

                n_buy  = int(np.sum(sigs == "BUY"))
                n_sell = int(np.sum(sigs == "SELL"))
                matcher_hold = int(np.sum(sigs == "HOLD"))
                regime_hold  = int(np.sum(bear_mask & (sigs != "HOLD")))
                n_hold_total = matcher_hold + regime_hold
                n_scored     = len(sigs) - n_hold_total

                print(f"  {d:>5.1f}  {avg_k:>6.1f}  {bss_val:>+9.5f}  "
                      f"{n_scored:>8}  {n_buy:>6}  {n_sell:>6}  {n_hold_total:>6}  "
                      f"{elapsed:>5.0f}")

                rows.append({
                    "max_distance": d,
                    "fold":         fi,
                    "fold_label":   fold["label"],
                    "AvgK":         round(avg_k, 3),
                    "BSS":          round(bss_val, 6),
                    "n_scored":     n_scored,
                    "n_buy":        n_buy,
                    "n_sell":       n_sell,
                    "n_hold":       n_hold_total,
                    "runtime_s":    round(elapsed, 2),
                })

            print()

    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    # ── Write TSV ──────────────────────────────────────────────────────────────
    tsv_path = RESULTS_DIR / "sweep_max_distance_23d_standardized.tsv"
    df_out = (
        pd.DataFrame(rows)
        .sort_values(["max_distance", "fold"])
        .reset_index(drop=True)
    )
    df_out.to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {tsv_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  SUMMARY — AvgK across all 6 folds per max_distance value")
    print("=" * 72)
    print(f"{'max_dist':>9}  {'mean_AvgK':>9}  {'folds≥20':>8}  "
          f"{'mean_BSS':>9}  note")
    print("-" * 72)

    winner: float | None = None
    for d in MAX_DISTANCE_VALUES:
        sub       = df_out[df_out["max_distance"] == d]
        mean_avgk = float(sub["AvgK"].mean())
        folds_ok  = int((sub["AvgK"] >= AVGK_GATE).sum())
        mean_bss  = float(sub["BSS"].mean())
        note = ""
        if folds_ok == 6 and winner is None:
            winner = d
            note = "★ WINNER"
        print(f"{d:>9.1f}  {mean_avgk:>9.1f}  {folds_ok:>8}  "
              f"{mean_bss:>+9.5f}  {note}")

    print("-" * 72)
    print(f"\nTotal runtime: {time.time() - t_total:.0f}s")

    if winner is not None:
        print(f"\n★ WINNER: max_distance = {winner}")
        print(f"  Smallest value where ALL 6 folds have AvgK ≥ {AVGK_GATE}.")
        print(f"  Provenance: {tsv_path}")
    else:
        print(f"\nNO WINNER found in sweep range [1.0, 5.0].")
        print("  Standardization may have broken feature scaling.")
        print("  Check _prepare_features output for NaN/Inf.")
        print("  Check that candlestick columns are being standardized (not zeros).")

    sys.exit(0)


if __name__ == "__main__":
    main()
