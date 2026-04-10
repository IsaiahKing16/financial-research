"""
scripts/phase7_baseline.py — Task T7.0: Phase 7 baseline re-establishment.

Runs 6-fold walk-forward with locked Phase 6 settings:
  feature_set=returns_candle (23D), max_distance=2.5, top_k=50,
  calibration=beta_abm, regime=H7 HOLD (bear rows → base_rate when
  SPY ret_90d < +0.05).

Output: results/phase7/baseline_23d.tsv

This is the single comparison target for all 6 Phase 7 enhancements (E1–E6).
Enhancement scripts import run_fold_with_config() to override specific
EngineConfig fields while keeping everything else locked.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase7_baseline.py

Provenance: Phase 7 implementation plan, Task T7.0 (2026-04-09)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root resolution (works in worktrees and main repo) ───────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent

# Worktree data fallback: if data/52t_features/ doesn't exist here,
# check the main repo (parent of .worktrees/).
def _find_data_dir() -> Path:
    candidate = REPO_ROOT / "data" / "52t_features"
    if candidate.exists():
        return candidate
    # Worktree layout: REPO_ROOT = .../financial-research/.worktrees/phase7-enhancements
    # Main repo:       REPO_ROOT.parent.parent = .../financial-research
    main_repo = REPO_ROOT.parent.parent
    alt = main_repo / "data" / "52t_features"
    if alt.exists():
        return alt
    raise RuntimeError(
        f"52T features data not found.\n"
        f"  Tried: {candidate}\n"
        f"  Tried: {alt}\n"
        f"  Run scripts/build_52t_features.py first."
    )


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
DATA_DIR    = _find_data_dir()
RESULTS_DIR = REPO_ROOT / "results" / "phase7"
OUTPUT_TSV  = RESULTS_DIR / "baseline_23d.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Locked settings (Phase 6 → Phase 7 baseline) ─────────────────────────────
FEATURE_SET   = "returns_candle"
FEATURE_COLS  = FeatureRegistry.get(FEATURE_SET).columns   # 23 columns
MAX_DISTANCE  = 2.5           # Phase 6 locked; see CLAUDE.md
TOP_K         = 50
HORIZON       = "fwd_7d_up"
SPY_THRESHOLD = 0.05          # H7 HOLD: bear = SPY ret_90d < +0.05
MURPHY_BINS   = 10            # bins for Murphy decomposition


# ── Beta calibrator (beta_abm — locked since H5) ──────────────────────────────

class _BetaCalibrator:
    """Drop-in replacement for _PlattCalibrator using BetaCalibration(parameters='abm')."""

    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ── BSS and Murphy decomposition ──────────────────────────────────────────────

def _bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Skill Score vs climatological baseline."""
    base_rate = float(y_true.mean())
    bs_ref = base_rate * (1.0 - base_rate)
    if bs_ref < 1e-10:
        return 0.0
    brier = float(np.mean((probs - y_true) ** 2))
    return 1.0 - brier / bs_ref


def _murphy_decomposition(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = MURPHY_BINS,
) -> tuple[float, float, float]:
    """Murphy decomposition of Brier Score into reliability + resolution + uncertainty.

    BSS = (resolution - reliability) / uncertainty   (when uncertainty > 0)

    Returns:
        reliability:  calibration term (lower = better)
        resolution:   discrimination term (higher = better)
        uncertainty:  climatological variance = base_rate*(1-base_rate)
    """
    base_rate = float(y_true.mean())
    uncertainty = base_rate * (1.0 - base_rate)

    if len(probs) == 0 or uncertainty < 1e-10:
        return float("nan"), float("nan"), uncertainty

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probs, bins, right=True)
    bin_idx = np.clip(bin_idx, 1, n_bins)

    reliability = 0.0
    resolution  = 0.0

    for b in range(1, n_bins + 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        n_b      = mask.sum()
        mean_p   = float(probs[mask].mean())
        mean_y   = float(y_true[mask].mean())
        w        = n_b / len(probs)
        reliability += w * (mean_p - mean_y) ** 2
        resolution  += w * (mean_y - base_rate) ** 2

    return reliability, resolution, uncertainty


# ── Regime labeler — H7 HOLD mode ─────────────────────────────────────────────

def _apply_h7_hold_regime(
    val_db: pd.DataFrame,
    train_db: pd.DataFrame,
    base_rate: float,
    probs: np.ndarray,
    threshold: float = SPY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply H7 HOLD regime: bear rows get base_rate instead of model probability.

    Bear row: SPY ret_90d on that date < threshold (+0.05).
    SPY dates from val_db; if SPY not in val_db, fall back to train_db SPY dates.

    Returns:
        probs_hold: probability array with bear rows set to base_rate
        bear_mask:  boolean array, True for rows that are "bear" (overridden)
    """
    # Extract SPY ret_90d time series from validation window
    spy_val = val_db[val_db["Ticker"] == "SPY"].copy()
    if spy_val.empty:
        # Fallback: use train SPY — rare edge for early folds
        spy_val = train_db[train_db["Ticker"] == "SPY"].copy()

    spy_val["Date"] = pd.to_datetime(spy_val["Date"])
    spy_val = spy_val.set_index("Date").sort_index()
    spy_ret90 = spy_val["ret_90d"]

    # Map each row's date to the nearest SPY ret_90d observation
    row_dates = pd.to_datetime(val_db["Date"])
    mapped = spy_ret90.reindex(row_dates.values, method="nearest")
    bear_mask = mapped.values < threshold

    probs_hold = probs.copy()
    probs_hold[bear_mask] = base_rate

    return probs_hold, bear_mask


# ── Candlestick augmentation ──────────────────────────────────────────────────

def _impute_candle_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs in candlestick columns with neutral values."""
    df = df.copy()
    prop_fill = {c: 0.0 for c in CANDLE_COLS if "direction" not in c}
    dir_fill  = {c: 1.0 for c in CANDLE_COLS if "direction" in c}
    df.fillna({**prop_fill, **dir_fill}, inplace=True)
    return df


def _augment_with_candlestick(full_db: pd.DataFrame) -> pd.DataFrame:
    """Compute and append candlestick features to full_db."""
    print("Computing candlestick features...", flush=True)
    t0 = time.time()
    candle_df = compute_candlestick_features(full_db)
    print(f"  {len(CANDLE_COLS)} candle columns computed in {time.time() - t0:.1f}s")
    augmented = pd.concat(
        [full_db.reset_index(drop=True), candle_df.reset_index(drop=True)], axis=1
    )
    augmented = _impute_candle_nans(augmented)
    remaining = int(augmented[CANDLE_COLS].isna().values.sum())
    if remaining > 0:
        raise RuntimeError(
            f"Candlestick imputation failed: {remaining} NaNs remain after fillna."
        )
    return augmented


# ── Config builder ────────────────────────────────────────────────────────────

def _build_cfg(cfg_overrides: dict | None = None) -> EngineConfig:
    """Build locked EngineConfig, then apply any overrides dict.

    Args:
        cfg_overrides: Dict of {field_name: value} to set after building the
                       locked baseline config. Used by E1–E6 enhancement scripts.

    Returns:
        EngineConfig with locked Phase 6 settings + any overrides applied.
    """
    cfg = EngineConfig()
    cfg.max_distance         = MAX_DISTANCE
    cfg.top_k                = TOP_K
    cfg.distance_weighting   = "uniform"
    cfg.distance_metric      = "euclidean"
    cfg.confidence_threshold = 0.65
    cfg.agreement_spread     = 0.05
    cfg.min_matches          = 5
    cfg.exclude_same_ticker  = True
    cfg.same_sector_only     = False
    cfg.regime_filter        = False    # HOLD applied post-hoc below
    cfg.regime_fallback      = False
    cfg.projection_horizon   = HORIZON
    cfg.cal_max_samples      = 100_000
    cfg.use_hnsw             = True
    cfg.nn_jobs              = 1        # MUST stay 1 — Windows/Py3.12 deadlock
    cfg.feature_set          = FEATURE_SET
    cfg.use_sax_filter       = False
    cfg.use_wfa_rerank       = False
    cfg.use_ib_compression   = False
    cfg.use_sector_conviction  = False
    cfg.use_momentum_filter    = False
    cfg.use_sentiment_veto     = False
    # Phase 7 flags all default False
    cfg.use_bma              = False
    cfg.use_owa              = False
    cfg.use_dtw_reranker     = False
    cfg.use_conformal        = False
    cfg.use_anomaly_filter   = False
    cfg.use_stumpy           = False

    if cfg_overrides:
        for k, v in cfg_overrides.items():
            if not hasattr(cfg, k):
                raise RuntimeError(
                    f"_build_cfg: unknown EngineConfig field {k!r}. "
                    "Check cfg_overrides keys."
                )
            setattr(cfg, k, v)

    return cfg


# ── Core fold runner (importable by E1–E6 scripts) ───────────────────────────

def run_fold_with_config(
    fold: dict,
    full_db: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    cfg_overrides: dict | None = None,
) -> dict:
    """Run one walk-forward fold and return a result dict.

    This is the primary entry point for Phase 7 enhancement scripts (E1–E6).
    They import this function, pass their own cfg_overrides, and compare the
    resulting BSS against the baseline TSV.

    Args:
        fold:          One entry from WALKFORWARD_FOLDS.
        full_db:       Augmented full database (train + val concatenated,
                       with candlestick features added).
        feature_cols:  Feature column list. Defaults to locked 23-col set.
        cfg_overrides: Dict of EngineConfig field overrides for the enhancement.

    Returns:
        Dict with keys: fold, bss, n_scored, n_total, base_rate,
                        mean_prob, reliability, resolution, uncertainty.
    """
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

    n_total   = len(val_db)
    y_true    = val_db[HORIZON].values.astype(float)
    base_rate = float(y_true.mean())

    nan_row = {
        "fold":        fold["label"],
        "bss":         float("nan"),
        "n_scored":    0,
        "n_total":     n_total,
        "base_rate":   round(base_rate, 6),
        "mean_prob":   float("nan"),
        "reliability": float("nan"),
        "resolution":  float("nan"),
        "uncertainty": round(base_rate * (1.0 - base_rate), 6),
    }

    if n_total == 0:
        return nan_row

    cfg = _build_cfg(cfg_overrides)

    # Monkey-patch _PlattCalibrator → _BetaCalibrator for this fold
    original_calibrator = _matcher_module._PlattCalibrator
    try:
        _matcher_module._PlattCalibrator = _BetaCalibrator

        matcher = PatternMatcher(cfg)
        matcher.fit(train_db, list(feature_cols))
        probs_raw, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    probs = np.asarray(probs_raw)
    sigs  = np.array(signals)

    # H7 HOLD regime: bear rows → base_rate probability
    probs_hold, bear_mask = _apply_h7_hold_regime(
        val_db=val_db,
        train_db=train_db,
        base_rate=base_rate,
        probs=probs,
    )

    # Scored rows: matcher gave BUY/SELL AND not overridden by bear regime
    scored_mask = (sigs != "HOLD") & ~bear_mask
    n_scored    = int(scored_mask.sum())
    mean_prob   = float(probs[scored_mask].mean()) if scored_mask.any() else float("nan")

    # BSS uses H7-overridden probabilities (bear rows clamped to base_rate)
    bss_val = _bss(probs_hold, y_true)

    # Murphy decomposition on scored rows only
    if scored_mask.any():
        rel, res, unc = _murphy_decomposition(probs[scored_mask], y_true[scored_mask])
    else:
        rel, res, unc = float("nan"), float("nan"), base_rate * (1.0 - base_rate)

    return {
        "fold":        fold["label"],
        "bss":         round(bss_val, 6),
        "n_scored":    n_scored,
        "n_total":     n_total,
        "base_rate":   round(base_rate, 6),
        "mean_prob":   round(mean_prob, 6) if not np.isnan(mean_prob) else float("nan"),
        "reliability": round(rel, 8)       if not np.isnan(rel)       else float("nan"),
        "resolution":  round(res, 8)       if not np.isnan(res)        else float("nan"),
        "uncertainty": round(unc, 6),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    print("=" * 72)
    print("  Task T7.0 — Phase 7 Baseline: returns_candle (23D)")
    print("=" * 72)
    print(f"  feature_set   : {FEATURE_SET} ({len(FEATURE_COLS)} columns)")
    print(f"  max_distance  : {MAX_DISTANCE}")
    print(f"  calibration   : beta_abm (H5 locked)")
    print(f"  regime        : H7 HOLD (SPY ret_90d < +{SPY_THRESHOLD:.2f} → base_rate)")
    print(f"  horizon       : {HORIZON}")
    print(f"  data          : {DATA_DIR}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"

    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        print("       Run scripts/build_52t_features.py first.")
        sys.exit(1)

    print(f"Loading 52T features from {DATA_DIR} ...", flush=True)
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"date range {full_db['Date'].min().date()} – {full_db['Date'].max().date()}")

    # Validate required columns
    missing = [c for c in ("Ticker", "Open", "High", "Low", "Close", "ret_90d", HORIZON)
               if c not in full_db.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        sys.exit(1)

    # ── Augment with candlestick features ──────────────────────────────────────
    full_db = _augment_with_candlestick(full_db)

    # ── Run 6-fold walk-forward ────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  Running 6-fold walk-forward (locked baseline config)")
    print(f"{'─'*72}")

    rows: list[dict] = []

    for fi, fold in enumerate(WALKFORWARD_FOLDS):
        print(f"\n  [{fi+1}/6] {fold['label']}", flush=True)
        t0 = time.time()

        result = run_fold_with_config(fold=fold, full_db=full_db)

        elapsed = time.time() - t0
        bss_s = f"{result['bss']:+.5f}" if not np.isnan(result["bss"]) else "  N/A"
        print(f"         BSS={bss_s}  n_scored={result['n_scored']}  "
              f"n_total={result['n_total']}  base_rate={result['base_rate']:.4f}  "
              f"t={elapsed:.0f}s")

        rows.append(result)

    # ── Write TSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.8f")
    print(f"\nSaved: {OUTPUT_TSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    bss_vals = [r["bss"] for r in rows if not np.isnan(r["bss"])]
    mean_bss = float(np.mean(bss_vals)) if bss_vals else float("nan")
    pos_folds = sum(1 for b in bss_vals if b > 0)

    total = time.time() - t_total
    print()
    print("=" * 72)
    print("  BASELINE SUMMARY")
    print("=" * 72)
    print(f"  {'Fold':<14}  {'BSS':>9}  {'n_scored':>8}  {'n_total':>7}  {'base_rate':>9}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*9}")
    for r in rows:
        bss_s = f"{r['bss']:+.5f}" if not np.isnan(r["bss"]) else "     N/A"
        print(f"  {r['fold']:<14}  {bss_s:>9}  {r['n_scored']:>8}  "
              f"{r['n_total']:>7}  {r['base_rate']:>9.4f}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*8}  {'─'*7}  {'─'*9}")
    mean_s = f"{mean_bss:+.5f}" if not np.isnan(mean_bss) else "     N/A"
    print(f"  {'MEAN':<14}  {mean_s:>9}  {pos_folds}/6 folds positive")
    print()
    print(f"  Total runtime: {total:.0f}s")
    print(f"  Output: {OUTPUT_TSV}")
    print()

    if np.isnan(mean_bss):
        print("[WARN] Mean BSS is NaN — check fold results above.")
    elif pos_folds >= 3:
        print(f"[PASS] {pos_folds}/6 positive folds (expected ≥ 3 from H7 baseline).")
    else:
        print(f"[WARN] Only {pos_folds}/6 positive folds — baseline may differ from H7 expectation.")
        print("       This could be normal given different 23D feature set vs 8D H7 sweep.")


if __name__ == "__main__":
    main()
