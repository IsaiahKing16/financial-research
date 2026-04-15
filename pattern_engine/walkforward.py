"""
pattern_engine/walkforward.py — Walk-forward fold runner, BSS, data loading.

Extracted from scripts/phase7_baseline.py into a production module for reuse
by Optuna sweep infrastructure (Phase 3 P3) and enhancement scripts.

Public API:
    run_fold(fold, full_db, feature_cols, cfg_overrides) -> dict
    run_walkforward(full_db, feature_cols, cfg_overrides, folds) -> dict
    load_and_augment_db(data_dir) -> DataFrame

Private helpers (logic identical to phase7_baseline.py):
    _BetaCalibrator, _bss, _murphy_decomposition, _apply_h7_hold_regime,
    _impute_candle_nans, _augment_with_candlestick, _build_cfg

Provenance: Phase 3 Optuna plan, Tasks 1-3 (2026-04-11)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

import pattern_engine.matcher as _matcher_module
from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import FeatureRegistry
from pattern_engine.candlestick import compute_candlestick_features, CANDLE_COLS

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError("betacal is not installed. Run: py -3.12 -m pip install betacal")

# ── Constants ────────────────────────────────────────────────────────────────

HORIZON = "fwd_7d_up"
SPY_THRESHOLD = 0.05
def _find_data_dir() -> Path:
    """Resolve 52T features directory, with worktree fallback."""
    # Try relative to this module's repo root
    module_root = Path(__file__).resolve().parent.parent
    candidate = module_root / "data" / "52t_features"
    if candidate.exists():
        return candidate
    # Worktree fallback: .worktrees/<name>/ → go up two levels to main repo
    main_repo = module_root.parent.parent
    alt = main_repo / "data" / "52t_features"
    if alt.exists():
        return alt
    return Path("data/52t_features")  # CWD-relative fallback for production

DATA_DIR = _find_data_dir()
FEATURE_COLS: list[str] = FeatureRegistry.get("returns_candle").columns   # 23 columns
CANDLE_FEATURE_COLS: list[str] = list(CANDLE_COLS)
MURPHY_BINS = 10
_NON_CONFIG_KEYS = {"cal_frac"}


# ── Beta calibrator (beta_abm — locked since H5) ────────────────────────────

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


# ── BSS and Murphy decomposition ────────────────────────────────────────────

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


# ── Regime labeler — H7 HOLD mode ───────────────────────────────────────────

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


# ── Candlestick augmentation ────────────────────────────────────────────────

def _impute_candle_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs in candlestick columns with neutral values."""
    df = df.copy()
    prop_fill = {c: 0.0 for c in CANDLE_COLS if "direction" not in c}
    dir_fill  = {c: 1.0 for c in CANDLE_COLS if "direction" in c}
    df.fillna({**prop_fill, **dir_fill}, inplace=True)
    return df


def _augment_with_candlestick(full_db: pd.DataFrame) -> pd.DataFrame:
    """Compute and append candlestick features to full_db."""
    t0 = time.time()
    candle_df = compute_candlestick_features(full_db)
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


# ── Config builder ───────────────────────────────────────────────────────────

FEATURE_SET = "returns_candle"
MAX_DISTANCE = 2.5
TOP_K = 50


def _build_cfg(cfg_overrides: dict | None = None) -> EngineConfig:
    """Build locked EngineConfig, then apply any overrides dict.

    IMPORTANT: _NON_CONFIG_KEYS (e.g. cal_frac) are stripped from a COPY of
    cfg_overrides before applying to EngineConfig, then injected via setattr.
    The caller's dict is never mutated.

    Args:
        cfg_overrides: Dict of {field_name: value} to set after building the
                       locked baseline config. Used by enhancement scripts.

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
        # Copy so we don't mutate the caller's dict
        overrides = dict(cfg_overrides)

        # Strip non-config keys (consumed by matcher via getattr, not EngineConfig fields)
        stripped = {}
        for key in _NON_CONFIG_KEYS:
            if key in overrides:
                stripped[key] = overrides.pop(key)

        # Apply remaining overrides to EngineConfig
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise RuntimeError(
                    f"_build_cfg: unknown EngineConfig field {k!r}. "
                    "Check cfg_overrides keys."
                )
            setattr(cfg, k, v)

        # Inject stripped keys via setattr so matcher.py can access them
        for k, v in stripped.items():
            setattr(cfg, k, v)

    return cfg


# ── Core fold runner ─────────────────────────────────────────────────────────

def run_fold(
    fold: dict,
    full_db: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg_overrides: dict | None = None,
) -> dict:
    """Run one walk-forward fold and return a result dict.

    Args:
        fold:          One entry from WALKFORWARD_FOLDS.
        full_db:       Augmented full database (train + val concatenated,
                       with candlestick features added).
        feature_cols:  Feature column list. Defaults to locked 23-col set.
        cfg_overrides: Dict of EngineConfig field overrides.

    Returns:
        Dict with keys: fold, bss, n_scored, n_total, base_rate,
                        mean_prob, reliability, resolution, uncertainty.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    # Temporal integrity check — prevent data leakage from training into validation
    if train_end >= val_start:
        raise RuntimeError(
            f"Fold '{fold.get('label', '?')}': train_end ({train_end.date()}) must be "
            f"strictly before val_start ({val_start.date()}). "
            "Overlapping folds cause data leakage. Check WALKFORWARD_FOLDS in config.py."
        )
    if val_start >= val_end:
        raise RuntimeError(
            f"Fold '{fold.get('label', '?')}': val_start ({val_start.date()}) must be "
            f"strictly before val_end ({val_end.date()})."
        )

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
    y_true    = val_db[HORIZON].values.astype(float) if n_total > 0 else np.array([])
    base_rate = float(y_true.mean()) if n_total > 0 else 0.0

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

    # Monkey-patch _PlattCalibrator -> _BetaCalibrator for this fold
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

    # H7 HOLD regime: bear rows -> base_rate probability
    probs_hold, bear_mask = _apply_h7_hold_regime(
        val_db=val_db,
        train_db=train_db,
        base_rate=base_rate,
        probs=probs,
    )

    # Scored rows: all non-bear rows (SPY ret_90d >= threshold), regardless of signal
    scored_mask = ~bear_mask
    n_scored    = int(scored_mask.sum())
    mean_prob   = float(probs_hold[scored_mask].mean()) if scored_mask.any() else float("nan")

    # BSS uses H7-overridden probabilities on scored (non-bear) rows only
    bss_val = _bss(probs_hold[scored_mask], y_true[scored_mask])

    # Murphy decomposition on scored rows only
    if scored_mask.any():
        rel, res, unc = _murphy_decomposition(probs_hold[scored_mask], y_true[scored_mask])
        # BSS identity guard: BS = REL - RES + UNC  (Murphy decomposition invariant)
        bs_val = float(np.mean((probs_hold[scored_mask] - y_true[scored_mask]) ** 2))
        identity_residual = abs(rel - res + unc - bs_val)
        if identity_residual > 0.05:  # Murphy binning noise up to ~0.004 (clustered probs);
                                         # 0.05 gives 13x margin while catching formula bugs
            raise RuntimeError(
                f"Murphy BSS identity violated in fold '{fold.get('label', '?')}': "
                f"|REL-RES+UNC-BS| = {identity_residual:.2e}. "
                "This indicates a bug in _murphy_decomposition. "
                f"REL={rel:.8f}, RES={res:.8f}, UNC={unc:.8f}, BS={bs_val:.8f}"
            )
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
        "resolution":  round(res, 8)       if not np.isnan(res)       else float("nan"),
        "uncertainty": round(unc, 6),
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_augment_db(data_dir: str | Path = DATA_DIR) -> pd.DataFrame:
    """Load train_db + val_db parquets, concatenate, augment with candlestick features.

    Args:
        data_dir: Directory containing train_db.parquet and val_db.parquet.
                  Defaults to DATA_DIR ("data/52t_features").

    Returns:
        Augmented DataFrame with candlestick features appended and NaNs imputed.
    """
    data_dir = Path(data_dir)

    t_path = data_dir / "train_db.parquet"
    v_path = data_dir / "val_db.parquet"

    if not t_path.exists() or not v_path.exists():
        raise RuntimeError(
            f"52T features data not found in {data_dir}. "
            "Run scripts/build_52t_features.py first."
        )

    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    return _augment_with_candlestick(full_db)


# ── Walk-forward orchestrator ────────────────────────────────────────────────

def run_walkforward(
    full_db: pd.DataFrame,
    feature_cols: list[str] | None = None,
    cfg_overrides: dict | None = None,
    folds: list[dict] | None = None,
) -> dict:
    """Run all walk-forward folds and compute aggregate metrics.

    Args:
        full_db:       Augmented full database.
        feature_cols:  Feature column list. Defaults to locked 23-col set.
        cfg_overrides: Dict of EngineConfig field overrides.
        folds:         List of fold dicts. Defaults to WALKFORWARD_FOLDS (6 folds).

    Returns:
        Dict with keys: mean_bss, trimmed_mean_bss, positive_folds,
                        fold_results, wilcoxon_p.
    """
    if folds is None:
        folds = WALKFORWARD_FOLDS

    fold_results = []
    for fold in folds:
        result = run_fold(
            fold=fold,
            full_db=full_db,
            feature_cols=feature_cols,
            cfg_overrides=cfg_overrides,
        )
        fold_results.append(result)

    # Compute aggregate metrics
    bss_vals = [r["bss"] for r in fold_results if not np.isnan(r["bss"])]

    if bss_vals:
        mean_bss = float(np.mean(bss_vals))
        # Trimmed mean: drop worst (lowest) fold
        sorted_bss = sorted(bss_vals)
        trimmed = sorted_bss[1:]  # drop worst
        trimmed_mean_bss = float(np.mean(trimmed)) if trimmed else mean_bss
        positive_folds = sum(1 for b in bss_vals if b > 0)
    else:
        mean_bss = float("nan")
        trimmed_mean_bss = float("nan")
        positive_folds = 0

    # Wilcoxon signed-rank test: H1: BSS > 0
    wilcoxon_p = None
    non_zero_bss = [b for b in bss_vals if b != 0.0]
    if len(non_zero_bss) >= 6:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(non_zero_bss, alternative="greater")
        wilcoxon_p = float(p)

    return {
        "mean_bss": mean_bss,
        "trimmed_mean_bss": trimmed_mean_bss,
        "positive_folds": positive_folds,
        "fold_results": fold_results,
        "wilcoxon_p": wilcoxon_p,
    }
