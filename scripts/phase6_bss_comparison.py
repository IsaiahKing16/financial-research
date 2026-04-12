"""
scripts/phase6_bss_comparison.py — Task 6.2: BSS head-to-head comparison.

Determines whether the 15 candlestick features improve prediction quality
over the 8D VOL_NORM baseline. Runs two full 6-fold walk-forward configs:

  Config A (baseline):  returns_only (8D),  max_distance=0.90
  Config B (candidate): returns_candle (23D), max_distance=WINNER (read from Task 6.1 TSV)

Gate: returns_candle BSS ≥ returns_only BSS on ≥ 4 of 6 folds → PASS
      Exactly 3/6 → DRAW (do NOT promote; expansion must show clear dominance)
      < 3/6 → FAIL

Both configs use beta_abm calibration + H7 HOLD regime (SPY threshold=+0.05).
Full fit+query per fold per config (no calibration shortcuts).

Prerequisite: Task 6.1 must be complete.
  results/phase6/sweep_max_distance_23d.tsv must exist and contain a WINNER.

Provenance: HANDOFF_Phase6-remainder_Phase7.md, Task 6.2

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase6_bss_comparison.py

Output:
    results/phase6/bss_comparison_candle_vs_baseline.tsv
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
from pattern_engine.features import FeatureRegistry, VOL_NORM_COLS
from pattern_engine.candlestick import compute_candlestick_features, CANDLE_COLS

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError("betacal is not installed. Run: py -3.12 -m pip install betacal")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = REPO_ROOT / "data" / "52t_features"
RESULTS_DIR    = REPO_ROOT / "results" / "phase6"
SWEEP_TSV      = RESULTS_DIR / "sweep_max_distance_23d.tsv"
OUTPUT_TSV     = RESULTS_DIR / "bss_comparison_candle_vs_baseline.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature sets ──────────────────────────────────────────────────────────────
RETURNS_ONLY_COLS: list[str]   = list(VOL_NORM_COLS)           # 8 cols
RETURNS_CANDLE_COLS: list[str] = FeatureRegistry.get("returns_candle").columns  # 23 cols

# ── Locked settings (CLAUDE.md) ───────────────────────────────────────────────
MAX_DISTANCE_BASELINE = 0.90   # H5 locked for returns_only (8D)
SPY_THRESHOLD         = 0.05   # H7 locked: bear = SPY ret_90d < +0.05
HORIZON               = "fwd_7d_up"
AVGK_GATE             = 20     # used when reading WINNER from sweep TSV

# ── Gate thresholds ───────────────────────────────────────────────────────────
PASS_THRESHOLD = 4   # ≥ 4/6 folds → PASS
DRAW_THRESHOLD = 3   # exactly 3/6 → DRAW


# ── Pure logic (testable without I/O) ────────────────────────────────────────

def _find_winner_from_df(
    df: pd.DataFrame,
    avgk_gate: int = AVGK_GATE,
) -> float | None:
    """Return the smallest max_distance where ALL folds have AvgK ≥ avgk_gate.

    Args:
        df: DataFrame with columns [max_distance, fold, AvgK].
        avgk_gate: Minimum AvgK required on every fold.

    Returns:
        The winning max_distance value, or None if no value passes.
    """
    for d in sorted(df["max_distance"].unique()):
        sub = df[df["max_distance"] == d]
        if (sub["AvgK"] >= avgk_gate).all():
            return float(d)
    return None


def _evaluate_gate(
    bss_a: list[float],
    bss_b: list[float],
) -> str:
    """Classify the head-to-head BSS comparison outcome.

    Args:
        bss_a: Per-fold BSS for Config A (returns_only baseline).
        bss_b: Per-fold BSS for Config B (returns_candle candidate).

    Returns:
        "PASS"  — B wins ≥ PASS_THRESHOLD folds
        "DRAW"  — B wins exactly DRAW_THRESHOLD folds
        "FAIL"  — B wins < DRAW_THRESHOLD folds
    """
    wins = sum(1 for a, b in zip(bss_a, bss_b) if b > a)
    if wins >= PASS_THRESHOLD:
        return "PASS"
    if wins == DRAW_THRESHOLD:
        return "DRAW"
    return "FAIL"


# ── Beta calibrator (beta_abm — locked since H5) ──────────────────────────────

class _BetaCalibrator:
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    base_rate = y_true.mean()
    bs_ref = base_rate * (1.0 - base_rate)
    if bs_ref < 1e-10:
        return 0.0
    return 1.0 - float(np.mean((probs - y_true) ** 2)) / bs_ref


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
    print(f"  {len(CANDLE_COLS)} columns in {time.time() - t0:.1f}s")
    augmented = pd.concat(
        [full_db.reset_index(drop=True), candle_df.reset_index(drop=True)], axis=1
    )
    augmented = _impute_candle_nans(augmented)
    remaining = int(augmented[CANDLE_COLS].isna().values.sum())
    if remaining > 0:
        raise RuntimeError(
            f"Imputation failed: {remaining} NaNs remain after fillna."
        )
    return augmented


def _build_cfg(max_distance: float, feature_set: str) -> EngineConfig:
    """Build a locked EngineConfig for either config A or B."""
    cfg = EngineConfig()
    cfg.max_distance         = max_distance
    cfg.top_k                = 50
    cfg.distance_weighting   = "uniform"
    cfg.confidence_threshold = 0.65
    cfg.agreement_spread     = 0.05
    cfg.min_matches          = 5
    cfg.exclude_same_ticker  = True
    cfg.same_sector_only     = False
    cfg.regime_filter        = False   # HOLD applied post-hoc
    cfg.regime_fallback      = False
    cfg.projection_horizon   = HORIZON
    cfg.cal_max_samples      = 100_000
    cfg.use_hnsw             = True
    cfg.nn_jobs              = 1        # MUST stay 1 — Windows/Py3.12 deadlock
    cfg.feature_set          = feature_set
    cfg.use_sax_filter       = False
    cfg.use_wfa_rerank       = False
    cfg.use_ib_compression   = False
    cfg.use_sector_conviction  = False
    cfg.use_momentum_filter    = False
    cfg.use_sentiment_veto     = False
    return cfg


def _run_config(
    label: str,
    feature_cols: list[str],
    feature_set: str,
    max_distance: float,
    full_db: pd.DataFrame,
) -> list[dict]:
    """Run 6-fold walk-forward for one config. Returns list of per-fold result dicts."""
    print(f"\n{'─'*68}")
    print(f"  Config {label}: feature_set={feature_set!r}  max_distance={max_distance}")
    print(f"{'─'*68}")

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
                rows.append({
                    "config": label, "fold": fi, "fold_label": fold["label"],
                    "BSS": float("nan"), "AvgK": 0.0,
                    "n_scored": 0, "n_hold": len(val_db),
                    "mean_confidence": float("nan"),
                })
                continue

            regime_labeler = _ThresholdRegimeLabeler(threshold=SPY_THRESHOLD)
            regime_labeler.fit(train_db)
            val_regime = regime_labeler.label(val_db)
            bear_mask  = (val_regime == 0)
            y_true     = val_db[HORIZON].values.astype(float)
            base_rate  = float(y_true.mean())

            cfg = _build_cfg(max_distance=max_distance, feature_set=feature_set)

            print(f"  [{fi+1}/6] {fold['label']}  train={len(train_db):,}  "
                  f"val={len(val_db):,}  bear={bear_mask.mean():.1%}", end=" ",
                  flush=True)
            t0 = time.time()
            matcher = PatternMatcher(cfg)
            matcher.fit(train_db, feature_cols)
            probs_raw, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
            elapsed = time.time() - t0

            probs = np.asarray(probs_raw)
            sigs  = np.array(signals)
            avg_k = float(np.mean(n_matches))

            # H7 HOLD: bear rows → base_rate
            probs_hold = probs.copy()
            probs_hold[bear_mask] = base_rate
            bss_val = _bss(probs_hold, y_true)

            # Scored rows: matcher says BUY/SELL AND not bear-overridden
            matcher_hold  = int(np.sum(sigs == "HOLD"))
            regime_hold   = int(np.sum(bear_mask & (sigs != "HOLD")))
            n_hold_total  = matcher_hold + regime_hold
            n_scored      = len(sigs) - n_hold_total

            # mean_confidence: avg |P(up) - 0.5| for rows that got BUY/SELL from matcher
            scored_mask   = (sigs != "HOLD") & ~bear_mask
            mean_conf     = float(np.mean(np.abs(probs[scored_mask] - 0.5))) \
                            if scored_mask.any() else float("nan")

            print(f"→ BSS={bss_val:+.5f}  AvgK={avg_k:.1f}  "
                  f"scored={n_scored}  t={elapsed:.0f}s")

            rows.append({
                "config":          label,
                "fold":            fi,
                "fold_label":      fold["label"],
                "BSS":             round(bss_val, 6),
                "AvgK":            round(avg_k, 3),
                "n_scored":        n_scored,
                "n_hold":          n_hold_total,
                "mean_confidence": round(mean_conf, 5) if not np.isnan(mean_conf) else float("nan"),
            })

    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    print("=" * 68)
    print("  Task 6.2 — BSS comparison: returns_candle vs returns_only")
    print("=" * 68)

    # ── Read WINNER from Task 6.1 ──────────────────────────────────────────────
    if not SWEEP_TSV.exists():
        print(f"ERROR: Task 6.1 output not found: {SWEEP_TSV}")
        print("       Run scripts/sweep_max_distance_23d.py first.")
        sys.exit(1)

    sweep_df = pd.read_csv(SWEEP_TSV, sep="\t")
    winner = _find_winner_from_df(sweep_df, avgk_gate=AVGK_GATE)
    if winner is None:
        print(f"ERROR: No WINNER found in {SWEEP_TSV}.")
        print(f"       No max_distance value achieves AvgK ≥ {AVGK_GATE} on all 6 folds.")
        sys.exit(1)

    print(f"\n  Config A (baseline) : returns_only (8D),   max_distance={MAX_DISTANCE_BASELINE}")
    print(f"  Config B (candidate): returns_candle (23D), max_distance={winner}")
    print(f"  Gate                : B wins ≥ {PASS_THRESHOLD}/6 folds → PASS")
    print(f"  Calibration         : beta_abm (H5 locked)")
    print(f"  Regime              : HOLD mode, SPY threshold=+{SPY_THRESHOLD:.2f} (H7 locked)")
    print(f"  Winner source       : {SWEEP_TSV.name}")

    # ── Load + augment data ────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"
    if not t_path.exists() or not v_path.exists():
        print(f"\nERROR: 52T features data not found in {DATA_DIR}")
        print("       Run scripts/build_52t_features.py first.")
        sys.exit(1)

    print(f"\nLoading 52T features data from {DATA_DIR}...")
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers")

    missing_ohlc = [c for c in ("Ticker", "Open", "High", "Low", "Close")
                    if c not in full_db.columns]
    if missing_ohlc:
        print(f"ERROR: Missing OHLC columns: {missing_ohlc}")
        sys.exit(1)

    # Augment once; Config A simply ignores the candle columns at fit time
    full_db = _augment_with_candlestick(full_db)

    # ── Run both configs ───────────────────────────────────────────────────────
    rows_a = _run_config(
        label="returns_only",
        feature_cols=RETURNS_ONLY_COLS,
        feature_set="returns_only",
        max_distance=MAX_DISTANCE_BASELINE,
        full_db=full_db,
    )
    rows_b = _run_config(
        label="returns_candle",
        feature_cols=RETURNS_CANDLE_COLS,
        feature_set="returns_candle",
        max_distance=winner,
        full_db=full_db,
    )

    # ── Write TSV ──────────────────────────────────────────────────────────────
    all_rows = rows_a + rows_b
    df_out = (
        pd.DataFrame(all_rows)
        .sort_values(["config", "fold"])
        .reset_index(drop=True)
    )
    df_out.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.6f")
    print(f"\nSaved: {OUTPUT_TSV}")

    # ── Gate evaluation ────────────────────────────────────────────────────────
    bss_a = [r["BSS"] for r in sorted(rows_a, key=lambda x: x["fold"])]
    bss_b = [r["BSS"] for r in sorted(rows_b, key=lambda x: x["fold"])]
    gate  = _evaluate_gate(bss_a, bss_b)

    print()
    print("=" * 68)
    print("  GATE EVALUATION")
    print("=" * 68)
    header = f"{'Fold':<14}  {'returns_only BSS':>16}  {'returns_candle BSS':>18}  WINNER"
    print(header)
    print("-" * 68)

    wins_b = 0
    for ra, rb in zip(
        sorted(rows_a, key=lambda x: x["fold"]),
        sorted(rows_b, key=lambda x: x["fold"]),
    ):
        winner_flag = "B (candle)" if rb["BSS"] > ra["BSS"] else "A (baseline)"
        if rb["BSS"] > ra["BSS"]:
            wins_b += 1
        bss_a_s = f"{ra['BSS']:+.5f}" if not np.isnan(ra["BSS"]) else "    N/A"
        bss_b_s = f"{rb['BSS']:+.5f}" if not np.isnan(rb["BSS"]) else "    N/A"
        print(f"{ra['fold_label']:<14}  {bss_a_s:>16}  {bss_b_s:>18}  {winner_flag}")

    print("-" * 68)
    print(f"  returns_candle wins: {wins_b}/6 folds")
    print(f"  GATE: {gate}")

    total = time.time() - t_total
    print(f"\nTotal runtime: {total:.0f}s")

    if gate == "PASS":
        print(f"\n[PASS] Candlestick features improve BSS on {wins_b}/6 folds.")
        print(f"  Proceed to Task 6.3 (body_position redundancy test).")
        print(f"  Recommended locked settings update for CLAUDE.md:")
        print(f"    feature_set=returns_candle, max_distance={winner} (23D calibration)")
    elif gate == "DRAW":
        print(f"\n[DRAW] Candlestick features win exactly 3/6 folds.")
        print(f"  Candlestick features do NOT demonstrate clear dominance.")
        print(f"  Reverting to returns_only. Phase 6 closes with baseline preserved.")
        print(f"  Candlestick feature code is archived; locked settings unchanged.")
    else:
        print(f"\n[FAIL] Candlestick features do not improve BSS ({wins_b}/6 folds).")
        print(f"  Reverting to returns_only. Phase 6 closes with baseline preserved.")
        print(f"  Candlestick feature code is archived; locked settings unchanged.")

    sys.exit(0)


if __name__ == "__main__":
    main()
