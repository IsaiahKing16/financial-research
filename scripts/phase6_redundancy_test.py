"""
scripts/phase6_redundancy_test.py — Task 6.3: body_position redundancy test.

Tests whether the `body_position` candlestick feature is redundant with
`lower_wick`. If zeroing body_position (3 columns, one per timeframe) does not
degrade BSS, we can drop to 22D.

Gate: 22D wins ≥ 3/6 folds → DROP body_position (update feature set code).
      22D wins < 3/6 → KEEP (redundancy may be acting as implicit weighting).

Comparison baseline: Config B from Task 6.2 (returns_candle 23D full).
These BSS values are read from results/phase6/bss_comparison_candle_vs_baseline.tsv
rather than re-run, to ensure an exact apples-to-apples comparison.

The 22D config zeroes body_position via feature_weights — the HNSW index still
has 23 dimensions but the 3 body_position columns contribute zero distance.

Locked settings (CLAUDE.md + Task 6.1/6.2 results):
    feature_set=returns_candle, max_distance=2.5 (WINNER from Task 6.1),
    calibration=beta_abm, regime=HOLD SPY_THRESHOLD=+0.05, nn_jobs=1

Prerequisite: Tasks 6.1 and 6.2 must be complete, and 6.2 GATE = PASS.

Provenance: HANDOFF_Phase6-remainder_Phase7.md, Task 6.3

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase6_redundancy_test.py

Output:
    results/phase6/redundancy_body_position.tsv
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
DATA_DIR        = REPO_ROOT / "data" / "52t_features"
RESULTS_DIR     = REPO_ROOT / "results" / "phase6"
COMPARISON_TSV  = RESULTS_DIR / "bss_comparison_candle_vs_baseline.tsv"
SWEEP_TSV       = RESULTS_DIR / "sweep_max_distance_23d.tsv"
OUTPUT_TSV      = RESULTS_DIR / "redundancy_body_position.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature set ───────────────────────────────────────────────────────────────
RETURNS_CANDLE_COLS: list[str] = FeatureRegistry.get("returns_candle").columns  # 23 cols

# ── body_position columns to zero (one per timeframe) ────────────────────────
BODY_POSITION_COLS = [
    "candle_1d_body_position",
    "candle_3d_body_position",
    "candle_5d_body_position",
]

# ── Locked settings ───────────────────────────────────────────────────────────
SPY_THRESHOLD = 0.05
HORIZON       = "fwd_7d_up"
AVGK_GATE     = 20

# ── Gate threshold ────────────────────────────────────────────────────────────
DROP_THRESHOLD = 3   # 22D wins ≥ 3/6 folds → DROP body_position


# ── Pure logic (testable without I/O) ────────────────────────────────────────

def _evaluate_redundancy_gate(
    bss_23d: list[float],
    bss_22d: list[float],
) -> str:
    """Classify the 22D vs 23D redundancy test outcome.

    Args:
        bss_23d: Per-fold BSS for full 23D config (returns_candle, body_position kept).
        bss_22d: Per-fold BSS for 22D config (body_position zeroed).

    Returns:
        "DROP"  — 22D wins ≥ DROP_THRESHOLD folds (body_position is redundant)
        "KEEP"  — 22D wins < DROP_THRESHOLD folds (body_position contributes signal)
    """
    wins_22d = sum(1 for a, b in zip(bss_23d, bss_22d) if b > a)
    return "DROP" if wins_22d >= DROP_THRESHOLD else "KEEP"


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


def _load_winner_from_sweep(sweep_tsv: Path) -> float:
    """Read WINNER max_distance from the Task 6.1 sweep TSV."""
    df = pd.read_csv(sweep_tsv, sep="\t")
    for d in sorted(df["max_distance"].unique()):
        sub = df[df["max_distance"] == d]
        if (sub["AvgK"] >= AVGK_GATE).all():
            return float(d)
    raise RuntimeError(
        f"No WINNER found in {sweep_tsv}. "
        f"No max_distance has AvgK ≥ {AVGK_GATE} on all 6 folds."
    )


def _load_23d_bss(comparison_tsv: Path) -> list[float]:
    """Read per-fold BSS for Config B (returns_candle 23D) from Task 6.2 TSV."""
    df = pd.read_csv(comparison_tsv, sep="\t")
    config_b = (
        df[df["config"] == "returns_candle"]
        .sort_values("fold")
        .reset_index(drop=True)
    )
    if len(config_b) != len(WALKFORWARD_FOLDS):
        raise RuntimeError(
            f"Expected {len(WALKFORWARD_FOLDS)} rows for returns_candle in "
            f"{comparison_tsv}, got {len(config_b)}."
        )
    return config_b["BSS"].tolist()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total = time.time()

    print("=" * 68)
    print("  Task 6.3 — body_position redundancy test (22D vs 23D)")
    print("=" * 68)
    print(f"  Zeroing : {BODY_POSITION_COLS}")
    print(f"  Gate    : 22D wins ≥ {DROP_THRESHOLD}/6 folds → DROP body_position")

    # ── Prerequisites ──────────────────────────────────────────────────────────
    for path, label in [(SWEEP_TSV, "Task 6.1 sweep TSV"),
                        (COMPARISON_TSV, "Task 6.2 comparison TSV")]:
        if not path.exists():
            print(f"\nERROR: {label} not found: {path}")
            sys.exit(1)

    winner = _load_winner_from_sweep(SWEEP_TSV)
    bss_23d_folds = _load_23d_bss(COMPARISON_TSV)

    print(f"  max_distance (WINNER from 6.1): {winner}")
    print(f"  23D BSS reference (from 6.2) : {[f'{v:+.5f}' for v in bss_23d_folds]}")

    # ── Load + augment data ────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"
    if not t_path.exists() or not v_path.exists():
        print(f"\nERROR: 52T features data not found in {DATA_DIR}")
        sys.exit(1)

    print(f"\nLoading 52T features data...")
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers")
    full_db = _augment_with_candlestick(full_db)

    # ── Build feature_weights: zero the 3 body_position columns ───────────────
    redundancy_weights = {col: 0.0 for col in BODY_POSITION_COLS}
    print(f"\nFeature weights applied: {redundancy_weights}")

    # ── Run 6-fold walk-forward (22D) ──────────────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  22D walk-forward (body_position zeroed), max_distance={winner}")
    print(f"{'─'*68}")

    original_calibrator = _matcher_module._PlattCalibrator
    _matcher_module._PlattCalibrator = _BetaCalibrator

    rows_22d: list[dict] = []

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
                rows_22d.append({
                    "fold": fi, "fold_label": fold["label"],
                    "BSS_22d": float("nan"), "BSS_23d": bss_23d_folds[fi],
                    "AvgK": 0.0, "n_scored": 0, "n_hold": 0,
                })
                continue

            regime_labeler = _ThresholdRegimeLabeler(threshold=SPY_THRESHOLD)
            regime_labeler.fit(train_db)
            val_regime = regime_labeler.label(val_db)
            bear_mask  = (val_regime == 0)
            y_true     = val_db[HORIZON].values.astype(float)
            base_rate  = float(y_true.mean())

            cfg = EngineConfig()
            cfg.max_distance         = winner
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
            cfg.nn_jobs              = 1     # MUST stay 1 — Windows/Py3.12 deadlock
            cfg.feature_set          = "returns_candle"
            cfg.feature_weights      = redundancy_weights
            cfg.use_sax_filter       = False
            cfg.use_wfa_rerank       = False
            cfg.use_ib_compression   = False
            cfg.use_sector_conviction  = False
            cfg.use_momentum_filter    = False
            cfg.use_sentiment_veto     = False

            print(f"  [{fi+1}/6] {fold['label']}  train={len(train_db):,}  "
                  f"val={len(val_db):,}", end=" ", flush=True)
            t0 = time.time()
            matcher = PatternMatcher(cfg)
            matcher.fit(train_db, RETURNS_CANDLE_COLS)
            probs_raw, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
            elapsed = time.time() - t0

            probs = np.asarray(probs_raw)
            sigs  = np.array(signals)
            avg_k = float(np.mean(n_matches))

            probs_hold = probs.copy()
            probs_hold[bear_mask] = base_rate
            bss_val = _bss(probs_hold, y_true)

            matcher_hold = int(np.sum(sigs == "HOLD"))
            regime_hold  = int(np.sum(bear_mask & (sigs != "HOLD")))
            n_hold_total = matcher_hold + regime_hold
            n_scored     = len(sigs) - n_hold_total

            bss_ref = bss_23d_folds[fi]
            delta   = bss_val - bss_ref
            winner_flag = "22D" if bss_val > bss_ref else "23D"

            print(f"→ BSS_22d={bss_val:+.5f}  BSS_23d={bss_ref:+.5f}  "
                  f"Δ={delta:+.5f}  winner={winner_flag}  t={elapsed:.0f}s")

            rows_22d.append({
                "fold":       fi,
                "fold_label": fold["label"],
                "BSS_22d":    round(bss_val, 6),
                "BSS_23d":    round(bss_ref, 6),
                "bss_delta":  round(delta, 6),
                "AvgK":       round(avg_k, 3),
                "n_scored":   n_scored,
                "n_hold":     n_hold_total,
            })

    finally:
        _matcher_module._PlattCalibrator = original_calibrator

    # ── Write TSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows_22d).sort_values("fold").reset_index(drop=True)
    df_out.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.6f")
    print(f"\nSaved: {OUTPUT_TSV}")

    # ── Gate evaluation ────────────────────────────────────────────────────────
    bss_22d_folds = [r["BSS_22d"] for r in sorted(rows_22d, key=lambda x: x["fold"])]
    gate = _evaluate_redundancy_gate(bss_23d_folds, bss_22d_folds)
    wins_22d = sum(1 for a, b in zip(bss_23d_folds, bss_22d_folds) if b > a)

    print()
    print("=" * 68)
    print("  GATE EVALUATION")
    print("=" * 68)
    print(f"{'Fold':<14}  {'BSS_23d':>9}  {'BSS_22d':>9}  {'Delta':>9}  Winner")
    print("-" * 68)
    for r in sorted(rows_22d, key=lambda x: x["fold"]):
        w = "22D" if r["BSS_22d"] > r["BSS_23d"] else "23D"
        print(f"{r['fold_label']:<14}  {r['BSS_23d']:>+9.5f}  "
              f"{r['BSS_22d']:>+9.5f}  {r['bss_delta']:>+9.5f}  {w}")
    print("-" * 68)
    print(f"  22D wins: {wins_22d}/6 folds → {gate} body_position")

    total = time.time() - t_total
    print(f"\nTotal runtime: {total:.0f}s")

    if gate == "DROP":
        print(f"\n[{gate}] 22D wins {wins_22d}/6 folds.")
        print(f"  body_position appears redundant. Remove candle_*_body_position")
        print(f"  from the returns_candle feature set in pattern_engine/features.py.")
        print(f"  This is a SEPARATE code change (not part of this script).")
        print(f"  After the code change, re-run Task 6.2 to confirm BSS is maintained.")
    else:
        print(f"\n[{gate}] 22D wins only {wins_22d}/6 folds.")
        print(f"  body_position contributes signal (or acts as implicit feature weighting).")
        print(f"  Retain all 23 columns. No code change required.")

    sys.exit(0)


if __name__ == "__main__":
    main()
