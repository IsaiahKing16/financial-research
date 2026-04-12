"""
scripts/smoke_test_candle_wf.py — Phase 6 candlestick smoke test.

Verifies that the returns_candle feature set (8 VOL_NORM + 15 candlestick = 23 cols)
completes 6 walk-forward folds without crash on the 52T dataset.

NOT gated on BSS — the goal is crash-free execution and a NaN audit.
BSS comparison vs. baseline will happen in a dedicated experiment.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/smoke_test_candle_wf.py

Data: data/52t_features/  (has OHLC + Ticker + VOL_NORM cols + labels)
Output: prints fold results + NaN summary to stdout. No files written.

NaN thresholds (from session log):
    >5% NaN after imputation in any candle feature = feature pollution risk,
    investigate data gaps before proceeding to BSS experiment.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import VOL_NORM_COLS, FeatureRegistry
from pattern_engine.candlestick import compute_candlestick_features, CANDLE_COLS

# ── Feature set ──────────────────────────────────────────────────────────────

RETURNS_CANDLE_COLS: list[str] = FeatureRegistry.get("returns_candle").columns  # 23 cols

# ── Data ─────────────────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "52t_features"

# ── Locked config (mirrors run_walkforward.py — do not change) ────────────────

@dataclass
class WalkForwardConfig:
    top_k: int = 50
    max_distance: float = 1.1019
    distance_weighting: str = "uniform"
    feature_weights: dict = field(default_factory=dict)
    batch_size: int = 256
    confidence_threshold: float = 0.65
    agreement_spread: float = 0.05
    min_matches: int = 5
    exclude_same_ticker: bool = True
    same_sector_only: bool = False
    regime_filter: bool = False
    regime_fallback: bool = False
    projection_horizon: str = "fwd_7d_up"
    calibration_method: str = "platt"
    cal_max_samples: int = 100_000
    use_hnsw: bool = True
    use_sax_filter: bool = False
    use_wfa_rerank: bool = False
    use_ib_compression: bool = False
    journal_top_n: int = 0
    use_sector_conviction: bool = False
    use_momentum_filter: bool = False
    use_sentiment_veto: bool = False
    sector_conviction_lift: float = 0.005
    momentum_min_outperformance: float = 0.015


FOLDS = [
    {"label": "2019",       "train_end": "2018-12-31", "val_start": "2019-01-01", "val_end": "2019-12-31"},
    {"label": "2020-COVID", "train_end": "2019-12-31", "val_start": "2020-01-01", "val_end": "2020-12-31"},
    {"label": "2021",       "train_end": "2020-12-31", "val_start": "2021-01-01", "val_end": "2021-12-31"},
    {"label": "2022-Bear",  "train_end": "2021-12-31", "val_start": "2022-01-01", "val_end": "2022-12-31"},
    {"label": "2023",       "train_end": "2022-12-31", "val_start": "2023-01-01", "val_end": "2023-12-31"},
    {"label": "2024-Val",   "train_end": "2023-12-31", "val_start": "2024-01-01", "val_end": "2024-12-31"},
]


# ── Metrics ───────────────────────────────────────────────────────────────────

def bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    brier = float(np.mean((probs - y_true) ** 2))
    brier_clim = float(np.var(y_true))
    return (1.0 - brier / brier_clim) if brier_clim > 0 else 0.0


def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((probs > 0.5).astype(int) == y_true.astype(int)))


# ── Candlestick augmentation ──────────────────────────────────────────────────

PROPORTION_COLS = [c for c in CANDLE_COLS if not c.endswith("_direction")]
DIRECTION_COLS  = [c for c in CANDLE_COLS if c.endswith("_direction")]


def _impute_candle_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaN in candlestick cols before passing to PatternMatcher.

    Proportions (body_to_range, upper_wick, lower_wick, body_position) → 0.0
    Direction → 1.0 (treat as bullish doji — neutral imputation)

    This is caller responsibility as documented in candlestick.py docstring.
    Imputation is applied on a copy; input is not mutated.
    """
    df = df.copy()
    prop_fill = {c: 0.0 for c in CANDLE_COLS if "direction" not in c}
    dir_fill  = {c: 1.0 for c in CANDLE_COLS if "direction" in c}
    df.fillna({**prop_fill, **dir_fill}, inplace=True)
    return df


def augment_with_candlestick(full_db: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute + impute candlestick features. Returns augmented df + NaN stats."""
    print("\nComputing candlestick features...")
    t0 = time.time()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candle_df = compute_candlestick_features(full_db)
        if caught:
            for w in caught:
                print(f"  [WARN] {w.message}")

    elapsed = time.time() - t0
    print(f"  Computed {len(CANDLE_COLS)} candlestick columns in {elapsed:.1f}s")

    # NaN audit before imputation
    nan_rates: dict[str, float] = {}
    for col in CANDLE_COLS:
        rate = float(candle_df[col].isna().mean())
        nan_rates[col] = rate

    overall_nan = float(candle_df.isna().values.mean())
    max_nan_col = max(nan_rates, key=nan_rates.get)
    max_nan_rate = nan_rates[max_nan_col]

    print(f"\n  NaN rate summary (before imputation):")
    print(f"    Overall: {overall_nan:.2%}")
    print(f"    Max single column: {max_nan_rate:.2%}  [{max_nan_col}]")

    if max_nan_rate > 0.05:
        print(f"  [WARNING] Max NaN rate {max_nan_rate:.2%} > 5% threshold.")
        print(f"            Investigate data gaps before BSS experiment.")

    # Merge candlestick into full_db
    augmented = pd.concat([full_db.reset_index(drop=True),
                           candle_df.reset_index(drop=True)], axis=1)

    # Impute NaNs in candle cols
    augmented = _impute_candle_nans(augmented)
    remaining_nan = augmented[CANDLE_COLS].isna().values.sum()
    if remaining_nan > 0:
        raise RuntimeError(
            f"Imputation failed: {remaining_nan} NaNs remain in candlestick cols after fillna."
        )
    print(f"  Imputation complete. 0 NaNs remain in candlestick cols.")

    return augmented, nan_rates


# ── Fold runner ───────────────────────────────────────────────────────────────

def run_fold(full_db: pd.DataFrame, fold: dict, cfg: WalkForwardConfig) -> dict:
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].copy()
    val_db   = full_db[(full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)].copy()

    train_db = train_db.dropna(subset=[cfg.projection_horizon])
    val_db   = val_db.dropna(subset=[cfg.projection_horizon])

    if len(val_db) == 0:
        return {"label": fold["label"], "bss": float("nan"), "accuracy": float("nan"),
                "n_train": len(train_db), "n_val": 0, "avg_k": 0.0,
                "buy": 0, "sell": 0, "hold": 0, "crashed": False}

    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, RETURNS_CANDLE_COLS)
    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    probs_arr = np.asarray(probs)
    y_true    = val_db[cfg.projection_horizon].values.astype(np.float64)

    return {
        "label":    fold["label"],
        "bss":      bss(probs_arr, y_true),
        "accuracy": accuracy(probs_arr, y_true),
        "n_train":  len(train_db),
        "n_val":    len(val_db),
        "avg_k":    float(np.mean(n_matches)),
        "buy":      int(np.sum(np.array(signals) == "BUY")),
        "sell":     int(np.sum(np.array(signals) == "SELL")),
        "hold":     int(np.sum(np.array(signals) == "HOLD")),
        "crashed":  False,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0_total = time.time()

    print("=" * 70)
    print("  Phase 6 Candlestick Smoke Test — returns_candle (23 features)")
    print("=" * 70)
    print(f"\nFeature set: {len(RETURNS_CANDLE_COLS)} columns")
    print(f"  VOL_NORM (8):    {VOL_NORM_COLS}")
    print(f"  CANDLE (15):     {CANDLE_COLS[:5]}... (first 5 shown)")
    print(f"\nGate: all 6 folds complete without crash")
    print(f"      NaN rate < 5% per candlestick column after imputation")

    # ── Load data ──────────────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"
    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        print("       Run scripts/build_52t_features.py first.")
        sys.exit(1)

    print(f"\nLoading 52T features data from {DATA_DIR}...")
    train_raw = pd.read_parquet(t_path)
    val_raw   = pd.read_parquet(v_path)
    full_db   = pd.concat([train_raw, val_raw], ignore_index=True)
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"{full_db['Date'].min().date()} → {full_db['Date'].max().date()}")

    # ── Verify OHLC columns present ────────────────────────────────────────────
    missing_ohlc = [c for c in ("Ticker", "Open", "High", "Low", "Close") if c not in full_db.columns]
    if missing_ohlc:
        print(f"ERROR: Missing OHLC columns: {missing_ohlc}")
        sys.exit(1)

    # ── Augment with candlestick features ──────────────────────────────────────
    try:
        full_db, nan_rates = augment_with_candlestick(full_db)
    except Exception as exc:
        print(f"\n[CRASH] Candlestick computation failed: {exc}")
        sys.exit(1)

    # ── Run 6 folds ────────────────────────────────────────────────────────────
    cfg = WalkForwardConfig()
    results = []

    print(f"\n{'Fold':<14} {'Train':>8} {'Val':>8} {'BSS':>9} {'Acc':>7} {'AvgK':>6} "
          f"{'BUY':>5} {'SELL':>5} {'HOLD':>6} {'Status':>8}")
    print("-" * 82)

    all_passed = True
    for i, fold in enumerate(FOLDS, 1):
        print(f"  [{i}/6] fitting {fold['label']}...", end=" ", flush=True)
        t_fold = time.time()
        try:
            r = run_fold(full_db, fold, cfg)
        except Exception as exc:
            elapsed = time.time() - t_fold
            print(f"CRASHED ({elapsed:.0f}s)")
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            r = {
                "label": fold["label"], "bss": float("nan"),
                "accuracy": float("nan"), "n_train": 0, "n_val": 0,
                "avg_k": 0.0, "buy": 0, "sell": 0, "hold": 0, "crashed": True,
            }
            all_passed = False

        results.append(r)
        elapsed = time.time() - t_fold
        if not r["crashed"]:
            bss_s = f"{r['bss']:+.5f}" if not np.isnan(r['bss']) else "   N/A "
            acc_s = f"{r['accuracy']:.1%}" if not np.isnan(r['accuracy']) else "  N/A "
            print(f"done ({elapsed:.0f}s)")
            print(f"{r['label']:<14} {r['n_train']:>8,} {r['n_val']:>8,} "
                  f"{bss_s:>9} {acc_s:>7} {r['avg_k']:>6.1f} "
                  f"{r['buy']:>5} {r['sell']:>5} {r['hold']:>6} {'OK':>8}")

    # ── Summary ────────────────────────────────────────────────────────────────
    total = time.time() - t0_total
    n_crashed = sum(1 for r in results if r["crashed"])
    n_complete = len(results) - n_crashed
    pos_bss = sum(1 for r in results if not r["crashed"] and r["bss"] > 0)
    mean_bss_vals = [r["bss"] for r in results if not r["crashed"] and not np.isnan(r["bss"])]
    mean_bss = float(np.mean(mean_bss_vals)) if mean_bss_vals else float("nan")

    print("-" * 82)
    print(f"\nFolds completed:   {n_complete}/6")
    print(f"Folds crashed:     {n_crashed}/6")
    print(f"Positive-BSS folds: {pos_bss}/6  (informational, not gated)")
    if not np.isnan(mean_bss):
        print(f"Mean BSS:          {mean_bss:+.5f}  (informational, not gated)")
    print(f"Total runtime:     {total:.0f}s")

    # NaN rate table for top offenders
    sorted_nans = sorted(nan_rates.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nNaN rate by column (top 5, before imputation):")
    for col, rate in sorted_nans[:5]:
        flag = " [WARNING >5%]" if rate > 0.05 else ""
        print(f"  {col:<35} {rate:.2%}{flag}")

    # ── Gate verdict ───────────────────────────────────────────────────────────
    nan_gate = all(r <= 0.05 for r in nan_rates.values())
    crash_gate = all_passed

    print(f"\n{'='*70}")
    print(f"SMOKE TEST GATE:")
    print(f"  [{'PASS' if crash_gate else 'FAIL'}] All 6 folds complete without crash: {n_complete}/6")
    print(f"  [{'PASS' if nan_gate else 'FAIL'}] All candlestick NaN rates < 5%:  "
          f"max={max(nan_rates.values()):.2%}")
    overall = crash_gate and nan_gate
    print(f"\n  RESULT: {'SMOKE TEST PASS' if overall else 'SMOKE TEST FAIL'}")
    print(f"{'='*70}")

    if overall:
        print("\nNext: Run BSS comparison experiment with returns_candle vs. returns_only")
        print("  scripts/feature_set_comparison.py  (or add --feature_set flag to run_walkforward.py)")
    else:
        print("\nNext: Investigate failures above before proceeding.")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
