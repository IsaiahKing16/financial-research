"""
sweep_max_distance.py — Sweep max_distance on Fold 6 (2024-Val) only.

max_distance=1.1019 was locked for the 52-ticker universe where it produced
AvgK≈39-42. With 585 tickers the index is denser; the same threshold admits
AvgK≈49, diluting signal with low-quality analogues.

Performance optimisation: max_distance is a Stage-4 post-filter — it has no
effect on index construction or neighbour fetching (HNSW always fetches
top_k*3=150 candidates). So we fit the matcher ONCE and mutate
config.max_distance between val queries. Sweep time: ~16 min instead of ~105 min.

Caveat: the Platt calibrator is fitted once at the original max_distance.
BSS values at other thresholds have slight miscalibration; the correct workflow
is to use this sweep to identify the winning threshold, then run a full
fit+query confirmation with that threshold.

Usage:
    python scripts/sweep_max_distance.py

Outputs a TSV to results/max_distance_sweep_YYYY-MM-DD.tsv.
New evidence required before updating the locked setting in CLAUDE.md.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import VOL_NORM_COLS

DATA_DIR    = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# M9: use vol-normalized features — same as run_walkforward.py.
# Walk-forward results 2026-03-24: vol-norm BSS=-0.00031 vs raw returns BSS=-0.00073
# on fold 6. Max_distance sweep must use the same feature space as the walk-forward.
FEATURE_COLS = VOL_NORM_COLS  # ret_Xd / rolling_std_Xd (8 features)

FOLD6 = {
    "label":     "2024-Val",
    "train_end": "2023-12-31",
    "val_start": "2024-01-01",
    "val_end":   "2024-12-31",
}

# Sweep: tight → current locked value
# Target: restore AvgK to 30–45 (the range seen in the 52-ticker baseline)
SWEEP_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                0.90, 0.95, 1.00, 1.05, 1.10, 1.1019]


@dataclass
class SweepConfig:
    top_k: int = 50
    max_distance: float = 1.1019          # mutated per sweep iteration
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


def bss(probs: np.ndarray, y_true: np.ndarray) -> float:
    brier      = float(np.mean((probs - y_true) ** 2))
    brier_clim = float(np.var(y_true))
    return (1.0 - brier / brier_clim) if brier_clim > 0 else 0.0


def directional_accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((probs > 0.5).astype(int) == y_true.astype(int)))


def main() -> None:
    print("=" * 72)
    print("  max_distance sweep — Fold 6 (2024-Val), 585-ticker universe")
    print("=" * 72)
    print(f"  Locked baseline : max_distance=1.1019  AvgK=41.9 (52-ticker)")
    print(f"  Observed (585T) : max_distance=1.1019  AvgK=48.9 → too many neighbours")
    print(f"  Target AvgK     : 30–45  (restore discriminative pruning)")
    print(f"  Sweep values    : {len(SWEEP_VALUES)} thresholds, fit index ONCE\n")

    # ── Load data ────────────────────────────────────────────────────────────
    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"
    if not t_path.exists():
        raise FileNotFoundError(f"train_db not found in {DATA_DIR}. Run prepare.py first.")

    full_db = pd.concat([pd.read_parquet(t_path), pd.read_parquet(v_path)],
                        ignore_index=True)
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  Loaded {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers\n")

    train_db = full_db[full_db["Date"] <= pd.Timestamp(FOLD6["train_end"])].dropna(
        subset=["fwd_7d_up"]).copy()
    val_db   = full_db[
        (full_db["Date"] >= pd.Timestamp(FOLD6["val_start"])) &
        (full_db["Date"] <= pd.Timestamp(FOLD6["val_end"]))
    ].dropna(subset=["fwd_7d_up"]).copy()
    y_true   = val_db["fwd_7d_up"].values.astype(np.float64)

    print(f"  Fold 6: train={len(train_db):,} rows, val={len(val_db):,} rows")

    # ── Fit index ONCE at default max_distance ───────────────────────────────
    cfg = SweepConfig()          # max_distance=1.1019 for calibration fit
    print(f"\n  Fitting HNSW index on {len(train_db):,} training rows "
          f"(cal_max_samples={cfg.cal_max_samples:,})...")
    t_fit = time.time()
    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, FEATURE_COLS)
    print(f"  Fit done in {time.time() - t_fit:.0f}s\n")

    # ── Sweep thresholds ─────────────────────────────────────────────────────
    print(f"{'max_d':>7}  {'BSS':>9}  {'Acc':>7}  {'AvgK':>6}  "
          f"{'%HOLD':>6}  {'BUY':>5}  {'SELL':>5}  {'t(s)':>5}")
    print("-" * 72)

    rows     = []
    best_bss = -np.inf
    best_d   = None

    for d in SWEEP_VALUES:
        matcher.config.max_distance = d           # mutate — no refit needed
        t0 = time.time()
        probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
        elapsed = time.time() - t0

        probs_arr = np.asarray(probs)
        sigs      = np.array(signals)

        r = {
            "max_distance": d,
            "bss":      bss(probs_arr, y_true),
            "accuracy": directional_accuracy(probs_arr, y_true),
            "avg_k":    float(np.mean(n_matches)),
            "pct_hold": float(np.mean(sigs == "HOLD")),
            "n_buy":    int(np.sum(sigs == "BUY")),
            "n_sell":   int(np.sum(sigs == "SELL")),
            "elapsed":  elapsed,
        }
        rows.append(r)

        if r["bss"] > best_bss:
            best_bss = r["bss"]
            best_d   = d

        flag = " ★" if r["bss"] > 0 else ""
        avk_flag = " ⚠" if r["avg_k"] > 47 else ""
        print(f"{d:>7.4f}  {r['bss']:>+9.5f}  {r['accuracy']:>7.1%}  "
              f"{r['avg_k']:>6.1f}{avk_flag}  {r['pct_hold']:>6.1%}  "
              f"{r['n_buy']:>5}  {r['n_sell']:>5}  {r['elapsed']:>5.0f}{flag}")

    print("-" * 72)
    print(f"\nBest BSS : {best_bss:+.5f}  at max_distance={best_d}")
    print(f"Baseline : BSS=+0.00103 at max_distance=1.1019 (52-ticker fold 6)")

    # ── Save TSV ─────────────────────────────────────────────────────────────
    tsv_path = RESULTS_DIR / f"max_distance_sweep_{date.today()}.tsv"
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False, float_format="%.6f")
    print(f"\nSaved  : {tsv_path}")

    print("\nNext step: if best_d differs from 1.1019, run a FULL fit+query")
    print("confirmation at that threshold (refits calibrator cleanly), then")
    print("update CLAUDE.md locked settings with the TSV as provenance.")


if __name__ == "__main__":
    main()
