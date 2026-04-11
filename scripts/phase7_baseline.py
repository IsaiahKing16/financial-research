"""
scripts/phase7_baseline.py — Task T7.0: Phase 7 baseline re-establishment.

Core logic now lives in pattern_engine.walkforward (P3 extraction).
This file re-exports symbols for backwards compatibility with E1–E4 scripts.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/phase7_baseline.py

Provenance: Phase 7 implementation plan, Task T7.0 (2026-04-09)
             P3 Optuna Infrastructure extraction (2026-04-11)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Repo root resolution ─────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Re-exports from pattern_engine.walkforward ──────────────────────────────
from pattern_engine.walkforward import (      # noqa: F401 — re-exports
    _BetaCalibrator,
    _augment_with_candlestick,
    _apply_h7_hold_regime,
    _build_cfg,
    FEATURE_COLS,
    HORIZON,
    SPY_THRESHOLD,
    run_fold as _run_fold,
)
from pattern_engine.config import WALKFORWARD_FOLDS


# ── DATA_DIR with worktree fallback ─────────────────────────────────────────
def _find_data_dir() -> Path:
    candidate = REPO_ROOT / "data" / "52t_features"
    if candidate.exists():
        return candidate
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


DATA_DIR    = _find_data_dir()
RESULTS_DIR = REPO_ROOT / "results" / "phase7"
OUTPUT_TSV  = RESULTS_DIR / "baseline_23d.tsv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Thin wrapper for backwards compatibility ─────────────────────────────────
def run_fold_with_config(
    fold: dict,
    full_db: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    cfg_overrides: dict | None = None,
) -> dict:
    """Thin wrapper — delegates to pattern_engine.walkforward.run_fold()."""
    return _run_fold(fold, full_db, feature_cols=feature_cols,
                     cfg_overrides=cfg_overrides)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t_total = time.time()
    from pattern_engine.features import FeatureRegistry

    FEATURE_SET = "returns_candle"
    feature_cols = list(FeatureRegistry.get(FEATURE_SET).columns)

    print("=" * 72)
    print("  Task T7.0 — Phase 7 Baseline: returns_candle (23D)")
    print("=" * 72)
    print(f"  feature_set   : {FEATURE_SET} ({len(feature_cols)} columns)")
    print(f"  max_distance  : 2.5")
    print(f"  calibration   : beta_abm (H5 locked)")
    print(f"  regime        : H7 HOLD (SPY ret_90d < +{SPY_THRESHOLD:.2f} → base_rate)")
    print(f"  horizon       : {HORIZON}")
    print(f"  data          : {DATA_DIR}")
    print()

    t_path = DATA_DIR / "train_db.parquet"
    v_path = DATA_DIR / "val_db.parquet"

    if not t_path.exists() or not v_path.exists():
        print(f"ERROR: 52T features data not found in {DATA_DIR}")
        sys.exit(1)

    print(f"Loading 52T features from {DATA_DIR} ...", flush=True)
    full_db = pd.concat(
        [pd.read_parquet(t_path), pd.read_parquet(v_path)], ignore_index=True
    )
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    print(f"  {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers, "
          f"date range {full_db['Date'].min().date()} – {full_db['Date'].max().date()}")

    full_db = _augment_with_candlestick(full_db)

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

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_TSV, sep="\t", index=False, float_format="%.8f")
    print(f"\nSaved: {OUTPUT_TSV}")

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
        print(f"[WARN] Only {pos_folds}/6 positive folds — baseline may differ.")


if __name__ == "__main__":
    main()
