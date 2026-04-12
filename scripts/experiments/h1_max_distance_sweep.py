"""
scripts/experiments/h1_max_distance_sweep.py

Hypothesis H1: Tighten max_distance + enable inverse-distance weighting.

Root cause: max_distance=1.1019 was calibrated on 52T. At 585T, expected NN-distance
scales as N^(-1/d). For d=8: (52/585)^(1/8) ≈ 0.79x — threshold is ~27% too permissive,
admitting noise neighbours that dilute probability estimates.

Sweep grid: max_distance=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0] x weighting=[uniform, inverse]
= 12 runs, all 6 folds each.

On completion, marks signal_fix_attempted=True in murphy_gate.json (unlocks H4).

Output: results/bss_fix_sweep_h1.tsv
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
from scripts.diagnostics.murphy_gate import MurphyGate


MAX_DISTANCE_VALUES   = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WEIGHTING_VALUES      = ["uniform", "inverse"]
BASELINE_MAX_DISTANCE = 1.1019
BASELINE_WEIGHTING    = "uniform"
GATE_THRESHOLD_POSITIVE_FOLDS = 3


# ---------------------------------------------------------------------------
# Shared helper: read best max_distance from H1 TSV
# (imported by H2, H3, H4 — keep this in H1)
# ---------------------------------------------------------------------------

def _get_best_h1_max_distance() -> float:
    """Read best max_distance from H1 TSV. Returns 0.80 if not available."""
    h1_path = project_root / "results" / "bss_fix_sweep_h1.tsv"
    if not h1_path.exists():
        print("  [H1] H1 TSV not found. Using max_distance=0.80 fallback.")
        return 0.80
    df = pd.read_csv(h1_path, comment="#", sep="\t")
    best = df.sort_values("mean_bss", ascending=False).iloc[0]
    best_d = float(best["max_distance"])
    print(f"  [H1] Best max_distance from H1 TSV: {best_d}")
    return best_d


# ---------------------------------------------------------------------------
# Shared data loader (mirrors run_walkforward.py load_full_db)
# ---------------------------------------------------------------------------

def _load_full_db() -> pd.DataFrame:
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"
    for base in (data_dir, processed_dir):
        t, v = base / "train_db.parquet", base / "val_db.parquet"
        if t.exists() and v.exists():
            combined = pd.concat([pd.read_parquet(t), pd.read_parquet(v)], ignore_index=True)
            return combined
    for fallback in (data_dir / "full_analogue_db.parquet", processed_dir / "full_db.parquet"):
        if fallback.exists():
            return pd.read_parquet(fallback)
    raise FileNotFoundError(f"No analogue database found in {data_dir}. Run prepare.py first.")


# ---------------------------------------------------------------------------
# Shared walk-forward runner
# ---------------------------------------------------------------------------

def run_walkforward(config: EngineConfig, full_db: pd.DataFrame) -> list[dict]:
    """Run 6-fold walk-forward with given config. Returns per-fold metrics."""
    full_db = full_db.copy()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    fold_results = []
    horizon = config.projection_horizon

    for fold in WALKFORWARD_FOLDS:
        train_end = pd.to_datetime(fold["train_end"])
        val_start = pd.to_datetime(fold["val_start"])
        val_end   = pd.to_datetime(fold["val_end"])

        train_db = full_db[full_db["Date"] <= train_end].dropna(subset=[horizon]).copy()
        val_db   = full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ].dropna(subset=[horizon]).copy()

        if len(train_db) == 0 or len(val_db) == 0:
            raise RuntimeError(
                f"Fold {fold['label']}: empty train ({len(train_db)}) "
                f"or val ({len(val_db)}) after dropna."
            )

        matcher = PatternMatcher(config)
        matcher.fit(train_db, VOL_NORM_COLS)
        probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

        y_true    = val_db[horizon].values.astype(float)
        base_rate = y_true.mean()
        bs_ref    = base_rate * (1 - base_rate)  # climatology Brier score
        bs_model  = float(np.mean((np.asarray(probs) - y_true) ** 2))
        bss       = 1.0 - bs_model / bs_ref if bs_ref > 1e-10 else 0.0
        avg_k     = float(np.mean(n_matches))

        # Warn if too many queries have zero matches (parameter too aggressive)
        zero_match_frac = float(np.mean(np.array(n_matches) == 0))
        if zero_match_frac > 0.10:
            print(f"    WARNING: {zero_match_frac:.1%} of queries returned 0 matches "
                  f"in fold {fold['label']}. max_distance may be too tight.")

        fold_results.append({
            "fold":      fold["label"],
            "bss":       round(bss, 6),
            "bs":        round(bs_model, 6),
            "avg_k":     round(avg_k, 2),
            "n_val":     len(val_db),
            "base_rate": round(base_rate, 4),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Phase A gate enforcement — raises RuntimeError if Murphy not run yet
    gate = MurphyGate.load_and_enforce(caller="H1")

    start = time.time()
    print("\n" + "=" * 70)
    print("  H1: MAX_DISTANCE × DISTANCE_WEIGHTING SWEEP")
    print("=" * 70)
    print(f"\n  Grid: max_distance={MAX_DISTANCE_VALUES}")
    print(f"        weighting={WEIGHTING_VALUES}")
    print(f"  Total runs: {len(MAX_DISTANCE_VALUES) * len(WEIGHTING_VALUES)} × 6 folds")

    # Load data once
    print(f"\n  Loading data...", end="", flush=True)
    full_db = _load_full_db()
    print(f" {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers")

    all_results = []
    best_mean_bss = -999.0
    best_row = None

    # --- Write TSV header once before the loop (incremental safety) ---
    # Each completed row is appended immediately so a mid-run crash loses at
    # most one in-progress config, not all prior results.
    results_path = project_root / "results" / "bss_fix_sweep_h1.tsv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-defined column order — consistent across all rows.
    _fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    _tsv_cols = (
        ["max_distance", "distance_weighting", "mean_bss", "positive_folds", "gate_met"]
        + [f"bss_{lbl}" for lbl in _fold_labels]
        + [f"avgk_{lbl}" for lbl in _fold_labels]
        + ["mean_avg_k"]
    )

    header_lines = [
        "# bss_fix_sweep_h1.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        f"# Hypothesis H1: max_distance × distance_weighting sweep",
        f"# Baseline: max_distance={BASELINE_MAX_DISTANCE}, weighting={BASELINE_WEIGHTING}",
        f"# Expected optimal: max_distance≈0.7-0.8 (N^(-1/d) scaling, d=8)",
        f"# Gate: mean_BSS > 0 AND positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        f"# Rows appended incrementally — file is valid even if run is interrupted.",
        "#",
    ]
    with open(results_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        f.write("\t".join(_tsv_cols) + "\n")

    for max_d in MAX_DISTANCE_VALUES:
        for weighting in WEIGHTING_VALUES:
            label = f"max_d={max_d:.1f} w={weighting}"
            print(f"\n  Running: {label} ...", end="", flush=True)

            cfg = EngineConfig()
            cfg.max_distance       = max_d
            cfg.distance_weighting = weighting
            # nn_jobs stays at 1 (EngineConfig default — never override)

            fold_metrics = run_walkforward(cfg, full_db)

            bss_values     = [m["bss"] for m in fold_metrics]
            mean_bss       = float(np.mean(bss_values))
            positive_folds = sum(1 for b in bss_values if b > 0)
            gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS

            row = {
                "max_distance":        max_d,
                "distance_weighting":  weighting,
                "mean_bss":            round(mean_bss, 6),
                "positive_folds":      positive_folds,
                "gate_met":            gate_met,
            }
            for m in fold_metrics:
                row[f"bss_{m['fold']}"] = m["bss"]
                row[f"avgk_{m['fold']}"] = m["avg_k"]
            row["mean_avg_k"] = round(float(np.mean([m["avg_k"] for m in fold_metrics])), 2)

            all_results.append(row)

            # --- Incremental append: flush this row to disk immediately ---
            with open(results_path, "a", encoding="utf-8") as f:
                f.write("\t".join(str(row.get(col, "")) for col in _tsv_cols) + "\n")

            print(f" mean_BSS={mean_bss:+.5f}  pos_folds={positive_folds}/6"
                  f"{'  *** GATE MET ***' if gate_met else ''}")

            if mean_bss > best_mean_bss:
                best_mean_bss = mean_bss
                best_row = row

    # --- Rewrite TSV at end: sorted by mean_bss (polished final version) ---
    df = pd.DataFrame(all_results, columns=_tsv_cols)
    df = df.sort_values("mean_bss", ascending=False)
    with open(results_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    # --- Console summary ---
    gate_met_rows = [r for r in all_results if r["gate_met"]]
    print("\n" + "=" * 70)
    print("  H1 SWEEP RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Config':<35s} {'mean_BSS':>10s} {'pos_folds':>10s} {'mean_k':>8s}")
    print(f"  {'-' * 63}")

    # Baseline row first
    print(f"  {'Baseline (1.1019, uniform)':<35s} {'N/A':>10s} {'0/6':>10s}")
    for r in sorted(all_results, key=lambda x: x["mean_bss"], reverse=True)[:8]:
        sign = "+" if r["mean_bss"] > 0 else ""
        flag = " *** GATE ***" if r["gate_met"] else ""
        cfg_label = f"max_d={r['max_distance']:.1f}, {r['distance_weighting']}"
        print(f"  {cfg_label:<35s} "
              f"{sign}{r['mean_bss']:>9.5f} {r['positive_folds']:>9}/6 "
              f"{r['mean_avg_k']:>8.1f}{flag}")

    if gate_met_rows:
        best = max(gate_met_rows, key=lambda x: x["mean_bss"])
        print(f"\n  GATE MET: max_distance={best['max_distance']}, "
              f"weighting={best['distance_weighting']}  "
              f"mean_BSS={best['mean_bss']:+.5f}  pos_folds={best['positive_folds']}/6")
    else:
        print(f"\n  GATE NOT MET by any H1 combination.")
        print(f"  Best: max_distance={best_row['max_distance']}, "
              f"weighting={best_row['distance_weighting']}  "
              f"mean_BSS={best_mean_bss:+.5f}")

    print(f"\n  TSV written: {results_path}")
    elapsed = (time.time() - start) / 60
    print(f"  Total time: {elapsed:.1f} min")

    # Unlock H4 (signal quality fix has now been attempted)
    MurphyGate.mark_signal_fix_attempted()

    print("\n  H4 (beta calibration) is now unlocked.")
    print("  Next step: Run h3_topk_sweep.py (if gate not met) OR h4_beta_calibration.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
