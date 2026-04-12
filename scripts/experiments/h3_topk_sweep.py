"""
scripts/experiments/h3_topk_sweep.py

Hypothesis H3: Reduce top_k to match 585T pool size.

k=50 was calibrated on 52T where k/N=96%. At 585T, k/N=8.5% — a fundamentally
different classifier operating point. Optimal k scales sub-linearly with N.
k≈sqrt(N)=24 is the heuristic starting point.

k* theoretical optimum: N^(4/(d+4)) for d=8 → k* ∝ N^(1/3).
For N=585: k*≈8.36. For N=52: k*≈4.4.

Sweep: top_k=[10, 15, 20, 24, 30, 40, 50]
Base:  best max_distance from H1 (or 0.80), distance_weighting=inverse

Run only if H1 and H2 fail the gate.

Output: results/bss_fix_sweep_h3.tsv
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
from scripts.diagnostics.murphy_gate import MurphyGate
from scripts.experiments.h1_max_distance_sweep import (
    run_walkforward,
    _load_full_db,
    _get_best_h1_max_distance,
    GATE_THRESHOLD_POSITIVE_FOLDS,
)


TOP_K_VALUES = [10, 15, 20, 24, 30, 40, 50]


def main() -> None:
    gate = MurphyGate.load_and_enforce(caller="H3")

    start = time.time()
    print("\n" + "=" * 70)
    print("  H3: TOP_K SWEEP")
    print("=" * 70)
    print(f"  Sweep: top_k={TOP_K_VALUES}")
    print(f"  k=24 = sqrt(585) heuristic  |  k=50 = current baseline")

    best_max_d = _get_best_h1_max_distance()

    print(f"\n  Loading data...", end="", flush=True)
    full_db = _load_full_db()
    print(f" {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers")

    _fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    _tsv_cols = (
        ["top_k", "max_distance", "distance_weighting",
         "mean_bss", "positive_folds", "gate_met", "mean_avg_k_actual"]
        + [f"bss_{lbl}" for lbl in _fold_labels]
    )
    results_path = project_root / "results" / "bss_fix_sweep_h3.tsv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "# bss_fix_sweep_h3.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        f"# Hypothesis H3: top_k sweep {TOP_K_VALUES}",
        f"# Base config: max_distance={best_max_d}, distance_weighting=inverse",
        f"# k=24 = sqrt(585) heuristic  |  k=50 = current baseline",
        f"# Gate: positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        f"# Rows appended incrementally — file is valid even if run is interrupted.",
        "#",
    ]

    # Resume: detect already-completed top_k values from existing TSV
    all_results = []
    _completed_ks: set[int] = set()
    if results_path.exists():
        try:
            _existing = pd.read_csv(results_path, comment="#", sep="\t")
            if "top_k" in _existing.columns and len(_existing) > 0:
                _completed_ks = set(int(v) for v in _existing["top_k"].tolist())
                for _, _row in _existing.iterrows():
                    all_results.append(_row.to_dict())
                print(f"\n  [RESUME] Skipping completed top_k values: {sorted(_completed_ks)}")
        except Exception:
            pass

    if not _completed_ks:
        with open(results_path, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(_tsv_cols) + "\n")

    for k in TOP_K_VALUES:
        if k in _completed_ks:
            print(f"\n  Skipping: top_k={k} (already in TSV)")
            continue

        print(f"\n  Running: top_k={k} max_d={best_max_d} w=inverse ...", end="", flush=True)

        cfg = EngineConfig()
        cfg.top_k              = k
        cfg.max_distance       = best_max_d
        cfg.distance_weighting = "inverse"

        fold_metrics   = run_walkforward(cfg, full_db)
        bss_values     = [m["bss"] for m in fold_metrics]
        mean_bss       = float(np.mean(bss_values))
        positive_folds = sum(1 for b in bss_values if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS
        mean_k_actual  = float(np.mean([m["avg_k"] for m in fold_metrics]))

        row = {
            "top_k":               k,
            "max_distance":        best_max_d,
            "distance_weighting":  "inverse",
            "mean_bss":            round(mean_bss, 6),
            "positive_folds":      positive_folds,
            "gate_met":            gate_met,
            "mean_avg_k_actual":   round(mean_k_actual, 2),
        }
        for m in fold_metrics:
            row[f"bss_{m['fold']}"] = m["bss"]
        all_results.append(row)

        # Incremental append
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in _tsv_cols) + "\n")

        print(f" mean_BSS={mean_bss:+.5f} pos_folds={positive_folds}/6 "
              f"actual_k={mean_k_actual:.1f}{'  *** GATE ***' if gate_met else ''}")

    # Final sorted rewrite
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_bss", ascending=False)
    with open(results_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    print(f"\n  TSV written: {results_path}")
    elapsed = (time.time() - start) / 60
    print(f"  Total time: {elapsed:.1f} min")

    MurphyGate.mark_signal_fix_attempted()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
