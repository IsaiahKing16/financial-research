"""
scripts/experiments/h2_sector_filter_sweep.py

Hypothesis H2: Sector filtering — hard filter vs soft prior boost.

Run only if H1 failed the gate. Check murphy_gate.json to confirm.

Tests:
  - same_sector_only=True (hard filter: strict domain restriction)
  - same_sector_boost_factor in [1.5, 2.0, 3.0] (soft prior: sector-weighted distance)

Both combined with:
  - distance_weighting='inverse' (best from H1 or default)
  - max_distance from best H1 run (read from TSV) or 0.80 fallback

Requires matcher.py to support same_sector_boost_factor (added in Phase 1 setup).
Safety check: warns if mean_avg_k < 5 with same_sector_only=True (pool too thin).

Output: results/bss_fix_sweep_h2.tsv
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

# Import shared helpers from H1
from scripts.experiments.h1_max_distance_sweep import (
    run_walkforward,
    _load_full_db,
    _get_best_h1_max_distance,
    GATE_THRESHOLD_POSITIVE_FOLDS,
)


SECTOR_BOOST_VALUES = [1.5, 2.0, 3.0]


def main() -> None:
    gate = MurphyGate.load_and_enforce(caller="H2")

    start = time.time()
    print("\n" + "=" * 70)
    print("  H2: SECTOR FILTER SWEEP (hard filter + soft prior)")
    print("=" * 70)

    best_max_d = _get_best_h1_max_distance()

    print(f"\n  Loading data...", end="", flush=True)
    full_db = _load_full_db()
    print(f" {len(full_db):,} rows, {full_db['Ticker'].nunique()} tickers")

    # Pre-load any rows already on disk so the final rewrite is complete
    all_results = []
    configs_to_test = [
        {"label": "hard_filter",  "same_sector_only": True,  "boost": 1.0},
        {"label": "boost_1.5x",   "same_sector_only": False, "boost": 1.5},
        {"label": "boost_2.0x",   "same_sector_only": False, "boost": 2.0},
        {"label": "boost_3.0x",   "same_sector_only": False, "boost": 3.0},
    ]

    # Pre-define column order and write TSV header before loop (incremental safety)
    _fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    _tsv_cols = (
        ["mode", "same_sector_only", "sector_boost", "max_distance",
         "distance_weighting", "mean_bss", "positive_folds", "gate_met", "mean_avg_k"]
        + [f"bss_{lbl}" for lbl in _fold_labels]
    )
    results_path = project_root / "results" / "bss_fix_sweep_h2.tsv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        "# bss_fix_sweep_h2.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        f"# Hypothesis H2: sector filter sweep (hard filter + soft prior boost)",
        f"# Base config: max_distance={best_max_d}, distance_weighting=inverse",
        f"# Gate: positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        f"# Rows appended incrementally — file is valid even if run is interrupted.",
        "#",
    ]

    # Resume support: detect already-completed configs from existing TSV
    _completed_labels: set[str] = set()
    if results_path.exists():
        try:
            _existing = pd.read_csv(results_path, comment="#", sep="\t")
            if "mode" in _existing.columns and len(_existing) > 0:
                _completed_labels = set(_existing["mode"].tolist())
                # Pre-load completed rows into all_results for final rewrite
                for _, _row in _existing.iterrows():
                    all_results.append(_row.to_dict())
                print(f"\n  [RESUME] Found {len(_completed_labels)} completed config(s) "
                      f"in existing TSV: {sorted(_completed_labels)}")
                print(f"  [RESUME] Appending to existing file — skipping completed configs.")
        except Exception:
            pass  # Corrupt/empty file — start fresh

    # Write header only if starting fresh
    if not _completed_labels:
        with open(results_path, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(_tsv_cols) + "\n")

    for cfg_spec in configs_to_test:
        label = cfg_spec["label"]

        # Skip if already completed in a previous (interrupted) run
        if label in _completed_labels:
            print(f"\n  Skipping: {label} (already in TSV)")
            continue

        print(f"\n  Running: {label} max_d={best_max_d} w=inverse ...", end="", flush=True)

        cfg = EngineConfig()
        cfg.max_distance             = best_max_d
        cfg.distance_weighting       = "inverse"
        cfg.same_sector_only         = cfg_spec["same_sector_only"]
        cfg.same_sector_boost_factor = cfg_spec["boost"]

        fold_metrics   = run_walkforward(cfg, full_db)
        bss_values     = [m["bss"] for m in fold_metrics]
        mean_bss       = float(np.mean(bss_values))
        positive_folds = sum(1 for b in bss_values if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS
        mean_k         = float(np.mean([m["avg_k"] for m in fold_metrics]))

        if cfg_spec["same_sector_only"] and mean_k < 5:
            print(f"\n  WARNING: mean_k={mean_k:.1f} < 5 with same_sector_only=True."
                  f" Pool is too thin for stable probability estimates.")

        row = {
            "mode":               label,
            "same_sector_only":   cfg_spec["same_sector_only"],
            "sector_boost":       cfg_spec["boost"],
            "max_distance":       best_max_d,
            "distance_weighting": "inverse",
            "mean_bss":           round(mean_bss, 6),
            "positive_folds":     positive_folds,
            "gate_met":           gate_met,
            "mean_avg_k":         round(mean_k, 2),
        }
        for m in fold_metrics:
            row[f"bss_{m['fold']}"] = m["bss"]
        all_results.append(row)

        # Incremental append — flush this row immediately
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in _tsv_cols) + "\n")

        print(f" mean_BSS={mean_bss:+.5f} pos_folds={positive_folds}/6 mean_k={mean_k:.1f}"
              f"{'  *** GATE ***' if gate_met else ''}")

    # Rewrite final sorted TSV
    df = pd.DataFrame(all_results, columns=_tsv_cols)
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
