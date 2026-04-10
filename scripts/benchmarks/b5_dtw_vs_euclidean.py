"""
scripts/benchmarks/b5_dtw_vs_euclidean.py — Competitive Benchmark B5.
Three configs on 2024-Val fold only (reference benchmark, not a gate).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pandas as pd
from pattern_engine.config import WALKFORWARD_FOLDS
from pattern_engine.features import FeatureRegistry
from phase7_baseline import run_fold_with_config, DATA_DIR, _augment_with_candlestick

FEATURE_COLS = list(FeatureRegistry.get("returns_candle").columns)

fold_2024 = next(f for f in WALKFORWARD_FOLDS if f["label"] == "2024-Val")

configs = [
    ("A: KNN (no cal)",         {"calibration_method": "none", "use_dtw_reranker": False}),
    ("B: KNN + beta_abm",       {"calibration_method": "beta_abm", "use_dtw_reranker": False}),
    ("C: KNN + beta_abm + DTW", {"calibration_method": "beta_abm", "use_dtw_reranker": True, "dtw_rerank_k": 20}),
]

print("=== Benchmark B5: DTW vs Euclidean (2024-Val fold) ===\n")

print("Loading data...")
all_files = sorted(DATA_DIR.glob("*.parquet"))
dfs = [pd.read_parquet(f) for f in all_files]
full_db = pd.concat(dfs, ignore_index=True)
full_db["Date"] = pd.to_datetime(full_db["Date"])
full_db = _augment_with_candlestick(full_db)
print()

for name, overrides in configs:
    t0 = time.time()
    result = run_fold_with_config(fold_2024, full_db, FEATURE_COLS, cfg_overrides=overrides)
    elapsed = time.time() - t0
    print(f"  {name:<35} BSS={result['bss']:.5f}  n={result['n_scored']}  ({elapsed:.1f}s)")

print("\nDone.")
