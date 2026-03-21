"""
feature_set_comparison.py — Walk-forward comparison of feature sets.

Runs WalkForwardRunner across returns_only, returns_overnight, and returns_session.
All locked settings are preserved (nn_jobs=1, top_k=50, max_distance=1.1019).
Results logged to data/results/experiments.tsv via ExperimentLogger.

Usage:
    python scripts/feature_set_comparison.py

Success gate: any feature set with mean BSS improvement >= 0.02 over returns_only.
"""

import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.config import EngineConfig
from pattern_engine.data import DataLoader
from pattern_engine.experiment_logging import ExperimentLogger
from pattern_engine.walkforward import WalkForwardRunner

# ---------------------------------------------------------------------------
# Load full DB (same path handling as bss_regression_test.py)
# ---------------------------------------------------------------------------
data_dir = project_root / "data"
raw_dir = data_dir / "raw"

if not raw_dir.exists() and list(data_dir.glob("*.csv")):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in data_dir.glob("*.csv"):
        dest = raw_dir / csv_file.name
        if not dest.exists():
            shutil.copy2(csv_file, dest)
    print(f"  Copied {len(list(raw_dir.glob('*.csv')))} CSVs to data/raw/")

print("Loading data...")
loader = DataLoader(data_dir=str(data_dir))
raw_data = loader.download(force_refresh=False)
full_db = loader.compute_features(raw_data)
print(f"  Full DB: {len(full_db)} rows, {len(full_db.columns)} columns")

# ---------------------------------------------------------------------------
# Run walk-forward for each feature set
# ---------------------------------------------------------------------------
results_dir = data_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)
logger = ExperimentLogger(results_dir=str(results_dir))

feature_sets = ["returns_only", "returns_overnight", "returns_session"]
bss_by_fs: dict[str, list[float]] = {}

for fs in feature_sets:
    print(f"\n{'=' * 60}")
    print(f"  Feature set: {fs}")
    print(f"{'=' * 60}")

    cfg = EngineConfig(
        feature_set=fs,
        nn_jobs=1,                   # LOCKED — Windows/Py3.12 joblib deadlock
        top_k=50,                    # LOCKED
        max_distance=1.1019,         # LOCKED
        confidence_threshold=0.65,   # LOCKED
    )

    runner = WalkForwardRunner(config=cfg, logger=logger)
    fold_metrics = runner.run(full_db, experiment_name=f"feature_set_{fs}", verbose=1)

    bss_scores = [m["brier_skill_score"] for m in fold_metrics if "brier_skill_score" in m]
    bss_by_fs[fs] = bss_scores

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  FEATURE SET COMPARISON — SUMMARY")
print("=" * 60)
print(f"  {'Feature Set':<25} {'Mean BSS':>10} {'2024 BSS':>10}")
print(f"  {'-' * 50}")

baseline_mean = None
for fs, scores in bss_by_fs.items():
    mean_bss = sum(scores) / len(scores) if scores else float("nan")
    last_bss = scores[-1] if scores else float("nan")
    if fs == "returns_only":
        baseline_mean = mean_bss
    print(f"  {fs:<25} {mean_bss:>+10.5f} {last_bss:>+10.5f}")

print(f"\n  Baseline (returns_only): mean BSS = {baseline_mean:+.5f}")
print(f"  Promotion gate: any set with mean BSS >= {(baseline_mean or 0) + 0.02:+.5f}")

if baseline_mean is not None:
    for fs, scores in bss_by_fs.items():
        if fs == "returns_only":
            continue
        mean_bss = sum(scores) / len(scores) if scores else float("nan")
        delta = mean_bss - baseline_mean
        cleared = delta >= 0.02
        print(f"  {fs}: delta={delta:+.5f} → gate {'CLEARED' if cleared else 'not cleared'}")

print(f"\n  Full results logged → {results_dir / 'experiments.tsv'}")
