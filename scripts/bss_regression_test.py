"""
BSS Regression Test - Validates the rebuilt PatternEngine against System A.

Uses real market data to run 6-fold walk-forward validation with proven
defaults. The critical benchmark: 2024 fold should yield ~+0.00100 BSS
with Platt calibration (matching System A's first positive BSS).

Usage:
    python scripts/bss_regression_test.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.data import DataLoader
from pattern_engine.walkforward import WalkForwardRunner
from pattern_engine.experiment_logging import ExperimentLogger
from pattern_engine.evaluation import print_metrics


def main():
    start = time.time()

    # -- Step 1: Build database from raw CSVs --
    # Try worktree data dir first, fall back to main repo data dir
    worktree_data = project_root / "data"
    main_repo_data = project_root.parent.parent.parent / "data"

    if (worktree_data / "raw").exists():
        data_dir = str(worktree_data)
    elif main_repo_data.exists() and (main_repo_data / "SPY.csv").exists():
        # Main repo has flat data/ with CSVs directly (not in raw/ subfolder)
        # We need to set up the raw dir for DataLoader
        raw_dir = worktree_data / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Symlink or copy CSVs to raw/
        import shutil
        for csv in main_repo_data.glob("*.csv"):
            dest = raw_dir / csv.name
            if not dest.exists():
                shutil.copy2(csv, dest)
        print(f"  Copied {len(list(raw_dir.glob('*.csv')))} CSVs from main repo")
        data_dir = str(worktree_data)
    else:
        data_dir = str(worktree_data)

    print("\n" + "=" * 70)
    print("  BSS REGRESSION TEST - PatternEngine v2.0 vs System A")
    print("=" * 70)

    loader = DataLoader(data_dir=data_dir)

    print("\n[1/4] Loading raw data...")
    raw_data = loader.download(force_refresh=False)
    print(f"  Loaded {len(raw_data)} tickers")

    print("\n[2/4] Computing features...")
    full_db = loader.compute_features(raw_data)

    print(f"\n[3/4] Running 6-fold walk-forward with proven defaults...")
    print(f"  Config: Euclidean distance, max_d=1.1019, TOP_K=50")
    print(f"  Calibration: Platt (cal_frac=0.76)")
    print(f"  Regime: binary (SPY ret_90d)")
    print(f"  Features: returns_only (8 features)")

    config = EngineConfig()  # All proven defaults
    logger = ExperimentLogger(results_dir=str(worktree_data / "results"))

    runner = WalkForwardRunner(config, folds=WALKFORWARD_FOLDS, logger=logger)
    fold_metrics = runner.run(full_db, experiment_name="bss_regression", verbose=1)

    # -- Step 4: Report results --
    print("\n" + "=" * 70)
    print("  BSS REGRESSION RESULTS")
    print("=" * 70)
    print(f"\n  {'Fold':<25s} {'BSS':>10s} {'BS':>10s} {'AvgK':>8s} {'Trades':>8s}")
    print(f"  {'-' * 63}")

    bss_values = []
    for m in fold_metrics:
        bss = m.get("brier_skill_score", 0)
        bs = m.get("brier_score", 0)
        avgk = m.get("avg_matches", 0)
        trades = m.get("confident_trades", 0)
        sign = "+" if bss > 0 else ""
        bss_values.append(bss)
        print(f"  {m['fold']:<25s} {sign}{bss:>9.5f} {bs:>10.5f} {avgk:>8.1f} {trades:>8d}")

    positive_folds = sum(1 for b in bss_values if b > 0)
    mean_bss = sum(bss_values) / len(bss_values) if bss_values else 0

    print(f"\n  {'Summary':>25s}")
    print(f"  {'-' * 63}")
    print(f"  Mean BSS:              {'+' if mean_bss > 0 else ''}{mean_bss:.5f}")
    print(f"  Positive BSS folds:    {positive_folds}/{len(bss_values)}")

    # -- Regression check --
    # System A achieved +0.00100 BSS on the 2024 fold with Platt
    fold_2024 = [m for m in fold_metrics if "2024" in m.get("fold", "")]
    if fold_2024:
        bss_2024 = fold_2024[0].get("brier_skill_score", 0)
        print(f"\n  2024 Fold BSS:          {'+' if bss_2024 > 0 else ''}{bss_2024:.5f}")
        print(f"  System A reference:     +0.00100")
        if bss_2024 > 0:
            print(f"  RESULT: POSITIVE BSS on 2024 fold [PASS]")
        else:
            print(f"  RESULT: Negative BSS on 2024 fold -- investigate")

    elapsed = (time.time() - start) / 60
    print(f"\n  Total time: {elapsed:.1f} min")
    print("=" * 70 + "\n")

    # -- Step 5: Integrity check on 2024 fold --
    print(f"\n[4/4] Running cross-model integrity check on 2024 fold...")
    from pattern_engine.cross_validation import CrossValidator
    import pandas as pd
    fold_2024_def = [f for f in WALKFORWARD_FOLDS if "2024" in f["label"]][0]
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    train_db = full_db[full_db["Date"] <= fold_2024_def["train_end"]].copy()
    val_db = full_db[
        (full_db["Date"] >= fold_2024_def["val_start"]) &
        (full_db["Date"] <= fold_2024_def["val_end"])
    ].copy()
    integrity = CrossValidator.integrity_check(config, train_db, val_db, verbose=1)

    # Save results summary
    results_dir = worktree_data / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "bss_regression_summary.txt", "w") as f:
        f.write(f"BSS Regression Test Results\n")
        f.write(f"{'=' * 50}\n")
        for m in fold_metrics:
            bss = m.get("brier_skill_score", 0)
            f.write(f"{m['fold']}: BSS={bss:+.5f}\n")
        f.write(f"\nMean BSS: {mean_bss:+.5f}\n")
        f.write(f"Positive folds: {positive_folds}/{len(bss_values)}\n")
        f.write(f"\nIntegrity: {'PASS' if integrity['all_passed'] else 'FAIL'}\n")
        for k, v in integrity.items():
            if k != "all_passed":
                f.write(f"  {k}: {v}\n")

    return fold_metrics


if __name__ == "__main__":
    main()
