"""
scripts/diagnostics/b3_murphy_decomposition.py

Murphy (1973) BSS decomposition across all 6 walk-forward folds.

Determines whether negative BSS is driven by:
  - Near-zero Resolution: model has no discriminative signal at 585T scale.
    Pool dilution destroyed the KNN signal entirely. Calibration cannot help.
  - High Reliability error: probabilities are systematically biased.
    Calibration fixes (H4) are the primary lever.
  - Mixed: both components contribute.

This script is PHASE A — it must complete before any Phase B experiment.
On completion it writes:
  results/murphy_gate.json                       (machine-readable enforcement gate)
  results/benchmarks/b3_murphy_decomposition.tsv (human-readable per-fold table)

Usage:
    PYTHONUTF8=1 py -3.12 scripts/diagnostics/b3_murphy_decomposition.py

References:
  Murphy (1973) "A New Vector Partition of the Probability Score"
  Stephenson (2008) — within-bin correction for binned decomposition
  KNN_Calibration_Phase_1.pdf §4 (FPPE project reference)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path setup ---
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pattern_engine.config import EngineConfig, WALKFORWARD_FOLDS
from pattern_engine.features import VOL_NORM_COLS
from scripts.diagnostics.murphy_gate import MurphyGate


# ---------------------------------------------------------------------------
# Data loading (parquet-based — mirrors run_walkforward.py load_full_db)
# ---------------------------------------------------------------------------

def _load_full_db() -> pd.DataFrame:
    """Load the full analogue database from parquet files produced by prepare.py."""
    data_dir = project_root / "data"
    processed_dir = data_dir / "processed"

    for base in (data_dir, processed_dir):
        t, v = base / "train_db.parquet", base / "val_db.parquet"
        if t.exists() and v.exists():
            combined = pd.concat([pd.read_parquet(t), pd.read_parquet(v)], ignore_index=True)
            print(f"  Loaded train+val from {base}: {len(combined):,} rows")
            return combined

    for fallback in (data_dir / "full_analogue_db.parquet", processed_dir / "full_db.parquet"):
        if fallback.exists():
            print(f"  WARNING: using fallback {fallback}")
            return pd.read_parquet(fallback)

    raise FileNotFoundError(
        f"No analogue database found in {data_dir}. Run prepare.py first."
    )


# ---------------------------------------------------------------------------
# Murphy decomposition (exact, no binning)
# ---------------------------------------------------------------------------

def murphy_decompose(probs: np.ndarray, y: np.ndarray) -> dict:
    """Compute exact Murphy (1973) BSS decomposition.

    Uses unique predicted probability values as strata — NOT equal-width bins.
    This avoids the within-bin correction artifacts described in Stephenson (2008)
    that distort interpretation when predicted probabilities are discrete (as they
    are with KNN, where prob = k_up / k and k is small).

    Args:
        probs: (N,) array of predicted probabilities in [0, 1].
        y:     (N,) array of binary outcomes {0, 1}.

    Returns:
        dict with keys: bs, bss, uncertainty, resolution, reliability, base_rate, n
    """
    if len(probs) != len(y):
        raise RuntimeError(
            f"murphy_decompose: len(probs)={len(probs)} != len(y)={len(y)}"
        )
    if len(probs) == 0:
        raise RuntimeError("murphy_decompose: empty arrays — no predictions to decompose")

    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(probs)
    base_rate = y.mean()

    # Uncertainty component (irreducible — only depends on base rate)
    uncertainty = base_rate * (1.0 - base_rate)

    # Group by unique predicted probability values
    unique_probs = np.unique(probs)
    resolution = 0.0
    reliability = 0.0

    for p_val in unique_probs:
        mask = probs == p_val
        n_k = mask.sum()
        o_k = y[mask].mean()  # observed frequency in this stratum
        resolution  += n_k * (o_k - base_rate) ** 2
        reliability += n_k * (p_val - o_k) ** 2

    resolution  /= n
    reliability /= n

    # Brier Score and BSS
    bs = reliability - resolution + uncertainty
    bss = 1.0 - bs / uncertainty if uncertainty > 1e-10 else 0.0

    return {
        "bs": bs,
        "bss": bss,
        "uncertainty": uncertainty,
        "resolution": resolution,
        "reliability": reliability,
        "base_rate": base_rate,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Walk-forward runner that returns (probs, y_true) per fold
# ---------------------------------------------------------------------------

def _run_fold(
    config: EngineConfig,
    full_db: pd.DataFrame,
    fold: dict,
) -> dict:
    """Run one walk-forward fold and return raw predictions + outcomes.

    Args:
        config:  EngineConfig instance.
        full_db: Full feature database (all tickers, all dates).
        fold:    Fold dict from WALKFORWARD_FOLDS with keys:
                 label, train_end, val_start, val_end.

    Returns:
        dict with keys: fold_label, probs, y_true, n_queries
    """
    from pattern_engine.matcher import PatternMatcher

    full_db = full_db.copy()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    train_end  = pd.to_datetime(fold["train_end"])
    val_start  = pd.to_datetime(fold["val_start"])
    val_end    = pd.to_datetime(fold["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].copy()
    val_db   = full_db[
        (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
    ].copy()

    # Drop rows where target is missing
    horizon = config.projection_horizon
    train_db = train_db.dropna(subset=[horizon])
    val_db   = val_db.dropna(subset=[horizon])

    if len(train_db) == 0 or len(val_db) == 0:
        raise RuntimeError(
            f"Fold {fold['label']}: empty train ({len(train_db)}) or "
            f"val ({len(val_db)}) set after dropna."
        )

    matcher = PatternMatcher(config)
    matcher.fit(train_db, VOL_NORM_COLS)
    probs, _, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    y_true = val_db[horizon].values.astype(np.float64)

    return {
        "fold_label": fold["label"],
        "probs": np.asarray(probs, dtype=np.float64),
        "y_true": y_true,
        "n_queries": len(val_db),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()

    print("\n" + "=" * 70)
    print("  B3: MURPHY BSS DECOMPOSITION — Phase A Diagnostic")
    print("=" * 70)
    print("\nThis is the blocking gate before Phase B experiments.")
    print("Output: results/murphy_gate.json + results/benchmarks/b3_murphy_decomposition.tsv")
    print("\nConfig: EngineConfig() defaults (max_distance=1.1019, top_k=50, uniform)")

    # --- Load data ---
    print("\n[1/3] Loading data...")
    full_db = _load_full_db()
    print(f"  {full_db['Ticker'].nunique()} tickers, {len(full_db):,} rows total")

    # --- Run all 6 folds ---
    config = EngineConfig()   # locked defaults — do NOT override
    fold_results = []
    all_probs_combined = []
    all_y_combined     = []

    print(f"\n[2/3] Running 6-fold walk-forward (EngineConfig defaults)...")
    for i, fold in enumerate(WALKFORWARD_FOLDS):
        print(f"\n  Fold {i+1}/6: {fold['label']}", end="", flush=True)
        t_fold = time.time()
        fold_data = _run_fold(config, full_db, fold)
        elapsed = time.time() - t_fold

        decomp = murphy_decompose(fold_data["probs"], fold_data["y_true"])

        fold_results.append({
            "fold":        fold_data["fold_label"],
            "bs":          decomp["bs"],
            "bss":         decomp["bss"],
            "uncertainty": decomp["uncertainty"],
            "resolution":  decomp["resolution"],
            "reliability": decomp["reliability"],
            "n_queries":   decomp["n"],
            "base_rate":   decomp["base_rate"],
        })

        all_probs_combined.extend(fold_data["probs"].tolist())
        all_y_combined.extend(fold_data["y_true"].tolist())

        sign = "+" if decomp["bss"] > 0 else ""
        print(f" ({elapsed:.0f}s)")
        print(f"    BSS={sign}{decomp['bss']:.5f}  "
              f"Resolution={decomp['resolution']:.6f}  "
              f"Reliability={decomp['reliability']:.6f}  "
              f"Uncertainty={decomp['uncertainty']:.4f}  "
              f"N={decomp['n']:,}")

    # --- Aggregate decomposition ---
    all_probs_arr = np.array(all_probs_combined)
    all_y_arr     = np.array(all_y_combined)
    pooled_decomp = murphy_decompose(all_probs_arr, all_y_arr)

    mean_resolution  = float(np.mean([r["resolution"]  for r in fold_results]))
    mean_reliability = float(np.mean([r["reliability"] for r in fold_results]))
    mean_bss         = float(np.mean([r["bss"]         for r in fold_results]))

    # --- Console report ---
    print("\n" + "=" * 70)
    print("  MURPHY DECOMPOSITION RESULTS")
    print("=" * 70)
    print(f"\n  {'Fold':<25s} {'BSS':>10s} {'Resolution':>12s} {'Reliability':>12s} {'Base Rate':>10s} {'N':>8s}")
    print(f"  {'-' * 79}")
    for r in fold_results:
        sign = "+" if r["bss"] > 0 else ""
        print(f"  {r['fold']:<25s} {sign}{r['bss']:>9.5f} {r['resolution']:>12.6f} "
              f"{r['reliability']:>12.6f} {r['base_rate']:>10.4f} {r['n_queries']:>8,}")
    print(f"  {'-' * 79}")
    print(f"  {'Mean (folds)':<25s} {'+' if mean_bss>0 else ''}{mean_bss:>9.5f} "
          f"{mean_resolution:>12.6f} {mean_reliability:>12.6f}")
    print(f"  {'Pooled (all folds)':<25s} {'+' if pooled_decomp['bss']>0 else ''}"
          f"{pooled_decomp['bss']:>9.5f} {pooled_decomp['resolution']:>12.6f} "
          f"{pooled_decomp['reliability']:>12.6f}")

    # --- Interpretation ---
    print("\n" + "=" * 70)
    print("  MURPHY DECOMPOSITION INTERPRETATION")
    print("=" * 70)
    print(f"\n  Resolution (mean across folds): {mean_resolution:.6f}")
    print(f"    Threshold for 'near zero':    < 0.001000")
    if mean_resolution < 0.001:
        print(f"    *** NEAR ZERO — pool dilution has destroyed discriminative signal ***")
        print(f"    H4 (beta calibration) CANNOT fix this on its own.")
        print(f"    Signal quality fixes (H1, H3) are ESSENTIAL first.")
    else:
        print(f"    OK — model retains some discriminative signal at 585T scale.")

    print(f"\n  Reliability error (mean across folds): {mean_reliability:.6f}")
    print(f"    Threshold for 'dominant':             > 0.002000")
    if mean_reliability > 0.002:
        print(f"    *** DOMINANT — probabilities are systematically biased ***")
        print(f"    H4 (beta calibration) is a primary fix.")
    else:
        print(f"    Low — calibration bias is not the primary problem.")

    print(f"\n  BSS = Resolution - Reliability + Uncertainty")
    print(f"  BSS ≈ {mean_bss:+.5f}  (mean across folds)")
    print(f"  Uncertainty ≈ {float(np.mean([r['uncertainty'] for r in fold_results])):.4f}")
    print(f"  (Uncertainty ≈ 0.25 expected for balanced binary task)")

    # --- Write TSV ---
    print(f"\n[3/3] Writing outputs...")
    results_dir = project_root / "results" / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = results_dir / "b3_murphy_decomposition.tsv"

    header_lines = [
        "# b3_murphy_decomposition.tsv",
        f"# Generated: {pd.Timestamp.utcnow().isoformat()}Z",
        "# Config: max_distance=1.1019, top_k=50, distance_weighting=uniform, calibration=platt",
        "# Purpose: Murphy (1973) BSS decomposition to determine Resolution vs Reliability failure mode",
        f"# Mean resolution: {mean_resolution:.6f}  |  Mean reliability: {mean_reliability:.6f}",
        f"# Mean BSS: {mean_bss:+.5f}",
        "#",
    ]
    df_out = pd.DataFrame(fold_results)
    df_out["bs"]          = df_out["bs"].round(6)
    df_out["bss"]         = df_out["bss"].round(6)
    df_out["uncertainty"] = df_out["uncertainty"].round(6)
    df_out["resolution"]  = df_out["resolution"].round(6)
    df_out["reliability"] = df_out["reliability"].round(6)
    df_out["base_rate"]   = df_out["base_rate"].round(4)

    with open(tsv_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df_out.to_csv(f, sep="\t", index=False)

    print(f"  TSV written: {tsv_path}")

    # --- Write gate file (MUST be last — other scripts block on this) ---
    gate = MurphyGate.write(
        mean_resolution=mean_resolution,
        mean_reliability=mean_reliability,
        fold_results=fold_results,
    )

    elapsed = (time.time() - start) / 60
    print(f"\n  Total time: {elapsed:.1f} min")
    print("\n" + "=" * 70)
    print("  PHASE A COMPLETE")
    print("=" * 70)
    print(f"\n  Dominant failure mode: {gate['diagnosis']['dominant_failure_mode'].upper()}")
    print(f"  Recommended branch:    {gate['diagnosis']['recommended_branch']}")
    print(f"\n  *** STOP — Report these results to the user before running Phase B ***")
    print(f"  Gate file: results/murphy_gate.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
