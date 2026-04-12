"""
scripts/experiments/validate_52t_best_config.py

Escalation Path 1 — validate Phase 1 best config at 52T universe.

Context (2026-04-02 session escalation):
  Phase 1 experiments (H1–H4) failed to meet the BSS gate on the 585T universe.
  Root cause: analogue pool dilution at 585T scale destroys resolution (Resolution≈0).
  The 52T baseline has Fold6 BSS=+0.00103 — discriminative signal exists there.

  Best config from Phase 1:
    - beta calibration: +0.00058 BSS on 585T, real and reproducible
    - max_distance=0.5: +0.00040 BSS, marginal but consistent
    - weighting=uniform: inverse is consistently worse (amplifies noise when Resolution≈0)

Test matrix (4 configs):
  1. baseline_52t        — max_d=1.1019, uniform, platt    (reproduces known baseline)
  2. beta_baseline_52t   — max_d=1.1019, uniform, beta_abm (beta cal only)
  3. platt_maxd05_52t    — max_d=0.5,    uniform, platt    (max_d only, control)
  4. beta_maxd05_52t     — max_d=0.5,    uniform, beta_abm (recommended: combined)

Gate: BSS > 0 on ≥ 3/6 folds.
  If met → Phase 1 gate satisfied at 52T. Lock settings, proceed to Phase 2 (Half-Kelly).
  If not met → escalate; investigate 52T signal further before Phase 2.

Data: data/processed/ (52T, 52 tickers, ~175K rows) — NOT the expanded 585T universe.

Output: results/validate_52t_best_config.tsv
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
from pattern_engine.features import RETURNS_ONLY_COLS, VOL_NORM_COLS

try:
    from betacal import BetaCalibration
except ImportError:
    raise RuntimeError(
        "betacal is not installed.\n"
        "Run: py -3.12 -m pip install betacal\n"
        "Then add 'betacal>=0.1.0' to requirements.txt."
    )


GATE_THRESHOLD_POSITIVE_FOLDS = 3
RESULTS_PATH = project_root / "results" / "validate_52t_best_config.tsv"

# ── 52T data path (pre-expansion legacy; do not use data/ which is 585T) ─────
_52T_DIR = project_root / "data" / "52t_volnorm"

# data/52t_volnorm/ was built by scripts/build_52t_volnorm.py (2026-04-02):
#   48 tickers filtered from 585T parquet (already has VOL_NORM_COLS)
#   4 missing tickers (ABBV, META, PYPL, TSLA) rebuilt from CSV caches
# This is the M9-equivalent 52T dataset — apples-to-apples with 585T experiments.
_52T_FEATURE_COLS = VOL_NORM_COLS


# ---------------------------------------------------------------------------
# Beta calibrator (identical to h4_beta_calibration.py — copy kept local so
# this script is self-contained and can run standalone)
# ---------------------------------------------------------------------------

class _BetaCalibrator:
    """Beta calibration drop-in for PatternMatcher._calibrator.

    Kull et al. (2017, AISTATS). parameters='abm': 3-parameter model;
    includes Platt sigmoid as special case so cannot harm already-calibrated scores.
    Interface is identical to _PlattCalibrator — no matcher.py modifications needed.
    """

    def __init__(self) -> None:
        self._cal = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_BetaCalibrator":
        self._cal = BetaCalibration(parameters="abm")
        self._cal.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._cal.predict(raw.reshape(-1, 1))


# ---------------------------------------------------------------------------
# Data loader — forced to 52T processed directory
# ---------------------------------------------------------------------------

def _load_52t_db() -> pd.DataFrame:
    """Load the 52T analogue database from data/processed/.

    Raises FileNotFoundError with a clear message if 52T data is missing —
    do NOT fall back to the root data/ directory (that is 585T).
    """
    t = _52T_DIR / "train_db.parquet"
    v = _52T_DIR / "val_db.parquet"
    if t.exists() and v.exists():
        combined = pd.concat([pd.read_parquet(t), pd.read_parquet(v)], ignore_index=True)
        n_tickers = combined["Ticker"].nunique()
        print(f"  Loaded 52T train+val: {len(combined):,} rows, {n_tickers} tickers from {_52T_DIR}")
        if n_tickers > 100:
            raise RuntimeError(
                f"Expected ~52 tickers in data/processed/ but found {n_tickers}. "
                "This is NOT the 52T dataset — check your data directory."
            )
        return combined

    raise FileNotFoundError(
        f"52T dataset not found at {_52T_DIR}.\n"
        "Expected: data/processed/train_db.parquet + data/processed/val_db.parquet\n"
        "Run prepare.py targeting the 52T universe or restore from backup."
    )


# ---------------------------------------------------------------------------
# Walk-forward runner — supports optional beta calibrator injection
# ---------------------------------------------------------------------------

def _run_walkforward(config: EngineConfig, full_db: pd.DataFrame,
                     use_beta: bool = False) -> list[dict]:
    """Run 6-fold walk-forward. If use_beta=True, monkey-patches _PlattCalibrator."""
    import pattern_engine.matcher as _matcher_module

    full_db = full_db.copy()
    full_db["Date"] = pd.to_datetime(full_db["Date"])
    horizon = config.projection_horizon
    fold_results = []

    original_calibrator = _matcher_module._PlattCalibrator

    for fold in WALKFORWARD_FOLDS:
        train_end = pd.to_datetime(fold["train_end"])
        val_start = pd.to_datetime(fold["val_start"])
        val_end   = pd.to_datetime(fold["val_end"])

        train_db = full_db[full_db["Date"] <= train_end].dropna(subset=[horizon]).copy()
        val_db   = full_db[
            (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
        ].dropna(subset=[horizon]).copy()

        if len(val_db) == 0:
            fold_results.append({
                "fold": fold["label"], "bss": float("nan"),
                "bs": float("nan"), "avg_k": 0.0, "n_val": 0,
            })
            continue

        try:
            if use_beta:
                _matcher_module._PlattCalibrator = _BetaCalibrator
            matcher = PatternMatcher(config)
            matcher.fit(train_db, _52T_FEATURE_COLS)
            probs, _, _, n_matches, _, _ = matcher.query(val_db, verbose=0)
        finally:
            _matcher_module._PlattCalibrator = original_calibrator

        y_true    = val_db[horizon].values.astype(float)
        base_rate = y_true.mean()
        bs_ref    = base_rate * (1 - base_rate)
        bs_model  = float(np.mean((np.asarray(probs) - y_true) ** 2))
        bss_val   = 1.0 - bs_model / bs_ref if bs_ref > 1e-10 else 0.0

        fold_results.append({
            "fold":  fold["label"],
            "bss":   round(bss_val, 6),
            "bs":    round(bs_model, 6),
            "avg_k": round(float(np.mean(n_matches)), 2),
            "n_val": len(val_db),
        })

    return fold_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fold_labels = [f["label"] for f in WALKFORWARD_FOLDS]
    tsv_cols = (
        ["config_label", "calibrator", "max_distance", "distance_weighting",
         "mean_bss", "positive_folds", "gate_met", "mean_avg_k"]
        + [f"bss_{lbl}" for lbl in fold_labels]
    )
    header_lines = [
        "# validate_52t_best_config.tsv",
        f"# Started: {pd.Timestamp.now('UTC').isoformat()}",
        "# Escalation Path 1 — Phase 1 best config at 52T universe",
        "# 4 configs: baseline, beta-only, maxd-only, beta+maxd",
        "# Features: VOL_NORM_COLS (M9 vol-normalized; rebuilt from 585T parquet + CSV caches)",
        f"# Gate: positive_folds >= {GATE_THRESHOLD_POSITIVE_FOLDS}",
        "# Rows appended incrementally — file is valid if run is interrupted.",
        "#",
    ]

    # Resume: detect already-completed configs
    all_results: list[dict] = []
    completed_labels: set[str] = set()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_PATH.exists():
        try:
            existing = pd.read_csv(RESULTS_PATH, comment="#", sep="\t")
            if "config_label" in existing.columns and len(existing) > 0:
                completed_labels = set(existing["config_label"].tolist())
                for _, row in existing.iterrows():
                    all_results.append(row.to_dict())
                print(f"\n  [RESUME] Skipping completed: {sorted(completed_labels)}")
        except Exception:
            pass

    if not completed_labels:
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            for line in header_lines:
                f.write(line + "\n")
            f.write("\t".join(tsv_cols) + "\n")

    start = time.time()
    print("\n" + "=" * 70)
    print("  VALIDATE 52T BEST CONFIG — Escalation Path 1")
    print("=" * 70)
    print("\n  Universe: 52T (data/processed/) — NOT the 585T expanded universe")
    print(f"  Gate: BSS > 0 on ≥ {GATE_THRESHOLD_POSITIVE_FOLDS}/6 folds")

    print(f"\n  Loading 52T data...", end="", flush=True)
    full_db = _load_52t_db()
    print()

    configs = [
        # 1. Reproduce known baseline (sanity check — must give Fold6 BSS≈+0.00103)
        {"label": "baseline_52t",      "max_d": 1.1019, "weighting": "uniform", "beta": False},
        # 2. Beta calibration only (isolate beta contribution at 52T)
        {"label": "beta_baseline_52t", "max_d": 1.1019, "weighting": "uniform", "beta": True},
        # 3. max_d=0.5 only (isolate distance tightening at 52T, Platt control)
        {"label": "platt_maxd05_52t",  "max_d": 0.5,    "weighting": "uniform", "beta": False},
        # 4. Recommended combined: max_d=0.5 + beta cal (the Phase 1 winner)
        {"label": "beta_maxd05_52t",   "max_d": 0.5,    "weighting": "uniform", "beta": True},
    ]

    for spec in configs:
        if spec["label"] in completed_labels:
            print(f"\n  Skipping: {spec['label']} (already completed)")
            continue

        cal_name = "beta_abm" if spec["beta"] else "platt"
        print(f"\n  Running: {spec['label']} (max_d={spec['max_d']}, {cal_name})...",
              end="", flush=True)

        cfg = EngineConfig()
        cfg.max_distance       = spec["max_d"]
        cfg.distance_weighting = spec["weighting"]

        t_cfg = time.time()
        fold_metrics   = _run_walkforward(cfg, full_db, use_beta=spec["beta"])
        elapsed        = time.time() - t_cfg

        bss_values     = [m["bss"] for m in fold_metrics if not np.isnan(m["bss"])]
        mean_bss       = float(np.mean(bss_values)) if bss_values else float("nan")
        positive_folds = sum(1 for b in bss_values if b > 0)
        gate_met       = positive_folds >= GATE_THRESHOLD_POSITIVE_FOLDS

        row: dict = {
            "config_label":       spec["label"],
            "calibrator":         cal_name,
            "max_distance":       spec["max_d"],
            "distance_weighting": spec["weighting"],
            "mean_bss":           round(mean_bss, 6),
            "positive_folds":     positive_folds,
            "gate_met":           gate_met,
            "mean_avg_k": round(
                float(np.mean([m["avg_k"] for m in fold_metrics if m["avg_k"] > 0])), 2
            ),
        }
        for m in fold_metrics:
            row[f"bss_{m['fold']}"] = m["bss"]
        all_results.append(row)

        # Incremental append — safe even if process is killed
        with open(RESULTS_PATH, "a", encoding="utf-8") as f:
            f.write("\t".join(str(row.get(col, "")) for col in tsv_cols) + "\n")

        flag = "  *** GATE MET ***" if gate_met else ""
        print(f" {elapsed:.0f}s — mean_BSS={mean_bss:+.5f} pos_folds={positive_folds}/6{flag}")

        # Per-fold detail
        for m in fold_metrics:
            bss_str = f"{m['bss']:+.6f}" if not np.isnan(m["bss"]) else "    N/A"
            marker  = " *" if m["bss"] > 0 else ""
            print(f"    {m['fold']:<14} BSS={bss_str}  avgK={m['avg_k']:5.1f}  n={m['n_val']:,}{marker}")

    # Final sorted rewrite
    df = pd.DataFrame(all_results)
    df = df.sort_values("mean_bss", ascending=False)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, sep="\t", index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  {'Config':<25} {'Cal':<10} {'maxD':>6} {'meanBSS':>10} {'posFolds':>9} {'Gate':>6}")
    print("  " + "-" * 65)
    for _, r in df.iterrows():
        gate_str = "YES ***" if r["gate_met"] else "no"
        print(f"  {r['config_label']:<25} {r['calibrator']:<10} {r['max_distance']:>6.4f} "
              f"{r['mean_bss']:>+10.5f} {int(r['positive_folds']):>5}/6    {gate_str}")

    gate_winners = df[df["gate_met"] == True]
    if len(gate_winners) > 0:
        best = gate_winners.iloc[0]
        print(f"\n  GATE MET: {best['config_label']} — mean_BSS={best['mean_bss']:+.5f}, "
              f"{int(best['positive_folds'])}/6 positive folds")
        print("\n  NEXT STEP: Lock beta_cal + max_d=0.5, proceed to Phase 2 (Half-Kelly).")
        print("  Update CLAUDE.md locked settings with beta calibration.")
    else:
        print(f"\n  Gate NOT met by any config. Best: {df.iloc[0]['config_label']} "
              f"(mean_BSS={df.iloc[0]['mean_bss']:+.5f})")
        print("\n  ESCALATE: 52T signal insufficient for Phase 2 gate.")
        print("  Investigate: feature quality at 52T, data integrity, calibration approach.")

    print(f"\n  TSV: {RESULTS_PATH}")
    print(f"  Total time: {(time.time() - start) / 60:.1f} min")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
