"""
run_experiment.py — Single walk-forward run with configurable filter flags.

Usage:
    python scripts/run_experiment.py --name E0
    python scripts/run_experiment.py --name E1 --sector-conviction
    python scripts/run_experiment.py --name E2 --momentum-filter
    python scripts/run_experiment.py --name E3 --sector-conviction --momentum-filter

All runs use confidence_threshold=0.55 (calibrated from diagnose_prob_dist.py:
max Platt prob on 585T universe is 0.6195; threshold=0.65 never fires).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.matcher import PatternMatcher
from pattern_engine.features import VOL_NORM_COLS
from scripts.run_walkforward import WalkForwardConfig, load_full_db, FOLDS, bss, accuracy

FEATURE_COLS = VOL_NORM_COLS
EXPERIMENT_THRESHOLD = 0.55


def run_fold(full_db: pd.DataFrame, fold: dict, cfg: WalkForwardConfig) -> dict:
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    train_db = full_db[full_db["Date"] <= train_end].copy()
    val_db   = full_db[
        (full_db["Date"] >= val_start) & (full_db["Date"] <= val_end)
    ].copy()
    train_db = train_db.dropna(subset=[cfg.projection_horizon])
    val_db   = val_db.dropna(subset=[cfg.projection_horizon])

    if len(val_db) == 0:
        return {"label": fold["label"], "bss": float("nan"), "accuracy": float("nan"),
                "n_val": 0, "avg_k": 0.0, "buy": 0, "sell": 0, "hold": 0,
                "buy_filt": 0, "sell_filt": 0, "acc_filt": float("nan")}

    matcher = PatternMatcher(cfg)
    matcher.fit(train_db, FEATURE_COLS)
    probs, signals, _, n_matches, _, _ = matcher.query(val_db, verbose=0)

    signals = list(signals)
    _active_filters = []

    if getattr(cfg, "use_sector_conviction", False):
        from pattern_engine.sector_conviction import SectorConvictionLayer
        from pattern_engine.sector import SECTOR_MAP
        layer = SectorConvictionLayer(
            SECTOR_MAP,
            min_sector_lift=getattr(cfg, "sector_conviction_lift", 0.005),
        )
        layer.fit(train_db, target_col=cfg.projection_horizon)
        _active_filters.append(layer)

    if getattr(cfg, "use_momentum_filter", False):
        from pattern_engine.momentum_signal import MomentumSignalFilter
        from pattern_engine.sector import SECTOR_MAP
        _mom_col = "ret_7d" if "ret_7d" in val_db.columns else "ret_7d_norm"
        mf = MomentumSignalFilter(
            SECTOR_MAP,
            lookback_col=_mom_col,
            min_outperformance=getattr(cfg, "momentum_min_outperformance", 0.015),
        )
        mf.fit(train_db)
        _active_filters.append(mf)

    if _active_filters:
        from pattern_engine.signal_pipeline import SignalPipeline
        pipeline = SignalPipeline(filters=_active_filters)
        signals, _ = pipeline.run(np.asarray(probs), signals, val_db)

    probs_arr = np.asarray(probs)
    sigs_arr  = np.asarray(signals)
    y_true    = val_db[cfg.projection_horizon].values.astype(np.float64)

    buy_mask  = sigs_arr == "BUY"
    sell_mask = sigs_arr == "SELL"
    act_mask  = buy_mask | sell_mask

    acc_filt = float("nan")
    if act_mask.sum() > 0:
        pred = np.where(buy_mask[act_mask], 1, 0)
        acc_filt = float(np.mean(pred == y_true[act_mask].astype(int)))

    return {
        "label":     fold["label"],
        "bss":       bss(probs_arr, y_true),
        "accuracy":  accuracy(probs_arr, y_true),
        "n_val":     len(val_db),
        "avg_k":     float(np.mean(n_matches)),
        "buy":       int(buy_mask.sum()),
        "sell":      int(sell_mask.sum()),
        "hold":      int(np.sum(sigs_arr == "HOLD")),
        "buy_filt":  int(buy_mask.sum()),
        "sell_filt": int(sell_mask.sum()),
        "acc_filt":  acc_filt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="EX", help="Experiment label")
    parser.add_argument("--sector-conviction", action="store_true")
    parser.add_argument("--momentum-filter",   action="store_true")
    args = parser.parse_args()

    cfg = WalkForwardConfig(
        confidence_threshold=EXPERIMENT_THRESHOLD,
        use_sector_conviction=args.sector_conviction,
        use_momentum_filter=args.momentum_filter,
    )

    flags = []
    if args.sector_conviction: flags.append("sector_conviction")
    if args.momentum_filter:   flags.append("momentum_filter")
    flag_str = ", ".join(flags) if flags else "none"

    print("=" * 70)
    print(f"  FPPE Experiment {args.name}  |  filters: {flag_str}")
    print(f"  threshold={EXPERIMENT_THRESHOLD}  (calibrated: max_prob=0.6195 on 585T)")
    print("=" * 70)

    t0 = time.time()
    full_db = load_full_db()
    full_db["Date"] = pd.to_datetime(full_db["Date"])

    results = []
    print(f"\n{'Fold':<14} {'Val':>8} {'BSS':>9} {'Acc':>7} {'AvgK':>6} "
          f"{'BUY':>7} {'SELL':>6} {'HOLD':>8} {'AccFilt':>8}")
    print("-" * 70)

    for i, fold in enumerate(FOLDS, 1):
        print(f"  [{i}/6] {fold['label']}...", end=" ", flush=True)
        t_fold = time.time()
        r = run_fold(full_db, fold, cfg)
        elapsed = time.time() - t_fold
        results.append(r)

        bss_s = f"{r['bss']:+.5f}" if not np.isnan(r["bss"]) else "   N/A "
        acc_s = f"{r['accuracy']:.1%}" if not np.isnan(r["accuracy"]) else "  N/A"
        af_s  = f"{r['acc_filt']:.3f}" if not np.isnan(r["acc_filt"]) else "  N/A"
        flag  = " *" if r["bss"] > 0 else ""
        print(f"done ({elapsed:.0f}s)")
        print(f"{r['label']:<14} {r['n_val']:>8,} {bss_s:>9} {acc_s:>7} {r['avg_k']:>6.1f} "
              f"{r['buy']:>7,} {r['sell']:>6,} {r['hold']:>8,} {af_s:>8}{flag}")

    total = time.time() - t0
    print("-" * 70)
    pos = sum(1 for r in results if r["bss"] > 0)
    mean_bss = np.nanmean([r["bss"] for r in results])
    total_buy  = sum(r["buy"] for r in results)
    total_sell = sum(r["sell"] for r in results)
    print(f"\nPositive-BSS folds: {pos}/6   Mean BSS: {mean_bss:+.5f}")
    print(f"Total BUY signals:  {total_buy:,}   Total SELL: {total_sell:,}")
    print(f"Runtime: {total:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
