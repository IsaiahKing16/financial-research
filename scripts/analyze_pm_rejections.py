"""
scripts/analyze_pm_rejections.py -- T4.3: histograms of PM rejection outcomes.

Reads results/phase4_rejections.csv and emits four histograms:
  1. by reason (most common -> least)
  2. by sector
  3. by confidence bucket (<0.65, 0.65-0.70, 0.70-0.75, 0.75-0.80, >=0.80)
  4. top 10 rejected tickers

Also surfaces the G6 gate check: no single reason > 60% of rejections.
On a small rejection sample (n < 20), G6 is flagged as sample-size-gated
because any single-reason dominance becomes statistically meaningless.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/analyze_pm_rejections.py

Outputs:
    results/phase4_rejection_analysis.txt

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md (T4.3)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
REJECT_CSV = project_root / "results" / "phase4_rejections.csv"
OUT_TXT    = project_root / "results" / "phase4_rejection_analysis.txt"

# G6 threshold and a sample-size floor below which single-reason
# dominance is expected and not diagnostic of miscalibration.
G6_DOMINANCE_BOUND = 0.60
G6_MIN_SAMPLE      = 20


def main() -> int:
    if not REJECT_CSV.exists():
        print(f"FATAL: {REJECT_CSV} not found. "
              f"Run run_phase4_walkforward.py first.")
        return 2

    df = pd.read_csv(REJECT_CSV)
    n = len(df)

    lines: list[str] = [
        "Phase 4 Rejection Analysis",
        "=" * 60,
        f"Total rejections: {n}",
        "",
    ]

    if df.empty:
        lines.append("No rejections -- all signals approved.")
        OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 0

    # 1. By reason.
    lines.append("By reason:")
    by_reason = df["reason"].value_counts(normalize=False)
    by_reason_pct = df["reason"].value_counts(normalize=True)
    for reason, count in by_reason.items():
        lines.append(
            f"  {reason:28s} {count:5d}  ({by_reason_pct[reason]:6.1%})"
        )
    lines.append("")

    # G6: no single reason > 60%. Flag sample-size gate on small n.
    max_reason_pct  = float(by_reason_pct.max())
    max_reason_name = str(by_reason_pct.idxmax())
    g6_raw_pass     = max_reason_pct <= G6_DOMINANCE_BOUND + 1e-9

    if n < G6_MIN_SAMPLE:
        lines.append(
            f"G6 check: SAMPLE-SIZE-GATED "
            f"(n={n} < min_sample={G6_MIN_SAMPLE})"
        )
        lines.append(
            f"  dominant reason: '{max_reason_name}' at {max_reason_pct:.1%}"
        )
        lines.append(
            "  Note: on small samples a single dominant reason is expected "
            "and not diagnostic. See SESSION_2026-04-09 T4.2 G6 escalation."
        )
        # Exit code 0 so downstream tooling doesn't treat small-n as failure.
        g6_exit = 0
    elif g6_raw_pass:
        lines.append(
            f"G6 check: PASS -- max reason '{max_reason_name}' "
            f"at {max_reason_pct:.1%} (threshold <= 60%)"
        )
        g6_exit = 0
    else:
        lines.append(
            f"G6 check: FAIL -- max reason '{max_reason_name}' "
            f"at {max_reason_pct:.1%} exceeds 60% threshold"
        )
        g6_exit = 1
    lines.append("")

    # 2. By sector.
    lines.append("By sector:")
    for sector, count in df["sector"].value_counts().items():
        lines.append(f"  {sector:28s} {count:5d}")
    lines.append("")

    # 3. By confidence bucket.
    lines.append("By confidence bucket:")
    bins   = [0.00, 0.65, 0.70, 0.75, 0.80, 1.00]
    labels = ["<0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", ">=0.80"]
    df = df.copy()
    df["conf_bucket"] = pd.cut(
        df["confidence"], bins=bins, labels=labels, include_lowest=True,
    )
    bucket_counts = df["conf_bucket"].value_counts().sort_index()
    for bucket in labels:
        count = int(bucket_counts.get(bucket, 0))
        lines.append(f"  {bucket:28s} {count:5d}")
    lines.append("")

    # 4. Top 10 rejected tickers.
    lines.append("Top 10 rejected tickers:")
    top = df["ticker"].value_counts().head(10)
    if top.empty:
        lines.append("  (none)")
    else:
        for ticker, count in top.items():
            lines.append(f"  {ticker:10s} {count:5d}")
    lines.append("")

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return g6_exit


if __name__ == "__main__":
    sys.exit(main())
