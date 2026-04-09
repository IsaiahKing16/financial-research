"""
scripts/compare_phase3_vs_phase4.py -- T4.4: Phase 3 vs Phase 4 head-to-head.

Reads results/phase3_walkforward.tsv and results/phase4_walkforward.tsv
and prints a delta table showing the marginal impact of the Portfolio
Manager filter on the 2024 fold.

The --no-pm parity contract (Phase 4 with PM disabled reproduces Phase 3
exactly) is verified in T4.1a and locked into commit history; this
script does NOT re-verify it on every run.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/compare_phase3_vs_phase4.py

Outputs:
    results/phase4_vs_phase3_comparison.txt

Plan: docs/superpowers/plans/2026-04-08-phase4-portfolio-manager-plan.md (T4.4)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
P3  = project_root / "results" / "phase3_walkforward.tsv"
P4  = project_root / "results" / "phase4_walkforward.tsv"
OUT = project_root / "results" / "phase4_vs_phase3_comparison.txt"


# field, display name, format, integer?
FIELDS: list[tuple[str, str, str, bool]] = [
    ("sharpe",          "Sharpe",         "{:12.4f}", False),
    ("max_dd",          "MaxDD",          "{:12.4f}", False),
    ("final_equity",    "Final equity",   "{:12.2f}", False),
    ("n_trades_placed", "Trades placed",  "{:12d}",   True),
    ("n_blocked",       "Trades blocked", "{:12d}",   True),
]


def main() -> int:
    if not P3.exists():
        print(f"FATAL: {P3} not found. Run run_phase3_walkforward.py first.")
        return 2
    if not P4.exists():
        print(f"FATAL: {P4} not found. Run run_phase4_walkforward.py first.")
        return 2

    p3 = pd.read_csv(P3, sep="\t").iloc[0]
    p4 = pd.read_csv(P4, sep="\t").iloc[0]

    lines: list[str] = [
        "Phase 3 vs Phase 4 -- 2024 fold",
        "=" * 70,
        "",
        f"{'Metric':<18s} {'Phase 3':>12s} {'Phase 4':>12s} "
        f"{'Delta':>12s} {'Delta %':>10s}",
        "-" * 70,
    ]

    for field, name, fmt, is_int in FIELDS:
        if field not in p3 or field not in p4:
            lines.append(f"{name:<18s}  (missing in one of the TSVs)")
            continue
        v3 = p3[field]
        v4 = p4[field]
        if is_int:
            v3i = int(v3)
            v4i = int(v4)
            delta_int = v4i - v3i
            rel = (delta_int / v3i * 100.0) if v3i != 0 else float("nan")
            rel_s = f"{rel:+9.1f}%" if v3i != 0 else "       n/a"
            lines.append(
                f"{name:<18s} {v3i:>12d} {v4i:>12d} "
                f"{delta_int:>+12d} {rel_s:>10s}"
            )
        else:
            delta = float(v4) - float(v3)
            rel = (delta / float(v3) * 100.0) if float(v3) != 0 else float("nan")
            rel_s = f"{rel:+9.2f}%" if float(v3) != 0 else "       n/a"
            lines.append(
                f"{name:<18s} {float(v3):>12.4f} {float(v4):>12.4f} "
                f"{delta:>+12.4f} {rel_s:>10s}"
            )

    lines.append("-" * 70)
    lines.append("")

    # Interpretive summary keyed to Phase 4's "don't regress" intent.
    sharpe_delta = float(p4["sharpe"]) - float(p3["sharpe"])
    sharpe_rel = sharpe_delta / float(p3["sharpe"]) * 100.0
    lines.append(
        f"Sharpe regression: {sharpe_rel:+.2f}% "
        f"(Phase 4 exit intent: stay within -25% of Phase 3)"
    )
    status = "within tolerance" if sharpe_rel > -25.0 else "EXCEEDS tolerance"
    lines.append(f"  Status: {status}")
    lines.append("")
    lines.append(
        "Note: --no-pm parity contract (Phase 4 --no-pm == Phase 3 exactly) "
        "was verified in T4.1a commit 13b81cf."
    )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
