"""
scripts/experiments/phase1_summary.py

Reads all completed sweep TSVs and produces a single ranked summary.
Identifies the winning configuration (if any) and calls
MurphyGate.record_winning_config() if the gate is met.

Only run after all desired Phase B experiments have completed.

Usage:
    PYTHONUTF8=1 py -3.12 scripts/experiments/phase1_summary.py

Output: results/phase1_sweep_summary.tsv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.diagnostics.murphy_gate import MurphyGate

GATE_MIN_POSITIVE_FOLDS = 3

TSV_FILES = {
    "H1": project_root / "results" / "bss_fix_sweep_h1.tsv",
    "H2": project_root / "results" / "bss_fix_sweep_h2.tsv",
    "H3": project_root / "results" / "bss_fix_sweep_h3.tsv",
    "H4": project_root / "results" / "bss_fix_sweep_h4.tsv",
}


def load_tsv(path: Path, experiment: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#", sep="\t")
    df["experiment"] = experiment
    return df


def main() -> None:
    gate = MurphyGate.load()   # Read-only — no enforcement needed for summary

    print("\n" + "=" * 70)
    print("  PHASE 1 SWEEP SUMMARY")
    print("=" * 70)

    all_rows = []
    for exp, path in TSV_FILES.items():
        df = load_tsv(path, exp)
        if not df.empty:
            all_rows.append(df)
            print(f"  Loaded {len(df)} rows from {path.name}")
        else:
            print(f"  {path.name} not found — skipping {exp}")

    if not all_rows:
        print("\n  No sweep results found. Run Phase B experiments first.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("mean_bss", ascending=False)

    # Console table
    print(f"\n  {'Experiment':<8} {'Config':<38} {'mean_BSS':>10} {'pos_folds':>10} {'Gate':>6}")
    print(f"  {'-' * 74}")

    # Baseline row
    print(f"  {'Baseline':<8} {'max_d=1.1019, uniform, Platt':<38} {'-0.00459':>10} {'0/6':>10} {'NO':>6}")

    for _, row in combined.head(12).iterrows():
        config_str = ""
        for col in ["max_distance", "distance_weighting", "top_k",
                    "same_sector_only", "calibrator", "config_label"]:
            if col in row and not pd.isna(row.get(col)):
                config_str += f"{col}={row[col]} "
        config_str = config_str.strip()[:37]
        sign = "+" if row["mean_bss"] > 0 else ""
        gate_flag = "YES" if row.get("gate_met", False) else "NO"
        print(f"  {row['experiment']:<8} {config_str:<38} "
              f"{sign}{row['mean_bss']:>9.5f} "
              f"{row['positive_folds']:>9}/6 "
              f"{gate_flag:>6}")

    # Find overall winner
    gate_met_rows = combined[combined["gate_met"] == True]
    if not gate_met_rows.empty:
        winner = gate_met_rows.sort_values("mean_bss", ascending=False).iloc[0]
        print(f"\n  *** PHASE 1 GATE MET ***")
        print(f"  Winner: Experiment={winner['experiment']}  "
              f"mean_BSS={winner['mean_bss']:+.5f}  "
              f"pos_folds={winner['positive_folds']}/6")

        # Build winning params dict
        params = {}
        for col in ["max_distance", "distance_weighting", "top_k",
                    "same_sector_only", "same_sector_boost_factor", "calibrator"]:
            if col in winner and not pd.isna(winner.get(col)):
                params[col] = winner[col]

        MurphyGate.record_winning_config(
            experiment=str(winner["experiment"]),
            params=params,
            mean_bss=float(winner["mean_bss"]),
            positive_folds=int(winner["positive_folds"]),
        )

        print(f"\n  Next action: Update locked settings in pattern_engine/config.py,")
        print(f"  CLAUDE.md, and fppe-roadmap-v2A.md with these params + provenance.")
    else:
        print(f"\n  *** PHASE 1 GATE NOT MET — all experiments failed ***")
        print(f"  Best result: mean_BSS={combined.iloc[0]['mean_bss']:+.5f}  "
              f"pos_folds={combined.iloc[0]['positive_folds']}/6")
        print(f"\n  3-STRIKE RULE: Stop and escalate. Do not attempt further variations.")
        print(f"  Create results/phase1_escalation_log.txt with exact numbers.")
        print(f"  Escalation paths:")
        print(f"    1. Revert to 52-ticker universe for live deployment")
        print(f"    2. Activate research/bma_calibrator.py as rescue calibrator")
        print(f"    3. Investigate if Resolution=0 is fundamental (signal extinction at 585T)")

    # Write summary TSV
    summary_path = project_root / "results" / "phase1_sweep_summary.tsv"
    header_lines = [
        "# phase1_sweep_summary.tsv",
        f"# Generated: {pd.Timestamp.utcnow().isoformat()}Z",
        "# All Phase 1 BSS fix experiments combined and ranked by mean_bss",
        "#",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        for line in header_lines:
            f.write(line + "\n")
        combined.to_csv(f, sep="\t", index=False)
    print(f"\n  Summary TSV: {summary_path}")


if __name__ == "__main__":
    main()
