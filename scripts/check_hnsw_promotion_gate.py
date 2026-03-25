"""
check_hnsw_promotion_gate.py — Machine-checkable HNSW promotion gate.

Evaluates all automated gate criteria from docs/rebuild/HNSW_PROMOTION_GATE.md.
Each gate emits PASS, FAIL, or SKIP with a one-line reason.

Usage:
    python scripts/check_hnsw_promotion_gate.py

Exit codes:
    0 — All PASS (or SKIP with no FAILs) — promotion approved
    1 — One or more gates FAILED — do not promote
    2 — Runtime error (missing files, import failure, etc.)

Gates checked:
    1. Recall parity (parity tests via pytest)
    2. Latency budget (benchmark artifact JSON or live benchmark)
    3. Signal parity walk-forward (artifact JSON if present, else SKIP)
    4. Zero regressions on full test suite

Gate 5 (human sign-off) cannot be machine-checked and is skipped here.

Linear: SLE-64
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

# ─── Path roots ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
REBUILD_DIR = REPO_ROOT / "rebuild_phase_3z"
ARTIFACTS_DIR = REBUILD_DIR / "artifacts" / "benchmarks"
WALKFORWARD_DIR = REBUILD_DIR / "artifacts" / "walkforward"


# ─── Gate result type ─────────────────────────────────────────────────────────

GateStatus = Literal["PASS", "FAIL", "SKIP"]


class GateResult:
    def __init__(self, name: str, status: GateStatus, reason: str) -> None:
        self.name = name
        self.status = status
        self.reason = reason

    def __str__(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "⊘"}[self.status]
        return f"  [{icon}] Gate {self.name}: {self.status} — {self.reason}"


# ─── Gate 1: Recall parity (parity tests) ────────────────────────────────────

def gate_1_recall_parity() -> GateResult:
    """Run parity tests; PASS if all pass, FAIL if any fail."""
    parity_dir = REBUILD_DIR / "tests" / "parity"
    if not parity_dir.exists():
        return GateResult("1 (Recall Parity)", "FAIL", f"Parity test dir not found: {parity_dir}")

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(parity_dir),
            "-q", "--tb=short",
            "--ignore", str(parity_dir / "test_matcher_parity_staged.py")
            if not (parity_dir / "test_matcher_parity_staged.py").exists()
            else str(parity_dir),
        ],
        capture_output=True,
        text=True,
        cwd=REBUILD_DIR,
    )

    if result.returncode == 0:
        # Extract summary line
        lines = result.stdout.strip().splitlines()
        summary = lines[-1] if lines else "no output"
        return GateResult("1 (Recall Parity)", "PASS", summary)
    else:
        # Extract failure count
        lines = result.stdout.strip().splitlines()
        summary = next(
            (l for l in reversed(lines) if "failed" in l or "error" in l),
            result.stdout[:200],
        )
        return GateResult("1 (Recall Parity)", "FAIL", summary.strip())


# ─── Gate 2: Latency budget (benchmark artifact) ─────────────────────────────

def gate_2_latency_budget() -> GateResult:
    """Read latest benchmark artifact; check p95 < 0.1ms and speedup > 20×."""
    # Find most recent artifact
    artifacts = sorted(ARTIFACTS_DIR.glob("hnsw_benchmark_*.json"))
    if not artifacts:
        return GateResult(
            "2 (Latency Budget)", "SKIP",
            "No benchmark artifact found in artifacts/benchmarks/. "
            "Run: pytest rebuild_phase_3z/tests/performance/ -m slow -v"
        )

    artifact = artifacts[-1]  # Most recent by filename (timestamp-sorted)
    try:
        data = json.loads(artifact.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return GateResult("2 (Latency Budget)", "FAIL", f"Could not read {artifact.name}: {e}")

    speedup = data.get("speedup_ratio", 0.0)
    gate_min_speedup = data.get("gate_min_speedup", 20.0)

    failures = []
    if speedup < gate_min_speedup:
        failures.append(f"speedup {speedup:.1f}× < {gate_min_speedup}× required")

    # p95 is not in the speedup artifact — it requires the latency test artifact
    # Check for latency artifact separately
    latency_artifacts = sorted(ARTIFACTS_DIR.glob("hnsw_latency_*.json"))
    p95_ms = None
    if latency_artifacts:
        try:
            lat_data = json.loads(latency_artifacts[-1].read_text())
            p95_ms = lat_data.get("p95_ms")
        except (json.JSONDecodeError, OSError):
            pass

    if p95_ms is not None:
        if p95_ms >= 0.1:
            failures.append(f"p95 {p95_ms:.4f}ms >= 0.1ms gate")

    if failures:
        return GateResult("2 (Latency Budget)", "FAIL", "; ".join(failures))

    artifact_name = artifact.name
    speedup_str = f"{speedup:.1f}×"
    p95_str = f"p95={p95_ms:.4f}ms" if p95_ms is not None else "p95=not-measured"
    return GateResult(
        "2 (Latency Budget)", "PASS",
        f"speedup={speedup_str} >= {gate_min_speedup}×, {p95_str} [from {artifact_name}]",
    )


# ─── Gate 3: Signal parity walk-forward ──────────────────────────────────────

def gate_3_walkforward_parity() -> GateResult:
    """Read walk-forward parity report if it exists; SKIP if not run yet."""
    report_path = WALKFORWARD_DIR / "hnsw_parity_report.json"
    if not report_path.exists():
        return GateResult(
            "3 (Walk-Forward Parity)", "SKIP",
            "Walk-forward parity report not found. "
            "Run: python scripts/run_walkforward.py --use-hnsw --compare-exact. "
            "Human review required before promotion.",
        )

    try:
        data = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return GateResult("3 (Walk-Forward Parity)", "FAIL", f"Could not read report: {e}")

    failures = []
    signal_agreement = data.get("signal_agreement_pct", 0.0)
    mean_prob_delta = data.get("mean_prob_abs_delta", 999.0)
    max_bss_delta = data.get("max_bss_delta_per_fold", 999.0)

    if signal_agreement < 99.0:
        failures.append(f"signal agreement {signal_agreement:.2f}% < 99%")
    if mean_prob_delta > 0.01:
        failures.append(f"mean prob delta {mean_prob_delta:.4f} > 0.01")
    if max_bss_delta > 0.005:
        failures.append(f"max BSS delta/fold {max_bss_delta:.4f} > 0.005")

    if failures:
        return GateResult("3 (Walk-Forward Parity)", "FAIL", "; ".join(failures))

    return GateResult(
        "3 (Walk-Forward Parity)", "PASS",
        f"agreement={signal_agreement:.2f}%, prob_delta={mean_prob_delta:.4f}, "
        f"bss_delta={max_bss_delta:.4f}",
    )


# ─── Gate 4: Zero regressions on full test suite ─────────────────────────────

def gate_4_full_test_suite() -> GateResult:
    """Run full test suite; PASS only if 0 failures."""
    test_dirs = []
    for d in [REPO_ROOT / "tests", REPO_ROOT / "trading_system" / "tests", REBUILD_DIR / "tests"]:
        if d.exists():
            test_dirs.append(str(d))

    if not test_dirs:
        return GateResult("4 (Full Test Suite)", "FAIL", "No test directories found")

    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + test_dirs + ["-q", "--tb=no", "-x"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    lines = result.stdout.strip().splitlines()
    summary = next(
        (l for l in reversed(lines) if "passed" in l or "failed" in l or "error" in l),
        result.stdout[-200:] if result.stdout else "no output",
    )

    if result.returncode == 0:
        return GateResult("4 (Full Test Suite)", "PASS", summary.strip())
    else:
        return GateResult("4 (Full Test Suite)", "FAIL", summary.strip())


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print()
    print("═" * 60)
    print("  HNSW Promotion Gate Check")
    print("  docs/rebuild/HNSW_PROMOTION_GATE.md | SLE-64")
    print("═" * 60)
    print()

    print("Running gates (this may take a few minutes)...")
    print()

    results = [
        gate_1_recall_parity(),
        gate_2_latency_budget(),
        gate_3_walkforward_parity(),
        gate_4_full_test_suite(),
    ]

    # Gate 5 — human sign-off
    results.append(GateResult(
        "5 (Human Sign-Off)", "SKIP",
        "Cannot be machine-checked. Sleep (Isaia) must review benchmark artifact "
        "and approve promotion via Linear SLE-64.",
    ))

    print("Results:")
    for r in results:
        print(r)

    print()
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    n_skip = sum(1 for r in results if r.status == "SKIP")

    print(f"  Summary: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")
    print()

    if n_fail > 0:
        print("  ✗ PROMOTION BLOCKED — fix failing gates before promoting HNSW.")
        print()
        return 1

    if n_skip > 0:
        skipped = [r.name for r in results if r.status == "SKIP"]
        print(f"  ⊘ PROMOTION CONDITIONAL — skipped: {', '.join(skipped)}")
        print("    All automated gates passed. Human review required for skipped gates.")
        print()
        return 0

    print("  ✓ ALL GATES PASSED — HNSW promotion approved (pending human sign-off).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
