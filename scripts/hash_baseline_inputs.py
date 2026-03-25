"""
hash_baseline_inputs.py — Automated dataset fingerprinting for Phase 3Z parity.

Generates SHA-256 hashes for all data files and compares against the frozen
baseline manifest. Exit code 0 = parity confirmed, exit code 1 = drift detected.

Usage:
    python scripts/hash_baseline_inputs.py                    # verify against manifest
    python scripts/hash_baseline_inputs.py --generate         # generate new manifest
    python scripts/hash_baseline_inputs.py --manifest PATH    # use custom manifest path

Design doc: docs/rebuild/PHASE_3Z_EXECUTION_PLAN.md §9
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "rebuild_phase_3z"
    / "artifacts"
    / "manifests"
    / "baseline_manifest_20260321.json"
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_hashes() -> dict:
    """Hash all data CSVs and cached signals."""
    hashes = {}

    # Data directory CSVs
    if DATA_DIR.exists():
        for csv_file in sorted(DATA_DIR.glob("*.csv")):
            key = f"data/{csv_file.name}"
            hashes[key] = sha256_file(csv_file)

    # Cached signals
    cached_signals = RESULTS_DIR / "cached_signals_2024.csv"
    if cached_signals.exists():
        hashes["results/cached_signals_2024.csv"] = sha256_file(cached_signals)

    return hashes


def generate_manifest(output_path: Path) -> None:
    """Generate a new baseline manifest with current hashes."""
    hashes = collect_hashes()
    manifest = {
        "manifest_version": "1.0.0",
        "description": "Auto-generated baseline hashes",
        "algorithm": "sha256",
        "file_count": len(hashes),
        "dataset_hashes": hashes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {output_path}")
    print(f"  Files hashed: {len(hashes)}")


def verify_against_manifest(manifest_path: Path) -> bool:
    """Compare current hashes against frozen manifest. Returns True if parity holds."""
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    frozen_hashes = manifest.get("dataset_hashes", {})
    # Filter out non-hash keys like "algorithm"
    frozen_hashes = {
        k: v for k, v in frozen_hashes.items() if k.startswith(("data/", "results/"))
    }

    current_hashes = collect_hashes()

    # Check for mismatches
    mismatches = []
    missing_in_current = []
    new_files = []

    for key, expected_hash in frozen_hashes.items():
        if key not in current_hashes:
            missing_in_current.append(key)
        elif current_hashes[key] != expected_hash:
            mismatches.append(
                {
                    "file": key,
                    "expected": expected_hash[:16] + "...",
                    "actual": current_hashes[key][:16] + "...",
                }
            )

    for key in current_hashes:
        if key not in frozen_hashes:
            new_files.append(key)

    # Report
    print("=" * 60)
    print("  Phase 3Z Baseline Parity Check")
    print("=" * 60)
    print(f"  Manifest:  {manifest_path.name}")
    print(f"  Expected:  {len(frozen_hashes)} files")
    print(f"  Found:     {len(current_hashes)} files")
    print()

    if not mismatches and not missing_in_current:
        print("  STATUS: PARITY CONFIRMED [OK]")
        print(f"  All {len(frozen_hashes)} files match frozen baseline.")
        if new_files:
            print(f"\n  NOTE: {len(new_files)} new files not in manifest:")
            for f in new_files:
                print(f"    + {f}")
        print("=" * 60)
        return True
    else:
        print("  STATUS: DRIFT DETECTED [FAIL]")
        if mismatches:
            print(f"\n  MISMATCHES ({len(mismatches)}):")
            for m in mismatches:
                print(f"    {m['file']}")
                print(f"      expected: {m['expected']}")
                print(f"      actual:   {m['actual']}")
        if missing_in_current:
            print(f"\n  MISSING ({len(missing_in_current)}):")
            for f in missing_in_current:
                print(f"    - {f}")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3Z baseline dataset fingerprinting"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate new manifest instead of verifying",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manifest file (default: {DEFAULT_MANIFEST.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for --generate (default: same as --manifest)",
    )
    args = parser.parse_args()

    if args.generate:
        output = args.output or args.manifest
        generate_manifest(output)
        sys.exit(0)
    else:
        ok = verify_against_manifest(args.manifest)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
