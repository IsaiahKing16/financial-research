"""
manifest.py — Immutable run manifests for provenance and prior-run retrieval.

Every overnight/sweep/walkforward run produces a manifest that records:
  - run_id, timestamps, git SHA, data version
  - config used, phases completed/failed
  - best BSS achieved, artifact paths

Manifests enable:
  - Full provenance (governance requirement)
  - Prior-run context loading (proactive memory injection)
  - Cache invalidation via data_version comparison
"""

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from pattern_engine.reliability import atomic_write_json, safe_read_json


def compute_data_version(tickers: list[str], feature_cols: list[str]) -> str:
    """Deterministic fingerprint of the data pipeline configuration.

    Changes when the ticker universe or feature column definitions change,
    signaling that cached processed data may be stale.
    """
    payload = json.dumps(
        {"tickers": sorted(tickers), "features": sorted(feature_cols)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def generate_run_id() -> str:
    """Generate a human-readable run ID: YYYYMMDD_HHMMSS_<short_uuid>."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{ts}_{short}"


@dataclass
class RunManifest:
    """Immutable record of a single run's provenance."""

    run_id: str = ""
    mode: str = ""                   # "static", "bayesian", "walkforward", "live"
    started_at: str = ""
    ended_at: str = ""
    git_sha: str = ""
    data_version: str = ""
    config_hash: str = ""
    phases_completed: int = 0
    phases_failed: int = 0
    phases_partial: int = 0
    best_bss: float | None = None
    total_folds: int = 0
    elapsed_minutes: float = 0.0
    artifact_paths: dict = field(default_factory=dict)
    notes: str = ""

    def save(self, runs_dir: str = "data/runs") -> Path:
        """Save manifest to data/runs/<run_id>/manifest.json."""
        run_path = Path(runs_dir) / self.run_id
        run_path.mkdir(parents=True, exist_ok=True)
        manifest_path = run_path / "manifest.json"
        atomic_write_json(manifest_path, asdict(self))
        return manifest_path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        """Load a manifest from JSON file."""
        data = safe_read_json(path)
        # Filter to only known fields
        known = {f.name for f in __import__("dataclasses").fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def load_prior_context(runs_dir: str = "data/runs",
                       n_recent: int = 5) -> dict:
    """Load recent run manifests for proactive context injection.

    Returns a dict with:
      - best_config_hash: config hash from the best prior run
      - best_bss: best BSS achieved across recent runs
      - failed_runs: list of run_ids that had all phases fail
      - last_data_version: data version from most recent run
      - recent_manifests: list of the N most recent RunManifest dicts
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return {
            "best_config_hash": None,
            "best_bss": None,
            "failed_runs": [],
            "last_data_version": None,
            "recent_manifests": [],
        }

    # Find all manifest.json files, sort by directory name (timestamp-based)
    manifest_files = sorted(
        runs_path.glob("*/manifest.json"),
        key=lambda p: p.parent.name,
        reverse=True,
    )[:n_recent]

    manifests = []
    for mf in manifest_files:
        try:
            m = RunManifest.load(mf)
            manifests.append(m)
        except Exception:
            continue

    if not manifests:
        return {
            "best_config_hash": None,
            "best_bss": None,
            "failed_runs": [],
            "last_data_version": None,
            "recent_manifests": [],
        }

    # Find best BSS
    valid_bss = [(m.config_hash, m.best_bss) for m in manifests
                 if m.best_bss is not None]
    best_hash, best_bss = max(valid_bss, key=lambda x: x[1]) if valid_bss else (None, None)

    # Find failed runs
    failed = [m.run_id for m in manifests
              if m.phases_completed == 0 and m.phases_failed > 0]

    return {
        "best_config_hash": best_hash,
        "best_bss": best_bss,
        "failed_runs": failed,
        "last_data_version": manifests[0].data_version if manifests else None,
        "recent_manifests": [asdict(m) for m in manifests],
    }


def check_data_staleness(current_version: str,
                         processed_dir: str = "data/processed") -> dict:
    """Check if cached processed data is stale vs current pipeline config.

    Returns:
        dict with 'stale' bool, 'cached_version', 'current_version'
    """
    version_file = Path(processed_dir) / "data_version.json"
    if not version_file.exists():
        return {"stale": True, "cached_version": None,
                "current_version": current_version,
                "reason": "No cached data version found"}

    cached = safe_read_json(version_file)
    cached_ver = cached.get("version", "")

    if cached_ver != current_version:
        return {"stale": True, "cached_version": cached_ver,
                "current_version": current_version,
                "reason": "Ticker universe or feature definitions changed"}

    return {"stale": False, "cached_version": cached_ver,
            "current_version": current_version, "reason": ""}


def save_data_version(version: str, tickers: list[str],
                      feature_cols: list[str],
                      processed_dir: str = "data/processed") -> None:
    """Save current data version fingerprint alongside processed data."""
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(processed_dir) / "data_version.json", {
        "version": version,
        "tickers": sorted(tickers),
        "n_features": len(feature_cols),
        "created_at": datetime.now().isoformat(),
    })
