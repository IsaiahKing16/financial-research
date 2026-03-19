"""
reliability.py — Shared utilities for crash-safe overnight execution.

Provides atomic file writes, advisory lock files, and timestamped
progress logging. Used by overnight.py, experiment_logging.py,
engine.py, and walkforward.py.

Design principle: every disk write should survive a mid-write crash.
The old file is preserved until the new one is fully written and fsynced.
"""

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path


def atomic_write(path: str | Path, data: str | bytes, mode: str = "w") -> None:
    """Write data to a file atomically via temp + fsync + rename.

    If the process crashes during write, the original file is preserved.
    On Windows, os.replace() is used (atomic on NTFS for same-volume).

    Args:
        path: target file path
        data: content to write (str for text, bytes for binary)
        mode: "w" for text, "wb" for binary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem = atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, obj: dict, indent: int = 2) -> None:
    """Atomically write a JSON object to a file."""
    atomic_write(path, json.dumps(obj, indent=indent, default=str))


def safe_read_json(path: str | Path, default: dict = None) -> dict:
    """Read a JSON file, returning default on missing or corrupt file.

    Args:
        path: JSON file path
        default: returned if file missing or corrupt (default: empty dict)

    Returns:
        Parsed JSON dict, or default
    """
    path = Path(path)
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # Corrupt checkpoint — warn and return default
        print(f"  WARNING: corrupt JSON at {path}: {e}")
        print(f"  Falling back to default: {default}")
        return default if default is not None else {}


class LockFile:
    """Advisory lock file to prevent concurrent overnight runs.

    Usage:
        with LockFile("data/results/overnight.lock") as lock:
            if not lock.acquired:
                print("Another run in progress")
                sys.exit(1)
            # ... do work ...

    On Windows, uses a simple PID-based lock file (no flock).
    Stale locks (PID no longer running) are automatically cleaned up.
    """

    def __init__(self, path: str | Path, stale_hours: float = 12.0):
        self.path = Path(path)
        self.stale_hours = stale_hours
        self.acquired = False
        self._pid = os.getpid()

    def __enter__(self) -> "LockFile":
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Check for existing lock and handle stale/dead PIDs
        if self.path.exists():
            lock_info = safe_read_json(self.path)
            old_pid = lock_info.get("pid")
            lock_time = lock_info.get("timestamp", "")

            # Check if lock holder is still running
            if old_pid and self._is_pid_alive(old_pid):
                print(f"  LOCK: Process {old_pid} is still running (locked at {lock_time})")
                self.acquired = False
                return self

            # Log why we're removing the stale lock
            if lock_time:
                try:
                    lock_dt = datetime.fromisoformat(lock_time)
                    age_hours = (datetime.now() - lock_dt).total_seconds() / 3600
                    if age_hours > self.stale_hours:
                        print(f"  LOCK: Stale lock detected ({age_hours:.1f}h old, PID {old_pid}). Removing.")
                    else:
                        print(f"  LOCK: Recent lock ({age_hours:.1f}h old, PID {old_pid} not running). Removing.")
                except (ValueError, TypeError):
                    print(f"  LOCK: Corrupt lock file. Removing.")

            # Remove stale lock
            try:
                self.path.unlink()
            except OSError:
                pass

        # Atomically create the lock file using exclusive create (no race window)
        lock_data = {
            "pid": self._pid,
            "timestamp": datetime.now().isoformat(),
            "hostname": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")),
        }
        try:
            with open(self.path, "x") as f:
                json.dump(lock_data, f)
                f.flush()
                os.fsync(f.fileno())
            self.acquired = True
        except FileExistsError:
            # Another process created the lock between our unlink and open
            self.acquired = False
        return self

    def __exit__(self, *args):
        if self.acquired:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.acquired = False

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Check if a process is running (cross-platform).

        On Windows, os.kill(pid, 0) sends CTRL_C_EVENT (signal.CTRL_C_EVENT == 0)
        to the process group — it does NOT check existence and will interrupt the
        calling process. Use ctypes OpenProcess/GetExitCodeProcess instead.

        On Unix, signal 0 is a null signal that only checks process existence.
        Guards against pid <= 0 (would target process group on Unix).
        Treats PermissionError as alive (process exists, different user).
        """
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            if sys.platform == "win32":
                import ctypes
                PROCESS_QUERY_INFORMATION = 0x0400
                STILL_ACTIVE = 259
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_INFORMATION, False, pid
                )
                if not handle:
                    return False
                exit_code = ctypes.c_ulong()
                ret = ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)
                )
                ctypes.windll.kernel32.CloseHandle(handle)
                return bool(ret) and exit_code.value == STILL_ACTIVE
            else:
                os.kill(pid, 0)  # Signal 0 = null check on Unix only
                return True
        except PermissionError:
            return True  # Process exists but owned by another user
        except (OSError, ProcessLookupError):
            return False


class ProgressLog:
    """Append-only timestamped progress log for overnight runs.

    Each entry is a single line: [timestamp] level: message
    Never truncated or overwritten — safe for post-mortem analysis.

    Usage:
        log = ProgressLog("data/results/overnight.log")
        log.info("Starting phase 3")
        log.error("Phase 3 failed: ValueError")
        log.info("Completed 4/8 phases")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {level}: {message}\n"
        with open(self.path, "a") as f:
            f.write(line)
            f.flush()

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def warn(self, message: str) -> None:
        self._write("WARN", message)

    def phase_start(self, phase_id: str, config_summary: str = "") -> None:
        self._write("PHASE_START", f"{phase_id} | {config_summary}")

    def phase_end(self, phase_id: str, duration_sec: float = 0,
                  status: str = "OK") -> None:
        self._write("PHASE_END", f"{phase_id} | {status} | {duration_sec:.1f}s")

    def fold_result(self, fold_label: str, bss: float = None,
                    error: str = None) -> None:
        if error:
            self._write("FOLD_FAIL", f"{fold_label} | {error}")
        else:
            sign = "+" if bss and bss > 0 else ""
            self._write("FOLD_OK", f"{fold_label} | BSS={sign}{bss:.6f}" if bss is not None
                         else f"{fold_label} | BSS=N/A")
