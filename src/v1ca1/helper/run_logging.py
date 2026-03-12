from __future__ import annotations

"""Helpers for recording reproducible script runs under one analysis session."""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from v1ca1 import __version__


def make_json_safe(value: Any) -> Any:
    """Convert Path-like and nested values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def get_run_log_dir(analysis_path: Path) -> Path:
    """Return the directory used to store one JSON log per script run."""
    return analysis_path / "v1ca1_log"


def get_git_commit() -> str | None:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def get_git_dirty() -> bool | None:
    """Return whether the current git worktree is dirty when available."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def write_run_log(
    analysis_path: Path,
    script_name: str,
    parameters: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> Path:
    """Write one run record to a unique JSON file for this session.

    Using one file per run avoids contention when multiple scripts are running
    simultaneously for the same session.
    """
    analysis_path.mkdir(parents=True, exist_ok=True)
    log_dir = get_run_log_dir(analysis_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp_utc = datetime.now(timezone.utc)
    timestamp_for_name = timestamp_utc.strftime("%Y%m%dT%H%M%S%fZ")
    script_slug = script_name.replace(".", "_")
    log_path = log_dir / (
        f"{script_slug}_{timestamp_for_name}_pid{os.getpid()}_{uuid4().hex[:8]}.json"
    )
    record = {
        "timestamp_utc": timestamp_utc.isoformat(),
        "script": script_name,
        "package_version": __version__,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "parameters": make_json_safe(parameters),
        "outputs": make_json_safe(outputs or {}),
    }
    with open(log_path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
        file.write("\n")
    return log_path
