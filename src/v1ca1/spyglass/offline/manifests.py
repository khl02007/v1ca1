"""Safe manifests for database-free Spyglass-equivalent analysis runs."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from v1ca1.spyglass.selection import canonical_json


DEFAULT_SCRATCH_ROOT = Path("/stelmo/kyu/analysis/spyglass")
MANIFEST_SCHEMA_VERSION = 1
CAMPAIGN_MANIFEST_FILENAME = "manifest.json"
SESSION_MANIFEST_FILENAME = "session_manifest.json"


def _path_component(value: Any, *, name: str) -> str:
    """Return one non-empty path component without traversal."""
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return value


def get_run_dir(
    run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> Path:
    """Return the guarded directory for one explicit campaign identifier."""
    run_id = _path_component(run_id, name="run_id")
    root = Path(scratch_root).expanduser().resolve(strict=False)
    return root / "runs" / run_id


def get_session_dir(
    run_dir: Path,
    *,
    animal_name: str,
    date: str,
) -> Path:
    """Return one guarded session directory within a campaign."""
    animal_name = _path_component(animal_name, name="animal_name")
    date = _path_component(date, name="date")
    guarded_run_dir = Path(run_dir).resolve(strict=False)
    session_dir = guarded_run_dir / animal_name / date
    if not session_dir.resolve(strict=False).is_relative_to(guarded_run_dir):
        raise ValueError("Session directory escapes the selected run directory.")
    return session_dir


def relative_run_path(path: Path, *, run_dir: Path) -> str:
    """Return a portable run-relative path, rejecting directory escapes."""
    guarded_run_dir = Path(run_dir).resolve(strict=False)
    resolved = Path(path).resolve(strict=False)
    if not resolved.is_relative_to(guarded_run_dir):
        raise ValueError(f"Artifact path escapes the run directory: {path}")
    return resolved.relative_to(guarded_run_dir).as_posix()


def resolve_run_path(value: str, *, run_dir: Path) -> Path:
    """Resolve one manifest path while requiring it to remain in the run."""
    relative = Path(str(value))
    if relative.is_absolute():
        raise ValueError("Manifest artifact paths must be run-relative.")
    return_path = (Path(run_dir).resolve(strict=False) / relative).resolve(
        strict=False
    )
    if not return_path.is_relative_to(Path(run_dir).resolve(strict=False)):
        raise ValueError(f"Manifest path escapes the run directory: {value!r}")
    return return_path


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one analysis artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nwb_fingerprint(nwb_path: Path, nwbfile: Any) -> dict[str, Any]:
    """Return cheap source provenance without hashing a potentially huge NWB."""
    path = Path(nwb_path).expanduser().resolve(strict=True)
    stat = path.stat()
    units = getattr(nwbfile, "units", None)
    return {
        "resolved_path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "nwb_identifier": str(getattr(nwbfile, "identifier", "")),
        "units_object_id": (
            None if units is None else str(getattr(units, "object_id", ""))
        ),
        "full_file_sha256": None,
    }


def code_provenance() -> dict[str, Any]:
    """Return lightweight version-control provenance without enforcing a pin."""
    import v1ca1

    repository = Path(__file__).resolve().parents[4]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "v1ca1_version": str(getattr(v1ca1, "__version__", "unknown")),
        "v1ca1_git_commit": commit,
        "v1ca1_git_dirty": dirty,
    }


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json_once(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomically create one JSON document without overwriting an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary manifest: {temporary}")
    try:
        temporary.write_text(
            json.dumps(
                json.loads(canonical_json(payload)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _replace_campaign_manifest(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomically update only the append-only campaign index."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Campaign update is already staged: {temporary}")
    try:
        temporary.write_text(
            json.dumps(
                json.loads(canonical_json(payload)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object as a plain dictionary."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be an object: {path}")
    return payload


def prepare_campaign(
    *,
    run_id: str,
    analysis_parameters: Mapping[str, Any],
    source_identity_policy: Mapping[str, Any],
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any]]:
    """Create or validate an append-only offline analysis campaign."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    manifest_path = run_dir / CAMPAIGN_MANIFEST_FILENAME
    expected_parameters = json.loads(canonical_json(analysis_parameters))
    expected_policy = json.loads(canonical_json(source_identity_policy))
    if manifest_path.exists():
        manifest = load_campaign_manifest(
            run_id,
            scratch_root=scratch_root,
            require_artifacts=False,
        )
        if manifest["analysis_parameters"] != expected_parameters:
            raise ValueError(
                "Existing campaign uses different analysis parameters; "
                "use a new run_id."
            )
        if manifest["source_identity_policy"] != expected_policy:
            raise ValueError(
                "Existing campaign uses a different unit identity policy; "
                "use a new run_id."
            )
        return run_dir, manifest
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory exists without a valid campaign manifest: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_utc": utc_now(),
        "updated_at_utc": None,
        "status": "in_progress",
        "code_provenance": code_provenance(),
        "analysis_parameters": expected_parameters,
        "source_identity_policy": expected_policy,
        "sessions": [],
    }
    write_json_once(manifest, manifest_path)
    return run_dir, manifest


def append_session_manifest(
    campaign: Mapping[str, Any],
    session_manifest: Mapping[str, Any],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Append one completed session to the campaign without replacing a session."""
    session_key = (
        str(session_manifest["animal_name"]),
        str(session_manifest["date"]),
    )
    session_path = get_session_dir(
        run_dir,
        animal_name=session_key[0],
        date=session_key[1],
    ) / SESSION_MANIFEST_FILENAME
    manifest_path = Path(run_dir) / CAMPAIGN_MANIFEST_FILENAME
    lock_path = Path(run_dir) / ".manifest.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        current = load_json(manifest_path)
        supplied = dict(campaign)
        for field in ("run_id", "analysis_parameters", "source_identity_policy"):
            if canonical_json(current.get(field)) != canonical_json(
                supplied.get(field)
            ):
                raise ValueError(
                    f"Current campaign {field} differs from the supplied snapshot."
                )
        sessions = list(current.get("sessions", ()))
        if any(
            (str(row.get("animal_name")), str(row.get("date"))) == session_key
            for row in sessions
        ):
            raise FileExistsError(
                f"Campaign already contains session {session_key!r}."
            )
        sessions.append(
            {
                "animal_name": session_key[0],
                "date": session_key[1],
                "nwb_file_name": str(session_manifest["nwb_file_name"]),
                "nwb_path": str(session_manifest["nwb_path"]),
                "session_manifest_path": relative_run_path(
                    session_path,
                    run_dir=run_dir,
                ),
                "status": str(session_manifest["status"]),
            }
        )
        current["sessions"] = sorted(
            sessions,
            key=lambda row: (str(row["animal_name"]), str(row["date"])),
        )
        current["updated_at_utc"] = utc_now()
        _replace_campaign_manifest(current, manifest_path)
    return current


def _validate_artifact_records(
    session: Mapping[str, Any],
    *,
    run_dir: Path,
    require_artifacts: bool,
) -> None:
    """Validate artifact path containment and optional existence."""
    path_fields = {
        "movement_firing_rate": (
            "artifact_dir",
            "firing_rate_path",
            "movement_intervals_path",
        ),
        "path_specific_place_tuning_curve": ("tuning_curve_path",),
        "path_specific_place_stability": ("stability_path",),
    }
    artifacts = session.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Session manifest artifacts must be a mapping.")
    for family, fields in path_fields.items():
        records = artifacts.get(family)
        if not isinstance(records, list):
            raise ValueError(f"Session artifact family {family!r} must be a list.")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Artifact record in {family!r} must be a mapping.")
            for field in fields:
                if field not in record:
                    raise ValueError(
                        f"Artifact record {family!r} is missing {field!r}."
                    )
                path = resolve_run_path(str(record[field]), run_dir=run_dir)
                if require_artifacts and not path.exists():
                    raise FileNotFoundError(f"Manifest artifact not found: {path}")


def load_session_manifest(
    path: Path,
    *,
    run_dir: Path,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    """Load and validate one offline session manifest."""
    manifest_path = Path(path).resolve(strict=require_artifacts)
    guarded_run_dir = Path(run_dir).resolve(strict=False)
    if not manifest_path.is_relative_to(guarded_run_dir):
        raise ValueError("Session manifest escapes the selected run directory.")
    session = load_json(manifest_path)
    if session.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported offline session manifest schema version.")
    if session.get("status") != "complete":
        raise ValueError("Offline session manifest is not complete.")
    _validate_artifact_records(
        session,
        run_dir=guarded_run_dir,
        require_artifacts=require_artifacts,
    )
    return session


def load_campaign_manifest(
    run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    require_artifacts: bool = True,
) -> dict[str, Any]:
    """Load a campaign and validate every registered session manifest."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    manifest = load_json(run_dir / CAMPAIGN_MANIFEST_FILENAME)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported offline campaign manifest schema version.")
    if str(manifest.get("run_id")) != str(run_id):
        raise ValueError("Campaign manifest run_id does not match its directory.")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("Campaign manifest sessions must be a list.")
    seen: set[tuple[str, str]] = set()
    for row in sessions:
        key = (str(row.get("animal_name")), str(row.get("date")))
        if key in seen:
            raise ValueError(f"Campaign contains duplicate session {key!r}.")
        seen.add(key)
        session_path = resolve_run_path(
            str(row.get("session_manifest_path", "")),
            run_dir=run_dir,
        )
        if require_artifacts or session_path.exists():
            session = load_session_manifest(
                session_path,
                run_dir=run_dir,
                require_artifacts=require_artifacts,
            )
            if (
                str(session["animal_name"]),
                str(session["date"]),
            ) != key:
                raise ValueError("Campaign and session manifest identities disagree.")
    return manifest


__all__ = [
    "CAMPAIGN_MANIFEST_FILENAME",
    "DEFAULT_SCRATCH_ROOT",
    "MANIFEST_SCHEMA_VERSION",
    "SESSION_MANIFEST_FILENAME",
    "append_session_manifest",
    "code_provenance",
    "file_sha256",
    "get_run_dir",
    "get_session_dir",
    "load_campaign_manifest",
    "load_session_manifest",
    "nwb_fingerprint",
    "prepare_campaign",
    "relative_run_path",
    "resolve_run_path",
    "utc_now",
    "write_json_once",
]
