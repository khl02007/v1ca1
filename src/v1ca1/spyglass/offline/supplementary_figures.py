"""Run shared supplementary-figure analyses from a Figure 2 campaign.

The campaign computes only database-free, NWB-backed artifacts: one V1 cvPCA
comparison, dark and AB epoch-motor summaries, and one empirical swap-tuning
comparison per manuscript session.  Every session pins its Figure 2 parent
manifest by path and SHA-256.  Invoke the CLI once per manuscript session with
the same ``--run-id`` to populate the full append-only campaign.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import fcntl
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures.datasets import get_processed_datasets
from v1ca1.spyglass import (
    cv_pca,
    epoch_motor_behavior,
    movement,
    path_specific_place,
    swap_tuning,
)
from v1ca1.spyglass.offline.figure_1 import (
    _curve_attributes,
    _tuning_selection,
)
from v1ca1.spyglass.offline.figure_1_full import FULL_W_CONFIGURATION_NAME
from v1ca1.spyglass.offline.figure_2 import (
    FIGURE_2_LIGHT_TEST_CONDITION,
    FIGURE_2_LIGHT_TEST_EPOCH,
    FIGURE_2_LIGHT_TRAIN_CONDITION,
    FIGURE_2_LIGHT_TRAIN_EPOCH,
    load_figure_2_campaign,
    load_figure_2_session_manifest,
)
from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    DEFAULT_SCRATCH_ROOT,
    MANIFEST_SCHEMA_VERSION,
    SESSION_MANIFEST_FILENAME,
    code_provenance,
    file_sha256,
    get_run_dir,
    get_session_dir,
    load_json,
    nwb_fingerprint,
    prepare_campaign,
    relative_run_path,
    resolve_run_path,
    utc_now,
    write_json_once,
)
from v1ca1.spyglass.offline.sources import (
    SOURCE_IDENTITY_POLICY,
    load_nwb_region_spikes,
    load_run_epoch_catalog_objects,
    select_run_epoch_catalog,
    validate_nwb_session_identity,
)
from v1ca1.spyglass.nwb import load_position
from v1ca1.spyglass.selection import (
    canonical_json,
    provenance_sha256,
    selection_uuid,
)
from v1ca1.spyglass.table_specs import (
    DEFAULT_MOVEMENT_PARAMETERS,
    LEGACY_TUNING_CURVE_PARAMETERS,
    MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS,
    MANUSCRIPT_V1_CV_PCA_PARAMETERS,
    MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS,
)


DEFAULT_PARENT_RUN_ID = "figure2-nwb-gpu-v1"
SUPPLEMENTARY_FIGURES_PIPELINE = "supplementary_figures"
SUPPLEMENTARY_FIGURES_REGION = "v1"
POSITION_ROLES = ("head", "body")
PRIMARY_POSITION_ROLE = "head"
ORIENTATION_REFERENCE_POSITION_ROLE = "body"
ARTIFACT_FAMILIES = (
    "cv_pca",
    "epoch_motor_behavior",
    "swap_tuning_curve_comparison",
)
_EXPECTED_SESSIONS = {
    (str(animal_name), str(date))
    for animal_name, date, _dark_epoch in get_processed_datasets()
}
_CV_PCA_PATH_FIELDS = (
    "result_path",
    "summary_path",
    "spectrum_path",
    "selected_units_path",
    "lap_assignments_path",
    "trajectory_qc_path",
    "manifest_path",
)
_EPOCH_MOTOR_PATH_FIELDS = (
    "artifact_manifest_path",
    "distribution_summary_path",
    "progression_summary_path",
    "trajectory_qc_path",
)
_SWAP_TUNING_PATH_FIELDS = (
    "artifact_manifest_path",
    "selected_units_path",
    "summary_path",
    "result_path",
)
_LIGHT_TEST_TUNING_PARAMETER_FIELDS = (
    "tuning_curve_param_name",
    "binning_mode",
    "place_bin_size_cm",
    "position_bin_count",
    "gaussian_smoothing_sigma_bins",
)


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: Any,
) -> dict[str, Any]:
    """Return exactly one record matching direct or selection fields."""
    matches = []
    for raw_record in records:
        record = dict(raw_record)
        selection = record.get("selection", {})
        if all(
            str(record.get(name, selection.get(name))) == str(value)
            for name, value in selectors.items()
        ):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {selectors!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _native(value: Any) -> Any:
    """Convert NumPy containers to JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    return value


def _seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a deterministic checksum to one artifact record."""
    output = _native(dict(record))
    if "record_sha256" in output:
        raise ValueError("Cannot reseal an artifact record.")
    output["record_sha256"] = provenance_sha256(output)
    return output


def _relative_artifact_record(
    record: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    run_dir: Path,
    path_fields: Sequence[str],
) -> dict[str, Any]:
    """Store guarded relative paths and SHA-256 values for one bundle."""
    output = dict(record)
    artifact_dir = Path(paths["artifact_dir"])
    output["artifact_dir"] = relative_run_path(artifact_dir, run_dir=run_dir)
    hashes = {}
    for field in path_fields:
        path = Path(paths[field])
        if not path.is_file():
            raise FileNotFoundError(f"Artifact writer omitted {field!r}: {path}")
        output[field] = relative_run_path(path, run_dir=run_dir)
        hashes[field] = file_sha256(path)
    output["artifact_sha256"] = hashes
    return _seal_record(output)


def _verify_record(record: Mapping[str, Any], *, run_dir: Path) -> None:
    """Verify one sealed, computed, run-local supplementary artifact."""
    if str(record.get("artifact_origin")) != "computed":
        raise ValueError("Supplementary artifacts must be computed de novo.")
    unhashed = dict(record)
    observed = str(unhashed.pop("record_sha256", ""))
    if not observed or provenance_sha256(unhashed) != observed:
        raise ValueError("Supplementary artifact record checksum mismatch.")
    hashes = record.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("Supplementary artifact record lacks file checksums.")
    for field, digest in hashes.items():
        path = resolve_run_path(str(record[field]), run_dir=run_dir)
        if not path.is_file() or file_sha256(path) != str(digest):
            raise ValueError(f"Supplementary artifact checksum mismatch: {path}")


def build_figure_2_parent_snapshot(
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Freeze a complete Figure 2 campaign and every session manifest hash."""
    parent_run_dir, campaign, sessions = load_figure_2_campaign(
        parent_run_id,
        scratch_root=scratch_root,
    )
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list) or len(summaries) != len(sessions):
        raise ValueError("Figure 2 parent has an incomplete session index.")
    observed = {
        (str(row.get("animal_name")), str(row.get("date"))) for row in summaries
    }
    if observed != _EXPECTED_SESSIONS:
        raise ValueError(
            "Supplementary analyses require exactly the manuscript sessions; "
            f"observed {sorted(observed)!r}."
        )
    snapshot_sessions = []
    for summary in summaries:
        if str(summary.get("status")) != "complete":
            raise ValueError("Every Figure 2 parent session must be complete.")
        manifest_path = resolve_run_path(
            str(summary["session_manifest_path"]),
            run_dir=parent_run_dir,
        )
        snapshot_sessions.append(
            {
                "animal_name": str(summary["animal_name"]),
                "date": str(summary["date"]),
                "session_manifest_path": str(summary["session_manifest_path"]),
                "session_manifest_sha256": file_sha256(manifest_path),
            }
        )
    return {
        "run_id": str(parent_run_id),
        "campaign_manifest_sha256": file_sha256(
            parent_run_dir / CAMPAIGN_MANIFEST_FILENAME
        ),
        "sessions": sorted(
            snapshot_sessions,
            key=lambda row: (row["animal_name"], row["date"]),
        ),
    }


def _parent_snapshot_from_campaign_or_sessions(
    campaign_or_sessions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract one common Figure 2 snapshot from a campaign or sessions."""
    if isinstance(campaign_or_sessions, Mapping):
        parameters = campaign_or_sessions.get("analysis_parameters", {})
        snapshot = (
            parameters.get("parent_figure_2")
            if isinstance(parameters, Mapping)
            else None
        )
    else:
        sessions = list(campaign_or_sessions)
        if not sessions:
            raise ValueError("At least one supplementary session is required.")
        snapshots = [row.get("parent_figure_2") for row in sessions]
        snapshot = snapshots[0]
        if any(
            canonical_json(value) != canonical_json(snapshot)
            for value in snapshots[1:]
        ):
            raise ValueError("Supplementary sessions use different Figure 2 parents.")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Supplementary campaign lacks its Figure 2 parent snapshot.")
    return dict(snapshot)


def load_parent_figure_2_sessions(
    campaign_or_sessions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, list[dict[str, Any]]]:
    """Load all checksum-pinned Figure 2 parent sessions in snapshot order."""
    snapshot = _parent_snapshot_from_campaign_or_sessions(campaign_or_sessions)
    current = build_figure_2_parent_snapshot(
        str(snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(snapshot):
        raise ValueError("Figure 2 parent changed after supplementary selection.")
    parent_run_dir = get_run_dir(str(snapshot["run_id"]), scratch_root=scratch_root)
    sessions = []
    for summary in snapshot["sessions"]:
        manifest_path = resolve_run_path(
            str(summary["session_manifest_path"]),
            run_dir=parent_run_dir,
        )
        if file_sha256(manifest_path) != str(summary["session_manifest_sha256"]):
            raise ValueError("Figure 2 parent session checksum changed.")
        session = load_figure_2_session_manifest(
            manifest_path,
            run_dir=parent_run_dir,
            scratch_root=scratch_root,
        )
        identity = (str(session["animal_name"]), str(session["date"]))
        expected = (str(summary["animal_name"]), str(summary["date"]))
        if identity != expected:
            raise ValueError("Figure 2 parent session identity changed.")
        sessions.append(session)
    return parent_run_dir, sessions


def build_supplementary_figures_configuration(
    parent_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable supplementary-analysis configuration."""
    return {
        "pipeline": SUPPLEMENTARY_FIGURES_PIPELINE,
        "parent_figure_2": dict(parent_snapshot),
        "region": SUPPLEMENTARY_FIGURES_REGION,
        "epochs": {
            "AB": FIGURE_2_LIGHT_TRAIN_EPOCH,
            "BA": FIGURE_2_LIGHT_TEST_EPOCH,
            "dark": "from_parent_figure_2_session",
        },
        "trajectory_types": list(TRAJECTORY_TYPES),
        "position_roles": list(POSITION_ROLES),
        "epoch_motor_position_roles": {
            "primary": PRIMARY_POSITION_ROLE,
            "orientation_reference": ORIENTATION_REFERENCE_POSITION_ROLE,
        },
        "cv_pca_parameters": dict(MANUSCRIPT_V1_CV_PCA_PARAMETERS),
        "epoch_motor_behavior_parameters": dict(
            MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS
        ),
        "epoch_motor_behavior_epochs": ["dark", "AB"],
        "movement_parameters": dict(DEFAULT_MOVEMENT_PARAMETERS),
        "light_test_tuning_curve_parameters": dict(
            LEGACY_TUNING_CURVE_PARAMETERS
        ),
        "swap_tuning_curve_comparison_parameters": dict(
            MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS
        ),
        "artifact_origin": "computed",
        "diagnostic_figures": False,
    }


def prepare_supplementary_figures_campaign(
    *,
    run_id: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create or validate one append-only supplementary campaign."""
    parent = build_figure_2_parent_snapshot(
        parent_run_id,
        scratch_root=scratch_root,
    )
    configuration = build_supplementary_figures_configuration(parent)
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    if (run_dir / CAMPAIGN_MANIFEST_FILENAME).exists():
        loaded_run_dir, campaign, _sessions = load_supplementary_figures_campaign(
            run_id,
            scratch_root=scratch_root,
        )
        if canonical_json(campaign.get("analysis_parameters")) != canonical_json(
            configuration
        ):
            raise ValueError(
                "Existing campaign uses different analysis parameters; "
                "use a new run_id."
            )
        if canonical_json(campaign.get("source_identity_policy")) != canonical_json(
            SOURCE_IDENTITY_POLICY
        ):
            raise ValueError(
                "Existing campaign uses a different source identity policy; "
                "use a new run_id."
            )
        return loaded_run_dir, campaign, parent
    run_dir, campaign = prepare_campaign(
        run_id=run_id,
        analysis_parameters=configuration,
        source_identity_policy=SOURCE_IDENTITY_POLICY,
        scratch_root=scratch_root,
    )
    return run_dir, campaign, parent


def _append_supplementary_session_manifest(
    campaign: Mapping[str, Any],
    session_manifest: Mapping[str, Any],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Append one session and its manifest checksum to the campaign index."""
    identity = (
        str(session_manifest["animal_name"]),
        str(session_manifest["date"]),
    )
    session_path = get_session_dir(
        run_dir,
        animal_name=identity[0],
        date=identity[1],
    ) / SESSION_MANIFEST_FILENAME
    if not session_path.is_file():
        raise FileNotFoundError(
            f"Supplementary session manifest was not written: {session_path}"
        )
    manifest_path = Path(run_dir) / CAMPAIGN_MANIFEST_FILENAME
    lock_path = Path(run_dir) / ".manifest.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        current = load_json(manifest_path)
        for field in ("run_id", "analysis_parameters", "source_identity_policy"):
            if canonical_json(current.get(field)) != canonical_json(
                campaign.get(field)
            ):
                raise ValueError(
                    f"Current campaign {field} differs from the supplied snapshot."
                )
        sessions = list(current.get("sessions", ()))
        if any(
            (str(row.get("animal_name")), str(row.get("date"))) == identity
            for row in sessions
        ):
            raise FileExistsError(
                f"Campaign already contains session {identity!r}."
            )
        sessions.append(
            {
                "animal_name": identity[0],
                "date": identity[1],
                "nwb_file_name": str(session_manifest["nwb_file_name"]),
                "nwb_path": str(session_manifest["nwb_path"]),
                "session_manifest_path": relative_run_path(
                    session_path,
                    run_dir=run_dir,
                ),
                "session_manifest_sha256": file_sha256(session_path),
                "status": str(session_manifest["status"]),
            }
        )
        current["sessions"] = sorted(
            sessions,
            key=lambda row: (str(row["animal_name"]), str(row["date"])),
        )
        current["updated_at_utc"] = utc_now()
        temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
        if temporary.exists():
            raise FileExistsError(
                f"Campaign update is already staged: {temporary}"
            )
        try:
            temporary.write_text(
                canonical_json(current) + "\n",
                encoding="utf-8",
            )
            temporary.replace(manifest_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return current


def _row_fields(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Copy selected catalog provenance fields into JSON-safe scalars."""
    return {name: _native(row[name]) for name in fields if name in row}


def _catalog_snapshot(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the NWB rows needed to reconstruct one run epoch."""
    pointer_fields = (
        "nwb_file_name",
        "source_table_path",
        "source_table_object_id",
        "source_object_path",
        "source_object_id",
    )
    return {
        "epoch_interval": _row_fields(
            selection["epoch_row"],
            (
                "epoch",
                "epoch_type",
                "condition",
                "is_light",
                "start_time",
                "stop_time",
                "time_unit",
                "time_reference",
                "metadata_table_path",
                "metadata_table_object_id",
                *pointer_fields,
            ),
        ),
        "positions": {
            role: _row_fields(
                row,
                (
                    "epoch",
                    "position_series_name",
                    "position_role",
                    "spatial_unit",
                    "source_row_index",
                    "start_index",
                    "stop_index_exclusive",
                    "sample_count",
                    "analysis_start_offset_samples",
                    "start_time",
                    "stop_time",
                    "first_frame",
                    "last_frame",
                    "video_series_name",
                    *pointer_fields,
                ),
            )
            for role, row in selection["position_rows"].items()
        },
        "trajectory_intervals": {
            trajectory_type: _row_fields(
                row,
                (
                    "epoch",
                    "trajectory_type",
                    "interval_list_name",
                    "interval_count",
                    "time_unit",
                    "time_reference",
                    *pointer_fields,
                ),
            )
            for trajectory_type, row in selection["trajectory_rows"].items()
        },
        "wtrack_graphs": {
            configuration_name: _row_fields(
                row,
                (
                    "configuration_name",
                    "coordinate_unit",
                    "use_hmm",
                    "source_row_index",
                    *pointer_fields,
                ),
            )
            for configuration_name, row in selection["graph_rows"].items()
        },
    }


def _load_cv_pca_positions(
    nwbfile: Any,
    selections: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Load untrimmed dark and AB head positions for cvPCA offset handling."""
    return {
        role: load_position(
            nwbfile,
            selections[role]["position_rows"][PRIMARY_POSITION_ROLE],
            apply_analysis_offset=False,
        )
        for role in ("dark", "light_train")
    }


def _record_value(record: Mapping[str, Any], field: str) -> Any:
    """Return one direct or nested selection value."""
    if field in record:
        return record[field]
    selection = record.get("selection")
    return selection.get(field) if isinstance(selection, Mapping) else None


def _record_path(
    record: Mapping[str, Any],
    field: str,
    *,
    run_dir: Path,
) -> Path:
    """Resolve and checksum one direct or nested parent artifact pointer."""
    artifacts = record.get("artifacts")
    if isinstance(artifacts, Mapping) and field in artifacts:
        metadata = artifacts[field]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Parent artifact metadata {field!r} is malformed.")
        relative_path = metadata.get("relative_path")
        digest = metadata.get("sha256")
    else:
        relative_path = record.get(field)
        hashes = record.get("artifact_sha256")
        digest = hashes.get(field) if isinstance(hashes, Mapping) else None
    if relative_path is None or digest is None:
        raise ValueError(f"Parent artifact lacks a checksummed {field!r} path.")
    path = resolve_run_path(str(relative_path), run_dir=run_dir)
    if not path.is_file() or file_sha256(path) != str(digest):
        raise ValueError(f"Parent artifact checksum mismatch for {field!r}.")
    return path


def _parent_session_pointer(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
) -> dict[str, str]:
    """Return the compact parent pointer stored in a child session."""
    summary = _one_record(
        parent_snapshot["sessions"],
        label="Figure 2 parent session",
        animal_name=animal_name,
        date=date,
    )
    return {
        "run_id": str(parent_snapshot["run_id"]),
        "session_manifest_path": str(summary["session_manifest_path"]),
        "session_manifest_sha256": str(summary["session_manifest_sha256"]),
    }


def _load_one_parent_session(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    """Load one checksum-pinned parent session selected by identity."""
    parent_run_dir, sessions = load_parent_figure_2_sessions(
        {
            "analysis_parameters": {
                "parent_figure_2": dict(parent_snapshot),
            }
        },
        scratch_root=scratch_root,
    )
    session = _one_record(
        sessions,
        label="loaded Figure 2 parent session",
        animal_name=animal_name,
        date=date,
    )
    pointer = _parent_session_pointer(
        parent_snapshot,
        animal_name=animal_name,
        date=date,
    )
    return parent_run_dir, session, pointer


def _validate_epoch_selection(
    selection: Mapping[str, Any],
    *,
    epoch: str,
    condition: str,
    is_light: bool,
) -> None:
    """Require one catalog selection to describe its fixed condition."""
    row = selection["epoch_row"]
    if (
        str(row.get("epoch")) != str(epoch)
        or str(row.get("epoch_type")) != "run"
        or str(row.get("condition", "")).casefold() != str(condition).casefold()
        or row.get("is_light") is not is_light
    ):
        raise ValueError(
            f"NWB epoch {epoch!r} does not describe condition {condition!r}."
        )


def _verify_parent_catalog_snapshot(
    parent_session: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require all Figure 2-selected head, lap, and graph rows to be unchanged."""
    parent_sources = parent_session.get("nwb_sources")
    if not isinstance(parent_sources, Mapping) or set(parent_sources) != set(snapshots):
        raise ValueError("Figure 2 parent NWB source snapshots are incomplete.")
    for epoch, snapshot in snapshots.items():
        parent = parent_sources[epoch]
        selected = {
            **snapshot,
            "positions": {"head": snapshot["positions"]["head"]},
        }
        if canonical_json(parent) != canonical_json(selected):
            raise ValueError(
                f"NWB catalog sources changed relative to Figure 2 for {epoch}."
            )


def _movement_context(
    parent_run_dir: Path,
    parent_session: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> dict[str, dict[str, Any]]:
    """Load dark, AB, and BA movement bundles from pinned parent records."""
    epochs = parent_session["epochs"]
    parent_artifacts = parent_session["parent_artifacts"]
    initial_run_dir = get_run_dir(
        str(parent_artifacts["initial_run_id"]),
        scratch_root=scratch_root,
    )
    records_and_roots = {
        "dark": (
            dict(parent_artifacts["dark_movement_firing_rate"]),
            initial_run_dir,
        ),
        "light_train": (
            _one_record(
                parent_session["artifacts"]["movement_firing_rate"],
                label="AB MovementFiringRate",
                epoch=epochs["AB"],
                region=SUPPLEMENTARY_FIGURES_REGION,
            ),
            parent_run_dir,
        ),
        "light_test": (
            _one_record(
                parent_session["artifacts"]["movement_firing_rate"],
                label="BA MovementFiringRate",
                epoch=epochs["BA"],
                region=SUPPLEMENTARY_FIGURES_REGION,
            ),
            parent_run_dir,
        ),
    }
    output = {}
    for role, (record, source_run_dir) in records_and_roots.items():
        firing_rate_path = _record_path(
            record,
            "firing_rate_path",
            run_dir=source_run_dir,
        )
        movement_intervals_path = _record_path(
            record,
            "movement_intervals_path",
            run_dir=source_run_dir,
        )
        loaded = {
            "table": movement.load_movement_firing_rate_artifact(firing_rate_path),
            "movement_intervals": movement.load_movement_interval_artifact(
                movement_intervals_path
            ),
        }
        movement.validate_movement_artifacts(
            loaded["table"],
            loaded["movement_intervals"],
        )
        loaded["analysis_status"] = (
            "no_units"
            if loaded["table"].empty
            else str(loaded["table"]["firing_rate_status"].iloc[0])
        )
        output[role] = {
            "record": record,
            "source_run_dir": source_run_dir,
            "firing_rate_path": firing_rate_path,
            "movement_intervals_path": movement_intervals_path,
            **loaded,
        }
    return output


def _tuning_context(
    parent_run_dir: Path,
    parent_session: Mapping[str, Any],
    *,
    scratch_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Resolve the eight dark/AB all-trial tuning curves from Figure 2."""
    epochs = parent_session["epochs"]
    parent_artifacts = parent_session["parent_artifacts"]
    initial_run_dir = get_run_dir(
        str(parent_artifacts["initial_run_id"]),
        scratch_root=scratch_root,
    )
    role_records = {
        "dark": (
            parent_artifacts["dark_tuning_curves"],
            initial_run_dir,
            str(epochs["dark"]),
        ),
        "light_train": (
            parent_session["artifacts"]["path_specific_place_tuning_curve"],
            parent_run_dir,
            str(epochs["AB"]),
        ),
    }
    output = {}
    for role, (records, source_run_dir, epoch) in role_records.items():
        output[role] = {}
        for trajectory_type in TRAJECTORY_TYPES:
            record = _one_record(
                records,
                label=f"{role} all-trial tuning curve",
                epoch=epoch,
                region=SUPPLEMENTARY_FIGURES_REGION,
                trajectory_type=trajectory_type,
                trial_subset="all",
            )
            output[role][trajectory_type] = {
                "record": record,
                "path": _record_path(
                    record,
                    "tuning_curve_path",
                    run_dir=source_run_dir,
                ),
            }
    return output


def _aligned_rates(
    movement_table: Any,
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return movement firing rates in the selected NWB unit order."""
    aligned = movement.align_movement_firing_rates(
        movement_table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )
    return np.asarray(aligned, dtype=float)


def _cv_pca_parameters(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated V1 cvPCA parameters including their preset name."""
    raw = dict(configuration["cv_pca_parameters"])
    parameter_name = str(raw.pop("cv_pca_param_name"))
    validated = cv_pca.validate_cv_pca_parameters(
        region=SUPPLEMENTARY_FIGURES_REGION,
        **raw,
    )
    return {"parameter_name": parameter_name, **validated}


def _run_cv_pca(context: Mapping[str, Any]) -> dict[str, Any]:
    """Compute and seal one canonical cvPCA bundle."""
    configuration = context["configuration"]
    parameters = _cv_pca_parameters(configuration)
    parameter_values = {
        name: value for name, value in parameters.items() if name != "parameter_name"
    }
    epochs = context["epochs"]
    source_identity = context["source_identity"]
    natural_key = {
        "animal_name": context["animal_name"],
        "date": context["date"],
        "region": SUPPLEMENTARY_FIGURES_REGION,
        "light_epoch": epochs["AB"],
        "dark_epoch": epochs["dark"],
        "parameter_name": parameters["parameter_name"],
        "parameters": parameter_values,
        "output_rule_sha256": cv_pca.OUTPUT_RULE_SHA256,
        "parent_session_manifest_sha256": context["parent_pointer"][
            "session_manifest_sha256"
        ],
        "selected_units_sha256": source_identity["selected_units_sha256"],
        "nwb_sources_sha256": provenance_sha256(context["nwb_sources"]),
    }
    result_id = str(selection_uuid("CVPCA", natural_key))
    dark_movement = context["movement"]["dark"]
    light_movement = context["movement"]["light_train"]
    position_offsets = {
        role: int(
            context["selections"][role]["position_rows"][
                PRIMARY_POSITION_ROLE
            ]["analysis_start_offset_samples"]
        )
        for role in ("dark", "light_train")
    }
    if len(set(position_offsets.values())) != 1:
        raise ValueError("cvPCA requires matching dark and AB position offsets.")
    result = cv_pca.compute_cv_pca(
        cv_pca_id=result_id,
        animal_name=context["animal_name"],
        date=context["date"],
        region=SUPPLEMENTARY_FIGURES_REGION,
        light_epoch=epochs["AB"],
        dark_epoch=epochs["dark"],
        spikes=context["spikes"]["ts_group"],
        stable_unit_ids=context["spikes"]["unit_ids"],
        light_position=context["cv_pca_positions"]["light_train"],
        dark_position=context["cv_pca_positions"]["dark"],
        light_movement_intervals=light_movement["movement_intervals"],
        dark_movement_intervals=dark_movement["movement_intervals"],
        light_movement_firing_rate_hz=_aligned_rates(
            light_movement["table"],
            spikes=context["spikes"]["ts_group"],
            stable_unit_ids=context["spikes"]["unit_ids"],
        ),
        dark_movement_firing_rate_hz=_aligned_rates(
            dark_movement["table"],
            spikes=context["spikes"]["ts_group"],
            stable_unit_ids=context["spikes"]["unit_ids"],
        ),
        light_trajectory_intervals=context["sources"]["light_train"][
            "trajectory_intervals"
        ],
        dark_trajectory_intervals=context["sources"]["dark"][
            "trajectory_intervals"
        ],
        graph_inputs={
            name: context["sources"]["dark"]["graph_inputs"][name]
            for name in TRAJECTORY_TYPES
        },
        upstream_provenance={
            "parent_figure_2": context["parent_pointer"],
            "dark_movement_firing_rate_sha256": file_sha256(
                dark_movement["firing_rate_path"]
            ),
            "light_movement_firing_rate_sha256": file_sha256(
                light_movement["firing_rate_path"]
            ),
            "nwb_sources_sha256": provenance_sha256(context["nwb_sources"]),
        },
        parameter_name=parameters["parameter_name"],
        parameter_sha256=provenance_sha256(parameter_values),
        position_offset_samples=position_offsets["dark"],
        **parameter_values,
    )
    written = cv_pca.write_cv_pca_artifact(
        result,
        artifact_root=context["run_dir"],
    )
    paths = written["artifact_paths"]
    return _relative_artifact_record(
        {
            "animal_name": context["animal_name"],
            "date": context["date"],
            "region": SUPPLEMENTARY_FIGURES_REGION,
            "dark_epoch": epochs["dark"],
            "light_epoch": epochs["AB"],
            "cv_pca_id": result_id,
            "parameter_name": parameters["parameter_name"],
            "random_seed": int(parameters["random_seed"]),
            "analysis_status": str(written["analysis_status"]),
            "artifact_origin": "computed",
        },
        paths,
        run_dir=context["run_dir"],
        path_fields=_CV_PCA_PATH_FIELDS,
    )


def _epoch_motor_parameters(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated motor parameters and their preset name."""
    raw = dict(configuration["epoch_motor_behavior_parameters"])
    name = str(raw.pop("epoch_motor_behavior_param_name"))
    values = epoch_motor_behavior.validate_epoch_motor_behavior_parameters(**raw)
    return {"parameter_name": name, **values}


def _run_epoch_motor_behavior(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute dark and AB epoch-motor bundles in fixed role order."""
    parameters = _epoch_motor_parameters(context["configuration"])
    movement_parameters = epoch_motor_behavior.validate_movement_parameter_snapshot(
        context["configuration"]["movement_parameters"]
    )
    parameter_payload = {
        "epoch_motor_behavior_param_name": parameters["parameter_name"],
        "progression_bin_size_cm": parameters["progression_bin_size_cm"],
    }
    output = []
    for role, condition in (("dark", "dark"), ("light_train", "AB")):
        epoch = str(context["epochs"][condition])
        selection = context["selections"][role]
        natural_key = {
            "animal_name": context["animal_name"],
            "date": context["date"],
            "epoch": epoch,
            "condition": condition,
            "parameters": parameter_payload,
            "movement_parameters": movement_parameters,
            "output_rule_sha256": provenance_sha256(
                dict(epoch_motor_behavior.OUTPUT_RULE)
            ),
            "parent_session_manifest_sha256": context["parent_pointer"][
                "session_manifest_sha256"
            ],
            "nwb_source_sha256": provenance_sha256(
                context["nwb_sources"][epoch]
            ),
        }
        result_id = str(selection_uuid("EpochMotorBehavior", natural_key))
        paths = epoch_motor_behavior.get_epoch_motor_behavior_artifact_paths(
            animal_name=context["animal_name"],
            date=context["date"],
            epoch=epoch,
            epoch_motor_behavior_id=result_id,
            artifact_root=context["run_dir"],
        )
        result = epoch_motor_behavior.compute_selected_epoch_motor_behavior(
            animal_name=context["animal_name"],
            date=context["date"],
            epoch=epoch,
            epoch_motor_behavior_id=result_id,
            primary_position=context["sources"][role]["positions"][
                PRIMARY_POSITION_ROLE
            ],
            orientation_reference_position=context["sources"][role]["positions"][
                ORIENTATION_REFERENCE_POSITION_ROLE
            ],
            primary_position_row=selection["position_rows"][PRIMARY_POSITION_ROLE],
            orientation_reference_position_row=selection["position_rows"][
                ORIENTATION_REFERENCE_POSITION_ROLE
            ],
            trajectory_intervals_by_type=context["sources"][role][
                "trajectory_intervals"
            ],
            graph_inputs_by_configuration={
                name: context["sources"][role]["graph_inputs"][name]
                for name in TRAJECTORY_TYPES
            },
            parameter_name=parameters["parameter_name"],
            parameter_sha256=provenance_sha256(parameter_payload),
            output_rule_sha256=provenance_sha256(
                dict(epoch_motor_behavior.OUTPUT_RULE)
            ),
            progression_bin_size_cm=parameters["progression_bin_size_cm"],
            movement_parameters=movement_parameters,
            movement_parameters_sha256=movement_parameters[
                "movement_parameters_sha256"
            ],
        )
        written = epoch_motor_behavior.write_epoch_motor_behavior_artifact(
            result,
            paths["artifact_dir"],
        )
        output.append(
            _relative_artifact_record(
                {
                    "animal_name": context["animal_name"],
                    "date": context["date"],
                    "epoch": epoch,
                    "condition": condition,
                    "epoch_motor_behavior_id": result_id,
                    "parameter_name": parameters["parameter_name"],
                    "analysis_status": str(result["analysis_status"]),
                    "artifact_origin": "computed",
                },
                written,
                run_dir=context["run_dir"],
                path_fields=_EPOCH_MOTOR_PATH_FIELDS,
            )
        )
    return output


def _swap_tuning_parameters(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated empirical swap parameters and their preset name."""
    raw = dict(configuration["swap_tuning_curve_comparison_parameters"])
    name = str(raw.pop("swap_tuning_curve_comparison_param_name"))
    values = swap_tuning.validate_swap_tuning_curve_comparison_parameters(**raw)
    return {"parameter_name": name, **values}


def _light_test_tuning_parameters(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the campaign-frozen BA tuning-curve parameter snapshot."""
    raw = configuration.get("light_test_tuning_curve_parameters")
    if not isinstance(raw, Mapping) or set(raw) != set(
        _LIGHT_TEST_TUNING_PARAMETER_FIELDS
    ):
        raise ValueError("BA tuning-curve parameter snapshot is incomplete.")
    name = str(raw["tuning_curve_param_name"]).strip()
    mode = str(raw["binning_mode"])
    if not name:
        raise ValueError("BA tuning-curve parameter name must be non-empty.")
    validated = path_specific_place.validate_binning_parameters(
        bin_size_cm=raw["place_bin_size_cm"],
        bin_count=raw["position_bin_count"],
        sigma_bins=raw["gaussian_smoothing_sigma_bins"],
    )
    expected_mode = (
        "bin_size_cm" if validated["bin_size_cm"] is not None else "bin_count"
    )
    if mode != expected_mode:
        raise ValueError("BA tuning-curve binning mode is inconsistent.")
    if (
        mode != "bin_size_cm"
        or not np.isclose(
            validated["bin_size_cm"],
            swap_tuning.REQUIRED_UPSTREAM_BIN_SIZE_CM,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.isclose(
            validated["sigma_bins"],
            swap_tuning.REQUIRED_UPSTREAM_SIGMA_BINS,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("BA tuning parameters are incompatible with SwapTuning.")
    return {
        "tuning_curve_param_name": name,
        "binning_mode": mode,
        "place_bin_size_cm": validated["bin_size_cm"],
        "position_bin_count": validated["bin_count"],
        "gaussian_smoothing_sigma_bins": validated["sigma_bins"],
    }


def _compute_light_test_tuning(
    context: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute the four BA curves absent from the Figure 2 parent campaign."""
    epoch = str(context["epochs"]["BA"])
    movement_result = context["movement"]["light_test"]
    parameters = _light_test_tuning_parameters(context["configuration"])
    movement_selection = {
        "nwb_file_name": context["nwb_file_name"],
        "epoch": epoch,
        "movement_firing_rate_id": str(
            _record_value(
                movement_result["record"],
                "movement_firing_rate_id",
            )
        ),
    }
    output = {}
    for trajectory_type in TRAJECTORY_TYPES:
        tuning_selection = _tuning_selection(
            movement_selection=movement_selection,
            trajectory_type=trajectory_type,
            trial_subset="all",
            parameters=parameters,
        )
        result_id = str(
            tuning_selection["path_specific_place_tuning_curve_id"]
        )
        result = path_specific_place.compute_selected_path_specific_place_tuning_curve(
            animal_name=context["animal_name"],
            date=context["date"],
            region=SUPPLEMENTARY_FIGURES_REGION,
            epoch=epoch,
            trajectory_type=trajectory_type,
            trial_subset="all",
            spikes=context["spikes"]["ts_group"],
            stable_unit_ids=context["spikes"]["unit_ids"],
            position=context["sources"]["light_test"]["positions"]["head"],
            trajectory_intervals=context["sources"]["light_test"][
                "trajectory_intervals"
            ][trajectory_type],
            graph_inputs=context["sources"]["light_test"]["graph_inputs"][
                trajectory_type
            ],
            movement_intervals=movement_result["movement_intervals"],
            movement_analysis_status=movement_result["analysis_status"],
            bin_size_cm=parameters["place_bin_size_cm"],
            bin_count=parameters["position_bin_count"],
            sigma_bins=float(parameters["gaussian_smoothing_sigma_bins"]),
        )
        result["tuning_curve"].attrs.update(
            _curve_attributes(
                tuning_selection,
                selected_units_sha256=context["source_identity"][
                    "selected_units_sha256"
                ],
            )
        )
        path = path_specific_place.get_path_specific_place_artifact_path(
            animal_name=context["animal_name"],
            date=context["date"],
            epoch=epoch,
            trajectory_type=trajectory_type,
            trial_subset="all",
            region=SUPPLEMENTARY_FIGURES_REGION,
            path_specific_place_tuning_curve_id=result_id,
            artifact_root=context["run_dir"],
        )
        written = path_specific_place.write_path_specific_place_artifact(
            result["tuning_curve"],
            path,
        )
        output[trajectory_type] = {
            "record": {
                **tuning_selection,
                "region": SUPPLEMENTARY_FIGURES_REGION,
                "analysis_status": str(result["analysis_status"]),
                "selected_units_sha256": context["source_identity"][
                    "selected_units_sha256"
                ],
                "artifact_origin": "computed",
            },
            "path": written,
        }
    return output


def _run_swap_tuning_curve_comparison(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute and seal one empirical dark/AB-to-BA swap comparison."""
    parameters = _swap_tuning_parameters(context["configuration"])
    parameter_values = {
        name: value for name, value in parameters.items() if name != "parameter_name"
    }
    parameter_payload = {
        "swap_tuning_curve_comparison_param_name": parameters["parameter_name"],
        **parameter_values,
    }
    epochs = context["epochs"]
    tuning = {
        role: dict(values) for role, values in context["tuning"].items()
    }
    if "light_test" not in tuning:
        tuning["light_test"] = _compute_light_test_tuning(context)
    tuning_paths = {
        role: {
            trajectory: values["path"]
            for trajectory, values in tuning[role].items()
        }
        for role in ("dark", "light_train", "light_test")
    }
    tuning_ids = {
        role: {
            trajectory: str(
                _record_value(values["record"], "path_specific_place_tuning_curve_id")
            )
            for trajectory, values in tuning[role].items()
        }
        for role in ("dark", "light_train", "light_test")
    }
    tuning_parameter_hashes = {
        role: {
            trajectory: str(
                _record_value(values["record"], "tuning_curve_parameters_sha256")
            )
            for trajectory, values in tuning[role].items()
        }
        for role in ("dark", "light_train", "light_test")
    }
    movement_ids = {
        role: str(
            _record_value(values["record"], "movement_firing_rate_id")
        )
        for role, values in context["movement"].items()
    }
    movement_table_hashes = {
        role: file_sha256(values["firing_rate_path"])
        for role, values in context["movement"].items()
    }
    movement_interval_hashes = {
        role: file_sha256(values["movement_intervals_path"])
        for role, values in context["movement"].items()
    }
    natural_key = {
        "animal_name": context["animal_name"],
        "date": context["date"],
        "region": SUPPLEMENTARY_FIGURES_REGION,
        "dark_epoch": epochs["dark"],
        "light_train_epoch": epochs["AB"],
        "light_test_epoch": epochs["BA"],
        "parameters": parameter_payload,
        "output_rule_sha256": swap_tuning.OUTPUT_RULE_SHA256,
        "tuning_curve_ids": tuning_ids,
        "movement_firing_rate_ids": movement_ids,
        "parent_session_manifest_sha256": context["parent_pointer"][
            "session_manifest_sha256"
        ],
    }
    result_id = str(selection_uuid("SwapTuningCurveComparison", natural_key))
    result = swap_tuning.compute_swap_tuning_curve_comparison(
        swap_tuning_curve_comparison_id=result_id,
        animal_name=context["animal_name"],
        date=context["date"],
        region=SUPPLEMENTARY_FIGURES_REGION,
        dark_epoch=epochs["dark"],
        light_train_epoch=epochs["AB"],
        light_test_epoch=epochs["BA"],
        tuning_curve_artifact_paths=tuning_paths,
        movement_firing_rate_tables_by_role={
            role: values["table"] for role, values in context["movement"].items()
        },
        spikes=context["spikes"]["ts_group"],
        stable_unit_ids=context["spikes"]["unit_ids"],
        position=context["sources"]["light_test"]["positions"]["head"],
        position_offset_samples=int(
            context["selections"]["light_test"]["position_rows"]["head"][
                "analysis_start_offset_samples"
            ]
        ),
        movement_interval=context["movement"]["light_test"][
            "movement_intervals"
        ],
        movement_analysis_status=context["movement"]["light_test"][
            "analysis_status"
        ],
        trajectory_intervals=context["sources"]["light_test"][
            "trajectory_intervals"
        ],
        graph_inputs_by_trajectory={
            name: context["sources"]["light_test"]["graph_inputs"][name]
            for name in TRAJECTORY_TYPES
        },
        parameter_name=parameters["parameter_name"],
        parameter_sha256=provenance_sha256(parameter_payload),
        output_rule_sha256=swap_tuning.OUTPUT_RULE_SHA256,
        source_tuning_curve_ids_by_role_trajectory=tuning_ids,
        source_tuning_parameters_sha256_by_role_trajectory=(
            tuning_parameter_hashes
        ),
        movement_firing_rate_ids_by_role=movement_ids,
        movement_firing_rate_table_sha256_by_role=movement_table_hashes,
        movement_intervals_sha256_by_role=movement_interval_hashes,
        sources={
            "parent_figure_2": context["parent_pointer"],
            "nwb_sources_sha256": provenance_sha256(context["nwb_sources"]),
        },
        **parameter_values,
    )
    paths = swap_tuning.get_swap_tuning_curve_comparison_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=SUPPLEMENTARY_FIGURES_REGION,
        dark_epoch=epochs["dark"],
        light_train_epoch=epochs["AB"],
        light_test_epoch=epochs["BA"],
        swap_tuning_curve_comparison_id=result_id,
        artifact_root=context["run_dir"],
    )
    written = swap_tuning.write_swap_tuning_curve_comparison_artifact(
        result,
        paths["artifact_dir"],
    )
    all_paths = dict(written)
    light_test_path_fields = []
    for trajectory_type, values in tuning["light_test"].items():
        field = f"light_test_{trajectory_type}_tuning_curve_path"
        all_paths[field] = Path(values["path"])
        light_test_path_fields.append(field)
    return _relative_artifact_record(
        {
            "animal_name": context["animal_name"],
            "date": context["date"],
            "region": SUPPLEMENTARY_FIGURES_REGION,
            "dark_epoch": epochs["dark"],
            "light_train_epoch": epochs["AB"],
            "light_test_epoch": epochs["BA"],
            "model_family": "empirical_swap_tuning",
            "models": list(swap_tuning.MODEL_NAMES),
            "swap_tuning_curve_comparison_id": result_id,
            "parameter_name": parameters["parameter_name"],
            "analysis_status": str(result["analysis_status"]),
            "artifact_origin": "computed",
        },
        all_paths,
        run_dir=context["run_dir"],
        path_fields=(*_SWAP_TUNING_PATH_FIELDS, *light_test_path_fields),
    )


def _compute_session_artifacts(
    context: Mapping[str, Any],
    *,
    cv_pca_runner: Callable[[Mapping[str, Any]], Mapping[str, Any]] = _run_cv_pca,
    epoch_motor_runner: Callable[
        [Mapping[str, Any]], Sequence[Mapping[str, Any]]
    ] = _run_epoch_motor_behavior,
    swap_tuning_runner: Callable[
        [Mapping[str, Any]], Mapping[str, Any]
    ] = _run_swap_tuning_curve_comparison,
) -> dict[str, list[dict[str, Any]]]:
    """Run the three artifact families through injectable production seams."""
    return {
        "cv_pca": [dict(cv_pca_runner(context))],
        "epoch_motor_behavior": [dict(row) for row in epoch_motor_runner(context)],
        "swap_tuning_curve_comparison": [dict(swap_tuning_runner(context))],
    }


def _source_identity(loaded_spikes: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable V1 source identity for one loaded NWB session."""
    return {
        "source": "ImportedSpikeSorting",
        "region": SUPPLEMENTARY_FIGURES_REGION,
        "spikesorting_merge_id": str(loaded_spikes["spikesorting_merge_id"]),
        "selected_units_sha256": str(loaded_spikes["selected_units_sha256"]),
        "n_units": int(loaded_spikes["n_units"]),
    }


def _run_supplementary_figures_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Compute and persist one supplementary-analysis session."""
    animal_name, date = str(animal_name), str(date)
    run_dir, campaign, parent_snapshot = prepare_supplementary_figures_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(run_dir, animal_name=animal_name, date=date)
    if session_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite a supplementary session: {session_dir}"
        )
    parent_run_dir, parent_session, parent_pointer = _load_one_parent_session(
        parent_snapshot,
        animal_name=animal_name,
        date=date,
        scratch_root=scratch_root,
    )
    epochs = {
        name: str(parent_session["epochs"][name]) for name in ("dark", "AB", "BA")
    }
    if (
        epochs["AB"] != FIGURE_2_LIGHT_TRAIN_EPOCH
        or epochs["BA"] != FIGURE_2_LIGHT_TEST_EPOCH
    ):
        raise ValueError("Figure 2 parent does not use the fixed AB/BA epochs.")
    nwb_path = Path(str(parent_session["nwb_path"])).resolve(strict=True)
    if nwb_path.is_relative_to(run_dir.resolve(strict=False)):
        raise ValueError("Source NWB must remain outside the output campaign.")

    import pynwb

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=animal_name,
            date=date,
        )
        fingerprint = nwb_fingerprint(nwb_path, nwbfile)
        if canonical_json(fingerprint) != canonical_json(
            parent_session["nwb_fingerprint"]
        ):
            raise ValueError("Source NWB changed relative to the Figure 2 parent.")
        selections = {
            "dark": select_run_epoch_catalog(
                nwbfile,
                nwb_file_name=nwb_path.name,
                epoch=epochs["dark"],
                position_roles=POSITION_ROLES,
                trajectory_types=TRAJECTORY_TYPES,
                graph_configurations=(FULL_W_CONFIGURATION_NAME,),
                require_dark=True,
            ),
            "light_train": select_run_epoch_catalog(
                nwbfile,
                nwb_file_name=nwb_path.name,
                epoch=epochs["AB"],
                position_roles=POSITION_ROLES,
                trajectory_types=TRAJECTORY_TYPES,
                graph_configurations=(FULL_W_CONFIGURATION_NAME,),
            ),
            "light_test": select_run_epoch_catalog(
                nwbfile,
                nwb_file_name=nwb_path.name,
                epoch=epochs["BA"],
                position_roles=POSITION_ROLES,
                trajectory_types=TRAJECTORY_TYPES,
                graph_configurations=(FULL_W_CONFIGURATION_NAME,),
            ),
        }
        _validate_epoch_selection(
            selections["dark"],
            epoch=epochs["dark"],
            condition="dark",
            is_light=False,
        )
        _validate_epoch_selection(
            selections["light_train"],
            epoch=epochs["AB"],
            condition=FIGURE_2_LIGHT_TRAIN_CONDITION,
            is_light=True,
        )
        _validate_epoch_selection(
            selections["light_test"],
            epoch=epochs["BA"],
            condition=FIGURE_2_LIGHT_TEST_CONDITION,
            is_light=True,
        )
        sources = {
            role: load_run_epoch_catalog_objects(nwbfile, selection)
            for role, selection in selections.items()
        }
        cv_pca_positions = _load_cv_pca_positions(nwbfile, selections)
        nwb_sources = {
            str(selection["epoch_row"]["epoch"]): _catalog_snapshot(selection)
            for selection in selections.values()
        }
        _verify_parent_catalog_snapshot(parent_session, nwb_sources)
        epoch_rows = [selection["epoch_row"] for selection in selections.values()]
        spike_bounds = (
            min(float(row["start_time"]) for row in epoch_rows),
            max(float(row["stop_time"]) for row in epoch_rows),
        )
        spikes = load_nwb_region_spikes(
            nwbfile,
            nwb_file_name=nwb_path.name,
            region=SUPPLEMENTARY_FIGURES_REGION,
            time_support=spike_bounds,
        )
        source_identity = _source_identity(spikes)
        parent_source = _one_record(
            parent_session["source_identity"],
            label="Figure 2 V1 spike source",
            region=SUPPLEMENTARY_FIGURES_REGION,
        )
        for field in (
            "spikesorting_merge_id",
            "selected_units_sha256",
            "n_units",
        ):
            if str(source_identity[field]) != str(parent_source[field]):
                raise ValueError(f"V1 source changed relative to Figure 2: {field}.")
        context = {
            "run_dir": run_dir,
            "configuration": campaign["analysis_parameters"],
            "animal_name": animal_name,
            "date": date,
            "nwb_file_name": nwb_path.name,
            "epochs": epochs,
            "parent_pointer": parent_pointer,
            "selections": selections,
            "sources": sources,
            "cv_pca_positions": cv_pca_positions,
            "nwb_sources": nwb_sources,
            "spikes": spikes,
            "source_identity": source_identity,
            "movement": _movement_context(
                parent_run_dir,
                parent_session,
                scratch_root=scratch_root,
            ),
            "tuning": _tuning_context(
                parent_run_dir,
                parent_session,
                scratch_root=scratch_root,
            ),
        }
        artifacts = _compute_session_artifacts(context)

    session_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": str(run_id),
        "created_at_utc": utc_now(),
        "code_provenance": code_provenance(),
        "status": "complete",
        "animal_name": animal_name,
        "date": date,
        "nwb_file_name": nwb_path.name,
        "nwb_path": str(nwb_path),
        "nwb_fingerprint": fingerprint,
        "epochs": epochs,
        "regions": [SUPPLEMENTARY_FIGURES_REGION],
        "parameters": campaign["analysis_parameters"],
        "parent_figure_2": dict(parent_snapshot),
        "parent_figure_2_session": parent_pointer,
        "source_identity": [source_identity],
        "nwb_sources": nwb_sources,
        "artifacts": artifacts,
    }
    manifest_path = session_dir / SESSION_MANIFEST_FILENAME
    write_json_once(session_manifest, manifest_path)
    _append_supplementary_session_manifest(
        campaign,
        session_manifest,
        run_dir=run_dir,
    )
    return session_manifest


def run_supplementary_figures_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Claim one session and clean only outputs created by a failed call."""
    run_dir, _campaign, _parent = prepare_supplementary_figures_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(run_dir, animal_name=animal_name, date=date)
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = session_dir.parent / f".{session_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Supplementary session is already running: "
                f"{animal_name} {date}"
            ) from exc
        preexisting = session_dir.exists()
        try:
            return _run_supplementary_figures_session(
                run_id=run_id,
                animal_name=animal_name,
                date=date,
                parent_run_id=parent_run_id,
                scratch_root=scratch_root,
            )
        except BaseException:
            if not preexisting and session_dir.exists():
                shutil.rmtree(session_dir)
            raise


def _validate_session_artifact_contract(
    session: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    """Validate exact family counts, roles, metadata, files, and bundles."""
    artifacts = session.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(
        ARTIFACT_FAMILIES
    ):
        raise ValueError("Supplementary session has an unexpected artifact schema.")
    expected_counts = {
        "cv_pca": 1,
        "epoch_motor_behavior": 2,
        "swap_tuning_curve_comparison": 1,
    }
    parameters = session.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Supplementary session lacks its parameter snapshot.")
    for family, count in expected_counts.items():
        records = artifacts[family]
        if not isinstance(records, list) or len(records) != count:
            raise ValueError(
                f"Supplementary family {family!r} must contain {count} records."
            )
        for record in records:
            _verify_record(record, run_dir=run_dir)
            if (
                str(record.get("animal_name")) != str(session.get("animal_name"))
                or str(record.get("date")) != str(session.get("date"))
            ):
                raise ValueError(
                    f"Supplementary family {family!r} has stale session identity."
                )

    epochs = session["epochs"]
    cv_record = artifacts["cv_pca"][0]
    if (
        str(cv_record.get("region")) != SUPPLEMENTARY_FIGURES_REGION
        or str(cv_record.get("dark_epoch")) != str(epochs["dark"])
        or str(cv_record.get("light_epoch")) != str(epochs["AB"])
        or str(cv_record.get("parameter_name"))
        != str(parameters["cv_pca_parameters"]["cv_pca_param_name"])
    ):
        raise ValueError("cvPCA record has stale region or epoch metadata.")
    cv_bundle = cv_pca.load_cv_pca_artifact(
        resolve_run_path(cv_record["artifact_dir"], run_dir=run_dir)
    )
    if (
        str(cv_bundle["cv_pca_id"]) != str(cv_record["cv_pca_id"])
        or str(cv_bundle["analysis_status"])
        != str(cv_record["analysis_status"])
    ):
        raise ValueError("cvPCA record disagrees with its bundle.")

    motor_records = artifacts["epoch_motor_behavior"]
    expected_motor = {
        (str(epochs["dark"]), "dark"),
        (str(epochs["AB"]), "AB"),
    }
    observed_motor = {
        (str(record.get("epoch")), str(record.get("condition")))
        for record in motor_records
    }
    if observed_motor != expected_motor:
        raise ValueError("EpochMotorBehavior dark/AB roles are incomplete.")
    for record in motor_records:
        if str(record.get("parameter_name")) != str(
            parameters["epoch_motor_behavior_parameters"][
                "epoch_motor_behavior_param_name"
            ]
        ):
            raise ValueError("EpochMotorBehavior parameter preset is stale.")
        bundle = epoch_motor_behavior.load_epoch_motor_behavior_artifact(
            resolve_run_path(record["artifact_dir"], run_dir=run_dir)
        )
        if (
            str(bundle["metadata"]["epoch_motor_behavior_id"])
            != str(record["epoch_motor_behavior_id"])
            or str(bundle["analysis_status"]) != str(record["analysis_status"])
        ):
            raise ValueError("EpochMotorBehavior record disagrees with its bundle.")

    swap_record = artifacts["swap_tuning_curve_comparison"][0]
    if (
        str(swap_record.get("region")) != SUPPLEMENTARY_FIGURES_REGION
        or str(swap_record.get("dark_epoch")) != str(epochs["dark"])
        or str(swap_record.get("light_train_epoch")) != str(epochs["AB"])
        or str(swap_record.get("light_test_epoch")) != str(epochs["BA"])
        or str(swap_record.get("model_family")) != "empirical_swap_tuning"
        or tuple(swap_record.get("models", ())) != tuple(swap_tuning.MODEL_NAMES)
        or str(swap_record.get("parameter_name"))
        != str(
            parameters["swap_tuning_curve_comparison_parameters"][
                "swap_tuning_curve_comparison_param_name"
            ]
        )
    ):
        raise ValueError("SwapTuningCurveComparison metadata is stale.")
    swap_bundle = swap_tuning.load_swap_tuning_curve_comparison_artifact(
        resolve_run_path(swap_record["artifact_dir"], run_dir=run_dir)
    )
    if (
        str(
            swap_bundle["metadata"]["swap_tuning_curve_comparison_id"]
        )
        != str(swap_record["swap_tuning_curve_comparison_id"])
        or str(swap_bundle["analysis_status"])
        != str(swap_record["analysis_status"])
    ):
        raise ValueError("SwapTuningCurveComparison record disagrees with its bundle.")
    source_hashes = swap_bundle.get("upstream_provenance", {}).get(
        "source_tuning_curve_sha256_by_role_trajectory"
    )
    if (
        not isinstance(source_hashes, Mapping)
        or not isinstance(source_hashes.get("light_test"), Mapping)
    ):
        raise ValueError("SwapTuningCurveComparison lacks BA tuning provenance.")
    record_hashes = swap_record["artifact_sha256"]
    for trajectory_type in TRAJECTORY_TYPES:
        field = f"light_test_{trajectory_type}_tuning_curve_path"
        if str(record_hashes.get(field, "")) != str(
            source_hashes["light_test"].get(trajectory_type, "")
        ):
            raise ValueError(
                "Recorded BA tuning hash disagrees with SwapTuning provenance."
            )


def load_supplementary_figures_session_manifest(
    path: Path,
    *,
    run_dir: Path,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load one completed session and verify every dependency."""
    manifest_path = Path(path).resolve(strict=True)
    guarded_run_dir = Path(run_dir).resolve(strict=True)
    if not manifest_path.is_relative_to(guarded_run_dir):
        raise ValueError("Supplementary session manifest escapes its run.")
    session = load_json(manifest_path)
    if (
        session.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or session.get("status") != "complete"
    ):
        raise ValueError("Supplementary session manifest is not complete.")
    epochs = session.get("epochs")
    if (
        not isinstance(epochs, Mapping)
        or set(epochs) != {"dark", "AB", "BA"}
        or str(epochs["AB"]) != FIGURE_2_LIGHT_TRAIN_EPOCH
        or str(epochs["BA"]) != FIGURE_2_LIGHT_TEST_EPOCH
        or len({str(value) for value in epochs.values()}) != 3
    ):
        raise ValueError("Supplementary session epochs are not canonical.")
    if session.get("regions") != [SUPPLEMENTARY_FIGURES_REGION]:
        raise ValueError("Supplementary session region changed.")
    _validate_session_artifact_contract(session, run_dir=guarded_run_dir)

    parent = session.get("parent_figure_2")
    pointer = session.get("parent_figure_2_session")
    if not isinstance(parent, Mapping) or not isinstance(pointer, Mapping):
        raise ValueError("Supplementary session lacks its Figure 2 parent pointers.")
    current = build_figure_2_parent_snapshot(
        str(parent["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(dict(parent)):
        raise ValueError("Supplementary Figure 2 parent changed after computation.")
    expected_pointer = _parent_session_pointer(
        parent,
        animal_name=str(session["animal_name"]),
        date=str(session["date"]),
    )
    if canonical_json(pointer) != canonical_json(expected_pointer):
        raise ValueError("Supplementary parent-session pointer changed.")
    parent_run_dir = get_run_dir(str(parent["run_id"]), scratch_root=scratch_root)
    parent_path = resolve_run_path(
        str(pointer["session_manifest_path"]),
        run_dir=parent_run_dir,
    )
    if file_sha256(parent_path) != str(pointer["session_manifest_sha256"]):
        raise ValueError("Supplementary parent session checksum changed.")
    parent_session = load_figure_2_session_manifest(
        parent_path,
        run_dir=parent_run_dir,
        scratch_root=scratch_root,
    )
    for field in ("nwb_file_name", "nwb_path", "nwb_fingerprint"):
        if canonical_json(session.get(field)) != canonical_json(
            parent_session.get(field)
        ):
            raise ValueError(
                f"Supplementary {field} differs from its Figure 2 parent."
            )
    if canonical_json(session.get("epochs")) != canonical_json(
        parent_session.get("epochs")
    ):
        raise ValueError("Supplementary epochs differ from Figure 2.")
    return session


def load_supplementary_figures_campaign(
    run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load a supplementary campaign and verify every completed session."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    campaign = load_json(run_dir / CAMPAIGN_MANIFEST_FILENAME)
    if campaign.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported supplementary campaign schema version.")
    if str(campaign.get("run_id")) != str(run_id):
        raise ValueError("Supplementary campaign run_id differs from its directory.")
    analysis_parameters = campaign.get("analysis_parameters")
    if not isinstance(analysis_parameters, Mapping) or analysis_parameters.get(
        "pipeline"
    ) != (
        SUPPLEMENTARY_FIGURES_PIPELINE
    ):
        raise ValueError("Selected campaign is not a supplementary-figures run.")
    parent = analysis_parameters.get("parent_figure_2")
    if not isinstance(parent, Mapping):
        raise ValueError("Supplementary campaign lacks its Figure 2 parent.")
    current = build_figure_2_parent_snapshot(
        str(parent["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(parent):
        raise ValueError("Figure 2 parent changed after campaign selection.")
    expected_parameters = build_supplementary_figures_configuration(parent)
    if canonical_json(analysis_parameters) != canonical_json(expected_parameters):
        raise ValueError("Supplementary campaign analysis parameters are stale.")
    if canonical_json(campaign.get("source_identity_policy")) != canonical_json(
        SOURCE_IDENTITY_POLICY
    ):
        raise ValueError("Supplementary campaign source identity policy is stale.")
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list):
        raise ValueError("Supplementary campaign sessions must be a list.")
    sessions = []
    seen = set()
    for summary in summaries:
        if str(summary.get("status")) != "complete":
            raise ValueError("Supplementary campaign indexes an incomplete session.")
        identity = (str(summary.get("animal_name")), str(summary.get("date")))
        if identity in seen:
            raise ValueError(
                f"Supplementary campaign duplicates session {identity!r}."
            )
        seen.add(identity)
        session_path = resolve_run_path(
            summary["session_manifest_path"],
            run_dir=run_dir,
        )
        expected_sha256 = str(summary.get("session_manifest_sha256", ""))
        if not expected_sha256 or file_sha256(session_path) != expected_sha256:
            raise ValueError(
                "Supplementary session manifest checksum mismatch for "
                f"{identity!r}."
            )
        session = load_supplementary_figures_session_manifest(
            session_path,
            run_dir=run_dir,
            scratch_root=scratch_root,
        )
        if (
            str(session["run_id"]) != str(run_id)
            or (str(session["animal_name"]), str(session["date"])) != identity
        ):
            raise ValueError("Supplementary campaign and session identities disagree.")
        if canonical_json(session["parameters"]) != canonical_json(
            campaign["analysis_parameters"]
        ):
            raise ValueError(
                "Supplementary session parameters changed after selection."
            )
        sessions.append(session)
    return run_dir, campaign, sessions


def _parser() -> argparse.ArgumentParser:
    """Build the explicit one-session supplementary campaign CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id", default=DEFAULT_PARENT_RUN_ID)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one supplementary-analysis session without DataJoint."""
    args = _parser().parse_args(argv)
    manifest = run_supplementary_figures_session(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        animal_name=args.animal_name,
        date=args.date,
        scratch_root=args.scratch_root,
    )
    print(
        "Completed supplementary analyses for "
        f"{manifest['animal_name']} {manifest['date']}."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ARTIFACT_FAMILIES",
    "DEFAULT_PARENT_RUN_ID",
    "SUPPLEMENTARY_FIGURES_PIPELINE",
    "SUPPLEMENTARY_FIGURES_REGION",
    "build_figure_2_parent_snapshot",
    "build_supplementary_figures_configuration",
    "load_parent_figure_2_sessions",
    "load_supplementary_figures_campaign",
    "load_supplementary_figures_session_manifest",
    "main",
    "prepare_supplementary_figures_campaign",
    "run_supplementary_figures_session",
]
