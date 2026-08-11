"""Run the complete Figure 2 analysis campaign without DataJoint."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import fcntl
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import (
    dark_light_glm,
    movement,
    path_progression_decoding,
    path_specific_decoding,
    path_specific_place,
    stability,
    swap_glm,
)
from v1ca1.spyglass.offline.figure_1 import (
    TRIAL_SUBSETS,
    _artifact_record_paths,
    _curve_attributes,
    _movement_selection,
    _tuning_selection,
)
from v1ca1.spyglass.offline.figure_1_decoding import (
    FIGURE_1_DECODING_PARAMETERS,
    run_figure_1_decoding,
)
from v1ca1.spyglass.offline.figure_1_examples import (
    DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    compute_nwb_example_payload,
    get_example_payload_path,
    write_example_payload,
)
from v1ca1.spyglass.offline.figure_1_full import (
    FULL_W_CONFIGURATION_NAME,
    build_parent_snapshot as build_initial_parent_snapshot,
    load_full_figure_campaign,
    load_full_figure_session_manifest,
    load_parent_analysis_inputs,
)
from v1ca1.spyglass.offline.figure_2_analyses import (
    FIGURE_2_DARK_LIGHT_GLM_PARAMETERS,
    FIGURE_2_LIGHT_TEST_CONDITION,
    FIGURE_2_LIGHT_TEST_EPOCH,
    FIGURE_2_LIGHT_TRAIN_CONDITION,
    FIGURE_2_LIGHT_TRAIN_EPOCH,
    FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS,
    FIGURE_2_SWAP_GLM_PARAMETERS,
    FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
    run_offline_dark_light_glm,
    run_offline_forward_swap_glm,
    run_offline_path_specific_decoding,
    run_offline_tuning_similarity,
)
from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    DEFAULT_SCRATCH_ROOT,
    MANIFEST_SCHEMA_VERSION,
    SESSION_MANIFEST_FILENAME,
    append_session_manifest,
    code_provenance,
    file_sha256,
    get_run_dir,
    get_session_dir,
    load_json,
    load_session_manifest,
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
from v1ca1.spyglass.selection import (
    canonical_json,
    provenance_sha256,
    selection_uuid,
    unit_identity_sha256,
)
from v1ca1.spyglass.table_specs import (
    DEFAULT_MOVEMENT_PARAMETERS,
    LEGACY_TUNING_CURVE_PARAMETERS,
)

DEFAULT_PARENT_RUN_ID = "figure1-full-nwb-v2"
FIGURE_2_PIPELINE = "figure_2"
FIGURE_2_REGION = "v1"
FIGURE_2_TUNING_PARAMETERS = dict(LEGACY_TUNING_CURVE_PARAMETERS)
FIGURE_2_PANEL_A_EXAMPLES = (
    {
        "animal_name": "L14",
        "date": "20240611",
        "sorting_unit_id": 34,
        "trajectory_types": ("center_to_left", "right_to_center"),
    },
    {
        "animal_name": "L15",
        "date": "20241121",
        "sorting_unit_id": 473,
        "trajectory_types": ("center_to_right", "left_to_center"),
    },
    {
        "animal_name": "L12",
        "date": "20240421",
        "sorting_unit_id": 37,
        "trajectory_types": ("center_to_right", "left_to_center"),
    },
    {
        "animal_name": "L14",
        "date": "20240611",
        "sorting_unit_id": 30,
        "trajectory_types": ("center_to_left", "right_to_center"),
    },
)
_CORE_ARTIFACT_FAMILIES = {
    "movement_firing_rate": 2,
    "path_specific_place_tuning_curve": 12,
    "path_specific_place_stability": 4,
}
_ANALYSIS_ARTIFACT_FAMILIES = {
    "path_specific_place_tuning_similarity": 2,
    "path_progression_decoding": 1,
    "path_specific_place_decoding": 2,
    "dark_light_glm": 1,
    "swap_glm": 1,
}
_PATH_PROGRESSION_ARTIFACT_FIELDS = (
    "artifact_manifest_path",
    "decoding_summary_path",
    "unit_eligibility_path",
    "selected_units_path",
    "binned_error_path",
)


def _one_record(
    records: Sequence[Mapping[str, Any]],
    *,
    label: str,
    **selectors: Any,
) -> dict[str, Any]:
    """Return exactly one record matching the requested fields."""
    matches = [
        dict(record)
        for record in records
        if all(
            str(record.get(field)) == str(value) for field, value in selectors.items()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {label} for {selectors!r}; " f"found {len(matches)}."
        )
    return matches[0]


def build_full_figure_1_parent_snapshot(
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Freeze a fully validated Figure 1 campaign and its session hashes."""
    parent_run_dir, campaign, sessions = load_full_figure_campaign(
        parent_run_id,
        scratch_root=scratch_root,
    )
    summaries = campaign.get("sessions", ())
    if not summaries or len(summaries) != len(sessions):
        raise ValueError("The full Figure 1 parent has incomplete sessions.")
    if any(str(row.get("status")) != "complete" for row in summaries):
        raise ValueError("Every full Figure 1 parent session must be complete.")
    snapshot_sessions = []
    for summary in summaries:
        session_path = resolve_run_path(
            summary["session_manifest_path"],
            run_dir=parent_run_dir,
        )
        snapshot_sessions.append(
            {
                "animal_name": str(summary["animal_name"]),
                "date": str(summary["date"]),
                "session_manifest_path": str(summary["session_manifest_path"]),
                "session_manifest_sha256": file_sha256(session_path),
            }
        )
    return {
        "run_id": str(parent_run_id),
        "manifest_sha256": file_sha256(parent_run_dir / CAMPAIGN_MANIFEST_FILENAME),
        "parent_figure_1d": dict(campaign["analysis_parameters"]["parent_figure_1d"]),
        "sessions": sorted(
            snapshot_sessions,
            key=lambda row: (row["animal_name"], row["date"]),
        ),
    }


def build_figure_2_configuration(
    parent_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable, manuscript-approved Figure 2 configuration."""
    return {
        "pipeline": FIGURE_2_PIPELINE,
        "parent_figure_1_full": dict(parent_snapshot),
        "region": FIGURE_2_REGION,
        "epochs": {
            "AB": FIGURE_2_LIGHT_TRAIN_EPOCH,
            "BA": FIGURE_2_LIGHT_TEST_EPOCH,
            "dark": "from_parent_figure_1_session",
        },
        "conditions": {
            "AB": FIGURE_2_LIGHT_TRAIN_CONDITION,
            "BA": FIGURE_2_LIGHT_TEST_CONDITION,
            "dark": "dark",
        },
        "trajectory_types": list(TRAJECTORY_TYPES),
        "full_w_configuration_name": FULL_W_CONFIGURATION_NAME,
        "position_roles": ["head"],
        "movement_parameters": dict(DEFAULT_MOVEMENT_PARAMETERS),
        "tuning_curve_parameters": dict(FIGURE_2_TUNING_PARAMETERS),
        "trial_subsets": list(TRIAL_SUBSETS),
        "tuning_similarity_parameters": dict(FIGURE_2_TUNING_SIMILARITY_PARAMETERS),
        "path_progression_decoding_parameters": dict(FIGURE_1_DECODING_PARAMETERS),
        "path_specific_place_decoding_parameters": dict(
            FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS
        ),
        "dark_light_glm_parameters": dict(FIGURE_2_DARK_LIGHT_GLM_PARAMETERS),
        "swap_glm_parameters": dict(FIGURE_2_SWAP_GLM_PARAMETERS),
        "panel_a_examples": [
            {
                **dict(spec),
                "trajectory_types": list(spec["trajectory_types"]),
            }
            for spec in FIGURE_2_PANEL_A_EXAMPLES
        ],
        "example_parameters": {
            "position_role": "head",
            "analysis_start_offset_samples": DEFAULT_POSITION_OFFSET,
            "speed_threshold_cm_s": DEFAULT_SPEED_THRESHOLD_CM_S,
            "position_bin_count": DEFAULT_POSITION_BIN_COUNT,
            "gaussian_smoothing_sigma_bins": (DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS),
            "trial_subsets": ["all"],
            "artifact_format": "npz",
        },
        "artifact_origin": "computed",
        "diagnostic_figures": False,
    }


def prepare_figure_2_campaign(
    *,
    run_id: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create or validate one append-only offline Figure 2 campaign."""
    parent_snapshot = build_full_figure_1_parent_snapshot(
        parent_run_id,
        scratch_root=scratch_root,
    )
    configuration = build_figure_2_configuration(parent_snapshot)
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    if (run_dir / CAMPAIGN_MANIFEST_FILENAME).exists():
        loaded_run_dir, campaign, _ = load_figure_2_campaign(
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
                "Existing campaign uses a different unit identity policy; "
                "use a new run_id."
            )
        return loaded_run_dir, campaign, parent_snapshot
    run_dir, campaign = prepare_campaign(
        run_id=run_id,
        analysis_parameters=configuration,
        source_identity_policy=SOURCE_IDENTITY_POLICY,
        scratch_root=scratch_root,
    )
    return run_dir, campaign, parent_snapshot


def _load_full_parent_session(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load one hash-pinned full Figure 1 session and its summary."""
    current = build_full_figure_1_parent_snapshot(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(dict(parent_snapshot)):
        raise ValueError("Full Figure 1 parent changed after campaign selection.")
    parent_run_dir = get_run_dir(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    summary = _one_record(
        parent_snapshot["sessions"],
        label="full Figure 1 parent session",
        animal_name=animal_name,
        date=date,
    )
    session_path = resolve_run_path(
        summary["session_manifest_path"],
        run_dir=parent_run_dir,
    )
    if file_sha256(session_path) != str(summary["session_manifest_sha256"]):
        raise ValueError("Full Figure 1 parent session checksum changed.")
    session = load_full_figure_session_manifest(
        session_path,
        run_dir=parent_run_dir,
    )
    return parent_run_dir, session, summary


def _load_initial_parent_session(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    scratch_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Load one transitive, hash-pinned initial Figure 1 session."""
    current = build_initial_parent_snapshot(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current) != canonical_json(dict(parent_snapshot)):
        raise ValueError("Initial Figure 1 parent changed after campaign selection.")
    parent_run_dir = get_run_dir(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    summary = _one_record(
        parent_snapshot["sessions"],
        label="initial Figure 1 parent session",
        animal_name=animal_name,
        date=date,
    )
    session_path = resolve_run_path(
        summary["session_manifest_path"],
        run_dir=parent_run_dir,
    )
    if file_sha256(session_path) != str(summary["session_manifest_sha256"]):
        raise ValueError("Initial Figure 1 parent session checksum changed.")
    session = load_session_manifest(
        session_path,
        run_dir=parent_run_dir,
        require_artifacts=True,
    )
    return parent_run_dir, session, summary


def _checked_parent_path(
    record: Mapping[str, Any],
    field: str,
    *,
    parent_run_dir: Path,
) -> Path:
    """Resolve and verify one checksummed parent artifact path."""
    path = resolve_run_path(record[field], run_dir=parent_run_dir)
    expected = record.get("artifact_sha256", {}).get(field)
    if expected is None or not path.is_file() or file_sha256(path) != str(expected):
        raise ValueError(f"Parent artifact checksum mismatch for {field!r}.")
    return path


def _load_parent_tuning_curves(
    parent_run_dir: Path,
    parent_session: Mapping[str, Any],
    *,
    epoch: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the four all-trial legacy tuning curves from the parent."""
    curves: dict[str, Any] = {}
    records: dict[str, dict[str, Any]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        record = _one_record(
            parent_session["artifacts"]["path_specific_place_tuning_curve"],
            label="parent all-trial tuning curve",
            epoch=epoch,
            region=FIGURE_2_REGION,
            trajectory_type=trajectory_type,
            trial_subset="all",
            tuning_curve_param_name=FIGURE_2_TUNING_PARAMETERS[
                "tuning_curve_param_name"
            ],
        )
        path = _checked_parent_path(
            record,
            "tuning_curve_path",
            parent_run_dir=parent_run_dir,
        )
        curves[trajectory_type] = path_specific_place.load_path_specific_place_artifact(
            path
        )
        records[trajectory_type] = record
    return curves, records


def _load_parent_inputs(
    parent_snapshot: Mapping[str, Any],
    *,
    animal_name: str,
    date: str,
    scratch_root: Path,
) -> dict[str, Any]:
    """Load all reusable dark inputs and the dark decoder from Figure 1."""
    full_run_dir, full_session, full_summary = _load_full_parent_session(
        parent_snapshot,
        animal_name=animal_name,
        date=date,
        scratch_root=scratch_root,
    )
    dark_epochs = tuple(str(value) for value in full_session.get("epochs", ()))
    if len(dark_epochs) != 1:
        raise ValueError("Full Figure 1 parent must declare one dark epoch.")
    dark_epoch = dark_epochs[0]
    initial_snapshot = parent_snapshot["parent_figure_1d"]
    initial_run_dir, initial_session, initial_summary = _load_initial_parent_session(
        initial_snapshot,
        animal_name=animal_name,
        date=date,
        scratch_root=scratch_root,
    )
    if dark_epoch not in {str(value) for value in initial_session["epochs"]}:
        raise ValueError("Full and initial Figure 1 dark epochs disagree.")
    parent_inputs = load_parent_analysis_inputs(
        initial_run_dir,
        initial_session,
        epoch=dark_epoch,
        region=FIGURE_2_REGION,
    )
    dark_curves, dark_curve_records = _load_parent_tuning_curves(
        initial_run_dir,
        initial_session,
        epoch=dark_epoch,
    )
    dark_decoder = _one_record(
        full_session["artifacts"]["path_progression_decoding"],
        label="dark path-progression decoder",
        epoch=dark_epoch,
        region=FIGURE_2_REGION,
    )
    for field in _PATH_PROGRESSION_ARTIFACT_FIELDS:
        _checked_parent_path(
            dark_decoder,
            field,
            parent_run_dir=full_run_dir,
        )
    dark_decoder_bundle = path_progression_decoding.load_decoding_artifact_bundle(
        _checked_parent_path(
            dark_decoder,
            "artifact_manifest_path",
            parent_run_dir=full_run_dir,
        )
    )
    dark_decoder_metadata = dark_decoder_bundle["metadata"]
    for field, expected in (
        ("animal_name", animal_name),
        ("date", date),
        ("epoch", dark_epoch),
        ("cohort_epoch", dark_epoch),
        ("region", FIGURE_2_REGION),
    ):
        if str(dark_decoder_metadata.get(field)) != str(expected):
            raise ValueError(f"Parent PathProgressionDecoding has mismatched {field}.")
    parent_artifacts = {
        "initial_run_id": str(initial_snapshot["run_id"]),
        "initial_session_manifest_path": str(initial_summary["session_manifest_path"]),
        "initial_session_manifest_sha256": str(
            initial_summary["session_manifest_sha256"]
        ),
        "dark_movement_firing_rate": dict(parent_inputs["movement_record"]),
        "dark_tuning_curves": [dark_curve_records[name] for name in TRAJECTORY_TYPES],
        "dark_stability": [
            parent_inputs["stability_records"][name] for name in TRAJECTORY_TYPES
        ],
        "full_run_id": str(parent_snapshot["run_id"]),
        "full_session_manifest_path": str(full_summary["session_manifest_path"]),
        "full_session_manifest_sha256": str(full_summary["session_manifest_sha256"]),
        "dark_path_progression_decoding": dark_decoder,
    }
    return {
        "dark_epoch": dark_epoch,
        "full_session": full_session,
        "initial_session": initial_session,
        "source_identity": parent_inputs["source_identity"],
        "movement_record": parent_inputs["movement_record"],
        "movement": parent_inputs["movement"],
        "stability_records": parent_inputs["stability_records"],
        "stability_tables": parent_inputs["stability_tables"],
        "tuning_curves": dark_curves,
        "tuning_curve_records": dark_curve_records,
        "path_progression_decoding_record": dark_decoder,
        "parent_artifacts": parent_artifacts,
    }


def _require_parent_fingerprint(
    nwb_path: Path,
    nwbfile: Any,
    parent_session: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the augmented NWB identity to match the Figure 1 parent."""
    observed = nwb_fingerprint(nwb_path, nwbfile)
    expected = dict(parent_session["nwb_fingerprint"])
    for field in (
        "resolved_path",
        "size_bytes",
        "mtime_ns",
        "nwb_identifier",
        "units_object_id",
    ):
        if observed.get(field) != expected.get(field):
            raise ValueError(f"Parent NWB fingerprint changed for {field!r}.")
    return observed


def _native(value: Any) -> Any:
    """Return NumPy-style scalars as plain Python values."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _row_fields(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Copy selected catalog provenance fields into JSON-safe scalars."""
    return {name: _native(row[name]) for name in fields if name in row}


def _catalog_snapshot(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the NWB rows needed for lap and path reconstruction."""
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


def _require_epoch_condition(
    selection: Mapping[str, Any],
    *,
    condition: str,
    is_light: bool,
) -> None:
    """Require one run epoch to carry its explicit dark/light condition."""
    row = selection["epoch_row"]
    if str(row.get("condition", "")).strip().casefold() != condition.casefold():
        raise ValueError(
            f"Epoch {row.get('epoch')!r} must have condition={condition!r}."
        )
    if row.get("is_light") is None or bool(row["is_light"]) is not is_light:
        raise ValueError(
            f"Epoch {row.get('epoch')!r} has inconsistent is_light metadata."
        )


def _seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return one immutable manifest record with its provenance digest."""
    output = dict(record)
    if "record_sha256" in output:
        raise ValueError("Cannot reseal an artifact record.")
    output["record_sha256"] = provenance_sha256(output)
    return output


def _compute_epoch_core(
    *,
    run_dir: Path,
    animal_name: str,
    date: str,
    epoch: str,
    nwb_file_name: str,
    region_group_id: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    selected_units_sha256: str,
    selection: Mapping[str, Any],
    sources: Mapping[str, Any],
    compute_tuning: bool,
) -> dict[str, Any]:
    """Compute one epoch's movement and optional all-unit tuning slice."""
    movement_parameters = dict(DEFAULT_MOVEMENT_PARAMETERS)
    movement_selection = _movement_selection(
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        position_series_name=selection["position_rows"]["head"]["position_series_name"],
        region_group_id=region_group_id,
        parameters=movement_parameters,
    )
    movement_result = movement.compute_selected_movement_firing_rate(
        animal_name=animal_name,
        date=date,
        region=FIGURE_2_REGION,
        epoch=epoch,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        position=sources["positions"]["head"],
        speed_threshold_cm_s=movement_parameters["speed_threshold_cm_s"],
        speed_smoothing_sigma_s=movement_parameters["speed_smoothing_sigma_s"],
    )
    movement_paths = movement.get_movement_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=FIGURE_2_REGION,
        movement_firing_rate_id=movement_selection["movement_firing_rate_id"],
        artifact_root=run_dir,
    )
    written_movement = movement.write_movement_artifacts(
        movement_result["table"],
        movement_result["movement_intervals"],
        movement_paths["artifact_dir"],
        overwrite=False,
    )
    movement_loaded = movement.load_movement_artifacts(movement_paths["artifact_dir"])
    movement_record = _seal_record(
        _artifact_record_paths(
            {
                **movement_selection,
                "region": FIGURE_2_REGION,
                "artifact_dir": movement_paths["artifact_dir"],
                "firing_rate_path": written_movement["firing_rate_path"],
                "movement_intervals_path": written_movement["movement_intervals_path"],
                "analysis_status": movement_result["analysis_status"],
                "n_units": movement_result["n_units"],
                "n_valid_units": movement_result["n_valid_units"],
                "selected_units_sha256": selected_units_sha256,
                "artifact_origin": "computed",
            },
            run_dir=run_dir,
            path_fields=(
                "artifact_dir",
                "firing_rate_path",
                "movement_intervals_path",
            ),
        )
    )
    result: dict[str, Any] = {
        "movement_selection": movement_selection,
        "movement": movement_loaded,
        "movement_record": movement_record,
        "tuning_curves": {},
        "tuning_curve_records": {},
        "stability_tables": {},
        "stability_records": {},
    }
    if not compute_tuning:
        return result

    tuning_parameters = dict(FIGURE_2_TUNING_PARAMETERS)
    curves: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
    tuning_records: dict[tuple[str, str], dict[str, Any]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        for trial_subset in TRIAL_SUBSETS:
            tuning_selection = _tuning_selection(
                movement_selection=movement_selection,
                trajectory_type=trajectory_type,
                trial_subset=trial_subset,
                parameters=tuning_parameters,
            )
            tuning_result = (
                path_specific_place.compute_selected_path_specific_place_tuning_curve(
                    animal_name=animal_name,
                    date=date,
                    region=FIGURE_2_REGION,
                    epoch=epoch,
                    trajectory_type=trajectory_type,
                    trial_subset=trial_subset,
                    spikes=spikes,
                    stable_unit_ids=stable_unit_ids,
                    position=(
                        sources["positions"]["head"]
                        if movement_loaded["analysis_status"] == "valid"
                        else None
                    ),
                    trajectory_intervals=sources["trajectory_intervals"][
                        trajectory_type
                    ],
                    graph_inputs=sources["graph_inputs"][trajectory_type],
                    movement_intervals=movement_loaded["movement_intervals"],
                    movement_analysis_status=movement_loaded["analysis_status"],
                    bin_size_cm=tuning_parameters["place_bin_size_cm"],
                    bin_count=tuning_parameters["position_bin_count"],
                    sigma_bins=tuning_parameters["gaussian_smoothing_sigma_bins"],
                )
            )
            tuning_result["tuning_curve"].attrs.update(
                _curve_attributes(
                    tuning_selection,
                    selected_units_sha256=selected_units_sha256,
                )
            )
            tuning_path = path_specific_place.get_path_specific_place_artifact_path(
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                trajectory_type=trajectory_type,
                trial_subset=trial_subset,
                region=FIGURE_2_REGION,
                path_specific_place_tuning_curve_id=tuning_selection[
                    "path_specific_place_tuning_curve_id"
                ],
                artifact_root=run_dir,
            )
            path_specific_place.write_path_specific_place_artifact(
                tuning_result["tuning_curve"],
                tuning_path,
                overwrite=False,
            )
            loaded_curve = path_specific_place.load_path_specific_place_artifact(
                tuning_path
            )
            key = (trajectory_type, trial_subset)
            curves[key] = (loaded_curve, tuning_selection)
            tuning_records[key] = _seal_record(
                _artifact_record_paths(
                    {
                        **tuning_selection,
                        "region": FIGURE_2_REGION,
                        "tuning_curve_path": tuning_path,
                        "analysis_status": tuning_result["analysis_status"],
                        "n_units": tuning_result["n_units"],
                        "n_valid_units": tuning_result["n_valid_units"],
                        "n_trials": tuning_result["n_trials"],
                        "n_position_bins": tuning_result["n_position_bins"],
                        "selected_units_sha256": selected_units_sha256,
                        "artifact_origin": "computed",
                    },
                    run_dir=run_dir,
                    path_fields=("tuning_curve_path",),
                )
            )

    stability_records: dict[str, dict[str, Any]] = {}
    stability_tables: dict[str, pd.DataFrame] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        odd_curve, odd_selection = curves[(trajectory_type, "odd")]
        even_curve, even_selection = curves[(trajectory_type, "even")]
        stability_payload = {
            "odd_path_specific_place_tuning_curve_id": odd_selection[
                "path_specific_place_tuning_curve_id"
            ],
            "even_path_specific_place_tuning_curve_id": even_selection[
                "path_specific_place_tuning_curve_id"
            ],
        }
        stability_id = str(
            selection_uuid("PathSpecificPlaceStability", stability_payload)
        )
        stability_result = stability.compute_selected_stability_from_tuning_curves(
            odd_tuning_curve=odd_curve,
            even_tuning_curve=even_curve,
            movement_firing_rate_table=movement_loaded["table"],
        )
        stability_path = stability.get_stability_artifact_path(
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            trajectory_type=trajectory_type,
            region=FIGURE_2_REGION,
            path_specific_place_stability_id=stability_id,
            artifact_root=run_dir,
        )
        stability.write_stability_artifact(
            stability_result["table"],
            stability_path,
            overwrite=False,
        )
        stability_tables[trajectory_type] = pd.read_parquet(stability_path)
        stability_records[trajectory_type] = _seal_record(
            _artifact_record_paths(
                {
                    "path_specific_place_stability_id": stability_id,
                    **stability_payload,
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch,
                    "region": FIGURE_2_REGION,
                    "trajectory_type": trajectory_type,
                    "tuning_curve_param_name": tuning_parameters[
                        "tuning_curve_param_name"
                    ],
                    "movement_firing_rate_id": movement_selection[
                        "movement_firing_rate_id"
                    ],
                    "stability_path": stability_path,
                    "analysis_status": stability_result["analysis_status"],
                    "n_units": stability_result["n_units"],
                    "n_valid_units": stability_result["n_valid_units"],
                    "selected_units_sha256": unit_identity_sha256(stable_unit_ids),
                    "artifact_origin": "computed",
                },
                run_dir=run_dir,
                path_fields=("stability_path",),
            )
        )
    result.update(
        {
            "tuning_curves": {
                trajectory_type: curves[(trajectory_type, "all")][0]
                for trajectory_type in TRAJECTORY_TYPES
            },
            "tuning_curve_records": tuning_records,
            "stability_tables": stability_tables,
            "stability_records": stability_records,
        }
    )
    return result


def _example_records_for_session(
    nwbfile: Any,
    *,
    run_dir: Path,
    nwb_file_name: str,
    animal_name: str,
    date: str,
    dark_epoch: str,
) -> list[dict[str, Any]]:
    """Compute dark and AB payloads for this session's fixed example cells."""
    records = []
    for spec in FIGURE_2_PANEL_A_EXAMPLES:
        if str(spec["animal_name"]) != animal_name or str(spec["date"]) != date:
            continue
        for condition, epoch in (
            ("dark", dark_epoch),
            (FIGURE_2_LIGHT_TRAIN_CONDITION, FIGURE_2_LIGHT_TRAIN_EPOCH),
        ):
            payload = compute_nwb_example_payload(
                nwbfile,
                nwb_file_name=nwb_file_name,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                region=FIGURE_2_REGION,
                sorting_unit_id=spec["sorting_unit_id"],
                trajectory_types=spec["trajectory_types"],
            )
            path = get_example_payload_path(
                run_dir,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                region=FIGURE_2_REGION,
                sorting_unit_id=spec["sorting_unit_id"],
            )
            written = write_example_payload(payload, path, run_dir=run_dir)
            records.append(
                _seal_record(
                    {
                        **dict(spec),
                        "trajectory_types": list(spec["trajectory_types"]),
                        "region": FIGURE_2_REGION,
                        "condition": condition,
                        "epoch": epoch,
                        "payload_path": relative_run_path(
                            written["payload_path"],
                            run_dir=run_dir,
                        ),
                        "artifact_sha256": written["artifact_sha256"],
                        "persistent_unit_identity": payload["metadata"][
                            "persistent_unit_identity"
                        ],
                        "artifact_origin": "computed",
                    }
                )
            )
    return records


def _run_figure_2_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Compute and persist one session's database-free Figure 2 inputs."""
    animal_name, date = map(str, (animal_name, date))
    run_dir, campaign, parent_snapshot = prepare_figure_2_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(
        run_dir,
        animal_name=animal_name,
        date=date,
    )
    if session_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite a Figure 2 session: {session_dir}"
        )
    parent = _load_parent_inputs(
        parent_snapshot,
        animal_name=animal_name,
        date=date,
        scratch_root=scratch_root,
    )
    dark_epoch = str(parent["dark_epoch"])
    epoch_by_condition = {
        "dark": dark_epoch,
        FIGURE_2_LIGHT_TRAIN_CONDITION: FIGURE_2_LIGHT_TRAIN_EPOCH,
        FIGURE_2_LIGHT_TEST_CONDITION: FIGURE_2_LIGHT_TEST_EPOCH,
    }
    if len(set(epoch_by_condition.values())) != 3:
        raise ValueError("Figure 2 requires three distinct run epochs.")
    nwb_path = Path(str(parent["full_session"]["nwb_path"])).resolve(strict=True)
    if nwb_path.is_relative_to(run_dir.resolve(strict=False)):
        raise ValueError("Source NWB must remain outside the output run.")

    import pynwb

    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        validate_nwb_session_identity(
            nwbfile,
            animal_name=animal_name,
            date=date,
        )
        fingerprint = _require_parent_fingerprint(
            nwb_path,
            nwbfile,
            parent["full_session"],
        )
        selections: dict[str, dict[str, Any]] = {}
        sources: dict[str, dict[str, Any]] = {}
        for condition, epoch in epoch_by_condition.items():
            selection = select_run_epoch_catalog(
                nwbfile,
                nwb_file_name=nwb_path.name,
                epoch=epoch,
                position_roles=("head",),
                trajectory_types=TRAJECTORY_TYPES,
                graph_configurations=(FULL_W_CONFIGURATION_NAME,),
                require_dark=condition == "dark",
            )
            _require_epoch_condition(
                selection,
                condition=condition,
                is_light=condition != "dark",
            )
            selections[epoch] = selection
            sources[epoch] = load_run_epoch_catalog_objects(
                nwbfile,
                selection,
            )

        epoch_starts = [
            float(selection["epoch_row"]["start_time"])
            for selection in selections.values()
        ]
        epoch_stops = [
            float(selection["epoch_row"]["stop_time"])
            for selection in selections.values()
        ]
        loaded_spikes = load_nwb_region_spikes(
            nwbfile,
            nwb_file_name=nwb_path.name,
            region=FIGURE_2_REGION,
            time_support=(min(epoch_starts), max(epoch_stops)),
        )
        parent_source = parent["source_identity"]
        for field in ("spikesorting_merge_id", "selected_units_sha256"):
            if str(loaded_spikes[field]) != str(parent_source[field]):
                raise ValueError(
                    "NWB regional units changed relative to Figure 1 " f"for {field!r}."
                )
        region_group_id = str(parent_source["offline_region_sorted_spikes_view_id"])
        common_core = {
            "run_dir": run_dir,
            "animal_name": animal_name,
            "date": date,
            "nwb_file_name": nwb_path.name,
            "region_group_id": region_group_id,
            "spikes": loaded_spikes["ts_group"],
            "stable_unit_ids": loaded_spikes["unit_ids"],
            "selected_units_sha256": loaded_spikes["selected_units_sha256"],
        }
        ab = _compute_epoch_core(
            epoch=FIGURE_2_LIGHT_TRAIN_EPOCH,
            selection=selections[FIGURE_2_LIGHT_TRAIN_EPOCH],
            sources=sources[FIGURE_2_LIGHT_TRAIN_EPOCH],
            compute_tuning=True,
            **common_core,
        )
        ba = _compute_epoch_core(
            epoch=FIGURE_2_LIGHT_TEST_EPOCH,
            selection=selections[FIGURE_2_LIGHT_TEST_EPOCH],
            sources=sources[FIGURE_2_LIGHT_TEST_EPOCH],
            compute_tuning=False,
            **common_core,
        )
        example_records = _example_records_for_session(
            nwbfile,
            run_dir=run_dir,
            nwb_file_name=nwb_path.name,
            animal_name=animal_name,
            date=date,
            dark_epoch=dark_epoch,
        )

        dark_tuning_ids = {
            trajectory_type: parent["tuning_curve_records"][trajectory_type][
                "path_specific_place_tuning_curve_id"
            ]
            for trajectory_type in TRAJECTORY_TYPES
        }
        ab_tuning_ids = {
            trajectory_type: ab["tuning_curve_records"][(trajectory_type, "all")][
                "path_specific_place_tuning_curve_id"
            ]
            for trajectory_type in TRAJECTORY_TYPES
        }
        dark_similarity = run_offline_tuning_similarity(
            output_dir=run_dir,
            animal_name=animal_name,
            date=date,
            region=FIGURE_2_REGION,
            epoch=dark_epoch,
            tuning_curve_ids_by_trajectory=dark_tuning_ids,
            tuning_curves_by_trajectory=parent["tuning_curves"],
            movement_firing_rate_table=parent["movement"]["table"],
            parameters=FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
        )
        ab_similarity = run_offline_tuning_similarity(
            output_dir=run_dir,
            animal_name=animal_name,
            date=date,
            region=FIGURE_2_REGION,
            epoch=FIGURE_2_LIGHT_TRAIN_EPOCH,
            tuning_curve_ids_by_trajectory=ab_tuning_ids,
            tuning_curves_by_trajectory=ab["tuning_curves"],
            movement_firing_rate_table=ab["movement"]["table"],
            parameters=FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
        )

        ab_stability_ids = {
            trajectory_type: ab["stability_records"][trajectory_type][
                "path_specific_place_stability_id"
            ]
            for trajectory_type in TRAJECTORY_TYPES
        }
        ab_path_progression = _seal_record(
            run_figure_1_decoding(
                animal_name=animal_name,
                date=date,
                epoch=FIGURE_2_LIGHT_TRAIN_EPOCH,
                nwb_file_name=nwb_path.name,
                region_sorted_spikes_group_id=region_group_id,
                movement_firing_rate_id=ab["movement_selection"][
                    "movement_firing_rate_id"
                ],
                stability_source_ids=ab_stability_ids,
                spikes=loaded_spikes["ts_group"],
                stable_unit_ids=loaded_spikes["unit_ids"],
                movement_firing_rate_table=ab["movement"]["table"],
                stability_tables_by_trajectory=ab["stability_tables"],
                position=sources[FIGURE_2_LIGHT_TRAIN_EPOCH]["positions"]["head"],
                trajectory_intervals=sources[FIGURE_2_LIGHT_TRAIN_EPOCH][
                    "trajectory_intervals"
                ],
                graph_inputs={
                    name: sources[FIGURE_2_LIGHT_TRAIN_EPOCH]["graph_inputs"][name]
                    for name in TRAJECTORY_TYPES
                },
                movement_interval=ab["movement"]["movement_intervals"],
                output_dir=run_dir,
            )["artifact_record"]
        )

        dark_path_specific = run_offline_path_specific_decoding(
            output_dir=run_dir,
            animal_name=animal_name,
            date=date,
            region=FIGURE_2_REGION,
            epoch=dark_epoch,
            nwb_file_name=nwb_path.name,
            region_sorted_spikes_group_id=region_group_id,
            movement_firing_rate_id=parent["movement_record"][
                "movement_firing_rate_id"
            ],
            spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            position=sources[dark_epoch]["positions"]["head"],
            trajectory_intervals_by_type=sources[dark_epoch]["trajectory_intervals"],
            graph_inputs_by_configuration={
                name: sources[dark_epoch]["graph_inputs"][name]
                for name in TRAJECTORY_TYPES
            },
            movement_intervals=parent["movement"]["movement_intervals"],
            parameters=FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS,
        )
        ab_path_specific = run_offline_path_specific_decoding(
            output_dir=run_dir,
            animal_name=animal_name,
            date=date,
            region=FIGURE_2_REGION,
            epoch=FIGURE_2_LIGHT_TRAIN_EPOCH,
            nwb_file_name=nwb_path.name,
            region_sorted_spikes_group_id=region_group_id,
            movement_firing_rate_id=ab["movement_selection"]["movement_firing_rate_id"],
            spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            position=sources[FIGURE_2_LIGHT_TRAIN_EPOCH]["positions"]["head"],
            trajectory_intervals_by_type=sources[FIGURE_2_LIGHT_TRAIN_EPOCH][
                "trajectory_intervals"
            ],
            graph_inputs_by_configuration={
                name: sources[FIGURE_2_LIGHT_TRAIN_EPOCH]["graph_inputs"][name]
                for name in TRAJECTORY_TYPES
            },
            movement_intervals=ab["movement"]["movement_intervals"],
            parameters=FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS,
        )

        path_graphs = {
            name: sources[dark_epoch]["graph_inputs"][name] for name in TRAJECTORY_TYPES
        }
        dark_light_record = run_offline_dark_light_glm(
            output_dir=run_dir,
            animal_name=animal_name,
            date=date,
            region=FIGURE_2_REGION,
            nwb_file_name=nwb_path.name,
            region_sorted_spikes_group_id=region_group_id,
            dark_epoch=dark_epoch,
            light_epoch=FIGURE_2_LIGHT_TRAIN_EPOCH,
            dark_movement_firing_rate_id=parent["movement_record"][
                "movement_firing_rate_id"
            ],
            light_movement_firing_rate_id=ab["movement_selection"][
                "movement_firing_rate_id"
            ],
            spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            dark_movement_firing_rate_table=parent["movement"]["table"],
            light_movement_firing_rate_table=ab["movement"]["table"],
            movement_by_epoch={
                dark_epoch: parent["movement"]["movement_intervals"],
                FIGURE_2_LIGHT_TRAIN_EPOCH: ab["movement"]["movement_intervals"],
            },
            trajectory_intervals_by_epoch={
                dark_epoch: sources[dark_epoch]["trajectory_intervals"],
                FIGURE_2_LIGHT_TRAIN_EPOCH: sources[FIGURE_2_LIGHT_TRAIN_EPOCH][
                    "trajectory_intervals"
                ],
            },
            graph_inputs_by_configuration=path_graphs,
            position_by_epoch={
                dark_epoch: sources[dark_epoch]["positions"]["head"],
                FIGURE_2_LIGHT_TRAIN_EPOCH: sources[FIGURE_2_LIGHT_TRAIN_EPOCH][
                    "positions"
                ]["head"],
            },
            parameters=FIGURE_2_DARK_LIGHT_GLM_PARAMETERS,
        )
        swap_record = run_offline_forward_swap_glm(
            output_dir=run_dir,
            animal_name=animal_name,
            date=date,
            region=FIGURE_2_REGION,
            nwb_file_name=nwb_path.name,
            region_sorted_spikes_group_id=region_group_id,
            dark_epoch=dark_epoch,
            light_train_epoch=FIGURE_2_LIGHT_TRAIN_EPOCH,
            light_test_epoch=FIGURE_2_LIGHT_TEST_EPOCH,
            dark_light_record=dark_light_record,
            light_test_movement_firing_rate_id=ba["movement_selection"][
                "movement_firing_rate_id"
            ],
            spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            movement_interval=ba["movement"]["movement_intervals"],
            movement_analysis_status=ba["movement"]["analysis_status"],
            trajectory_intervals_by_type=sources[FIGURE_2_LIGHT_TEST_EPOCH][
                "trajectory_intervals"
            ],
            graph_inputs_by_configuration={
                name: sources[FIGURE_2_LIGHT_TEST_EPOCH]["graph_inputs"][name]
                for name in TRAJECTORY_TYPES
            },
            position=sources[FIGURE_2_LIGHT_TEST_EPOCH]["positions"]["head"],
            dark_condition="dark",
            light_train_condition=FIGURE_2_LIGHT_TRAIN_CONDITION,
            light_test_condition=FIGURE_2_LIGHT_TEST_CONDITION,
            parameters=FIGURE_2_SWAP_GLM_PARAMETERS,
        )

    artifacts = {
        "movement_firing_rate": [
            ab["movement_record"],
            ba["movement_record"],
        ],
        "path_specific_place_tuning_curve": [
            ab["tuning_curve_records"][(trajectory_type, trial_subset)]
            for trajectory_type in TRAJECTORY_TYPES
            for trial_subset in TRIAL_SUBSETS
        ],
        "path_specific_place_stability": [
            ab["stability_records"][trajectory_type]
            for trajectory_type in TRAJECTORY_TYPES
        ],
        "path_specific_place_tuning_similarity": [
            dark_similarity,
            ab_similarity,
        ],
        "path_progression_decoding": [ab_path_progression],
        "path_specific_place_decoding": [
            dark_path_specific,
            ab_path_specific,
        ],
        "figure_examples": example_records,
        "dark_light_glm": [dark_light_record],
        "swap_glm": [swap_record],
    }
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
        "epochs": epoch_by_condition,
        "regions": [FIGURE_2_REGION],
        "trajectories": list(TRAJECTORY_TYPES),
        "parameters": campaign["analysis_parameters"],
        "parent_figure_1_full": dict(parent_snapshot),
        "parent_artifacts": parent["parent_artifacts"],
        "source_identity": [dict(parent_source)],
        "nwb_sources": {
            epoch: _catalog_snapshot(selection)
            for epoch, selection in selections.items()
        },
        "upstream_artifacts": {
            "dark_movement_firing_rate_id": parent["movement_record"][
                "movement_firing_rate_id"
            ],
            "AB_movement_firing_rate_id": ab["movement_selection"][
                "movement_firing_rate_id"
            ],
            "BA_movement_firing_rate_id": ba["movement_selection"][
                "movement_firing_rate_id"
            ],
            "dark_stability_ids_by_trajectory": {
                name: parent["stability_records"][name][
                    "path_specific_place_stability_id"
                ]
                for name in TRAJECTORY_TYPES
            },
            "AB_stability_ids_by_trajectory": ab_stability_ids,
            "dark_path_progression_decoding_id": parent[
                "path_progression_decoding_record"
            ]["path_progression_decoding_id"],
        },
        "artifacts": artifacts,
    }
    manifest_path = session_dir / SESSION_MANIFEST_FILENAME
    write_json_once(session_manifest, manifest_path)
    append_session_manifest(
        campaign,
        session_manifest,
        run_dir=run_dir,
    )
    return session_manifest


def run_figure_2_session(
    *,
    run_id: str,
    animal_name: str,
    date: str,
    parent_run_id: str = DEFAULT_PARENT_RUN_ID,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Claim and run one session, cleaning only this call's failed outputs."""
    run_dir, _campaign, _parent = prepare_figure_2_campaign(
        run_id=run_id,
        parent_run_id=parent_run_id,
        scratch_root=scratch_root,
    )
    session_dir = get_session_dir(
        run_dir,
        animal_name=animal_name,
        date=date,
    )
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = session_dir.parent / f".{session_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(
                lock_stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Figure 2 session is already running: {animal_name} {date}"
            ) from exc
        session_preexisted = session_dir.exists()
        try:
            return _run_figure_2_session(
                run_id=run_id,
                animal_name=animal_name,
                date=date,
                parent_run_id=parent_run_id,
                scratch_root=scratch_root,
            )
        except BaseException:
            if not session_preexisted and session_dir.exists():
                shutil.rmtree(session_dir)
            raise


def _verify_artifact(path: Path, expected_sha256: str) -> None:
    """Require one declared Figure 2 artifact and checksum."""
    if not path.is_file() or file_sha256(path) != str(expected_sha256):
        raise ValueError(f"Figure 2 artifact checksum mismatch: {path}")


def _verify_computed_record(
    record: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    """Verify one run-local computed record and all declared files."""
    if str(record.get("artifact_origin")) != "computed":
        raise ValueError("Figure 2 artifacts must be computed de novo.")
    if "record_sha256" not in record:
        raise ValueError("Figure 2 result record is missing its checksum.")
    unhashed = dict(record)
    observed = str(unhashed.pop("record_sha256"))
    if observed != provenance_sha256(unhashed):
        raise ValueError("Figure 2 result record checksum mismatch.")

    nested = record.get("artifacts")
    path_hashes = record.get("artifact_sha256")
    if isinstance(nested, Mapping):
        if not nested:
            raise ValueError("Figure 2 result declares an empty artifact bundle.")
        for metadata in nested.values():
            if not isinstance(metadata, Mapping):
                raise ValueError("Artifact file metadata must be a mapping.")
            path = resolve_run_path(metadata["relative_path"], run_dir=run_dir)
            _verify_artifact(path, metadata["sha256"])
            if int(metadata["file_size_bytes"]) != int(path.stat().st_size):
                raise ValueError(f"Figure 2 artifact size changed: {path}")
        return
    if isinstance(path_hashes, Mapping):
        if not path_hashes:
            raise ValueError("Figure 2 result declares no file checksums.")
        for field, digest in path_hashes.items():
            path = resolve_run_path(record[field], run_dir=run_dir)
            _verify_artifact(path, digest)
        return
    if isinstance(path_hashes, str) and "payload_path" in record:
        path = resolve_run_path(record["payload_path"], run_dir=run_dir)
        _verify_artifact(path, path_hashes)
        return
    raise ValueError("Figure 2 result does not declare its artifacts.")


def _expected_example_count(animal_name: str, date: str) -> int:
    """Return two epoch payloads per fixed example cell in one session."""
    return 2 * sum(
        str(spec["animal_name"]) == str(animal_name) and str(spec["date"]) == str(date)
        for spec in FIGURE_2_PANEL_A_EXAMPLES
    )


def _validate_nwb_source_snapshots(
    session: Mapping[str, Any],
    epochs: Mapping[str, Any],
) -> None:
    """Require complete catalog selectors for all three Figure 2 epochs."""
    snapshots = session.get("nwb_sources")
    nwb_file_name = str(session.get("nwb_file_name", ""))
    if not nwb_file_name:
        raise ValueError("Figure 2 session is missing its NWB file name.")
    expected_epochs = {str(value) for value in epochs.values()}
    if not isinstance(snapshots, Mapping) or set(snapshots) != expected_epochs:
        raise ValueError("Figure 2 NWB source snapshots are incomplete.")
    condition_by_epoch = {
        str(epochs["dark"]): ("dark", False),
        str(epochs["AB"]): (FIGURE_2_LIGHT_TRAIN_CONDITION, True),
        str(epochs["BA"]): (FIGURE_2_LIGHT_TEST_CONDITION, True),
    }
    for epoch, (condition, is_light) in condition_by_epoch.items():
        snapshot = snapshots[epoch]
        if not isinstance(snapshot, Mapping):
            raise ValueError("Figure 2 NWB source snapshot must be a mapping.")
        interval = snapshot.get("epoch_interval")
        observed_is_light = (
            interval.get("is_light") if isinstance(interval, Mapping) else None
        )
        if not isinstance(interval, Mapping) or (
            str(interval.get("epoch")) != epoch
            or str(interval.get("condition")) != condition
            or not isinstance(observed_is_light, bool)
            or observed_is_light is not is_light
            or str(interval.get("nwb_file_name")) != nwb_file_name
        ):
            raise ValueError("Figure 2 epoch condition provenance is stale.")
        positions = snapshot.get("positions")
        if not isinstance(positions, Mapping) or set(positions) != {"head"}:
            raise ValueError("Figure 2 requires one head-position selector.")
        position = positions["head"]
        required_position = {
            "position_series_name",
            "position_role",
            "source_row_index",
            "start_index",
            "stop_index_exclusive",
            "sample_count",
            "analysis_start_offset_samples",
            "start_time",
            "stop_time",
            "source_table_path",
            "source_table_object_id",
            "source_object_path",
            "source_object_id",
        }
        if not isinstance(position, Mapping) or not required_position.issubset(
            position
        ):
            raise ValueError("Figure 2 head-position selector is incomplete.")
        if (
            str(position.get("epoch")) != epoch
            or str(position.get("position_role")) != "head"
            or str(position.get("nwb_file_name")) != nwb_file_name
        ):
            raise ValueError("Figure 2 head-position selector is stale.")
        trajectory_rows = snapshot.get("trajectory_intervals")
        if not isinstance(trajectory_rows, Mapping) or set(trajectory_rows) != set(
            TRAJECTORY_TYPES
        ):
            raise ValueError("Figure 2 trajectory selectors are incomplete.")
        graph_rows = snapshot.get("wtrack_graphs")
        expected_graphs = {*TRAJECTORY_TYPES, FULL_W_CONFIGURATION_NAME}
        if not isinstance(graph_rows, Mapping) or set(graph_rows) != expected_graphs:
            raise ValueError("Figure 2 W-track selectors are incomplete.")
        for trajectory_type, row in trajectory_rows.items():
            if not {
                "epoch",
                "trajectory_type",
                "interval_count",
                "source_table_path",
                "source_table_object_id",
                "source_object_path",
                "source_object_id",
            }.issubset(row):
                raise ValueError("Figure 2 trajectory selector is incomplete.")
            if (
                str(row.get("epoch")) != epoch
                or str(row.get("trajectory_type")) != trajectory_type
                or str(row.get("nwb_file_name")) != nwb_file_name
            ):
                raise ValueError("Figure 2 trajectory selector is stale.")
        for configuration_name, row in graph_rows.items():
            if not {
                "configuration_name",
                "coordinate_unit",
                "use_hmm",
                "source_row_index",
                "source_table_path",
                "source_table_object_id",
                "source_object_path",
                "source_object_id",
            }.issubset(row):
                raise ValueError("Figure 2 W-track selector is incomplete.")
            if (
                str(row.get("configuration_name")) != configuration_name
                or str(row.get("nwb_file_name")) != nwb_file_name
            ):
                raise ValueError("Figure 2 W-track selector is stale.")


def _record_value(record: Mapping[str, Any], field: str) -> Any:
    """Return one direct or nested selection value from a result record."""
    if field in record:
        return record[field]
    selection = record.get("selection")
    return selection.get(field) if isinstance(selection, Mapping) else None


def _record_artifact_path(
    record: Mapping[str, Any],
    field: str,
    *,
    run_dir: Path,
) -> Path:
    """Resolve one nested or flat run-relative artifact file."""
    nested = record.get("artifacts")
    if isinstance(nested, Mapping) and field in nested:
        metadata = nested[field]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Artifact metadata for {field!r} is malformed.")
        value = metadata["relative_path"]
    else:
        value = record[field]
    return resolve_run_path(value, run_dir=run_dir)


def _validate_analysis_roles(
    session: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    """Require exact Figure 2 epoch roles and validate figure-facing bundles."""
    animal_name = str(session["animal_name"])
    date = str(session["date"])
    nwb_file_name = str(session["nwb_file_name"])
    dark_epoch = str(session["epochs"]["dark"])
    ab_epoch = str(session["epochs"]["AB"])
    ba_epoch = str(session["epochs"]["BA"])

    for records in artifacts.values():
        for record in records:
            for field, expected in (
                ("animal_name", animal_name),
                ("date", date),
                ("region", FIGURE_2_REGION),
                ("nwb_file_name", nwb_file_name),
            ):
                observed = _record_value(record, field)
                if observed is not None and str(observed) != expected:
                    raise ValueError(
                        f"Figure 2 record has mismatched {field}: {observed!r}."
                    )

    similarities = artifacts["path_specific_place_tuning_similarity"]
    if any(
        str(row.get("analysis_name")) != "PathSpecificPlaceTuningSimilarity"
        for row in similarities
    ):
        raise ValueError("Figure 2 similarity records have a stale analysis name.")
    if {str(_record_value(row, "epoch")) for row in similarities} != {
        dark_epoch,
        ab_epoch,
    }:
        raise ValueError("Figure 2 similarities must cover exactly dark and AB.")

    progression = artifacts["path_progression_decoding"][0]
    if str(_record_value(progression, "epoch")) != ab_epoch:
        raise ValueError("Figure 2 computed path progression must be AB.")
    progression_path = _record_artifact_path(
        progression,
        "artifact_manifest_path",
        run_dir=run_dir,
    )
    progression_bundle = path_progression_decoding.load_decoding_artifact_bundle(
        progression_path
    )
    progression_metadata = progression_bundle["metadata"]
    for field, expected in (
        ("animal_name", animal_name),
        ("date", date),
        ("epoch", ab_epoch),
        ("cohort_epoch", ab_epoch),
        ("region", FIGURE_2_REGION),
    ):
        if str(progression_metadata.get(field)) != expected:
            raise ValueError(f"PathProgressionDecoding has mismatched {field}.")

    place_records = artifacts["path_specific_place_decoding"]
    if any(
        str(row.get("analysis_name")) != "PathSpecificPlaceDecoding"
        for row in place_records
    ):
        raise ValueError("Figure 2 place decoder records have a stale name.")
    if {str(_record_value(row, "epoch")) for row in place_records} != {
        dark_epoch,
        ab_epoch,
    }:
        raise ValueError("PathSpecificPlaceDecoding must cover dark and AB.")
    for record in place_records:
        artifact_dir = resolve_run_path(record["artifact_dir"], run_dir=run_dir)
        bundle = path_specific_decoding.load_path_specific_decoding_artifact(
            artifact_dir
        )
        metadata = bundle["metadata"]
        for field, expected in (
            ("animal_name", animal_name),
            ("date", date),
            ("epoch", str(_record_value(record, "epoch"))),
            ("region", FIGURE_2_REGION),
        ):
            if str(metadata.get(field)) != expected:
                raise ValueError(f"PathSpecificPlaceDecoding has mismatched {field}.")

    dark_light = artifacts["dark_light_glm"][0]
    if str(dark_light.get("analysis_name")) != "DarkLightGLM":
        raise ValueError("Figure 2 DarkLightGLM record has a stale name.")
    for field, expected in (
        ("dark_epoch", dark_epoch),
        ("light_epoch", ab_epoch),
        ("region", FIGURE_2_REGION),
        ("nwb_file_name", nwb_file_name),
    ):
        if str(_record_value(dark_light, field)) != expected:
            raise ValueError(f"DarkLightGLM has mismatched {field}.")
    # The outer record explicitly hashes the manifest and every selected model
    # consumed by SwapGLM/Figure 2. Candidate files are intentionally not
    # reopened on every campaign read; creation performed a full bundle load.

    swap = artifacts["swap_glm"][0]
    if str(swap.get("analysis_name")) != "SwapGLM":
        raise ValueError("Figure 2 SwapGLM record has a stale name.")
    for field, expected in (
        ("dark_epoch", dark_epoch),
        ("light_train_epoch", ab_epoch),
        ("light_test_epoch", ba_epoch),
        ("region", FIGURE_2_REGION),
        ("nwb_file_name", nwb_file_name),
    ):
        if str(_record_value(swap, field)) != expected:
            raise ValueError(f"SwapGLM has mismatched {field}.")
    if str(_record_value(swap, "dark_light_glm_id")) != str(
        _record_value(dark_light, "dark_light_glm_id")
    ):
        raise ValueError("SwapGLM does not reference this session's DarkLightGLM.")
    swap_dir = resolve_run_path(swap["artifact_dir"], run_dir=run_dir)
    swap_bundle = swap_glm.load_swap_glm_artifact(swap_dir)
    swap_metadata = swap_bundle["metadata"]
    for field, expected in (
        ("animal_name", animal_name),
        ("date", date),
        ("dark_epoch", dark_epoch),
        ("light_train_epoch", ab_epoch),
        ("light_test_epoch", ba_epoch),
        ("region", FIGURE_2_REGION),
    ):
        if str(swap_metadata.get(field)) != expected:
            raise ValueError(f"SwapGLM bundle has mismatched {field}.")


def load_figure_2_session_manifest(
    path: Path,
    *,
    run_dir: Path,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> dict[str, Any]:
    """Load one completed Figure 2 session and verify every dependency."""
    manifest_path = Path(path).resolve(strict=True)
    guarded_run_dir = Path(run_dir).resolve(strict=True)
    if not manifest_path.is_relative_to(guarded_run_dir):
        raise ValueError("Figure 2 session manifest escapes its run.")
    session = load_json(manifest_path)
    if (
        session.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or session.get("status") != "complete"
    ):
        raise ValueError("Figure 2 session manifest is not complete.")
    epochs = session.get("epochs")
    if not isinstance(epochs, Mapping) or set(epochs) != {"dark", "AB", "BA"}:
        raise ValueError("Figure 2 session must declare dark, AB, and BA epochs.")
    if (
        str(epochs["AB"]) != FIGURE_2_LIGHT_TRAIN_EPOCH
        or str(epochs["BA"]) != FIGURE_2_LIGHT_TEST_EPOCH
    ):
        raise ValueError("Figure 2 light epoch assignments changed.")
    _validate_nwb_source_snapshots(session, epochs)

    artifacts = session.get("artifacts")
    required = {
        *_CORE_ARTIFACT_FAMILIES,
        *_ANALYSIS_ARTIFACT_FAMILIES,
        "figure_examples",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required:
        raise ValueError("Figure 2 session has an unexpected artifact schema.")
    expected_counts = {
        **_CORE_ARTIFACT_FAMILIES,
        **_ANALYSIS_ARTIFACT_FAMILIES,
        "figure_examples": _expected_example_count(
            session["animal_name"],
            session["date"],
        ),
    }
    for family, expected_count in expected_counts.items():
        records = artifacts[family]
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(
                f"Figure 2 family {family!r} must contain " f"{expected_count} records."
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(f"Figure 2 family {family!r} is malformed.")
            _verify_computed_record(record, run_dir=guarded_run_dir)
    _validate_analysis_roles(
        session,
        artifacts,
        run_dir=guarded_run_dir,
    )

    ab_tuning = artifacts["path_specific_place_tuning_curve"]
    expected_tuning_keys = {
        (trajectory_type, trial_subset)
        for trajectory_type in TRAJECTORY_TYPES
        for trial_subset in TRIAL_SUBSETS
    }
    observed_tuning_keys = {
        (str(row.get("trajectory_type")), str(row.get("trial_subset")))
        for row in ab_tuning
        if str(row.get("epoch")) == FIGURE_2_LIGHT_TRAIN_EPOCH
    }
    if observed_tuning_keys != expected_tuning_keys:
        raise ValueError("Figure 2 AB tuning rows are incomplete or duplicated.")
    observed_stability = {
        str(row.get("trajectory_type"))
        for row in artifacts["path_specific_place_stability"]
        if str(row.get("epoch")) == FIGURE_2_LIGHT_TRAIN_EPOCH
    }
    if observed_stability != set(TRAJECTORY_TYPES):
        raise ValueError("Figure 2 AB stability rows are incomplete.")
    movement_epochs = {
        str(row.get("epoch")) for row in artifacts["movement_firing_rate"]
    }
    if movement_epochs != {
        FIGURE_2_LIGHT_TRAIN_EPOCH,
        FIGURE_2_LIGHT_TEST_EPOCH,
    }:
        raise ValueError("Figure 2 movement rows must be AB and BA.")
    expected_examples = {
        (
            str(spec["sorting_unit_id"]),
            epoch,
            tuple(str(value) for value in spec["trajectory_types"]),
        )
        for spec in FIGURE_2_PANEL_A_EXAMPLES
        if str(spec["animal_name"]) == str(session["animal_name"])
        and str(spec["date"]) == str(session["date"])
        for epoch in (str(epochs["dark"]), str(epochs["AB"]))
    }
    observed_examples = {
        (
            str(row.get("sorting_unit_id")),
            str(row.get("epoch")),
            tuple(str(value) for value in row.get("trajectory_types", ())),
        )
        for row in artifacts["figure_examples"]
    }
    if observed_examples != expected_examples:
        raise ValueError("Figure 2 example payload selections are noncanonical.")

    parent_snapshot = session.get("parent_figure_1_full")
    if not isinstance(parent_snapshot, Mapping):
        raise ValueError("Figure 2 session lacks its Figure 1 parent snapshot.")
    current_parent = build_full_figure_1_parent_snapshot(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current_parent) != canonical_json(dict(parent_snapshot)):
        raise ValueError("Figure 2 parent changed after session computation.")
    expected_parent = _load_parent_inputs(
        parent_snapshot,
        animal_name=str(session["animal_name"]),
        date=str(session["date"]),
        scratch_root=scratch_root,
    )
    if canonical_json(session.get("parent_artifacts")) != canonical_json(
        expected_parent["parent_artifacts"]
    ):
        raise ValueError("Figure 2 parent artifact pointers changed.")
    if str(epochs["dark"]) != str(expected_parent["dark_epoch"]):
        raise ValueError("Figure 2 dark epoch changed relative to its parent.")
    full_parent_session = expected_parent["full_session"]
    for field in ("nwb_file_name", "nwb_path", "nwb_fingerprint"):
        if canonical_json(session.get(field)) != canonical_json(
            full_parent_session.get(field)
        ):
            raise ValueError(
                f"Figure 2 session {field} differs from its Figure 1 parent."
            )
    if canonical_json(session.get("source_identity")) != canonical_json(
        [expected_parent["source_identity"]]
    ):
        raise ValueError(
            "Figure 2 regional spike source differs from its Figure 1 parent."
        )
    return session


def load_figure_2_campaign(
    run_id: str,
    *,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Load a Figure 2 campaign and verify all session artifacts."""
    run_dir = get_run_dir(run_id, scratch_root=scratch_root)
    campaign = load_json(run_dir / CAMPAIGN_MANIFEST_FILENAME)
    if campaign.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported Figure 2 campaign schema version.")
    if str(campaign.get("run_id")) != str(run_id):
        raise ValueError("Figure 2 campaign run_id does not match its directory.")
    if campaign.get("analysis_parameters", {}).get("pipeline") != (FIGURE_2_PIPELINE):
        raise ValueError("Selected campaign is not a Figure 2 run.")
    parent_snapshot = campaign["analysis_parameters"]["parent_figure_1_full"]
    current_parent = build_full_figure_1_parent_snapshot(
        str(parent_snapshot["run_id"]),
        scratch_root=scratch_root,
    )
    if canonical_json(current_parent) != canonical_json(parent_snapshot):
        raise ValueError("Figure 2 parent changed after campaign selection.")
    summaries = campaign.get("sessions")
    if not isinstance(summaries, list):
        raise ValueError("Figure 2 campaign sessions must be a list.")
    sessions = []
    seen: set[tuple[str, str]] = set()
    for summary in summaries:
        identity = (str(summary.get("animal_name")), str(summary.get("date")))
        if identity in seen:
            raise ValueError(
                f"Figure 2 campaign contains duplicate session {identity!r}."
            )
        seen.add(identity)
        session_path = resolve_run_path(
            summary["session_manifest_path"],
            run_dir=run_dir,
        )
        session = load_figure_2_session_manifest(
            session_path,
            run_dir=run_dir,
            scratch_root=scratch_root,
        )
        if (
            str(session.get("run_id")) != str(run_id)
            or str(session["animal_name"]) != identity[0]
            or str(session["date"]) != identity[1]
        ):
            raise ValueError("Figure 2 campaign and session identities disagree.")
        if canonical_json(session["parent_figure_1_full"]) != canonical_json(
            parent_snapshot
        ):
            raise ValueError("Figure 2 sessions use different parent snapshots.")
        if canonical_json(session.get("parameters")) != canonical_json(
            campaign["analysis_parameters"]
        ):
            raise ValueError("Figure 2 session parameters changed after selection.")
        sessions.append(session)
    return run_dir, campaign, sessions


def _parser() -> argparse.ArgumentParser:
    """Build the explicit one-session Figure 2 CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run-id", default=DEFAULT_PARENT_RUN_ID)
    parser.add_argument("--animal-name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one Figure 2 session without plotting or database access."""
    args = _parser().parse_args(argv)
    manifest = run_figure_2_session(
        run_id=args.run_id,
        parent_run_id=args.parent_run_id,
        animal_name=args.animal_name,
        date=args.date,
        scratch_root=args.scratch_root,
    )
    print(
        "Completed offline Figure 2 inputs for "
        f"{manifest['animal_name']} {manifest['date']} "
        f"(dark={manifest['epochs']['dark']}, "
        f"AB={manifest['epochs']['AB']}, BA={manifest['epochs']['BA']})."
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PARENT_RUN_ID",
    "FIGURE_2_PANEL_A_EXAMPLES",
    "FIGURE_2_PIPELINE",
    "FIGURE_2_REGION",
    "build_figure_2_configuration",
    "build_full_figure_1_parent_snapshot",
    "load_figure_2_campaign",
    "load_figure_2_session_manifest",
    "main",
    "prepare_figure_2_campaign",
    "run_figure_2_session",
]
