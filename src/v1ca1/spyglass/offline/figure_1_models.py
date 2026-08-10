"""Run Figure 1 encoding models without DataJoint or legacy artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import dpp_encoding, motor_encoding
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import (
    canonical_json,
    provenance_sha256,
    selection_uuid,
)
from v1ca1.spyglass.table_specs import (
    MANUSCRIPT_DPP_ENCODING_PARAMETERS,
    MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS,
)


FIGURE_1_MOTOR_ENCODING_PARAMETERS = MappingProxyType(
    dict(MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS)
)
FIGURE_1_DPP_ENCODING_PARAMETERS = MappingProxyType(
    dict(MANUSCRIPT_DPP_ENCODING_PARAMETERS)
)


def _nonempty_string(value: Any, *, name: str) -> str:
    """Return one stripped, non-empty source identifier."""
    if value is None:
        raise ValueError(f"{name} must be non-empty.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _approved_parameters(
    parameters: Mapping[str, Any],
    *,
    approved: Mapping[str, Any],
    analysis_name: str,
) -> dict[str, Any]:
    """Require the exact, explicitly approved Figure 1 parameter row."""
    supplied = dict(parameters)
    if canonical_json(supplied) != canonical_json(dict(approved)):
        raise ValueError(
            f"{analysis_name} parameters must exactly match the approved "
            "Figure 1 parameters."
        )
    return supplied


def _source_fields(
    *,
    stability_ids_by_trajectory: Mapping[str, Any],
) -> dict[str, str]:
    """Return the four fixed trajectory, graph, and stability source slots."""
    expected = set(TRAJECTORY_TYPES)
    actual = set(stability_ids_by_trajectory)
    if actual != expected:
        raise ValueError(
            "stability_ids_by_trajectory must contain exactly the four "
            f"trajectory types; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )
    fields: dict[str, str] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        fields[f"{trajectory_type}_trajectory_type"] = trajectory_type
        fields[f"{trajectory_type}_configuration_name"] = trajectory_type
        fields[f"{trajectory_type}_stability_id"] = _nonempty_string(
            stability_ids_by_trajectory[trajectory_type],
            name=f"{trajectory_type}_stability_id",
        )
    fields["full_w_configuration_name"] = "full_w"
    return fields


def build_motor_encoding_selection_snapshot(
    *,
    nwb_file_name: str,
    epoch: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    primary_position_series_name: str,
    orientation_reference_position_series_name: str,
    stability_ids_by_trajectory: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic offline equivalent of a motor selection row."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_1_MOTOR_ENCODING_PARAMETERS,
        analysis_name="MotorEncoding",
    )
    primary_name = _nonempty_string(
        primary_position_series_name,
        name="primary_position_series_name",
    )
    reference_name = _nonempty_string(
        orientation_reference_position_series_name,
        name="orientation_reference_position_series_name",
    )
    if primary_name == reference_name:
        raise ValueError(
            "MotorEncoding requires distinct primary and "
            "orientation-reference position series."
        )
    natural_key = {
        "nwb_file_name": _nonempty_string(
            nwb_file_name,
            name="nwb_file_name",
        ),
        "epoch": _nonempty_string(epoch, name="epoch"),
        "region_sorted_spikes_group_id": _nonempty_string(
            region_sorted_spikes_group_id,
            name="region_sorted_spikes_group_id",
        ),
        "movement_firing_rate_id": _nonempty_string(
            movement_firing_rate_id,
            name="movement_firing_rate_id",
        ),
        "primary_position_series_name": primary_name,
        "orientation_reference_position_series_name": reference_name,
        **_source_fields(
            stability_ids_by_trajectory=stability_ids_by_trajectory,
        ),
        "motor_encoding_param_name": approved[
            "motor_encoding_param_name"
        ],
        "motor_encoding_parameters_sha256": provenance_sha256(approved),
        "motor_encoding_model_spec_sha256": (
            motor_encoding.MODEL_SPEC_SHA256
        ),
        "motor_encoding_output_rule_sha256": (
            motor_encoding.OUTPUT_RULE_SHA256
        ),
    }
    return {
        "motor_encoding_id": str(
            selection_uuid("MotorEncoding", natural_key)
        ),
        **natural_key,
    }


def build_dpp_encoding_selection_snapshot(
    *,
    nwb_file_name: str,
    epoch: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    stability_ids_by_trajectory: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic offline equivalent of a DPP selection row."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_1_DPP_ENCODING_PARAMETERS,
        analysis_name="DPPEncoding",
    )
    natural_key = {
        "nwb_file_name": _nonempty_string(
            nwb_file_name,
            name="nwb_file_name",
        ),
        "epoch": _nonempty_string(epoch, name="epoch"),
        "region_sorted_spikes_group_id": _nonempty_string(
            region_sorted_spikes_group_id,
            name="region_sorted_spikes_group_id",
        ),
        "movement_firing_rate_id": _nonempty_string(
            movement_firing_rate_id,
            name="movement_firing_rate_id",
        ),
        **_source_fields(
            stability_ids_by_trajectory=stability_ids_by_trajectory,
        ),
        "dpp_encoding_param_name": approved["dpp_encoding_param_name"],
        "dpp_encoding_parameters_sha256": provenance_sha256(approved),
    }
    return {
        "dpp_encoding_id": str(
            selection_uuid("DPPEncoding", natural_key)
        ),
        **natural_key,
    }


def _guarded_output_root(output_dir: Path) -> Path:
    """Return one explicit output root, rejecting existing non-directories."""
    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Output root is not a directory: {root}")
    return root


def _artifact_metadata(path: Path, *, output_root: Path) -> dict[str, Any]:
    """Return one portable, checksum-bearing artifact record."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(output_root):
        raise ValueError(f"Artifact escaped the explicit output root: {path}")
    return {
        "relative_path": resolved.relative_to(output_root).as_posix(),
        "file_size_bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def _motor_output_paths(
    *,
    output_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    motor_encoding_id: str,
) -> dict[str, Path]:
    """Return guarded, session-first paths for one motor result."""
    paths = motor_encoding.get_motor_encoding_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        motor_encoding_id=motor_encoding_id,
        artifact_root=output_root,
    )
    artifact_dir = paths["artifact_dir"].resolve(strict=False)
    if not artifact_dir.is_relative_to(output_root):
        raise ValueError("MotorEncoding output escaped the explicit output root.")
    return paths


def _dpp_output_path(
    *,
    output_root: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    dpp_encoding_id: str,
) -> Path:
    """Return the guarded, session-first path for one DPP result."""
    path = dpp_encoding.get_dpp_encoding_artifact_path(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        dpp_encoding_id=dpp_encoding_id,
        artifact_root=output_root,
    ).resolve(strict=False)
    if not path.is_relative_to(output_root):
        raise ValueError("DPPEncoding output escaped the explicit output root.")
    return path


def run_offline_motor_encoding(
    *,
    output_dir: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    primary_position_series_name: str,
    orientation_reference_position_series_name: str,
    stability_ids_by_trajectory: Mapping[str, Any],
    parameters: Mapping[str, Any],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    primary_position: Any,
    orientation_reference_position: Any,
    primary_position_source: str,
    orientation_reference_position_source: str,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compute and write one de novo Figure 1 MotorEncoding result."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_1_MOTOR_ENCODING_PARAMETERS,
        analysis_name="MotorEncoding",
    )
    selection = build_motor_encoding_selection_snapshot(
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        region_sorted_spikes_group_id=region_sorted_spikes_group_id,
        movement_firing_rate_id=movement_firing_rate_id,
        primary_position_series_name=primary_position_series_name,
        orientation_reference_position_series_name=(
            orientation_reference_position_series_name
        ),
        stability_ids_by_trajectory=stability_ids_by_trajectory,
        parameters=approved,
    )
    output_root = _guarded_output_root(output_dir)
    paths = _motor_output_paths(
        output_root=output_root,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        motor_encoding_id=selection["motor_encoding_id"],
    )
    if paths["artifact_dir"].exists():
        raise FileExistsError(
            "Refusing to overwrite offline MotorEncoding output: "
            f"{paths['artifact_dir']}"
        )
    effective = {
        name: value
        for name, value in approved.items()
        if name != "motor_encoding_param_name"
    }
    result = motor_encoding.compute_motor_encoding(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        motor_encoding_id=selection["motor_encoding_id"],
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        primary_position=primary_position,
        orientation_reference_position=orientation_reference_position,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        trajectory_intervals_by_type=trajectory_intervals_by_type,
        graph_inputs_by_configuration=graph_inputs_by_configuration,
        movement_intervals=movement_intervals,
        movement_firing_rate_table=movement_firing_rate_table,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        parameter_name=approved["motor_encoding_param_name"],
        parameter_sha256=selection["motor_encoding_parameters_sha256"],
        model_spec_sha256=selection[
            "motor_encoding_model_spec_sha256"
        ],
        output_rule_sha256=selection[
            "motor_encoding_output_rule_sha256"
        ],
        **effective,
    )
    if str(result.get("artifact_origin", "")) != "computed":
        raise ValueError("Offline MotorEncoding must be computed de novo.")
    metadata = dict(result.get("metadata", {}))
    if str(metadata.get("motor_encoding_id", "")) != selection[
        "motor_encoding_id"
    ]:
        raise ValueError("MotorEncoding result does not match its selection UUID.")
    motor_encoding.write_motor_encoding_artifact(
        result,
        paths["artifact_dir"],
        overwrite=False,
    )
    artifact_fields = (
        "artifact_manifest_path",
        "selected_units_path",
        "nested_cv_path",
        "full_refit_path",
    )
    record = {
        "analysis_name": "MotorEncoding",
        "selection": selection,
        "effective_parameters": approved,
        "artifact_origin": "computed",
        "analysis_status": str(result["analysis_status"]),
        "n_units_input": int(result["n_units_input"]),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_units_valid": int(result["n_units_valid"]),
        "artifacts": {
            name: _artifact_metadata(
                paths[name],
                output_root=output_root,
            )
            for name in artifact_fields
        },
    }
    record["record_sha256"] = provenance_sha256(record)
    return record


def run_offline_dpp_encoding(
    *,
    output_dir: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    stability_ids_by_trajectory: Mapping[str, Any],
    parameters: Mapping[str, Any],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    """Compute and write one de novo Figure 1 DPPEncoding result."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_1_DPP_ENCODING_PARAMETERS,
        analysis_name="DPPEncoding",
    )
    selection = build_dpp_encoding_selection_snapshot(
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        region_sorted_spikes_group_id=region_sorted_spikes_group_id,
        movement_firing_rate_id=movement_firing_rate_id,
        stability_ids_by_trajectory=stability_ids_by_trajectory,
        parameters=approved,
    )
    output_root = _guarded_output_root(output_dir)
    path = _dpp_output_path(
        output_root=output_root,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        dpp_encoding_id=selection["dpp_encoding_id"],
    )
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite offline DPPEncoding output: {path}"
        )
    effective = {
        name: value
        for name, value in approved.items()
        if name != "dpp_encoding_param_name"
    }
    result = dpp_encoding.compute_selected_dpp_encoding(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        position=position,
        trajectory_intervals_by_type=trajectory_intervals_by_type,
        graph_inputs_by_configuration=graph_inputs_by_configuration,
        movement_intervals=movement_intervals,
        movement_firing_rate_table=movement_firing_rate_table,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        dpp_encoding_id=selection["dpp_encoding_id"],
        **effective,
    )
    table = result["table"]
    observed_ids = (
        []
        if table.empty
        else table["dpp_encoding_id"].astype(str).unique().tolist()
    )
    if observed_ids and observed_ids != [selection["dpp_encoding_id"]]:
        raise ValueError("DPPEncoding result does not match its selection UUID.")
    dpp_encoding.write_dpp_encoding_artifact(
        table,
        path,
        overwrite=False,
    )
    artifact = _artifact_metadata(path, output_root=output_root)
    record = {
        "analysis_name": "DPPEncoding",
        "selection": selection,
        "effective_parameters": approved,
        "artifact_origin": "computed",
        "analysis_status": str(result["analysis_status"]),
        "n_units_input": int(len(stable_unit_ids)),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_units_valid": int(result["n_units_valid"]),
        "eligible_units_sha256": str(result["eligible_units_sha256"]),
        "artifacts": {"dpp_encoding_path": artifact},
    }
    record["record_sha256"] = provenance_sha256(record)
    return record


__all__ = [
    "FIGURE_1_DPP_ENCODING_PARAMETERS",
    "FIGURE_1_MOTOR_ENCODING_PARAMETERS",
    "build_dpp_encoding_selection_snapshot",
    "build_motor_encoding_selection_snapshot",
    "run_offline_dpp_encoding",
    "run_offline_motor_encoding",
]
