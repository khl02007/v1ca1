"""Run focused Figure 2 analyses without DataJoint or legacy artifacts."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import (
    dark_light_glm,
    path_specific_decoding,
    swap_glm,
    tuning_similarity,
)
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import (
    canonical_json,
    provenance_sha256,
    selection_uuid,
    unit_identity_sha256,
)
from v1ca1.spyglass.table_specs import (
    ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS,
    DEFAULT_SWAP_GLM_PARAMETERS,
    DARK_LIGHT_GLM_OUTPUT_RULE,
    LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS,
    MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS,
)

FIGURE_2_TUNING_SIMILARITY_PARAMETERS = MappingProxyType(
    dict(ABSOLUTE_OVERLAP_TUNING_SIMILARITY_PARAMETERS)
)
FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS = MappingProxyType(
    dict(MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS)
)
FIGURE_2_DARK_LIGHT_GLM_PARAMETERS = MappingProxyType(
    dict(LEGACY_V4_V1_DARK_LIGHT_GLM_PARAMETERS)
)
FIGURE_2_SWAP_GLM_PARAMETERS = MappingProxyType(dict(DEFAULT_SWAP_GLM_PARAMETERS))

FIGURE_2_LIGHT_TRAIN_EPOCH = "02_r1"
FIGURE_2_LIGHT_TEST_EPOCH = "06_r3"
FIGURE_2_LIGHT_TRAIN_CONDITION = "AB"
FIGURE_2_LIGHT_TEST_CONDITION = "BA"


def _path_component(value: Any, *, name: str) -> str:
    """Return one non-empty path-safe label."""
    component = str(value)
    if not component or Path(component).name != component or component in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return component


def _source_uuid(value: Any, *, name: str) -> str:
    """Return one canonical UUID identifying an upstream offline result."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def _approved_parameters(
    parameters: Mapping[str, Any],
    *,
    approved: Mapping[str, Any],
    analysis_name: str,
) -> dict[str, Any]:
    """Require the exact approved Figure 2 parameter row."""
    supplied = dict(parameters)
    if canonical_json(supplied) != canonical_json(dict(approved)):
        raise ValueError(
            f"{analysis_name} parameters must exactly match the approved "
            "Figure 2 parameters."
        )
    return supplied


def _trajectory_ids(
    values: Mapping[str, Any],
    *,
    description: str,
) -> dict[str, str]:
    """Return exactly four trajectory-keyed upstream UUIDs."""
    expected = set(TRAJECTORY_TYPES)
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"{description} must contain exactly the four trajectories; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )
    return {
        trajectory_type: _source_uuid(
            values[trajectory_type],
            name=f"{trajectory_type} {description}",
        )
        for trajectory_type in TRAJECTORY_TYPES
    }


def _require_trajectory_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
) -> None:
    """Require a mapping with exactly the four manuscript paths."""
    if set(values) != set(TRAJECTORY_TYPES):
        raise ValueError(f"{name} must contain exactly the four trajectories.")


def _require_tuning_curve_identity(
    tuning_curves_by_trajectory: Mapping[str, Any],
    *,
    expected: Mapping[str, str],
) -> None:
    """Require every tuning curve to match the declared session slice."""
    for trajectory_type, curve in tuning_curves_by_trajectory.items():
        attributes = getattr(curve, "attrs", None)
        if not isinstance(attributes, Mapping):
            raise ValueError(
                f"Tuning curve {trajectory_type!r} has no provenance attributes."
            )
        for field, value in expected.items():
            if str(attributes.get(field)) != str(value):
                raise ValueError(
                    f"Tuning curve {trajectory_type!r} has mismatched {field}."
                )


def _guarded_output_root(output_dir: Path) -> Path:
    """Return one explicit output root without creating or replacing it."""
    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {root}")
    return root


def _guarded_artifact_dir(path: Path, *, output_root: Path) -> Path:
    """Return a canonical artifact directory confined below output_root."""
    artifact_dir = Path(path).resolve(strict=False)
    if not artifact_dir.is_relative_to(output_root) or artifact_dir == output_root:
        raise ValueError("Canonical artifact path escapes output_dir.")
    return artifact_dir


def _relative_dir(path: Path, *, output_root: Path) -> str:
    """Return a guarded run-relative directory path."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_dir() or not resolved.is_relative_to(output_root):
        raise ValueError(f"Artifact directory escaped output_dir: {path}")
    return resolved.relative_to(output_root).as_posix()


def _artifact_metadata(path: Path, *, output_root: Path) -> dict[str, Any]:
    """Return one portable, checksum-bearing file record."""
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(output_root):
        raise ValueError(f"Artifact escaped output_dir: {path}")
    return {
        "relative_path": resolved.relative_to(output_root).as_posix(),
        "file_size_bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def _record_digest(record: Mapping[str, Any]) -> str:
    """Return the provenance digest for a manifest-ready result record."""
    return provenance_sha256(dict(record))


def _validate_record_digest(record: Mapping[str, Any], *, name: str) -> None:
    """Reject a mutated upstream offline result record."""
    payload = dict(record)
    observed = str(payload.pop("record_sha256", ""))
    if not observed or observed != _record_digest(payload):
        raise ValueError(f"{name} record_sha256 is missing or stale.")


def _path_source_fields() -> dict[str, str]:
    """Return fixed trajectory and graph aliases used by Figure 2."""
    fields: dict[str, str] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        fields[f"{trajectory_type}_trajectory_type"] = trajectory_type
        fields[f"{trajectory_type}_configuration_name"] = trajectory_type
    return fields


def build_tuning_similarity_selection_snapshot(
    *,
    tuning_curve_ids_by_trajectory: Mapping[str, Any],
    parameters: Mapping[str, Any] = FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
) -> dict[str, Any]:
    """Build the deterministic four-path absolute-overlap selection."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
        analysis_name="PathSpecificPlaceTuningSimilarity",
    )
    curve_ids = _trajectory_ids(
        tuning_curve_ids_by_trajectory,
        description="tuning_curve_ids_by_trajectory",
    )
    natural_key = {
        **{
            f"{trajectory_type}_tuning_curve_id": curve_ids[trajectory_type]
            for trajectory_type in TRAJECTORY_TYPES
        },
        "tuning_similarity_param_name": approved["tuning_similarity_param_name"],
        "tuning_similarity_parameters_sha256": provenance_sha256(approved),
    }
    return {
        "path_specific_place_tuning_similarity_id": str(
            selection_uuid(
                "PathSpecificPlaceTuningSimilarity",
                natural_key,
            )
        ),
        **natural_key,
    }


def build_path_specific_decoding_selection_snapshot(
    *,
    nwb_file_name: str,
    epoch: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    parameters: Mapping[str, Any] = (FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS),
) -> dict[str, Any]:
    """Build the deterministic within-epoch physical-place selection."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS,
        analysis_name="PathSpecificPlaceDecoding",
    )
    natural_key = {
        "nwb_file_name": _path_component(
            nwb_file_name,
            name="nwb_file_name",
        ),
        "epoch": _path_component(epoch, name="epoch"),
        "region_sorted_spikes_group_id": _source_uuid(
            region_sorted_spikes_group_id,
            name="region_sorted_spikes_group_id",
        ),
        "movement_firing_rate_id": _source_uuid(
            movement_firing_rate_id,
            name="movement_firing_rate_id",
        ),
        **_path_source_fields(),
        "path_specific_place_decoding_param_name": approved[
            "path_specific_place_decoding_param_name"
        ],
        "path_specific_place_decoding_parameters_sha256": (provenance_sha256(approved)),
        "path_specific_place_decoding_output_rule_sha256": provenance_sha256(
            dict(path_specific_decoding.OUTPUT_RULE)
        ),
    }
    return {
        "path_specific_place_decoding_id": str(
            selection_uuid("PathSpecificPlaceDecoding", natural_key)
        ),
        **natural_key,
    }


def build_dark_light_glm_selection_snapshot(
    *,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    dark_epoch: str,
    light_epoch: str,
    dark_movement_firing_rate_id: Any,
    light_movement_firing_rate_id: Any,
    parameters: Mapping[str, Any] = FIGURE_2_DARK_LIGHT_GLM_PARAMETERS,
) -> dict[str, Any]:
    """Build the approved legacy-v4 dark/02_r1 GLM selection."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_DARK_LIGHT_GLM_PARAMETERS,
        analysis_name="DarkLightGLM",
    )
    dark_epoch = _path_component(dark_epoch, name="dark_epoch")
    light_epoch = _path_component(light_epoch, name="light_epoch")
    if light_epoch != FIGURE_2_LIGHT_TRAIN_EPOCH:
        raise ValueError("Figure 2 DarkLightGLM requires light_epoch='02_r1'.")
    if dark_epoch == light_epoch:
        raise ValueError("DarkLightGLM requires distinct dark and light epochs.")
    source_fields: dict[str, str] = {
        "dark_epoch": dark_epoch,
        "light_epoch": light_epoch,
    }
    for epoch_role in ("dark", "light"):
        for trajectory_type in TRAJECTORY_TYPES:
            source_fields[f"{epoch_role}_{trajectory_type}_trajectory_type"] = (
                trajectory_type
            )
    for trajectory_type in TRAJECTORY_TYPES:
        source_fields[f"{trajectory_type}_configuration_name"] = trajectory_type
    natural_key = {
        "nwb_file_name": _path_component(
            nwb_file_name,
            name="nwb_file_name",
        ),
        "region_sorted_spikes_group_id": _source_uuid(
            region_sorted_spikes_group_id,
            name="region_sorted_spikes_group_id",
        ),
        "dark_movement_firing_rate_id": _source_uuid(
            dark_movement_firing_rate_id,
            name="dark_movement_firing_rate_id",
        ),
        "light_movement_firing_rate_id": _source_uuid(
            light_movement_firing_rate_id,
            name="light_movement_firing_rate_id",
        ),
        **source_fields,
        "dark_light_glm_param_name": approved["dark_light_glm_param_name"],
        "dark_light_glm_parameters_sha256": provenance_sha256(approved),
        "dark_light_glm_output_rule_sha256": provenance_sha256(
            dict(DARK_LIGHT_GLM_OUTPUT_RULE)
        ),
    }
    return {
        "dark_light_glm_id": str(selection_uuid("DarkLightGLM", natural_key)),
        **natural_key,
    }


def build_forward_swap_glm_selection_snapshot(
    *,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    dark_light_glm_id: Any,
    light_test_movement_firing_rate_id: Any,
    dark_light_snapshot: Mapping[str, Any],
    dark_condition: str = "dark",
    light_train_condition: str = FIGURE_2_LIGHT_TRAIN_CONDITION,
    light_test_condition: str = FIGURE_2_LIGHT_TEST_CONDITION,
    parameters: Mapping[str, Any] = FIGURE_2_SWAP_GLM_PARAMETERS,
) -> dict[str, Any]:
    """Build the fixed 02_r1-to-06_r3 held-out SwapGLM selection."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_SWAP_GLM_PARAMETERS,
        analysis_name="SwapGLM",
    )
    epochs = {
        "dark_epoch": _path_component(dark_epoch, name="dark_epoch"),
        "light_train_epoch": _path_component(
            light_train_epoch,
            name="light_train_epoch",
        ),
        "light_test_epoch": _path_component(
            light_test_epoch,
            name="light_test_epoch",
        ),
    }
    if epochs["light_train_epoch"] != FIGURE_2_LIGHT_TRAIN_EPOCH or (
        epochs["light_test_epoch"] != FIGURE_2_LIGHT_TEST_EPOCH
    ):
        raise ValueError(
            "Figure 2 SwapGLM requires the forward 02_r1-to-06_r3 transfer."
        )
    if len(set(epochs.values())) != 3:
        raise ValueError("SwapGLM requires three distinct run epochs.")
    conditions = {
        "dark_condition": str(dark_condition),
        "light_train_condition": str(light_train_condition),
        "light_test_condition": str(light_test_condition),
    }
    expected_conditions = {
        "dark_condition": "dark",
        "light_train_condition": FIGURE_2_LIGHT_TRAIN_CONDITION,
        "light_test_condition": FIGURE_2_LIGHT_TEST_CONDITION,
    }
    if conditions != expected_conditions:
        raise ValueError("Figure 2 SwapGLM requires dark/AB/BA epoch conditions.")

    snapshot = dict(dark_light_snapshot)
    required_snapshot_fields = {
        "dark_light_manifest_sha256",
        "dark_light_selected_sha256_by_model",
        "dark_light_parameter_sha256",
        "dark_light_output_rule_sha256",
        "upstream_analysis_status",
        "metadata",
    }
    missing = sorted(required_snapshot_fields.difference(snapshot))
    if missing:
        raise ValueError(f"dark_light_snapshot is missing fields {missing!r}.")
    metadata = dict(snapshot["metadata"])
    expected_metadata = {
        "dark_light_glm_id": _source_uuid(
            dark_light_glm_id,
            name="dark_light_glm_id",
        ),
        "dark_epoch": epochs["dark_epoch"],
        "light_epoch": epochs["light_train_epoch"],
    }
    for name, expected in expected_metadata.items():
        if str(metadata.get(name)) != str(expected):
            raise ValueError(f"DarkLightGLM snapshot has mismatched {name}.")
    source_fields: dict[str, str] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        source_fields[f"light_test_{trajectory_type}_trajectory_type"] = trajectory_type
        source_fields[f"{trajectory_type}_configuration_name"] = trajectory_type
    natural_key = {
        "nwb_file_name": _path_component(
            nwb_file_name,
            name="nwb_file_name",
        ),
        "dark_light_glm_id": expected_metadata["dark_light_glm_id"],
        "region_sorted_spikes_group_id": _source_uuid(
            region_sorted_spikes_group_id,
            name="region_sorted_spikes_group_id",
        ),
        "light_test_movement_firing_rate_id": _source_uuid(
            light_test_movement_firing_rate_id,
            name="light_test_movement_firing_rate_id",
        ),
        **epochs,
        **source_fields,
        "swap_glm_param_name": approved["swap_glm_param_name"],
        **conditions,
        "dark_light_manifest_sha256": str(snapshot["dark_light_manifest_sha256"]),
        "dark_light_selected_sha256_by_model": dict(
            snapshot["dark_light_selected_sha256_by_model"]
        ),
        "dark_light_parameter_sha256": str(snapshot["dark_light_parameter_sha256"]),
        "dark_light_output_rule_sha256": str(snapshot["dark_light_output_rule_sha256"]),
        "upstream_analysis_status": str(snapshot["upstream_analysis_status"]),
        "swap_glm_parameters_sha256": provenance_sha256(approved),
        "swap_glm_output_rule_sha256": swap_glm.OUTPUT_RULE_SHA256,
    }
    return {
        "swap_glm_id": str(selection_uuid("SwapGLM", natural_key)),
        **natural_key,
    }


def run_offline_tuning_similarity(
    *,
    output_dir: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    tuning_curve_ids_by_trajectory: Mapping[str, Any],
    tuning_curves_by_trajectory: Mapping[str, Any],
    movement_firing_rate_table: pd.DataFrame,
    parameters: Mapping[str, Any] = FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
) -> dict[str, Any]:
    """Compute and write one de novo absolute-overlap result."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_TUNING_SIMILARITY_PARAMETERS,
        analysis_name="PathSpecificPlaceTuningSimilarity",
    )
    _require_trajectory_mapping(
        tuning_curves_by_trajectory,
        name="tuning_curves_by_trajectory",
    )
    selection = build_tuning_similarity_selection_snapshot(
        tuning_curve_ids_by_trajectory=tuning_curve_ids_by_trajectory,
        parameters=approved,
    )
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
        }.items()
    }
    _require_tuning_curve_identity(
        tuning_curves_by_trajectory,
        expected=components,
    )
    output_root = _guarded_output_root(output_dir)
    path = tuning_similarity.get_tuning_similarity_artifact_path(
        **components,
        similarity_metric=approved["similarity_metric"],
        path_specific_place_tuning_similarity_id=selection[
            "path_specific_place_tuning_similarity_id"
        ],
        artifact_root=output_root,
    ).resolve(strict=False)
    _guarded_artifact_dir(path.parent, output_root=output_root)
    if path.exists():
        raise FileExistsError(
            "Refusing to overwrite offline tuning similarity output: " f"{path}"
        )
    result = tuning_similarity.compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=tuning_curves_by_trajectory,
        movement_firing_rate_table=movement_firing_rate_table,
        similarity_metric=approved["similarity_metric"],
    )
    tuning_similarity.write_tuning_similarity_artifact(
        result["table"],
        path,
        overwrite=False,
    )
    loaded = tuning_similarity.load_tuning_similarity_artifact(path)
    if not loaded.empty:
        for field, value in components.items():
            if set(loaded[field].astype(str)) != {str(value)}:
                raise ValueError(f"Reloaded tuning similarity has mismatched {field}.")
    record = {
        "analysis_name": "PathSpecificPlaceTuningSimilarity",
        **components,
        "selection": selection,
        "effective_parameters": approved,
        "artifact_origin": "computed",
        "analysis_status": str(result["analysis_status"]),
        "n_units": int(result["n_units"]),
        "n_valid_comparisons": int(result["n_valid_comparisons"]),
        "n_units_with_valid_comparison": int(result["n_units_with_valid_comparison"]),
        "artifacts": {
            "similarity_path": _artifact_metadata(
                path,
                output_root=output_root,
            )
        },
    }
    record["record_sha256"] = _record_digest(record)
    return record


def run_offline_path_specific_decoding(
    *,
    output_dir: Path,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    movement_firing_rate_id: Any,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    parameters: Mapping[str, Any] = (FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS),
) -> dict[str, Any]:
    """Compute and write one de novo within-epoch place decoder."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS,
        analysis_name="PathSpecificPlaceDecoding",
    )
    _require_trajectory_mapping(
        trajectory_intervals_by_type,
        name="trajectory_intervals_by_type",
    )
    _require_trajectory_mapping(
        graph_inputs_by_configuration,
        name="graph_inputs_by_configuration",
    )
    selection = build_path_specific_decoding_selection_snapshot(
        nwb_file_name=nwb_file_name,
        epoch=epoch,
        region_sorted_spikes_group_id=region_sorted_spikes_group_id,
        movement_firing_rate_id=movement_firing_rate_id,
        parameters=approved,
    )
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
        }.items()
    }
    output_root = _guarded_output_root(output_dir)
    paths = path_specific_decoding.get_path_specific_decoding_artifact_paths(
        **components,
        path_specific_place_decoding_id=selection["path_specific_place_decoding_id"],
        artifact_root=output_root,
    )
    artifact_dir = _guarded_artifact_dir(
        paths["artifact_dir"],
        output_root=output_root,
    )
    if artifact_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite offline place-decoding output: " f"{artifact_dir}"
        )
    stable_unit_ids = tuple(dict(row) for row in stable_unit_ids)
    effective = {
        name: value
        for name, value in approved.items()
        if name != "path_specific_place_decoding_param_name"
    }
    result = path_specific_decoding.compute_path_specific_place_decoding(
        **components,
        path_specific_place_decoding_id=selection["path_specific_place_decoding_id"],
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        position=position,
        trajectory_intervals=trajectory_intervals_by_type,
        graph_inputs=graph_inputs_by_configuration,
        movement_interval=movement_intervals,
        parameter_name=approved["path_specific_place_decoding_param_name"],
        parameter_sha256=selection["path_specific_place_decoding_parameters_sha256"],
        output_rule_sha256=selection["path_specific_place_decoding_output_rule_sha256"],
        **effective,
    )
    if str(result.get("artifact_origin", "")) != "computed":
        raise ValueError("Offline PathSpecificPlaceDecoding must be de novo.")
    if (
        str(
            result.get("metadata", {}).get(
                "path_specific_place_decoding_id",
                "",
            )
        )
        != selection["path_specific_place_decoding_id"]
    ):
        raise ValueError("Place-decoding result has a mismatched selection UUID.")
    written = path_specific_decoding.write_path_specific_decoding_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    loaded = path_specific_decoding.load_path_specific_decoding_artifact(artifact_dir)
    if str(loaded["metadata"]["path_specific_place_decoding_id"]) != (
        selection["path_specific_place_decoding_id"]
    ):
        raise ValueError("Reloaded place-decoding artifact has a stale UUID.")
    artifact_fields = (
        "artifact_manifest_path",
        "selected_units_path",
        "fold_qc_path",
        "decoding_summary_path",
        "binned_error_path",
        "true_path",
        "decoded_path",
    )
    record = {
        "analysis_name": "PathSpecificPlaceDecoding",
        **components,
        "selection": selection,
        "effective_parameters": approved,
        "artifact_origin": "computed",
        "analysis_status": str(result["analysis_status"]),
        "n_units": int(result["n_units"]),
        "n_folds_expected": int(result["n_folds_expected"]),
        "n_folds_valid": int(result["n_folds_valid"]),
        "n_decoded_samples": int(result["n_decoded_samples"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "input_units_sha256": unit_identity_sha256(stable_unit_ids),
        "artifact_dir": _relative_dir(
            artifact_dir,
            output_root=output_root,
        ),
        "artifacts": {
            name: _artifact_metadata(
                Path(written[name]),
                output_root=output_root,
            )
            for name in artifact_fields
        },
    }
    record["record_sha256"] = _record_digest(record)
    return record


def run_offline_dark_light_glm(
    *,
    output_dir: Path,
    animal_name: str,
    date: str,
    region: str,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    dark_epoch: str,
    light_epoch: str,
    dark_movement_firing_rate_id: Any,
    light_movement_firing_rate_id: Any,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    dark_movement_firing_rate_table: pd.DataFrame,
    light_movement_firing_rate_table: pd.DataFrame,
    movement_by_epoch: Mapping[str, Any],
    trajectory_intervals_by_epoch: Mapping[str, Mapping[str, Any]],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    position_by_epoch: Mapping[str, Any],
    parameters: Mapping[str, Any] = FIGURE_2_DARK_LIGHT_GLM_PARAMETERS,
) -> dict[str, Any]:
    """Fit and write the approved legacy-v4 dark/02_r1 Figure 2 GLM."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_DARK_LIGHT_GLM_PARAMETERS,
        analysis_name="DarkLightGLM",
    )
    epochs = {
        _path_component(dark_epoch, name="dark_epoch"),
        _path_component(light_epoch, name="light_epoch"),
    }
    for name, values in (
        ("movement_by_epoch", movement_by_epoch),
        ("trajectory_intervals_by_epoch", trajectory_intervals_by_epoch),
        ("position_by_epoch", position_by_epoch),
    ):
        if set(values) != epochs:
            raise ValueError(f"{name} must contain exactly dark and light epochs.")
    for epoch, values in trajectory_intervals_by_epoch.items():
        _require_trajectory_mapping(
            values,
            name=f"trajectory_intervals_by_epoch[{epoch!r}]",
        )
    _require_trajectory_mapping(
        graph_inputs_by_configuration,
        name="graph_inputs_by_configuration",
    )
    selection = build_dark_light_glm_selection_snapshot(
        nwb_file_name=nwb_file_name,
        region_sorted_spikes_group_id=region_sorted_spikes_group_id,
        dark_epoch=dark_epoch,
        light_epoch=light_epoch,
        dark_movement_firing_rate_id=dark_movement_firing_rate_id,
        light_movement_firing_rate_id=light_movement_firing_rate_id,
        parameters=approved,
    )
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "dark_epoch": dark_epoch,
            "light_epoch": light_epoch,
        }.items()
    }
    output_root = _guarded_output_root(output_dir)
    paths = dark_light_glm.get_dark_light_glm_artifact_paths(
        **components,
        dark_light_glm_id=selection["dark_light_glm_id"],
        artifact_root=output_root,
    )
    artifact_dir = _guarded_artifact_dir(
        paths["artifact_dir"],
        output_root=output_root,
    )
    if artifact_dir.exists():
        raise FileExistsError(
            "Refusing to overwrite offline DarkLightGLM output: " f"{artifact_dir}"
        )
    stable_unit_ids = tuple(dict(row) for row in stable_unit_ids)
    effective = {
        name: value
        for name, value in approved.items()
        if name != "dark_light_glm_param_name"
    }
    result = dark_light_glm.compute_dark_light_glm(
        dark_light_glm_id=selection["dark_light_glm_id"],
        **components,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        dark_movement_firing_rate_table=dark_movement_firing_rate_table,
        light_movement_firing_rate_table=light_movement_firing_rate_table,
        movement_by_epoch=movement_by_epoch,
        trajectory_intervals_by_epoch=trajectory_intervals_by_epoch,
        graph_inputs_by_trajectory=graph_inputs_by_configuration,
        position_by_epoch=position_by_epoch,
        parameter_name=approved["dark_light_glm_param_name"],
        parameter_sha256=selection["dark_light_glm_parameters_sha256"],
        output_rule_sha256=selection["dark_light_glm_output_rule_sha256"],
        **effective,
    )
    if str(result.get("artifact_origin", "")) != "computed":
        raise ValueError("Offline DarkLightGLM must be computed de novo.")
    if str(result.get("metadata", {}).get("dark_light_glm_id", "")) != (
        selection["dark_light_glm_id"]
    ):
        raise ValueError("DarkLightGLM result has a mismatched selection UUID.")
    written = dark_light_glm.write_dark_light_glm_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    loaded = dark_light_glm.load_dark_light_glm_artifact(artifact_dir)
    if str(loaded["metadata"]["dark_light_glm_id"]) != selection["dark_light_glm_id"]:
        raise ValueError("Reloaded DarkLightGLM artifact has a stale UUID.")
    artifact_paths: dict[str, Path] = {
        "artifact_manifest_path": Path(written["artifact_manifest_path"]),
        "selected_units_path": Path(written["selected_units_path"]),
        "selection_summary_path": Path(written["selection_summary_path"]),
    }
    artifact_paths.update(
        {
            f"selected_model_{model_name}_path": Path(path)
            for model_name, path in written["selected_model_paths"].items()
        }
    )
    record = {
        "analysis_name": "DarkLightGLM",
        **components,
        "selection": selection,
        "effective_parameters": approved,
        "artifact_origin": "computed",
        "analysis_status": str(result["analysis_status"]),
        "n_units": int(result["n_units"]),
        "n_candidates": int(result["n_candidates"]),
        "n_selected_models": int(result["n_selected_models"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "input_units_sha256": unit_identity_sha256(stable_unit_ids),
        "artifact_dir": _relative_dir(
            artifact_dir,
            output_root=output_root,
        ),
        "artifacts": {
            name: _artifact_metadata(path, output_root=output_root)
            for name, path in artifact_paths.items()
        },
    }
    record["record_sha256"] = _record_digest(record)
    return record


def _load_dark_light_snapshot(
    record: Mapping[str, Any],
    *,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Load and freeze one computed DarkLightGLM wrapper result."""
    _validate_record_digest(record, name="DarkLightGLM")
    if record.get("analysis_name") != "DarkLightGLM" or (
        record.get("artifact_origin") != "computed"
    ):
        raise ValueError("SwapGLM requires a computed DarkLightGLM record.")
    artifact_dir = Path(output_root) / str(record.get("artifact_dir", ""))
    artifact_dir = artifact_dir.resolve(strict=True)
    if (
        not artifact_dir.is_dir()
        or artifact_dir == output_root
        or not artifact_dir.is_relative_to(output_root)
    ):
        raise ValueError("DarkLightGLM artifact directory escapes output_dir.")
    manifest_record = dict(record.get("artifacts", {})).get("artifact_manifest_path")
    if not isinstance(manifest_record, Mapping):
        raise ValueError("DarkLightGLM record is missing its manifest metadata.")
    manifest_path = artifact_dir / dark_light_glm.MANIFEST_FILENAME
    expected_manifest_path = Path(output_root) / str(
        manifest_record.get("relative_path", "")
    )
    if expected_manifest_path.resolve(strict=True) != manifest_path:
        raise ValueError("DarkLightGLM record points to a different manifest.")
    if int(
        manifest_record.get("file_size_bytes", -1)
    ) != manifest_path.stat().st_size or str(
        manifest_record.get("sha256", "")
    ) != file_sha256(
        manifest_path
    ):
        raise ValueError("DarkLightGLM manifest changed after its offline run.")
    loaded = dark_light_glm.load_dark_light_glm_artifact(artifact_dir)
    if loaded.get("artifact_origin") != "computed":
        raise ValueError("SwapGLM upstream must be a de novo computed artifact.")
    selection = dict(record.get("selection", {}))
    expected_parameter_sha256 = provenance_sha256(
        dict(FIGURE_2_DARK_LIGHT_GLM_PARAMETERS)
    )
    if (
        str(selection.get("dark_light_glm_parameters_sha256", ""))
        != expected_parameter_sha256
        or str(selection.get("dark_light_glm_output_rule_sha256", ""))
        != provenance_sha256(dict(DARK_LIGHT_GLM_OUTPUT_RULE))
        or canonical_json(record.get("effective_parameters", {}))
        != canonical_json(dict(FIGURE_2_DARK_LIGHT_GLM_PARAMETERS))
    ):
        raise ValueError("DarkLightGLM record does not use legacy_v4_v1.")
    expected_metadata = {
        "dark_light_glm_id": selection.get("dark_light_glm_id"),
        "dark_epoch": selection.get("dark_epoch"),
        "light_epoch": selection.get("light_epoch"),
    }
    if any(
        str(loaded["metadata"].get(name)) != str(expected)
        for name, expected in expected_metadata.items()
    ):
        raise ValueError("DarkLightGLM artifact disagrees with its selection.")
    if (
        str(loaded["parameters"].get("parameter_sha256")) != expected_parameter_sha256
        or str(loaded["parameters"].get("output_rule_sha256"))
        != provenance_sha256(dict(DARK_LIGHT_GLM_OUTPUT_RULE))
        or str(loaded.get("analysis_status")) != str(record.get("analysis_status"))
    ):
        raise ValueError("DarkLightGLM artifact provenance is stale.")
    manifest = loaded["manifest"]
    selected_sha256_by_model: dict[str, str] = {}
    for model_name in swap_glm.SOURCE_MODEL_NAMES:
        rows = manifest[
            manifest["artifact_key"].astype(str) == f"selected:{model_name}"
        ]
        if len(rows) != 1:
            raise ValueError(f"DarkLightGLM manifest requires selected:{model_name}.")
        selected_sha256_by_model[model_name] = str(rows.iloc[0]["sha256"])
    snapshot = {
        "dark_light_manifest_sha256": file_sha256(manifest_path),
        "dark_light_selected_sha256_by_model": selected_sha256_by_model,
        "dark_light_parameter_sha256": str(loaded["parameters"]["parameter_sha256"]),
        "dark_light_output_rule_sha256": str(
            loaded["parameters"]["output_rule_sha256"]
        ),
        "upstream_analysis_status": str(loaded["analysis_status"]),
        "metadata": dict(loaded["metadata"]),
    }
    return artifact_dir, snapshot


def run_offline_forward_swap_glm(
    *,
    output_dir: Path,
    animal_name: str,
    date: str,
    region: str,
    nwb_file_name: str,
    region_sorted_spikes_group_id: Any,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    dark_light_record: Mapping[str, Any],
    light_test_movement_firing_rate_id: Any,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_interval: Any,
    movement_analysis_status: str,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    position: Any,
    dark_condition: str = "dark",
    light_train_condition: str = FIGURE_2_LIGHT_TRAIN_CONDITION,
    light_test_condition: str = FIGURE_2_LIGHT_TEST_CONDITION,
    parameters: Mapping[str, Any] = FIGURE_2_SWAP_GLM_PARAMETERS,
) -> dict[str, Any]:
    """Score only the Figure 2 02_r1-to-06_r3 held-out transfer."""
    approved = _approved_parameters(
        parameters,
        approved=FIGURE_2_SWAP_GLM_PARAMETERS,
        analysis_name="SwapGLM",
    )
    _require_trajectory_mapping(
        trajectory_intervals_by_type,
        name="trajectory_intervals_by_type",
    )
    _require_trajectory_mapping(
        graph_inputs_by_configuration,
        name="graph_inputs_by_configuration",
    )
    output_root = _guarded_output_root(output_dir)
    dark_light_artifact_dir, dark_light_snapshot = _load_dark_light_snapshot(
        dark_light_record,
        output_root=output_root,
    )
    dark_light_selection = dict(dark_light_record["selection"])
    selection = build_forward_swap_glm_selection_snapshot(
        nwb_file_name=nwb_file_name,
        region_sorted_spikes_group_id=region_sorted_spikes_group_id,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
        dark_light_glm_id=dark_light_selection["dark_light_glm_id"],
        light_test_movement_firing_rate_id=(light_test_movement_firing_rate_id),
        dark_light_snapshot=dark_light_snapshot,
        dark_condition=dark_condition,
        light_train_condition=light_train_condition,
        light_test_condition=light_test_condition,
        parameters=approved,
    )
    if str(dark_light_selection.get("nwb_file_name")) != str(nwb_file_name) or (
        str(dark_light_selection.get("region_sorted_spikes_group_id"))
        != str(selection["region_sorted_spikes_group_id"])
    ):
        raise ValueError("SwapGLM and DarkLightGLM must share NWB and spike group.")
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "dark_epoch": dark_epoch,
            "light_train_epoch": light_train_epoch,
            "light_test_epoch": light_test_epoch,
        }.items()
    }
    paths = swap_glm.get_swap_glm_artifact_paths(
        **components,
        swap_glm_id=selection["swap_glm_id"],
        artifact_root=output_root,
    )
    artifact_dir = _guarded_artifact_dir(
        paths["artifact_dir"],
        output_root=output_root,
    )
    if artifact_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite offline SwapGLM output: {artifact_dir}"
        )
    stable_unit_ids = tuple(dict(row) for row in stable_unit_ids)
    effective = {
        name: value for name, value in approved.items() if name != "swap_glm_param_name"
    }
    result = swap_glm.compute_swap_glm(
        swap_glm_id=selection["swap_glm_id"],
        **components,
        dark_light_glm_artifact_path=dark_light_artifact_dir,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        movement_interval=movement_interval,
        movement_analysis_status=movement_analysis_status,
        trajectory_intervals=trajectory_intervals_by_type,
        graph_inputs_by_trajectory=graph_inputs_by_configuration,
        position=position,
        parameter_name=approved["swap_glm_param_name"],
        parameter_sha256=selection["swap_glm_parameters_sha256"],
        output_rule_sha256=selection["swap_glm_output_rule_sha256"],
        **effective,
    )
    if str(result.get("artifact_origin", "")) != "computed":
        raise ValueError("Offline SwapGLM must be computed de novo.")
    if (
        str(result.get("metadata", {}).get("swap_glm_id", ""))
        != selection["swap_glm_id"]
    ):
        raise ValueError("SwapGLM result has a mismatched selection UUID.")
    written = swap_glm.write_swap_glm_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    loaded = swap_glm.load_swap_glm_artifact(artifact_dir)
    if str(loaded["metadata"]["swap_glm_id"]) != selection["swap_glm_id"]:
        raise ValueError("Reloaded SwapGLM artifact has a stale UUID.")
    artifact_fields = (
        "artifact_manifest_path",
        "selected_units_path",
        "result_path",
    )
    record = {
        "analysis_name": "SwapGLM",
        **components,
        "selection": selection,
        "effective_parameters": approved,
        "artifact_origin": "computed",
        "analysis_status": str(result["analysis_status"]),
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "input_units_sha256": unit_identity_sha256(stable_unit_ids),
        "artifact_dir": _relative_dir(
            artifact_dir,
            output_root=output_root,
        ),
        "artifacts": {
            name: _artifact_metadata(
                Path(written[name]),
                output_root=output_root,
            )
            for name in artifact_fields
        },
    }
    record["record_sha256"] = _record_digest(record)
    return record


__all__ = [
    "FIGURE_2_DARK_LIGHT_GLM_PARAMETERS",
    "FIGURE_2_LIGHT_TEST_CONDITION",
    "FIGURE_2_LIGHT_TEST_EPOCH",
    "FIGURE_2_LIGHT_TRAIN_CONDITION",
    "FIGURE_2_LIGHT_TRAIN_EPOCH",
    "FIGURE_2_PATH_SPECIFIC_DECODING_PARAMETERS",
    "FIGURE_2_SWAP_GLM_PARAMETERS",
    "FIGURE_2_TUNING_SIMILARITY_PARAMETERS",
    "build_dark_light_glm_selection_snapshot",
    "build_forward_swap_glm_selection_snapshot",
    "build_path_specific_decoding_selection_snapshot",
    "build_tuning_similarity_selection_snapshot",
    "run_offline_dark_light_glm",
    "run_offline_forward_swap_glm",
    "run_offline_path_specific_decoding",
    "run_offline_tuning_similarity",
]
