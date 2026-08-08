"""Database-free epoch-level motor-behavior artifact bundles.

The adapter keeps the scientific implementation in
``v1ca1.motor.compare_epoch_motor_behavior`` authoritative.  Each result is
one run epoch and contains the six legacy motor-variable distribution rows,
the four-path progression summary, and explicit trajectory QC.  Position
series names and roles are deliberately not hard-coded to ``head`` or
``body``: the selected primary series defines translation and the selected
orientation-reference series defines the primary-minus-reference head vector.

For manuscript compatibility, jointly non-finite position rows are removed
before the existing motor helper is called.  Derivatives therefore span any
gaps created by that removal, and progression medians and quartiles use the
legacy linear-angle policy even for head direction.  Both choices are fixed in
``OUTPUT_RULE`` rather than exposed as silent options.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES, build_movement_interval


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "epoch_motor_behavior"
MANIFEST_FILENAME = "manifest.parquet"
DISTRIBUTION_FILENAME = "distribution_summary.parquet"
PROGRESSION_FILENAME = "progression_summary.parquet"
TRAJECTORY_QC_FILENAME = "trajectory_qc.parquet"
SCHEMA_VERSION = "1"
BUNDLE_SCHEMA_VERSION = "1"

DEFAULT_PROGRESSION_BIN_SIZE_CM = 4.0
DEFAULT_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_SPEED_SMOOTHING_SIGMA_S = 0.1

MANUSCRIPT_PARAMETERS = MappingProxyType(
    {
        "progression_bin_size_cm": DEFAULT_PROGRESSION_BIN_SIZE_CM,
    }
)
MANUSCRIPT_MOVEMENT_PARAMETERS = MappingProxyType(
    {
        "movement_param_name": "default",
        "speed_threshold_cm_s": DEFAULT_SPEED_THRESHOLD_CM_S,
        "speed_smoothing_sigma_s": DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    }
)

OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "row_granularity": "one_session_run_epoch",
        "primary_position_semantics": "translation_and_track_linearization",
        "orientation_reference_semantics": "primary_minus_reference_head_vector",
        "position_unit": "cm",
        "time_unit": "s",
        "position_offset_policy": "use_catalog_offset_without_additional_truncation",
        "nonfinite_policy": "drop_joint_rows_then_differentiate_across_remaining_gaps",
        "speed_smoothing_implementation": "compare_epoch_motor_behavior_fixed_sigma_0.1_s",
        "movement_threshold_policy": "strictly_greater_than_threshold",
        "trajectory_order": tuple(TRAJECTORY_TYPES),
        "trajectory_interval_boundary_policy": "closed",
        "graph_orientation": "natural_trajectory_direction",
        "graph_edge_spacing_cm": 0.0,
        "progression_range": (0.0, 1.0),
        "progression_empty_bin_policy": "omit",
        "head_direction_distribution_policy": "circular_mean_and_resultant_length",
        "head_direction_progression_policy": "legacy_linear_median_and_quartiles",
        "motor_variables": (
            "speed_cm_s",
            "acceleration_cm_s2",
            "head_direction_deg",
            "head_angular_velocity_deg_s",
            "head_angular_acceleration_deg_s2",
            "head_angular_speed_deg_s",
        ),
        "pairwise_distribution_artifact": "excluded",
    }
)

ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_valid_position",
    "no_movement",
    "no_trials",
)
TRAJECTORY_STATUSES = (
    "valid",
    "no_valid_position",
    "no_movement",
    "no_trials",
    "no_movement_samples",
)

DISTRIBUTION_COLUMNS = (
    "epoch",
    "variable",
    "sample_count",
    "movement_duration_s",
    "mean",
    "median",
    "std",
    "p10",
    "p90",
    "circular_mean_deg",
    "resultant_length",
)
PROGRESSION_COLUMNS = (
    "epoch",
    "trajectory_type",
    "variable",
    "progression_bin_index",
    "progression_bin_start",
    "progression_bin_end",
    "progression_bin_center",
    "sample_count",
    "median",
    "q25",
    "q75",
)
TRAJECTORY_QC_COLUMNS = (
    "epoch_motor_behavior_id",
    "animal_name",
    "date",
    "epoch",
    "trajectory_type",
    "trajectory_interval_count",
    "trajectory_interval_duration_s",
    "movement_supported_duration_s",
    "movement_supported_sample_count",
    "finite_progression_sample_count",
    "occupied_progression_bin_count",
    "graph_length_cm",
    "trajectory_status",
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "schema_version",
    "bundle_schema_version",
    "epoch_motor_behavior_id",
    "animal_name",
    "date",
    "epoch",
    "epoch_type",
    "primary_position_source",
    "primary_position_role",
    "orientation_reference_position_source",
    "orientation_reference_position_role",
    "position_offset_samples",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "progression_bin_size_cm",
    "movement_param_name",
    "movement_parameters_sha256",
    "speed_threshold_cm_s",
    "speed_smoothing_sigma_s",
    "n_position_samples_input",
    "n_finite_position_samples",
    "n_dropped_nonfinite_samples",
    "n_movement_samples",
    "movement_duration_s",
    "n_supported_trajectories",
    "sampling_rate_hz",
    "median_sample_interval_s",
    "maximum_sample_gap_s",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
)

_POSITION_ROW_REQUIRED_FIELDS = (
    "nwb_file_name",
    "epoch",
    "position_series_name",
    "position_role",
    "spatial_unit",
    "start_index",
    "stop_index_exclusive",
    "sample_count",
    "analysis_start_offset_samples",
    "start_time",
    "stop_time",
    "first_frame",
    "last_frame",
    "video_series_name",
)
_ALIGNED_POSITION_ROW_FIELDS = (
    "nwb_file_name",
    "epoch",
    "start_index",
    "stop_index_exclusive",
    "sample_count",
    "analysis_start_offset_samples",
    "start_time",
    "stop_time",
    "first_frame",
    "last_frame",
    "video_series_name",
)


def _motor_module() -> Any:
    """Return the authoritative legacy motor helper module."""
    motor = importlib.import_module("v1ca1.motor.compare_epoch_motor_behavior")

    if tuple(motor.MOTOR_VARIABLES) != tuple(OUTPUT_RULE["motor_variables"]):
        raise RuntimeError("The fixed six-variable motor specification changed.")
    return motor


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe non-empty path component."""
    component = str(value)
    if not component or Path(component).name != component or component in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return component


def _uuid_string(value: Any, *, name: str) -> str:
    """Return one canonical UUID string."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def _json_safe(value: Any) -> Any:
    """Return one stable JSON-compatible provenance payload."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _provenance_sha256(value: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for one mapping."""
    payload = json.dumps(
        _json_safe(dict(value)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_epoch_motor_behavior_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    epoch_motor_behavior_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first motor artifact bundle."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
        }.items()
    }
    result_id = _uuid_string(
        epoch_motor_behavior_id,
        name="epoch_motor_behavior_id",
    )
    artifact_dir = (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / result_id
    )
    return _paths_for_directory(artifact_dir)


def _paths_for_directory(directory: Path) -> dict[str, Path]:
    """Return canonical child paths for one artifact directory."""
    directory = Path(directory)
    return {
        "artifact_dir": directory,
        "artifact_manifest_path": directory / MANIFEST_FILENAME,
        "distribution_summary_path": directory / DISTRIBUTION_FILENAME,
        "progression_summary_path": directory / PROGRESSION_FILENAME,
        "trajectory_qc_path": directory / TRAJECTORY_QC_FILENAME,
    }


def _finite_float(value: Any, *, name: str) -> float:
    """Return one finite non-boolean numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one numeric scalar.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def validate_epoch_motor_behavior_parameters(
    *,
    progression_bin_size_cm: float = DEFAULT_PROGRESSION_BIN_SIZE_CM,
) -> dict[str, float]:
    """Return validated EpochMotorBehavior-owned parameters."""
    bin_size = _finite_float(
        progression_bin_size_cm,
        name="progression_bin_size_cm",
    )
    if bin_size <= 0.0:
        raise ValueError("progression_bin_size_cm must be positive.")
    return {"progression_bin_size_cm": bin_size}


def validate_movement_parameter_snapshot(
    movement_parameters: Mapping[str, Any] | None = None,
    *,
    movement_parameters_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the separately selected upstream MovementParameters row."""
    values = dict(MANUSCRIPT_MOVEMENT_PARAMETERS)
    if movement_parameters is not None:
        if not isinstance(movement_parameters, Mapping):
            raise TypeError("movement_parameters must be a mapping.")
        values = dict(movement_parameters)
    embedded_sha256 = values.pop("movement_parameters_sha256", None)
    if movement_parameters_sha256 is None:
        movement_parameters_sha256 = embedded_sha256
    elif embedded_sha256 is not None and str(embedded_sha256) != str(
        movement_parameters_sha256
    ):
        raise ValueError("Conflicting movement_parameters_sha256 values.")
    expected_fields = set(MANUSCRIPT_MOVEMENT_PARAMETERS)
    if set(values) != expected_fields:
        raise ValueError(
            "movement_parameters must contain exactly movement_param_name, "
            "speed_threshold_cm_s, and speed_smoothing_sigma_s."
        )
    name = str(values["movement_param_name"]).strip()
    if not name or len(name) > 64:
        raise ValueError("movement_param_name must be non-empty and at most 64 characters.")
    threshold = _finite_float(
        values["speed_threshold_cm_s"],
        name="speed_threshold_cm_s",
    )
    sigma = _finite_float(
        values["speed_smoothing_sigma_s"],
        name="speed_smoothing_sigma_s",
    )
    if not np.isclose(
        threshold,
        DEFAULT_SPEED_THRESHOLD_CM_S,
        rtol=0.0,
        atol=1e-15,
    ) or not np.isclose(
        sigma,
        DEFAULT_SPEED_SMOOTHING_SIGMA_S,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError(
            "EpochMotorBehavior currently requires the manuscript MovementParameters "
            "values speed_threshold_cm_s=4.0 and speed_smoothing_sigma_s=0.1."
        )
    snapshot = {
        "movement_param_name": name,
        "speed_threshold_cm_s": threshold,
        "speed_smoothing_sigma_s": sigma,
    }
    expected_sha256 = _provenance_sha256(snapshot)
    if movement_parameters_sha256 is None:
        movement_parameters_sha256 = expected_sha256
    if str(movement_parameters_sha256) != expected_sha256:
        raise ValueError(
            "movement_parameters_sha256 does not match the selected "
            "MovementParameters row."
        )
    return {**snapshot, "movement_parameters_sha256": expected_sha256}


def _integer(value: Any, *, name: str) -> int:
    """Return one non-negative integer scalar."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _position_arrays(position: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned XY values and timestamps without rejecting joint NaNs."""
    values = np.asarray(getattr(position, "d", position), dtype=float)
    timestamps = np.asarray(getattr(position, "t", ()), dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_samples, 2).")
    if values.shape[0] != timestamps.size:
        raise ValueError(f"{name} values and timestamps must align.")
    return values, timestamps


def validate_position_inputs(
    *,
    epoch: str,
    primary_position: Any,
    orientation_reference_position: Any,
    primary_position_row: Mapping[str, Any],
    orientation_reference_position_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate two distinct, already-offset centimeter position selections."""
    rows = {
        "primary": dict(primary_position_row),
        "orientation-reference": dict(orientation_reference_position_row),
    }
    for label, row in rows.items():
        missing = sorted(set(_POSITION_ROW_REQUIRED_FIELDS).difference(row))
        if missing:
            raise ValueError(f"{label} position row is missing fields {missing!r}.")
        if str(row["epoch"]) != str(epoch):
            raise ValueError(f"{label} position row does not match epoch {epoch!r}.")
        if str(row["spatial_unit"]) != "cm":
            raise ValueError(f"{label} position must use centimeters.")
        if not str(row["position_series_name"]).strip():
            raise ValueError(f"{label} position series name must be non-empty.")
        if not str(row["position_role"]).strip():
            raise ValueError(f"{label} position role must be non-empty.")

    primary_row = rows["primary"]
    reference_row = rows["orientation-reference"]
    if str(primary_row["position_series_name"]) == str(
        reference_row["position_series_name"]
    ):
        raise ValueError(
            "Primary and orientation-reference positions must be distinct series."
        )
    for field in _ALIGNED_POSITION_ROW_FIELDS:
        if str(primary_row[field]) != str(reference_row[field]):
            raise ValueError(
                "Primary and orientation-reference position rows must share exact "
                f"sampling metadata: {field}."
            )

    start = _integer(primary_row["start_index"], name="start_index")
    stop = _integer(
        primary_row["stop_index_exclusive"],
        name="stop_index_exclusive",
    )
    sample_count = _integer(primary_row["sample_count"], name="sample_count")
    offset = _integer(
        primary_row["analysis_start_offset_samples"],
        name="analysis_start_offset_samples",
    )
    if stop < start or stop - start != sample_count:
        raise ValueError("Position half-open bounds do not match sample_count.")
    if sample_count == 0 or offset >= sample_count:
        raise ValueError("Position analysis offset must lie within stored samples.")

    primary_values, primary_times = _position_arrays(
        primary_position,
        name="primary_position",
    )
    reference_values, reference_times = _position_arrays(
        orientation_reference_position,
        name="orientation_reference_position",
    )
    expected_loaded_count = sample_count - offset
    if primary_values.shape[0] != expected_loaded_count or (
        reference_values.shape[0] != expected_loaded_count
    ):
        raise ValueError(
            "Loaded position sample count must equal catalog sample_count minus "
            "analysis_start_offset_samples; do not truncate a second time."
        )
    if not np.array_equal(primary_times, reference_times, equal_nan=True):
        raise ValueError(
            "Primary and orientation-reference timestamps must match exactly."
        )
    return {
        "primary_values": primary_values,
        "orientation_reference_values": reference_values,
        "timestamps": primary_times,
        "primary_position_source": str(primary_row["position_series_name"]),
        "primary_position_role": str(primary_row["position_role"]),
        "orientation_reference_position_source": str(
            reference_row["position_series_name"]
        ),
        "orientation_reference_position_role": str(
            reference_row["position_role"]
        ),
        "position_offset_samples": offset,
        "n_position_samples_input": expected_loaded_count,
    }


def _interval_bounds(intervals: Any, *, name: str) -> np.ndarray:
    """Return validated sorted, non-overlapping second-based bounds."""
    try:
        starts = np.asarray(intervals.start, dtype=float).reshape(-1)
        ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must expose numeric start and end arrays.") from exc
    if starts.shape != ends.shape:
        raise ValueError(f"{name} starts and ends must align.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError(f"{name} bounds must be finite seconds.")
    if np.any(ends <= starts):
        raise ValueError(f"{name} stops must be strictly after starts.")
    if starts.size > 1 and (
        np.any(np.diff(starts) < 0.0) or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError(f"{name} bounds must be sorted and non-overlapping.")
    return np.column_stack((starts, ends))


def _validate_trajectory_inputs(
    trajectory_intervals_by_type: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Return exact four-path trajectory bounds."""
    if not isinstance(trajectory_intervals_by_type, Mapping):
        raise TypeError("trajectory_intervals_by_type must be a mapping.")
    expected = set(TRAJECTORY_TYPES)
    actual = set(trajectory_intervals_by_type)
    if actual != expected:
        raise ValueError(
            "trajectory_intervals_by_type must contain exactly four paths; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )
    return {
        trajectory: _interval_bounds(
            trajectory_intervals_by_type[trajectory],
            name=f"{trajectory} intervals",
        )
        for trajectory in TRAJECTORY_TYPES
    }


def _validate_graph_inputs(
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Return equal natural-direction graph lengths in centimeters."""
    if not isinstance(graph_inputs_by_configuration, Mapping):
        raise TypeError("graph_inputs_by_configuration must be a mapping.")
    expected = set(TRAJECTORY_TYPES)
    actual = set(graph_inputs_by_configuration)
    if actual != expected:
        raise ValueError(
            "graph_inputs_by_configuration must contain exactly the four natural "
            f"trajectory graphs; missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )
    from v1ca1.spyglass.stability import _ordered_graph_length

    lengths: dict[str, float] = {}
    for trajectory in TRAJECTORY_TYPES:
        graph = dict(graph_inputs_by_configuration[trajectory])
        if str(graph.get("configuration_name", "")) != trajectory:
            raise ValueError(
                "WTrackGraph configuration_name must match its trajectory key."
            )
        if graph.get("coordinate_unit") != "cm":
            raise ValueError("WTrackGraph coordinate_unit must be 'cm'.")
        track_kwargs = dict(graph.get("track_graph_kwargs", {}))
        linear_kwargs = dict(graph.get("linearization_kwargs", {}))
        if set(track_kwargs) != {"node_positions", "edges"}:
            raise ValueError("WTrackGraph track_graph_kwargs are incomplete.")
        spacing = np.asarray(
            linear_kwargs.get("edge_spacing", ()),
            dtype=float,
        ).reshape(-1)
        if np.any(spacing != 0.0):
            raise ValueError(
                "Epoch motor behavior requires zero graph edge spacing."
            )
        use_hmm = bool(linear_kwargs.get("use_HMM", graph.get("use_hmm", False)))
        if use_hmm:
            raise ValueError(
                "Epoch motor behavior requires deterministic non-HMM linearization."
            )
        lengths[trajectory] = _ordered_graph_length(
            np.asarray(track_kwargs["node_positions"], dtype=float),
            linear_kwargs.get("edge_order", ()),
            spacing,
        )
    reference = lengths[TRAJECTORY_TYPES[0]]
    if any(
        not np.isclose(lengths[name], reference, rtol=1e-10, atol=1e-12)
        for name in TRAJECTORY_TYPES[1:]
    ):
        raise ValueError("The four trajectory graph lengths must match.")
    return lengths


def _progression_bin_edges(
    *,
    graph_length_cm: float,
    progression_bin_size_cm: float,
) -> np.ndarray:
    """Return legacy-compatible normalized edges from selected graph length."""
    bin_size = float(progression_bin_size_cm) / float(graph_length_cm)
    edges = np.arange(0.0, 1.0 + bin_size, bin_size, dtype=float)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)
    else:
        edges[-1] = 1.0
    return edges


def _intersection_duration(first: np.ndarray, second: np.ndarray) -> float:
    """Return the duration of two internally non-overlapping interval sets."""
    duration = 0.0
    for first_start, first_stop in first:
        for second_start, second_stop in second:
            duration += max(
                0.0,
                min(float(first_stop), float(second_stop))
                - max(float(first_start), float(second_start)),
            )
    return float(duration)


def empty_trajectory_qc_table() -> pd.DataFrame:
    """Return an empty trajectory-QC table with canonical columns."""
    return pd.DataFrame(
        {
            "epoch_motor_behavior_id": pd.Series(dtype=str),
            "animal_name": pd.Series(dtype=str),
            "date": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "trajectory_type": pd.Series(dtype=str),
            "trajectory_interval_count": pd.Series(dtype=np.int64),
            "trajectory_interval_duration_s": pd.Series(dtype=float),
            "movement_supported_duration_s": pd.Series(dtype=float),
            "movement_supported_sample_count": pd.Series(dtype=np.int64),
            "finite_progression_sample_count": pd.Series(dtype=np.int64),
            "occupied_progression_bin_count": pd.Series(dtype=np.int64),
            "graph_length_cm": pd.Series(dtype=float),
            "trajectory_status": pd.Series(dtype=str),
        }
    ).loc[:, list(TRAJECTORY_QC_COLUMNS)]


def _empty_controlled_data(
    motor_variables: Sequence[str],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return empty progression/value vectors for every path and variable."""
    return {
        trajectory: {
            variable: (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
            )
            for variable in motor_variables
        }
        for trajectory in TRAJECTORY_TYPES
    }


def _sampling_qc(timestamps: np.ndarray) -> tuple[float, float, float]:
    """Return inferred rate, median sample interval, and maximum gap."""
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    if timestamps.size < 2:
        return np.nan, np.nan, np.nan
    differences = np.diff(timestamps)
    rate = float((timestamps.size - 1) / (timestamps[-1] - timestamps[0]))
    return rate, float(np.median(differences)), float(np.max(differences))


def _make_filtered_position(timestamps: np.ndarray, values: np.ndarray) -> Any:
    """Return one Pynapple position frame for graph linearization."""
    import pynapple as nap

    return nap.TsdFrame(
        t=np.asarray(timestamps, dtype=float),
        d=np.asarray(values, dtype=float),
        columns=["x", "y"],
        time_units="s",
    )


def _trajectory_progression_values(
    *,
    filtered_position: Any,
    filtered_timestamps: np.ndarray,
    trajectory_intervals: Any,
    trajectory_bounds: np.ndarray,
    graph_inputs: Mapping[str, Any],
    trajectory_type: str,
    expected_graph_length_cm: float,
) -> np.ndarray:
    """Return selected natural-direction progression aligned to path samples."""
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    progression, graph_length = build_task_progression_from_graph(
        position=filtered_position,
        trajectory_interval=trajectory_intervals,
        graph_inputs=graph_inputs,
        trajectory_type=trajectory_type,
    )
    if not np.isclose(
        graph_length,
        expected_graph_length_cm,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Linearization and selected WTrackGraph length disagree.")
    motor = _motor_module()
    trajectory_mask = motor.build_interval_mask(
        filtered_timestamps,
        trajectory_bounds,
    )
    expected_times = np.asarray(filtered_timestamps[trajectory_mask], dtype=float)
    progression_times = np.asarray(progression.t, dtype=float).reshape(-1)
    progression_values = np.asarray(progression.d, dtype=float).reshape(-1)
    if progression_times.shape != progression_values.shape or not np.array_equal(
        progression_times,
        expected_times,
    ):
        raise ValueError(
            "Graph progression timestamps do not match closed trajectory samples."
        )
    return np.clip(progression_values, 0.0, 1.0)


def _parameter_snapshot(
    *,
    parameter_name: str,
    parameters: Mapping[str, float],
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
) -> dict[str, Any]:
    """Return validated named parameters and immutable digests."""
    name = str(parameter_name).strip()
    if not name or len(name) > 64:
        raise ValueError("parameter_name must be non-empty and at most 64 characters.")
    payload = {
        "epoch_motor_behavior_param_name": name,
        **dict(parameters),
    }
    expected_parameter_sha256 = _provenance_sha256(payload)
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    if str(parameter_sha256) != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    expected_output_rule_sha256 = _provenance_sha256(dict(OUTPUT_RULE))
    if output_rule_sha256 is None:
        output_rule_sha256 = expected_output_rule_sha256
    if str(output_rule_sha256) != expected_output_rule_sha256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": name,
        "parameter_sha256": str(parameter_sha256),
        "output_rule_sha256": str(output_rule_sha256),
        **dict(parameters),
    }


def compute_selected_epoch_motor_behavior(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    epoch_motor_behavior_id: Any,
    primary_position: Any,
    orientation_reference_position: Any,
    primary_position_row: Mapping[str, Any],
    orientation_reference_position_row: Mapping[str, Any],
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    epoch_type: str = "run",
    parameter_name: str = "manuscript_4cm",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    progression_bin_size_cm: float = DEFAULT_PROGRESSION_BIN_SIZE_CM,
    movement_parameters: Mapping[str, Any] | None = None,
    movement_parameters_sha256: str | None = None,
    artifact_origin: str = "computed",
    legacy_artifact_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one epoch's canonical motor summaries from selected NWB data."""
    animal = _path_component(animal_name, name="animal_name")
    session_date = _path_component(date, name="date")
    epoch_name = _path_component(epoch, name="epoch")
    result_id = _uuid_string(
        epoch_motor_behavior_id,
        name="epoch_motor_behavior_id",
    )
    if str(epoch_type) != "run":
        raise ValueError("EpochMotorBehavior requires a run epoch.")
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("artifact_origin must be computed or registered_existing.")
    parameters = validate_epoch_motor_behavior_parameters(
        progression_bin_size_cm=progression_bin_size_cm,
    )
    movement_snapshot = validate_movement_parameter_snapshot(
        movement_parameters,
        movement_parameters_sha256=movement_parameters_sha256,
    )
    parameter_snapshot = _parameter_snapshot(
        parameter_name=parameter_name,
        parameters=parameters,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
    )
    position_inputs = validate_position_inputs(
        epoch=epoch_name,
        primary_position=primary_position,
        orientation_reference_position=orientation_reference_position,
        primary_position_row=primary_position_row,
        orientation_reference_position_row=orientation_reference_position_row,
    )
    trajectory_bounds = _validate_trajectory_inputs(
        trajectory_intervals_by_type
    )
    graph_lengths = _validate_graph_inputs(graph_inputs_by_configuration)
    common_graph_length = graph_lengths[TRAJECTORY_TYPES[0]]
    progression_edges = _progression_bin_edges(
        graph_length_cm=common_graph_length,
        progression_bin_size_cm=parameters["progression_bin_size_cm"],
    )
    motor = _motor_module()
    motor_variables = tuple(motor.MOTOR_VARIABLES)

    raw_timestamps = np.asarray(position_inputs["timestamps"], dtype=float)
    raw_primary = np.asarray(position_inputs["primary_values"], dtype=float)
    raw_reference = np.asarray(
        position_inputs["orientation_reference_values"],
        dtype=float,
    )
    finite_mask = (
        np.isfinite(raw_timestamps)
        & np.all(np.isfinite(raw_primary), axis=1)
        & np.all(np.isfinite(raw_reference), axis=1)
    )
    n_finite = int(np.sum(finite_mask))
    n_dropped = int(finite_mask.size - n_finite)
    filtered_timestamps = raw_timestamps[finite_mask]
    filtered_primary = raw_primary[finite_mask]
    filtered_reference = raw_reference[finite_mask]

    movement_values = {
        variable: np.asarray([], dtype=float) for variable in motor_variables
    }
    controlled_data = _empty_controlled_data(motor_variables)
    movement_duration_s = 0.0
    movement_sample_count = 0
    movement_bounds = np.empty((0, 2), dtype=float)
    trajectory_rows: list[dict[str, Any]] = []

    if n_finite < 2:
        analysis_status = "no_valid_position"
        sampling_rate_hz, median_dt_s, maximum_gap_s = (np.nan, np.nan, np.nan)
        for trajectory in TRAJECTORY_TYPES:
            bounds = trajectory_bounds[trajectory]
            trajectory_rows.append(
                {
                    "trajectory_type": trajectory,
                    "trajectory_interval_count": len(bounds),
                    "trajectory_interval_duration_s": float(
                        np.sum(bounds[:, 1] - bounds[:, 0])
                    ),
                    "movement_supported_duration_s": 0.0,
                    "movement_supported_sample_count": 0,
                    "finite_progression_sample_count": 0,
                    "occupied_progression_bin_count": 0,
                    "graph_length_cm": graph_lengths[trajectory],
                    "trajectory_status": "no_valid_position",
                }
            )
    else:
        (
            filtered_timestamps,
            filtered_primary,
            filtered_reference,
            helper_dropped_count,
        ) = motor.filter_finite_position_samples(
            raw_timestamps,
            raw_primary,
            raw_reference,
        )
        if helper_dropped_count != n_dropped:
            raise RuntimeError("Finite-position filtering policy changed.")
        sampling_rate_hz, median_dt_s, maximum_gap_s = _sampling_qc(
            filtered_timestamps
        )
        computed_variables = motor.compute_motor_variables(
            filtered_primary,
            filtered_reference,
            filtered_timestamps,
        )
        if set(computed_variables) != set(motor_variables):
            raise RuntimeError("Motor helper returned an unexpected variable set.")
        import pynapple as nap

        speed_tsd = nap.Tsd(
            t=filtered_timestamps,
            d=np.asarray(computed_variables["speed_cm_s"], dtype=float),
            time_units="s",
        )
        movement_interval = build_movement_interval(
            speed_tsd,
            speed_threshold_cm_s=movement_snapshot["speed_threshold_cm_s"],
        )
        movement_bounds = _interval_bounds(
            movement_interval,
            name="computed movement intervals",
        )
        movement_duration_s = float(movement_interval.tot_length())
        movement_mask = motor.build_interval_mask(
            filtered_timestamps,
            movement_bounds,
        )
        movement_sample_count = int(np.sum(movement_mask))
        movement_values = {
            variable: np.asarray(computed_variables[variable][movement_mask], dtype=float)
            for variable in motor_variables
        }

        if movement_duration_s <= 0.0 or movement_sample_count == 0:
            analysis_status = "no_movement"
            for trajectory in TRAJECTORY_TYPES:
                bounds = trajectory_bounds[trajectory]
                trajectory_rows.append(
                    {
                        "trajectory_type": trajectory,
                        "trajectory_interval_count": len(bounds),
                        "trajectory_interval_duration_s": float(
                            np.sum(bounds[:, 1] - bounds[:, 0])
                        ),
                        "movement_supported_duration_s": 0.0,
                        "movement_supported_sample_count": 0,
                        "finite_progression_sample_count": 0,
                        "occupied_progression_bin_count": 0,
                        "graph_length_cm": graph_lengths[trajectory],
                        "trajectory_status": "no_movement",
                    }
                )
        else:
            filtered_position = _make_filtered_position(
                filtered_timestamps,
                filtered_primary,
            )
            supported_trajectory_count = 0
            for trajectory in TRAJECTORY_TYPES:
                bounds = trajectory_bounds[trajectory]
                trajectory_mask = motor.build_interval_mask(
                    filtered_timestamps,
                    bounds,
                )
                combined_mask = movement_mask & trajectory_mask
                support_count = int(np.sum(combined_mask))
                interval_count = len(bounds)
                interval_duration = float(np.sum(bounds[:, 1] - bounds[:, 0]))
                support_duration = _intersection_duration(bounds, movement_bounds)
                if interval_count == 0:
                    trajectory_status = "no_trials"
                    finite_progression_count = 0
                    progression_values = np.asarray([], dtype=float)
                elif support_count == 0:
                    trajectory_status = "no_movement_samples"
                    finite_progression_count = 0
                    progression_values = np.asarray([], dtype=float)
                else:
                    progression_path_values = _trajectory_progression_values(
                        filtered_position=filtered_position,
                        filtered_timestamps=filtered_timestamps,
                        trajectory_intervals=trajectory_intervals_by_type[trajectory],
                        trajectory_bounds=bounds,
                        graph_inputs=graph_inputs_by_configuration[trajectory],
                        trajectory_type=trajectory,
                        expected_graph_length_cm=graph_lengths[trajectory],
                    )
                    path_movement_mask = motor.build_interval_mask(
                        filtered_timestamps[trajectory_mask],
                        movement_bounds,
                    )
                    progression_values = progression_path_values[path_movement_mask]
                    if progression_values.size != support_count:
                        raise ValueError(
                            "Movement-supported progression and motor samples do not align."
                        )
                    finite_progression_count = int(
                        np.sum(np.isfinite(progression_values))
                    )
                    trajectory_status = (
                        "valid"
                        if finite_progression_count > 0
                        else "no_movement_samples"
                    )
                if trajectory_status == "valid":
                    supported_trajectory_count += 1
                for variable in motor_variables:
                    controlled_data[trajectory][variable] = (
                        progression_values,
                        np.asarray(computed_variables[variable][combined_mask], dtype=float),
                    )
                occupied_bin_count = 0
                if finite_progression_count:
                    bin_indices = np.searchsorted(
                        progression_edges,
                        progression_values[np.isfinite(progression_values)],
                        side="right",
                    ) - 1
                    bin_indices = np.clip(
                        bin_indices,
                        0,
                        len(progression_edges) - 2,
                    )
                    occupied_bin_count = int(np.unique(bin_indices).size)
                trajectory_rows.append(
                    {
                        "trajectory_type": trajectory,
                        "trajectory_interval_count": interval_count,
                        "trajectory_interval_duration_s": interval_duration,
                        "movement_supported_duration_s": support_duration,
                        "movement_supported_sample_count": support_count,
                        "finite_progression_sample_count": finite_progression_count,
                        "occupied_progression_bin_count": occupied_bin_count,
                        "graph_length_cm": graph_lengths[trajectory],
                        "trajectory_status": trajectory_status,
                    }
                )
            if supported_trajectory_count == 0:
                analysis_status = "no_trials"
            elif supported_trajectory_count < len(TRAJECTORY_TYPES):
                analysis_status = "partial_valid"
            else:
                analysis_status = "valid"

    distribution = motor.build_distribution_summary_table(
        [epoch_name],
        {epoch_name: movement_values},
        {epoch_name: movement_duration_s},
    ).loc[:, list(DISTRIBUTION_COLUMNS)]
    progression = motor.build_progression_summary_table(
        selected_epochs=[epoch_name],
        controlled_data_by_epoch={epoch_name: controlled_data},
        progression_bin_edges=progression_edges,
    ).loc[:, list(PROGRESSION_COLUMNS)]
    trajectory_qc = pd.DataFrame.from_records(trajectory_rows)
    common_qc = {
        "epoch_motor_behavior_id": result_id,
        "animal_name": animal,
        "date": session_date,
        "epoch": epoch_name,
    }
    for name, value in reversed(tuple(common_qc.items())):
        trajectory_qc.insert(0, name, value)
    trajectory_qc = trajectory_qc.loc[:, list(TRAJECTORY_QC_COLUMNS)]
    supported_count = int((trajectory_qc["trajectory_status"] == "valid").sum())

    metadata = {
        "epoch_motor_behavior_id": result_id,
        "animal_name": animal,
        "date": session_date,
        "epoch": epoch_name,
        "epoch_type": "run",
        "primary_position_source": position_inputs["primary_position_source"],
        "primary_position_role": position_inputs["primary_position_role"],
        "orientation_reference_position_source": position_inputs[
            "orientation_reference_position_source"
        ],
        "orientation_reference_position_role": position_inputs[
            "orientation_reference_position_role"
        ],
        "position_offset_samples": position_inputs["position_offset_samples"],
    }
    result = {
        "metadata": metadata,
        "parameters": parameter_snapshot,
        "movement_parameters": movement_snapshot,
        "distribution_summary": distribution,
        "progression_summary": progression,
        "trajectory_qc": trajectory_qc,
        "n_position_samples_input": int(
            position_inputs["n_position_samples_input"]
        ),
        "n_finite_position_samples": n_finite,
        "n_dropped_nonfinite_samples": n_dropped,
        "n_movement_samples": movement_sample_count,
        "movement_duration_s": movement_duration_s,
        "n_supported_trajectories": supported_count,
        "sampling_rate_hz": sampling_rate_hz,
        "median_sample_interval_s": median_dt_s,
        "maximum_sample_gap_s": maximum_gap_s,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance": (
            None
            if legacy_artifact_provenance is None
            else _json_safe(dict(legacy_artifact_provenance))
        ),
    }
    return validate_epoch_motor_behavior_result(result)


def _validate_exact_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    """Require exact canonical table columns."""
    if list(table.columns) != list(columns):
        raise ValueError(f"{name} columns do not match the canonical schema.")


def _nonnegative_integer_column(
    table: pd.DataFrame,
    column: str,
    *,
    name: str,
) -> np.ndarray:
    """Return one validated non-negative integer column."""
    values = table[column].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(
        values != np.floor(values)
    ):
        raise ValueError(f"{name} must contain non-negative integers.")
    return values.astype(np.int64)


def _validate_distribution_semantics(
    distribution: pd.DataFrame,
    *,
    motor_variables: Sequence[str],
    movement_sample_count: int,
    movement_duration_s: float,
) -> None:
    """Validate summary statistics and the fixed empty-distribution policy."""
    sample_counts = _nonnegative_integer_column(
        distribution,
        "sample_count",
        name="Distribution sample_count",
    )
    if not np.all(sample_counts == movement_sample_count):
        raise ValueError(
            "Distribution sample counts do not match n_movement_samples."
        )
    durations = distribution["movement_duration_s"].to_numpy(dtype=float)
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0) or not (
        np.allclose(
            durations,
            movement_duration_s,
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Distribution movement duration does not match the result."
        )

    summary_columns = (
        "mean",
        "median",
        "std",
        "p10",
        "p90",
        "circular_mean_deg",
        "resultant_length",
    )
    for _, row in distribution.iterrows():
        count = int(row["sample_count"])
        variable = str(row["variable"])
        values = np.asarray([row[name] for name in summary_columns], dtype=float)
        if count == 0:
            if not np.all(np.isnan(values)):
                raise ValueError(
                    "Empty distributions must use NaN for every summary statistic."
                )
            continue
        if variable == "head_direction_deg":
            required = np.asarray(
                [
                    row["median"],
                    row["p10"],
                    row["p90"],
                    row["circular_mean_deg"],
                    row["resultant_length"],
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(required)) or not (
                np.isnan(float(row["mean"])) and np.isnan(float(row["std"]))
            ):
                raise ValueError(
                    "Non-empty head-direction summaries must use the fixed circular "
                    "statistic policy."
                )
            circular_mean = float(row["circular_mean_deg"])
            resultant = float(row["resultant_length"])
            if not (-180.0 <= circular_mean <= 180.0) or not (
                0.0 <= resultant <= 1.0 + 1e-12
            ):
                raise ValueError("Head-direction circular statistics are out of range.")
        else:
            required = np.asarray(
                [
                    row["mean"],
                    row["median"],
                    row["std"],
                    row["p10"],
                    row["p90"],
                ],
                dtype=float,
            )
            if not np.all(np.isfinite(required)) or not (
                np.isnan(float(row["circular_mean_deg"]))
                and np.isnan(float(row["resultant_length"]))
            ):
                raise ValueError(
                    "Non-angular distributions must use finite linear statistics "
                    "and NaN circular statistics."
                )
            if float(row["std"]) < 0.0:
                raise ValueError("Distribution standard deviations cannot be negative.")
        if not (
            float(row["p10"])
            <= float(row["median"])
            <= float(row["p90"])
        ):
            raise ValueError("Distribution quantiles must be ordered.")


def _validate_progression_and_qc_semantics(
    progression: pd.DataFrame,
    trajectory_qc: pd.DataFrame,
    *,
    motor_variables: Sequence[str],
    progression_bin_size_cm: float,
) -> None:
    """Cross-validate trajectory QC, progression bins, and summary counts."""
    count_columns = (
        "trajectory_interval_count",
        "movement_supported_sample_count",
        "finite_progression_sample_count",
        "occupied_progression_bin_count",
    )
    qc_counts = {
        column: _nonnegative_integer_column(
            trajectory_qc,
            column,
            name=f"trajectory_qc {column}",
        )
        for column in count_columns
    }
    duration_columns = (
        "trajectory_interval_duration_s",
        "movement_supported_duration_s",
        "graph_length_cm",
    )
    qc_values = {
        column: trajectory_qc[column].to_numpy(dtype=float)
        for column in duration_columns
    }
    for column, values in qc_values.items():
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"trajectory_qc {column} must be finite and non-negative.")
    if np.any(qc_values["graph_length_cm"] <= 0.0) or not np.allclose(
        qc_values["graph_length_cm"],
        qc_values["graph_length_cm"][0],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("trajectory_qc graph lengths must be equal and positive.")
    interval_counts = qc_counts["trajectory_interval_count"]
    interval_durations = qc_values["trajectory_interval_duration_s"]
    if np.any((interval_counts == 0) != (interval_durations == 0.0)):
        raise ValueError(
            "Trajectory interval counts and positive durations must be consistent."
        )
    support_durations = qc_values["movement_supported_duration_s"]
    if np.any(support_durations > interval_durations + 1e-12):
        raise ValueError("Movement-supported duration exceeds trajectory duration.")
    support_counts = qc_counts["movement_supported_sample_count"]
    finite_counts = qc_counts["finite_progression_sample_count"]
    occupied_counts = qc_counts["occupied_progression_bin_count"]
    if np.any(finite_counts > support_counts) or np.any(
        occupied_counts > finite_counts
    ):
        raise ValueError("Trajectory progression counts exceed their sample support.")

    if not progression.empty:
        _nonnegative_integer_column(
            progression,
            "progression_bin_index",
            name="Progression bin indices",
        )
        sample_counts = _nonnegative_integer_column(
            progression,
            "sample_count",
            name="Progression sample_count",
        )
        if np.any(sample_counts == 0):
            raise ValueError("Progression rows must describe occupied bins.")
        numeric = progression.loc[
            :,
            [
                "progression_bin_start",
                "progression_bin_end",
                "progression_bin_center",
                "median",
                "q25",
                "q75",
            ],
        ].to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Progression rows must contain finite values.")
        starts = progression["progression_bin_start"].to_numpy(dtype=float)
        ends = progression["progression_bin_end"].to_numpy(dtype=float)
        centers = progression["progression_bin_center"].to_numpy(dtype=float)
        if np.any(starts < 0.0) or np.any(ends > 1.0) or np.any(ends <= starts):
            raise ValueError("Progression bins must be ordered within [0, 1].")
        if not np.allclose(
            centers,
            (starts + ends) / 2.0,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Progression bin centers must be exact midpoints.")
        q25 = progression["q25"].to_numpy(dtype=float)
        median = progression["median"].to_numpy(dtype=float)
        q75 = progression["q75"].to_numpy(dtype=float)
        if np.any(q25 > median) or np.any(median > q75):
            raise ValueError("Progression quantiles must be ordered.")

    for _, qc_row in trajectory_qc.iterrows():
        trajectory = str(qc_row["trajectory_type"])
        status = str(qc_row["trajectory_status"])
        interval_count = int(qc_row["trajectory_interval_count"])
        support_count = int(qc_row["movement_supported_sample_count"])
        finite_count = int(qc_row["finite_progression_sample_count"])
        occupied_count = int(qc_row["occupied_progression_bin_count"])
        path_progression = progression.loc[
            progression["trajectory_type"].astype(str) == trajectory
        ]
        if status == "valid":
            if not (
                interval_count > 0
                and support_count > 0
                and finite_count > 0
                and occupied_count > 0
            ):
                raise ValueError("Valid trajectory QC rows require positive support.")
        elif status == "no_trials":
            if not (
                interval_count == 0
                and support_count == 0
                and finite_count == 0
                and occupied_count == 0
            ):
                raise ValueError("no_trials trajectory QC is internally inconsistent.")
        elif status in {"no_valid_position", "no_movement"}:
            if not (
                float(qc_row["movement_supported_duration_s"]) == 0.0
                and support_count == 0
                and finite_count == 0
                and occupied_count == 0
            ):
                raise ValueError(f"{status} trajectory QC must have zero support.")
        elif status == "no_movement_samples":
            if interval_count == 0 or finite_count != 0 or occupied_count != 0:
                raise ValueError(
                    "no_movement_samples trajectory QC is internally inconsistent."
                )

        if status != "valid":
            if not path_progression.empty:
                raise ValueError(
                    "Only valid trajectories may have progression summary rows."
                )
            continue
        if path_progression.empty:
            raise ValueError("Valid trajectories require progression summary rows.")
        edges = _progression_bin_edges(
            graph_length_cm=float(qc_row["graph_length_cm"]),
            progression_bin_size_cm=progression_bin_size_cm,
        )
        indices = path_progression["progression_bin_index"].to_numpy(dtype=int)
        if np.any(indices >= len(edges) - 1):
            raise ValueError("Progression bin index exceeds the selected graph geometry.")
        if not np.allclose(
            path_progression["progression_bin_start"].to_numpy(dtype=float),
            edges[indices],
            rtol=1e-12,
            atol=1e-12,
        ) or not np.allclose(
            path_progression["progression_bin_end"].to_numpy(dtype=float),
            edges[indices + 1],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Progression rows do not match selected bin geometry.")
        occupied_bins = np.unique(indices)
        if occupied_bins.size != occupied_count:
            raise ValueError(
                "Progression occupied bins do not match trajectory_qc."
            )
        for bin_index in occupied_bins:
            rows = path_progression.loc[
                path_progression["progression_bin_index"].to_numpy(dtype=int)
                == bin_index
            ]
            if len(rows) != len(motor_variables) or set(
                rows["variable"].astype(str)
            ) != set(motor_variables):
                raise ValueError(
                    "Each occupied path bin must contain all motor variables."
                )
            if rows["sample_count"].astype(int).nunique() != 1:
                raise ValueError(
                    "Motor variables in one path bin must share sample support."
                )
        reference_rows = path_progression.loc[
            path_progression["variable"].astype(str) == str(motor_variables[0])
        ]
        if int(reference_rows["sample_count"].sum()) != finite_count:
            raise ValueError(
                "Progression sample counts do not match trajectory_qc."
            )


def validate_epoch_motor_behavior_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and shallow-copy one in-memory motor artifact result."""
    copied = dict(result)
    metadata = dict(copied["metadata"])
    parameters = dict(copied["parameters"])
    movement_parameters = validate_movement_parameter_snapshot(
        copied["movement_parameters"],
        movement_parameters_sha256=dict(copied["movement_parameters"])[
            "movement_parameters_sha256"
        ],
    )
    result_id = _uuid_string(
        metadata["epoch_motor_behavior_id"],
        name="epoch_motor_behavior_id",
    )
    metadata["epoch_motor_behavior_id"] = result_id
    if str(metadata.get("epoch_type")) != "run":
        raise ValueError("Epoch motor behavior metadata must identify a run epoch.")
    effective = validate_epoch_motor_behavior_parameters(
        progression_bin_size_cm=parameters["progression_bin_size_cm"],
    )
    expected_parameters = _parameter_snapshot(
        parameter_name=parameters["parameter_name"],
        parameters=effective,
        parameter_sha256=parameters["parameter_sha256"],
        output_rule_sha256=parameters["output_rule_sha256"],
    )
    for name, columns in (
        ("distribution_summary", DISTRIBUTION_COLUMNS),
        ("progression_summary", PROGRESSION_COLUMNS),
        ("trajectory_qc", TRAJECTORY_QC_COLUMNS),
    ):
        if not isinstance(copied[name], pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        _validate_exact_columns(copied[name], columns, name=name)

    distribution = copied["distribution_summary"]
    progression = copied["progression_summary"]
    trajectory_qc = copied["trajectory_qc"]
    motor_variables = tuple(_motor_module().MOTOR_VARIABLES)
    if len(distribution) != len(motor_variables):
        raise ValueError("Distribution summary must have one row per motor variable.")
    if set(distribution["variable"].astype(str)) != set(motor_variables) or (
        distribution["variable"].astype(str).duplicated().any()
    ):
        raise ValueError("Distribution motor-variable rows are incomplete or duplicated.")
    if not np.all(distribution["epoch"].astype(str) == str(metadata["epoch"])):
        raise ValueError("Distribution rows do not match the selected epoch.")
    if len(trajectory_qc) != len(TRAJECTORY_TYPES) or (
        trajectory_qc["trajectory_type"].astype(str).duplicated().any()
    ):
        raise ValueError("trajectory_qc must contain one row per trajectory.")
    if set(trajectory_qc["trajectory_type"].astype(str)) != set(TRAJECTORY_TYPES):
        raise ValueError("trajectory_qc trajectory rows are incomplete.")
    for field in ("epoch_motor_behavior_id", "animal_name", "date", "epoch"):
        if not np.all(trajectory_qc[field].astype(str) == str(metadata[field])):
            raise ValueError(f"trajectory_qc does not match metadata field {field!r}.")
    if not set(trajectory_qc["trajectory_status"].astype(str)).issubset(
        TRAJECTORY_STATUSES
    ):
        raise ValueError("trajectory_qc contains an unsupported status.")

    if not progression.empty:
        if not np.all(progression["epoch"].astype(str) == str(metadata["epoch"])):
            raise ValueError("Progression rows do not match the selected epoch.")
        if not set(progression["trajectory_type"].astype(str)).issubset(
            TRAJECTORY_TYPES
        ) or not set(progression["variable"].astype(str)).issubset(motor_variables):
            raise ValueError("Progression rows contain unsupported labels.")
        keys = [
            "trajectory_type",
            "variable",
            "progression_bin_index",
        ]
        if progression[keys].astype(str).duplicated().any():
            raise ValueError("Progression rows contain duplicate path-variable bins.")

    scalar_integer_names = (
        "n_position_samples_input",
        "n_finite_position_samples",
        "n_dropped_nonfinite_samples",
        "n_movement_samples",
        "n_supported_trajectories",
    )
    for name in scalar_integer_names:
        _integer(copied[name], name=name)
    if int(copied["n_finite_position_samples"]) + int(
        copied["n_dropped_nonfinite_samples"]
    ) != int(copied["n_position_samples_input"]):
        raise ValueError("Finite and dropped position counts do not match input count.")
    duration = _finite_float(
        copied["movement_duration_s"],
        name="movement_duration_s",
    )
    if duration < 0.0:
        raise ValueError("movement_duration_s must be non-negative.")
    n_movement = int(copied["n_movement_samples"])
    n_finite = int(copied["n_finite_position_samples"])
    if n_movement > n_finite:
        raise ValueError("Movement sample count exceeds finite position support.")
    _validate_distribution_semantics(
        distribution,
        motor_variables=motor_variables,
        movement_sample_count=n_movement,
        movement_duration_s=duration,
    )
    _validate_progression_and_qc_semantics(
        progression,
        trajectory_qc,
        motor_variables=motor_variables,
        progression_bin_size_cm=effective["progression_bin_size_cm"],
    )
    if np.any(
        trajectory_qc["movement_supported_sample_count"].to_numpy(dtype=int)
        > n_movement
    ):
        raise ValueError(
            "Trajectory movement support exceeds epoch-wide movement samples."
        )
    supported = int((trajectory_qc["trajectory_status"] == "valid").sum())
    if supported != int(copied["n_supported_trajectories"]):
        raise ValueError("Supported trajectory count does not match trajectory_qc.")

    status = str(copied["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError("Unsupported analysis_status.")
    if status == "valid" and supported != len(TRAJECTORY_TYPES):
        raise ValueError("valid results require all four supported trajectories.")
    if status == "partial_valid" and not (0 < supported < len(TRAJECTORY_TYPES)):
        raise ValueError("partial_valid requires one to three supported trajectories.")
    if status == "no_valid_position" and n_finite >= 2:
        raise ValueError("no_valid_position requires fewer than two finite samples.")
    if status == "no_valid_position" and (
        n_movement != 0
        or duration != 0.0
        or set(trajectory_qc["trajectory_status"].astype(str))
        != {"no_valid_position"}
    ):
        raise ValueError("no_valid_position result status is inconsistent with QC.")
    if status == "no_movement" and (n_movement != 0 or duration != 0.0):
        raise ValueError("no_movement requires zero sampled movement and duration.")
    if status == "no_movement" and set(
        trajectory_qc["trajectory_status"].astype(str)
    ) != {"no_movement"}:
        raise ValueError("no_movement result status is inconsistent with QC.")
    if status == "no_trials" and (
        n_finite < 2 or n_movement == 0 or duration <= 0.0 or supported != 0
    ):
        raise ValueError(
            "no_trials requires valid moving position but no supported trajectory."
        )
    artifact_origin = str(copied["artifact_origin"])
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("Unsupported artifact_origin.")
    provenance = copied.get("legacy_artifact_provenance")
    if artifact_origin == "computed" and provenance is not None:
        raise ValueError("Computed artifacts cannot have legacy provenance.")
    if artifact_origin == "registered_existing" and (
        not isinstance(provenance, Mapping) or not provenance
    ):
        raise ValueError(
            "Registered-existing artifacts require non-empty legacy provenance."
        )
    for name in (
        "sampling_rate_hz",
        "median_sample_interval_s",
        "maximum_sample_gap_s",
    ):
        value = float(copied[name])
        if n_finite >= 2 and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be positive for valid position data.")
        if n_finite < 2 and not np.isnan(value):
            raise ValueError(f"{name} must be NaN without valid position data.")
    copied["metadata"] = metadata
    copied["parameters"] = expected_parameters
    copied["movement_parameters"] = movement_parameters
    copied["artifact_origin"] = artifact_origin
    copied["legacy_artifact_provenance"] = (
        None if provenance is None else _json_safe(dict(provenance))
    )
    return copied


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields repeated across one artifact manifest."""
    metadata = dict(result["metadata"])
    parameters = dict(result["parameters"])
    movement_parameters = dict(result["movement_parameters"])
    provenance_json = json.dumps(
        _json_safe(result.get("legacy_artifact_provenance") or {}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **metadata,
        **parameters,
        **movement_parameters,
        "schema_version": SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_position_samples_input": result["n_position_samples_input"],
        "n_finite_position_samples": result["n_finite_position_samples"],
        "n_dropped_nonfinite_samples": result["n_dropped_nonfinite_samples"],
        "n_movement_samples": result["n_movement_samples"],
        "movement_duration_s": result["movement_duration_s"],
        "n_supported_trajectories": result["n_supported_trajectories"],
        "sampling_rate_hz": result["sampling_rate_hz"],
        "median_sample_interval_s": result["median_sample_interval_s"],
        "maximum_sample_gap_s": result["maximum_sample_gap_s"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": provenance_json,
    }


def write_epoch_motor_behavior_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write and reload one immutable motor artifact bundle."""
    result = validate_epoch_motor_behavior_result(result)
    destination = Path(path)
    result_id = result["metadata"]["epoch_motor_behavior_id"]
    if destination.name != result_id:
        raise ValueError("Artifact directory name must equal the result UUID.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite epoch motor artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.backup")
    temporary.mkdir()
    try:
        result["distribution_summary"].to_parquet(
            temporary / DISTRIBUTION_FILENAME,
            index=False,
        )
        result["progression_summary"].to_parquet(
            temporary / PROGRESSION_FILENAME,
            index=False,
        )
        result["trajectory_qc"].to_parquet(
            temporary / TRAJECTORY_QC_FILENAME,
            index=False,
        )
        common = _manifest_common(result)
        specs = (
            ("distribution_summary", DISTRIBUTION_FILENAME),
            ("progression_summary", PROGRESSION_FILENAME),
            ("trajectory_qc", TRAJECTORY_QC_FILENAME),
        )
        rows = []
        for artifact_key, filename in specs:
            artifact_path = temporary / filename
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "relative_path": filename,
                    "artifact_kind": "parquet",
                    "file_size_bytes": artifact_path.stat().st_size,
                    "sha256": _file_sha256(artifact_path),
                    **common,
                }
            )
        pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_parquet(
            temporary / MANIFEST_FILENAME,
            index=False,
        )
        load_epoch_motor_behavior_artifact(
            temporary,
            _allow_temporary_name=True,
        )
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    return _paths_for_directory(destination)


def load_epoch_motor_behavior_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one canonical motor artifact bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Epoch motor manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    _validate_exact_columns(manifest, MANIFEST_COLUMNS, name="manifest")
    if manifest.empty or manifest["artifact_key"].duplicated().any():
        raise ValueError("Manifest artifact keys must be unique and non-empty.")
    expected = {
        "distribution_summary": DISTRIBUTION_FILENAME,
        "progression_summary": PROGRESSION_FILENAME,
        "trajectory_qc": TRAJECTORY_QC_FILENAME,
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("Manifest does not contain the canonical artifact set.")
    for _, row in manifest.iterrows():
        key = str(row["artifact_key"])
        filename = expected[key]
        if str(row["relative_path"]) != filename or str(
            row["artifact_kind"]
        ) != "parquet":
            raise ValueError("Manifest artifact names or kinds are inconsistent.")
        if Path(str(row["relative_path"])).name != str(row["relative_path"]):
            raise ValueError("Manifest paths must be direct child filenames.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(f"Manifest checksum mismatch for {artifact_path}.")

    first = manifest.iloc[0]
    if str(first["schema_version"]) != SCHEMA_VERSION or str(
        first["bundle_schema_version"]
    ) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported epoch motor artifact schema version.")
    metadata_names = (
        "epoch_motor_behavior_id",
        "animal_name",
        "date",
        "epoch",
        "epoch_type",
        "primary_position_source",
        "primary_position_role",
        "orientation_reference_position_source",
        "orientation_reference_position_role",
    )
    metadata = {name: str(first[name]) for name in metadata_names}
    metadata["position_offset_samples"] = int(first["position_offset_samples"])
    if not _allow_temporary_name and directory.name != metadata[
        "epoch_motor_behavior_id"
    ]:
        raise ValueError("Artifact directory name does not match its result UUID.")
    parameters = {
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        "progression_bin_size_cm": float(first["progression_bin_size_cm"]),
    }
    movement_parameters = {
        "movement_param_name": str(first["movement_param_name"]),
        "movement_parameters_sha256": str(first["movement_parameters_sha256"]),
        "speed_threshold_cm_s": float(first["speed_threshold_cm_s"]),
        "speed_smoothing_sigma_s": float(first["speed_smoothing_sigma_s"]),
    }
    provenance_payload = json.loads(str(first["legacy_artifact_provenance_json"]))
    common = {
        **metadata,
        **parameters,
        **movement_parameters,
        "schema_version": SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_position_samples_input": int(first["n_position_samples_input"]),
        "n_finite_position_samples": int(first["n_finite_position_samples"]),
        "n_dropped_nonfinite_samples": int(
            first["n_dropped_nonfinite_samples"]
        ),
        "n_movement_samples": int(first["n_movement_samples"]),
        "movement_duration_s": float(first["movement_duration_s"]),
        "n_supported_trajectories": int(first["n_supported_trajectories"]),
        "sampling_rate_hz": float(first["sampling_rate_hz"]),
        "median_sample_interval_s": float(first["median_sample_interval_s"]),
        "maximum_sample_gap_s": float(first["maximum_sample_gap_s"]),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance_json": str(
            first["legacy_artifact_provenance_json"]
        ),
    }
    for name, value in common.items():
        if not np.all(manifest[name].astype(str) == str(value)):
            raise ValueError(f"Manifest has inconsistent {name!r} values.")
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "movement_parameters": movement_parameters,
        "distribution_summary": pd.read_parquet(
            directory / DISTRIBUTION_FILENAME
        ),
        "progression_summary": pd.read_parquet(directory / PROGRESSION_FILENAME),
        "trajectory_qc": pd.read_parquet(directory / TRAJECTORY_QC_FILENAME),
        "n_position_samples_input": int(first["n_position_samples_input"]),
        "n_finite_position_samples": int(first["n_finite_position_samples"]),
        "n_dropped_nonfinite_samples": int(
            first["n_dropped_nonfinite_samples"]
        ),
        "n_movement_samples": int(first["n_movement_samples"]),
        "movement_duration_s": float(first["movement_duration_s"]),
        "n_supported_trajectories": int(first["n_supported_trajectories"]),
        "sampling_rate_hz": float(first["sampling_rate_hz"]),
        "median_sample_interval_s": float(first["median_sample_interval_s"]),
        "maximum_sample_gap_s": float(first["maximum_sample_gap_s"]),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance": provenance_payload or None,
        "manifest": manifest,
    }
    return validate_epoch_motor_behavior_result(result)


def _normalize_motor_table_for_comparison(
    table: pd.DataFrame,
    *,
    kind: str,
) -> pd.DataFrame:
    """Return one deterministically sorted legacy/canonical table."""
    normalized = table.copy()
    for column in ("epoch", "trajectory_type", "variable"):
        if column in normalized:
            normalized[column] = normalized[column].astype(str)
    variable_order = {
        value: index for index, value in enumerate(_motor_module().MOTOR_VARIABLES)
    }
    normalized["_variable_order"] = normalized["variable"].map(variable_order)
    if kind == "distribution":
        sort_columns = ["_variable_order"]
    elif kind == "progression":
        trajectory_order = {
            value: index for index, value in enumerate(TRAJECTORY_TYPES)
        }
        normalized["_trajectory_order"] = normalized["trajectory_type"].map(
            trajectory_order
        )
        sort_columns = [
            "_variable_order",
            "_trajectory_order",
            "progression_bin_index",
        ]
    else:
        raise ValueError(f"Unknown motor table kind {kind!r}.")
    normalized = normalized.sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )
    return normalized.drop(
        columns=[column for column in normalized if column.startswith("_")]
    )


def _assert_recomputed_table_matches(
    *,
    source: pd.DataFrame,
    recomputed: pd.DataFrame,
    kind: str,
) -> None:
    """Require one legacy epoch subset to match NWB recomputation."""
    source_normalized = _normalize_motor_table_for_comparison(source, kind=kind)
    recomputed_normalized = _normalize_motor_table_for_comparison(
        recomputed,
        kind=kind,
    )
    try:
        pd.testing.assert_frame_equal(
            source_normalized,
            recomputed_normalized,
            check_dtype=False,
            check_categorical=False,
            check_exact=False,
            rtol=1e-7,
            atol=1e-9,
        )
    except AssertionError as exc:
        raise ValueError(
            f"Legacy {kind} rows do not match exact NWB recomputation."
        ) from exc


def _read_legacy_run_log(
    path: Path | None,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    position_offset_samples: int,
    parameters: Mapping[str, float],
    movement_parameters: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate an optional legacy script run log and return provenance."""
    if path is None:
        return None
    log_path = Path(path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Legacy motor run log not found: {log_path}")
    with log_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("script") != "v1ca1.motor.compare_epoch_motor_behavior":
        raise ValueError("Legacy run log has the wrong script name.")
    logged = dict(payload.get("parameters", {}))
    if str(logged.get("animal_name")) != str(animal_name) or str(
        logged.get("date")
    ) != str(date):
        raise ValueError("Legacy run log session does not match registration.")
    if str(epoch) not in [str(value) for value in logged.get("epochs", ())]:
        raise ValueError("Legacy run log does not include the selected epoch.")
    comparisons = {
        "position_offset": float(position_offset_samples),
        "speed_threshold_cm_s": movement_parameters["speed_threshold_cm_s"],
        "progression_bin_size_cm": parameters["progression_bin_size_cm"],
    }
    for name, expected in comparisons.items():
        if name not in logged or not np.isclose(
            float(logged[name]),
            float(expected),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Legacy run log parameter {name!r} does not match.")
    return {
        "source_run_log_path": str(log_path.resolve(strict=True)),
        "source_run_log_sha256": _file_sha256(log_path),
        "source_v1ca1_git_commit": payload.get("git_commit"),
        "source_git_dirty": payload.get("git_dirty"),
        "source_timestamp_utc": payload.get("timestamp_utc"),
    }


def register_existing_epoch_motor_behavior_artifact(
    *,
    source_distribution_path: Path,
    source_progression_path: Path,
    destination_path: Path,
    animal_name: str,
    date: str,
    epoch: str,
    epoch_motor_behavior_id: Any,
    primary_position: Any,
    orientation_reference_position: Any,
    primary_position_row: Mapping[str, Any],
    orientation_reference_position_row: Mapping[str, Any],
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    source_run_log_path: Path | None = None,
    epoch_type: str = "run",
    parameter_name: str = "manuscript_4cm",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    progression_bin_size_cm: float = DEFAULT_PROGRESSION_BIN_SIZE_CM,
    movement_parameters: Mapping[str, Any] | None = None,
    movement_parameters_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Strictly verify and register one epoch from legacy session Parquets."""
    distribution_path = Path(source_distribution_path)
    progression_path = Path(source_progression_path)
    for name, source in (
        ("source_distribution_path", distribution_path),
        ("source_progression_path", progression_path),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"{name} does not exist: {source}")
    parameters = validate_epoch_motor_behavior_parameters(
        progression_bin_size_cm=progression_bin_size_cm,
    )
    movement_snapshot = validate_movement_parameter_snapshot(
        movement_parameters,
        movement_parameters_sha256=movement_parameters_sha256,
    )
    position_offset_samples = _integer(
        primary_position_row.get("analysis_start_offset_samples"),
        name="analysis_start_offset_samples",
    )
    log_provenance = _read_legacy_run_log(
        source_run_log_path,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        position_offset_samples=position_offset_samples,
        parameters=parameters,
        movement_parameters=movement_snapshot,
    )
    provenance = {
        "source_distribution_path": str(distribution_path.resolve(strict=True)),
        "source_distribution_sha256": _file_sha256(distribution_path),
        "source_progression_path": str(progression_path.resolve(strict=True)),
        "source_progression_sha256": _file_sha256(progression_path),
        "source_epoch": str(epoch),
        "verification": "exact_epoch_rows_recomputed_from_selected_nwb_inputs",
        **(log_provenance or {}),
    }
    recomputed = compute_selected_epoch_motor_behavior(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        epoch_motor_behavior_id=epoch_motor_behavior_id,
        primary_position=primary_position,
        orientation_reference_position=orientation_reference_position,
        primary_position_row=primary_position_row,
        orientation_reference_position_row=orientation_reference_position_row,
        trajectory_intervals_by_type=trajectory_intervals_by_type,
        graph_inputs_by_configuration=graph_inputs_by_configuration,
        epoch_type=epoch_type,
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        progression_bin_size_cm=progression_bin_size_cm,
        movement_parameters=movement_snapshot,
        movement_parameters_sha256=movement_snapshot[
            "movement_parameters_sha256"
        ],
        artifact_origin="registered_existing",
        legacy_artifact_provenance=provenance,
    )

    source_distribution = pd.read_parquet(distribution_path)
    source_progression = pd.read_parquet(progression_path)
    _validate_exact_columns(
        source_distribution,
        DISTRIBUTION_COLUMNS,
        name="legacy distribution summary",
    )
    _validate_exact_columns(
        source_progression,
        PROGRESSION_COLUMNS,
        name="legacy progression summary",
    )
    distribution_epoch = source_distribution.loc[
        source_distribution["epoch"].astype(str) == str(epoch)
    ].copy()
    progression_epoch = source_progression.loc[
        source_progression["epoch"].astype(str) == str(epoch)
    ].copy()
    if len(distribution_epoch) != len(_motor_module().MOTOR_VARIABLES):
        raise ValueError(
            "Legacy distribution file must contain exactly six selected-epoch rows."
        )
    _assert_recomputed_table_matches(
        source=distribution_epoch,
        recomputed=recomputed["distribution_summary"],
        kind="distribution",
    )
    _assert_recomputed_table_matches(
        source=progression_epoch,
        recomputed=recomputed["progression_summary"],
        kind="progression",
    )
    paths = write_epoch_motor_behavior_artifact(
        recomputed,
        destination_path,
        overwrite=overwrite,
    )
    return {
        **recomputed,
        **paths,
        "_created_artifact_paths": [
            str(paths["artifact_manifest_path"]),
            str(paths["distribution_summary_path"]),
            str(paths["progression_summary_path"]),
            str(paths["trajectory_qc_path"]),
        ],
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_PROGRESSION_BIN_SIZE_CM",
    "DEFAULT_SPEED_SMOOTHING_SIGMA_S",
    "DEFAULT_SPEED_THRESHOLD_CM_S",
    "DISTRIBUTION_COLUMNS",
    "DISTRIBUTION_FILENAME",
    "MANIFEST_COLUMNS",
    "MANIFEST_FILENAME",
    "MANUSCRIPT_MOVEMENT_PARAMETERS",
    "MANUSCRIPT_PARAMETERS",
    "OUTPUT_RULE",
    "PROGRESSION_COLUMNS",
    "PROGRESSION_FILENAME",
    "SCHEMA_VERSION",
    "TRAJECTORY_QC_COLUMNS",
    "TRAJECTORY_QC_FILENAME",
    "compute_selected_epoch_motor_behavior",
    "empty_trajectory_qc_table",
    "get_epoch_motor_behavior_artifact_paths",
    "load_epoch_motor_behavior_artifact",
    "register_existing_epoch_motor_behavior_artifact",
    "validate_epoch_motor_behavior_parameters",
    "validate_epoch_motor_behavior_result",
    "validate_movement_parameter_snapshot",
    "validate_position_inputs",
    "write_epoch_motor_behavior_artifact",
]
