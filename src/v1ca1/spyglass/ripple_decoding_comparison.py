"""Database-free CA1/V1 ripple decoding comparison artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "ripple_decoding_comparison"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
RIPPLE_QC_FILENAME = "ripple_qc.parquet"
RIPPLE_METRICS_FILENAME = "ripple_metrics.parquet"
EPOCH_SUMMARY_FILENAME = "epoch_summary.parquet"
RESULT_FILENAME = "ripple_decoding_comparison.nc"
CA1_DECODED_FILENAME = "ca1_decoded.npz"
V1_DECODED_FILENAME = "v1_decoded.npz"
BUNDLE_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"

REPRESENTATIONS = ("path_specific_place", "dpp")
LEGACY_REPRESENTATION = {
    "path_specific_place": "place",
    "dpp": "task_progression",
}
LEGACY_TO_REPRESENTATION = {
    value: key for key, value in LEGACY_REPRESENTATION.items()
}
SCORING_SCHEMES = ("trajectory", "turn_group", "arm_identity")
SOURCE_REGION = "ca1"
TARGET_REGION = "v1"

DEFAULT_DECODE_BIN_SIZE_S = 0.002
DEFAULT_SPATIAL_BIN_SIZE_CM = 4.0
DEFAULT_TUNING_SMOOTHING_SIGMA_BINS = 0.0
DEFAULT_CA1_MIN_MOVEMENT_RATE_HZ = 0.0
DEFAULT_V1_MIN_MOVEMENT_RATE_HZ = 0.5
DEFAULT_N_SHUFFLES = 100
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD = 2.0
DEFAULT_REQUIRE_SPEED_GATED = True
DEFAULT_EXPECTED_MOVEMENT_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_EXPECTED_SPEED_SIGMA_S = 0.1

TURN_GROUP_BY_TRAJECTORY = {
    "center_to_left": "left",
    "right_to_center": "left",
    "center_to_right": "right",
    "left_to_center": "right",
}
PHYSICAL_ARM_BY_TRAJECTORY = {
    "center_to_left": "left",
    "left_to_center": "left",
    "center_to_right": "right",
    "right_to_center": "right",
}
DPP_OFFSET_BY_TRAJECTORY = {
    trajectory: 0.0 if turn == "left" else 1.0
    for trajectory, turn in TURN_GROUP_BY_TRAJECTORY.items()
}
TURN_GROUP_LABELS = ("left", "right")
ARM_IDENTITY_LABELS = ("other", "left", "right")

OUTPUT_RULE = {
    "version": 1,
    "comparison_direction": "ca1_vs_v1_decoded_state",
    "result_granularity": (
        "session_x_train_epoch_x_decode_epoch_x_representation"
    ),
    "representations": REPRESENTATIONS,
    "path_specific_place_coordinate": (
        "four_concatenated_trajectory_specific_center_to_arm_blocks_cm"
    ),
    "dpp_coordinate": (
        "two_same_turn_blocks_left_then_right_natural_direction_progression"
    ),
    "trajectory_order": tuple(TRAJECTORY_TYPES),
    "turn_group_by_trajectory": TURN_GROUP_BY_TRAJECTORY,
    "physical_arm_by_trajectory": PHYSICAL_ARM_BY_TRAJECTORY,
    "arm_boundary_policy": (
        "graph_derived_midpoint_of_penultimate_center_to_arm_edge"
    ),
    "categorical_scoring_schemes": SCORING_SCHEMES,
    "path_specific_place_schemes": SCORING_SCHEMES,
    "dpp_schemes": ("turn_group",),
    "continuous_scoring": False,
    "unit_filter_policy": (
        "inclusive_train_epoch_movement_firing_rate_threshold_by_region"
    ),
    "training_policy": "movement_supported_train_epoch_tuning_curves",
    "decoding_policy": (
        "independent_ca1_and_v1_bayesian_decoders_per_exact_detected_ripple"
    ),
    "alignment_policy": "common_ripple_source_index_and_rounded_bin_time",
    "shuffle_policy": (
        "deranged_v1_ripple_blocks_within_equal_bin_count_groups"
    ),
    "ripple_input_policy": (
        "one_decode_epoch_detector_zscore_threshold_2_speed_gated_exact_intervals"
    ),
    "graph_policy": "four_directional_path_wtrack_graphs_in_centimeters",
    "terminal_policy": "explicit_expected_terminal_and_partial_artifacts",
    "unexpected_error_policy": "raise",
    "legacy_registration_policy": (
        "imported_sorting_identity_resolution_exact_nwb_redecode_rescore_and_"
        "complete_five_file_comparison_under_corrected_graph_scoring"
    ),
    "legacy_arm_bug_policy": "reject_turn_group_labeled_inbound_arm_outputs",
    "time_unit": "s",
    "time_reference": "augmented_nwb_ephys_timestamps",
}

IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (
    "region",
    *IDENTITY_COLUMNS,
    "input_unit_index",
    "movement_firing_rate_hz",
    "minimum_movement_firing_rate_hz",
    "passes_movement_firing_rate",
    "included_in_decoder",
    "unit_qc_status",
)
UNIT_QC_STATUSES = (
    "excluded_movement_firing_rate",
    "not_computed",
    "included",
)
RIPPLE_QC_COLUMNS = (
    "ripple_decoding_comparison_id",
    "animal_name",
    "date",
    "train_epoch",
    "decode_epoch",
    "representation",
    "ripple_source_index",
    "ripple_start_time_s",
    "ripple_end_time_s",
    "ca1_decoding_status",
    "ca1_decoding_reason",
    "ca1_n_bins",
    "v1_decoding_status",
    "v1_decoding_reason",
    "v1_n_bins",
    "alignment_status",
    "alignment_reason",
    "n_aligned_bins",
)
RIPPLE_METRIC_BASE_COLUMNS = (
    "ripple_decoding_comparison_id",
    "animal_name",
    "date",
    "train_epoch",
    "decode_epoch",
    "representation",
    "ripple_id",
    "ripple_source_index",
    "ripple_start_time_s",
    "ripple_end_time_s",
    "n_bins",
)
RIPPLE_METRIC_COLUMNS = RIPPLE_METRIC_BASE_COLUMNS + tuple(
    column
    for scheme in SCORING_SCHEMES
    for column in (
        f"{scheme}_scheme_requested",
        f"{scheme}_scheme_applicable",
        f"{scheme}_match_rate",
        f"{scheme}_n_matching_bins",
        f"{scheme}_n_valid_labeled_bins",
    )
)
EPOCH_SUMMARY_BASE_COLUMNS = (
    "ripple_decoding_comparison_id",
    "animal_name",
    "date",
    "train_epoch",
    "decode_epoch",
    "representation",
    "n_ripple_events_input",
    "n_ripples",
    "n_ripple_bins",
    "n_effective_shuffles",
)
EPOCH_SUMMARY_COLUMNS = EPOCH_SUMMARY_BASE_COLUMNS + tuple(
    column
    for scheme in SCORING_SCHEMES
    for column in (
        f"{scheme}_scheme_requested",
        f"{scheme}_scheme_applicable",
        f"{scheme}_scheme_reason",
        f"{scheme}_n_valid_ripples",
        f"{scheme}_match_rate",
        f"{scheme}_match_rate_shuffle_mean",
        f"{scheme}_match_rate_shuffle_sd",
        f"{scheme}_match_rate_p_value",
    )
) + ("analysis_status",)

ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_ripples",
    "no_ca1_units",
    "no_v1_units",
    "no_eligible_ca1_units",
    "no_eligible_v1_units",
    "no_train_movement",
    "no_train_trajectory_samples",
    "no_common_decoded_bins",
    "no_valid_metrics",
)
FITTED_STATUSES = ("valid", "partial_valid", "no_valid_metrics")

MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "ripple_decoding_comparison_id",
    "animal_name",
    "date",
    "train_epoch",
    "decode_epoch",
    "representation",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "graph_policy_sha256",
    "upstream_provenance_json",
    "selected_ripple_intervals_sha256",
    "n_ripple_events_input",
    "n_ripples",
    "n_ripple_bins",
    "n_ca1_units",
    "n_v1_units",
    "n_ca1_units_in_decoder",
    "n_v1_units_in_decoder",
    "selected_units_sha256",
    "ripple_qc_sha256",
    "ripple_metrics_sha256",
    "epoch_summary_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "bundle_schema_version",
)


def _scientific_module() -> Any:
    """Import the existing ripple-decoding implementation lazily."""
    from v1ca1.ripple import ripple_decoding_comparison

    return ripple_decoding_comparison


def _provenance_sha256(value: Any) -> str:
    """Return the shared deterministic provenance digest."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(value)


OUTPUT_RULE_SHA256 = _provenance_sha256(OUTPUT_RULE)


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


def _file_sha256(path: Path) -> str:
    """Return one streaming file digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_sha256(table: pd.DataFrame) -> str:
    """Return one deterministic canonical table digest."""
    values = pd.util.hash_pandas_object(table, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _database_bool(value: Any, *, name: str) -> bool:
    """Normalize a bool or database integer 0/1 without truthy coercion."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(int(value))
    raise TypeError(f"{name} must be a bool or database integer 0/1.")


def _fixed_float(value: Any, expected: float, *, name: str) -> float:
    """Require one finite numeric scalar to equal a fixed value."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric scalar.")
    output = float(value)
    if not np.isfinite(output) or not np.isclose(
        output, float(expected), rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"{name} must equal the fixed value {expected!r}.")
    return output


def _fixed_integer(value: Any, expected: int, *, name: str) -> int:
    """Require one integer scalar to equal a fixed value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    if int(value) != int(expected):
        raise ValueError(f"{name} must equal the fixed value {expected!r}.")
    return int(value)


def validate_ripple_decoding_comparison_parameters(
    *,
    decode_bin_size_s: float = DEFAULT_DECODE_BIN_SIZE_S,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    tuning_smoothing_sigma_bins: float = DEFAULT_TUNING_SMOOTHING_SIGMA_BINS,
    ca1_min_movement_rate_hz: float = DEFAULT_CA1_MIN_MOVEMENT_RATE_HZ,
    v1_min_movement_rate_hz: float = DEFAULT_V1_MIN_MOVEMENT_RATE_HZ,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    expected_detector_zscore_threshold: float = (
        DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD
    ),
    require_speed_gated: bool = DEFAULT_REQUIRE_SPEED_GATED,
) -> dict[str, Any]:
    """Return the fixed manuscript ripple-decoding parameters."""
    parameters = {
        "decode_bin_size_s": _fixed_float(
            decode_bin_size_s, DEFAULT_DECODE_BIN_SIZE_S, name="decode_bin_size_s"
        ),
        "spatial_bin_size_cm": _fixed_float(
            spatial_bin_size_cm,
            DEFAULT_SPATIAL_BIN_SIZE_CM,
            name="spatial_bin_size_cm",
        ),
        "tuning_smoothing_sigma_bins": _fixed_float(
            tuning_smoothing_sigma_bins,
            DEFAULT_TUNING_SMOOTHING_SIGMA_BINS,
            name="tuning_smoothing_sigma_bins",
        ),
        "ca1_min_movement_rate_hz": _fixed_float(
            ca1_min_movement_rate_hz,
            DEFAULT_CA1_MIN_MOVEMENT_RATE_HZ,
            name="ca1_min_movement_rate_hz",
        ),
        "v1_min_movement_rate_hz": _fixed_float(
            v1_min_movement_rate_hz,
            DEFAULT_V1_MIN_MOVEMENT_RATE_HZ,
            name="v1_min_movement_rate_hz",
        ),
        "n_shuffles": _fixed_integer(
            n_shuffles, DEFAULT_N_SHUFFLES, name="n_shuffles"
        ),
        "shuffle_seed": _fixed_integer(
            shuffle_seed, DEFAULT_SHUFFLE_SEED, name="shuffle_seed"
        ),
        "expected_detector_zscore_threshold": _fixed_float(
            expected_detector_zscore_threshold,
            DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD,
            name="expected_detector_zscore_threshold",
        ),
    }
    if not _database_bool(require_speed_gated, name="require_speed_gated"):
        raise ValueError("RippleDecodingComparison requires speed-gated ripples.")
    parameters["require_speed_gated"] = True
    return parameters


def _effective_parameters(
    *,
    parameter_name: str,
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
    **values: Any,
) -> dict[str, Any]:
    """Validate parameters and their immutable hashes."""
    parameters = validate_ripple_decoding_comparison_parameters(**values)
    name = _path_component(parameter_name, name="parameter_name")
    expected = _provenance_sha256(
        {"ripple_decoding_comparison_param_name": name, **parameters}
    )
    if parameter_sha256 is None:
        parameter_sha256 = expected
    if str(parameter_sha256) != expected:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    if output_rule_sha256 is None:
        output_rule_sha256 = OUTPUT_RULE_SHA256
    if str(output_rule_sha256) != OUTPUT_RULE_SHA256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": name,
        "parameter_sha256": str(parameter_sha256),
        "output_rule_sha256": str(output_rule_sha256),
        **parameters,
    }


def _metadata(
    *,
    ripple_decoding_comparison_id: Any,
    animal_name: str,
    date: str,
    train_epoch: str,
    decode_epoch: str,
    representation: str,
) -> dict[str, str]:
    """Return canonical immutable result metadata."""
    representation = str(representation)
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS!r}.")
    return {
        "ripple_decoding_comparison_id": _uuid_string(
            ripple_decoding_comparison_id,
            name="ripple_decoding_comparison_id",
        ),
        "animal_name": _path_component(animal_name, name="animal_name"),
        "date": _path_component(date, name="date"),
        "train_epoch": _path_component(train_epoch, name="train_epoch"),
        "decode_epoch": _path_component(decode_epoch, name="decode_epoch"),
        "representation": representation,
    }


def get_ripple_decoding_comparison_artifact_paths(
    *,
    animal_name: str,
    date: str,
    train_epoch: str,
    decode_epoch: str,
    representation: str,
    ripple_decoding_comparison_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return the session-first UUID bundle paths for one comparison."""
    metadata = _metadata(
        ripple_decoding_comparison_id=ripple_decoding_comparison_id,
        animal_name=animal_name,
        date=date,
        train_epoch=train_epoch,
        decode_epoch=decode_epoch,
        representation=representation,
    )
    pair = f"{metadata['train_epoch']}_train_to_{metadata['decode_epoch']}_decode"
    directory = (
        Path(artifact_root)
        / metadata["animal_name"]
        / metadata["date"]
        / ARTIFACT_DIRNAME
        / pair
        / metadata["representation"]
        / metadata["ripple_decoding_comparison_id"]
    )
    return {
        "artifact_dir": directory,
        "artifact_manifest_path": directory / MANIFEST_FILENAME,
        "selected_units_path": directory / SELECTED_UNITS_FILENAME,
        "ripple_qc_path": directory / RIPPLE_QC_FILENAME,
        "ripple_metrics_path": directory / RIPPLE_METRICS_FILENAME,
        "epoch_summary_path": directory / EPOCH_SUMMARY_FILENAME,
        "result_path": directory / RESULT_FILENAME,
        "ca1_decoded_path": directory / CA1_DECODED_FILENAME,
        "v1_decoded_path": directory / V1_DECODED_FILENAME,
    }


def _canonical_provenance(
    upstream_provenance: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Validate direct ripple and movement provenance and canonicalize JSON."""
    if not isinstance(upstream_provenance, Mapping) or not upstream_provenance:
        raise ValueError("upstream_provenance must be a non-empty mapping.")
    raw = dict(upstream_provenance)
    missing = [
        name
        for name in (
            "detector_zscore_threshold",
            "speed_gated",
            "movement_speed_threshold_cm_s",
            "movement_speed_sigma_s",
        )
        if name not in raw
    ]
    if missing:
        raise ValueError(f"upstream_provenance is missing {missing!r}.")
    detector = _fixed_float(
        raw["detector_zscore_threshold"],
        parameters["expected_detector_zscore_threshold"],
        name="upstream detector_zscore_threshold",
    )
    movement_threshold = _fixed_float(
        raw["movement_speed_threshold_cm_s"],
        DEFAULT_EXPECTED_MOVEMENT_SPEED_THRESHOLD_CM_S,
        name="upstream movement_speed_threshold_cm_s",
    )
    speed_sigma = _fixed_float(
        raw["movement_speed_sigma_s"],
        DEFAULT_EXPECTED_SPEED_SIGMA_S,
        name="upstream movement_speed_sigma_s",
    )
    speed_gated = _database_bool(raw["speed_gated"], name="upstream speed_gated")
    if not speed_gated:
        raise ValueError("Selected Ripples provenance must have speed_gated=True.")
    native = {
        **raw,
        "detector_zscore_threshold": detector,
        "speed_gated": True,
        "movement_speed_threshold_cm_s": movement_threshold,
        "movement_speed_sigma_s": speed_sigma,
    }
    try:
        encoded = json.dumps(native, sort_keys=True, separators=(",", ":"))
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError("upstream_provenance must be JSON serializable.") from exc
    return normalized, encoded


def _normalize_ripple_table(ripple_table: Any, *, epoch: str) -> pd.DataFrame:
    """Return exact finite non-overlapping ripple bounds for one decode epoch."""
    if isinstance(ripple_table, pd.DataFrame):
        table = ripple_table.copy()
    elif callable(getattr(ripple_table, "as_dataframe", None)):
        table = ripple_table.as_dataframe().copy()
    elif callable(getattr(ripple_table, "to_dataframe", None)):
        table = ripple_table.to_dataframe().copy()
    else:
        table = pd.DataFrame(ripple_table)
    rename = {}
    if "start" in table and "start_time" not in table:
        rename["start"] = "start_time"
    if "stop_time" in table and "end_time" not in table:
        rename["stop_time"] = "end_time"
    if "end" in table and "end_time" not in table:
        rename["end"] = "end_time"
    table = table.rename(columns=rename)
    if "epoch" in table:
        table = table.loc[table["epoch"].astype(str) == str(epoch)]
    missing = [name for name in ("start_time", "end_time") if name not in table]
    if missing:
        raise ValueError(f"ripple_table is missing required columns {missing!r}.")
    output = table.loc[:, ["start_time", "end_time"]].copy().reset_index(drop=True)
    for name in ("start_time", "end_time"):
        output[name] = pd.to_numeric(output[name], errors="raise")
    bounds = output.to_numpy(dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Ripple bounds must be finite seconds values.")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Every ripple must have start_time < end_time.")
    output = output.sort_values(["start_time", "end_time"], kind="stable").reset_index(
        drop=True
    )
    starts = output["start_time"].to_numpy(dtype=float)
    ends = output["end_time"].to_numpy(dtype=float)
    if len(output) > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError("Exact selected ripple intervals must not overlap.")
    return output


def _ripple_hash(table: pd.DataFrame) -> str:
    """Return a deterministic digest of exact ripple bounds."""
    return _provenance_sha256(
        {
            "start_time_s": table["start_time"].to_numpy(dtype=float).tolist(),
            "end_time_s": table["end_time"].to_numpy(dtype=float).tolist(),
        }
    )


def prepare_ripple_decoding_comparison_event_selection(
    *, decode_epoch: str, ripple_table: Any
) -> dict[str, Any]:
    """Return canonical exact decode-epoch ripples and their frozen hash."""
    table = _normalize_ripple_table(
        ripple_table, epoch=_path_component(decode_epoch, name="decode_epoch")
    )
    return {
        "selected_ripple_table": table,
        "n_ripple_events_input": int(len(table)),
        "ripple_duration_s": float(
            np.sum(
                table["end_time"].to_numpy(dtype=float)
                - table["start_time"].to_numpy(dtype=float)
            )
        ),
        "selected_ripple_intervals_sha256": _ripple_hash(table),
    }


def _interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite aligned interval bounds in seconds."""
    starts = np.asarray(getattr(intervals, "start"), dtype=float).reshape(-1)
    ends = np.asarray(getattr(intervals, "end"), dtype=float).reshape(-1)
    if starts.shape != ends.shape or not np.all(np.isfinite(starts)) or not np.all(
        np.isfinite(ends)
    ):
        raise ValueError("Interval bounds must be aligned and finite.")
    if np.any(ends <= starts) or (len(starts) > 1 and np.any(starts[1:] < ends[:-1])):
        raise ValueError("Intervals must be positive and non-overlapping.")
    return starts, ends


def _interval_duration(intervals: Any) -> float:
    """Return exact interval duration without relying on one implementation."""
    starts, ends = _interval_bounds(intervals)
    return float(np.sum(ends - starts))


def _require_ripples_inside_epoch(table: pd.DataFrame, epoch_interval: Any) -> None:
    """Require every selected ripple to remain exact inside the decode epoch."""
    epoch_starts, epoch_ends = _interval_bounds(epoch_interval)
    for start, end in table.to_numpy(dtype=float):
        if not np.any((start >= epoch_starts) & (end <= epoch_ends)):
            raise ValueError(
                "Every selected ripple must lie completely inside decode_epoch_interval."
            )


def _graph_geometry(
    graph_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate four path graphs and derive path lengths and physical-arm bounds."""
    if set(graph_inputs) != set(TRAJECTORY_TYPES):
        raise ValueError("graph_inputs must contain exactly four trajectory graphs.")
    from v1ca1.spyglass.path_specific_place import graph_length_from_inputs

    lengths: dict[str, float] = {}
    arm_starts: dict[str, float] = {}
    graph_payload: dict[str, Any] = {}
    for trajectory in TRAJECTORY_TYPES:
        graph = dict(graph_inputs[trajectory])
        length = float(
            graph_length_from_inputs(graph, trajectory_type=trajectory)
        )
        track = dict(graph.get("track_graph_kwargs", {}))
        linear = dict(graph.get("linearization_kwargs", {}))
        nodes = np.asarray(track.get("node_positions"), dtype=float)
        edges = np.asarray(track.get("edges"), dtype=int)
        edge_order = np.asarray(linear.get("edge_order", ()), dtype=int)
        spacing = np.asarray(linear.get("edge_spacing", ()), dtype=float).reshape(-1)
        if edge_order.ndim != 2 or edge_order.shape[1:] != (2,):
            raise ValueError("Each graph edge_order must have shape (n_edges, 2).")
        if spacing.size != max(0, len(edge_order) - 1):
            raise ValueError("Each graph edge_spacing must align to edge_order.")
        center_order = edge_order.copy()
        center_spacing = spacing.copy()
        if trajectory.endswith("_to_center"):
            center_order = center_order[::-1, ::-1]
            center_spacing = center_spacing[::-1]
        segment_lengths = np.linalg.norm(
            nodes[center_order[:, 1]] - nodes[center_order[:, 0]], axis=1
        )
        if len(segment_lengths) == 0 or not np.all(np.isfinite(segment_lengths)):
            raise ValueError("Each path graph must contain finite ordered segments.")
        if len(segment_lengths) == 1:
            arm_start = 0.0
        else:
            arm_start = float(
                np.sum(segment_lengths[:-2])
                + np.sum(center_spacing[: max(0, len(segment_lengths) - 2)])
                + 0.5 * segment_lengths[-2]
            )
        if not 0.0 <= arm_start < length:
            raise ValueError("Graph-derived terminal-arm boundary is invalid.")
        lengths[trajectory] = length
        arm_starts[trajectory] = arm_start
        graph_payload[trajectory] = {
            "configuration_name": str(graph.get("configuration_name", "")),
            "coordinate_unit": str(graph.get("coordinate_unit", "")),
            "node_positions": nodes.tolist(),
            "edges": edges.tolist(),
            "edge_order": edge_order.tolist(),
            "edge_spacing": spacing.tolist(),
            "use_HMM": bool(linear.get("use_HMM", False)),
        }
    common_length = lengths[TRAJECTORY_TYPES[0]]
    if common_length <= 0.0 or any(
        not np.isclose(value, common_length, rtol=1e-10, atol=1e-12)
        for value in lengths.values()
    ):
        raise ValueError("The four directional path graphs must have equal length.")
    geometry = {
        "trajectory_order": list(TRAJECTORY_TYPES),
        "path_length_cm": common_length,
        "path_length_cm_by_trajectory": lengths,
        "arm_start_cm_by_trajectory": arm_starts,
        "physical_arm_by_trajectory": PHYSICAL_ARM_BY_TRAJECTORY,
        "turn_group_by_trajectory": TURN_GROUP_BY_TRAJECTORY,
        "graphs": graph_payload,
    }
    geometry["graph_policy_sha256"] = _provenance_sha256(geometry)
    return geometry


def _build_repeated_edges(
    *, block_length: float, block_count: int, bin_width: float
) -> np.ndarray:
    """Build strictly increasing bin edges with exact block boundaries."""
    edges = [0.0]
    for index in range(int(block_count)):
        start = index * float(block_length)
        edges.extend(
            (start + np.arange(bin_width, block_length, bin_width)).tolist()
        )
        edges.append((index + 1) * float(block_length))
    output = np.asarray(edges, dtype=float)
    if np.any(np.diff(output) <= 0.0):
        raise ValueError("Repeated-block bin edges must be increasing.")
    return output


def _build_representation_feature(
    *,
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_interval: Any,
    representation: str,
    spatial_bin_size_cm: float,
    geometry: Mapping[str, Any],
) -> tuple[Any, np.ndarray, int]:
    """Build graph-derived four-block place or two-block DPP training input."""
    from v1ca1.spyglass.dpp import _pool_interval_sets
    from v1ca1.spyglass.path_specific_place import _intersect_intervals
    from v1ca1.spyglass.stability import build_task_progression_from_graph
    import pynapple as nap

    if set(trajectory_intervals) != set(TRAJECTORY_TYPES):
        raise ValueError("trajectory_intervals must contain exactly four paths.")
    supports = {
        trajectory: _intersect_intervals(
            trajectory_intervals[trajectory], movement_interval
        )
        for trajectory in TRAJECTORY_TYPES
    }
    pooled_support = _pool_interval_sets(supports)
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    common_length = float(geometry["path_length_cm"])
    for index, trajectory in enumerate(TRAJECTORY_TYPES):
        progression, length = build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals[trajectory],
            graph_inputs=graph_inputs[trajectory],
            trajectory_type=trajectory,
        )
        if not np.isclose(length, common_length, rtol=1e-10, atol=1e-12):
            raise ValueError("Linearization and graph-policy lengths disagree.")
        restricted = progression.restrict(supports[trajectory])
        times = np.asarray(restricted.t, dtype=float).reshape(-1)
        values = np.asarray(restricted.d, dtype=float).reshape(-1)
        finite = np.isfinite(times) & np.isfinite(values)
        times, values = times[finite], values[finite]
        if representation == "path_specific_place":
            if trajectory.endswith("_to_center"):
                values = 1.0 - values
            values = values * common_length + index * common_length
        else:
            values = values + DPP_OFFSET_BY_TRAJECTORY[trajectory]
        time_chunks.append(times)
        value_chunks.append(values)
    times = np.concatenate(time_chunks) if time_chunks else np.array([], dtype=float)
    values = np.concatenate(value_chunks) if value_chunks else np.array([], dtype=float)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    if len(times) > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Pooled trajectory feature timestamps must be unique.")
    feature = nap.Tsd(
        t=times,
        d=values,
        time_support=pooled_support,
        time_units="s",
    )
    if representation == "path_specific_place":
        bins = _build_repeated_edges(
            block_length=common_length,
            block_count=len(TRAJECTORY_TYPES),
            bin_width=float(spatial_bin_size_cm),
        )
    else:
        bins = _build_repeated_edges(
            block_length=1.0,
            block_count=2,
            bin_width=float(spatial_bin_size_cm) / common_length,
        )
    return feature, bins, int(len(times))


def _state_labels(
    state: np.ndarray,
    *,
    representation: str,
    scheme: str,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    """Map decoded states to corrected graph-derived categorical labels."""
    values = np.asarray(state, dtype=float).reshape(-1)
    labels = np.full(values.size, -1, dtype=int)
    path_length = float(geometry["path_length_cm"])
    if representation == "dpp":
        if scheme != "turn_group":
            return labels
        valid = np.isfinite(values) & (values >= 0.0) & (values <= 2.0)
        clipped = np.clip(values[valid], 0.0, np.nextafter(2.0, 0.0))
        labels[valid] = (clipped >= 1.0).astype(int)
        return labels
    upper = path_length * len(TRAJECTORY_TYPES)
    valid = np.isfinite(values) & (values >= 0.0) & (values <= upper)
    clipped = np.clip(values[valid], 0.0, np.nextafter(upper, 0.0))
    trajectories = np.floor(clipped / path_length).astype(int)
    local = clipped - trajectories * path_length
    if scheme == "trajectory":
        labels[valid] = trajectories
    elif scheme == "turn_group":
        labels[valid] = np.asarray(
            [
                TURN_GROUP_LABELS.index(
                    TURN_GROUP_BY_TRAJECTORY[TRAJECTORY_TYPES[index]]
                )
                for index in trajectories
            ],
            dtype=int,
        )
    elif scheme == "arm_identity":
        arm_labels = np.zeros(len(clipped), dtype=int)
        for index, trajectory in enumerate(TRAJECTORY_TYPES):
            trajectory_mask = trajectories == index
            arm_start = float(
                geometry["arm_start_cm_by_trajectory"][trajectory]
            )
            in_arm = trajectory_mask & (local >= arm_start)
            arm_labels[in_arm] = ARM_IDENTITY_LABELS.index(
                PHYSICAL_ARM_BY_TRAJECTORY[trajectory]
            )
        labels[valid] = arm_labels
    else:
        raise ValueError(f"Unsupported scoring scheme {scheme!r}.")
    return labels


def _scheme_availability(representation: str) -> dict[str, dict[str, Any]]:
    """Return fixed categorical applicability for one representation."""
    if representation == "path_specific_place":
        return {scheme: {"applicable": True, "reason": "ok"} for scheme in SCORING_SCHEMES}
    return {
        "trajectory": {
            "applicable": False,
            "reason": "dpp collapses four trajectories into two turn groups",
        },
        "turn_group": {"applicable": True, "reason": "ok"},
        "arm_identity": {
            "applicable": False,
            "reason": "dpp does not preserve within-path physical-arm occupancy",
        },
    }


def _boolean_array(values: Sequence[Any], *, name: str) -> np.ndarray:
    """Return a strictly validated boolean vector."""
    return np.asarray(
        [_database_bool(value, name=name) for value in values], dtype=bool
    )


def _identity_table(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    *,
    region: str,
    movement_firing_rates_hz: Sequence[float],
    minimum_rate_hz: float,
    allow_nonfinite_rates: bool = False,
) -> pd.DataFrame:
    """Return all input units with persistent identity and rate-selection audit."""
    from v1ca1.spyglass.movement import _stable_identity_rows

    rows = _stable_identity_rows(spikes, stable_unit_ids)
    rates = np.asarray(movement_firing_rates_hz, dtype=float).reshape(-1)
    if rates.size != len(rows):
        raise ValueError(
            f"{region} movement firing rates must align to the input TsGroup."
        )
    finite = np.isfinite(rates)
    if (not allow_nonfinite_rates and not np.all(finite)) or np.any(
        rates[finite] < 0.0
    ):
        raise ValueError(
            f"{region} movement firing rates must be finite and non-negative "
            "unless movement support is terminal."
        )
    records = []
    for index, (identity, rate) in enumerate(zip(rows, rates, strict=True)):
        passes = bool(
            not allow_nonfinite_rates
            and np.isfinite(rate)
            and rate >= float(minimum_rate_hz)
        )
        records.append(
            {
                "region": region,
                "spikesorting_merge_id": str(identity["spikesorting_merge_id"]),
                "unit_id": str(identity["unit_id"]),
                "stable_unit_id": str(identity["stable_unit_id"]),
                "group_unit_id": str(identity["group_unit_id"]),
                "input_unit_index": int(index),
                "movement_firing_rate_hz": float(rate),
                "minimum_movement_firing_rate_hz": float(minimum_rate_hz),
                "passes_movement_firing_rate": passes,
                "included_in_decoder": False,
                "unit_qc_status": (
                    "not_computed"
                    if allow_nonfinite_rates or passes
                    else "excluded_movement_firing_rate"
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SELECTED_UNIT_COLUMNS)


def _empty_decoded() -> dict[str, Any]:
    """Return one typed empty decoded-ripple payload."""
    return {
        "decoded_state": np.array([], dtype=float),
        "bin_times_s": np.array([], dtype=float),
        "ripple_ids": np.array([], dtype=int),
        "n_ripples_kept": 0,
        "n_bins": 0,
        "ripple_start_times_s": np.array([], dtype=float),
        "ripple_end_times_s": np.array([], dtype=float),
        "ripple_source_indices": np.array([], dtype=int),
        "skipped_ripples": [],
    }


def _canonical_decoded(decoded: Mapping[str, Any], *, region: str) -> dict[str, Any]:
    """Validate and normalize one regional decoded-ripple payload."""
    required = (
        "decoded_state",
        "bin_times_s",
        "ripple_ids",
        "ripple_start_times_s",
        "ripple_end_times_s",
        "ripple_source_indices",
    )
    missing = [name for name in required if name not in decoded]
    if missing:
        raise ValueError(f"{region} decoded payload is missing {missing!r}.")
    state = np.asarray(decoded["decoded_state"], dtype=float).reshape(-1)
    times = np.asarray(decoded["bin_times_s"], dtype=float).reshape(-1)
    ripple_ids = np.asarray(decoded["ripple_ids"], dtype=int).reshape(-1)
    starts = np.asarray(decoded["ripple_start_times_s"], dtype=float).reshape(-1)
    ends = np.asarray(decoded["ripple_end_times_s"], dtype=float).reshape(-1)
    source = np.asarray(decoded["ripple_source_indices"], dtype=int).reshape(-1)
    if state.shape != times.shape or state.shape != ripple_ids.shape:
        raise ValueError(f"{region} decoded bin arrays do not align.")
    if starts.shape != ends.shape or starts.shape != source.shape:
        raise ValueError(f"{region} decoded ripple arrays do not align.")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(times)):
        raise ValueError(f"{region} decoded bins must be finite.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError(f"{region} decoded ripple bounds must be finite.")
    if np.any(ends <= starts) or len(np.unique(source)) != len(source):
        raise ValueError(f"{region} decoded ripple metadata is invalid.")
    if len(source) == 0:
        if state.size or ripple_ids.size:
            raise ValueError(f"{region} empty ripple metadata has decoded bins.")
    else:
        if np.any(ripple_ids < 0) or not np.array_equal(
            np.unique(ripple_ids), np.arange(len(source), dtype=int)
        ):
            raise ValueError(f"{region} decoded ripple ids are not contiguous.")
        for ripple_id in range(len(source)):
            mask = ripple_ids == ripple_id
            if not np.any(mask):
                raise ValueError(f"{region} decoded ripple has no bins.")
            ripple_times = times[mask]
            if np.any(np.diff(ripple_times) <= 0.0):
                raise ValueError(f"{region} decoded times must increase per ripple.")
            tolerance = 1e-9
            if np.any(ripple_times < starts[ripple_id] - tolerance) or np.any(
                ripple_times > ends[ripple_id] + tolerance
            ):
                raise ValueError(f"{region} decoded bins lie outside ripple bounds.")
    skipped = []
    for row in decoded.get("skipped_ripples", []):
        if not isinstance(row, Mapping) or "ripple_index" not in row or "reason" not in row:
            raise ValueError(f"{region} skipped-ripple audit is malformed.")
        skipped.append(
            {"ripple_index": int(row["ripple_index"]), "reason": str(row["reason"])}
        )
    return {
        "decoded_state": state,
        "bin_times_s": times,
        "ripple_ids": ripple_ids,
        "n_ripples_kept": int(len(source)),
        "n_bins": int(len(state)),
        "ripple_start_times_s": starts,
        "ripple_end_times_s": ends,
        "ripple_source_indices": source,
        "skipped_ripples": skipped,
    }


def _empty_metric_table() -> pd.DataFrame:
    """Return an empty per-ripple metric table with canonical columns."""
    return pd.DataFrame({name: pd.Series(dtype=object) for name in RIPPLE_METRIC_COLUMNS})


def _metric_tables(
    *,
    aligned: Mapping[str, Any],
    metadata: Mapping[str, str],
    geometry: Mapping[str, Any],
    n_events_input: int,
    n_shuffles: int,
    shuffle_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Score corrected categorical agreement and the fixed ripple-block null."""
    science = _scientific_module()
    availability = _scheme_availability(metadata["representation"])
    ca1_state = np.asarray(aligned["ca1_decoded_state"], dtype=float)
    v1_state = np.asarray(aligned["v1_decoded_state"], dtype=float)
    ripple_ids = np.asarray(aligned["ripple_ids"], dtype=int)
    ca1_labels = {
        scheme: _state_labels(
            ca1_state,
            representation=metadata["representation"],
            scheme=scheme,
            geometry=geometry,
        )
        for scheme in SCORING_SCHEMES
    }
    v1_labels = {
        scheme: _state_labels(
            v1_state,
            representation=metadata["representation"],
            scheme=scheme,
            geometry=geometry,
        )
        for scheme in SCORING_SCHEMES
    }
    rows: list[dict[str, Any]] = []
    for ripple_id, source_index in enumerate(
        np.asarray(aligned["ripple_source_indices"], dtype=int)
    ):
        mask = ripple_ids == ripple_id
        row: dict[str, Any] = {
            **metadata,
            "ripple_id": int(ripple_id),
            "ripple_source_index": int(source_index),
            "ripple_start_time_s": float(aligned["ripple_start_times_s"][ripple_id]),
            "ripple_end_time_s": float(aligned["ripple_end_times_s"][ripple_id]),
            "n_bins": int(np.sum(mask)),
        }
        for scheme in SCORING_SCHEMES:
            applicable = bool(availability[scheme]["applicable"])
            row[f"{scheme}_scheme_requested"] = True
            row[f"{scheme}_scheme_applicable"] = applicable
            if applicable:
                values = science.compute_per_ripple_categorical_metrics(
                    ca1_labels=ca1_labels[scheme][mask],
                    v1_labels=v1_labels[scheme][mask],
                )
            else:
                values = {
                    "match_rate": np.nan,
                    "n_matching_bins": 0,
                    "n_valid_labeled_bins": 0,
                }
            row[f"{scheme}_match_rate"] = values["match_rate"]
            row[f"{scheme}_n_matching_bins"] = int(values["n_matching_bins"])
            row[f"{scheme}_n_valid_labeled_bins"] = int(
                values["n_valid_labeled_bins"]
            )
        rows.append(row)
    metrics = pd.DataFrame.from_records(rows, columns=RIPPLE_METRIC_COLUMNS)
    null = {
        scheme: np.full(int(n_shuffles), np.nan, dtype=float)
        for scheme in SCORING_SCHEMES
    }
    effective = 0
    if int(aligned["n_ripples"]) >= 2:
        rng = np.random.default_rng(int(shuffle_seed))
        for shuffle_index in range(int(n_shuffles)):
            shuffled, changed = science.shuffle_ripple_state_blocks_by_length(
                v1_state, ripple_ids, rng
            )
            if not changed:
                continue
            effective += 1
            for scheme in SCORING_SCHEMES:
                if not availability[scheme]["applicable"]:
                    continue
                shuffled_labels = _state_labels(
                    shuffled,
                    representation=metadata["representation"],
                    scheme=scheme,
                    geometry=geometry,
                )
                per_ripple = []
                for ripple_id in range(int(aligned["n_ripples"])):
                    mask = ripple_ids == ripple_id
                    value = science.compute_per_ripple_categorical_metrics(
                        ca1_labels=ca1_labels[scheme][mask],
                        v1_labels=shuffled_labels[mask],
                    )["match_rate"]
                    per_ripple.append(float(value))
                finite = np.asarray(per_ripple, dtype=float)
                if np.any(np.isfinite(finite)):
                    null[scheme][shuffle_index] = float(np.nanmean(finite))
    summary: dict[str, Any] = {
        **metadata,
        "n_ripple_events_input": int(n_events_input),
        "n_ripples": int(aligned["n_ripples"]),
        "n_ripple_bins": int(aligned["n_bins"]),
        "n_effective_shuffles": int(effective),
    }
    any_valid = False
    for scheme in SCORING_SCHEMES:
        applicable = bool(availability[scheme]["applicable"])
        summary[f"{scheme}_scheme_requested"] = True
        summary[f"{scheme}_scheme_applicable"] = applicable
        summary[f"{scheme}_scheme_reason"] = str(availability[scheme]["reason"])
        if applicable and not metrics.empty:
            observed_values = metrics[f"{scheme}_match_rate"].to_numpy(dtype=float)
            finite = np.isfinite(observed_values)
            n_valid = int(np.sum(finite))
            observed = float(np.nanmean(observed_values)) if n_valid else np.nan
            any_valid = any_valid or n_valid > 0
            shuffle = science.summarize_metric_against_shuffle(
                observed, null[scheme], direction="higher"
            )
        else:
            n_valid = 0
            observed = np.nan
            shuffle = {"shuffle_mean": np.nan, "shuffle_sd": np.nan, "p_value": np.nan}
        summary[f"{scheme}_n_valid_ripples"] = n_valid
        summary[f"{scheme}_match_rate"] = observed
        summary[f"{scheme}_match_rate_shuffle_mean"] = shuffle["shuffle_mean"]
        summary[f"{scheme}_match_rate_shuffle_sd"] = shuffle["shuffle_sd"]
        summary[f"{scheme}_match_rate_p_value"] = shuffle["p_value"]
    summary["analysis_status"] = "valid" if any_valid else "no_valid_metrics"
    labels = {
        f"{scheme}_{region}_label": values
        for scheme in SCORING_SCHEMES
        for region, values in (
            ("ca1", ca1_labels[scheme]),
            ("v1", v1_labels[scheme]),
        )
    }
    return (
        metrics,
        pd.DataFrame.from_records([summary], columns=EPOCH_SUMMARY_COLUMNS),
        null,
        labels,
    )


def _ripple_qc_table(
    *,
    ripple_table: pd.DataFrame,
    metadata: Mapping[str, str],
    ca1_decoded: Mapping[str, Any],
    v1_decoded: Mapping[str, Any],
    aligned: Mapping[str, Any],
    terminal_status: str | None = None,
) -> pd.DataFrame:
    """Return one row per exact input ripple with regional and alignment audit."""
    ca1_lookup = {
        int(source): local
        for local, source in enumerate(ca1_decoded["ripple_source_indices"])
    }
    v1_lookup = {
        int(source): local
        for local, source in enumerate(v1_decoded["ripple_source_indices"])
    }
    aligned_lookup = {
        int(source): local
        for local, source in enumerate(aligned["ripple_source_indices"])
    }
    ca1_skip = {
        int(row["ripple_index"]): str(row["reason"])
        for row in ca1_decoded.get("skipped_ripples", [])
    }
    v1_skip = {
        int(row["ripple_index"]): str(row["reason"])
        for row in v1_decoded.get("skipped_ripples", [])
    }
    align_skip = {
        int(row["ripple_index"]): str(row["reason"])
        for row in aligned.get("skipped_ripples", [])
    }
    rows = []
    for source_index, ripple in ripple_table.reset_index(drop=True).iterrows():
        if terminal_status is not None:
            ca1_status = v1_status = alignment_status = "not_computed"
            ca1_reason = v1_reason = alignment_reason = terminal_status
        else:
            ca1_status = "valid" if source_index in ca1_lookup else "skipped"
            v1_status = "valid" if source_index in v1_lookup else "skipped"
            alignment_status = (
                "valid" if source_index in aligned_lookup else "skipped"
            )
            ca1_reason = "ok" if ca1_status == "valid" else ca1_skip.get(
                source_index, "decoding did not retain this ripple"
            )
            v1_reason = "ok" if v1_status == "valid" else v1_skip.get(
                source_index, "decoding did not retain this ripple"
            )
            if alignment_status == "valid":
                alignment_reason = "ok"
            elif ca1_status != "valid" or v1_status != "valid":
                alignment_reason = "one or both regional decoders skipped this ripple"
            else:
                alignment_reason = align_skip.get(
                    source_index, "regional decoded bins did not align"
                )
        ca1_local = ca1_lookup.get(source_index)
        v1_local = v1_lookup.get(source_index)
        aligned_local = aligned_lookup.get(source_index)
        rows.append(
            {
                **metadata,
                "ripple_source_index": int(source_index),
                "ripple_start_time_s": float(ripple["start_time"]),
                "ripple_end_time_s": float(ripple["end_time"]),
                "ca1_decoding_status": ca1_status,
                "ca1_decoding_reason": ca1_reason,
                "ca1_n_bins": (
                    0
                    if ca1_local is None
                    else int(np.sum(ca1_decoded["ripple_ids"] == ca1_local))
                ),
                "v1_decoding_status": v1_status,
                "v1_decoding_reason": v1_reason,
                "v1_n_bins": (
                    0
                    if v1_local is None
                    else int(np.sum(v1_decoded["ripple_ids"] == v1_local))
                ),
                "alignment_status": alignment_status,
                "alignment_reason": alignment_reason,
                "n_aligned_bins": (
                    0
                    if aligned_local is None
                    else int(np.sum(aligned["ripple_ids"] == aligned_local))
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=RIPPLE_QC_COLUMNS)


def _terminal_summary(
    *,
    metadata: Mapping[str, str],
    n_events_input: int,
    status: str,
) -> pd.DataFrame:
    """Return one canonical summary row for an expected terminal condition."""
    availability = _scheme_availability(metadata["representation"])
    row: dict[str, Any] = {
        **metadata,
        "n_ripple_events_input": int(n_events_input),
        "n_ripples": 0,
        "n_ripple_bins": 0,
        "n_effective_shuffles": 0,
        "analysis_status": status,
    }
    for scheme in SCORING_SCHEMES:
        row[f"{scheme}_scheme_requested"] = True
        row[f"{scheme}_scheme_applicable"] = bool(
            availability[scheme]["applicable"]
        )
        row[f"{scheme}_scheme_reason"] = str(availability[scheme]["reason"])
        row[f"{scheme}_n_valid_ripples"] = 0
        row[f"{scheme}_match_rate"] = np.nan
        row[f"{scheme}_match_rate_shuffle_mean"] = np.nan
        row[f"{scheme}_match_rate_shuffle_sd"] = np.nan
        row[f"{scheme}_match_rate_p_value"] = np.nan
    return pd.DataFrame.from_records([row], columns=EPOCH_SUMMARY_COLUMNS)


def _dataset(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    selected_ripple_intervals_sha256: str,
    geometry: Mapping[str, Any],
    ripple_table: pd.DataFrame,
    selected_units: pd.DataFrame,
    ripple_qc: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ca1_decoded: Mapping[str, Any],
    v1_decoded: Mapping[str, Any],
    aligned: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    null_samples: Mapping[str, np.ndarray],
    bin_edges: np.ndarray,
    feature_sample_count: int,
    analysis_status: str,
    artifact_origin: str = "computed",
    legacy_artifact_provenance: Mapping[str, Any] | None = None,
) -> Any:
    """Build the self-describing NetCDF payload for one comparison."""
    import xarray as xr

    legacy = {} if legacy_artifact_provenance is None else dict(
        legacy_artifact_provenance
    )
    ca1_units = selected_units.loc[selected_units["region"] == SOURCE_REGION]
    v1_units = selected_units.loc[selected_units["region"] == TARGET_REGION]
    availability = _scheme_availability(metadata["representation"])
    attrs = {
        "ripple_decoding_comparison_result_schema_version": RESULT_SCHEMA_VERSION,
        **metadata,
        "comparison_direction": "ca1_vs_v1_decoded_state",
        "parameter_name": str(parameters["parameter_name"]),
        "parameter_sha256": str(parameters["parameter_sha256"]),
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        "effective_parameters_json": json.dumps(
            {
                name: value
                for name, value in parameters.items()
                if name not in {
                    "parameter_name",
                    "parameter_sha256",
                    "output_rule_sha256",
                }
            },
            sort_keys=True,
        ),
        "upstream_provenance_json": upstream_provenance_json,
        "selected_ripple_intervals_sha256": selected_ripple_intervals_sha256,
        "graph_policy_sha256": str(geometry["graph_policy_sha256"]),
        "graph_geometry_json": json.dumps(dict(geometry), sort_keys=True),
        "scheme_availability_json": json.dumps(availability, sort_keys=True),
        "feature_sample_count": int(feature_sample_count),
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance_json": json.dumps(legacy, sort_keys=True),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
    data_vars: dict[str, Any] = {
        "bin_time_s": (("bin",), np.asarray(aligned["bin_times_s"], dtype=float)),
        "ripple_id": (("bin",), np.asarray(aligned["ripple_ids"], dtype=int)),
        "ca1_decoded_state": (
            ("bin",),
            np.asarray(aligned["ca1_decoded_state"], dtype=float),
        ),
        "v1_decoded_state": (
            ("bin",),
            np.asarray(aligned["v1_decoded_state"], dtype=float),
        ),
        "ripple_source_index": (
            ("ripple",),
            np.asarray(aligned["ripple_source_indices"], dtype=int),
        ),
        "ripple_start_time_s": (
            ("ripple",),
            np.asarray(aligned["ripple_start_times_s"], dtype=float),
        ),
        "ripple_end_time_s": (
            ("ripple",),
            np.asarray(aligned["ripple_end_times_s"], dtype=float),
        ),
        "input_ripple_start_time_s": (
            ("input_ripple",),
            ripple_table["start_time"].to_numpy(dtype=float),
        ),
        "input_ripple_end_time_s": (
            ("input_ripple",),
            ripple_table["end_time"].to_numpy(dtype=float),
        ),
        "input_ripple_alignment_status": (
            ("input_ripple",),
            ripple_qc["alignment_status"].to_numpy(dtype=str),
        ),
        "input_ripple_alignment_reason": (
            ("input_ripple",),
            ripple_qc["alignment_reason"].to_numpy(dtype=str),
        ),
        "ca1_movement_firing_rate_hz": (
            ("ca1_unit",),
            ca1_units["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "ca1_keep_unit": (
            ("ca1_unit",),
            ca1_units["included_in_decoder"].to_numpy(dtype=bool),
        ),
        "v1_movement_firing_rate_hz": (
            ("v1_unit",),
            v1_units["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "v1_keep_unit": (
            ("v1_unit",),
            v1_units["included_in_decoder"].to_numpy(dtype=bool),
        ),
    }
    for scheme in SCORING_SCHEMES:
        ca1_label = np.asarray(
            labels.get(f"{scheme}_ca1_label", np.full(aligned["n_bins"], -1)),
            dtype=int,
        )
        v1_label = np.asarray(
            labels.get(f"{scheme}_v1_label", np.full(aligned["n_bins"], -1)),
            dtype=int,
        )
        valid = (ca1_label >= 0) & (v1_label >= 0)
        matches = np.zeros(valid.shape, dtype=bool)
        matches[valid] = ca1_label[valid] == v1_label[valid]
        data_vars[f"{scheme}_ca1_label"] = (("bin",), ca1_label)
        data_vars[f"{scheme}_v1_label"] = (("bin",), v1_label)
        data_vars[f"{scheme}_bin_match"] = (("bin",), matches)
        for metric_name in (
            "match_rate",
            "n_matching_bins",
            "n_valid_labeled_bins",
        ):
            column = f"{scheme}_{metric_name}"
            metric_dtype = float if metric_name == "match_rate" else int
            data_vars[column] = (
                ("ripple",),
                metrics.get(column, pd.Series(dtype=metric_dtype)).to_numpy(
                    dtype=metric_dtype
                ),
            )
        null = np.asarray(
            null_samples.get(scheme, np.full(parameters["n_shuffles"], np.nan)),
            dtype=float,
        )
        data_vars[f"{scheme}_match_rate_shuffle"] = (("shuffle",), null)
        for suffix in (
            "match_rate",
            "match_rate_shuffle_mean",
            "match_rate_shuffle_sd",
            "match_rate_p_value",
        ):
            column = f"{scheme}_{suffix}"
            data_vars[f"{scheme}_{suffix}_observed" if suffix == "match_rate" else column] = (
                (),
                float(summary.iloc[0][column]),
            )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "bin": np.arange(int(aligned["n_bins"]), dtype=int),
            "ripple": np.arange(int(aligned["n_ripples"]), dtype=int),
            "input_ripple": np.arange(len(ripple_table), dtype=int),
            "shuffle": np.arange(int(parameters["n_shuffles"]), dtype=int),
            "spatial_bin_edge": np.asarray(bin_edges, dtype=float),
            "ca1_unit": ca1_units["stable_unit_id"].to_numpy(dtype=str),
            "v1_unit": v1_units["stable_unit_id"].to_numpy(dtype=str),
            "ca1_spikesorting_merge_id": (
                ("ca1_unit",),
                ca1_units["spikesorting_merge_id"].to_numpy(dtype=str),
            ),
            "ca1_source_unit_id": (
                ("ca1_unit",), ca1_units["unit_id"].to_numpy(dtype=str)
            ),
            "v1_spikesorting_merge_id": (
                ("v1_unit",),
                v1_units["spikesorting_merge_id"].to_numpy(dtype=str),
            ),
            "v1_source_unit_id": (
                ("v1_unit",), v1_units["unit_id"].to_numpy(dtype=str)
            ),
        },
        attrs=attrs,
    )


def _empty_aligned() -> dict[str, Any]:
    """Return one typed empty CA1/V1 alignment payload."""
    return {
        "ca1_decoded_state": np.array([], dtype=float),
        "v1_decoded_state": np.array([], dtype=float),
        "bin_times_s": np.array([], dtype=float),
        "ripple_ids": np.array([], dtype=int),
        "n_ripples": 0,
        "n_bins": 0,
        "ripple_source_indices": np.array([], dtype=int),
        "ripple_start_times_s": np.array([], dtype=float),
        "ripple_end_times_s": np.array([], dtype=float),
        "skipped_ripples": [],
    }


def _canonical_aligned(aligned: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one aligned regional decoded-state payload."""
    output = _empty_aligned()
    for name in (
        "ca1_decoded_state",
        "v1_decoded_state",
        "bin_times_s",
    ):
        output[name] = np.asarray(aligned[name], dtype=float).reshape(-1)
    output["ripple_ids"] = np.asarray(aligned["ripple_ids"], dtype=int).reshape(-1)
    for name in (
        "ripple_source_indices",
        "ripple_start_times_s",
        "ripple_end_times_s",
    ):
        dtype = int if name == "ripple_source_indices" else float
        output[name] = np.asarray(aligned[name], dtype=dtype).reshape(-1)
    if not (
        output["ca1_decoded_state"].shape
        == output["v1_decoded_state"].shape
        == output["bin_times_s"].shape
        == output["ripple_ids"].shape
    ):
        raise ValueError("Aligned decoded bin arrays do not share a shape.")
    if not (
        output["ripple_source_indices"].shape
        == output["ripple_start_times_s"].shape
        == output["ripple_end_times_s"].shape
    ):
        raise ValueError("Aligned decoded ripple arrays do not share a shape.")
    if not np.all(np.isfinite(output["ca1_decoded_state"])) or not np.all(
        np.isfinite(output["v1_decoded_state"])
    ):
        raise ValueError("Aligned decoded states must be finite.")
    if not np.all(np.isfinite(output["bin_times_s"])):
        raise ValueError("Aligned decoded times must be finite.")
    n_ripples = len(output["ripple_source_indices"])
    if len(set(output["ripple_source_indices"].tolist())) != n_ripples:
        raise ValueError("Aligned ripple source indices must be unique.")
    if n_ripples:
        expected = np.arange(n_ripples, dtype=int)
        if not np.array_equal(np.unique(output["ripple_ids"]), expected):
            raise ValueError("Aligned ripple ids must be contiguous.")
    elif output["ripple_ids"].size:
        raise ValueError("Aligned bins cannot exist without ripple rows.")
    output["n_ripples"] = int(n_ripples)
    output["n_bins"] = int(len(output["bin_times_s"]))
    output["skipped_ripples"] = [
        {"ripple_index": int(row["ripple_index"]), "reason": str(row["reason"])}
        for row in aligned.get("skipped_ripples", [])
    ]
    return output


def _representation_bins(
    *, representation: str, geometry: Mapping[str, Any], spatial_bin_size_cm: float
) -> np.ndarray:
    """Return the graph-derived fixed spatial grid without position samples."""
    path_length = float(geometry["path_length_cm"])
    if representation == "path_specific_place":
        return _build_repeated_edges(
            block_length=path_length,
            block_count=len(TRAJECTORY_TYPES),
            bin_width=float(spatial_bin_size_cm),
        )
    return _build_repeated_edges(
        block_length=1.0,
        block_count=2,
        bin_width=float(spatial_bin_size_cm) / path_length,
    )


def _terminal_from_inputs(
    *, selected_units: pd.DataFrame, n_ripples: int, movement_duration_s: float
) -> str | None:
    """Return the first fixed expected terminal status implied by inputs."""
    ca1 = selected_units.loc[selected_units["region"] == SOURCE_REGION]
    v1 = selected_units.loc[selected_units["region"] == TARGET_REGION]
    if movement_duration_s <= 0.0:
        return "no_train_movement"
    if n_ripples == 0:
        return "no_ripples"
    if ca1.empty:
        return "no_ca1_units"
    if v1.empty:
        return "no_v1_units"
    if not ca1["passes_movement_firing_rate"].any():
        return "no_eligible_ca1_units"
    if not v1["passes_movement_firing_rate"].any():
        return "no_eligible_v1_units"
    return None


def _finish_result(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
    upstream_provenance_json: str,
    selected_hash: str,
    geometry: Mapping[str, Any],
    ripple_table: pd.DataFrame,
    selected_units: pd.DataFrame,
    ripple_qc: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    ca1_decoded: Mapping[str, Any],
    v1_decoded: Mapping[str, Any],
    aligned: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    null_samples: Mapping[str, np.ndarray],
    bin_edges: np.ndarray,
    feature_sample_count: int,
    analysis_status: str,
    artifact_origin: str = "computed",
    legacy_artifact_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble and strictly validate one in-memory result bundle."""
    dataset = _dataset(
        metadata=metadata,
        parameters=parameters,
        upstream_provenance_json=upstream_provenance_json,
        selected_ripple_intervals_sha256=selected_hash,
        geometry=geometry,
        ripple_table=ripple_table,
        selected_units=selected_units,
        ripple_qc=ripple_qc,
        metrics=metrics,
        summary=summary,
        ca1_decoded=ca1_decoded,
        v1_decoded=v1_decoded,
        aligned=aligned,
        labels=labels,
        null_samples=null_samples,
        bin_edges=bin_edges,
        feature_sample_count=feature_sample_count,
        analysis_status=analysis_status,
        artifact_origin=artifact_origin,
        legacy_artifact_provenance=legacy_artifact_provenance,
    )
    return validate_ripple_decoding_comparison_result(
        {
            **metadata,
            "parameters": dict(parameters),
            "upstream_provenance": dict(upstream_provenance),
            "selected_ripple_intervals_sha256": selected_hash,
            "graph_geometry": dict(geometry),
            "selected_units": selected_units,
            "ripple_qc": ripple_qc,
            "ripple_metrics": metrics,
            "epoch_summary": summary,
            "ca1_decoded": dict(ca1_decoded),
            "v1_decoded": dict(v1_decoded),
            "dataset": dataset,
            "analysis_status": analysis_status,
            "artifact_origin": artifact_origin,
            "legacy_artifact_provenance": (
                {} if legacy_artifact_provenance is None else dict(legacy_artifact_provenance)
            ),
        }
    )


def compute_ripple_decoding_comparison(
    *,
    ripple_decoding_comparison_id: Any,
    animal_name: str,
    date: str,
    train_epoch: str,
    decode_epoch: str,
    representation: str,
    ca1_spikes: Any,
    ca1_stable_unit_ids: Sequence[Mapping[str, Any]],
    ca1_movement_firing_rates_hz: Sequence[float],
    v1_spikes: Any,
    v1_stable_unit_ids: Sequence[Mapping[str, Any]],
    v1_movement_firing_rates_hz: Sequence[float],
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_interval: Any,
    ripple_table: Any,
    decode_epoch_interval: Any,
    upstream_provenance: Mapping[str, Any],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    expected_selected_ripple_intervals_sha256: str | None = None,
    expected_graph_policy_sha256: str | None = None,
    decode_bin_size_s: float = DEFAULT_DECODE_BIN_SIZE_S,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    tuning_smoothing_sigma_bins: float = DEFAULT_TUNING_SMOOTHING_SIGMA_BINS,
    ca1_min_movement_rate_hz: float = DEFAULT_CA1_MIN_MOVEMENT_RATE_HZ,
    v1_min_movement_rate_hz: float = DEFAULT_V1_MIN_MOVEMENT_RATE_HZ,
    n_shuffles: int = DEFAULT_N_SHUFFLES,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    expected_detector_zscore_threshold: float = DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD,
    require_speed_gated: bool = DEFAULT_REQUIRE_SPEED_GATED,
) -> dict[str, Any]:
    """Compute one NWB-derived CA1/V1 ripple decoding comparison."""
    metadata = _metadata(
        ripple_decoding_comparison_id=ripple_decoding_comparison_id,
        animal_name=animal_name,
        date=date,
        train_epoch=train_epoch,
        decode_epoch=decode_epoch,
        representation=representation,
    )
    parameters = _effective_parameters(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        decode_bin_size_s=decode_bin_size_s,
        spatial_bin_size_cm=spatial_bin_size_cm,
        tuning_smoothing_sigma_bins=tuning_smoothing_sigma_bins,
        ca1_min_movement_rate_hz=ca1_min_movement_rate_hz,
        v1_min_movement_rate_hz=v1_min_movement_rate_hz,
        n_shuffles=n_shuffles,
        shuffle_seed=shuffle_seed,
        expected_detector_zscore_threshold=expected_detector_zscore_threshold,
        require_speed_gated=require_speed_gated,
    )
    provenance, provenance_json = _canonical_provenance(
        upstream_provenance, parameters=parameters
    )
    events = prepare_ripple_decoding_comparison_event_selection(
        decode_epoch=metadata["decode_epoch"], ripple_table=ripple_table
    )
    ripples = events["selected_ripple_table"]
    selected_hash = str(events["selected_ripple_intervals_sha256"])
    if expected_selected_ripple_intervals_sha256 is not None and str(
        expected_selected_ripple_intervals_sha256
    ) != selected_hash:
        raise ValueError("Selected ripple intervals changed after input selection.")
    _require_ripples_inside_epoch(ripples, decode_epoch_interval)
    geometry = _graph_geometry(graph_inputs)
    if expected_graph_policy_sha256 is not None and str(
        expected_graph_policy_sha256
    ) != str(geometry["graph_policy_sha256"]):
        raise ValueError("WTrackGraph inputs changed after selection.")
    movement_duration_s = _interval_duration(movement_interval)
    movement_is_terminal = movement_duration_s <= 0.0
    ca1_units = _identity_table(
        ca1_spikes,
        ca1_stable_unit_ids,
        region=SOURCE_REGION,
        movement_firing_rates_hz=ca1_movement_firing_rates_hz,
        minimum_rate_hz=parameters["ca1_min_movement_rate_hz"],
        allow_nonfinite_rates=movement_is_terminal,
    )
    v1_units = _identity_table(
        v1_spikes,
        v1_stable_unit_ids,
        region=TARGET_REGION,
        movement_firing_rates_hz=v1_movement_firing_rates_hz,
        minimum_rate_hz=parameters["v1_min_movement_rate_hz"],
        allow_nonfinite_rates=movement_is_terminal,
    )
    selected_units = pd.concat([ca1_units, v1_units], ignore_index=True)
    bin_edges = _representation_bins(
        representation=metadata["representation"],
        geometry=geometry,
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
    )
    terminal = _terminal_from_inputs(
        selected_units=selected_units,
        n_ripples=len(ripples),
        movement_duration_s=movement_duration_s,
    )
    ca1_decoded = _empty_decoded()
    v1_decoded = _empty_decoded()
    aligned = _empty_aligned()
    feature_sample_count = 0
    labels: dict[str, np.ndarray] = {}
    null = {
        scheme: np.full(parameters["n_shuffles"], np.nan, dtype=float)
        for scheme in SCORING_SCHEMES
    }
    if terminal is None:
        feature, observed_bins, feature_sample_count = _build_representation_feature(
            position=position,
            trajectory_intervals=trajectory_intervals,
            graph_inputs=graph_inputs,
            movement_interval=movement_interval,
            representation=metadata["representation"],
            spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
            geometry=geometry,
        )
        if not np.array_equal(observed_bins, bin_edges):
            raise ValueError("Feature builder returned a stale graph-derived spatial grid.")
        if feature_sample_count == 0:
            terminal = "no_train_trajectory_samples"
    if terminal is None:
        science = _scientific_module()
        ca1_mask = ca1_units["passes_movement_firing_rate"].to_numpy(dtype=bool)
        v1_mask = v1_units["passes_movement_firing_rate"].to_numpy(dtype=bool)
        ca1_keys = np.asarray(list(ca1_spikes.keys()), dtype=object)[ca1_mask].tolist()
        v1_keys = np.asarray(list(v1_spikes.keys()), dtype=object)[v1_mask].tolist()
        filtered_ca1 = science._subset_spikes(ca1_spikes, ca1_keys)
        filtered_v1 = science._subset_spikes(v1_spikes, v1_keys)
        ca1_tuning = science.compute_tuning_curves_for_epoch(
            spikes=filtered_ca1,
            feature=feature,
            movement_interval=movement_interval,
            bin_edges=bin_edges,
            feature_name=metadata["representation"],
        )
        v1_tuning = science.compute_tuning_curves_for_epoch(
            spikes=filtered_v1,
            feature=feature,
            movement_interval=movement_interval,
            bin_edges=bin_edges,
            feature_name=metadata["representation"],
        )
        ca1_decoded = _canonical_decoded(
            science.assemble_decoded_ripple_epoch_data(
                spikes=filtered_ca1,
                tuning_curves=ca1_tuning,
                ripple_table=ripples,
                epoch_interval=decode_epoch_interval,
                bin_size_s=parameters["decode_bin_size_s"],
            ),
            region=SOURCE_REGION,
        )
        v1_decoded = _canonical_decoded(
            science.assemble_decoded_ripple_epoch_data(
                spikes=filtered_v1,
                tuning_curves=v1_tuning,
                ripple_table=ripples,
                epoch_interval=decode_epoch_interval,
                bin_size_s=parameters["decode_bin_size_s"],
            ),
            region=TARGET_REGION,
        )
        aligned = _canonical_aligned(
            science.align_decoded_ripple_data(ca1_decoded, v1_decoded)
        )
        selected_units.loc[
            selected_units["passes_movement_firing_rate"], "included_in_decoder"
        ] = True
        selected_units.loc[
            selected_units["passes_movement_firing_rate"], "unit_qc_status"
        ] = "included"
        if aligned["n_bins"] == 0:
            terminal = "no_common_decoded_bins"
    if terminal is not None:
        ripple_qc = _ripple_qc_table(
            ripple_table=ripples,
            metadata=metadata,
            ca1_decoded=ca1_decoded,
            v1_decoded=v1_decoded,
            aligned=aligned,
            terminal_status=(
                terminal
                if terminal not in {"no_common_decoded_bins"}
                else None
            ),
        )
        metrics = _empty_metric_table()
        summary = _terminal_summary(
            metadata=metadata, n_events_input=len(ripples), status=terminal
        )
        analysis_status = terminal
    else:
        metrics, summary, null, labels = _metric_tables(
            aligned=aligned,
            metadata=metadata,
            geometry=geometry,
            n_events_input=len(ripples),
            n_shuffles=parameters["n_shuffles"],
            shuffle_seed=parameters["shuffle_seed"],
        )
        if str(summary.iloc[0]["analysis_status"]) == "no_valid_metrics":
            analysis_status = "no_valid_metrics"
        elif aligned["n_ripples"] < len(ripples):
            analysis_status = "partial_valid"
        else:
            analysis_status = "valid"
        summary.loc[:, "analysis_status"] = analysis_status
        ripple_qc = _ripple_qc_table(
            ripple_table=ripples,
            metadata=metadata,
            ca1_decoded=ca1_decoded,
            v1_decoded=v1_decoded,
            aligned=aligned,
        )
    return _finish_result(
        metadata=metadata,
        parameters=parameters,
        upstream_provenance=provenance,
        upstream_provenance_json=provenance_json,
        selected_hash=selected_hash,
        geometry=geometry,
        ripple_table=ripples,
        selected_units=selected_units,
        ripple_qc=ripple_qc,
        metrics=metrics,
        summary=summary,
        ca1_decoded=ca1_decoded,
        v1_decoded=v1_decoded,
        aligned=aligned,
        labels=labels,
        null_samples=null,
        bin_edges=bin_edges,
        feature_sample_count=feature_sample_count,
        analysis_status=analysis_status,
    )


def _assert_frame_equal(
    observed: pd.DataFrame, expected: pd.DataFrame, *, name: str
) -> None:
    """Raise a concise error when one scientific table differs."""
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(f"{name} differs from its canonical reconstruction.") from exc


def _validate_selected_units(
    table: Any, *, parameters: Mapping[str, Any], analysis_status: str
) -> pd.DataFrame:
    """Validate all-unit persistent identity and inclusion decisions."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != SELECTED_UNIT_COLUMNS:
        raise ValueError("selected_units does not match its canonical schema.")
    output = table.copy().reset_index(drop=True)
    if not output["region"].astype(str).isin((SOURCE_REGION, TARGET_REGION)).all():
        raise ValueError("selected_units contains an unsupported region.")
    for region in (SOURCE_REGION, TARGET_REGION):
        subset = output.loc[output["region"].astype(str) == region]
        indices = pd.to_numeric(subset["input_unit_index"], errors="raise").to_numpy(
            dtype=int
        )
        if not np.array_equal(indices, np.arange(len(subset), dtype=int)):
            raise ValueError(f"{region} input unit indices are not contiguous.")
        for column in IDENTITY_COLUMNS:
            if subset[column].map(str).eq("").any():
                raise ValueError(f"{region} contains empty unit identities.")
        expected_stable = (
            subset["spikesorting_merge_id"].map(str)
            + ":"
            + subset["unit_id"].map(str)
        )
        if not expected_stable.equals(subset["stable_unit_id"].map(str)):
            raise ValueError(f"{region} stable unit ids are noncanonical.")
        if subset["stable_unit_id"].map(str).duplicated().any():
            raise ValueError(f"{region} stable unit ids must be unique.")
        rates = pd.to_numeric(
            subset["movement_firing_rate_hz"], errors="raise"
        ).to_numpy(dtype=float)
        finite = np.isfinite(rates)
        movement_terminal = analysis_status == "no_train_movement"
        if (not movement_terminal and not np.all(finite)) or np.any(
            rates[finite] < 0.0
        ):
            raise ValueError(f"{region} movement rates are invalid.")
        threshold_name = f"{region}_min_movement_rate_hz"
        thresholds = pd.to_numeric(
            subset["minimum_movement_firing_rate_hz"], errors="raise"
        ).to_numpy(dtype=float)
        if not np.allclose(
            thresholds, parameters[threshold_name], rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{region} unit threshold differs from parameters.")
        passes = _boolean_array(
            subset["passes_movement_firing_rate"].tolist(),
            name=f"{region} passes_movement_firing_rate",
        )
        expected_passes = (
            np.zeros(len(rates), dtype=bool)
            if movement_terminal
            else rates >= parameters[threshold_name]
        )
        if not np.array_equal(passes, expected_passes):
            raise ValueError(f"{region} unit selection flags are stale.")
        included = _boolean_array(
            subset["included_in_decoder"].tolist(),
            name=f"{region} included_in_decoder",
        )
        if np.any(included & ~passes):
            raise ValueError(f"{region} excluded units entered the decoder.")
        statuses = subset["unit_qc_status"].astype(str).to_numpy()
        expected = (
            np.full(len(subset), "not_computed", dtype=object)
            if movement_terminal
            else np.where(
                ~passes,
                "excluded_movement_firing_rate",
                np.where(included, "included", "not_computed"),
            )
        )
        if not np.array_equal(statuses, expected):
            raise ValueError(f"{region} unit QC statuses are inconsistent.")
        if analysis_status in FITTED_STATUSES + ("no_common_decoded_bins",) and len(
            subset
        ) and not np.array_equal(included, passes):
            raise ValueError(f"{region} eligible fitted units must enter the decoder.")
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _validate_dataset_identity(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    selected_hash: str,
    geometry: Mapping[str, Any],
    selected_units: pd.DataFrame,
    ripple_table: pd.DataFrame,
    ripple_qc: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    aligned: Mapping[str, Any],
    labels: Mapping[str, np.ndarray],
    null_samples: Mapping[str, np.ndarray],
    analysis_status: str,
    artifact_origin: str,
    legacy_artifact_provenance: Mapping[str, Any],
) -> None:
    """Validate NetCDF metadata and all reconstructable scientific variables."""
    if dataset is None or not hasattr(dataset, "attrs") or not hasattr(dataset, "sizes"):
        raise TypeError("dataset must be xarray Dataset-like.")
    attrs = {
        "ripple_decoding_comparison_result_schema_version": RESULT_SCHEMA_VERSION,
        **metadata,
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        "upstream_provenance_json": upstream_provenance_json,
        "selected_ripple_intervals_sha256": selected_hash,
        "graph_policy_sha256": geometry["graph_policy_sha256"],
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
    for name, expected in attrs.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"dataset has mismatched {name!r}.")
    try:
        effective = json.loads(str(dataset.attrs["effective_parameters_json"]))
        stored_geometry = json.loads(str(dataset.attrs["graph_geometry_json"]))
        stored_legacy = json.loads(
            str(dataset.attrs["legacy_artifact_provenance_json"])
        )
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("dataset contains malformed JSON provenance.") from exc
    expected_effective = {
        name: value
        for name, value in parameters.items()
        if name not in {"parameter_name", "parameter_sha256", "output_rule_sha256"}
    }
    if effective != expected_effective or stored_geometry != dict(geometry):
        raise ValueError("dataset parameters or graph geometry differ from the row.")
    if stored_legacy != dict(legacy_artifact_provenance):
        raise ValueError("dataset legacy provenance differs from the row.")
    expected_sizes = {
        "bin": int(aligned["n_bins"]),
        "ripple": int(aligned["n_ripples"]),
        "input_ripple": len(ripple_table),
        "shuffle": int(parameters["n_shuffles"]),
    }
    for name, size in expected_sizes.items():
        if int(dataset.sizes.get(name, -1)) != int(size):
            raise ValueError(f"dataset has mismatched {name!r} size.")
    array_checks = {
        "bin_time_s": aligned["bin_times_s"],
        "ripple_id": aligned["ripple_ids"],
        "ca1_decoded_state": aligned["ca1_decoded_state"],
        "v1_decoded_state": aligned["v1_decoded_state"],
        "ripple_source_index": aligned["ripple_source_indices"],
        "ripple_start_time_s": aligned["ripple_start_times_s"],
        "ripple_end_time_s": aligned["ripple_end_times_s"],
        "input_ripple_start_time_s": ripple_table["start_time"].to_numpy(),
        "input_ripple_end_time_s": ripple_table["end_time"].to_numpy(),
    }
    for name, expected in array_checks.items():
        if name not in dataset or not np.allclose(
            np.asarray(dataset[name].values),
            np.asarray(expected),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"dataset variable {name!r} is mismatched.")
    if not np.array_equal(
        np.asarray(dataset["input_ripple_alignment_status"].values).astype(str),
        ripple_qc["alignment_status"].to_numpy(dtype=str),
    ):
        raise ValueError("dataset ripple QC status is mismatched.")
    for region in (SOURCE_REGION, TARGET_REGION):
        units = selected_units.loc[selected_units["region"] == region]
        if not np.array_equal(
            np.asarray(dataset.coords[f"{region}_unit"].values).astype(str),
            units["stable_unit_id"].to_numpy(dtype=str),
        ):
            raise ValueError(f"dataset {region} unit identity is misaligned.")
        if not np.allclose(
            np.asarray(dataset[f"{region}_movement_firing_rate_hz"].values),
            units["movement_firing_rate_hz"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ) or not np.array_equal(
            np.asarray(dataset[f"{region}_keep_unit"].values, dtype=bool),
            units["included_in_decoder"].to_numpy(dtype=bool),
        ):
            raise ValueError(f"dataset {region} unit audit is mismatched.")
    for scheme in SCORING_SCHEMES:
        for region in (SOURCE_REGION, TARGET_REGION):
            name = f"{scheme}_{region}_label"
            expected = labels.get(name, np.full(int(aligned["n_bins"]), -1))
            if name not in dataset or not np.array_equal(
                np.asarray(dataset[name].values, dtype=int), np.asarray(expected, dtype=int)
            ):
                raise ValueError(f"dataset corrected labels {name!r} are mismatched.")
        name = f"{scheme}_match_rate_shuffle"
        if name not in dataset or not np.allclose(
            np.asarray(dataset[name].values, dtype=float),
            np.asarray(null_samples[scheme], dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"dataset shuffle samples for {scheme!r} are mismatched.")
        for metric_name in (
            "match_rate",
            "n_matching_bins",
            "n_valid_labeled_bins",
        ):
            column = f"{scheme}_{metric_name}"
            metric_dtype = float if metric_name == "match_rate" else int
            if column not in dataset or not np.allclose(
                np.asarray(dataset[column].values, dtype=metric_dtype),
                metrics.get(column, pd.Series(dtype=metric_dtype)).to_numpy(
                    dtype=metric_dtype
                ),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(f"dataset ripple metric {column!r} is mismatched.")


def validate_ripple_decoding_comparison_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate identity, provenance, decoded arrays, rescoring, and terminals."""
    if not isinstance(result, Mapping):
        raise TypeError("RippleDecodingComparison result must be a mapping.")
    metadata = _metadata(
        ripple_decoding_comparison_id=result["ripple_decoding_comparison_id"],
        animal_name=result["animal_name"],
        date=result["date"],
        train_epoch=result["train_epoch"],
        decode_epoch=result["decode_epoch"],
        representation=result["representation"],
    )
    raw_parameters = dict(result["parameters"])
    effective_names = (
        "decode_bin_size_s",
        "spatial_bin_size_cm",
        "tuning_smoothing_sigma_bins",
        "ca1_min_movement_rate_hz",
        "v1_min_movement_rate_hz",
        "n_shuffles",
        "shuffle_seed",
        "expected_detector_zscore_threshold",
        "require_speed_gated",
    )
    parameters = _effective_parameters(
        parameter_name=raw_parameters.get("parameter_name", "default"),
        parameter_sha256=raw_parameters.get("parameter_sha256"),
        output_rule_sha256=raw_parameters.get("output_rule_sha256"),
        **{name: raw_parameters[name] for name in effective_names},
    )
    if set(raw_parameters) != set(parameters):
        raise ValueError("parameters contains missing or unsupported fields.")
    provenance, provenance_json = _canonical_provenance(
        result["upstream_provenance"], parameters=parameters
    )
    geometry = dict(result["graph_geometry"])
    graph_hash = str(geometry.pop("graph_policy_sha256", ""))
    if graph_hash != _provenance_sha256(geometry):
        raise ValueError("graph geometry digest is stale or tampered.")
    geometry["graph_policy_sha256"] = graph_hash
    analysis_status = str(result["analysis_status"])
    if analysis_status not in ANALYSIS_STATUSES:
        raise ValueError(f"Unsupported analysis_status {analysis_status!r}.")
    artifact_origin = str(result.get("artifact_origin", "computed"))
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("artifact_origin must be computed or registered_existing.")
    legacy = dict(result.get("legacy_artifact_provenance", {}))
    if artifact_origin == "computed" and legacy:
        raise ValueError("Computed results cannot claim legacy provenance.")
    selected_units = _validate_selected_units(
        result["selected_units"],
        parameters=parameters,
        analysis_status=analysis_status,
    )
    ca1_decoded = _canonical_decoded(result["ca1_decoded"], region=SOURCE_REGION)
    v1_decoded = _canonical_decoded(result["v1_decoded"], region=TARGET_REGION)
    aligned = _canonical_aligned(
        _scientific_module().align_decoded_ripple_data(ca1_decoded, v1_decoded)
    )
    dataset = result["dataset"]
    ripple_table = pd.DataFrame(
        {
            "start_time": np.asarray(
                dataset["input_ripple_start_time_s"].values, dtype=float
            ),
            "end_time": np.asarray(
                dataset["input_ripple_end_time_s"].values, dtype=float
            ),
        }
    )
    normalized_ripples = _normalize_ripple_table(
        ripple_table, epoch=metadata["decode_epoch"]
    )
    if not np.array_equal(
        normalized_ripples.to_numpy(dtype=float), ripple_table.to_numpy(dtype=float)
    ):
        raise ValueError("Persisted ripple intervals are not in canonical order.")
    selected_hash = _ripple_hash(ripple_table)
    if str(result["selected_ripple_intervals_sha256"]) != selected_hash:
        raise ValueError("Persisted ripple bounds do not match their selection digest.")
    ripple_qc = result["ripple_qc"]
    if not isinstance(ripple_qc, pd.DataFrame) or tuple(
        ripple_qc.columns
    ) != RIPPLE_QC_COLUMNS:
        raise ValueError("ripple_qc does not match its canonical schema.")
    ripple_qc = ripple_qc.copy().reset_index(drop=True)
    if len(ripple_qc) != len(ripple_table):
        raise ValueError("ripple_qc must contain every exact input event.")
    metrics = result["ripple_metrics"]
    summary = result["epoch_summary"]
    if not isinstance(metrics, pd.DataFrame) or tuple(metrics.columns) != RIPPLE_METRIC_COLUMNS:
        raise ValueError("ripple_metrics does not match its canonical schema.")
    if not isinstance(summary, pd.DataFrame) or tuple(summary.columns) != EPOCH_SUMMARY_COLUMNS or len(summary) != 1:
        raise ValueError("epoch_summary does not match its canonical schema.")
    metrics = metrics.copy().reset_index(drop=True)
    summary = summary.copy().reset_index(drop=True)
    if str(summary.iloc[0]["analysis_status"]) != analysis_status:
        raise ValueError("epoch_summary analysis status differs from the result.")
    if analysis_status in FITTED_STATUSES:
        expected_metrics, expected_summary, null, labels = _metric_tables(
            aligned=aligned,
            metadata=metadata,
            geometry=geometry,
            n_events_input=len(ripple_table),
            n_shuffles=parameters["n_shuffles"],
            shuffle_seed=parameters["shuffle_seed"],
        )
        expected_status = "no_valid_metrics" if str(
            expected_summary.iloc[0]["analysis_status"]
        ) == "no_valid_metrics" else (
            "partial_valid" if aligned["n_ripples"] < len(ripple_table) else "valid"
        )
        expected_summary.loc[:, "analysis_status"] = expected_status
        if analysis_status != expected_status:
            raise ValueError("analysis_status differs from decoded/scored content.")
    else:
        if not metrics.empty:
            raise ValueError("Terminal results must have empty ripple_metrics.")
        expected_metrics = _empty_metric_table()
        expected_summary = _terminal_summary(
            metadata=metadata,
            n_events_input=len(ripple_table),
            status=analysis_status,
        )
        null = {
            scheme: np.full(parameters["n_shuffles"], np.nan, dtype=float)
            for scheme in SCORING_SCHEMES
        }
        labels = {}
    _assert_frame_equal(metrics, expected_metrics, name="ripple_metrics")
    _assert_frame_equal(summary, expected_summary, name="epoch_summary")
    expected_qc = _ripple_qc_table(
        ripple_table=ripple_table,
        metadata=metadata,
        ca1_decoded=ca1_decoded,
        v1_decoded=v1_decoded,
        aligned=aligned,
        terminal_status=(
            analysis_status
            if analysis_status not in FITTED_STATUSES + ("no_common_decoded_bins",)
            else None
        ),
    )
    _assert_frame_equal(ripple_qc, expected_qc, name="ripple_qc")
    _validate_dataset_identity(
        dataset,
        metadata=metadata,
        parameters=parameters,
        upstream_provenance_json=provenance_json,
        selected_hash=selected_hash,
        geometry=geometry,
        selected_units=selected_units,
        ripple_table=ripple_table,
        ripple_qc=ripple_qc,
        metrics=metrics,
        summary=summary,
        aligned=aligned,
        labels=labels,
        null_samples=null,
        analysis_status=analysis_status,
        artifact_origin=artifact_origin,
        legacy_artifact_provenance=legacy,
    )
    ca1_units = selected_units["region"].eq(SOURCE_REGION)
    v1_units = selected_units["region"].eq(TARGET_REGION)
    return {
        **metadata,
        "parameters": parameters,
        "upstream_provenance": provenance,
        "selected_ripple_intervals_sha256": selected_hash,
        "graph_geometry": geometry,
        "graph_policy_sha256": graph_hash,
        "selected_units": selected_units,
        "ripple_qc": ripple_qc,
        "ripple_metrics": metrics,
        "epoch_summary": summary,
        "ca1_decoded": ca1_decoded,
        "v1_decoded": v1_decoded,
        "dataset": dataset,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance": legacy,
        "n_ripple_events_input": int(len(ripple_table)),
        "n_ripples": int(aligned["n_ripples"]),
        "n_ripple_bins": int(aligned["n_bins"]),
        "n_ca1_units": int(ca1_units.sum()),
        "n_v1_units": int(v1_units.sum()),
        "n_ca1_units_in_decoder": int(
            selected_units.loc[ca1_units, "included_in_decoder"].sum()
        ),
        "n_v1_units_in_decoder": int(
            selected_units.loc[v1_units, "included_in_decoder"].sum()
        ),
        "selected_units_sha256": _table_sha256(selected_units),
        "ripple_qc_sha256": _table_sha256(ripple_qc),
        "ripple_metrics_sha256": _table_sha256(metrics),
        "epoch_summary_sha256": _table_sha256(summary),
    }


def _decoded_npz_payload(
    decoded: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    region: str,
    selected_stable_unit_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return the pickle-free canonical decoded NPZ payload."""
    canonical = _canonical_decoded(decoded, region=region)
    return {
        "schema_version": np.asarray(RESULT_SCHEMA_VERSION),
        "region": np.asarray(region),
        "ripple_decoding_comparison_id": np.asarray(
            metadata["ripple_decoding_comparison_id"]
        ),
        "animal_name": np.asarray(metadata["animal_name"]),
        "date": np.asarray(metadata["date"]),
        "train_epoch": np.asarray(metadata["train_epoch"]),
        "decode_epoch": np.asarray(metadata["decode_epoch"]),
        "representation": np.asarray(metadata["representation"]),
        "decoded_state": canonical["decoded_state"],
        "bin_times_s": canonical["bin_times_s"],
        "ripple_ids": canonical["ripple_ids"],
        "ripple_start_times_s": canonical["ripple_start_times_s"],
        "ripple_end_times_s": canonical["ripple_end_times_s"],
        "ripple_source_indices": canonical["ripple_source_indices"],
        "skipped_ripples_json": np.asarray(
            json.dumps(canonical["skipped_ripples"], sort_keys=True)
        ),
        "selected_stable_unit_ids": np.asarray(
            list(selected_stable_unit_ids), dtype=str
        ),
        "time_unit": np.asarray("s"),
        "time_reference": np.asarray("augmented_nwb_ephys_timestamps"),
    }


def _write_decoded_npz(
    path: Path,
    decoded: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
    region: str,
    selected_stable_unit_ids: Sequence[str],
) -> None:
    """Write one pickle-free regional decoded-ripple NPZ."""
    np.savez_compressed(
        path,
        **_decoded_npz_payload(
            decoded,
            metadata=metadata,
            region=region,
            selected_stable_unit_ids=selected_stable_unit_ids,
        ),
    )


def _load_decoded_npz(
    path: Path,
    *,
    metadata: Mapping[str, str],
    region: str,
    expected_stable_unit_ids: Sequence[str],
) -> dict[str, Any]:
    """Load and semantically validate one canonical regional decoded NPZ."""
    expected_keys = set(
        _decoded_npz_payload(
            _empty_decoded(),
            metadata=metadata,
            region=region,
            selected_stable_unit_ids=expected_stable_unit_ids,
        )
    )
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"{region} decoded NPZ does not have the canonical schema.")
        scalar_expected = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "region": region,
            **metadata,
            "time_unit": "s",
            "time_reference": "augmented_nwb_ephys_timestamps",
        }
        for name, expected in scalar_expected.items():
            if str(np.asarray(archive[name]).item()) != str(expected):
                raise ValueError(f"{region} decoded NPZ has mismatched {name!r}.")
        stable = np.asarray(archive["selected_stable_unit_ids"]).astype(str)
        if not np.array_equal(stable, np.asarray(expected_stable_unit_ids, dtype=str)):
            raise ValueError(f"{region} decoded NPZ unit selection is mismatched.")
        try:
            skipped = json.loads(str(np.asarray(archive["skipped_ripples_json"]).item()))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{region} decoded NPZ has malformed skipped-ripple JSON.") from exc
        decoded = {
            name: np.asarray(archive[name]).copy()
            for name in (
                "decoded_state",
                "bin_times_s",
                "ripple_ids",
                "ripple_start_times_s",
                "ripple_end_times_s",
                "ripple_source_indices",
            )
        }
    decoded["skipped_ripples"] = skipped
    return _canonical_decoded(decoded, region=region)


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return values repeated on every manifest artifact row."""
    parameters = result["parameters"]
    return {
        "ripple_decoding_comparison_id": result["ripple_decoding_comparison_id"],
        "animal_name": result["animal_name"],
        "date": result["date"],
        "train_epoch": result["train_epoch"],
        "decode_epoch": result["decode_epoch"],
        "representation": result["representation"],
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "graph_policy_sha256": result["graph_policy_sha256"],
        "upstream_provenance_json": json.dumps(
            result["upstream_provenance"], sort_keys=True, separators=(",", ":")
        ),
        "selected_ripple_intervals_sha256": result[
            "selected_ripple_intervals_sha256"
        ],
        "n_ripple_events_input": result["n_ripple_events_input"],
        "n_ripples": result["n_ripples"],
        "n_ripple_bins": result["n_ripple_bins"],
        "n_ca1_units": result["n_ca1_units"],
        "n_v1_units": result["n_v1_units"],
        "n_ca1_units_in_decoder": result["n_ca1_units_in_decoder"],
        "n_v1_units_in_decoder": result["n_v1_units_in_decoder"],
        "selected_units_sha256": result["selected_units_sha256"],
        "ripple_qc_sha256": result["ripple_qc_sha256"],
        "ripple_metrics_sha256": result["ripple_metrics_sha256"],
        "epoch_summary_sha256": result["epoch_summary_sha256"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": json.dumps(
            result["legacy_artifact_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def _load_dataset(path: Path) -> Any:
    """Eagerly load and close one NetCDF dataset."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def write_ripple_decoding_comparison_artifact(
    result: Mapping[str, Any], path: Path, *, overwrite: bool = False
) -> dict[str, Path]:
    """Atomically write, checksum, and reload one complete seven-file bundle."""
    validated = validate_ripple_decoding_comparison_result(result)
    destination = Path(path)
    if destination.name != validated["ripple_decoding_comparison_id"]:
        raise ValueError(
            "Artifact directory name must equal ripple_decoding_comparison_id."
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite RippleDecodingComparison: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    metadata = {name: validated[name] for name in (
        "ripple_decoding_comparison_id",
        "animal_name",
        "date",
        "train_epoch",
        "decode_epoch",
        "representation",
    )}
    try:
        validated["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME, index=False
        )
        validated["ripple_qc"].to_parquet(temporary / RIPPLE_QC_FILENAME, index=False)
        validated["ripple_metrics"].to_parquet(
            temporary / RIPPLE_METRICS_FILENAME, index=False
        )
        validated["epoch_summary"].to_parquet(
            temporary / EPOCH_SUMMARY_FILENAME, index=False
        )
        validated["dataset"].to_netcdf(temporary / RESULT_FILENAME)
        for region, filename in (
            (SOURCE_REGION, CA1_DECODED_FILENAME),
            (TARGET_REGION, V1_DECODED_FILENAME),
        ):
            selected = validated["selected_units"].loc[
                (validated["selected_units"]["region"] == region)
                & validated["selected_units"]["included_in_decoder"]
            ]
            _write_decoded_npz(
                temporary / filename,
                validated[f"{region}_decoded"],
                metadata=metadata,
                region=region,
                selected_stable_unit_ids=selected["stable_unit_id"].to_list(),
            )
        common = _manifest_common(validated)
        artifacts = (
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("ripple_qc", RIPPLE_QC_FILENAME, "parquet"),
            ("ripple_metrics", RIPPLE_METRICS_FILENAME, "parquet"),
            ("epoch_summary", EPOCH_SUMMARY_FILENAME, "parquet"),
            ("ripple_decoding_comparison", RESULT_FILENAME, "netcdf"),
            ("ca1_decoded", CA1_DECODED_FILENAME, "npz"),
            ("v1_decoded", V1_DECODED_FILENAME, "npz"),
        )
        rows = []
        for key, filename, kind in artifacts:
            artifact_path = temporary / filename
            rows.append(
                {
                    "artifact_key": key,
                    "relative_path": filename,
                    "artifact_kind": kind,
                    "file_size_bytes": artifact_path.stat().st_size,
                    "sha256": _file_sha256(artifact_path),
                    **common,
                }
            )
        pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_parquet(
            temporary / MANIFEST_FILENAME, index=False
        )
        load_ripple_decoding_comparison_artifact(
            temporary, _allow_temporary_name=True
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "artifact_dir": destination,
        "artifact_manifest_path": destination / MANIFEST_FILENAME,
        "selected_units_path": destination / SELECTED_UNITS_FILENAME,
        "ripple_qc_path": destination / RIPPLE_QC_FILENAME,
        "ripple_metrics_path": destination / RIPPLE_METRICS_FILENAME,
        "epoch_summary_path": destination / EPOCH_SUMMARY_FILENAME,
        "result_path": destination / RESULT_FILENAME,
        "ca1_decoded_path": destination / CA1_DECODED_FILENAME,
        "v1_decoded_path": destination / V1_DECODED_FILENAME,
    }


def load_ripple_decoding_comparison_artifact(
    path: Path, *, _allow_temporary_name: bool = False
) -> dict[str, Any]:
    """Checksum and semantically validate one complete artifact bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    expected = {
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "ripple_qc": (RIPPLE_QC_FILENAME, "parquet"),
        "ripple_metrics": (RIPPLE_METRICS_FILENAME, "parquet"),
        "epoch_summary": (EPOCH_SUMMARY_FILENAME, "parquet"),
        "ripple_decoding_comparison": (RESULT_FILENAME, "netcdf"),
        "ca1_decoded": (CA1_DECODED_FILENAME, "npz"),
        "v1_decoded": (V1_DECODED_FILENAME, "npz"),
    }
    if tuple(manifest.columns) != MANIFEST_COLUMNS or len(manifest) != len(expected):
        raise ValueError("Artifact manifest does not match its canonical schema.")
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("Artifact manifest lacks canonical files.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("Artifact manifest contains stale names or kinds.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
        if int(row["file_size_bytes"]) != artifact_path.stat().st_size or str(
            row["sha256"]
        ) != _file_sha256(artifact_path):
            raise ValueError(f"Artifact checksum mismatch: {artifact_path}")
    first = manifest.iloc[0]
    for name in MANIFEST_COLUMNS[5:]:
        if not manifest[name].astype(str).eq(str(first[name])).all():
            raise ValueError(f"Artifact manifest has inconsistent {name!r}.")
    result_id = str(first["ripple_decoding_comparison_id"])
    if not _allow_temporary_name and directory.name != result_id:
        raise ValueError("Artifact directory name does not match result UUID.")
    dataset = _load_dataset(directory / RESULT_FILENAME)
    try:
        effective = json.loads(str(dataset.attrs["effective_parameters_json"]))
        provenance = json.loads(str(first["upstream_provenance_json"]))
        geometry = json.loads(str(dataset.attrs["graph_geometry_json"]))
        legacy = json.loads(str(first["legacy_artifact_provenance_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("Artifact bundle contains malformed provenance JSON.") from exc
    metadata = {
        name: str(first[name])
        for name in (
            "ripple_decoding_comparison_id",
            "animal_name",
            "date",
            "train_epoch",
            "decode_epoch",
            "representation",
        )
    }
    selected_units = pd.read_parquet(directory / SELECTED_UNITS_FILENAME)
    decoded = {}
    for region, filename in (
        (SOURCE_REGION, CA1_DECODED_FILENAME),
        (TARGET_REGION, V1_DECODED_FILENAME),
    ):
        stable = selected_units.loc[
            (selected_units["region"] == region)
            & selected_units["included_in_decoder"],
            "stable_unit_id",
        ].to_list()
        decoded[region] = _load_decoded_npz(
            directory / filename,
            metadata=metadata,
            region=region,
            expected_stable_unit_ids=stable,
        )
    validated = validate_ripple_decoding_comparison_result(
        {
            **metadata,
            "parameters": {
                "parameter_name": str(first["parameter_name"]),
                "parameter_sha256": str(first["parameter_sha256"]),
                "output_rule_sha256": str(first["output_rule_sha256"]),
                **effective,
            },
            "upstream_provenance": provenance,
            "selected_ripple_intervals_sha256": str(
                first["selected_ripple_intervals_sha256"]
            ),
            "graph_geometry": geometry,
            "selected_units": selected_units,
            "ripple_qc": pd.read_parquet(directory / RIPPLE_QC_FILENAME),
            "ripple_metrics": pd.read_parquet(directory / RIPPLE_METRICS_FILENAME),
            "epoch_summary": pd.read_parquet(directory / EPOCH_SUMMARY_FILENAME),
            "ca1_decoded": decoded[SOURCE_REGION],
            "v1_decoded": decoded[TARGET_REGION],
            "dataset": dataset,
            "analysis_status": str(first["analysis_status"]),
            "artifact_origin": str(first["artifact_origin"]),
            "legacy_artifact_provenance": legacy,
        }
    )
    integer_fields = (
        "n_ripple_events_input",
        "n_ripples",
        "n_ripple_bins",
        "n_ca1_units",
        "n_v1_units",
        "n_ca1_units_in_decoder",
        "n_v1_units_in_decoder",
    )
    if any(int(first[name]) != validated[name] for name in integer_fields):
        raise ValueError("Artifact manifest counts differ from persisted content.")
    for name in (
        "selected_units_sha256",
        "ripple_qc_sha256",
        "ripple_metrics_sha256",
        "epoch_summary_sha256",
        "graph_policy_sha256",
    ):
        if str(first[name]) != str(validated[name]):
            raise ValueError(f"Artifact manifest {name!r} is mismatched.")
    return {**validated, "manifest": manifest}


def _load_legacy_decoded(path: Path, *, region: str) -> dict[str, np.ndarray]:
    """Load one historical Pynapple Tsd NPZ for strict scientific comparison."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy {region} decoded artifact not found: {source}")
    import pynapple as nap

    decoded = nap.load_file(source)
    try:
        state = np.asarray(decoded.d, dtype=float).reshape(-1)
        times = np.asarray(decoded.t, dtype=float).reshape(-1)
        starts = np.asarray(decoded.time_support.start, dtype=float).reshape(-1)
        ends = np.asarray(decoded.time_support.end, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Legacy {region} decoded NPZ is not a Pynapple Tsd.") from exc
    if state.shape != times.shape or not np.all(np.isfinite(state)) or not np.all(
        np.isfinite(times)
    ):
        raise ValueError(f"Legacy {region} decoded NPZ is malformed.")
    return {"state": state, "times": times, "starts": starts, "ends": ends}


def _compare_legacy_decoded(
    observed: Mapping[str, np.ndarray],
    expected: Mapping[str, Any],
    *,
    region: str,
) -> None:
    """Require an old decoded Tsd to equal the graph-derived NWB redecode."""
    checks = {
        "state": np.asarray(expected["decoded_state"], dtype=float),
        "times": np.asarray(expected["bin_times_s"], dtype=float),
        "starts": np.asarray(expected["ripple_start_times_s"], dtype=float),
        "ends": np.asarray(expected["ripple_end_times_s"], dtype=float),
    }
    for name, values in checks.items():
        if np.asarray(observed[name]).shape != values.shape or not np.allclose(
            observed[name], values, rtol=2e-7, atol=2e-8, equal_nan=True
        ):
            raise ValueError(
                f"Legacy {region} decoded {name} differs from exact NWB redecode."
            )


def _legacy_representation(value: Any) -> str:
    """Normalize an old representation label to the canonical vocabulary."""
    text = str(value)
    if text in LEGACY_TO_REPRESENTATION:
        return LEGACY_TO_REPRESENTATION[text]
    if text in REPRESENTATIONS:
        return text
    raise ValueError(f"Unsupported legacy representation {text!r}.")


def _compare_legacy_tables(
    *,
    metrics_path: Path,
    summary_path: Path,
    expected_metrics: pd.DataFrame,
    expected_summary: pd.DataFrame,
    metadata: Mapping[str, str],
) -> None:
    """Compare every historical categorical score under corrected labels."""
    if not Path(metrics_path).is_file() or not Path(summary_path).is_file():
        raise FileNotFoundError("Complete legacy ripple metric and summary files are required.")
    observed_metrics = pd.read_parquet(metrics_path).copy()
    observed_summary = pd.read_parquet(summary_path).copy()
    required_metric_columns = (
        "representation",
        "train_epoch",
        "decode_epoch",
        "ripple_id",
        "ripple_source_index",
        "ripple_start_time_s",
        "ripple_end_time_s",
    ) + tuple(
        column
        for scheme in SCORING_SCHEMES
        for column in (
            f"{scheme}_scheme_requested",
            f"{scheme}_scheme_applicable",
            f"{scheme}_match_rate",
            f"{scheme}_n_matching_bins",
            f"{scheme}_n_valid_labeled_bins",
        )
    )
    required_summary_columns = (
        "representation",
        "train_epoch",
        "decode_epoch",
        "n_ripples",
        "n_ripple_bins",
        "n_ripple_events_input",
        "n_effective_shuffles",
    ) + tuple(
        column
        for scheme in SCORING_SCHEMES
        for column in (
            f"{scheme}_scheme_requested",
            f"{scheme}_scheme_applicable",
            f"{scheme}_scheme_reason",
            f"{scheme}_n_valid_ripples",
            f"{scheme}_match_rate",
            f"{scheme}_match_rate_shuffle_mean",
            f"{scheme}_match_rate_shuffle_sd",
            f"{scheme}_match_rate_p_value",
        )
    )
    missing_metrics = [name for name in required_metric_columns if name not in observed_metrics]
    missing_summary = [name for name in required_summary_columns if name not in observed_summary]
    if missing_metrics or missing_summary or len(observed_summary) != 1:
        raise ValueError("Legacy metric/summary files do not have the strict five-file schema.")
    observed_metrics["representation"] = observed_metrics["representation"].map(
        _legacy_representation
    )
    observed_summary["representation"] = observed_summary["representation"].map(
        _legacy_representation
    )
    for table in (observed_metrics, observed_summary):
        if not table["representation"].eq(metadata["representation"]).all() or not table[
            "train_epoch"
        ].astype(str).eq(metadata["train_epoch"]).all() or not table[
            "decode_epoch"
        ].astype(str).eq(metadata["decode_epoch"]).all():
            raise ValueError("Legacy artifact identity differs from the selected result.")
    _assert_frame_equal(
        observed_metrics.loc[:, list(required_metric_columns)],
        expected_metrics.loc[:, list(required_metric_columns)],
        name=(
            "legacy ripple metrics under corrected physical-arm graph scoring; "
            "old turn-group-labeled inbound-arm outputs are intentionally rejected"
        ),
    )
    _assert_frame_equal(
        observed_summary.loc[:, list(required_summary_columns)],
        expected_summary.loc[:, list(required_summary_columns)],
        name="legacy epoch summary under corrected graph scoring",
    )


def _resolve_legacy_unit_axis(
    values: Sequence[Any],
    *,
    resolver: Callable[[Sequence[Any]], Sequence[Mapping[str, Any]]],
    region: str,
    expected_units: pd.DataFrame,
) -> list[str]:
    """Resolve one historical unit coordinate independently by region."""
    if not callable(resolver):
        raise TypeError(f"{region}_legacy_identity_resolver must be callable.")
    raw = list(values)
    resolved = [dict(row) for row in resolver(raw)]
    if len(resolved) != len(raw):
        raise ValueError(f"{region} legacy identity resolver returned the wrong length.")
    stable = []
    for row in resolved:
        if "spikesorting_merge_id" not in row or "unit_id" not in row:
            raise ValueError(f"{region} resolver omitted persistent identity fields.")
        canonical = f"{row['spikesorting_merge_id']}:{row['unit_id']}"
        if str(row.get("stable_unit_id", canonical)) != canonical:
            raise ValueError(f"{region} resolver returned a noncanonical stable id.")
        stable.append(canonical)
    if len(set(stable)) != len(stable) or set(stable) != set(
        expected_units["stable_unit_id"].astype(str)
    ):
        raise ValueError(f"Legacy {region} unit axis differs from NWB input units.")
    return stable


def _compare_legacy_dataset(
    path: Path,
    *,
    expected: Any,
    expected_units: pd.DataFrame,
    metadata: Mapping[str, str],
    ca1_resolver: Callable[[Sequence[Any]], Sequence[Mapping[str, Any]]],
    v1_resolver: Callable[[Sequence[Any]], Sequence[Mapping[str, Any]]],
) -> None:
    """Require historical NetCDF arrays to equal corrected NWB rescoring."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy comparison NetCDF not found: {source_path}")
    observed = _load_dataset(source_path)
    if _legacy_representation(observed.attrs.get("representation", "")) != metadata[
        "representation"
    ] or str(observed.attrs.get("train_epoch", "")) != metadata["train_epoch"] or str(
        observed.attrs.get("decode_epoch", "")
    ) != metadata["decode_epoch"]:
        raise ValueError("Legacy NetCDF identity differs from the selected result.")
    region_resolvers = {SOURCE_REGION: ca1_resolver, TARGET_REGION: v1_resolver}
    for region in (SOURCE_REGION, TARGET_REGION):
        units = expected_units.loc[expected_units["region"] == region]
        stable = _resolve_legacy_unit_axis(
            np.asarray(observed.coords[f"{region}_unit"].values).tolist(),
            resolver=region_resolvers[region],
            region=region,
            expected_units=units,
        )
        expected_stable = np.asarray(expected.coords[f"{region}_unit"].values).astype(str)
        index = {value: offset for offset, value in enumerate(stable)}
        order = np.asarray([index[value] for value in expected_stable], dtype=int)
        for name in (f"{region}_movement_firing_rate_hz", f"{region}_keep_unit"):
            if name not in observed or not np.allclose(
                np.asarray(observed[name].values)[order],
                np.asarray(expected[name].values),
                rtol=2e-7,
                atol=2e-8,
                equal_nan=True,
            ):
                raise ValueError(f"Legacy NetCDF {region} unit audit differs from NWB.")
    applicable_schemes = (
        SCORING_SCHEMES
        if metadata["representation"] == "path_specific_place"
        else ("turn_group",)
    )
    variables = (
        "bin_time_s",
        "ripple_id",
        "ca1_decoded_state",
        "v1_decoded_state",
        "ripple_source_index",
        "ripple_start_time_s",
        "ripple_end_time_s",
    ) + tuple(
        name
        for scheme in applicable_schemes
        for name in (
            f"{scheme}_ca1_label",
            f"{scheme}_v1_label",
            f"{scheme}_bin_match",
            f"{scheme}_match_rate",
            f"{scheme}_n_matching_bins",
            f"{scheme}_n_valid_labeled_bins",
            f"{scheme}_match_rate_shuffle",
            f"{scheme}_match_rate_observed",
            f"{scheme}_match_rate_shuffle_mean",
            f"{scheme}_match_rate_shuffle_sd",
            f"{scheme}_match_rate_p_value",
        )
    )
    for name in variables:
        if name not in observed or name not in expected:
            raise ValueError(f"Legacy NetCDF lacks required scientific variable {name!r}.")
        if np.asarray(observed[name].values).shape != np.asarray(expected[name].values).shape or not np.allclose(
            observed[name].values,
            expected[name].values,
            rtol=2e-7,
            atol=2e-8,
            equal_nan=True,
        ):
            rationale = (
                " Correct graph-based physical-arm scoring intentionally rejects "
                "legacy place bundles that labeled inbound arms by turn group."
                if name.startswith("arm_identity")
                else ""
            )
            raise ValueError(f"Legacy NetCDF variable {name!r} differs from NWB reconstruction.{rationale}")


def register_existing_ripple_decoding_comparison_artifact(
    *,
    source_ca1_decoded_path: Path,
    source_v1_decoded_path: Path,
    source_ripple_metrics_path: Path,
    source_epoch_summary_path: Path,
    source_result_path: Path,
    destination_path: Path,
    ca1_legacy_identity_resolver: Callable[
        [Sequence[Any]], Sequence[Mapping[str, Any]]
    ],
    v1_legacy_identity_resolver: Callable[
        [Sequence[Any]], Sequence[Mapping[str, Any]]
    ],
    ca1_sorting_type: str,
    v1_sorting_type: str,
    source_v1ca1_git_commit: str | None = None,
    source_spyglass_git_commit: str | None = None,
    overwrite: bool = False,
    **compute_kwargs: Any,
) -> dict[str, Any]:
    """Strictly redecode/rescore NWB inputs before registering all five legacy files."""
    if str(ca1_sorting_type) != "ImportedSpikeSorting" or str(
        v1_sorting_type
    ) != "ImportedSpikeSorting":
        raise ValueError(
            "Legacy registration requires ImportedSpikeSorting for CA1 and V1."
        )
    sources = {
        "ca1_decoded": Path(source_ca1_decoded_path),
        "v1_decoded": Path(source_v1_decoded_path),
        "ripple_metrics": Path(source_ripple_metrics_path),
        "epoch_summary": Path(source_epoch_summary_path),
        "result": Path(source_result_path),
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Complete five-file legacy bundle is missing {missing!r}.")
    recomputed = compute_ripple_decoding_comparison(**compute_kwargs)
    if recomputed["analysis_status"] not in FITTED_STATUSES:
        raise ValueError("Legacy registration requires a recomputed fitted result.")
    for region in (SOURCE_REGION, TARGET_REGION):
        _compare_legacy_decoded(
            _load_legacy_decoded(sources[f"{region}_decoded"], region=region),
            recomputed[f"{region}_decoded"],
            region=region,
        )
    metadata = {name: recomputed[name] for name in (
        "ripple_decoding_comparison_id",
        "animal_name",
        "date",
        "train_epoch",
        "decode_epoch",
        "representation",
    )}
    _compare_legacy_tables(
        metrics_path=sources["ripple_metrics"],
        summary_path=sources["epoch_summary"],
        expected_metrics=recomputed["ripple_metrics"],
        expected_summary=recomputed["epoch_summary"],
        metadata=metadata,
    )
    _compare_legacy_dataset(
        sources["result"],
        expected=recomputed["dataset"],
        expected_units=recomputed["selected_units"],
        metadata=metadata,
        ca1_resolver=ca1_legacy_identity_resolver,
        v1_resolver=v1_legacy_identity_resolver,
    )
    legacy = {
        "registration_policy": (
            "exact_nwb_redecode_rescore_and_all_five_legacy_files_strict_equal"
        ),
        "corrected_arm_policy": (
            "physical_arm_from_graph_not_turn_group; stale inbound labels rejected"
        ),
        "ca1_sorting_type": str(ca1_sorting_type),
        "v1_sorting_type": str(v1_sorting_type),
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_sha256": {name: _file_sha256(path) for name, path in sources.items()},
        "source_v1ca1_git_commit": (
            "unknown" if source_v1ca1_git_commit is None else str(source_v1ca1_git_commit)
        ),
        "source_spyglass_git_commit": (
            "unknown" if source_spyglass_git_commit is None else str(source_spyglass_git_commit)
        ),
    }
    registered = dict(recomputed)
    registered["artifact_origin"] = "registered_existing"
    registered["legacy_artifact_provenance"] = legacy
    registered["dataset"] = recomputed["dataset"].copy(deep=True)
    registered["dataset"].attrs["artifact_origin"] = "registered_existing"
    registered["dataset"].attrs["legacy_artifact_provenance_json"] = json.dumps(
        legacy, sort_keys=True
    )
    registered = validate_ripple_decoding_comparison_result(registered)
    paths = write_ripple_decoding_comparison_artifact(
        registered, destination_path, overwrite=overwrite
    )
    return {**registered, **paths}


__all__ = [
    "ANALYSIS_STATUSES",
    "DEFAULT_ARTIFACT_ROOT",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "REPRESENTATIONS",
    "compute_ripple_decoding_comparison",
    "get_ripple_decoding_comparison_artifact_paths",
    "load_ripple_decoding_comparison_artifact",
    "prepare_ripple_decoding_comparison_event_selection",
    "register_existing_ripple_decoding_comparison_artifact",
    "validate_ripple_decoding_comparison_parameters",
    "validate_ripple_decoding_comparison_result",
    "write_ripple_decoding_comparison_artifact",
]
