"""Database-free nine-model motor-encoding artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "motor_encoding"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
NESTED_CV_FILENAME = "nested_cv.nc"
FULL_REFIT_FILENAME = "full_refit.nc"
FULL_W_CONFIGURATION_NAME = "full_w"

RESULT_SCHEMA_VERSION = "2"
NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_SELECTED_UNITS_TABLE_NAME = "motor_encoding_selected_units"
NWB_DATASET_INDEX_TABLE_NAME = "motor_encoding_dataset_index"
NWB_COORDINATES_TABLE_NAME = "motor_encoding_coordinates"
NWB_NESTED_CV_ARRAYS_TABLE_NAME = "motor_encoding_nested_cv_arrays"
NWB_FULL_REFIT_ARRAYS_TABLE_NAME = "motor_encoding_full_refit_arrays"
NWB_PROVENANCE_TABLE_NAME = "motor_encoding_provenance"

DEFAULT_BIN_SIZE_S = 0.05
DEFAULT_N_FOLDS = 5
DEFAULT_INNER_N_FOLDS = 3
DEFAULT_RANDOM_SEED = 0
DEFAULT_RIDGE_VALUES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
DEFAULT_SPATIAL_BIN_SIZES_CM = (2.0, 4.0, 8.0)
DEFAULT_MOTOR_FEATURE_MODE = "zscore"
DEFAULT_MOTOR_ZSCORE_EPS = 1e-12
DEFAULT_MOTOR_SPLINE_N_BASIS = 5
DEFAULT_MOTOR_SPLINE_ORDER = 4
DEFAULT_POSITION_SPLINE_ORDER = 4
DEFAULT_SPEED_SMOOTHING_SIGMA_S = 0.1
DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM = 15.0

MODEL_NAMES = (
    "motor",
    "motor_tp",
    "tp_only",
    "motor_place",
    "place_only",
    "motor_generalized_place",
    "generalized_place_only",
    "motor_generalized_task_progression",
    "generalized_task_progression_only",
)
MODEL_SPEC = MappingProxyType(
    {
        "motor": "strict motor covariates only",
        "motor_tp": (
            "motor covariates plus TP group offset and TP group-specific "
            "spline fields"
        ),
        "tp_only": "TP group offset plus TP group-specific spline fields",
        "motor_place": (
            "motor covariates plus trajectory offset and "
            "trajectory-specific place fields"
        ),
        "place_only": "trajectory offset plus trajectory-specific place fields",
        "motor_generalized_place": (
            "motor covariates plus one generalized full-W place spline field"
        ),
        "generalized_place_only": "one generalized full-W place spline field",
        "motor_generalized_task_progression": (
            "motor covariates plus one generalized task-progression spline field"
        ),
        "generalized_task_progression_only": (
            "one generalized task-progression spline field"
        ),
    }
)
OUTPUT_RULE = MappingProxyType(
    {
        "version": 2,
        "model_names": MODEL_NAMES,
        "cross_validation": "nested_lap_level_by_trajectory_movement_only",
        "hyperparameter_selection": (
            "per_model_population_median_unit_information_bits_per_spike"
        ),
        "unit_fit_failure_policy": (
            "retry_nonfinite_or_failed_population_units_independently"
        ),
        "movement_operator": "greater_than_or_equal",
        "stability_aggregation": "at_least_one_trajectory",
        "stability_operator": "greater_than_or_equal",
        "full_refit_role": "visualization_not_heldout_inference",
        "primary_position_role": "speed_acceleration_and_track_linearization",
        "orientation_reference_role": "primary_minus_reference_head_direction",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "position_unit": "cm",
    }
)
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_units",
    "no_eligible_units",
    "no_trials",
    "no_valid_position",
    "no_movement",
    "no_valid_units",
)
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "selection_index",
    "movement_firing_rate_hz",
    "passes_movement_firing_rate",
    *(f"{trajectory_type}_stability_correlation" for trajectory_type in TRAJECTORY_TYPES),
    "passes_stability",
    "eligible",
    "n_outer_folds_selected",
    "n_outer_folds_with_finite_evidence",
    "valid_nested_cv",
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "motor_encoding_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "parameter_name",
    "parameter_sha256",
    "model_spec_sha256",
    "output_rule_sha256",
    "n_units_input",
    "n_units_eligible",
    "n_units_valid",
    "n_outer_folds_expected",
    "n_outer_folds_valid",
    "selected_units_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
)

DATASET_INDEX_COLUMNS = (
    "dataset_key",
    "dataset_index",
    "dimensions_json",
    "sizes_json",
    "coordinate_names_json",
    "variable_names_json",
    "attrs_json",
)
ARRAY_COMPONENT_COLUMNS = (
    "dataset_key",
    "component_index",
    "component_name",
    "dimensions_json",
    "shape_json",
    "dtype",
    "numeric_count",
    "numeric_values",
    "text_values_json",
    "attrs_json",
)
PROVENANCE_COLUMNS = (
    "metadata_json",
    "parameters_json",
    "model_spec_json",
    "output_rule_json",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "artifact_schema_version",
    "result_schema_version",
)


def _provenance_sha256(value: Any) -> str:
    """Return the shared deterministic provenance digest."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(value)


MODEL_SPEC_SHA256 = _provenance_sha256(dict(MODEL_SPEC))
OUTPUT_RULE_SHA256 = _provenance_sha256(dict(OUTPUT_RULE))


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


def get_motor_encoding_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    motor_encoding_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first motor artifact bundle."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "region": region,
        }.items()
    }
    result_id = _uuid_string(
        motor_encoding_id,
        name="motor_encoding_id",
    )
    artifact_dir = (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["region"]
        / result_id
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "selected_units_path": artifact_dir / SELECTED_UNITS_FILENAME,
        "nested_cv_path": artifact_dir / NESTED_CV_FILENAME,
        "full_refit_path": artifact_dir / FULL_REFIT_FILENAME,
    }


def validate_motor_encoding_parameters(
    *,
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
    evaluation_bin_size_s: float = DEFAULT_BIN_SIZE_S,
    outer_n_folds: int = DEFAULT_N_FOLDS,
    inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ridge_values: Sequence[float] = DEFAULT_RIDGE_VALUES,
    spatial_bin_sizes_cm: Sequence[float] = DEFAULT_SPATIAL_BIN_SIZES_CM,
    motor_feature_mode: str = DEFAULT_MOTOR_FEATURE_MODE,
    motor_zscore_eps: float = DEFAULT_MOTOR_ZSCORE_EPS,
    motor_spline_n_basis: int = DEFAULT_MOTOR_SPLINE_N_BASIS,
    motor_spline_order: int = DEFAULT_MOTOR_SPLINE_ORDER,
    position_spline_order: int = DEFAULT_POSITION_SPLINE_ORDER,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    generalized_place_branch_gap_cm: float = (
        DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM
    ),
) -> dict[str, Any]:
    """Return validated parameters for the fixed nine-model comparison."""
    integers = {
        "outer_n_folds": outer_n_folds,
        "inner_n_folds": inner_n_folds,
        "random_seed": random_seed,
        "motor_spline_n_basis": motor_spline_n_basis,
        "motor_spline_order": motor_spline_order,
        "position_spline_order": position_spline_order,
    }
    for name, value in integers.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        integers[name] = int(value)
    if integers["outer_n_folds"] < 2 or integers["inner_n_folds"] < 2:
        raise ValueError(
            "outer_n_folds and inner_n_folds must each be at least 2."
        )
    if integers["random_seed"] < 0:
        raise ValueError("random_seed must be non-negative.")
    for name in (
        "motor_spline_n_basis",
        "motor_spline_order",
        "position_spline_order",
    ):
        if integers[name] < 1:
            raise ValueError(f"{name} must be positive.")

    numeric = {
        "minimum_movement_firing_rate_hz": float(
            minimum_movement_firing_rate_hz
        ),
        "minimum_stability_correlation": float(
            minimum_stability_correlation
        ),
        "evaluation_bin_size_s": float(evaluation_bin_size_s),
        "motor_zscore_eps": float(motor_zscore_eps),
        "speed_smoothing_sigma_s": float(speed_smoothing_sigma_s),
        "generalized_place_branch_gap_cm": float(
            generalized_place_branch_gap_cm
        ),
    }
    if not np.isfinite(numeric["minimum_movement_firing_rate_hz"]) or (
        numeric["minimum_movement_firing_rate_hz"] < 0.0
    ):
        raise ValueError(
            "minimum_movement_firing_rate_hz must be finite and non-negative."
        )
    if not np.isfinite(numeric["minimum_stability_correlation"]) or not (
        -1.0 <= numeric["minimum_stability_correlation"] <= 1.0
    ):
        raise ValueError(
            "minimum_stability_correlation must be finite and within [-1, 1]."
        )
    for name in ("evaluation_bin_size_s", "motor_zscore_eps"):
        if not np.isfinite(numeric[name]) or numeric[name] <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
    for name in (
        "speed_smoothing_sigma_s",
        "generalized_place_branch_gap_cm",
    ):
        if not np.isfinite(numeric[name]) or numeric[name] < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    mode = str(motor_feature_mode)
    if mode not in {"zscore", "spline"}:
        raise ValueError("motor_feature_mode must be 'zscore' or 'spline'.")
    ridges = tuple(float(value) for value in ridge_values)
    spatial_bins = tuple(float(value) for value in spatial_bin_sizes_cm)
    if not ridges or len(ridges) != len(set(ridges)) or any(
        not np.isfinite(value) or value < 0.0 for value in ridges
    ):
        raise ValueError("ridge_values must be unique, finite, and non-negative.")
    if not spatial_bins or len(spatial_bins) != len(set(spatial_bins)) or any(
        not np.isfinite(value) or value <= 0.0 for value in spatial_bins
    ):
        raise ValueError(
            "spatial_bin_sizes_cm must be unique, positive, and finite."
        )
    return {
        **numeric,
        **integers,
        "ridge_values": ridges,
        "spatial_bin_sizes_cm": spatial_bins,
        "motor_feature_mode": mode,
    }


MANUSCRIPT_PARAMETERS_BY_REGION = MappingProxyType(
    {
        "v1": MappingProxyType(
            validate_motor_encoding_parameters(
                minimum_movement_firing_rate_hz=0.5,
                minimum_stability_correlation=0.5,
            )
        ),
        "ca1": MappingProxyType(
            validate_motor_encoding_parameters(
                minimum_movement_firing_rate_hz=0.0,
                minimum_stability_correlation=0.5,
            )
        ),
    }
)


def _motor_module() -> Any:
    """Import the existing fit implementation lazily."""
    from v1ca1.task_progression import motor

    if tuple(motor.MODEL_NAMES) != MODEL_NAMES:
        raise RuntimeError("The fixed nine-model motor specification has changed.")
    return motor


def _graph_length_cm(graph_inputs: Mapping[str, Any]) -> float:
    """Return an ordered graph length using the shared linearization helper."""
    from v1ca1.spyglass.stability import _ordered_graph_length

    if graph_inputs.get("coordinate_unit") != "cm":
        raise ValueError("WTrackGraph coordinate_unit must be 'cm'.")
    track_kwargs = dict(graph_inputs.get("track_graph_kwargs", {}))
    linear_kwargs = dict(graph_inputs.get("linearization_kwargs", {}))
    if set(track_kwargs) != {"node_positions", "edges"}:
        raise ValueError("WTrackGraph track_graph_kwargs are incomplete.")
    return _ordered_graph_length(
        np.asarray(track_kwargs["node_positions"], dtype=float),
        linear_kwargs.get("edge_order", ()),
        linear_kwargs.get("edge_spacing", ()),
    )


def build_graph_derived_position_basis_configs(
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    *,
    spatial_bin_sizes_cm: Sequence[float],
    spline_order: int,
    generalized_place_branch_gap_cm: float,
) -> list[dict[str, Any]]:
    """Build motor basis candidates only from selected NWB graph geometry."""
    expected = {*TRAJECTORY_TYPES, FULL_W_CONFIGURATION_NAME}
    actual = set(graph_inputs_by_configuration)
    if actual != expected:
        raise ValueError(
            "Graph inputs must contain the four trajectories and full_w; "
            f"missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}."
        )
    lengths: dict[str, float] = {}
    for configuration_name in (*TRAJECTORY_TYPES, FULL_W_CONFIGURATION_NAME):
        graph = graph_inputs_by_configuration[configuration_name]
        if str(graph.get("configuration_name", "")) != configuration_name:
            raise ValueError(
                "WTrackGraph configuration_name does not match its mapping key."
            )
        lengths[configuration_name] = _graph_length_cm(graph)
    common_path_length = lengths[TRAJECTORY_TYPES[0]]
    if any(
        not np.isclose(
            lengths[trajectory], common_path_length, rtol=1e-10, atol=1e-12
        )
        for trajectory in TRAJECTORY_TYPES[1:]
    ):
        raise ValueError("The four directional path graphs must have equal length.")
    full_graph = graph_inputs_by_configuration[FULL_W_CONFIGURATION_NAME]
    full_spacing = np.asarray(
        dict(full_graph.get("linearization_kwargs", {})).get(
            "edge_spacing", ()
        ),
        dtype=float,
    )
    observed_gap = float(np.sum(full_spacing))
    if not np.isclose(
        observed_gap,
        float(generalized_place_branch_gap_cm),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "generalized_place_branch_gap_cm does not match the selected "
            f"full_w graph spacing ({observed_gap:g} cm)."
        )
    motor = _motor_module()
    return [
        motor.build_position_basis_config_from_lengths(
            trajectory_length_cm=common_path_length,
            generalized_place_length_cm=lengths[FULL_W_CONFIGURATION_NAME],
            spatial_bin_size_cm=float(spatial_bin_size),
            spline_order=int(spline_order),
            generalized_place_branch_gap_cm=float(
                generalized_place_branch_gap_cm
            ),
        )
        for spatial_bin_size in spatial_bin_sizes_cm
    ]


def _position_arrays(position: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned two-dimensional position values and timestamps."""
    values = np.asarray(getattr(position, "d", position), dtype=float)
    timestamps = np.asarray(getattr(position, "t", ()), dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_samples, 2).")
    if values.shape[0] != timestamps.size or timestamps.size < 2:
        raise ValueError(f"{name} values and timestamps must align.")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} timestamps must be finite and increasing.")
    return values, timestamps


def validate_position_pair(
    primary_position: Any,
    orientation_reference_position: Any,
) -> None:
    """Require exact timestamp alignment for motor-vector construction."""
    _primary_values, primary_times = _position_arrays(
        primary_position, name="primary_position"
    )
    _reference_values, reference_times = _position_arrays(
        orientation_reference_position,
        name="orientation_reference_position",
    )
    if not np.array_equal(primary_times, reference_times):
        raise ValueError(
            "Primary and orientation-reference position timestamps must match exactly."
        )


def build_motor_model_features(
    *,
    primary_position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
) -> dict[str, Any]:
    """Build graph-derived path, full-W, and generalized progression features."""
    from v1ca1.spyglass.dpp import _pool_interval_sets
    from v1ca1.spyglass.path_specific_place import _intersect_intervals
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    expected = set(TRAJECTORY_TYPES)
    if set(trajectory_intervals_by_type) != expected:
        raise ValueError("trajectory_intervals_by_type must contain four paths.")
    progressions: dict[str, Any] = {}
    path_lengths: dict[str, float] = {}
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    support_by_trajectory = {
        trajectory: _intersect_intervals(
            trajectory_intervals_by_type[trajectory], movement_intervals
        )
        for trajectory in TRAJECTORY_TYPES
    }
    for trajectory in TRAJECTORY_TYPES:
        progression, length = build_task_progression_from_graph(
            position=primary_position,
            trajectory_interval=trajectory_intervals_by_type[trajectory],
            graph_inputs=graph_inputs_by_configuration[trajectory],
            trajectory_type=trajectory,
        )
        progressions[trajectory] = progression
        path_lengths[trajectory] = float(length)
        restricted = progression.restrict(support_by_trajectory[trajectory])
        time_chunks.append(np.asarray(restricted.t, dtype=float))
        value_chunks.append(np.asarray(restricted.d, dtype=float))
    common_path_length = path_lengths[TRAJECTORY_TYPES[0]]
    if any(
        not np.isclose(path_lengths[name], common_path_length)
        for name in TRAJECTORY_TYPES[1:]
    ):
        raise ValueError("The four directional path graphs must have equal length.")
    pooled_support = _pool_interval_sets(support_by_trajectory)
    if not time_chunks or not any(chunk.size for chunk in time_chunks):
        raise ValueError("No movement-supported trajectory samples are available.")
    times = np.concatenate(time_chunks)
    values = np.concatenate(value_chunks)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Trajectory feature timestamps must be unique.")

    full_progression, full_length = build_task_progression_from_graph(
        position=primary_position,
        trajectory_interval=movement_intervals,
        graph_inputs=graph_inputs_by_configuration[FULL_W_CONFIGURATION_NAME],
        trajectory_type=FULL_W_CONFIGURATION_NAME,
    )
    import pynapple as nap

    generalized_task_progression = nap.Tsd(
        t=times,
        d=values,
        time_support=movement_intervals,
        time_units="s",
    )
    generalized_place_position = nap.Tsd(
        t=np.asarray(full_progression.t, dtype=float),
        d=np.asarray(full_progression.d, dtype=float) * float(full_length),
        time_support=movement_intervals,
        time_units="s",
    )
    return {
        "task_progression_by_trajectory": progressions,
        "generalized_task_progression": generalized_task_progression,
        "generalized_place_position": generalized_place_position,
        "support_by_trajectory": support_by_trajectory,
        "pooled_support": pooled_support,
        "common_path_length_cm": common_path_length,
        "full_w_length_cm": float(full_length),
    }


def _selected_identity_table(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return stable unit identities aligned to the current TsGroup."""
    from v1ca1.spyglass.movement import _stable_identity_rows

    rows = _stable_identity_rows(spikes, stable_unit_ids)
    return pd.DataFrame.from_records(rows, columns=IDENTITY_COLUMNS)


def _aligned_movement_rates(
    table: pd.DataFrame,
    *,
    identity: pd.DataFrame,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    movement_intervals: Any,
) -> tuple[str, np.ndarray]:
    """Return the canonical movement status and rates in TsGroup order."""
    from v1ca1.spyglass.movement import validate_movement_firing_rate_table

    observed = validate_movement_firing_rate_table(table).copy()
    if identity.empty:
        if not observed.empty:
            raise ValueError("No selected units require an empty movement table.")
        return "no_units", np.asarray([], dtype=float)
    for field, expected in {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
    }.items():
        if observed[field].astype(str).unique().tolist() != [str(expected)]:
            raise ValueError(f"MovementFiringRate has mismatched {field}.")
    observed["stable_unit_id"] = observed["stable_unit_id"].astype(str)
    if observed["stable_unit_id"].duplicated().any():
        raise ValueError("MovementFiringRate stable_unit_id values must be unique.")
    observed = observed.set_index("stable_unit_id", drop=False)
    stable_ids = identity["stable_unit_id"].astype(str).tolist()
    if set(observed.index) != set(stable_ids):
        raise ValueError("MovementFiringRate identities do not match selected units.")
    observed = observed.loc[stable_ids]
    for field in ("spikesorting_merge_id", "unit_id"):
        if observed[field].astype(str).tolist() != identity[field].astype(str).tolist():
            raise ValueError(f"MovementFiringRate {field} does not match units.")
    statuses = observed["firing_rate_status"].astype(str).unique().tolist()
    if len(statuses) != 1 or statuses[0] not in {
        "valid",
        "no_valid_position",
        "no_movement",
    }:
        raise ValueError("MovementFiringRate has an unsupported mixed status.")
    status = statuses[0]
    duration = float(movement_intervals.tot_length())
    observed_duration = pd.to_numeric(
        observed["movement_duration_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.allclose(observed_duration, duration, rtol=1e-9, atol=1e-12):
        raise ValueError("MovementFiringRate duration does not match intervals.")
    rates = pd.to_numeric(
        observed["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if status == "valid" and (
        not np.all(np.isfinite(rates)) or np.any(rates < 0.0)
    ):
        raise ValueError("Movement firing rates must be finite and non-negative.")
    if status != "valid" and not np.all(np.isnan(rates)):
        raise ValueError(f"{status} movement firing rates must be NaN.")
    return status, rates


def _build_unit_eligibility_table(
    *,
    identity: pd.DataFrame,
    movement_firing_rates_hz: np.ndarray,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
) -> pd.DataFrame:
    """Align movement and four path-stability inputs to the selected units."""
    rates = np.asarray(movement_firing_rates_hz, dtype=float).reshape(-1)
    if rates.size != len(identity):
        raise ValueError("Movement firing rates do not align to selected units.")
    actual_trajectories = set(stability_tables_by_trajectory)
    expected_trajectories = set(TRAJECTORY_TYPES)
    if actual_trajectories != expected_trajectories:
        raise ValueError(
            "stability_tables_by_trajectory must contain exactly the four "
            "trajectory types; "
            f"missing={sorted(expected_trajectories - actual_trajectories)!r}, "
            f"extra={sorted(actual_trajectories - expected_trajectories)!r}."
        )

    output = identity.copy()
    output["movement_firing_rate_hz"] = rates
    stability_arrays: list[np.ndarray] = []
    expected_ids = identity["stable_unit_id"].astype(str).tolist()
    for trajectory_type in TRAJECTORY_TYPES:
        table = stability_tables_by_trajectory[trajectory_type]
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"Stability[{trajectory_type}] must be one pandas DataFrame."
            )
        if identity.empty:
            if not table.empty:
                raise ValueError(
                    "No selected units require empty stability tables."
                )
            correlations = np.asarray([], dtype=float)
        else:
            required = {
                "spikesorting_merge_id",
                "unit_id",
                "stable_unit_id",
                "animal_name",
                "date",
                "region",
                "epoch",
                "trajectory_type",
                "stability_correlation",
            }
            missing = sorted(required.difference(table.columns))
            if missing:
                raise ValueError(
                    f"Stability[{trajectory_type}] is missing columns "
                    f"{missing!r}."
                )
            observed = table.copy()
            for field_name, expected_value in {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
            }.items():
                values = observed[field_name].astype(str).unique().tolist()
                if values != [str(expected_value)]:
                    raise ValueError(
                        f"Stability[{trajectory_type}] has mismatched "
                        f"{field_name}."
                    )
            for field_name in (
                "spikesorting_merge_id",
                "unit_id",
                "stable_unit_id",
            ):
                observed[field_name] = observed[field_name].astype(str)
            if observed["stable_unit_id"].duplicated().any():
                raise ValueError(
                    f"Stability[{trajectory_type}] has duplicate stable units."
                )
            observed = observed.set_index("stable_unit_id", drop=False)
            if set(observed.index) != set(expected_ids):
                raise ValueError(
                    f"Stability[{trajectory_type}] identities do not exactly "
                    "match the selected units."
                )
            observed = observed.loc[expected_ids]
            for field_name in ("spikesorting_merge_id", "unit_id"):
                if observed[field_name].tolist() != identity[
                    field_name
                ].astype(str).tolist():
                    raise ValueError(
                        f"Stability[{trajectory_type}] {field_name} does not "
                        "match the selected units."
                    )
            correlations = pd.to_numeric(
                observed["stability_correlation"],
                errors="coerce",
            ).to_numpy(dtype=float)
            finite = np.isfinite(correlations)
            if np.any(np.isinf(correlations)) or np.any(
                (correlations[finite] < -1.0 - 1e-9)
                | (correlations[finite] > 1.0 + 1e-9)
            ):
                raise ValueError(
                    "Stability correlations must be within [-1, 1] or NaN."
                )
            if "stability_status" in observed:
                statuses = observed["stability_status"].astype(str).to_numpy()
                if not np.array_equal(finite, statuses == "valid"):
                    raise ValueError(
                        "Finite stability correlations must correspond exactly "
                        "to stability_status='valid'."
                    )
            if "firing_rate_hz" in observed:
                stability_rates = pd.to_numeric(
                    observed["firing_rate_hz"],
                    errors="coerce",
                ).to_numpy(dtype=float)
                if not np.allclose(
                    stability_rates,
                    rates,
                    rtol=1e-9,
                    atol=1e-12,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"Stability[{trajectory_type}] firing rates disagree "
                        "with MovementFiringRate."
                    )
        output[f"{trajectory_type}_stability_correlation"] = correlations
        stability_arrays.append(correlations)

    if identity.empty:
        passes_stability = np.asarray([], dtype=bool)
    else:
        stability_matrix = np.column_stack(stability_arrays)
        passes_stability = np.any(
            np.isfinite(stability_matrix)
            & (
                stability_matrix
                >= float(minimum_stability_correlation)
            ),
            axis=1,
        )
    passes_movement = (
        np.isfinite(rates)
        & (rates >= float(minimum_movement_firing_rate_hz))
    )
    output["passes_movement_firing_rate"] = passes_movement
    output["passes_stability"] = passes_stability
    output["eligible"] = passes_movement & passes_stability
    return output


def _parameter_metadata(
    *,
    parameter_name: str,
    parameter_sha256: str | None,
    model_spec_sha256: str | None,
    output_rule_sha256: str | None,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return validated parameter and fixed-rule provenance."""
    name = str(parameter_name).strip()
    if not name or len(name) > 64:
        raise ValueError(
            "parameter_name must be non-empty and at most 64 characters."
        )
    expected_parameter_sha256 = _provenance_sha256(
        {"motor_encoding_param_name": name, **dict(parameters)}
    )
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    if str(parameter_sha256) != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    if model_spec_sha256 is None:
        model_spec_sha256 = MODEL_SPEC_SHA256
    if str(model_spec_sha256) != MODEL_SPEC_SHA256:
        raise ValueError("model_spec_sha256 does not match the fixed model spec.")
    if output_rule_sha256 is None:
        output_rule_sha256 = OUTPUT_RULE_SHA256
    if str(output_rule_sha256) != OUTPUT_RULE_SHA256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": name,
        "parameter_sha256": str(parameter_sha256),
        "model_spec_sha256": str(model_spec_sha256),
        "output_rule_sha256": str(output_rule_sha256),
        **dict(parameters),
    }


def _unit_index_by_group(identity: pd.DataFrame) -> dict[str, int]:
    """Return a string-normalized ephemeral-unit lookup."""
    keys = identity["group_unit_id"].map(str).tolist()
    if len(keys) != len(set(keys)):
        raise ValueError("String-normalized group unit identifiers must be unique.")
    return {key: index for index, key in enumerate(keys)}


def _build_selected_units_table(
    *,
    eligibility: pd.DataFrame,
    nested_cv: Any,
    full_refit: Any,
) -> pd.DataFrame:
    """Build the all-input unit audit for eligibility and held-out validity."""
    identity = eligibility.loc[:, list(IDENTITY_COLUMNS)]
    n_units = len(eligibility)
    nested_units = [str(value) for value in np.asarray(nested_cv.unit.values)]
    group_lookup = _unit_index_by_group(identity)
    if len(nested_units) != n_units or set(nested_units) != set(group_lookup):
        raise ValueError("Nested-CV units do not match all selected input units.")
    nested_order = np.asarray([group_lookup[value] for value in nested_units], dtype=int)
    if not np.array_equal(nested_order, np.arange(n_units)):
        raise ValueError("Nested-CV units must preserve selected TsGroup order.")

    eligible = eligibility["eligible"].to_numpy(dtype=bool)
    full_units = {str(value) for value in np.asarray(full_refit.unit.values)}
    expected_full_units = {
        str(identity.iloc[index]["group_unit_id"])
        for index in np.flatnonzero(eligible)
    }
    if full_units != expected_full_units:
        raise ValueError(
            "Full-refit units do not match the epoch-wide movement-rate filter."
        )
    selected = np.asarray(nested_cv["outer_unit_selected"].values, dtype=bool)
    evidence = np.asarray(
        nested_cv["outer_info_bits_per_spike"].values,
        dtype=float,
    )
    if selected.shape != (int(nested_cv.sizes["outer_fold"]), n_units):
        raise ValueError("outer_unit_selected has an unexpected shape.")
    if evidence.shape != (
        int(nested_cv.sizes["outer_fold"]),
        len(MODEL_NAMES),
        n_units,
    ):
        raise ValueError("outer_info_bits_per_spike has an unexpected shape.")
    finite_by_fold = selected & np.all(np.isfinite(evidence), axis=1)
    pooled = np.asarray(
        nested_cv["pooled_info_bits_per_spike"].values,
        dtype=float,
    )
    pooled_spikes = np.asarray(
        nested_cv["pooled_spike_sum"].values,
        dtype=float,
    )
    valid = (
        eligible
        & (pooled_spikes > 0.0)
        & np.all(np.isfinite(pooled), axis=0)
        & np.any(finite_by_fold, axis=0)
    )
    output = eligibility.copy()
    for field in IDENTITY_COLUMNS:
        output[field] = output[field].map(str)
    output["selection_index"] = np.arange(n_units, dtype=int)
    output["n_outer_folds_selected"] = np.sum(selected, axis=0).astype(int)
    output["n_outer_folds_with_finite_evidence"] = np.sum(
        finite_by_fold, axis=0
    ).astype(int)
    output["valid_nested_cv"] = valid
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _analysis_status(selected_units: pd.DataFrame) -> str:
    """Return the fixed all-unit audit status."""
    n_eligible = int(selected_units["eligible"].sum())
    n_valid = int(selected_units["valid_nested_cv"].sum())
    if n_valid == 0:
        return "no_valid_units"
    return "valid" if n_valid == n_eligible else "partial_valid"


def _canonicalize_computed_dataset(
    dataset: Any,
    *,
    role: str,
    selected_units: pd.DataFrame,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    primary_position_source: str,
    orientation_reference_position_source: str,
    artifact_origin: str,
) -> Any:
    """Attach persistent unit coordinates and canonical result provenance."""
    group_to_row = {
        str(row["group_unit_id"]): row
        for row in selected_units.to_dict("records")
    }
    group_ids = [str(value) for value in np.asarray(dataset.unit.values)]
    if len(group_ids) != len(set(group_ids)) or any(
        group_id not in group_to_row for group_id in group_ids
    ):
        raise ValueError("Dataset unit coordinates do not map to selected units.")
    rows = [group_to_row[group_id] for group_id in group_ids]
    stable_ids = np.asarray([str(row["stable_unit_id"]) for row in rows], dtype=str)
    canonical = dataset.assign_coords(
        {
            "unit": ("unit", stable_ids),
            "spikesorting_merge_id": (
                "unit",
                np.asarray(
                    [str(row["spikesorting_merge_id"]) for row in rows],
                    dtype=str,
                ),
            ),
            "unit_id": (
                "unit",
                np.asarray([str(row["unit_id"]) for row in rows], dtype=str),
            ),
            "stable_unit_id": ("unit", stable_ids),
            "group_unit_id": ("unit", np.asarray(group_ids, dtype=str)),
        }
    )
    if role == "nested_cv":
        rate_lookup = {
            str(row["stable_unit_id"]): float(row["movement_firing_rate_hz"])
            for row in selected_units.to_dict("records")
        }
        canonical["movement_firing_rate_hz"] = (
            "unit",
            np.asarray([rate_lookup[value] for value in stable_ids], dtype=float),
        )
    parameter_json = json.dumps(
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in parameters.items()
            if key not in {
                "parameter_name",
                "parameter_sha256",
                "model_spec_sha256",
                "output_rule_sha256",
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    attrs = dict(canonical.attrs)
    attrs.update(
        {
            **metadata,
            "artifact_role": role,
            "parameter_name": str(parameters["parameter_name"]),
            "parameter_sha256": str(parameters["parameter_sha256"]),
            "model_spec_sha256": str(parameters["model_spec_sha256"]),
            "output_rule_sha256": str(parameters["output_rule_sha256"]),
            "effective_parameters_json": parameter_json,
            "primary_position_source": str(primary_position_source),
            "orientation_reference_position_source": str(
                orientation_reference_position_source
            ),
            "artifact_origin": str(artifact_origin),
        }
    )
    canonical.attrs = attrs
    return canonical


def _common_metadata(
    *,
    motor_encoding_id: Any,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> dict[str, str]:
    """Return validated result metadata."""
    return {
        "motor_encoding_id": _uuid_string(
            motor_encoding_id,
            name="motor_encoding_id",
        ),
        **{
            name: _path_component(value, name=name)
            for name, value in {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "epoch": epoch,
            }.items()
        },
    }


def _terminal_selected_units(
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Return an all-input audit for a terminal computation."""
    output = eligibility.copy()
    for field in IDENTITY_COLUMNS:
        output[field] = output[field].map(str)
    output["selection_index"] = np.arange(len(output), dtype=int)
    output["n_outer_folds_selected"] = np.zeros(len(output), dtype=int)
    output["n_outer_folds_with_finite_evidence"] = np.zeros(
        len(output), dtype=int
    )
    output["valid_nested_cv"] = np.zeros(len(output), dtype=bool)
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _terminal_datasets(
    *,
    selected_units: pd.DataFrame,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    primary_position_source: str,
    orientation_reference_position_source: str,
    analysis_status: str,
) -> tuple[Any, Any]:
    """Return canonical empty NetCDF datasets for one terminal status."""
    import xarray as xr

    group_ids = selected_units["group_unit_id"].astype(str).to_numpy()
    n_units = len(group_ids)
    nested = xr.Dataset(
        data_vars={
            "outer_unit_selected": (
                ("outer_fold", "unit"),
                np.zeros((0, n_units), dtype=np.int8),
            ),
            "outer_info_bits_per_spike": (
                ("outer_fold", "model", "unit"),
                np.empty((0, len(MODEL_NAMES), n_units), dtype=float),
            ),
            "pooled_info_bits_per_spike": (
                ("model", "unit"),
                np.full((len(MODEL_NAMES), n_units), np.nan, dtype=float),
            ),
            "pooled_spike_sum": ("unit", np.zeros(n_units, dtype=float)),
            "outer_train_bin_count": (
                "outer_fold",
                np.asarray([], dtype=int),
            ),
            "outer_test_bin_count": (
                "outer_fold",
                np.asarray([], dtype=int),
            ),
        },
        coords={
            "outer_fold": np.asarray([], dtype=int),
            "model": np.asarray(MODEL_NAMES, dtype=str),
            "unit": group_ids,
        },
        attrs={"fit_stage": "terminal", "analysis_status": analysis_status},
    )
    full_refit = xr.Dataset(
        data_vars={
            "selected_ridge": (
                "model",
                np.full(len(MODEL_NAMES), np.nan, dtype=float),
            ),
            "movement_firing_rate_hz": (
                "unit",
                np.asarray([], dtype=float),
            ),
        },
        coords={
            "model": np.asarray(MODEL_NAMES, dtype=str),
            "unit": np.asarray([], dtype=str),
        },
        attrs={"fit_stage": "terminal", "analysis_status": analysis_status},
    )
    nested = _canonicalize_computed_dataset(
        nested,
        role="nested_cv",
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        artifact_origin="computed",
    )
    full_refit = _canonicalize_computed_dataset(
        full_refit,
        role="full_refit",
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        artifact_origin="computed",
    )
    return nested, full_refit


def _terminal_result(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    eligibility: pd.DataFrame,
    analysis_status: str,
    primary_position_source: str,
    orientation_reference_position_source: str,
) -> dict[str, Any]:
    """Build one validated terminal artifact bundle."""
    if analysis_status not in ANALYSIS_STATUSES:
        raise ValueError("Unsupported terminal analysis status.")
    selected_units = _terminal_selected_units(
        eligibility,
    )
    nested_cv, full_refit = _terminal_datasets(
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        analysis_status=analysis_status,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    selected_units_sha256 = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    result = {
        "metadata": dict(metadata),
        "parameters": dict(parameters),
        "selected_units": selected_units,
        "nested_cv": nested_cv,
        "full_refit": full_refit,
        "n_units_input": len(selected_units),
        "n_units_eligible": int(
            selected_units["eligible"].sum()
        ),
        "n_units_valid": 0,
        "n_outer_folds_expected": int(parameters["outer_n_folds"]),
        "n_outer_folds_valid": 0,
        "selected_units_sha256": selected_units_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }
    return validate_motor_encoding_result(result)


def compute_motor_encoding(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    motor_encoding_id: Any,
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
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    model_spec_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    evaluation_bin_size_s: float = DEFAULT_BIN_SIZE_S,
    outer_n_folds: int = DEFAULT_N_FOLDS,
    inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ridge_values: Sequence[float] = DEFAULT_RIDGE_VALUES,
    spatial_bin_sizes_cm: Sequence[float] = DEFAULT_SPATIAL_BIN_SIZES_CM,
    motor_feature_mode: str = DEFAULT_MOTOR_FEATURE_MODE,
    motor_zscore_eps: float = DEFAULT_MOTOR_ZSCORE_EPS,
    motor_spline_n_basis: int = DEFAULT_MOTOR_SPLINE_N_BASIS,
    motor_spline_order: int = DEFAULT_MOTOR_SPLINE_ORDER,
    position_spline_order: int = DEFAULT_POSITION_SPLINE_ORDER,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    generalized_place_branch_gap_cm: float = (
        DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM
    ),
) -> dict[str, Any]:
    """Run the existing nested-CV motor implementation on selected NWB inputs."""
    metadata = _common_metadata(
        motor_encoding_id=motor_encoding_id,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
    )
    parameters = validate_motor_encoding_parameters(
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
        evaluation_bin_size_s=evaluation_bin_size_s,
        outer_n_folds=outer_n_folds,
        inner_n_folds=inner_n_folds,
        random_seed=random_seed,
        ridge_values=ridge_values,
        spatial_bin_sizes_cm=spatial_bin_sizes_cm,
        motor_feature_mode=motor_feature_mode,
        motor_zscore_eps=motor_zscore_eps,
        motor_spline_n_basis=motor_spline_n_basis,
        motor_spline_order=motor_spline_order,
        position_spline_order=position_spline_order,
        speed_smoothing_sigma_s=speed_smoothing_sigma_s,
        generalized_place_branch_gap_cm=generalized_place_branch_gap_cm,
    )
    parameters = _parameter_metadata(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        model_spec_sha256=model_spec_sha256,
        output_rule_sha256=output_rule_sha256,
        parameters=parameters,
    )
    for name, value in {
        "primary_position_source": primary_position_source,
        "orientation_reference_position_source": (
            orientation_reference_position_source
        ),
    }.items():
        if not str(value).strip():
            raise ValueError(f"{name} must be non-empty.")
    validate_position_pair(primary_position, orientation_reference_position)
    identity = _selected_identity_table(spikes, stable_unit_ids)
    movement_status, movement_rates = _aligned_movement_rates(
        movement_firing_rate_table,
        identity=identity,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        epoch=metadata["epoch"],
        movement_intervals=movement_intervals,
    )
    eligibility = _build_unit_eligibility_table(
        identity=identity,
        movement_firing_rates_hz=movement_rates,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        epoch=metadata["epoch"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
    )
    if movement_status in {"no_units", "no_valid_position", "no_movement"}:
        return _terminal_result(
            metadata=metadata,
            parameters=parameters,
            eligibility=eligibility,
            analysis_status=movement_status,
            primary_position_source=primary_position_source,
            orientation_reference_position_source=(
                orientation_reference_position_source
            ),
        )
    motor = _motor_module()
    full_refit_unit_mask = eligibility["eligible"].to_numpy(dtype=bool)
    if not np.any(full_refit_unit_mask):
        return _terminal_result(
            metadata=metadata,
            parameters=parameters,
            eligibility=eligibility,
            analysis_status="no_eligible_units",
            primary_position_source=primary_position_source,
            orientation_reference_position_source=(
                orientation_reference_position_source
            ),
        )
    feasible, reasons, _summary = motor.summarize_lap_cv_feasibility(
        dict(trajectory_intervals_by_type),
        n_folds=parameters["outer_n_folds"],
        inner_n_folds=parameters["inner_n_folds"],
    )
    if not feasible:
        return _terminal_result(
            metadata=metadata,
            parameters=parameters,
            eligibility=eligibility,
            analysis_status="no_trials",
            primary_position_source=primary_position_source,
            orientation_reference_position_source=(
                orientation_reference_position_source
            ),
        )
    basis_configs = build_graph_derived_position_basis_configs(
        graph_inputs_by_configuration,
        spatial_bin_sizes_cm=parameters["spatial_bin_sizes_cm"],
        spline_order=parameters["position_spline_order"],
        generalized_place_branch_gap_cm=parameters[
            "generalized_place_branch_gap_cm"
        ],
    )
    features = build_motor_model_features(
        primary_position=primary_position,
        trajectory_intervals_by_type=trajectory_intervals_by_type,
        graph_inputs_by_configuration=graph_inputs_by_configuration,
        movement_intervals=movement_intervals,
    )
    epoch_data = motor.prepare_motor_epoch_data(
        spikes=spikes,
        position_tsd=primary_position,
        body_position_tsd=orientation_reference_position,
        generalized_place_position=features["generalized_place_position"],
        generalized_task_progression=features["generalized_task_progression"],
        trajectory_intervals=dict(trajectory_intervals_by_type),
        task_progression_by_trajectory=features[
            "task_progression_by_trajectory"
        ],
        movement_interval=movement_intervals,
        bin_size_s=parameters["evaluation_bin_size_s"],
        speed_smoothing_sigma_s=parameters["speed_smoothing_sigma_s"],
    )
    if [str(value) for value in np.asarray(epoch_data["unit_ids"])] != identity[
        "group_unit_id"
    ].map(str).tolist():
        raise ValueError("Binned response units do not preserve selected-unit order.")
    outer_folds = motor.build_lap_cv_folds_for_epoch(
        dict(trajectory_intervals_by_type),
        n_folds=parameters["outer_n_folds"],
        seed=parameters["random_seed"],
    )
    nested_result = motor.run_nested_lap_cv(
        epoch_data,
        outer_folds,
        ridge_values=parameters["ridge_values"],
        position_basis_configs=basis_configs,
        inner_n_folds=parameters["inner_n_folds"],
        seed=parameters["random_seed"],
        min_firing_rate_hz=parameters["minimum_movement_firing_rate_hz"],
        motor_feature_mode=parameters["motor_feature_mode"],
        motor_zscore_eps=parameters["motor_zscore_eps"],
        motor_spline_k=parameters["motor_spline_n_basis"],
        motor_spline_order=parameters["motor_spline_order"],
        allowed_unit_mask=full_refit_unit_mask,
        isolate_unit_failures=True,
    )
    sources = {
        "primary_position_source": str(primary_position_source),
        "orientation_reference_position_source": str(
            orientation_reference_position_source
        ),
        "graph_configurations": [*TRAJECTORY_TYPES, FULL_W_CONFIGURATION_NAME],
    }
    fit_parameters = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in parameters.items()
        if key not in {
            "parameter_name",
            "parameter_sha256",
            "model_spec_sha256",
            "output_rule_sha256",
        }
    }
    nested_cv = motor.build_nested_cv_dataset(
        nested_result,
        outer_folds,
        epoch_data,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        epoch=metadata["epoch"],
        min_firing_rate_hz=parameters["minimum_movement_firing_rate_hz"],
        sources=sources,
        fit_parameters=fit_parameters,
    )
    full_ridge_cv = motor.compute_hyperparameter_cv_scores(
        epoch_data,
        outer_folds,
        unit_mask=full_refit_unit_mask,
        ridge_values=parameters["ridge_values"],
        position_basis_configs=basis_configs,
        motor_feature_mode=parameters["motor_feature_mode"],
        motor_zscore_eps=parameters["motor_zscore_eps"],
        motor_spline_k=parameters["motor_spline_n_basis"],
        motor_spline_order=parameters["motor_spline_order"],
        isolate_unit_failures=True,
    )
    ridge_by_model = {
        model_name: float(full_ridge_cv["selected_ridge"][model_index])
        for model_index, model_name in enumerate(MODEL_NAMES)
    }
    position_basis_by_model = {
        model_name: basis_configs[
            int(full_ridge_cv["selected_spatial_index"][model_index])
        ]
        for model_index, model_name in enumerate(MODEL_NAMES)
    }
    full_fit = motor.fit_full_refit_models(
        epoch_data,
        unit_mask=full_refit_unit_mask,
        ridge_by_model=ridge_by_model,
        position_basis_by_model=position_basis_by_model,
        motor_feature_mode=parameters["motor_feature_mode"],
        motor_zscore_eps=parameters["motor_zscore_eps"],
        motor_spline_k=parameters["motor_spline_n_basis"],
        motor_spline_order=parameters["motor_spline_order"],
        isolate_unit_failures=True,
    )
    full_refit = motor.build_full_refit_dataset(
        full_fit,
        full_ridge_cv,
        epoch_data,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        epoch=metadata["epoch"],
        movement_firing_rates=movement_rates,
        min_firing_rate_hz=parameters["minimum_movement_firing_rate_hz"],
        sources=sources,
        fit_parameters=fit_parameters,
    )
    selected_units = _build_selected_units_table(
        eligibility=eligibility,
        nested_cv=nested_cv,
        full_refit=full_refit,
    )
    nested_cv = _canonicalize_computed_dataset(
        nested_cv,
        role="nested_cv",
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        artifact_origin="computed",
    )
    full_refit = _canonicalize_computed_dataset(
        full_refit,
        role="full_refit",
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        artifact_origin="computed",
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    selected_units_sha256 = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    train_counts = np.asarray(nested_cv["outer_train_bin_count"], dtype=int)
    test_counts = np.asarray(nested_cv["outer_test_bin_count"], dtype=int)
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": selected_units,
        "nested_cv": nested_cv,
        "full_refit": full_refit,
        "n_units_input": len(selected_units),
        "n_units_eligible": int(
            selected_units["eligible"].sum()
        ),
        "n_units_valid": int(selected_units["valid_nested_cv"].sum()),
        "n_outer_folds_expected": parameters["outer_n_folds"],
        "n_outer_folds_valid": int(np.sum((train_counts > 0) & (test_counts > 0))),
        "selected_units_sha256": selected_units_sha256,
        "analysis_status": _analysis_status(selected_units),
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }
    return validate_motor_encoding_result(result)


def _validate_dataset(
    dataset: Any,
    *,
    role: str,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    artifact_origin: str,
) -> Any:
    """Validate one canonical nested-CV or full-refit dataset."""
    import xarray as xr

    if not isinstance(dataset, xr.Dataset):
        raise TypeError(f"{role} must be an xarray.Dataset.")
    if "model" not in dataset.coords or tuple(
        str(value) for value in np.asarray(dataset.model.values)
    ) != MODEL_NAMES:
        raise ValueError(f"{role} must contain the fixed nine-model coordinate.")
    if "unit" not in dataset.dims:
        raise ValueError(f"{role} must contain a unit dimension.")
    for coordinate in (*IDENTITY_COLUMNS,):
        if coordinate not in dataset.coords:
            raise ValueError(f"{role} is missing unit coordinate {coordinate!r}.")
        if dataset[coordinate].dims != ("unit",):
            raise ValueError(f"{role} {coordinate} must index the unit dimension.")
    stable_ids = [str(value) for value in np.asarray(dataset.stable_unit_id)]
    if [str(value) for value in np.asarray(dataset.unit)] != stable_ids:
        raise ValueError(f"{role} unit coordinate must equal stable_unit_id.")
    if len(stable_ids) != len(set(stable_ids)):
        raise ValueError(f"{role} stable unit identities must be unique.")
    required_variables = (
        {
            "outer_unit_selected",
            "outer_info_bits_per_spike",
            "pooled_info_bits_per_spike",
            "pooled_spike_sum",
            "outer_train_bin_count",
            "outer_test_bin_count",
            "movement_firing_rate_hz",
        }
        if role == "nested_cv"
        else {"selected_ridge", "movement_firing_rate_hz"}
    )
    missing = sorted(required_variables.difference(dataset.data_vars))
    if missing:
        raise ValueError(f"{role} is missing required variables {missing!r}.")
    for name, expected in {
        **metadata,
        "artifact_role": role,
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "model_spec_sha256": MODEL_SPEC_SHA256,
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        "artifact_origin": artifact_origin,
    }.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"{role} has stale attribute {name!r}.")
    try:
        stored_parameters = json.loads(
            str(dataset.attrs["effective_parameters_json"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} has invalid effective parameter metadata.") from exc
    expected_parameters = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in parameters.items()
        if key not in {
            "parameter_name",
            "parameter_sha256",
            "model_spec_sha256",
            "output_rule_sha256",
        }
    }
    if stored_parameters != expected_parameters:
        raise ValueError(f"{role} effective parameters are stale.")
    return dataset


def validate_motor_encoding_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one in-memory motor artifact bundle."""
    copied = dict(result)
    metadata = _common_metadata(**dict(copied["metadata"]))
    parameter_values = dict(copied["parameters"])
    effective = validate_motor_encoding_parameters(
        **{
            name: parameter_values[name]
            for name in (
                "minimum_movement_firing_rate_hz",
                "minimum_stability_correlation",
                "evaluation_bin_size_s",
                "outer_n_folds",
                "inner_n_folds",
                "random_seed",
                "ridge_values",
                "spatial_bin_sizes_cm",
                "motor_feature_mode",
                "motor_zscore_eps",
                "motor_spline_n_basis",
                "motor_spline_order",
                "position_spline_order",
                "speed_smoothing_sigma_s",
                "generalized_place_branch_gap_cm",
            )
        }
    )
    parameters = _parameter_metadata(
        parameter_name=parameter_values["parameter_name"],
        parameter_sha256=parameter_values["parameter_sha256"],
        model_spec_sha256=parameter_values["model_spec_sha256"],
        output_rule_sha256=parameter_values["output_rule_sha256"],
        parameters=effective,
    )
    selected_units = copied["selected_units"]
    if not isinstance(selected_units, pd.DataFrame) or list(
        selected_units.columns
    ) != list(SELECTED_UNIT_COLUMNS):
        raise ValueError("selected_units does not match the canonical schema.")
    if selected_units["stable_unit_id"].astype(str).duplicated().any():
        raise ValueError("selected_units stable identities must be unique.")
    if selected_units["selection_index"].tolist() != list(
        range(len(selected_units))
    ):
        raise ValueError("selected_units selection_index must be contiguous.")
    for field in IDENTITY_COLUMNS:
        if selected_units[field].map(str).str.len().eq(0).any():
            raise ValueError(f"selected_units {field} must be non-empty.")
    bool_columns = (
        "passes_movement_firing_rate",
        "passes_stability",
        "eligible",
        "valid_nested_cv",
    )
    if any(selected_units[name].isna().any() for name in bool_columns):
        raise ValueError("selected_units eligibility fields cannot be null.")
    if not np.array_equal(
        selected_units["eligible"].to_numpy(dtype=bool),
        selected_units["passes_movement_firing_rate"].to_numpy(dtype=bool)
        & selected_units["passes_stability"].to_numpy(dtype=bool),
    ):
        raise ValueError(
            "selected_units eligible must be movement-rate AND stability."
        )
    status = str(copied["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError("Unsupported analysis_status.")
    artifact_origin = str(copied["artifact_origin"])
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("Unsupported artifact_origin.")
    for role in ("nested_cv", "full_refit"):
        _validate_dataset(
            copied[role],
            role=role,
            metadata=metadata,
            parameters=parameters,
            artifact_origin=artifact_origin,
        )
    from v1ca1.spyglass.selection import unit_identity_sha256

    digest = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    if str(copied["selected_units_sha256"]) != digest:
        raise ValueError("selected_units_sha256 does not match selected_units.")
    expected_counts = {
        "n_units_input": len(selected_units),
        "n_units_eligible": int(
            selected_units["eligible"].sum()
        ),
        "n_units_valid": int(selected_units["valid_nested_cv"].sum()),
        "n_outer_folds_expected": parameters["outer_n_folds"],
    }
    for name, expected in expected_counts.items():
        if int(copied[name]) != int(expected):
            raise ValueError(f"{name} does not match the artifact bundle.")
    n_outer_valid = int(copied["n_outer_folds_valid"])
    if not 0 <= n_outer_valid <= parameters["outer_n_folds"]:
        raise ValueError("n_outer_folds_valid is outside the expected range.")
    train_counts = np.asarray(
        copied["nested_cv"]["outer_train_bin_count"], dtype=int
    )
    test_counts = np.asarray(
        copied["nested_cv"]["outer_test_bin_count"], dtype=int
    )
    observed_outer_valid = int(np.sum((train_counts > 0) & (test_counts > 0)))
    if n_outer_valid != observed_outer_valid:
        raise ValueError(
            "n_outer_folds_valid does not match nested-CV bin counts."
        )
    nested_ids = [str(value) for value in np.asarray(copied["nested_cv"].unit)]
    if nested_ids != selected_units["stable_unit_id"].astype(str).tolist():
        raise ValueError("Nested-CV units do not match the all-input unit audit.")
    nested_rates = np.asarray(
        copied["nested_cv"]["movement_firing_rate_hz"], dtype=float
    )
    audit_rates = selected_units["movement_firing_rate_hz"].to_numpy(dtype=float)
    if not np.allclose(nested_rates, audit_rates, equal_nan=True):
        raise ValueError("Nested-CV movement rates do not match selected_units.")
    full_ids = {str(value) for value in np.asarray(copied["full_refit"].unit)}
    eligible_ids = set(
        selected_units.loc[
            selected_units["eligible"], "stable_unit_id"
        ].astype(str)
    )
    if status in {"valid", "partial_valid", "no_valid_units"} and (
        full_ids != eligible_ids
    ):
        raise ValueError("Full-refit units do not match eligible unit identities.")
    if status not in {"valid", "partial_valid", "no_valid_units"} and full_ids:
        raise ValueError("Terminal artifacts cannot contain full-refit units.")
    if full_ids:
        rate_by_id = dict(
            zip(
                selected_units["stable_unit_id"].astype(str),
                audit_rates,
                strict=True,
            )
        )
        full_order = [str(value) for value in np.asarray(copied["full_refit"].unit)]
        expected_full_rates = np.asarray(
            [rate_by_id[value] for value in full_order], dtype=float
        )
        if not np.allclose(
            np.asarray(
                copied["full_refit"]["movement_firing_rate_hz"], dtype=float
            ),
            expected_full_rates,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "Full-refit movement rates do not match selected_units."
            )
    if status == "valid" and expected_counts["n_units_valid"] != expected_counts[
        "n_units_eligible"
    ]:
        raise ValueError("valid status requires every eligible unit to be valid.")
    if status == "partial_valid" and not (
        0 < expected_counts["n_units_valid"] < expected_counts["n_units_eligible"]
    ):
        raise ValueError("partial_valid status has inconsistent unit counts.")
    if status in {
        "no_units",
        "no_eligible_units",
        "no_trials",
        "no_valid_position",
        "no_movement",
        "no_valid_units",
    } and expected_counts["n_units_valid"] != 0:
        raise ValueError(f"{status} artifacts cannot contain valid units.")
    copied.update(
        {
            "metadata": metadata,
            "parameters": parameters,
            "selected_units": selected_units.copy(),
            "selected_units_sha256": digest,
            "analysis_status": status,
            "artifact_origin": artifact_origin,
        }
    )
    return copied


def _json_ready(value: Any) -> Any:
    """Return nested metadata using JSON-native scalar types."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(current) for key, current in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(current) for current in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _nwb_column_description(name: str) -> str:
    """Return a compact description for one scratch-table column."""
    return name.replace("_", " ") + "."


def _empty_nwb_dynamic_table(
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    ragged_columns: Sequence[str] = (),
) -> Any:
    """Construct a typed zero-row DynamicTable without row inference."""
    from hdmf.common import DynamicTable, VectorData, VectorIndex

    output_columns = []
    for column in columns:
        if column in ragged_columns:
            data = VectorData(
                name=column,
                description=_nwb_column_description(column),
                data=np.asarray([], dtype=float),
            )
            output_columns.extend(
                (
                    data,
                    VectorIndex(
                        name=f"{column}_index",
                        data=np.asarray([], dtype=np.int64),
                        target=data,
                    ),
                )
            )
            continue
        if column in text_columns:
            values = np.asarray([], dtype="S1")
        elif column in integer_columns:
            values = np.asarray([], dtype=np.int64)
        elif column in boolean_columns:
            values = np.asarray([], dtype=bool)
        else:
            values = np.asarray([], dtype=float)
        output_columns.append(
            VectorData(
                name=column,
                description=_nwb_column_description(column),
                data=values,
            )
        )
    return DynamicTable(
        name=name,
        description=description,
        columns=output_columns,
    )


def _normalize_nwb_frame(
    table: pd.DataFrame,
    *,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    vector_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return one canonical typed MotorEncoding scratch-table frame."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != tuple(
        columns
    ):
        raise ValueError(
            "MotorEncoding NWB table does not have its canonical schema."
        )
    output = table.copy().reset_index(drop=True)
    for column in text_columns:
        output[column] = output[column].map(str)
    for column in integer_columns:
        output[column] = pd.to_numeric(
            output[column], errors="raise"
        ).astype(np.int64)
    for column in boolean_columns:
        values = output[column].tolist()
        if not all(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(
                f"MotorEncoding NWB column {column!r} must be boolean."
            )
        output[column] = np.asarray(values, dtype=bool)
    for column in vector_columns:
        vectors = [np.asarray(value, dtype=float) for value in output[column]]
        if any(vector.ndim != 1 or np.isinf(vector).any() for vector in vectors):
            raise ValueError(
                f"MotorEncoding NWB vector column {column!r} is invalid."
            )
        output[column] = vectors
    for column in columns:
        if column in (
            *text_columns,
            *integer_columns,
            *boolean_columns,
            *vector_columns,
        ):
            continue
        output[column] = pd.to_numeric(
            output[column], errors="raise"
        ).astype(float)
        if np.isinf(output[column].to_numpy(dtype=float)).any():
            raise ValueError(
                f"MotorEncoding NWB numeric column {column!r} contains infinity."
            )
    return output.loc[:, list(columns)]


def _dynamic_table_from_frame(
    table: pd.DataFrame,
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> Any:
    """Convert one scalar frame to an NWB DynamicTable."""
    from hdmf.common import DynamicTable

    canonical = _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
    )
    if canonical.empty:
        return _empty_nwb_dynamic_table(
            name=name,
            description=description,
            columns=columns,
            text_columns=text_columns,
            integer_columns=integer_columns,
            boolean_columns=boolean_columns,
        )
    return DynamicTable.from_dataframe(
        name=name,
        df=canonical,
        table_description=description,
        columns=[
            {"name": column, "description": _nwb_column_description(column)}
            for column in columns
        ],
    )


def _ragged_dynamic_table_from_frame(
    table: pd.DataFrame,
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    vector_columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> Any:
    """Convert scalar keys and numeric vectors to an NWB DynamicTable."""
    from hdmf.common import DynamicTable

    canonical = _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
        vector_columns=vector_columns,
    )
    if canonical.empty:
        return _empty_nwb_dynamic_table(
            name=name,
            description=description,
            columns=columns,
            text_columns=text_columns,
            integer_columns=integer_columns,
            boolean_columns=boolean_columns,
            ragged_columns=vector_columns,
        )
    scalar_columns = tuple(
        column for column in columns if column not in vector_columns
    )
    output = DynamicTable.from_dataframe(
        name=name,
        df=canonical.loc[:, list(scalar_columns)],
        table_description=description,
        columns=[
            {"name": column, "description": _nwb_column_description(column)}
            for column in scalar_columns
        ],
    )
    for column in vector_columns:
        vectors = canonical[column].tolist()
        if all(vector.size == 0 for vector in vectors):
            vectors = [np.asarray([np.nan], dtype=float) for _ in vectors]
        output.add_column(
            name=column,
            description=_nwb_column_description(column),
            data=vectors,
            index=True,
        )
    return output


def _decode_nwb_text(value: Any) -> str:
    """Return text after an HDF5-backed DynamicTable round trip."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _frame_from_dynamic_table(
    nwb_table: Any,
    *,
    expected_name: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    vector_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Load one DynamicTable or Spyglass-fetched DataFrame."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table.copy()
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != expected_name:
            raise ValueError(
                f"Unexpected MotorEncoding NWB object {nwb_table.name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("MotorEncoding NWB objects must be DynamicTables.")
    table = table.reset_index(drop=True)
    observed = tuple(str(column) for column in table.columns)
    if set(observed) != set(columns) or len(observed) != len(columns):
        raise ValueError("MotorEncoding NWB object has a noncanonical schema.")
    table = table.loc[:, list(columns)]
    for column in text_columns:
        table[column] = table[column].map(_decode_nwb_text)
    for column in vector_columns:
        table[column] = [
            np.asarray(value, dtype=float) for value in table[column]
        ]
    return _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
        vector_columns=vector_columns,
    )


def _dataset_records(result: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return the nested-CV and full-refit datasets in storage order."""
    canonical = validate_motor_encoding_result(result)
    return (
        ("nested_cv", canonical["nested_cv"]),
        ("full_refit", canonical["full_refit"]),
    )


def _dataset_index_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return ordered structure and attributes for both xarray datasets."""
    rows = []
    for dataset_index, (dataset_key, dataset) in enumerate(
        _dataset_records(result)
    ):
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset_index": dataset_index,
                "dimensions_json": json.dumps(
                    list(dataset.dims), separators=(",", ":")
                ),
                "sizes_json": json.dumps(
                    [int(dataset.sizes[name]) for name in dataset.dims],
                    separators=(",", ":"),
                ),
                "coordinate_names_json": json.dumps(
                    [str(name) for name in dataset.coords],
                    separators=(",", ":"),
                ),
                "variable_names_json": json.dumps(
                    [str(name) for name in dataset.data_vars],
                    separators=(",", ":"),
                ),
                "attrs_json": json.dumps(
                    _json_ready(dict(dataset.attrs)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=DATASET_INDEX_COLUMNS)


def _array_component_record(
    *,
    dataset_key: str,
    component_index: int,
    component_name: str,
    array: Any,
) -> dict[str, Any]:
    """Flatten one xarray coordinate or numeric variable without data JSON."""
    values = np.asarray(array.values)
    kind = values.dtype.kind
    if kind in {"O", "S", "U"}:
        flattened = [_decode_nwb_text(value) for value in values.reshape(-1)]
        dtype = "str"
        numeric = np.asarray([], dtype=float)
        text_json = json.dumps(flattened, separators=(",", ":"))
    elif kind in {"b", "i", "u", "f"}:
        numeric = values.astype(float, copy=False).reshape(-1)
        if np.isinf(numeric).any():
            raise ValueError(
                f"MotorEncoding array {component_name!r} contains infinity."
            )
        dtype = str(values.dtype)
        text_json = "[]"
    else:
        raise TypeError(
            f"MotorEncoding array {component_name!r} has unsupported dtype "
            f"{values.dtype}."
        )
    return {
        "dataset_key": dataset_key,
        "component_index": int(component_index),
        "component_name": component_name,
        "dimensions_json": json.dumps(list(array.dims), separators=(",", ":")),
        "shape_json": json.dumps(list(values.shape), separators=(",", ":")),
        "dtype": dtype,
        "numeric_count": int(numeric.size),
        "numeric_values": np.asarray(numeric, dtype=float),
        "text_values_json": text_json,
        "attrs_json": json.dumps(
            _json_ready(dict(array.attrs)),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _coordinates_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return every xarray coordinate as one ordered ragged row."""
    rows = []
    for dataset_key, dataset in _dataset_records(result):
        rows.extend(
            _array_component_record(
                dataset_key=dataset_key,
                component_index=index,
                component_name=str(name),
                array=coordinate,
            )
            for index, (name, coordinate) in enumerate(dataset.coords.items())
        )
    return pd.DataFrame.from_records(rows, columns=ARRAY_COMPONENT_COLUMNS)


def _variables_frame(result: Mapping[str, Any], *, dataset_key: str) -> pd.DataFrame:
    """Return every numeric variable for one dataset as an ordered ragged row."""
    datasets = dict(_dataset_records(result))
    if dataset_key not in datasets:
        raise ValueError(f"Unknown MotorEncoding dataset key {dataset_key!r}.")
    dataset = datasets[dataset_key]
    rows = []
    for index, (name, variable) in enumerate(dataset.data_vars.items()):
        if np.asarray(variable.values).dtype.kind not in {"b", "i", "u", "f"}:
            raise TypeError(
                f"MotorEncoding data variable {name!r} must be numeric."
            )
        rows.append(
            _array_component_record(
                dataset_key=dataset_key,
                component_index=index,
                component_name=str(name),
                array=variable,
            )
        )
    return pd.DataFrame.from_records(rows, columns=ARRAY_COMPONENT_COLUMNS)


def _provenance_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return detached result metadata and fixed analysis definitions."""
    canonical = validate_motor_encoding_result(result)
    return pd.DataFrame.from_records(
        [
            {
                "metadata_json": json.dumps(
                    _json_ready(canonical["metadata"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "parameters_json": json.dumps(
                    _json_ready(canonical["parameters"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "model_spec_json": json.dumps(
                    _json_ready(dict(MODEL_SPEC)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "output_rule_json": json.dumps(
                    _json_ready(dict(OUTPUT_RULE)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "analysis_status": str(canonical["analysis_status"]),
                "artifact_origin": str(canonical["artifact_origin"]),
                "legacy_artifact_provenance_json": json.dumps(
                    _json_ready(canonical["legacy_artifact_provenance"] or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
                "result_schema_version": RESULT_SCHEMA_VERSION,
            }
        ],
        columns=PROVENANCE_COLUMNS,
    )


def motor_encoding_result_to_nwb_objects(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one MotorEncoding result to six NWB scratch tables."""
    canonical = validate_motor_encoding_result(result)
    selected_integer = (
        "selection_index",
        "n_outer_folds_selected",
        "n_outer_folds_with_finite_evidence",
    )
    selected_boolean = (
        "passes_movement_firing_rate",
        "passes_stability",
        "eligible",
        "valid_nested_cv",
    )
    component_text = (
        "dataset_key",
        "component_name",
        "dimensions_json",
        "shape_json",
        "dtype",
        "text_values_json",
        "attrs_json",
    )
    return {
        "selected_units": _dynamic_table_from_frame(
            canonical["selected_units"],
            name=NWB_SELECTED_UNITS_TABLE_NAME,
            description=(
                "All-unit identity, eligibility, and nested-CV audit for "
                f"MotorEncoding NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
            ),
            columns=SELECTED_UNIT_COLUMNS,
            text_columns=IDENTITY_COLUMNS,
            integer_columns=selected_integer,
            boolean_columns=selected_boolean,
        ),
        "dataset_index": _dynamic_table_from_frame(
            _dataset_index_frame(canonical),
            name=NWB_DATASET_INDEX_TABLE_NAME,
            description="Ordered nested-CV and full-refit xarray dataset index.",
            columns=DATASET_INDEX_COLUMNS,
            text_columns=tuple(
                column
                for column in DATASET_INDEX_COLUMNS
                if column != "dataset_index"
            ),
            integer_columns=("dataset_index",),
        ),
        "coordinates": _ragged_dynamic_table_from_frame(
            _coordinates_frame(canonical),
            name=NWB_COORDINATES_TABLE_NAME,
            description="Coordinates for both MotorEncoding xarray datasets.",
            columns=ARRAY_COMPONENT_COLUMNS,
            vector_columns=("numeric_values",),
            text_columns=component_text,
            integer_columns=("component_index", "numeric_count"),
        ),
        "nested_cv_arrays": _ragged_dynamic_table_from_frame(
            _variables_frame(canonical, dataset_key="nested_cv"),
            name=NWB_NESTED_CV_ARRAYS_TABLE_NAME,
            description="Complete nested-CV data variables for MotorEncoding.",
            columns=ARRAY_COMPONENT_COLUMNS,
            vector_columns=("numeric_values",),
            text_columns=component_text,
            integer_columns=("component_index", "numeric_count"),
        ),
        "full_refit_arrays": _ragged_dynamic_table_from_frame(
            _variables_frame(canonical, dataset_key="full_refit"),
            name=NWB_FULL_REFIT_ARRAYS_TABLE_NAME,
            description="Complete full-refit data variables for MotorEncoding.",
            columns=ARRAY_COMPONENT_COLUMNS,
            vector_columns=("numeric_values",),
            text_columns=component_text,
            integer_columns=("component_index", "numeric_count"),
        ),
        "provenance": _dynamic_table_from_frame(
            _provenance_frame(canonical),
            name=NWB_PROVENANCE_TABLE_NAME,
            description=(
                "Detached MotorEncoding identity, parameters, definitions, "
                "status, and legacy provenance."
            ),
            columns=PROVENANCE_COLUMNS,
            text_columns=PROVENANCE_COLUMNS,
        ),
    }


def _parse_json(value: str, *, name: str, expected_type: type) -> Any:
    """Parse one JSON value with a field-specific error."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MotorEncoding {name} contains malformed JSON.") from exc
    if not isinstance(decoded, expected_type):
        raise ValueError(
            f"MotorEncoding {name} must encode {expected_type.__name__}."
        )
    return decoded


def _decode_array_component(
    record: Mapping[str, Any],
) -> tuple[tuple[str, ...], Any, dict[str, Any]]:
    """Restore one xarray coordinate or variable from a flattened row."""
    dimensions = tuple(
        str(value)
        for value in _parse_json(
            str(record["dimensions_json"]),
            name="array dimensions",
            expected_type=list,
        )
    )
    try:
        shape = tuple(
            int(value)
            for value in _parse_json(
                str(record["shape_json"]),
                name="array shape",
                expected_type=list,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("MotorEncoding array shape is malformed.") from exc
    if len(dimensions) != len(shape) or any(value < 0 for value in shape):
        raise ValueError("MotorEncoding array dimensions and shape disagree.")
    expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    dtype = str(record["dtype"])
    if dtype == "str":
        values = _parse_json(
            str(record["text_values_json"]),
            name="text array",
            expected_type=list,
        )
        if len(values) != expected_size:
            raise ValueError("MotorEncoding text array size disagrees with shape.")
        array = np.asarray([str(value) for value in values], dtype=str)
    else:
        count = int(record["numeric_count"])
        values = np.asarray(record["numeric_values"], dtype=float)[:count]
        if count != expected_size or values.size != expected_size:
            raise ValueError(
                "MotorEncoding numeric array size disagrees with its shape."
            )
        try:
            target_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError("MotorEncoding array dtype is unsupported.") from exc
        if target_dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("MotorEncoding numeric array dtype is unsupported.")
        array = values.astype(target_dtype)
    attrs = _parse_json(
        str(record["attrs_json"]),
        name="array attributes",
        expected_type=dict,
    )
    return dimensions, array.reshape(shape), attrs


def _ordered_component_records(
    table: pd.DataFrame,
    *,
    dataset_key: str,
) -> list[dict[str, Any]]:
    """Return uniquely named, contiguously indexed component records."""
    selected = table.loc[table["dataset_key"].astype(str) == dataset_key].copy()
    selected = selected.sort_values("component_index", kind="stable")
    if selected["component_name"].astype(str).duplicated().any():
        raise ValueError("MotorEncoding NWB component names must be unique.")
    if selected["component_index"].tolist() != list(range(len(selected))):
        raise ValueError("MotorEncoding NWB component indices must be contiguous.")
    return selected.to_dict("records")


def _dataset_from_nwb_frames(
    *,
    index_record: Mapping[str, Any],
    coordinates: pd.DataFrame,
    variables: pd.DataFrame,
) -> Any:
    """Rebuild one exact xarray Dataset from indexed component rows."""
    import xarray as xr

    dataset_key = str(index_record["dataset_key"])
    coordinate_records = _ordered_component_records(
        coordinates, dataset_key=dataset_key
    )
    variable_records = _ordered_component_records(
        variables, dataset_key=dataset_key
    )
    coordinate_names = _parse_json(
        str(index_record["coordinate_names_json"]),
        name="coordinate names",
        expected_type=list,
    )
    variable_names = _parse_json(
        str(index_record["variable_names_json"]),
        name="variable names",
        expected_type=list,
    )
    if [str(record["component_name"]) for record in coordinate_records] != [
        str(value) for value in coordinate_names
    ]:
        raise ValueError("MotorEncoding coordinate order disagrees with its index.")
    if [str(record["component_name"]) for record in variable_records] != [
        str(value) for value in variable_names
    ]:
        raise ValueError("MotorEncoding variable order disagrees with its index.")
    coords = {}
    coord_attrs = {}
    for record in coordinate_records:
        name = str(record["component_name"])
        dims, values, attrs = _decode_array_component(record)
        coords[name] = (dims, values)
        coord_attrs[name] = attrs
    data_vars = {}
    variable_attrs = {}
    for record in variable_records:
        name = str(record["component_name"])
        dims, values, attrs = _decode_array_component(record)
        data_vars[name] = (dims, values)
        variable_attrs[name] = attrs
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=_parse_json(
            str(index_record["attrs_json"]),
            name="dataset attributes",
            expected_type=dict,
        ),
    )
    for name, attrs in coord_attrs.items():
        dataset.coords[name].attrs.update(attrs)
    for name, attrs in variable_attrs.items():
        dataset[name].attrs.update(attrs)
    dimensions = [
        str(value)
        for value in _parse_json(
            str(index_record["dimensions_json"]),
            name="dataset dimensions",
            expected_type=list,
        )
    ]
    sizes = [
        int(value)
        for value in _parse_json(
            str(index_record["sizes_json"]),
            name="dataset sizes",
            expected_type=list,
        )
    ]
    if dimensions != list(dataset.dims) or sizes != [
        int(dataset.sizes[name]) for name in dataset.dims
    ]:
        raise ValueError("MotorEncoding dataset dimensions disagree with its index.")
    return dataset


def motor_encoding_result_from_nwb_objects(
    *,
    selected_units: Any,
    dataset_index: Any,
    coordinates: Any,
    nested_cv_arrays: Any,
    full_refit_arrays: Any,
    provenance: Any,
) -> dict[str, Any]:
    """Reconstruct and validate one result from six NWB scratch tables."""
    selected = _frame_from_dynamic_table(
        selected_units,
        expected_name=NWB_SELECTED_UNITS_TABLE_NAME,
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=IDENTITY_COLUMNS,
        integer_columns=(
            "selection_index",
            "n_outer_folds_selected",
            "n_outer_folds_with_finite_evidence",
        ),
        boolean_columns=(
            "passes_movement_firing_rate",
            "passes_stability",
            "eligible",
            "valid_nested_cv",
        ),
    )
    index = _frame_from_dynamic_table(
        dataset_index,
        expected_name=NWB_DATASET_INDEX_TABLE_NAME,
        columns=DATASET_INDEX_COLUMNS,
        text_columns=tuple(
            column for column in DATASET_INDEX_COLUMNS if column != "dataset_index"
        ),
        integer_columns=("dataset_index",),
    ).sort_values("dataset_index", kind="stable")
    if index["dataset_key"].tolist() != ["nested_cv", "full_refit"] or index[
        "dataset_index"
    ].tolist() != [0, 1]:
        raise ValueError("MotorEncoding NWB dataset index is noncanonical.")
    component_text = (
        "dataset_key",
        "component_name",
        "dimensions_json",
        "shape_json",
        "dtype",
        "text_values_json",
        "attrs_json",
    )
    component_kwargs = {
        "columns": ARRAY_COMPONENT_COLUMNS,
        "text_columns": component_text,
        "integer_columns": ("component_index", "numeric_count"),
        "vector_columns": ("numeric_values",),
    }
    coordinate_frame = _frame_from_dynamic_table(
        coordinates,
        expected_name=NWB_COORDINATES_TABLE_NAME,
        **component_kwargs,
    )
    nested_frame = _frame_from_dynamic_table(
        nested_cv_arrays,
        expected_name=NWB_NESTED_CV_ARRAYS_TABLE_NAME,
        **component_kwargs,
    )
    full_frame = _frame_from_dynamic_table(
        full_refit_arrays,
        expected_name=NWB_FULL_REFIT_ARRAYS_TABLE_NAME,
        **component_kwargs,
    )
    provenance_frame = _frame_from_dynamic_table(
        provenance,
        expected_name=NWB_PROVENANCE_TABLE_NAME,
        columns=PROVENANCE_COLUMNS,
        text_columns=PROVENANCE_COLUMNS,
    )
    if len(provenance_frame) != 1:
        raise ValueError("MotorEncoding provenance must contain exactly one row.")
    source = provenance_frame.iloc[0].to_dict()
    if source["artifact_schema_version"] != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("MotorEncoding NWB artifact schema version is unsupported.")
    if source["result_schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("MotorEncoding result schema version is unsupported.")
    if _parse_json(
        source["model_spec_json"], name="model spec", expected_type=dict
    ) != dict(MODEL_SPEC):
        raise ValueError("MotorEncoding model spec is stale.")
    if _parse_json(
        source["output_rule_json"], name="output rule", expected_type=dict
    ) != _json_ready(dict(OUTPUT_RULE)):
        raise ValueError("MotorEncoding output rule is stale.")
    records = {str(row["dataset_key"]): row for row in index.to_dict("records")}
    nested = _dataset_from_nwb_frames(
        index_record=records["nested_cv"],
        coordinates=coordinate_frame,
        variables=nested_frame,
    )
    full = _dataset_from_nwb_frames(
        index_record=records["full_refit"],
        coordinates=coordinate_frame,
        variables=full_frame,
    )
    legacy = _parse_json(
        source["legacy_artifact_provenance_json"],
        name="legacy provenance",
        expected_type=dict,
    )
    parameters = _parse_json(
        source["parameters_json"], name="parameters", expected_type=dict
    )
    return validate_motor_encoding_result(
        {
            "metadata": _parse_json(
                source["metadata_json"], name="metadata", expected_type=dict
            ),
            "parameters": parameters,
            "selected_units": selected,
            "nested_cv": nested,
            "full_refit": full,
            "n_units_input": len(selected),
            "n_units_eligible": int(selected["eligible"].sum()),
            "n_units_valid": int(selected["valid_nested_cv"].sum()),
            "n_outer_folds_expected": int(parameters["outer_n_folds"]),
            "n_outer_folds_valid": int(
                np.sum(
                    (np.asarray(nested["outer_train_bin_count"], dtype=int) > 0)
                    & (np.asarray(nested["outer_test_bin_count"], dtype=int) > 0)
                )
            ),
            "selected_units_sha256": _selected_units_identity_sha256(selected),
            "analysis_status": source["analysis_status"],
            "artifact_origin": source["artifact_origin"],
            "legacy_artifact_provenance": legacy or None,
        }
    )


def _selected_units_identity_sha256(table: pd.DataFrame) -> str:
    """Return the established selected-unit identity digest."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    return unit_identity_sha256(
        table.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict("records")
    )


def _semantic_frame_sha256(
    table: pd.DataFrame,
    *,
    columns: Sequence[str],
    vector_columns: Sequence[str] = (),
) -> str:
    """Hash one canonical frame without HDF5 or object-ID dependence."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(columns), separators=(",", ":")).encode())
    for record in table.loc[:, list(columns)].to_dict("records"):
        for column in columns:
            digest.update(column.encode())
            value = record[column]
            if column in vector_columns:
                vector = np.asarray(value, dtype=np.float64).copy()
                vector[np.isnan(vector)] = np.nan
                digest.update(np.asarray([vector.size], dtype=np.int64).tobytes())
                digest.update(vector.tobytes(order="C"))
            else:
                digest.update(
                    json.dumps(
                        _json_ready(value),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                )
    return digest.hexdigest()


def motor_encoding_nwb_hashes(result: Mapping[str, Any]) -> dict[str, str]:
    """Return semantic hashes for all six NWB objects and the full result."""
    canonical = validate_motor_encoding_result(result)
    frames = {
        "selected_units_table_sha256": (
            canonical["selected_units"],
            SELECTED_UNIT_COLUMNS,
            (),
        ),
        "dataset_index_sha256": (
            _dataset_index_frame(canonical),
            DATASET_INDEX_COLUMNS,
            (),
        ),
        "coordinates_sha256": (
            _coordinates_frame(canonical),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "nested_cv_arrays_sha256": (
            _variables_frame(canonical, dataset_key="nested_cv"),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "full_refit_arrays_sha256": (
            _variables_frame(canonical, dataset_key="full_refit"),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "provenance_sha256": (
            _provenance_frame(canonical),
            PROVENANCE_COLUMNS,
            (),
        ),
    }
    hashes = {
        name: _semantic_frame_sha256(
            frame,
            columns=columns,
            vector_columns=vector_columns,
        )
        for name, (frame, columns, vector_columns) in frames.items()
    }
    hashes["motor_encoding_sha256"] = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return hashes


def _file_sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_motor_encoding_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write and reload one motor-encoding artifact bundle."""
    result = validate_motor_encoding_result(result)
    destination = Path(path)
    result_id = result["metadata"]["motor_encoding_id"]
    if destination.name != result_id:
        raise ValueError("Artifact directory name must equal the result UUID.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite motor encoding artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        result["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME,
            index=False,
        )
        result["nested_cv"].to_netcdf(temporary / NESTED_CV_FILENAME)
        result["full_refit"].to_netcdf(temporary / FULL_REFIT_FILENAME)
        provenance = result.get("legacy_artifact_provenance")
        provenance_json = (
            ""
            if provenance is None
            else json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        )
        common = {
            **result["metadata"],
            "parameter_name": result["parameters"]["parameter_name"],
            "parameter_sha256": result["parameters"]["parameter_sha256"],
            "model_spec_sha256": result["parameters"]["model_spec_sha256"],
            "output_rule_sha256": result["parameters"]["output_rule_sha256"],
            "n_units_input": result["n_units_input"],
            "n_units_eligible": result["n_units_eligible"],
            "n_units_valid": result["n_units_valid"],
            "n_outer_folds_expected": result["n_outer_folds_expected"],
            "n_outer_folds_valid": result["n_outer_folds_valid"],
            "selected_units_sha256": result["selected_units_sha256"],
            "analysis_status": result["analysis_status"],
            "artifact_origin": result["artifact_origin"],
            "legacy_artifact_provenance_json": provenance_json,
        }
        rows = []
        for key, filename, kind in (
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("nested_cv", NESTED_CV_FILENAME, "netcdf"),
            ("full_refit", FULL_REFIT_FILENAME, "netcdf"),
        ):
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
            temporary / MANIFEST_FILENAME,
            index=False,
        )
        load_motor_encoding_artifact(temporary, _allow_temporary_name=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return get_motor_encoding_artifact_paths(
        animal_name=result["metadata"]["animal_name"],
        date=result["metadata"]["date"],
        epoch=result["metadata"]["epoch"],
        region=result["metadata"]["region"],
        motor_encoding_id=result_id,
        artifact_root=destination.parents[5],
    )


def load_motor_encoding_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one motor-encoding artifact bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Motor encoding manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if list(manifest.columns) != list(MANIFEST_COLUMNS):
        raise ValueError("Motor encoding manifest columns are not canonical.")
    expected = {
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "nested_cv": (NESTED_CV_FILENAME, "netcdf"),
        "full_refit": (FULL_REFIT_FILENAME, "netcdf"),
    }
    if manifest.empty or set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("Motor encoding manifest artifact set is incomplete.")
    if manifest["artifact_key"].duplicated().any():
        raise ValueError("Motor encoding manifest artifact keys must be unique.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("Motor encoding manifest names or kinds are stale.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(f"Manifest checksum mismatch for {artifact_path}.")
    first = manifest.iloc[0]
    metadata = {
        name: str(first[name])
        for name in (
            "motor_encoding_id",
            "animal_name",
            "date",
            "region",
            "epoch",
        )
    }
    if not _allow_temporary_name and directory.name != metadata[
        "motor_encoding_id"
    ]:
        raise ValueError("Artifact directory name does not match its result UUID.")
    import xarray as xr

    with xr.open_dataset(directory / NESTED_CV_FILENAME) as opened:
        nested_cv = opened.load()
    with xr.open_dataset(directory / FULL_REFIT_FILENAME) as opened:
        full_refit = opened.load()
    try:
        effective = json.loads(str(nested_cv.attrs["effective_parameters_json"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Nested-CV parameters could not be decoded.") from exc
    parameters = validate_motor_encoding_parameters(**effective)
    parameters = _parameter_metadata(
        parameter_name=str(first["parameter_name"]),
        parameter_sha256=str(first["parameter_sha256"]),
        model_spec_sha256=str(first["model_spec_sha256"]),
        output_rule_sha256=str(first["output_rule_sha256"]),
        parameters=parameters,
    )
    common = {
        **metadata,
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "model_spec_sha256": parameters["model_spec_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "n_units_input": int(first["n_units_input"]),
        "n_units_eligible": int(first["n_units_eligible"]),
        "n_units_valid": int(first["n_units_valid"]),
        "n_outer_folds_expected": int(first["n_outer_folds_expected"]),
        "n_outer_folds_valid": int(first["n_outer_folds_valid"]),
        "selected_units_sha256": str(first["selected_units_sha256"]),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance_json": str(
            first["legacy_artifact_provenance_json"]
        ),
    }
    for name, value in common.items():
        if not np.all(manifest[name].astype(str) == str(value)):
            raise ValueError(f"Motor encoding manifest has inconsistent {name!r}.")
    provenance_json = common["legacy_artifact_provenance_json"]
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": pd.read_parquet(directory / SELECTED_UNITS_FILENAME),
        "nested_cv": nested_cv,
        "full_refit": full_refit,
        "n_units_input": common["n_units_input"],
        "n_units_eligible": common["n_units_eligible"],
        "n_units_valid": common["n_units_valid"],
        "n_outer_folds_expected": common["n_outer_folds_expected"],
        "n_outer_folds_valid": common["n_outer_folds_valid"],
        "selected_units_sha256": common["selected_units_sha256"],
        "analysis_status": common["analysis_status"],
        "artifact_origin": common["artifact_origin"],
        "legacy_artifact_provenance": (
            None if not provenance_json else json.loads(provenance_json)
        ),
        "manifest": manifest,
    }
    return validate_motor_encoding_result(result)


def _legacy_identity_table(
    legacy_unit_ids: Sequence[Any],
    resolver: Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]],
) -> pd.DataFrame:
    """Resolve every legacy nested-CV unit to one current persistent identity."""
    rows: list[dict[str, str]] = []
    stable_ids: set[str] = set()
    group_ids: set[str] = set()
    for legacy_unit_id in legacy_unit_ids:
        if isinstance(resolver, Mapping):
            matches = [
                value
                for key, value in resolver.items()
                if str(key) == str(legacy_unit_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Legacy unit {legacy_unit_id!r} must resolve exactly once."
                )
            identity = dict(matches[0])
        elif callable(resolver):
            try:
                identity = dict(resolver(legacy_unit_id))
            except LookupError as exc:
                raise ValueError(
                    f"Legacy unit {legacy_unit_id!r} could not be resolved."
                ) from exc
        else:
            raise TypeError("unit_identity_resolver must be a mapping or callable.")
        missing = sorted(
            {"spikesorting_merge_id", "unit_id"}.difference(identity)
        )
        if missing:
            raise ValueError(f"Resolved legacy identity is missing {missing!r}.")
        merge_id = str(identity["spikesorting_merge_id"])
        source_unit_id = str(identity["unit_id"])
        stable_id = f"{merge_id}:{source_unit_id}"
        group_id = str(identity.get("group_unit_id", legacy_unit_id))
        if not merge_id or not source_unit_id or not group_id:
            raise ValueError("Resolved legacy identity fields must be non-empty.")
        if stable_id in stable_ids or group_id in group_ids:
            raise ValueError("Resolved legacy identities must be unique.")
        stable_ids.add(stable_id)
        group_ids.add(group_id)
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": source_unit_id,
                "stable_unit_id": stable_id,
                "group_unit_id": group_id,
                "_legacy_unit_id": str(legacy_unit_id),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=(*IDENTITY_COLUMNS, "_legacy_unit_id"),
    )


def _registration_movement_rates(
    movement_firing_rate_table: pd.DataFrame,
    *,
    identity: pd.DataFrame,
    metadata: Mapping[str, str],
) -> np.ndarray:
    """Align one current movement-rate artifact to resolved legacy identities."""
    from v1ca1.spyglass.movement import validate_movement_firing_rate_table

    observed = validate_movement_firing_rate_table(
        movement_firing_rate_table
    ).copy()
    for field in ("animal_name", "date", "region", "epoch"):
        if observed[field].astype(str).unique().tolist() != [metadata[field]]:
            raise ValueError(f"MovementFiringRate has mismatched {field}.")
    if observed["firing_rate_status"].astype(str).unique().tolist() != ["valid"]:
        raise ValueError("Legacy motor registration requires valid movement rates.")
    observed["stable_unit_id"] = observed["stable_unit_id"].astype(str)
    observed = observed.set_index("stable_unit_id", drop=False)
    stable_ids = identity["stable_unit_id"].astype(str).tolist()
    if set(observed.index) != set(stable_ids):
        raise ValueError("Movement rates do not match resolved legacy identities.")
    observed = observed.loc[stable_ids]
    for field in ("spikesorting_merge_id", "unit_id"):
        if observed[field].astype(str).tolist() != identity[field].astype(str).tolist():
            raise ValueError(f"MovementFiringRate {field} does not match identities.")
    rates = pd.to_numeric(
        observed["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("Movement firing rates must be finite and non-negative.")
    return rates


def _validate_legacy_dataset_pair(
    nested_cv: Any,
    full_refit: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require one exact current schema-v2 legacy motor output pair."""
    for name, dataset, required in (
        (
            "legacy nested_cv",
            nested_cv,
            {
                "outer_unit_selected",
                "outer_info_bits_per_spike",
                "pooled_info_bits_per_spike",
                "pooled_spike_sum",
                "outer_train_bin_count",
                "outer_test_bin_count",
            },
        ),
        (
            "legacy full_refit",
            full_refit,
            {"selected_ridge", "movement_firing_rate_hz"},
        ),
    ):
        if "model" not in dataset.coords or tuple(
            str(value) for value in np.asarray(dataset.model)
        ) != MODEL_NAMES:
            raise ValueError(f"{name} does not use the fixed nine-model order.")
        missing = sorted(required.difference(dataset.data_vars))
        if missing:
            raise ValueError(f"{name} is missing variables {missing!r}.")
        for field in ("animal_name", "date", "region", "epoch"):
            if str(dataset.attrs.get(field, "")) != metadata[field]:
                raise ValueError(f"{name} has mismatched {field} metadata.")
        if str(dataset.attrs.get("schema_version", "")) != "2":
            raise ValueError(f"{name} must use motor output schema version 2.")
        try:
            definitions = json.loads(str(dataset.attrs["model_definitions_json"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} has invalid model definitions.") from exc
        if definitions != dict(MODEL_SPEC):
            raise ValueError(f"{name} model definitions do not match the fixed spec.")
    if int(nested_cv.sizes.get("outer_fold", -1)) != int(
        parameters["outer_n_folds"]
    ):
        raise ValueError("Legacy nested_cv outer-fold count is inconsistent.")
    try:
        nested_fit = json.loads(str(nested_cv.attrs["fit_parameters_json"]))
        full_fit = json.loads(str(full_refit.attrs["fit_parameters_json"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Legacy fit_parameters_json is invalid.") from exc
    checks = {
        "bin_size_s": parameters["evaluation_bin_size_s"],
        "n_folds": parameters["outer_n_folds"],
        "inner_n_folds": parameters["inner_n_folds"],
        "seed": parameters["random_seed"],
        "ridges": list(parameters["ridge_values"]),
        "spatial_bin_sizes_cm": list(parameters["spatial_bin_sizes_cm"]),
        "motor_feature_mode": parameters["motor_feature_mode"],
        "motor_zscore_eps": parameters["motor_zscore_eps"],
        "motor_spline_k": parameters["motor_spline_n_basis"],
        "motor_spline_order": parameters["motor_spline_order"],
        "tp_spline_order": parameters["position_spline_order"],
        "speed_sigma_s": parameters["speed_smoothing_sigma_s"],
        "generalized_place_branch_gap_cm": parameters[
            "generalized_place_branch_gap_cm"
        ],
    }
    for field, expected in checks.items():
        if field not in nested_fit or field not in full_fit:
            raise ValueError(f"Legacy fit parameters are missing {field!r}.")
        for observed in (nested_fit[field], full_fit[field]):
            if isinstance(expected, list):
                if not np.allclose(
                    np.asarray(observed, dtype=float),
                    np.asarray(expected, dtype=float),
                    rtol=1e-12,
                    atol=1e-15,
                ):
                    raise ValueError(f"Legacy fit parameter {field!r} is stale.")
            elif isinstance(expected, str):
                if str(observed) != expected:
                    raise ValueError(f"Legacy fit parameter {field!r} is stale.")
            elif not np.isclose(
                float(observed), float(expected), rtol=1e-12, atol=1e-15
            ):
                raise ValueError(f"Legacy fit parameter {field!r} is stale.")
    expected_basis_configs = build_graph_derived_position_basis_configs(
        graph_inputs_by_configuration,
        spatial_bin_sizes_cm=parameters["spatial_bin_sizes_cm"],
        spline_order=parameters["position_spline_order"],
        generalized_place_branch_gap_cm=parameters[
            "generalized_place_branch_gap_cm"
        ],
    )
    for fit_parameters in (nested_fit, full_fit):
        observed_configs = fit_parameters.get("position_basis_configs")
        if not isinstance(observed_configs, list) or len(observed_configs) != len(
            expected_basis_configs
        ):
            raise ValueError("Legacy position basis configurations are incomplete.")
        for observed, expected in zip(
            observed_configs,
            expected_basis_configs,
            strict=True,
        ):
            for field in (
                "spatial_bin_size_cm",
                "spline_order",
                "trajectory_length_cm",
                "generalized_place_length_cm",
                "trajectory_n_splines",
                "generalized_place_n_splines",
                "generalized_place_branch_gap_cm",
            ):
                if field not in observed or not np.isclose(
                    float(observed[field]),
                    float(expected[field]),
                    rtol=1e-10,
                    atol=1e-12,
                ):
                    raise ValueError(
                        "Legacy position basis geometry does not match selected "
                        f"NWB graphs for {field!r}."
                    )
    for dataset in (nested_cv, full_refit):
        if not np.isclose(
            float(dataset.attrs.get("min_firing_rate_hz", np.nan)),
            float(parameters["minimum_movement_firing_rate_hz"]),
        ):
            raise ValueError("Legacy movement firing-rate threshold is stale.")


def _replace_legacy_units_with_group_ids(
    dataset: Any,
    identity: pd.DataFrame,
) -> Any:
    """Replace legacy unit labels with their resolved current group labels."""
    legacy_to_group = dict(
        zip(
            identity["_legacy_unit_id"].astype(str),
            identity["group_unit_id"].astype(str),
            strict=True,
        )
    )
    legacy_ids = [str(value) for value in np.asarray(dataset.unit)]
    if any(value not in legacy_to_group for value in legacy_ids):
        raise ValueError("A legacy dataset unit does not resolve to the nested cohort.")
    return dataset.assign_coords(
        unit=("unit", np.asarray([legacy_to_group[value] for value in legacy_ids]))
    )


def register_existing_motor_encoding_artifact(
    *,
    source_nested_cv_path: Path,
    source_full_refit_path: Path,
    destination_path: Path | None,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    motor_encoding_id: Any,
    movement_firing_rate_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    unit_identity_resolver: (
        Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]]
    ),
    primary_position_source: str,
    orientation_reference_position_source: str,
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float,
    parameter_name: str,
    parameter_sha256: str | None = None,
    model_spec_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    evaluation_bin_size_s: float = DEFAULT_BIN_SIZE_S,
    outer_n_folds: int = DEFAULT_N_FOLDS,
    inner_n_folds: int = DEFAULT_INNER_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ridge_values: Sequence[float] = DEFAULT_RIDGE_VALUES,
    spatial_bin_sizes_cm: Sequence[float] = DEFAULT_SPATIAL_BIN_SIZES_CM,
    motor_feature_mode: str = DEFAULT_MOTOR_FEATURE_MODE,
    motor_zscore_eps: float = DEFAULT_MOTOR_ZSCORE_EPS,
    motor_spline_n_basis: int = DEFAULT_MOTOR_SPLINE_N_BASIS,
    motor_spline_order: int = DEFAULT_MOTOR_SPLINE_ORDER,
    position_spline_order: int = DEFAULT_POSITION_SPLINE_ORDER,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    generalized_place_branch_gap_cm: float = (
        DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM
    ),
    source_v1ca1_git_commit: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and copy one exact schema-v2 legacy motor output pair."""
    source_nested_cv_path = Path(source_nested_cv_path)
    source_full_refit_path = Path(source_full_refit_path)
    for name, source_path in (
        ("source_nested_cv_path", source_nested_cv_path),
        ("source_full_refit_path", source_full_refit_path),
    ):
        if not source_path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {source_path}")
    import xarray as xr

    with xr.open_dataset(source_nested_cv_path) as opened:
        nested_cv = opened.load()
    with xr.open_dataset(source_full_refit_path) as opened:
        full_refit = opened.load()
    metadata = _common_metadata(
        motor_encoding_id=motor_encoding_id,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
    )
    effective = validate_motor_encoding_parameters(
        minimum_movement_firing_rate_hz=minimum_movement_firing_rate_hz,
        minimum_stability_correlation=minimum_stability_correlation,
        evaluation_bin_size_s=evaluation_bin_size_s,
        outer_n_folds=outer_n_folds,
        inner_n_folds=inner_n_folds,
        random_seed=random_seed,
        ridge_values=ridge_values,
        spatial_bin_sizes_cm=spatial_bin_sizes_cm,
        motor_feature_mode=motor_feature_mode,
        motor_zscore_eps=motor_zscore_eps,
        motor_spline_n_basis=motor_spline_n_basis,
        motor_spline_order=motor_spline_order,
        position_spline_order=position_spline_order,
        speed_smoothing_sigma_s=speed_smoothing_sigma_s,
        generalized_place_branch_gap_cm=generalized_place_branch_gap_cm,
    )
    parameters = _parameter_metadata(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        model_spec_sha256=model_spec_sha256,
        output_rule_sha256=output_rule_sha256,
        parameters=effective,
    )
    _validate_legacy_dataset_pair(
        nested_cv,
        full_refit,
        metadata=metadata,
        parameters=parameters,
        graph_inputs_by_configuration=graph_inputs_by_configuration,
    )
    identity = _legacy_identity_table(
        np.asarray(nested_cv.unit.values),
        unit_identity_resolver,
    )
    rates = _registration_movement_rates(
        movement_firing_rate_table,
        identity=identity,
        metadata=metadata,
    )
    eligibility = _build_unit_eligibility_table(
        identity=identity.loc[:, list(IDENTITY_COLUMNS)],
        movement_firing_rates_hz=rates,
        stability_tables_by_trajectory=stability_tables_by_trajectory,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        epoch=metadata["epoch"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
    )
    nested_cv = _replace_legacy_units_with_group_ids(nested_cv, identity)
    full_refit = _replace_legacy_units_with_group_ids(full_refit, identity)
    eligible_group_ids = eligibility.loc[
        eligibility["eligible"], "group_unit_id"
    ].astype(str).tolist()
    full_group_ids = [str(value) for value in np.asarray(full_refit.unit)]
    missing_eligible = sorted(set(eligible_group_ids) - set(full_group_ids))
    if missing_eligible:
        raise ValueError(
            "Legacy full-refit output is missing stability-eligible units: "
            f"{missing_eligible!r}."
        )
    full_refit = full_refit.sel(unit=eligible_group_ids)
    selected_units = _build_selected_units_table(
        eligibility=eligibility,
        nested_cv=nested_cv,
        full_refit=full_refit,
    )
    nested_cv = _canonicalize_computed_dataset(
        nested_cv,
        role="nested_cv",
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        artifact_origin="registered_existing",
    )
    full_refit = _canonicalize_computed_dataset(
        full_refit,
        role="full_refit",
        selected_units=selected_units,
        metadata=metadata,
        parameters=parameters,
        primary_position_source=primary_position_source,
        orientation_reference_position_source=(
            orientation_reference_position_source
        ),
        artifact_origin="registered_existing",
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    selected_units_sha256 = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    train_counts = np.asarray(nested_cv["outer_train_bin_count"], dtype=int)
    test_counts = np.asarray(nested_cv["outer_test_bin_count"], dtype=int)
    provenance = {
        "source_nested_cv_path": str(source_nested_cv_path),
        "source_nested_cv_sha256": _file_sha256(source_nested_cv_path),
        "source_full_refit_path": str(source_full_refit_path),
        "source_full_refit_sha256": _file_sha256(source_full_refit_path),
        "source_schema_version": "2",
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "unit_identity_validation": "caller_resolver_for_every_nested_unit",
        "parameter_validation": "exact_fit_parameters_json_and_attrs",
    }
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": selected_units,
        "nested_cv": nested_cv,
        "full_refit": full_refit,
        "n_units_input": len(selected_units),
        "n_units_eligible": int(
            selected_units["eligible"].sum()
        ),
        "n_units_valid": int(selected_units["valid_nested_cv"].sum()),
        "n_outer_folds_expected": parameters["outer_n_folds"],
        "n_outer_folds_valid": int(np.sum((train_counts > 0) & (test_counts > 0))),
        "selected_units_sha256": selected_units_sha256,
        "analysis_status": _analysis_status(selected_units),
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
    }
    result = validate_motor_encoding_result(result)
    if destination_path is None:
        return result
    paths = write_motor_encoding_artifact(
        result,
        destination_path,
        overwrite=overwrite,
    )
    return {**result, "artifact_paths": paths}


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "DEFAULT_ARTIFACT_ROOT",
    "MANUSCRIPT_PARAMETERS_BY_REGION",
    "MODEL_NAMES",
    "MODEL_SPEC",
    "MODEL_SPEC_SHA256",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "RESULT_SCHEMA_VERSION",
    "build_graph_derived_position_basis_configs",
    "build_motor_model_features",
    "compute_motor_encoding",
    "get_motor_encoding_artifact_paths",
    "load_motor_encoding_artifact",
    "motor_encoding_nwb_hashes",
    "motor_encoding_result_from_nwb_objects",
    "motor_encoding_result_to_nwb_objects",
    "register_existing_motor_encoding_artifact",
    "validate_motor_encoding_parameters",
    "validate_motor_encoding_result",
    "validate_position_pair",
    "write_motor_encoding_artifact",
]
