"""Database-free light/dark cvPCA artifacts for the custom Spyglass pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "cv_pca"
RESULT_FILENAME = "cv_pca.nc"
SUMMARY_FILENAME = "summary.parquet"
SPECTRUM_FILENAME = "within_spectrum.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
LAP_ASSIGNMENTS_FILENAME = "lap_assignments.parquet"
TRAJECTORY_QC_FILENAME = "trajectory_qc.parquet"
MANIFEST_FILENAME = "manifest.parquet"
BUNDLE_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"

TRAJECTORY_TYPES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
CONDITIONS = ("dark", "light")
DEFAULT_BIN_SIZE_CM = 4.0
DEFAULT_N_GROUPS = 4
DEFAULT_MIN_OCCUPANCY_S = 0.01
DEFAULT_UNIT_FILTER_MODE = "shared-active"
DEFAULT_NORMALIZATION = "zscore"
DEFAULT_MIN_CONDITION_SD_HZ = 1e-6
DEFAULT_MIN_SCALE = 1e-6
DEFAULT_RANDOM_SEED = 47
DEFAULT_POSITION_OFFSET_SAMPLES = 10
DEFAULT_REGION_MIN_FIRING_RATE_HZ = {"v1": 0.5, "ca1": 0.0}

ANALYSIS_STATUSES = (
    "valid",
    "no_units",
    "no_valid_position",
    "no_movement",
    "no_trials",
    "insufficient_laps",
    "no_shared_position_bins",
    "no_eligible_units",
)

OUTPUT_RULE = {
    "version": 1,
    "row_scope": "one_session_light_epoch_dark_epoch_region_random_seed",
    "representation": "four_concatenated_path_specific_physical_place_trajectories",
    "linearization_direction": "from_center_for_both_inbound_and_outbound_laps",
    "trajectory_order": TRAJECTORY_TYPES,
    "lap_policy": "all_laps_randomly_partitioned_into_disjoint_groups",
    "condition_bin_policy": "intersection_across_all_dark_and_light_lap_groups",
    "unit_policy": "same_sorted_units_and_stable_identity_in_both_epochs",
    "position_input_policy": (
        "untrimmed_position_series_then_discard_position_offset_samples_once"
    ),
    "default_unit_filter": "shared_active_in_both_epochs",
    "full_residual_matrix_storage": False,
    "residual_fraction_by_unit_class_storage": False,
    "terminal_artifact_policy": "explicit_expected_empty_input_statuses",
    "legacy_registration_policy": (
        "readable_complete_netcdf_and_summary_then_exact_nwb_recomputation"
    ),
    "time_unit": "s",
    "position_unit": "cm",
}

IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "input_unit_index",
    "dark_movement_firing_rate_hz",
    "light_movement_firing_rate_hz",
    "passes_dark_firing_rate",
    "passes_light_firing_rate",
    "dark_condition_sd_hz",
    "light_condition_sd_hz",
    "passes_dark_condition_sd",
    "passes_light_condition_sd",
    "unit_class",
    "included_in_cv_pca",
    "unit_qc_status",
)
SUMMARY_COLUMNS = (
    "cv_pca_id",
    "animal_name",
    "date",
    "region",
    "light_epoch",
    "dark_epoch",
    "random_seed",
    "condition",
    "epoch",
    "n_units",
    "within_cv_participation_ratio",
    "within_cv_n_components_80",
    "within_cv_n_components_90",
    "analysis_status",
)
SPECTRUM_COLUMNS = (
    "cv_pca_id",
    "animal_name",
    "date",
    "region",
    "light_epoch",
    "dark_epoch",
    "random_seed",
    "condition",
    "component",
    "within_cv_spectrum_signed",
    "within_cv_spectrum_positive",
    "within_cv_cumulative_shared_variance",
    "analysis_status",
)
LAP_ASSIGNMENT_COLUMNS = (
    "condition",
    "epoch",
    "trajectory_type",
    "lap_index",
    "lap_number",
    "group_index",
    "start_time_s",
    "end_time_s",
)
TRAJECTORY_QC_COLUMNS = (
    "condition",
    "epoch",
    "trajectory_type",
    "n_trials",
    "n_groups_required",
    "movement_supported_duration_s",
    "graph_length_cm",
    "n_shared_valid_bins",
    "trajectory_status",
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "cv_pca_id",
    "animal_name",
    "date",
    "region",
    "light_epoch",
    "dark_epoch",
    "random_seed",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "upstream_provenance_json",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "n_input_units",
    "n_selected_units",
    "bundle_schema_version",
)

# This list intentionally excludes the scientifically invalid legacy
# residual_fraction_by_unit_class field and the enormous residual_matrix field.
SCIENTIFIC_RESULT_VARIABLES = (
    "score_covariance_by_component",
    "test_variance_by_component",
    "cumulative_test_variance_captured",
    "residual_fraction",
    "total_test_variance",
    "valid_projection",
    "within_cv_spectrum_signed",
    "within_cv_spectrum_positive",
    "within_cv_cumulative_shared_variance",
    "within_cv_participation_ratio",
    "within_cv_n_components_80",
    "within_cv_n_components_90",
    "residualized_light_cutoff_components",
    "residualized_light_cv_spectrum_signed_by_fold",
    "residualized_light_cv_spectrum_signed",
    "residualized_light_cv_spectrum_positive",
    "residualized_light_cv_cumulative_shared_variance",
    "residualized_light_cv_participation_ratio",
    "residualized_light_cv_n_components_80",
    "residualized_light_cv_n_components_90",
)
PROHIBITED_RESULT_VARIABLES = (
    "residual_matrix",
    "residual_fraction_by_unit_class",
)
VALID_RESULT_SUPPORT_VARIABLES = (
    "dark_firing_rate_hz",
    "light_firing_rate_hz",
    "dark_condition_sd_hz",
    "light_condition_sd_hz",
    "unit_class_per_unit",
    "condition_trajectory",
    "condition_bin_center",
    "condition_bin_index",
)


def _science_module() -> Any:
    """Import the existing scientific implementation only when needed."""
    from v1ca1.signal_dim import cv_pca

    return cv_pca


def _provenance_sha256(value: Any) -> str:
    """Return the shared deterministic provenance digest."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(value)


OUTPUT_RULE_SHA256 = _provenance_sha256(OUTPUT_RULE)


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe, non-empty path component."""
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


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    """Return one finite, non-boolean floating-point value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return result


def _positive_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    """Return one non-boolean integer at least ``minimum``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _canonical_json_mapping(
    value: Mapping[str, Any], *, name: str
) -> tuple[dict[str, Any], str]:
    """Return one JSON-roundtripped non-empty provenance mapping."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping.")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible values.") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError(f"{name} must encode one non-empty object.")
    return decoded, encoded


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_cv_pca_artifact_paths(
    *,
    animal_name: str,
    date: str,
    light_epoch: str,
    dark_epoch: str,
    region: str,
    cv_pca_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one immutable session-first cvPCA artifact bundle."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
            "region": region,
        }.items()
    }
    result_id = _uuid_string(cv_pca_id, name="cv_pca_id")
    directory = (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / f"{components['light_epoch']}_vs_{components['dark_epoch']}"
        / components["region"]
        / result_id
    )
    return {
        "artifact_dir": directory,
        "result_path": directory / RESULT_FILENAME,
        "summary_path": directory / SUMMARY_FILENAME,
        "spectrum_path": directory / SPECTRUM_FILENAME,
        "selected_units_path": directory / SELECTED_UNITS_FILENAME,
        "lap_assignments_path": directory / LAP_ASSIGNMENTS_FILENAME,
        "trajectory_qc_path": directory / TRAJECTORY_QC_FILENAME,
        "manifest_path": directory / MANIFEST_FILENAME,
    }


def get_legacy_cv_pca_paths(
    analysis_path: Path,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    random_seed: int = DEFAULT_RANDOM_SEED,
    seed_specific: bool = True,
) -> dict[str, Path]:
    """Return the complete legacy NetCDF/summary pair for one seed.

    The repeat workflow's seed-specific files are canonical because a custom
    table row represents one random seed.  ``seed_specific=False`` remains
    available for auditing the older single-run files.
    """
    seed = _positive_integer(random_seed, name="random_seed", minimum=0)
    stem = f"{region}_{light_epoch}_vs_{dark_epoch}_cv_pca"
    if seed_specific:
        stem = f"{stem}_seed{seed}"
    directory = Path(analysis_path) / "signal_dim" / "cv_pca"
    return {
        "result_path": directory / f"{stem}.nc",
        "summary_path": directory / f"{stem}_summary.parquet",
    }


def validate_cv_pca_parameters(
    *,
    region: str,
    bin_size_cm: float = DEFAULT_BIN_SIZE_CM,
    n_groups: int = DEFAULT_N_GROUPS,
    min_occupancy_s: float = DEFAULT_MIN_OCCUPANCY_S,
    unit_filter_mode: str = DEFAULT_UNIT_FILTER_MODE,
    min_firing_rate_hz: float | None = None,
    min_condition_sd_hz: float = DEFAULT_MIN_CONDITION_SD_HZ,
    normalization: str = DEFAULT_NORMALIZATION,
    min_scale: float = DEFAULT_MIN_SCALE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Validate one cvPCA parameter row; repeats are separate selections."""
    region = str(region).lower()
    if min_firing_rate_hz is None:
        if region not in DEFAULT_REGION_MIN_FIRING_RATE_HZ:
            raise ValueError(
                "min_firing_rate_hz is required outside the v1/ca1 presets."
            )
        min_firing_rate_hz = DEFAULT_REGION_MIN_FIRING_RATE_HZ[region]
    filter_mode = str(unit_filter_mode)
    if filter_mode not in {"shared-active", "dark-active", "union-active"}:
        raise ValueError("unit_filter_mode has an unsupported value.")
    normalization = str(normalization)
    if normalization not in {"zscore", "center"}:
        raise ValueError("normalization must be 'zscore' or 'center'.")
    groups = _positive_integer(n_groups, name="n_groups", minimum=3)
    seed = _positive_integer(random_seed, name="random_seed", minimum=0)
    parameters = {
        "bin_size_cm": _finite_float(
            bin_size_cm, name="bin_size_cm", minimum=np.nextafter(0.0, 1.0)
        ),
        "n_groups": groups,
        "min_occupancy_s": _finite_float(
            min_occupancy_s,
            name="min_occupancy_s",
            minimum=np.nextafter(0.0, 1.0),
        ),
        "unit_filter_mode": filter_mode,
        "min_firing_rate_hz": _finite_float(
            min_firing_rate_hz, name="min_firing_rate_hz", minimum=0.0
        ),
        "min_condition_sd_hz": _finite_float(
            min_condition_sd_hz,
            name="min_condition_sd_hz",
            minimum=np.nextafter(0.0, 1.0),
        ),
        "normalization": normalization,
        "min_scale": _finite_float(
            min_scale, name="min_scale", minimum=np.nextafter(0.0, 1.0)
        ),
        "random_seed": seed,
    }
    return parameters


def _interval_bounds(intervals: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return validated sorted interval starts and ends in seconds."""
    if isinstance(intervals, pd.DataFrame):
        candidates = (("start", "end"), ("start_time", "end_time"))
        columns = next(
            (pair for pair in candidates if set(pair).issubset(intervals.columns)),
            None,
        )
        if columns is None:
            raise ValueError(f"{name} DataFrame lacks interval columns.")
        starts = intervals[columns[0]].to_numpy(dtype=float)
        ends = intervals[columns[1]].to_numpy(dtype=float)
    else:
        try:
            starts = np.asarray(intervals.start, dtype=float).reshape(-1)
            ends = np.asarray(intervals.end, dtype=float).reshape(-1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must expose start and end arrays.") from exc
    if starts.shape != ends.shape:
        raise ValueError(f"{name} interval starts and ends do not align.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError(f"{name} interval bounds must be finite.")
    if np.any(ends < starts):
        raise ValueError(f"{name} interval ends precede starts.")
    if starts.size > 1 and (
        np.any(np.diff(starts) < 0.0) or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError(f"{name} intervals must be sorted and non-overlapping.")
    return starts, ends


def _interval_duration(intervals: Any, *, name: str) -> float:
    """Return the finite total duration of one interval collection."""
    starts, ends = _interval_bounds(intervals, name=name)
    return float(np.sum(ends - starts))


def _intersection_duration(first: Any, second: Any) -> float:
    """Return overlap duration for two sorted interval collections."""
    first_start, first_end = _interval_bounds(first, name="trajectory_intervals")
    second_start, second_end = _interval_bounds(second, name="movement_intervals")
    total = 0.0
    first_index = second_index = 0
    while first_index < first_start.size and second_index < second_start.size:
        start = max(first_start[first_index], second_start[second_index])
        end = min(first_end[first_index], second_end[second_index])
        total += max(0.0, end - start)
        if first_end[first_index] <= second_end[second_index]:
            first_index += 1
        else:
            second_index += 1
    return float(total)


def _position_arrays(position: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned two-dimensional position and second-based timestamps."""
    values = np.asarray(getattr(position, "d", position), dtype=float)
    timestamps = np.asarray(getattr(position, "t", ()), dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_samples, 2).")
    if timestamps.shape != (values.shape[0],):
        raise ValueError(f"{name} position values and timestamps must align.")
    if not np.all(np.isfinite(timestamps)) or (
        timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError(f"{name} timestamps must be finite and increasing.")
    return values, timestamps


def _identity_rows(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable identities aligned to the selected TsGroup runtime keys."""
    group_keys = [] if spikes is None else list(spikes.keys())
    identities = [dict(value) for value in stable_unit_ids]
    if len(group_keys) != len(identities):
        raise ValueError("TsGroup and stable unit identity lengths must match.")
    group_strings = [str(value) for value in group_keys]
    if any(not value for value in group_strings) or len(set(group_strings)) != len(
        group_strings
    ):
        raise ValueError("TsGroup runtime unit identifiers must be unique and non-empty.")
    rows: list[dict[str, Any]] = []
    stable_ids: set[str] = set()
    for index, (group_key, identity) in enumerate(
        zip(group_keys, identities, strict=True)
    ):
        missing = [
            field
            for field in ("spikesorting_merge_id", "unit_id")
            if field not in identity
        ]
        if missing:
            raise ValueError(f"Stable unit identity is missing fields {missing!r}.")
        merge_id = str(identity["spikesorting_merge_id"])
        unit_id = str(identity["unit_id"])
        stable_id = f"{merge_id}:{unit_id}"
        if not merge_id or not unit_id or stable_id in stable_ids:
            raise ValueError("Stable unit identities must be unique and non-empty.")
        stable_ids.add(stable_id)
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": stable_id,
                "group_unit_id": str(group_key),
                "input_unit_index": index,
                "_group_key": group_key,
            }
        )
    return rows


def _firing_rate_array(values: Sequence[float], *, name: str, n_units: int) -> np.ndarray:
    """Return one non-negative per-unit movement firing-rate vector."""
    output = np.asarray(values, dtype=float).reshape(-1)
    if output.shape != (n_units,):
        raise ValueError(f"{name} must have one value per selected unit.")
    if np.any(np.isinf(output)) or np.any(output < 0.0):
        raise ValueError(f"{name} may contain non-negative finite values or NaN.")
    return output


def _graph_length(graph: Mapping[str, Any], *, expected_name: str) -> float:
    """Validate one from-center graph payload and return its ordered length."""
    if str(graph.get("configuration_name", "")) != expected_name:
        raise ValueError(
            f"WTrackGraph configuration_name must be {expected_name!r}."
        )
    if str(graph.get("coordinate_unit", "")) != "cm":
        raise ValueError("WTrackGraph coordinate_unit must be 'cm'.")
    track_kwargs = dict(graph.get("track_graph_kwargs", {}))
    linear_kwargs = dict(graph.get("linearization_kwargs", {}))
    if set(track_kwargs) != {"node_positions", "edges"}:
        raise ValueError("WTrackGraph track_graph_kwargs are incomplete.")
    nodes = np.asarray(track_kwargs["node_positions"], dtype=float)
    edges = np.asarray(track_kwargs["edges"], dtype=int)
    edge_order = np.asarray(linear_kwargs.get("edge_order"), dtype=int)
    spacing = np.asarray(linear_kwargs.get("edge_spacing", ()), dtype=float).reshape(-1)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.all(np.isfinite(nodes)):
        raise ValueError("WTrackGraph nodes must be finite two-dimensional positions.")
    if edges.ndim != 2 or edges.shape[1] != 2 or edge_order.shape != edges.shape:
        raise ValueError("WTrackGraph edges and edge_order must align.")
    if spacing.shape != (max(0, len(edge_order) - 1),):
        raise ValueError("WTrackGraph edge_spacing has an unexpected length.")
    if np.any(spacing < 0.0) or not np.all(np.isfinite(spacing)):
        raise ValueError("WTrackGraph edge spacing must be finite and non-negative.")
    if edges.size and (
        np.any(edges < 0)
        or np.any(edges >= len(nodes))
        or np.any(edge_order < 0)
        or np.any(edge_order >= len(nodes))
    ):
        raise ValueError("WTrackGraph edges contain invalid node indices.")
    graph_edges = {frozenset(map(int, edge)) for edge in edges}
    ordered_edges = {frozenset(map(int, edge)) for edge in edge_order}
    if graph_edges != ordered_edges or len(graph_edges) != len(edges):
        raise ValueError("WTrackGraph edge_order must order every graph edge once.")
    segment_lengths = np.linalg.norm(
        nodes[edge_order[:, 1]] - nodes[edge_order[:, 0]], axis=1
    )
    length = float(np.sum(segment_lengths) + np.sum(spacing))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("WTrackGraph ordered length must be positive.")
    return length


def _normalize_graph_inputs(
    graph_inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], float, list[float]]:
    """Select left/right from-center graphs from a two- or four-row mapping."""
    supplied = {str(key): dict(value) for key, value in graph_inputs.items()}
    if set(TRAJECTORY_TYPES).issubset(supplied):
        selected = {
            "left": supplied["center_to_left"],
            "right": supplied["center_to_right"],
        }
    elif set(supplied) == {"center_to_left", "center_to_right"}:
        selected = {
            "left": supplied["center_to_left"],
            "right": supplied["center_to_right"],
        }
    elif set(supplied) == {"left", "right"}:
        selected = {"left": supplied["left"], "right": supplied["right"]}
    else:
        raise ValueError(
            "graph_inputs must contain left/right from-center graphs or all four "
            "trajectory graphs."
        )
    lengths = {
        "left": _graph_length(selected["left"], expected_name="center_to_left"),
        "right": _graph_length(selected["right"], expected_name="center_to_right"),
    }
    if not np.isclose(lengths["left"], lengths["right"], rtol=1e-9, atol=1e-9):
        raise ValueError("Left and right WTrackGraph path lengths must match.")
    spacings = {
        side: list(
            map(
                float,
                dict(graph["linearization_kwargs"]).get("edge_spacing", ()),
            )
        )
        for side, graph in selected.items()
    }
    if not np.allclose(spacings["left"], spacings["right"], rtol=0.0, atol=0.0):
        raise ValueError(
            "The reused cvPCA implementation requires equal left/right edge spacing."
        )
    return selected, float(lengths["left"]), spacings["left"]


def _validate_trajectory_mapping(
    value: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    """Return one exactly-four-trajectory interval mapping."""
    output = {str(key): intervals for key, intervals in value.items()}
    if set(output) != set(TRAJECTORY_TYPES):
        raise ValueError(f"{name} must contain exactly the four W-track trajectories.")
    for trajectory, intervals in output.items():
        _interval_bounds(intervals, name=f"{name}[{trajectory!r}]")
    return output


def _build_lap_assignments(
    *,
    epoch: str,
    condition: str,
    trajectory_intervals: Mapping[str, Any],
    n_groups: int,
    random_seed: int,
) -> pd.DataFrame:
    """Reproduce the scientific implementation's deterministic lap grouping."""
    science = _science_module()
    rng = science._rng_for_epoch(random_seed, epoch)
    rows: list[dict[str, Any]] = []
    for trajectory in TRAJECTORY_TYPES:
        starts, ends = _interval_bounds(
            trajectory_intervals[trajectory],
            name=f"{condition}_{trajectory}_intervals",
        )
        if starts.size < n_groups:
            continue
        selected = science.sample_lap_indices(
            starts.size,
            lap_fraction=1.0,
            n_groups=n_groups,
            rng=rng,
        )
        groups = science.split_lap_indices_into_groups(selected, n_groups, rng=rng)
        for group_index, group in enumerate(groups):
            for lap_index in np.sort(group):
                rows.append(
                    {
                        "condition": condition,
                        "epoch": epoch,
                        "trajectory_type": trajectory,
                        "lap_index": int(lap_index),
                        "lap_number": int(lap_index) + 1,
                        "group_index": group_index,
                        "start_time_s": float(starts[lap_index]),
                        "end_time_s": float(ends[lap_index]),
                    }
                )
    return pd.DataFrame.from_records(rows, columns=LAP_ASSIGNMENT_COLUMNS)


def _make_trajectory_qc(
    *,
    dark_epoch: str,
    light_epoch: str,
    dark_intervals: Mapping[str, Any],
    light_intervals: Mapping[str, Any],
    dark_movement: Any,
    light_movement: Any,
    graph_length_cm: float,
    n_groups: int,
) -> pd.DataFrame:
    """Build one eight-row input and retained-bin trajectory audit."""
    rows = []
    for condition, epoch, intervals_by_trajectory, movement in (
        ("dark", dark_epoch, dark_intervals, dark_movement),
        ("light", light_epoch, light_intervals, light_movement),
    ):
        for trajectory in TRAJECTORY_TYPES:
            starts, _ = _interval_bounds(
                intervals_by_trajectory[trajectory],
                name=f"{condition}_{trajectory}_intervals",
            )
            support_duration = _intersection_duration(
                intervals_by_trajectory[trajectory], movement
            )
            if starts.size == 0:
                status = "no_trials"
            elif starts.size < n_groups:
                status = "insufficient_laps"
            elif support_duration <= 0.0:
                status = "no_movement_support"
            else:
                status = "pending"
            rows.append(
                {
                    "condition": condition,
                    "epoch": epoch,
                    "trajectory_type": trajectory,
                    "n_trials": int(starts.size),
                    "n_groups_required": n_groups,
                    "movement_supported_duration_s": support_duration,
                    "graph_length_cm": graph_length_cm,
                    "n_shared_valid_bins": 0,
                    "trajectory_status": status,
                }
            )
    return pd.DataFrame.from_records(rows, columns=TRAJECTORY_QC_COLUMNS)


def _build_science_session(
    *,
    region: str,
    spikes: Any,
    dark_epoch: str,
    light_epoch: str,
    dark_position: Any,
    light_position: Any,
    dark_movement_intervals: Any,
    light_movement_intervals: Any,
    dark_trajectory_intervals: Mapping[str, Any],
    light_trajectory_intervals: Mapping[str, Any],
    dark_firing_rates: np.ndarray,
    light_firing_rates: np.ndarray,
    selected_graphs: Mapping[str, Mapping[str, Any]],
    graph_length_cm: float,
    edge_spacing: Sequence[float],
    position_offset_samples: int,
) -> dict[str, Any]:
    """Construct the legacy scientific session dictionary from NWB objects."""
    import track_linearization as tl

    dark_values, dark_times = _position_arrays(dark_position, name="dark_position")
    light_values, light_times = _position_arrays(light_position, name="light_position")
    graphs = {
        side: tl.make_track_graph(**dict(graph["track_graph_kwargs"]))
        for side, graph in selected_graphs.items()
    }
    edge_orders = {
        side: [
            tuple(map(int, edge))
            for edge in dict(graph["linearization_kwargs"])["edge_order"]
        ]
        for side, graph in selected_graphs.items()
    }
    trajectory_times: dict[str, dict[str, np.ndarray]] = {}
    for epoch, source in (
        (dark_epoch, dark_trajectory_intervals),
        (light_epoch, light_trajectory_intervals),
    ):
        trajectory_times[epoch] = {}
        for trajectory in TRAJECTORY_TYPES:
            starts, ends = _interval_bounds(
                source[trajectory], name=f"{epoch}_{trajectory}_intervals"
            )
            trajectory_times[epoch][trajectory] = np.column_stack((starts, ends))
    return {
        "position_offset": position_offset_samples,
        "trajectory_times": trajectory_times,
        "movement_by_epoch": {
            dark_epoch: dark_movement_intervals,
            light_epoch: light_movement_intervals,
        },
        "position_dict": {dark_epoch: dark_values, light_epoch: light_values},
        "timestamps_position_dict": {
            dark_epoch: dark_times,
            light_epoch: light_times,
        },
        "spikes_by_region": {region: spikes},
        "track_graphs_by_side": graphs,
        "edge_orders_by_side": edge_orders,
        "linear_edge_spacing": list(edge_spacing),
        "track_total_length": graph_length_cm,
        "movement_firing_rates_by_region": {
            region: {dark_epoch: dark_firing_rates, light_epoch: light_firing_rates}
        },
    }


def _initial_selected_units(
    identity_rows: Sequence[Mapping[str, Any]],
    *,
    dark_firing_rates: np.ndarray,
    light_firing_rates: np.ndarray,
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Build an all-input-unit table before condition modulation is known."""
    threshold = float(parameters["min_firing_rate_hz"])
    rows = []
    for identity, dark_rate, light_rate in zip(
        identity_rows, dark_firing_rates, light_firing_rates, strict=True
    ):
        rows.append(
            {
                **{name: str(identity[name]) for name in IDENTITY_COLUMNS},
                "input_unit_index": int(identity["input_unit_index"]),
                "dark_movement_firing_rate_hz": float(dark_rate),
                "light_movement_firing_rate_hz": float(light_rate),
                "passes_dark_firing_rate": bool(
                    np.isfinite(dark_rate) and dark_rate >= threshold
                ),
                "passes_light_firing_rate": bool(
                    np.isfinite(light_rate) and light_rate >= threshold
                ),
                "dark_condition_sd_hz": np.nan,
                "light_condition_sd_hz": np.nan,
                "passes_dark_condition_sd": False,
                "passes_light_condition_sd": False,
                "unit_class": "not_computed",
                "included_in_cv_pca": False,
                "unit_qc_status": "not_computed",
            }
        )
    return pd.DataFrame.from_records(rows, columns=SELECTED_UNIT_COLUMNS)


def _firing_rate_candidate_mask(
    table: pd.DataFrame, *, unit_filter_mode: str
) -> np.ndarray:
    """Return the firing-rate-only candidate mask for early terminal handling."""
    dark = table["passes_dark_firing_rate"].to_numpy(dtype=bool)
    light = table["passes_light_firing_rate"].to_numpy(dtype=bool)
    if unit_filter_mode == "shared-active":
        return dark & light
    if unit_filter_mode == "dark-active":
        return dark
    return dark | light


def _annotate_selected_units(
    table: pd.DataFrame,
    *,
    pair_tensors: Any,
    identity_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Add condition modulation and final inclusion to the all-unit audit."""
    output = table.copy()
    selection = pair_tensors.unit_selection
    n_units = len(output)
    for name, values in {
        "dark_active": selection.dark_active,
        "light_active": selection.light_active,
        "dark_modulated": selection.dark_modulated,
        "light_modulated": selection.light_modulated,
        "keep_mask": selection.keep_mask,
        "unit_classes": selection.unit_classes,
    }.items():
        if np.asarray(values).shape != (n_units,):
            raise ValueError(f"Scientific unit-selection field {name} is misaligned.")
    output["passes_dark_firing_rate"] = np.asarray(selection.dark_active, dtype=bool)
    output["passes_light_firing_rate"] = np.asarray(selection.light_active, dtype=bool)
    output["passes_dark_condition_sd"] = np.asarray(
        selection.dark_modulated, dtype=bool
    )
    output["passes_light_condition_sd"] = np.asarray(
        selection.light_modulated, dtype=bool
    )
    output["unit_class"] = np.asarray(selection.unit_classes, dtype=str)
    output["included_in_cv_pca"] = np.asarray(selection.keep_mask, dtype=bool)

    runtime_to_index = {
        str(identity["_group_key"]): int(identity["input_unit_index"])
        for identity in identity_rows
    }
    retained_runtime_ids = np.asarray(pair_tensors.unit_ids).reshape(-1)
    retained_indices: list[int] = []
    for runtime_id in retained_runtime_ids:
        try:
            retained_indices.append(runtime_to_index[str(runtime_id)])
        except KeyError as exc:
            raise ValueError(
                "Scientific output contains an unknown runtime unit identifier."
            ) from exc
    if retained_indices != np.flatnonzero(selection.keep_mask).tolist():
        raise ValueError("Scientific retained units do not follow stable input order.")
    output.loc[retained_indices, "dark_condition_sd_hz"] = np.asarray(
        pair_tensors.dark_condition_sd_hz, dtype=float
    )
    output.loc[retained_indices, "light_condition_sd_hz"] = np.asarray(
        pair_tensors.light_condition_sd_hz, dtype=float
    )
    output["unit_qc_status"] = np.where(
        output["included_in_cv_pca"], "included", "excluded"
    )
    return output.loc[:, SELECTED_UNIT_COLUMNS]


def _terminal_dataset(
    *,
    metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    status: str,
    provenance_json: str,
    position_offset_samples: int,
) -> Any:
    """Build one compact, readable terminal NetCDF dataset."""
    import xarray as xr

    dataset = xr.Dataset(
        data_vars={
            "within_cv_spectrum_signed": (
                ("within_condition", "component"),
                np.empty((2, 0), dtype=np.float32),
            ),
            "within_cv_spectrum_positive": (
                ("within_condition", "component"),
                np.empty((2, 0), dtype=np.float32),
            ),
            "within_cv_cumulative_shared_variance": (
                ("within_condition", "component"),
                np.empty((2, 0), dtype=np.float32),
            ),
            "within_cv_participation_ratio": (
                "within_condition", np.full(2, np.nan, dtype=np.float32)
            ),
            "within_cv_n_components_80": (
                "within_condition", np.full(2, np.nan, dtype=np.float32)
            ),
            "within_cv_n_components_90": (
                "within_condition", np.full(2, np.nan, dtype=np.float32)
            ),
        },
        coords={
            "within_condition": np.asarray(CONDITIONS, dtype=str),
            "component": np.asarray([], dtype=int),
            "unit": np.asarray([], dtype=str),
        },
    )
    dataset.attrs.update(
        _dataset_attrs(
            metadata=metadata,
            parameters=parameters,
            status=status,
            provenance_json=provenance_json,
            position_offset_samples=position_offset_samples,
        )
    )
    return dataset


def _dataset_attrs(
    *,
    metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    status: str,
    provenance_json: str,
    position_offset_samples: int,
) -> dict[str, Any]:
    """Return NetCDF-safe canonical result attributes."""
    return {
        **{name: str(metadata[name]) for name in (
            "cv_pca_id",
            "animal_name",
            "date",
            "region",
            "light_epoch",
            "dark_epoch",
            "parameter_name",
            "parameter_sha256",
        )},
        "random_seed": int(parameters["random_seed"]),
        "analysis_status": str(status),
        "effective_parameters_json": json.dumps(
            dict(parameters), sort_keys=True, separators=(",", ":")
        ),
        "upstream_provenance_json": provenance_json,
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        "position_offset_samples": int(position_offset_samples),
        "result_schema_version": RESULT_SCHEMA_VERSION,
    }


def _stable_identity_dataset(
    dataset: Any,
    *,
    selected_units: pd.DataFrame,
) -> Any:
    """Replace ephemeral retained-unit coordinates with stable identities."""
    kept = selected_units.loc[selected_units["included_in_cv_pca"]].reset_index(drop=True)
    if int(dataset.sizes.get("unit", 0)) != len(kept):
        raise ValueError("Scientific result unit dimension differs from selected units.")
    return dataset.assign_coords(
        unit=("unit", kept["stable_unit_id"].to_numpy(dtype=str)),
        spikesorting_merge_id=(
            "unit", kept["spikesorting_merge_id"].to_numpy(dtype=str)
        ),
        unit_id=("unit", kept["unit_id"].to_numpy(dtype=str)),
        stable_unit_id=("unit", kept["stable_unit_id"].to_numpy(dtype=str)),
        group_unit_id=("unit", kept["group_unit_id"].to_numpy(dtype=str)),
    )


def _summary_from_dataset(dataset: Any) -> pd.DataFrame:
    """Return only the canonical within-condition dimensionality metrics."""
    attrs = dataset.attrs
    rows = []
    for condition in CONDITIONS:
        values = {}
        for variable in (
            "within_cv_participation_ratio",
            "within_cv_n_components_80",
            "within_cv_n_components_90",
        ):
            values[variable] = float(
                dataset[variable].sel(within_condition=condition).values
            )
        rows.append(
            {
                "cv_pca_id": str(attrs["cv_pca_id"]),
                "animal_name": str(attrs["animal_name"]),
                "date": str(attrs["date"]),
                "region": str(attrs["region"]),
                "light_epoch": str(attrs["light_epoch"]),
                "dark_epoch": str(attrs["dark_epoch"]),
                "random_seed": int(attrs["random_seed"]),
                "condition": condition,
                "epoch": str(attrs[f"{condition}_epoch"]),
                "n_units": int(dataset.sizes.get("unit", 0)),
                **values,
                "analysis_status": str(attrs["analysis_status"]),
            }
        )
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def _spectrum_from_dataset(dataset: Any) -> pd.DataFrame:
    """Return the canonical within-condition spectrum table."""
    attrs = dataset.attrs
    components = np.asarray(dataset.coords["component"].values, dtype=int)
    rows = []
    for condition in CONDITIONS:
        arrays = {
            variable: np.asarray(
                dataset[variable].sel(within_condition=condition).values,
                dtype=float,
            )
            for variable in (
                "within_cv_spectrum_signed",
                "within_cv_spectrum_positive",
                "within_cv_cumulative_shared_variance",
            )
        }
        for index, component in enumerate(components):
            rows.append(
                {
                    "cv_pca_id": str(attrs["cv_pca_id"]),
                    "animal_name": str(attrs["animal_name"]),
                    "date": str(attrs["date"]),
                    "region": str(attrs["region"]),
                    "light_epoch": str(attrs["light_epoch"]),
                    "dark_epoch": str(attrs["dark_epoch"]),
                    "random_seed": int(attrs["random_seed"]),
                    "condition": condition,
                    "component": int(component),
                    **{name: float(values[index]) for name, values in arrays.items()},
                    "analysis_status": str(attrs["analysis_status"]),
                }
            )
    return pd.DataFrame.from_records(rows, columns=SPECTRUM_COLUMNS)


def compute_cv_pca(
    *,
    cv_pca_id: Any,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    light_position: Any,
    dark_position: Any,
    light_movement_intervals: Any,
    dark_movement_intervals: Any,
    light_movement_firing_rate_hz: Sequence[float],
    dark_movement_firing_rate_hz: Sequence[float],
    light_trajectory_intervals: Mapping[str, Any],
    dark_trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    upstream_provenance: Mapping[str, Any],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    position_offset_samples: int = DEFAULT_POSITION_OFFSET_SAMPLES,
    bin_size_cm: float = DEFAULT_BIN_SIZE_CM,
    n_groups: int = DEFAULT_N_GROUPS,
    min_occupancy_s: float = DEFAULT_MIN_OCCUPANCY_S,
    unit_filter_mode: str = DEFAULT_UNIT_FILTER_MODE,
    min_firing_rate_hz: float | None = None,
    min_condition_sd_hz: float = DEFAULT_MIN_CONDITION_SD_HZ,
    normalization: str = DEFAULT_NORMALIZATION,
    min_scale: float = DEFAULT_MIN_SCALE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Compute one cvPCA result exclusively from selected NWB-derived inputs."""
    metadata = {
        "cv_pca_id": _uuid_string(cv_pca_id, name="cv_pca_id"),
        "animal_name": _path_component(animal_name, name="animal_name"),
        "date": _path_component(date, name="date"),
        "region": _path_component(str(region).lower(), name="region"),
        "light_epoch": _path_component(light_epoch, name="light_epoch"),
        "dark_epoch": _path_component(dark_epoch, name="dark_epoch"),
        "parameter_name": _path_component(parameter_name, name="parameter_name"),
    }
    if metadata["light_epoch"] == metadata["dark_epoch"]:
        raise ValueError("light_epoch and dark_epoch must differ.")
    parameters = validate_cv_pca_parameters(
        region=metadata["region"],
        bin_size_cm=bin_size_cm,
        n_groups=n_groups,
        min_occupancy_s=min_occupancy_s,
        unit_filter_mode=unit_filter_mode,
        min_firing_rate_hz=min_firing_rate_hz,
        min_condition_sd_hz=min_condition_sd_hz,
        normalization=normalization,
        min_scale=min_scale,
        random_seed=random_seed,
    )
    expected_parameter_sha256 = _provenance_sha256(parameters)
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    elif str(parameter_sha256) != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    metadata["parameter_sha256"] = str(parameter_sha256)
    provenance, provenance_json = _canonical_json_mapping(
        upstream_provenance, name="upstream_provenance"
    )
    position_offset = _positive_integer(
        position_offset_samples, name="position_offset_samples", minimum=0
    )
    identities = _identity_rows(spikes, stable_unit_ids)
    dark_rates = _firing_rate_array(
        dark_movement_firing_rate_hz,
        name="dark_movement_firing_rate_hz",
        n_units=len(identities),
    )
    light_rates = _firing_rate_array(
        light_movement_firing_rate_hz,
        name="light_movement_firing_rate_hz",
        n_units=len(identities),
    )
    dark_trajectories = _validate_trajectory_mapping(
        dark_trajectory_intervals, name="dark_trajectory_intervals"
    )
    light_trajectories = _validate_trajectory_mapping(
        light_trajectory_intervals, name="light_trajectory_intervals"
    )
    selected_graphs, graph_length_cm, edge_spacing = _normalize_graph_inputs(
        graph_inputs
    )
    dark_values, _ = _position_arrays(dark_position, name="dark_position")
    light_values, _ = _position_arrays(light_position, name="light_position")
    if position_offset > len(dark_values) or position_offset > len(light_values):
        raise ValueError("position_offset_samples exceeds a selected position series.")
    dark_movement_duration = _interval_duration(
        dark_movement_intervals, name="dark_movement_intervals"
    )
    light_movement_duration = _interval_duration(
        light_movement_intervals, name="light_movement_intervals"
    )
    selected_units = _initial_selected_units(
        identities,
        dark_firing_rates=dark_rates,
        light_firing_rates=light_rates,
        parameters=parameters,
    )
    trajectory_qc = _make_trajectory_qc(
        dark_epoch=metadata["dark_epoch"],
        light_epoch=metadata["light_epoch"],
        dark_intervals=dark_trajectories,
        light_intervals=light_trajectories,
        dark_movement=dark_movement_intervals,
        light_movement=light_movement_intervals,
        graph_length_cm=graph_length_cm,
        n_groups=parameters["n_groups"],
    )
    lap_assignments = pd.concat(
        [
            _build_lap_assignments(
                epoch=metadata["dark_epoch"],
                condition="dark",
                trajectory_intervals=dark_trajectories,
                n_groups=parameters["n_groups"],
                random_seed=parameters["random_seed"],
            ),
            _build_lap_assignments(
                epoch=metadata["light_epoch"],
                condition="light",
                trajectory_intervals=light_trajectories,
                n_groups=parameters["n_groups"],
                random_seed=parameters["random_seed"],
            ),
        ],
        ignore_index=True,
    ).loc[:, LAP_ASSIGNMENT_COLUMNS]

    terminal_status: str | None = None
    if not identities:
        terminal_status = "no_units"
    elif (
        np.sum(np.all(np.isfinite(dark_values[position_offset:]), axis=1)) < 2
        or np.sum(np.all(np.isfinite(light_values[position_offset:]), axis=1)) < 2
    ):
        terminal_status = "no_valid_position"
    elif dark_movement_duration <= 0.0 or light_movement_duration <= 0.0:
        terminal_status = "no_movement"
    elif trajectory_qc["n_trials"].eq(0).any():
        terminal_status = "no_trials"
    elif trajectory_qc["n_trials"].lt(parameters["n_groups"]).any():
        terminal_status = "insufficient_laps"
    elif not np.any(
        _firing_rate_candidate_mask(
            selected_units, unit_filter_mode=parameters["unit_filter_mode"]
        )
    ):
        terminal_status = "no_eligible_units"

    pair_tensors = None
    if terminal_status is None:
        session = _build_science_session(
            region=metadata["region"],
            spikes=spikes,
            dark_epoch=metadata["dark_epoch"],
            light_epoch=metadata["light_epoch"],
            dark_position=dark_position,
            light_position=light_position,
            dark_movement_intervals=dark_movement_intervals,
            light_movement_intervals=light_movement_intervals,
            dark_trajectory_intervals=dark_trajectories,
            light_trajectory_intervals=light_trajectories,
            dark_firing_rates=dark_rates,
            light_firing_rates=light_rates,
            selected_graphs=selected_graphs,
            graph_length_cm=graph_length_cm,
            edge_spacing=edge_spacing,
            position_offset_samples=position_offset,
        )
        science = _science_module()
        try:
            pair_tensors = science.build_pairwise_tuning_tensors(
                session,
                region=metadata["region"],
                light_epoch=metadata["light_epoch"],
                dark_epoch=metadata["dark_epoch"],
                bin_size_cm=parameters["bin_size_cm"],
                n_groups=parameters["n_groups"],
                min_occupancy_s=parameters["min_occupancy_s"],
                group_seed=parameters["random_seed"],
                min_firing_rate_hz=parameters["min_firing_rate_hz"],
                min_condition_sd_hz=parameters["min_condition_sd_hz"],
                unit_filter_mode=parameters["unit_filter_mode"],
            )
        except ValueError as exc:
            message = str(exc)
            if message.startswith("Too few shared valid bins for "):
                terminal_status = "no_shared_position_bins"
            elif message == (
                "No units remain after applying firing-rate and condition-SD filters."
            ):
                terminal_status = "no_eligible_units"
            else:
                raise

    if terminal_status is None:
        assert pair_tensors is not None
        selected_units = _annotate_selected_units(
            selected_units,
            pair_tensors=pair_tensors,
            identity_rows=identities,
        )
        metrics = science.compute_cv_pca_metrics(
            {"dark": pair_tensors.dark, "light": pair_tensors.light},
            unit_classes=pair_tensors.unit_classes,
            normalization=parameters["normalization"],
            min_scale=parameters["min_scale"],
            save_residual_matrices=False,
        )
        if "residual_matrix" in metrics:
            raise RuntimeError("The compact cvPCA computation created residual matrices.")
        dataset = science.build_result_dataset(
            pair_tensors=pair_tensors,
            metrics=metrics,
            animal_name=metadata["animal_name"],
            date=metadata["date"],
            region=metadata["region"],
            light_epoch=metadata["light_epoch"],
            dark_epoch=metadata["dark_epoch"],
            settings={
                **parameters,
                "n_random_repeats": 1,
            },
        )
        dataset = dataset.drop_vars(
            list(PROHIBITED_RESULT_VARIABLES), errors="ignore"
        )
        dataset = _stable_identity_dataset(dataset, selected_units=selected_units)
        status = "valid"
        valid_bins = pair_tensors.n_valid_bins_by_trajectory
        trajectory_qc["n_shared_valid_bins"] = trajectory_qc[
            "trajectory_type"
        ].map(lambda value: int(valid_bins[str(value)]))
        trajectory_qc["trajectory_status"] = "valid"
    else:
        status = terminal_status
        selected_units["unit_qc_status"] = np.where(
            _firing_rate_candidate_mask(
                selected_units, unit_filter_mode=parameters["unit_filter_mode"]
            ),
            "not_computed",
            "excluded_firing_rate",
        )
        trajectory_qc.loc[
            trajectory_qc["trajectory_status"].eq("pending"),
            "trajectory_status",
        ] = "not_computed"
        dataset = _terminal_dataset(
            metadata=metadata,
            parameters=parameters,
            status=status,
            provenance_json=provenance_json,
            position_offset_samples=position_offset,
        )

    dataset.attrs.update(
        _dataset_attrs(
            metadata=metadata,
            parameters=parameters,
            status=status,
            provenance_json=provenance_json,
            position_offset_samples=position_offset,
        )
    )
    dataset.attrs["artifact_origin"] = "computed"
    dataset.attrs["legacy_artifact_provenance_json"] = "{}"
    summary = _summary_from_dataset(dataset)
    spectrum = _spectrum_from_dataset(dataset)
    result = {
        **metadata,
        "parameters": parameters,
        "upstream_provenance": provenance,
        "position_offset_samples": position_offset,
        "selected_units": selected_units,
        "lap_assignments": lap_assignments,
        "trajectory_qc": trajectory_qc,
        "summary": summary,
        "spectrum": spectrum,
        "dataset": dataset,
        "analysis_status": status,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": {},
    }
    return validate_cv_pca_result(result)


def _validate_table_schema(table: Any, columns: Sequence[str], *, name: str) -> pd.DataFrame:
    """Return a copy of one table with an exact canonical column order."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != tuple(columns):
        raise ValueError(f"{name} does not match its canonical schema.")
    return table.copy().reset_index(drop=True)


def _boolean_column(table: pd.DataFrame, name: str) -> np.ndarray:
    """Return one strictly boolean table column without truth-value coercion."""
    values = table[name].tolist()
    if any(not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{name} must contain only explicit boolean values.")
    return np.asarray(values, dtype=bool)


def _integer_column(
    table: pd.DataFrame, name: str, *, minimum: int = 0
) -> np.ndarray:
    """Return one finite integer-valued table column."""
    values = table[name].to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values < minimum)
        or not np.array_equal(values, np.floor(values))
    ):
        raise ValueError(f"{name} must contain integers >= {minimum}.")
    return values.astype(int)


def _allclose(
    observed: Any,
    expected: Any,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-7,
) -> bool:
    """Return whether two numerical arrays agree, including aligned NaNs."""
    observed_values = np.asarray(observed)
    expected_values = np.asarray(expected)
    return observed_values.shape == expected_values.shape and np.allclose(
        observed_values,
        expected_values,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def _validate_selected_unit_semantics(
    table: pd.DataFrame,
    *,
    parameters: Mapping[str, Any],
    status: str,
) -> np.ndarray:
    """Validate unit-selection flags, classes, and terminal-state semantics."""
    dark_rate = table["dark_movement_firing_rate_hz"].to_numpy(dtype=float)
    light_rate = table["light_movement_firing_rate_hz"].to_numpy(dtype=float)
    dark_sd = table["dark_condition_sd_hz"].to_numpy(dtype=float)
    light_sd = table["light_condition_sd_hz"].to_numpy(dtype=float)
    for name, values in (
        ("dark_movement_firing_rate_hz", dark_rate),
        ("light_movement_firing_rate_hz", light_rate),
        ("dark_condition_sd_hz", dark_sd),
        ("light_condition_sd_hz", light_sd),
    ):
        if np.any(np.isinf(values)) or np.any(values < 0.0):
            raise ValueError(f"selected_units {name} contains invalid values.")

    dark_active = _boolean_column(table, "passes_dark_firing_rate")
    light_active = _boolean_column(table, "passes_light_firing_rate")
    dark_modulated = _boolean_column(table, "passes_dark_condition_sd")
    light_modulated = _boolean_column(table, "passes_light_condition_sd")
    included = _boolean_column(table, "included_in_cv_pca")
    firing_rate_threshold = float(parameters["min_firing_rate_hz"])
    if not np.array_equal(
        dark_active, np.isfinite(dark_rate) & (dark_rate >= firing_rate_threshold)
    ) or not np.array_equal(
        light_active, np.isfinite(light_rate) & (light_rate >= firing_rate_threshold)
    ):
        raise ValueError("Selected-unit firing-rate flags are inconsistent.")

    if status == "valid":
        condition_threshold = float(parameters["min_condition_sd_hz"])
        for values, flags, name in (
            (dark_sd, dark_modulated, "dark"),
            (light_sd, light_modulated, "light"),
        ):
            # The scientific implementation retains condition-SD values only
            # for selected units.  Its flags remain available for every unit.
            if np.any(np.isfinite(values) & ~included):
                raise ValueError(
                    f"Excluded units cannot retain {name} condition-SD values."
                )
            retained_expected = np.isfinite(values) & (
                values >= condition_threshold
            )
            if not np.array_equal(flags[included], retained_expected[included]):
                raise ValueError(
                    f"Selected-unit {name} condition-SD flags are inconsistent."
                )
        dark_usable = dark_active & dark_modulated
        light_usable = light_active & light_modulated
        expected_classes = np.full(
            len(table), "inactive_or_low_mod", dtype=object
        )
        expected_classes[dark_usable & ~light_usable] = "dark_only"
        expected_classes[light_usable & ~dark_usable] = "light_only"
        expected_classes[dark_usable & light_usable] = "shared_active"
        if not np.array_equal(
            table["unit_class"].to_numpy(dtype=str),
            expected_classes.astype(str),
        ):
            raise ValueError("Selected-unit classes are inconsistent with QC flags.")
        filter_mode = str(parameters["unit_filter_mode"])
        if filter_mode == "shared-active":
            expected_included = dark_usable & light_usable
        elif filter_mode == "dark-active":
            expected_included = dark_usable
        else:
            expected_included = dark_usable | light_usable
        if not np.array_equal(included, expected_included):
            raise ValueError("Selected-unit inclusion is inconsistent with parameters.")
        expected_qc = np.where(included, "included", "excluded")
        if not np.array_equal(
            table["unit_qc_status"].to_numpy(dtype=str), expected_qc
        ):
            raise ValueError("Selected-unit QC status is inconsistent.")
    else:
        if np.any(dark_modulated) or np.any(light_modulated):
            raise ValueError("Terminal cvPCA units cannot have condition-SD flags.")
        if not table["unit_class"].astype(str).eq("not_computed").all():
            raise ValueError("Terminal cvPCA unit classes must be not_computed.")
        candidate = _firing_rate_candidate_mask(
            table, unit_filter_mode=str(parameters["unit_filter_mode"])
        )
        expected_qc = np.where(candidate, "not_computed", "excluded_firing_rate")
        if not np.array_equal(
            table["unit_qc_status"].to_numpy(dtype=str), expected_qc
        ):
            raise ValueError("Terminal selected-unit QC status is inconsistent.")
        if np.any(included):
            raise ValueError("Terminal cvPCA results cannot include units.")
    return included


def _validate_trajectory_qc_semantics(
    table: pd.DataFrame,
    *,
    dark_epoch: str,
    light_epoch: str,
    n_groups: int,
    status: str,
) -> None:
    """Validate the fixed eight-row trajectory audit."""
    expected_keys = [
        (condition, epoch, trajectory)
        for condition, epoch in (("dark", dark_epoch), ("light", light_epoch))
        for trajectory in TRAJECTORY_TYPES
    ]
    observed_keys = list(
        zip(
            table["condition"].astype(str),
            table["epoch"].astype(str),
            table["trajectory_type"].astype(str),
            strict=True,
        )
    )
    if observed_keys != expected_keys:
        raise ValueError("trajectory_qc rows are not in canonical epoch/path order.")
    n_trials = _integer_column(table, "n_trials")
    required_groups = _integer_column(table, "n_groups_required", minimum=3)
    n_valid_bins = _integer_column(table, "n_shared_valid_bins")
    if not np.all(required_groups == n_groups):
        raise ValueError("trajectory_qc group counts differ from parameters.")
    support = table["movement_supported_duration_s"].to_numpy(dtype=float)
    graph_length = table["graph_length_cm"].to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(support))
        or np.any(support < 0.0)
        or not np.all(np.isfinite(graph_length))
        or np.any(graph_length <= 0.0)
        or not np.allclose(graph_length, graph_length[0], rtol=1e-9, atol=1e-9)
    ):
        raise ValueError("trajectory_qc support or graph lengths are invalid.")
    observed_status = table["trajectory_status"].to_numpy(dtype=str)
    if status == "valid":
        if (
            np.any(n_trials < n_groups)
            or np.any(support <= 0.0)
            or np.any(n_valid_bins < 2)
            or not np.all(observed_status == "valid")
        ):
            raise ValueError("Valid cvPCA trajectory QC is inconsistent.")
        return
    expected_status = np.full(len(table), "not_computed", dtype=object)
    expected_status[n_trials == 0] = "no_trials"
    expected_status[(n_trials > 0) & (n_trials < n_groups)] = "insufficient_laps"
    expected_status[(n_trials >= n_groups) & (support <= 0.0)] = (
        "no_movement_support"
    )
    if not np.array_equal(observed_status, expected_status.astype(str)):
        raise ValueError("Terminal cvPCA trajectory QC is inconsistent.")
    if np.any(n_valid_bins != 0):
        raise ValueError("Terminal cvPCA trajectories cannot retain valid-bin counts.")


def _validate_lap_assignment_semantics(
    table: pd.DataFrame,
    *,
    trajectory_qc: pd.DataFrame,
    n_groups: int,
) -> None:
    """Validate that every groupable lap occurs exactly once."""
    if table.empty:
        if trajectory_qc["n_trials"].ge(n_groups).any():
            raise ValueError("Lap assignments omit groupable trajectories.")
        return
    allowed_conditions = set(CONDITIONS)
    if not set(table["condition"].astype(str)).issubset(allowed_conditions):
        raise ValueError("Lap assignments contain an unknown condition.")
    if not set(table["trajectory_type"].astype(str)).issubset(
        set(TRAJECTORY_TYPES)
    ):
        raise ValueError("Lap assignments contain an unknown trajectory.")
    starts = table["start_time_s"].to_numpy(dtype=float)
    ends = table["end_time_s"].to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(starts))
        or not np.all(np.isfinite(ends))
        or np.any(ends < starts)
    ):
        raise ValueError("Lap assignment times are invalid.")
    lap_index = _integer_column(table, "lap_index")
    lap_number = _integer_column(table, "lap_number", minimum=1)
    group_index = _integer_column(table, "group_index")
    if not np.array_equal(lap_number, lap_index + 1) or np.any(
        group_index >= n_groups
    ):
        raise ValueError("Lap assignment indices are inconsistent.")
    if table.duplicated(["condition", "trajectory_type", "lap_index"]).any():
        raise ValueError("A lap is assigned more than once.")

    for row in trajectory_qc.itertuples(index=False):
        selected = table.loc[
            table["condition"].astype(str).eq(str(row.condition))
            & table["epoch"].astype(str).eq(str(row.epoch))
            & table["trajectory_type"].astype(str).eq(str(row.trajectory_type))
        ]
        expected_count = int(row.n_trials) if int(row.n_trials) >= n_groups else 0
        if len(selected) != expected_count:
            raise ValueError("Lap assignments do not cover every groupable input lap.")
        if expected_count:
            if set(selected["lap_index"].astype(int)) != set(
                range(expected_count)
            ) or set(selected["group_index"].astype(int)) != set(range(n_groups)):
                raise ValueError("Lap assignment groups or indices are incomplete.")


def _validate_valid_dataset_semantics(
    dataset: Any,
    *,
    selected_units: pd.DataFrame,
    trajectory_qc: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> None:
    """Validate dimensions, stable units, support data, and within spectra."""
    missing = [name for name in VALID_RESULT_SUPPORT_VARIABLES if name not in dataset]
    if missing:
        raise ValueError(f"Valid cvPCA result lacks support variables {missing!r}.")
    expected_dimensions = {
        "score_covariance_by_component": (
            "source_condition",
            "target_condition",
            "source_fold",
            "target_group",
            "component",
        ),
        "test_variance_by_component": (
            "source_condition",
            "target_condition",
            "source_fold",
            "target_group",
            "component",
        ),
        "cumulative_test_variance_captured": (
            "source_condition",
            "target_condition",
            "source_fold",
            "target_group",
            "component",
        ),
        "residual_fraction": (
            "source_condition",
            "target_condition",
            "source_fold",
            "target_group",
            "component",
        ),
        "total_test_variance": (
            "source_condition",
            "target_condition",
            "source_fold",
            "target_group",
        ),
        "valid_projection": (
            "source_condition",
            "target_condition",
            "source_fold",
            "target_group",
        ),
        "within_cv_spectrum_signed": ("within_condition", "component"),
        "within_cv_spectrum_positive": ("within_condition", "component"),
        "within_cv_cumulative_shared_variance": (
            "within_condition",
            "component",
        ),
        "within_cv_participation_ratio": ("within_condition",),
        "within_cv_n_components_80": ("within_condition",),
        "within_cv_n_components_90": ("within_condition",),
        "residualized_light_cutoff_components": ("residualized_component",),
        "residualized_light_cv_spectrum_signed_by_fold": (
            "residualized_component",
            "source_fold",
            "component",
        ),
        "residualized_light_cv_spectrum_signed": (
            "residualized_component",
            "component",
        ),
        "residualized_light_cv_spectrum_positive": (
            "residualized_component",
            "component",
        ),
        "residualized_light_cv_cumulative_shared_variance": (
            "residualized_component",
            "component",
        ),
        "residualized_light_cv_participation_ratio": ("residualized_component",),
        "residualized_light_cv_n_components_80": ("residualized_component",),
        "residualized_light_cv_n_components_90": ("residualized_component",),
        "dark_firing_rate_hz": ("unit",),
        "light_firing_rate_hz": ("unit",),
        "dark_condition_sd_hz": ("unit",),
        "light_condition_sd_hz": ("unit",),
        "unit_class_per_unit": ("unit",),
        "condition_trajectory": ("condition",),
        "condition_bin_center": ("condition",),
        "condition_bin_index": ("condition",),
    }
    for name, dimensions in expected_dimensions.items():
        if tuple(dataset[name].dims) != dimensions:
            raise ValueError(f"cvPCA variable {name!r} has stale dimensions.")
        values = np.asarray(dataset[name].values)
        if values.dtype.kind in "fc" and np.any(np.isinf(values)):
            raise ValueError(f"cvPCA variable {name!r} contains infinity.")

    n_groups = int(parameters["n_groups"])
    if int(dataset.sizes.get("source_fold", -1)) != n_groups or int(
        dataset.sizes.get("target_group", -1)
    ) != n_groups:
        raise ValueError("cvPCA fold dimensions differ from parameters.")
    for coordinate, expected in (
        ("source_condition", CONDITIONS),
        ("target_condition", CONDITIONS),
        ("within_condition", CONDITIONS),
        ("residualized_component", ("pr", "80", "90")),
    ):
        if tuple(np.asarray(dataset.coords[coordinate].values).astype(str)) != expected:
            raise ValueError(f"cvPCA coordinate {coordinate!r} is stale.")
    for coordinate in ("source_fold", "target_group"):
        if not np.array_equal(
            np.asarray(dataset.coords[coordinate].values, dtype=int),
            np.arange(n_groups, dtype=int),
        ):
            raise ValueError(f"cvPCA coordinate {coordinate!r} is stale.")
    n_components = int(dataset.sizes["component"])
    if not np.array_equal(
        np.asarray(dataset.coords["component"].values, dtype=int),
        np.arange(1, n_components + 1, dtype=int),
    ) or n_components != min(int(dataset.sizes["condition"]), int(dataset.sizes["unit"])):
        raise ValueError("cvPCA component coordinate or count is inconsistent.")

    kept = selected_units.loc[
        selected_units["included_in_cv_pca"].to_numpy(dtype=bool)
    ].reset_index(drop=True)
    expected_coordinates = {
        "unit": kept["stable_unit_id"].to_numpy(dtype=str),
        **{
            name: kept[name].to_numpy(dtype=str)
            for name in IDENTITY_COLUMNS
        },
    }
    for name, expected in expected_coordinates.items():
        if name not in dataset.coords or not np.array_equal(
            np.asarray(dataset.coords[name].values).astype(str), expected
        ):
            raise ValueError(f"cvPCA stable unit coordinate {name!r} is inconsistent.")
    for variable, column in (
        ("dark_firing_rate_hz", "dark_movement_firing_rate_hz"),
        ("light_firing_rate_hz", "light_movement_firing_rate_hz"),
        ("dark_condition_sd_hz", "dark_condition_sd_hz"),
        ("light_condition_sd_hz", "light_condition_sd_hz"),
    ):
        if not _allclose(dataset[variable].values, kept[column].to_numpy(dtype=float)):
            raise ValueError(f"cvPCA unit variable {variable!r} is inconsistent.")
    if not np.array_equal(
        np.asarray(dataset["unit_class_per_unit"].values).astype(str),
        kept["unit_class"].to_numpy(dtype=str),
    ):
        raise ValueError("cvPCA unit classes differ from selected_units.")

    valid_bins: dict[str, int] = {}
    for trajectory in TRAJECTORY_TYPES:
        values = trajectory_qc.loc[
            trajectory_qc["trajectory_type"].astype(str).eq(trajectory),
            "n_shared_valid_bins",
        ].to_numpy(dtype=int)
        if values.shape != (2,) or values[0] != values[1]:
            raise ValueError("Shared valid-bin counts must match across conditions.")
        valid_bins[trajectory] = int(values[0])
    try:
        attribute_bins = json.loads(
            str(dataset.attrs["n_valid_bins_by_trajectory_json"])
        )
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("cvPCA valid-bin metadata is malformed.") from exc
    if attribute_bins != valid_bins:
        raise ValueError("cvPCA valid-bin metadata differs from trajectory_qc.")
    expected_trajectory = np.concatenate(
        [np.repeat(trajectory, valid_bins[trajectory]) for trajectory in TRAJECTORY_TYPES]
    )
    observed_trajectory = np.asarray(dataset["condition_trajectory"].values).astype(str)
    if not np.array_equal(observed_trajectory, expected_trajectory):
        raise ValueError("cvPCA condition trajectories are inconsistent.")
    bin_index = np.asarray(dataset["condition_bin_index"].values, dtype=int)
    bin_center = np.asarray(dataset["condition_bin_center"].values, dtype=float)
    if np.any(bin_index < 0) or not np.all(np.isfinite(bin_center)):
        raise ValueError("cvPCA retained-bin metadata is invalid.")
    offset = 0
    for trajectory in TRAJECTORY_TYPES:
        count = valid_bins[trajectory]
        selected_index = bin_index[offset : offset + count]
        selected_center = bin_center[offset : offset + count]
        if np.any(np.diff(selected_index) <= 0) or np.any(np.diff(selected_center) <= 0.0):
            raise ValueError("cvPCA retained bins must increase within each trajectory.")
        offset += count

    score = np.asarray(dataset["score_covariance_by_component"].values, dtype=float)
    signed = np.asarray(dataset["within_cv_spectrum_signed"].values, dtype=float)
    expected_signed = np.full_like(signed, np.nan)
    for condition_index in range(len(CONDITIONS)):
        diagonal = np.stack(
            [
                score[
                    condition_index,
                    condition_index,
                    fold,
                    fold,
                    :,
                ]
                for fold in range(n_groups)
            ],
            axis=0,
        )
        valid_count = np.sum(np.isfinite(diagonal), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected_signed[condition_index] = np.where(
                valid_count > 0,
                np.nansum(diagonal, axis=0) / valid_count,
                np.nan,
            )
    positive = np.clip(expected_signed, 0.0, np.inf)
    cumulative = np.full_like(positive, np.nan)
    participation = np.full(len(CONDITIONS), np.nan, dtype=float)
    n80 = np.full(len(CONDITIONS), np.nan, dtype=float)
    n90 = np.full(len(CONDITIONS), np.nan, dtype=float)
    for index in range(len(CONDITIONS)):
        total = float(np.nansum(positive[index]))
        denominator = float(np.nansum(positive[index] ** 2))
        if denominator > 0.0:
            participation[index] = total**2 / denominator
        if total > 0.0:
            cumulative[index] = np.cumsum(positive[index]) / total
            for fraction, destination in ((0.8, n80), (0.9, n90)):
                reached = np.flatnonzero(
                    np.isfinite(cumulative[index])
                    & (cumulative[index] >= fraction)
                )
                if reached.size:
                    destination[index] = float(reached[0] + 1)
    for variable, expected in (
        ("within_cv_spectrum_signed", expected_signed),
        ("within_cv_spectrum_positive", positive),
        ("within_cv_cumulative_shared_variance", cumulative),
        ("within_cv_participation_ratio", participation),
        ("within_cv_n_components_80", n80),
        ("within_cv_n_components_90", n90),
    ):
        if not _allclose(dataset[variable].values, expected):
            raise ValueError(f"cvPCA within-condition metric {variable!r} is stale.")


def validate_cv_pca_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one in-memory canonical cvPCA result."""
    required = {
        "cv_pca_id",
        "animal_name",
        "date",
        "region",
        "light_epoch",
        "dark_epoch",
        "parameter_name",
        "parameter_sha256",
        "parameters",
        "upstream_provenance",
        "position_offset_samples",
        "selected_units",
        "lap_assignments",
        "trajectory_qc",
        "summary",
        "spectrum",
        "dataset",
        "analysis_status",
        "artifact_origin",
        "legacy_artifact_provenance",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"cvPCA result is missing fields {missing!r}.")
    output = dict(result)
    output["cv_pca_id"] = _uuid_string(output["cv_pca_id"], name="cv_pca_id")
    status = str(output["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Unknown cvPCA analysis_status {status!r}.")
    parameters = validate_cv_pca_parameters(
        region=str(output["region"]), **dict(output["parameters"])
    )
    if _provenance_sha256(parameters) != str(output["parameter_sha256"]):
        raise ValueError("cvPCA parameter digest is stale.")
    provenance, provenance_json = _canonical_json_mapping(
        output["upstream_provenance"], name="upstream_provenance"
    )
    origin = str(output["artifact_origin"])
    if origin not in {"computed", "registered_existing"}:
        raise ValueError("artifact_origin must be computed or registered_existing.")
    if not isinstance(output["legacy_artifact_provenance"], Mapping):
        raise TypeError("legacy_artifact_provenance must be a mapping.")
    legacy = dict(output["legacy_artifact_provenance"])
    if origin == "computed" and legacy:
        raise ValueError("Computed cvPCA results cannot have legacy provenance.")
    if origin == "registered_existing" and not legacy:
        raise ValueError("Registered cvPCA results require legacy provenance.")
    if legacy:
        legacy, _ = _canonical_json_mapping(
            legacy, name="legacy_artifact_provenance"
        )
    position_offset = _positive_integer(
        output["position_offset_samples"],
        name="position_offset_samples",
        minimum=0,
    )
    selected = _validate_table_schema(
        output["selected_units"], SELECTED_UNIT_COLUMNS, name="selected_units"
    )
    laps = _validate_table_schema(
        output["lap_assignments"],
        LAP_ASSIGNMENT_COLUMNS,
        name="lap_assignments",
    )
    trajectory_qc = _validate_table_schema(
        output["trajectory_qc"], TRAJECTORY_QC_COLUMNS, name="trajectory_qc"
    )
    summary = _validate_table_schema(output["summary"], SUMMARY_COLUMNS, name="summary")
    spectrum = _validate_table_schema(
        output["spectrum"], SPECTRUM_COLUMNS, name="spectrum"
    )
    if len(trajectory_qc) != 8 or set(trajectory_qc["condition"]) != set(CONDITIONS):
        raise ValueError("trajectory_qc must contain four rows per condition.")
    if len(summary) != 2 or summary["condition"].tolist() != list(CONDITIONS):
        raise ValueError("summary must contain ordered dark and light rows.")
    for name in IDENTITY_COLUMNS:
        selected[name] = selected[name].map(str)
        if selected[name].eq("").any():
            raise ValueError(f"selected_units has empty {name} values.")
    if selected["stable_unit_id"].duplicated().any() or selected[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError("Selected unit identities must be unique.")
    expected_stable = selected["spikesorting_merge_id"] + ":" + selected["unit_id"]
    if not selected["stable_unit_id"].equals(expected_stable):
        raise ValueError("Selected stable unit identities are inconsistent.")
    expected_indices = np.arange(len(selected), dtype=int)
    if not np.array_equal(
        selected["input_unit_index"].to_numpy(dtype=int), expected_indices
    ):
        raise ValueError("Selected input unit indices must be contiguous.")
    included = _validate_selected_unit_semantics(
        selected, parameters=parameters, status=status
    )
    _validate_trajectory_qc_semantics(
        trajectory_qc,
        dark_epoch=str(output["dark_epoch"]),
        light_epoch=str(output["light_epoch"]),
        n_groups=int(parameters["n_groups"]),
        status=status,
    )
    _validate_lap_assignment_semantics(
        laps,
        trajectory_qc=trajectory_qc,
        n_groups=int(parameters["n_groups"]),
    )
    dataset = output["dataset"]
    for prohibited in PROHIBITED_RESULT_VARIABLES:
        if prohibited in dataset:
            raise ValueError(f"Compact cvPCA result contains prohibited {prohibited!r}.")
    for variable in (
        "within_cv_spectrum_signed",
        "within_cv_spectrum_positive",
        "within_cv_cumulative_shared_variance",
        "within_cv_participation_ratio",
        "within_cv_n_components_80",
        "within_cv_n_components_90",
    ):
        if variable not in dataset:
            raise ValueError(f"cvPCA dataset is missing {variable!r}.")
    if tuple(np.asarray(dataset.coords["within_condition"].values).astype(str)) != CONDITIONS:
        raise ValueError("cvPCA condition coordinate order is stale.")
    if int(dataset.sizes.get("unit", 0)) != int(np.sum(included)):
        raise ValueError("cvPCA selected-unit count differs from its dataset.")
    if status == "valid":
        missing_science = [
            name for name in SCIENTIFIC_RESULT_VARIABLES if name not in dataset
        ]
        if missing_science:
            raise ValueError(f"Valid cvPCA result lacks variables {missing_science!r}.")
        if not np.any(included) or dataset.sizes.get("component", 0) < 1:
            raise ValueError("Valid cvPCA results require units and components.")
        if not selected.loc[included, "unit_qc_status"].eq("included").all():
            raise ValueError("Included cvPCA units need included QC status.")
        if len(laps) != int(trajectory_qc["n_trials"].sum()):
            raise ValueError("Lap assignments must include every input lap.")
        if not trajectory_qc["trajectory_status"].eq("valid").all():
            raise ValueError("Valid cvPCA trajectory rows must all be valid.")
        _validate_valid_dataset_semantics(
            dataset,
            selected_units=selected,
            trajectory_qc=trajectory_qc,
            parameters=parameters,
        )
    else:
        if np.any(included) or dataset.sizes.get("component", 0) != 0:
            raise ValueError("Terminal cvPCA results cannot retain units or components.")
        metric_values = summary[
            [
                "within_cv_participation_ratio",
                "within_cv_n_components_80",
                "within_cv_n_components_90",
            ]
        ].to_numpy(dtype=float)
        if not np.isnan(metric_values).all() or not spectrum.empty:
            raise ValueError("Terminal cvPCA summaries must contain only NaN metrics.")
    attrs = dataset.attrs
    for name in (
        "cv_pca_id",
        "animal_name",
        "date",
        "region",
        "light_epoch",
        "dark_epoch",
        "parameter_name",
        "parameter_sha256",
        "analysis_status",
        "artifact_origin",
    ):
        if str(attrs.get(name, "")) != str(output[name]):
            raise ValueError(f"cvPCA dataset attribute {name!r} is inconsistent.")
    if int(attrs.get("random_seed", -1)) != parameters["random_seed"]:
        raise ValueError("cvPCA dataset random seed is inconsistent.")
    try:
        effective_parameters = json.loads(
            str(attrs.get("effective_parameters_json", ""))
        )
    except json.JSONDecodeError as exc:
        raise ValueError("cvPCA dataset parameters are malformed.") from exc
    if effective_parameters != parameters:
        raise ValueError("cvPCA dataset effective parameters are inconsistent.")
    if int(attrs.get("position_offset_samples", -1)) != position_offset:
        raise ValueError("cvPCA dataset position offset is inconsistent.")
    if str(attrs.get("result_schema_version", "")) != RESULT_SCHEMA_VERSION:
        raise ValueError("cvPCA result schema version is stale.")
    if attrs.get("output_rule_sha256") != OUTPUT_RULE_SHA256:
        raise ValueError("cvPCA output-rule digest is stale.")
    if str(attrs.get("upstream_provenance_json")) != provenance_json:
        raise ValueError("cvPCA dataset upstream provenance is inconsistent.")
    if json.loads(str(attrs.get("legacy_artifact_provenance_json", "{}"))) != legacy:
        raise ValueError("cvPCA dataset legacy provenance is inconsistent.")
    expected_summary = _summary_from_dataset(dataset)
    pd.testing.assert_frame_equal(summary, expected_summary, check_dtype=False)
    expected_spectrum = _spectrum_from_dataset(dataset)
    pd.testing.assert_frame_equal(spectrum, expected_spectrum, check_dtype=False)
    output.update(
        {
            "parameters": parameters,
            "upstream_provenance": provenance,
            "selected_units": selected,
            "lap_assignments": laps,
            "trajectory_qc": trajectory_qc,
            "summary": summary,
            "spectrum": spectrum,
            "analysis_status": status,
            "artifact_origin": origin,
            "legacy_artifact_provenance": legacy,
            "position_offset_samples": position_offset,
            "n_input_units": len(selected),
            "n_selected_units": int(np.sum(included)),
        }
    )
    return output


def _manifest_for_directory(result: Mapping[str, Any], directory: Path) -> pd.DataFrame:
    """Build the six-row immutable bundle manifest."""
    artifacts = {
        "cv_pca": (RESULT_FILENAME, "netcdf"),
        "summary": (SUMMARY_FILENAME, "parquet"),
        "within_spectrum": (SPECTRUM_FILENAME, "parquet"),
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "lap_assignments": (LAP_ASSIGNMENTS_FILENAME, "parquet"),
        "trajectory_qc": (TRAJECTORY_QC_FILENAME, "parquet"),
    }
    provenance_json = json.dumps(
        result["upstream_provenance"], sort_keys=True, separators=(",", ":")
    )
    legacy_json = json.dumps(
        result["legacy_artifact_provenance"],
        sort_keys=True,
        separators=(",", ":"),
    )
    rows = []
    for key, (filename, kind) in artifacts.items():
        path = Path(directory) / filename
        rows.append(
            {
                "artifact_key": key,
                "relative_path": filename,
                "artifact_kind": kind,
                "file_size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
                "cv_pca_id": result["cv_pca_id"],
                "animal_name": result["animal_name"],
                "date": result["date"],
                "region": result["region"],
                "light_epoch": result["light_epoch"],
                "dark_epoch": result["dark_epoch"],
                "random_seed": result["parameters"]["random_seed"],
                "parameter_name": result["parameter_name"],
                "parameter_sha256": result["parameter_sha256"],
                "output_rule_sha256": OUTPUT_RULE_SHA256,
                "upstream_provenance_json": provenance_json,
                "analysis_status": result["analysis_status"],
                "artifact_origin": result["artifact_origin"],
                "legacy_artifact_provenance_json": legacy_json,
                "n_input_units": result["n_input_units"],
                "n_selected_units": result["n_selected_units"],
                "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            }
        )
    return pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS)


def _write_bundle_files(result: Mapping[str, Any], directory: Path) -> None:
    """Write all canonical scientific files and their manifest."""
    directory.mkdir(parents=True, exist_ok=False)
    result["dataset"].to_netcdf(directory / RESULT_FILENAME)
    result["summary"].to_parquet(directory / SUMMARY_FILENAME, index=False)
    result["spectrum"].to_parquet(directory / SPECTRUM_FILENAME, index=False)
    result["selected_units"].to_parquet(
        directory / SELECTED_UNITS_FILENAME, index=False
    )
    result["lap_assignments"].to_parquet(
        directory / LAP_ASSIGNMENTS_FILENAME, index=False
    )
    result["trajectory_qc"].to_parquet(
        directory / TRAJECTORY_QC_FILENAME, index=False
    )
    _manifest_for_directory(result, directory).to_parquet(
        directory / MANIFEST_FILENAME, index=False
    )


def write_cv_pca_artifact(
    result: Mapping[str, Any],
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write and reload one complete immutable cvPCA bundle."""
    validated = validate_cv_pca_result(result)
    paths = get_cv_pca_artifact_paths(
        animal_name=validated["animal_name"],
        date=validated["date"],
        light_epoch=validated["light_epoch"],
        dark_epoch=validated["dark_epoch"],
        region=validated["region"],
        cv_pca_id=validated["cv_pca_id"],
        artifact_root=artifact_root,
    )
    destination = paths["artifact_dir"]
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite cvPCA artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.backup")
    had_existing = destination.exists()
    installed_destination = False
    try:
        _write_bundle_files(validated, temporary)
        load_cv_pca_artifact(temporary, _allow_temporary_name=True)
        if had_existing:
            os.replace(destination, backup)
        os.replace(temporary, destination)
        installed_destination = True
        loaded = load_cv_pca_artifact(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        elif installed_destination and destination.exists():
            shutil.rmtree(destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    return {**loaded, "artifact_paths": paths, "_created_artifact_paths": [str(destination)]}


def load_cv_pca_artifact(
    artifact_dir: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load and checksum-validate one complete cvPCA bundle."""
    directory = Path(artifact_dir)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cvPCA manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or len(manifest) != 6:
        raise ValueError("cvPCA manifest does not have its canonical schema.")
    expected = {
        "cv_pca": (RESULT_FILENAME, "netcdf"),
        "summary": (SUMMARY_FILENAME, "parquet"),
        "within_spectrum": (SPECTRUM_FILENAME, "parquet"),
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "lap_assignments": (LAP_ASSIGNMENTS_FILENAME, "parquet"),
        "trajectory_qc": (TRAJECTORY_QC_FILENAME, "parquet"),
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("cvPCA manifest lacks canonical artifacts.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("cvPCA manifest contains stale artifact names.")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"cvPCA artifact not found: {path}")
        if path.stat().st_size != int(row["file_size_bytes"]) or _file_sha256(
            path
        ) != str(row["sha256"]):
            raise ValueError(f"cvPCA checksum mismatch: {path}")
    first = manifest.iloc[0]
    for name in MANIFEST_COLUMNS[5:]:
        if not manifest[name].astype(str).eq(str(first[name])).all():
            raise ValueError(f"cvPCA manifest has inconsistent {name!r}.")
    if str(first["bundle_schema_version"]) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("cvPCA bundle schema version is stale.")
    if not _allow_temporary_name and directory.name != str(first["cv_pca_id"]):
        raise ValueError("cvPCA artifact directory name does not match cv_pca_id.")
    import xarray as xr

    with xr.open_dataset(directory / RESULT_FILENAME) as opened:
        dataset = opened.load()
    try:
        parameters = json.loads(str(dataset.attrs["effective_parameters_json"]))
        provenance = json.loads(str(first["upstream_provenance_json"]))
        legacy = json.loads(str(first["legacy_artifact_provenance_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError("cvPCA artifact contains malformed provenance.") from exc
    result = validate_cv_pca_result(
        {
            "cv_pca_id": str(first["cv_pca_id"]),
            "animal_name": str(first["animal_name"]),
            "date": str(first["date"]),
            "region": str(first["region"]),
            "light_epoch": str(first["light_epoch"]),
            "dark_epoch": str(first["dark_epoch"]),
            "parameter_name": str(first["parameter_name"]),
            "parameter_sha256": str(first["parameter_sha256"]),
            "parameters": parameters,
            "upstream_provenance": provenance,
            "position_offset_samples": int(dataset.attrs["position_offset_samples"]),
            "selected_units": pd.read_parquet(directory / SELECTED_UNITS_FILENAME),
            "lap_assignments": pd.read_parquet(directory / LAP_ASSIGNMENTS_FILENAME),
            "trajectory_qc": pd.read_parquet(directory / TRAJECTORY_QC_FILENAME),
            "summary": pd.read_parquet(directory / SUMMARY_FILENAME),
            "spectrum": pd.read_parquet(directory / SPECTRUM_FILENAME),
            "dataset": dataset,
            "analysis_status": str(first["analysis_status"]),
            "artifact_origin": str(first["artifact_origin"]),
            "legacy_artifact_provenance": legacy,
        }
    )
    if result["n_input_units"] != int(first["n_input_units"]) or result[
        "n_selected_units"
    ] != int(first["n_selected_units"]):
        raise ValueError("cvPCA manifest unit counts differ from the bundle.")
    return {**result, "manifest": manifest}


def _load_compact_legacy_dataset(path: Path) -> Any:
    """Read only canonical scientific variables from a potentially huge legacy file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy cvPCA NetCDF not found: {source}")
    import xarray as xr

    try:
        with xr.open_dataset(source) as opened:
            missing = [name for name in SCIENTIFIC_RESULT_VARIABLES if name not in opened]
            if missing:
                raise ValueError(f"Legacy cvPCA NetCDF lacks variables {missing!r}.")
            compact = opened[list(SCIENTIFIC_RESULT_VARIABLES)].load()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Legacy cvPCA NetCDF is unreadable or incomplete: {source}") from exc
    return compact


def _validate_legacy_summary(path: Path, dataset: Any) -> pd.DataFrame:
    """Require the complete four-direction legacy summary and dataset agreement."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy cvPCA summary not found: {source}")
    table = pd.read_parquet(source)
    required = {
        "animal_name",
        "date",
        "region",
        "dark_epoch",
        "light_epoch",
        "unit_filter_mode",
        "normalization",
        "source_condition",
        "target_condition",
        "projection_direction",
        "is_cross_condition",
        "n_units",
        "source_cv_participation_ratio",
        "source_n_components_80",
        "source_n_components_90",
        "min_firing_rate_hz",
        "min_condition_sd_hz",
        "bin_size_cm",
        "n_groups",
        "min_occupancy_s",
    }
    if not required.issubset(table.columns) or len(table) != 4:
        raise ValueError("Legacy cvPCA summary is incomplete.")
    expected_directions = {(source, target) for source in CONDITIONS for target in CONDITIONS}
    observed_directions = set(
        zip(
            table["source_condition"].astype(str),
            table["target_condition"].astype(str),
            strict=True,
        )
    )
    if observed_directions != expected_directions:
        raise ValueError("Legacy cvPCA summary lacks a projection direction.")
    attrs = dataset.attrs
    for column, attribute in (
        ("animal_name", "animal_name"),
        ("date", "date"),
        ("region", "region"),
        ("dark_epoch", "dark_epoch"),
        ("light_epoch", "light_epoch"),
        ("unit_filter_mode", "unit_filter_mode"),
        ("normalization", "normalization"),
    ):
        if not table[column].astype(str).eq(str(attrs.get(attribute, ""))).all():
            raise ValueError(
                f"Legacy cvPCA summary {column!r} disagrees with its NetCDF."
            )
    for column, attribute in (
        ("n_units", "n_units"),
        ("n_groups", "n_groups"),
    ):
        values = table[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(
            values == float(attrs.get(attribute, np.nan))
        ):
            raise ValueError(
                f"Legacy cvPCA summary {column!r} disagrees with its NetCDF."
            )
    for column, attribute in (
        ("min_firing_rate_hz", "min_firing_rate_hz"),
        ("min_condition_sd_hz", "min_condition_sd_hz"),
        ("bin_size_cm", "bin_size_cm"),
        ("min_occupancy_s", "min_occupancy_s"),
    ):
        if not np.allclose(
            table[column].to_numpy(dtype=float),
            float(attrs.get(attribute, np.nan)),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                f"Legacy cvPCA summary {column!r} disagrees with its NetCDF."
            )
    if "random_seed" in table and not table["random_seed"].astype(int).eq(
        int(attrs.get("random_seed", -1))
    ).all():
        raise ValueError("Legacy cvPCA summary random_seed disagrees with its NetCDF.")
    expected_projection = (
        table["source_condition"].astype(str)
        + "_to_"
        + table["target_condition"].astype(str)
    )
    expected_cross = ~table["source_condition"].astype(str).eq(
        table["target_condition"].astype(str)
    )
    if not table["projection_direction"].astype(str).equals(expected_projection):
        raise ValueError("Legacy cvPCA summary projection labels are inconsistent.")
    if not np.array_equal(
        table["is_cross_condition"].to_numpy(dtype=bool),
        expected_cross.to_numpy(dtype=bool),
    ):
        raise ValueError("Legacy cvPCA summary cross-condition flags are inconsistent.")
    for condition in CONDITIONS:
        selected = table.loc[table["source_condition"].astype(str).eq(condition)]
        for column, variable in (
            ("source_cv_participation_ratio", "within_cv_participation_ratio"),
            ("source_n_components_80", "within_cv_n_components_80"),
            ("source_n_components_90", "within_cv_n_components_90"),
        ):
            values = selected[column].to_numpy(dtype=float)
            expected = float(dataset[variable].sel(within_condition=condition).values)
            if not np.allclose(values, expected, rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError("Legacy cvPCA summary disagrees with its NetCDF.")
    return table


def _compare_scientific_datasets(observed: Any, expected: Any) -> None:
    """Require every canonical scientific array to match an NWB recomputation."""
    for name in SCIENTIFIC_RESULT_VARIABLES:
        if name not in expected:
            raise ValueError(f"Recomputed cvPCA result lacks {name!r}.")
        if tuple(observed[name].dims) != tuple(expected[name].dims):
            raise ValueError(f"Legacy cvPCA {name} dimensions differ from recomputation.")
        for dimension in observed[name].dims:
            if dimension not in observed.coords or dimension not in expected.coords:
                raise ValueError(
                    f"Legacy cvPCA {name} lacks its {dimension!r} coordinate."
                )
            observed_coordinate = np.asarray(observed.coords[dimension].values)
            expected_coordinate = np.asarray(expected.coords[dimension].values)
            if observed_coordinate.dtype.kind in "OUS" or expected_coordinate.dtype.kind in "OUS":
                coordinate_matches = np.array_equal(
                    observed_coordinate.astype(str), expected_coordinate.astype(str)
                )
            else:
                coordinate_matches = _allclose(
                    observed_coordinate,
                    expected_coordinate,
                    rtol=0.0,
                    atol=0.0,
                )
            if not coordinate_matches:
                raise ValueError(
                    f"Legacy cvPCA {dimension!r} coordinate differs from recomputation."
                )
        observed_values = np.asarray(observed[name].values)
        expected_values = np.asarray(expected[name].values)
        if observed_values.shape != expected_values.shape:
            raise ValueError(f"Legacy cvPCA {name} shape differs from recomputation.")
        if observed_values.dtype.kind == "b" or expected_values.dtype.kind == "b":
            matches = np.array_equal(observed_values, expected_values)
        else:
            matches = np.allclose(
                observed_values,
                expected_values,
                rtol=1e-7,
                atol=1e-9,
                equal_nan=True,
            )
        if not matches:
            raise ValueError(f"Legacy cvPCA {name} differs from exact NWB recomputation.")


def _validate_legacy_dataset_metadata(observed: Any, expected: Any) -> None:
    """Require legacy identity, parameters, and audit metadata to match NWB data."""
    observed_attrs = observed.attrs
    expected_attrs = expected.attrs
    for name in (
        "animal_name",
        "date",
        "region",
        "light_epoch",
        "dark_epoch",
        "script",
        "normalization",
        "unit_filter_mode",
    ):
        if str(observed_attrs.get(name, "")) != str(expected_attrs.get(name, "")):
            raise ValueError(f"Legacy cvPCA metadata {name!r} differs from recomputation.")
    for name in (
        "n_groups",
        "n_conditions",
        "n_units",
        "n_components",
        "random_seed",
    ):
        try:
            observed_value = int(observed_attrs[name])
            expected_value = int(expected_attrs[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Legacy cvPCA metadata {name!r} is malformed.") from exc
        if observed_value != expected_value:
            raise ValueError(f"Legacy cvPCA metadata {name!r} differs from recomputation.")
    for name in (
        "min_firing_rate_hz",
        "min_condition_sd_hz",
        "bin_size_cm",
        "min_occupancy_s",
    ):
        try:
            observed_value = float(observed_attrs[name])
            expected_value = float(expected_attrs[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Legacy cvPCA metadata {name!r} is malformed.") from exc
        if not np.isclose(
            observed_value, expected_value, rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"Legacy cvPCA metadata {name!r} differs from recomputation.")
    for name in (
        "n_valid_bins_by_trajectory_json",
        "unit_class_counts_kept_json",
        "unit_class_counts_all_json",
    ):
        try:
            observed_value = json.loads(str(observed_attrs[name]))
            expected_value = json.loads(str(expected_attrs[name]))
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Legacy cvPCA metadata {name!r} is malformed.") from exc
        if observed_value != expected_value:
            raise ValueError(f"Legacy cvPCA metadata {name!r} differs from recomputation.")


def register_existing_cv_pca_artifact(
    *,
    legacy_result_path: Path,
    legacy_summary_path: Path,
    compute_inputs: Mapping[str, Any],
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Strictly verify, compact, and register one legacy cvPCA result pair."""
    observed = _load_compact_legacy_dataset(legacy_result_path)
    _validate_legacy_summary(legacy_summary_path, observed)
    recomputed = compute_cv_pca(**dict(compute_inputs))
    if recomputed["analysis_status"] != "valid":
        raise ValueError("A legacy cvPCA result cannot register against a terminal recomputation.")
    _validate_legacy_dataset_metadata(observed, recomputed["dataset"])
    _compare_scientific_datasets(observed, recomputed["dataset"])

    # Preserve the verified legacy numerical arrays while taking stable identities,
    # QC, lap assignments, and provenance from the exact NWB recomputation.
    dataset = recomputed["dataset"].copy(deep=True)
    for name in SCIENTIFIC_RESULT_VARIABLES:
        dataset[name].data = np.asarray(observed[name].values).copy()
    legacy_provenance = {
        "legacy_result_path": str(Path(legacy_result_path).resolve(strict=True)),
        "legacy_result_sha256": _file_sha256(Path(legacy_result_path)),
        "legacy_summary_path": str(Path(legacy_summary_path).resolve(strict=True)),
        "legacy_summary_sha256": _file_sha256(Path(legacy_summary_path)),
        "verification": "exact_nwb_recomputation",
        "excluded_legacy_variables": list(PROHIBITED_RESULT_VARIABLES),
    }
    dataset.attrs["artifact_origin"] = "registered_existing"
    dataset.attrs["legacy_artifact_provenance_json"] = json.dumps(
        legacy_provenance, sort_keys=True, separators=(",", ":")
    )
    registered = {
        **recomputed,
        "dataset": dataset,
        "summary": _summary_from_dataset(dataset),
        "spectrum": _spectrum_from_dataset(dataset),
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": legacy_provenance,
    }
    registered = validate_cv_pca_result(registered)
    return write_cv_pca_artifact(
        registered, artifact_root=artifact_root, overwrite=overwrite
    )


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "BUNDLE_SCHEMA_VERSION",
    "CONDITIONS",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_BIN_SIZE_CM",
    "DEFAULT_MIN_CONDITION_SD_HZ",
    "DEFAULT_MIN_OCCUPANCY_S",
    "DEFAULT_MIN_SCALE",
    "DEFAULT_N_GROUPS",
    "DEFAULT_NORMALIZATION",
    "DEFAULT_POSITION_OFFSET_SAMPLES",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_REGION_MIN_FIRING_RATE_HZ",
    "DEFAULT_UNIT_FILTER_MODE",
    "LAP_ASSIGNMENT_COLUMNS",
    "MANIFEST_COLUMNS",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "PROHIBITED_RESULT_VARIABLES",
    "SCIENTIFIC_RESULT_VARIABLES",
    "SELECTED_UNIT_COLUMNS",
    "SPECTRUM_COLUMNS",
    "SUMMARY_COLUMNS",
    "TRAJECTORY_QC_COLUMNS",
    "TRAJECTORY_TYPES",
    "compute_cv_pca",
    "get_cv_pca_artifact_paths",
    "get_legacy_cv_pca_paths",
    "load_cv_pca_artifact",
    "register_existing_cv_pca_artifact",
    "validate_cv_pca_parameters",
    "validate_cv_pca_result",
    "write_cv_pca_artifact",
]
