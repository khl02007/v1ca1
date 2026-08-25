"""Database-free dark/light GLM computation and artifact bundles.

The current analysis contract (schema v5) chooses place bases from spatial-bin
sizes.  Manuscript-era artifacts (schema v4) instead chose explicit spline
counts.  This module keeps those contracts distinct through
``basis_candidate_mode`` and never relabels one schema as the other.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import gc
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
ARTIFACT_DIRNAME = "dark_light_glm"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
SELECTION_SUMMARY_FILENAME = "selection_summary.nc"
CANDIDATE_DIRNAME = "candidates"
SELECTED_DIRNAME = "selected"

NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_SELECTED_UNITS_TABLE_NAME = "dark_light_glm_selected_units"
NWB_DATASET_INDEX_TABLE_NAME = "dark_light_glm_dataset_index"
NWB_AXES_TABLE_NAME = "dark_light_glm_axes"
NWB_CANDIDATE_RESULTS_TABLE_NAME = "dark_light_glm_candidate_results"
NWB_SELECTED_RESULTS_TABLE_NAME = "dark_light_glm_selected_results"
NWB_SELECTION_SUMMARY_TABLE_NAME = "dark_light_glm_selection_summary"
NWB_PROVENANCE_TABLE_NAME = "dark_light_glm_provenance"

MODEL_NAMES = (
    "visual",
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
)
BASIS_CANDIDATE_MODES = ("spatial_bin_size_cm", "n_splines")
SCHEMA_VERSION_BY_MODE = MappingProxyType(
    {"spatial_bin_size_cm": "5", "n_splines": "4"}
)
DEFAULT_BIN_SIZES_S = (0.02, 0.05)
DEFAULT_SPATIAL_BIN_SIZES_CM = (2.0, 4.0, 8.0)
LEGACY_N_SPLINES = (25, 40, 60)
DEFAULT_RIDGES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
DEFAULT_N_FOLDS = 5
DEFAULT_RANDOM_SEED = 47
DEFAULT_SPLINE_ORDER = 4
DEFAULT_USE_SPEED = True
DEFAULT_SPEED_FEATURE_MODE = "linear"
DEFAULT_N_SPLINES_SPEED = 5
DEFAULT_SPLINE_ORDER_SPEED = 4
DEFAULT_SPEED_SMOOTHING_SIGMA_S = 0.1
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_units",
    "no_eligible_units",
    "no_valid_position",
    "no_movement",
    "no_valid_units",
)

CURRENT_PARAMETERS = MappingProxyType(
    {
        "basis_candidate_mode": "spatial_bin_size_cm",
        "basis_candidates": DEFAULT_SPATIAL_BIN_SIZES_CM,
        "bin_sizes_s": DEFAULT_BIN_SIZES_S,
        "ridges": DEFAULT_RIDGES,
        "n_folds": DEFAULT_N_FOLDS,
        "random_seed": DEFAULT_RANDOM_SEED,
        "spline_order": DEFAULT_SPLINE_ORDER,
        "use_speed": DEFAULT_USE_SPEED,
        "speed_feature_mode": DEFAULT_SPEED_FEATURE_MODE,
        "n_splines_speed": DEFAULT_N_SPLINES_SPEED,
        "spline_order_speed": DEFAULT_SPLINE_ORDER_SPEED,
        "speed_smoothing_sigma_s": DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    }
)
LEGACY_PARAMETERS = MappingProxyType(
    {
        **dict(CURRENT_PARAMETERS),
        "basis_candidate_mode": "n_splines",
        "basis_candidates": LEGACY_N_SPLINES,
    }
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
    "dark_movement_firing_rate_hz",
    "light_movement_firing_rate_hz",
    "n_selected_model_trajectory_fits",
    "valid_glm_fit",
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "dark_light_glm_id",
    "animal_name",
    "date",
    "region",
    "light_epoch",
    "dark_epoch",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "basis_candidate_mode",
    "basis_candidates_json",
    "schema_version",
    "n_candidates",
    "n_selected_models",
    "n_units",
    "selected_units_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
)

DATASET_INDEX_COLUMNS = (
    "dataset_key",
    "fit_stage",
    "model_name",
    "attrs_json",
)
ARRAY_COMPONENT_COLUMNS = (
    "dataset_key",
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
    "trajectory_length_cm",
    "segment_edges",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "artifact_schema_version",
)


def _analysis_module() -> Any:
    """Import the existing fit implementation only when computation is requested."""
    from v1ca1.task_progression import dark_light_glm

    return dark_light_glm


def _clear_jax_fit_caches() -> None:
    """Best-effort cleanup of JAX caches between large candidate fits."""
    try:
        import jax
    except ModuleNotFoundError:
        pass
    else:
        jax.clear_caches()
    gc.collect()


class _IsolatingPopulationGLM:
    """PopulationGLM-compatible adapter with per-unit failure isolation."""

    def __init__(self, *_args: Any, regularizer_strength: float, **_kwargs: Any):
        self._ridge = float(regularizer_strength)

    def fit(self, design: np.ndarray, response: np.ndarray) -> Any:
        """Fit via the shared motor helper and expose the expected attributes."""
        from v1ca1.task_progression.motor import (
            fit_population_glm_isolating_unit_failures,
        )

        fitted = fit_population_glm_isolating_unit_failures(
            design,
            response,
            ridge=self._ridge,
        )
        self.coef_ = fitted.coef_
        self.intercept_ = fitted.intercept_
        return self


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


def _float_token(value: float) -> str:
    """Return one compact path-safe floating-point token."""
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _candidate_key(
    model_name: str,
    *,
    bin_size_s: float,
    basis_candidate_mode: str,
    basis_value: float | int,
) -> str:
    """Return the canonical key and filename stem for one candidate dataset."""
    if basis_candidate_mode == "spatial_bin_size_cm":
        basis_token = f"spatial_bin_{_float_token(float(basis_value))}cm"
    else:
        basis_token = f"n_splines_{int(basis_value)}"
    return f"{model_name}__bin_{_float_token(bin_size_s)}s__{basis_token}"


def get_dark_light_glm_artifact_paths(
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    dark_light_glm_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Any]:
    """Return one UUID-keyed, session-first coupled dark/light bundle."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
        }.items()
    }
    result_id = _uuid_string(dark_light_glm_id, name="dark_light_glm_id")
    pair_name = f"{components['light_epoch']}_vs_{components['dark_epoch']}"
    artifact_dir = (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / pair_name
        / components["region"]
        / result_id
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "selected_units_path": artifact_dir / SELECTED_UNITS_FILENAME,
        "candidate_dir": artifact_dir / CANDIDATE_DIRNAME,
        "selected_dir": artifact_dir / SELECTED_DIRNAME,
        "selection_summary_path": artifact_dir / SELECTION_SUMMARY_FILENAME,
        "selected_model_paths": {
            model_name: artifact_dir / SELECTED_DIRNAME / f"{model_name}.nc"
            for model_name in MODEL_NAMES
        },
    }


def _positive_unique_floats(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    """Return ordered, unique positive finite floating-point values."""
    normalized = tuple(dict.fromkeys(float(value) for value in values))
    if not normalized or any(not np.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError(f"{name} must contain positive finite values.")
    return normalized


def validate_dark_light_glm_parameters(
    *,
    basis_candidate_mode: str,
    basis_candidates: Sequence[float | int],
    bin_sizes_s: Sequence[float] = DEFAULT_BIN_SIZES_S,
    ridges: Sequence[float] = DEFAULT_RIDGES,
    n_folds: int = DEFAULT_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    spline_order: int = DEFAULT_SPLINE_ORDER,
    min_dark_firing_rate_hz: float = 0.0,
    min_light_firing_rate_hz: float = 0.0,
    use_speed: bool = DEFAULT_USE_SPEED,
    speed_feature_mode: str = DEFAULT_SPEED_FEATURE_MODE,
    n_splines_speed: int = DEFAULT_N_SPLINES_SPEED,
    spline_order_speed: int = DEFAULT_SPLINE_ORDER_SPEED,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    speed_bounds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Validate the explicit schema mode and effective fit parameters."""
    mode = str(basis_candidate_mode)
    if mode not in BASIS_CANDIDATE_MODES:
        raise ValueError(f"basis_candidate_mode must be one of {BASIS_CANDIDATE_MODES!r}.")
    if mode == "spatial_bin_size_cm":
        normalized_basis: tuple[float | int, ...] = _positive_unique_floats(
            basis_candidates,
            name="basis_candidates",
        )
    else:
        candidate_floats = _positive_unique_floats(
            basis_candidates,
            name="basis_candidates",
        )
        if any(not float(value).is_integer() for value in candidate_floats):
            raise ValueError("n_splines basis candidates must be positive integers.")
        normalized_basis = tuple(int(value) for value in candidate_floats)
    bins = _positive_unique_floats(bin_sizes_s, name="bin_sizes_s")
    ridge_values = _positive_unique_floats(ridges, name="ridges")
    n_folds = int(n_folds)
    spline_order = int(spline_order)
    n_splines_speed = int(n_splines_speed)
    spline_order_speed = int(spline_order_speed)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if spline_order < 1:
        raise ValueError("spline_order must be positive.")
    if mode == "n_splines" and any(
        int(value) < spline_order for value in normalized_basis
    ):
        raise ValueError("Every n_splines candidate must be at least spline_order.")
    if n_splines_speed < 1 or spline_order_speed < 1:
        raise ValueError("Speed spline counts and order must be positive.")
    speed_smoothing_sigma_s = float(speed_smoothing_sigma_s)
    if (
        not np.isfinite(speed_smoothing_sigma_s)
        or speed_smoothing_sigma_s <= 0.0
    ):
        raise ValueError(
            "speed_smoothing_sigma_s must be positive and finite."
        )
    dark_threshold = float(min_dark_firing_rate_hz)
    light_threshold = float(min_light_firing_rate_hz)
    if any(
        not np.isfinite(value) or value < 0.0
        for value in (dark_threshold, light_threshold)
    ):
        raise ValueError("Movement firing-rate thresholds must be non-negative and finite.")
    speed_mode = str(speed_feature_mode)
    if speed_mode not in {"linear", "bspline"}:
        raise ValueError("speed_feature_mode must be 'linear' or 'bspline'.")
    normalized_bounds: tuple[float, float] | None = None
    if speed_bounds is not None:
        bounds = np.asarray(speed_bounds, dtype=float).reshape(-1)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[1] <= bounds[0]:
            raise ValueError("speed_bounds must contain two increasing finite values.")
        normalized_bounds = (float(bounds[0]), float(bounds[1]))
    return {
        "basis_candidate_mode": mode,
        "basis_candidates": normalized_basis,
        "schema_version": SCHEMA_VERSION_BY_MODE[mode],
        "bin_sizes_s": bins,
        "ridges": ridge_values,
        "n_folds": n_folds,
        "random_seed": int(random_seed),
        "spline_order": spline_order,
        "min_dark_firing_rate_hz": dark_threshold,
        "min_light_firing_rate_hz": light_threshold,
        "use_speed": bool(use_speed),
        "speed_feature_mode": speed_mode,
        "n_splines_speed": n_splines_speed,
        "spline_order_speed": spline_order_speed,
        "speed_smoothing_sigma_s": speed_smoothing_sigma_s,
        "speed_bounds": normalized_bounds,
    }


def derive_graph_geometry(
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
) -> tuple[float, np.ndarray]:
    """Derive common path length and three-segment boundaries from NWB graphs."""
    actual = set(graph_inputs_by_trajectory)
    expected = set(TRAJECTORY_TYPES)
    if actual != expected:
        raise ValueError(
            "graph_inputs_by_trajectory must contain exactly the four trajectories; "
            f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}."
        )
    geometries: list[tuple[float, np.ndarray]] = []
    for trajectory_type in TRAJECTORY_TYPES:
        graph = dict(graph_inputs_by_trajectory[trajectory_type])
        if str(graph.get("configuration_name", "")) != trajectory_type:
            raise ValueError("WTrackGraph configuration_name must match trajectory_type.")
        if str(graph.get("coordinate_unit", "")) != "cm":
            raise ValueError("WTrackGraph coordinate_unit must be 'cm'.")
        kwargs = dict(graph.get("track_graph_kwargs", {}))
        linearization = dict(graph.get("linearization_kwargs", {}))
        nodes = np.asarray(kwargs.get("node_positions"), dtype=float)
        edge_order = np.asarray(linearization.get("edge_order"), dtype=int)
        spacing = np.asarray(linearization.get("edge_spacing", ()), dtype=float).reshape(-1)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.all(np.isfinite(nodes)):
            raise ValueError("WTrackGraph node_positions must be finite with shape (n, 2).")
        if edge_order.shape != (5, 2):
            raise ValueError("Dark/light segment geometry requires five ordered path edges.")
        if spacing.shape != (4,) or np.any(spacing < 0.0) or not np.all(np.isfinite(spacing)):
            raise ValueError("WTrackGraph edge_spacing must contain four non-negative values.")
        if np.any(edge_order < 0) or np.any(edge_order >= len(nodes)):
            raise ValueError("WTrackGraph edge_order contains invalid node indices.")
        lengths = np.linalg.norm(
            nodes[edge_order[:, 1]] - nodes[edge_order[:, 0]],
            axis=1,
        )
        total = float(np.sum(lengths) + np.sum(spacing))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("WTrackGraph path length must be positive and finite.")
        cumulative_before = np.concatenate(
            ([0.0], np.cumsum(lengths[:-1] + spacing))
        )
        segment_edges = np.asarray(
            [
                0.0,
                (cumulative_before[1] + 0.5 * lengths[1]) / total,
                (cumulative_before[3] + 0.5 * lengths[3]) / total,
                1.0,
            ],
            dtype=float,
        )
        if np.any(np.diff(segment_edges) <= 0.0):
            raise ValueError("Derived dark/light segment edges are not increasing.")
        geometries.append((total, segment_edges))
    reference_length, reference_edges = geometries[0]
    for length, edges in geometries[1:]:
        if not np.isclose(length, reference_length, rtol=1e-9, atol=1e-9):
            raise ValueError("The four trajectory graphs do not have a common path length.")
        if not np.allclose(edges, reference_edges, rtol=1e-9, atol=1e-9):
            raise ValueError("The four trajectory graphs imply different segment boundaries.")
    return float(reference_length), reference_edges


def _position_basis_configs(
    *,
    parameters: Mapping[str, Any],
    trajectory_length_cm: float,
) -> list[dict[str, Any]]:
    """Return source-compatible basis configs while preserving the public mode."""
    analysis = _analysis_module()
    configs = []
    for value in parameters["basis_candidates"]:
        if parameters["basis_candidate_mode"] == "spatial_bin_size_cm":
            spatial_bin_size_cm = float(value)
            n_splines = analysis.n_splines_from_spatial_bin_size(
                trajectory_length_cm,
                spatial_bin_size_cm,
                spline_order=int(parameters["spline_order"]),
            )
        else:
            n_splines = int(value)
            # The current fitter accepts both quantities, but n_splines is the
            # operative v4 candidate.  This proxy is removed from v4 outputs.
            spatial_bin_size_cm = trajectory_length_cm / n_splines
        configs.append(
            {
                "spatial_bin_size_cm": float(spatial_bin_size_cm),
                "trajectory_length_cm": float(trajectory_length_cm),
                "n_splines": int(n_splines),
                "spline_order": int(parameters["spline_order"]),
                "pos_bounds": (0.0, 1.0),
                "basis_value": value,
            }
        )
    return configs


def _identity_and_rates(
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    dark_movement_firing_rate_table: pd.DataFrame,
    light_movement_firing_rate_table: pd.DataFrame,
    min_dark_firing_rate_hz: float,
    min_light_firing_rate_hz: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]:
    """Align movement-rate rows and return selection plus upstream status."""
    from v1ca1.spyglass.movement import (
        align_movement_firing_rates,
        validate_movement_firing_rate_table,
    )
    from v1ca1.spyglass.path_specific_place import _identity_rows

    dark_table = validate_movement_firing_rate_table(
        dark_movement_firing_rate_table
    )
    light_table = validate_movement_firing_rate_table(
        light_movement_firing_rate_table
    )
    identities = pd.DataFrame.from_records(
        _identity_rows(spikes, stable_unit_ids),
        columns=(*IDENTITY_COLUMNS, "_group_key"),
    )
    dark_rates = align_movement_firing_rates(
        dark_table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    ).to_numpy(dtype=float)
    light_rates = align_movement_firing_rates(
        light_table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    ).to_numpy(dtype=float)
    if identities.empty:
        return (
            pd.DataFrame(columns=SELECTED_UNIT_COLUMNS),
            np.asarray([], dtype=bool),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            "no_units",
        )
    statuses = {
        str(dark_table["firing_rate_status"].iloc[0]),
        str(light_table["firing_rate_status"].iloc[0]),
    }
    if "no_valid_position" in statuses:
        status = "no_valid_position"
    elif "no_movement" in statuses:
        status = "no_movement"
    else:
        status = "valid"
    if status != "valid":
        return (
            pd.DataFrame(columns=SELECTED_UNIT_COLUMNS),
            np.zeros(len(identities), dtype=bool),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            status,
        )
    masks = _analysis_module().build_train_epoch_fr_mask(
        dark_rates,
        light_rates,
        min_dark_fr_hz=min_dark_firing_rate_hz,
        min_light_fr_hz=min_light_firing_rate_hz,
    )
    unit_mask = np.asarray(masks["combined"], dtype=bool)
    selected = identities.loc[unit_mask, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    selected["selection_index"] = np.arange(len(selected), dtype=np.int64)
    selected["dark_movement_firing_rate_hz"] = dark_rates[unit_mask]
    selected["light_movement_firing_rate_hz"] = light_rates[unit_mask]
    selected["n_selected_model_trajectory_fits"] = 0
    selected["valid_glm_fit"] = False
    selected = selected.loc[:, list(SELECTED_UNIT_COLUMNS)]
    status = "valid" if np.any(unit_mask) else "no_eligible_units"
    return (
        selected,
        unit_mask,
        dark_rates[unit_mask],
        light_rates[unit_mask],
        status,
    )


def _derive_task_progression(
    *,
    position_by_epoch: Mapping[str, Any],
    trajectory_intervals_by_epoch: Mapping[str, Mapping[str, Any]],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    epochs: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Linearize selected NWB position into graph-derived task progression."""
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    output: dict[str, dict[str, Any]] = {}
    for epoch in epochs:
        output[epoch] = {}
        for trajectory_type in TRAJECTORY_TYPES:
            task_progression, _length = build_task_progression_from_graph(
                position=position_by_epoch[epoch],
                trajectory_interval=trajectory_intervals_by_epoch[epoch][trajectory_type],
                graph_inputs=graph_inputs_by_trajectory[trajectory_type],
                trajectory_type=trajectory_type,
            )
            output[epoch][trajectory_type] = task_progression
    return output


def _derive_speed(
    position_by_epoch: Mapping[str, Any],
    epochs: Sequence[str],
    *,
    speed_smoothing_sigma_s: float,
) -> dict[str, Any]:
    """Compute speed from already-offset selected position rows."""
    from v1ca1.helper.session import build_speed_tsd

    speed = {}
    for epoch in epochs:
        position = position_by_epoch[epoch]
        speed[epoch] = build_speed_tsd(
            np.asarray(position.d, dtype=float),
            np.asarray(position.t, dtype=float),
            position_offset=0,
            speed_smoothing_sigma_s=float(speed_smoothing_sigma_s),
        )
    return speed


def _fit_candidate_dataset(
    *,
    model_name: str,
    spikes: Any,
    trajectory_intervals_by_epoch: Mapping[str, Mapping[str, Any]],
    task_progression_by_epoch: Mapping[str, Mapping[str, Any]],
    speed_by_epoch: Mapping[str, Any] | None,
    light_epoch: str,
    dark_epoch: str,
    folds_by_trajectory: Mapping[str, list[dict[str, Any]]],
    movement_by_epoch: Mapping[str, Any],
    bin_size_s: float,
    position_basis: Mapping[str, Any],
    ridges: Sequence[float],
    unit_mask: np.ndarray,
    segment_edges: np.ndarray,
    animal_name: str,
    date: str,
    region: str,
    selected_dark_rates: np.ndarray,
    selected_light_rates: np.ndarray,
    parameters: Mapping[str, Any],
    sources: Mapping[str, Any],
    fit_parameters: Mapping[str, Any],
) -> Any:
    """Fit one candidate by reusing the existing per-trajectory GLM routines."""
    analysis = _analysis_module()
    results_by_trajectory: dict[str, dict[float, dict[str, Any]]] = {
        trajectory_type: {} for trajectory_type in TRAJECTORY_TYPES
    }
    for trajectory_type in TRAJECTORY_TYPES:
        for ridge in ridges:
            results_by_trajectory[trajectory_type][float(ridge)] = (
                analysis._fit_selected_full_model_per_traj(
                    model_name=model_name,
                    spikes=spikes,
                    trajectory_ep_by_epoch=dict(
                        trajectory_intervals_by_epoch
                    ),
                    tp_by_epoch=dict(task_progression_by_epoch),
                    speed_by_epoch=(
                        None if speed_by_epoch is None else dict(speed_by_epoch)
                    ),
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    traj_name=trajectory_type,
                    folds=folds_by_trajectory[trajectory_type],
                    movement_by_run=dict(movement_by_epoch),
                    bin_size_s=float(bin_size_s),
                    n_splines=int(position_basis["n_splines"]),
                    spline_order=int(parameters["spline_order"]),
                    spatial_bin_size_cm=float(
                        position_basis["spatial_bin_size_cm"]
                    ),
                    trajectory_length_cm=float(
                        position_basis["trajectory_length_cm"]
                    ),
                    ridge=float(ridge),
                    unit_mask=unit_mask,
                    segment_edges=segment_edges,
                    speed_feature_mode=str(parameters["speed_feature_mode"]),
                    n_splines_speed=int(parameters["n_splines_speed"]),
                    spline_order_speed=int(
                        parameters["spline_order_speed"]
                    ),
                    speed_bounds=parameters["speed_bounds"],
                    population_glm_class=_IsolatingPopulationGLM,
                )
            )
    dataset = analysis.build_selected_candidate_dataset(
        model_name=model_name,
        results_by_traj=results_by_trajectory,
        ridge_values=[float(value) for value in ridges],
        animal_name=animal_name,
        date=date,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        dark_movement_firing_rates=selected_dark_rates,
        light_movement_firing_rates=selected_light_rates,
        min_dark_firing_rate_hz=float(parameters["min_dark_firing_rate_hz"]),
        min_light_firing_rate_hz=float(parameters["min_light_firing_rate_hz"]),
        segment_edges=segment_edges,
        sources=dict(sources),
        fit_parameters=dict(fit_parameters),
    )
    return dataset


def _normalize_dataset_schema(dataset: Any, *, basis_candidate_mode: str) -> Any:
    """Return one dataset with an honest v4 or v5 basis contract."""
    normalized = dataset.copy(deep=True)
    normalized.attrs["schema_version"] = SCHEMA_VERSION_BY_MODE[basis_candidate_mode]
    normalized.attrs["basis_candidate_mode"] = basis_candidate_mode
    if basis_candidate_mode == "n_splines":
        normalized.attrs.pop("spatial_bin_size_cm", None)
        normalized.attrs.pop("selected_spatial_bin_size_cm", None)
    return normalized


def _add_selected_unit_fit_qc(
    selected_units: pd.DataFrame,
    selected_datasets: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Record finite selected-model fits without dropping failed units."""
    counts = np.zeros(len(selected_units), dtype=int)
    expected = 0
    for model_name in MODEL_NAMES:
        dataset = selected_datasets[model_name]
        if "coef_intercept" not in dataset:
            raise ValueError(
                f"Selected dataset {model_name!r} is missing coef_intercept."
            )
        coefficients = dataset["coef_intercept"]
        if "unit" not in coefficients.dims:
            raise ValueError("Selected coef_intercept must index unit.")
        values = np.asarray(
            coefficients.transpose(
                *(dim for dim in coefficients.dims if dim != "unit"),
                "unit",
            ),
            dtype=float,
        )
        reshaped = values.reshape(-1, len(selected_units))
        counts += np.sum(np.isfinite(reshaped), axis=0).astype(int)
        expected += reshaped.shape[0]
    valid = counts == expected
    output = selected_units.copy()
    output["n_selected_model_trajectory_fits"] = counts
    output["valid_glm_fit"] = valid
    if not np.any(valid):
        status = "no_valid_units"
    elif np.all(valid):
        status = "valid"
    else:
        status = "partial_valid"
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)], status


def _summary_for_mode(
    *,
    selection_records: Sequence[dict[str, Any]],
    position_basis_configs: Sequence[dict[str, Any]],
    selected_by_model: Mapping[str, dict[str, Any]],
    shared_selection: Mapping[str, Any],
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    fit_parameters: Mapping[str, Any],
) -> Any:
    """Build the existing summary and expose the selected public basis dimension."""
    analysis = _analysis_module()
    summary = analysis.build_selection_summary_dataset(
        selection_records=selection_records,
        model_names=MODEL_NAMES,
        bin_sizes_s=parameters["bin_sizes_s"],
        position_basis_configs=position_basis_configs,
        ridge_values=parameters["ridges"],
        selected_by_model=dict(selected_by_model),
        shared_selection=dict(shared_selection),
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        light_epoch=metadata["light_epoch"],
        dark_epoch=metadata["dark_epoch"],
        fit_parameters=dict(fit_parameters),
    )
    mode = str(parameters["basis_candidate_mode"])
    summary = _normalize_dataset_schema(summary, basis_candidate_mode=mode)
    if mode == "n_splines":
        summary = summary.drop_vars("n_splines_by_spatial_bin_size")
        summary = summary.rename({"spatial_bin_size_cm": "n_splines"})
        summary = summary.assign_coords(
            n_splines=np.asarray(parameters["basis_candidates"], dtype=int)
        )
        summary.attrs.pop("trajectory_length_cm", None)
    return summary


def _metadata(
    *,
    dark_light_glm_id: Any,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
) -> dict[str, str]:
    """Return canonical coupled-result metadata."""
    metadata = {
        "dark_light_glm_id": _uuid_string(
            dark_light_glm_id,
            name="dark_light_glm_id",
        )
    }
    metadata.update(
        {
            name: _path_component(value, name=name)
            for name, value in {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "light_epoch": light_epoch,
                "dark_epoch": dark_epoch,
            }.items()
        }
    )
    if metadata["light_epoch"] == metadata["dark_epoch"]:
        raise ValueError("light_epoch and dark_epoch must differ.")
    return metadata


def _validate_movement_table_context(
    table: pd.DataFrame,
    *,
    metadata: Mapping[str, str],
    epoch: str,
) -> None:
    """Require one movement-rate table to match this session and epoch."""
    if table.empty:
        return
    expected = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "epoch": epoch,
    }
    for field, value in expected.items():
        observed = table[field].astype(str).unique().tolist()
        if observed != [str(value)]:
            raise ValueError(
                f"Movement firing-rate table has mismatched {field!r}."
            )


def _terminal_result(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    trajectory_length_cm: float,
    segment_edges: np.ndarray,
    analysis_status: str,
) -> dict[str, Any]:
    """Return one canonical, persistable terminal result bundle."""
    if analysis_status not in {
        "no_units",
        "no_eligible_units",
        "no_valid_position",
        "no_movement",
    }:
        raise ValueError("Unsupported terminal dark/light analysis status.")
    import xarray as xr

    fit_parameters = {
        **dict(parameters),
        "basis_candidates": list(parameters["basis_candidates"]),
        "bin_sizes_s": list(parameters["bin_sizes_s"]),
        "ridges": list(parameters["ridges"]),
        "models": list(MODEL_NAMES),
        "segment_edges": np.asarray(segment_edges, dtype=float).tolist(),
        "trajectory_length_cm": float(trajectory_length_cm),
        "cv_fold_scope": "lap_level_by_trajectory_movement_only",
        "fold_metadata": {},
        "unit_failure_policy": "isolate_population_glm_failures_per_unit",
    }
    for name in ("parameter_sha256", "output_rule_sha256"):
        fit_parameters.pop(name, None)
    mode = str(parameters["basis_candidate_mode"])
    summary = xr.Dataset(
        coords={
            "model": np.asarray(MODEL_NAMES, dtype=str),
            mode: np.asarray(parameters["basis_candidates"]),
        },
        attrs={
            **{
                name: metadata[name]
                for name in (
                    "animal_name",
                    "date",
                    "region",
                    "light_epoch",
                    "dark_epoch",
                )
            },
            "schema_version": parameters["schema_version"],
            "basis_candidate_mode": mode,
            "fit_stage": "terminal",
            "analysis_status": analysis_status,
            "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        },
    )
    selected_datasets = {}
    for model_name in MODEL_NAMES:
        selected_datasets[model_name] = xr.Dataset(
            data_vars={
                "dark_movement_firing_rate_hz": (
                    "unit",
                    np.asarray([], dtype=float),
                ),
                "light_movement_firing_rate_hz": (
                    "unit",
                    np.asarray([], dtype=float),
                ),
                "coef_intercept": (
                    ("unit", "trajectory"),
                    np.empty((0, len(TRAJECTORY_TYPES)), dtype=float),
                ),
            },
            coords={
                "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
                "unit": np.asarray([], dtype=str),
            },
            attrs={
                **{
                    name: metadata[name]
                    for name in (
                        "animal_name",
                        "date",
                        "region",
                        "light_epoch",
                        "dark_epoch",
                    )
                },
                "schema_version": parameters["schema_version"],
                "basis_candidate_mode": mode,
                "model_name": model_name,
                "fit_stage": "terminal",
                "analysis_status": analysis_status,
            },
        )
    return validate_dark_light_glm_result(
        {
            "metadata": dict(metadata),
            "parameters": dict(parameters),
            "selected_units": pd.DataFrame(columns=SELECTED_UNIT_COLUMNS),
            "candidate_datasets": {},
            "selected_datasets": selected_datasets,
            "selection_summary": summary,
            "trajectory_length_cm": trajectory_length_cm,
            "segment_edges": segment_edges,
            "analysis_status": analysis_status,
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
    )


def compute_dark_light_glm(
    *,
    dark_light_glm_id: Any,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    dark_movement_firing_rate_table: pd.DataFrame,
    light_movement_firing_rate_table: pd.DataFrame,
    movement_by_epoch: Mapping[str, Any],
    trajectory_intervals_by_epoch: Mapping[str, Mapping[str, Any]],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    position_by_epoch: Mapping[str, Any] | None = None,
    task_progression_by_epoch: Mapping[str, Mapping[str, Any]] | None = None,
    speed_by_epoch: Mapping[str, Any] | None = None,
    parameter_name: str = "current",
    parameter_sha256: str = "",
    output_rule_sha256: str = "",
    basis_candidate_mode: str = "spatial_bin_size_cm",
    basis_candidates: Sequence[float | int] = DEFAULT_SPATIAL_BIN_SIZES_CM,
    bin_sizes_s: Sequence[float] = DEFAULT_BIN_SIZES_S,
    ridges: Sequence[float] = DEFAULT_RIDGES,
    n_folds: int = DEFAULT_N_FOLDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    spline_order: int = DEFAULT_SPLINE_ORDER,
    min_dark_firing_rate_hz: float = 0.0,
    min_light_firing_rate_hz: float = 0.0,
    use_speed: bool = DEFAULT_USE_SPEED,
    speed_feature_mode: str = DEFAULT_SPEED_FEATURE_MODE,
    n_splines_speed: int = DEFAULT_N_SPLINES_SPEED,
    spline_order_speed: int = DEFAULT_SPLINE_ORDER_SPEED,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    speed_bounds: Sequence[float] | None = None,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one coupled dark/light result using existing fitting functions."""
    metadata = _metadata(
        dark_light_glm_id=dark_light_glm_id,
        animal_name=animal_name,
        date=date,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    parameters = validate_dark_light_glm_parameters(
        basis_candidate_mode=basis_candidate_mode,
        basis_candidates=basis_candidates,
        bin_sizes_s=bin_sizes_s,
        ridges=ridges,
        n_folds=n_folds,
        random_seed=random_seed,
        spline_order=spline_order,
        min_dark_firing_rate_hz=min_dark_firing_rate_hz,
        min_light_firing_rate_hz=min_light_firing_rate_hz,
        use_speed=use_speed,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_smoothing_sigma_s=speed_smoothing_sigma_s,
        speed_bounds=speed_bounds,
    )
    parameter_name = _path_component(parameter_name, name="parameter_name")
    parameters.update(
        {
            "parameter_name": parameter_name,
            "parameter_sha256": str(parameter_sha256),
            "output_rule_sha256": str(output_rule_sha256),
        }
    )
    epochs = (dark_epoch, light_epoch)
    for mapping_name, mapping in (
        ("movement_by_epoch", movement_by_epoch),
        ("trajectory_intervals_by_epoch", trajectory_intervals_by_epoch),
    ):
        if set(mapping) != set(epochs):
            raise ValueError(f"{mapping_name} must contain exactly the dark and light epochs.")
    for epoch in epochs:
        if set(trajectory_intervals_by_epoch[epoch]) != set(TRAJECTORY_TYPES):
            raise ValueError("Each epoch must contain exactly four trajectory intervals.")
    trajectory_length_cm, segment_edges = derive_graph_geometry(
        graph_inputs_by_trajectory
    )
    _validate_movement_table_context(
        dark_movement_firing_rate_table,
        metadata=metadata,
        epoch=dark_epoch,
    )
    _validate_movement_table_context(
        light_movement_firing_rate_table,
        metadata=metadata,
        epoch=light_epoch,
    )
    (
        selected_units,
        unit_mask,
        selected_dark_rates,
        selected_light_rates,
        input_status,
    ) = _identity_and_rates(
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        dark_movement_firing_rate_table=dark_movement_firing_rate_table,
        light_movement_firing_rate_table=light_movement_firing_rate_table,
        min_dark_firing_rate_hz=parameters["min_dark_firing_rate_hz"],
        min_light_firing_rate_hz=parameters["min_light_firing_rate_hz"],
    )
    if input_status != "valid":
        return _terminal_result(
            metadata=metadata,
            parameters=parameters,
            trajectory_length_cm=trajectory_length_cm,
            segment_edges=segment_edges,
            analysis_status=input_status,
        )
    if task_progression_by_epoch is None:
        if position_by_epoch is None or set(position_by_epoch) != set(epochs):
            raise ValueError(
                "position_by_epoch is required to derive task progression when "
                "task_progression_by_epoch is not supplied."
            )
        task_progression_by_epoch = _derive_task_progression(
            position_by_epoch=position_by_epoch,
            trajectory_intervals_by_epoch=trajectory_intervals_by_epoch,
            graph_inputs_by_trajectory=graph_inputs_by_trajectory,
            epochs=epochs,
        )
    if set(task_progression_by_epoch) != set(epochs) or any(
        set(task_progression_by_epoch[epoch]) != set(TRAJECTORY_TYPES)
        for epoch in epochs
    ):
        raise ValueError("task_progression_by_epoch must contain both epochs and four paths.")
    if parameters["use_speed"]:
        if speed_by_epoch is None:
            if position_by_epoch is None:
                raise ValueError("Speed-enabled fits require speed_by_epoch or position_by_epoch.")
            speed_by_epoch = _derive_speed(
                position_by_epoch,
                epochs,
                speed_smoothing_sigma_s=parameters[
                    "speed_smoothing_sigma_s"
                ],
            )
        if set(speed_by_epoch) != set(epochs):
            raise ValueError("speed_by_epoch must contain exactly the dark and light epochs.")
    else:
        speed_by_epoch = None

    analysis = _analysis_module()
    folds_by_trajectory = {
        trajectory_type: analysis.build_lap_cv_folds_for_trajectory(
            trajectory_intervals=dict(trajectory_intervals_by_epoch),
            movement_by_run=dict(movement_by_epoch),
            dark_epoch=dark_epoch,
            light_epoch=light_epoch,
            trajectory=trajectory_type,
            n_folds=parameters["n_folds"],
            seed=parameters["random_seed"],
        )
        for trajectory_type in TRAJECTORY_TYPES
    }
    position_basis_configs = _position_basis_configs(
        parameters=parameters,
        trajectory_length_cm=trajectory_length_cm,
    )
    fold_metadata = {
        trajectory_type: [dict(fold["metadata"]) for fold in folds]
        for trajectory_type, folds in folds_by_trajectory.items()
    }
    fit_parameters = {
        **parameters,
        "basis_candidates": list(parameters["basis_candidates"]),
        "bin_sizes_s": list(parameters["bin_sizes_s"]),
        "ridges": list(parameters["ridges"]),
        "models": list(MODEL_NAMES),
        "segment_edges": segment_edges.tolist(),
        "trajectory_length_cm": trajectory_length_cm,
        "cv_fold_scope": "lap_level_by_trajectory_movement_only",
        "fold_metadata": fold_metadata,
        "unit_failure_policy": "isolate_population_glm_failures_per_unit",
    }
    for name in ("parameter_sha256", "output_rule_sha256"):
        fit_parameters.pop(name, None)
    candidate_datasets: dict[str, Any] = {}
    candidate_lookup: dict[tuple[str, float, float], tuple[str, Any]] = {}
    selection_records: list[dict[str, Any]] = []
    source_metadata = {} if sources is None else dict(sources)

    for bin_size_s in parameters["bin_sizes_s"]:
        for position_basis in position_basis_configs:
            try:
                dataset = _fit_candidate_dataset(
                    model_name="visual",
                    spikes=spikes,
                    trajectory_intervals_by_epoch=trajectory_intervals_by_epoch,
                    task_progression_by_epoch=task_progression_by_epoch,
                    speed_by_epoch=speed_by_epoch,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    folds_by_trajectory=folds_by_trajectory,
                    movement_by_epoch=movement_by_epoch,
                    bin_size_s=bin_size_s,
                    position_basis=position_basis,
                    ridges=parameters["ridges"],
                    unit_mask=unit_mask,
                    segment_edges=segment_edges,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    selected_dark_rates=selected_dark_rates,
                    selected_light_rates=selected_light_rates,
                    parameters=parameters,
                    sources=source_metadata,
                    fit_parameters=fit_parameters,
                )
            finally:
                _clear_jax_fit_caches()
            key = _candidate_key(
                "visual",
                bin_size_s=bin_size_s,
                basis_candidate_mode=parameters["basis_candidate_mode"],
                basis_value=position_basis["basis_value"],
            )
            candidate_datasets[key] = _normalize_dataset_schema(
                dataset,
                basis_candidate_mode=parameters["basis_candidate_mode"],
            )
            candidate_lookup[("visual", float(bin_size_s), float(position_basis["spatial_bin_size_cm"]))] = (key, dataset)
            selection_records.extend(analysis.score_candidate_dataset(dataset))

    shared_selection = analysis.choose_visual_shared_hyperparameters(selection_records)
    selected_position_basis = next(
        config
        for config in position_basis_configs
        if np.isclose(
            float(config["spatial_bin_size_cm"]),
            float(shared_selection["spatial_bin_size_cm"]),
        )
    )
    selected_bin_size_s = float(shared_selection["bin_size_s"])
    for model_name in MODEL_NAMES[1:]:
        try:
            dataset = _fit_candidate_dataset(
                model_name=model_name,
                spikes=spikes,
                trajectory_intervals_by_epoch=trajectory_intervals_by_epoch,
                task_progression_by_epoch=task_progression_by_epoch,
                speed_by_epoch=speed_by_epoch,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                folds_by_trajectory=folds_by_trajectory,
                movement_by_epoch=movement_by_epoch,
                bin_size_s=selected_bin_size_s,
                position_basis=selected_position_basis,
                ridges=parameters["ridges"],
                unit_mask=unit_mask,
                segment_edges=segment_edges,
                animal_name=animal_name,
                date=date,
                region=region,
                selected_dark_rates=selected_dark_rates,
                selected_light_rates=selected_light_rates,
                parameters=parameters,
                sources=source_metadata,
                fit_parameters=fit_parameters,
            )
        finally:
            _clear_jax_fit_caches()
        key = _candidate_key(
            model_name,
            bin_size_s=selected_bin_size_s,
            basis_candidate_mode=parameters["basis_candidate_mode"],
            basis_value=selected_position_basis["basis_value"],
        )
        candidate_datasets[key] = _normalize_dataset_schema(
            dataset,
            basis_candidate_mode=parameters["basis_candidate_mode"],
        )
        candidate_lookup[(model_name, selected_bin_size_s, float(selected_position_basis["spatial_bin_size_cm"]))] = (key, dataset)
        selection_records.extend(analysis.score_candidate_dataset(dataset))

    selected_by_model: dict[str, dict[str, Any]] = {}
    selected_datasets: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        selected_record = analysis.choose_model_ridge(
            selection_records,
            model_name=model_name,
            bin_size_s=selected_bin_size_s,
            spatial_bin_size_cm=float(selected_position_basis["spatial_bin_size_cm"]),
        )
        selected_by_model[model_name] = selected_record
        candidate = candidate_lookup[
            (
                model_name,
                selected_bin_size_s,
                float(selected_position_basis["spatial_bin_size_cm"]),
            )
        ][1]
        selected = analysis.build_selected_model_dataset(
            candidate,
            selected_ridge=float(selected_record["ridge"]),
            selection_score=float(selected_record["score_median"]),
            shared_selection=shared_selection,
        )
        selected_datasets[model_name] = _normalize_dataset_schema(
            selected,
            basis_candidate_mode=parameters["basis_candidate_mode"],
        )
    selected_units, analysis_status = _add_selected_unit_fit_qc(
        selected_units,
        selected_datasets,
    )
    summary = _summary_for_mode(
        selection_records=selection_records,
        position_basis_configs=position_basis_configs,
        selected_by_model=selected_by_model,
        shared_selection=shared_selection,
        metadata=metadata,
        parameters=parameters,
        fit_parameters=fit_parameters,
    )
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": selected_units,
        "candidate_datasets": candidate_datasets,
        "selected_datasets": selected_datasets,
        "selection_summary": summary,
        "trajectory_length_cm": trajectory_length_cm,
        "segment_edges": segment_edges,
        "analysis_status": analysis_status,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }
    return validate_dark_light_glm_result(result)


def _dataset_metadata(dataset: Any) -> dict[str, str]:
    """Return coupled metadata extracted from one NetCDF dataset."""
    return {
        name: str(dataset.attrs.get(name, ""))
        for name in ("animal_name", "date", "region", "light_epoch", "dark_epoch")
    }


def _validate_dataset(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    fit_stage: str | None,
) -> None:
    """Validate schema, metadata, stage, and basis semantics of one dataset."""
    if _dataset_metadata(dataset) != {
        name: metadata[name]
        for name in ("animal_name", "date", "region", "light_epoch", "dark_epoch")
    }:
        raise ValueError("Dark/light dataset metadata does not match the result.")
    if str(dataset.attrs.get("schema_version", "")) != str(parameters["schema_version"]):
        raise ValueError("Dark/light dataset schema_version does not match basis mode.")
    declared_mode = dataset.attrs.get("basis_candidate_mode")
    if declared_mode is not None and str(declared_mode) != parameters["basis_candidate_mode"]:
        raise ValueError("Dark/light dataset basis_candidate_mode is inconsistent.")
    if parameters["basis_candidate_mode"] == "n_splines" and (
        "spatial_bin_size_cm" in dataset.attrs
        or "selected_spatial_bin_size_cm" in dataset.attrs
    ):
        raise ValueError("Schema v4 artifacts must not claim spatial-bin candidates.")
    if fit_stage is not None and str(dataset.attrs.get("fit_stage", "")) != fit_stage:
        raise ValueError(f"Expected a {fit_stage!r} dark/light dataset.")


def _validate_dataset_units_and_rates(
    dataset: Any,
    *,
    selected_units: pd.DataFrame,
    role: str,
) -> None:
    """Require exact selected group-unit order and movement-rate vectors."""
    if "unit" not in dataset.dims:
        raise ValueError(f"{role} is missing its unit dimension.")
    expected_ids = selected_units["group_unit_id"].astype(str).tolist()
    observed_ids = [str(value) for value in np.asarray(dataset.unit.values)]
    if observed_ids != expected_ids:
        raise ValueError(
            f"{role} unit coordinate/order does not match selected_units "
            "group_unit_id."
        )
    for variable, column in (
        (
            "dark_movement_firing_rate_hz",
            "dark_movement_firing_rate_hz",
        ),
        (
            "light_movement_firing_rate_hz",
            "light_movement_firing_rate_hz",
        ),
    ):
        if variable not in dataset or dataset[variable].dims != ("unit",):
            raise ValueError(f"{role} is missing unit vector {variable!r}.")
        observed = np.asarray(dataset[variable], dtype=float)
        expected = selected_units[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError(f"{role} contains invalid movement firing rates.")
        if not np.allclose(observed, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(
                f"{role} {variable} does not match selected_units."
            )


def _selected_units_sha256(selected_units: pd.DataFrame) -> str:
    """Return the canonical persistent-unit identity digest."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    return unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict("records")
    )


def validate_dark_light_glm_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one in-memory coupled result and return defensive copies."""
    required = {
        "metadata",
        "parameters",
        "selected_units",
        "candidate_datasets",
        "selected_datasets",
        "selection_summary",
        "trajectory_length_cm",
        "segment_edges",
        "analysis_status",
        "artifact_origin",
        "legacy_artifact_provenance",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Dark/light result is missing fields {missing!r}.")
    copied = dict(result)
    metadata = _metadata(**copied["metadata"])
    raw_parameters = dict(copied["parameters"])
    effective = validate_dark_light_glm_parameters(
        **{
            name: raw_parameters[name]
            for name in (
                "basis_candidate_mode",
                "basis_candidates",
                "bin_sizes_s",
                "ridges",
                "n_folds",
                "random_seed",
                "spline_order",
                "min_dark_firing_rate_hz",
                "min_light_firing_rate_hz",
                "use_speed",
                "speed_feature_mode",
                "n_splines_speed",
                "spline_order_speed",
                "speed_smoothing_sigma_s",
                "speed_bounds",
            )
        }
    )
    for name in ("parameter_name", "parameter_sha256", "output_rule_sha256"):
        effective[name] = str(raw_parameters.get(name, ""))
    status = str(copied["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Unsupported dark/light analysis_status {status!r}.")
    terminal_statuses = {
        "no_units",
        "no_eligible_units",
        "no_valid_position",
        "no_movement",
    }
    is_terminal = status in terminal_statuses
    selected_units = copied["selected_units"].copy()
    if tuple(selected_units.columns) != SELECTED_UNIT_COLUMNS:
        raise ValueError("selected_units does not have the canonical columns.")
    if is_terminal and not selected_units.empty:
        raise ValueError("Terminal dark/light results require empty selected_units.")
    if not is_terminal and selected_units.empty:
        raise ValueError("Non-terminal dark/light results require selected units.")
    for field in IDENTITY_COLUMNS:
        selected_units[field] = selected_units[field].astype(str)
    if (
        selected_units["stable_unit_id"].duplicated().any()
        or selected_units["group_unit_id"].duplicated().any()
    ):
        raise ValueError("selected_units identities must be unique.")
    expected_stable_ids = (
        selected_units["spikesorting_merge_id"]
        + ":"
        + selected_units["unit_id"]
    )
    if not np.array_equal(
        selected_units["stable_unit_id"].to_numpy(dtype=str),
        expected_stable_ids.to_numpy(dtype=str),
    ):
        raise ValueError("selected_units stable identities are inconsistent.")
    if not selected_units.empty:
        if not np.array_equal(
            selected_units["selection_index"].to_numpy(dtype=int),
            np.arange(len(selected_units), dtype=int),
        ):
            raise ValueError("selected_units selection_index is not contiguous.")
        for column in (
            "dark_movement_firing_rate_hz",
            "light_movement_firing_rate_hz",
        ):
            rates = pd.to_numeric(
                selected_units[column], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
                raise ValueError(
                    "selected_units movement firing rates must be finite and "
                    "non-negative."
                )
        fit_counts = pd.to_numeric(
            selected_units["n_selected_model_trajectory_fits"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if (
            not np.all(np.isfinite(fit_counts))
            or np.any(fit_counts < 0.0)
            or np.any(fit_counts > len(MODEL_NAMES) * len(TRAJECTORY_TYPES))
            or not np.allclose(fit_counts, np.rint(fit_counts))
        ):
            raise ValueError("selected_units fit counts are invalid.")
        selected_units["n_selected_model_trajectory_fits"] = np.rint(
            fit_counts
        ).astype(int)
        selected_units["valid_glm_fit"] = selected_units[
            "valid_glm_fit"
        ].astype(bool)
    candidate_datasets = dict(copied["candidate_datasets"])
    selected_datasets = dict(copied["selected_datasets"])
    expected_candidate_count = (
        len(effective["bin_sizes_s"]) * len(effective["basis_candidates"])
        + len(MODEL_NAMES)
        - 1
    )
    if is_terminal and candidate_datasets:
        raise ValueError("Terminal dark/light results cannot contain candidates.")
    if not is_terminal and len(candidate_datasets) != expected_candidate_count:
        raise ValueError("Dark/light result does not contain every expected candidate.")
    if set(selected_datasets) != set(MODEL_NAMES):
        raise ValueError("Dark/light result must contain all four selected models.")
    for key, dataset in candidate_datasets.items():
        _path_component(key, name="candidate key")
        _validate_dataset(
            dataset,
            metadata=metadata,
            parameters=effective,
            fit_stage="candidate",
        )
        if str(dataset.attrs.get("model_name", "")) not in MODEL_NAMES:
            raise ValueError("Candidate dataset has an unsupported model_name.")
        _validate_dataset_units_and_rates(
            dataset,
            selected_units=selected_units,
            role=f"candidate {key!r}",
        )
    for model_name, dataset in selected_datasets.items():
        _validate_dataset(
            dataset,
            metadata=metadata,
            parameters=effective,
            fit_stage="terminal" if is_terminal else "selected",
        )
        if str(dataset.attrs.get("model_name", "")) != model_name:
            raise ValueError("Selected dataset model_name does not match its key.")
        if is_terminal and str(dataset.attrs.get("analysis_status", "")) != status:
            raise ValueError("Terminal selected dataset has a stale status.")
        _validate_dataset_units_and_rates(
            dataset,
            selected_units=selected_units,
            role=f"selected {model_name!r}",
        )
    summary = copied["selection_summary"]
    _validate_dataset(
        summary,
        metadata=metadata,
        parameters=effective,
        fit_stage=None,
    )
    basis_dim = effective["basis_candidate_mode"]
    if basis_dim not in summary.dims:
        raise ValueError("Selection summary is missing its declared basis dimension.")
    expected_basis = np.asarray(effective["basis_candidates"])
    if not np.allclose(np.asarray(summary.coords[basis_dim]), expected_basis):
        raise ValueError("Selection summary basis coordinates do not match parameters.")
    if is_terminal:
        if str(summary.attrs.get("fit_stage", "")) != "terminal" or str(
            summary.attrs.get("analysis_status", "")
        ) != status:
            raise ValueError("Terminal selection summary has stale status metadata.")
    else:
        try:
            selected_bin_size_s = float(summary.attrs["selected_bin_size_s"])
            selected_basis_value = (
                float(summary.attrs["selected_spatial_bin_size_cm"])
                if effective["basis_candidate_mode"] == "spatial_bin_size_cm"
                else int(summary.attrs["selected_n_splines"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Selection summary is missing its selected bin or basis value."
            ) from exc
        expected_candidate_keys = {
            _candidate_key(
                "visual",
                bin_size_s=bin_size_s,
                basis_candidate_mode=effective["basis_candidate_mode"],
                basis_value=basis_value,
            )
            for bin_size_s in effective["bin_sizes_s"]
            for basis_value in effective["basis_candidates"]
        }
        expected_candidate_keys.update(
            _candidate_key(
                model_name,
                bin_size_s=selected_bin_size_s,
                basis_candidate_mode=effective["basis_candidate_mode"],
                basis_value=selected_basis_value,
            )
            for model_name in MODEL_NAMES[1:]
        )
        if set(candidate_datasets) != expected_candidate_keys:
            raise ValueError(
                "Dark/light candidates do not match the full visual grid and "
                "three selected-basis comparison fits."
            )
        expected_units, expected_status = _add_selected_unit_fit_qc(
            selected_units,
            selected_datasets,
        )
        for column in (
            "n_selected_model_trajectory_fits",
            "valid_glm_fit",
        ):
            if not np.array_equal(
                selected_units[column].to_numpy(),
                expected_units[column].to_numpy(),
            ):
                raise ValueError(f"selected_units {column!r} is stale.")
        if status != expected_status:
            raise ValueError("analysis_status does not match selected-unit fit QC.")
    trajectory_length_cm = float(copied["trajectory_length_cm"])
    segment_edges = np.asarray(copied["segment_edges"], dtype=float).reshape(-1)
    if not np.isfinite(trajectory_length_cm) or trajectory_length_cm <= 0.0:
        raise ValueError("trajectory_length_cm must be positive and finite.")
    if segment_edges.shape != (4,) or not np.all(np.isfinite(segment_edges)) or (
        not np.all(np.diff(segment_edges) > 0.0)
    ) or not np.allclose(segment_edges[[0, -1]], [0.0, 1.0]):
        raise ValueError("segment_edges must be four increasing values from zero to one.")
    origin = str(copied["artifact_origin"])
    if origin not in {"computed", "registered_existing"}:
        raise ValueError("Unsupported artifact_origin.")
    provenance = copied["legacy_artifact_provenance"]
    if origin == "computed" and provenance is not None:
        raise ValueError("Computed artifacts cannot carry legacy provenance.")
    if origin == "registered_existing":
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError("Registered artifacts require legacy provenance.")
        provenance = dict(provenance)
    copied.update(
        {
            "metadata": metadata,
            "parameters": effective,
            "selected_units": selected_units,
            "candidate_datasets": candidate_datasets,
            "selected_datasets": selected_datasets,
            "trajectory_length_cm": trajectory_length_cm,
            "segment_edges": segment_edges,
            "analysis_status": status,
            "artifact_origin": origin,
            "legacy_artifact_provenance": provenance,
            "selected_units_sha256": _selected_units_sha256(selected_units),
            "n_units": len(selected_units),
            "n_candidates": len(candidate_datasets),
            "n_selected_models": len(selected_datasets),
        }
    )
    return copied


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return manifest values shared by every artifact file."""
    provenance = result.get("legacy_artifact_provenance")
    return {
        **result["metadata"],
        "parameter_name": result["parameters"]["parameter_name"],
        "parameter_sha256": result["parameters"]["parameter_sha256"],
        "output_rule_sha256": result["parameters"]["output_rule_sha256"],
        "basis_candidate_mode": result["parameters"]["basis_candidate_mode"],
        "basis_candidates_json": json.dumps(list(result["parameters"]["basis_candidates"])),
        "schema_version": result["parameters"]["schema_version"],
        "n_candidates": result["n_candidates"],
        "n_selected_models": result["n_selected_models"],
        "n_units": result["n_units"],
        "selected_units_sha256": result["selected_units_sha256"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": (
            ""
            if provenance is None
            else json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        ),
    }


def write_dark_light_glm_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write and reload one complete coupled result bundle."""
    result = validate_dark_light_glm_result(result)
    destination = Path(path)
    if destination.name != result["metadata"]["dark_light_glm_id"]:
        raise ValueError("Artifact directory name must equal dark_light_glm_id.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite dark/light artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        candidate_dir = temporary / CANDIDATE_DIRNAME
        selected_dir = temporary / SELECTED_DIRNAME
        candidate_dir.mkdir()
        selected_dir.mkdir()
        result["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME,
            index=False,
        )
        for key, dataset in sorted(result["candidate_datasets"].items()):
            dataset.to_netcdf(candidate_dir / f"{key}.nc")
        for model_name, dataset in sorted(result["selected_datasets"].items()):
            dataset.to_netcdf(selected_dir / f"{model_name}.nc")
        result["selection_summary"].to_netcdf(
            temporary / SELECTION_SUMMARY_FILENAME
        )
        artifact_specs = [
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("selection_summary", SELECTION_SUMMARY_FILENAME, "netcdf"),
        ]
        artifact_specs.extend(
            (f"candidate:{key}", f"{CANDIDATE_DIRNAME}/{key}.nc", "netcdf")
            for key in sorted(result["candidate_datasets"])
        )
        artifact_specs.extend(
            (f"selected:{model}", f"{SELECTED_DIRNAME}/{model}.nc", "netcdf")
            for model in MODEL_NAMES
        )
        common = _manifest_common(result)
        rows = []
        for artifact_key, relative_path, artifact_kind in artifact_specs:
            artifact_path = temporary / relative_path
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "relative_path": relative_path,
                    "artifact_kind": artifact_kind,
                    "file_size_bytes": artifact_path.stat().st_size,
                    "sha256": _file_sha256(artifact_path),
                    **common,
                }
            )
        pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_parquet(
            temporary / MANIFEST_FILENAME,
            index=False,
        )
        load_dark_light_glm_artifact(temporary, _allow_temporary_name=True)
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
        "candidate_dir": destination / CANDIDATE_DIRNAME,
        "selected_dir": destination / SELECTED_DIRNAME,
        "selection_summary_path": destination / SELECTION_SUMMARY_FILENAME,
        "selected_model_paths": {
            model_name: destination / SELECTED_DIRNAME / f"{model_name}.nc"
            for model_name in MODEL_NAMES
        },
    }


def _load_dataset(path: Path) -> Any:
    """Load one NetCDF dataset eagerly and close its backing file."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def load_dark_light_glm_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one complete coupled artifact."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dark/light manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or manifest.empty:
        raise ValueError("Dark/light manifest does not have the canonical schema.")
    if manifest["artifact_key"].duplicated().any():
        raise ValueError("Dark/light manifest artifact keys must be unique.")
    for _, row in manifest.iterrows():
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Manifest contains an unsafe relative path.")
        artifact_path = directory / relative_path
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
            "dark_light_glm_id",
            "animal_name",
            "date",
            "region",
            "light_epoch",
            "dark_epoch",
        )
    }
    if not _allow_temporary_name and directory.name != metadata["dark_light_glm_id"]:
        raise ValueError("Artifact directory name does not match dark_light_glm_id.")
    parameters = {
        "basis_candidate_mode": str(first["basis_candidate_mode"]),
        "basis_candidates": tuple(json.loads(str(first["basis_candidates_json"]))),
        "schema_version": str(first["schema_version"]),
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
    }
    summary = _load_dataset(directory / SELECTION_SUMMARY_FILENAME)
    fit_parameters = json.loads(str(summary.attrs.get("fit_parameters_json", "{}")))
    for name in (
        "bin_sizes_s",
        "ridges",
        "n_folds",
        "random_seed",
        "spline_order",
        "min_dark_firing_rate_hz",
        "min_light_firing_rate_hz",
        "use_speed",
        "speed_feature_mode",
        "n_splines_speed",
        "spline_order_speed",
        "speed_smoothing_sigma_s",
        "speed_bounds",
    ):
        if name not in fit_parameters:
            raise ValueError(f"Selection summary fit parameters are missing {name!r}.")
        parameters[name] = fit_parameters[name]
    candidate_datasets = {}
    selected_datasets = {}
    for _, row in manifest.iterrows():
        key = str(row["artifact_key"])
        if key.startswith("candidate:"):
            candidate_datasets[key.split(":", 1)[1]] = _load_dataset(
                directory / str(row["relative_path"])
            )
        elif key.startswith("selected:"):
            selected_datasets[key.split(":", 1)[1]] = _load_dataset(
                directory / str(row["relative_path"])
            )
    for name in (
        "dark_light_glm_id",
        "animal_name",
        "date",
        "region",
        "light_epoch",
        "dark_epoch",
        "parameter_name",
        "parameter_sha256",
        "output_rule_sha256",
        "basis_candidate_mode",
        "basis_candidates_json",
        "schema_version",
        "n_candidates",
        "n_selected_models",
        "n_units",
        "selected_units_sha256",
        "analysis_status",
        "artifact_origin",
        "legacy_artifact_provenance_json",
    ):
        if not np.all(manifest[name].astype(str) == str(first[name])):
            raise ValueError(f"Dark/light manifest has inconsistent {name!r}.")
    provenance_json = str(first["legacy_artifact_provenance_json"])
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": pd.read_parquet(directory / SELECTED_UNITS_FILENAME),
        "candidate_datasets": candidate_datasets,
        "selected_datasets": selected_datasets,
        "selection_summary": summary,
        "trajectory_length_cm": float(fit_parameters["trajectory_length_cm"]),
        "segment_edges": np.asarray(fit_parameters["segment_edges"], dtype=float),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance": (
            None if not provenance_json else json.loads(provenance_json)
        ),
        "manifest": manifest,
    }
    validated = validate_dark_light_glm_result(result)
    for field in (
        "n_candidates",
        "n_selected_models",
        "n_units",
        "selected_units_sha256",
        "analysis_status",
    ):
        if str(validated[field]) != str(first[field]):
            raise ValueError(
                f"Dark/light manifest has stale derived field {field!r}."
            )
    return validated


def _legacy_identity_table(
    legacy_unit_ids: Sequence[Any],
    resolver: Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]],
) -> pd.DataFrame:
    """Resolve every legacy unit into one persistent and group identity."""
    rows = []
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
        unit_id = str(identity["unit_id"])
        stable_id = f"{merge_id}:{unit_id}"
        group_id = str(identity.get("group_unit_id", legacy_unit_id))
        if not merge_id or not unit_id or not group_id:
            raise ValueError("Resolved legacy identity fields must be non-empty.")
        if stable_id in stable_ids or group_id in group_ids:
            raise ValueError("Resolved legacy identities must be unique.")
        stable_ids.add(stable_id)
        group_ids.add(group_id)
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": stable_id,
                "group_unit_id": group_id,
                "_legacy_unit_id": str(legacy_unit_id),
            }
        )
    return pd.DataFrame.from_records(
        rows,
        columns=(*IDENTITY_COLUMNS, "_legacy_unit_id"),
    )


def _legacy_unit_vectors(
    datasets: Sequence[Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Validate exact legacy unit order and firing-rate vectors everywhere."""
    if not datasets:
        raise ValueError("Legacy registration requires at least one dataset.")
    expected_ids: list[str] | None = None
    expected_dark: np.ndarray | None = None
    expected_light: np.ndarray | None = None
    for dataset in datasets:
        if "unit" not in dataset.dims:
            raise ValueError("Legacy dataset is missing its unit dimension.")
        unit_ids = [str(value) for value in np.asarray(dataset.unit.values)]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Legacy dataset unit identities must be unique.")
        if expected_ids is None:
            expected_ids = unit_ids
        elif unit_ids != expected_ids:
            raise ValueError("Legacy dataset unit coordinate/order is inconsistent.")
        current_vectors = []
        for variable in (
            "dark_movement_firing_rate_hz",
            "light_movement_firing_rate_hz",
        ):
            if variable not in dataset or dataset[variable].dims != ("unit",):
                raise ValueError(
                    f"Legacy dataset is missing unit vector {variable!r}."
                )
            values = np.asarray(dataset[variable], dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("Legacy firing-rate vectors must be finite and non-negative.")
            current_vectors.append(values)
        if expected_dark is None:
            expected_dark, expected_light = current_vectors
        elif not np.allclose(
            current_vectors[0], expected_dark, rtol=1e-10, atol=1e-12
        ) or not np.allclose(
            current_vectors[1], expected_light, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Legacy firing-rate vectors are inconsistent.")
    assert expected_ids is not None
    assert expected_dark is not None
    assert expected_light is not None
    return expected_ids, expected_dark, expected_light


def _legacy_selected_units(
    *,
    legacy_unit_ids: Sequence[str],
    dark_rates: np.ndarray,
    light_rates: np.ndarray,
    selected_units: pd.DataFrame | None,
    unit_identity_resolver: (
        Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]] | None
    ),
) -> pd.DataFrame:
    """Build canonical selected-unit rows from one explicit identity source."""
    if (selected_units is None) == (unit_identity_resolver is None):
        raise ValueError(
            "Provide exactly one of selected_units or unit_identity_resolver."
        )
    if unit_identity_resolver is not None:
        identity = _legacy_identity_table(
            legacy_unit_ids,
            unit_identity_resolver,
        ).loc[:, list(IDENTITY_COLUMNS)]
    else:
        assert selected_units is not None
        missing = [
            column
            for column in (*IDENTITY_COLUMNS, "selection_index")
            if column not in selected_units
        ]
        if missing:
            raise ValueError(f"selected_units is missing columns {missing!r}.")
        identity = selected_units.loc[:, list(IDENTITY_COLUMNS)].copy()
        for field in IDENTITY_COLUMNS:
            identity[field] = identity[field].astype(str)
        if identity["group_unit_id"].tolist() != list(legacy_unit_ids):
            raise ValueError(
                "Legacy unit coordinate/order does not match selected_units "
                "group_unit_id."
            )
        if not np.array_equal(
            pd.to_numeric(
                selected_units["selection_index"], errors="coerce"
            ).to_numpy(dtype=float),
            np.arange(len(selected_units), dtype=float),
        ):
            raise ValueError("selected_units selection_index is not contiguous.")
        for column, expected in (
            ("dark_movement_firing_rate_hz", dark_rates),
            ("light_movement_firing_rate_hz", light_rates),
        ):
            if column in selected_units and not np.allclose(
                pd.to_numeric(
                    selected_units[column], errors="coerce"
                ).to_numpy(dtype=float),
                np.asarray(expected, dtype=float),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"selected_units {column} does not match legacy vectors."
                )
    output = identity.copy()
    output["selection_index"] = np.arange(len(output), dtype=int)
    output["dark_movement_firing_rate_hz"] = np.asarray(dark_rates, dtype=float)
    output["light_movement_firing_rate_hz"] = np.asarray(light_rates, dtype=float)
    output["n_selected_model_trajectory_fits"] = 0
    output["valid_glm_fit"] = False
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _replace_legacy_unit_coordinate(dataset: Any, selected_units: pd.DataFrame) -> Any:
    """Replace an already-validated legacy unit coordinate with group IDs."""
    return dataset.assign_coords(
        unit=(
            "unit",
            selected_units["group_unit_id"].astype(str).to_numpy(),
        )
    )


def register_existing_dark_light_glm_artifact(
    *,
    source_candidate_paths: Sequence[Path],
    source_selected_paths_by_model: Mapping[str, Path],
    source_selection_summary_path: Path,
    destination_path: Path | None,
    dark_light_glm_id: Any,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    selected_units: pd.DataFrame | None = None,
    unit_identity_resolver: (
        Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]] | None
    ) = None,
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    basis_candidate_mode: str,
    basis_candidates: Sequence[float | int],
    parameter_name: str,
    parameter_sha256: str = "",
    output_rule_sha256: str = "",
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SMOOTHING_SIGMA_S,
    source_v1ca1_git_commit: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and copy one exact legacy/current dark-light artifact bundle."""
    metadata = _metadata(
        dark_light_glm_id=dark_light_glm_id,
        animal_name=animal_name,
        date=date,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    summary_path = Path(source_selection_summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Selection summary not found: {summary_path}")
    summary = _load_dataset(summary_path)
    fit_parameters = json.loads(str(summary.attrs.get("fit_parameters_json", "{}")))
    required_parameters = (
        "bin_sizes_s",
        "ridges",
        "n_folds",
        "spline_order",
        "use_speed",
        "speed_feature_mode",
        "n_splines_speed",
        "spline_order_speed",
        "speed_bounds",
        "segment_edges",
    )
    missing = [name for name in required_parameters if name not in fit_parameters]
    if "seed" not in fit_parameters and "random_seed" not in fit_parameters:
        missing.append("seed or random_seed")
    if missing:
        raise ValueError(f"Legacy selection summary is missing fit parameters {missing!r}.")
    if "seed" in fit_parameters and "random_seed" in fit_parameters and (
        int(fit_parameters["seed"]) != int(fit_parameters["random_seed"])
    ):
        raise ValueError("Legacy seed and random_seed values disagree.")
    random_seed = int(
        fit_parameters.get("random_seed", fit_parameters.get("seed"))
    )
    source_speed_sigma = float(
        fit_parameters.get(
            "speed_smoothing_sigma_s",
            speed_smoothing_sigma_s,
        )
    )
    if not np.isclose(
        source_speed_sigma,
        float(speed_smoothing_sigma_s),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Legacy speed_smoothing_sigma_s does not match the selected "
            "parameters."
        )
    parameters = validate_dark_light_glm_parameters(
        basis_candidate_mode=basis_candidate_mode,
        basis_candidates=basis_candidates,
        bin_sizes_s=fit_parameters["bin_sizes_s"],
        ridges=fit_parameters["ridges"],
        n_folds=fit_parameters["n_folds"],
        random_seed=random_seed,
        spline_order=fit_parameters["spline_order"],
        min_dark_firing_rate_hz=fit_parameters.get(
            "min_dark_firing_rate_hz",
            fit_parameters.get(
                "dark_region_threshold_hz",
                fit_parameters.get("region_threshold_hz", 0.0),
            ),
        ),
        min_light_firing_rate_hz=fit_parameters.get(
            "min_light_firing_rate_hz",
            fit_parameters.get(
                "light_region_threshold_hz",
                fit_parameters.get("region_threshold_hz", 0.0),
            ),
        ),
        use_speed=fit_parameters["use_speed"],
        speed_feature_mode=fit_parameters["speed_feature_mode"],
        n_splines_speed=fit_parameters["n_splines_speed"],
        spline_order_speed=fit_parameters["spline_order_speed"],
        speed_smoothing_sigma_s=speed_smoothing_sigma_s,
        speed_bounds=fit_parameters["speed_bounds"],
    )
    parameters.update(
        {
            "parameter_name": _path_component(parameter_name, name="parameter_name"),
            "parameter_sha256": str(parameter_sha256),
            "output_rule_sha256": str(output_rule_sha256),
        }
    )
    if str(summary.attrs.get("schema_version", "")) != parameters["schema_version"]:
        raise ValueError(
            "Legacy schema does not match basis_candidate_mode; v4 requires "
            "n_splines and v5 requires spatial_bin_size_cm."
        )
    summary = _normalize_dataset_schema(
        summary,
        basis_candidate_mode=parameters["basis_candidate_mode"],
    )
    candidate_datasets = {}
    for source_path in source_candidate_paths:
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Candidate artifact not found: {source_path}")
        dataset = _load_dataset(source_path)
        _validate_dataset(dataset, metadata=metadata, parameters=parameters, fit_stage="candidate")
        model_name = str(dataset.attrs["model_name"])
        bin_size_s = float(dataset.attrs["bin_size_s"])
        if parameters["basis_candidate_mode"] == "spatial_bin_size_cm":
            basis_value: float | int = float(dataset.attrs["spatial_bin_size_cm"])
        else:
            basis_value = int(dataset.attrs["n_splines"])
        key = _candidate_key(
            model_name,
            bin_size_s=bin_size_s,
            basis_candidate_mode=parameters["basis_candidate_mode"],
            basis_value=basis_value,
        )
        if key in candidate_datasets:
            raise ValueError(f"Duplicate legacy candidate {key!r}.")
        candidate_datasets[key] = dataset
    if set(source_selected_paths_by_model) != set(MODEL_NAMES):
        raise ValueError("source_selected_paths_by_model must contain all four models.")
    selected_datasets = {}
    for model_name in MODEL_NAMES:
        source_path = Path(source_selected_paths_by_model[model_name])
        if not source_path.is_file():
            raise FileNotFoundError(f"Selected artifact not found: {source_path}")
        dataset = _load_dataset(source_path)
        _validate_dataset(dataset, metadata=metadata, parameters=parameters, fit_stage="selected")
        if str(dataset.attrs.get("model_name", "")) != model_name:
            raise ValueError("Selected artifact model_name does not match its source key.")
        selected_datasets[model_name] = dataset
    legacy_unit_ids, dark_rates, light_rates = _legacy_unit_vectors(
        [*candidate_datasets.values(), *selected_datasets.values()]
    )
    selected_units = _legacy_selected_units(
        legacy_unit_ids=legacy_unit_ids,
        dark_rates=dark_rates,
        light_rates=light_rates,
        selected_units=selected_units,
        unit_identity_resolver=unit_identity_resolver,
    )
    if np.any(dark_rates <= parameters["min_dark_firing_rate_hz"]) or np.any(
        light_rates <= parameters["min_light_firing_rate_hz"]
    ):
        raise ValueError(
            "Legacy selected units do not pass both strict movement-rate "
            "thresholds."
        )
    candidate_datasets = {
        key: _replace_legacy_unit_coordinate(dataset, selected_units)
        for key, dataset in candidate_datasets.items()
    }
    selected_datasets = {
        model_name: _replace_legacy_unit_coordinate(dataset, selected_units)
        for model_name, dataset in selected_datasets.items()
    }
    selected_units, analysis_status = _add_selected_unit_fit_qc(
        selected_units,
        selected_datasets,
    )
    trajectory_length_cm, graph_segment_edges = derive_graph_geometry(
        graph_inputs_by_trajectory
    )
    source_segment_edges = np.asarray(fit_parameters["segment_edges"], dtype=float)
    if not np.allclose(
        source_segment_edges,
        graph_segment_edges,
        rtol=1e-8,
        atol=1e-8,
    ):
        raise ValueError(
            "Legacy segment edges do not match the selected NWB W-track graphs."
        )
    bundle_fit_parameters = {
        **fit_parameters,
        "basis_candidate_mode": parameters["basis_candidate_mode"],
        "basis_candidates": list(parameters["basis_candidates"]),
        "random_seed": parameters["random_seed"],
        "min_dark_firing_rate_hz": parameters["min_dark_firing_rate_hz"],
        "min_light_firing_rate_hz": parameters["min_light_firing_rate_hz"],
        "speed_smoothing_sigma_s": parameters["speed_smoothing_sigma_s"],
        "trajectory_length_cm": trajectory_length_cm,
        "segment_edges": graph_segment_edges.tolist(),
    }
    summary = summary.copy(deep=True)
    summary.attrs["fit_parameters_json"] = json.dumps(
        bundle_fit_parameters,
        sort_keys=True,
    )
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": selected_units,
        "candidate_datasets": candidate_datasets,
        "selected_datasets": selected_datasets,
        "selection_summary": summary,
        "trajectory_length_cm": trajectory_length_cm,
        "segment_edges": graph_segment_edges,
        "analysis_status": analysis_status,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": {
            "source_candidate_paths": [str(Path(path)) for path in source_candidate_paths],
            "source_candidate_sha256": [
                _file_sha256(Path(path)) for path in source_candidate_paths
            ],
            "source_selected_paths": {
                name: str(Path(path))
                for name, path in source_selected_paths_by_model.items()
            },
            "source_selected_sha256": {
                name: _file_sha256(Path(path))
                for name, path in source_selected_paths_by_model.items()
            },
            "source_selection_summary_path": str(summary_path),
            "source_selection_summary_sha256": _file_sha256(summary_path),
            "source_v1ca1_git_commit": source_v1ca1_git_commit,
            "unit_identity_validation": (
                "caller_resolver_for_every_imported_unit"
                if unit_identity_resolver is not None
                else "caller_selected_units_exact_group_order"
            ),
            "speed_smoothing_sigma_s_assumed": (
                "speed_smoothing_sigma_s" not in fit_parameters
            ),
        },
    }
    validated = validate_dark_light_glm_result(result)
    if destination_path is None:
        return validated
    write_dark_light_glm_artifact(validated, destination_path, overwrite=overwrite)
    return load_dark_light_glm_artifact(destination_path)


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
    """Return one canonical typed DarkLightGLM scratch-table frame."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != tuple(
        columns
    ):
        raise ValueError(
            "DarkLightGLM NWB table does not have its canonical schema."
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
                f"DarkLightGLM NWB column {column!r} must be boolean."
            )
        output[column] = np.asarray(values, dtype=bool)
    for column in vector_columns:
        vectors = [np.asarray(value, dtype=float) for value in output[column]]
        if any(vector.ndim != 1 or np.isinf(vector).any() for vector in vectors):
            raise ValueError(
                f"DarkLightGLM NWB vector column {column!r} is invalid."
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
                f"DarkLightGLM NWB numeric column {column!r} contains infinity."
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
    """Convert scalar keys and aligned numeric vectors to a DynamicTable."""
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
                f"Unexpected DarkLightGLM NWB object {nwb_table.name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("DarkLightGLM NWB objects must be DynamicTables.")
    table = table.reset_index(drop=True)
    observed = tuple(str(column) for column in table.columns)
    if set(observed) != set(columns) or len(observed) != len(columns):
        raise ValueError("DarkLightGLM NWB object has a noncanonical schema.")
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


def _dataset_records(result: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    """Return every xarray dataset in deterministic storage order."""
    canonical = validate_dark_light_glm_result(result)
    records = [
        (key, "candidate", dataset)
        for key, dataset in sorted(canonical["candidate_datasets"].items())
    ]
    records.extend(
        (
            model_name,
            (
                "terminal"
                if canonical["analysis_status"]
                in {"no_units", "no_eligible_units", "no_valid_position", "no_movement"}
                else "selected"
            ),
            canonical["selected_datasets"][model_name],
        )
        for model_name in MODEL_NAMES
    )
    records.append(
        ("selection_summary", "selection_summary", canonical["selection_summary"])
    )
    return records


def _dataset_index_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return one metadata row per candidate, selected model, and summary."""
    rows = []
    for dataset_key, fit_stage, dataset in _dataset_records(result):
        rows.append(
            {
                "dataset_key": dataset_key,
                "fit_stage": fit_stage,
                "model_name": str(dataset.attrs.get("model_name", "")),
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
    component_name: str,
    array: Any,
) -> dict[str, Any]:
    """Flatten one xarray coordinate or variable without numeric JSON."""
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
                f"DarkLightGLM array {component_name!r} contains infinity."
            )
        dtype = str(values.dtype)
        text_json = "[]"
    else:
        raise TypeError(
            f"DarkLightGLM array {component_name!r} has unsupported dtype "
            f"{values.dtype}."
        )
    return {
        "dataset_key": dataset_key,
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


def _axes_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return every named coordinate as one keyed ragged row."""
    rows = []
    for dataset_key, _fit_stage, dataset in _dataset_records(result):
        rows.extend(
            _array_component_record(
                dataset_key=dataset_key,
                component_name=str(name),
                array=coordinate,
            )
            for name, coordinate in dataset.coords.items()
        )
    return pd.DataFrame.from_records(rows, columns=ARRAY_COMPONENT_COLUMNS)


def _variables_frame(
    result: Mapping[str, Any],
    *,
    fit_stages: set[str],
) -> pd.DataFrame:
    """Return flattened data variables for selected dataset stages."""
    rows = []
    for dataset_key, fit_stage, dataset in _dataset_records(result):
        if fit_stage not in fit_stages:
            continue
        rows.extend(
            _array_component_record(
                dataset_key=dataset_key,
                component_name=str(name),
                array=variable,
            )
            for name, variable in dataset.data_vars.items()
        )
    return pd.DataFrame.from_records(rows, columns=ARRAY_COMPONENT_COLUMNS)


def _provenance_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return detached result metadata and global geometry."""
    canonical = validate_dark_light_glm_result(result)
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
                "trajectory_length_cm": float(canonical["trajectory_length_cm"]),
                "segment_edges": np.asarray(canonical["segment_edges"], dtype=float),
                "analysis_status": str(canonical["analysis_status"]),
                "artifact_origin": str(canonical["artifact_origin"]),
                "legacy_artifact_provenance_json": json.dumps(
                    _json_ready(canonical["legacy_artifact_provenance"] or {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
            }
        ],
        columns=PROVENANCE_COLUMNS,
    )


def dark_light_glm_selected_units_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert the complete selected-unit audit to an NWB DynamicTable."""
    return _dynamic_table_from_frame(
        table,
        name=NWB_SELECTED_UNITS_TABLE_NAME,
        description=(
            "Selected-unit identity and fit audit for DarkLightGLM; "
            f"v1ca1 NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=IDENTITY_COLUMNS,
        integer_columns=("selection_index", "n_selected_model_trajectory_fits"),
        boolean_columns=("valid_glm_fit",),
    )


def dark_light_glm_selected_units_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load the selected-unit audit from its NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_SELECTED_UNITS_TABLE_NAME,
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=IDENTITY_COLUMNS,
        integer_columns=("selection_index", "n_selected_model_trajectory_fits"),
        boolean_columns=("valid_glm_fit",),
    )


def _component_frame_to_dynamic_table(
    frame: pd.DataFrame,
    *,
    name: str,
    description: str,
) -> Any:
    """Store flattened xarray components as keyed ragged rows."""
    return _ragged_dynamic_table_from_frame(
        frame,
        name=name,
        description=description,
        columns=ARRAY_COMPONENT_COLUMNS,
        vector_columns=("numeric_values",),
        text_columns=(
            "dataset_key",
            "component_name",
            "dimensions_json",
            "shape_json",
            "dtype",
            "text_values_json",
            "attrs_json",
        ),
        integer_columns=("numeric_count",),
    )


def _component_frame_from_dynamic_table(nwb_table: Any, *, name: str) -> pd.DataFrame:
    """Load flattened xarray components from one scratch table."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=name,
        columns=ARRAY_COMPONENT_COLUMNS,
        vector_columns=("numeric_values",),
        text_columns=(
            "dataset_key",
            "component_name",
            "dimensions_json",
            "shape_json",
            "dtype",
            "text_values_json",
            "attrs_json",
        ),
        integer_columns=("numeric_count",),
    )


def dark_light_glm_result_to_nwb_objects(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one DarkLightGLM result to seven NWB scratch objects."""
    canonical = validate_dark_light_glm_result(result)
    return {
        "selected_units": dark_light_glm_selected_units_to_dynamic_table(
            canonical["selected_units"]
        ),
        "dataset_index": _dynamic_table_from_frame(
            _dataset_index_frame(canonical),
            name=NWB_DATASET_INDEX_TABLE_NAME,
            description="Candidate, selected, and selection-summary dataset index.",
            columns=DATASET_INDEX_COLUMNS,
            text_columns=DATASET_INDEX_COLUMNS,
        ),
        "axes": _component_frame_to_dynamic_table(
            _axes_frame(canonical),
            name=NWB_AXES_TABLE_NAME,
            description="Coordinates for every stored DarkLightGLM dataset.",
        ),
        "candidate_results": _component_frame_to_dynamic_table(
            _variables_frame(canonical, fit_stages={"candidate"}),
            name=NWB_CANDIDATE_RESULTS_TABLE_NAME,
            description="Complete candidate-search arrays for DarkLightGLM.",
        ),
        "selected_results": _component_frame_to_dynamic_table(
            _variables_frame(canonical, fit_stages={"selected", "terminal"}),
            name=NWB_SELECTED_RESULTS_TABLE_NAME,
            description="Selected-ridge model arrays used by downstream analyses.",
        ),
        "selection_summary": _component_frame_to_dynamic_table(
            _variables_frame(canonical, fit_stages={"selection_summary"}),
            name=NWB_SELECTION_SUMMARY_TABLE_NAME,
            description="Complete DarkLightGLM hyperparameter-search summary arrays.",
        ),
        "provenance": _ragged_dynamic_table_from_frame(
            _provenance_frame(canonical),
            name=NWB_PROVENANCE_TABLE_NAME,
            description=(
                "Detached DarkLightGLM identity, parameters, geometry, and status."
            ),
            columns=PROVENANCE_COLUMNS,
            vector_columns=("segment_edges",),
            text_columns=(
                "metadata_json",
                "parameters_json",
                "analysis_status",
                "artifact_origin",
                "legacy_artifact_provenance_json",
                "artifact_schema_version",
            ),
        ),
    }


def _parse_json_mapping(value: str, *, name: str) -> dict[str, Any]:
    """Parse one JSON object with a field-specific error."""
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DarkLightGLM {name} contains malformed JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"DarkLightGLM {name} must encode a mapping.")
    return dict(decoded)


def _decode_array_component(
    record: Mapping[str, Any],
) -> tuple[tuple[str, ...], Any, dict[str, Any]]:
    """Restore one xarray component from its flattened NWB row."""
    try:
        dimensions = tuple(
            str(value) for value in json.loads(record["dimensions_json"])
        )
        shape = tuple(int(value) for value in json.loads(record["shape_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("DarkLightGLM array shape metadata is malformed.") from exc
    if len(dimensions) != len(shape) or any(value < 0 for value in shape):
        raise ValueError("DarkLightGLM array dimensions and shape disagree.")
    expected_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    dtype = str(record["dtype"])
    if dtype == "str":
        try:
            values = json.loads(record["text_values_json"])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "DarkLightGLM text array contains malformed JSON."
            ) from exc
        if not isinstance(values, list) or len(values) != expected_size:
            raise ValueError("DarkLightGLM text array size disagrees with its shape.")
        array = np.asarray([str(value) for value in values], dtype=str)
    else:
        count = int(record["numeric_count"])
        values = np.asarray(record["numeric_values"], dtype=float)[:count]
        if count != expected_size or values.size != expected_size:
            raise ValueError(
                "DarkLightGLM numeric array size disagrees with its shape."
            )
        try:
            target_dtype = np.dtype(dtype)
        except TypeError as exc:
            raise ValueError("DarkLightGLM array dtype is unsupported.") from exc
        array = values.astype(target_dtype)
    attrs = _parse_json_mapping(str(record["attrs_json"]), name="array attrs")
    return dimensions, array.reshape(shape), attrs


def _dataset_from_nwb_frames(
    *,
    index_record: Mapping[str, Any],
    axes: pd.DataFrame,
    variables: pd.DataFrame,
) -> Any:
    """Rebuild one exact xarray Dataset from keyed component rows."""
    import xarray as xr

    dataset_key = str(index_record["dataset_key"])
    coordinate_values = {}
    coordinate_attrs = {}
    for record in axes.loc[
        axes["dataset_key"].astype(str) == dataset_key
    ].to_dict("records"):
        name = str(record["component_name"])
        dimensions, values, attrs = _decode_array_component(record)
        coordinate_values[name] = (dimensions, values)
        coordinate_attrs[name] = attrs
    data_values = {}
    data_attrs = {}
    for record in variables.loc[
        variables["dataset_key"].astype(str) == dataset_key
    ].to_dict("records"):
        name = str(record["component_name"])
        dimensions, values, attrs = _decode_array_component(record)
        data_values[name] = (dimensions, values)
        data_attrs[name] = attrs
    dataset = xr.Dataset(
        data_vars=data_values,
        coords=coordinate_values,
        attrs=_parse_json_mapping(
            str(index_record["attrs_json"]), name="dataset attrs"
        ),
    )
    for name, attrs in coordinate_attrs.items():
        dataset.coords[name].attrs.update(attrs)
    for name, attrs in data_attrs.items():
        dataset[name].attrs.update(attrs)
    return dataset


def dark_light_glm_result_from_nwb_objects(
    *,
    selected_units: Any,
    dataset_index: Any,
    axes: Any,
    candidate_results: Any,
    selected_results: Any,
    selection_summary: Any,
    provenance: Any,
) -> dict[str, Any]:
    """Reconstruct and validate one result from seven NWB objects."""
    selected = dark_light_glm_selected_units_from_dynamic_table(selected_units)
    index = _frame_from_dynamic_table(
        dataset_index,
        expected_name=NWB_DATASET_INDEX_TABLE_NAME,
        columns=DATASET_INDEX_COLUMNS,
        text_columns=DATASET_INDEX_COLUMNS,
    )
    if index["dataset_key"].duplicated().any():
        raise ValueError("DarkLightGLM NWB dataset keys must be unique.")
    axes_frame = _component_frame_from_dynamic_table(
        axes, name=NWB_AXES_TABLE_NAME
    )
    candidate_frame = _component_frame_from_dynamic_table(
        candidate_results, name=NWB_CANDIDATE_RESULTS_TABLE_NAME
    )
    selected_frame = _component_frame_from_dynamic_table(
        selected_results, name=NWB_SELECTED_RESULTS_TABLE_NAME
    )
    summary_frame = _component_frame_from_dynamic_table(
        selection_summary, name=NWB_SELECTION_SUMMARY_TABLE_NAME
    )
    provenance_frame = _frame_from_dynamic_table(
        provenance,
        expected_name=NWB_PROVENANCE_TABLE_NAME,
        columns=PROVENANCE_COLUMNS,
        vector_columns=("segment_edges",),
        text_columns=(
            "metadata_json",
            "parameters_json",
            "analysis_status",
            "artifact_origin",
            "legacy_artifact_provenance_json",
            "artifact_schema_version",
        ),
    )
    if len(provenance_frame) != 1:
        raise ValueError("DarkLightGLM provenance must contain exactly one row.")
    source = provenance_frame.iloc[0].to_dict()
    if source["artifact_schema_version"] != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("DarkLightGLM NWB artifact schema version is unsupported.")
    candidate_datasets = {}
    selected_datasets = {}
    summary = None
    for record in index.to_dict("records"):
        fit_stage = str(record["fit_stage"])
        if fit_stage == "candidate":
            dataset = _dataset_from_nwb_frames(
                index_record=record,
                axes=axes_frame,
                variables=candidate_frame,
            )
            candidate_datasets[str(record["dataset_key"])] = dataset
        elif fit_stage in {"selected", "terminal"}:
            dataset = _dataset_from_nwb_frames(
                index_record=record,
                axes=axes_frame,
                variables=selected_frame,
            )
            selected_datasets[str(record["dataset_key"])] = dataset
        elif fit_stage == "selection_summary":
            if summary is not None:
                raise ValueError("DarkLightGLM NWB contains multiple summaries.")
            summary = _dataset_from_nwb_frames(
                index_record=record,
                axes=axes_frame,
                variables=summary_frame,
            )
        else:
            raise ValueError(f"Unsupported DarkLightGLM NWB fit stage {fit_stage!r}.")
    if summary is None:
        raise ValueError("DarkLightGLM NWB is missing its selection summary.")
    legacy = _parse_json_mapping(
        str(source["legacy_artifact_provenance_json"]),
        name="legacy provenance",
    )
    return validate_dark_light_glm_result(
        {
            "metadata": _parse_json_mapping(
                str(source["metadata_json"]), name="metadata"
            ),
            "parameters": _parse_json_mapping(
                str(source["parameters_json"]), name="parameters"
            ),
            "selected_units": selected,
            "candidate_datasets": candidate_datasets,
            "selected_datasets": selected_datasets,
            "selection_summary": summary,
            "trajectory_length_cm": float(source["trajectory_length_cm"]),
            "segment_edges": np.asarray(source["segment_edges"], dtype=float),
            "analysis_status": str(source["analysis_status"]),
            "artifact_origin": str(source["artifact_origin"]),
            "legacy_artifact_provenance": legacy or None,
        }
    )


def _semantic_frame_sha256(
    table: pd.DataFrame,
    *,
    columns: Sequence[str],
    vector_columns: Sequence[str] = (),
) -> str:
    """Hash one canonical frame without depending on HDF5 object identity."""
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


def _dataset_semantic_sha256(dataset: Any, *, dataset_key: str) -> str:
    """Hash one xarray dataset through its NWB storage representation."""
    index = pd.DataFrame.from_records(
        [
            {
                "dataset_key": dataset_key,
                "fit_stage": str(dataset.attrs.get("fit_stage", "")),
                "model_name": str(dataset.attrs.get("model_name", "")),
                "attrs_json": json.dumps(
                    _json_ready(dict(dataset.attrs)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        ],
        columns=DATASET_INDEX_COLUMNS,
    )
    axes = pd.DataFrame.from_records(
        [
            _array_component_record(
                dataset_key=dataset_key,
                component_name=str(name),
                array=value,
            )
            for name, value in dataset.coords.items()
        ],
        columns=ARRAY_COMPONENT_COLUMNS,
    )
    variables = pd.DataFrame.from_records(
        [
            _array_component_record(
                dataset_key=dataset_key,
                component_name=str(name),
                array=value,
            )
            for name, value in dataset.data_vars.items()
        ],
        columns=ARRAY_COMPONENT_COLUMNS,
    )
    component_hashes = (
        _semantic_frame_sha256(index, columns=DATASET_INDEX_COLUMNS),
        _semantic_frame_sha256(
            axes,
            columns=ARRAY_COMPONENT_COLUMNS,
            vector_columns=("numeric_values",),
        ),
        _semantic_frame_sha256(
            variables,
            columns=ARRAY_COMPONENT_COLUMNS,
            vector_columns=("numeric_values",),
        ),
    )
    return hashlib.sha256("".join(component_hashes).encode()).hexdigest()


def dark_light_glm_selected_model_sha256s(
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Return one semantic digest for each selected DarkLightGLM model."""
    canonical = validate_dark_light_glm_result(result)
    return {
        model_name: _dataset_semantic_sha256(
            canonical["selected_datasets"][model_name],
            dataset_key=model_name,
        )
        for model_name in MODEL_NAMES
    }


def dark_light_glm_nwb_hashes(result: Mapping[str, Any]) -> dict[str, str]:
    """Return semantic hashes for all seven NWB objects and the result."""
    canonical = validate_dark_light_glm_result(result)
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
        "axes_sha256": (
            _axes_frame(canonical),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "candidate_results_sha256": (
            _variables_frame(canonical, fit_stages={"candidate"}),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "selected_results_sha256": (
            _variables_frame(canonical, fit_stages={"selected", "terminal"}),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "selection_summary_sha256": (
            _variables_frame(canonical, fit_stages={"selection_summary"}),
            ARRAY_COMPONENT_COLUMNS,
            ("numeric_values",),
        ),
        "provenance_sha256": (
            _provenance_frame(canonical),
            PROVENANCE_COLUMNS,
            ("segment_edges",),
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
    hashes["dark_light_glm_sha256"] = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return hashes


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "BASIS_CANDIDATE_MODES",
    "CURRENT_PARAMETERS",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_BIN_SIZES_S",
    "DEFAULT_RIDGES",
    "DEFAULT_SPEED_SMOOTHING_SIGMA_S",
    "DEFAULT_SPATIAL_BIN_SIZES_CM",
    "LEGACY_N_SPLINES",
    "LEGACY_PARAMETERS",
    "MODEL_NAMES",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "SCHEMA_VERSION_BY_MODE",
    "compute_dark_light_glm",
    "dark_light_glm_nwb_hashes",
    "dark_light_glm_result_from_nwb_objects",
    "dark_light_glm_result_to_nwb_objects",
    "dark_light_glm_selected_model_sha256s",
    "dark_light_glm_selected_units_from_dynamic_table",
    "dark_light_glm_selected_units_to_dynamic_table",
    "derive_graph_geometry",
    "get_dark_light_glm_artifact_paths",
    "load_dark_light_glm_artifact",
    "register_existing_dark_light_glm_artifact",
    "validate_dark_light_glm_parameters",
    "validate_dark_light_glm_result",
    "write_dark_light_glm_artifact",
]
