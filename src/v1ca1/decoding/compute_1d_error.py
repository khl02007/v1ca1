from __future__ import annotations

"""Compute trajectory-aligned 1D ahead/behind decoding error.

The script loads combined `predict_1d` posterior outputs, maps each 1D MAP
position back onto the fitted W-track graph, and computes signed
ahead/behind distance using the animal's head direction. Only bins inside
trajectory intervals are included. Per-region raw tables, trajectory-position
summaries, and PNG figures are saved under the modern analysis directory.
"""

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.decoding._1d import (
    DEFAULT_BRANCH_GAP_CM,
    DEFAULT_DATA_ROOT,
    DEFAULT_MOVEMENT_VAR,
    DEFAULT_N_FOLDS,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_POSITION_STD,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    DEFAULT_TIME_BIN_SIZE_S,
    DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD,
    DISCRETE_VAR_CHOICES,
    REGIONS,
    build_classifier_output_paths,
    build_prediction_output_paths,
    build_time_grid,
    get_analysis_path_for_session,
    get_fit_output_dir,
    get_predict_output_dir,
    get_unit_selection_label,
    interpolate_position_to_time,
    intervalset_to_arrays,
    load_classifier,
    load_required_session_inputs,
    preflight_no_existing,
    require_existing_paths,
    select_regions,
)
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import TRAJECTORY_TYPES, load_clean_dlc_position_data
from v1ca1.helper.wtrack import (
    get_wtrack_branch_graph,
    get_wtrack_branch_side,
    get_wtrack_direction,
    get_wtrack_total_length,
)


SCRIPT_NAME = "v1ca1.decoding.compute_1d_error"
POSTERIOR_NAME = "acausal_posterior"
DEFAULT_MIN_BIN_COUNT = 20
MAP_CHUNK_SIZE = 100_000
TRAJECTORY_PANEL_ORDER = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
RAW_COLUMNS = (
    "time",
    "region",
    "trajectory_type",
    "lap_index",
    "cv_fold",
    "true_full_w_position_cm",
    "true_trajectory_position_cm",
    "decoded_full_w_position_cm",
    "ahead_behind_distance_cm",
    "abs_ahead_behind_distance_cm",
    "speed_cm_s",
    "actual_edge_id",
    "decoded_edge_id",
    "posterior_max_probability",
)
SUMMARY_COLUMNS = (
    "region",
    "trajectory_type",
    "bin_left",
    "bin_right",
    "bin_center",
    "n",
    "ahead_behind_median",
    "ahead_behind_q25",
    "ahead_behind_q75",
    "abs_ahead_behind_median",
    "abs_ahead_behind_q25",
    "abs_ahead_behind_q75",
)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments."""
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if args.time_bin_size_s <= 0:
        raise ValueError("--time-bin-size-s must be positive.")
    if args.position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if args.speed_threshold_cm_s < 0:
        raise ValueError("--speed-threshold-cm-s must be non-negative.")
    if args.position_std <= 0:
        raise ValueError("--position-std must be positive.")
    if args.place_bin_size <= 0:
        raise ValueError("--place-bin-size must be positive.")
    if args.movement_var <= 0:
        raise ValueError("--movement-var must be positive.")
    if args.branch_gap_cm < 0:
        raise ValueError("--branch-gap-cm must be non-negative.")
    if args.v1_ripple_glm_devexp_threshold is not None:
        if not np.isfinite(args.v1_ripple_glm_devexp_threshold):
            raise ValueError("--v1-ripple-glm-devexp-threshold must be finite.")
    if args.position_bin_size_cm is not None and args.position_bin_size_cm <= 0:
        raise ValueError("--position-bin-size-cm must be positive.")
    if args.min_bin_count < 1:
        raise ValueError("--min-bin-count must be at least 1.")
    if args.ylim_cm is not None:
        if not np.isfinite(args.ylim_cm):
            raise ValueError("--ylim-cm must be finite.")
        if args.ylim_cm <= 0:
            raise ValueError("--ylim-cm must be positive.")


def get_error_output_dir(analysis_path: Path) -> Path:
    """Return the output directory for 1D decoding-error tables."""
    return analysis_path / "decoding" / "error" / "1d"


def get_error_figure_dir(analysis_path: Path) -> Path:
    """Return the output directory for 1D decoding-error figures."""
    return analysis_path / "figs" / "decoding" / "error" / "1d"


def format_compact_path_value(value: Any) -> str:
    """Return a compact filesystem-safe representation of one path token."""
    if isinstance(value, bool):
        return "T" if value else "F"
    if isinstance(value, (float, np.floating)):
        text = np.format_float_positional(float(value), trim="-")
        return text.replace("-", "m").replace(".", "p")
    text = str(value)
    return (
        text.replace(" ", "")
        .replace("/", "-")
        .replace(".", "p")
        .replace("-", "m")
    )


def format_compact_unit_selection_label(unit_selection_label: str) -> str:
    """Return a compact filename token for one unit-selection mode."""
    if unit_selection_label == "all_units":
        return "all"
    if unit_selection_label == "v1_units_ripple_devexp_p_lt_0p05":
        return "v1glm-p0p05"
    threshold_prefix = "v1_units_ripple_devexp_ge_"
    if unit_selection_label.startswith(threshold_prefix):
        threshold = unit_selection_label.removeprefix(threshold_prefix)
        return f"v1glm-devexpge{threshold}"
    return format_compact_path_value(unit_selection_label)


def get_short_prediction_hash(prediction_path: Path) -> str:
    """Return a short stable hash for the exact prediction filename stem."""
    return hashlib.sha1(prediction_path.stem.encode("utf-8")).hexdigest()[:8]


def build_region_output_paths(
    *,
    prediction_paths: dict[str, Path],
    table_output_dir: Path,
    figure_output_dir: Path,
    epoch: str,
    n_folds: int,
    random_state: int,
    time_bin_size_s: float,
    speed_threshold_cm_s: float,
    position_std: float,
    place_bin_size: float,
    movement_var: float,
    discrete_var: str,
    branch_gap_cm: float,
    direction: bool,
    movement: bool,
    unit_selection_label: str,
    position_bin_size_cm: float,
    min_bin_count: int,
) -> dict[str, dict[str, Path]]:
    """Return raw-table, summary-table, and figure paths for each region."""
    output_paths: dict[str, dict[str, Path]] = {}
    for region, prediction_path in prediction_paths.items():
        stem = "_".join(
            (
                region,
                epoch,
                f"cv{n_folds}",
                f"rs{random_state}",
                f"tb{format_compact_path_value(time_bin_size_s)}",
                f"spd{format_compact_path_value(speed_threshold_cm_s)}",
                f"std{format_compact_path_value(position_std)}",
                f"pb{format_compact_path_value(place_bin_size)}",
                f"mv{format_compact_path_value(movement_var)}",
                f"disc{format_compact_path_value(discrete_var)}",
                f"gap{format_compact_path_value(branch_gap_cm)}",
                f"dir{format_compact_path_value(direction)}",
                f"mov{format_compact_path_value(movement)}",
                f"units{format_compact_unit_selection_label(unit_selection_label)}",
                f"pbin{format_compact_path_value(position_bin_size_cm)}",
                f"h{get_short_prediction_hash(prediction_path)}",
            )
        )
        summary_stem = f"{stem}_min{min_bin_count}"
        output_paths[region] = {
            "raw": table_output_dir / f"ahead_behind_{stem}.parquet",
            "summary": table_output_dir / f"ahead_behind_summary_{summary_stem}.parquet",
            "figure": figure_output_dir / f"ahead_behind_{summary_stem}.png",
        }
    return output_paths


def flatten_output_paths(output_paths: dict[str, dict[str, Path]]) -> list[Path]:
    """Return all output paths from a nested region output-path mapping."""
    return [path for paths_by_kind in output_paths.values() for path in paths_by_kind.values()]


def get_classifier_environment(classifier: Any) -> Any:
    """Return the single RTC environment from one saved classifier."""
    if hasattr(classifier, "environments"):
        environments = classifier.environments
        if len(environments) != 1:
            raise ValueError(
                "Expected exactly one classifier environment for 1D decoding. "
                f"Found {len(environments)}."
            )
        return environments[0]
    if hasattr(classifier, "environment"):
        return classifier.environment
    raise ValueError("Could not find an RTC environment on the saved classifier.")


def get_environment_positions(environment: Any) -> np.ndarray:
    """Return 1D position-bin centers from one RTC environment."""
    if hasattr(environment, "place_bin_centers_"):
        positions = np.asarray(environment.place_bin_centers_, dtype=float).squeeze()
    elif hasattr(environment, "place_bin_centers_nodes_df_"):
        positions = np.asarray(
            environment.place_bin_centers_nodes_df_["linear_position"],
            dtype=float,
        )
    else:
        raise ValueError("Classifier environment does not expose position-bin centers.")

    if positions.ndim != 1:
        raise ValueError(
            "Classifier environment position-bin centers must be one-dimensional. "
            f"Got shape {positions.shape}."
        )
    return positions


def validate_prediction_dataset(
    dataset: Any,
    *,
    prediction_path: Path,
    time_grid: np.ndarray,
) -> None:
    """Validate one prediction dataset against expected coordinates."""
    required_variables = (
        POSTERIOR_NAME,
        "true_linear_position",
        "speed_cm_s",
        "cv_fold",
    )
    missing_variables = [name for name in required_variables if name not in dataset]
    if missing_variables:
        raise ValueError(
            f"Prediction output is missing required variables {missing_variables!r}: "
            f"{prediction_path}"
        )
    if "position" not in dataset.coords:
        raise ValueError(f"Prediction output is missing 'position' coordinate: {prediction_path}")

    predicted_time = np.asarray(dataset.time, dtype=float)
    expected_time = np.asarray(time_grid, dtype=float)
    if predicted_time.size != expected_time.size:
        raise ValueError(
            f"Prediction output {prediction_path} has {predicted_time.size} time bins, "
            f"but the current settings expect {expected_time.size}."
        )
    if not np.allclose(predicted_time, expected_time, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Prediction output {prediction_path} does not match the current time grid. "
            "Check --time-bin-size-s, --position-offset, --animal-name, --date, "
            "and --epoch."
        )


def validate_environment_matches_prediction(
    environment: Any,
    dataset: Any,
    *,
    classifier_path: Path,
    prediction_path: Path,
) -> np.ndarray:
    """Validate classifier environment position bins against prediction bins."""
    environment_positions = get_environment_positions(environment)
    prediction_positions = np.asarray(dataset.position, dtype=float)
    if environment_positions.shape != prediction_positions.shape:
        raise ValueError(
            "Classifier and prediction position-bin counts differ. "
            f"Classifier {classifier_path}: {environment_positions.shape}; "
            f"prediction {prediction_path}: {prediction_positions.shape}."
        )
    if not np.allclose(environment_positions, prediction_positions, rtol=0.0, atol=1e-9):
        raise ValueError(
            "Classifier environment position bins do not match prediction output. "
            f"Classifier: {classifier_path}; prediction: {prediction_path}"
        )
    return environment_positions


def points_toward_node(track_graph: Any, edge: np.ndarray, head_direction: float) -> Any:
    """Return the graph node one head direction points toward along an edge."""
    edge = np.asarray(edge)
    node1_position = np.asarray(track_graph.nodes[edge[0]]["pos"], dtype=float)
    node2_position = np.asarray(track_graph.nodes[edge[1]]["pos"], dtype=float)
    edge_vector = node2_position - node1_position
    head_vector = np.asarray([np.cos(head_direction), np.sin(head_direction)])
    return edge[(edge_vector @ head_vector >= 0).astype(int)]


def get_distance_between_nodes(track_graph: Any, node1: Any, node2: Any) -> float:
    """Return Euclidean distance between two positioned graph nodes."""
    node1_position = np.asarray(track_graph.nodes[node1]["pos"], dtype=float)
    node2_position = np.asarray(track_graph.nodes[node2]["pos"], dtype=float)
    return float(np.sqrt(np.sum((node1_position - node2_position) ** 2)))


def setup_track_graph(
    track_graph: Any,
    actual_position: np.ndarray,
    actual_edge: np.ndarray,
    head_direction: float,
    mental_position: np.ndarray,
    mental_edge: np.ndarray,
) -> Any:
    """Insert actual, mental, and head nodes into a track graph copy."""
    track_graph.add_node("actual_position", pos=actual_position)
    track_graph.add_node("head", pos=actual_position)
    track_graph.add_node("mental_position", pos=mental_position)

    node_ahead = points_toward_node(track_graph, actual_edge, head_direction)
    node_behind = actual_edge[~np.isin(actual_edge, node_ahead)][0]

    if np.all(actual_edge == mental_edge):
        actual_position_distance = get_distance_between_nodes(
            track_graph,
            "actual_position",
            node_ahead,
        )
        mental_position_distance = get_distance_between_nodes(
            track_graph,
            "mental_position",
            node_ahead,
        )
        if actual_position_distance < mental_position_distance:
            node_order = [
                node_ahead,
                "head",
                "actual_position",
                "mental_position",
                node_behind,
            ]
        else:
            node_order = [
                node_ahead,
                "mental_position",
                "head",
                "actual_position",
                node_behind,
            ]
    else:
        node_order = [node_ahead, "head", "actual_position", node_behind]

        distance = get_distance_between_nodes(
            track_graph,
            mental_edge[0],
            "mental_position",
        )
        track_graph.add_edge(mental_edge[0], "mental_position", distance=distance)

        distance = get_distance_between_nodes(
            track_graph,
            "mental_position",
            mental_edge[1],
        )
        track_graph.add_edge("mental_position", mental_edge[1], distance=distance)

    for node1, node2 in zip(node_order[:-1], node_order[1:], strict=True):
        distance = get_distance_between_nodes(track_graph, node1, node2)
        track_graph.add_edge(node1, node2, distance=distance)

    return track_graph


def calculate_ahead_behind(
    track_graph: Any,
    *,
    source: str = "actual_position",
    target: str = "mental_position",
) -> int:
    """Return 1 if the graph path to target goes through the head node, else -1."""
    import networkx as nx

    path = nx.shortest_path(
        track_graph,
        source=source,
        target=target,
        weight="distance",
    )
    return 1 if "head" in path else -1


def calculate_distance(
    track_graph: Any,
    *,
    source: str = "actual_position",
    target: str = "mental_position",
) -> float:
    """Return shortest-path graph distance between inserted actual and mental nodes."""
    import networkx as nx

    return float(
        nx.shortest_path_length(
            track_graph,
            source=source,
            target=target,
            weight="distance",
        )
    )


def compute_ahead_behind_distance(
    *,
    track_graph: Any,
    actual_projected_position: np.ndarray,
    actual_edge_id: np.ndarray,
    head_direction: np.ndarray,
    mental_position_2d: np.ndarray,
    mental_edge_id: np.ndarray,
) -> np.ndarray:
    """Return signed ahead/behind distance for all valid time bins."""
    graph_edges = np.asarray(list(track_graph.edges), dtype=object)
    n_edges = graph_edges.shape[0]
    actual_edge_id = np.asarray(actual_edge_id, dtype=int)
    mental_edge_id = np.asarray(mental_edge_id, dtype=int)
    ahead_behind_distance = np.full(actual_edge_id.shape, np.nan, dtype=float)

    valid = (
        np.isfinite(actual_projected_position).all(axis=1)
        & np.isfinite(head_direction)
        & np.isfinite(mental_position_2d).all(axis=1)
        & (actual_edge_id >= 0)
        & (actual_edge_id < n_edges)
        & (mental_edge_id >= 0)
        & (mental_edge_id < n_edges)
    )
    copy_graph = track_graph.copy()
    for index in np.flatnonzero(valid):
        copy_graph = setup_track_graph(
            copy_graph,
            actual_projected_position[index],
            graph_edges[actual_edge_id[index]],
            float(head_direction[index]),
            mental_position_2d[index],
            graph_edges[mental_edge_id[index]],
        )
        distance = calculate_distance(copy_graph)
        ahead_behind = calculate_ahead_behind(copy_graph)
        ahead_behind_distance[index] = ahead_behind * distance

        copy_graph.remove_node("actual_position")
        copy_graph.remove_node("head")
        copy_graph.remove_node("mental_position")

    return ahead_behind_distance


def build_trajectory_samples(
    *,
    animal_name: str,
    time_grid: np.ndarray,
    head_position: np.ndarray,
    trajectory_intervals: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return time-bin indices and branch-aligned positions inside trajectories."""
    import track_linearization as tl

    index_chunks: list[np.ndarray] = []
    trajectory_chunks: list[np.ndarray] = []
    lap_chunks: list[np.ndarray] = []
    position_chunks: list[np.ndarray] = []

    for trajectory_type in TRAJECTORY_TYPES:
        starts, ends = intervalset_to_arrays(trajectory_intervals[trajectory_type])
        trajectory_indices: list[np.ndarray] = []
        trajectory_laps: list[np.ndarray] = []
        for lap_index, (start_time, end_time) in enumerate(
            zip(starts, ends, strict=True)
        ):
            indices = np.flatnonzero((time_grid > start_time) & (time_grid <= end_time))
            if indices.size == 0:
                continue
            trajectory_indices.append(indices)
            trajectory_laps.append(np.full(indices.shape, lap_index, dtype=int))

        if not trajectory_indices:
            continue

        indices = np.concatenate(trajectory_indices)
        laps = np.concatenate(trajectory_laps)
        track_graph, edge_order = get_wtrack_branch_graph(
            animal_name=animal_name,
            branch_side=get_wtrack_branch_side(trajectory_type),
            direction=get_wtrack_direction(trajectory_type),
        )
        trajectory_position = np.full(indices.shape, np.nan, dtype=float)
        finite_position = np.isfinite(head_position[indices]).all(axis=1)
        if np.any(finite_position):
            position_df = tl.get_linearized_position(
                position=head_position[indices][finite_position],
                track_graph=track_graph,
                edge_order=edge_order,
                edge_spacing=0,
            )
            trajectory_position[finite_position] = np.asarray(
                position_df["linear_position"],
                dtype=float,
            )

        index_chunks.append(indices)
        trajectory_chunks.append(np.repeat(trajectory_type, indices.size))
        lap_chunks.append(laps)
        position_chunks.append(trajectory_position)

    if not index_chunks:
        raise ValueError("No decoder time bins fall inside trajectory intervals.")

    time_indices = np.concatenate(index_chunks)
    order = np.argsort(time_indices)
    sorted_time_indices = time_indices[order]
    if sorted_time_indices.size > 1 and np.any(np.diff(sorted_time_indices) == 0):
        raise ValueError("Trajectory intervals overlap on the decoder time grid.")

    return {
        "time_index": sorted_time_indices,
        "trajectory_type": np.concatenate(trajectory_chunks)[order],
        "lap_index": np.concatenate(lap_chunks)[order],
        "trajectory_position": np.concatenate(position_chunks)[order],
    }


def compute_head_direction(head_position: np.ndarray, body_position: np.ndarray) -> np.ndarray:
    """Return legacy-compatible head direction from head/body positions."""
    return (
        np.arctan2(
            body_position[:, 1] - head_position[:, 1],
            body_position[:, 0] - head_position[:, 0],
        )
        + np.pi
    )


def compute_map_estimates(
    posterior: Any,
    *,
    time_indices: np.ndarray,
    environment: Any,
    environment_positions: np.ndarray,
    chunk_size: int = MAP_CHUNK_SIZE,
) -> dict[str, np.ndarray]:
    """Return MAP index, position, edge, and probability for selected time bins."""
    posterior = posterior.transpose("time", "position")
    n_time = int(time_indices.size)
    map_indices = np.full(n_time, -1, dtype=int)
    posterior_max = np.full(n_time, np.nan, dtype=float)

    is_track_interior = np.asarray(environment.is_track_interior_, dtype=bool).ravel()
    if is_track_interior.size != environment_positions.size:
        raise ValueError(
            "Classifier environment interior mask does not match position-bin count. "
            f"Interior mask: {is_track_interior.size}; positions: {environment_positions.size}."
        )

    for start in range(0, n_time, chunk_size):
        stop = min(start + chunk_size, n_time)
        chunk_indices = time_indices[start:stop]
        values = np.array(
            posterior.isel(time=chunk_indices).values,
            dtype=float,
            copy=True,
        )
        if values.ndim != 2:
            raise ValueError(
                "Expected summed posterior to be two-dimensional "
                f"(time, position). Got shape {values.shape}."
            )
        values[:, ~is_track_interior] = np.nan
        finite = np.isfinite(values)
        has_finite = np.any(finite, axis=1)
        if not np.any(has_finite):
            continue

        safe_values = np.where(finite, values, -np.inf)
        local_indices = np.argmax(safe_values[has_finite], axis=1)
        chunk_map_indices = np.full(chunk_indices.shape, -1, dtype=int)
        chunk_posterior_max = np.full(chunk_indices.shape, np.nan, dtype=float)
        chunk_map_indices[has_finite] = local_indices
        map_indices[start:stop] = chunk_map_indices
        row_indices = np.flatnonzero(has_finite)
        chunk_posterior_max[has_finite] = values[row_indices, local_indices]
        posterior_max[start:stop] = chunk_posterior_max

    node_table = environment.place_bin_centers_nodes_df_
    mental_position_2d = np.full((n_time, 2), np.nan, dtype=float)
    mental_edge_id = np.full(n_time, -1, dtype=int)
    decoded_position = np.full(n_time, np.nan, dtype=float)
    valid = map_indices >= 0
    if np.any(valid):
        mental_position_2d[valid] = node_table.iloc[map_indices[valid]][
            ["x_position", "y_position"]
        ].to_numpy(dtype=float)
        mental_edge_id[valid] = node_table.iloc[map_indices[valid]][
            "edge_id"
        ].to_numpy(dtype=int)
        decoded_position[valid] = environment_positions[map_indices[valid]]

    return {
        "map_index": map_indices,
        "decoded_position": decoded_position,
        "mental_position_2d": mental_position_2d,
        "mental_edge_id": mental_edge_id,
        "posterior_max": posterior_max,
    }


def compute_actual_graph_position(
    *,
    environment: Any,
    head_position: np.ndarray,
    time_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project actual head position onto one classifier environment graph."""
    import track_linearization as tl

    actual_position = np.asarray(head_position[time_indices], dtype=float)
    projected_position = np.full(actual_position.shape, np.nan, dtype=float)
    edge_id = np.full(time_indices.shape, -1, dtype=int)
    finite_position = np.isfinite(actual_position).all(axis=1)
    if not np.any(finite_position):
        return {
            "projected_position": projected_position,
            "edge_id": edge_id,
        }

    position_df = tl.get_linearized_position(
        position=actual_position[finite_position],
        track_graph=environment.track_graph,
        edge_order=environment.edge_order,
        edge_spacing=environment.edge_spacing,
    )
    projected_position[finite_position] = position_df[
        ["projected_x_position", "projected_y_position"]
    ].to_numpy(dtype=float)
    edge_id[finite_position] = np.asarray(position_df["track_segment_id"], dtype=int)
    return {
        "projected_position": projected_position,
        "edge_id": edge_id,
    }


def build_raw_table(
    *,
    region: str,
    time: np.ndarray,
    trajectory_type: np.ndarray,
    lap_index: np.ndarray,
    cv_fold: np.ndarray,
    true_full_w_position: np.ndarray,
    true_trajectory_position: np.ndarray,
    speed: np.ndarray,
    actual_edge_id: np.ndarray,
    decoded_edge_id: np.ndarray,
    decoded_full_w_position: np.ndarray,
    posterior_max_probability: np.ndarray,
    ahead_behind_distance: np.ndarray,
) -> Any:
    """Build one valid-bin raw ahead/behind table."""
    import pandas as pd

    valid = (
        np.isfinite(time)
        & np.isfinite(true_full_w_position)
        & np.isfinite(true_trajectory_position)
        & np.isfinite(speed)
        & np.isfinite(decoded_full_w_position)
        & np.isfinite(posterior_max_probability)
        & np.isfinite(ahead_behind_distance)
        & (actual_edge_id >= 0)
        & (decoded_edge_id >= 0)
    )
    if not np.any(valid):
        return pd.DataFrame(columns=RAW_COLUMNS)

    table = pd.DataFrame(
        {
            "time": time[valid],
            "region": np.repeat(region, int(np.sum(valid))),
            "trajectory_type": trajectory_type[valid],
            "lap_index": lap_index[valid].astype(int),
            "cv_fold": cv_fold[valid].astype(int),
            "true_full_w_position_cm": true_full_w_position[valid],
            "true_trajectory_position_cm": true_trajectory_position[valid],
            "decoded_full_w_position_cm": decoded_full_w_position[valid],
            "ahead_behind_distance_cm": ahead_behind_distance[valid],
            "abs_ahead_behind_distance_cm": np.abs(ahead_behind_distance[valid]),
            "speed_cm_s": speed[valid],
            "actual_edge_id": actual_edge_id[valid].astype(int),
            "decoded_edge_id": decoded_edge_id[valid].astype(int),
            "posterior_max_probability": posterior_max_probability[valid],
        }
    )
    return table.loc[:, RAW_COLUMNS]


def summarize_region_table(
    raw_table: Any,
    *,
    region: str,
    animal_name: str,
    position_bin_size_cm: float,
    min_bin_count: int,
) -> Any:
    """Summarize signed and absolute ahead/behind distance by trajectory position."""
    import pandas as pd

    total_length = get_wtrack_total_length(animal_name)
    bin_edges = np.arange(0.0, total_length + position_bin_size_cm, position_bin_size_cm)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rows: list[dict[str, Any]] = []

    for trajectory_type in TRAJECTORY_PANEL_ORDER:
        trajectory_table = raw_table.loc[
            raw_table["trajectory_type"] == trajectory_type
        ] if not raw_table.empty else raw_table
        positions = np.asarray(
            trajectory_table.get("true_trajectory_position_cm", []),
            dtype=float,
        )
        signed_error = np.asarray(
            trajectory_table.get("ahead_behind_distance_cm", []),
            dtype=float,
        )
        absolute_error = np.asarray(
            trajectory_table.get("abs_ahead_behind_distance_cm", []),
            dtype=float,
        )
        bin_indices = np.digitize(positions, bin_edges) - 1
        in_range = (bin_indices >= 0) & (bin_indices < bin_centers.size)
        bin_indices = bin_indices[in_range]
        signed_error = signed_error[in_range]
        absolute_error = absolute_error[in_range]

        for bin_index, (bin_left, bin_right, bin_center) in enumerate(
            zip(bin_edges[:-1], bin_edges[1:], bin_centers, strict=True)
        ):
            in_bin = bin_indices == bin_index
            count = int(np.sum(in_bin))
            if count < min_bin_count:
                rows.append(
                    {
                        "region": region,
                        "trajectory_type": trajectory_type,
                        "bin_left": float(bin_left),
                        "bin_right": float(bin_right),
                        "bin_center": float(bin_center),
                        "n": count,
                        "ahead_behind_median": np.nan,
                        "ahead_behind_q25": np.nan,
                        "ahead_behind_q75": np.nan,
                        "abs_ahead_behind_median": np.nan,
                        "abs_ahead_behind_q25": np.nan,
                        "abs_ahead_behind_q75": np.nan,
                    }
                )
                continue

            signed_bin = signed_error[in_bin]
            absolute_bin = absolute_error[in_bin]
            rows.append(
                {
                    "region": region,
                    "trajectory_type": trajectory_type,
                    "bin_left": float(bin_left),
                    "bin_right": float(bin_right),
                    "bin_center": float(bin_center),
                    "n": count,
                    "ahead_behind_median": float(np.median(signed_bin)),
                    "ahead_behind_q25": float(np.quantile(signed_bin, 0.25)),
                    "ahead_behind_q75": float(np.quantile(signed_bin, 0.75)),
                    "abs_ahead_behind_median": float(np.median(absolute_bin)),
                    "abs_ahead_behind_q25": float(np.quantile(absolute_bin, 0.25)),
                    "abs_ahead_behind_q75": float(np.quantile(absolute_bin, 0.75)),
                }
            )

    return pd.DataFrame(rows).loc[:, SUMMARY_COLUMNS]


def plot_region_summary(
    summary_table: Any,
    *,
    region: str,
    animal_name: str,
    date: str,
    epoch: str,
    unit_selection_label: str,
    ylim_cm: float | None,
    output_path: Path,
) -> Path:
    """Save one 2x2 trajectory summary figure for a region."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_by_trajectory = {
        "center_to_left": axes[0, 0],
        "center_to_right": axes[0, 1],
        "left_to_center": axes[1, 0],
        "right_to_center": axes[1, 1],
    }

    for trajectory_type, axis in axes_by_trajectory.items():
        trajectory_table = summary_table.loc[
            summary_table["trajectory_type"] == trajectory_type
        ]
        valid = (
            np.isfinite(trajectory_table["ahead_behind_median"].to_numpy(dtype=float))
            & np.isfinite(trajectory_table["ahead_behind_q25"].to_numpy(dtype=float))
            & np.isfinite(trajectory_table["ahead_behind_q75"].to_numpy(dtype=float))
        )
        x = trajectory_table["bin_center"].to_numpy(dtype=float)
        median = trajectory_table["ahead_behind_median"].to_numpy(dtype=float)
        q25 = trajectory_table["ahead_behind_q25"].to_numpy(dtype=float)
        q75 = trajectory_table["ahead_behind_q75"].to_numpy(dtype=float)
        axis.axhline(0.0, color="0.3", linewidth=0.8, linestyle="--")
        axis.plot(x[valid], median[valid], color="tab:blue", linewidth=1.8)
        axis.fill_between(x[valid], q25[valid], q75[valid], color="tab:blue", alpha=0.25)
        axis.set_title(trajectory_type)
        axis.set_xlabel("Trajectory position (cm)")
        axis.grid(alpha=0.2)
        if ylim_cm is not None:
            axis.set_ylim(-float(ylim_cm), float(ylim_cm))

    axes[0, 0].set_ylabel("Ahead/behind distance (cm)")
    axes[1, 0].set_ylabel("Ahead/behind distance (cm)")
    figure.suptitle(
        f"{animal_name} {date} {epoch} {region.upper()} 1D ahead/behind "
        f"unit_selection {unit_selection_label}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def count_bins_by_trajectory(trajectory_type: np.ndarray) -> dict[str, int]:
    """Return candidate-bin counts by trajectory type."""
    return {
        trajectory: int(np.sum(trajectory_type == trajectory))
        for trajectory in TRAJECTORY_PANEL_ORDER
    }


def compute_dropped_counts(
    *,
    candidate_counts: dict[str, int],
    raw_table: Any,
) -> dict[str, dict[str, int]]:
    """Return kept and dropped valid-bin counts by trajectory type."""
    dropped_counts: dict[str, dict[str, int]] = {}
    for trajectory_type in TRAJECTORY_PANEL_ORDER:
        candidate = int(candidate_counts.get(trajectory_type, 0))
        if raw_table.empty:
            kept = 0
        else:
            kept = int(np.sum(raw_table["trajectory_type"] == trajectory_type))
        dropped_counts[trajectory_type] = {
            "candidate": candidate,
            "kept": kept,
            "dropped": candidate - kept,
        }
    return dropped_counts


def process_region(
    *,
    region: str,
    prediction_path: Path,
    classifier_path: Path,
    output_paths: dict[str, Path],
    animal_name: str,
    date: str,
    epoch: str,
    unit_selection_label: str,
    time_grid: np.ndarray,
    head_position: np.ndarray,
    head_direction: np.ndarray,
    trajectory_samples: dict[str, np.ndarray],
    position_bin_size_cm: float,
    min_bin_count: int,
    ylim_cm: float | None,
) -> dict[str, Any]:
    """Compute and save 1D ahead/behind outputs for one region."""
    import xarray as xr

    classifier = load_classifier(classifier_path)
    environment = get_classifier_environment(classifier)
    time_indices = trajectory_samples["time_index"]

    with xr.open_dataset(prediction_path) as dataset:
        validate_prediction_dataset(
            dataset,
            prediction_path=prediction_path,
            time_grid=time_grid,
        )
        environment_positions = validate_environment_matches_prediction(
            environment,
            dataset,
            classifier_path=classifier_path,
            prediction_path=prediction_path,
        )

        posterior = dataset[POSTERIOR_NAME]
        if "state" in posterior.dims:
            posterior = posterior.sum("state")

        map_data = compute_map_estimates(
            posterior,
            time_indices=time_indices,
            environment=environment,
            environment_positions=environment_positions,
        )
        actual_graph_data = compute_actual_graph_position(
            environment=environment,
            head_position=head_position,
            time_indices=time_indices,
        )
        ahead_behind_distance = compute_ahead_behind_distance(
            track_graph=environment.track_graph,
            actual_projected_position=actual_graph_data["projected_position"],
            actual_edge_id=actual_graph_data["edge_id"],
            head_direction=head_direction[time_indices],
            mental_position_2d=map_data["mental_position_2d"],
            mental_edge_id=map_data["mental_edge_id"],
        )

        raw_table = build_raw_table(
            region=region,
            time=time_grid[time_indices],
            trajectory_type=trajectory_samples["trajectory_type"],
            lap_index=trajectory_samples["lap_index"],
            cv_fold=np.asarray(dataset["cv_fold"].isel(time=time_indices).values),
            true_full_w_position=np.asarray(
                dataset["true_linear_position"].isel(time=time_indices).values,
                dtype=float,
            ),
            true_trajectory_position=trajectory_samples["trajectory_position"],
            speed=np.asarray(dataset["speed_cm_s"].isel(time=time_indices).values, dtype=float),
            actual_edge_id=actual_graph_data["edge_id"],
            decoded_edge_id=map_data["mental_edge_id"],
            decoded_full_w_position=map_data["decoded_position"],
            posterior_max_probability=map_data["posterior_max"],
            ahead_behind_distance=ahead_behind_distance,
        )

    summary_table = summarize_region_table(
        raw_table,
        region=region,
        animal_name=animal_name,
        position_bin_size_cm=position_bin_size_cm,
        min_bin_count=min_bin_count,
    )

    output_paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    raw_table.to_parquet(output_paths["raw"])
    summary_table.to_parquet(output_paths["summary"])
    figure_path = plot_region_summary(
        summary_table,
        region=region,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        unit_selection_label=unit_selection_label,
        ylim_cm=ylim_cm,
        output_path=output_paths["figure"],
    )

    candidate_counts = count_bins_by_trajectory(trajectory_samples["trajectory_type"])
    dropped_counts = compute_dropped_counts(
        candidate_counts=candidate_counts,
        raw_table=raw_table,
    )
    return {
        "raw_path": output_paths["raw"],
        "summary_path": output_paths["summary"],
        "figure_path": figure_path,
        "n_rows": int(len(raw_table)),
        "dropped_counts": dropped_counts,
    }


def run(args: argparse.Namespace) -> None:
    """Run the 1D ahead/behind decoding-error workflow."""
    validate_arguments(args)

    analysis_path = get_analysis_path_for_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    selected_regions = select_regions(args.region)
    position_bin_size_cm = (
        float(args.place_bin_size)
        if args.position_bin_size_cm is None
        else float(args.position_bin_size_cm)
    )
    v1_unit_selection_requested = (
        args.v1_ripple_glm_units
        or args.v1_ripple_glm_devexp_threshold is not None
    )
    unit_selection_label = get_unit_selection_label(
        args.v1_ripple_glm_units and "v1" in selected_regions,
        (
            args.v1_ripple_glm_devexp_threshold
            if "v1" in selected_regions
            else None
        ),
    )
    if v1_unit_selection_requested and "v1" not in selected_regions:
        print(
            "A V1 ripple GLM unit-selection option was passed, but V1 is not selected; "
            "CA1 outputs are unchanged."
        )

    classifier_paths_by_fold = build_classifier_output_paths(
        get_fit_output_dir(analysis_path),
        regions=selected_regions,
        epoch=args.epoch,
        n_folds=args.n_folds,
        random_state=args.random_state,
        direction=args.direction,
        movement=args.movement,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        position_std=args.position_std,
        discrete_var=args.discrete_var,
        place_bin_size=args.place_bin_size,
        movement_var=args.movement_var,
        branch_gap_cm=args.branch_gap_cm,
        unit_selection_label=unit_selection_label,
    )
    classifier_paths = {
        region: classifier_paths_by_fold[(region, 0)]
        for region in selected_regions
    }
    prediction_paths = build_prediction_output_paths(
        get_predict_output_dir(analysis_path),
        regions=selected_regions,
        epoch=args.epoch,
        n_folds=args.n_folds,
        random_state=args.random_state,
        direction=args.direction,
        movement=args.movement,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        position_std=args.position_std,
        discrete_var=args.discrete_var,
        place_bin_size=args.place_bin_size,
        movement_var=args.movement_var,
        branch_gap_cm=args.branch_gap_cm,
        unit_selection_label=unit_selection_label,
    )
    output_paths = build_region_output_paths(
        prediction_paths=prediction_paths,
        table_output_dir=get_error_output_dir(analysis_path),
        figure_output_dir=get_error_figure_dir(analysis_path),
        epoch=args.epoch,
        n_folds=args.n_folds,
        random_state=args.random_state,
        time_bin_size_s=args.time_bin_size_s,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        position_std=args.position_std,
        place_bin_size=args.place_bin_size,
        movement_var=args.movement_var,
        discrete_var=args.discrete_var,
        branch_gap_cm=args.branch_gap_cm,
        direction=args.direction,
        movement=args.movement,
        unit_selection_label=unit_selection_label,
        position_bin_size_cm=position_bin_size_cm,
        min_bin_count=args.min_bin_count,
    )

    preflight_no_existing(
        flatten_output_paths(output_paths),
        overwrite=args.overwrite,
        output_kind="decoding-error",
    )
    require_existing_paths(
        list(prediction_paths.values()),
        input_kind="prediction",
    )
    require_existing_paths(
        list(classifier_paths.values()),
        input_kind="classifier",
    )

    print(
        f"Computing 1D ahead/behind error for {args.animal_name} {args.date} "
        f"epoch {args.epoch}; regions={list(selected_regions)}, "
        f"unit_selection={unit_selection_label}, position_bin_size_cm={position_bin_size_cm}."
    )

    session = load_required_session_inputs(
        analysis_path=analysis_path,
        animal_name=args.animal_name,
        epoch=args.epoch,
    )
    _parquet_epoch_order, _head_position_by_epoch, body_position_by_epoch = (
        load_clean_dlc_position_data(analysis_path, validate_timestamps=True)
    )
    if args.epoch not in body_position_by_epoch:
        raise ValueError(f"Cleaned DLC position parquet does not contain epoch {args.epoch!r}.")

    time_grid = build_time_grid(
        session["timestamps_position"],
        position_offset=args.position_offset,
        time_bin_size_s=args.time_bin_size_s,
    )
    head_position = interpolate_position_to_time(
        session["position"],
        session["timestamps_position"],
        time_grid,
        position_offset=args.position_offset,
    )
    body_position = interpolate_position_to_time(
        body_position_by_epoch[args.epoch],
        session["timestamps_position"],
        time_grid,
        position_offset=args.position_offset,
    )
    head_direction = compute_head_direction(head_position, body_position)
    trajectory_samples = build_trajectory_samples(
        animal_name=args.animal_name,
        time_grid=time_grid,
        head_position=head_position,
        trajectory_intervals=session["trajectory_intervals"],
    )

    region_results: dict[str, Any] = {}
    for region in selected_regions:
        region_result = process_region(
            region=region,
            prediction_path=prediction_paths[region],
            classifier_path=classifier_paths[region],
            output_paths=output_paths[region],
            animal_name=args.animal_name,
            date=args.date,
            epoch=args.epoch,
            unit_selection_label=unit_selection_label,
            time_grid=time_grid,
            head_position=head_position,
            head_direction=head_direction,
            trajectory_samples=trajectory_samples,
            position_bin_size_cm=position_bin_size_cm,
            min_bin_count=args.min_bin_count,
            ylim_cm=args.ylim_cm,
        )
        region_results[region] = region_result
        print(
            f"Saved {region} ahead/behind outputs: "
            f"{region_result['raw_path']}, "
            f"{region_result['summary_path']}, "
            f"{region_result['figure_path']}"
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name=SCRIPT_NAME,
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "epoch": args.epoch,
            "regions": list(selected_regions),
            "data_root": args.data_root,
            "n_folds": args.n_folds,
            "random_state": args.random_state,
            "time_bin_size_s": args.time_bin_size_s,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "position_std": args.position_std,
            "place_bin_size": args.place_bin_size,
            "movement_var": args.movement_var,
            "discrete_var": args.discrete_var,
            "branch_gap_cm": args.branch_gap_cm,
            "direction": args.direction,
            "movement": args.movement,
            "v1_ripple_glm_units": args.v1_ripple_glm_units,
            "v1_ripple_glm_devexp_threshold": args.v1_ripple_glm_devexp_threshold,
            "unit_selection_label": unit_selection_label,
            "position_bin_size_cm": position_bin_size_cm,
            "min_bin_count": args.min_bin_count,
            "ylim_cm": args.ylim_cm,
            "overwrite": args.overwrite,
        },
        outputs={
            "sources": {
                **session["sources"],
                "prediction_paths": prediction_paths,
                "classifier_paths": classifier_paths,
            },
            "trajectory_sample_count": int(trajectory_samples["time_index"].size),
            "output_paths": output_paths,
            "region_results": region_results,
        },
    )
    print(f"Saved run metadata to {log_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for 1D decoding-error computation."""
    parser = argparse.ArgumentParser(
        description="Compute trajectory-aligned 1D ahead/behind decoding error."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name, e.g. L14.")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format.")
    parser.add_argument("--epoch", required=True, help="Run epoch name, e.g. 02_r1.")
    parser.add_argument(
        "--region",
        choices=REGIONS,
        help="Optional single region to process. Default: process all regions.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Number of shuffled lap-wise cross-validation folds. Default: {DEFAULT_N_FOLDS}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed used for shuffled lap-wise folds. Default: {DEFAULT_RANDOM_STATE}",
    )
    parser.add_argument(
        "--time-bin-size-s",
        type=float,
        default=DEFAULT_TIME_BIN_SIZE_S,
        help=f"Prediction time bin size in seconds. Default: {DEFAULT_TIME_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Leading position samples dropped by predict_1d. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Prediction path setting matching fit_1d/predict_1d. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--position-std",
        type=float,
        default=DEFAULT_POSITION_STD,
        help=f"Prediction path setting matching fit_1d. Default: {DEFAULT_POSITION_STD}",
    )
    parser.add_argument(
        "--place-bin-size",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=f"Prediction path setting matching fit_1d. Default: {DEFAULT_PLACE_BIN_SIZE_CM}",
    )
    parser.add_argument(
        "--movement-var",
        type=float,
        default=DEFAULT_MOVEMENT_VAR,
        help=f"Prediction path setting matching fit_1d. Default: {DEFAULT_MOVEMENT_VAR}",
    )
    parser.add_argument(
        "--discrete-var",
        choices=DISCRETE_VAR_CHOICES,
        default="switching",
        help="Prediction path setting matching fit_1d. Default: switching.",
    )
    parser.add_argument(
        "--branch-gap-cm",
        type=float,
        default=DEFAULT_BRANCH_GAP_CM,
        help=(
            "Prediction path setting matching fit_1d/predict_1d. "
            f"Default: {DEFAULT_BRANCH_GAP_CM}"
        ),
    )
    parser.add_argument(
        "--direction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use outputs fit with inbound/outbound groups. Default: enabled.",
    )
    parser.add_argument(
        "--movement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use outputs fit with movement-restricted training. Default: enabled.",
    )
    unit_selection_group = parser.add_mutually_exclusive_group()
    unit_selection_group.add_argument(
        "--v1-ripple-glm-units",
        action="store_true",
        help=(
            "Use V1 ripple GLM p-value-selected outputs matching predict_1d. "
            f"Threshold: {DEFAULT_V1_RIPPLE_GLM_P_VALUE_THRESHOLD}."
        ),
    )
    unit_selection_group.add_argument(
        "--v1-ripple-glm-devexp-threshold",
        type=float,
        help=(
            "Use V1 outputs restricted to units with ripple_devexp_mean greater "
            "than or equal to this value."
        ),
    )
    parser.add_argument(
        "--position-bin-size-cm",
        type=float,
        help=(
            "Spatial bin size for trajectory-position summaries. "
            "Default: --place-bin-size."
        ),
    )
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=DEFAULT_MIN_BIN_COUNT,
        help=f"Minimum samples required to summarize a spatial bin. Default: {DEFAULT_MIN_BIN_COUNT}",
    )
    parser.add_argument(
        "--ylim-cm",
        type=float,
        help="Optional symmetric y-axis limit for figures, e.g. --ylim-cm 80.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing decoding-error outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run from the command line."""
    run(parse_arguments())


if __name__ == "__main__":
    main()
