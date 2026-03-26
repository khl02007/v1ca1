"""Fit one draft W-track geometry from session position data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    POSITION_SOURCE_CHOICES,
    get_analysis_path,
    get_run_epochs,
    load_epoch_tags,
    load_position_data_with_precedence,
)
from v1ca1.helper.wtrack import get_wtrack_geometry


DEFAULT_OUTPUT_DIRNAME = "wtrack_geometry_fit"
DEFAULT_DRAFT_NAME = "wtrack_geometry_draft.json"
DEFAULT_QC_NAME = "wtrack_fit_qc.png"
DEFAULT_SNIPPET_NAME = "wtrack_geometry_snippet.py.txt"
DEFAULT_OCCUPANCY_BIN_SIZE_CM = 1.0
DEFAULT_SKELETON_STEP_CM = 1.0
DEFAULT_MATCH_THRESHOLD_CM = 4.0
DEFAULT_MIN_OCCUPANCY_BIN_COUNT = 25
_WTRACK_EDGE_LIST = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def _as_xy_array(position_xy: np.ndarray) -> np.ndarray:
    """Return one validated finite XY array."""
    position_array = np.asarray(position_xy, dtype=float)
    if position_array.ndim != 2 or position_array.shape[1] != 2:
        raise ValueError(
            f"Expected one position array of shape (n_samples, 2), got {position_array.shape}."
        )
    finite_mask = np.isfinite(position_array).all(axis=1)
    filtered = position_array[finite_mask]
    if filtered.size == 0:
        raise ValueError("Position array does not contain any finite XY samples.")
    return filtered


def _wrap_angle(theta: float) -> float:
    """Wrap one angle to [-pi, pi)."""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def build_wtrack_node_positions(
    center_well_xy: np.ndarray | list[float] | tuple[float, float],
    theta: float,
    long_segment_length: float,
    short_segment_length: float,
    dx: float,
    dy: float,
) -> dict[str, np.ndarray]:
    """Build mirrored left/right node positions from one rotated W-template."""
    center = np.asarray(center_well_xy, dtype=float).reshape(2)
    if not np.all(np.isfinite(center)):
        raise ValueError("center_well_xy must be finite.")

    long_length = float(long_segment_length)
    short_length = float(short_segment_length)
    diagonal_dx = float(dx)
    diagonal_dy = float(dy)
    if min(long_length, short_length, diagonal_dx, diagonal_dy) <= 0:
        raise ValueError("All W-track segment parameters must be positive.")

    forward = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    lateral = np.array([-forward[1], forward[0]], dtype=float)

    center_junction = center + long_length * forward

    def _build_branch(sign: float) -> np.ndarray:
        branch_lateral = sign * lateral
        diagonal_node = center_junction + diagonal_dy * forward + diagonal_dx * branch_lateral
        branch_floor = diagonal_node + short_length * branch_lateral
        outer_junction = branch_floor - diagonal_dy * forward + diagonal_dx * branch_lateral
        outer_well = outer_junction - long_length * forward
        return np.vstack(
            [
                center,
                center_junction,
                diagonal_node,
                branch_floor,
                outer_junction,
                outer_well,
            ]
        )

    return {
        "left": _build_branch(+1.0),
        "right": _build_branch(-1.0),
    }


def _get_wtrack_segments(node_positions: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Return all unique W-track segments as endpoint pairs."""
    center = np.asarray(node_positions["left"][0], dtype=float)
    center_junction = np.asarray(node_positions["left"][1], dtype=float)
    left = np.asarray(node_positions["left"], dtype=float)
    right = np.asarray(node_positions["right"], dtype=float)
    return [
        np.vstack([center, center_junction]),
        left[[1, 2]],
        left[[2, 3]],
        left[[3, 4]],
        left[[4, 5]],
        right[[1, 2]],
        right[[2, 3]],
        right[[3, 4]],
        right[[4, 5]],
    ]


def _sample_segment(segment: np.ndarray, step_cm: float) -> np.ndarray:
    """Sample approximately uniform points along one segment."""
    start = np.asarray(segment[0], dtype=float)
    end = np.asarray(segment[1], dtype=float)
    length = float(np.linalg.norm(end - start))
    if length == 0:
        return start.reshape(1, 2)
    n_samples = max(int(np.ceil(length / step_cm)), 1)
    weights = np.linspace(0.0, 1.0, n_samples + 1, dtype=float)
    return start + weights[:, None] * (end - start)


def sample_wtrack_skeleton(
    node_positions: dict[str, np.ndarray],
    step_cm: float = DEFAULT_SKELETON_STEP_CM,
) -> np.ndarray:
    """Sample one point cloud along the full W-track skeleton."""
    if step_cm <= 0:
        raise ValueError("step_cm must be positive.")
    sampled = np.vstack([_sample_segment(segment, step_cm) for segment in _get_wtrack_segments(node_positions)])
    rounded = np.round(sampled, decimals=6)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return sampled[np.sort(unique_indices)]


def make_wtrack_geometry_dict(
    center_well_xy: np.ndarray | list[float] | tuple[float, float],
    theta: float,
    long_segment_length: float,
    short_segment_length: float,
    dx: float,
    dy: float,
) -> dict[str, Any]:
    """Return one geometry dict compatible with the shared W-track helper layout."""
    node_positions = build_wtrack_node_positions(
        center_well_xy=center_well_xy,
        theta=theta,
        long_segment_length=long_segment_length,
        short_segment_length=short_segment_length,
        dx=dx,
        dy=dy,
    )
    return {
        "dx": float(dx),
        "dy": float(dy),
        "long_segment_length": float(long_segment_length),
        "short_segment_length": float(short_segment_length),
        "node_positions_right": node_positions["right"].tolist(),
        "node_positions_left": node_positions["left"].tolist(),
        "edges_from_center": [list(edge) for edge in _WTRACK_EDGE_LIST],
        "edges_to_center": [list(edge[::-1]) for edge in _WTRACK_EDGE_LIST[::-1]],
        "edge_order_from_center": [list(edge) for edge in _WTRACK_EDGE_LIST],
        "edge_order_to_center": [list(edge[::-1]) for edge in _WTRACK_EDGE_LIST[::-1]],
    }


def compute_total_wtrack_length(geometry: dict[str, Any]) -> float:
    """Return the total path length implied by one geometry dict."""
    diagonal_segment_length = float(np.hypot(float(geometry["dx"]), float(geometry["dy"])))
    return float(
        float(geometry["long_segment_length"]) * 2.0
        + float(geometry["short_segment_length"])
        + 2.0 * diagonal_segment_length
    )


def compute_occupied_bin_centers(
    position_xy: np.ndarray,
    bin_size_cm: float = DEFAULT_OCCUPANCY_BIN_SIZE_CM,
) -> np.ndarray:
    """Collapse one position cloud into equal-weight occupied spatial bins."""
    if bin_size_cm <= 0:
        raise ValueError("bin_size_cm must be positive.")
    position_array = _as_xy_array(position_xy)
    origin = np.min(position_array, axis=0)
    bin_indices = np.floor((position_array - origin) / float(bin_size_cm)).astype(int)
    unique_indices = np.unique(bin_indices, axis=0)
    return origin + (unique_indices.astype(float) + 0.5) * float(bin_size_cm)


def _compute_multistart_thetas(position_xy: np.ndarray) -> list[float]:
    """Return PCA-derived candidate stem orientations."""
    centered = _as_xy_array(position_xy) - np.mean(position_xy, axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_vector = eigenvectors[:, int(np.argmax(eigenvalues))]
    major_angle = math.atan2(float(major_vector[1]), float(major_vector[0]))
    return [_wrap_angle(major_angle), _wrap_angle(major_angle + np.pi / 2.0)]


def _get_reference_fit_parameters() -> dict[str, float]:
    """Return the scalar parameters stored for the L14 reference geometry."""
    geometry = get_wtrack_geometry("L14")
    return {
        "long_segment_length": float(geometry["long_segment_length"]),
        "short_segment_length": float(geometry["short_segment_length"]),
        "dx": float(geometry["dx"]),
        "dy": float(geometry["dy"]),
    }


def _estimate_initial_scale(position_xy: np.ndarray, reference_params: dict[str, float]) -> float:
    """Estimate one global size scale from occupancy span."""
    reference_geometry = build_wtrack_node_positions(
        center_well_xy=np.zeros(2, dtype=float),
        theta=-np.pi / 2.0,
        long_segment_length=reference_params["long_segment_length"],
        short_segment_length=reference_params["short_segment_length"],
        dx=reference_params["dx"],
        dy=reference_params["dy"],
    )
    reference_points = sample_wtrack_skeleton(reference_geometry)
    reference_span = float(np.linalg.norm(np.ptp(reference_points, axis=0)))
    occupancy_span = float(np.linalg.norm(np.ptp(_as_xy_array(position_xy), axis=0)))
    if reference_span <= 0 or occupancy_span <= 0:
        return 1.0
    return max(occupancy_span / reference_span, 0.25)


def _make_initial_parameter_vector(
    occupied_centers: np.ndarray,
    reference_params: dict[str, float],
    scale: float,
    theta: float,
) -> np.ndarray:
    """Return one multistart parameter vector aligned to the occupancy centroid."""
    scaled_params = {
        key: max(value * scale, 1.0)
        for key, value in reference_params.items()
    }
    centered_geometry = build_wtrack_node_positions(
        center_well_xy=np.zeros(2, dtype=float),
        theta=theta,
        long_segment_length=scaled_params["long_segment_length"],
        short_segment_length=scaled_params["short_segment_length"],
        dx=scaled_params["dx"],
        dy=scaled_params["dy"],
    )
    template_centroid = np.mean(sample_wtrack_skeleton(centered_geometry), axis=0)
    occupancy_centroid = np.mean(occupied_centers, axis=0)
    center = occupancy_centroid - template_centroid
    return np.array(
        [
            center[0],
            center[1],
            theta,
            scaled_params["long_segment_length"],
            scaled_params["short_segment_length"],
            scaled_params["dx"],
            scaled_params["dy"],
        ],
        dtype=float,
    )


def _build_parameter_bounds(
    occupied_centers: np.ndarray,
    initial_scale: float,
    reference_params: dict[str, float],
) -> list[tuple[float, float]]:
    """Return optimizer bounds for one session fit."""
    mins = np.min(occupied_centers, axis=0)
    maxs = np.max(occupied_centers, axis=0)
    span = np.maximum(maxs - mins, 10.0)
    center_bounds = [
        (float(mins[0] - span[0]), float(maxs[0] + span[0])),
        (float(mins[1] - span[1]), float(maxs[1] + span[1])),
    ]

    def _scale_bounds(value: float) -> tuple[float, float]:
        scaled = max(value * initial_scale, 1.0)
        return (max(0.5 * scaled, 1.0), max(2.5 * scaled, 2.0))

    return center_bounds + [
        (-np.pi, np.pi),
        _scale_bounds(reference_params["long_segment_length"]),
        _scale_bounds(reference_params["short_segment_length"]),
        _scale_bounds(reference_params["dx"]),
        _scale_bounds(reference_params["dy"]),
    ]


def _objective_from_parameter_vector(
    params: np.ndarray,
    occupied_centers: np.ndarray,
    skeleton_step_cm: float,
) -> float:
    """Return the symmetric Chamfer-style loss for one parameter vector."""
    center_x, center_y, theta, long_length, short_length, dx, dy = params
    if min(long_length, short_length, dx, dy) <= 0:
        return float("inf")

    node_positions = build_wtrack_node_positions(
        center_well_xy=np.array([center_x, center_y], dtype=float),
        theta=float(theta),
        long_segment_length=float(long_length),
        short_segment_length=float(short_length),
        dx=float(dx),
        dy=float(dy),
    )
    skeleton_points = sample_wtrack_skeleton(node_positions, step_cm=skeleton_step_cm)

    occupancy_tree = cKDTree(occupied_centers)
    skeleton_tree = cKDTree(skeleton_points)
    occupancy_to_skeleton = skeleton_tree.query(occupied_centers, k=1)[0]
    skeleton_to_occupancy = occupancy_tree.query(skeleton_points, k=1)[0]
    return float(
        np.mean(occupancy_to_skeleton**2) + np.mean(skeleton_to_occupancy**2)
    )


def _clip_to_bounds(
    params: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Project one initial guess into closed optimizer bounds."""
    clipped = np.asarray(params, dtype=float).copy()
    for index, (lower, upper) in enumerate(bounds):
        clipped[index] = np.clip(clipped[index], lower, upper)
    return clipped


def assess_wtrack_fit(
    occupied_centers: np.ndarray,
    geometry: dict[str, Any],
    *,
    skeleton_step_cm: float = DEFAULT_SKELETON_STEP_CM,
    match_threshold_cm: float = DEFAULT_MATCH_THRESHOLD_CM,
) -> dict[str, float | int | bool]:
    """Return summary metrics describing one fitted geometry."""
    node_positions = {
        "left": np.asarray(geometry["node_positions_left"], dtype=float),
        "right": np.asarray(geometry["node_positions_right"], dtype=float),
    }
    skeleton_points = sample_wtrack_skeleton(node_positions, step_cm=skeleton_step_cm)
    occupancy_tree = cKDTree(occupied_centers)
    skeleton_tree = cKDTree(skeleton_points)
    occupancy_to_skeleton = skeleton_tree.query(occupied_centers, k=1)[0]
    skeleton_to_occupancy = occupancy_tree.query(skeleton_points, k=1)[0]

    return {
        "occupied_bin_count": int(occupied_centers.shape[0]),
        "skeleton_point_count": int(skeleton_points.shape[0]),
        "occupancy_to_skeleton_mean_cm": float(np.mean(occupancy_to_skeleton)),
        "occupancy_to_skeleton_p90_cm": float(np.percentile(occupancy_to_skeleton, 90)),
        "skeleton_to_occupancy_mean_cm": float(np.mean(skeleton_to_occupancy)),
        "skeleton_to_occupancy_p90_cm": float(np.percentile(skeleton_to_occupancy, 90)),
        "occupancy_within_threshold_fraction": float(
            np.mean(occupancy_to_skeleton <= match_threshold_cm)
        ),
        "skeleton_within_threshold_fraction": float(
            np.mean(skeleton_to_occupancy <= match_threshold_cm)
        ),
        "match_threshold_cm": float(match_threshold_cm),
        "track_total_length_cm": compute_total_wtrack_length(geometry),
    }


def _is_fit_acceptable(metrics: dict[str, float | int | bool]) -> bool:
    """Return whether one fit passes conservative draft-QC thresholds."""
    return bool(
        int(metrics["occupied_bin_count"]) >= DEFAULT_MIN_OCCUPANCY_BIN_COUNT
        and float(metrics["occupancy_within_threshold_fraction"]) >= 0.70
        and float(metrics["skeleton_within_threshold_fraction"]) >= 0.60
        and float(metrics["occupancy_to_skeleton_p90_cm"]) <= 8.0
        and float(metrics["skeleton_to_occupancy_p90_cm"]) <= 8.0
    )


def fit_wtrack_geometry_from_positions(
    position_xy: np.ndarray,
    *,
    occupancy_bin_size_cm: float = DEFAULT_OCCUPANCY_BIN_SIZE_CM,
    skeleton_step_cm: float = DEFAULT_SKELETON_STEP_CM,
    match_threshold_cm: float = DEFAULT_MATCH_THRESHOLD_CM,
) -> dict[str, Any]:
    """Fit one draft W-track geometry directly from XY samples."""
    position_array = _as_xy_array(position_xy)
    occupied_centers = compute_occupied_bin_centers(
        position_array,
        bin_size_cm=occupancy_bin_size_cm,
    )
    reference_params = _get_reference_fit_parameters()
    initial_scale = _estimate_initial_scale(occupied_centers, reference_params)
    bounds = _build_parameter_bounds(occupied_centers, initial_scale, reference_params)

    best_result: dict[str, Any] | None = None
    for theta in _compute_multistart_thetas(occupied_centers):
        x0 = _make_initial_parameter_vector(
            occupied_centers=occupied_centers,
            reference_params=reference_params,
            scale=initial_scale,
            theta=theta,
        )
        x0 = _clip_to_bounds(x0, bounds)
        result = minimize(
            _objective_from_parameter_vector,
            x0=x0,
            args=(occupied_centers, skeleton_step_cm),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 250, "disp": False},
        )
        if not result.success and not np.isfinite(result.fun):
            continue
        if best_result is None or float(result.fun) < float(best_result["objective"]):
            best_result = {
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "params": np.asarray(result.x, dtype=float),
                "theta_seed_rad": float(theta),
            }

    if best_result is None:
        raise RuntimeError("Could not fit a W-track geometry from the provided position samples.")

    params = best_result["params"]
    geometry = make_wtrack_geometry_dict(
        center_well_xy=params[:2],
        theta=float(params[2]),
        long_segment_length=float(params[3]),
        short_segment_length=float(params[4]),
        dx=float(params[5]),
        dy=float(params[6]),
    )
    metrics = assess_wtrack_fit(
        occupied_centers,
        geometry,
        skeleton_step_cm=skeleton_step_cm,
        match_threshold_cm=match_threshold_cm,
    )
    fit_ok = _is_fit_acceptable(metrics)

    return {
        "geometry": geometry,
        "occupied_bin_centers": occupied_centers,
        "fit_ok": fit_ok,
        "fit_metrics": metrics,
        "optimizer": {
            "objective": float(best_result["objective"]),
            "success": bool(best_result["success"]),
            "message": str(best_result["message"]),
            "theta_seed_rad": float(best_result["theta_seed_rad"]),
            "theta_rad": float(params[2]),
            "center_well_xy": [float(params[0]), float(params[1])],
        },
    }


def _select_position_epochs(
    position_by_epoch: dict[str, np.ndarray],
    requested_epochs: list[str] | None,
    run_epochs: list[str],
) -> list[str]:
    """Return the ordered run epochs used for fitting."""
    if requested_epochs is not None and len(requested_epochs) > 0:
        missing = [epoch for epoch in requested_epochs if epoch not in position_by_epoch]
        if missing:
            raise ValueError(f"Requested epochs are missing position data: {missing!r}")
        return [str(epoch) for epoch in requested_epochs]

    selected = [epoch for epoch in run_epochs if epoch in position_by_epoch]
    if not selected:
        raise ValueError("No run epochs with position data were found for W-track fitting.")
    return selected


def _concatenate_position_epochs(
    position_by_epoch: dict[str, np.ndarray],
    selected_epochs: list[str],
) -> np.ndarray:
    """Concatenate selected epochs into one XY array."""
    concatenated = np.vstack([_as_xy_array(position_by_epoch[epoch]) for epoch in selected_epochs])
    if concatenated.size == 0:
        raise ValueError("Selected epochs did not contain any finite position samples.")
    return concatenated


def _serialize_result(
    *,
    animal_name: str,
    date: str,
    position_source: str,
    selected_epochs: list[str],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    """Return one JSON-serializable draft record."""
    return {
        "animal_name": animal_name,
        "date": date,
        "position_source": position_source,
        "selected_epochs": selected_epochs,
        "fit_ok": bool(fit_result["fit_ok"]),
        "fit_metrics": {
            key: float(value) if isinstance(value, (np.floating, float)) else int(value)
            for key, value in fit_result["fit_metrics"].items()
        },
        "optimizer": {
            key: (
                [float(item) for item in value]
                if isinstance(value, list)
                else float(value)
                if isinstance(value, (np.floating, float))
                else bool(value)
                if isinstance(value, (np.bool_, bool))
                else value
            )
            for key, value in fit_result["optimizer"].items()
        },
        "geometry": fit_result["geometry"],
    }


def _format_geometry_snippet(animal_name: str, geometry: dict[str, Any]) -> str:
    """Return one paste-ready Python snippet for `wtrack.py`."""
    left = json.dumps(geometry["node_positions_left"], indent=12)
    right = json.dumps(geometry["node_positions_right"], indent=12)
    return (
        f'_WTRACK_GEOMETRY_BY_ANIMAL["{animal_name}"] = {{\n'
        f'    "dx": {geometry["dx"]:.6f},\n'
        f'    "dy": {geometry["dy"]:.6f},\n'
        f'    "long_segment_length": {geometry["long_segment_length"]:.6f},\n'
        f'    "short_segment_length": {geometry["short_segment_length"]:.6f},\n'
        f'    "node_positions_right": np.array({right}, dtype=float),\n'
        f'    "node_positions_left": np.array({left}, dtype=float),\n'
        f'    "edges_from_center": np.array({geometry["edges_from_center"]}, dtype=int),\n'
        f'    "edges_to_center": np.array({geometry["edges_to_center"]}, dtype=int),\n'
        f'    "edge_order_from_center": {geometry["edge_order_from_center"]},\n'
        f'    "edge_order_to_center": {geometry["edge_order_to_center"]},\n'
        "}\n"
    )


def _save_qc_figure(
    output_path: Path,
    occupied_centers: np.ndarray,
    geometry: dict[str, Any],
    metrics: dict[str, float | int | bool],
    *,
    fit_ok: bool,
    animal_name: str,
    date: str,
) -> None:
    """Save one draft overlay figure for visual QC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    node_positions = {
        "left": np.asarray(geometry["node_positions_left"], dtype=float),
        "right": np.asarray(geometry["node_positions_right"], dtype=float),
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        occupied_centers[:, 0],
        occupied_centers[:, 1],
        s=10,
        alpha=0.5,
        color="0.6",
        label="occupied bins",
        zorder=1,
    )
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        points = node_positions[side]
        for i, j in _WTRACK_EDGE_LIST:
            segment = points[[i, j]]
            ax.plot(segment[:, 0], segment[:, 1], color=color, lw=2.0, zorder=2)
        ax.scatter(points[:, 0], points[:, 1], s=35, color=color, zorder=3, label=f"{side} nodes")
        for index, (x_coord, y_coord) in enumerate(points):
            ax.text(x_coord, y_coord, f"{side[0]}{index}", fontsize=8, color=color)

    summary = (
        f"fit_ok={fit_ok}\n"
        f"occ<=thr: {metrics['occupancy_within_threshold_fraction']:.2f}\n"
        f"skel<=thr: {metrics['skeleton_within_threshold_fraction']:.2f}\n"
        f"occ p90: {metrics['occupancy_to_skeleton_p90_cm']:.2f} cm\n"
        f"skel p90: {metrics['skeleton_to_occupancy_p90_cm']:.2f} cm\n"
        f"length: {metrics['track_total_length_cm']:.2f} cm"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"{animal_name} {date} draft W-track fit")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def fit_wtrack_geometry(
    animal_name: str,
    date: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    epochs: list[str] | None = None,
    position_source: str = "auto",
    clean_dlc_input_dirname: str = DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    clean_dlc_input_name: str = DEFAULT_CLEAN_DLC_POSITION_NAME,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
) -> dict[str, Path | bool]:
    """Fit and save one draft W-track geometry for a session."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not output_dirname:
        raise ValueError("--output-dirname must be a non-empty string.")

    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    run_epochs = get_run_epochs(epoch_tags)
    position_by_epoch, resolved_position_source = load_position_data_with_precedence(
        analysis_path,
        position_source=position_source,
        clean_dlc_input_dirname=clean_dlc_input_dirname,
        clean_dlc_input_name=clean_dlc_input_name,
        validate_timestamps=True,
    )
    selected_epochs = _select_position_epochs(position_by_epoch, epochs, run_epochs)
    position_xy = _concatenate_position_epochs(position_by_epoch, selected_epochs)
    fit_result = fit_wtrack_geometry_from_positions(position_xy)

    output_dir = analysis_path / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    draft_record = _serialize_result(
        animal_name=animal_name,
        date=date,
        position_source=resolved_position_source,
        selected_epochs=selected_epochs,
        fit_result=fit_result,
    )
    draft_path = output_dir / DEFAULT_DRAFT_NAME
    with open(draft_path, "w", encoding="utf-8") as file:
        json.dump(draft_record, file, indent=2)
        file.write("\n")

    snippet_path = output_dir / DEFAULT_SNIPPET_NAME
    with open(snippet_path, "w", encoding="utf-8") as file:
        file.write(_format_geometry_snippet(animal_name, fit_result["geometry"]))

    qc_path = output_dir / DEFAULT_QC_NAME
    _save_qc_figure(
        qc_path,
        fit_result["occupied_bin_centers"],
        fit_result["geometry"],
        fit_result["fit_metrics"],
        fit_ok=bool(fit_result["fit_ok"]),
        animal_name=animal_name,
        date=date,
    )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.position.fit_wtrack_geometry",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "epochs": selected_epochs,
            "position_source": position_source,
            "clean_dlc_input_dirname": clean_dlc_input_dirname,
            "clean_dlc_input_name": clean_dlc_input_name,
            "output_dirname": output_dirname,
        },
        outputs={
            "position_source_resolved": resolved_position_source,
            "selected_epochs": selected_epochs,
            "fit_ok": bool(fit_result["fit_ok"]),
            "fit_metrics": draft_record["fit_metrics"],
            "draft_path": draft_path,
            "snippet_path": snippet_path,
            "qc_path": qc_path,
        },
    )
    print(f"Saved draft W-track geometry to {draft_path}")
    print(f"Saved QC overlay to {qc_path}")
    print(f"Saved paste-ready geometry snippet to {snippet_path}")
    print(f"Saved run metadata to {log_path}")
    return {
        "draft_path": draft_path,
        "qc_path": qc_path,
        "snippet_path": snippet_path,
        "log_path": log_path,
        "fit_ok": bool(fit_result["fit_ok"]),
    }


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the W-track fitter."""
    parser = argparse.ArgumentParser(
        description="Fit one draft W-track geometry from session position data."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name.")
    parser.add_argument("--date", required=True, help="Recording date in YYYYMMDD format.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        default=None,
        help="Optional run epochs to fit. Default: all run epochs with position data.",
    )
    parser.add_argument(
        "--position-source",
        default="auto",
        choices=POSITION_SOURCE_CHOICES,
        help=(
            "Which session position source to use. "
            "Default: auto (cleaned DLC head, then position.pkl, then body_position.pkl)."
        ),
    )
    parser.add_argument(
        "--clean-dlc-input-dirname",
        default=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        help=(
            "Directory under the session analysis path containing combined cleaned DLC output. "
            f"Default: {DEFAULT_CLEAN_DLC_POSITION_DIRNAME}"
        ),
    )
    parser.add_argument(
        "--clean-dlc-input-name",
        default=DEFAULT_CLEAN_DLC_POSITION_NAME,
        help=f"Combined cleaned DLC filename. Default: {DEFAULT_CLEAN_DLC_POSITION_NAME}",
    )
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=(
            "Directory under the session analysis path used for fitter outputs. "
            f"Default: {DEFAULT_OUTPUT_DIRNAME}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the W-track geometry fitter CLI."""
    args = parse_arguments(argv)
    fit_wtrack_geometry(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        epochs=args.epochs,
        position_source=args.position_source,
        clean_dlc_input_dirname=args.clean_dlc_input_dirname,
        clean_dlc_input_name=args.clean_dlc_input_name,
        output_dirname=args.output_dirname,
    )


if __name__ == "__main__":
    main()
