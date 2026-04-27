from __future__ import annotations

"""Compare motor behavior across run epochs for one session.

This script compares motor-variable distributions across run epochs using two
complementary views:

- pooled movement-only distributions, which reveal overall epoch-to-epoch drift
- trajectory- and task-progression-conditioned summaries, which reduce
  confounds from changes in route occupancy along the W-track

By default the script saves three parquet tables under ``analysis_path /
"motor"`` and one raw-distribution figure plus one controlled progression
figure for each motor variable under ``analysis_path / "figs" / "motor"``.
"""

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    TRAJECTORY_TYPES,
    build_movement_interval,
    get_analysis_path,
    get_run_epochs,
    load_available_position_pickle_data,
    load_clean_dlc_position_data,
    load_epoch_tags,
    load_position_timestamps,
    load_trajectory_time_bounds,
)
from v1ca1.helper.wtrack import get_wtrack_total_length


DEFAULT_PROGRESSION_BIN_SIZE_CM = 4.0
DEFAULT_N_HIST_BINS = 40
MOTOR_VARIABLES = (
    "speed_cm_s",
    "acceleration_cm_s2",
    "head_direction_deg",
    "head_angular_velocity_deg_s",
    "head_angular_acceleration_deg_s2",
    "head_angular_speed_deg_s",
)
VARIABLE_LABELS = {
    "speed_cm_s": "Speed (cm/s)",
    "acceleration_cm_s2": "Acceleration (cm/s^2)",
    "head_direction_deg": "Head direction (deg)",
    "head_angular_velocity_deg_s": "Head angular velocity (deg/s)",
    "head_angular_acceleration_deg_s2": "Head angular acceleration (deg/s^2)",
    "head_angular_speed_deg_s": "Head angular speed (deg/s)",
}


def select_run_epochs(
    run_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return the requested run epochs, defaulting to all available run epochs."""
    if not requested_epochs:
        return list(run_epochs)

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in available run epochs {run_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return list(requested_epochs)


def load_motor_position_data(
    analysis_path: Path,
    required_epochs: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    """Load per-epoch head and body XY arrays with cleaned-DLC precedence."""
    clean_dlc_path = (
        analysis_path
        / DEFAULT_CLEAN_DLC_POSITION_DIRNAME
        / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    if clean_dlc_path.exists():
        _epoch_order, head_position, body_position = load_clean_dlc_position_data(
            analysis_path,
            input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
            input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
            validate_timestamps=True,
        )
        source = str(clean_dlc_path)
    else:
        head_position = load_available_position_pickle_data(
            analysis_path,
            input_name="position.pkl",
        )
        body_position = load_available_position_pickle_data(
            analysis_path,
            input_name="body_position.pkl",
        )
        source = "position.pkl + body_position.pkl"

    required_epoch_list = list(required_epochs or [])
    missing_head_epochs = [epoch for epoch in required_epoch_list if epoch not in head_position]
    missing_body_epochs = [epoch for epoch in required_epoch_list if epoch not in body_position]
    if missing_head_epochs or missing_body_epochs:
        raise ValueError(
            "Motor position epochs do not cover all requested run epochs. "
            f"Missing head epochs: {missing_head_epochs!r}; "
            f"missing body epochs: {missing_body_epochs!r}"
        )

    return (
        {
            str(epoch): np.asarray(values, dtype=float)
            for epoch, values in head_position.items()
        },
        {
            str(epoch): np.asarray(values, dtype=float)
            for epoch, values in body_position.items()
        },
        source,
    )


def compute_motor_variables(
    position_xy: np.ndarray,
    body_xy: np.ndarray,
    timestamps_position: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-sample motor variables on a shared timestamp grid."""
    import position_tools as pt

    position_xy = np.asarray(position_xy, dtype=float)
    body_xy = np.asarray(body_xy, dtype=float)
    timestamps_position = np.asarray(timestamps_position, dtype=float)

    if position_xy.shape != body_xy.shape:
        raise ValueError(
            "Head and body position arrays must have matching shapes. "
            f"Got {position_xy.shape} and {body_xy.shape}."
        )
    if position_xy.ndim != 2 or position_xy.shape[1] != 2:
        raise ValueError(f"Expected XY position arrays, got shape {position_xy.shape}.")
    if position_xy.shape[0] != timestamps_position.size:
        raise ValueError(
            "Position samples and timestamps must have matching lengths. "
            f"Got {position_xy.shape[0]} and {timestamps_position.size}."
        )
    if timestamps_position.size < 2:
        raise ValueError("At least two timestamps are required to compute motor variables.")
    if not np.all(np.isfinite(position_xy)) or not np.all(np.isfinite(body_xy)):
        raise ValueError("Position arrays contain non-finite values.")
    if not np.all(np.isfinite(timestamps_position)):
        raise ValueError("Position timestamps contain non-finite values.")
    if np.any(np.diff(timestamps_position) <= 0):
        raise ValueError("Position timestamps must be strictly increasing.")

    sampling_rate = (timestamps_position.size - 1) / (
        timestamps_position[-1] - timestamps_position[0]
    )
    speed = np.asarray(
        pt.get_speed(
            position=position_xy,
            time=timestamps_position,
            sampling_frequency=float(sampling_rate),
            sigma=0.1,
        ),
        dtype=float,
    )
    if speed.shape[0] != timestamps_position.size:
        raise ValueError("Speed computation returned the wrong number of samples.")

    acceleration = np.asarray(
        np.gradient(speed, timestamps_position),
        dtype=float,
    )
    head_vector = position_xy - body_xy
    head_direction_rad = np.arctan2(head_vector[:, 1], head_vector[:, 0])
    head_direction_deg = np.rad2deg(head_direction_rad)
    head_direction_unwrapped = np.unwrap(head_direction_rad)
    head_angular_velocity_deg_s = np.rad2deg(
        np.gradient(head_direction_unwrapped, timestamps_position)
    )
    head_angular_acceleration_deg_s2 = np.asarray(
        np.gradient(head_angular_velocity_deg_s, timestamps_position),
        dtype=float,
    )

    return {
        "speed_cm_s": speed,
        "acceleration_cm_s2": acceleration,
        "head_direction_deg": head_direction_deg,
        "head_angular_velocity_deg_s": head_angular_velocity_deg_s,
        "head_angular_acceleration_deg_s2": head_angular_acceleration_deg_s2,
        "head_angular_speed_deg_s": np.abs(head_angular_velocity_deg_s),
    }


def filter_finite_position_samples(
    timestamps_position: np.ndarray,
    position_xy: np.ndarray,
    body_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop samples with non-finite timestamps or head/body XY coordinates."""
    timestamps_position = np.asarray(timestamps_position, dtype=float)
    position_xy = np.asarray(position_xy, dtype=float)
    body_xy = np.asarray(body_xy, dtype=float)

    finite_mask = (
        np.isfinite(timestamps_position)
        & np.all(np.isfinite(position_xy), axis=1)
        & np.all(np.isfinite(body_xy), axis=1)
    )
    dropped_count = int((~finite_mask).sum())
    filtered_timestamps = timestamps_position[finite_mask]
    filtered_head = position_xy[finite_mask]
    filtered_body = body_xy[finite_mask]

    if filtered_timestamps.size < 2:
        raise ValueError(
            "Fewer than two finite position samples remain after dropping non-finite values."
        )
    if np.any(np.diff(filtered_timestamps) <= 0):
        raise ValueError(
            "Finite position timestamps must remain strictly increasing after filtering."
        )

    return filtered_timestamps, filtered_head, filtered_body, dropped_count


def build_interval_mask(
    timestamps: np.ndarray,
    interval_bounds: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask for timestamps that fall within any interval."""
    timestamps = np.asarray(timestamps, dtype=float)
    interval_bounds = np.asarray(interval_bounds, dtype=float)

    mask = np.zeros(timestamps.shape, dtype=bool)
    if interval_bounds.size == 0:
        return mask
    if interval_bounds.ndim != 2 or interval_bounds.shape[1] != 2:
        raise ValueError(
            "interval_bounds must have shape (n, 2). "
            f"Got {interval_bounds.shape}."
        )

    for start_time, end_time in interval_bounds:
        mask |= (timestamps >= float(start_time)) & (timestamps <= float(end_time))
    return mask


def compute_progression_by_trajectory(
    animal_name: str,
    position_xy: np.ndarray,
    _trajectory_times: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute normalized within-trajectory progression for each trajectory type."""
    import track_linearization as tl

    from v1ca1.helper.session import get_track_graph_for_trajectory

    total_length = get_wtrack_total_length(animal_name)
    progression_by_trajectory: dict[str, np.ndarray] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        track_graph, edge_order = get_track_graph_for_trajectory(
            animal_name,
            trajectory_type,
        )
        position_df = tl.get_linearized_position(
            position=np.asarray(position_xy, dtype=float),
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=0,
        )
        progression_by_trajectory[trajectory_type] = np.clip(
            np.asarray(position_df["linear_position"], dtype=float) / float(total_length),
            0.0,
            1.0,
        )
    return progression_by_trajectory


def build_progression_bin_edges(
    animal_name: str,
    progression_bin_size_cm: float,
) -> np.ndarray:
    """Return normalized within-trajectory progression bin edges."""
    if progression_bin_size_cm <= 0:
        raise ValueError("--progression-bin-size-cm must be positive.")

    bin_size = float(progression_bin_size_cm) / float(get_wtrack_total_length(animal_name))
    edges = np.arange(0.0, 1.0 + bin_size, bin_size, dtype=float)
    if edges[-1] < 1.0:
        edges = np.append(edges, 1.0)
    else:
        edges[-1] = 1.0
    return edges


def circular_mean_deg(values_deg: np.ndarray) -> tuple[float, float]:
    """Return the circular mean angle and resultant length for degree samples."""
    values_deg = np.asarray(values_deg, dtype=float)
    if values_deg.size == 0:
        return np.nan, np.nan

    angles_rad = np.deg2rad(values_deg)
    sin_mean = float(np.mean(np.sin(angles_rad)))
    cos_mean = float(np.mean(np.cos(angles_rad)))
    resultant_length = float(np.hypot(sin_mean, cos_mean))
    circular_mean = float(np.rad2deg(np.arctan2(sin_mean, cos_mean)))
    return circular_mean, resultant_length


def summarize_distribution_values(
    epoch: str,
    variable_name: str,
    values: np.ndarray,
    movement_duration_s: float,
) -> dict[str, Any]:
    """Return one summary row for one epoch and motor variable."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        row = {
            "epoch": epoch,
            "variable": variable_name,
            "sample_count": 0,
            "movement_duration_s": float(movement_duration_s),
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "circular_mean_deg": np.nan,
            "resultant_length": np.nan,
        }
        return row

    row = {
        "epoch": epoch,
        "variable": variable_name,
        "sample_count": int(values.size),
        "movement_duration_s": float(movement_duration_s),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "p10": float(np.percentile(values, 10.0)),
        "p90": float(np.percentile(values, 90.0)),
        "circular_mean_deg": np.nan,
        "resultant_length": np.nan,
    }
    if variable_name == "head_direction_deg":
        circular_mean, resultant_length = circular_mean_deg(values)
        row["mean"] = np.nan
        row["std"] = np.nan
        row["circular_mean_deg"] = circular_mean
        row["resultant_length"] = resultant_length
    return row


def build_distribution_summary_table(
    selected_epochs: list[str],
    movement_values_by_epoch: dict[str, dict[str, np.ndarray]],
    movement_durations_by_epoch: dict[str, float],
) -> pd.DataFrame:
    """Build the per-epoch motor distribution summary table."""
    rows: list[dict[str, Any]] = []
    for epoch in selected_epochs:
        for variable_name in MOTOR_VARIABLES:
            rows.append(
                summarize_distribution_values(
                    epoch=epoch,
                    variable_name=variable_name,
                    values=movement_values_by_epoch[epoch][variable_name],
                    movement_duration_s=movement_durations_by_epoch[epoch],
                )
            )

    table = pd.DataFrame.from_records(rows)
    table["epoch"] = pd.Categorical(table["epoch"], categories=selected_epochs, ordered=True)
    table["variable"] = pd.Categorical(
        table["variable"],
        categories=list(MOTOR_VARIABLES),
        ordered=True,
    )
    return table.sort_values(["variable", "epoch"], kind="stable").reset_index(drop=True)


def build_histogram_bin_edges(
    variable_name: str,
    values_by_epoch: dict[str, np.ndarray],
    n_hist_bins: int,
) -> np.ndarray:
    """Return fixed histogram edges for one variable across all epochs."""
    if n_hist_bins < 2:
        raise ValueError("--n-hist-bins must be at least 2.")
    if variable_name == "head_direction_deg":
        return np.linspace(-180.0, 180.0, n_hist_bins + 1, dtype=float)

    finite_value_parts = [
        np.asarray(values, dtype=float)[np.isfinite(values)]
        for values in values_by_epoch.values()
        if np.asarray(values).size > 0
    ]
    if not finite_value_parts:
        return np.linspace(0.0, 1.0, n_hist_bins + 1, dtype=float)

    finite_values = np.concatenate(finite_value_parts)
    if finite_values.size == 0:
        return np.linspace(0.0, 1.0, n_hist_bins + 1, dtype=float)

    lower = float(np.percentile(finite_values, 1.0))
    upper = float(np.percentile(finite_values, 99.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
    if not np.isfinite(lower) or not np.isfinite(upper):
        lower, upper = 0.0, 1.0
    if upper <= lower:
        pad = max(abs(lower) * 0.05, 1.0)
        lower -= pad
        upper += pad
    return np.linspace(lower, upper, n_hist_bins + 1, dtype=float)


def histogram_probabilities(
    values: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """Return normalized histogram probabilities for one value vector."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)

    counts, _ = np.histogram(values, bins=np.asarray(bin_edges, dtype=float))
    total = int(counts.sum())
    if total <= 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    return counts.astype(float) / float(total)


def build_pairwise_distance_table(
    *,
    selected_epochs: list[str],
    values_by_epoch: dict[str, dict[str, np.ndarray]],
    bin_edges_by_variable: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build the full pairwise Jensen-Shannon divergence table."""
    rows: list[dict[str, Any]] = []

    for variable_name in MOTOR_VARIABLES:
        epoch_probabilities = {
            epoch: histogram_probabilities(
                values_by_epoch[epoch][variable_name],
                bin_edges_by_variable[variable_name],
            )
            for epoch in selected_epochs
        }
        for epoch_a in selected_epochs:
            for epoch_b in selected_epochs:
                probability_a = epoch_probabilities[epoch_a]
                probability_b = epoch_probabilities[epoch_b]
                if probability_a.sum() == 0 or probability_b.sum() == 0:
                    divergence = np.nan
                else:
                    distance = float(jensenshannon(probability_a, probability_b, base=2.0))
                    divergence = distance**2
                rows.append(
                    {
                        "variable": variable_name,
                        "epoch_a": epoch_a,
                        "epoch_b": epoch_b,
                        "jensen_shannon_divergence": divergence,
                    }
                )

    table = pd.DataFrame.from_records(rows)
    table["variable"] = pd.Categorical(
        table["variable"],
        categories=list(MOTOR_VARIABLES),
        ordered=True,
    )
    table["epoch_a"] = pd.Categorical(
        table["epoch_a"],
        categories=selected_epochs,
        ordered=True,
    )
    table["epoch_b"] = pd.Categorical(
        table["epoch_b"],
        categories=selected_epochs,
        ordered=True,
    )
    return table.sort_values(["variable", "epoch_a", "epoch_b"], kind="stable").reset_index(
        drop=True
    )


def summarize_progression_values(
    *,
    epoch: str,
    trajectory_type: str,
    variable_name: str,
    progression: np.ndarray,
    values: np.ndarray,
    progression_bin_edges: np.ndarray,
) -> pd.DataFrame:
    """Summarize one variable along normalized progression for one epoch and trajectory."""
    progression = np.asarray(progression, dtype=float)
    values = np.asarray(values, dtype=float)
    if progression.shape != values.shape:
        raise ValueError(
            "Progression and value arrays must have matching shapes. "
            f"Got {progression.shape} and {values.shape}."
        )

    finite_mask = np.isfinite(progression) & np.isfinite(values)
    progression = progression[finite_mask]
    values = values[finite_mask]
    if progression.size == 0:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    rows: list[dict[str, Any]] = []
    n_bins = len(progression_bin_edges) - 1
    for bin_index in range(n_bins):
        bin_start = float(progression_bin_edges[bin_index])
        bin_end = float(progression_bin_edges[bin_index + 1])
        if bin_index == n_bins - 1:
            in_bin = (progression >= bin_start) & (progression <= bin_end)
        else:
            in_bin = (progression >= bin_start) & (progression < bin_end)
        if not np.any(in_bin):
            continue

        bin_values = values[in_bin]
        rows.append(
            {
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "variable": variable_name,
                "progression_bin_index": int(bin_index),
                "progression_bin_start": bin_start,
                "progression_bin_end": bin_end,
                "progression_bin_center": float((bin_start + bin_end) / 2.0),
                "sample_count": int(bin_values.size),
                "median": float(np.median(bin_values)),
                "q25": float(np.percentile(bin_values, 25.0)),
                "q75": float(np.percentile(bin_values, 75.0)),
            }
        )

    return pd.DataFrame.from_records(rows)


def build_progression_summary_table(
    *,
    selected_epochs: list[str],
    controlled_data_by_epoch: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]],
    progression_bin_edges: np.ndarray,
) -> pd.DataFrame:
    """Build the trajectory- and progression-conditioned motor summary table."""
    tables: list[pd.DataFrame] = []
    for epoch in selected_epochs:
        for trajectory_type in TRAJECTORY_TYPES:
            for variable_name in MOTOR_VARIABLES:
                progression, values = controlled_data_by_epoch[epoch][trajectory_type][
                    variable_name
                ]
                tables.append(
                    summarize_progression_values(
                        epoch=epoch,
                        trajectory_type=trajectory_type,
                        variable_name=variable_name,
                        progression=progression,
                        values=values,
                        progression_bin_edges=progression_bin_edges,
                    )
                )

    if tables:
        table = pd.concat(tables, ignore_index=True)
    else:
        table = pd.DataFrame(
            columns=[
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
            ]
        )

    if not table.empty:
        table["epoch"] = pd.Categorical(
            table["epoch"],
            categories=selected_epochs,
            ordered=True,
        )
        table["trajectory_type"] = pd.Categorical(
            table["trajectory_type"],
            categories=list(TRAJECTORY_TYPES),
            ordered=True,
        )
        table["variable"] = pd.Categorical(
            table["variable"],
            categories=list(MOTOR_VARIABLES),
            ordered=True,
        )
        table = table.sort_values(
            ["variable", "trajectory_type", "epoch", "progression_bin_index"],
            kind="stable",
        ).reset_index(drop=True)
    return table


def plot_ecdf(ax: Any, values: np.ndarray, label: str) -> None:
    """Plot one empirical CDF."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    ax.plot(x, y, linewidth=1.5, label=label)


def plot_pairwise_distance_heatmap(
    ax: Any,
    distance_table: pd.DataFrame,
    selected_epochs: list[str],
) -> None:
    """Plot a square epoch-by-epoch divergence heatmap."""
    if distance_table.empty:
        matrix = np.full((len(selected_epochs), len(selected_epochs)), np.nan, dtype=float)
    else:
        matrix = (
            distance_table.pivot(
                index="epoch_a",
                columns="epoch_b",
                values="jensen_shannon_divergence",
            )
            .reindex(index=selected_epochs, columns=selected_epochs)
            .to_numpy(dtype=float)
        )

    image = ax.imshow(matrix, origin="upper", cmap="magma")
    ax.set_xticks(np.arange(len(selected_epochs)))
    ax.set_xticklabels(selected_epochs, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(selected_epochs)))
    ax.set_yticklabels(selected_epochs)
    ax.set_title("Pairwise JS divergence")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_raw_distribution_figure(
    *,
    output_path: Path,
    variable_name: str,
    selected_epochs: list[str],
    movement_values_by_epoch: dict[str, dict[str, np.ndarray]],
    distance_table: pd.DataFrame,
    bin_edges: np.ndarray,
) -> None:
    """Save the pooled movement distribution figure for one variable."""
    label = VARIABLE_LABELS[variable_name]
    if variable_name == "head_direction_deg":
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        hist_ax, heatmap_ax = axes
    else:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        hist_ax, ecdf_ax, heatmap_ax = axes

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    for epoch in selected_epochs:
        values = movement_values_by_epoch[epoch][variable_name]
        probabilities = histogram_probabilities(values, bin_edges)
        hist_ax.plot(bin_centers, probabilities, drawstyle="steps-mid", linewidth=1.5, label=epoch)
        if variable_name != "head_direction_deg":
            plot_ecdf(ecdf_ax, values, epoch)

    hist_ax.set_title(f"{label} distribution")
    hist_ax.set_xlabel(label)
    hist_ax.set_ylabel("Probability")
    hist_ax.legend(frameon=False, fontsize=8)

    if variable_name == "head_direction_deg":
        hist_ax.set_xlim(-180.0, 180.0)
        hist_ax.set_xticks(np.linspace(-180.0, 180.0, 5))
    else:
        ecdf_ax.set_title(f"{label} ECDF")
        ecdf_ax.set_xlabel(label)
        ecdf_ax.set_ylabel("ECDF")
        ecdf_ax.set_ylim(0.0, 1.0)
        ecdf_ax.legend(frameon=False, fontsize=8)

    plot_pairwise_distance_heatmap(heatmap_ax, distance_table, selected_epochs)
    figure.suptitle(label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_controlled_progression_figure(
    *,
    output_path: Path,
    variable_name: str,
    selected_epochs: list[str],
    progression_summary: pd.DataFrame,
) -> None:
    """Save the trajectory-conditioned progression summary figure for one variable."""
    label = VARIABLE_LABELS[variable_name]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    axes_flat = axes.ravel()

    variable_table = progression_summary.loc[
        progression_summary["variable"].astype(str) == variable_name
    ].copy()
    for ax, trajectory_type in zip(axes_flat, TRAJECTORY_TYPES, strict=True):
        trajectory_table = variable_table.loc[
            variable_table["trajectory_type"].astype(str) == trajectory_type
        ].copy()
        for epoch in selected_epochs:
            epoch_table = trajectory_table.loc[
                trajectory_table["epoch"].astype(str) == epoch
            ].sort_values("progression_bin_index", kind="stable")
            if epoch_table.empty:
                continue

            x = epoch_table["progression_bin_center"].to_numpy(dtype=float)
            median = epoch_table["median"].to_numpy(dtype=float)
            q25 = epoch_table["q25"].to_numpy(dtype=float)
            q75 = epoch_table["q75"].to_numpy(dtype=float)

            ax.plot(x, median, linewidth=1.8, label=epoch)
            if epoch_table["sample_count"].to_numpy(dtype=int).max(initial=0) >= 3:
                ax.fill_between(x, q25, q75, alpha=0.15)

        ax.set_title(trajectory_type)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Normalized task progression")
        ax.set_ylabel(label)

    axes_flat[0].legend(frameon=False, fontsize=8)
    figure.suptitle(f"{label} by trajectory and progression")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def compare_epoch_motor_behavior(
    *,
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    requested_epochs: list[str] | None = None,
    progression_bin_size_cm: float = DEFAULT_PROGRESSION_BIN_SIZE_CM,
    n_hist_bins: int = DEFAULT_N_HIST_BINS,
) -> dict[str, Any]:
    """Run the epoch motor comparison workflow for one session."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if speed_threshold_cm_s < 0:
        raise ValueError("--speed-threshold-cm-s must be non-negative.")

    print(f"Processing {animal_name} {date}.")
    print(f"Using analysis path: {analysis_path}")

    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    run_epochs = get_run_epochs(epoch_tags)
    selected_epochs = select_run_epochs(run_epochs, requested_epochs)
    print(f"Found {len(run_epochs)} run epochs: {run_epochs!r}")
    if requested_epochs:
        print(f"Requested epochs: {requested_epochs!r}")

    position_epoch_tags, timestamps_position, position_timestamp_source = load_position_timestamps(
        analysis_path
    )
    position_epoch_set = set(position_epoch_tags)
    head_position_all, body_position_all, position_source = load_motor_position_data(
        analysis_path,
    )
    available_epochs = [
        epoch
        for epoch in selected_epochs
        if epoch in position_epoch_set
        and epoch in head_position_all
        and epoch in body_position_all
    ]
    skipped_epochs = [epoch for epoch in selected_epochs if epoch not in set(available_epochs)]
    if skipped_epochs:
        if requested_epochs:
            raise ValueError(
                "Requested run epochs are missing position timestamps or motor position arrays: "
                f"{skipped_epochs!r}"
            )
        print(
            "Skipping run epochs missing position timestamps or motor position arrays: "
            f"{skipped_epochs!r}"
        )
    if not available_epochs:
        raise ValueError("No run epochs had the required timestamps and motor position inputs.")

    selected_epochs = available_epochs
    print(f"Using epochs: {selected_epochs!r}")
    head_position_by_epoch = {epoch: head_position_all[epoch] for epoch in selected_epochs}
    body_position_by_epoch = {epoch: body_position_all[epoch] for epoch in selected_epochs}
    trajectory_times, trajectory_source = load_trajectory_time_bounds(analysis_path, selected_epochs)
    print(f"Using position source: {position_source}")
    print(f"Using position timestamp source: {position_timestamp_source}")
    print(f"Using trajectory source: {trajectory_source}")

    motor_output_dir = analysis_path / "motor"
    figure_output_dir = analysis_path / "figs" / "motor"
    motor_output_dir.mkdir(parents=True, exist_ok=True)
    figure_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing tables to {motor_output_dir}")
    print(f"Writing figures to {figure_output_dir}")

    movement_values_by_epoch: dict[str, dict[str, np.ndarray]] = {}
    movement_durations_by_epoch: dict[str, float] = {}
    controlled_data_by_epoch: dict[str, dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    for epoch in selected_epochs:
        print(f"Computing motor variables for epoch {epoch}...")
        trimmed_head = np.asarray(head_position_by_epoch[epoch][position_offset:], dtype=float)
        trimmed_body = np.asarray(body_position_by_epoch[epoch][position_offset:], dtype=float)
        trimmed_timestamps = np.asarray(
            timestamps_position[epoch][position_offset:],
            dtype=float,
        )
        (
            filtered_timestamps,
            filtered_head,
            filtered_body,
            dropped_nonfinite_samples,
        ) = filter_finite_position_samples(
            trimmed_timestamps,
            trimmed_head,
            trimmed_body,
        )
        if dropped_nonfinite_samples:
            print(
                f"  dropped {dropped_nonfinite_samples} non-finite head/body position samples "
                "before computing motor variables"
            )

        motor_variables = compute_motor_variables(
            filtered_head,
            filtered_body,
            filtered_timestamps,
        )
        import pynapple as nap

        speed_tsd = nap.Tsd(
            t=filtered_timestamps,
            d=np.asarray(motor_variables["speed_cm_s"], dtype=float),
            time_units="s",
        )
        movement_interval = build_movement_interval(
            speed_tsd,
            speed_threshold_cm_s=speed_threshold_cm_s,
        )
        movement_bounds = np.column_stack(
            (
                np.asarray(movement_interval.start, dtype=float),
                np.asarray(movement_interval.end, dtype=float),
            )
        )
        movement_mask = build_interval_mask(filtered_timestamps, movement_bounds)
        progression_by_trajectory = compute_progression_by_trajectory(
            animal_name,
            filtered_head,
            trajectory_times[epoch],
        )

        movement_values_by_epoch[epoch] = {
            variable_name: np.asarray(values[movement_mask], dtype=float)
            for variable_name, values in motor_variables.items()
        }
        movement_durations_by_epoch[epoch] = float(movement_interval.tot_length())
        movement_sample_count = int(movement_mask.sum())
        trajectory_counts = {
            trajectory_type: int(
                (
                    movement_mask
                    & build_interval_mask(
                        filtered_timestamps,
                        trajectory_times[epoch][trajectory_type],
                    )
                ).sum()
            )
            for trajectory_type in TRAJECTORY_TYPES
        }
        print(
            f"  movement duration: {movement_durations_by_epoch[epoch]:.2f} s, "
            f"samples kept: {movement_sample_count}, "
            f"trajectory samples: {trajectory_counts}"
        )
        controlled_data_by_epoch[epoch] = {}

        for trajectory_type in TRAJECTORY_TYPES:
            trajectory_mask = build_interval_mask(
                filtered_timestamps,
                trajectory_times[epoch][trajectory_type],
            )
            combined_mask = movement_mask & trajectory_mask
            controlled_data_by_epoch[epoch][trajectory_type] = {}
            progression_values = progression_by_trajectory[trajectory_type][combined_mask]
            for variable_name in MOTOR_VARIABLES:
                controlled_data_by_epoch[epoch][trajectory_type][variable_name] = (
                    progression_values,
                    np.asarray(motor_variables[variable_name][combined_mask], dtype=float),
                )

    print("Building summary tables...")
    distribution_summary = build_distribution_summary_table(
        selected_epochs,
        movement_values_by_epoch,
        movement_durations_by_epoch,
    )
    progression_bin_edges = build_progression_bin_edges(
        animal_name,
        progression_bin_size_cm,
    )
    progression_summary = build_progression_summary_table(
        selected_epochs=selected_epochs,
        controlled_data_by_epoch=controlled_data_by_epoch,
        progression_bin_edges=progression_bin_edges,
    )
    print(
        "Summary table sizes: "
        f"distribution={len(distribution_summary)}, "
        f"progression={len(progression_summary)}"
    )

    bin_edges_by_variable = {
        variable_name: build_histogram_bin_edges(
            variable_name,
            {
                epoch: movement_values_by_epoch[epoch][variable_name]
                for epoch in selected_epochs
            },
            n_hist_bins,
        )
        for variable_name in MOTOR_VARIABLES
    }
    print("Computing pairwise distribution distances...")
    pairwise_distances = build_pairwise_distance_table(
        selected_epochs=selected_epochs,
        values_by_epoch=movement_values_by_epoch,
        bin_edges_by_variable=bin_edges_by_variable,
    )
    print(f"Pairwise distance rows: {len(pairwise_distances)}")

    distribution_summary_path = motor_output_dir / "epoch_motor_distribution_summary.parquet"
    pairwise_distance_path = motor_output_dir / "epoch_motor_pairwise_distances.parquet"
    progression_summary_path = motor_output_dir / "epoch_motor_progression_summary.parquet"
    print("Saving parquet outputs...")
    distribution_summary.to_parquet(distribution_summary_path, index=False)
    pairwise_distances.to_parquet(pairwise_distance_path, index=False)
    progression_summary.to_parquet(progression_summary_path, index=False)
    print(f"Saved distribution summary to {distribution_summary_path}")
    print(f"Saved pairwise distances to {pairwise_distance_path}")
    print(f"Saved progression summary to {progression_summary_path}")

    raw_figure_paths: dict[str, Path] = {}
    controlled_figure_paths: dict[str, Path] = {}
    print("Saving figures...")
    for variable_name in MOTOR_VARIABLES:
        raw_figure_path = figure_output_dir / f"epoch_motor_{variable_name}_raw.png"
        controlled_figure_path = (
            figure_output_dir / f"epoch_motor_{variable_name}_controlled.png"
        )
        print(f"  Rendering figures for {variable_name}...")
        save_raw_distribution_figure(
            output_path=raw_figure_path,
            variable_name=variable_name,
            selected_epochs=selected_epochs,
            movement_values_by_epoch=movement_values_by_epoch,
            distance_table=pairwise_distances.loc[
                pairwise_distances["variable"].astype(str) == variable_name
            ].copy(),
            bin_edges=bin_edges_by_variable[variable_name],
        )
        save_controlled_progression_figure(
            output_path=controlled_figure_path,
            variable_name=variable_name,
            selected_epochs=selected_epochs,
            progression_summary=progression_summary,
        )
        raw_figure_paths[variable_name] = raw_figure_path
        controlled_figure_paths[variable_name] = controlled_figure_path
        print(f"    saved raw figure to {raw_figure_path}")
        print(f"    saved controlled figure to {controlled_figure_path}")

    outputs: dict[str, Any] = {
        "selected_epochs": selected_epochs,
        "skipped_epochs": skipped_epochs,
        "distribution_summary_path": distribution_summary_path,
        "pairwise_distance_path": pairwise_distance_path,
        "progression_summary_path": progression_summary_path,
        "raw_figure_paths": raw_figure_paths,
        "controlled_figure_paths": controlled_figure_paths,
        "position_source": position_source,
        "position_timestamp_source": position_timestamp_source,
        "trajectory_source": trajectory_source,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.motor.compare_epoch_motor_behavior",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "epochs": selected_epochs,
            "position_offset": position_offset,
            "speed_threshold_cm_s": speed_threshold_cm_s,
            "progression_bin_size_cm": progression_bin_size_cm,
            "n_hist_bins": n_hist_bins,
        },
        outputs=outputs,
    )
    outputs["log_path"] = log_path
    print(f"Saved motor comparison outputs to {motor_output_dir}")
    print(f"Saved run log to {log_path}")
    return outputs


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the epoch motor comparison CLI."""
    parser = argparse.ArgumentParser(
        description="Compare motor behavior across run epochs for one session"
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="Specific run epoch labels to compare. Defaults to all run epochs.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore per epoch. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--progression-bin-size-cm",
        type=float,
        default=DEFAULT_PROGRESSION_BIN_SIZE_CM,
        help=(
            "Within-trajectory progression bin size in cm. "
            f"Default: {DEFAULT_PROGRESSION_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--n-hist-bins",
        type=int,
        default=DEFAULT_N_HIST_BINS,
        help=f"Number of fixed histogram bins per variable. Default: {DEFAULT_N_HIST_BINS}",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run the epoch motor comparison CLI."""
    args = parse_arguments()
    compare_epoch_motor_behavior(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        requested_epochs=args.epochs,
        progression_bin_size_cm=args.progression_bin_size_cm,
        n_hist_bins=args.n_hist_bins,
    )


if __name__ == "__main__":
    main()
