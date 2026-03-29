from __future__ import annotations

"""Shared task-progression session loading and preprocessing helpers.

This module now keeps task-progression-specific representations local while
reusing the generic session-loading and movement helpers from
`v1ca1.helper.session`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    POSITION_SOURCE_CHOICES,
    REGIONS,
    TRAJECTORY_TYPES,
    TURN_TRAJECTORY_PAIRS,
    _get_track_graph_for_trajectory,
    build_epoch_interval,
    build_movement_interval,
    build_speed_tsd,
    compute_movement_firing_rates,
    get_analysis_path,
    get_run_epochs,
    load_ephys_timestamps_all,
    load_body_position_data_with_precedence,
    load_epoch_tags,
    load_position_data_with_precedence,
    load_position_timestamps,
    load_spikes_by_region,
    load_trajectory_intervals,
)
from v1ca1.helper.wtrack import (
    get_wtrack_branch_graph,
    get_wtrack_branch_side,
    get_wtrack_total_length,
)

if TYPE_CHECKING:
    import pandas as pd
    import pynapple as nap
    import xarray as xr



def build_task_progression_by_trajectory(
    animal_name: str,
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, "nap.IntervalSet"],
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> dict[str, "nap.Tsd"]:
    """Build normalized task-progression Tsds for each trajectory in one epoch."""
    import pynapple as nap
    import track_linearization as tl

    epoch_position = position[position_offset:]
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    total_length_per_trajectory = get_wtrack_total_length(animal_name)
    progression_by_trajectory: dict[str, nap.Tsd] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        track_graph, edge_order = _get_track_graph_for_trajectory(
            animal_name,
            trajectory_type,
        )
        position_df = tl.get_linearized_position(
            position=epoch_position,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=0,
        )
        progression_by_trajectory[trajectory_type] = nap.Tsd(
            t=epoch_timestamps,
            d=np.asarray(position_df["linear_position"], dtype=float)
            / total_length_per_trajectory,
            time_support=trajectory_intervals[trajectory_type],
            time_units="s",
        )
    return progression_by_trajectory


def build_linear_position(
    animal_name: str,
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, "nap.IntervalSet"],
    movement_interval: "nap.IntervalSet",
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> "nap.Tsd":
    """Build the concatenated four-trajectory linear-position coordinate for one epoch."""
    import pynapple as nap
    import track_linearization as tl

    epoch_position = position[position_offset:]
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    total_length_per_trajectory = get_wtrack_total_length(animal_name)
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        track_graph, edge_order = get_wtrack_branch_graph(
            animal_name=animal_name,
            branch_side=get_wtrack_branch_side(trajectory_type),
            direction="from_center",
        )
        position_df = tl.get_linearized_position(
            position=epoch_position,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=0,
        )
        linear_position = nap.Tsd(
            t=epoch_timestamps,
            d=np.asarray(position_df["linear_position"], dtype=float)
            + total_length_per_trajectory * trajectory_index,
            time_support=trajectory_intervals[trajectory_type],
            time_units="s",
        )
        time_chunks.append(np.asarray(linear_position.t, dtype=float))
        value_chunks.append(np.asarray(linear_position.d, dtype=float))

    if not time_chunks:
        return nap.Tsd(t=np.array([], dtype=float), d=np.array([], dtype=float), time_units="s")

    all_times = np.concatenate(time_chunks)
    all_values = np.concatenate(value_chunks)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=movement_interval,
        time_units="s",
    )


def build_task_progression(
    animal_name: str,
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, "nap.IntervalSet"],
    movement_interval: "nap.IntervalSet",
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> "nap.Tsd":
    """Build the two-branch task-progression coordinate for one epoch."""
    import pynapple as nap
    import track_linearization as tl

    epoch_position = position[position_offset:]
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    total_length_per_trajectory = get_wtrack_total_length(animal_name)
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []

    for trajectory_type in TRAJECTORY_TYPES:
        track_graph, edge_order = _get_track_graph_for_trajectory(
            animal_name,
            trajectory_type,
        )
        position_df = tl.get_linearized_position(
            position=epoch_position,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=0,
        )
        offset = 0 if trajectory_type in {"center_to_left", "right_to_center"} else 1
        task_progression = nap.Tsd(
            t=epoch_timestamps,
            d=np.asarray(position_df["linear_position"], dtype=float)
            / total_length_per_trajectory
            + offset,
            time_support=trajectory_intervals[trajectory_type],
            time_units="s",
        )
        time_chunks.append(np.asarray(task_progression.t, dtype=float))
        value_chunks.append(np.asarray(task_progression.d, dtype=float))

    if not time_chunks:
        return nap.Tsd(t=np.array([], dtype=float), d=np.array([], dtype=float), time_units="s")

    all_times = np.concatenate(time_chunks)
    all_values = np.concatenate(value_chunks)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=movement_interval,
        time_units="s",
    )


def build_task_progression_bins(
    animal_name: str,
    place_bin_size_cm: float = DEFAULT_PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    """Return the normalized task-progression bin edges."""
    task_progression_bin_size = place_bin_size_cm / get_wtrack_total_length(animal_name)
    return np.arange(0, 1 + task_progression_bin_size, task_progression_bin_size)


def build_combined_task_progression_bins(
    animal_name: str,
    place_bin_size_cm: float = DEFAULT_PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    """Return the bin edges for the combined left/right task-progression coordinate."""
    task_progression_bin_size = place_bin_size_cm / get_wtrack_total_length(animal_name)
    return np.arange(0, 2 + task_progression_bin_size, task_progression_bin_size)


def build_linear_position_bins(
    animal_name: str,
    place_bin_size_cm: float = DEFAULT_PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    """Return the concatenated linear-position bin edges across all four trajectories."""
    total_length_per_trajectory = get_wtrack_total_length(animal_name)
    return np.arange(
        0,
        total_length_per_trajectory * len(TRAJECTORY_TYPES) + place_bin_size_cm,
        place_bin_size_cm,
    )


def prepare_task_progression_session(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    regions: tuple[str, ...] = REGIONS,
    selected_run_epochs: list[str] | None = None,
    position_source: str = "auto",
    clean_dlc_input_dirname: str = DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    clean_dlc_input_name: str = DEFAULT_CLEAN_DLC_POSITION_NAME,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
) -> dict[str, Any]:
    """Load one session and build shared task-progression preprocessing outputs."""
    epoch_source = ""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, epoch_source = load_epoch_tags(analysis_path)
    available_run_epochs = get_run_epochs(epoch_tags)
    if selected_run_epochs is None:
        run_epochs = available_run_epochs
    else:
        missing_run_epochs = [
            epoch for epoch in selected_run_epochs if epoch not in available_run_epochs
        ]
        if missing_run_epochs:
            raise ValueError(
                "Requested run epochs were not found in the saved session epochs. "
                f"Available run epochs: {available_run_epochs!r}; "
                f"requested: {selected_run_epochs!r}; missing: {missing_run_epochs!r}"
            )
        run_epochs = list(dict.fromkeys(selected_run_epochs))
    position_epoch_tags, timestamps_position, position_timestamp_source = load_position_timestamps(
        analysis_path
    )
    if position_epoch_tags != epoch_tags:
        raise ValueError(
            "Saved position timestamp epochs do not match saved ephys epochs. "
            f"Ephys epochs: {epoch_tags!r}; position epochs: {position_epoch_tags!r}"
        )

    if position_source not in POSITION_SOURCE_CHOICES:
        raise ValueError(
            f"Unknown position_source {position_source!r}. "
            f"Expected one of {POSITION_SOURCE_CHOICES!r}."
        )

    loaded_position_by_epoch, position_source_path = load_position_data_with_precedence(
        analysis_path,
        position_source=position_source,
        clean_dlc_input_dirname=clean_dlc_input_dirname,
        clean_dlc_input_name=clean_dlc_input_name,
        validate_timestamps=True,
    )
    loaded_body_position_by_epoch, body_position_source_path = (
        load_body_position_data_with_precedence(
            analysis_path,
            clean_dlc_input_dirname=clean_dlc_input_dirname,
            clean_dlc_input_name=clean_dlc_input_name,
            validate_timestamps=True,
        )
    )
    missing_position_epochs = [epoch for epoch in run_epochs if epoch not in loaded_position_by_epoch]
    if missing_position_epochs:
        raise ValueError(
            "Selected run epochs are missing loaded position data. "
            f"Requested run epochs: {run_epochs!r}; "
            f"missing from {position_source_path}: {missing_position_epochs!r}"
        )
    missing_body_position_epochs = [
        epoch for epoch in run_epochs if epoch not in loaded_body_position_by_epoch
    ]
    if missing_body_position_epochs:
        raise ValueError(
            "Selected run epochs are missing body position data. "
            f"Requested run epochs: {run_epochs!r}; "
            f"missing from {body_position_source_path}: {missing_body_position_epochs!r}"
        )
    position_by_epoch = {epoch: loaded_position_by_epoch[epoch] for epoch in run_epochs}
    body_position_by_epoch = {
        epoch: loaded_body_position_by_epoch[epoch] for epoch in run_epochs
    }
    timestamps_ephys_all, ephys_all_source = load_ephys_timestamps_all(analysis_path)
    trajectory_intervals, trajectory_source = load_trajectory_intervals(analysis_path, run_epochs)
    spikes_by_region = load_spikes_by_region(analysis_path, timestamps_ephys_all, regions=regions)

    all_epoch_by_run: dict[str, Any] = {}
    speed_by_run: dict[str, Any] = {}
    movement_by_run: dict[str, Any] = {}
    task_progression_by_trajectory: dict[str, dict[str, Any]] = {}
    linear_position_by_run: dict[str, Any] = {}
    task_progression_by_run: dict[str, Any] = {}
    for epoch in run_epochs:
        all_epoch_by_run[epoch] = build_epoch_interval(
            timestamps_position[epoch],
            position_offset=position_offset,
        )
        speed_by_run[epoch] = build_speed_tsd(
            position_by_epoch[epoch],
            timestamps_position[epoch],
            position_offset=position_offset,
        )
        movement_by_run[epoch] = build_movement_interval(
            speed_by_run[epoch],
            speed_threshold_cm_s=speed_threshold_cm_s,
        )
        task_progression_by_trajectory[epoch] = build_task_progression_by_trajectory(
            animal_name,
            position_by_epoch[epoch],
            timestamps_position[epoch],
            trajectory_intervals[epoch],
            position_offset=position_offset,
        )
        linear_position_by_run[epoch] = build_linear_position(
            animal_name,
            position_by_epoch[epoch],
            timestamps_position[epoch],
            trajectory_intervals[epoch],
            movement_by_run[epoch],
            position_offset=position_offset,
        )
        task_progression_by_run[epoch] = build_task_progression(
            animal_name,
            position_by_epoch[epoch],
            timestamps_position[epoch],
            trajectory_intervals[epoch],
            movement_by_run[epoch],
            position_offset=position_offset,
        )

    return {
        "analysis_path": analysis_path,
        "epoch_tags": epoch_tags,
        "available_run_epochs": available_run_epochs,
        "run_epochs": run_epochs,
        "timestamps_position": timestamps_position,
        "position_by_epoch": position_by_epoch,
        "body_position_by_epoch": body_position_by_epoch,
        "timestamps_ephys_all": timestamps_ephys_all,
        "spikes_by_region": spikes_by_region,
        "trajectory_intervals": trajectory_intervals,
        "all_epoch_by_run": all_epoch_by_run,
        "speed_by_run": speed_by_run,
        "movement_by_run": movement_by_run,
        "task_progression_by_trajectory": task_progression_by_trajectory,
        "linear_position_by_run": linear_position_by_run,
        "task_progression_by_run": task_progression_by_run,
        "sources": {
            "epoch_tags": epoch_source,
            "timestamps_position": position_timestamp_source,
            "timestamps_ephys_all": ephys_all_source,
            "trajectory_intervals": trajectory_source,
            "position": position_source_path,
            "body_position": body_position_source_path,
            "sorting": "spikeinterface",
            "track_geometry": animal_name,
        },
    }


def compute_trajectory_task_progression_tuning_curves(
    spikes: "nap.TsGroup",
    task_progression_by_trajectory: dict[str, "nap.Tsd"],
    movement_interval: "nap.IntervalSet",
    bins: np.ndarray,
) -> dict[str, "xr.DataArray"]:
    """Compute trajectory-specific task-progression tuning curves for one region and epoch."""
    import pynapple as nap

    tuning_curves: dict[str, xr.DataArray] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        tuning_curves[trajectory_type] = nap.compute_tuning_curves(
            data=spikes,
            features=task_progression_by_trajectory[trajectory_type],
            bins=[bins],
            epochs=movement_interval,
            feature_names=["tp"],
        )
    return tuning_curves
