from __future__ import annotations

"""Shared helpers for signal-dimensionality analyses."""

from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    build_movement_interval,
    build_speed_tsd,
    compute_movement_firing_rates,
    get_analysis_path,
    get_run_epochs,
    load_ephys_timestamps_all,
    load_epoch_tags,
    load_position_data_with_precedence,
    load_position_timestamps,
    load_spikes_by_region,
    load_trajectory_time_bounds,
)
from v1ca1.helper.wtrack import (
    get_wtrack_branch_graph,
    get_wtrack_total_length,
)


DEFAULT_DATA_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_REGIONS = ("v1", "ca1")
DEFAULT_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_POSITION_OFFSET = 10
DEFAULT_RUN_EPOCH_LIMIT = 4
DEFAULT_BIN_SIZE_CM = 4.0
DEFAULT_N_GROUPS = 4
DEFAULT_MIN_OCCUPANCY_S = 0.01
DEFAULT_RANDOM_SEED = 47

# This is the signal-dimensionality condition-axis order. It is intentionally
# explicit because `v1ca1.helper.session.TRAJECTORY_TYPES` uses a different
# branch-paired order.
SIGNAL_DIM_TRAJECTORY_TYPES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
TRAJECTORY_TYPES = SIGNAL_DIM_TRAJECTORY_TYPES


def get_signal_dim_output_dir(analysis_path: Path, script_name: str) -> Path:
    """Return the standardized data/output directory for one signal-dim script."""
    return analysis_path / "signal_dim" / str(script_name)


def get_signal_dim_figure_dir(analysis_path: Path, script_name: str) -> Path:
    """Return the standardized figure directory for one signal-dim script."""
    return analysis_path / "figs" / "signal_dim" / str(script_name)


def _get_default_run_epochs(run_epochs: list[str]) -> list[str]:
    """Return the legacy MEME default subset of run epochs."""
    return list(run_epochs[:DEFAULT_RUN_EPOCH_LIMIT])


def select_run_epochs(
    run_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return selected run epochs, defaulting to the legacy first-four subset."""
    if requested_epochs is None:
        return _get_default_run_epochs(run_epochs)

    missing_run_epochs = [epoch for epoch in requested_epochs if epoch not in run_epochs]
    if missing_run_epochs:
        raise ValueError(
            "Selected run epochs were not found among available run epochs "
            f"{run_epochs!r}: {missing_run_epochs!r}"
        )
    return list(requested_epochs)


def get_light_and_dark_epochs(
    run_epochs: list[str],
    light_epoch: str | None,
    dark_epoch: str,
) -> tuple[list[str], str]:
    """Return validated light and dark epoch selections for one session."""
    if not run_epochs:
        raise ValueError("No run epochs were selected.")

    if dark_epoch not in run_epochs:
        raise ValueError(
            f"Requested dark epoch was not found in run epochs {run_epochs!r}: "
            f"{dark_epoch!r}"
        )

    if light_epoch is not None:
        if light_epoch not in run_epochs:
            raise ValueError(
                f"Requested light epoch was not found in run epochs {run_epochs!r}: "
                f"{light_epoch!r}"
            )
        if light_epoch == dark_epoch:
            raise ValueError("Light and dark epochs must be different.")
        selected_light_epochs = [light_epoch]
    else:
        selected_light_epochs = [epoch for epoch in run_epochs if epoch != dark_epoch]
        if not selected_light_epochs:
            raise ValueError(
                "No light epochs remain after excluding the requested dark epoch."
            )

    return selected_light_epochs, dark_epoch


def load_signal_dim_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    regions: list[str],
    position_offset: int,
    speed_threshold_cm_s: float,
) -> dict[str, Any]:
    """Load one session and derive reusable state for signal-dim analyses."""
    analysis_path = get_analysis_path(animal_name, date, data_root)
    epoch_list, _ = load_epoch_tags(analysis_path)
    position_epoch_tags, timestamps_position_dict, _ = load_position_timestamps(analysis_path)
    if position_epoch_tags != epoch_list:
        raise ValueError(
            "Saved position timestamp epochs do not match saved ephys epochs. "
            f"Ephys epochs: {epoch_list!r}; position epochs: {position_epoch_tags!r}"
        )
    timestamps_ephys_all_ptp, _ = load_ephys_timestamps_all(analysis_path)
    available_run_epochs = get_run_epochs(epoch_list)
    position_dict_all, position_source = load_position_data_with_precedence(
        analysis_path,
        position_source="clean_dlc_head",
        clean_dlc_input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        clean_dlc_input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    run_epochs: list[str] = []
    skipped_run_epochs: list[dict[str, str]] = []
    for epoch in available_run_epochs:
        if epoch not in position_dict_all:
            skipped_run_epochs.append(
                {"epoch": epoch, "reason": f"head position missing from {position_source}"}
            )
            continue
        position_xy = np.asarray(position_dict_all[epoch], dtype=float)
        if not np.isfinite(position_xy).any():
            skipped_run_epochs.append(
                {"epoch": epoch, "reason": f"head position is all NaN in {position_source}"}
            )
            continue
        run_epochs.append(epoch)

    if not run_epochs:
        raise ValueError(
            "No run epochs have usable head position in the combined cleaned DLC "
            f"position parquet {position_source}."
        )

    position_dict = {epoch: position_dict_all[epoch] for epoch in run_epochs}
    trajectory_times, _trajectory_source = load_trajectory_time_bounds(
        analysis_path,
        run_epochs,
    )
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all_ptp,
        regions=tuple(regions),
    )
    speed_by_epoch = {
        epoch: build_speed_tsd(
            position_dict[epoch],
            timestamps_position_dict[epoch],
            position_offset=position_offset,
        )
        for epoch in run_epochs
    }
    movement_by_epoch = {
        epoch: build_movement_interval(
            speed_by_epoch[epoch],
            speed_threshold_cm_s=speed_threshold_cm_s,
        )
        for epoch in run_epochs
    }
    movement_firing_rates_by_region = compute_movement_firing_rates(
        spikes_by_region,
        movement_by_epoch,
        run_epochs,
    )
    track_graphs_by_side: dict[str, Any] = {}
    edge_orders_by_side: dict[str, list[tuple[int, int]]] = {}
    for side in ("left", "right"):
        track_graph, edge_order = get_wtrack_branch_graph(
            animal_name,
            branch_side=side,
            direction="from_center",
        )
        track_graphs_by_side[side] = track_graph
        edge_orders_by_side[side] = edge_order

    return {
        "analysis_path": analysis_path,
        "epoch_list": epoch_list,
        "run_epochs": run_epochs,
        "skipped_run_epochs": skipped_run_epochs,
        "position_source": position_source,
        "timestamps_position_dict": timestamps_position_dict,
        "position_dict": position_dict,
        "trajectory_times": trajectory_times,
        "spikes_by_region": spikes_by_region,
        "movement_by_epoch": movement_by_epoch,
        "movement_firing_rates_by_region": movement_firing_rates_by_region,
        "track_total_length": get_wtrack_total_length(animal_name),
        "track_graphs_by_side": track_graphs_by_side,
        "edge_orders_by_side": edge_orders_by_side,
        "linear_edge_spacing": 0,
        "position_offset": position_offset,
    }


def sample_lap_indices(
    n_available: int,
    *,
    lap_fraction: float,
    n_groups: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample laps without replacement while retaining enough groups."""
    if not (0.0 < lap_fraction <= 1.0):
        raise ValueError("lap_fraction must be in (0, 1].")
    if n_available < n_groups:
        raise ValueError(
            f"Need at least n_groups laps. Got n_available={n_available}, "
            f"n_groups={n_groups}."
        )

    n_select = int(np.ceil(float(lap_fraction) * n_available))
    n_select = min(n_available, max(n_groups, n_select))
    return np.sort(rng.choice(n_available, size=n_select, replace=False))


def split_lap_indices_into_groups(
    lap_indices: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Randomly split lap indices into disjoint groups as evenly as possible."""
    lap_indices = np.asarray(lap_indices, dtype=np.int64)
    if lap_indices.ndim != 1:
        raise ValueError("lap_indices must be a 1D array.")
    if lap_indices.size < n_groups:
        raise ValueError(
            f"Need at least n_groups sampled laps. Got n={lap_indices.size}, "
            f"n_groups={n_groups}."
        )

    groups = np.array_split(rng.permutation(lap_indices), n_groups)
    groups = [g.astype(np.int64, copy=False) for g in groups if len(g) > 0]
    if len(groups) < 2:
        raise ValueError(
            "Grouping produced <2 non-empty groups; increase laps or reduce n_groups."
        )
    return groups


def occupancy_mask_1d(
    linpos_tsd: Any,
    epochs: Any,
    bin_edges: np.ndarray,
    *,
    min_occupancy_s: float,
) -> np.ndarray:
    """Return bins with at least `min_occupancy_s` of sampled occupancy."""
    if min_occupancy_s <= 0:
        raise ValueError("min_occupancy_s must be > 0.")

    pos = linpos_tsd.restrict(epochs)
    t = np.asarray(pos.t)
    x = np.asarray(pos.d)

    if x.size < 2:
        return np.zeros(len(bin_edges) - 1, dtype=bool)

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        dt = float((t[-1] - t[0]) / max(len(t) - 1, 1))

    counts, _ = np.histogram(x, bins=bin_edges)
    occ_s = counts.astype(np.float64) * dt
    return occ_s >= float(min_occupancy_s)
