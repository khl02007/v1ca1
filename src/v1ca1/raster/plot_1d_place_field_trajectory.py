from __future__ import annotations

"""Plot branch-aligned 1D trajectory rasters with overlaid place fields.

This workflow replaces the old classifier-backed legacy script with a modern
session loader built on shared helpers and pynapple tuning curves. For each
unit in each requested region, it plots one panel per run epoch and trajectory
using the legacy two-column layout:

- top block: `center_to_left`, `center_to_right`
- bottom block: `left_to_center`, `right_to_center`

Within each panel, spike positions are shown as a trial-by-trial raster using
all spikes inside each detected trajectory interval, while the overlaid place
field is computed from movement-restricted samples only.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    load_ephys_timestamps_all,
    load_epoch_tags,
    load_position_data_with_precedence,
    load_position_timestamps,
    load_spikes_by_region,
    load_trajectory_intervals,
    build_movement_interval,
    build_speed_tsd,
    get_analysis_path,
    get_run_epochs,
)
from v1ca1.helper.wtrack import (
    get_wtrack_branch_graph,
    get_wtrack_geometry,
    get_wtrack_total_length,
)


DEFAULT_SIGMA_BINS = 1.0
DEFAULT_SMOOTHING_MODE = "interpolate_short_gaps"
DEFAULT_MAX_INTERPOLATED_GAP_BINS = 2
DEFAULT_FIG_WIDTH = 12.0
DEFAULT_FIG_HEIGHT = 10.0
DEFAULT_SPACER_HEIGHT_RATIO = 0.5
DEFAULT_CLEAN_DLC_INPUT_DIRNAME = "dlc_position_cleaned"
DEFAULT_CLEAN_DLC_INPUT_NAME = "position.parquet"
TRAJECTORY_TYPES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)


def smooth_values_nan_aware(
    values: np.ndarray,
    sigma_bins: float,
    *,
    axis: int = -1,
    eps: float = 1e-12,
    mode: str = "nearest",
) -> np.ndarray:
    """Smooth one array without turning unsupported bins into zeros."""
    values = np.asarray(values, dtype=float)
    if sigma_bins <= 0:
        return values.copy()

    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)
    numerator = gaussian_filter1d(filled, sigma=sigma_bins, axis=axis, mode=mode)
    denominator = gaussian_filter1d(
        mask.astype(float),
        sigma=sigma_bins,
        axis=axis,
        mode=mode,
    )
    smoothed = numerator / np.maximum(denominator, eps)
    return np.where(denominator > eps, smoothed, np.nan)


def interpolate_short_nans_1d(
    values: np.ndarray,
    *,
    max_gap_bins: int,
) -> np.ndarray:
    """Linearly interpolate only short interior NaN runs in one 1D array."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if max_gap_bins < 0:
        raise ValueError("--max-interpolated-gap-bins must be non-negative.")
    if values.size == 0:
        return values.copy()

    output = values.copy()
    finite = np.isfinite(output)
    if not np.any(finite) or np.all(finite):
        return output

    n_values = output.size
    index = 0
    while index < n_values:
        if np.isfinite(output[index]):
            index += 1
            continue

        gap_start = index
        while index < n_values and not np.isfinite(output[index]):
            index += 1
        gap_end = index
        gap_length = gap_end - gap_start

        left_index = gap_start - 1
        right_index = gap_end
        has_left = left_index >= 0 and np.isfinite(output[left_index])
        has_right = right_index < n_values and np.isfinite(output[right_index])
        if not has_left or not has_right or gap_length > max_gap_bins:
            continue

        interpolate_indices = np.arange(gap_start, gap_end, dtype=float)
        output[gap_start:gap_end] = np.interp(
            interpolate_indices,
            np.array([left_index, right_index], dtype=float),
            np.array([output[left_index], output[right_index]], dtype=float),
        )

    return output


def interpolate_short_nans(
    values: np.ndarray,
    *,
    max_gap_bins: int,
    axis: int = -1,
) -> np.ndarray:
    """Interpolate short interior NaN runs along one array axis."""
    values = np.asarray(values, dtype=float)
    moved = np.moveaxis(values, axis, -1)
    output = moved.copy()
    leading_shape = output.shape[:-1]
    for index in np.ndindex(leading_shape):
        output[index] = interpolate_short_nans_1d(
            output[index],
            max_gap_bins=max_gap_bins,
        )
    return np.moveaxis(output, -1, axis)


def smooth_tuning_curve_nan_aware(
    tuning_curve: Any,
    *,
    pos_dim: str = "linpos",
    sigma_bins: float,
) -> Any:
    """Return one nan-aware smoothed tuning curve."""
    axis = tuning_curve.get_axis_num(pos_dim)
    smoothed = smooth_values_nan_aware(
        np.asarray(tuning_curve.values, dtype=float),
        sigma_bins=sigma_bins,
        axis=axis,
    )
    return tuning_curve.copy(data=smoothed)


def smooth_tuning_curve_interpolate_short_gaps(
    tuning_curve: Any,
    *,
    pos_dim: str = "linpos",
    sigma_bins: float,
    max_gap_bins: int,
) -> Any:
    """Interpolate short interior gaps before nan-aware Gaussian smoothing."""
    axis = tuning_curve.get_axis_num(pos_dim)
    interpolated = interpolate_short_nans(
        np.asarray(tuning_curve.values, dtype=float),
        max_gap_bins=max_gap_bins,
        axis=axis,
    )
    smoothed = smooth_values_nan_aware(
        interpolated,
        sigma_bins=sigma_bins,
        axis=axis,
    )
    return tuning_curve.copy(data=smoothed)


def build_place_bin_edges(
    animal_name: str,
    place_bin_size_cm: float,
) -> np.ndarray:
    """Return shared place-field bin edges for one branch-aligned trajectory."""
    if place_bin_size_cm <= 0:
        raise ValueError("--place-bin-size-cm must be positive.")
    total_length = get_wtrack_total_length(animal_name)
    return np.arange(0.0, total_length + place_bin_size_cm, place_bin_size_cm)


def get_guide_line_positions(animal_name: str) -> np.ndarray:
    """Return the legacy guide-line positions for one W-track trajectory axis."""
    geometry = get_wtrack_geometry(animal_name)
    diagonal_segment_length = float(np.hypot(geometry["dx"], geometry["dy"]))
    return np.array(
        [
            geometry["long_segment_length"] + diagonal_segment_length * 0.5,
            (
                geometry["long_segment_length"]
                + diagonal_segment_length * 1.5
                + geometry["short_segment_length"]
            ),
        ],
        dtype=float,
    )


def get_branch_aligned_track_graph(
    animal_name: str,
    trajectory_type: str,
) -> tuple[Any, list[tuple[int, int]]]:
    """Return the branch-aligned graph used by the legacy 1D raster figure."""
    branch_side = "left" if "left" in trajectory_type else "right"
    return get_wtrack_branch_graph(
        animal_name=animal_name,
        branch_side=branch_side,
        direction="from_center",
    )


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start and end arrays from one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "IntervalSet start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts, ends


def _intervalset_is_empty(intervals: Any) -> bool:
    """Return whether one IntervalSet contains no intervals."""
    starts, _ends = _extract_interval_bounds(intervals)
    return starts.size == 0


def build_linear_position_by_trajectory(
    animal_name: str,
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, Any],
    *,
    position_offset: int,
) -> dict[str, Any]:
    """Build branch-aligned linear position Tsds for the four trajectories."""
    import pynapple as nap
    import track_linearization as tl

    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if position.shape[0] <= position_offset or timestamps_position.size <= position_offset:
        raise ValueError(
            "Position offset removes all samples for this epoch. "
            f"position count: {position.shape[0]}, timestamp count: {timestamps_position.size}, "
            f"position_offset: {position_offset}"
        )

    epoch_position = np.asarray(position[position_offset:], dtype=float)
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    linear_position_by_trajectory: dict[str, Any] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        track_graph, edge_order = get_branch_aligned_track_graph(
            animal_name,
            trajectory_type,
        )
        position_df = tl.get_linearized_position(
            position=epoch_position,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=0,
        )
        linear_position_by_trajectory[trajectory_type] = nap.Tsd(
            t=epoch_timestamps,
            d=np.asarray(position_df["linear_position"], dtype=float),
            time_support=trajectory_intervals[trajectory_type],
            time_units="s",
        )
    return linear_position_by_trajectory


def compute_place_tuning_curve(
    spikes: Any,
    linear_position: Any,
    epochs: Any,
    *,
    bin_edges: np.ndarray,
    sigma_bins: float,
    smoothing_mode: str,
    max_interpolated_gap_bins: int,
) -> Any | None:
    """Compute one smoothed place tuning curve or return `None` for empty epochs."""
    import pynapple as nap

    if _intervalset_is_empty(epochs):
        return None

    tuning_curve = nap.compute_tuning_curves(
        data=spikes,
        features=linear_position,
        epochs=epochs,
        bins=[np.asarray(bin_edges, dtype=float)],
        feature_names=["linpos"],
    )
    if sigma_bins > 0:
        if smoothing_mode == "interpolate_short_gaps":
            tuning_curve = smooth_tuning_curve_interpolate_short_gaps(
                tuning_curve,
                pos_dim="linpos",
                sigma_bins=sigma_bins,
                max_gap_bins=max_interpolated_gap_bins,
            )
        elif smoothing_mode == "nan_aware":
            tuning_curve = smooth_tuning_curve_nan_aware(
                tuning_curve,
                pos_dim="linpos",
                sigma_bins=sigma_bins,
            )
        else:
            raise ValueError(
                f"Unknown smoothing_mode {smoothing_mode!r}. "
                "Expected one of {'interpolate_short_gaps', 'nan_aware'}."
            )
    return tuning_curve


def _extract_tuning_curve_arrays(tuning_curve: Any) -> tuple[np.ndarray, np.ndarray, str]:
    """Return unit labels, unit-by-position values, and the position dimension name."""
    if len(tuning_curve.dims) != 2:
        raise ValueError(
            "Expected a 2D tuning curve with unit and position dimensions. "
            f"Got dims {tuning_curve.dims!r}."
        )
    unit_dim, pos_dim = tuning_curve.dims
    values = np.asarray(
        tuning_curve.transpose(unit_dim, pos_dim).values,
        dtype=float,
    )
    units = np.asarray(tuning_curve.coords[unit_dim].values)
    return units, values, pos_dim


def extract_unit_place_field(
    tuning_curve: Any | None,
    unit_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return position bins and one unit's place field from a tuning curve."""
    if tuning_curve is None:
        return None

    units, values, pos_dim = _extract_tuning_curve_arrays(tuning_curve)
    matching_indices = np.flatnonzero(units == unit_id)
    if matching_indices.size == 0:
        return None

    field = np.asarray(values[int(matching_indices[0])], dtype=float)
    position = np.asarray(tuning_curve.coords[pos_dim].values, dtype=float)
    return position, field


def make_linear_position_interpolator(linear_position: Any) -> Any:
    """Return a linear interpolator for one branch-aligned linear position Tsd."""
    times = np.asarray(linear_position.t, dtype=float)
    values = np.asarray(linear_position.d, dtype=float)
    if times.size < 2:
        return None
    return interp1d(
        times,
        values,
        kind="linear",
        bounds_error=False,
        assume_sorted=True,
    )


def compute_trial_spike_positions(
    spike_times_s: np.ndarray,
    trajectory_intervals: Any,
    linear_position_interpolator: Any,
) -> list[np.ndarray]:
    """Return branch-aligned spike positions for each detected trajectory interval."""
    starts, ends = _extract_interval_bounds(trajectory_intervals)
    trial_positions: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        trial_spike_times = spike_times_s[
            (spike_times_s > float(start)) & (spike_times_s <= float(end))
        ]
        if trial_spike_times.size == 0 or linear_position_interpolator is None:
            trial_positions.append(np.array([], dtype=float))
            continue
        positions = np.asarray(linear_position_interpolator(trial_spike_times), dtype=float)
        trial_positions.append(positions[np.isfinite(positions)])
    return trial_positions


def get_panel_location(
    trajectory_type: str,
    n_run_epochs: int,
) -> tuple[int, int]:
    """Return the `(row_offset, column)` location for one trajectory block."""
    if trajectory_type == "center_to_left":
        return 0, 0
    if trajectory_type == "center_to_right":
        return 0, 1
    if trajectory_type == "left_to_center":
        return n_run_epochs + 1, 0
    if trajectory_type == "right_to_center":
        return n_run_epochs + 1, 1
    raise ValueError(f"Unknown trajectory type: {trajectory_type!r}")


def get_figure_height(n_run_epochs: int) -> float:
    """Scale figure height while preserving the legacy 9-row proportions."""
    legacy_nrows = 9
    current_nrows = n_run_epochs * 2 + 1
    return DEFAULT_FIG_HEIGHT * (current_nrows / legacy_nrows)


def get_available_trajectory_epochs(
    analysis_path: Path,
    run_epochs: list[str],
) -> list[str]:
    """Return the run epochs that actually have saved trajectory intervals."""
    parquet_path = analysis_path / "trajectory_times.parquet"
    if parquet_path.exists():
        import pandas as pd

        trajectory_table = pd.read_parquet(parquet_path)
        if "epoch" not in trajectory_table.columns:
            raise ValueError(
                "trajectory_times.parquet is missing the required 'epoch' column."
            )
        available_epochs = set(trajectory_table["epoch"].astype(str).unique().tolist())
        filtered_epochs = [epoch for epoch in run_epochs if epoch in available_epochs]
        if filtered_epochs:
            return filtered_epochs

    pickle_path = analysis_path / "trajectory_times.pkl"
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Could not find trajectory_times.parquet or trajectory_times.pkl under {analysis_path}."
        )

    with open(pickle_path, "rb") as file:
        trajectory_times = pickle.load(file)
    available_epochs = {str(epoch) for epoch in trajectory_times.keys()}
    filtered_epochs = [epoch for epoch in run_epochs if epoch in available_epochs]
    if not filtered_epochs:
        raise ValueError(
            "No run epochs have saved trajectory intervals. "
            f"Run epochs: {run_epochs!r}; available trajectory epochs: {sorted(available_epochs)!r}."
        )
    return filtered_epochs


def get_available_position_epochs(
    position_by_epoch: dict[str, np.ndarray],
    run_epochs: list[str],
    *,
    position_source: str,
) -> list[str]:
    """Return the run epochs that actually have loaded position samples."""
    available_epochs = {str(epoch) for epoch in position_by_epoch.keys()}
    filtered_epochs = [epoch for epoch in run_epochs if epoch in available_epochs]
    if not filtered_epochs:
        raise ValueError(
            "No run epochs have loaded position data. "
            f"Run epochs: {run_epochs!r}; available position epochs from {position_source}: "
            f"{sorted(available_epochs)!r}."
        )
    return filtered_epochs


def prepare_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    regions: tuple[str, ...],
    position_offset: int,
    speed_threshold_cm_s: float,
) -> dict[str, Any]:
    """Load the session inputs required by the 1D trajectory raster workflow."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    run_epochs = get_run_epochs(epoch_tags)
    run_epochs = get_available_trajectory_epochs(analysis_path, run_epochs)
    position_epoch_tags, timestamps_position, _timestamp_source = load_position_timestamps(
        analysis_path
    )
    if position_epoch_tags != epoch_tags:
        raise ValueError(
            "Saved position timestamp epochs do not match saved ephys epochs. "
            f"Ephys epochs: {epoch_tags!r}; position epochs: {position_epoch_tags!r}"
        )

    position_by_epoch, position_source = load_position_data_with_precedence(
        analysis_path,
        position_source="auto",
        clean_dlc_input_dirname=DEFAULT_CLEAN_DLC_INPUT_DIRNAME,
        clean_dlc_input_name=DEFAULT_CLEAN_DLC_INPUT_NAME,
        validate_timestamps=True,
    )
    run_epochs = get_available_position_epochs(
        position_by_epoch,
        run_epochs,
        position_source=position_source,
    )
    timestamps_ephys_all, _ephys_source = load_ephys_timestamps_all(analysis_path)
    trajectory_intervals, _trajectory_source = load_trajectory_intervals(
        analysis_path,
        run_epochs,
    )
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all,
        regions=regions,
    )

    movement_by_run: dict[str, Any] = {}
    linear_position_by_run: dict[str, dict[str, Any]] = {}
    for epoch in run_epochs:
        speed_tsd = build_speed_tsd(
            position_by_epoch[epoch],
            timestamps_position[epoch],
            position_offset=position_offset,
        )
        movement_by_run[epoch] = build_movement_interval(
            speed_tsd,
            speed_threshold_cm_s=speed_threshold_cm_s,
        )
        linear_position_by_run[epoch] = build_linear_position_by_trajectory(
            animal_name,
            position_by_epoch[epoch],
            timestamps_position[epoch],
            trajectory_intervals[epoch],
            position_offset=position_offset,
        )

    return {
        "analysis_path": analysis_path,
        "run_epochs": run_epochs,
        "timestamps_ephys_all": timestamps_ephys_all,
        "trajectory_intervals": trajectory_intervals,
        "movement_by_run": movement_by_run,
        "linear_position_by_run": linear_position_by_run,
        "spikes_by_region": spikes_by_region,
        "position_source": position_source,
    }


def compute_region_place_fields(
    *,
    spikes: Any,
    linear_position_by_run: dict[str, dict[str, Any]],
    trajectory_intervals: dict[str, dict[str, Any]],
    movement_by_run: dict[str, Any],
    run_epochs: list[str],
    bin_edges: np.ndarray,
    sigma_bins: float,
    smoothing_mode: str,
    max_interpolated_gap_bins: int,
) -> dict[str, dict[str, Any | None]]:
    """Compute one place-field tuning curve per epoch and trajectory."""
    tuning_curves: dict[str, dict[str, Any | None]] = {}
    for epoch in run_epochs:
        tuning_curves[epoch] = {}
        for trajectory_type in TRAJECTORY_TYPES:
            trajectory_epochs = trajectory_intervals[epoch][trajectory_type].intersect(
                movement_by_run[epoch]
            )
            tuning_curves[epoch][trajectory_type] = compute_place_tuning_curve(
                spikes,
                linear_position_by_run[epoch][trajectory_type],
                trajectory_epochs,
                bin_edges=bin_edges,
                sigma_bins=sigma_bins,
                smoothing_mode=smoothing_mode,
                max_interpolated_gap_bins=max_interpolated_gap_bins,
            )
    return tuning_curves


def plot_unit_figure(
    *,
    animal_name: str,
    date: str,
    region: str,
    unit_id: int,
    run_epochs: list[str],
    spike_times_s: np.ndarray,
    linear_position_by_run: dict[str, dict[str, Any]],
    trajectory_intervals: dict[str, dict[str, Any]],
    place_fields_by_run: dict[str, dict[str, Any | None]],
    output_dir: Path,
    show: bool,
) -> Path:
    """Plot and save one unit's branch-aligned trajectory raster figure."""
    import matplotlib.pyplot as plt

    n_run_epochs = len(run_epochs)
    nrows = n_run_epochs * 2 + 1
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(DEFAULT_FIG_WIDTH, get_figure_height(n_run_epochs)),
        squeeze=False,
        gridspec_kw={
            "height_ratios": ([1.0] * n_run_epochs)
            + [DEFAULT_SPACER_HEIGHT_RATIO]
            + ([1.0] * n_run_epochs),
        },
    )

    total_length = get_wtrack_total_length(animal_name)
    guide_positions = get_guide_line_positions(animal_name)
    tick_positions = [0.0, *guide_positions.tolist(), total_length]
    tick_labels = [
        "0",
        f"{guide_positions[0]:.1f}",
        f"{guide_positions[1]:.1f}",
        f"{total_length:.1f}",
    ]

    pf_axes: list[Any] = []
    pf_max_vals: list[float] = []

    for trajectory_type in TRAJECTORY_TYPES:
        row_offset, col_index = get_panel_location(trajectory_type, n_run_epochs)
        for epoch_idx, epoch in enumerate(run_epochs):
            axis = axes[row_offset + epoch_idx, col_index]
            linear_position = linear_position_by_run[epoch][trajectory_type]
            linear_position_interpolator = make_linear_position_interpolator(linear_position)
            trial_positions = compute_trial_spike_positions(
                spike_times_s,
                trajectory_intervals[epoch][trajectory_type],
                linear_position_interpolator,
            )

            for trial_index, positions in enumerate(trial_positions, start=1):
                if positions.size == 0:
                    continue
                axis.plot(
                    positions,
                    np.full(positions.shape, trial_index, dtype=float),
                    "|",
                    color="black",
                    markersize=1.0,
                )

            n_trials = len(trial_positions)
            axis.set_ylim(0.0, max(1, n_trials) + 1.0)
            axis.set_xlim(0.0, total_length)
            axis.set_yticklabels([])
            axis.set_xticks(tick_positions)
            if epoch_idx == n_run_epochs - 1:
                axis.set_xticklabels(tick_labels)
            else:
                axis.set_xticklabels([])

            for guide_position in guide_positions:
                axis.axvline(guide_position, color="gray", alpha=0.4)

            if col_index == 0:
                axis.set_ylabel(str(epoch))

            if epoch_idx == 0:
                axis.set_title(trajectory_type)

            place_field = extract_unit_place_field(
                place_fields_by_run[epoch][trajectory_type],
                unit_id,
            )
            if place_field is not None:
                position_bins, field = place_field
                pf_axis = axis.twinx()
                pf_axis.plot(position_bins, field, color="blue")
                pf_axes.append(pf_axis)

                finite = field[np.isfinite(field)]
                if finite.size > 0:
                    pf_max_vals.append(float(np.nanmax(finite)))

    spacer_row = n_run_epochs
    for col_index in range(2):
        axes[spacer_row, col_index].axis("off")

    pf_max = 1e-6 if not pf_max_vals else max(1e-6, np.round(float(np.nanmax(pf_max_vals)), 1))
    for pf_axis in pf_axes:
        pf_axis.set_ylim([0.0, pf_max])
        pf_axis.set_yticks([0.0, pf_max])
        pf_axis.set_yticklabels(["0", f"{pf_max:g}"])

    axes[-1, -1].set_xlabel("Position (cm)")
    axes[-1, -1].set_ylabel("Trials")

    fig.suptitle(f"{animal_name} {date} {region} {unit_id}")
    output_path = output_dir / f"{region}_{unit_id}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the 1D trajectory raster workflow."""
    parser = argparse.ArgumentParser(
        description="Plot branch-aligned 1D trajectory rasters with place fields",
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
        "--region",
        choices=REGIONS,
        help="Only plot one region. Default: plot all regions.",
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        help="Only plot one unit. Default: plot all units in the requested regions.",
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
        "--place-bin-size-cm",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=(
            "Place-field bin size in cm along each branch-aligned trajectory. "
            f"Default: {DEFAULT_PLACE_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in place-field bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--smoothing-mode",
        choices=("interpolate_short_gaps", "nan_aware"),
        default=DEFAULT_SMOOTHING_MODE,
        help=(
            "How to smooth tuning curves. "
            f"Default: {DEFAULT_SMOOTHING_MODE}."
        ),
    )
    parser.add_argument(
        "--max-interpolated-gap-bins",
        type=int,
        default=DEFAULT_MAX_INTERPOLATED_GAP_BINS,
        help=(
            "Maximum length of an interior NaN run to linearly interpolate before "
            "smoothing when --smoothing-mode=interpolate_short_gaps. "
            f"Default: {DEFAULT_MAX_INTERPOLATED_GAP_BINS}."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the 1D trajectory place-field raster workflow."""
    args = parse_arguments(argv)
    selected_regions = (args.region,) if args.region is not None else REGIONS
    session = prepare_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )

    output_dir = session["analysis_path"] / "figs" / "place_field_1d_trajectory"
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = build_place_bin_edges(
        args.animal_name,
        place_bin_size_cm=args.place_bin_size_cm,
    )

    for region in selected_regions:
        spikes = session["spikes_by_region"][region]
        region_unit_ids = list(spikes.keys())
        if args.unit_id is not None:
            if args.unit_id not in region_unit_ids:
                raise ValueError(
                    f"Requested unit {args.unit_id} was not found in region {region!r}. "
                    f"Available unit count: {len(region_unit_ids)}."
                )
            unit_ids = [args.unit_id]
        else:
            unit_ids = region_unit_ids

        place_fields_by_run = compute_region_place_fields(
            spikes=spikes,
            linear_position_by_run=session["linear_position_by_run"],
            trajectory_intervals=session["trajectory_intervals"],
            movement_by_run=session["movement_by_run"],
            run_epochs=session["run_epochs"],
            bin_edges=bin_edges,
            sigma_bins=args.sigma_bins,
            smoothing_mode=args.smoothing_mode,
            max_interpolated_gap_bins=args.max_interpolated_gap_bins,
        )

        for unit_id in unit_ids:
            spike_times_s = np.asarray(spikes[unit_id].t, dtype=float)
            plot_unit_figure(
                animal_name=args.animal_name,
                date=args.date,
                region=region,
                unit_id=unit_id,
                run_epochs=session["run_epochs"],
                spike_times_s=spike_times_s,
                linear_position_by_run=session["linear_position_by_run"],
                trajectory_intervals=session["trajectory_intervals"],
                place_fields_by_run=place_fields_by_run,
                output_dir=output_dir,
                show=args.show,
            )


if __name__ == "__main__":
    main()
