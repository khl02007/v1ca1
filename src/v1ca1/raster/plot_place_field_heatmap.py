from __future__ import annotations

"""Plot odd/even cross-trajectory place-field heatmaps for one session."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    build_movement_interval,
    build_speed_tsd,
    get_analysis_path,
    get_run_epochs,
    load_ephys_timestamps_all,
    load_epoch_tags,
    load_position_data_with_precedence,
    load_position_timestamps,
    load_spikes_by_region,
    load_trajectory_intervals,
)
from v1ca1.helper.wtrack import (
    get_wtrack_branch_graph,
    get_wtrack_geometry,
    get_wtrack_total_length,
)


DEFAULT_SIGMA_BINS = 1.5
DEFAULT_CLEAN_DLC_INPUT_DIRNAME = "dlc_position_cleaned"
DEFAULT_CLEAN_DLC_INPUT_NAME = "position.parquet"


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


def build_place_bin_edges(
    animal_name: str,
    place_bin_size_cm: float = DEFAULT_PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    """Return shared per-trajectory place-field bin edges."""
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


def split_odd_even_bounds(
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split aligned interval bounds into odd and even traversals."""
    starts = np.asarray(starts, dtype=float).ravel()
    ends = np.asarray(ends, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Interval start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts[::2], ends[::2], starts[1::2], ends[1::2]


def compute_unit_order(values: np.ndarray) -> np.ndarray:
    """Order units by the peak-bin index of one odd-trial tuning matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D tuning matrix, got shape {values.shape}.")

    n_units, n_bins = values.shape
    row_max = np.full(n_units, np.nan, dtype=float)
    finite_rows = np.isfinite(values).any(axis=1)
    if np.any(finite_rows):
        row_max[finite_rows] = np.nanmax(values[finite_rows], axis=1)

    valid_rows = np.isfinite(row_max) & (row_max > 0)
    normalized = np.full_like(values, np.nan, dtype=float)
    if np.any(valid_rows):
        normalized[valid_rows] = values[valid_rows] / row_max[valid_rows, None]

    safe = np.where(np.isfinite(normalized), normalized, -np.inf)
    peak_index = np.argmax(safe, axis=1)
    peak_index = np.where(valid_rows, peak_index, n_bins)
    return np.argsort(peak_index, kind="stable")


def align_and_normalize_panel_values(
    display_values: np.ndarray,
    display_units: np.ndarray,
    reference_units: np.ndarray,
    unit_order: np.ndarray,
) -> np.ndarray:
    """Align one display matrix to the reference units and normalize per unit."""
    display_values = np.asarray(display_values, dtype=float)
    display_units = np.asarray(display_units)
    reference_units = np.asarray(reference_units)
    unit_order = np.asarray(unit_order, dtype=int)

    if display_values.ndim != 2:
        raise ValueError(f"Expected a 2D tuning matrix, got shape {display_values.shape}.")
    if display_values.shape[0] != display_units.size:
        raise ValueError(
            "Display matrix rows must match the number of display units. "
            f"Got {display_values.shape[0]} rows and {display_units.size} units."
        )
    if unit_order.shape != (reference_units.size,):
        raise ValueError(
            "unit_order must contain one index per reference unit. "
            f"Got shape {unit_order.shape} for {reference_units.size} units."
        )

    aligned = np.full((reference_units.size, display_values.shape[1]), np.nan, dtype=float)
    index_by_unit = {
        unit: index for index, unit in enumerate(display_units.tolist())
    }
    for reference_index, unit in enumerate(reference_units.tolist()):
        display_index = index_by_unit.get(unit)
        if display_index is not None:
            aligned[reference_index] = display_values[display_index]

    sorted_values = aligned[unit_order]
    row_max = np.full(sorted_values.shape[0], np.nan, dtype=float)
    finite_rows = np.isfinite(sorted_values).any(axis=1)
    if np.any(finite_rows):
        row_max[finite_rows] = np.nanmax(sorted_values[finite_rows], axis=1)

    valid_rows = np.isfinite(row_max) & (row_max > 0)
    normalized = np.full_like(sorted_values, np.nan, dtype=float)
    if np.any(valid_rows):
        normalized[valid_rows] = sorted_values[valid_rows] / row_max[valid_rows, None]
    return normalized


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start/end arrays from one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "IntervalSet start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts, ends


def _make_interval_set(starts: np.ndarray, ends: np.ndarray) -> Any:
    """Return one pynapple IntervalSet from aligned bounds."""
    import pynapple as nap

    return nap.IntervalSet(
        start=np.asarray(starts, dtype=float),
        end=np.asarray(ends, dtype=float),
        time_units="s",
    )


def _intervalset_is_empty(intervals: Any) -> bool:
    """Return whether one IntervalSet contains no intervals."""
    starts, _ends = _extract_interval_bounds(intervals)
    return starts.size == 0


def _has_plottable_values(values: np.ndarray) -> bool:
    """Return whether one tuning matrix contains any positive finite values."""
    values = np.asarray(values, dtype=float)
    return bool(np.isfinite(values).any() and np.nanmax(values) > 0)


def _extract_tuning_curve_arrays(tuning_curve: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return unit labels and a unit-by-position value matrix from one tuning curve."""
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
    return units, values


def get_legacy_heatmap_track_graph(
    animal_name: str,
    trajectory_type: str,
) -> tuple[Any, list[tuple[int, int]]]:
    """Return the legacy branch-only linearization used by the old heatmap script."""
    branch_side = "left" if "left" in trajectory_type else "right"
    return get_wtrack_branch_graph(
        animal_name=animal_name,
        branch_side=branch_side,
        direction="from_center",
    )


def build_linear_position_by_trajectory(
    animal_name: str,
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, Any],
    *,
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> dict[str, Any]:
    """Build legacy-coordinate linear position Tsds for the four trajectories."""
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
        track_graph, edge_order = get_legacy_heatmap_track_graph(
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


def load_position_data_with_fallback(
    analysis_path: Path,
    *,
    clean_dlc_input_dirname: str = DEFAULT_CLEAN_DLC_INPUT_DIRNAME,
    clean_dlc_input_name: str = DEFAULT_CLEAN_DLC_INPUT_NAME,
) -> tuple[dict[str, np.ndarray], str]:
    """Load head XY from cleaned DLC when available, else fall back to saved pickles."""
    position_by_epoch, source = load_position_data_with_precedence(
        analysis_path,
        position_source="auto",
        clean_dlc_input_dirname=clean_dlc_input_dirname,
        clean_dlc_input_name=clean_dlc_input_name,
        validate_timestamps=True,
    )
    if source.endswith("position.pkl") or source.endswith("body_position.pkl"):
        print(f"{source} is being used for position data.")
    return position_by_epoch, source


def load_heatmap_trajectory_intervals(
    analysis_path: Path,
    run_epochs: list[str],
) -> dict[str, dict[str, Any]]:
    """Load heatmap trajectory intervals and require the canonical parquet export."""
    trajectory_intervals, trajectory_source = load_trajectory_intervals(
        analysis_path,
        run_epochs,
    )
    if trajectory_source != "parquet":
        parquet_path = analysis_path / "trajectory_times.parquet"
        raise FileNotFoundError(
            "plot_place_field_heatmap.py requires a readable trajectory_times.parquet "
            "export and does not accept trajectory_times.pkl fallback. "
            f"Expected parquet at {parquet_path}. Regenerate it with "
            "`python -m v1ca1.helper.get_trajectory_times --animal-name ... --date ...`."
        )
    return trajectory_intervals


def prepare_heatmap_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    regions: tuple[str, ...],
    position_offset: int,
    speed_threshold_cm_s: float,
    requested_epoch: str | None = None,
) -> dict[str, Any]:
    """Load only the session inputs needed by the heatmap plotting workflow."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    run_epochs = get_run_epochs(epoch_tags)
    position_epoch_tags, timestamps_position, _position_source = load_position_timestamps(
        analysis_path
    )
    position_by_epoch_all, loaded_position_source = load_position_data_with_fallback(
        analysis_path,
    )

    available_run_epochs = [
        epoch
        for epoch in run_epochs
        if epoch in position_epoch_tags and epoch in position_by_epoch_all
    ]
    skipped_run_epochs = [epoch for epoch in run_epochs if epoch not in available_run_epochs]
    if skipped_run_epochs and requested_epoch is None:
        print(
            "Skipping run epochs missing position timestamps or position samples: "
            f"{skipped_run_epochs!r}"
        )

    if requested_epoch is not None:
        if requested_epoch not in run_epochs:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} was not found in run epochs {run_epochs!r}."
            )
        if requested_epoch not in position_epoch_tags:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} is missing position timestamps."
            )
        if requested_epoch not in position_by_epoch_all:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} is missing position samples in position.pkl."
            )
        selected_run_epochs = [requested_epoch]
    else:
        selected_run_epochs = available_run_epochs

    if not selected_run_epochs:
        raise ValueError(
            "No run epochs have both position timestamps and position samples for heatmap plotting."
        )

    timestamps_ephys_all, _ephys_source = load_ephys_timestamps_all(analysis_path)
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all,
        regions=regions,
    )
    trajectory_intervals = load_heatmap_trajectory_intervals(
        analysis_path,
        selected_run_epochs,
    )

    movement_by_run: dict[str, Any] = {}
    for epoch in selected_run_epochs:
        speed_tsd = build_speed_tsd(
            position_by_epoch_all[epoch],
            timestamps_position[epoch],
            position_offset=position_offset,
        )
        movement_by_run[epoch] = build_movement_interval(
            speed_tsd,
            speed_threshold_cm_s=speed_threshold_cm_s,
        )

    return {
        "analysis_path": analysis_path,
        "run_epochs": selected_run_epochs,
        "timestamps_position": {
            epoch: timestamps_position[epoch] for epoch in selected_run_epochs
        },
        "position_by_epoch": {
            epoch: position_by_epoch_all[epoch] for epoch in selected_run_epochs
        },
        "trajectory_intervals": trajectory_intervals,
        "movement_by_run": movement_by_run,
        "spikes_by_region": spikes_by_region,
        "position_source": loaded_position_source,
    }


def compute_place_tuning_curve(
    spikes: Any,
    linear_position: Any,
    epochs: Any,
    *,
    bin_edges: np.ndarray,
    sigma_bins: float,
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
        tuning_curve = smooth_tuning_curve_nan_aware(
            tuning_curve,
            pos_dim="linpos",
            sigma_bins=sigma_bins,
        )
    return tuning_curve


def compute_odd_even_place_tuning_curves(
    spikes: Any,
    linear_position_by_trajectory: dict[str, Any],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    *,
    bin_edges: np.ndarray,
    sigma_bins: float,
) -> tuple[dict[str, Any | None], dict[str, Any | None]]:
    """Compute odd- and even-trial place tuning curves for each trajectory."""
    odd_curves: dict[str, Any | None] = {}
    even_curves: dict[str, Any | None] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        starts, ends = _extract_interval_bounds(trajectory_intervals[trajectory_type])
        odd_starts, odd_ends, even_starts, even_ends = split_odd_even_bounds(starts, ends)

        odd_epochs = _make_interval_set(odd_starts, odd_ends).intersect(movement_interval)
        even_epochs = _make_interval_set(even_starts, even_ends).intersect(movement_interval)

        odd_curves[trajectory_type] = compute_place_tuning_curve(
            spikes,
            linear_position_by_trajectory[trajectory_type],
            odd_epochs,
            bin_edges=bin_edges,
            sigma_bins=sigma_bins,
        )
        even_curves[trajectory_type] = compute_place_tuning_curve(
            spikes,
            linear_position_by_trajectory[trajectory_type],
            even_epochs,
            bin_edges=bin_edges,
            sigma_bins=sigma_bins,
        )
    return odd_curves, even_curves


def plot_place_field_heatmap_trajectories(
    *,
    animal_name: str,
    date: str,
    analysis_path: Path,
    region: str,
    epoch: str,
    spikes: Any,
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    position_offset: int,
    place_bin_size_cm: float,
    sigma_bins: float,
    show: bool = False,
) -> Path | None:
    """Plot and save one 4x4 odd/even cross-trajectory heatmap figure."""
    import matplotlib.pyplot as plt

    print(f"Working on {animal_name} {date} {epoch} {region}.")

    linear_position_by_trajectory = build_linear_position_by_trajectory(
        animal_name,
        position,
        timestamps_position,
        trajectory_intervals,
        position_offset=position_offset,
    )
    bin_edges = build_place_bin_edges(
        animal_name,
        place_bin_size_cm=place_bin_size_cm,
    )
    odd_curves, even_curves = compute_odd_even_place_tuning_curves(
        spikes,
        linear_position_by_trajectory,
        trajectory_intervals,
        movement_interval,
        bin_edges=bin_edges,
        sigma_bins=sigma_bins,
    )

    guide_line_positions = get_guide_line_positions(animal_name)
    fig, axes = plt.subplots(
        nrows=len(TRAJECTORY_TYPES),
        ncols=len(TRAJECTORY_TYPES),
        figsize=(18, 12),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )

    row_ordering: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    y_limit = max(int(len(spikes.keys())), 1)
    for trajectory_type, order_curve in odd_curves.items():
        if order_curve is None:
            continue
        reference_units, order_values = _extract_tuning_curve_arrays(order_curve)
        if not _has_plottable_values(order_values):
            continue
        row_ordering[trajectory_type] = (
            reference_units,
            compute_unit_order(order_values),
        )
        y_limit = max(y_limit, int(reference_units.size))

    color_image = None
    plotted_any = False
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])

    for row_index, order_trajectory in enumerate(TRAJECTORY_TYPES):
        reference_units, unit_order = row_ordering.get(
            order_trajectory,
            (np.asarray([]), np.asarray([], dtype=int)),
        )
        order_curve = odd_curves[order_trajectory]

        for col_index, plot_trajectory in enumerate(TRAJECTORY_TYPES):
            axis = axes[row_index, col_index]
            axis.set_xlim(x_min, x_max)
            axis.set_ylim(y_limit, 0)
            for position_cm in guide_line_positions:
                axis.axvline(position_cm, color="black", alpha=0.5, linewidth=0.8)

            if row_index == 0:
                axis.set_title(f"even: {plot_trajectory}")
            if col_index == 0:
                axis.set_ylabel(f"odd order:\n{order_trajectory}")
            if row_index == len(TRAJECTORY_TYPES) - 1:
                axis.set_xlabel("linear position (cm)")

            display_curve = even_curves[plot_trajectory]
            if order_curve is None or display_curve is None or unit_order.size == 0:
                continue

            display_units, display_values = _extract_tuning_curve_arrays(display_curve)
            panel_values = align_and_normalize_panel_values(
                display_values,
                display_units,
                reference_units,
                unit_order,
            )
            if not _has_plottable_values(panel_values):
                continue

            image = axis.imshow(
                panel_values,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
                extent=[x_min, x_max, panel_values.shape[0], 0],
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            color_image = image if color_image is None else color_image
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(
            "Skipping place-field heatmap because no odd/even trajectory panels had "
            f"plottable data for {animal_name} {date} {region} {epoch}."
        )
        return None

    assert color_image is not None
    fig.colorbar(color_image, ax=axes, shrink=0.8, pad=0.01)
    fig.suptitle(f"{animal_name} {date} {region} {epoch}")

    output_dir = analysis_path / "figs" / "place_field_heatmap"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{region}_{epoch}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return output_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the heatmap workflow."""
    parser = argparse.ArgumentParser(
        description="Plot odd/even trajectory place-field heatmaps",
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
        "--epoch",
        help="Only plot one run epoch. Default: plot all run epochs.",
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
            "Place-field bin size in cm along each trajectory. "
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
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the odd/even trajectory place-field heatmap workflow."""
    args = parse_arguments(argv)
    selected_regions = (args.region,) if args.region is not None else REGIONS
    session = prepare_heatmap_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        requested_epoch=args.epoch,
    )

    for region in selected_regions:
        for epoch in session["run_epochs"]:
            plot_place_field_heatmap_trajectories(
                animal_name=args.animal_name,
                date=args.date,
                analysis_path=session["analysis_path"],
                region=region,
                epoch=epoch,
                spikes=session["spikes_by_region"][region],
                position=session["position_by_epoch"][epoch],
                timestamps_position=session["timestamps_position"][epoch],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                position_offset=args.position_offset,
                place_bin_size_cm=args.place_bin_size_cm,
                sigma_bins=args.sigma_bins,
                show=args.show,
            )


if __name__ == "__main__":
    main()
