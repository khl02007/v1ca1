from __future__ import annotations

"""Plot theta phase-preference heatmaps for one session."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    build_movement_interval,
    build_speed_tsd,
    get_analysis_path,
    get_run_epochs,
    load_ephys_timestamps_all,
    load_epoch_tags,
    load_position_data_with_precedence,
    load_position_timestamps,
    load_spikes_by_region,
)
from v1ca1.oscillation.get_theta_phase import (
    OUTPUT_DIRNAME as THETA_OUTPUT_DIRNAME,
    THETA_METADATA_FILENAME,
    THETA_PHASE_DIRNAME,
)

DEFAULT_PHASE_BIN_COUNT = 24
DEFAULT_MIN_SPIKES = 20
DEFAULT_FIGURE_DIRNAME = "theta_phase_preference"
DEFAULT_UNIT_ORDERING = "movement_rate"
UNIT_ORDERING_CHOICES = ("movement_rate", "preferred_phase")


def load_theta_phase_metadata_epochs(analysis_path: Path) -> list[str] | None:
    """Return saved theta metadata epochs when the metadata manifest exists."""
    metadata_path = analysis_path / THETA_OUTPUT_DIRNAME / THETA_METADATA_FILENAME
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    epochs = metadata.get("epochs")
    if epochs is None:
        return None
    if not isinstance(epochs, list):
        raise ValueError(
            f"Theta metadata field 'epochs' must be a list in {metadata_path}."
        )
    return [str(epoch) for epoch in epochs]


def get_theta_phase_epoch_paths(analysis_path: Path) -> dict[str, Path]:
    """Return saved theta-phase `.npz` files keyed by epoch."""
    theta_phase_dir = analysis_path / THETA_OUTPUT_DIRNAME / THETA_PHASE_DIRNAME
    theta_phase_paths = {
        path.stem: path
        for path in sorted(theta_phase_dir.glob("*.npz"))
    }
    if theta_phase_paths:
        return theta_phase_paths

    raise FileNotFoundError(
        "Could not find theta-phase `.npz` outputs under "
        f"{theta_phase_dir}. Run `python -m v1ca1.oscillation.get_theta_phase "
        "--animal-name ... --date ...` first."
    )


def select_theta_phase_epochs(
    run_epochs: list[str],
    theta_phase_epoch_paths: dict[str, Path],
    *,
    requested_epoch: str | None = None,
    metadata_epochs: list[str] | None = None,
) -> list[str]:
    """Select run epochs that have saved theta-phase outputs."""
    if metadata_epochs is None:
        available_epochs = set(theta_phase_epoch_paths)
    else:
        available_epochs = {
            epoch for epoch in metadata_epochs if epoch in theta_phase_epoch_paths
        }

    if requested_epoch is not None:
        if requested_epoch not in run_epochs:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} was not found in run epochs {run_epochs!r}."
            )
        if requested_epoch not in available_epochs:
            missing_path = (
                theta_phase_epoch_paths.get(requested_epoch)
                or (
                    next(iter(theta_phase_epoch_paths.values())).parent
                    / f"{requested_epoch}.npz"
                )
            )
            raise FileNotFoundError(
                f"Requested epoch {requested_epoch!r} does not have a saved theta-phase "
                f"output at {missing_path}."
            )
        return [requested_epoch]

    selected_epochs = [epoch for epoch in run_epochs if epoch in available_epochs]
    if selected_epochs:
        return selected_epochs

    theta_phase_dir = next(iter(theta_phase_epoch_paths.values())).parent
    raise ValueError(
        "No run epochs have saved theta-phase outputs under "
        f"{theta_phase_dir}. Available theta-phase epochs: "
        f"{sorted(available_epochs)!r}."
    )


def load_theta_phase_tsd(theta_phase_path: Path) -> Any:
    """Load one saved theta-phase time series."""
    try:
        import pynapple as nap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pynapple is required to load saved theta-phase `.npz` files."
        ) from exc

    if not theta_phase_path.exists():
        raise FileNotFoundError(f"Theta-phase file not found: {theta_phase_path}")
    return nap.load_file(theta_phase_path)


def get_movement_ready_epochs(
    run_epochs: list[str],
    position_epoch_tags: list[str],
    position_by_epoch: dict[str, np.ndarray],
    *,
    requested_epoch: str | None = None,
) -> list[str]:
    """Return run epochs that have both position timestamps and position samples."""
    available_epochs = [
        epoch
        for epoch in run_epochs
        if epoch in position_epoch_tags and epoch in position_by_epoch
    ]
    skipped_epochs = [epoch for epoch in run_epochs if epoch not in available_epochs]
    if skipped_epochs and requested_epoch is None:
        print(
            "Skipping run epochs missing position timestamps or position samples: "
            f"{skipped_epochs!r}"
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
        if requested_epoch not in position_by_epoch:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} is missing position samples."
            )
        return [requested_epoch]

    if available_epochs:
        return available_epochs

    raise ValueError(
        "No run epochs have both position timestamps and position samples for "
        "movement-restricted theta phase plotting."
    )


def extract_theta_phase_arrays(theta_phase_tsd: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned timestamp and phase arrays from one theta-phase time series."""
    timestamps = np.asarray(theta_phase_tsd.t, dtype=float).ravel()
    phase_values = np.asarray(theta_phase_tsd.d, dtype=float).ravel()

    if timestamps.ndim != 1 or phase_values.ndim != 1:
        raise ValueError("Theta phase timestamps and values must both be 1D.")
    if timestamps.size == 0:
        raise ValueError("Theta phase time series is empty.")
    if timestamps.size != phase_values.size:
        raise ValueError(
            "Theta phase timestamps and values must have matching lengths, got "
            f"{timestamps.size} and {phase_values.size}."
        )
    if np.any(~np.isfinite(timestamps)) or np.any(~np.isfinite(phase_values)):
        raise ValueError("Theta phase timestamps and values must be finite.")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Theta phase timestamps must be strictly increasing.")
    return timestamps, phase_values


def sample_phase_at_spike_times(
    spike_times: np.ndarray,
    theta_phase_timestamps: np.ndarray,
    theta_phase_values: np.ndarray,
) -> np.ndarray:
    """Sample theta phase at spike times using complex interpolation."""
    spike_time_array = np.asarray(spike_times, dtype=float).ravel()
    theta_time_array = np.asarray(theta_phase_timestamps, dtype=float).ravel()
    theta_phase_array = np.asarray(theta_phase_values, dtype=float).ravel()

    if spike_time_array.size == 0:
        return np.array([], dtype=float)
    if theta_time_array.size != theta_phase_array.size:
        raise ValueError(
            "Theta phase timestamps and values must have matching lengths, got "
            f"{theta_time_array.size} and {theta_phase_array.size}."
        )
    if theta_time_array.size == 0:
        return np.array([], dtype=float)

    in_support = (
        np.isfinite(spike_time_array)
        & (spike_time_array >= theta_time_array[0])
        & (spike_time_array <= theta_time_array[-1])
    )
    if not np.any(in_support):
        return np.array([], dtype=float)

    spike_times_supported = spike_time_array[in_support]
    complex_phase = np.exp(1j * theta_phase_array)
    real_interp = np.interp(
        spike_times_supported,
        theta_time_array,
        complex_phase.real,
    )
    imag_interp = np.interp(
        spike_times_supported,
        theta_time_array,
        complex_phase.imag,
    )
    return np.angle(real_interp + 1j * imag_interp)


def restrict_spike_times_to_interval(
    spike_times: np.ndarray,
    intervals: Any,
) -> np.ndarray:
    """Return spike times that fall within one IntervalSet-like object."""
    spike_time_array = np.asarray(spike_times, dtype=float).ravel()
    if spike_time_array.size == 0:
        return np.array([], dtype=float)

    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Interval start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    if starts.size == 0:
        return np.array([], dtype=float)

    kept_spike_times: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        start_index = int(np.searchsorted(spike_time_array, float(start), side="left"))
        end_index = int(np.searchsorted(spike_time_array, float(end), side="right"))
        if end_index > start_index:
            kept_spike_times.append(spike_time_array[start_index:end_index])

    if not kept_spike_times:
        return np.array([], dtype=float)
    return np.concatenate(kept_spike_times)


def build_phase_bin_edges(phase_bin_count: int) -> np.ndarray:
    """Return evenly spaced theta-phase bin edges spanning [-pi, pi)."""
    if phase_bin_count <= 0:
        raise ValueError("--phase-bin-count must be positive.")
    return np.linspace(-np.pi, np.pi, int(phase_bin_count) + 1, dtype=float)


def get_interval_total_duration(intervals: Any) -> float:
    """Return the total duration of one IntervalSet-like object in seconds."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Interval start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    if starts.size == 0:
        return 0.0
    return float(np.sum(ends - starts))


def build_unit_phase_histograms(
    spikes: Any,
    theta_phase_timestamps: np.ndarray,
    theta_phase_values: np.ndarray,
    *,
    phase_bin_edges: np.ndarray,
    movement_interval: Any,
    min_spikes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one unit-by-phase histogram matrix for a region and epoch."""
    if min_spikes <= 0:
        raise ValueError("--min-spikes must be positive.")

    movement_duration_s = get_interval_total_duration(movement_interval)
    if movement_duration_s < 0:
        raise ValueError("Movement interval duration must be non-negative.")

    unit_ids: list[Any] = []
    histogram_rows: list[np.ndarray] = []
    movement_firing_rates_hz: list[float] = []
    for unit_id in spikes.keys():
        spike_times = restrict_spike_times_to_interval(
            np.asarray(spikes[unit_id].t, dtype=float).ravel(),
            movement_interval,
        )
        spike_phases = sample_phase_at_spike_times(
            spike_times,
            theta_phase_timestamps,
            theta_phase_values,
        )
        if spike_phases.size < min_spikes:
            continue

        counts, _ = np.histogram(spike_phases, bins=phase_bin_edges)
        if counts.sum() <= 0:
            continue
        unit_ids.append(unit_id)
        histogram_rows.append(np.asarray(counts, dtype=float))
        movement_firing_rates_hz.append(
            float(spike_times.size) / movement_duration_s
            if movement_duration_s > 0
            else np.nan
        )

    if not histogram_rows:
        return np.asarray([], dtype=object), np.empty(
            (0, phase_bin_edges.size - 1),
            dtype=float,
        ), np.array([], dtype=float)
    return np.asarray(unit_ids), np.vstack(histogram_rows), np.asarray(
        movement_firing_rates_hz,
        dtype=float,
    )


def normalize_heatmap_rows(values: np.ndarray) -> np.ndarray:
    """Peak-normalize each heatmap row independently."""
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap matrix, got shape {value_array.shape}.")

    row_max = np.full(value_array.shape[0], np.nan, dtype=float)
    finite_rows = np.isfinite(value_array).any(axis=1)
    if np.any(finite_rows):
        row_max[finite_rows] = np.nanmax(value_array[finite_rows], axis=1)

    valid_rows = np.isfinite(row_max) & (row_max > 0)
    normalized = np.full_like(value_array, np.nan, dtype=float)
    if np.any(valid_rows):
        normalized[valid_rows] = value_array[valid_rows] / row_max[valid_rows, None]
    return normalized


def compute_unit_order(values: np.ndarray) -> np.ndarray:
    """Order units by the phase-bin index of their maximum response."""
    value_array = np.asarray(values, dtype=float)
    if value_array.ndim != 2:
        raise ValueError(f"Expected a 2D heatmap matrix, got shape {value_array.shape}.")

    n_units, n_bins = value_array.shape
    if n_units == 0:
        return np.array([], dtype=int)

    row_max = np.full(n_units, np.nan, dtype=float)
    finite_rows = np.isfinite(value_array).any(axis=1)
    if np.any(finite_rows):
        row_max[finite_rows] = np.nanmax(value_array[finite_rows], axis=1)

    valid_rows = np.isfinite(row_max) & (row_max > 0)
    safe_values = np.where(np.isfinite(value_array), value_array, -np.inf)
    peak_index = np.argmax(safe_values, axis=1)
    peak_index = np.where(valid_rows, peak_index, n_bins)
    return np.argsort(peak_index, kind="stable")


def compute_unit_order_by_movement_rate(movement_firing_rates_hz: np.ndarray) -> np.ndarray:
    """Order units by movement firing rate from highest to lowest."""
    value_array = np.asarray(movement_firing_rates_hz, dtype=float).ravel()
    if value_array.ndim != 1:
        raise ValueError(
            "Expected a 1D movement firing-rate array, got shape "
            f"{value_array.shape}."
        )
    if value_array.size == 0:
        return np.array([], dtype=int)

    valid_rows = np.isfinite(value_array)
    safe_values = np.where(valid_rows, value_array, -np.inf)
    return np.argsort(-safe_values, kind="stable")


def compute_heatmap_unit_order(
    histogram_matrix: np.ndarray,
    movement_firing_rates_hz: np.ndarray,
    *,
    unit_ordering: str,
) -> np.ndarray:
    """Return the requested row ordering for one heatmap."""
    if unit_ordering == "movement_rate":
        return compute_unit_order_by_movement_rate(movement_firing_rates_hz)
    if unit_ordering == "preferred_phase":
        return compute_unit_order(histogram_matrix)
    raise ValueError(
        f"Unsupported --unit-ordering {unit_ordering!r}. "
        f"Expected one of {UNIT_ORDERING_CHOICES!r}."
    )


def plot_theta_phase_heatmap(
    *,
    animal_name: str,
    date: str,
    analysis_path: Path,
    region: str,
    epoch: str,
    spikes: Any,
    theta_phase_tsd: Any,
    movement_interval: Any,
    phase_bin_count: int,
    min_spikes: int,
    unit_ordering: str,
    show: bool = False,
) -> Path | None:
    """Plot and save one theta phase-preference heatmap."""
    import matplotlib.pyplot as plt

    theta_phase_timestamps, theta_phase_values = extract_theta_phase_arrays(theta_phase_tsd)
    phase_bin_edges = build_phase_bin_edges(phase_bin_count)
    unit_ids, histogram_matrix, movement_firing_rates_hz = build_unit_phase_histograms(
        spikes,
        theta_phase_timestamps,
        theta_phase_values,
        phase_bin_edges=phase_bin_edges,
        movement_interval=movement_interval,
        min_spikes=min_spikes,
    )
    if unit_ids.size == 0:
        print(
            "Skipping theta phase-preference heatmap because no units met the minimum "
            f"spike count for {animal_name} {date} {region} {epoch}."
        )
        return None

    normalized_matrix = normalize_heatmap_rows(histogram_matrix)
    unit_order = compute_heatmap_unit_order(
        histogram_matrix,
        movement_firing_rates_hz,
        unit_ordering=unit_ordering,
    )
    ordered_matrix = normalized_matrix[unit_order]

    fig, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = axis.imshow(
        ordered_matrix,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        extent=[phase_bin_edges[0], phase_bin_edges[-1], ordered_matrix.shape[0], 0],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axis.set_xlabel("theta phase (rad)")
    if unit_ordering == "movement_rate":
        axis.set_ylabel("units ordered by movement firing rate")
    else:
        axis.set_ylabel("units ordered by preferred phase")
    axis.set_title(f"{animal_name} {date} {region} {epoch}")
    axis.set_xticks([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi])
    axis.set_xticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"])
    axis.set_yticks([])
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85, pad=0.02)
    colorbar.set_label("peak-normalized spike count")

    output_dir = analysis_path / "figs" / DEFAULT_FIGURE_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{region}_{epoch}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return output_path


def prepare_theta_phase_preference_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    regions: tuple[str, ...],
    position_offset: int,
    speed_threshold_cm_s: float,
    requested_epoch: str | None = None,
) -> dict[str, Any]:
    """Load the session inputs needed by the theta phase-preference workflow."""
    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    run_epochs = get_run_epochs(epoch_tags)
    position_epoch_tags, timestamps_position, _position_source = load_position_timestamps(
        analysis_path
    )
    position_by_epoch = load_position_data_with_precedence(
        analysis_path,
        position_source="auto",
        clean_dlc_input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        clean_dlc_input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )[0]
    movement_ready_epochs = get_movement_ready_epochs(
        run_epochs,
        position_epoch_tags,
        position_by_epoch,
        requested_epoch=requested_epoch,
    )
    metadata_epochs = load_theta_phase_metadata_epochs(analysis_path)
    theta_phase_epoch_paths = get_theta_phase_epoch_paths(analysis_path)
    selected_epochs = select_theta_phase_epochs(
        movement_ready_epochs,
        theta_phase_epoch_paths,
        requested_epoch=requested_epoch,
        metadata_epochs=metadata_epochs,
    )

    timestamps_ephys_all, _ephys_source = load_ephys_timestamps_all(analysis_path)
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all,
        regions=regions,
    )
    movement_by_epoch: dict[str, Any] = {}
    for epoch in selected_epochs:
        speed_tsd = build_speed_tsd(
            position_by_epoch[epoch],
            timestamps_position[epoch],
            position_offset=position_offset,
        )
        movement_by_epoch[epoch] = build_movement_interval(
            speed_tsd,
            speed_threshold_cm_s=speed_threshold_cm_s,
        )
    return {
        "analysis_path": analysis_path,
        "selected_epochs": selected_epochs,
        "movement_by_epoch": movement_by_epoch,
        "theta_phase_epoch_paths": {
            epoch: theta_phase_epoch_paths[epoch] for epoch in selected_epochs
        },
        "spikes_by_region": spikes_by_region,
    }


def plot_theta_phase_preference_for_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    region: str | None = None,
    epoch: str | None = None,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    phase_bin_count: int = DEFAULT_PHASE_BIN_COUNT,
    min_spikes: int = DEFAULT_MIN_SPIKES,
    unit_ordering: str = DEFAULT_UNIT_ORDERING,
    show: bool = False,
) -> dict[str, dict[str, Path | None]]:
    """Run the theta phase-preference workflow for one session."""
    selected_regions = (region,) if region is not None else REGIONS
    session = prepare_theta_phase_preference_session(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
        regions=selected_regions,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        requested_epoch=epoch,
    )

    output_paths: dict[str, dict[str, Path | None]] = {}
    for selected_region in selected_regions:
        output_paths[selected_region] = {}
        for selected_epoch in session["selected_epochs"]:
            print(
                "Processing theta phase-preference heatmap for "
                f"region={selected_region}, epoch={selected_epoch}."
            )
            theta_phase_tsd = load_theta_phase_tsd(
                session["theta_phase_epoch_paths"][selected_epoch]
            )
            output_paths[selected_region][selected_epoch] = plot_theta_phase_heatmap(
                animal_name=animal_name,
                date=date,
                analysis_path=session["analysis_path"],
                region=selected_region,
                epoch=selected_epoch,
                spikes=session["spikes_by_region"][selected_region],
                theta_phase_tsd=theta_phase_tsd,
                movement_interval=session["movement_by_epoch"][selected_epoch],
                phase_bin_count=phase_bin_count,
                min_spikes=min_spikes,
                unit_ordering=unit_ordering,
                show=show,
            )
    return output_paths


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for theta phase-preference heatmaps."""
    parser = argparse.ArgumentParser(
        description="Plot theta phase-preference heatmaps by region and epoch",
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
        help="Only plot one run epoch. Default: plot all run epochs with saved theta phase.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples to ignore per epoch when defining "
            f"movement. Default: {DEFAULT_POSITION_OFFSET}"
        ),
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
        "--phase-bin-count",
        type=int,
        default=DEFAULT_PHASE_BIN_COUNT,
        help=(
            "Number of phase bins spanning [-pi, pi). "
            f"Default: {DEFAULT_PHASE_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--min-spikes",
        type=int,
        default=DEFAULT_MIN_SPIKES,
        help=(
            "Minimum number of spike-assigned phases required to keep one unit. "
            f"Default: {DEFAULT_MIN_SPIKES}"
        ),
    )
    parser.add_argument(
        "--unit-ordering",
        choices=UNIT_ORDERING_CHOICES,
        default=DEFAULT_UNIT_ORDERING,
        help=(
            "How to order unit rows in the heatmap. "
            f"Default: {DEFAULT_UNIT_ORDERING}"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the theta phase-preference heatmap CLI."""
    args = parse_arguments(argv)
    plot_theta_phase_preference_for_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        region=args.region,
        epoch=args.epoch,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        phase_bin_count=args.phase_bin_count,
        min_spikes=args.min_spikes,
        unit_ordering=args.unit_ordering,
        show=args.show,
    )


if __name__ == "__main__":
    main()
