from __future__ import annotations

"""Plot movement-restricted firing-rate histograms for one session."""

import argparse
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
    load_ephys_timestamps_all,
    load_epoch_tags,
    load_position_data_with_precedence,
    load_position_timestamps,
    load_spikes_by_region,
)


DEFAULT_HISTOGRAM_BIN_SIZE_HZ = 0.5
DEFAULT_OUTPUT_DIRNAME = "movement_firing_rate_histogram"


def get_selected_epochs(
    epoch_tags: list[str],
    position_epoch_tags: list[str],
    position_by_epoch: dict[str, np.ndarray],
    *,
    requested_epoch: str | None = None,
) -> list[str]:
    """Return ordered epochs with both timestamps and position samples."""
    available_epochs = [
        epoch
        for epoch in epoch_tags
        if epoch in position_epoch_tags and epoch in position_by_epoch
    ]
    skipped_epochs = [epoch for epoch in epoch_tags if epoch not in available_epochs]
    if skipped_epochs and requested_epoch is None:
        print(
            "Skipping epochs missing position timestamps or position samples: "
            f"{skipped_epochs!r}"
        )

    if requested_epoch is not None:
        if requested_epoch not in epoch_tags:
            raise ValueError(
                f"Requested epoch {requested_epoch!r} was not found in session epochs {epoch_tags!r}."
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

    if not available_epochs:
        raise ValueError(
            "No epochs have both position timestamps and position samples for firing-rate plotting."
        )
    return available_epochs


def prepare_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path,
    regions: tuple[str, ...],
    position_offset: int,
    speed_threshold_cm_s: float,
    requested_epoch: str | None = None,
) -> dict[str, Any]:
    """Load the session inputs required by the histogram workflow."""
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    position_epoch_tags, timestamps_position, _timestamp_source = load_position_timestamps(
        analysis_path
    )
    position_by_epoch, position_source = load_position_data_with_precedence(
        analysis_path,
        position_source="auto",
        clean_dlc_input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        clean_dlc_input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    selected_epochs = get_selected_epochs(
        epoch_tags,
        position_epoch_tags,
        position_by_epoch,
        requested_epoch=requested_epoch,
    )
    timestamps_ephys_all, _ephys_source = load_ephys_timestamps_all(analysis_path)
    spikes_by_region = load_spikes_by_region(
        analysis_path,
        timestamps_ephys_all,
        regions=regions,
    )

    movement_by_epoch: dict[str, Any] = {}
    movement_duration_s: dict[str, float] = {}
    for epoch in selected_epochs:
        speed_tsd = build_speed_tsd(
            position_by_epoch[epoch],
            timestamps_position[epoch],
            position_offset=position_offset,
        )
        movement_interval = build_movement_interval(
            speed_tsd,
            speed_threshold_cm_s=speed_threshold_cm_s,
        )
        movement_by_epoch[epoch] = movement_interval
        movement_duration_s[epoch] = float(movement_interval.tot_length())

    return {
        "analysis_path": analysis_path,
        "selected_epochs": selected_epochs,
        "movement_by_epoch": movement_by_epoch,
        "movement_duration_s": movement_duration_s,
        "spikes_by_region": spikes_by_region,
        "position_source": position_source,
    }


def compute_epoch_movement_firing_rates(
    spikes: Any,
    movement_interval: Any,
    movement_duration_s: float,
) -> np.ndarray | None:
    """Return per-unit movement firing rates for one epoch or `None` if empty."""
    if movement_duration_s <= 0:
        return None

    starts = np.asarray(movement_interval.start, dtype=float).ravel()
    ends = np.asarray(movement_interval.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Movement interval start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )

    firing_rates_hz = np.empty(len(list(spikes.keys())), dtype=float)
    for unit_index, unit_id in enumerate(spikes.keys()):
        spike_times_s = np.asarray(spikes[unit_id].t, dtype=float)
        start_indices = np.searchsorted(spike_times_s, starts, side="left")
        end_indices = np.searchsorted(spike_times_s, ends, side="right")
        spike_count = int(np.sum(end_indices - start_indices))
        firing_rates_hz[unit_index] = spike_count / movement_duration_s

    return firing_rates_hz


def get_region_histogram_edges(
    firing_rates_by_epoch: dict[str, np.ndarray | None],
    bin_size_hz: float = DEFAULT_HISTOGRAM_BIN_SIZE_HZ,
) -> np.ndarray:
    """Return shared histogram bin edges for one region figure."""
    if bin_size_hz <= 0:
        raise ValueError("--hist-bin-size-hz must be positive.")

    finite_rate_arrays = [
        rates[np.isfinite(rates)]
        for rates in firing_rates_by_epoch.values()
        if rates is not None and np.isfinite(rates).any()
    ]
    if not finite_rate_arrays:
        return np.array([0.0, bin_size_hz], dtype=float)

    combined_rates = np.concatenate(finite_rate_arrays)
    max_rate = float(np.nanmax(combined_rates))
    upper_edge = max(bin_size_hz, np.ceil(max_rate / bin_size_hz) * bin_size_hz)
    return np.arange(0.0, upper_edge + bin_size_hz, bin_size_hz, dtype=float)


def plot_region_histogram(
    *,
    animal_name: str,
    date: str,
    region: str,
    selected_epochs: list[str],
    firing_rates_by_epoch: dict[str, np.ndarray | None],
    movement_duration_s: dict[str, float],
    output_dir: Path,
    requested_epoch: str | None,
    hist_bin_size_hz: float,
    show: bool,
) -> Path:
    """Plot and save the per-epoch movement firing-rate histogram for one region."""
    import matplotlib.pyplot as plt

    n_epochs = len(selected_epochs)
    fig_width = max(4.0 * n_epochs, 5.0)
    fig, axes = plt.subplots(
        ncols=n_epochs,
        figsize=(fig_width, 4.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    bin_edges = get_region_histogram_edges(
        firing_rates_by_epoch,
        bin_size_hz=hist_bin_size_hz,
    )
    x_max = float(bin_edges[-1])

    for axis, epoch in zip(axes, selected_epochs):
        rates = firing_rates_by_epoch[epoch]
        axis.set_title(epoch)
        axis.set_xlim(float(bin_edges[0]), x_max)

        if rates is None:
            axis.text(
                0.5,
                0.5,
                "No movement samples",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            axis.text(
                0.5,
                0.2,
                f"duration = {movement_duration_s[epoch]:.3f} s",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            continue

        finite_rates = rates[np.isfinite(rates)]
        total_units = int(rates.size)
        weights = np.full(finite_rates.shape, 1.0 / total_units, dtype=float)
        axis.hist(
            finite_rates,
            bins=bin_edges,
            weights=weights,
            color="0.7",
            edgecolor="black",
            linewidth=1.0,
        )

    axes[0].set_ylabel("Fraction of all units")
    fig.supxlabel("Firing rate during movement (Hz)")
    fig.suptitle(f"{animal_name} {date} {region}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{region}_{requested_epoch}.png" if requested_epoch is not None else f"{region}.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return output_path


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the histogram workflow."""
    parser = argparse.ArgumentParser(
        description="Plot movement-restricted firing-rate histograms for all units",
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
        help="Only plot one epoch. Default: plot all epochs with usable position.",
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
        "--hist-bin-size-hz",
        type=float,
        default=DEFAULT_HISTOGRAM_BIN_SIZE_HZ,
        help=(
            "Histogram bin width in Hz. "
            f"Default: {DEFAULT_HISTOGRAM_BIN_SIZE_HZ}"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the movement firing-rate histogram workflow."""
    args = parse_arguments(argv)
    selected_regions = (args.region,) if args.region is not None else REGIONS
    session = prepare_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        requested_epoch=args.epoch,
    )

    output_dir = session["analysis_path"] / "figs" / DEFAULT_OUTPUT_DIRNAME
    for region in selected_regions:
        firing_rates_by_epoch = {
            epoch: compute_epoch_movement_firing_rates(
                session["spikes_by_region"][region],
                session["movement_by_epoch"][epoch],
                session["movement_duration_s"][epoch],
            )
            for epoch in session["selected_epochs"]
        }
        plot_region_histogram(
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            selected_epochs=session["selected_epochs"],
            firing_rates_by_epoch=firing_rates_by_epoch,
            movement_duration_s=session["movement_duration_s"],
            output_dir=output_dir,
            requested_epoch=args.epoch,
            hist_bin_size_hz=args.hist_bin_size_hz,
            show=args.show,
        )


if __name__ == "__main__":
    main()
