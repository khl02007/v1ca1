from __future__ import annotations

"""Plot behavioral and neural state traces across sleep and run epochs."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, DEFAULT_NWB_ROOT, DEFAULT_POSITION_OFFSET
from v1ca1.sleep._session import (
    DEFAULT_FIRING_RATE_BIN_SIZE_S,
    DEFAULT_PLOT_CUTOFF_HZ,
    DEFAULT_PLOT_SPECTROGRAM_NOVERLAP,
    DEFAULT_PLOT_SPECTROGRAM_NPERSEG,
    DEFAULT_TIME_BIN_SIZE_S,
    DEFAULT_V1_LFP_CHANNEL,
    butter_filter_and_decimate,
    compute_firing_rate_matrix,
    compute_spectrogram_principal_component,
    compute_theta_delta_ratio,
    decimate_signal,
    get_analysis_path,
    get_epoch_trace,
    get_nwb_path,
    get_recording_sampling_frequency,
    get_speed_trace,
    get_time_spike_indicator,
    load_recording,
    load_sleep_session_inputs,
    load_sleep_sortings,
    lowpass_filter,
    order_units_by_epoch_spike_count,
    resolve_ripple_channel,
    validate_epochs,
    validate_recording_channel,
    validate_selected_epochs_across_sources,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for sleep phase figures."""
    parser = argparse.ArgumentParser(description="Plot behavioral and neural sleep-phase traces")
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base analysis directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="Optional subset of epoch labels to process. Default: all saved epochs.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples to ignore per epoch when plotting speed. "
            f"Default: {DEFAULT_POSITION_OFFSET}"
        ),
    )
    parser.add_argument(
        "--v1-lfp-channel",
        type=int,
        default=DEFAULT_V1_LFP_CHANNEL,
        help=f"Recording channel id used for the V1 LFP trace. Default: {DEFAULT_V1_LFP_CHANNEL}",
    )
    parser.add_argument(
        "--ripple-channel",
        type=int,
        help=(
            "Recording channel id used for the ripple-band CA1 LFP trace. "
            "Default: the first configured session ripple channel."
        ),
    )
    parser.add_argument(
        "--region",
        choices=("ca1", "v1"),
        help="Optional heatmap region to plot. Default: include both CA1 and V1 heatmaps.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure in addition to saving it.",
    )
    return parser.parse_args(argv)


def is_run_epoch(epoch: str) -> bool:
    """Return whether one epoch label follows the lab run-epoch naming convention."""
    return "r" in str(epoch).lower()


def select_plot_epochs(
    requested_epochs: list[str],
    *,
    timestamps_position: dict[str, np.ndarray],
    position_by_epoch: dict[str, np.ndarray],
) -> tuple[list[str], list[str]]:
    """Return plottable epochs and skipped epochs based on available position inputs."""
    plottable_epochs: list[str] = []
    skipped_epochs: list[str] = []
    for epoch in requested_epochs:
        if epoch in timestamps_position and epoch in position_by_epoch:
            plottable_epochs.append(epoch)
        else:
            skipped_epochs.append(epoch)
    return plottable_epochs, skipped_epochs


def get_linearized_run_position(position: np.ndarray) -> np.ndarray:
    """Return the legacy full-track linearized position used in sleep-phase figures."""
    import track_linearization as tl

    node_positions = np.array(
        [
            (55.0, 81.0),
            (23.0, 81.0),
            (87.0, 81.0),
            (55.0, 10.0),
            (23.0, 10.0),
            (87.0, 10.0),
        ],
        dtype=float,
    )
    edges = np.array(
        [
            (0, 3),
            (3, 4),
            (3, 5),
            (4, 1),
            (5, 2),
        ],
        dtype=int,
    )
    track_graph = tl.make_track_graph(node_positions, edges)
    position_df = tl.get_linearized_position(
        position=np.asarray(position, dtype=float),
        track_graph=track_graph,
        edge_order=[(0, 3), (3, 4), (4, 1), (3, 5), (5, 2)],
        edge_spacing=10,
    )
    return np.asarray(position_df["linear_position"], dtype=float)


def compute_multiunit_population_rate(
    spike_indicator: np.ndarray,
    *,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Return the ripple-detection population firing-rate trace for one spike matrix."""
    import ripple_detection as rd

    time_array = np.asarray(timestamps, dtype=float)
    if time_array.size < 2:
        raise ValueError("timestamps must contain at least two samples.")
    sampling_frequency = int(round(1.0 / np.median(np.diff(time_array))))
    return np.asarray(
        rd.get_multiunit_population_firing_rate(
            np.asarray(spike_indicator, dtype=float),
            sampling_frequency=sampling_frequency,
        ),
        dtype=float,
    ).reshape(-1)


def plot_sleep_phases_for_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    epochs: list[str] | None = None,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    v1_lfp_channel: int = DEFAULT_V1_LFP_CHANNEL,
    ripple_channel: int | None = None,
    region: str | None = None,
    show: bool = False,
) -> dict[str, Any]:
    """Plot and save one multi-panel sleep-phase figure per selected epoch."""
    import matplotlib.pyplot as plt

    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")

    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    session = load_sleep_session_inputs(analysis_path)
    selected_epochs = validate_epochs(session["epoch_tags"], epochs)
    validate_selected_epochs_across_sources(
        selected_epochs,
        source_epochs={
            "timestamps_ephys": session["timestamps_ephys"],
        },
    )

    recording = load_recording(get_nwb_path(animal_name=animal_name, date=date, nwb_root=nwb_root))
    validated_v1_channel = validate_recording_channel(
        recording,
        v1_lfp_channel,
        channel_name="V1 LFP channel",
    )
    selected_ripple_channel = resolve_ripple_channel(
        animal_name=animal_name,
        date=date,
        ripple_channel=ripple_channel,
    )
    validated_ripple_channel = validate_recording_channel(
        recording,
        selected_ripple_channel,
        channel_name="Ripple channel",
    )
    sampling_frequency = get_recording_sampling_frequency(recording)
    sortings = load_sleep_sortings(analysis_path)

    figure_dir = analysis_path / "figs" / "sleep"
    figure_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {animal_name} {date}.")
    print(
        "Using position source: "
        f"{session.get('sources', {}).get('position', 'unknown position source')}"
    )
    print(f"Saving figures under {figure_dir}")

    heatmap_regions = ("ca1", "v1") if region is None else (str(region),)
    heatmap_label = "_".join(heatmap_regions)
    print(f"Plotting heatmap regions: {', '.join(heatmap_regions)}")

    selected_epochs, skipped_epochs = select_plot_epochs(
        selected_epochs,
        timestamps_position=session["timestamps_position"],
        position_by_epoch=session["position_by_epoch"],
    )
    if skipped_epochs:
        print(
            "Skipping epochs without combined position data: "
            + ", ".join(skipped_epochs)
        )
    if not selected_epochs:
        raise ValueError(
            "No epochs have both saved timestamps and usable position data for sleep-phase plotting."
        )
    print(f"Plotting {len(selected_epochs)} epoch(s): {', '.join(selected_epochs)}")

    output_paths: dict[str, Path] = {}
    epoch_summaries: dict[str, dict[str, float]] = {}
    downsample_factor = max(1, int(sampling_frequency // 150))
    downsampled_sampling_frequency = sampling_frequency / float(downsample_factor)

    for epoch in selected_epochs:
        print(f"Plotting {date} {epoch}.")

        epoch_timestamps = session["timestamps_ephys"][epoch]
        epoch_time, ca1_signal = get_epoch_trace(
            recording,
            epoch_timestamps=epoch_timestamps,
            timestamps_ephys_all=session["timestamps_ephys_all"],
            channel_id=validated_ripple_channel,
        )
        _epoch_time_v1, v1_signal = get_epoch_trace(
            recording,
            epoch_timestamps=epoch_timestamps,
            timestamps_ephys_all=session["timestamps_ephys_all"],
            channel_id=validated_v1_channel,
        )
        ripple_lfp_time, ripple_lfp = butter_filter_and_decimate(
            epoch_time,
            ca1_signal,
            sampling_frequency=sampling_frequency,
            new_sampling_frequency=1000.0,
            lowcut_hz=150.0,
            highcut_hz=250.0,
        )

        ca1_lowpass = lowpass_filter(ca1_signal, DEFAULT_PLOT_CUTOFF_HZ, sampling_frequency)
        v1_lowpass = lowpass_filter(v1_signal, DEFAULT_PLOT_CUTOFF_HZ, sampling_frequency)
        ca1_downsampled = decimate_signal(ca1_lowpass, downsample_factor)
        v1_downsampled = decimate_signal(v1_lowpass, downsample_factor)

        v1_spectrogram_time, v1_pc1 = compute_spectrogram_principal_component(
            v1_downsampled,
            downsampled_sampling_frequency,
            max_frequency_hz=60.0,
            nperseg=DEFAULT_PLOT_SPECTROGRAM_NPERSEG,
            noverlap=DEFAULT_PLOT_SPECTROGRAM_NOVERLAP,
        )
        ca1_spectrogram_time, theta_delta_ratio = compute_theta_delta_ratio(
            ca1_downsampled,
            downsampled_sampling_frequency,
            nperseg=DEFAULT_PLOT_SPECTROGRAM_NPERSEG,
            noverlap=DEFAULT_PLOT_SPECTROGRAM_NOVERLAP,
        )

        spike_time, spike_indicator_ca1 = get_time_spike_indicator(
            session["timestamps_position"][epoch],
            position_offset=position_offset,
            time_bin_size_s=DEFAULT_TIME_BIN_SIZE_S,
            sorting=sortings["ca1"],
            timestamps_ephys_all=session["timestamps_ephys_all"],
            temporal_overlap=False,
        )
        _spike_time_v1, spike_indicator_v1 = get_time_spike_indicator(
            session["timestamps_position"][epoch],
            position_offset=position_offset,
            time_bin_size_s=DEFAULT_TIME_BIN_SIZE_S,
            sorting=sortings["v1"],
            timestamps_ephys_all=session["timestamps_ephys_all"],
            temporal_overlap=False,
        )
        multiunit_ca1 = compute_multiunit_population_rate(
            spike_indicator_ca1,
            timestamps=spike_time,
        )
        multiunit_v1 = compute_multiunit_population_rate(
            spike_indicator_v1,
            timestamps=spike_time,
        )

        trimmed_position, position_time, speed = get_speed_trace(
            session["position_by_epoch"][epoch],
            session["timestamps_position"][epoch],
            position_offset=position_offset,
        )
        firing_rate_by_region: dict[str, np.ndarray] = {}
        firing_rate_time: np.ndarray | None = None
        for selected_region in heatmap_regions:
            firing_rate_matrix, region_time = compute_firing_rate_matrix(
                sortings[selected_region],
                epoch_timestamps=epoch_timestamps,
                timestamps_ephys_all=session["timestamps_ephys_all"],
                bin_size_s=DEFAULT_FIRING_RATE_BIN_SIZE_S,
                zscore_values=True,
            )
            unit_order = order_units_by_epoch_spike_count(
                sortings[selected_region],
                start_time=float(epoch_timestamps[0]),
                end_time=float(epoch_timestamps[-1]),
            )
            firing_rate_by_region[selected_region] = firing_rate_matrix[unit_order]
            if firing_rate_time is None:
                firing_rate_time = region_time
        if firing_rate_time is None:
            raise ValueError("Could not build a firing-rate time axis for the selected regions.")

        fig, axes = plt.subplots(
            nrows=5 + len(heatmap_regions),
            figsize=(40, 3 * (5 + len(heatmap_regions))),
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1] + [3] * len(heatmap_regions)},
        )
        axes = np.atleast_1d(axes)

        if is_run_epoch(epoch):
            linear_position = get_linearized_run_position(trimmed_position)
            axes[0].plot(position_time, linear_position)
            axes[0].twinx().plot(position_time, speed, "k")
            axes[0].set_ylabel("Position (cm)\nSpeed (cm/s)")
        else:
            axes[0].plot(position_time, speed, "k")
            axes[0].set_ylabel("Speed (cm/s)")

        axes[1].plot(v1_spectrogram_time + epoch_time[0], v1_pc1)
        axes[2].plot(ca1_spectrogram_time + epoch_time[0], theta_delta_ratio)
        axes[3].plot(ripple_lfp_time, ripple_lfp)
        axes[4].plot(spike_time, multiunit_ca1, label="CA1")
        axes[4].plot(spike_time, multiunit_v1, label="V1")
        axes[4].legend()
        axes[1].set_ylabel("V1 LFP spectrogram\nPC1 (zscore)")
        axes[2].set_ylabel("CA1 Theta / Delta")
        axes[3].set_ylabel(r"CA1 ripple band LFP ($\mu$V)")
        axes[4].set_ylabel("Multiunit firing rate (Hz)")
        for axis, selected_region in zip(axes[5:], heatmap_regions, strict=True):
            firing_rate_matrix = firing_rate_by_region[selected_region]
            axis.imshow(
                firing_rate_matrix,
                vmin=-1.0,
                vmax=1.0,
                aspect="auto",
                extent=[firing_rate_time[0], firing_rate_time[-1], 0, firing_rate_matrix.shape[0]],
            )
            axis.set_ylabel(f"Firing rate {selected_region.upper()} (zscore)")

        for axis in axes:
            axis.set_xlim([firing_rate_time[0], firing_rate_time[-1]])
        axes[1].set_ylim([0, 10])
        axes[2].set_ylim([0, 50])
        axes[-1].set_xlabel("Time (s)")
        axes[0].set_title(f"{date} {epoch} sleep phase")

        output_path = figure_dir / f"{epoch}_{heatmap_label}.png"
        fig.savefig(output_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        print(f"Saved figure to {output_path}")

        output_paths[epoch] = output_path
        epoch_summaries[epoch] = {
            **{
                f"{selected_region}_unit_count": float(firing_rate_by_region[selected_region].shape[0])
                for selected_region in heatmap_regions
            },
            "firing_rate_bin_count": float(firing_rate_time.shape[0]),
            "spike_indicator_sample_count": float(spike_time.shape[0]),
            "position_sample_count": float(position_time.shape[0]),
            "heatmap_region_count": float(len(heatmap_regions)),
        }

    outputs = {
        "figure_paths": output_paths,
        "selected_epochs": selected_epochs,
        "sources": session["sources"],
        "epoch_summaries": epoch_summaries,
        "v1_lfp_channel": validated_v1_channel,
        "ripple_channel": validated_ripple_channel,
        "heatmap_regions": list(heatmap_regions),
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.sleep.plot_sleep_phases",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "epochs": epochs,
            "position_offset": position_offset,
            "v1_lfp_channel": validated_v1_channel,
            "ripple_channel": validated_ripple_channel,
            "region": region,
            "show": show,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")
    outputs["log_path"] = log_path
    return outputs


def main(argv: list[str] | None = None) -> None:
    """Run the sleep-phase plotting CLI."""
    args = parse_arguments(argv)
    plot_sleep_phases_for_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        epochs=args.epochs,
        position_offset=args.position_offset,
        v1_lfp_channel=args.v1_lfp_channel,
        ripple_channel=args.ripple_channel,
        region=args.region,
        show=args.show,
    )


if __name__ == "__main__":
    main()
