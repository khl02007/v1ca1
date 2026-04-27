from __future__ import annotations

"""Compute sleep intervals for one session from V1 LFP and movement speed.

The canonical output is a single parquet table with one row per sleep interval
and columns `start`, `end`, and `epoch`.
"""

import argparse
from pathlib import Path
from typing import Any

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, DEFAULT_NWB_ROOT, DEFAULT_POSITION_OFFSET
from v1ca1.sleep._session import (
    DEFAULT_SLEEP_DOWNSAMPLED_FREQUENCY_HZ,
    DEFAULT_SLEEP_MAX_GAP_S,
    DEFAULT_SLEEP_MIN_DURATION_S,
    DEFAULT_SLEEP_PC1_THRESHOLD,
    DEFAULT_SLEEP_SPEED_THRESHOLD_CM_S,
    DEFAULT_SLEEP_SPECTROGRAM_NOVERLAP,
    DEFAULT_SLEEP_SPECTROGRAM_NPERSEG,
    DEFAULT_TIME_BIN_SIZE_S,
    DEFAULT_V1_LFP_CHANNEL,
    build_uniform_time_grid,
    butter_filter_and_decimate,
    compute_sleep_time_intervals,
    compute_spectrogram_principal_component,
    get_analysis_path,
    get_epoch_trace,
    get_nwb_path,
    get_recording_sampling_frequency,
    get_speed_trace,
    interpolate_to_grid,
    load_recording,
    load_sleep_session_inputs,
    save_interval_table_output,
    validate_epochs,
    validate_recording_channel,
    validate_selected_epochs_across_sources,
)


def select_sleep_ready_epochs(
    selected_epochs: list[str],
    *,
    timestamps_position: dict[str, Any],
    position_by_epoch: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Return epochs that have both position timestamps and position samples."""
    selected_ready = [
        epoch
        for epoch in selected_epochs
        if epoch in timestamps_position and epoch in position_by_epoch
    ]
    skipped_epochs = [epoch for epoch in selected_epochs if epoch not in selected_ready]
    return selected_ready, skipped_epochs


def get_sleep_times_for_session(
    *,
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    epochs: list[str] | None = None,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SLEEP_SPEED_THRESHOLD_CM_S,
    pc1_threshold: float = DEFAULT_SLEEP_PC1_THRESHOLD,
    min_duration_s: float = DEFAULT_SLEEP_MIN_DURATION_S,
    max_gap_s: float = DEFAULT_SLEEP_MAX_GAP_S,
    v1_lfp_channel: int = DEFAULT_V1_LFP_CHANNEL,
) -> dict[str, Any]:
    """Compute and save session sleep intervals as one root-level parquet table."""
    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if min_duration_s < 0:
        raise ValueError("--min-duration-s must be non-negative.")
    if max_gap_s < 0:
        raise ValueError("--max-gap-s must be non-negative.")

    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    session = load_sleep_session_inputs(analysis_path)
    selected_epochs = validate_epochs(session["epoch_tags"], epochs)
    selected_epochs, skipped_epochs_missing_position = select_sleep_ready_epochs(
        selected_epochs,
        timestamps_position=session["timestamps_position"],
        position_by_epoch=session["position_by_epoch"],
    )
    if skipped_epochs_missing_position:
        print(
            "Skipping epochs missing position timestamps or position samples: "
            f"{skipped_epochs_missing_position!r}"
        )
    if not selected_epochs:
        raise ValueError("No selected epochs have both position timestamps and position samples.")
    validate_selected_epochs_across_sources(
        selected_epochs,
        source_epochs={
            "timestamps_ephys": session["timestamps_ephys"],
            "timestamps_position": session["timestamps_position"],
            "position": session["position_by_epoch"],
        },
    )

    recording = load_recording(get_nwb_path(animal_name=animal_name, date=date, nwb_root=nwb_root))
    validated_v1_channel = validate_recording_channel(
        recording,
        v1_lfp_channel,
        channel_name="V1 LFP channel",
    )
    sampling_frequency = get_recording_sampling_frequency(recording)

    print(f"Processing {animal_name} {date}.")

    intervals_by_epoch: dict[str, Any] = {}
    epoch_summaries: dict[str, dict[str, float]] = {}
    for epoch in selected_epochs:
        position = session["position_by_epoch"][epoch]
        timestamps_position = session["timestamps_position"][epoch]
        _trimmed_position, position_time, speed = get_speed_trace(
            position,
            timestamps_position,
            position_offset=position_offset,
        )
        lfp_time, v1_signal = get_epoch_trace(
            recording,
            epoch_timestamps=session["timestamps_ephys"][epoch],
            timestamps_ephys_all=session["timestamps_ephys_all"],
            channel_id=validated_v1_channel,
        )
        decimated_time, decimated_signal = butter_filter_and_decimate(
            lfp_time,
            v1_signal,
            sampling_frequency=sampling_frequency,
            new_sampling_frequency=DEFAULT_SLEEP_DOWNSAMPLED_FREQUENCY_HZ,
            cutoff_hz=150.0,
        )
        spectrogram_time, v1_pc1 = compute_spectrogram_principal_component(
            decimated_signal,
            DEFAULT_SLEEP_DOWNSAMPLED_FREQUENCY_HZ,
            max_frequency_hz=60.0,
            nperseg=DEFAULT_SLEEP_SPECTROGRAM_NPERSEG,
            noverlap=DEFAULT_SLEEP_SPECTROGRAM_NOVERLAP,
        )
        time_grid = build_uniform_time_grid(
            float(position_time[0]),
            float(position_time[-1]),
            1.0 / DEFAULT_TIME_BIN_SIZE_S,
        )
        pc1_interp = interpolate_to_grid(decimated_time[0] + spectrogram_time, v1_pc1, time_grid)
        speed_interp = interpolate_to_grid(position_time, speed, time_grid)
        sleep_intervals = compute_sleep_time_intervals(
            time_grid,
            pc1_interp,
            speed_interp,
            pc1_threshold=pc1_threshold,
            speed_threshold_cm_s=speed_threshold_cm_s,
            min_duration_s=min_duration_s,
            max_gap_s=max_gap_s,
        )
        intervals_by_epoch[epoch] = sleep_intervals
        epoch_summaries[epoch] = {
            "sleep_interval_count": float(sleep_intervals.shape[0]),
            "sleep_total_duration_s": float(
                0.0 if sleep_intervals.size == 0 else (sleep_intervals[:, 1] - sleep_intervals[:, 0]).sum()
            ),
            "grid_sample_count": float(time_grid.shape[0]),
            "position_sample_count": float(position_time.shape[0]),
            "spectrogram_sample_count": float(v1_pc1.shape[0]),
        }

    output_path = save_interval_table_output(
        analysis_path,
        output_name="sleep_times",
        intervals_by_epoch=intervals_by_epoch,
    )
    outputs = {
        "sleep_times_path": output_path,
        "selected_epochs": selected_epochs,
        "skipped_epochs_missing_position": skipped_epochs_missing_position,
        "sources": session["sources"],
        "epoch_summaries": epoch_summaries,
        "v1_lfp_channel": validated_v1_channel,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.sleep.get_sleep_times",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "epochs": epochs,
            "position_offset": position_offset,
            "speed_threshold_cm_s": speed_threshold_cm_s,
            "pc1_threshold": pc1_threshold,
            "min_duration_s": min_duration_s,
            "max_gap_s": max_gap_s,
            "v1_lfp_channel": validated_v1_channel,
        },
        outputs=outputs,
    )
    print(f"Saved sleep intervals to {output_path}")
    print(f"Saved run metadata to {log_path}")
    outputs["log_path"] = log_path
    outputs["intervals_by_epoch"] = intervals_by_epoch
    return outputs


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for sleep interval extraction."""
    parser = argparse.ArgumentParser(description="Compute sleep intervals for one session")
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
            "Number of leading position samples to ignore per epoch when defining speed. "
            f"Default: {DEFAULT_POSITION_OFFSET}"
        ),
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SLEEP_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define low-movement periods. "
            f"Default: {DEFAULT_SLEEP_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--pc1-threshold",
        type=float,
        default=DEFAULT_SLEEP_PC1_THRESHOLD,
        help=(
            "Threshold on the V1 spectrogram PC1 used to define sleep. "
            f"Default: {DEFAULT_SLEEP_PC1_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=DEFAULT_SLEEP_MIN_DURATION_S,
        help=f"Minimum interval duration in seconds. Default: {DEFAULT_SLEEP_MIN_DURATION_S}",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=DEFAULT_SLEEP_MAX_GAP_S,
        help=(
            "Merge adjacent sleep intervals when the gap is at most this many seconds. "
            f"Default: {DEFAULT_SLEEP_MAX_GAP_S}"
        ),
    )
    parser.add_argument(
        "--v1-lfp-channel",
        type=int,
        default=DEFAULT_V1_LFP_CHANNEL,
        help=f"Recording channel id used for the V1 LFP trace. Default: {DEFAULT_V1_LFP_CHANNEL}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the sleep interval extraction CLI."""
    args = parse_arguments(argv)
    get_sleep_times_for_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        epochs=args.epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        pc1_threshold=args.pc1_threshold,
        min_duration_s=args.min_duration_s,
        max_gap_s=args.max_gap_s,
        v1_lfp_channel=args.v1_lfp_channel,
    )


if __name__ == "__main__":
    main()
