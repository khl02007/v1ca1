from __future__ import annotations

"""Compute theta-band LFP and Hilbert phase for one session."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
)
from v1ca1.oscillation._channels import get_session_theta_channel

DEFAULT_LOWCUT_HZ = 4.0
DEFAULT_HIGHCUT_HZ = 12.0
DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY = 1000.0
OUTPUT_DIRNAME = "oscillation"
THETA_LFP_DIRNAME = "theta_lfp"
THETA_PHASE_DIRNAME = "theta_phase"
THETA_METADATA_FILENAME = "theta_metadata.json"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for theta phase extraction."""
    parser = argparse.ArgumentParser(description="Compute theta-band LFP and phase")
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
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
        "--theta-channel",
        type=int,
        help=(
            "Recording channel id used for theta extraction. "
            "Default: use the session-specific registry when available."
        ),
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        help="Optional subset of epoch labels to process. Default: all saved epochs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing theta outputs if they already exist.",
    )
    return parser.parse_args()


def validate_epochs(
    available_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return selected epochs after validating an optional user subset."""
    if requested_epochs is None:
        return available_epochs

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in available_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in the saved session epochs {available_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return requested_epochs


def load_theta_inputs(
    analysis_path: Path,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray, dict[str, str]]:
    """Load per-epoch and concatenated ephys timestamps for one session."""
    epoch_tags, timestamps_by_epoch, timestamps_source = load_ephys_timestamps_by_epoch(
        analysis_path
    )
    timestamps_ephys_all, timestamps_all_source = load_ephys_timestamps_all(analysis_path)
    return (
        epoch_tags,
        timestamps_by_epoch,
        timestamps_ephys_all,
        {
            "timestamps_ephys": timestamps_source,
            "timestamps_ephys_all": timestamps_all_source,
        },
    )


def load_theta_filtered_recording(
    nwb_path: Path,
    *,
    lowcut_hz: float = DEFAULT_LOWCUT_HZ,
    highcut_hz: float = DEFAULT_HIGHCUT_HZ,
) -> Any:
    """Load the NWB recording and wrap it with a theta-band filter."""
    try:
        import spikeinterface.full as si
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface.full is required to load NWB recordings for theta extraction."
        ) from exc

    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    recording = si.read_nwb_recording(nwb_path)
    return si.bandpass_filter(
        recording,
        freq_min=lowcut_hz,
        freq_max=highcut_hz,
        dtype=np.float64,
    )


def validate_theta_channel(recording: Any, theta_channel: int) -> int:
    """Validate that the requested theta channel exists in the recording."""
    channel_ids = np.asarray(recording.get_channel_ids())
    if channel_ids.size == 0:
        raise ValueError("Recording does not contain any channel ids.")
    if theta_channel not in channel_ids.tolist():
        raise ValueError(
            f"Theta channel {theta_channel} was not found in the recording channel ids "
            f"{channel_ids.tolist()!r}."
        )
    return int(theta_channel)


def resolve_theta_channel(
    animal_name: str,
    date: str,
    theta_channel: int | None,
) -> tuple[int, str]:
    """Resolve the theta channel from the CLI or the session registry."""
    if theta_channel is not None:
        return int(theta_channel), "argument"

    try:
        return get_session_theta_channel(animal_name=animal_name, date=date), "registry"
    except KeyError as exc:
        raise ValueError(
            "No theta channel was provided and no session mapping is configured. "
            "Pass --theta-channel or add the session to "
            "v1ca1.oscillation._channels.THETA_CHANNEL_BY_SESSION."
        ) from exc


def compute_instantaneous_phase(theta_lfp: np.ndarray) -> np.ndarray:
    """Return the Hilbert phase of one theta-band trace."""
    try:
        from scipy.signal import hilbert
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required to compute theta phase from theta-band LFP."
        ) from exc
    return np.angle(hilbert(np.asarray(theta_lfp, dtype=float)))


def get_recording_sampling_frequency(recording: Any) -> float:
    """Return the recording sampling frequency in Hz."""
    try:
        sampling_frequency = float(recording.get_sampling_frequency())
    except Exception as exc:
        raise ValueError(
            "Could not read the recording sampling frequency required for theta downsampling."
        ) from exc
    if sampling_frequency <= 0:
        raise ValueError(f"Recording sampling frequency must be positive, got {sampling_frequency}.")
    return sampling_frequency


def downsample_theta_trace(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    sampling_frequency: float,
    target_new_sampling_frequency: float = DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downsample one filtered theta trace to a lower sampling frequency."""
    try:
        import scipy.signal
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required to downsample theta traces."
        ) from exc

    time_array = np.asarray(timestamps, dtype=float)
    value_array = np.asarray(values, dtype=float).reshape(-1)
    if time_array.ndim != 1 or value_array.ndim != 1:
        raise ValueError("Theta timestamps and values must both be 1D.")
    if time_array.size != value_array.size:
        raise ValueError(
            "Theta timestamps and values must have matching lengths, got "
            f"{time_array.size} and {value_array.size}."
        )

    fs = float(sampling_frequency)
    target_fs = float(target_new_sampling_frequency)
    if target_fs <= 0 or target_fs > fs:
        raise ValueError("target_new_sampling_frequency must be in (0, sampling_frequency].")

    q = max(int(round(fs / target_fs)), 1)
    actual_new_fs = fs / q
    if q == 1:
        return time_array, value_array, actual_new_fs

    decimated_values = np.asarray(
        scipy.signal.decimate(value_array, q, ftype="iir", zero_phase=True),
        dtype=float,
    )
    decimated_timestamps = np.asarray(time_array[::q], dtype=float)
    sample_count = min(decimated_timestamps.size, decimated_values.size)
    return (
        decimated_timestamps[:sample_count],
        decimated_values[:sample_count],
        actual_new_fs,
    )


def extract_theta_for_epoch(
    recording: Any,
    *,
    epoch: str,
    epoch_timestamps: np.ndarray,
    timestamps_ephys_all: np.ndarray,
    theta_channel: int,
    target_new_sampling_frequency: float = DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract one epoch of theta-band LFP and phase from a filtered recording."""
    epoch_timestamps_array = np.asarray(epoch_timestamps, dtype=float)
    if epoch_timestamps_array.ndim != 1 or epoch_timestamps_array.size == 0:
        raise ValueError(f"Expected non-empty 1D timestamps for epoch {epoch!r}.")

    start_frame = int(
        np.searchsorted(timestamps_ephys_all, float(epoch_timestamps_array[0]), side="left")
    )
    end_frame = int(
        np.searchsorted(timestamps_ephys_all, float(epoch_timestamps_array[-1]), side="right")
    )

    traces = np.asarray(
        recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[theta_channel],
            return_in_uV=True,
        ),
        dtype=float,
    ).reshape(-1)
    timestamps_epoch = np.asarray(timestamps_ephys_all[start_frame:end_frame], dtype=float)

    sample_count = min(timestamps_epoch.size, traces.size)
    timestamps_epoch = timestamps_epoch[:sample_count]
    theta_lfp = traces[:sample_count]
    if timestamps_epoch.size == 0:
        raise ValueError(f"No theta samples were extracted for epoch {epoch!r}.")

    sampling_frequency = get_recording_sampling_frequency(recording)
    timestamps_epoch, theta_lfp, output_sampling_frequency = downsample_theta_trace(
        timestamps_epoch,
        theta_lfp,
        sampling_frequency=sampling_frequency,
        target_new_sampling_frequency=target_new_sampling_frequency,
    )
    theta_phase = compute_instantaneous_phase(theta_lfp)
    if theta_phase.shape != theta_lfp.shape:
        raise ValueError(
            f"Theta phase shape {theta_phase.shape} does not match theta LFP shape "
            f"{theta_lfp.shape} for epoch {epoch!r}."
        )
    return timestamps_epoch, theta_lfp, theta_phase, output_sampling_frequency


def get_theta_output_paths(
    analysis_path: Path,
    epochs: list[str],
) -> dict[str, Any]:
    """Return all theta output paths for one run."""
    output_dir = analysis_path / OUTPUT_DIRNAME
    theta_lfp_dir = output_dir / THETA_LFP_DIRNAME
    theta_phase_dir = output_dir / THETA_PHASE_DIRNAME
    return {
        "output_dir": output_dir,
        "theta_lfp_dir": theta_lfp_dir,
        "theta_phase_dir": theta_phase_dir,
        "theta_lfp_npz_paths": {epoch: theta_lfp_dir / f"{epoch}.npz" for epoch in epochs},
        "theta_phase_npz_paths": {epoch: theta_phase_dir / f"{epoch}.npz" for epoch in epochs},
        "metadata_path": output_dir / THETA_METADATA_FILENAME,
    }


def ensure_output_paths(
    output_paths: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Create output directories and validate overwrite policy."""
    path_candidates = [
        output_paths["metadata_path"],
        *output_paths["theta_lfp_npz_paths"].values(),
        *output_paths["theta_phase_npz_paths"].values(),
    ]
    existing_paths = [path for path in path_candidates if path.exists()]
    if existing_paths and not overwrite:
        raise FileExistsError(
            "Theta outputs already exist. Pass --overwrite to replace them. "
            f"First existing path: {existing_paths[0]}"
        )

    output_paths["theta_lfp_dir"].mkdir(parents=True, exist_ok=True)
    output_paths["theta_phase_dir"].mkdir(parents=True, exist_ok=True)


def save_theta_series_npz(
    output_path: Path,
    *,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> Path:
    """Write one pynapple-backed theta time series artifact."""
    try:
        import pynapple as nap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pynapple is required to save theta time-series artifacts as .npz files."
        ) from exc

    time_array = np.asarray(timestamps, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if time_array.ndim != 1 or value_array.ndim != 1:
        raise ValueError("Theta timestamps and values must both be 1D.")
    if time_array.size != value_array.size:
        raise ValueError(
            "Theta timestamps and values must have matching lengths, got "
            f"{time_array.size} and {value_array.size}."
        )

    nap.Tsd(t=time_array, d=value_array, time_units="s").save(output_path)
    return output_path


def build_theta_metadata(
    *,
    animal_name: str,
    date: str,
    theta_channel: int,
    theta_channel_source: str,
    epochs: list[str],
    timestamps: dict[str, np.ndarray],
    output_paths: dict[str, Any],
    sources: dict[str, str],
    lowcut_hz: float,
    highcut_hz: float,
    target_new_sampling_frequency: float,
    output_sampling_frequency: float,
) -> dict[str, Any]:
    """Build one JSON-safe metadata manifest for theta outputs."""
    return {
        "animal_name": str(animal_name),
        "date": str(date),
        "theta_channel": int(theta_channel),
        "theta_channel_source": str(theta_channel_source),
        "theta_band_hz": [float(lowcut_hz), float(highcut_hz)],
        "target_output_sampling_frequency_hz": float(target_new_sampling_frequency),
        "output_sampling_frequency_hz": float(output_sampling_frequency),
        "epochs": list(epochs),
        "sources": dict(sources),
        "epoch_time_bounds_s": {
            epoch: [float(np.asarray(timestamps[epoch], dtype=float)[0]), float(np.asarray(timestamps[epoch], dtype=float)[-1])]
            for epoch in epochs
        },
        "theta_lfp_npz_paths": {
            epoch: str(path) for epoch, path in output_paths["theta_lfp_npz_paths"].items()
        },
        "theta_phase_npz_paths": {
            epoch: str(path) for epoch, path in output_paths["theta_phase_npz_paths"].items()
        },
    }


def save_theta_metadata(output_path: Path, metadata: dict[str, Any]) -> Path:
    """Write one theta metadata manifest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    return output_path


def get_theta_phase_for_session(
    *,
    animal_name: str,
    date: str,
    theta_channel: int | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    epochs: list[str] | None = None,
    overwrite: bool = False,
    lowcut_hz: float = DEFAULT_LOWCUT_HZ,
    highcut_hz: float = DEFAULT_HIGHCUT_HZ,
    target_new_sampling_frequency: float = DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
) -> dict[str, Any]:
    """Compute theta-band LFP and Hilbert phase for one session."""
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    analysis_path.mkdir(parents=True, exist_ok=True)
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    resolved_theta_channel, theta_channel_source = resolve_theta_channel(
        animal_name=animal_name,
        date=date,
        theta_channel=theta_channel,
    )

    available_epochs, timestamps_by_epoch, timestamps_ephys_all, sources = load_theta_inputs(
        analysis_path
    )
    selected_epochs = validate_epochs(available_epochs, epochs)

    output_paths = get_theta_output_paths(analysis_path, selected_epochs)
    ensure_output_paths(output_paths, overwrite=overwrite)

    filtered_recording = load_theta_filtered_recording(
        nwb_path,
        lowcut_hz=lowcut_hz,
        highcut_hz=highcut_hz,
    )
    validated_channel = validate_theta_channel(filtered_recording, resolved_theta_channel)

    output_sampling_frequency: float | None = None
    for epoch in selected_epochs:
        (
            timestamps_epoch,
            theta_lfp_epoch,
            theta_phase_epoch,
            epoch_output_sampling_frequency,
        ) = extract_theta_for_epoch(
            filtered_recording,
            epoch=epoch,
            epoch_timestamps=timestamps_by_epoch[epoch],
            timestamps_ephys_all=timestamps_ephys_all,
            theta_channel=validated_channel,
            target_new_sampling_frequency=target_new_sampling_frequency,
        )
        if output_sampling_frequency is None:
            output_sampling_frequency = epoch_output_sampling_frequency
        elif not np.isclose(output_sampling_frequency, epoch_output_sampling_frequency):
            raise ValueError(
                "Theta output sampling frequency changed across epochs: "
                f"{output_sampling_frequency} vs {epoch_output_sampling_frequency}."
            )
        save_theta_series_npz(
            output_paths["theta_lfp_npz_paths"][epoch],
            timestamps=timestamps_epoch,
            values=theta_lfp_epoch,
        )
        save_theta_series_npz(
            output_paths["theta_phase_npz_paths"][epoch],
            timestamps=timestamps_epoch,
            values=theta_phase_epoch,
        )

    metadata = build_theta_metadata(
        animal_name=animal_name,
        date=date,
        theta_channel=validated_channel,
        theta_channel_source=theta_channel_source,
        epochs=selected_epochs,
        timestamps=timestamps_by_epoch,
            output_paths=output_paths,
            sources=sources,
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz,
            target_new_sampling_frequency=target_new_sampling_frequency,
            output_sampling_frequency=(
                float(output_sampling_frequency)
                if output_sampling_frequency is not None
                else float(target_new_sampling_frequency)
            ),
    )
    metadata_path = save_theta_metadata(output_paths["metadata_path"], metadata)

    outputs = {
        "analysis_path": analysis_path,
        "selected_epochs": selected_epochs,
        "theta_channel": validated_channel,
        "theta_channel_source": theta_channel_source,
        "theta_band_hz": [float(lowcut_hz), float(highcut_hz)],
        "target_output_sampling_frequency_hz": float(target_new_sampling_frequency),
        "output_sampling_frequency_hz": (
            float(output_sampling_frequency)
            if output_sampling_frequency is not None
            else float(target_new_sampling_frequency)
        ),
        "sources": sources,
        "theta_lfp_npz_paths": output_paths["theta_lfp_npz_paths"],
        "theta_phase_npz_paths": output_paths["theta_phase_npz_paths"],
        "metadata_path": metadata_path,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.oscillation.get_theta_phase",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "theta_channel": validated_channel,
            "theta_channel_source": theta_channel_source,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "epochs": selected_epochs,
            "overwrite": overwrite,
            "lowcut_hz": lowcut_hz,
            "highcut_hz": highcut_hz,
            "target_new_sampling_frequency": target_new_sampling_frequency,
        },
        outputs=outputs,
    )
    outputs["log_path"] = log_path
    print(f"Saved run metadata to {log_path}")
    return outputs


def main() -> None:
    """Run the theta phase extraction CLI."""
    args = parse_arguments()
    get_theta_phase_for_session(
        animal_name=args.animal_name,
        date=args.date,
        theta_channel=args.theta_channel,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        epochs=args.epochs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
