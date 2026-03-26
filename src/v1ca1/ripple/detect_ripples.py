from __future__ import annotations

"""Detect hippocampal ripples for one session.

This modernized CLI replaces the legacy ripple-detection lab script with
validated session loading, explicit CLI arguments, and modern pynapple outputs
under the session analysis directory.
"""

import argparse
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    get_analysis_path,
)
from v1ca1.ripple._channels import get_session_ripple_channels

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_ZSCORE_THRESHOLD = 2.0
DEFAULT_LOWCUT_HZ = 150.0
DEFAULT_HIGHCUT_HZ = 250.0
DEFAULT_FILTER_ORDER = 4
DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY = 1000.0
DEFAULT_NOTCH_BASE_FREQ = 60.0
DEFAULT_NOTCH_HARMONICS = 10
DEFAULT_NOTCH_QUALITY = 50.0
DEFAULT_ENABLE_NOTCH_FILTER = True
LFP_CACHE_DIRNAME = "ripple_channels_lfp"
LFP_CACHE_FORMAT = "ripple_channels_lfp_netcdf"
LFP_CACHE_FORMAT_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for ripple detection."""
    parser = argparse.ArgumentParser(description="Detect SWR ripples for one session")
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
        "--epochs",
        nargs="+",
        help="Optional subset of epoch labels to process. Default: all saved epochs.",
    )
    parser.add_argument(
        "--zscore-threshold",
        type=float,
        default=DEFAULT_ZSCORE_THRESHOLD,
        help=f"Ripple detector z-score threshold. Default: {DEFAULT_ZSCORE_THRESHOLD}",
    )
    parser.add_argument(
        "--disable-speed-gating",
        action="store_true",
        help="Disable speed gating and write the no-speed detector outputs.",
    )
    parser.add_argument(
        "--disable-notch-filter",
        action="store_true",
        help="Disable notch filtering before ripple-band extraction. Default: enabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute the cached ripple-band LFP even if it already exists.",
    )
    return parser.parse_args()


def validate_epochs(
    available_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return the selected epoch list after validating any user subset."""
    if requested_epochs is None:
        return available_epochs

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in available_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in the saved session epochs {available_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return requested_epochs


def validate_selected_epochs_across_sources(
    selected_epochs: list[str],
    *,
    source_epochs: dict[str, list[str] | dict[str, Any]],
) -> None:
    """Ensure every selected epoch exists in each required per-epoch source."""
    missing_by_source: dict[str, list[str]] = {}
    for source_name, source in source_epochs.items():
        available_epochs = set(source.keys() if isinstance(source, dict) else source)
        missing_epochs = [epoch for epoch in selected_epochs if epoch not in available_epochs]
        if missing_epochs:
            missing_by_source[source_name] = missing_epochs

    if missing_by_source:
        details = "; ".join(
            f"{source_name}: {missing_epochs!r}"
            for source_name, missing_epochs in missing_by_source.items()
        )
        raise ValueError(
            "Selected epochs are missing required session inputs. "
            f"Missing epochs by source: {details}"
        )


def select_speed_gated_epochs(
    available_epochs: list[str],
    *,
    requested_epochs: list[str] | None,
    position_timestamp_epochs: dict[str, np.ndarray],
    clean_dlc_epochs: list[str],
) -> list[str]:
    """Select speed-gated epochs, filtering incomplete epochs only for implicit runs."""
    if requested_epochs is not None:
        return validate_epochs(available_epochs, requested_epochs)

    clean_dlc_epoch_set = set(clean_dlc_epochs)
    selected_epochs = [
        epoch
        for epoch in available_epochs
        if epoch in position_timestamp_epochs and epoch in clean_dlc_epoch_set
    ]
    if not selected_epochs:
        raise ValueError(
            "No epochs have both cleaned position data and position timestamps for "
            "speed-gated ripple detection."
        )
    return selected_epochs


def validate_ripple_channels(channels: list[int]) -> list[int]:
    """Return a validated list of ripple channel ids."""
    if not channels:
        raise ValueError("At least one ripple channel is required.")
    if any(channel < 0 for channel in channels):
        raise ValueError(f"Ripple channels must be non-negative integers: {channels!r}")
    return list(dict.fromkeys(int(channel) for channel in channels))


def get_ripple_channels_for_session(animal_name: str, date: str) -> list[int]:
    """Return the configured ripple channels for one session from the code registry."""
    try:
        configured_channels = get_session_ripple_channels(animal_name=animal_name, date=date)
    except KeyError as exc:
        raise ValueError(
            "No ripple channels are configured for this session. "
            "Add the animal/date entry to "
            "v1ca1.ripple._channels.RIPPLE_CHANNELS_BY_SESSION."
        ) from exc
    return validate_ripple_channels(configured_channels)


def get_output_paths(output_dir: Path, use_speed_gating: bool) -> dict[str, Path]:
    """Return the ripple event-table and LFP cache paths for one detector mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_speed_gating:
        stem = "ripple_times"
    else:
        stem = "ripple_times_no_speed"
    return {
        "interval_parquet": output_dir / f"{stem}.parquet",
        "lfp_cache_dir": output_dir / LFP_CACHE_DIRNAME,
    }


def get_epoch_lfp_cache_path(cache_dir: Path, epoch: str) -> Path:
    """Return the NetCDF cache path for one epoch."""
    return cache_dir / f"{epoch}_ripple_channels_lfp.nc"


def build_epoch_lfp_dataset(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    timestamps: np.ndarray,
    filtered_lfp: np.ndarray,
    sampling_frequency: float,
    channel_ids: list[int],
    enable_notch_filter: bool,
):
    """Build one epoch-level ripple-band LFP cache dataset."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to store ripple-band LFP caches as NetCDF files."
        ) from exc

    time_array = np.asarray(timestamps, dtype=float)
    lfp_array = np.asarray(filtered_lfp, dtype=float)
    channel_array = np.asarray(channel_ids, dtype=int)
    if lfp_array.ndim != 2:
        raise ValueError(
            f"filtered_lfp must be a 2D array, got shape {lfp_array.shape} for epoch {epoch!r}."
        )
    if time_array.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape {time_array.shape} for epoch {epoch!r}.")
    if lfp_array.shape[0] != time_array.size:
        raise ValueError(
            "filtered_lfp sample count does not match timestamps for epoch "
            f"{epoch!r}: {lfp_array.shape[0]} vs {time_array.size}."
        )
    if lfp_array.shape[1] != channel_array.size:
        raise ValueError(
            "filtered_lfp channel count does not match channel_ids for epoch "
            f"{epoch!r}: {lfp_array.shape[1]} vs {channel_array.size}."
        )

    return xr.Dataset(
        data_vars={
            "filtered_lfp": (("sample", "channel"), lfp_array),
            "sampling_frequency_hz": ((), float(sampling_frequency)),
        },
        coords={
            "time": ("sample", time_array),
            "channel": ("channel", channel_array),
        },
        attrs={
            "animal_name": str(animal_name),
            "date": str(date),
            "epoch": str(epoch),
            "notch_filter_enabled": int(bool(enable_notch_filter)),
            "notch_base_freq_hz": float(DEFAULT_NOTCH_BASE_FREQ),
            "notch_harmonics": int(DEFAULT_NOTCH_HARMONICS),
            "notch_quality": float(DEFAULT_NOTCH_QUALITY),
            "lowcut_hz": float(DEFAULT_LOWCUT_HZ),
            "highcut_hz": float(DEFAULT_HIGHCUT_HZ),
            "target_sampling_frequency_hz": float(DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY),
            "cache_format": LFP_CACHE_FORMAT,
            "cache_format_version": int(LFP_CACHE_FORMAT_VERSION),
        },
    )


def save_epoch_lfp_dataset(path: Path, dataset: Any) -> Path:
    """Save one epoch-level ripple-band LFP cache as NetCDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path, engine="scipy")
    return path


def load_epoch_lfp_dataset(path: Path):
    """Load one epoch-level ripple-band LFP cache dataset."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to load ripple-band LFP caches stored as NetCDF files."
        ) from exc
    return xr.load_dataset(path, engine="scipy")


def epoch_lfp_dataset_matches_request(
    dataset: Any,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    channel_ids: list[int],
    enable_notch_filter: bool,
) -> bool:
    """Return whether one cached epoch dataset matches the current request."""
    if "filtered_lfp" not in dataset.data_vars or "sampling_frequency_hz" not in dataset.data_vars:
        return False
    if "time" not in dataset.coords or "channel" not in dataset.coords:
        return False
    if "sample" not in dataset.dims or "channel" not in dataset.dims:
        return False

    filtered_lfp = np.asarray(dataset["filtered_lfp"].values, dtype=float)
    time_array = np.asarray(dataset["time"].values, dtype=float)
    channel_array = np.asarray(dataset["channel"].values, dtype=int)
    if filtered_lfp.ndim != 2:
        return False
    if time_array.ndim != 1:
        return False
    if filtered_lfp.shape[0] != time_array.size:
        return False
    if filtered_lfp.shape[1] != channel_array.size:
        return False
    if list(channel_array.tolist()) != list(channel_ids):
        return False

    attrs = dataset.attrs
    if attrs.get("cache_format") != LFP_CACHE_FORMAT:
        return False
    if int(attrs.get("cache_format_version", -1)) != LFP_CACHE_FORMAT_VERSION:
        return False
    if str(attrs.get("animal_name")) != str(animal_name):
        return False
    if str(attrs.get("date")) != str(date):
        return False
    if str(attrs.get("epoch")) != str(epoch):
        return False
    if int(attrs.get("notch_filter_enabled", -1)) != int(bool(enable_notch_filter)):
        return False
    if float(attrs.get("notch_base_freq_hz", np.nan)) != float(DEFAULT_NOTCH_BASE_FREQ):
        return False
    if int(attrs.get("notch_harmonics", -1)) != int(DEFAULT_NOTCH_HARMONICS):
        return False
    if float(attrs.get("notch_quality", np.nan)) != float(DEFAULT_NOTCH_QUALITY):
        return False
    if float(attrs.get("lowcut_hz", np.nan)) != float(DEFAULT_LOWCUT_HZ):
        return False
    if float(attrs.get("highcut_hz", np.nan)) != float(DEFAULT_HIGHCUT_HZ):
        return False
    if float(attrs.get("target_sampling_frequency_hz", np.nan)) != float(
        DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY
    ):
        return False
    return True


def _extract_interval_dataframe(intervals: Any):
    """Return a dataframe-like view of a pynapple IntervalSet."""
    if hasattr(intervals, "as_dataframe"):
        return intervals.as_dataframe()
    if hasattr(intervals, "_metadata"):
        return intervals._metadata.copy()  # type: ignore[attr-defined]
    raise ValueError("Could not read metadata from timestamps_ephys.npz.")


def _extract_epoch_tags_from_intervalset(epoch_intervals: Any) -> list[str]:
    """Extract saved epoch labels from a pynapple IntervalSet."""
    try:
        epoch_info = epoch_intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is not None:
        epoch_array = np.asarray(epoch_info)
        if epoch_array.size:
            return [str(epoch) for epoch in epoch_array.tolist()]

    interval_df = _extract_interval_dataframe(epoch_intervals)
    if "epoch" in interval_df.columns:
        return [str(epoch) for epoch in interval_df["epoch"].tolist()]

    raise ValueError("timestamps_ephys.npz does not contain saved epoch labels.")


def _extract_interval_bounds_from_intervalset(
    intervals: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned start/end arrays from a pynapple IntervalSet."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "timestamps_ephys.npz has mismatched start/end arrays: "
            f"{starts.shape} vs {ends.shape}."
        )
    return starts, ends


def _extract_epoch_tags_from_tsgroup(position_group: Any) -> list[str]:
    """Extract saved epoch labels from a pynapple TsGroup."""
    try:
        epoch_info = position_group["epoch"]
    except Exception:
        epoch_info = None

    if epoch_info is None:
        raise ValueError("timestamps_position.npz does not contain the saved epoch labels.")

    epoch_array = np.asarray(epoch_info)
    if epoch_array.size == 0:
        raise ValueError("timestamps_position.npz does not contain any saved epoch labels.")
    return [str(epoch) for epoch in epoch_array.tolist()]


def _load_required_npz(path: Path, artifact_name: str) -> Any:
    """Load one required pynapple-backed `.npz` artifact."""
    try:
        import pynapple as nap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"pynapple is required to read {artifact_name} for ripple detection."
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Required {artifact_name} not found: {path}")

    try:
        return nap.load_file(path)
    except Exception as exc:
        raise ValueError(f"Failed to load required {artifact_name}: {path}") from exc


def load_ephys_timestamps_all_npz(analysis_path: Path) -> np.ndarray:
    """Load concatenated ephys timestamps from the required `.npz` export."""
    npz_path = analysis_path / "timestamps_ephys_all.npz"
    timestamps_all = _load_required_npz(npz_path, "timestamps_ephys_all.npz")
    return np.asarray(timestamps_all.t, dtype=float)


def load_ephys_timestamps_by_epoch_npz(
    analysis_path: Path,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load per-epoch ephys timestamps from the required `.npz` export."""
    npz_path = analysis_path / "timestamps_ephys.npz"
    epoch_intervals = _load_required_npz(npz_path, "timestamps_ephys.npz")
    epoch_tags = _extract_epoch_tags_from_intervalset(epoch_intervals)
    starts, ends = _extract_interval_bounds_from_intervalset(epoch_intervals)
    if len(epoch_tags) != starts.size:
        raise ValueError(f"Mismatch between epoch labels and saved epoch intervals in {npz_path}.")

    timestamps_all = load_ephys_timestamps_all_npz(analysis_path)
    timestamps_by_epoch: dict[str, np.ndarray] = {}
    for epoch, start, end in zip(epoch_tags, starts, ends):
        start_index = int(np.searchsorted(timestamps_all, float(start), side="left"))
        end_index = int(np.searchsorted(timestamps_all, float(end), side="right"))
        epoch_timestamps = np.asarray(timestamps_all[start_index:end_index], dtype=float)
        if epoch_timestamps.size == 0:
            raise ValueError(
                "Could not reconstruct any ephys timestamps for epoch "
                f"{epoch!r} from {npz_path}."
            )
        timestamps_by_epoch[epoch] = epoch_timestamps
    return epoch_tags, timestamps_by_epoch


def load_position_timestamps_npz(analysis_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Load per-epoch position timestamps from the required `.npz` export."""
    npz_path = analysis_path / "timestamps_position.npz"
    position_group = _load_required_npz(npz_path, "timestamps_position.npz")
    epoch_tags = _extract_epoch_tags_from_tsgroup(position_group)
    if len(epoch_tags) != len(position_group):
        raise ValueError(
            f"Mismatch between epoch labels and time series count in {npz_path}."
        )
    timestamps_position = {
        epoch: np.asarray(position_group[index].t, dtype=float)
        for index, epoch in enumerate(epoch_tags)
    }
    return epoch_tags, timestamps_position


def load_clean_dlc_head_position(
    analysis_path: Path,
    *,
    selected_epochs: list[str],
    timestamps_position_by_epoch: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], Path]:
    """Load cleaned DLC head position from the combined session parquet."""
    import pandas as pd

    input_path = (
        analysis_path
        / DEFAULT_CLEAN_DLC_POSITION_DIRNAME
        / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    if not input_path.exists():
        raise FileNotFoundError(
            "Combined cleaned DLC position file not found: "
            f"{input_path}. Generate it with "
            f"`python -m v1ca1.position.combine_clean_dlc_position --animal-name {analysis_path.parent.name} "
            f"--date {analysis_path.name}`."
        )

    table = pd.read_parquet(input_path)
    required_columns = (
        "epoch",
        "frame",
        "frame_time_s",
        "head_x_cm",
        "head_y_cm",
    )
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "Combined cleaned DLC position parquet is missing required columns: "
            f"{missing_columns!r}"
        )
    if table.empty:
        raise ValueError(f"Combined cleaned DLC position parquet is empty: {input_path}")

    epoch_series = table["epoch"].astype(str)
    available_epochs = set(epoch_series.tolist())
    missing_epochs = [epoch for epoch in selected_epochs if epoch not in available_epochs]
    if missing_epochs:
        raise ValueError(
            "Selected epochs are missing required session inputs. "
            f"Missing epochs by source: clean_dlc_position: {missing_epochs!r}"
        )

    head_position: dict[str, np.ndarray] = {}
    for epoch in selected_epochs:
        epoch_table = table.loc[epoch_series == epoch].reset_index(drop=True)
        frame_numbers = epoch_table["frame"].to_numpy(dtype=int)
        if np.unique(frame_numbers).size != frame_numbers.size:
            raise ValueError(
                f"Combined cleaned DLC position contains duplicate frames for epoch {epoch!r}."
            )
        if frame_numbers.size > 1 and np.any(np.diff(frame_numbers) < 0):
            raise ValueError(
                f"Combined cleaned DLC position frames are not monotonic for epoch {epoch!r}."
            )

        frame_times = epoch_table["frame_time_s"].to_numpy(dtype=float)
        expected_times = np.asarray(timestamps_position_by_epoch[epoch], dtype=float)
        if frame_times.size != expected_times.size:
            raise ValueError(
                "Combined cleaned DLC position row count does not match saved position timestamps "
                f"for epoch {epoch!r}: {frame_times.size} vs {expected_times.size}."
            )
        if not np.allclose(frame_times, expected_times, rtol=0.0, atol=1e-9):
            raise ValueError(
                "Combined cleaned DLC position timestamps do not match saved position timestamps "
                f"for epoch {epoch!r}."
            )

        head_position[epoch] = epoch_table[["head_x_cm", "head_y_cm"]].to_numpy(dtype=float)

    return head_position, input_path


def load_available_clean_dlc_epochs(analysis_path: Path) -> tuple[list[str], Path]:
    """Return the cleaned DLC epochs available in the combined session parquet."""
    import pandas as pd

    input_path = (
        analysis_path
        / DEFAULT_CLEAN_DLC_POSITION_DIRNAME
        / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    if not input_path.exists():
        raise FileNotFoundError(
            "Combined cleaned DLC position file not found: "
            f"{input_path}. Generate it with "
            f"`python -m v1ca1.position.combine_clean_dlc_position --animal-name {analysis_path.parent.name} "
            f"--date {analysis_path.name}`."
        )

    table = pd.read_parquet(input_path)
    required_columns = (
        "epoch",
        "frame",
        "frame_time_s",
        "head_x_cm",
        "head_y_cm",
    )
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "Combined cleaned DLC position parquet is missing required columns: "
            f"{missing_columns!r}"
        )
    if table.empty:
        raise ValueError(f"Combined cleaned DLC position parquet is empty: {input_path}")

    epoch_series = table["epoch"].astype(str)
    available_epochs = list(dict.fromkeys(epoch_series.tolist()))
    return available_epochs, input_path


def get_recording(animal_name: str, date: str, nwb_root: Path):
    """Load the NWB recording for one session."""
    try:
        import spikeinterface.full as si
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface is required to read NWB recordings for ripple detection."
        ) from exc

    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")
    return si.read_nwb_recording(nwb_path)


def apply_notch_filters_multichannel(
    signal_data: np.ndarray,
    fs: float,
    base_freq: float = DEFAULT_NOTCH_BASE_FREQ,
    n_harmonics: int = DEFAULT_NOTCH_HARMONICS,
    quality: float = DEFAULT_NOTCH_QUALITY,
) -> np.ndarray:
    """Apply a stack of notch filters to a multichannel signal."""
    import scipy.signal

    filtered = np.asarray(signal_data, dtype=float)
    for harmonic in range(1, n_harmonics + 1):
        notch_freq = harmonic * base_freq
        if notch_freq >= fs / 2:
            break
        b, a = scipy.signal.iirnotch(w0=notch_freq, Q=quality, fs=fs)
        sos = scipy.signal.tf2sos(b, a)
        filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=0)
    return filtered


def butter_filter_and_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    sampling_frequency: float,
    target_new_sampling_frequency: float = DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
    lowcut: float = DEFAULT_LOWCUT_HZ,
    highcut: float = DEFAULT_HIGHCUT_HZ,
    order: int = DEFAULT_FILTER_ORDER,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Bandpass filter a multichannel trace and decimate it by stride."""
    import scipy.signal

    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a 1D array.")

    signal_data = np.asarray(data)
    if signal_data.ndim == 1:
        signal_data = signal_data[:, np.newaxis]

    if signal_data.shape[0] != timestamps.shape[0]:
        raise ValueError(
            "The first dimension of the signal must match the timestamp count. "
            f"Got {signal_data.shape[0]} and {timestamps.shape[0]}."
        )

    fs = float(sampling_frequency)
    target_fs = float(target_new_sampling_frequency)
    if target_fs <= 0 or target_fs > fs:
        raise ValueError("target_new_sampling_frequency must be in (0, sampling_frequency].")

    q = max(int(round(fs / target_fs)), 1)
    actual_new_fs = fs / q

    nyquist = 0.5 * fs
    low = max(1e-6, min(float(lowcut), nyquist * 0.999))
    high = max(low * 1.001, min(float(highcut), nyquist * 0.999))
    if high >= 0.5 * actual_new_fs:
        raise ValueError(
            f"Highcut {high} Hz is too high for the decimated Nyquist frequency "
            f"{0.5 * actual_new_fs} Hz."
        )

    sos = scipy.signal.butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    filtered = scipy.signal.sosfiltfilt(sos, signal_data, axis=0)

    decimated_timestamps = np.asarray(timestamps[::q], dtype=float)
    decimated_data = np.asarray(filtered[::q], dtype=float)
    sample_count = min(decimated_timestamps.shape[0], decimated_data.shape[0])
    return (
        decimated_timestamps[:sample_count],
        decimated_data[:sample_count],
        actual_new_fs,
    )


def filter_ripple_band_for_epoch(
    recording: Any,
    *,
    epoch: str,
    epoch_timestamps: np.ndarray,
    timestamps_ephys_all: np.ndarray,
    ripple_channels: list[int],
    enable_notch_filter: bool,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Extract ripple-band LFP for one epoch from the NWB recording."""
    start_frame = int(np.searchsorted(timestamps_ephys_all, float(epoch_timestamps[0]), side="left"))
    end_frame = int(np.searchsorted(timestamps_ephys_all, float(epoch_timestamps[-1]), side="right"))
    traces = np.asarray(
        recording.get_traces(
            channel_ids=ripple_channels,
            start_frame=start_frame,
            end_frame=end_frame,
            return_in_uV=True,
        ),
        dtype=float,
    )

    try:
        sampling_frequency = float(recording.get_sampling_frequency())
    except Exception:
        sampling_frequency = 30000.0

    timestamps_epoch = np.asarray(timestamps_ephys_all[start_frame:end_frame], dtype=float)
    sample_count = min(timestamps_epoch.shape[0], traces.shape[0])
    timestamps_epoch = timestamps_epoch[:sample_count]
    traces = traces[:sample_count]

    notch_applied = bool(enable_notch_filter)
    if notch_applied:
        traces = apply_notch_filters_multichannel(traces, fs=sampling_frequency)

    return (*butter_filter_and_decimate(timestamps_epoch, traces, sampling_frequency), notch_applied)


def initialize_lfp_cache(
    channel_ids: list[int],
    *,
    enable_notch_filter: bool,
) -> dict[str, Any]:
    """Return an empty in-memory ripple-band cache structure."""
    return {
        "time": {},
        "data": {},
        "fs": {},
        "channel_ids": list(channel_ids),
        "notch_filter_enabled": bool(enable_notch_filter),
        "notch_base_freq_hz": DEFAULT_NOTCH_BASE_FREQ,
        "notch_harmonics": DEFAULT_NOTCH_HARMONICS,
        "notch_quality": DEFAULT_NOTCH_QUALITY,
        "lowcut_hz": DEFAULT_LOWCUT_HZ,
        "highcut_hz": DEFAULT_HIGHCUT_HZ,
        "target_sampling_frequency_hz": DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY,
        "epoch_cache_actions": {},
    }


def record_lfp_cache_action(
    cache: dict[str, Any],
    *,
    epoch: str,
    action: str,
    reason: str,
    cache_path: Path,
) -> None:
    """Record and print one ripple LFP cache action for an epoch."""
    cache["epoch_cache_actions"][epoch] = {
        "action": str(action),
        "reason": str(reason),
        "cache_path": str(cache_path),
    }
    print(
        f"Ripple LFP cache {action} for {epoch}: {reason}. "
        f"Cache path: {cache_path}"
    )


def compute_or_load_ripple_lfp_cache(
    *,
    cache_dir: Path,
    animal_name: str,
    date: str,
    recording: Any,
    selected_epochs: list[str],
    timestamps_by_epoch: dict[str, np.ndarray],
    timestamps_ephys_all: np.ndarray,
    channel_ids: list[int],
    enable_notch_filter: bool,
    overwrite: bool,
) -> tuple[dict[str, Any], str]:
    """Load or compute the cached ripple-band LFP traces."""
    cache = initialize_lfp_cache(channel_ids, enable_notch_filter=enable_notch_filter)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for epoch in selected_epochs:
        cache_path = get_epoch_lfp_cache_path(cache_dir, epoch)
        if overwrite:
            if cache_path.exists():
                record_lfp_cache_action(
                    cache,
                    epoch=epoch,
                    action="recompute",
                    reason="overwrite requested; existing cache will be replaced",
                    cache_path=cache_path,
                )
            else:
                record_lfp_cache_action(
                    cache,
                    epoch=epoch,
                    action="compute",
                    reason="overwrite requested and no existing cache was found",
                    cache_path=cache_path,
                )
        elif cache_path.exists():
            try:
                dataset = load_epoch_lfp_dataset(cache_path)
            except Exception as exc:
                record_lfp_cache_action(
                    cache,
                    epoch=epoch,
                    action="recompute",
                    reason=f"existing cache could not be loaded ({type(exc).__name__}: {exc})",
                    cache_path=cache_path,
                )
            else:
                if epoch_lfp_dataset_matches_request(
                    dataset,
                    animal_name=animal_name,
                    date=date,
                    epoch=epoch,
                    channel_ids=channel_ids,
                    enable_notch_filter=enable_notch_filter,
                ):
                    cache["time"][epoch] = np.asarray(dataset["time"].values, dtype=float)
                    cache["data"][epoch] = np.asarray(dataset["filtered_lfp"].values, dtype=float)
                    cache["fs"][epoch] = float(dataset["sampling_frequency_hz"].values)
                    record_lfp_cache_action(
                        cache,
                        epoch=epoch,
                        action="reuse",
                        reason="existing cache matches the current request",
                        cache_path=cache_path,
                    )
                    continue
                record_lfp_cache_action(
                    cache,
                    epoch=epoch,
                    action="recompute",
                    reason="existing cache does not match the current request",
                    cache_path=cache_path,
                )
        else:
            record_lfp_cache_action(
                cache,
                epoch=epoch,
                action="compute",
                reason="no existing cache was found",
                cache_path=cache_path,
            )

        timestamps_decimated, filtered_lfp, actual_fs, _ = filter_ripple_band_for_epoch(
            recording,
            epoch=epoch,
            epoch_timestamps=timestamps_by_epoch[epoch],
            timestamps_ephys_all=timestamps_ephys_all,
            ripple_channels=channel_ids,
            enable_notch_filter=enable_notch_filter,
        )
        cache["time"][epoch] = timestamps_decimated
        cache["data"][epoch] = filtered_lfp
        cache["fs"][epoch] = actual_fs
        dataset = build_epoch_lfp_dataset(
            animal_name=animal_name,
            date=date,
            epoch=epoch,
            timestamps=timestamps_decimated,
            filtered_lfp=filtered_lfp,
            sampling_frequency=actual_fs,
            channel_ids=channel_ids,
            enable_notch_filter=enable_notch_filter,
        )
        save_epoch_lfp_dataset(cache_path, dataset)

    return cache, "netcdf"


def compute_speed(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    position_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute speed from one epoch of XY position data."""
    import position_tools as pt

    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if position.shape[0] <= position_offset or timestamps_position.size <= position_offset:
        raise ValueError(
            "Position offset removes all samples for one epoch. "
            f"position samples: {position.shape[0]}, timestamp samples: {timestamps_position.size}, "
            f"position_offset: {position_offset}"
        )
    if position.shape[0] != timestamps_position.size:
        raise ValueError(
            "Position samples and timestamps must have the same length. "
            f"Got {position.shape[0]} and {timestamps_position.size}."
        )

    epoch_position = np.asarray(position[position_offset:], dtype=float)
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    duration = float(epoch_timestamps[-1] - epoch_timestamps[0])
    if duration <= 0:
        raise ValueError("Position timestamps must span a positive duration.")
    position_sampling_rate = (len(epoch_timestamps) - 1) / duration
    speed = np.asarray(
        pt.get_speed(
            epoch_position,
            epoch_timestamps,
            sigma=DEFAULT_SPEED_SIGMA_S,
            sampling_frequency=position_sampling_rate,
        ),
        dtype=float,
    )
    return epoch_timestamps, speed


def interpolate_speed_to_lfp(
    lfp_timestamps: np.ndarray,
    position_timestamps: np.ndarray,
    speed: np.ndarray,
) -> np.ndarray:
    """Interpolate speed onto the ripple-band LFP timestamps."""
    import scipy.interpolate

    speed_interpolator = scipy.interpolate.interp1d(
        position_timestamps,
        speed,
        axis=0,
        bounds_error=False,
        kind="linear",
    )
    return np.asarray(speed_interpolator(lfp_timestamps), dtype=float)


def run_kay_ripple_detector(
    *,
    time: np.ndarray,
    filtered_lfps: np.ndarray,
    speed: np.ndarray,
    sampling_frequency: float,
    zscore_threshold: float,
) -> Any:
    """Run the Kay ripple detector with lazy dependency loading."""
    try:
        import ripple_detection as rd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ripple_detection is required to detect ripples. Install it in the active environment."
        ) from exc

    return rd.Kay_ripple_detector(
        time=time,
        filtered_lfps=filtered_lfps,
        speed=speed,
        sampling_frequency=sampling_frequency,
        zscore_threshold=zscore_threshold,
    )


def normalize_ripple_table(result: Any, epoch: str) -> "pd.DataFrame":
    """Normalize one detector result into a pandas DataFrame."""
    import pandas as pd

    if result is None or (isinstance(result, dict) and not result):
        dataframe = pd.DataFrame(columns=["start_time", "end_time"])
    elif isinstance(result, pd.DataFrame):
        dataframe = result.copy()
    elif hasattr(result, "to_dataframe"):
        dataframe = result.to_dataframe()
    else:
        dataframe = pd.DataFrame(result)

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Ripple detector output for epoch {epoch!r} could not be normalized to a DataFrame."
        )

    dataframe = dataframe.reset_index(drop=True)
    if dataframe.empty:
        for column_name in ("start_time", "end_time"):
            if column_name not in dataframe.columns:
                dataframe[column_name] = pd.Series(dtype=float)

    missing_columns = [
        column_name
        for column_name in ("start_time", "end_time")
        if column_name not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Ripple detector output for epoch {epoch!r} is missing required columns: "
            f"{missing_columns!r}."
        )

    dataframe["start_time"] = np.asarray(dataframe["start_time"], dtype=float)
    dataframe["end_time"] = np.asarray(dataframe["end_time"], dtype=float)
    return dataframe


def detect_ripples_for_epoch(
    *,
    epoch: str,
    ripple_lfp_cache: dict[str, Any],
    position_by_epoch: dict[str, np.ndarray],
    position_timestamps_by_epoch: dict[str, np.ndarray],
    position_offset: int,
    zscore_threshold: float,
    use_speed_gating: bool,
) -> "pd.DataFrame":
    """Run ripple detection for one epoch and return a normalized event table."""
    lfp_timestamps = np.asarray(ripple_lfp_cache["time"][epoch], dtype=float)
    filtered_lfps = np.asarray(ripple_lfp_cache["data"][epoch], dtype=float)
    sampling_frequency = float(ripple_lfp_cache["fs"][epoch])

    if use_speed_gating:
        position_timestamps, speed = compute_speed(
            position_by_epoch[epoch],
            position_timestamps_by_epoch[epoch],
            position_offset,
        )
        speed_interp = interpolate_speed_to_lfp(lfp_timestamps, position_timestamps, speed)
    else:
        speed_interp = np.zeros(lfp_timestamps.shape[0], dtype=float)

    detector_result = run_kay_ripple_detector(
        time=lfp_timestamps,
        filtered_lfps=filtered_lfps,
        speed=speed_interp,
        sampling_frequency=sampling_frequency,
        zscore_threshold=zscore_threshold,
    )
    return normalize_ripple_table(detector_result, epoch=epoch)


def flatten_ripple_tables(ripple_tables_by_epoch: dict[str, "pd.DataFrame"]) -> "pd.DataFrame":
    """Flatten per-epoch detector tables into one event table with an epoch column."""
    import pandas as pd

    tables: list[pd.DataFrame] = []
    for epoch, table in ripple_tables_by_epoch.items():
        epoch_table = table.copy()
        epoch_table.insert(0, "epoch", epoch)
        tables.append(epoch_table)

    if not tables:
        return pd.DataFrame(columns=["epoch", "start_time", "end_time"])
    return pd.concat(tables, ignore_index=True, sort=False)

def save_ripple_interval_output(path: Path, flat_table: "pd.DataFrame") -> Path:
    """Write the flattened ripple event table as one canonical parquet table."""
    output_table = flat_table.copy()
    rename_columns = {}
    if "start_time" in output_table.columns:
        rename_columns["start_time"] = "start"
    if "end_time" in output_table.columns:
        rename_columns["end_time"] = "end"
    output_table = output_table.rename(columns=rename_columns)

    preferred_columns = ["start", "end", "epoch"]
    ordered_columns = [column for column in preferred_columns if column in output_table.columns]
    ordered_columns.extend(
        column for column in output_table.columns if column not in ordered_columns
    )
    output_table = output_table.loc[:, ordered_columns]

    sort_columns = [column for column in ["start", "end", "epoch"] if column in output_table.columns]
    if sort_columns:
        output_table = output_table.sort_values(by=sort_columns, kind="mergesort").reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    output_table.to_parquet(path, index=False)
    return path


def build_epoch_summaries(
    *,
    selected_epochs: list[str],
    ripple_lfp_cache: dict[str, Any],
    ripple_tables_by_epoch: dict[str, "pd.DataFrame"],
    enable_notch_filter: bool,
) -> dict[str, dict[str, Any]]:
    """Return per-epoch run-log summaries."""
    summaries: dict[str, dict[str, Any]] = {}
    for epoch in selected_epochs:
        filtered_lfp = np.asarray(ripple_lfp_cache["data"][epoch], dtype=float)
        summaries[epoch] = {
            "filtered_lfp_sample_count": int(filtered_lfp.shape[0]),
            "filtered_lfp_channel_count": int(filtered_lfp.shape[1]) if filtered_lfp.ndim == 2 else 1,
            "actual_sampling_frequency_hz": float(ripple_lfp_cache["fs"][epoch]),
            "notch_applied": bool(enable_notch_filter),
            "ripple_count": int(len(ripple_tables_by_epoch[epoch])),
        }
    return summaries


def get_ripple_times(
    *,
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    epochs: list[str] | None = None,
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    disable_speed_gating: bool = False,
    enable_notch_filter: bool = DEFAULT_ENABLE_NOTCH_FILTER,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Detect ripples for one session and save the configured outputs."""
    if zscore_threshold <= 0:
        raise ValueError("--zscore-threshold must be positive.")

    analysis_path = get_analysis_path(animal_name=animal_name, date=date, data_root=data_root)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, timestamps_ephys_by_epoch = load_ephys_timestamps_by_epoch_npz(analysis_path)
    timestamps_ephys_all = load_ephys_timestamps_all_npz(analysis_path)
    use_speed_gating = not disable_speed_gating

    timestamps_position_by_epoch: dict[str, np.ndarray] = {}
    position_by_epoch: dict[str, np.ndarray] = {}
    if use_speed_gating:
        _position_epoch_tags, timestamps_position_by_epoch = load_position_timestamps_npz(
            analysis_path
        )
        available_clean_dlc_epochs, position_source_path = load_available_clean_dlc_epochs(
            analysis_path
        )
        selected_epochs = select_speed_gated_epochs(
            epoch_tags,
            requested_epochs=epochs,
            position_timestamp_epochs=timestamps_position_by_epoch,
            clean_dlc_epochs=available_clean_dlc_epochs,
        )
        validate_selected_epochs_across_sources(
            selected_epochs,
            source_epochs={
                "ephys_timestamps": timestamps_ephys_by_epoch,
                "position_timestamps": timestamps_position_by_epoch,
                "clean_dlc_position": available_clean_dlc_epochs,
            },
        )
        position_by_epoch, position_source_path = load_clean_dlc_head_position(
            analysis_path,
            selected_epochs=selected_epochs,
            timestamps_position_by_epoch=timestamps_position_by_epoch,
        )
        position_timestamp_source = "npz"
        position_source = str(position_source_path)
    else:
        selected_epochs = validate_epochs(epoch_tags, epochs)
        validate_selected_epochs_across_sources(
            selected_epochs,
            source_epochs={"ephys_timestamps": timestamps_ephys_by_epoch},
        )
        position_timestamp_source = "not_required_disable_speed_gating"
        position_source = "not_required_disable_speed_gating"

    channel_ids = get_ripple_channels_for_session(animal_name=animal_name, date=date)

    output_paths = get_output_paths(analysis_path / "ripple", use_speed_gating=use_speed_gating)
    recording = get_recording(animal_name=animal_name, date=date, nwb_root=nwb_root)
    ripple_lfp_cache, lfp_cache_source = compute_or_load_ripple_lfp_cache(
        cache_dir=output_paths["lfp_cache_dir"],
        animal_name=animal_name,
        date=date,
        recording=recording,
        selected_epochs=selected_epochs,
        timestamps_by_epoch=timestamps_ephys_by_epoch,
        timestamps_ephys_all=timestamps_ephys_all,
        channel_ids=channel_ids,
        enable_notch_filter=enable_notch_filter,
        overwrite=overwrite,
    )

    updated_tables: dict[str, pd.DataFrame] = {}
    for epoch in selected_epochs:
        print(f"Detecting ripples for {animal_name} {date} {epoch}")
        updated_tables[epoch] = detect_ripples_for_epoch(
            epoch=epoch,
            ripple_lfp_cache=ripple_lfp_cache,
            position_by_epoch=position_by_epoch,
            position_timestamps_by_epoch=timestamps_position_by_epoch,
            position_offset=position_offset,
            zscore_threshold=zscore_threshold,
            use_speed_gating=use_speed_gating,
        )

    flat_table = flatten_ripple_tables(updated_tables)
    interval_parquet_path = save_ripple_interval_output(output_paths["interval_parquet"], flat_table)

    epoch_summaries = build_epoch_summaries(
        selected_epochs=selected_epochs,
        ripple_lfp_cache=ripple_lfp_cache,
        ripple_tables_by_epoch=updated_tables,
        enable_notch_filter=enable_notch_filter,
    )

    print(f"Saved ripple intervals to {interval_parquet_path}")

    return {
        "analysis_path": analysis_path,
        "interval_parquet_path": interval_parquet_path,
        "lfp_cache_dir": output_paths["lfp_cache_dir"],
        "lfp_cache_epoch_paths": {
            epoch: get_epoch_lfp_cache_path(output_paths["lfp_cache_dir"], epoch)
            for epoch in selected_epochs
        },
        "sources": {
            "timestamps_ephys": "npz",
            "timestamps_ephys_all": "npz",
            "timestamps_position": position_timestamp_source,
            "position": position_source,
            "lfp_cache": lfp_cache_source,
        },
        "lfp_cache_epoch_actions": dict(ripple_lfp_cache["epoch_cache_actions"]),
        "available_epochs": epoch_tags,
        "selected_epochs": selected_epochs,
        "epoch_summaries": epoch_summaries,
        "ripple_channels": channel_ids,
        "position_offset": int(position_offset),
        "use_speed_gating": use_speed_gating,
        "enable_notch_filter": bool(enable_notch_filter),
    }


def main() -> None:
    """Run the ripple detection CLI."""
    args = parse_arguments()
    result = get_ripple_times(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        epochs=args.epochs,
        zscore_threshold=args.zscore_threshold,
        disable_speed_gating=args.disable_speed_gating,
        enable_notch_filter=not args.disable_notch_filter,
        overwrite=args.overwrite,
    )
    log_path = write_run_log(
        analysis_path=result["analysis_path"],
        script_name="v1ca1.ripple.detect_ripples",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "nwb_root": args.nwb_root,
            "epochs": result["selected_epochs"],
            "ripple_channels": result["ripple_channels"],
            "zscore_threshold": float(args.zscore_threshold),
            "position_offset": result["position_offset"],
            "use_speed_gating": result["use_speed_gating"],
            "overwrite": args.overwrite,
            "notch_filter_enabled": result["enable_notch_filter"],
            "notch_base_freq_hz": DEFAULT_NOTCH_BASE_FREQ,
            "notch_harmonics": DEFAULT_NOTCH_HARMONICS,
            "notch_quality": DEFAULT_NOTCH_QUALITY,
        },
        outputs={
            "sources": result["sources"],
            "saved_interval_parquet": result["interval_parquet_path"],
            "saved_lfp_cache_dir": result["lfp_cache_dir"],
            "saved_lfp_cache_epoch_paths": result["lfp_cache_epoch_paths"],
            "lfp_cache_epoch_actions": result["lfp_cache_epoch_actions"],
            "available_epochs": result["available_epochs"],
            "selected_epochs": result["selected_epochs"],
            "epoch_summaries": result["epoch_summaries"],
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
