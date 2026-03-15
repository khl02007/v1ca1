from __future__ import annotations

"""Detect hippocampal ripples for one session.

This modernized CLI replaces the legacy ripple-detection lab script with
validated session loading, explicit CLI arguments, legacy-compatible detector
pickles, and modern pynapple outputs under the session analysis directory.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
    load_position_data,
    load_position_timestamps,
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
    """Return the legacy and modern output paths for one detector mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_speed_gating:
        legacy_pickle = output_dir / "Kay_ripple_detector.pkl"
        stem = "ripple_times"
    else:
        legacy_pickle = output_dir / "Kay_ripple_detector_no_speed.pkl"
        stem = "ripple_times_no_speed"
    return {
        "legacy_pickle": legacy_pickle,
        "interval_npz": output_dir / f"{stem}.npz",
        "lfp_cache": output_dir / "ripple_channels_lfp.pkl",
    }


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
    """Return an empty ripple-band cache structure."""
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
    }


def cache_matches_request(
    cache: dict[str, Any],
    selected_epochs: list[str],
    channel_ids: list[int],
    *,
    enable_notch_filter: bool,
) -> bool:
    """Return whether an existing LFP cache can be reused for this request."""
    if "time" not in cache or "data" not in cache or "fs" not in cache:
        return False
    stored_channels = cache.get("channel_ids")
    if stored_channels is None:
        return False
    if list(stored_channels) != list(channel_ids):
        return False
    if bool(cache.get("notch_filter_enabled")) is not bool(enable_notch_filter):
        return False
    if cache.get("notch_base_freq_hz") != DEFAULT_NOTCH_BASE_FREQ:
        return False
    if cache.get("notch_harmonics") != DEFAULT_NOTCH_HARMONICS:
        return False
    if cache.get("notch_quality") != DEFAULT_NOTCH_QUALITY:
        return False
    return all(epoch in cache["time"] and epoch in cache["data"] and epoch in cache["fs"] for epoch in selected_epochs)


def cache_channels_match(
    cache: dict[str, Any],
    channel_ids: list[int],
    *,
    enable_notch_filter: bool,
) -> bool:
    """Return whether an existing cache was built from the requested channels."""
    stored_channels = cache.get("channel_ids")
    if stored_channels is None:
        return False
    if bool(cache.get("notch_filter_enabled")) is not bool(enable_notch_filter):
        return False
    if cache.get("notch_base_freq_hz") != DEFAULT_NOTCH_BASE_FREQ:
        return False
    if cache.get("notch_harmonics") != DEFAULT_NOTCH_HARMONICS:
        return False
    if cache.get("notch_quality") != DEFAULT_NOTCH_QUALITY:
        return False
    return list(stored_channels) == list(channel_ids)


def compute_or_load_ripple_lfp_cache(
    *,
    cache_path: Path,
    recording: Any,
    selected_epochs: list[str],
    timestamps_by_epoch: dict[str, np.ndarray],
    timestamps_ephys_all: np.ndarray,
    channel_ids: list[int],
    enable_notch_filter: bool,
    overwrite: bool,
) -> tuple[dict[str, Any], str]:
    """Load or compute the cached ripple-band LFP traces."""
    cache: dict[str, Any] | None = None
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            loaded_cache = pickle.load(file)
        if isinstance(loaded_cache, dict) and not overwrite:
            if cache_matches_request(
                loaded_cache,
                selected_epochs,
                channel_ids,
                enable_notch_filter=enable_notch_filter,
            ):
                return loaded_cache, "existing_pickle"
            if (
                "time" in loaded_cache
                and "data" in loaded_cache
                and "fs" in loaded_cache
                and cache_channels_match(
                    loaded_cache,
                    channel_ids,
                    enable_notch_filter=enable_notch_filter,
                )
            ):
                cache = loaded_cache

    if cache is None or not isinstance(cache, dict):
        cache = initialize_lfp_cache(channel_ids, enable_notch_filter=enable_notch_filter)
    else:
        cache["channel_ids"] = list(channel_ids)
        cache["notch_filter_enabled"] = bool(enable_notch_filter)
        cache["notch_base_freq_hz"] = DEFAULT_NOTCH_BASE_FREQ
        cache["notch_harmonics"] = DEFAULT_NOTCH_HARMONICS
        cache["notch_quality"] = DEFAULT_NOTCH_QUALITY

    for epoch in selected_epochs:
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as file:
        pickle.dump(cache, file)
    return cache, "computed"


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


def load_existing_ripple_tables(path: Path) -> dict[str, "pd.DataFrame"]:
    """Load an existing legacy detector pickle into normalized dataframes."""
    if not path.exists():
        return {}

    with open(path, "rb") as file:
        loaded = pickle.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Existing ripple detector pickle is not a dictionary: {path}")

    return {str(epoch): normalize_ripple_table(table, epoch=str(epoch)) for epoch, table in loaded.items()}


def merge_ripple_tables(
    epoch_order: list[str],
    existing_tables: dict[str, "pd.DataFrame"],
    updated_tables: dict[str, "pd.DataFrame"],
) -> dict[str, "pd.DataFrame"]:
    """Merge updated per-epoch detector tables with any existing saved results."""
    combined = {**existing_tables, **updated_tables}
    ordered_combined: dict[str, pd.DataFrame] = {}
    remaining_epochs = [epoch for epoch in combined if epoch not in epoch_order]
    for epoch in [*epoch_order, *sorted(remaining_epochs)]:
        if epoch in combined:
            ordered_combined[epoch] = combined[epoch]
    return ordered_combined


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


def save_legacy_detector_pickle(
    path: Path,
    ripple_tables_by_epoch: dict[str, "pd.DataFrame"],
) -> Path:
    """Write the legacy per-epoch ripple detector pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(ripple_tables_by_epoch, file)
    return path


def _to_metadata_list(values: list[Any]) -> list[Any] | None:
    """Convert one metadata column to pynapple-compatible scalar values."""
    metadata_values: list[Any] = []
    for value in values:
        if isinstance(value, np.generic):
            metadata_values.append(value.item())
            continue
        if value is None or isinstance(value, (str, bool, int, float)):
            metadata_values.append(value)
            continue
        if isinstance(value, Path):
            metadata_values.append(str(value))
            continue
        if np.isscalar(value):
            metadata_values.append(value)
            continue
        return None
    return metadata_values


def save_ripple_interval_output(path: Path, flat_table: "pd.DataFrame") -> Path:
    """Write the flattened ripple event table as one pynapple IntervalSet."""
    import pynapple as nap

    start_array = np.asarray(flat_table["start_time"], dtype=float)
    end_array = np.asarray(flat_table["end_time"], dtype=float)
    interval_set = nap.IntervalSet(start=start_array, end=end_array, time_units="s")

    metadata: dict[str, list[Any]] = {}
    if "epoch" in flat_table.columns:
        metadata["epoch"] = [str(epoch) for epoch in flat_table["epoch"].tolist()]

    for column_name in flat_table.columns:
        if column_name in {"start_time", "end_time", "epoch"}:
            continue
        metadata_values = _to_metadata_list(flat_table[column_name].tolist())
        if metadata_values is not None:
            metadata[column_name] = metadata_values

    if metadata:
        interval_set.set_info(**metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    interval_set.save(path)
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

    epoch_tags, timestamps_ephys_by_epoch, ephys_source = load_ephys_timestamps_by_epoch(analysis_path)
    timestamps_ephys_all, ephys_all_source = load_ephys_timestamps_all(analysis_path)
    selected_epochs = validate_epochs(epoch_tags, epochs)
    _, timestamps_position_by_epoch, position_timestamp_source = load_position_timestamps(
        analysis_path
    )
    validate_selected_epochs_across_sources(
        selected_epochs,
        source_epochs={
            "ephys_timestamps": timestamps_ephys_by_epoch,
            "position_timestamps": timestamps_position_by_epoch,
        },
    )
    position_by_epoch = load_position_data(analysis_path, selected_epochs)
    channel_ids = get_ripple_channels_for_session(animal_name=animal_name, date=date)
    use_speed_gating = not disable_speed_gating

    output_paths = get_output_paths(analysis_path / "ripple", use_speed_gating=use_speed_gating)
    recording = get_recording(animal_name=animal_name, date=date, nwb_root=nwb_root)
    ripple_lfp_cache, lfp_cache_source = compute_or_load_ripple_lfp_cache(
        cache_path=output_paths["lfp_cache"],
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

    existing_tables = load_existing_ripple_tables(output_paths["legacy_pickle"])
    combined_tables = merge_ripple_tables(epoch_tags, existing_tables, updated_tables)
    flat_table = flatten_ripple_tables(combined_tables)

    legacy_pickle_path = save_legacy_detector_pickle(output_paths["legacy_pickle"], combined_tables)
    interval_npz_path = save_ripple_interval_output(output_paths["interval_npz"], flat_table)

    epoch_summaries = build_epoch_summaries(
        selected_epochs=selected_epochs,
        ripple_lfp_cache=ripple_lfp_cache,
        ripple_tables_by_epoch=updated_tables,
        enable_notch_filter=enable_notch_filter,
    )

    print(f"Saved legacy ripple detector pickle to {legacy_pickle_path}")
    print(f"Saved ripple intervals to {interval_npz_path}")

    return {
        "analysis_path": analysis_path,
        "legacy_pickle_path": legacy_pickle_path,
        "interval_npz_path": interval_npz_path,
        "lfp_cache_path": output_paths["lfp_cache"],
        "sources": {
            "timestamps_ephys": ephys_source,
            "timestamps_ephys_all": ephys_all_source,
            "timestamps_position": position_timestamp_source,
            "position": "pickle",
            "lfp_cache": lfp_cache_source,
        },
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
            "saved_legacy_pickle": result["legacy_pickle_path"],
            "saved_interval_npz": result["interval_npz_path"],
            "saved_lfp_cache": result["lfp_cache_path"],
            "available_epochs": result["available_epochs"],
            "selected_epochs": result["selected_epochs"],
            "epoch_summaries": result["epoch_summaries"],
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
