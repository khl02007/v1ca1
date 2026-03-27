from __future__ import annotations

"""Shared session-loading and signal helpers for sleep analyses."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    DEFAULT_POSITION_OFFSET,
    REGIONS,
    build_speed_tsd,
    get_analysis_path,
    load_ephys_timestamps_all,
    load_ephys_timestamps_by_epoch,
    load_position_data_with_precedence,
    load_position_timestamps,
)

DEFAULT_V1_LFP_CHANNEL = 12
DEFAULT_RIPPLE_CHANNEL = 16 + 32 * 2 + 128 * 2
DEFAULT_SLEEP_PC1_THRESHOLD = 0.5
DEFAULT_SLEEP_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_SLEEP_MIN_DURATION_S = 0.1
DEFAULT_SLEEP_MAX_GAP_S = 0.1
DEFAULT_SLEEP_SPECTROGRAM_NPERSEG = 128
DEFAULT_SLEEP_SPECTROGRAM_NOVERLAP = 64
DEFAULT_SLEEP_DOWNSAMPLED_FREQUENCY_HZ = 60.0
DEFAULT_PLOT_SPECTROGRAM_NPERSEG = 256
DEFAULT_PLOT_SPECTROGRAM_NOVERLAP = 128
DEFAULT_PLOT_CUTOFF_HZ = 70.0
DEFAULT_TIME_BIN_SIZE_S = 2e-3
DEFAULT_FIRING_RATE_BIN_SIZE_S = 100e-3
EPSILON = 1e-12


def get_nwb_path(
    animal_name: str,
    date: str,
    nwb_root: Path = DEFAULT_NWB_ROOT,
) -> Path:
    """Return the NWB path for one session."""
    return nwb_root / f"{animal_name}{date}.nwb"


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


def load_sleep_session_inputs(analysis_path: Path) -> dict[str, Any]:
    """Load the saved timestamp and position inputs needed by sleep analyses."""
    epoch_tags, timestamps_ephys, timestamps_ephys_source = load_ephys_timestamps_by_epoch(
        analysis_path
    )
    timestamps_ephys_all, timestamps_ephys_all_source = load_ephys_timestamps_all(analysis_path)
    position_epoch_tags, timestamps_position, timestamps_position_source = load_position_timestamps(
        analysis_path
    )
    position_by_epoch, position_source = load_position_data_with_precedence(
        analysis_path,
        position_source="auto",
        clean_dlc_input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        clean_dlc_input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    return {
        "epoch_tags": epoch_tags,
        "timestamps_ephys": timestamps_ephys,
        "timestamps_ephys_all": timestamps_ephys_all,
        "timestamps_position": timestamps_position,
        "position_by_epoch": position_by_epoch,
        "sources": {
            "timestamps_ephys": timestamps_ephys_source,
            "timestamps_ephys_all": timestamps_ephys_all_source,
            "timestamps_position": timestamps_position_source,
            "position": position_source,
        },
    }


def load_sleep_sortings(
    analysis_path: Path,
    regions: tuple[str, ...] = REGIONS,
) -> dict[str, Any]:
    """Load the requested SpikeInterface sorting extractors."""
    try:
        import spikeinterface.full as si
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface.full is required to load sorting outputs for sleep analyses."
        ) from exc

    return {
        region: si.load(analysis_path / f"sorting_{region}")
        for region in regions
    }


def load_recording(nwb_path: Path) -> Any:
    """Load the session NWB recording."""
    try:
        import spikeinterface.full as si
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "spikeinterface.full is required to load NWB recordings for sleep analyses."
        ) from exc

    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")
    return si.read_nwb_recording(nwb_path)


def get_recording_sampling_frequency(recording: Any) -> float:
    """Return the recording sampling frequency in Hz."""
    try:
        sampling_frequency = float(recording.get_sampling_frequency())
    except Exception as exc:
        raise ValueError("Could not read the recording sampling frequency.") from exc
    if sampling_frequency <= 0:
        raise ValueError(f"Recording sampling frequency must be positive, got {sampling_frequency}.")
    return sampling_frequency


def validate_recording_channel(
    recording: Any,
    channel_id: int,
    *,
    channel_name: str,
) -> int:
    """Validate that one channel exists in the recording."""
    channel_ids = np.asarray(recording.get_channel_ids())
    if channel_ids.size == 0:
        raise ValueError("Recording does not contain any channel ids.")
    if int(channel_id) not in channel_ids.tolist():
        raise ValueError(
            f"{channel_name} {channel_id} was not found in the recording channel ids "
            f"{channel_ids.tolist()!r}."
        )
    return int(channel_id)


def get_epoch_frame_bounds(
    epoch_timestamps: np.ndarray,
    timestamps_ephys_all: np.ndarray,
) -> tuple[int, int]:
    """Return inclusive epoch frame bounds as SpikeInterface start/end indices."""
    epoch_timestamps = np.asarray(epoch_timestamps, dtype=float)
    timestamps_ephys_all = np.asarray(timestamps_ephys_all, dtype=float)
    if epoch_timestamps.ndim != 1 or epoch_timestamps.size == 0:
        raise ValueError("Epoch timestamps must be a non-empty one-dimensional array.")

    start_frame = int(np.searchsorted(timestamps_ephys_all, epoch_timestamps[0], side="left"))
    end_frame = int(np.searchsorted(timestamps_ephys_all, epoch_timestamps[-1], side="right"))
    if start_frame >= end_frame:
        raise ValueError(
            "Could not resolve a non-empty frame range for the requested epoch timestamps."
        )
    return start_frame, end_frame


def get_epoch_trace(
    recording: Any,
    *,
    epoch_timestamps: np.ndarray,
    timestamps_ephys_all: np.ndarray,
    channel_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one single-channel trace for one epoch."""
    start_frame, end_frame = get_epoch_frame_bounds(epoch_timestamps, timestamps_ephys_all)
    trace = np.asarray(
        recording.get_traces(
            channel_ids=[int(channel_id)],
            return_in_uV=True,
            start_frame=start_frame,
            end_frame=end_frame,
        ),
        dtype=float,
    ).reshape(-1)
    timestamps = np.asarray(timestamps_ephys_all[start_frame:end_frame], dtype=float)
    if trace.shape[0] != timestamps.shape[0]:
        raise ValueError(
            "Loaded trace length does not match the selected timestamp segment: "
            f"{trace.shape[0]} vs {timestamps.shape[0]}."
        )
    return timestamps, trace


def butter_filter_and_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    sampling_frequency: float,
    new_sampling_frequency: float,
    *,
    cutoff_hz: float | None = None,
    lowcut_hz: float | None = None,
    highcut_hz: float | None = None,
    order: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter one trace and decimate it to a lower sampling rate."""
    import scipy.signal

    timestamps = np.asarray(timestamps, dtype=float)
    data_array = np.asarray(data, dtype=float)
    if timestamps.ndim != 1:
        raise ValueError("Timestamps must be one-dimensional.")
    if data_array.shape[0] != timestamps.shape[0]:
        raise ValueError("Trace length must match the timestamp length.")
    if new_sampling_frequency <= 0:
        raise ValueError("new_sampling_frequency must be positive.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if new_sampling_frequency > sampling_frequency:
        raise ValueError("new_sampling_frequency cannot exceed sampling_frequency.")

    q = int(round(sampling_frequency / new_sampling_frequency))
    if q < 1:
        raise ValueError(
            "Could not derive a positive integer decimation factor from "
            f"{sampling_frequency} Hz to {new_sampling_frequency} Hz."
        )

    nyquist = 0.5 * sampling_frequency
    if cutoff_hz is not None:
        normalized = float(cutoff_hz) / nyquist
        b, a = scipy.signal.butter(order, normalized, btype="low", analog=False)
    elif lowcut_hz is not None and highcut_hz is not None:
        low = float(lowcut_hz) / nyquist
        high = float(highcut_hz) / nyquist
        b, a = scipy.signal.butter(order, [low, high], btype="band")
    else:
        raise ValueError("Pass either cutoff_hz or both lowcut_hz and highcut_hz.")

    if data_array.ndim == 1:
        data_array = data_array[:, np.newaxis]

    filtered = np.apply_along_axis(
        lambda values: scipy.signal.filtfilt(b, a, values),
        axis=0,
        arr=data_array,
    )
    decimated = np.apply_along_axis(
        lambda values: scipy.signal.decimate(values, q, ftype="iir", zero_phase=True),
        axis=0,
        arr=filtered,
    )
    decimated_timestamps = timestamps[::q]
    min_len = min(decimated_timestamps.shape[0], decimated.shape[0])
    decimated_timestamps = decimated_timestamps[:min_len]
    decimated = decimated[:min_len]
    if decimated.shape[1] == 1:
        return decimated_timestamps, decimated[:, 0]
    return decimated_timestamps, decimated


def lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float,
    sampling_frequency: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth lowpass filter to one trace."""
    import scipy.signal

    signal_array = np.asarray(signal, dtype=float)
    if signal_array.ndim != 1:
        raise ValueError("signal must be one-dimensional.")

    nyquist = 0.5 * float(sampling_frequency)
    normalized = float(cutoff_hz) / nyquist
    b, a = scipy.signal.butter(order, normalized, btype="low", analog=False)
    return np.asarray(scipy.signal.filtfilt(b, a, signal_array), dtype=float)


def decimate_signal(
    signal: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Decimate one trace when the factor is greater than one."""
    import scipy.signal

    signal_array = np.asarray(signal, dtype=float)
    if factor <= 1:
        return signal_array
    return np.asarray(
        scipy.signal.decimate(signal_array, int(factor), ftype="iir", zero_phase=True),
        dtype=float,
    )


def compute_spectrogram_principal_component(
    signal: np.ndarray,
    sampling_frequency: float,
    *,
    max_frequency_hz: float,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the spectrogram time axis and first z-scored PC below one cutoff."""
    import scipy.signal
    import scipy.stats
    from sklearn.decomposition import PCA

    frequencies, times, spectrogram_values = scipy.signal.spectrogram(
        np.asarray(signal, dtype=float),
        fs=float(sampling_frequency),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
    )
    keep = frequencies < float(max_frequency_hz)
    if not np.any(keep):
        raise ValueError(
            f"No spectrogram frequencies were below the requested cutoff {max_frequency_hz} Hz."
        )

    values = spectrogram_values[keep].T
    pc1 = scipy.stats.zscore(PCA(n_components=1).fit_transform(values), axis=0).reshape(-1)
    return np.asarray(times, dtype=float), np.asarray(pc1, dtype=float)


def compute_theta_delta_ratio(
    signal: np.ndarray,
    sampling_frequency: float,
    *,
    nperseg: int,
    noverlap: int,
    theta_range_hz: tuple[float, float] = (5.0, 10.0),
    delta_range_hz: tuple[float, float] = (0.5, 4.0),
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the spectrogram time axis and CA1 theta/delta ratio trace."""
    import scipy.signal

    frequencies, times, spectrogram_values = scipy.signal.spectrogram(
        np.asarray(signal, dtype=float),
        fs=float(sampling_frequency),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
    )
    theta_power = spectrogram_values[
        (frequencies > float(theta_range_hz[0])) & (frequencies <= float(theta_range_hz[1]))
    ].sum(axis=0)
    delta_power = spectrogram_values[
        (frequencies > float(delta_range_hz[0])) & (frequencies <= float(delta_range_hz[1]))
    ].sum(axis=0)
    return np.asarray(times, dtype=float), np.asarray(
        theta_power / np.maximum(delta_power, float(epsilon)),
        dtype=float,
    )


def build_uniform_time_grid(
    start_time: float,
    end_time: float,
    sampling_frequency: float,
) -> np.ndarray:
    """Build an evenly sampled time grid spanning one closed interval."""
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive.")
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time.")

    n_samples = int(np.ceil((float(end_time) - float(start_time)) * float(sampling_frequency))) + 1
    return np.linspace(float(start_time), float(end_time), n_samples)


def interpolate_to_grid(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one trace onto a requested time grid."""
    import scipy.interpolate

    interpolator = scipy.interpolate.interp1d(
        np.asarray(source_times, dtype=float),
        np.asarray(source_values, dtype=float),
        axis=0,
        bounds_error=False,
        kind="linear",
    )
    return np.asarray(interpolator(np.asarray(target_times, dtype=float)), dtype=float)


def get_speed_trace(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    *,
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return trimmed position samples, timestamps, and speed for one epoch."""
    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")

    speed_tsd = build_speed_tsd(
        position=np.asarray(position, dtype=float),
        timestamps_position=np.asarray(timestamps_position, dtype=float),
        position_offset=position_offset,
    )
    trimmed_position = np.asarray(position[position_offset:], dtype=float)
    trimmed_timestamps = np.asarray(speed_tsd.t, dtype=float)
    speed = np.asarray(speed_tsd.d, dtype=float)
    if trimmed_position.shape[0] != trimmed_timestamps.shape[0]:
        raise ValueError(
            "Trimmed position samples and speed timestamps must have matching lengths."
        )
    return trimmed_position, trimmed_timestamps, speed


def get_time_grid_from_position_timestamps(
    timestamps_position: np.ndarray,
    *,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    time_bin_size_s: float = DEFAULT_TIME_BIN_SIZE_S,
) -> np.ndarray:
    """Return the uniform spike-count time grid derived from position timestamps."""
    trimmed_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    if trimmed_timestamps.size == 0:
        raise ValueError("Position offset removed all position timestamps for the epoch.")
    return build_uniform_time_grid(
        float(trimmed_timestamps[0]),
        float(trimmed_timestamps[-1]),
        1.0 / float(time_bin_size_s),
    )


def get_spike_indicator(
    sorting: Any,
    *,
    timestamps_ephys_all: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    """Bin spike trains from one sorting onto a requested time grid."""
    spike_indicator: list[np.ndarray] = []
    all_timestamps = np.asarray(timestamps_ephys_all, dtype=float)
    time_array = np.asarray(time_grid, dtype=float)

    for unit_id in sorting.get_unit_ids():
        spike_times = all_timestamps[sorting.get_unit_spike_train(unit_id)]
        spike_times = spike_times[(spike_times > time_array[0]) & (spike_times <= time_array[-1])]
        spike_indicator.append(
            np.bincount(
                np.digitize(spike_times, time_array[1:-1]),
                minlength=time_array.shape[0],
            )
        )
    if not spike_indicator:
        return np.zeros((time_array.shape[0], 0), dtype=float)
    return np.asarray(spike_indicator, dtype=float).T


def get_time_spike_indicator(
    timestamps_position: np.ndarray,
    *,
    position_offset: int,
    time_bin_size_s: float,
    sorting: Any,
    timestamps_ephys_all: np.ndarray,
    temporal_overlap: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the spike-count matrix on one position-derived time grid."""
    time_grid = get_time_grid_from_position_timestamps(
        timestamps_position,
        position_offset=position_offset,
        time_bin_size_s=time_bin_size_s,
    )
    if not temporal_overlap:
        return time_grid, get_spike_indicator(
            sorting,
            timestamps_ephys_all=timestamps_ephys_all,
            time_grid=time_grid,
        )

    offset_grid = time_grid[:-1] + float(time_bin_size_s) / 2.0
    interleaved_time = np.sort(np.concatenate([time_grid, offset_grid]))
    spike_indicator_primary = get_spike_indicator(
        sorting,
        timestamps_ephys_all=timestamps_ephys_all,
        time_grid=time_grid,
    )
    spike_indicator_offset = get_spike_indicator(
        sorting,
        timestamps_ephys_all=timestamps_ephys_all,
        time_grid=offset_grid,
    )
    spike_indicator = np.zeros((interleaved_time.shape[0], spike_indicator_primary.shape[1]), dtype=float)
    spike_indicator[0::2] = spike_indicator_primary
    spike_indicator[1::2] = spike_indicator_offset
    return interleaved_time, spike_indicator


def compute_firing_rate_matrix(
    sorting: Any,
    *,
    epoch_timestamps: np.ndarray,
    timestamps_ephys_all: np.ndarray,
    bin_size_s: float,
    zscore_values: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one per-unit firing-rate matrix and bin-center time axis."""
    epoch_timestamps = np.asarray(epoch_timestamps, dtype=float)
    if epoch_timestamps.ndim != 1 or epoch_timestamps.size < 2:
        raise ValueError("epoch_timestamps must contain at least two samples.")

    start_time = float(epoch_timestamps[0])
    stop_time = float(epoch_timestamps[-1])
    bin_edges = np.arange(start_time, stop_time, float(bin_size_s))
    if bin_edges.size < 2:
        raise ValueError(
            f"Epoch duration is shorter than the requested bin size {bin_size_s} s."
        )
    time = bin_edges[:-1] + float(bin_size_s) / 2.0
    firing_rates = np.zeros((len(sorting.get_unit_ids()), bin_edges.shape[0] - 1), dtype=float)

    all_timestamps = np.asarray(timestamps_ephys_all, dtype=float)
    for index, unit_id in enumerate(sorting.get_unit_ids()):
        spike_times = all_timestamps[sorting.get_unit_spike_train(unit_id)]
        spike_times = spike_times[(spike_times > start_time) & (spike_times <= stop_time)]
        spike_counts, _ = np.histogram(spike_times, bin_edges)
        if np.all(spike_counts == 0):
            continue

        firing_rate = spike_counts / float(bin_size_s)
        if zscore_values:
            mean_firing_rate = float(np.mean(firing_rate))
            std_firing_rate = float(np.std(firing_rate))
            if std_firing_rate > 0:
                firing_rates[index] = (firing_rate - mean_firing_rate) / std_firing_rate
            else:
                firing_rates[index] = np.zeros_like(firing_rate, dtype=float)
        else:
            firing_rates[index] = firing_rate
    return firing_rates, np.asarray(time, dtype=float)


def order_units_by_epoch_spike_count(
    sorting: Any,
    *,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    """Return unit indices ordered by spike count within one epoch."""
    sliced_sorting = sorting.time_slice(start_time=float(start_time), end_time=float(end_time))
    return np.argsort(
        [
            len(sliced_sorting.get_unit_spike_train(unit_id))
            for unit_id in sliced_sorting.get_unit_ids()
        ]
    )


def find_joint_threshold_intervals(
    timestamps: NDArray[np.generic],
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    *,
    threshold_1: float,
    threshold_2: float,
    min_duration_s: float | None = None,
) -> NDArray[np.int64]:
    """Return contiguous index intervals where x1 is above and x2 is below threshold."""
    if timestamps.shape != x1.shape or timestamps.shape != x2.shape:
        raise ValueError("timestamps, x1, and x2 must have matching shapes.")

    mask = (x1 > float(threshold_1)) & (x2 < float(threshold_2))
    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int64)

    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.shape[0]]))

    intervals = np.column_stack((starts, ends)).astype(np.int64)
    if min_duration_s is None:
        return intervals

    durations = timestamps[intervals[:, 1] - 1] - timestamps[intervals[:, 0]]
    return intervals[durations >= float(min_duration_s)].astype(np.int64)


def intervals_to_time_bounds(
    timestamps: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    """Convert half-open index intervals into start/end time bounds."""
    time_array = np.asarray(timestamps, dtype=float)
    interval_array = np.asarray(intervals, dtype=np.int64)
    if interval_array.size == 0:
        return np.empty((0, 2), dtype=float)
    starts = time_array[interval_array[:, 0]]
    ends = time_array[interval_array[:, 1] - 1]
    return np.column_stack((starts, ends)).astype(float)


def merge_close_intervals(
    intervals: np.ndarray,
    *,
    max_gap_s: float,
) -> np.ndarray:
    """Merge consecutive time intervals separated by at most one gap threshold."""
    interval_array = np.asarray(intervals, dtype=float)
    if interval_array.size == 0:
        return np.empty((0, 2), dtype=float)

    sorted_intervals = interval_array[np.argsort(interval_array[:, 0])]
    merged: list[list[float]] = [[float(sorted_intervals[0, 0]), float(sorted_intervals[0, 1])]]
    for start, end in sorted_intervals[1:]:
        if float(start) - merged[-1][1] <= float(max_gap_s):
            merged[-1][1] = max(merged[-1][1], float(end))
        else:
            merged.append([float(start), float(end)])
    return np.asarray(merged, dtype=float)


def compute_sleep_time_intervals(
    timestamps: np.ndarray,
    pc1_trace: np.ndarray,
    speed_trace: np.ndarray,
    *,
    pc1_threshold: float,
    speed_threshold_cm_s: float,
    min_duration_s: float,
    max_gap_s: float,
) -> np.ndarray:
    """Return merged sleep intervals from interpolated PC1 and speed traces."""
    intervals = find_joint_threshold_intervals(
        np.asarray(timestamps, dtype=float),
        np.asarray(pc1_trace, dtype=float),
        np.asarray(speed_trace, dtype=float),
        threshold_1=pc1_threshold,
        threshold_2=speed_threshold_cm_s,
        min_duration_s=min_duration_s,
    )
    return merge_close_intervals(
        intervals_to_time_bounds(np.asarray(timestamps, dtype=float), intervals),
        max_gap_s=max_gap_s,
    )


def save_interval_table_output(
    analysis_path: Path,
    *,
    output_name: str,
    intervals_by_epoch: dict[str, np.ndarray],
) -> Path:
    """Write one canonical parquet table of interval rows."""
    import pandas as pd

    rows: list[dict[str, float | str]] = []
    for epoch, intervals in intervals_by_epoch.items():
        interval_array = np.asarray(intervals, dtype=float)
        if interval_array.size == 0:
            continue
        if interval_array.ndim != 2 or interval_array.shape[1] != 2:
            raise ValueError(
                f"Expected {output_name!r} intervals with shape (n, 2) for epoch {epoch!r}, "
                f"got {interval_array.shape}."
            )
        for start, end in interval_array:
            rows.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "epoch": str(epoch),
                }
            )

    interval_table = pd.DataFrame.from_records(rows, columns=["start", "end", "epoch"])
    if not interval_table.empty:
        interval_table = interval_table.sort_values(
            by=["start", "end", "epoch"],
            kind="stable",
        ).reset_index(drop=True)

    output_path = analysis_path / f"{output_name}.parquet"
    interval_table.to_parquet(output_path, index=False)
    return output_path
