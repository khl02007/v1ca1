from __future__ import annotations

from numpy.typing import NDArray
from typing import Tuple, Union
from pathlib import Path
import pandas as pd
import kyutils
import pickle
import scipy.signal
import scipy.stats
import spikeinterface.full as si
import numpy as np
import position_tools as pt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import track_linearization as tl

import argparse

animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / "20240611"

num_sleep_epochs = 5
num_run_epochs = 4

nwb_file_base_path = Path("/stelmo/nwb/raw")

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

regions = ["v1", "ca1"]

position_offset = 10

trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]
with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

position_offset = 10

time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)

save_dir = analysis_path / "sleep_times"
save_dir.mkdir(parents=True, exist_ok=True)


nwb_file_name = f"{animal_name}{date}.nwb"

nperseg = 128  # Length of each segment for FFT
noverlap = 64  # Overlap between segments

lfp_channel = 12


def butter_filter_and_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    sampling_frequency: float,
    new_sampling_frequency: float,
    cutoff: float = 150.0,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Butterworth lowpass filter and decimate signal.

    Parameters
    ----------
    timestamps : np.ndarray
        Time vector in seconds. Shape: (n_times,)
    data : np.ndarray
        Time series signal. Shape: (n_times, n_channels)
    sampling_frequency : float
        Original sampling rate in Hz.
    new_sampling_frequency : float
        Desired sampling rate in Hz.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Decimated timestamps and filtered data. Shapes: (n_times_new,), (n_times_new, n_channels)
    """
    if timestamps.ndim != 1 or data.shape[0] != timestamps.shape[0]:
        raise ValueError("Timestamps must be 1D and match first dimension of data.")

    q = int(round(sampling_frequency / new_sampling_frequency))

    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Zero-phase filter along time axis
    filtered = np.apply_along_axis(
        lambda x: scipy.signal.filtfilt(b, a, x), axis=0, arr=data
    )

    # Decimate
    decimated_data = np.apply_along_axis(
        lambda x: scipy.signal.decimate(x, q, ftype="iir", zero_phase=True),
        axis=0,
        arr=filtered,
    )

    decimated_timestamps = timestamps[::q]
    min_len = min(len(decimated_timestamps), decimated_data.shape[0])
    return decimated_timestamps[:min_len], decimated_data[:min_len].flatten()


def find_joint_threshold_intervals(
    t: NDArray[np.generic],
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    thr1: float,
    thr2: float,
    min_duration: float | None = None,
) -> NDArray[np.int64]:
    """
    Identify contiguous index intervals where both time series exceed thresholds.

    Parameters
    ----------
    t : ndarray of shape (n_time,)
        Monotonic timestamps.
    x1 : ndarray of shape (n_time,)
        First time series.
    x2 : ndarray of shape (n_time,)
        Second time series.
    thr1 : float
        Threshold for x1.
    thr2 : float
        Threshold for x2.
    min_duration : float or None
        Minimum interval duration in same units as t. If None, no filtering.

    Returns
    -------
    intervals : ndarray of shape (n_intervals, 2)
        Each row = [start_idx, end_idx) in index coordinates.
    """
    if t.shape != x1.shape or t.shape != x2.shape:
        raise ValueError("Shapes of t, x1, x2 must match.")

    mask = (x1 > thr1) & (x2 < thr2)

    if not np.any(mask):
        return np.empty((0, 2), dtype=np.int64)

    mask_int = mask.astype(np.int8)
    diff = np.diff(mask_int)

    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate(([0], starts))

    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))

    intervals = np.column_stack((starts, ends))

    if min_duration is not None:
        durations = t[intervals[:, 1] - 1] - t[intervals[:, 0]]
        intervals = intervals[durations >= min_duration]

    return intervals.astype(np.int64)


def intervals_to_times(
    t: NDArray[np.generic],
    intervals: NDArray[np.int64],
) -> NDArray[np.generic]:
    """
    Convert index intervals to timestamp intervals.

    Parameters
    ----------
    t : ndarray of shape (n_time,)
        Timestamps corresponding to the time axis.
    intervals : ndarray of shape (n_intervals, 2)
        Index intervals [start_idx, end_idx), as returned by
        `find_joint_threshold_intervals`.

    Returns
    -------
    time_intervals : ndarray of shape (n_intervals, 2)
        Each row is [start_time, end_time], where:
        - start_time = t[start_idx]
        - end_time   = t[end_idx - 1]
    """
    if intervals.size == 0:
        return np.empty((0, 2), dtype=t.dtype)

    starts = t[intervals[:, 0]]
    ends = t[intervals[:, 1] - 1]  # end index is exclusive
    time_intervals = np.column_stack((starts, ends))
    return time_intervals


def build_mask_from_intervals(
    n_time: int,
    intervals: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """
    Build a Boolean mask of length `n_time` that is True inside given intervals.

    Parameters
    ----------
    n_time : int
        Length of the time axis.
    intervals : ndarray of shape (n_intervals, 2)
        Index intervals [start_idx, end_idx).

    Returns
    -------
    mask : ndarray of shape (n_time,)
        Boolean mask with True inside intervals and False outside.
    """
    mask = np.zeros(n_time, dtype=bool)
    for start, end in intervals:
        mask[start:end] = True
    return mask


def plot_time_series_with_intervals(
    t: NDArray[np.generic],
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    thr1: float,
    thr2: float,
    min_duration: float | None = None,
    figsize: Tuple[float, float] = (10.0, 5.0),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot two time series, thresholds, and highlight joint-exceedance intervals.

    Parameters
    ----------
    t : ndarray of shape (n_time,)
        Timestamps.
    x1 : ndarray of shape (n_time,)
        First time series.
    x2 : ndarray of shape (n_time,)
        Second time series.
    thr1 : float
        Threshold for `x1`.
    thr2 : float
        Threshold for `x2`.
    min_duration : float or None, optional
        Minimum duration of intervals to highlight (same units as `t`).
        If None, all joint-exceedance intervals are highlighted.
    figsize : tuple of float, optional
        Size of the figure (width, height).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes with the plot.
    """
    # Get filtered intervals (duration constraint applied here)
    intervals = find_joint_threshold_intervals(t, x1, x2, thr1, thr2, min_duration)

    # Build a mask that is True only on the kept intervals
    mask = build_mask_from_intervals(t.shape[0], intervals)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot time series
    ax.plot(t, x1, label="x1", linewidth=1.5)
    ax.plot(t, x2, label="x2", linewidth=1.5)

    # Plot thresholds
    ax.axhline(thr1, linestyle="--", linewidth=1.0, label=f"thr1 = {thr1}")
    ax.axhline(thr2, linestyle="--", linewidth=1.0, label=f"thr2 = {thr2}")

    # Highlight only the intervals that satisfy the duration constraint
    ymin, ymax = ax.get_ylim()
    ax.fill_between(t, ymin, ymax, where=mask, alpha=0.15)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(
        "Time series with joint threshold-exceedance intervals "
        f"(min_duration={min_duration})"
    )
    ax.legend(loc="best")
    ax.grid(True, linestyle=":")

    fig.tight_layout()
    return fig, ax


def intervals_to_dataframe(
    t: NDArray[np.generic], intervals: NDArray[np.int64]
) -> pd.DataFrame:
    """
    Convert index intervals to a pandas DataFrame.

    Parameters
    ----------
    t : ndarray (n_time,)
        Timestamps.
    intervals : ndarray (n_intervals, 2)
        Index intervals [start_idx, end_idx).

    Returns
    -------
    df : pandas.DataFrame
        Columns:
        - start_time
        - end_time
        - duration
    """
    # print("len(t):", len(t))
    # print("intervals.shape:", intervals.shape)
    # print("intervals min start:", intervals[:, 0].min())
    # print("intervals max start:", intervals[:, 0].max())
    # print("intervals min end:", intervals[:, 1].min())
    # print("intervals max end:", intervals[:, 1].max())

    if intervals.size == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "duration"])

    starts = t[intervals[:, 0]]
    ends = t[intervals[:, 1] - 1]
    durations = ends - starts

    return pd.DataFrame(
        {
            "start_time": starts,
            "end_time": ends,
            "duration": durations,
        }
    )


def merge_close_intervals(
    df: pd.DataFrame,
    max_gap: Union[float, np.timedelta64, pd.Timedelta],
) -> pd.DataFrame:
    """
    Merge consecutive intervals if the gap between them is less than or equal
    to a given threshold.

    Intervals are assumed to be half-open [start_time, end_time], and the gap
    between interval i and i+1 is defined as:

        gap_i = start_time_{i+1} - end_time_i

    Consecutive intervals with gap_i <= max_gap are merged into a single
    interval spanning from the first start_time to the last end_time.

    Parameters
    ----------
    df : pandas.DataFrame, shape (n_intervals, 3)
        DataFrame with columns:
        - "start_time"
        - "end_time"
        - "duration"
        Typically produced by `intervals_to_dataframe`.
    max_gap : float or numpy.timedelta64 or pandas.Timedelta
        Maximum allowed gap between consecutive intervals for them to be
        merged. Must be compatible with the dtype of `start_time` / `end_time`.
        For numeric time, use a float; for datetime-like time, use a Timedelta.

    Returns
    -------
    merged_df : pandas.DataFrame, shape (n_merged_intervals, 3)
        DataFrame with merged intervals, with columns:
        - "start_time"
        - "end_time"
        - "duration"
    """
    if df.empty:
        return df.copy()

    # Ensure sorted by start_time
    df_sorted = df.sort_values("start_time").reset_index(drop=True)

    start_vals = df_sorted["start_time"].to_numpy()
    end_vals = df_sorted["end_time"].to_numpy()

    n_intervals = len(df_sorted)
    if n_intervals == 1:
        # Nothing to merge; just recompute duration to be safe
        out = df_sorted[["start_time", "end_time"]].copy()
        out["duration"] = out["end_time"] - out["start_time"]
        return out

    # Compute gaps between consecutive intervals: start_{i+1} - end_i
    gaps = start_vals[1:] - end_vals[:-1]

    # Start a new group where gap > max_gap; keep same group where gap <= max_gap
    new_group = np.empty(n_intervals, dtype=bool)
    new_group[0] = True
    new_group[1:] = gaps > max_gap

    group_ids = np.cumsum(new_group.astype(int)) - 1  # 0-based group IDs

    df_sorted = df_sorted.assign(_group=group_ids)

    merged = (
        df_sorted.groupby("_group", sort=False)
        .agg(
            start_time=("start_time", "min"),
            end_time=("end_time", "max"),
        )
        .reset_index(drop=True)
    )

    merged["duration"] = merged["end_time"] - merged["start_time"]

    return merged


def get_sleep_times(
    epoch, speed_threshold=4.0, pc1_threshold=0.5, min_duration=0.1, max_gap=0.1
):
    recording = si.read_nwb_recording(nwb_file_base_path / nwb_file_name)
    fs = recording.get_sampling_frequency()

    # get speed
    position = position_dict[epoch][position_offset:]
    t_position = timestamps_position_dict[epoch][position_offset:]
    position_sampling_rate = len(position) / (t_position[-1] - t_position[0])

    speed = pt.get_speed(
        position,
        time=t_position,
        sampling_frequency=position_sampling_rate,
        sigma=0.1,
    )

    # get first PC of spectrogram from a cortical channel
    i1 = np.searchsorted(timestamps_ephys_all_ptp, timestamps_ephys[epoch][0])
    i2 = np.searchsorted(timestamps_ephys_all_ptp, timestamps_ephys[epoch][-1])

    v1_signal = recording.get_traces(
        channel_ids=[lfp_channel],
        return_in_uV=True,
        start_frame=i1,
        end_frame=i2,
    ).flatten()

    new_sampling_frequency = 60
    t, v1_signal_decimated = butter_filter_and_decimate(
        timestamps=timestamps_ephys_all_ptp[i1:i2],
        data=v1_signal,
        sampling_frequency=fs,
        new_sampling_frequency=new_sampling_frequency,
        cutoff=150,
    )

    v1_frequencies, v1_times, v1_Sxx = scipy.signal.spectrogram(
        x=v1_signal_decimated,
        fs=new_sampling_frequency,
        nperseg=nperseg,
        noverlap=noverlap,
    )
    v1_times_spectrogram = v1_times + timestamps_ephys_all_ptp[i1]
    v1_Sxx_filtered = v1_Sxx[v1_frequencies < 60, :]
    Sxx_flat = v1_Sxx_filtered.T  # Transpose to have time as rows for PCA
    pca = PCA(n_components=1)
    v1_lfp_spectrogram_pc1 = scipy.stats.zscore(pca.fit_transform(Sxx_flat)).squeeze()

    # interpolate to have same timestamps
    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    f_pc1 = scipy.interpolate.interp1d(
        v1_times_spectrogram,
        v1_lfp_spectrogram_pc1,
        axis=0,
        bounds_error=False,
        kind="linear",
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )

    pc1_interp = f_pc1(time)
    speed_interp = f_speed(time)

    # detect threshold crossing
    intervals = find_joint_threshold_intervals(
        t=time,
        x1=pc1_interp,
        x2=speed_interp,
        thr1=pc1_threshold,
        thr2=speed_threshold,
        min_duration=min_duration,
    )
    df = intervals_to_dataframe(time, intervals)
    # plot_time_series_with_intervals(t, x1, x2, thr1, thr2, min_duration=min_duration)
    df = merge_close_intervals(df, max_gap=max_gap)

    with open(save_dir / f"{epoch}.pkl", "wb") as f:
        pickle.dump(df, f)

    return df


def main():
    # args = parse_arguments()
    for epoch in epoch_list:
        get_sleep_times(epoch)


if __name__ == "__main__":
    main()
