"Detects the rhythmic V1 activity at center reward wells"

# utf-8
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
import ripple_detection as rd
from pathlib import Path
import kyutils
import time
import pandas as pd
import pickle
import spikeinterface.full as si
import position_tools as pt


animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)

regions = ["v1", "ca1"]

sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

position_offset = 10

temporal_bin_size_s = 2e-3
sampling_rate = int(1 / temporal_bin_size_s)
speed_threshold = 4  # cm/s


# def get_spike_indicator(sorting, timestamps_ephys_all, time):
#     spike_indicator = []
#     for unit_id in sorting.get_unit_ids():
#         spike_times = timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)]
#         spike_times = spike_times[(spike_times > time[0]) & (spike_times <= time[-1])]
#         spike_indicator.append(
#             np.bincount(np.digitize(spike_times, time[1:-1]), minlength=time.shape[0])
#         )
#     spike_indicator = np.asarray(spike_indicator).T
#     return spike_indicator


def get_spike_indicator(sorting, timestamps_ephys_all, time):
    # time: (n_time,) sampled uniformly; interpret as left edges of bins of width dt
    dt = float(time[1] - time[0])
    t0 = float(time[0])
    n_time = time.size

    spike_indicator = np.zeros((n_time, len(sorting.get_unit_ids())), dtype=np.int32)

    for j, unit_id in enumerate(sorting.get_unit_ids()):
        spike_times = timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)]
        # map to bin index
        idx = np.floor((spike_times - t0) / dt).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_time)]
        spike_indicator[:, j] = np.bincount(idx, minlength=n_time)

    return spike_indicator


def detect_theta_bursts(
    x: np.ndarray,
    fs: float,
    theta_band: Tuple[float, float] = (8.0, 12.0),
    threshold_start_z: float = 2.5,
    threshold_end_z: float = 1.5,
    min_duration_s: float = 1.0,
    min_cycles: float | None = None,
    merge_gap_s: float = 0.150,
    envelope_smooth_s: float = 0.200,
) -> Dict[str, Any]:
    """
    Detect transient theta-band (8–12 Hz) oscillations using robust envelope
    thresholding with hysteresis, smoothing, gap merging, and duration/cycle
    constraints. Returns only plain NumPy arrays (no classes).

    Parameters
    ----------
    x : np.ndarray
        Input time series, shape (n_time,).
    fs : float
        Sampling frequency in Hz.
    theta_band : tuple[float, float], optional
        Bandpass range in Hz. Default is (8.0, 12.0).
    threshold_start_z : float, optional
        Robust z-score to START a burst. Default is 2.5.
    threshold_end_z : float, optional
        Robust z-score to END a burst. Default is 1.5 (must be < start).
    min_duration_s : float, optional
        Minimum event duration in seconds. Default is 1.0.
    min_cycles : float | None, optional
        Minimum number of cycles at band center frequency. If None, only
        duration is enforced. Default is None.
    merge_gap_s : float, optional
        Merge events separated by a gap smaller than this (seconds).
        Default is 0.150.
    envelope_smooth_s : float, optional
        Moving-average smoothing of the envelope in seconds. Default is 0.200.

    Returns
    -------
    result : dict
        {
            "mask": np.ndarray,            # (n_time,) boolean mask of detected bursts
            "events": {                    # per-event arrays, all length n_events
                "start_idx": np.ndarray,   # (n_events,)
                "end_idx": np.ndarray,     # (n_events,) end is exclusive
                "start_time": np.ndarray,  # (n_events,)
                "end_time": np.ndarray,    # (n_events,)
                "duration": np.ndarray,    # (n_events,)
                "peak_envelope": np.ndarray,# (n_events,)
                "peak_time": np.ndarray,   # (n_events,)
            },
            "envelope": np.ndarray,        # (n_time,) smoothed envelope
            "zscore": np.ndarray,          # (n_time,) robust z-score of envelope
            "thresholds": {
                "start_z": float,
                "end_z": float,
                "median": float,
                "mad": float,
            },
        }

    Notes
    -----
    - Zero-phase bandpass (filtfilt) → Hilbert envelope → smoothing.
    - Robust baseline via median/MAD; hysteresis avoids flicker.
    - Merges short gaps; keeps only events that meet duration/cycle criteria.
    """
    # ---- Input checks
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if threshold_start_z <= threshold_end_z:
        raise ValueError("threshold_start_z must be > threshold_end_z.")

    n = x.size

    # ---- Bandpass in theta band
    nyq = 0.5 * fs
    low = theta_band[0] / nyq
    high = theta_band[1] / nyq
    b, a = butter(4, [low, high], btype="band")
    x_filt = filtfilt(b, a, x, method="pad")

    # ---- Envelope via Hilbert
    envelope = np.abs(hilbert(x_filt))

    # ---- Envelope smoothing (moving average, reflect at edges)
    win = max(1, int(round(envelope_smooth_s * fs)))
    if win > 1:
        kernel = np.ones(win, dtype=float) / win
        env_s = np.convolve(envelope, kernel, mode="same")
    else:
        env_s = envelope

    # ---- Robust z-score (median/MAD)
    med = float(np.median(env_s))
    mad = float(np.median(np.abs(env_s - med)))
    scale = 1.4826 * mad + 1e-12
    z = (env_s - med) / scale

    # ---- Hysteresis binarization
    mask = np.zeros(n, dtype=bool)
    active = False
    for i in range(n):
        if not active and z[i] >= threshold_start_z:
            active = True
        if active:
            mask[i] = True
            if z[i] < threshold_end_z:
                active = False

    # ---- Extract segments
    xint = mask.astype(np.int8)
    starts = np.where(np.diff(np.r_[0, xint]) == 1)[0]
    ends = np.where(np.diff(np.r_[xint, 0]) == -1)[0]

    # ---- Merge close segments
    if starts.size and ends.size:
        min_gap = int(round(merge_gap_s * fs))
        merged_starts = [int(starts[0])]
        merged_ends = [int(ends[0])]
        for s, e in zip(starts[1:], ends[1:]):
            if s - merged_ends[-1] < min_gap:
                merged_ends[-1] = int(e)
            else:
                merged_starts.append(int(s))
                merged_ends.append(int(e))
        starts = np.asarray(merged_starts, dtype=int)
        ends = np.asarray(merged_ends, dtype=int)

    # ---- Enforce duration and (optional) cycle count
    keep = []
    min_len = int(round(min_duration_s * fs))
    fc = 0.5 * (theta_band[0] + theta_band[1])
    for s, e in zip(starts, ends):
        length = e - s
        if length < min_len:
            continue
        if min_cycles is not None:
            dur_s = length / fs
            if fc * dur_s < float(min_cycles):
                continue
        keep.append((s, e))

    # ---- Build final mask and event features
    final_mask = np.zeros(n, dtype=bool)
    ev_start_idx: list[int] = []
    ev_end_idx: list[int] = []
    ev_start_t: list[float] = []
    ev_end_t: list[float] = []
    ev_dur: list[float] = []
    ev_peak_env: list[float] = []
    ev_peak_t: list[float] = []

    for s, e in keep:
        final_mask[s:e] = True
        seg = env_s[s:e]
        p_rel = int(np.argmax(seg))
        p_idx = s + p_rel

        ev_start_idx.append(int(s))
        ev_end_idx.append(int(e))
        ev_start_t.append(float(s / fs))
        ev_end_t.append(float(e / fs))
        ev_dur.append(float((e - s) / fs))
        ev_peak_env.append(float(seg[p_rel]))
        ev_peak_t.append(float(p_idx / fs))

    events = {
        "start_idx": np.asarray(ev_start_idx, dtype=int),
        "end_idx": np.asarray(ev_end_idx, dtype=int),
        "start_time": np.asarray(ev_start_t, dtype=float),
        "end_time": np.asarray(ev_end_t, dtype=float),
        "duration": np.asarray(ev_dur, dtype=float),
        "peak_envelope": np.asarray(ev_peak_env, dtype=float),
        "peak_time": np.asarray(ev_peak_t, dtype=float),
    }

    return {
        "mask": final_mask,
        "events": events,
        "envelope": env_s,
        "zscore": z,
        "thresholds": {
            "start_z": float(threshold_start_z),
            "end_z": float(threshold_end_z),
            "median": med,
            "mad": mad,
        },
    }


def theta_burst_events_to_df(
    result: Dict[str, Any],
    *,
    t0: float = 0.0,
) -> pd.DataFrame:
    """
    Convert detect_theta_bursts() output to a DataFrame of (start_time, stop_time).

    Parameters
    ----------
    result : dict
        Output of detect_theta_bursts(...).
    t0 : float, optional
        Offset added to all times (e.g. absolute start time). Default is 0.0.

    Returns
    -------
    df : pd.DataFrame
        Columns:
        - start_time_s
        - stop_time_s
        - duration_s
        - peak_time_s
        - peak_envelope
    """
    ev = result["events"]

    start = np.asarray(ev["start_time"], dtype=float) + float(t0)
    stop = np.asarray(ev["end_time"], dtype=float) + float(t0)

    df = pd.DataFrame(
        {
            "start_time_s": start,
            "stop_time_s": stop,
            "duration_s": np.asarray(ev["duration"], dtype=float),
            "peak_time_s": np.asarray(ev["peak_time"], dtype=float) + float(t0),
            "peak_envelope": np.asarray(ev["peak_envelope"], dtype=float),
        }
    )

    # keep robust behavior for empty case + sorting
    if len(df) == 0:
        return df

    df = df.sort_values("start_time_s", kind="mergesort").reset_index(drop=True)
    return df


def main():

    start = time.perf_counter()

    v1_theta_times = {}
    for epoch in run_epoch_list:
        t_position = timestamps_position[epoch][position_offset:]
        start_time = t_position[0]
        end_time = t_position[-1]
        n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1
        t = np.linspace(start_time, end_time, n_samples)

        spike_indicator = get_spike_indicator(
            sorting["v1"], timestamps_ephys_all=timestamps_ephys_all_ptp, time=t
        )
        mua = rd.get_multiunit_population_firing_rate(
            multiunit=spike_indicator, sampling_frequency=sampling_rate
        )
        results = detect_theta_bursts(
            x=mua,
            fs=sampling_rate,
            theta_band=(8.0, 12.0),
            threshold_start_z=2.5,
            threshold_end_z=1.5,
            min_duration_s=1.0,
            merge_gap_s=0.150,
            envelope_smooth_s=0.200,
        )
        v1_theta_times[epoch] = theta_burst_events_to_df(results, t0=t[0])

    with open(analysis_path / "v1_theta_times.pkl", "wb") as f:
        pickle.dump(v1_theta_times, f)

    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
