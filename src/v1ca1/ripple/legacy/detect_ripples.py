import spikeinterface.full as si
import numpy as np
import kyutils
from pathlib import Path
import pickle
import scipy.signal
from typing import Tuple
import ripple_detection as rd
import scipy
import position_tools as pt
from typing import Tuple

import argparse

animal_name = "L14"
data_path = Path("/nimbus/kyu") / animal_name
date = "20240611"
analysis_path = data_path / "singleday_sort" / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

nwb_file_base_path = Path("/stelmo/nwb/raw")

(analysis_path / "ripple").mkdir(parents=True, exist_ok=True)

position_offset = 10

channels_with_ripples = [
    2 + 32 * 2 + 128 * 1,
    3 + 32 * 2 + 128 * 1,
    4 + 32 * 2 + 128 * 1,
    15 + 32 * 2 + 128 * 2,
    16 + 32 * 2 + 128 * 2,
    17 + 32 * 2 + 128 * 2,
]

si.set_global_job_kwargs(chunk_size=30000, n_jobs=8, progress_bar=True)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)


def design_notch_filter(
    freq: float, fs: float, quality: float = 50.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a second-order IIR notch filter.

    Parameters
    ----------
    freq : float
        Frequency to notch (Hz).
    fs : float
        Sampling frequency (Hz).
    quality : float, optional
        Quality factor (default is 50.0).

    Returns
    -------
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.
    """
    w0 = freq / (fs / 2)
    b, a = scipy.signal.iirnotch(w0, quality)
    return b, a


def apply_notch_filters(
    signal_data: np.ndarray, fs: float, base_freq: float = 60.0, n_harmonics: int = 10
) -> np.ndarray:
    """
    Apply multiple notch filters to remove 60 Hz noise and its harmonics.

    Parameters
    ----------
    signal_data : np.ndarray
        Signal to be filtered, shape (n_samples,).
    fs : float
        Sampling frequency (Hz).
    base_freq : float, optional
        Base noise frequency (default is 60.0).
    n_harmonics : int, optional
        Number of harmonics to filter (default is 10).

    Returns
    -------
    filtered_signal : np.ndarray
        Filtered signal.
    """
    filtered_signal = signal_data.copy()
    for i in range(1, n_harmonics + 1):
        freq = i * base_freq
        b, a = design_notch_filter(freq, fs)
        filtered_signal = scipy.signal.filtfilt(b, a, filtered_signal)
    return filtered_signal


def apply_notch_filters_multichannel(
    signal_data: np.ndarray,
    fs: float,
    base_freq: float = 60.0,
    n_harmonics: int = 10,
    quality: float = 50.0,
) -> np.ndarray:
    """
    Apply multiple notch filters to a multichannel signal.

    Parameters
    ----------
    signal_data : np.ndarray
        Input signal of shape (n_times, n_channels).
    fs : float
        Sampling frequency (Hz).
    base_freq : float, optional
        Base noise frequency (default 60.0 Hz).
    n_harmonics : int, optional
        Number of harmonics to filter (default 10).
    quality : float, optional
        Quality factor.

    Returns
    -------
    np.ndarray
        Filtered signal, shape (n_times, n_channels).
    """
    filtered = np.asarray(signal_data, dtype=float)
    for i in range(1, n_harmonics + 1):
        notch_freq = i * base_freq
        if notch_freq >= fs / 2:
            break
        b, a = scipy.signal.iirnotch(w0=notch_freq, Q=quality, fs=fs)
        sos = scipy.signal.tf2sos(b, a)
        filtered = scipy.signal.sosfiltfilt(sos, filtered, axis=0)
    return filtered


def butter_bandpass_filter(
    sampling_frequency: float,
    lowcut: float = 150.0,
    highcut: float = 250.0,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter.

    Parameters
    ----------
    sampling_frequency : float
        Sampling rate in Hz.
    lowcut : float
        Low frequency cutoff in Hz.
    highcut : float
        High frequency cutoff in Hz.
    order : int
        Filter order.

    Returns
    -------
    b : np.ndarray
        Numerator filter coefficients.
    a : np.ndarray
        Denominator filter coefficients.
    """
    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return b, a


def butter_filter_and_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    sampling_frequency: float,
    target_new_sampling_frequency: float,
    lowcut: float = 150.0,
    highcut: float = 250.0,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Bandpass filter (zero-phase) and decimate using integer factor q≈fs/target_fs.

    Parameters
    ----------
    timestamps : np.ndarray, shape (n_times,)
        Time vector in seconds.
    data : np.ndarray, shape (n_times, n_channels) or (n_times,)
        Time series signal.
    sampling_frequency : float
        Original sampling rate in Hz.
    target_new_sampling_frequency : float
        Desired sampling rate in Hz (target). Actual new fs will be fs / q.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    order : int
        Butterworth filter order.

    Returns
    -------
    decimated_timestamps : np.ndarray, shape (n_times_new,)
        Decimated timestamps.
    decimated_data : np.ndarray, shape (n_times_new, n_channels)
        Filtered/decimated data.
    actual_new_fs : float
        The actual new sampling rate fs' = fs / q.
    """
    if timestamps.ndim != 1:
        raise ValueError("`timestamps` must be 1D.")

    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if data.shape[0] != timestamps.shape[0]:
        raise ValueError("First dimension of `data` must match `timestamps` length.")

    fs = float(sampling_frequency)
    target_fs = float(target_new_sampling_frequency)
    if target_fs <= 0 or target_fs > fs:
        raise ValueError("`target_new_sampling_frequency` must be in (0, fs].")

    # Integer decimation factor q and implied new fs'
    q_float = fs / target_fs
    q: int = int(round(q_float))
    if q < 1:
        q = 1
    actual_new_fs = fs / q

    # Safety: keep band within (0, Nyquist)
    nyq = 0.5 * fs
    lo = max(1e-6, min(lowcut, nyq * 0.999))
    hi = max(lo * 1.001, min(highcut, nyq * 0.999))

    # Check that new Nyquist supports the desired band
    new_nyq = 0.5 * actual_new_fs
    if hi >= new_nyq:
        raise ValueError(
            f"Highcut {hi} Hz is too high for decimated Nyquist {new_nyq} Hz. "
            "Decrease highcut or decimation factor."
        )

    sos = scipy.signal.butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")

    # Zero-phase filter all channels at once (axis=0 is time)
    filtered = scipy.signal.sosfiltfilt(sos, data, axis=0)

    # Simple decimation by stride (no extra filter)
    decimated_data = filtered[::q]

    # Downsample timestamps by stride q
    decimated_timestamps = timestamps[::q]

    # Align lengths (just in case)
    n = min(decimated_timestamps.shape[0], decimated_data.shape[0])
    return decimated_timestamps[:n], decimated_data[:n], actual_new_fs


def filter_ripple_band(
    recording,
    channels_with_ripple=[260, 256, 318, 278, 282, 274, 338, 372, 342, 359, 367, 336],
):
    ripple_lfp = {"time": {}, "data": {}, "fs": {}}

    for epoch in epoch_list:
        start_frame = np.searchsorted(
            timestamps_ephys_all_ptp, timestamps_ephys[epoch][0]
        )
        end_frame = np.searchsorted(
            timestamps_ephys_all_ptp, timestamps_ephys[epoch][-1]
        )

        traces = recording.get_traces(
            channel_ids=channels_with_ripple,
            start_frame=start_frame,
            end_frame=end_frame,
            return_in_uV=True,
        )

        # Use recording sampling frequency if available
        try:
            fs = float(recording.get_sampling_frequency())
        except Exception:
            fs = 30_000.0  # fallback if extractor does not expose fs

        t_epoch = timestamps_ephys_all_ptp[start_frame:end_frame]

        # Optional: conditional notch
        if (date in ["20240605", "20240606", "20240607", "20240609"]) and (
            epoch == "08_r4"
        ):
            traces = apply_notch_filters_multichannel(
                traces, fs=fs, n_harmonics=10, quality=30
            )

        target_new_fs = 1000.0
        t_dec, x_dec, actual_new_fs = butter_filter_and_decimate(
            timestamps=t_epoch,
            data=traces,
            sampling_frequency=fs,
            target_new_sampling_frequency=target_new_fs,
            lowcut=150.0,
            highcut=250.0,
        )

        ripple_lfp["time"][epoch] = t_dec
        ripple_lfp["data"][epoch] = x_dec
        ripple_lfp["fs"][epoch] = actual_new_fs  # store actual fs'

    out_dir = analysis_path / "ripple"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ripple_channels_lfp.pkl", "wb") as f:
        pickle.dump(ripple_lfp, f)

    return ripple_lfp


def detect_ripples(epoch, ripple_lfp, zscore_threshold=2.0, use_speed=True):
    t_position = timestamps_position[epoch][position_offset:]
    position = position_dict[epoch][position_offset:]

    position_sampling_rate = (len(t_position) - 1) / (t_position[-1] - t_position[0])

    speed = pt.get_speed(
        position,
        t_position,
        sigma=0.1,
        sampling_frequency=position_sampling_rate,
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )

    if not use_speed:
        speed_interp = np.zeros(len(ripple_lfp["time"][epoch]))
    else:
        speed_interp = f_speed(ripple_lfp["time"][epoch])

    return rd.Kay_ripple_detector(
        time=ripple_lfp["time"][epoch],
        filtered_lfps=ripple_lfp["data"][epoch],
        speed=speed_interp,
        sampling_frequency=ripple_lfp["fs"][epoch],  # ← actual fs'
        zscore_threshold=zscore_threshold,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="detect SWR ripples")
    parser.add_argument("--use_speed", type=str, help="whether to use speed")

    parser.add_argument(
        "--overwrite", type=str, help="whether to overwrite existing lfp and detection"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.use_speed == "True":
        use_speed = True
    else:
        use_speed = False

    if args.overwrite == "True":
        overwrite = True
    else:
        overwrite = False

    if not use_speed:
        detector_save_path = (
            analysis_path / "ripple" / "Kay_ripple_detector_no_speed.pkl"
        )
    else:
        detector_save_path = analysis_path / "ripple" / "Kay_ripple_detector.pkl"

    Kay_ripple_times_dict = {}
    recording = si.read_nwb_recording(nwb_file_base_path / f"{animal_name}{date}.nwb")

    ripple_lfp_path = analysis_path / "ripple" / "ripple_channels_lfp.pkl"

    if ripple_lfp_path.exists():
        if overwrite:
            print("Overwriting existing ripple band filtered LFP.")
            ripple_lfp = filter_ripple_band(recording, channels_with_ripples)
        else:
            print("Ripple band filtered LFP exists. Not recomputing.")
            with open(ripple_lfp_path, "rb") as f:
                ripple_lfp = pickle.load(f)
    else:
        print("Ripple band filtered LFP doesn't exist. Computing...")
        ripple_lfp = filter_ripple_band(recording, channels_with_ripples)

    for epoch in epoch_list:
        print(f"detecting ripples for {date} {epoch}")
        Kay_ripple_times_dict[epoch] = detect_ripples(
            epoch, ripple_lfp, use_speed=use_speed
        )

    with open(
        detector_save_path,
        "wb",
    ) as f:
        pickle.dump(Kay_ripple_times_dict, f)


if __name__ == "__main__":
    main()
