import spikeinterface.full as si
import probeinterface as pi
import numpy as np
import kyutils
from pathlib import Path
import os
import pickle
from scipy.signal import butter, filtfilt, decimate, hilbert
from typing import Tuple

import argparse

animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date


num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

nwb_file_base_path = Path("/stelmo/nwb/raw")


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
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_filter_and_decimate(
    timestamps: np.ndarray,
    data: np.ndarray,
    sampling_frequency: float,
    new_sampling_frequency: float,
    lowcut: float = 150.0,
    highcut: float = 250.0,
    order: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply Butterworth bandpass filter and decimate signal.

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
    if sampling_frequency % new_sampling_frequency != 0:
        raise ValueError(
            "sampling_frequency must be divisible by new_sampling_frequency."
        )

    b, a = butter_bandpass_filter(sampling_frequency, lowcut, highcut, order)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Zero-phase filter along time axis
    filtered = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=0, arr=data)

    # Decimate
    decimated_data = np.apply_along_axis(
        lambda x: decimate(x, q, ftype="iir", zero_phase=True), axis=0, arr=filtered
    )

    decimated_timestamps = timestamps[::q]
    min_len = min(len(decimated_timestamps), decimated_data.shape[0])
    return decimated_timestamps[:min_len], decimated_data[:min_len]


def get_theta_phase(theta_channel=260):

    recording = si.read_nwb_recording(nwb_file_base_path / f"{animal_name}{date}.nwb")
    filtered_recording = si.bandpass_filter(
        recording, freq_min=4.0, freq_max=12.0, dtype=np.float64
    )

    theta_lfp = {}
    theta_phase = {}
    for epoch in epoch_list:
        start_frame = np.searchsorted(
            timestamps_ephys_all_ptp, timestamps_ephys[epoch][0]
        )
        end_frame = np.searchsorted(
            timestamps_ephys_all_ptp, timestamps_ephys[epoch][-1]
        )
        theta_lfp[epoch] = filtered_recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            channel_ids=[theta_channel],
            return_in_uV=True,
        ).flatten()
        theta_phase[epoch] = np.angle(hilbert(theta_lfp[epoch]))

    with open(
        analysis_path / "theta_lfp.pkl",
        "wb",
    ) as f:
        pickle.dump(theta_lfp, f)

    with open(
        analysis_path / "theta_phase.pkl",
        "wb",
    ) as f:
        pickle.dump(theta_phase, f)

    return None


def main():
    theta_channel = 162
    get_theta_phase(theta_channel)


if __name__ == "__main__":
    main()
