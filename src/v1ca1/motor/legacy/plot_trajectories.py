from typing import Tuple
from pathlib import Path

import kyutils
import pickle
import spikeinterface.full as si
import numpy as np
import position_tools as pt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, decimate, spectrogram
from scipy.stats import zscore
from sklearn.decomposition import PCA
import ripple_detection as rd
import replay_trajectory_classification as rtc
import track_linearization as tl
import pynapple as nap

animal_name = "L14"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = (
    data_path / "multiday_sort" / "20240605_20240606_20240607_20240609_20240611"
)
dates = ["20240605", "20240606", "20240607", "20240609", "20240611"]

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

nwb_file_base_path = Path("/stelmo/nwb/raw")

trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]
position_offset = 10
time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)
t_all = np.concatenate([timestamps_ephys_all_ptp[date] for date in dates])

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)


eps = 1e-12

sorting_path = analysis_path / "sorting_v1"
sorting = si.load(sorting_path)

data = {}
for unit_id in sorting.get_unit_ids():
    data[unit_id] = nap.Ts(t=t_all[sorting.get_unit_spike_train(unit_id)])
tsgroup = nap.TsGroup(data, time_units="s")


lfp_channel = 12
ripple_channel = 16 + 32 * 2 + 128 * 2

position_dict = {}
for date in dates:
    position_dict[date] = {}
    for epoch in epoch_list:
        with open(
            analysis_path / "position" / f"{date}_{animal_name}_{epoch}.pkl", "rb"
        ) as f:
            position_dict[date][epoch] = pickle.load(f)


def get_time(date: str, epoch: str, time_bin_size: float, ptp: bool = True):
    sampling_rate = int(1 / (time_bin_size))

    # define reference time offset and subtract it from the timestamps
    t_position = timestamps_position[date][epoch][position_offset:]
    if not ptp:
        t_position = t_position - timestamps_ephys[date][epoch][0]

    # define time vector for decoding (temporal resolution: 2 ms)
    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    return time


def get_spike_indicator(date, epoch, sorting, t_all, time, ptp):
    if not ptp:
        t_all = t_all - timestamps_ephys[date][epoch][0]
    spike_indicator = []
    for unit_id in sorting.get_unit_ids():
        spike_times = t_all[sorting.get_unit_spike_train(unit_id)]
        spike_times = spike_times[(spike_times > time[0]) & (spike_times <= time[-1])]
        spike_indicator.append(
            np.bincount(np.digitize(spike_times, time[1:-1]), minlength=time.shape[0])
        )
    spike_indicator = np.asarray(spike_indicator).T
    return spike_indicator


def get_time_spike_indicator(
    date, epoch, time_bin_size, ptp, sorting, t_all, temporal_overlap
):
    if temporal_overlap:
        time1 = get_time(date, epoch, time_bin_size, ptp)
        time2 = time1 + time_bin_size / 2
        time2 = time2[:-1]
        time = np.sort(np.concatenate([time1, time2]))

        spike_indicator1 = get_spike_indicator(date, epoch, sorting, t_all, time1, ptp)
        spike_indicator2 = get_spike_indicator(date, epoch, sorting, t_all, time2, ptp)
        spike_indicator = np.zeros((len(time), spike_indicator1.shape[1]))
        spike_indicator[0::2] = spike_indicator1
        spike_indicator[1::2] = spike_indicator2
    else:
        time = get_time(date, epoch, time_bin_size, ptp)
        spike_indicator = get_spike_indicator(date, epoch, sorting, t_all, time, ptp)

    return time, spike_indicator


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


def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(signal, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, signal)


def get_firing_rate_matrix(sorting, date, epoch, bin_size, zscore=True):

    start_time = timestamps_ephys[date][epoch][0]
    stop_time = timestamps_ephys[date][epoch][-1]

    bin_edges = np.arange(start_time, stop_time, bin_size)
    time = bin_edges[:-1] + bin_size / 2

    fr = np.zeros((len(sorting.get_unit_ids()), len(bin_edges) - 1))

    for k, unit_id in enumerate(sorting.get_unit_ids()):

        spike_times = t_all[sorting.get_unit_spike_train(unit_id)]
        spike_times = spike_times[
            (spike_times > start_time) & (spike_times <= stop_time)
        ]

        spike_counts, _ = np.histogram(spike_times, bin_edges)
        if np.all(spike_counts == 0):
            continue

        firing_rate = spike_counts / bin_size
        if zscore:
            mean_fr = np.mean(firing_rate)
            std_fr = np.std(firing_rate)
            fr[k] = (firing_rate - mean_fr) / std_fr
        else:
            fr[k] = firing_rate
    return fr, time


def gaussian_kernel(
    sigma: float,
    dt: float,
    support: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a discrete-time Gaussian kernel with unit area.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel in seconds.
    dt : float
        Time step (bin width) in seconds.
    support : float, optional
        Half-width in units of sigma for truncation, by default 5.0.

    Returns
    -------
    kernel : ndarray, shape (n_kernel,)
        Discrete Gaussian kernel normalized so that sum(kernel) * dt == 1.
    t_kernel : ndarray, shape (n_kernel,)
        Kernel time axis (s), centered at 0.
    """
    if sigma <= 0 or dt <= 0 or support <= 0:
        raise ValueError("sigma, dt, and support must be positive.")

    half_width = int(np.ceil(support * sigma / dt))
    t_kernel = dt * np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (t_kernel / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    kernel /= kernel.sum() * dt  # unit area in discrete time
    return kernel, t_kernel


def rate_from_binary_train(
    spikes: np.ndarray,
    dt: float,
    sigma: float,
    support: float = 5.0,
    mode: str = "same",
) -> np.ndarray:
    """Estimate firing rate (Hz) from a binned spike train using a Gaussian kernel.

    Parameters
    ----------
    spikes : ndarray, shape (n_time,) or (n_time, n_neurons)
        Spike counts per bin (0/1 or integer counts).
    dt : float
        Bin width in seconds (e.g., 0.002 for 2 ms).
    sigma : float
        Gaussian kernel standard deviation in seconds.
    support : float, optional
        Half-width in units of sigma for kernel truncation, by default 5.0.
    mode : {"same", "full", "valid"}, optional
        Convolution mode passed to `np.convolve`, by default "same".

    Returns
    -------
    rate_hz : ndarray
        Estimated firing rate in Hz. Shape is (n_time,) for 1D input or
        (n_time, n_neurons) for 2D input.
    """
    kernel, _ = gaussian_kernel(sigma=sigma, dt=dt, support=support)

    if spikes.ndim == 1:
        smoothed = np.convolve(spikes.astype(float), kernel, mode=mode)
        return smoothed

    elif spikes.ndim == 2:
        # Vectorized convolution across neurons
        smoothed = np.apply_along_axis(
            lambda s: np.convolve(s.astype(float), kernel, mode=mode),
            axis=0,
            arr=spikes,
        )
        return smoothed

    else:
        raise ValueError("spikes must be 1D or 2D (n_time,) or (n_time, n_neurons).")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_contour_two_distributions(
    data1: np.ndarray, data2: np.ndarray, title: str, grid_size: int = 200, ax=None
) -> None:
    """
    Plot contour lines for two 2D sample distributions.

    Parameters
    ----------
    data1 : np.ndarray
        Samples from distribution 1. Shape: (n_samples, 2).
    data2 : np.ndarray
        Samples from distribution 2. Shape: (n_samples, 2).
    grid_size : int, optional
        Resolution of the evaluation grid, by default 200.
    """

    # Kernel density estimates
    kde1 = gaussian_kde(data1.T)
    kde2 = gaussian_kde(data2.T)

    # Determine grid bounds based on all samples
    xmin = min(data1[:, 0].min(), data2[:, 0].min())
    xmax = max(data1[:, 0].max(), data2[:, 0].max())
    ymin = min(data1[:, 1].min(), data2[:, 1].min())
    ymax = max(data1[:, 1].max(), data2[:, 1].max())

    # Create grid
    x_grid, y_grid = np.mgrid[
        xmin : xmax : complex(grid_size), ymin : ymax : complex(grid_size)
    ]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Evaluate KDE on the grid
    z1 = kde1(positions).reshape(x_grid.shape)
    z2 = kde2(positions).reshape(x_grid.shape)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax.contour(x_grid, y_grid, z1, levels=8, cmap="Blues")
        ax.contour(x_grid, y_grid, z2, levels=8, linestyles="dashed", cmap="Reds")

        ax.set_xlabel("X position (cm)")
        ax.set_ylabel("Y position (cm)")
        ax.set_title(title)


center_to_left_colors = ["purple", "green"]
center_to_right_colors = ["blue", "red"]
date_list = ["20240607", "20240611"]
epoch = "06_r3"

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4 * 1))

for date_idx, date in enumerate(date_list):

    # for epoch_idx, epoch in enumerate(['02_r1']):

    for i, j in trajectory_times[date][epoch]["left_to_center"]:
        ax[date_idx].plot(
            position_dict[date][epoch][
                (timestamps_position[date][epoch] > i)
                & (timestamps_position[date][epoch] <= j),
                0,
            ],
            position_dict[date][epoch][
                (timestamps_position[date][epoch] > i)
                & (timestamps_position[date][epoch] <= j),
                1,
            ],
            color=center_to_left_colors[1],
            alpha=0.3,
        )

    for i, j in trajectory_times[date][epoch]["right_to_center"]:
        ax[date_idx].plot(
            position_dict[date][epoch][
                (timestamps_position[date][epoch] > i)
                & (timestamps_position[date][epoch] <= j),
                0,
            ],
            position_dict[date][epoch][
                (timestamps_position[date][epoch] > i)
                & (timestamps_position[date][epoch] <= j),
                1,
            ],
            color=center_to_right_colors[1],
            alpha=0.3,
        )

    long_segment_length = 71
    short_segment_length = 32
    node_positions_left = np.array(
        [
            (55, 81),  # center well
            (55 - short_segment_length, 81),  # left well
            (55, 81 - long_segment_length),  # center junction
            (55 - short_segment_length, 81 - long_segment_length),  # left junction
        ]
    )
    ax[date_idx].plot(node_positions_left[:, 0], node_positions_left[:, 1], "ko")

    node_positions_right = np.array(
        [
            (55, 81),  # center well
            # (55-short_segment_length, 81),  # left well
            (55 + short_segment_length, 81),  # right well
            (55, 81 - long_segment_length),  # center junction
            # (55-short_segment_length, 81-long_segment_length),  # left junction
            (55 + short_segment_length, 81 - long_segment_length),  # right junction
        ]
    )
    ax[date_idx].plot(node_positions_right[:, 0], node_positions_right[:, 1], "ko")
    ax[date_idx].plot([55 - 5, 55 - 5], [55, 40], color="black")
    ax[date_idx].plot([55 + 5, 55 + 5], [55, 40], color="black")
    ax[date_idx].plot([55 - 5, 55 + 5], [40, 40], color="black")
    ax[date_idx].plot([55 - 5, 55 + 5], [55, 55], color="black")
    ax[date_idx].set_title(date)


# for (i,j) in trajectory_times[date1][epoch]['center_to_left']:
#     plt.plot(position_dict[date1][epoch][(timestamps_position[date1][epoch]>i) & (timestamps_position[date1][epoch]<=j), 0],
#              position_dict[date1][epoch][(timestamps_position[date1][epoch]>i) & (timestamps_position[date1][epoch]<=j), 1],
#              color='blue', alpha=0.3)

# for (i,j) in trajectory_times[date1][epoch]['center_to_right']:
#     plt.plot(position_dict[date1][epoch][(timestamps_position[date1][epoch]>i) & (timestamps_position[date1][epoch]<=j), 0],
#              position_dict[date1][epoch][(timestamps_position[date1][epoch]>i) & (timestamps_position[date1][epoch]<=j), 1],
#              color='red', alpha=0.3)

center_to_left_colors = ["red", "magenta"]
center_to_right_colors = ["blue", "cyan"]
date_list = ["20240607", "20240611"]
fig, ax = plt.subplots(ncols=2, nrows=len(date_list), figsize=(8, 4 * len(date_list)))
for date_idx, date in enumerate(date_list):
    for epoch_idx, epoch in enumerate(["06_r3", "08_r4"]):

        for i, j in trajectory_times[date][epoch]["center_to_left"]:
            ax[date_idx, epoch_idx].plot(
                position_dict[date][epoch][
                    (timestamps_position[date][epoch] > i)
                    & (timestamps_position[date][epoch] <= j),
                    0,
                ],
                position_dict[date][epoch][
                    (timestamps_position[date][epoch] > i)
                    & (timestamps_position[date][epoch] <= j),
                    1,
                ],
                color=center_to_left_colors[date_idx],
                alpha=0.3,
            )

        for i, j in trajectory_times[date][epoch]["center_to_right"]:
            ax[date_idx, epoch_idx].plot(
                position_dict[date][epoch][
                    (timestamps_position[date][epoch] > i)
                    & (timestamps_position[date][epoch] <= j),
                    0,
                ],
                position_dict[date][epoch][
                    (timestamps_position[date][epoch] > i)
                    & (timestamps_position[date][epoch] <= j),
                    1,
                ],
                color=center_to_right_colors[date_idx],
                alpha=0.3,
            )

        long_segment_length = 71
        short_segment_length = 32
        node_positions_left = np.array(
            [
                (55, 81),  # center well
                (55 - short_segment_length, 81),  # left well
                (55, 81 - long_segment_length),  # center junction
                (55 - short_segment_length, 81 - long_segment_length),  # left junction
            ]
        )
        ax[date_idx, epoch_idx].plot(
            node_positions_left[:, 0], node_positions_left[:, 1], "ko"
        )

        node_positions_right = np.array(
            [
                (55, 81),  # center well
                # (55-short_segment_length, 81),  # left well
                (55 + short_segment_length, 81),  # right well
                (55, 81 - long_segment_length),  # center junction
                # (55-short_segment_length, 81-long_segment_length),  # left junction
                (55 + short_segment_length, 81 - long_segment_length),  # right junction
            ]
        )
        ax[date_idx, epoch_idx].plot(
            node_positions_right[:, 0], node_positions_right[:, 1], "ko"
        )
        ax[date_idx, epoch_idx].plot([55, 55], [55, 40], color="black")

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
plot_contour_two_distributions(
    np.concatenate(splitting_position["center_to_left"]),
    np.concatenate(splitting_position["center_to_right"]),
    title="Outbound",
    ax=ax[0],
)
plot_contour_two_distributions(
    np.concatenate(splitting_position["left_to_center"]),
    np.concatenate(splitting_position["right_to_center"]),
    title="Inbound",
    ax=ax[1],
)
