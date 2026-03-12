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

eps = 1e-12

lfp_channel = 12
ripple_channel = 16 + 32 * 2 + 128 * 2


# define track graph for linearization
node_positions = np.array(
    [
        (55, 81),  # center well
        (23, 81),  # left well
        (87, 81),  # right well
        (55, 10),  # center junction
        (23, 10),  # left junction
        (87, 10),  # right junction
    ]
)

edges = np.array(
    [
        (0, 3),
        (3, 4),
        (3, 5),
        (4, 1),
        (5, 2),
    ]
)

linear_edge_order = [
    (0, 3),
    (3, 4),
    (4, 1),
    (3, 5),
    (5, 2),
]
linear_edge_spacing = 10
track_graph = tl.make_track_graph(node_positions, edges)


def get_time(epoch: str, time_bin_size: float, ptp: bool = True):
    sampling_rate = int(1 / (time_bin_size))

    # define reference time offset and subtract it from the timestamps
    t_position = timestamps_position_dict[epoch][position_offset:]
    if not ptp:
        t_position = t_position - timestamps_ephys[epoch][0]

    # define time vector for decoding (temporal resolution: 2 ms)
    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    return time


def get_spike_indicator(epoch, sorting, t_all, time, ptp):
    if not ptp:
        t_all = t_all - timestamps_ephys[epoch][0]
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
    epoch, time_bin_size, ptp, sorting, t_all, temporal_overlap
):
    if temporal_overlap:
        time1 = get_time(epoch, time_bin_size, ptp)
        time2 = time1 + time_bin_size / 2
        time2 = time2[:-1]
        time = np.sort(np.concatenate([time1, time2]))

        spike_indicator1 = get_spike_indicator(epoch, sorting, t_all, time1, ptp)
        spike_indicator2 = get_spike_indicator(epoch, sorting, t_all, time2, ptp)
        spike_indicator = np.zeros((len(time), spike_indicator1.shape[1]))
        spike_indicator[0::2] = spike_indicator1
        spike_indicator[1::2] = spike_indicator2
    else:
        time = get_time(epoch, time_bin_size, ptp)
        spike_indicator = get_spike_indicator(epoch, sorting, t_all, time, ptp)

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


def get_firing_rate_matrix(sorting, epoch, bin_size, zscore=True):

    start_time = timestamps_ephys[epoch][0]
    stop_time = timestamps_ephys[epoch][-1]

    bin_edges = np.arange(start_time, stop_time, bin_size)
    time = bin_edges[:-1] + bin_size / 2

    fr = np.zeros((len(sorting.get_unit_ids()), len(bin_edges) - 1))

    for k, unit_id in enumerate(sorting.get_unit_ids()):

        spike_times = timestamps_ephys_all_ptp[sorting.get_unit_spike_train(unit_id)]
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


def plot_sleep_phases():
    fig_save_dir = analysis_path / "figs" / "sleep"
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    nperseg = 256  # Length of each segment for FFT
    noverlap = 128  # Overlap between segments
    cutoff = 70  # Hz, frequency below which to keep for spectrogram

    nwb_file_name = f"{animal_name}{date}.nwb"
    recording = si.read_nwb_recording(nwb_file_base_path / nwb_file_name)
    fs = recording.get_sampling_frequency()
    downsample_factor = int(fs // 150)
    fr_bin_size = 100e-3
    for epoch in epoch_list:

        print(f"plotting {date} {epoch}")

        i1 = np.searchsorted(timestamps_ephys_all_ptp, timestamps_ephys[epoch][0])
        i2 = np.searchsorted(timestamps_ephys_all_ptp, timestamps_ephys[epoch][-1])
        ca1_signal = recording.get_traces(
            channel_ids=[ripple_channel],
            return_in_uV=True,
            start_frame=i1,
            end_frame=i2,
        ).flatten()
        v1_signal = recording.get_traces(
            channel_ids=[lfp_channel],
            return_in_uV=True,
            start_frame=i1,
            end_frame=i2,
        ).flatten()

        t = timestamps_ephys_all_ptp[i1:i2]

        ripple_lfp_time, ripple_lfp = butter_filter_and_decimate(
            timestamps=t,
            data=ca1_signal,
            sampling_frequency=fs,
            new_sampling_frequency=1000,
            lowcut=150.0,
            highcut=250.0,
            order=4,
        )

        ca1_filtered_signal = lowpass_filter(ca1_signal, cutoff, fs)
        v1_filtered_signal = lowpass_filter(v1_signal, cutoff, fs)

        v1_downsampled_signal = decimate(v1_filtered_signal, downsample_factor)
        ca1_downsampled_signal = decimate(ca1_filtered_signal, downsample_factor)
        fs_downsampled = fs // downsample_factor
        t_downsampled = t[::downsample_factor]

        v1_frequencies, v1_times, v1_Sxx = spectrogram(
            v1_downsampled_signal,
            fs_downsampled,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        v1_time_spectrogram = (
            v1_times * (nperseg - noverlap)
        ) / fs_downsampled  # Time in seconds in the downsampled signal

        v1_Sxx_filtered = v1_Sxx[v1_frequencies < 60, :]
        Sxx_flat = v1_Sxx_filtered.T  # Transpose to have time as rows for PCA
        pca = PCA(n_components=1)
        v1_lfp_spectrogram_pc1 = zscore(pca.fit_transform(Sxx_flat)).squeeze()

        ca1_frequencies, ca1_times, ca1_Sxx = spectrogram(
            ca1_downsampled_signal,
            fs_downsampled,
            nperseg=nperseg,
            noverlap=noverlap,
        )

        theta_pow = ca1_Sxx[(ca1_frequencies > 5) & (ca1_frequencies <= 10)].sum(axis=0)
        delta_pow = ca1_Sxx[(ca1_frequencies > 0.5) & (ca1_frequencies <= 4)].sum(
            axis=0
        )
        ca1_theta_delta = theta_pow / np.maximum(delta_pow, eps)

        tt, spike_indicator_ca1 = get_time_spike_indicator(
            epoch=epoch,
            time_bin_size=0.002,
            ptp=True,
            sorting=sorting["ca1"],
            t_all=timestamps_ephys_all_ptp,
            temporal_overlap=False,
        )

        tt, spike_indicator_v1 = get_time_spike_indicator(
            epoch=epoch,
            time_bin_size=0.002,
            ptp=True,
            sorting=sorting["v1"],
            t_all=timestamps_ephys_all_ptp,
            temporal_overlap=False,
        )

        multiunit_ca1 = rd.get_multiunit_population_firing_rate(
            spike_indicator_ca1,
            sampling_frequency=int(1 / np.median(np.diff(tt))),
        )
        multiunit_v1 = rd.get_multiunit_population_firing_rate(
            spike_indicator_v1,
            sampling_frequency=int(1 / np.median(np.diff(tt))),
        )

        position = position_dict[epoch][position_offset:]
        t_position = timestamps_position_dict[epoch][position_offset:]
        position_sampling_rate = len(position) / (t_position[-1] - t_position[0])

        speed = pt.get_speed(
            position,
            time=t_position,
            sampling_frequency=position_sampling_rate,
            sigma=0.1,
        )

        fr_ca1, time = get_firing_rate_matrix(
            sorting["ca1"], epoch, bin_size=fr_bin_size, zscore=True
        )
        fr_v1, time = get_firing_rate_matrix(
            sorting["v1"], epoch, bin_size=fr_bin_size, zscore=True
        )

        sliced_sorting_ca1 = sorting["ca1"].time_slice(
            start_time=t[0],
            end_time=t[-1],
        )
        sliced_sorting_v1 = sorting["v1"].time_slice(
            start_time=t[0],
            end_time=t[-1],
        )

        order_ca1 = np.argsort(
            [
                len(sliced_sorting_ca1.get_unit_spike_train(unit_id))
                for unit_id in sliced_sorting_ca1.get_unit_ids()
            ]
        )
        order_v1 = np.argsort(
            [
                len(sliced_sorting_v1.get_unit_spike_train(unit_id))
                for unit_id in sliced_sorting_v1.get_unit_ids()
            ]
        )

        fr_ca1 = fr_ca1[order_ca1]
        fr_v1 = fr_v1[order_v1]

        fig, ax = plt.subplots(
            nrows=7,
            figsize=(40, 7 * 3),
            gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 3, 3]},
        )

        if epoch[3] == "r":

            position_df = tl.get_linearized_position(
                position=position,
                track_graph=track_graph,
                edge_order=linear_edge_order,
                edge_spacing=linear_edge_spacing,
            )
            linear_position = position_df["linear_position"]

            ax[0].plot(t_position, linear_position)
            ax[0].twinx().plot(t_position, speed, "k")
            ax[0].set_ylabel("Position (cm)\nSpeed (cm/s)")
        else:
            ax[0].plot(t_position, speed, "k")
            ax[0].set_ylabel("Speed (cm/s)")

        ax[1].plot(v1_times + t[0], v1_lfp_spectrogram_pc1)
        ax[2].plot(ca1_times + t[0], ca1_theta_delta)
        ax[3].plot(ripple_lfp_time, ripple_lfp)
        ax[4].plot(tt, np.asarray(multiunit_ca1).squeeze(), label="CA1")
        ax[4].plot(tt, np.asarray(multiunit_v1).squeeze(), label="V1")
        ax[4].legend()
        ax[5].imshow(
            fr_ca1,
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
            extent=[time[0], time[-1], 0, fr_ca1.shape[0]],
        )
        ax[6].imshow(
            fr_v1,
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
            extent=[time[0], time[-1], 0, fr_v1.shape[0]],
        )

        ax[1].set_ylabel("V1 LFP spectrogram\nPC1 (zscore)")
        ax[2].set_ylabel("CA1 Theta / Delta")
        ax[3].set_ylabel("CA1 ripple band LFP ($\mu$V)")
        ax[4].set_ylabel("Multiunit firing rate (Hz)")
        ax[5].set_ylabel("Firing rate CA1 (zscore)")
        ax[6].set_ylabel("Firing rate V1 (zscore)")

        for a in ax:
            a.set_xlim([time[0], time[-1]])
        ax[1].set_ylim([0, 10])
        ax[2].set_ylim([0, 50])
        ax[-1].set_xlabel("Time (s)")

        ax[0].set_title(f"{date} {epoch} sleep phase")

        fig.savefig(
            fig_save_dir / f"{epoch}.pdf",
            # dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot behavioral and neural data during sleep"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    plot_sleep_phases()


if __name__ == "__main__":
    main()
