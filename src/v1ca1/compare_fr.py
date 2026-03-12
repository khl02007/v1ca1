import spikeinterface.full as si
import numpy as np
import kyutils
import time
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import argparse
import position_tools as pt
import scipy

animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

num_sleep_epochs = 5
num_run_epochs = 4


epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

regions = ["v1", "ca1"]
position_offset = 10


with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")


time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)
speed_threshold = 4  # cm/s


# helper functions for processing
def get_spike_indicator(sorting, timestamps_ephys_all, time):
    spike_indicator = []
    for unit_id in sorting.get_unit_ids():
        spike_times = timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)]
        spike_times = spike_times[(spike_times > time[0]) & (spike_times <= time[-1])]
        spike_indicator.append(
            np.bincount(np.digitize(spike_times, time[1:-1]), minlength=time.shape[0])
        )
    spike_indicator = np.asarray(spike_indicator).T
    return spike_indicator


def get_spike_indicator2(sorting, timestamps_ephys_all, time):
    """
    Bin spikes with `time` as left bin edges.

    Bin i counts spikes in [time[i], time[i+1]) for i=0..n_time-2.
    The last bin (i=n_time-1) counts spikes in [time[-1], time[-1] + dt),
    where dt = time[1] - time[0].

    Parameters
    ----------
    sorting : object
        Must implement get_unit_ids() and get_unit_spike_train(unit_id).
    timestamps_ephys_all : np.ndarray, shape (n_ephys_samples,)
        Timestamps indexed by spike-train indices.
    time : np.ndarray, shape (n_time,)
        Uniformly spaced, monotonically increasing time grid (left edges).

    Returns
    -------
    spike_indicator : np.ndarray, shape (n_time, n_units)
        Spike counts per left-edge bin.
    """
    n_time = time.shape[0]
    dt = time[1] - time[0]
    t0 = time[0]
    t_end_exclusive = time[-1] + dt  # makes the last left-edge bin real

    spike_indicator = []
    for unit_id in sorting.get_unit_ids():
        spike_times = timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)]

        # Keep spikes within [t0, t_end_exclusive)
        mask = (spike_times >= t0) & (spike_times < t_end_exclusive)
        spike_times = spike_times[mask]

        # Left-edge bin index: floor((t - t0) / dt)
        idx = np.floor((spike_times - t0) / dt).astype(np.int64)

        # Robust to tiny float drift
        idx = np.clip(idx, 0, n_time - 1)

        counts = np.bincount(idx, minlength=n_time)
        spike_indicator.append(counts)

    return np.asarray(spike_indicator, dtype=np.int64).T


def estimate_fr(region, epoch, state):
    t_position = timestamps_position[epoch][position_offset:]

    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    position = position_dict[epoch][position_offset:]
    position_sampling_rate = len(position) / (t_position[-1] - t_position[0])

    speed = pt.get_speed(
        position, time=t_position, sampling_frequency=position_sampling_rate, sigma=0.1
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )
    speed_interp = f_speed(time)

    spike_indicator = get_spike_indicator(
        sorting[region], timestamps_ephys_all_ptp, time
    )

    if state == "run":
        running = speed_interp > speed_threshold
        fr = np.sum(spike_indicator[running], axis=0) / (
            np.sum(running) * time_bin_size
        )
    elif state == "immobility":
        immobile = speed_interp <= speed_threshold
        fr = np.sum(spike_indicator[immobile], axis=0) / (
            np.sum(immobile) * time_bin_size
        )

    else:
        fr = np.sum(spike_indicator, axis=0) / (len(time) * time_bin_size)

    return fr


def plot_fr_distribution(state: str):
    fig_save_path = analysis_path / "figs" / "fr_distribution"
    fig_save_path.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(0, 100, 101)
    for region in regions:
        fig, ax = plt.subplots(ncols=len(run_epoch_list), figsize=(12, 4))
        for i, epoch in enumerate(run_epoch_list):
            fr_stim = estimate_fr(region=region, epoch=epoch, state=state)
            ax[i].hist(fr_stim, bins=bin_edges, histtype="step", color="red", alpha=0.7)
            ax[i].set_title(epoch)
        fig.supxlabel(f"Firing rate (Hz)")
        fig.suptitle(f"{animal_name} {date} firing rate distribution")
        fig.savefig(
            fig_save_path / f"{region}_{state}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return None


def plot_fr_stim_vs_dark(state: str):
    fig_save_path = analysis_path / "figs" / "fr_comparison"
    fig_save_path.mkdir(parents=True, exist_ok=True)

    for region in regions:
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        fr_dark = estimate_fr(region=region, epoch=run_epoch_list[3], state=state)
        for i, epoch in enumerate(run_epoch_list[:3]):
            fr_stim = estimate_fr(region=region, epoch=epoch, state=state)
            ax[i].scatter(
                fr_stim,
                fr_dark,
                marker="o",
                edgecolors="black",
                facecolors="none",
                alpha=0.3,
            )
            ax[i].set_xscale("symlog", linthresh=1)
            ax[i].set_yscale("symlog", linthresh=1)
            m = max(np.max(fr_stim), np.max(fr_dark))
            ax[i].plot([0, m], [0, m], "r--")
            ax[i].set_xlim(0, m)
            ax[i].set_ylim(0, m)
            ax[i].set_title(epoch)
            ax[i].set_xlabel(f"Firing rate, {epoch} (Hz)")
        ax[0].set_ylabel(f"Firing rate, dark (Hz)")
        fig.suptitle(f"{animal_name} {date} firing rate comparison")
        fig.savefig(
            fig_save_path / f"{region}_{state}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
    return None


def main():
    state = "run"
    start = time.perf_counter()
    plot_fr_stim_vs_dark(state=state)
    plot_fr_distribution(state=state)
    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
