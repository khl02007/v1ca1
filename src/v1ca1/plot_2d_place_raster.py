# import non_local_detector.analysis as analysis
import spikeinterface.full as si
import numpy as np
import kyutils
import pickle
import matplotlib.pyplot as plt
import scipy
import position_tools as pt

from pathlib import Path


animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / "20240611"

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


time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)
speed_threshold = 4  # cm/s


def plot_2d_place_raster():
    fig_save_dir = analysis_path / "figs" / "place_raster_2d"
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        for unit_id in sorting[region].get_unit_ids():

            fig, ax = plt.subplots(
                figsize=(3 * len(epoch_list), 3 * 3),
                nrows=3,
                ncols=len(epoch_list),
            )

            spike_times_s = timestamps_ephys_all_ptp[
                sorting[region].get_unit_spike_train(unit_id)
            ]

            for a in ax.flatten():
                a.set_xticks([])
                a.set_yticks([])
                a.spines["top"].set_visible(False)
                a.spines["right"].set_visible(False)
                a.spines["left"].set_visible(False)
                a.spines["bottom"].set_visible(False)
                a.set_aspect("equal")
            ax[0, 0].set_ylabel("All spikes")
            ax[1, 0].set_ylabel("Spikes during movement")
            ax[2, 0].set_ylabel("Spikes during immobility")

            for epoch_idx, epoch in enumerate(epoch_list):

                t_position = timestamps_position_dict[epoch][position_offset:]
                position = position_dict[epoch][position_offset:]
                f_position = scipy.interpolate.interp1d(
                    t_position,
                    position,
                    axis=0,
                    bounds_error=False,
                    kind="linear",
                )
                position_sampling_rate = len(position) / (
                    t_position[-1] - t_position[0]
                )
                speed = pt.get_speed(
                    position,
                    time=t_position,
                    sampling_frequency=position_sampling_rate,
                    sigma=0.1,
                )
                f_speed = scipy.interpolate.interp1d(
                    t_position, speed, axis=0, bounds_error=False, kind="linear"
                )

                ax[0, epoch_idx].plot(
                    position[:, 0],
                    position[:, 1],
                    color="gray",
                    alpha=0.2,
                )
                ax[1, epoch_idx].plot(
                    position[:, 0],
                    position[:, 1],
                    color="gray",
                    alpha=0.2,
                )
                ax[2, epoch_idx].plot(
                    position[:, 0],
                    position[:, 1],
                    color="gray",
                    alpha=0.2,
                )

                spike_times_epoch = spike_times_s[
                    (spike_times_s > t_position[0]) & (spike_times_s <= t_position[-1])
                ]
                spike_times_movement = spike_times_s[
                    (spike_times_s > t_position[0])
                    & (spike_times_s <= t_position[-1])
                    & (f_speed(spike_times_s) > speed_threshold)
                ]
                spike_times_immobility = spike_times_s[
                    (spike_times_s > t_position[0])
                    & (spike_times_s <= t_position[-1])
                    & (f_speed(spike_times_s) <= speed_threshold)
                ]

                position_given_spike = f_position(spike_times_epoch)
                position_given_spike_movement = f_position(spike_times_movement)
                position_given_spike_immobility = f_position(spike_times_immobility)

                ax[0, epoch_idx].scatter(
                    position_given_spike[:, 0],
                    position_given_spike[:, 1],
                    color="black",
                    s=0.75,
                    alpha=0.5,
                    marker=".",
                )
                ax[1, epoch_idx].scatter(
                    position_given_spike_movement[:, 0],
                    position_given_spike_movement[:, 1],
                    color="black",
                    s=0.75,
                    alpha=0.5,
                    marker=".",
                )

                ax[2, epoch_idx].scatter(
                    position_given_spike_immobility[:, 0],
                    position_given_spike_immobility[:, 1],
                    color="black",
                    s=0.75,
                    alpha=0.5,
                    marker=".",
                )

                ax[0, epoch_idx].set_title(epoch)

            fig.suptitle(f"{animal_name} {region} {unit_id}")
            fig_save_path = fig_save_dir / f"{region}_{unit_id}.png"

            fig.savefig(
                fig_save_path,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    return None


def main():
    plot_2d_place_raster()


if __name__ == "__main__":
    main()
