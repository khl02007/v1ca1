# import non_local_detector.analysis as analysis
import spikeinterface.full as si
import numpy as np
import kyutils
import pickle
import matplotlib.pyplot as plt
import scipy

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


time_bin_size = 2e-3
sampling_rate = int(1 / time_bin_size)


def plot_2d_place_raster():
    fig_save_dir = analysis_path / "figs" / "place_raster_2d_trajectory"
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        for unit_id in sorting[region].get_unit_ids():

            fig, ax = plt.subplots(
                figsize=(3 * len(trajectory_types), 3 * len(run_epoch_list)),
                ncols=len(trajectory_types),
                nrows=len(run_epoch_list),
            )

            spike_times_s = timestamps_ephys_all_ptp[
                sorting[region].get_unit_spike_train(unit_id)
            ]
            for trajectory_type_idx, trajectory_type in enumerate(trajectory_types):

                for epoch_idx, epoch in enumerate(run_epoch_list):

                    t_position = timestamps_position_dict[epoch][position_offset:]
                    position = position_dict[epoch][position_offset:]

                    ax[epoch_idx, trajectory_type_idx].plot(
                        position[:, 0],
                        position[:, 1],
                        color="gray",
                        alpha=0.2,
                    )

                    f_position = scipy.interpolate.interp1d(
                        t_position,
                        position,
                        axis=0,
                        bounds_error=False,
                        kind="linear",
                    )

                    for traj_start_time, traj_end_time in trajectory_times[epoch][
                        trajectory_type
                    ]:
                        spike_times_trajectory = spike_times_s[
                            (spike_times_s > traj_start_time)
                            & (spike_times_s <= traj_end_time)
                        ]

                        position_given_spike = f_position(spike_times_trajectory)
                        ax[epoch_idx, trajectory_type_idx].scatter(
                            position_given_spike[:, 0],
                            position_given_spike[:, 1],
                            color="black",
                            s=0.75,
                            alpha=0.5,
                            marker=".",
                        )

                    ax[epoch_idx, trajectory_type_idx].set_xticks([])
                    ax[epoch_idx, trajectory_type_idx].set_yticks([])

                    ax[epoch_idx, trajectory_type_idx].spines["top"].set_visible(False)
                    ax[epoch_idx, trajectory_type_idx].spines["right"].set_visible(
                        False
                    )
                    ax[epoch_idx, trajectory_type_idx].spines["left"].set_visible(False)
                    ax[epoch_idx, trajectory_type_idx].spines["bottom"].set_visible(
                        False
                    )

                    ax[epoch_idx, trajectory_type_idx].set_aspect("equal")

                    if epoch_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_title(trajectory_type)
                    if trajectory_type_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_ylabel(epoch)

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
