import replay_trajectory_classification as rtc
import spikeinterface.full as si
import numpy as np
import kyutils
import pickle
import scipy

import matplotlib.pyplot as plt
import position_tools as pt
import pandas as pd
import track_linearization as tl
import xarray as xr
import trajectory_analysis_tools as tat

from pathlib import Path

import track_linearization as tl

import numpy as np
import matplotlib.pyplot as plt

import os
import argparse

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


def load_classifier_dict(
    use_half="all",
    movement=True,
    speed_std=1.0,
    discrete_var="switching",
    speed_bin_size=1.0,
    movement_var=4.0,
):
    classifier_save_dir = analysis_path / "classifier_speed"

    classifier_dict = {}
    for region in regions:
        classifier_dict[region] = {}
        for epoch in run_epoch_list:
            classifier_save_path = classifier_save_dir / (
                f"classifier_{region}_{date}_{epoch}"
                f"_speed_use_half_{use_half}"
                f"_movement_{movement}"
                f"_fit_speed_std_{speed_std}"
                f"_discrete_var_{discrete_var}"
                f"_speed_bin_size_{speed_bin_size}"
                f"_movement_var_{movement_var}"
                ".pkl"
            )

            classifier_dict[region][epoch] = rtc.SortedSpikesClassifier.load_model(
                filename=classifier_save_path
            )

    return classifier_dict


def plot_speed_field(
    speed_std=1.0,
):

    fig_save_path = analysis_path / "figs" / "speed_field"
    fig_save_path.mkdir(parents=True, exist_ok=True)

    classifier_dict = load_classifier_dict(speed_std=speed_std)

    for region in regions:

        for unit_id in sorting[region].get_unit_ids():

            fig, ax = plt.subplots(
                nrows=len(run_epoch_list) * 2 + 1,
                ncols=2,
                figsize=(12, 10),
                gridspec_kw={"height_ratios": [1, 1, 1, 1, 0.5, 1, 1, 1, 1]},
            )

            spike_times_frames = sorting[region].get_unit_spike_train(unit_id)
            spike_times_s = timestamps_ephys_all_ptp[spike_times_frames]

            pf_axes = []
            pf_max_vals = []

            for trajectory_type in trajectory_types:

                if trajectory_type == "center_to_left":
                    ax_col_offset = 0
                    ax_row_offset = 0
                elif trajectory_type == "left_to_center":
                    ax_col_offset = 0
                    ax_row_offset = len(run_epoch_list) + 1
                elif trajectory_type == "center_to_right":
                    ax_col_offset = 1
                    ax_row_offset = 0
                elif trajectory_type == "right_to_center":
                    ax_col_offset = 1
                    ax_row_offset = len(run_epoch_list) + 1
                for epoch_idx, epoch in enumerate(run_epoch_list):

                    timestamps_position = timestamps_position_dict[epoch][
                        position_offset:
                    ]
                    position = position_dict[epoch][position_offset:]

                    position_sampling_rate = len(position) / (
                        timestamps_position[-1] - timestamps_position[0]
                    )
                    speed = pt.get_speed(
                        position,
                        time=timestamps_position,
                        sampling_frequency=position_sampling_rate,
                        sigma=0.1,
                    )
                    f_speed = scipy.interpolate.interp1d(
                        timestamps_position,
                        speed,
                        axis=0,
                        bounds_error=False,
                        kind="linear",
                    )

                    speed_list = []
                    for traj_start_time, traj_end_time in trajectory_times[epoch][
                        trajectory_type
                    ]:
                        spike_times_trajectory = spike_times_s[
                            (spike_times_s > traj_start_time)
                            & (spike_times_s <= traj_end_time)
                        ]

                        if len(spike_times_trajectory) == 0:
                            speed_list.append([])
                            continue

                        speed_list.append(f_speed(spike_times_trajectory))

                    for i, pos_seg in enumerate(speed_list):
                        if len(pos_seg) == 0:
                            continue
                        ax[ax_row_offset + epoch_idx, ax_col_offset].plot(
                            pos_seg,
                            np.ones(len(pos_seg)) + i,
                            "|",
                            color="black",
                            markersize=1,
                        )

                    num_trials = sum(len(seg) > 0 for seg in speed_list)
                    ax[ax_row_offset + epoch_idx, ax_col_offset].set_ylim(
                        [0, max(1, num_trials) + 1]
                    )

                    ax_pf = ax[ax_row_offset + epoch_idx, ax_col_offset].twinx()
                    pf_axes.append(ax_pf)
                    ax_pf.plot(
                        classifier_dict[region][epoch]
                        .place_fields_[("", 0)]
                        .position.to_numpy(),
                        classifier_dict[region][epoch]
                        .place_fields_[("", 0)][:, unit_id]
                        .to_numpy()
                        / time_bin_size,
                        color="blue",
                    )
                    pf_max_vals.append(
                        np.max(
                            classifier_dict[region][epoch]
                            .place_fields_[("", 0)][:, unit_id]
                            .to_numpy()
                            / time_bin_size
                        )
                    )

                    # alpha = 0.4
                    # color = "gray"

                    # ax[ax_row_offset + epoch_idx, ax_col_offset].axvline(
                    #     long_segment_length + diagonal_segment_length * 0.5,
                    #     color=color,
                    #     alpha=alpha,
                    # )
                    # ax[ax_row_offset + epoch_idx, ax_col_offset].axvline(
                    #     long_segment_length
                    #     + diagonal_segment_length * 1.5
                    #     + short_segment_length,
                    #     color=color,
                    #     alpha=alpha,
                    # )
                    ax[ax_row_offset + epoch_idx, ax_col_offset].set_yticklabels("")

                    # ax[ax_row_offset + epoch_idx, ax_col_offset].set_xticks(
                    #     [
                    #         0,
                    #         long_segment_length + diagonal_segment_length * 0.5,
                    #         long_segment_length
                    #         + short_segment_length
                    #         + diagonal_segment_length * 1.5,
                    #         long_segment_length * 2
                    #         + short_segment_length
                    #         + diagonal_segment_length * 2,
                    #     ]
                    # )

                    if epoch_idx != 3:
                        ax[ax_row_offset + epoch_idx, ax_col_offset].set_xticklabels("")
                    # else:
                    #     ax[ax_row_offset + epoch_idx, ax_col_offset].set_xticklabels(
                    #         [
                    #             0,
                    #             np.round(
                    #                 long_segment_length + diagonal_segment_length * 0.5,
                    #                 1,
                    #             ),
                    #             np.round(
                    #                 long_segment_length
                    #                 + short_segment_length
                    #                 + diagonal_segment_length * 1.5,
                    #                 1,
                    #             ),
                    #             np.round(
                    #                 long_segment_length * 2
                    #                 + short_segment_length
                    #                 + diagonal_segment_length * 2,
                    #                 1,
                    #             ),
                    #         ]
                    #     )

                    ax[ax_row_offset + epoch_idx, ax_col_offset].set_xlim([0, 75])

                    if epoch_idx == 0:
                        ax[ax_row_offset + epoch_idx, ax_col_offset].set_title(
                            trajectory_type
                        )

            if len(pf_max_vals) == 0:
                pf_max = 1e-6
            else:
                # nan-safe, and enforce strictly positive upper bound
                pf_max = max(1e-6, np.round(float(np.nanmax(pf_max_vals)), 1))
            for pf_ax in pf_axes:
                pf_ax.set_ylim([0, pf_max])
                pf_ax.set_yticks([0, pf_max])
                pf_ax.set_yticklabels([0, pf_max])

            ax[4, 0].axis("off")
            ax[4, 1].axis("off")

            ax[0, 0].set_ylabel("Stim1")
            ax[1, 0].set_ylabel("Stim2")
            ax[2, 0].set_ylabel("Stim3")
            ax[3, 0].set_ylabel("Dark")

            ax[-1, -1].set_xlabel("Speed (cm/s)")
            ax[-1, -1].set_ylabel("Trials")

            fig.suptitle(f"{animal_name} {date} {region} {unit_id}")

            fig.savefig(
                fig_save_path / f"{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    return None


def parse_arguments():

    parser = argparse.ArgumentParser(description="Fit 2D decoder on CA1 and V1 spikes")
    parser.add_argument(
        "--speed_std",
        type=float,
        help="std of gaussian kernel for estimating speed fields, in cm/s",
    )

    return parser.parse_args()


def main():
    # args = parse_arguments()
    speed_std = 1.0
    plot_speed_field(
        speed_std=speed_std,
    )


if __name__ == "__main__":
    main()
