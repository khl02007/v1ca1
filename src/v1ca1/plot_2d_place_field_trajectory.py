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

from scipy.ndimage import gaussian_filter1d
from pathlib import Path

import non_local_detector
import track_linearization as tl
from track_linearization import make_track_graph, plot_track_graph

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

import os
import argparse


from matplotlib.patches import FancyArrowPatch

animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date


regions = ["ca1", "v1"]
# regions = ["v1"]

time_bin_size = 2e-3
trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

# run_epoch_list = ["02_r1", "06_r3", "08_r4"]
epoch_labels = ["Stim 1", "Stim 2", "Stim 3", "Dark"]
traj_labels = ["Outbound-right", "Outbound-left", "Inbound-right", "Inbound-left"]

width_ratios = len(trajectory_types) * [1]
width_ratios.append(0.1)


with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

# with open(data_path / "run_epochs" / "reliable_pf_cells.pkl", "rb") as f:
#     reliable_pf_cells = pickle.load(f)

classifier_save_dir = analysis_path / "classifier_2d_trajectory"


def load_classifier_dict(
    use_half="all",
    position_std=4.0,
    movement=True,
    place_bin_size=1.0,
    movement_var=4.0,
    discrete_var="switching",
):

    classifier_dict = {}
    for region in regions:
        classifier_dict[region] = {}
        for epoch in run_epoch_list:
            classifier_dict[region][epoch] = {}
            for trajectory_type in trajectory_types:
                classifier_path = classifier_save_dir / (
                    f"classifier_{region}_{date}_{epoch}"
                    f"_2d_{trajectory_type}"
                    f"_use_half_{use_half}"
                    f"_movement_{movement}"
                    f"_fit_position_std_{position_std}"
                    f"_discrete_var_{discrete_var}"
                    f"_place_bin_size_{place_bin_size}"
                    f"_movement_var_{movement_var}"
                    ".pkl"
                )

                classifier_dict[region][epoch][trajectory_type] = (
                    rtc.SortedSpikesClassifier.load_model(filename=classifier_path)
                )

    return classifier_dict


def plot_2d_place_field_trajectory(position_std=4.0):
    fig_save_path = analysis_path / "figs" / "place_field_2d_trajectory"
    fig_save_path.mkdir(parents=True, exist_ok=True)

    classifier_dict = load_classifier_dict(position_std=position_std)

    for region in regions:

        n_units = (
            classifier_dict[region][run_epoch_list[0]][trajectory_types[0]]
            .place_fields_[("", 0)]
            .shape[1]
        )

        for unit_idx in range(n_units):

            max_fr = []
            for epoch in run_epoch_list:
                for trajectory_type in trajectory_types:
                    max_fr.append(
                        np.nanmax(
                            classifier_dict[region][epoch][trajectory_type]
                            .place_fields_[("", 0)]
                            .to_numpy()[:, unit_idx]
                            / time_bin_size
                        )
                    )

            max_fr = int(np.ceil(np.max(max_fr)))

            fig, ax = plt.subplots(
                nrows=len(run_epoch_list),
                ncols=len(trajectory_types) + 1,
                figsize=(3 * len(trajectory_types) + 0.3, 3 * len(run_epoch_list)),
                gridspec_kw={
                    "width_ratios": width_ratios,
                },
            )

            for epoch_idx, epoch in enumerate(run_epoch_list):
                for trajectory_idx, trajectory_type in enumerate(trajectory_types):

                    x = (
                        classifier_dict[region][epoch][trajectory_type]
                        .place_fields_[("", 0)]
                        .x_position.to_numpy()
                    )
                    y = (
                        classifier_dict[region][epoch][trajectory_type]
                        .place_fields_[("", 0)]
                        .y_position.to_numpy()
                    )
                    c = (
                        classifier_dict[region][epoch][trajectory_type]
                        .place_fields_[("", 0)]
                        .to_numpy()[:, unit_idx]
                        / time_bin_size
                    )

                    ax[epoch_idx, trajectory_idx].plot(
                        position_dict[epoch][10:, 0],
                        position_dict[epoch][10:, 1],
                        color="black",
                        alpha=0.1,
                        linewidth=2.5,
                    )

                    sc = ax[epoch_idx, trajectory_idx].scatter(
                        x,
                        y,
                        c=c,
                        s=0.6,
                        marker="s",
                        vmin=0,
                        vmax=max_fr,
                        cmap="inferno",
                    )

                    ax[epoch_idx, 0].set_ylabel(epoch_labels[epoch_idx], fontsize=12)

                    if epoch_idx == 0:
                        ax[epoch_idx, trajectory_idx].set_title(
                            traj_labels[trajectory_idx]
                        )

                    if trajectory_idx == 0:

                        # Define the start and end points of the arrow
                        start_point = (65, 50)
                        end_point = (80, 50)

                        arrow = FancyArrowPatch(
                            start_point,
                            end_point,
                            connectionstyle="arc3,rad=3",  # 'arc3
                            # connectionstyle="angle3,angleA=90,angleB=0",  # Gradual curve (angleA controls the curve direction)
                            arrowstyle="-|>",  # Larger arrowhead
                            mutation_scale=10,
                            linewidth=2,
                            shrinkA=10,
                            shrinkB=0,
                            color="black",
                        )

                        ax[epoch_idx, trajectory_idx].add_patch(arrow)
                    elif trajectory_idx == 2:
                        # Define the start and end points of the arrow
                        start_point = (80, 50)
                        end_point = (65, 50)

                        arrow = FancyArrowPatch(
                            start_point,
                            end_point,
                            connectionstyle="arc3,rad=-3.0",  # 'arc3
                            arrowstyle="-|>",  # Larger arrowhead
                            mutation_scale=10,
                            shrinkA=10,
                            shrinkB=0,
                            linewidth=2,
                            color="black",
                        )
                        ax[epoch_idx, trajectory_idx].add_patch(arrow)

                    elif trajectory_idx == 1:
                        # Define the start and end points of the arrow
                        start_point = (46, 50)
                        end_point = (31, 50)

                        arrow = FancyArrowPatch(
                            start_point,
                            end_point,
                            connectionstyle="arc3,rad=-3.0",  # 'arc3
                            arrowstyle="-|>",  # Larger arrowhead
                            mutation_scale=10,
                            shrinkA=10,
                            shrinkB=0,
                            linewidth=2,
                            color="black",
                        )
                        ax[epoch_idx, trajectory_idx].add_patch(arrow)

                    elif trajectory_idx == 3:
                        # Define the start and end points of the arrow
                        start_point = (31, 50)
                        end_point = (46, 50)

                        arrow = FancyArrowPatch(
                            start_point,
                            end_point,
                            connectionstyle="arc3,rad=3.0",  # 'arc3
                            arrowstyle="-|>",  # Larger arrowhead
                            mutation_scale=10,
                            shrinkA=10,
                            shrinkB=0,
                            linewidth=2,
                            color="black",
                        )
                        ax[epoch_idx, trajectory_idx].add_patch(arrow)

                    ax[epoch_idx, trajectory_idx].set_xlim(0, 100)
                    ax[epoch_idx, trajectory_idx].set_ylim(0, 100)
                    ax[epoch_idx, trajectory_idx].set_aspect("equal")
                    ax[epoch_idx, trajectory_idx].spines["top"].set_visible(False)
                    ax[epoch_idx, trajectory_idx].spines["right"].set_visible(False)
                    ax[epoch_idx, trajectory_idx].spines["left"].set_visible(False)
                    ax[epoch_idx, trajectory_idx].spines["bottom"].set_visible(False)
                    ax[epoch_idx, trajectory_idx].set_xlabel("")
                    ax[epoch_idx, trajectory_idx].set_xticks([])
                    ax[epoch_idx, trajectory_idx].set_yticks([])

                    if epoch_idx < len(run_epoch_list) - 1:
                        ax[epoch_idx, -1].axis("off")

            cbar = fig.colorbar(sc, cax=ax[-1, -1], shrink=0.5)
            ax[-1, -1].set_ylim([0, max_fr])
            ax[-1, -1].set_yticks([0, max_fr])
            ax[-1, -1].set_yticklabels([0, max_fr])
            cbar.set_label("Hz", rotation=0)

            fig.suptitle(f"{animal_name} {date} {region} {unit_idx}")

            fig.savefig(
                fig_save_path / f"{region}_{unit_idx}.png",
                dpi=300,
            )

            plt.close(fig)

    return None


# def parse_arguments():

#     parser = argparse.ArgumentParser(description="Fit 2D decoder on CA1 and V1 spikes")
#     parser.add_argument("--date_idx", type=int, help="width of gaussian")
#     parser.add_argument("--position_std", type=float, help="width of gaussian")

#     return parser.parse_args()


def main():
    # args = parse_arguments()
    position_std = 4.0
    plot_2d_place_field_trajectory(
        position_std=position_std,
    )


if __name__ == "__main__":
    main()
