import replay_trajectory_classification as rtc
import spikeinterface.full as si
import numpy as np
import kyutils
import pickle

import matplotlib.pyplot as plt
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

import os
import argparse


animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]

regions = ["v1", "ca1"]

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

position_offset = 10

temporal_bin_size_s = 2e-3
sampling_rate = int(1 / temporal_bin_size_s)
speed_threshold = 4  # cm/s

sleep_epoch_list = [epoch for epoch in epoch_list if epoch not in run_epoch_list]


def load_classifier_dict(
    use_half="all",
    position_std=4.0,
    movement=True,
    place_bin_size=1.0,
    movement_var=4.0,
    discrete_var="switching",
):
    classifier_save_dir = analysis_path / "classifier_2d_sleep_box"
    classifier_dict = {}
    for region in regions:
        classifier_dict[region] = {}
        for sleep_epoch in sleep_epoch_list:
            classifier_path = classifier_save_dir / (
                f"classifier_{region}_{date}_{sleep_epoch}_2d_sleep_box"
                f"_use_half_{use_half}"
                f"_movement_{movement}"
                f"_fit_position_std_{position_std}"
                f"_discrete_var_{discrete_var}"
                f"_place_bin_size_{place_bin_size}"
                f"_movement_var_{movement_var}"
                ".pkl"
            )
            classifier_dict[region][sleep_epoch] = (
                rtc.SortedSpikesClassifier.load_model(filename=classifier_path)
            )
    return classifier_dict


width_ratios = len(sleep_epoch_list) * [1]
width_ratios.append(0.1)


def plot_2d_place_field_sleep_box(position_std):

    fig_save_dir = analysis_path / "figs" / "place_field_2d_sleep_box"
    fig_save_dir.mkdir(parents=True, exist_ok=True)

    classifier_dict = load_classifier_dict(position_std=position_std)

    for region in regions:

        n_units = (
            classifier_dict[region][sleep_epoch_list[0]].place_fields_[("", 0)].shape[1]
        )

        for unit_idx in range(n_units):

            max_fr = []
            for epoch in sleep_epoch_list:

                max_fr.append(
                    np.nanmax(
                        classifier_dict[region][epoch]
                        .place_fields_[("", 0)]
                        .to_numpy()[:, unit_idx]
                        / temporal_bin_size_s
                    )
                )

            max_fr = int(np.ceil(np.max(max_fr)))

            fig, ax = plt.subplots(
                ncols=len(sleep_epoch_list) + 1,
                figsize=(3 * len(sleep_epoch_list) + 0.3, 3),
                gridspec_kw={
                    "width_ratios": width_ratios,
                },
            )

            for epoch_idx, epoch in enumerate(sleep_epoch_list):

                x = (
                    classifier_dict[region][epoch]
                    .place_fields_[("", 0)]
                    .x_position.to_numpy()
                )
                y = (
                    classifier_dict[region][epoch]
                    .place_fields_[("", 0)]
                    .y_position.to_numpy()
                )
                c = (
                    classifier_dict[region][epoch]
                    .place_fields_[("", 0)]
                    .to_numpy()[:, unit_idx]
                    / temporal_bin_size_s
                )

                # ax[epoch_idx].plot(
                #     position_dict[epoch][10:, 0],
                #     position_dict[epoch][10:, 1],
                #     color="black",
                #     alpha=0.1,
                #     linewidth=2.5,
                # )

                sc = ax[epoch_idx].scatter(
                    x,
                    y,
                    c=c,
                    s=9.3,
                    marker="s",
                    vmin=0,
                    vmax=max_fr,
                    cmap="inferno",
                )

                ax[epoch_idx].set_title(epoch)

                ax[epoch_idx].set_xlim(0, 35)
                ax[epoch_idx].set_ylim(0, 35)
                ax[epoch_idx].set_aspect("equal")
                # ax[epoch_idx].spines["top"].set_visible(False)
                # ax[epoch_idx].spines["right"].set_visible(False)
                # ax[epoch_idx].spines["left"].set_visible(False)
                # ax[epoch_idx].spines["bottom"].set_visible(False)
                ax[epoch_idx].set_xlabel("")
                ax[epoch_idx].set_xticks([])
                ax[epoch_idx].set_yticks([])

                # ax[epoch_idx].axis("off")

            cbar = fig.colorbar(sc, cax=ax[-1], shrink=0.5)
            ax[-1].set_ylim([0, max_fr])
            ax[-1].set_yticks([0, max_fr])
            ax[-1].set_yticklabels([0, max_fr])
            cbar.set_label("Hz", rotation=0)

            fig.suptitle(f"{animal_name} {date} {region} {unit_idx}")

            fig.savefig(
                fig_save_dir / f"{region}_{unit_idx}.png",
                dpi=300,
            )

            plt.close(fig)

    return None


# def parse_arguments():

#     parser = argparse.ArgumentParser(description="Fit 2D decoder on CA1 and V1 spikes")
#     parser.add_argument("--position_std", type=float, help="width of gaussian")

#     return parser.parse_args()


def main():
    # args = parse_arguments()
    plot_2d_place_field_sleep_box(
        position_std=2.0
        # position_std=args.position_std,
    )


if __name__ == "__main__":
    main()
