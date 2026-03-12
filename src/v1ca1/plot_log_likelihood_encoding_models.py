import numpy as np
import kyutils
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import argparse


animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)
trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]
regions = ["v1", "ca1"]


with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)

model_labels = ["trajectory", "speed", "headdir", "headdir_velocity", "headdir_speed"]

diff_labels = [
    "place − speed",
    "place − headdir",
    "place − headdir\nvelocity",
    "place − headdir\nspeed",
]

feature_sizes = [0.5, 1.0, 2.0, 4.0, 8.0]


def load_log_likelihoods(feature_std=4.0):

    log_likelihood_place = {}
    log_likelihood_speed = {}
    log_likelihood_headdir = {}
    log_likelihood_headdir_velocity = {}
    log_likelihood_headdir_speed = {}
    spike_indicator_trial = {}

    for epoch in run_epoch_list:
        log_likelihood_place_path = (
            analysis_path
            / "log_likelihood_trajectory"
            / f"log_likelihood_{epoch}_use_half_all_movement_True_position_std_{feature_std}_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_speed_path = (
            analysis_path
            / "log_likelihood_speed"
            / f"log_likelihood_speed_{epoch}_use_half_all_movement_True_speed_std_{feature_std}_discrete_var_switching_speed_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_headdir_path = (
            analysis_path
            / "log_likelihood_headdir"
            / f"log_likelihood_headdir_{epoch}_use_half_all_movement_True_headdir_std_{feature_std}_discrete_var_switching_headdir_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_headdir_velocity_path = (
            analysis_path
            / "log_likelihood_headdir_velocity"
            / f"log_likelihood_headdir_velocity_{epoch}_use_half_all_movement_True_headdir_velocity_std_{feature_std}_discrete_var_switching_headdir_velocity_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_headdir_speed_path = (
            analysis_path
            / "log_likelihood_headdir_speed"
            / f"log_likelihood_headdir_speed_{epoch}_use_half_all_movement_True_headdir_speed_std_{feature_std}_discrete_var_switching_headdir_speed_bin_size_1.0_movement_var_4.0.pkl"
        )

        with open(log_likelihood_place_path, "rb") as f:
            log_likelihood_place[epoch] = pickle.load(f)

        with open(log_likelihood_speed_path, "rb") as f:
            log_likelihood_speed[epoch] = pickle.load(f)

        with open(log_likelihood_headdir_path, "rb") as f:
            log_likelihood_headdir[epoch] = pickle.load(f)

        with open(log_likelihood_headdir_velocity_path, "rb") as f:
            log_likelihood_headdir_velocity[epoch] = pickle.load(f)

        with open(log_likelihood_headdir_speed_path, "rb") as f:
            log_likelihood_headdir_speed[epoch] = pickle.load(f)

        with open(
            analysis_path
            / "spike_indicators_trajectory_trial"
            / f"spike_indicators_trajectory_trial_{epoch}_use_half_all_movement_True_position_std_{feature_std}_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl",
            "rb",
        ) as f:
            spike_indicator_trial[epoch] = pickle.load(f)
    return (
        log_likelihood_place,
        log_likelihood_speed,
        log_likelihood_headdir,
        log_likelihood_headdir_velocity,
        log_likelihood_headdir_speed,
        spike_indicator_trial,
    )


def load_log_likelihoods_cv(feature_std=4.0):

    log_likelihood_place = {}
    log_likelihood_speed = {}
    log_likelihood_headdir = {}
    log_likelihood_headdir_velocity = {}
    log_likelihood_headdir_speed = {}
    spike_indicator_trial = {}

    for epoch in run_epoch_list:
        log_likelihood_place_path = (
            analysis_path
            / "log_likelihood_2d_trajectory_cv"
            / f"log_likelihood_2d_trajectory_{epoch}_n_folds_10_movement_True_position_std_{feature_std}_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_speed_path = (
            analysis_path
            / "log_likelihood_speed_cv"
            / f"log_likelihood_speed_{epoch}_n_folds_10_movement_True_speed_std_{feature_std}_discrete_var_switching_speed_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_headdir_path = (
            analysis_path
            / "log_likelihood_headdir_cv"
            / f"log_likelihood_headdir_{epoch}_n_folds_10_movement_True_headdir_std_{feature_std}_discrete_var_switching_headdir_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_headdir_velocity_path = (
            analysis_path
            / "log_likelihood_headdir_velocity_cv"
            / f"log_likelihood_headdir_velocity_{epoch}_n_folds_10_movement_True_headdir_velocity_std_{feature_std}_discrete_var_switching_headdir_velocity_bin_size_1.0_movement_var_4.0.pkl"
        )
        log_likelihood_headdir_speed_path = (
            analysis_path
            / "log_likelihood_headdir_speed_cv"
            / f"log_likelihood_headdir_speed_{epoch}_n_folds_10_movement_True_headdir_speed_std_{feature_std}_discrete_var_switching_headdir_speed_bin_size_1.0_movement_var_4.0.pkl"
        )

        with open(log_likelihood_place_path, "rb") as f:
            log_likelihood_place[epoch] = pickle.load(f)

        with open(log_likelihood_speed_path, "rb") as f:
            log_likelihood_speed[epoch] = pickle.load(f)

        with open(log_likelihood_headdir_path, "rb") as f:
            log_likelihood_headdir[epoch] = pickle.load(f)

        with open(log_likelihood_headdir_velocity_path, "rb") as f:
            log_likelihood_headdir_velocity[epoch] = pickle.load(f)

        with open(log_likelihood_headdir_speed_path, "rb") as f:
            log_likelihood_headdir_speed[epoch] = pickle.load(f)

        with open(
            analysis_path
            / "spike_indicators_2d_trajectory_cv_trial"
            / f"spike_indicators_2d_trajectory_trial_{epoch}_n_folds_10_movement_True_position_std_2.0_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl",
            "rb",
        ) as f:
            spike_indicator_trial[epoch] = pickle.load(f)
    return (
        log_likelihood_place,
        log_likelihood_speed,
        log_likelihood_headdir,
        log_likelihood_headdir_velocity,
        log_likelihood_headdir_speed,
        spike_indicator_trial,
    )


def plot_log_likelihood_across_models(feature_std=2.0):
    fig_save_path = (
        analysis_path / "figs" / "log_likelihood" / f"feature_std_{feature_std}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)
    print(f"loading log likelihoods with feature_std {feature_std}")

    (
        log_likelihood_place,
        log_likelihood_speed,
        log_likelihood_headdir,
        log_likelihood_headdir_velocity,
        log_likelihood_headdir_speed,
        spike_indicator_trial,
    ) = load_log_likelihoods(feature_std=feature_std)

    for region in regions:
        print(f"in region {region}")
        unit_ids = list(
            log_likelihood_place[run_epoch_list[0]][region][trajectory_types[0]].keys()
        )
        for unit_id in unit_ids:
            fig, ax = plt.subplots(
                figsize=(5 * len(trajectory_types), 5 * len(run_epoch_list)),
                ncols=len(trajectory_types),
                nrows=len(run_epoch_list),
            )
            for epoch_idx, epoch in enumerate(run_epoch_list):
                for trajectory_type_idx, trajectory_type in enumerate(trajectory_types):

                    ll_trial_place = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_place[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    ll_trial_speed = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_speed[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    ll_trial_headdir = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_headdir[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    ll_trial_headdir_velocity = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_headdir_velocity[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    ll_trial_headdir_speed = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_headdir_speed[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )

                    n_spikes_trial = np.array(
                        [
                            np.sum(i)
                            for i in spike_indicator_trial[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    valid = n_spikes_trial > 0
                    if not np.any(valid):
                        # No valid trials for this trajectory; skip plotting
                        continue

                    denom = np.log(2) * n_spikes_trial[valid]
                    place_bits = ll_trial_place[valid] / denom
                    speed_bits = ll_trial_speed[valid] / denom
                    headdir_bits = ll_trial_headdir[valid] / denom
                    headdir_vel_bits = ll_trial_headdir_velocity[valid] / denom
                    headdir_speed_bits = ll_trial_headdir_speed[valid] / denom

                    diff_speed = place_bits - speed_bits
                    diff_headdir = place_bits - headdir_bits
                    diff_headdir_vel = place_bits - headdir_vel_bits
                    diff_headdir_speed = place_bits - headdir_speed_bits

                    diff_data = np.stack(
                        [
                            diff_speed,
                            diff_headdir,
                            diff_headdir_vel,
                            diff_headdir_speed,
                        ],
                        axis=1,
                    )  # shape (n_trials, 4)

                    # data = np.stack(
                    #     [
                    #         ll_trial_place / np.log(2) / n_spikes_trial,
                    #         ll_trial_speed / np.log(2) / n_spikes_trial,
                    #         ll_trial_headdir / np.log(2) / n_spikes_trial,
                    #         ll_trial_headdir_velocity / np.log(2) / n_spikes_trial,
                    #         ll_trial_headdir_speed / np.log(2) / n_spikes_trial,
                    #     ],
                    #     axis=0,
                    # )

                    x = np.arange(1, diff_data.shape[1] + 1)
                    ax[epoch_idx, trajectory_type_idx].boxplot(
                        diff_data,
                        positions=x,
                        labels=diff_labels,
                        showmeans=True,
                        showfliers=False,
                    )
                    ax[epoch_idx, trajectory_type_idx].axhline(
                        0.0, linestyle="--", linewidth=1
                    )

                    ax[epoch_idx, trajectory_type_idx].tick_params(
                        axis="x", rotation=30
                    )

                    # ax[epoch_idx, trajectory_type_idx].boxplot(
                    #     data.T,
                    #     labels=model_labels,
                    #     showmeans=True,
                    #     patch_artist=False,
                    #     showfliers=False,
                    # )
                    if epoch_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_title(trajectory_type)
                    if trajectory_type_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_ylabel(epoch)
                    if epoch_idx != len(run_epoch_list) - 1:
                        ax[epoch_idx, trajectory_type_idx].set_xticks([])

            ax[0, -1].set_ylabel(
                "Place field model advantage\nlog likliehood (bits / spike)"
            )

            fig.suptitle(
                f"{animal_name} {date} {region} {unit_id} model log likelihood comparison"
            )

            fig.savefig(
                fig_save_path / f"{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    return None


def plot_log_likelihood_across_models_cv(feature_std=2.0):
    fig_save_path = (
        analysis_path / "figs" / "log_likelihood_cv" / f"feature_std_{feature_std}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)
    print(f"loading log likelihoods with feature_std {feature_std}")

    (
        log_likelihood_place,
        log_likelihood_speed,
        log_likelihood_headdir,
        log_likelihood_headdir_velocity,
        log_likelihood_headdir_speed,
        spike_indicator_trial,
    ) = load_log_likelihoods_cv(feature_std=feature_std)

    for region in regions:
        print(f"in region {region}")
        unit_ids = list(log_likelihood_place[run_epoch_list[0]][region].keys())
        for unit_id in unit_ids:
            fig, ax = plt.subplots(
                figsize=(5 * len(trajectory_types), 5 * len(run_epoch_list)),
                ncols=len(trajectory_types),
                nrows=len(run_epoch_list),
            )
            for epoch_idx, epoch in enumerate(run_epoch_list):
                for trajectory_type_idx, trajectory_type in enumerate(trajectory_types):

                    ll_trial_place = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_place[epoch][region][unit_id][
                                trajectory_type
                            ]
                        ]
                    )
                    ll_trial_speed = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_speed[epoch][region][unit_id][
                                trajectory_type
                            ]
                        ]
                    )
                    ll_trial_headdir = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_headdir[epoch][region][unit_id][
                                trajectory_type
                            ]
                        ]
                    )
                    ll_trial_headdir_velocity = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_headdir_velocity[epoch][region][
                                unit_id
                            ][trajectory_type]
                        ]
                    )
                    ll_trial_headdir_speed = np.array(
                        [
                            np.sum(i)
                            for i in log_likelihood_headdir_speed[epoch][region][
                                unit_id
                            ][trajectory_type]
                        ]
                    )

                    n_spikes_trial = np.array(
                        [
                            np.sum(i)
                            for i in spike_indicator_trial[epoch][region][unit_id][
                                trajectory_type
                            ]
                        ]
                    )
                    valid = n_spikes_trial > 0
                    if not np.any(valid):
                        # No valid trials for this trajectory; skip plotting
                        continue

                    denom = np.log(2) * n_spikes_trial[valid]
                    place_bits = ll_trial_place[valid] / denom
                    speed_bits = ll_trial_speed[valid] / denom
                    headdir_bits = ll_trial_headdir[valid] / denom
                    headdir_vel_bits = ll_trial_headdir_velocity[valid] / denom
                    headdir_speed_bits = ll_trial_headdir_speed[valid] / denom

                    diff_speed = place_bits - speed_bits
                    diff_headdir = place_bits - headdir_bits
                    diff_headdir_vel = place_bits - headdir_vel_bits
                    diff_headdir_speed = place_bits - headdir_speed_bits

                    diff_data = np.stack(
                        [
                            diff_speed,
                            diff_headdir,
                            diff_headdir_vel,
                            diff_headdir_speed,
                        ],
                        axis=1,
                    )  # shape (n_trials, 4)

                    # data = np.stack(
                    #     [
                    #         ll_trial_place / np.log(2) / n_spikes_trial,
                    #         ll_trial_speed / np.log(2) / n_spikes_trial,
                    #         ll_trial_headdir / np.log(2) / n_spikes_trial,
                    #         ll_trial_headdir_velocity / np.log(2) / n_spikes_trial,
                    #         ll_trial_headdir_speed / np.log(2) / n_spikes_trial,
                    #     ],
                    #     axis=0,
                    # )

                    x = np.arange(1, diff_data.shape[1] + 1)
                    ax[epoch_idx, trajectory_type_idx].boxplot(
                        diff_data,
                        positions=x,
                        labels=diff_labels,
                        showmeans=True,
                        showfliers=False,
                    )
                    ax[epoch_idx, trajectory_type_idx].axhline(
                        0.0, linestyle="--", linewidth=1
                    )

                    ax[epoch_idx, trajectory_type_idx].tick_params(
                        axis="x", rotation=30
                    )

                    # ax[epoch_idx, trajectory_type_idx].boxplot(
                    #     data.T,
                    #     labels=model_labels,
                    #     showmeans=True,
                    #     patch_artist=False,
                    #     showfliers=False,
                    # )
                    if epoch_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_title(trajectory_type)
                    if trajectory_type_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_ylabel(epoch)
                    if epoch_idx != len(run_epoch_list) - 1:
                        ax[epoch_idx, trajectory_type_idx].set_xticks([])

            ax[0, -1].set_ylabel(
                "Place field model advantage\nlog likliehood (bits / spike)"
            )

            fig.suptitle(
                f"{animal_name} {date} {region} {unit_id} model log likelihood comparison"
            )

            fig.savefig(
                fig_save_path / f"{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    return None


def plot_log_likelihood_across_feature_size(model_type="headdir", plotmode="median"):
    assert model_type in [
        "speed",
        "headdir",
        "headdir_speed",
        "headdir_velocity",
    ], ValueError("unexpected model type")
    assert plotmode in ["median", "box"], ValueError("unexpected plotmode")

    fig_save_path = (
        analysis_path
        / "figs"
        / "log_likelihood"
        / "across_feature_size_comparison"
        / f"{model_type}_{plotmode}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    print(
        f"plotting log likelihood advantage of place model over model_type {model_type} plotmode {plotmode}"
    )

    log_likelihood_trajectory = {}
    for epoch in run_epoch_list:
        log_likelihood_trajectory[epoch] = {}
        for feature_size in feature_sizes:
            log_likelihood_trajectory_path = (
                analysis_path
                / "log_likelihood_trajectory"
                / f"log_likelihood_{epoch}_use_half_all_movement_True_position_std_{feature_size}_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl"
            )
            with open(log_likelihood_trajectory_path, "rb") as f:
                log_likelihood_trajectory[epoch][feature_size] = pickle.load(f)

    log_likelihood_model = {}
    for epoch in run_epoch_list:
        log_likelihood_model[epoch] = {}
        for feature_size in feature_sizes:
            if model_type == "headdir":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_headdir"
                    / f"log_likelihood_headdir_{epoch}_use_half_all_movement_True_headdir_std_{feature_size}_discrete_var_switching_headdir_bin_size_1.0_movement_var_4.0.pkl"
                )
            elif model_type == "headdir_speed":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_headdir_speed"
                    / f"log_likelihood_headdir_speed_{epoch}_use_half_all_movement_True_headdir_speed_std_{feature_size}_discrete_var_switching_headdir_speed_bin_size_1.0_movement_var_4.0.pkl"
                )
            elif model_type == "headdir_velocity":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_headdir_velocity"
                    / f"log_likelihood_headdir_velocity_{epoch}_use_half_all_movement_True_headdir_velocity_std_{feature_size}_discrete_var_switching_headdir_velocity_bin_size_1.0_movement_var_4.0.pkl"
                )

            elif model_type == "speed":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_speed"
                    / f"log_likelihood_speed_{epoch}_use_half_all_movement_True_speed_std_{feature_size}_discrete_var_switching_speed_bin_size_1.0_movement_var_4.0.pkl"
                )
            with open(log_likelihood_model_path, "rb") as f:
                log_likelihood_model[epoch][feature_size] = pickle.load(f)

    spike_indicator_trial = {}
    for epoch in run_epoch_list:
        with open(
            analysis_path
            / "spike_indicators_trajectory_trial"
            / f"spike_indicators_trajectory_trial_{epoch}_use_half_all_movement_True_position_std_1.0_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl",
            "rb",
        ) as f:
            spike_indicator_trial[epoch] = pickle.load(f)

    for region in regions:
        unit_ids = list(
            log_likelihood_trajectory[run_epoch_list[0]][feature_sizes[0]][region][
                trajectory_types[0]
            ].keys()
        )
        for unit_id in unit_ids:
            log_likelihood_difference_per_trial = {}
            fig, ax = plt.subplots(
                nrows=len(run_epoch_list),
                ncols=len(trajectory_types),
                figsize=(5 * len(trajectory_types), 5 * len(run_epoch_list)),
            )

            for epoch_idx, epoch in enumerate(run_epoch_list):

                log_likelihood_difference_per_trial[epoch] = {}
                for trajectory_type in trajectory_types:
                    log_likelihood_difference_per_trial[epoch][trajectory_type] = {}
                    for feature_size in feature_sizes:
                        place_log_likelihood = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_trajectory[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        model_log_likelihood = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_model[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        log_likelihood_difference_per_trial[epoch][trajectory_type][
                            feature_size
                        ] = (place_log_likelihood - model_log_likelihood)
                for trajectory_type_idx, trajectory_type in enumerate(trajectory_types):

                    n_spikes_trial = np.array(
                        [
                            np.sum(i)
                            for i in spike_indicator_trial[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    valid = n_spikes_trial > 0
                    if not np.any(valid):
                        # No valid trials for this trajectory; skip plotting
                        continue

                    denom = np.log(2) * n_spikes_trial[valid]

                    # Build per-feature-size normalized arrays (each is length n_valid)
                    labels = list(feature_sizes)
                    norm_by_fs = []
                    medians = []

                    for feature_size in feature_sizes:
                        place_ll = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_trajectory[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        model_ll = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_model[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        diff = place_ll - model_ll  # (n_trials,)
                        norm = diff[valid] / denom  # (n_valid,)
                        norm_by_fs.append(norm)
                        medians.append(np.median(norm))

                    positions = np.asarray(feature_sizes, dtype=float)

                    if plotmode == "median":
                        ax[epoch_idx, trajectory_type_idx].plot(positions, medians)
                    elif plotmode == "box":
                        ax[epoch_idx, trajectory_type_idx].boxplot(
                            norm_by_fs,
                            positions=positions,
                            labels=labels,
                            showfliers=False,
                        )
                    if epoch_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_title(trajectory_type)
                    if trajectory_type_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_ylabel(epoch)
                    if epoch_idx != len(run_epoch_list) - 1:
                        ax[epoch_idx, trajectory_type_idx].set_xticks([])

                    ax[epoch_idx, trajectory_type_idx].axhline(
                        0.0, linestyle="--", linewidth=1
                    )
                    ax[epoch_idx, trajectory_type_idx].set_xticks(positions)
                    ax[epoch_idx, trajectory_type_idx].set_xticklabels(
                        [str(fs) for fs in feature_sizes], rotation=30
                    )
                    ax[epoch_idx, trajectory_type_idx].tick_params(
                        axis="x", rotation=30
                    )

            ax[-1, 0].set_xlabel("Feature size")
            ax[0, -1].set_ylabel(
                f"Place field model advantage over {model_type}\nlog likliehood (bits / spike)"
            )

            fig.suptitle(
                f"{region} {unit_id} log likelihood advantage of place model over {model_type}"
            )

            fig.savefig(
                fig_save_path / f"{region}_{unit_id}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    return None


def plot_log_likelihood_across_feature_size_cv(model_type="headdir", plotmode="median"):
    assert model_type in [
        "speed",
        "headdir",
        "headdir_speed",
        "headdir_velocity",
    ], ValueError("unexpected model type")
    assert plotmode in ["median", "box"], ValueError("unexpected plotmode")

    fig_save_path = (
        analysis_path
        / "figs"
        / "log_likelihood"
        / "across_feature_size_comparison"
        / f"{model_type}_{plotmode}"
    )
    fig_save_path.mkdir(parents=True, exist_ok=True)

    print(
        f"plotting log likelihood advantage of place model over model_type {model_type} plotmode {plotmode}"
    )

    log_likelihood_trajectory = {}
    for epoch in run_epoch_list:
        log_likelihood_trajectory[epoch] = {}
        for feature_size in feature_sizes:
            log_likelihood_trajectory_path = (
                analysis_path
                / "log_likelihood_trajectory"
                / f"log_likelihood_{epoch}_use_half_all_movement_True_position_std_{feature_size}_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl"
            )
            with open(log_likelihood_trajectory_path, "rb") as f:
                log_likelihood_trajectory[epoch][feature_size] = pickle.load(f)

    log_likelihood_model = {}
    for epoch in run_epoch_list:
        log_likelihood_model[epoch] = {}
        for feature_size in feature_sizes:
            if model_type == "headdir":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_headdir"
                    / f"log_likelihood_headdir_{epoch}_use_half_all_movement_True_headdir_std_{feature_size}_discrete_var_switching_headdir_bin_size_1.0_movement_var_4.0.pkl"
                )
            elif model_type == "headdir_speed":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_headdir_speed"
                    / f"log_likelihood_headdir_speed_{epoch}_use_half_all_movement_True_headdir_speed_std_{feature_size}_discrete_var_switching_headdir_speed_bin_size_1.0_movement_var_4.0.pkl"
                )
            elif model_type == "headdir_velocity":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_headdir_velocity"
                    / f"log_likelihood_headdir_velocity_{epoch}_use_half_all_movement_True_headdir_velocity_std_{feature_size}_discrete_var_switching_headdir_velocity_bin_size_1.0_movement_var_4.0.pkl"
                )

            elif model_type == "speed":
                log_likelihood_model_path = (
                    analysis_path
                    / "log_likelihood_speed"
                    / f"log_likelihood_speed_{epoch}_use_half_all_movement_True_speed_std_{feature_size}_discrete_var_switching_speed_bin_size_1.0_movement_var_4.0.pkl"
                )
            with open(log_likelihood_model_path, "rb") as f:
                log_likelihood_model[epoch][feature_size] = pickle.load(f)

    spike_indicator_trial = {}
    for epoch in run_epoch_list:
        with open(
            analysis_path
            / "spike_indicators_trajectory_trial"
            / f"spike_indicators_trajectory_trial_{epoch}_use_half_all_movement_True_position_std_1.0_discrete_var_switching_place_bin_size_1.0_movement_var_4.0.pkl",
            "rb",
        ) as f:
            spike_indicator_trial[epoch] = pickle.load(f)

    for region in regions:
        unit_ids = list(
            log_likelihood_trajectory[run_epoch_list[0]][feature_sizes[0]][region][
                trajectory_types[0]
            ].keys()
        )
        for unit_id in unit_ids:
            log_likelihood_difference_per_trial = {}
            fig, ax = plt.subplots(
                nrows=len(run_epoch_list),
                ncols=len(trajectory_types),
                figsize=(5 * len(trajectory_types), 5 * len(run_epoch_list)),
            )

            for epoch_idx, epoch in enumerate(run_epoch_list):

                log_likelihood_difference_per_trial[epoch] = {}
                for trajectory_type in trajectory_types:
                    log_likelihood_difference_per_trial[epoch][trajectory_type] = {}
                    for feature_size in feature_sizes:
                        place_log_likelihood = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_trajectory[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        model_log_likelihood = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_model[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        log_likelihood_difference_per_trial[epoch][trajectory_type][
                            feature_size
                        ] = (place_log_likelihood - model_log_likelihood)
                for trajectory_type_idx, trajectory_type in enumerate(trajectory_types):

                    n_spikes_trial = np.array(
                        [
                            np.sum(i)
                            for i in spike_indicator_trial[epoch][region][
                                trajectory_type
                            ][unit_id]
                        ]
                    )
                    valid = n_spikes_trial > 0
                    if not np.any(valid):
                        # No valid trials for this trajectory; skip plotting
                        continue

                    denom = np.log(2) * n_spikes_trial[valid]

                    # Build per-feature-size normalized arrays (each is length n_valid)
                    labels = list(feature_sizes)
                    norm_by_fs = []
                    medians = []

                    for feature_size in feature_sizes:
                        place_ll = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_trajectory[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        model_ll = np.asarray(
                            [
                                np.sum(i)
                                for i in log_likelihood_model[epoch][feature_size][
                                    region
                                ][trajectory_type][unit_id]
                            ]
                        )
                        diff = place_ll - model_ll  # (n_trials,)
                        norm = diff[valid] / denom  # (n_valid,)
                        norm_by_fs.append(norm)
                        medians.append(np.median(norm))

                    positions = np.asarray(feature_sizes, dtype=float)

                    if plotmode == "median":
                        ax[epoch_idx, trajectory_type_idx].plot(positions, medians)
                    elif plotmode == "box":
                        ax[epoch_idx, trajectory_type_idx].boxplot(
                            norm_by_fs,
                            positions=positions,
                            labels=labels,
                            showfliers=False,
                        )
                    if epoch_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_title(trajectory_type)
                    if trajectory_type_idx == 0:
                        ax[epoch_idx, trajectory_type_idx].set_ylabel(epoch)
                    if epoch_idx != len(run_epoch_list) - 1:
                        ax[epoch_idx, trajectory_type_idx].set_xticks([])

                    ax[epoch_idx, trajectory_type_idx].axhline(
                        0.0, linestyle="--", linewidth=1
                    )
                    ax[epoch_idx, trajectory_type_idx].set_xticks(positions)
                    ax[epoch_idx, trajectory_type_idx].set_xticklabels(
                        [str(fs) for fs in feature_sizes], rotation=30
                    )
                    ax[epoch_idx, trajectory_type_idx].tick_params(
                        axis="x", rotation=30
                    )

            ax[-1, 0].set_xlabel("Feature size")
            ax[0, -1].set_ylabel(
                f"Place field model advantage over {model_type}\nlog likliehood (bits / spike)"
            )

            fig.suptitle(
                f"{region} {unit_id} log likelihood advantage of place model over {model_type}"
            )

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
        "--feature_std",
        type=float,
        help="std of gaussian kernel for estimating place fields, in cm",
    )

    return parser.parse_args()


def main():
    # args = parse_arguments()
    for feature_std in [2.0, 1.0, 0.5, 4.0, 8.0]:
        plot_log_likelihood_across_models_cv(
            feature_std=feature_std,
        )


if __name__ == "__main__":
    main()
