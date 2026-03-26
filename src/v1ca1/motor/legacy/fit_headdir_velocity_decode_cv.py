import spikeinterface.full as si
import numpy as np
import kyutils
import pickle

import replay_trajectory_classification as rtc
import scipy
import argparse

import position_tools as pt
from pathlib import Path

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

with open(analysis_path / "position.pkl", "rb") as f:
    position_dict = pickle.load(f)
with open(analysis_path / "body_position.pkl", "rb") as f:
    body_position_dict = pickle.load(f)

with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(analysis_path / "trajectory_times.pkl", "rb") as f:
    trajectory_times = pickle.load(f)


sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

position_offset = 10

temporal_bin_size_s = 2e-3
sampling_rate = int(1 / temporal_bin_size_s)
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


def get_start_stop_times_n_fold_cv(epoch, n_folds, ptp=True):

    trajectory_type_list = np.concatenate(
        [
            [trajectory_type] * len(trajectory_times[epoch][trajectory_type])
            for trajectory_type in trajectory_types
        ]
    )
    trajectory_type_list = trajectory_type_list[
        np.argsort(
            np.concatenate(
                [
                    trajectory_times[epoch][trajectory_type]
                    for trajectory_type in trajectory_types
                ]
            )[:, 0]
        )
    ]
    first_trajectory_type = trajectory_type_list[0]
    first_trajectory_type_start_times = np.asarray(
        trajectory_times[epoch][first_trajectory_type]
    )[:, 0]

    folds = np.array_split(first_trajectory_type_start_times, n_folds)
    folds_start_stop_times = np.asarray(
        [[folds[i][0], folds[i + 1][0]] for i in range(len(folds) - 1)]
    )
    folds_start_stop_times = np.vstack(
        (
            folds_start_stop_times,
            [folds_start_stop_times[-1][1], timestamps_ephys[epoch][-1]],
        )
    )
    folds_start_stop_times[0] = [
        timestamps_position[epoch][position_offset:][0],
        folds_start_stop_times[0][1],
    ]
    folds_start_stop_times = np.asarray(folds_start_stop_times)
    if not ptp:
        folds_start_stop_times = folds_start_stop_times - timestamps_ephys[epoch][0]
    return folds_start_stop_times


# fit
def fit_headdir_velocity_decode_cv(
    epoch_idx,
    n_folds=10,
    movement="True",
    headdir_velocity_std=1.0,
    discrete_var="switching",
    headdir_velocity_bin_size=1.0,
    movement_var=4.0,
    overwrite="False",
):
    if movement == "False":
        movement = False
    elif movement == "True":
        movement = True
    else:
        raise ValueError("`movement` must be either True or False")

    assert discrete_var in ["switching", "random_walk", "uniform"], ValueError(
        "`discrete_var` must be either 'switching', 'random_walk', or 'uniform'"
    )

    # define epoch name
    epoch = run_epoch_list[epoch_idx]
    n_folds = int(n_folds)

    print(
        f"Fitting head direction velocity decoder "
        f"{animal_name} {date} "
        f"epoch {epoch}, "
        f"n_folds {n_folds}, "
        f"headdir_velocity_bin_size {headdir_velocity_bin_size}, "
        f"movement_var {movement_var}, "
        f"headdir_velocity_std {headdir_velocity_std}, "
        f"movement {movement}, "
        f"discrete_var {discrete_var}, "
        f"overwrite {overwrite}."
    )

    # define reference time offset and subtract it from the timestamps
    t_position = timestamps_position[epoch][position_offset:]

    # define time vector for decoding (temporal resolution: 2 ms)
    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    # define position
    head_position_2d = position_dict[epoch][position_offset:]
    body_position_2d = body_position_dict[epoch][position_offset:]
    position_sampling_rate = len(head_position_2d) / (t_position[-1] - t_position[0])

    head_direction_rad = pt.get_angle(body_position_2d, head_position_2d)
    head_direction_rad_unwrapped = np.unwrap(head_direction_rad)
    head_direction_deg_unwrapped = np.rad2deg(head_direction_rad_unwrapped)

    head_direction_deg_velocity = np.diff(
        head_direction_deg_unwrapped, prepend=head_direction_deg_unwrapped[0]
    )
    head_direction_deg_velocity = pt.core.gaussian_smooth(
        head_direction_deg_velocity,
        sigma=0.1,
        sampling_frequency=position_sampling_rate,
    )
    f_head_direction_velocity = scipy.interpolate.interp1d(
        t_position,
        head_direction_deg_velocity,
        axis=0,
        bounds_error=False,
        kind="linear",
    )
    head_direction_velocity_interp = f_head_direction_velocity(time)

    # compute speed (for figuring out movement vs. stationary times)
    speed = pt.get_speed(
        head_position_2d,
        time=t_position,
        sampling_frequency=position_sampling_rate,
        sigma=0.1,
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )
    speed_interp = f_speed(time)
    exceeds_speed = speed_interp > speed_threshold

    cv_start_stop_times = get_start_stop_times_n_fold_cv(
        epoch=epoch, n_folds=n_folds, ptp=True
    )

    # define environment
    environment = rtc.Environment(
        place_bin_size=headdir_velocity_bin_size,
    )

    # define movement model, observation type, and encoding labels
    random_walk = rtc.RandomWalk(movement_var=movement_var)
    uniform = rtc.Uniform()
    if discrete_var == "switching":
        continuous_transition_types = [[random_walk, uniform], [uniform, uniform]]
    elif discrete_var == "random_walk":
        continuous_transition_types = [
            [
                random_walk,
                random_walk,
            ],
            [
                random_walk,
                random_walk,
            ],
        ]
    elif discrete_var == "uniform":
        continuous_transition_types = [
            [
                uniform,
                uniform,
            ],
            [
                uniform,
                uniform,
            ],
        ]
    observation_models = None
    encoding_labels = None

    classifier_save_dir = analysis_path / "classifier_headdir_velocity_cv"
    classifier_save_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        spike_indicator = get_spike_indicator(
            sorting[region], timestamps_ephys_all_ptp, time
        )
        print(f"in region {region}")
        for fold_idx in range(n_folds):
            fold_start_time, fold_stop_time = cv_start_stop_times[fold_idx]
            inside_fold = (time > fold_start_time) & (time <= fold_stop_time)
            outside_fold = ~inside_fold

            print(
                f"Working on fold {fold_idx+1} out of {n_folds}, using {np.sum(outside_fold)} out of {len(time)} time bins to train"
            )

            classifier_save_path = classifier_save_dir / (
                f"classifier_{region}_{date}_{epoch}_headdir_velocity"
                f"_{fold_idx}_of_{n_folds}_folds"
                f"_movement_{movement}"
                f"_fit_headdir_velocity_std_{headdir_velocity_std}"
                f"_discrete_var_{discrete_var}"
                f"_headdir_velocity_bin_size_{headdir_velocity_bin_size}"
                f"_movement_var_{movement_var}"
                ".pkl"
            )

            if classifier_save_path.exists() and (not overwrite):
                print("Aleady exists, skipping...")
                return

            if movement:
                is_training = exceeds_speed & outside_fold
            else:
                is_training = outside_fold

            # define classifiers
            classifier = rtc.SortedSpikesClassifier(
                environments=environment,
                continuous_transition_types=continuous_transition_types,
                observation_models=observation_models,
                sorted_spikes_algorithm="spiking_likelihood_kde_gpu",
                sorted_spikes_algorithm_params={
                    "position_std": headdir_velocity_std,
                    "use_diffusion": False,
                    "block_size": int(2**11),
                },
            )

            # fit classifiers
            classifier.fit(
                head_direction_velocity_interp,
                spike_indicator,
                is_training=is_training,
                encoding_group_labels=encoding_labels,
            )

            classifier.save_model(classifier_save_path)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit head direction decoder on CA1 and V1 for the four trajectories"
    )
    parser.add_argument("--epoch_idx", type=int, help="Epoch to fit")
    parser.add_argument(
        "--n_folds",
        type=int,
        help="number of folds for cv",
    )
    parser.add_argument(
        "--movement",
        type=str,
        help="Whether to encode based on movement vs. all times; bool",
    )
    parser.add_argument(
        "--headdir_velocity_std",
        type=float,
        help="Width of gaussian for estimating speed fields; in degrees",
    )
    parser.add_argument(
        "--discrete_var",
        type=str,
        help="Type of movement model for the discrete latent variable; one of ['switching', 'random_walk', 'uniform']",
    )
    parser.add_argument(
        "--headdir_velocity_bin_size",
        type=float,
        help="size of speed bin, in degrees",
    )
    parser.add_argument(
        "--movement_var",
        type=float,
        help="variance of random walk for movement dynamics, in degrees^2",
    )

    parser.add_argument(
        "--overwrite",
        type=str,
        help="Overwrite classifier if exists; one of ['True', 'False']",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    fit_headdir_velocity_decode_cv(
        epoch_idx=args.epoch_idx,
        n_folds=args.n_folds,
        movement=args.movement,
        headdir_velocity_std=args.headdir_velocity_std,
        discrete_var=args.discrete_var,
        headdir_velocity_bin_size=args.headdir_velocity_bin_size,
        movement_var=args.movement_var,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
