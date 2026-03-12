import replay_trajectory_classification as rtc
import spikeinterface.full as si

import numpy as np
import kyutils
import pickle

import scipy
import position_tools as pt

from pathlib import Path


import argparse

animal_name = "L14"
date = "20240611"
analysis_path = Path("/stelmo/kyu/analysis") / animal_name / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

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

trajectory_types = [
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
]
regions = ["v1", "ca1"]
sorting = {}
for region in regions:
    sorting[region] = si.load(analysis_path / f"sorting_{region}")

speed_threshold = 4  # cm/s

temporal_bin_size_s = 2e-3
sampling_rate = int(1 / temporal_bin_size_s)

position_offset = 10


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


def load_classifiers(
    epoch_idx: int,
    n_folds: int,
    movement: str = "True",
    headdir_speed_std: float = 4.0,
    discrete_var: str = "switching",
    headdir_speed_bin_size: float = 1.0,
    movement_var: float = 4.0,
):
    epoch = run_epoch_list[epoch_idx]

    classifier_save_dir = analysis_path / "classifier_headdir_speed_cv"

    classifier = {}
    for region in regions:
        classifier[region] = {}
        for fold_idx in range(n_folds):
            classifier_save_path = classifier_save_dir / (
                f"classifier_{region}_{date}_{epoch}"
                f"_headdir_speed_{fold_idx}_of_{n_folds}_folds"
                f"_movement_{movement}"
                f"_fit_headdir_speed_std_{headdir_speed_std}"
                f"_discrete_var_{discrete_var}"
                f"_headdir_speed_bin_size_{headdir_speed_bin_size}"
                f"_movement_var_{movement_var}"
                ".pkl"
            )
            classifier[region][fold_idx] = rtc.SortedSpikesClassifier.load_model(
                filename=classifier_save_path
            )
    return classifier


def compute_rate_function_headdir_speed_cv(
    epoch_idx: int,
    n_folds: int,
    movement: str = "True",
    headdir_speed_std: float = 4.0,
    discrete_var: str = "switching",
    headdir_speed_bin_size: float = 1.0,
    movement_var: float = 4.0,
    overwrite: str = "True",
):

    epoch = run_epoch_list[epoch_idx]

    if movement == "False":
        movement = False
    elif movement == "True":
        movement = True
    else:
        raise ValueError("`movement` must be either True or False")

    assert discrete_var in ["switching", "random_walk", "uniform"], ValueError(
        "`discrete_var` must be either 'switching', 'random_walk', or 'uniform'"
    )

    if overwrite == "False":
        overwrite = False
    elif overwrite == "True":
        overwrite = True
    else:
        raise ValueError("`overwrite` must be either True or False")

    rate_function_save_dir = analysis_path / "rate_function_headdir_speed_cv"
    rate_function_save_dir.mkdir(parents=True, exist_ok=True)

    log_likelihood_save_dir = analysis_path / "log_likelihood_headdir_speed_cv"
    log_likelihood_save_dir.mkdir(parents=True, exist_ok=True)

    classifier_dict = load_classifiers(
        epoch_idx=epoch_idx,
        n_folds=n_folds,
        movement=movement,
        headdir_speed_std=headdir_speed_std,
        discrete_var=discrete_var,
        headdir_speed_bin_size=headdir_speed_bin_size,
        movement_var=movement_var,
    )

    rate_function_save_path = rate_function_save_dir / (
        "rate_function_headdir_speed"
        f"_{epoch}"
        f"_n_folds_{n_folds}"
        f"_movement_{movement}"
        f"_headdir_speed_std_{headdir_speed_std}"
        f"_discrete_var_{discrete_var}"
        f"_headdir_speed_bin_size_{headdir_speed_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    if rate_function_save_path.exists() and (overwrite == False):
        print("Aleady exists, skipping...")
        return

    log_likelihood_save_path = log_likelihood_save_dir / (
        "log_likelihood_headdir_speed"
        f"_{epoch}"
        f"_n_folds_{n_folds}"
        f"_movement_{movement}"
        f"_headdir_speed_std_{headdir_speed_std}"
        f"_discrete_var_{discrete_var}"
        f"_headdir_speed_bin_size_{headdir_speed_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    if log_likelihood_save_path.exists() and (overwrite == False):
        print("Log likelihood aleady exists, skipping...")
        return

    rate_function = {}
    log_likelihood = {}

    print(
        "Computing rate function by trajectory from head direction speed fields "
        f"{animal_name} {date} "
        f"epoch {epoch}, "
        f"n_folds {n_folds}, "
        f"movement {movement}, "
        f"headdir_speed_std {headdir_speed_std}, "
        f"discrete_var {discrete_var}, "
        f"headdir_speed_bin_size {headdir_speed_bin_size}, "
        f"movement_var {movement_var}, "
        f"overwrite {overwrite}."
    )

    for region in regions:
        rate_function[region] = {}
        log_likelihood[region] = {}

        print(f"in region {region}")

        t_position = timestamps_position[epoch][position_offset:]
        start_time = t_position[0]
        end_time = t_position[-1]
        n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1
        time = np.linspace(start_time, end_time, n_samples)

        head_position_2d = position_dict[epoch][position_offset:]
        body_position_2d = body_position_dict[epoch][position_offset:]

        position_sampling_rate = len(head_position_2d) / (
            t_position[-1] - t_position[0]
        )

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
        head_direction_speed_interp = np.abs(head_direction_velocity_interp)

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
            epoch, n_folds=n_folds, ptp=True
        )

        spike_indicator = get_spike_indicator(
            sorting[region], timestamps_ephys_all_ptp, time
        )

        for unit_id in sorting[region].get_unit_ids():
            rate_function[region][unit_id] = {}
            log_likelihood[region][unit_id] = {}

            for trajectory_type in trajectory_types:
                rate_function[region][unit_id][trajectory_type] = []
                log_likelihood[region][unit_id][trajectory_type] = []

                for fold_idx in range(n_folds):

                    fold_start_time, fold_stop_time = cv_start_stop_times[fold_idx]
                    inside_fold = (time > fold_start_time) & (time <= fold_stop_time)

                    classifier = classifier_dict[region][fold_idx]

                    headdir_bin_centers = classifier.place_fields_[("", 0)][
                        "position"
                    ].to_numpy()

                    headdir_speed_field = classifier.place_fields_[("", 0)][
                        :, unit_id
                    ].to_numpy()

                    f_rate = scipy.interpolate.interp1d(
                        headdir_bin_centers,
                        headdir_speed_field,
                        axis=0,
                        bounds_error=False,
                        kind="linear",
                    )

                    for traj_start_time, traj_stop_time in trajectory_times[epoch][
                        trajectory_type
                    ]:
                        within_trajectory = (time > traj_start_time) & (
                            time <= traj_stop_time
                        )

                        if movement:
                            inds_to_keep = (
                                exceeds_speed & within_trajectory & inside_fold
                            )
                        else:
                            inds_to_keep = within_trajectory & inside_fold
                        if inds_to_keep.sum() == 0:
                            continue

                        r = f_rate(head_direction_speed_interp[inds_to_keep])
                        nans = np.isnan(r)
                        if np.any(nans):
                            good = ~nans
                            if good.sum() == 0:
                                continue
                            r[nans] = np.interp(
                                np.flatnonzero(nans), np.flatnonzero(good), r[good]
                            )

                        rate_function[region][unit_id][trajectory_type].append(r)

                        bins_with_spikes = spike_indicator[
                            inds_to_keep, unit_id
                        ].astype(bool)

                        prob_spike = r * np.exp(-r)
                        prob_no_spike = np.exp(-r)
                        likelihood = np.zeros(bins_with_spikes.shape)

                        likelihood[bins_with_spikes] = prob_spike[bins_with_spikes]
                        likelihood[~bins_with_spikes] = prob_no_spike[~bins_with_spikes]

                        eps = 1e-12  # numerical floor to avoid log(0)
                        log_likelihood_chunk = np.log(np.maximum(likelihood, eps))

                        log_likelihood[region][unit_id][trajectory_type].append(
                            log_likelihood_chunk
                        )

    with open(rate_function_save_path, "wb") as f:
        pickle.dump(rate_function, f)
    with open(log_likelihood_save_path, "wb") as f:
        pickle.dump(log_likelihood, f)

    return rate_function, log_likelihood


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute time varying firing rate functions of neurons in a region from head direction speed fields and actual position"
    )
    parser.add_argument("--epoch_idx", type=int, help="Epoch index")
    parser.add_argument(
        "--n_folds",
        type=int,
        help="number of cv folds",
    )
    parser.add_argument(
        "--movement",
        type=str,
        help="Whether to use place fields learned during movement vs. all times; `True` or `False`",
    )
    parser.add_argument(
        "--headdir_speed_std",
        type=float,
        help="Width of Gaussian used to estimate the place fields during encoding, in cm; float",
    )
    parser.add_argument(
        "--discrete_var",
        type=str,
        help="Type of model for the discrete latent variable; one of ['switching', 'random_walk', 'uniform']",
    )
    parser.add_argument(
        "--headdir_speed_bin_size",
        type=float,
        help="Bin size for space, in cm",
    )
    parser.add_argument(
        "--movement_var",
        type=float,
        help="Variance of random walk prior, in cm^2",
    )
    parser.add_argument(
        "--overwrite",
        type=str,
        help="Overwrite classifier if exists; one of ['True', 'False']",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    compute_rate_function_headdir_speed_cv(
        epoch_idx=args.epoch_idx,
        n_folds=args.n_folds,
        movement=args.movement,
        headdir_speed_std=args.headdir_speed_std,
        discrete_var=args.discrete_var,
        headdir_speed_bin_size=args.headdir_speed_bin_size,
        movement_var=args.movement_var,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
