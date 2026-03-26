import replay_trajectory_classification as rtc
import spikeinterface.full as si

import numpy as np
import kyutils
import pickle

import scipy
import track_linearization as tl
import position_tools as pt

from pathlib import Path

from track_linearization import make_track_graph

import argparse

animal_name = "L14"
date = "20240611"
data_path = Path("/stelmo/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / date

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

classifier_save_dir = analysis_path / "classifier_headdir_velocity"


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


def interpolate_nonmonotonic_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN values in y given non-monotonic x values.
    Sorts by x for interpolation, then restores original order.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates, shape (n,).
    y : np.ndarray
        1D array of y-values, shape (n,), may include NaN.

    Returns
    -------
    np.ndarray
        Interpolated y array, shape (n,).
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D.")

    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    valid = ~np.isnan(y)

    if not np.any(valid):
        return y.copy()

    # Sort by x into increasing order
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    valid_sorted = valid[order]

    # Interpolate along sorted axis values
    y_interp_sorted = np.interp(
        x_sorted, x_sorted[valid_sorted], y_sorted[valid_sorted]
    )

    # Return to original ordering
    y_interp = np.empty_like(y)
    y_interp[order] = y_interp_sorted

    return y_interp


def unwrap_degrees(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap angle sequence in degrees, originally in range [-180, 180].

    Parameters
    ----------
    angles : np.ndarray
        Array of angles (n_time,) in degrees.

    Returns
    -------
    np.ndarray
        Unwrapped angle sequence (n_time,) in degrees.
    """
    angles = np.asarray(angles)
    diffs = np.diff(angles)
    diffs = np.where(diffs > 180, diffs - 360, diffs)
    diffs = np.where(diffs < -180, diffs + 360, diffs)
    return np.concatenate(([angles[0]], angles[0] + np.cumsum(diffs)))


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


def compute_rate_function_headdir_velocity(
    epoch_idx: int,
    use_half: str,
    movement: str = "True",
    headdir_velocity_std: float = 4.0,
    discrete_var: str = "switching",
    headdir_velocity_bin_size: float = 1.0,
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

    rate_function_save_dir = analysis_path / "rate_function_headdir_velocity"
    rate_function_save_dir.mkdir(parents=True, exist_ok=True)

    log_likelihood_save_dir = analysis_path / "log_likelihood_headdir_velocity"
    log_likelihood_save_dir.mkdir(parents=True, exist_ok=True)

    rate_function_save_path = rate_function_save_dir / (
        "rate_function_headdir_velocity"
        f"_{epoch}"
        f"_use_half_{use_half}"
        f"_movement_{movement}"
        f"_headdir_velocity_std_{headdir_velocity_std}"
        f"_discrete_var_{discrete_var}"
        f"_headdir_velocity_bin_size_{headdir_velocity_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    if rate_function_save_path.exists() and (overwrite == False):
        print("Aleady exists, skipping...")
        return

    log_likelihood_save_path = log_likelihood_save_dir / (
        "log_likelihood_headdir_velocity"
        f"_{epoch}"
        f"_use_half_{use_half}"
        f"_movement_{movement}"
        f"_headdir_velocity_std_{headdir_velocity_std}"
        f"_discrete_var_{discrete_var}"
        f"_headdir_velocity_bin_size_{headdir_velocity_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    if log_likelihood_save_path.exists() and (overwrite == False):
        print("Log likelihood aleady exists, skipping...")
        return

    rate_function = {}
    log_likelihood = {}

    print(
        "Computing rate function by trajectory from head direction velocity fields "
        f"{animal_name} {date} "
        f"epoch {epoch}, "
        f"use_half {use_half}, "
        f"movement {movement}, "
        f"headdir_velocity_std {headdir_velocity_std}, "
        f"discrete_var {discrete_var}, "
        f"headdir_velocity_bin_size {headdir_velocity_bin_size}, "
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

        spike_indicator = get_spike_indicator(
            sorting[region], timestamps_ephys_all_ptp, time
        )

        for trajectory_type in trajectory_types:

            rate_function[region][trajectory_type] = {}
            log_likelihood[region][trajectory_type] = {}

            classifier_save_path = classifier_save_dir / (
                f"classifier_{region}_{date}_{epoch}_headdir_velocity"
                f"_use_half_{use_half}"
                f"_movement_{movement}"
                f"_fit_headdir_velocity_std_{headdir_velocity_std}"
                f"_discrete_var_{discrete_var}"
                f"_headdir_velocity_bin_size_{headdir_velocity_bin_size}"
                f"_movement_var_{movement_var}"
                ".pkl"
            )
            if not classifier_save_path.exists():
                print("Classifier doesn't exist, skipping...")
                return

            classifier = rtc.SortedSpikesClassifier.load_model(
                filename=classifier_save_path
            )

            unit_indices = classifier.place_fields_[("", 0)].neuron.to_numpy()
            headdir_bin_centers = classifier.place_fields_[("", 0)][
                "position"
            ].to_numpy()

            for unit_index in unit_indices:
                headdir_velocity_field = classifier.place_fields_[("", 0)][
                    :, unit_index
                ].to_numpy()

                f_rate = scipy.interpolate.interp1d(
                    headdir_bin_centers,
                    headdir_velocity_field,
                    axis=0,
                    bounds_error=False,
                    kind="linear",
                )

                rate_trajectory_trial_list = []
                log_likelihood_trial_list = []
                for traj_start_time, traj_stop_time in trajectory_times[epoch][
                    trajectory_type
                ]:
                    if movement:
                        inds_to_keep = (
                            (time > traj_start_time)
                            & (time <= traj_stop_time)
                            & (speed_interp > speed_threshold)
                        )
                    else:
                        inds_to_keep = (time > traj_start_time) & (
                            time <= traj_stop_time
                        )
                    if inds_to_keep.sum() == 0:
                        continue

                    r = f_rate(head_direction_velocity_interp[inds_to_keep])
                    r = interpolate_nonmonotonic_x(
                        head_direction_velocity_interp[inds_to_keep], r
                    )
                    rate_trajectory_trial_list.append(r)

                    bins_with_spikes = spike_indicator[inds_to_keep, unit_index].astype(
                        bool
                    )
                    prob_spike = r * np.exp(-r)
                    prob_no_spike = np.exp(-r)
                    likelihood = np.zeros(bins_with_spikes.shape)

                    likelihood[bins_with_spikes] = prob_spike[bins_with_spikes]
                    likelihood[~bins_with_spikes] = prob_no_spike[~bins_with_spikes]
                    log_likelihood_trial_list.append(np.log(likelihood))

                rate_function[region][trajectory_type][
                    unit_index
                ] = rate_trajectory_trial_list
                log_likelihood[region][trajectory_type][
                    unit_index
                ] = log_likelihood_trial_list

    with open(rate_function_save_path, "wb") as f:
        pickle.dump(rate_function, f)
    with open(log_likelihood_save_path, "wb") as f:
        pickle.dump(log_likelihood, f)

    return rate_function, log_likelihood


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute time varying firing rate functions of neurons in a region from 2D place fields and actual position"
    )
    parser.add_argument("--epoch_idx", type=int, help="Epoch index")
    parser.add_argument(
        "--use_half",
        type=str,
        help="Time range to fit; one of ['first', 'second', 'all']",
    )
    parser.add_argument(
        "--movement",
        type=str,
        help="Whether to use place fields learned during movement vs. all times; `True` or `False`",
    )
    parser.add_argument(
        "--headdir_velocity_std",
        type=float,
        help="Width of Gaussian used to estimate the place fields during encoding, in cm; float",
    )
    parser.add_argument(
        "--discrete_var",
        type=str,
        help="Type of model for the discrete latent variable; one of ['switching', 'random_walk', 'uniform']",
    )
    parser.add_argument(
        "--headdir_velocity_bin_size",
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
    compute_rate_function_headdir_velocity(
        epoch_idx=args.epoch_idx,
        use_half=args.use_half,
        movement=args.movement,
        headdir_velocity_std=args.headdir_velocity_std,
        discrete_var=args.discrete_var,
        headdir_velocity_bin_size=args.headdir_velocity_bin_size,
        movement_var=args.movement_var,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
