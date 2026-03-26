import replay_trajectory_classification as rtc
import spikeinterface.full as si

import numpy as np
import kyutils
import pickle

import scipy
import track_linearization as tl
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
    sorting[region] = si.load(
        Path("/nimbus/kyu")
        / animal_name
        / "singleday_sort"
        / date
        / f"sorting_{region}"
    )

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


def get_track_graph():
    # track graphs
    long_segment_length = 71
    short_segment_length = 32
    node_positions_left = np.array(
        [
            (55, 81),  # center well
            (55 - short_segment_length, 81),  # left well
            (55, 81 - long_segment_length),  # center junction
            (55 - short_segment_length, 81 - long_segment_length),  # left junction
        ]
    )
    node_positions_right = np.array(
        [
            (55, 81),  # center well
            # (55-short_segment_length, 81),  # left well
            (55 + short_segment_length, 81),  # right well
            (55, 81 - long_segment_length),  # center junction
            # (55-short_segment_length, 81-long_segment_length),  # left junction
            (55 + short_segment_length, 81 - long_segment_length),  # right junction
        ]
    )

    edges = np.array(
        [
            (0, 2),
            (2, 3),
            (3, 1),
        ]
    )

    linear_edge_order = [
        (0, 2),
        (2, 3),
        (3, 1),
    ]
    linear_edge_spacing = 0
    track_graph_left = tl.make_track_graph(node_positions_left, edges)
    track_graph_right = tl.make_track_graph(node_positions_right, edges)

    return track_graph_left, track_graph_right, linear_edge_order, linear_edge_spacing


def get_track_graph_new():

    dx = 9.5
    dy = 9

    long_segment_length = 81 - 17 - 2
    short_segment_length = 13.5

    node_positions_left = np.array(
        [
            (55.5, 81),  # center well
            (55.5, 81 - long_segment_length),
            (55.5 - dx, 81 - long_segment_length - dy),
            (55.5 - dx - short_segment_length, 81 - long_segment_length - dy),
            (55.5 - 2 * dx - short_segment_length, 81 - long_segment_length),
            (55.5 - 2 * dx - short_segment_length, 81),
        ]
    )

    node_positions_right = np.array(
        [
            (55.5, 81),  # center well
            (55.5, 81 - long_segment_length),
            (55.5 + dx, 81 - long_segment_length - dy),
            (55.5 + dx + short_segment_length, 81 - long_segment_length - dy),
            (55.5 + 2 * dx + short_segment_length, 81 - long_segment_length),
            (55.5 + 2 * dx + short_segment_length, 81),
        ]
    )

    edges = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    linear_edge_order = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    linear_edge_spacing = 0
    track_graph_left = tl.make_track_graph(node_positions_left, edges)
    track_graph_right = tl.make_track_graph(node_positions_right, edges)

    return track_graph_left, track_graph_right, linear_edge_order, linear_edge_spacing


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
    if len(first_trajectory_type_start_times) < n_folds:
        raise ValueError("n_folds exceeds number of trajectories available for CV.")
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
    position_std: float = 4.0,
    discrete_var: str = "switching",
    place_bin_size: float = 1.0,
    movement_var: float = 4.0,
):
    epoch = run_epoch_list[epoch_idx]

    classifier_save_dir = analysis_path / "classifier_2d_trajectory_cv"

    classifier = {}
    for region in regions:
        classifier[region] = {}
        for trajectory_type in trajectory_types:
            classifier[region][trajectory_type] = {}
            for fold_idx in range(n_folds):
                classifier_save_path = classifier_save_dir / (
                    f"classifier_{region}_{date}_{epoch}"
                    f"_2d_{trajectory_type}"
                    f"_{fold_idx}_of_{n_folds}_folds"
                    f"_movement_{movement}"
                    f"_fit_position_std_{position_std}"
                    f"_discrete_var_{discrete_var}"
                    f"_place_bin_size_{place_bin_size}"
                    f"_movement_var_{movement_var}"
                    ".pkl"
                )
                classifier[region][trajectory_type][fold_idx] = (
                    rtc.SortedSpikesClassifier.load_model(filename=classifier_save_path)
                )
    return classifier


def compute_rate_function_2d_trajectory_cv(
    epoch_idx: int,
    n_folds: int,
    movement: str = "True",
    position_std: float = 4.0,
    discrete_var: str = "switching",
    place_bin_size: float = 1.0,
    movement_var: float = 4.0,
    overwrite: str = "True",
):

    epoch = run_epoch_list[epoch_idx]
    n_folds = int(n_folds)

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

    rate_function_save_dir = analysis_path / "rate_function_2d_trajectory_cv"
    rate_function_save_dir.mkdir(parents=True, exist_ok=True)

    log_likelihood_save_dir = analysis_path / "log_likelihood_2d_trajectory_cv"
    log_likelihood_save_dir.mkdir(parents=True, exist_ok=True)

    spike_indicators_save_dir = (
        analysis_path / "spike_indicators_2d_trajectory_cv_trial"
    )
    spike_indicators_save_dir.mkdir(parents=True, exist_ok=True)

    classifier_dict = load_classifiers(
        epoch_idx=epoch_idx,
        n_folds=n_folds,
        movement=movement,
        position_std=position_std,
        discrete_var=discrete_var,
        place_bin_size=place_bin_size,
        movement_var=movement_var,
    )

    track_graph_left, track_graph_right, linear_edge_order, linear_edge_spacing = (
        get_track_graph_new()
    )

    rate_function_save_path = rate_function_save_dir / (
        "rate_function_2d_trajectory"
        f"_{epoch}"
        f"_n_folds_{n_folds}"
        f"_movement_{movement}"
        f"_position_std_{position_std}"
        f"_discrete_var_{discrete_var}"
        f"_place_bin_size_{place_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    if rate_function_save_path.exists() and (overwrite == False):
        print("Rate function already exists, skipping...")
        return

    log_likelihood_save_path = log_likelihood_save_dir / (
        "log_likelihood_2d_trajectory"
        f"_{epoch}"
        f"_n_folds_{n_folds}"
        f"_movement_{movement}"
        f"_position_std_{position_std}"
        f"_discrete_var_{discrete_var}"
        f"_place_bin_size_{place_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    if log_likelihood_save_path.exists() and (overwrite == False):
        print("Log likelihood aleady exists, skipping...")
        return

    spike_indicators_save_path = spike_indicators_save_dir / (
        "spike_indicators_2d_trajectory_trial"
        f"_{epoch}"
        f"_n_folds_{n_folds}"
        f"_movement_{movement}"
        f"_position_std_{position_std}"
        f"_discrete_var_{discrete_var}"
        f"_place_bin_size_{place_bin_size}"
        f"_movement_var_{movement_var}.pkl"
    )

    rate_function = {}
    log_likelihood = {}
    spike_indicators = {}

    print(
        f"Computing rate function by trajectory from 2d place fields "
        f"{animal_name} {date} "
        f"epoch {epoch}, "
        f"n_folds {n_folds}, "
        f"movement {movement}, "
        f"position_std {position_std}, "
        f"discrete_var {discrete_var}, "
        f"place_bin_size {place_bin_size}, "
        f"movement_var {movement_var}, "
        f"overwrite {overwrite}."
    )

    for region in regions:
        rate_function[region] = {}
        log_likelihood[region] = {}
        spike_indicators[region] = {}

        print(f"in region {region}")

        t_position = timestamps_position[epoch][position_offset:]
        start_time = t_position[0]
        end_time = t_position[-1]
        n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1
        time = np.linspace(start_time, end_time, n_samples)

        head_position_2d = position_dict[epoch][position_offset:]
        f_pos = scipy.interpolate.interp1d(
            t_position, head_position_2d, axis=0, bounds_error=False, kind="linear"
        )
        head_position_2d_interp = f_pos(time)

        position_sampling_rate = len(head_position_2d) / (
            t_position[-1] - t_position[0]
        )
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
            spike_indicators[region][unit_id] = {}

            for trajectory_type in trajectory_types:
                rate_function[region][unit_id][trajectory_type] = []
                log_likelihood[region][unit_id][trajectory_type] = []
                spike_indicators[region][unit_id][trajectory_type] = []

                for fold_idx in range(n_folds):

                    fold_start_time, fold_stop_time = cv_start_stop_times[fold_idx]
                    inside_fold = (time > fold_start_time) & (time <= fold_stop_time)

                    classifier = classifier_dict[region][trajectory_type][fold_idx]

                    pos = classifier.place_fields_[("", 0)]["position"].to_numpy()
                    pos = np.asarray([np.asarray(i) for i in pos])

                    pf = classifier.place_fields_[("", 0)][:, unit_id].to_numpy()

                    tree = scipy.spatial.cKDTree(pos)  # pos shape (n_bins, 2)
                    _, pos_idx_per_t = tree.query(
                        head_position_2d_interp, k=1
                    )  # vectorized

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

                        r = pf[pos_idx_per_t[inds_to_keep]]
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
                        spike_indicators[region][unit_id][trajectory_type].append(
                            bins_with_spikes
                        )

    with open(rate_function_save_path, "wb") as f:
        pickle.dump(rate_function, f)
    with open(log_likelihood_save_path, "wb") as f:
        pickle.dump(log_likelihood, f)
    with open(spike_indicators_save_path, "wb") as f:
        pickle.dump(spike_indicators, f)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute time varying firing rate functions of neurons in a region from 2D place fields and actual position"
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
        "--position_std",
        type=float,
        help="Width of Gaussian used to estimate the place fields during encoding, in cm; float",
    )
    parser.add_argument(
        "--discrete_var",
        type=str,
        help="Type of model for the discrete latent variable; one of ['switching', 'random_walk', 'uniform']",
    )
    parser.add_argument(
        "--place_bin_size",
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
    compute_rate_function_2d_trajectory_cv(
        epoch_idx=args.epoch_idx,
        n_folds=args.n_folds,
        movement=args.movement,
        position_std=args.position_std,
        discrete_var=args.discrete_var,
        place_bin_size=args.place_bin_size,
        movement_var=args.movement_var,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
