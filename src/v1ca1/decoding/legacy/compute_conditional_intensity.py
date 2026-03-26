import replay_trajectory_classification as rtc
import spikeinterface.full as si

import numpy as np
import kyutils
import pickle

import scipy
import track_linearization as tl

from pathlib import Path

from track_linearization import make_track_graph

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
trajectory_type_list = [
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

temporal_bin_size_s = 2e-3
sampling_rate = int(1 / temporal_bin_size_s)

position_offset = 10

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


def compute_rate_function_for_one_region(
    classifier,
    pos_idx_per_t,
    time,
    is_inbound=None,
):

    if is_inbound is not None:
        unit_ids = classifier.place_fields_[("", "Inbound")].neuron.to_numpy()
    else:
        unit_ids = classifier.place_fields_[("", 0)].neuron.to_numpy()

    rate_functions = {}

    for unit_idx in unit_ids:

        if is_inbound is not None:
            place_field_inbound = classifier.place_fields_[("", "Inbound")][
                :, unit_idx
            ].to_numpy()
            place_field_outbound = classifier.place_fields_[("", "Outbound")][
                :, unit_idx
            ].to_numpy()

            rate_function = np.zeros(len(time))
            rate_function[is_inbound] = place_field_inbound[pos_idx_per_t][is_inbound]
            rate_function[~is_inbound] = place_field_outbound[pos_idx_per_t][
                ~is_inbound
            ]

        else:
            place_field = classifier.place_fields_[("", 0)][:, unit_idx].to_numpy()

            rate_function = place_field[pos_idx_per_t]

        nans = np.isnan(rate_function)
        if nans.sum() != 0:
            rate_function[nans] = (
                rate_function[np.where(nans)[0] - 1]
                + rate_function[np.where(nans)[0] + 1]
            ) / 2

        rate_functions[unit_idx] = rate_function

    return rate_functions


def compute_rate_function(
    epoch_idx: int,
    region: str,
    use_half: str,
    direction: str = "True",
    movement: str = "False",
    position_std: float = 6.0,
    discrete_var: str = "switching",
    place_bin_size: float = 2.0,
    movement_var: float = 6.0,
    overwrite: str = "False",
):

    epoch = run_epoch_list[epoch_idx]

    n_folds = int(n_folds)

    if direction == "False":
        direction = False
    elif direction == "True":
        direction = True
    else:
        raise ValueError("`direction` must be either True or False")

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

    direction_str = "_direction" if direction else ""
    movement_str = "_movement" if movement else ""

    print(
        f"Computing rate function for "
        f"{animal_name} {date} "
        f"epoch {epoch}, "
        f"region {region}, "
        f"{n_folds} fold cv, "
        f"direction {str(direction)}, "
        f"movement {str(movement)}, "
        f"position_std {position_std}, "
        f"discrete_var {discrete_var}, "
        f"place_bin_size {place_bin_size}, "
        f"movement_var {movement_var}, "
        f"overwrite {overwrite}."
    )

    rate_function_save_path = (
        data_path
        / "run_epochs"
        / "rate_function_cv"
        / (
            "rate_function"
            f"_{epoch}"
            f"_{region}"
            f"_{n_folds}_fold_cv"
            f"{direction_str}"
            f"{movement_str}"
            f"_position_std_{position_std}"
            f"_discrete_var_{discrete_var}"
            f"_place_bin_size_{place_bin_size}"
            f"_movement_var_{movement_var}.pkl"
        )
    )
    rate_function_save_path.parent.mkdir(parents=True, exist_ok=True)
    if rate_function_save_path.exists() and (overwrite == False):
        print("Aleady exists, skipping...")
        return

    # define reference time offset and subtract it from the timestamps
    t_position = timestamps_position[epoch][position_offset:]

    # define time vector for decoding (temporal resolution: 2 ms)
    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    # define position
    position = position_dict[epoch][position_offset:]

    # interpolate 2d position
    f_pos = scipy.interpolate.interp1d(
        t_position, position, axis=0, bounds_error=False, kind="linear"
    )
    position_interp = f_pos(time)

    classifier_path = (
        data_path
        / "run_epochs"
        / "classifier_2d_cv"
        / (
            "classifier"
            f"_{region}"
            f"_{epoch}"
            f"_2d"
            f"_0_of_{n_folds}_fold_cv"
            f"{direction_str}"
            f"{movement_str}"
            f"_fit_position_std_{position_std}"
            f"_discrete_var_{discrete_var}"
            f"_place_bin_size_{place_bin_size}"
            f"_movement_var_{movement_var}.pkl"
        )
    )
    if not classifier_path.exists():
        print("Classifier doesn't exist, skipping...")
        return

    classifier = rtc.SortedSpikesClassifier.load_model(filename=classifier_path)

    if direction:
        position_df = tl.get_linearized_position(
            position=position_interp,
            track_graph=track_graph,
            edge_order=linear_edge_order,
            edge_spacing=linear_edge_spacing,
        )
        position_interp_linear = position_df["linear_position"].to_numpy()

        is_inbound = np.insert(np.diff(position_interp_linear) < 0, 0, False)

        pos = classifier.place_fields_[("", "Inbound")]["position"].to_numpy()
        pos = np.array([list(tup) for tup in pos])
    else:
        is_inbound = None

        pos = classifier.place_fields_[("", 0)]["position"].to_numpy()
        pos = np.array([list(tup) for tup in pos])

    pos_idx_per_t = np.zeros(len(time), dtype=int)
    for i in range(len(time)):
        distances = np.linalg.norm(pos - position_interp[i], axis=1)
        pos_idx_per_t[i] = np.argmin(distances)

    # load classifier
    classifier_path = (
        data_path
        / "run_epochs"
        / "classifier_2d_cv"
        / (
            "classifier"
            f"_{region}"
            f"_{epoch}"
            f"_2d"
            f"_{fold_idx}_of_{n_folds}_fold_cv"
            f"{direction_str}"
            f"{movement_str}"
            f"_fit_position_std_{position_std}"
            f"_discrete_var_{discrete_var}"
            f"_place_bin_size_{place_bin_size}"
            f"_movement_var_{movement_var}.pkl"
        )
    )
    if not classifier_path.exists():
        print("Classifier doesn't exist, skipping...")
        return

    classifier = rtc.SortedSpikesClassifier.load_model(filename=classifier_path)

    rate_functions_cv.append(
        compute_rate_function_for_one_region(
            classifier,
            pos_idx_per_t[during_fold],
            time[during_fold],
            is_inbound[during_fold],
        )
    )

    rate_function = {}
    for unit_id in rate_functions_cv[0].keys():
        rate_function[unit_id] = np.concatenate(
            [rate_functions_cv[fold_idx][unit_id] for fold_idx in range(n_folds)]
        )
    rate_function["time"] = np.concatenate(time_list)

    with open(rate_function_save_path, "wb") as f:
        pickle.dump(rate_function, f)

    return rate_function


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute time varying firing rate functions of neurons in a region from 2D place fields and actual position"
    )
    parser.add_argument("--epoch_idx", type=int, help="Epoch index")
    parser.add_argument("--region", type=str, help="Brain region, e.g. `ca1`, `v1`")
    parser.add_argument(
        "--use_half",
        type=str,
        help="Time range to fit; one of ['first', 'second', 'all']",
    )
    parser.add_argument(
        "--direction",
        type=str,
        help="Whether to use different place fields for inbound vs. outbound directions; `True` or `False`",
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
    compute_rate_function(
        args.epoch_idx,
        args.region,
        args.use_half,
        args.direction,
        args.movement,
        args.position_std,
        args.discrete_var,
        args.place_bin_size,
        args.movement_var,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
