import spikeinterface.full as si
import numpy as np
import kyutils
import pickle

import replay_trajectory_classification as rtc
import scipy
import track_linearization as tl
from track_linearization import make_track_graph
import argparse

import position_tools as pt
from pathlib import Path

animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / date

num_sleep_epochs = 5
num_run_epochs = 4

epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

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


# fit
def fit_speed_decode(
    epoch_idx,
    use_half="all",
    movement="True",
    speed_std=2.0,
    discrete_var="switching",
    speed_bin_size=0.5,
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

    print(
        f"Fitting speed decoder "
        f"{animal_name} {date} "
        f"epoch {epoch}, "
        f"use_half {use_half}, "
        f"movement {movement}, "
        f"speed_std {speed_std}, "
        f"discrete_var {discrete_var}, "
        f"speed_bin_size {speed_bin_size}, "
        f"movement_var {movement_var}, "
        f"overwrite {overwrite}."
    )

    # define reference time offset and subtract it from the timestamps
    t_position = timestamps_position[epoch][position_offset:]

    # define time vector for decoding (temporal resolution: 2 ms)
    if use_half == "first":
        start_time = t_position[0]
        end_time = t_position[int(len(t_position) / 2)]
    elif use_half == "second":
        start_time = t_position[int(len(t_position) / 2) + 1]
        end_time = t_position[-1]
    else:
        start_time = t_position[0]
        end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    # define position
    position = position_dict[epoch][position_offset:]

    position_sampling_rate = len(position) / (t_position[-1] - t_position[0])
    speed = pt.get_speed(
        position, time=t_position, sampling_frequency=position_sampling_rate, sigma=0.1
    )
    f_speed = scipy.interpolate.interp1d(
        t_position, speed, axis=0, bounds_error=False, kind="linear"
    )
    speed_interp = f_speed(time)

    classifier_save_dir = analysis_path / "classifier_speed"
    classifier_save_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        spike_indicator = get_spike_indicator(
            sorting[region], timestamps_ephys_all_ptp, time
        )

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

        if classifier_save_path.exists() and (not overwrite):
            print("Aleady exists, skipping...")
            return

        if movement:
            inds_to_keep = speed_interp > speed_threshold
        else:
            inds_to_keep = np.ones(len(speed_interp), dtype=bool)

        # define environment
        environment = rtc.Environment(
            place_bin_size=speed_bin_size,
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

        # define classifiers
        classifier = rtc.SortedSpikesClassifier(
            environments=environment,
            continuous_transition_types=continuous_transition_types,
            observation_models=observation_models,
            sorted_spikes_algorithm="spiking_likelihood_kde_gpu",
            sorted_spikes_algorithm_params={
                "position_std": speed_std,
                "use_diffusion": False,
                "block_size": int(2**11),
            },
        )

        # fit classifiers
        classifier.fit(
            speed_interp[inds_to_keep],
            spike_indicator[inds_to_keep],
            encoding_group_labels=encoding_labels,
        )

        classifier.save_model(classifier_save_path)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fit speed decoder on CA1 and V1 for the four trajectories"
    )
    parser.add_argument("--epoch_idx", type=int, help="Epoch to fit")
    parser.add_argument(
        "--use_half",
        type=str,
        help="Time range to fit; one of ['first', 'second', 'all']",
    )
    parser.add_argument(
        "--movement",
        type=str,
        help="Whether to encode based on movement vs. all times; bool",
    )
    parser.add_argument(
        "--speed_std",
        type=float,
        help="Width of gaussian for estimating speed fields; float",
    )
    parser.add_argument(
        "--discrete_var",
        type=str,
        help="Type of movement model for the discrete latent variable; one of ['switching', 'random_walk', 'uniform']",
    )
    parser.add_argument(
        "--speed_bin_size",
        type=float,
        help="size of speed bin, in cm/s",
    )
    parser.add_argument(
        "--movement_var",
        type=float,
        help="variance of random walk for movement dynamics, in cm^2",
    )

    parser.add_argument(
        "--overwrite",
        type=str,
        help="Overwrite classifier if exists; one of ['True', 'False']",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    fit_speed_decode(
        epoch_idx=args.epoch_idx,
        use_half=args.use_half,
        movement=args.movement,
        speed_std=args.speed_std,
        discrete_var=args.discrete_var,
        speed_bin_size=args.speed_bin_size,
        movement_var=args.movement_var,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
