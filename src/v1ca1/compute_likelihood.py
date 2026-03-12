from typing import Dict

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
data_path = Path("/stelmo/kyu") / animal_name / date

num_sleep_epochs = 6
num_run_epochs = 5


epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)


with open(data_path / "run_epochs" / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)

with open(data_path / "run_epochs" / "position_interp.pkl", "rb") as f:
    position_dict = pickle.load(f)

with open(data_path / "run_epochs" / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(data_path / "run_epochs" / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

sampling_rate = int(1 / (2e-3))

position_offset = 10


def get_time(epoch: str, ptp: bool = True):
    # define reference time offset and subtract it from the timestamps
    t_position = timestamps_position[epoch][position_offset:]
    if not ptp:
        t_position = t_position - timestamps_ephys[epoch][0]

    # define time vector for decoding (temporal resolution: 2 ms)
    start_time = t_position[0]
    end_time = t_position[-1]

    n_samples = int(np.ceil((end_time - start_time) * sampling_rate)) + 1

    time = np.linspace(start_time, end_time, n_samples)

    return time


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


def compute_likelihood_for_one_region(
    rate_function: Dict[int, np.ndarray],
    spike_indicator: np.ndarray,
):

    likelihood = np.zeros(spike_indicator.shape)
    for unit_idx, rate in rate_function.items():
        prob_spike = rate * np.exp(-rate)
        prob_no_spike = np.exp(-rate)
        bins_with_spikes = spike_indicator[:, unit_idx].astype(bool)
        likelihood[:, unit_idx][bins_with_spikes] = prob_spike[bins_with_spikes]
        likelihood[:, unit_idx][~bins_with_spikes] = prob_no_spike[~bins_with_spikes]
    return likelihood


def compute_likelihood(
    epoch_idx: int,
    region: str,
    use_half: str = "all",
    direction: str = "True",
    movement: str = "False",
    position_std: float = 6.0,
    discrete_var: str = "switching",
    place_bin_size: float = 2.0,
    movement_var: float = 6.0,
    overwrite: str = "False",
):
    epoch = run_epoch_list[epoch_idx]

    assert use_half in ["first", "second", "all"], ValueError(
        "`use_half` must be either 'first', 'second', or 'all'"
    )

    if direction == "False":
        direction = False
    elif direction == "True":
        direction = True
    else:
        raise ValueError("`direction` must be either 'True' or 'False'")

    if movement == "False":
        movement = False
    elif movement == "True":
        movement = True
    else:
        raise ValueError("`movement` must be either 'True' or 'False'")

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
        f"Computing likelihood of real spikes based on encoding model from "
        f"{animal_name} {date}, "
        f"epoch {epoch}, "
        f"region {region}, "
        f"use_half {use_half}, "
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
        / "rate_function"
        / (
            "rate_function"
            f"_{epoch}"
            f"_{region}"
            f"_{use_half}"
            f"{direction_str}"
            f"{movement_str}"
            f"_position_std_{position_std}"
            f"_discrete_var_{discrete_var}"
            f"_place_bin_size_{place_bin_size}"
            f"_movement_var_{movement_var}.pkl"
        )
    )

    if not rate_function_save_path.exists():
        print("Rate function does not exist!")
        return
    else:
        with open(rate_function_save_path, "rb") as f:
            rate_function = pickle.load(f)

    likelihood_save_path = (
        data_path
        / "run_epochs"
        / "likelihood"
        / (
            "likelihood"
            f"_{epoch}"
            f"_{region}"
            f"_{use_half}"
            f"{direction_str}"
            f"{movement_str}"
            f"_position_std_{position_std}"
            f"_discrete_var_{discrete_var}"
            f"_place_bin_size_{place_bin_size}"
            f"_movement_var_{movement_var}.pkl"
        )
    )
    likelihood_save_path.parent.mkdir(parents=True, exist_ok=True)
    if likelihood_save_path.exists() and (not overwrite):
        print("Aleady exists, skipping...")
        return

    time = get_time(epoch, ptp=True)

    sorting = si.load_extractor(data_path / "run_epochs" / f"sorting_{region}")

    # bin spikes
    spike_indicator = get_spike_indicator(sorting, timestamps_ephys_all_ptp, time)

    likelihood = compute_likelihood_for_one_region(
        rate_function,
        spike_indicator,
    )

    with open(likelihood_save_path, "wb") as f:
        pickle.dump(likelihood, f)

    return likelihood


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute likelihood for each time bin of each neuron"
    )
    parser.add_argument("--epoch_idx", type=int, help="Epoch index")
    parser.add_argument("--region", type=str, help="Brain region")
    parser.add_argument(
        "--use_half",
        type=str,
        help="Time range to fit; one of ['first', 'second', 'all']",
    )
    parser.add_argument(
        "--direction",
        type=str,
        help="Whether to simulate based on inbound vs. outbound; bool",
    )
    parser.add_argument(
        "--movement",
        type=str,
        help="Whether to simulate based on movement vs. all times; bool",
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
        help="Place bin size in cm; float",
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
    compute_likelihood(
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
