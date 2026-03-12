import numpy as np
import kyutils
import pickle
import pynwb
from pathlib import Path

animal_name = "L14"
date = "20240611"
data_path = Path("/nimbus/kyu") / animal_name
analysis_path = data_path / "singleday_sort" / "20240611"

num_sleep_epochs = 5
num_run_epochs = 4
epoch_list, run_epoch_list = kyutils.get_epoch_list(
    num_sleep_epochs=num_sleep_epochs, num_run_epochs=num_run_epochs
)

nwb_file_base_path = Path("/stelmo/nwb/raw")


with open(analysis_path / "timestamps_ephys.pkl", "rb") as f:
    timestamps_ephys = pickle.load(f)

with open(analysis_path / "timestamps_position.pkl", "rb") as f:
    timestamps_position = pickle.load(f)

with open(analysis_path / "timestamps_ephys_all.pkl", "rb") as f:
    timestamps_ephys_all_ptp = pickle.load(f)


def get_first_and_last_lick_times(run_epoch, time, din):
    t_epoch = timestamps_ephys_all_ptp[
        (timestamps_ephys_all_ptp >= timestamps_ephys[run_epoch][0])
        & (timestamps_ephys_all_ptp < timestamps_ephys[run_epoch][-1])
    ]
    left_lick_times = t_epoch[
        np.searchsorted(
            time["data"]["time"], din["data"]["time"][din["data"]["state"] == 1]
        )
    ]

    first_lick_mask = np.concatenate(([True], np.diff(left_lick_times) > 13))
    first_lick_ind = np.where(first_lick_mask)[0]

    first_lick_times = left_lick_times[first_lick_mask]
    last_lick_times = np.concatenate(
        (left_lick_times[first_lick_ind[1:] - 1], [left_lick_times[-1]])
    )

    return np.sort(first_lick_times), np.sort(last_lick_times)


def get_trajectory_times():
    "based on poke times"
    trajectory_times = {}

    nwb_file_path = nwb_file_base_path / f"{animal_name}{date}.nwb"
    with pynwb.NWBHDF5IO(path=nwb_file_path, mode="r") as io:
        nwbf = io.read()
        poke_left_data = (
            nwbf.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["Poke Left"]
            .data[:]
        )
        poke_left_timestamps = (
            nwbf.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["Poke Left"]
            .timestamps[:]
        )
        poke_right_data = (
            nwbf.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["Poke Right"]
            .data[:]
        )
        poke_right_timestamps = (
            nwbf.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["Poke Right"]
            .timestamps[:]
        )
        poke_center_data = (
            nwbf.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["Poke Center"]
            .data[:]
        )
        poke_center_timestamps = (
            nwbf.processing["behavior"]
            .data_interfaces["behavioral_events"]
            .time_series["Poke Center"]
            .timestamps[:]
        )
        poke_left_times = poke_left_timestamps[poke_left_data == 1]
        poke_right_times = poke_right_timestamps[poke_right_data == 1]
        poke_center_times = poke_center_timestamps[poke_center_data == 1]

    for run_epoch in run_epoch_list:
        trajectory_times[run_epoch] = {}

        poke_left_times_epoch = poke_left_times[
            (poke_left_times >= timestamps_ephys[run_epoch][0])
            & (poke_left_times < timestamps_ephys[run_epoch][-1])
        ]
        poke_right_times_epoch = poke_right_times[
            (poke_right_times >= timestamps_ephys[run_epoch][0])
            & (poke_right_times < timestamps_ephys[run_epoch][-1])
        ]
        poke_center_times_epoch = poke_center_times[
            (poke_center_times >= timestamps_ephys[run_epoch][0])
            & (poke_center_times < timestamps_ephys[run_epoch][-1])
        ]

        poke_times = np.concatenate(
            (poke_left_times_epoch, poke_center_times_epoch, poke_right_times_epoch)
        )

        # 0=left, 1=center, 2=right
        well_type = np.concatenate(
            (
                0 * np.ones(poke_left_times_epoch.shape),
                1 * np.ones(poke_center_times_epoch.shape),
                2 * np.ones(poke_right_times_epoch.shape),
            )
        )

        poke_time_order_idx = np.argsort(poke_times)

        poke_times = poke_times[poke_time_order_idx]
        well_type = well_type[poke_time_order_idx]

        # trajectory times
        left_to_center_traj_times = []
        center_to_left_traj_times = []
        right_to_center_traj_times = []
        center_to_right_traj_times = []
        for i in range(len(poke_times) - 1):
            if (well_type[i] == 0) and (well_type[i + 1] == 1):
                left_to_center_traj_times.append([poke_times[i], poke_times[i + 1]])
            elif (well_type[i] == 1) and (well_type[i + 1] == 0):
                center_to_left_traj_times.append([poke_times[i], poke_times[i + 1]])
            elif (well_type[i] == 2) and (well_type[i + 1] == 1):
                right_to_center_traj_times.append([poke_times[i], poke_times[i + 1]])
            elif (well_type[i] == 1) and (well_type[i + 1] == 2):
                center_to_right_traj_times.append([poke_times[i], poke_times[i + 1]])

        trajectory_times[run_epoch]["left_to_center"] = np.asarray(
            left_to_center_traj_times
        )
        trajectory_times[run_epoch]["center_to_left"] = np.asarray(
            center_to_left_traj_times
        )
        trajectory_times[run_epoch]["right_to_center"] = np.asarray(
            right_to_center_traj_times
        )
        trajectory_times[run_epoch]["center_to_right"] = np.asarray(
            center_to_right_traj_times
        )

    with open(analysis_path / "trajectory_times.pkl", "wb") as f:
        pickle.dump(trajectory_times, f)
    return None


def main():
    get_trajectory_times()


if __name__ == "__main__":
    main()
