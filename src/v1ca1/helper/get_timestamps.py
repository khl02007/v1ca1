import numpy as np
import kyutils
from pathlib import Path
import os
import pickle
import pynwb

import argparse

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


def get_timestamps():

    timestamps_position = {}
    timestamps_ephys = {}

    nwb_file_path = nwb_file_base_path / f"{animal_name}{date}.nwb"

    with pynwb.NWBHDF5IO(nwb_file_path, "r") as io:
        nwbfile = io.read()
        epoch_tags = np.concatenate(list(nwbfile.epochs[:]["tags"]))
        timestamps_ephys_all = nwbfile.acquisition["e-series"].timestamps[:]

        video_files = list(
            nwbfile.processing["video_files"]
            .data_interfaces["video"]
            .time_series.keys()
        )
        for i, video_file in enumerate(video_files):
            print(f"processing epoch {epoch_tags[i]}")

            timestamps_position[epoch_tags[i]] = (
                nwbfile.processing["video_files"]
                .data_interfaces["video"]
                .time_series[video_file]
                .timestamps[:]
            )

    for epoch in epoch_tags:
        t_ephys = kyutils.readTrodesExtractedDataFile(
            data_path
            / date
            / f"{date}_{animal_name}_{epoch}"
            / f"{date}_{animal_name}_{epoch}.time"
            / f"{date}_{animal_name}_{epoch}.timestamps.dat"
        )
        timestamps_ephys[epoch] = kyutils.nwb.utils.generate_ephys_timestamps(t_ephys)

    with open(analysis_path / "timestamps_ephys.pkl", "wb") as f:
        pickle.dump(timestamps_ephys, f)
    with open(analysis_path / "timestamps_position.pkl", "wb") as f:
        pickle.dump(timestamps_position, f)
    with open(analysis_path / "timestamps_ephys_all.pkl", "wb") as f:
        pickle.dump(timestamps_ephys_all, f)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Save timestamps")
    return parser.parse_args()


def main():
    # args = parse_arguments()
    get_timestamps()


if __name__ == "__main__":
    main()
