from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import kyutils
import numpy as np
import pynwb


DEFAULT_DATA_ROOT = Path("/nimbus/kyu")
DEFAULT_NWB_ROOT = Path("/stelmo/nwb/raw")


def get_timestamps(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
) -> None:
    data_path = data_root / animal_name
    analysis_path = data_path / date
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"

    timestamps_position = {}
    timestamps_ephys = {}

    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    print(f"Processing {animal_name} {date}.")
    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save timestamps")
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing extracted Trodes data. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    get_timestamps(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
    )


if __name__ == "__main__":
    main()
