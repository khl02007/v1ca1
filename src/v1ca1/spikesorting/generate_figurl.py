import spikeinterface.full as si
import probeinterface as pi
import numpy as np
import kyutils
import pickle
import time
from pathlib import Path
import pandas as pd

import argparse

animal_name = "L14"
data_path = Path("/nimbus/kyu") / animal_name
date = "20240611"
analysis_path = Path("/nimbus/kyu") / animal_name / "singleday_sort" / "20240611"

nwb_file_base_path = Path("/stelmo/nwb/raw")
nwb_file_name = f"{animal_name}{date}.nwb"

si.set_global_job_kwargs(chunk_size=30000, n_jobs=8, progress_bar=True)


def generate_figurl(probe_idx, shank_idx):
    print(f"Generating figurl for {animal_name} probe{probe_idx} shank{shank_idx}.")

    recording = si.read_nwb_recording(nwb_file_base_path / nwb_file_name)

    recording_shank = recording.select_channels(
        channel_ids=np.arange(
            128 * probe_idx + 32 * shank_idx, 128 * probe_idx + 32 * (shank_idx + 1)
        )
    )

    sorting_path = (
        analysis_path / "sorting" / f"sorting_probe{probe_idx}_shank{shank_idx}_ms4"
    )
    sorting = si.NpzSortingExtractor(sorting_path / "sorter_output" / "firings.npz")

    metrics_path = (
        analysis_path
        / "sorting_analyzer"
        / f"sorting_analyzer_probe{probe_idx}_shank{shank_idx}_ms4"
        / "extensions"
        / "quality_metrics"
        / "metrics.csv"
    )
    metrics = pd.read_csv(metrics_path)
    pc_metrics_path = (
        analysis_path
        / "sorting_analyzer"
        / f"sorting_analyzer_probe{probe_idx}_shank{shank_idx}_ms4"
        / "pc_metrics.pkl"
    )
    with open(pc_metrics_path, "rb") as file:
        pc_metrics = pickle.load(file)
    reformatted_metric = kyutils.spikesorting.figurl.reformat_metrics(
        {
            "nn_isolation": pc_metrics["nn_isolation"],
            "nn_noise_overlap": pc_metrics["nn_noise_overlap"],
            "isi_violations_ratio": metrics["isi_violations_ratio"],
            "snr": metrics["snr"],
            "amplitude_median": metrics["amplitude_median"],
        }
    )

    txt_file_name = (
        analysis_path / "figurl" / f"figurl_probe{probe_idx}_shank{shank_idx}_ms4.txt"
    )
    txt_file_name.parent.mkdir(parents=True, exist_ok=True)

    figurl = kyutils.create_figurl_spikesorting(
        recording=recording_shank,
        sorting=sorting,
        label=f"{animal_name} 20240611 probe{probe_idx} shank{shank_idx} ms4",
        metrics=reformatted_metric,
        curation_uri=f"gh://LorenFrankLab/sorting-curations/main/khl02007/{animal_name}/20240611/probe{probe_idx}/shank{shank_idx}/ms4/curation.json",
    )
    with open(txt_file_name, "w") as file:
        file.write(figurl)

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generates figurl")
    parser.add_argument("--probe_idx", type=int, help="Probe index")
    parser.add_argument("--shank_idx", type=int, help="Shank index")
    return parser.parse_args()


def main():
    args = parse_arguments()
    start = time.perf_counter()

    generate_figurl(args.probe_idx, args.shank_idx)

    end = time.perf_counter()

    elapsed = end - start
    print(f"Execution time: {elapsed:.6f} seconds")


if __name__ == "__main__":
    main()
