import spikeinterface.full as si
import probeinterface as pi
import numpy as np
import kyutils
import time
from pathlib import Path
import os
import pickle
from datetime import datetime, timedelta
import argparse

animal_name = "L14"
date = "20240611"
analysis_path = Path("/nimbus/kyu") / animal_name / "singleday_sort" / date

nwb_file_base_path = Path("/stelmo/nwb/raw")
nwb_file_name = f"{animal_name}{date}.nwb"

si.set_global_job_kwargs(chunk_size=30000, n_jobs=8, progress_bar=True)


def sort(
    probe_idx, shank_idx, recompute_sorting=False, recompute_sorting_analyzer=False
):

    print(f"Processing probe {probe_idx} shank {shank_idx}.")
    sort_save_path = (
        analysis_path / "sorting" / f"sorting_probe{probe_idx}_shank{shank_idx}_ms4"
    )

    print("Loading recording...")
    recording = si.read_nwb_recording(nwb_file_base_path / nwb_file_name)

    recording_shank = recording.select_channels(
        channel_ids=np.arange(
            128 * probe_idx + 32 * shank_idx, 128 * probe_idx + 32 * (shank_idx + 1)
        )
    )

    recording_filt = si.bandpass_filter(recording_shank, dtype=np.float64)

    print("Sorting...")
    if os.path.exists(sort_save_path) and (recompute_sorting == False):
        print("Sorting already exists.")
        sorting = si.NpzSortingExtractor(
            sort_save_path / "sorter_output" / "firings.npz"
        )
        print(sorting)
    else:
        recording_filt_whiten = si.whiten(recording_filt, dtype=np.float64)

        ms4_params = si.get_default_sorter_params("mountainsort4")

        ms4_params["adjacency_radius"] = 150
        ms4_params["filter"] = False
        ms4_params["whiten"] = False
        ms4_params["num_workers"] = 8

        start_sort_time = time.time()
        sorting = si.run_sorter(
            "mountainsort4",
            recording=recording_filt_whiten,
            folder=sort_save_path,
            **ms4_params,
        )
        end_sort_time = time.time()
        print(f"Sorting took {end_sort_time - start_sort_time} seconds.")

    sorting_analyzer_path = (
        analysis_path
        / "sorting_analyzer"
        / f"sorting_analyzer_probe{probe_idx}_shank{shank_idx}_ms4"
    )
    if os.path.exists(sorting_analyzer_path) and (recompute_sorting_analyzer == False):
        print("Sorting analyzer already exists.")
        sorting_analyzer = si.load_sorting_analyzer(
            folder=sorting_analyzer_path, load_extensions=False
        )
    else:
        print("Computing sorting analyzer extensions...")
        start_sorting_analyzer_compute_time = time.time()

        sorting_analyzer = si.create_sorting_analyzer(
            sorting,
            recording_filt,
            format="binary_folder",
            folder=sorting_analyzer_path,
            sparse=False,
            overwrite=recompute_sorting_analyzer,
        )
        sorting_analyzer.compute(
            input=[
                "random_spikes",
                "noise_levels",
                "templates",
                "waveforms",
                "principal_components",
                "quality_metrics",
                "spike_amplitudes",
                "correlograms",
            ],
            extension_params={
                "random_spikes": {"max_spikes_per_unit": 10000, "seed": 0},
                "waveforms": {"ms_before": 1.0, "ms_after": 1.0},
                "templates": {
                    "ms_before": 1.0,
                    "ms_after": 1.0,
                    "operators": ["median", "average"],
                },
                "principal_components": {
                    "n_components": 10,
                    "mode": "by_channel_local",
                    "whiten": True,
                    "dtype": "float32",
                },
                "noise_levels": {
                    "num_chunks_per_segment": 20,
                    # "chunk_size": 10000,
                    "seed": 1,
                },
                "correlograms": {"window_ms": 50, "bin_ms": 1.0, "method": "auto"},
                "spike_amplitudes": {"peak_sign": "neg"},
                "quality_metrics": {
                    "metric_names": [
                        "isi_violation",
                        "snr",
                        "firing_range",
                        "amplitude_median",
                    ],
                    "qm_params": {
                        "isi_violation": {"isi_threshold_ms": 1.5, "min_isi_ms": 0},
                        "snr": {"peak_sign": "neg", "peak_mode": "extremum"},
                        "firi+ng_range": {"bin_size_s": 5, "percentiles": (5, 95)},
                        "amplitude_median": {"peak_sign": "neg"},
                    },
                    "peak_sign": "neg",
                    "seed": 1,
                },
            },
        )
        pc_metrics = si.compute_pc_metrics(
            sorting_analyzer=sorting_analyzer,
            metric_names=["nn_isolation", "nn_noise_overlap"],
            qm_params={
                "nn_isolation": {
                    "max_spikes": 5000,
                    "min_spikes": 10,
                    "min_fr": 0.0,
                    "n_neighbors": 4,
                    "n_components": 10,
                    "radius_um": 100,
                    "peak_sign": "neg",
                },
                "nn_noise_overlap": {
                    "max_spikes": 5000,
                    "min_spikes": 10,
                    "min_fr": 0.0,
                    "n_neighbors": 4,
                    "n_components": 10,
                    "radius_um": 100,
                    "peak_sign": "neg",
                },
            },
            unit_ids=None,
            seed=1,
            progress_bar=True,
        )
        with open(
            sorting_analyzer_path / "pc_metrics.pkl",
            "wb",
        ) as f:
            pickle.dump(pc_metrics, f)

        end_sorting_analyzer_compute_time = time.time()
        print(
            f"Sorting analyzer took {end_sorting_analyzer_compute_time - start_sorting_analyzer_compute_time} seconds."
        )

    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run ms4 on concatenated run epochs")
    parser.add_argument("--probe_idx", type=int, help="Probe index")
    parser.add_argument("--shank_idx", type=int, help="Shank index")
    return parser.parse_args()


def main():
    args = parse_arguments()
    sort(
        args.probe_idx,
        args.shank_idx,
        recompute_sorting=False,
        recompute_sorting_analyzer=True,
    )


if __name__ == "__main__":
    main()
