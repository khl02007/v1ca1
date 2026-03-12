from __future__ import annotations

"""Run Mountainsort4 spike sorting for one animal/date probe shank session.

This script reads a single-session NWB recording, selects one shank from one
probe, runs Mountainsort4 if sorter output does not already exist, and then
computes the corresponding SpikeInterface sorting analyzer outputs and quality
metrics under the configured analysis directory.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ANALYSIS_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_NWB_ROOT = Path("/stelmo/nwb/raw")
_SPIKEINTERFACE = None


def get_spikeinterface():
    """Import SpikeInterface lazily and configure shared job settings once."""
    global _SPIKEINTERFACE
    if _SPIKEINTERFACE is None:
        import spikeinterface.full as si

        si.set_global_job_kwargs(chunk_size=30000, n_jobs=8, progress_bar=True)
        _SPIKEINTERFACE = si
    return _SPIKEINTERFACE


def get_analysis_path(
    animal_name: str,
    date: str,
    analysis_root: Path,
) -> Path:
    """Return the analysis directory for one animal/date session."""
    return analysis_root / animal_name / date


def get_nwb_path(animal_name: str, date: str, nwb_root: Path) -> Path:
    """Return the NWB file path for one animal/date session."""
    return nwb_root / f"{animal_name}{date}.nwb"


def get_sorting_path(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
) -> Path:
    """Return the sorter output directory for one probe/shank."""
    return (
        get_analysis_path(animal_name, date, analysis_root)
        / "sorting"
        / f"sorting_probe{probe_idx}_shank{shank_idx}_ms4"
    )


def get_sorting_analyzer_path(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
) -> Path:
    """Return the sorting analyzer directory for one probe/shank."""
    return (
        get_analysis_path(animal_name, date, analysis_root)
        / "sorting_analyzer"
        / f"sorting_analyzer_probe{probe_idx}_shank{shank_idx}_ms4"
    )


def get_recording_shank(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    nwb_root: Path,
) -> Any:
    """Load the NWB recording and select channels for the requested shank."""
    si = get_spikeinterface()
    nwb_path = get_nwb_path(animal_name, date, nwb_root)
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    print(f"Loading recording from {nwb_path}...")
    recording = si.read_nwb_recording(nwb_path)
    channel_ids = np.arange(
        128 * probe_idx + 32 * shank_idx, 128 * probe_idx + 32 * (shank_idx + 1)
    )
    return recording.select_channels(channel_ids=channel_ids)


def run_mountainsort4(
    recording_filt: Any,
    sort_save_path: Path,
    ms4_params: dict[str, Any],
    recompute_sorting: bool,
) -> Any:
    """Run Mountainsort4 or load existing sorter output from disk."""
    si = get_spikeinterface()
    print("Sorting...")
    firings_path = sort_save_path / "sorter_output" / "firings.npz"
    if firings_path.exists() and not recompute_sorting:
        print("Sorting already exists.")
        sorting = si.NpzSortingExtractor(firings_path)
        print(sorting)
        return sorting

    sort_save_path.parent.mkdir(parents=True, exist_ok=True)

    start_sort_time = time.time()
    sorting = si.run_sorter(
        "mountainsort4",
        recording=recording_filt,
        folder=sort_save_path,
        **ms4_params,
    )
    end_sort_time = time.time()
    print(f"Sorting took {end_sort_time - start_sort_time} seconds.")
    return sorting


def compute_sorting_analyzer(
    sorting: Any,
    recording_filt: Any,
    sorting_analyzer_path: Path,
    recompute_sorting_analyzer: bool,
) -> None:
    """Compute or load analyzer extensions and derived quality metrics."""
    si = get_spikeinterface()
    if sorting_analyzer_path.exists() and not recompute_sorting_analyzer:
        print("Sorting analyzer already exists.")
        si.load_sorting_analyzer(
            folder=sorting_analyzer_path, load_extensions=False
        )
        return

    print("Computing sorting analyzer extensions...")
    start_sorting_analyzer_compute_time = time.time()
    sorting_analyzer_path.parent.mkdir(parents=True, exist_ok=True)

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
            "noise_levels": {"num_chunks_per_segment": 20, "seed": 1},
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
                    "firing_range": {"bin_size_s": 5, "percentiles": (5, 95)},
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
    with open(sorting_analyzer_path / "pc_metrics.pkl", "wb") as file:
        pickle.dump(pc_metrics, file)

    end_sorting_analyzer_compute_time = time.time()
    print(
        "Sorting analyzer took "
        f"{end_sorting_analyzer_compute_time - start_sorting_analyzer_compute_time} "
        "seconds."
    )


def sort(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    ms4_params: dict[str, Any],
    recompute_sorting: bool = False,
    recompute_sorting_analyzer: bool = False,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
) -> None:
    """Run filtering, whitening, sorting, and analyzer computation for one shank."""
    si = get_spikeinterface()
    print(f"Processing {animal_name} {date} probe {probe_idx} shank {shank_idx}.")
    recording_shank = get_recording_shank(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        nwb_root=nwb_root,
    )
    recording_filt = si.bandpass_filter(recording_shank, dtype=np.float64)
    recording_filt_whiten = si.whiten(recording_filt, dtype=np.float64)
    sorting = run_mountainsort4(
        recording_filt=recording_filt_whiten,
        sort_save_path=get_sorting_path(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            analysis_root=analysis_root,
        ),
        ms4_params=ms4_params,
        recompute_sorting=recompute_sorting,
    )
    compute_sorting_analyzer(
        sorting=sorting,
        recording_filt=recording_filt,
        sorting_analyzer_path=get_sorting_analyzer_path(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            analysis_root=analysis_root,
        ),
        recompute_sorting_analyzer=recompute_sorting_analyzer,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for spike sorting."""
    parser = argparse.ArgumentParser(description="Run mountainsort4 spike sorting")
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
    parser.add_argument("--probe-idx", type=int, required=True, help="Probe index")
    parser.add_argument("--shank-idx", type=int, required=True, help="Shank index")
    parser.add_argument(
        "--recompute-sorting",
        action="store_true",
        help="Re-run mountainsort4 even if sorter output already exists",
    )
    parser.add_argument(
        "--recompute-sorting-analyzer",
        action="store_true",
        help="Recompute the sorting analyzer even if it already exists",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help=f"Base directory for analysis output. Default: {DEFAULT_ANALYSIS_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint."""
    si = get_spikeinterface()
    args = parse_arguments()
    ms4_params = si.get_default_sorter_params("mountainsort4")
    ms4_params["adjacency_radius"] = 150
    ms4_params["filter"] = False
    ms4_params["whiten"] = False
    ms4_params["num_workers"] = 8
    sort(
        animal_name=args.animal_name,
        date=args.date,
        probe_idx=args.probe_idx,
        shank_idx=args.shank_idx,
        ms4_params=ms4_params,
        recompute_sorting=args.recompute_sorting,
        recompute_sorting_analyzer=args.recompute_sorting_analyzer,
        analysis_root=args.analysis_root,
        nwb_root=args.nwb_root,
    )


if __name__ == "__main__":
    main()
