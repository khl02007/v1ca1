from __future__ import annotations

"""Generate a figurl for interactive spike sorting curation.

This script creates a figurl URL for one animal/date/probe/shank session and
stores that URL in the analysis directory. Opening the figurl lets a curator
review units in the browser and assign curation labels such as accept/reject.

The figurl is configured with a `sortingCuration` URI that points to a JSON
file in the `LorenFrankLab/sorting-curations` GitHub repository. Those labels
are not applied inside this script. Instead, downstream code such as
`consolidate_sorting.py` reads the saved `curation.json` and applies it to the
SpikeInterface sorting object via `si.apply_sortingview_curation(...)`.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

DEFAULT_ANALYSIS_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_NWB_ROOT = Path("/stelmo/nwb/raw")
DEFAULT_CURATION_BASE_URI = (
    "gh://LorenFrankLab/sorting-curations/main/khl02007"
)
_SPIKEINTERFACE = None


def get_spikeinterface():
    """Import SpikeInterface lazily and configure shared job settings once."""
    global _SPIKEINTERFACE
    if _SPIKEINTERFACE is None:
        import spikeinterface.full as si

        si.set_global_job_kwargs(chunk_size=30000, n_jobs=8, progress_bar=True)
        _SPIKEINTERFACE = si
    return _SPIKEINTERFACE


def reformat_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert metric dictionaries into the format expected by sortingview."""
    reformatted_metrics = {}
    for metric_name, metric_values in metrics.items():
        reformatted_metrics[metric_name] = {
            str(unit_id): metric_value
            for unit_id, metric_value in metric_values.items()
        }

    return [
        {
            "name": metric_name,
            "label": metric_name,
            "tooltip": metric_name,
            "data": metric_values,
        }
        for metric_name, metric_values in reformatted_metrics.items()
    ]


def create_figurl_spikesorting(
    recording: Any,
    sorting: Any,
    label: str,
    metrics: list[dict[str, Any]] | None = None,
    curation_uri: str | None = None,
) -> str:
    """Create a figurl URL for interactive spike sorting review.

    If `curation_uri` is provided, the figurl loads and writes curation state
    against that remote JSON path so manual labels can be reused downstream.
    """
    try:
        import sortingview.views as vv
        from sortingview.SpikeSortingView import SpikeSortingView
    except ImportError as exc:
        raise ImportError(
            "generate_figurl.py requires `sortingview` and `kachery-cloud` "
            "to create figurls."
        ) from exc

    spike_sorting_view = SpikeSortingView.create(
        recording=recording,
        sorting=sorting,
        segment_duration_sec=60 * 20,
        snippet_len=(20, 20),
        max_num_snippets_per_segment=300,
        channel_neighborhood_size=12,
    )

    raster_plot_subsample_max_firing_rate = 50
    spike_amplitudes_subsample_max_firing_rate = 50
    layout = vv.MountainLayout(
        items=[
            vv.MountainLayoutItem(
                label="Summary",
                view=spike_sorting_view.sorting_summary_view(),
            ),
            vv.MountainLayoutItem(
                label="Units table",
                view=spike_sorting_view.units_table_view(
                    unit_ids=spike_sorting_view.unit_ids,
                    unit_metrics=metrics,
                ),
            ),
            vv.MountainLayoutItem(
                label="Raster plot",
                view=spike_sorting_view.raster_plot_view(
                    unit_ids=spike_sorting_view.unit_ids,
                    _subsample_max_firing_rate=raster_plot_subsample_max_firing_rate,
                ),
            ),
            vv.MountainLayoutItem(
                label="Spike amplitudes",
                view=spike_sorting_view.spike_amplitudes_view(
                    unit_ids=spike_sorting_view.unit_ids,
                    _subsample_max_firing_rate=spike_amplitudes_subsample_max_firing_rate,
                ),
            ),
            vv.MountainLayoutItem(
                label="Autocorrelograms",
                view=spike_sorting_view.autocorrelograms_view(
                    unit_ids=spike_sorting_view.unit_ids
                ),
            ),
            vv.MountainLayoutItem(
                label="Cross correlograms",
                view=spike_sorting_view.cross_correlograms_view(
                    unit_ids=spike_sorting_view.unit_ids
                ),
            ),
            vv.MountainLayoutItem(
                label="Avg waveforms",
                view=spike_sorting_view.average_waveforms_view(
                    unit_ids=spike_sorting_view.unit_ids
                ),
            ),
            vv.MountainLayoutItem(
                label="Electrode geometry",
                view=spike_sorting_view.electrode_geometry_view(),
            ),
            vv.MountainLayoutItem(
                label="Curation",
                view=vv.SortingCuration2(),
                is_control=True,
            ),
        ]
    )

    if curation_uri:
        url_state = {"sortingCuration": curation_uri}
    else:
        url_state = None
    return layout.url(label=label, state=url_state)


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


def get_figurl_path(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
) -> Path:
    """Return the output text file path for the generated figurl."""
    return (
        get_analysis_path(animal_name, date, analysis_root)
        / "figurl"
        / f"figurl_probe{probe_idx}_shank{shank_idx}_ms4.txt"
    )


def get_curation_uri(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    curation_base_uri: str,
) -> str:
    """Return the curation JSON URI for one probe/shank."""
    return (
        f"{curation_base_uri}/{animal_name}/{date}/probe{probe_idx}/"
        f"shank{shank_idx}/ms4/curation.json"
    )


def get_recording_shank(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    nwb_root: Path,
):
    """Load the NWB recording and select channels for the requested shank."""
    import numpy as np

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


def load_sorting(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
):
    """Load sorter output for one probe/shank."""
    si = get_spikeinterface()
    firings_path = get_sorting_path(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    ) / "sorter_output" / (
        "firings.npz"
    )
    if not firings_path.exists():
        raise FileNotFoundError(f"Sorting output not found: {firings_path}")
    return si.NpzSortingExtractor(firings_path)


def load_metrics(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
):
    """Load quality metrics and PC metrics for one probe/shank."""
    import pandas as pd

    sorting_analyzer_path = get_sorting_analyzer_path(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )
    metrics_path = (
        sorting_analyzer_path / "extensions" / "quality_metrics" / "metrics.csv"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(f"Quality metrics not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path)

    pc_metrics_path = sorting_analyzer_path / "pc_metrics.pkl"
    if not pc_metrics_path.exists():
        raise FileNotFoundError(f"PC metrics not found: {pc_metrics_path}")
    with open(pc_metrics_path, "rb") as file:
        pc_metrics = pickle.load(file)

    return metrics, pc_metrics


def generate_figurl(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    curation_base_uri: str = DEFAULT_CURATION_BASE_URI,
) -> None:
    """Generate and save a figurl for one animal/date probe/shank session.

    The saved text file contains a URL that opens an interactive sortingview
    session. Curators can use that page to mark units with labels stored in the
    `sorting-curations` GitHub repository. Applying those labels to a
    SpikeInterface sorting object happens later in downstream scripts, not here.
    """
    print(
        f"Generating figurl for {animal_name} {date} "
        f"probe {probe_idx} shank {shank_idx}."
    )

    recording_shank = get_recording_shank(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        nwb_root=nwb_root,
    )
    sorting = load_sorting(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )
    metrics, pc_metrics = load_metrics(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )

    reformatted_metric = reformat_metrics(
        {
            "nn_isolation": pc_metrics["nn_isolation"],
            "nn_noise_overlap": pc_metrics["nn_noise_overlap"],
            "isi_violations_ratio": metrics["isi_violations_ratio"],
            "snr": metrics["snr"],
            "amplitude_median": metrics["amplitude_median"],
        }
    )

    figurl_path = get_figurl_path(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )
    figurl_path.parent.mkdir(parents=True, exist_ok=True)

    figurl = create_figurl_spikesorting(
        recording=recording_shank,
        sorting=sorting,
        label=f"{animal_name} {date} probe{probe_idx} shank{shank_idx} ms4",
        metrics=reformatted_metric,
        curation_uri=get_curation_uri(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            curation_base_uri=curation_base_uri,
        ),
    )
    with open(figurl_path, "w") as file:
        file.write(figurl)

    print(f"Saved figurl to {figurl_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for figurl generation."""
    parser = argparse.ArgumentParser(description="Generate a spikesorting figurl")
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
    parser.add_argument(
        "--curation-base-uri",
        default=DEFAULT_CURATION_BASE_URI,
        help=(
            "Base URI for remote curation JSON files used by figurl and "
            "downstream curation application. "
            f"Default: {DEFAULT_CURATION_BASE_URI}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint and report execution time."""
    args = parse_arguments()
    start = time.perf_counter()

    generate_figurl(
        animal_name=args.animal_name,
        date=args.date,
        probe_idx=args.probe_idx,
        shank_idx=args.shank_idx,
        analysis_root=args.analysis_root,
        nwb_root=args.nwb_root,
        curation_base_uri=args.curation_base_uri,
    )

    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")


if __name__ == "__main__":
    main()
