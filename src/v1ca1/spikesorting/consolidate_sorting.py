from __future__ import annotations

"""Apply manual curation JSON files and consolidate curated sortings by region.

This script reads per-shank Mountainsort4 outputs, applies sortingview curation
JSON files from the local `sorting-curations` checkout, saves curated per-shank
sortings, and then aggregates those curated sortings into region-level V1 and
CA1 outputs.
"""

import argparse
from pathlib import Path
from typing import Any

from v1ca1.helper.run_logging import write_run_log

DEFAULT_ANALYSIS_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_CURATION_ROOT = Path("/home/kyu/repos/sorting-curations/khl02007")
DEFAULT_REGION_PROBES = {
    "v1": [0, 3],
    "ca1": [1, 2],
}
_SPIKEINTERFACE = None


def parse_probe_list(probe_list: str) -> list[int]:
    """Parse a comma-separated probe list such as '0,3' into integers."""
    return [int(token.strip()) for token in probe_list.split(",") if token.strip()]


def validate_region_probes(region_probes: dict[str, list[int]]) -> None:
    """Validate that region probe assignments are non-empty and disjoint."""
    for region, probe_indices in region_probes.items():
        if not probe_indices:
            raise ValueError(f"Probe list for {region} cannot be empty.")

    if set(region_probes["v1"]) & set(region_probes["ca1"]):
        raise ValueError("V1 and CA1 probe lists must be disjoint.")


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


def get_curated_sorting_path(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
) -> Path:
    """Return the save directory for one curated probe/shank sorting."""
    return (
        get_analysis_path(animal_name, date, analysis_root)
        / "sorting"
        / f"curated_sorting_probe{probe_idx}_shank{shank_idx}_ms4"
    )


def get_region_sorting_path(
    animal_name: str,
    date: str,
    region: str,
    analysis_root: Path,
) -> Path:
    """Return the save directory for one region-level aggregated sorting."""
    return get_analysis_path(animal_name, date, analysis_root) / f"sorting_{region}"


def get_sorting_curation_path(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    curation_root: Path,
) -> Path:
    """Return the local curation JSON path for one probe/shank."""
    return (
        curation_root
        / animal_name
        / date
        / f"probe{probe_idx}"
        / f"shank{shank_idx}"
        / "ms4"
        / "curation.json"
    )


def load_sorting(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
) -> Any:
    """Load sorter output for one probe/shank."""
    si = get_spikeinterface()
    firings_path = (
        get_sorting_path(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            analysis_root=analysis_root,
        )
        / "sorter_output"
        / "firings.npz"
    )
    if not firings_path.exists():
        raise FileNotFoundError(f"Sorting output not found: {firings_path}")
    return si.NpzSortingExtractor(firings_path)


def apply_curation(
    sorting: Any,
    sorting_curations_path: Path,
) -> Any:
    """Apply sortingview curation labels and postprocess the curated sorting."""
    si = get_spikeinterface()
    curated_sorting = si.apply_sortingview_curation(
        sorting,
        sorting_curations_path,
        exclude_labels=["noise", "reject"],
        include_labels=None,
        skip_merge=False,
    )
    curated_sorting = si.remove_duplicated_spikes(
        curated_sorting,
        censored_period_ms=0.1,
    )
    curated_sorting = si.remove_redundant_units(
        curated_sorting,
        duplicate_threshold=0.9,
        remove_strategy="max_spikes",
        align=False,
    )
    return curated_sorting


def stamp_sorting_provenance(
    sorting: Any,
    region: str,
    probe_idx: int,
    shank_idx: int,
    sorting_curations_path: Path,
    curation_root: Path,
) -> Any:
    """Attach per-unit provenance needed by downstream NWB export."""
    unit_ids = sorting.get_unit_ids()
    curation_json_relpath = sorting_curations_path.relative_to(curation_root).as_posix()
    sorting.set_property("region", [region] * len(unit_ids))
    sorting.set_property("probe_idx", [int(probe_idx)] * len(unit_ids))
    sorting.set_property("shank_idx", [int(shank_idx)] * len(unit_ids))
    sorting.set_property("curation_json_relpath", [curation_json_relpath] * len(unit_ids))
    return sorting


def consolidate_region_sorting(
    animal_name: str,
    date: str,
    region: str,
    probe_indices: list[int],
    analysis_root: Path,
    curation_root: Path,
) -> Any:
    """Apply curation across a region's probes and save the aggregated sorting."""
    si = get_spikeinterface()
    curated_sortings = []

    for probe_idx in probe_indices:
        print(f"probe{probe_idx}")
        for shank_idx in range(4):
            print(f"shank{shank_idx}")
            sorting_curations_path = get_sorting_curation_path(
                animal_name=animal_name,
                date=date,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                curation_root=curation_root,
            )
            if not sorting_curations_path.exists():
                print(f"Skipping missing curation file: {sorting_curations_path}")
                continue

            sorting = load_sorting(
                animal_name=animal_name,
                date=date,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                analysis_root=analysis_root,
            )
            print(
                f"{sorting.get_num_units()} units in probe {probe_idx} "
                f"shank {shank_idx}"
            )

            curated_sorting = apply_curation(
                sorting=sorting,
                sorting_curations_path=sorting_curations_path,
            )
            curated_sorting = stamp_sorting_provenance(
                sorting=curated_sorting,
                region=region,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                sorting_curations_path=sorting_curations_path,
                curation_root=curation_root,
            )
            print(
                f"after curation {curated_sorting.get_num_units()} units "
                f"in probe {probe_idx} shank {shank_idx}"
            )

            curated_sorting.save_to_folder(
                get_curated_sorting_path(
                    animal_name=animal_name,
                    date=date,
                    probe_idx=probe_idx,
                    shank_idx=shank_idx,
                    analysis_root=analysis_root,
                ),
                overwrite=True,
            )
            curated_sortings.append(curated_sorting)

    if not curated_sortings:
        raise FileNotFoundError(
            f"No curated sortings found for {region} {animal_name} {date}."
        )

    region_sorting = si.UnitsAggregationSorting(curated_sortings)
    region_sorting.save_to_folder(
        get_region_sorting_path(
            animal_name=animal_name,
            date=date,
            region=region,
            analysis_root=analysis_root,
        ),
        overwrite=True,
    )
    print(f"done with {region}")
    return region_sorting


def consolidate_sorting(
    animal_name: str,
    date: str,
    region_probes: dict[str, list[int]],
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    curation_root: Path = DEFAULT_CURATION_ROOT,
) -> tuple[Any, Any]:
    """Consolidate curated sortings into region-level V1 and CA1 outputs."""
    sorting_v1 = consolidate_region_sorting(
        animal_name=animal_name,
        date=date,
        region="v1",
        probe_indices=region_probes["v1"],
        analysis_root=analysis_root,
        curation_root=curation_root,
    )
    sorting_ca1 = consolidate_region_sorting(
        animal_name=animal_name,
        date=date,
        region="ca1",
        probe_indices=region_probes["ca1"],
        analysis_root=analysis_root,
        curation_root=curation_root,
    )
    log_path = write_run_log(
        analysis_path=get_analysis_path(animal_name, date, analysis_root),
        script_name="v1ca1.spikesorting.consolidate_sorting",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "region_probes": region_probes,
            "analysis_root": analysis_root,
            "curation_root": curation_root,
        },
        outputs={
            "sorting_v1_path": get_region_sorting_path(
                animal_name=animal_name,
                date=date,
                region="v1",
                analysis_root=analysis_root,
            ),
            "sorting_ca1_path": get_region_sorting_path(
                animal_name=animal_name,
                date=date,
                region="ca1",
                analysis_root=analysis_root,
            ),
        },
    )
    print(f"Saved run metadata to {log_path}")
    return sorting_v1, sorting_ca1


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for curated sorting consolidation."""
    parser = argparse.ArgumentParser(
        description="Apply sortingview curations and consolidate sortings"
    )
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
        "--analysis-root",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT,
        help=f"Base directory for analysis output. Default: {DEFAULT_ANALYSIS_ROOT}",
    )
    parser.add_argument(
        "--curation-root",
        type=Path,
        default=DEFAULT_CURATION_ROOT,
        help=(
            "Base directory for the local sorting-curations checkout. "
            f"Default: {DEFAULT_CURATION_ROOT}"
        ),
    )
    parser.add_argument(
        "--v1-probes",
        default=",".join(str(idx) for idx in DEFAULT_REGION_PROBES["v1"]),
        help=(
            "Comma-separated probe indices assigned to V1 for this session. "
            "Default: 0,3"
        ),
    )
    parser.add_argument(
        "--ca1-probes",
        default=",".join(str(idx) for idx in DEFAULT_REGION_PROBES["ca1"]),
        help=(
            "Comma-separated probe indices assigned to CA1 for this session. "
            "Default: 1,2"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint."""
    args = parse_arguments()
    region_probes = {
        "v1": parse_probe_list(args.v1_probes),
        "ca1": parse_probe_list(args.ca1_probes),
    }
    validate_region_probes(region_probes)
    consolidate_sorting(
        animal_name=args.animal_name,
        date=args.date,
        region_probes=region_probes,
        analysis_root=args.analysis_root,
        curation_root=args.curation_root,
    )


if __name__ == "__main__":
    main()
