from __future__ import annotations

"""Apply manual curation JSON files and consolidate curated sortings by region.

This script reads per-shank Mountainsort4 outputs, applies sortingview curation
JSON files from the local `sorting-curations` checkout, saves curated per-shank
sortings, and then aggregates region-specific subsets of those curated
sortings into one `sorting_{region}` output per configured region.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_NWB_ROOT
from v1ca1.spikesorting._region_assignment import (
    get_region_names_from_rules,
    get_session_region_assignment_rules,
    index_rules_by_probe_shank,
)

DEFAULT_ANALYSIS_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_CURATION_ROOT = Path("/home/kyu/repos/sorting-curations/khl02007")
EXCLUDED_CURATION_LABELS = ("noise", "reject")
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


def get_nwb_path(
    animal_name: str,
    date: str,
    nwb_root: Path,
) -> Path:
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


def load_sorting_analyzer(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
) -> Any:
    """Load the saved sorting analyzer for one probe/shank."""
    si = get_spikeinterface()
    sorting_analyzer_path = get_sorting_analyzer_path(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )
    if not sorting_analyzer_path.exists():
        raise FileNotFoundError(f"Sorting analyzer not found: {sorting_analyzer_path}")
    if not hasattr(si, "load_sorting_analyzer"):
        raise RuntimeError(
            "consolidate_sorting.py requires a SpikeInterface version with "
            "`load_sorting_analyzer()` support."
        )
    # Load the analyzer shell only and let SpikeInterface pull extensions lazily.
    return si.load_sorting_analyzer(folder=sorting_analyzer_path, load_extensions=False)


def load_curation_payload(sorting_curations_path: Path) -> dict[str, Any]:
    """Load one raw sortingview curation JSON payload."""
    if not sorting_curations_path.exists():
        raise FileNotFoundError(f"Curation JSON not found: {sorting_curations_path}")
    return json.loads(sorting_curations_path.read_text(encoding="utf-8"))


def apply_sortingview_curation(
    sorting: Any,
    sorting_curations_path: Path,
) -> Any:
    """Apply sortingview merges and label filtering for one probe/shank."""
    si = get_spikeinterface()
    return si.apply_sortingview_curation(
        sorting,
        sorting_curations_path,
        exclude_labels=list(EXCLUDED_CURATION_LABELS),
        include_labels=None,
        skip_merge=False,
    )


def postprocess_curated_sorting(sorting: Any) -> Any:
    """Remove duplicated spikes and redundant units after manual curation."""
    si = get_spikeinterface()
    curated_sorting = si.remove_duplicated_spikes(
        sorting,
        censored_period_ms=0.3,
    )
    curated_sorting = si.remove_redundant_units(
        curated_sorting,
        duplicate_threshold=0.9,
        remove_strategy="max_spikes",
        align=False,
    )
    return curated_sorting


def get_original_unit_spike_counts(sorting: Any) -> dict[int, int]:
    """Return the spike count for each original sorter unit."""
    return {
        int(unit_id): int(len(sorting.get_unit_spike_train(unit_id)))
        for unit_id in sorting.get_unit_ids()
    }


def _decode_group_labels(raw_values: np.ndarray) -> np.ndarray:
    """Convert NWB group labels into a string array."""
    labels = []
    for raw_value in raw_values.tolist():
        if isinstance(raw_value, bytes):
            labels.append(raw_value.decode("utf-8"))
        else:
            labels.append(str(raw_value))
    return np.asarray(labels, dtype=str)


def get_shank_channel_depths_from_arrays(
    electrode_ids: np.ndarray,
    probe_group_labels: np.ndarray,
    probe_shanks: np.ndarray,
    rel_y_values: np.ndarray,
    y_values: np.ndarray,
    probe_idx: int,
    shank_idx: int,
) -> dict[int, float]:
    """Return one shank's global electrode-id to depth mapping.

    Depth selection prefers `rel_y` and falls back to `y` only when `rel_y`
    does not vary within the requested shank.
    """
    electrode_ids = np.asarray(electrode_ids, dtype=int)
    probe_group_labels = np.asarray(probe_group_labels, dtype=str)
    probe_shanks = np.asarray(probe_shanks, dtype=int)
    rel_y_values = np.asarray(rel_y_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    ordered_probe_labels = sorted(np.unique(probe_group_labels).tolist())
    probe_idx_by_label = {
        probe_label: idx for idx, probe_label in enumerate(ordered_probe_labels)
    }
    probe_indices = np.asarray(
        [probe_idx_by_label[label] for label in probe_group_labels],
        dtype=int,
    )

    shank_mask = (probe_indices == int(probe_idx)) & (probe_shanks == int(shank_idx))
    if not np.any(shank_mask):
        raise ValueError(
            f"Could not find electrode metadata for probe {probe_idx} shank {shank_idx}."
        )

    def has_depth_variation(values: np.ndarray) -> bool:
        finite_values = values[np.isfinite(values)]
        return finite_values.size > 0 and np.unique(finite_values).size > 1

    if has_depth_variation(rel_y_values[shank_mask]):
        depth_values = rel_y_values[shank_mask]
    elif has_depth_variation(y_values[shank_mask]):
        depth_values = y_values[shank_mask]
    else:
        raise ValueError(
            "Could not determine within-shank depth coordinates because neither "
            f"`rel_y` nor `y` varies for probe {probe_idx} shank {shank_idx}."
        )

    return {
        int(electrode_id): float(depth_value)
        for electrode_id, depth_value in zip(
            electrode_ids[shank_mask].tolist(),
            depth_values.tolist(),
            strict=True,
        )
    }


def load_shank_channel_depths(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    nwb_root: Path,
) -> dict[int, float]:
    """Load shank-local depth coordinates keyed by global electrode id."""
    import h5py

    nwb_path = get_nwb_path(animal_name, date, nwb_root)
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    with h5py.File(nwb_path, "r") as nwb_file:
        try:
            electrodes = nwb_file["general"]["extracellular_ephys"]["electrodes"]
        except KeyError as exc:
            raise ValueError(
                "Could not locate the NWB electrodes table under "
                "`general/extracellular_ephys/electrodes`."
            ) from exc

        if "id" not in electrodes:
            raise ValueError("The NWB electrodes table does not expose an `id` column.")

        electrode_ids = np.asarray(electrodes["id"][()], dtype=int)
        if "group_name" in electrodes:
            probe_group_labels = _decode_group_labels(electrodes["group_name"][()])
        else:
            probe_group_labels = np.asarray(electrode_ids // 128, dtype=str)

        if "probe_shank" in electrodes:
            probe_shanks = np.asarray(electrodes["probe_shank"][()], dtype=int)
        else:
            probe_shanks = (electrode_ids % 128) // 32

        rel_y_values = (
            np.asarray(electrodes["rel_y"][()], dtype=float)
            if "rel_y" in electrodes
            else np.full(electrode_ids.shape, np.nan, dtype=float)
        )
        y_values = (
            np.asarray(electrodes["y"][()], dtype=float)
            if "y" in electrodes
            else np.full(electrode_ids.shape, np.nan, dtype=float)
        )

    return get_shank_channel_depths_from_arrays(
        electrode_ids=electrode_ids,
        probe_group_labels=probe_group_labels,
        probe_shanks=probe_shanks,
        rel_y_values=rel_y_values,
        y_values=y_values,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
    )


def get_template_extremum_channel_ids(sorting_analyzer: Any) -> dict[int, int]:
    """Return the extremum channel id for each original unit."""
    si = get_spikeinterface()
    get_extremum_channel = getattr(si, "get_template_extremum_channel", None)
    if get_extremum_channel is None:
        from spikeinterface.postprocessing import get_template_extremum_channel
    else:
        get_template_extremum_channel = get_extremum_channel

    extremum_channel_ids = get_template_extremum_channel(
        sorting_analyzer,
        peak_sign="neg",
        outputs="id",
    )
    return {
        int(unit_id): int(channel_id)
        for unit_id, channel_id in extremum_channel_ids.items()
    }


def load_original_unit_extremum_depths(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    analysis_root: Path,
    nwb_root: Path,
) -> dict[int, float]:
    """Return one original sorter's unit extremum depth per unit id."""
    sorting_analyzer = load_sorting_analyzer(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )
    channel_depths = load_shank_channel_depths(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        nwb_root=nwb_root,
    )
    extremum_channel_ids = get_template_extremum_channel_ids(sorting_analyzer)
    missing_channel_ids = sorted(
        channel_id
        for channel_id in extremum_channel_ids.values()
        if channel_id not in channel_depths
    )
    if missing_channel_ids:
        raise ValueError(
            "Could not map extremum channels to NWB depth coordinates for "
            f"probe {probe_idx} shank {shank_idx}. Missing channel ids: "
            f"{missing_channel_ids!r}."
        )
    return {
        int(unit_id): float(channel_depths[int(channel_id)])
        for unit_id, channel_id in extremum_channel_ids.items()
    }


def build_curated_unit_groups(
    original_unit_ids: Iterable[int],
    curation_payload: dict[str, Any],
    excluded_labels: Iterable[str] = EXCLUDED_CURATION_LABELS,
) -> list[list[int]]:
    """Reconstruct curated units from the raw sortingview curation payload.

    This mirrors the local SpikeInterface behavior used by this workflow:
    merges are applied first, then exclude labels remove only surviving
    singleton units whose ids directly match labeled original unit ids.
    """
    original_unit_ids = [int(unit_id) for unit_id in original_unit_ids]
    original_unit_id_set = set(original_unit_ids)
    excluded_labels = {str(label) for label in excluded_labels}

    labels_by_unit = curation_payload.get("labelsByUnit", {})
    excluded_original_unit_ids = {
        int(unit_id)
        for unit_id, labels in labels_by_unit.items()
        if any(str(label) in excluded_labels for label in labels)
    }

    merge_groups_raw = curation_payload.get("mergeGroups", [])
    merge_groups = [
        [int(unit_id) for unit_id in merge_group]
        for merge_group in merge_groups_raw
    ]
    merged_unit_ids: set[int] = set()
    for merge_group in merge_groups:
        if len(merge_group) < 2:
            raise ValueError(
                f"Invalid merge group {merge_group!r}. Merge groups must contain at least two units."
            )
        unknown_unit_ids = sorted(set(merge_group) - original_unit_id_set)
        if unknown_unit_ids:
            raise ValueError(
                f"Curation merge group contains unknown unit ids {unknown_unit_ids!r}."
            )
        overlapping_unit_ids = sorted(set(merge_group) & merged_unit_ids)
        if overlapping_unit_ids:
            raise ValueError(
                "Curation merge groups must be disjoint. Overlapping unit ids: "
                f"{overlapping_unit_ids!r}."
            )
        merged_unit_ids.update(merge_group)

    curated_unit_groups = [[int(unit_id)] for unit_id in original_unit_ids]
    for merge_group in merge_groups:
        merge_group_set = set(merge_group)
        curated_unit_groups = [
            curated_group
            for curated_group in curated_unit_groups
            if merge_group_set.isdisjoint(curated_group)
        ] + [list(merge_group)]

    curated_unit_groups = [
        curated_group
        for curated_group in curated_unit_groups
        if not (
            len(curated_group) == 1
            and curated_group[0] in excluded_original_unit_ids
        )
    ]
    return curated_unit_groups


def compute_curated_unit_extremum_depths(
    curated_unit_groups: list[list[int]],
    original_unit_extremum_depths: dict[int, float],
    original_unit_spike_counts: dict[int, int],
) -> list[float]:
    """Return one weighted extremum depth per curated unit."""
    curated_unit_extremum_depths: list[float] = []
    for curated_unit_group in curated_unit_groups:
        spike_counts = np.asarray(
            [
                original_unit_spike_counts[int(unit_id)]
                for unit_id in curated_unit_group
            ],
            dtype=float,
        )
        extremum_depths = np.asarray(
            [
                original_unit_extremum_depths[int(unit_id)]
                for unit_id in curated_unit_group
            ],
            dtype=float,
        )
        total_spike_count = float(np.sum(spike_counts))
        if total_spike_count <= 0:
            raise ValueError(
                "Cannot compute a weighted extremum depth for curated unit group "
                f"{curated_unit_group!r} because it has no spikes."
            )
        curated_unit_extremum_depths.append(
            float(np.average(extremum_depths, weights=spike_counts))
        )
    return curated_unit_extremum_depths


def assign_regions_to_depths(
    extremum_depths: list[float],
    rule: dict[str, Any],
) -> list[str]:
    """Assign each curated unit depth to one region for one probe/shank rule."""
    mode = str(rule["mode"])
    if mode == "all":
        return [str(rule["region"])] * len(extremum_depths)
    if mode != "threshold_by_extremum_depth":
        raise ValueError(f"Unsupported region assignment mode {mode!r}.")

    reference_depth = float(rule["reference_depth"])
    below_region = str(rule["below_region"])
    above_or_equal_region = str(rule["above_or_equal_region"])
    return [
        below_region if float(depth) < reference_depth else above_or_equal_region
        for depth in extremum_depths
    ]


def stamp_sorting_provenance(
    sorting: Any,
    region_by_unit_id: dict[int, str],
    extremum_depth_by_unit_id: dict[int, float],
    probe_idx: int,
    shank_idx: int,
    sorting_curations_path: Path,
    curation_root: Path,
) -> Any:
    """Attach per-unit provenance needed by downstream consumers."""
    unit_ids = [int(unit_id) for unit_id in sorting.get_unit_ids()]
    curation_json_relpath = sorting_curations_path.relative_to(curation_root).as_posix()
    sorting.set_property(
        "region",
        [str(region_by_unit_id[unit_id]) for unit_id in unit_ids],
    )
    sorting.set_property("probe_idx", [int(probe_idx)] * len(unit_ids))
    sorting.set_property("shank_idx", [int(shank_idx)] * len(unit_ids))
    sorting.set_property(
        "curation_json_relpath",
        [curation_json_relpath] * len(unit_ids),
    )
    sorting.set_property(
        "extremum_depth",
        [float(extremum_depth_by_unit_id[unit_id]) for unit_id in unit_ids],
    )
    return sorting


def split_sorting_by_region(sorting: Any) -> dict[str, Any]:
    """Split one curated shank sorting into region-specific subsets."""
    region_values = np.asarray(sorting.get_property("region")).astype(str)
    unit_ids = list(sorting.get_unit_ids())
    sorting_by_region: dict[str, Any] = {}
    for region in sorted(set(region_values.tolist())):
        region_unit_ids = [
            unit_id
            for unit_id, unit_region in zip(unit_ids, region_values.tolist(), strict=True)
            if unit_region == region
        ]
        sorting_by_region[region] = sorting.select_units(region_unit_ids)
    return sorting_by_region


def curate_probe_shank_sorting(
    animal_name: str,
    date: str,
    probe_idx: int,
    shank_idx: int,
    rule: dict[str, Any],
    analysis_root: Path,
    curation_root: Path,
    nwb_root: Path,
) -> dict[str, Any]:
    """Curate one probe/shank sorting and split the surviving units by region."""
    print(f"probe{probe_idx} shank{shank_idx}")
    sorting_curations_path = get_sorting_curation_path(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        curation_root=curation_root,
    )
    curation_payload = load_curation_payload(sorting_curations_path)
    sorting = load_sorting(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
    )
    print(
        f"{sorting.get_num_units()} units before curation in probe {probe_idx} "
        f"shank {shank_idx}"
    )

    original_unit_ids = [int(unit_id) for unit_id in sorting.get_unit_ids()]
    original_unit_spike_counts = get_original_unit_spike_counts(sorting)
    original_unit_extremum_depths = load_original_unit_extremum_depths(
        animal_name=animal_name,
        date=date,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        analysis_root=analysis_root,
        nwb_root=nwb_root,
    )

    curated_sorting = apply_sortingview_curation(
        sorting=sorting,
        sorting_curations_path=sorting_curations_path,
    )
    curated_unit_groups = build_curated_unit_groups(
        original_unit_ids=original_unit_ids,
        curation_payload=curation_payload,
    )
    curated_unit_ids = [int(unit_id) for unit_id in curated_sorting.get_unit_ids()]
    if len(curated_unit_ids) != len(curated_unit_groups):
        raise ValueError(
            "Reconstructed curated unit groups do not match the number of units "
            f"returned by SpikeInterface for probe {probe_idx} shank {shank_idx}: "
            f"{len(curated_unit_groups)} groups vs {len(curated_unit_ids)} units."
        )

    curated_unit_extremum_depths = compute_curated_unit_extremum_depths(
        curated_unit_groups=curated_unit_groups,
        original_unit_extremum_depths=original_unit_extremum_depths,
        original_unit_spike_counts=original_unit_spike_counts,
    )
    assigned_regions = assign_regions_to_depths(
        extremum_depths=curated_unit_extremum_depths,
        rule=rule,
    )
    region_by_unit_id = {
        unit_id: region
        for unit_id, region in zip(curated_unit_ids, assigned_regions, strict=True)
    }
    extremum_depth_by_unit_id = {
        unit_id: float(extremum_depth)
        for unit_id, extremum_depth in zip(
            curated_unit_ids,
            curated_unit_extremum_depths,
            strict=True,
        )
    }

    curated_sorting = postprocess_curated_sorting(curated_sorting)
    curated_sorting = stamp_sorting_provenance(
        sorting=curated_sorting,
        region_by_unit_id=region_by_unit_id,
        extremum_depth_by_unit_id=extremum_depth_by_unit_id,
        probe_idx=probe_idx,
        shank_idx=shank_idx,
        sorting_curations_path=sorting_curations_path,
        curation_root=curation_root,
    )
    print(
        f"{curated_sorting.get_num_units()} units after postprocessing in probe "
        f"{probe_idx} shank {shank_idx}"
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
    return split_sorting_by_region(curated_sorting)


def save_region_sortings(
    animal_name: str,
    date: str,
    region_sortings: dict[str, list[Any]],
    analysis_root: Path,
) -> dict[str, Path]:
    """Aggregate and save one consolidated sorting per region."""
    si = get_spikeinterface()
    output_paths: dict[str, Path] = {}
    for region, sortings in sorted(region_sortings.items()):
        if not sortings:
            continue
        region_sorting = si.UnitsAggregationSorting(sortings)
        output_path = get_region_sorting_path(
            animal_name=animal_name,
            date=date,
            region=region,
            analysis_root=analysis_root,
        )
        region_sorting.save_to_folder(output_path, overwrite=True)
        print(f"saved {region} sorting to {output_path}")
        output_paths[f"sorting_{region}_path"] = output_path
    return output_paths


def consolidate_sorting(
    animal_name: str,
    date: str,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    curation_root: Path = DEFAULT_CURATION_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
) -> None:
    """Consolidate curated per-shank sortings into region-level outputs."""
    session_rules = get_session_region_assignment_rules(
        animal_name=animal_name,
        date=date,
    )
    rules_by_probe_shank = index_rules_by_probe_shank(session_rules)
    region_names = get_region_names_from_rules(session_rules)
    region_sortings: dict[str, list[Any]] = {
        region: [] for region in region_names
    }

    for probe_idx, shank_idx in sorted(rules_by_probe_shank):
        sorting_by_region = curate_probe_shank_sorting(
            animal_name=animal_name,
            date=date,
            probe_idx=probe_idx,
            shank_idx=shank_idx,
            rule=rules_by_probe_shank[(probe_idx, shank_idx)],
            analysis_root=analysis_root,
            curation_root=curation_root,
            nwb_root=nwb_root,
        )
        for region, sorting in sorting_by_region.items():
            region_sortings.setdefault(region, []).append(sorting)

    output_paths = save_region_sortings(
        animal_name=animal_name,
        date=date,
        region_sortings=region_sortings,
        analysis_root=analysis_root,
    )
    if not output_paths:
        raise FileNotFoundError(
            "No consolidated region sortings were produced for "
            f"{animal_name} {date}."
        )

    log_path = write_run_log(
        analysis_path=get_analysis_path(animal_name, date, analysis_root),
        script_name="v1ca1.spikesorting.consolidate_sorting",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "analysis_root": analysis_root,
            "curation_root": curation_root,
            "nwb_root": nwb_root,
            "region_assignment_rules": session_rules,
        },
        outputs=output_paths,
    )
    print(f"Saved run metadata to {log_path}")


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
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory for source NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CLI entrypoint."""
    args = parse_arguments()
    consolidate_sorting(
        animal_name=args.animal_name,
        date=args.date,
        analysis_root=args.analysis_root,
        curation_root=args.curation_root,
        nwb_root=args.nwb_root,
    )


if __name__ == "__main__":
    main()
