from __future__ import annotations

"""Export an augmented copy of one NWB file from saved analysis outputs.

This workflow replaces epoch bounds from saved ephys intervals and can also
replace the NWB units table with curated spike sorting results from
consolidated SpikeInterface outputs.
"""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    get_analysis_path,
    load_ephys_timestamps_all,
)
from v1ca1.spikesorting.consolidate_sorting import DEFAULT_CURATION_ROOT

if TYPE_CHECKING:
    import pandas as pd
    import pynapple as nap
    import pynwb


DEFAULT_OUTPUT_SUFFIX = "_epoch_bounds_replaced.nwb"
EPHYS_COMPRESSION = "gzip"
EPHYS_COMPRESSION_LEVEL = 4
EPHYS_COMPRESSION_SHUFFLE = True
REGIONS = ("v1", "ca1")
REQUIRED_SORTING_PROPERTIES = (
    "region",
    "probe_idx",
    "shank_idx",
    "curation_json_relpath",
)
SORTING_PROVENANCE_SCRATCH_NAME = "spike_sorting_curation_provenance"
UNITS_TABLE_DESCRIPTION = (
    "Curated spike sorting units exported from consolidated SpikeInterface sortings."
)
_UNITS_COLUMN_DESCRIPTIONS = {
    "region": "Brain region for the curated unit.",
    "probe_idx": "Probe index that produced the curated unit.",
    "shank_idx": "Shank index that produced the curated unit.",
    "sorting_unit_id": "Unit id from the consolidated SpikeInterface sorting.",
    "curation_json_relpath": "Path to the raw sortingview curation JSON relative to the curation root.",
    "is_merged": "Whether the curated unit resulted from a manual merge.",
}
_SPIKEINTERFACE = None


def get_spikeinterface():
    """Import SpikeInterface lazily."""
    global _SPIKEINTERFACE
    if _SPIKEINTERFACE is None:
        import spikeinterface.full as si

        _SPIKEINTERFACE = si
    return _SPIKEINTERFACE


def _extract_interval_dataframe(intervals: "nap.IntervalSet") -> "pd.DataFrame":
    """Return a dataframe-like view of one pynapple IntervalSet."""
    if hasattr(intervals, "as_dataframe"):
        return intervals.as_dataframe()
    if hasattr(intervals, "_metadata"):
        return intervals._metadata.copy()  # type: ignore[attr-defined]
    raise ValueError("Could not read metadata from timestamps_ephys.npz.")


def _extract_epoch_tags(intervals: "nap.IntervalSet") -> list[str]:
    """Extract saved epoch labels from timestamps_ephys.npz."""
    try:
        epoch_info = intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is not None:
        epoch_array = np.asarray(epoch_info)
        if epoch_array.size:
            return [str(epoch) for epoch in epoch_array.tolist()]

    interval_df = _extract_interval_dataframe(intervals)
    if "epoch" in interval_df.columns:
        return [str(epoch) for epoch in interval_df["epoch"].tolist()]

    raise ValueError("timestamps_ephys.npz does not contain saved epoch labels.")


def _extract_interval_bounds(intervals: "nap.IntervalSet") -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned start and end arrays from timestamps_ephys.npz."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "timestamps_ephys.npz has mismatched start/end arrays: "
            f"{starts.shape} vs {ends.shape}."
        )
    return starts, ends


def load_epoch_bounds_npz(analysis_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load saved epoch labels and bounds from timestamps_ephys.npz."""
    import pynapple as nap

    npz_path = analysis_path / "timestamps_ephys.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"timestamps_ephys.npz not found: {npz_path}")

    try:
        intervals = nap.load_file(npz_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {npz_path}.") from exc

    epoch_tags = _extract_epoch_tags(intervals)
    start_times, stop_times = _extract_interval_bounds(intervals)
    if len(epoch_tags) != start_times.size:
        raise ValueError(
            "Mismatch between saved epoch labels and interval bounds in "
            f"{npz_path}."
        )
    if start_times.size == 0:
        raise ValueError(f"timestamps_ephys.npz does not contain any epochs: {npz_path}")
    if np.any(stop_times < start_times):
        raise ValueError(f"timestamps_ephys.npz contains an interval with stop < start: {npz_path}")

    return epoch_tags, start_times, stop_times


def resolve_output_path(
    nwb_path: Path,
    animal_name: str,
    date: str,
    output_path: Path | None,
) -> Path:
    """Return the requested output path, defaulting to a sibling NWB copy."""
    if output_path is not None:
        return output_path
    return nwb_path.with_name(f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}")


def validate_output_path(source_path: Path, output_path: Path, overwrite: bool) -> None:
    """Validate the destination path for the corrected NWB file."""
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Output path must differ from the source NWB path.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. Pass --overwrite to replace it."
        )


def validate_epoch_tags(
    nwb_epoch_tags: list[str],
    replacement_epoch_tags: list[str],
) -> None:
    """Require exact epoch count and order agreement between NWB and npz."""
    if nwb_epoch_tags != replacement_epoch_tags:
        raise ValueError(
            "Epoch labels from timestamps_ephys.npz do not match the NWB epochs table. "
            f"NWB: {nwb_epoch_tags!r}; timestamps_ephys.npz: {replacement_epoch_tags!r}"
        )


def _set_column_values(column: Any, values: np.ndarray, column_name: str) -> None:
    """Replace one DynamicTable column with new values."""
    value_array = np.asarray(values, dtype=float)
    errors: list[str] = []

    if hasattr(column, "transform"):
        try:
            # Detach read-only HDF5-backed columns into memory before export.
            column.transform(lambda _data: value_array.copy())
            return
        except Exception as exc:
            errors.append(f"{type(column).__name__} transform failed: {exc}")

    targets: list[Any] = []
    if hasattr(column, "data"):
        targets.append(column.data)
    targets.append(column)

    for target in targets:
        try:
            target[:] = value_array
            return
        except Exception as exc:
            errors.append(f"{type(target).__name__} slice assignment failed: {exc}")

        try:
            for index, value in enumerate(value_array.tolist()):
                target[index] = float(value)
            return
        except Exception as exc:
            errors.append(f"{type(target).__name__} item assignment failed: {exc}")

    if hasattr(column, "_Data__data"):
        try:
            setattr(column, "_Data__data", value_array.copy())
            return
        except Exception as exc:
            errors.append(f"{type(column).__name__} internal data replacement failed: {exc}")

    raise TypeError(
        f"Could not update epochs column {column_name!r}. "
        + " ".join(errors)
    )


def update_epoch_bounds_in_memory(
    nwbfile: "pynwb.NWBFile",
    replacement_start_times: np.ndarray,
    replacement_stop_times: np.ndarray,
) -> list[dict[str, Any]]:
    """Replace epoch start/stop times on one in-memory NWB object."""
    if nwbfile.epochs is None:
        raise ValueError("NWB file does not contain an epochs table.")

    epoch_tags, old_start_times, old_stop_times = extract_epoch_metadata(nwbfile)
    if not (
        len(epoch_tags)
        == replacement_start_times.size
        == replacement_stop_times.size
    ):
        raise ValueError("Replacement epoch bounds do not match the NWB epoch count.")

    epoch_table = nwbfile.epochs
    _set_column_values(epoch_table["start_time"], replacement_start_times, "start_time")
    _set_column_values(epoch_table["stop_time"], replacement_stop_times, "stop_time")

    return [
        {
            "epoch": epoch,
            "old_start_time_s": float(old_start),
            "old_stop_time_s": float(old_stop),
            "new_start_time_s": float(new_start),
            "new_stop_time_s": float(new_stop),
        }
        for epoch, old_start, old_stop, new_start, new_stop in zip(
            epoch_tags,
            old_start_times,
            old_stop_times,
            replacement_start_times,
            replacement_stop_times,
            strict=True,
        )
    ]


def _get_string_attribute(value: Any) -> str | None:
    """Return one HDF5 string-like attribute as a Python string."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    return None


def _iter_dataset_selections(
    dataset: Any,
    row_block_size: int = 4096,
) -> Iterator[tuple[slice, ...] | tuple[()]]:
    """Yield chunk-like selections for copying one HDF5 dataset."""
    if dataset.shape == ():
        yield ()
        return
    if dataset.chunks is not None:
        yield from dataset.iter_chunks()
        return
    if dataset.shape[0] == 0:
        return

    block_size = min(row_block_size, dataset.shape[0])
    for start in range(0, dataset.shape[0], block_size):
        stop = min(start + block_size, dataset.shape[0])
        selection = [slice(None)] * dataset.ndim
        selection[0] = slice(start, stop)
        yield tuple(selection)


def _copy_dataset_contents(source_dataset: Any, dest_dataset: Any) -> None:
    """Copy one HDF5 dataset without loading the entire array at once."""
    if source_dataset.shape == ():
        dest_dataset[()] = source_dataset[()]
        return

    for selection in _iter_dataset_selections(source_dataset):
        dest_dataset[selection] = source_dataset[selection]


def _copy_dataset_attributes(source_dataset: Any, dest_dataset: Any) -> None:
    """Copy HDF5 dataset attributes to a replacement dataset."""
    for key, value in source_dataset.attrs.items():
        dest_dataset.attrs[key] = value


def _get_available_dataset_name(parent_group: Any, base_name: str) -> str:
    """Return a temporary child name that does not collide within one HDF5 group."""
    if base_name not in parent_group:
        return base_name

    index = 1
    while f"{base_name}_{index}" in parent_group:
        index += 1
    return f"{base_name}_{index}"


def _replace_dataset_with_compression(h5_file: Any, dataset_path: str) -> None:
    """Rewrite one dataset in place with the configured compression settings."""
    dataset = h5_file[dataset_path]
    parent_group = dataset.parent
    dataset_name = dataset.name.rsplit("/", maxsplit=1)[-1]
    temp_name = _get_available_dataset_name(parent_group, f"__tmp_{dataset_name}_compressed__")
    backup_name = _get_available_dataset_name(parent_group, f"__tmp_{dataset_name}_original__")

    temp_dataset = None
    try:
        temp_dataset = parent_group.create_dataset(
            temp_name,
            shape=dataset.shape,
            dtype=dataset.dtype,
            chunks=dataset.chunks if dataset.chunks is not None else True,
            maxshape=dataset.maxshape,
            fillvalue=dataset.fillvalue,
            compression=EPHYS_COMPRESSION,
            compression_opts=EPHYS_COMPRESSION_LEVEL,
            shuffle=EPHYS_COMPRESSION_SHUFFLE,
        )
        _copy_dataset_contents(dataset, temp_dataset)
        _copy_dataset_attributes(dataset, temp_dataset)
    except Exception:
        if temp_dataset is not None and temp_name in parent_group:
            del parent_group[temp_name]
        raise

    parent_group.move(dataset_name, backup_name)
    try:
        parent_group.move(temp_name, dataset_name)
    except Exception:
        if backup_name in parent_group and dataset_name not in parent_group:
            parent_group.move(backup_name, dataset_name)
        if temp_name in parent_group:
            del parent_group[temp_name]
        raise

    del parent_group[backup_name]


def _find_electrical_series_data_paths(h5_file: Any) -> list[str]:
    """Return all ElectricalSeries data dataset paths in one NWB HDF5 file."""
    import h5py

    dataset_paths: list[str] = []

    def visitor(_name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Group):
            return
        neurodata_type = _get_string_attribute(obj.attrs.get("neurodata_type"))
        if neurodata_type != "ElectricalSeries":
            return
        data_dataset = obj.get("data")
        if isinstance(data_dataset, h5py.Dataset):
            dataset_paths.append(data_dataset.name)

    h5_file.visititems(visitor)
    return sorted(dataset_paths)


def update_epoch_bounds_in_hdf5(
    output_path: Path,
    replacement_start_times: np.ndarray,
    replacement_stop_times: np.ndarray,
) -> None:
    """Rewrite the epochs table bounds directly in one staged NWB HDF5 file."""
    import h5py

    with h5py.File(output_path, "r+") as h5_file:
        epochs_group = h5_file["intervals"]["epochs"]
        start_dataset = epochs_group["start_time"]
        stop_dataset = epochs_group["stop_time"]
        if start_dataset.shape != replacement_start_times.shape:
            raise ValueError(
                "Replacement epoch start times do not match the staged NWB epochs table. "
                f"NWB: {start_dataset.shape}; replacement: {replacement_start_times.shape}"
            )
        if stop_dataset.shape != replacement_stop_times.shape:
            raise ValueError(
                "Replacement epoch stop times do not match the staged NWB epochs table. "
                f"NWB: {stop_dataset.shape}; replacement: {replacement_stop_times.shape}"
            )
        start_dataset[:] = np.asarray(replacement_start_times, dtype=float)
        stop_dataset[:] = np.asarray(replacement_stop_times, dtype=float)


def recompress_ephys_data(output_path: Path) -> list[str]:
    """Apply compression to all ElectricalSeries data datasets in one NWB file."""
    import h5py

    with h5py.File(output_path, "r+") as h5_file:
        dataset_paths = _find_electrical_series_data_paths(h5_file)
        for dataset_path in dataset_paths:
            _replace_dataset_with_compression(h5_file, dataset_path)
    return dataset_paths


def _create_temporary_output_path(output_path: Path) -> Path:
    """Return a temporary sibling path for staging the rewritten NWB file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=output_path.suffix,
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        return Path(temp_file.name)


def get_region_sorting_path(analysis_path: Path, region: str) -> Path:
    """Return the consolidated sorting folder for one region."""
    return analysis_path / f"sorting_{region}"


def load_consolidated_region_sortings(analysis_path: Path) -> dict[str, Any]:
    """Load any consolidated region sortings available for NWB export."""
    si = get_spikeinterface()
    region_sortings: dict[str, Any] = {}
    for region in REGIONS:
        sorting_path = get_region_sorting_path(analysis_path, region)
        if sorting_path.exists():
            region_sortings[region] = si.load(sorting_path)
    if not region_sortings:
        raise FileNotFoundError(
            "No consolidated region sorting folders were found under "
            f"{analysis_path}. Run v1ca1.spikesorting.consolidate_sorting first."
        )
    return region_sortings


def validate_sorting_provenance(region_sortings: dict[str, Any]) -> None:
    """Require the consolidated sortings to retain stamped provenance properties."""
    for region, sorting in region_sortings.items():
        property_keys = set(sorting.get_property_keys())
        missing_properties = [
            property_name
            for property_name in REQUIRED_SORTING_PROPERTIES
            if property_name not in property_keys
        ]
        if missing_properties:
            raise ValueError(
                "Consolidated sorting for region "
                f"{region!r} is missing required properties {missing_properties!r}. "
                "Rerun v1ca1.spikesorting.consolidate_sorting after updating it to stamp "
                "sorting provenance properties."
            )

        region_values = np.asarray(sorting.get_property("region")).astype(str)
        if np.any(region_values != region):
            raise ValueError(
                f"Consolidated sorting for region {region!r} has inconsistent 'region' property values."
            )


def _get_electrode_row_indices_by_id(nwbfile: "pynwb.NWBFile") -> dict[int, int]:
    """Return NWB electrode table row indices keyed by the electrode id column."""
    if nwbfile.electrodes is None:
        raise ValueError(
            "NWB file does not contain an electrodes table required for spike sorting export."
        )

    electrode_df = nwbfile.electrodes.to_dataframe()
    electrode_ids = [int(electrode_id) for electrode_id in electrode_df.index.tolist()]
    if len(electrode_ids) != len(set(electrode_ids)):
        raise ValueError("NWB electrode table contains duplicate electrode ids.")
    return {
        electrode_id: row_index
        for row_index, electrode_id in enumerate(electrode_ids)
    }


def _get_shank_electrode_rows(
    electrode_rows_by_id: dict[int, int],
    probe_idx: int,
    shank_idx: int,
) -> np.ndarray:
    """Return NWB electrode row indices for one probe/shank using lab channel ids."""
    electrode_ids = list(range(128 * probe_idx + 32 * shank_idx, 128 * probe_idx + 32 * (shank_idx + 1)))
    missing_ids = [electrode_id for electrode_id in electrode_ids if electrode_id not in electrode_rows_by_id]
    if missing_ids:
        raise ValueError(
            "NWB electrode table is missing the expected channel ids for "
            f"probe {probe_idx} shank {shank_idx}: missing {missing_ids!r}."
        )
    return np.asarray([electrode_rows_by_id[electrode_id] for electrode_id in electrode_ids], dtype=int)


def _create_units_table(nwbfile: "pynwb.NWBFile") -> None:
    """Create a fresh NWB units table for curated export."""
    from pynwb.misc import Units

    if nwbfile.electrodes is None:
        raise ValueError(
            "NWB file does not contain an electrodes table required for spike sorting export."
        )
    if nwbfile.units is not None:
        raise ValueError(
            "NWB file already contains a units table. "
            "Spike sorting export currently requires a source NWB without existing units."
        )

    nwbfile.units = Units(
        name="units",
        description=UNITS_TABLE_DESCRIPTION,
        electrode_table=nwbfile.electrodes,
    )
    for column_name, description in _UNITS_COLUMN_DESCRIPTIONS.items():
        nwbfile.add_unit_column(column_name, description)


def _load_curation_provenance_record(
    curation_root: Path,
    region: str,
    probe_idx: int,
    shank_idx: int,
    curation_json_relpath: str,
) -> dict[str, Any]:
    """Load one raw curation JSON payload for storage in NWB scratch."""
    curation_json_path = curation_root / curation_json_relpath
    if not curation_json_path.exists():
        raise FileNotFoundError(
            f"Curation JSON not found for exported units: {curation_json_path}"
        )

    curation_payload = json.loads(curation_json_path.read_text(encoding="utf-8"))
    return {
        "region": region,
        "probe_idx": int(probe_idx),
        "shank_idx": int(shank_idx),
        "curation_json_relpath": curation_json_relpath,
        "labels_by_unit_json": json.dumps(curation_payload.get("labelsByUnit", {}), sort_keys=True),
        "merge_groups_json": json.dumps(curation_payload.get("mergeGroups", [])),
    }


def export_curated_sorting_to_nwb(
    nwbfile: "pynwb.NWBFile",
    analysis_path: Path,
    curation_root: Path,
    replacement_start_times: np.ndarray,
    replacement_stop_times: np.ndarray,
) -> dict[str, Any]:
    """Replace the NWB units table with curated sorting results and provenance."""
    import pandas as pd

    region_sortings = load_consolidated_region_sortings(analysis_path)
    validate_sorting_provenance(region_sortings)
    timestamps_ephys_all, timestamps_source = load_ephys_timestamps_all(analysis_path)
    obs_intervals = np.column_stack(
        (
            np.asarray(replacement_start_times, dtype=float),
            np.asarray(replacement_stop_times, dtype=float),
        )
    )
    electrode_rows_by_id = _get_electrode_row_indices_by_id(nwbfile)

    if SORTING_PROVENANCE_SCRATCH_NAME in nwbfile.scratch:
        del nwbfile.scratch[SORTING_PROVENANCE_SCRATCH_NAME]

    _create_units_table(nwbfile)

    provenance_records: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    sorting_source_paths: dict[str, Path] = {}
    sorting_unit_counts: dict[str, int] = {}
    next_unit_id = 0

    for region in REGIONS:
        sorting = region_sortings.get(region)
        if sorting is None:
            continue

        sorting_source_paths[region] = get_region_sorting_path(analysis_path, region)
        property_keys = set(sorting.get_property_keys())
        sorting_unit_ids = list(sorting.get_unit_ids())
        unit_index_by_id = {int(unit_id): index for index, unit_id in enumerate(sorting_unit_ids)}
        region_values = np.asarray(sorting.get_property("region")).astype(str)
        probe_values = np.asarray(sorting.get_property("probe_idx"))
        shank_values = np.asarray(sorting.get_property("shank_idx"))
        curation_relpaths = np.asarray(sorting.get_property("curation_json_relpath")).astype(str)
        is_merged_values = (
            np.asarray(sorting.get_property("is_merged"))
            if "is_merged" in property_keys
            else None
        )

        exported_region_units = 0
        for sorting_unit_id in sorted((int(unit_id) for unit_id in sorting_unit_ids)):
            unit_index = unit_index_by_id[sorting_unit_id]
            region_value = str(region_values[unit_index])
            probe_idx = int(probe_values[unit_index])
            shank_idx = int(shank_values[unit_index])
            curation_json_relpath = str(curation_relpaths[unit_index])
            is_merged = bool(is_merged_values[unit_index]) if is_merged_values is not None else False

            if region_value != region:
                raise ValueError(
                    f"Consolidated sorting for region {region!r} has unit rows tagged as {region_value!r}."
                )
            if not curation_json_relpath:
                raise ValueError(
                    f"Consolidated sorting for region {region!r} has an empty curation_json_relpath value."
                )

            spike_sample_indices = np.asarray(
                sorting.get_unit_spike_train(sorting_unit_id),
                dtype=int,
            ).ravel()
            if spike_sample_indices.size:
                if np.any(spike_sample_indices < 0):
                    raise ValueError(
                        f"Sorting unit {sorting_unit_id} contains negative spike sample indices."
                    )
                if np.any(spike_sample_indices >= timestamps_ephys_all.size):
                    raise ValueError(
                        f"Sorting unit {sorting_unit_id} contains spike samples beyond timestamps_ephys_all."
                    )
            spike_times_s = np.asarray(timestamps_ephys_all[spike_sample_indices], dtype=float)
            electrodes = _get_shank_electrode_rows(
                electrode_rows_by_id=electrode_rows_by_id,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
            )

            nwbfile.add_unit(
                id=next_unit_id,
                spike_times=spike_times_s,
                obs_intervals=obs_intervals,
                electrodes=electrodes,
                region=region_value,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                sorting_unit_id=int(sorting_unit_id),
                curation_json_relpath=curation_json_relpath,
                is_merged=is_merged,
            )
            next_unit_id += 1
            exported_region_units += 1

            provenance_key = (region_value, probe_idx, shank_idx, curation_json_relpath)
            if provenance_key not in provenance_records:
                provenance_records[provenance_key] = _load_curation_provenance_record(
                    curation_root=curation_root,
                    region=region_value,
                    probe_idx=probe_idx,
                    shank_idx=shank_idx,
                    curation_json_relpath=curation_json_relpath,
                )

        sorting_unit_counts[region] = exported_region_units

    provenance_dataframe = pd.DataFrame(
        [
            provenance_records[key]
            for key in sorted(provenance_records)
        ],
        columns=[
            "region",
            "probe_idx",
            "shank_idx",
            "curation_json_relpath",
            "labels_by_unit_json",
            "merge_groups_json",
        ],
    )
    nwbfile.add_scratch(
        provenance_dataframe,
        name=SORTING_PROVENANCE_SCRATCH_NAME,
        description="Raw sortingview curation payloads used to produce the curated NWB units table.",
    )
    nwbfile.set_modified()
    return {
        "sorting_source_paths": sorting_source_paths,
        "sorting_unit_counts": sorting_unit_counts,
        "timestamps_ephys_all_source": timestamps_source,
        "sorting_curation_provenance_scratch_name": SORTING_PROVENANCE_SCRATCH_NAME,
        "units_table_row_count": int(next_unit_id),
    }


def write_nwb_copy(
    read_io: Any,
    nwbfile: "pynwb.NWBFile",
    output_path: Path,
) -> None:
    """Write the corrected NWB object to a new file."""
    import pynwb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pynwb.NWBHDF5IO(output_path, "w") as write_io:
        export = getattr(write_io, "export", None)
        if callable(export):
            export(src_io=read_io, nwbfile=nwbfile)
        else:
            write_io.write(nwbfile)


def export_augmented_nwb(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    output_path: Path | None = None,
    overwrite: bool = False,
    add_sorting: bool = False,
    curation_root: Path = DEFAULT_CURATION_ROOT,
) -> Path:
    """Write a new NWB file with epoch bounds replaced from timestamps_ephys.npz."""
    import pynwb

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    resolved_output_path = resolve_output_path(
        nwb_path=nwb_path,
        animal_name=animal_name,
        date=date,
        output_path=output_path,
    )

    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    replacement_epoch_tags, replacement_start_times, replacement_stop_times = (
        load_epoch_bounds_npz(analysis_path)
    )
    validate_output_path(source_path=nwb_path, output_path=resolved_output_path, overwrite=overwrite)

    print(f"Processing {animal_name} {date}.")
    temp_output_path = _create_temporary_output_path(resolved_output_path)
    compressed_dataset_paths: list[str] = []
    sorting_outputs: dict[str, Any] = {
        "sorting_export_enabled": bool(add_sorting),
    }
    with pynwb.NWBHDF5IO(nwb_path, "r") as read_io:
        nwbfile = read_io.read()
        nwb_epoch_tags, _nwb_start_times, _nwb_stop_times = extract_epoch_metadata(nwbfile)
        validate_epoch_tags(
            nwb_epoch_tags=nwb_epoch_tags,
            replacement_epoch_tags=replacement_epoch_tags,
        )
        if not (
            len(nwb_epoch_tags)
            == replacement_start_times.size
            == replacement_stop_times.size
        ):
            raise ValueError("Replacement epoch bounds do not match the NWB epoch count.")
        epoch_replacements = [
            {
                "epoch": epoch,
                "old_start_time_s": float(old_start),
                "old_stop_time_s": float(old_stop),
                "new_start_time_s": float(new_start),
                "new_stop_time_s": float(new_stop),
            }
            for epoch, old_start, old_stop, new_start, new_stop in zip(
                nwb_epoch_tags,
                _nwb_start_times,
                _nwb_stop_times,
                replacement_start_times,
                replacement_stop_times,
                strict=True,
            )
        ]
        if add_sorting:
            sorting_outputs.update(
                export_curated_sorting_to_nwb(
                    nwbfile=nwbfile,
                    analysis_path=analysis_path,
                    curation_root=curation_root,
                    replacement_start_times=replacement_start_times,
                    replacement_stop_times=replacement_stop_times,
                )
            )
        try:
            write_nwb_copy(
                read_io=read_io,
                nwbfile=nwbfile,
                output_path=temp_output_path,
            )
            update_epoch_bounds_in_hdf5(
                output_path=temp_output_path,
                replacement_start_times=replacement_start_times,
                replacement_stop_times=replacement_stop_times,
            )
            compressed_dataset_paths = recompress_ephys_data(temp_output_path)
            temp_output_path.replace(resolved_output_path)
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()

    outputs = {
        "source_nwb_path": nwb_path,
        "output_nwb_path": resolved_output_path,
        "epoch_column_names": list(getattr(nwbfile.epochs, "colnames", [])),
        "epoch_replacements": epoch_replacements,
        "compressed_ephys_data_paths": compressed_dataset_paths,
        "ephys_data_compression": {
            "compression": EPHYS_COMPRESSION,
            "compression_opts": EPHYS_COMPRESSION_LEVEL,
            "shuffle": EPHYS_COMPRESSION_SHUFFLE,
        },
        **sorting_outputs,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.nwb.export_augmented_nwb",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "output_path": resolved_output_path,
            "overwrite": overwrite,
            "add_sorting": add_sorting,
            "curation_root": curation_root,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")
    return resolved_output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the augmented NWB export CLI."""
    parser = argparse.ArgumentParser(
        description="Export an augmented NWB copy from timestamps_ephys.npz and optional curated sorting"
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
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path for the augmented NWB copy. Default: sibling *_epoch_bounds_replaced.nwb",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--add-sorting",
        action="store_true",
        help="Replace the NWB units table with curated consolidated spike sorting output.",
    )
    parser.add_argument(
        "--curation-root",
        type=Path,
        default=DEFAULT_CURATION_ROOT,
        help=f"Base directory for the local sorting-curations checkout. Default: {DEFAULT_CURATION_ROOT}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the augmented NWB export CLI."""
    args = parse_arguments()
    export_augmented_nwb(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        output_path=args.output_path,
        overwrite=args.overwrite,
        add_sorting=args.add_sorting,
        curation_root=args.curation_root,
    )


if __name__ == "__main__":
    main()
