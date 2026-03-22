from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.nwb.export_augmented_nwb import (
    DEFAULT_OUTPUT_SUFFIX,
    SORTING_PROVENANCE_SCRATCH_NAME,
    export_augmented_nwb,
)


ACQUISITION_DATASET_PATH = "/acquisition/ElectricalSeries/data"
ACQUISITION_TIMESTAMPS_PATH = "/acquisition/ElectricalSeries/timestamps"
LFP_DATASET_PATH = "/processing/ecephys/LFP/LFPSeries/data"


def _get_shank_channel_ids(probe_idx: int, shank_idx: int) -> list[int]:
    """Return the lab channel ids for one probe/shank."""
    start = 128 * probe_idx + 32 * shank_idx
    return list(range(start, start + 32))


def _add_test_electrodes(nwbfile: Any, electrode_ids: list[int]) -> None:
    """Add one NWB electrodes table with the requested id values."""
    device = nwbfile.create_device("test-device")
    electrode_group = nwbfile.create_electrode_group(
        name="test-electrode-group",
        description="test electrode group",
        location="test location",
        device=device,
    )
    for electrode_id in electrode_ids:
        nwbfile.add_electrode(
            id=int(electrode_id),
            x=0.0,
            y=0.0,
            z=0.0,
            imp=np.nan,
            location="test location",
            filtering="none",
            group=electrode_group,
        )


def _add_test_ephys_series(nwbfile: Any) -> None:
    """Add acquisition and processed ElectricalSeries objects to one test NWB file."""
    pynwb = pytest.importorskip("pynwb")

    electrode_region = nwbfile.create_electrode_table_region([0], "test electrode")
    acquisition_series = pynwb.ecephys.ElectricalSeries(
        name="ElectricalSeries",
        data=np.arange(12, dtype=np.int16).reshape(12, 1),
        electrodes=electrode_region,
        timestamps=np.linspace(0.0, 0.11, num=12),
        filtering="none",
    )
    nwbfile.add_acquisition(acquisition_series)

    lfp = pynwb.ecephys.LFP(name="LFP")
    processing_module = nwbfile.create_processing_module("ecephys", "test ecephys module")
    processing_module.add(lfp)
    lfp.add_electrical_series(
        pynwb.ecephys.ElectricalSeries(
            name="LFPSeries",
            data=np.arange(20, dtype=np.int16).reshape(20, 1),
            electrodes=electrode_region,
            rate=1000.0,
            filtering="none",
        )
    )


def _add_source_unit(nwbfile: Any) -> None:
    """Add one pre-existing source unit so the export path can replace it."""
    from pynwb.misc import Units

    nwbfile.units = Units(
        name="units",
        description="source units",
        electrode_table=nwbfile.electrodes,
    )
    nwbfile.add_unit(
        id=99,
        spike_times=np.asarray([0.05], dtype=float),
        obs_intervals=np.asarray([[0.0, 1.0]], dtype=float),
        electrodes=np.asarray([0], dtype=int),
    )


def _write_test_nwb(
    nwb_path: Path,
    epoch_tags: list[str],
    epoch_bounds: list[tuple[float, float]],
    include_ephys: bool = False,
    electrode_ids: list[int] | None = None,
    include_source_units: bool = False,
) -> None:
    pynwb = pytest.importorskip("pynwb")

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    for epoch_tag, (start_time, stop_time) in zip(epoch_tags, epoch_bounds, strict=True):
        nwbfile.add_epoch(
            start_time=float(start_time),
            stop_time=float(stop_time),
            tags=[str(epoch_tag)],
        )

    if electrode_ids is not None:
        _add_test_electrodes(nwbfile, electrode_ids)
    if include_ephys:
        _add_test_ephys_series(nwbfile)
    if include_source_units:
        _add_source_unit(nwbfile)

    with pynwb.NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)


def _write_timestamps_ephys_npz(
    analysis_path: Path,
    epoch_tags: list[str],
    epoch_bounds: list[tuple[float, float]],
) -> None:
    nap = pytest.importorskip("pynapple")

    starts = np.asarray([bounds[0] for bounds in epoch_bounds], dtype=float)
    stops = np.asarray([bounds[1] for bounds in epoch_bounds], dtype=float)
    intervals = nap.IntervalSet(start=starts, end=stops, time_units="s")
    intervals.set_info(epoch=list(epoch_tags))
    intervals.save(analysis_path / "timestamps_ephys.npz")


def _write_timestamps_ephys_all_npz(analysis_path: Path, timestamps_s: np.ndarray) -> None:
    """Save concatenated ephys timestamps in the canonical pynapple format."""
    nap = pytest.importorskip("pynapple")

    nap.Ts(t=np.asarray(timestamps_s, dtype=float), time_units="s").save(
        analysis_path / "timestamps_ephys_all.npz"
    )


def _write_sorting_folder(
    sorting_path: Path,
    unit_spikes_by_id: dict[int, list[int]],
    unit_metadata: dict[int, dict[str, Any]],
) -> None:
    """Create one saved SpikeInterface sorting folder with per-unit properties."""
    si = pytest.importorskip("spikeinterface.full")

    sorting = si.NumpySorting.from_unit_dict(
        {int(unit_id): np.asarray(spikes, dtype=int) for unit_id, spikes in unit_spikes_by_id.items()},
        sampling_frequency=1000.0,
    )
    ordered_unit_ids = [int(unit_id) for unit_id in sorting.get_unit_ids()]
    property_names = sorted(
        {
            property_name
            for metadata in unit_metadata.values()
            for property_name in metadata
        }
    )
    for property_name in property_names:
        sorting.set_property(
            property_name,
            [unit_metadata[unit_id][property_name] for unit_id in ordered_unit_ids],
        )
    sorting.save_to_folder(sorting_path, overwrite=True)


def _write_curation_json(
    curation_root: Path,
    curation_json_relpath: str,
    labels_by_unit: dict[str, list[str]],
    merge_groups: list[list[int]],
) -> None:
    """Write one raw sortingview curation payload under the requested root."""
    curation_json_path = curation_root / curation_json_relpath
    curation_json_path.parent.mkdir(parents=True, exist_ok=True)
    curation_json_path.write_text(
        json.dumps(
            {
                "labelsByUnit": labels_by_unit,
                "mergeGroups": merge_groups,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _read_epoch_bounds(nwb_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    pynwb = pytest.importorskip("pynwb")

    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        epoch_tags, start_times, stop_times = extract_epoch_metadata(nwbfile)
    return epoch_tags, start_times, stop_times


def _read_dataset_storage(nwb_path: Path, dataset_path: str) -> dict[str, Any]:
    h5py = pytest.importorskip("h5py")

    with h5py.File(nwb_path, "r") as h5_file:
        dataset = h5_file[dataset_path]
        return {
            "compression": dataset.compression,
            "compression_opts": dataset.compression_opts,
            "shuffle": dataset.shuffle,
            "data": dataset[()],
        }


def _read_units_table(nwb_path: Path) -> Any:
    """Return the NWB units table as a DataFrame, or None if absent."""
    pynwb = pytest.importorskip("pynwb")

    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        if nwbfile.units is None:
            return None
        return nwbfile.units.to_dataframe()


def _read_units_export(nwb_path: Path) -> tuple[Any, Any]:
    """Return the exported NWB units and scratch provenance tables as DataFrames."""
    pynwb = pytest.importorskip("pynwb")

    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        provenance_df = nwbfile.scratch[SORTING_PROVENANCE_SCRATCH_NAME].to_dataframe()
    return units_df, provenance_df


def test_export_augmented_nwb_writes_new_nwb_file(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    source_epoch_tags = ["01_s1", "02_r1"]
    source_epoch_bounds = [(0.0, 1.0), (2.0, 3.0)]
    replacement_epoch_bounds = [(0.25, 0.95), (2.1, 2.85)]
    _write_test_nwb(
        nwb_path,
        source_epoch_tags,
        source_epoch_bounds,
        include_ephys=True,
        electrode_ids=[0],
    )
    _write_timestamps_ephys_npz(analysis_path, source_epoch_tags, replacement_epoch_bounds)

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
    )

    assert output_path == nwb_root / f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}"
    assert output_path.exists()

    source_tags, source_starts, source_stops = _read_epoch_bounds(nwb_path)
    output_tags, output_starts, output_stops = _read_epoch_bounds(output_path)

    assert source_tags == source_epoch_tags
    assert np.allclose(source_starts, [0.0, 2.0])
    assert np.allclose(source_stops, [1.0, 3.0])
    assert output_tags == source_epoch_tags
    assert np.allclose(output_starts, [0.25, 2.1])
    assert np.allclose(output_stops, [0.95, 2.85])

    source_acquisition_data = _read_dataset_storage(nwb_path, ACQUISITION_DATASET_PATH)
    output_acquisition_data = _read_dataset_storage(output_path, ACQUISITION_DATASET_PATH)
    source_lfp_data = _read_dataset_storage(nwb_path, LFP_DATASET_PATH)
    output_lfp_data = _read_dataset_storage(output_path, LFP_DATASET_PATH)
    source_timestamps = _read_dataset_storage(nwb_path, ACQUISITION_TIMESTAMPS_PATH)
    output_timestamps = _read_dataset_storage(output_path, ACQUISITION_TIMESTAMPS_PATH)

    assert source_acquisition_data["compression"] is None
    assert output_acquisition_data["compression"] == "gzip"
    assert output_acquisition_data["compression_opts"] == 4
    assert output_acquisition_data["shuffle"] is True
    assert np.array_equal(output_acquisition_data["data"], source_acquisition_data["data"])

    assert source_lfp_data["compression"] is None
    assert output_lfp_data["compression"] == "gzip"
    assert output_lfp_data["compression_opts"] == 4
    assert output_lfp_data["shuffle"] is True
    assert np.array_equal(output_lfp_data["data"], source_lfp_data["data"])

    assert source_timestamps["compression"] is None
    assert output_timestamps["compression"] is None
    assert np.array_equal(output_timestamps["data"], source_timestamps["data"])


def test_export_augmented_nwb_can_export_curated_sorting(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    curation_root = tmp_path / "curations"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)
    curation_root.mkdir(parents=True)

    source_epoch_tags = ["01_s1", "02_r1"]
    source_epoch_bounds = [(0.0, 1.0), (2.0, 3.0)]
    replacement_epoch_bounds = [(0.25, 0.95), (2.1, 2.85)]
    electrode_ids = _get_shank_channel_ids(0, 0) + _get_shank_channel_ids(1, 0)
    _write_test_nwb(
        nwb_path,
        source_epoch_tags,
        source_epoch_bounds,
        include_ephys=True,
        electrode_ids=electrode_ids,
    )
    _write_timestamps_ephys_npz(analysis_path, source_epoch_tags, replacement_epoch_bounds)

    timestamps_ephys_all = np.linspace(0.0, 0.299, num=300)
    _write_timestamps_ephys_all_npz(analysis_path, timestamps_ephys_all)

    v1_curation_relpath = f"{animal_name}/{date}/probe0/shank0/ms4/curation.json"
    ca1_curation_relpath = f"{animal_name}/{date}/probe1/shank0/ms4/curation.json"
    _write_sorting_folder(
        analysis_path / "sorting_v1",
        unit_spikes_by_id={11: [10, 30, 50]},
        unit_metadata={
            11: {
                "region": "v1",
                "probe_idx": 0,
                "shank_idx": 0,
                "curation_json_relpath": v1_curation_relpath,
                "is_merged": False,
            }
        },
    )
    _write_sorting_folder(
        analysis_path / "sorting_ca1",
        unit_spikes_by_id={21: [20, 40, 80]},
        unit_metadata={
            21: {
                "region": "ca1",
                "probe_idx": 1,
                "shank_idx": 0,
                "curation_json_relpath": ca1_curation_relpath,
                "is_merged": True,
            }
        },
    )
    _write_curation_json(
        curation_root,
        v1_curation_relpath,
        labels_by_unit={"11": ["good"]},
        merge_groups=[],
    )
    _write_curation_json(
        curation_root,
        ca1_curation_relpath,
        labels_by_unit={"21": ["mua"]},
        merge_groups=[[1, 2]],
    )

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_sorting=True,
        curation_root=curation_root,
    )

    source_tags, source_starts, source_stops = _read_epoch_bounds(nwb_path)
    output_tags, output_starts, output_stops = _read_epoch_bounds(output_path)
    source_acquisition_data = _read_dataset_storage(nwb_path, ACQUISITION_DATASET_PATH)
    output_acquisition_data = _read_dataset_storage(output_path, ACQUISITION_DATASET_PATH)
    source_lfp_data = _read_dataset_storage(nwb_path, LFP_DATASET_PATH)
    output_lfp_data = _read_dataset_storage(output_path, LFP_DATASET_PATH)
    output_timestamps = _read_dataset_storage(output_path, ACQUISITION_TIMESTAMPS_PATH)
    units_df, provenance_df = _read_units_export(output_path)
    source_units_df = _read_units_table(nwb_path)

    assert source_tags == source_epoch_tags
    assert np.allclose(source_starts, [0.0, 2.0])
    assert np.allclose(source_stops, [1.0, 3.0])
    assert output_tags == source_epoch_tags
    assert np.allclose(output_starts, [0.25, 2.1])
    assert np.allclose(output_stops, [0.95, 2.85])

    assert source_acquisition_data["compression"] is None
    assert output_acquisition_data["compression"] == "gzip"
    assert output_acquisition_data["compression_opts"] == 4
    assert output_acquisition_data["shuffle"] is True
    assert np.array_equal(output_acquisition_data["data"], source_acquisition_data["data"])

    assert source_lfp_data["compression"] is None
    assert output_lfp_data["compression"] == "gzip"
    assert output_lfp_data["compression_opts"] == 4
    assert output_lfp_data["shuffle"] is True
    assert np.array_equal(output_lfp_data["data"], source_lfp_data["data"])

    assert output_timestamps["compression"] is None
    assert source_units_df is None

    assert len(units_df) == 2
    assert units_df.index.tolist() == [0, 1]
    assert {
        "region",
        "probe_idx",
        "shank_idx",
        "sorting_unit_id",
        "curation_json_relpath",
        "is_merged",
        "spike_times",
        "obs_intervals",
        "electrodes",
    }.issubset(units_df.columns)

    units_by_sorting_id = units_df.set_index("sorting_unit_id", drop=False)
    expected_obs_intervals = np.asarray(replacement_epoch_bounds, dtype=float)

    assert np.allclose(units_by_sorting_id.loc[11, "spike_times"], timestamps_ephys_all[[10, 30, 50]])
    assert np.allclose(units_by_sorting_id.loc[21, "spike_times"], timestamps_ephys_all[[20, 40, 80]])
    assert np.allclose(units_by_sorting_id.loc[11, "obs_intervals"], expected_obs_intervals)
    assert np.allclose(units_by_sorting_id.loc[21, "obs_intervals"], expected_obs_intervals)
    assert units_by_sorting_id.loc[11, "region"] == "v1"
    assert units_by_sorting_id.loc[21, "region"] == "ca1"
    assert units_by_sorting_id.loc[11, "probe_idx"] == 0
    assert units_by_sorting_id.loc[21, "probe_idx"] == 1
    assert units_by_sorting_id.loc[11, "shank_idx"] == 0
    assert units_by_sorting_id.loc[21, "shank_idx"] == 0
    assert units_by_sorting_id.loc[11, "curation_json_relpath"] == v1_curation_relpath
    assert units_by_sorting_id.loc[21, "curation_json_relpath"] == ca1_curation_relpath
    assert bool(units_by_sorting_id.loc[11, "is_merged"]) is False
    assert bool(units_by_sorting_id.loc[21, "is_merged"]) is True
    assert units_by_sorting_id.loc[11, "electrodes"].index.tolist() == _get_shank_channel_ids(0, 0)
    assert units_by_sorting_id.loc[21, "electrodes"].index.tolist() == _get_shank_channel_ids(1, 0)

    assert len(provenance_df) == 2
    assert set(provenance_df["curation_json_relpath"].tolist()) == {v1_curation_relpath, ca1_curation_relpath}
    provenance_by_relpath = provenance_df.set_index("curation_json_relpath", drop=False)
    assert json.loads(provenance_by_relpath.loc[v1_curation_relpath, "labels_by_unit_json"]) == {"11": ["good"]}
    assert json.loads(provenance_by_relpath.loc[ca1_curation_relpath, "merge_groups_json"]) == [[1, 2]]


def test_export_augmented_nwb_rejects_mismatched_epoch_labels(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1", "02_r1"], [(0.0, 1.0), (2.0, 3.0)])
    _write_timestamps_ephys_npz(
        analysis_path,
        ["02_r1", "01_s1"],
        [(2.1, 2.9), (0.1, 0.9)],
    )

    with pytest.raises(ValueError, match="do not match the NWB epochs table"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_export_augmented_nwb_requires_timestamps_ephys_npz(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])

    with pytest.raises(FileNotFoundError, match="timestamps_ephys.npz not found"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_export_augmented_nwb_rejects_unreadable_timestamps_npz(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    (analysis_path / "timestamps_ephys.npz").write_text("not a valid npz", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to load"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_export_augmented_nwb_requires_overwrite_for_existing_output(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    output_path = nwb_root / f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    output_path.write_bytes(b"existing output")

    with pytest.raises(FileExistsError, match="Output path already exists"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
        )


def test_export_augmented_nwb_requires_consolidated_sorting_when_add_sorting_enabled(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])

    with pytest.raises(FileNotFoundError, match="No consolidated region sorting folders were found"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_sorting=True,
            curation_root=tmp_path / "curations",
        )


def test_export_augmented_nwb_requires_stamped_sorting_provenance(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    _write_sorting_folder(
        analysis_path / "sorting_v1",
        unit_spikes_by_id={11: [1, 3, 5]},
        unit_metadata={
            11: {
                "region": "v1",
                "probe_idx": 0,
            }
        },
    )

    with pytest.raises(ValueError, match="missing required properties"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_sorting=True,
            curation_root=tmp_path / "curations",
        )


def test_export_augmented_nwb_requires_curation_json_for_exported_units(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(
        nwb_path,
        ["01_s1"],
        [(0.0, 1.0)],
        electrode_ids=_get_shank_channel_ids(0, 0),
    )
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    _write_timestamps_ephys_all_npz(analysis_path, np.linspace(0.0, 0.099, num=100))
    _write_sorting_folder(
        analysis_path / "sorting_v1",
        unit_spikes_by_id={11: [1, 3, 5]},
        unit_metadata={
            11: {
                "region": "v1",
                "probe_idx": 0,
                "shank_idx": 0,
                "curation_json_relpath": f"{animal_name}/{date}/probe0/shank0/ms4/curation.json",
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="Curation JSON not found for exported units"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_sorting=True,
            curation_root=tmp_path / "curations",
        )


def test_export_augmented_nwb_rejects_existing_units_table_when_add_sorting_enabled(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    curation_root = tmp_path / "curations"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)
    curation_root.mkdir(parents=True)

    curation_json_relpath = f"{animal_name}/{date}/probe0/shank0/ms4/curation.json"
    _write_test_nwb(
        nwb_path,
        ["01_s1"],
        [(0.0, 1.0)],
        electrode_ids=_get_shank_channel_ids(0, 0),
        include_source_units=True,
    )
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    _write_timestamps_ephys_all_npz(analysis_path, np.linspace(0.0, 0.099, num=100))
    _write_sorting_folder(
        analysis_path / "sorting_v1",
        unit_spikes_by_id={11: [1, 3, 5]},
        unit_metadata={
            11: {
                "region": "v1",
                "probe_idx": 0,
                "shank_idx": 0,
                "curation_json_relpath": curation_json_relpath,
            }
        },
    )
    _write_curation_json(
        curation_root,
        curation_json_relpath,
        labels_by_unit={"11": ["good"]},
        merge_groups=[],
    )

    with pytest.raises(ValueError, match="already contains a units table"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_sorting=True,
            curation_root=curation_root,
        )


def test_export_augmented_nwb_requires_matching_shank_electrode_ids(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    curation_root = tmp_path / "curations"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)
    curation_root.mkdir(parents=True)

    curation_json_relpath = f"{animal_name}/{date}/probe0/shank0/ms4/curation.json"
    _write_test_nwb(
        nwb_path,
        ["01_s1"],
        [(0.0, 1.0)],
        electrode_ids=list(range(1000, 1032)),
    )
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    _write_timestamps_ephys_all_npz(analysis_path, np.linspace(0.0, 0.099, num=100))
    _write_sorting_folder(
        analysis_path / "sorting_v1",
        unit_spikes_by_id={11: [1, 3, 5]},
        unit_metadata={
            11: {
                "region": "v1",
                "probe_idx": 0,
                "shank_idx": 0,
                "curation_json_relpath": curation_json_relpath,
            }
        },
    )
    _write_curation_json(
        curation_root,
        curation_json_relpath,
        labels_by_unit={"11": ["good"]},
        merge_groups=[],
    )

    with pytest.raises(ValueError, match="missing the expected channel ids"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_sorting=True,
            curation_root=curation_root,
        )
