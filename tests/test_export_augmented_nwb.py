from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.helper.session import DEFAULT_POSITION_OFFSET, TRAJECTORY_TYPES
from v1ca1.helper.wtrack import (
    DEFAULT_WTRACK_BRANCH_GAP_CM,
    get_wtrack_branch_graph_inputs,
    get_wtrack_branch_side,
    get_wtrack_direction,
    get_wtrack_full_graph_inputs,
)
from v1ca1.nwb.export_augmented_nwb import (
    BODY_POSITION_SERIES_NAME,
    DEFAULT_OUTPUT_SUFFIX,
    EPHYS_INTERVALS_TABLE_NAME,
    HEAD_POSITION_SERIES_NAME,
    POSITION_COLUMNS,
    POSITION_EPOCHS_TABLE_NAME,
    POSITION_PROVENANCE_SCRATCH_NAME,
    POSITION_RELATIVE_PATH,
    POSITION_SAMPLE_METADATA_NAME,
    RIPPLE_PROVENANCE_SCRATCH_NAME,
    RIPPLES_INTERVALS_TABLE_NAME,
    SORTING_PROVENANCE_SCRATCH_NAME,
    TRAJECTORY_INTERVALS_TABLE_NAME,
    WTRACK_FULL_GRAPH_CONFIGURATION_NAME,
    WTRACK_LINEARIZATION_TABLE_NAME,
    add_wtrack_linearization_to_nwb,
    build_wtrack_linearization_configurations,
    export_augmented_nwb,
    load_position_parquet,
    load_position_producer_provenance,
    load_ripple_times_parquet,
    parse_arguments,
)


ACQUISITION_DATASET_PATH = "/acquisition/ElectricalSeries/data"
ACQUISITION_TIMESTAMPS_PATH = "/acquisition/ElectricalSeries/timestamps"
LFP_DATASET_PATH = "/processing/ecephys/LFP/LFPSeries/data"
RIPPLE_METRIC_COLUMNS = (
    "duration",
    "max_thresh",
    "mean_zscore",
    "median_zscore",
    "max_zscore",
    "min_zscore",
    "area",
    "total_energy",
    "speed_at_start",
    "speed_at_end",
    "max_speed",
    "min_speed",
    "median_speed",
    "mean_speed",
)
RIPPLE_COLUMNS = ("start", "end", "epoch", *RIPPLE_METRIC_COLUMNS)


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


def _add_test_ephys_series(
    nwbfile: Any,
    acquisition_data: np.ndarray | None = None,
    lfp_data: np.ndarray | None = None,
    acquisition_maxshape: tuple[int, ...] | None = None,
) -> None:
    """Add acquisition and processed ElectricalSeries objects to one test NWB file."""
    pynwb = pytest.importorskip("pynwb")

    acquisition_data = (
        np.arange(12, dtype=np.int16).reshape(12, 1)
        if acquisition_data is None
        else np.asarray(acquisition_data)
    )
    lfp_data = (
        np.arange(20, dtype=np.int16).reshape(20, 1)
        if lfp_data is None
        else np.asarray(lfp_data)
    )
    if acquisition_data.ndim != 2 or lfp_data.ndim != 2:
        raise ValueError("Test ElectricalSeries data must be two-dimensional.")
    if acquisition_data.shape[1] != lfp_data.shape[1]:
        raise ValueError("Test acquisition and LFP data must have matching channels.")

    electrode_region = nwbfile.create_electrode_table_region(
        list(range(acquisition_data.shape[1])),
        "test electrodes",
    )
    acquisition_storage: Any = acquisition_data
    acquisition_time_kwargs: dict[str, Any] = {
        "timestamps": np.arange(acquisition_data.shape[0], dtype=float) / 100.0,
    }
    if acquisition_maxshape is not None:
        from hdmf.backends.hdf5 import H5DataIO

        acquisition_storage = H5DataIO(
            acquisition_data,
            chunks=True,
            maxshape=acquisition_maxshape,
        )
        acquisition_time_kwargs = {
            "starting_time": 0.0,
            "rate": 100.0,
        }
    acquisition_series = pynwb.ecephys.ElectricalSeries(
        name="ElectricalSeries",
        data=acquisition_storage,
        electrodes=electrode_region,
        filtering="none",
        **acquisition_time_kwargs,
    )
    nwbfile.add_acquisition(acquisition_series)

    lfp = pynwb.ecephys.LFP(name="LFP")
    processing_module = nwbfile.create_processing_module("ecephys", "test ecephys module")
    processing_module.add(lfp)
    lfp.add_electrical_series(
        pynwb.ecephys.ElectricalSeries(
            name="LFPSeries",
            data=lfp_data,
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
    video_timestamps_by_epoch: dict[str, np.ndarray] | None = None,
    video_series_order: list[str] | None = None,
    include_empty_position: bool = False,
    existing_position_series_name: str | None = None,
    ephys_acquisition_data: np.ndarray | None = None,
    ephys_lfp_data: np.ndarray | None = None,
    ephys_acquisition_maxshape: tuple[int, ...] | None = None,
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
        _add_test_ephys_series(
            nwbfile,
            acquisition_data=ephys_acquisition_data,
            lfp_data=ephys_lfp_data,
            acquisition_maxshape=ephys_acquisition_maxshape,
        )
    if include_source_units:
        _add_source_unit(nwbfile)
    if include_empty_position or existing_position_series_name is not None:
        behavior_module = nwbfile.create_processing_module(
            "behavior",
            "test behavior module",
        )
        position = pynwb.behavior.Position(name="position")
        behavior_module.add(position)
        if existing_position_series_name is not None:
            position.create_spatial_series(
                name=existing_position_series_name,
                data=np.asarray([[0.0, 0.0]]),
                timestamps=np.asarray([0.0]),
                unit="centimeters",
                reference_frame="test frame",
            )
    if video_timestamps_by_epoch is not None:
        video_module = nwbfile.create_processing_module(
            "video_files",
            "test video module",
        )
        video = pynwb.behavior.BehavioralEvents(name="video")
        ordered_video_epochs = (
            list(video_series_order)
            if video_series_order is not None
            else list(epoch_tags)
        )
        if set(ordered_video_epochs) != set(epoch_tags):
            raise ValueError(
                "Test video series order must contain each epoch exactly once."
            )
        for epoch_tag in ordered_video_epochs:
            if epoch_tag not in video_timestamps_by_epoch:
                raise ValueError(
                    f"Missing test video timestamps for epoch {epoch_tag!r}."
                )
            video.add_timeseries(
                pynwb.image.ImageSeries(
                    name=f"{epoch_tag}.h264",
                    external_file=[f"{epoch_tag}.h264"],
                    format="external",
                    starting_frame=[0],
                    timestamps=np.asarray(
                        video_timestamps_by_epoch[epoch_tag],
                        dtype=float,
                    ),
                )
            )
        video_module.add(video)

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


def _write_trajectory_times_parquet(
    analysis_path: Path,
    rows: list[dict[str, float | str]],
) -> None:
    """Write one canonical poke-defined trajectory table."""
    pd.DataFrame.from_records(
        rows,
        columns=["start", "end", "epoch", "trajectory_type"],
    ).to_parquet(analysis_path / "trajectory_times.parquet", index=False)


def _write_position_parquet(
    analysis_path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    """Write one canonical combined head/body position table."""
    position_path = analysis_path / POSITION_RELATIVE_PATH
    position_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows, columns=POSITION_COLUMNS).to_parquet(
        position_path,
        index=False,
    )
    return position_path


def _make_ripple_row(
    start: float,
    end: float,
    epoch: str,
    *,
    strength: float,
) -> dict[str, float | str]:
    """Return one representative canonical ripple-event row."""
    return {
        "start": float(start),
        "end": float(end),
        "epoch": str(epoch),
        "duration": float(end - start),
        "max_thresh": float(strength - 0.5),
        "mean_zscore": float(strength),
        "median_zscore": float(strength - 0.1),
        "max_zscore": float(strength + 1.0),
        "min_zscore": float(strength - 1.0),
        "area": float(strength / 10.0),
        "total_energy": float(strength**2 / 10.0),
        "speed_at_start": 0.2,
        "speed_at_end": 0.3,
        "max_speed": 0.5,
        "min_speed": 0.1,
        "median_speed": 0.25,
        "mean_speed": 0.27,
    }


def _write_ripple_times_parquet(
    analysis_path: Path,
    rows: list[dict[str, float | str]],
) -> Path:
    """Write one canonical speed-gated ripple table."""
    ripple_path = analysis_path / "ripple" / "ripple_times.parquet"
    ripple_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows, columns=RIPPLE_COLUMNS).to_parquet(
        ripple_path,
        index=False,
    )
    return ripple_path


def _write_ripple_run_log(
    analysis_path: Path,
    *,
    animal_name: str,
    date: str,
    ripple_rows: list[dict[str, float | str]],
    selected_epochs: list[str],
    use_speed_gating: bool = True,
    log_suffix: str = "matching",
    saved_root: Path = Path("/migrated/analysis"),
) -> tuple[Path, dict[str, Any]]:
    """Write one detector run log with explicit per-epoch ripple coverage."""
    event_counts = {
        epoch: sum(str(row["epoch"]) == epoch for row in ripple_rows)
        for epoch in selected_epochs
    }
    output_name = (
        "ripple_times.parquet"
        if use_speed_gating
        else "ripple_times_no_speed.parquet"
    )
    record = {
        "timestamp_utc": "2026-01-02T03:04:05+00:00",
        "script": "v1ca1.ripple.detect_ripples",
        "package_version": "0.1.0",
        "git_commit": "abc123",
        "git_dirty": False,
        "parameters": {
            "animal_name": animal_name,
            "date": date,
            "epochs": list(selected_epochs),
            "ripple_channels": [10, 11],
            "zscore_threshold": 2.0,
            "position_offset": 10,
            "use_speed_gating": bool(use_speed_gating),
            "overwrite": True,
            "notch_filter_enabled": False,
            "notch_base_freq_hz": 60.0,
            "notch_harmonics": 10,
            "notch_quality": 50.0,
        },
        "outputs": {
            "saved_interval_parquet": str(
                saved_root
                / animal_name
                / date
                / "ripple"
                / output_name
            ),
            "selected_epochs": list(selected_epochs),
            "run_epochs": list(selected_epochs),
            "skipped_existing_output_epochs": [],
            "epoch_summaries": {
                epoch: {
                    "actual_sampling_frequency_hz": 998.6438095238095,
                    "ripple_count": event_counts[epoch],
                }
                for epoch in selected_epochs
            },
        },
    }
    log_dir = analysis_path / "v1ca1_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = (
        log_dir
        / f"v1ca1_ripple_detect_ripples_20260102T030405000000Z_{log_suffix}.json"
    )
    log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return log_path, record


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
            "shape": dataset.shape,
            "maxshape": dataset.maxshape,
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


def _read_interval_table(nwb_path: Path, table_name: str) -> tuple[tuple[str, ...], Any]:
    """Return one NWB intervals table's column names and dataframe."""
    pynwb = pytest.importorskip("pynwb")

    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        interval_table = io.read().intervals[table_name]
        return tuple(interval_table.colnames), interval_table.to_dataframe()


def _read_scratch_json(nwb_path: Path, scratch_name: str) -> dict[str, Any]:
    """Return one JSON scratch item as a dictionary."""
    pynwb = pytest.importorskip("pynwb")

    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        scratch_data = io.read().scratch[scratch_name].data
        if isinstance(scratch_data, np.ndarray):
            scratch_data = scratch_data.item()
        if isinstance(scratch_data, bytes):
            scratch_data = scratch_data.decode()
    return json.loads(str(scratch_data))


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
    ephys_epoch_bounds = [(0.25, 0.95), (2.1, 2.85)]
    _write_test_nwb(
        nwb_path,
        source_epoch_tags,
        source_epoch_bounds,
        include_ephys=True,
        electrode_ids=[0],
    )
    _write_timestamps_ephys_npz(analysis_path, source_epoch_tags, ephys_epoch_bounds)
    trajectory_rows = [
        {
            "start": 2.2,
            "end": 2.4,
            "epoch": "02_r1",
            "trajectory_type": "left_to_center",
        },
        {
            "start": 2.5,
            "end": 2.7,
            "epoch": "02_r1",
            "trajectory_type": "center_to_right",
        },
    ]
    _write_trajectory_times_parquet(analysis_path, trajectory_rows)
    source_sha256 = hashlib.sha256(nwb_path.read_bytes()).hexdigest()

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_trajectory_times=True,
    )

    assert DEFAULT_OUTPUT_SUFFIX == "_augmented.nwb"
    assert output_path == nwb_root / f"{animal_name}{date}_augmented.nwb"
    assert output_path.exists()
    assert hashlib.sha256(nwb_path.read_bytes()).hexdigest() == source_sha256
    assert list(nwb_root.glob(f"{output_path.stem}.*{output_path.suffix}")) == []
    pynwb = pytest.importorskip("pynwb")
    assert pynwb.validate(path=output_path) == []

    source_tags, source_starts, source_stops = _read_epoch_bounds(nwb_path)
    output_tags, output_starts, output_stops = _read_epoch_bounds(output_path)

    assert source_tags == source_epoch_tags
    assert np.allclose(source_starts, [0.0, 2.0])
    assert np.allclose(source_stops, [1.0, 3.0])
    assert output_tags == source_epoch_tags
    assert np.allclose(output_starts, [0.0, 2.0])
    assert np.allclose(output_stops, [1.0, 3.0])

    ephys_colnames, ephys_df = _read_interval_table(
        output_path,
        EPHYS_INTERVALS_TABLE_NAME,
    )
    trajectory_colnames, trajectory_df = _read_interval_table(
        output_path,
        TRAJECTORY_INTERVALS_TABLE_NAME,
    )
    assert ephys_colnames == ("start_time", "stop_time", "epoch")
    assert ephys_df.reset_index(drop=True).to_dict("records") == [
        {
            "start_time": 0.25,
            "stop_time": 0.95,
            "epoch": "01_s1",
        },
        {
            "start_time": 2.1,
            "stop_time": 2.85,
            "epoch": "02_r1",
        },
    ]
    assert trajectory_colnames == (
        "start_time",
        "stop_time",
        "epoch",
        "trajectory_type",
    )
    assert trajectory_df.reset_index(drop=True).to_dict("records") == [
        {
            "start_time": 2.2,
            "stop_time": 2.4,
            "epoch": "02_r1",
            "trajectory_type": "left_to_center",
        },
        {
            "start_time": 2.5,
            "stop_time": 2.7,
            "epoch": "02_r1",
            "trajectory_type": "center_to_right",
        },
    ]

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


def test_export_augmented_nwb_writes_compact_streamed_ephys(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    channel_count = 32
    sample_count = 32_768
    compressible_data = np.zeros(
        (sample_count, channel_count),
        dtype=np.int16,
    )
    _write_test_nwb(
        nwb_path,
        ["01_s1"],
        [(0.0, 1.0)],
        include_ephys=True,
        electrode_ids=list(range(channel_count)),
        ephys_acquisition_data=compressible_data,
        ephys_lfp_data=compressible_data,
    )
    _write_timestamps_ephys_npz(
        analysis_path,
        ["01_s1"],
        [(0.1, 0.9)],
    )
    source_size_bytes = nwb_path.stat().st_size

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
    )

    assert output_path.stat().st_size < 0.75 * source_size_bytes
    for dataset_path in (ACQUISITION_DATASET_PATH, LFP_DATASET_PATH):
        output_storage = _read_dataset_storage(output_path, dataset_path)
        assert output_storage["compression"] == "gzip"
        assert output_storage["compression_opts"] == 4
        assert output_storage["shuffle"] is True
        assert np.array_equal(output_storage["data"], compressible_data)


def test_export_augmented_nwb_preserves_resizable_ephys_shape(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    acquisition_data = np.arange(12, dtype=np.int16).reshape(12, 1)
    _write_test_nwb(
        nwb_path,
        ["01_s1"],
        [(0.0, 1.0)],
        include_ephys=True,
        electrode_ids=[0],
        ephys_acquisition_data=acquisition_data,
        ephys_acquisition_maxshape=(24, 1),
    )
    _write_timestamps_ephys_npz(
        analysis_path,
        ["01_s1"],
        [(0.1, 0.9)],
    )

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
    )

    output_storage = _read_dataset_storage(output_path, ACQUISITION_DATASET_PATH)
    assert output_storage["shape"] == acquisition_data.shape
    assert output_storage["maxshape"] == (24, 1)
    assert np.array_equal(output_storage["data"], acquisition_data)


def test_export_augmented_nwb_can_export_cleaned_position(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    epoch_tags = ["01_s1", "02_r1", "03_s2"]
    video_timestamps_by_epoch = {
        "01_s1": np.asarray([0.05, 0.15]),
        "02_r1": np.asarray([2.05, 2.15]),
        "03_s2": np.asarray([4.05, 4.15, 4.25]),
    }
    _write_test_nwb(
        nwb_path,
        epoch_tags,
        [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)],
        video_timestamps_by_epoch=video_timestamps_by_epoch,
        video_series_order=["03_s2", "02_r1", "01_s1"],
        include_empty_position=True,
    )
    _write_timestamps_ephys_npz(
        analysis_path,
        epoch_tags,
        [(0.1, 0.9), (2.1, 2.9), (4.1, 4.9)],
    )
    position_rows = [
        {
            "epoch": "01_s1",
            "frame": 10,
            "frame_time_s": 0.05,
            "head_x_cm": 1.0,
            "head_y_cm": 2.0,
            "body_x_cm": np.nan,
            "body_y_cm": np.nan,
        },
        {
            "epoch": "01_s1",
            "frame": 11,
            "frame_time_s": 0.15,
            "head_x_cm": np.nan,
            "head_y_cm": np.nan,
            "body_x_cm": np.nan,
            "body_y_cm": np.nan,
        },
        {
            "epoch": "03_s2",
            "frame": 5,
            "frame_time_s": 4.05,
            "head_x_cm": 3.0,
            "head_y_cm": 4.0,
            "body_x_cm": 2.5,
            "body_y_cm": 3.5,
        },
        {
            "epoch": "03_s2",
            "frame": 7,
            "frame_time_s": 4.15,
            "head_x_cm": 5.0,
            "head_y_cm": 6.0,
            "body_x_cm": 4.5,
            "body_y_cm": 5.5,
        },
        {
            "epoch": "03_s2",
            "frame": 9,
            "frame_time_s": 4.25,
            "head_x_cm": 7.0,
            "head_y_cm": 8.0,
            "body_x_cm": 6.5,
            "body_y_cm": 7.5,
        },
    ]
    position_path = _write_position_parquet(
        analysis_path,
        position_rows,
    )
    source_sha256 = hashlib.sha256(position_path.read_bytes()).hexdigest()

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_position=True,
    )

    pynwb = pytest.importorskip("pynwb")
    assert pynwb.validate(path=output_path) == []
    expected_table = pd.DataFrame.from_records(
        position_rows,
        columns=POSITION_COLUMNS,
    )
    with pynwb.NWBHDF5IO(output_path, "r") as io:
        nwbfile = io.read()
        behavior = nwbfile.processing["behavior"]
        position = behavior.data_interfaces["position"]
        head_series = position.spatial_series[HEAD_POSITION_SERIES_NAME]
        body_series = position.spatial_series[BODY_POSITION_SERIES_NAME]

        assert set(position.spatial_series) == {
            HEAD_POSITION_SERIES_NAME,
            BODY_POSITION_SERIES_NAME,
        }
        assert head_series.unit == "centimeters"
        assert body_series.unit == "centimeters"
        np.testing.assert_allclose(
            head_series.data[:],
            expected_table[["head_x_cm", "head_y_cm"]].to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            body_series.data[:],
            expected_table[["body_x_cm", "body_y_cm"]].to_numpy(),
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            head_series.timestamps[:],
            expected_table["frame_time_s"].to_numpy(),
        )
        assert body_series.timestamps is head_series.timestamps

        sample_metadata = behavior.data_interfaces[
            POSITION_SAMPLE_METADATA_NAME
        ].to_dataframe()
        epoch_intervals = nwbfile.intervals[
            POSITION_EPOCHS_TABLE_NAME
        ].to_dataframe()
        position_offset_description = nwbfile.intervals[
            POSITION_EPOCHS_TABLE_NAME
        ]["analysis_start_offset_samples"].description

    assert sample_metadata.index.tolist() == [0, 1, 2, 3, 4]
    assert sample_metadata["epoch"].tolist() == [
        "01_s1",
        "01_s1",
        "03_s2",
        "03_s2",
        "03_s2",
    ]
    assert sample_metadata["frame"].tolist() == [10, 11, 5, 7, 9]
    assert "local indices 0 through 9 remain stored" in position_offset_description
    assert epoch_intervals.reset_index(drop=True).to_dict("records") == [
        {
            "start_time": 0.05,
            "stop_time": 0.15,
            "epoch": "01_s1",
            "start_index": 0,
            "stop_index_exclusive": 2,
            "sample_count": 2,
            "analysis_start_offset_samples": DEFAULT_POSITION_OFFSET,
            "first_frame": 10,
            "last_frame": 11,
            "video_series_name": "01_s1.h264",
        },
        {
            "start_time": 4.05,
            "stop_time": 4.25,
            "epoch": "03_s2",
            "start_index": 2,
            "stop_index_exclusive": 5,
            "sample_count": 3,
            "analysis_start_offset_samples": DEFAULT_POSITION_OFFSET,
            "first_frame": 5,
            "last_frame": 9,
            "video_series_name": "03_s2.h264",
        },
    ]

    provenance = _read_scratch_json(
        output_path,
        POSITION_PROVENANCE_SCRATCH_NAME,
    )
    assert provenance["source_artifact"]["sha256"] == source_sha256
    assert provenance["source_artifact"]["row_count"] == 5
    assert provenance["source_artifact"]["epoch_order"] == [
        "01_s1",
        "03_s2",
    ]
    assert provenance["source_artifact"]["frame_counts_by_epoch"] == {
        "01_s1": 2,
        "03_s2": 3,
    }
    assert provenance["source_artifact"]["nan_counts_by_epoch"]["01_s1"] == {
        "head_x_cm": 1,
        "head_y_cm": 1,
        "body_x_cm": 2,
        "body_y_cm": 2,
    }
    assert provenance["timestamps"]["validation_by_epoch"]["01_s1"][
        "video_series_name"
    ] == "01_s1.h264"
    assert provenance["producer"]["status"] == "missing"

    export_logs = list(
        (analysis_path / "v1ca1_log").glob(
            "v1ca1_nwb_export_augmented_nwb_*.json"
        )
    )
    assert len(export_logs) == 1
    export_record = json.loads(export_logs[0].read_text(encoding="utf-8"))
    assert export_record["parameters"]["add_position"] is True
    assert export_record["parameters"]["position_offset"] == DEFAULT_POSITION_OFFSET
    assert export_record["outputs"]["position_sample_count"] == 5
    assert export_record["outputs"][
        "position_analysis_start_offset_samples"
    ] == DEFAULT_POSITION_OFFSET
    assert export_record["outputs"]["position_epoch_order"] == [
        "01_s1",
        "03_s2",
    ]


def test_export_augmented_nwb_can_export_wtrack_linearization(
    tmp_path: Path,
) -> None:
    animal_name = "L14"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_wtrack_linearization=True,
    )

    pynwb = pytest.importorskip("pynwb")
    tl = pytest.importorskip("track_linearization")
    assert pynwb.validate(path=output_path) == []
    with pynwb.NWBHDF5IO(output_path, "r") as io:
        nwbfile = io.read()
        table = nwbfile.processing["behavior"].data_interfaces[
            WTRACK_LINEARIZATION_TABLE_NAME
        ]
        assert tuple(table.colnames) == (
            "configuration_name",
            "node_positions_cm",
            "edges",
            "edge_order",
            "edge_spacing_cm",
            "use_hmm",
        )
        table_frame = table.to_dataframe()

    expected_configuration_names = [
        *TRAJECTORY_TYPES,
        WTRACK_FULL_GRAPH_CONFIGURATION_NAME,
    ]
    assert table_frame["configuration_name"].tolist() == (
        expected_configuration_names
    )
    expected_by_name: dict[str, tuple[Any, Any, Any, Any]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        node_positions, edges, edge_order = get_wtrack_branch_graph_inputs(
            animal_name=animal_name,
            branch_side=get_wtrack_branch_side(trajectory_type),
            direction=get_wtrack_direction(trajectory_type),
        )
        expected_by_name[trajectory_type] = (
            node_positions,
            edges,
            edge_order,
            np.zeros(len(edge_order) - 1),
        )
    expected_by_name[WTRACK_FULL_GRAPH_CONFIGURATION_NAME] = (
        get_wtrack_full_graph_inputs(
            animal_name,
            branch_gap_cm=DEFAULT_WTRACK_BRANCH_GAP_CM,
        )
    )

    for row in table_frame.to_dict("records"):
        configuration_name = str(row["configuration_name"])
        node_positions = np.asarray(row["node_positions_cm"], dtype=float)
        edges = np.asarray(row["edges"], dtype=int)
        edge_order = np.asarray(row["edge_order"], dtype=int)
        edge_spacing = np.asarray(row["edge_spacing_cm"], dtype=float)
        expected_nodes, expected_edges, expected_order, expected_spacing = (
            expected_by_name[configuration_name]
        )
        np.testing.assert_allclose(node_positions, expected_nodes)
        np.testing.assert_array_equal(edges, expected_edges)
        np.testing.assert_array_equal(edge_order, expected_order)
        np.testing.assert_allclose(edge_spacing, expected_spacing)
        assert bool(row["use_hmm"]) is False

        graph = tl.make_track_graph(node_positions, edges)
        linearized = tl.get_linearized_position(
            position=node_positions,
            track_graph=graph,
            edge_order=[tuple(edge) for edge in edge_order.tolist()],
            edge_spacing=edge_spacing.tolist(),
            use_HMM=bool(row["use_hmm"]),
        )
        assert len(linearized) == len(node_positions)
        assert np.all(np.isfinite(linearized["linear_position"]))

    export_logs = list(
        (analysis_path / "v1ca1_log").glob(
            "v1ca1_nwb_export_augmented_nwb_*.json"
        )
    )
    assert len(export_logs) == 1
    export_record = json.loads(export_logs[0].read_text(encoding="utf-8"))
    assert export_record["parameters"]["add_wtrack_linearization"] is True
    assert export_record["outputs"][
        "wtrack_linearization_table_path"
    ] == f"/processing/behavior/{WTRACK_LINEARIZATION_TABLE_NAME}"
    assert export_record["outputs"][
        "wtrack_linearization_configuration_names"
    ] == expected_configuration_names


def test_add_wtrack_linearization_rejects_position_unit_mismatch() -> None:
    pynwb = pytest.importorskip("pynwb")
    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    behavior = nwbfile.create_processing_module("behavior", "Behavioral data.")
    position = pynwb.behavior.Position(name="position")
    position.create_spatial_series(
        name=HEAD_POSITION_SERIES_NAME,
        data=np.asarray([[0.0, 0.0]]),
        timestamps=np.asarray([0.0]),
        unit="meters",
        reference_frame="test frame",
    )
    behavior.add(position)

    with pytest.raises(ValueError, match="require 'centimeters'"):
        add_wtrack_linearization_to_nwb(
            nwbfile=nwbfile,
            configurations=build_wtrack_linearization_configurations("L14"),
        )


def test_add_wtrack_linearization_rejects_existing_table() -> None:
    pynwb = pytest.importorskip("pynwb")
    from hdmf.common import DynamicTable

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    behavior = nwbfile.create_processing_module("behavior", "Behavioral data.")
    behavior.add(
        DynamicTable(
            name=WTRACK_LINEARIZATION_TABLE_NAME,
            description="existing table",
        )
    )

    with pytest.raises(ValueError, match="already contains a data interface"):
        add_wtrack_linearization_to_nwb(
            nwbfile=nwbfile,
            configurations=build_wtrack_linearization_configurations("L14"),
        )


@pytest.mark.parametrize(
    ("malformation", "expected_message"),
    [
        ("missing_column", "missing required columns"),
        ("fractional_frame", "non-integer frame values"),
        ("duplicate_frame", "duplicate frames"),
        ("infinite_coordinate", "infinite position coordinates"),
        ("interleaved_epochs", "repeated non-contiguous epoch block"),
    ],
)
def test_load_position_parquet_rejects_malformed_tables(
    tmp_path: Path,
    malformation: str,
    expected_message: str,
) -> None:
    analysis_path = tmp_path / "analysis"
    position_path = analysis_path / POSITION_RELATIVE_PATH
    position_path.parent.mkdir(parents=True)
    table = pd.DataFrame.from_records(
        [
            {
                "epoch": "01_s1",
                "frame": 0,
                "frame_time_s": 0.1,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": 0.5,
                "body_y_cm": 1.5,
            },
            {
                "epoch": "01_s1",
                "frame": 1,
                "frame_time_s": 0.2,
                "head_x_cm": 2.0,
                "head_y_cm": 3.0,
                "body_x_cm": 1.5,
                "body_y_cm": 2.5,
            },
            {
                "epoch": "02_r1",
                "frame": 0,
                "frame_time_s": 2.1,
                "head_x_cm": 3.0,
                "head_y_cm": 4.0,
                "body_x_cm": 2.5,
                "body_y_cm": 3.5,
            },
        ],
        columns=POSITION_COLUMNS,
    )

    if malformation == "missing_column":
        table = table.drop(columns="body_y_cm")
    elif malformation == "fractional_frame":
        table["frame"] = table["frame"].astype(float)
        table.loc[1, "frame"] = 1.5
    elif malformation == "duplicate_frame":
        table.loc[1, "frame"] = 0
    elif malformation == "infinite_coordinate":
        table.loc[1, "head_x_cm"] = np.inf
    elif malformation == "interleaved_epochs":
        table = table.iloc[[0, 2, 1]].reset_index(drop=True)
    else:
        raise AssertionError(f"Unexpected malformation: {malformation}")
    table.to_parquet(position_path, index=False)

    with pytest.raises(ValueError, match=expected_message):
        load_position_parquet(analysis_path)


def test_load_position_producer_provenance_matches_legacy_converter(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    position_path = _write_position_parquet(
        analysis_path,
        [
            {
                "epoch": "01_s1",
                "frame": 0,
                "frame_time_s": 0.05,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": np.nan,
                "body_y_cm": np.nan,
            },
        ],
    )
    producer_record = {
        "timestamp_utc": "2026-01-02T03:04:05+00:00",
        "script": "v1ca1.position.convert_legacy_position_pickles",
        "package_version": "0.1.0",
        "git_commit": "abc123",
        "git_dirty": True,
        "parameters": {
            "animal_name": animal_name,
            "date": date,
            "head_position_path": "/migrated/position.pkl",
            "body_position_path": "/migrated/body_position.pkl",
        },
        "outputs": {
            "written_epochs": ["01_s1"],
            "missing_body_epochs": ["01_s1"],
            "combined_frame_count": 1,
            "combined_output_path": str(
                Path("/migrated")
                / animal_name
                / date
                / POSITION_RELATIVE_PATH
            ),
        },
    }
    log_dir = analysis_path / "v1ca1_log"
    log_dir.mkdir(parents=True)
    log_path = (
        log_dir
        / "v1ca1_position_convert_legacy_position_pickles_test.json"
    )
    log_path.write_text(
        json.dumps(producer_record),
        encoding="utf-8",
    )

    position_table, _ = load_position_parquet(analysis_path)
    provenance = load_position_producer_provenance(
        analysis_path=analysis_path,
        animal_name=animal_name,
        date=date,
        position_path=position_path,
        position_table=position_table,
    )

    assert provenance["status"] == "matched"
    assert provenance["producer_script"] == (
        "v1ca1.position.convert_legacy_position_pickles"
    )
    assert provenance["analysis_relative_path"] == (
        f"v1ca1_log/{log_path.name}"
    )
    assert provenance["record"] == producer_record


def test_export_augmented_nwb_validates_position_video_timestamps(
    tmp_path: Path,
) -> None:
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
        video_timestamps_by_epoch={
            "01_s1": np.asarray([0.05, 0.15]),
        },
    )
    _write_timestamps_ephys_npz(
        analysis_path,
        ["01_s1"],
        [(0.1, 0.9)],
    )
    _write_position_parquet(
        analysis_path,
        [
            {
                "epoch": "01_s1",
                "frame": 0,
                "frame_time_s": 0.05,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": 0.5,
                "body_y_cm": 1.5,
            },
            {
                "epoch": "01_s1",
                "frame": 1,
                "frame_time_s": 0.1501,
                "head_x_cm": 2.0,
                "head_y_cm": 3.0,
                "body_x_cm": 1.5,
                "body_y_cm": 2.5,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="timestamps do not match NWB video series '01_s1.h264'",
    ):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_position=True,
        )


def test_export_augmented_nwb_requires_position_parquet_when_enabled(
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
    _write_timestamps_ephys_npz(
        analysis_path,
        ["01_s1"],
        [(0.1, 0.9)],
    )

    with pytest.raises(FileNotFoundError, match="Combined position parquet"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_position=True,
        )


def test_export_augmented_nwb_rejects_existing_position_series(
    tmp_path: Path,
) -> None:
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
        video_timestamps_by_epoch={
            "01_s1": np.asarray([0.05]),
        },
        existing_position_series_name=HEAD_POSITION_SERIES_NAME,
    )
    _write_timestamps_ephys_npz(
        analysis_path,
        ["01_s1"],
        [(0.1, 0.9)],
    )
    _write_position_parquet(
        analysis_path,
        [
            {
                "epoch": "01_s1",
                "frame": 0,
                "frame_time_s": 0.05,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": 0.5,
                "body_y_cm": 1.5,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="already contains destination SpatialSeries names",
    ):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_position=True,
        )


def test_export_augmented_nwb_can_export_speed_gated_ripples(
    tmp_path: Path,
) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    epoch_tags = ["01_s1", "02_r1"]
    _write_test_nwb(
        nwb_path,
        epoch_tags,
        [(0.0, 1.0), (2.0, 3.0)],
    )
    _write_timestamps_ephys_npz(
        analysis_path,
        epoch_tags,
        [(0.1, 0.9), (2.1, 2.9)],
    )
    ripple_rows = [
        _make_ripple_row(0.2, 0.24, "01_s1", strength=4.5),
        _make_ripple_row(2.2, 2.27, "02_r1", strength=5.0),
    ]
    ripple_path = _write_ripple_times_parquet(analysis_path, ripple_rows)
    _matching_log_path, matching_record = _write_ripple_run_log(
        analysis_path,
        animal_name=animal_name,
        date=date,
        ripple_rows=ripple_rows,
        selected_epochs=epoch_tags,
    )
    repeated_log_path, repeated_record = _write_ripple_run_log(
        analysis_path,
        animal_name=animal_name,
        date=date,
        ripple_rows=ripple_rows,
        selected_epochs=epoch_tags,
        log_suffix="repeated",
    )
    assert repeated_record == matching_record

    no_speed_path = analysis_path / "ripple" / "ripple_times_no_speed.parquet"
    pd.DataFrame.from_records(
        [_make_ripple_row(0.3, 0.35, "01_s1", strength=9.0)],
        columns=RIPPLE_COLUMNS,
    ).to_parquet(no_speed_path, index=False)
    _write_ripple_run_log(
        analysis_path,
        animal_name=animal_name,
        date=date,
        ripple_rows=[
            _make_ripple_row(0.3, 0.35, "01_s1", strength=9.0),
        ],
        selected_epochs=epoch_tags,
        use_speed_gating=False,
        log_suffix="newer_no_speed",
    )

    source_sha256 = hashlib.sha256(ripple_path.read_bytes()).hexdigest()
    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_ripples=True,
    )

    pynwb = pytest.importorskip("pynwb")
    assert pynwb.validate(path=output_path) == []

    ripple_colnames, ripple_df = _read_interval_table(
        output_path,
        RIPPLES_INTERVALS_TABLE_NAME,
    )
    assert ripple_colnames == (
        "start_time",
        "stop_time",
        "epoch",
        *RIPPLE_METRIC_COLUMNS,
    )
    assert ripple_df.index.tolist() == [0, 1]
    assert ripple_df["epoch"].tolist() == epoch_tags
    assert np.allclose(ripple_df["start_time"], [0.2, 2.2])
    assert np.allclose(ripple_df["stop_time"], [0.24, 2.27])
    for column_name in RIPPLE_METRIC_COLUMNS:
        assert np.allclose(
            ripple_df[column_name].to_numpy(dtype=float),
            [float(row[column_name]) for row in ripple_rows],
        )

    provenance = _read_scratch_json(
        output_path,
        RIPPLE_PROVENANCE_SCRATCH_NAME,
    )
    assert provenance["schema_version"] == 1
    assert provenance["intervals_table_name"] == RIPPLES_INTERVALS_TABLE_NAME
    assert provenance["source_artifact"] == {
        "analysis_relative_path": "ripple/ripple_times.parquet",
        "sha256": source_sha256,
        "row_count": 2,
        "imported_columns": list(RIPPLE_COLUMNS),
        "imported_dtypes": {
            "start": "float64",
            "end": "float64",
            "epoch": "object",
            **{column_name: "float64" for column_name in RIPPLE_METRIC_COLUMNS},
        },
        "event_counts_by_epoch": {"01_s1": 1, "02_r1": 1},
        "source_row_order_preserved": True,
    }
    assert provenance["run_log"]["analysis_relative_path"] == (
        f"v1ca1_log/{repeated_log_path.name}"
    )
    assert provenance["run_log"]["record"] == matching_record
    assert provenance["run_log"]["record"]["parameters"]["use_speed_gating"] is True

    export_logs = list(
        (analysis_path / "v1ca1_log").glob(
            "v1ca1_nwb_export_augmented_nwb_*.json"
        )
    )
    assert len(export_logs) == 1
    export_record = json.loads(export_logs[0].read_text(encoding="utf-8"))
    assert export_record["parameters"]["add_ripples"] is True
    assert export_record["outputs"]["ripple_intervals_table_name"] == "ripples"
    assert export_record["outputs"]["ripple_intervals_row_count"] == 2
    assert export_record["outputs"][
        "ripple_detection_provenance_scratch_name"
    ] == RIPPLE_PROVENANCE_SCRATCH_NAME


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
    ephys_epoch_bounds = [(0.25, 0.95), (2.1, 2.85)]
    electrode_ids = _get_shank_channel_ids(0, 0) + _get_shank_channel_ids(1, 0)
    _write_test_nwb(
        nwb_path,
        source_epoch_tags,
        source_epoch_bounds,
        include_ephys=True,
        electrode_ids=electrode_ids,
    )
    _write_timestamps_ephys_npz(analysis_path, source_epoch_tags, ephys_epoch_bounds)

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
    assert np.allclose(output_starts, [0.0, 2.0])
    assert np.allclose(output_stops, [1.0, 3.0])

    _ephys_colnames, ephys_df = _read_interval_table(
        output_path,
        EPHYS_INTERVALS_TABLE_NAME,
    )
    assert np.allclose(ephys_df["start_time"], [0.25, 2.1])
    assert np.allclose(ephys_df["stop_time"], [0.95, 2.85])
    assert ephys_df["epoch"].tolist() == source_epoch_tags

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
    expected_obs_intervals = np.asarray(ephys_epoch_bounds, dtype=float)

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


def test_export_augmented_nwb_requires_ripple_parquet_when_enabled(
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

    with pytest.raises(FileNotFoundError, match="ripple_times.parquet not found"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_ripples=True,
        )


@pytest.mark.parametrize(
    ("malformation", "expected_message"),
    [
        ("missing_epoch", "missing required columns"),
        ("bad_duration", "inconsistent with end - start"),
        ("non_finite_metric", "non-finite values in 'mean_zscore'"),
        ("unsorted", "not in canonical start/end/epoch order"),
        ("reserved_tags", "reserved by NWB TimeIntervals"),
    ],
)
def test_load_ripple_times_parquet_rejects_malformed_tables(
    tmp_path: Path,
    malformation: str,
    expected_message: str,
) -> None:
    analysis_path = tmp_path / "analysis"
    ripple_path = analysis_path / "ripple" / "ripple_times.parquet"
    ripple_path.parent.mkdir(parents=True)
    rows = [
        _make_ripple_row(0.2, 0.24, "01_s1", strength=4.5),
        _make_ripple_row(0.4, 0.46, "01_s1", strength=5.0),
    ]
    table = pd.DataFrame.from_records(rows, columns=RIPPLE_COLUMNS)

    if malformation == "missing_epoch":
        table = table.drop(columns="epoch")
    elif malformation == "bad_duration":
        table.loc[0, "duration"] = 1.0
    elif malformation == "non_finite_metric":
        table.loc[0, "mean_zscore"] = np.nan
    elif malformation == "unsorted":
        table = table.iloc[::-1].reset_index(drop=True)
    elif malformation == "reserved_tags":
        table["tags"] = "unexpected"
    else:
        raise AssertionError(f"Unexpected malformation: {malformation}")
    table.to_parquet(ripple_path, index=False)

    with pytest.raises(ValueError, match=expected_message):
        load_ripple_times_parquet(analysis_path)


def test_export_augmented_nwb_requires_matching_ripple_provenance(
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
    ripple_rows = [
        _make_ripple_row(0.2, 0.24, "01_s1", strength=4.5),
    ]
    _write_ripple_times_parquet(analysis_path, ripple_rows)
    _write_ripple_run_log(
        analysis_path,
        animal_name=animal_name,
        date=date,
        ripple_rows=ripple_rows,
        selected_epochs=["01_s1"],
        use_speed_gating=False,
        log_suffix="no_speed_only",
    )

    with pytest.raises(
        ValueError,
        match="No complete speed-gated ripple detector run log matches",
    ):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_ripples=True,
        )


@pytest.mark.parametrize(
    ("ripple_epoch", "start", "end", "expected_message"),
    [
        ("02_r1", 0.2, 0.24, "epochs absent"),
        ("01_s1", 0.05, 0.24, "falls outside the ephys bounds"),
    ],
)
def test_export_augmented_nwb_validates_ripple_epoch_bounds(
    tmp_path: Path,
    ripple_epoch: str,
    start: float,
    end: float,
    expected_message: str,
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
    ripple_rows = [
        _make_ripple_row(start, end, ripple_epoch, strength=4.5),
    ]
    _write_ripple_times_parquet(analysis_path, ripple_rows)
    _write_ripple_run_log(
        analysis_path,
        animal_name=animal_name,
        date=date,
        ripple_rows=ripple_rows,
        selected_epochs=[ripple_epoch],
    )

    with pytest.raises(ValueError, match=expected_message):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_ripples=True,
        )


def test_export_augmented_nwb_writes_empty_ripple_table_with_coverage(
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
    _write_ripple_times_parquet(analysis_path, [])
    _write_ripple_run_log(
        analysis_path,
        animal_name=animal_name,
        date=date,
        ripple_rows=[],
        selected_epochs=["01_s1"],
    )

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_ripples=True,
    )

    ripple_colnames, ripple_df = _read_interval_table(
        output_path,
        RIPPLES_INTERVALS_TABLE_NAME,
    )
    assert set(ripple_colnames) == {
        "start_time",
        "stop_time",
        "epoch",
        *RIPPLE_METRIC_COLUMNS,
    }
    assert ripple_df.empty
    provenance = _read_scratch_json(
        output_path,
        RIPPLE_PROVENANCE_SCRATCH_NAME,
    )
    assert provenance["source_artifact"]["event_counts_by_epoch"] == {}
    assert provenance["run_log"]["record"]["outputs"]["epoch_summaries"] == {
        "01_s1": {
            "actual_sampling_frequency_hz": 998.6438095238095,
            "ripple_count": 0,
        }
    }


def test_parse_arguments_accepts_add_ripples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_augmented_nwb.py",
            "--animal-name",
            "animal",
            "--date",
            "20240101",
            "--add-ripples",
            "--add-position",
            "--position-offset",
            "7",
            "--add-wtrack-linearization",
        ],
    )

    args = parse_arguments()

    assert args.add_ripples is True
    assert args.add_position is True
    assert args.position_offset == 7
    assert args.add_wtrack_linearization is True


def test_export_augmented_nwb_requires_trajectory_parquet_when_enabled(
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

    with pytest.raises(FileNotFoundError, match="trajectory_times.parquet not found"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_trajectory_times=True,
        )


def test_export_augmented_nwb_rejects_malformed_trajectory_parquet(
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
    pd.DataFrame(
        [{"start": 0.2, "end": 0.3, "epoch": "01_s1"}]
    ).to_parquet(analysis_path / "trajectory_times.parquet", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            add_trajectory_times=True,
        )


def test_export_augmented_nwb_writes_empty_trajectory_table(tmp_path: Path) -> None:
    animal_name = "animal"
    date = "20240101"
    analysis_path = tmp_path / "analysis" / animal_name / date
    nwb_root = tmp_path / "raw"
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    analysis_path.mkdir(parents=True)
    nwb_root.mkdir(parents=True)

    _write_test_nwb(nwb_path, ["01_s1"], [(0.0, 1.0)])
    _write_timestamps_ephys_npz(analysis_path, ["01_s1"], [(0.1, 0.9)])
    _write_trajectory_times_parquet(analysis_path, [])

    output_path = export_augmented_nwb(
        animal_name=animal_name,
        date=date,
        data_root=tmp_path / "analysis",
        nwb_root=nwb_root,
        add_trajectory_times=True,
    )

    colnames, trajectory_df = _read_interval_table(
        output_path,
        TRAJECTORY_INTERVALS_TABLE_NAME,
    )
    assert set(colnames) == {"start_time", "stop_time", "epoch", "trajectory_type"}
    assert trajectory_df.empty


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


def test_export_augmented_nwb_preserves_existing_output_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    source_sha256 = hashlib.sha256(nwb_path.read_bytes()).hexdigest()
    existing_output = b"existing augmented output"
    output_path.write_bytes(existing_output)

    def fail_after_partial_write(
        read_io: Any,
        nwbfile: Any,
        output_path: Path,
    ) -> None:
        del read_io, nwbfile
        output_path.write_bytes(b"partial temporary output")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        "v1ca1.nwb.export_augmented_nwb.write_nwb_copy",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="simulated write failure"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            overwrite=True,
        )

    assert hashlib.sha256(nwb_path.read_bytes()).hexdigest() == source_sha256
    assert output_path.read_bytes() == existing_output
    assert list(nwb_root.glob(f"{output_path.stem}.*{output_path.suffix}")) == []


def test_export_augmented_nwb_rejects_source_as_output_even_with_overwrite(
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
    source_sha256 = hashlib.sha256(nwb_path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="Output path must differ from the source"):
        export_augmented_nwb(
            animal_name=animal_name,
            date=date,
            data_root=tmp_path / "analysis",
            nwb_root=nwb_root,
            output_path=nwb_path,
            overwrite=True,
        )

    assert hashlib.sha256(nwb_path.read_bytes()).hexdigest() == source_sha256


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
