"""Tests for standalone epoch RippleBandLFP artifact bundles."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.ripple import detect_ripples
from v1ca1.spyglass import ripple_band_lfp


RESULT_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
ELECTRODE_IDS = [201, 17]
SOURCE_FS = 4096.0
GAIN_TO_UV = [2.0, 0.5]
OFFSET_TO_UV = [10.0, -3.0]


def _raw_input() -> tuple[np.ndarray, np.ndarray]:
    """Return one second of two ordered int16 acquisition channels."""
    timestamps = np.arange(int(SOURCE_FS), dtype=float) / SOURCE_FS
    traces = np.column_stack(
        (
            1000.0 * np.sin(2.0 * np.pi * 200.0 * timestamps),
            700.0 * np.sin(2.0 * np.pi * 180.0 * timestamps)
            + 120.0 * np.sin(2.0 * np.pi * 60.0 * timestamps),
        )
    ).astype(np.int16)
    return timestamps, traces


def _sampling_frequency_provenance(
    timestamps: np.ndarray,
    *,
    sampling_frequency_hz: float = SOURCE_FS,
) -> dict[str, object]:
    """Return the frozen SpikeInterface first-timestamp estimator snapshot."""
    reference = np.asarray(timestamps[:1000], dtype=float)
    return {
        "method": ripple_band_lfp.SAMPLING_FREQUENCY_ESTIMATION_METHOD,
        "samples_for_rate_estimation": len(reference),
        "reference_start_index": 0,
        "reference_stop_index_exclusive": len(reference),
        "reference_timestamps_sha256": ripple_band_lfp._array_sha256(reference),
        "estimated_sampling_frequency_hz": sampling_frequency_hz,
    }


def _source_slice_provenance(
    timestamps: np.ndarray,
    *,
    start_index: int = 100,
) -> dict[str, object]:
    """Return canonical source paths, bounds, and ordered channel mappings."""
    return {
        "epoch": "08_r4",
        "data_path": "/acquisition/ElectricalSeries/data",
        "timestamps_path": "/acquisition/ElectricalSeries/timestamps",
        "electrodes_path": "/acquisition/ElectricalSeries/electrodes",
        "interval_table_path": "/intervals/ephys_recording_intervals",
        "electrodes_table_path": "/general/extracellular_ephys/electrodes",
        "interval_table_row_index": 7,
        "source_start_index": start_index,
        "source_stop_index_exclusive": start_index + len(timestamps),
        "epoch_start_time_s": float(timestamps[0]) if len(timestamps) else None,
        "epoch_stop_time_s": float(timestamps[-1]) if len(timestamps) else None,
        "electrical_series_object_id": "electrical-series-object-id",
        "electrodes_region_object_id": "electrodes-region-object-id",
        "electrodes_table_object_id": "electrodes-object-id",
        "interval_table_object_id": "intervals-object-id",
        "electrodes_region_sha256": "d" * 64,
        "electrodes_table_ids_sha256": "e" * 64,
        "selected_data_column_indices": [3, 1],
        "selected_electrode_table_rows": [23, 7],
    }


def _compute_kwargs(**overrides: object) -> dict[str, object]:
    """Return canonical explicit raw-NWB computation arguments."""
    timestamps, traces = _raw_input()
    kwargs: dict[str, object] = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "ripple_band_lfp_id": RESULT_ID,
        "source_nwb_file_name": "L14_20240611_augmented.nwb",
        "source_electrical_series_path": "/acquisition/ElectricalSeries",
        "raw_timestamps": timestamps,
        "raw_traces": traces,
        "electrode_ids": ELECTRODE_IDS,
        "gain_to_uv": GAIN_TO_UV,
        "offset_to_uv": OFFSET_TO_UV,
        "source_sampling_frequency_hz": SOURCE_FS,
        "sampling_frequency_provenance": _sampling_frequency_provenance(
            timestamps
        ),
        "source_slice_provenance": _source_slice_provenance(timestamps),
        "upstream_provenance": {
            "nwb_object_id": "electrical-series-object-id",
            "electrode_table_object_id": "electrodes-object-id",
        },
    }
    kwargs.update(overrides)
    return kwargs


def _legacy_dataset(result: dict[str, object]):
    """Return the exact NetCDF schema written by detect_ripples.py."""
    dataset = result["dataset"]
    parameters = result["parameters"]
    return detect_ripples.build_epoch_lfp_dataset(
        animal_name=result["metadata"]["animal_name"],
        date=result["metadata"]["date"],
        epoch=result["metadata"]["epoch"],
        timestamps=np.asarray(dataset["time"].values, dtype=float),
        filtered_lfp=np.asarray(dataset["filtered_lfp"].values, dtype=float),
        sampling_frequency=float(dataset["sampling_frequency_hz"].values),
        channel_ids=list(result["ordered_electrode_ids"]),
        enable_notch_filter=bool(parameters["enable_notch_filter"]),
    )


def _legacy_run_log_payload(source: Path) -> dict[str, object]:
    """Return the exact detector run-log pairing for one epoch cache."""
    return {
        "script": "v1ca1.ripple.detect_ripples",
        "parameters": {
            "animal_name": "L14",
            "date": "20240611",
            "epochs": ["08_r4"],
            "ripple_channels": ELECTRODE_IDS,
            "notch_filter_enabled": False,
            "notch_base_freq_hz": 60.0,
            "notch_harmonics": 10,
            "notch_quality": 50.0,
        },
        "outputs": {
            "selected_epochs": ["08_r4"],
            "saved_lfp_cache_dir": str(source.parent),
            "saved_lfp_cache_epoch_paths": {"08_r4": str(source)},
        },
        "git_commit": "abc123",
        "git_dirty": False,
        "timestamp_utc": "2026-08-07T00:00:00Z",
    }


def _write_nwb_input_fixture(
    tmp_path: Path,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write one small NWB-like HDF5 source with nontrivial channel mapping."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "audit_fixture.nwb"
    sample_count = 4000
    nominal_sampling_frequency_hz = 29959.314285714285
    timestamps = (
        1_700_000_000.0
        + np.arange(sample_count, dtype=float) / nominal_sampling_frequency_hz
    )
    timestamps[1200:] += np.linspace(0.0, 1e-4, sample_count - 1200)
    raw = np.arange(sample_count * 4, dtype=np.int16).reshape(sample_count, 4)
    with h5py.File(path, "w") as nwb_file:
        series = nwb_file.create_group("/acquisition/e-series")
        series.attrs["neurodata_type"] = "ElectricalSeries"
        series.attrs["object_id"] = "electrical-series-object-id"
        data = series.create_dataset("data", data=raw)
        data.attrs["conversion"] = 2e-6
        data.attrs["offset"] = 0.0
        data.attrs["unit"] = "volts"
        series.create_dataset("timestamps", data=timestamps)
        series.create_dataset(
            "channel_conversion",
            data=np.asarray([0.5, 1.0, 2.0, 1.5]),
        )

        table = nwb_file.create_group(
            "/general/extracellular_ephys/electrodes"
        )
        table.attrs["object_id"] = "electrodes-object-id"
        table.create_dataset("id", data=np.asarray([101, 305, 17, 900]))
        table.create_dataset(
            "offset",
            data=np.asarray([1e-6, 2e-6, 3e-6, 4e-6]),
        )
        region = series.create_dataset(
            "electrodes",
            data=np.asarray([2, 0, 3, 1]),
        )
        region.attrs["object_id"] = "electrodes-region-object-id"
        region.attrs["table"] = table.ref

        intervals = nwb_file.create_group(
            "/intervals/ephys_recording_intervals"
        )
        intervals.attrs["object_id"] = "intervals-object-id"
        intervals.create_dataset(
            "epoch",
            data=np.asarray(["08_r4"], dtype=h5py.string_dtype("utf-8")),
        )
        intervals.create_dataset("start_time", data=[timestamps[500]])
        intervals.create_dataset("stop_time", data=[timestamps[3499]])
    return path, timestamps, raw


def test_paths_parameters_and_noninterchangeable_science(tmp_path: Path) -> None:
    """The bundle is session-first and documents the exact legacy transform."""
    paths = ripple_band_lfp.get_ripple_band_lfp_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_band_lfp_id=RESULT_ID,
        artifact_root=tmp_path,
    )
    expected = (
        tmp_path
        / "L14"
        / "20240611"
        / "ripple_band_lfp"
        / "08_r4"
        / str(RESULT_ID)
    )
    assert paths["artifact_dir"] == expected
    assert dict(ripple_band_lfp.MANUSCRIPT_PARAMETERS) == {
        "lowcut_hz": 150.0,
        "highcut_hz": 250.0,
        "filter_order": 4,
        "target_sampling_frequency_hz": 1000.0,
        "enable_notch_filter": False,
        "notch_base_freq_hz": 60.0,
        "notch_harmonics": 10,
        "notch_quality": 50.0,
    }
    assert (
        ripple_band_lfp.OUTPUT_RULE["standard_spyglass_lfp_interchangeable"]
        is False
    )
    assert ripple_band_lfp.OUTPUT_RULE["source_nwb_mutation"] is False


@pytest.mark.parametrize("enable_notch", [False, True])
def test_compute_matches_legacy_helpers_and_preserves_electrode_order(
    enable_notch: bool,
) -> None:
    """Filtering is helper-identical and never sorts the selected electrodes."""
    timestamps, traces = _raw_input()
    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs(enable_notch_filter=enable_notch)
    )
    helper_input = (
        traces.astype(np.float32)
        * np.asarray(GAIN_TO_UV, dtype=np.float32)[None, :]
        + np.asarray(OFFSET_TO_UV, dtype=np.float32)[None, :]
    )
    helper_input = np.asarray(helper_input, dtype=float)
    if enable_notch:
        helper_input = detect_ripples.apply_notch_filters_multichannel(
            helper_input,
            fs=SOURCE_FS,
            base_freq=60.0,
            n_harmonics=10,
            quality=50.0,
        )
    expected_time, expected_lfp, expected_fs = (
        detect_ripples.butter_filter_and_decimate(
            timestamps,
            helper_input,
            SOURCE_FS,
            target_new_sampling_frequency=1000.0,
            lowcut=150.0,
            highcut=250.0,
            order=4,
        )
    )

    assert result["analysis_status"] == "valid"
    assert result["ordered_electrode_ids"] == [201, 17]
    assert result["dataset"]["channel"].values.tolist() == [201, 17]
    assert result["channel_qc"]["electrode_id"].tolist() == [201, 17]
    np.testing.assert_array_equal(result["dataset"]["time"].values, expected_time)
    np.testing.assert_array_equal(
        result["dataset"]["filtered_lfp"].values,
        expected_lfp,
    )
    assert result["actual_sampling_frequency_hz"] == expected_fs == 1024.0
    assert result["ordered_gain_to_uv"] == GAIN_TO_UV
    assert result["ordered_offset_to_uv"] == OFFSET_TO_UV
    assert result["channel_qc"]["gain_to_uV"].tolist() == GAIN_TO_UV
    assert result["dataset"].attrs["filter_input_unit"] == "microvolts"
    assert int(result["dataset"].attrs["notch_filter_enabled"]) == int(
        enable_notch
    )


def test_scaling_matches_si_float32_then_detector_float64_filtering() -> None:
    """Non-exact gains use SI float32 math, then detector float64 input."""
    timestamps, traces = _raw_input()
    gains = [1.0 / 3.0, np.pi / 7.0]
    offsets = [0.1, -0.2]
    si_scaled_float32 = (
        traces.astype(np.float32)
        * np.asarray(gains, dtype=np.float32)[None, :]
        + np.asarray(offsets, dtype=np.float32)[None, :]
    )
    detector_filter_input = np.asarray(si_scaled_float32, dtype=float)
    float64_scaled = (
        traces.astype(float) * np.asarray(gains)[None, :]
        + np.asarray(offsets)[None, :]
    )
    assert np.any(detector_filter_input != float64_scaled)
    expected_time, expected_lfp, expected_fs = (
        detect_ripples.butter_filter_and_decimate(
            timestamps,
            detector_filter_input,
            SOURCE_FS,
            target_new_sampling_frequency=1000.0,
            lowcut=150.0,
            highcut=250.0,
            order=4,
        )
    )
    direct_float32_lfp = detect_ripples.butter_filter_and_decimate(
        timestamps,
        si_scaled_float32,
        SOURCE_FS,
        target_new_sampling_frequency=1000.0,
        lowcut=150.0,
        highcut=250.0,
        order=4,
    )[1]
    assert not np.array_equal(direct_float32_lfp, expected_lfp)

    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs(gain_to_uv=gains, offset_to_uv=offsets)
    )

    np.testing.assert_array_equal(result["dataset"]["time"], expected_time)
    np.testing.assert_array_equal(
        result["dataset"]["filtered_lfp"], expected_lfp
    )
    assert result["actual_sampling_frequency_hz"] == expected_fs
    assert result["trace_scaling_sha256"] == ripple_band_lfp._provenance_sha256(
        {
            "electrode_ids": ELECTRODE_IDS,
            "gain_to_uV": gains,
            "offset_to_uV": offsets,
            "scaling_operation_dtype": "float32",
            "filter_input_dtype": "float64",
        }
    )


def test_database_scalars_normalize_to_python_values() -> None:
    """NumPy/DataJoint scalar values do not leak into database-facing metadata."""
    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs(
            animal_name=b"L14",
            date=np.str_("20240611"),
            electrode_ids=[np.int64(201), np.int64(17)],
            gain_to_uv=[np.float64(2.0), np.float64(0.5)],
            offset_to_uv=[np.float64(10.0), np.float64(-3.0)],
            source_sampling_frequency_hz=np.float64(SOURCE_FS),
            filter_order=np.int64(4),
            enable_notch_filter=np.int64(0),
            notch_harmonics=np.int64(10),
        )
    )
    summary = ripple_band_lfp.summarize_ripple_band_lfp_artifact_bundle(result)

    assert summary["animal_name"] == "L14"
    assert type(summary["n_channels"]) is int
    assert type(summary["input_sample_count"]) is int
    assert type(summary["source_sampling_frequency_hz"]) is float
    assert type(summary["analysis_status"]) is str
    assert result["parameters"]["enable_notch_filter"] is False


def test_raw_input_validation_rejects_channel_and_timestamp_ambiguity() -> None:
    """Only exact int16 samples with unique electrode-table IDs are accepted."""
    with pytest.raises(ValueError, match="int16 dtype"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(raw_traces=_raw_input()[1].astype(float))
        )
    with pytest.raises(ValueError, match="repeated channel_id"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(electrode_ids=[17, 17])
        )
    with pytest.raises(ValueError, match="at least 0"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(electrode_ids=[17, -1])
        )
    timestamps, _ = _raw_input()
    changed = timestamps.copy()
    changed[100] = changed[99]
    with pytest.raises(ValueError, match="strictly increasing"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(raw_timestamps=changed)
        )
    with pytest.raises(ValueError, match="must align exactly"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(raw_timestamps=timestamps[:-1])
        )


def test_epoch_endpoint_average_does_not_override_frozen_source_rate() -> None:
    """Epoch endpoints may imply a different average rate than the SI estimate."""
    timestamps, _ = _raw_input()
    changed = timestamps.copy()
    changed[-1] += 0.01
    endpoint_rate = (len(changed) - 1) / (changed[-1] - changed[0])

    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs(
            raw_timestamps=changed,
            source_slice_provenance=_source_slice_provenance(changed),
        )
    )

    assert endpoint_rate != SOURCE_FS
    assert result["source_sampling_frequency_hz"] == SOURCE_FS


def test_scaling_and_frequency_provenance_are_strict() -> None:
    """Per-channel scaling and the frozen rate estimate cannot drift silently."""
    with pytest.raises(ValueError, match="align exactly"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(gain_to_uv=[1.0])
        )
    timestamps, _ = _raw_input()
    wrong_rate = _sampling_frequency_provenance(timestamps)
    wrong_rate["estimated_sampling_frequency_hz"] = SOURCE_FS + 1.0
    with pytest.raises(ValueError, match="does not match the selected source rate"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(sampling_frequency_provenance=wrong_rate)
        )

    result = ripple_band_lfp.compute_selected_ripple_band_lfp(**_compute_kwargs())
    result["ordered_gain_to_uv"] = [1.0, 0.5]
    with pytest.raises(ValueError, match="trace_scaling_sha256"):
        ripple_band_lfp.validate_ripple_band_lfp_result(result)


def test_hdf5_inspection_freezes_metadata_without_epoch_array_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selection-time inspection reads metadata and the 1,000-time rate sample."""
    path, timestamps, _ = _write_nwb_input_fixture(tmp_path)
    reads: list[tuple[str, object]] = []
    original_read = ripple_band_lfp._read_hdf5_array

    def recording_read(dataset, selection):
        reads.append((str(dataset.name), selection))
        return original_read(dataset, selection)

    monkeypatch.setattr(ripple_band_lfp, "_read_hdf5_array", recording_read)
    snapshot = ripple_band_lfp.inspect_selected_ripple_band_lfp_nwb_inputs(
        nwb_path=path,
        epoch="08_r4",
        electrode_ids=[900, 101],
    )

    assert "raw_timestamps" not in snapshot
    assert "raw_traces" not in snapshot
    assert snapshot["source_nwb_file_name"] == path.name
    assert snapshot["electrode_ids"] == [900, 101]
    assert snapshot["gain_to_uv"] == [4.0, 2.0]
    assert snapshot["offset_to_uv"] == [4.0, 1.0]
    source_slice = snapshot["source_slice_provenance"]
    assert source_slice["epoch"] == "08_r4"
    assert source_slice["interval_table_row_index"] == 0
    assert source_slice["source_start_index"] == 500
    assert source_slice["source_stop_index_exclusive"] == 3500
    assert source_slice["epoch_start_time_s"] == timestamps[500]
    assert source_slice["epoch_stop_time_s"] == timestamps[3499]
    assert source_slice["selected_data_column_indices"] == [2, 1]
    assert source_slice["selected_electrode_table_rows"] == [3, 0]
    assert source_slice["electrodes_region_object_id"] == (
        "electrodes-region-object-id"
    )
    assert len(source_slice["electrodes_region_sha256"]) == 64
    assert len(source_slice["electrodes_table_ids_sha256"]) == 64
    assert not any(name.endswith("/data") for name, _ in reads)
    timestamp_reads = [
        selection
        for name, selection in reads
        if name.endswith("/timestamps")
    ]
    assert timestamp_reads == [slice(0, 1000)]


def test_hdf5_loader_rejects_frozen_snapshot_mutations_before_epoch_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scale, rate, object, and column drift fail before trace materialization."""
    path, _, _ = _write_nwb_input_fixture(tmp_path)
    snapshot = ripple_band_lfp.inspect_selected_ripple_band_lfp_nwb_inputs(
        nwb_path=path,
        epoch="08_r4",
        electrode_ids=[900, 101],
    )
    reads: list[str] = []
    original_read = ripple_band_lfp._read_hdf5_array

    def recording_read(dataset, selection):
        reads.append(str(dataset.name))
        return original_read(dataset, selection)

    monkeypatch.setattr(ripple_band_lfp, "_read_hdf5_array", recording_read)
    mutations = []
    changed_gain = deepcopy(snapshot)
    changed_gain["gain_to_uv"][0] = 4.5
    mutations.append(changed_gain)
    changed_rate_hash = deepcopy(snapshot)
    changed_rate_hash["sampling_frequency_provenance"][
        "reference_timestamps_sha256"
    ] = "f" * 64
    mutations.append(changed_rate_hash)
    changed_object = deepcopy(snapshot)
    changed_object["source_slice_provenance"][
        "electrical_series_object_id"
    ] = "different-electrical-series-object-id"
    mutations.append(changed_object)
    changed_column = deepcopy(snapshot)
    changed_column["source_slice_provenance"][
        "selected_data_column_indices"
    ] = [3, 1]
    mutations.append(changed_column)

    for changed_snapshot in mutations:
        reads.clear()
        with pytest.raises(ValueError, match="snapshot changed after selection"):
            ripple_band_lfp.load_selected_ripple_band_lfp_nwb_inputs(
                nwb_path=path,
                epoch="08_r4",
                electrode_ids=[900, 101],
                expected_snapshot=changed_snapshot,
            )
        assert not any(name.endswith("/data") for name in reads)


def test_hdf5_loader_slices_epoch_maps_ids_and_matches_si_scaling(
    tmp_path: Path,
) -> None:
    """The loader reads one slice and preserves requested NWB electrode-ID order."""
    path, timestamps, raw = _write_nwb_input_fixture(tmp_path)
    snapshot = ripple_band_lfp.inspect_selected_ripple_band_lfp_nwb_inputs(
        nwb_path=path,
        epoch="08_r4",
        electrode_ids=[900, 101],
    )

    loaded = ripple_band_lfp.load_selected_ripple_band_lfp_nwb_inputs(
        nwb_path=path,
        epoch="08_r4",
        electrode_ids=[900, 101],
        expected_snapshot=snapshot,
    )

    assert {
        field_name: loaded[field_name]
        for field_name in snapshot
    } == snapshot
    np.testing.assert_array_equal(loaded["raw_timestamps"], timestamps[500:3500])
    np.testing.assert_array_equal(loaded["raw_traces"], raw[500:3500, [2, 1]])
    assert loaded["electrode_ids"] == [900, 101]
    assert loaded["gain_to_uv"] == [4.0, 2.0]
    assert loaded["offset_to_uv"] == [4.0, 1.0]
    assert loaded["source_slice_provenance"][
        "selected_data_column_indices"
    ] == [2, 1]
    assert loaded["source_slice_provenance"][
        "selected_electrode_table_rows"
    ] == [3, 0]
    expected_source_fs = 1.0 / np.median(np.diff(timestamps[:1000]))
    assert loaded["source_sampling_frequency_hz"] == expected_source_fs
    endpoint_fs = 2999.0 / (timestamps[3499] - timestamps[500])
    assert endpoint_fs != expected_source_fs

    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_band_lfp_id=RESULT_ID,
        **loaded,
    )
    assert result["source_sampling_frequency_hz"] == expected_source_fs
    assert result["input_sample_count"] == 3000


def test_truly_empty_epoch_has_explicit_terminal_artifact() -> None:
    """Only a truly empty aligned epoch becomes a terminal result."""
    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs(
            raw_timestamps=np.asarray([], dtype=float),
            raw_traces=np.empty((0, 2), dtype=np.int16),
            source_slice_provenance=_source_slice_provenance(
                np.asarray([], dtype=float)
            ),
        )
    )

    assert result["analysis_status"] == "empty_input"
    assert result["input_sample_count"] == 0
    assert result["output_sample_count"] == 0
    assert result["dataset"]["filtered_lfp"].shape == (0, 2)
    assert result["channel_qc"]["analysis_status"].tolist() == [
        "empty_input",
        "empty_input",
    ]
    with pytest.raises(ValueError, match="at least two timestamps"):
        ripple_band_lfp.compute_selected_ripple_band_lfp(
            **_compute_kwargs(
                raw_timestamps=np.asarray([1.0]),
                raw_traces=np.zeros((1, 2), dtype=np.int16),
            )
        )


def test_write_load_checksum_and_no_overwrite(tmp_path: Path) -> None:
    """The NetCDF/QC bundle round trips and tampering is detected."""
    result = ripple_band_lfp.compute_selected_ripple_band_lfp(**_compute_kwargs())
    destination = tmp_path / str(RESULT_ID)
    paths = ripple_band_lfp.write_ripple_band_lfp_artifact(result, destination)
    loaded = ripple_band_lfp.load_ripple_band_lfp_artifact(destination)

    assert loaded["manifest"]["artifact_key"].tolist() == [
        "ripple_band_lfp",
        "channel_qc",
    ]
    np.testing.assert_array_equal(
        loaded["dataset"]["filtered_lfp"].values,
        result["dataset"]["filtered_lfp"].values,
    )
    pd.testing.assert_frame_equal(
        loaded["channel_qc"],
        result["channel_qc"],
        check_dtype=False,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        ripple_band_lfp.write_ripple_band_lfp_artifact(result, destination)

    qc = pd.read_parquet(paths["channel_qc_path"])
    qc.loc[0, "electrode_id"] = 99
    qc.to_parquet(paths["channel_qc_path"], index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        ripple_band_lfp.load_ripple_band_lfp_artifact(destination)


def test_artifact_origin_requires_matching_legacy_provenance() -> None:
    """Computed and registered bundles cannot spoof each other's provenance."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs()
    )
    computed["legacy_artifact_provenance"] = {"source": "legacy.nc"}
    with pytest.raises(ValueError, match="cannot contain legacy provenance"):
        ripple_band_lfp.validate_ripple_band_lfp_result(computed)

    registered = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs(
            artifact_origin="registered_existing",
            legacy_artifact_provenance={"source": "legacy.nc"},
        )
    )
    registered["legacy_artifact_provenance"] = {}
    with pytest.raises(ValueError, match="require legacy provenance"):
        ripple_band_lfp.validate_ripple_band_lfp_result(registered)


def test_result_semantics_reject_count_and_timestamp_mutations() -> None:
    """Stride counts and output time support remain tied to raw input."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs()
    )

    wrong_count = deepcopy(computed)
    wrong_count["dataset"] = wrong_count["dataset"].isel(
        sample=slice(None, -1)
    )
    wrong_count["output_sample_count"] -= 1
    wrong_count["channel_qc"] = wrong_count["channel_qc"].copy()
    wrong_count["channel_qc"]["output_sample_count"] -= 1
    with pytest.raises(ValueError, match=r"ceil\(input samples / stride\)"):
        ripple_band_lfp.validate_ripple_band_lfp_result(wrong_count)

    wrong_start = deepcopy(computed)
    wrong_start["dataset"] = wrong_start["dataset"].copy(deep=True)
    wrong_start["dataset"]["time"].values[0] += 1e-6
    with pytest.raises(ValueError, match="first filtered timestamp"):
        ripple_band_lfp.validate_ripple_band_lfp_result(wrong_start)

    outside_epoch = deepcopy(computed)
    outside_epoch["dataset"] = outside_epoch["dataset"].copy(deep=True)
    outside_epoch["dataset"]["time"].values[-1] = (
        outside_epoch["input_stop_time_s"] + 1e-6
    )
    with pytest.raises(ValueError, match="within the raw epoch input"):
        ripple_band_lfp.validate_ripple_band_lfp_result(outside_epoch)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("bundle_schema_version", "999", "bundle_schema_version"),
        ("n_channels", 99, "n_channels"),
        ("output_start_time_s", 1.25, "output_start_time_s"),
        ("output_stop_time_s", 2.5, "output_stop_time_s"),
    ],
)
def test_manifest_semantics_reject_mutations(
    tmp_path: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    """Manifest summaries cannot drift from the checksummed result dataset."""
    result = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs()
    )
    destination = tmp_path / field_name / str(RESULT_ID)
    paths = ripple_band_lfp.write_ripple_band_lfp_artifact(
        result, destination
    )
    manifest = pd.read_parquet(paths["artifact_manifest_path"])
    manifest[field_name] = value
    manifest.to_parquet(paths["artifact_manifest_path"], index=False)

    with pytest.raises(ValueError, match=message):
        ripple_band_lfp.load_ripple_band_lfp_artifact(destination)


def test_strict_legacy_registration_recomputes_exact_nwb_input(
    tmp_path: Path,
) -> None:
    """A complete matching legacy epoch cache registers with source provenance."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(**_compute_kwargs())
    source = tmp_path / "08_r4_ripple_channels_lfp.nc"
    _legacy_dataset(computed).to_netcdf(source, engine="scipy")
    run_log = tmp_path / "run_log.json"
    run_log.write_text(
        json.dumps(_legacy_run_log_payload(source)),
        encoding="utf-8",
    )
    destination = tmp_path / "registered" / str(RESULT_ID)

    registered = ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
        source_result_path=source,
        source_run_log_path=run_log,
        destination_path=destination,
        **_compute_kwargs(),
    )

    assert registered["artifact_origin"] == "registered_existing"
    assert registered["legacy_artifact_provenance"][
        "source_v1ca1_git_commit"
    ] == "abc123"
    assert registered["legacy_artifact_provenance"]["source_epoch"] == "08_r4"
    assert len(registered["_created_artifact_paths"]) == 3
    assert not list(destination.rglob("*.js"))


def test_legacy_run_log_requires_exact_channels_filter_and_path(
    tmp_path: Path,
) -> None:
    """A supplied detector log must identify the exact legacy cache bytes."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs()
    )
    source = tmp_path / "08_r4_ripple_channels_lfp.nc"
    _legacy_dataset(computed).to_netcdf(source, engine="scipy")
    mutations = []
    changed = _legacy_run_log_payload(source)
    changed["parameters"]["ripple_channels"] = list(reversed(ELECTRODE_IDS))
    mutations.append((changed, "electrode order"))
    changed = _legacy_run_log_payload(source)
    changed["parameters"]["notch_quality"] = 49.0
    mutations.append((changed, "notch_quality"))
    changed = _legacy_run_log_payload(source)
    changed["outputs"]["saved_lfp_cache_epoch_paths"]["08_r4"] = str(
        tmp_path / "different.nc"
    )
    mutations.append((changed, "different epoch cache"))

    for index, (payload, message) in enumerate(mutations):
        run_log = tmp_path / f"run_log_{index}.json"
        run_log.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
                source_result_path=source,
                source_run_log_path=run_log,
                destination_path=tmp_path / f"registered_{index}" / str(RESULT_ID),
                **_compute_kwargs(),
            )


def test_legacy_registration_rejects_non_detector_filter_parameters(
    tmp_path: Path,
) -> None:
    """Legacy caches cannot be registered under a custom filter definition."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs()
    )
    source = tmp_path / "legacy.nc"
    _legacy_dataset(computed).to_netcdf(source, engine="scipy")
    with pytest.raises(ValueError, match="filter_order differs"):
        ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
            source_result_path=source,
            destination_path=tmp_path / "registered" / str(RESULT_ID),
            **_compute_kwargs(filter_order=3),
        )


def test_legacy_registration_rejects_netcdf_toctou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation and recorded checksum cannot refer to different file bytes."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(
        **_compute_kwargs()
    )
    source = tmp_path / "legacy.nc"
    _legacy_dataset(computed).to_netcdf(source, engine="scipy")
    original_load = ripple_band_lfp._load_dataset

    def load_then_mutate(path):
        loaded = original_load(path)
        with Path(path).open("ab") as stream:
            stream.write(b"changed-after-load")
        return loaded

    monkeypatch.setattr(ripple_band_lfp, "_load_dataset", load_then_mutate)
    with pytest.raises(ValueError, match="changed during validation"):
        ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
            source_result_path=source,
            destination_path=tmp_path / "registered" / str(RESULT_ID),
            **_compute_kwargs(),
        )


def test_legacy_registration_rejects_channel_value_and_metadata_drift(
    tmp_path: Path,
) -> None:
    """No mismatched legacy cache can be blessed as a canonical NWB result."""
    computed = ripple_band_lfp.compute_selected_ripple_band_lfp(**_compute_kwargs())
    legacy = _legacy_dataset(computed)

    wrong_channels = legacy.assign_coords(channel=("channel", [17, 201]))
    wrong_channel_path = tmp_path / "wrong_channels.nc"
    wrong_channels.to_netcdf(wrong_channel_path, engine="scipy")
    with pytest.raises(ValueError, match="electrode order"):
        ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
            source_result_path=wrong_channel_path,
            destination_path=tmp_path / "channel" / str(RESULT_ID),
            **_compute_kwargs(),
        )

    changed = legacy.copy(deep=True)
    changed["filtered_lfp"].values[10, 0] += 1.0
    changed_path = tmp_path / "changed.nc"
    changed.to_netcdf(changed_path, engine="scipy")
    with pytest.raises(ValueError, match="scientific values"):
        ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
            source_result_path=changed_path,
            destination_path=tmp_path / "value" / str(RESULT_ID),
            **_compute_kwargs(),
        )

    wrong_metadata = legacy.copy(deep=True)
    wrong_metadata.attrs["epoch"] = "06_r3"
    wrong_metadata_path = tmp_path / "wrong_metadata.nc"
    wrong_metadata.to_netcdf(wrong_metadata_path, engine="scipy")
    with pytest.raises(ValueError, match="attribute 'epoch'"):
        ripple_band_lfp.register_existing_ripple_band_lfp_artifact(
            source_result_path=wrong_metadata_path,
            destination_path=tmp_path / "metadata" / str(RESULT_ID),
            **_compute_kwargs(),
        )
