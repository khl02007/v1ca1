"""Database-free epoch ripple-band LFP artifact bundles.

The scientific transform remains the one used by
``v1ca1.ripple.detect_ripples``: optional stacked legacy notch filters,
followed by a fourth-order 150--250 Hz Butterworth bandpass and stride
decimation toward 1 kHz.  This is intentionally not interchangeable with
Spyglass's standard FIR/two-stage LFP pipeline.

Inputs are explicit slices from the raw NWB acquisition ElectricalSeries.
Columns are identified by ordered NWB electrodes-table IDs, not by the
repeated ``channel_id`` metadata column.  That order is immutable because the
paper figure displays the first selected channel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.ripple import detect_ripples


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "ripple_band_lfp"
MANIFEST_FILENAME = "manifest.parquet"
QC_FILENAME = "channel_qc.parquet"
RESULT_FILENAME = "ripple_band_lfp.nc"
BUNDLE_SCHEMA_VERSION = "1"
SAMPLING_FREQUENCY_ESTIMATION_METHOD = (
    "spikeinterface_nwb_first_timestamp_differences_median"
)
DEFAULT_SAMPLES_FOR_RATE_ESTIMATION = 1000

DEFAULT_LOWCUT_HZ = detect_ripples.DEFAULT_LOWCUT_HZ
DEFAULT_HIGHCUT_HZ = detect_ripples.DEFAULT_HIGHCUT_HZ
DEFAULT_FILTER_ORDER = detect_ripples.DEFAULT_FILTER_ORDER
DEFAULT_TARGET_SAMPLING_FREQUENCY_HZ = (
    detect_ripples.DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY
)
DEFAULT_ENABLE_NOTCH_FILTER = detect_ripples.DEFAULT_ENABLE_NOTCH_FILTER
DEFAULT_NOTCH_BASE_FREQ_HZ = detect_ripples.DEFAULT_NOTCH_BASE_FREQ
DEFAULT_NOTCH_HARMONICS = detect_ripples.DEFAULT_NOTCH_HARMONICS
DEFAULT_NOTCH_QUALITY = detect_ripples.DEFAULT_NOTCH_QUALITY

MANUSCRIPT_PARAMETERS = MappingProxyType(
    {
        "lowcut_hz": float(DEFAULT_LOWCUT_HZ),
        "highcut_hz": float(DEFAULT_HIGHCUT_HZ),
        "filter_order": int(DEFAULT_FILTER_ORDER),
        "target_sampling_frequency_hz": float(
            DEFAULT_TARGET_SAMPLING_FREQUENCY_HZ
        ),
        "enable_notch_filter": bool(DEFAULT_ENABLE_NOTCH_FILTER),
        "notch_base_freq_hz": float(DEFAULT_NOTCH_BASE_FREQ_HZ),
        "notch_harmonics": int(DEFAULT_NOTCH_HARMONICS),
        "notch_quality": float(DEFAULT_NOTCH_QUALITY),
    }
)

OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "row_granularity": "one_session_epoch",
        "source": "raw_nwb_acquisition_electrical_series_int16",
        "time_source": "explicit_nwb_ephys_timestamps_seconds",
        "channel_identity": "ordered_nwb_electrodes_table_id",
        "electrode_membership_validation": "upstream_selected_nwb_loader",
        "channel_order_policy": "preserve_exactly_first_channel_is_figure_channel",
        "notch_policy": "optional_legacy_iirnotch_stack_before_bandpass",
        "bandpass_implementation": "detect_ripples.butter_filter_and_decimate",
        "default_band_hz": (150.0, 250.0),
        "default_filter_order": 4,
        "default_target_sampling_frequency_hz": 1000.0,
        "decimation": "integer_stride_round_source_fs_over_target_fs",
        "actual_sampling_frequency": "source_fs_divided_by_stride",
        "voltage_scaling": (
            "spikeinterface_0_103_2_float32_scaling_then_float64_filter_input"
        ),
        "registered_raw_identity": (
            "datajoint_filepath_contents_hash_and_size"
        ),
        "in_place_mutation_outside_datajoint": "unsupported",
        "sampling_frequency_estimation": SAMPLING_FREQUENCY_ESTIMATION_METHOD,
        "standard_spyglass_lfp_interchangeable": False,
        "source_nwb_mutation": False,
        "time_unit": "s",
    }
)

ANALYSIS_STATUSES = ("valid", "empty_input")
QC_COLUMNS = (
    "ripple_band_lfp_id",
    "animal_name",
    "date",
    "epoch",
    "channel_index",
    "electrode_id",
    "input_sample_count",
    "output_sample_count",
    "input_dtype",
    "gain_to_uV",
    "offset_to_uV",
    "source_sampling_frequency_hz",
    "actual_sampling_frequency_hz",
    "analysis_status",
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "ripple_band_lfp_id",
    "animal_name",
    "date",
    "epoch",
    "source_nwb_file_name",
    "source_electrical_series_path",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "upstream_provenance_json",
    "sampling_frequency_provenance_json",
    "sampling_frequency_provenance_sha256",
    "source_slice_provenance_json",
    "source_slice_provenance_sha256",
    "ordered_electrode_ids_json",
    "ordered_gain_to_uV_json",
    "ordered_offset_to_uV_json",
    "trace_scaling_sha256",
    "raw_timestamps_sha256",
    "raw_traces_sha256",
    "n_channels",
    "input_sample_count",
    "output_sample_count",
    "input_start_time_s",
    "input_stop_time_s",
    "output_start_time_s",
    "output_stop_time_s",
    "source_sampling_frequency_hz",
    "actual_sampling_frequency_hz",
    "decimation_factor",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "bundle_schema_version",
)


def _python_scalar(value: Any) -> Any:
    """Normalize NumPy/database scalars and byte strings to Python values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _text(value: Any, *, name: str) -> str:
    """Return one non-empty normalized database text scalar."""
    text = str(_python_scalar(value)).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe non-empty path component."""
    component = _text(value, name=name)
    if Path(component).name != component or component in {".", ".."}:
        raise ValueError(f"{name} must be one safe path component.")
    return component


def _uuid_string(value: Any, *, name: str) -> str:
    """Return one canonical UUID string from a database scalar."""
    try:
        return str(uuid.UUID(_text(value, name=name)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def _database_bool(value: Any, *, name: str) -> bool:
    """Normalize bool and database integer 0/1 without truthy coercion."""
    value = _python_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(int(value))
    raise TypeError(f"{name} must be a bool or database integer 0/1.")


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    """Return one normalized database integer at or above a lower bound."""
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _finite_float(value: Any, *, name: str, positive: bool = False) -> float:
    """Return one finite normalized database numeric scalar."""
    value = _python_scalar(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _json_safe(value: Any) -> Any:
    """Return one recursively normalized JSON-compatible value."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return _python_scalar(value)


def _provenance_sha256(value: Any) -> str:
    """Return a stable digest for JSON-compatible provenance."""
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    """Return a dtype-, shape-, and byte-sensitive array digest."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    """Return a streaming file digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    """Return fields that reveal replacement or mutation during a read."""
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _stable_file_fingerprint(path: Path) -> dict[str, Any]:
    """Hash a file and reject replacement or mutation during that hash."""
    file_path = Path(path)
    before = file_path.stat()
    digest = _file_sha256(file_path)
    after = file_path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"File changed while it was being hashed: {file_path}")
    return {
        "sha256": digest,
        "stat_identity": _stat_identity(after),
    }


def _read_stable_json(path: Path) -> tuple[dict[str, Any], str]:
    """Parse JSON from one stable byte read and return that exact byte digest."""
    file_path = Path(path)
    before = file_path.stat()
    payload_bytes = file_path.read_bytes()
    after = file_path.stat()
    if _stat_identity(before) != _stat_identity(after) or len(
        payload_bytes
    ) != after.st_size:
        raise ValueError(f"JSON file changed while it was read: {file_path}")
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON file is unreadable: {file_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file must contain one object: {file_path}")
    return dict(payload), hashlib.sha256(payload_bytes).hexdigest()


OUTPUT_RULE_SHA256 = _provenance_sha256(dict(OUTPUT_RULE))


def get_ripple_band_lfp_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_band_lfp_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first epoch artifact bundle."""
    animal = _path_component(animal_name, name="animal_name")
    session_date = _path_component(date, name="date")
    epoch_name = _path_component(epoch, name="epoch")
    result_id = _uuid_string(ripple_band_lfp_id, name="ripple_band_lfp_id")
    directory = (
        Path(artifact_root)
        / animal
        / session_date
        / ARTIFACT_DIRNAME
        / epoch_name
        / result_id
    )
    return _paths_for_directory(directory)


def _paths_for_directory(directory: Path) -> dict[str, Path]:
    """Return canonical child paths for one artifact directory."""
    directory = Path(directory)
    return {
        "artifact_dir": directory,
        "artifact_manifest_path": directory / MANIFEST_FILENAME,
        "channel_qc_path": directory / QC_FILENAME,
        "result_path": directory / RESULT_FILENAME,
    }


def validate_ripple_band_lfp_parameters(
    *,
    source_sampling_frequency_hz: Any,
    lowcut_hz: Any = DEFAULT_LOWCUT_HZ,
    highcut_hz: Any = DEFAULT_HIGHCUT_HZ,
    filter_order: Any = DEFAULT_FILTER_ORDER,
    target_sampling_frequency_hz: Any = DEFAULT_TARGET_SAMPLING_FREQUENCY_HZ,
    enable_notch_filter: Any = DEFAULT_ENABLE_NOTCH_FILTER,
    notch_base_freq_hz: Any = DEFAULT_NOTCH_BASE_FREQ_HZ,
    notch_harmonics: Any = DEFAULT_NOTCH_HARMONICS,
    notch_quality: Any = DEFAULT_NOTCH_QUALITY,
) -> dict[str, Any]:
    """Return validated legacy-filter parameters and the resulting stride."""
    source_fs = _finite_float(
        source_sampling_frequency_hz,
        name="source_sampling_frequency_hz",
        positive=True,
    )
    lowcut = _finite_float(lowcut_hz, name="lowcut_hz", positive=True)
    highcut = _finite_float(highcut_hz, name="highcut_hz", positive=True)
    order = _integer(filter_order, name="filter_order", minimum=1)
    target_fs = _finite_float(
        target_sampling_frequency_hz,
        name="target_sampling_frequency_hz",
        positive=True,
    )
    notch_enabled = _database_bool(
        enable_notch_filter,
        name="enable_notch_filter",
    )
    notch_base = _finite_float(
        notch_base_freq_hz,
        name="notch_base_freq_hz",
        positive=True,
    )
    harmonics = _integer(notch_harmonics, name="notch_harmonics", minimum=1)
    quality = _finite_float(
        notch_quality,
        name="notch_quality",
        positive=True,
    )
    if highcut <= lowcut:
        raise ValueError("highcut_hz must exceed lowcut_hz.")
    if highcut >= source_fs / 2.0:
        raise ValueError("highcut_hz must be below the source Nyquist frequency.")
    if target_fs > source_fs:
        raise ValueError(
            "target_sampling_frequency_hz cannot exceed source sampling frequency."
        )
    stride = max(int(round(source_fs / target_fs)), 1)
    actual_fs = source_fs / stride
    if highcut >= actual_fs / 2.0:
        raise ValueError(
            "highcut_hz must be below the actual decimated Nyquist frequency."
        )
    return {
        "source_sampling_frequency_hz": source_fs,
        "lowcut_hz": lowcut,
        "highcut_hz": highcut,
        "filter_order": order,
        "target_sampling_frequency_hz": target_fs,
        "enable_notch_filter": notch_enabled,
        "notch_base_freq_hz": notch_base,
        "notch_harmonics": harmonics,
        "notch_quality": quality,
        "decimation_factor": stride,
        "actual_sampling_frequency_hz": actual_fs,
    }


def _parameter_snapshot(
    *,
    parameter_name: Any,
    parameters: Mapping[str, Any],
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
) -> dict[str, Any]:
    """Return one immutable named parameter snapshot."""
    name = _text(parameter_name, name="parameter_name")
    if len(name) > 64:
        raise ValueError("parameter_name must be at most 64 characters.")
    scientific = {
        key: parameters[key]
        for key in (
            "lowcut_hz",
            "highcut_hz",
            "filter_order",
            "target_sampling_frequency_hz",
            "enable_notch_filter",
            "notch_base_freq_hz",
            "notch_harmonics",
            "notch_quality",
        )
    }
    expected_parameter_sha256 = _provenance_sha256(
        {"ripple_band_lfp_param_name": name, **scientific}
    )
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    if _text(parameter_sha256, name="parameter_sha256") != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    if output_rule_sha256 is None:
        output_rule_sha256 = OUTPUT_RULE_SHA256
    if _text(output_rule_sha256, name="output_rule_sha256") != OUTPUT_RULE_SHA256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": name,
        "parameter_sha256": expected_parameter_sha256,
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        **scientific,
    }


def _electrode_ids(values: Sequence[Any]) -> np.ndarray:
    """Return ordered unique NWB electrodes-table IDs in the valid range."""
    if isinstance(values, (str, bytes)):
        raise TypeError("electrode_ids must be an ordered integer sequence.")
    result = np.asarray(
        [_integer(value, name="electrode_id", minimum=0) for value in values],
        dtype=np.int64,
    )
    if result.size == 0:
        raise ValueError("At least one NWB electrode ID is required.")
    if np.unique(result).size != result.size:
        raise ValueError(
            "NWB electrode IDs must be unique; do not use repeated channel_id values."
        )
    return result


def _validate_raw_inputs(
    *,
    raw_timestamps: Any,
    raw_traces: Any,
    electrode_ids: Sequence[Any],
) -> dict[str, Any]:
    """Return exact raw acquisition arrays without reordering or coercing traces."""
    timestamps = np.asarray(raw_timestamps, dtype=float)
    traces = np.asarray(raw_traces)
    electrodes = _electrode_ids(electrode_ids)
    if timestamps.ndim != 1:
        raise ValueError("raw_timestamps must be one-dimensional.")
    if traces.ndim != 2:
        raise ValueError("raw_traces must have shape (sample, electrode).")
    if traces.dtype != np.dtype(np.int16):
        raise ValueError(
            "raw_traces must preserve the acquisition ElectricalSeries int16 dtype."
        )
    if traces.shape[0] != timestamps.size:
        raise ValueError("raw traces and explicit ephys timestamps must align exactly.")
    if traces.shape[1] != electrodes.size:
        raise ValueError("raw trace columns must match the ordered electrode IDs.")
    if timestamps.size == 1:
        raise ValueError("A nonempty raw epoch must contain at least two timestamps.")
    if timestamps.size:
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("raw_timestamps must be finite seconds.")
        differences = np.diff(timestamps)
        if np.any(differences <= 0.0):
            raise ValueError("raw_timestamps must be strictly increasing.")
    return {
        "timestamps": timestamps,
        "traces": traces,
        "electrode_ids": electrodes,
        "raw_timestamps_sha256": _array_sha256(timestamps),
        "raw_traces_sha256": _array_sha256(traces),
    }


def _ordered_voltage_scaling(
    *,
    gain_to_uv: Sequence[Any],
    offset_to_uv: Sequence[Any],
    electrode_ids: np.ndarray,
) -> dict[str, Any]:
    """Return finite per-electrode scaling aligned to the selected order."""
    gains = np.asarray(
        [
            _finite_float(value, name="gain_to_uv", positive=True)
            for value in gain_to_uv
        ],
        dtype=float,
    )
    offsets = np.asarray(
        [_finite_float(value, name="offset_to_uv") for value in offset_to_uv],
        dtype=float,
    )
    if gains.shape != electrode_ids.shape or offsets.shape != electrode_ids.shape:
        raise ValueError(
            "gain_to_uv and offset_to_uv must align exactly to ordered electrode_ids."
        )
    payload = {
        "electrode_ids": electrode_ids.astype(int).tolist(),
        "gain_to_uV": gains.tolist(),
        "offset_to_uV": offsets.tolist(),
        "scaling_operation_dtype": "float32",
        "filter_input_dtype": "float64",
    }
    return {
        "gain_to_uv": gains,
        "offset_to_uv": offsets,
        "trace_scaling_sha256": _provenance_sha256(payload),
    }


def _sampling_frequency_snapshot(
    value: Mapping[str, Any],
    *,
    source_sampling_frequency_hz: float,
) -> dict[str, Any]:
    """Validate the exact legacy SpikeInterface frequency-estimator snapshot."""
    if not isinstance(value, Mapping):
        raise TypeError("sampling_frequency_provenance must be a mapping.")
    expected_fields = {
        "method",
        "samples_for_rate_estimation",
        "reference_start_index",
        "reference_stop_index_exclusive",
        "reference_timestamps_sha256",
        "estimated_sampling_frequency_hz",
    }
    if set(value) != expected_fields:
        raise ValueError(
            "sampling_frequency_provenance does not have the canonical fields."
        )
    method = _text(value["method"], name="sampling_frequency method")
    if method != SAMPLING_FREQUENCY_ESTIMATION_METHOD:
        raise ValueError("Unsupported sampling-frequency estimation method.")
    sample_count = _integer(
        value["samples_for_rate_estimation"],
        name="samples_for_rate_estimation",
        minimum=2,
    )
    start = _integer(
        value["reference_start_index"],
        name="reference_start_index",
    )
    stop = _integer(
        value["reference_stop_index_exclusive"],
        name="reference_stop_index_exclusive",
    )
    if start != 0 or stop != sample_count:
        raise ValueError(
            "Sampling-frequency reference must be the first source timestamps."
        )
    digest = _text(
        value["reference_timestamps_sha256"],
        name="reference_timestamps_sha256",
    )
    if len(digest) != 64:
        raise ValueError("reference_timestamps_sha256 must be a SHA-256 digest.")
    estimated = _finite_float(
        value["estimated_sampling_frequency_hz"],
        name="estimated_sampling_frequency_hz",
        positive=True,
    )
    if not np.isclose(
        estimated,
        source_sampling_frequency_hz,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Sampling-frequency provenance does not match the selected source rate."
        )
    return {
        "method": method,
        "samples_for_rate_estimation": sample_count,
        "reference_start_index": start,
        "reference_stop_index_exclusive": stop,
        "reference_timestamps_sha256": digest,
        "estimated_sampling_frequency_hz": estimated,
    }


_SOURCE_SLICE_FIELDS = {
    "epoch",
    "data_path",
    "timestamps_path",
    "electrodes_path",
    "interval_table_path",
    "electrodes_table_path",
    "interval_table_row_index",
    "source_start_index",
    "source_stop_index_exclusive",
    "epoch_start_time_s",
    "epoch_stop_time_s",
    "electrical_series_object_id",
    "electrodes_region_object_id",
    "electrodes_table_object_id",
    "interval_table_object_id",
    "electrodes_region_sha256",
    "electrodes_table_ids_sha256",
    "selected_data_column_indices",
    "selected_electrode_table_rows",
}


def _source_slice_snapshot(
    value: Mapping[str, Any],
    *,
    epoch: str,
    source_electrical_series_path: str,
    input_sample_count: int,
    input_start_time_s: float | None,
    input_stop_time_s: float | None,
    electrode_ids: np.ndarray,
) -> dict[str, Any]:
    """Validate the exact source paths, bounds, and electrode-column mapping."""
    if not isinstance(value, Mapping):
        raise TypeError("source_slice_provenance must be a mapping.")
    if set(value) != _SOURCE_SLICE_FIELDS:
        raise ValueError("source_slice_provenance does not have canonical fields.")
    normalized = {
        name: _text(value[name], name=name)
        for name in (
            "epoch",
            "data_path",
            "timestamps_path",
            "electrodes_path",
            "interval_table_path",
            "electrodes_table_path",
            "electrical_series_object_id",
            "electrodes_region_object_id",
            "electrodes_table_object_id",
            "interval_table_object_id",
            "electrodes_region_sha256",
            "electrodes_table_ids_sha256",
        )
    }
    if normalized["epoch"] != epoch:
        raise ValueError("source_slice_provenance epoch is inconsistent.")
    for digest_name in (
        "electrodes_region_sha256",
        "electrodes_table_ids_sha256",
    ):
        if len(normalized[digest_name]) != 64:
            raise ValueError(f"{digest_name} must be a SHA-256 digest.")
    expected_paths = {
        "data_path": f"{source_electrical_series_path}/data",
        "timestamps_path": f"{source_electrical_series_path}/timestamps",
        "electrodes_path": f"{source_electrical_series_path}/electrodes",
    }
    for name, expected in expected_paths.items():
        if normalized[name] != expected:
            raise ValueError(f"source_slice_provenance {name} is inconsistent.")
    start = _integer(value["source_start_index"], name="source_start_index")
    interval_row = _integer(
        value["interval_table_row_index"],
        name="interval_table_row_index",
    )
    stop = _integer(
        value["source_stop_index_exclusive"],
        name="source_stop_index_exclusive",
    )
    if stop < start or stop - start != input_sample_count:
        raise ValueError("Source slice indices do not match raw epoch samples.")
    columns = np.asarray(
        [
            _integer(item, name="selected_data_column_index")
            for item in value["selected_data_column_indices"]
        ],
        dtype=np.int64,
    )
    table_rows = np.asarray(
        [
            _integer(item, name="selected_electrode_table_row")
            for item in value["selected_electrode_table_rows"]
        ],
        dtype=np.int64,
    )
    if columns.shape != electrode_ids.shape or table_rows.shape != electrode_ids.shape:
        raise ValueError("Source electrode mappings must align to electrode_ids.")
    if (
        np.unique(columns).size != columns.size
        or np.unique(table_rows).size != table_rows.size
    ):
        raise ValueError("Source electrode mappings must be unique.")
    epoch_start = value["epoch_start_time_s"]
    epoch_stop = value["epoch_stop_time_s"]
    if input_sample_count:
        epoch_start = _finite_float(epoch_start, name="epoch_start_time_s")
        epoch_stop = _finite_float(epoch_stop, name="epoch_stop_time_s")
        if input_start_time_s != epoch_start or input_stop_time_s != epoch_stop:
            raise ValueError("Source epoch boundaries do not match exact timestamps.")
    elif epoch_start is not None or epoch_stop is not None:
        raise ValueError("Empty source slices require null epoch boundaries.")
    return {
        **normalized,
        "interval_table_row_index": interval_row,
        "source_start_index": start,
        "source_stop_index_exclusive": stop,
        "epoch_start_time_s": epoch_start,
        "epoch_stop_time_s": epoch_stop,
        "selected_data_column_indices": columns.astype(int).tolist(),
        "selected_electrode_table_rows": table_rows.astype(int).tolist(),
    }


def _absolute_hdf5_path(value: Any, *, name: str) -> str:
    """Return one normalized absolute HDF5 object path."""
    path = _text(value, name=name)
    if not path.startswith("/") or path == "/" or "//" in path:
        raise ValueError(f"{name} must be an absolute HDF5 object path.")
    return path.rstrip("/")


def _hdf5_searchsorted(dataset: Any, value: float, *, side: str) -> int:
    """Binary-search one sorted HDF5 dataset without materializing it."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    lower = 0
    upper = int(dataset.shape[0])
    while lower < upper:
        middle = (lower + upper) // 2
        middle_value = float(dataset[middle])
        move_right = middle_value < value or (
            side == "right" and middle_value == value
        )
        if move_right:
            lower = middle + 1
        else:
            upper = middle
    return lower


def _hdf5_object_id(value: Any, *, object_path: str) -> str:
    """Return the required NWB object_id from one HDF5 group or dataset."""
    try:
        object_id = value.attrs["object_id"]
    except KeyError as exc:
        raise ValueError(f"NWB object {object_path!r} is missing object_id.") from exc
    return _text(object_id, name=f"{object_path} object_id")


def _decode_hdf5_text(values: Any) -> list[str]:
    """Decode one one-dimensional NWB text column."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("NWB text columns must be one-dimensional.")
    return [str(_python_scalar(value)) for value in array.tolist()]


def _read_hdf5_array(dataset: Any, selection: Any) -> np.ndarray:
    """Materialize one explicit HDF5 selection.

    Keeping bulk reads behind this helper makes the metadata-only inspection
    contract testable. Scalar binary-search probes intentionally remain direct
    dataset reads.
    """
    return np.asarray(dataset[selection])


_NWB_INPUT_SNAPSHOT_FIELDS = {
    "source_nwb_file_name",
    "source_electrical_series_path",
    "electrode_ids",
    "gain_to_uv",
    "offset_to_uv",
    "source_sampling_frequency_hz",
    "sampling_frequency_provenance",
    "source_slice_provenance",
}


def _normalize_nwb_input_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical metadata-only NWB input snapshot."""
    if not isinstance(value, Mapping):
        raise TypeError("expected_snapshot must be a mapping.")
    if set(value) != _NWB_INPUT_SNAPSHOT_FIELDS:
        raise ValueError("NWB input snapshot does not have canonical fields.")
    nwb_file_name = _path_component(
        value["source_nwb_file_name"],
        name="source_nwb_file_name",
    )
    series_path = _absolute_hdf5_path(
        value["source_electrical_series_path"],
        name="source_electrical_series_path",
    )
    electrodes = _electrode_ids(value["electrode_ids"])
    scaling = _ordered_voltage_scaling(
        gain_to_uv=value["gain_to_uv"],
        offset_to_uv=value["offset_to_uv"],
        electrode_ids=electrodes,
    )
    source_fs = _finite_float(
        value["source_sampling_frequency_hz"],
        name="source_sampling_frequency_hz",
        positive=True,
    )
    frequency_snapshot = _sampling_frequency_snapshot(
        value["sampling_frequency_provenance"],
        source_sampling_frequency_hz=source_fs,
    )
    raw_slice = value["source_slice_provenance"]
    if not isinstance(raw_slice, Mapping):
        raise TypeError("source_slice_provenance must be a mapping.")
    start_index = _integer(
        raw_slice.get("source_start_index"),
        name="source_start_index",
    )
    stop_index = _integer(
        raw_slice.get("source_stop_index_exclusive"),
        name="source_stop_index_exclusive",
    )
    source_slice = _source_slice_snapshot(
        raw_slice,
        epoch=_text(raw_slice.get("epoch"), name="source slice epoch"),
        source_electrical_series_path=series_path,
        input_sample_count=stop_index - start_index,
        input_start_time_s=raw_slice.get("epoch_start_time_s"),
        input_stop_time_s=raw_slice.get("epoch_stop_time_s"),
        electrode_ids=electrodes,
    )
    return {
        "source_nwb_file_name": nwb_file_name,
        "source_electrical_series_path": series_path,
        "electrode_ids": electrodes.astype(int).tolist(),
        "gain_to_uv": scaling["gain_to_uv"].tolist(),
        "offset_to_uv": scaling["offset_to_uv"].tolist(),
        "source_sampling_frequency_hz": source_fs,
        "sampling_frequency_provenance": frequency_snapshot,
        "source_slice_provenance": source_slice,
    }


def _normalize_nwb_input_request(
    *,
    nwb_path: Path,
    epoch: Any,
    electrode_ids: Sequence[Any],
    source_electrical_series_path: Any,
    interval_table_path: Any,
    samples_for_rate_estimation: Any,
) -> dict[str, Any]:
    """Normalize one public inspection or loading request."""
    path = Path(nwb_path)
    if not path.is_file():
        raise FileNotFoundError(f"NWB file not found: {path}")
    return {
        "nwb_path": path,
        "epoch": _text(epoch, name="epoch"),
        "electrode_ids": _electrode_ids(electrode_ids),
        "source_electrical_series_path": _absolute_hdf5_path(
            source_electrical_series_path,
            name="source_electrical_series_path",
        ),
        "interval_table_path": _absolute_hdf5_path(
            interval_table_path,
            name="interval_table_path",
        ),
        "samples_for_rate_estimation": _integer(
            samples_for_rate_estimation,
            name="samples_for_rate_estimation",
            minimum=2,
        ),
    }


def _inspect_open_nwb_input_source(
    nwb_file: Any,
    *,
    source_nwb_file_name: str,
    epoch: str,
    electrode_ids: np.ndarray,
    source_electrical_series_path: str,
    interval_table_path: str,
    samples_for_rate_estimation: int,
) -> dict[str, Any]:
    """Inspect one open NWB source without reading epoch-scale arrays."""
    try:
        series = nwb_file[source_electrical_series_path]
    except KeyError as exc:
        raise ValueError(
            f"ElectricalSeries not found at {source_electrical_series_path!r}."
        ) from exc
    if _text(
        series.attrs.get("neurodata_type", ""),
        name="ElectricalSeries neurodata_type",
    ) != "ElectricalSeries":
        raise ValueError(
            f"NWB object {source_electrical_series_path!r} is not an "
            "ElectricalSeries."
        )
    for child_name in ("data", "timestamps", "electrodes"):
        if child_name not in series:
            raise ValueError(
                f"ElectricalSeries {source_electrical_series_path!r} is "
                f"missing {child_name!r}."
            )
    data = series["data"]
    timestamps = series["timestamps"]
    electrode_region = series["electrodes"]
    if data.ndim != 2 or data.dtype != np.dtype(np.int16):
        raise ValueError("Raw ElectricalSeries data must be two-dimensional int16.")
    if timestamps.ndim != 1 or timestamps.shape[0] != data.shape[0]:
        raise ValueError("ElectricalSeries timestamps must align to data rows.")
    if electrode_region.ndim != 1 or electrode_region.shape[0] != data.shape[1]:
        raise ValueError(
            "ElectricalSeries electrode rows must align to data columns."
        )
    unit = _text(data.attrs.get("unit", ""), name="ElectricalSeries data unit")
    if unit.lower() not in {"v", "volt", "volts"}:
        raise ValueError("Raw ElectricalSeries data unit must be volts.")

    try:
        intervals = nwb_file[interval_table_path]
    except KeyError as exc:
        raise ValueError(
            f"Interval table not found at {interval_table_path!r}."
        ) from exc
    for column_name in ("epoch", "start_time", "stop_time"):
        if column_name not in intervals:
            raise ValueError(
                f"Interval table {interval_table_path!r} is missing "
                f"{column_name!r}."
            )
    epochs = _decode_hdf5_text(
        _read_hdf5_array(intervals["epoch"], slice(None))
    )
    matches = [index for index, value in enumerate(epochs) if value == epoch]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one interval row for epoch {epoch!r}; "
            f"found {len(matches)}."
        )
    interval_row = matches[0]
    epoch_start = _finite_float(
        intervals["start_time"][interval_row],
        name="epoch start_time",
    )
    epoch_stop = _finite_float(
        intervals["stop_time"][interval_row],
        name="epoch stop_time",
    )
    if epoch_stop <= epoch_start:
        raise ValueError("Epoch interval stop_time must exceed start_time.")

    total_samples = int(timestamps.shape[0])
    start_index = _hdf5_searchsorted(timestamps, epoch_start, side="left")
    stop_index = _hdf5_searchsorted(timestamps, epoch_stop, side="right")
    if (
        start_index >= total_samples
        or float(timestamps[start_index]) != epoch_start
        or stop_index <= start_index
        or float(timestamps[stop_index - 1]) != epoch_stop
    ):
        raise ValueError(
            "Epoch boundaries must exactly match ElectricalSeries timestamps."
        )
    if start_index and float(timestamps[start_index - 1]) >= epoch_start:
        raise ValueError("Epoch start slice boundary is ambiguous.")
    if stop_index < total_samples and float(timestamps[stop_index]) <= epoch_stop:
        raise ValueError("Epoch stop slice boundary is ambiguous.")

    reference_count = min(samples_for_rate_estimation, total_samples)
    if reference_count < 2:
        raise ValueError("ElectricalSeries needs at least two timestamps.")
    reference_timestamps = np.asarray(
        _read_hdf5_array(timestamps, slice(0, reference_count)),
        dtype=float,
    )
    if not np.all(np.isfinite(reference_timestamps)) or np.any(
        np.diff(reference_timestamps) <= 0.0
    ):
        raise ValueError("Sampling-rate reference timestamps must increase.")
    source_sampling_frequency_hz = float(
        1.0 / np.median(np.diff(reference_timestamps))
    )
    sampling_frequency_provenance = {
        "method": SAMPLING_FREQUENCY_ESTIMATION_METHOD,
        "samples_for_rate_estimation": reference_count,
        "reference_start_index": 0,
        "reference_stop_index_exclusive": reference_count,
        "reference_timestamps_sha256": _array_sha256(reference_timestamps),
        "estimated_sampling_frequency_hz": source_sampling_frequency_hz,
    }

    try:
        table_reference = electrode_region.attrs["table"]
        electrodes_table = nwb_file[table_reference]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "ElectricalSeries electrodes region has no valid target table."
        ) from exc
    if "id" not in electrodes_table:
        raise ValueError("NWB electrodes table is missing its id column.")
    table_ids = np.asarray(
        _read_hdf5_array(electrodes_table["id"], slice(None)),
        dtype=np.int64,
    )
    table_rows_by_id = {
        int(electrode_id): table_row
        for table_row, electrode_id in enumerate(table_ids.tolist())
    }
    if len(table_rows_by_id) != table_ids.size:
        raise ValueError("NWB electrodes-table IDs must be unique.")
    region_rows = np.asarray(
        _read_hdf5_array(electrode_region, slice(None)),
        dtype=np.int64,
    )
    if np.any(region_rows < 0) or np.any(region_rows >= table_ids.size):
        raise ValueError("ElectricalSeries references invalid electrode rows.")
    if np.unique(region_rows).size != region_rows.size:
        raise ValueError(
            "ElectricalSeries electrode-row references must be unique."
        )
    data_column_by_table_row = {
        int(table_row): data_column
        for data_column, table_row in enumerate(region_rows.tolist())
    }
    selected_table_rows = []
    selected_data_columns = []
    for electrode_id in electrode_ids.tolist():
        if int(electrode_id) not in table_rows_by_id:
            raise ValueError(
                f"Electrode ID {int(electrode_id)} is absent from the NWB table."
            )
        table_row = table_rows_by_id[int(electrode_id)]
        if table_row not in data_column_by_table_row:
            raise ValueError(
                f"Electrode ID {int(electrode_id)} is absent from the "
                "ElectricalSeries."
            )
        selected_table_rows.append(table_row)
        selected_data_columns.append(data_column_by_table_row[table_row])
    selected_columns = np.asarray(selected_data_columns, dtype=np.int64)

    conversion = _finite_float(
        data.attrs.get("conversion"),
        name="ElectricalSeries data conversion",
        positive=True,
    )
    all_gains_to_uv = np.full(data.shape[1], conversion * 1e6, dtype=float)
    if "channel_conversion" in series:
        channel_conversion = np.asarray(
            _read_hdf5_array(series["channel_conversion"], slice(None)),
            dtype=float,
        )
        if channel_conversion.shape != (data.shape[1],) or not np.all(
            np.isfinite(channel_conversion) & (channel_conversion > 0.0)
        ):
            raise ValueError("ElectricalSeries channel_conversion is invalid.")
        all_gains_to_uv *= channel_conversion

    data_offset = _finite_float(
        data.attrs.get("offset", 0.0),
        name="ElectricalSeries data offset",
    )
    if data_offset == 0.0 and "offset" in electrodes_table:
        table_offsets = np.asarray(
            _read_hdf5_array(electrodes_table["offset"], slice(None)),
            dtype=float,
        )
        if table_offsets.shape != table_ids.shape or not np.all(
            np.isfinite(table_offsets)
        ):
            raise ValueError("NWB electrodes-table offsets are invalid.")
        all_offsets_to_uv = table_offsets[region_rows] * 1e6
    else:
        all_offsets_to_uv = np.full(
            data.shape[1],
            data_offset * 1e6,
            dtype=float,
        )

    snapshot = {
        "source_nwb_file_name": source_nwb_file_name,
        "source_electrical_series_path": series.name,
        "electrode_ids": electrode_ids.astype(int).tolist(),
        "gain_to_uv": all_gains_to_uv[selected_columns].tolist(),
        "offset_to_uv": all_offsets_to_uv[selected_columns].tolist(),
        "source_sampling_frequency_hz": source_sampling_frequency_hz,
        "sampling_frequency_provenance": sampling_frequency_provenance,
        "source_slice_provenance": {
            "epoch": epoch,
            "data_path": data.name,
            "timestamps_path": timestamps.name,
            "electrodes_path": electrode_region.name,
            "interval_table_path": intervals.name,
            "electrodes_table_path": electrodes_table.name,
            "interval_table_row_index": interval_row,
            "source_start_index": start_index,
            "source_stop_index_exclusive": stop_index,
            "epoch_start_time_s": epoch_start,
            "epoch_stop_time_s": epoch_stop,
            "electrical_series_object_id": _hdf5_object_id(
                series,
                object_path=series.name,
            ),
            "electrodes_region_object_id": _hdf5_object_id(
                electrode_region,
                object_path=electrode_region.name,
            ),
            "electrodes_table_object_id": _hdf5_object_id(
                electrodes_table,
                object_path=electrodes_table.name,
            ),
            "interval_table_object_id": _hdf5_object_id(
                intervals,
                object_path=intervals.name,
            ),
            "electrodes_region_sha256": _array_sha256(region_rows),
            "electrodes_table_ids_sha256": _array_sha256(table_ids),
            "selected_data_column_indices": (
                selected_columns.astype(int).tolist()
            ),
            "selected_electrode_table_rows": [
                int(value) for value in selected_table_rows
            ],
        },
    }
    return _normalize_nwb_input_snapshot(snapshot)


def _h5py_module() -> Any:
    """Import the optional HDF5 dependency only when NWB access is requested."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError("h5py is required to load raw NWB LFP inputs.") from exc
    return h5py


def inspect_selected_ripple_band_lfp_nwb_inputs(
    *,
    nwb_path: Path,
    epoch: Any,
    electrode_ids: Sequence[Any],
    source_electrical_series_path: Any = "/acquisition/e-series",
    interval_table_path: Any = "/intervals/ephys_recording_intervals",
    samples_for_rate_estimation: Any = DEFAULT_SAMPLES_FOR_RATE_ESTIMATION,
) -> dict[str, Any]:
    """Freeze one epoch's NWB paths, slice, scaling, and rate metadata only."""
    request = _normalize_nwb_input_request(
        nwb_path=nwb_path,
        epoch=epoch,
        electrode_ids=electrode_ids,
        source_electrical_series_path=source_electrical_series_path,
        interval_table_path=interval_table_path,
        samples_for_rate_estimation=samples_for_rate_estimation,
    )
    h5py = _h5py_module()
    with h5py.File(request["nwb_path"], "r") as nwb_file:
        return _inspect_open_nwb_input_source(
            nwb_file,
            source_nwb_file_name=request["nwb_path"].name,
            epoch=request["epoch"],
            electrode_ids=request["electrode_ids"],
            source_electrical_series_path=request[
                "source_electrical_series_path"
            ],
            interval_table_path=request["interval_table_path"],
            samples_for_rate_estimation=request[
                "samples_for_rate_estimation"
            ],
        )


def _require_expected_nwb_input_snapshot(
    current: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    """Reject any source metadata drift before materializing epoch arrays."""
    if expected is None:
        return
    normalized = _normalize_nwb_input_snapshot(expected)
    for field_name in sorted(_NWB_INPUT_SNAPSHOT_FIELDS):
        if _provenance_sha256(current[field_name]) != _provenance_sha256(
            normalized[field_name]
        ):
            raise ValueError(
                "NWB input snapshot changed after selection: "
                f"{field_name}."
            )


def load_selected_ripple_band_lfp_nwb_inputs(
    *,
    nwb_path: Path,
    epoch: Any,
    electrode_ids: Sequence[Any],
    source_electrical_series_path: Any = "/acquisition/e-series",
    interval_table_path: Any = "/intervals/ephys_recording_intervals",
    samples_for_rate_estimation: Any = DEFAULT_SAMPLES_FOR_RATE_ESTIMATION,
    expected_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one frozen epoch/electrode slice from an augmented NWB file.

    Inspection and expected-snapshot validation finish before the selected
    trace or timestamp slice is materialized. The source-rate and voltage
    semantics deliberately mirror the SpikeInterface extractor used by the
    legacy ripple detector.
    """
    request = _normalize_nwb_input_request(
        nwb_path=nwb_path,
        epoch=epoch,
        electrode_ids=electrode_ids,
        source_electrical_series_path=source_electrical_series_path,
        interval_table_path=interval_table_path,
        samples_for_rate_estimation=samples_for_rate_estimation,
    )
    h5py = _h5py_module()
    with h5py.File(request["nwb_path"], "r") as nwb_file:
        snapshot = _inspect_open_nwb_input_source(
            nwb_file,
            source_nwb_file_name=request["nwb_path"].name,
            epoch=request["epoch"],
            electrode_ids=request["electrode_ids"],
            source_electrical_series_path=request[
                "source_electrical_series_path"
            ],
            interval_table_path=request["interval_table_path"],
            samples_for_rate_estimation=request[
                "samples_for_rate_estimation"
            ],
        )
        _require_expected_nwb_input_snapshot(snapshot, expected_snapshot)
        source_slice = snapshot["source_slice_provenance"]
        start_index = int(source_slice["source_start_index"])
        stop_index = int(source_slice["source_stop_index_exclusive"])
        selected_columns = np.asarray(
            source_slice["selected_data_column_indices"],
            dtype=np.int64,
        )
        sorted_order = np.argsort(selected_columns)
        inverse_order = np.argsort(sorted_order)
        sorted_columns = selected_columns[sorted_order]
        series = nwb_file[snapshot["source_electrical_series_path"]]
        raw_traces = _read_hdf5_array(
            series["data"],
            (slice(start_index, stop_index), sorted_columns.tolist()),
        )[:, inverse_order]
        raw_timestamps = np.asarray(
            _read_hdf5_array(
                series["timestamps"],
                slice(start_index, stop_index),
            ),
            dtype=float,
        )
        if raw_traces.dtype != np.dtype(np.int16):
            raise ValueError("Raw HDF5 slice did not preserve int16 data.")
        if raw_timestamps.shape != (stop_index - start_index,):
            raise ValueError("Raw HDF5 timestamp slice has the wrong length.")
        if (
            raw_timestamps.size == 0
            or float(raw_timestamps[0]) != source_slice["epoch_start_time_s"]
            or float(raw_timestamps[-1]) != source_slice["epoch_stop_time_s"]
        ):
            raise ValueError("Raw HDF5 timestamp slice boundaries changed.")
    return {
        **snapshot,
        "raw_timestamps": raw_timestamps,
        "raw_traces": raw_traces,
    }


def _canonical_upstream_provenance(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a JSON-safe optional upstream selection snapshot."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("upstream_provenance must be a mapping.")
    normalized = _json_safe(dict(value))
    json.dumps(normalized, sort_keys=True, allow_nan=False)
    return normalized


def _build_dataset(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    result_id: str,
    source_nwb_file_name: str,
    source_electrical_series_path: str,
    timestamps: np.ndarray,
    filtered_lfp: np.ndarray,
    electrode_ids: np.ndarray,
    gain_to_uv: np.ndarray,
    offset_to_uv: np.ndarray,
    trace_scaling_sha256: str,
    sampling_frequency_provenance: Mapping[str, Any],
    source_slice_provenance: Mapping[str, Any],
    parameters: Mapping[str, Any],
    source_sampling_frequency_hz: float,
    analysis_status: str,
    artifact_origin: str,
    raw_timestamps_sha256: str,
    raw_traces_sha256: str,
) -> Any:
    """Build the legacy-compatible NetCDF schema plus canonical provenance attrs."""
    dataset = detect_ripples.build_epoch_lfp_dataset(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        timestamps=timestamps,
        filtered_lfp=filtered_lfp,
        sampling_frequency=float(parameters["actual_sampling_frequency_hz"]),
        channel_ids=electrode_ids.astype(int).tolist(),
        enable_notch_filter=bool(parameters["enable_notch_filter"]),
    )
    dataset.attrs.update(
        {
            "ripple_band_lfp_id": result_id,
            "source_nwb_file_name": source_nwb_file_name,
            "source_electrical_series_path": source_electrical_series_path,
            "source_trace_dtype": "int16",
            "filter_input_unit": "microvolts",
            "source_sampling_frequency_hz": float(source_sampling_frequency_hz),
            "electrode_coordinate_semantics": "nwb_electrodes_table_id",
            "electrode_order_preserved": 1,
            "ordered_gain_to_uV_json": json.dumps(
                gain_to_uv.tolist(), separators=(",", ":")
            ),
            "ordered_offset_to_uV_json": json.dumps(
                offset_to_uv.tolist(), separators=(",", ":")
            ),
            "trace_scaling_sha256": trace_scaling_sha256,
            "sampling_frequency_provenance_sha256": _provenance_sha256(
                sampling_frequency_provenance
            ),
            "source_slice_provenance_sha256": _provenance_sha256(
                source_slice_provenance
            ),
            "filter_order": int(parameters["filter_order"]),
            "lowcut_hz": float(parameters["lowcut_hz"]),
            "highcut_hz": float(parameters["highcut_hz"]),
            "target_sampling_frequency_hz": float(
                parameters["target_sampling_frequency_hz"]
            ),
            "notch_filter_enabled": int(parameters["enable_notch_filter"]),
            "notch_base_freq_hz": float(parameters["notch_base_freq_hz"]),
            "notch_harmonics": int(parameters["notch_harmonics"]),
            "notch_quality": float(parameters["notch_quality"]),
            "decimation_factor": int(parameters["decimation_factor"]),
            "actual_sampling_frequency_hz": float(
                parameters["actual_sampling_frequency_hz"]
            ),
            "parameter_name": str(parameters["parameter_name"]),
            "parameter_sha256": str(parameters["parameter_sha256"]),
            "output_rule_sha256": str(parameters["output_rule_sha256"]),
            "raw_timestamps_sha256": raw_timestamps_sha256,
            "raw_traces_sha256": raw_traces_sha256,
            "analysis_status": analysis_status,
            "artifact_origin": artifact_origin,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        }
    )
    return dataset


def _channel_qc_table(
    *,
    result_id: str,
    animal_name: str,
    date: str,
    epoch: str,
    electrode_ids: np.ndarray,
    gain_to_uv: np.ndarray,
    offset_to_uv: np.ndarray,
    input_sample_count: int,
    output_sample_count: int,
    source_sampling_frequency_hz: float,
    actual_sampling_frequency_hz: float,
    analysis_status: str,
) -> pd.DataFrame:
    """Return one ordered audit row per selected electrode."""
    rows = [
        {
            "ripple_band_lfp_id": result_id,
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "channel_index": channel_index,
            "electrode_id": int(electrode_id),
            "input_sample_count": input_sample_count,
            "output_sample_count": output_sample_count,
            "input_dtype": "int16",
            "gain_to_uV": float(gain_to_uv[channel_index]),
            "offset_to_uV": float(offset_to_uv[channel_index]),
            "source_sampling_frequency_hz": source_sampling_frequency_hz,
            "actual_sampling_frequency_hz": actual_sampling_frequency_hz,
            "analysis_status": analysis_status,
        }
        for channel_index, electrode_id in enumerate(electrode_ids)
    ]
    return pd.DataFrame.from_records(rows, columns=QC_COLUMNS)


def compute_selected_ripple_band_lfp(
    *,
    animal_name: Any,
    date: Any,
    epoch: Any,
    ripple_band_lfp_id: Any,
    source_nwb_file_name: Any,
    source_electrical_series_path: Any,
    raw_timestamps: Any,
    raw_traces: Any,
    electrode_ids: Sequence[Any],
    gain_to_uv: Sequence[Any],
    offset_to_uv: Sequence[Any],
    source_sampling_frequency_hz: Any,
    sampling_frequency_provenance: Mapping[str, Any],
    source_slice_provenance: Mapping[str, Any],
    parameter_name: Any = "manuscript_150_250hz_1khz",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    lowcut_hz: Any = DEFAULT_LOWCUT_HZ,
    highcut_hz: Any = DEFAULT_HIGHCUT_HZ,
    filter_order: Any = DEFAULT_FILTER_ORDER,
    target_sampling_frequency_hz: Any = DEFAULT_TARGET_SAMPLING_FREQUENCY_HZ,
    enable_notch_filter: Any = DEFAULT_ENABLE_NOTCH_FILTER,
    notch_base_freq_hz: Any = DEFAULT_NOTCH_BASE_FREQ_HZ,
    notch_harmonics: Any = DEFAULT_NOTCH_HARMONICS,
    notch_quality: Any = DEFAULT_NOTCH_QUALITY,
    upstream_provenance: Mapping[str, Any] | None = None,
    artifact_origin: str = "computed",
    legacy_artifact_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one epoch's ripple-band LFP from exact raw NWB inputs."""
    animal = _path_component(animal_name, name="animal_name")
    session_date = _path_component(date, name="date")
    epoch_name = _path_component(epoch, name="epoch")
    result_id = _uuid_string(ripple_band_lfp_id, name="ripple_band_lfp_id")
    nwb_file_name = _text(source_nwb_file_name, name="source_nwb_file_name")
    series_path = _text(
        source_electrical_series_path,
        name="source_electrical_series_path",
    )
    if not series_path.startswith("/"):
        raise ValueError("source_electrical_series_path must be an absolute NWB path.")
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("artifact_origin must be computed or registered_existing.")
    effective = validate_ripple_band_lfp_parameters(
        source_sampling_frequency_hz=source_sampling_frequency_hz,
        lowcut_hz=lowcut_hz,
        highcut_hz=highcut_hz,
        filter_order=filter_order,
        target_sampling_frequency_hz=target_sampling_frequency_hz,
        enable_notch_filter=enable_notch_filter,
        notch_base_freq_hz=notch_base_freq_hz,
        notch_harmonics=notch_harmonics,
        notch_quality=notch_quality,
    )
    parameters = {
        **_parameter_snapshot(
            parameter_name=parameter_name,
            parameters=effective,
            parameter_sha256=parameter_sha256,
            output_rule_sha256=output_rule_sha256,
        ),
        "source_sampling_frequency_hz": effective[
            "source_sampling_frequency_hz"
        ],
        "decimation_factor": effective["decimation_factor"],
        "actual_sampling_frequency_hz": effective[
            "actual_sampling_frequency_hz"
        ],
    }
    raw = _validate_raw_inputs(
        raw_timestamps=raw_timestamps,
        raw_traces=raw_traces,
        electrode_ids=electrode_ids,
    )
    timestamps = raw["timestamps"]
    traces = raw["traces"]
    electrodes = raw["electrode_ids"]
    scaling = _ordered_voltage_scaling(
        gain_to_uv=gain_to_uv,
        offset_to_uv=offset_to_uv,
        electrode_ids=electrodes,
    )
    frequency_snapshot = _sampling_frequency_snapshot(
        sampling_frequency_provenance,
        source_sampling_frequency_hz=effective["source_sampling_frequency_hz"],
    )
    slice_snapshot = _source_slice_snapshot(
        source_slice_provenance,
        epoch=epoch_name,
        source_electrical_series_path=series_path,
        input_sample_count=int(timestamps.size),
        input_start_time_s=(float(timestamps[0]) if timestamps.size else None),
        input_stop_time_s=(float(timestamps[-1]) if timestamps.size else None),
        electrode_ids=electrodes,
    )
    if timestamps.size == 0:
        analysis_status = "empty_input"
        output_timestamps = np.asarray([], dtype=float)
        filtered_lfp = np.empty((0, electrodes.size), dtype=float)
    else:
        analysis_status = "valid"
        # Match the complete detector path exactly: SpikeInterface 0.103.2
        # performs gain/offset scaling in float32, then detect_ripples casts
        # return_in_uV traces to Python float (float64) before filtering.
        scaled_traces_float32 = (
            np.asarray(traces, dtype=np.float32)
            * np.asarray(scaling["gain_to_uv"], dtype=np.float32)[None, :]
            + np.asarray(scaling["offset_to_uv"], dtype=np.float32)[None, :]
        )
        filter_input = np.asarray(scaled_traces_float32, dtype=float)
        if parameters["enable_notch_filter"]:
            filter_input = detect_ripples.apply_notch_filters_multichannel(
                filter_input,
                fs=effective["source_sampling_frequency_hz"],
                base_freq=parameters["notch_base_freq_hz"],
                n_harmonics=parameters["notch_harmonics"],
                quality=parameters["notch_quality"],
            )
        try:
            (
                output_timestamps,
                filtered_lfp,
                actual_sampling_frequency_hz,
            ) = detect_ripples.butter_filter_and_decimate(
                timestamps,
                filter_input,
                effective["source_sampling_frequency_hz"],
                target_new_sampling_frequency=parameters[
                    "target_sampling_frequency_hz"
                ],
                lowcut=parameters["lowcut_hz"],
                highcut=parameters["highcut_hz"],
                order=parameters["filter_order"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Ripple-band filtering failed for the nonempty raw epoch input."
            ) from exc
        if not np.isclose(
            actual_sampling_frequency_hz,
            effective["actual_sampling_frequency_hz"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError("Legacy helper returned an unexpected actual frequency.")
        if output_timestamps.size == 0 or filtered_lfp.shape[0] == 0:
            raise ValueError("Nonempty raw input produced an empty filtered result.")
    dataset = _build_dataset(
        animal_name=animal,
        date=session_date,
        epoch=epoch_name,
        result_id=result_id,
        source_nwb_file_name=nwb_file_name,
        source_electrical_series_path=series_path,
        timestamps=output_timestamps,
        filtered_lfp=filtered_lfp,
        electrode_ids=electrodes,
        gain_to_uv=scaling["gain_to_uv"],
        offset_to_uv=scaling["offset_to_uv"],
        trace_scaling_sha256=scaling["trace_scaling_sha256"],
        sampling_frequency_provenance=frequency_snapshot,
        source_slice_provenance=slice_snapshot,
        parameters=parameters,
        source_sampling_frequency_hz=effective["source_sampling_frequency_hz"],
        analysis_status=analysis_status,
        artifact_origin=artifact_origin,
        raw_timestamps_sha256=raw["raw_timestamps_sha256"],
        raw_traces_sha256=raw["raw_traces_sha256"],
    )
    qc = _channel_qc_table(
        result_id=result_id,
        animal_name=animal,
        date=session_date,
        epoch=epoch_name,
        electrode_ids=electrodes,
        gain_to_uv=scaling["gain_to_uv"],
        offset_to_uv=scaling["offset_to_uv"],
        input_sample_count=int(timestamps.size),
        output_sample_count=int(output_timestamps.size),
        source_sampling_frequency_hz=effective["source_sampling_frequency_hz"],
        actual_sampling_frequency_hz=effective["actual_sampling_frequency_hz"],
        analysis_status=analysis_status,
    )
    result = {
        "metadata": {
            "ripple_band_lfp_id": result_id,
            "animal_name": animal,
            "date": session_date,
            "epoch": epoch_name,
            "source_nwb_file_name": nwb_file_name,
            "source_electrical_series_path": series_path,
        },
        "parameters": parameters,
        "upstream_provenance": _canonical_upstream_provenance(
            upstream_provenance
        ),
        "sampling_frequency_provenance": frequency_snapshot,
        "source_slice_provenance": slice_snapshot,
        "dataset": dataset,
        "channel_qc": qc,
        "ordered_electrode_ids": electrodes.astype(int).tolist(),
        "ordered_gain_to_uv": scaling["gain_to_uv"].tolist(),
        "ordered_offset_to_uv": scaling["offset_to_uv"].tolist(),
        "trace_scaling_sha256": scaling["trace_scaling_sha256"],
        "raw_timestamps_sha256": raw["raw_timestamps_sha256"],
        "raw_traces_sha256": raw["raw_traces_sha256"],
        "input_sample_count": int(timestamps.size),
        "output_sample_count": int(output_timestamps.size),
        "input_start_time_s": (
            float(timestamps[0]) if timestamps.size else np.nan
        ),
        "input_stop_time_s": (
            float(timestamps[-1]) if timestamps.size else np.nan
        ),
        "source_sampling_frequency_hz": effective[
            "source_sampling_frequency_hz"
        ],
        "actual_sampling_frequency_hz": effective[
            "actual_sampling_frequency_hz"
        ],
        "decimation_factor": effective["decimation_factor"],
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance": _canonical_upstream_provenance(
            legacy_artifact_provenance
        ),
    }
    return validate_ripple_band_lfp_result(result)


def _validate_dataset(
    dataset: Any,
    *,
    metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    ordered_electrode_ids: Sequence[int],
    ordered_gain_to_uv: Sequence[float],
    ordered_offset_to_uv: Sequence[float],
    trace_scaling_sha256: str,
    sampling_frequency_provenance: Mapping[str, Any],
    source_slice_provenance: Mapping[str, Any],
    input_sample_count: int,
    output_sample_count: int,
    input_start_time_s: float,
    input_stop_time_s: float,
    analysis_status: str,
    artifact_origin: str,
    raw_timestamps_sha256: str,
    raw_traces_sha256: str,
) -> None:
    """Validate the complete canonical NetCDF schema and metadata."""
    required_data_vars = {"filtered_lfp", "sampling_frequency_hz"}
    required_coords = {"time", "channel"}
    if not required_data_vars.issubset(dataset.data_vars):
        raise ValueError("Ripple-band dataset is missing required data variables.")
    if not required_coords.issubset(dataset.coords):
        raise ValueError("Ripple-band dataset is missing required coordinates.")
    if tuple(dataset["filtered_lfp"].dims) != ("sample", "channel"):
        raise ValueError("filtered_lfp must have dimensions (sample, channel).")
    if tuple(dataset["time"].dims) != ("sample",):
        raise ValueError("time must have dimension sample.")
    if tuple(dataset["channel"].dims) != ("channel",):
        raise ValueError("channel must have dimension channel.")
    filtered_lfp = np.asarray(dataset["filtered_lfp"].values, dtype=float)
    timestamps = np.asarray(dataset["time"].values, dtype=float)
    electrode_ids = np.asarray(dataset["channel"].values, dtype=np.int64)
    if filtered_lfp.shape != (output_sample_count, len(ordered_electrode_ids)):
        raise ValueError("Filtered LFP shape does not match canonical sample counts.")
    if timestamps.shape != (output_sample_count,):
        raise ValueError("Filtered timestamps do not match output_sample_count.")
    if electrode_ids.tolist() != list(ordered_electrode_ids):
        raise ValueError("Dataset electrode order differs from the selected order.")
    if timestamps.size and (
        not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("Filtered timestamps must be finite and increasing.")
    expected_output_count = (
        input_sample_count + int(parameters["decimation_factor"]) - 1
    ) // int(parameters["decimation_factor"])
    if output_sample_count != expected_output_count:
        raise ValueError(
            "output_sample_count must equal ceil(input samples / stride)."
        )
    if timestamps.size:
        if timestamps[0] != input_start_time_s:
            raise ValueError(
                "The first filtered timestamp must equal the input start."
            )
        if np.any(timestamps < input_start_time_s) or np.any(
            timestamps > input_stop_time_s
        ):
            raise ValueError(
                "Filtered timestamps must remain within the raw epoch input."
            )
    if not np.all(np.isfinite(filtered_lfp)):
        raise ValueError("Filtered LFP values must be finite.")
    sampling_frequency = np.asarray(
        dataset["sampling_frequency_hz"].values
    )
    if sampling_frequency.ndim != 0 or not np.isclose(
        float(sampling_frequency),
        float(parameters["actual_sampling_frequency_hz"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Dataset actual sampling frequency is inconsistent.")

    attrs = dataset.attrs
    expected_attrs = {
        "ripple_band_lfp_id": metadata["ripple_band_lfp_id"],
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "epoch": metadata["epoch"],
        "source_nwb_file_name": metadata["source_nwb_file_name"],
        "source_electrical_series_path": metadata[
            "source_electrical_series_path"
        ],
        "source_trace_dtype": "int16",
        "filter_input_unit": "microvolts",
        "electrode_coordinate_semantics": "nwb_electrodes_table_id",
        "ordered_gain_to_uV_json": json.dumps(
            list(ordered_gain_to_uv), separators=(",", ":")
        ),
        "ordered_offset_to_uV_json": json.dumps(
            list(ordered_offset_to_uv), separators=(",", ":")
        ),
        "trace_scaling_sha256": trace_scaling_sha256,
        "sampling_frequency_provenance_sha256": _provenance_sha256(
            sampling_frequency_provenance
        ),
        "source_slice_provenance_sha256": _provenance_sha256(
            source_slice_provenance
        ),
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "raw_timestamps_sha256": raw_timestamps_sha256,
        "raw_traces_sha256": raw_traces_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "cache_format": detect_ripples.LFP_CACHE_FORMAT,
    }
    for name, expected in expected_attrs.items():
        if str(_python_scalar(attrs.get(name, ""))) != str(expected):
            raise ValueError(f"Dataset attribute {name!r} is inconsistent.")
    numeric_attrs = {
        "source_sampling_frequency_hz": parameters[
            "source_sampling_frequency_hz"
        ],
        "actual_sampling_frequency_hz": parameters[
            "actual_sampling_frequency_hz"
        ],
        "lowcut_hz": parameters["lowcut_hz"],
        "highcut_hz": parameters["highcut_hz"],
        "target_sampling_frequency_hz": parameters[
            "target_sampling_frequency_hz"
        ],
        "notch_base_freq_hz": parameters["notch_base_freq_hz"],
        "notch_quality": parameters["notch_quality"],
    }
    for name, expected in numeric_attrs.items():
        try:
            actual = float(_python_scalar(attrs[name]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Dataset attribute {name!r} is missing.") from exc
        if not np.isclose(actual, float(expected), rtol=0.0, atol=1e-12):
            raise ValueError(f"Dataset attribute {name!r} is inconsistent.")
    integer_attrs = {
        "filter_order": parameters["filter_order"],
        "notch_filter_enabled": int(parameters["enable_notch_filter"]),
        "notch_harmonics": parameters["notch_harmonics"],
        "decimation_factor": parameters["decimation_factor"],
        "electrode_order_preserved": 1,
        "cache_format_version": detect_ripples.LFP_CACHE_FORMAT_VERSION,
    }
    for name, expected in integer_attrs.items():
        try:
            actual = int(_python_scalar(attrs[name]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Dataset attribute {name!r} is missing.") from exc
        if actual != int(expected):
            raise ValueError(f"Dataset attribute {name!r} is inconsistent.")
    if analysis_status == "valid" and (
        input_sample_count <= 0 or output_sample_count <= 0
    ):
        raise ValueError("valid status requires nonempty input and output.")
    if analysis_status == "empty_input" and (
        input_sample_count != 0 or output_sample_count != 0
    ):
        raise ValueError("empty_input status requires zero input and output samples.")


def validate_ripple_band_lfp_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and shallow-copy one in-memory RippleBandLFP result."""
    copied = dict(result)
    metadata = dict(copied["metadata"])
    result_id = _uuid_string(
        metadata["ripple_band_lfp_id"],
        name="ripple_band_lfp_id",
    )
    metadata = {
        "ripple_band_lfp_id": result_id,
        "animal_name": _path_component(metadata["animal_name"], name="animal_name"),
        "date": _path_component(metadata["date"], name="date"),
        "epoch": _path_component(metadata["epoch"], name="epoch"),
        "source_nwb_file_name": _text(
            metadata["source_nwb_file_name"],
            name="source_nwb_file_name",
        ),
        "source_electrical_series_path": _text(
            metadata["source_electrical_series_path"],
            name="source_electrical_series_path",
        ),
    }
    parameters_input = dict(copied["parameters"])
    effective = validate_ripple_band_lfp_parameters(
        source_sampling_frequency_hz=parameters_input[
            "source_sampling_frequency_hz"
        ],
        lowcut_hz=parameters_input["lowcut_hz"],
        highcut_hz=parameters_input["highcut_hz"],
        filter_order=parameters_input["filter_order"],
        target_sampling_frequency_hz=parameters_input[
            "target_sampling_frequency_hz"
        ],
        enable_notch_filter=parameters_input["enable_notch_filter"],
        notch_base_freq_hz=parameters_input["notch_base_freq_hz"],
        notch_harmonics=parameters_input["notch_harmonics"],
        notch_quality=parameters_input["notch_quality"],
    )
    parameters = {
        **_parameter_snapshot(
            parameter_name=parameters_input["parameter_name"],
            parameters=effective,
            parameter_sha256=parameters_input["parameter_sha256"],
            output_rule_sha256=parameters_input["output_rule_sha256"],
        ),
        "source_sampling_frequency_hz": effective[
            "source_sampling_frequency_hz"
        ],
        "decimation_factor": effective["decimation_factor"],
        "actual_sampling_frequency_hz": effective[
            "actual_sampling_frequency_hz"
        ],
    }
    ordered_electrodes = _electrode_ids(copied["ordered_electrode_ids"])
    scaling = _ordered_voltage_scaling(
        gain_to_uv=copied["ordered_gain_to_uv"],
        offset_to_uv=copied["ordered_offset_to_uv"],
        electrode_ids=ordered_electrodes,
    )
    trace_scaling_sha256 = _text(
        copied["trace_scaling_sha256"],
        name="trace_scaling_sha256",
    )
    if trace_scaling_sha256 != scaling["trace_scaling_sha256"]:
        raise ValueError("trace_scaling_sha256 does not match ordered scaling.")
    sampling_frequency_provenance = _sampling_frequency_snapshot(
        copied["sampling_frequency_provenance"],
        source_sampling_frequency_hz=parameters["source_sampling_frequency_hz"],
    )
    input_count = _integer(
        copied["input_sample_count"],
        name="input_sample_count",
    )
    output_count = _integer(
        copied["output_sample_count"],
        name="output_sample_count",
    )
    input_start_time_s = float(copied["input_start_time_s"])
    input_stop_time_s = float(copied["input_stop_time_s"])
    source_slice_provenance = _source_slice_snapshot(
        copied["source_slice_provenance"],
        epoch=metadata["epoch"],
        source_electrical_series_path=metadata["source_electrical_series_path"],
        input_sample_count=input_count,
        input_start_time_s=(input_start_time_s if input_count else None),
        input_stop_time_s=(input_stop_time_s if input_count else None),
        electrode_ids=ordered_electrodes,
    )
    if not np.isclose(
        _finite_float(
            copied["source_sampling_frequency_hz"],
            name="source_sampling_frequency_hz",
            positive=True,
        ),
        parameters["source_sampling_frequency_hz"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Result source sampling frequency is inconsistent.")
    if not np.isclose(
        _finite_float(
            copied["actual_sampling_frequency_hz"],
            name="actual_sampling_frequency_hz",
            positive=True,
        ),
        parameters["actual_sampling_frequency_hz"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Result actual sampling frequency is inconsistent.")
    if _integer(
        copied["decimation_factor"],
        name="decimation_factor",
        minimum=1,
    ) != parameters["decimation_factor"]:
        raise ValueError("Result decimation factor is inconsistent.")
    status = _text(copied["analysis_status"], name="analysis_status")
    if status not in ANALYSIS_STATUSES:
        raise ValueError("Unsupported RippleBandLFP analysis_status.")
    origin = _text(copied["artifact_origin"], name="artifact_origin")
    if origin not in {"computed", "registered_existing"}:
        raise ValueError("Unsupported RippleBandLFP artifact_origin.")
    legacy_provenance = _canonical_upstream_provenance(
        copied.get("legacy_artifact_provenance")
    )
    if origin == "computed" and legacy_provenance:
        raise ValueError(
            "Computed RippleBandLFP results cannot contain legacy provenance."
        )
    if origin == "registered_existing" and not legacy_provenance:
        raise ValueError(
            "Registered RippleBandLFP results require legacy provenance."
        )
    if input_count:
        if not np.isfinite(input_start_time_s) or not np.isfinite(input_stop_time_s):
            raise ValueError("Nonempty input requires finite input time boundaries.")
        if input_stop_time_s <= input_start_time_s:
            raise ValueError("Nonempty input time boundaries must increase.")
    elif not (np.isnan(input_start_time_s) and np.isnan(input_stop_time_s)):
        raise ValueError("Empty input requires NaN input time boundaries.")
    for name in ("raw_timestamps_sha256", "raw_traces_sha256"):
        if len(_text(copied[name], name=name)) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest.")
    qc = copied["channel_qc"]
    if not isinstance(qc, pd.DataFrame) or list(qc.columns) != list(QC_COLUMNS):
        raise ValueError("channel_qc does not match the canonical schema.")
    if len(qc) != ordered_electrodes.size:
        raise ValueError("channel_qc must contain one row per selected electrode.")
    if qc["channel_index"].to_numpy(dtype=int).tolist() != list(
        range(ordered_electrodes.size)
    ):
        raise ValueError("channel_qc channel_index does not preserve order.")
    if qc["electrode_id"].to_numpy(dtype=int).tolist() != ordered_electrodes.tolist():
        raise ValueError("channel_qc electrode IDs do not preserve selection order.")
    expected_qc_scalars = {
        "ripple_band_lfp_id": result_id,
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "epoch": metadata["epoch"],
        "input_sample_count": input_count,
        "output_sample_count": output_count,
        "input_dtype": "int16",
        "analysis_status": status,
    }
    for name, expected in expected_qc_scalars.items():
        if not np.all(qc[name].astype(str) == str(expected)):
            raise ValueError(f"channel_qc field {name!r} is inconsistent.")
    for name, expected in {
        "gain_to_uV": scaling["gain_to_uv"],
        "offset_to_uV": scaling["offset_to_uv"],
    }.items():
        if not np.allclose(
            qc[name].to_numpy(dtype=float),
            np.asarray(expected, dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"channel_qc field {name!r} is inconsistent.")
    for name, expected in {
        "source_sampling_frequency_hz": parameters[
            "source_sampling_frequency_hz"
        ],
        "actual_sampling_frequency_hz": parameters[
            "actual_sampling_frequency_hz"
        ],
    }.items():
        if not np.allclose(
            qc[name].to_numpy(dtype=float),
            float(expected),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"channel_qc field {name!r} is inconsistent.")
    _validate_dataset(
        copied["dataset"],
        metadata=metadata,
        parameters=parameters,
        ordered_electrode_ids=ordered_electrodes.tolist(),
        ordered_gain_to_uv=scaling["gain_to_uv"].tolist(),
        ordered_offset_to_uv=scaling["offset_to_uv"].tolist(),
        trace_scaling_sha256=trace_scaling_sha256,
        sampling_frequency_provenance=sampling_frequency_provenance,
        source_slice_provenance=source_slice_provenance,
        input_sample_count=input_count,
        output_sample_count=output_count,
        input_start_time_s=input_start_time_s,
        input_stop_time_s=input_stop_time_s,
        analysis_status=status,
        artifact_origin=origin,
        raw_timestamps_sha256=_text(
            copied["raw_timestamps_sha256"], name="raw_timestamps_sha256"
        ),
        raw_traces_sha256=_text(
            copied["raw_traces_sha256"], name="raw_traces_sha256"
        ),
    )
    copied.update(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": _canonical_upstream_provenance(
                copied.get("upstream_provenance")
            ),
            "ordered_electrode_ids": ordered_electrodes.astype(int).tolist(),
            "ordered_gain_to_uv": scaling["gain_to_uv"].tolist(),
            "ordered_offset_to_uv": scaling["offset_to_uv"].tolist(),
            "trace_scaling_sha256": trace_scaling_sha256,
            "sampling_frequency_provenance": sampling_frequency_provenance,
            "source_slice_provenance": source_slice_provenance,
            "input_sample_count": input_count,
            "output_sample_count": output_count,
            "input_start_time_s": input_start_time_s,
            "input_stop_time_s": input_stop_time_s,
            "source_sampling_frequency_hz": parameters[
                "source_sampling_frequency_hz"
            ],
            "actual_sampling_frequency_hz": parameters[
                "actual_sampling_frequency_hz"
            ],
            "decimation_factor": parameters["decimation_factor"],
            "analysis_status": status,
            "artifact_origin": origin,
            "legacy_artifact_provenance": legacy_provenance,
        }
    )
    return copied


def _time_boundary(dataset: Any, index: int) -> float:
    """Return one output timestamp boundary or NaN for empty data."""
    timestamps = np.asarray(dataset["time"].values, dtype=float)
    return float(timestamps[index]) if timestamps.size else np.nan


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return immutable manifest values repeated for every artifact."""
    metadata = result["metadata"]
    parameters = result["parameters"]
    input_count = int(result["input_sample_count"])
    source_fs = float(result["source_sampling_frequency_hz"])
    return {
        **metadata,
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "upstream_provenance_json": json.dumps(
            result["upstream_provenance"], sort_keys=True, separators=(",", ":")
        ),
        "sampling_frequency_provenance_json": json.dumps(
            result["sampling_frequency_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "sampling_frequency_provenance_sha256": _provenance_sha256(
            result["sampling_frequency_provenance"]
        ),
        "source_slice_provenance_json": json.dumps(
            result["source_slice_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source_slice_provenance_sha256": _provenance_sha256(
            result["source_slice_provenance"]
        ),
        "ordered_electrode_ids_json": json.dumps(
            result["ordered_electrode_ids"], separators=(",", ":")
        ),
        "ordered_gain_to_uV_json": json.dumps(
            result["ordered_gain_to_uv"], separators=(",", ":")
        ),
        "ordered_offset_to_uV_json": json.dumps(
            result["ordered_offset_to_uv"], separators=(",", ":")
        ),
        "trace_scaling_sha256": result["trace_scaling_sha256"],
        "raw_timestamps_sha256": result["raw_timestamps_sha256"],
        "raw_traces_sha256": result["raw_traces_sha256"],
        "n_channels": len(result["ordered_electrode_ids"]),
        "input_sample_count": input_count,
        "output_sample_count": result["output_sample_count"],
        "input_start_time_s": result["input_start_time_s"],
        "input_stop_time_s": result["input_stop_time_s"],
        "output_start_time_s": _time_boundary(result["dataset"], 0),
        "output_stop_time_s": _time_boundary(result["dataset"], -1),
        "source_sampling_frequency_hz": source_fs,
        "actual_sampling_frequency_hz": result[
            "actual_sampling_frequency_hz"
        ],
        "decimation_factor": result["decimation_factor"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": json.dumps(
            result["legacy_artifact_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def write_ripple_band_lfp_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write, checksum, and reload one immutable epoch bundle."""
    validated = validate_ripple_band_lfp_result(result)
    destination = Path(path)
    if destination.name != validated["metadata"]["ripple_band_lfp_id"]:
        raise ValueError("Artifact directory name must equal ripple_band_lfp_id.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite RippleBandLFP artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    backup = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.backup")
    temporary.mkdir()
    try:
        validated["dataset"].to_netcdf(
            temporary / RESULT_FILENAME,
            engine="scipy",
        )
        validated["channel_qc"].to_parquet(
            temporary / QC_FILENAME,
            index=False,
        )
        common = _manifest_common(validated)
        rows = []
        for artifact_key, filename, artifact_kind in (
            ("ripple_band_lfp", RESULT_FILENAME, "netcdf"),
            ("channel_qc", QC_FILENAME, "parquet"),
        ):
            artifact_path = temporary / filename
            rows.append(
                {
                    "artifact_key": artifact_key,
                    "relative_path": filename,
                    "artifact_kind": artifact_kind,
                    "file_size_bytes": artifact_path.stat().st_size,
                    "sha256": _file_sha256(artifact_path),
                    **common,
                }
            )
        pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_parquet(
            temporary / MANIFEST_FILENAME,
            index=False,
        )
        load_ripple_band_lfp_artifact(
            temporary,
            _allow_temporary_name=True,
        )
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    return _paths_for_directory(destination)


def _load_dataset(path: Path) -> Any:
    """Eagerly load one SciPy-NetCDF dataset and close the backing file."""
    import xarray as xr

    with xr.open_dataset(path, engine="scipy") as dataset:
        return dataset.load()


def _manifest_values_equal(series: pd.Series, value: Any) -> bool:
    """Return whether every manifest value equals one normalized scalar."""
    value = _python_scalar(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        expected = float(value)
        actual = series.to_numpy(dtype=float)
        if np.isnan(expected):
            return bool(np.all(np.isnan(actual)))
        return bool(np.allclose(actual, expected, rtol=0.0, atol=1e-12))
    return bool(np.all(series.astype(str) == str(value)))


def load_ripple_band_lfp_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one canonical epoch bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RippleBandLFP manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if list(manifest.columns) != list(MANIFEST_COLUMNS):
        raise ValueError("RippleBandLFP manifest has the wrong schema.")
    expected_artifacts = {
        "ripple_band_lfp": (RESULT_FILENAME, "netcdf"),
        "channel_qc": (QC_FILENAME, "parquet"),
    }
    if len(manifest) != len(expected_artifacts) or (
        manifest["artifact_key"].duplicated().any()
    ):
        raise ValueError("RippleBandLFP manifest artifact rows are incomplete.")
    if set(manifest["artifact_key"].astype(str)) != set(expected_artifacts):
        raise ValueError("RippleBandLFP manifest has an unexpected artifact set.")
    for _, row in manifest.iterrows():
        key = str(row["artifact_key"])
        filename, artifact_kind = expected_artifacts[key]
        if str(row["relative_path"]) != filename or str(
            row["artifact_kind"]
        ) != artifact_kind:
            raise ValueError("RippleBandLFP manifest names or kinds are inconsistent.")
        if Path(str(row["relative_path"])).name != str(row["relative_path"]):
            raise ValueError("Manifest artifact paths must be direct child names.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(f"Manifest checksum mismatch for {artifact_path}.")

    first = manifest.iloc[0]
    common_fields = [
        name
        for name in MANIFEST_COLUMNS
        if name
        not in {
            "artifact_key",
            "relative_path",
            "artifact_kind",
            "file_size_bytes",
            "sha256",
        }
    ]
    for name in common_fields:
        if not _manifest_values_equal(manifest[name], first[name]):
            raise ValueError(f"Manifest field {name!r} is inconsistent across rows.")
    result_id = _uuid_string(first["ripple_band_lfp_id"], name="ripple_band_lfp_id")
    if not _allow_temporary_name and directory.name != result_id:
        raise ValueError("Artifact directory name does not match ripple_band_lfp_id.")
    dataset = _load_dataset(directory / RESULT_FILENAME)
    sampling_frequency_provenance = json.loads(
        str(first["sampling_frequency_provenance_json"])
    )
    source_slice_provenance = json.loads(
        str(first["source_slice_provenance_json"])
    )
    if str(first["sampling_frequency_provenance_sha256"]) != _provenance_sha256(
        sampling_frequency_provenance
    ):
        raise ValueError("Manifest sampling-frequency provenance digest is invalid.")
    if str(first["source_slice_provenance_sha256"]) != _provenance_sha256(
        source_slice_provenance
    ):
        raise ValueError("Manifest source-slice provenance digest is invalid.")
    parameters = {
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        "source_sampling_frequency_hz": float(
            first["source_sampling_frequency_hz"]
        ),
        "lowcut_hz": float(dataset.attrs["lowcut_hz"]),
        "highcut_hz": float(dataset.attrs["highcut_hz"]),
        "filter_order": int(dataset.attrs["filter_order"]),
        "target_sampling_frequency_hz": float(
            dataset.attrs["target_sampling_frequency_hz"]
        ),
        "enable_notch_filter": _database_bool(
            dataset.attrs["notch_filter_enabled"],
            name="notch_filter_enabled",
        ),
        "notch_base_freq_hz": float(dataset.attrs["notch_base_freq_hz"]),
        "notch_harmonics": int(dataset.attrs["notch_harmonics"]),
        "notch_quality": float(dataset.attrs["notch_quality"]),
        "decimation_factor": int(first["decimation_factor"]),
        "actual_sampling_frequency_hz": float(
            first["actual_sampling_frequency_hz"]
        ),
    }
    result = {
        "metadata": {
            "ripple_band_lfp_id": result_id,
            "animal_name": str(first["animal_name"]),
            "date": str(first["date"]),
            "epoch": str(first["epoch"]),
            "source_nwb_file_name": str(first["source_nwb_file_name"]),
            "source_electrical_series_path": str(
                first["source_electrical_series_path"]
            ),
        },
        "parameters": parameters,
        "upstream_provenance": json.loads(str(first["upstream_provenance_json"])),
        "sampling_frequency_provenance": sampling_frequency_provenance,
        "source_slice_provenance": source_slice_provenance,
        "dataset": dataset,
        "channel_qc": pd.read_parquet(directory / QC_FILENAME),
        "ordered_electrode_ids": json.loads(
            str(first["ordered_electrode_ids_json"])
        ),
        "ordered_gain_to_uv": json.loads(
            str(first["ordered_gain_to_uV_json"])
        ),
        "ordered_offset_to_uv": json.loads(
            str(first["ordered_offset_to_uV_json"])
        ),
        "trace_scaling_sha256": str(first["trace_scaling_sha256"]),
        "raw_timestamps_sha256": str(first["raw_timestamps_sha256"]),
        "raw_traces_sha256": str(first["raw_traces_sha256"]),
        "input_sample_count": int(first["input_sample_count"]),
        "output_sample_count": int(first["output_sample_count"]),
        "input_start_time_s": float(first["input_start_time_s"]),
        "input_stop_time_s": float(first["input_stop_time_s"]),
        "source_sampling_frequency_hz": float(
            first["source_sampling_frequency_hz"]
        ),
        "actual_sampling_frequency_hz": float(
            first["actual_sampling_frequency_hz"]
        ),
        "decimation_factor": int(first["decimation_factor"]),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance": json.loads(
            str(first["legacy_artifact_provenance_json"])
        ),
        "manifest": manifest,
    }
    if str(first["bundle_schema_version"]) != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Manifest bundle_schema_version is unsupported.")
    if int(first["n_channels"]) != len(result["ordered_electrode_ids"]):
        raise ValueError("Manifest n_channels disagrees with the dataset.")
    for field_name, expected in (
        ("output_start_time_s", _time_boundary(dataset, 0)),
        ("output_stop_time_s", _time_boundary(dataset, -1)),
    ):
        actual = float(first[field_name])
        matches = (
            np.isnan(actual) and np.isnan(expected)
        ) or actual == expected
        if not matches:
            raise ValueError(
                f"Manifest {field_name} disagrees with the dataset."
            )
    return validate_ripple_band_lfp_result(result)


def summarize_ripple_band_lfp_artifact_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Return Python-native database-facing scalar metadata."""
    validated = validate_ripple_band_lfp_result(bundle)
    metadata = validated["metadata"]
    return {
        "ripple_band_lfp_id": str(metadata["ripple_band_lfp_id"]),
        "animal_name": str(metadata["animal_name"]),
        "date": str(metadata["date"]),
        "epoch": str(metadata["epoch"]),
        "n_channels": int(len(validated["ordered_electrode_ids"])),
        "input_sample_count": int(validated["input_sample_count"]),
        "output_sample_count": int(validated["output_sample_count"]),
        "source_sampling_frequency_hz": float(
            validated["source_sampling_frequency_hz"]
        ),
        "actual_sampling_frequency_hz": float(
            validated["actual_sampling_frequency_hz"]
        ),
        "decimation_factor": int(validated["decimation_factor"]),
        "analysis_status": str(validated["analysis_status"]),
    }


def _validate_legacy_dataset_against_recomputation(
    *,
    legacy: Any,
    recomputed: Mapping[str, Any],
) -> None:
    """Require complete legacy metadata and exact recomputed scientific arrays."""
    expected = recomputed["dataset"]
    required_data_vars = {"filtered_lfp", "sampling_frequency_hz"}
    required_coords = {"time", "channel"}
    if not required_data_vars.issubset(
        legacy.data_vars
    ) or not required_coords.issubset(legacy.coords):
        raise ValueError("Legacy RippleBandLFP NetCDF is incomplete.")
    if tuple(legacy["filtered_lfp"].dims) != ("sample", "channel"):
        raise ValueError("Legacy filtered_lfp dimensions are not canonical.")
    metadata = recomputed["metadata"]
    parameters = recomputed["parameters"]
    required_attrs = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "epoch": metadata["epoch"],
        "cache_format": detect_ripples.LFP_CACHE_FORMAT,
    }
    for name, value in required_attrs.items():
        if str(_python_scalar(legacy.attrs.get(name, ""))) != str(value):
            raise ValueError(f"Legacy NetCDF attribute {name!r} does not match.")
    integer_attrs = {
        "cache_format_version": detect_ripples.LFP_CACHE_FORMAT_VERSION,
        "notch_filter_enabled": int(parameters["enable_notch_filter"]),
        "notch_harmonics": parameters["notch_harmonics"],
    }
    for name, value in integer_attrs.items():
        try:
            actual = int(_python_scalar(legacy.attrs[name]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Legacy NetCDF attribute {name!r} is missing.") from exc
        if actual != int(value):
            raise ValueError(f"Legacy NetCDF attribute {name!r} does not match.")
    numeric_attrs = {
        "notch_base_freq_hz": parameters["notch_base_freq_hz"],
        "notch_quality": parameters["notch_quality"],
        "lowcut_hz": parameters["lowcut_hz"],
        "highcut_hz": parameters["highcut_hz"],
        "target_sampling_frequency_hz": parameters[
            "target_sampling_frequency_hz"
        ],
    }
    for name, value in numeric_attrs.items():
        try:
            actual = float(_python_scalar(legacy.attrs[name]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Legacy NetCDF attribute {name!r} is missing.") from exc
        if not np.isclose(actual, float(value), rtol=0.0, atol=1e-12):
            raise ValueError(f"Legacy NetCDF attribute {name!r} does not match.")
    if np.asarray(legacy["channel"].values, dtype=int).tolist() != recomputed[
        "ordered_electrode_ids"
    ]:
        raise ValueError("Legacy NetCDF electrode order does not match NWB input.")
    try:
        np.testing.assert_array_equal(
            np.asarray(legacy["time"].values, dtype=float),
            np.asarray(expected["time"].values, dtype=float),
        )
        np.testing.assert_array_equal(
            np.asarray(legacy["filtered_lfp"].values, dtype=float),
            np.asarray(expected["filtered_lfp"].values, dtype=float),
        )
        np.testing.assert_array_equal(
            np.asarray(legacy["sampling_frequency_hz"].values, dtype=float),
            np.asarray(expected["sampling_frequency_hz"].values, dtype=float),
        )
    except AssertionError as exc:
        raise ValueError(
            "Legacy RippleBandLFP scientific values do not match exact NWB "
            "recomputation."
        ) from exc


def _legacy_run_log_provenance(
    path: Path | None,
    *,
    source_result_path: Path,
    animal_name: str,
    date: str,
    epoch: str,
    electrode_ids: Sequence[int],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an optional detector run log and return its provenance."""
    if path is None:
        return {}
    log_path = Path(path)
    if not log_path.is_file():
        raise FileNotFoundError(f"Legacy ripple run log not found: {log_path}")
    payload, run_log_sha256 = _read_stable_json(log_path)
    if payload.get("script") != "v1ca1.ripple.detect_ripples":
        raise ValueError("Legacy ripple run log has the wrong script.")
    logged = dict(payload.get("parameters", {}))
    if str(logged.get("animal_name")) != animal_name or str(
        logged.get("date")
    ) != date:
        raise ValueError("Legacy ripple run log session does not match.")
    epochs = [str(value) for value in logged.get("epochs", ())]
    if epoch not in epochs:
        raise ValueError("Legacy ripple run log does not include the selected epoch.")
    try:
        logged_electrodes = _electrode_ids(logged["ripple_channels"]).tolist()
    except KeyError as exc:
        raise ValueError(
            "Legacy ripple run log is missing ripple_channels."
        ) from exc
    if logged_electrodes != [int(value) for value in electrode_ids]:
        raise ValueError(
            "Legacy ripple run log electrode order does not match."
        )
    detector_parameters = {
        "notch_filter_enabled": bool(parameters["enable_notch_filter"]),
        "notch_base_freq_hz": float(parameters["notch_base_freq_hz"]),
        "notch_harmonics": int(parameters["notch_harmonics"]),
        "notch_quality": float(parameters["notch_quality"]),
    }
    for field_name, expected in detector_parameters.items():
        if field_name not in logged:
            raise ValueError(
                f"Legacy ripple run log is missing {field_name}."
            )
        if field_name == "notch_filter_enabled":
            matches = _database_bool(
                logged[field_name], name=field_name
            ) == expected
        elif field_name == "notch_harmonics":
            matches = _integer(
                logged[field_name], name=field_name, minimum=1
            ) == expected
        else:
            matches = np.isclose(
                _finite_float(logged[field_name], name=field_name),
                expected,
                rtol=0.0,
                atol=1e-12,
            )
        if not matches:
            raise ValueError(
                f"Legacy ripple run log {field_name} does not match."
            )
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Legacy ripple run log is missing outputs.")
    selected_epochs = [str(value) for value in outputs.get("selected_epochs", ())]
    if epoch not in selected_epochs:
        raise ValueError(
            "Legacy ripple run log outputs do not select this epoch."
        )
    epoch_paths = outputs.get("saved_lfp_cache_epoch_paths")
    if not isinstance(epoch_paths, Mapping) or epoch not in epoch_paths:
        raise ValueError(
            "Legacy ripple run log has no LFP cache path for this epoch."
        )
    source_path = Path(source_result_path).resolve(strict=True)
    logged_epoch_path = Path(str(epoch_paths[epoch])).resolve(strict=False)
    if logged_epoch_path != source_path:
        raise ValueError(
            "Legacy ripple run log is paired with a different epoch cache."
        )
    cache_dir = outputs.get("saved_lfp_cache_dir")
    if cache_dir is None or Path(str(cache_dir)).resolve(
        strict=False
    ) != source_path.parent:
        raise ValueError(
            "Legacy ripple run log cache directory does not match the artifact."
        )
    return {
        "source_run_log_path": str(log_path.resolve(strict=True)),
        "source_run_log_sha256": run_log_sha256,
        "source_v1ca1_git_commit": payload.get("git_commit"),
        "source_git_dirty": payload.get("git_dirty"),
        "source_timestamp_utc": payload.get("timestamp_utc"),
    }


def register_existing_ripple_band_lfp_artifact(
    *,
    source_result_path: Path,
    destination_path: Path,
    source_run_log_path: Path | None = None,
    overwrite: bool = False,
    **compute_kwargs: Any,
) -> dict[str, Any]:
    """Strictly register one complete legacy epoch NetCDF after NWB recomputation."""
    source_path = Path(source_result_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy RippleBandLFP NetCDF not found: {source_path}")
    recomputed = compute_selected_ripple_band_lfp(
        **compute_kwargs,
        artifact_origin="computed",
    )
    legacy_parameters = recomputed["parameters"]
    fixed_detector_parameters = {
        "lowcut_hz": float(detect_ripples.DEFAULT_LOWCUT_HZ),
        "highcut_hz": float(detect_ripples.DEFAULT_HIGHCUT_HZ),
        "filter_order": int(detect_ripples.DEFAULT_FILTER_ORDER),
        "target_sampling_frequency_hz": float(
            detect_ripples.DEFAULT_TARGET_NEW_SAMPLING_FREQUENCY
        ),
        "notch_base_freq_hz": float(detect_ripples.DEFAULT_NOTCH_BASE_FREQ),
        "notch_harmonics": int(detect_ripples.DEFAULT_NOTCH_HARMONICS),
        "notch_quality": float(detect_ripples.DEFAULT_NOTCH_QUALITY),
    }
    for field_name, expected in fixed_detector_parameters.items():
        if legacy_parameters[field_name] != expected:
            raise ValueError(
                "Legacy RippleBandLFP registration requires the detector's "
                f"fixed parameters; {field_name} differs."
            )
    run_log_provenance = _legacy_run_log_provenance(
        source_run_log_path,
        source_result_path=source_path,
        animal_name=recomputed["metadata"]["animal_name"],
        date=recomputed["metadata"]["date"],
        epoch=recomputed["metadata"]["epoch"],
        electrode_ids=recomputed["ordered_electrode_ids"],
        parameters=legacy_parameters,
    )
    source_fingerprint_before = _stable_file_fingerprint(source_path)
    try:
        legacy = _load_dataset(source_path)
    except Exception as exc:
        raise ValueError(
            f"Legacy RippleBandLFP NetCDF is unreadable: {source_path}"
        ) from exc
    _validate_legacy_dataset_against_recomputation(
        legacy=legacy,
        recomputed=recomputed,
    )
    source_fingerprint_after = _stable_file_fingerprint(source_path)
    if source_fingerprint_before != source_fingerprint_after:
        raise ValueError(
            "Legacy RippleBandLFP NetCDF changed during validation."
        )
    provenance = {
        "source_result_path": str(source_path.resolve(strict=True)),
        "source_result_sha256": source_fingerprint_before["sha256"],
        "source_epoch": recomputed["metadata"]["epoch"],
        "verification": "exact_scientific_values_recomputed_from_selected_nwb_input",
        **run_log_provenance,
    }
    recomputed["legacy_artifact_provenance"] = provenance
    recomputed["artifact_origin"] = "registered_existing"
    recomputed["dataset"].attrs["artifact_origin"] = "registered_existing"
    recomputed = validate_ripple_band_lfp_result(recomputed)
    paths = write_ripple_band_lfp_artifact(
        recomputed,
        destination_path,
        overwrite=overwrite,
    )
    return {
        **recomputed,
        **paths,
        "_created_artifact_paths": [
            str(paths["artifact_manifest_path"]),
            str(paths["channel_qc_path"]),
            str(paths["result_path"]),
        ],
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_ENABLE_NOTCH_FILTER",
    "DEFAULT_FILTER_ORDER",
    "DEFAULT_HIGHCUT_HZ",
    "DEFAULT_LOWCUT_HZ",
    "DEFAULT_NOTCH_BASE_FREQ_HZ",
    "DEFAULT_NOTCH_HARMONICS",
    "DEFAULT_NOTCH_QUALITY",
    "DEFAULT_SAMPLES_FOR_RATE_ESTIMATION",
    "DEFAULT_TARGET_SAMPLING_FREQUENCY_HZ",
    "MANIFEST_COLUMNS",
    "MANIFEST_FILENAME",
    "MANUSCRIPT_PARAMETERS",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "QC_COLUMNS",
    "QC_FILENAME",
    "RESULT_FILENAME",
    "SAMPLING_FREQUENCY_ESTIMATION_METHOD",
    "compute_selected_ripple_band_lfp",
    "get_ripple_band_lfp_artifact_paths",
    "inspect_selected_ripple_band_lfp_nwb_inputs",
    "load_selected_ripple_band_lfp_nwb_inputs",
    "load_ripple_band_lfp_artifact",
    "register_existing_ripple_band_lfp_artifact",
    "summarize_ripple_band_lfp_artifact_bundle",
    "validate_ripple_band_lfp_parameters",
    "validate_ripple_band_lfp_result",
    "write_ripple_band_lfp_artifact",
]
