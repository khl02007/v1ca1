"""Database-free within-epoch path-specific place decoding artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "path_specific_place_decoding"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
FOLD_QC_FILENAME = "fold_qc.parquet"
SUMMARY_FILENAME = "decoding_summary.parquet"
BINNED_ERROR_FILENAME = "decoding_error_by_position.parquet"
TRUE_FILENAME = "true_place.npz"
DECODED_FILENAME = "decoded_place.npz"

NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_SELECTED_UNITS_TABLE_NAME = "path_specific_place_decoding_selected_units"
NWB_FOLD_QC_TABLE_NAME = "path_specific_place_decoding_fold_qc"
NWB_SUMMARY_TABLE_NAME = "path_specific_place_decoding_summary"
NWB_BINNED_ERROR_TABLE_NAME = "path_specific_place_decoding_error_by_position"
NWB_TRUE_POSITION_TIMESERIES_NAME = "path_specific_place_decoding_true_position"
NWB_DECODED_POSITION_TIMESERIES_NAME = (
    "path_specific_place_decoding_decoded_position"
)
NWB_DECODING_SUPPORT_NAME = "path_specific_place_decoding_support"
NWB_PROVENANCE_TABLE_NAME = "path_specific_place_decoding_provenance"

DEFAULT_N_FOLDS = 5
DEFAULT_DECODING_BIN_SIZE_S = 0.02
DEFAULT_SLIDING_WINDOW_SIZE_BINS = 4
DEFAULT_SPATIAL_BIN_SIZE_CM = 4.0
DEFAULT_RANDOM_SEED = 47
DEFAULT_ERROR_MODE = "signed"
DEFAULT_ERROR_SUMMARY = "median_iqr"
DEFAULT_MIN_BIN_COUNT = 5

MANUSCRIPT_PARAMETERS = MappingProxyType(
    {
        "n_folds": DEFAULT_N_FOLDS,
        "decoding_bin_size_s": DEFAULT_DECODING_BIN_SIZE_S,
        "sliding_window_size_bins": DEFAULT_SLIDING_WINDOW_SIZE_BINS,
        "spatial_bin_size_cm": DEFAULT_SPATIAL_BIN_SIZE_CM,
        "random_seed": DEFAULT_RANDOM_SEED,
    }
)
OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "coordinate": "concatenated_path_specific_linear_position",
        "coordinate_unit": "cm",
        "trajectory_order": tuple(TRAJECTORY_TYPES),
        "path_orientation": "from_center",
        "unit_policy": "all_region_sorted_spikes_group_units",
        "cross_validation": "lap_wise_kfold_per_trajectory_then_pooled",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": DEFAULT_ERROR_MODE,
        "error_summary": DEFAULT_ERROR_SUMMARY,
        "min_bin_count": DEFAULT_MIN_BIN_COUNT,
    }
)
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_units",
    "no_valid_decodes",
)
FOLD_STATUSES = (
    "valid",
    "no_units",
    "no_train_movement",
    "no_test_movement",
    "no_test_count_bins",
)
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (*IDENTITY_COLUMNS, "selection_index")
FOLD_QC_COLUMNS = (
    "path_specific_place_decoding_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "fold",
    "n_train_laps",
    "n_test_laps",
    "train_duration_s",
    "test_duration_s",
    "n_decoded_samples",
    "qc_status",
    "qc_message",
)
SUMMARY_COLUMNS = (
    "path_specific_place_decoding_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "model",
    "coordinate_unit",
    "n_units",
    "n_folds_expected",
    "n_folds_valid",
    "mae",
    "rmse",
    "mean_signed_error",
    "median_abs_error",
    "n_samples",
    "analysis_status",
)
BINNED_VALUE_COLUMNS = (
    "bin_left",
    "bin_right",
    "bin_center",
    "n",
    "center",
    "yerr_low",
    "yerr_high",
)
BINNED_ERROR_COLUMNS = (
    "path_specific_place_decoding_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "coordinate_unit",
    *BINNED_VALUE_COLUMNS,
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "path_specific_place_decoding_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "n_folds",
    "decoding_bin_size_s",
    "sliding_window_size_bins",
    "spatial_bin_size_cm",
    "random_seed",
    "n_units",
    "n_folds_valid",
    "n_decoded_samples",
    "selected_units_sha256",
    "analysis_status",
    "artifact_origin",
)


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe non-empty path component."""
    component = str(value)
    if not component or Path(component).name != component or component in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return component


def _uuid_string(value: Any, *, name: str) -> str:
    """Return one canonical UUID string."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def get_path_specific_decoding_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    path_specific_place_decoding_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first decoding artifact bundle."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "region": region,
        }.items()
    }
    result_id = _uuid_string(
        path_specific_place_decoding_id,
        name="path_specific_place_decoding_id",
    )
    artifact_dir = (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["region"]
        / result_id
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "selected_units_path": artifact_dir / SELECTED_UNITS_FILENAME,
        "fold_qc_path": artifact_dir / FOLD_QC_FILENAME,
        "decoding_summary_path": artifact_dir / SUMMARY_FILENAME,
        "binned_error_path": artifact_dir / BINNED_ERROR_FILENAME,
        "true_path": artifact_dir / TRUE_FILENAME,
        "decoded_path": artifact_dir / DECODED_FILENAME,
    }


def validate_path_specific_decoding_parameters(
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    decoding_bin_size_s: float = DEFAULT_DECODING_BIN_SIZE_S,
    sliding_window_size_bins: int = DEFAULT_SLIDING_WINDOW_SIZE_BINS,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Return validated lap-wise Bayesian decoding parameters."""
    integer_values = {
        "n_folds": n_folds,
        "sliding_window_size_bins": sliding_window_size_bins,
        "random_seed": random_seed,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        integer_values[name] = int(value)
    if integer_values["n_folds"] < 2:
        raise ValueError("n_folds must be at least 2.")
    if integer_values["sliding_window_size_bins"] < 1:
        raise ValueError("sliding_window_size_bins must be positive.")
    if integer_values["random_seed"] < 0:
        raise ValueError("random_seed must be non-negative.")

    numeric_values = {
        "decoding_bin_size_s": float(decoding_bin_size_s),
        "spatial_bin_size_cm": float(spatial_bin_size_cm),
    }
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in numeric_values.values()
    ):
        raise ValueError("Decoding and spatial bin sizes must be positive and finite.")
    return {**integer_values, **numeric_values}


def _interval_summary(intervals: Any) -> tuple[int, float]:
    """Return the count and duration of one IntervalSet-like value."""
    starts = np.asarray(intervals.start, dtype=float).reshape(-1)
    ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    if starts.shape != ends.shape or np.any(ends < starts):
        raise ValueError("Interval starts and ends must align and be ordered.")
    return int(starts.size), float(np.sum(ends - starts))


def _make_tsd(times: np.ndarray, values: np.ndarray, support: Any) -> Any:
    """Build one second-based Pynapple Tsd lazily."""
    import pynapple as nap

    return nap.Tsd(
        t=np.asarray(times, dtype=float),
        d=np.asarray(values, dtype=float),
        time_support=support,
        time_units="s",
    )


def _empty_tsd(support: Any) -> Any:
    """Return one empty second-based Tsd."""
    return _make_tsd(np.array([], dtype=float), np.array([], dtype=float), support)


def build_concatenated_path_specific_position(
    *,
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_interval: Any,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
) -> tuple[Any, np.ndarray, float]:
    """Build the legacy four-segment path-specific place coordinate."""
    expected = set(TRAJECTORY_TYPES)
    if set(trajectory_intervals) != expected:
        raise ValueError("Trajectory intervals must contain exactly four paths.")
    if set(graph_inputs) != expected:
        raise ValueError("Path graphs must contain exactly four configurations.")
    parameters = validate_path_specific_decoding_parameters(
        spatial_bin_size_cm=spatial_bin_size_cm
    )
    from v1ca1.spyglass.stability import build_task_progression_from_graph
    from v1ca1.task_progression.decoding_comparison import _make_intervalset

    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    support_starts: list[np.ndarray] = []
    support_ends: list[np.ndarray] = []
    lengths: list[float] = []
    normalized_by_trajectory: list[tuple[Any, np.ndarray]] = []
    for trajectory_type in TRAJECTORY_TYPES:
        progression, graph_length_cm = build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals[trajectory_type],
            graph_inputs=graph_inputs[trajectory_type],
            trajectory_type=trajectory_type,
        )
        restricted = progression.restrict(trajectory_intervals[trajectory_type])
        values = np.asarray(restricted.d, dtype=float).reshape(-1)
        if trajectory_type.endswith("_to_center"):
            values = 1.0 - values
        normalized_by_trajectory.append((restricted, values))
        lengths.append(float(graph_length_cm))

    common_length = lengths[0]
    if common_length <= 0.0 or any(
        not np.isclose(length, common_length, rtol=1e-10, atol=1e-12)
        for length in lengths
    ):
        raise ValueError("The four path graphs must have one common length.")

    for trajectory_index, (trajectory_type, item) in enumerate(
        zip(TRAJECTORY_TYPES, normalized_by_trajectory, strict=True)
    ):
        restricted, normalized_values = item
        times = np.asarray(restricted.t, dtype=float).reshape(-1)
        values = (
            normalized_values * common_length
            + trajectory_index * common_length
        )
        finite = np.isfinite(times) & np.isfinite(values)
        time_chunks.append(times[finite])
        value_chunks.append(values[finite])
        starts = np.asarray(trajectory_intervals[trajectory_type].start, dtype=float)
        ends = np.asarray(trajectory_intervals[trajectory_type].end, dtype=float)
        support_starts.append(starts)
        support_ends.append(ends)
    all_times = np.concatenate(time_chunks) if time_chunks else np.array([])
    all_values = np.concatenate(value_chunks) if value_chunks else np.array([])
    order = np.argsort(all_times, kind="stable")
    path_support = _make_intervalset(support_starts, support_ends).intersect(
        movement_interval
    )
    feature = _make_tsd(all_times[order], all_values[order], path_support)
    bin_size = float(parameters["spatial_bin_size_cm"])
    bins = np.arange(
        0.0,
        common_length * len(TRAJECTORY_TYPES) + bin_size,
        bin_size,
        dtype=float,
    )
    return feature, bins, common_length


def _selected_unit_table(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return all selected regional units in stable identity order."""
    from v1ca1.spyglass.path_specific_place import _identity_rows

    rows = _identity_rows(spikes, stable_unit_ids)
    records = [
        {
            **{name: str(row[name]) for name in IDENTITY_COLUMNS},
            "selection_index": index,
        }
        for index, row in enumerate(rows)
    ]
    return pd.DataFrame.from_records(records, columns=SELECTED_UNIT_COLUMNS)


def _validate_tsd(tsd: Any, *, name: str, allow_empty: bool = False) -> None:
    """Validate one second-based true or decoded Tsd."""
    times = np.asarray(tsd.t, dtype=float).reshape(-1)
    values = np.asarray(tsd.d, dtype=float).reshape(-1)
    if times.shape != values.shape or (not allow_empty and not times.size):
        raise ValueError(f"{name} must contain aligned samples.")
    if not np.all(np.isfinite(times)) or (
        times.size > 1 and np.any(np.diff(times) <= 0)
    ):
        raise ValueError(f"{name} timestamps must be finite and increasing.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} values must be finite.")


def _decode_fold(
    *,
    spikes: Any,
    feature: Any,
    train_interval: Any,
    test_interval: Any,
    movement_interval: Any,
    bins: np.ndarray,
    decoding_bin_size_s: float,
    sliding_window_size_bins: int,
) -> tuple[Any | None, Any | None, str, str, float, float]:
    """Decode one fold, returning explicit support-only terminal states."""
    import pynapple as nap

    from v1ca1.task_progression.decoding_comparison import (
        _filter_epochs_with_count_bins,
    )

    train = train_interval.intersect(movement_interval)
    test = test_interval.intersect(movement_interval)
    _, train_duration = _interval_summary(train)
    _, test_duration = _interval_summary(test)
    if train_duration <= 0.0:
        return (
            None,
            None,
            "no_train_movement",
            "No training movement support.",
            train_duration,
            test_duration,
        )
    if test_duration <= 0.0:
        return (
            None,
            None,
            "no_test_movement",
            "No test movement support.",
            train_duration,
            test_duration,
        )
    filtered_test = _filter_epochs_with_count_bins(
        spikes,
        test,
        float(decoding_bin_size_s),
    )
    _, filtered_duration = _interval_summary(filtered_test)
    if filtered_duration <= 0.0:
        return (
            None,
            None,
            "no_test_count_bins",
            "No test interval produced a count bin.",
            train_duration,
            0.0,
        )
    tuning_curves = nap.compute_tuning_curves(
        data=spikes,
        features=feature,
        bins=[np.asarray(bins, dtype=float)],
        epochs=train,
    )
    decoded, _ = nap.decode_bayes(
        tuning_curves=tuning_curves,
        data=spikes,
        epochs=filtered_test,
        sliding_window_size=int(sliding_window_size_bins),
        bin_size=float(decoding_bin_size_s),
    )
    true = feature.restrict(filtered_test)
    _validate_tsd(true, name="fold true")
    _validate_tsd(decoded, name="fold decoded")
    return true, decoded, "valid", "", train_duration, filtered_duration


def _terminal_summary(
    metadata: Mapping[str, str],
    *,
    n_units: int,
    n_folds: int,
    status: str,
) -> pd.DataFrame:
    """Return one explicit terminal summary row."""
    return pd.DataFrame.from_records(
        [
            {
                **metadata,
                "model": "path_specific_place",
                "coordinate_unit": "cm",
                "n_units": int(n_units),
                "n_folds_expected": int(n_folds),
                "n_folds_valid": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "mean_signed_error": np.nan,
                "median_abs_error": np.nan,
                "n_samples": 0,
                "analysis_status": status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )


def _analysis_status(n_units: int, n_folds: int, n_valid: int) -> str:
    """Return one canonical decoder status."""
    if n_units == 0:
        return "no_units"
    if n_valid == n_folds:
        return "valid"
    if n_valid == 0:
        return "no_valid_decodes"
    return "partial_valid"


def compute_path_specific_place_decoding(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    path_specific_place_decoding_id: Any,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_interval: Any,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    decoding_bin_size_s: float = DEFAULT_DECODING_BIN_SIZE_S,
    sliding_window_size_bins: int = DEFAULT_SLIDING_WINDOW_SIZE_BINS,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Compute one all-unit within-epoch path-specific place decoder."""
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256
    from v1ca1.task_progression.decoding_comparison import (
        _concatenate_tsds,
        build_train_test_folds,
        summarize_decoding_error_by_position,
        summarize_decoding_metrics,
    )

    parameters = validate_path_specific_decoding_parameters(
        n_folds=n_folds,
        decoding_bin_size_s=decoding_bin_size_s,
        sliding_window_size_bins=sliding_window_size_bins,
        spatial_bin_size_cm=spatial_bin_size_cm,
        random_seed=random_seed,
    )
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
        }.items()
    }
    result_id = _uuid_string(
        path_specific_place_decoding_id,
        name="path_specific_place_decoding_id",
    )
    parameter_name = str(parameter_name).strip()
    if not parameter_name or len(parameter_name) > 64:
        raise ValueError("parameter_name must be non-empty and at most 64 characters.")
    parameter_payload = {
        "path_specific_place_decoding_param_name": parameter_name,
        **parameters,
    }
    expected_parameter_sha256 = provenance_sha256(parameter_payload)
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    if str(parameter_sha256) != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match the effective parameters.")
    expected_output_sha256 = provenance_sha256(dict(OUTPUT_RULE))
    if output_rule_sha256 is None:
        output_rule_sha256 = expected_output_sha256
    if str(output_rule_sha256) != expected_output_sha256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")

    metadata = {
        "path_specific_place_decoding_id": result_id,
        **components,
    }
    selected_units = _selected_unit_table(spikes, stable_unit_ids)
    selected_units_sha256 = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    n_units = len(selected_units)
    feature, bins, path_length_cm = build_concatenated_path_specific_position(
        position=position,
        trajectory_intervals=trajectory_intervals,
        graph_inputs=graph_inputs,
        movement_interval=movement_interval,
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
    )
    train_folds, test_folds = build_train_test_folds(
        dict(trajectory_intervals),
        n_folds=parameters["n_folds"],
        random_state=parameters["random_seed"],
    )
    true_chunks: list[Any] = []
    decoded_chunks: list[Any] = []
    fold_rows: list[dict[str, Any]] = []
    for fold in range(parameters["n_folds"]):
        train_laps, _ = _interval_summary(train_folds[fold])
        test_laps, _ = _interval_summary(test_folds[fold])
        if n_units == 0:
            true = decoded = None
            qc_status = "no_units"
            qc_message = "No regional units were selected."
            train_duration = test_duration = 0.0
        else:
            (
                true,
                decoded,
                qc_status,
                qc_message,
                train_duration,
                test_duration,
            ) = _decode_fold(
                spikes=spikes,
                feature=feature,
                train_interval=train_folds[fold],
                test_interval=test_folds[fold],
                movement_interval=movement_interval,
                bins=bins,
                decoding_bin_size_s=parameters["decoding_bin_size_s"],
                sliding_window_size_bins=parameters[
                    "sliding_window_size_bins"
                ],
            )
        n_samples = 0 if decoded is None else len(np.asarray(decoded.t))
        if true is not None and decoded is not None:
            true_chunks.append(true)
            decoded_chunks.append(decoded)
        fold_rows.append(
            {
                **metadata,
                "fold": fold,
                "n_train_laps": train_laps,
                "n_test_laps": test_laps,
                "train_duration_s": train_duration,
                "test_duration_s": test_duration,
                "n_decoded_samples": n_samples,
                "qc_status": qc_status,
                "qc_message": qc_message,
            }
        )
    fold_qc = pd.DataFrame.from_records(fold_rows).loc[:, list(FOLD_QC_COLUMNS)]
    n_folds_valid = int((fold_qc["qc_status"] == "valid").sum())
    status = _analysis_status(n_units, parameters["n_folds"], n_folds_valid)
    true = _concatenate_tsds(true_chunks, movement_interval)
    decoded = _concatenate_tsds(decoded_chunks, movement_interval)
    if n_folds_valid:
        metrics = summarize_decoding_metrics(true, decoded)
        summary = pd.DataFrame.from_records(
            [
                {
                    **metadata,
                    "model": "path_specific_place",
                    "coordinate_unit": "cm",
                    "n_units": n_units,
                    "n_folds_expected": parameters["n_folds"],
                    "n_folds_valid": n_folds_valid,
                    **metrics,
                    "analysis_status": status,
                }
            ],
            columns=SUMMARY_COLUMNS,
        )
        binned = summarize_decoding_error_by_position(
            true,
            decoded,
            bin_edges=bins,
            error_mode=DEFAULT_ERROR_MODE,
            summary=DEFAULT_ERROR_SUMMARY,
            min_count=DEFAULT_MIN_BIN_COUNT,
        )
        binned_error = binned.assign(
            **metadata,
            coordinate_unit="cm",
        ).loc[:, list(BINNED_ERROR_COLUMNS)]
    else:
        summary = _terminal_summary(
            metadata,
            n_units=n_units,
            n_folds=parameters["n_folds"],
            status=status,
        )
        binned_error = pd.DataFrame(columns=BINNED_ERROR_COLUMNS)
    return {
        "metadata": metadata,
        "parameters": {
            "parameter_name": parameter_name,
            "parameter_sha256": str(parameter_sha256),
            "output_rule_sha256": str(output_rule_sha256),
            **parameters,
        },
        "selected_units": selected_units,
        "fold_qc": fold_qc,
        "summary": summary,
        "binned_error": binned_error,
        "true": true,
        "decoded": decoded,
        "path_length_cm": path_length_cm,
        "n_units": n_units,
        "n_folds_expected": parameters["n_folds"],
        "n_folds_valid": n_folds_valid,
        "n_decoded_samples": int(summary.iloc[0]["n_samples"]),
        "selected_units_sha256": selected_units_sha256,
        "analysis_status": status,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def _file_sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_exact_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    name: str,
) -> None:
    """Require one table to contain exactly the canonical columns."""
    if list(table.columns) != list(columns):
        raise ValueError(f"{name} columns do not match the canonical schema.")


def validate_path_specific_decoding_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy one in-memory decoding result bundle."""
    copied = dict(result)
    metadata = dict(copied["metadata"])
    parameters = dict(copied["parameters"])
    result_id = _uuid_string(
        metadata["path_specific_place_decoding_id"],
        name="path_specific_place_decoding_id",
    )
    metadata["path_specific_place_decoding_id"] = result_id
    effective = validate_path_specific_decoding_parameters(
        **{name: parameters[name] for name in MANUSCRIPT_PARAMETERS}
    )
    if any(effective[name] != parameters[name] for name in effective):
        raise ValueError("Result parameters are not canonical.")
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256

    expected_parameter_sha256 = provenance_sha256(
        {
            "path_specific_place_decoding_param_name": parameters[
                "parameter_name"
            ],
            **effective,
        }
    )
    if str(parameters["parameter_sha256"]) != expected_parameter_sha256:
        raise ValueError("Result parameter digest is stale.")
    if str(parameters["output_rule_sha256"]) != provenance_sha256(
        dict(OUTPUT_RULE)
    ):
        raise ValueError("Result output-rule digest is stale.")
    for name, columns in (
        ("selected_units", SELECTED_UNIT_COLUMNS),
        ("fold_qc", FOLD_QC_COLUMNS),
        ("summary", SUMMARY_COLUMNS),
        ("binned_error", BINNED_ERROR_COLUMNS),
    ):
        _validate_exact_columns(copied[name], columns, name=name)
    _validate_tsd(copied["true"], name="true", allow_empty=True)
    _validate_tsd(copied["decoded"], name="decoded", allow_empty=True)
    if copied["analysis_status"] not in ANALYSIS_STATUSES:
        raise ValueError("Unsupported analysis_status.")
    if copied["artifact_origin"] not in {"computed", "registered_existing"}:
        raise ValueError("Unsupported artifact_origin.")
    selected_digest = copied["selected_units_sha256"]
    expected_digest = unit_identity_sha256(
        copied["selected_units"]
        .loc[:, ["spikesorting_merge_id", "unit_id"]]
        .to_dict("records")
    )
    if str(selected_digest) != expected_digest:
        raise ValueError("Selected-unit digest does not match selected_units.")
    if int(copied["n_units"]) != len(copied["selected_units"]):
        raise ValueError("n_units does not match selected_units.")
    n_folds_valid = int((copied["fold_qc"]["qc_status"] == "valid").sum())
    if int(copied["n_folds_valid"]) != n_folds_valid:
        raise ValueError("n_folds_valid does not match fold_qc.")
    if len(copied["fold_qc"]) != int(copied["n_folds_expected"]):
        raise ValueError("fold_qc does not contain one row per expected fold.")
    if len(copied["summary"]) != 1:
        raise ValueError("summary must contain exactly one row.")
    summary_row = copied["summary"].iloc[0]
    if (
        int(summary_row["n_samples"]) != int(copied["n_decoded_samples"])
        or str(summary_row["analysis_status"]) != str(copied["analysis_status"])
    ):
        raise ValueError("Summary scalars do not match the result bundle.")
    for table_name in ("fold_qc", "summary", "binned_error"):
        table = copied[table_name]
        for field_name, expected in metadata.items():
            if not table.empty and not np.all(
                table[field_name].astype(str) == str(expected)
            ):
                raise ValueError(
                    f"{table_name} does not match metadata field {field_name!r}."
                )
    copied["metadata"] = metadata
    copied["parameters"] = {**parameters, **effective}
    return copied


_NWB_TABLE_COLUMNS = {
    "selected_units": SELECTED_UNIT_COLUMNS,
    "fold_qc": FOLD_QC_COLUMNS,
    "summary": SUMMARY_COLUMNS,
    "binned_error": BINNED_ERROR_COLUMNS,
}
_NWB_TABLE_NAMES = {
    "selected_units": NWB_SELECTED_UNITS_TABLE_NAME,
    "fold_qc": NWB_FOLD_QC_TABLE_NAME,
    "summary": NWB_SUMMARY_TABLE_NAME,
    "binned_error": NWB_BINNED_ERROR_TABLE_NAME,
}
_NWB_TABLE_TEXT_COLUMNS = {
    "selected_units": {
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "group_unit_id",
    },
    "fold_qc": {
        "path_specific_place_decoding_id",
        "animal_name",
        "date",
        "region",
        "epoch",
        "qc_status",
        "qc_message",
    },
    "summary": {
        "path_specific_place_decoding_id",
        "animal_name",
        "date",
        "region",
        "epoch",
        "model",
        "coordinate_unit",
        "analysis_status",
    },
    "binned_error": {
        "path_specific_place_decoding_id",
        "animal_name",
        "date",
        "region",
        "epoch",
        "coordinate_unit",
    },
}
_NWB_TABLE_INTEGER_COLUMNS = {
    "selected_units": {"selection_index"},
    "fold_qc": {
        "fold",
        "n_train_laps",
        "n_test_laps",
        "n_decoded_samples",
    },
    "summary": {
        "n_units",
        "n_folds_expected",
        "n_folds_valid",
        "n_samples",
    },
    "binned_error": {"n"},
}
_NWB_PROVENANCE_COLUMNS = (
    "artifact_schema_version",
    "metadata_json",
)


def _decoded_nwb_text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    """Return one UTF-8 string fetched from an NWB object."""
    if isinstance(value, (bytes, np.bytes_)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} is not valid UTF-8.") from exc
    value = str(value)
    if not value and not allow_empty:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _canonical_nwb_table(
    table: pd.DataFrame,
    *,
    artifact_name: str,
) -> pd.DataFrame:
    """Return one exact-schema decoding table with deterministic dtypes."""
    if artifact_name not in _NWB_TABLE_COLUMNS:
        raise ValueError(f"Unknown decoding table {artifact_name!r}.")
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Path-specific decoding tables must be pandas DataFrames.")
    columns = _NWB_TABLE_COLUMNS[artifact_name]
    observed = tuple(str(column) for column in table.columns)
    if len(observed) != len(columns) or set(observed) != set(columns):
        raise ValueError(
            f"{artifact_name} must contain exactly columns {tuple(columns)!r}."
        )
    output = table.loc[:, list(columns)].copy().reset_index(drop=True)
    text_columns = _NWB_TABLE_TEXT_COLUMNS[artifact_name]
    integer_columns = _NWB_TABLE_INTEGER_COLUMNS[artifact_name]
    for column in columns:
        if column in text_columns:
            output[column] = output[column].map(
                lambda value, column=column: _decoded_nwb_text(
                    value,
                    name=f"{artifact_name}.{column}",
                    allow_empty=(column == "qc_message"),
                )
            )
        elif column in integer_columns:
            values = pd.to_numeric(output[column], errors="raise").to_numpy(
                dtype=float
            )
            if not np.all(np.isfinite(values)) or not np.allclose(
                values,
                np.rint(values),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"{artifact_name}.{column} must contain finite integers."
                )
            output[column] = np.rint(values).astype(np.int64)
        else:
            values = pd.to_numeric(output[column], errors="raise").to_numpy(
                dtype=float
            )
            if np.any(np.isinf(values)):
                raise ValueError(f"{artifact_name}.{column} cannot contain infinity.")
            output[column] = values.astype(float)
    return output


def _decoding_table_to_dynamic_table(
    table: pd.DataFrame,
    *,
    artifact_name: str,
) -> Any:
    """Convert one canonical decoding DataFrame to an NWB DynamicTable."""
    from hdmf.common import DynamicTable, VectorData

    canonical = _canonical_nwb_table(table, artifact_name=artifact_name)
    columns = _NWB_TABLE_COLUMNS[artifact_name]
    description = (
        f"PathSpecificPlaceDecoding {artifact_name.replace('_', ' ')}; "
        f"v1ca1 NWB artifact schema {NWB_ARTIFACT_SCHEMA_VERSION}."
    )
    if canonical.empty:
        vector_columns = []
        for column in columns:
            if column in _NWB_TABLE_TEXT_COLUMNS[artifact_name]:
                data = np.asarray([], dtype="S1")
            elif column in _NWB_TABLE_INTEGER_COLUMNS[artifact_name]:
                data = np.asarray([], dtype=np.int64)
            else:
                data = np.asarray([], dtype=float)
            vector_columns.append(
                VectorData(
                    name=column,
                    description=f"Canonical {artifact_name} field {column!r}.",
                    data=data,
                )
            )
        return DynamicTable(
            name=_NWB_TABLE_NAMES[artifact_name],
            description=description,
            columns=vector_columns,
        )
    return DynamicTable.from_dataframe(
        name=_NWB_TABLE_NAMES[artifact_name],
        df=canonical,
        table_description=description,
        columns=[
            {
                "name": column,
                "description": f"Canonical {artifact_name} field {column!r}.",
            }
            for column in columns
        ],
    )


def _decoding_table_from_dynamic_table(
    nwb_table: Any,
    *,
    artifact_name: str,
) -> pd.DataFrame:
    """Return one canonical decoding table from a fetched NWB object."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table
    elif isinstance(nwb_table, DynamicTable):
        expected_name = _NWB_TABLE_NAMES[artifact_name]
        if str(nwb_table.name) != expected_name:
            raise ValueError(
                f"Unexpected decoding NWB object name {nwb_table.name!r}; "
                f"expected {expected_name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("Decoding tabular NWB objects must be DynamicTables.")
    return _canonical_nwb_table(table, artifact_name=artifact_name)


def selected_units_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert selected decoding units to an NWB DynamicTable."""
    return _decoding_table_to_dynamic_table(table, artifact_name="selected_units")


def selected_units_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Return selected decoding units from a fetched NWB object."""
    return _decoding_table_from_dynamic_table(
        nwb_table,
        artifact_name="selected_units",
    )


def fold_qc_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert fold-level decoding QC to an NWB DynamicTable."""
    return _decoding_table_to_dynamic_table(table, artifact_name="fold_qc")


def fold_qc_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Return fold-level decoding QC from a fetched NWB object."""
    return _decoding_table_from_dynamic_table(nwb_table, artifact_name="fold_qc")


def decoding_summary_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert the decoding summary to an NWB DynamicTable."""
    return _decoding_table_to_dynamic_table(table, artifact_name="summary")


def decoding_summary_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Return the decoding summary from a fetched NWB object."""
    return _decoding_table_from_dynamic_table(nwb_table, artifact_name="summary")


def binned_error_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert position-binned decoding error to an NWB DynamicTable."""
    return _decoding_table_to_dynamic_table(table, artifact_name="binned_error")


def binned_error_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Return position-binned decoding error from a fetched NWB object."""
    return _decoding_table_from_dynamic_table(
        nwb_table,
        artifact_name="binned_error",
    )


def _position_to_time_series(tsd: Any, *, kind: str) -> Any:
    """Convert one true or decoded position Tsd to an NWB TimeSeries."""
    from pynwb import TimeSeries

    if kind not in {"true", "decoded"}:
        raise ValueError("Position TimeSeries kind must be 'true' or 'decoded'.")
    _validate_tsd(tsd, name=kind, allow_empty=True)
    name = (
        NWB_TRUE_POSITION_TIMESERIES_NAME
        if kind == "true"
        else NWB_DECODED_POSITION_TIMESERIES_NAME
    )
    return TimeSeries(
        name=name,
        data=np.asarray(tsd.d, dtype=float).reshape(-1),
        unit="cm",
        timestamps=np.asarray(tsd.t, dtype=float).reshape(-1),
        description=(
            f"{kind.capitalize()} concatenated path-specific position in "
            "ephys-reference time; "
            f"v1ca1 NWB artifact schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
    )


def true_position_to_time_series(tsd: Any) -> Any:
    """Convert measured path-specific position to an NWB TimeSeries."""
    return _position_to_time_series(tsd, kind="true")


def decoded_position_to_time_series(tsd: Any) -> Any:
    """Convert decoded path-specific position to an NWB TimeSeries."""
    return _position_to_time_series(tsd, kind="decoded")


def _time_series_arrays(nwb_series: Any, *, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return validated timestamps and values from one decoding TimeSeries."""
    from pynwb import TimeSeries

    if not isinstance(nwb_series, TimeSeries):
        raise TypeError("Decoded-position NWB objects must be TimeSeries objects.")
    expected_name = (
        NWB_TRUE_POSITION_TIMESERIES_NAME
        if kind == "true"
        else NWB_DECODED_POSITION_TIMESERIES_NAME
    )
    if str(nwb_series.name) != expected_name:
        raise ValueError(
            f"Unexpected {kind} position TimeSeries name {nwb_series.name!r}."
        )
    if str(nwb_series.unit) != "cm":
        raise ValueError(f"{kind} position TimeSeries must use centimeters.")
    if nwb_series.timestamps is None:
        raise ValueError(f"{kind} position TimeSeries must use explicit timestamps.")
    times = np.asarray(nwb_series.timestamps[:], dtype=float).reshape(-1)
    values = np.asarray(nwb_series.data[:], dtype=float).reshape(-1)
    if times.shape != values.shape:
        raise ValueError(f"{kind} position timestamps and values must align.")
    if not np.all(np.isfinite(times)) or (
        times.size > 1 and np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(f"{kind} position timestamps must be finite and increasing.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{kind} position values must be finite.")
    return times, values


def _support_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return validated ordered support bounds in seconds."""
    starts = np.asarray(intervals.start, dtype=float).reshape(-1)
    ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    if starts.shape != ends.shape or not np.all(np.isfinite(starts)) or (
        not np.all(np.isfinite(ends))
    ):
        raise ValueError("Decoding support bounds must be aligned and finite.")
    if np.any(ends < starts) or (
        starts.size > 1 and np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError("Decoding support intervals must be ordered and disjoint.")
    return starts, ends


def decoding_support_to_time_intervals(true: Any, decoded: Any) -> Any:
    """Return one native TimeIntervals object for the shared Tsd support."""
    from pynwb.epoch import TimeIntervals

    true_starts, true_ends = _support_bounds(true.time_support)
    decoded_starts, decoded_ends = _support_bounds(decoded.time_support)
    if not np.array_equal(true_starts, decoded_starts) or not np.array_equal(
        true_ends,
        decoded_ends,
    ):
        raise ValueError("True and decoded positions must share exact time support.")
    output = TimeIntervals(
        name=NWB_DECODING_SUPPORT_NAME,
        description=(
            "Shared true/decoded support in ephys-reference seconds; "
            f"v1ca1 NWB artifact schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
    )
    for start, end in zip(true_starts, true_ends, strict=True):
        output.add_interval(start_time=float(start), stop_time=float(end))
    return output


def decoding_support_from_time_intervals(nwb_intervals: Any) -> Any:
    """Return a seconds-based IntervalSet from a fetched TimeIntervals object."""
    from pynwb.epoch import TimeIntervals

    if isinstance(nwb_intervals, pd.DataFrame):
        table = nwb_intervals
    elif isinstance(nwb_intervals, TimeIntervals):
        if str(nwb_intervals.name) != NWB_DECODING_SUPPORT_NAME:
            raise ValueError(
                f"Unexpected decoding support name {nwb_intervals.name!r}."
            )
        table = nwb_intervals.to_dataframe()
    else:
        raise TypeError("Decoding support must be TimeIntervals or a DataFrame.")
    if tuple(str(column) for column in table.columns) != (
        "start_time",
        "stop_time",
    ):
        raise ValueError(
            "Decoding TimeIntervals must contain only start_time and stop_time."
        )
    import pynapple as nap

    support = nap.IntervalSet(
        start=np.asarray(table["start_time"], dtype=float),
        end=np.asarray(table["stop_time"], dtype=float),
        time_units="s",
    )
    _support_bounds(support)
    return support


def position_from_time_series(nwb_series: Any, support: Any, *, kind: str) -> Any:
    """Reconstruct one true or decoded Pynapple Tsd from fetched NWB objects."""
    times, values = _time_series_arrays(nwb_series, kind=kind)
    return _make_tsd(times, values, support)


def _provenance_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return storage-independent scalar provenance for one decoding result."""
    validated = validate_path_specific_decoding_result(result)
    path_length_cm = float(validated["path_length_cm"])
    return {
        "metadata": dict(validated["metadata"]),
        "parameters": dict(validated["parameters"]),
        "path_length_cm": (
            path_length_cm if np.isfinite(path_length_cm) else None
        ),
        "n_units": int(validated["n_units"]),
        "n_folds_expected": int(validated["n_folds_expected"]),
        "n_folds_valid": int(validated["n_folds_valid"]),
        "n_decoded_samples": int(validated["n_decoded_samples"]),
        "selected_units_sha256": str(validated["selected_units_sha256"]),
        "analysis_status": str(validated["analysis_status"]),
        "artifact_origin": str(validated["artifact_origin"]),
        "legacy_artifact_provenance": validated[
            "legacy_artifact_provenance"
        ],
    }


def decoding_provenance_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store one canonical JSON provenance record for a decoding result."""
    from hdmf.common import DynamicTable

    from v1ca1.spyglass.selection import canonical_json

    table = pd.DataFrame(
        [
            {
                "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
                "metadata_json": canonical_json(_provenance_payload(result)),
            }
        ],
        columns=list(_NWB_PROVENANCE_COLUMNS),
    )
    return DynamicTable.from_dataframe(
        name=NWB_PROVENANCE_TABLE_NAME,
        df=table,
        table_description=(
            "One provenance record for PathSpecificPlaceDecoding; "
            f"v1ca1 NWB artifact schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=[
            {
                "name": "artifact_schema_version",
                "description": "v1ca1 NWB artifact schema version.",
            },
            {
                "name": "metadata_json",
                "description": "Canonical JSON result metadata and parameters.",
            },
        ],
    )


def _provenance_from_dynamic_table(nwb_table: Any) -> dict[str, Any]:
    """Return and validate the scalar payload from a provenance table."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != NWB_PROVENANCE_TABLE_NAME:
            raise ValueError(
                f"Unexpected decoding provenance name {nwb_table.name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("Decoding provenance must be a DynamicTable or DataFrame.")
    if len(table) != 1 or set(table.columns) != set(_NWB_PROVENANCE_COLUMNS):
        raise ValueError("Decoding provenance must contain one canonical row.")
    version = _decoded_nwb_text(
        table.iloc[0]["artifact_schema_version"],
        name="artifact_schema_version",
    )
    if version != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Decoding NWB artifact schema version is unsupported.")
    payload_text = _decoded_nwb_text(
        table.iloc[0]["metadata_json"],
        name="metadata_json",
    )
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Decoding provenance is not valid JSON.") from exc
    expected = {
        "metadata",
        "parameters",
        "path_length_cm",
        "n_units",
        "n_folds_expected",
        "n_folds_valid",
        "n_decoded_samples",
        "selected_units_sha256",
        "analysis_status",
        "artifact_origin",
        "legacy_artifact_provenance",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Decoding provenance has an invalid schema.")
    return dict(payload)


def path_specific_place_decoding_result_from_nwb_objects(
    *,
    selected_units: Any,
    fold_qc: Any,
    summary: Any,
    binned_error: Any,
    true_position: Any,
    decoded_position: Any,
    decoding_support: Any,
    provenance: Any,
) -> dict[str, Any]:
    """Reconstruct and validate one result from its eight fetched objects."""
    payload = _provenance_from_dynamic_table(provenance)
    support = decoding_support_from_time_intervals(decoding_support)
    result = {
        "metadata": dict(payload["metadata"]),
        "parameters": dict(payload["parameters"]),
        "selected_units": selected_units_from_dynamic_table(selected_units),
        "fold_qc": fold_qc_from_dynamic_table(fold_qc),
        "summary": decoding_summary_from_dynamic_table(summary),
        "binned_error": binned_error_from_dynamic_table(binned_error),
        "true": position_from_time_series(true_position, support, kind="true"),
        "decoded": position_from_time_series(
            decoded_position,
            support,
            kind="decoded",
        ),
        "path_length_cm": (
            np.nan
            if payload["path_length_cm"] is None
            else float(payload["path_length_cm"])
        ),
        "n_units": int(payload["n_units"]),
        "n_folds_expected": int(payload["n_folds_expected"]),
        "n_folds_valid": int(payload["n_folds_valid"]),
        "n_decoded_samples": int(payload["n_decoded_samples"]),
        "selected_units_sha256": str(payload["selected_units_sha256"]),
        "analysis_status": str(payload["analysis_status"]),
        "artifact_origin": str(payload["artifact_origin"]),
        "legacy_artifact_provenance": payload["legacy_artifact_provenance"],
    }
    return validate_path_specific_decoding_result(result)


def _table_sha256(table: pd.DataFrame, *, artifact_name: str) -> str:
    """Digest one canonical decoding table independently of storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    canonical = _canonical_nwb_table(table, artifact_name=artifact_name)
    records = []
    for record in canonical.to_dict("records"):
        normalized = {}
        for column in _NWB_TABLE_COLUMNS[artifact_name]:
            value = record[column]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and np.isnan(value):
                value = None
            normalized[column] = value
        records.append(normalized)
    return provenance_sha256(
        {
            "columns": list(_NWB_TABLE_COLUMNS[artifact_name]),
            "records": records,
        }
    )


def selected_units_sha256(table: pd.DataFrame) -> str:
    """Digest the complete selected-unit table."""
    return _table_sha256(table, artifact_name="selected_units")


def fold_qc_sha256(table: pd.DataFrame) -> str:
    """Digest the complete fold-QC table."""
    return _table_sha256(table, artifact_name="fold_qc")


def decoding_summary_sha256(table: pd.DataFrame) -> str:
    """Digest the complete decoding-summary table."""
    return _table_sha256(table, artifact_name="summary")


def binned_error_sha256(table: pd.DataFrame) -> str:
    """Digest the complete binned-error table."""
    return _table_sha256(table, artifact_name="binned_error")


def position_time_series_sha256(tsd: Any, *, kind: str) -> str:
    """Digest one true or decoded position series independent of storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    _validate_tsd(tsd, name=kind, allow_empty=True)
    return provenance_sha256(
        {
            "kind": kind,
            "timestamps_s": np.asarray(tsd.t, dtype=float).reshape(-1).tolist(),
            "position_cm": np.asarray(tsd.d, dtype=float).reshape(-1).tolist(),
        }
    )


def decoding_support_sha256(true: Any, decoded: Any) -> str:
    """Digest the exact shared true/decoded support bounds."""
    from v1ca1.spyglass.selection import provenance_sha256

    true_starts, true_ends = _support_bounds(true.time_support)
    decoded_starts, decoded_ends = _support_bounds(decoded.time_support)
    if not np.array_equal(true_starts, decoded_starts) or not np.array_equal(
        true_ends,
        decoded_ends,
    ):
        raise ValueError("True and decoded positions must share exact time support.")
    return provenance_sha256(
        {
            "start_time_s": true_starts.tolist(),
            "stop_time_s": true_ends.tolist(),
        }
    )


def decoding_provenance_sha256(result: Mapping[str, Any]) -> str:
    """Digest scalar result provenance independently of physical storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(_provenance_payload(result))


def write_path_specific_decoding_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write and reload one canonical decoding bundle."""
    result = validate_path_specific_decoding_result(result)
    destination = Path(path)
    result_id = result["metadata"]["path_specific_place_decoding_id"]
    if destination.name != result_id:
        raise ValueError("Artifact directory name must equal the result UUID.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite decoding artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        result["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME, index=False
        )
        result["fold_qc"].to_parquet(temporary / FOLD_QC_FILENAME, index=False)
        result["summary"].to_parquet(temporary / SUMMARY_FILENAME, index=False)
        result["binned_error"].to_parquet(
            temporary / BINNED_ERROR_FILENAME, index=False
        )
        result["true"].save(temporary / TRUE_FILENAME)
        result["decoded"].save(temporary / DECODED_FILENAME)
        common = {
            **result["metadata"],
            **result["parameters"],
            "n_units": result["n_units"],
            "n_folds_valid": result["n_folds_valid"],
            "n_decoded_samples": result["n_decoded_samples"],
            "selected_units_sha256": result["selected_units_sha256"],
            "analysis_status": result["analysis_status"],
            "artifact_origin": result["artifact_origin"],
        }
        specs = (
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("fold_qc", FOLD_QC_FILENAME, "parquet"),
            ("decoding_summary", SUMMARY_FILENAME, "parquet"),
            ("binned_error", BINNED_ERROR_FILENAME, "parquet"),
            ("true", TRUE_FILENAME, "pynapple_npz"),
            ("decoded", DECODED_FILENAME, "pynapple_npz"),
        )
        rows = []
        for key, filename, kind in specs:
            artifact_path = temporary / filename
            rows.append(
                {
                    "artifact_key": key,
                    "relative_path": filename,
                    "artifact_kind": kind,
                    "file_size_bytes": artifact_path.stat().st_size,
                    "sha256": _file_sha256(artifact_path),
                    **common,
                }
            )
        pd.DataFrame.from_records(rows, columns=MANIFEST_COLUMNS).to_parquet(
            temporary / MANIFEST_FILENAME,
            index=False,
        )
        load_path_specific_decoding_artifact(
            temporary,
            _allow_temporary_name=True,
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return get_path_specific_decoding_artifact_paths(
        animal_name=result["metadata"]["animal_name"],
        date=result["metadata"]["date"],
        epoch=result["metadata"]["epoch"],
        region=result["metadata"]["region"],
        path_specific_place_decoding_id=result_id,
        artifact_root=destination.parents[5],
    )


def _load_npz(path: Path) -> Any:
    """Load one Pynapple Tsd artifact."""
    import pynapple as nap

    loaded = nap.load_file(path)
    if not hasattr(loaded, "t") or not hasattr(loaded, "d"):
        raise ValueError(f"Expected a Tsd artifact at {path}.")
    return loaded


def load_path_specific_decoding_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one canonical decoding artifact."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Decoding manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    _validate_exact_columns(manifest, MANIFEST_COLUMNS, name="manifest")
    if manifest.empty or manifest["artifact_key"].duplicated().any():
        raise ValueError("Manifest artifact keys must be unique and non-empty.")
    expected_artifacts = {
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "fold_qc": (FOLD_QC_FILENAME, "parquet"),
        "decoding_summary": (SUMMARY_FILENAME, "parquet"),
        "binned_error": (BINNED_ERROR_FILENAME, "parquet"),
        "true": (TRUE_FILENAME, "pynapple_npz"),
        "decoded": (DECODED_FILENAME, "pynapple_npz"),
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected_artifacts):
        raise ValueError("Manifest does not contain the canonical artifact set.")
    for _, row in manifest.iterrows():
        expected_filename, expected_kind = expected_artifacts[
            str(row["artifact_key"])
        ]
        if (
            str(row["relative_path"]) != expected_filename
            or str(row["artifact_kind"]) != expected_kind
        ):
            raise ValueError("Manifest artifact names or kinds are inconsistent.")
        relative_path = Path(str(row["relative_path"]))
        if relative_path.name != str(row["relative_path"]):
            raise ValueError("Manifest paths must be direct child filenames.")
        artifact_path = directory / relative_path
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Manifest artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(f"Manifest checksum mismatch for {artifact_path}.")
    first = manifest.iloc[0]
    metadata = {
        name: str(first[name])
        for name in (
            "path_specific_place_decoding_id",
            "animal_name",
            "date",
            "region",
            "epoch",
        )
    }
    if not _allow_temporary_name and directory.name != metadata[
        "path_specific_place_decoding_id"
    ]:
        raise ValueError("Artifact directory name does not match its result UUID.")
    parameters = {
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        "n_folds": int(first["n_folds"]),
        "decoding_bin_size_s": float(first["decoding_bin_size_s"]),
        "sliding_window_size_bins": int(first["sliding_window_size_bins"]),
        "spatial_bin_size_cm": float(first["spatial_bin_size_cm"]),
        "random_seed": int(first["random_seed"]),
    }
    common_values = {
        **metadata,
        **parameters,
        "n_units": int(first["n_units"]),
        "n_folds_valid": int(first["n_folds_valid"]),
        "n_decoded_samples": int(first["n_decoded_samples"]),
        "selected_units_sha256": str(first["selected_units_sha256"]),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
    }
    for name, expected in common_values.items():
        if not np.all(manifest[name].astype(str) == str(expected)):
            raise ValueError(f"Manifest has inconsistent {name!r} values.")
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": pd.read_parquet(directory / SELECTED_UNITS_FILENAME),
        "fold_qc": pd.read_parquet(directory / FOLD_QC_FILENAME),
        "summary": pd.read_parquet(directory / SUMMARY_FILENAME),
        "binned_error": pd.read_parquet(directory / BINNED_ERROR_FILENAME),
        "true": _load_npz(directory / TRUE_FILENAME),
        "decoded": _load_npz(directory / DECODED_FILENAME),
        "path_length_cm": np.nan,
        "n_units": int(first["n_units"]),
        "n_folds_expected": int(first["n_folds"]),
        "n_folds_valid": int(first["n_folds_valid"]),
        "n_decoded_samples": int(first["n_decoded_samples"]),
        "selected_units_sha256": str(first["selected_units_sha256"]),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance": None,
        "manifest": manifest,
    }
    validate_path_specific_decoding_result(result)
    return result


def register_existing_path_specific_decoding_artifact(
    *,
    source_true_path: Path,
    source_decoded_path: Path,
    destination_path: Path | None,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    path_specific_place_decoding_id: Any,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    trajectory_intervals: Mapping[str, Any],
    movement_interval: Any,
    path_length_cm: float,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    decoding_bin_size_s: float = DEFAULT_DECODING_BIN_SIZE_S,
    sliding_window_size_bins: int = DEFAULT_SLIDING_WINDOW_SIZE_BINS,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    random_seed: int = DEFAULT_RANDOM_SEED,
    source_v1ca1_git_commit: str | None = None,
) -> dict[str, Any]:
    """Validate legacy Tsds and optionally write the offline artifact bundle."""
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256
    from v1ca1.task_progression.decoding_comparison import (
        build_train_test_folds,
        summarize_decoding_error_by_position,
        summarize_decoding_metrics,
    )

    source_true_path = Path(source_true_path)
    source_decoded_path = Path(source_decoded_path)
    for name, source_path in (
        ("source_true_path", source_true_path),
        ("source_decoded_path", source_decoded_path),
    ):
        if not source_path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {source_path}")
    parameters = validate_path_specific_decoding_parameters(
        n_folds=n_folds,
        decoding_bin_size_s=decoding_bin_size_s,
        sliding_window_size_bins=sliding_window_size_bins,
        spatial_bin_size_cm=spatial_bin_size_cm,
        random_seed=random_seed,
    )
    parameter_name = str(parameter_name).strip()
    if not parameter_name or len(parameter_name) > 64:
        raise ValueError("parameter_name must be non-empty and at most 64 characters.")
    parameter_payload = {
        "path_specific_place_decoding_param_name": parameter_name,
        **parameters,
    }
    expected_parameter_sha256 = provenance_sha256(parameter_payload)
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    if str(parameter_sha256) != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match the effective parameters.")
    expected_output_sha256 = provenance_sha256(dict(OUTPUT_RULE))
    if output_rule_sha256 is None:
        output_rule_sha256 = expected_output_sha256
    if str(output_rule_sha256) != expected_output_sha256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")

    path_length_cm = float(path_length_cm)
    if not np.isfinite(path_length_cm) or path_length_cm <= 0.0:
        raise ValueError("path_length_cm must be positive and finite.")
    true = _load_npz(source_true_path)
    decoded = _load_npz(source_decoded_path)
    _validate_tsd(true, name="legacy true")
    _validate_tsd(decoded, name="legacy decoded")
    true_values = np.asarray(true.d, dtype=float)
    tolerance = max(1e-9, path_length_cm * 1e-9)
    maximum = path_length_cm * len(TRAJECTORY_TYPES)
    if np.any(true_values < -tolerance) or np.any(true_values > maximum + tolerance):
        raise ValueError(
            "Legacy true-place values lie outside the concatenated path coordinate."
        )

    metadata = {
        "path_specific_place_decoding_id": _uuid_string(
            path_specific_place_decoding_id,
            name="path_specific_place_decoding_id",
        ),
        **{
            name: _path_component(value, name=name)
            for name, value in {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "epoch": epoch,
            }.items()
        },
    }
    selected_units = _selected_unit_table(spikes, stable_unit_ids)
    selected_units_sha256 = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    train_folds, test_folds = build_train_test_folds(
        dict(trajectory_intervals),
        n_folds=parameters["n_folds"],
        random_state=parameters["random_seed"],
    )
    fold_rows = []
    for fold in range(parameters["n_folds"]):
        train = train_folds[fold].intersect(movement_interval)
        test = test_folds[fold].intersect(movement_interval)
        n_train_laps, train_duration = _interval_summary(train)
        n_test_laps, test_duration = _interval_summary(test)
        n_samples = len(np.asarray(decoded.restrict(test).t))
        status = "valid" if n_samples else "no_test_count_bins"
        fold_rows.append(
            {
                **metadata,
                "fold": fold,
                "n_train_laps": n_train_laps,
                "n_test_laps": n_test_laps,
                "train_duration_s": train_duration,
                "test_duration_s": test_duration,
                "n_decoded_samples": n_samples,
                "qc_status": status,
                "qc_message": (
                    ""
                    if n_samples
                    else "Legacy output has no decoded samples in this test fold."
                ),
            }
        )
    fold_qc = pd.DataFrame.from_records(fold_rows).loc[:, list(FOLD_QC_COLUMNS)]
    n_folds_valid = int((fold_qc["qc_status"] == "valid").sum())
    n_units = len(selected_units)
    analysis_status = _analysis_status(
        n_units,
        parameters["n_folds"],
        n_folds_valid,
    )
    metrics = summarize_decoding_metrics(true, decoded)
    if int(metrics["n_samples"]) <= 0:
        raise ValueError(
            "Legacy true and decoded artifacts contain no aligned samples."
        )
    summary = pd.DataFrame.from_records(
        [
            {
                **metadata,
                "model": "path_specific_place",
                "coordinate_unit": "cm",
                "n_units": n_units,
                "n_folds_expected": parameters["n_folds"],
                "n_folds_valid": n_folds_valid,
                **metrics,
                "analysis_status": analysis_status,
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
    bins = np.arange(
        0.0,
        maximum + parameters["spatial_bin_size_cm"],
        parameters["spatial_bin_size_cm"],
        dtype=float,
    )
    binned = summarize_decoding_error_by_position(
        true,
        decoded,
        bin_edges=bins,
        error_mode=DEFAULT_ERROR_MODE,
        summary=DEFAULT_ERROR_SUMMARY,
        min_count=DEFAULT_MIN_BIN_COUNT,
    )
    binned_error = binned.assign(
        **metadata,
        coordinate_unit="cm",
    ).loc[:, list(BINNED_ERROR_COLUMNS)]
    provenance = {
        "source_true_path": str(source_true_path.resolve()),
        "source_true_sha256": _file_sha256(source_true_path),
        "source_decoded_path": str(source_decoded_path.resolve()),
        "source_decoded_sha256": _file_sha256(source_decoded_path),
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "assumed_parameters": {**parameter_payload},
        "source_parameter_validation": (
            "caller_attested; legacy filenames do not encode decoder parameters"
        ),
        "fold_validation": "reconstructed from selected lap-wise KFold splits",
    }
    result = {
        "metadata": metadata,
        "parameters": {
            "parameter_name": parameter_name,
            "parameter_sha256": str(parameter_sha256),
            "output_rule_sha256": str(output_rule_sha256),
            **parameters,
        },
        "selected_units": selected_units,
        "fold_qc": fold_qc,
        "summary": summary,
        "binned_error": binned_error,
        "true": true,
        "decoded": decoded,
        "path_length_cm": path_length_cm,
        "n_units": n_units,
        "n_folds_expected": parameters["n_folds"],
        "n_folds_valid": n_folds_valid,
        "n_decoded_samples": int(metrics["n_samples"]),
        "selected_units_sha256": selected_units_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
    }
    if destination_path is None:
        return validate_path_specific_decoding_result(result)
    write_path_specific_decoding_artifact(result, destination_path)
    return {
        **result,
        "artifact_paths": get_path_specific_decoding_artifact_paths(
            animal_name=metadata["animal_name"],
            date=metadata["date"],
            epoch=metadata["epoch"],
            region=metadata["region"],
            path_specific_place_decoding_id=metadata[
                "path_specific_place_decoding_id"
            ],
            artifact_root=Path(destination_path).parents[5],
        ),
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "DEFAULT_ARTIFACT_ROOT",
    "MANUSCRIPT_PARAMETERS",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "NWB_BINNED_ERROR_TABLE_NAME",
    "NWB_DECODING_SUPPORT_NAME",
    "NWB_DECODED_POSITION_TIMESERIES_NAME",
    "NWB_FOLD_QC_TABLE_NAME",
    "NWB_PROVENANCE_TABLE_NAME",
    "NWB_SELECTED_UNITS_TABLE_NAME",
    "NWB_SUMMARY_TABLE_NAME",
    "NWB_TRUE_POSITION_TIMESERIES_NAME",
    "OUTPUT_RULE",
    "binned_error_from_dynamic_table",
    "binned_error_sha256",
    "binned_error_to_dynamic_table",
    "build_concatenated_path_specific_position",
    "compute_path_specific_place_decoding",
    "decoded_position_to_time_series",
    "decoding_provenance_sha256",
    "decoding_provenance_to_dynamic_table",
    "decoding_summary_from_dynamic_table",
    "decoding_summary_sha256",
    "decoding_summary_to_dynamic_table",
    "decoding_support_from_time_intervals",
    "decoding_support_sha256",
    "decoding_support_to_time_intervals",
    "fold_qc_from_dynamic_table",
    "fold_qc_sha256",
    "fold_qc_to_dynamic_table",
    "get_path_specific_decoding_artifact_paths",
    "load_path_specific_decoding_artifact",
    "path_specific_place_decoding_result_from_nwb_objects",
    "position_from_time_series",
    "position_time_series_sha256",
    "register_existing_path_specific_decoding_artifact",
    "selected_units_from_dynamic_table",
    "selected_units_sha256",
    "selected_units_to_dynamic_table",
    "true_position_to_time_series",
    "validate_path_specific_decoding_parameters",
    "validate_path_specific_decoding_result",
    "write_path_specific_decoding_artifact",
]
