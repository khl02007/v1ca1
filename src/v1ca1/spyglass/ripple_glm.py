"""Database-free CA1-to-V1 ripple population GLM artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "ripple_glm"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
SUMMARY_FILENAME = "summary.parquet"
RESULT_FILENAME = "ripple_glm.nc"
BUNDLE_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"

SOURCE_REGION = "ca1"
TARGET_REGION = "v1"
SOURCE_ROLE = "source"
TARGET_ROLE = "target"
METRIC_NAMES = ("pseudo_r2", "mae", "devexp", "bits_per_spike")
RIPPLE_SELECTION_MODES = ("allripples", "deduped", "single")
SOURCE_PREDICTOR_MODES = ("unit_vector", "mean_activity")

DEFAULT_RIPPLE_WINDOW_S = 0.2
DEFAULT_RIPPLE_WINDOW_OFFSET_S = 0.0
DEFAULT_MIN_SPIKES_PER_RIPPLE = 0.1
DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE = 0.0
DEFAULT_N_SPLITS = 5
# The manuscript pipeline used 100 shuffle refits; the legacy CLI's broad
# exploratory default remains 1000 in ``v1ca1.ripple.ripple_glm``.
DEFAULT_N_SHUFFLES_RIPPLE = 100
DEFAULT_RIDGE_STRENGTH = 1e-1
DEFAULT_SHUFFLE_SEED = 45
DEFAULT_MAXITER = 6000
DEFAULT_TOL = 1e-7
DEFAULT_RIPPLE_SELECTION_MODE = "allripples"
DEFAULT_SOURCE_PREDICTOR_MODE = "unit_vector"
DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD = 2.0
DEFAULT_REQUIRE_SPEED_GATED = True

OUTPUT_RULE = {
    "version": 1,
    "model_direction": "ca1_to_v1",
    "model_family": "ridge_regularized_poisson_population_glm",
    "cv_policy": "contiguous_unshuffled_kfold_over_selected_ripples",
    "shuffle_policy": (
        "independently_permute_each_target_unit_training_response_and_refit"
    ),
    "ripple_selection_modes": RIPPLE_SELECTION_MODES,
    "source_predictor_modes": SOURCE_PREDICTOR_MODES,
    "ripple_input_policy": (
        "detector_zscore_threshold_2_and_speed_gated_events_required"
    ),
    "source_preprocessing": (
        "drop_near_constant_train_features_then_zscore_divide_by_sqrt_n_and_clip_10"
    ),
    "source_coefficient_space": "full_data_preprocessed_predictor",
    "unit_filter_policy": (
        "inclusive_mean_spike_count_per_selected_ripple_threshold"
    ),
    "unit_audit_policy": "retain_all_source_and_target_input_units",
    "unit_failure_policy": (
        "shared_population_fit_preserved_and_nonfinite_target_metrics_isolated_in_audit"
    ),
    "terminal_artifact_policy": (
        "explicit_for_no_units_no_or_insufficient_ripples_and_no_eligible_units"
    ),
    "legacy_registration_policy": (
        "imported_sorting_identity_resolved_then_verify_nwb_event_windows_target_counts_"
        "fold_layout_metric_self_consistency_and_coefficient_axes_shape_finiteness"
    ),
    "time_unit": "s",
    "time_reference": "augmented_nwb_ephys_timestamps",
}

IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (
    "role",
    "region",
    *IDENTITY_COLUMNS,
    "input_unit_index",
    "mean_spikes_per_ripple",
    "minimum_mean_spikes_per_ripple",
    "passes_spike_threshold",
    "included_in_fit",
    "included_in_full_coefficient",
    "valid_glm_metrics",
    "unit_qc_status",
)
UNIT_QC_STATUSES = (
    "excluded_spike_threshold",
    "included_source_predictor",
    "included_source_no_full_coefficient",
    "valid",
    "partial_nonfinite_metrics",
    "no_finite_metrics",
    "not_computed",
)
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_source_units",
    "no_target_units",
    "no_ripples",
    "insufficient_ripples",
    "no_eligible_source_units",
    "no_eligible_target_units",
    "no_valid_target_units",
)
FITTED_ANALYSIS_STATUSES = ("valid", "partial_valid", "no_valid_target_units")
SUMMARY_COLUMNS = (
    *IDENTITY_COLUMNS,
    "ripple_glm_id",
    "animal_name",
    "date",
    "epoch",
    "source_region",
    "target_region",
    "n_ripples",
    "valid_glm_metrics",
    "unit_qc_status",
    *tuple(
        f"ripple_{metric}_{suffix}"
        for metric in METRIC_NAMES
        for suffix in ("mean", "sem", "shuffle_mean", "shuffle_sd", "p_value")
    ),
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "ripple_glm_id",
    "animal_name",
    "date",
    "epoch",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "upstream_provenance_json",
    "selected_ripple_events_sha256",
    "n_source_units",
    "n_target_units",
    "n_source_units_in_fit",
    "n_target_units_in_fit",
    "n_valid_target_units",
    "n_ripples",
    "selected_units_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "bundle_schema_version",
)


def _ripple_module() -> Any:
    """Import the existing scientific implementation only when required."""
    from v1ca1.ripple import ripple_glm

    return ripple_glm


def _provenance_sha256(value: Any) -> str:
    """Return the shared deterministic provenance digest."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(value)


OUTPUT_RULE_SHA256 = _provenance_sha256(OUTPUT_RULE)


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe, non-empty path component."""
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


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_sha256(table: pd.DataFrame) -> str:
    """Return a deterministic digest for one canonical table."""
    hashed = pd.util.hash_pandas_object(table, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _database_bool(value: Any, *, name: str) -> bool:
    """Normalize one bool or database integer 0/1 without accepting truthy junk."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(int(value))
    raise TypeError(f"{name} must be a bool or database integer 0/1.")


def get_ripple_glm_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_glm_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first ripple GLM bundle."""
    animal_name = _path_component(animal_name, name="animal_name")
    date = _path_component(date, name="date")
    epoch = _path_component(epoch, name="epoch")
    result_id = _uuid_string(ripple_glm_id, name="ripple_glm_id")
    artifact_dir = (
        Path(artifact_root)
        / animal_name
        / date
        / ARTIFACT_DIRNAME
        / epoch
        / result_id
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "selected_units_path": artifact_dir / SELECTED_UNITS_FILENAME,
        "summary_path": artifact_dir / SUMMARY_FILENAME,
        "result_path": artifact_dir / RESULT_FILENAME,
    }


def validate_ripple_glm_parameters(
    *,
    ripple_window_s: float = DEFAULT_RIPPLE_WINDOW_S,
    ripple_window_offset_s: float = DEFAULT_RIPPLE_WINDOW_OFFSET_S,
    source_window_s: float | None = None,
    source_window_offset_s: float | None = None,
    target_window_s: float | None = None,
    target_window_offset_s: float | None = None,
    ripple_selection_mode: str = DEFAULT_RIPPLE_SELECTION_MODE,
    source_predictor_mode: str = DEFAULT_SOURCE_PREDICTOR_MODE,
    min_spikes_per_ripple: float = DEFAULT_MIN_SPIKES_PER_RIPPLE,
    min_ca1_spikes_per_ripple: float = DEFAULT_MIN_CA1_SPIKES_PER_RIPPLE,
    n_splits: int = DEFAULT_N_SPLITS,
    n_shuffles_ripple: int = DEFAULT_N_SHUFFLES_RIPPLE,
    ridge_strength: float = DEFAULT_RIDGE_STRENGTH,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    maxiter: int = DEFAULT_MAXITER,
    tol: float = DEFAULT_TOL,
    expected_detector_zscore_threshold: float = (
        DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD
    ),
    require_speed_gated: bool = DEFAULT_REQUIRE_SPEED_GATED,
) -> dict[str, Any]:
    """Return validated parameters with effective source and target windows."""
    ripple = _ripple_module()
    windows = ripple._resolve_model_window_parameters(
        ripple_window_s=float(ripple_window_s),
        ripple_window_offset_s=float(ripple_window_offset_s),
        source_window_s=source_window_s,
        source_window_offset_s=source_window_offset_s,
        target_window_s=target_window_s,
        target_window_offset_s=target_window_offset_s,
    )
    selection_mode = str(ripple_selection_mode)
    if selection_mode not in RIPPLE_SELECTION_MODES:
        raise ValueError(
            f"ripple_selection_mode must be one of {RIPPLE_SELECTION_MODES!r}."
        )
    predictor_mode = str(source_predictor_mode)
    if predictor_mode not in SOURCE_PREDICTOR_MODES:
        raise ValueError(
            f"source_predictor_mode must be one of {SOURCE_PREDICTOR_MODES!r}."
        )
    floats = {
        "min_spikes_per_ripple": float(min_spikes_per_ripple),
        "min_ca1_spikes_per_ripple": float(min_ca1_spikes_per_ripple),
        "ridge_strength": float(ridge_strength),
        "tol": float(tol),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in floats.values()):
        raise ValueError("Spike thresholds, ridge_strength, and tol must be finite and non-negative.")
    if floats["tol"] <= 0.0:
        raise ValueError("tol must be positive.")
    detector_threshold = float(expected_detector_zscore_threshold)
    if not np.isclose(
        detector_threshold,
        DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "RippleGLM requires detector events selected at z-score threshold 2.0."
        )
    if not _database_bool(require_speed_gated, name="require_speed_gated"):
        raise ValueError("RippleGLM requires speed-gated ripple events.")
    integers: dict[str, int] = {}
    for name, raw, minimum in (
        ("n_splits", n_splits, 2),
        ("n_shuffles_ripple", n_shuffles_ripple, 0),
        ("maxiter", maxiter, 1),
    ):
        if isinstance(raw, bool) or int(raw) != raw or int(raw) < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
        integers[name] = int(raw)
    if isinstance(shuffle_seed, bool) or int(shuffle_seed) != shuffle_seed:
        raise ValueError("shuffle_seed must be an integer.")
    return {
        "ripple_window_s": float(windows["target_window_s"]),
        "ripple_window_offset_s": float(windows["target_window_offset_s"]),
        "source_window_s": float(windows["source_window_s"]),
        "source_window_offset_s": float(windows["source_window_offset_s"]),
        "target_window_s": float(windows["target_window_s"]),
        "target_window_offset_s": float(windows["target_window_offset_s"]),
        "source_target_windows_differ": bool(windows["windows_differ"]),
        "ripple_selection_mode": selection_mode,
        "source_predictor_mode": predictor_mode,
        **floats,
        **integers,
        "shuffle_seed": int(shuffle_seed),
        "expected_detector_zscore_threshold": detector_threshold,
        "require_speed_gated": True,
    }


def _effective_parameters(
    *,
    parameter_name: str,
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
    **parameter_values: Any,
) -> dict[str, Any]:
    """Validate parameters and immutable parameter/rule hashes."""
    values = validate_ripple_glm_parameters(**parameter_values)
    name = _path_component(parameter_name, name="parameter_name")
    expected = _provenance_sha256({"ripple_glm_param_name": name, **values})
    if parameter_sha256 is None:
        parameter_sha256 = expected
    if str(parameter_sha256) != expected:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    if output_rule_sha256 is None:
        output_rule_sha256 = OUTPUT_RULE_SHA256
    if str(output_rule_sha256) != OUTPUT_RULE_SHA256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": name,
        "parameter_sha256": str(parameter_sha256),
        "output_rule_sha256": str(output_rule_sha256),
        **values,
    }


def _metadata(
    *, ripple_glm_id: Any, animal_name: str, date: str, epoch: str
) -> dict[str, str]:
    """Return validated result metadata."""
    return {
        "ripple_glm_id": _uuid_string(ripple_glm_id, name="ripple_glm_id"),
        "animal_name": _path_component(animal_name, name="animal_name"),
        "date": _path_component(date, name="date"),
        "epoch": _path_component(epoch, name="epoch"),
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
    }


def _canonical_json_mapping(value: Mapping[str, Any], *, name: str) -> tuple[dict[str, Any], str]:
    """Return a JSON-roundtripped non-empty provenance mapping and its JSON."""
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping.")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be JSON serializable.") from exc
    return normalized, encoded


def _validate_upstream_provenance(
    upstream_provenance: Mapping[str, Any],
    *,
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Require provenance for the fixed threshold-2 speed-gated ripple input."""
    normalized, encoded = _canonical_json_mapping(
        upstream_provenance, name="upstream_provenance"
    )
    try:
        detector_threshold = normalized["detector_zscore_threshold"]
        speed_gated = normalized["speed_gated"]
    except KeyError as exc:
        raise ValueError(
            "upstream_provenance must contain detector_zscore_threshold and "
            "speed_gated from the selected RippleIntervals row."
        ) from exc
    if isinstance(detector_threshold, bool) or not isinstance(
        detector_threshold, Real
    ):
        raise TypeError(
            "upstream_provenance detector_zscore_threshold must be numeric."
        )
    detector_threshold = float(detector_threshold)
    if not np.isfinite(detector_threshold) or not np.isclose(
        detector_threshold,
        float(parameters["expected_detector_zscore_threshold"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Selected RippleIntervals detector threshold must equal 2.0.")
    if not _database_bool(
        speed_gated, name="upstream_provenance speed_gated"
    ):
        raise ValueError("Selected RippleIntervals provenance must have speed_gated=True.")
    return normalized, encoded


def _identity_table(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    *,
    role: str,
    region: str,
) -> pd.DataFrame:
    """Return persistent identities aligned to one runtime TsGroup."""
    from v1ca1.spyglass.movement import _stable_identity_rows

    rows = _stable_identity_rows(spikes, stable_unit_ids)
    table = pd.DataFrame.from_records(rows, columns=IDENTITY_COLUMNS)
    if table.empty:
        table = pd.DataFrame({column: pd.Series(dtype=object) for column in IDENTITY_COLUMNS})
    table.insert(0, "region", region)
    table.insert(0, "role", role)
    table["input_unit_index"] = np.arange(len(table), dtype=int)
    return table


def _normalize_ripples(ripple_table: Any, *, epoch: str) -> pd.DataFrame:
    """Return finite, ordered ripple intervals for exactly one epoch."""
    as_dataframe = getattr(ripple_table, "as_dataframe", None)
    if callable(as_dataframe):
        table = as_dataframe().copy()
    elif isinstance(ripple_table, pd.DataFrame):
        table = ripple_table.copy()
    else:
        table = pd.DataFrame(ripple_table)
    rename = {}
    if "start" in table.columns and "start_time" not in table.columns:
        rename["start"] = "start_time"
    if "stop_time" in table.columns and "end_time" not in table.columns:
        rename["stop_time"] = "end_time"
    if "end" in table.columns and "end_time" not in table.columns:
        rename["end"] = "end_time"
    table = table.rename(columns=rename)
    if "epoch" in table.columns:
        table = table.loc[table["epoch"].astype(str) == str(epoch)]
    missing = [name for name in ("start_time", "end_time") if name not in table]
    if missing:
        raise ValueError(f"ripple_table is missing required columns {missing!r}.")
    table = table.copy().reset_index(drop=True)
    table["start_time"] = pd.to_numeric(table["start_time"], errors="raise")
    table["end_time"] = pd.to_numeric(table["end_time"], errors="raise")
    bounds = table[["start_time", "end_time"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Ripple bounds must be finite seconds values.")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Every ripple interval must have start_time < end_time.")
    if table["start_time"].duplicated().any():
        raise ValueError("Ripple start times must be unique.")
    return table.sort_values("start_time", kind="stable").reset_index(drop=True)


def _epoch_bounds(epoch_interval: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, non-overlapping epoch interval bounds."""
    if not hasattr(epoch_interval, "start") or not hasattr(epoch_interval, "end"):
        raise TypeError("epoch_interval must expose start and end seconds.")
    starts = np.asarray(epoch_interval.start, dtype=float).reshape(-1)
    ends = np.asarray(epoch_interval.end, dtype=float).reshape(-1)
    if starts.shape != ends.shape or starts.size == 0:
        raise ValueError("epoch_interval must contain aligned non-empty bounds.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("epoch_interval bounds must be finite.")
    if np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("epoch_interval bounds must be positive and non-overlapping.")
    return starts, ends


def _select_and_count_inputs(
    *,
    epoch: str,
    ripple_table: Any,
    epoch_interval: Any,
    source_spikes: Any,
    target_spikes: Any,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact selected windows and input spike-count matrices."""
    ripple = _ripple_module()
    table = _normalize_ripples(ripple_table, epoch=epoch)
    before_selection = len(table)
    selection_window_s, selection_offset_s = ripple._resolve_selection_window(
        source_window_s=parameters["source_window_s"],
        source_window_offset_s=parameters["source_window_offset_s"],
        target_window_s=parameters["target_window_s"],
        target_window_offset_s=parameters["target_window_offset_s"],
    )
    if parameters["ripple_selection_mode"] == "deduped":
        table, _ = ripple.remove_duplicate_ripples(
            table,
            ripple_window_s=selection_window_s,
            ripple_window_offset_s=selection_offset_s,
        )
    elif parameters["ripple_selection_mode"] == "single":
        table, _ = ripple.keep_single_ripple_windows(
            table,
            ripple_window_s=selection_window_s,
            ripple_window_offset_s=selection_offset_s,
        )
    after_selection = len(table)
    source_starts, source_ends = ripple._build_ripple_sample_windows(
        table,
        ripple_window_s=parameters["source_window_s"],
        ripple_window_offset_s=parameters["source_window_offset_s"],
    )
    target_starts, target_ends = ripple._build_ripple_sample_windows(
        table,
        ripple_window_s=parameters["target_window_s"],
        ripple_window_offset_s=parameters["target_window_offset_s"],
    )
    _epoch_bounds(epoch_interval)
    keep = ripple._windows_within_epoch_mask(
        epoch_interval=epoch_interval,
        source_window_starts=source_starts,
        source_window_ends=source_ends,
        target_window_starts=target_starts,
        target_window_ends=target_ends,
    )
    table = table.loc[keep].reset_index(drop=True)
    source_starts, source_ends = source_starts[keep], source_ends[keep]
    target_starts, target_ends = target_starts[keep], target_ends[keep]
    source_counts, source_group_ids = ripple._count_spikes_in_windows(
        source_spikes,
        window_starts=source_starts,
        window_ends=source_ends,
    )
    target_counts, target_group_ids = ripple._count_spikes_in_windows(
        target_spikes,
        window_starts=target_starts,
        window_ends=target_ends,
    )
    n_ripples = len(table)
    source_means = source_counts.sum(axis=0) / max(n_ripples, 1)
    target_means = target_counts.sum(axis=0) / max(n_ripples, 1)
    source_keep = source_means >= parameters["min_ca1_spikes_per_ripple"]
    target_keep = target_means >= parameters["min_spikes_per_ripple"]
    return {
        "selected_ripple_table": table,
        "ripple_start_time_s": table["start_time"].to_numpy(dtype=float),
        "source_window_start_s": np.asarray(source_starts, dtype=float),
        "source_window_end_s": np.asarray(source_ends, dtype=float),
        "target_window_start_s": np.asarray(target_starts, dtype=float),
        "target_window_end_s": np.asarray(target_ends, dtype=float),
        "source_counts": np.asarray(source_counts, dtype=float),
        "target_counts": np.asarray(target_counts, dtype=float),
        "source_group_ids": np.asarray(source_group_ids),
        "target_group_ids": np.asarray(target_group_ids),
        "source_means": np.asarray(source_means, dtype=float),
        "target_means": np.asarray(target_means, dtype=float),
        "source_keep": np.asarray(source_keep, dtype=bool),
        "target_keep": np.asarray(target_keep, dtype=bool),
        "n_ripples_before_selection": int(before_selection),
        "n_ripples_removed_by_selection": int(before_selection - after_selection),
        "n_ripples_after_selection": int(after_selection),
        "n_ripples_before_window_bounds": int(after_selection),
        "n_ripples_removed_by_window_bounds": int(after_selection - n_ripples),
        "n_ripples_after_window_bounds": int(n_ripples),
    }


def prepare_ripple_glm_event_selection(
    *,
    epoch: str,
    ripple_table: Any,
    epoch_interval: Any,
    **parameter_values: Any,
) -> dict[str, Any]:
    """Select exact modeled ripple windows without loading any spike data.

    This helper lets a selection table freeze the event hash before an
    expensive GLM population. It applies the same all/deduped/single rule,
    source/target union selection window, and epoch-bound clipping as
    :func:`compute_ripple_glm`.
    """
    parameters = validate_ripple_glm_parameters(**parameter_values)
    prepared = _select_and_count_inputs(
        epoch=_path_component(epoch, name="epoch"),
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        source_spikes={},
        target_spikes={},
        parameters=parameters,
    )
    return {
        key: prepared[key]
        for key in (
            "selected_ripple_table",
            "ripple_start_time_s",
            "source_window_start_s",
            "source_window_end_s",
            "target_window_start_s",
            "target_window_end_s",
            "n_ripples_before_selection",
            "n_ripples_removed_by_selection",
            "n_ripples_after_selection",
            "n_ripples_before_window_bounds",
            "n_ripples_removed_by_window_bounds",
            "n_ripples_after_window_bounds",
        )
    } | {"selected_ripple_events_sha256": _selected_ripple_events_sha256(prepared)}


def _group_id_lookup(identity: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return a unique string-normalized runtime-unit identity lookup."""
    rows = {}
    for row in identity.to_dict("records"):
        key = str(row["group_unit_id"])
        if key in rows:
            raise ValueError("String-normalized group unit identifiers must be unique.")
        rows[key] = row
    return rows


def _initial_selected_units(
    source_identity: pd.DataFrame,
    target_identity: pd.DataFrame,
    inputs: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Build the all-input unit audit before fit-quality annotation."""
    source = source_identity.copy()
    target = target_identity.copy()
    if [str(value) for value in inputs["source_group_ids"]] != source[
        "group_unit_id"
    ].map(str).tolist():
        raise ValueError("Source spike keys changed while constructing the unit audit.")
    if [str(value) for value in inputs["target_group_ids"]] != target[
        "group_unit_id"
    ].map(str).tolist():
        raise ValueError("Target spike keys changed while constructing the unit audit.")
    source["mean_spikes_per_ripple"] = inputs["source_means"]
    source["minimum_mean_spikes_per_ripple"] = parameters[
        "min_ca1_spikes_per_ripple"
    ]
    source["passes_spike_threshold"] = inputs["source_keep"]
    target["mean_spikes_per_ripple"] = inputs["target_means"]
    target["minimum_mean_spikes_per_ripple"] = parameters["min_spikes_per_ripple"]
    target["passes_spike_threshold"] = inputs["target_keep"]
    for table in (source, target):
        table["included_in_fit"] = table["passes_spike_threshold"].astype(bool)
        table["included_in_full_coefficient"] = False
        table["valid_glm_metrics"] = False
        table["unit_qc_status"] = np.where(
            table["passes_spike_threshold"],
            "not_computed",
            "excluded_spike_threshold",
        )
    output = pd.concat([source, target], ignore_index=True)
    for field in IDENTITY_COLUMNS:
        output[field] = output[field].map(str)
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _terminal_status(
    *, source_identity: pd.DataFrame, target_identity: pd.DataFrame, inputs: Mapping[str, Any], n_splits: int
) -> str | None:
    """Return the fixed terminal status for weak or empty inputs."""
    if source_identity.empty:
        return "no_source_units"
    if target_identity.empty:
        return "no_target_units"
    n_ripples = int(inputs["n_ripples_after_window_bounds"])
    if n_ripples == 0:
        return "no_ripples"
    if n_ripples < int(n_splits):
        return "insufficient_ripples"
    if not np.any(inputs["source_keep"]):
        return "no_eligible_source_units"
    if not np.any(inputs["target_keep"]):
        return "no_eligible_target_units"
    return None


def _selected_ripple_events_sha256(inputs: Mapping[str, Any]) -> str:
    """Return a digest of selected anchors and effective source/target windows."""
    return _provenance_sha256(
        {
            name: np.asarray(inputs[name], dtype=float).tolist()
            for name in (
                "ripple_start_time_s",
                "source_window_start_s",
                "source_window_end_s",
                "target_window_start_s",
                "target_window_end_s",
            )
        }
    )


def _require_expected_event_hash(
    inputs: Mapping[str, Any], expected_sha256: str | None
) -> None:
    """Require a selection-row event digest when one was frozen upstream."""
    if expected_sha256 is None:
        return
    observed = _selected_ripple_events_sha256(inputs)
    if str(expected_sha256) != observed:
        raise ValueError(
            "Selected ripple events differ from the event selection frozen upstream."
        )


def _dataset_json_attrs(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    legacy_artifact_provenance: Mapping[str, Any] | None,
    analysis_status: str,
    selected_ripple_events_sha256: str,
    artifact_origin: str,
) -> dict[str, Any]:
    """Return canonical scalar attributes shared by all result datasets."""
    effective = {
        key: value
        for key, value in parameters.items()
        if key not in {"parameter_name", "parameter_sha256", "output_rule_sha256"}
    }
    return {
        "ripple_glm_result_schema_version": RESULT_SCHEMA_VERSION,
        **metadata,
        "model_direction": "ca1_to_v1",
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "effective_parameters_json": json.dumps(
            effective, sort_keys=True, separators=(",", ":")
        ),
        "upstream_provenance_json": upstream_provenance_json,
        "selected_ripple_events_sha256": selected_ripple_events_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance_json": json.dumps(
            dict(legacy_artifact_provenance or {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }


def _terminal_dataset(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    source_identity: pd.DataFrame,
    target_identity: pd.DataFrame,
    upstream_provenance_json: str,
    analysis_status: str,
) -> Any:
    """Return a typed terminal NetCDF dataset without pretending a fit ran."""
    import xarray as xr

    n_samples = int(inputs["n_ripples_after_window_bounds"])
    attrs = _dataset_json_attrs(
        metadata=metadata,
        parameters=parameters,
        upstream_provenance_json=upstream_provenance_json,
        legacy_artifact_provenance=None,
        analysis_status=analysis_status,
        selected_ripple_events_sha256=_selected_ripple_events_sha256(inputs),
        artifact_origin="computed",
    )
    attrs.update(
        {
            key: int(inputs[key])
            for key in (
                "n_ripples_before_selection",
                "n_ripples_removed_by_selection",
                "n_ripples_after_selection",
                "n_ripples_before_window_bounds",
                "n_ripples_removed_by_window_bounds",
                "n_ripples_after_window_bounds",
            )
        }
    )
    return xr.Dataset(
        data_vars={
            "ripple_start_time_s": ("sample", inputs["ripple_start_time_s"]),
            "source_window_start_s": ("sample", inputs["source_window_start_s"]),
            "source_window_end_s": ("sample", inputs["source_window_end_s"]),
            "target_window_start_s": ("sample", inputs["target_window_start_s"]),
            "target_window_end_s": ("sample", inputs["target_window_end_s"]),
        },
        coords={
            "sample": np.arange(n_samples, dtype=int),
            "unit": np.asarray([], dtype=str),
            "source_unit": np.asarray([], dtype=str),
            "coef_source_unit": np.asarray([], dtype=str),
            "fold": np.asarray([], dtype=int),
            "shuffle": np.asarray([], dtype=int),
        },
        attrs=attrs,
    )


def _canonicalize_dataset(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    source_identity: pd.DataFrame,
    target_identity: pd.DataFrame,
    upstream_provenance_json: str,
    artifact_origin: str,
    legacy_artifact_provenance: Mapping[str, Any] | None,
) -> tuple[Any, pd.DataFrame]:
    """Replace runtime unit identifiers and derive target-unit fit QC."""
    source_lookup = _group_id_lookup(source_identity)
    target_lookup = _group_id_lookup(target_identity)
    raw_targets = [str(value) for value in np.asarray(dataset.coords["unit"].values)]
    raw_sources = [str(value) for value in np.asarray(dataset["ca1_unit_id"].values)]
    if any(value not in target_lookup for value in raw_targets):
        raise ValueError("Ripple GLM target units do not match supplied identities.")
    if any(value not in source_lookup for value in raw_sources):
        raise ValueError("Ripple GLM source units do not match supplied identities.")
    if len(raw_targets) != len(set(raw_targets)) or len(raw_sources) != len(set(raw_sources)):
        raise ValueError("Ripple GLM unit identifiers must be unique.")
    target_rows = [target_lookup[value] for value in raw_targets]
    source_rows = [source_lookup[value] for value in raw_sources]
    target_stable = np.asarray([row["stable_unit_id"] for row in target_rows], dtype=str)
    source_stable = np.asarray([row["stable_unit_id"] for row in source_rows], dtype=str)
    canonical = dataset.assign_coords(
        {
            "unit": target_stable,
            "spikesorting_merge_id": (
                "unit",
                np.asarray([row["spikesorting_merge_id"] for row in target_rows], dtype=str),
            ),
            "unit_id": ("unit", np.asarray([row["unit_id"] for row in target_rows], dtype=str)),
            "stable_unit_id": ("unit", target_stable),
            "group_unit_id": ("unit", np.asarray(raw_targets, dtype=str)),
            "source_unit": source_stable,
            "source_spikesorting_merge_id": (
                "source_unit",
                np.asarray([row["spikesorting_merge_id"] for row in source_rows], dtype=str),
            ),
            "source_unit_id": (
                "source_unit", np.asarray([row["unit_id"] for row in source_rows], dtype=str)
            ),
            "source_stable_unit_id": ("source_unit", source_stable),
            "source_group_unit_id": ("source_unit", np.asarray(raw_sources, dtype=str)),
        }
    )
    coefficient_group_ids = [str(value) for value in np.asarray(dataset["coef_ca1_unit_id"].values)]
    if parameters["source_predictor_mode"] == "unit_vector":
        if any(value not in source_lookup for value in coefficient_group_ids):
            raise ValueError("Full-fit coefficients contain unknown source units.")
        coefficient_rows = [source_lookup[value] for value in coefficient_group_ids]
        coefficient_stable = np.asarray(
            [row["stable_unit_id"] for row in coefficient_rows], dtype=str
        )
    else:
        if len(coefficient_group_ids) != 1:
            raise ValueError("mean_activity must produce exactly one source coefficient.")
        coefficient_stable = np.asarray(["mean_ca1_activity"], dtype=str)
    canonical = canonical.assign_coords(
        {
            "coef_source_unit": coefficient_stable,
            "coef_source_group_unit_id": (
                "coef_source_unit", np.asarray(coefficient_group_ids, dtype=str)
            ),
        }
    )
    canonical["ca1_unit_id"] = ("source_unit", source_stable)
    canonical["coef_ca1_unit_id"] = ("coef_source_unit", coefficient_stable)
    metric_matrix = np.vstack(
        [
            np.asarray(canonical[f"ripple_{metric}_mean"].values, dtype=float)
            for metric in METRIC_NAMES
        ]
    )
    finite_counts = np.isfinite(metric_matrix).sum(axis=0)
    target_valid = finite_counts == len(METRIC_NAMES)
    canonical.attrs.update(
        _dataset_json_attrs(
            metadata=metadata,
            parameters=parameters,
            upstream_provenance_json=upstream_provenance_json,
            legacy_artifact_provenance=legacy_artifact_provenance,
            analysis_status=(
                "valid"
                if np.all(target_valid)
                else ("partial_valid" if np.any(target_valid) else "no_valid_target_units")
            ),
            selected_ripple_events_sha256=_selected_ripple_events_sha256(inputs),
            artifact_origin=artifact_origin,
        )
    )
    canonical.attrs.update(
        {
            key: int(inputs[key])
            for key in (
                "n_ripples_before_selection",
                "n_ripples_removed_by_selection",
                "n_ripples_after_selection",
                "n_ripples_before_window_bounds",
                "n_ripples_removed_by_window_bounds",
                "n_ripples_after_window_bounds",
            )
        }
    )
    quality = pd.DataFrame(
        {
            "stable_unit_id": target_stable,
            "finite_metric_count": finite_counts,
            "valid_glm_metrics": target_valid,
            "unit_qc_status": np.where(
                target_valid,
                "valid",
                np.where(finite_counts > 0, "partial_nonfinite_metrics", "no_finite_metrics"),
            ),
        }
    )
    return canonical, quality


def _annotate_selected_units(
    selected_units: pd.DataFrame,
    *,
    dataset: Any,
    parameters: Mapping[str, Any],
    quality: pd.DataFrame | None,
    terminal: bool,
) -> pd.DataFrame:
    """Finish coefficient and fit-quality annotations in the all-unit audit."""
    output = selected_units.copy()
    if terminal:
        output["included_in_fit"] = False
        output["included_in_full_coefficient"] = False
        output["valid_glm_metrics"] = False
        output.loc[output["passes_spike_threshold"], "unit_qc_status"] = (
            "not_computed"
        )
        return output
    fitted_source = set(np.asarray(dataset["source_stable_unit_id"].values, dtype=str))
    fitted_target = set(np.asarray(dataset["stable_unit_id"].values, dtype=str))
    coefficient_sources = (
        set(np.asarray(dataset["coef_ca1_unit_id"].values, dtype=str))
        if parameters["source_predictor_mode"] == "unit_vector"
        else set()
    )
    source_mask = output["role"] == SOURCE_ROLE
    target_mask = output["role"] == TARGET_ROLE
    output.loc[source_mask, "included_in_fit"] = output.loc[
        source_mask, "stable_unit_id"
    ].isin(fitted_source)
    output.loc[target_mask, "included_in_fit"] = output.loc[
        target_mask, "stable_unit_id"
    ].isin(fitted_target)
    output.loc[source_mask, "included_in_full_coefficient"] = output.loc[
        source_mask, "stable_unit_id"
    ].isin(coefficient_sources)
    source_included = source_mask & output["included_in_fit"]
    output.loc[source_included, "unit_qc_status"] = np.where(
        output.loc[source_included, "included_in_full_coefficient"],
        "included_source_predictor",
        "included_source_no_full_coefficient",
    )
    if quality is not None:
        quality = quality.set_index("stable_unit_id")
        for index in output.index[target_mask & output["included_in_fit"]]:
            stable_id = output.at[index, "stable_unit_id"]
            if stable_id not in quality.index:
                raise ValueError("Target fit-quality rows do not match selected units.")
            output.at[index, "valid_glm_metrics"] = bool(
                quality.at[stable_id, "valid_glm_metrics"]
            )
            output.at[index, "unit_qc_status"] = str(
                quality.at[stable_id, "unit_qc_status"]
            )
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _summary_table(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    selected_units: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per fitted target unit with figure-facing metrics."""
    target = selected_units.loc[
        (selected_units["role"] == TARGET_ROLE) & selected_units["included_in_fit"]
    ].copy()
    if target.empty:
        return pd.DataFrame({column: pd.Series(dtype=object) for column in SUMMARY_COLUMNS})
    lookup = target.set_index("stable_unit_id", drop=False)
    stable_ids = [str(value) for value in np.asarray(dataset.coords["unit"].values)]
    rows = []
    for unit_index, stable_id in enumerate(stable_ids):
        audit = lookup.loc[stable_id]
        row = {
            **{field: str(audit[field]) for field in IDENTITY_COLUMNS},
            **metadata,
            "n_ripples": int(dataset.sizes["sample"]),
            "valid_glm_metrics": bool(audit["valid_glm_metrics"]),
            "unit_qc_status": str(audit["unit_qc_status"]),
        }
        for metric in METRIC_NAMES:
            for suffix in ("mean", "sem", "shuffle_mean", "shuffle_sd", "p_value"):
                row[f"ripple_{metric}_{suffix}"] = float(
                    dataset[f"ripple_{metric}_{suffix}"].values[unit_index]
                )
        rows.append(row)
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def _base_result(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
    selected_units: pd.DataFrame,
    summary: pd.DataFrame,
    dataset: Any,
    inputs: Mapping[str, Any],
    analysis_status: str,
    artifact_origin: str,
    legacy_artifact_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble one in-memory artifact bundle."""
    return {
        **metadata,
        "parameters": dict(parameters),
        "upstream_provenance": dict(upstream_provenance),
        "selected_ripple_events_sha256": _selected_ripple_events_sha256(inputs),
        "selected_units": selected_units,
        "summary": summary,
        "dataset": dataset,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance": dict(legacy_artifact_provenance or {}),
    }


def compute_ripple_glm(
    *,
    ripple_glm_id: Any,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_table: Any,
    epoch_interval: Any,
    source_spikes: Any,
    source_stable_unit_ids: Sequence[Mapping[str, Any]],
    target_spikes: Any,
    target_stable_unit_ids: Sequence[Mapping[str, Any]],
    upstream_provenance: Mapping[str, Any],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    expected_selected_ripple_events_sha256: str | None = None,
    **parameter_values: Any,
) -> dict[str, Any]:
    """Compute one fixed-direction ripple population GLM artifact in memory."""
    metadata = _metadata(
        ripple_glm_id=ripple_glm_id,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    parameters = _effective_parameters(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        **parameter_values,
    )
    normalized_provenance, provenance_json = _validate_upstream_provenance(
        upstream_provenance, parameters=parameters
    )
    source_identity = _identity_table(
        source_spikes, source_stable_unit_ids, role=SOURCE_ROLE, region=SOURCE_REGION
    )
    target_identity = _identity_table(
        target_spikes, target_stable_unit_ids, role=TARGET_ROLE, region=TARGET_REGION
    )
    inputs = _select_and_count_inputs(
        epoch=metadata["epoch"],
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        source_spikes=source_spikes,
        target_spikes=target_spikes,
        parameters=parameters,
    )
    _require_expected_event_hash(inputs, expected_selected_ripple_events_sha256)
    selected_units = _initial_selected_units(
        source_identity, target_identity, inputs, parameters
    )
    terminal_status = _terminal_status(
        source_identity=source_identity,
        target_identity=target_identity,
        inputs=inputs,
        n_splits=parameters["n_splits"],
    )
    if terminal_status is not None:
        selected_units = _annotate_selected_units(
            selected_units,
            dataset=None,
            parameters=parameters,
            quality=None,
            terminal=True,
        )
        dataset = _terminal_dataset(
            metadata=metadata,
            parameters=parameters,
            inputs=inputs,
            source_identity=source_identity,
            target_identity=target_identity,
            upstream_provenance_json=provenance_json,
            analysis_status=terminal_status,
        )
        result = _base_result(
            metadata=metadata,
            parameters=parameters,
            upstream_provenance=normalized_provenance,
            selected_units=selected_units,
            summary=_summary_table(dataset, metadata=metadata, selected_units=selected_units),
            dataset=dataset,
            inputs=inputs,
            analysis_status=terminal_status,
            artifact_origin="computed",
            legacy_artifact_provenance=None,
        )
        return validate_ripple_glm_result(result)

    ripple = _ripple_module()
    prepared = ripple._prepare_ripple_glm_epoch_inputs(
        metadata["epoch"],
        spikes={SOURCE_REGION: source_spikes, TARGET_REGION: target_spikes},
        epoch_interval=epoch_interval,
        ripple_table=inputs["selected_ripple_table"],
        min_spikes_per_ripple=parameters["min_spikes_per_ripple"],
        min_ca1_spikes_per_ripple=parameters["min_ca1_spikes_per_ripple"],
        ripple_window_s=parameters["ripple_window_s"],
        ripple_window_offset_s=parameters["ripple_window_offset_s"],
        source_window_s=parameters["source_window_s"],
        source_window_offset_s=parameters["source_window_offset_s"],
        target_window_s=parameters["target_window_s"],
        target_window_offset_s=parameters["target_window_offset_s"],
        n_splits=parameters["n_splits"],
    )
    results = ripple._fit_ripple_glm_on_prepared_epoch(
        metadata["epoch"],
        prepared_epoch=prepared,
        source_predictor_mode=parameters["source_predictor_mode"],
        n_shuffles_ripple=parameters["n_shuffles_ripple"],
        shuffle_seed=parameters["shuffle_seed"],
        ripple_window_s=parameters["ripple_window_s"],
        ripple_window_offset_s=parameters["ripple_window_offset_s"],
        source_window_s=parameters["source_window_s"],
        source_window_offset_s=parameters["source_window_offset_s"],
        target_window_s=parameters["target_window_s"],
        target_window_offset_s=parameters["target_window_offset_s"],
        ridge_strength=parameters["ridge_strength"],
        maxiter=parameters["maxiter"],
        tol=parameters["tol"],
    )
    results["min_spikes_per_ripple"] = parameters["min_spikes_per_ripple"]
    results["min_ca1_spikes_per_ripple"] = parameters["min_ca1_spikes_per_ripple"]
    fit_parameters = {
        **{
            key: value
            for key, value in parameters.items()
            if key not in {"parameter_name", "parameter_sha256", "output_rule_sha256"}
        },
        **{
            key: inputs[key]
            for key in (
                "n_ripples_before_selection",
                "n_ripples_removed_by_selection",
                "n_ripples_after_selection",
                "n_ripples_before_window_bounds",
                "n_ripples_removed_by_window_bounds",
                "n_ripples_after_window_bounds",
            )
        },
    }
    dataset = ripple.build_epoch_fit_dataset(
        results,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        epoch=metadata["epoch"],
        sources=normalized_provenance,
        fit_parameters=fit_parameters,
    )
    dataset, quality = _canonicalize_dataset(
        dataset,
        metadata=metadata,
        parameters=parameters,
        inputs=inputs,
        source_identity=source_identity,
        target_identity=target_identity,
        upstream_provenance_json=provenance_json,
        artifact_origin="computed",
        legacy_artifact_provenance=None,
    )
    selected_units = _annotate_selected_units(
        selected_units,
        dataset=dataset,
        parameters=parameters,
        quality=quality,
        terminal=False,
    )
    status = str(dataset.attrs["analysis_status"])
    result = _base_result(
        metadata=metadata,
        parameters=parameters,
        upstream_provenance=normalized_provenance,
        selected_units=selected_units,
        summary=_summary_table(dataset, metadata=metadata, selected_units=selected_units),
        dataset=dataset,
        inputs=inputs,
        analysis_status=status,
        artifact_origin="computed",
        legacy_artifact_provenance=None,
    )
    return validate_ripple_glm_result(result)


def _expected_fold_index(n_samples: int, n_splits: int) -> np.ndarray:
    """Return the deterministic test-fold assignment produced by sklearn KFold."""
    if n_samples < n_splits:
        return np.asarray([], dtype=int)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    output = np.empty(n_samples, dtype=int)
    start = 0
    for fold, size in enumerate(fold_sizes):
        output[start : start + size] = fold
        start += size
    return output


def _assert_array_close(
    observed: Any,
    expected: Any,
    *,
    name: str,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> None:
    """Require matching shapes and tightly equal numeric values."""
    observed_array = np.asarray(observed)
    expected_array = np.asarray(expected)
    if observed_array.shape != expected_array.shape or not np.allclose(
        observed_array,
        expected_array,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ):
        raise ValueError(f"{name} does not match its required scientific invariant.")


def _timestamp_subtraction_atol(left: Any, right: Any) -> np.ndarray:
    """Return one-float-ULP tolerances for differences of absolute times."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("Timestamp subtraction arrays must have matching shapes.")
    scale = np.maximum(np.abs(left_array), np.abs(right_array))
    return np.maximum(1e-12, np.spacing(scale))


def _validate_selected_units(
    table: Any,
    *,
    parameters: Mapping[str, Any],
    analysis_status: str,
) -> pd.DataFrame:
    """Validate and normalize the all-source/all-target unit audit."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("selected_units must be a pandas DataFrame.")
    if list(table.columns) != list(SELECTED_UNIT_COLUMNS):
        raise ValueError("selected_units columns do not match the canonical schema.")
    output = table.copy().reset_index(drop=True)
    if not output["role"].isin((SOURCE_ROLE, TARGET_ROLE)).all():
        raise ValueError("selected_units contains an unsupported unit role.")
    expected_region = output["role"].map(
        {SOURCE_ROLE: SOURCE_REGION, TARGET_ROLE: TARGET_REGION}
    )
    if output["region"].astype(str).tolist() != expected_region.tolist():
        raise ValueError("selected_units region does not match its source/target role.")
    for field in IDENTITY_COLUMNS:
        output[field] = output[field].map(str)
        if output[field].eq("").any():
            raise ValueError(f"selected_units {field} values must be non-empty.")
    expected_stable = (
        output["spikesorting_merge_id"] + ":" + output["unit_id"]
    )
    if output["stable_unit_id"].tolist() != expected_stable.tolist():
        raise ValueError("selected_units stable_unit_id is not the canonical identity.")
    if output["stable_unit_id"].duplicated().any():
        raise ValueError("selected_units stable identities must be unique across both groups.")
    for role in (SOURCE_ROLE, TARGET_ROLE):
        role_rows = output.loc[output["role"] == role]
        expected_index = np.arange(len(role_rows), dtype=int)
        observed_index = pd.to_numeric(
            role_rows["input_unit_index"], errors="raise"
        ).to_numpy(dtype=int)
        if not np.array_equal(observed_index, expected_index):
            raise ValueError(f"{role} input_unit_index must be contiguous and ordered.")
    means = pd.to_numeric(output["mean_spikes_per_ripple"], errors="raise").to_numpy(
        dtype=float
    )
    minimum = pd.to_numeric(
        output["minimum_mean_spikes_per_ripple"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(means)) or np.any(means < 0.0):
        raise ValueError("Mean spikes per ripple must be finite and non-negative.")
    expected_minimum = np.where(
        output["role"].to_numpy() == SOURCE_ROLE,
        parameters["min_ca1_spikes_per_ripple"],
        parameters["min_spikes_per_ripple"],
    )
    _assert_array_close(minimum, expected_minimum, name="unit spike thresholds")
    for column in (
        "passes_spike_threshold",
        "included_in_fit",
        "included_in_full_coefficient",
        "valid_glm_metrics",
    ):
        if not output[column].isin((True, False)).all():
            raise ValueError(f"selected_units {column} must be boolean.")
        output[column] = output[column].astype(bool)
    expected_pass = means >= minimum
    if not np.array_equal(output["passes_spike_threshold"], expected_pass):
        raise ValueError("passes_spike_threshold does not match the inclusive threshold.")
    if np.any(output["included_in_fit"] & ~output["passes_spike_threshold"]):
        raise ValueError("Only threshold-passing units may be included in the fit.")
    if np.any(output["included_in_full_coefficient"] & ~output["included_in_fit"]):
        raise ValueError("Only fitted source units may have full-fit coefficients.")
    if np.any(
        output.loc[output["role"] == TARGET_ROLE, "included_in_full_coefficient"]
    ):
        raise ValueError("Target units cannot be source coefficients.")
    if np.any(output.loc[output["role"] == SOURCE_ROLE, "valid_glm_metrics"]):
        raise ValueError("Source units do not have target GLM metrics.")
    if not output["unit_qc_status"].isin(UNIT_QC_STATUSES).all():
        raise ValueError("selected_units contains an unsupported unit_qc_status.")
    excluded = ~output["passes_spike_threshold"]
    if not output.loc[excluded, "unit_qc_status"].eq(
        "excluded_spike_threshold"
    ).all():
        raise ValueError("Excluded unit QC statuses do not match their threshold result.")
    terminal = analysis_status not in FITTED_ANALYSIS_STATUSES
    if terminal:
        if output["included_in_fit"].any() or output["valid_glm_metrics"].any():
            raise ValueError("Terminal results cannot claim fitted or valid units.")
        if not output.loc[~excluded, "unit_qc_status"].eq("not_computed").all():
            raise ValueError("Threshold-passing terminal units must be not_computed.")
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _validate_dataset_attrs(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance_json: str,
    selected_ripple_events_sha256: str,
    analysis_status: str,
    artifact_origin: str,
    legacy_artifact_provenance: Mapping[str, Any],
) -> None:
    """Require immutable metadata, parameters, and provenance in NetCDF attrs."""
    expected_scalars = {
        "ripple_glm_result_schema_version": RESULT_SCHEMA_VERSION,
        **metadata,
        "model_direction": "ca1_to_v1",
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": OUTPUT_RULE_SHA256,
        "upstream_provenance_json": upstream_provenance_json,
        "selected_ripple_events_sha256": selected_ripple_events_sha256,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
    }
    for name, expected in expected_scalars.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"Ripple GLM dataset has mismatched {name}.")
    effective = {
        key: value
        for key, value in parameters.items()
        if key not in {"parameter_name", "parameter_sha256", "output_rule_sha256"}
    }
    if json.loads(str(dataset.attrs.get("effective_parameters_json", "{}"))) != effective:
        raise ValueError("Ripple GLM dataset effective parameters differ from the row.")
    if json.loads(str(dataset.attrs.get("legacy_artifact_provenance_json", "{}"))) != dict(
        legacy_artifact_provenance
    ):
        raise ValueError("Ripple GLM dataset legacy provenance differs from the row.")


def _validate_window_variables(
    dataset: Any, *, parameters: Mapping[str, Any], selected_events_sha256: str
) -> None:
    """Validate exact sample windows and their deterministic event digest."""
    required = (
        "ripple_start_time_s",
        "source_window_start_s",
        "source_window_end_s",
        "target_window_start_s",
        "target_window_end_s",
    )
    for name in required:
        if name not in dataset or dataset[name].dims != ("sample",):
            raise ValueError(f"Ripple GLM dataset is missing canonical {name}.")
    arrays = {name: np.asarray(dataset[name].values, dtype=float) for name in required}
    n_samples = int(dataset.sizes["sample"])
    if any(value.shape != (n_samples,) for value in arrays.values()):
        raise ValueError("Ripple window arrays do not align to sample.")
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Ripple window arrays must be finite.")
    if n_samples > 1 and np.any(np.diff(arrays["ripple_start_time_s"]) <= 0.0):
        raise ValueError("Ripple anchors must be strictly increasing.")
    _assert_array_close(
        arrays["source_window_start_s"] - arrays["ripple_start_time_s"],
        np.full(n_samples, parameters["source_window_offset_s"], dtype=float),
        name="source window offset",
        rtol=0.0,
        atol=_timestamp_subtraction_atol(
            arrays["source_window_start_s"], arrays["ripple_start_time_s"]
        ),
    )
    _assert_array_close(
        arrays["source_window_end_s"] - arrays["source_window_start_s"],
        np.full(n_samples, parameters["source_window_s"], dtype=float),
        name="source window width",
        rtol=0.0,
        atol=_timestamp_subtraction_atol(
            arrays["source_window_end_s"], arrays["source_window_start_s"]
        ),
    )
    _assert_array_close(
        arrays["target_window_start_s"] - arrays["ripple_start_time_s"],
        np.full(n_samples, parameters["target_window_offset_s"], dtype=float),
        name="target window offset",
        rtol=0.0,
        atol=_timestamp_subtraction_atol(
            arrays["target_window_start_s"], arrays["ripple_start_time_s"]
        ),
    )
    _assert_array_close(
        arrays["target_window_end_s"] - arrays["target_window_start_s"],
        np.full(n_samples, parameters["target_window_s"], dtype=float),
        name="target window width",
        rtol=0.0,
        atol=_timestamp_subtraction_atol(
            arrays["target_window_end_s"], arrays["target_window_start_s"]
        ),
    )
    digest = _provenance_sha256({name: value.tolist() for name, value in arrays.items()})
    if digest != selected_events_sha256:
        raise ValueError("Selected ripple event hash does not match dataset windows.")
    if "ripple_window_start_s" in dataset:
        _assert_array_close(
            dataset["ripple_window_start_s"],
            arrays["target_window_start_s"],
            name="legacy target-window start alias",
        )
    if "ripple_window_end_s" in dataset:
        _assert_array_close(
            dataset["ripple_window_end_s"],
            arrays["target_window_end_s"],
            name="legacy target-window end alias",
        )
    count_names = (
        "n_ripples_before_selection",
        "n_ripples_removed_by_selection",
        "n_ripples_after_selection",
        "n_ripples_before_window_bounds",
        "n_ripples_removed_by_window_bounds",
        "n_ripples_after_window_bounds",
    )
    try:
        counts = {name: int(dataset.attrs[name]) for name in count_names}
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Ripple GLM dataset lacks valid ripple selection counts.") from exc
    if any(value < 0 for value in counts.values()):
        raise ValueError("Ripple GLM selection counts must be non-negative.")
    if (
        counts["n_ripples_before_selection"]
        - counts["n_ripples_removed_by_selection"]
        != counts["n_ripples_after_selection"]
        or counts["n_ripples_before_window_bounds"]
        != counts["n_ripples_after_selection"]
        or counts["n_ripples_before_window_bounds"]
        - counts["n_ripples_removed_by_window_bounds"]
        != counts["n_ripples_after_window_bounds"]
        or counts["n_ripples_after_window_bounds"] != n_samples
    ):
        raise ValueError("Ripple GLM selection-count arithmetic is inconsistent.")


def _validate_metric_arithmetic(dataset: Any, *, parameters: Mapping[str, Any]) -> None:
    """Recompute real fold metrics and all shuffle-derived summaries."""
    ripple = _ripple_module()
    n_samples = int(dataset.sizes["sample"])
    n_units = int(dataset.sizes["unit"])
    n_folds = int(dataset.sizes["fold"])
    n_shuffles = int(dataset.sizes["shuffle"])
    if n_folds != parameters["n_splits"] or n_shuffles != parameters["n_shuffles_ripple"]:
        raise ValueError("Ripple GLM fold/shuffle dimensions differ from parameters.")
    required = ("ripple_fold_index", "ripple_observed_count_oof", "ripple_predicted_count_oof")
    if any(name not in dataset for name in required):
        raise ValueError("Ripple GLM dataset lacks samplewise fit arrays.")
    fold_index = np.asarray(dataset["ripple_fold_index"].values, dtype=int)
    expected_folds = _expected_fold_index(n_samples, n_folds)
    if not np.array_equal(fold_index, expected_folds):
        raise ValueError("ripple_fold_index does not match contiguous KFold.")
    observed = np.asarray(dataset["ripple_observed_count_oof"].values, dtype=float)
    predicted = np.asarray(dataset["ripple_predicted_count_oof"].values, dtype=float)
    if observed.shape != (n_samples, n_units) or predicted.shape != observed.shape:
        raise ValueError("Samplewise observed/predicted count shapes are invalid.")
    if not np.all(np.isfinite(observed)) or np.any(observed < 0.0) or not np.allclose(
        observed, np.rint(observed), rtol=0.0, atol=1e-12
    ):
        raise ValueError("Observed ripple counts must be finite non-negative integers.")
    if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0.0):
        raise ValueError("Predicted ripple counts must be finite and positive.")
    recomputed: dict[str, np.ndarray] = {
        name: np.full((n_folds, n_units), np.nan, dtype=float) for name in METRIC_NAMES
    }
    for fold in range(n_folds):
        test = fold_index == fold
        train = ~test
        recomputed["pseudo_r2"][fold] = ripple.mcfadden_pseudo_r2_per_neuron(
            observed[test], predicted[test], observed[train]
        )
        recomputed["mae"][fold] = ripple.mae_per_neuron(
            observed[test], predicted[test]
        )
        recomputed["devexp"][fold] = ripple.deviance_explained_per_neuron(
            observed[test], predicted[test], observed[train]
        )
        recomputed["bits_per_spike"][fold] = ripple.bits_per_spike_per_neuron(
            observed[test], predicted[test], observed[train]
        )
    for metric in METRIC_NAMES:
        fold_name = f"{metric}_ripple_folds"
        shuffle_name = f"{metric}_ripple_shuff_folds"
        if fold_name not in dataset or shuffle_name not in dataset:
            raise ValueError(f"Ripple GLM dataset is missing {metric} fold arrays.")
        folds = np.asarray(dataset[fold_name].values, dtype=float)
        shuffles = np.asarray(dataset[shuffle_name].values, dtype=float)
        if folds.shape != (n_folds, n_units) or shuffles.shape != (
            n_folds,
            n_shuffles,
            n_units,
        ):
            raise ValueError(f"Ripple GLM {metric} fold dimensions are invalid.")
        _assert_array_close(folds, recomputed[metric], name=f"{metric} real folds", rtol=2e-6, atol=2e-7)
        summary = ripple.summarize_ripple_metric_against_shuffle(
            folds,
            shuffles,
            higher_is_better=ripple.HIGHER_IS_BETTER_BY_METRIC[metric],
        )
        for suffix, key in (
            ("mean", "real_mean"),
            ("sem", "real_sem"),
            ("shuffle_mean", "shuffle_mean"),
            ("shuffle_sd", "shuffle_sd"),
            ("p_value", "unit_p_value"),
        ):
            variable = f"ripple_{metric}_{suffix}"
            if variable not in dataset or dataset[variable].dims != ("unit",):
                raise ValueError(f"Ripple GLM dataset is missing {variable}.")
            _assert_array_close(
                dataset[variable].values,
                summary[key],
                name=variable,
                rtol=2e-6,
                atol=2e-7,
            )
        p_values = np.asarray(dataset[f"ripple_{metric}_p_value"].values, dtype=float)
        finite = np.isfinite(p_values)
        if np.any((p_values[finite] < 0.0) | (p_values[finite] > 1.0)):
            raise ValueError(f"Ripple GLM {metric} p-values must lie in [0, 1].")


def _validate_fitted_dataset(
    dataset: Any,
    *,
    selected_units: pd.DataFrame,
    parameters: Mapping[str, Any],
) -> None:
    """Validate unit identity coordinates, coefficients, and metric arithmetic."""
    for dimension in ("sample", "unit", "source_unit", "coef_source_unit", "fold", "shuffle"):
        if dimension not in dataset.dims:
            raise ValueError(f"Ripple GLM dataset is missing dimension {dimension}.")
    target = selected_units.loc[
        (selected_units["role"] == TARGET_ROLE) & selected_units["included_in_fit"]
    ]
    source = selected_units.loc[
        (selected_units["role"] == SOURCE_ROLE) & selected_units["included_in_fit"]
    ]
    if list(np.asarray(dataset.coords["unit"].values, dtype=str)) != target[
        "stable_unit_id"
    ].tolist():
        raise ValueError("Dataset target units do not match the selected-unit audit.")
    if list(np.asarray(dataset.coords["source_unit"].values, dtype=str)) != source[
        "stable_unit_id"
    ].tolist():
        raise ValueError("Dataset source units do not match the selected-unit audit.")
    identity_coordinates = {
        "stable_unit_id": target["stable_unit_id"],
        "spikesorting_merge_id": target["spikesorting_merge_id"],
        "unit_id": target["unit_id"],
        "group_unit_id": target["group_unit_id"],
    }
    for name, expected in identity_coordinates.items():
        if name not in dataset.coords or list(
            np.asarray(dataset.coords[name].values, dtype=str)
        ) != expected.tolist():
            raise ValueError(f"Dataset target coordinate {name} is misaligned.")
    for name, expected in {
        "source_stable_unit_id": source["stable_unit_id"],
        "source_spikesorting_merge_id": source["spikesorting_merge_id"],
        "source_unit_id": source["unit_id"],
        "source_group_unit_id": source["group_unit_id"],
    }.items():
        if name not in dataset.coords or list(
            np.asarray(dataset.coords[name].values, dtype=str)
        ) != expected.tolist():
            raise ValueError(f"Dataset source coordinate {name} is misaligned.")
    coefficients = np.asarray(dataset["coef_ca1_full_all"].values, dtype=float)
    intercepts = np.asarray(dataset["coef_intercept_full_all"].values, dtype=float)
    if coefficients.shape != (dataset.sizes["coef_source_unit"], dataset.sizes["unit"]):
        raise ValueError("Full-fit source coefficient shape is invalid.")
    if intercepts.shape != (dataset.sizes["unit"],):
        raise ValueError("Full-fit intercept shape is invalid.")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(intercepts)):
        raise ValueError("Full-fit coefficients and intercepts must be finite.")
    coefficient_units = set(np.asarray(dataset["coef_ca1_unit_id"].values, dtype=str))
    expected_coefficients = set(
        source.loc[source["included_in_full_coefficient"], "stable_unit_id"]
    )
    if parameters["source_predictor_mode"] == "unit_vector":
        if coefficient_units != expected_coefficients:
            raise ValueError("Coefficient source units do not match the unit audit.")
    elif coefficient_units != {"mean_ca1_activity"}:
        raise ValueError("mean_activity must have one synthetic source coefficient.")
    _validate_metric_arithmetic(dataset, parameters=parameters)


def _expected_terminal_status(selected_units: pd.DataFrame, n_ripples: int, n_splits: int) -> str | None:
    """Derive terminal status from the persisted all-unit audit and event count."""
    source = selected_units["role"] == SOURCE_ROLE
    target = selected_units["role"] == TARGET_ROLE
    if not source.any():
        return "no_source_units"
    if not target.any():
        return "no_target_units"
    if n_ripples == 0:
        return "no_ripples"
    if n_ripples < n_splits:
        return "insufficient_ripples"
    if not selected_units.loc[source, "passes_spike_threshold"].any():
        return "no_eligible_source_units"
    if not selected_units.loc[target, "passes_spike_threshold"].any():
        return "no_eligible_target_units"
    return None


def validate_ripple_glm_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one canonical ripple GLM artifact bundle."""
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    metadata = _metadata(
        ripple_glm_id=result.get("ripple_glm_id"),
        animal_name=result.get("animal_name"),
        date=result.get("date"),
        epoch=result.get("epoch"),
    )
    raw_parameters = result.get("parameters")
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("result parameters must be a mapping.")
    parameter_keys = (
        "ripple_window_s",
        "ripple_window_offset_s",
        "source_window_s",
        "source_window_offset_s",
        "target_window_s",
        "target_window_offset_s",
        "ripple_selection_mode",
        "source_predictor_mode",
        "min_spikes_per_ripple",
        "min_ca1_spikes_per_ripple",
        "n_splits",
        "n_shuffles_ripple",
        "ridge_strength",
        "shuffle_seed",
        "maxiter",
        "tol",
        "expected_detector_zscore_threshold",
        "require_speed_gated",
    )
    missing = [key for key in parameter_keys if key not in raw_parameters]
    if missing:
        raise ValueError(f"Ripple GLM parameters are missing {missing!r}.")
    parameters = _effective_parameters(
        parameter_name=raw_parameters.get("parameter_name"),
        parameter_sha256=raw_parameters.get("parameter_sha256"),
        output_rule_sha256=raw_parameters.get("output_rule_sha256"),
        **{key: raw_parameters[key] for key in parameter_keys},
    )
    normalized_provenance, provenance_json = _validate_upstream_provenance(
        result.get("upstream_provenance"), parameters=parameters
    )
    selected_event_hash = str(result.get("selected_ripple_events_sha256", ""))
    if len(selected_event_hash) != 64:
        raise ValueError("selected_ripple_events_sha256 must be one SHA-256 digest.")
    analysis_status = str(result.get("analysis_status", ""))
    if analysis_status not in ANALYSIS_STATUSES:
        raise ValueError("Ripple GLM result has an unsupported analysis_status.")
    artifact_origin = str(result.get("artifact_origin", ""))
    if artifact_origin not in {"computed", "registered_existing"}:
        raise ValueError("Ripple GLM artifact_origin is unsupported.")
    legacy = result.get("legacy_artifact_provenance", {})
    if not isinstance(legacy, Mapping):
        raise TypeError("legacy_artifact_provenance must be a mapping.")
    legacy = dict(legacy)
    if artifact_origin == "computed" and legacy:
        raise ValueError("Computed ripple GLM results cannot claim legacy provenance.")
    if artifact_origin == "registered_existing" and not legacy:
        raise ValueError("Registered ripple GLM results require legacy provenance.")
    selected_units = _validate_selected_units(
        result.get("selected_units"),
        parameters=parameters,
        analysis_status=analysis_status,
    )
    dataset = result.get("dataset")
    if dataset is None or not hasattr(dataset, "attrs") or not hasattr(dataset, "sizes"):
        raise TypeError("result dataset must be an xarray Dataset-like object.")
    _validate_dataset_attrs(
        dataset,
        metadata=metadata,
        parameters=parameters,
        upstream_provenance_json=provenance_json,
        selected_ripple_events_sha256=selected_event_hash,
        analysis_status=analysis_status,
        artifact_origin=artifact_origin,
        legacy_artifact_provenance=legacy,
    )
    _validate_window_variables(
        dataset,
        parameters=parameters,
        selected_events_sha256=selected_event_hash,
    )
    n_ripples = int(dataset.sizes.get("sample", -1))
    expected_terminal = _expected_terminal_status(
        selected_units, n_ripples, parameters["n_splits"]
    )
    if analysis_status in FITTED_ANALYSIS_STATUSES:
        if expected_terminal is not None:
            raise ValueError(
                "Non-terminal Ripple GLM result has terminal inputs: " + expected_terminal
            )
        _validate_fitted_dataset(
            dataset, selected_units=selected_units, parameters=parameters
        )
        target = selected_units.loc[
            (selected_units["role"] == TARGET_ROLE)
            & selected_units["included_in_fit"]
        ]
        expected_status = (
            "valid"
            if target["valid_glm_metrics"].all()
            else (
                "partial_valid"
                if target["valid_glm_metrics"].any()
                else "no_valid_target_units"
            )
        )
        if analysis_status != expected_status:
            raise ValueError("Ripple GLM analysis_status differs from target-unit QC.")
    else:
        if expected_terminal != analysis_status:
            raise ValueError(
                "Ripple GLM terminal status differs from its persisted inputs."
            )
        if int(dataset.sizes.get("unit", -1)) != 0 or int(
            dataset.sizes.get("source_unit", -1)
        ) != 0:
            raise ValueError("Terminal Ripple GLM datasets must have empty fit dimensions.")
    summary = result.get("summary")
    if not isinstance(summary, pd.DataFrame) or list(summary.columns) != list(
        SUMMARY_COLUMNS
    ):
        raise ValueError("Ripple GLM summary does not match its canonical schema.")
    expected_summary = _summary_table(
        dataset, metadata=metadata, selected_units=selected_units
    )
    try:
        pd.testing.assert_frame_equal(
            summary.reset_index(drop=True),
            expected_summary.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError("Ripple GLM summary differs from its NetCDF dataset.") from exc
    source_mask = selected_units["role"] == SOURCE_ROLE
    target_mask = selected_units["role"] == TARGET_ROLE
    return {
        **metadata,
        "parameters": parameters,
        "upstream_provenance": normalized_provenance,
        "selected_ripple_events_sha256": selected_event_hash,
        "selected_units": selected_units,
        "summary": expected_summary,
        "dataset": dataset,
        "analysis_status": analysis_status,
        "artifact_origin": artifact_origin,
        "legacy_artifact_provenance": legacy,
        "n_source_units": int(source_mask.sum()),
        "n_target_units": int(target_mask.sum()),
        "n_source_units_in_fit": int(
            selected_units.loc[source_mask, "included_in_fit"].sum()
        ),
        "n_target_units_in_fit": int(
            selected_units.loc[target_mask, "included_in_fit"].sum()
        ),
        "n_valid_target_units": int(
            selected_units.loc[target_mask, "valid_glm_metrics"].sum()
        ),
        "n_ripples": n_ripples,
        "selected_units_sha256": _table_sha256(selected_units),
    }


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return repeated immutable manifest fields for one validated result."""
    return {
        "ripple_glm_id": result["ripple_glm_id"],
        "animal_name": result["animal_name"],
        "date": result["date"],
        "epoch": result["epoch"],
        "parameter_name": result["parameters"]["parameter_name"],
        "parameter_sha256": result["parameters"]["parameter_sha256"],
        "output_rule_sha256": result["parameters"]["output_rule_sha256"],
        "upstream_provenance_json": json.dumps(
            result["upstream_provenance"], sort_keys=True, separators=(",", ":")
        ),
        "selected_ripple_events_sha256": result["selected_ripple_events_sha256"],
        "n_source_units": result["n_source_units"],
        "n_target_units": result["n_target_units"],
        "n_source_units_in_fit": result["n_source_units_in_fit"],
        "n_target_units_in_fit": result["n_target_units_in_fit"],
        "n_valid_target_units": result["n_valid_target_units"],
        "n_ripples": result["n_ripples"],
        "selected_units_sha256": result["selected_units_sha256"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": json.dumps(
            result["legacy_artifact_provenance"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def _load_dataset(path: Path) -> Any:
    """Load one NetCDF dataset eagerly and close its backing file."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def write_ripple_glm_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write, checksum, and reload one ripple GLM bundle."""
    validated = validate_ripple_glm_result(result)
    destination = Path(path)
    if destination.name != validated["ripple_glm_id"]:
        raise ValueError("Artifact directory name must equal ripple_glm_id.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite ripple GLM artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        validated["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME, index=False
        )
        validated["summary"].to_parquet(temporary / SUMMARY_FILENAME, index=False)
        validated["dataset"].to_netcdf(temporary / RESULT_FILENAME)
        common = _manifest_common(validated)
        rows = []
        for key, filename, kind in (
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("summary", SUMMARY_FILENAME, "parquet"),
            ("ripple_glm", RESULT_FILENAME, "netcdf"),
        ):
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
            temporary / MANIFEST_FILENAME, index=False
        )
        load_ripple_glm_artifact(temporary, _allow_temporary_name=True)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "artifact_dir": destination,
        "artifact_manifest_path": destination / MANIFEST_FILENAME,
        "selected_units_path": destination / SELECTED_UNITS_FILENAME,
        "summary_path": destination / SUMMARY_FILENAME,
        "result_path": destination / RESULT_FILENAME,
    }


def load_ripple_glm_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one complete ripple GLM bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ripple GLM manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or len(manifest) != 3:
        raise ValueError("Ripple GLM manifest does not have the canonical schema.")
    expected = {
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "summary": (SUMMARY_FILENAME, "parquet"),
        "ripple_glm": (RESULT_FILENAME, "netcdf"),
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("Ripple GLM manifest lacks canonical artifacts.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("Ripple GLM manifest names or kinds are stale.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Ripple GLM artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or _file_sha256(
            artifact_path
        ) != str(row["sha256"]):
            raise ValueError(f"Ripple GLM checksum mismatch: {artifact_path}")
    first = manifest.iloc[0]
    for name in MANIFEST_COLUMNS[5:]:
        if not np.all(manifest[name].astype(str) == str(first[name])):
            raise ValueError(f"Ripple GLM manifest has inconsistent {name!r}.")
    result_id = str(first["ripple_glm_id"])
    if not _allow_temporary_name and directory.name != result_id:
        raise ValueError("Artifact directory name does not match ripple_glm_id.")
    dataset = _load_dataset(directory / RESULT_FILENAME)
    effective = json.loads(str(dataset.attrs.get("effective_parameters_json", "{}")))
    parameters = {
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        **effective,
    }
    result = validate_ripple_glm_result(
        {
            "ripple_glm_id": result_id,
            "animal_name": str(first["animal_name"]),
            "date": str(first["date"]),
            "epoch": str(first["epoch"]),
            "parameters": parameters,
            "upstream_provenance": json.loads(str(first["upstream_provenance_json"])),
            "selected_ripple_events_sha256": str(
                first["selected_ripple_events_sha256"]
            ),
            "selected_units": pd.read_parquet(
                directory / SELECTED_UNITS_FILENAME
            ),
            "summary": pd.read_parquet(directory / SUMMARY_FILENAME),
            "dataset": dataset,
            "analysis_status": str(first["analysis_status"]),
            "artifact_origin": str(first["artifact_origin"]),
            "legacy_artifact_provenance": json.loads(
                str(first["legacy_artifact_provenance_json"])
            ),
        }
    )
    count_fields = (
        "n_source_units",
        "n_target_units",
        "n_source_units_in_fit",
        "n_target_units_in_fit",
        "n_valid_target_units",
        "n_ripples",
    )
    if any(result[name] != int(first[name]) for name in count_fields):
        raise ValueError("Ripple GLM manifest counts differ from loaded artifacts.")
    if result["selected_units_sha256"] != str(first["selected_units_sha256"]):
        raise ValueError("Ripple GLM selected-unit digest differs from its manifest.")
    return result


def _legacy_effective_parameters(dataset: Any) -> dict[str, Any]:
    """Extract effective scientific parameters from one supported legacy file."""
    try:
        fit = json.loads(str(dataset.attrs["fit_parameters_json"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Legacy ripple GLM lacks valid fit_parameters_json.") from exc
    if not isinstance(fit, dict):
        raise ValueError("Legacy ripple GLM fit_parameters_json must encode a mapping.")

    def attr_or_fit(name: str, default: Any = None) -> Any:
        if name in dataset.attrs:
            return dataset.attrs[name]
        return fit.get(name, default)

    target_window_s = attr_or_fit(
        "target_window_s", attr_or_fit("ripple_window_s", None)
    )
    target_offset = attr_or_fit(
        "target_window_offset_s", attr_or_fit("ripple_window_offset_s", 0.0)
    )
    source_window_s = attr_or_fit("source_window_s", target_window_s)
    source_offset = attr_or_fit("source_window_offset_s", target_offset)
    required = {
        "ripple_window_s": target_window_s,
        "source_window_s": source_window_s,
        "target_window_s": target_window_s,
        "min_spikes_per_ripple": fit.get("min_spikes_per_ripple"),
        "min_ca1_spikes_per_ripple": fit.get("min_ca1_spikes_per_ripple"),
        "ridge_strength": fit.get("ridge_strength"),
        "shuffle_seed": fit.get("shuffle_seed"),
        "maxiter": fit.get("maxiter"),
        "tol": fit.get("tol"),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Legacy ripple GLM parameters are missing {missing!r}.")
    return validate_ripple_glm_parameters(
        ripple_window_s=float(target_window_s),
        ripple_window_offset_s=float(target_offset),
        source_window_s=float(source_window_s),
        source_window_offset_s=float(source_offset),
        target_window_s=float(target_window_s),
        target_window_offset_s=float(target_offset),
        ripple_selection_mode=str(
            attr_or_fit("ripple_selection_mode", "allripples")
        ),
        source_predictor_mode=str(
            attr_or_fit("source_predictor_mode", "unit_vector")
        ),
        min_spikes_per_ripple=float(required["min_spikes_per_ripple"]),
        min_ca1_spikes_per_ripple=float(required["min_ca1_spikes_per_ripple"]),
        n_splits=int(fit.get("n_splits", dataset.sizes.get("fold", -1))),
        n_shuffles_ripple=int(
            fit.get("n_shuffles_ripple", dataset.sizes.get("shuffle", -1))
        ),
        ridge_strength=float(required["ridge_strength"]),
        shuffle_seed=int(required["shuffle_seed"]),
        maxiter=int(required["maxiter"]),
        tol=float(required["tol"]),
        expected_detector_zscore_threshold=(
            DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD
        ),
        require_speed_gated=True,
    )


def _validate_legacy_parameters(
    dataset: Any, parameters: Mapping[str, Any]
) -> None:
    """Require every model-affecting legacy parameter to match the selected row."""
    observed = _legacy_effective_parameters(dataset)
    for name, expected in parameters.items():
        if name in {"parameter_name", "parameter_sha256", "output_rule_sha256"}:
            continue
        actual = observed[name]
        if isinstance(expected, float):
            if not np.isclose(float(actual), expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Legacy ripple GLM has mismatched parameter {name}.")
        elif actual != expected:
            raise ValueError(f"Legacy ripple GLM has mismatched parameter {name}.")


def _normalize_legacy_dataset(dataset: Any, *, parameters: Mapping[str, Any]) -> Any:
    """Add only deterministic aliases omitted by older legacy schemas."""
    normalized = dataset.copy(deep=True)
    if "ripple_start_time_s" not in normalized:
        if "ripple_window_start_s" not in normalized:
            raise ValueError("Legacy ripple GLM lacks ripple anchor/window times.")
        normalized["ripple_start_time_s"] = (
            normalized["ripple_window_start_s"]
            - parameters["target_window_offset_s"]
        )
    if "target_window_start_s" not in normalized:
        normalized["target_window_start_s"] = normalized["ripple_window_start_s"]
    if "target_window_end_s" not in normalized:
        normalized["target_window_end_s"] = normalized["ripple_window_end_s"]
    if "source_window_start_s" not in normalized:
        normalized["source_window_start_s"] = (
            normalized["ripple_start_time_s"]
            + parameters["source_window_offset_s"]
        )
    if "source_window_end_s" not in normalized:
        normalized["source_window_end_s"] = (
            normalized["source_window_start_s"] + parameters["source_window_s"]
        )
    if "coef_ca1_unit_id" not in normalized:
        raise ValueError("Legacy ripple GLM lacks coefficient-aligned source IDs.")
    if "coef_source_feature_name" not in normalized:
        raw = np.asarray(normalized["coef_ca1_unit_id"].values)
        names = (
            np.asarray([f"ca1_unit_{value}" for value in raw], dtype=str)
            if parameters["source_predictor_mode"] == "unit_vector"
            else np.asarray(["mean_ca1_activity"], dtype=str)
        )
        normalized["coef_source_feature_name"] = ("coef_source_unit", names)
    return normalized


def _resolver_value(
    resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    legacy_unit_id: Any,
    *,
    role: str,
) -> dict[str, Any]:
    """Resolve one legacy sorting ID without silently accepting ambiguity."""
    if callable(resolver):
        resolved = resolver(legacy_unit_id)
    elif isinstance(resolver, Mapping):
        try:
            resolved = resolver[legacy_unit_id]
        except (KeyError, TypeError):
            matches = [
                value
                for key, value in resolver.items()
                if str(key) == str(legacy_unit_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Legacy {role} unit {legacy_unit_id!r} has "
                    f"{len(matches)} identity resolver matches."
                )
            resolved = matches[0]
    else:
        raise TypeError(f"{role}_legacy_unit_identity_resolver must be a mapping or callable.")
    if not isinstance(resolved, Mapping):
        raise TypeError(f"Resolved legacy {role} identity must be a mapping.")
    missing = [
        name
        for name in ("group_unit_id", "spikesorting_merge_id", "unit_id")
        if name not in resolved
    ]
    if missing:
        raise ValueError(
            f"Resolved legacy {role} identity is missing fields {missing!r}."
        )
    return {name: resolved[name] for name in resolved}


def _resolve_legacy_units(
    legacy_unit_ids: Sequence[Any],
    *,
    resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    current_identity: pd.DataFrame,
    role: str,
) -> list[str]:
    """Map a legacy axis to current runtime keys and verify persistent identity."""
    current = _group_id_lookup(current_identity)
    resolved_group_ids: list[str] = []
    for legacy_unit_id in legacy_unit_ids:
        resolved = _resolver_value(
            resolver, legacy_unit_id, role=role
        )
        group_id = str(resolved["group_unit_id"])
        if group_id not in current:
            raise ValueError(
                f"Legacy {role} unit {legacy_unit_id!r} resolves to an unknown runtime key."
            )
        expected = current[group_id]
        for name in ("spikesorting_merge_id", "unit_id"):
            if str(resolved[name]) != str(expected[name]):
                raise ValueError(
                    f"Legacy {role} unit {legacy_unit_id!r} resolves to mismatched {name}."
                )
        if "stable_unit_id" in resolved and str(resolved["stable_unit_id"]) != str(
            expected["stable_unit_id"]
        ):
            raise ValueError(
                f"Legacy {role} unit {legacy_unit_id!r} resolves to mismatched stable_unit_id."
            )
        resolved_group_ids.append(group_id)
    if len(resolved_group_ids) != len(set(resolved_group_ids)):
        raise ValueError(f"Legacy {role} unit resolver maps multiple IDs to one runtime key.")
    return resolved_group_ids


def _rekey_legacy_dataset(
    dataset: Any,
    *,
    source_group_ids: Sequence[str],
    target_group_ids: Sequence[str],
    coefficient_source_group_ids: Sequence[str],
    source_predictor_mode: str,
) -> Any:
    """Replace verified legacy sorting IDs with current ephemeral runtime keys."""
    rekeyed = dataset.assign_coords(
        {
            "unit": np.asarray(target_group_ids, dtype=str),
            "source_unit": np.arange(len(source_group_ids), dtype=int),
            "coef_source_unit": np.arange(
                len(coefficient_source_group_ids), dtype=int
            ),
        }
    )
    rekeyed["ca1_unit_id"] = (
        "source_unit", np.asarray(source_group_ids, dtype=str)
    )
    if source_predictor_mode == "unit_vector":
        coefficient_ids = np.asarray(coefficient_source_group_ids, dtype=str)
    else:
        coefficient_ids = np.asarray([-1], dtype=int)
    rekeyed["coef_ca1_unit_id"] = ("coef_source_unit", coefficient_ids)
    return rekeyed


def _validate_legacy_against_inputs(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    inputs: Mapping[str, Any],
    source_identity: pd.DataFrame,
    target_identity: pd.DataFrame,
    source_legacy_unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    target_legacy_unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
) -> Any:
    """Verify legacy input alignment, metric arithmetic, and coefficient shape."""
    expected_attrs = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "epoch": metadata["epoch"],
        "source_region": SOURCE_REGION,
        "target_region": TARGET_REGION,
        "model_direction": "ca1_to_v1",
    }
    for name, expected in expected_attrs.items():
        if str(dataset.attrs.get(name, "")) != expected:
            raise ValueError(f"Legacy ripple GLM has mismatched {name}.")
    _validate_legacy_parameters(dataset, parameters)
    n_samples = int(inputs["n_ripples_after_window_bounds"])
    if int(dataset.sizes.get("sample", -1)) != n_samples:
        raise ValueError("Legacy ripple GLM sample count differs from selected events.")
    for variable, input_name in (
        ("ripple_start_time_s", "ripple_start_time_s"),
        ("source_window_start_s", "source_window_start_s"),
        ("source_window_end_s", "source_window_end_s"),
        ("target_window_start_s", "target_window_start_s"),
        ("target_window_end_s", "target_window_end_s"),
    ):
        _assert_array_close(
            dataset[variable].values,
            inputs[input_name],
            name=f"legacy {variable}",
            rtol=1e-10,
            atol=1e-10,
        )
    expected_source_ids = [
        str(value)
        for value in np.asarray(inputs["source_group_ids"])[inputs["source_keep"]]
    ]
    expected_target_ids = [
        str(value)
        for value in np.asarray(inputs["target_group_ids"])[inputs["target_keep"]]
    ]
    raw_source_ids = np.asarray(dataset["ca1_unit_id"].values).tolist()
    raw_target_ids = np.asarray(dataset.coords["unit"].values).tolist()
    observed_source_ids = _resolve_legacy_units(
        raw_source_ids,
        resolver=source_legacy_unit_identity_resolver,
        current_identity=source_identity,
        role=SOURCE_ROLE,
    )
    observed_target_ids = _resolve_legacy_units(
        raw_target_ids,
        resolver=target_legacy_unit_identity_resolver,
        current_identity=target_identity,
        role=TARGET_ROLE,
    )
    if observed_source_ids != expected_source_ids:
        raise ValueError("Legacy ripple GLM source units differ from NWB-derived inputs.")
    if observed_target_ids != expected_target_ids:
        raise ValueError("Legacy ripple GLM target units differ from NWB-derived inputs.")
    expected_observed = np.asarray(inputs["target_counts"], dtype=float)[
        :, inputs["target_keep"]
    ]
    _assert_array_close(
        dataset["ripple_observed_count_oof"].values,
        expected_observed,
        name="legacy target spike counts",
        rtol=0.0,
        atol=1e-12,
    )
    expected_fold = _expected_fold_index(n_samples, parameters["n_splits"])
    if not np.array_equal(
        np.asarray(dataset["ripple_fold_index"].values, dtype=int), expected_fold
    ):
        raise ValueError("Legacy ripple GLM folds differ from contiguous KFold.")
    source_counts = np.asarray(inputs["source_counts"], dtype=float)[
        :, inputs["source_keep"]
    ]
    if parameters["source_predictor_mode"] == "unit_vector":
        coefficient_keep = source_counts.std(axis=0) > 1e-6
        expected_coef_ids = np.asarray(expected_source_ids)[coefficient_keep].tolist()
        raw_coefficient_ids = np.asarray(dataset["coef_ca1_unit_id"].values).tolist()
        observed_coef_ids = _resolve_legacy_units(
            raw_coefficient_ids,
            resolver=source_legacy_unit_identity_resolver,
            current_identity=source_identity,
            role="coefficient_source",
        )
        if observed_coef_ids != expected_coef_ids:
            raise ValueError(
                "Legacy full-fit coefficient units differ from reconstructed source variance."
            )
    else:
        raw_coefficient_ids = np.asarray(dataset["coef_ca1_unit_id"].values).tolist()
        observed_coef_ids = [
            str(value) for value in np.asarray(dataset["coef_ca1_unit_id"].values)
        ]
        if observed_coef_ids != ["-1"]:
            raise ValueError("Legacy mean-activity fit lacks its synthetic coefficient ID.")
    _validate_metric_arithmetic(dataset, parameters=parameters)
    coefficients = np.asarray(dataset["coef_ca1_full_all"].values, dtype=float)
    intercepts = np.asarray(dataset["coef_intercept_full_all"].values, dtype=float)
    if coefficients.shape != (
        dataset.sizes["coef_source_unit"],
        dataset.sizes["unit"],
    ) or intercepts.shape != (dataset.sizes["unit"],):
        raise ValueError("Legacy ripple GLM full-fit coefficient dimensions are invalid.")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(intercepts)):
        raise ValueError("Legacy ripple GLM coefficients must be finite.")
    expected_counts = {
        "n_ripples": n_samples,
        "n_units": len(expected_target_ids),
        "n_ca1_units": len(expected_source_ids),
    }
    for name, expected in expected_counts.items():
        if int(dataset.attrs.get(name, -1)) != expected:
            raise ValueError(f"Legacy ripple GLM has stale {name} metadata.")
    selection_counts = {
        "n_ripples_before_selection": inputs["n_ripples_before_selection"],
        "n_ripples_removed_by_selection": inputs[
            "n_ripples_removed_by_selection"
        ],
        "n_ripples_after_selection": inputs["n_ripples_after_selection"],
    }
    for name, expected in selection_counts.items():
        if int(dataset.attrs.get(name, -1)) != int(expected):
            raise ValueError(f"Legacy ripple GLM has stale {name} metadata.")
    return _rekey_legacy_dataset(
        dataset,
        source_group_ids=observed_source_ids,
        target_group_ids=observed_target_ids,
        coefficient_source_group_ids=(
            observed_coef_ids if parameters["source_predictor_mode"] == "unit_vector" else ["-1"]
        ),
        source_predictor_mode=parameters["source_predictor_mode"],
    )


def register_existing_ripple_glm_artifact(
    *,
    source_result_path: Path,
    destination_path: Path,
    ripple_glm_id: Any,
    animal_name: str,
    date: str,
    epoch: str,
    ripple_table: Any,
    epoch_interval: Any,
    source_spikes: Any,
    source_stable_unit_ids: Sequence[Mapping[str, Any]],
    target_spikes: Any,
    target_stable_unit_ids: Sequence[Mapping[str, Any]],
    source_sorting_type: str,
    target_sorting_type: str,
    source_legacy_unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    target_legacy_unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    upstream_provenance: Mapping[str, Any],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    expected_selected_ripple_events_sha256: str | None = None,
    source_v1ca1_git_commit: str | None = None,
    source_spyglass_git_commit: str | None = None,
    overwrite: bool = False,
    **parameter_values: Any,
) -> dict[str, Any]:
    """Strictly validate and register one existing legacy NetCDF artifact."""
    if str(source_sorting_type) != "ImportedSpikeSorting" or str(
        target_sorting_type
    ) != "ImportedSpikeSorting":
        raise ValueError(
            "Legacy RippleGLM registration requires ImportedSpikeSorting for "
            "both source and target groups."
        )
    source_path = Path(source_result_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy ripple GLM artifact not found: {source_path}")
    metadata = _metadata(
        ripple_glm_id=ripple_glm_id,
        animal_name=animal_name,
        date=date,
        epoch=epoch,
    )
    parameters = _effective_parameters(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        **parameter_values,
    )
    normalized_provenance, provenance_json = _validate_upstream_provenance(
        upstream_provenance, parameters=parameters
    )
    source_identity = _identity_table(
        source_spikes, source_stable_unit_ids, role=SOURCE_ROLE, region=SOURCE_REGION
    )
    target_identity = _identity_table(
        target_spikes, target_stable_unit_ids, role=TARGET_ROLE, region=TARGET_REGION
    )
    inputs = _select_and_count_inputs(
        epoch=metadata["epoch"],
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        source_spikes=source_spikes,
        target_spikes=target_spikes,
        parameters=parameters,
    )
    _require_expected_event_hash(inputs, expected_selected_ripple_events_sha256)
    terminal_status = _terminal_status(
        source_identity=source_identity,
        target_identity=target_identity,
        inputs=inputs,
        n_splits=parameters["n_splits"],
    )
    if terminal_status is not None:
        raise ValueError(
            "Cannot register a fitted legacy ripple GLM for terminal inputs: "
            f"{terminal_status}. Compute the explicit terminal artifact instead."
        )
    legacy_dataset = _normalize_legacy_dataset(
        _load_dataset(source_path), parameters=parameters
    )
    legacy_dataset = _validate_legacy_against_inputs(
        legacy_dataset,
        metadata=metadata,
        parameters=parameters,
        inputs=inputs,
        source_identity=source_identity,
        target_identity=target_identity,
        source_legacy_unit_identity_resolver=(
            source_legacy_unit_identity_resolver
        ),
        target_legacy_unit_identity_resolver=(
            target_legacy_unit_identity_resolver
        ),
    )
    selected_units = _initial_selected_units(
        source_identity, target_identity, inputs, parameters
    )
    legacy_provenance = {
        "source_result_path": str(source_path.resolve()),
        "source_result_sha256": _file_sha256(source_path),
        "source_schema_version": str(legacy_dataset.attrs.get("schema_version", "unknown")),
        "source_fit_parameters_sha256": _provenance_sha256(
            json.loads(str(legacy_dataset.attrs["fit_parameters_json"]))
        ),
        "source_v1ca1_git_commit": (
            "unknown" if source_v1ca1_git_commit is None else str(source_v1ca1_git_commit)
        ),
        "source_spyglass_git_commit": (
            "unknown" if source_spyglass_git_commit is None else str(source_spyglass_git_commit)
        ),
        "registration_validation": OUTPUT_RULE["legacy_registration_policy"],
    }
    dataset, quality = _canonicalize_dataset(
        legacy_dataset,
        metadata=metadata,
        parameters=parameters,
        inputs=inputs,
        source_identity=source_identity,
        target_identity=target_identity,
        upstream_provenance_json=provenance_json,
        artifact_origin="registered_existing",
        legacy_artifact_provenance=legacy_provenance,
    )
    selected_units = _annotate_selected_units(
        selected_units,
        dataset=dataset,
        parameters=parameters,
        quality=quality,
        terminal=False,
    )
    status = str(dataset.attrs["analysis_status"])
    result = validate_ripple_glm_result(
        _base_result(
            metadata=metadata,
            parameters=parameters,
            upstream_provenance=normalized_provenance,
            selected_units=selected_units,
            summary=_summary_table(
                dataset, metadata=metadata, selected_units=selected_units
            ),
            dataset=dataset,
            inputs=inputs,
            analysis_status=status,
            artifact_origin="registered_existing",
            legacy_artifact_provenance=legacy_provenance,
        )
    )
    paths = write_ripple_glm_artifact(
        result, Path(destination_path), overwrite=overwrite
    )
    return {**result, "artifact_paths": paths}


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_EXPECTED_DETECTOR_ZSCORE_THRESHOLD",
    "DEFAULT_N_SHUFFLES_RIPPLE",
    "DEFAULT_REQUIRE_SPEED_GATED",
    "MANIFEST_COLUMNS",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "RESULT_SCHEMA_VERSION",
    "SELECTED_UNIT_COLUMNS",
    "SUMMARY_COLUMNS",
    "compute_ripple_glm",
    "get_ripple_glm_artifact_paths",
    "load_ripple_glm_artifact",
    "prepare_ripple_glm_event_selection",
    "register_existing_ripple_glm_artifact",
    "validate_ripple_glm_parameters",
    "validate_ripple_glm_result",
    "write_ripple_glm_artifact",
]
