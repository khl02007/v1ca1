"""Database-free empirical swap-tuning comparison and artifact bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import TRAJECTORY_TYPES


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "swap_tuning_curve_comparison"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
SUMMARY_FILENAME = "summary.parquet"
RESULT_FILENAME = "swap_tuning.nc"
BUNDLE_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "3"
LEGACY_RESULT_SCHEMA_VERSION = "2"
NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_SELECTED_UNITS_TABLE_NAME = "swap_tuning_selected_units"
NWB_SCORE_SUMMARY_TABLE_NAME = "swap_tuning_score_summary"
NWB_SOURCE_PROFILES_TABLE_NAME = "swap_tuning_source_profiles"
NWB_MODEL_PROFILES_TABLE_NAME = "swap_tuning_model_profiles"
NWB_GEOMETRY_TABLE_NAME = "swap_tuning_geometry"
NWB_PROVENANCE_TABLE_NAME = "swap_tuning_provenance"

MODEL_NAMES = (
    "empirical_visual",
    "empirical_dark",
    "empirical_pointwise_multiplicative_ratio",
    "empirical_segment_multiplicative_ratio",
    "empirical_pointwise_additive_delta",
    "empirical_segment_additive_delta",
)
SWAP_CONFIGURATION = {
    "center_to_left": {"source_trajectory": "center_to_right", "segment_index": 2},
    "center_to_right": {"source_trajectory": "center_to_left", "segment_index": 2},
    "left_to_center": {"source_trajectory": "right_to_center", "segment_index": 0},
    "right_to_center": {"source_trajectory": "left_to_center", "segment_index": 0},
}
EMPIRICAL_MODEL_FORMULAS = {
    "empirical_visual": "other_light",
    "empirical_dark": "same_dark",
    "empirical_pointwise_multiplicative_ratio": (
        "same_dark * other_light / max(other_dark, epsilon)"
    ),
    "empirical_segment_multiplicative_ratio": (
        "same_dark * sum(other_light_in_swap_segment) / "
        "max(sum(other_dark_in_swap_segment), epsilon)"
    ),
    "empirical_pointwise_additive_delta": "same_dark + other_light - other_dark",
    "empirical_segment_additive_delta": (
        "same_dark + mean(other_light_in_swap_segment - "
        "other_dark_in_swap_segment)"
    ),
}
DEFAULT_EVALUATION_BIN_SIZE_S = 0.05
DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS = 1.0
DEFAULT_MIN_DARK_FIRING_RATE_HZ = 0.5
DEFAULT_MIN_LIGHT_FIRING_RATE_HZ = 0.5
REQUIRED_UPSTREAM_BIN_SIZE_CM = 4.0
REQUIRED_UPSTREAM_SIGMA_BINS = 0.0
LEGACY_POSITION_OFFSET_SAMPLES = 10
LEGACY_SPEED_THRESHOLD_CM_S = 4.0
EMPIRICAL_EPSILON = 1e-10

OUTPUT_RULE = {
    "version": 1,
    "models": MODEL_NAMES,
    "swap_configuration": SWAP_CONFIGURATION,
    "empirical_model_formulas": EMPIRICAL_MODEL_FORMULAS,
    "training_tuning_source": (
        "twelve_all_trial_path_specific_place_tuning_curves"
    ),
    "training_tuning_input_policy": (
        "four_cm_unsmoothed_curves_interpolate_nans_then_gaussian_smooth"
    ),
    "all_nan_tuning_fallback": (
        "trajectory_spike_count_divided_by_movement_support_duration"
    ),
    "evaluation_scope": "heldout_light_swapped_segment_movement_laps",
    "evaluation_bin_size_unit": "s",
    "eligibility_policy": (
        "strict_dark_and_light_train_epoch_wide_movement_firing_rate_thresholds"
    ),
    "heldout_firing_rate_filter": False,
    "trajectory_support_policy": (
        "all_or_none_terminal_if_any_heldout_path_has_no_scoring_bins"
    ),
    "movement_interval_provenance_policy": (
        "exact_artifact_sha256_frozen_for_dark_light_train_and_light_test"
    ),
    "unit_failure_policy": "retain_and_isolate_nonfinite_scores_per_unit",
    "unit_audit_policy": "retain_all_upstream_units_with_explicit_eligibility",
    "runtime_unit_key_policy": (
        "persistent_identity_aligned_native_tsgroup_keys_with_stable_output_ids"
    ),
    "legacy_registration_policy": (
        "imported_spike_sorting_only_exact_nwb_reconstruction_and_rescore"
    ),
    "legacy_preprocessing_policy": (
        "position_offset_10_and_speed_threshold_4_cm_s_required"
    ),
    "legacy_comparison_policy": "all_scientific_values_tight_equal",
    "time_unit": "s",
    "time_reference": "augmented_nwb_ephys_timestamps",
    "position_unit": "cm",
}

IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "source_unit_index",
    "eligible_unit_index",
    "dark_movement_firing_rate_hz",
    "light_train_movement_firing_rate_hz",
    "light_test_movement_firing_rate_hz",
    "passes_dark_firing_rate_threshold",
    "passes_light_firing_rate_threshold",
    "eligible_for_comparison",
    "test_light_spike_count",
    "n_finite_scores",
    "n_expected_scores",
    "valid_swap_tuning_score",
    "unit_qc_status",
)
UNIT_QC_STATUSES = (
    "excluded_dark_firing_rate",
    "excluded_light_firing_rate",
    "excluded_both_firing_rates",
    "not_computed",
    "valid",
    "zero_test_spikes",
    "partial_nonfinite_score",
    "no_valid_scores",
)
SCORE_QC_STATUSES = ("valid", "zero_test_spikes", "nonfinite_score")
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_units",
    "no_eligible_units",
    "upstream_terminal",
    "no_valid_position",
    "no_movement",
    "no_trajectory_samples",
    "no_valid_units",
)
SUMMARY_COLUMNS = (
    *IDENTITY_COLUMNS,
    "swap_tuning_curve_comparison_id",
    "animal_name",
    "date",
    "region",
    "dark_train_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "trajectory",
    "model",
    "evaluation_bin_size_s",
    "gaussian_smoothing_sigma_bins",
    "min_dark_firing_rate_hz",
    "min_light_firing_rate_hz",
    "dark_train_movement_firing_rate_hz",
    "light_train_movement_firing_rate_hz",
    "light_test_movement_firing_rate_hz",
    "swap_source_trajectory",
    "swap_segment_index_1based",
    "swap_segment_start",
    "swap_segment_end",
    "train_dark_same_rate_hz",
    "train_dark_other_rate_hz",
    "train_light_other_rate_hz",
    "test_light_target_rate_hz",
    "test_light_spike_sum",
    "test_light_bin_count",
    "test_light_duration_s",
    "ll_sum",
    "ll_bits_per_spike",
    "ll_bits_per_s",
    "score_qc_status",
    "unit_valid",
)
SCIENTIFIC_VARIABLE_DIMS = {
    "dark_train_movement_firing_rate_hz": ("unit",),
    "light_train_movement_firing_rate_hz": ("unit",),
    "light_test_movement_firing_rate_hz": ("unit",),
    "ll_sum": ("model", "trajectory", "unit"),
    "ll_bits_per_spike": ("model", "trajectory", "unit"),
    "ll_bits_per_s": ("model", "trajectory", "unit"),
    "model_tuning_hz": ("model", "trajectory", "tp_bin", "unit"),
    "train_dark_same_rate_hz": ("trajectory", "unit"),
    "train_dark_other_rate_hz": ("trajectory", "unit"),
    "train_light_other_rate_hz": ("trajectory", "unit"),
    "test_light_target_rate_hz": ("trajectory", "unit"),
    "test_light_spike_sum": ("trajectory", "unit"),
    "same_dark_train_tuning_hz": ("trajectory", "tp_bin", "unit"),
    "other_dark_train_tuning_hz": ("trajectory", "tp_bin", "unit"),
    "other_light_train_tuning_hz": ("trajectory", "tp_bin", "unit"),
    "test_light_tuning_hz": ("trajectory", "tp_bin", "unit"),
    "segment_bin_mask": ("trajectory", "tp_bin"),
    "swap_source_trajectory": ("trajectory",),
    "swap_segment_index_1based": ("trajectory",),
    "swap_segment_start": ("trajectory",),
    "swap_segment_end": ("trajectory",),
    "test_light_bin_count": ("trajectory",),
    "test_light_duration_s": ("trajectory",),
    "segment_edges": ("segment_edge",),
}
SOURCE_PROFILE_COLUMNS = (
    "trajectory",
    *IDENTITY_COLUMNS,
    "same_dark_train_tuning_hz",
    "other_dark_train_tuning_hz",
    "other_light_train_tuning_hz",
    "test_light_tuning_hz",
)
SOURCE_PROFILE_VECTOR_COLUMNS = SOURCE_PROFILE_COLUMNS[-4:]
MODEL_PROFILE_COLUMNS = (
    "model",
    "trajectory",
    *IDENTITY_COLUMNS,
    "model_tuning_hz",
)
GEOMETRY_COLUMNS = (
    "trajectory",
    "swap_source_trajectory",
    "swap_segment_index_1based",
    "swap_segment_start",
    "swap_segment_end",
    "test_light_bin_count",
    "test_light_duration_s",
    "tp_bin",
    "segment_edges",
    "segment_bin_mask",
)
GEOMETRY_VECTOR_COLUMNS = GEOMETRY_COLUMNS[-3:]
PROVENANCE_COLUMNS = (
    "metadata_json",
    "parameters_json",
    "upstream_provenance_json",
    "dataset_attrs_json",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "artifact_schema_version",
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "file_size_bytes",
    "sha256",
    "swap_tuning_curve_comparison_id",
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "upstream_provenance_json",
    "n_source_units",
    "n_units",
    "n_valid_units",
    "selected_units_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "bundle_schema_version",
)


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


def get_swap_tuning_curve_comparison_artifact_paths(
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    swap_tuning_curve_comparison_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first empirical swap bundle."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "dark_epoch": dark_epoch,
            "light_train_epoch": light_train_epoch,
            "light_test_epoch": light_test_epoch,
        }.items()
    }
    result_id = _uuid_string(
        swap_tuning_curve_comparison_id,
        name="swap_tuning_curve_comparison_id",
    )
    light_pair = (
        f"{components['light_train_epoch']}_train_to_"
        f"{components['light_test_epoch']}_test"
    )
    artifact_dir = (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / light_pair
        / f"dark_{components['dark_epoch']}"
        / components["region"]
        / result_id
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "selected_units_path": artifact_dir / SELECTED_UNITS_FILENAME,
        "summary_path": artifact_dir / SUMMARY_FILENAME,
        "result_path": artifact_dir / RESULT_FILENAME,
    }


def validate_swap_tuning_curve_comparison_parameters(
    *,
    evaluation_bin_size_s: float = DEFAULT_EVALUATION_BIN_SIZE_S,
    gaussian_smoothing_sigma_bins: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
    min_dark_firing_rate_hz: float = DEFAULT_MIN_DARK_FIRING_RATE_HZ,
    min_light_firing_rate_hz: float = DEFAULT_MIN_LIGHT_FIRING_RATE_HZ,
) -> dict[str, float]:
    """Return validated empirical swap-tuning parameters."""
    values = {
        "evaluation_bin_size_s": float(evaluation_bin_size_s),
        "gaussian_smoothing_sigma_bins": float(gaussian_smoothing_sigma_bins),
        "min_dark_firing_rate_hz": float(min_dark_firing_rate_hz),
        "min_light_firing_rate_hz": float(min_light_firing_rate_hz),
    }
    if not np.isfinite(values["evaluation_bin_size_s"]) or values[
        "evaluation_bin_size_s"
    ] <= 0.0:
        raise ValueError("evaluation_bin_size_s must be positive and finite.")
    for name in (
        "gaussian_smoothing_sigma_bins",
        "min_dark_firing_rate_hz",
        "min_light_firing_rate_hz",
    ):
        if not np.isfinite(values[name]) or values[name] < 0.0:
            raise ValueError(f"{name} must be non-negative and finite.")
    return values


def _effective_parameters(
    *,
    parameter_name: str,
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
    evaluation_bin_size_s: float,
    gaussian_smoothing_sigma_bins: float,
    min_dark_firing_rate_hz: float,
    min_light_firing_rate_hz: float,
) -> dict[str, Any]:
    """Validate parameters and their deterministic provenance hashes."""
    values = validate_swap_tuning_curve_comparison_parameters(
        evaluation_bin_size_s=evaluation_bin_size_s,
        gaussian_smoothing_sigma_bins=gaussian_smoothing_sigma_bins,
        min_dark_firing_rate_hz=min_dark_firing_rate_hz,
        min_light_firing_rate_hz=min_light_firing_rate_hz,
    )
    parameter_name = _path_component(parameter_name, name="parameter_name")
    payload = {
        "swap_tuning_curve_comparison_param_name": parameter_name,
        **values,
    }
    expected_parameter_sha256 = _provenance_sha256(payload)
    if parameter_sha256 is None:
        parameter_sha256 = expected_parameter_sha256
    if str(parameter_sha256) != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match effective parameters.")
    if output_rule_sha256 is None:
        output_rule_sha256 = OUTPUT_RULE_SHA256
    if str(output_rule_sha256) != OUTPUT_RULE_SHA256:
        raise ValueError("output_rule_sha256 does not match the fixed output rule.")
    return {
        "parameter_name": parameter_name,
        "parameter_sha256": str(parameter_sha256),
        "output_rule_sha256": str(output_rule_sha256),
        **values,
    }


def _metadata(
    *,
    swap_tuning_curve_comparison_id: Any,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
) -> dict[str, str]:
    """Return validated metadata for one empirical swap comparison."""
    metadata = {
        "swap_tuning_curve_comparison_id": _uuid_string(
            swap_tuning_curve_comparison_id,
            name="swap_tuning_curve_comparison_id",
        ),
        **{
            name: _path_component(value, name=name)
            for name, value in {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "dark_epoch": dark_epoch,
                "light_train_epoch": light_train_epoch,
                "light_test_epoch": light_test_epoch,
            }.items()
        },
    }
    epochs = (
        metadata["dark_epoch"],
        metadata["light_train_epoch"],
        metadata["light_test_epoch"],
    )
    if len(set(epochs)) != 3:
        raise ValueError("Dark, light-train, and light-test epochs must be distinct.")
    return metadata


def _file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_sha256(table: pd.DataFrame) -> str:
    """Return a deterministic digest for one validated tabular input."""
    hashed = pd.util.hash_pandas_object(table, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _role_epochs(metadata: Mapping[str, str]) -> dict[str, str]:
    """Return the fixed analysis-role to concrete epoch mapping."""
    return {
        "dark": str(metadata["dark_epoch"]),
        "light_train": str(metadata["light_train_epoch"]),
        "light_test": str(metadata["light_test_epoch"]),
    }


def _normalize_nested_role_mapping(
    values: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> dict[str, dict[str, Any]]:
    """Return one exact three-role by four-trajectory mapping."""
    expected_roles = {"dark", "light_train", "light_test"}
    if set(values) != expected_roles:
        raise ValueError(f"{name} must contain exactly {sorted(expected_roles)!r}.")
    normalized = {}
    for role in ("dark", "light_train", "light_test"):
        current = dict(values[role])
        if set(current) != set(TRAJECTORY_TYPES):
            raise ValueError(f"{name}[{role!r}] must contain exactly four paths.")
        normalized[role] = {
            trajectory: current[trajectory] for trajectory in TRAJECTORY_TYPES
        }
    return normalized


def _curve_identity_table(curve: Any) -> pd.DataFrame:
    """Return persistent identities in one PathSpecificPlace curve's order."""
    from v1ca1.spyglass.path_specific_place import IDENTITY_COORDINATES

    return pd.DataFrame(
        {
            name: np.asarray(curve.coords[name].values).astype(str)
            for name in IDENTITY_COORDINATES
        }
    ).loc[:, list(IDENTITY_COLUMNS)]


def _load_and_validate_tuning_inputs(
    *,
    tuning_curve_artifact_paths: Mapping[str, Mapping[str, Path]] | None,
    tuning_curves_by_role_trajectory: Mapping[str, Mapping[str, Any]] | None,
    metadata: Mapping[str, str],
    graph_length_cm: float,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, dict[str, str]]]:
    """Load the twelve exact tuning inputs from paths or fetched NWB objects."""
    from v1ca1.spyglass.path_specific_place import (
        PATH_FRACTION_COORDINATE,
        POSITION_DIM,
        load_path_specific_place_artifact,
        path_specific_place_tuning_curve_sha256,
        validate_path_specific_place_tuning_curve,
    )

    if (tuning_curve_artifact_paths is None) == (
        tuning_curves_by_role_trajectory is None
    ):
        raise ValueError(
            "Provide exactly one of tuning_curve_artifact_paths and "
            "tuning_curves_by_role_trajectory."
        )
    paths = (
        None
        if tuning_curve_artifact_paths is None
        else _normalize_nested_role_mapping(
            tuning_curve_artifact_paths,
            name="tuning_curve_artifact_paths",
        )
    )
    supplied_curves = (
        None
        if tuning_curves_by_role_trajectory is None
        else _normalize_nested_role_mapping(
            tuning_curves_by_role_trajectory,
            name="tuning_curves_by_role_trajectory",
        )
    )
    role_epochs = _role_epochs(metadata)
    curves: dict[str, dict[str, Any]] = {}
    hashes: dict[str, dict[str, str]] = {}
    reference_identity: pd.DataFrame | None = None
    reference_position: np.ndarray | None = None
    reference_fraction: np.ndarray | None = None
    reference_edges: np.ndarray | None = None
    for role in ("dark", "light_train", "light_test"):
        curves[role] = {}
        hashes[role] = {}
        for trajectory in TRAJECTORY_TYPES:
            path = None if paths is None else Path(paths[role][trajectory])
            curve = (
                validate_path_specific_place_tuning_curve(
                    supplied_curves[role][trajectory]
                )
                if supplied_curves is not None
                else load_path_specific_place_artifact(path)
            )
            expected_attrs = {
                "animal_name": metadata["animal_name"],
                "date": metadata["date"],
                "region": metadata["region"],
                "epoch": role_epochs[role],
                "trajectory_type": trajectory,
                "trial_subset": "all",
                "binning_mode": "bin_size_cm",
            }
            for field, expected in expected_attrs.items():
                if str(curve.attrs.get(field, "")) != str(expected):
                    raise ValueError(
                        f"Tuning input {role}/{trajectory} has mismatched {field!r}."
                    )
            if not np.isclose(
                float(curve.attrs.get("bin_size_cm", np.nan)),
                REQUIRED_UPSTREAM_BIN_SIZE_CM,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Swap tuning requires 4-cm PathSpecificPlace inputs.")
            if not np.isclose(
                float(curve.attrs.get("sigma_bins", np.nan)),
                REQUIRED_UPSTREAM_SIGMA_BINS,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Swap tuning requires unsmoothed PathSpecificPlace inputs.")
            if not np.isclose(
                float(curve.attrs.get("graph_length_cm", np.nan)),
                float(graph_length_cm),
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError("Tuning input graph length differs from WTrackGraph.")

            identity = _curve_identity_table(curve)
            position = np.asarray(curve.coords[POSITION_DIM].values, dtype=float)
            fraction = np.asarray(
                curve.coords[PATH_FRACTION_COORDINATE].values,
                dtype=float,
            )
            try:
                edges = np.asarray(
                    json.loads(str(curve.attrs["bin_edges_cm_json"])),
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("Tuning input has invalid centimeter bin edges.") from exc
            if reference_identity is None:
                reference_identity = identity
                reference_position = position
                reference_fraction = fraction
                reference_edges = edges
            elif not reference_identity.equals(identity):
                raise ValueError("The twelve tuning inputs disagree on unit identities/order.")
            elif not (
                np.array_equal(reference_position, position)
                and np.array_equal(reference_fraction, fraction)
                and np.array_equal(reference_edges, edges)
            ):
                raise ValueError("The twelve tuning inputs must use the same exact grid.")
            curves[role][trajectory] = curve
            hashes[role][trajectory] = (
                path_specific_place_tuning_curve_sha256(curve)
                if path is None
                else _file_sha256(path)
            )
    if reference_identity is None:
        raise ValueError("No PathSpecificPlace tuning inputs were supplied.")
    return curves, reference_identity, hashes


def _validate_movement_table_context(
    table: pd.DataFrame,
    *,
    metadata: Mapping[str, str],
    epoch: str,
) -> pd.DataFrame:
    """Validate one movement table and require the selected session context."""
    from v1ca1.spyglass.movement import validate_movement_firing_rate_table

    table = validate_movement_firing_rate_table(table).copy()
    if table.empty:
        return table
    for field, expected in {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "epoch": epoch,
    }.items():
        values = table[field].astype(str).unique().tolist()
        if values != [str(expected)]:
            raise ValueError(f"Movement input has mismatched {field!r}.")
    return table


def _selected_units_sha256(selected_units: pd.DataFrame) -> str:
    """Return the canonical all-source-unit identity digest."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    return unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )


def _align_unit_inputs(
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    curve_identity: pd.DataFrame,
    movement_tables_by_role: Mapping[str, pd.DataFrame],
    parameters: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, str]:
    """Align identities/rates and build an all-source-unit eligibility audit."""
    from v1ca1.spyglass.movement import align_movement_firing_rates
    from v1ca1.spyglass.path_specific_place import _identity_rows

    identities = pd.DataFrame.from_records(
        _identity_rows(spikes, stable_unit_ids),
        columns=(*IDENTITY_COLUMNS, "_group_key"),
    )
    comparable_identity = identities.loc[:, list(IDENTITY_COLUMNS)].astype(str)
    if not comparable_identity.equals(curve_identity.astype(str)):
        raise ValueError(
            "Held-out RegionSortedSpikesGroup and tuning inputs disagree on "
            "persistent identities/order."
        )
    rates = {
        role: align_movement_firing_rates(
            table,
            spikes=spikes,
            stable_unit_ids=stable_unit_ids,
        ).to_numpy(dtype=float)
        for role, table in movement_tables_by_role.items()
    }
    n_units = len(identities)
    statuses = {
        role: (
            "no_units"
            if table.empty
            else str(table["firing_rate_status"].iloc[0])
        )
        for role, table in movement_tables_by_role.items()
    }
    if n_units == 0:
        return (
            pd.DataFrame(columns=SELECTED_UNIT_COLUMNS),
            np.asarray([], dtype=bool),
            np.asarray([], dtype=object),
            "no_units",
        )
    if any(value == "no_valid_position" for value in statuses.values()):
        input_status = "no_valid_position"
    elif any(value == "no_movement" for value in statuses.values()):
        input_status = "no_movement"
    elif any(value == "no_units" for value in statuses.values()):
        raise ValueError("A non-empty source group cannot have no_units movement input.")
    else:
        input_status = "valid"

    dark_pass = np.isfinite(rates["dark"]) & (
        rates["dark"] > float(parameters["min_dark_firing_rate_hz"])
    )
    light_pass = np.isfinite(rates["light_train"]) & (
        rates["light_train"] > float(parameters["min_light_firing_rate_hz"])
    )
    eligible = dark_pass & light_pass & (input_status == "valid")
    eligible_indices = np.full(n_units, -1, dtype=np.int64)
    eligible_indices[eligible] = np.arange(int(np.sum(eligible)), dtype=np.int64)
    qc = np.full(n_units, "not_computed", dtype=object)
    qc[~dark_pass & light_pass] = "excluded_dark_firing_rate"
    qc[dark_pass & ~light_pass] = "excluded_light_firing_rate"
    qc[~dark_pass & ~light_pass] = "excluded_both_firing_rates"
    records = []
    for index, row in identities.iterrows():
        records.append(
            {
                **{name: str(row[name]) for name in IDENTITY_COLUMNS},
                "source_unit_index": int(index),
                "eligible_unit_index": int(eligible_indices[index]),
                "dark_movement_firing_rate_hz": float(rates["dark"][index]),
                "light_train_movement_firing_rate_hz": float(
                    rates["light_train"][index]
                ),
                "light_test_movement_firing_rate_hz": float(
                    rates["light_test"][index]
                ),
                "passes_dark_firing_rate_threshold": bool(dark_pass[index]),
                "passes_light_firing_rate_threshold": bool(light_pass[index]),
                "eligible_for_comparison": bool(eligible[index]),
                "test_light_spike_count": 0.0,
                "n_finite_scores": 0,
                "n_expected_scores": len(MODEL_NAMES) * len(TRAJECTORY_TYPES),
                "valid_swap_tuning_score": False,
                "unit_qc_status": str(qc[index]),
            }
        )
    audit = pd.DataFrame.from_records(records, columns=SELECTED_UNIT_COLUMNS)
    native_keys = identities.loc[eligible, "_group_key"].to_numpy(dtype=object)
    if input_status != "valid":
        return audit, eligible, native_keys, input_status
    return (
        audit,
        eligible,
        native_keys,
        "valid" if np.any(eligible) else "no_eligible_units",
    )


def _interval_duration_s(intervals: Any) -> float:
    """Return validated total interval duration in seconds."""
    from v1ca1.spyglass.movement import movement_interval_summary

    return float(movement_interval_summary(intervals)[1])


def _has_valid_position_samples(position: Any) -> bool:
    """Return whether position has at least two finite timed samples."""
    try:
        times = np.asarray(position.t, dtype=float).reshape(-1)
        values = np.asarray(position.d, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return False
    if values.shape[0] != times.size:
        return False
    finite_values = (
        np.isfinite(values)
        if values.ndim == 1
        else np.all(np.isfinite(values.reshape(times.size, -1)), axis=1)
    )
    return bool(np.sum(np.isfinite(times) & finite_values) >= 2)


def _select_spikes(spikes: Any, mask: np.ndarray) -> Any:
    """Return selected TsGroup units in source order."""
    return spikes[np.asarray(mask, dtype=bool)]


def _derive_task_progression(
    *,
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Linearize held-out position for each selected trajectory graph."""
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    return {
        trajectory: build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals[trajectory],
            graph_inputs=graph_inputs_by_trajectory[trajectory],
            trajectory_type=trajectory,
        )[0]
        for trajectory in TRAJECTORY_TYPES
    }


def _analysis_module() -> Any:
    """Import the existing empirical analysis implementation lazily."""
    from v1ca1.task_progression import swap_tuning_curve_comparison

    return swap_tuning_curve_comparison


def _validate_analysis_contract(analysis: Any) -> None:
    """Fail if the fixed pipeline semantics drift from the reused analysis."""
    if tuple(getattr(analysis, "EMPIRICAL_MODEL_NAMES", ())) != MODEL_NAMES:
        raise ValueError("Empirical model order differs from the source analysis.")
    if dict(getattr(analysis, "SWAP_CONFIG", {})) != SWAP_CONFIGURATION:
        raise ValueError("Trajectory swap configuration differs from source analysis.")


def _speed_threshold_from_tables(
    movement_tables_by_role: Mapping[str, pd.DataFrame],
) -> float:
    """Return one speed threshold shared by all movement inputs."""
    thresholds = []
    for table in movement_tables_by_role.values():
        if table.empty:
            continue
        values = pd.to_numeric(
            table["speed_threshold_cm_s"],
            errors="coerce",
        ).drop_duplicates()
        if len(values) != 1 or not np.isfinite(float(values.iloc[0])):
            raise ValueError("Each movement input must have one finite speed threshold.")
        thresholds.append(float(values.iloc[0]))
    if not thresholds:
        return float("nan")
    if not np.allclose(thresholds, thresholds[0], rtol=0.0, atol=1e-12):
        raise ValueError("The three movement inputs must use one speed threshold.")
    return float(thresholds[0])


def _validate_position_offset(value: int) -> int:
    """Return one non-negative integer position offset."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError("position_offset_samples must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError("position_offset_samples must be non-negative.")
    return value


def _optional_nested_strings(
    values: Mapping[str, Mapping[str, Any]] | None,
    *,
    name: str,
) -> dict[str, dict[str, str]]:
    """Validate an optional complete role/trajectory provenance mapping."""
    if values is None:
        return {}
    normalized = _normalize_nested_role_mapping(values, name=name)
    output = {
        role: {
            trajectory: str(normalized[role][trajectory])
            for trajectory in TRAJECTORY_TYPES
        }
        for role in ("dark", "light_train", "light_test")
    }
    if any(not value for role in output.values() for value in role.values()):
        raise ValueError(f"{name} values must be non-empty.")
    return output


def _optional_role_strings(
    values: Mapping[str, Any] | None,
    *,
    name: str,
) -> dict[str, str]:
    """Validate an optional complete three-role provenance mapping."""
    if values is None:
        return {}
    expected = {"dark", "light_train", "light_test"}
    if set(values) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)!r}.")
    output = {role: str(values[role]) for role in sorted(expected)}
    if any(not value for value in output.values()):
        raise ValueError(f"{name} values must be non-empty.")
    return output


def _is_sha256(value: Any) -> bool:
    """Return whether one value is a 64-character hexadecimal digest."""
    token = str(value)
    return len(token) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in token
    )


def _validate_role_sha256_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, str]:
    """Return one complete three-role mapping of SHA-256 digests."""
    output = _optional_role_strings(values, name=name)
    if not output or any(not _is_sha256(value) for value in output.values()):
        raise ValueError(f"{name} must contain one SHA-256 digest per role.")
    return output


def _validate_nested_sha256_mapping(
    values: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> dict[str, dict[str, str]]:
    """Return one complete role/trajectory mapping of SHA-256 digests."""
    output = _optional_nested_strings(values, name=name)
    if not output or any(
        not _is_sha256(value)
        for role_values in output.values()
        for value in role_values.values()
    ):
        raise ValueError(
            f"{name} must contain one SHA-256 digest per role/trajectory."
        )
    return output


def movement_firing_rate_table_content_sha256(table: pd.DataFrame) -> str:
    """Return the semantic hash used when no source-file hash is supplied."""
    from v1ca1.spyglass.movement import validate_movement_firing_rate_table

    return _table_sha256(validate_movement_firing_rate_table(table))


def _terminal_dataset(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
    selected_units: pd.DataFrame,
    segment_edges: np.ndarray,
    analysis_status: str,
    terminal_detail: Mapping[str, Any] | None = None,
) -> Any:
    """Return one persistable terminal NetCDF marker."""
    import xarray as xr

    eligible = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool)
    ]
    dataset = xr.Dataset(
        coords={
            "model": np.asarray(MODEL_NAMES, dtype=str),
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "unit": np.asarray(eligible["stable_unit_id"], dtype=str),
            "segment_edge": np.arange(len(segment_edges), dtype=np.int64),
            "spikesorting_merge_id": (
                "unit",
                np.asarray(eligible["spikesorting_merge_id"], dtype=str),
            ),
            "unit_id": ("unit", np.asarray(eligible["unit_id"], dtype=str)),
            "stable_unit_id": (
                "unit",
                np.asarray(eligible["stable_unit_id"], dtype=str),
            ),
            "group_unit_id": (
                "unit",
                np.asarray(eligible["group_unit_id"], dtype=str),
            ),
        },
        attrs={
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "analysis_status": analysis_status,
            "terminal_reason": analysis_status,
            "fit_stage": "terminal",
            **{
                name: metadata[name]
                for name in (
                    "animal_name",
                    "date",
                    "region",
                    "dark_epoch",
                    "light_train_epoch",
                    "light_test_epoch",
                )
            },
            "dark_train_epoch": metadata["dark_epoch"],
            **{
                name: parameters[name]
                for name in (
                    "parameter_name",
                    "parameter_sha256",
                    "output_rule_sha256",
                    "evaluation_bin_size_s",
                    "gaussian_smoothing_sigma_bins",
                    "min_dark_firing_rate_hz",
                    "min_light_firing_rate_hz",
                )
            },
            "upstream_provenance_json": json.dumps(
                dict(upstream_provenance),
                sort_keys=True,
            ),
            "segment_edges_json": json.dumps(
                np.asarray(segment_edges, dtype=float).tolist()
            ),
        },
    )
    dataset.attrs.update(dict(terminal_detail or {}))
    return dataset


def _trajectory_rate_from_curve(curve: Any, eligible: np.ndarray) -> np.ndarray:
    """Return legacy fallback rates from curve spike counts and support."""
    counts = np.asarray(curve.coords["spike_count"].values, dtype=float)[eligible]
    duration = float(curve.attrs["support_duration_s"])
    if not np.isfinite(duration) or duration <= 0.0:
        return np.full(counts.shape, np.nan, dtype=float)
    return counts / duration


def _prepare_tuning_inputs(
    curves: Mapping[str, Mapping[str, Any]],
    *,
    eligible: np.ndarray,
    sigma_bins: float,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
]:
    """Interpolate and smooth all twelve upstream tuning matrices."""
    analysis = _analysis_module()
    tunings: dict[str, dict[str, np.ndarray]] = {}
    rates: dict[str, dict[str, np.ndarray]] = {}
    for role in ("dark", "light_train", "light_test"):
        tunings[role] = {}
        rates[role] = {}
        for trajectory in TRAJECTORY_TYPES:
            curve = curves[role][trajectory]
            fallback = _trajectory_rate_from_curve(curve, eligible)
            rates[role][trajectory] = fallback
            matrix = np.asarray(curve.values, dtype=float)[eligible].T
            tunings[role][trajectory] = analysis.smooth_interpolated_tuning_matrix(
                matrix,
                fallback_rates_hz=np.nan_to_num(fallback, nan=0.0),
                sigma_bins=float(sigma_bins),
            )
    return tunings, rates


def _compute_empirical_arrays(
    *,
    curves: Mapping[str, Mapping[str, Any]],
    eligible: np.ndarray,
    spikes: Any,
    native_unit_keys: np.ndarray,
    task_progression_by_trajectory: Mapping[str, Any],
    trajectory_intervals: Mapping[str, Any],
    movement_interval: Any,
    bin_edges: np.ndarray,
    segment_edges: np.ndarray,
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Reuse empirical formulas/scoring and return source-compatible arrays."""
    analysis = _analysis_module()
    tunings, rates = _prepare_tuning_inputs(
        curves,
        eligible=eligible,
        sigma_bins=float(parameters["gaussian_smoothing_sigma_bins"]),
    )
    selected_spikes = _select_spikes(spikes, eligible)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_units = int(np.sum(eligible))
    n_trajectories = len(TRAJECTORY_TYPES)
    metrics = {
        name: np.full((len(MODEL_NAMES), n_trajectories, n_units), np.nan)
        for name in ("ll_sum", "ll_bits_per_spike", "ll_bits_per_s")
    }
    model_tuning = np.full(
        (len(MODEL_NAMES), n_trajectories, len(bin_centers), n_units),
        np.nan,
    )
    test_spikes = np.full((n_trajectories, n_units), np.nan)
    test_bin_count = np.zeros(n_trajectories, dtype=float)
    test_duration = np.zeros(n_trajectories, dtype=float)
    segment_masks = np.zeros((n_trajectories, len(bin_centers)), dtype=bool)
    swap_source = np.empty(n_trajectories, dtype=object)
    swap_segment_index = np.zeros(n_trajectories, dtype=np.int64)

    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        config = SWAP_CONFIGURATION[trajectory]
        source = str(config["source_trajectory"])
        segment_index = int(config["segment_index"])
        swap_source[trajectory_index] = source
        swap_segment_index[trajectory_index] = segment_index
        segment_masks[trajectory_index] = analysis.segment_mask(
            bin_centers,
            segment_edges,
            segment_index,
        )
        predictions = analysis.build_empirical_swap_tunings(
            tunings["dark"][trajectory],
            tunings["light_train"][source],
            tunings["dark"][source],
            bin_centers,
            segment_edges,
            segment_index,
            epsilon=EMPIRICAL_EPSILON,
        )
        for model_index, model_name in enumerate(MODEL_NAMES):
            model_tuning[model_index, trajectory_index] = predictions[model_name]
        support = trajectory_intervals[trajectory].intersect(movement_interval)
        if _interval_duration_s(support) <= 0.0:
            return None, trajectory
        score = analysis.score_tuning_curves_on_segment(
            spikes=selected_spikes,
            task_progression=task_progression_by_trajectory[trajectory],
            epoch=support,
            tunings_by_model=predictions,
            bin_edges=bin_edges,
            segment_edges=segment_edges,
            segment_index=segment_index,
            bin_size_s=float(parameters["evaluation_bin_size_s"]),
            unit_ids=native_unit_keys,
        )
        if float(score["test_light_bin_count"]) <= 0.0:
            return None, trajectory
        for name in metrics:
            metrics[name][:, trajectory_index] = np.asarray(score[name], dtype=float)
        test_spikes[trajectory_index] = np.asarray(
            score["test_light_spike_sum"],
            dtype=float,
        )
        test_bin_count[trajectory_index] = float(score["test_light_bin_count"])
        test_duration[trajectory_index] = float(score["test_light_duration_s"])

    return {
        "unit_ids": np.asarray(native_unit_keys),
        "model_names": np.asarray(MODEL_NAMES, dtype=str),
        "bin_edges": np.asarray(bin_edges, dtype=float),
        "bin_centers": bin_centers,
        "segment_edges": np.asarray(segment_edges, dtype=float),
        "swap_source_trajectory": swap_source,
        "swap_segment_index": swap_segment_index,
        "segment_bin_mask": segment_masks,
        "model_tuning": model_tuning,
        "same_dark_tuning": np.stack(
            [tunings["dark"][trajectory] for trajectory in TRAJECTORY_TYPES]
        ),
        "other_dark_tuning": np.stack(
            [
                tunings["dark"][SWAP_CONFIGURATION[trajectory]["source_trajectory"]]
                for trajectory in TRAJECTORY_TYPES
            ]
        ),
        "other_light_tuning": np.stack(
            [
                tunings["light_train"][
                    SWAP_CONFIGURATION[trajectory]["source_trajectory"]
                ]
                for trajectory in TRAJECTORY_TYPES
            ]
        ),
        "test_light_tuning": np.stack(
            [tunings["light_test"][trajectory] for trajectory in TRAJECTORY_TYPES]
        ),
        "train_dark_same_rate_hz": np.stack(
            [rates["dark"][trajectory] for trajectory in TRAJECTORY_TYPES]
        ),
        "train_dark_other_rate_hz": np.stack(
            [
                rates["dark"][SWAP_CONFIGURATION[trajectory]["source_trajectory"]]
                for trajectory in TRAJECTORY_TYPES
            ]
        ),
        "train_light_other_rate_hz": np.stack(
            [
                rates["light_train"][
                    SWAP_CONFIGURATION[trajectory]["source_trajectory"]
                ]
                for trajectory in TRAJECTORY_TYPES
            ]
        ),
        "test_light_target_rate_hz": np.stack(
            [rates["light_test"][trajectory] for trajectory in TRAJECTORY_TYPES]
        ),
        "test_light_spike_sum": test_spikes,
        "test_light_bin_count": test_bin_count,
        "test_light_duration_s": test_duration,
        "metrics": metrics,
    }, None


def empty_swap_tuning_summary() -> pd.DataFrame:
    """Return an empty long-form summary with the canonical columns."""
    return pd.DataFrame(columns=SUMMARY_COLUMNS)


def _build_dataset(
    *,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
    selected_units: pd.DataFrame,
    source_payload: Mapping[str, Any],
) -> Any:
    """Build the complete legacy-compatible scientific NetCDF schema."""
    analysis = _analysis_module()
    eligible = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool)
    ].reset_index(drop=True)
    dataset = analysis.build_region_dataset(
        dict(arrays),
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        dark_train_epoch=metadata["dark_epoch"],
        light_train_epoch=metadata["light_train_epoch"],
        light_test_epoch=metadata["light_test_epoch"],
        dark_movement_firing_rates=eligible[
            "dark_movement_firing_rate_hz"
        ].to_numpy(dtype=float),
        light_movement_firing_rates=eligible[
            "light_train_movement_firing_rate_hz"
        ].to_numpy(dtype=float),
        bin_size_s=float(parameters["evaluation_bin_size_s"]),
        sigma_bins=float(parameters["gaussian_smoothing_sigma_bins"]),
        place_bin_size_cm=REQUIRED_UPSTREAM_BIN_SIZE_CM,
        apply_fr_filter=True,
        min_dark_fr_hz=float(parameters["min_dark_firing_rate_hz"]),
        min_light_fr_hz=float(parameters["min_light_firing_rate_hz"]),
        sources=dict(source_payload),
    )
    dataset["light_test_movement_firing_rate_hz"] = (
        "unit",
        eligible["light_test_movement_firing_rate_hz"].to_numpy(dtype=float),
    )
    dataset = dataset.assign_coords(
        unit=np.asarray(eligible["stable_unit_id"], dtype=str),
        spikesorting_merge_id=(
            "unit",
            np.asarray(eligible["spikesorting_merge_id"], dtype=str),
        ),
        unit_id=("unit", np.asarray(eligible["unit_id"], dtype=str)),
        stable_unit_id=(
            "unit",
            np.asarray(eligible["stable_unit_id"], dtype=str),
        ),
        group_unit_id=(
            "unit",
            np.asarray(eligible["group_unit_id"], dtype=str),
        ),
    )
    dataset.attrs.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "analysis_status": "valid",
            "fit_stage": "evaluated",
            "dark_epoch": metadata["dark_epoch"],
            "parameter_name": parameters["parameter_name"],
            "parameter_sha256": parameters["parameter_sha256"],
            "output_rule_sha256": parameters["output_rule_sha256"],
            "evaluation_bin_size_s": parameters["evaluation_bin_size_s"],
            "gaussian_smoothing_sigma_bins": parameters[
                "gaussian_smoothing_sigma_bins"
            ],
            "min_dark_firing_rate_hz": parameters["min_dark_firing_rate_hz"],
            "min_light_firing_rate_hz": parameters["min_light_firing_rate_hz"],
            "empirical_epsilon": EMPIRICAL_EPSILON,
            "upstream_provenance_json": json.dumps(
                dict(upstream_provenance),
                sort_keys=True,
            ),
        }
    )
    return dataset


def _audit_unit_scores(
    selected_units: pd.DataFrame,
    dataset: Any,
) -> tuple[pd.DataFrame, str]:
    """Attach isolated per-unit score QC to the all-source-unit audit."""
    audit = selected_units.copy()
    eligible_rows = audit["eligible_for_comparison"].astype(bool).to_numpy()
    if not np.any(eligible_rows):
        return audit, "no_eligible_units" if len(audit) else "no_units"
    values = np.asarray(dataset["ll_bits_per_spike"].values, dtype=float)
    expected_shape = (
        len(MODEL_NAMES),
        len(TRAJECTORY_TYPES),
        int(np.sum(eligible_rows)),
    )
    if values.shape != expected_shape:
        raise ValueError("Empirical score array has an unexpected shape.")
    finite_counts = np.sum(np.isfinite(values), axis=(0, 1)).astype(np.int64)
    expected_count = len(MODEL_NAMES) * len(TRAJECTORY_TYPES)
    spike_counts = np.sum(
        np.asarray(dataset["test_light_spike_sum"].values, dtype=float),
        axis=0,
    )
    audit.loc[eligible_rows, "test_light_spike_count"] = spike_counts
    audit.loc[eligible_rows, "n_finite_scores"] = finite_counts
    valid = finite_counts == expected_count
    audit.loc[eligible_rows, "valid_swap_tuning_score"] = valid
    qc = np.full(valid.shape, "partial_nonfinite_score", dtype=object)
    qc[finite_counts == 0] = "no_valid_scores"
    qc[spike_counts <= 0.0] = "zero_test_spikes"
    qc[valid] = "valid"
    audit.loc[eligible_rows, "unit_qc_status"] = qc
    if np.all(valid):
        status = "valid"
    elif np.any(valid):
        status = "partial_valid"
    else:
        status = "no_valid_units"
    return audit.loc[:, list(SELECTED_UNIT_COLUMNS)], status


def _build_summary(
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    selected_units: pd.DataFrame,
    dataset: Any,
) -> pd.DataFrame:
    """Return a canonical long-form view of every score and its QC."""
    eligible = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool)
    ].reset_index(drop=True)
    if eligible.empty:
        return empty_swap_tuning_summary()
    rows = []
    segment_edges = np.asarray(dataset["segment_edges"].values, dtype=float)
    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        source = str(dataset["swap_source_trajectory"].values[trajectory_index])
        segment_index = int(
            dataset["swap_segment_index_1based"].values[trajectory_index]
        ) - 1
        for unit_index, unit in eligible.iterrows():
            spike_sum = float(
                dataset["test_light_spike_sum"].values[
                    trajectory_index,
                    unit_index,
                ]
            )
            for model_index, model_name in enumerate(MODEL_NAMES):
                ll_sum = float(
                    dataset["ll_sum"].values[
                        model_index,
                        trajectory_index,
                        unit_index,
                    ]
                )
                ll_bits_per_spike = float(
                    dataset["ll_bits_per_spike"].values[
                        model_index,
                        trajectory_index,
                        unit_index,
                    ]
                )
                ll_bits_per_s = float(
                    dataset["ll_bits_per_s"].values[
                        model_index,
                        trajectory_index,
                        unit_index,
                    ]
                )
                if spike_sum <= 0.0:
                    score_status = "zero_test_spikes"
                elif not (
                    np.isfinite(ll_sum)
                    and np.isfinite(ll_bits_per_spike)
                    and np.isfinite(ll_bits_per_s)
                ):
                    score_status = "nonfinite_score"
                else:
                    score_status = "valid"
                rows.append(
                    {
                        **{name: str(unit[name]) for name in IDENTITY_COLUMNS},
                        "swap_tuning_curve_comparison_id": metadata[
                            "swap_tuning_curve_comparison_id"
                        ],
                        "animal_name": metadata["animal_name"],
                        "date": metadata["date"],
                        "region": metadata["region"],
                        "dark_train_epoch": metadata["dark_epoch"],
                        "light_train_epoch": metadata["light_train_epoch"],
                        "light_test_epoch": metadata["light_test_epoch"],
                        "trajectory": trajectory,
                        "model": model_name,
                        "evaluation_bin_size_s": parameters[
                            "evaluation_bin_size_s"
                        ],
                        "gaussian_smoothing_sigma_bins": parameters[
                            "gaussian_smoothing_sigma_bins"
                        ],
                        "min_dark_firing_rate_hz": parameters[
                            "min_dark_firing_rate_hz"
                        ],
                        "min_light_firing_rate_hz": parameters[
                            "min_light_firing_rate_hz"
                        ],
                        "dark_train_movement_firing_rate_hz": unit[
                            "dark_movement_firing_rate_hz"
                        ],
                        "light_train_movement_firing_rate_hz": unit[
                            "light_train_movement_firing_rate_hz"
                        ],
                        "light_test_movement_firing_rate_hz": unit[
                            "light_test_movement_firing_rate_hz"
                        ],
                        "swap_source_trajectory": source,
                        "swap_segment_index_1based": segment_index + 1,
                        "swap_segment_start": segment_edges[segment_index],
                        "swap_segment_end": segment_edges[segment_index + 1],
                        "train_dark_same_rate_hz": dataset[
                            "train_dark_same_rate_hz"
                        ].values[trajectory_index, unit_index],
                        "train_dark_other_rate_hz": dataset[
                            "train_dark_other_rate_hz"
                        ].values[trajectory_index, unit_index],
                        "train_light_other_rate_hz": dataset[
                            "train_light_other_rate_hz"
                        ].values[trajectory_index, unit_index],
                        "test_light_target_rate_hz": dataset[
                            "test_light_target_rate_hz"
                        ].values[trajectory_index, unit_index],
                        "test_light_spike_sum": spike_sum,
                        "test_light_bin_count": dataset[
                            "test_light_bin_count"
                        ].values[trajectory_index],
                        "test_light_duration_s": dataset[
                            "test_light_duration_s"
                        ].values[trajectory_index],
                        "ll_sum": ll_sum,
                        "ll_bits_per_spike": ll_bits_per_spike,
                        "ll_bits_per_s": ll_bits_per_s,
                        "score_qc_status": score_status,
                        "unit_valid": bool(unit["valid_swap_tuning_score"]),
                    }
                )
    return pd.DataFrame.from_records(rows, columns=SUMMARY_COLUMNS)


def compute_swap_tuning_curve_comparison(
    *,
    swap_tuning_curve_comparison_id: Any,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    tuning_curve_artifact_paths: Mapping[str, Mapping[str, Path]] | None,
    movement_firing_rate_tables_by_role: Mapping[str, pd.DataFrame],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    position_offset_samples: int,
    movement_interval: Any,
    movement_analysis_status: str,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    evaluation_bin_size_s: float = DEFAULT_EVALUATION_BIN_SIZE_S,
    gaussian_smoothing_sigma_bins: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
    min_dark_firing_rate_hz: float = DEFAULT_MIN_DARK_FIRING_RATE_HZ,
    min_light_firing_rate_hz: float = DEFAULT_MIN_LIGHT_FIRING_RATE_HZ,
    source_tuning_curve_ids_by_role_trajectory: Mapping[
        str, Mapping[str, Any]
    ]
    | None = None,
    source_tuning_parameters_sha256_by_role_trajectory: Mapping[
        str, Mapping[str, str]
    ]
    | None = None,
    movement_firing_rate_ids_by_role: Mapping[str, Any] | None = None,
    movement_firing_rate_table_sha256_by_role: Mapping[str, str] | None = None,
    movement_intervals_sha256_by_role: Mapping[str, str],
    sources: Mapping[str, Any] | None = None,
    tuning_curves_by_role_trajectory: Mapping[str, Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Compute one empirical swap comparison from selected NWB-backed inputs."""
    metadata = _metadata(
        swap_tuning_curve_comparison_id=swap_tuning_curve_comparison_id,
        animal_name=animal_name,
        date=date,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
    )
    parameters = _effective_parameters(
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        evaluation_bin_size_s=evaluation_bin_size_s,
        gaussian_smoothing_sigma_bins=gaussian_smoothing_sigma_bins,
        min_dark_firing_rate_hz=min_dark_firing_rate_hz,
        min_light_firing_rate_hz=min_light_firing_rate_hz,
    )
    movement_interval_hashes = _validate_role_sha256_mapping(
        movement_intervals_sha256_by_role,
        name="movement_intervals_sha256_by_role",
    )
    analysis = _analysis_module()
    _validate_analysis_contract(analysis)
    position_offset = _validate_position_offset(position_offset_samples)
    if set(trajectory_intervals) != set(TRAJECTORY_TYPES):
        raise ValueError("trajectory_intervals must contain exactly four paths.")
    if set(graph_inputs_by_trajectory) != set(TRAJECTORY_TYPES):
        raise ValueError("graph_inputs_by_trajectory must contain exactly four paths.")
    role_epochs = _role_epochs(metadata)
    if set(movement_firing_rate_tables_by_role) != set(role_epochs):
        raise ValueError(
            "movement_firing_rate_tables_by_role must contain dark, light_train, "
            "and light_test."
        )
    movement_tables = {
        role: _validate_movement_table_context(
            movement_firing_rate_tables_by_role[role],
            metadata=metadata,
            epoch=role_epochs[role],
        )
        for role in role_epochs
    }
    speed_threshold = _speed_threshold_from_tables(movement_tables)
    from v1ca1.spyglass.dark_light_glm import derive_graph_geometry

    graph_length_cm, segment_edges = derive_graph_geometry(
        graph_inputs_by_trajectory
    )
    curves, curve_identity, curve_hashes = _load_and_validate_tuning_inputs(
        tuning_curve_artifact_paths=tuning_curve_artifact_paths,
        tuning_curves_by_role_trajectory=tuning_curves_by_role_trajectory,
        metadata=metadata,
        graph_length_cm=graph_length_cm,
    )
    selected_units, eligible, native_unit_keys, input_status = _align_unit_inputs(
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        curve_identity=curve_identity,
        movement_tables_by_role=movement_tables,
        parameters=parameters,
    )
    upstream_provenance = {
        "selected_units_sha256": _selected_units_sha256(selected_units),
        "source_tuning_curve_sha256_by_role_trajectory": curve_hashes,
        "source_tuning_curve_ids_by_role_trajectory": _optional_nested_strings(
            source_tuning_curve_ids_by_role_trajectory,
            name="source_tuning_curve_ids_by_role_trajectory",
        ),
        "source_tuning_parameters_sha256_by_role_trajectory": (
            _optional_nested_strings(
                source_tuning_parameters_sha256_by_role_trajectory,
                name="source_tuning_parameters_sha256_by_role_trajectory",
            )
        ),
        "movement_firing_rate_table_sha256_by_role": (
            _optional_role_strings(
                movement_firing_rate_table_sha256_by_role,
                name="movement_firing_rate_table_sha256_by_role",
            )
            or {
                role: movement_firing_rate_table_content_sha256(table)
                for role, table in movement_tables.items()
            }
        ),
        "movement_firing_rate_ids_by_role": _optional_role_strings(
            movement_firing_rate_ids_by_role,
            name="movement_firing_rate_ids_by_role",
        ),
        "movement_intervals_sha256_by_role": movement_interval_hashes,
        "position_offset_samples": position_offset,
        "speed_threshold_cm_s": speed_threshold,
    }

    def terminal(status: str, **detail: Any) -> dict[str, Any]:
        dataset = _terminal_dataset(
            metadata=metadata,
            parameters=parameters,
            upstream_provenance=upstream_provenance,
            selected_units=selected_units,
            segment_edges=segment_edges,
            analysis_status=status,
            terminal_detail=detail,
        )
        return validate_swap_tuning_curve_comparison_result(
            {
                "metadata": metadata,
                "parameters": parameters,
                "upstream_provenance": upstream_provenance,
                "selected_units": selected_units,
                "summary": empty_swap_tuning_summary(),
                "dataset": dataset,
                "analysis_status": status,
                "artifact_origin": "computed",
                "legacy_artifact_provenance": None,
            }
        )

    if input_status != "valid":
        return terminal(input_status)
    upstream_statuses = {
        str(curve.attrs["analysis_status"])
        for by_trajectory in curves.values()
        for curve in by_trajectory.values()
    }
    if upstream_statuses != {"valid"}:
        return terminal(
            "upstream_terminal",
            upstream_tuning_statuses_json=json.dumps(sorted(upstream_statuses)),
        )
    if not _has_valid_position_samples(position):
        if str(movement_analysis_status) == "valid":
            raise ValueError("Selected Position conflicts with valid held-out movement.")
        return terminal("no_valid_position")
    movement_status = str(movement_analysis_status)
    if movement_status not in {
        "valid",
        "no_units",
        "no_valid_position",
        "no_movement",
    }:
        raise ValueError("movement_analysis_status is unsupported.")
    if movement_status == "no_units":
        raise ValueError("Non-empty eligible units conflict with no_units movement.")
    movement_duration = _interval_duration_s(movement_interval)
    if movement_status in {"no_valid_position", "no_movement"}:
        if movement_duration > 0.0:
            raise ValueError("Terminal held-out movement status has positive duration.")
        return terminal(movement_status)
    if movement_duration <= 0.0:
        raise ValueError("Valid held-out movement must have positive duration.")

    first_curve = curves["dark"][TRAJECTORY_TYPES[0]]
    bin_edges_cm = np.asarray(
        json.loads(str(first_curve.attrs["bin_edges_cm_json"])),
        dtype=float,
    )
    bin_edges = bin_edges_cm / float(graph_length_cm)
    task_progression = _derive_task_progression(
        position=position,
        trajectory_intervals=trajectory_intervals,
        graph_inputs_by_trajectory=graph_inputs_by_trajectory,
    )
    arrays, missing_trajectory = _compute_empirical_arrays(
        curves=curves,
        eligible=eligible,
        spikes=spikes,
        native_unit_keys=native_unit_keys,
        task_progression_by_trajectory=task_progression,
        trajectory_intervals=trajectory_intervals,
        movement_interval=movement_interval,
        bin_edges=bin_edges,
        segment_edges=segment_edges,
        parameters=parameters,
    )
    if arrays is None:
        return terminal(
            "no_trajectory_samples",
            missing_trajectory=str(missing_trajectory),
        )
    source_payload = {} if sources is None else dict(sources)
    source_payload["swap_tuning_upstream"] = upstream_provenance
    dataset = _build_dataset(
        arrays=arrays,
        metadata=metadata,
        parameters=parameters,
        upstream_provenance=upstream_provenance,
        selected_units=selected_units,
        source_payload=source_payload,
    )
    selected_units, analysis_status = _audit_unit_scores(selected_units, dataset)
    dataset.attrs["analysis_status"] = analysis_status
    summary = _build_summary(
        metadata=metadata,
        parameters=parameters,
        selected_units=selected_units,
        dataset=dataset,
    )
    return validate_swap_tuning_curve_comparison_result(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream_provenance,
            "selected_units": selected_units,
            "summary": summary,
            "dataset": dataset,
            "analysis_status": analysis_status,
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
    )


def _validate_selected_units(
    table: pd.DataFrame,
    *,
    parameters: Mapping[str, Any],
    analysis_status: str,
) -> pd.DataFrame:
    """Validate the exact all-source-unit audit schema and arithmetic."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("selected_units must be a pandas DataFrame.")
    if tuple(table.columns) != SELECTED_UNIT_COLUMNS:
        raise ValueError("selected_units does not have the canonical schema.")
    if table.empty:
        return table.copy()
    output = table.copy()
    for name in IDENTITY_COLUMNS:
        output[name] = output[name].astype(str)
    expected_stable = output["spikesorting_merge_id"] + ":" + output["unit_id"]
    if not np.array_equal(
        output["stable_unit_id"].to_numpy(dtype=str),
        expected_stable.to_numpy(dtype=str),
    ):
        raise ValueError("selected_units stable identities are inconsistent.")
    if output["stable_unit_id"].duplicated().any() or output[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError("selected_units identities must be unique.")
    source_index = pd.to_numeric(
        output["source_unit_index"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.array_equal(source_index, np.arange(len(output), dtype=float)):
        raise ValueError("selected_units source_unit_index must be contiguous.")
    eligible = output["eligible_for_comparison"].to_numpy()
    dark_pass = output["passes_dark_firing_rate_threshold"].to_numpy()
    light_pass = output["passes_light_firing_rate_threshold"].to_numpy()
    for name, values in {
        "eligible_for_comparison": eligible,
        "passes_dark_firing_rate_threshold": dark_pass,
        "passes_light_firing_rate_threshold": light_pass,
        "valid_swap_tuning_score": output["valid_swap_tuning_score"].to_numpy(),
    }.items():
        if not all(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"selected_units {name} must contain booleans.")
    eligible = eligible.astype(bool)
    dark_pass = dark_pass.astype(bool)
    light_pass = light_pass.astype(bool)
    rate_inputs_valid = analysis_status not in {
        "no_units",
        "no_valid_position",
        "no_movement",
    }
    expected_eligible = (
        dark_pass & light_pass
        if rate_inputs_valid
        else np.zeros(len(output), dtype=bool)
    )
    if not np.array_equal(eligible, expected_eligible):
        raise ValueError(
            "selected_units eligibility must equal the two strict firing-rate "
            "masks whenever movement inputs are valid."
        )
    eligible_index = pd.to_numeric(
        output["eligible_unit_index"],
        errors="coerce",
    ).to_numpy(dtype=float)
    expected_index = np.full(len(output), -1.0)
    expected_index[eligible] = np.arange(int(np.sum(eligible)), dtype=float)
    if not np.array_equal(eligible_index, expected_index):
        raise ValueError("selected_units eligible_unit_index is inconsistent.")
    for name in (
        "dark_movement_firing_rate_hz",
        "light_train_movement_firing_rate_hz",
        "light_test_movement_firing_rate_hz",
        "test_light_spike_count",
    ):
        values = pd.to_numeric(output[name], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isinf(values)) or np.any(values[np.isfinite(values)] < 0.0):
            raise ValueError(f"selected_units {name} must be non-negative or NaN.")
    dark_rates = pd.to_numeric(
        output["dark_movement_firing_rate_hz"],
        errors="coerce",
    ).to_numpy(dtype=float)
    light_rates = pd.to_numeric(
        output["light_train_movement_firing_rate_hz"],
        errors="coerce",
    ).to_numpy(dtype=float)
    expected_dark_pass = np.isfinite(dark_rates) & (
        dark_rates > float(parameters["min_dark_firing_rate_hz"])
    )
    expected_light_pass = np.isfinite(light_rates) & (
        light_rates > float(parameters["min_light_firing_rate_hz"])
    )
    if not np.array_equal(dark_pass, expected_dark_pass) or not np.array_equal(
        light_pass,
        expected_light_pass,
    ):
        raise ValueError("selected_units firing-rate threshold audit is stale.")
    finite_counts = pd.to_numeric(
        output["n_finite_scores"],
        errors="coerce",
    ).to_numpy(dtype=float)
    expected_counts = pd.to_numeric(
        output["n_expected_scores"],
        errors="coerce",
    ).to_numpy(dtype=float)
    expected_score_count = len(MODEL_NAMES) * len(TRAJECTORY_TYPES)
    if (
        np.any(~np.isfinite(finite_counts))
        or np.any(finite_counts < 0.0)
        or np.any(finite_counts > expected_score_count)
        or not np.allclose(finite_counts, np.rint(finite_counts))
        or not np.all(expected_counts == expected_score_count)
    ):
        raise ValueError("selected_units score counts are invalid.")
    valid = output["valid_swap_tuning_score"].astype(bool).to_numpy()
    if not np.array_equal(valid, eligible & (finite_counts == expected_score_count)):
        raise ValueError("selected_units valid-score flags are stale.")
    test_spike_counts = pd.to_numeric(
        output["test_light_spike_count"],
        errors="coerce",
    ).to_numpy(dtype=float)
    evaluated = analysis_status in {"valid", "partial_valid", "no_valid_units"}
    if np.any((~eligible) & ((finite_counts != 0.0) | (test_spike_counts != 0.0))):
        raise ValueError("Ineligible units cannot contain empirical score audits.")
    if not evaluated and np.any(
        (finite_counts != 0.0) | (test_spike_counts != 0.0) | valid
    ):
        raise ValueError("Terminal unit audits cannot contain computed scores.")
    expected_qc = np.full(len(output), "not_computed", dtype=object)
    expected_qc[~dark_pass & light_pass] = "excluded_dark_firing_rate"
    expected_qc[dark_pass & ~light_pass] = "excluded_light_firing_rate"
    expected_qc[~dark_pass & ~light_pass] = "excluded_both_firing_rates"
    if evaluated:
        eligible_qc = np.full(int(np.sum(eligible)), "partial_nonfinite_score", dtype=object)
        eligible_finite = finite_counts[eligible]
        eligible_spikes = test_spike_counts[eligible]
        eligible_qc[eligible_finite == 0] = "no_valid_scores"
        eligible_qc[eligible_spikes <= 0.0] = "zero_test_spikes"
        eligible_qc[eligible_finite == expected_score_count] = "valid"
        expected_qc[eligible] = eligible_qc
    statuses = output["unit_qc_status"].astype(str).to_numpy()
    if not np.array_equal(statuses, expected_qc.astype(str)):
        raise ValueError("selected_units unit_qc_status labels are stale.")
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)]


def _validate_dataset(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
    selected_units: pd.DataFrame,
    analysis_status: str,
) -> Any:
    """Validate terminal or complete empirical swap NetCDF content."""
    for name, expected in {
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "analysis_status": analysis_status,
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_epoch": metadata["dark_epoch"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
    }.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"Swap-tuning dataset has mismatched {name!r}.")
    for name in (
        "evaluation_bin_size_s",
        "gaussian_smoothing_sigma_bins",
        "min_dark_firing_rate_hz",
        "min_light_firing_rate_hz",
    ):
        if not np.isclose(
            float(dataset.attrs.get(name, np.nan)),
            float(parameters[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Swap-tuning dataset has mismatched {name!r}.")
    try:
        saved_upstream = json.loads(str(dataset.attrs["upstream_provenance_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Swap-tuning dataset has invalid upstream provenance.") from exc
    if saved_upstream != dict(upstream_provenance):
        raise ValueError("Swap-tuning dataset upstream provenance is stale.")
    for coordinate, expected in {
        "model": MODEL_NAMES,
        "trajectory": TRAJECTORY_TYPES,
    }.items():
        if coordinate not in dataset.coords or not np.array_equal(
            np.asarray(dataset.coords[coordinate].values, dtype=str),
            np.asarray(expected, dtype=str),
        ):
            raise ValueError(f"Swap-tuning dataset has noncanonical {coordinate}.")
    eligible = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool)
    ]
    for name in IDENTITY_COLUMNS:
        if name not in dataset.coords:
            raise ValueError(f"Swap-tuning dataset lacks {name!r} coordinate.")
        expected = eligible[name].astype(str).to_numpy()
        observed = np.asarray(dataset.coords[name].values, dtype=str)
        if not np.array_equal(observed, expected):
            raise ValueError(f"Swap-tuning dataset has stale {name!r} coordinate.")
    if not np.array_equal(
        np.asarray(dataset.coords["unit"].values, dtype=str),
        eligible["stable_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Swap-tuning unit coordinate is not persistent identity.")
    is_terminal = analysis_status not in {"valid", "partial_valid", "no_valid_units"}
    if is_terminal:
        if dataset.data_vars:
            raise ValueError("Terminal swap-tuning datasets cannot contain data variables.")
        if str(dataset.attrs.get("fit_stage")) != "terminal":
            raise ValueError("Terminal swap-tuning dataset has stale fit_stage.")
        if str(dataset.attrs.get("terminal_reason", "")) != analysis_status:
            raise ValueError("Terminal swap-tuning dataset has stale terminal_reason.")
        if analysis_status == "no_trajectory_samples" and str(
            dataset.attrs.get("missing_trajectory", "")
        ) not in TRAJECTORY_TYPES:
            raise ValueError("no_trajectory_samples requires the missing path name.")
        if analysis_status == "upstream_terminal":
            try:
                upstream_statuses = json.loads(
                    str(dataset.attrs["upstream_tuning_statuses_json"])
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "upstream_terminal requires upstream tuning statuses."
                ) from exc
            if not isinstance(upstream_statuses, list) or not upstream_statuses or (
                set(map(str, upstream_statuses)) == {"valid"}
            ):
                raise ValueError("upstream_terminal tuning statuses are stale.")
        return dataset
    if set(dataset.data_vars) != set(SCIENTIFIC_VARIABLE_DIMS):
        missing = sorted(set(SCIENTIFIC_VARIABLE_DIMS).difference(dataset.data_vars))
        extra = sorted(set(dataset.data_vars).difference(SCIENTIFIC_VARIABLE_DIMS))
        raise ValueError(
            f"Swap-tuning scientific schema is incomplete: missing={missing!r}, "
            f"extra={extra!r}."
        )
    for name, dims in SCIENTIFIC_VARIABLE_DIMS.items():
        if tuple(dataset[name].dims) != dims:
            raise ValueError(f"Swap-tuning variable {name!r} has stale dimensions.")
        values = np.asarray(dataset[name].values)
        if values.dtype.kind not in "OUSb" and np.any(np.isinf(values.astype(float))):
            raise ValueError(f"Swap-tuning variable {name!r} contains infinities.")
    for name, expected in {
        "bin_size_s": parameters["evaluation_bin_size_s"],
        "sigma_bins": parameters["gaussian_smoothing_sigma_bins"],
        "place_bin_size_cm": REQUIRED_UPSTREAM_BIN_SIZE_CM,
        "min_dark_fr_hz": parameters["min_dark_firing_rate_hz"],
        "min_light_fr_hz": parameters["min_light_firing_rate_hz"],
        "empirical_epsilon": EMPIRICAL_EPSILON,
    }.items():
        if not np.isclose(
            float(dataset.attrs.get(name, np.nan)),
            float(expected),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Swap-tuning dataset has stale scientific {name!r}.")
    if not _strict_boolean(dataset.attrs.get("apply_fr_filter"), name="apply_fr_filter"):
        raise ValueError("Swap-tuning dataset must apply train firing-rate filtering.")
    for name, expected in {
        "scoring_scope": "light_test_swapped_segment_only",
        "training_tuning_scope": "full_trajectory_movement_interval",
    }.items():
        if str(dataset.attrs.get(name, "")) != expected:
            raise ValueError(f"Swap-tuning dataset has stale {name!r}.")
    try:
        formulas = json.loads(str(dataset.attrs["empirical_model_formulas_json"]))
        swap_rule = json.loads(str(dataset.attrs["swap_rule_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Swap-tuning dataset has invalid model provenance.") from exc
    if formulas != EMPIRICAL_MODEL_FORMULAS or swap_rule != SWAP_CONFIGURATION:
        raise ValueError("Swap-tuning dataset model formulas or swap rule are stale.")

    segment_edges = np.asarray(dataset["segment_edges"].values, dtype=float)
    if (
        segment_edges.shape != (4,)
        or not np.all(np.isfinite(segment_edges))
        or np.any(np.diff(segment_edges) <= 0.0)
        or not np.allclose(
            segment_edges[[0, -1]],
            [0.0, 1.0],
            rtol=0.0,
            atol=1e-12,
        )
        or not np.array_equal(
            np.asarray(dataset.coords["segment_edge"].values, dtype=int),
            np.arange(4, dtype=int),
        )
    ):
        raise ValueError("Swap-tuning segment edges are noncanonical.")
    expected_sources = np.asarray(
        [SWAP_CONFIGURATION[name]["source_trajectory"] for name in TRAJECTORY_TYPES],
        dtype=str,
    )
    expected_segment_indices = np.asarray(
        [SWAP_CONFIGURATION[name]["segment_index"] + 1 for name in TRAJECTORY_TYPES],
        dtype=int,
    )
    observed_sources = np.asarray(
        dataset["swap_source_trajectory"].values,
        dtype=str,
    )
    observed_indices = np.asarray(
        dataset["swap_segment_index_1based"].values,
        dtype=int,
    )
    if not np.array_equal(observed_sources, expected_sources) or not np.array_equal(
        observed_indices,
        expected_segment_indices,
    ):
        raise ValueError("Swap-tuning source trajectories or segment indices are stale.")
    zero_based_indices = expected_segment_indices - 1
    expected_starts = segment_edges[zero_based_indices]
    expected_ends = segment_edges[zero_based_indices + 1]
    if not np.allclose(
        dataset["swap_segment_start"],
        expected_starts,
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        dataset["swap_segment_end"],
        expected_ends,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Swap-tuning saved segment boundaries are stale.")
    tp_bins = np.asarray(dataset.coords["tp_bin"].values, dtype=float)
    if (
        tp_bins.ndim != 1
        or not np.all(np.isfinite(tp_bins))
        or (tp_bins.size > 1 and np.any(np.diff(tp_bins) <= 0.0))
    ):
        raise ValueError("Swap-tuning TP-bin centers must be finite and increasing.")
    analysis = _analysis_module()
    expected_masks = np.stack(
        [
            analysis.segment_mask(tp_bins, segment_edges, int(segment_index))
            for segment_index in zero_based_indices
        ]
    )
    if not np.array_equal(
        np.asarray(dataset["segment_bin_mask"].values, dtype=bool),
        expected_masks,
    ):
        raise ValueError("Swap-tuning segment-bin masks are stale.")

    finite_nonnegative_variables = (
        "dark_train_movement_firing_rate_hz",
        "light_train_movement_firing_rate_hz",
        "light_test_movement_firing_rate_hz",
        "model_tuning_hz",
        "train_dark_same_rate_hz",
        "train_dark_other_rate_hz",
        "train_light_other_rate_hz",
        "test_light_target_rate_hz",
        "test_light_spike_sum",
        "same_dark_train_tuning_hz",
        "other_dark_train_tuning_hz",
        "other_light_train_tuning_hz",
        "test_light_tuning_hz",
        "test_light_bin_count",
        "test_light_duration_s",
    )
    for name in finite_nonnegative_variables:
        values = np.asarray(dataset[name].values, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"Swap-tuning variable {name!r} must be finite/nonnegative.")
    model_tuning = np.asarray(dataset["model_tuning_hz"].values, dtype=float)
    if np.any(model_tuning < EMPIRICAL_EPSILON):
        raise ValueError("Empirical model tunings must respect the fixed epsilon floor.")
    spike_counts = np.asarray(dataset["test_light_spike_sum"].values, dtype=float)
    if not np.allclose(spike_counts, np.rint(spike_counts), rtol=0.0, atol=1e-9):
        raise ValueError("Held-out spike sums must be integer counts.")
    bin_counts = np.asarray(dataset["test_light_bin_count"].values, dtype=float)
    duration = np.asarray(dataset["test_light_duration_s"].values, dtype=float)
    if (
        np.any(bin_counts <= 0.0)
        or not np.allclose(bin_counts, np.rint(bin_counts), rtol=0.0, atol=1e-9)
        or np.any(duration <= 0.0)
        or not np.allclose(
            duration,
            bin_counts * float(parameters["evaluation_bin_size_s"]),
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError("Held-out bin counts and durations are inconsistent.")

    same_dark = np.asarray(dataset["same_dark_train_tuning_hz"].values, dtype=float)
    other_dark = np.asarray(dataset["other_dark_train_tuning_hz"].values, dtype=float)
    other_light = np.asarray(dataset["other_light_train_tuning_hz"].values, dtype=float)
    same_dark_rates = np.asarray(dataset["train_dark_same_rate_hz"].values, dtype=float)
    other_dark_rates = np.asarray(dataset["train_dark_other_rate_hz"].values, dtype=float)
    trajectory_index = {name: index for index, name in enumerate(TRAJECTORY_TYPES)}
    for target_index, target in enumerate(TRAJECTORY_TYPES):
        source_index = trajectory_index[SWAP_CONFIGURATION[target]["source_trajectory"]]
        if not np.allclose(
            other_dark[target_index],
            same_dark[source_index],
            rtol=1e-10,
            atol=1e-12,
        ) or not np.allclose(
            other_dark_rates[target_index],
            same_dark_rates[source_index],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Saved other-dark source trajectories are inconsistent.")
        expected_tunings = analysis.build_empirical_swap_tunings(
            same_dark[target_index],
            other_light[target_index],
            other_dark[target_index],
            tp_bins,
            segment_edges,
            int(zero_based_indices[target_index]),
            epsilon=EMPIRICAL_EPSILON,
        )
        for model_index, model_name in enumerate(MODEL_NAMES):
            if not np.allclose(
                model_tuning[model_index, target_index],
                expected_tunings[model_name],
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Empirical model tuning formula is stale for {model_name!r}."
                )

    eligible_reset = eligible.reset_index(drop=True)
    for variable, column in (
        ("dark_train_movement_firing_rate_hz", "dark_movement_firing_rate_hz"),
        (
            "light_train_movement_firing_rate_hz",
            "light_train_movement_firing_rate_hz",
        ),
        (
            "light_test_movement_firing_rate_hz",
            "light_test_movement_firing_rate_hz",
        ),
    ):
        if not np.allclose(
            dataset[variable],
            eligible_reset[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Swap-tuning {variable!r} differs from unit audit.")
    ll = np.asarray(dataset["ll_sum"].values, dtype=float)
    if not np.all(np.isfinite(ll)) or np.any(ll > 1e-10):
        raise ValueError("Poisson log likelihoods must be finite and non-positive.")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_bits_per_spike = np.where(
            spike_counts[None, :, :] > 0.0,
            ll / (np.log(2.0) * spike_counts[None, :, :]),
            np.nan,
        )
    expected_bits_per_s = ll / (np.log(2.0) * duration[None, :, None])
    if not np.allclose(
        dataset["ll_bits_per_spike"],
        expected_bits_per_spike,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ) or not np.allclose(
        dataset["ll_bits_per_s"],
        expected_bits_per_s,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError("Swap-tuning likelihood unit conversion is inconsistent.")
    expected_finite_counts = np.sum(
        np.isfinite(expected_bits_per_spike),
        axis=(0, 1),
    )
    if not np.array_equal(
        eligible_reset["n_finite_scores"].to_numpy(dtype=int),
        expected_finite_counts,
    ) or not np.allclose(
        eligible_reset["test_light_spike_count"].to_numpy(dtype=float),
        np.sum(spike_counts, axis=0),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Swap-tuning unit score audit differs from scientific arrays.")
    return dataset


def _validate_upstream_provenance(
    upstream: Mapping[str, Any],
    *,
    selected_units: pd.DataFrame,
) -> tuple[dict[str, Any], str]:
    """Validate every frozen upstream identity and file digest."""
    output = dict(upstream)
    required = {
        "selected_units_sha256",
        "source_tuning_curve_sha256_by_role_trajectory",
        "source_tuning_curve_ids_by_role_trajectory",
        "source_tuning_parameters_sha256_by_role_trajectory",
        "movement_firing_rate_table_sha256_by_role",
        "movement_firing_rate_ids_by_role",
        "movement_intervals_sha256_by_role",
        "position_offset_samples",
        "speed_threshold_cm_s",
    }
    if set(output) != required:
        raise ValueError("upstream_provenance does not have the canonical keys.")
    selected_hash = _selected_units_sha256(selected_units)
    if not _is_sha256(output["selected_units_sha256"]) or str(
        output["selected_units_sha256"]
    ) != selected_hash:
        raise ValueError("upstream selected-unit identity hash is stale.")
    output["source_tuning_curve_sha256_by_role_trajectory"] = (
        _validate_nested_sha256_mapping(
            output["source_tuning_curve_sha256_by_role_trajectory"],
            name="source_tuning_curve_sha256_by_role_trajectory",
        )
    )
    tuning_ids = dict(output["source_tuning_curve_ids_by_role_trajectory"])
    output["source_tuning_curve_ids_by_role_trajectory"] = (
        {}
        if not tuning_ids
        else _optional_nested_strings(
            tuning_ids,
            name="source_tuning_curve_ids_by_role_trajectory",
        )
    )
    tuning_parameter_hashes = dict(
        output["source_tuning_parameters_sha256_by_role_trajectory"]
    )
    output["source_tuning_parameters_sha256_by_role_trajectory"] = (
        {}
        if not tuning_parameter_hashes
        else _validate_nested_sha256_mapping(
            tuning_parameter_hashes,
            name="source_tuning_parameters_sha256_by_role_trajectory",
        )
    )
    output["movement_firing_rate_table_sha256_by_role"] = (
        _validate_role_sha256_mapping(
            output["movement_firing_rate_table_sha256_by_role"],
            name="movement_firing_rate_table_sha256_by_role",
        )
    )
    movement_ids = dict(output["movement_firing_rate_ids_by_role"])
    output["movement_firing_rate_ids_by_role"] = (
        {}
        if not movement_ids
        else _optional_role_strings(
            movement_ids,
            name="movement_firing_rate_ids_by_role",
        )
    )
    output["movement_intervals_sha256_by_role"] = _validate_role_sha256_mapping(
        output["movement_intervals_sha256_by_role"],
        name="movement_intervals_sha256_by_role",
    )
    output["position_offset_samples"] = _validate_position_offset(
        output["position_offset_samples"]
    )
    speed_threshold = float(output["speed_threshold_cm_s"])
    if not np.isfinite(speed_threshold) or speed_threshold < 0.0:
        raise ValueError("upstream speed_threshold_cm_s must be non-negative and finite.")
    output["speed_threshold_cm_s"] = speed_threshold
    return output, selected_hash


def _validate_summary(
    table: pd.DataFrame,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
    selected_units: pd.DataFrame,
    dataset: Any,
    analysis_status: str,
) -> pd.DataFrame:
    """Validate the long-form summary and its exact NetCDF correspondence."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("summary must be a pandas DataFrame.")
    if tuple(table.columns) != SUMMARY_COLUMNS:
        raise ValueError("summary does not have the canonical schema.")
    eligible = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool)
    ]
    is_terminal = analysis_status not in {"valid", "partial_valid", "no_valid_units"}
    if is_terminal:
        if not table.empty:
            raise ValueError("Terminal swap-tuning results require an empty summary.")
        return table
    expected_rows = len(eligible) * len(MODEL_NAMES) * len(TRAJECTORY_TYPES)
    if len(table) != expected_rows:
        raise ValueError("Swap-tuning summary has an unexpected row count.")
    if set(table["score_qc_status"].astype(str)) - set(SCORE_QC_STATUSES):
        raise ValueError("Swap-tuning summary contains unsupported score QC.")
    for field, expected in {
        "swap_tuning_curve_comparison_id": metadata[
            "swap_tuning_curve_comparison_id"
        ],
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_train_epoch": metadata["dark_epoch"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
    }.items():
        if table[field].astype(str).unique().tolist() != [str(expected)]:
            raise ValueError(f"Swap-tuning summary has mismatched {field!r}.")
    expected = _build_summary(
        metadata=metadata,
        parameters=parameters,
        selected_units=selected_units,
        dataset=dataset,
    )
    for name in SUMMARY_COLUMNS:
        observed_values = table[name].to_numpy()
        expected_values = expected[name].to_numpy()
        if observed_values.dtype.kind in "OUSb" or expected_values.dtype.kind in "OUSb":
            matches = np.array_equal(
                observed_values.astype(str),
                expected_values.astype(str),
            )
        else:
            matches = np.allclose(
                observed_values.astype(float),
                expected_values.astype(float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        if not matches:
            raise ValueError(f"Swap-tuning summary column {name!r} is stale.")
    return table


def validate_swap_tuning_curve_comparison_result(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize one complete in-memory result bundle."""
    required = {
        "metadata",
        "parameters",
        "upstream_provenance",
        "selected_units",
        "summary",
        "dataset",
        "analysis_status",
        "artifact_origin",
        "legacy_artifact_provenance",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Swap-tuning result is missing fields {missing!r}.")
    metadata = dict(result["metadata"])
    metadata = _metadata(**metadata)
    raw_parameters = dict(result["parameters"])
    parameters = _effective_parameters(
        parameter_name=raw_parameters["parameter_name"],
        parameter_sha256=raw_parameters["parameter_sha256"],
        output_rule_sha256=raw_parameters["output_rule_sha256"],
        evaluation_bin_size_s=raw_parameters["evaluation_bin_size_s"],
        gaussian_smoothing_sigma_bins=raw_parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        min_dark_firing_rate_hz=raw_parameters["min_dark_firing_rate_hz"],
        min_light_firing_rate_hz=raw_parameters["min_light_firing_rate_hz"],
    )
    status = str(result["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Unsupported swap-tuning analysis_status {status!r}.")
    selected_units = _validate_selected_units(
        result["selected_units"],
        parameters=parameters,
        analysis_status=status,
    )
    upstream, selected_hash = _validate_upstream_provenance(
        result["upstream_provenance"],
        selected_units=selected_units,
    )
    dataset = _validate_dataset(
        result["dataset"],
        metadata=metadata,
        parameters=parameters,
        upstream_provenance=upstream,
        selected_units=selected_units,
        analysis_status=status,
    )
    summary = _validate_summary(
        result["summary"],
        metadata=metadata,
        parameters=parameters,
        selected_units=selected_units,
        dataset=dataset,
        analysis_status=status,
    )
    origin = str(result["artifact_origin"])
    if origin not in {"computed", "registered_existing"}:
        raise ValueError("artifact_origin must be computed or registered_existing.")
    legacy = result["legacy_artifact_provenance"]
    if origin == "computed" and legacy is not None:
        raise ValueError("Computed results cannot have legacy provenance.")
    if origin == "registered_existing" and not isinstance(legacy, Mapping):
        raise ValueError("Registered results require legacy provenance.")
    eligible = selected_units["eligible_for_comparison"].astype(bool)
    valid = selected_units["valid_swap_tuning_score"].astype(bool)
    n_source_units = len(selected_units)
    n_units = int(eligible.sum())
    n_valid_units = int(valid.sum())
    has_scientific_output = bool(dataset.data_vars)
    if has_scientific_output:
        if n_source_units == 0 or n_units == 0:
            raise ValueError("Evaluated swap-tuning output requires eligible units.")
        expected_status = (
            "valid"
            if n_valid_units == n_units
            else "partial_valid"
            if n_valid_units > 0
            else "no_valid_units"
        )
    elif n_source_units == 0:
        expected_status = "no_units"
    elif status == "no_eligible_units":
        if n_units != 0:
            raise ValueError("no_eligible_units cannot contain eligible units.")
        expected_status = "no_eligible_units"
    elif status in {"no_valid_position", "no_movement"}:
        if n_units != 0:
            raise ValueError(f"{status} cannot contain eligible units.")
        expected_status = status
    elif status in {"upstream_terminal", "no_trajectory_samples"}:
        if n_units == 0:
            raise ValueError(f"{status} requires threshold-eligible units.")
        expected_status = status
    else:
        raise ValueError("Terminal swap-tuning state is inconsistent with unit counts.")
    if status != expected_status:
        raise ValueError("Swap-tuning analysis_status is inconsistent with unit counts.")
    return {
        "metadata": metadata,
        "parameters": parameters,
        "upstream_provenance": upstream,
        "selected_units": selected_units,
        "summary": summary,
        "dataset": dataset,
        "analysis_status": status,
        "artifact_origin": origin,
        "legacy_artifact_provenance": None if legacy is None else dict(legacy),
        "n_source_units": n_source_units,
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "selected_units_sha256": selected_hash,
    }


_SELECTED_UNIT_TEXT_COLUMNS = (*IDENTITY_COLUMNS, "unit_qc_status")
_SELECTED_UNIT_INTEGER_COLUMNS = (
    "source_unit_index",
    "eligible_unit_index",
    "n_finite_scores",
    "n_expected_scores",
)
_SELECTED_UNIT_BOOLEAN_COLUMNS = (
    "passes_dark_firing_rate_threshold",
    "passes_light_firing_rate_threshold",
    "eligible_for_comparison",
    "valid_swap_tuning_score",
)
_SUMMARY_TEXT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "swap_tuning_curve_comparison_id",
    "animal_name",
    "date",
    "region",
    "dark_train_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "trajectory",
    "model",
    "swap_source_trajectory",
    "score_qc_status",
)
_SUMMARY_INTEGER_COLUMNS = ("swap_segment_index_1based",)
_SUMMARY_BOOLEAN_COLUMNS = ("unit_valid",)


def _nwb_column_description(name: str) -> str:
    """Return a compact description for one self-describing scratch column."""
    return name.replace("_", " ") + "."


def _empty_nwb_dynamic_table(
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    ragged_columns: Sequence[str] = (),
) -> Any:
    """Construct a typed zero-row DynamicTable without HDMF row inference."""
    from hdmf.common import DynamicTable, VectorData, VectorIndex

    output_columns = []
    for column in columns:
        if column in ragged_columns:
            data = VectorData(
                name=column,
                description=_nwb_column_description(column),
                data=np.asarray([], dtype=float),
            )
            index = VectorIndex(
                name=f"{column}_index",
                data=np.asarray([], dtype=np.int64),
                target=data,
            )
            output_columns.extend((data, index))
            continue
        if column in text_columns:
            values = np.asarray([], dtype="S1")
        elif column in integer_columns:
            values = np.asarray([], dtype=np.int64)
        elif column in boolean_columns:
            values = np.asarray([], dtype=bool)
        else:
            values = np.asarray([], dtype=float)
        output_columns.append(
            VectorData(
                name=column,
                description=_nwb_column_description(column),
                data=values,
            )
        )
    return DynamicTable(
        name=name,
        description=description,
        columns=output_columns,
    )


def _normalize_nwb_frame(
    table: pd.DataFrame,
    *,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return one canonical, explicitly typed DataFrame for NWB storage."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != tuple(columns):
        raise ValueError("NWB table does not have its canonical schema.")
    output = table.copy().reset_index(drop=True)
    for column in text_columns:
        output[column] = output[column].map(str)
    for column in integer_columns:
        output[column] = pd.to_numeric(output[column], errors="raise").astype(
            np.int64
        )
    for column in boolean_columns:
        values = output[column].tolist()
        if not all(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"NWB column {column!r} must contain booleans.")
        output[column] = np.asarray(values, dtype=bool)
    for column in columns:
        if column in text_columns or column in integer_columns or column in boolean_columns:
            continue
        output[column] = pd.to_numeric(output[column], errors="raise").astype(
            float
        )
    return output.loc[:, list(columns)]


def _dynamic_table_from_frame(
    table: pd.DataFrame,
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> Any:
    """Convert one explicitly typed frame to an NWB DynamicTable."""
    from hdmf.common import DynamicTable

    canonical = _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
    )
    if canonical.empty:
        return _empty_nwb_dynamic_table(
            name=name,
            description=description,
            columns=columns,
            text_columns=text_columns,
            integer_columns=integer_columns,
            boolean_columns=boolean_columns,
        )
    return DynamicTable.from_dataframe(
        name=name,
        df=canonical,
        table_description=description,
        columns=[
            {"name": column, "description": _nwb_column_description(column)}
            for column in columns
        ],
    )


def _decode_nwb_text(value: Any) -> str:
    """Return one text value after an HDF5-backed DynamicTable roundtrip."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _decode_nwb_float(value: Any) -> float:
    """Parse one NWB text scalar with Python's exact float round-trip."""
    return float(_decode_nwb_text(value))


def _frame_from_dynamic_table(
    nwb_table: Any,
    *,
    expected_name: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Load one scalar DynamicTable or Spyglass-fetched DataFrame."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table.copy()
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != expected_name:
            raise ValueError(f"Unexpected swap-tuning NWB object {nwb_table.name!r}.")
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("Swap-tuning tabular NWB objects must be DynamicTables.")
    table = table.reset_index(drop=True)
    observed_columns = tuple(str(column) for column in table.columns)
    if set(observed_columns) != set(columns) or len(observed_columns) != len(columns):
        raise ValueError("Swap-tuning NWB object has a noncanonical schema.")
    table = table.loc[:, list(columns)]
    for column in text_columns:
        table[column] = table[column].map(_decode_nwb_text)
    return _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
    )


def swap_tuning_selected_units_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert the all-source-unit audit to one NWB DynamicTable."""
    return _dynamic_table_from_frame(
        table,
        name=NWB_SELECTED_UNITS_TABLE_NAME,
        description=(
            "All source-unit eligibility and score audit for empirical swap "
            f"tuning; v1ca1 NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=_SELECTED_UNIT_TEXT_COLUMNS,
        integer_columns=_SELECTED_UNIT_INTEGER_COLUMNS,
        boolean_columns=_SELECTED_UNIT_BOOLEAN_COLUMNS,
    )


def swap_tuning_selected_units_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load the all-source-unit audit from its NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_SELECTED_UNITS_TABLE_NAME,
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=_SELECTED_UNIT_TEXT_COLUMNS,
        integer_columns=_SELECTED_UNIT_INTEGER_COLUMNS,
        boolean_columns=_SELECTED_UNIT_BOOLEAN_COLUMNS,
    )


def swap_tuning_score_summary_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert the long-form score summary to one NWB DynamicTable."""
    return _dynamic_table_from_frame(
        table,
        name=NWB_SCORE_SUMMARY_TABLE_NAME,
        description=(
            "Long-form empirical swap-model scores and scalar evaluation "
            f"metadata; v1ca1 NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=SUMMARY_COLUMNS,
        text_columns=_SUMMARY_TEXT_COLUMNS,
        integer_columns=_SUMMARY_INTEGER_COLUMNS,
        boolean_columns=_SUMMARY_BOOLEAN_COLUMNS,
    )


def swap_tuning_score_summary_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load the long-form score summary from its NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_SCORE_SUMMARY_TABLE_NAME,
        columns=SUMMARY_COLUMNS,
        text_columns=_SUMMARY_TEXT_COLUMNS,
        integer_columns=_SUMMARY_INTEGER_COLUMNS,
        boolean_columns=_SUMMARY_BOOLEAN_COLUMNS,
    )


def _profile_rows(result: Mapping[str, Any], *, model_profiles: bool) -> pd.DataFrame:
    """Return identity-stable ragged profile rows from one canonical result."""
    canonical = validate_swap_tuning_curve_comparison_result(result)
    dataset = canonical["dataset"]
    eligible = canonical["selected_units"].loc[
        canonical["selected_units"]["eligible_for_comparison"].astype(bool)
    ].reset_index(drop=True)
    if not dataset.data_vars:
        columns = MODEL_PROFILE_COLUMNS if model_profiles else SOURCE_PROFILE_COLUMNS
        return pd.DataFrame(columns=columns)
    rows = []
    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        for unit_index, unit in eligible.iterrows():
            identity = {name: str(unit[name]) for name in IDENTITY_COLUMNS}
            if model_profiles:
                for model_index, model in enumerate(MODEL_NAMES):
                    rows.append(
                        {
                            "model": model,
                            "trajectory": trajectory,
                            **identity,
                            "model_tuning_hz": np.asarray(
                                dataset["model_tuning_hz"].values[
                                    model_index, trajectory_index, :, unit_index
                                ],
                                dtype=float,
                            ),
                        }
                    )
            else:
                rows.append(
                    {
                        "trajectory": trajectory,
                        **identity,
                        **{
                            name: np.asarray(
                                dataset[name].values[
                                    trajectory_index, :, unit_index
                                ],
                                dtype=float,
                            )
                            for name in SOURCE_PROFILE_VECTOR_COLUMNS
                        },
                    }
                )
    columns = MODEL_PROFILE_COLUMNS if model_profiles else SOURCE_PROFILE_COLUMNS
    return pd.DataFrame.from_records(rows, columns=columns)


def _ragged_profile_to_dynamic_table(
    table: pd.DataFrame,
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    vector_columns: Sequence[str],
) -> Any:
    """Convert scalar identities plus aligned vectors to one DynamicTable."""
    from hdmf.common import DynamicTable

    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != tuple(columns):
        raise ValueError("Swap-tuning profile table has a noncanonical schema.")
    scalar_columns = tuple(column for column in columns if column not in vector_columns)
    scalar = table.loc[:, list(scalar_columns)].copy().reset_index(drop=True)
    for column in scalar_columns:
        scalar[column] = scalar[column].map(str)
    if table.empty:
        return _empty_nwb_dynamic_table(
            name=name,
            description=description,
            columns=columns,
            text_columns=scalar_columns,
            ragged_columns=vector_columns,
        )
    output = DynamicTable.from_dataframe(
        name=name,
        df=scalar,
        table_description=description,
        columns=[
            {"name": column, "description": _nwb_column_description(column)}
            for column in scalar_columns
        ],
    )
    for column in vector_columns:
        vectors = [np.asarray(value, dtype=float) for value in table[column]]
        if any(vector.ndim != 1 or np.isinf(vector).any() for vector in vectors):
            raise ValueError(f"Swap-tuning profile column {column!r} is invalid.")
        if all(vector.size == 0 for vector in vectors):
            # HDMF cannot encode a ragged column with nonzero rows and zero
            # flattened values. A single NaN is an explicit empty-vector
            # sentinel; the inverse converter removes it before validation.
            vectors = [np.asarray([np.nan], dtype=float) for _ in vectors]
        output.add_column(
            name=column,
            description=_nwb_column_description(column),
            data=vectors,
            index=True,
        )
    return output


def _ragged_profile_from_dynamic_table(
    nwb_table: Any,
    *,
    expected_name: str,
    columns: Sequence[str],
    vector_columns: Sequence[str],
) -> pd.DataFrame:
    """Load identity-stable ragged profile rows from one NWB object."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table.copy()
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != expected_name:
            raise ValueError(f"Unexpected swap-tuning NWB object {nwb_table.name!r}.")
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("Swap-tuning profile NWB objects must be DynamicTables.")
    table = table.reset_index(drop=True)
    observed_columns = tuple(str(column) for column in table.columns)
    if set(observed_columns) != set(columns) or len(observed_columns) != len(columns):
        raise ValueError("Swap-tuning profile NWB object has a noncanonical schema.")
    table = table.loc[:, list(columns)]
    for column in columns:
        if column in vector_columns:
            table[column] = [np.asarray(value, dtype=float) for value in table[column]]
            if table[column].map(
                lambda value: value.shape == (1,) and np.isnan(value[0])
            ).all():
                table[column] = [np.asarray([], dtype=float) for _ in table[column]]
            if any(value.ndim != 1 or np.isinf(value).any() for value in table[column]):
                raise ValueError(f"Swap-tuning profile column {column!r} is invalid.")
        else:
            table[column] = table[column].map(_decode_nwb_text)
    return table.loc[:, list(columns)]


def swap_tuning_source_profiles_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store the four source tuning profiles for every path and eligible unit."""
    return _ragged_profile_to_dynamic_table(
        _profile_rows(result, model_profiles=False),
        name=NWB_SOURCE_PROFILES_TABLE_NAME,
        description=(
            "Dark, light-train, and held-out light source tuning profiles on "
            "the normalized progression grid."
        ),
        columns=SOURCE_PROFILE_COLUMNS,
        vector_columns=SOURCE_PROFILE_VECTOR_COLUMNS,
    )


def swap_tuning_source_profiles_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load the per-path source tuning profiles from their NWB object."""
    return _ragged_profile_from_dynamic_table(
        nwb_table,
        expected_name=NWB_SOURCE_PROFILES_TABLE_NAME,
        columns=SOURCE_PROFILE_COLUMNS,
        vector_columns=SOURCE_PROFILE_VECTOR_COLUMNS,
    )


def swap_tuning_model_profiles_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store every empirical model tuning profile by model, path, and unit."""
    return _ragged_profile_to_dynamic_table(
        _profile_rows(result, model_profiles=True),
        name=NWB_MODEL_PROFILES_TABLE_NAME,
        description=(
            "Empirical swap-model tuning profiles on the normalized progression "
            "grid."
        ),
        columns=MODEL_PROFILE_COLUMNS,
        vector_columns=("model_tuning_hz",),
    )


def swap_tuning_model_profiles_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load empirical model tuning profiles from their NWB object."""
    return _ragged_profile_from_dynamic_table(
        nwb_table,
        expected_name=NWB_MODEL_PROFILES_TABLE_NAME,
        columns=MODEL_PROFILE_COLUMNS,
        vector_columns=("model_tuning_hz",),
    )


def _geometry_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return four rows describing the shared grid and swap-segment geometry."""
    canonical = validate_swap_tuning_curve_comparison_result(result)
    dataset = canonical["dataset"]
    evaluated = bool(dataset.data_vars)
    if evaluated:
        segment_edges = np.asarray(dataset["segment_edges"].values, dtype=float)
        tp_bin = np.asarray(dataset.coords["tp_bin"].values, dtype=float)
    else:
        segment_edges = np.asarray(
            json.loads(str(dataset.attrs["segment_edges_json"])), dtype=float
        )
        tp_bin = np.asarray([], dtype=float)
    rows = []
    for index, trajectory in enumerate(TRAJECTORY_TYPES):
        segment_index = int(SWAP_CONFIGURATION[trajectory]["segment_index"])
        rows.append(
            {
                "trajectory": trajectory,
                "swap_source_trajectory": SWAP_CONFIGURATION[trajectory][
                    "source_trajectory"
                ],
                "swap_segment_index_1based": segment_index + 1,
                "swap_segment_start": float(segment_edges[segment_index]),
                "swap_segment_end": float(segment_edges[segment_index + 1]),
                "test_light_bin_count": (
                    float(dataset["test_light_bin_count"].values[index])
                    if evaluated
                    else np.nan
                ),
                "test_light_duration_s": (
                    float(dataset["test_light_duration_s"].values[index])
                    if evaluated
                    else np.nan
                ),
                "tp_bin": tp_bin.copy(),
                "segment_edges": segment_edges.copy(),
                "segment_bin_mask": (
                    np.asarray(dataset["segment_bin_mask"].values[index], dtype=float)
                    if evaluated
                    else np.asarray([], dtype=float)
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=GEOMETRY_COLUMNS)


def swap_tuning_geometry_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store progression-bin and swap-segment geometry in one DynamicTable."""
    return _ragged_profile_to_dynamic_table(
        _geometry_frame(result),
        name=NWB_GEOMETRY_TABLE_NAME,
        description=(
            "Normalized progression grid, swap segments, scoring support, and "
            "segment masks for each path."
        ),
        columns=GEOMETRY_COLUMNS,
        vector_columns=GEOMETRY_VECTOR_COLUMNS,
    )


def swap_tuning_geometry_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load and validate the four-row swap geometry object."""
    table = _ragged_profile_from_dynamic_table(
        nwb_table,
        expected_name=NWB_GEOMETRY_TABLE_NAME,
        columns=GEOMETRY_COLUMNS,
        vector_columns=GEOMETRY_VECTOR_COLUMNS,
    )
    for column in ("swap_segment_index_1based",):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(np.int64)
    for column in (
        "swap_segment_start",
        "swap_segment_end",
        "test_light_bin_count",
        "test_light_duration_s",
    ):
        table[column] = table[column].map(_decode_nwb_float).astype(float)
    if table["trajectory"].tolist() != list(TRAJECTORY_TYPES):
        raise ValueError("Swap-tuning geometry paths are not in canonical order.")
    return table.loc[:, list(GEOMETRY_COLUMNS)]


def _json_ready(value: Any) -> Any:
    """Return nested metadata using JSON-native scalar and sequence types."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(current) for key, current in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(current) for current in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if value is None:
        return None
    return str(value)


def _provenance_record(result: Mapping[str, Any]) -> dict[str, str]:
    """Return one detached, self-describing swap-tuning provenance record."""
    canonical = validate_swap_tuning_curve_comparison_result(result)
    return {
        "metadata_json": json.dumps(
            _json_ready(canonical["metadata"]), sort_keys=True, separators=(",", ":")
        ),
        "parameters_json": json.dumps(
            _json_ready(canonical["parameters"]), sort_keys=True, separators=(",", ":")
        ),
        "upstream_provenance_json": json.dumps(
            _json_ready(canonical["upstream_provenance"]),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "dataset_attrs_json": json.dumps(
            _json_ready(dict(canonical["dataset"].attrs)),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "analysis_status": str(canonical["analysis_status"]),
        "artifact_origin": str(canonical["artifact_origin"]),
        "legacy_artifact_provenance_json": json.dumps(
            _json_ready(canonical["legacy_artifact_provenance"] or {}),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
    }


def swap_tuning_provenance_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Convert result provenance to a single-row NWB DynamicTable."""
    return _dynamic_table_from_frame(
        pd.DataFrame.from_records([_provenance_record(result)], columns=PROVENANCE_COLUMNS),
        name=NWB_PROVENANCE_TABLE_NAME,
        description=(
            "Detached swap-tuning selection, parameter, source, and dataset "
            f"provenance; v1ca1 NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=PROVENANCE_COLUMNS,
        text_columns=PROVENANCE_COLUMNS,
    )


def swap_tuning_provenance_from_dynamic_table(nwb_table: Any) -> dict[str, Any]:
    """Load and parse one swap-tuning provenance record."""
    table = _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_PROVENANCE_TABLE_NAME,
        columns=PROVENANCE_COLUMNS,
        text_columns=PROVENANCE_COLUMNS,
    )
    if len(table) != 1:
        raise ValueError("Swap-tuning provenance must contain exactly one row.")
    record = table.iloc[0].to_dict()
    if record["artifact_schema_version"] != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Swap-tuning NWB artifact schema version is unsupported.")
    try:
        metadata = json.loads(record["metadata_json"])
        parameters = json.loads(record["parameters_json"])
        upstream = json.loads(record["upstream_provenance_json"])
        dataset_attrs = json.loads(record["dataset_attrs_json"])
        legacy = json.loads(record["legacy_artifact_provenance_json"])
    except json.JSONDecodeError as exc:
        raise ValueError("Swap-tuning NWB provenance contains malformed JSON.") from exc
    if not all(
        isinstance(value, Mapping)
        for value in (metadata, parameters, upstream, dataset_attrs, legacy)
    ):
        raise ValueError("Swap-tuning NWB provenance JSON must encode mappings.")
    return {
        "metadata": dict(metadata),
        "parameters": dict(parameters),
        "upstream_provenance": dict(upstream),
        "dataset_attrs": dict(dataset_attrs),
        "analysis_status": record["analysis_status"],
        "artifact_origin": record["artifact_origin"],
        "legacy_artifact_provenance": dict(legacy) or None,
    }


def swap_tuning_curve_comparison_result_to_nwb_objects(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert one complete canonical result to six NWB scratch objects."""
    canonical = validate_swap_tuning_curve_comparison_result(result)
    return {
        "selected_units": swap_tuning_selected_units_to_dynamic_table(
            canonical["selected_units"]
        ),
        "score_summary": swap_tuning_score_summary_to_dynamic_table(
            canonical["summary"]
        ),
        "source_profiles": swap_tuning_source_profiles_to_dynamic_table(canonical),
        "model_profiles": swap_tuning_model_profiles_to_dynamic_table(canonical),
        "geometry": swap_tuning_geometry_to_dynamic_table(canonical),
        "provenance": swap_tuning_provenance_to_dynamic_table(canonical),
    }


def _require_profile_coverage(
    table: pd.DataFrame,
    *,
    selected_units: pd.DataFrame,
    include_model: bool,
) -> None:
    """Require each eligible unit/path/model profile exactly once."""
    eligible_ids = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool), "stable_unit_id"
    ].astype(str)
    expected = {
        ((model,) if include_model else ()) + (trajectory, stable_unit_id)
        for model in (MODEL_NAMES if include_model else (None,))
        for trajectory in TRAJECTORY_TYPES
        for stable_unit_id in eligible_ids
    }
    if include_model:
        observed = set(
            zip(
                table["model"].astype(str),
                table["trajectory"].astype(str),
                table["stable_unit_id"].astype(str),
                strict=True,
            )
        )
    else:
        observed = set(
            zip(
                table["trajectory"].astype(str),
                table["stable_unit_id"].astype(str),
                strict=True,
            )
        )
    if len(observed) != len(table) or observed != expected:
        raise ValueError("Swap-tuning NWB profiles do not cover the expected grid.")


def _dataset_from_nwb_frames(
    *,
    selected_units: pd.DataFrame,
    summary: pd.DataFrame,
    source_profiles: pd.DataFrame,
    model_profiles: pd.DataFrame,
    geometry: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> Any:
    """Reconstruct the exact scientific xarray dataset from NWB tables."""
    import xarray as xr

    eligible = selected_units.loc[
        selected_units["eligible_for_comparison"].astype(bool)
    ].reset_index(drop=True)
    status = str(provenance["analysis_status"])
    attrs = dict(provenance["dataset_attrs"])
    segment_edges = np.asarray(geometry.iloc[0]["segment_edges"], dtype=float)
    if any(
        not np.array_equal(np.asarray(value, dtype=float), segment_edges)
        for value in geometry["segment_edges"]
    ):
        raise ValueError("Swap-tuning geometry rows disagree on segment edges.")
    coords = {
        "model": np.asarray(MODEL_NAMES, dtype=str),
        "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
        "unit": np.asarray(eligible["stable_unit_id"], dtype=str),
        "segment_edge": np.arange(len(segment_edges), dtype=np.int64),
        **{
            name: ("unit", np.asarray(eligible[name], dtype=str))
            for name in IDENTITY_COLUMNS
        },
    }
    evaluated = status in {"valid", "partial_valid", "no_valid_units"}
    if not evaluated:
        if not summary.empty or not source_profiles.empty or not model_profiles.empty:
            raise ValueError("Terminal swap-tuning NWB objects contain profiles or scores.")
        return xr.Dataset(coords=coords, attrs=attrs)

    _require_profile_coverage(
        source_profiles, selected_units=selected_units, include_model=False
    )
    _require_profile_coverage(
        model_profiles, selected_units=selected_units, include_model=True
    )
    tp_bin = np.asarray(geometry.iloc[0]["tp_bin"], dtype=float)
    if any(
        not np.array_equal(np.asarray(value, dtype=float), tp_bin)
        for value in geometry["tp_bin"]
    ):
        raise ValueError("Swap-tuning geometry rows disagree on progression bins.")
    n_models = len(MODEL_NAMES)
    n_trajectories = len(TRAJECTORY_TYPES)
    n_units = len(eligible)
    n_bins = len(tp_bin)
    model_index = {name: index for index, name in enumerate(MODEL_NAMES)}
    trajectory_index = {
        name: index for index, name in enumerate(TRAJECTORY_TYPES)
    }
    unit_index = {
        str(value): index for index, value in enumerate(eligible["stable_unit_id"])
    }
    ll_arrays = {
        name: np.full((n_models, n_trajectories, n_units), np.nan, dtype=float)
        for name in ("ll_sum", "ll_bits_per_spike", "ll_bits_per_s")
    }
    scalar_arrays = {
        name: np.full((n_trajectories, n_units), np.nan, dtype=float)
        for name in (
            "train_dark_same_rate_hz",
            "train_dark_other_rate_hz",
            "train_light_other_rate_hz",
            "test_light_target_rate_hz",
            "test_light_spike_sum",
        )
    }
    for row in summary.to_dict("records"):
        model_i = model_index[str(row["model"])]
        trajectory_i = trajectory_index[str(row["trajectory"])]
        unit_i = unit_index[str(row["stable_unit_id"])]
        for name, values in ll_arrays.items():
            values[model_i, trajectory_i, unit_i] = float(row[name])
        for name, values in scalar_arrays.items():
            observed = float(row[name])
            previous = values[trajectory_i, unit_i]
            if np.isfinite(previous) and not np.isclose(
                previous, observed, rtol=0.0, atol=1e-12, equal_nan=True
            ):
                raise ValueError("Swap-tuning score rows disagree on scalar metadata.")
            values[trajectory_i, unit_i] = observed
    source_arrays = {
        name: np.full((n_trajectories, n_bins, n_units), np.nan, dtype=float)
        for name in SOURCE_PROFILE_VECTOR_COLUMNS
    }
    for row in source_profiles.to_dict("records"):
        trajectory_i = trajectory_index[str(row["trajectory"])]
        unit_i = unit_index[str(row["stable_unit_id"])]
        for name, values in source_arrays.items():
            profile = np.asarray(row[name], dtype=float)
            if profile.shape != (n_bins,):
                raise ValueError("Swap-tuning source profile has the wrong length.")
            values[trajectory_i, :, unit_i] = profile
    model_tuning = np.full(
        (n_models, n_trajectories, n_bins, n_units), np.nan, dtype=float
    )
    for row in model_profiles.to_dict("records"):
        profile = np.asarray(row["model_tuning_hz"], dtype=float)
        if profile.shape != (n_bins,):
            raise ValueError("Swap-tuning model profile has the wrong length.")
        model_tuning[
            model_index[str(row["model"])],
            trajectory_index[str(row["trajectory"])],
            :,
            unit_index[str(row["stable_unit_id"])],
        ] = profile
    masks = np.stack(
        [np.asarray(value, dtype=bool) for value in geometry["segment_bin_mask"]]
    )
    if masks.shape != (n_trajectories, n_bins):
        raise ValueError("Swap-tuning segment masks have the wrong shape.")
    coords["tp_bin"] = tp_bin
    data_vars = {
        "dark_train_movement_firing_rate_hz": (
            "unit",
            eligible["dark_movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "light_train_movement_firing_rate_hz": (
            "unit",
            eligible["light_train_movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "light_test_movement_firing_rate_hz": (
            "unit",
            eligible["light_test_movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        **{
            name: (("model", "trajectory", "unit"), values)
            for name, values in ll_arrays.items()
        },
        "model_tuning_hz": (
            ("model", "trajectory", "tp_bin", "unit"),
            model_tuning,
        ),
        **{
            name: (("trajectory", "unit"), values)
            for name, values in scalar_arrays.items()
        },
        **{
            name: (("trajectory", "tp_bin", "unit"), values)
            for name, values in source_arrays.items()
        },
        "segment_bin_mask": (("trajectory", "tp_bin"), masks),
        "swap_source_trajectory": (
            "trajectory",
            geometry["swap_source_trajectory"].to_numpy(dtype=str),
        ),
        "swap_segment_index_1based": (
            "trajectory",
            geometry["swap_segment_index_1based"].to_numpy(dtype=np.int64),
        ),
        "swap_segment_start": (
            "trajectory",
            geometry["swap_segment_start"].to_numpy(dtype=float),
        ),
        "swap_segment_end": (
            "trajectory",
            geometry["swap_segment_end"].to_numpy(dtype=float),
        ),
        "test_light_bin_count": (
            "trajectory",
            geometry["test_light_bin_count"].to_numpy(dtype=float),
        ),
        "test_light_duration_s": (
            "trajectory",
            geometry["test_light_duration_s"].to_numpy(dtype=float),
        ),
        "segment_edges": ("segment_edge", segment_edges),
    }
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def swap_tuning_curve_comparison_result_from_nwb_objects(
    *,
    selected_units: Any,
    score_summary: Any,
    source_profiles: Any,
    model_profiles: Any,
    geometry: Any,
    provenance: Any,
) -> dict[str, Any]:
    """Reconstruct and validate one result from its six NWB scratch objects."""
    provenance_record = swap_tuning_provenance_from_dynamic_table(provenance)
    selected_table = swap_tuning_selected_units_from_dynamic_table(selected_units)
    summary_table = swap_tuning_score_summary_from_dynamic_table(score_summary)
    source_table = swap_tuning_source_profiles_from_dynamic_table(source_profiles)
    model_table = swap_tuning_model_profiles_from_dynamic_table(model_profiles)
    geometry_table = swap_tuning_geometry_from_dynamic_table(geometry)
    dataset = _dataset_from_nwb_frames(
        selected_units=selected_table,
        summary=summary_table,
        source_profiles=source_table,
        model_profiles=model_table,
        geometry=geometry_table,
        provenance=provenance_record,
    )
    return validate_swap_tuning_curve_comparison_result(
        {
            "metadata": provenance_record["metadata"],
            "parameters": provenance_record["parameters"],
            "upstream_provenance": provenance_record["upstream_provenance"],
            "selected_units": selected_table,
            "summary": summary_table,
            "dataset": dataset,
            "analysis_status": provenance_record["analysis_status"],
            "artifact_origin": provenance_record["artifact_origin"],
            "legacy_artifact_provenance": provenance_record[
                "legacy_artifact_provenance"
            ],
        }
    )


def _float_array_sha256(values: Any) -> str:
    """Return a deterministic digest for one float array, including shape."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    if np.isnan(array).any():
        array = array.copy()
        array[np.isnan(array)] = np.nan
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _ragged_frame_sha256(
    table: pd.DataFrame,
    *,
    vector_columns: Sequence[str],
) -> str:
    """Hash scalar row identity and every ordered ragged vector."""
    scalar_columns = [column for column in table.columns if column not in vector_columns]
    scalar_table = table.loc[:, scalar_columns].copy()
    for column in scalar_table.select_dtypes(include=[np.floating]).columns:
        values = scalar_table[column].to_numpy(copy=True)
        values[np.isnan(values)] = np.nan
        scalar_table[column] = values
    vectors = {
        column: [
            _float_array_sha256(np.asarray(value, dtype=float))
            for value in table[column]
        ]
        for column in vector_columns
    }
    return _provenance_sha256(
        {
            "scalar_sha256": _table_sha256(scalar_table),
            "vectors": vectors,
        }
    )


def swap_tuning_curve_comparison_nwb_hashes(
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Return storage-independent hashes for all six NWB scratch objects."""
    canonical = validate_swap_tuning_curve_comparison_result(result)
    source_profiles = _profile_rows(canonical, model_profiles=False)
    model_profiles = _profile_rows(canonical, model_profiles=True)
    geometry = _geometry_frame(canonical)
    return {
        "selected_units_table_sha256": _table_sha256(
            canonical["selected_units"]
        ),
        "score_summary_sha256": _table_sha256(canonical["summary"]),
        "source_profiles_sha256": _ragged_frame_sha256(
            source_profiles, vector_columns=SOURCE_PROFILE_VECTOR_COLUMNS
        ),
        "model_profiles_sha256": _ragged_frame_sha256(
            model_profiles, vector_columns=("model_tuning_hz",)
        ),
        "geometry_sha256": _ragged_frame_sha256(
            geometry, vector_columns=GEOMETRY_VECTOR_COLUMNS
        ),
        "provenance_sha256": _provenance_sha256(_provenance_record(canonical)),
    }


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return manifest fields shared by every bundle file."""
    metadata = result["metadata"]
    parameters = result["parameters"]
    return {
        "swap_tuning_curve_comparison_id": metadata[
            "swap_tuning_curve_comparison_id"
        ],
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_epoch": metadata["dark_epoch"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "upstream_provenance_json": json.dumps(
            result["upstream_provenance"],
            sort_keys=True,
        ),
        "n_source_units": result["n_source_units"],
        "n_units": result["n_units"],
        "n_valid_units": result["n_valid_units"],
        "selected_units_sha256": result["selected_units_sha256"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": json.dumps(
            result["legacy_artifact_provenance"] or {},
            sort_keys=True,
        ),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def write_swap_tuning_curve_comparison_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write and reload one complete empirical swap bundle."""
    result = validate_swap_tuning_curve_comparison_result(result)
    destination = Path(path)
    if destination.name != result["metadata"]["swap_tuning_curve_comparison_id"]:
        raise ValueError(
            "Artifact directory name must equal swap_tuning_curve_comparison_id."
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite swap-tuning artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        result["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME,
            index=False,
        )
        result["summary"].to_parquet(temporary / SUMMARY_FILENAME, index=False)
        result["dataset"].to_netcdf(temporary / RESULT_FILENAME)
        common = _manifest_common(result)
        rows = []
        for key, filename, kind in (
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("summary", SUMMARY_FILENAME, "parquet"),
            ("swap_tuning", RESULT_FILENAME, "netcdf"),
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
            temporary / MANIFEST_FILENAME,
            index=False,
        )
        load_swap_tuning_curve_comparison_artifact(
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
    return {
        "artifact_dir": destination,
        "artifact_manifest_path": destination / MANIFEST_FILENAME,
        "selected_units_path": destination / SELECTED_UNITS_FILENAME,
        "summary_path": destination / SUMMARY_FILENAME,
        "result_path": destination / RESULT_FILENAME,
    }


def _load_dataset(path: Path) -> Any:
    """Load one NetCDF dataset eagerly and close its backing file."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def load_swap_tuning_curve_comparison_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one complete empirical swap bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Swap-tuning manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or len(manifest) != 3:
        raise ValueError("Swap-tuning manifest does not have the canonical schema.")
    expected = {
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "summary": (SUMMARY_FILENAME, "parquet"),
        "swap_tuning": (RESULT_FILENAME, "netcdf"),
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("Swap-tuning manifest lacks canonical artifacts.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("Swap-tuning manifest names or kinds are stale.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Swap-tuning artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(f"Swap-tuning checksum mismatch: {artifact_path}")
    first = manifest.iloc[0]
    for name in MANIFEST_COLUMNS[5:]:
        if not np.all(manifest[name].astype(str) == str(first[name])):
            raise ValueError(f"Swap-tuning manifest has inconsistent {name!r}.")
    metadata = {
        name: str(first[name])
        for name in (
            "swap_tuning_curve_comparison_id",
            "animal_name",
            "date",
            "region",
            "dark_epoch",
            "light_train_epoch",
            "light_test_epoch",
        )
    }
    if not _allow_temporary_name and directory.name != metadata[
        "swap_tuning_curve_comparison_id"
    ]:
        raise ValueError("Artifact directory name does not match result UUID.")
    dataset = _load_dataset(directory / RESULT_FILENAME)
    parameters = {
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        **{
            name: float(dataset.attrs[name])
            for name in (
                "evaluation_bin_size_s",
                "gaussian_smoothing_sigma_bins",
                "min_dark_firing_rate_hz",
                "min_light_firing_rate_hz",
            )
        },
    }
    legacy = json.loads(str(first["legacy_artifact_provenance_json"]))
    result = validate_swap_tuning_curve_comparison_result(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": json.loads(
                str(first["upstream_provenance_json"])
            ),
            "selected_units": pd.read_parquet(
                directory / SELECTED_UNITS_FILENAME
            ),
            "summary": pd.read_parquet(directory / SUMMARY_FILENAME),
            "dataset": dataset,
            "analysis_status": str(first["analysis_status"]),
            "artifact_origin": str(first["artifact_origin"]),
            "legacy_artifact_provenance": legacy or None,
        }
    )
    if (
        result["n_source_units"] != int(first["n_source_units"])
        or result["n_units"] != int(first["n_units"])
        or result["n_valid_units"] != int(first["n_valid_units"])
        or result["selected_units_sha256"] != str(first["selected_units_sha256"])
    ):
        raise ValueError("Swap-tuning manifest summary is stale.")
    result["manifest"] = manifest
    return result


LEGACY_SUMMARY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "dark_train_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "trajectory",
    "unit",
    "apply_fr_filter",
    "min_dark_fr_hz",
    "min_light_fr_hz",
    "dark_train_movement_firing_rate_hz",
    "light_train_movement_firing_rate_hz",
    "swap_source_trajectory",
    "swap_segment_index_1based",
    "swap_segment_start",
    "swap_segment_end",
    "train_dark_same_rate_hz",
    "train_dark_other_rate_hz",
    "train_light_other_rate_hz",
    "test_light_target_rate_hz",
    "test_light_spike_sum",
    "test_light_bin_count",
    "test_light_duration_s",
    "model",
    "ll_sum",
    "ll_bits_per_spike",
    "ll_bits_per_s",
)


def _strict_boolean(value: Any, *, name: str) -> bool:
    """Return one legacy NetCDF-compatible boolean scalar."""
    if isinstance(value, str) and value in {"0", "1"}:
        return value == "1"
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"Legacy {name} must encode one boolean.")


def _validate_legacy_dataset_contract(
    dataset: Any,
    *,
    metadata: Mapping[str, str],
    parameters: Mapping[str, Any],
) -> None:
    """Require the complete supported schema-2 legacy scientific contract."""
    if str(dataset.attrs.get("schema_version", "")) != LEGACY_RESULT_SCHEMA_VERSION:
        raise ValueError("Legacy swap-tuning result must use schema version 2.")
    if set(dataset.data_vars) != set(SCIENTIFIC_VARIABLE_DIMS).difference(
        {"light_test_movement_firing_rate_hz"}
    ):
        raise ValueError("Legacy swap-tuning NetCDF does not have all 23 variables.")
    for name, dims in SCIENTIFIC_VARIABLE_DIMS.items():
        if name == "light_test_movement_firing_rate_hz":
            continue
        if tuple(dataset[name].dims) != dims:
            raise ValueError(f"Legacy swap-tuning variable {name!r} has stale dims.")
    for coordinate, expected in {
        "model": MODEL_NAMES,
        "trajectory": TRAJECTORY_TYPES,
    }.items():
        if not np.array_equal(
            np.asarray(dataset.coords[coordinate].values, dtype=str),
            np.asarray(expected, dtype=str),
        ):
            raise ValueError(f"Legacy swap-tuning {coordinate} order is stale.")
    for name, expected in {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_train_epoch": metadata["dark_epoch"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
        "scoring_scope": "light_test_swapped_segment_only",
        "training_tuning_scope": "full_trajectory_movement_interval",
    }.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"Legacy swap-tuning has mismatched {name!r}.")
    for name, expected in {
        "bin_size_s": parameters["evaluation_bin_size_s"],
        "sigma_bins": parameters["gaussian_smoothing_sigma_bins"],
        "place_bin_size_cm": REQUIRED_UPSTREAM_BIN_SIZE_CM,
        "min_dark_fr_hz": parameters["min_dark_firing_rate_hz"],
        "min_light_fr_hz": parameters["min_light_firing_rate_hz"],
    }.items():
        if not np.isclose(
            float(dataset.attrs.get(name, np.nan)),
            float(expected),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Legacy swap-tuning has mismatched {name!r}.")
    if not _strict_boolean(
        dataset.attrs.get("apply_fr_filter"),
        name="apply_fr_filter",
    ):
        raise ValueError("Legacy swap-tuning must apply the train firing-rate filter.")
    try:
        swap_rule = json.loads(str(dataset.attrs["swap_rule_json"]))
        formulas = json.loads(str(dataset.attrs["empirical_model_formulas_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Legacy swap-tuning formula provenance is invalid.") from exc
    if swap_rule != SWAP_CONFIGURATION:
        raise ValueError("Legacy swap-tuning swap rule is stale.")
    if formulas != EMPIRICAL_MODEL_FORMULAS:
        raise ValueError("Legacy empirical model formulas are stale.")


def _resolve_legacy_identity(
    legacy_unit_id: Any,
    resolver: Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]],
) -> dict[str, str]:
    """Resolve one raw imported-sorting unit to persistent identity."""
    if isinstance(resolver, Mapping):
        matches = [
            value
            for key, value in resolver.items()
            if str(key) == str(legacy_unit_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Legacy unit {legacy_unit_id!r} must resolve exactly once."
            )
        identity = matches[0]
    elif callable(resolver):
        try:
            identity = resolver(legacy_unit_id)
        except LookupError as exc:
            raise ValueError(
                f"Legacy unit {legacy_unit_id!r} could not be resolved."
            ) from exc
    else:
        raise TypeError("unit_identity_resolver must be a mapping or callable.")
    if not isinstance(identity, Mapping):
        raise TypeError("Resolved legacy unit identity must be a mapping.")
    missing = sorted(
        {"spikesorting_merge_id", "unit_id"}.difference(identity)
    )
    if missing:
        raise ValueError(f"Resolved legacy identity is missing {missing!r}.")
    if "sorting_unit_id" in identity and str(identity["sorting_unit_id"]) != str(
        legacy_unit_id
    ):
        raise ValueError("Resolved sorting_unit_id conflicts with legacy unit.")
    merge_id = str(identity["spikesorting_merge_id"])
    unit_id = str(identity["unit_id"])
    return {
        "spikesorting_merge_id": merge_id,
        "unit_id": unit_id,
        "stable_unit_id": f"{merge_id}:{unit_id}",
        "group_unit_id": str(identity.get("group_unit_id", legacy_unit_id)),
    }


def _legacy_identity_alignment(
    dataset: Any,
    *,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    expected_selected_units: pd.DataFrame,
) -> tuple[Any, dict[str, str]]:
    """Map/reorder raw legacy unit coordinates to canonical eligible identities."""
    legacy_ids = list(np.asarray(dataset.coords["unit"].values).reshape(-1))
    if len({str(value) for value in legacy_ids}) != len(legacy_ids):
        raise ValueError("Legacy swap-tuning unit coordinates must be unique.")
    resolved = [
        _resolve_legacy_identity(value, unit_identity_resolver)
        for value in legacy_ids
    ]
    stable_to_index = {
        row["stable_unit_id"]: index for index, row in enumerate(resolved)
    }
    if len(stable_to_index) != len(resolved):
        raise ValueError("Resolved legacy persistent identities must be unique.")
    eligible = expected_selected_units.loc[
        expected_selected_units["eligible_for_comparison"].astype(bool)
    ].reset_index(drop=True)
    expected_ids = eligible["stable_unit_id"].astype(str).tolist()
    if set(stable_to_index) != set(expected_ids):
        raise ValueError(
            "Legacy selected units do not exactly match recomputed eligible units."
        )
    order = [stable_to_index[stable_id] for stable_id in expected_ids]
    ordered = dataset.isel(unit=order).assign_coords(
        unit=np.asarray(expected_ids, dtype=str)
    )
    mapping = {
        str(legacy_ids[index]): expected_ids[output_index]
        for output_index, index in enumerate(order)
    }
    return ordered, mapping


def _compare_scientific_datasets(source: Any, recomputed: Any) -> None:
    """Require every legacy scientific coordinate and variable to match."""
    for name in ("model", "trajectory", "unit", "tp_bin", "segment_edge"):
        source_values = np.asarray(source.coords[name].values)
        expected_values = np.asarray(recomputed.coords[name].values)
        if source_values.dtype.kind in "OUS" or expected_values.dtype.kind in "OUS":
            matches = np.array_equal(
                source_values.astype(str),
                expected_values.astype(str),
            )
        else:
            matches = np.allclose(
                source_values.astype(float),
                expected_values.astype(float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )
        if not matches:
            raise ValueError(
                f"Legacy swap-tuning coordinate {name!r} differs from NWB re-score."
            )
    for name in sorted(
        set(SCIENTIFIC_VARIABLE_DIMS).difference(
            {"light_test_movement_firing_rate_hz"}
        )
    ):
        source_values = np.asarray(source[name].values)
        expected_values = np.asarray(recomputed[name].values)
        if source_values.dtype.kind in "OUSb" or expected_values.dtype.kind in "OUSb":
            matches = np.array_equal(
                source_values.astype(str),
                expected_values.astype(str),
            )
        else:
            matches = np.allclose(
                source_values.astype(float),
                expected_values.astype(float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )
        if not matches:
            raise ValueError(
                f"Legacy swap-tuning variable {name!r} differs from NWB re-score."
            )


def _compare_legacy_summary(
    source: pd.DataFrame,
    *,
    recomputed: pd.DataFrame,
    legacy_to_stable_id: Mapping[str, str],
) -> None:
    """Require every legacy Parquet value to match the canonical summary."""
    if tuple(source.columns) != LEGACY_SUMMARY_COLUMNS:
        raise ValueError("Legacy swap-tuning summary does not have 28 columns.")
    if len(source) != len(recomputed):
        raise ValueError("Legacy swap-tuning summary has a stale row count.")
    normalized = source.copy()
    normalized["unit"] = normalized["unit"].map(
        lambda value: legacy_to_stable_id.get(str(value), "")
    )
    if (normalized["unit"] == "").any():
        raise ValueError("Legacy summary contains an unresolved unit.")
    expected = pd.DataFrame(
        {
            "animal_name": recomputed["animal_name"],
            "date": recomputed["date"],
            "region": recomputed["region"],
            "dark_train_epoch": recomputed["dark_train_epoch"],
            "light_train_epoch": recomputed["light_train_epoch"],
            "light_test_epoch": recomputed["light_test_epoch"],
            "trajectory": recomputed["trajectory"],
            "unit": recomputed["stable_unit_id"],
            "apply_fr_filter": True,
            "min_dark_fr_hz": recomputed["min_dark_firing_rate_hz"],
            "min_light_fr_hz": recomputed["min_light_firing_rate_hz"],
            **{
                name: recomputed[name]
                for name in LEGACY_SUMMARY_COLUMNS[11:24]
            },
            "model": recomputed["model"],
            "ll_sum": recomputed["ll_sum"],
            "ll_bits_per_spike": recomputed["ll_bits_per_spike"],
            "ll_bits_per_s": recomputed["ll_bits_per_s"],
        },
        columns=LEGACY_SUMMARY_COLUMNS,
    )
    for name in LEGACY_SUMMARY_COLUMNS:
        observed_values = normalized[name].to_numpy()
        expected_values = expected[name].to_numpy()
        if observed_values.dtype.kind in "OUSb" or expected_values.dtype.kind in "OUSb":
            matches = np.array_equal(
                observed_values.astype(str),
                expected_values.astype(str),
            )
        else:
            matches = np.allclose(
                observed_values.astype(float),
                expected_values.astype(float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
            )
        if not matches:
            raise ValueError(
                f"Legacy swap-tuning summary column {name!r} differs from NWB re-score."
            )


def register_existing_swap_tuning_curve_comparison_artifact(
    *,
    source_result_path: Path,
    source_summary_path: Path,
    destination_path: Path | None,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    source_sorting_type: str,
    swap_tuning_curve_comparison_id: Any,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    tuning_curve_artifact_paths: Mapping[str, Mapping[str, Path]] | None,
    movement_firing_rate_tables_by_role: Mapping[str, pd.DataFrame],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    position_offset_samples: int,
    movement_interval: Any,
    movement_analysis_status: str,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    evaluation_bin_size_s: float = DEFAULT_EVALUATION_BIN_SIZE_S,
    gaussian_smoothing_sigma_bins: float = DEFAULT_GAUSSIAN_SMOOTHING_SIGMA_BINS,
    min_dark_firing_rate_hz: float = DEFAULT_MIN_DARK_FIRING_RATE_HZ,
    min_light_firing_rate_hz: float = DEFAULT_MIN_LIGHT_FIRING_RATE_HZ,
    source_tuning_curve_ids_by_role_trajectory: Mapping[
        str, Mapping[str, Any]
    ]
    | None = None,
    source_tuning_parameters_sha256_by_role_trajectory: Mapping[
        str, Mapping[str, str]
    ]
    | None = None,
    movement_firing_rate_ids_by_role: Mapping[str, Any] | None = None,
    movement_firing_rate_table_sha256_by_role: Mapping[str, str] | None = None,
    movement_intervals_sha256_by_role: Mapping[str, str],
    sources: Mapping[str, Any] | None = None,
    source_v1ca1_git_commit: str | None = None,
    source_spyglass_git_commit: str | None = None,
    tuning_curves_by_role_trajectory: Mapping[str, Mapping[str, Any]]
    | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Re-score exact NWB inputs and register only matching legacy outputs."""
    if str(source_sorting_type) != "ImportedSpikeSorting":
        raise ValueError(
            "Legacy swap-tuning registration requires ImportedSpikeSorting."
        )
    if _validate_position_offset(position_offset_samples) != LEGACY_POSITION_OFFSET_SAMPLES:
        raise ValueError("Legacy swap-tuning registration requires position offset 10.")
    source_result_path = Path(source_result_path)
    source_summary_path = Path(source_summary_path)
    for path in (source_result_path, source_summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Legacy swap-tuning artifact not found: {path}")
    recomputed = compute_swap_tuning_curve_comparison(
        swap_tuning_curve_comparison_id=swap_tuning_curve_comparison_id,
        animal_name=animal_name,
        date=date,
        region=region,
        dark_epoch=dark_epoch,
        light_train_epoch=light_train_epoch,
        light_test_epoch=light_test_epoch,
        tuning_curve_artifact_paths=tuning_curve_artifact_paths,
        tuning_curves_by_role_trajectory=tuning_curves_by_role_trajectory,
        movement_firing_rate_tables_by_role=movement_firing_rate_tables_by_role,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        position=position,
        position_offset_samples=position_offset_samples,
        movement_interval=movement_interval,
        movement_analysis_status=movement_analysis_status,
        trajectory_intervals=trajectory_intervals,
        graph_inputs_by_trajectory=graph_inputs_by_trajectory,
        parameter_name=parameter_name,
        parameter_sha256=parameter_sha256,
        output_rule_sha256=output_rule_sha256,
        evaluation_bin_size_s=evaluation_bin_size_s,
        gaussian_smoothing_sigma_bins=gaussian_smoothing_sigma_bins,
        min_dark_firing_rate_hz=min_dark_firing_rate_hz,
        min_light_firing_rate_hz=min_light_firing_rate_hz,
        source_tuning_curve_ids_by_role_trajectory=(
            source_tuning_curve_ids_by_role_trajectory
        ),
        source_tuning_parameters_sha256_by_role_trajectory=(
            source_tuning_parameters_sha256_by_role_trajectory
        ),
        movement_firing_rate_ids_by_role=movement_firing_rate_ids_by_role,
        movement_firing_rate_table_sha256_by_role=(
            movement_firing_rate_table_sha256_by_role
        ),
        movement_intervals_sha256_by_role=movement_intervals_sha256_by_role,
        sources=sources,
    )
    if recomputed["analysis_status"] not in {"valid", "partial_valid", "no_valid_units"}:
        raise ValueError("A valid legacy result cannot register to a terminal re-score.")
    speed_threshold = float(
        recomputed["upstream_provenance"]["speed_threshold_cm_s"]
    )
    if not np.isclose(
        speed_threshold,
        LEGACY_SPEED_THRESHOLD_CM_S,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Legacy swap-tuning registration requires speed threshold 4.")
    source_dataset = _load_dataset(source_result_path)
    _validate_legacy_dataset_contract(
        source_dataset,
        metadata=recomputed["metadata"],
        parameters=recomputed["parameters"],
    )
    aligned_source, identity_mapping = _legacy_identity_alignment(
        source_dataset,
        unit_identity_resolver=unit_identity_resolver,
        expected_selected_units=recomputed["selected_units"],
    )
    _compare_scientific_datasets(aligned_source, recomputed["dataset"])
    source_summary = pd.read_parquet(source_summary_path)
    _compare_legacy_summary(
        source_summary,
        recomputed=recomputed["summary"],
        legacy_to_stable_id=identity_mapping,
    )
    provenance = {
        "source_result_path": str(source_result_path.resolve()),
        "source_result_sha256": _file_sha256(source_result_path),
        "source_summary_path": str(source_summary_path.resolve()),
        "source_summary_sha256": _file_sha256(source_summary_path),
        "source_v1ca1_git_commit": (
            "unknown" if source_v1ca1_git_commit is None else str(source_v1ca1_git_commit)
        ),
        "source_spyglass_git_commit": (
            "unknown"
            if source_spyglass_git_commit is None
            else str(source_spyglass_git_commit)
        ),
        "source_sorting_type": "ImportedSpikeSorting",
        "source_schema_version": LEGACY_RESULT_SCHEMA_VERSION,
        "unit_identity_mapping_sha256": _provenance_sha256(identity_mapping),
        "position_offset_samples": LEGACY_POSITION_OFFSET_SAMPLES,
        "speed_threshold_cm_s": LEGACY_SPEED_THRESHOLD_CM_S,
        "comparison_policy": OUTPUT_RULE["legacy_comparison_policy"],
        "registration_policy": OUTPUT_RULE["legacy_registration_policy"],
    }
    registered = {
        **recomputed,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
    }
    registered["dataset"] = recomputed["dataset"].copy()
    registered["dataset"].attrs.update(
        {
            "artifact_origin": "registered_existing",
            "legacy_artifact_provenance_json": json.dumps(
                provenance,
                sort_keys=True,
            ),
        }
    )
    registered = validate_swap_tuning_curve_comparison_result(registered)
    if destination_path is None:
        return registered
    paths = write_swap_tuning_curve_comparison_artifact(
        registered,
        destination_path,
        overwrite=overwrite,
    )
    registered.update(paths)
    registered["_created_artifact_paths"] = [str(paths["artifact_dir"])]
    return registered
