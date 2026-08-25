"""Database-free held-out swapped-light GLM comparison artifacts."""

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

from v1ca1.helper.session import DEFAULT_PLACE_BIN_SIZE_CM, TRAJECTORY_TYPES


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "swap_glm"
MANIFEST_FILENAME = "manifest.parquet"
SELECTED_UNITS_FILENAME = "selected_units.parquet"
RESULT_FILENAME = "swap_glm.nc"

VISUAL_ADDITIVE_MODEL_NAME = "visual_additive_delta"
MODEL_NAMES = (
    "visual",
    VISUAL_ADDITIVE_MODEL_NAME,
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
    "dark",
)
SOURCE_MODEL_NAMES = (
    "visual",
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
)
DERIVED_MODEL_SOURCES = MappingProxyType(
    {
        VISUAL_ADDITIVE_MODEL_NAME: "visual",
        "dark": "task_segment_bump",
    }
)
LEGACY_SCHEMA4_MODEL_NAMES = (
    "visual",
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
    "dark",
)
LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES = MappingProxyType(
    {"dark": "task_segment_bump"}
)
LEGACY_SCHEMA6_MODEL_NAMES = (
    "visual",
    "visual_additive_delta",
    "visual_segment_additive_delta",
    "visual_multiplicative_ratio",
    "visual_segment_multiplicative_ratio",
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
    "dark",
)
LEGACY_SCHEMA6_DERIVED_MODEL_SOURCES = MappingProxyType(
    {
        "dark": "task_segment_bump",
        "visual_additive_delta": "visual",
        "visual_segment_additive_delta": "visual",
        "visual_multiplicative_ratio": "visual",
        "visual_segment_multiplicative_ratio": "visual",
    }
)
LEGACY_SCHEMA6_DIAGNOSTIC_VARIABLES = (
    "test_light_swapped_segment_clipped_fraction",
    "test_light_swapped_segment_source_denominator_clipped_fraction",
    "test_light_swapped_segment_source_segment_denominator_clipped",
)
LEGACY_SCHEMA6_VISUAL_EMPIRICAL_MODEL_DEFINITIONS = MappingProxyType(
    {
        "outside_swapped_segment": (
            "All visual empirical models use the target selected visual light "
            "count outside the swapped segment."
        ),
        "visual_additive_delta": (
            "On the swapped segment, target visual dark count at held-out speed "
            "plus source visual light-minus-dark count delta at reference speed."
        ),
        "visual_multiplicative_ratio": (
            "On the swapped segment, target visual dark count at held-out speed "
            "times source visual light/dark count ratio at reference speed."
        ),
        "visual_segment_additive_delta": (
            "On the swapped segment, target visual dark count at held-out speed "
            "plus the source segment mean visual light-minus-dark count delta, "
            "computed on the saved TP grid at reference speed."
        ),
        "visual_segment_multiplicative_ratio": (
            "On the swapped segment, target visual dark count at held-out speed "
            "times the source segment summed visual light/dark count ratio, "
            "computed on the saved TP grid at reference speed."
        ),
    }
)
VISUAL_EMPIRICAL_MODEL_DEFINITIONS = MappingProxyType(
    {
        "outside_swapped_segment": (
            "The target selected visual light count at held-out speed."
        ),
        VISUAL_ADDITIVE_MODEL_NAME: (
            "On the swapped segment, target visual dark count at held-out speed "
            "plus source visual light-minus-dark count delta at reference speed."
        ),
    }
)
PREDICTION_COUNT_CLIP_EPS = 1e-12
DEFAULT_SWAP_LIGHT_OFFSET = False
DEFAULT_OBSERVED_SPATIAL_BIN_SIZE_CM = DEFAULT_PLACE_BIN_SIZE_CM
RESULT_SCHEMA_VERSION = "5"
BUNDLE_SCHEMA_VERSION = "1"
NWB_ARTIFACT_SCHEMA_VERSION = "1"

NWB_SELECTED_UNITS_TABLE_NAME = "swap_glm_selected_units"
NWB_MODEL_METADATA_TABLE_NAME = "swap_glm_model_metadata"
NWB_AXES_TABLE_NAME = "swap_glm_axes"
NWB_TRAJECTORY_METADATA_TABLE_NAME = "swap_glm_trajectory_metadata"
NWB_MODEL_RESULTS_TABLE_NAME = "swap_glm_model_results"
NWB_OBSERVED_RESPONSE_TABLE_NAME = "swap_glm_observed_response"
NWB_PROVENANCE_TABLE_NAME = "swap_glm_provenance"

METRIC_PREFIXES = (
    "test_light_swapped_segment_unswapped",
    "test_light_swapped_segment_swapped",
    "test_light_full_unswapped",
    "test_light_full_swapped",
)
PRIMARY_METRIC = (
    "test_light_swapped_segment_swapped_delta_model_minus_visual_"
    "raw_ll_bits_per_spike"
)
CANONICAL_COORDINATE_DIMS = MappingProxyType(
    {
        "model": ("model",),
        "trajectory": ("trajectory",),
        "unit": ("unit",),
        "tp_grid": ("tp_grid",),
        "tp_observed_bin": ("tp_observed_bin",),
        "tp_observed_edge": ("tp_observed_edge",),
        "segment_edge": ("segment_edge",),
    }
)
CANONICAL_DATA_VARIABLE_DIMS = MappingProxyType(
    {
        "selected_model_path": ("model",),
        "selected_source_model": ("model",),
        "selected_ridge": ("model",),
        "selected_score": ("model",),
        "swap_source_trajectory": ("trajectory",),
        "swap_segment_index_1based": ("trajectory",),
        "swap_segment_start": ("trajectory",),
        "swap_segment_end": ("trajectory",),
        "dark_hz_grid": ("model", "trajectory", "tp_grid", "unit"),
        "train_light_hz_grid": ("model", "trajectory", "tp_grid", "unit"),
        "test_light_unswapped_hz_grid": (
            "model",
            "trajectory",
            "tp_grid",
            "unit",
        ),
        "test_light_swapped_hz_grid": (
            "model",
            "trajectory",
            "tp_grid",
            "unit",
        ),
        "test_light_swapped_segment_n_bins": ("trajectory",),
        "test_light_full_n_bins": ("trajectory",),
        "test_light_occupancy_s": ("trajectory", "tp_observed_bin"),
        "test_light_spike_count": (
            "trajectory",
            "tp_observed_bin",
            "unit",
        ),
        "test_light_observed_rate_hz": (
            "trajectory",
            "tp_observed_bin",
            "unit",
        ),
        **{
            f"{metric_prefix}_{suffix}": ("model", "trajectory", "unit")
            for metric_prefix in METRIC_PREFIXES
            for suffix in (
                "raw_ll_sum",
                "spike_sum",
                "raw_ll_bits_per_spike",
            )
        },
        PRIMARY_METRIC: ("model", "trajectory", "unit"),
    }
)
FIXED_SCIENTIFIC_ATTRS = MappingProxyType(
    {
        "fit_source": "dark_light_glm_selected",
        "test_scoring_scope": (
            "swapped_segment_primary_and_full_diagnostic"
        ),
        "primary_metric": PRIMARY_METRIC,
        "heldout_epoch_scope": "all_movement_laps",
        "raw_ll_bits_per_spike_definition": (
            "raw_poisson_ll_sum / spike_sum / log(2)"
        ),
    }
)

OUTPUT_RULE = MappingProxyType(
    {
        "version": 3,
        "models": MODEL_NAMES,
        "derived_model_sources": dict(DERIVED_MODEL_SOURCES),
        "swap_configuration": {
            "center_to_left": {
                "source_trajectory": "center_to_right",
                "segment_index": 2,
            },
            "center_to_right": {
                "source_trajectory": "center_to_left",
                "segment_index": 2,
            },
            "left_to_center": {
                "source_trajectory": "right_to_center",
                "segment_index": 0,
            },
            "right_to_center": {
                "source_trajectory": "left_to_center",
                "segment_index": 0,
            },
        },
        "fit_source": "exact_selected_dark_light_glm_artifact",
        "visual_additive_delta": {
            "outside_swapped_segment": "target_visual_light_at_heldout_speed",
            "inside_swapped_segment": (
                "target_visual_dark_at_heldout_speed_plus_"
                "source_visual_light_minus_dark_at_reference_speed"
            ),
            "prediction_count_clip_eps": PREDICTION_COUNT_CLIP_EPS,
        },
        "evaluation_epoch": "held_out_light_movement_laps",
        "primary_metric": PRIMARY_METRIC,
        "unit_policy": "all_upstream_dark_light_selected_units_in_saved_order",
        "unit_validity_policy": (
            "upstream_valid_glm_fit_and_all_expected_primary_scores_finite"
        ),
        "unit_failure_policy": (
            "retain_all_units_and_isolate_nonfinite_scores_by_unit_model_trajectory"
        ),
        "runtime_unit_key_policy": (
            "persistent_identity_aligned_native_tsgroup_keys_with_canonical_output_ids"
        ),
        "trajectory_support_policy": (
            "all_or_none_terminal_if_any_path_has_no_movement_bins"
        ),
        "movement_terminal_status_policy": (
            "selected_movement_firing_rate_status_precedes_interval_fallback"
        ),
        "legacy_registration_policy": (
            "normalize_verified_schema4_or_schema6_then_exact_nwb_rescore_without_refit"
        ),
        "legacy_comparison_policy": (
            "all_scientific_coordinates_and_variables_tight_equal"
        ),
        "legacy_missing_derived_model_policy": (
            "schema4_four_source_models_compare_all_available_then_rescore_dark_from_verified_task_segment_bump"
        ),
        "legacy_preprocessing_provenance_policy": (
            "historical_schema4_or_schema6_position_offset_and_speed_threshold_must_match_selected_nwb_inputs"
        ),
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "position_unit": "cm",
    }
)
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "upstream_terminal",
    "no_units",
    "no_valid_position",
    "no_movement",
    "no_trajectory_samples",
    "no_valid_units",
)
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SELECTED_UNIT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "selection_index",
    "dark_movement_firing_rate_hz",
    "light_movement_firing_rate_hz",
    "upstream_valid_glm_fit",
    "test_light_spike_count",
    "n_finite_primary_scores",
    "n_expected_primary_scores",
    "valid_swap_score",
)
MODEL_METADATA_COLUMNS = (
    "model",
    "selected_model_path",
    "selected_source_model",
    "selected_ridge",
    "selected_score",
)
AXES_COLUMNS = ("axis_name", "values")
TRAJECTORY_METADATA_COLUMNS = (
    "trajectory",
    "swap_source_trajectory",
    "swap_segment_index_1based",
    "swap_segment_start",
    "swap_segment_end",
    "test_light_swapped_segment_n_bins",
    "test_light_full_n_bins",
    "test_light_occupancy_s",
)
MODEL_RESULT_PROFILE_COLUMNS = (
    "dark_hz_grid",
    "train_light_hz_grid",
    "test_light_unswapped_hz_grid",
    "test_light_swapped_hz_grid",
)
MODEL_RESULT_SCORE_COLUMNS = (
    *(
        f"{metric_prefix}_{suffix}"
        for metric_prefix in METRIC_PREFIXES
        for suffix in (
            "raw_ll_sum",
            "spike_sum",
            "raw_ll_bits_per_spike",
        )
    ),
    PRIMARY_METRIC,
)
MODEL_RESULT_COLUMNS = (
    "model",
    "trajectory",
    *IDENTITY_COLUMNS,
    "selection_index",
    *MODEL_RESULT_PROFILE_COLUMNS,
    *MODEL_RESULT_SCORE_COLUMNS,
)
OBSERVED_RESPONSE_COLUMNS = (
    "trajectory",
    *IDENTITY_COLUMNS,
    "selection_index",
    "test_light_spike_count",
    "test_light_observed_rate_hz",
)
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
    "swap_glm_id",
    "animal_name",
    "date",
    "region",
    "dark_epoch",
    "light_train_epoch",
    "light_test_epoch",
    "parameter_name",
    "parameter_sha256",
    "output_rule_sha256",
    "swap_light_offset",
    "observed_spatial_bin_size_cm",
    "dark_light_glm_id",
    "dark_light_manifest_sha256",
    "dark_light_selected_sha256_json",
    "dark_light_parameter_sha256",
    "dark_light_output_rule_sha256",
    "upstream_analysis_status",
    "n_units",
    "n_valid_units",
    "selected_units_sha256",
    "analysis_status",
    "artifact_origin",
    "legacy_artifact_provenance_json",
    "bundle_schema_version",
)


def _analysis_module() -> Any:
    """Import the existing evaluator and initialize its spline dependency."""
    from v1ca1.task_progression import swap_glm_comparison

    swap_glm_comparison._require_spline_basis()
    return swap_glm_comparison


def _validate_reused_analysis_contract(analysis: Any) -> None:
    """Fail if the fixed Spyglass contract drifts from the reused evaluator."""
    expected_models = tuple(getattr(analysis, "DEFAULT_MODEL_NAMES", ()))
    if expected_models != LEGACY_SCHEMA4_MODEL_NAMES:
        raise ValueError(
            "SwapGLM model order differs from swap_glm_comparison."
        )
    expected_sources = dict(
        getattr(analysis, "DERIVED_SELECTED_MODEL_SOURCES", {})
    )
    if expected_sources != dict(LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES):
        raise ValueError(
            "SwapGLM derived-model sources differ from "
            "swap_glm_comparison."
        )
    expected_swap = dict(getattr(analysis, "SWAP_CONFIG", {}))
    if expected_swap != dict(OUTPUT_RULE["swap_configuration"]):
        raise ValueError(
            "SwapGLM trajectory swap configuration differs from "
            "swap_glm_comparison."
        )


def _provenance_sha256(value: Any) -> str:
    """Return the shared deterministic provenance digest."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(value)


OUTPUT_RULE_SHA256 = _provenance_sha256(dict(OUTPUT_RULE))
LEGACY_SCHEMA4_OUTPUT_RULE_SHA256 = (
    "e730dcdee948232670ece391af58a8053417a172e1229ddd38e5d269bd24f408"
)


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


def get_swap_glm_artifact_paths(
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    swap_glm_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return one UUID-keyed, session-first held-out swap bundle."""
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
    result_id = _uuid_string(swap_glm_id, name="swap_glm_id")
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
        "result_path": artifact_dir / RESULT_FILENAME,
    }


def validate_swap_glm_parameters(
    *,
    swap_light_offset: bool = DEFAULT_SWAP_LIGHT_OFFSET,
    observed_spatial_bin_size_cm: float = DEFAULT_OBSERVED_SPATIAL_BIN_SIZE_CM,
) -> dict[str, Any]:
    """Return validated held-out scoring parameters."""
    if not isinstance(swap_light_offset, (bool, np.bool_)):
        raise TypeError("swap_light_offset must be boolean.")
    observed_spatial_bin_size_cm = float(observed_spatial_bin_size_cm)
    if (
        not np.isfinite(observed_spatial_bin_size_cm)
        or observed_spatial_bin_size_cm <= 0.0
    ):
        raise ValueError(
            "observed_spatial_bin_size_cm must be positive and finite."
        )
    return {
        "swap_light_offset": bool(swap_light_offset),
        "observed_spatial_bin_size_cm": observed_spatial_bin_size_cm,
    }


def _metadata(
    *,
    swap_glm_id: Any,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
) -> dict[str, str]:
    """Return validated metadata for one held-out comparison."""
    metadata = {
        "swap_glm_id": _uuid_string(swap_glm_id, name="swap_glm_id"),
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


def _load_dataset(path: Path) -> Any:
    """Load one NetCDF dataset eagerly and close its backing file."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return dataset.load()


def _load_dark_light_input(path: Path) -> dict[str, Any]:
    """Load one exact upstream DarkLight artifact and its selected-file hashes."""
    from v1ca1.spyglass.dark_light_glm import (
        MANIFEST_FILENAME as DARK_LIGHT_MANIFEST_FILENAME,
        load_dark_light_glm_artifact,
    )

    directory = Path(path)
    result = load_dark_light_glm_artifact(directory)
    manifest_path = directory / DARK_LIGHT_MANIFEST_FILENAME
    manifest = result.get("manifest")
    if manifest is None:
        manifest = pd.read_parquet(manifest_path)
    selected_hashes = {}
    for model_name in SOURCE_MODEL_NAMES:
        rows = manifest.loc[
            manifest["artifact_key"].astype(str) == f"selected:{model_name}"
        ]
        if len(rows) != 1:
            raise ValueError(
                f"DarkLight manifest must contain selected:{model_name} exactly once."
            )
        selected_hashes[model_name] = str(rows.iloc[0]["sha256"])
    legacy_provenance = result.get("legacy_artifact_provenance")
    legacy_selected_hashes = (
        {}
        if legacy_provenance is None
        else dict(legacy_provenance.get("source_selected_sha256", {}))
    )
    result["upstream_provenance"] = {
        "dark_light_glm_id": str(result["metadata"]["dark_light_glm_id"]),
        "dark_light_manifest_sha256": _file_sha256(manifest_path),
        "dark_light_selected_sha256_by_model": selected_hashes,
        "dark_light_parameter_sha256": str(
            result["parameters"].get("parameter_sha256", "")
        ),
        "dark_light_output_rule_sha256": str(
            result["parameters"].get("output_rule_sha256", "")
        ),
        "upstream_analysis_status": str(result["analysis_status"]),
        "dark_light_artifact_path": str(directory.resolve()),
        "dark_light_legacy_selected_sha256_by_model": legacy_selected_hashes,
    }
    return result


def _load_dark_light_nwb_input(
    result: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an in-memory DarkLightGLM result and attach NWB provenance."""
    from v1ca1.spyglass.dark_light_glm import validate_dark_light_glm_result

    canonical = validate_dark_light_glm_result(result)
    upstream = dict(upstream_provenance)
    required = {
        "dark_light_glm_id",
        "dark_light_glm_sha256",
        "dark_light_selected_model_sha256_by_model",
        "dark_light_parameter_sha256",
        "dark_light_output_rule_sha256",
        "upstream_analysis_status",
    }
    missing = sorted(required.difference(upstream))
    if missing:
        raise ValueError(
            f"DarkLightGLM NWB provenance is missing fields {missing!r}."
        )
    canonical["upstream_provenance"] = upstream
    return canonical


def _dark_light_provenance_style(
    upstream: Mapping[str, Any],
) -> str:
    """Return the mutually exclusive file-bundle or NWB provenance style."""
    legacy = {
        "dark_light_manifest_sha256",
        "dark_light_selected_sha256_by_model",
    }.issubset(upstream)
    nwb = {
        "dark_light_glm_sha256",
        "dark_light_selected_model_sha256_by_model",
    }.issubset(upstream)
    if legacy == nwb:
        raise ValueError(
            "DarkLight upstream must use exactly one file-bundle or NWB "
            "provenance contract."
        )
    return "legacy" if legacy else "nwb"


def _dark_light_selected_hashes(
    upstream: Mapping[str, Any],
) -> dict[str, str]:
    """Return selected-model hashes from either supported provenance style."""
    legacy_field = "dark_light_selected_sha256_by_model"
    nwb_field = "dark_light_selected_model_sha256_by_model"
    if legacy_field in upstream and nwb_field not in upstream:
        field = legacy_field
    elif nwb_field in upstream and legacy_field not in upstream:
        field = nwb_field
    else:
        raise ValueError(
            "DarkLight selected-model provenance must use exactly one "
            "file-bundle or NWB hash field."
        )
    return {str(key): str(value) for key, value in dict(upstream[field]).items()}


def _dark_light_dataset_provenance_attrs(
    upstream: Mapping[str, Any],
) -> dict[str, Any]:
    """Return style-specific DarkLight provenance attrs for a swap dataset."""
    style = _dark_light_provenance_style(upstream)
    if style == "legacy":
        return {
            "dark_light_manifest_sha256": str(
                upstream["dark_light_manifest_sha256"]
            ),
            "dark_light_selected_sha256_json": json.dumps(
                upstream["dark_light_selected_sha256_by_model"],
                sort_keys=True,
            ),
        }
    return {
        "dark_light_glm_sha256": str(upstream["dark_light_glm_sha256"]),
        "dark_light_selected_model_sha256_json": json.dumps(
            upstream["dark_light_selected_model_sha256_by_model"],
            sort_keys=True,
        ),
    }


def _validate_upstream_context(
    upstream: Mapping[str, Any],
    metadata: Mapping[str, str],
) -> None:
    """Require the selected DarkLight row to match this session and train pair."""
    expected = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_epoch": metadata["dark_epoch"],
        "light_epoch": metadata["light_train_epoch"],
    }
    observed = dict(upstream["metadata"])
    for name, value in expected.items():
        if str(observed.get(name, "")) != str(value):
            raise ValueError(f"DarkLight upstream has mismatched {name!r}.")


def _selected_unit_ids(upstream: Mapping[str, Any]) -> np.ndarray:
    """Return selected dataset unit IDs after checking every upstream model."""
    selected = dict(upstream["selected_datasets"])
    if set(selected) != set(SOURCE_MODEL_NAMES):
        raise ValueError("DarkLight upstream must contain all four selected models.")
    reference = np.asarray(selected["visual"].coords["unit"].values)
    for model_name in SOURCE_MODEL_NAMES[1:]:
        candidate = np.asarray(selected[model_name].coords["unit"].values)
        if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
            raise ValueError("DarkLight selected models disagree on unit order.")
    return reference


def _heldout_identity_alignment(
    *,
    upstream: Mapping[str, Any],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Match canonical selected units to held-out native TsGroup keys."""
    from v1ca1.spyglass.path_specific_place import _identity_rows

    upstream_units = upstream["selected_units"].copy()
    required = {
        *IDENTITY_COLUMNS,
        "dark_movement_firing_rate_hz",
        "light_movement_firing_rate_hz",
        "valid_glm_fit",
    }
    missing = sorted(required.difference(upstream_units.columns))
    if missing:
        raise ValueError(f"DarkLight selected-unit audit is missing {missing!r}.")
    identities = pd.DataFrame.from_records(
        _identity_rows(spikes, stable_unit_ids),
        columns=(*IDENTITY_COLUMNS, "_group_key"),
    )
    heldout_lookup = {
        (str(row["spikesorting_merge_id"]), str(row["unit_id"])): row
        for _, row in identities.iterrows()
    }
    selected_ids = _selected_unit_ids(upstream)
    if len(upstream_units) != len(selected_ids) or not np.array_equal(
        upstream_units["group_unit_id"].astype(str).to_numpy(),
        np.asarray([str(value) for value in selected_ids]),
    ):
        raise ValueError(
            "DarkLight selected-unit group keys do not match selected NetCDF unit order."
        )
    records = []
    native_group_keys = []
    for selection_index, (_, source_row) in enumerate(upstream_units.iterrows()):
        key = (
            str(source_row["spikesorting_merge_id"]),
            str(source_row["unit_id"]),
        )
        heldout_row = heldout_lookup.get(key)
        if heldout_row is None:
            raise ValueError(
                "Held-out RegionSortedSpikesGroup is missing DarkLight selected "
                f"unit {key!r}."
            )
        for name in ("stable_unit_id", "group_unit_id"):
            if str(heldout_row[name]) != str(source_row[name]):
                raise ValueError(
                    f"Held-out sorting has conflicting {name} for unit {key!r}."
                )
        native_group_keys.append(heldout_row["_group_key"])
        records.append(
            {
                **{name: str(source_row[name]) for name in IDENTITY_COLUMNS},
                "selection_index": selection_index,
                "dark_movement_firing_rate_hz": float(
                    source_row["dark_movement_firing_rate_hz"]
                ),
                "light_movement_firing_rate_hz": float(
                    source_row["light_movement_firing_rate_hz"]
                ),
                "upstream_valid_glm_fit": bool(source_row["valid_glm_fit"]),
                "test_light_spike_count": 0.0,
                "n_finite_primary_scores": 0,
                "n_expected_primary_scores": (
                    (len(MODEL_NAMES) - 1) * len(TRAJECTORY_TYPES)
                ),
                "valid_swap_score": False,
            }
        )
    if len({repr(key) for key in native_group_keys}) != len(native_group_keys):
        raise ValueError("Held-out native TsGroup keys must be unique.")
    return (
        pd.DataFrame.from_records(records, columns=SELECTED_UNIT_COLUMNS),
        np.asarray(native_group_keys, dtype=object),
    )


def _heldout_identity_audit(
    *,
    upstream: Mapping[str, Any],
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return the selected-unit audit after persistent identity alignment."""
    return _heldout_identity_alignment(
        upstream=upstream,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )[0]


def _derive_task_progression(
    *,
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Linearize one held-out epoch from selected NWB graph inputs."""
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    return {
        trajectory_type: build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals[trajectory_type],
            graph_inputs=graph_inputs_by_trajectory[trajectory_type],
            trajectory_type=trajectory_type,
        )[0]
        for trajectory_type in TRAJECTORY_TYPES
    }


def _derive_speed(position: Any, *, smoothing_sigma_s: float) -> Any:
    """Compute speed from an already-offset selected NWB Position row."""
    from v1ca1.helper.session import build_speed_tsd

    return build_speed_tsd(
        np.asarray(position.d, dtype=float),
        np.asarray(position.t, dtype=float),
        position_offset=0,
        speed_smoothing_sigma_s=float(smoothing_sigma_s),
    )


def _has_valid_position_samples(position: Any) -> bool:
    """Return whether Position has at least two finite, timed samples."""
    try:
        times = np.asarray(position.t, dtype=float).reshape(-1)
        values = np.asarray(position.d, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return False
    if not times.size or values.shape[0] != times.size:
        return False
    finite_values = (
        np.isfinite(values)
        if values.ndim == 1
        else np.all(np.isfinite(values.reshape(times.size, -1)), axis=1)
    )
    return bool(np.sum(np.isfinite(times) & finite_values) >= 2)


def _interval_duration(intervals: Any) -> float:
    """Return total interval duration in seconds."""
    starts = np.asarray(intervals.start, dtype=float).reshape(-1)
    ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    if starts.shape != ends.shape or np.any(ends < starts):
        raise ValueError("Interval starts and ends must be aligned and ordered.")
    return float(np.sum(ends - starts))


def _observed_bin_edges(path_length_cm: float, bin_size_cm: float) -> np.ndarray:
    """Return legacy-compatible normalized observed task-progression bins."""
    path_length_cm = float(path_length_cm)
    if not np.isfinite(path_length_cm) or path_length_cm <= 0.0:
        raise ValueError("path_length_cm must be positive and finite.")
    return np.arange(0.0, 1.0 + bin_size_cm / path_length_cm, bin_size_cm / path_length_cm)


def _terminal_dataset(
    *,
    metadata: Mapping[str, str],
    unit_ids: np.ndarray,
    segment_edges: np.ndarray,
    parameters: Mapping[str, Any],
    upstream_provenance: Mapping[str, Any],
    analysis_status: str,
    terminal_detail: Mapping[str, Any] | None = None,
) -> Any:
    """Return one persistable NetCDF terminal marker."""
    import xarray as xr

    dataset = xr.Dataset(
        coords={
            "model": np.asarray(MODEL_NAMES, dtype=str),
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "unit": np.asarray(unit_ids),
            "segment_edge": np.asarray(segment_edges, dtype=float),
        },
        attrs={
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
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
            "fit_stage": "terminal",
            "analysis_status": analysis_status,
            "parameter_name": str(parameters["parameter_name"]),
            "parameter_sha256": str(parameters["parameter_sha256"]),
            "swap_light_offset": bool(parameters["swap_light_offset"]),
            "observed_spatial_bin_size_cm": float(
                parameters["observed_spatial_bin_size_cm"]
            ),
            "output_rule_sha256": str(parameters["output_rule_sha256"]),
            "dark_light_glm_id": str(upstream_provenance["dark_light_glm_id"]),
            **_dark_light_dataset_provenance_attrs(upstream_provenance),
            "dark_light_parameter_sha256": str(
                upstream_provenance["dark_light_parameter_sha256"]
            ),
            "dark_light_output_rule_sha256": str(
                upstream_provenance["dark_light_output_rule_sha256"]
            ),
            "upstream_analysis_status": str(
                upstream_provenance["upstream_analysis_status"]
            ),
        },
    )
    dataset.attrs.update(dict(terminal_detail or {}))
    return dataset


def _audit_scores(
    selected_units: pd.DataFrame,
    dataset: Any,
    *,
    model_names: Sequence[str] = MODEL_NAMES,
) -> tuple[pd.DataFrame, str]:
    """Attach per-unit score QC without allowing one unit to affect another."""
    output = selected_units.copy()
    if output.empty:
        return output.loc[:, list(SELECTED_UNIT_COLUMNS)], "no_units"
    primary_name = (
        "test_light_swapped_segment_swapped_delta_model_minus_visual_"
        "raw_ll_bits_per_spike"
    )
    raw_name = "test_light_full_swapped_spike_sum"
    if primary_name not in dataset or raw_name not in dataset:
        raise ValueError("Swap result is missing primary score or spike-count variables.")
    expected_dimensions = ("model", "trajectory", "unit")
    if tuple(dataset[primary_name].dims) != expected_dimensions:
        raise ValueError("Primary swap-score dimensions are noncanonical.")
    if tuple(dataset[raw_name].dims) != expected_dimensions:
        raise ValueError("Swap spike-count dimensions are noncanonical.")
    for coordinate, expected in (
        ("model", model_names),
        ("trajectory", TRAJECTORY_TYPES),
    ):
        if coordinate not in dataset.coords or not np.array_equal(
            np.asarray(dataset.coords[coordinate].values, dtype=str),
            np.asarray(expected, dtype=str),
        ):
            raise ValueError(
                f"Swap score audit has a noncanonical {coordinate!r} coordinate."
            )
    if "unit" not in dataset.coords or not np.array_equal(
        np.asarray(dataset.coords["unit"].values, dtype=str),
        output["group_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Swap score audit unit order differs from selected_units.")
    primary = np.asarray(dataset[primary_name].values, dtype=float)
    expected_shape = (
        len(model_names),
        len(TRAJECTORY_TYPES),
        len(output),
    )
    if primary.shape != expected_shape:
        raise ValueError("Primary swap-score array has an unexpected shape.")
    if not np.all(np.isnan(primary[0])):
        raise ValueError("Visual-reference primary swap scores must be NaN.")
    delta = primary[1:]
    if delta.shape != (
        len(model_names) - 1,
        len(TRAJECTORY_TYPES),
        len(output),
    ):
        raise ValueError("Primary swap-score array has an unexpected shape.")
    raw_spike_sum = np.asarray(dataset[raw_name].values, dtype=float)
    if raw_spike_sum.shape != expected_shape:
        raise ValueError("Swap spike-count array has an unexpected shape.")
    spike_sum = raw_spike_sum[0]
    if spike_sum.shape != (len(TRAJECTORY_TYPES), len(output)):
        raise ValueError("Swap spike-count array has an unexpected shape.")
    finite_count = np.sum(np.isfinite(delta), axis=(0, 1)).astype(int)
    expected_count = (len(model_names) - 1) * len(TRAJECTORY_TYPES)
    output["test_light_spike_count"] = np.nansum(spike_sum, axis=0)
    output["n_finite_primary_scores"] = finite_count
    output["n_expected_primary_scores"] = expected_count
    output["valid_swap_score"] = (
        output["upstream_valid_glm_fit"].astype(bool)
        & (finite_count == expected_count)
    )
    if np.all(output["valid_swap_score"]):
        status = "valid"
    elif np.any(output["valid_swap_score"]):
        status = "partial_valid"
    else:
        status = "no_valid_units"
    return output.loc[:, list(SELECTED_UNIT_COLUMNS)], status


def _effective_parameters(
    *,
    parameter_name: str,
    parameter_sha256: str | None,
    output_rule_sha256: str | None,
    swap_light_offset: bool,
    observed_spatial_bin_size_cm: float,
) -> dict[str, Any]:
    """Validate parameters and their deterministic provenance hashes."""
    validated = validate_swap_glm_parameters(
        swap_light_offset=swap_light_offset,
        observed_spatial_bin_size_cm=observed_spatial_bin_size_cm,
    )
    parameter_name = _path_component(parameter_name, name="parameter_name")
    payload = {"swap_glm_param_name": parameter_name, **validated}
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
        **validated,
    }


def _visual_additive_delta_count(
    *,
    target_light_count: np.ndarray,
    target_dark_count: np.ndarray,
    source_light_reference_count: np.ndarray,
    source_dark_reference_count: np.ndarray,
    swap_mask: np.ndarray,
) -> np.ndarray:
    """Return the exact count-space additive visual swap prediction."""
    target_light_count = np.asarray(target_light_count, dtype=float)
    target_dark_count = np.asarray(target_dark_count, dtype=float)
    source_light_reference_count = np.asarray(
        source_light_reference_count,
        dtype=float,
    )
    source_dark_reference_count = np.asarray(
        source_dark_reference_count,
        dtype=float,
    )
    if not (
        target_dark_count.shape
        == source_light_reference_count.shape
        == source_dark_reference_count.shape
        == target_light_count.shape
    ):
        raise ValueError("Visual additive prediction counts must share one shape.")
    mask = np.asarray(swap_mask, dtype=bool).reshape(-1)
    if target_light_count.ndim != 2 or mask.shape != (
        target_light_count.shape[0],
    ):
        raise ValueError(
            "Visual additive prediction requires a bin-by-unit count matrix "
            "and one aligned swap mask."
        )
    swapped = target_light_count.copy()
    additive = (
        target_dark_count
        + source_light_reference_count
        - source_dark_reference_count
    )
    swapped[mask] = np.maximum(
        additive[mask],
        PREDICTION_COUNT_CLIP_EPS,
    )
    return swapped


def _evaluate_visual_additive_delta_on_test_epoch(
    *,
    analysis: Any,
    dataset: Any,
    test_inputs_by_traj: Mapping[str, Mapping[str, Any]],
    segment_edges: np.ndarray,
    bin_size_s: float,
    n_splines: int,
    spline_order: int,
) -> dict[str, dict[str, Any]]:
    """Score the count-space additive visual swap on held-out light data."""
    results: dict[str, dict[str, Any]] = {}
    grid = np.asarray(dataset.coords["tp_grid"].values, dtype=float)
    unit_ids = np.asarray(dataset.coords["unit"].values)
    for trajectory in TRAJECTORY_TYPES:
        swap_info = OUTPUT_RULE["swap_configuration"][trajectory]
        source_trajectory = str(swap_info["source_trajectory"])
        swap_segment_index = int(swap_info["segment_index"])
        test_inputs = test_inputs_by_traj[trajectory]
        p_test = np.asarray(test_inputs["p"], dtype=float)
        y_test = np.asarray(test_inputs["y"], dtype=float)
        swap_mask = analysis._segment_mask(
            p_test,
            segment_edges,
            swap_segment_index,
        )

        target_light = analysis.predict_selected_light_eta(
            dataset=dataset,
            model_name="visual",
            trajectory=trajectory,
            p_eval=p_test,
            speed_values=test_inputs["v"],
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
        )
        target_dark = analysis.predict_selected_dark_eta(
            dataset=dataset,
            trajectory=trajectory,
            p_eval=p_test,
            speed_values=test_inputs["v"],
            n_splines=n_splines,
            spline_order=spline_order,
        )
        source_light_reference = analysis.predict_selected_light_eta(
            dataset=dataset,
            model_name="visual",
            trajectory=source_trajectory,
            p_eval=p_test,
            speed_values=None,
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
            allow_missing_speed=True,
        )
        source_dark_reference = analysis.predict_selected_dark_eta(
            dataset=dataset,
            trajectory=source_trajectory,
            p_eval=p_test,
            speed_values=None,
            n_splines=n_splines,
            spline_order=spline_order,
            allow_missing_speed=True,
        )
        target_light_count = np.exp(target_light["light_eta"])
        swapped_count = _visual_additive_delta_count(
            target_light_count=target_light_count,
            target_dark_count=np.exp(target_dark["dark_eta"]),
            source_light_reference_count=np.exp(
                source_light_reference["light_eta"]
            ),
            source_dark_reference_count=np.exp(
                source_dark_reference["dark_eta"]
            ),
            swap_mask=swap_mask,
        )

        grid_mask = analysis._segment_mask(
            grid,
            segment_edges,
            swap_segment_index,
        )
        target_dark_grid_hz = analysis._selected_var_by_trajectory(
            dataset,
            "dark_hz_grid",
            trajectory,
        )
        target_light_grid_hz = analysis._selected_var_by_trajectory(
            dataset,
            "train_light_hz_grid",
            trajectory,
        )
        source_dark_grid_hz = analysis._selected_var_by_trajectory(
            dataset,
            "dark_hz_grid",
            source_trajectory,
        )
        source_light_grid_hz = analysis._selected_var_by_trajectory(
            dataset,
            "train_light_hz_grid",
            source_trajectory,
        )
        swapped_grid_count = _visual_additive_delta_count(
            target_light_count=target_light_grid_hz * float(bin_size_s),
            target_dark_count=target_dark_grid_hz * float(bin_size_s),
            source_light_reference_count=(
                source_light_grid_hz * float(bin_size_s)
            ),
            source_dark_reference_count=(
                source_dark_grid_hz * float(bin_size_s)
            ),
            swap_mask=grid_mask,
        )

        y_segment = y_test[swap_mask] if np.any(swap_mask) else y_test[:0]
        unswapped_segment_count = (
            target_light_count[swap_mask]
            if np.any(swap_mask)
            else target_light_count[:0]
        )
        swapped_segment_count = (
            swapped_count[swap_mask]
            if np.any(swap_mask)
            else swapped_count[:0]
        )
        results[trajectory] = {
            "unit_ids": unit_ids,
            "swap_source_trajectory": source_trajectory,
            "swap_segment_index_1based": swap_segment_index + 1,
            "swap_segment_start": float(segment_edges[swap_segment_index]),
            "swap_segment_end": float(segment_edges[swap_segment_index + 1]),
            "dark_hz_grid": target_dark_grid_hz,
            "train_light_hz_grid": target_light_grid_hz,
            "test_light_unswapped_hz_grid": target_light_grid_hz,
            "test_light_swapped_hz_grid": (
                swapped_grid_count / float(bin_size_s)
            ),
            "test_light_full_unswapped_metrics": (
                analysis.summarize_raw_poisson_metrics(
                    y_test,
                    target_light_count,
                )
            ),
            "test_light_full_swapped_metrics": (
                analysis.summarize_raw_poisson_metrics(
                    y_test,
                    swapped_count,
                )
            ),
            "test_light_swapped_segment_unswapped_metrics": (
                analysis.summarize_raw_poisson_metrics(
                    y_segment,
                    unswapped_segment_count,
                )
            ),
            "test_light_swapped_segment_swapped_metrics": (
                analysis.summarize_raw_poisson_metrics(
                    y_segment,
                    swapped_segment_count,
                )
            ),
            "test_light_full_n_bins": int(y_test.shape[0]),
            "test_light_swapped_segment_n_bins": int(y_segment.shape[0]),
        }
    return results


def compute_swap_glm(
    *,
    swap_glm_id: Any,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    dark_light_glm_artifact_path: Path | None = None,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_interval: Any,
    movement_analysis_status: str | None = None,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    position: Any | None = None,
    task_progression_by_trajectory: Mapping[str, Any] | None = None,
    speed: Any | None = None,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    swap_light_offset: bool = DEFAULT_SWAP_LIGHT_OFFSET,
    observed_spatial_bin_size_cm: float = DEFAULT_OBSERVED_SPATIAL_BIN_SIZE_CM,
    sources: Mapping[str, Any] | None = None,
    dark_light_glm_result: Mapping[str, Any] | None = None,
    dark_light_glm_upstream_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score selected DarkLight models on one held-out light epoch."""
    metadata = _metadata(
        swap_glm_id=swap_glm_id,
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
        swap_light_offset=swap_light_offset,
        observed_spatial_bin_size_cm=observed_spatial_bin_size_cm,
    )
    analysis = _analysis_module()
    _validate_reused_analysis_contract(analysis)
    if set(trajectory_intervals) != set(TRAJECTORY_TYPES):
        raise ValueError("trajectory_intervals must contain exactly four paths.")
    if set(graph_inputs_by_trajectory) != set(TRAJECTORY_TYPES):
        raise ValueError("graph_inputs_by_trajectory must contain exactly four paths.")

    if (dark_light_glm_artifact_path is None) == (dark_light_glm_result is None):
        raise ValueError(
            "Provide exactly one DarkLightGLM artifact path or in-memory result."
        )
    if dark_light_glm_result is None:
        upstream = _load_dark_light_input(Path(dark_light_glm_artifact_path))
    else:
        if dark_light_glm_upstream_provenance is None:
            raise ValueError(
                "In-memory DarkLightGLM input requires frozen NWB provenance."
            )
        upstream = _load_dark_light_nwb_input(
            dark_light_glm_result,
            dark_light_glm_upstream_provenance,
        )
    _validate_upstream_context(upstream, metadata)
    upstream_provenance = dict(upstream["upstream_provenance"])
    selected_unit_ids = _selected_unit_ids(upstream)
    selected_units, native_unit_keys = _heldout_identity_alignment(
        upstream=upstream,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )
    from v1ca1.spyglass.dark_light_glm import derive_graph_geometry

    path_length_cm, segment_edges = derive_graph_geometry(
        graph_inputs_by_trajectory
    )
    upstream_edges = np.asarray(upstream["segment_edges"], dtype=float)
    if not np.allclose(segment_edges, upstream_edges, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "Held-out WTrackGraph geometry differs from the DarkLight fit geometry."
        )
    upstream_status = str(upstream.get("analysis_status", "valid"))
    if upstream_status not in {"valid", "partial_valid"}:
        dataset = _terminal_dataset(
            metadata=metadata,
            unit_ids=selected_unit_ids,
            segment_edges=segment_edges,
            parameters=parameters,
            upstream_provenance=upstream_provenance,
            analysis_status="upstream_terminal",
        )
        result = {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream_provenance,
            "selected_units": selected_units,
            "dataset": dataset,
            "analysis_status": "upstream_terminal",
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
        return validate_swap_glm_result(result)
    if selected_units.empty:
        dataset = _terminal_dataset(
            metadata=metadata,
            unit_ids=selected_unit_ids,
            segment_edges=segment_edges,
            parameters=parameters,
            upstream_provenance=upstream_provenance,
            analysis_status="no_units",
        )
        return validate_swap_glm_result(
            {
                "metadata": metadata,
                "parameters": parameters,
                "upstream_provenance": upstream_provenance,
                "selected_units": selected_units,
                "dataset": dataset,
                "analysis_status": "no_units",
                "artifact_origin": "computed",
                "legacy_artifact_provenance": None,
            }
        )
    if movement_analysis_status is not None:
        movement_analysis_status = str(movement_analysis_status)
        if movement_analysis_status not in {
            "valid",
            "no_units",
            "no_valid_position",
            "no_movement",
        }:
            raise ValueError(
                "movement_analysis_status is not a supported MovementFiringRate "
                "terminal status."
            )
        if movement_analysis_status == "no_units":
            raise ValueError(
                "MovementFiringRate reports no_units for a non-empty SwapGLM "
                "unit selection."
            )
        if movement_analysis_status in {"no_valid_position", "no_movement"}:
            if _interval_duration(movement_interval) > 0.0:
                raise ValueError(
                    "Terminal MovementFiringRate status conflicts with a "
                    "positive-duration movement interval."
                )
            dataset = _terminal_dataset(
                metadata=metadata,
                unit_ids=selected_unit_ids,
                segment_edges=segment_edges,
                parameters=parameters,
                upstream_provenance=upstream_provenance,
                analysis_status=movement_analysis_status,
            )
            return validate_swap_glm_result(
                {
                    "metadata": metadata,
                    "parameters": parameters,
                    "upstream_provenance": upstream_provenance,
                    "selected_units": selected_units,
                    "dataset": dataset,
                    "analysis_status": movement_analysis_status,
                    "artifact_origin": "computed",
                    "legacy_artifact_provenance": None,
                }
            )
    if position is not None and not _has_valid_position_samples(position):
        if movement_analysis_status == "valid":
            raise ValueError(
                "Selected Position no longer agrees with valid MovementFiringRate."
            )
        dataset = _terminal_dataset(
            metadata=metadata,
            unit_ids=selected_unit_ids,
            segment_edges=segment_edges,
            parameters=parameters,
            upstream_provenance=upstream_provenance,
            analysis_status="no_valid_position",
        )
        return validate_swap_glm_result(
            {
                "metadata": metadata,
                "parameters": parameters,
                "upstream_provenance": upstream_provenance,
                "selected_units": selected_units,
                "dataset": dataset,
                "analysis_status": "no_valid_position",
                "artifact_origin": "computed",
                "legacy_artifact_provenance": None,
            }
        )
    if _interval_duration(movement_interval) <= 0.0:
        if movement_analysis_status == "valid":
            raise ValueError(
                "Valid MovementFiringRate must contain positive-duration movement."
            )
        dataset = _terminal_dataset(
            metadata=metadata,
            unit_ids=selected_unit_ids,
            segment_edges=segment_edges,
            parameters=parameters,
            upstream_provenance=upstream_provenance,
            analysis_status="no_movement",
        )
        return validate_swap_glm_result(
            {
                "metadata": metadata,
                "parameters": parameters,
                "upstream_provenance": upstream_provenance,
                "selected_units": selected_units,
                "dataset": dataset,
                "analysis_status": "no_movement",
                "artifact_origin": "computed",
                "legacy_artifact_provenance": None,
            }
        )
    if task_progression_by_trajectory is None:
        if position is None:
            raise ValueError(
                "position is required when task_progression_by_trajectory is absent."
            )
        task_progression_by_trajectory = _derive_task_progression(
            position=position,
            trajectory_intervals=trajectory_intervals,
            graph_inputs_by_trajectory=graph_inputs_by_trajectory,
        )
    if set(task_progression_by_trajectory) != set(TRAJECTORY_TYPES):
        raise ValueError("task_progression_by_trajectory must contain four paths.")

    source_selected_datasets = dict(upstream["selected_datasets"])
    shared_metadata = analysis.validate_selected_dark_light_glms(
        source_selected_datasets,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        dark_train_epoch=metadata["dark_epoch"],
        light_train_epoch=metadata["light_train_epoch"],
    )
    if bool(shared_metadata["has_speed"]) and speed is None:
        if position is None:
            raise ValueError("A speed-enabled DarkLight model requires speed or position.")
        smoothing = float(
            upstream["parameters"].get("speed_smoothing_sigma_s", 0.1)
        )
        speed = _derive_speed(position, smoothing_sigma_s=smoothing)

    test_inputs_by_traj = {}
    observed_summaries_by_traj = {}
    observed_bin_edges = _observed_bin_edges(
        path_length_cm,
        parameters["observed_spatial_bin_size_cm"],
    )
    for trajectory_type in TRAJECTORY_TYPES:
        test_inputs = analysis._prepare_test_epoch_inputs_for_units(
            spikes=spikes,
            unit_ids=native_unit_keys,
            trajectory_ep_by_epoch={
                metadata["light_test_epoch"]: dict(trajectory_intervals)
            },
            tp_by_epoch={
                metadata["light_test_epoch"]: dict(
                    task_progression_by_trajectory
                )
            },
            speed_by_epoch=(
                None
                if not bool(shared_metadata["has_speed"])
                else {metadata["light_test_epoch"]: speed}
            ),
            traj_name=trajectory_type,
            epoch=metadata["light_test_epoch"],
            bin_size_s=float(shared_metadata["bin_size_s"]),
            restrict_interval=movement_interval,
        )
        if int(np.asarray(test_inputs["y"]).shape[0]) == 0:
            dataset = _terminal_dataset(
                metadata=metadata,
                unit_ids=selected_unit_ids,
                segment_edges=segment_edges,
                parameters=parameters,
                upstream_provenance=upstream_provenance,
                analysis_status="no_trajectory_samples",
                terminal_detail={"missing_trajectory": trajectory_type},
            )
            return validate_swap_glm_result(
                {
                    "metadata": metadata,
                    "parameters": parameters,
                    "upstream_provenance": upstream_provenance,
                    "selected_units": selected_units,
                    "dataset": dataset,
                    "analysis_status": "no_trajectory_samples",
                    "artifact_origin": "computed",
                    "legacy_artifact_provenance": None,
                }
            )
        test_inputs_by_traj[trajectory_type] = test_inputs
        observed_summaries_by_traj[trajectory_type] = analysis.build_observed_summary(
            test_inputs["y"],
            test_inputs["p"],
            observed_bin_edges,
            bin_size_s=float(shared_metadata["bin_size_s"]),
        )

    selected_datasets = {
        model_name: source_selected_datasets[
            DERIVED_MODEL_SOURCES.get(model_name, model_name)
        ]
        for model_name in MODEL_NAMES
    }
    if dark_light_glm_artifact_path is not None:
        selected_paths = {
            model_name: Path(dark_light_glm_artifact_path)
            / "selected"
            / f"{DERIVED_MODEL_SOURCES.get(model_name, model_name)}.nc"
            for model_name in MODEL_NAMES
        }
    else:
        upstream_id = str(upstream_provenance["dark_light_glm_id"])
        selected_paths = {
            model_name: (
                f"analysis-nwb://{upstream_id}/selected/"
                f"{DERIVED_MODEL_SOURCES.get(model_name, model_name)}"
            )
            for model_name in MODEL_NAMES
        }
    results_by_model = {}
    for model_name in MODEL_NAMES:
        if model_name == VISUAL_ADDITIVE_MODEL_NAME:
            results_by_model[model_name] = (
                _evaluate_visual_additive_delta_on_test_epoch(
                    analysis=analysis,
                    dataset=source_selected_datasets["visual"],
                    test_inputs_by_traj=test_inputs_by_traj,
                    segment_edges=segment_edges,
                    bin_size_s=float(shared_metadata["bin_size_s"]),
                    n_splines=int(shared_metadata["n_splines"]),
                    spline_order=int(shared_metadata["spline_order"]),
                )
            )
            continue
        results_by_model[model_name] = (
            analysis.evaluate_selected_model_on_test_epoch(
                model_name=model_name,
                datasets_by_model=selected_datasets,
                test_inputs_by_traj=test_inputs_by_traj,
                segment_edges=segment_edges,
                bin_size_s=float(shared_metadata["bin_size_s"]),
                n_splines=int(shared_metadata["n_splines"]),
                spline_order=int(shared_metadata["spline_order"]),
                swap_light_offset=parameters["swap_light_offset"],
            )
        )
    source_payload = {} if sources is None else dict(sources)
    source_payload["dark_light_glm_artifact"] = upstream_provenance
    fit_parameters = {
        **parameters,
        "models": list(MODEL_NAMES),
        "scoring_epoch_scope": "all_movement_laps",
        "swapped_component": (
            "local_model_component_with_scalar_light_offset"
            if parameters["swap_light_offset"]
            else "local_model_component_without_scalar_light_offset"
        ),
        "derived_model_sources": dict(DERIVED_MODEL_SOURCES),
        "observed_spatial_bin_size_cm": parameters[
            "observed_spatial_bin_size_cm"
        ],
    }
    dataset = analysis.build_selected_swap_dataset(
        model_names=MODEL_NAMES,
        selected_datasets=selected_datasets,
        selected_paths=selected_paths,
        results_by_model=results_by_model,
        test_inputs_by_traj=test_inputs_by_traj,
        observed_summaries_by_traj=observed_summaries_by_traj,
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        dark_train_epoch=metadata["dark_epoch"],
        light_train_epoch=metadata["light_train_epoch"],
        light_test_epoch=metadata["light_test_epoch"],
        segment_edges=segment_edges,
        observed_bin_edges=observed_bin_edges,
        shared_metadata=shared_metadata,
        sources=source_payload,
        fit_parameters=fit_parameters,
    )
    dataset["selected_source_model"] = (
        ("model",),
        _expected_selected_source_models(MODEL_NAMES, DERIVED_MODEL_SOURCES),
    )
    dataset.attrs.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "analysis_status": "valid",
            "parameter_name": parameters["parameter_name"],
            "parameter_sha256": parameters["parameter_sha256"],
            "output_rule_sha256": parameters["output_rule_sha256"],
            "dark_light_glm_id": upstream_provenance["dark_light_glm_id"],
            **_dark_light_dataset_provenance_attrs(upstream_provenance),
            "dark_light_parameter_sha256": upstream_provenance[
                "dark_light_parameter_sha256"
            ],
            "dark_light_output_rule_sha256": upstream_provenance[
                "dark_light_output_rule_sha256"
            ],
            "upstream_analysis_status": upstream_provenance[
                "upstream_analysis_status"
            ],
            "observed_spatial_bin_size_cm": parameters[
                "observed_spatial_bin_size_cm"
            ],
            "prediction_count_clip_eps": PREDICTION_COUNT_CLIP_EPS,
            "visual_empirical_model_definitions_json": json.dumps(
                dict(VISUAL_EMPIRICAL_MODEL_DEFINITIONS),
                sort_keys=True,
            ),
            "derived_model_sources_json": json.dumps(
                dict(DERIVED_MODEL_SOURCES),
                sort_keys=True,
            ),
            "fit_parameters_json": json.dumps(
                fit_parameters,
                sort_keys=True,
            ),
        }
    )
    selected_units, analysis_status = _audit_scores(selected_units, dataset)
    dataset.attrs["analysis_status"] = analysis_status
    return validate_swap_glm_result(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream_provenance,
            "selected_units": selected_units,
            "dataset": dataset,
            "analysis_status": analysis_status,
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
    )


def _selected_units_sha256(selected_units: pd.DataFrame) -> str:
    """Return the canonical selected-unit identity digest."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    return unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )


def _is_sha256(value: Any) -> bool:
    """Return whether one value is a lowercase or uppercase SHA-256 token."""
    token = str(value)
    return len(token) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in token
    )


def _json_mapping_attr(dataset: Any, name: str) -> dict[str, Any]:
    """Load one required JSON-object dataset attribute."""
    try:
        value = json.loads(str(dataset.attrs[name]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Swap dataset has invalid {name!r}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Swap dataset {name!r} must encode a JSON object.")
    return value


def _boolean_attr(dataset: Any, name: str) -> bool:
    """Return one strict NetCDF-compatible boolean attribute."""
    value = dataset.attrs.get(name)
    if not isinstance(value, (bool, np.bool_, int, np.integer)) or int(
        value
    ) not in (0, 1):
        raise ValueError(f"Swap dataset has invalid boolean attribute {name!r}.")
    return bool(value)


def _allclose_equal_nan(
    observed: Any,
    expected: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> bool:
    """Return strict numeric equality while treating paired NaNs as equal."""
    return np.allclose(
        np.asarray(observed, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


def _expected_selected_source_models(
    model_names: Sequence[str],
    derived_sources: Mapping[str, str],
) -> np.ndarray:
    """Return the selected-file source for each scored model."""
    return np.asarray(
        [derived_sources.get(name, name) for name in model_names],
        dtype=str,
    )


def _validate_swap_score_arithmetic(dataset: Any) -> None:
    """Validate raw-LL, bits/spike, visual-delta, and observed-rate arithmetic."""
    for prefix in METRIC_PREFIXES:
        raw_ll = np.asarray(dataset[f"{prefix}_raw_ll_sum"].values, dtype=float)
        spike_sum = np.asarray(dataset[f"{prefix}_spike_sum"].values, dtype=float)
        bits = np.asarray(
            dataset[f"{prefix}_raw_ll_bits_per_spike"].values,
            dtype=float,
        )
        if np.any(~np.isfinite(spike_sum)) or np.any(spike_sum < 0.0):
            raise ValueError(f"Swap dataset {prefix!r} has invalid spike sums.")
        with np.errstate(divide="ignore", invalid="ignore"):
            expected_bits = np.where(
                spike_sum > 0.0,
                raw_ll / (spike_sum * np.log(2.0)),
                np.nan,
            )
        if not _allclose_equal_nan(bits, expected_bits):
            raise ValueError(
                f"Swap dataset {prefix!r} raw-LL bits/spike arithmetic is stale."
            )
        if not _allclose_equal_nan(
            spike_sum,
            np.broadcast_to(spike_sum[[0]], spike_sum.shape),
        ):
            raise ValueError(
                f"Swap dataset {prefix!r} spike sums differ across models."
            )

    for scope in ("swapped_segment", "full"):
        unswapped = dataset[
            f"test_light_{scope}_unswapped_spike_sum"
        ].values
        swapped = dataset[f"test_light_{scope}_swapped_spike_sum"].values
        if not _allclose_equal_nan(unswapped, swapped):
            raise ValueError(
                f"Swap dataset {scope!r} spike sums differ by prediction rule."
            )

    swapped_bits = np.asarray(
        dataset[
            "test_light_swapped_segment_swapped_raw_ll_bits_per_spike"
        ].values,
        dtype=float,
    )
    expected_delta = swapped_bits - swapped_bits[[0]]
    expected_delta[0] = np.nan
    if not _allclose_equal_nan(dataset[PRIMARY_METRIC].values, expected_delta):
        raise ValueError("Swap dataset model-minus-visual score arithmetic is stale.")

    occupancy = np.asarray(dataset["test_light_occupancy_s"].values, dtype=float)
    spike_count = np.asarray(dataset["test_light_spike_count"].values, dtype=float)
    observed_rate = np.asarray(
        dataset["test_light_observed_rate_hz"].values,
        dtype=float,
    )
    if (
        np.any(~np.isfinite(occupancy))
        or np.any(occupancy < 0.0)
        or np.any(~np.isfinite(spike_count))
        or np.any(spike_count < 0.0)
    ):
        raise ValueError("Swap observed occupancy or spike counts are invalid.")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_rate = np.where(
            occupancy[:, :, None] > 0.0,
            spike_count / occupancy[:, :, None],
            np.nan,
        )
    if not _allclose_equal_nan(observed_rate, expected_rate):
        raise ValueError("Swap observed-rate arithmetic is stale.")
    full_spikes = np.asarray(
        dataset["test_light_full_swapped_spike_sum"].values,
        dtype=float,
    )[0]
    if not _allclose_equal_nan(full_spikes, np.sum(spike_count, axis=1)):
        raise ValueError("Swap full-epoch spike sums disagree with observed bins.")

    bin_size_s = float(dataset.attrs["bin_size_s"])
    full_n_bins = np.asarray(dataset["test_light_full_n_bins"].values, dtype=float)
    segment_n_bins = np.asarray(
        dataset["test_light_swapped_segment_n_bins"].values,
        dtype=float,
    )
    if (
        np.any(~np.isfinite(full_n_bins))
        or np.any(~np.isfinite(segment_n_bins))
        or np.any(full_n_bins != np.rint(full_n_bins))
        or np.any(segment_n_bins != np.rint(segment_n_bins))
        or np.any(segment_n_bins < 0.0)
        or np.any(segment_n_bins > full_n_bins)
        or not _allclose_equal_nan(
            full_n_bins,
            np.sum(occupancy, axis=1) / bin_size_s,
        )
    ):
        raise ValueError("Swap held-out bin counts are inconsistent.")


def _validate_nonterminal_swap_dataset(
    dataset: Any,
    *,
    model_names: Sequence[str] = MODEL_NAMES,
    require_selected_source_model: bool = True,
    derived_sources: Mapping[str, str] = DERIVED_MODEL_SOURCES,
    expected_schema_version: str = RESULT_SCHEMA_VERSION,
    extra_data_variables: Sequence[str] = (),
) -> None:
    """Validate a complete current or explicitly described legacy swap schema."""
    model_names = tuple(model_names)
    expected_variables = set(CANONICAL_DATA_VARIABLE_DIMS)
    if not require_selected_source_model:
        expected_variables.remove("selected_source_model")
    expected_variables.update(extra_data_variables)
    if set(dataset.data_vars) != expected_variables:
        missing = sorted(expected_variables.difference(dataset.data_vars))
        extra = sorted(set(dataset.data_vars).difference(expected_variables))
        raise ValueError(
            "Swap dataset variables differ from the complete schema; "
            f"missing={missing!r}, extra={extra!r}."
        )
    if set(dataset.coords) != set(CANONICAL_COORDINATE_DIMS):
        raise ValueError("Swap dataset coordinates differ from the complete schema.")
    for name, dims in CANONICAL_COORDINATE_DIMS.items():
        if tuple(dataset.coords[name].dims) != dims:
            raise ValueError(f"Swap coordinate {name!r} has noncanonical dimensions.")
    for name, dims in CANONICAL_DATA_VARIABLE_DIMS.items():
        if name == "selected_source_model" and not require_selected_source_model:
            continue
        if tuple(dataset[name].dims) != dims:
            raise ValueError(f"Swap variable {name!r} has noncanonical dimensions.")
    for name in extra_data_variables:
        if tuple(dataset[name].dims) != ("model", "trajectory", "unit"):
            raise ValueError(
                f"Legacy schema-6 diagnostic {name!r} has invalid dimensions."
            )
    if str(dataset.attrs.get("schema_version", "")) != str(
        expected_schema_version
    ):
        raise ValueError("Swap dataset has an unexpected schema_version.")
    for name, expected in FIXED_SCIENTIFIC_ATTRS.items():
        if str(dataset.attrs.get(name, "")) != expected:
            raise ValueError(f"Swap dataset has mismatched fixed attribute {name!r}.")
    if not np.array_equal(
        np.asarray(dataset.coords["model"].values, dtype=str),
        np.asarray(model_names, dtype=str),
    ):
        raise ValueError("Swap dataset has a noncanonical model coordinate.")
    if not np.array_equal(
        np.asarray(dataset.coords["trajectory"].values, dtype=str),
        np.asarray(TRAJECTORY_TYPES, dtype=str),
    ):
        raise ValueError("Swap dataset has a noncanonical trajectory coordinate.")
    unit_ids = np.asarray(dataset.coords["unit"].values).astype(str)
    if len(set(unit_ids.tolist())) != len(unit_ids) or any(
        not value for value in unit_ids
    ):
        raise ValueError("Swap dataset unit coordinates must be unique and non-empty.")
    for name in ("tp_grid", "tp_observed_edge", "segment_edge"):
        values = np.asarray(dataset.coords[name].values, dtype=float)
        if (
            values.ndim != 1
            or values.size < 2
            or np.any(~np.isfinite(values))
            or np.any(np.diff(values) <= 0.0)
        ):
            raise ValueError(f"Swap coordinate {name!r} must strictly increase.")
    tp_grid = np.asarray(dataset.coords["tp_grid"].values, dtype=float)
    segment_edges = np.asarray(dataset.coords["segment_edge"].values, dtype=float)
    observed_edges = np.asarray(
        dataset.coords["tp_observed_edge"].values,
        dtype=float,
    )
    observed_bins = np.asarray(
        dataset.coords["tp_observed_bin"].values,
        dtype=float,
    )
    if (
        not np.allclose(tp_grid[[0, -1]], [0.0, 1.0])
        or not np.allclose(segment_edges[[0, -1]], [0.0, 1.0])
        or not np.isclose(observed_edges[0], 0.0)
        or observed_edges[-1] < 1.0
        or not _allclose_equal_nan(
            observed_bins,
            0.5 * (observed_edges[:-1] + observed_edges[1:]),
        )
    ):
        raise ValueError("Swap task-progression coordinates are inconsistent.")

    if require_selected_source_model and not np.array_equal(
        np.asarray(dataset["selected_source_model"].values, dtype=str),
        _expected_selected_source_models(model_names, derived_sources),
    ):
        raise ValueError("Swap selected-source model mapping is inconsistent.")
    selected_paths = np.asarray(dataset["selected_model_path"].values, dtype=str)
    if any(not value for value in selected_paths):
        raise ValueError("Swap selected-model paths must be non-empty.")
    selected_ridges = np.asarray(dataset["selected_ridge"].values, dtype=float)
    selected_scores = np.asarray(dataset["selected_score"].values, dtype=float)
    if (
        np.any(~np.isfinite(selected_ridges))
        or np.any(selected_ridges < 0.0)
        or np.any(~np.isfinite(selected_scores))
    ):
        raise ValueError("Swap selected-model ridge or score metadata is invalid.")

    expected_sources = np.asarray(
        [
            OUTPUT_RULE["swap_configuration"][trajectory]["source_trajectory"]
            for trajectory in TRAJECTORY_TYPES
        ],
        dtype=str,
    )
    expected_segment_indices = np.asarray(
        [
            OUTPUT_RULE["swap_configuration"][trajectory]["segment_index"] + 1
            for trajectory in TRAJECTORY_TYPES
        ],
        dtype=int,
    )
    if not np.array_equal(
        np.asarray(dataset["swap_source_trajectory"].values, dtype=str),
        expected_sources,
    ) or not np.array_equal(
        np.asarray(dataset["swap_segment_index_1based"].values, dtype=int),
        expected_segment_indices,
    ):
        raise ValueError("Swap source-trajectory or segment mapping is inconsistent.")
    zero_based = expected_segment_indices - 1
    if not _allclose_equal_nan(
        dataset["swap_segment_start"].values,
        segment_edges[zero_based],
    ) or not _allclose_equal_nan(
        dataset["swap_segment_end"].values,
        segment_edges[zero_based + 1],
    ):
        raise ValueError("Swap segment bounds disagree with segment_edge.")

    if _boolean_attr(dataset, "swap_light_offset") not in (True, False):
        raise AssertionError("Unreachable invalid swap_light_offset.")
    for name in ("bin_size_s",):
        value = float(dataset.attrs.get(name, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Swap dataset has invalid {name!r}.")
    for name in ("n_splines", "spline_order", "n_speed_features"):
        try:
            value = int(dataset.attrs[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Swap dataset has invalid {name!r}.") from exc
        if value < (0 if name == "n_speed_features" else 1):
            raise ValueError(f"Swap dataset has invalid {name!r}.")
    _boolean_attr(dataset, "has_speed")
    if str(dataset.attrs.get("speed_feature_mode", "")) not in {
        "none",
        "linear",
        "bspline",
    }:
        raise ValueError("Swap dataset has invalid speed_feature_mode.")
    if _json_mapping_attr(dataset, "swap_rule_json") != dict(
        OUTPUT_RULE["swap_configuration"]
    ):
        raise ValueError("Swap dataset has a stale swap_rule_json mapping.")
    if _json_mapping_attr(dataset, "derived_model_sources_json") != dict(
        derived_sources
    ):
        raise ValueError("Swap dataset has a stale derived-model mapping.")
    _json_mapping_attr(dataset, "sources_json")
    fit_parameters = _json_mapping_attr(dataset, "fit_parameters_json")
    if tuple(fit_parameters.get("models", ())) != model_names or dict(
        fit_parameters.get("derived_model_sources", {})
    ) != dict(derived_sources):
        raise ValueError("Swap fit_parameters_json has stale model mappings.")
    if str(fit_parameters.get("scoring_epoch_scope", "")) != "all_movement_laps":
        raise ValueError("Swap fit_parameters_json has a stale scoring scope.")
    if bool(fit_parameters.get("swap_light_offset")) != _boolean_attr(
        dataset,
        "swap_light_offset",
    ):
        raise ValueError("Swap fit parameters disagree on swap_light_offset.")
    expected_component = (
        "local_model_component_with_scalar_light_offset"
        if _boolean_attr(dataset, "swap_light_offset")
        else "local_model_component_without_scalar_light_offset"
    )
    if str(fit_parameters.get("swapped_component", "")) != expected_component:
        raise ValueError("Swap fit parameters have a stale swapped component.")
    if str(expected_schema_version) == RESULT_SCHEMA_VERSION:
        prediction_clip = float(
            dataset.attrs.get("prediction_count_clip_eps", np.nan)
        )
        if not np.isclose(
            prediction_clip,
            PREDICTION_COUNT_CLIP_EPS,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise ValueError(
                "Swap dataset has stale additive prediction clipping."
            )
        if _json_mapping_attr(
            dataset,
            "visual_empirical_model_definitions_json",
        ) != dict(VISUAL_EMPIRICAL_MODEL_DEFINITIONS):
            raise ValueError(
                "Swap dataset has stale visual empirical-model definitions."
            )
    _validate_swap_score_arithmetic(dataset)


def _validate_loaded_schema4_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one immutable historical schema-4 bundle."""
    copied = dict(result)
    metadata = _metadata(**dict(copied["metadata"]))
    raw_parameters = dict(copied["parameters"])
    validated_parameters = validate_swap_glm_parameters(
        swap_light_offset=raw_parameters["swap_light_offset"],
        observed_spatial_bin_size_cm=raw_parameters[
            "observed_spatial_bin_size_cm"
        ],
    )
    parameter_name = _path_component(
        raw_parameters["parameter_name"],
        name="parameter_name",
    )
    expected_parameter_sha256 = _provenance_sha256(
        {
            "swap_glm_param_name": parameter_name,
            **validated_parameters,
        }
    )
    if str(raw_parameters["parameter_sha256"]) != expected_parameter_sha256:
        raise ValueError("Historical parameter_sha256 is stale.")
    if str(raw_parameters["output_rule_sha256"]) != (
        LEGACY_SCHEMA4_OUTPUT_RULE_SHA256
    ):
        raise ValueError("Historical output_rule_sha256 is stale.")
    parameters = {
        "parameter_name": parameter_name,
        "parameter_sha256": expected_parameter_sha256,
        "output_rule_sha256": LEGACY_SCHEMA4_OUTPUT_RULE_SHA256,
        **validated_parameters,
    }

    upstream = dict(copied["upstream_provenance"])
    selected_hashes = dict(
        upstream.get("dark_light_selected_sha256_by_model", {})
    )
    if set(selected_hashes) != set(SOURCE_MODEL_NAMES) or any(
        not _is_sha256(value) for value in selected_hashes.values()
    ):
        raise ValueError("Historical DarkLight selected-model digests are incomplete.")
    for name in (
        "dark_light_manifest_sha256",
        "dark_light_parameter_sha256",
        "dark_light_output_rule_sha256",
    ):
        if not _is_sha256(upstream.get(name, "")):
            raise ValueError(f"Historical DarkLight {name} must be SHA-256.")
    _uuid_string(upstream.get("dark_light_glm_id"), name="dark_light_glm_id")

    selected_units = copied["selected_units"].copy()
    if tuple(selected_units.columns) != SELECTED_UNIT_COLUMNS:
        raise ValueError("Historical selected_units schema is stale.")
    for name in IDENTITY_COLUMNS:
        selected_units[name] = selected_units[name].astype(str)
    if (
        selected_units["stable_unit_id"].duplicated().any()
        or selected_units["group_unit_id"].duplicated().any()
        or selected_units.duplicated(
            subset=["spikesorting_merge_id", "unit_id"]
        ).any()
    ):
        raise ValueError("Historical selected-unit identities are not unique.")
    if not np.array_equal(
        selected_units["selection_index"].to_numpy(dtype=int),
        np.arange(len(selected_units), dtype=int),
    ):
        raise ValueError("Historical selection_index is not contiguous.")

    dataset = copied["dataset"]
    if str(dataset.attrs.get("schema_version", "")) != "4" or not np.array_equal(
        np.asarray(dataset.coords.get("model", ()), dtype=str),
        np.asarray(LEGACY_SCHEMA4_MODEL_NAMES, dtype=str),
    ):
        raise ValueError("Historical bundle is not canonical schema 4.")
    expected_metadata = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_train_epoch": metadata["dark_epoch"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
    }
    for name, expected in expected_metadata.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"Historical swap dataset has mismatched {name!r}.")
    if not np.array_equal(
        np.asarray(dataset.coords.get("unit", ()), dtype=str),
        selected_units["group_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Historical swap dataset unit order is stale.")
    status = str(copied["analysis_status"])
    if status not in ANALYSIS_STATUSES or str(
        dataset.attrs.get("analysis_status", "")
    ) != status:
        raise ValueError("Historical swap analysis_status is stale.")
    expected_score_count = (
        (len(LEGACY_SCHEMA4_MODEL_NAMES) - 1) * len(TRAJECTORY_TYPES)
    )
    if np.any(
        selected_units["n_expected_primary_scores"].to_numpy(dtype=int)
        != expected_score_count
    ):
        raise ValueError("Historical selected-unit score counts are stale.")
    terminal_statuses = {
        "upstream_terminal",
        "no_units",
        "no_valid_position",
        "no_movement",
        "no_trajectory_samples",
    }
    if status in terminal_statuses:
        if str(dataset.attrs.get("fit_stage", "")) != "terminal":
            raise ValueError("Historical terminal swap marker is stale.")
    else:
        _validate_nonterminal_swap_dataset(
            dataset,
            model_names=LEGACY_SCHEMA4_MODEL_NAMES,
            derived_sources=LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES,
            expected_schema_version="4",
        )
        selected_units, derived_status = _audit_scores(
            selected_units,
            dataset,
            model_names=LEGACY_SCHEMA4_MODEL_NAMES,
        )
        if derived_status != status:
            raise ValueError("Historical swap score audit is stale.")

    copied.update(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream,
            "selected_units": selected_units,
            "analysis_status": status,
            "selected_units_sha256": _selected_units_sha256(selected_units),
            "n_units": len(selected_units),
            "n_valid_units": int(selected_units["valid_swap_score"].sum()),
            "historical_schema_version": "4",
        }
    )
    return copied


def validate_swap_glm_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one in-memory held-out swap result."""
    required = {
        "metadata",
        "parameters",
        "upstream_provenance",
        "selected_units",
        "dataset",
        "analysis_status",
        "artifact_origin",
        "legacy_artifact_provenance",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"Swap result is missing fields {missing!r}.")
    if str(result["dataset"].attrs.get("schema_version", "")) == "4":
        return _validate_loaded_schema4_result(result)
    copied = dict(result)
    metadata = _metadata(**dict(copied["metadata"]))
    raw_parameters = dict(copied["parameters"])
    parameters = _effective_parameters(
        parameter_name=raw_parameters["parameter_name"],
        parameter_sha256=raw_parameters["parameter_sha256"],
        output_rule_sha256=raw_parameters["output_rule_sha256"],
        swap_light_offset=raw_parameters["swap_light_offset"],
        observed_spatial_bin_size_cm=raw_parameters[
            "observed_spatial_bin_size_cm"
        ],
    )
    upstream = dict(copied["upstream_provenance"])
    for name in (
        "dark_light_glm_id",
        "dark_light_parameter_sha256",
        "dark_light_output_rule_sha256",
        "upstream_analysis_status",
    ):
        if name not in upstream:
            raise ValueError(f"Upstream provenance is missing {name!r}.")
    _uuid_string(upstream["dark_light_glm_id"], name="dark_light_glm_id")
    provenance_style = _dark_light_provenance_style(upstream)
    result_digest_field = (
        "dark_light_manifest_sha256"
        if provenance_style == "legacy"
        else "dark_light_glm_sha256"
    )
    if not _is_sha256(upstream[result_digest_field]):
        raise ValueError("DarkLight result digest must be SHA-256.")
    for name in (
        "dark_light_parameter_sha256",
        "dark_light_output_rule_sha256",
    ):
        if not _is_sha256(upstream[name]):
            raise ValueError(f"DarkLight {name} must be SHA-256.")
    selected_hashes = _dark_light_selected_hashes(upstream)
    if set(selected_hashes) != set(SOURCE_MODEL_NAMES) or any(
        not _is_sha256(value) for value in selected_hashes.values()
    ):
        raise ValueError("DarkLight selected-model digests are incomplete.")
    upstream_status = str(upstream["upstream_analysis_status"])
    from v1ca1.spyglass.dark_light_glm import (
        ANALYSIS_STATUSES as DARK_LIGHT_ANALYSIS_STATUSES,
    )

    if upstream_status not in DARK_LIGHT_ANALYSIS_STATUSES:
        raise ValueError("DarkLight upstream_analysis_status is unsupported.")
    selected_units = copied["selected_units"].copy()
    if tuple(selected_units.columns) != SELECTED_UNIT_COLUMNS:
        raise ValueError("selected_units does not have the canonical schema.")
    for field in IDENTITY_COLUMNS:
        selected_units[field] = selected_units[field].astype(str)
        if (selected_units[field].str.len() == 0).any():
            raise ValueError(f"selected_units {field} values must be non-empty.")
    if (
        selected_units["stable_unit_id"].duplicated().any()
        or selected_units["group_unit_id"].duplicated().any()
        or selected_units.duplicated(
            subset=["spikesorting_merge_id", "unit_id"]
        ).any()
    ):
        raise ValueError("selected_units identities must be unique.")
    expected_stable_ids = (
        selected_units["spikesorting_merge_id"]
        + ":"
        + selected_units["unit_id"]
    )
    if not np.array_equal(
        selected_units["stable_unit_id"].to_numpy(dtype=str),
        expected_stable_ids.to_numpy(dtype=str),
    ):
        raise ValueError("selected_units stable identities are inconsistent.")
    if not np.array_equal(
        selected_units["selection_index"].to_numpy(dtype=int),
        np.arange(len(selected_units), dtype=int),
    ):
        raise ValueError("selected_units selection_index must be contiguous.")
    for field in (
        "dark_movement_firing_rate_hz",
        "light_movement_firing_rate_hz",
        "test_light_spike_count",
    ):
        values = selected_units[field].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"selected_units {field} must be finite and non-negative."
            )
    for field in ("upstream_valid_glm_fit", "valid_swap_score"):
        values = selected_units[field].to_numpy()
        if any(not isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"selected_units {field} must be boolean.")
        selected_units[field] = values.astype(bool)
    finite_counts = selected_units["n_finite_primary_scores"].to_numpy(
        dtype=float
    )
    expected_counts = selected_units["n_expected_primary_scores"].to_numpy(
        dtype=float
    )
    expected_score_count = (len(MODEL_NAMES) - 1) * len(TRAJECTORY_TYPES)
    if (
        np.any(~np.isfinite(finite_counts))
        or np.any(~np.isfinite(expected_counts))
        or np.any(finite_counts != np.rint(finite_counts))
        or np.any(expected_counts != expected_score_count)
        or np.any(finite_counts < 0)
        or np.any(finite_counts > expected_score_count)
    ):
        raise ValueError("selected_units primary-score QC counts are invalid.")
    expected_valid = (
        selected_units["upstream_valid_glm_fit"].to_numpy(dtype=bool)
        & (finite_counts == expected_score_count)
    )
    if not np.array_equal(
        selected_units["valid_swap_score"].to_numpy(dtype=bool),
        expected_valid,
    ):
        raise ValueError("selected_units valid_swap_score is inconsistent.")
    dataset = copied["dataset"]
    expected_attrs = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
    }
    for name, expected in expected_attrs.items():
        if str(dataset.attrs.get(name, "")) != str(expected):
            raise ValueError(f"Swap dataset has mismatched {name!r}.")
    observed_dark = dataset.attrs.get(
        "dark_train_epoch", dataset.attrs.get("dark_epoch", "")
    )
    if str(observed_dark) != metadata["dark_epoch"]:
        raise ValueError("Swap dataset has mismatched dark epoch.")
    for coord, expected in (
        ("model", MODEL_NAMES),
        ("trajectory", TRAJECTORY_TYPES),
    ):
        if coord not in dataset.coords or not np.array_equal(
            np.asarray(dataset.coords[coord].values, dtype=str),
            np.asarray(expected, dtype=str),
        ):
            raise ValueError(f"Swap dataset has a noncanonical {coord!r} coordinate.")
    dataset_units = np.asarray(dataset.coords["unit"].values).astype(str)
    audit_units = selected_units["group_unit_id"].astype(str).to_numpy()
    if not np.array_equal(dataset_units, audit_units):
        raise ValueError("Swap dataset unit order differs from selected_units.")
    status = str(copied["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError("Unsupported swap analysis_status.")
    if str(dataset.attrs.get("analysis_status", "")) != status:
        raise ValueError("Swap dataset analysis_status does not match the bundle.")
    if str(dataset.attrs.get("schema_version", "")) != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Swap dataset schema_version is not canonical v{RESULT_SCHEMA_VERSION}."
        )
    expected_dataset_attrs = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "parameter_name": parameters["parameter_name"],
        "parameter_sha256": parameters["parameter_sha256"],
        "output_rule_sha256": parameters["output_rule_sha256"],
        "swap_light_offset": parameters["swap_light_offset"],
        "observed_spatial_bin_size_cm": parameters[
            "observed_spatial_bin_size_cm"
        ],
        "dark_light_glm_id": upstream["dark_light_glm_id"],
        "dark_light_parameter_sha256": upstream[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": upstream[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": upstream_status,
        **{
            key: value
            for key, value in _dark_light_dataset_provenance_attrs(
                upstream
            ).items()
            if not key.endswith("_json")
        },
    }
    for name, expected in expected_dataset_attrs.items():
        observed = dataset.attrs.get(name)
        if isinstance(expected, bool):
            matches = isinstance(
                observed,
                (bool, np.bool_, int, np.integer),
            ) and int(observed) in (0, 1) and bool(observed) == expected
        elif isinstance(expected, float):
            try:
                matches = np.isclose(
                    float(observed), expected, rtol=1e-12, atol=1e-12
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(observed) == str(expected)
        if not matches:
            raise ValueError(f"Swap dataset has mismatched {name!r}.")
    try:
        selected_attr = (
            "dark_light_selected_sha256_json"
            if provenance_style == "legacy"
            else "dark_light_selected_model_sha256_json"
        )
        dataset_selected_hashes = json.loads(str(dataset.attrs[selected_attr]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Swap dataset lacks DarkLight selected-model provenance."
        ) from exc
    if dataset_selected_hashes != selected_hashes:
        raise ValueError(
            "Swap dataset DarkLight selected-model provenance is mismatched."
        )
    terminal_statuses = {
        "upstream_terminal",
        "no_units",
        "no_valid_position",
        "no_movement",
        "no_trajectory_samples",
    }
    if status in terminal_statuses:
        if str(dataset.attrs.get("fit_stage", "")) != "terminal":
            raise ValueError("Terminal swap datasets must have fit_stage='terminal'.")
        if np.any(selected_units["valid_swap_score"].to_numpy(dtype=bool)):
            raise ValueError("Terminal swap results cannot contain valid scores.")
    elif str(dataset.attrs.get("fit_source", "")) != "dark_light_glm_selected":
        raise ValueError("Nonterminal swap dataset has an invalid fit_source.")
    if status == "upstream_terminal":
        if upstream_status in {"valid", "partial_valid"}:
            raise ValueError("upstream_terminal requires a terminal DarkLight status.")
    elif upstream_status not in {"valid", "partial_valid"}:
        raise ValueError("Non-upstream-terminal result has terminal DarkLight input.")
    if status == "no_units" and not selected_units.empty:
        raise ValueError("no_units requires an empty selected-unit audit.")
    if status == "valid" and (
        selected_units.empty or not np.all(expected_valid)
    ):
        raise ValueError("valid requires every selected unit to pass score QC.")
    if status == "partial_valid" and not (
        np.any(expected_valid) and not np.all(expected_valid)
    ):
        raise ValueError("partial_valid requires some but not all valid units.")
    if status == "no_valid_units" and (
        selected_units.empty or np.any(expected_valid)
    ):
        raise ValueError("no_valid_units requires selected units and no valid scores.")
    if status == "no_trajectory_samples" and str(
        dataset.attrs.get("missing_trajectory", "")
    ) not in TRAJECTORY_TYPES:
        raise ValueError(
            "no_trajectory_samples requires the missing trajectory identity."
        )
    if status in {"valid", "partial_valid", "no_valid_units"}:
        _validate_nonterminal_swap_dataset(dataset)
        recomputed_units, recomputed_status = _audit_scores(
            selected_units,
            dataset,
        )
        for field in (
            "test_light_spike_count",
            "n_finite_primary_scores",
            "n_expected_primary_scores",
            "valid_swap_score",
        ):
            if not np.array_equal(
                selected_units[field].to_numpy(),
                recomputed_units[field].to_numpy(),
            ):
                raise ValueError(
                    f"selected_units {field} does not match the swap dataset."
                )
        if status != recomputed_status:
            raise ValueError(
                "Swap analysis_status does not match dataset-derived unit QC."
            )
    origin = str(copied["artifact_origin"])
    if origin not in {"computed", "registered_existing"}:
        raise ValueError("Unsupported artifact_origin.")
    legacy_provenance = copied["legacy_artifact_provenance"]
    if origin == "computed" and legacy_provenance is not None:
        raise ValueError("Computed swap artifacts cannot carry legacy provenance.")
    if origin == "registered_existing" and (
        not isinstance(legacy_provenance, Mapping) or not legacy_provenance
    ):
        raise ValueError(
            "Registered swap artifacts require non-empty legacy provenance."
        )
    selected_digest = _selected_units_sha256(selected_units)
    copied.update(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream,
            "selected_units": selected_units,
            "analysis_status": status,
            "artifact_origin": origin,
            "legacy_artifact_provenance": (
                None
                if legacy_provenance is None
                else dict(legacy_provenance)
            ),
            "selected_units_sha256": selected_digest,
            "n_units": len(selected_units),
            "n_valid_units": int(selected_units["valid_swap_score"].sum()),
        }
    )
    return copied


def _manifest_common(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return manifest values shared by both bundle artifacts."""
    upstream = result["upstream_provenance"]
    legacy = result["legacy_artifact_provenance"]
    return {
        **result["metadata"],
        **result["parameters"],
        "dark_light_glm_id": upstream["dark_light_glm_id"],
        "dark_light_manifest_sha256": upstream[
            "dark_light_manifest_sha256"
        ],
        "dark_light_selected_sha256_json": json.dumps(
            upstream["dark_light_selected_sha256_by_model"], sort_keys=True
        ),
        "dark_light_parameter_sha256": upstream[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": upstream[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": upstream["upstream_analysis_status"],
        "n_units": result["n_units"],
        "n_valid_units": result["n_valid_units"],
        "selected_units_sha256": result["selected_units_sha256"],
        "analysis_status": result["analysis_status"],
        "artifact_origin": result["artifact_origin"],
        "legacy_artifact_provenance_json": json.dumps(
            {} if legacy is None else legacy,
            sort_keys=True,
        ),
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
    }


def write_swap_glm_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write and reload one complete held-out swap bundle."""
    result = validate_swap_glm_result(result)
    destination = Path(path)
    if destination.name != result["metadata"]["swap_glm_id"]:
        raise ValueError("Artifact directory name must equal swap_glm_id.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite swap artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    try:
        result["selected_units"].to_parquet(
            temporary / SELECTED_UNITS_FILENAME,
            index=False,
        )
        result["dataset"].to_netcdf(temporary / RESULT_FILENAME)
        common = _manifest_common(result)
        rows = []
        for artifact_key, filename, artifact_kind in (
            ("selected_units", SELECTED_UNITS_FILENAME, "parquet"),
            ("swap_glm", RESULT_FILENAME, "netcdf"),
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
        load_swap_glm_artifact(temporary, _allow_temporary_name=True)
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
        "result_path": destination / RESULT_FILENAME,
    }


def load_swap_glm_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one complete held-out swap bundle."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Swap manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or len(manifest) != 2:
        raise ValueError("Swap manifest does not have the canonical schema.")
    expected = {
        "selected_units": (SELECTED_UNITS_FILENAME, "parquet"),
        "swap_glm": (RESULT_FILENAME, "netcdf"),
    }
    if set(manifest["artifact_key"].astype(str)) != set(expected):
        raise ValueError("Swap manifest does not contain the canonical artifacts.")
    for _, row in manifest.iterrows():
        filename, kind = expected[str(row["artifact_key"])]
        if str(row["relative_path"]) != filename or str(row["artifact_kind"]) != kind:
            raise ValueError("Swap manifest artifact names or kinds are inconsistent.")
        artifact_path = directory / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Swap artifact not found: {artifact_path}")
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(f"Swap artifact checksum mismatch: {artifact_path}")
    first = manifest.iloc[0]
    for name in MANIFEST_COLUMNS[5:]:
        if not np.all(manifest[name].astype(str) == str(first[name])):
            raise ValueError(f"Swap manifest has inconsistent {name!r} values.")
    metadata = {
        name: str(first[name])
        for name in (
            "swap_glm_id",
            "animal_name",
            "date",
            "region",
            "dark_epoch",
            "light_train_epoch",
            "light_test_epoch",
        )
    }
    if not _allow_temporary_name and directory.name != metadata["swap_glm_id"]:
        raise ValueError("Artifact directory name does not match swap_glm_id.")
    parameters = {
        "parameter_name": str(first["parameter_name"]),
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        "swap_light_offset": bool(first["swap_light_offset"]),
        "observed_spatial_bin_size_cm": float(
            first["observed_spatial_bin_size_cm"]
        ),
    }
    upstream = {
        "dark_light_glm_id": str(first["dark_light_glm_id"]),
        "dark_light_manifest_sha256": str(first["dark_light_manifest_sha256"]),
        "dark_light_selected_sha256_by_model": json.loads(
            str(first["dark_light_selected_sha256_json"])
        ),
        "dark_light_parameter_sha256": str(
            first["dark_light_parameter_sha256"]
        ),
        "dark_light_output_rule_sha256": str(
            first["dark_light_output_rule_sha256"]
        ),
        "upstream_analysis_status": str(first["upstream_analysis_status"]),
    }
    legacy = json.loads(str(first["legacy_artifact_provenance_json"]))
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "upstream_provenance": upstream,
        "selected_units": pd.read_parquet(directory / SELECTED_UNITS_FILENAME),
        "dataset": _load_dataset(directory / RESULT_FILENAME),
        "analysis_status": str(first["analysis_status"]),
        "artifact_origin": str(first["artifact_origin"]),
        "legacy_artifact_provenance": legacy or None,
        "manifest": manifest,
    }
    validated = validate_swap_glm_result(result)
    if (
        validated["n_units"] != int(first["n_units"])
        or validated["n_valid_units"] != int(first["n_valid_units"])
        or validated["selected_units_sha256"]
        != str(first["selected_units_sha256"])
    ):
        raise ValueError("Swap manifest summary does not match loaded artifacts.")
    return validated


_SELECTED_UNIT_TEXT_COLUMNS = IDENTITY_COLUMNS
_SELECTED_UNIT_INTEGER_COLUMNS = (
    "selection_index",
    "n_finite_primary_scores",
    "n_expected_primary_scores",
)
_SELECTED_UNIT_BOOLEAN_COLUMNS = (
    "upstream_valid_glm_fit",
    "valid_swap_score",
)


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
    vector_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return a canonical typed frame for one SwapGLM NWB object."""
    if not isinstance(table, pd.DataFrame) or tuple(table.columns) != tuple(
        columns
    ):
        raise ValueError("SwapGLM NWB table does not have its canonical schema.")
    output = table.copy().reset_index(drop=True)
    for column in text_columns:
        output[column] = output[column].map(str)
    for column in integer_columns:
        output[column] = pd.to_numeric(
            output[column], errors="raise"
        ).astype(np.int64)
    for column in boolean_columns:
        values = output[column].tolist()
        if not all(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError(f"SwapGLM NWB column {column!r} must be boolean.")
        output[column] = np.asarray(values, dtype=bool)
    for column in vector_columns:
        vectors = [np.asarray(value, dtype=float) for value in output[column]]
        if any(vector.ndim != 1 or np.isinf(vector).any() for vector in vectors):
            raise ValueError(
                f"SwapGLM NWB vector column {column!r} is invalid."
            )
        output[column] = vectors
    for column in columns:
        if column in (
            *text_columns,
            *integer_columns,
            *boolean_columns,
            *vector_columns,
        ):
            continue
        output[column] = pd.to_numeric(
            output[column], errors="raise"
        ).astype(float)
        if np.isinf(output[column].to_numpy(dtype=float)).any():
            raise ValueError(
                f"SwapGLM NWB numeric column {column!r} contains infinity."
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
    """Convert a scalar frame to an NWB DynamicTable."""
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


def _ragged_dynamic_table_from_frame(
    table: pd.DataFrame,
    *,
    name: str,
    description: str,
    columns: Sequence[str],
    vector_columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> Any:
    """Convert scalar keys and aligned numeric vectors to a DynamicTable."""
    from hdmf.common import DynamicTable

    canonical = _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
        vector_columns=vector_columns,
    )
    if canonical.empty:
        return _empty_nwb_dynamic_table(
            name=name,
            description=description,
            columns=columns,
            text_columns=text_columns,
            integer_columns=integer_columns,
            boolean_columns=boolean_columns,
            ragged_columns=vector_columns,
        )
    scalar_columns = tuple(
        column for column in columns if column not in vector_columns
    )
    output = DynamicTable.from_dataframe(
        name=name,
        df=canonical.loc[:, list(scalar_columns)],
        table_description=description,
        columns=[
            {"name": column, "description": _nwb_column_description(column)}
            for column in scalar_columns
        ],
    )
    for column in vector_columns:
        vectors = canonical[column].tolist()
        if all(vector.size == 0 for vector in vectors):
            vectors = [np.asarray([np.nan], dtype=float) for _ in vectors]
        output.add_column(
            name=column,
            description=_nwb_column_description(column),
            data=vectors,
            index=True,
        )
    return output


def _decode_nwb_text(value: Any) -> str:
    """Return text after an HDF5-backed DynamicTable round trip."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _frame_from_dynamic_table(
    nwb_table: Any,
    *,
    expected_name: str,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    vector_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Load one SwapGLM DynamicTable or Spyglass-fetched DataFrame."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table.copy()
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != expected_name:
            raise ValueError(
                f"Unexpected SwapGLM NWB object {nwb_table.name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("SwapGLM NWB objects must be DynamicTables.")
    table = table.reset_index(drop=True)
    observed = tuple(str(column) for column in table.columns)
    if set(observed) != set(columns) or len(observed) != len(columns):
        raise ValueError("SwapGLM NWB object has a noncanonical schema.")
    table = table.loc[:, list(columns)]
    for column in text_columns:
        table[column] = table[column].map(_decode_nwb_text)
    for column in vector_columns:
        table[column] = [
            np.asarray(value, dtype=float) for value in table[column]
        ]
        if table[column].map(
            lambda value: value.shape == (1,) and np.isnan(value[0])
        ).all():
            table[column] = [
                np.asarray([], dtype=float) for _ in table[column]
            ]
    return _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
        vector_columns=vector_columns,
    )


def _current_nwb_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical current-schema result suitable for NWB storage."""
    canonical = validate_swap_glm_result(result)
    schema_version = str(canonical["dataset"].attrs.get("schema_version", ""))
    if schema_version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "SwapGLM NWB storage requires a normalized current-schema result."
        )
    return canonical


def _selected_units_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return the canonical selected-unit audit."""
    return _current_nwb_result(result)["selected_units"].copy()


def _model_metadata_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return one row per fitted or derived model."""
    dataset = _current_nwb_result(result)["dataset"]
    if not dataset.data_vars:
        return pd.DataFrame(columns=MODEL_METADATA_COLUMNS)
    rows = []
    for model_index, model in enumerate(MODEL_NAMES):
        rows.append(
            {
                "model": model,
                "selected_model_path": str(
                    dataset["selected_model_path"].values[model_index]
                ),
                "selected_source_model": str(
                    dataset["selected_source_model"].values[model_index]
                ),
                "selected_ridge": float(
                    dataset["selected_ridge"].values[model_index]
                ),
                "selected_score": float(
                    dataset["selected_score"].values[model_index]
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=MODEL_METADATA_COLUMNS)


def _axes_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return all numerical SwapGLM coordinates as named vectors."""
    dataset = _current_nwb_result(result)["dataset"]
    rows = []
    for axis_name in (
        "tp_grid",
        "tp_observed_bin",
        "tp_observed_edge",
        "segment_edge",
    ):
        values = (
            np.asarray(dataset.coords[axis_name].values, dtype=float)
            if axis_name in dataset.coords
            else np.asarray([], dtype=float)
        )
        rows.append({"axis_name": axis_name, "values": values})
    return pd.DataFrame.from_records(rows, columns=AXES_COLUMNS)


def _trajectory_metadata_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return swap geometry and held-out support for each trajectory."""
    dataset = _current_nwb_result(result)["dataset"]
    if not dataset.data_vars:
        return pd.DataFrame(columns=TRAJECTORY_METADATA_COLUMNS)
    rows = []
    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        rows.append(
            {
                "trajectory": trajectory,
                "swap_source_trajectory": str(
                    dataset["swap_source_trajectory"].values[trajectory_index]
                ),
                "swap_segment_index_1based": int(
                    dataset["swap_segment_index_1based"].values[
                        trajectory_index
                    ]
                ),
                "swap_segment_start": float(
                    dataset["swap_segment_start"].values[trajectory_index]
                ),
                "swap_segment_end": float(
                    dataset["swap_segment_end"].values[trajectory_index]
                ),
                "test_light_swapped_segment_n_bins": int(
                    dataset["test_light_swapped_segment_n_bins"].values[
                        trajectory_index
                    ]
                ),
                "test_light_full_n_bins": int(
                    dataset["test_light_full_n_bins"].values[trajectory_index]
                ),
                "test_light_occupancy_s": np.asarray(
                    dataset["test_light_occupancy_s"].values[trajectory_index],
                    dtype=float,
                ),
            }
        )
    return pd.DataFrame.from_records(
        rows, columns=TRAJECTORY_METADATA_COLUMNS
    )


def _unit_identity_record(unit: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit identity carried by every unit-bearing NWB row."""
    return {
        **{name: str(unit[name]) for name in IDENTITY_COLUMNS},
        "selection_index": int(unit["selection_index"]),
    }


def _model_results_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return model scores and progression profiles keyed by path and unit."""
    canonical = _current_nwb_result(result)
    dataset = canonical["dataset"]
    if not dataset.data_vars:
        return pd.DataFrame(columns=MODEL_RESULT_COLUMNS)
    selected_units = canonical["selected_units"].reset_index(drop=True)
    rows = []
    for model_index, model in enumerate(MODEL_NAMES):
        for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
            for unit_index, unit in selected_units.iterrows():
                rows.append(
                    {
                        "model": model,
                        "trajectory": trajectory,
                        **_unit_identity_record(unit),
                        **{
                            name: np.asarray(
                                dataset[name].values[
                                    model_index,
                                    trajectory_index,
                                    :,
                                    unit_index,
                                ],
                                dtype=float,
                            )
                            for name in MODEL_RESULT_PROFILE_COLUMNS
                        },
                        **{
                            name: float(
                                dataset[name].values[
                                    model_index, trajectory_index, unit_index
                                ]
                            )
                            for name in MODEL_RESULT_SCORE_COLUMNS
                        },
                    }
                )
    return pd.DataFrame.from_records(rows, columns=MODEL_RESULT_COLUMNS)


def _observed_response_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return observed held-out counts and rates keyed by path and unit."""
    canonical = _current_nwb_result(result)
    dataset = canonical["dataset"]
    if not dataset.data_vars:
        return pd.DataFrame(columns=OBSERVED_RESPONSE_COLUMNS)
    selected_units = canonical["selected_units"].reset_index(drop=True)
    rows = []
    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        for unit_index, unit in selected_units.iterrows():
            rows.append(
                {
                    "trajectory": trajectory,
                    **_unit_identity_record(unit),
                    "test_light_spike_count": np.asarray(
                        dataset["test_light_spike_count"].values[
                            trajectory_index, :, unit_index
                        ],
                        dtype=float,
                    ),
                    "test_light_observed_rate_hz": np.asarray(
                        dataset["test_light_observed_rate_hz"].values[
                            trajectory_index, :, unit_index
                        ],
                        dtype=float,
                    ),
                }
            )
    return pd.DataFrame.from_records(rows, columns=OBSERVED_RESPONSE_COLUMNS)


def _json_ready(value: Any) -> Any:
    """Return nested provenance using JSON-native scalar types."""
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


def _provenance_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Return one detached, self-describing SwapGLM provenance row."""
    canonical = _current_nwb_result(result)
    record = {
        "metadata_json": json.dumps(
            _json_ready(canonical["metadata"]),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "parameters_json": json.dumps(
            _json_ready(canonical["parameters"]),
            sort_keys=True,
            separators=(",", ":"),
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
    return pd.DataFrame.from_records([record], columns=PROVENANCE_COLUMNS)


def swap_glm_selected_units_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert the complete selected-unit audit to an NWB DynamicTable."""
    return _dynamic_table_from_frame(
        table,
        name=NWB_SELECTED_UNITS_TABLE_NAME,
        description=(
            "Selected-unit identity and held-out score audit for SwapGLM; "
            f"v1ca1 NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=_SELECTED_UNIT_TEXT_COLUMNS,
        integer_columns=_SELECTED_UNIT_INTEGER_COLUMNS,
        boolean_columns=_SELECTED_UNIT_BOOLEAN_COLUMNS,
    )


def swap_glm_selected_units_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load the selected-unit audit from its NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_SELECTED_UNITS_TABLE_NAME,
        columns=SELECTED_UNIT_COLUMNS,
        text_columns=_SELECTED_UNIT_TEXT_COLUMNS,
        integer_columns=_SELECTED_UNIT_INTEGER_COLUMNS,
        boolean_columns=_SELECTED_UNIT_BOOLEAN_COLUMNS,
    )


def swap_glm_model_metadata_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store selected model identity, ridge, and source metadata."""
    return _dynamic_table_from_frame(
        _model_metadata_frame(result),
        name=NWB_MODEL_METADATA_TABLE_NAME,
        description="Selected and derived SwapGLM model metadata.",
        columns=MODEL_METADATA_COLUMNS,
        text_columns=("model", "selected_model_path", "selected_source_model"),
    )


def swap_glm_model_metadata_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load selected model metadata from its NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_MODEL_METADATA_TABLE_NAME,
        columns=MODEL_METADATA_COLUMNS,
        text_columns=("model", "selected_model_path", "selected_source_model"),
    )


def swap_glm_axes_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store progression, observed-bin, and segment axes."""
    return _ragged_dynamic_table_from_frame(
        _axes_frame(result),
        name=NWB_AXES_TABLE_NAME,
        description="Named numerical axes for the SwapGLM model arrays.",
        columns=AXES_COLUMNS,
        vector_columns=("values",),
        text_columns=("axis_name",),
    )


def swap_glm_axes_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load the named SwapGLM axes from their NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_AXES_TABLE_NAME,
        columns=AXES_COLUMNS,
        text_columns=("axis_name",),
        vector_columns=("values",),
    )


def swap_glm_trajectory_metadata_to_dynamic_table(
    result: Mapping[str, Any],
) -> Any:
    """Store per-path swap geometry, bin counts, and occupancy."""
    return _ragged_dynamic_table_from_frame(
        _trajectory_metadata_frame(result),
        name=NWB_TRAJECTORY_METADATA_TABLE_NAME,
        description="Swap geometry and held-out evaluation support by path.",
        columns=TRAJECTORY_METADATA_COLUMNS,
        vector_columns=("test_light_occupancy_s",),
        text_columns=("trajectory", "swap_source_trajectory"),
        integer_columns=(
            "swap_segment_index_1based",
            "test_light_swapped_segment_n_bins",
            "test_light_full_n_bins",
        ),
    )


def swap_glm_trajectory_metadata_from_dynamic_table(
    nwb_table: Any,
) -> pd.DataFrame:
    """Load per-path SwapGLM metadata from its NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_TRAJECTORY_METADATA_TABLE_NAME,
        columns=TRAJECTORY_METADATA_COLUMNS,
        vector_columns=("test_light_occupancy_s",),
        text_columns=("trajectory", "swap_source_trajectory"),
        integer_columns=(
            "swap_segment_index_1based",
            "test_light_swapped_segment_n_bins",
            "test_light_full_n_bins",
        ),
    )


def swap_glm_model_results_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store model predictions and scores keyed by path and selected unit."""
    return _ragged_dynamic_table_from_frame(
        _model_results_frame(result),
        name=NWB_MODEL_RESULTS_TABLE_NAME,
        description=(
            "SwapGLM progression profiles and score metrics keyed by model, "
            "path, and persistent unit identity."
        ),
        columns=MODEL_RESULT_COLUMNS,
        vector_columns=MODEL_RESULT_PROFILE_COLUMNS,
        text_columns=("model", "trajectory", *IDENTITY_COLUMNS),
        integer_columns=("selection_index",),
    )


def swap_glm_model_results_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Load keyed model predictions and scores from their NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_MODEL_RESULTS_TABLE_NAME,
        columns=MODEL_RESULT_COLUMNS,
        vector_columns=MODEL_RESULT_PROFILE_COLUMNS,
        text_columns=("model", "trajectory", *IDENTITY_COLUMNS),
        integer_columns=("selection_index",),
    )


def swap_glm_observed_response_to_dynamic_table(
    result: Mapping[str, Any],
) -> Any:
    """Store held-out observed counts and rates keyed by path and unit."""
    return _ragged_dynamic_table_from_frame(
        _observed_response_frame(result),
        name=NWB_OBSERVED_RESPONSE_TABLE_NAME,
        description=(
            "Held-out observed spike counts and rates across progression bins."
        ),
        columns=OBSERVED_RESPONSE_COLUMNS,
        vector_columns=(
            "test_light_spike_count",
            "test_light_observed_rate_hz",
        ),
        text_columns=("trajectory", *IDENTITY_COLUMNS),
        integer_columns=("selection_index",),
    )


def swap_glm_observed_response_from_dynamic_table(
    nwb_table: Any,
) -> pd.DataFrame:
    """Load held-out observed responses from their NWB object."""
    return _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_OBSERVED_RESPONSE_TABLE_NAME,
        columns=OBSERVED_RESPONSE_COLUMNS,
        vector_columns=(
            "test_light_spike_count",
            "test_light_observed_rate_hz",
        ),
        text_columns=("trajectory", *IDENTITY_COLUMNS),
        integer_columns=("selection_index",),
    )


def swap_glm_provenance_to_dynamic_table(result: Mapping[str, Any]) -> Any:
    """Store detached SwapGLM parameters, sources, attrs, and status."""
    return _dynamic_table_from_frame(
        _provenance_frame(result),
        name=NWB_PROVENANCE_TABLE_NAME,
        description=(
            "Detached SwapGLM identity, parameters, upstream provenance, and "
            f"dataset attributes; v1ca1 NWB schema {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=PROVENANCE_COLUMNS,
        text_columns=PROVENANCE_COLUMNS,
    )


def swap_glm_provenance_from_dynamic_table(nwb_table: Any) -> dict[str, Any]:
    """Load and parse one detached SwapGLM provenance row."""
    table = _frame_from_dynamic_table(
        nwb_table,
        expected_name=NWB_PROVENANCE_TABLE_NAME,
        columns=PROVENANCE_COLUMNS,
        text_columns=PROVENANCE_COLUMNS,
    )
    if len(table) != 1:
        raise ValueError("SwapGLM provenance must contain exactly one row.")
    record = table.iloc[0].to_dict()
    if record["artifact_schema_version"] != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("SwapGLM NWB artifact schema version is unsupported.")
    decoded = {}
    for field in (
        "metadata_json",
        "parameters_json",
        "upstream_provenance_json",
        "dataset_attrs_json",
        "legacy_artifact_provenance_json",
    ):
        try:
            value = json.loads(record[field])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"SwapGLM provenance {field!r} contains malformed JSON."
            ) from exc
        if not isinstance(value, Mapping):
            raise ValueError(
                f"SwapGLM provenance {field!r} must encode a mapping."
            )
        decoded[field] = dict(value)
    return {**record, **decoded}


def swap_glm_result_to_nwb_objects(result: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one current SwapGLM result to seven NWB scratch objects."""
    canonical = _current_nwb_result(result)
    return {
        "selected_units": swap_glm_selected_units_to_dynamic_table(
            canonical["selected_units"]
        ),
        "model_metadata": swap_glm_model_metadata_to_dynamic_table(canonical),
        "axes": swap_glm_axes_to_dynamic_table(canonical),
        "trajectory_metadata": swap_glm_trajectory_metadata_to_dynamic_table(
            canonical
        ),
        "model_results": swap_glm_model_results_to_dynamic_table(canonical),
        "observed_response": swap_glm_observed_response_to_dynamic_table(
            canonical
        ),
        "provenance": swap_glm_provenance_to_dynamic_table(canonical),
    }


def _require_nwb_row_order(
    observed: Sequence[Any],
    expected: Sequence[Any],
    *,
    name: str,
) -> None:
    """Require deterministic complete keyed rows in canonical order."""
    if list(observed) != list(expected):
        raise ValueError(f"SwapGLM NWB {name} rows are incomplete or reordered.")


def _require_unit_identity_rows(
    table: pd.DataFrame,
    selected_units: pd.DataFrame,
) -> None:
    """Require every long-form row to carry the selected-unit identity."""
    identities = selected_units.set_index("selection_index")
    for row in table.to_dict("records"):
        index = int(row["selection_index"])
        if index not in identities.index:
            raise ValueError("SwapGLM NWB row has an unknown selection index.")
        expected = identities.loc[index]
        if any(str(row[name]) != str(expected[name]) for name in IDENTITY_COLUMNS):
            raise ValueError("SwapGLM NWB row has a mismatched unit identity.")


def swap_glm_result_from_nwb_objects(
    *,
    selected_units: Any,
    model_metadata: Any,
    axes: Any,
    trajectory_metadata: Any,
    model_results: Any,
    observed_response: Any,
    provenance: Any,
) -> dict[str, Any]:
    """Reconstruct and validate one SwapGLM result from seven NWB objects."""
    import xarray as xr

    selected = swap_glm_selected_units_from_dynamic_table(selected_units)
    models = swap_glm_model_metadata_from_dynamic_table(model_metadata)
    axes_table = swap_glm_axes_from_dynamic_table(axes)
    trajectories = swap_glm_trajectory_metadata_from_dynamic_table(
        trajectory_metadata
    )
    results = swap_glm_model_results_from_dynamic_table(model_results)
    observed = swap_glm_observed_response_from_dynamic_table(observed_response)
    source = swap_glm_provenance_from_dynamic_table(provenance)
    _require_nwb_row_order(
        axes_table["axis_name"].tolist(),
        ("tp_grid", "tp_observed_bin", "tp_observed_edge", "segment_edge"),
        name="axis",
    )
    axis_values = {
        str(row["axis_name"]): np.asarray(row["values"], dtype=float)
        for row in axes_table.to_dict("records")
    }
    metadata = source["metadata_json"]
    parameters = source["parameters_json"]
    upstream = source["upstream_provenance_json"]
    attrs = source["dataset_attrs_json"]
    status = str(source["analysis_status"])
    origin = str(source["artifact_origin"])
    legacy = source["legacy_artifact_provenance_json"] or None
    unit_ids = selected["group_unit_id"].astype(str).to_numpy()
    if str(attrs.get("fit_stage", "")) == "terminal":
        if not models.empty or not trajectories.empty or not results.empty or not observed.empty:
            raise ValueError("Terminal SwapGLM NWB objects contain model results.")
        dataset = xr.Dataset(
            coords={
                "model": np.asarray(MODEL_NAMES, dtype=str),
                "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
                "unit": unit_ids,
                "segment_edge": axis_values["segment_edge"],
            },
            attrs=attrs,
        )
    else:
        _require_nwb_row_order(models["model"].tolist(), MODEL_NAMES, name="model")
        _require_nwb_row_order(
            trajectories["trajectory"].tolist(),
            TRAJECTORY_TYPES,
            name="trajectory",
        )
        expected_model_keys = [
            (model, trajectory, index)
            for model in MODEL_NAMES
            for trajectory in TRAJECTORY_TYPES
            for index in selected["selection_index"].astype(int)
        ]
        observed_model_keys = list(
            zip(
                results["model"],
                results["trajectory"],
                results["selection_index"].astype(int),
                strict=True,
            )
        )
        _require_nwb_row_order(
            observed_model_keys, expected_model_keys, name="model-result"
        )
        expected_observed_keys = [
            (trajectory, index)
            for trajectory in TRAJECTORY_TYPES
            for index in selected["selection_index"].astype(int)
        ]
        observed_keys = list(
            zip(
                observed["trajectory"],
                observed["selection_index"].astype(int),
                strict=True,
            )
        )
        _require_nwb_row_order(
            observed_keys, expected_observed_keys, name="observed-response"
        )
        _require_unit_identity_rows(results, selected)
        _require_unit_identity_rows(observed, selected)
        n_models = len(MODEL_NAMES)
        n_trajectories = len(TRAJECTORY_TYPES)
        n_units = len(selected)
        n_grid = len(axis_values["tp_grid"])
        n_observed = len(axis_values["tp_observed_bin"])
        data_vars: dict[str, Any] = {
            "selected_model_path": (
                "model",
                models["selected_model_path"].astype(str).to_numpy(),
            ),
            "selected_source_model": (
                "model",
                models["selected_source_model"].astype(str).to_numpy(),
            ),
            "selected_ridge": (
                "model",
                models["selected_ridge"].to_numpy(dtype=float),
            ),
            "selected_score": (
                "model",
                models["selected_score"].to_numpy(dtype=float),
            ),
            "swap_source_trajectory": (
                "trajectory",
                trajectories["swap_source_trajectory"].astype(str).to_numpy(),
            ),
            "swap_segment_index_1based": (
                "trajectory",
                trajectories["swap_segment_index_1based"].to_numpy(
                    dtype=np.int64
                ),
            ),
            "swap_segment_start": (
                "trajectory",
                trajectories["swap_segment_start"].to_numpy(dtype=float),
            ),
            "swap_segment_end": (
                "trajectory",
                trajectories["swap_segment_end"].to_numpy(dtype=float),
            ),
            "test_light_swapped_segment_n_bins": (
                "trajectory",
                trajectories["test_light_swapped_segment_n_bins"].to_numpy(
                    dtype=np.int64
                ),
            ),
            "test_light_full_n_bins": (
                "trajectory",
                trajectories["test_light_full_n_bins"].to_numpy(dtype=np.int64),
            ),
            "test_light_occupancy_s": (
                ("trajectory", "tp_observed_bin"),
                np.stack(trajectories["test_light_occupancy_s"].tolist()),
            ),
        }
        for name in MODEL_RESULT_PROFILE_COLUMNS:
            values = np.full(
                (n_models, n_trajectories, n_grid, n_units),
                np.nan,
                dtype=float,
            )
            for row_index, value in enumerate(results[name]):
                vector = np.asarray(value, dtype=float)
                if vector.shape != (n_grid,):
                    raise ValueError(
                        f"SwapGLM NWB profile {name!r} has the wrong length."
                    )
                model_index = row_index // (n_trajectories * n_units)
                within_model = row_index % (n_trajectories * n_units)
                trajectory_index = within_model // n_units
                unit_index = within_model % n_units
                values[model_index, trajectory_index, :, unit_index] = vector
            data_vars[name] = (
                ("model", "trajectory", "tp_grid", "unit"),
                values,
            )
        for name in MODEL_RESULT_SCORE_COLUMNS:
            values = results[name].to_numpy(dtype=float).reshape(
                n_models, n_trajectories, n_units
            )
            data_vars[name] = (("model", "trajectory", "unit"), values)
        for name in (
            "test_light_spike_count",
            "test_light_observed_rate_hz",
        ):
            values = np.full(
                (n_trajectories, n_observed, n_units),
                np.nan,
                dtype=float,
            )
            for row_index, value in enumerate(observed[name]):
                vector = np.asarray(value, dtype=float)
                if vector.shape != (n_observed,):
                    raise ValueError(
                        f"SwapGLM NWB observed vector {name!r} has the wrong length."
                    )
                trajectory_index = row_index // n_units
                unit_index = row_index % n_units
                values[trajectory_index, :, unit_index] = vector
            data_vars[name] = (
                ("trajectory", "tp_observed_bin", "unit"),
                values,
            )
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "model": np.asarray(MODEL_NAMES, dtype=str),
                "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
                "unit": unit_ids,
                **axis_values,
            },
            attrs=attrs,
        )
    return validate_swap_glm_result(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream,
            "selected_units": selected,
            "dataset": dataset,
            "analysis_status": status,
            "artifact_origin": origin,
            "legacy_artifact_provenance": legacy,
        }
    )


def _semantic_frame_sha256(
    table: pd.DataFrame,
    *,
    columns: Sequence[str],
    text_columns: Sequence[str] = (),
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
    vector_columns: Sequence[str] = (),
) -> str:
    """Hash a canonical scalar/ragged frame independent of NWB identifiers."""
    canonical = _normalize_nwb_frame(
        table,
        columns=columns,
        text_columns=text_columns,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
        vector_columns=vector_columns,
    )
    digest = hashlib.sha256()
    digest.update(json.dumps(list(columns), separators=(",", ":")).encode())
    digest.update(np.asarray([len(canonical)], dtype="<i8").tobytes())
    for row in canonical.to_dict("records"):
        for column in columns:
            value = row[column]
            if column in text_columns:
                encoded = str(value).encode("utf-8")
                digest.update(np.asarray([len(encoded)], dtype="<i8").tobytes())
                digest.update(encoded)
            elif column in integer_columns:
                digest.update(np.asarray([value], dtype="<i8").tobytes())
            elif column in boolean_columns:
                digest.update(np.asarray([value], dtype=np.uint8).tobytes())
            elif column in vector_columns:
                array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
                if np.isnan(array).any():
                    array = array.copy()
                    array[np.isnan(array)] = np.nan
                digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
                digest.update(array.tobytes())
            else:
                array = np.asarray([value], dtype="<f8")
                if np.isnan(array[0]):
                    array[0] = np.nan
                digest.update(array.tobytes())
    return digest.hexdigest()


def swap_glm_nwb_hashes(result: Mapping[str, Any]) -> dict[str, str]:
    """Return deterministic semantic hashes for all seven NWB objects."""
    canonical = _current_nwb_result(result)
    frames = {
        "selected_units_table_sha256": (
            _selected_units_frame(canonical),
            SELECTED_UNIT_COLUMNS,
            _SELECTED_UNIT_TEXT_COLUMNS,
            _SELECTED_UNIT_INTEGER_COLUMNS,
            _SELECTED_UNIT_BOOLEAN_COLUMNS,
            (),
        ),
        "model_metadata_sha256": (
            _model_metadata_frame(canonical),
            MODEL_METADATA_COLUMNS,
            ("model", "selected_model_path", "selected_source_model"),
            (),
            (),
            (),
        ),
        "axes_sha256": (
            _axes_frame(canonical),
            AXES_COLUMNS,
            ("axis_name",),
            (),
            (),
            ("values",),
        ),
        "trajectory_metadata_sha256": (
            _trajectory_metadata_frame(canonical),
            TRAJECTORY_METADATA_COLUMNS,
            ("trajectory", "swap_source_trajectory"),
            (
                "swap_segment_index_1based",
                "test_light_swapped_segment_n_bins",
                "test_light_full_n_bins",
            ),
            (),
            ("test_light_occupancy_s",),
        ),
        "model_results_sha256": (
            _model_results_frame(canonical),
            MODEL_RESULT_COLUMNS,
            ("model", "trajectory", *IDENTITY_COLUMNS),
            ("selection_index",),
            (),
            MODEL_RESULT_PROFILE_COLUMNS,
        ),
        "observed_response_sha256": (
            _observed_response_frame(canonical),
            OBSERVED_RESPONSE_COLUMNS,
            ("trajectory", *IDENTITY_COLUMNS),
            ("selection_index",),
            (),
            ("test_light_spike_count", "test_light_observed_rate_hz"),
        ),
        "provenance_sha256": (
            _provenance_frame(canonical),
            PROVENANCE_COLUMNS,
            PROVENANCE_COLUMNS,
            (),
            (),
            (),
        ),
    }
    return {
        name: _semantic_frame_sha256(
            frame,
            columns=columns,
            text_columns=text_columns,
            integer_columns=integer_columns,
            boolean_columns=boolean_columns,
            vector_columns=vector_columns,
        )
        for name, (
            frame,
            columns,
            text_columns,
            integer_columns,
            boolean_columns,
            vector_columns,
        ) in frames.items()
    }


def _verify_legacy_selected_sources(
    dataset: Any,
    upstream_provenance: Mapping[str, Any],
    *,
    model_names: Sequence[str] | None = None,
) -> dict[str, str]:
    """Verify that legacy selected-file paths have exact upstream checksums."""
    selected_paths = _legacy_selected_source_paths(dataset)
    expected_hashes = _dark_light_selected_hashes(upstream_provenance)
    legacy_hashes = dict(
        upstream_provenance.get(
            "dark_light_legacy_selected_sha256_by_model",
            {},
        )
    )
    verified = {}
    if model_names is None:
        model_names = tuple(np.asarray(dataset.coords["model"].values, dtype=str))
    for model_name in model_names:
        if model_name not in selected_paths:
            raise ValueError(f"Legacy source mapping is missing {model_name!r}.")
        source_path = Path(selected_paths[model_name])
        if not source_path.is_file():
            raise FileNotFoundError(f"Legacy selected source not found: {source_path}")
        source_model = DERIVED_MODEL_SOURCES.get(model_name, model_name)
        digest = _file_sha256(source_path)
        allowed_hashes = {str(expected_hashes[source_model])}
        if source_model in legacy_hashes:
            allowed_hashes.add(str(legacy_hashes[source_model]))
        if digest not in allowed_hashes:
            raise ValueError(
                f"Legacy {model_name!r} source does not match upstream DarkLight."
            )
        verified[model_name] = digest
    return verified


def _legacy_selected_source_paths(dataset: Any) -> dict[str, str]:
    """Return the legacy swap file's selected DarkLight source mapping."""
    try:
        sources = json.loads(str(dataset.attrs["sources_json"]))
        selected_paths = dict(sources["dark_light_glm_selected"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Legacy swap artifact lacks selected DarkLight source provenance."
        ) from exc
    return {str(name): str(path) for name, path in selected_paths.items()}


def _normalize_legacy_swap_dataset(dataset: Any) -> tuple[Any, dict[str, Any]]:
    """Validate a supported legacy schema and return its strict source view."""
    schema_version = str(dataset.attrs.get("schema_version", ""))
    model_names = tuple(
        np.asarray(dataset.coords.get("model", ()), dtype=str).tolist()
    )
    if schema_version == "6":
        if model_names != LEGACY_SCHEMA6_MODEL_NAMES:
            raise ValueError("Legacy schema-6 swap artifact has stale model order.")
        _validate_nonterminal_swap_dataset(
            dataset,
            model_names=LEGACY_SCHEMA6_MODEL_NAMES,
            derived_sources=LEGACY_SCHEMA6_DERIVED_MODEL_SOURCES,
            expected_schema_version="6",
            extra_data_variables=LEGACY_SCHEMA6_DIAGNOSTIC_VARIABLES,
        )
        prediction_clip = float(
            dataset.attrs.get("prediction_count_clip_eps", np.nan)
        )
        if not np.isclose(prediction_clip, 1e-12, rtol=1e-12, atol=1e-15):
            raise ValueError(
                "Legacy schema-6 swap artifact has stale prediction clipping."
            )
        if _json_mapping_attr(
            dataset,
            "visual_empirical_model_definitions_json",
        ) != dict(LEGACY_SCHEMA6_VISUAL_EMPIRICAL_MODEL_DEFINITIONS):
            raise ValueError(
                "Legacy schema-6 visual empirical-model definitions are stale."
            )
        normalized = dataset.sel(model=list(MODEL_NAMES)).drop_vars(
            list(LEGACY_SCHEMA6_DIAGNOSTIC_VARIABLES)
        )
        normalized.attrs = dict(dataset.attrs)
        normalized.attrs["schema_version"] = RESULT_SCHEMA_VERSION
        normalized.attrs["prediction_count_clip_eps"] = (
            PREDICTION_COUNT_CLIP_EPS
        )
        normalized.attrs["visual_empirical_model_definitions_json"] = (
            json.dumps(
                dict(VISUAL_EMPIRICAL_MODEL_DEFINITIONS),
                sort_keys=True,
            )
        )
        normalized.attrs["derived_model_sources_json"] = json.dumps(
            dict(DERIVED_MODEL_SOURCES),
            sort_keys=True,
        )
        fit_parameters = _json_mapping_attr(dataset, "fit_parameters_json")
        fit_parameters["models"] = list(MODEL_NAMES)
        if "requested_models" in fit_parameters:
            fit_parameters["requested_models"] = list(MODEL_NAMES)
        fit_parameters["derived_model_sources"] = dict(DERIVED_MODEL_SOURCES)
        normalized.attrs["fit_parameters_json"] = json.dumps(
            fit_parameters,
            sort_keys=True,
        )
        _validate_nonterminal_swap_dataset(normalized)
        return normalized, {
            "source_schema_version": "6",
            "source_model_names": list(LEGACY_SCHEMA6_MODEL_NAMES),
            "compared_model_names": list(MODEL_NAMES),
            "dropped_schema6_diagnostics": list(
                LEGACY_SCHEMA6_DIAGNOSTIC_VARIABLES
            ),
            "synthesized_selected_source_model": False,
            "dark_score_source": "legacy_source_and_exact_nwb_rescore",
            "requires_legacy_preprocessing_provenance": True,
        }
    if schema_version == RESULT_SCHEMA_VERSION:
        if model_names != MODEL_NAMES:
            raise ValueError("Current swap artifact has stale model order.")
        _validate_nonterminal_swap_dataset(dataset)
        return dataset, {
            "source_schema_version": RESULT_SCHEMA_VERSION,
            "source_model_names": list(MODEL_NAMES),
            "compared_model_names": list(MODEL_NAMES),
            "dropped_schema6_diagnostics": [],
            "synthesized_selected_source_model": False,
            "dark_score_source": "legacy_source_and_exact_nwb_rescore",
            "requires_legacy_preprocessing_provenance": True,
        }
    if schema_version != "4":
        raise ValueError("Existing swap artifact must use schema version 4, 5, or 6.")
    if model_names == LEGACY_SCHEMA4_MODEL_NAMES:
        _validate_nonterminal_swap_dataset(
            dataset,
            model_names=LEGACY_SCHEMA4_MODEL_NAMES,
            derived_sources=LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES,
            expected_schema_version="4",
        )
        return dataset, {
            "source_schema_version": "4",
            "source_model_names": list(LEGACY_SCHEMA4_MODEL_NAMES),
            "compared_model_names": list(LEGACY_SCHEMA4_MODEL_NAMES),
            "dropped_schema6_diagnostics": [],
            "synthesized_selected_source_model": False,
            "dark_score_source": "legacy_source_and_exact_nwb_rescore",
            "requires_legacy_preprocessing_provenance": True,
        }
    if model_names != SOURCE_MODEL_NAMES:
        raise ValueError(
            "Legacy schema-4 swap artifact must contain either the four "
            "source models or the complete canonical five-model order."
        )

    normalized = dataset.assign(
        selected_source_model=(
            ("model",),
            np.asarray(SOURCE_MODEL_NAMES, dtype=str),
        )
    )
    normalized.attrs = dict(dataset.attrs)
    normalized.attrs["derived_model_sources_json"] = json.dumps({}, sort_keys=True)
    fit_parameters = _json_mapping_attr(dataset, "fit_parameters_json")
    fit_parameters["derived_model_sources"] = {}
    normalized.attrs["fit_parameters_json"] = json.dumps(
        fit_parameters,
        sort_keys=True,
    )
    _validate_nonterminal_swap_dataset(
        normalized,
        model_names=SOURCE_MODEL_NAMES,
        derived_sources={},
        expected_schema_version="4",
    )
    return normalized, {
        "source_schema_version": "4",
        "source_model_names": list(SOURCE_MODEL_NAMES),
        "compared_model_names": list(SOURCE_MODEL_NAMES),
        "dropped_schema6_diagnostics": [],
        "synthesized_selected_source_model": True,
        "dark_score_source": (
            "exact_nwb_rescore_from_verified_task_segment_bump_not_legacy_clone"
        ),
        "requires_legacy_preprocessing_provenance": True,
    }


def _compare_scientific_swap_datasets(
    source: Any,
    recomputed: Any,
    *,
    compared_model_names: Sequence[str],
) -> None:
    """Require every available legacy scientific value to match an exact re-score."""
    compared_model_names = tuple(compared_model_names)
    recomputed_view = recomputed.sel(model=list(compared_model_names))
    for name in CANONICAL_COORDINATE_DIMS:
        source_values = np.asarray(source.coords[name].values)
        recomputed_values = np.asarray(recomputed_view.coords[name].values)
        if source_values.dtype.kind in "OUS" or recomputed_values.dtype.kind in "OUS":
            matches = np.array_equal(
                source_values.astype(str),
                recomputed_values.astype(str),
            )
        else:
            matches = _allclose_equal_nan(
                source_values,
                recomputed_values,
                rtol=1e-9,
                atol=1e-9,
            )
        if not matches:
            raise ValueError(
                f"Legacy swap coordinate {name!r} differs from exact NWB re-score."
            )

    compared_variables = set(CANONICAL_DATA_VARIABLE_DIMS).intersection(
        source.data_vars
    )
    compared_variables.remove("selected_model_path")
    for name in sorted(compared_variables):
        source_values = np.asarray(source[name].values)
        recomputed_values = np.asarray(recomputed_view[name].values)
        if source_values.dtype.kind in "OUS" or recomputed_values.dtype.kind in "OUS":
            matches = np.array_equal(
                source_values.astype(str),
                recomputed_values.astype(str),
            )
        else:
            matches = _allclose_equal_nan(
                source_values,
                recomputed_values,
                rtol=1e-9,
                atol=1e-9,
            )
        if not matches:
            raise ValueError(
                f"Legacy swap variable {name!r} differs from exact NWB re-score."
            )

    for name in (
        *FIXED_SCIENTIFIC_ATTRS,
        "swap_light_offset",
        "bin_size_s",
        "spatial_bin_size_cm",
        "n_splines",
        "spline_order",
        "has_speed",
        "speed_feature_mode",
        "n_speed_features",
        "speed_spline_order",
    ):
        if name not in source.attrs:
            continue
        source_value = source.attrs[name]
        recomputed_value = recomputed_view.attrs.get(name)
        if isinstance(source_value, (float, int, np.number)) and not isinstance(
            source_value,
            (bool, np.bool_),
        ):
            matches = _allclose_equal_nan(
                [source_value],
                [recomputed_value],
                rtol=1e-12,
                atol=1e-12,
            )
        else:
            matches = str(source_value) == str(recomputed_value)
        if not matches:
            raise ValueError(
                f"Legacy swap scientific attribute {name!r} differs from exact "
                "NWB re-score."
            )


def _remap_legacy_unit_coordinate(
    dataset: Any,
    upstream: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Map legacy selected-file unit coordinates to canonical group IDs."""
    selected_paths = _legacy_selected_source_paths(dataset)
    legacy_unit_ids: np.ndarray | None = None
    for model_name in SOURCE_MODEL_NAMES:
        if model_name not in selected_paths:
            raise ValueError(
                f"Legacy source mapping is missing {model_name!r}."
            )
        source_dataset = _load_dataset(Path(selected_paths[model_name]))
        if "unit" not in source_dataset.coords:
            raise ValueError(
                f"Legacy DarkLight source {model_name!r} lacks unit coordinates."
            )
        current = np.asarray(source_dataset.coords["unit"].values).astype(str)
        if legacy_unit_ids is None:
            legacy_unit_ids = current
        elif not np.array_equal(current, legacy_unit_ids):
            raise ValueError(
                "Legacy DarkLight selected sources disagree on unit order."
            )
    if legacy_unit_ids is None:
        raise ValueError("Legacy DarkLight selected sources contain no models.")
    if len(set(legacy_unit_ids.tolist())) != len(legacy_unit_ids):
        raise ValueError("Legacy DarkLight unit coordinates must be unique.")

    canonical_unit_ids = _selected_unit_ids(upstream)
    canonical_strings = canonical_unit_ids.astype(str)
    upstream_units = upstream["selected_units"]
    if len(legacy_unit_ids) != len(upstream_units) or not np.array_equal(
        canonical_strings,
        upstream_units["group_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError(
            "Legacy DarkLight unit order cannot be aligned to canonical "
            "persistent identities."
        )
    source_result_units = np.asarray(dataset.coords["unit"].values).astype(str)
    if not np.array_equal(source_result_units, legacy_unit_ids):
        raise ValueError(
            "Legacy swap unit order differs from its verified DarkLight sources."
        )
    remapped = dataset.assign_coords(unit=canonical_unit_ids)
    audit = {
        "legacy_unit_coordinate_sha256": _provenance_sha256(
            legacy_unit_ids.tolist()
        ),
        "canonical_unit_coordinate_sha256": _provenance_sha256(
            canonical_strings.tolist()
        ),
        "unit_coordinate_remapped": not np.array_equal(
            legacy_unit_ids,
            canonical_strings,
        ),
        "unit_identity_alignment": (
            "verified_dark_light_source_order_to_canonical_persistent_order"
        ),
    }
    return remapped, audit


def register_existing_swap_glm_artifact(
    *,
    source_result_path: Path,
    destination_path: Path | None,
    swap_glm_id: Any,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    dark_light_glm_artifact_path: Path | None = None,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_interval: Any,
    movement_analysis_status: str,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    position: Any,
    position_offset_samples: int,
    speed_threshold_cm_s: float,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    output_rule_sha256: str | None = None,
    swap_light_offset: bool = DEFAULT_SWAP_LIGHT_OFFSET,
    observed_spatial_bin_size_cm: float = DEFAULT_OBSERVED_SPATIAL_BIN_SIZE_CM,
    source_v1ca1_git_commit: str | None = None,
    overwrite: bool = False,
    dark_light_glm_result: Mapping[str, Any] | None = None,
    dark_light_glm_upstream_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-score exact NWB inputs and register only a matching legacy result."""
    source_path = Path(source_result_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Existing swap result not found: {source_path}")
    metadata = _metadata(
        swap_glm_id=swap_glm_id,
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
        swap_light_offset=swap_light_offset,
        observed_spatial_bin_size_cm=observed_spatial_bin_size_cm,
    )
    if (dark_light_glm_artifact_path is None) == (dark_light_glm_result is None):
        raise ValueError(
            "Provide exactly one DarkLightGLM artifact path or in-memory result."
        )
    if dark_light_glm_result is None:
        upstream = _load_dark_light_input(Path(dark_light_glm_artifact_path))
    else:
        if dark_light_glm_upstream_provenance is None:
            raise ValueError(
                "In-memory DarkLightGLM input requires frozen NWB provenance."
            )
        upstream = _load_dark_light_nwb_input(
            dark_light_glm_result,
            dark_light_glm_upstream_provenance,
        )
    _validate_upstream_context(upstream, metadata)
    upstream_provenance = dict(upstream["upstream_provenance"])
    source_dataset = _load_dataset(source_path)
    normalized_source, normalization_audit = _normalize_legacy_swap_dataset(
        source_dataset
    )
    if not isinstance(position_offset_samples, (int, np.integer)) or isinstance(
        position_offset_samples,
        (bool, np.bool_),
    ):
        raise TypeError("position_offset_samples must be an integer.")
    normalized_position_offset = int(position_offset_samples)
    if normalized_position_offset < 0:
        raise ValueError("position_offset_samples must be a non-negative integer.")
    normalized_speed_threshold = float(speed_threshold_cm_s)
    if (
        not np.isfinite(normalized_speed_threshold)
        or normalized_speed_threshold < 0.0
    ):
        raise ValueError("speed_threshold_cm_s must be finite and non-negative.")
    preprocessing_audit = {
        "selected_position_offset_samples": normalized_position_offset,
        "selected_speed_threshold_cm_s": normalized_speed_threshold,
        "legacy_fields_required": bool(
            normalization_audit["requires_legacy_preprocessing_provenance"]
        ),
    }
    if normalization_audit["requires_legacy_preprocessing_provenance"]:
        legacy_fit_parameters = _json_mapping_attr(
            normalized_source,
            "fit_parameters_json",
        )
        try:
            raw_legacy_position_offset = legacy_fit_parameters[
                "position_offset"
            ]
            raw_legacy_speed_threshold = legacy_fit_parameters[
                "speed_threshold_cm_s"
            ]
        except KeyError as exc:
            raise ValueError(
                "Historical swap artifact lacks position-offset or speed-threshold "
                "provenance."
            ) from exc
        if isinstance(
            raw_legacy_position_offset,
            (bool, np.bool_),
        ) or not isinstance(
            raw_legacy_position_offset,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(
                "Historical swap position_offset must be a finite integer."
            )
        legacy_position_value = float(raw_legacy_position_offset)
        if (
            not np.isfinite(legacy_position_value)
            or not legacy_position_value.is_integer()
        ):
            raise ValueError(
                "Historical swap position_offset must be a finite integer."
            )
        if isinstance(
            raw_legacy_speed_threshold,
            (bool, np.bool_),
        ) or not isinstance(
            raw_legacy_speed_threshold,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(
                "Historical swap speed_threshold_cm_s must be finite and "
                "non-negative."
            )
        legacy_position_offset = int(legacy_position_value)
        legacy_speed_threshold = float(raw_legacy_speed_threshold)
        if not np.isfinite(legacy_speed_threshold) or legacy_speed_threshold < 0.0:
            raise ValueError(
                "Historical swap speed_threshold_cm_s must be finite and "
                "non-negative."
            )
        if legacy_position_offset != normalized_position_offset:
            raise ValueError(
                "Historical swap position_offset differs from selected Position."
            )
        if not np.isclose(
            legacy_speed_threshold,
            normalized_speed_threshold,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                "Historical swap speed_threshold_cm_s differs from selected "
                "MovementParameters."
            )
        preprocessing_audit.update(
            {
                "legacy_position_offset_samples": legacy_position_offset,
                "legacy_speed_threshold_cm_s": legacy_speed_threshold,
            }
        )
    expected_attrs = {
        "animal_name": metadata["animal_name"],
        "date": metadata["date"],
        "region": metadata["region"],
        "dark_train_epoch": metadata["dark_epoch"],
        "light_train_epoch": metadata["light_train_epoch"],
        "light_test_epoch": metadata["light_test_epoch"],
        "fit_source": "dark_light_glm_selected",
    }
    for name, expected in expected_attrs.items():
        if str(normalized_source.attrs.get(name, "")) != str(expected):
            raise ValueError(f"Existing swap artifact has mismatched {name!r}.")
    if _boolean_attr(
        normalized_source,
        "swap_light_offset",
    ) != parameters["swap_light_offset"]:
        raise ValueError("Existing swap artifact has a different swap-light-offset rule.")
    verified_sources = _verify_legacy_selected_sources(
        normalized_source,
        upstream_provenance,
        model_names=normalization_audit["compared_model_names"],
    )
    normalized_source, unit_coordinate_audit = _remap_legacy_unit_coordinate(
        normalized_source,
        upstream,
    )
    selected_ids = _selected_unit_ids(upstream).astype(str)
    if not np.array_equal(
        np.asarray(normalized_source.coords["unit"].values).astype(str),
        selected_ids,
    ):
        raise ValueError("Existing swap units do not match exact DarkLight units.")
    recomputed = compute_swap_glm(
        swap_glm_id=metadata["swap_glm_id"],
        animal_name=metadata["animal_name"],
        date=metadata["date"],
        region=metadata["region"],
        dark_epoch=metadata["dark_epoch"],
        light_train_epoch=metadata["light_train_epoch"],
        light_test_epoch=metadata["light_test_epoch"],
        dark_light_glm_artifact_path=dark_light_glm_artifact_path,
        dark_light_glm_result=(
            upstream if dark_light_glm_artifact_path is None else None
        ),
        dark_light_glm_upstream_provenance=(
            upstream_provenance
            if dark_light_glm_artifact_path is None
            else None
        ),
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        movement_interval=movement_interval,
        movement_analysis_status=movement_analysis_status,
        trajectory_intervals=trajectory_intervals,
        graph_inputs_by_trajectory=graph_inputs_by_trajectory,
        position=position,
        parameter_name=parameters["parameter_name"],
        parameter_sha256=parameters["parameter_sha256"],
        output_rule_sha256=parameters["output_rule_sha256"],
        swap_light_offset=parameters["swap_light_offset"],
        observed_spatial_bin_size_cm=parameters[
            "observed_spatial_bin_size_cm"
        ],
    )
    if recomputed["analysis_status"] in {
        "upstream_terminal",
        "no_units",
        "no_valid_position",
        "no_movement",
        "no_trajectory_samples",
    }:
        raise ValueError(
            "Legacy swap registration requires a nonterminal exact NWB re-score."
        )
    _compare_scientific_swap_datasets(
        normalized_source,
        recomputed["dataset"],
        compared_model_names=normalization_audit["compared_model_names"],
    )
    provenance = {
        "source_result_path": str(source_path.resolve()),
        "source_result_sha256": _file_sha256(source_path),
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "verified_selected_source_sha256_by_model": verified_sources,
        **unit_coordinate_audit,
        "source_normalization": normalization_audit,
        "preprocessing_provenance": preprocessing_audit,
        "validation": (
            "verified legacy schema and source hashes, then exact held-out NWB "
            "re-score comparison of every available scientific coordinate and variable"
        ),
    }
    result = validate_swap_glm_result(
        {
            **recomputed,
            "artifact_origin": "registered_existing",
            "legacy_artifact_provenance": provenance,
        }
    )
    if destination_path is None:
        return result
    artifact_paths = write_swap_glm_artifact(
        result,
        destination_path,
        overwrite=overwrite,
    )
    return {**result, "artifact_paths": artifact_paths}


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "DEFAULT_ARTIFACT_ROOT",
    "MODEL_NAMES",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "OUTPUT_RULE",
    "OUTPUT_RULE_SHA256",
    "compute_swap_glm",
    "get_swap_glm_artifact_paths",
    "load_swap_glm_artifact",
    "register_existing_swap_glm_artifact",
    "swap_glm_axes_from_dynamic_table",
    "swap_glm_axes_to_dynamic_table",
    "swap_glm_model_metadata_from_dynamic_table",
    "swap_glm_model_metadata_to_dynamic_table",
    "swap_glm_model_results_from_dynamic_table",
    "swap_glm_model_results_to_dynamic_table",
    "swap_glm_nwb_hashes",
    "swap_glm_observed_response_from_dynamic_table",
    "swap_glm_observed_response_to_dynamic_table",
    "swap_glm_provenance_from_dynamic_table",
    "swap_glm_provenance_to_dynamic_table",
    "swap_glm_result_from_nwb_objects",
    "swap_glm_result_to_nwb_objects",
    "swap_glm_selected_units_from_dynamic_table",
    "swap_glm_selected_units_to_dynamic_table",
    "swap_glm_trajectory_metadata_from_dynamic_table",
    "swap_glm_trajectory_metadata_to_dynamic_table",
    "validate_swap_glm_parameters",
    "validate_swap_glm_result",
    "write_swap_glm_artifact",
]
