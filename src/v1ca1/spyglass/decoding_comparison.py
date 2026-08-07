"""Database-free cross-path path-progression Bayesian decoding artifacts."""

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

from v1ca1.helper.session import TRAJECTORY_TYPES, TURN_TRAJECTORY_PAIRS


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "path_progression_decoding_comparison"
MANIFEST_FILENAME = "manifest.parquet"
UNIT_FILENAME = "selected_units.parquet"
METRICS_FILENAME = "decoding_summary.parquet"
ELIGIBILITY_FILENAME = "unit_eligibility.parquet"
BINNED_FILENAME = "cross_path_error_by_position.parquet"

DEFAULT_DECODING_BIN_SIZE_S = 0.02
DEFAULT_SLIDING_WINDOW_SIZE_BINS = 4
DEFAULT_SPATIAL_BIN_SIZE_CM = 4.0
DEFAULT_ERROR_MODE = "signed"
DEFAULT_ERROR_SUMMARY = "median_iqr"
DEFAULT_MIN_BIN_COUNT = 5

SAME_INBOUND_OUTBOUND_CROSS_ARM_FAMILY = (
    "same_inbound_outbound_cross_arm"
)
SAME_INBOUND_OUTBOUND_CROSS_ARM_PAIRS = (
    ("center_to_left", "center_to_right"),
    ("center_to_right", "center_to_left"),
    ("left_to_center", "right_to_center"),
    ("right_to_center", "left_to_center"),
)
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
UNIT_TABLE_COLUMNS = (*IDENTITY_COLUMNS, "selection_index")
METRIC_NAMES = (
    "mae",
    "rmse",
    "mean_signed_error",
    "median_abs_error",
    "n_samples",
)
METRIC_COLUMNS = (
    "path_progression_decoding_comparison_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "cohort_epoch",
    "transfer_family",
    "source_trajectory",
    "target_trajectory",
    "flip_tuning_curve",
    "coordinate_unit",
    "n_units",
    *METRIC_NAMES,
    "qc_status",
    "qc_message",
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
BINNED_COLUMNS = (
    "path_progression_decoding_comparison_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "cohort_epoch",
    "transfer_family",
    "source_trajectory",
    "target_trajectory",
    "flip_tuning_curve",
    "coordinate_unit",
    *BINNED_VALUE_COLUMNS,
)
ELIGIBILITY_BASE_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
    "target_epoch",
    "cohort_epoch",
    "minimum_movement_firing_rate_hz",
    "minimum_stability_correlation",
    "target_movement_firing_rate_hz",
    "cohort_movement_firing_rate_hz",
)
ELIGIBILITY_FLAG_COLUMNS = (
    "target_passes_movement_firing_rate",
    "target_passes_stability",
    "target_eligible",
    "cohort_passes_movement_firing_rate",
    "cohort_passes_stability",
    "cohort_eligible",
    "shared_eligible",
)
ELIGIBILITY_COLUMNS = (
    *ELIGIBILITY_BASE_COLUMNS,
    *tuple(
        f"target_{trajectory_type}_stability_correlation"
        for trajectory_type in TRAJECTORY_TYPES
    ),
    *tuple(
        f"cohort_{trajectory_type}_stability_correlation"
        for trajectory_type in TRAJECTORY_TYPES
    ),
    *ELIGIBILITY_FLAG_COLUMNS,
)
MANIFEST_COLUMNS = (
    "artifact_key",
    "relative_path",
    "artifact_kind",
    "transfer_family",
    "source_trajectory",
    "target_trajectory",
    "value_role",
    "file_size_bytes",
    "sha256",
    "path_progression_decoding_comparison_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "cohort_epoch",
    "parameter_name",
    "parameter_sha256",
    "eligibility_rule_sha256",
    "transfer_spec_sha256",
    "decoding_output_rule_sha256",
    "decoding_bin_size_s",
    "sliding_window_size_bins",
    "spatial_bin_size_cm",
    "minimum_movement_firing_rate_hz",
    "minimum_stability_correlation",
    "error_mode",
    "error_summary",
    "min_bin_count",
    "n_units",
    "selected_units_sha256",
    "n_units_input",
    "n_units_eligible",
    "n_transfer_pairs_expected",
    "n_transfer_pairs_valid",
    "n_decoded_samples",
    "analysis_status",
)
MANUSCRIPT_PARAMETERS = {
    "decoding_bin_size_s": DEFAULT_DECODING_BIN_SIZE_S,
    "sliding_window_size_bins": DEFAULT_SLIDING_WINDOW_SIZE_BINS,
    "spatial_bin_size_cm": DEFAULT_SPATIAL_BIN_SIZE_CM,
    "error_mode": DEFAULT_ERROR_MODE,
    "error_summary": DEFAULT_ERROR_SUMMARY,
    "min_bin_count": DEFAULT_MIN_BIN_COUNT,
}
DECODING_OUTPUT_RULE = MappingProxyType(
    {
        "version": 1,
        "coordinate_unit": "normalized_path_progression",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": DEFAULT_ERROR_MODE,
        "error_summary": DEFAULT_ERROR_SUMMARY,
        "min_bin_count": DEFAULT_MIN_BIN_COUNT,
    }
)


def _decoding_output_rule(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the frozen output-summary semantics for effective parameters."""
    return {
        "version": 1,
        "coordinate_unit": "normalized_path_progression",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": str(parameters["error_mode"]),
        "error_summary": str(parameters["error_summary"]),
        "min_bin_count": int(parameters["min_bin_count"]),
    }


def build_cross_path_transfer_specs() -> tuple[dict[str, Any], ...]:
    """Return the fixed 16 directed path-progression transfers."""
    specs: list[dict[str, Any]] = []
    for turn_type, (trajectory_a, trajectory_b) in TURN_TRAJECTORY_PAIRS.items():
        for source, target in (
            (trajectory_a, trajectory_b),
            (trajectory_b, trajectory_a),
        ):
            specs.append(
                {
                    "transfer_family": "same_turn_cross_arm",
                    "source_trajectory": source,
                    "target_trajectory": target,
                    "source_turn_type": turn_type,
                    "turn_type": turn_type,
                    "flip_tuning_curve": False,
                }
            )
    for source, target in (
        ("center_to_left", "left_to_center"),
        ("left_to_center", "center_to_left"),
        ("center_to_right", "right_to_center"),
        ("right_to_center", "center_to_right"),
    ):
        for flip in (False, True):
            specs.append(
                {
                    "transfer_family": (
                        "opposite_turn_same_arm_flipped"
                        if flip
                        else "opposite_turn_same_arm"
                    ),
                    "source_trajectory": source,
                    "target_trajectory": target,
                    "flip_tuning_curve": flip,
                }
            )
    specs.extend(
        {
            "transfer_family": SAME_INBOUND_OUTBOUND_CROSS_ARM_FAMILY,
            "source_trajectory": source,
            "target_trajectory": target,
            "flip_tuning_curve": False,
        }
        for source, target in SAME_INBOUND_OUTBOUND_CROSS_ARM_PAIRS
    )
    keys = [
        (
            spec["transfer_family"],
            spec["source_trajectory"],
            spec["target_trajectory"],
        )
        for spec in specs
    ]
    if len(specs) != 16 or len(keys) != len(set(keys)):
        raise RuntimeError("Cross-path transfer specifications are incomplete.")
    return tuple(specs)


EXPECTED_TRANSFER_PAIR_COUNT = 16
CROSS_PATH_TRANSFER_SPECS = tuple(
    MappingProxyType(spec) for spec in build_cross_path_transfer_specs()
)
TRANSFER_PAIR_SPECS = CROSS_PATH_TRANSFER_SPECS
TRANSFER_SPEC_SHA256 = hashlib.sha256(
    json.dumps(
        [dict(spec) for spec in TRANSFER_PAIR_SPECS],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
ELIGIBILITY_RULE = MappingProxyType(
    {
        "version": 1,
        "cohort_policy": "target_and_cohort_intersection",
        "movement_operator": "greater_than_or_equal",
        "stability_aggregation": "at_least_one_trajectory",
        "stability_operator": "greater_than_or_equal",
        "null_stability_threshold": "disabled",
    }
)
TRANSFER_QC_STATUSES = (
    "valid",
    "no_units",
    "no_eligible_units",
    "no_source_movement",
    "no_target_movement",
    "no_target_count_bins",
)
ANALYSIS_STATUSES = (
    "valid",
    "partial_valid",
    "no_units",
    "no_eligible_units",
    "no_valid_decodes",
)


class TransferSupportError(ValueError):
    """Expected lack of temporal support for one directed transfer."""

    def __init__(self, status: str, message: str):
        if status not in TRANSFER_QC_STATUSES or status == "valid":
            raise ValueError(f"Unsupported transfer support status: {status!r}.")
        super().__init__(message)
        self.status = status


def _path_component(value: Any, *, name: str) -> str:
    """Return one safe non-empty path component."""
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return value


def _uuid_string(value: Any, *, name: str) -> str:
    """Return one canonical UUID string."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def get_decoding_comparison_artifact_path(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    path_progression_decoding_comparison_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return the canonical session-first UUID artifact directory."""
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
        path_progression_decoding_comparison_id,
        name="path_progression_decoding_comparison_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["region"]
        / result_id
    )


def get_decoding_artifact_paths(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    cohort_epoch: str,
    region: str,
    path_progression_decoding_comparison_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> dict[str, Path]:
    """Return all canonical paths for one UUID-keyed artifact bundle."""
    _path_component(cohort_epoch, name="cohort_epoch")
    artifact_dir = get_decoding_comparison_artifact_path(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        region=region,
        path_progression_decoding_comparison_id=(
            path_progression_decoding_comparison_id
        ),
        artifact_root=artifact_root,
    )
    return {
        "artifact_dir": artifact_dir,
        "artifact_manifest_path": artifact_dir / MANIFEST_FILENAME,
        "decoding_summary_path": artifact_dir / METRICS_FILENAME,
        "unit_eligibility_path": artifact_dir / ELIGIBILITY_FILENAME,
        "selected_units_path": artifact_dir / UNIT_FILENAME,
        "binned_error_path": artifact_dir / BINNED_FILENAME,
    }


def validate_decoding_parameters(
    *,
    decoding_bin_size_s: float = DEFAULT_DECODING_BIN_SIZE_S,
    sliding_window_size_bins: int = DEFAULT_SLIDING_WINDOW_SIZE_BINS,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    error_mode: str = DEFAULT_ERROR_MODE,
    error_summary: str = DEFAULT_ERROR_SUMMARY,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
) -> dict[str, Any]:
    """Return validated cross-path Bayesian decoding parameters."""
    for name, value in (
        ("sliding_window_size_bins", sliding_window_size_bins),
        ("min_bin_count", min_bin_count),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")
        if int(value) < 1:
            raise ValueError(f"{name} must be positive.")
    numeric = {
        "decoding_bin_size_s": float(decoding_bin_size_s),
        "spatial_bin_size_cm": float(spatial_bin_size_cm),
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in numeric.values()):
        raise ValueError("Decoding and spatial bin sizes must be positive and finite.")
    if error_mode not in {"signed", "absolute"}:
        raise ValueError("error_mode must be 'signed' or 'absolute'.")
    if error_summary not in {"mean_std", "median_iqr"}:
        raise ValueError("error_summary must be 'mean_std' or 'median_iqr'.")
    return {
        **numeric,
        "sliding_window_size_bins": int(sliding_window_size_bins),
        "error_mode": str(error_mode),
        "error_summary": str(error_summary),
        "min_bin_count": int(min_bin_count),
    }


def _identity_index(table: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return one unique persistent-unit table indexed by stable identity."""
    required = {
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{name} is missing identity columns {missing!r}.")
    output = table.copy()
    for column in required:
        output[column] = output[column].astype(str)
    expected = output["spikesorting_merge_id"] + ":" + output["unit_id"]
    if not np.array_equal(output["stable_unit_id"], expected) or output[
        "stable_unit_id"
    ].duplicated().any():
        raise ValueError(f"{name} persistent unit identities are inconsistent.")
    if "group_unit_id" in output:
        output["group_unit_id"] = output["group_unit_id"].astype(str)
        if (output["group_unit_id"] == "").any() or output[
            "group_unit_id"
        ].duplicated().any():
            raise ValueError(f"{name} group unit identities are inconsistent.")
    return output.set_index("stable_unit_id", drop=False)


def _aligned_epoch_eligibility_inputs(
    *,
    movement_table: pd.DataFrame,
    stability_tables_by_trajectory: Mapping[str, pd.DataFrame] | None,
    stable_ids: Sequence[str] | None,
    role: str,
    epoch: str,
    expected_metadata: Mapping[str, str] | None,
    minimum_stability_correlation: float | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, str]]:
    """Return one epoch's rates and four correlations in persistent-unit order."""
    movement = _identity_index(movement_table, name=f"{role} movement table")
    missing = sorted(
        {"group_unit_id", "movement_firing_rate_hz"}.difference(movement)
    )
    if missing:
        raise ValueError(f"{role} movement table is missing columns {missing!r}.")
    if "epoch" in movement and not np.all(
        movement["epoch"].astype(str) == str(epoch)
    ):
        raise ValueError(f"{role} movement table does not match epoch {epoch!r}.")
    observed_metadata = {}
    for column in ("animal_name", "date", "region"):
        if column not in movement or movement.empty:
            continue
        values = movement[column].astype(str).unique().tolist()
        if len(values) != 1 or not values[0]:
            raise ValueError(
                f"{role} movement table has inconsistent {column!r} metadata."
            )
        observed_metadata[column] = values[0]
        expected = None if expected_metadata is None else expected_metadata.get(column)
        if expected is not None and values[0] != str(expected):
            raise ValueError(
                f"{role} movement table does not match {column}={expected!r}."
            )
    if stable_ids is None:
        stable_ids = sorted(movement.index.astype(str).tolist())
    if set(movement.index) != set(stable_ids):
        raise ValueError(f"{role} movement unit identities do not match the cohort.")
    movement = movement.loc[list(stable_ids)]
    correlations: dict[str, np.ndarray] = {}
    if stability_tables_by_trajectory is None:
        if minimum_stability_correlation is not None:
            raise ValueError(
                f"{role} stability inputs are required when filtering is enabled."
            )
        correlations = {
            trajectory_type: np.full(len(stable_ids), np.nan, dtype=float)
            for trajectory_type in TRAJECTORY_TYPES
        }
    else:
        if set(stability_tables_by_trajectory) != set(TRAJECTORY_TYPES):
            raise ValueError(
                f"{role} stability inputs must contain exactly four trajectories."
            )
        for trajectory_type in TRAJECTORY_TYPES:
            table = _identity_index(
                stability_tables_by_trajectory[trajectory_type],
                name=f"{role} {trajectory_type} stability table",
            )
            if set(table.index) != set(stable_ids):
                raise ValueError(
                    f"{role} {trajectory_type} identities do not match the cohort."
                )
            if "stability_correlation" not in table:
                raise ValueError(
                    f"{role} {trajectory_type} is missing stability_correlation."
                )
            if "epoch" in table and not np.all(
                table["epoch"].astype(str) == str(epoch)
            ):
                raise ValueError(
                    f"{role} {trajectory_type} stability does not match "
                    f"epoch {epoch!r}."
                )
            if "trajectory_type" in table and not np.all(
                table["trajectory_type"].astype(str) == trajectory_type
            ):
                raise ValueError(
                    f"{role} stability table does not match {trajectory_type!r}."
                )
            for column, expected in observed_metadata.items():
                if column in table and not np.all(
                    table[column].astype(str) == expected
                ):
                    raise ValueError(
                        f"{role} {trajectory_type} stability does not match "
                        f"{column}={expected!r}."
                    )
            correlations[trajectory_type] = pd.to_numeric(
                table.loc[list(stable_ids), "stability_correlation"],
                errors="coerce",
            ).to_numpy(dtype=float)
    return movement, correlations, observed_metadata


def build_symmetric_cohort_eligibility_table(
    *,
    target_epoch: str,
    cohort_epoch: str,
    target_movement_firing_rate_table: pd.DataFrame,
    cohort_movement_firing_rate_table: pd.DataFrame,
    target_stability_tables_by_trajectory: Mapping[str, pd.DataFrame] | None,
    cohort_stability_tables_by_trajectory: Mapping[str, pd.DataFrame] | None,
    minimum_movement_firing_rate_hz: float,
    minimum_stability_correlation: float | None = None,
    animal_name: str | None = None,
    date: str | None = None,
    region: str | None = None,
) -> pd.DataFrame:
    """Return symmetric target/cohort eligibility for persistent units."""
    firing_rate_threshold = float(minimum_movement_firing_rate_hz)
    if not np.isfinite(firing_rate_threshold) or firing_rate_threshold < 0.0:
        raise ValueError(
            "minimum_movement_firing_rate_hz must be non-negative and finite."
        )
    stability_threshold = (
        None
        if minimum_stability_correlation is None
        else float(minimum_stability_correlation)
    )
    if stability_threshold is not None and (
        not np.isfinite(stability_threshold)
        or stability_threshold < -1.0
        or stability_threshold > 1.0
    ):
        raise ValueError(
            "minimum_stability_correlation must be None or within [-1, 1]."
        )
    expected_metadata = {
        name: str(value)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
        }.items()
        if value is not None
    }
    (
        target_movement,
        target_correlations,
        target_metadata,
    ) = _aligned_epoch_eligibility_inputs(
        movement_table=target_movement_firing_rate_table,
        stability_tables_by_trajectory=target_stability_tables_by_trajectory,
        stable_ids=None,
        role="target",
        epoch=str(target_epoch),
        expected_metadata=expected_metadata,
        minimum_stability_correlation=stability_threshold,
    )
    stable_ids = target_movement.index.astype(str).tolist()
    (
        cohort_movement,
        cohort_correlations,
        cohort_metadata,
    ) = _aligned_epoch_eligibility_inputs(
        movement_table=cohort_movement_firing_rate_table,
        stability_tables_by_trajectory=cohort_stability_tables_by_trajectory,
        stable_ids=stable_ids,
        role="cohort",
        epoch=str(cohort_epoch),
        expected_metadata=expected_metadata,
        minimum_stability_correlation=stability_threshold,
    )
    for column in set(target_metadata).intersection(cohort_metadata):
        if target_metadata[column] != cohort_metadata[column]:
            raise ValueError(
                f"Target and cohort {column!r} metadata do not match."
            )
    target_rates = pd.to_numeric(
        target_movement["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    cohort_rates = pd.to_numeric(
        cohort_movement["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.any(np.isinf(target_rates)) or np.any(np.isinf(cohort_rates)):
        raise ValueError("Movement firing rates may be finite or NaN, not infinite.")
    if np.any(target_rates[np.isfinite(target_rates)] < 0.0) or np.any(
        cohort_rates[np.isfinite(cohort_rates)] < 0.0
    ):
        raise ValueError("Finite movement firing rates must be non-negative.")
    target_rate_pass = np.isfinite(target_rates) & (
        target_rates >= firing_rate_threshold
    )
    cohort_rate_pass = np.isfinite(cohort_rates) & (
        cohort_rates >= firing_rate_threshold
    )
    if stability_threshold is None:
        target_stability_pass = np.ones(len(stable_ids), dtype=bool)
        cohort_stability_pass = np.ones(len(stable_ids), dtype=bool)
    else:
        target_matrix = np.column_stack(
            [target_correlations[name] for name in TRAJECTORY_TYPES]
        )
        cohort_matrix = np.column_stack(
            [cohort_correlations[name] for name in TRAJECTORY_TYPES]
        )
        if np.any(np.isinf(target_matrix)) or np.any(np.isinf(cohort_matrix)):
            raise ValueError(
                "Stability correlations may be finite or NaN, not infinite."
            )
        for matrix in (target_matrix, cohort_matrix):
            finite = matrix[np.isfinite(matrix)]
            if np.any(finite < -1.0 - 1e-9) or np.any(finite > 1.0 + 1e-9):
                raise ValueError(
                    "Finite stability correlations must be within [-1, 1]."
                )
        target_stability_pass = np.any(
            np.isfinite(target_matrix) & (target_matrix >= stability_threshold),
            axis=1,
        )
        cohort_stability_pass = np.any(
            np.isfinite(cohort_matrix) & (cohort_matrix >= stability_threshold),
            axis=1,
        )
    target_eligible = target_rate_pass & target_stability_pass
    cohort_eligible = cohort_rate_pass & cohort_stability_pass
    output = pd.DataFrame(
        {
            "spikesorting_merge_id": target_movement[
                "spikesorting_merge_id"
            ].to_numpy(),
            "unit_id": target_movement["unit_id"].to_numpy(),
            "stable_unit_id": stable_ids,
            "group_unit_id": target_movement["group_unit_id"].to_numpy(),
            "target_epoch": str(target_epoch),
            "cohort_epoch": str(cohort_epoch),
            "minimum_movement_firing_rate_hz": firing_rate_threshold,
            "minimum_stability_correlation": (
                np.nan if stability_threshold is None else stability_threshold
            ),
            "target_movement_firing_rate_hz": target_rates,
            "cohort_movement_firing_rate_hz": cohort_rates,
            **{
                f"target_{name}_stability_correlation": target_correlations[name]
                for name in TRAJECTORY_TYPES
            },
            **{
                f"cohort_{name}_stability_correlation": cohort_correlations[name]
                for name in TRAJECTORY_TYPES
            },
            "target_passes_movement_firing_rate": target_rate_pass,
            "target_passes_stability": target_stability_pass,
            "target_eligible": target_eligible,
            "cohort_passes_movement_firing_rate": cohort_rate_pass,
            "cohort_passes_stability": cohort_stability_pass,
            "cohort_eligible": cohort_eligible,
            "shared_eligible": target_eligible & cohort_eligible,
        }
    )
    return output.loc[:, list(ELIGIBILITY_COLUMNS)]


def get_shared_eligible_stable_unit_ids(table: pd.DataFrame) -> list[str]:
    """Return persistent unit IDs passing both target and cohort eligibility."""
    if list(table.columns) != list(ELIGIBILITY_COLUMNS):
        raise ValueError("Eligibility table does not use its canonical schema.")
    return table.loc[table["shared_eligible"], "stable_unit_id"].astype(str).tolist()


def _selected_spikes_and_units(
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    selected_unit_ids: Sequence[Any],
) -> tuple[Any, pd.DataFrame]:
    """Return the exact caller-selected population and persistent identities."""
    from v1ca1.spyglass.path_specific_place import _identity_rows

    selected = list(selected_unit_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("selected_unit_ids must be non-empty and unique.")
    available = list(spikes.keys())
    missing = [unit_id for unit_id in selected if unit_id not in available]
    if missing:
        raise ValueError(f"Selected units are absent from spikes: {missing!r}.")
    identities = _identity_rows(spikes, stable_unit_ids)
    by_group_key = {row["_group_key"]: row for row in identities}
    rows = [
        {
            **{name: str(by_group_key[key][name]) for name in IDENTITY_COLUMNS},
            "selection_index": index,
        }
        for index, key in enumerate(selected)
    ]
    unit_table = pd.DataFrame.from_records(rows).loc[:, list(UNIT_TABLE_COLUMNS)]

    import pynapple as nap

    selected_spikes = nap.TsGroup(
        {unit_id: spikes[unit_id] for unit_id in selected},
        time_support=spikes.time_support,
        time_units="s",
    )
    if set(selected_spikes.keys()) != set(selected):
        raise ValueError("Selected spike population is inconsistent.")
    return selected_spikes, unit_table


def _validate_tsd(tsd: Any, *, name: str, decoded: bool) -> None:
    """Validate one non-empty second-based true or decoded Tsd."""
    times = np.asarray(tsd.t, dtype=float).reshape(-1)
    values = np.asarray(tsd.d, dtype=float).reshape(-1)
    if times.shape != values.shape or not times.size:
        raise ValueError(f"{name} must contain aligned non-empty samples.")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} timestamps must be finite and increasing.")
    if np.any(np.isinf(values)) or (decoded and not np.all(np.isfinite(values))):
        raise ValueError(f"{name} contains invalid values.")


def _require_count_bins(
    spikes: Any,
    intervals: Any,
    *,
    bin_size_s: float,
    name: str,
) -> Any:
    """Require every target interval to produce at least one count bin."""
    from v1ca1.task_progression.decoding_comparison import (
        _filter_epochs_with_count_bins,
        _intervalset_to_arrays,
    )

    filtered = _filter_epochs_with_count_bins(spikes, intervals, bin_size_s)
    filtered_starts, _ = _intervalset_to_arrays(filtered)
    if not filtered_starts.size:
        raise TransferSupportError(
            "no_target_count_bins",
            f"No {name} interval produces a decoding bin.",
        )
    return filtered


def _build_path_progressions(
    *,
    position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_configuration: Mapping[str, Mapping[str, Any]],
    spatial_bin_size_cm: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build four natural-direction normalized path coordinates and bins."""
    if set(trajectory_intervals_by_type) != set(TRAJECTORY_TYPES):
        raise ValueError("Trajectory intervals must contain exactly four paths.")
    if set(graph_inputs_by_configuration) != set(TRAJECTORY_TYPES):
        raise ValueError("Path graphs must contain exactly four configurations.")
    from v1ca1.spyglass.dpp import build_dpp_bin_edges
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    progressions = {}
    lengths = {}
    for trajectory_type in TRAJECTORY_TYPES:
        progression, length = build_task_progression_from_graph(
            position=position,
            trajectory_interval=trajectory_intervals_by_type[trajectory_type],
            graph_inputs=graph_inputs_by_configuration[trajectory_type],
            trajectory_type=trajectory_type,
        )
        progressions[trajectory_type] = progression
        lengths[trajectory_type] = float(length)
    common_length = lengths[TRAJECTORY_TYPES[0]]
    if common_length <= 0.0 or any(
        not np.isclose(length, common_length, rtol=1e-10, atol=1e-12)
        for length in lengths.values()
    ):
        raise ValueError("The four path graphs must have one common length.")
    bins = build_dpp_bin_edges(
        common_length,
        bin_size_cm=spatial_bin_size_cm,
    )
    return progressions, bins


def _decode_cross_path_pair(
    *,
    spikes: Any,
    progressions: Mapping[str, Any],
    trajectory_intervals_by_type: Mapping[str, Any],
    movement_intervals: Any,
    bins: np.ndarray,
    spec: Mapping[str, Any],
    decoding_bin_size_s: float,
    sliding_window_size_bins: int,
) -> tuple[Any, Any]:
    """Decode one target path from a tuning curve fit on a source path."""
    import pynapple as nap

    from v1ca1.task_progression.decoding_comparison import (
        _flip_tuning_curves_along_position_axis,
    )

    source = str(spec["source_trajectory"])
    target = str(spec["target_trajectory"])
    train_epoch = trajectory_intervals_by_type[source].intersect(
        movement_intervals
    )
    test_epoch = trajectory_intervals_by_type[target].intersect(
        movement_intervals
    )
    if float(train_epoch.tot_length()) <= 0.0:
        raise TransferSupportError(
            "no_source_movement",
            f"Source path {source!r} has no movement support.",
        )
    if float(test_epoch.tot_length()) <= 0.0:
        raise TransferSupportError(
            "no_target_movement",
            f"Target path {target!r} has no movement support.",
        )
    test_epoch = _require_count_bins(
        spikes,
        test_epoch,
        bin_size_s=decoding_bin_size_s,
        name=f"target path {target!r}",
    )
    tuning_curves = nap.compute_tuning_curves(
        data=spikes,
        features=progressions[source],
        bins=[np.asarray(bins, dtype=float)],
        epochs=train_epoch,
    )
    if bool(spec.get("flip_tuning_curve", False)):
        tuning_curves = _flip_tuning_curves_along_position_axis(tuning_curves)
    decoded, _ = nap.decode_bayes(
        tuning_curves=tuning_curves,
        data=spikes,
        epochs=test_epoch,
        sliding_window_size=sliding_window_size_bins,
        bin_size=decoding_bin_size_s,
    )
    true = progressions[target].restrict(test_epoch)
    _validate_tsd(true, name="cross-path true", decoded=False)
    _validate_tsd(decoded, name="cross-path decoded", decoded=True)
    return true, decoded


def _metrics(true: Any, decoded: Any) -> dict[str, Any]:
    """Return strict non-empty decoding metrics."""
    from v1ca1.task_progression.decoding_comparison import (
        summarize_decoding_metrics,
    )

    metrics = summarize_decoding_metrics(true, decoded)
    if int(metrics["n_samples"]) <= 0 or any(
        not np.isfinite(float(metrics[name]))
        for name in METRIC_NAMES
        if name != "n_samples"
    ):
        raise ValueError("A decoding output has invalid summary metrics.")
    return metrics


def _binned_error(
    true: Any,
    decoded: Any,
    *,
    bins: np.ndarray,
    parameters: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the legacy-equivalent binned decoding-error summary."""
    from v1ca1.task_progression.decoding_comparison import (
        summarize_decoding_error_by_position,
    )

    return summarize_decoding_error_by_position(
        true,
        decoded,
        bin_edges=np.asarray(bins, dtype=float),
        error_mode=str(parameters["error_mode"]),
        summary=str(parameters["error_summary"]),
        min_count=int(parameters["min_bin_count"]),
    )


def _sha256_string(value: Any, *, name: str) -> str:
    """Return one normalized hexadecimal SHA-256 digest."""
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be one hexadecimal SHA-256 digest.")
    return digest


def _input_identity_rows(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return and validate all caller-supplied persistent unit identities."""
    from v1ca1.spyglass.path_specific_place import _identity_rows

    return _identity_rows(spikes, stable_unit_ids)


def _terminal_metric_rows(
    *,
    metadata: Mapping[str, Any],
    n_units: int,
    qc_status: str,
    qc_message: str,
) -> pd.DataFrame:
    """Return one explicit terminal-QC row for every directed transfer."""
    rows = []
    for spec in TRANSFER_PAIR_SPECS:
        rows.append(
            {
                **metadata,
                "transfer_family": str(spec["transfer_family"]),
                "source_trajectory": str(spec["source_trajectory"]),
                "target_trajectory": str(spec["target_trajectory"]),
                "flip_tuning_curve": bool(
                    spec.get("flip_tuning_curve", False)
                ),
                "coordinate_unit": "normalized_path_progression",
                "n_units": int(n_units),
                "mae": np.nan,
                "rmse": np.nan,
                "mean_signed_error": np.nan,
                "median_abs_error": np.nan,
                "n_samples": 0,
                "qc_status": str(qc_status),
                "qc_message": str(qc_message),
            }
        )
    return pd.DataFrame.from_records(rows).loc[:, list(METRIC_COLUMNS)]


def _empty_binned_error_table() -> pd.DataFrame:
    """Return an empty cross-path error table with its canonical columns."""
    return pd.DataFrame(columns=list(BINNED_COLUMNS))


def _analysis_status(
    *,
    n_units_input: int,
    n_units_eligible: int,
    n_transfer_pairs_valid: int,
) -> str:
    """Return the canonical terminal status for one comparison."""
    if n_units_input == 0:
        return "no_units"
    if n_units_eligible == 0:
        return "no_eligible_units"
    if n_transfer_pairs_valid == EXPECTED_TRANSFER_PAIR_COUNT:
        return "valid"
    if n_transfer_pairs_valid == 0:
        return "no_valid_decodes"
    return "partial_valid"


def compute_path_progression_decoding_comparison(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    cohort_epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    target_movement_firing_rate_table: pd.DataFrame,
    cohort_movement_firing_rate_table: pd.DataFrame,
    target_stability_tables_by_trajectory: Mapping[
        str, pd.DataFrame
    ] | None,
    cohort_stability_tables_by_trajectory: Mapping[
        str, pd.DataFrame
    ] | None,
    position: Any,
    trajectory_intervals: Mapping[str, Any],
    graph_inputs: Mapping[str, Mapping[str, Any]],
    movement_interval: Any,
    path_progression_decoding_comparison_id: Any,
    decoding_bin_size_s: float = DEFAULT_DECODING_BIN_SIZE_S,
    sliding_window_size_bins: int = DEFAULT_SLIDING_WINDOW_SIZE_BINS,
    spatial_bin_size_cm: float = DEFAULT_SPATIAL_BIN_SIZE_CM,
    minimum_movement_firing_rate_hz: float = 0.5,
    minimum_stability_correlation: float | None = None,
    parameter_name: str = "default",
    parameter_sha256: str | None = None,
    eligibility_rule_sha256: str | None = None,
    transfer_spec_sha256: str = TRANSFER_SPEC_SHA256,
    decoding_output_rule_sha256: str | None = None,
    error_mode: str = DEFAULT_ERROR_MODE,
    error_summary: str = DEFAULT_ERROR_SUMMARY,
    min_bin_count: int = DEFAULT_MIN_BIN_COUNT,
) -> dict[str, Any]:
    """Compute all directed transfers with one symmetric shared unit cohort."""
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256

    parameters = validate_decoding_parameters(
        decoding_bin_size_s=decoding_bin_size_s,
        sliding_window_size_bins=sliding_window_size_bins,
        spatial_bin_size_cm=spatial_bin_size_cm,
        error_mode=error_mode,
        error_summary=error_summary,
        min_bin_count=min_bin_count,
    )
    metadata_components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "cohort_epoch": cohort_epoch,
        }.items()
    }
    result_id = _uuid_string(
        path_progression_decoding_comparison_id,
        name="path_progression_decoding_comparison_id",
    )
    parameter_name = str(parameter_name).strip()
    if not parameter_name or len(parameter_name) > 64:
        raise ValueError("parameter_name must be non-empty and at most 64 characters.")
    parameter_snapshot = {
        "path_progression_decoding_param_name": parameter_name,
        "decoding_bin_size_s": parameters["decoding_bin_size_s"],
        "sliding_window_size_bins": parameters["sliding_window_size_bins"],
        "spatial_bin_size_cm": parameters["spatial_bin_size_cm"],
        "minimum_movement_firing_rate_hz": float(
            minimum_movement_firing_rate_hz
        ),
        "minimum_stability_correlation": (
            None
            if minimum_stability_correlation is None
            else float(minimum_stability_correlation)
        ),
    }
    expected_parameter_sha256 = provenance_sha256(parameter_snapshot)
    parameter_sha256 = (
        expected_parameter_sha256
        if parameter_sha256 is None
        else _sha256_string(parameter_sha256, name="parameter_sha256")
    )
    if parameter_sha256 != expected_parameter_sha256:
        raise ValueError("parameter_sha256 does not match the supplied parameters.")
    expected_eligibility_sha256 = provenance_sha256(dict(ELIGIBILITY_RULE))
    eligibility_rule_sha256 = (
        expected_eligibility_sha256
        if eligibility_rule_sha256 is None
        else _sha256_string(
            eligibility_rule_sha256,
            name="eligibility_rule_sha256",
        )
    )
    if eligibility_rule_sha256 != expected_eligibility_sha256:
        raise ValueError("eligibility_rule_sha256 does not match the fixed rule.")
    transfer_spec_sha256 = _sha256_string(
        transfer_spec_sha256,
        name="transfer_spec_sha256",
    )
    if transfer_spec_sha256 != TRANSFER_SPEC_SHA256:
        raise ValueError("transfer_spec_sha256 does not match the fixed transfers.")
    expected_output_rule_sha256 = provenance_sha256(
        _decoding_output_rule(parameters)
    )
    decoding_output_rule_sha256 = (
        expected_output_rule_sha256
        if decoding_output_rule_sha256 is None
        else _sha256_string(
            decoding_output_rule_sha256,
            name="decoding_output_rule_sha256",
        )
    )
    if decoding_output_rule_sha256 != expected_output_rule_sha256:
        raise ValueError(
            "decoding_output_rule_sha256 does not match the fixed output rule."
        )

    input_identities = _input_identity_rows(spikes, stable_unit_ids)
    n_units_input = len(input_identities)
    eligibility = build_symmetric_cohort_eligibility_table(
        target_epoch=metadata_components["epoch"],
        cohort_epoch=metadata_components["cohort_epoch"],
        target_movement_firing_rate_table=(
            target_movement_firing_rate_table
        ),
        cohort_movement_firing_rate_table=(
            cohort_movement_firing_rate_table
        ),
        target_stability_tables_by_trajectory=(
            target_stability_tables_by_trajectory
        ),
        cohort_stability_tables_by_trajectory=(
            cohort_stability_tables_by_trajectory
        ),
        minimum_movement_firing_rate_hz=(
            minimum_movement_firing_rate_hz
        ),
        minimum_stability_correlation=minimum_stability_correlation,
        animal_name=metadata_components["animal_name"],
        date=metadata_components["date"],
        region=metadata_components["region"],
    )
    input_by_stable_id = {
        str(row["stable_unit_id"]): row for row in input_identities
    }
    if set(eligibility["stable_unit_id"].astype(str)) != set(
        input_by_stable_id
    ):
        raise ValueError("Eligibility inputs do not match the supplied spike units.")
    # TsGroup keys are ephemeral and may differ across saved upstream
    # artifacts or a later reload. Join on persistent identities, then retain
    # this compute-time TsGroup's keys only as local artifact provenance.
    eligibility = eligibility.copy()
    eligibility["group_unit_id"] = [
        str(input_by_stable_id[stable_id]["group_unit_id"])
        for stable_id in eligibility["stable_unit_id"].astype(str)
    ]
    selected_stable_ids = get_shared_eligible_stable_unit_ids(eligibility)
    selected_group_ids = [
        input_by_stable_id[stable_id]["_group_key"]
        for stable_id in selected_stable_ids
    ]
    if selected_group_ids:
        selected_spikes, unit_table = _selected_spikes_and_units(
            spikes=spikes,
            stable_unit_ids=stable_unit_ids,
            selected_unit_ids=selected_group_ids,
        )
    else:
        selected_spikes = None
        unit_table = pd.DataFrame(columns=list(UNIT_TABLE_COLUMNS))
    n_units_eligible = len(unit_table)
    eligible_units_sha256 = unit_identity_sha256(
        unit_table.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    metadata = {
        "path_progression_decoding_comparison_id": result_id,
        **metadata_components,
        "parameter_name": parameter_name,
        "parameter_sha256": parameter_sha256,
        "eligibility_rule_sha256": eligibility_rule_sha256,
        "transfer_spec_sha256": transfer_spec_sha256,
        "decoding_output_rule_sha256": decoding_output_rule_sha256,
    }
    table_metadata = {
        name: metadata[name]
        for name in (
            "path_progression_decoding_comparison_id",
            "animal_name",
            "date",
            "region",
            "epoch",
            "cohort_epoch",
        )
    }

    outputs: dict[tuple[str, str, str], dict[str, Any]] = {}
    if n_units_input == 0:
        metrics = _terminal_metric_rows(
            metadata=table_metadata,
            n_units=0,
            qc_status="no_units",
            qc_message="The regional spike source contains no units.",
        )
        binned_error = _empty_binned_error_table()
    elif n_units_eligible == 0:
        metrics = _terminal_metric_rows(
            metadata=table_metadata,
            n_units=0,
            qc_status="no_eligible_units",
            qc_message="No units pass the symmetric target/cohort filter.",
        )
        binned_error = _empty_binned_error_table()
    else:
        progressions, bins = _build_path_progressions(
            position=position,
            trajectory_intervals_by_type=trajectory_intervals,
            graph_inputs_by_configuration=graph_inputs,
            spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        )
        metric_rows = []
        binned_tables = []
        for spec in TRANSFER_PAIR_SPECS:
            key = (
                str(spec["transfer_family"]),
                str(spec["source_trajectory"]),
                str(spec["target_trajectory"]),
            )
            transfer_metadata = {
                "transfer_family": key[0],
                "source_trajectory": key[1],
                "target_trajectory": key[2],
                "flip_tuning_curve": bool(
                    spec.get("flip_tuning_curve", False)
                ),
                "coordinate_unit": "normalized_path_progression",
            }
            try:
                true, decoded = _decode_cross_path_pair(
                    spikes=selected_spikes,
                    progressions=progressions,
                    trajectory_intervals_by_type=trajectory_intervals,
                    movement_intervals=movement_interval,
                    bins=bins,
                    spec=spec,
                    decoding_bin_size_s=parameters["decoding_bin_size_s"],
                    sliding_window_size_bins=parameters[
                        "sliding_window_size_bins"
                    ],
                )
            except TransferSupportError as exc:
                metric_rows.append(
                    {
                        **table_metadata,
                        **transfer_metadata,
                        "n_units": n_units_eligible,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mean_signed_error": np.nan,
                        "median_abs_error": np.nan,
                        "n_samples": 0,
                        "qc_status": exc.status,
                        "qc_message": str(exc),
                    }
                )
                continue
            outputs[key] = {"true": true, "decoded": decoded}
            metric_rows.append(
                {
                    **table_metadata,
                    **transfer_metadata,
                    "n_units": n_units_eligible,
                    **_metrics(true, decoded),
                    "qc_status": "valid",
                    "qc_message": "",
                }
            )
            binned = _binned_error(
                true,
                decoded,
                bins=bins,
                parameters=parameters,
            )
            for column, value in reversed(tuple(transfer_metadata.items())):
                binned.insert(0, column, value)
            for column, value in reversed(tuple(table_metadata.items())):
                binned.insert(0, column, value)
            binned_tables.append(binned)
        metrics = pd.DataFrame.from_records(metric_rows).loc[
            :, list(METRIC_COLUMNS)
        ]
        binned_error = (
            pd.concat(binned_tables, ignore_index=True).loc[
                :, list(BINNED_COLUMNS)
            ]
            if binned_tables
            else _empty_binned_error_table()
        )

    n_transfer_pairs_valid = int((metrics["qc_status"] == "valid").sum())
    n_decoded_samples = int(
        pd.to_numeric(metrics["n_samples"], errors="raise").sum()
    )
    status = _analysis_status(
        n_units_input=n_units_input,
        n_units_eligible=n_units_eligible,
        n_transfer_pairs_valid=n_transfer_pairs_valid,
    )
    result = {
        "metadata": metadata,
        "parameters": {
            **parameters,
            "minimum_movement_firing_rate_hz": float(
                minimum_movement_firing_rate_hz
            ),
            "minimum_stability_correlation": (
                None
                if minimum_stability_correlation is None
                else float(minimum_stability_correlation)
            ),
        },
        "unit_eligibility": eligibility,
        "selected_units": unit_table,
        "cross_path_outputs": outputs,
        "cross_path_metrics": metrics,
        "cross_path_binned_error": binned_error,
        "n_units_input": n_units_input,
        "n_units_eligible": n_units_eligible,
        "n_transfer_pairs_expected": EXPECTED_TRANSFER_PAIR_COUNT,
        "n_transfer_pairs_valid": n_transfer_pairs_valid,
        "n_decoded_samples": n_decoded_samples,
        "analysis_status": status,
        "eligible_units_sha256": eligible_units_sha256,
    }
    return validate_decoding_comparison_result(result)


def _validate_exact_columns(
    table: pd.DataFrame,
    expected: Sequence[str],
    *,
    name: str,
) -> None:
    """Require one table to use its exact canonical schema."""
    if list(table.columns) != list(expected):
        raise ValueError(f"{name} does not use its exact canonical schema.")


def validate_decoding_comparison_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return one in-memory cross-path decoding result."""
    required = {
        "metadata",
        "parameters",
        "unit_eligibility",
        "selected_units",
        "cross_path_outputs",
        "cross_path_metrics",
        "cross_path_binned_error",
        "n_units_input",
        "n_units_eligible",
        "n_transfer_pairs_expected",
        "n_transfer_pairs_valid",
        "n_decoded_samples",
        "analysis_status",
        "eligible_units_sha256",
    }
    if set(result) != required:
        raise ValueError("Decoding result keys do not match the canonical schema.")
    metadata = dict(result["metadata"])
    metadata_fields = {
        "path_progression_decoding_comparison_id",
        "animal_name",
        "date",
        "region",
        "epoch",
        "cohort_epoch",
        "parameter_name",
        "parameter_sha256",
        "eligibility_rule_sha256",
        "transfer_spec_sha256",
        "decoding_output_rule_sha256",
    }
    if set(metadata) != metadata_fields:
        raise ValueError("Decoding result metadata is incomplete.")
    _uuid_string(
        metadata["path_progression_decoding_comparison_id"],
        name="path_progression_decoding_comparison_id",
    )
    for name in ("animal_name", "date", "region", "epoch", "cohort_epoch"):
        _path_component(metadata[name], name=name)
    parameters = dict(result["parameters"])
    parameter_fields = {
        *MANUSCRIPT_PARAMETERS,
        "minimum_movement_firing_rate_hz",
        "minimum_stability_correlation",
    }
    if set(parameters) != parameter_fields:
        raise ValueError("Decoding parameters do not use the canonical schema.")
    validate_decoding_parameters(
        **{name: parameters[name] for name in MANUSCRIPT_PARAMETERS}
    )
    firing_rate_threshold = float(
        parameters["minimum_movement_firing_rate_hz"]
    )
    stability_value = parameters["minimum_stability_correlation"]
    stability_threshold = (
        None if stability_value is None else float(stability_value)
    )
    if not np.isfinite(firing_rate_threshold) or firing_rate_threshold < 0.0:
        raise ValueError("The movement firing-rate threshold is invalid.")
    if stability_threshold is not None and (
        not np.isfinite(stability_threshold)
        or not -1.0 <= stability_threshold <= 1.0
    ):
        raise ValueError("The stability threshold is invalid.")
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256

    parameter_snapshot = {
        "path_progression_decoding_param_name": metadata["parameter_name"],
        "decoding_bin_size_s": parameters["decoding_bin_size_s"],
        "sliding_window_size_bins": parameters["sliding_window_size_bins"],
        "spatial_bin_size_cm": parameters["spatial_bin_size_cm"],
        "minimum_movement_firing_rate_hz": firing_rate_threshold,
        "minimum_stability_correlation": stability_threshold,
    }
    if _sha256_string(
        metadata["parameter_sha256"], name="parameter_sha256"
    ) != provenance_sha256(parameter_snapshot):
        raise ValueError("Decoding parameter provenance is inconsistent.")
    if _sha256_string(
        metadata["eligibility_rule_sha256"],
        name="eligibility_rule_sha256",
    ) != provenance_sha256(dict(ELIGIBILITY_RULE)):
        raise ValueError("Decoding eligibility-rule provenance is inconsistent.")
    if _sha256_string(
        metadata["transfer_spec_sha256"], name="transfer_spec_sha256"
    ) != TRANSFER_SPEC_SHA256:
        raise ValueError("Decoding transfer-spec provenance is inconsistent.")
    if _sha256_string(
        metadata["decoding_output_rule_sha256"],
        name="decoding_output_rule_sha256",
    ) != provenance_sha256(_decoding_output_rule(parameters)):
        raise ValueError("Decoding output-rule provenance is inconsistent.")
    if not str(metadata["parameter_name"]).strip() or len(
        str(metadata["parameter_name"])
    ) > 64:
        raise ValueError("Decoding parameter_name is invalid.")

    eligibility = result["unit_eligibility"]
    _validate_exact_columns(
        eligibility,
        ELIGIBILITY_COLUMNS,
        name="unit_eligibility",
    )
    identity = eligibility.loc[:, list(IDENTITY_COLUMNS)].astype(str)
    expected_stable = identity["spikesorting_merge_id"] + ":" + identity["unit_id"]
    if not np.array_equal(
        identity["stable_unit_id"].to_numpy(), expected_stable.to_numpy()
    ) or identity["stable_unit_id"].duplicated().any() or identity[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError("Eligibility unit identities are inconsistent.")
    if not eligibility.empty:
        if not np.all(
            eligibility["target_epoch"].astype(str) == str(metadata["epoch"])
        ) or not np.all(
            eligibility["cohort_epoch"].astype(str)
            == str(metadata["cohort_epoch"])
        ):
            raise ValueError("Eligibility epoch metadata is inconsistent.")
        if not np.allclose(
            pd.to_numeric(
                eligibility["minimum_movement_firing_rate_hz"],
                errors="coerce",
            ),
            firing_rate_threshold,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Eligibility firing-rate threshold is inconsistent.")
        observed_stability_threshold = pd.to_numeric(
            eligibility["minimum_stability_correlation"], errors="coerce"
        ).to_numpy(dtype=float)
        if stability_threshold is None:
            if not np.all(np.isnan(observed_stability_threshold)):
                raise ValueError("Disabled stability filtering must use NaN.")
        elif not np.allclose(
            observed_stability_threshold,
            stability_threshold,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Eligibility stability threshold is inconsistent.")
    for role in ("target", "cohort"):
        rates = pd.to_numeric(
            eligibility[f"{role}_movement_firing_rate_hz"], errors="coerce"
        ).to_numpy(dtype=float)
        if np.any(np.isinf(rates)) or np.any(rates[np.isfinite(rates)] < 0.0):
            raise ValueError("Eligibility firing rates contain invalid values.")
        expected_rate_pass = np.isfinite(rates) & (
            rates >= firing_rate_threshold
        )
        observed_rate_pass = eligibility[
            f"{role}_passes_movement_firing_rate"
        ].to_numpy(dtype=bool)
        if not np.array_equal(observed_rate_pass, expected_rate_pass):
            raise ValueError("Eligibility firing-rate flags are inconsistent.")
        correlations = np.column_stack(
            [
                pd.to_numeric(
                    eligibility[
                        f"{role}_{trajectory_type}_stability_correlation"
                    ],
                    errors="coerce",
                ).to_numpy(dtype=float)
                for trajectory_type in TRAJECTORY_TYPES
            ]
        )
        if np.any(np.isinf(correlations)):
            raise ValueError("Eligibility correlations may not be infinite.")
        finite_correlations = correlations[np.isfinite(correlations)]
        if np.any(finite_correlations < -1.0 - 1e-9) or np.any(
            finite_correlations > 1.0 + 1e-9
        ):
            raise ValueError("Eligibility correlations must be within [-1, 1].")
        expected_stability_pass = (
            np.ones(len(eligibility), dtype=bool)
            if stability_threshold is None
            else np.any(
                np.isfinite(correlations)
                & (correlations >= stability_threshold),
                axis=1,
            )
        )
        observed_stability_pass = eligibility[
            f"{role}_passes_stability"
        ].to_numpy(dtype=bool)
        if not np.array_equal(
            observed_stability_pass, expected_stability_pass
        ):
            raise ValueError("Eligibility stability flags are inconsistent.")
        expected_eligible = expected_rate_pass & expected_stability_pass
        if not np.array_equal(
            eligibility[f"{role}_eligible"].to_numpy(dtype=bool),
            expected_eligible,
        ):
            raise ValueError("Per-epoch eligibility flags are inconsistent.")
    expected_shared = eligibility["target_eligible"].to_numpy(dtype=bool) & (
        eligibility["cohort_eligible"].to_numpy(dtype=bool)
    )
    if not np.array_equal(
        eligibility["shared_eligible"].to_numpy(dtype=bool), expected_shared
    ):
        raise ValueError("Shared eligibility flags are inconsistent.")

    units = result["selected_units"]
    _validate_exact_columns(units, UNIT_TABLE_COLUMNS, name="selected_units")
    if units["stable_unit_id"].duplicated().any() or units[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError("selected_units must contain unique identities.")
    selected_expected_stable = (
        units["spikesorting_merge_id"].astype(str)
        + ":"
        + units["unit_id"].astype(str)
    )
    if not np.array_equal(
        units["stable_unit_id"].astype(str).to_numpy(),
        selected_expected_stable.to_numpy(),
    ) or not np.array_equal(
        units["selection_index"].to_numpy(), np.arange(len(units))
    ):
        raise ValueError("selected_units identities or order are inconsistent.")
    expected_selected = eligibility.loc[
        expected_shared, list(IDENTITY_COLUMNS)
    ].reset_index(drop=True)
    observed_selected = units.loc[:, list(IDENTITY_COLUMNS)].reset_index(
        drop=True
    )
    if not observed_selected.astype(str).equals(expected_selected.astype(str)):
        raise ValueError("selected_units do not equal the shared eligible set.")

    expected_keys = tuple(
        (
            str(spec["transfer_family"]),
            str(spec["source_trajectory"]),
            str(spec["target_trajectory"]),
        )
        for spec in CROSS_PATH_TRANSFER_SPECS
    )
    metrics = result["cross_path_metrics"]
    binned = result["cross_path_binned_error"]
    _validate_exact_columns(metrics, METRIC_COLUMNS, name="cross_path_metrics")
    _validate_exact_columns(
        binned, BINNED_COLUMNS, name="cross_path_binned_error"
    )
    if len(metrics) != EXPECTED_TRANSFER_PAIR_COUNT:
        raise ValueError("The decoding summary must contain all 16 transfers.")
    for name, table in (
        ("cross_path_metrics", metrics),
        ("cross_path_binned_error", binned),
    ):
        for column in (
            "path_progression_decoding_comparison_id",
            "animal_name",
            "date",
            "region",
            "epoch",
            "cohort_epoch",
        ):
            expected = metadata[column]
            if not table.empty and not np.all(
                table[column].astype(str) == str(expected)
            ):
                raise ValueError(f"{name} has inconsistent {column!r} values.")
        numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if np.any(np.isinf(numeric)):
            raise ValueError(f"{name} may contain finite values or NaN, not inf.")
    observed_keys = list(
        zip(
            metrics["transfer_family"],
            metrics["source_trajectory"],
            metrics["target_trajectory"],
            strict=True,
        )
    )
    if observed_keys != list(expected_keys) or not np.all(
        metrics["n_units"] == len(units)
    ):
        raise ValueError("Cross-path metric rows, counts, or QC are inconsistent.")
    for row_index, spec in enumerate(TRANSFER_PAIR_SPECS):
        if bool(metrics.iloc[row_index]["flip_tuning_curve"]) != bool(
            spec.get("flip_tuning_curve", False)
        ) or str(metrics.iloc[row_index]["coordinate_unit"]) != (
            "normalized_path_progression"
        ):
            raise ValueError("Transfer coordinate or flip metadata is inconsistent.")
    statuses = metrics["qc_status"].astype(str)
    if not set(statuses).issubset(TRANSFER_QC_STATUSES):
        raise ValueError("The decoding summary contains unsupported QC statuses.")
    valid_keys = tuple(
        key
        for key, status in zip(expected_keys, statuses, strict=True)
        if status == "valid"
    )
    outputs = result["cross_path_outputs"]
    if tuple(outputs) != valid_keys:
        raise ValueError("Time outputs must exactly match valid transfer rows.")
    metric_values = metrics.loc[:, list(METRIC_NAMES)].apply(
        pd.to_numeric, errors="coerce"
    )
    for row_index, status in enumerate(statuses):
        row = metrics.iloc[row_index]
        if status == "valid":
            if not np.all(np.isfinite(metric_values.iloc[row_index])) or int(
                row["n_samples"]
            ) <= 0 or str(row["qc_message"]):
                raise ValueError("Valid transfer QC requires finite metrics.")
        elif not (
            np.all(np.isnan(metric_values.iloc[row_index][:-1]))
            and int(row["n_samples"]) == 0
            and bool(str(row["qc_message"]))
        ):
            raise ValueError("Invalid transfer QC requires empty metrics and a reason.")
    for key, output in outputs.items():
        if set(output) != {"true", "decoded"}:
            raise ValueError(f"Invalid output roles for transfer {key!r}.")
        _validate_tsd(output["true"], name=f"{key} true", decoded=False)
        _validate_tsd(output["decoded"], name=f"{key} decoded", decoded=True)
        expected = _metrics(output["true"], output["decoded"])
        row = metrics[
            (metrics["transfer_family"] == key[0])
            & (metrics["source_trajectory"] == key[1])
            & (metrics["target_trajectory"] == key[2])
        ].iloc[0]
        for metric in METRIC_NAMES:
            if not np.isclose(
                float(row[metric]),
                float(expected[metric]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"Cross-path metric {metric!r} is inconsistent.")

        subset = binned[
            (binned["transfer_family"] == key[0])
            & (binned["source_trajectory"] == key[1])
            & (binned["target_trajectory"] == key[2])
        ].reset_index(drop=True)
        if subset.empty:
            raise ValueError("Every valid transfer needs one binned-error grid.")
        metric_row = metrics[
            (metrics["transfer_family"] == key[0])
            & (metrics["source_trajectory"] == key[1])
            & (metrics["target_trajectory"] == key[2])
        ].iloc[0]
        if not np.all(
            subset["flip_tuning_curve"].to_numpy(dtype=bool)
            == bool(metric_row["flip_tuning_curve"])
        ) or not np.all(
            subset["coordinate_unit"].astype(str)
            == str(metric_row["coordinate_unit"])
        ):
            raise ValueError("Binned-error transfer metadata is inconsistent.")
        left = subset["bin_left"].to_numpy(dtype=float)
        right = subset["bin_right"].to_numpy(dtype=float)
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)) or (
            np.any(right <= left)
        ) or not np.allclose(left[1:], right[:-1], rtol=0.0, atol=1e-12):
            raise ValueError("Binned-error edges are not one contiguous grid.")
        bins = np.concatenate((left[:1], right))
        expected_binned = _binned_error(
            output["true"],
            output["decoded"],
            bins=bins,
            parameters=parameters,
        ).loc[:, list(BINNED_VALUE_COLUMNS)]
        try:
            pd.testing.assert_frame_equal(
                subset.loc[:, list(BINNED_VALUE_COLUMNS)],
                expected_binned,
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as exc:
            raise ValueError("Cross-path binned errors are inconsistent.") from exc
    if set(
        zip(
            binned["transfer_family"],
            binned["source_trajectory"],
            binned["target_trajectory"],
            strict=True,
        )
    ) != set(valid_keys):
        raise ValueError("Binned-error rows must exactly match valid transfers.")

    scalar_counts = {
        "n_units_input": len(eligibility),
        "n_units_eligible": len(units),
        "n_transfer_pairs_expected": EXPECTED_TRANSFER_PAIR_COUNT,
        "n_transfer_pairs_valid": len(outputs),
        "n_decoded_samples": int(metrics["n_samples"].sum()),
    }
    for name, expected in scalar_counts.items():
        value = result[name]
        if isinstance(value, bool) or int(value) != expected:
            raise ValueError(f"{name} is inconsistent with the decoding bundle.")
    expected_status = _analysis_status(
        n_units_input=scalar_counts["n_units_input"],
        n_units_eligible=scalar_counts["n_units_eligible"],
        n_transfer_pairs_valid=scalar_counts["n_transfer_pairs_valid"],
    )
    if str(result["analysis_status"]) != expected_status:
        raise ValueError("analysis_status is inconsistent with bundle counts.")
    digest = unit_identity_sha256(
        units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict("records")
    )
    if _sha256_string(
        result["eligible_units_sha256"], name="eligible_units_sha256"
    ) != digest:
        raise ValueError("eligible_units_sha256 is inconsistent.")
    return dict(result)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_units_sha256(units: pd.DataFrame) -> str:
    """Return the ordered selected persistent-unit digest."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    return unit_identity_sha256(
        units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict("records")
    )


def _npz_filename(key: tuple[str, str, str], role: str) -> str:
    """Return one canonical transfer time-domain filename."""
    family, source, target = key
    return f"cross_{family}_{source}_to_{target}_{role}.npz"


def _manifest_row(
    path: Path,
    *,
    artifact_key: str,
    artifact_kind: str,
    common: Mapping[str, Any],
    transfer_key: tuple[str, str, str] | None = None,
    value_role: str = "",
) -> dict[str, Any]:
    """Return one checksummed file-manifest row."""
    family, source, target = transfer_key or ("", "", "")
    return {
        "artifact_key": artifact_key,
        "relative_path": path.name,
        "artifact_kind": artifact_kind,
        "transfer_family": family,
        "source_trajectory": source,
        "target_trajectory": target,
        "value_role": value_role,
        "file_size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
        **common,
    }


def write_decoding_comparison_artifact(
    result: Mapping[str, Any],
    path: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically write and validate one UUID decoding artifact directory."""
    result = validate_decoding_comparison_result(result)
    destination = Path(path)
    result_id = _uuid_string(
        result["metadata"]["path_progression_decoding_comparison_id"],
        name="path_progression_decoding_comparison_id",
    )
    if destination.name != result_id:
        raise ValueError("Artifact directory name must equal the result UUID.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite decoding artifact directory: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir()
    common = {
        **result["metadata"],
        **result["parameters"],
        "n_units": len(result["selected_units"]),
        "selected_units_sha256": result["eligible_units_sha256"],
        **{
            name: result[name]
            for name in (
                "n_units_input",
                "n_units_eligible",
                "n_transfer_pairs_expected",
                "n_transfer_pairs_valid",
                "n_decoded_samples",
                "analysis_status",
            )
        },
    }
    rows = []
    backup = None
    try:
        for artifact_key, filename, table in (
            (
                "unit_eligibility",
                ELIGIBILITY_FILENAME,
                result["unit_eligibility"],
            ),
            ("selected_units", UNIT_FILENAME, result["selected_units"]),
            (
                "decoding_summary",
                METRICS_FILENAME,
                result["cross_path_metrics"],
            ),
            (
                "cross_path_binned_error",
                BINNED_FILENAME,
                result["cross_path_binned_error"],
            ),
        ):
            file_path = temporary / filename
            table.to_parquet(file_path, index=False)
            rows.append(
                _manifest_row(
                    file_path,
                    artifact_key=artifact_key,
                    artifact_kind="parquet",
                    common=common,
                )
            )
        for key, output in result["cross_path_outputs"].items():
            for role in ("true", "decoded"):
                file_path = temporary / _npz_filename(key, role)
                output[role].save(file_path)
                rows.append(
                    _manifest_row(
                        file_path,
                        artifact_key=f"cross:{key[0]}:{key[1]}:{key[2]}:{role}",
                        artifact_kind="pynapple_npz",
                        common=common,
                        transfer_key=key,
                        value_role=role,
                    )
                )
        manifest = pd.DataFrame.from_records(rows).loc[:, list(MANIFEST_COLUMNS)]
        manifest.to_parquet(temporary / MANIFEST_FILENAME, index=False)
        load_decoding_comparison_artifact(
            temporary,
            _allow_temporary_name=True,
        )
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.backup"
            )
            os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    return {
        "path": destination,
        "manifest": pd.read_parquet(destination / MANIFEST_FILENAME),
        "artifact_manifest_path": destination / MANIFEST_FILENAME,
        "decoding_summary_path": destination / METRICS_FILENAME,
        "unit_eligibility_path": destination / ELIGIBILITY_FILENAME,
        "selected_units_path": destination / UNIT_FILENAME,
        "binned_error_path": destination / BINNED_FILENAME,
        "n_units": len(result["selected_units"]),
        "selected_units_sha256": common["selected_units_sha256"],
        "_created_artifact_paths": [str(destination)],
    }


def _validate_artifact_paths(paths: Mapping[str, Path]) -> Path:
    """Return the artifact directory after validating one path plan."""
    expected_names = {
        "artifact_manifest_path": MANIFEST_FILENAME,
        "decoding_summary_path": METRICS_FILENAME,
        "unit_eligibility_path": ELIGIBILITY_FILENAME,
        "selected_units_path": UNIT_FILENAME,
        "binned_error_path": BINNED_FILENAME,
    }
    expected_keys = {"artifact_dir", *expected_names}
    if set(paths) != expected_keys:
        raise ValueError("Decoding artifact paths do not use the canonical keys.")
    artifact_dir = Path(paths["artifact_dir"])
    for key, filename in expected_names.items():
        if Path(paths[key]) != artifact_dir / filename:
            raise ValueError(f"Decoding artifact path {key!r} is inconsistent.")
    return artifact_dir


def write_decoding_artifact_bundle(
    result: Mapping[str, Any],
    artifact_paths: Mapping[str, Path],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one canonical bundle using an explicit preplanned path mapping."""
    artifact_dir = _validate_artifact_paths(artifact_paths)
    return write_decoding_comparison_artifact(
        result,
        artifact_dir,
        overwrite=overwrite,
    )


def _load_npz(path: Path) -> Any:
    """Load one pynapple NPZ and require a Tsd-like result."""
    import pynapple as nap

    loaded = nap.load_file(path)
    if not hasattr(loaded, "t") or not hasattr(loaded, "d"):
        raise ValueError(f"Expected a Tsd artifact at {path}.")
    return loaded


def load_decoding_comparison_artifact(
    path: Path,
    *,
    _allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Load, checksum, and validate one decoding artifact directory."""
    directory = Path(path)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Decoding manifest not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    _validate_exact_columns(manifest, MANIFEST_COLUMNS, name="manifest")
    if manifest.empty or manifest["artifact_key"].duplicated().any() or manifest[
        "relative_path"
    ].duplicated().any():
        raise ValueError("Manifest keys and paths must be unique and non-empty.")
    for _, row in manifest.iterrows():
        relative = Path(str(row["relative_path"]))
        if relative.name != str(row["relative_path"]):
            raise ValueError("Manifest paths must be direct child filenames.")
        file_path = directory / relative
        if not file_path.is_file():
            raise FileNotFoundError(f"Manifest artifact not found: {file_path}")
        if file_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(file_path) != str(row["sha256"])
        ):
            raise ValueError(f"Manifest checksum mismatch for {file_path}.")
    first = manifest.iloc[0]
    metadata = {
        name: str(first[name])
        for name in (
            "path_progression_decoding_comparison_id",
            "animal_name",
            "date",
            "region",
            "epoch",
            "cohort_epoch",
            "parameter_name",
            "parameter_sha256",
            "eligibility_rule_sha256",
            "transfer_spec_sha256",
            "decoding_output_rule_sha256",
        )
    }
    result_id = _uuid_string(
        metadata["path_progression_decoding_comparison_id"],
        name="path_progression_decoding_comparison_id",
    )
    if not _allow_temporary_name and directory.name != result_id:
        raise ValueError("Manifest UUID does not match its artifact directory.")
    parameters = validate_decoding_parameters(
        decoding_bin_size_s=float(first["decoding_bin_size_s"]),
        sliding_window_size_bins=int(first["sliding_window_size_bins"]),
        spatial_bin_size_cm=float(first["spatial_bin_size_cm"]),
        error_mode=str(first["error_mode"]),
        error_summary=str(first["error_summary"]),
        min_bin_count=int(first["min_bin_count"]),
    )
    parameters["minimum_movement_firing_rate_hz"] = float(
        first["minimum_movement_firing_rate_hz"]
    )
    minimum_stability = first["minimum_stability_correlation"]
    parameters["minimum_stability_correlation"] = (
        None if pd.isna(minimum_stability) else float(minimum_stability)
    )
    for name, expected in {**metadata, **parameters}.items():
        if name == "minimum_stability_correlation" and expected is None:
            if not manifest[name].isna().all():
                raise ValueError(
                    "Manifest has inconsistent minimum_stability_correlation."
                )
            continue
        if not np.all(manifest[name].astype(str) == str(expected)):
            raise ValueError(f"Manifest has inconsistent {name!r} values.")
    for name in (
        "n_units",
        "n_units_input",
        "n_units_eligible",
        "n_transfer_pairs_expected",
        "n_transfer_pairs_valid",
        "n_decoded_samples",
    ):
        if not np.all(manifest[name] == int(first[name])):
            raise ValueError(f"Manifest has inconsistent {name!r} values.")
    for name in ("analysis_status", "selected_units_sha256"):
        if not np.all(manifest[name].astype(str) == str(first[name])):
            raise ValueError(f"Manifest has inconsistent {name!r} values.")

    def _table(artifact_key: str) -> pd.DataFrame:
        rows = manifest[manifest["artifact_key"] == artifact_key]
        if len(rows) != 1 or rows.iloc[0]["artifact_kind"] != "parquet":
            raise ValueError(f"Manifest is missing table {artifact_key!r}.")
        return pd.read_parquet(directory / str(rows.iloc[0]["relative_path"]))

    eligibility = _table("unit_eligibility")
    selected_units = _table("selected_units")
    metrics = _table("decoding_summary")
    binned_error = _table("cross_path_binned_error")
    valid_rows = metrics[metrics["qc_status"].astype(str) == "valid"]
    valid_keys = tuple(
        zip(
            valid_rows["transfer_family"].astype(str),
            valid_rows["source_trajectory"].astype(str),
            valid_rows["target_trajectory"].astype(str),
            strict=True,
        )
    )
    expected_keys = {
        "unit_eligibility",
        "selected_units",
        "decoding_summary",
        "cross_path_binned_error",
        *{
            f"cross:{key[0]}:{key[1]}:{key[2]}:{role}"
            for key in valid_keys
            for role in ("true", "decoded")
        },
    }
    if set(manifest["artifact_key"].astype(str)) != expected_keys:
        raise ValueError("Manifest does not contain the exact canonical file set.")
    canonical_filenames = {
        "unit_eligibility": ELIGIBILITY_FILENAME,
        "selected_units": UNIT_FILENAME,
        "decoding_summary": METRICS_FILENAME,
        "cross_path_binned_error": BINNED_FILENAME,
        **{
            f"cross:{key[0]}:{key[1]}:{key[2]}:{role}": _npz_filename(
                key, role
            )
            for key in valid_keys
            for role in ("true", "decoded")
        },
    }
    for _, row in manifest.iterrows():
        if str(row["relative_path"]) != canonical_filenames[
            str(row["artifact_key"])
        ]:
            raise ValueError("Manifest artifact filenames are not canonical.")
    outputs = {}
    for key in valid_keys:
        outputs[key] = {}
        for role in ("true", "decoded"):
            artifact_key = f"cross:{key[0]}:{key[1]}:{key[2]}:{role}"
            rows = manifest[manifest["artifact_key"] == artifact_key]
            if len(rows) != 1 or rows.iloc[0]["artifact_kind"] != "pynapple_npz":
                raise ValueError(f"Manifest is missing output {artifact_key!r}.")
            outputs[key][role] = _load_npz(
                directory / str(rows.iloc[0]["relative_path"])
            )
    result = {
        "metadata": metadata,
        "parameters": parameters,
        "unit_eligibility": eligibility,
        "selected_units": selected_units,
        "cross_path_outputs": outputs,
        "cross_path_metrics": metrics,
        "cross_path_binned_error": binned_error,
        **{
            name: int(first[name])
            for name in (
                "n_units_input",
                "n_units_eligible",
                "n_transfer_pairs_expected",
                "n_transfer_pairs_valid",
                "n_decoded_samples",
            )
        },
        "analysis_status": str(first["analysis_status"]),
        "eligible_units_sha256": str(first["selected_units_sha256"]),
    }
    result = validate_decoding_comparison_result(result)
    digest = _selected_units_sha256(result["selected_units"])
    if not np.all(manifest["n_units"] == len(result["selected_units"])) or not np.all(
        manifest["selected_units_sha256"].astype(str) == digest
    ):
        raise ValueError("Manifest selected-unit snapshot is inconsistent.")
    result["manifest"] = manifest
    result["path"] = directory
    return result


def load_decoding_artifact_bundle(manifest_path: Path) -> dict[str, Any]:
    """Load one canonical bundle from the manifest path stored by DataJoint."""
    path = Path(manifest_path)
    if path.name != MANIFEST_FILENAME:
        raise ValueError(
            f"Expected a {MANIFEST_FILENAME!r} path, got {path.name!r}."
        )
    return load_decoding_comparison_artifact(path.parent)


def summarize_decoding_artifact_bundle(
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Return database-facing scalar metadata from a validated bundle."""
    result_keys = {
        "metadata",
        "parameters",
        "unit_eligibility",
        "selected_units",
        "cross_path_outputs",
        "cross_path_metrics",
        "cross_path_binned_error",
        "n_units_input",
        "n_units_eligible",
        "n_transfer_pairs_expected",
        "n_transfer_pairs_valid",
        "n_decoded_samples",
        "analysis_status",
        "eligible_units_sha256",
    }
    missing = sorted(result_keys.difference(bundle))
    if missing:
        raise ValueError(f"Decoding bundle is missing result fields {missing!r}.")
    result = validate_decoding_comparison_result(
        {name: bundle[name] for name in result_keys}
    )
    return {
        **result["metadata"],
        **{
            name: result[name]
            for name in (
                "n_units_input",
                "n_units_eligible",
                "n_transfer_pairs_expected",
                "n_transfer_pairs_valid",
                "n_decoded_samples",
                "analysis_status",
                "eligible_units_sha256",
            )
        },
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "CROSS_PATH_TRANSFER_SPECS",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_DECODING_BIN_SIZE_S",
    "DEFAULT_SLIDING_WINDOW_SIZE_BINS",
    "DECODING_OUTPUT_RULE",
    "ELIGIBILITY_COLUMNS",
    "ELIGIBILITY_FILENAME",
    "ELIGIBILITY_RULE",
    "EXPECTED_TRANSFER_PAIR_COUNT",
    "MANIFEST_COLUMNS",
    "MANIFEST_FILENAME",
    "MANUSCRIPT_PARAMETERS",
    "METRIC_COLUMNS",
    "TRANSFER_QC_STATUSES",
    "TRANSFER_PAIR_SPECS",
    "TRANSFER_SPEC_SHA256",
    "build_cross_path_transfer_specs",
    "build_symmetric_cohort_eligibility_table",
    "compute_path_progression_decoding_comparison",
    "get_decoding_artifact_paths",
    "get_decoding_comparison_artifact_path",
    "get_shared_eligible_stable_unit_ids",
    "load_decoding_artifact_bundle",
    "load_decoding_comparison_artifact",
    "summarize_decoding_artifact_bundle",
    "validate_decoding_comparison_result",
    "validate_decoding_parameters",
    "write_decoding_artifact_bundle",
    "write_decoding_comparison_artifact",
]
