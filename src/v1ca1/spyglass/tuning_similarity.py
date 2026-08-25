"""Path-specific tuning-similarity computation and Parquet/NWB adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.spyglass import movement
from v1ca1.spyglass.path_specific_place import (
    PATH_FRACTION_COORDINATE,
    POSITION_DIM,
    validate_path_specific_place_tuning_curve,
)
from v1ca1.task_progression.similarity import (
    DIRECT_COMPARISON_LABELS,
    DIRECT_COMPARISON_SPECS,
    SIMILARITY_METRICS,
    compute_similarity_score_with_qc,
    flip_curve_if_requested,
)


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "path_specific_place_tuning_similarity"
ARTIFACT_FILENAME = "similarity.parquet"
NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_TUNING_SIMILARITY_TABLE_NAME = "path_specific_place_tuning_similarity"
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
SIMILARITY_STATUSES = (
    "valid",
    "no_finite_bins_both_trajectories",
    "no_finite_bins_trajectory_a",
    "no_finite_bins_trajectory_b",
    "nonfinite_similarity",
)
ANALYSIS_STATUSES = (
    "no_units",
    "valid",
    "no_valid_comparisons",
)
REQUIRED_TRAJECTORIES = tuple(
    dict.fromkeys(
        trajectory
        for spec in DIRECT_COMPARISON_SPECS
        for trajectory in (spec["trajectory_a"], spec["trajectory_b"])
    )
)
TABLE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "animal_name",
    "date",
    "region",
    "epoch",
    "similarity_metric",
    "comparison_family",
    "comparison_label",
    "side",
    "trajectory_a",
    "trajectory_b",
    "flip_trajectory_b",
    "movement_firing_rate_hz",
    "similarity",
    "n_trajectory_a_finite_bins",
    "n_trajectory_b_finite_bins",
    "n_paired_finite_bins",
    "similarity_status",
)
_TEXT_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
    "animal_name",
    "date",
    "region",
    "epoch",
    "similarity_metric",
    "comparison_family",
    "comparison_label",
    "side",
    "trajectory_a",
    "trajectory_b",
    "similarity_status",
)
_INTEGER_COLUMNS = (
    "n_trajectory_a_finite_bins",
    "n_trajectory_b_finite_bins",
    "n_paired_finite_bins",
)
_FLOAT_COLUMNS = (
    "movement_firing_rate_hz",
    "similarity",
)
_COLUMN_DESCRIPTIONS = {
    "spikesorting_merge_id": "Persistent Spyglass spike-sorting merge identifier.",
    "unit_id": "Unit identifier within the spike-sorting merge.",
    "stable_unit_id": "Composite persistent unit identifier, merge_id:unit_id.",
    "group_unit_id": "Unit key in the upstream path-specific tuning curves.",
    "animal_name": "Subject identifier used by the analysis.",
    "date": "Session date formatted as YYYYMMDD.",
    "region": "Canonical analyzed brain region.",
    "epoch": "Selected epoch name.",
    "similarity_metric": "Selected correlation or overlap metric.",
    "comparison_family": "Direct path-comparison family.",
    "comparison_label": "Canonical direct path-comparison label.",
    "side": "Left or right comparison side.",
    "trajectory_a": "First physical path in the comparison.",
    "trajectory_b": "Second physical path in the comparison.",
    "flip_trajectory_b": "Whether the second curve was reversed before comparison.",
    "movement_firing_rate_hz": "Whole-epoch movement firing rate in hertz.",
    "similarity": "Path-specific tuning-curve similarity score.",
    "n_trajectory_a_finite_bins": "Finite bins in the first tuning curve.",
    "n_trajectory_b_finite_bins": "Finite bins in the second tuning curve.",
    "n_paired_finite_bins": "Bins finite in both compared tuning curves.",
    "similarity_status": "Similarity-score QC status.",
}
LEGACY_COLUMNS = (
    "unit",
    "region",
    "epoch",
    "comparison_family",
    "comparison_label",
    "side",
    "trajectory_a",
    "trajectory_b",
    "flip_trajectory_b",
    "firing_rate_hz",
    "similarity",
    "n_trajectory_a_finite_bins",
    "n_trajectory_b_finite_bins",
    "n_paired_finite_bins",
    "similarity_status",
)
_COMPARISON_ORDER = {
    label: index for index, label in enumerate(DIRECT_COMPARISON_LABELS)
}


def _path_component(value: Any, *, name: str) -> str:
    """Return one non-empty path component without traversal."""
    value = str(value)
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{name} must be one non-empty path component.")
    return value


def _uuid_component(value: Any, *, name: str) -> str:
    """Return one canonical UUID path component."""
    try:
        return str(uuid.UUID(str(value)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UUID, got {value!r}.") from exc


def validate_similarity_metric(similarity_metric: str) -> str:
    """Return one supported fixed similarity metric."""
    metric = str(similarity_metric)
    if metric not in SIMILARITY_METRICS:
        raise ValueError(
            f"similarity_metric must be one of {SIMILARITY_METRICS!r}."
        )
    return metric


def get_tuning_similarity_artifact_path(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    similarity_metric: str,
    path_specific_place_tuning_similarity_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return one UUID-keyed, session-first similarity Parquet path."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "region": region,
        }.items()
    }
    metric = validate_similarity_metric(similarity_metric)
    similarity_id = _uuid_component(
        path_specific_place_tuning_similarity_id,
        name="path_specific_place_tuning_similarity_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["region"]
        / metric
        / similarity_id
        / ARTIFACT_FILENAME
    )


def empty_tuning_similarity_table() -> pd.DataFrame:
    """Return an empty all-unit similarity table with its canonical schema."""
    return pd.DataFrame(
        {
            "spikesorting_merge_id": pd.Series(dtype=str),
            "unit_id": pd.Series(dtype=str),
            "stable_unit_id": pd.Series(dtype=str),
            "group_unit_id": pd.Series(dtype=str),
            "animal_name": pd.Series(dtype=str),
            "date": pd.Series(dtype=str),
            "region": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "similarity_metric": pd.Series(dtype=str),
            "comparison_family": pd.Series(dtype=str),
            "comparison_label": pd.Series(dtype=str),
            "side": pd.Series(dtype=str),
            "trajectory_a": pd.Series(dtype=str),
            "trajectory_b": pd.Series(dtype=str),
            "flip_trajectory_b": pd.Series(dtype=bool),
            "movement_firing_rate_hz": pd.Series(dtype=float),
            "similarity": pd.Series(dtype=float),
            "n_trajectory_a_finite_bins": pd.Series(dtype=np.int64),
            "n_trajectory_b_finite_bins": pd.Series(dtype=np.int64),
            "n_paired_finite_bins": pd.Series(dtype=np.int64),
            "similarity_status": pd.Series(dtype=str),
        }
    ).loc[:, list(TABLE_COLUMNS)]


def _curve_identity_table(curve: Any) -> pd.DataFrame:
    """Return ordered persistent identities from one canonical tuning curve."""
    validate_path_specific_place_tuning_curve(curve)
    return pd.DataFrame(
        {
            name: np.asarray(curve.coords[name].values).astype(str)
            for name in IDENTITY_COLUMNS
        }
    )


def _validate_tuning_curves(
    tuning_curves_by_trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return four matching all-trial tuning curves and ordered identities."""
    if not isinstance(tuning_curves_by_trajectory, Mapping):
        raise TypeError("tuning_curves_by_trajectory must be a mapping.")
    actual = set(tuning_curves_by_trajectory)
    expected = set(REQUIRED_TRAJECTORIES)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            "Tuning similarity requires exactly the four path-specific curves; "
            f"missing={missing!r}, extra={extra!r}."
        )

    curves = {
        trajectory: tuning_curves_by_trajectory[trajectory]
        for trajectory in REQUIRED_TRAJECTORIES
    }
    reference_trajectory = REQUIRED_TRAJECTORIES[0]
    reference = curves[reference_trajectory]
    reference_identity = _curve_identity_table(reference)
    if str(reference.attrs["trial_subset"]) != "all":
        raise ValueError("Tuning similarity requires all-trial tuning curves.")

    shared_attributes = (
        "animal_name",
        "date",
        "region",
        "epoch",
        "trial_subset",
        "binning_mode",
        "bin_size_cm",
        "bin_count",
        "sigma_bins",
        "graph_length_cm",
        "bin_edges_cm_json",
        "n_units",
    )
    reference_position = np.asarray(
        reference.coords[POSITION_DIM].values,
        dtype=float,
    )
    reference_fraction = np.asarray(
        reference.coords[PATH_FRACTION_COORDINATE].values,
        dtype=float,
    )
    for trajectory, curve in curves.items():
        identity = _curve_identity_table(curve)
        if str(curve.attrs["trajectory_type"]) != trajectory:
            raise ValueError(
                f"Tuning curve {trajectory!r} has a mismatched trajectory_type."
            )
        if str(curve.attrs["trial_subset"]) != "all":
            raise ValueError("Tuning similarity requires all-trial tuning curves.")
        if not reference_identity.equals(identity):
            raise ValueError(
                "All four tuning curves must contain the same ordered unit "
                f"identities; mismatch for {trajectory!r}."
            )
        position = np.asarray(curve.coords[POSITION_DIM].values, dtype=float)
        fraction = np.asarray(
            curve.coords[PATH_FRACTION_COORDINATE].values,
            dtype=float,
        )
        if not np.array_equal(reference_position, position) or not np.array_equal(
            reference_fraction,
            fraction,
        ):
            raise ValueError(
                "All four tuning curves must use exactly matching centimeter "
                f"and path-fraction bin grids; mismatch for {trajectory!r}."
            )
        for name in shared_attributes:
            reference_value = reference.attrs.get(name)
            value = curve.attrs.get(name)
            if isinstance(reference_value, (int, float, np.number)) and isinstance(
                value,
                (int, float, np.number),
            ):
                matches = bool(
                    np.array_equal(
                        np.asarray(reference_value),
                        np.asarray(value),
                        equal_nan=True,
                    )
                )
            else:
                matches = reference_value == value
            if not matches:
                raise ValueError(
                    f"All four tuning curves must agree on {name!r}; "
                    f"mismatch for {trajectory!r}."
                )
    return curves, reference_identity


def _align_movement_rates(
    movement_firing_rate_table: pd.DataFrame,
    *,
    identity: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> pd.Series:
    """Validate and align upstream movement rates to curve identities."""
    movement.validate_movement_firing_rate_table(movement_firing_rate_table)
    expected_ids = identity["stable_unit_id"].astype(str).tolist()
    if not expected_ids:
        if not movement_firing_rate_table.empty:
            raise ValueError(
                "Movement firing-rate table must be empty when curves have no units."
            )
        return pd.Series(dtype=float, name="movement_firing_rate_hz")
    if movement_firing_rate_table.empty:
        raise ValueError(
            "Movement firing-rate table must contain every tuning-curve unit."
        )
    for name in ("animal_name", "date", "region", "epoch"):
        values = movement_firing_rate_table[name].astype(str).unique().tolist()
        if values != [str(metadata[name])]:
            raise ValueError(
                f"Movement firing-rate table does not match tuning-curve {name}."
            )

    observed = movement_firing_rate_table.copy()
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        observed[name] = observed[name].astype(str)
    observed = observed.set_index("stable_unit_id", drop=False)
    if set(observed.index) != set(expected_ids):
        raise ValueError(
            "Movement firing-rate identities do not exactly match tuning-curve units."
        )
    observed = observed.loc[expected_ids]
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        if observed[name].astype(str).tolist() != identity[name].astype(str).tolist():
            raise ValueError(
                f"Movement firing-rate {name} does not match curve identity."
            )
    rates = pd.to_numeric(
        observed["movement_firing_rate_hz"],
        errors="coerce",
    ).to_numpy(dtype=float)
    return pd.Series(
        rates,
        index=expected_ids,
        dtype=float,
        name="movement_firing_rate_hz",
    )


def compute_tuning_similarity_from_curves(
    *,
    tuning_curves_by_trajectory: Mapping[str, Any],
    movement_firing_rate_table: pd.DataFrame,
    similarity_metric: str,
) -> dict[str, Any]:
    """Compute four direct path comparisons for every selected unit."""
    metric = validate_similarity_metric(similarity_metric)
    curves, identity = _validate_tuning_curves(tuning_curves_by_trajectory)
    reference = curves[REQUIRED_TRAJECTORIES[0]]
    metadata = reference.attrs
    rates = _align_movement_rates(
        movement_firing_rate_table,
        identity=identity,
        metadata=metadata,
    )
    if identity.empty:
        return {
            "table": empty_tuning_similarity_table(),
            "analysis_status": "no_units",
            "n_units": 0,
            "n_valid_comparisons": 0,
            "n_units_with_valid_comparison": 0,
        }

    rows: list[dict[str, Any]] = []
    for unit_index, unit in identity.iterrows():
        stable_unit_id = str(unit["stable_unit_id"])
        for spec in DIRECT_COMPARISON_SPECS:
            trajectory_a = str(spec["trajectory_a"])
            trajectory_b = str(spec["trajectory_b"])
            curve_a = np.asarray(
                curves[trajectory_a].values[unit_index],
                dtype=float,
            )
            curve_b = flip_curve_if_requested(
                curves[trajectory_b].values[unit_index],
                should_flip=bool(spec["flip_trajectory_b"]),
            )
            score = compute_similarity_score_with_qc(
                curve_a,
                curve_b,
                similarity_metric=metric,
            )
            rows.append(
                {
                    **{name: str(unit[name]) for name in IDENTITY_COLUMNS},
                    "animal_name": str(metadata["animal_name"]),
                    "date": str(metadata["date"]),
                    "region": str(metadata["region"]),
                    "epoch": str(metadata["epoch"]),
                    "similarity_metric": metric,
                    "comparison_family": str(spec["comparison_family"]),
                    "comparison_label": str(spec["comparison_label"]),
                    "side": str(spec["side"]),
                    "trajectory_a": trajectory_a,
                    "trajectory_b": trajectory_b,
                    "flip_trajectory_b": bool(spec["flip_trajectory_b"]),
                    "movement_firing_rate_hz": float(rates.loc[stable_unit_id]),
                    **score,
                }
            )
    table = pd.DataFrame.from_records(rows).loc[:, list(TABLE_COLUMNS)]
    validate_tuning_similarity_table(table)
    return {"table": table, **summarize_tuning_similarity_table(table)}


def _single_value(table: pd.DataFrame, column: str) -> Any:
    """Return one value shared by every row of a non-empty table."""
    values = table[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Similarity table column {column!r} must be constant.")
    return values.iloc[0]


def _validate_integer_column(table: pd.DataFrame, column: str) -> np.ndarray:
    """Return one validated non-negative integer column."""
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not np.allclose(values, np.rint(values), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            f"Similarity table column {column!r} must contain non-negative integers."
        )
    return np.rint(values).astype(np.int64)


def validate_tuning_similarity_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and return one canonical all-unit similarity table."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Tuning-similarity artifact must be a pandas DataFrame.")
    missing = sorted(set(TABLE_COLUMNS).difference(table.columns))
    extra = sorted(set(table.columns).difference(TABLE_COLUMNS))
    if missing or extra:
        raise ValueError(
            "Tuning-similarity table must have the exact canonical schema; "
            f"missing={missing!r}, extra={extra!r}."
        )
    if table.empty:
        return table

    metric = validate_similarity_metric(
        str(_single_value(table, "similarity_metric"))
    )
    for column in ("animal_name", "date", "region", "epoch"):
        if not str(_single_value(table, column)):
            raise ValueError(f"Similarity table column {column!r} must be non-empty.")

    identity_rows = table.loc[:, list(IDENTITY_COLUMNS)].astype(str)
    expected_stable_ids = (
        identity_rows["spikesorting_merge_id"]
        + ":"
        + identity_rows["unit_id"]
    )
    if not np.array_equal(
        expected_stable_ids.to_numpy(dtype=str),
        identity_rows["stable_unit_id"].to_numpy(dtype=str),
    ):
        raise ValueError("Similarity-table stable unit identities are inconsistent.")
    units = identity_rows.drop_duplicates("stable_unit_id")
    if units["stable_unit_id"].duplicated().any() or units[
        "group_unit_id"
    ].duplicated().any():
        raise ValueError("Similarity-table unit identities must be one-to-one.")
    for stable_unit_id, unit_rows in table.groupby("stable_unit_id", sort=False):
        if len(unit_rows) != len(DIRECT_COMPARISON_SPECS):
            raise ValueError(
                f"Unit {stable_unit_id!r} must have exactly four comparisons."
            )
        for column in IDENTITY_COLUMNS:
            if unit_rows[column].astype(str).nunique() != 1:
                raise ValueError(
                    f"Unit {stable_unit_id!r} has inconsistent {column}."
                )
        labels = unit_rows["comparison_label"].astype(str).tolist()
        if set(labels) != set(DIRECT_COMPARISON_LABELS) or len(set(labels)) != len(
            labels
        ):
            raise ValueError(
                f"Unit {stable_unit_id!r} must contain each direct comparison once."
            )
        rates = pd.to_numeric(
            unit_rows["movement_firing_rate_hz"],
            errors="coerce",
        ).to_numpy(dtype=float)
        if not np.allclose(rates, rates[0], rtol=0.0, atol=0.0, equal_nan=True):
            raise ValueError(
                f"Unit {stable_unit_id!r} has inconsistent movement firing rates."
            )

    rates = pd.to_numeric(
        table["movement_firing_rate_hz"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if np.any(np.isinf(rates)) or np.any(rates[np.isfinite(rates)] < 0.0):
        raise ValueError("Movement firing rates must be non-negative or NaN.")

    count_a = _validate_integer_column(table, "n_trajectory_a_finite_bins")
    count_b = _validate_integer_column(table, "n_trajectory_b_finite_bins")
    paired = _validate_integer_column(table, "n_paired_finite_bins")
    if np.any(paired > np.minimum(count_a, count_b)):
        raise ValueError("Paired finite-bin counts exceed trajectory support.")

    scores = pd.to_numeric(table["similarity"], errors="coerce").to_numpy(
        dtype=float
    )
    if np.any(np.isinf(scores)):
        raise ValueError("Similarity values may be finite or NaN, not infinite.")
    valid = table["similarity_status"].astype(str).eq("valid").to_numpy()
    if not np.array_equal(np.isfinite(scores), valid):
        raise ValueError(
            "Finite similarity values must correspond exactly to valid status."
        )
    finite_scores = scores[valid]
    lower = -1.0 if metric == "correlation" else 0.0
    if np.any(finite_scores < lower - 1e-12) or np.any(
        finite_scores > 1.0 + 1e-12
    ):
        raise ValueError(f"Similarity values are outside the range for {metric!r}.")

    for row_index, row in table.reset_index(drop=True).iterrows():
        label = str(row["comparison_label"])
        spec = next(
            (
                candidate
                for candidate in DIRECT_COMPARISON_SPECS
                if candidate["comparison_label"] == label
            ),
            None,
        )
        if spec is None:
            raise ValueError(f"Unsupported direct comparison label {label!r}.")
        for column in (
            "comparison_family",
            "side",
            "trajectory_a",
            "trajectory_b",
        ):
            if str(row[column]) != str(spec[column]):
                raise ValueError(
                    f"Comparison {label!r} has a mismatched {column}."
                )
        flip = row["flip_trajectory_b"]
        if not isinstance(flip, (bool, np.bool_)) or bool(flip) != bool(
            spec["flip_trajectory_b"]
        ):
            raise ValueError(
                f"Comparison {label!r} has a mismatched flip_trajectory_b."
            )
        status = str(row["similarity_status"])
        if status not in SIMILARITY_STATUSES:
            raise ValueError(f"Unsupported similarity_status {status!r}.")
        n_a = count_a[row_index]
        n_b = count_b[row_index]
        if status == "no_finite_bins_both_trajectories" and not (
            n_a == 0 and n_b == 0
        ):
            raise ValueError("Both-trajectory no-finite-bin QC is inconsistent.")
        if status == "no_finite_bins_trajectory_a" and not (
            n_a == 0 and n_b > 0
        ):
            raise ValueError("Trajectory-a no-finite-bin QC is inconsistent.")
        if status == "no_finite_bins_trajectory_b" and not (
            n_a > 0 and n_b == 0
        ):
            raise ValueError("Trajectory-b no-finite-bin QC is inconsistent.")
        if status == "nonfinite_similarity" and not (n_a > 0 and n_b > 0):
            raise ValueError("Nonfinite-similarity QC requires both trajectories.")
    return table


def _decode_text(value: Any, *, column: str) -> str:
    """Return one NWB-loaded scalar as UTF-8 text."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Tuning-similarity column {column!r} contains invalid UTF-8."
            ) from exc
    return str(value)


def _canonical_nwb_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return one validated similarity table with deterministic NWB dtypes."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Tuning-similarity artifact must be a pandas DataFrame.")
    missing = sorted(set(TABLE_COLUMNS).difference(table.columns))
    extra = sorted(set(table.columns).difference(TABLE_COLUMNS))
    if missing or extra:
        raise ValueError(
            "Tuning-similarity table must have the exact canonical schema; "
            f"missing={missing!r}, extra={extra!r}."
        )
    if table.empty:
        return empty_tuning_similarity_table()

    output = table.loc[:, list(TABLE_COLUMNS)].copy().reset_index(drop=True)
    for column in _TEXT_COLUMNS:
        output[column] = output[column].map(
            lambda value, column=column: _decode_text(value, column=column)
        )
        if output[column].eq("").any():
            raise ValueError(
                f"Tuning-similarity column {column!r} cannot be empty."
            )
    flips = output["flip_trajectory_b"].tolist()
    if not all(isinstance(value, (bool, np.bool_)) for value in flips):
        raise TypeError("flip_trajectory_b must contain boolean values.")
    output["flip_trajectory_b"] = np.asarray(flips, dtype=bool)
    for column in _INTEGER_COLUMNS:
        output[column] = _validate_integer_column(output, column)
    for column in _FLOAT_COLUMNS:
        output[column] = pd.to_numeric(
            output[column],
            errors="raise",
        ).astype(float)
    validate_tuning_similarity_table(output)
    return output


def tuning_similarity_table_to_dynamic_table(table: pd.DataFrame) -> Any:
    """Convert one canonical tuning-similarity table to an NWB DynamicTable."""
    from hdmf.common import DynamicTable, VectorData

    canonical = _canonical_nwb_table(table)
    description = (
        "All-unit direct path-specific tuning comparisons and QC; "
        f"v1ca1 schema version {NWB_ARTIFACT_SCHEMA_VERSION}."
    )
    if canonical.empty:
        columns = []
        for name in TABLE_COLUMNS:
            if name in _TEXT_COLUMNS:
                data = np.asarray([], dtype="S1")
            elif name == "flip_trajectory_b":
                data = np.asarray([], dtype=bool)
            elif name in _INTEGER_COLUMNS:
                data = np.asarray([], dtype=np.int64)
            else:
                data = np.asarray([], dtype=float)
            columns.append(
                VectorData(
                    name=name,
                    description=_COLUMN_DESCRIPTIONS[name],
                    data=data,
                )
            )
        return DynamicTable(
            name=NWB_TUNING_SIMILARITY_TABLE_NAME,
            description=description,
            columns=columns,
        )

    return DynamicTable.from_dataframe(
        name=NWB_TUNING_SIMILARITY_TABLE_NAME,
        df=canonical,
        table_description=description,
        columns=[
            {
                "name": name,
                "description": _COLUMN_DESCRIPTIONS[name],
            }
            for name in TABLE_COLUMNS
        ],
    )


def tuning_similarity_table_from_dynamic_table(nwb_table: Any) -> pd.DataFrame:
    """Return a canonical DataFrame from a DynamicTable or fetched DataFrame."""
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != NWB_TUNING_SIMILARITY_TABLE_NAME:
            raise ValueError(
                "Unexpected tuning-similarity NWB object name "
                f"{nwb_table.name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError(
            "Tuning-similarity NWB object must be a DynamicTable or DataFrame."
        )
    return _canonical_nwb_table(table.reset_index(drop=True))


def tuning_similarity_table_sha256(table: pd.DataFrame) -> str:
    """Digest the complete canonical similarity table independent of storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    canonical = _canonical_nwb_table(table)
    records = []
    for record in canonical.to_dict("records"):
        normalized = {}
        for column in TABLE_COLUMNS:
            value = record[column]
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and np.isnan(value):
                value = None
            normalized[column] = value
        records.append(normalized)
    return provenance_sha256(
        {
            "columns": list(TABLE_COLUMNS),
            "records": records,
        }
    )


def summarize_tuning_similarity_table(table: pd.DataFrame) -> dict[str, Any]:
    """Return result-level counts and status for one validated table."""
    validate_tuning_similarity_table(table)
    if table.empty:
        return {
            "analysis_status": "no_units",
            "n_units": 0,
            "n_valid_comparisons": 0,
            "n_units_with_valid_comparison": 0,
        }
    valid = table["similarity_status"].astype(str).eq("valid")
    n_valid = int(valid.sum())
    n_units_with_valid = int(table.loc[valid, "stable_unit_id"].nunique())
    return {
        "analysis_status": "valid" if n_valid else "no_valid_comparisons",
        "n_units": int(table["stable_unit_id"].nunique()),
        "n_valid_comparisons": n_valid,
        "n_units_with_valid_comparison": n_units_with_valid,
    }


def load_tuning_similarity_artifact(path: Path) -> pd.DataFrame:
    """Load and validate one canonical tuning-similarity Parquet."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tuning-similarity artifact not found: {path}")
    table = pd.read_parquet(path)
    return validate_tuning_similarity_table(table)


def write_tuning_similarity_artifact(
    table: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated Parquet without implicit overwrite."""
    validate_tuning_similarity_table(table)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite tuning-similarity artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.backup")
    had_existing = path.exists()
    try:
        table.to_parquet(temporary, index=False)
        load_tuning_similarity_artifact(temporary)
        if had_existing:
            os.replace(path, backup)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        if backup.exists():
            path.unlink(missing_ok=True)
            os.replace(backup, path)
        raise
    else:
        backup.unlink(missing_ok=True)
    return path


def _resolve_legacy_identity(
    legacy_unit_id: Any,
    resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
) -> dict[str, str] | None:
    """Resolve one selected legacy unit id, or return ``None`` if unselected."""
    if isinstance(resolver, Mapping):
        matches = [
            value
            for key, value in resolver.items()
            if str(key) == str(legacy_unit_id)
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError(
                f"Legacy unit {legacy_unit_id!r} must resolve exactly once."
            )
        identity = matches[0]
    elif callable(resolver):
        try:
            identity = resolver(legacy_unit_id)
        except LookupError:
            return None
    else:
        raise TypeError("unit_identity_resolver must be a mapping or callable.")
    if not isinstance(identity, Mapping):
        raise TypeError("Each resolved legacy unit identity must be a mapping.")
    missing = [
        name
        for name in ("spikesorting_merge_id", "unit_id")
        if name not in identity
    ]
    if missing:
        raise ValueError(f"Resolved legacy identity is missing fields {missing!r}.")
    merge_id = str(identity["spikesorting_merge_id"])
    unit_id = str(identity["unit_id"])
    if not merge_id or not unit_id:
        raise ValueError("Resolved persistent unit identity must be non-empty.")
    return {
        "spikesorting_merge_id": merge_id,
        "unit_id": unit_id,
        "stable_unit_id": f"{merge_id}:{unit_id}",
    }


def _tables_match(
    normalized: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    """Require one normalized legacy table to equal recomputed canonical rows."""
    key = ["stable_unit_id", "comparison_label"]
    normalized = normalized.sort_values(key, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(key, kind="stable").reset_index(drop=True)
    if len(normalized) != len(expected):
        raise ValueError("Legacy similarity row count does not match selected curves.")
    string_columns = [
        column
        for column in TABLE_COLUMNS
        if column
        not in {
            "flip_trajectory_b",
            "movement_firing_rate_hz",
            "similarity",
            "n_trajectory_a_finite_bins",
            "n_trajectory_b_finite_bins",
            "n_paired_finite_bins",
        }
    ]
    for column in string_columns:
        if not np.array_equal(
            normalized[column].astype(str).to_numpy(),
            expected[column].astype(str).to_numpy(),
        ):
            raise ValueError(
                f"Legacy similarity column {column!r} does not match selected curves."
            )
    if not np.array_equal(
        normalized["flip_trajectory_b"].to_numpy(dtype=bool),
        expected["flip_trajectory_b"].to_numpy(dtype=bool),
    ):
        raise ValueError("Legacy comparison reversal does not match selected curves.")
    for column in (
        "movement_firing_rate_hz",
        "similarity",
    ):
        if not np.allclose(
            pd.to_numeric(normalized[column], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Legacy similarity column {column!r} does not match selected curves."
            )
    for column in (
        "n_trajectory_a_finite_bins",
        "n_trajectory_b_finite_bins",
        "n_paired_finite_bins",
    ):
        if not np.array_equal(
            pd.to_numeric(normalized[column], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(expected[column], errors="coerce").to_numpy(dtype=float),
        ):
            raise ValueError(
                f"Legacy similarity column {column!r} does not match selected curves."
            )


def validate_tuning_similarity_against_inputs(
    table: pd.DataFrame,
    *,
    tuning_curves_by_trajectory: Mapping[str, Any],
    movement_firing_rate_table: pd.DataFrame,
    similarity_metric: str,
) -> pd.DataFrame:
    """Require one artifact to equal scores recomputed from its upstream data."""
    validate_tuning_similarity_table(table)
    expected = compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=tuning_curves_by_trajectory,
        movement_firing_rate_table=movement_firing_rate_table,
        similarity_metric=similarity_metric,
    )["table"]
    try:
        _tables_match(table, expected)
    except ValueError as exc:
        raise ValueError(
            "Tuning-similarity artifact does not match its selected upstream "
            "tuning curves and movement firing rates."
        ) from exc
    return table


def normalize_legacy_all_units_similarity_table(
    legacy_table: pd.DataFrame,
    *,
    tuning_curves_by_trajectory: Mapping[str, Any],
    movement_firing_rate_table: pd.DataFrame,
    similarity_metric: str,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
) -> pd.DataFrame:
    """Validate and normalize one direct-only legacy all-unit similarity table."""
    if not isinstance(legacy_table, pd.DataFrame):
        raise TypeError("Legacy similarity artifact must be a pandas DataFrame.")
    missing = sorted(set(LEGACY_COLUMNS).difference(legacy_table.columns))
    if missing:
        raise ValueError(f"Legacy all-unit similarity is missing columns {missing!r}.")

    expected_result = compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=tuning_curves_by_trajectory,
        movement_firing_rate_table=movement_firing_rate_table,
        similarity_metric=similarity_metric,
    )
    expected = expected_result["table"]
    if legacy_table.empty and not expected.empty:
        raise ValueError("Legacy similarity is empty for a non-empty unit selection.")

    expected_identity = (
        expected.loc[:, list(IDENTITY_COLUMNS)]
        .drop_duplicates("stable_unit_id")
        .set_index("stable_unit_id", drop=False)
    )
    resolved: dict[str, dict[str, str]] = {}
    for legacy_unit_id in legacy_table["unit"].drop_duplicates().tolist():
        identity = _resolve_legacy_identity(legacy_unit_id, unit_identity_resolver)
        if identity is None:
            continue
        stable_id = identity["stable_unit_id"]
        if stable_id in resolved:
            raise ValueError("Legacy units must resolve to unique persistent identities.")
        resolved[stable_id] = {**identity, "legacy_unit_id": legacy_unit_id}
    if set(resolved) != set(expected_identity.index.astype(str)):
        raise ValueError(
            "Resolved legacy identities do not exactly match selected tuning units."
        )
    if expected.empty:
        return expected
    legacy_to_stable = {
        str(item["legacy_unit_id"]): stable_id
        for stable_id, item in resolved.items()
    }

    rows: list[dict[str, Any]] = []
    metadata = expected.iloc[0]
    expected_rates = (
        expected.loc[:, ["stable_unit_id", "movement_firing_rate_hz"]]
        .drop_duplicates("stable_unit_id")
        .set_index("stable_unit_id")["movement_firing_rate_hz"]
    )
    for legacy_row in legacy_table.loc[:, list(LEGACY_COLUMNS)].to_dict("records"):
        stable_id = legacy_to_stable.get(str(legacy_row["unit"]))
        if stable_id is None:
            continue
        identity = expected_identity.loc[stable_id]
        legacy_rate = float(legacy_row["firing_rate_hz"])
        expected_rate = float(expected_rates.loc[stable_id])
        if not np.isclose(
            legacy_rate,
            expected_rate,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                "Legacy firing_rate_hz does not match upstream MovementFiringRate."
            )
        rows.append(
            {
                **{name: str(identity[name]) for name in IDENTITY_COLUMNS},
                "animal_name": str(metadata["animal_name"]),
                "date": str(metadata["date"]),
                "region": legacy_row["region"],
                "epoch": legacy_row["epoch"],
                "similarity_metric": validate_similarity_metric(similarity_metric),
                "comparison_family": legacy_row["comparison_family"],
                "comparison_label": legacy_row["comparison_label"],
                "side": legacy_row["side"],
                "trajectory_a": legacy_row["trajectory_a"],
                "trajectory_b": legacy_row["trajectory_b"],
                "flip_trajectory_b": legacy_row["flip_trajectory_b"],
                "movement_firing_rate_hz": expected_rate,
                "similarity": legacy_row["similarity"],
                "n_trajectory_a_finite_bins": legacy_row[
                    "n_trajectory_a_finite_bins"
                ],
                "n_trajectory_b_finite_bins": legacy_row[
                    "n_trajectory_b_finite_bins"
                ],
                "n_paired_finite_bins": legacy_row["n_paired_finite_bins"],
                "similarity_status": legacy_row["similarity_status"],
            }
        )
    normalized = pd.DataFrame.from_records(rows).loc[:, list(TABLE_COLUMNS)]
    validate_tuning_similarity_table(normalized)
    _tables_match(normalized, expected)
    order = {
        stable_id: index
        for index, stable_id in enumerate(
            expected["stable_unit_id"].drop_duplicates().astype(str)
        )
    }
    normalized = normalized.assign(
        _unit_order=normalized["stable_unit_id"].astype(str).map(order),
        _comparison_order=normalized["comparison_label"].astype(str).map(
            _COMPARISON_ORDER
        ),
    ).sort_values(["_unit_order", "_comparison_order"], kind="stable")
    return normalized.drop(
        columns=["_unit_order", "_comparison_order"]
    ).reset_index(drop=True)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_existing_tuning_similarity_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    tuning_curves_by_trajectory: Mapping[str, Any],
    movement_firing_rate_table: pd.DataFrame,
    similarity_metric: str,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize and atomically register one legacy ``*_all_units`` Parquet."""
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy tuning-similarity artifact not found: {source}")
    if not source.stem.endswith("_all_units"):
        raise ValueError("Legacy registration requires a *_all_units.parquet source.")
    if source.suffix != ".parquet":
        raise ValueError("Legacy tuning-similarity source must be a Parquet file.")
    if source.resolve() == destination.resolve(strict=False):
        raise ValueError("Legacy source and canonical destination must differ.")

    legacy_table = pd.read_parquet(source)
    table = normalize_legacy_all_units_similarity_table(
        legacy_table,
        tuning_curves_by_trajectory=tuning_curves_by_trajectory,
        movement_firing_rate_table=movement_firing_rate_table,
        similarity_metric=similarity_metric,
        unit_identity_resolver=unit_identity_resolver,
    )
    source_sha256 = _file_sha256(source)
    written = write_tuning_similarity_artifact(
        table,
        destination,
        overwrite=overwrite,
    )
    summary = summarize_tuning_similarity_table(table)
    return {
        "table": table,
        "similarity_path": written,
        **summary,
        "legacy_artifact_provenance": {
            "source_path": str(source.resolve(strict=True)),
            "source_sha256": source_sha256,
            "legacy_unit_column": "unit",
        },
        "_created_artifact_paths": [str(written)],
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "ARTIFACT_FILENAME",
    "DEFAULT_ARTIFACT_ROOT",
    "IDENTITY_COLUMNS",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "NWB_TUNING_SIMILARITY_TABLE_NAME",
    "REQUIRED_TRAJECTORIES",
    "SIMILARITY_METRICS",
    "SIMILARITY_STATUSES",
    "TABLE_COLUMNS",
    "compute_tuning_similarity_from_curves",
    "empty_tuning_similarity_table",
    "get_tuning_similarity_artifact_path",
    "load_tuning_similarity_artifact",
    "normalize_legacy_all_units_similarity_table",
    "register_existing_tuning_similarity_artifact",
    "summarize_tuning_similarity_table",
    "tuning_similarity_table_from_dynamic_table",
    "tuning_similarity_table_sha256",
    "tuning_similarity_table_to_dynamic_table",
    "validate_tuning_similarity_against_inputs",
    "validate_similarity_metric",
    "validate_tuning_similarity_table",
    "write_tuning_similarity_artifact",
]
