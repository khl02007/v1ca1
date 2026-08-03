"""Database-free movement intervals and epoch-wide firing-rate artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import build_movement_interval, build_speed_tsd


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "movement_firing_rate"
FIRING_RATE_FILENAME = "movement_firing_rate.parquet"
INTERVALS_FILENAME = "movement_intervals.npz"
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
MOVEMENT_TABLE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "animal_name",
    "date",
    "region",
    "epoch",
    "movement_spike_count",
    "movement_duration_s",
    "movement_firing_rate_hz",
    "firing_rate_status",
    "position_sample_count",
    "finite_position_sample_count",
    "finite_speed_sample_count",
    "movement_interval_count",
    "speed_threshold_cm_s",
    "speed_smoothing_sigma_s",
)
ANALYSIS_STATUSES = (
    "no_units",
    "no_valid_position",
    "no_movement",
    "valid",
)


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


def get_movement_artifact_dir(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str,
    movement_firing_rate_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return one UUID-keyed, session-first movement artifact directory."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "region": region,
        }.items()
    }
    movement_id = _uuid_component(
        movement_firing_rate_id,
        name="movement_firing_rate_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["region"]
        / movement_id
    )


def get_movement_artifact_paths(**kwargs: Any) -> dict[str, Path]:
    """Return canonical Parquet and Pynapple paths for one selection."""
    artifact_dir = get_movement_artifact_dir(**kwargs)
    return {
        "artifact_dir": artifact_dir,
        "firing_rate_path": artifact_dir / FIRING_RATE_FILENAME,
        "movement_intervals_path": artifact_dir / INTERVALS_FILENAME,
    }


def _validate_parameters(
    speed_threshold_cm_s: float,
    speed_smoothing_sigma_s: float,
) -> tuple[float, float]:
    """Return validated movement parameters as finite floats."""
    threshold = float(speed_threshold_cm_s)
    sigma = float(speed_smoothing_sigma_s)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("speed_threshold_cm_s must be non-negative and finite.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("speed_smoothing_sigma_s must be positive and finite.")
    return threshold, sigma


def _position_arrays(position: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned selected-position values and timestamps without trimming."""
    values = np.asarray(getattr(position, "d", position), dtype=float)
    timestamps = np.asarray(getattr(position, "t", ()), dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Selected position must have shape (n_samples, 2).")
    if timestamps.size != values.shape[0]:
        raise ValueError("Selected position samples and timestamps must align.")
    if not np.all(np.isfinite(timestamps)) or (
        timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("Position timestamps must be finite and strictly increasing.")
    return values, timestamps


def _stable_identity_rows(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return persistent identities aligned to ephemeral TsGroup keys."""
    group_keys = [] if spikes is None else list(spikes.keys())
    identities = [dict(identity) for identity in stable_unit_ids]
    if len(group_keys) != len(identities):
        raise ValueError("TsGroup and stable unit identity lengths must match.")
    try:
        unique_group_key_count = len(set(group_keys))
    except TypeError as exc:
        raise ValueError("Ephemeral TsGroup unit identifiers must be hashable.") from exc
    if unique_group_key_count != len(group_keys):
        raise ValueError("Ephemeral TsGroup unit identifiers must be unique.")

    rows: list[dict[str, Any]] = []
    persistent_ids: set[tuple[str, str]] = set()
    for group_key, identity in zip(group_keys, identities, strict=True):
        missing = {
            field
            for field in ("spikesorting_merge_id", "unit_id")
            if field not in identity
        }
        if missing:
            raise ValueError(
                f"Stable unit identity is missing fields {sorted(missing)!r}."
            )
        merge_id = str(identity["spikesorting_merge_id"])
        unit_id = str(identity["unit_id"])
        if not merge_id or not unit_id:
            raise ValueError("Persistent unit identity fields must be non-empty.")
        persistent_id = (merge_id, unit_id)
        if persistent_id in persistent_ids:
            raise ValueError("Persistent unit identities must be unique.")
        persistent_ids.add(persistent_id)
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": f"{merge_id}:{unit_id}",
                "group_unit_id": group_key,
            }
        )
    return rows


def empty_movement_firing_rate_table() -> pd.DataFrame:
    """Return an empty movement-rate table with its canonical schema."""
    return pd.DataFrame(
        {
            "spikesorting_merge_id": pd.Series(dtype=str),
            "unit_id": pd.Series(dtype=str),
            "stable_unit_id": pd.Series(dtype=str),
            "group_unit_id": pd.Series(dtype=object),
            "animal_name": pd.Series(dtype=str),
            "date": pd.Series(dtype=str),
            "region": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "movement_spike_count": pd.Series(dtype=np.int64),
            "movement_duration_s": pd.Series(dtype=float),
            "movement_firing_rate_hz": pd.Series(dtype=float),
            "firing_rate_status": pd.Series(dtype=str),
            "position_sample_count": pd.Series(dtype=np.int64),
            "finite_position_sample_count": pd.Series(dtype=np.int64),
            "finite_speed_sample_count": pd.Series(dtype=np.int64),
            "movement_interval_count": pd.Series(dtype=np.int64),
            "speed_threshold_cm_s": pd.Series(dtype=float),
            "speed_smoothing_sigma_s": pd.Series(dtype=float),
        }
    ).loc[:, list(MOVEMENT_TABLE_COLUMNS)]


def _empty_intervalset() -> Any:
    """Return a valid empty second-based Pynapple IntervalSet."""
    import pynapple as nap

    return nap.IntervalSet(
        start=np.array([], dtype=float),
        end=np.array([], dtype=float),
        time_units="s",
    )


def movement_interval_summary(movement_intervals: Any) -> tuple[int, float]:
    """Validate one IntervalSet-like object and return count and duration."""
    try:
        starts = np.asarray(movement_intervals.start, dtype=float).reshape(-1)
        ends = np.asarray(movement_intervals.end, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Movement intervals must expose numeric start and end arrays.") from exc
    if starts.shape != ends.shape:
        raise ValueError("Movement interval start and end arrays must align.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Movement interval bounds must be finite.")
    if np.any(ends < starts):
        raise ValueError("Movement interval stop times must not precede start times.")
    if starts.size > 1 and (
        np.any(np.diff(starts) < 0.0) or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError("Movement intervals must be sorted and non-overlapping.")
    duration = float(np.sum(ends - starts))
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("Movement interval duration must be non-negative and finite.")
    if hasattr(movement_intervals, "tot_length"):
        reported_duration = float(movement_intervals.tot_length())
        if not np.isfinite(reported_duration) or not np.isclose(
            reported_duration,
            duration,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Movement interval bounds and reported duration disagree.")
    return int(starts.size), duration


def _movement_spike_counts(spikes: Any, movement_intervals: Any) -> np.ndarray:
    """Return total movement spike counts aligned to TsGroup key order."""
    n_units = len(list(spikes.keys()))
    counts = np.asarray(
        spikes.count(ep=movement_intervals).to_numpy(),
        dtype=float,
    )
    if counts.ndim == 0:
        totals = counts.reshape(1)
    elif counts.ndim == 1:
        if n_units == 1:
            totals = np.asarray([np.sum(counts)], dtype=float)
        elif counts.size == n_units:
            totals = counts
        else:
            raise ValueError("Movement spike counts do not align with TsGroup units.")
    else:
        if counts.shape[-1] != n_units:
            raise ValueError("Movement spike counts do not align with TsGroup units.")
        totals = np.sum(counts, axis=tuple(range(counts.ndim - 1))).reshape(-1)
    if totals.size != n_units:
        raise ValueError("Movement spike counts do not align with TsGroup units.")
    if not np.all(np.isfinite(totals)) or np.any(totals < 0.0) or not np.allclose(
        totals,
        np.rint(totals),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Movement spike counts must be finite non-negative integers.")
    return np.rint(totals).astype(np.int64)


def _build_movement_table(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    movement_spike_counts: np.ndarray,
    movement_duration_s: float,
    movement_firing_rates_hz: np.ndarray,
    firing_rate_status: str,
    position_sample_count: int,
    finite_position_sample_count: int,
    finite_speed_sample_count: int,
    movement_interval_count: int,
    speed_threshold_cm_s: float,
    speed_smoothing_sigma_s: float,
) -> pd.DataFrame:
    """Return one canonical all-unit movement-rate table."""
    rows: list[dict[str, Any]] = []
    for identity, spike_count, firing_rate in zip(
        identity_rows,
        movement_spike_counts,
        movement_firing_rates_hz,
        strict=True,
    ):
        rows.append(
            {
                **dict(identity),
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "epoch": str(epoch),
                "movement_spike_count": int(spike_count),
                "movement_duration_s": float(movement_duration_s),
                "movement_firing_rate_hz": float(firing_rate),
                "firing_rate_status": str(firing_rate_status),
                "position_sample_count": int(position_sample_count),
                "finite_position_sample_count": int(finite_position_sample_count),
                "finite_speed_sample_count": int(finite_speed_sample_count),
                "movement_interval_count": int(movement_interval_count),
                "speed_threshold_cm_s": float(speed_threshold_cm_s),
                "speed_smoothing_sigma_s": float(speed_smoothing_sigma_s),
            }
        )
    table = pd.DataFrame(rows, columns=MOVEMENT_TABLE_COLUMNS)
    validate_movement_firing_rate_table(table)
    return table


def compute_selected_movement_firing_rate(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    speed_threshold_cm_s: float,
    speed_smoothing_sigma_s: float,
) -> dict[str, Any]:
    """Compute all-unit movement rates from one already-offset Position row."""
    threshold, sigma = _validate_parameters(
        speed_threshold_cm_s,
        speed_smoothing_sigma_s,
    )
    identity_rows = _stable_identity_rows(spikes, stable_unit_ids)
    n_units = len(identity_rows)
    if n_units == 0:
        return {
            "table": empty_movement_firing_rate_table(),
            "movement_intervals": _empty_intervalset(),
            "analysis_status": "no_units",
            "n_units": 0,
            "n_valid_units": 0,
            "n_units_with_spikes": 0,
            "position_sample_count": 0,
            "finite_position_sample_count": 0,
            "finite_speed_sample_count": 0,
            "movement_interval_count": 0,
            "movement_duration_s": 0.0,
        }

    position_values, position_times = _position_arrays(position)
    position_sample_count = int(position_values.shape[0])
    finite_position_count = int(
        np.sum(np.all(np.isfinite(position_values), axis=1))
    )
    empty_counts = np.zeros(n_units, dtype=np.int64)
    undefined_rates = np.full(n_units, np.nan, dtype=float)
    if finite_position_count < 2:
        movement_intervals = _empty_intervalset()
        table = _build_movement_table(
            identity_rows=identity_rows,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            movement_spike_counts=empty_counts,
            movement_duration_s=0.0,
            movement_firing_rates_hz=undefined_rates,
            firing_rate_status="no_valid_position",
            position_sample_count=position_sample_count,
            finite_position_sample_count=finite_position_count,
            finite_speed_sample_count=0,
            movement_interval_count=0,
            speed_threshold_cm_s=threshold,
            speed_smoothing_sigma_s=sigma,
        )
        return {
            "table": table,
            "movement_intervals": movement_intervals,
            "analysis_status": "no_valid_position",
            "n_units": n_units,
            "n_valid_units": 0,
            "n_units_with_spikes": 0,
            "position_sample_count": position_sample_count,
            "finite_position_sample_count": finite_position_count,
            "finite_speed_sample_count": 0,
            "movement_interval_count": 0,
            "movement_duration_s": 0.0,
        }

    speed = build_speed_tsd(
        position_values,
        position_times,
        position_offset=0,
        speed_smoothing_sigma_s=sigma,
    )
    speed_values = np.asarray(getattr(speed, "d", speed), dtype=float).reshape(-1)
    if speed_values.size != position_sample_count:
        raise ValueError("Computed speed does not align with selected position samples.")
    finite_speed_count = int(np.sum(np.isfinite(speed_values)))
    if finite_speed_count < 2:
        movement_intervals = _empty_intervalset()
        table = _build_movement_table(
            identity_rows=identity_rows,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            movement_spike_counts=empty_counts,
            movement_duration_s=0.0,
            movement_firing_rates_hz=undefined_rates,
            firing_rate_status="no_valid_position",
            position_sample_count=position_sample_count,
            finite_position_sample_count=finite_position_count,
            finite_speed_sample_count=finite_speed_count,
            movement_interval_count=0,
            speed_threshold_cm_s=threshold,
            speed_smoothing_sigma_s=sigma,
        )
        return {
            "table": table,
            "movement_intervals": movement_intervals,
            "analysis_status": "no_valid_position",
            "n_units": n_units,
            "n_valid_units": 0,
            "n_units_with_spikes": 0,
            "position_sample_count": position_sample_count,
            "finite_position_sample_count": finite_position_count,
            "finite_speed_sample_count": finite_speed_count,
            "movement_interval_count": 0,
            "movement_duration_s": 0.0,
        }

    movement_intervals = build_movement_interval(
        speed,
        speed_threshold_cm_s=threshold,
    )
    interval_count, duration = movement_interval_summary(movement_intervals)
    if duration <= 0.0:
        table = _build_movement_table(
            identity_rows=identity_rows,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            movement_spike_counts=empty_counts,
            movement_duration_s=0.0,
            movement_firing_rates_hz=undefined_rates,
            firing_rate_status="no_movement",
            position_sample_count=position_sample_count,
            finite_position_sample_count=finite_position_count,
            finite_speed_sample_count=finite_speed_count,
            movement_interval_count=interval_count,
            speed_threshold_cm_s=threshold,
            speed_smoothing_sigma_s=sigma,
        )
        return {
            "table": table,
            "movement_intervals": movement_intervals,
            "analysis_status": "no_movement",
            "n_units": n_units,
            "n_valid_units": 0,
            "n_units_with_spikes": 0,
            "position_sample_count": position_sample_count,
            "finite_position_sample_count": finite_position_count,
            "finite_speed_sample_count": finite_speed_count,
            "movement_interval_count": interval_count,
            "movement_duration_s": 0.0,
        }

    spike_counts = _movement_spike_counts(spikes, movement_intervals)
    firing_rates = spike_counts.astype(float) / duration
    table = _build_movement_table(
        identity_rows=identity_rows,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        movement_spike_counts=spike_counts,
        movement_duration_s=duration,
        movement_firing_rates_hz=firing_rates,
        firing_rate_status="valid",
        position_sample_count=position_sample_count,
        finite_position_sample_count=finite_position_count,
        finite_speed_sample_count=finite_speed_count,
        movement_interval_count=interval_count,
        speed_threshold_cm_s=threshold,
        speed_smoothing_sigma_s=sigma,
    )
    return {
        "table": table,
        "movement_intervals": movement_intervals,
        "analysis_status": "valid",
        "n_units": n_units,
        "n_valid_units": n_units,
        "n_units_with_spikes": int(np.sum(spike_counts > 0)),
        "position_sample_count": position_sample_count,
        "finite_position_sample_count": finite_position_count,
        "finite_speed_sample_count": finite_speed_count,
        "movement_interval_count": interval_count,
        "movement_duration_s": duration,
    }


def _single_table_value(table: pd.DataFrame, column: str) -> Any:
    """Return one value shared by every row of a non-empty table."""
    values = table[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Movement table column {column!r} must be constant.")
    return values.iloc[0]


def validate_movement_firing_rate_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and return one canonical all-unit movement-rate table."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("Movement firing-rate artifact must be a pandas DataFrame.")
    missing = [column for column in MOVEMENT_TABLE_COLUMNS if column not in table]
    if missing:
        raise ValueError(f"Movement firing-rate table is missing columns {missing!r}.")
    if table.empty:
        return table

    for column in ("stable_unit_id", "group_unit_id"):
        if table[column].duplicated().any():
            raise ValueError(f"Movement table contains duplicate {column!r} values.")
    expected_stable_ids = (
        table["spikesorting_merge_id"].astype(str)
        + ":"
        + table["unit_id"].astype(str)
    )
    if not np.array_equal(
        expected_stable_ids.to_numpy(dtype=str),
        table["stable_unit_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Movement table stable unit identities are inconsistent.")

    status = str(_single_table_value(table, "firing_rate_status"))
    if status not in ANALYSIS_STATUSES[1:]:
        raise ValueError(f"Unsupported firing_rate_status {status!r}.")
    threshold, sigma = _validate_parameters(
        _single_table_value(table, "speed_threshold_cm_s"),
        _single_table_value(table, "speed_smoothing_sigma_s"),
    )
    del threshold, sigma
    for column in ("animal_name", "date", "region", "epoch"):
        if not str(_single_table_value(table, column)):
            raise ValueError(f"Movement table column {column!r} must be non-empty.")

    integer_columns = (
        "movement_spike_count",
        "position_sample_count",
        "finite_position_sample_count",
        "finite_speed_sample_count",
        "movement_interval_count",
    )
    integer_values: dict[str, np.ndarray] = {}
    for column in integer_columns:
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0) or not np.allclose(
            values,
            np.rint(values),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(f"Movement table column {column!r} must be non-negative integers.")
        integer_values[column] = np.rint(values).astype(np.int64)
        if column != "movement_spike_count" and np.any(
            integer_values[column] != integer_values[column][0]
        ):
            raise ValueError(f"Movement table column {column!r} must be constant.")
    if np.any(
        integer_values["finite_position_sample_count"]
        > integer_values["position_sample_count"]
    ) or np.any(
        integer_values["finite_speed_sample_count"]
        > integer_values["position_sample_count"]
    ):
        raise ValueError("Movement table finite sample counts exceed position samples.")

    duration = pd.to_numeric(
        table["movement_duration_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(duration)) or np.any(duration < 0.0) or not np.allclose(
        duration,
        duration[0],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Movement duration must be one non-negative finite value.")
    rates = pd.to_numeric(
        table["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    counts = integer_values["movement_spike_count"].astype(float)
    position_count = int(integer_values["position_sample_count"][0])
    finite_position_count = int(
        integer_values["finite_position_sample_count"][0]
    )
    finite_speed_count = int(integer_values["finite_speed_sample_count"][0])
    interval_count = int(integer_values["movement_interval_count"][0])
    if status == "valid":
        if (
            position_count < 2
            or finite_position_count < 2
            or finite_speed_count < 2
            or interval_count < 1
            or duration[0] <= 0.0
            or not np.all(np.isfinite(rates))
        ):
            raise ValueError(
                "Valid movement rows require sufficient finite samples, at least one "
                "interval, positive duration, and finite rates."
            )
        if not np.allclose(rates, counts / duration[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Movement firing rates do not equal spike count divided by duration.")
    else:
        if duration[0] != 0.0 or np.any(counts != 0.0) or np.any(np.isfinite(rates)):
            raise ValueError(
                f"{status} rows require zero duration/counts and undefined rates."
            )
        if interval_count != 0:
            raise ValueError(f"{status} rows require an empty movement IntervalSet.")
        if status == "no_valid_position" and (
            finite_position_count >= 2 and finite_speed_count >= 2
        ):
            raise ValueError(
                "no_valid_position requires fewer than two finite position or speed samples."
            )
        if status == "no_movement" and (
            finite_position_count < 2 or finite_speed_count < 2
        ):
            raise ValueError(
                "no_movement requires sufficient finite position and speed samples."
            )
    return table


def validate_movement_artifacts(
    table: pd.DataFrame,
    movement_intervals: Any,
) -> tuple[pd.DataFrame, Any]:
    """Cross-validate one movement-rate table and its exact IntervalSet."""
    validate_movement_firing_rate_table(table)
    interval_count, duration = movement_interval_summary(movement_intervals)
    if table.empty:
        if interval_count != 0 or duration != 0.0:
            raise ValueError("An empty no-unit table requires empty movement intervals.")
        return table, movement_intervals
    expected_count = int(_single_table_value(table, "movement_interval_count"))
    expected_duration = float(_single_table_value(table, "movement_duration_s"))
    if interval_count != expected_count or not np.isclose(
        duration,
        expected_duration,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Movement-rate table and IntervalSet metadata disagree.")
    return table, movement_intervals


def load_movement_firing_rate_artifact(path: Path) -> pd.DataFrame:
    """Load and validate one canonical movement-rate Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Movement firing-rate artifact not found: {path}")
    table = pd.read_parquet(path)
    validate_movement_firing_rate_table(table)
    return table


def load_movement_interval_artifact(path: Path) -> Any:
    """Load and validate one exact Pynapple movement IntervalSet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Movement interval artifact not found: {path}")
    import pynapple as nap

    intervals = nap.load_file(path)
    if getattr(intervals, "nap_class", None) != "IntervalSet":
        raise ValueError(f"Movement interval artifact is not an IntervalSet: {path}")
    movement_interval_summary(intervals)
    return intervals


def load_movement_artifacts(artifact_dir: Path) -> dict[str, Any]:
    """Load and cross-validate both canonical movement artifacts."""
    artifact_dir = Path(artifact_dir)
    firing_rate_path = artifact_dir / FIRING_RATE_FILENAME
    movement_intervals_path = artifact_dir / INTERVALS_FILENAME
    table = load_movement_firing_rate_artifact(firing_rate_path)
    movement_intervals = load_movement_interval_artifact(movement_intervals_path)
    validate_movement_artifacts(table, movement_intervals)
    status = "no_units" if table.empty else str(table["firing_rate_status"].iloc[0])
    return {
        "table": table,
        "movement_intervals": movement_intervals,
        "analysis_status": status,
        "firing_rate_path": firing_rate_path,
        "movement_intervals_path": movement_intervals_path,
    }


def align_movement_firing_rates(
    table: pd.DataFrame,
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.Series:
    """Return persistent artifact rates indexed by ephemeral TsGroup keys."""
    validate_movement_firing_rate_table(table)
    identity_rows = _stable_identity_rows(spikes, stable_unit_ids)
    if not identity_rows:
        if not table.empty:
            raise ValueError("A non-empty movement table cannot align to no units.")
        return pd.Series(dtype=float, name="movement_firing_rate_hz")
    expected_ids = [row["stable_unit_id"] for row in identity_rows]
    actual_ids = table["stable_unit_id"].astype(str).tolist()
    if set(expected_ids) != set(actual_ids) or len(expected_ids) != len(actual_ids):
        raise ValueError("Movement artifact unit identities do not match the selected group.")
    rates_by_id = table.set_index("stable_unit_id")["movement_firing_rate_hz"]
    return pd.Series(
        [float(rates_by_id.loc[stable_id]) for stable_id in expected_ids],
        index=[row["group_unit_id"] for row in identity_rows],
        dtype=float,
        name="movement_firing_rate_hz",
    )


def write_movement_artifacts(
    table: pd.DataFrame,
    movement_intervals: Any,
    artifact_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically write the Parquet and exact Pynapple IntervalSet as one bundle."""
    validate_movement_artifacts(table, movement_intervals)
    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite movement artifacts: {artifact_dir}")
    if artifact_dir.exists() and not artifact_dir.is_dir():
        raise ValueError(f"Movement artifact destination is not a directory: {artifact_dir}")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_dir.name}.",
            suffix=".tmp",
            dir=artifact_dir.parent,
        )
    )
    backup_dir = artifact_dir.with_name(
        f".{artifact_dir.name}.{uuid.uuid4().hex}.backup"
    )
    had_existing = artifact_dir.exists()
    try:
        table.to_parquet(stage_dir / FIRING_RATE_FILENAME, index=False)
        movement_intervals.save(stage_dir / INTERVALS_FILENAME)
        load_movement_artifacts(stage_dir)
        if had_existing:
            os.replace(artifact_dir, backup_dir)
        os.replace(stage_dir, artifact_dir)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if backup_dir.exists():
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            os.replace(backup_dir, artifact_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    return {
        "firing_rate_path": artifact_dir / FIRING_RATE_FILENAME,
        "movement_intervals_path": artifact_dir / INTERVALS_FILENAME,
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "DEFAULT_ARTIFACT_ROOT",
    "FIRING_RATE_FILENAME",
    "IDENTITY_COLUMNS",
    "INTERVALS_FILENAME",
    "MOVEMENT_TABLE_COLUMNS",
    "align_movement_firing_rates",
    "compute_selected_movement_firing_rate",
    "empty_movement_firing_rate_table",
    "get_movement_artifact_dir",
    "get_movement_artifact_paths",
    "load_movement_artifacts",
    "load_movement_firing_rate_artifact",
    "load_movement_interval_artifact",
    "movement_interval_summary",
    "validate_movement_artifacts",
    "validate_movement_firing_rate_table",
    "write_movement_artifacts",
]
