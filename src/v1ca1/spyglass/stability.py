"""Database-free task-progression stability adapter and Parquet artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.task_progression.stability import compute_trajectory_stability_table


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "task_progression_stability"
ARTIFACT_FILENAME = "stability.parquet"
IDENTITY_COLUMNS = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
MOVEMENT_RATE_COLUMNS = (
    *IDENTITY_COLUMNS,
    "movement_spike_count",
    "movement_duration_s",
    "movement_firing_rate_hz",
    "firing_rate_status",
)
MOVEMENT_RATE_STATUSES = ("valid", "no_valid_position", "no_movement")


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


def get_stability_artifact_path(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    trajectory_type: str,
    region: str,
    task_progression_stability_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return one UUID-keyed, session-first stability Parquet path."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "trajectory_type": trajectory_type,
            "region": region,
        }.items()
    }
    stability_id = _uuid_component(
        task_progression_stability_id,
        name="task_progression_stability_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["trajectory_type"]
        / components["region"]
        / stability_id
        / ARTIFACT_FILENAME
    )


def _position_arrays(position: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned two-dimensional centimeter position and seconds."""
    values = np.asarray(getattr(position, "d", position), dtype=float)
    timestamps = np.asarray(getattr(position, "t", ()), dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Selected position must have shape (n_samples, 2).")
    if timestamps.size != values.shape[0]:
        raise ValueError("Selected position samples and timestamps must align.")
    if not np.all(np.isfinite(timestamps)) or (
        timestamps.size > 1 and np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError("Position timestamps must be finite and strictly increasing.")
    return values, timestamps


def _ordered_graph_length(
    node_positions: np.ndarray,
    edge_order: Sequence[Sequence[int]],
    edge_spacing: Sequence[float],
) -> float:
    """Return the selected ordered track length in centimeters."""
    node_positions = np.asarray(node_positions, dtype=float)
    edges = np.asarray(edge_order, dtype=int)
    spacing = np.asarray(edge_spacing, dtype=float).reshape(-1)
    if edges.ndim != 2 or edges.shape[1] != 2 or not len(edges):
        raise ValueError("W-track edge_order must contain at least one edge.")
    if spacing.size not in {0, len(edges) - 1}:
        raise ValueError("W-track edge_spacing must have n_edges - 1 entries.")
    segment_lengths = np.linalg.norm(
        node_positions[edges[:, 1]] - node_positions[edges[:, 0]],
        axis=1,
    )
    length = float(np.sum(segment_lengths) + np.sum(spacing))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("W-track ordered length must be positive and finite.")
    return length


def build_task_progression_from_graph(
    *,
    position: Any,
    trajectory_interval: Any,
    graph_inputs: Mapping[str, Any],
    trajectory_type: str,
) -> tuple[Any, float]:
    """Linearize one selected position series using one selected graph row."""
    configuration_name = str(graph_inputs.get("configuration_name", ""))
    if configuration_name != str(trajectory_type):
        raise ValueError(
            "WTrackGraph configuration_name must equal the selected trajectory_type."
        )
    if graph_inputs.get("coordinate_unit") != "cm":
        raise ValueError("WTrackGraph coordinate_unit must be 'cm'.")
    values, timestamps = _position_arrays(position)
    track_graph_kwargs = dict(graph_inputs.get("track_graph_kwargs", {}))
    linearization_kwargs = dict(graph_inputs.get("linearization_kwargs", {}))
    if set(track_graph_kwargs) != {"node_positions", "edges"}:
        raise ValueError("WTrackGraph track_graph_kwargs are incomplete.")

    import pynapple as nap
    import track_linearization as tl

    track_graph = tl.make_track_graph(**track_graph_kwargs)
    linearized = tl.get_linearized_position(
        position=values,
        track_graph=track_graph,
        **linearization_kwargs,
    )
    graph_length_cm = _ordered_graph_length(
        np.asarray(track_graph_kwargs["node_positions"], dtype=float),
        linearization_kwargs.get("edge_order", ()),
        linearization_kwargs.get("edge_spacing", ()),
    )
    progression = np.asarray(linearized["linear_position"], dtype=float)
    if progression.shape != timestamps.shape:
        raise ValueError("Linearized position does not align with position timestamps.")
    return (
        nap.Tsd(
            t=timestamps,
            d=progression / graph_length_cm,
            time_support=trajectory_interval,
            time_units="s",
        ),
        graph_length_cm,
    )


def _selected_unit_identity(
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Return the exact persistent-to-ephemeral identity selected for analysis."""
    if spikes is None:
        group_keys: list[Any] = []
    else:
        group_keys = list(spikes.keys())
    selected_ids = [dict(unit_id) for unit_id in stable_unit_ids]
    if len(group_keys) != len(selected_ids):
        raise ValueError("TsGroup and stable unit identity lengths must match.")

    rows: list[dict[str, Any]] = []
    for group_key, unit_id in zip(group_keys, selected_ids, strict=True):
        missing = [
            name
            for name in ("spikesorting_merge_id", "unit_id")
            if name not in unit_id
        ]
        if missing:
            raise ValueError(
                f"Stable unit identity is missing required fields {missing!r}."
            )
        merge_id = str(unit_id["spikesorting_merge_id"])
        source_unit_id = str(unit_id["unit_id"])
        if not merge_id or not source_unit_id:
            raise ValueError("Persistent unit identity fields must be non-empty.")
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": source_unit_id,
                "stable_unit_id": f"{merge_id}:{source_unit_id}",
                "group_unit_id": group_key,
            }
        )

    identity = pd.DataFrame.from_records(rows, columns=IDENTITY_COLUMNS)
    if identity.empty:
        return identity
    if identity["stable_unit_id"].duplicated().any():
        raise ValueError("Persistent unit identities must be unique and aligned.")
    if identity["group_unit_id"].duplicated().any():
        raise ValueError("Ephemeral TsGroup unit identifiers must be unique.")
    return identity


def _movement_interval_duration(movement_interval: Any) -> float:
    """Return a finite, non-negative movement duration in seconds."""
    if movement_interval is None or not callable(
        getattr(movement_interval, "tot_length", None)
    ):
        raise TypeError("movement_interval must be a Pynapple-like IntervalSet.")
    duration = float(movement_interval.tot_length())
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("Movement interval duration must be finite and non-negative.")
    return duration


def _validate_movement_firing_rate_table(
    table: pd.DataFrame,
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    movement_interval: Any,
) -> dict[str, Any]:
    """Validate and align one canonical movement-rate artifact to the TsGroup."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("movement_firing_rate_table must be a pandas DataFrame.")
    missing = sorted(set(MOVEMENT_RATE_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(
            "Movement firing-rate table is missing canonical columns "
            f"{missing!r}."
        )

    expected = _selected_unit_identity(
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )
    interval_duration = _movement_interval_duration(movement_interval)
    if expected.empty:
        if not table.empty:
            raise ValueError(
                "Movement firing-rate table must be empty when no units are selected."
            )
        if interval_duration != 0.0:
            raise ValueError(
                "No-unit movement artifacts require an empty movement interval."
            )
        return {
            "status": "no_units",
            "rates": pd.Series(dtype=float),
            "duration_s": interval_duration,
        }
    if table.empty:
        raise ValueError(
            "Movement firing-rate table must contain every selected unit."
        )

    observed = table.loc[:, MOVEMENT_RATE_COLUMNS].copy()
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        observed[name] = observed[name].astype(str)
    if observed["stable_unit_id"].duplicated().any():
        raise ValueError("Movement firing-rate table has duplicate stable_unit_id rows.")

    observed_by_id = observed.set_index("stable_unit_id", drop=False)
    expected_ids = expected["stable_unit_id"].tolist()
    if set(observed_by_id.index) != set(expected_ids):
        raise ValueError(
            "Movement firing-rate identities do not exactly match the selected units."
        )
    observed = observed_by_id.loc[expected_ids].reset_index(drop=True)
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        if observed[name].tolist() != expected[name].astype(str).tolist():
            raise ValueError(
                f"Movement firing-rate {name} does not match selected identity."
            )
    # The saved group_unit_id belongs to the TsGroup used to create the movement
    # artifact. It is provenance, not persistent identity, and may differ from
    # the group keys reconstructed for this downstream computation.

    statuses = observed["firing_rate_status"].astype(str)
    if statuses.nunique() != 1:
        raise ValueError(
            "Movement firing-rate status must be uniform across selected units."
        )
    status = str(statuses.iloc[0])
    if status not in MOVEMENT_RATE_STATUSES:
        raise ValueError(f"Unknown movement firing-rate status {status!r}.")

    durations = pd.to_numeric(
        observed["movement_duration_s"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.all(np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError(
            "Movement firing-rate durations must be finite and non-negative."
        )
    if not np.allclose(durations, interval_duration, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "Movement firing-rate duration does not match movement_interval."
        )

    counts = pd.to_numeric(
        observed["movement_spike_count"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(counts))
        or np.any(counts < 0.0)
        or not np.allclose(counts, np.rint(counts), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "Movement spike counts must be finite non-negative integers."
        )
    rates = pd.to_numeric(
        observed["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if status == "valid":
        if interval_duration <= 0.0:
            raise ValueError("Valid movement rates require positive movement duration.")
        expected_rates = counts / interval_duration
        if (
            not np.all(np.isfinite(rates))
            or np.any(rates < 0.0)
            or not np.allclose(rates, expected_rates, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(
                "Valid movement firing rates must equal spike count divided by duration."
            )
    else:
        if interval_duration != 0.0 or np.any(counts != 0.0):
            raise ValueError(
                f"{status} movement rows require zero duration and zero spike counts."
            )
        if not np.all(np.isnan(rates)):
            raise ValueError(f"{status} movement firing rates must be NaN.")

    return {
        "status": status,
        "rates": pd.Series(
            rates,
            index=expected["group_unit_id"].tolist(),
            dtype=float,
        ),
        "duration_s": interval_duration,
    }


def _attach_unit_identity(
    table: pd.DataFrame,
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Replace ephemeral tuning keys with persistent composite identity."""
    identity_table = _selected_unit_identity(
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )
    identity = identity_table.set_index("group_unit_id").to_dict("index")

    output = table.copy()
    if output.empty:
        output = output.rename(columns={"unit": "group_unit_id"})
        output["spikesorting_merge_id"] = pd.Series(dtype=str)
        output["unit_id"] = pd.Series(dtype=str)
        output["stable_unit_id"] = pd.Series(dtype=str)
        ordered = [*IDENTITY_COLUMNS]
        ordered.extend(column for column in output if column not in ordered)
        return output.loc[:, ordered]
    if "unit" not in output:
        raise ValueError("Stability output is missing its ephemeral unit column.")
    unknown = [unit for unit in output["unit"] if unit not in identity]
    if unknown:
        raise ValueError(f"Stability output contains unknown group unit keys {unknown!r}.")
    identities = [identity[unit] for unit in output["unit"]]
    output = output.rename(columns={"unit": "group_unit_id"})
    output["spikesorting_merge_id"] = [
        item["spikesorting_merge_id"] for item in identities
    ]
    output["unit_id"] = [item["unit_id"] for item in identities]
    output["stable_unit_id"] = [item["stable_unit_id"] for item in identities]
    ordered = [*IDENTITY_COLUMNS]
    ordered.extend(column for column in output if column not in ordered)
    return output.loc[:, ordered]


def empty_stability_table() -> pd.DataFrame:
    """Return an empty Spyglass stability table with persistent identity columns."""
    from v1ca1.task_progression.stability import _empty_stability_table

    return _attach_unit_identity(
        _empty_stability_table(),
        spikes={},
        stable_unit_ids=[],
    )


def _terminal_stability_table(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    firing_rates: pd.Series,
    status: str,
) -> pd.DataFrame:
    """Return one explicit terminal-QC row per selected unit."""
    from v1ca1.task_progression.stability import _empty_stability_table

    rows = []
    for group_unit_id in list(spikes.keys()):
        rows.append(
            {
                "animal_name": str(animal_name),
                "date": str(date),
                "unit": group_unit_id,
                "region": str(region),
                "epoch": str(epoch),
                "trajectory_type": str(trajectory_type),
                "firing_rate_hz": float(firing_rates.loc[group_unit_id]),
                "stability_correlation": np.nan,
                "n_odd_trials": 0,
                "n_even_trials": 0,
                "odd_duration_s": 0.0,
                "even_duration_s": 0.0,
                "n_odd_feature_samples": 0,
                "n_even_feature_samples": 0,
                "n_odd_spikes": 0,
                "n_even_spikes": 0,
                "n_odd_finite_bins": 0,
                "n_even_finite_bins": 0,
                "n_paired_finite_bins": 0,
                "stability_status": str(status),
            }
        )
    table = pd.DataFrame.from_records(rows, columns=_empty_stability_table().columns)
    return _attach_unit_identity(
        table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )


def compute_selected_stability(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_interval: Any,
    graph_inputs: Mapping[str, Any],
    movement_interval: Any,
    movement_firing_rate_table: pd.DataFrame,
    place_bin_size_cm: float,
) -> dict[str, Any]:
    """Compute one selected trajectory artifact from already loaded inputs."""
    movement = _validate_movement_firing_rate_table(
        movement_firing_rate_table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
        movement_interval=movement_interval,
    )
    if movement["status"] == "no_units":
        return {
            "table": empty_stability_table(),
            "analysis_status": "no_units",
            "n_units": 0,
            "n_valid_units": 0,
        }
    if movement["status"] in {"no_valid_position", "no_movement"}:
        terminal_table = _terminal_stability_table(
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory_type=trajectory_type,
            spikes=spikes,
            stable_unit_ids=stable_unit_ids,
            firing_rates=movement["rates"],
            status=movement["status"],
        )
        return {
            "table": terminal_table,
            "analysis_status": movement["status"],
            "n_units": len(stable_unit_ids),
            "n_valid_units": 0,
        }

    task_progression, graph_length_cm = build_task_progression_from_graph(
        position=position,
        trajectory_interval=trajectory_interval,
        graph_inputs=graph_inputs,
        trajectory_type=trajectory_type,
    )
    place_bin_size_cm = float(place_bin_size_cm)
    if not np.isfinite(place_bin_size_cm) or place_bin_size_cm <= 0:
        raise ValueError("place_bin_size_cm must be positive and finite.")
    normalized_bin_size = place_bin_size_cm / graph_length_cm
    bins = np.arange(0.0, 1.0 + normalized_bin_size, normalized_bin_size)
    table = compute_trajectory_stability_table(
        animal_name=str(animal_name),
        date=str(date),
        region=str(region),
        epoch=str(epoch),
        trajectory_type=str(trajectory_type),
        spikes=spikes,
        task_progression=task_progression,
        trajectory_interval=trajectory_interval,
        movement_interval=movement_interval,
        bins=bins,
        epoch_firing_rates=movement["rates"],
    )
    expected_group_units = list(spikes.keys())
    if (
        "unit" not in table
        or table["unit"].duplicated().any()
        or set(table["unit"]) != set(expected_group_units)
    ):
        raise ValueError(
            "Stability output must contain exactly one row per selected TsGroup unit."
        )
    table = table.set_index("unit", drop=False).loc[expected_group_units].reset_index(
        drop=True
    )
    output_rates = pd.to_numeric(table["firing_rate_hz"], errors="coerce").to_numpy(
        dtype=float
    )
    expected_rates = movement["rates"].loc[expected_group_units].to_numpy(dtype=float)
    if not np.allclose(
        output_rates,
        expected_rates,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise ValueError(
            "Stability firing_rate_hz values do not match the upstream movement artifact."
        )
    table = _attach_unit_identity(
        table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )
    valid = table["stability_status"].astype(str).eq("valid")
    n_valid_units = int(valid.sum())
    return {
        "table": table,
        "analysis_status": "valid" if n_valid_units else "no_valid_units",
        "n_units": len(stable_unit_ids),
        "n_valid_units": n_valid_units,
    }


def write_stability_artifact(
    table: pd.DataFrame,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one stability Parquet without implicit overwrite."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite stability artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.parquet")
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.backup")
    had_existing = path.exists()
    try:
        table.to_parquet(temporary, index=False)
        if had_existing:
            os.replace(path, backup)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        if backup.exists():
            os.replace(backup, path)
        raise
    else:
        backup.unlink(missing_ok=True)
    return path


__all__ = [
    "ARTIFACT_DIRNAME",
    "ARTIFACT_FILENAME",
    "DEFAULT_ARTIFACT_ROOT",
    "MOVEMENT_RATE_COLUMNS",
    "build_task_progression_from_graph",
    "compute_selected_stability",
    "empty_stability_table",
    "get_stability_artifact_path",
    "write_stability_artifact",
]
