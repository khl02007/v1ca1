"""Database-free task-progression stability adapter and Parquet artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.helper.session import build_movement_interval, build_speed_tsd
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


def _movement_firing_rates(spikes: Any, movement_interval: Any) -> pd.Series:
    """Return one epoch-wide movement firing rate per ephemeral group key."""
    unit_keys = list(spikes.keys())
    duration = float(movement_interval.tot_length())
    if duration <= 0:
        values = np.zeros(len(unit_keys), dtype=float)
    else:
        counts = np.asarray(spikes.count(ep=movement_interval).to_numpy(), dtype=float)
        values = np.sum(counts, axis=0).reshape(-1) / duration
    if values.size != len(unit_keys):
        raise ValueError("Movement firing rates do not align with TsGroup units.")
    return pd.Series(values, index=unit_keys, dtype=float)


def _attach_unit_identity(
    table: pd.DataFrame,
    *,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Replace ephemeral tuning keys with persistent composite identity."""
    group_keys = list(spikes.keys())
    stable_unit_ids = [dict(unit_id) for unit_id in stable_unit_ids]
    if len(group_keys) != len(stable_unit_ids):
        raise ValueError("TsGroup and stable unit identity lengths must match.")
    identity = {
        group_key: (
            str(unit_id["spikesorting_merge_id"]),
            str(unit_id["unit_id"]),
        )
        for group_key, unit_id in zip(group_keys, stable_unit_ids)
    }
    if len(identity) != len(group_keys) or len(set(identity.values())) != len(identity):
        raise ValueError("Persistent unit identities must be unique and aligned.")

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
    output["spikesorting_merge_id"] = [item[0] for item in identities]
    output["unit_id"] = [item[1] for item in identities]
    output["stable_unit_id"] = [f"{item[0]}:{item[1]}" for item in identities]
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
    speed_threshold_cm_s: float,
    speed_smoothing_sigma_s: float,
    place_bin_size_cm: float,
) -> dict[str, Any]:
    """Compute one selected trajectory artifact from already loaded inputs."""
    if spikes is None or not stable_unit_ids:
        return {
            "table": empty_stability_table(),
            "analysis_status": "no_units",
            "n_units": 0,
            "n_valid_units": 0,
        }
    position_values, position_times = _position_arrays(position)
    speed = build_speed_tsd(
        position_values,
        position_times,
        position_offset=0,
        speed_smoothing_sigma_s=float(speed_smoothing_sigma_s),
    )
    movement_interval = build_movement_interval(
        speed,
        speed_threshold_cm_s=float(speed_threshold_cm_s),
    )
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
        epoch_firing_rates=_movement_firing_rates(spikes, movement_interval),
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
    "build_task_progression_from_graph",
    "compute_selected_stability",
    "empty_stability_table",
    "get_stability_artifact_path",
    "write_stability_artifact",
]
