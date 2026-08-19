"""Database-free path-specific place-stability adapter and Parquet artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from numbers import Real
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import pandas as pd

from v1ca1.task_progression.stability import compute_trajectory_stability_table


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "path_specific_place_stability"
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
    path_specific_place_stability_id: Any,
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
        path_specific_place_stability_id,
        name="path_specific_place_stability_id",
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
                "stability_shape_overlap": np.nan,
                "odd_tuning_curve_area": np.nan,
                "even_tuning_curve_area": np.nan,
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
                "shape_overlap_status": str(status),
                "stability_segmented_shape_overlap": np.nan,
                "segment_stability_shape_overlaps": "[NaN, NaN, NaN]",
                "segment_shape_overlap_statuses": json.dumps([str(status)] * 3),
                "odd_segment_mean_firing_rates_hz": "[NaN, NaN, NaN]",
                "even_segment_mean_firing_rates_hz": "[NaN, NaN, NaN]",
                "odd_segment_tuning_curve_areas": "[NaN, NaN, NaN]",
                "even_segment_tuning_curve_areas": "[NaN, NaN, NaN]",
                "segment_edges_normalized": "[0.0, NaN, NaN, 1.0]",
                "segmented_shape_overlap_status": str(status),
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


def _curve_identity_table(curve: Any) -> pd.DataFrame:
    """Return persistent unit coordinates from one canonical tuning curve."""
    from v1ca1.spyglass.path_specific_place import (
        validate_path_specific_place_tuning_curve,
    )

    validate_path_specific_place_tuning_curve(curve)
    return pd.DataFrame(
        {
            name: np.asarray(curve.coords[name].values).astype(str)
            for name in IDENTITY_COLUMNS
        }
    )


def _validate_curve_pair(odd_curve: Any, even_curve: Any) -> pd.DataFrame:
    """Validate one matching odd/even curve pair and return its identities."""
    odd_identity = _curve_identity_table(odd_curve)
    even_identity = _curve_identity_table(even_curve)
    if str(odd_curve.attrs["trial_subset"]) != "odd" or str(
        even_curve.attrs["trial_subset"]
    ) != "even":
        raise ValueError("Stability requires one odd and one even tuning curve.")
    if not odd_identity.equals(even_identity):
        raise ValueError(
            "Odd and even tuning curves must contain the same ordered unit identities."
        )
    if tuple(odd_curve.dims) != tuple(even_curve.dims) or not np.allclose(
        np.asarray(odd_curve.coords[odd_curve.dims[1]].values, dtype=float),
        np.asarray(even_curve.coords[even_curve.dims[1]].values, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Odd and even tuning curves must use identical position bins.")

    shared_attributes = (
        "animal_name",
        "date",
        "region",
        "epoch",
        "trajectory_type",
        "binning_mode",
        "sigma_bins",
        "graph_length_cm",
        "bin_edges_cm_json",
        "n_units",
    )
    for name in shared_attributes:
        odd_value = odd_curve.attrs[name]
        even_value = even_curve.attrs[name]
        if isinstance(odd_value, Real) and isinstance(even_value, Real):
            matches = np.isclose(
                float(odd_value),
                float(even_value),
                rtol=1e-10,
                atol=1e-12,
            )
        else:
            matches = str(odd_value) == str(even_value)
        if not bool(matches):
            raise ValueError(
                f"Odd and even tuning curves disagree on {name}."
            )
    return odd_identity


def _movement_rates_for_curve_pair(
    table: pd.DataFrame,
    *,
    identity: pd.DataFrame,
) -> dict[str, Any]:
    """Validate and align movement firing rates to tuning-curve identities."""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("movement_firing_rate_table must be a pandas DataFrame.")
    missing = sorted(set(MOVEMENT_RATE_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(
            "Movement firing-rate table is missing canonical columns "
            f"{missing!r}."
        )
    expected_ids = identity["stable_unit_id"].astype(str).tolist()
    if not expected_ids:
        if not table.empty:
            raise ValueError(
                "Movement firing-rate table must be empty when tuning curves have no units."
            )
        return {
            "status": "no_units",
            "rates": pd.Series(dtype=float),
        }
    if table.empty:
        raise ValueError(
            "Movement firing-rate table must contain every tuning-curve unit."
        )

    observed = table.loc[:, MOVEMENT_RATE_COLUMNS].copy()
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        observed[name] = observed[name].astype(str)
    if observed["stable_unit_id"].duplicated().any():
        raise ValueError("Movement firing-rate table has duplicate stable units.")
    observed = observed.set_index("stable_unit_id", drop=False)
    if set(observed.index) != set(expected_ids):
        raise ValueError(
            "Movement firing-rate identities do not match the tuning-curve units."
        )
    observed = observed.loc[expected_ids].reset_index(drop=True)
    for name in ("spikesorting_merge_id", "unit_id", "stable_unit_id"):
        if observed[name].astype(str).tolist() != identity[name].astype(str).tolist():
            raise ValueError(
                f"Movement firing-rate {name} does not match tuning-curve identity."
            )
    statuses = observed["firing_rate_status"].astype(str)
    if statuses.nunique() != 1:
        raise ValueError("Movement firing-rate status must be uniform across units.")
    status = str(statuses.iloc[0])
    if status not in MOVEMENT_RATE_STATUSES:
        raise ValueError(f"Unknown movement firing-rate status {status!r}.")
    rates = pd.to_numeric(
        observed["movement_firing_rate_hz"], errors="coerce"
    ).to_numpy(dtype=float)
    if status == "valid":
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError("Valid movement firing rates must be finite and non-negative.")
    elif not np.all(np.isnan(rates)):
        raise ValueError(f"{status} movement firing rates must be NaN.")
    return {
        "status": status,
        "rates": pd.Series(rates, index=expected_ids, dtype=float),
    }


def _curve_spike_counts(curve: Any) -> pd.Series:
    """Return validated per-unit spike counts from one computed curve."""
    if "spike_count" not in curve.coords:
        raise ValueError("Tuning curve is missing its per-unit spike_count coordinate.")
    stable_ids = np.asarray(curve.coords["stable_unit_id"].values).astype(str)
    counts = np.asarray(curve.coords["spike_count"].values, dtype=float)
    if counts.shape != stable_ids.shape or (
        not np.all(np.isfinite(counts))
        or np.any(counts < 0.0)
        or not np.allclose(counts, np.rint(counts), rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "Computed odd/even tuning curves require finite non-negative integer "
            "spike_count coordinates."
        )
    return pd.Series(np.rint(counts).astype(int), index=stable_ids, dtype=int)


def _curve_pair_trajectory_status(odd_curve: Any, even_curve: Any) -> str | None:
    """Return the fixed trajectory-level QC status for one curve pair."""
    from v1ca1.task_progression.stability import MIN_FEATURE_SAMPLES_PER_SPLIT

    if int(odd_curve.attrs["n_trials"]) <= 0:
        return "no_odd_trials"
    if int(even_curve.attrs["n_trials"]) <= 0:
        return "no_even_trials"
    if float(odd_curve.attrs["support_duration_s"]) <= 0.0:
        return "no_odd_movement_support"
    if float(even_curve.attrs["support_duration_s"]) <= 0.0:
        return "no_even_movement_support"
    if int(odd_curve.attrs["n_valid_position_samples"]) < (
        MIN_FEATURE_SAMPLES_PER_SPLIT
    ):
        return "insufficient_odd_feature_samples"
    if int(even_curve.attrs["n_valid_position_samples"]) < (
        MIN_FEATURE_SAMPLES_PER_SPLIT
    ):
        return "insufficient_even_feature_samples"
    return None


def compute_selected_stability_from_tuning_curves(
    *,
    odd_tuning_curve: Any,
    even_tuning_curve: Any,
    movement_firing_rate_table: pd.DataFrame,
) -> dict[str, Any]:
    """Compute fixed-QC stability from matching persisted odd/even curves."""
    from v1ca1.task_progression.stability import (
        _evaluate_segmented_stability_shape_overlap,
        _evaluate_stability_correlation,
        _evaluate_stability_shape_overlap,
    )
    from v1ca1.helper.wtrack import get_wtrack_segment_edges

    identity = _validate_curve_pair(odd_tuning_curve, even_tuning_curve)
    movement = _movement_rates_for_curve_pair(
        movement_firing_rate_table,
        identity=identity,
    )
    n_units = len(identity)
    if n_units == 0:
        return {
            "table": empty_stability_table(),
            "analysis_status": "no_units",
            "n_units": 0,
            "n_valid_units": 0,
        }

    odd_counts = _curve_spike_counts(odd_tuning_curve)
    even_counts = _curve_spike_counts(even_tuning_curve)
    trajectory_status = _curve_pair_trajectory_status(
        odd_tuning_curve,
        even_tuning_curve,
    )
    if movement["status"] in {"no_valid_position", "no_movement"}:
        trajectory_status = movement["status"]

    metadata = odd_tuning_curve.attrs
    progression = np.asarray(
        odd_tuning_curve.coords[odd_tuning_curve.dims[1]].values,
        dtype=float,
    )
    if np.nanmax(progression) > 1.0 + 1e-9:
        progression = progression / float(metadata["graph_length_cm"])
    segment_edges = get_wtrack_segment_edges(str(metadata["animal_name"]))
    rows: list[dict[str, Any]] = []
    for index, unit in identity.iterrows():
        stable_id = str(unit["stable_unit_id"])
        n_odd_spikes = int(odd_counts.loc[stable_id])
        n_even_spikes = int(even_counts.loc[stable_id])
        if trajectory_status is None:
            correlation_qc = _evaluate_stability_correlation(
                np.asarray(odd_tuning_curve.values[index], dtype=float),
                np.asarray(even_tuning_curve.values[index], dtype=float),
                n_odd_spikes=n_odd_spikes,
                n_even_spikes=n_even_spikes,
            )
            shape_overlap_qc = _evaluate_stability_shape_overlap(
                np.asarray(odd_tuning_curve.values[index], dtype=float),
                np.asarray(even_tuning_curve.values[index], dtype=float),
                n_odd_spikes=n_odd_spikes,
                n_even_spikes=n_even_spikes,
            )
            segmented_shape_overlap_qc = (
                _evaluate_segmented_stability_shape_overlap(
                    np.asarray(odd_tuning_curve.values[index], dtype=float),
                    np.asarray(even_tuning_curve.values[index], dtype=float),
                    progression,
                    segment_edges,
                    n_odd_spikes=n_odd_spikes,
                    n_even_spikes=n_even_spikes,
                )
            )
        else:
            correlation_qc = {
                "stability_correlation": np.nan,
                "stability_status": trajectory_status,
                "n_odd_finite_bins": 0,
                "n_even_finite_bins": 0,
                "n_paired_finite_bins": 0,
            }
            shape_overlap_qc = {
                "stability_shape_overlap": np.nan,
                "odd_tuning_curve_area": np.nan,
                "even_tuning_curve_area": np.nan,
                "shape_overlap_status": trajectory_status,
            }
            segmented_shape_overlap_qc = {
                "stability_segmented_shape_overlap": np.nan,
                "segment_stability_shape_overlaps": "[NaN, NaN, NaN]",
                "segment_shape_overlap_statuses": json.dumps(
                    [trajectory_status] * 3
                ),
                "odd_segment_mean_firing_rates_hz": "[NaN, NaN, NaN]",
                "even_segment_mean_firing_rates_hz": "[NaN, NaN, NaN]",
                "odd_segment_tuning_curve_areas": "[NaN, NaN, NaN]",
                "even_segment_tuning_curve_areas": "[NaN, NaN, NaN]",
                "segment_edges_normalized": json.dumps(segment_edges.tolist()),
                "segmented_shape_overlap_status": trajectory_status,
            }
        rows.append(
            {
                **{name: str(unit[name]) for name in IDENTITY_COLUMNS},
                "animal_name": str(metadata["animal_name"]),
                "date": str(metadata["date"]),
                "region": str(metadata["region"]),
                "epoch": str(metadata["epoch"]),
                "trajectory_type": str(metadata["trajectory_type"]),
                "firing_rate_hz": float(movement["rates"].loc[stable_id]),
                "stability_correlation": correlation_qc["stability_correlation"],
                "stability_shape_overlap": shape_overlap_qc[
                    "stability_shape_overlap"
                ],
                "odd_tuning_curve_area": shape_overlap_qc[
                    "odd_tuning_curve_area"
                ],
                "even_tuning_curve_area": shape_overlap_qc[
                    "even_tuning_curve_area"
                ],
                "n_odd_trials": int(odd_tuning_curve.attrs["n_trials"]),
                "n_even_trials": int(even_tuning_curve.attrs["n_trials"]),
                "odd_duration_s": float(
                    odd_tuning_curve.attrs["support_duration_s"]
                ),
                "even_duration_s": float(
                    even_tuning_curve.attrs["support_duration_s"]
                ),
                "n_odd_feature_samples": int(
                    odd_tuning_curve.attrs["n_valid_position_samples"]
                ),
                "n_even_feature_samples": int(
                    even_tuning_curve.attrs["n_valid_position_samples"]
                ),
                "n_odd_spikes": n_odd_spikes,
                "n_even_spikes": n_even_spikes,
                "n_odd_finite_bins": correlation_qc["n_odd_finite_bins"],
                "n_even_finite_bins": correlation_qc["n_even_finite_bins"],
                "n_paired_finite_bins": correlation_qc["n_paired_finite_bins"],
                "stability_status": correlation_qc["stability_status"],
                "shape_overlap_status": shape_overlap_qc[
                    "shape_overlap_status"
                ],
                **segmented_shape_overlap_qc,
            }
        )
    columns = empty_stability_table().columns
    table = pd.DataFrame.from_records(rows).loc[:, columns]
    n_valid_units = int(table["stability_status"].astype(str).eq("valid").sum())
    if movement["status"] in {"no_valid_position", "no_movement"}:
        analysis_status = movement["status"]
    else:
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    return {
        "table": table,
        "analysis_status": analysis_status,
        "n_units": n_units,
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
    "compute_selected_stability_from_tuning_curves",
    "empty_stability_table",
    "get_stability_artifact_path",
    "write_stability_artifact",
]
