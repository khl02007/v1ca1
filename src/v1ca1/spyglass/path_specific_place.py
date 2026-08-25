"""Database-free path-specific place tuning curves and artifact adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "path_specific_place_tuning_curve"
ARTIFACT_FILENAME = "tuning_curve.nc"
NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_TUNING_TABLE_NAME = "path_specific_place_tuning"
NWB_BINS_TABLE_NAME = "path_specific_place_bins"
NWB_PROVENANCE_TABLE_NAME = "path_specific_place_provenance"
BINNING_MODES = ("bin_size_cm", "bin_count")
TRIAL_SUBSETS = ("all", "odd", "even")
ANALYSIS_STATUSES = (
    "valid",
    "no_units",
    "no_trials",
    "no_valid_position",
    "no_movement",
    "no_valid_units",
)
IDENTITY_COORDINATES = (
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
)
POSITION_DIM = "linear_position_cm"
PATH_FRACTION_COORDINATE = "path_fraction"
SPIKE_COUNT_COORDINATE = "spike_count"
UNIT_DIM = "unit"

_NWB_TUNING_COLUMNS = (
    "curve_row",
    "spikesorting_merge_id",
    "unit_id",
    "stable_unit_id",
    "group_unit_id",
    SPIKE_COUNT_COORDINATE,
    "firing_rate_hz",
)
_NWB_BIN_COLUMNS = (
    "position_bin",
    "left_edge_cm",
    POSITION_DIM,
    "right_edge_cm",
    PATH_FRACTION_COORDINATE,
)
_NWB_PROVENANCE_COLUMNS = (
    "artifact_schema_version",
    "metadata_json",
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


def get_path_specific_place_artifact_path(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    trajectory_type: str,
    trial_subset: str,
    region: str,
    path_specific_place_tuning_curve_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return one UUID-keyed, session-first tuning-curve path."""
    components = {
        name: _path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "trajectory_type": trajectory_type,
            "trial_subset": validate_trial_subset(trial_subset),
            "region": region,
        }.items()
    }
    selection_id = _uuid_component(
        path_specific_place_tuning_curve_id,
        name="path_specific_place_tuning_curve_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["trajectory_type"]
        / components["trial_subset"]
        / components["region"]
        / selection_id
        / ARTIFACT_FILENAME
    )


def _numeric_scalar(value: Any, *, name: str) -> float:
    """Return one finite, non-boolean numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one numeric scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def validate_binning_parameters(
    *,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
) -> dict[str, Any]:
    """Validate mutually exclusive spatial binning and smoothing parameters."""
    if (bin_size_cm is None) == (bin_count is None):
        raise ValueError(
            "Exactly one of bin_size_cm and bin_count must be provided."
        )

    sigma = _numeric_scalar(sigma_bins, name="sigma_bins")
    if sigma < 0.0:
        raise ValueError("sigma_bins must be non-negative.")

    if bin_size_cm is not None:
        size = _numeric_scalar(bin_size_cm, name="bin_size_cm")
        if size <= 0.0:
            raise ValueError("bin_size_cm must be positive.")
        return {
            "binning_mode": "bin_size_cm",
            "bin_size_cm": size,
            "bin_count": None,
            "sigma_bins": sigma,
        }

    if isinstance(bin_count, bool) or not isinstance(bin_count, Integral):
        raise TypeError("bin_count must be one integer scalar.")
    count = int(bin_count)
    if count <= 0:
        raise ValueError("bin_count must be positive.")
    return {
        "binning_mode": "bin_count",
        "bin_size_cm": None,
        "bin_count": count,
        "sigma_bins": sigma,
    }


def validate_trial_subset(trial_subset: str) -> str:
    """Return one supported fixed trial subset."""
    subset = str(trial_subset)
    if subset not in TRIAL_SUBSETS:
        raise ValueError(f"trial_subset must be one of {TRIAL_SUBSETS!r}.")
    return subset


def build_position_bin_edges(
    graph_length_cm: float,
    *,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
) -> np.ndarray:
    """Return path-position bin edges in centimeters."""
    graph_length = _numeric_scalar(graph_length_cm, name="graph_length_cm")
    if graph_length <= 0.0:
        raise ValueError("graph_length_cm must be positive.")
    parameters = validate_binning_parameters(
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=0.0,
    )
    if parameters["binning_mode"] == "bin_count":
        return np.linspace(
            0.0,
            graph_length,
            int(parameters["bin_count"]) + 1,
            dtype=float,
        )
    size = float(parameters["bin_size_cm"])
    return np.arange(0.0, graph_length + size, size, dtype=float)


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return validated sorted non-overlapping interval bounds."""
    try:
        starts = np.asarray(intervals.start, dtype=float).reshape(-1)
        ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Intervals must expose numeric start and end arrays.") from exc
    if starts.shape != ends.shape:
        raise ValueError("Interval start and end arrays must align.")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Interval bounds must be finite.")
    if np.any(ends < starts):
        raise ValueError("Interval stops must not precede starts.")
    if starts.size > 1 and (
        np.any(np.diff(starts) < 0.0) or np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError("Intervals must be sorted and non-overlapping.")
    return starts, ends


def _make_interval_set(starts: np.ndarray, ends: np.ndarray) -> Any:
    """Build one second-based Pynapple IntervalSet lazily."""
    import pynapple as nap

    return nap.IntervalSet(
        start=np.asarray(starts, dtype=float),
        end=np.asarray(ends, dtype=float),
        time_units="s",
    )


def select_trial_subset_intervals(
    trajectory_intervals: Any,
    trial_subset: str,
) -> Any:
    """Return all, one-indexed odd, or one-indexed even trajectory trials."""
    subset = validate_trial_subset(trial_subset)
    starts, ends = _extract_interval_bounds(trajectory_intervals)
    if subset == "odd":
        starts, ends = starts[::2], ends[::2]
    elif subset == "even":
        starts, ends = starts[1::2], ends[1::2]
    return _make_interval_set(starts, ends)


def _interval_summary(intervals: Any) -> tuple[int, float]:
    """Return one interval count and duration after validating its bounds."""
    starts, ends = _extract_interval_bounds(intervals)
    duration = float(np.sum(ends - starts))
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("Interval duration must be non-negative and finite.")
    return int(starts.size), duration


def _intersect_intervals(first: Any, second: Any) -> Any:
    """Intersect two Pynapple-like interval objects."""
    intersect = getattr(first, "intersect", None)
    if not callable(intersect):
        raise TypeError("Selected trial intervals must expose intersect().")
    result = intersect(second)
    _interval_summary(result)
    return result


def _subset_spike_counts(spikes: Any, support: Any, *, n_units: int) -> np.ndarray:
    """Return integer spike counts within one trial/movement support."""
    if n_units == 0:
        return np.zeros(0, dtype=np.int64)
    from v1ca1.spyglass.movement import _movement_spike_counts

    counts = np.asarray(
        _movement_spike_counts(spikes, support),
        dtype=np.int64,
    ).reshape(-1)
    if counts.shape != (n_units,):
        raise ValueError("Subset spike counts do not align with selected units.")
    return counts


def _validate_movement_analysis_status(status: str, *, n_units: int) -> str:
    """Validate one upstream movement result status."""
    status = str(status)
    allowed = ("valid", "no_units", "no_valid_position", "no_movement")
    if status not in allowed:
        raise ValueError(f"movement_analysis_status must be one of {allowed!r}.")
    if status == "no_units" and n_units:
        raise ValueError("A no_units movement result cannot supply selected units.")
    return status


def graph_length_from_inputs(
    graph_inputs: Mapping[str, Any],
    *,
    trajectory_type: str,
) -> float:
    """Return the selected ordered path length from one WTrackGraph payload."""
    configuration_name = str(graph_inputs.get("configuration_name", ""))
    if configuration_name != str(trajectory_type):
        raise ValueError(
            "WTrackGraph configuration_name must equal trajectory_type."
        )
    if graph_inputs.get("coordinate_unit") != "cm":
        raise ValueError("WTrackGraph coordinate_unit must be 'cm'.")
    track_graph_kwargs = dict(graph_inputs.get("track_graph_kwargs", {}))
    linearization_kwargs = dict(graph_inputs.get("linearization_kwargs", {}))
    if set(track_graph_kwargs) != {"node_positions", "edges"}:
        raise ValueError("WTrackGraph track_graph_kwargs are incomplete.")

    from v1ca1.spyglass.stability import _ordered_graph_length

    return _ordered_graph_length(
        np.asarray(track_graph_kwargs["node_positions"], dtype=float),
        linearization_kwargs.get("edge_order", ()),
        linearization_kwargs.get("edge_spacing", ()),
    )


def build_path_specific_linear_position(
    *,
    position: Any,
    trajectory_intervals: Any,
    graph_inputs: Mapping[str, Any],
    trajectory_type: str,
) -> tuple[Any, float]:
    """Linearize selected NWB-backed position into path centimeters."""
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    task_progression, graph_length_cm = build_task_progression_from_graph(
        position=position,
        trajectory_interval=trajectory_intervals,
        graph_inputs=graph_inputs,
        trajectory_type=trajectory_type,
    )
    import pynapple as nap

    linear_position = nap.Tsd(
        t=np.asarray(task_progression.t, dtype=float),
        d=np.asarray(task_progression.d, dtype=float) * graph_length_cm,
        time_support=task_progression.time_support,
        time_units="s",
    )
    return linear_position, float(graph_length_cm)


def _identity_rows(
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable coordinates aligned to ephemeral spike-group keys."""
    group_keys = [] if spikes is None else list(spikes.keys())
    identities = [dict(identity) for identity in stable_unit_ids]
    if len(group_keys) != len(identities):
        raise ValueError("TsGroup and stable unit identity lengths must match.")
    try:
        if len(set(group_keys)) != len(group_keys):
            raise ValueError("Ephemeral group unit identifiers must be unique.")
    except TypeError as exc:
        raise ValueError("Ephemeral group unit identifiers must be hashable.") from exc

    rows: list[dict[str, Any]] = []
    persistent_ids: set[tuple[str, str]] = set()
    for group_key, identity in zip(group_keys, identities, strict=True):
        missing = [
            name
            for name in ("spikesorting_merge_id", "unit_id")
            if name not in identity
        ]
        if missing:
            raise ValueError(f"Stable unit identity is missing fields {missing!r}.")
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
                "group_unit_id": str(group_key),
                "_group_key": group_key,
            }
        )
    return rows


def _build_curve(
    values: np.ndarray,
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    spike_counts: Sequence[float],
    bin_edges_cm: np.ndarray,
    attrs: Mapping[str, Any],
) -> Any:
    """Build one canonical stable-identity xarray DataArray."""
    import xarray as xr

    values = np.asarray(values, dtype=float)
    edges = np.asarray(bin_edges_cm, dtype=float).reshape(-1)
    n_units = len(identity_rows)
    n_bins = max(0, edges.size - 1)
    if values.shape != (n_units, n_bins):
        raise ValueError(
            "Tuning values must have shape (n_units, n_position_bins); "
            f"got {values.shape}, expected {(n_units, n_bins)}."
        )
    centers = (edges[:-1] + edges[1:]) / 2.0
    graph_length_cm = _numeric_scalar(
        attrs.get("graph_length_cm"),
        name="graph_length_cm",
    )
    if graph_length_cm <= 0.0:
        raise ValueError("graph_length_cm must be positive.")
    counts = np.asarray(spike_counts, dtype=float).reshape(-1)
    if counts.shape != (n_units,):
        raise ValueError("Spike counts must align with the unit dimension.")
    stored_counts: np.ndarray
    if np.all(np.isfinite(counts)):
        if np.any(counts < 0.0) or not np.allclose(
            counts,
            np.rint(counts),
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("Finite spike counts must be non-negative integers.")
        stored_counts = counts.astype(np.int64)
    else:
        stored_counts = counts
    stable_ids = [str(row["stable_unit_id"]) for row in identity_rows]
    curve = xr.DataArray(
        values,
        dims=(UNIT_DIM, POSITION_DIM),
        coords={
            UNIT_DIM: (UNIT_DIM, np.asarray(stable_ids, dtype=str)),
            "spikesorting_merge_id": (
                UNIT_DIM,
                np.asarray(
                    [str(row["spikesorting_merge_id"]) for row in identity_rows],
                    dtype=str,
                ),
            ),
            "unit_id": (
                UNIT_DIM,
                np.asarray([str(row["unit_id"]) for row in identity_rows], dtype=str),
            ),
            "stable_unit_id": (UNIT_DIM, np.asarray(stable_ids, dtype=str)),
            "group_unit_id": (
                UNIT_DIM,
                np.asarray([str(row["group_unit_id"]) for row in identity_rows], dtype=str),
            ),
            SPIKE_COUNT_COORDINATE: (UNIT_DIM, stored_counts),
            POSITION_DIM: (POSITION_DIM, centers),
            PATH_FRACTION_COORDINATE: (
                POSITION_DIM,
                centers / graph_length_cm,
            ),
        },
        name="firing_rate_hz",
        attrs=dict(attrs),
    )
    curve.attrs["units"] = "Hz"
    curve.coords[POSITION_DIM].attrs["units"] = "cm"
    curve.coords[PATH_FRACTION_COORDINATE].attrs["units"] = "1"
    curve.coords[PATH_FRACTION_COORDINATE].attrs["definition"] = (
        "Centimeter bin center divided by graph_length_cm. Fixed-width "
        "binning retains the legacy padded final edge, so the last center may "
        "be slightly greater than 1."
    )
    curve.coords[SPIKE_COUNT_COORDINATE].attrs["definition"] = (
        "Spikes in this row's trial subset intersected with movement intervals."
    )
    return curve


def _curve_attributes(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    trial_subset: str,
    parameters: Mapping[str, Any],
    graph_length_cm: float,
    bin_edges_cm: np.ndarray,
    n_trials: int,
    support_duration_s: float,
    n_feature_samples: int,
    n_valid_position_samples: int,
    n_units: int,
    n_valid_units: int,
    analysis_status: str,
) -> dict[str, Any]:
    """Return NetCDF-safe canonical curve metadata."""
    attrs: dict[str, Any] = {
        "animal_name": str(animal_name),
        "date": str(date),
        "region": str(region),
        "epoch": str(epoch),
        "trajectory_type": str(trajectory_type),
        "trial_subset": validate_trial_subset(trial_subset),
        "binning_mode": str(parameters["binning_mode"]),
        "sigma_bins": float(parameters["sigma_bins"]),
        "graph_length_cm": float(graph_length_cm),
        "bin_edges_cm_json": json.dumps(
            np.asarray(bin_edges_cm, dtype=float).tolist(),
            separators=(",", ":"),
        ),
        "n_trials": int(n_trials),
        "support_duration_s": float(support_duration_s),
        "n_feature_samples": int(n_feature_samples),
        "n_valid_position_samples": int(n_valid_position_samples),
        "n_units": int(n_units),
        "n_valid_units": int(n_valid_units),
        "analysis_status": str(analysis_status),
    }
    if parameters["binning_mode"] == "bin_count":
        attrs["bin_count"] = int(parameters["bin_count"])
    else:
        attrs["bin_size_cm"] = float(parameters["bin_size_cm"])
    return attrs


def _terminal_curve(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    spike_counts: Sequence[float],
    bin_edges_cm: np.ndarray,
    attrs: Mapping[str, Any],
) -> Any:
    """Return one all-unit NaN curve for a terminal analysis status."""
    values = np.full(
        (len(identity_rows), len(np.asarray(bin_edges_cm).reshape(-1)) - 1),
        np.nan,
        dtype=float,
    )
    curve = _build_curve(
        values,
        identity_rows=identity_rows,
        spike_counts=spike_counts,
        bin_edges_cm=bin_edges_cm,
        attrs=attrs,
    )
    return validate_path_specific_place_tuning_curve(curve)


def _raw_tuning_curve(
    *,
    spikes: Any,
    linear_position: Any,
    support: Any,
    bin_edges_cm: np.ndarray,
) -> Any:
    """Compute one unsmoothed Pynapple tuning curve lazily."""
    import pynapple as nap

    return nap.compute_tuning_curves(
        data=spikes,
        features=linear_position,
        epochs=support,
        bins=[np.asarray(bin_edges_cm, dtype=float)],
        feature_names=[POSITION_DIM],
    )


def _aligned_raw_values(
    raw_curve: Any,
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    n_bins: int,
) -> np.ndarray:
    """Return raw tuning values in selected stable-unit order."""
    if len(getattr(raw_curve, "dims", ())) != 2:
        raise ValueError("Pynapple tuning output must be two-dimensional.")
    unit_dim = raw_curve.dims[0]
    raw_units = list(np.asarray(raw_curve.coords[unit_dim].values).reshape(-1))
    if len(raw_units) != len(set(raw_units)):
        raise ValueError("Pynapple tuning output contains duplicate unit ids.")
    raw_index = {
        unit_id.item() if isinstance(unit_id, np.generic) else unit_id: index
        for index, unit_id in enumerate(raw_units)
    }
    expected_keys = [row["_group_key"] for row in identity_rows]
    try:
        missing = [key for key in expected_keys if key not in raw_index]
        extra = [key for key in raw_index if key not in set(expected_keys)]
    except TypeError as exc:
        raise ValueError("Pynapple tuning unit identifiers must be hashable.") from exc
    if missing or extra:
        raise ValueError(
            "Pynapple tuning output units do not match the selected spike group: "
            f"missing={missing!r}, extra={extra!r}."
        )
    values = np.asarray(raw_curve.values, dtype=float)[
        [raw_index[key] for key in expected_keys]
    ]
    if values.shape != (len(identity_rows), n_bins):
        raise ValueError("Pynapple tuning output has an unexpected bin shape.")
    values[np.isinf(values)] = np.nan
    return values


def smooth_tuning_values(values: np.ndarray, *, sigma_bins: float) -> np.ndarray:
    """Apply the repository's NaN-aware non-circular Gaussian smoothing."""
    sigma = _numeric_scalar(sigma_bins, name="sigma_bins")
    if sigma < 0.0:
        raise ValueError("sigma_bins must be non-negative.")
    from v1ca1.raster.plot_place_field_heatmap import smooth_values_nan_aware

    return smooth_values_nan_aware(
        np.asarray(values, dtype=float),
        sigma_bins=sigma,
        axis=1,
        mode="nearest",
    )


def compute_selected_path_specific_place_tuning_curve(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    trial_subset: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_intervals: Any,
    graph_inputs: Mapping[str, Any],
    movement_intervals: Any,
    movement_analysis_status: str = "valid",
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
) -> dict[str, Any]:
    """Compute one all-unit tuning curve from already selected source objects."""
    parameters = validate_binning_parameters(
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=sigma_bins,
    )
    subset = validate_trial_subset(trial_subset)
    identities = _identity_rows(spikes, stable_unit_ids)
    movement_status = _validate_movement_analysis_status(
        movement_analysis_status,
        n_units=len(identities),
    )
    selected_trials = select_trial_subset_intervals(
        trajectory_intervals,
        subset,
    )
    n_trials, _trial_duration = _interval_summary(selected_trials)
    graph_length_cm = graph_length_from_inputs(
        graph_inputs,
        trajectory_type=trajectory_type,
    )
    bin_edges_cm = build_position_bin_edges(
        graph_length_cm,
        bin_size_cm=parameters["bin_size_cm"],
        bin_count=parameters["bin_count"],
    )
    support = _intersect_intervals(selected_trials, movement_intervals)
    _support_count, support_duration_s = _interval_summary(support)
    spike_counts = np.zeros(len(identities), dtype=np.int64)

    def terminal(
        status: str,
        *,
        n_feature_samples: int = 0,
        n_valid_position_samples: int = 0,
    ) -> dict[str, Any]:
        attrs = _curve_attributes(
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            trajectory_type=trajectory_type,
            trial_subset=subset,
            parameters=parameters,
            graph_length_cm=graph_length_cm,
            bin_edges_cm=bin_edges_cm,
            n_trials=n_trials,
            support_duration_s=support_duration_s,
            n_feature_samples=n_feature_samples,
            n_valid_position_samples=n_valid_position_samples,
            n_units=len(identities),
            n_valid_units=0,
            analysis_status=status,
        )
        return {
            "tuning_curve": _terminal_curve(
                identity_rows=identities,
                spike_counts=spike_counts,
                bin_edges_cm=bin_edges_cm,
                attrs=attrs,
            ),
            "analysis_status": status,
            "n_units": len(identities),
            "n_valid_units": 0,
            "n_trials": n_trials,
            "support_duration_s": support_duration_s,
            "n_feature_samples": n_feature_samples,
            "n_position_bins": len(bin_edges_cm) - 1,
        }

    if not identities:
        return terminal("no_units")
    if movement_status in {"no_valid_position", "no_movement"}:
        return terminal(movement_status)
    if n_trials == 0:
        return terminal("no_trials")
    if support_duration_s <= 0.0:
        return terminal("no_movement")

    spike_counts = _subset_spike_counts(
        spikes,
        support,
        n_units=len(identities),
    )

    linear_position, computed_graph_length = build_path_specific_linear_position(
        position=position,
        trajectory_intervals=trajectory_intervals,
        graph_inputs=graph_inputs,
        trajectory_type=trajectory_type,
    )
    if not np.isclose(
        computed_graph_length,
        graph_length_cm,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Linearization and WTrackGraph lengths disagree.")
    restrict = getattr(linear_position, "restrict", None)
    if not callable(restrict):
        raise TypeError("Linearized position must expose restrict().")
    restricted_position = restrict(support)
    restricted_values = np.asarray(
        getattr(restricted_position, "d", restricted_position),
        dtype=float,
    ).reshape(-1)
    n_feature_samples = int(restricted_values.size)
    n_valid_position_samples = int(np.sum(np.isfinite(restricted_values)))
    if n_valid_position_samples < 2:
        return terminal(
            "no_valid_position",
            n_feature_samples=n_feature_samples,
            n_valid_position_samples=n_valid_position_samples,
        )

    raw_curve = _raw_tuning_curve(
        spikes=spikes,
        linear_position=linear_position,
        support=support,
        bin_edges_cm=bin_edges_cm,
    )
    values = _aligned_raw_values(
        raw_curve,
        identity_rows=identities,
        n_bins=len(bin_edges_cm) - 1,
    )
    values = smooth_tuning_values(values, sigma_bins=parameters["sigma_bins"])
    n_valid_units = int(np.sum(np.any(np.isfinite(values), axis=1)))
    status = "valid" if n_valid_units else "no_valid_units"
    attrs = _curve_attributes(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        trajectory_type=trajectory_type,
        trial_subset=subset,
        parameters=parameters,
        graph_length_cm=graph_length_cm,
        bin_edges_cm=bin_edges_cm,
        n_trials=n_trials,
        support_duration_s=support_duration_s,
        n_feature_samples=n_feature_samples,
        n_valid_position_samples=n_valid_position_samples,
        n_units=len(identities),
        n_valid_units=n_valid_units,
        analysis_status=status,
    )
    curve = _build_curve(
        values,
        identity_rows=identities,
        spike_counts=spike_counts,
        bin_edges_cm=bin_edges_cm,
        attrs=attrs,
    )
    validate_path_specific_place_tuning_curve(curve)
    return {
        "tuning_curve": curve,
        "analysis_status": status,
        "n_units": len(identities),
        "n_valid_units": n_valid_units,
        "n_trials": n_trials,
        "support_duration_s": support_duration_s,
        "n_feature_samples": n_feature_samples,
        "n_position_bins": len(bin_edges_cm) - 1,
    }


def _parse_bin_edges(value: Any, *, name: str) -> np.ndarray:
    """Return one strictly increasing finite edge array."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not valid JSON.") from exc
    edges = np.asarray(value, dtype=float)
    if edges.ndim == 2 and edges.shape[0] == 1:
        edges = edges[0]
    edges = edges.reshape(-1)
    if edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{name} must contain finite, strictly increasing edges.")
    return edges


def validate_path_specific_place_tuning_curve(curve: Any) -> Any:
    """Validate and return one canonical tuning-curve DataArray."""
    if getattr(curve, "name", None) != "firing_rate_hz":
        raise ValueError("Tuning curve must be named 'firing_rate_hz'.")
    if tuple(getattr(curve, "dims", ())) != (UNIT_DIM, POSITION_DIM):
        raise ValueError(
            f"Tuning curve dimensions must be {(UNIT_DIM, POSITION_DIM)!r}."
        )
    for coordinate in (
        UNIT_DIM,
        POSITION_DIM,
        PATH_FRACTION_COORDINATE,
        SPIKE_COUNT_COORDINATE,
        *IDENTITY_COORDINATES,
    ):
        if coordinate not in curve.coords:
            raise ValueError(f"Tuning curve is missing coordinate {coordinate!r}.")
    values = np.asarray(curve.values, dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("Tuning-curve values may be finite or NaN, not infinite.")
    n_units, n_bins = values.shape
    stable_ids = np.asarray(curve.coords["stable_unit_id"].values).astype(str)
    unit_coordinate = np.asarray(curve.coords[UNIT_DIM].values).astype(str)
    merge_ids = np.asarray(curve.coords["spikesorting_merge_id"].values).astype(str)
    unit_ids = np.asarray(curve.coords["unit_id"].values).astype(str)
    group_ids = np.asarray(curve.coords["group_unit_id"].values).astype(str)
    spike_counts = np.asarray(
        curve.coords[SPIKE_COUNT_COORDINATE].values,
        dtype=float,
    )
    for name, coordinate in {
        "stable_unit_id": stable_ids,
        UNIT_DIM: unit_coordinate,
        "spikesorting_merge_id": merge_ids,
        "unit_id": unit_ids,
        "group_unit_id": group_ids,
    }.items():
        if coordinate.shape != (n_units,):
            raise ValueError(f"Coordinate {name!r} does not align with units.")
    expected_stable_ids = np.asarray(
        [f"{merge_id}:{unit_id}" for merge_id, unit_id in zip(merge_ids, unit_ids, strict=True)],
        dtype=str,
    )
    if not np.array_equal(stable_ids, expected_stable_ids) or not np.array_equal(
        unit_coordinate,
        stable_ids,
    ):
        raise ValueError("Tuning-curve stable unit identities are inconsistent.")
    if len(set(stable_ids.tolist())) != n_units or len(set(group_ids.tolist())) != n_units:
        raise ValueError("Tuning-curve unit coordinates must be unique.")
    if spike_counts.shape != (n_units,) or np.any(np.isinf(spike_counts)):
        raise ValueError("Spike counts must align with units and may not be infinite.")
    finite_spike_counts = spike_counts[np.isfinite(spike_counts)]
    if np.any(finite_spike_counts < 0.0) or not np.allclose(
        finite_spike_counts,
        np.rint(finite_spike_counts),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Finite spike counts must be non-negative integers.")
    legacy_normalized = str(curve.attrs.get("legacy_normalized", "false")).lower()
    if np.any(np.isnan(spike_counts)) and not (
        legacy_normalized == "true" and curve.attrs.get("trial_subset") == "all"
    ):
        raise ValueError(
            "Unknown spike counts are allowed only for legacy-normalized all-trial "
            "artifacts."
        )

    position = np.asarray(curve.coords[POSITION_DIM].values, dtype=float)
    if position.shape != (n_bins,) or not np.all(np.isfinite(position)):
        raise ValueError("Path-position coordinates must be finite and align with bins.")
    if position.size > 1 and np.any(np.diff(position) <= 0.0):
        raise ValueError("Path-position coordinates must be strictly increasing.")
    if curve.coords[POSITION_DIM].attrs.get("units") != "cm":
        raise ValueError("Path-position coordinate units must be centimeters.")
    path_fraction = np.asarray(
        curve.coords[PATH_FRACTION_COORDINATE].values,
        dtype=float,
    )
    if path_fraction.shape != (n_bins,) or not np.all(np.isfinite(path_fraction)):
        raise ValueError("path_fraction must be finite and align with position bins.")
    if curve.coords[PATH_FRACTION_COORDINATE].attrs.get("units") != "1":
        raise ValueError("path_fraction must be dimensionless.")
    if curve.attrs.get("units") != "Hz":
        raise ValueError("Tuning-curve value units must be Hz.")

    required_attrs = {
        "animal_name",
        "date",
        "region",
        "epoch",
        "trajectory_type",
        "trial_subset",
        "binning_mode",
        "sigma_bins",
        "graph_length_cm",
        "bin_edges_cm_json",
        "n_trials",
        "support_duration_s",
        "n_feature_samples",
        "n_valid_position_samples",
        "n_units",
        "n_valid_units",
        "analysis_status",
    }
    missing = sorted(required_attrs.difference(curve.attrs))
    if missing:
        raise ValueError(f"Tuning curve is missing attributes {missing!r}.")
    validate_trial_subset(curve.attrs["trial_subset"])
    mode = str(curve.attrs["binning_mode"])
    if mode not in BINNING_MODES:
        raise ValueError(f"Unknown binning_mode {mode!r}.")
    validate_binning_parameters(
        bin_size_cm=(curve.attrs.get("bin_size_cm") if mode == "bin_size_cm" else None),
        bin_count=(curve.attrs.get("bin_count") if mode == "bin_count" else None),
        sigma_bins=curve.attrs["sigma_bins"],
    )
    edges = _parse_bin_edges(
        curve.attrs["bin_edges_cm_json"],
        name="bin_edges_cm_json",
    )
    if edges.size != n_bins + 1 or not np.allclose(
        position,
        (edges[:-1] + edges[1:]) / 2.0,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Path-position coordinates do not match saved bin edges.")
    graph_length = _numeric_scalar(
        curve.attrs["graph_length_cm"],
        name="graph_length_cm",
    )
    if graph_length <= 0.0:
        raise ValueError("graph_length_cm must be positive.")
    if not np.allclose(
        path_fraction,
        position / graph_length,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("path_fraction does not match centimeter position.")
    for name in (
        "n_trials",
        "n_feature_samples",
        "n_valid_position_samples",
        "n_units",
        "n_valid_units",
    ):
        value = curve.attrs[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if int(curve.attrs["n_valid_position_samples"]) > int(
        curve.attrs["n_feature_samples"]
    ):
        raise ValueError("n_valid_position_samples cannot exceed n_feature_samples.")
    if int(curve.attrs["n_units"]) != n_units:
        raise ValueError("Saved n_units does not match the unit dimension.")
    n_valid_units = int(np.sum(np.any(np.isfinite(values), axis=1)))
    if int(curve.attrs["n_valid_units"]) != n_valid_units:
        raise ValueError("Saved n_valid_units does not match tuning values.")
    duration = _numeric_scalar(
        curve.attrs["support_duration_s"],
        name="support_duration_s",
    )
    if duration < 0.0:
        raise ValueError("support_duration_s must be non-negative.")
    status = str(curve.attrs["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Unknown analysis_status {status!r}.")
    n_trials = int(curve.attrs["n_trials"])
    n_valid_position_samples = int(
        curve.attrs["n_valid_position_samples"]
    )
    if status == "valid" and (
        n_units == 0
        or n_valid_units == 0
        or n_trials == 0
        or duration <= 0.0
        or n_valid_position_samples < 2
    ):
        raise ValueError(
            "Valid tuning curves require units, trials, positive support, at "
            "least two valid position samples, and at least one valid unit."
        )
    if status == "no_units" and n_units != 0:
        raise ValueError("no_units tuning curves must have no units.")
    if status == "no_trials" and (
        n_units == 0 or n_trials != 0 or duration != 0.0
    ):
        raise ValueError(
            "no_trials tuning curves require selected units, zero trials, and "
            "zero support duration."
        )
    if status == "no_movement" and (n_units == 0 or duration != 0.0):
        raise ValueError(
            "no_movement tuning curves require selected units and zero "
            "support duration."
        )
    if status == "no_valid_position" and (
        n_units == 0 or n_valid_position_samples >= 2
    ):
        raise ValueError(
            "no_valid_position tuning curves require selected units and fewer "
            "than two valid position samples."
        )
    if status == "no_valid_units" and (
        n_units == 0
        or n_trials == 0
        or duration <= 0.0
        or n_valid_position_samples < 2
    ):
        raise ValueError(
            "no_valid_units tuning curves require units, trials, positive "
            "support, and at least two valid position samples."
        )
    if status != "valid" and n_valid_units != 0:
        raise ValueError(f"{status} tuning curves cannot contain valid unit rows.")
    return curve


def _decoded_nwb_text(value: Any, *, name: str) -> str:
    """Return one UTF-8 string fetched from an NWB DynamicTable."""
    if isinstance(value, (bytes, np.bytes_)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} is not valid UTF-8.") from exc
    value = str(value)
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _nwb_table_frame(
    nwb_table: Any,
    *,
    expected_name: str,
    expected_columns: Sequence[str],
) -> Any:
    """Return one fetched DynamicTable as an exact-column DataFrame."""
    import pandas as pd
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table.copy()
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != expected_name:
            raise ValueError(
                f"Unexpected path-specific place NWB object name "
                f"{nwb_table.name!r}; expected {expected_name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError(
            "Path-specific place NWB objects must be DynamicTables or "
            "DataFrames."
        )
    observed_columns = tuple(str(column) for column in table.columns)
    if set(observed_columns) != set(expected_columns) or len(
        observed_columns
    ) != len(expected_columns):
        raise ValueError(
            f"{expected_name} must contain exactly columns "
            f"{tuple(expected_columns)!r}."
        )
    return table.loc[:, list(expected_columns)].reset_index(drop=True)


def _curve_metadata_payload(curve: Any) -> dict[str, Any]:
    """Return scalar and coordinate metadata needed to rebuild one curve."""
    validate_path_specific_place_tuning_curve(curve)
    return {
        "attrs": dict(curve.attrs),
        "coordinate_attrs": {
            str(name): dict(curve.coords[name].attrs)
            for name in (
                POSITION_DIM,
                PATH_FRACTION_COORDINATE,
                SPIKE_COUNT_COORDINATE,
            )
        },
    }


def path_specific_place_tuning_to_dynamic_table(curve: Any) -> Any:
    """Store unit identities, spike counts, and fixed-length tuning vectors."""
    import pandas as pd
    from hdmf.common import DynamicTable, VectorData

    canonical = validate_path_specific_place_tuning_curve(curve)
    n_units = int(canonical.sizes[UNIT_DIM])
    n_bins = int(canonical.sizes[POSITION_DIM])
    description = (
        "One row per selected unit with its path-specific firing-rate vector; "
        f"v1ca1 schema version {NWB_ARTIFACT_SCHEMA_VERSION}."
    )
    if n_units == 0:
        string_columns = {
            "spikesorting_merge_id",
            "unit_id",
            "stable_unit_id",
            "group_unit_id",
        }
        columns = []
        for name in _NWB_TUNING_COLUMNS:
            if name in string_columns:
                data = np.asarray([], dtype="S1")
            elif name == "curve_row":
                data = np.asarray([], dtype=np.int64)
            elif name == "firing_rate_hz":
                data = np.empty((0, n_bins), dtype=float)
            else:
                data = np.asarray([], dtype=float)
            columns.append(
                VectorData(
                    name=name,
                    description=(
                        "Zero-based row in the unit-by-position tuning matrix."
                        if name == "curve_row"
                        else "Canonical path-specific place-tuning field."
                    ),
                    data=data,
                )
            )
        return DynamicTable(
            name=NWB_TUNING_TABLE_NAME,
            description=description,
            columns=columns,
        )

    table = pd.DataFrame(
        {
            "curve_row": np.arange(n_units, dtype=np.int64),
            "spikesorting_merge_id": np.asarray(
                canonical.coords["spikesorting_merge_id"].values
            ).astype(str),
            "unit_id": np.asarray(canonical.coords["unit_id"].values).astype(str),
            "stable_unit_id": np.asarray(
                canonical.coords["stable_unit_id"].values
            ).astype(str),
            "group_unit_id": np.asarray(
                canonical.coords["group_unit_id"].values
            ).astype(str),
            SPIKE_COUNT_COORDINATE: np.asarray(
                canonical.coords[SPIKE_COUNT_COORDINATE].values,
                dtype=float,
            ),
            "firing_rate_hz": [
                np.asarray(row, dtype=float)
                for row in np.asarray(canonical.values, dtype=float)
            ],
        },
        columns=list(_NWB_TUNING_COLUMNS),
    )
    return DynamicTable.from_dataframe(
        name=NWB_TUNING_TABLE_NAME,
        df=table,
        table_description=description,
        columns=[
            {
                "name": name,
                "description": (
                    "Zero-based row in the unit-by-position tuning matrix."
                    if name == "curve_row"
                    else "Canonical path-specific place-tuning field."
                ),
            }
            for name in _NWB_TUNING_COLUMNS
        ],
    )


def path_specific_place_bins_to_dynamic_table(curve: Any) -> Any:
    """Store one typed row for every column of the tuning vectors."""
    import pandas as pd
    from hdmf.common import DynamicTable

    canonical = validate_path_specific_place_tuning_curve(curve)
    edges = _parse_bin_edges(
        canonical.attrs["bin_edges_cm_json"],
        name="bin_edges_cm_json",
    )
    n_bins = int(canonical.sizes[POSITION_DIM])
    table = pd.DataFrame(
        {
            "position_bin": np.arange(n_bins, dtype=np.int64),
            "left_edge_cm": edges[:-1],
            POSITION_DIM: np.asarray(
                canonical.coords[POSITION_DIM].values,
                dtype=float,
            ),
            "right_edge_cm": edges[1:],
            PATH_FRACTION_COORDINATE: np.asarray(
                canonical.coords[PATH_FRACTION_COORDINATE].values,
                dtype=float,
            ),
        },
        columns=list(_NWB_BIN_COLUMNS),
    )
    return DynamicTable.from_dataframe(
        name=NWB_BINS_TABLE_NAME,
        df=table,
        table_description=(
            "One row per path-position column of path_specific_place_tuning; "
            f"v1ca1 schema version {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=[
            {
                "name": name,
                "description": (
                    "Zero-based tuning-vector column."
                    if name == "position_bin"
                    else "Canonical path-position bin field."
                ),
            }
            for name in _NWB_BIN_COLUMNS
        ],
    )


def path_specific_place_provenance_to_dynamic_table(curve: Any) -> Any:
    """Store one canonical JSON metadata record for the complete curve."""
    import pandas as pd
    from hdmf.common import DynamicTable

    from v1ca1.spyglass.selection import canonical_json

    payload = _curve_metadata_payload(curve)
    table = pd.DataFrame(
        [
            {
                "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
                "metadata_json": canonical_json(payload),
            }
        ],
        columns=list(_NWB_PROVENANCE_COLUMNS),
    )
    return DynamicTable.from_dataframe(
        name=NWB_PROVENANCE_TABLE_NAME,
        df=table,
        table_description=(
            "One provenance record for path-specific place tuning; "
            f"v1ca1 schema version {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=[
            {
                "name": "artifact_schema_version",
                "description": "v1ca1 NWB artifact schema version.",
            },
            {
                "name": "metadata_json",
                "description": (
                    "Canonical JSON containing DataArray and coordinate metadata."
                ),
            },
        ],
    )


def _path_specific_place_nwb_frames(
    tuning: Any,
    bins: Any,
    provenance: Any,
) -> tuple[Any, Any, Any]:
    """Return the three canonical NWB table frames after structural checks."""
    tuning_table = _nwb_table_frame(
        tuning,
        expected_name=NWB_TUNING_TABLE_NAME,
        expected_columns=_NWB_TUNING_COLUMNS,
    )
    bins_table = _nwb_table_frame(
        bins,
        expected_name=NWB_BINS_TABLE_NAME,
        expected_columns=_NWB_BIN_COLUMNS,
    )
    provenance_table = _nwb_table_frame(
        provenance,
        expected_name=NWB_PROVENANCE_TABLE_NAME,
        expected_columns=_NWB_PROVENANCE_COLUMNS,
    )
    if len(provenance_table) != 1:
        raise ValueError(
            "Path-specific place provenance must contain exactly one row."
        )
    return tuning_table, bins_table, provenance_table


def path_specific_place_tuning_curve_from_nwb_objects(
    tuning: Any,
    bins: Any,
    provenance: Any,
) -> Any:
    """Reconstruct the canonical DataArray from three fetched NWB objects."""
    tuning_table, bins_table, provenance_table = _path_specific_place_nwb_frames(
        tuning,
        bins,
        provenance,
    )
    schema_version = _decoded_nwb_text(
        provenance_table.loc[0, "artifact_schema_version"],
        name="artifact_schema_version",
    )
    if schema_version != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Path-specific place NWB artifact schema version is unsupported."
        )
    metadata_json = _decoded_nwb_text(
        provenance_table.loc[0, "metadata_json"],
        name="metadata_json",
    )
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Path-specific place provenance is not valid JSON.") from exc
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "attrs",
        "coordinate_attrs",
    }:
        raise ValueError("Path-specific place provenance has an invalid schema.")
    attrs = dict(metadata["attrs"])
    coordinate_attrs = dict(metadata["coordinate_attrs"])

    n_units = len(tuning_table)
    n_bins = len(bins_table)
    curve_rows = np.asarray(tuning_table["curve_row"], dtype=np.int64)
    if not np.array_equal(curve_rows, np.arange(n_units, dtype=np.int64)):
        raise ValueError("Path-specific tuning curve_row values must be consecutive.")
    bin_rows = np.asarray(bins_table["position_bin"], dtype=np.int64)
    if not np.array_equal(bin_rows, np.arange(n_bins, dtype=np.int64)):
        raise ValueError("Path-specific position_bin values must be consecutive.")
    if n_bins == 0:
        raise ValueError("Path-specific place tuning requires at least one bin.")

    left_edges = np.asarray(bins_table["left_edge_cm"], dtype=float)
    right_edges = np.asarray(bins_table["right_edge_cm"], dtype=float)
    if (
        not np.all(np.isfinite(left_edges))
        or not np.all(np.isfinite(right_edges))
        or np.any(right_edges <= left_edges)
        or not np.allclose(
            left_edges[1:],
            right_edges[:-1],
            rtol=1e-10,
            atol=1e-12,
        )
    ):
        raise ValueError("Path-specific place bin edges are invalid or discontinuous.")
    edges = np.concatenate((left_edges[:1], right_edges))

    if n_units:
        tuning_rows = [
            np.asarray(value, dtype=float).reshape(-1)
            for value in tuning_table["firing_rate_hz"]
        ]
        if any(row.shape != (n_bins,) for row in tuning_rows):
            raise ValueError(
                "Every path-specific tuning vector must align with the bin table."
            )
        values = np.stack(tuning_rows, axis=0)
    else:
        values = np.empty((0, n_bins), dtype=float)
    identity_rows = [
        {
            "spikesorting_merge_id": _decoded_nwb_text(
                row.spikesorting_merge_id,
                name="spikesorting_merge_id",
            ),
            "unit_id": _decoded_nwb_text(row.unit_id, name="unit_id"),
            "stable_unit_id": _decoded_nwb_text(
                row.stable_unit_id,
                name="stable_unit_id",
            ),
            "group_unit_id": _decoded_nwb_text(
                row.group_unit_id,
                name="group_unit_id",
            ),
        }
        for row in tuning_table.itertuples(index=False)
    ]
    curve = _build_curve(
        values,
        identity_rows=identity_rows,
        spike_counts=np.asarray(tuning_table[SPIKE_COUNT_COORDINATE], dtype=float),
        bin_edges_cm=edges,
        attrs=attrs,
    )
    observed_position = np.asarray(
        bins_table[POSITION_DIM],
        dtype=float,
    )
    observed_fraction = np.asarray(
        bins_table[PATH_FRACTION_COORDINATE],
        dtype=float,
    )
    if not np.allclose(
        observed_position,
        np.asarray(curve.coords[POSITION_DIM].values, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ) or not np.allclose(
        observed_fraction,
        np.asarray(curve.coords[PATH_FRACTION_COORDINATE].values, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "Path-specific place bin metadata disagrees with curve provenance."
        )
    for coordinate_name, values_by_name in coordinate_attrs.items():
        if coordinate_name not in curve.coords or not isinstance(
            values_by_name,
            Mapping,
        ):
            raise ValueError(
                "Path-specific place coordinate provenance is invalid."
            )
        curve.coords[coordinate_name].attrs.update(dict(values_by_name))
    return validate_path_specific_place_tuning_curve(curve)


def _normalized_float_values(values: Any) -> list[Any]:
    """Return floats with NaN represented canonically for semantic hashing."""
    normalized = []
    for value in np.asarray(values, dtype=float).reshape(-1):
        normalized.append(None if np.isnan(value) else float(value))
    return normalized


def path_specific_place_tuning_sha256(curve: Any) -> str:
    """Digest unit metadata and tuning values independently of NWB storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    canonical = validate_path_specific_place_tuning_curve(curve)
    values = np.asarray(canonical.values, dtype=float)
    records = []
    for row_index in range(int(canonical.sizes[UNIT_DIM])):
        records.append(
            {
                "curve_row": row_index,
                "spikesorting_merge_id": str(
                    canonical.coords["spikesorting_merge_id"].values[row_index]
                ),
                "unit_id": str(canonical.coords["unit_id"].values[row_index]),
                "stable_unit_id": str(
                    canonical.coords["stable_unit_id"].values[row_index]
                ),
                "group_unit_id": str(
                    canonical.coords["group_unit_id"].values[row_index]
                ),
                SPIKE_COUNT_COORDINATE: _normalized_float_values(
                    [canonical.coords[SPIKE_COUNT_COORDINATE].values[row_index]]
                )[0],
                "firing_rate_hz": _normalized_float_values(values[row_index]),
            }
        )
    return provenance_sha256(
        {"columns": list(_NWB_TUNING_COLUMNS), "records": records}
    )


def path_specific_place_bins_sha256(curve: Any) -> str:
    """Digest ordered path-bin metadata independently of NWB storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    canonical = validate_path_specific_place_tuning_curve(curve)
    edges = _parse_bin_edges(
        canonical.attrs["bin_edges_cm_json"],
        name="bin_edges_cm_json",
    )
    return provenance_sha256(
        {
            "position_bin": list(range(int(canonical.sizes[POSITION_DIM]))),
            "left_edge_cm": edges[:-1].tolist(),
            POSITION_DIM: np.asarray(
                canonical.coords[POSITION_DIM].values,
                dtype=float,
            ).tolist(),
            "right_edge_cm": edges[1:].tolist(),
            PATH_FRACTION_COORDINATE: np.asarray(
                canonical.coords[PATH_FRACTION_COORDINATE].values,
                dtype=float,
            ).tolist(),
        }
    )


def path_specific_place_provenance_sha256(curve: Any) -> str:
    """Digest the canonical scalar and coordinate provenance payload."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
            "metadata": _curve_metadata_payload(curve),
        }
    )


def path_specific_place_tuning_curve_sha256(curve: Any) -> str:
    """Digest all three logical curve objects as one scientific artifact."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            "path_specific_place_tuning_sha256": (
                path_specific_place_tuning_sha256(curve)
            ),
            "path_specific_place_bins_sha256": (
                path_specific_place_bins_sha256(curve)
            ),
            "path_specific_place_provenance_sha256": (
                path_specific_place_provenance_sha256(curve)
            ),
        }
    )


def load_path_specific_place_artifact(path: Path) -> Any:
    """Load and validate one canonical NetCDF tuning curve."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Path-specific place artifact not found: {path}")
    import xarray as xr

    with xr.open_dataarray(path) as opened:
        curve = opened.load()
    return validate_path_specific_place_tuning_curve(curve)


def write_path_specific_place_artifact(
    curve: Any,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated NetCDF without implicit overwrite."""
    validate_path_specific_place_tuning_curve(curve)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite path-specific place artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.nc")
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.backup")
    had_existing = path.exists()
    try:
        curve.to_netcdf(temporary)
        load_path_specific_place_artifact(temporary)
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


def _legacy_identity_rows(
    legacy_unit_ids: Sequence[Any],
    resolver: Mapping[Any, Mapping[str, Any]] | Callable[[Any], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Return selected identities and legacy row indices in group order."""
    selected: list[tuple[int, Any, Mapping[str, Any]]] = []
    if isinstance(resolver, Mapping):
        used_indices: set[int] = set()
        for selected_legacy_id, identity in resolver.items():
            matching_indices = [
                index
                for index, legacy_id in enumerate(legacy_unit_ids)
                if str(legacy_id) == str(selected_legacy_id)
            ]
            if len(matching_indices) != 1:
                raise ValueError(
                    f"Selected legacy unit {selected_legacy_id!r} must appear "
                    "exactly once in the legacy artifact."
                )
            legacy_index = matching_indices[0]
            if legacy_index in used_indices:
                raise ValueError("Selected legacy unit ids must be unique.")
            used_indices.add(legacy_index)
            selected.append(
                (legacy_index, legacy_unit_ids[legacy_index], identity)
            )
    elif callable(resolver):
        for legacy_index, legacy_id in enumerate(legacy_unit_ids):
            try:
                identity = resolver(legacy_id)
            except LookupError:
                continue
            selected.append((legacy_index, legacy_id, identity))
    else:
        raise TypeError("unit_identity_resolver must be a mapping or callable.")

    rows: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    stable_ids: set[str] = set()
    group_ids: set[str] = set()
    for legacy_index, legacy_id, resolved_identity in selected:
        if not isinstance(resolved_identity, Mapping):
            raise TypeError("Resolved legacy unit identities must be mappings.")
        identity = dict(resolved_identity)
        missing = [
            name
            for name in ("spikesorting_merge_id", "unit_id")
            if name not in identity
        ]
        if missing:
            raise ValueError(
                f"Resolved legacy unit identity is missing fields {missing!r}."
            )
        if "sorting_unit_id" in identity and str(identity["sorting_unit_id"]) != str(
            legacy_id
        ):
            raise ValueError(
                f"Resolved sorting_unit_id does not match legacy unit {legacy_id!r}."
            )
        merge_id = str(identity["spikesorting_merge_id"])
        unit_id = str(identity["unit_id"])
        stable_id = f"{merge_id}:{unit_id}"
        group_id = str(identity.get("group_unit_id", legacy_id))
        if not merge_id or not unit_id or stable_id in stable_ids:
            raise ValueError("Resolved legacy identities must be non-empty and unique.")
        if group_id in group_ids:
            raise ValueError("Resolved group_unit_id values must be unique.")
        stable_ids.add(stable_id)
        group_ids.add(group_id)
        selected_indices.append(legacy_index)
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": stable_id,
                "group_unit_id": group_id,
                "_group_key": identity.get("group_unit_id", legacy_id),
            }
        )
    return rows, np.asarray(selected_indices, dtype=int)


def normalize_legacy_all_trial_tuning_curve(
    legacy_curve: Any,
    *,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    graph_length_cm: float,
    n_trials: int,
    support_duration_s: float,
    n_feature_samples: int,
    n_valid_position_samples: int | None = None,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
) -> Any:
    """Normalize one legacy all-trial NetCDF to stable unit coordinates."""
    parameters = validate_binning_parameters(
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=sigma_bins,
    )
    if parameters["sigma_bins"] != 0.0:
        raise ValueError("Legacy tuning curves can only register with sigma_bins=0.")
    if getattr(legacy_curve, "name", None) != "firing_rate_hz":
        raise ValueError("Legacy tuning curve must be named 'firing_rate_hz'.")
    if tuple(getattr(legacy_curve, "dims", ())) != ("unit", "linpos"):
        raise ValueError("Legacy place tuning curve dimensions must be ('unit', 'linpos').")
    expected_attrs = {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
        "trajectory_type": trajectory_type,
        "model_name": "place",
    }
    for name, expected in expected_attrs.items():
        if str(legacy_curve.attrs.get(name, "")) != str(expected):
            raise ValueError(
                f"Legacy tuning curve {name}={legacy_curve.attrs.get(name)!r}, "
                f"expected {expected!r}."
            )
    legacy_units = list(np.asarray(legacy_curve.coords["unit"].values).reshape(-1))
    if len(legacy_units) != len(set(legacy_units)):
        raise ValueError("Legacy tuning curve contains duplicate unit ids.")
    identities, selected_indices = _legacy_identity_rows(
        legacy_units,
        unit_identity_resolver,
    )
    legacy_values = np.asarray(legacy_curve.values, dtype=float)
    if legacy_values.shape[0] != len(legacy_units) or np.any(np.isinf(legacy_values)):
        raise ValueError("Legacy tuning values must align with units and contain no infinities.")
    values = legacy_values[selected_indices]
    if "bin_edges" not in legacy_curve.attrs:
        raise ValueError("Legacy tuning curve is missing bin_edges metadata.")
    edges = _parse_bin_edges(legacy_curve.attrs["bin_edges"], name="legacy bin_edges")
    expected_edges = build_position_bin_edges(
        graph_length_cm,
        bin_size_cm=parameters["bin_size_cm"],
        bin_count=parameters["bin_count"],
    )
    if edges.shape != expected_edges.shape or not np.allclose(
        edges,
        expected_edges,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Legacy bin_edges do not match selected binning parameters.")
    legacy_position = np.asarray(legacy_curve.coords["linpos"].values, dtype=float)
    if legacy_position.shape != (edges.size - 1,) or not np.allclose(
        legacy_position,
        (edges[:-1] + edges[1:]) / 2.0,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Legacy linpos coordinates do not match bin_edges.")
    n_trials_value = int(n_trials)
    if isinstance(n_trials, bool) or n_trials_value < 0 or n_trials_value != n_trials:
        raise ValueError("n_trials must be a non-negative integer.")
    support_duration = _numeric_scalar(
        support_duration_s,
        name="support_duration_s",
    )
    if support_duration < 0.0:
        raise ValueError("support_duration_s must be non-negative.")
    n_feature_samples_value = int(n_feature_samples)
    if (
        isinstance(n_feature_samples, bool)
        or n_feature_samples_value < 0
        or n_feature_samples_value != n_feature_samples
    ):
        raise ValueError("n_feature_samples must be a non-negative integer.")
    if n_valid_position_samples is None:
        n_valid_position_samples = n_feature_samples_value
    n_valid_position_samples_value = int(n_valid_position_samples)
    if (
        isinstance(n_valid_position_samples, bool)
        or n_valid_position_samples_value < 0
        or n_valid_position_samples_value != n_valid_position_samples
        or n_valid_position_samples_value > n_feature_samples_value
    ):
        raise ValueError(
            "n_valid_position_samples must be a non-negative integer no larger "
            "than n_feature_samples."
        )
    n_valid_units = int(np.sum(np.any(np.isfinite(values), axis=1)))
    status = (
        "no_units"
        if not identities
        else "valid"
        if n_valid_units
        else "no_valid_units"
    )
    attrs = _curve_attributes(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        trajectory_type=trajectory_type,
        trial_subset="all",
        parameters=parameters,
        graph_length_cm=graph_length_cm,
        bin_edges_cm=edges,
        n_trials=n_trials_value,
        support_duration_s=support_duration,
        n_feature_samples=n_feature_samples_value,
        n_valid_position_samples=n_valid_position_samples_value,
        n_units=len(identities),
        n_valid_units=n_valid_units,
        analysis_status=status,
    )
    attrs["legacy_normalized"] = "true"
    curve = _build_curve(
        values,
        identity_rows=identities,
        spike_counts=np.full(len(identities), np.nan, dtype=float),
        bin_edges_cm=edges,
        attrs=attrs,
    )
    return validate_path_specific_place_tuning_curve(curve)


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one existing file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_existing_path_specific_place_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    graph_length_cm: float,
    n_trials: int,
    support_duration_s: float,
    n_feature_samples: int,
    n_valid_position_samples: int | None = None,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
    artifact_attributes: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize and atomically register one legacy all-trial NetCDF."""
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy tuning curve not found: {source}")
    if source.resolve() == destination.resolve(strict=False):
        raise ValueError("Legacy source and canonical destination must differ.")
    import xarray as xr

    with xr.open_dataarray(source) as opened:
        legacy_curve = opened.load()
    curve = normalize_legacy_all_trial_tuning_curve(
        legacy_curve,
        unit_identity_resolver=unit_identity_resolver,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        trajectory_type=trajectory_type,
        graph_length_cm=graph_length_cm,
        n_trials=n_trials,
        support_duration_s=support_duration_s,
        n_feature_samples=n_feature_samples,
        n_valid_position_samples=n_valid_position_samples,
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=sigma_bins,
    )
    curve.attrs.update(
        {
            str(name): str(value)
            for name, value in dict(artifact_attributes or {}).items()
        }
    )
    written = write_path_specific_place_artifact(
        curve,
        destination,
        overwrite=overwrite,
    )
    return {
        "tuning_curve": curve,
        "tuning_curve_path": written,
        "analysis_status": str(curve.attrs["analysis_status"]),
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "support_duration_s": float(curve.attrs["support_duration_s"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[POSITION_DIM]),
        "legacy_artifact_provenance": {
            "source_path": str(source.resolve(strict=True)),
            "source_sha256": _file_sha256(source),
            "legacy_unit_coordinate": "sorting_unit_id",
        },
        "_created_artifact_paths": [str(written)],
    }


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "ARTIFACT_FILENAME",
    "BINNING_MODES",
    "DEFAULT_ARTIFACT_ROOT",
    "IDENTITY_COORDINATES",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "NWB_BINS_TABLE_NAME",
    "NWB_PROVENANCE_TABLE_NAME",
    "NWB_TUNING_TABLE_NAME",
    "PATH_FRACTION_COORDINATE",
    "POSITION_DIM",
    "SPIKE_COUNT_COORDINATE",
    "TRIAL_SUBSETS",
    "UNIT_DIM",
    "build_path_specific_linear_position",
    "build_position_bin_edges",
    "compute_selected_path_specific_place_tuning_curve",
    "get_path_specific_place_artifact_path",
    "graph_length_from_inputs",
    "load_path_specific_place_artifact",
    "normalize_legacy_all_trial_tuning_curve",
    "path_specific_place_bins_sha256",
    "path_specific_place_bins_to_dynamic_table",
    "path_specific_place_provenance_sha256",
    "path_specific_place_provenance_to_dynamic_table",
    "path_specific_place_tuning_curve_from_nwb_objects",
    "path_specific_place_tuning_curve_sha256",
    "path_specific_place_tuning_sha256",
    "path_specific_place_tuning_to_dynamic_table",
    "register_existing_path_specific_place_artifact",
    "select_trial_subset_intervals",
    "smooth_tuning_values",
    "validate_binning_parameters",
    "validate_path_specific_place_tuning_curve",
    "validate_trial_subset",
    "write_path_specific_place_artifact",
]
