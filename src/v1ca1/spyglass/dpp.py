"""Directional path-progression tuning and artifact adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from numbers import Integral
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from v1ca1.helper.session import TURN_TRAJECTORY_PAIRS
from v1ca1.spyglass import path_specific_place as place


DEFAULT_ARTIFACT_ROOT = Path("/stelmo/nwb/analysis/kyu/v1ca1")
ARTIFACT_DIRNAME = "dpp_tuning_curve"
ARTIFACT_FILENAME = "tuning_curve.nc"
TURN_TYPES = tuple(TURN_TRAJECTORY_PAIRS)
TRIAL_SUBSETS = place.TRIAL_SUBSETS
ANALYSIS_STATUSES = place.ANALYSIS_STATUSES
UNIT_DIM = "unit"
DPP_DIM = "dpp"
PATH_FRACTION_COORDINATE = "path_fraction"
LINEAR_POSITION_COORDINATE = "linear_position_cm"
SPIKE_COUNT_COORDINATE = place.SPIKE_COUNT_COORDINATE
IDENTITY_COORDINATES = place.IDENTITY_COORDINATES
NWB_ARTIFACT_SCHEMA_VERSION = "1"
NWB_TUNING_TABLE_NAME = "dpp_tuning"
NWB_BINS_TABLE_NAME = "dpp_bins"
NWB_PROVENANCE_TABLE_NAME = "dpp_provenance"

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
    "dpp_bin",
    "left_edge_dpp",
    DPP_DIM,
    "right_edge_dpp",
    PATH_FRACTION_COORDINATE,
    "left_edge_cm",
    LINEAR_POSITION_COORDINATE,
    "right_edge_cm",
)
_NWB_PROVENANCE_COLUMNS = (
    "artifact_schema_version",
    "metadata_json",
)


def validate_turn_type(turn_type: str) -> str:
    """Return one supported same-turn directional-progression label."""
    value = str(turn_type)
    if value not in TURN_TRAJECTORY_PAIRS:
        raise ValueError(f"turn_type must be one of {TURN_TYPES!r}.")
    return value


def get_dpp_trajectory_pair(turn_type: str) -> tuple[str, str]:
    """Return the fixed ordered trajectory pair for one DPP turn type."""
    return tuple(TURN_TRAJECTORY_PAIRS[validate_turn_type(turn_type)])


def get_dpp_artifact_path(
    *,
    animal_name: str,
    date: str,
    epoch: str,
    turn_type: str,
    trial_subset: str,
    region: str,
    dpp_tuning_curve_id: Any,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Path:
    """Return one UUID-keyed, session-first DPP tuning-curve path."""
    components = {
        name: place._path_component(value, name=name)
        for name, value in {
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "turn_type": validate_turn_type(turn_type),
            "trial_subset": place.validate_trial_subset(trial_subset),
            "region": region,
        }.items()
    }
    selection_id = place._uuid_component(
        dpp_tuning_curve_id,
        name="dpp_tuning_curve_id",
    )
    return (
        Path(artifact_root)
        / components["animal_name"]
        / components["date"]
        / ARTIFACT_DIRNAME
        / components["epoch"]
        / components["turn_type"]
        / components["trial_subset"]
        / components["region"]
        / selection_id
        / ARTIFACT_FILENAME
    )


def _pair_mapping(
    values: Mapping[str, Any],
    *,
    turn_type: str,
    name: str,
) -> dict[str, Any]:
    """Return values for exactly the two trajectories needed by one DPP row."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by trajectory type.")
    pair = get_dpp_trajectory_pair(turn_type)
    missing = [trajectory for trajectory in pair if trajectory not in values]
    if missing:
        raise ValueError(f"{name} is missing trajectories {missing!r}.")
    return {trajectory: values[trajectory] for trajectory in pair}


def common_graph_length_from_inputs(
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    *,
    turn_type: str,
) -> tuple[float, dict[str, float]]:
    """Validate paired graph lengths and return the fixed first-path length.

    The legacy DPP binning converts centimeters to normalized progression with
    one W-track path length. The first trajectory in the repository's fixed
    pair is the deterministic canonical source, and the other must match it.
    """
    graphs = _pair_mapping(
        graph_inputs_by_trajectory,
        turn_type=turn_type,
        name="graph_inputs_by_trajectory",
    )
    lengths = {
        trajectory: place.graph_length_from_inputs(
            graph,
            trajectory_type=trajectory,
        )
        for trajectory, graph in graphs.items()
    }
    pair = get_dpp_trajectory_pair(turn_type)
    canonical = float(lengths[pair[0]])
    if not np.isclose(
        canonical,
        float(lengths[pair[1]]),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(
            "DPP requires the two trajectory graphs to have one common path "
            f"length; got {lengths!r}."
        )
    return canonical, lengths


def build_dpp_bin_edges(
    common_graph_length_cm: float,
    *,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
) -> np.ndarray:
    """Return normalized DPP edges using the shared tuning parameters."""
    parameters = place.validate_binning_parameters(
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=0.0,
    )
    graph_length = place._numeric_scalar(
        common_graph_length_cm,
        name="common_graph_length_cm",
    )
    if graph_length <= 0.0:
        raise ValueError("common_graph_length_cm must be positive.")
    if parameters["binning_mode"] == "bin_count":
        return np.linspace(
            0.0,
            1.0,
            int(parameters["bin_count"]) + 1,
            dtype=float,
        )
    step = float(parameters["bin_size_cm"]) / graph_length
    return np.arange(0.0, 1.0 + step, step, dtype=float)


def _pool_interval_sets(intervals_by_trajectory: Mapping[str, Any]) -> Any:
    """Return the sorted, non-overlapping union of paired interval rows."""
    start_chunks: list[np.ndarray] = []
    end_chunks: list[np.ndarray] = []
    for intervals in intervals_by_trajectory.values():
        starts, ends = place._extract_interval_bounds(intervals)
        if starts.size:
            start_chunks.append(starts)
            end_chunks.append(ends)
    if not start_chunks:
        return place._make_interval_set(
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    starts = np.concatenate(start_chunks)
    ends = np.concatenate(end_chunks)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError("Paired DPP trajectory intervals must not overlap.")
    return place._make_interval_set(starts, ends)


def select_dpp_trial_intervals(
    trajectory_intervals_by_type: Mapping[str, Any],
    *,
    turn_type: str,
    trial_subset: str,
) -> tuple[dict[str, Any], Any]:
    """Split trials within each trajectory, then return their pooled support."""
    subset = place.validate_trial_subset(trial_subset)
    intervals = _pair_mapping(
        trajectory_intervals_by_type,
        turn_type=turn_type,
        name="trajectory_intervals_by_type",
    )
    selected = {
        trajectory: place.select_trial_subset_intervals(value, subset)
        for trajectory, value in intervals.items()
    }
    return selected, _pool_interval_sets(selected)


def _source_integer_mapping(
    values: Mapping[str, Any],
    *,
    turn_type: str,
    name: str,
) -> dict[str, int]:
    """Validate one non-negative integer mapping for the paired trajectories."""
    paired = _pair_mapping(values, turn_type=turn_type, name=name)
    output: dict[str, int] = {}
    for trajectory, value in paired.items():
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{name}[{trajectory!r}] must be a non-negative integer.")
        output[trajectory] = int(value)
    return output


def _source_float_mapping(
    values: Mapping[str, Any],
    *,
    turn_type: str,
    name: str,
) -> dict[str, float]:
    """Validate one non-negative numeric mapping for the paired trajectories."""
    paired = _pair_mapping(values, turn_type=turn_type, name=name)
    output: dict[str, float] = {}
    for trajectory, value in paired.items():
        numeric = place._numeric_scalar(value, name=f"{name}[{trajectory!r}]")
        if numeric < 0.0:
            raise ValueError(f"{name}[{trajectory!r}] must be non-negative.")
        output[trajectory] = numeric
    return output


def _json_mapping(values: Mapping[str, Any]) -> str:
    """Return one compact, deterministic JSON object for NetCDF metadata."""
    return json.dumps(dict(values), sort_keys=True, separators=(",", ":"))


def _curve_attributes(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    turn_type: str,
    trial_subset: str,
    parameters: Mapping[str, Any],
    common_graph_length_cm: float,
    graph_length_cm_by_trajectory: Mapping[str, float],
    bin_edges_dpp: np.ndarray,
    n_trials_by_trajectory: Mapping[str, int],
    support_duration_s_by_trajectory: Mapping[str, float],
    n_feature_samples_by_trajectory: Mapping[str, int],
    n_valid_position_samples_by_trajectory: Mapping[str, int],
    n_units: int,
    n_valid_units: int,
    analysis_status: str,
) -> dict[str, Any]:
    """Return canonical NetCDF-safe DPP tuning metadata."""
    pair = get_dpp_trajectory_pair(turn_type)
    edges = np.asarray(bin_edges_dpp, dtype=float)
    attrs: dict[str, Any] = {
        "animal_name": str(animal_name),
        "date": str(date),
        "region": str(region),
        "epoch": str(epoch),
        "turn_type": validate_turn_type(turn_type),
        "trial_subset": place.validate_trial_subset(trial_subset),
        "outbound_trajectory_type": pair[0],
        "inbound_trajectory_type": pair[1],
        "source_trajectory_types_json": json.dumps(pair, separators=(",", ":")),
        "binning_mode": str(parameters["binning_mode"]),
        "sigma_bins": float(parameters["sigma_bins"]),
        "common_graph_length_cm": float(common_graph_length_cm),
        "graph_length_cm_by_trajectory_json": _json_mapping(
            graph_length_cm_by_trajectory
        ),
        "bin_edges_dpp_json": json.dumps(edges.tolist(), separators=(",", ":")),
        "bin_edges_cm_json": json.dumps(
            (edges * float(common_graph_length_cm)).tolist(),
            separators=(",", ":"),
        ),
        "n_trials_by_trajectory_json": _json_mapping(n_trials_by_trajectory),
        "support_duration_s_by_trajectory_json": _json_mapping(
            support_duration_s_by_trajectory
        ),
        "n_feature_samples_by_trajectory_json": _json_mapping(
            n_feature_samples_by_trajectory
        ),
        "n_valid_position_samples_by_trajectory_json": _json_mapping(
            n_valid_position_samples_by_trajectory
        ),
        "n_trials": int(sum(n_trials_by_trajectory.values())),
        "n_outbound_trials": int(n_trials_by_trajectory[pair[0]]),
        "n_inbound_trials": int(n_trials_by_trajectory[pair[1]]),
        "support_duration_s": float(sum(support_duration_s_by_trajectory.values())),
        "n_feature_samples": int(sum(n_feature_samples_by_trajectory.values())),
        "n_valid_position_samples": int(
            sum(n_valid_position_samples_by_trajectory.values())
        ),
        "n_units": int(n_units),
        "n_valid_units": int(n_valid_units),
        "analysis_status": str(analysis_status),
        "pooling_policy": (
            "Split one-indexed odd/even trials independently within each source "
            "trajectory; intersect each with movement; pool raw normalized DPP "
            "samples and support before one tuning estimate."
        ),
    }
    if parameters["binning_mode"] == "bin_count":
        attrs["bin_count"] = int(parameters["bin_count"])
    else:
        attrs["bin_size_cm"] = float(parameters["bin_size_cm"])
    return attrs


def _build_curve(
    values: np.ndarray,
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    spike_counts: Sequence[float],
    bin_edges_dpp: np.ndarray,
    attrs: Mapping[str, Any],
) -> Any:
    """Build one canonical stable-identity DPP tuning DataArray."""
    import xarray as xr

    edges = np.asarray(bin_edges_dpp, dtype=float).reshape(-1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    values = np.asarray(values, dtype=float)
    n_units = len(identity_rows)
    if values.shape != (n_units, len(centers)):
        raise ValueError("DPP tuning values do not match unit and bin dimensions.")
    counts = np.asarray(spike_counts, dtype=float).reshape(-1)
    if counts.shape != (n_units,):
        raise ValueError("Spike counts must align with the unit dimension.")
    if np.all(np.isfinite(counts)):
        if np.any(counts < 0.0) or not np.allclose(counts, np.rint(counts)):
            raise ValueError("Finite spike counts must be non-negative integers.")
        counts = counts.astype(np.int64)
    stable_ids = [str(row["stable_unit_id"]) for row in identity_rows]
    common_length = float(attrs["common_graph_length_cm"])
    curve = xr.DataArray(
        values,
        dims=(UNIT_DIM, DPP_DIM),
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
            SPIKE_COUNT_COORDINATE: (UNIT_DIM, counts),
            DPP_DIM: (DPP_DIM, centers),
            PATH_FRACTION_COORDINATE: (DPP_DIM, centers),
            LINEAR_POSITION_COORDINATE: (DPP_DIM, centers * common_length),
        },
        name="firing_rate_hz",
        attrs=dict(attrs),
    )
    curve.attrs["units"] = "Hz"
    curve.coords[DPP_DIM].attrs["units"] = "1"
    curve.coords[PATH_FRACTION_COORDINATE].attrs["units"] = "1"
    curve.coords[LINEAR_POSITION_COORDINATE].attrs["units"] = "cm"
    curve.coords[SPIKE_COUNT_COORDINATE].attrs["definition"] = (
        "Spikes in the pooled paired-trajectory trial subset intersected with "
        "movement intervals."
    )
    return curve


def _terminal_curve(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    spike_counts: Sequence[float],
    bin_edges_dpp: np.ndarray,
    attrs: Mapping[str, Any],
) -> Any:
    """Return an all-NaN curve for one terminal analysis status."""
    curve = _build_curve(
        np.full(
            (len(identity_rows), len(np.asarray(bin_edges_dpp).reshape(-1)) - 1),
            np.nan,
            dtype=float,
        ),
        identity_rows=identity_rows,
        spike_counts=spike_counts,
        bin_edges_dpp=bin_edges_dpp,
        attrs=attrs,
    )
    return validate_dpp_tuning_curve(curve)


def _build_trajectory_progression(
    *,
    position: Any,
    trajectory_intervals: Any,
    graph_inputs: Mapping[str, Any],
    trajectory_type: str,
) -> tuple[Any, float]:
    """Build normalized path progression from one selected graph row."""
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    return build_task_progression_from_graph(
        position=position,
        trajectory_interval=trajectory_intervals,
        graph_inputs=graph_inputs,
        trajectory_type=trajectory_type,
    )


def _build_pooled_progression(
    *,
    position: Any,
    selected_intervals_by_trajectory: Mapping[str, Any],
    support_by_trajectory: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    turn_type: str,
    common_graph_length_cm: float,
) -> tuple[Any, dict[str, int], dict[str, int]]:
    """Pool raw normalized progression samples after per-source restriction."""
    import pynapple as nap

    pair = get_dpp_trajectory_pair(turn_type)
    graphs = _pair_mapping(
        graph_inputs_by_trajectory,
        turn_type=turn_type,
        name="graph_inputs_by_trajectory",
    )
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    n_features: dict[str, int] = {}
    n_valid: dict[str, int] = {}
    for trajectory in pair:
        progression, graph_length = _build_trajectory_progression(
            position=position,
            trajectory_intervals=selected_intervals_by_trajectory[trajectory],
            graph_inputs=graphs[trajectory],
            trajectory_type=trajectory,
        )
        if not np.isclose(
            graph_length,
            common_graph_length_cm,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Linearization and validated DPP graph lengths disagree.")
        restrict = getattr(progression, "restrict", None)
        if not callable(restrict):
            raise TypeError("Task progression must expose restrict().")
        restricted = restrict(support_by_trajectory[trajectory])
        times = np.asarray(restricted.t, dtype=float).reshape(-1)
        values = np.asarray(restricted.d, dtype=float).reshape(-1)
        if times.shape != values.shape or not np.all(np.isfinite(times)):
            raise ValueError("DPP timestamps and progression values must align.")
        n_features[trajectory] = int(values.size)
        n_valid[trajectory] = int(np.sum(np.isfinite(values)))
        if times.size:
            time_chunks.append(times)
            value_chunks.append(values)

    pooled_support = _pool_interval_sets(support_by_trajectory)
    if not time_chunks:
        return (
            nap.Tsd(
                t=np.asarray([], dtype=float),
                d=np.asarray([], dtype=float),
                time_support=pooled_support,
                time_units="s",
            ),
            n_features,
            n_valid,
        )
    times = np.concatenate(time_chunks)
    values = np.concatenate(value_chunks)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Pooled DPP feature timestamps must be unique and increasing.")
    return (
        nap.Tsd(
            t=times,
            d=values,
            time_support=pooled_support,
            time_units="s",
        ),
        n_features,
        n_valid,
    )


def _raw_tuning_curve(
    *,
    spikes: Any,
    dpp: Any,
    support: Any,
    bin_edges_dpp: np.ndarray,
) -> Any:
    """Compute one tuning curve after raw paired-trajectory pooling."""
    import pynapple as nap

    return nap.compute_tuning_curves(
        data=spikes,
        features=dpp,
        epochs=support,
        bins=[np.asarray(bin_edges_dpp, dtype=float)],
        feature_names=[DPP_DIM],
    )


def compute_selected_dpp_tuning_curve(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    turn_type: str,
    trial_subset: str,
    spikes: Any,
    stable_unit_ids: Sequence[Mapping[str, Any]],
    position: Any,
    trajectory_intervals_by_type: Mapping[str, Any],
    graph_inputs_by_trajectory: Mapping[str, Mapping[str, Any]],
    movement_intervals: Any,
    movement_analysis_status: str = "valid",
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
) -> dict[str, Any]:
    """Compute one DPP curve by pooling samples before rate estimation."""
    turn = validate_turn_type(turn_type)
    subset = place.validate_trial_subset(trial_subset)
    parameters = place.validate_binning_parameters(
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=sigma_bins,
    )
    identities = place._identity_rows(spikes, stable_unit_ids)
    movement_status = place._validate_movement_analysis_status(
        movement_analysis_status,
        n_units=len(identities),
    )
    common_length, graph_lengths = common_graph_length_from_inputs(
        graph_inputs_by_trajectory,
        turn_type=turn,
    )
    bin_edges = build_dpp_bin_edges(
        common_length,
        bin_size_cm=parameters["bin_size_cm"],
        bin_count=parameters["bin_count"],
    )
    selected, _selected_support = select_dpp_trial_intervals(
        trajectory_intervals_by_type,
        turn_type=turn,
        trial_subset=subset,
    )
    n_trials_by_trajectory = {
        trajectory: place._interval_summary(intervals)[0]
        for trajectory, intervals in selected.items()
    }
    support_by_trajectory = {
        trajectory: place._intersect_intervals(intervals, movement_intervals)
        for trajectory, intervals in selected.items()
    }
    support_duration_by_trajectory = {
        trajectory: place._interval_summary(intervals)[1]
        for trajectory, intervals in support_by_trajectory.items()
    }
    pooled_support = _pool_interval_sets(support_by_trajectory)
    spike_counts = np.zeros(len(identities), dtype=np.int64)
    empty_feature_counts = {trajectory: 0 for trajectory in get_dpp_trajectory_pair(turn)}

    def terminal(
        status: str,
        *,
        feature_counts: Mapping[str, int] = empty_feature_counts,
        valid_counts: Mapping[str, int] = empty_feature_counts,
    ) -> dict[str, Any]:
        attrs = _curve_attributes(
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=epoch,
            turn_type=turn,
            trial_subset=subset,
            parameters=parameters,
            common_graph_length_cm=common_length,
            graph_length_cm_by_trajectory=graph_lengths,
            bin_edges_dpp=bin_edges,
            n_trials_by_trajectory=n_trials_by_trajectory,
            support_duration_s_by_trajectory=support_duration_by_trajectory,
            n_feature_samples_by_trajectory=feature_counts,
            n_valid_position_samples_by_trajectory=valid_counts,
            n_units=len(identities),
            n_valid_units=0,
            analysis_status=status,
        )
        curve = _terminal_curve(
            identity_rows=identities,
            spike_counts=spike_counts,
            bin_edges_dpp=bin_edges,
            attrs=attrs,
        )
        return _result_summary(curve)

    if not identities:
        return terminal("no_units")
    if movement_status in {"no_valid_position", "no_movement"}:
        return terminal(movement_status)
    if sum(n_trials_by_trajectory.values()) == 0:
        return terminal("no_trials")
    if sum(support_duration_by_trajectory.values()) <= 0.0:
        return terminal("no_movement")

    spike_counts = place._subset_spike_counts(
        spikes,
        pooled_support,
        n_units=len(identities),
    )

    progression, feature_counts, valid_counts = _build_pooled_progression(
        position=position,
        selected_intervals_by_trajectory=selected,
        support_by_trajectory=support_by_trajectory,
        graph_inputs_by_trajectory=graph_inputs_by_trajectory,
        turn_type=turn,
        common_graph_length_cm=common_length,
    )
    if sum(valid_counts.values()) < 2:
        return terminal(
            "no_valid_position",
            feature_counts=feature_counts,
            valid_counts=valid_counts,
        )
    raw = _raw_tuning_curve(
        spikes=spikes,
        dpp=progression,
        support=pooled_support,
        bin_edges_dpp=bin_edges,
    )
    values = place._aligned_raw_values(
        raw,
        identity_rows=identities,
        n_bins=len(bin_edges) - 1,
    )
    values = place.smooth_tuning_values(
        values,
        sigma_bins=parameters["sigma_bins"],
    )
    n_valid_units = int(np.sum(np.any(np.isfinite(values), axis=1)))
    status = "valid" if n_valid_units else "no_valid_units"
    attrs = _curve_attributes(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        turn_type=turn,
        trial_subset=subset,
        parameters=parameters,
        common_graph_length_cm=common_length,
        graph_length_cm_by_trajectory=graph_lengths,
        bin_edges_dpp=bin_edges,
        n_trials_by_trajectory=n_trials_by_trajectory,
        support_duration_s_by_trajectory=support_duration_by_trajectory,
        n_feature_samples_by_trajectory=feature_counts,
        n_valid_position_samples_by_trajectory=valid_counts,
        n_units=len(identities),
        n_valid_units=n_valid_units,
        analysis_status=status,
    )
    curve = _build_curve(
        values,
        identity_rows=identities,
        spike_counts=spike_counts,
        bin_edges_dpp=bin_edges,
        attrs=attrs,
    )
    validate_dpp_tuning_curve(curve)
    return _result_summary(curve)


def _parse_json_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Parse one JSON-object NetCDF attribute."""
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"{name} must contain a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return parsed


def validate_dpp_tuning_curve(curve: Any) -> Any:
    """Validate and return one canonical DPP tuning DataArray."""
    if getattr(curve, "name", None) != "firing_rate_hz":
        raise ValueError("DPP tuning curve must be named 'firing_rate_hz'.")
    if tuple(getattr(curve, "dims", ())) != (UNIT_DIM, DPP_DIM):
        raise ValueError(f"DPP tuning dimensions must be {(UNIT_DIM, DPP_DIM)!r}.")
    for coordinate in (
        UNIT_DIM,
        DPP_DIM,
        PATH_FRACTION_COORDINATE,
        LINEAR_POSITION_COORDINATE,
        SPIKE_COUNT_COORDINATE,
        *IDENTITY_COORDINATES,
    ):
        if coordinate not in curve.coords:
            raise ValueError(f"DPP tuning curve is missing coordinate {coordinate!r}.")
    values = np.asarray(curve.values, dtype=float)
    if np.any(np.isinf(values)):
        raise ValueError("DPP tuning values may be finite or NaN, not infinite.")
    n_units, n_bins = values.shape
    stable_ids = np.asarray(curve.coords["stable_unit_id"].values).astype(str)
    unit_ids = np.asarray(curve.coords["unit_id"].values).astype(str)
    merge_ids = np.asarray(curve.coords["spikesorting_merge_id"].values).astype(str)
    group_ids = np.asarray(curve.coords["group_unit_id"].values).astype(str)
    unit_coordinate = np.asarray(curve.coords[UNIT_DIM].values).astype(str)
    expected = np.asarray(
        [f"{merge}:{unit}" for merge, unit in zip(merge_ids, unit_ids, strict=True)],
        dtype=str,
    )
    for name, coordinate in {
        "stable_unit_id": stable_ids,
        "unit_id": unit_ids,
        "spikesorting_merge_id": merge_ids,
        "group_unit_id": group_ids,
        UNIT_DIM: unit_coordinate,
    }.items():
        if coordinate.shape != (n_units,):
            raise ValueError(f"Coordinate {name!r} does not align with units.")
    if not np.array_equal(stable_ids, expected) or not np.array_equal(
        unit_coordinate,
        stable_ids,
    ):
        raise ValueError("DPP stable unit identities are inconsistent.")
    if len(set(stable_ids.tolist())) != n_units or len(set(group_ids.tolist())) != n_units:
        raise ValueError("DPP unit coordinates must be unique.")
    spike_counts = np.asarray(curve.coords[SPIKE_COUNT_COORDINATE].values, dtype=float)
    if spike_counts.shape != (n_units,) or np.any(np.isinf(spike_counts)):
        raise ValueError("Spike counts must align with units and may not be infinite.")
    finite_counts = spike_counts[np.isfinite(spike_counts)]
    if np.any(finite_counts < 0.0) or not np.allclose(finite_counts, np.rint(finite_counts)):
        raise ValueError("Finite spike counts must be non-negative integers.")
    legacy = str(curve.attrs.get("legacy_normalized", "false")).lower() == "true"
    if np.any(np.isnan(spike_counts)) and not (
        legacy and curve.attrs.get("trial_subset") == "all"
    ):
        raise ValueError("Unknown spike counts are only valid for legacy all-trial DPP.")

    required_attrs = {
        "animal_name",
        "date",
        "region",
        "epoch",
        "turn_type",
        "trial_subset",
        "outbound_trajectory_type",
        "inbound_trajectory_type",
        "source_trajectory_types_json",
        "binning_mode",
        "sigma_bins",
        "common_graph_length_cm",
        "graph_length_cm_by_trajectory_json",
        "bin_edges_dpp_json",
        "bin_edges_cm_json",
        "n_trials_by_trajectory_json",
        "support_duration_s_by_trajectory_json",
        "n_feature_samples_by_trajectory_json",
        "n_valid_position_samples_by_trajectory_json",
        "n_trials",
        "n_outbound_trials",
        "n_inbound_trials",
        "support_duration_s",
        "n_feature_samples",
        "n_valid_position_samples",
        "n_units",
        "n_valid_units",
        "analysis_status",
        "pooling_policy",
    }
    missing = sorted(required_attrs.difference(curve.attrs))
    if missing:
        raise ValueError(f"DPP tuning curve is missing attributes {missing!r}.")
    turn = validate_turn_type(curve.attrs["turn_type"])
    subset = place.validate_trial_subset(curve.attrs["trial_subset"])
    pair = get_dpp_trajectory_pair(turn)
    try:
        source_pair = tuple(json.loads(str(curve.attrs["source_trajectory_types_json"])))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("source_trajectory_types_json is invalid.") from exc
    if source_pair != pair:
        raise ValueError("Saved DPP source trajectories do not match turn_type.")
    if (
        str(curve.attrs["outbound_trajectory_type"]) != pair[0]
        or str(curve.attrs["inbound_trajectory_type"]) != pair[1]
    ):
        raise ValueError("Saved outbound/inbound trajectories do not match turn_type.")
    mode = str(curve.attrs["binning_mode"])
    parameters = place.validate_binning_parameters(
        bin_size_cm=curve.attrs.get("bin_size_cm") if mode == "bin_size_cm" else None,
        bin_count=curve.attrs.get("bin_count") if mode == "bin_count" else None,
        sigma_bins=curve.attrs["sigma_bins"],
    )
    common_length = place._numeric_scalar(
        curve.attrs["common_graph_length_cm"],
        name="common_graph_length_cm",
    )
    if common_length <= 0.0:
        raise ValueError("common_graph_length_cm must be positive.")
    graph_lengths = _source_float_mapping(
        _parse_json_mapping(
            curve.attrs["graph_length_cm_by_trajectory_json"],
            name="graph_length_cm_by_trajectory_json",
        ),
        turn_type=turn,
        name="graph_length_cm_by_trajectory",
    )
    if any(
        not np.isclose(length, common_length, rtol=1e-10, atol=1e-12)
        for length in graph_lengths.values()
    ):
        raise ValueError("Saved DPP graph lengths do not match the common length.")
    edges = place._parse_bin_edges(
        curve.attrs["bin_edges_dpp_json"],
        name="bin_edges_dpp_json",
    )
    expected_edges = build_dpp_bin_edges(
        common_length,
        bin_size_cm=parameters["bin_size_cm"],
        bin_count=parameters["bin_count"],
    )
    if edges.shape != expected_edges.shape or not np.allclose(
        edges,
        expected_edges,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Saved DPP bin edges do not match the parameters.")
    if edges.size != n_bins + 1:
        raise ValueError("Saved DPP bin edges do not match the data shape.")
    centers = (edges[:-1] + edges[1:]) / 2.0
    dpp_coordinate = np.asarray(curve.coords[DPP_DIM].values, dtype=float)
    fraction = np.asarray(curve.coords[PATH_FRACTION_COORDINATE].values, dtype=float)
    centimeters = np.asarray(curve.coords[LINEAR_POSITION_COORDINATE].values, dtype=float)
    if not (
        dpp_coordinate.shape == fraction.shape == centimeters.shape == (n_bins,)
        and np.allclose(dpp_coordinate, centers, rtol=1e-10, atol=1e-12)
        and np.allclose(fraction, centers, rtol=1e-10, atol=1e-12)
        and np.allclose(centimeters, centers * common_length, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("DPP coordinates do not match saved bin edges.")
    if (
        curve.coords[DPP_DIM].attrs.get("units") != "1"
        or curve.coords[PATH_FRACTION_COORDINATE].attrs.get("units") != "1"
        or curve.coords[LINEAR_POSITION_COORDINATE].attrs.get("units") != "cm"
        or curve.attrs.get("units") != "Hz"
    ):
        raise ValueError("DPP coordinate or value units are invalid.")

    n_trials_by_source = _source_integer_mapping(
        _parse_json_mapping(
            curve.attrs["n_trials_by_trajectory_json"],
            name="n_trials_by_trajectory_json",
        ),
        turn_type=turn,
        name="n_trials_by_trajectory",
    )
    support_by_source = _source_float_mapping(
        _parse_json_mapping(
            curve.attrs["support_duration_s_by_trajectory_json"],
            name="support_duration_s_by_trajectory_json",
        ),
        turn_type=turn,
        name="support_duration_s_by_trajectory",
    )
    features_by_source = _source_integer_mapping(
        _parse_json_mapping(
            curve.attrs["n_feature_samples_by_trajectory_json"],
            name="n_feature_samples_by_trajectory_json",
        ),
        turn_type=turn,
        name="n_feature_samples_by_trajectory",
    )
    valid_by_source = _source_integer_mapping(
        _parse_json_mapping(
            curve.attrs["n_valid_position_samples_by_trajectory_json"],
            name="n_valid_position_samples_by_trajectory_json",
        ),
        turn_type=turn,
        name="n_valid_position_samples_by_trajectory",
    )
    if any(valid_by_source[name] > features_by_source[name] for name in pair):
        raise ValueError("Valid DPP samples cannot exceed feature samples.")
    aggregates = {
        "n_trials": sum(n_trials_by_source.values()),
        "support_duration_s": sum(support_by_source.values()),
        "n_feature_samples": sum(features_by_source.values()),
        "n_valid_position_samples": sum(valid_by_source.values()),
    }
    for name, expected_value in aggregates.items():
        value = place._numeric_scalar(curve.attrs[name], name=name)
        if name != "support_duration_s" and value != int(value):
            raise ValueError(f"{name} must be an integer.")
        if not np.isclose(value, expected_value, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{name} does not equal its source-trajectory sum.")
    if int(curve.attrs["n_outbound_trials"]) != n_trials_by_source[pair[0]]:
        raise ValueError("n_outbound_trials does not match source metadata.")
    if int(curve.attrs["n_inbound_trials"]) != n_trials_by_source[pair[1]]:
        raise ValueError("n_inbound_trials does not match source metadata.")
    if int(curve.attrs["n_units"]) != n_units:
        raise ValueError("Saved n_units does not match the unit dimension.")
    n_valid_units = int(np.sum(np.any(np.isfinite(values), axis=1)))
    if int(curve.attrs["n_valid_units"]) != n_valid_units:
        raise ValueError("Saved n_valid_units does not match DPP values.")
    status = str(curve.attrs["analysis_status"])
    if status not in ANALYSIS_STATUSES:
        raise ValueError(f"Unknown DPP analysis_status {status!r}.")
    n_trials = int(aggregates["n_trials"])
    duration = float(aggregates["support_duration_s"])
    n_valid_position = int(aggregates["n_valid_position_samples"])
    if status == "valid" and (
        n_units == 0
        or n_valid_units == 0
        or n_trials == 0
        or duration <= 0.0
        or n_valid_position < 2
    ):
        raise ValueError("Valid DPP curves require units, trials, support, and position.")
    if status == "no_units" and n_units != 0:
        raise ValueError("no_units DPP curves must have no units.")
    if status == "no_trials" and (n_units == 0 or n_trials != 0 or duration != 0.0):
        raise ValueError("no_trials DPP curves require units and zero trials/support.")
    if status == "no_movement" and (n_units == 0 or duration != 0.0):
        raise ValueError("no_movement DPP curves require units and zero support.")
    if status == "no_valid_position" and (
        n_units == 0 or n_valid_position >= 2
    ):
        raise ValueError("no_valid_position DPP curves require fewer than two samples.")
    if status == "no_valid_units" and (
        n_units == 0 or n_trials == 0 or duration <= 0.0 or n_valid_position < 2
    ):
        raise ValueError("no_valid_units DPP metadata is inconsistent.")
    if status != "valid" and n_valid_units != 0:
        raise ValueError(f"{status} DPP curves cannot contain valid unit rows.")
    if subset not in TRIAL_SUBSETS:
        raise ValueError("Unknown trial subset.")
    return curve


def _decoded_nwb_text(value: Any, *, name: str) -> str:
    """Return one UTF-8 string fetched from a DPP DynamicTable."""
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
    """Return one fetched DPP DynamicTable as an exact-column DataFrame."""
    import pandas as pd
    from hdmf.common import DynamicTable

    if isinstance(nwb_table, pd.DataFrame):
        table = nwb_table.copy()
    elif isinstance(nwb_table, DynamicTable):
        if str(nwb_table.name) != expected_name:
            raise ValueError(
                f"Unexpected DPP NWB object name {nwb_table.name!r}; "
                f"expected {expected_name!r}."
            )
        table = nwb_table.to_dataframe()
    else:
        raise TypeError("DPP NWB objects must be DynamicTables or DataFrames.")
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
    validate_dpp_tuning_curve(curve)
    return {
        "attrs": dict(curve.attrs),
        "coordinate_attrs": {
            str(name): dict(curve.coords[name].attrs)
            for name in (
                DPP_DIM,
                PATH_FRACTION_COORDINATE,
                LINEAR_POSITION_COORDINATE,
                SPIKE_COUNT_COORDINATE,
            )
        },
    }


def dpp_tuning_to_dynamic_table(curve: Any) -> Any:
    """Store unit identities, spike counts, and fixed-length DPP vectors."""
    import pandas as pd
    from hdmf.common import DynamicTable, VectorData

    canonical = validate_dpp_tuning_curve(curve)
    n_units = int(canonical.sizes[UNIT_DIM])
    n_bins = int(canonical.sizes[DPP_DIM])
    description = (
        "One row per selected unit with its directional path-progression "
        f"firing-rate vector; v1ca1 schema version {NWB_ARTIFACT_SCHEMA_VERSION}."
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
                        "Zero-based row in the unit-by-DPP tuning matrix."
                        if name == "curve_row"
                        else "Canonical directional progression-tuning field."
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
            "unit_id": np.asarray(canonical.coords["unit_id"].values).astype(
                str
            ),
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
                    "Zero-based row in the unit-by-DPP tuning matrix."
                    if name == "curve_row"
                    else "Canonical directional progression-tuning field."
                ),
            }
            for name in _NWB_TUNING_COLUMNS
        ],
    )


def dpp_bins_to_dynamic_table(curve: Any) -> Any:
    """Store normalized and physical coordinates for every DPP vector bin."""
    import pandas as pd
    from hdmf.common import DynamicTable

    canonical = validate_dpp_tuning_curve(curve)
    dpp_edges = place._parse_bin_edges(
        canonical.attrs["bin_edges_dpp_json"],
        name="bin_edges_dpp_json",
    )
    cm_edges = place._parse_bin_edges(
        canonical.attrs["bin_edges_cm_json"],
        name="bin_edges_cm_json",
    )
    common_length = float(canonical.attrs["common_graph_length_cm"])
    if cm_edges.shape != dpp_edges.shape or not np.allclose(
        cm_edges,
        dpp_edges * common_length,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("DPP centimeter bin edges do not match normalized bins.")
    n_bins = int(canonical.sizes[DPP_DIM])
    table = pd.DataFrame(
        {
            "dpp_bin": np.arange(n_bins, dtype=np.int64),
            "left_edge_dpp": dpp_edges[:-1],
            DPP_DIM: np.asarray(canonical.coords[DPP_DIM].values, dtype=float),
            "right_edge_dpp": dpp_edges[1:],
            PATH_FRACTION_COORDINATE: np.asarray(
                canonical.coords[PATH_FRACTION_COORDINATE].values,
                dtype=float,
            ),
            "left_edge_cm": cm_edges[:-1],
            LINEAR_POSITION_COORDINATE: np.asarray(
                canonical.coords[LINEAR_POSITION_COORDINATE].values,
                dtype=float,
            ),
            "right_edge_cm": cm_edges[1:],
        },
        columns=list(_NWB_BIN_COLUMNS),
    )
    return DynamicTable.from_dataframe(
        name=NWB_BINS_TABLE_NAME,
        df=table,
        table_description=(
            "One row per directional path-progression column of dpp_tuning; "
            f"v1ca1 schema version {NWB_ARTIFACT_SCHEMA_VERSION}."
        ),
        columns=[
            {
                "name": name,
                "description": (
                    "Zero-based tuning-vector column."
                    if name == "dpp_bin"
                    else "Canonical directional progression-bin field."
                ),
            }
            for name in _NWB_BIN_COLUMNS
        ],
    )


def dpp_provenance_to_dynamic_table(curve: Any) -> Any:
    """Store one canonical JSON metadata record for the complete DPP curve."""
    import pandas as pd
    from hdmf.common import DynamicTable

    from v1ca1.spyglass.selection import canonical_json

    table = pd.DataFrame(
        [
            {
                "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
                "metadata_json": canonical_json(_curve_metadata_payload(curve)),
            }
        ],
        columns=list(_NWB_PROVENANCE_COLUMNS),
    )
    return DynamicTable.from_dataframe(
        name=NWB_PROVENANCE_TABLE_NAME,
        df=table,
        table_description=(
            "One provenance record for directional path-progression tuning; "
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


def _dpp_nwb_frames(
    tuning: Any,
    bins: Any,
    provenance: Any,
) -> tuple[Any, Any, Any]:
    """Return the three canonical DPP table frames after structural checks."""
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
        raise ValueError("DPP provenance must contain exactly one row.")
    return tuning_table, bins_table, provenance_table


def dpp_tuning_curve_from_nwb_objects(
    tuning: Any,
    bins: Any,
    provenance: Any,
) -> Any:
    """Reconstruct the canonical DPP DataArray from three fetched objects."""
    tuning_table, bins_table, provenance_table = _dpp_nwb_frames(
        tuning,
        bins,
        provenance,
    )
    schema_version = _decoded_nwb_text(
        provenance_table.loc[0, "artifact_schema_version"],
        name="artifact_schema_version",
    )
    if schema_version != NWB_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("DPP NWB artifact schema version is unsupported.")
    metadata_json = _decoded_nwb_text(
        provenance_table.loc[0, "metadata_json"],
        name="metadata_json",
    )
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise ValueError("DPP provenance is not valid JSON.") from exc
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "attrs",
        "coordinate_attrs",
    }:
        raise ValueError("DPP provenance has an invalid schema.")
    attrs = dict(metadata["attrs"])
    coordinate_attrs = dict(metadata["coordinate_attrs"])

    n_units = len(tuning_table)
    n_bins = len(bins_table)
    curve_rows = np.asarray(tuning_table["curve_row"], dtype=np.int64)
    if not np.array_equal(curve_rows, np.arange(n_units, dtype=np.int64)):
        raise ValueError("DPP curve_row values must be consecutive.")
    bin_rows = np.asarray(bins_table["dpp_bin"], dtype=np.int64)
    if not np.array_equal(bin_rows, np.arange(n_bins, dtype=np.int64)):
        raise ValueError("DPP dpp_bin values must be consecutive.")
    if n_bins == 0:
        raise ValueError("DPP tuning requires at least one bin.")

    left_dpp = np.asarray(bins_table["left_edge_dpp"], dtype=float)
    right_dpp = np.asarray(bins_table["right_edge_dpp"], dtype=float)
    left_cm = np.asarray(bins_table["left_edge_cm"], dtype=float)
    right_cm = np.asarray(bins_table["right_edge_cm"], dtype=float)
    if (
        not np.all(np.isfinite(left_dpp))
        or not np.all(np.isfinite(right_dpp))
        or not np.all(np.isfinite(left_cm))
        or not np.all(np.isfinite(right_cm))
        or np.any(right_dpp <= left_dpp)
        or np.any(right_cm <= left_cm)
        or not np.allclose(
            left_dpp[1:], right_dpp[:-1], rtol=1e-10, atol=1e-12
        )
        or not np.allclose(
            left_cm[1:], right_cm[:-1], rtol=1e-10, atol=1e-12
        )
    ):
        raise ValueError("DPP bin edges are invalid or discontinuous.")
    dpp_edges = np.concatenate((left_dpp[:1], right_dpp))
    cm_edges = np.concatenate((left_cm[:1], right_cm))
    common_length = float(attrs["common_graph_length_cm"])
    if not np.allclose(
        cm_edges,
        dpp_edges * common_length,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("DPP physical and normalized bin edges disagree.")

    if n_units:
        tuning_rows = [
            np.asarray(value, dtype=float).reshape(-1)
            for value in tuning_table["firing_rate_hz"]
        ]
        if any(row.shape != (n_bins,) for row in tuning_rows):
            raise ValueError("Every DPP tuning vector must align with the bin table.")
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
        spike_counts=np.asarray(
            tuning_table[SPIKE_COUNT_COORDINATE],
            dtype=float,
        ),
        bin_edges_dpp=dpp_edges,
        attrs=attrs,
    )
    expected_bin_values = {
        DPP_DIM: np.asarray(bins_table[DPP_DIM], dtype=float),
        PATH_FRACTION_COORDINATE: np.asarray(
            bins_table[PATH_FRACTION_COORDINATE],
            dtype=float,
        ),
        LINEAR_POSITION_COORDINATE: np.asarray(
            bins_table[LINEAR_POSITION_COORDINATE],
            dtype=float,
        ),
    }
    for coordinate_name, expected_values in expected_bin_values.items():
        if not np.allclose(
            expected_values,
            np.asarray(curve.coords[coordinate_name].values, dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "DPP bin metadata disagrees with curve provenance: "
                f"{coordinate_name}."
            )
    for coordinate_name, values_by_name in coordinate_attrs.items():
        if coordinate_name not in curve.coords or not isinstance(
            values_by_name,
            Mapping,
        ):
            raise ValueError("DPP coordinate provenance is invalid.")
        curve.coords[coordinate_name].attrs.update(dict(values_by_name))
    return validate_dpp_tuning_curve(curve)


def _normalized_float_values(values: Any) -> list[Any]:
    """Return floats with NaN represented canonically for semantic hashing."""
    return [
        None if np.isnan(value) else float(value)
        for value in np.asarray(values, dtype=float).reshape(-1)
    ]


def dpp_tuning_sha256(curve: Any) -> str:
    """Digest DPP unit metadata and tuning values independent of storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    canonical = validate_dpp_tuning_curve(curve)
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


def dpp_bins_sha256(curve: Any) -> str:
    """Digest ordered DPP-bin metadata independent of NWB storage."""
    from v1ca1.spyglass.selection import provenance_sha256

    canonical = validate_dpp_tuning_curve(curve)
    dpp_edges = place._parse_bin_edges(
        canonical.attrs["bin_edges_dpp_json"],
        name="bin_edges_dpp_json",
    )
    cm_edges = place._parse_bin_edges(
        canonical.attrs["bin_edges_cm_json"],
        name="bin_edges_cm_json",
    )
    return provenance_sha256(
        {
            "dpp_bin": list(range(int(canonical.sizes[DPP_DIM]))),
            "left_edge_dpp": dpp_edges[:-1].tolist(),
            DPP_DIM: np.asarray(
                canonical.coords[DPP_DIM].values,
                dtype=float,
            ).tolist(),
            "right_edge_dpp": dpp_edges[1:].tolist(),
            PATH_FRACTION_COORDINATE: np.asarray(
                canonical.coords[PATH_FRACTION_COORDINATE].values,
                dtype=float,
            ).tolist(),
            "left_edge_cm": cm_edges[:-1].tolist(),
            LINEAR_POSITION_COORDINATE: np.asarray(
                canonical.coords[LINEAR_POSITION_COORDINATE].values,
                dtype=float,
            ).tolist(),
            "right_edge_cm": cm_edges[1:].tolist(),
        }
    )


def dpp_provenance_sha256(curve: Any) -> str:
    """Digest the canonical DPP scalar and coordinate provenance payload."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
            "metadata": _curve_metadata_payload(curve),
        }
    )


def dpp_tuning_curve_sha256(curve: Any) -> str:
    """Digest all three logical DPP objects as one scientific artifact."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            "dpp_tuning_sha256": dpp_tuning_sha256(curve),
            "dpp_bins_sha256": dpp_bins_sha256(curve),
            "dpp_provenance_sha256": dpp_provenance_sha256(curve),
        }
    )


def _result_summary(curve: Any) -> dict[str, Any]:
    """Return table-friendly scalar metadata around one canonical curve."""
    validate_dpp_tuning_curve(curve)
    return {
        "tuning_curve": curve,
        "analysis_status": str(curve.attrs["analysis_status"]),
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "n_outbound_trials": int(curve.attrs["n_outbound_trials"]),
        "n_inbound_trials": int(curve.attrs["n_inbound_trials"]),
        "support_duration_s": float(curve.attrs["support_duration_s"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[DPP_DIM]),
    }


def normalize_legacy_all_trial_dpp_tuning_curve(
    legacy_curve: Any,
    *,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    turn_type: str,
    common_graph_length_cm: float,
    graph_length_cm_by_trajectory: Mapping[str, float],
    n_trials_by_trajectory: Mapping[str, int],
    support_duration_s_by_trajectory: Mapping[str, float],
    n_feature_samples_by_trajectory: Mapping[str, int],
    n_valid_position_samples_by_trajectory: Mapping[str, int] | None = None,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
) -> Any:
    """Normalize one compatible legacy all-trial task-progression curve."""
    turn = validate_turn_type(turn_type)
    parameters = place.validate_binning_parameters(
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=sigma_bins,
    )
    if parameters["sigma_bins"] != 0.0:
        raise ValueError("Legacy DPP curves can only register with sigma_bins=0.")
    if getattr(legacy_curve, "name", None) != "firing_rate_hz":
        raise ValueError("Legacy DPP curve must be named 'firing_rate_hz'.")
    if tuple(getattr(legacy_curve, "dims", ())) != ("unit", "tp"):
        raise ValueError("Legacy DPP dimensions must be ('unit', 'tp').")
    expected_attrs = {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
        "model_name": "task_progression",
        "turn_type": turn,
    }
    for name, expected in expected_attrs.items():
        if str(legacy_curve.attrs.get(name, "")) != str(expected):
            raise ValueError(
                f"Legacy DPP {name}={legacy_curve.attrs.get(name)!r}, "
                f"expected {expected!r}."
            )
    graph_lengths = _source_float_mapping(
        graph_length_cm_by_trajectory,
        turn_type=turn,
        name="graph_length_cm_by_trajectory",
    )
    common_length = place._numeric_scalar(
        common_graph_length_cm,
        name="common_graph_length_cm",
    )
    if common_length <= 0.0 or any(
        not np.isclose(length, common_length, rtol=1e-10, atol=1e-12)
        for length in graph_lengths.values()
    ):
        raise ValueError("Legacy DPP registration requires one common graph length.")
    trials = _source_integer_mapping(
        n_trials_by_trajectory,
        turn_type=turn,
        name="n_trials_by_trajectory",
    )
    durations = _source_float_mapping(
        support_duration_s_by_trajectory,
        turn_type=turn,
        name="support_duration_s_by_trajectory",
    )
    features = _source_integer_mapping(
        n_feature_samples_by_trajectory,
        turn_type=turn,
        name="n_feature_samples_by_trajectory",
    )
    valid = (
        dict(features)
        if n_valid_position_samples_by_trajectory is None
        else _source_integer_mapping(
            n_valid_position_samples_by_trajectory,
            turn_type=turn,
            name="n_valid_position_samples_by_trajectory",
        )
    )
    if any(valid[name] > features[name] for name in get_dpp_trajectory_pair(turn)):
        raise ValueError("Valid DPP samples cannot exceed feature samples.")
    legacy_units = list(np.asarray(legacy_curve.coords["unit"].values).reshape(-1))
    if len(legacy_units) != len(set(legacy_units)):
        raise ValueError("Legacy DPP curve contains duplicate unit ids.")
    identities, selected_indices = place._legacy_identity_rows(
        legacy_units,
        unit_identity_resolver,
    )
    values = np.asarray(legacy_curve.values, dtype=float)
    if values.shape[0] != len(legacy_units) or np.any(np.isinf(values)):
        raise ValueError("Legacy DPP values must align with units and contain no infinities.")
    values = values[selected_indices]
    if "bin_edges" not in legacy_curve.attrs:
        raise ValueError("Legacy DPP curve is missing bin_edges metadata.")
    edges = place._parse_bin_edges(legacy_curve.attrs["bin_edges"], name="legacy bin_edges")
    expected_edges = build_dpp_bin_edges(
        common_length,
        bin_size_cm=parameters["bin_size_cm"],
        bin_count=parameters["bin_count"],
    )
    if edges.shape != expected_edges.shape or not np.allclose(
        edges,
        expected_edges,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Legacy DPP bin edges do not match selected parameters.")
    legacy_position = np.asarray(legacy_curve.coords["tp"].values, dtype=float)
    if legacy_position.shape != (edges.size - 1,) or not np.allclose(
        legacy_position,
        (edges[:-1] + edges[1:]) / 2.0,
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("Legacy tp coordinates do not match bin_edges.")
    n_valid_units = int(np.sum(np.any(np.isfinite(values), axis=1)))
    status = "no_units" if not identities else "valid" if n_valid_units else "no_valid_units"
    attrs = _curve_attributes(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        turn_type=turn,
        trial_subset="all",
        parameters=parameters,
        common_graph_length_cm=common_length,
        graph_length_cm_by_trajectory=graph_lengths,
        bin_edges_dpp=edges,
        n_trials_by_trajectory=trials,
        support_duration_s_by_trajectory=durations,
        n_feature_samples_by_trajectory=features,
        n_valid_position_samples_by_trajectory=valid,
        n_units=len(identities),
        n_valid_units=n_valid_units,
        analysis_status=status,
    )
    attrs["legacy_normalized"] = "true"
    curve = _build_curve(
        values,
        identity_rows=identities,
        spike_counts=np.full(len(identities), np.nan, dtype=float),
        bin_edges_dpp=edges,
        attrs=attrs,
    )
    return validate_dpp_tuning_curve(curve)


def load_dpp_artifact(path: Path) -> Any:
    """Load and validate one canonical DPP NetCDF artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DPP tuning artifact not found: {path}")
    import xarray as xr

    with xr.open_dataarray(path) as opened:
        curve = opened.load()
    return validate_dpp_tuning_curve(curve)


def write_dpp_artifact(
    curve: Any,
    path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one validated DPP NetCDF without implicit overwrite."""
    validate_dpp_tuning_curve(curve)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite DPP tuning artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.nc")
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.backup")
    had_existing = path.exists()
    try:
        curve.to_netcdf(temporary)
        load_dpp_artifact(temporary)
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


def register_existing_dpp_artifact(
    *,
    source_path: Path,
    destination_path: Path,
    unit_identity_resolver: Mapping[Any, Mapping[str, Any]]
    | Callable[[Any], Mapping[str, Any]],
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    turn_type: str,
    common_graph_length_cm: float,
    graph_length_cm_by_trajectory: Mapping[str, float],
    n_trials_by_trajectory: Mapping[str, int],
    support_duration_s_by_trajectory: Mapping[str, float],
    n_feature_samples_by_trajectory: Mapping[str, int],
    n_valid_position_samples_by_trajectory: Mapping[str, int] | None = None,
    bin_size_cm: float | None = None,
    bin_count: int | None = None,
    sigma_bins: float = 0.0,
    artifact_attributes: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize and atomically register one legacy all-trial DPP artifact."""
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Legacy DPP tuning curve not found: {source}")
    if source.resolve() == destination.resolve(strict=False):
        raise ValueError("Legacy source and canonical destination must differ.")
    import xarray as xr

    with xr.open_dataarray(source) as opened:
        legacy_curve = opened.load()
    curve = normalize_legacy_all_trial_dpp_tuning_curve(
        legacy_curve,
        unit_identity_resolver=unit_identity_resolver,
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        turn_type=turn_type,
        common_graph_length_cm=common_graph_length_cm,
        graph_length_cm_by_trajectory=graph_length_cm_by_trajectory,
        n_trials_by_trajectory=n_trials_by_trajectory,
        support_duration_s_by_trajectory=support_duration_s_by_trajectory,
        n_feature_samples_by_trajectory=n_feature_samples_by_trajectory,
        n_valid_position_samples_by_trajectory=n_valid_position_samples_by_trajectory,
        bin_size_cm=bin_size_cm,
        bin_count=bin_count,
        sigma_bins=sigma_bins,
    )
    curve.attrs.update(
        {str(name): str(value) for name, value in dict(artifact_attributes or {}).items()}
    )
    written = write_dpp_artifact(curve, destination, overwrite=overwrite)
    result = _result_summary(curve)
    result.update(
        {
            "tuning_curve_path": written,
            "legacy_artifact_provenance": {
                "source_path": str(source.resolve(strict=True)),
                "source_sha256": place._file_sha256(source),
                "legacy_unit_coordinate": "sorting_unit_id",
            },
            "_created_artifact_paths": [str(written)],
        }
    )
    return result


__all__ = [
    "ANALYSIS_STATUSES",
    "ARTIFACT_DIRNAME",
    "ARTIFACT_FILENAME",
    "DEFAULT_ARTIFACT_ROOT",
    "DPP_DIM",
    "IDENTITY_COORDINATES",
    "LINEAR_POSITION_COORDINATE",
    "NWB_ARTIFACT_SCHEMA_VERSION",
    "NWB_BINS_TABLE_NAME",
    "NWB_PROVENANCE_TABLE_NAME",
    "NWB_TUNING_TABLE_NAME",
    "PATH_FRACTION_COORDINATE",
    "SPIKE_COUNT_COORDINATE",
    "TRIAL_SUBSETS",
    "TURN_TYPES",
    "UNIT_DIM",
    "build_dpp_bin_edges",
    "common_graph_length_from_inputs",
    "compute_selected_dpp_tuning_curve",
    "dpp_bins_sha256",
    "dpp_bins_to_dynamic_table",
    "dpp_provenance_sha256",
    "dpp_provenance_to_dynamic_table",
    "dpp_tuning_curve_from_nwb_objects",
    "dpp_tuning_curve_sha256",
    "dpp_tuning_sha256",
    "dpp_tuning_to_dynamic_table",
    "get_dpp_artifact_path",
    "get_dpp_trajectory_pair",
    "load_dpp_artifact",
    "normalize_legacy_all_trial_dpp_tuning_curve",
    "register_existing_dpp_artifact",
    "select_dpp_trial_intervals",
    "validate_dpp_tuning_curve",
    "validate_turn_type",
    "write_dpp_artifact",
]
