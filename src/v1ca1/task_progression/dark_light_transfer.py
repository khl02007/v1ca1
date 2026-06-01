"""Score dark-trained task-progression models on held-out light trajectories.

This script asks whether task-progression tuning learned in a dark run predicts
held-out light-run spiking better than a circularly shifted light-train null.
Light and shuffle models are fit separately for each trajectory. Dark models are
fit separately for each turn direction, pooling the two same-turn trajectories.
All three model classes are scored on the same intact light-test folds.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from v1ca1.helper.cuda import (
    configure_cuda_visible_devices,
    pop_cuda_visible_devices_argument,
)


_CUDA_VISIBLE_DEVICES_CLI = pop_cuda_visible_devices_argument()
configure_cuda_visible_devices(_CUDA_VISIBLE_DEVICES_CLI)


from v1ca1.helper.run_logging import write_run_log
from v1ca1.paper_figures.datasets import (
    get_dataset_dark_epoch,
    get_dataset_light_epoch,
)
from v1ca1.task_progression import dark_light_glm as dlg
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    compute_movement_firing_rates,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)
from v1ca1.task_progression.tuning_analysis import (
    _movement_interval_axis,
    _movement_positions_to_times,
    _times_to_movement_positions,
)


DEFAULT_BIN_SIZE_S = 0.05
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 3
DEFAULT_N_SHUFFLES = 100
DEFAULT_SEED = 47
DEFAULT_SHUFFLE_MIN_SHIFT_S = 0.5
DEFAULT_MIN_SELECTION_UNITS = 5
DEFAULT_EMPIRICAL_SPATIAL_BIN_SIZE_CM = 4.0
DEFAULT_EMPIRICAL_SIGMA_BINS = 1.0
DEFAULT_MIN_TUNING_STABILITY_CORRELATION = 0.5
STABILITY_TABLE_FILENAME = "odd_even_task_progression_stability.parquet"
DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
ESTIMATOR_CHOICES = ("empirical", "glm")
DARK_MODEL_SCOPE = "trajectory"
TRANSFER_DENOMINATOR_EPS = 1e-12
_EMPIRICAL_BACKEND_NOTICE_PRINTED = False


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted start and end arrays from one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).reshape(-1)
    ends = np.asarray(intervals.end, dtype=float).reshape(-1)
    if starts.shape != ends.shape:
        raise ValueError(
            "Interval start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts)
    return starts[order], ends[order]


def _make_intervalset_like(reference: Any, starts: np.ndarray, ends: np.ndarray) -> Any:
    """Return an IntervalSet using the same class as `reference`."""
    intervalset_class = reference.__class__
    return intervalset_class(
        start=np.asarray(starts, dtype=float),
        end=np.asarray(ends, dtype=float),
        time_units="s",
    )


def _subset_intervalset(intervals: Any, indices: np.ndarray) -> Any:
    """Return one IntervalSet containing selected rows from `intervals`."""
    starts, ends = _extract_interval_bounds(intervals)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    return _make_intervalset_like(intervals, starts[indices], ends[indices])


def _combine_intervalsets(intervals: Sequence[Any]) -> Any:
    """Return one sorted IntervalSet containing all intervals in `intervals`."""
    if not intervals:
        raise ValueError("At least one IntervalSet is required.")
    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    for interval in intervals:
        starts, ends = _extract_interval_bounds(interval)
        if starts.size:
            starts_list.append(starts)
            ends_list.append(ends)
    if not starts_list:
        return _make_intervalset_like(
            intervals[0],
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    starts = np.concatenate(starts_list).astype(float, copy=False)
    ends = np.concatenate(ends_list).astype(float, copy=False)
    order = np.argsort(starts)
    return _make_intervalset_like(intervals[0], starts[order], ends[order])


def _restrict_laps_to_movement(
    trajectory_intervals: dict[str, dict[str, Any]],
    movement_by_run: dict[str, Any],
    *,
    epoch: str,
    trajectory: str,
    lap_indices: np.ndarray,
) -> Any:
    """Return selected trajectory laps intersected with movement intervals."""
    return _subset_intervalset(
        trajectory_intervals[epoch][trajectory],
        np.asarray(lap_indices, dtype=int),
    ).intersect(movement_by_run[epoch])


def build_lap_index_folds(
    trajectory_intervals: dict[str, Any],
    *,
    n_folds: int,
    seed: int,
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Build balanced lap-index CV folds independently within each trajectory."""
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    folds_by_trajectory: dict[str, list[dict[str, np.ndarray]]] = {}
    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        starts, _ends = _extract_interval_bounds(trajectory_intervals[trajectory])
        n_laps = int(starts.size)
        if n_laps < n_folds:
            raise ValueError(
                f"Trajectory {trajectory!r} has {n_laps} lap(s), fewer than "
                f"n_folds={n_folds}."
            )
        kfold = KFold(
            n_splits=int(n_folds),
            shuffle=True,
            random_state=int(seed) + trajectory_index,
        )
        trajectory_folds: list[dict[str, np.ndarray]] = []
        for train_indices, test_indices in kfold.split(np.arange(n_laps)):
            trajectory_folds.append(
                {
                    "train_indices": np.sort(train_indices.astype(int, copy=False)),
                    "test_indices": np.sort(test_indices.astype(int, copy=False)),
                }
            )
        folds_by_trajectory[trajectory] = trajectory_folds
    return folds_by_trajectory


def build_inner_lap_index_folds(
    outer_train_indices: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """Split one outer-training lap set into inner train/validation folds."""
    lap_indices = np.asarray(outer_train_indices, dtype=int).reshape(-1)
    if n_folds < 2:
        raise ValueError("--inner-n-folds must be at least 2.")
    if lap_indices.size < n_folds:
        raise ValueError(
            f"Outer train set has {lap_indices.size} lap(s), fewer than "
            f"inner_n_folds={n_folds}."
        )

    kfold = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    folds: list[dict[str, np.ndarray]] = []
    for train_pos, validation_pos in kfold.split(np.arange(lap_indices.size)):
        folds.append(
            {
                "train_indices": np.sort(lap_indices[train_pos].astype(int, copy=False)),
                "validation_indices": np.sort(
                    lap_indices[validation_pos].astype(int, copy=False)
                ),
            }
        )
    return folds


def validate_shift_range(
    *,
    duration_s: float,
    min_shift_s: float,
    max_shift_s: float | None,
) -> tuple[float, float]:
    """Return a valid circular-shift range for one concatenated interval axis."""
    duration = float(duration_s)
    min_shift = float(min_shift_s)
    if duration <= 0.0:
        raise ValueError("Cannot circularly shift spikes in an empty interval.")
    if min_shift < 0.0:
        raise ValueError("--shuffle-min-shift-s must be non-negative.")

    auto_max_shift = duration - min_shift
    max_shift = auto_max_shift if max_shift_s is None else float(max_shift_s)
    if max_shift > auto_max_shift:
        raise ValueError(
            "Requested shuffle max shift exceeds the movement-axis duration after "
            "leaving the minimum wrap margin: "
            f"duration={duration:.6g}, min_shift={min_shift:.6g}, "
            f"max_shift={max_shift:.6g}."
        )
    if max_shift <= min_shift:
        raise ValueError(
            "Trajectory train interval is too short for the requested circular "
            "shift range: "
            f"duration={duration:.6g}, min_shift={min_shift:.6g}, "
            f"max_shift={max_shift:.6g}."
        )
    return min_shift, max_shift


def circular_shift_unit_spikes_on_interval_axis(
    unit_spikes: Any,
    intervals: Any,
    *,
    rng: np.random.Generator,
    min_shift_s: float,
    max_shift_s: float | None,
) -> Any:
    """Circularly shift one unit's spikes on a concatenated interval axis."""
    import pynapple as nap

    restricted_spikes = unit_spikes.restrict(intervals)
    spike_times = np.asarray(restricted_spikes.t, dtype=float)
    if spike_times.size == 0:
        return nap.Ts(t=np.asarray([], dtype=float), time_units="s")

    starts, ends, axis_starts, total_length = _movement_interval_axis(intervals)
    min_shift, max_shift = validate_shift_range(
        duration_s=total_length,
        min_shift_s=min_shift_s,
        max_shift_s=max_shift_s,
    )
    shift_amount = float(rng.uniform(min_shift, max_shift))
    positions = _times_to_movement_positions(
        spike_times,
        starts=starts,
        ends=ends,
        axis_starts=axis_starts,
    )
    shifted_positions = np.mod(positions + shift_amount, total_length)
    shifted_times = _movement_positions_to_times(
        shifted_positions,
        starts=starts,
        ends=ends,
        axis_starts=axis_starts,
    )
    shifted_times.sort()
    return nap.Ts(t=shifted_times, time_units="s")


def circular_shift_spikes_on_interval_axis(
    spikes: Any,
    intervals: Any,
    *,
    rng: np.random.Generator,
    min_shift_s: float,
    max_shift_s: float | None,
) -> Any:
    """Circularly shift each unit independently on a concatenated interval axis."""
    import pynapple as nap

    shifted = {
        unit: circular_shift_unit_spikes_on_interval_axis(
            spikes[unit],
            intervals,
            rng=rng,
            min_shift_s=min_shift_s,
            max_shift_s=max_shift_s,
        )
        for unit in spikes.keys()
    }
    return nap.TsGroup(shifted)


def _seed_from_parts(base_seed: int, *parts: object) -> np.random.Generator:
    """Return a deterministic RNG from a base seed and stable text components."""
    import zlib

    seed_parts = [int(base_seed) & 0xFFFFFFFF]
    for part in parts:
        seed_parts.append(zlib.crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
    return np.random.default_rng(np.random.SeedSequence(seed_parts))


def _selected_spikes(spikes: Any, unit_mask: np.ndarray) -> Any:
    """Return a TsGroup-like object containing only selected units."""
    return spikes[np.asarray(unit_mask, dtype=bool)]


def get_tuning_stability_table_path(analysis_path: Path) -> Path:
    """Return the saved odd/even task-progression stability table path."""
    return (
        get_task_progression_output_dir(analysis_path, "stability")
        / STABILITY_TABLE_FILENAME
    )


def load_tuning_stability_table(
    *,
    analysis_path: Path,
    animal_name: str,
    date: str,
) -> pd.DataFrame:
    """Load saved odd/even tuning stability rows or raise an actionable error."""
    table_path = get_tuning_stability_table_path(analysis_path)
    if not table_path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{table_path}. Run "
            "`python -m v1ca1.task_progression.stability "
            f"--animal-name {animal_name} --date {date}` first."
        )
    return pd.read_parquet(table_path)


def select_stable_units_for_epoch(
    stability_table: pd.DataFrame,
    *,
    region: str,
    epoch: str,
    min_correlation: float,
) -> np.ndarray:
    """Return units stable in at least one trajectory for one region/epoch."""
    required_columns = {"unit", "region", "epoch", "trajectory_type", "stability_correlation"}
    missing_columns = required_columns.difference(stability_table.columns)
    if missing_columns:
        raise ValueError(
            "Stability table is missing required column(s): "
            + ", ".join(sorted(missing_columns))
        )

    correlations = stability_table["stability_correlation"].to_numpy(dtype=float)
    stable_rows = stability_table[
        (stability_table["region"].astype(str) == str(region))
        & (stability_table["epoch"].astype(str) == str(epoch))
        & (stability_table["trajectory_type"].astype(str).isin(TRAJECTORY_TYPES))
        & np.isfinite(correlations)
        & (correlations >= float(min_correlation))
    ]
    return np.asarray(stable_rows["unit"].drop_duplicates())


def build_unit_filter_diagnostics(
    *,
    unit_ids: np.ndarray,
    fr_mask: np.ndarray,
    light_stable_units: np.ndarray | None,
    dark_stable_units: np.ndarray | None,
    min_stability_correlation: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return final unit mask and filter-count diagnostics."""
    units = np.asarray(unit_ids)
    fr = np.asarray(fr_mask, dtype=bool).reshape(-1)
    if units.shape[0] != fr.shape[0]:
        raise ValueError(
            "Unit ids and FR mask must have matching length. "
            f"Got {units.shape[0]} and {fr.shape[0]}."
        )

    filter_enabled = (
        light_stable_units is not None
        and dark_stable_units is not None
        and min_stability_correlation is not None
    )
    if filter_enabled:
        light_stable_mask = np.isin(units, np.asarray(light_stable_units))
        dark_stable_mask = np.isin(units, np.asarray(dark_stable_units))
    else:
        light_stable_mask = np.ones(units.shape[0], dtype=bool)
        dark_stable_mask = np.ones(units.shape[0], dtype=bool)

    final_mask = fr & light_stable_mask & dark_stable_mask
    diagnostics = {
        "tuning_stability_filter_enabled": bool(filter_enabled),
        "min_tuning_stability_correlation": (
            np.nan
            if min_stability_correlation is None
            else float(min_stability_correlation)
        ),
        "n_units_total": int(units.shape[0]),
        "n_units_fr_pass": int(np.sum(fr)),
        "n_units_light_stable": int(np.sum(light_stable_mask)),
        "n_units_dark_stable": int(np.sum(dark_stable_mask)),
        "n_units_fr_and_light_stable": int(np.sum(fr & light_stable_mask)),
        "n_units_fr_and_dark_stable": int(np.sum(fr & dark_stable_mask)),
        "n_units_final": int(np.sum(final_mask)),
    }
    return final_mask, diagnostics


def prepare_glm_inputs(
    *,
    spikes: Any,
    epoch: str,
    trajectories: Sequence[str],
    intervals_by_trajectory: dict[str, Any],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    bin_size_s: float,
) -> dict[str, Any]:
    """Assemble binned counts, task progression, and optional speed values."""
    responses: list[np.ndarray] = []
    task_progression: list[np.ndarray] = []
    speed_values: list[np.ndarray] = []
    trajectory_labels: list[np.ndarray] = []
    unit_ids: np.ndarray | None = None

    has_speed = speed_by_epoch is not None
    for trajectory in trajectories:
        interval = intervals_by_trajectory[trajectory]
        if float(interval.tot_length()) <= 0.0:
            raise ValueError(
                f"Trajectory interval {epoch!r}/{trajectory!r} is empty after "
                "movement restriction."
            )

        counts = spikes.count(float(bin_size_s), ep=interval)
        current_unit_ids = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = current_unit_ids
        elif current_unit_ids.shape != unit_ids.shape or not np.all(
            current_unit_ids == unit_ids
        ):
            raise ValueError("Spike count columns differ across trajectories.")

        y = np.asarray(counts.d, dtype=float)
        p = tp_by_epoch[epoch][trajectory].interpolate(counts).to_numpy().reshape(-1)
        if has_speed:
            speed = speed_by_epoch[epoch].interpolate(counts).to_numpy().reshape(-1)
            good = np.isfinite(p) & np.isfinite(speed)
            speed_values.append(np.asarray(speed[good], dtype=float))
        else:
            good = np.isfinite(p)

        if not np.any(good):
            raise ValueError(
                f"No finite task-progression samples for {epoch!r}/{trajectory!r}."
            )
        responses.append(np.asarray(y[good], dtype=float))
        task_progression.append(np.asarray(p[good], dtype=float))
        trajectory_labels.append(
            np.asarray([trajectory] * int(np.sum(good)), dtype=object)
        )

    if unit_ids is None:
        raise ValueError("No trajectories were provided.")

    y_all = np.concatenate(responses, axis=0)
    p_all = np.concatenate(task_progression, axis=0)
    if has_speed:
        v_all = np.concatenate(speed_values, axis=0)
    else:
        v_all = None
    return {
        "unit_ids": np.asarray(unit_ids),
        "y": np.asarray(y_all, dtype=float),
        "p": np.asarray(p_all, dtype=float).reshape(-1),
        "v": None if v_all is None else np.asarray(v_all, dtype=float).reshape(-1),
        "trajectory": np.concatenate(trajectory_labels),
    }


def _numba_jit_disabled() -> bool:
    """Return whether numba JIT has been disabled for this process."""
    value = os.environ.get("NUMBA_DISABLE_JIT", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _print_empirical_backend_notice(reason: str) -> None:
    """Print the empirical backend fallback notice at most once."""
    global _EMPIRICAL_BACKEND_NOTICE_PRINTED
    if _EMPIRICAL_BACKEND_NOTICE_PRINTED:
        return
    print(f"  Empirical tuning backend: binned counts ({reason}).")
    _EMPIRICAL_BACKEND_NOTICE_PRINTED = True


def compute_binned_empirical_tuning_curves(
    train_inputs: dict[str, Any],
    *,
    bin_edges: np.ndarray,
    bin_size_s: float,
) -> np.ndarray:
    """Return firing-rate tuning curves from binned counts and TP samples."""
    y = np.asarray(train_inputs["y"], dtype=float)
    p = np.asarray(train_inputs["p"], dtype=float).reshape(-1)
    edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    if y.ndim != 2:
        raise ValueError("train_inputs['y'] must be a 2-D array.")
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            "Task-progression samples and spike-count rows must match. "
            f"Got {p.shape[0]} and {y.shape[0]}."
        )
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be a strictly increasing 1-D array.")
    if bin_size_s <= 0.0:
        raise ValueError("bin_size_s must be positive.")

    n_bins = edges.size - 1
    rates_hz = np.full((n_bins, y.shape[1]), np.nan, dtype=float)
    good = np.isfinite(p) & (p >= edges[0]) & (p <= edges[-1])
    if not np.any(good):
        return rates_hz

    bin_index = np.digitize(p[good], edges) - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    y_good = y[good]
    for current_bin in range(n_bins):
        in_bin = bin_index == current_bin
        if np.any(in_bin):
            duration_s = float(np.sum(in_bin)) * float(bin_size_s)
            rates_hz[current_bin] = np.sum(y_good[in_bin], axis=0) / duration_s
    return rates_hz


def build_task_progression_bin_edges(
    *,
    animal_name: str,
    spatial_bin_size_cm: float,
) -> tuple[np.ndarray, int, float]:
    """Return normalized TP bin edges from a spatial bin size in centimeters."""
    spatial_bin_size = float(spatial_bin_size_cm)
    if spatial_bin_size <= 0.0:
        raise ValueError("--spatial-bin-size-cm must be positive.")
    trajectory_length_cm = float(dlg.get_wtrack_total_length(animal_name))
    bin_count = max(1, int(np.ceil(trajectory_length_cm / spatial_bin_size)))
    return np.linspace(0.0, 1.0, bin_count + 1), bin_count, trajectory_length_cm


def interpolate_empirical_curve(
    curve_hz: np.ndarray,
    *,
    fallback_rate_hz: float,
) -> tuple[np.ndarray, bool]:
    """Interpolate NaNs in one empirical tuning curve or use a mean-rate fallback."""
    values = np.asarray(curve_hz, dtype=float).reshape(-1)
    fallback = max(float(fallback_rate_hz), 1e-12)
    if values.size == 0:
        return np.asarray([], dtype=float), True
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, fallback, dtype=float), True
    if np.all(finite):
        return values, False

    bin_index = np.arange(values.size, dtype=float)
    interpolated = np.interp(bin_index, bin_index[finite], values[finite])
    return interpolated.astype(float, copy=False), False


def smooth_empirical_curve(
    curve_hz: np.ndarray,
    *,
    sigma_bins: float,
    fallback_rate_hz: float,
) -> tuple[np.ndarray, bool]:
    """Interpolate, smooth, and sanitize one empirical tuning curve."""
    from scipy.ndimage import gaussian_filter1d

    interpolated, used_fallback = interpolate_empirical_curve(
        curve_hz,
        fallback_rate_hz=fallback_rate_hz,
    )
    if interpolated.size == 0:
        return interpolated, used_fallback

    if float(sigma_bins) > 0.0:
        smoothed = gaussian_filter1d(
            interpolated,
            sigma=float(sigma_bins),
            mode="nearest",
        )
    else:
        smoothed = interpolated
    if not np.all(np.isfinite(smoothed)):
        smoothed, fallback_after_smoothing = interpolate_empirical_curve(
            smoothed,
            fallback_rate_hz=fallback_rate_hz,
        )
        used_fallback = used_fallback or fallback_after_smoothing
    return np.maximum(np.asarray(smoothed, dtype=float), 1e-12), used_fallback


def _combined_task_progression_feature(
    *,
    epoch: str,
    trajectories: Sequence[str],
    intervals_by_trajectory: dict[str, Any],
    tp_by_epoch: dict[str, dict[str, Any]],
) -> tuple[Any, Any]:
    """Return one Tsd and IntervalSet pooled across selected trajectories."""
    import pynapple as nap

    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    intervals: list[Any] = []
    for trajectory in trajectories:
        interval = intervals_by_trajectory[trajectory]
        intervals.append(interval)
        restricted = tp_by_epoch[epoch][trajectory].restrict(interval)
        t = np.asarray(restricted.t, dtype=float).reshape(-1)
        d = np.asarray(restricted.to_numpy(), dtype=float).reshape(-1)
        if t.shape != d.shape:
            raise ValueError(
                f"Task-progression samples have mismatched time/value shapes for "
                f"{epoch!r}/{trajectory!r}: {t.shape} vs {d.shape}."
            )
        good = np.isfinite(t) & np.isfinite(d)
        if np.any(good):
            times.append(t[good])
            values.append(d[good])

    combined_interval = _combine_intervalsets(intervals)
    if not times:
        raise ValueError(
            f"No finite task-progression samples were available for {epoch!r} "
            f"trajectories {list(trajectories)!r}."
        )
    t_all = np.concatenate(times)
    d_all = np.concatenate(values)
    order = np.argsort(t_all)
    return (
        nap.Tsd(t=t_all[order], d=d_all[order], time_units="s"),
        combined_interval,
    )


def fit_empirical_tuning(
    *,
    spikes: Any,
    epoch: str,
    trajectories: Sequence[str],
    intervals_by_trajectory: dict[str, Any],
    tp_by_epoch: dict[str, dict[str, Any]],
    bin_edges: np.ndarray,
    bin_size_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Estimate one empirical TP tuning curve model with pynapple."""
    import pynapple as nap

    train_inputs = prepare_glm_inputs(
        spikes=spikes,
        epoch=epoch,
        trajectories=trajectories,
        intervals_by_trajectory=intervals_by_trajectory,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=None,
        bin_size_s=bin_size_s,
    )
    tuning_curves = None
    raw_rate_matrix: np.ndarray | None = None
    empirical_backend = "pynapple"
    if _numba_jit_disabled():
        empirical_backend = "binned_counts"
        _print_empirical_backend_notice(
            "NUMBA_DISABLE_JIT is set, so pynapple value_from is bypassed"
        )
        raw_rate_matrix = compute_binned_empirical_tuning_curves(
            train_inputs,
            bin_edges=bin_edges,
            bin_size_s=bin_size_s,
        )
    else:
        feature, combined_interval = _combined_task_progression_feature(
            epoch=epoch,
            trajectories=trajectories,
            intervals_by_trajectory=intervals_by_trajectory,
            tp_by_epoch=tp_by_epoch,
        )
        try:
            tuning_curves = nap.compute_tuning_curves(
                data=spikes,
                features=feature,
                bins=[np.asarray(bin_edges, dtype=float)],
                epochs=combined_interval,
                feature_names=["tp"],
            )
        except UnboundLocalError as exc:
            empirical_backend = "binned_counts"
            _print_empirical_backend_notice(
                f"pynapple compute_tuning_curves failed with {exc.__class__.__name__}"
            )
            raw_rate_matrix = compute_binned_empirical_tuning_curves(
                train_inputs,
                bin_edges=bin_edges,
                bin_size_s=bin_size_s,
            )

    unit_ids = np.asarray(train_inputs["unit_ids"])
    curves: list[np.ndarray] = []
    fallback_flags: list[bool] = []
    y_train = np.asarray(train_inputs["y"], dtype=float)
    for unit_index, unit in enumerate(unit_ids):
        if raw_rate_matrix is None:
            raw_curve = np.asarray(
                tuning_curves.sel(unit=unit).values,
                dtype=float,
            ).reshape(-1)
        else:
            raw_curve = np.asarray(raw_rate_matrix[:, unit_index], dtype=float)
        train_duration = max(y_train.shape[0] * float(bin_size_s), 1e-12)
        fallback_rate = float(np.sum(y_train[:, unit_index]) / train_duration)
        curve, used_fallback = smooth_empirical_curve(
            raw_curve,
            sigma_bins=sigma_bins,
            fallback_rate_hz=fallback_rate,
        )
        curves.append(curve)
        fallback_flags.append(bool(used_fallback))

    return {
        "estimator": "empirical",
        "unit_ids": unit_ids,
        "bin_edges": np.asarray(bin_edges, dtype=float),
        "rates_hz": np.stack(curves, axis=1),
        "fallback_to_mean": np.asarray(fallback_flags, dtype=bool),
        "train_inputs": train_inputs,
        "sigma_bins": float(sigma_bins),
        "empirical_backend": empirical_backend,
    }


def predict_empirical_counts(
    fit: dict[str, Any],
    inputs: dict[str, Any],
    *,
    bin_size_s: float,
) -> np.ndarray:
    """Return predicted spike counts from one empirical tuning model."""
    bin_edges = np.asarray(fit["bin_edges"], dtype=float)
    rates_hz = np.asarray(fit["rates_hz"], dtype=float)
    positions = np.asarray(inputs["p"], dtype=float).reshape(-1)
    bin_index = np.digitize(positions, bin_edges) - 1
    bin_index = np.clip(bin_index, 0, rates_hz.shape[0] - 1)
    return np.maximum(rates_hz[bin_index] * float(bin_size_s), 1e-12)


def score_empirical_model(
    *,
    fit: dict[str, Any],
    test_inputs: dict[str, Any],
    light_train_inputs: dict[str, Any],
    bin_size_s: float,
) -> dict[str, np.ndarray]:
    """Predict and score one empirical tuning model on light-test inputs."""
    lam_pred = predict_empirical_counts(fit, test_inputs, bin_size_s=bin_size_s)
    return score_glm_prediction(
        y_test=test_inputs["y"],
        lam_pred=lam_pred,
        y_light_train_baseline=light_train_inputs["y"],
    )


def _build_tp_design(
    task_progression: np.ndarray,
    *,
    n_splines: int,
    spline_order: int,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Return the task-progression spline design matrix."""
    dlg._require_nemos()
    basis = dlg.BSplineEval(
        n_basis_funcs=int(n_splines),
        order=int(spline_order),
        bounds=pos_bounds,
    )
    return np.asarray(
        basis.compute_features(np.asarray(task_progression, dtype=float).reshape(-1)),
        dtype=float,
    )


def fit_poisson_tp_glm(
    train_inputs: dict[str, Any],
    *,
    n_splines: int,
    spline_order: int,
    ridge: float,
    use_speed: bool,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    """Fit one Poisson task-progression GLM and return prediction metadata."""
    dlg._require_nemos()
    y_train = np.asarray(train_inputs["y"], dtype=float)
    if y_train.ndim != 2 or y_train.shape[0] == 0:
        raise ValueError(f"Expected nonempty 2D train response, got {y_train.shape}.")

    x_tp = _build_tp_design(
        train_inputs["p"],
        n_splines=n_splines,
        spline_order=spline_order,
    )
    if use_speed:
        if train_inputs["v"] is None:
            raise ValueError("Speed inputs are required when use_speed=True.")
        speed_transform = dlg._fit_speed_feature_transform(
            train_inputs["v"],
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
        x_speed = dlg._transform_speed_with_feature_transform(
            train_inputs["v"],
            speed_transform,
        )
    else:
        speed_transform = dlg._empty_speed_feature_transform()
        x_speed = dlg._empty_speed_design(y_train.shape[0])

    x_train = np.concatenate([x_tp, x_speed], axis=1)
    model = dlg.PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model.fit(x_train, y_train)
    return {
        "model": model,
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "speed_transform": speed_transform,
    }


def _subset_train_input_units(
    train_inputs: dict[str, Any],
    unit_mask: np.ndarray,
) -> dict[str, Any]:
    """Return train inputs with response columns limited to selected units."""
    mask = np.asarray(unit_mask, dtype=bool).reshape(-1)
    y = np.asarray(train_inputs["y"], dtype=float)
    if y.ndim != 2 or y.shape[1] != mask.shape[0]:
        raise ValueError(
            "Unit mask must match train response columns. "
            f"Got {mask.shape[0]} and {y.shape[1]}."
        )
    subset = dict(train_inputs)
    subset["y"] = y[:, mask]
    subset["unit_ids"] = np.asarray(train_inputs["unit_ids"])[mask]
    return subset


def fit_poisson_tp_glm_with_zero_count_fallback(
    train_inputs: dict[str, Any],
    *,
    n_splines: int,
    spline_order: int,
    ridge: float,
    use_speed: bool,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    """Fit a GLM, using dark-train mean fallback for all-zero unit columns."""
    y_train = np.asarray(train_inputs["y"], dtype=float)
    if y_train.ndim != 2 or y_train.shape[0] == 0:
        raise ValueError(f"Expected nonempty 2D train response, got {y_train.shape}.")

    fallback_to_mean = np.sum(y_train, axis=0) <= 0.0
    fallback_mean_count = np.clip(np.mean(y_train, axis=0), 1e-12, None)
    fit_unit_mask = ~fallback_to_mean
    if np.any(fit_unit_mask):
        fit = fit_poisson_tp_glm(
            _subset_train_input_units(train_inputs, fit_unit_mask),
            n_splines=n_splines,
            spline_order=spline_order,
            ridge=ridge,
            use_speed=use_speed,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
    else:
        fit = {
            "model": None,
            "n_splines": int(n_splines),
            "spline_order": int(spline_order),
            "speed_transform": dlg._empty_speed_feature_transform(),
        }

    fit.update(
        {
            "fit_unit_mask": fit_unit_mask,
            "fallback_to_mean": fallback_to_mean,
            "fallback_mean_count": fallback_mean_count,
            "unit_ids": np.asarray(train_inputs["unit_ids"]),
        }
    )
    return fit


def predict_poisson_counts(
    fit: dict[str, Any],
    inputs: dict[str, Any],
) -> np.ndarray:
    """Return predicted spike counts per bin and unit from one fit."""
    fit_unit_mask = fit.get("fit_unit_mask")
    if fit_unit_mask is not None:
        mask = np.asarray(fit_unit_mask, dtype=bool).reshape(-1)
        fallback_mean_count = np.asarray(
            fit["fallback_mean_count"],
            dtype=float,
        ).reshape(-1)
        n_time = np.asarray(inputs["p"], dtype=float).reshape(-1).shape[0]
        if mask.shape[0] != fallback_mean_count.shape[0]:
            raise ValueError(
                "fit_unit_mask and fallback_mean_count must have matching length."
            )
        lam_full = np.repeat(
            np.clip(fallback_mean_count, 1e-12, None)[None, :],
            n_time,
            axis=0,
        )
        if fit.get("model") is None:
            return lam_full
        fit_without_fallback = {
            key: value
            for key, value in fit.items()
            if key
            not in {
                "fit_unit_mask",
                "fallback_to_mean",
                "fallback_mean_count",
                "unit_ids",
            }
        }
        lam_full[:, mask] = predict_poisson_counts(fit_without_fallback, inputs)
        return lam_full

    x_tp = _build_tp_design(
        inputs["p"],
        n_splines=int(fit["n_splines"]),
        spline_order=int(fit["spline_order"]),
    )
    speed_transform = fit["speed_transform"]
    if str(speed_transform["mode"]) == "none":
        x_speed = dlg._empty_speed_design(x_tp.shape[0])
    else:
        if inputs["v"] is None:
            raise ValueError("Speed inputs are required for a speed-enabled fit.")
        x_speed = dlg._transform_speed_with_feature_transform(
            inputs["v"],
            speed_transform,
        )
    x = np.concatenate([x_tp, x_speed], axis=1)
    model = fit["model"]
    coef = dlg._coef_feat_by_unit(model, n_features=x.shape[1])
    eta = np.asarray(model.intercept_).reshape(1, -1) + (x @ coef)
    return np.exp(np.clip(eta, -30.0, 30.0))


def bits_per_spike_from_sums(
    ll_sum: np.ndarray,
    null_ll_sum: np.ndarray,
    spike_sum: np.ndarray,
) -> np.ndarray:
    """Return null-referenced information gain in bits/spike."""
    ll_sum = np.asarray(ll_sum, dtype=float)
    null_ll_sum = np.asarray(null_ll_sum, dtype=float)
    spike_sum = np.asarray(spike_sum, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            spike_sum > 0,
            (ll_sum - null_ll_sum) / (spike_sum * np.log(2.0)),
            np.nan,
        )


def score_glm_prediction(
    *,
    y_test: np.ndarray,
    lam_pred: np.ndarray,
    y_light_train_baseline: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return held-out LL, target-light null LL, spikes, and bits/spike."""
    y_test = np.asarray(y_test, dtype=float)
    lam_pred = np.asarray(lam_pred, dtype=float)
    y_null_fit = np.asarray(y_light_train_baseline, dtype=float)
    if y_test.shape[1] != y_null_fit.shape[1]:
        raise ValueError(
            "Test response and null-fit response must have matching unit counts. "
            f"Got {y_test.shape[1]} and {y_null_fit.shape[1]}."
        )
    ll_sum = dlg._poisson_ll_sum(y_test, lam_pred)
    null_rate = np.clip(np.mean(y_null_fit, axis=0), 1e-12, None)
    null_lam = np.repeat(null_rate[None, :], y_test.shape[0], axis=0)
    null_ll_sum = dlg._poisson_ll_sum(y_test, null_lam)
    spike_sum = np.asarray(np.sum(y_test, axis=0), dtype=float)
    return {
        "ll_sum": ll_sum,
        "null_ll_sum": null_ll_sum,
        "spike_sum": spike_sum,
        "bits_per_spike": bits_per_spike_from_sums(
            ll_sum,
            null_ll_sum,
            spike_sum,
        ),
    }


def score_one_model(
    *,
    fit: dict[str, Any],
    test_inputs: dict[str, Any],
    light_train_inputs: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Predict and score one fit on intact light-test inputs."""
    lam_pred = predict_poisson_counts(fit, test_inputs)
    return score_glm_prediction(
        y_test=test_inputs["y"],
        lam_pred=lam_pred,
        y_light_train_baseline=light_train_inputs["y"],
    )


def make_score_rows(
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model: str,
    fold: int,
    trajectory: str,
    unit_ids: np.ndarray,
    metrics: dict[str, np.ndarray],
    selected: dict[str, Any],
    bin_size_s: float,
    shuffle_index: int | None = None,
    fallback_to_mean: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Return long-form per-unit score rows for one fold/trajectory/model."""
    rows: list[dict[str, Any]] = []
    if fallback_to_mean is None:
        fallback = np.full(unit_ids.shape, False, dtype=bool)
    else:
        fallback = np.asarray(fallback_to_mean, dtype=bool).reshape(-1)
        if fallback.shape[0] != unit_ids.shape[0]:
            raise ValueError(
                "fallback_to_mean must have one value per unit. "
                f"Got {fallback.shape[0]} and {unit_ids.shape[0]}."
            )
    for unit_index, unit in enumerate(unit_ids):
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "light_epoch": light_epoch,
                "dark_epoch": dark_epoch,
                "estimator": selected.get("estimator", "glm"),
                "fold": int(fold),
                "trajectory": trajectory,
                "dark_model_scope": selected.get(
                    "dark_model_scope",
                    DARK_MODEL_SCOPE,
                ),
                "model": model,
                "shuffle_index": (
                    np.nan if shuffle_index is None else int(shuffle_index)
                ),
                "unit": unit,
                "ll_sum": float(metrics["ll_sum"][unit_index]),
                "null_ll_sum": float(metrics["null_ll_sum"][unit_index]),
                "spike_sum": float(metrics["spike_sum"][unit_index]),
                "bits_per_spike": float(metrics["bits_per_spike"][unit_index]),
                "selected_ridge": float(selected.get("ridge", np.nan)),
                "selected_spatial_bin_size_cm": float(
                    selected.get("spatial_bin_size_cm", np.nan)
                ),
                "selected_n_splines": selected.get("n_splines", np.nan),
                "selected_tp_bin_count": selected.get("tp_bin_count", np.nan),
                "sigma_bins": selected.get("sigma_bins", np.nan),
                "bin_size_s": float(bin_size_s),
                "tuning_curve_fallback_to_mean": bool(fallback[unit_index]),
            }
        )
    return rows


def _is_better_hyperparameter_record(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> bool:
    """Return whether one candidate wins score and deterministic tie-breaks."""
    candidate_score = float(candidate["score_median"])
    if not np.isfinite(candidate_score):
        return False
    if incumbent is None:
        return True
    incumbent_score = float(incumbent["score_median"])
    if candidate_score > incumbent_score + dlg.HYPERPARAMETER_TIE_ATOL:
        return True
    if candidate_score < incumbent_score - dlg.HYPERPARAMETER_TIE_ATOL:
        return False
    if not np.isclose(
        float(candidate["spatial_bin_size_cm"]),
        float(incumbent["spatial_bin_size_cm"]),
    ):
        return float(candidate["spatial_bin_size_cm"]) > float(
            incumbent["spatial_bin_size_cm"]
        )
    return float(candidate["ridge"]) > float(incumbent["ridge"])


def choose_hyperparameter_record(
    records: Sequence[dict[str, Any]],
    *,
    min_selection_units: int,
) -> dict[str, Any]:
    """Choose ridge and TP basis from finite inner-CV median bits/spike."""
    best: dict[str, Any] | None = None
    for record in records:
        if int(record["n_finite_units"]) < int(min_selection_units):
            continue
        if _is_better_hyperparameter_record(record, best):
            best = dict(record)
    if best is None:
        raise ValueError(
            "No hyperparameter candidate had enough finite inner-CV unit scores "
            f"(min_selection_units={min_selection_units})."
        )
    return best


def select_outer_fold_hyperparameters(
    *,
    spikes: Any,
    light_epoch: str,
    outer_fold_index: int,
    outer_folds: dict[str, list[dict[str, np.ndarray]]],
    trajectory_intervals: dict[str, dict[str, Any]],
    movement_by_run: dict[str, Any],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    position_basis_configs: Sequence[dict[str, Any]],
    ridge_values: Sequence[float],
    inner_n_folds: int,
    seed: int,
    bin_size_s: float,
    spline_order: int,
    use_speed: bool,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
    min_selection_units: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select ridge and task-progression basis using intact light-train inner CV."""
    reference_unit_ids: np.ndarray | None = None
    selection_records: list[dict[str, Any]] = []

    for basis_config in position_basis_configs:
        n_splines = int(basis_config["n_splines"])
        spatial_bin_size_cm = float(basis_config["spatial_bin_size_cm"])
        for ridge in ridge_values:
            ll_sum: np.ndarray | None = None
            null_ll_sum: np.ndarray | None = None
            spike_sum: np.ndarray | None = None

            for trajectory in TRAJECTORY_TYPES:
                outer_train_indices = outer_folds[trajectory][outer_fold_index][
                    "train_indices"
                ]
                inner_folds = build_inner_lap_index_folds(
                    outer_train_indices,
                    n_folds=inner_n_folds,
                    seed=int(seed) + (1000 * int(outer_fold_index)),
                )
                for inner_fold in inner_folds:
                    train_interval = _restrict_laps_to_movement(
                        trajectory_intervals,
                        movement_by_run,
                        epoch=light_epoch,
                        trajectory=trajectory,
                        lap_indices=inner_fold["train_indices"],
                    )
                    validation_interval = _restrict_laps_to_movement(
                        trajectory_intervals,
                        movement_by_run,
                        epoch=light_epoch,
                        trajectory=trajectory,
                        lap_indices=inner_fold["validation_indices"],
                    )
                    interval_map_train = {trajectory: train_interval}
                    interval_map_validation = {trajectory: validation_interval}
                    train_inputs = prepare_glm_inputs(
                        spikes=spikes,
                        epoch=light_epoch,
                        trajectories=[trajectory],
                        intervals_by_trajectory=interval_map_train,
                        tp_by_epoch=tp_by_epoch,
                        speed_by_epoch=speed_by_epoch,
                        bin_size_s=bin_size_s,
                    )
                    validation_inputs = prepare_glm_inputs(
                        spikes=spikes,
                        epoch=light_epoch,
                        trajectories=[trajectory],
                        intervals_by_trajectory=interval_map_validation,
                        tp_by_epoch=tp_by_epoch,
                        speed_by_epoch=speed_by_epoch,
                        bin_size_s=bin_size_s,
                    )
                    fit = fit_poisson_tp_glm(
                        train_inputs,
                        n_splines=n_splines,
                        spline_order=spline_order,
                        ridge=float(ridge),
                        use_speed=use_speed,
                        speed_feature_mode=speed_feature_mode,
                        n_splines_speed=n_splines_speed,
                        spline_order_speed=spline_order_speed,
                        speed_bounds=speed_bounds,
                    )
                    metrics = score_one_model(
                        fit=fit,
                        test_inputs=validation_inputs,
                        light_train_inputs=train_inputs,
                    )
                    current_unit_ids = np.asarray(train_inputs["unit_ids"])
                    if reference_unit_ids is None:
                        reference_unit_ids = current_unit_ids
                    elif current_unit_ids.shape != reference_unit_ids.shape or not np.all(
                        current_unit_ids == reference_unit_ids
                    ):
                        raise ValueError(
                            "Inner-CV unit ids changed across folds or trajectories."
                        )
                    if ll_sum is None:
                        ll_sum = np.zeros(current_unit_ids.size, dtype=float)
                        null_ll_sum = np.zeros(current_unit_ids.size, dtype=float)
                        spike_sum = np.zeros(current_unit_ids.size, dtype=float)
                    ll_sum += np.asarray(metrics["ll_sum"], dtype=float)
                    null_ll_sum += np.asarray(metrics["null_ll_sum"], dtype=float)
                    spike_sum += np.asarray(metrics["spike_sum"], dtype=float)

            if (
                reference_unit_ids is None
                or ll_sum is None
                or null_ll_sum is None
                or spike_sum is None
            ):
                raise ValueError("No inner-CV data were available for selection.")
            bits = bits_per_spike_from_sums(ll_sum, null_ll_sum, spike_sum)
            finite = bits[np.isfinite(bits)]
            selection_records.append(
                {
                    "fold": int(outer_fold_index),
                    "spatial_bin_size_cm": spatial_bin_size_cm,
                    "trajectory_length_cm": float(
                        basis_config["trajectory_length_cm"]
                    ),
                    "n_splines": n_splines,
                    "spline_order": int(spline_order),
                    "ridge": float(ridge),
                    "score_median": (
                        float(np.median(finite)) if finite.size else np.nan
                    ),
                    "score_mean": float(np.mean(finite)) if finite.size else np.nan,
                    "n_finite_units": int(finite.size),
                }
            )

    selected = choose_hyperparameter_record(
        selection_records,
        min_selection_units=min_selection_units,
    )
    return selected, selection_records


def compute_transfer_index(
    *,
    light_bits: float,
    dark_bits: float,
    shuffle_bits: float,
    eps: float = TRANSFER_DENOMINATOR_EPS,
) -> tuple[float, str]:
    """Return transfer index and NaN reason when the denominator is unstable."""
    values = {
        "light": float(light_bits),
        "dark": float(dark_bits),
        "shuffle": float(shuffle_bits),
    }
    for name, value in values.items():
        if not np.isfinite(value):
            return np.nan, f"nonfinite_{name}"
    denominator = values["light"] - values["shuffle"]
    if denominator <= float(eps):
        return np.nan, "light_not_above_shuffle"
    return (values["dark"] - values["shuffle"]) / denominator, "valid"


def _aggregate_model_table(score_table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Aggregate fold/trajectory score rows for one non-shuffle model."""
    columns = [
        "unit",
        f"{model}_ll_sum",
        f"{model}_null_ll_sum",
        f"{model}_spike_sum",
        f"{model}_bits_per_spike",
    ]
    subset = score_table[score_table["model"] == model].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        subset.groupby("unit", observed=False)[["ll_sum", "null_ll_sum", "spike_sum"]]
        .sum()
        .reset_index()
    )
    grouped[f"{model}_ll_sum"] = grouped.pop("ll_sum")
    grouped[f"{model}_null_ll_sum"] = grouped.pop("null_ll_sum")
    grouped[f"{model}_spike_sum"] = grouped.pop("spike_sum")
    grouped[f"{model}_bits_per_spike"] = bits_per_spike_from_sums(
        grouped[f"{model}_ll_sum"].to_numpy(dtype=float),
        grouped[f"{model}_null_ll_sum"].to_numpy(dtype=float),
        grouped[f"{model}_spike_sum"].to_numpy(dtype=float),
    )
    return grouped


def _aggregate_shuffle_table(
    shuffle_table: pd.DataFrame,
    *,
    expected_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate raw shuffle rows into per-shuffle and per-unit summaries."""
    summary_columns = [
        "unit",
        "shuffle_mean_ll_sum",
        "shuffle_median_ll_sum",
        "shuffle_mean_bits_per_spike",
        "shuffle_median_bits_per_spike",
        "shuffle_mean_null_ll_sum",
        "shuffle_mean_spike_sum",
        "shuffle_complete_count",
    ]
    if shuffle_table.empty:
        return pd.DataFrame(), pd.DataFrame(columns=summary_columns)

    grouped = (
        shuffle_table.groupby(["unit", "shuffle_index"], observed=False)
        .agg(
            ll_sum=("ll_sum", "sum"),
            null_ll_sum=("null_ll_sum", "sum"),
            spike_sum=("spike_sum", "sum"),
            component_count=("trajectory", "count"),
        )
        .reset_index()
    )
    grouped["complete"] = grouped["component_count"] == int(expected_components)
    grouped["bits_per_spike"] = bits_per_spike_from_sums(
        grouped["ll_sum"].to_numpy(dtype=float),
        grouped["null_ll_sum"].to_numpy(dtype=float),
        grouped["spike_sum"].to_numpy(dtype=float),
    )

    complete = grouped[grouped["complete"]].copy()
    if complete.empty:
        return grouped, pd.DataFrame(columns=summary_columns)
    summary = (
        complete.groupby("unit", observed=False)
        .agg(
            shuffle_mean_ll_sum=("ll_sum", "mean"),
            shuffle_median_ll_sum=("ll_sum", "median"),
            shuffle_mean_bits_per_spike=("bits_per_spike", "mean"),
            shuffle_median_bits_per_spike=("bits_per_spike", "median"),
            shuffle_mean_null_ll_sum=("null_ll_sum", "mean"),
            shuffle_mean_spike_sum=("spike_sum", "mean"),
            shuffle_complete_count=("shuffle_index", "count"),
        )
        .reset_index()
    )
    return grouped, summary


def build_unit_summary(
    *,
    unit_ids: np.ndarray,
    score_table: pd.DataFrame,
    shuffle_table: pd.DataFrame,
    light_rates_hz: np.ndarray,
    dark_rates_hz: np.ndarray,
    expected_shuffle_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-unit transfer summaries and per-shuffle aggregate scores."""
    summary = pd.DataFrame({"unit": np.asarray(unit_ids)})
    summary["light_movement_firing_rate_hz"] = np.asarray(light_rates_hz, dtype=float)
    summary["dark_movement_firing_rate_hz"] = np.asarray(dark_rates_hz, dtype=float)

    for model in ("light", "dark"):
        model_summary = _aggregate_model_table(score_table, model)
        summary = summary.merge(model_summary, on="unit", how="left")

    shuffle_by_index, shuffle_summary = _aggregate_shuffle_table(
        shuffle_table,
        expected_components=expected_shuffle_components,
    )
    summary = summary.merge(shuffle_summary, on="unit", how="left")
    summary["intercept_bits_per_spike"] = 0.0
    summary["light_minus_shuffle_bits_per_spike"] = (
        summary["light_bits_per_spike"] - summary["shuffle_mean_bits_per_spike"]
    )
    summary["dark_minus_shuffle_bits_per_spike"] = (
        summary["dark_bits_per_spike"] - summary["shuffle_mean_bits_per_spike"]
    )
    summary["dark_minus_light_bits_per_spike"] = (
        summary["dark_bits_per_spike"] - summary["light_bits_per_spike"]
    )

    transfer_values: list[float] = []
    transfer_reasons: list[str] = []
    for row in summary.itertuples(index=False):
        value, reason = compute_transfer_index(
            light_bits=getattr(row, "light_bits_per_spike", np.nan),
            dark_bits=getattr(row, "dark_bits_per_spike", np.nan),
            shuffle_bits=getattr(row, "shuffle_mean_bits_per_spike", np.nan),
        )
        transfer_values.append(float(value))
        transfer_reasons.append(reason)
    summary["transfer_index"] = transfer_values
    summary["transfer_index_status"] = transfer_reasons
    return summary, shuffle_by_index


def build_session_summary(
    unit_summary: pd.DataFrame,
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    estimator: str,
    filter_diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return one session-level summary row for a region."""
    transfer_valid = unit_summary["transfer_index"].to_numpy(dtype=float)
    transfer_valid = transfer_valid[np.isfinite(transfer_valid)]
    row = {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "estimator": estimator,
        "n_units": int(len(unit_summary)),
        "n_transfer_index_valid": int(transfer_valid.size),
        "n_transfer_index_nan": int(len(unit_summary) - transfer_valid.size),
        "median_light_bits_per_spike": float(
            np.nanmedian(unit_summary["light_bits_per_spike"])
        ),
        "median_dark_bits_per_spike": float(
            np.nanmedian(unit_summary["dark_bits_per_spike"])
        ),
        "median_shuffle_bits_per_spike": float(
            np.nanmedian(unit_summary["shuffle_mean_bits_per_spike"])
        ),
        "median_dark_minus_shuffle_bits_per_spike": float(
            np.nanmedian(unit_summary["dark_minus_shuffle_bits_per_spike"])
        ),
        "median_light_minus_shuffle_bits_per_spike": float(
            np.nanmedian(unit_summary["light_minus_shuffle_bits_per_spike"])
        ),
        "median_transfer_index": (
            float(np.nanmedian(transfer_valid)) if transfer_valid.size else np.nan
        ),
        "mean_transfer_index": (
            float(np.nanmean(transfer_valid)) if transfer_valid.size else np.nan
        ),
    }
    status_counts = (
        unit_summary["transfer_index_status"].value_counts(dropna=False).to_dict()
    )
    row["transfer_index_status_counts_json"] = json.dumps(status_counts, sort_keys=True)
    if filter_diagnostics is not None:
        row.update(filter_diagnostics)
    return pd.DataFrame([row])


def plot_region_summary(
    unit_summary: pd.DataFrame,
    *,
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    save_path: Path,
) -> None:
    """Save the four-panel per-region dark/light transfer summary figure."""
    import matplotlib.pyplot as plt

    def _boxplot_ready(arrays: list[np.ndarray]) -> list[np.ndarray]:
        return [array if array.size else np.asarray([np.nan]) for array in arrays]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    score_columns = [
        ("light_bits_per_spike", "light"),
        ("dark_bits_per_spike", "dark"),
        ("shuffle_mean_bits_per_spike", "shuffle"),
    ]
    score_values = [
        unit_summary[column].dropna().to_numpy(dtype=float)
        for column, _label in score_columns
    ]
    axes[0].boxplot(
        _boxplot_ready(score_values),
        labels=[label for _column, label in score_columns],
    )
    axes[0].axhline(0.0, color="k", linewidth=1)
    axes[0].set_ylabel("Held-out information (bits/spike)")
    axes[0].set_title("Model scores")

    deltas = [
        unit_summary["dark_minus_shuffle_bits_per_spike"].dropna().to_numpy(dtype=float),
        unit_summary["light_minus_shuffle_bits_per_spike"].dropna().to_numpy(dtype=float),
    ]
    axes[1].boxplot(
        _boxplot_ready(deltas),
        labels=["dark - shuffle", "light - shuffle"],
    )
    axes[1].axhline(0.0, color="k", linewidth=1)
    axes[1].set_ylabel("Delta bits/spike")
    axes[1].set_title("Paired deltas")

    transfer = unit_summary["transfer_index"].dropna().to_numpy(dtype=float)
    if transfer.size:
        axes[2].hist(transfer, bins=30, color="0.35", edgecolor="white")
        axes[2].axvline(float(np.nanmedian(transfer)), color="tab:red", linewidth=1.5)
    axes[2].axvline(0.0, color="k", linewidth=1)
    axes[2].axvline(1.0, color="k", linewidth=1, linestyle="--")
    n_total = int(len(unit_summary))
    n_valid = int(transfer.size)
    axes[2].text(
        0.02,
        0.98,
        f"n plotted = {n_valid}/{n_total}\nn NaN = {n_total - n_valid}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
    )
    axes[2].set_xlabel("Transfer index")
    axes[2].set_ylabel("Units")
    axes[2].set_title("Dark transfer fraction")

    x = unit_summary["light_bits_per_spike"].to_numpy(dtype=float)
    y = unit_summary["dark_bits_per_spike"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    axes[3].scatter(x[valid], y[valid], s=16, alpha=0.65)
    if np.any(valid):
        lower = float(np.nanmin([x[valid].min(), y[valid].min()]))
        upper = float(np.nanmax([x[valid].max(), y[valid].max()]))
        axes[3].plot([lower, upper], [lower, upper], color="k", linewidth=1)
    axes[3].set_xlabel("Light-trained bits/spike")
    axes[3].set_ylabel("Dark-trained bits/spike")
    axes[3].set_title("Light vs dark transfer")

    fig.suptitle(
        f"{animal_name} {date} {region.upper()} | light {light_epoch} | dark {dark_epoch}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _output_stem(region: str, light_epoch: str, dark_epoch: str, estimator: str) -> str:
    """Return a filesystem stem for one region and light/dark epoch pair."""
    return f"{region}_{light_epoch}_light_{dark_epoch}_dark_transfer_{estimator}"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for dark/light transfer scoring."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit light, dark, and shuffled-light task-progression models and "
            "score them on held-out light trajectory folds."
        )
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=_CUDA_VISIBLE_DEVICES_CLI,
        help="Optional CUDA_VISIBLE_DEVICES value applied before importing JAX.",
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--regions",
        "--region",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Regions to fit. Default: {' '.join(REGIONS)}",
    )
    parser.add_argument(
        "--light-epoch",
        help="Light run epoch to split and score. Default: registry value.",
    )
    parser.add_argument(
        "--dark-epoch",
        help="Dark run epoch used for dark-trained models. Default: registry value.",
    )
    parser.add_argument(
        "--estimator",
        choices=ESTIMATOR_CHOICES,
        default="empirical",
        help="Encoding estimator to use. Default: empirical",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--v1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help=f"Minimum V1 movement firing rate in both epochs. Default: {DEFAULT_REGION_FR_THRESHOLDS['v1']}",
    )
    parser.add_argument(
        "--ca1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=f"Minimum CA1 movement firing rate in both epochs. Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Spike-count bin size in seconds. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_OUTER_FOLDS,
        help=f"Outer light lap-CV folds. Default: {DEFAULT_OUTER_FOLDS}",
    )
    parser.add_argument(
        "--inner-n-folds",
        type=int,
        default=DEFAULT_INNER_FOLDS,
        help=f"Inner light-train CV folds for hyperparameter selection. Default: {DEFAULT_INNER_FOLDS}",
    )
    parser.add_argument(
        "--ridges",
        nargs="+",
        type=float,
        default=list(dlg.DEFAULT_RIDGES),
        help="GLM ridge strengths to sweep. Default matches dark_light_glm.py.",
    )
    parser.add_argument(
        "--spatial-bin-sizes-cm",
        nargs="+",
        type=float,
        default=list(dlg.DEFAULT_SPATIAL_BIN_SIZES_CM),
        help="GLM spatial-bin candidates used to derive TP spline counts.",
    )
    parser.add_argument(
        "--spatial-bin-size-cm",
        type=float,
        default=DEFAULT_EMPIRICAL_SPATIAL_BIN_SIZE_CM,
        help=(
            "Empirical tuning spatial bin size in cm. "
            f"Default: {DEFAULT_EMPIRICAL_SPATIAL_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_EMPIRICAL_SIGMA_BINS,
        help=(
            "Empirical tuning Gaussian smoothing width in bins. "
            f"Default: {DEFAULT_EMPIRICAL_SIGMA_BINS}"
        ),
    )
    parser.add_argument(
        "--spline-order",
        type=int,
        default=4,
        help="Spline order for task-progression basis functions. Default: 4",
    )
    parser.add_argument(
        "--min-selection-units",
        type=int,
        default=DEFAULT_MIN_SELECTION_UNITS,
        help=f"Minimum finite units required for hyperparameter selection. Default: {DEFAULT_MIN_SELECTION_UNITS}",
    )
    parser.add_argument(
        "--min-tuning-stability-correlation",
        type=float,
        default=DEFAULT_MIN_TUNING_STABILITY_CORRELATION,
        help=(
            "Minimum odd/even tuning correlation required in at least one "
            "trajectory for both light and dark epochs. "
            f"Default: {DEFAULT_MIN_TUNING_STABILITY_CORRELATION}"
        ),
    )
    parser.add_argument(
        "--no-tuning-stability-filter",
        dest="use_tuning_stability_filter",
        action="store_false",
        help="Disable the saved odd/even tuning-stability unit filter.",
    )
    parser.set_defaults(use_tuning_stability_filter=True)
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help=f"Number of circular-shift shuffled-light fits. Default: {DEFAULT_N_SHUFFLES}",
    )
    parser.add_argument(
        "--shuffle-min-shift-s",
        type=float,
        default=DEFAULT_SHUFFLE_MIN_SHIFT_S,
        help=f"Minimum circular shift on the trajectory-train axis. Default: {DEFAULT_SHUFFLE_MIN_SHIFT_S}",
    )
    parser.add_argument(
        "--shuffle-max-shift-s",
        type=float,
        help="Optional maximum circular shift. Default: trajectory train duration minus min shift.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for folds and shuffles. Default: {DEFAULT_SEED}",
    )
    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        "--use-speed",
        dest="use_speed",
        action="store_true",
        help="Include speed as a nuisance covariate.",
    )
    speed_group.add_argument(
        "--no-speed",
        dest="use_speed",
        action="store_false",
        help="Exclude speed covariates.",
    )
    parser.set_defaults(use_speed=False)
    parser.add_argument(
        "--speed-feature-mode",
        choices=("linear", "bspline"),
        default="linear",
        help="Speed covariate parameterization when speed is enabled. Default: linear",
    )
    parser.add_argument(
        "--n-splines-speed",
        type=int,
        default=5,
        help="Number of spline basis functions for bspline speed. Default: 5",
    )
    parser.add_argument(
        "--spline-order-speed",
        type=int,
        default=4,
        help="Spline order for bspline speed. Default: 4",
    )
    parser.add_argument(
        "--speed-bounds",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Optional explicit bounds for bspline speed features.",
    )
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    """Validate CLI values before session loading and model fitting."""
    if args.estimator not in ESTIMATOR_CHOICES:
        raise ValueError(f"--estimator must be one of {ESTIMATOR_CHOICES!r}.")
    if args.bin_size_s <= 0.0:
        raise ValueError("--bin-size-s must be positive.")
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if args.inner_n_folds < 2:
        raise ValueError("--inner-n-folds must be at least 2.")
    if args.n_shuffles < 1:
        raise ValueError("--n-shuffles must be at least 1.")
    if args.spline_order <= 0:
        raise ValueError("--spline-order must be positive.")
    if args.n_splines_speed <= 0:
        raise ValueError("--n-splines-speed must be positive.")
    if args.spline_order_speed <= 0:
        raise ValueError("--spline-order-speed must be positive.")
    if args.min_selection_units < 1:
        raise ValueError("--min-selection-units must be at least 1.")
    if args.use_tuning_stability_filter and not (
        -1.0 <= args.min_tuning_stability_correlation <= 1.0
    ):
        raise ValueError(
            "--min-tuning-stability-correlation must be between -1 and 1."
        )
    if args.estimator == "empirical" and args.use_speed:
        raise ValueError("--use-speed is not supported for --estimator empirical.")
    if args.spatial_bin_size_cm <= 0.0:
        raise ValueError("--spatial-bin-size-cm must be positive.")
    if args.sigma_bins < 0.0:
        raise ValueError("--sigma-bins must be non-negative.")
    if any(float(ridge) < 0.0 for ridge in args.ridges):
        raise ValueError("Ridge strengths must be non-negative.")
    if any(float(value) <= 0.0 for value in args.spatial_bin_sizes_cm):
        raise ValueError("--spatial-bin-sizes-cm values must be positive.")


def main() -> None:
    """Run the dark/light transfer workflow for one session."""
    args = parse_arguments()
    _validate_arguments(args)

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    light_epoch = (
        get_dataset_light_epoch(args.animal_name)
        if args.light_epoch is None
        else str(args.light_epoch)
    )
    dark_epoch = (
        get_dataset_dark_epoch(args.animal_name)
        if args.dark_epoch is None
        else str(args.dark_epoch)
    )
    if light_epoch == dark_epoch:
        raise ValueError("--light-epoch and --dark-epoch must differ.")

    stability_table: pd.DataFrame | None = None
    stability_table_path: Path | None = None
    if args.use_tuning_stability_filter:
        stability_table_path = get_tuning_stability_table_path(analysis_path)
        stability_table = load_tuning_stability_table(
            analysis_path=analysis_path,
            animal_name=args.animal_name,
            date=args.date,
        )

    if args.estimator == "glm":
        ridge_values = [float(value) for value in dict.fromkeys(args.ridges)]
        spatial_bin_sizes_cm = [
            float(value) for value in dict.fromkeys(args.spatial_bin_sizes_cm)
        ]
        position_basis_configs = dlg.build_position_basis_configs(
            animal_name=args.animal_name,
            spatial_bin_sizes_cm=spatial_bin_sizes_cm,
            spline_order=args.spline_order,
        )
        empirical_bin_edges = np.asarray([], dtype=float)
        empirical_tp_bin_count = 0
        trajectory_length_cm = float(position_basis_configs[0]["trajectory_length_cm"])
    else:
        ridge_values = []
        spatial_bin_sizes_cm = [float(args.spatial_bin_size_cm)]
        position_basis_configs = []
        (
            empirical_bin_edges,
            empirical_tp_bin_count,
            trajectory_length_cm,
        ) = build_task_progression_bin_edges(
            animal_name=args.animal_name,
            spatial_bin_size_cm=args.spatial_bin_size_cm,
        )
    print(
        f"Loading session {args.animal_name} {args.date}: "
        f"light={light_epoch}, dark={dark_epoch}."
    )
    if args.estimator == "glm":
        print(
            "GLM hyperparameter candidates: spatial bins "
            + ", ".join(f"{value:g}cm" for value in spatial_bin_sizes_cm)
            + "; ridges "
            + ", ".join(f"{value:g}" for value in ridge_values)
            + f"; bin={args.bin_size_s:g}s."
        )
    else:
        print(
            "Empirical tuning parameters: "
            f"spatial_bin={args.spatial_bin_size_cm:g}cm "
            f"({empirical_tp_bin_count} TP bins), "
            f"sigma={args.sigma_bins:g} bins; bin={args.bin_size_s:g}s."
        )
    if args.use_tuning_stability_filter:
        print(
            "Using tuning-stability unit filter: "
            f"odd/even correlation >= {args.min_tuning_stability_correlation:g} "
            f"in at least one trajectory for both {light_epoch} and {dark_epoch}."
        )

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=tuple(args.regions),
        selected_run_epochs=[light_epoch, dark_epoch],
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    speed_by_run = session["speed_by_run"] if args.use_speed else None
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    region_thresholds = {
        "v1": float(args.v1_min_fr_hz),
        "ca1": float(args.ca1_min_fr_hz),
    }
    saved_outputs: list[Path] = []
    region_filter_diagnostics: list[dict[str, Any]] = []
    for region in args.regions:
        print(f"Preparing {region.upper()}.")
        spikes_region = session["spikes_by_region"][region]
        region_unit_ids = np.asarray(list(spikes_region.keys()))
        dark_rates = np.asarray(movement_firing_rates[region][dark_epoch], dtype=float)
        light_rates = np.asarray(movement_firing_rates[region][light_epoch], dtype=float)
        fr_mask = dlg.build_train_epoch_fr_mask(
            dark_rates,
            light_rates,
            min_dark_fr_hz=region_thresholds[region],
            min_light_fr_hz=region_thresholds[region],
        )["combined"]
        if args.use_tuning_stability_filter:
            if stability_table is None:
                raise RuntimeError("Tuning stability table was not loaded.")
            light_stable_units = select_stable_units_for_epoch(
                stability_table,
                region=region,
                epoch=light_epoch,
                min_correlation=args.min_tuning_stability_correlation,
            )
            dark_stable_units = select_stable_units_for_epoch(
                stability_table,
                region=region,
                epoch=dark_epoch,
                min_correlation=args.min_tuning_stability_correlation,
            )
            min_stability_correlation = float(
                args.min_tuning_stability_correlation
            )
        else:
            light_stable_units = None
            dark_stable_units = None
            min_stability_correlation = None

        unit_mask, filter_diagnostics = build_unit_filter_diagnostics(
            unit_ids=region_unit_ids,
            fr_mask=fr_mask,
            light_stable_units=light_stable_units,
            dark_stable_units=dark_stable_units,
            min_stability_correlation=min_stability_correlation,
        )
        filter_diagnostics.update(
            {
                "animal_name": args.animal_name,
                "date": args.date,
                "region": region,
                "light_epoch": light_epoch,
                "dark_epoch": dark_epoch,
            }
        )
        region_filter_diagnostics.append(dict(filter_diagnostics))
        if not np.any(unit_mask):
            raise ValueError(
                f"No {region.upper()} units passed the movement firing-rate and "
                "tuning-stability filters. Counts: "
                f"{filter_diagnostics!r}."
            )
        selected_spikes = _selected_spikes(spikes_region, unit_mask)
        selected_light_rates = light_rates[unit_mask]
        selected_dark_rates = dark_rates[unit_mask]
        selected_unit_ids = np.asarray(list(selected_spikes.keys()))
        print(
            f"  Selected {selected_unit_ids.size}/{region_unit_ids.size} units "
            f"after FR and tuning-stability filters "
            f"(FR pass: {filter_diagnostics['n_units_fr_pass']}, "
            f"light stable: {filter_diagnostics['n_units_light_stable']}, "
            f"dark stable: {filter_diagnostics['n_units_dark_stable']})."
        )

        outer_folds = build_lap_index_folds(
            session["trajectory_intervals"][light_epoch],
            n_folds=args.n_folds,
            seed=args.seed,
        )
        fold_score_rows: list[dict[str, Any]] = []
        shuffle_score_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        all_candidate_rows: list[dict[str, Any]] = []
        skipped_shuffle_rows: list[dict[str, Any]] = []

        for fold in range(args.n_folds):
            if args.estimator == "glm":
                print(
                    f"  Outer fold {fold + 1}/{args.n_folds}: "
                    "selecting GLM hyperparameters."
                )
                selected, candidates = select_outer_fold_hyperparameters(
                    spikes=selected_spikes,
                    light_epoch=light_epoch,
                    outer_fold_index=fold,
                    outer_folds=outer_folds,
                    trajectory_intervals=session["trajectory_intervals"],
                    movement_by_run=session["movement_by_run"],
                    tp_by_epoch=session["task_progression_by_trajectory"],
                    speed_by_epoch=speed_by_run,
                    position_basis_configs=position_basis_configs,
                    ridge_values=ridge_values,
                    inner_n_folds=args.inner_n_folds,
                    seed=args.seed,
                    bin_size_s=args.bin_size_s,
                    spline_order=args.spline_order,
                    use_speed=args.use_speed,
                    speed_feature_mode=args.speed_feature_mode,
                    n_splines_speed=args.n_splines_speed,
                    spline_order_speed=args.spline_order_speed,
                    speed_bounds=(
                        None if args.speed_bounds is None else tuple(args.speed_bounds)
                    ),
                    min_selection_units=args.min_selection_units,
                )
                selected.update(
                    {
                        "fold": int(fold),
                        "estimator": "glm",
                        "dark_model_scope": DARK_MODEL_SCOPE,
                    }
                )
                selection_rows.append(
                    {
                        "animal_name": args.animal_name,
                        "date": args.date,
                        "region": region,
                        "light_epoch": light_epoch,
                        "dark_epoch": dark_epoch,
                        **selected,
                    }
                )
                for candidate in candidates:
                    all_candidate_rows.append(
                        {
                            "animal_name": args.animal_name,
                            "date": args.date,
                            "region": region,
                            "light_epoch": light_epoch,
                            "dark_epoch": dark_epoch,
                            "estimator": "glm",
                            "dark_model_scope": DARK_MODEL_SCOPE,
                            **candidate,
                        }
                    )
                print(
                    f"    selected ridge={selected['ridge']:.3g}, "
                    f"spatial_bin={selected['spatial_bin_size_cm']:g}cm "
                    f"({selected['n_splines']} splines), "
                    f"median={selected['score_median']:.5g} bits/spike."
                )
            else:
                print(
                    f"  Outer fold {fold + 1}/{args.n_folds}: "
                    "using fixed empirical tuning parameters."
                )
                selected = {
                    "fold": int(fold),
                    "estimator": "empirical",
                    "dark_model_scope": DARK_MODEL_SCOPE,
                    "ridge": np.nan,
                    "spatial_bin_size_cm": float(args.spatial_bin_size_cm),
                    "trajectory_length_cm": float(trajectory_length_cm),
                    "n_splines": np.nan,
                    "tp_bin_count": int(empirical_tp_bin_count),
                    "spline_order": np.nan,
                    "sigma_bins": float(args.sigma_bins),
                    "score_median": np.nan,
                    "score_mean": np.nan,
                    "n_finite_units": int(selected_unit_ids.size),
                }

            dark_fits_by_trajectory: dict[str, dict[str, Any]] = {}
            for trajectory in TRAJECTORY_TYPES:
                dark_intervals = {
                    trajectory: session["trajectory_intervals"][dark_epoch][
                        trajectory
                    ].intersect(session["movement_by_run"][dark_epoch])
                }
                if args.estimator == "glm":
                    dark_inputs = prepare_glm_inputs(
                        spikes=selected_spikes,
                        epoch=dark_epoch,
                        trajectories=[trajectory],
                        intervals_by_trajectory=dark_intervals,
                        tp_by_epoch=session["task_progression_by_trajectory"],
                        speed_by_epoch=speed_by_run,
                        bin_size_s=args.bin_size_s,
                    )
                    dark_fits_by_trajectory[trajectory] = (
                        fit_poisson_tp_glm_with_zero_count_fallback(
                            dark_inputs,
                            n_splines=int(selected["n_splines"]),
                            spline_order=args.spline_order,
                            ridge=float(selected["ridge"]),
                            use_speed=args.use_speed,
                            speed_feature_mode=args.speed_feature_mode,
                            n_splines_speed=args.n_splines_speed,
                            spline_order_speed=args.spline_order_speed,
                            speed_bounds=(
                                None
                                if args.speed_bounds is None
                                else tuple(args.speed_bounds)
                            ),
                        )
                    )
                else:
                    dark_fits_by_trajectory[trajectory] = fit_empirical_tuning(
                        spikes=selected_spikes,
                        epoch=dark_epoch,
                        trajectories=[trajectory],
                        intervals_by_trajectory=dark_intervals,
                        tp_by_epoch=session["task_progression_by_trajectory"],
                        bin_edges=empirical_bin_edges,
                        bin_size_s=args.bin_size_s,
                        sigma_bins=args.sigma_bins,
                    )

            for trajectory in TRAJECTORY_TYPES:
                train_interval = _restrict_laps_to_movement(
                    session["trajectory_intervals"],
                    session["movement_by_run"],
                    epoch=light_epoch,
                    trajectory=trajectory,
                    lap_indices=outer_folds[trajectory][fold]["train_indices"],
                )
                test_interval = _restrict_laps_to_movement(
                    session["trajectory_intervals"],
                    session["movement_by_run"],
                    epoch=light_epoch,
                    trajectory=trajectory,
                    lap_indices=outer_folds[trajectory][fold]["test_indices"],
                )
                train_interval_by_trajectory = {trajectory: train_interval}
                test_interval_by_trajectory = {trajectory: test_interval}

                light_train_inputs = prepare_glm_inputs(
                    spikes=selected_spikes,
                    epoch=light_epoch,
                    trajectories=[trajectory],
                    intervals_by_trajectory=train_interval_by_trajectory,
                    tp_by_epoch=session["task_progression_by_trajectory"],
                    speed_by_epoch=speed_by_run,
                    bin_size_s=args.bin_size_s,
                )
                light_test_inputs = prepare_glm_inputs(
                    spikes=selected_spikes,
                    epoch=light_epoch,
                    trajectories=[trajectory],
                    intervals_by_trajectory=test_interval_by_trajectory,
                    tp_by_epoch=session["task_progression_by_trajectory"],
                    speed_by_epoch=speed_by_run,
                    bin_size_s=args.bin_size_s,
                )
                if args.estimator == "glm":
                    light_fit = fit_poisson_tp_glm(
                        light_train_inputs,
                        n_splines=int(selected["n_splines"]),
                        spline_order=args.spline_order,
                        ridge=float(selected["ridge"]),
                        use_speed=args.use_speed,
                        speed_feature_mode=args.speed_feature_mode,
                        n_splines_speed=args.n_splines_speed,
                        spline_order_speed=args.spline_order_speed,
                        speed_bounds=(
                            None
                            if args.speed_bounds is None
                            else tuple(args.speed_bounds)
                        ),
                    )
                    light_metrics = score_one_model(
                        fit=light_fit,
                        test_inputs=light_test_inputs,
                        light_train_inputs=light_train_inputs,
                    )
                    dark_metrics = score_one_model(
                        fit=dark_fits_by_trajectory[trajectory],
                        test_inputs=light_test_inputs,
                        light_train_inputs=light_train_inputs,
                    )
                else:
                    light_fit = fit_empirical_tuning(
                        spikes=selected_spikes,
                        epoch=light_epoch,
                        trajectories=[trajectory],
                        intervals_by_trajectory=train_interval_by_trajectory,
                        tp_by_epoch=session["task_progression_by_trajectory"],
                        bin_edges=empirical_bin_edges,
                        bin_size_s=args.bin_size_s,
                        sigma_bins=args.sigma_bins,
                    )
                    light_metrics = score_empirical_model(
                        fit=light_fit,
                        test_inputs=light_test_inputs,
                        light_train_inputs=light_train_inputs,
                        bin_size_s=args.bin_size_s,
                    )
                    dark_metrics = score_empirical_model(
                        fit=dark_fits_by_trajectory[trajectory],
                        test_inputs=light_test_inputs,
                        light_train_inputs=light_train_inputs,
                        bin_size_s=args.bin_size_s,
                    )
                fold_score_rows.extend(
                    make_score_rows(
                        animal_name=args.animal_name,
                        date=args.date,
                        region=region,
                        light_epoch=light_epoch,
                        dark_epoch=dark_epoch,
                        model="light",
                        fold=fold,
                        trajectory=trajectory,
                        unit_ids=light_test_inputs["unit_ids"],
                        metrics=light_metrics,
                        selected=selected,
                        bin_size_s=args.bin_size_s,
                        fallback_to_mean=(
                            light_fit.get("fallback_to_mean")
                            if args.estimator == "empirical"
                            else None
                        ),
                    )
                )
                fold_score_rows.extend(
                    make_score_rows(
                        animal_name=args.animal_name,
                        date=args.date,
                        region=region,
                        light_epoch=light_epoch,
                        dark_epoch=dark_epoch,
                        model="dark",
                        fold=fold,
                        trajectory=trajectory,
                        unit_ids=light_test_inputs["unit_ids"],
                        metrics=dark_metrics,
                        selected=selected,
                        bin_size_s=args.bin_size_s,
                        fallback_to_mean=(
                            dark_fits_by_trajectory[trajectory].get("fallback_to_mean")
                            if args.estimator in ("empirical", "glm")
                            else None
                        ),
                    )
                )

                for shuffle_index in range(args.n_shuffles):
                    rng = _seed_from_parts(
                        args.seed,
                        region,
                        fold,
                        trajectory,
                        shuffle_index,
                    )
                    try:
                        shifted_spikes = circular_shift_spikes_on_interval_axis(
                            selected_spikes,
                            train_interval,
                            rng=rng,
                            min_shift_s=args.shuffle_min_shift_s,
                            max_shift_s=args.shuffle_max_shift_s,
                        )
                    except ValueError as exc:
                        skipped_shuffle_rows.append(
                            {
                                "animal_name": args.animal_name,
                                "date": args.date,
                                "region": region,
                                "light_epoch": light_epoch,
                                "dark_epoch": dark_epoch,
                                "fold": int(fold),
                                "trajectory": trajectory,
                                "shuffle_index": int(shuffle_index),
                                "reason": str(exc),
                            }
                        )
                        continue
                    if args.estimator == "glm":
                        shuffle_train_inputs = prepare_glm_inputs(
                            spikes=shifted_spikes,
                            epoch=light_epoch,
                            trajectories=[trajectory],
                            intervals_by_trajectory=train_interval_by_trajectory,
                            tp_by_epoch=session["task_progression_by_trajectory"],
                            speed_by_epoch=speed_by_run,
                            bin_size_s=args.bin_size_s,
                        )
                        shuffle_fit = fit_poisson_tp_glm(
                            shuffle_train_inputs,
                            n_splines=int(selected["n_splines"]),
                            spline_order=args.spline_order,
                            ridge=float(selected["ridge"]),
                            use_speed=args.use_speed,
                            speed_feature_mode=args.speed_feature_mode,
                            n_splines_speed=args.n_splines_speed,
                            spline_order_speed=args.spline_order_speed,
                            speed_bounds=(
                                None
                                if args.speed_bounds is None
                                else tuple(args.speed_bounds)
                            ),
                        )
                        shuffle_metrics = score_one_model(
                            fit=shuffle_fit,
                            test_inputs=light_test_inputs,
                            light_train_inputs=light_train_inputs,
                        )
                    else:
                        shuffle_fit = fit_empirical_tuning(
                            spikes=shifted_spikes,
                            epoch=light_epoch,
                            trajectories=[trajectory],
                            intervals_by_trajectory=train_interval_by_trajectory,
                            tp_by_epoch=session["task_progression_by_trajectory"],
                            bin_edges=empirical_bin_edges,
                            bin_size_s=args.bin_size_s,
                            sigma_bins=args.sigma_bins,
                        )
                        shuffle_metrics = score_empirical_model(
                            fit=shuffle_fit,
                            test_inputs=light_test_inputs,
                            light_train_inputs=light_train_inputs,
                            bin_size_s=args.bin_size_s,
                        )
                    shuffle_score_rows.extend(
                        make_score_rows(
                            animal_name=args.animal_name,
                            date=args.date,
                            region=region,
                            light_epoch=light_epoch,
                            dark_epoch=dark_epoch,
                            model="shuffle",
                            fold=fold,
                            trajectory=trajectory,
                            unit_ids=light_test_inputs["unit_ids"],
                            metrics=shuffle_metrics,
                            selected=selected,
                            bin_size_s=args.bin_size_s,
                            shuffle_index=shuffle_index,
                            fallback_to_mean=(
                                shuffle_fit.get("fallback_to_mean")
                                if args.estimator == "empirical"
                                else None
                            ),
                        )
                    )

        fold_score_table = pd.DataFrame(fold_score_rows)
        shuffle_score_table = pd.DataFrame(shuffle_score_rows)
        expected_shuffle_components = int(args.n_folds * len(TRAJECTORY_TYPES))
        unit_summary, shuffle_by_index = build_unit_summary(
            unit_ids=selected_unit_ids,
            score_table=fold_score_table,
            shuffle_table=shuffle_score_table,
            light_rates_hz=selected_light_rates,
            dark_rates_hz=selected_dark_rates,
            expected_shuffle_components=expected_shuffle_components,
        )
        session_summary = build_session_summary(
            unit_summary,
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            estimator=args.estimator,
            filter_diagnostics=filter_diagnostics,
        )
        skipped_shuffle_table = pd.DataFrame(skipped_shuffle_rows)

        stem = _output_stem(region, light_epoch, dark_epoch, args.estimator)
        output_paths = {
            "fold_scores": data_dir / f"{stem}_fold_scores.parquet",
            "shuffle_scores": data_dir / f"{stem}_shuffle_scores.parquet",
            "shuffle_summary": data_dir / f"{stem}_shuffle_by_index.parquet",
            "unit_summary": data_dir / f"{stem}_unit_summary.parquet",
            "session_summary": data_dir / f"{stem}_session_summary.parquet",
            "skipped_shuffles": data_dir / f"{stem}_skipped_shuffles.parquet",
        }
        if args.estimator == "glm":
            selection_table = pd.DataFrame(selection_rows)
            candidate_table = pd.DataFrame(all_candidate_rows)
            output_paths["selected_hyperparameters"] = (
                data_dir / f"{stem}_selected_hyperparameters.parquet"
            )
            output_paths["candidate_hyperparameters"] = (
                data_dir / f"{stem}_candidate_hyperparameters.parquet"
            )
        fold_score_table.to_parquet(output_paths["fold_scores"], index=False)
        shuffle_score_table.to_parquet(output_paths["shuffle_scores"], index=False)
        shuffle_by_index.to_parquet(output_paths["shuffle_summary"], index=False)
        unit_summary.to_parquet(output_paths["unit_summary"], index=False)
        session_summary.to_parquet(output_paths["session_summary"], index=False)
        if args.estimator == "glm":
            selection_table.to_parquet(
                output_paths["selected_hyperparameters"],
                index=False,
            )
            candidate_table.to_parquet(
                output_paths["candidate_hyperparameters"],
                index=False,
            )
        skipped_shuffle_table.to_parquet(output_paths["skipped_shuffles"], index=False)
        saved_outputs.extend(output_paths.values())

        figure_path = fig_dir / f"{stem}.png"
        plot_region_summary(
            unit_summary,
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            save_path=figure_path,
        )
        saved_outputs.append(figure_path)
        print(f"  Saved {region.upper()} outputs with stem {stem}.")

    write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.dark_light_transfer",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "regions": list(args.regions),
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
            "dark_model_scope": DARK_MODEL_SCOPE,
            "estimator": args.estimator,
            "bin_size_s": float(args.bin_size_s),
            "n_folds": int(args.n_folds),
            "inner_n_folds": int(args.inner_n_folds),
            "ridges": ridge_values,
            "spatial_bin_sizes_cm": spatial_bin_sizes_cm,
            "spatial_bin_size_cm": float(args.spatial_bin_size_cm),
            "empirical_tp_bin_count": int(empirical_tp_bin_count),
            "sigma_bins": float(args.sigma_bins),
            "n_shuffles": int(args.n_shuffles),
            "shuffle_min_shift_s": float(args.shuffle_min_shift_s),
            "shuffle_max_shift_s": args.shuffle_max_shift_s,
            "use_speed": bool(args.use_speed),
            "speed_feature_mode": args.speed_feature_mode,
            "region_thresholds_hz": region_thresholds,
            "tuning_stability_filter_enabled": bool(
                args.use_tuning_stability_filter
            ),
            "min_tuning_stability_correlation": (
                float(args.min_tuning_stability_correlation)
                if args.use_tuning_stability_filter
                else None
            ),
            "tuning_stability_table_path": (
                None if stability_table_path is None else str(stability_table_path)
            ),
            "seed": int(args.seed),
        },
        outputs={
            "saved_outputs": [str(path) for path in saved_outputs],
            "region_filter_diagnostics": region_filter_diagnostics,
        },
    )
    print("Saved outputs:")
    for path in saved_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
