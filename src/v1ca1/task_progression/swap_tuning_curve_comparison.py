from __future__ import annotations

"""Compare swapped-light activity with empirical task-progression tuning curves.

This script fits trajectory-specific task-progression tuning curves with
`pynapple.compute_tuning_curves` in a dark training epoch, a light training
epoch, and a light test epoch. For each trajectory, only the configured swapped
segment in the light test epoch is scored against empirical visual and
task-coordinate transfer predictions.
"""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.wtrack import get_wtrack_geometry, get_wtrack_total_length
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    build_task_progression_bins,
    compute_movement_firing_rates,
    get_analysis_path,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    import xarray as xr


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_BIN_SIZE_S = 0.02
DEFAULT_SIGMA_BINS = 1.0
POS_BOUNDS = (0.0, 1.0)
EMPIRICAL_MODEL_NAMES = (
    "empirical_visual",
    "empirical_dark",
    "empirical_pointwise_multiplicative_ratio",
    "empirical_segment_multiplicative_ratio",
    "empirical_pointwise_additive_delta",
    "empirical_segment_additive_delta",
)
SWAP_CONFIG = {
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
}


def derive_default_segment_edges(animal_name: str) -> np.ndarray:
    """Return the three-segment task-progression edges from W-track geometry."""
    geometry = get_wtrack_geometry(animal_name)
    diagonal_segment_length = float(
        np.sqrt(geometry["dx"] ** 2 + geometry["dy"] ** 2)
    )
    total_length = get_wtrack_total_length(animal_name)
    segment_border1 = (
        geometry["long_segment_length"] + diagonal_segment_length / 2.0
    ) / total_length
    segment_border2 = (
        geometry["long_segment_length"]
        + diagonal_segment_length
        + geometry["short_segment_length"]
        + diagonal_segment_length / 2.0
    ) / total_length
    return np.asarray([0.0, segment_border1, segment_border2, 1.0], dtype=float)


def validate_segment_edges(
    segment_edges: list[float] | np.ndarray,
    *,
    pos_bounds: tuple[float, float] = POS_BOUNDS,
) -> np.ndarray:
    """Validate and sanitize task-progression segment edges."""
    edges = np.asarray(segment_edges, dtype=float).reshape(-1)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            f"segment_edges must be a 1D sequence with len>=2. Got shape={edges.shape}."
        )
    if not np.all(np.isfinite(edges)):
        raise ValueError("segment_edges must contain only finite values.")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("segment_edges must be strictly increasing.")

    lower, upper = map(float, pos_bounds)
    tol = 1e-9
    if edges[0] < lower - tol or edges[-1] > upper + tol:
        raise ValueError(
            f"segment_edges must lie within pos_bounds={pos_bounds}. "
            f"Got [{edges[0]}, {edges[-1]}]."
        )
    edges[0] = max(edges[0], lower)
    edges[-1] = min(edges[-1], upper)
    return edges


def segment_mask(
    values: np.ndarray,
    segment_edges: np.ndarray,
    segment_index: int,
) -> np.ndarray:
    """Return a mask for one TP segment, including the last segment right edge."""
    values = np.asarray(values, dtype=float).reshape(-1)
    edges = np.asarray(segment_edges, dtype=float).reshape(-1)
    start = float(edges[int(segment_index)])
    end = float(edges[int(segment_index) + 1])
    if int(segment_index) == edges.size - 2:
        return (values >= start) & (values <= end)
    return (values >= start) & (values < end)


def validate_model_comparison_epochs(
    run_epochs: list[str],
    *,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
) -> tuple[str, str, str]:
    """Validate the requested dark/light train/test epoch triplet."""
    selected_epochs = [dark_train_epoch, light_train_epoch, light_test_epoch]
    missing_epochs = [epoch for epoch in selected_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in available run epochs {run_epochs!r}: "
            f"{missing_epochs!r}"
        )
    if len(set(selected_epochs)) != 3:
        raise ValueError(
            "dark_train_epoch, light_train_epoch, and light_test_epoch must all be "
            "distinct."
        )
    return dark_train_epoch, light_train_epoch, light_test_epoch


def build_train_epoch_fr_mask(
    dark_epoch_rates: np.ndarray,
    light_epoch_rates: np.ndarray,
    *,
    min_dark_fr_hz: float,
    min_light_fr_hz: float,
) -> dict[str, np.ndarray]:
    """Return dark, light, and combined train-epoch firing-rate masks."""
    dark_rates = np.asarray(dark_epoch_rates, dtype=float)
    light_rates = np.asarray(light_epoch_rates, dtype=float)
    if dark_rates.shape != light_rates.shape:
        raise ValueError(
            "dark_epoch_rates and light_epoch_rates must have matching shapes. "
            f"Got {dark_rates.shape} and {light_rates.shape}."
        )

    dark_mask = np.isfinite(dark_rates) & (dark_rates > float(min_dark_fr_hz))
    light_mask = np.isfinite(light_rates) & (light_rates > float(min_light_fr_hz))
    return {
        "dark": dark_mask,
        "light": light_mask,
        "combined": dark_mask & light_mask,
    }


def interpolate_nans_1d(values: np.ndarray, *, fallback_value: float = 0.0) -> np.ndarray:
    """Linearly interpolate NaNs and nearest-fill edge NaNs in one curve."""
    curve = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(curve)
    if np.all(finite):
        return curve.copy()
    if not np.any(finite):
        fallback = 0.0 if not np.isfinite(fallback_value) else float(fallback_value)
        return np.full(curve.shape, fallback, dtype=float)

    x = np.arange(curve.size, dtype=float)
    return np.interp(x, x[finite], curve[finite])


def smooth_interpolated_tuning_matrix(
    tuning_matrix: np.ndarray,
    *,
    fallback_rates_hz: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    """Interpolate NaNs per unit, then smooth a `(tp_bin, unit)` rate matrix."""
    matrix = np.asarray(tuning_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"tuning_matrix must be 2D. Got shape={matrix.shape}.")
    fallback_rates = np.asarray(fallback_rates_hz, dtype=float).reshape(-1)
    if fallback_rates.size != matrix.shape[1]:
        raise ValueError(
            "fallback_rates_hz must have one value per unit. "
            f"Got {fallback_rates.size} rates for {matrix.shape[1]} units."
        )

    interpolated = np.empty_like(matrix, dtype=float)
    for unit_index in range(matrix.shape[1]):
        interpolated[:, unit_index] = interpolate_nans_1d(
            matrix[:, unit_index],
            fallback_value=float(fallback_rates[unit_index]),
        )

    if float(sigma_bins) <= 0.0:
        return interpolated
    return gaussian_filter1d(
        interpolated,
        sigma=float(sigma_bins),
        axis=0,
        mode="nearest",
    )


def _extract_tuning_matrix(
    tuning_curve: Any,
    unit_ids: np.ndarray,
    *,
    pos_dim: str = "tp",
) -> np.ndarray:
    """Return a tuning curve as `(tp_bin, unit)` in the requested unit order."""
    if "unit" not in tuning_curve.dims:
        raise ValueError(f"tuning_curve must have a 'unit' dimension: {tuning_curve.dims}")
    if pos_dim not in tuning_curve.dims:
        other_dims = [dim for dim in tuning_curve.dims if dim != "unit"]
        if len(other_dims) != 1:
            raise ValueError(
                f"Could not infer position dimension from dims={tuning_curve.dims}."
            )
        pos_dim = str(other_dims[0])

    selected = tuning_curve.sel(unit=np.asarray(unit_ids))
    return np.asarray(selected.transpose(pos_dim, "unit").values, dtype=float)


def prepare_tuning_matrix(
    tuning_curve: Any,
    unit_ids: np.ndarray,
    *,
    fallback_rates_hz: np.ndarray,
    sigma_bins: float,
    pos_dim: str = "tp",
) -> np.ndarray:
    """Extract, interpolate, and smooth one pynapple tuning-curve output."""
    return smooth_interpolated_tuning_matrix(
        _extract_tuning_matrix(tuning_curve, unit_ids, pos_dim=pos_dim),
        fallback_rates_hz=fallback_rates_hz,
        sigma_bins=sigma_bins,
    )


def _sanitize_tuning_matrix(values: np.ndarray) -> np.ndarray:
    """Return a finite nonnegative tuning matrix."""
    matrix = np.asarray(values, dtype=float)
    return np.clip(np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def build_empirical_swap_tunings(
    same_dark_tuning: np.ndarray,
    other_light_tuning: np.ndarray,
    other_dark_tuning: np.ndarray,
    bin_centers: np.ndarray,
    segment_edges: np.ndarray,
    segment_index: int,
    *,
    epsilon: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Return empirical visual and task-coordinate swap tuning predictions."""
    same_dark = _sanitize_tuning_matrix(same_dark_tuning)
    other_light = _sanitize_tuning_matrix(other_light_tuning)
    other_dark = _sanitize_tuning_matrix(other_dark_tuning)
    if same_dark.shape != other_light.shape or same_dark.shape != other_dark.shape:
        raise ValueError(
            "same_dark_tuning, other_light_tuning, and other_dark_tuning must "
            f"have matching shapes. Got {same_dark.shape}, {other_light.shape}, "
            f"and {other_dark.shape}."
        )

    segment_bins = segment_mask(bin_centers, segment_edges, segment_index)
    if not np.any(segment_bins):
        raise ValueError(
            f"No TP bins fell inside segment_index={segment_index} for empirical "
            "segment-scalar tuning construction."
        )

    eps = float(epsilon)
    pointwise_ratio = other_light / np.clip(other_dark, eps, None)
    segment_ratio = np.sum(other_light[segment_bins], axis=0) / np.clip(
        np.sum(other_dark[segment_bins], axis=0),
        eps,
        None,
    )
    pointwise_delta = other_light - other_dark
    segment_delta = np.mean(
        other_light[segment_bins] - other_dark[segment_bins],
        axis=0,
    )

    tunings = {
        "empirical_visual": other_light,
        "empirical_dark": same_dark,
        "empirical_pointwise_multiplicative_ratio": same_dark * pointwise_ratio,
        "empirical_segment_multiplicative_ratio": same_dark * segment_ratio[None, :],
        "empirical_pointwise_additive_delta": same_dark + pointwise_delta,
        "empirical_segment_additive_delta": same_dark + segment_delta[None, :],
    }
    return {
        model_name: np.clip(np.asarray(tunings[model_name], dtype=float), eps, None)
        for model_name in EMPIRICAL_MODEL_NAMES
    }


def _poisson_log_likelihood(
    spike_counts: np.ndarray,
    rates_hz: np.ndarray,
    *,
    bin_size_s: float,
    epsilon: float,
) -> np.ndarray:
    """Return summed Poisson log likelihood per unit."""
    counts = np.asarray(spike_counts, dtype=float)
    rates = np.maximum(np.asarray(rates_hz, dtype=float), float(epsilon))
    lam = np.maximum(rates * float(bin_size_s), float(epsilon))
    return np.sum(
        counts * np.log(lam) - lam - gammaln(counts + 1.0),
        axis=0,
    )


def score_segment_binned_counts(
    spike_counts: np.ndarray,
    positions: np.ndarray,
    tunings_by_model: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    segment_edges: np.ndarray,
    segment_index: int,
    *,
    bin_size_s: float,
    epsilon: float = 1e-10,
) -> dict[str, np.ndarray | float]:
    """Score empirical tuning-curve models on binned counts within one TP segment."""
    counts = np.asarray(spike_counts, dtype=float)
    if counts.ndim == 1:
        counts = counts[:, None]
    positions = np.asarray(positions, dtype=float).reshape(-1)
    if counts.shape[0] != positions.size:
        raise ValueError(
            "spike_counts and positions must have matching sample counts. "
            f"Got {counts.shape[0]} and {positions.size}."
        )

    model_curves = []
    for model_name in EMPIRICAL_MODEL_NAMES:
        if model_name not in tunings_by_model:
            raise ValueError(f"Missing empirical tuning curve for model={model_name!r}.")
        model_curves.append(np.asarray(tunings_by_model[model_name], dtype=float))
    reference_shape = model_curves[0].shape
    for model_name, curve in zip(EMPIRICAL_MODEL_NAMES, model_curves, strict=True):
        if curve.shape != reference_shape:
            raise ValueError(
                "All empirical tuning curves must have matching shapes. "
                f"Model {model_name!r} has shape {curve.shape}; expected "
                f"{reference_shape}."
            )
    if len(reference_shape) != 2 or reference_shape[1] != counts.shape[1]:
        raise ValueError(
            "Tuning curves must be shaped `(tp_bin, unit)` and match counts. "
            f"Got curve shape={reference_shape}, count shape={counts.shape}."
        )

    valid = np.isfinite(positions) & segment_mask(positions, segment_edges, segment_index)
    counts = counts[valid]
    positions = positions[valid]
    n_models = len(model_curves)
    n_units = reference_shape[1]
    if counts.size == 0:
        empty = np.full((n_models, n_units), np.nan, dtype=float)
        zeros = np.zeros((n_models, n_units), dtype=float)
        return {
            "test_light_spike_sum": np.zeros((n_units,), dtype=float),
            "test_light_bin_count": 0.0,
            "test_light_duration_s": 0.0,
            "ll_sum": zeros,
            "ll_bits_per_spike": empty.copy(),
            "ll_bits_per_s": empty.copy(),
        }

    bin_edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    bin_index = np.digitize(positions, bin_edges) - 1
    bin_index = np.clip(bin_index, 0, reference_shape[0] - 1)
    ll_sum = np.stack(
        [
            _poisson_log_likelihood(
                counts,
                curve[bin_index, :],
                bin_size_s=bin_size_s,
                epsilon=epsilon,
            )
            for curve in model_curves
        ],
        axis=0,
    )
    spike_sum = np.sum(counts, axis=0)
    duration_s = float(counts.shape[0] * float(bin_size_s))
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_bits_per_spike = np.where(
            spike_sum[None, :] > 0,
            ll_sum / (np.log(2.0) * spike_sum[None, :]),
            np.nan,
        )
        ll_bits_per_s = ll_sum / (np.log(2.0) * duration_s)

    return {
        "test_light_spike_sum": np.asarray(spike_sum, dtype=float),
        "test_light_bin_count": float(counts.shape[0]),
        "test_light_duration_s": duration_s,
        "ll_sum": np.asarray(ll_sum, dtype=float),
        "ll_bits_per_spike": np.asarray(ll_bits_per_spike, dtype=float),
        "ll_bits_per_s": np.asarray(ll_bits_per_s, dtype=float),
    }


def score_tuning_curves_on_segment(
    *,
    spikes: Any,
    task_progression: Any,
    epoch: Any,
    tunings_by_model: dict[str, np.ndarray],
    bin_edges: np.ndarray,
    segment_edges: np.ndarray,
    segment_index: int,
    bin_size_s: float,
    unit_ids: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Bin spikes and score empirical tuning-curve models on one swapped segment."""
    counts = spikes.count(float(bin_size_s), ep=epoch)
    count_unit_ids = np.asarray(counts.columns)
    if count_unit_ids.shape != unit_ids.shape or not np.all(count_unit_ids == unit_ids):
        raise ValueError(
            "Binned spike columns do not match selected unit order. "
            f"Got {count_unit_ids.tolist()!r}; expected {unit_ids.tolist()!r}."
        )

    spike_counts = np.asarray(counts.d, dtype=float)
    if spike_counts.ndim == 1:
        spike_counts = spike_counts[:, None]
    positions = np.asarray(task_progression.interpolate(counts).to_numpy(), dtype=float)
    return score_segment_binned_counts(
        spike_counts,
        positions.reshape(-1),
        tunings_by_model,
        bin_edges,
        segment_edges,
        segment_index,
        bin_size_s=bin_size_s,
    )


def _interval_duration_s(interval: Any) -> float:
    """Return an IntervalSet-like object's duration in seconds."""
    return float(interval.tot_length())


def _trajectory_movement_interval(
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    trajectory: str,
) -> Any:
    """Return one trajectory interval restricted to movement."""
    return trajectory_intervals[trajectory].intersect(movement_interval)


def compute_unit_rates(
    spikes: Any,
    unit_ids: np.ndarray,
    epoch: Any,
) -> np.ndarray:
    """Return firing rates for selected units in one IntervalSet-like epoch."""
    duration_s = _interval_duration_s(epoch)
    rates = np.full((unit_ids.size,), np.nan, dtype=float)
    if duration_s <= 0.0:
        return rates

    for unit_index, unit_id in enumerate(unit_ids):
        rates[unit_index] = len(spikes[unit_id].restrict(epoch).t) / duration_s
    return rates


def compute_trajectory_rates(
    spikes: Any,
    unit_ids: np.ndarray,
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
) -> dict[str, np.ndarray]:
    """Return movement-restricted firing rates by trajectory."""
    rates: dict[str, np.ndarray] = {}
    for trajectory in TRAJECTORY_TYPES:
        epoch = _trajectory_movement_interval(
            trajectory_intervals,
            movement_interval,
            trajectory,
        )
        rates[trajectory] = compute_unit_rates(spikes, unit_ids, epoch)
    return rates


def compute_smoothed_tuning_by_trajectory(
    *,
    spikes: Any,
    unit_ids: np.ndarray,
    task_progression_by_trajectory: dict[str, Any],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    bin_edges: np.ndarray,
    fallback_rates_by_trajectory: dict[str, np.ndarray],
    sigma_bins: float,
) -> dict[str, np.ndarray]:
    """Compute smoothed empirical TP tuning curves for all trajectories."""
    import pynapple as nap

    n_bins = int(np.asarray(bin_edges).size - 1)
    tuning_by_trajectory: dict[str, np.ndarray] = {}
    for trajectory in TRAJECTORY_TYPES:
        epoch = _trajectory_movement_interval(
            trajectory_intervals,
            movement_interval,
            trajectory,
        )
        fallback_rates = np.nan_to_num(
            np.asarray(fallback_rates_by_trajectory[trajectory], dtype=float),
            nan=0.0,
        )
        if _interval_duration_s(epoch) <= 0.0:
            tuning_by_trajectory[trajectory] = np.repeat(
                fallback_rates[None, :],
                n_bins,
                axis=0,
            )
            continue

        tuning_curve = nap.compute_tuning_curves(
            data=spikes,
            features=task_progression_by_trajectory[trajectory],
            bins=[bin_edges],
            epochs=epoch,
            feature_names=["tp"],
        )
        tuning_by_trajectory[trajectory] = prepare_tuning_matrix(
            tuning_curve,
            unit_ids,
            fallback_rates_hz=fallback_rates,
            sigma_bins=sigma_bins,
            pos_dim="tp",
        )
    return tuning_by_trajectory


def run_region_comparison(
    *,
    spikes: Any,
    unit_ids: np.ndarray,
    trajectory_intervals: dict[str, dict[str, Any]],
    task_progression_by_epoch: dict[str, dict[str, Any]],
    movement_by_run: dict[str, Any],
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    bin_edges: np.ndarray,
    segment_edges: np.ndarray,
    bin_size_s: float,
    sigma_bins: float,
) -> dict[str, Any]:
    """Fit empirical tuning curves and compare swapped-segment explanations."""
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_trajectories = len(TRAJECTORY_TYPES)
    n_units = unit_ids.size

    dark_rates = compute_trajectory_rates(
        spikes,
        unit_ids,
        trajectory_intervals[dark_train_epoch],
        movement_by_run[dark_train_epoch],
    )
    light_rates = compute_trajectory_rates(
        spikes,
        unit_ids,
        trajectory_intervals[light_train_epoch],
        movement_by_run[light_train_epoch],
    )
    test_rates = compute_trajectory_rates(
        spikes,
        unit_ids,
        trajectory_intervals[light_test_epoch],
        movement_by_run[light_test_epoch],
    )

    dark_tuning = compute_smoothed_tuning_by_trajectory(
        spikes=spikes,
        unit_ids=unit_ids,
        task_progression_by_trajectory=task_progression_by_epoch[dark_train_epoch],
        trajectory_intervals=trajectory_intervals[dark_train_epoch],
        movement_interval=movement_by_run[dark_train_epoch],
        bin_edges=bin_edges,
        fallback_rates_by_trajectory=dark_rates,
        sigma_bins=sigma_bins,
    )
    light_tuning = compute_smoothed_tuning_by_trajectory(
        spikes=spikes,
        unit_ids=unit_ids,
        task_progression_by_trajectory=task_progression_by_epoch[light_train_epoch],
        trajectory_intervals=trajectory_intervals[light_train_epoch],
        movement_interval=movement_by_run[light_train_epoch],
        bin_edges=bin_edges,
        fallback_rates_by_trajectory=light_rates,
        sigma_bins=sigma_bins,
    )
    test_tuning = compute_smoothed_tuning_by_trajectory(
        spikes=spikes,
        unit_ids=unit_ids,
        task_progression_by_trajectory=task_progression_by_epoch[light_test_epoch],
        trajectory_intervals=trajectory_intervals[light_test_epoch],
        movement_interval=movement_by_run[light_test_epoch],
        bin_edges=bin_edges,
        fallback_rates_by_trajectory=test_rates,
        sigma_bins=sigma_bins,
    )

    model_names = np.asarray(EMPIRICAL_MODEL_NAMES, dtype=str)
    metric_names = ("ll_sum", "ll_bits_per_spike", "ll_bits_per_s")
    metrics = {
        metric_name: np.full(
            (model_names.size, n_trajectories, n_units),
            np.nan,
            dtype=float,
        )
        for metric_name in metric_names
    }
    model_tuning = np.full(
        (model_names.size, n_trajectories, bin_centers.size, n_units),
        np.nan,
        dtype=float,
    )
    test_light_spike_sum = np.full((n_trajectories, n_units), np.nan, dtype=float)
    test_light_bin_count = np.zeros((n_trajectories,), dtype=float)
    test_light_duration_s = np.zeros((n_trajectories,), dtype=float)
    segment_bin_masks = np.zeros((n_trajectories, bin_centers.size), dtype=bool)
    swap_source = np.empty((n_trajectories,), dtype=object)
    swap_segment_index = np.zeros((n_trajectories,), dtype=int)

    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        swap_info = SWAP_CONFIG[trajectory]
        source_trajectory = str(swap_info["source_trajectory"])
        segment_index = int(swap_info["segment_index"])
        swap_source[trajectory_index] = source_trajectory
        swap_segment_index[trajectory_index] = segment_index
        segment_bin_masks[trajectory_index] = segment_mask(
            bin_centers,
            segment_edges,
            segment_index,
        )

        same_dark_curve = dark_tuning[trajectory]
        other_dark_curve = dark_tuning[source_trajectory]
        other_light_curve = light_tuning[source_trajectory]
        tunings_by_model = build_empirical_swap_tunings(
            same_dark_curve,
            other_light_curve,
            other_dark_curve,
            bin_centers,
            segment_edges,
            segment_index,
        )
        for model_index, model_name in enumerate(EMPIRICAL_MODEL_NAMES):
            model_tuning[model_index, trajectory_index] = tunings_by_model[model_name]
        epoch = _trajectory_movement_interval(
            trajectory_intervals[light_test_epoch],
            movement_by_run[light_test_epoch],
            trajectory,
        )
        score = score_tuning_curves_on_segment(
            spikes=spikes,
            task_progression=task_progression_by_epoch[light_test_epoch][trajectory],
            epoch=epoch,
            tunings_by_model=tunings_by_model,
            bin_edges=bin_edges,
            segment_edges=segment_edges,
            segment_index=segment_index,
            bin_size_s=bin_size_s,
            unit_ids=unit_ids,
        )
        for metric_name in metric_names:
            metrics[metric_name][:, trajectory_index] = np.asarray(
                score[metric_name],
                dtype=float,
            )
        test_light_spike_sum[trajectory_index] = np.asarray(
            score["test_light_spike_sum"],
            dtype=float,
        )
        test_light_bin_count[trajectory_index] = float(score["test_light_bin_count"])
        test_light_duration_s[trajectory_index] = float(score["test_light_duration_s"])

    return {
        "unit_ids": np.asarray(unit_ids),
        "model_names": model_names,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "segment_edges": np.asarray(segment_edges, dtype=float),
        "swap_source_trajectory": swap_source,
        "swap_segment_index": swap_segment_index,
        "segment_bin_mask": segment_bin_masks,
        "model_tuning": model_tuning,
        "same_dark_tuning": np.stack(
            [dark_tuning[trajectory] for trajectory in TRAJECTORY_TYPES],
            axis=0,
        ),
        "other_dark_tuning": np.stack(
            [
                dark_tuning[str(SWAP_CONFIG[trajectory]["source_trajectory"])]
                for trajectory in TRAJECTORY_TYPES
            ],
            axis=0,
        ),
        "other_light_tuning": np.stack(
            [
                light_tuning[str(SWAP_CONFIG[trajectory]["source_trajectory"])]
                for trajectory in TRAJECTORY_TYPES
            ],
            axis=0,
        ),
        "test_light_tuning": np.stack(
            [test_tuning[trajectory] for trajectory in TRAJECTORY_TYPES],
            axis=0,
        ),
        "train_dark_same_rate_hz": np.stack(
            [dark_rates[trajectory] for trajectory in TRAJECTORY_TYPES],
            axis=0,
        ),
        "train_dark_other_rate_hz": np.stack(
            [
                dark_rates[str(SWAP_CONFIG[trajectory]["source_trajectory"])]
                for trajectory in TRAJECTORY_TYPES
            ],
            axis=0,
        ),
        "train_light_other_rate_hz": np.stack(
            [
                light_rates[str(SWAP_CONFIG[trajectory]["source_trajectory"])]
                for trajectory in TRAJECTORY_TYPES
            ],
            axis=0,
        ),
        "test_light_target_rate_hz": np.stack(
            [test_rates[trajectory] for trajectory in TRAJECTORY_TYPES],
            axis=0,
        ),
        "test_light_spike_sum": test_light_spike_sum,
        "test_light_bin_count": test_light_bin_count,
        "test_light_duration_s": test_light_duration_s,
        "metrics": metrics,
    }


def build_results_table(
    result: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    dark_movement_firing_rates: np.ndarray,
    light_movement_firing_rates: np.ndarray,
    apply_fr_filter: bool,
    min_dark_fr_hz: float,
    min_light_fr_hz: float,
) -> pd.DataFrame:
    """Return one long-form metric table with one row per model, trajectory, and unit."""
    rows: list[dict[str, Any]] = []
    unit_ids = np.asarray(result["unit_ids"])
    model_names = np.asarray(result["model_names"], dtype=str)
    segment_edges = np.asarray(result["segment_edges"], dtype=float)
    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        segment_index = int(result["swap_segment_index"][trajectory_index])
        for unit_index, unit_id in enumerate(unit_ids):
            base_row = {
                "animal_name": animal_name,
                "date": date,
                "region": region,
                "dark_train_epoch": dark_train_epoch,
                "light_train_epoch": light_train_epoch,
                "light_test_epoch": light_test_epoch,
                "trajectory": trajectory,
                "unit": int(unit_id),
                "apply_fr_filter": bool(apply_fr_filter),
                "min_dark_fr_hz": float(min_dark_fr_hz),
                "min_light_fr_hz": float(min_light_fr_hz),
                "dark_train_movement_firing_rate_hz": float(
                    dark_movement_firing_rates[unit_index]
                ),
                "light_train_movement_firing_rate_hz": float(
                    light_movement_firing_rates[unit_index]
                ),
                "swap_source_trajectory": str(
                    result["swap_source_trajectory"][trajectory_index]
                ),
                "swap_segment_index_1based": int(segment_index + 1),
                "swap_segment_start": float(segment_edges[segment_index]),
                "swap_segment_end": float(segment_edges[segment_index + 1]),
                "train_dark_same_rate_hz": float(
                    result["train_dark_same_rate_hz"][trajectory_index, unit_index]
                ),
                "train_dark_other_rate_hz": float(
                    result["train_dark_other_rate_hz"][trajectory_index, unit_index]
                ),
                "train_light_other_rate_hz": float(
                    result["train_light_other_rate_hz"][trajectory_index, unit_index]
                ),
                "test_light_target_rate_hz": float(
                    result["test_light_target_rate_hz"][trajectory_index, unit_index]
                ),
                "test_light_spike_sum": float(
                    result["test_light_spike_sum"][trajectory_index, unit_index]
                ),
                "test_light_bin_count": float(
                    result["test_light_bin_count"][trajectory_index]
                ),
                "test_light_duration_s": float(
                    result["test_light_duration_s"][trajectory_index]
                ),
            }
            for model_index, model_name in enumerate(model_names):
                row = dict(base_row)
                row["model"] = str(model_name)
                for metric_name, values in result["metrics"].items():
                    row[metric_name] = float(
                        values[model_index, trajectory_index, unit_index]
                    )
                rows.append(row)

    return pd.DataFrame(rows)


def build_region_dataset(
    result: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    dark_movement_firing_rates: np.ndarray,
    light_movement_firing_rates: np.ndarray,
    bin_size_s: float,
    sigma_bins: float,
    place_bin_size_cm: float,
    apply_fr_filter: bool,
    min_dark_fr_hz: float,
    min_light_fr_hz: float,
    sources: dict[str, Any],
) -> "xr.Dataset":
    """Build one NetCDF-ready region dataset."""
    import xarray as xr

    unit_ids = np.asarray(result["unit_ids"])
    model_names = np.asarray(result["model_names"], dtype=str)
    segment_edges = np.asarray(result["segment_edges"], dtype=float)
    swap_segment_index = np.asarray(result["swap_segment_index"], dtype=int)
    empirical_model_formulas = {
        "empirical_visual": "other_light",
        "empirical_dark": "same_dark",
        "empirical_pointwise_multiplicative_ratio": (
            "same_dark * other_light / max(other_dark, epsilon)"
        ),
        "empirical_segment_multiplicative_ratio": (
            "same_dark * sum(other_light_in_swap_segment) / "
            "max(sum(other_dark_in_swap_segment), epsilon)"
        ),
        "empirical_pointwise_additive_delta": "same_dark + other_light - other_dark",
        "empirical_segment_additive_delta": (
            "same_dark + mean(other_light_in_swap_segment - "
            "other_dark_in_swap_segment)"
        ),
    }
    dataset = xr.Dataset(
        data_vars={
            "dark_train_movement_firing_rate_hz": (
                "unit",
                np.asarray(dark_movement_firing_rates, dtype=float),
            ),
            "light_train_movement_firing_rate_hz": (
                "unit",
                np.asarray(light_movement_firing_rates, dtype=float),
            ),
            "same_dark_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray(result["same_dark_tuning"], dtype=float),
            ),
            "other_dark_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray(result["other_dark_tuning"], dtype=float),
            ),
            "other_light_train_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray(result["other_light_tuning"], dtype=float),
            ),
            "model_tuning_hz": (
                ("model", "trajectory", "tp_bin", "unit"),
                np.asarray(result["model_tuning"], dtype=float),
            ),
            "test_light_tuning_hz": (
                ("trajectory", "tp_bin", "unit"),
                np.asarray(result["test_light_tuning"], dtype=float),
            ),
            "train_dark_same_rate_hz": (
                ("trajectory", "unit"),
                np.asarray(result["train_dark_same_rate_hz"], dtype=float),
            ),
            "train_dark_other_rate_hz": (
                ("trajectory", "unit"),
                np.asarray(result["train_dark_other_rate_hz"], dtype=float),
            ),
            "train_light_other_rate_hz": (
                ("trajectory", "unit"),
                np.asarray(result["train_light_other_rate_hz"], dtype=float),
            ),
            "test_light_target_rate_hz": (
                ("trajectory", "unit"),
                np.asarray(result["test_light_target_rate_hz"], dtype=float),
            ),
            "test_light_spike_sum": (
                ("trajectory", "unit"),
                np.asarray(result["test_light_spike_sum"], dtype=float),
            ),
            "segment_bin_mask": (
                ("trajectory", "tp_bin"),
                np.asarray(result["segment_bin_mask"], dtype=bool),
            ),
            "swap_source_trajectory": (
                "trajectory",
                np.asarray(result["swap_source_trajectory"], dtype=str),
            ),
            "swap_segment_index_1based": (
                "trajectory",
                swap_segment_index + 1,
            ),
            "swap_segment_start": (
                "trajectory",
                np.asarray(
                    [segment_edges[index] for index in swap_segment_index],
                    dtype=float,
                ),
            ),
            "swap_segment_end": (
                "trajectory",
                np.asarray(
                    [segment_edges[index + 1] for index in swap_segment_index],
                    dtype=float,
                ),
            ),
            "test_light_bin_count": (
                "trajectory",
                np.asarray(result["test_light_bin_count"], dtype=float),
            ),
            "test_light_duration_s": (
                "trajectory",
                np.asarray(result["test_light_duration_s"], dtype=float),
            ),
            "segment_edges": ("segment_edge", segment_edges),
        },
        coords={
            "model": model_names,
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "unit": unit_ids,
            "tp_bin": np.asarray(result["bin_centers"], dtype=float),
            "segment_edge": np.arange(segment_edges.size, dtype=int),
        },
        attrs={
            "schema_version": "2",
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "dark_train_epoch": dark_train_epoch,
            "light_train_epoch": light_train_epoch,
            "light_test_epoch": light_test_epoch,
            "bin_size_s": float(bin_size_s),
            "sigma_bins": float(sigma_bins),
            "place_bin_size_cm": float(place_bin_size_cm),
            "apply_fr_filter": bool(apply_fr_filter),
            "min_dark_fr_hz": float(min_dark_fr_hz),
            "min_light_fr_hz": float(min_light_fr_hz),
            "scoring_scope": "light_test_swapped_segment_only",
            "training_tuning_scope": "full_trajectory_movement_interval",
            "empirical_model_formulas_json": json.dumps(
                empirical_model_formulas,
                sort_keys=True,
            ),
            "swap_rule_json": json.dumps(SWAP_CONFIG, sort_keys=True),
            "sources_json": json.dumps(sources, sort_keys=True),
        },
    )

    for metric_name, values in result["metrics"].items():
        dataset[metric_name] = (
            ("model", "trajectory", "unit"),
            np.asarray(values, dtype=float),
        )
    return dataset


def _output_stem(
    *,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
) -> str:
    """Return a stable output stem for one region and epoch triplet."""
    return (
        f"{region}_{dark_train_epoch}_traindark_"
        f"{light_train_epoch}_trainlight_"
        f"{light_test_epoch}_testlight_swap_tuning_curve_comparison"
    )


def _selected_spikes(spikes: Any, unit_mask: np.ndarray) -> Any:
    """Return the selected units from one TsGroup-like object."""
    return spikes[np.asarray(unit_mask, dtype=bool)]


def _unit_ids_from_spikes(spikes: Any) -> np.ndarray:
    """Return unit ids from a TsGroup-like object."""
    return np.asarray(list(spikes.keys()))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare swapped-light activity using empirical task-progression "
            "tuning curves."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal/session name.")
    parser.add_argument("--date", required=True, help="Session date, e.g. 20240611.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Analysis data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Brain regions to analyze. Default: {' '.join(REGIONS)}",
    )
    parser.add_argument(
        "--dark-train-epoch",
        required=True,
        help="Dark run epoch used for fitting same-arm dark tuning.",
    )
    parser.add_argument(
        "--light-train-epoch",
        required=True,
        help="Light run epoch used for fitting other-arm light tuning.",
    )
    parser.add_argument(
        "--light-test-epoch",
        required=True,
        help="Light run epoch whose swapped segment is evaluated.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples to ignore per epoch. "
            f"Default: {DEFAULT_POSITION_OFFSET}"
        ),
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
        "--place-bin-size-cm",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=(
            "Physical bin size used to derive normalized TP bins. "
            f"Default: {DEFAULT_PLACE_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=(
            "Spike-count bin size in seconds for Poisson likelihood scoring. "
            f"Default: {DEFAULT_BIN_SIZE_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in TP bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--v1-min-dark-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help=(
            "Minimum dark-train movement firing rate for V1 units. "
            f"Default: {DEFAULT_REGION_FR_THRESHOLDS['v1']}"
        ),
    )
    parser.add_argument(
        "--v1-min-light-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help=(
            "Minimum light-train movement firing rate for V1 units. "
            f"Default: {DEFAULT_REGION_FR_THRESHOLDS['v1']}"
        ),
    )
    parser.add_argument(
        "--ca1-min-dark-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=(
            "Minimum dark-train movement firing rate for CA1 units. "
            f"Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}"
        ),
    )
    parser.add_argument(
        "--ca1-min-light-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=(
            "Minimum light-train movement firing rate for CA1 units. "
            f"Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}"
        ),
    )
    parser.add_argument(
        "--no-fr-filter",
        action="store_false",
        dest="apply_fr_filter",
        help="Include all units instead of applying train-epoch firing-rate thresholds.",
    )
    parser.set_defaults(apply_fr_filter=True)
    parser.add_argument(
        "--segment-edges",
        nargs="+",
        type=float,
        help=(
            "Explicit normalized task-progression segment edges. Defaults to "
            "geometry-derived three-segment edges."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Accepted for compatibility; plots are not generated by this script.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the swapped-light empirical tuning-curve comparison."""
    args = parse_arguments()
    selected_epochs = [
        args.dark_train_epoch,
        args.light_train_epoch,
        args.light_test_epoch,
    ]
    print(
        "Loading session for "
        f"{args.animal_name} {args.date} with dark-train={args.dark_train_epoch}, "
        f"light-train={args.light_train_epoch}, light-test={args.light_test_epoch}."
    )
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        selected_run_epochs=selected_epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    validate_model_comparison_epochs(
        session["run_epochs"],
        dark_train_epoch=args.dark_train_epoch,
        light_train_epoch=args.light_train_epoch,
        light_test_epoch=args.light_test_epoch,
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)

    segment_edges = (
        derive_default_segment_edges(args.animal_name)
        if args.segment_edges is None
        else validate_segment_edges(args.segment_edges)
    )
    print(
        "Using segment edges: "
        + ", ".join(f"{edge:.4f}" for edge in np.asarray(segment_edges, dtype=float))
    )
    bin_edges = build_task_progression_bins(
        args.animal_name,
        place_bin_size_cm=args.place_bin_size_cm,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    dark_region_thresholds = {
        "v1": float(args.v1_min_dark_fr_hz),
        "ca1": float(args.ca1_min_dark_fr_hz),
    }
    light_region_thresholds = {
        "v1": float(args.v1_min_light_fr_hz),
        "ca1": float(args.ca1_min_light_fr_hz),
    }

    saved_tables: list[Path] = []
    saved_datasets: list[Path] = []
    skipped_regions: list[dict[str, Any]] = []
    for region in args.regions:
        print(f"Preparing region {region.upper()}.")
        dark_epoch_rates = np.asarray(
            movement_firing_rates[region][args.dark_train_epoch],
            dtype=float,
        )
        light_epoch_rates = np.asarray(
            movement_firing_rates[region][args.light_train_epoch],
            dtype=float,
        )
        train_fr_masks = build_train_epoch_fr_mask(
            dark_epoch_rates,
            light_epoch_rates,
            min_dark_fr_hz=dark_region_thresholds[region],
            min_light_fr_hz=light_region_thresholds[region],
        )
        if args.apply_fr_filter:
            unit_mask = train_fr_masks["combined"]
        else:
            unit_mask = np.ones(dark_epoch_rates.shape, dtype=bool)

        n_units_total = int(unit_mask.size)
        print(
            f"{region.upper()} FR mask: "
            f"dark>{dark_region_thresholds[region]:.3f} Hz passed "
            f"{int(train_fr_masks['dark'].sum())}/{n_units_total}; "
            f"light>{light_region_thresholds[region]:.3f} Hz passed "
            f"{int(train_fr_masks['light'].sum())}/{n_units_total}; "
            f"selected {int(unit_mask.sum())}/{n_units_total}."
        )
        if not np.any(unit_mask):
            skipped_regions.append(
                {
                    "region": region,
                    "reason": "no units selected after firing-rate filtering",
                    "dark_threshold_hz": dark_region_thresholds[region],
                    "light_threshold_hz": light_region_thresholds[region],
                }
            )
            print(f"Skipping {region.upper()}: no units selected.")
            continue

        selected_spikes = _selected_spikes(session["spikes_by_region"][region], unit_mask)
        unit_ids = _unit_ids_from_spikes(selected_spikes)
        selected_dark_rates = dark_epoch_rates[unit_mask]
        selected_light_rates = light_epoch_rates[unit_mask]
        print(
            f"Selected {unit_ids.size} {region.upper()} units for empirical "
            "tuning-curve comparison."
        )

        result = run_region_comparison(
            spikes=selected_spikes,
            unit_ids=unit_ids,
            trajectory_intervals=session["trajectory_intervals"],
            task_progression_by_epoch=session["task_progression_by_trajectory"],
            movement_by_run=session["movement_by_run"],
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
            light_test_epoch=args.light_test_epoch,
            bin_edges=bin_edges,
            segment_edges=segment_edges,
            bin_size_s=args.bin_size_s,
            sigma_bins=args.sigma_bins,
        )
        table = build_results_table(
            result,
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
            light_test_epoch=args.light_test_epoch,
            dark_movement_firing_rates=selected_dark_rates,
            light_movement_firing_rates=selected_light_rates,
            apply_fr_filter=args.apply_fr_filter,
            min_dark_fr_hz=dark_region_thresholds[region],
            min_light_fr_hz=light_region_thresholds[region],
        )
        dataset = build_region_dataset(
            result,
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
            light_test_epoch=args.light_test_epoch,
            dark_movement_firing_rates=selected_dark_rates,
            light_movement_firing_rates=selected_light_rates,
            bin_size_s=args.bin_size_s,
            sigma_bins=args.sigma_bins,
            place_bin_size_cm=args.place_bin_size_cm,
            apply_fr_filter=args.apply_fr_filter,
            min_dark_fr_hz=dark_region_thresholds[region],
            min_light_fr_hz=light_region_thresholds[region],
            sources=session["sources"],
        )

        stem = _output_stem(
            region=region,
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
            light_test_epoch=args.light_test_epoch,
        )
        table_path = data_dir / f"{stem}.parquet"
        dataset_path = data_dir / f"{stem}.nc"
        table.to_parquet(table_path, index=False)
        dataset.to_netcdf(dataset_path)
        dataset.close()
        saved_tables.append(table_path)
        saved_datasets.append(dataset_path)
        print(f"Saved table to {table_path}.")
        print(f"Saved dataset to {dataset_path}.")

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.swap_tuning_curve_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": str(args.data_root),
            "regions": list(args.regions),
            "dark_train_epoch": args.dark_train_epoch,
            "light_train_epoch": args.light_train_epoch,
            "light_test_epoch": args.light_test_epoch,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "place_bin_size_cm": args.place_bin_size_cm,
            "bin_size_s": args.bin_size_s,
            "sigma_bins": args.sigma_bins,
            "apply_fr_filter": args.apply_fr_filter,
            "region_dark_thresholds_hz": dark_region_thresholds,
            "region_light_thresholds_hz": light_region_thresholds,
            "segment_edges": segment_edges.tolist(),
            "swap_rule": SWAP_CONFIG,
            "saved_tables": [str(path) for path in saved_tables],
            "saved_datasets": [str(path) for path in saved_datasets],
            "skipped_regions": skipped_regions,
        },
    )
    print(f"Saved run log to {log_path}.")
    if saved_tables:
        print(f"Saved {len(saved_tables)} table(s) and {len(saved_datasets)} dataset(s).")
    if skipped_regions:
        print(f"Skipped {len(skipped_regions)} region(s): {skipped_regions!r}.")


if __name__ == "__main__":
    main()
