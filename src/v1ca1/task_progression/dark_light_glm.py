from __future__ import annotations

"""Fit dark/light task-progression GLMs for one session.

This module reuses the shared task-progression session loaders, fits
swap-compatible dark/light GLMs per trajectory, selects shared bin-size and
spatial-bin-derived place-basis hyperparameters from the visual model by
lap-level cross-validation, selects ridge per model, then saves candidate,
selected, and selection-summary NetCDF-backed `xarray.Dataset` outputs.

Supported selected-fit models:

- `visual`: separate dark and light task-progression fields
- `task_segment_bump`: shared dark field plus segment raised-cosine light gain
- `task_segment_scalar`: shared dark field plus segment scalar light gain
- `task_dense_gain`: shared dark field plus dense spline light gain
"""

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from v1ca1.helper.cuda import (
    configure_cuda_visible_devices,
    pop_cuda_visible_devices_argument,
)
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.wtrack import get_wtrack_geometry, get_wtrack_total_length
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    compute_movement_firing_rates,
    get_analysis_path,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    import xarray as xr


_CUDA_VISIBLE_DEVICES_CLI = pop_cuda_visible_devices_argument()
configure_cuda_visible_devices(_CUDA_VISIBLE_DEVICES_CLI)


try:
    import scipy
except ModuleNotFoundError:
    scipy = None

try:
    from nemos.basis import BSplineEval
    from nemos.glm import PopulationGLM
except ModuleNotFoundError:
    BSplineEval = None
    PopulationGLM = None


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_RIDGES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
DEFAULT_BIN_SIZES_S = (0.02, 0.05)
DEFAULT_SPATIAL_BIN_SIZES_CM = (2.0, 4.0, 8.0)
DEFAULT_MODEL_NAMES = (
    "visual",
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
)
DEPRECATED_MODEL_NAME_ALIASES = {
    "independent_light_field": "visual",
    "segment_bump_gain": "task_segment_bump",
    "segment_scalar_gain": "task_segment_scalar",
    "dense_gain": "task_dense_gain",
}
MODEL_NAME_CHOICES = (
    *DEFAULT_MODEL_NAMES,
    *DEPRECATED_MODEL_NAME_ALIASES,
)
DEFAULT_SEED = 47
SELECTION_METRIC = "ll_bits_per_spike_cv_combined"
SELECTION_AGGREGATION = "median_trajectory_unit"
SELECTION_MODEL_NAME = "visual"
HYPERPARAMETER_TIE_ATOL = 1e-6
CV_SCORE_SUFFIXES = ("combined", "dark", "light")


def _require_nemos() -> None:
    """Ensure the core GLM dependencies are available before fitting models."""
    if BSplineEval is None or PopulationGLM is None:
        raise ModuleNotFoundError(
            "This script requires `nemos` to fit dark/light GLM models, but it is not installed."
        )


def _require_scipy() -> None:
    """Ensure SciPy is available before using the additive model path."""
    if scipy is None:
        raise ModuleNotFoundError(
            "The additive dark/light model requires `scipy`, but it is not installed."
        )


def select_light_dark_pairs(
    run_epochs: list[str],
    *,
    dark_epoch: str,
    light_epochs: list[str] | None,
) -> list[tuple[str, str]]:
    """Return validated `(light_epoch, dark_epoch)` pairs to fit."""
    epoch_pool = list(run_epochs)
    if dark_epoch not in epoch_pool:
        raise ValueError(
            "Requested dark epoch was not found in the valid run epochs. "
            f"Valid run epochs: {epoch_pool!r}; dark_epoch: {dark_epoch!r}"
        )
    if light_epochs is not None:
        selected_light_epochs = list(dict.fromkeys(light_epochs))
        missing_light_epochs = [
            epoch for epoch in selected_light_epochs if epoch not in epoch_pool
        ]
        if missing_light_epochs:
            raise ValueError(
                "Requested light epochs were not found in the valid run epochs. "
                f"Valid run epochs: {epoch_pool!r}; missing: {missing_light_epochs!r}"
            )
    else:
        selected_light_epochs = [epoch for epoch in epoch_pool if epoch != dark_epoch]

    invalid_light_epochs = [
        epoch for epoch in selected_light_epochs if epoch == dark_epoch
    ]
    if invalid_light_epochs:
        raise ValueError(
            "Light epochs must differ from the requested dark epoch. "
            f"dark_epoch={dark_epoch!r}; invalid light epochs: {invalid_light_epochs!r}"
        )
    if not selected_light_epochs:
        raise ValueError("No light epochs remain after excluding the requested dark epoch.")

    return [(light_epoch, dark_epoch) for light_epoch in selected_light_epochs]


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


def normalize_model_names(model_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return active swap-compatible model names and human-readable messages."""
    normalized: list[str] = []
    messages: list[str] = []
    for model_name in model_names:
        name = str(model_name)
        if name in DEPRECATED_MODEL_NAME_ALIASES:
            replacement = DEPRECATED_MODEL_NAME_ALIASES[name]
            messages.append(
                f"Model {name!r} is deprecated; using {replacement!r} instead."
            )
            name = replacement
        if name not in DEFAULT_MODEL_NAMES:
            raise ValueError(
                f"Unknown model {name!r}. Expected one of {DEFAULT_MODEL_NAMES!r}."
            )
        if name not in normalized:
            normalized.append(name)

    if SELECTION_MODEL_NAME not in normalized:
        normalized.insert(0, SELECTION_MODEL_NAME)
        messages.append(
            f"Added required selection-anchor model {SELECTION_MODEL_NAME!r}."
        )
    return normalized, messages


def normalize_candidate_values(args: argparse.Namespace) -> dict[str, list[float]]:
    """Return normalized hyperparameter candidate lists from CLI arguments."""
    if args.bin_sizes_s is not None:
        bin_sizes = [float(value) for value in args.bin_sizes_s]
    elif args.bin_size_s is not None:
        bin_sizes = [float(args.bin_size_s)]
    else:
        bin_sizes = [float(value) for value in DEFAULT_BIN_SIZES_S]

    spatial_bin_sizes = [float(value) for value in args.spatial_bin_sizes_cm]

    if any(value <= 0 for value in bin_sizes):
        raise ValueError("All bin sizes must be positive.")
    if any(value <= 0 for value in spatial_bin_sizes):
        raise ValueError("All spatial bin-size candidates must be positive.")
    return {
        "bin_sizes_s": list(dict.fromkeys(bin_sizes)),
        "spatial_bin_sizes_cm": list(dict.fromkeys(spatial_bin_sizes)),
    }


def n_splines_from_spatial_bin_size(
    length_cm: float,
    spatial_bin_size_cm: float,
    *,
    spline_order: int,
) -> int:
    """Return a positive spline count from a spatial bin size."""
    if not np.isfinite(length_cm) or length_cm <= 0.0:
        raise ValueError(f"Track length must be positive and finite. Got {length_cm!r}.")
    if not np.isfinite(spatial_bin_size_cm) or spatial_bin_size_cm <= 0.0:
        raise ValueError(
            "Spatial bin size must be positive and finite. "
            f"Got {spatial_bin_size_cm!r}."
        )
    return max(
        int(spline_order),
        int(np.ceil(float(length_cm) / float(spatial_bin_size_cm))),
    )


def build_position_basis_configs(
    *,
    animal_name: str,
    spatial_bin_sizes_cm: Sequence[float],
    spline_order: int,
) -> list[dict[str, Any]]:
    """Build ordered spatial-bin-derived basis configurations."""
    trajectory_length_cm = float(get_wtrack_total_length(animal_name))
    spatial_bin_sizes = [float(value) for value in dict.fromkeys(spatial_bin_sizes_cm)]
    if not spatial_bin_sizes:
        raise ValueError("At least one spatial bin size is required.")
    return [
        {
            "spatial_bin_size_cm": float(spatial_bin_size_cm),
            "trajectory_length_cm": trajectory_length_cm,
            "n_splines": n_splines_from_spatial_bin_size(
                trajectory_length_cm,
                spatial_bin_size_cm,
                spline_order=spline_order,
            ),
            "spline_order": int(spline_order),
            "pos_bounds": (0.0, 1.0),
        }
        for spatial_bin_size_cm in spatial_bin_sizes
    ]


def derive_default_segment_edges(animal_name: str) -> np.ndarray:
    """Derive the three-segment default for segment-gain models."""
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



def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start/end arrays from an IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Interval start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts, ends


def _subset_intervalset(intervals: Any, indices: np.ndarray) -> Any:
    """Return one IntervalSet containing only selected interval rows."""
    intervalset_class = intervals.__class__
    starts, ends = _extract_interval_bounds(intervals)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    return intervalset_class(
        start=np.asarray(starts[indices], dtype=float),
        end=np.asarray(ends[indices], dtype=float),
        time_units="s",
    )


def build_lap_cv_folds_for_trajectory(
    *,
    trajectory_intervals: dict[str, dict[str, Any]],
    movement_by_run: dict[str, Any],
    dark_epoch: str,
    light_epoch: str,
    trajectory: str,
    n_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Return movement-restricted lap-level CV folds for one trajectory."""
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    rng = np.random.default_rng(int(seed))
    intervals_by_epoch = {
        dark_epoch: trajectory_intervals[dark_epoch][trajectory],
        light_epoch: trajectory_intervals[light_epoch][trajectory],
    }
    split_indices: dict[str, list[np.ndarray]] = {}
    for epoch, intervals in intervals_by_epoch.items():
        starts, _ends = _extract_interval_bounds(intervals)
        n_laps = int(starts.size)
        if n_laps < n_folds:
            raise ValueError(
                f"Epoch {epoch!r} trajectory {trajectory!r} has only {n_laps} lap(s), "
                f"fewer than n_folds={n_folds}."
            )
        split_indices[epoch] = [
            np.sort(chunk.astype(int, copy=False))
            for chunk in np.array_split(rng.permutation(n_laps), n_folds)
        ]

    folds: list[dict[str, Any]] = []
    for fold_index in range(n_folds):
        train_restrict: dict[str, Any] = {}
        validation_restrict: dict[str, Any] = {}
        fold_metadata: dict[str, Any] = {"fold_index": int(fold_index)}
        for epoch, intervals in intervals_by_epoch.items():
            starts, _ends = _extract_interval_bounds(intervals)
            all_indices = np.arange(starts.size, dtype=int)
            validation_indices = split_indices[epoch][fold_index]
            train_indices = np.setdiff1d(all_indices, validation_indices, assume_unique=True)
            train_interval = _subset_intervalset(intervals, train_indices).intersect(
                movement_by_run[epoch]
            )
            validation_interval = _subset_intervalset(intervals, validation_indices).intersect(
                movement_by_run[epoch]
            )
            train_restrict[epoch] = train_interval
            validation_restrict[epoch] = validation_interval
            fold_metadata[f"{epoch}_train_indices"] = train_indices.tolist()
            fold_metadata[f"{epoch}_validation_indices"] = validation_indices.tolist()

        folds.append(
            {
                "fold_index": int(fold_index),
                "train_restrict": train_restrict,
                "validation_restrict": validation_restrict,
                "metadata": fold_metadata,
            }
        )
    return folds


def _zscore_train_apply(
    values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Z-score using the training set and apply that transform to test samples."""
    mean = float(np.mean(values[train_idx]))
    std = float(np.std(values[train_idx]) + eps)
    return (values[train_idx] - mean) / std, (values[test_idx] - mean) / std, mean, std


def _as_interval_set(interval_like: Any) -> Any:
    """Return an IntervalSet whether the input is one directly or via `time_support`."""
    if hasattr(interval_like, "time_support"):
        return interval_like.time_support
    return interval_like


def _coef_feat_by_unit(model: PopulationGLM, n_features: int) -> np.ndarray:
    """Return coefficients as `(feature, unit)` regardless of backend layout."""
    coef = np.asarray(model.coef_)
    if coef.shape[0] != n_features:
        coef = coef.T
    return coef


def _empty_speed_design(n_rows: int) -> np.ndarray:
    """Return an empty speed design matrix with the requested row count."""
    return np.zeros((int(n_rows), 0), dtype=float)


def _normalize_speed_feature_mode(speed_feature_mode: str) -> str:
    """Validate the requested speed parameterization mode."""
    mode = str(speed_feature_mode).lower()
    if mode not in {"linear", "bspline"}:
        raise ValueError(
            "speed_feature_mode must be one of {'linear', 'bspline'}. "
            f"Got {speed_feature_mode!r}."
        )
    return mode


def _empty_speed_feature_transform() -> dict[str, Any]:
    """Return a sentinel speed-transform metadata object for no-speed fits."""
    return {
        "mode": "none",
        "basis": "none",
        "n_features": 0,
        "spline_order": np.nan,
        "bounds": np.asarray([np.nan, np.nan], dtype=float),
        "reference_value": np.nan,
        "mean": np.nan,
        "std": np.nan,
    }


def _sanitize_feature_bounds(
    lower: float,
    upper: float,
    *,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Return finite, strictly increasing bounds for spline features."""
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Feature bounds must be finite.")
    if upper <= lower:
        pad = max(abs(lower), abs(upper), 1.0) * eps
        lower -= pad
        upper += pad
    return float(lower), float(upper)


def _fit_speed_feature_transform(
    values: np.ndarray,
    *,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Fit the speed-feature transform from training or full-data samples."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Speed feature transform requires at least one sample.")

    mode = _normalize_speed_feature_mode(speed_feature_mode)
    reference_value = float(np.mean(values))

    if mode == "linear":
        return {
            "mode": "linear",
            "basis": "linear",
            "n_features": 1,
            "spline_order": np.nan,
            "bounds": np.asarray([np.nan, np.nan], dtype=float),
            "reference_value": reference_value,
            "mean": float(np.mean(values)),
            "std": float(np.std(values) + eps),
        }

    if speed_bounds is None:
        lower = float(np.min(values))
        upper = float(np.max(values))
    else:
        bounds_arr = np.asarray(speed_bounds, dtype=float).reshape(-1)
        if bounds_arr.shape != (2,):
            raise ValueError(
                f"speed_bounds must be a length-2 tuple. Got shape={bounds_arr.shape}."
            )
        lower, upper = float(bounds_arr[0]), float(bounds_arr[1])

    lower, upper = _sanitize_feature_bounds(lower, upper)
    return {
        "mode": "bspline",
        "basis": "bspline",
        "n_features": int(n_splines_speed),
        "spline_order": int(spline_order_speed),
        "bounds": np.asarray([lower, upper], dtype=float),
        "reference_value": reference_value,
        "mean": np.nan,
        "std": np.nan,
    }


def _transform_speed_with_feature_transform(
    values: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    """Apply a fitted speed transform to one vector of speed samples."""
    values = np.asarray(values, dtype=float).reshape(-1)
    mode = str(transform["mode"])

    if mode == "none":
        return _empty_speed_design(values.size)

    if mode == "linear":
        return ((values - float(transform["mean"])) / float(transform["std"]))[:, None]

    lower, upper = np.asarray(transform["bounds"], dtype=float)
    basis = BSplineEval(
        n_basis_funcs=int(transform["n_features"]),
        order=int(transform["spline_order"]),
        bounds=(float(lower), float(upper)),
    )
    return np.asarray(
        basis.compute_features(np.clip(values, float(lower), float(upper))),
        dtype=float,
    )


def _make_speed_design_train_test(
    values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a speed transform on train indices and apply it to train/test."""
    transform = _fit_speed_feature_transform(
        values[train_idx],
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
        eps=eps,
    )
    return (
        _transform_speed_with_feature_transform(values[train_idx], transform),
        _transform_speed_with_feature_transform(values[test_idx], transform),
        transform,
    )


def _make_speed_design_all(
    values: np.ndarray,
    *,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and apply a speed transform using all available samples."""
    transform = _fit_speed_feature_transform(
        values,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
        eps=eps,
    )
    return _transform_speed_with_feature_transform(values, transform), transform


def _speed_reference_design(transform: dict[str, Any]) -> np.ndarray:
    """Return the design row evaluated at the transform reference speed."""
    if transform["mode"] == "none":
        return _empty_speed_design(1)
    return _transform_speed_with_feature_transform(
        np.asarray([transform["reference_value"]], dtype=float),
        transform,
    )


def _speed_reference_effect(
    transform: dict[str, Any],
    coef_speed_basis: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Return the speed contribution used for TP grid predictions."""
    coef_speed_basis = np.asarray(coef_speed_basis, dtype=float).reshape(-1, n_units)
    if coef_speed_basis.shape[0] == 0:
        return np.zeros((1, n_units), dtype=float)
    return _speed_reference_design(transform) @ coef_speed_basis


def _format_speed_outputs(
    *,
    transform: dict[str, Any],
    coef_speed_basis_base: np.ndarray | None,
    coef_speed_basis_full: np.ndarray | None,
    n_units: int,
) -> dict[str, Any]:
    """Normalize speed metadata and coefficient outputs across modes."""
    nan_u = np.full((n_units,), np.nan)
    empty_speed = np.full((0, n_units), np.nan)

    if coef_speed_basis_base is None:
        coef_speed_basis_base = empty_speed
    else:
        coef_speed_basis_base = np.asarray(coef_speed_basis_base, dtype=float).reshape(
            -1,
            n_units,
        )

    if coef_speed_basis_full is None:
        coef_speed_basis_full = empty_speed
    else:
        coef_speed_basis_full = np.asarray(coef_speed_basis_full, dtype=float).reshape(
            -1,
            n_units,
        )

    mode = str(transform["mode"])
    if mode == "linear":
        coef_speed_base = (
            coef_speed_basis_base[0, :]
            if coef_speed_basis_base.shape[0] > 0
            else nan_u
        )
        coef_speed_full = (
            coef_speed_basis_full[0, :]
            if coef_speed_basis_full.shape[0] > 0
            else nan_u
        )
        speed_mean = float(transform["mean"])
        speed_std = float(transform["std"])
    else:
        coef_speed_base = nan_u
        coef_speed_full = nan_u
        speed_mean = np.nan
        speed_std = np.nan

    return {
        "speed_feature_mode": mode,
        "n_speed_features": int(transform["n_features"]),
        "speed_basis": str(transform["basis"]),
        "speed_spline_order": float(transform["spline_order"]),
        "speed_basis_bounds": np.asarray(transform["bounds"], dtype=float),
        "speed_reference_value": (
            float(transform["reference_value"])
            if np.isfinite(transform["reference_value"])
            else np.nan
        ),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "coef_speed_base_all": coef_speed_base,
        "coef_speed_full_all": coef_speed_full,
        "coef_speed_basis_base_all": coef_speed_basis_base,
        "coef_speed_basis_full_all": coef_speed_basis_full,
    }


def _validate_segment_edges(
    segment_edges: Sequence[float],
    *,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Validate and sanitize TP segment edges."""
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


def _segment_center_raised_cosine_basis(
    x: np.ndarray,
    segment_edges: Sequence[float],
    *,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    overlap_frac: float = 0.0,
) -> np.ndarray:
    """Return one raised-cosine bump per TP segment."""
    edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    x = np.asarray(x, dtype=float).reshape(-1)
    n_segments = edges.size - 1
    if x.size == 0:
        return np.zeros((0, n_segments), dtype=float)

    overlap_frac = float(overlap_frac)
    if overlap_frac < 0:
        raise ValueError("overlap_frac must be >= 0.")

    basis = np.zeros((x.size, n_segments), dtype=float)
    for segment_index in range(n_segments):
        start = float(edges[segment_index])
        end = float(edges[segment_index + 1])
        width = end - start
        if width <= 0:
            continue

        center = 0.5 * (start + end)
        left_ext = overlap_frac * width if segment_index > 0 else 0.0
        right_ext = overlap_frac * width if segment_index < (n_segments - 1) else 0.0
        left = max(pos_bounds[0], start - left_ext)
        right = min(pos_bounds[1], end + right_ext)
        half_left = center - left
        half_right = right - center
        if half_left <= 0 or half_right <= 0:
            continue

        normalized = np.empty_like(x, dtype=float)
        left_mask = x <= center
        normalized[left_mask] = (x[left_mask] - center) / half_left
        normalized[~left_mask] = (x[~left_mask] - center) / half_right
        inside_mask = np.abs(normalized) <= 1.0
        basis[inside_mask, segment_index] = 0.5 * (
            1.0 + np.cos(np.pi * normalized[inside_mask])
        )
    return basis


def _segment_onehot_basis(
    x: np.ndarray,
    segment_edges: Sequence[float],
    *,
    pos_bounds: tuple[float, float],
    drop_first: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-hot segment membership, optionally dropping the reference segment."""
    edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    x = np.asarray(x, dtype=float).reshape(-1)
    segment_index = np.searchsorted(edges, x, side="right") - 1
    segment_index = np.clip(segment_index, 0, edges.size - 2)

    n_segments = edges.size - 1
    basis = np.zeros((x.size, n_segments), dtype=float)
    basis[np.arange(x.size), segment_index] = 1.0
    if drop_first:
        return basis[:, 1:], edges
    return basis, edges



def _poisson_ll_sum(y_true: np.ndarray, lam_pred: np.ndarray) -> np.ndarray:
    """Return Poisson log-likelihood sums per unit."""
    _require_scipy()
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.clip(np.asarray(lam_pred, dtype=float), 1e-12, None)
    return np.sum(
        y_true * np.log(lam_pred) - lam_pred - scipy.special.gammaln(y_true + 1.0),
        axis=0,
    )


def _build_full_model_nospeed_design(
    model_name: str,
    p: np.ndarray,
    light: np.ndarray,
    *,
    n_splines: int,
    spline_order: int,
    segment_edges: np.ndarray,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the no-speed full-model design matrix and design metadata."""
    _require_nemos()
    p = np.asarray(p, dtype=float).reshape(-1)
    light = np.asarray(light, dtype=float).reshape(-1)
    if p.shape != light.shape:
        raise ValueError(f"p and light must have matching shapes. Got {p.shape} and {light.shape}.")

    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    dark_basis = np.asarray(basis.compute_features(p), dtype=float)
    n_place_basis = int(dark_basis.shape[1])
    metadata: dict[str, Any] = {
        "basis": basis,
        "n_place_basis": n_place_basis,
        "pos_bounds": tuple(float(value) for value in pos_bounds),
    }

    if model_name == "visual":
        basis_dark = dark_basis * (1.0 - light[:, None])
        basis_light = dark_basis * light[:, None]
        design = np.concatenate([basis_dark, basis_light, light[:, None]], axis=1)
        metadata["light_basis_count"] = n_place_basis
        return design, metadata

    if model_name == "task_segment_bump":
        gain_basis = _segment_center_raised_cosine_basis(
            p,
            segment_edges,
            pos_bounds=pos_bounds,
            overlap_frac=0.0,
        )
        design = np.concatenate([dark_basis, light[:, None], gain_basis * light[:, None]], axis=1)
        metadata["gain_basis_count"] = int(gain_basis.shape[1])
        metadata["gain_basis_name"] = "segment_raised_cosine"
        metadata["segment_edges"] = np.asarray(segment_edges, dtype=float)
        return design, metadata

    if model_name == "task_segment_scalar":
        gain_basis, gain_edges = _segment_onehot_basis(
            p,
            segment_edges,
            pos_bounds=pos_bounds,
            drop_first=False,
        )
        design = np.concatenate([dark_basis, light[:, None], gain_basis * light[:, None]], axis=1)
        metadata["gain_basis_count"] = int(gain_basis.shape[1])
        metadata["gain_basis_name"] = "segment_scalar"
        metadata["segment_edges"] = np.asarray(gain_edges, dtype=float)
        return design, metadata

    if model_name == "task_dense_gain":
        gain_basis = np.asarray(basis.compute_features(p), dtype=float)
        design = np.concatenate([dark_basis, light[:, None], gain_basis * light[:, None]], axis=1)
        metadata["gain_basis_count"] = int(gain_basis.shape[1])
        metadata["gain_basis_name"] = "bspline"
        return design, metadata

    raise ValueError(f"Unknown model_name: {model_name!r}")


def _build_full_model_grid_components(
    model_name: str,
    *,
    grid: np.ndarray,
    n_splines: int,
    spline_order: int,
    segment_edges: np.ndarray,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, np.ndarray]:
    """Return grid basis matrices for selected full-model predictions."""
    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    place_basis = np.asarray(basis.compute_features(grid), dtype=float)
    output: dict[str, np.ndarray] = {"place_basis": place_basis}
    if model_name == "task_segment_bump":
        output["gain_basis"] = _segment_center_raised_cosine_basis(
            grid,
            segment_edges,
            pos_bounds=pos_bounds,
            overlap_frac=0.0,
        )
    elif model_name == "task_segment_scalar":
        output["gain_basis"], _ = _segment_onehot_basis(
            grid,
            segment_edges,
            pos_bounds=pos_bounds,
            drop_first=False,
        )
    elif model_name == "task_dense_gain":
        output["gain_basis"] = np.asarray(basis.compute_features(grid), dtype=float)
    return output


def _score_full_model_fold(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    light_test: np.ndarray,
    y_train: np.ndarray,
    light_train: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return combined, dark-only, and light-only CV likelihood metrics."""
    coef = _coef_feat_by_unit(model, n_features=x_test.shape[1])
    eta = np.asarray(model.intercept_).reshape(1, -1) + (x_test @ coef)
    lam = np.exp(eta)
    result: dict[str, np.ndarray] = {}
    for suffix, mask in (
        ("combined", np.ones(light_test.shape, dtype=bool)),
        ("dark", light_test < 0.5),
        ("light", light_test > 0.5),
    ):
        n_units = y_test.shape[1]
        if not np.any(mask):
            result[f"ll_sum_cv_{suffix}"] = np.zeros(n_units, dtype=float)
            result[f"null_ll_sum_cv_{suffix}"] = np.zeros(n_units, dtype=float)
            result[f"spike_sum_cv_{suffix}"] = np.zeros(n_units, dtype=float)
            continue

        if suffix == "combined":
            train_mask = np.ones(light_train.shape, dtype=bool)
        elif suffix == "dark":
            train_mask = light_train < 0.5
        else:
            train_mask = light_train > 0.5
        if not np.any(train_mask):
            train_mask = np.ones(light_train.shape, dtype=bool)

        y_subset = y_test[mask]
        lam_subset = lam[mask]
        null_rate = np.clip(np.mean(y_train[train_mask], axis=0), 1e-12, None)
        null_lam = np.repeat(null_rate[None, :], y_subset.shape[0], axis=0)
        result[f"ll_sum_cv_{suffix}"] = _poisson_ll_sum(y_subset, lam_subset)
        result[f"null_ll_sum_cv_{suffix}"] = _poisson_ll_sum(y_subset, null_lam)
        result[f"spike_sum_cv_{suffix}"] = np.asarray(np.sum(y_subset, axis=0), dtype=float)
    return result


def _bits_per_spike_from_ll(
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


def _prepare_dark_light_fit_inputs(
    *,
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    bin_size_s: float,
    unit_mask: np.ndarray | None,
    restrict_ep_by_epoch: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble binned response, TP, light labels, and optional speed for one trajectory."""
    all_epochs = list(light_epochs) + list(dark_epochs)
    if not all_epochs:
        raise ValueError("Provide at least one epoch in light_epochs or dark_epochs.")

    has_speed = speed_by_epoch is not None
    selected_spikes = spikes if unit_mask is None else spikes[unit_mask]

    responses: list[np.ndarray] = []
    task_progression: list[np.ndarray] = []
    light_labels: list[np.ndarray] = []
    speed_values: list[np.ndarray] = []
    unit_ids: np.ndarray | None = None

    for epoch in all_epochs:
        trajectory_interval = trajectory_ep_by_epoch[epoch][traj_name]
        if restrict_ep_by_epoch is not None:
            trajectory_interval = trajectory_interval.intersect(
                _as_interval_set(restrict_ep_by_epoch[epoch])
            )

        counts = selected_spikes.count(bin_size_s, ep=trajectory_interval)
        current_unit_ids = np.asarray(counts.columns)
        if unit_ids is None:
            unit_ids = current_unit_ids
        elif current_unit_ids.shape != unit_ids.shape or not np.all(current_unit_ids == unit_ids):
            raise ValueError("Spike count columns (unit order) differ across epochs.")

        response_epoch = np.asarray(counts.d, dtype=float)
        tp_epoch = (
            tp_by_epoch[epoch][traj_name].interpolate(counts).to_numpy().reshape(-1)
        )
        light_epoch = (
            np.ones_like(tp_epoch, dtype=float)
            if epoch in light_epochs
            else np.zeros_like(tp_epoch, dtype=float)
        )
        if has_speed:
            speed_epoch = speed_by_epoch[epoch].interpolate(counts).to_numpy().reshape(-1)
            good_mask = np.isfinite(tp_epoch) & np.isfinite(speed_epoch)
            speed_values.append(np.asarray(speed_epoch[good_mask], dtype=float))
        else:
            good_mask = np.isfinite(tp_epoch)

        responses.append(np.asarray(response_epoch[good_mask], dtype=float))
        task_progression.append(np.asarray(tp_epoch[good_mask], dtype=float))
        light_labels.append(np.asarray(light_epoch[good_mask], dtype=float))

    if unit_ids is None:
        raise ValueError("No data remained after parsing epochs and trajectory intervals.")

    y_all = np.concatenate(responses, axis=0)
    p_all = np.concatenate(task_progression, axis=0).reshape(-1)
    l_all = np.concatenate(light_labels, axis=0).reshape(-1)
    if (l_all == 1).sum() == 0 or (l_all == 0).sum() == 0:
        raise ValueError("Need both light and dark bins across the provided epochs.")

    if has_speed:
        v_all = np.concatenate(speed_values, axis=0).reshape(-1)
    else:
        v_all = None

    return {
        "unit_ids": np.asarray(unit_ids),
        "y_all": y_all,
        "p_all": p_all,
        "l_all": l_all,
        "v_all": v_all,
        "has_speed": bool(has_speed),
    }



def _first_result(results_by_traj: dict[str, dict[float, dict[str, Any]]]) -> dict[str, Any]:
    """Return the first fit result from a `{trajectory: {ridge: result}}` mapping."""
    for trajectory in TRAJECTORY_TYPES:
        by_ridge = results_by_traj[trajectory]
        if by_ridge:
            return next(iter(by_ridge.values()))
    raise ValueError("No fit results were available to build a dataset.")



def _fit_selected_full_model_per_traj(
    *,
    model_name: str,
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    light_epoch: str,
    dark_epoch: str,
    traj_name: str,
    folds: list[dict[str, Any]],
    movement_by_run: dict[str, Any],
    bin_size_s: float,
    n_splines: int,
    spline_order: int,
    spatial_bin_size_cm: float,
    trajectory_length_cm: float,
    ridge: float,
    unit_mask: np.ndarray,
    segment_edges: np.ndarray,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit one swap-compatible model with lap-CV and a full-data refit."""
    _require_nemos()
    n_units: int | None = None
    cv_accumulator: dict[str, np.ndarray] = {}

    for fold in folds:
        train_inputs = _prepare_dark_light_fit_inputs(
            spikes=spikes,
            trajectory_ep_by_epoch=trajectory_ep_by_epoch,
            tp_by_epoch=tp_by_epoch,
            speed_by_epoch=speed_by_epoch,
            traj_name=traj_name,
            light_epochs=[light_epoch],
            dark_epochs=[dark_epoch],
            bin_size_s=bin_size_s,
            unit_mask=unit_mask,
            restrict_ep_by_epoch=fold["train_restrict"],
        )
        validation_inputs = _prepare_dark_light_fit_inputs(
            spikes=spikes,
            trajectory_ep_by_epoch=trajectory_ep_by_epoch,
            tp_by_epoch=tp_by_epoch,
            speed_by_epoch=speed_by_epoch,
            traj_name=traj_name,
            light_epochs=[light_epoch],
            dark_epochs=[dark_epoch],
            bin_size_s=bin_size_s,
            unit_mask=unit_mask,
            restrict_ep_by_epoch=fold["validation_restrict"],
        )
        y_train = np.asarray(train_inputs["y_all"], dtype=float)
        y_validation = np.asarray(validation_inputs["y_all"], dtype=float)
        if y_train.shape[0] == 0 or y_validation.shape[0] == 0:
            raise ValueError(
                f"Fold {fold['fold_index']} has no binned train or validation samples."
            )
        if n_units is None:
            n_units = int(y_train.shape[1])
            for suffix in CV_SCORE_SUFFIXES:
                cv_accumulator[f"ll_sum_cv_{suffix}"] = np.zeros(n_units, dtype=float)
                cv_accumulator[f"null_ll_sum_cv_{suffix}"] = np.zeros(n_units, dtype=float)
                cv_accumulator[f"spike_sum_cv_{suffix}"] = np.zeros(n_units, dtype=float)

        x_train_nospeed, _ = _build_full_model_nospeed_design(
            model_name,
            train_inputs["p_all"],
            train_inputs["l_all"],
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
            pos_bounds=pos_bounds,
        )
        x_validation_nospeed, _ = _build_full_model_nospeed_design(
            model_name,
            validation_inputs["p_all"],
            validation_inputs["l_all"],
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
            pos_bounds=pos_bounds,
        )
        if speed_by_epoch is None:
            speed_transform = _empty_speed_feature_transform()
            v_train = _empty_speed_design(y_train.shape[0])
            v_validation = _empty_speed_design(y_validation.shape[0])
        else:
            speed_transform = _fit_speed_feature_transform(
                train_inputs["v_all"],
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
            )
            v_train = _transform_speed_with_feature_transform(
                train_inputs["v_all"],
                speed_transform,
            )
            v_validation = _transform_speed_with_feature_transform(
                validation_inputs["v_all"],
                speed_transform,
            )
        x_train = np.concatenate([x_train_nospeed, v_train], axis=1)
        x_validation = np.concatenate([x_validation_nospeed, v_validation], axis=1)

        model = PopulationGLM(
            "Poisson",
            regularizer="Ridge",
            regularizer_strength=float(ridge),
        )
        model.fit(x_train, y_train)
        fold_scores = _score_full_model_fold(
            model,
            x_validation,
            y_validation,
            validation_inputs["l_all"],
            y_train,
            train_inputs["l_all"],
        )
        for key, value in fold_scores.items():
            cv_accumulator[key] += np.asarray(value, dtype=float)

    if n_units is None:
        raise ValueError("No CV folds were available to fit.")

    for suffix in CV_SCORE_SUFFIXES:
        cv_accumulator[f"ll_bits_per_spike_cv_{suffix}"] = _bits_per_spike_from_ll(
            cv_accumulator[f"ll_sum_cv_{suffix}"],
            cv_accumulator[f"null_ll_sum_cv_{suffix}"],
            cv_accumulator[f"spike_sum_cv_{suffix}"],
        )

    full_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=[light_epoch],
        dark_epochs=[dark_epoch],
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch={
            dark_epoch: movement_by_run[dark_epoch],
            light_epoch: movement_by_run[light_epoch],
        },
    )
    x_full_nospeed, design_metadata = _build_full_model_nospeed_design(
        model_name,
        full_inputs["p_all"],
        full_inputs["l_all"],
        n_splines=n_splines,
        spline_order=spline_order,
        segment_edges=segment_edges,
        pos_bounds=pos_bounds,
    )
    if speed_by_epoch is None:
        speed_transform_all = _empty_speed_feature_transform()
        v_full = _empty_speed_design(full_inputs["y_all"].shape[0])
    else:
        speed_transform_all = _fit_speed_feature_transform(
            full_inputs["v_all"],
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
        v_full = _transform_speed_with_feature_transform(
            full_inputs["v_all"],
            speed_transform_all,
        )
    x_full = np.concatenate([x_full_nospeed, v_full], axis=1)
    model_full = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model_full.fit(x_full, full_inputs["y_all"])

    coef = _coef_feat_by_unit(model_full, n_features=x_full.shape[1])
    intercept = np.asarray(model_full.intercept_).reshape(-1)
    unit_ids = np.asarray(full_inputs["unit_ids"])
    n_units = int(unit_ids.size)
    n_place_basis = int(design_metadata["n_place_basis"])
    n_speed_features = int(speed_transform_all["n_features"])
    model_specific_coef: dict[str, np.ndarray] = {}

    if model_name == "visual":
        coef_place_dark = coef[:n_place_basis]
        coef_place_light = coef[n_place_basis : (2 * n_place_basis)]
        coef_light_offset = coef[2 * n_place_basis]
        speed_start = (2 * n_place_basis) + 1
        model_specific_coef["coef_place_light"] = coef_place_light
    else:
        coef_place_dark = coef[:n_place_basis]
        coef_light_offset = coef[n_place_basis]
        gain_basis_count = int(design_metadata["gain_basis_count"])
        gain_start = n_place_basis + 1
        gain_end = gain_start + gain_basis_count
        coef_gain = coef[gain_start:gain_end]
        speed_start = gain_end
        if model_name == "task_segment_bump":
            model_specific_coef["coef_segment_bump_gain"] = coef_gain
        elif model_name == "task_segment_scalar":
            model_specific_coef["coef_segment_scalar_gain"] = coef_gain
        elif model_name == "task_dense_gain":
            model_specific_coef["coef_gain_spline"] = coef_gain

    coef_speed_basis = coef[speed_start : (speed_start + n_speed_features)]
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=None,
        coef_speed_basis_full=coef_speed_basis,
        n_units=n_units,
    )

    grid_tp = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    grid_components = _build_full_model_grid_components(
        model_name,
        grid=grid_tp,
        n_splines=n_splines,
        spline_order=spline_order,
        segment_edges=segment_edges,
        pos_bounds=pos_bounds,
    )
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis,
        n_units,
    )
    dark_eta = (
        intercept[None, :]
        + (grid_components["place_basis"] @ coef_place_dark)
        + speed_ref_effect
    )
    if model_name == "visual":
        light_eta = (
            intercept[None, :]
            + coef_light_offset[None, :]
            + (grid_components["place_basis"] @ model_specific_coef["coef_place_light"])
            + speed_ref_effect
        )
    else:
        gain_key = {
            "task_segment_bump": "coef_segment_bump_gain",
            "task_segment_scalar": "coef_segment_scalar_gain",
            "task_dense_gain": "coef_gain_spline",
        }[model_name]
        light_eta = (
            dark_eta
            + coef_light_offset[None, :]
            + (grid_components["gain_basis"] @ model_specific_coef[gain_key])
        )

    result = {
        "model_name": model_name,
        "unit_ids": unit_ids,
        "has_speed": speed_by_epoch is not None,
        "bin_size_s": float(bin_size_s),
        "spatial_bin_size_cm": float(spatial_bin_size_cm),
        "trajectory_length_cm": float(trajectory_length_cm),
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "ridge": float(ridge),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        "grid_tp": grid_tp,
        "dark_hz_grid": np.exp(dark_eta) / float(bin_size_s),
        "light_hz_grid": np.exp(light_eta) / float(bin_size_s),
        "train_light_hz_grid": np.exp(light_eta) / float(bin_size_s),
        "coef_intercept": intercept,
        "coef_light_offset": coef_light_offset,
        "coef_place_dark": coef_place_dark,
        "speed_feature_mode": speed_outputs["speed_feature_mode"],
        "n_speed_features": speed_outputs["n_speed_features"],
        "speed_basis": speed_outputs["speed_basis"],
        "speed_spline_order": speed_outputs["speed_spline_order"],
        "speed_basis_bounds": speed_outputs["speed_basis_bounds"],
        "speed_reference_value": speed_outputs["speed_reference_value"],
        "speed_mean": speed_outputs["speed_mean"],
        "speed_std": speed_outputs["speed_std"],
        "coef_speed": speed_outputs["coef_speed_full_all"],
        "coef_speed_basis": speed_outputs["coef_speed_basis_full_all"],
        **cv_accumulator,
        **model_specific_coef,
    }
    if model_name in {"task_segment_bump", "task_segment_scalar"}:
        result["segment_edges"] = np.asarray(design_metadata["segment_edges"], dtype=float)
    else:
        result["segment_edges"] = np.asarray(segment_edges, dtype=float)
    return result


def _stack_selected_array(
    results_by_traj: dict[str, dict[float, dict[str, Any]]],
    ridge_values: list[float],
    key: str,
) -> np.ndarray:
    """Stack one selected-fit field into `(trajectory, ridge, ...)` order."""
    return np.stack(
        [
            np.stack(
                [
                    np.asarray(results_by_traj[trajectory][float(ridge)][key])
                    for ridge in ridge_values
                ],
                axis=0,
            )
            for trajectory in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def build_selected_candidate_dataset(
    *,
    model_name: str,
    results_by_traj: dict[str, dict[float, dict[str, Any]]],
    ridge_values: list[float],
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    dark_movement_firing_rates: np.ndarray,
    light_movement_firing_rates: np.ndarray,
    min_dark_firing_rate_hz: float,
    min_light_firing_rate_hz: float,
    segment_edges: np.ndarray,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build one candidate dataset for a model/bin/spline grid point."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task-progression dark/light fits as NetCDF."
        ) from exc

    first = _first_result(results_by_traj)
    unit_ids = np.asarray(first["unit_ids"])
    tp_grid = np.asarray(first["grid_tp"], dtype=float)
    place_basis_count = int(np.asarray(first["coef_place_dark"]).shape[0])
    speed_basis_count = int(np.asarray(first["coef_speed_basis"]).shape[0])
    segment_edges = np.asarray(segment_edges, dtype=float)

    attrs = {
        "schema_version": "5",
        "fit_stage": "candidate",
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "model_name": model_name,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "light_train_epoch": light_epoch,
        "dark_train_epoch": dark_epoch,
        "bin_size_s": float(first["bin_size_s"]),
        "spatial_bin_size_cm": float(first["spatial_bin_size_cm"]),
        "trajectory_length_cm": float(first["trajectory_length_cm"]),
        "n_splines": int(first["n_splines"]),
        "spline_order": int(first["spline_order"]),
        "ridge_candidates_json": json.dumps([float(ridge) for ridge in ridge_values]),
        "has_speed": bool(first["has_speed"]),
        "has_light_offset_term": True,
        "pos_bounds_lower": float(first["pos_bounds"][0]),
        "pos_bounds_upper": float(first["pos_bounds"][1]),
        "speed_feature_mode": str(first["speed_feature_mode"]),
        "n_speed_features": int(first["n_speed_features"]),
        "speed_basis": str(first["speed_basis"]),
        "speed_spline_order": float(first["speed_spline_order"]),
        "selection_metric": SELECTION_METRIC,
        "selection_metric_description": (
            "Cross-validated Poisson information gain versus a train-split "
            "constant-rate null, in bits/spike."
        ),
        "selection_aggregation": SELECTION_AGGREGATION,
        "min_dark_firing_rate_hz": float(min_dark_firing_rate_hz),
        "min_light_firing_rate_hz": float(min_light_firing_rate_hz),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
    }

    data_vars: dict[str, tuple[tuple[str, ...] | str, np.ndarray]] = {
        "dark_movement_firing_rate_hz": (
            "unit",
            np.asarray(dark_movement_firing_rates, dtype=float),
        ),
        "light_movement_firing_rate_hz": (
            "unit",
            np.asarray(light_movement_firing_rates, dtype=float),
        ),
        "speed_feature_mode": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_feature_mode"),
        ),
        "n_speed_features": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "n_speed_features"),
        ),
        "speed_basis": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_basis"),
        ),
        "speed_spline_order": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_spline_order"),
        ),
        "speed_basis_bounds": (
            ("trajectory", "ridge", "speed_bound"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_basis_bounds"),
        ),
        "speed_reference_value": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_reference_value"),
        ),
        "speed_mean": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_mean"),
        ),
        "speed_std": (
            ("trajectory", "ridge"),
            _stack_selected_array(results_by_traj, ridge_values, "speed_std"),
        ),
        "dark_hz_grid": (
            ("trajectory", "ridge", "tp_grid", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "dark_hz_grid"),
        ),
        "light_hz_grid": (
            ("trajectory", "ridge", "tp_grid", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "light_hz_grid"),
        ),
        "train_light_hz_grid": (
            ("trajectory", "ridge", "tp_grid", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "train_light_hz_grid"),
        ),
        "coef_intercept": (
            ("trajectory", "ridge", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_intercept"),
        ),
        "coef_light_offset": (
            ("trajectory", "ridge", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_light_offset"),
        ),
        "coef_place_dark": (
            ("trajectory", "ridge", "place_basis", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_place_dark"),
        ),
        "coef_speed": (
            ("trajectory", "ridge", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_speed"),
        ),
        "coef_speed_basis": (
            ("trajectory", "ridge", "speed_basis_feature", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_speed_basis"),
        ),
        "segment_edges": (
            ("segment_edge",),
            segment_edges,
        ),
    }
    for suffix in CV_SCORE_SUFFIXES:
        for metric_prefix in ("ll_sum", "null_ll_sum", "spike_sum", "ll_bits_per_spike"):
            key = f"{metric_prefix}_cv_{suffix}"
            data_vars[key] = (
                ("trajectory", "ridge", "unit"),
                _stack_selected_array(results_by_traj, ridge_values, key),
            )

    coords: dict[str, np.ndarray] = {
        "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
        "ridge": np.asarray(ridge_values, dtype=float),
        "unit": unit_ids,
        "tp_grid": tp_grid,
        "place_basis": np.arange(place_basis_count, dtype=int),
        "speed_basis_feature": np.arange(speed_basis_count, dtype=int),
        "speed_bound": np.asarray(["lower", "upper"], dtype=str),
        "segment_edge": segment_edges,
    }
    if model_name == "visual":
        data_vars["coef_place_light"] = (
            ("trajectory", "ridge", "place_basis", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_place_light"),
        )
    elif model_name == "task_segment_bump":
        segment_basis_count = int(
            np.asarray(first["coef_segment_bump_gain"]).shape[0]
        )
        coords["segment_basis"] = np.arange(segment_basis_count, dtype=int)
        data_vars["coef_segment_bump_gain"] = (
            ("trajectory", "ridge", "segment_basis", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_segment_bump_gain"),
        )
    elif model_name == "task_segment_scalar":
        segment_basis_count = int(
            np.asarray(first["coef_segment_scalar_gain"]).shape[0]
        )
        coords["segment_basis"] = np.arange(segment_basis_count, dtype=int)
        data_vars["coef_segment_scalar_gain"] = (
            ("trajectory", "ridge", "segment_basis", "unit"),
            _stack_selected_array(
                results_by_traj,
                ridge_values,
                "coef_segment_scalar_gain",
            ),
        )
    elif model_name == "task_dense_gain":
        gain_basis_count = int(np.asarray(first["coef_gain_spline"]).shape[0])
        coords["gain_basis_feature"] = np.arange(gain_basis_count, dtype=int)
        data_vars["coef_gain_spline"] = (
            ("trajectory", "ridge", "gain_basis_feature", "unit"),
            _stack_selected_array(results_by_traj, ridge_values, "coef_gain_spline"),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    dataset["coef_light_offset"].attrs["description"] = (
        "Explicit scalar light indicator coefficient, fit for every selected-fit model."
    )
    return dataset


def score_candidate_dataset(dataset: "xr.Dataset") -> list[dict[str, Any]]:
    """Return pooled CV scores for each ridge in one candidate dataset."""
    score_values = np.asarray(dataset[SELECTION_METRIC].values, dtype=float)
    records: list[dict[str, Any]] = []
    for ridge_index, ridge in enumerate(np.asarray(dataset.coords["ridge"].values, dtype=float)):
        values = score_values[:, ridge_index, :].reshape(-1)
        finite = values[np.isfinite(values)]
        records.append(
            {
                "model_name": str(dataset.attrs["model_name"]),
                "bin_size_s": float(dataset.attrs["bin_size_s"]),
                "spatial_bin_size_cm": float(dataset.attrs["spatial_bin_size_cm"]),
                "trajectory_length_cm": float(dataset.attrs["trajectory_length_cm"]),
                "n_splines": int(dataset.attrs["n_splines"]),
                "ridge": float(ridge),
                "score_median": (
                    float(np.median(finite)) if finite.size > 0 else np.nan
                ),
                "score_mean": float(np.mean(finite)) if finite.size > 0 else np.nan,
                "n_finite": int(finite.size),
            }
        )
    return records


def _is_better_selection_record(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    compare_bin_and_spatial: bool,
) -> bool:
    """Return whether `candidate` wins the deterministic selection tie-break."""
    candidate_score = float(candidate["score_median"])
    if not np.isfinite(candidate_score):
        return False
    if incumbent is None:
        return True
    incumbent_score = float(incumbent["score_median"])
    if not np.isfinite(incumbent_score):
        return True
    if candidate_score > incumbent_score + HYPERPARAMETER_TIE_ATOL:
        return True
    if candidate_score < incumbent_score - HYPERPARAMETER_TIE_ATOL:
        return False
    if compare_bin_and_spatial:
        if float(candidate["bin_size_s"]) != float(incumbent["bin_size_s"]):
            return float(candidate["bin_size_s"]) > float(incumbent["bin_size_s"])
        if not np.isclose(
            float(candidate["spatial_bin_size_cm"]),
            float(incumbent["spatial_bin_size_cm"]),
        ):
            return float(candidate["spatial_bin_size_cm"]) > float(
                incumbent["spatial_bin_size_cm"]
            )
    return float(candidate["ridge"]) > float(incumbent["ridge"])


def choose_visual_shared_hyperparameters(
    selection_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Choose shared bin size and spatial-bin basis from visual CV scores."""
    best: dict[str, Any] | None = None
    for record in selection_records:
        if record["model_name"] != SELECTION_MODEL_NAME:
            continue
        if _is_better_selection_record(
            record,
            best,
            compare_bin_and_spatial=True,
        ):
            best = dict(record)
    if best is None:
        raise ValueError("No finite visual CV score was available for selection.")
    return {
        "bin_size_s": float(best["bin_size_s"]),
        "spatial_bin_size_cm": float(best["spatial_bin_size_cm"]),
        "trajectory_length_cm": float(best["trajectory_length_cm"]),
        "n_splines": int(best["n_splines"]),
        "visual_ridge": float(best["ridge"]),
        "score_median": float(best["score_median"]),
        "score_mean": float(best["score_mean"]),
        "n_finite": int(best["n_finite"]),
    }


def choose_model_ridge(
    selection_records: Sequence[dict[str, Any]],
    *,
    model_name: str,
    bin_size_s: float,
    spatial_bin_size_cm: float,
) -> dict[str, Any]:
    """Choose the ridge value for one model at the shared bin/spatial choice."""
    best: dict[str, Any] | None = None
    for record in selection_records:
        if record["model_name"] != model_name:
            continue
        if not np.isclose(float(record["bin_size_s"]), float(bin_size_s)):
            continue
        if not np.isclose(
            float(record["spatial_bin_size_cm"]),
            float(spatial_bin_size_cm),
        ):
            continue
        if _is_better_selection_record(
            record,
            best,
            compare_bin_and_spatial=False,
        ):
            best = dict(record)
    if best is None:
        raise ValueError(
            f"No finite CV score was available for model {model_name!r} at "
            f"bin_size_s={bin_size_s}, spatial_bin_size_cm={spatial_bin_size_cm}."
        )
    return best


def build_selected_model_dataset(
    candidate_dataset: "xr.Dataset",
    *,
    selected_ridge: float,
    selection_score: float,
    shared_selection: dict[str, Any],
) -> "xr.Dataset":
    """Return the selected-ridge view of a candidate dataset."""
    selected = candidate_dataset.sel(ridge=float(selected_ridge), drop=True).copy(deep=True)
    selected.attrs.update(
        {
            "fit_stage": "selected",
            "selected_bin_size_s": float(shared_selection["bin_size_s"]),
            "selected_spatial_bin_size_cm": float(
                shared_selection["spatial_bin_size_cm"]
            ),
            "selected_n_splines": int(shared_selection["n_splines"]),
            "selected_ridge": float(selected_ridge),
            "selection_metric": SELECTION_METRIC,
            "selection_score": float(selection_score),
            "selection_model_name": SELECTION_MODEL_NAME,
            "selection_visual_ridge": float(shared_selection["visual_ridge"]),
            "selection_visual_score": float(shared_selection["score_median"]),
        }
    )
    return selected


def build_selection_summary_dataset(
    *,
    selection_records: Sequence[dict[str, Any]],
    model_names: Sequence[str],
    bin_sizes_s: Sequence[float],
    position_basis_configs: Sequence[dict[str, Any]],
    ridge_values: Sequence[float],
    selected_by_model: dict[str, dict[str, Any]],
    shared_selection: dict[str, Any],
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build the CV selection summary across model and hyperparameter candidates."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task-progression selection summaries."
        ) from exc

    model_names = list(model_names)
    bin_sizes = [float(value) for value in bin_sizes_s]
    position_basis_configs = list(position_basis_configs)
    spatial_bin_sizes = [
        float(config["spatial_bin_size_cm"]) for config in position_basis_configs
    ]
    n_splines_by_spatial = [
        int(config["n_splines"]) for config in position_basis_configs
    ]
    trajectory_lengths = [
        float(config["trajectory_length_cm"]) for config in position_basis_configs
    ]
    ridges = [float(value) for value in ridge_values]
    shape = (len(model_names), len(bin_sizes), len(spatial_bin_sizes), len(ridges))
    score_median = np.full(shape, np.nan, dtype=float)
    score_mean = np.full(shape, np.nan, dtype=float)
    n_finite = np.zeros(shape, dtype=int)
    model_index = {name: index for index, name in enumerate(model_names)}
    bin_index = {value: index for index, value in enumerate(bin_sizes)}
    spatial_index = {value: index for index, value in enumerate(spatial_bin_sizes)}
    ridge_index = {value: index for index, value in enumerate(ridges)}
    for record in selection_records:
        key = (
            model_index.get(str(record["model_name"])),
            bin_index.get(float(record["bin_size_s"])),
            spatial_index.get(float(record["spatial_bin_size_cm"])),
            ridge_index.get(float(record["ridge"])),
        )
        if any(value is None for value in key):
            continue
        mi, bi, si, ri = key
        score_median[mi, bi, si, ri] = float(record["score_median"])
        score_mean[mi, bi, si, ri] = float(record["score_mean"])
        n_finite[mi, bi, si, ri] = int(record["n_finite"])

    selected_ridge = np.full((len(model_names),), np.nan, dtype=float)
    selected_score = np.full((len(model_names),), np.nan, dtype=float)
    for index, model_name in enumerate(model_names):
        if model_name in selected_by_model:
            selected_ridge[index] = float(selected_by_model[model_name]["ridge"])
            selected_score[index] = float(selected_by_model[model_name]["score_median"])

    return xr.Dataset(
        data_vars={
            "cv_score_median": (
                ("model", "bin_size_s", "spatial_bin_size_cm", "ridge"),
                score_median,
            ),
            "cv_score_mean": (
                ("model", "bin_size_s", "spatial_bin_size_cm", "ridge"),
                score_mean,
            ),
            "cv_score_n_finite": (
                ("model", "bin_size_s", "spatial_bin_size_cm", "ridge"),
                n_finite,
            ),
            "n_splines_by_spatial_bin_size": (
                "spatial_bin_size_cm",
                np.asarray(n_splines_by_spatial, dtype=int),
            ),
            "selected_ridge": ("model", selected_ridge),
            "selected_score_median": ("model", selected_score),
        },
        coords={
            "model": np.asarray(model_names, dtype=str),
            "bin_size_s": np.asarray(bin_sizes, dtype=float),
            "spatial_bin_size_cm": np.asarray(spatial_bin_sizes, dtype=float),
            "ridge": np.asarray(ridges, dtype=float),
        },
        attrs={
            "schema_version": "5",
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
            "light_train_epoch": light_epoch,
            "dark_train_epoch": dark_epoch,
            "selection_metric": SELECTION_METRIC,
            "selection_metric_description": (
                "Cross-validated Poisson information gain versus a train-split "
                "constant-rate null, in bits/spike."
            ),
            "selection_aggregation": SELECTION_AGGREGATION,
            "selection_model_name": SELECTION_MODEL_NAME,
            "selected_bin_size_s": float(shared_selection["bin_size_s"]),
            "selected_spatial_bin_size_cm": float(
                shared_selection["spatial_bin_size_cm"]
            ),
            "selected_n_splines": int(shared_selection["n_splines"]),
            "trajectory_length_cm": float(np.unique(trajectory_lengths)[0]),
            "selection_visual_ridge": float(shared_selection["visual_ridge"]),
            "selection_visual_score": float(shared_selection["score_median"]),
            "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        },
    )


def _format_float_token(value: float) -> str:
    """Return a path-safe compact token for one float value."""
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def candidate_output_path(
    data_dir: Path,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_name: str,
    bin_size_s: float,
    spatial_bin_size_cm: float,
    n_splines: int,
    spline_order: int,
) -> Path:
    """Return the candidate output path for one model/hyperparameter point."""
    return (
        data_dir
        / "candidates"
        / (
            f"{region}_{light_epoch}_vs_{dark_epoch}_{model_name}_"
            f"bin{_format_float_token(bin_size_s)}s_"
            f"spbin{_format_float_token(spatial_bin_size_cm)}cm_"
            f"nspl{int(n_splines)}_"
            f"order{int(spline_order)}.nc"
        )
    )


def selected_output_path(
    data_dir: Path,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    model_name: str,
) -> Path:
    """Return the selected model output path."""
    return data_dir / "selected" / f"{region}_{light_epoch}_vs_{dark_epoch}_{model_name}_selected.nc"


def selection_summary_output_path(
    data_dir: Path,
    *,
    region: str,
    light_epoch: str,
    dark_epoch: str,
) -> Path:
    """Return the shared selection-summary output path."""
    return data_dir / "selection_summary" / f"{region}_{light_epoch}_vs_{dark_epoch}_selection_summary.nc"



def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the dark/light task-progression GLM script."""
    parser = argparse.ArgumentParser(
        description="Fit selected dark/light task-progression GLMs for one session"
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=_CUDA_VISIBLE_DEVICES_CLI,
        help=(
            "Optional CUDA_VISIBLE_DEVICES value applied before importing JAX. "
            "Default: unset"
        ),
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
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Regions to fit. Default: {' '.join(REGIONS)}",
    )
    parser.add_argument(
        "--light-epochs",
        nargs="+",
        help="Explicit run epoch labels to use as light epochs.",
    )
    parser.add_argument(
        "--dark-epoch",
        required=True,
        help="Run epoch label to use as the dark epoch.",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore per epoch. Default: {DEFAULT_POSITION_OFFSET}",
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
        "--v1-min-dark-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help=f"Minimum dark-epoch movement firing rate for V1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['v1']}",
    )
    parser.add_argument(
        "--v1-min-light-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help=f"Minimum light-epoch movement firing rate for V1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['v1']}",
    )
    parser.add_argument(
        "--ca1-min-dark-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=f"Minimum dark-epoch movement firing rate for CA1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}",
    )
    parser.add_argument(
        "--ca1-min-light-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=f"Minimum light-epoch movement firing rate for CA1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        help=(
            "Legacy single spike-count bin-size candidate in seconds. "
            "Use --bin-sizes-s for a sweep."
        ),
    )
    parser.add_argument(
        "--bin-sizes-s",
        nargs="+",
        type=float,
        help="Spike-count bin-size candidates in seconds. Default: 0.02 0.05",
    )
    parser.add_argument(
        "--ridges",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGES),
        help="Ridge strengths to sweep for each model. Default: 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of lap-level CV folds. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for fold assignment. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--models",
        "--model-families",
        dest="models",
        nargs="+",
        choices=MODEL_NAME_CHOICES,
        default=list(DEFAULT_MODEL_NAMES),
        help=(
            "Swap-compatible models to fit. `visual` is always included. "
            "Deprecated aliases are accepted and normalized."
        ),
    )
    parser.add_argument(
        "--spatial-bin-sizes-cm",
        type=float,
        nargs="+",
        default=list(DEFAULT_SPATIAL_BIN_SIZES_CM),
        help=(
            "Spatial bin-size candidates used to derive place-field spline "
            "counts by ceil(track_length / bin_size). Default: "
            + " ".join(f"{value:g}" for value in DEFAULT_SPATIAL_BIN_SIZES_CM)
        ),
    )
    parser.add_argument(
        "--spline-order",
        type=int,
        default=4,
        help="Spline order for the shared dark/place field. Default: 4",
    )
    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        "--use-speed",
        dest="use_speed",
        action="store_true",
        help="Include speed in the dark/light GLM fits.",
    )
    speed_group.add_argument(
        "--no-speed",
        dest="use_speed",
        action="store_false",
        help="Exclude speed from the dark/light GLM fits.",
    )
    parser.set_defaults(use_speed=True)
    parser.add_argument(
        "--speed-feature-mode",
        choices=("linear", "bspline"),
        default="linear",
        help="Speed covariate parameterization to use when speed is enabled. Default: linear",
    )
    parser.add_argument(
        "--n-splines-speed",
        type=int,
        default=5,
        help="Number of spline basis functions for bspline speed features. Default: 5",
    )
    parser.add_argument(
        "--spline-order-speed",
        type=int,
        default=4,
        help="Spline order for bspline speed features. Default: 4",
    )
    parser.add_argument(
        "--speed-bounds",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Optional explicit bounds for bspline speed features.",
    )
    parser.add_argument(
        "--segment-edges",
        nargs="+",
        type=float,
        help="Explicit TP segment edges for segment-based models. Defaults to geometry-derived edges.",
    )
    args = parser.parse_args()
    args.model_families = args.models
    return args


def main() -> None:
    """Run the selected dark/light task-progression GLM workflow."""
    args = parse_arguments()
    model_names, model_messages = normalize_model_names(args.models)
    candidate_values = normalize_candidate_values(args)
    bin_sizes_s = [float(value) for value in candidate_values["bin_sizes_s"]]
    spatial_bin_sizes_cm = [
        float(value) for value in candidate_values["spatial_bin_sizes_cm"]
    ]
    ridge_values = [float(value) for value in dict.fromkeys(args.ridges)]
    if any(value < 0 for value in ridge_values):
        raise ValueError("Ridge strengths must be non-negative.")
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if args.spline_order <= 0:
        raise ValueError("--spline-order must be positive.")
    if args.n_splines_speed <= 0:
        raise ValueError("--n-splines-speed must be positive.")
    if args.spline_order_speed <= 0:
        raise ValueError("--spline-order-speed must be positive.")

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    position_basis_configs = build_position_basis_configs(
        animal_name=args.animal_name,
        spatial_bin_sizes_cm=spatial_bin_sizes_cm,
        spline_order=args.spline_order,
    )
    print(f"Loading session for {args.animal_name} {args.date}.")
    if args.cuda_visible_devices is not None:
        print(f"Using CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}.")
    for message in model_messages:
        print(message)
    print(
        "Candidate bin sizes: "
        + ", ".join(f"{value:g}" for value in bin_sizes_s)
        + "; spatial bin sizes: "
        + ", ".join(f"{value:g}cm" for value in spatial_bin_sizes_cm)
        + "; ridges: "
        + ", ".join(f"{value:g}" for value in ridge_values)
    )
    print(
        "Derived place basis configs: "
        + ", ".join(
            (
                f"{config['spatial_bin_size_cm']:g}cm -> "
                f"{config['n_splines']} splines"
            )
            for config in position_basis_configs
        )
    )
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    epoch_pairs = select_light_dark_pairs(
        session["run_epochs"],
        dark_epoch=args.dark_epoch,
        light_epochs=args.light_epochs,
    )
    print(
        "Using valid run epochs: "
        + ", ".join(str(epoch) for epoch in session["run_epochs"])
    )
    print(
        "Fitting light/dark pairs: "
        + ", ".join(f"{light} vs {dark}" for light, dark in epoch_pairs)
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    segment_edges = (
        derive_default_segment_edges(args.animal_name)
        if args.segment_edges is None
        else _validate_segment_edges(args.segment_edges)
    )
    print(
        "Using segment edges: "
        + ", ".join(f"{edge:.4f}" for edge in np.asarray(segment_edges, dtype=float))
    )
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("candidates", "selected", "selection_summary"):
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)

    dark_region_thresholds = {
        "v1": float(args.v1_min_dark_fr_hz),
        "ca1": float(args.ca1_min_dark_fr_hz),
    }
    light_region_thresholds = {
        "v1": float(args.v1_min_light_fr_hz),
        "ca1": float(args.ca1_min_light_fr_hz),
    }
    saved_datasets: list[Path] = []
    skipped_fits: list[dict[str, Any]] = []
    speed_by_run = session["speed_by_run"] if args.use_speed else None

    for region in args.regions:
        print(f"Preparing region {region.upper()}.")
        spikes_region = session["spikes_by_region"][region]
        for light_epoch, dark_epoch in epoch_pairs:
            print(f"  Pair {light_epoch} vs {dark_epoch}.")
            dark_epoch_rates = np.asarray(
                movement_firing_rates[region][dark_epoch],
                dtype=float,
            )
            light_epoch_rates = np.asarray(
                movement_firing_rates[region][light_epoch],
                dtype=float,
            )
            train_fr_masks = build_train_epoch_fr_mask(
                dark_epoch_rates,
                light_epoch_rates,
                min_dark_fr_hz=dark_region_thresholds[region],
                min_light_fr_hz=light_region_thresholds[region],
            )
            unit_mask = train_fr_masks["combined"]
            n_units_total = int(unit_mask.size)
            print(
                f"    {region.upper()} FR mask: "
                f"dark>{dark_region_thresholds[region]:.3f} Hz passed "
                f"{int(train_fr_masks['dark'].sum())}/{n_units_total}; "
                f"light>{light_region_thresholds[region]:.3f} Hz passed "
                f"{int(train_fr_masks['light'].sum())}/{n_units_total}; "
                f"both passed {int(unit_mask.sum())}/{n_units_total}."
            )
            if not np.any(unit_mask):
                skipped_fits.append(
                    {
                        "region": region,
                        "light_epoch": light_epoch,
                        "dark_epoch": dark_epoch,
                        "reason": (
                            "no units passed the combined dark/light train firing-rate "
                            "thresholds"
                        ),
                        "dark_threshold_hz": dark_region_thresholds[region],
                        "light_threshold_hz": light_region_thresholds[region],
                    }
                )
                print(
                    f"    Skipping pair: no {region.upper()} units passed the "
                    "combined dark/light train thresholds "
                    f"(dark>{dark_region_thresholds[region]:.3f} Hz, "
                    f"light>{light_region_thresholds[region]:.3f} Hz)."
                )
                continue

            selected_dark_rates = dark_epoch_rates[unit_mask]
            selected_light_rates = light_epoch_rates[unit_mask]
            print(
                f"    Selected {int(unit_mask.sum())} {region.upper()} units above "
                f"{dark_region_thresholds[region]:.3f} Hz in {dark_epoch} and above "
                f"{light_region_thresholds[region]:.3f} Hz in {light_epoch}."
            )
            try:
                folds_by_traj = {
                    trajectory: build_lap_cv_folds_for_trajectory(
                        trajectory_intervals=session["trajectory_intervals"],
                        movement_by_run=session["movement_by_run"],
                        dark_epoch=dark_epoch,
                        light_epoch=light_epoch,
                        trajectory=trajectory,
                        n_folds=args.n_folds,
                        seed=args.seed,
                    )
                    for trajectory in TRAJECTORY_TYPES
                }
            except Exception as exc:
                skipped_fits.append(
                    {
                        "region": region,
                        "light_epoch": light_epoch,
                        "dark_epoch": dark_epoch,
                        "reason": "lap CV fold construction failed",
                        "error": str(exc),
                    }
                )
                print(f"    Skipping pair after lap-CV setup failure: {exc}")
                continue

            fold_metadata = {
                trajectory: [
                    dict(fold["metadata"]) for fold in folds_by_traj[trajectory]
                ]
                for trajectory in TRAJECTORY_TYPES
            }
            fit_parameters_common = {
                "cuda_visible_devices": args.cuda_visible_devices,
                "position_offset": args.position_offset,
                "speed_threshold_cm_s": args.speed_threshold_cm_s,
                "bin_sizes_s": bin_sizes_s,
                "spatial_bin_sizes_cm": spatial_bin_sizes_cm,
                "position_basis_configs": position_basis_configs,
                "ridges": ridge_values,
                "n_folds": args.n_folds,
                "seed": args.seed,
                "models": model_names,
                "selection_model_name": SELECTION_MODEL_NAME,
                "selection_metric": SELECTION_METRIC,
                "selection_aggregation": SELECTION_AGGREGATION,
                "spline_order": args.spline_order,
                "use_speed": args.use_speed,
                "speed_feature_mode": args.speed_feature_mode,
                "n_splines_speed": args.n_splines_speed,
                "spline_order_speed": args.spline_order_speed,
                "speed_bounds": args.speed_bounds,
                "segment_edges": [float(edge) for edge in segment_edges],
                "cv_fold_scope": "lap_level_by_trajectory_movement_only",
                "fold_metadata": fold_metadata,
                "region_threshold_hz": dark_region_thresholds[region],
                "dark_region_threshold_hz": dark_region_thresholds[region],
                "light_region_threshold_hz": light_region_thresholds[region],
            }

            candidate_datasets: dict[tuple[str, float, float], Any] = {}
            selection_records: list[dict[str, Any]] = []

            def fit_candidate_dataset(
                model_name: str,
                *,
                bin_size_s: float,
                position_basis: dict[str, Any],
            ) -> Any | None:
                spatial_bin_size_cm = float(position_basis["spatial_bin_size_cm"])
                trajectory_length_cm = float(position_basis["trajectory_length_cm"])
                n_splines = int(position_basis["n_splines"])
                print(
                    f"    Fitting {model_name} candidate "
                    f"bin={bin_size_s:g}s, spatial_bin={spatial_bin_size_cm:g}cm "
                    f"({n_splines} splines) "
                    f"across {len(TRAJECTORY_TYPES)} trajectories and "
                    f"{len(ridge_values)} ridge value(s)."
                )
                results_by_traj: dict[str, dict[float, dict[str, Any]]] = {
                    trajectory: {} for trajectory in TRAJECTORY_TYPES
                }
                for trajectory in TRAJECTORY_TYPES:
                    for ridge in ridge_values:
                        print(
                            f"      {trajectory}: ridge={float(ridge):.3g}"
                        )
                        try:
                            results_by_traj[trajectory][float(ridge)] = (
                                _fit_selected_full_model_per_traj(
                                    model_name=model_name,
                                    spikes=spikes_region,
                                    trajectory_ep_by_epoch=session[
                                        "trajectory_intervals"
                                    ],
                                    tp_by_epoch=session[
                                        "task_progression_by_trajectory"
                                    ],
                                    speed_by_epoch=speed_by_run,
                                    light_epoch=light_epoch,
                                    dark_epoch=dark_epoch,
                                    traj_name=trajectory,
                                    folds=folds_by_traj[trajectory],
                                    movement_by_run=session["movement_by_run"],
                                    bin_size_s=bin_size_s,
                                    n_splines=n_splines,
                                    spline_order=args.spline_order,
                                    spatial_bin_size_cm=spatial_bin_size_cm,
                                    trajectory_length_cm=trajectory_length_cm,
                                    ridge=float(ridge),
                                    unit_mask=unit_mask,
                                    segment_edges=segment_edges,
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
                        except Exception as exc:
                            skipped_fits.append(
                                {
                                    "region": region,
                                    "light_epoch": light_epoch,
                                    "dark_epoch": dark_epoch,
                                    "model_name": model_name,
                                    "trajectory": trajectory,
                                    "ridge": float(ridge),
                                    "bin_size_s": float(bin_size_s),
                                    "spatial_bin_size_cm": spatial_bin_size_cm,
                                    "trajectory_length_cm": trajectory_length_cm,
                                    "n_splines": int(n_splines),
                                    "reason": "candidate fit failed",
                                    "error": str(exc),
                                }
                            )
                            print(
                                f"      Failed on {trajectory} at "
                                f"ridge={float(ridge):.3g}: {exc}"
                            )
                            return None

                fit_parameters = {
                    **fit_parameters_common,
                    "model_name": model_name,
                    "bin_size_s": float(bin_size_s),
                    "spatial_bin_size_cm": spatial_bin_size_cm,
                    "trajectory_length_cm": trajectory_length_cm,
                    "n_splines": int(n_splines),
                }
                dataset = build_selected_candidate_dataset(
                    model_name=model_name,
                    results_by_traj=results_by_traj,
                    ridge_values=ridge_values,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    dark_movement_firing_rates=selected_dark_rates,
                    light_movement_firing_rates=selected_light_rates,
                    min_dark_firing_rate_hz=dark_region_thresholds[region],
                    min_light_firing_rate_hz=light_region_thresholds[region],
                    segment_edges=segment_edges,
                    sources=session["sources"],
                    fit_parameters=fit_parameters,
                )
                output_path = candidate_output_path(
                    data_dir,
                    region=region,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    model_name=model_name,
                    bin_size_s=bin_size_s,
                    spatial_bin_size_cm=spatial_bin_size_cm,
                    n_splines=n_splines,
                    spline_order=args.spline_order,
                )
                dataset.to_netcdf(output_path)
                saved_datasets.append(output_path)
                print(f"    Saved candidate dataset to {output_path}.")
                return dataset

            print(
                "    Fitting visual candidates first for shared bin-size and "
                "spatial-bin-size selection."
            )
            for bin_size_s in bin_sizes_s:
                for position_basis in position_basis_configs:
                    dataset = fit_candidate_dataset(
                        SELECTION_MODEL_NAME,
                        bin_size_s=bin_size_s,
                        position_basis=position_basis,
                    )
                    if dataset is None:
                        continue
                    key = (
                        SELECTION_MODEL_NAME,
                        float(bin_size_s),
                        float(position_basis["spatial_bin_size_cm"]),
                    )
                    candidate_datasets[key] = dataset
                    selection_records.extend(score_candidate_dataset(dataset))

            try:
                shared_selection = choose_visual_shared_hyperparameters(
                    selection_records
                )
            except Exception as exc:
                skipped_fits.append(
                    {
                        "region": region,
                        "light_epoch": light_epoch,
                        "dark_epoch": dark_epoch,
                        "reason": "visual shared hyperparameter selection failed",
                        "error": str(exc),
                    }
                )
                print(f"    Skipping pair after visual selection failure: {exc}")
                continue

            selected_bin_size_s = float(shared_selection["bin_size_s"])
            selected_spatial_bin_size_cm = float(
                shared_selection["spatial_bin_size_cm"]
            )
            selected_n_splines = int(shared_selection["n_splines"])
            selected_position_basis = next(
                config
                for config in position_basis_configs
                if np.isclose(
                    float(config["spatial_bin_size_cm"]),
                    selected_spatial_bin_size_cm,
                )
            )
            print(
                "    Selected shared hyperparameters from visual CV: "
                f"bin={selected_bin_size_s:g}s, "
                f"spatial_bin={selected_spatial_bin_size_cm:g}cm "
                f"({selected_n_splines} splines), "
                f"visual ridge={shared_selection['visual_ridge']:.3g}, "
                f"score={shared_selection['score_median']:.5g} bits/spike."
            )

            for model_name in model_names:
                if model_name == SELECTION_MODEL_NAME:
                    continue
                dataset = fit_candidate_dataset(
                    model_name,
                    bin_size_s=selected_bin_size_s,
                    position_basis=selected_position_basis,
                )
                if dataset is None:
                    continue
                key = (model_name, selected_bin_size_s, selected_spatial_bin_size_cm)
                candidate_datasets[key] = dataset
                selection_records.extend(score_candidate_dataset(dataset))

            selected_by_model: dict[str, dict[str, Any]] = {}
            for model_name in model_names:
                key = (model_name, selected_bin_size_s, selected_spatial_bin_size_cm)
                if key not in candidate_datasets:
                    print(
                        f"    No candidate dataset available for {model_name}; "
                        "skipping selected output."
                    )
                    continue
                try:
                    selected_record = choose_model_ridge(
                        selection_records,
                        model_name=model_name,
                        bin_size_s=selected_bin_size_s,
                        spatial_bin_size_cm=selected_spatial_bin_size_cm,
                    )
                except Exception as exc:
                    skipped_fits.append(
                        {
                            "region": region,
                            "light_epoch": light_epoch,
                            "dark_epoch": dark_epoch,
                            "model_name": model_name,
                            "reason": "ridge selection failed",
                            "error": str(exc),
                        }
                    )
                    print(f"    Ridge selection failed for {model_name}: {exc}")
                    continue
                selected_by_model[model_name] = selected_record
                selected_dataset = build_selected_model_dataset(
                    candidate_datasets[key],
                    selected_ridge=float(selected_record["ridge"]),
                    selection_score=float(selected_record["score_median"]),
                    shared_selection=shared_selection,
                )
                output_path = selected_output_path(
                    data_dir,
                    region=region,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    model_name=model_name,
                )
                selected_dataset.to_netcdf(output_path)
                saved_datasets.append(output_path)
                print(
                    f"    Saved selected {model_name} dataset to {output_path} "
                    f"(ridge={float(selected_record['ridge']):.3g}, "
                    f"score={float(selected_record['score_median']):.5g})."
                )

            summary_dataset = build_selection_summary_dataset(
                selection_records=selection_records,
                model_names=model_names,
                bin_sizes_s=bin_sizes_s,
                position_basis_configs=position_basis_configs,
                ridge_values=ridge_values,
                selected_by_model=selected_by_model,
                shared_selection=shared_selection,
                animal_name=args.animal_name,
                date=args.date,
                region=region,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                fit_parameters=fit_parameters_common,
            )
            summary_path = selection_summary_output_path(
                data_dir,
                region=region,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
            )
            summary_dataset.to_netcdf(summary_path)
            saved_datasets.append(summary_path)
            print(f"    Saved selection summary to {summary_path}.")

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.dark_light_glm",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "cuda_visible_devices": args.cuda_visible_devices,
            "regions": args.regions,
            "light_epochs": args.light_epochs,
            "dark_epoch": args.dark_epoch,
            "epoch_pairs": epoch_pairs,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "region_dark_thresholds_hz": dark_region_thresholds,
            "region_light_thresholds_hz": light_region_thresholds,
            "bin_sizes_s": bin_sizes_s,
            "ridges": ridge_values,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "models": model_names,
            "selection_model_name": SELECTION_MODEL_NAME,
            "selection_metric": SELECTION_METRIC,
            "spatial_bin_sizes_cm": spatial_bin_sizes_cm,
            "position_basis_configs": position_basis_configs,
            "spline_order": args.spline_order,
            "use_speed": args.use_speed,
            "speed_feature_mode": args.speed_feature_mode,
            "n_splines_speed": args.n_splines_speed,
            "spline_order_speed": args.spline_order_speed,
            "speed_bounds": args.speed_bounds,
            "segment_edges": segment_edges.tolist(),
        },
        outputs={
            "sources": session["sources"],
            "saved_datasets": saved_datasets,
            "skipped_fits": skipped_fits,
        },
    )

    if saved_datasets:
        print(f"Saved {len(saved_datasets)} NetCDF fit dataset(s) to {data_dir}")
    if skipped_fits:
        print(f"Skipped {len(skipped_fits)} fit item(s); see run log for details.")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
