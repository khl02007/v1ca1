from __future__ import annotations

"""Fit dark/light task-progression GLMs for one session.

This module modernizes the legacy `task_progression_glm_updated3.py` workflow.
It reuses the shared task-progression session loaders, exposes a CLI for
session/model settings, fits several dark/light model families per trajectory,
and writes one NetCDF-backed `xarray.Dataset` per region/light-dark/model
combination.

Supported model families:

- `mult`: shared place field with spline gain modulation in light
- `mult_per_segment`: shared place field with segment-anchored raised-cosine gain
- `mult_fewer_gain_splines`: shared place field with fewer gain splines than dark-field splines
- `mult_per_segment_scalar`: shared place field with one scalar gain per segment
- `mult_per_segment_overlap`: segment-anchored raised-cosine gain with overlap
- `add_jax`: additive-in-rate light component fit with JAX/JAXopt when available
- `sep`: separate dark and light place fields
"""

import argparse
import json
import os
from dataclasses import dataclass
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
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    import xarray as xr


_CUDA_VISIBLE_DEVICES_CLI = pop_cuda_visible_devices_argument()
configure_cuda_visible_devices(_CUDA_VISIBLE_DEVICES_CLI)


try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    jax = None
    jnp = np

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
DEFAULT_RIDGES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
DEFAULT_LIGHT_EPOCHS = ("02_r1", "04_r2", "06_r3")
DEFAULT_DARK_EPOCH = "08_r4"
DEFAULT_MODEL_FAMILIES = (
    "mult",
    "mult_per_segment",
    "mult_fewer_gain_splines",
    "mult_per_segment_scalar",
    "mult_per_segment_overlap",
    "add_jax",
    "sep",
)
DEFAULT_SEED = 47
DEFAULT_GAIN_OVERLAP_FRAC = 0.05
CV_METRIC_NAMES = (
    "spike_sum_cv",
    "ll_base_sum_cv",
    "ll_full_sum_cv",
    "dLL_sum_cv",
    "ll_base_per_spike_cv",
    "ll_full_per_spike_cv",
    "dll_per_spike_cv",
    "ll_base_bits_per_spike_cv",
    "ll_full_bits_per_spike_cv",
    "dll_bits_per_spike_cv",
)


def _register_pytree_node_class(cls: type[Any]) -> type[Any]:
    """Register a dataclass as a JAX pytree when JAX is available."""
    if jax is None:
        return cls
    return jax.tree_util.register_pytree_node_class(cls)


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


def select_run_epochs(
    run_epochs: list[str],
    requested_epochs: list[str] | None,
) -> list[str]:
    """Return requested run epochs, defaulting to all available run epochs."""
    if not requested_epochs:
        return list(run_epochs)
    missing_epochs = [epoch for epoch in requested_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in available run epochs {run_epochs!r}: "
            f"{missing_epochs!r}"
        )
    return list(requested_epochs)


def select_light_dark_pairs(
    run_epochs: list[str],
    *,
    selected_epochs: list[str],
    light_epochs: list[str] | None,
    dark_epochs: list[str] | None,
) -> list[tuple[str, str]]:
    """Return validated `(light_epoch, dark_epoch)` pairs to fit."""
    epoch_pool = selected_epochs or run_epochs
    for epoch_list, label in ((light_epochs, "light"), (dark_epochs, "dark")):
        if epoch_list is None:
            continue
        missing = [epoch for epoch in epoch_list if epoch not in epoch_pool]
        if missing:
            raise ValueError(
                f"Requested {label} epochs were not found in the selected run epochs "
                f"{epoch_pool!r}: {missing!r}"
            )

    selected_dark_epochs = (
        list(dark_epochs) if dark_epochs else [DEFAULT_DARK_EPOCH]
    )
    selected_light_epochs = (
        list(light_epochs)
        if light_epochs
        else [
            epoch
            for epoch in DEFAULT_LIGHT_EPOCHS
            if epoch not in selected_dark_epochs
        ]
    )
    missing_defaults = [
        epoch
        for epoch in selected_light_epochs + selected_dark_epochs
        if epoch not in epoch_pool
    ]
    if missing_defaults:
        raise ValueError(
            "Default light/dark epochs were not found in the selected run epochs. "
            f"Selected run epochs: {epoch_pool!r}; missing defaults: {missing_defaults!r}"
        )
    if not selected_light_epochs:
        raise ValueError("No light epochs remain after applying the selected dark epochs.")

    pairs: list[tuple[str, str]] = []
    for light_epoch in selected_light_epochs:
        for dark_epoch in selected_dark_epochs:
            if light_epoch == dark_epoch:
                continue
            pairs.append((light_epoch, dark_epoch))
    if not pairs:
        raise ValueError("No distinct light/dark epoch pairs were selected.")
    return pairs


def derive_default_segment_edges(animal_name: str) -> np.ndarray:
    """Derive the three-segment default used by the legacy workflow."""
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


def _contiguous_folds(light: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split light and dark bins into contiguous folds and shuffle fold assignment."""
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    rng = np.random.default_rng(seed)
    light = np.asarray(light, dtype=int).reshape(-1)
    all_idx = np.arange(light.size, dtype=np.int64)

    idx_light = np.where(light == 1)[0].astype(np.int64)
    idx_dark = np.where(light == 0)[0].astype(np.int64)

    chunks_light = np.array_split(idx_light, n_folds)
    chunks_dark = np.array_split(idx_dark, n_folds)
    perm = rng.permutation(n_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_folds):
        perm_index = int(perm[fold_index])
        test_indices = np.concatenate(
            [chunks_light[perm_index], chunks_dark[perm_index]]
        ).astype(np.int64, copy=False)
        test_indices.sort()

        is_test = np.zeros(light.size, dtype=bool)
        is_test[test_indices] = True
        train_indices = all_idx[~is_test]
        folds.append((train_indices, test_indices))
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
    if mode == "linear" and coef_speed_basis_base.shape[0] > 0:
        coef_speed_base = coef_speed_basis_base[0, :]
        coef_speed_full = coef_speed_basis_full[0, :]
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


def _raised_cosine_taper_window(
    x: np.ndarray,
    *,
    core_lo: float,
    core_hi: float,
    ext_lo: float,
    ext_hi: float,
) -> np.ndarray:
    """Return a taper that is one in the core and decays smoothly in overlap."""
    x = np.asarray(x, dtype=float).reshape(-1)
    window = np.zeros_like(x, dtype=float)

    core_mask = (x >= core_lo) & (x <= core_hi)
    window[core_mask] = 1.0

    if ext_lo < core_lo:
        left_mask = (x >= ext_lo) & (x < core_lo)
        left_phase = (x[left_mask] - ext_lo) / (core_lo - ext_lo)
        window[left_mask] = 0.5 * (1.0 - np.cos(np.pi * left_phase))

    if core_hi < ext_hi:
        right_mask = (x > core_hi) & (x <= ext_hi)
        right_phase = (x[right_mask] - core_hi) / (ext_hi - core_hi)
        window[right_mask] = 0.5 * (1.0 + np.cos(np.pi * right_phase))
    return window


def _segment_overlap_bspline_basis(
    x: np.ndarray,
    segment_edges: Sequence[float],
    *,
    n_basis_per_segment: int,
    spline_order: int,
    overlap_frac: float,
    pos_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tapered per-segment spline blocks with optional overlap."""
    edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    x = np.asarray(x, dtype=float).reshape(-1)

    overlap_frac = float(overlap_frac)
    if not (0.0 <= overlap_frac < 0.5):
        raise ValueError("overlap_frac must be in [0, 0.5).")

    n_segments = edges.size - 1
    core_bounds = np.zeros((n_segments, 2), dtype=float)
    ext_bounds = np.zeros((n_segments, 2), dtype=float)
    pieces: list[np.ndarray] = []

    for segment_index in range(n_segments):
        core_lo = float(edges[segment_index])
        core_hi = float(edges[segment_index + 1])
        width = core_hi - core_lo
        overlap = overlap_frac * width
        ext_lo = max(float(pos_bounds[0]), core_lo - overlap)
        ext_hi = min(float(pos_bounds[1]), core_hi + overlap)

        core_bounds[segment_index] = (core_lo, core_hi)
        ext_bounds[segment_index] = (ext_lo, ext_hi)

        basis = BSplineEval(
            n_basis_funcs=int(n_basis_per_segment),
            order=int(spline_order),
            bounds=(ext_lo, ext_hi),
        )
        clipped_x = np.clip(x, ext_lo, ext_hi)
        piece = np.asarray(basis.compute_features(clipped_x), dtype=float)
        piece *= _raised_cosine_taper_window(
            x,
            core_lo=core_lo,
            core_hi=core_hi,
            ext_lo=ext_lo,
            ext_hi=ext_hi,
        )[:, None]
        pieces.append(piece)

    if not pieces:
        return np.zeros((x.size, 0), dtype=float), edges, core_bounds, ext_bounds
    return np.concatenate(pieces, axis=1), edges, core_bounds, ext_bounds


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
                restrict_ep_by_epoch[epoch].time_support
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


def _fit_exp_poisson_models(
    *,
    y_all: np.ndarray,
    l_all: np.ndarray,
    v_all: np.ndarray | None,
    x_base_nospeed: np.ndarray,
    x_full_nospeed: np.ndarray,
    ridge: float,
    n_folds: int,
    seed: int,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Run CV and full-data fits for nested exp-link Poisson GLMs."""
    _require_nemos()
    n_units = y_all.shape[1]
    has_speed = v_all is not None
    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    aggregate_sample_scores = lambda array: jnp.sum(array, axis=0)

    ll_base_sum = np.zeros(n_units, dtype=float)
    ll_full_sum = np.zeros(n_units, dtype=float)
    spike_sum = np.zeros(n_units, dtype=float)

    if has_speed:
        v_all = np.asarray(v_all, dtype=float).reshape(-1)
        V_all, speed_transform_all = _make_speed_design_all(
            v_all,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
    else:
        V_all = _empty_speed_design(y_all.shape[0])
        speed_transform_all = _empty_speed_feature_transform()

    for train_idx, test_idx in folds:
        if has_speed:
            V_train, V_test, _ = _make_speed_design_train_test(
                v_all,
                train_idx,
                test_idx,
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
            )
            x_base_train = np.concatenate(
                [x_base_nospeed[train_idx], V_train],
                axis=1,
            )
            x_base_test = np.concatenate(
                [x_base_nospeed[test_idx], V_test],
                axis=1,
            )
            x_full_train = np.concatenate(
                [x_full_nospeed[train_idx], V_train],
                axis=1,
            )
            x_full_test = np.concatenate(
                [x_full_nospeed[test_idx], V_test],
                axis=1,
            )
        else:
            x_base_train = x_base_nospeed[train_idx]
            x_base_test = x_base_nospeed[test_idx]
            x_full_train = x_full_nospeed[train_idx]
            x_full_test = x_full_nospeed[test_idx]

        model_base = PopulationGLM(
            "Poisson",
            regularizer="Ridge",
            regularizer_strength=ridge,
        )
        model_full = PopulationGLM(
            "Poisson",
            regularizer="Ridge",
            regularizer_strength=ridge,
        )
        model_base.fit(x_base_train, y_all[train_idx])
        model_full.fit(x_full_train, y_all[train_idx])

        ll_base_sum += np.asarray(
            model_base.score(
                x_base_test,
                y_all[test_idx],
                score_type="log-likelihood",
                aggregate_sample_scores=aggregate_sample_scores,
            )
        )
        ll_full_sum += np.asarray(
            model_full.score(
                x_full_test,
                y_all[test_idx],
                score_type="log-likelihood",
                aggregate_sample_scores=aggregate_sample_scores,
            )
        )
        spike_sum += np.asarray(y_all[test_idx].sum(axis=0), dtype=float)

    d_ll_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spike = np.where(spike_sum > 0, ll_base_sum / spike_sum, np.nan)
        ll_full_per_spike = np.where(spike_sum > 0, ll_full_sum / spike_sum, np.nan)
        d_ll_per_spike = np.where(spike_sum > 0, d_ll_sum / spike_sum, np.nan)

    x_base_all = np.concatenate([x_base_nospeed, V_all], axis=1)
    x_full_all = np.concatenate([x_full_nospeed, V_all], axis=1)

    model_base_all = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=ridge,
    )
    model_full_all = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=ridge,
    )
    model_base_all.fit(x_base_all, y_all)
    model_full_all.fit(x_full_all, y_all)

    coef_base = _coef_feat_by_unit(model_base_all, n_features=x_base_all.shape[1])
    coef_full = _coef_feat_by_unit(model_full_all, n_features=x_full_all.shape[1])
    n_speed_features = int(speed_transform_all["n_features"])
    if n_speed_features > 0:
        coef_speed_basis_base = coef_base[-n_speed_features:, :]
        coef_speed_basis_full = coef_full[-n_speed_features:, :]
    else:
        coef_speed_basis_base = np.full((0, n_units), np.nan)
        coef_speed_basis_full = np.full((0, n_units), np.nan)

    return {
        "coef_base": coef_base,
        "coef_full": coef_full,
        "coef_speed_basis_base": coef_speed_basis_base,
        "coef_speed_basis_full": coef_speed_basis_full,
        "intercept_base": np.asarray(model_base_all.intercept_).reshape(-1),
        "intercept_full": np.asarray(model_full_all.intercept_).reshape(-1),
        "spike_sum_cv": spike_sum,
        "ll_base_sum_cv": ll_base_sum,
        "ll_full_sum_cv": ll_full_sum,
        "dLL_sum_cv": d_ll_sum,
        "ll_base_per_spike_cv": ll_base_per_spike,
        "ll_full_per_spike_cv": ll_full_per_spike,
        "dll_per_spike_cv": d_ll_per_spike,
        "ll_base_bits_per_spike_cv": ll_base_per_spike * inv_log2,
        "ll_full_bits_per_spike_cv": ll_full_per_spike * inv_log2,
        "dll_bits_per_spike_cv": d_ll_per_spike * inv_log2,
        "has_speed": has_speed,
        "speed_transform": speed_transform_all,
    }


def _standard_result_dict(
    *,
    unit_ids: np.ndarray,
    bin_size_s: float,
    has_speed: bool,
    speed_outputs: dict[str, Any],
    pos_bounds: tuple[float, float],
    n_splines: int,
    spline_order: int,
    cv_result: dict[str, Any],
    coef_intercept_base_all: np.ndarray,
    coef_place_base_all: np.ndarray,
    coef_light_base_all: np.ndarray,
    coef_place_x_light_base_all: np.ndarray,
    coef_speed_base_all: np.ndarray,
    coef_intercept_full_all: np.ndarray,
    coef_place_dark_full_all: np.ndarray,
    coef_light_full_all: np.ndarray,
    coef_speed_full_all: np.ndarray,
    coef_add_intercept_full_all: np.ndarray,
    coef_add_place_full_all: np.ndarray,
    grid_tp: np.ndarray,
    dark_hz_grid: np.ndarray,
    light_hz_grid: np.ndarray,
    family_specific_full_coef: dict[str, np.ndarray] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the shared result schema used across all model families."""
    result = {
        "unit_ids": np.asarray(unit_ids),
        "has_speed": bool(has_speed),
        **speed_outputs,
        "bin_size_s": float(bin_size_s),
        "n_splines": int(n_splines),
        "spline_order": int(spline_order),
        "pos_bounds": np.asarray(pos_bounds, dtype=float),
        "spike_sum_cv": np.asarray(cv_result["spike_sum_cv"], dtype=float),
        "ll_base_sum_cv": np.asarray(cv_result["ll_base_sum_cv"], dtype=float),
        "ll_full_sum_cv": np.asarray(cv_result["ll_full_sum_cv"], dtype=float),
        "dLL_sum_cv": np.asarray(cv_result["dLL_sum_cv"], dtype=float),
        "ll_base_per_spike_cv": np.asarray(cv_result["ll_base_per_spike_cv"], dtype=float),
        "ll_full_per_spike_cv": np.asarray(cv_result["ll_full_per_spike_cv"], dtype=float),
        "dll_per_spike_cv": np.asarray(cv_result["dll_per_spike_cv"], dtype=float),
        "ll_base_bits_per_spike_cv": np.asarray(cv_result["ll_base_bits_per_spike_cv"], dtype=float),
        "ll_full_bits_per_spike_cv": np.asarray(cv_result["ll_full_bits_per_spike_cv"], dtype=float),
        "dll_bits_per_spike_cv": np.asarray(cv_result["dll_bits_per_spike_cv"], dtype=float),
        "coef_intercept_base_all": np.asarray(coef_intercept_base_all, dtype=float),
        "coef_place_base_all": np.asarray(coef_place_base_all, dtype=float),
        "coef_light_base_all": np.asarray(coef_light_base_all, dtype=float),
        "coef_place_x_light_base_all": np.asarray(
            coef_place_x_light_base_all, dtype=float
        ),
        "coef_speed_base_all": np.asarray(coef_speed_base_all, dtype=float),
        "coef_intercept_full_all": np.asarray(coef_intercept_full_all, dtype=float),
        "coef_place_dark_full_all": np.asarray(coef_place_dark_full_all, dtype=float),
        "coef_light_full_all": np.asarray(coef_light_full_all, dtype=float),
        "coef_speed_full_all": np.asarray(coef_speed_full_all, dtype=float),
        "coef_add_intercept_full_all": np.asarray(
            coef_add_intercept_full_all, dtype=float
        ),
        "coef_add_place_full_all": np.asarray(coef_add_place_full_all, dtype=float),
        "grid_tp": np.asarray(grid_tp, dtype=float),
        "dark_hz_grid": np.asarray(dark_hz_grid, dtype=float),
        "light_hz_grid": np.asarray(light_hz_grid, dtype=float),
    }
    if family_specific_full_coef:
        result.update(
            {
                name: np.asarray(values, dtype=float)
                for name, values in family_specific_full_coef.items()
            }
        )
    if extra:
        result.update(extra)
    return result


def fit_shared_place_light_mod_nemos_per_traj(
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None = None,
    *,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: dict[str, Any] | None = None,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit the shared-place multiplicative light modulation model."""
    _require_nemos()
    fit_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=light_epochs,
        dark_epochs=dark_epochs,
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch=restrict_ep_by_epoch,
    )
    y_all = fit_inputs["y_all"]
    p_all = fit_inputs["p_all"]
    l_all = fit_inputs["l_all"]
    v_all = fit_inputs["v_all"]

    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    place_basis = np.asarray(basis.compute_features(p_all), dtype=float)
    n_basis = place_basis.shape[1]
    n_units = y_all.shape[1]
    x_base_nospeed = np.concatenate([place_basis, l_all[:, None]], axis=1)
    x_full_nospeed = np.concatenate(
        [place_basis, l_all[:, None], place_basis * l_all[:, None]],
        axis=1,
    )
    cv_result = _fit_exp_poisson_models(
        y_all=y_all,
        l_all=l_all,
        v_all=v_all,
        x_base_nospeed=x_base_nospeed,
        x_full_nospeed=x_full_nospeed,
        ridge=ridge,
        n_folds=n_folds,
        seed=seed,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
    )

    coef0 = cv_result["coef_base"]
    coef1 = cv_result["coef_full"]
    speed_transform_all = cv_result["speed_transform"]
    coef_speed_basis_base = cv_result["coef_speed_basis_base"]
    coef_speed_basis_full = cv_result["coef_speed_basis_full"]
    coef_place_base = coef0[:n_basis]
    coef_light_base = coef0[n_basis]
    coef_place_full = coef1[:n_basis]
    coef_light_full = coef1[n_basis]
    coef_place_x_light_full = coef1[(n_basis + 1) : (n_basis + 1 + n_basis)]
    nan_u = np.full((n_units,), np.nan)
    nan_basis = np.full((n_basis, n_units), np.nan)
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=coef_speed_basis_base,
        coef_speed_basis_full=coef_speed_basis_full,
        n_units=n_units,
    )

    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    grid_basis = np.asarray(basis.compute_features(grid), dtype=float)
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis_full,
        n_units,
    )
    dark_lin = (
        cv_result["intercept_full"][None, :]
        + (grid_basis @ coef_place_full)
        + speed_ref_effect
    )
    light_lin = dark_lin + coef_light_full[None, :] + (
        grid_basis @ coef_place_x_light_full
    )
    return _standard_result_dict(
        unit_ids=fit_inputs["unit_ids"],
        bin_size_s=bin_size_s,
        has_speed=cv_result["has_speed"],
        speed_outputs=speed_outputs,
        pos_bounds=pos_bounds,
        n_splines=n_splines,
        spline_order=spline_order,
        cv_result=cv_result,
        coef_intercept_base_all=cv_result["intercept_base"],
        coef_place_base_all=coef_place_base,
        coef_light_base_all=coef_light_base,
        coef_place_x_light_base_all=nan_basis,
        coef_speed_base_all=speed_outputs["coef_speed_base_all"],
        coef_intercept_full_all=cv_result["intercept_full"],
        coef_place_dark_full_all=coef_place_full,
        coef_light_full_all=coef_light_full,
        coef_speed_full_all=speed_outputs["coef_speed_full_all"],
        coef_add_intercept_full_all=nan_u,
        coef_add_place_full_all=nan_basis,
        grid_tp=grid,
        dark_hz_grid=np.exp(dark_lin) / bin_size_s,
        light_hz_grid=np.exp(light_lin) / bin_size_s,
        family_specific_full_coef={
            "coef_light_gain_spline_full_all": coef_place_x_light_full,
        },
    )


def fit_shared_place_light_mod_nemos_per_traj_segment_gain(
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None = None,
    *,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    segment_edges: Sequence[float],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: dict[str, Any] | None = None,
    gain_overlap_frac: float = 0.0,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit the segment-raised-cosine multiplicative light modulation model."""
    _require_nemos()
    fit_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=light_epochs,
        dark_epochs=dark_epochs,
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch=restrict_ep_by_epoch,
    )
    y_all = fit_inputs["y_all"]
    p_all = fit_inputs["p_all"]
    l_all = fit_inputs["l_all"]
    v_all = fit_inputs["v_all"]

    basis_dark = BSplineEval(
        n_basis_funcs=n_splines,
        order=spline_order,
        bounds=pos_bounds,
    )
    dark_basis = np.asarray(basis_dark.compute_features(p_all), dtype=float)
    gain_edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    gain_basis = _segment_center_raised_cosine_basis(
        p_all,
        gain_edges,
        pos_bounds=pos_bounds,
        overlap_frac=gain_overlap_frac,
    )
    n_dark_basis = dark_basis.shape[1]
    n_gain_basis = gain_basis.shape[1]
    n_units = y_all.shape[1]

    x_base_nospeed = np.concatenate([dark_basis, l_all[:, None]], axis=1)
    x_full_nospeed = np.concatenate(
        [dark_basis, l_all[:, None], gain_basis * l_all[:, None]],
        axis=1,
    )
    cv_result = _fit_exp_poisson_models(
        y_all=y_all,
        l_all=l_all,
        v_all=v_all,
        x_base_nospeed=x_base_nospeed,
        x_full_nospeed=x_full_nospeed,
        ridge=ridge,
        n_folds=n_folds,
        seed=seed,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
    )

    coef0 = cv_result["coef_base"]
    coef1 = cv_result["coef_full"]
    speed_transform_all = cv_result["speed_transform"]
    coef_speed_basis_base = cv_result["coef_speed_basis_base"]
    coef_speed_basis_full = cv_result["coef_speed_basis_full"]
    coef_place_base = coef0[:n_dark_basis]
    coef_light_base = coef0[n_dark_basis]
    coef_place_full = coef1[:n_dark_basis]
    coef_light_full = coef1[n_dark_basis]
    coef_gain_full = coef1[(n_dark_basis + 1) : (n_dark_basis + 1 + n_gain_basis)]
    nan_u = np.full((n_units,), np.nan)
    nan_dark = np.full((n_dark_basis, n_units), np.nan)
    nan_gain = np.full((n_gain_basis, n_units), np.nan)
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=coef_speed_basis_base,
        coef_speed_basis_full=coef_speed_basis_full,
        n_units=n_units,
    )

    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    grid_dark = np.asarray(basis_dark.compute_features(grid), dtype=float)
    grid_gain = _segment_center_raised_cosine_basis(
        grid,
        gain_edges,
        pos_bounds=pos_bounds,
        overlap_frac=gain_overlap_frac,
    )
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis_full,
        n_units,
    )
    dark_lin = (
        cv_result["intercept_full"][None, :]
        + (grid_dark @ coef_place_full)
        + speed_ref_effect
    )
    light_lin = dark_lin + coef_light_full[None, :] + (grid_gain @ coef_gain_full)
    return _standard_result_dict(
        unit_ids=fit_inputs["unit_ids"],
        bin_size_s=bin_size_s,
        has_speed=cv_result["has_speed"],
        speed_outputs=speed_outputs,
        pos_bounds=pos_bounds,
        n_splines=n_splines,
        spline_order=spline_order,
        cv_result=cv_result,
        coef_intercept_base_all=cv_result["intercept_base"],
        coef_place_base_all=coef_place_base,
        coef_light_base_all=coef_light_base,
        coef_place_x_light_base_all=nan_gain,
        coef_speed_base_all=speed_outputs["coef_speed_base_all"],
        coef_intercept_full_all=cv_result["intercept_full"],
        coef_place_dark_full_all=coef_place_full,
        coef_light_full_all=coef_light_full,
        coef_speed_full_all=speed_outputs["coef_speed_full_all"],
        coef_add_intercept_full_all=nan_u,
        coef_add_place_full_all=nan_dark,
        grid_tp=grid,
        dark_hz_grid=np.exp(dark_lin) / bin_size_s,
        light_hz_grid=np.exp(light_lin) / bin_size_s,
        family_specific_full_coef={
            "coef_segment_bump_gain_full_all": coef_gain_full,
        },
        extra={
            "gain_basis": "segment_raised_cosine",
            "segment_edges": gain_edges,
            "n_splines_gain": int(n_gain_basis),
            "gain_overlap_frac": float(gain_overlap_frac),
        },
    )


def fit_shared_place_light_mod_nemos_per_traj_gain_splines(
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None = None,
    *,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    n_splines_gain: int | None = None,
    spline_order_gain: int | None = None,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: dict[str, Any] | None = None,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit the multiplicative light modulation model with a separate gain basis."""
    _require_nemos()
    if n_splines_gain is None:
        n_splines_gain = int(n_splines)
    if spline_order_gain is None:
        spline_order_gain = int(spline_order)

    fit_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=light_epochs,
        dark_epochs=dark_epochs,
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch=restrict_ep_by_epoch,
    )
    y_all = fit_inputs["y_all"]
    p_all = fit_inputs["p_all"]
    l_all = fit_inputs["l_all"]
    v_all = fit_inputs["v_all"]

    basis_dark = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    basis_gain = BSplineEval(
        n_basis_funcs=n_splines_gain,
        order=spline_order_gain,
        bounds=pos_bounds,
    )
    dark_basis = np.asarray(basis_dark.compute_features(p_all), dtype=float)
    gain_basis = np.asarray(basis_gain.compute_features(p_all), dtype=float)
    n_dark_basis = dark_basis.shape[1]
    n_gain_basis = gain_basis.shape[1]
    n_units = y_all.shape[1]

    x_base_nospeed = np.concatenate([dark_basis, l_all[:, None]], axis=1)
    x_full_nospeed = np.concatenate(
        [dark_basis, l_all[:, None], gain_basis * l_all[:, None]],
        axis=1,
    )
    cv_result = _fit_exp_poisson_models(
        y_all=y_all,
        l_all=l_all,
        v_all=v_all,
        x_base_nospeed=x_base_nospeed,
        x_full_nospeed=x_full_nospeed,
        ridge=ridge,
        n_folds=n_folds,
        seed=seed,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
    )

    coef0 = cv_result["coef_base"]
    coef1 = cv_result["coef_full"]
    speed_transform_all = cv_result["speed_transform"]
    coef_speed_basis_base = cv_result["coef_speed_basis_base"]
    coef_speed_basis_full = cv_result["coef_speed_basis_full"]
    coef_place_base = coef0[:n_dark_basis]
    coef_light_base = coef0[n_dark_basis]
    coef_place_full = coef1[:n_dark_basis]
    coef_light_full = coef1[n_dark_basis]
    coef_gain_full = coef1[(n_dark_basis + 1) : (n_dark_basis + 1 + n_gain_basis)]
    nan_u = np.full((n_units,), np.nan)
    nan_dark = np.full((n_dark_basis, n_units), np.nan)
    nan_gain = np.full((n_gain_basis, n_units), np.nan)
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=coef_speed_basis_base,
        coef_speed_basis_full=coef_speed_basis_full,
        n_units=n_units,
    )

    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    grid_dark = np.asarray(basis_dark.compute_features(grid), dtype=float)
    grid_gain = np.asarray(basis_gain.compute_features(grid), dtype=float)
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis_full,
        n_units,
    )
    dark_lin = (
        cv_result["intercept_full"][None, :]
        + (grid_dark @ coef_place_full)
        + speed_ref_effect
    )
    light_lin = dark_lin + coef_light_full[None, :] + (grid_gain @ coef_gain_full)
    return _standard_result_dict(
        unit_ids=fit_inputs["unit_ids"],
        bin_size_s=bin_size_s,
        has_speed=cv_result["has_speed"],
        speed_outputs=speed_outputs,
        pos_bounds=pos_bounds,
        n_splines=n_splines,
        spline_order=spline_order,
        cv_result=cv_result,
        coef_intercept_base_all=cv_result["intercept_base"],
        coef_place_base_all=coef_place_base,
        coef_light_base_all=coef_light_base,
        coef_place_x_light_base_all=nan_gain,
        coef_speed_base_all=speed_outputs["coef_speed_base_all"],
        coef_intercept_full_all=cv_result["intercept_full"],
        coef_place_dark_full_all=coef_place_full,
        coef_light_full_all=coef_light_full,
        coef_speed_full_all=speed_outputs["coef_speed_full_all"],
        coef_add_intercept_full_all=nan_u,
        coef_add_place_full_all=nan_dark,
        grid_tp=grid,
        dark_hz_grid=np.exp(dark_lin) / bin_size_s,
        light_hz_grid=np.exp(light_lin) / bin_size_s,
        family_specific_full_coef={
            "coef_light_gain_spline_full_all": coef_gain_full,
        },
        extra={
            "gain_basis": "cyclic_bspline",
            "n_splines_gain": int(n_splines_gain),
            "spline_order_gain": int(spline_order_gain),
        },
    )


def fit_shared_place_light_mod_nemos_per_traj_segment_scalar_gain(
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None = None,
    *,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    segment_edges: Sequence[float],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: dict[str, Any] | None = None,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit the piecewise-constant segment gain model."""
    _require_nemos()
    fit_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=light_epochs,
        dark_epochs=dark_epochs,
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch=restrict_ep_by_epoch,
    )
    y_all = fit_inputs["y_all"]
    p_all = fit_inputs["p_all"]
    l_all = fit_inputs["l_all"]
    v_all = fit_inputs["v_all"]

    basis_dark = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    dark_basis = np.asarray(basis_dark.compute_features(p_all), dtype=float)
    segment_basis, gain_edges = _segment_onehot_basis(
        p_all,
        segment_edges,
        pos_bounds=pos_bounds,
        drop_first=False,
    )
    n_dark_basis = dark_basis.shape[1]
    n_gain_basis = segment_basis.shape[1]
    n_units = y_all.shape[1]

    x_base_nospeed = np.concatenate([dark_basis, l_all[:, None]], axis=1)
    x_full_nospeed = np.concatenate(
        [dark_basis, l_all[:, None], segment_basis * l_all[:, None]],
        axis=1,
    )
    cv_result = _fit_exp_poisson_models(
        y_all=y_all,
        l_all=l_all,
        v_all=v_all,
        x_base_nospeed=x_base_nospeed,
        x_full_nospeed=x_full_nospeed,
        ridge=ridge,
        n_folds=n_folds,
        seed=seed,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
    )

    coef0 = cv_result["coef_base"]
    coef1 = cv_result["coef_full"]
    speed_transform_all = cv_result["speed_transform"]
    coef_speed_basis_base = cv_result["coef_speed_basis_base"]
    coef_speed_basis_full = cv_result["coef_speed_basis_full"]
    coef_place_base = coef0[:n_dark_basis]
    coef_light_base = coef0[n_dark_basis]
    coef_place_full = coef1[:n_dark_basis]
    coef_light_full = coef1[n_dark_basis]
    coef_gain_full = coef1[(n_dark_basis + 1) : (n_dark_basis + 1 + n_gain_basis)]
    nan_u = np.full((n_units,), np.nan)
    nan_dark = np.full((n_dark_basis, n_units), np.nan)
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=coef_speed_basis_base,
        coef_speed_basis_full=coef_speed_basis_full,
        n_units=n_units,
    )

    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    grid_dark = np.asarray(basis_dark.compute_features(grid), dtype=float)
    grid_segment, _ = _segment_onehot_basis(
        grid,
        gain_edges,
        pos_bounds=pos_bounds,
        drop_first=False,
    )
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis_full,
        n_units,
    )
    dark_lin = (
        cv_result["intercept_full"][None, :]
        + (grid_dark @ coef_place_full)
        + speed_ref_effect
    )
    light_lin = dark_lin + coef_light_full[None, :] + (grid_segment @ coef_gain_full)
    return _standard_result_dict(
        unit_ids=fit_inputs["unit_ids"],
        bin_size_s=bin_size_s,
        has_speed=cv_result["has_speed"],
        speed_outputs=speed_outputs,
        pos_bounds=pos_bounds,
        n_splines=n_splines,
        spline_order=spline_order,
        cv_result=cv_result,
        coef_intercept_base_all=cv_result["intercept_base"],
        coef_place_base_all=coef_place_base,
        coef_light_base_all=coef_light_base,
        coef_place_x_light_base_all=nan_dark,
        coef_speed_base_all=speed_outputs["coef_speed_base_all"],
        coef_intercept_full_all=cv_result["intercept_full"],
        coef_place_dark_full_all=coef_place_full,
        coef_light_full_all=coef_light_full,
        coef_speed_full_all=speed_outputs["coef_speed_full_all"],
        coef_add_intercept_full_all=nan_u,
        coef_add_place_full_all=nan_dark,
        grid_tp=grid,
        dark_hz_grid=np.exp(dark_lin) / bin_size_s,
        light_hz_grid=np.exp(light_lin) / bin_size_s,
        family_specific_full_coef={
            "coef_segment_scalar_gain_full_all": coef_gain_full,
        },
        extra={
            "gain_basis": "segment_scalar",
            "segment_edges": gain_edges,
            "n_splines_gain": int(n_gain_basis),
        },
    )


@_register_pytree_node_class
@dataclass
class _AddRateParamsBatched:
    theta0: jnp.ndarray
    beta: jnp.ndarray
    beta_speed: jnp.ndarray
    phi0: jnp.ndarray
    delta: jnp.ndarray

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], Any]:
        return (self.theta0, self.beta, self.beta_speed, self.phi0, self.delta), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Any,
        children: tuple[jnp.ndarray, ...],
    ) -> "_AddRateParamsBatched":
        theta0, beta, beta_speed, phi0, delta = children
        return cls(theta0=theta0, beta=beta, beta_speed=beta_speed, phi0=phi0, delta=delta)


@_register_pytree_node_class
@dataclass
class _AddRateParamsBatchedNoSpeed:
    theta0: jnp.ndarray
    beta: jnp.ndarray
    phi0: jnp.ndarray
    delta: jnp.ndarray

    def tree_flatten(self) -> tuple[tuple[jnp.ndarray, ...], Any]:
        return (self.theta0, self.beta, self.phi0, self.delta), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: Any,
        children: tuple[jnp.ndarray, ...],
    ) -> "_AddRateParamsBatchedNoSpeed":
        theta0, beta, phi0, delta = children
        return cls(theta0=theta0, beta=beta, phi0=phi0, delta=delta)


def _init_additive_from_base(
    *,
    y: np.ndarray,
    basis: np.ndarray,
    speed: np.ndarray | None,
    light: np.ndarray,
    ridge: float,
) -> _AddRateParamsBatched | _AddRateParamsBatchedNoSpeed:
    """Warm-start additive fits from the base exp-link Poisson solution."""
    _require_nemos()
    n_time, n_basis = basis.shape
    n_units = y.shape[1]
    del n_time

    if speed is not None and speed.shape[1] > 0:
        n_speed_features = int(speed.shape[1])
        x_base = np.concatenate([basis, speed], axis=1)
        model = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        model.fit(x_base, y)
        coef = _coef_feat_by_unit(model, n_features=n_basis + n_speed_features)
        theta0 = np.asarray(model.intercept_).reshape(-1).astype(np.float32)
        beta = coef[:n_basis].astype(np.float32)
        beta_speed = coef[n_basis : (n_basis + n_speed_features)].astype(np.float32)
    else:
        model = PopulationGLM("Poisson", regularizer="Ridge", regularizer_strength=ridge)
        model.fit(basis, y)
        coef = _coef_feat_by_unit(model, n_features=n_basis)
        theta0 = np.asarray(model.intercept_).reshape(-1).astype(np.float32)
        beta = coef.astype(np.float32)
        beta_speed = None

    eps = 1e-6
    light_mask = light.astype(bool)
    mean_light = y[light_mask].mean(axis=0) if np.any(light_mask) else y.mean(axis=0)
    mean_dark = y[~light_mask].mean(axis=0) if np.any(~light_mask) else y.mean(axis=0)
    add_mu = np.maximum(mean_light - mean_dark, eps)
    phi0 = np.log(add_mu + eps).astype(np.float32)
    delta = np.zeros((n_basis, n_units), dtype=np.float32)

    if beta_speed is not None:
        return _AddRateParamsBatched(
            theta0=jnp.asarray(theta0),
            beta=jnp.asarray(beta),
            beta_speed=jnp.asarray(beta_speed),
            phi0=jnp.asarray(phi0),
            delta=jnp.asarray(delta),
        )
    return _AddRateParamsBatchedNoSpeed(
        theta0=jnp.asarray(theta0),
        beta=jnp.asarray(beta),
        phi0=jnp.asarray(phi0),
        delta=jnp.asarray(delta),
    )


def _try_import_jaxopt() -> Any | None:
    """Import `jaxopt` if it is available in the current environment."""
    if jax is None:
        return None
    try:
        import jaxopt  # type: ignore
    except Exception:
        return None
    return jaxopt


def _fit_additive_rate_batched_jax(
    *,
    y: np.ndarray,
    basis: np.ndarray,
    speed: np.ndarray | None,
    light: np.ndarray,
    ridge: float,
    maxiter: int = 200,
    lr: float = 5e-2,
) -> _AddRateParamsBatched | _AddRateParamsBatchedNoSpeed:
    """Fit the additive-in-rate light component with JAX and optional JAXopt."""
    if jax is None:
        raise ModuleNotFoundError(
            "The `add_jax` model family requires `jax`, but it is not installed in this environment."
        )

    y_j = jnp.asarray(y, dtype=jnp.float32)
    basis_j = jnp.asarray(basis, dtype=jnp.float32)
    light_j = jnp.asarray(light, dtype=jnp.float32).reshape(-1)

    idx_light = jnp.where(light_j > 0.5)[0]
    idx_dark = jnp.where(light_j <= 0.5)[0]

    if speed is not None and speed.shape[1] > 0:
        speed_j = jnp.asarray(speed, dtype=jnp.float32)

        def nll(params: _AddRateParamsBatched) -> jnp.ndarray:
            eta = (
                params.theta0[None, :]
                + (basis_j @ params.beta)
                + (speed_j @ params.beta_speed)
            )
            z_add = params.phi0[None, :] + (basis_j @ params.delta)
            eta_dark = eta[idx_dark]
            y_dark = y_j[idx_dark]
            ll_dark = jnp.sum(y_dark * eta_dark - jnp.exp(eta_dark))

            eta_light = eta[idx_light]
            z_light = z_add[idx_light]
            y_light = y_j[idx_light]
            log_rate_light = jnp.logaddexp(eta_light, z_light)
            rate_light = jnp.exp(eta_light) + jnp.exp(z_light)
            ll_light = jnp.sum(y_light * log_rate_light - rate_light)

            penalty = 0.5 * ridge * (
                jnp.sum(params.beta**2)
                + jnp.sum(params.delta**2)
                + jnp.sum(params.beta_speed**2)
            )
            return -((ll_dark + ll_light) / y_j.shape[0]) + penalty

        init = _init_additive_from_base(
            y=y,
            basis=basis,
            speed=speed,
            light=light,
            ridge=ridge,
        )
        nll_jit = jax.jit(nll)
        jaxopt = _try_import_jaxopt()
        if jaxopt is not None:
            solver = jaxopt.LBFGS(fun=nll_jit, maxiter=int(maxiter))
            params_hat, _ = solver.run(init)
            return params_hat

        grad_jit = jax.jit(jax.grad(nll))
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        def zeros_like(p: _AddRateParamsBatched) -> _AddRateParamsBatched:
            return _AddRateParamsBatched(
                theta0=jnp.zeros_like(p.theta0),
                beta=jnp.zeros_like(p.beta),
                beta_speed=jnp.zeros_like(p.beta_speed),
                phi0=jnp.zeros_like(p.phi0),
                delta=jnp.zeros_like(p.delta),
            )

        params = init
        first_moment = zeros_like(init)
        second_moment = zeros_like(init)
        for step in range(int(maxiter)):
            grad = grad_jit(params)
            first_moment = _AddRateParamsBatched(
                theta0=beta1 * first_moment.theta0 + (1 - beta1) * grad.theta0,
                beta=beta1 * first_moment.beta + (1 - beta1) * grad.beta,
                beta_speed=beta1 * first_moment.beta_speed + (1 - beta1) * grad.beta_speed,
                phi0=beta1 * first_moment.phi0 + (1 - beta1) * grad.phi0,
                delta=beta1 * first_moment.delta + (1 - beta1) * grad.delta,
            )
            second_moment = _AddRateParamsBatched(
                theta0=beta2 * second_moment.theta0 + (1 - beta2) * (grad.theta0**2),
                beta=beta2 * second_moment.beta + (1 - beta2) * (grad.beta**2),
                beta_speed=beta2 * second_moment.beta_speed
                + (1 - beta2) * (grad.beta_speed**2),
                phi0=beta2 * second_moment.phi0 + (1 - beta2) * (grad.phi0**2),
                delta=beta2 * second_moment.delta + (1 - beta2) * (grad.delta**2),
            )
            step_num = step + 1
            m_hat = _AddRateParamsBatched(
                theta0=first_moment.theta0 / (1 - beta1**step_num),
                beta=first_moment.beta / (1 - beta1**step_num),
                beta_speed=first_moment.beta_speed / (1 - beta1**step_num),
                phi0=first_moment.phi0 / (1 - beta1**step_num),
                delta=first_moment.delta / (1 - beta1**step_num),
            )
            v_hat = _AddRateParamsBatched(
                theta0=second_moment.theta0 / (1 - beta2**step_num),
                beta=second_moment.beta / (1 - beta2**step_num),
                beta_speed=second_moment.beta_speed / (1 - beta2**step_num),
                phi0=second_moment.phi0 / (1 - beta2**step_num),
                delta=second_moment.delta / (1 - beta2**step_num),
            )
            params = _AddRateParamsBatched(
                theta0=params.theta0 - lr * m_hat.theta0 / (jnp.sqrt(v_hat.theta0) + eps),
                beta=params.beta - lr * m_hat.beta / (jnp.sqrt(v_hat.beta) + eps),
                beta_speed=params.beta_speed
                - lr * m_hat.beta_speed / (jnp.sqrt(v_hat.beta_speed) + eps),
                phi0=params.phi0 - lr * m_hat.phi0 / (jnp.sqrt(v_hat.phi0) + eps),
                delta=params.delta - lr * m_hat.delta / (jnp.sqrt(v_hat.delta) + eps),
            )
        return params

    def nll_no_speed(params: _AddRateParamsBatchedNoSpeed) -> jnp.ndarray:
        eta = params.theta0[None, :] + (basis_j @ params.beta)
        z_add = params.phi0[None, :] + (basis_j @ params.delta)
        eta_dark = eta[idx_dark]
        y_dark = y_j[idx_dark]
        ll_dark = jnp.sum(y_dark * eta_dark - jnp.exp(eta_dark))

        eta_light = eta[idx_light]
        z_light = z_add[idx_light]
        y_light = y_j[idx_light]
        log_rate_light = jnp.logaddexp(eta_light, z_light)
        rate_light = jnp.exp(eta_light) + jnp.exp(z_light)
        ll_light = jnp.sum(y_light * log_rate_light - rate_light)
        penalty = 0.5 * ridge * (jnp.sum(params.beta**2) + jnp.sum(params.delta**2))
        return -((ll_dark + ll_light) / y_j.shape[0]) + penalty

    init_no_speed = _init_additive_from_base(
        y=y,
        basis=basis,
        speed=None,
        light=light,
        ridge=ridge,
    )
    nll_no_speed_jit = jax.jit(nll_no_speed)
    jaxopt = _try_import_jaxopt()
    if jaxopt is not None:
        solver = jaxopt.LBFGS(fun=nll_no_speed_jit, maxiter=int(maxiter))
        params_hat, _ = solver.run(init_no_speed)
        return params_hat

    grad_jit = jax.jit(jax.grad(nll_no_speed))
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    def zeros_like_no_speed(
        p: _AddRateParamsBatchedNoSpeed,
    ) -> _AddRateParamsBatchedNoSpeed:
        return _AddRateParamsBatchedNoSpeed(
            theta0=jnp.zeros_like(p.theta0),
            beta=jnp.zeros_like(p.beta),
            phi0=jnp.zeros_like(p.phi0),
            delta=jnp.zeros_like(p.delta),
        )

    params = init_no_speed
    first_moment = zeros_like_no_speed(init_no_speed)
    second_moment = zeros_like_no_speed(init_no_speed)
    for step in range(int(maxiter)):
        grad = grad_jit(params)
        first_moment = _AddRateParamsBatchedNoSpeed(
            theta0=beta1 * first_moment.theta0 + (1 - beta1) * grad.theta0,
            beta=beta1 * first_moment.beta + (1 - beta1) * grad.beta,
            phi0=beta1 * first_moment.phi0 + (1 - beta1) * grad.phi0,
            delta=beta1 * first_moment.delta + (1 - beta1) * grad.delta,
        )
        second_moment = _AddRateParamsBatchedNoSpeed(
            theta0=beta2 * second_moment.theta0 + (1 - beta2) * (grad.theta0**2),
            beta=beta2 * second_moment.beta + (1 - beta2) * (grad.beta**2),
            phi0=beta2 * second_moment.phi0 + (1 - beta2) * (grad.phi0**2),
            delta=beta2 * second_moment.delta + (1 - beta2) * (grad.delta**2),
        )
        step_num = step + 1
        m_hat = _AddRateParamsBatchedNoSpeed(
            theta0=first_moment.theta0 / (1 - beta1**step_num),
            beta=first_moment.beta / (1 - beta1**step_num),
            phi0=first_moment.phi0 / (1 - beta1**step_num),
            delta=first_moment.delta / (1 - beta1**step_num),
        )
        v_hat = _AddRateParamsBatchedNoSpeed(
            theta0=second_moment.theta0 / (1 - beta2**step_num),
            beta=second_moment.beta / (1 - beta2**step_num),
            phi0=second_moment.phi0 / (1 - beta2**step_num),
            delta=second_moment.delta / (1 - beta2**step_num),
        )
        params = _AddRateParamsBatchedNoSpeed(
            theta0=params.theta0 - lr * m_hat.theta0 / (jnp.sqrt(v_hat.theta0) + eps),
            beta=params.beta - lr * m_hat.beta / (jnp.sqrt(v_hat.beta) + eps),
            phi0=params.phi0 - lr * m_hat.phi0 / (jnp.sqrt(v_hat.phi0) + eps),
            delta=params.delta - lr * m_hat.delta / (jnp.sqrt(v_hat.delta) + eps),
        )
    return params


def fit_true_additive_rate_poisson_fast_per_traj(
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None = None,
    *,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: dict[str, Any] | None = None,
    maxiter_full: int = 200,
    lr_full: float = 5e-2,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit the additive-in-rate light model."""
    _require_nemos()
    _require_scipy()
    fit_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=light_epochs,
        dark_epochs=dark_epochs,
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch=restrict_ep_by_epoch,
    )
    y_all = np.asarray(fit_inputs["y_all"], dtype=np.float32)
    p_all = np.asarray(fit_inputs["p_all"], dtype=np.float32)
    l_all = np.asarray(fit_inputs["l_all"], dtype=np.float32)
    v_all = None if fit_inputs["v_all"] is None else np.asarray(fit_inputs["v_all"], dtype=np.float32)
    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    basis_all = np.asarray(basis.compute_features(p_all), dtype=np.float32)
    n_basis = basis_all.shape[1]
    n_units = y_all.shape[1]
    if v_all is not None:
        V_all, speed_transform_all = _make_speed_design_all(
            v_all,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
            eps=1e-6,
        )
        V_all = np.asarray(V_all, dtype=np.float32)
    else:
        V_all = _empty_speed_design(y_all.shape[0]).astype(np.float32)
        speed_transform_all = _empty_speed_feature_transform()
    n_speed_features = int(speed_transform_all["n_features"])

    folds = _contiguous_folds(l_all, n_folds=n_folds, seed=seed)
    ll_base_sum = np.zeros(n_units, dtype=np.float64)
    ll_full_sum = np.zeros(n_units, dtype=np.float64)
    spike_sum = np.zeros(n_units, dtype=np.float64)

    for train_idx, test_idx in folds:
        basis_train = basis_all[train_idx]
        basis_test = basis_all[test_idx]
        light_train = l_all[train_idx]
        light_test = l_all[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        if v_all is not None:
            V_train, V_test, _ = _make_speed_design_train_test(
                v_all,
                train_idx,
                test_idx,
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
                eps=1e-6,
            )
            V_train = np.asarray(V_train, dtype=np.float32)
            V_test = np.asarray(V_test, dtype=np.float32)
            x_base_train = np.concatenate([basis_train, V_train], axis=1)
            model_base = PopulationGLM(
                "Poisson",
                regularizer="Ridge",
                regularizer_strength=ridge,
            )
            model_base.fit(x_base_train, y_train)
            coef_base = _coef_feat_by_unit(
                model_base,
                n_features=n_basis + n_speed_features,
            )
            theta0_base = np.asarray(model_base.intercept_).reshape(-1).astype(np.float32)
            beta_base = coef_base[:n_basis].astype(np.float32)
            beta_speed_base = coef_base[n_basis : (n_basis + n_speed_features)].astype(
                np.float32
            )
            eta_base_test = theta0_base[None, :] + (basis_test @ beta_base)
            if n_speed_features > 0:
                eta_base_test = eta_base_test + (V_test @ beta_speed_base)
        else:
            model_base = PopulationGLM(
                "Poisson",
                regularizer="Ridge",
                regularizer_strength=ridge,
            )
            model_base.fit(basis_train, y_train)
            coef_base = _coef_feat_by_unit(model_base, n_features=n_basis)
            theta0_base = np.asarray(model_base.intercept_).reshape(-1).astype(np.float32)
            beta_base = coef_base.astype(np.float32)
            eta_base_test = theta0_base[None, :] + (basis_test @ beta_base)
            V_train = None
            V_test = None

        ll_base = np.sum(y_test * eta_base_test - np.exp(eta_base_test), axis=0)
        ll_norm = np.sum(scipy.special.gammaln(y_test + 1.0), axis=0)
        ll_base -= ll_norm

        pars = _fit_additive_rate_batched_jax(
            y=np.asarray(y_train, dtype=np.float32),
            basis=np.asarray(basis_train, dtype=np.float32),
            speed=None if V_train is None else np.asarray(V_train, dtype=np.float32),
            light=np.asarray(light_train, dtype=np.float32),
            ridge=ridge,
            maxiter=maxiter_full,
            lr=lr_full,
        )

        if V_test is not None and n_speed_features > 0:
            eta_dark = (
                np.asarray(pars.theta0)[None, :]
                + (basis_test @ np.asarray(pars.beta))
                + (V_test @ np.asarray(pars.beta_speed))
            )
            z_add = np.asarray(pars.phi0)[None, :] + (basis_test @ np.asarray(pars.delta))
        else:
            eta_dark = np.asarray(pars.theta0)[None, :] + (basis_test @ np.asarray(pars.beta))
            z_add = np.asarray(pars.phi0)[None, :] + (basis_test @ np.asarray(pars.delta))

        is_light = light_test > 0.5
        ll_full = np.zeros(n_units, dtype=np.float64)
        if np.any(~is_light):
            eta_dark_test = eta_dark[~is_light]
            y_dark_test = y_test[~is_light]
            ll_full += np.sum(y_dark_test * eta_dark_test - np.exp(eta_dark_test), axis=0)
        if np.any(is_light):
            eta_light = eta_dark[is_light]
            z_light = z_add[is_light]
            y_light = y_test[is_light]
            log_rate = np.logaddexp(eta_light, z_light)
            rate = np.exp(eta_light) + np.exp(z_light)
            ll_full += np.sum(y_light * log_rate - rate, axis=0)
        ll_full -= ll_norm

        ll_base_sum += ll_base
        ll_full_sum += ll_full
        spike_sum += np.sum(y_test, axis=0)

    d_ll_sum = ll_full_sum - ll_base_sum
    inv_log2 = 1.0 / np.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll_base_per_spike = np.where(spike_sum > 0, ll_base_sum / spike_sum, np.nan)
        ll_full_per_spike = np.where(spike_sum > 0, ll_full_sum / spike_sum, np.nan)
        d_ll_per_spike = np.where(spike_sum > 0, d_ll_sum / spike_sum, np.nan)

    x_base_all = np.concatenate([basis_all, V_all], axis=1)
    model_base_all = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=ridge,
    )
    model_base_all.fit(x_base_all, y_all)
    coef0 = _coef_feat_by_unit(model_base_all, n_features=n_basis + n_speed_features)
    intercept0 = np.asarray(model_base_all.intercept_).reshape(-1)
    coef_place_base = coef0[:n_basis]
    coef_speed_basis_base = coef0[n_basis : (n_basis + n_speed_features)]

    pars_all = _fit_additive_rate_batched_jax(
        y=np.asarray(y_all, dtype=np.float32),
        basis=np.asarray(basis_all, dtype=np.float32),
        speed=None if n_speed_features == 0 else np.asarray(V_all, dtype=np.float32),
        light=np.asarray(l_all, dtype=np.float32),
        ridge=ridge,
        maxiter=maxiter_full,
        lr=lr_full,
    )

    theta0_all = np.asarray(pars_all.theta0)
    beta_all = np.asarray(pars_all.beta)
    phi0_all = np.asarray(pars_all.phi0)
    delta_all = np.asarray(pars_all.delta)
    if n_speed_features > 0:
        coef_speed_basis_full = np.asarray(pars_all.beta_speed)
    else:
        coef_speed_basis_full = np.full((0, n_units), np.nan)
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=coef_speed_basis_base,
        coef_speed_basis_full=coef_speed_basis_full,
        n_units=n_units,
    )

    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200).astype(np.float32)
    grid_basis = np.asarray(basis.compute_features(grid), dtype=np.float32)
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis_full,
        n_units,
    )
    dark_bin = np.exp(theta0_all[None, :] + (grid_basis @ beta_all) + speed_ref_effect)
    add_bin = np.exp(phi0_all[None, :] + (grid_basis @ delta_all))
    nan_u = np.full((n_units,), np.nan)
    nan_basis = np.full((n_basis, n_units), np.nan)
    return _standard_result_dict(
        unit_ids=fit_inputs["unit_ids"],
        bin_size_s=bin_size_s,
        has_speed=v_all is not None,
        speed_outputs=speed_outputs,
        pos_bounds=pos_bounds,
        n_splines=n_splines,
        spline_order=spline_order,
        cv_result={
            "spike_sum_cv": spike_sum,
            "ll_base_sum_cv": ll_base_sum,
            "ll_full_sum_cv": ll_full_sum,
            "dLL_sum_cv": d_ll_sum,
            "ll_base_per_spike_cv": ll_base_per_spike,
            "ll_full_per_spike_cv": ll_full_per_spike,
            "dll_per_spike_cv": d_ll_per_spike,
            "ll_base_bits_per_spike_cv": ll_base_per_spike * inv_log2,
            "ll_full_bits_per_spike_cv": ll_full_per_spike * inv_log2,
            "dll_bits_per_spike_cv": d_ll_per_spike * inv_log2,
        },
        coef_intercept_base_all=intercept0,
        coef_place_base_all=coef_place_base,
        coef_light_base_all=nan_u,
        coef_place_x_light_base_all=nan_basis,
        coef_speed_base_all=speed_outputs["coef_speed_base_all"],
        coef_intercept_full_all=theta0_all,
        coef_place_dark_full_all=beta_all,
        coef_light_full_all=nan_u,
        coef_speed_full_all=speed_outputs["coef_speed_full_all"],
        coef_add_intercept_full_all=phi0_all,
        coef_add_place_full_all=delta_all,
        grid_tp=np.asarray(grid),
        dark_hz_grid=dark_bin / bin_size_s,
        light_hz_grid=(dark_bin + add_bin) / bin_size_s,
        extra={
            "maxiter_full": int(maxiter_full),
            "lr_full": float(lr_full),
        },
    )


def fit_separate_dark_light_fields_nemos_per_traj(
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None = None,
    *,
    traj_name: str,
    light_epochs: Sequence[str],
    dark_epochs: Sequence[str],
    bin_size_s: float = 0.02,
    n_splines: int = 25,
    spline_order: int = 4,
    pos_bounds: tuple[float, float] = (0.0, 1.0),
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
    restrict_ep_by_epoch: dict[str, Any] | None = None,
    speed_feature_mode: str = "linear",
    n_splines_speed: int = 5,
    spline_order_speed: int = 4,
    speed_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Fit separate dark and light place fields."""
    _require_nemos()
    fit_inputs = _prepare_dark_light_fit_inputs(
        spikes=spikes,
        trajectory_ep_by_epoch=trajectory_ep_by_epoch,
        tp_by_epoch=tp_by_epoch,
        speed_by_epoch=speed_by_epoch,
        traj_name=traj_name,
        light_epochs=light_epochs,
        dark_epochs=dark_epochs,
        bin_size_s=bin_size_s,
        unit_mask=unit_mask,
        restrict_ep_by_epoch=restrict_ep_by_epoch,
    )
    y_all = fit_inputs["y_all"]
    p_all = fit_inputs["p_all"]
    l_all = fit_inputs["l_all"]
    v_all = fit_inputs["v_all"]

    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=pos_bounds)
    place_basis = np.asarray(basis.compute_features(p_all), dtype=float)
    n_basis = place_basis.shape[1]
    n_units = y_all.shape[1]
    basis_dark = place_basis * (1.0 - l_all[:, None])
    basis_light = place_basis * l_all[:, None]
    x_base_nospeed = np.concatenate([place_basis, l_all[:, None]], axis=1)
    x_full_nospeed = np.concatenate([basis_dark, basis_light, l_all[:, None]], axis=1)
    cv_result = _fit_exp_poisson_models(
        y_all=y_all,
        l_all=l_all,
        v_all=v_all,
        x_base_nospeed=x_base_nospeed,
        x_full_nospeed=x_full_nospeed,
        ridge=ridge,
        n_folds=n_folds,
        seed=seed,
        speed_feature_mode=speed_feature_mode,
        n_splines_speed=n_splines_speed,
        spline_order_speed=spline_order_speed,
        speed_bounds=speed_bounds,
    )

    coef0 = cv_result["coef_base"]
    coef1 = cv_result["coef_full"]
    speed_transform_all = cv_result["speed_transform"]
    coef_speed_basis_base = cv_result["coef_speed_basis_base"]
    coef_speed_basis_full = cv_result["coef_speed_basis_full"]
    coef_place_base = coef0[:n_basis]
    coef_light_base = coef0[n_basis]
    coef_place_dark = coef1[:n_basis]
    coef_place_light = coef1[n_basis : (2 * n_basis)]
    coef_light_full = coef1[2 * n_basis]
    coef_place_change = coef_place_light - coef_place_dark
    nan_u = np.full((n_units,), np.nan)
    nan_basis = np.full((n_basis, n_units), np.nan)
    speed_outputs = _format_speed_outputs(
        transform=speed_transform_all,
        coef_speed_basis_base=coef_speed_basis_base,
        coef_speed_basis_full=coef_speed_basis_full,
        n_units=n_units,
    )

    grid = np.linspace(pos_bounds[0], pos_bounds[1], 200)
    grid_basis = np.asarray(basis.compute_features(grid), dtype=float)
    speed_ref_effect = _speed_reference_effect(
        speed_transform_all,
        coef_speed_basis_full,
        n_units,
    )
    dark_lin = (
        cv_result["intercept_full"][None, :]
        + (grid_basis @ coef_place_dark)
        + speed_ref_effect
    )
    light_lin = (
        cv_result["intercept_full"][None, :]
        + coef_light_full[None, :]
        + (grid_basis @ coef_place_light)
        + speed_ref_effect
    )
    return _standard_result_dict(
        unit_ids=fit_inputs["unit_ids"],
        bin_size_s=bin_size_s,
        has_speed=cv_result["has_speed"],
        speed_outputs=speed_outputs,
        pos_bounds=pos_bounds,
        n_splines=n_splines,
        spline_order=spline_order,
        cv_result=cv_result,
        coef_intercept_base_all=cv_result["intercept_base"],
        coef_place_base_all=coef_place_base,
        coef_light_base_all=coef_light_base,
        coef_place_x_light_base_all=nan_basis,
        coef_speed_base_all=speed_outputs["coef_speed_base_all"],
        coef_intercept_full_all=cv_result["intercept_full"],
        coef_place_dark_full_all=coef_place_dark,
        coef_light_full_all=coef_light_full,
        coef_speed_full_all=speed_outputs["coef_speed_full_all"],
        coef_add_intercept_full_all=nan_u,
        coef_add_place_full_all=nan_basis,
        grid_tp=grid,
        dark_hz_grid=np.exp(dark_lin) / bin_size_s,
        light_hz_grid=np.exp(light_lin) / bin_size_s,
        family_specific_full_coef={
            "coef_place_light_full_all": coef_place_light,
        },
    )


def _first_result(results_by_traj: dict[str, dict[float, dict[str, Any]]]) -> dict[str, Any]:
    """Return the first fit result from a `{trajectory: {ridge: result}}` mapping."""
    for trajectory in TRAJECTORY_TYPES:
        by_ridge = results_by_traj[trajectory]
        if by_ridge:
            return next(iter(by_ridge.values()))
    raise ValueError("No fit results were available to build a dataset.")


def _stack_family_array(
    results_by_traj: dict[str, dict[float, dict[str, Any]]],
    ridge_values: list[float],
    key: str,
) -> np.ndarray:
    """Stack one result field into `(trajectory, ridge, ...)` order."""
    return np.stack(
        [
            np.stack(
                [
                    np.asarray(results_by_traj[trajectory][ridge][key])
                    for ridge in ridge_values
                ],
                axis=0,
            )
            for trajectory in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def build_family_dataset(
    *,
    family_name: str,
    results_by_traj: dict[str, dict[float, dict[str, Any]]],
    ridge_values: list[float],
    animal_name: str,
    date: str,
    region: str,
    light_epoch: str,
    dark_epoch: str,
    dark_movement_firing_rates: np.ndarray,
    min_dark_firing_rate_hz: float,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build one family-level `xarray.Dataset` across trajectories and ridges."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task-progression dark/light fits as NetCDF."
        ) from exc

    first = _first_result(results_by_traj)
    unit_ids = np.asarray(first["unit_ids"])
    tp_grid = np.asarray(first["grid_tp"], dtype=float)
    place_basis_count = int(np.asarray(first["coef_place_base_all"]).shape[0])
    light_mod_basis_count = int(np.asarray(first["coef_place_x_light_base_all"]).shape[0])
    add_basis_count = int(np.asarray(first["coef_add_place_full_all"]).shape[0])
    speed_basis_count = int(np.asarray(first["coef_speed_basis_full_all"]).shape[0])

    attrs = {
        "schema_version": "2",
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "model_family": family_name,
        "bin_size_s": float(first["bin_size_s"]),
        "n_splines": int(first["n_splines"]),
        "spline_order": int(first["spline_order"]),
        "has_speed": bool(first["has_speed"]),
        "pos_bounds_lower": float(first["pos_bounds"][0]),
        "pos_bounds_upper": float(first["pos_bounds"][1]),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
    }
    for optional_key in (
        "n_splines_gain",
        "spline_order_gain",
        "gain_basis",
        "gain_overlap_frac",
        "maxiter_full",
        "lr_full",
    ):
        if optional_key in first:
            attrs[optional_key] = (
                json.dumps(first[optional_key])
                if isinstance(first[optional_key], (list, dict, tuple))
                else first[optional_key]
            )

    data_vars: dict[str, tuple[tuple[str, ...] | str, np.ndarray]] = {
        "dark_movement_firing_rate_hz": (
            "unit",
            np.asarray(dark_movement_firing_rates, dtype=float),
        ),
        "speed_feature_mode": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "speed_feature_mode"),
        ),
        "n_speed_features": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "n_speed_features"),
        ),
        "speed_basis": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "speed_basis"),
        ),
        "speed_spline_order": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "speed_spline_order"),
        ),
        "speed_basis_bounds": (
            ("trajectory", "ridge", "speed_bound"),
            _stack_family_array(results_by_traj, ridge_values, "speed_basis_bounds"),
        ),
        "speed_reference_value": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "speed_reference_value"),
        ),
        "speed_mean": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "speed_mean"),
        ),
        "speed_std": (
            ("trajectory", "ridge"),
            _stack_family_array(results_by_traj, ridge_values, "speed_std"),
        ),
        "spike_sum_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "spike_sum_cv"),
        ),
        "ll_base_sum_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "ll_base_sum_cv"),
        ),
        "ll_full_sum_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "ll_full_sum_cv"),
        ),
        "dLL_sum_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "dLL_sum_cv"),
        ),
        "ll_base_per_spike_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "ll_base_per_spike_cv"),
        ),
        "ll_full_per_spike_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "ll_full_per_spike_cv"),
        ),
        "dll_per_spike_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "dll_per_spike_cv"),
        ),
        "ll_base_bits_per_spike_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "ll_base_bits_per_spike_cv"),
        ),
        "ll_full_bits_per_spike_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "ll_full_bits_per_spike_cv"),
        ),
        "dll_bits_per_spike_cv": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "dll_bits_per_spike_cv"),
        ),
        "coef_intercept_base_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_intercept_base_all"),
        ),
        "coef_place_base_all": (
            ("trajectory", "ridge", "place_basis", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_place_base_all"),
        ),
        "coef_light_base_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_light_base_all"),
        ),
        "coef_place_x_light_base_all": (
            ("trajectory", "ridge", "light_mod_basis", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_place_x_light_base_all"),
        ),
        "coef_speed_base_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_speed_base_all"),
        ),
        "coef_speed_basis_base_all": (
            ("trajectory", "ridge", "speed_basis_feature", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_speed_basis_base_all"),
        ),
        "coef_intercept_full_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_intercept_full_all"),
        ),
        "coef_place_dark_full_all": (
            ("trajectory", "ridge", "place_basis", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_place_dark_full_all"),
        ),
        "coef_light_full_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_light_full_all"),
        ),
        "coef_speed_full_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_speed_full_all"),
        ),
        "coef_speed_basis_full_all": (
            ("trajectory", "ridge", "speed_basis_feature", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_speed_basis_full_all"),
        ),
        "coef_add_intercept_full_all": (
            ("trajectory", "ridge", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_add_intercept_full_all"),
        ),
        "coef_add_place_full_all": (
            ("trajectory", "ridge", "add_basis", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "coef_add_place_full_all"),
        ),
        "dark_hz_grid": (
            ("trajectory", "ridge", "tp_grid", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "dark_hz_grid"),
        ),
        "light_hz_grid": (
            ("trajectory", "ridge", "tp_grid", "unit"),
            _stack_family_array(results_by_traj, ridge_values, "light_hz_grid"),
        ),
    }
    coords: dict[str, np.ndarray] = {
        "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
        "ridge": np.asarray(ridge_values, dtype=float),
        "unit": unit_ids,
        "tp_grid": tp_grid,
        "place_basis": np.arange(place_basis_count, dtype=int),
        "light_mod_basis": np.arange(light_mod_basis_count, dtype=int),
        "speed_basis_feature": np.arange(speed_basis_count, dtype=int),
        "speed_bound": np.asarray(["lower", "upper"], dtype=str),
        "add_basis": np.arange(add_basis_count, dtype=int),
        "cv_metric": np.asarray(CV_METRIC_NAMES, dtype=str),
    }
    if family_name in {"mult", "mult_fewer_gain_splines"}:
        gain_basis_count = int(
            np.asarray(first["coef_light_gain_spline_full_all"]).shape[0]
        )
        coords["gain_basis_feature"] = np.arange(gain_basis_count, dtype=int)
        data_vars["coef_light_gain_spline_full_all"] = (
            ("trajectory", "ridge", "gain_basis_feature", "unit"),
            _stack_family_array(
                results_by_traj,
                ridge_values,
                "coef_light_gain_spline_full_all",
            ),
        )
    elif family_name in {"mult_per_segment", "mult_per_segment_overlap"}:
        segment_basis_count = int(
            np.asarray(first["coef_segment_bump_gain_full_all"]).shape[0]
        )
        coords["segment_basis"] = np.arange(segment_basis_count, dtype=int)
        data_vars["coef_segment_bump_gain_full_all"] = (
            ("trajectory", "ridge", "segment_basis", "unit"),
            _stack_family_array(
                results_by_traj,
                ridge_values,
                "coef_segment_bump_gain_full_all",
            ),
        )
    elif family_name == "mult_per_segment_scalar":
        segment_basis_count = int(
            np.asarray(first["coef_segment_scalar_gain_full_all"]).shape[0]
        )
        coords["segment_basis"] = np.arange(segment_basis_count, dtype=int)
        data_vars["coef_segment_scalar_gain_full_all"] = (
            ("trajectory", "ridge", "segment_basis", "unit"),
            _stack_family_array(
                results_by_traj,
                ridge_values,
                "coef_segment_scalar_gain_full_all",
            ),
        )
    elif family_name == "sep":
        data_vars["coef_place_light_full_all"] = (
            ("trajectory", "ridge", "place_basis", "unit"),
            _stack_family_array(
                results_by_traj,
                ridge_values,
                "coef_place_light_full_all",
            ),
        )

    dataset = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    dataset.attrs["min_dark_firing_rate_hz"] = float(min_dark_firing_rate_hz)
    if "segment_edges" in first:
        segment_edges = np.asarray(first["segment_edges"], dtype=float)
        dataset = dataset.assign_coords({"segment_edge": segment_edges})
        dataset["segment_edges"] = ("segment_edge", segment_edges)
    return dataset


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the dark/light task-progression GLM script."""
    parser = argparse.ArgumentParser(
        description="Fit dark/light task-progression GLM families for one session"
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
        "--epochs",
        nargs="+",
        help="Specific run epoch labels to consider before choosing light/dark pairs.",
    )
    parser.add_argument(
        "--light-epochs",
        nargs="+",
        help="Explicit run epoch labels to use as light epochs.",
    )
    parser.add_argument(
        "--dark-epochs",
        nargs="+",
        help="Explicit run epoch labels to use as dark epochs.",
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
        "--ca1-min-dark-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=f"Minimum dark-epoch movement firing rate for CA1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=0.02,
        help="Spike-count bin size in seconds. Default: 0.02",
    )
    parser.add_argument(
        "--ridges",
        nargs="+",
        type=float,
        default=list(DEFAULT_RIDGES),
        help="Ridge strengths to fit for each model family. Default: 1e-1 1e-2 1e-3 1e-4 1e-5",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of light/dark contiguous CV folds. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used for fold assignment. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--model-families",
        nargs="+",
        choices=DEFAULT_MODEL_FAMILIES,
        default=list(DEFAULT_MODEL_FAMILIES),
        help="Model families to fit.",
    )
    parser.add_argument(
        "--n-splines",
        type=int,
        default=25,
        help="Number of spline basis functions for the shared dark/place field. Default: 25",
    )
    parser.add_argument(
        "--spline-order",
        type=int,
        default=4,
        help="Spline order for the shared dark/place field. Default: 4",
    )
    parser.add_argument(
        "--n-splines-gain",
        type=int,
        default=6,
        help="Number of spline basis functions for `mult_fewer_gain_splines`. Default: 6",
    )
    parser.add_argument(
        "--spline-order-gain",
        type=int,
        default=4,
        help="Spline order for `mult_fewer_gain_splines`. Default: 4",
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
    parser.add_argument(
        "--gain-overlap-frac",
        type=float,
        default=DEFAULT_GAIN_OVERLAP_FRAC,
        help=f"Overlap fraction for `mult_per_segment_overlap`. Default: {DEFAULT_GAIN_OVERLAP_FRAC}",
    )
    return parser.parse_args()


def _fit_model_family(
    family_name: str,
    *,
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    light_epoch: str,
    dark_epoch: str,
    ridge: float,
    traj_name: str,
    unit_mask: np.ndarray,
    movement_by_run: dict[str, Any],
    args: argparse.Namespace,
    segment_edges: np.ndarray,
) -> dict[str, Any]:
    """Dispatch one model-family fit."""
    common_kwargs = {
        "spikes": spikes,
        "trajectory_ep_by_epoch": trajectory_ep_by_epoch,
        "tp_by_epoch": tp_by_epoch,
        "speed_by_epoch": speed_by_epoch,
        "traj_name": traj_name,
        "light_epochs": [light_epoch],
        "dark_epochs": [dark_epoch],
        "bin_size_s": args.bin_size_s,
        "n_splines": args.n_splines,
        "spline_order": args.spline_order,
        "ridge": ridge,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "unit_mask": unit_mask,
        "restrict_ep_by_epoch": movement_by_run,
        "speed_feature_mode": args.speed_feature_mode,
        "n_splines_speed": args.n_splines_speed,
        "spline_order_speed": args.spline_order_speed,
        "speed_bounds": None if args.speed_bounds is None else tuple(args.speed_bounds),
    }
    if family_name == "mult":
        return fit_shared_place_light_mod_nemos_per_traj(**common_kwargs)
    if family_name == "mult_per_segment":
        return fit_shared_place_light_mod_nemos_per_traj_segment_gain(
            **common_kwargs,
            segment_edges=segment_edges,
            gain_overlap_frac=0.0,
        )
    if family_name == "mult_fewer_gain_splines":
        return fit_shared_place_light_mod_nemos_per_traj_gain_splines(
            **common_kwargs,
            n_splines_gain=args.n_splines_gain,
            spline_order_gain=args.spline_order_gain,
        )
    if family_name == "mult_per_segment_scalar":
        return fit_shared_place_light_mod_nemos_per_traj_segment_scalar_gain(
            **common_kwargs,
            segment_edges=segment_edges,
        )
    if family_name == "mult_per_segment_overlap":
        return fit_shared_place_light_mod_nemos_per_traj_segment_gain(
            **common_kwargs,
            segment_edges=segment_edges,
            gain_overlap_frac=args.gain_overlap_frac,
        )
    if family_name == "add_jax":
        return fit_true_additive_rate_poisson_fast_per_traj(**common_kwargs)
    if family_name == "sep":
        return fit_separate_dark_light_fields_nemos_per_traj(**common_kwargs)
    raise ValueError(f"Unknown model family: {family_name}")


def main() -> None:
    """Run the modernized dark/light task-progression GLM workflow."""
    args = parse_arguments()
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    selected_epochs = select_run_epochs(session["run_epochs"], args.epochs)
    epoch_pairs = select_light_dark_pairs(
        session["run_epochs"],
        selected_epochs=selected_epochs,
        light_epochs=args.light_epochs,
        dark_epochs=args.dark_epochs,
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
    data_dir = analysis_path / "task_progression_dark_light"
    data_dir.mkdir(parents=True, exist_ok=True)

    region_thresholds = {
        "v1": float(args.v1_min_dark_fr_hz),
        "ca1": float(args.ca1_min_dark_fr_hz),
    }
    saved_datasets: list[Path] = []
    skipped_fits: list[dict[str, Any]] = []
    speed_by_run = session["speed_by_run"] if args.use_speed else None

    for region in args.regions:
        spikes_region = session["spikes_by_region"][region]
        for light_epoch, dark_epoch in epoch_pairs:
            dark_epoch_rates = np.asarray(
                movement_firing_rates[region][dark_epoch],
                dtype=float,
            )
            unit_mask = np.isfinite(dark_epoch_rates) & (
                dark_epoch_rates > region_thresholds[region]
            )
            if not np.any(unit_mask):
                skipped_fits.append(
                    {
                        "region": region,
                        "light_epoch": light_epoch,
                        "dark_epoch": dark_epoch,
                        "reason": "no units passed the dark-epoch firing-rate threshold",
                        "threshold_hz": region_thresholds[region],
                    }
                )
                continue

            selected_dark_rates = dark_epoch_rates[unit_mask]
            for family_name in args.model_families:
                family_results: dict[str, dict[float, dict[str, Any]]] = {
                    trajectory: {} for trajectory in TRAJECTORY_TYPES
                }
                family_failed = False
                for trajectory in TRAJECTORY_TYPES:
                    for ridge in args.ridges:
                        try:
                            family_results[trajectory][float(ridge)] = _fit_model_family(
                                family_name,
                                spikes=spikes_region,
                                trajectory_ep_by_epoch=session["trajectory_intervals"],
                                tp_by_epoch=session["task_progression_by_trajectory"],
                                speed_by_epoch=speed_by_run,
                                light_epoch=light_epoch,
                                dark_epoch=dark_epoch,
                                ridge=float(ridge),
                                traj_name=trajectory,
                                unit_mask=unit_mask,
                                movement_by_run=session["movement_by_run"],
                                args=args,
                                segment_edges=segment_edges,
                            )
                        except Exception as exc:
                            skipped_fits.append(
                                {
                                    "region": region,
                                    "light_epoch": light_epoch,
                                    "dark_epoch": dark_epoch,
                                    "model_family": family_name,
                                    "trajectory": trajectory,
                                    "ridge": float(ridge),
                                    "reason": "fit failed",
                                    "error": str(exc),
                                }
                            )
                            family_failed = True
                            break
                    if family_failed:
                        break

                if family_failed:
                    continue

                fit_parameters = {
                    "cuda_visible_devices": args.cuda_visible_devices,
                    "position_offset": args.position_offset,
                    "speed_threshold_cm_s": args.speed_threshold_cm_s,
                    "bin_size_s": args.bin_size_s,
                    "ridges": [float(ridge) for ridge in args.ridges],
                    "n_folds": args.n_folds,
                    "seed": args.seed,
                    "n_splines": args.n_splines,
                    "spline_order": args.spline_order,
                    "n_splines_gain": args.n_splines_gain,
                    "spline_order_gain": args.spline_order_gain,
                    "use_speed": args.use_speed,
                    "speed_feature_mode": args.speed_feature_mode,
                    "n_splines_speed": args.n_splines_speed,
                    "spline_order_speed": args.spline_order_speed,
                    "speed_bounds": args.speed_bounds,
                    "segment_edges": [float(edge) for edge in segment_edges],
                    "gain_overlap_frac": args.gain_overlap_frac,
                    "model_family": family_name,
                }
                dataset = build_family_dataset(
                    family_name=family_name,
                    results_by_traj=family_results,
                    ridge_values=[float(ridge) for ridge in args.ridges],
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    dark_movement_firing_rates=selected_dark_rates,
                    min_dark_firing_rate_hz=region_thresholds[region],
                    sources=session["sources"],
                    fit_parameters=fit_parameters,
                )
                result_path = (
                    data_dir
                    / f"{region}_{light_epoch}_vs_{dark_epoch}_{family_name}.nc"
                )
                dataset.to_netcdf(result_path)
                saved_datasets.append(result_path)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.dark_light_glm",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "cuda_visible_devices": args.cuda_visible_devices,
            "regions": args.regions,
            "epochs": selected_epochs,
            "light_epochs": args.light_epochs,
            "dark_epochs": args.dark_epochs,
            "epoch_pairs": epoch_pairs,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "region_dark_thresholds_hz": region_thresholds,
            "bin_size_s": args.bin_size_s,
            "ridges": args.ridges,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "model_families": args.model_families,
            "n_splines": args.n_splines,
            "spline_order": args.spline_order,
            "n_splines_gain": args.n_splines_gain,
            "spline_order_gain": args.spline_order_gain,
            "use_speed": args.use_speed,
            "speed_feature_mode": args.speed_feature_mode,
            "n_splines_speed": args.n_splines_speed,
            "spline_order_speed": args.spline_order_speed,
            "speed_bounds": args.speed_bounds,
            "segment_edges": segment_edges.tolist(),
            "gain_overlap_frac": args.gain_overlap_frac,
        },
        outputs={
            "sources": session["sources"],
            "saved_datasets": saved_datasets,
            "skipped_fits": skipped_fits,
        },
    )

    if saved_datasets:
        print(f"Saved {len(saved_datasets)} NetCDF fit dataset(s) to {data_dir}")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
