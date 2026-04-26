from __future__ import annotations

"""Compare selected dark/light task-progression GLMs on swapped light data.

This module consumes selected model outputs from `dark_light_glm.py`, loads only
the held-out light test epoch, and evaluates whether task-coordinate models
predict the swapped segment better than the visual model.

Supported models:

- `visual`: separate dark and light fields, analogous to `independent_light_field`
- `task_segment_bump`: dark field plus segment raised-cosine light gain
- `task_segment_scalar`: dark field plus segment scalar light gain
- `task_dense_gain`: dark field plus dense spline light gain
"""

import argparse
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from scipy.special import gammaln

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.wtrack import get_wtrack_geometry, get_wtrack_total_length
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    build_task_progression_bins,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    from nemos.basis import BSplineEval as BSplineEvalType

    import xarray as xr


BSplineEval = None
PopulationGLM = None


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_RIDGES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
DEFAULT_MODEL_NAMES = (
    "visual",
    "task_segment_bump",
    "task_segment_scalar",
    "task_dense_gain",
)
DEFAULT_SEGMENT_OVERLAP_FRAC = 0.0
DEFAULT_BIN_SIZE_S = 0.02
DEFAULT_N_SPLINES = 25
DEFAULT_SPLINE_ORDER = 4
DEFAULT_N_SPLINES_SPEED = 5
DEFAULT_SPLINE_ORDER_SPEED = 4
DEFAULT_SEED = 47
RIDGE_TIE_ATOL = 1e-6
POS_BOUNDS = (0.0, 1.0)
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


def _require_spline_basis() -> None:
    """Ensure NeMoS basis evaluation is available for selected GLM outputs."""
    global BSplineEval
    if BSplineEval is None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        try:
            from nemos.basis import BSplineEval as bspline_eval
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "This script requires `nemos` to evaluate selected task-progression "
                "GLM spline bases, but it is not installed."
            ) from exc
        BSplineEval = bspline_eval
    if BSplineEval is None:
        raise ModuleNotFoundError(
            "This script requires `nemos` to evaluate selected task-progression "
            "GLM spline bases, but it is not installed."
        )


def _require_nemos() -> None:
    """Ensure NeMoS fitting dependencies are available for legacy fit helpers."""
    global PopulationGLM
    _require_spline_basis()
    if PopulationGLM is None:
        try:
            from nemos.glm import PopulationGLM as population_glm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The legacy fitting helpers require `nemos.glm.PopulationGLM`, "
                "but it is not installed."
            ) from exc
        PopulationGLM = population_glm


def derive_default_segment_edges(animal_name: str) -> np.ndarray:
    """Derive the legacy three-segment TP edges from the W-track geometry."""
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


def _validate_segment_edges(
    segment_edges: Sequence[float],
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


def _normalize_speed_feature_mode(speed_feature_mode: str) -> str:
    """Validate the requested speed parameterization."""
    mode = str(speed_feature_mode).lower()
    if mode not in {"linear", "bspline"}:
        raise ValueError(
            "speed_feature_mode must be one of {'linear', 'bspline'}. "
            f"Got {speed_feature_mode!r}."
        )
    return mode


def _empty_speed_design(n_rows: int) -> np.ndarray:
    """Return an empty speed design matrix with the requested row count."""
    return np.zeros((int(n_rows), 0), dtype=float)


def _empty_speed_feature_transform() -> dict[str, Any]:
    """Return a sentinel speed transform when speed is disabled."""
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
    n_splines_speed: int = DEFAULT_N_SPLINES_SPEED,
    spline_order_speed: int = DEFAULT_SPLINE_ORDER_SPEED,
    speed_bounds: tuple[float, float] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Fit the speed-feature transform from training samples."""
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
    """Apply a fitted speed transform to one vector of samples."""
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


def _speed_reference_design(transform: dict[str, Any]) -> np.ndarray:
    """Return the design row evaluated at the reference speed."""
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
    """Return the speed contribution used for the smooth TP grids."""
    coef_speed_basis = np.asarray(coef_speed_basis, dtype=float).reshape(-1, n_units)
    if coef_speed_basis.shape[0] == 0:
        return np.zeros((1, n_units), dtype=float)
    return _speed_reference_design(transform) @ coef_speed_basis


def _format_speed_outputs(
    *,
    transform: dict[str, Any],
    coef_speed_basis: np.ndarray,
    n_units: int,
) -> dict[str, Any]:
    """Normalize speed metadata and coefficients for saved outputs."""
    coef_speed_basis = np.asarray(coef_speed_basis, dtype=float).reshape(-1, n_units)
    coef_speed = np.full((n_units,), np.nan, dtype=float)
    if str(transform["mode"]) == "linear" and coef_speed_basis.shape[0] > 0:
        coef_speed = coef_speed_basis[0, :]

    return {
        "speed_feature_mode": str(transform["mode"]),
        "n_speed_features": int(transform["n_features"]),
        "speed_basis": str(transform["basis"]),
        "speed_spline_order": float(transform["spline_order"]),
        "speed_basis_bounds": np.asarray(transform["bounds"], dtype=float),
        "speed_reference_value": (
            float(transform["reference_value"])
            if np.isfinite(transform["reference_value"])
            else np.nan
        ),
        "speed_mean": (
            float(transform["mean"])
            if np.isfinite(transform["mean"])
            else np.nan
        ),
        "speed_std": (
            float(transform["std"])
            if np.isfinite(transform["std"])
            else np.nan
        ),
        "coef_speed": coef_speed,
        "coef_speed_basis": coef_speed_basis,
    }


def normalize_requested_models(model_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return selected-file model names, always including the visual baseline."""
    normalized: list[str] = []
    messages: list[str] = []
    for model_name in model_names:
        name = str(model_name)
        if name not in DEFAULT_MODEL_NAMES:
            raise ValueError(
                f"Unknown model {name!r}. Expected one of {DEFAULT_MODEL_NAMES!r}."
            )
        if name not in normalized:
            normalized.append(name)
    if "visual" not in normalized:
        normalized.insert(0, "visual")
        messages.append("Added required baseline model 'visual'.")
    return normalized, messages


def selected_dark_light_glm_dir(analysis_path: Path) -> Path:
    """Return the default selected-output directory from `dark_light_glm.py`."""
    return (
        get_task_progression_output_dir(analysis_path, "dark_light_glm")
        / "selected"
    )


def selected_dark_light_glm_path(
    selected_dir: Path,
    *,
    region: str,
    light_train_epoch: str,
    dark_train_epoch: str,
    model_name: str,
) -> Path:
    """Return the expected selected dark/light GLM path for one model."""
    return (
        Path(selected_dir)
        / f"{region}_{light_train_epoch}_vs_{dark_train_epoch}_{model_name}_selected.nc"
    )


def load_selected_dark_light_glm(
    selected_dir: Path,
    *,
    region: str,
    light_train_epoch: str,
    dark_train_epoch: str,
    model_name: str,
) -> tuple["xr.Dataset", Path]:
    """Load one selected dark/light GLM dataset into memory."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to load selected dark/light GLM NetCDF files."
        ) from exc

    path = selected_dark_light_glm_path(
        selected_dir,
        region=region,
        light_train_epoch=light_train_epoch,
        dark_train_epoch=dark_train_epoch,
        model_name=model_name,
    )
    if not path.exists():
        raise FileNotFoundError(
            "Missing selected dark-light GLM file:\n"
            f"  expected {path}\n"
            "Run dark_light_glm.py first with the matching dark and light train epochs."
        )
    dataset = xr.open_dataset(path)
    dataset.load()
    dataset.close()
    return dataset, path


def _attr_epoch(dataset: "xr.Dataset", *names: str) -> str | None:
    """Return the first matching epoch attr value from a selected dataset."""
    for name in names:
        if name in dataset.attrs:
            return str(dataset.attrs[name])
    return None


def _assert_coord_equal(
    reference: "xr.Dataset",
    candidate: "xr.Dataset",
    coord_name: str,
    *,
    model_name: str,
) -> None:
    """Raise if two selected datasets disagree on one coordinate."""
    reference_values = np.asarray(reference.coords[coord_name].values)
    candidate_values = np.asarray(candidate.coords[coord_name].values)
    if reference_values.shape != candidate_values.shape or not np.array_equal(
        reference_values,
        candidate_values,
    ):
        raise ValueError(
            f"Selected dark-light file for model {model_name!r} disagrees on "
            f"coordinate {coord_name!r}."
        )


def _assert_attr_equal(
    reference: "xr.Dataset",
    candidate: "xr.Dataset",
    attr_name: str,
    *,
    model_name: str,
    numeric: bool = False,
) -> None:
    """Raise if two selected datasets disagree on one attribute."""
    if attr_name not in reference.attrs or attr_name not in candidate.attrs:
        raise ValueError(
            f"Selected dark-light file for model {model_name!r} is missing "
            f"required attr {attr_name!r}."
        )
    if numeric:
        if not np.isclose(
            float(reference.attrs[attr_name]),
            float(candidate.attrs[attr_name]),
            equal_nan=True,
        ):
            raise ValueError(
                f"Selected dark-light file for model {model_name!r} disagrees on "
                f"attr {attr_name!r}."
            )
    elif reference.attrs[attr_name] != candidate.attrs[attr_name]:
        raise ValueError(
            f"Selected dark-light file for model {model_name!r} disagrees on "
            f"attr {attr_name!r}."
        )


def validate_selected_dark_light_glms(
    datasets_by_model: dict[str, "xr.Dataset"],
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
) -> dict[str, Any]:
    """Validate selected dark/light outputs and return shared fit metadata."""
    if "visual" not in datasets_by_model:
        raise ValueError("The selected visual model is required as the baseline.")
    reference = datasets_by_model["visual"]

    for model_name, dataset in datasets_by_model.items():
        if str(dataset.attrs.get("fit_stage", "")) != "selected":
            raise ValueError(
                f"Selected dark-light file for model {model_name!r} is not marked "
                "fit_stage='selected'."
            )
        expected_attrs = {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "model_name": model_name,
        }
        for attr_name, expected in expected_attrs.items():
            if str(dataset.attrs.get(attr_name, "")) != str(expected):
                raise ValueError(
                    f"Selected dark-light file for model {model_name!r} has "
                    f"{attr_name}={dataset.attrs.get(attr_name)!r}, expected {expected!r}."
                )
        dark_epoch = _attr_epoch(dataset, "dark_train_epoch", "dark_epoch")
        light_epoch = _attr_epoch(dataset, "light_train_epoch", "light_epoch")
        if dark_epoch != str(dark_train_epoch) or light_epoch != str(light_train_epoch):
            raise ValueError(
                f"Selected dark-light file for model {model_name!r} has "
                f"dark/light train epochs {dark_epoch!r}/{light_epoch!r}, expected "
                f"{dark_train_epoch!r}/{light_train_epoch!r}."
            )
        if bool(dataset.attrs.get("has_light_offset_term", False)) is not True:
            raise ValueError(
                f"Selected dark-light file for model {model_name!r} must include "
                "the scalar light offset term."
            )

    for model_name, dataset in datasets_by_model.items():
        for coord_name in ("unit", "trajectory", "tp_grid", "segment_edge"):
            _assert_coord_equal(reference, dataset, coord_name, model_name=model_name)
        for attr_name in ("bin_size_s", "selected_bin_size_s"):
            _assert_attr_equal(
                reference,
                dataset,
                attr_name,
                model_name=model_name,
                numeric=True,
            )
        for attr_name in (
            "selected_n_splines",
            "spline_order",
            "has_speed",
            "speed_feature_mode",
            "n_speed_features",
        ):
            _assert_attr_equal(reference, dataset, attr_name, model_name=model_name)
        _assert_attr_equal(
            reference,
            dataset,
            "speed_spline_order",
            model_name=model_name,
            numeric=True,
        )
        if bool(dataset.attrs.get("has_speed", False)):
            for trajectory in np.asarray(dataset.coords["trajectory"].values, dtype=str):
                _speed_transform_from_selected_dataset(dataset, str(trajectory))

    return {
        "unit_ids": np.asarray(reference.coords["unit"].values),
        "trajectory": np.asarray(reference.coords["trajectory"].values, dtype=str),
        "tp_grid": np.asarray(reference.coords["tp_grid"].values, dtype=float),
        "segment_edges": np.asarray(reference.coords["segment_edge"].values, dtype=float),
        "bin_size_s": float(reference.attrs["selected_bin_size_s"]),
        "n_splines": int(reference.attrs["selected_n_splines"]),
        "spline_order": int(reference.attrs["spline_order"]),
        "has_speed": bool(reference.attrs["has_speed"]),
        "speed_feature_mode": str(reference.attrs["speed_feature_mode"]),
        "n_speed_features": int(reference.attrs["n_speed_features"]),
        "speed_spline_order": float(reference.attrs["speed_spline_order"]),
    }


def _segment_center_raised_cosine_basis(
    x: np.ndarray,
    segment_edges: Sequence[float],
    *,
    pos_bounds: tuple[float, float] = POS_BOUNDS,
    overlap_frac: float = 0.0,
) -> np.ndarray:
    """Return one raised-cosine bump per task-progression segment."""
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
    pos_bounds: tuple[float, float] = POS_BOUNDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-hot segment membership without dropping any segment."""
    edges = _validate_segment_edges(segment_edges, pos_bounds=pos_bounds)
    x = np.asarray(x, dtype=float).reshape(-1)
    segment_index = np.searchsorted(edges, x, side="right") - 1
    segment_index = np.clip(segment_index, 0, edges.size - 2)

    n_segments = edges.size - 1
    basis = np.zeros((x.size, n_segments), dtype=float)
    basis[np.arange(x.size), segment_index] = 1.0
    return basis, edges


def _coef_feat_by_unit(model: PopulationGLM, n_features: int) -> np.ndarray:
    """Return coefficients as `(feature, unit)` regardless of backend layout."""
    coef = np.asarray(model.coef_)
    if coef.shape[0] != n_features:
        coef = coef.T
    return coef


def _prepare_epoch_inputs(
    *,
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    traj_name: str,
    epoch: str,
    bin_size_s: float,
    unit_mask: np.ndarray | None,
    restrict_interval: Any | None,
) -> dict[str, Any]:
    """Assemble binned spikes, TP, and optional speed for one epoch/trajectory."""
    selected_spikes = spikes if unit_mask is None else spikes[unit_mask]
    trajectory_interval = trajectory_ep_by_epoch[epoch][traj_name]
    if restrict_interval is not None:
        trajectory_interval = trajectory_interval.intersect(restrict_interval)

    counts = selected_spikes.count(bin_size_s, ep=trajectory_interval)
    unit_ids = np.asarray(counts.columns)
    y_epoch = np.asarray(counts.d, dtype=float)
    p_epoch = tp_by_epoch[epoch][traj_name].interpolate(counts).to_numpy().reshape(-1)

    if speed_by_epoch is None:
        valid_mask = np.isfinite(p_epoch)
        v_epoch = None
    else:
        speed_epoch = speed_by_epoch[epoch].interpolate(counts).to_numpy().reshape(-1)
        valid_mask = np.isfinite(p_epoch) & np.isfinite(speed_epoch)
        v_epoch = np.asarray(speed_epoch[valid_mask], dtype=float)

    return {
        "unit_ids": unit_ids,
        "y": np.asarray(y_epoch[valid_mask], dtype=float),
        "p": np.asarray(p_epoch[valid_mask], dtype=float),
        "v": v_epoch,
    }


def _prepare_test_epoch_inputs_for_units(
    *,
    spikes: Any,
    unit_ids: np.ndarray,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    traj_name: str,
    epoch: str,
    bin_size_s: float,
    restrict_interval: Any | None,
) -> dict[str, Any]:
    """Assemble held-out inputs for selected-file units in exact saved order."""
    requested_unit_ids = np.asarray(unit_ids)
    available_unit_ids = np.asarray(list(spikes.keys()))
    missing = [
        unit_id for unit_id in requested_unit_ids if unit_id not in set(available_unit_ids)
    ]
    if missing:
        raise ValueError(
            f"Held-out spikes are missing selected units for trajectory {traj_name!r}: "
            f"{missing!r}."
        )
    unit_mask = np.isin(available_unit_ids, requested_unit_ids)
    selected_spikes = spikes[unit_mask]
    trajectory_interval = trajectory_ep_by_epoch[epoch][traj_name]
    if restrict_interval is not None:
        trajectory_interval = trajectory_interval.intersect(restrict_interval)

    counts = selected_spikes.count(bin_size_s, ep=trajectory_interval)
    count_unit_ids = np.asarray(counts.columns)
    order = []
    for unit_id in requested_unit_ids:
        matches = np.where(count_unit_ids == unit_id)[0]
        if matches.size != 1:
            raise ValueError(
                f"Could not recover selected unit {unit_id!r} from held-out spike counts."
            )
        order.append(int(matches[0]))
    order_array = np.asarray(order, dtype=int)
    y_epoch = np.asarray(counts.d, dtype=float)[:, order_array]
    p_epoch = tp_by_epoch[epoch][traj_name].interpolate(counts).to_numpy().reshape(-1)

    if speed_by_epoch is None:
        valid_mask = np.isfinite(p_epoch)
        v_epoch = None
    else:
        speed_epoch = speed_by_epoch[epoch].interpolate(counts).to_numpy().reshape(-1)
        valid_mask = np.isfinite(p_epoch) & np.isfinite(speed_epoch)
        v_epoch = np.asarray(speed_epoch[valid_mask], dtype=float)

    return {
        "unit_ids": requested_unit_ids,
        "y": np.asarray(y_epoch[valid_mask], dtype=float),
        "p": np.asarray(p_epoch[valid_mask], dtype=float),
        "v": v_epoch,
    }


def _validate_unit_ids(epoch_inputs: dict[str, dict[str, Any]]) -> np.ndarray:
    """Validate that all epoch inputs share the same unit ordering."""
    reference_ids: np.ndarray | None = None
    for epoch_name, inputs in epoch_inputs.items():
        unit_ids = np.asarray(inputs["unit_ids"])
        if reference_ids is None:
            reference_ids = unit_ids
            continue
        if unit_ids.shape != reference_ids.shape or not np.all(unit_ids == reference_ids):
            raise ValueError(
                "Spike count columns (unit order) differ across epochs for "
                f"{epoch_name!r}."
            )
    if reference_ids is None:
        raise ValueError("No epoch inputs were available.")
    return reference_ids


def _poisson_log_likelihood_per_neuron(
    y_true: np.ndarray,
    lam_pred: np.ndarray,
) -> np.ndarray:
    """Return the Poisson log-likelihood per neuron."""
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.clip(np.asarray(lam_pred, dtype=float), 1e-12, None)
    return np.sum(y_true * np.log(lam_pred) - lam_pred - gammaln(y_true + 1.0), axis=0)


def _poisson_log_likelihood_saturated_per_neuron(y_true: np.ndarray) -> np.ndarray:
    """Return the saturated-model Poisson log-likelihood per neuron."""
    y_true = np.asarray(y_true, dtype=float)
    y_safe = np.clip(y_true, 1e-12, None)
    return np.sum(y_true * np.log(y_safe) - y_true - gammaln(y_true + 1.0), axis=0)


def compute_bits_per_spike_per_neuron(
    y_true: np.ndarray,
    lam_pred: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """Return information gain in bits/spike per neuron."""
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.asarray(lam_pred, dtype=float)
    y_null_fit = np.asarray(y_null_fit, dtype=float)
    n_units = y_true.shape[1] if y_true.ndim == 2 else lam_pred.shape[1]
    if y_true.size == 0 or y_null_fit.size == 0:
        return np.full((n_units,), np.nan, dtype=float)

    ll_model = _poisson_log_likelihood_per_neuron(y_true, lam_pred)
    lam0 = np.mean(y_null_fit, axis=0)
    lam0 = np.clip(lam0, 1e-12, None)
    ll_null = np.sum(
        y_true * np.log(lam0[None, :]) - lam0[None, :] - gammaln(y_true + 1.0),
        axis=0,
    )
    spikes = np.sum(y_true, axis=0)
    denom = spikes * np.log(2.0)
    bits_per_spike = (ll_model - ll_null) / np.clip(denom, 1e-12, None)
    return np.where(spikes > 0, bits_per_spike, np.nan)


def compute_deviance_explained_per_neuron(
    y_true: np.ndarray,
    lam_pred: np.ndarray,
    y_null_fit: np.ndarray,
) -> np.ndarray:
    """Return Poisson deviance explained per neuron."""
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.asarray(lam_pred, dtype=float)
    y_null_fit = np.asarray(y_null_fit, dtype=float)
    n_units = y_true.shape[1] if y_true.ndim == 2 else lam_pred.shape[1]
    if y_true.size == 0 or y_null_fit.size == 0:
        return np.full((n_units,), np.nan, dtype=float)

    ll_model = _poisson_log_likelihood_per_neuron(y_true, lam_pred)
    lam0 = np.mean(y_null_fit, axis=0, keepdims=True)
    lam0 = np.repeat(lam0, y_true.shape[0], axis=0)
    ll_null = _poisson_log_likelihood_per_neuron(y_true, lam0)
    ll_sat = _poisson_log_likelihood_saturated_per_neuron(y_true)
    denom = ll_sat - ll_null
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    return (ll_model - ll_null) / denom


def summarize_poisson_metrics(
    y_true: np.ndarray,
    lam_pred: np.ndarray,
    y_null_fit: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the saved scalar metrics for one epoch."""
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.asarray(lam_pred, dtype=float)
    n_units = y_true.shape[1] if y_true.ndim == 2 else lam_pred.shape[1]
    if y_true.size == 0:
        ll_sum = np.zeros((n_units,), dtype=float)
        spike_sum = np.zeros((n_units,), dtype=float)
        ll_per_spike = np.full((n_units,), np.nan, dtype=float)
    else:
        ll_sum = _poisson_log_likelihood_per_neuron(y_true, lam_pred)
        spike_sum = np.sum(y_true, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll_per_spike = np.where(spike_sum > 0, ll_sum / spike_sum, np.nan)
    return {
        "spike_sum": np.asarray(spike_sum, dtype=float),
        "ll_sum": np.asarray(ll_sum, dtype=float),
        "ll_bits_per_spike": compute_bits_per_spike_per_neuron(
            y_true,
            lam_pred,
            y_null_fit,
        ),
        "deviance_explained": compute_deviance_explained_per_neuron(
            y_true,
            lam_pred,
            y_null_fit,
        ),
        "ll_per_spike": np.asarray(ll_per_spike, dtype=float),
    }


def summarize_raw_poisson_metrics(
    y_true: np.ndarray,
    lam_pred: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return raw held-out Poisson LL summaries without a null baseline."""
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.asarray(lam_pred, dtype=float)
    n_units = y_true.shape[1] if y_true.ndim == 2 else lam_pred.shape[1]
    if y_true.size == 0:
        ll_sum = np.zeros((n_units,), dtype=float)
        spike_sum = np.zeros((n_units,), dtype=float)
    else:
        ll_sum = _poisson_log_likelihood_per_neuron(y_true, lam_pred)
        spike_sum = np.sum(y_true, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ll_bits_per_spike = np.where(
            spike_sum > 0,
            ll_sum / (spike_sum * np.log(2.0)),
            np.nan,
        )
    return {
        "raw_ll_sum": np.asarray(ll_sum, dtype=float),
        "spike_sum": np.asarray(spike_sum, dtype=float),
        "raw_ll_bits_per_spike": np.asarray(raw_ll_bits_per_spike, dtype=float),
    }


def build_observed_summary(
    y_epoch: np.ndarray,
    p_epoch: np.ndarray,
    bin_edges: np.ndarray,
    *,
    bin_size_s: float,
) -> dict[str, np.ndarray]:
    """Build occupancy, spike-count, and observed-rate summaries on TP bins."""
    y_epoch = np.asarray(y_epoch, dtype=float)
    p_epoch = np.asarray(p_epoch, dtype=float).reshape(-1)
    bin_edges = np.asarray(bin_edges, dtype=float).reshape(-1)
    n_bins = bin_edges.size - 1
    if n_bins <= 0:
        raise ValueError("Observed-summary bin_edges must contain at least two values.")

    occupancy_s = np.zeros((n_bins,), dtype=float)
    spike_count = np.zeros((n_bins, y_epoch.shape[1]), dtype=float)
    bin_index = np.searchsorted(bin_edges, p_epoch, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    np.add.at(occupancy_s, bin_index, float(bin_size_s))
    np.add.at(spike_count, bin_index, y_epoch)

    observed_rate_hz = np.full_like(spike_count, np.nan, dtype=float)
    valid_occ = occupancy_s > 0
    observed_rate_hz[valid_occ] = spike_count[valid_occ] / occupancy_s[valid_occ, None]
    return {
        "occupancy_s": occupancy_s,
        "spike_count": spike_count,
        "observed_rate_hz": observed_rate_hz,
    }


def _segment_mask(
    x: np.ndarray,
    segment_edges: np.ndarray,
    segment_index: int,
) -> np.ndarray:
    """Return the bin mask for one segment, including the right edge on the last bin."""
    x = np.asarray(x, dtype=float).reshape(-1)
    start = float(segment_edges[segment_index])
    end = float(segment_edges[segment_index + 1])
    if segment_index == (segment_edges.size - 2):
        return (x >= start) & (x <= end)
    return (x >= start) & (x < end)


def _format_ridge_token(ridge: float) -> str:
    """Return a stable filename token for one ridge value."""
    return np.format_float_scientific(float(ridge), trim="-", exp_digits=2)


def _format_bin_size_token(bin_size_s: float) -> str:
    """Return a stable filename token for one spike-count bin size."""
    token = np.format_float_positional(float(bin_size_s), trim="-")
    return token.replace("-", "m").replace(".", "p")


def _extract_interval_bounds(
    intervals: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start/end arrays from one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "Trajectory interval bounds have mismatched shapes: "
            f"{starts.shape} vs {ends.shape}."
        )
    return starts, ends


def _subset_intervalset(
    intervals: Any,
    indices: np.ndarray,
) -> Any:
    """Return one IntervalSet containing only the requested lap indices."""
    intervalset_class = intervals.__class__
    starts, ends = _extract_interval_bounds(intervals)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    if indices.size == 0:
        return intervalset_class(
            start=np.asarray([], dtype=float),
            end=np.asarray([], dtype=float),
            time_units="s",
        )
    return intervalset_class(
        start=np.asarray(starts[indices], dtype=float),
        end=np.asarray(ends[indices], dtype=float),
        time_units="s",
    )


def split_test_light_laps_by_trajectory(
    trajectory_intervals_by_type: dict[str, Any],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Split one held-out light epoch into validation/test laps per trajectory."""
    rng = np.random.default_rng(int(seed))
    split_by_traj: dict[str, dict[str, Any]] = {}
    for trajectory in TRAJECTORY_TYPES:
        intervals = trajectory_intervals_by_type[trajectory]
        lap_start_s, lap_end_s = _extract_interval_bounds(intervals)
        n_laps = int(lap_start_s.size)
        if n_laps < 2:
            raise ValueError(
                f"Held-out light epoch has fewer than 2 laps for trajectory "
                f"{trajectory!r} (n_laps={n_laps})."
            )

        permuted = np.asarray(rng.permutation(n_laps), dtype=int)
        validation_idx, test_idx = [
            np.sort(chunk.astype(int, copy=False))
            for chunk in np.array_split(permuted, 2)
        ]
        split_labels = np.full((n_laps,), "", dtype="<U10")
        split_labels[validation_idx] = "validation"
        split_labels[test_idx] = "test"

        split_by_traj[trajectory] = {
            "lap_start_s": lap_start_s,
            "lap_end_s": lap_end_s,
            "lap_split": split_labels,
            "validation_indices": validation_idx,
            "test_indices": test_idx,
            "validation_interval": _subset_intervalset(intervals, validation_idx),
            "test_interval": _subset_intervalset(intervals, test_idx),
        }
    return split_by_traj


def choose_best_ridge(
    ridge_values: Sequence[float],
    validation_metric_by_ridge: dict[float, float],
    *,
    atol: float = RIDGE_TIE_ATOL,
) -> float:
    """Choose the best ridge, preferring stronger regularization in near-ties."""
    ordered_ridges = [float(ridge) for ridge in ridge_values]
    best_ridge: float | None = None
    best_metric = -np.inf
    for ridge in ordered_ridges:
        metric = float(validation_metric_by_ridge[ridge])
        if not np.isfinite(metric):
            continue
        if best_ridge is None or metric > (best_metric + float(atol)):
            best_ridge = ridge
            best_metric = metric
            continue
        if (
            best_ridge is not None
            and abs(metric - best_metric) <= float(atol)
            and ridge > best_ridge
        ):
            best_ridge = ridge
    if best_ridge is None:
        raise ValueError("No finite validation scores were available for ridge selection.")
    return float(best_ridge)


def _output_stem(
    *,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    model_name: str,
    bin_size_s: float,
    ridge: float,
    include_light_offset_term: bool = True,
) -> str:
    """Return the shared filename stem for one saved model artifact."""
    bin_size_token = _format_bin_size_token(bin_size_s)
    ridge_token = _format_ridge_token(ridge)
    light_offset_token = "" if include_light_offset_term else "_no_light_offset"
    return (
        f"{region}_{dark_train_epoch}_traindark_"
        f"{light_train_epoch}_trainlight_"
        f"{light_test_epoch}_testlight_"
        f"bin{bin_size_token}s_"
        f"{model_name}{light_offset_token}_ridge{ridge_token}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the model-comparison workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Score selected dark/light task-progression GLMs on a held-out "
            "swapped-light epoch."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base analysis directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Regions to fit. Default: {' '.join(REGIONS)}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODEL_NAMES,
        default=list(DEFAULT_MODEL_NAMES),
        help=(
            "Selected model families to score. The visual baseline is always "
            "included even if omitted. "
            f"Default: {' '.join(DEFAULT_MODEL_NAMES)}"
        ),
    )
    parser.add_argument(
        "--dark-train-epoch",
        required=True,
        help="Dark run epoch used for fitting.",
    )
    parser.add_argument(
        "--light-train-epoch",
        required=True,
        help="Light run epoch used for fitting.",
    )
    parser.add_argument(
        "--light-test-epoch",
        required=True,
        help="Held-out light run epoch used for evaluation.",
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
        "--dark-light-glm-dir",
        type=Path,
        help=(
            "Directory containing dark_light_glm selected NetCDF files. "
            "Default: analysis_path/task_progression/dark_light_glm/selected"
        ),
    )
    parser.add_argument(
        "--swap-light-offset",
        action="store_true",
        help=(
            "Also swap the selected trajectory's scalar light-offset coefficient "
            "inside the swapped segment. Default: swap only the local "
            "trajectory/segment-dependent component."
        ),
    )
    return parser.parse_args()


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


def _predict_visual_components(
    *,
    p_eval: np.ndarray,
    speed_design: np.ndarray,
    basis: BSplineEval,
    intercept: np.ndarray,
    coef_light: np.ndarray,
    coef_place_dark: np.ndarray,
    coef_place_light: np.ndarray,
    coef_speed_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return dark, light, and light-place components for the visual model."""
    grid_basis = np.asarray(basis.compute_features(p_eval), dtype=float)
    speed_part = speed_design @ coef_speed_basis
    dark_eta = intercept[None, :] + (grid_basis @ coef_place_dark) + speed_part
    light_place = grid_basis @ coef_place_light
    light_eta = intercept[None, :] + coef_light[None, :] + light_place + speed_part
    return {
        "dark_eta": dark_eta,
        "light_eta": light_eta,
        "light_place": light_place,
    }


def _predict_task_components(
    *,
    p_eval: np.ndarray,
    speed_design: np.ndarray,
    dark_basis: BSplineEval,
    segment_edges: np.ndarray,
    intercept: np.ndarray,
    coef_light: np.ndarray,
    coef_place_dark: np.ndarray,
    coef_gain: np.ndarray,
    coef_speed_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return dark, light, and segment-gain components for the task model."""
    grid_dark = np.asarray(dark_basis.compute_features(p_eval), dtype=float)
    grid_gain = _segment_center_raised_cosine_basis(
        p_eval,
        segment_edges,
        pos_bounds=POS_BOUNDS,
        overlap_frac=DEFAULT_SEGMENT_OVERLAP_FRAC,
    )
    speed_part = speed_design @ coef_speed_basis
    dark_eta = intercept[None, :] + (grid_dark @ coef_place_dark) + speed_part
    gain_part = grid_gain @ coef_gain
    light_eta = dark_eta + coef_light[None, :] + gain_part
    return {
        "dark_eta": dark_eta,
        "light_eta": light_eta,
        "gain_part": gain_part,
        "gain_basis": grid_gain,
    }


def _predict_task_scalar_components(
    *,
    p_eval: np.ndarray,
    speed_design: np.ndarray,
    dark_basis: BSplineEval,
    segment_edges: np.ndarray,
    intercept: np.ndarray,
    coef_light: np.ndarray,
    coef_place_dark: np.ndarray,
    coef_segment_scalar_gain: np.ndarray,
    coef_speed_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return dark, light, and scalar segment-gain components."""
    grid_dark = np.asarray(dark_basis.compute_features(p_eval), dtype=float)
    grid_gain, _ = _segment_onehot_basis(
        p_eval,
        segment_edges,
        pos_bounds=POS_BOUNDS,
    )
    speed_part = speed_design @ coef_speed_basis
    dark_eta = intercept[None, :] + (grid_dark @ coef_place_dark) + speed_part
    gain_part = grid_gain @ coef_segment_scalar_gain
    light_eta = dark_eta + coef_light[None, :] + gain_part
    return {
        "dark_eta": dark_eta,
        "light_eta": light_eta,
        "gain_part": gain_part,
        "gain_basis": grid_gain,
    }


def _predict_task_dense_gain_components(
    *,
    p_eval: np.ndarray,
    speed_design: np.ndarray,
    dark_basis: BSplineEval,
    gain_basis: BSplineEval,
    intercept: np.ndarray,
    coef_light: np.ndarray,
    coef_place_dark: np.ndarray,
    coef_gain_spline: np.ndarray,
    coef_speed_basis: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return dark, light, and dense gain-spline components."""
    grid_dark = np.asarray(dark_basis.compute_features(p_eval), dtype=float)
    grid_gain = np.asarray(gain_basis.compute_features(p_eval), dtype=float)
    speed_part = speed_design @ coef_speed_basis
    dark_eta = intercept[None, :] + (grid_dark @ coef_place_dark) + speed_part
    gain_part = grid_gain @ coef_gain_spline
    light_eta = dark_eta + coef_light[None, :] + gain_part
    return {
        "dark_eta": dark_eta,
        "light_eta": light_eta,
        "gain_part": gain_part,
        "gain_basis": grid_gain,
    }


def build_visual_swapped_light_eta(
    *,
    own_light_eta: np.ndarray,
    own_light_place: np.ndarray,
    paired_light_place: np.ndarray,
    own_light_offset: np.ndarray | float = 0.0,
    paired_light_offset: np.ndarray | float = 0.0,
    swap_mask: np.ndarray,
    swap_light_offset: bool = False,
) -> np.ndarray:
    """Return light prediction after swapping the local visual light component."""
    swapped_eta = np.asarray(own_light_eta, dtype=float).copy()
    delta = (
        np.asarray(paired_light_place, dtype=float)
        - np.asarray(own_light_place, dtype=float)
    )
    if swap_light_offset:
        offset_delta = (
            np.asarray(paired_light_offset, dtype=float)
            - np.asarray(own_light_offset, dtype=float)
        )
        delta = delta + (
            offset_delta if offset_delta.ndim == 0 else offset_delta.reshape(1, -1)
        )
    swapped_eta[np.asarray(swap_mask, dtype=bool)] += delta[np.asarray(swap_mask, dtype=bool)]
    return swapped_eta


def build_task_swapped_light_eta(
    *,
    own_light_eta: np.ndarray,
    own_gain_basis: np.ndarray,
    own_coef_gain: np.ndarray,
    paired_coef_gain: np.ndarray,
    own_light_offset: np.ndarray | float = 0.0,
    paired_light_offset: np.ndarray | float = 0.0,
    swap_segment_index: int,
    swap_mask: np.ndarray,
    swap_light_offset: bool = False,
) -> np.ndarray:
    """Return task-model light prediction after swapping one light segment."""
    swapped_eta = np.asarray(own_light_eta, dtype=float).copy()
    gain_column = np.asarray(own_gain_basis[:, swap_segment_index], dtype=float)[:, None]
    own_contribution = gain_column * np.asarray(own_coef_gain[swap_segment_index], dtype=float)[None, :]
    paired_contribution = gain_column * np.asarray(paired_coef_gain[swap_segment_index], dtype=float)[None, :]
    delta = paired_contribution - own_contribution
    if swap_light_offset:
        offset_delta = (
            np.asarray(paired_light_offset, dtype=float)
            - np.asarray(own_light_offset, dtype=float)
        )
        delta = delta + (
            offset_delta if offset_delta.ndim == 0 else offset_delta.reshape(1, -1)
        )
    mask = np.asarray(swap_mask, dtype=bool)
    swapped_eta[mask] += delta[mask]
    return swapped_eta


def _selected_var_by_trajectory(
    dataset: "xr.Dataset",
    var_name: str,
    trajectory: str,
    *,
    dtype: type = float,
) -> np.ndarray:
    """Return one selected-dataset variable sliced by trajectory."""
    return np.asarray(dataset[var_name].sel(trajectory=trajectory).values, dtype=dtype)


def _selected_scalar_by_trajectory(
    dataset: "xr.Dataset",
    var_name: str,
    trajectory: str,
) -> Any:
    """Return one scalar selected-dataset variable sliced by trajectory."""
    value = dataset[var_name].sel(trajectory=trajectory).values
    if np.asarray(value).shape == ():
        return np.asarray(value).item()
    return np.asarray(value).reshape(-1)[0].item()


def _speed_transform_from_selected_dataset(
    dataset: "xr.Dataset",
    trajectory: str,
) -> dict[str, Any]:
    """Return a saved speed transform for one model and trajectory."""
    model_name = str(dataset.attrs.get("model_name", "unknown"))
    transform = {
        "mode": str(_selected_scalar_by_trajectory(dataset, "speed_feature_mode", trajectory)),
        "basis": str(_selected_scalar_by_trajectory(dataset, "speed_basis", trajectory)),
        "n_features": int(_selected_scalar_by_trajectory(dataset, "n_speed_features", trajectory)),
        "spline_order": float(_selected_scalar_by_trajectory(dataset, "speed_spline_order", trajectory)),
        "bounds": _selected_var_by_trajectory(dataset, "speed_basis_bounds", trajectory),
        "reference_value": float(
            _selected_scalar_by_trajectory(dataset, "speed_reference_value", trajectory)
        ),
        "mean": float(_selected_scalar_by_trajectory(dataset, "speed_mean", trajectory)),
        "std": float(_selected_scalar_by_trajectory(dataset, "speed_std", trajectory)),
    }
    mode = str(transform["mode"])
    n_features = int(transform["n_features"])
    if mode == "none" or n_features == 0:
        return transform
    if mode == "linear":
        if (
            not np.isfinite(transform["mean"])
            or not np.isfinite(transform["std"])
            or float(transform["std"]) <= 0.0
        ):
            raise ValueError(
                "Selected dark-light GLM file has invalid linear speed metadata "
                f"for model {model_name!r}, trajectory {trajectory!r}. "
                "Expected finite speed_mean and positive finite speed_std. "
                "Regenerate the selected dark_light_glm outputs with the updated writer."
            )
        return transform
    if mode == "bspline":
        bounds = np.asarray(transform["bounds"], dtype=float)
        if (
            bounds.shape != (2,)
            or not np.all(np.isfinite(bounds))
            or not np.isfinite(transform["spline_order"])
        ):
            raise ValueError(
                "Selected dark-light GLM file has invalid bspline speed metadata "
                f"for model {model_name!r}, trajectory {trajectory!r}."
            )
        return transform
    raise ValueError(
        "Selected dark-light GLM file has unsupported speed_feature_mode "
        f"{mode!r} for model {model_name!r}, trajectory {trajectory!r}."
    )


def _place_basis(values: np.ndarray, *, n_splines: int, spline_order: int) -> np.ndarray:
    """Evaluate the selected task-progression B-spline basis."""
    basis = BSplineEval(
        n_basis_funcs=int(n_splines),
        order=int(spline_order),
        bounds=POS_BOUNDS,
    )
    values = np.clip(np.asarray(values, dtype=float).reshape(-1), *POS_BOUNDS)
    return np.asarray(basis.compute_features(values), dtype=float)


def _selected_speed_effect(
    dataset: "xr.Dataset",
    trajectory: str,
    values: np.ndarray | None,
    *,
    n_rows: int,
    n_units: int,
    allow_missing_speed: bool = False,
) -> np.ndarray:
    """Return speed eta from selected per-trajectory speed metadata."""
    transform = _speed_transform_from_selected_dataset(dataset, trajectory)
    if str(transform["mode"]) == "none" or int(transform["n_features"]) == 0:
        return np.zeros((int(n_rows), int(n_units)), dtype=float)
    if values is None:
        if allow_missing_speed:
            return np.zeros((int(n_rows), int(n_units)), dtype=float)
        raise ValueError(
            f"Selected model uses speed, but held-out speed is unavailable for {trajectory!r}."
        )
    speed_design = _transform_speed_with_feature_transform(values, transform)
    coef_speed_basis = _selected_var_by_trajectory(
        dataset,
        "coef_speed_basis",
        trajectory,
    ).reshape(speed_design.shape[1], n_units)
    return speed_design @ coef_speed_basis


def predict_selected_light_eta(
    *,
    dataset: "xr.Dataset",
    model_name: str,
    trajectory: str,
    p_eval: np.ndarray,
    speed_values: np.ndarray | None,
    n_splines: int,
    spline_order: int,
    segment_edges: np.ndarray,
    allow_missing_speed: bool = False,
) -> dict[str, np.ndarray]:
    """Predict selected-model light eta and the swappable component."""
    p_eval = np.asarray(p_eval, dtype=float).reshape(-1)
    n_units = int(np.asarray(dataset.coords["unit"].values).size)
    intercept = _selected_var_by_trajectory(dataset, "coef_intercept", trajectory)
    coef_light = _selected_var_by_trajectory(dataset, "coef_light_offset", trajectory)
    speed_eta = _selected_speed_effect(
        dataset,
        trajectory,
        speed_values,
        n_rows=p_eval.size,
        n_units=n_units,
        allow_missing_speed=allow_missing_speed,
    )
    place_basis = _place_basis(
        p_eval,
        n_splines=n_splines,
        spline_order=spline_order,
    )

    if model_name == "visual":
        coef_place_light = _selected_var_by_trajectory(
            dataset,
            "coef_place_light",
            trajectory,
        )
        light_component = place_basis @ coef_place_light
        return {
            "light_eta": (
                intercept[None, :]
                + coef_light[None, :]
                + light_component
                + speed_eta
            ),
            "light_component": light_component,
            "light_offset": coef_light,
            "basis": place_basis,
        }

    coef_place_dark = _selected_var_by_trajectory(dataset, "coef_place_dark", trajectory)
    dark_component = place_basis @ coef_place_dark
    if model_name == "task_segment_bump":
        gain_basis = _segment_center_raised_cosine_basis(
            p_eval,
            segment_edges,
            pos_bounds=POS_BOUNDS,
            overlap_frac=DEFAULT_SEGMENT_OVERLAP_FRAC,
        )
        coef_gain = _selected_var_by_trajectory(
            dataset,
            "coef_segment_bump_gain",
            trajectory,
        )
    elif model_name == "task_segment_scalar":
        gain_basis, _ = _segment_onehot_basis(
            p_eval,
            segment_edges,
            pos_bounds=POS_BOUNDS,
        )
        coef_gain = _selected_var_by_trajectory(
            dataset,
            "coef_segment_scalar_gain",
            trajectory,
        )
    elif model_name == "task_dense_gain":
        gain_basis = _place_basis(
            p_eval,
            n_splines=n_splines,
            spline_order=spline_order,
        )
        coef_gain = _selected_var_by_trajectory(dataset, "coef_gain_spline", trajectory)
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    gain_component = gain_basis @ coef_gain
    return {
        "light_eta": (
            intercept[None, :]
            + dark_component
            + coef_light[None, :]
            + gain_component
            + speed_eta
        ),
        "light_component": gain_component,
        "light_offset": coef_light,
        "gain_basis": gain_basis,
        "coef_gain": coef_gain,
    }


def build_selected_swapped_eta(
    *,
    model_name: str,
    own_prediction: dict[str, np.ndarray],
    paired_dataset: "xr.Dataset",
    paired_trajectory: str,
    p_eval: np.ndarray,
    segment_edges: np.ndarray,
    swap_segment_index: int,
    swap_mask: np.ndarray,
    n_splines: int,
    spline_order: int,
    swap_light_offset: bool = False,
) -> np.ndarray:
    """Return light eta after swapping only the local model component."""
    if model_name == "visual":
        paired_prediction = predict_selected_light_eta(
            dataset=paired_dataset,
            model_name=model_name,
            trajectory=paired_trajectory,
            p_eval=p_eval,
            speed_values=None,
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
            allow_missing_speed=True,
        )
        return build_visual_swapped_light_eta(
            own_light_eta=own_prediction["light_eta"],
            own_light_place=own_prediction["light_component"],
            paired_light_place=paired_prediction["light_component"],
            own_light_offset=own_prediction["light_offset"],
            paired_light_offset=paired_prediction["light_offset"],
            swap_mask=swap_mask,
            swap_light_offset=swap_light_offset,
        )

    if model_name in {"task_segment_bump", "task_segment_scalar"}:
        coef_name = {
            "task_segment_bump": "coef_segment_bump_gain",
            "task_segment_scalar": "coef_segment_scalar_gain",
        }[model_name]
        paired_coef_gain = _selected_var_by_trajectory(
            paired_dataset,
            coef_name,
            paired_trajectory,
        )
        paired_light_offset = _selected_var_by_trajectory(
            paired_dataset,
            "coef_light_offset",
            paired_trajectory,
        )
        return build_task_swapped_light_eta(
            own_light_eta=own_prediction["light_eta"],
            own_gain_basis=own_prediction["gain_basis"],
            own_coef_gain=own_prediction["coef_gain"],
            paired_coef_gain=paired_coef_gain,
            own_light_offset=own_prediction["light_offset"],
            paired_light_offset=paired_light_offset,
            swap_segment_index=swap_segment_index,
            swap_mask=swap_mask,
            swap_light_offset=swap_light_offset,
        )

    if model_name == "task_dense_gain":
        paired_prediction = predict_selected_light_eta(
            dataset=paired_dataset,
            model_name=model_name,
            trajectory=paired_trajectory,
            p_eval=p_eval,
            speed_values=None,
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
            allow_missing_speed=True,
        )
        return build_visual_swapped_light_eta(
            own_light_eta=own_prediction["light_eta"],
            own_light_place=own_prediction["light_component"],
            paired_light_place=paired_prediction["light_component"],
            own_light_offset=own_prediction["light_offset"],
            paired_light_offset=paired_prediction["light_offset"],
            swap_mask=swap_mask,
            swap_light_offset=swap_light_offset,
        )

    raise ValueError(f"Unknown model_name: {model_name!r}")


def evaluate_selected_model_on_test_epoch(
    *,
    model_name: str,
    datasets_by_model: dict[str, "xr.Dataset"],
    test_inputs_by_traj: dict[str, dict[str, Any]],
    segment_edges: np.ndarray,
    bin_size_s: float,
    n_splines: int,
    spline_order: int,
    swap_light_offset: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute raw held-out swap metrics for one selected model."""
    dataset = datasets_by_model[model_name]
    results: dict[str, dict[str, Any]] = {}
    for trajectory in TRAJECTORY_TYPES:
        swap_info = SWAP_CONFIG[trajectory]
        paired_trajectory = str(swap_info["source_trajectory"])
        swap_segment_index = int(swap_info["segment_index"])
        test_inputs = test_inputs_by_traj[trajectory]
        p_test = np.asarray(test_inputs["p"], dtype=float)
        y_test = np.asarray(test_inputs["y"], dtype=float)
        swap_mask = _segment_mask(p_test, segment_edges, swap_segment_index)

        own_prediction = predict_selected_light_eta(
            dataset=dataset,
            model_name=model_name,
            trajectory=trajectory,
            p_eval=p_test,
            speed_values=test_inputs["v"],
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
        )
        swapped_eta = build_selected_swapped_eta(
            model_name=model_name,
            own_prediction=own_prediction,
            paired_dataset=dataset,
            paired_trajectory=paired_trajectory,
            p_eval=p_test,
            segment_edges=segment_edges,
            swap_segment_index=swap_segment_index,
            swap_mask=swap_mask,
            n_splines=n_splines,
            spline_order=spline_order,
            swap_light_offset=swap_light_offset,
        )

        unswapped_count = np.exp(own_prediction["light_eta"])
        swapped_count = np.exp(swapped_eta)
        grid = np.asarray(dataset.coords["tp_grid"].values, dtype=float)
        grid_mask = _segment_mask(grid, segment_edges, swap_segment_index)
        own_grid_hz = _selected_var_by_trajectory(
            dataset,
            "train_light_hz_grid",
            trajectory,
        )
        own_grid_eta = np.log(np.clip(own_grid_hz * float(bin_size_s), 1e-12, None))
        own_grid_prediction = predict_selected_light_eta(
            dataset=dataset,
            model_name=model_name,
            trajectory=trajectory,
            p_eval=grid,
            speed_values=None,
            n_splines=n_splines,
            spline_order=spline_order,
            segment_edges=segment_edges,
            allow_missing_speed=True,
        )
        own_grid_prediction["light_eta"] = own_grid_eta
        swapped_grid_eta = build_selected_swapped_eta(
            model_name=model_name,
            own_prediction=own_grid_prediction,
            paired_dataset=dataset,
            paired_trajectory=paired_trajectory,
            p_eval=grid,
            segment_edges=segment_edges,
            swap_segment_index=swap_segment_index,
            swap_mask=grid_mask,
            n_splines=n_splines,
            spline_order=spline_order,
            swap_light_offset=swap_light_offset,
        )

        y_segment = y_test[swap_mask] if np.any(swap_mask) else y_test[:0]
        unswapped_segment_count = (
            unswapped_count[swap_mask] if np.any(swap_mask) else unswapped_count[:0]
        )
        swapped_segment_count = (
            swapped_count[swap_mask] if np.any(swap_mask) else swapped_count[:0]
        )

        results[trajectory] = {
            "unit_ids": np.asarray(dataset.coords["unit"].values),
            "swap_source_trajectory": paired_trajectory,
            "swap_segment_index_1based": int(swap_segment_index + 1),
            "swap_segment_start": float(segment_edges[swap_segment_index]),
            "swap_segment_end": float(segment_edges[swap_segment_index + 1]),
            "dark_hz_grid": _selected_var_by_trajectory(
                dataset,
                "dark_hz_grid",
                trajectory,
            ),
            "train_light_hz_grid": own_grid_hz,
            "test_light_unswapped_hz_grid": own_grid_hz,
            "test_light_swapped_hz_grid": np.exp(swapped_grid_eta) / float(bin_size_s),
            "test_light_full_unswapped_metrics": summarize_raw_poisson_metrics(
                y_test,
                unswapped_count,
            ),
            "test_light_full_swapped_metrics": summarize_raw_poisson_metrics(
                y_test,
                swapped_count,
            ),
            "test_light_swapped_segment_unswapped_metrics": summarize_raw_poisson_metrics(
                y_segment,
                unswapped_segment_count,
            ),
            "test_light_swapped_segment_swapped_metrics": summarize_raw_poisson_metrics(
                y_segment,
                swapped_segment_count,
            ),
            "test_light_full_n_bins": int(y_test.shape[0]),
            "test_light_swapped_segment_n_bins": int(y_segment.shape[0]),
        }
    return results


def fit_visual_model_for_trajectory(
    *,
    epoch_inputs: dict[str, dict[str, Any]],
    ridge: float,
    n_splines: int,
    spline_order: int,
    include_light_offset_term: bool = True,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    """Fit the separate dark/light visual model for one trajectory."""
    _require_nemos()
    unit_ids = _validate_unit_ids(epoch_inputs)
    train_dark = epoch_inputs["train_dark"]
    train_light = epoch_inputs["train_light"]
    validation_light = epoch_inputs["validation_light"]
    test_light = epoch_inputs["test_light"]

    y_dark = np.asarray(train_dark["y"], dtype=float)
    y_light = np.asarray(train_light["y"], dtype=float)
    y_train = np.concatenate([y_dark, y_light], axis=0)
    p_train = np.concatenate([train_dark["p"], train_light["p"]], axis=0)
    l_train = np.concatenate(
        [
            np.zeros(train_dark["p"].shape[0], dtype=float),
            np.ones(train_light["p"].shape[0], dtype=float),
        ],
        axis=0,
    )

    if train_dark["v"] is None:
        speed_transform = _empty_speed_feature_transform()
        v_train = _empty_speed_design(y_train.shape[0])
        v_dark = _empty_speed_design(y_dark.shape[0])
        v_light = _empty_speed_design(y_light.shape[0])
        v_validation = _empty_speed_design(validation_light["y"].shape[0])
        v_test = _empty_speed_design(test_light["y"].shape[0])
    else:
        train_speed = np.concatenate([train_dark["v"], train_light["v"]], axis=0)
        speed_transform = _fit_speed_feature_transform(
            train_speed,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
        v_train = _transform_speed_with_feature_transform(train_speed, speed_transform)
        v_dark = _transform_speed_with_feature_transform(train_dark["v"], speed_transform)
        v_light = _transform_speed_with_feature_transform(train_light["v"], speed_transform)
        v_validation = _transform_speed_with_feature_transform(
            validation_light["v"],
            speed_transform,
        )
        v_test = _transform_speed_with_feature_transform(test_light["v"], speed_transform)

    basis = BSplineEval(n_basis_funcs=n_splines, order=spline_order, bounds=POS_BOUNDS)
    place_basis_train = np.asarray(basis.compute_features(p_train), dtype=float)
    n_place_basis = int(place_basis_train.shape[1])
    basis_dark = place_basis_train * (1.0 - l_train[:, None])
    basis_light = place_basis_train * l_train[:, None]
    if include_light_offset_term:
        x_train = np.concatenate(
            [basis_dark, basis_light, l_train[:, None], v_train],
            axis=1,
        )
    else:
        x_train = np.concatenate([basis_dark, basis_light, v_train], axis=1)

    model = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model.fit(x_train, y_train)

    coef = _coef_feat_by_unit(model, n_features=x_train.shape[1])
    n_speed_features = int(speed_transform["n_features"])
    coef_place_dark = coef[:n_place_basis]
    coef_place_light = coef[n_place_basis : (2 * n_place_basis)]
    light_index = 2 * n_place_basis
    if include_light_offset_term:
        coef_light = coef[light_index]
        speed_start = light_index + 1
    else:
        coef_light = np.zeros((unit_ids.size,), dtype=float)
        speed_start = light_index
    coef_speed_basis = coef[speed_start : (speed_start + n_speed_features)]
    intercept = np.asarray(model.intercept_).reshape(-1)

    pred_dark = _predict_visual_components(
        p_eval=train_dark["p"],
        speed_design=v_dark,
        basis=basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_place_light=coef_place_light,
        coef_speed_basis=coef_speed_basis,
    )
    pred_light = _predict_visual_components(
        p_eval=train_light["p"],
        speed_design=v_light,
        basis=basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_place_light=coef_place_light,
        coef_speed_basis=coef_speed_basis,
    )
    pred_validation = _predict_visual_components(
        p_eval=validation_light["p"],
        speed_design=v_validation,
        basis=basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_place_light=coef_place_light,
        coef_speed_basis=coef_speed_basis,
    )
    pred_test = _predict_visual_components(
        p_eval=test_light["p"],
        speed_design=v_test,
        basis=basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_place_light=coef_place_light,
        coef_speed_basis=coef_speed_basis,
    )

    speed_outputs = _format_speed_outputs(
        transform=speed_transform,
        coef_speed_basis=coef_speed_basis,
        n_units=unit_ids.size,
    )
    return {
        "model_name": "visual",
        "unit_ids": unit_ids,
        "epoch_inputs": epoch_inputs,
        "basis": basis,
        "has_light_offset_term": bool(include_light_offset_term),
        "intercept": intercept,
        "coef_light": coef_light,
        "coef_place_dark": coef_place_dark,
        "coef_place_light": coef_place_light,
        "coef_speed_basis": coef_speed_basis,
        "speed_transform": speed_transform,
        "speed_outputs": speed_outputs,
        "pred_train_dark": pred_dark,
        "pred_train_light": pred_light,
        "pred_validation_light": pred_validation,
        "pred_test_light": pred_test,
    }


def fit_task_segment_bump_model_for_trajectory(
    *,
    epoch_inputs: dict[str, dict[str, Any]],
    ridge: float,
    n_splines: int,
    spline_order: int,
    segment_edges: np.ndarray,
    include_light_offset_term: bool = True,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    """Fit the segment-bump gain model for one trajectory."""
    _require_nemos()
    unit_ids = _validate_unit_ids(epoch_inputs)
    train_dark = epoch_inputs["train_dark"]
    train_light = epoch_inputs["train_light"]
    validation_light = epoch_inputs["validation_light"]
    test_light = epoch_inputs["test_light"]

    y_dark = np.asarray(train_dark["y"], dtype=float)
    y_light = np.asarray(train_light["y"], dtype=float)
    y_train = np.concatenate([y_dark, y_light], axis=0)
    p_train = np.concatenate([train_dark["p"], train_light["p"]], axis=0)
    l_train = np.concatenate(
        [
            np.zeros(train_dark["p"].shape[0], dtype=float),
            np.ones(train_light["p"].shape[0], dtype=float),
        ],
        axis=0,
    )

    if train_dark["v"] is None:
        speed_transform = _empty_speed_feature_transform()
        v_train = _empty_speed_design(y_train.shape[0])
        v_dark = _empty_speed_design(y_dark.shape[0])
        v_light = _empty_speed_design(y_light.shape[0])
        v_validation = _empty_speed_design(validation_light["y"].shape[0])
        v_test = _empty_speed_design(test_light["y"].shape[0])
    else:
        train_speed = np.concatenate([train_dark["v"], train_light["v"]], axis=0)
        speed_transform = _fit_speed_feature_transform(
            train_speed,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
        v_train = _transform_speed_with_feature_transform(train_speed, speed_transform)
        v_dark = _transform_speed_with_feature_transform(train_dark["v"], speed_transform)
        v_light = _transform_speed_with_feature_transform(train_light["v"], speed_transform)
        v_validation = _transform_speed_with_feature_transform(
            validation_light["v"],
            speed_transform,
        )
        v_test = _transform_speed_with_feature_transform(test_light["v"], speed_transform)

    dark_basis = BSplineEval(
        n_basis_funcs=n_splines,
        order=spline_order,
        bounds=POS_BOUNDS,
    )
    dark_basis_train = np.asarray(dark_basis.compute_features(p_train), dtype=float)
    gain_basis_train = _segment_center_raised_cosine_basis(
        p_train,
        segment_edges,
        pos_bounds=POS_BOUNDS,
        overlap_frac=DEFAULT_SEGMENT_OVERLAP_FRAC,
    )
    n_dark_basis = int(dark_basis_train.shape[1])
    n_gain_basis = int(gain_basis_train.shape[1])
    if include_light_offset_term:
        x_train = np.concatenate(
            [
                dark_basis_train,
                l_train[:, None],
                gain_basis_train * l_train[:, None],
                v_train,
            ],
            axis=1,
        )
    else:
        x_train = np.concatenate(
            [dark_basis_train, gain_basis_train * l_train[:, None], v_train],
            axis=1,
        )

    model = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model.fit(x_train, y_train)

    coef = _coef_feat_by_unit(model, n_features=x_train.shape[1])
    n_speed_features = int(speed_transform["n_features"])
    coef_place_dark = coef[:n_dark_basis]
    if include_light_offset_term:
        coef_light = coef[n_dark_basis]
        gain_start = n_dark_basis + 1
    else:
        coef_light = np.zeros((unit_ids.size,), dtype=float)
        gain_start = n_dark_basis
    coef_gain = coef[gain_start : (gain_start + n_gain_basis)]
    speed_start = gain_start + n_gain_basis
    coef_speed_basis = coef[speed_start : (speed_start + n_speed_features)]
    intercept = np.asarray(model.intercept_).reshape(-1)

    pred_dark = _predict_task_components(
        p_eval=train_dark["p"],
        speed_design=v_dark,
        dark_basis=dark_basis,
        segment_edges=segment_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain=coef_gain,
        coef_speed_basis=coef_speed_basis,
    )
    pred_light = _predict_task_components(
        p_eval=train_light["p"],
        speed_design=v_light,
        dark_basis=dark_basis,
        segment_edges=segment_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain=coef_gain,
        coef_speed_basis=coef_speed_basis,
    )
    pred_validation = _predict_task_components(
        p_eval=validation_light["p"],
        speed_design=v_validation,
        dark_basis=dark_basis,
        segment_edges=segment_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain=coef_gain,
        coef_speed_basis=coef_speed_basis,
    )
    pred_test = _predict_task_components(
        p_eval=test_light["p"],
        speed_design=v_test,
        dark_basis=dark_basis,
        segment_edges=segment_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain=coef_gain,
        coef_speed_basis=coef_speed_basis,
    )

    speed_outputs = _format_speed_outputs(
        transform=speed_transform,
        coef_speed_basis=coef_speed_basis,
        n_units=unit_ids.size,
    )
    return {
        "model_name": "task_segment_bump",
        "unit_ids": unit_ids,
        "epoch_inputs": epoch_inputs,
        "basis": dark_basis,
        "segment_edges": np.asarray(segment_edges, dtype=float),
        "has_light_offset_term": bool(include_light_offset_term),
        "intercept": intercept,
        "coef_light": coef_light,
        "coef_place_dark": coef_place_dark,
        "coef_segment_bump_gain": coef_gain,
        "coef_speed_basis": coef_speed_basis,
        "speed_transform": speed_transform,
        "speed_outputs": speed_outputs,
        "pred_train_dark": pred_dark,
        "pred_train_light": pred_light,
        "pred_validation_light": pred_validation,
        "pred_test_light": pred_test,
    }


def fit_task_segment_scalar_model_for_trajectory(
    *,
    epoch_inputs: dict[str, dict[str, Any]],
    ridge: float,
    n_splines: int,
    spline_order: int,
    segment_edges: np.ndarray,
    include_light_offset_term: bool = True,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    """Fit the segment-scalar gain model for one trajectory."""
    _require_nemos()
    unit_ids = _validate_unit_ids(epoch_inputs)
    train_dark = epoch_inputs["train_dark"]
    train_light = epoch_inputs["train_light"]
    validation_light = epoch_inputs["validation_light"]
    test_light = epoch_inputs["test_light"]

    y_dark = np.asarray(train_dark["y"], dtype=float)
    y_light = np.asarray(train_light["y"], dtype=float)
    y_train = np.concatenate([y_dark, y_light], axis=0)
    p_train = np.concatenate([train_dark["p"], train_light["p"]], axis=0)
    l_train = np.concatenate(
        [
            np.zeros(train_dark["p"].shape[0], dtype=float),
            np.ones(train_light["p"].shape[0], dtype=float),
        ],
        axis=0,
    )

    if train_dark["v"] is None:
        speed_transform = _empty_speed_feature_transform()
        v_train = _empty_speed_design(y_train.shape[0])
        v_dark = _empty_speed_design(y_dark.shape[0])
        v_light = _empty_speed_design(y_light.shape[0])
        v_validation = _empty_speed_design(validation_light["y"].shape[0])
        v_test = _empty_speed_design(test_light["y"].shape[0])
    else:
        train_speed = np.concatenate([train_dark["v"], train_light["v"]], axis=0)
        speed_transform = _fit_speed_feature_transform(
            train_speed,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
        v_train = _transform_speed_with_feature_transform(train_speed, speed_transform)
        v_dark = _transform_speed_with_feature_transform(train_dark["v"], speed_transform)
        v_light = _transform_speed_with_feature_transform(train_light["v"], speed_transform)
        v_validation = _transform_speed_with_feature_transform(
            validation_light["v"],
            speed_transform,
        )
        v_test = _transform_speed_with_feature_transform(test_light["v"], speed_transform)

    dark_basis = BSplineEval(
        n_basis_funcs=n_splines,
        order=spline_order,
        bounds=POS_BOUNDS,
    )
    dark_basis_train = np.asarray(dark_basis.compute_features(p_train), dtype=float)
    gain_basis_train, gain_edges = _segment_onehot_basis(
        p_train,
        segment_edges,
        pos_bounds=POS_BOUNDS,
    )
    n_dark_basis = int(dark_basis_train.shape[1])
    n_gain_basis = int(gain_basis_train.shape[1])
    if include_light_offset_term:
        x_train = np.concatenate(
            [
                dark_basis_train,
                l_train[:, None],
                gain_basis_train * l_train[:, None],
                v_train,
            ],
            axis=1,
        )
    else:
        x_train = np.concatenate(
            [dark_basis_train, gain_basis_train * l_train[:, None], v_train],
            axis=1,
        )

    model = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model.fit(x_train, y_train)

    coef = _coef_feat_by_unit(model, n_features=x_train.shape[1])
    n_speed_features = int(speed_transform["n_features"])
    coef_place_dark = coef[:n_dark_basis]
    if include_light_offset_term:
        coef_light = coef[n_dark_basis]
        gain_start = n_dark_basis + 1
    else:
        coef_light = np.zeros((unit_ids.size,), dtype=float)
        gain_start = n_dark_basis
    coef_segment_scalar_gain = coef[gain_start : (gain_start + n_gain_basis)]
    speed_start = gain_start + n_gain_basis
    coef_speed_basis = coef[speed_start : (speed_start + n_speed_features)]
    intercept = np.asarray(model.intercept_).reshape(-1)

    pred_dark = _predict_task_scalar_components(
        p_eval=train_dark["p"],
        speed_design=v_dark,
        dark_basis=dark_basis,
        segment_edges=gain_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_segment_scalar_gain=coef_segment_scalar_gain,
        coef_speed_basis=coef_speed_basis,
    )
    pred_light = _predict_task_scalar_components(
        p_eval=train_light["p"],
        speed_design=v_light,
        dark_basis=dark_basis,
        segment_edges=gain_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_segment_scalar_gain=coef_segment_scalar_gain,
        coef_speed_basis=coef_speed_basis,
    )
    pred_validation = _predict_task_scalar_components(
        p_eval=validation_light["p"],
        speed_design=v_validation,
        dark_basis=dark_basis,
        segment_edges=gain_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_segment_scalar_gain=coef_segment_scalar_gain,
        coef_speed_basis=coef_speed_basis,
    )
    pred_test = _predict_task_scalar_components(
        p_eval=test_light["p"],
        speed_design=v_test,
        dark_basis=dark_basis,
        segment_edges=gain_edges,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_segment_scalar_gain=coef_segment_scalar_gain,
        coef_speed_basis=coef_speed_basis,
    )

    speed_outputs = _format_speed_outputs(
        transform=speed_transform,
        coef_speed_basis=coef_speed_basis,
        n_units=unit_ids.size,
    )
    return {
        "model_name": "task_segment_scalar",
        "unit_ids": unit_ids,
        "epoch_inputs": epoch_inputs,
        "basis": dark_basis,
        "segment_edges": np.asarray(gain_edges, dtype=float),
        "has_light_offset_term": bool(include_light_offset_term),
        "intercept": intercept,
        "coef_light": coef_light,
        "coef_place_dark": coef_place_dark,
        "coef_segment_scalar_gain": coef_segment_scalar_gain,
        "coef_speed_basis": coef_speed_basis,
        "speed_transform": speed_transform,
        "speed_outputs": speed_outputs,
        "pred_train_dark": pred_dark,
        "pred_train_light": pred_light,
        "pred_validation_light": pred_validation,
        "pred_test_light": pred_test,
    }


def fit_task_dense_gain_model_for_trajectory(
    *,
    epoch_inputs: dict[str, dict[str, Any]],
    ridge: float,
    n_splines: int,
    spline_order: int,
    include_light_offset_term: bool = True,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    """Fit the dense gain-spline modulation model for one trajectory."""
    _require_nemos()
    unit_ids = _validate_unit_ids(epoch_inputs)
    train_dark = epoch_inputs["train_dark"]
    train_light = epoch_inputs["train_light"]
    validation_light = epoch_inputs["validation_light"]
    test_light = epoch_inputs["test_light"]

    y_dark = np.asarray(train_dark["y"], dtype=float)
    y_light = np.asarray(train_light["y"], dtype=float)
    y_train = np.concatenate([y_dark, y_light], axis=0)
    p_train = np.concatenate([train_dark["p"], train_light["p"]], axis=0)
    l_train = np.concatenate(
        [
            np.zeros(train_dark["p"].shape[0], dtype=float),
            np.ones(train_light["p"].shape[0], dtype=float),
        ],
        axis=0,
    )

    if train_dark["v"] is None:
        speed_transform = _empty_speed_feature_transform()
        v_train = _empty_speed_design(y_train.shape[0])
        v_dark = _empty_speed_design(y_dark.shape[0])
        v_light = _empty_speed_design(y_light.shape[0])
        v_validation = _empty_speed_design(validation_light["y"].shape[0])
        v_test = _empty_speed_design(test_light["y"].shape[0])
    else:
        train_speed = np.concatenate([train_dark["v"], train_light["v"]], axis=0)
        speed_transform = _fit_speed_feature_transform(
            train_speed,
            speed_feature_mode=speed_feature_mode,
            n_splines_speed=n_splines_speed,
            spline_order_speed=spline_order_speed,
            speed_bounds=speed_bounds,
        )
        v_train = _transform_speed_with_feature_transform(train_speed, speed_transform)
        v_dark = _transform_speed_with_feature_transform(train_dark["v"], speed_transform)
        v_light = _transform_speed_with_feature_transform(train_light["v"], speed_transform)
        v_validation = _transform_speed_with_feature_transform(
            validation_light["v"],
            speed_transform,
        )
        v_test = _transform_speed_with_feature_transform(test_light["v"], speed_transform)

    dark_basis = BSplineEval(
        n_basis_funcs=n_splines,
        order=spline_order,
        bounds=POS_BOUNDS,
    )
    gain_basis = BSplineEval(
        n_basis_funcs=n_splines,
        order=spline_order,
        bounds=POS_BOUNDS,
    )
    dark_basis_train = np.asarray(dark_basis.compute_features(p_train), dtype=float)
    gain_basis_train = np.asarray(gain_basis.compute_features(p_train), dtype=float)
    n_dark_basis = int(dark_basis_train.shape[1])
    n_gain_basis = int(gain_basis_train.shape[1])
    if include_light_offset_term:
        x_train = np.concatenate(
            [
                dark_basis_train,
                l_train[:, None],
                gain_basis_train * l_train[:, None],
                v_train,
            ],
            axis=1,
        )
    else:
        x_train = np.concatenate(
            [dark_basis_train, gain_basis_train * l_train[:, None], v_train],
            axis=1,
        )

    model = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model.fit(x_train, y_train)

    coef = _coef_feat_by_unit(model, n_features=x_train.shape[1])
    n_speed_features = int(speed_transform["n_features"])
    coef_place_dark = coef[:n_dark_basis]
    if include_light_offset_term:
        coef_light = coef[n_dark_basis]
        gain_start = n_dark_basis + 1
    else:
        coef_light = np.zeros((unit_ids.size,), dtype=float)
        gain_start = n_dark_basis
    coef_gain_spline = coef[gain_start : (gain_start + n_gain_basis)]
    speed_start = gain_start + n_gain_basis
    coef_speed_basis = coef[speed_start : (speed_start + n_speed_features)]
    intercept = np.asarray(model.intercept_).reshape(-1)

    pred_dark = _predict_task_dense_gain_components(
        p_eval=train_dark["p"],
        speed_design=v_dark,
        dark_basis=dark_basis,
        gain_basis=gain_basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain_spline=coef_gain_spline,
        coef_speed_basis=coef_speed_basis,
    )
    pred_light = _predict_task_dense_gain_components(
        p_eval=train_light["p"],
        speed_design=v_light,
        dark_basis=dark_basis,
        gain_basis=gain_basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain_spline=coef_gain_spline,
        coef_speed_basis=coef_speed_basis,
    )
    pred_validation = _predict_task_dense_gain_components(
        p_eval=validation_light["p"],
        speed_design=v_validation,
        dark_basis=dark_basis,
        gain_basis=gain_basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain_spline=coef_gain_spline,
        coef_speed_basis=coef_speed_basis,
    )
    pred_test = _predict_task_dense_gain_components(
        p_eval=test_light["p"],
        speed_design=v_test,
        dark_basis=dark_basis,
        gain_basis=gain_basis,
        intercept=intercept,
        coef_light=coef_light,
        coef_place_dark=coef_place_dark,
        coef_gain_spline=coef_gain_spline,
        coef_speed_basis=coef_speed_basis,
    )

    speed_outputs = _format_speed_outputs(
        transform=speed_transform,
        coef_speed_basis=coef_speed_basis,
        n_units=unit_ids.size,
    )
    return {
        "model_name": "task_dense_gain",
        "unit_ids": unit_ids,
        "epoch_inputs": epoch_inputs,
        "basis": dark_basis,
        "gain_basis": gain_basis,
        "has_light_offset_term": bool(include_light_offset_term),
        "intercept": intercept,
        "coef_light": coef_light,
        "coef_place_dark": coef_place_dark,
        "coef_gain_spline": coef_gain_spline,
        "coef_speed_basis": coef_speed_basis,
        "speed_transform": speed_transform,
        "speed_outputs": speed_outputs,
        "pred_train_dark": pred_dark,
        "pred_train_light": pred_light,
        "pred_validation_light": pred_validation,
        "pred_test_light": pred_test,
    }


def finalize_model_results_by_trajectory(
    *,
    model_name: str,
    fit_by_traj: dict[str, dict[str, Any]],
    segment_edges: np.ndarray,
    ridge: float,
    observed_bin_edges: np.ndarray,
    bin_size_s: float,
) -> dict[str, dict[str, Any]]:
    """Add swapped predictions, metrics, and summaries to one fitted model."""
    results: dict[str, dict[str, Any]] = {}
    for trajectory in TRAJECTORY_TYPES:
        swap_info = SWAP_CONFIG[trajectory]
        paired_trajectory = str(swap_info["source_trajectory"])
        swap_segment_index = int(swap_info["segment_index"])

        fit_result = fit_by_traj[trajectory]
        paired_result = fit_by_traj[paired_trajectory]
        epoch_inputs = fit_result["epoch_inputs"]
        train_dark = epoch_inputs["train_dark"]
        train_light = epoch_inputs["train_light"]
        validation_light = epoch_inputs["validation_light"]
        test_light = epoch_inputs["test_light"]
        validation_mask = _segment_mask(
            validation_light["p"],
            segment_edges,
            swap_segment_index,
        )
        test_mask = _segment_mask(test_light["p"], segment_edges, swap_segment_index)

        if model_name == "visual":
            paired_validation_light_place = np.asarray(
                paired_result["basis"].compute_features(validation_light["p"]),
                dtype=float,
            ) @ np.asarray(paired_result["coef_place_light"], dtype=float)
            paired_test_light_place = np.asarray(
                paired_result["basis"].compute_features(test_light["p"]),
                dtype=float,
            ) @ np.asarray(paired_result["coef_place_light"], dtype=float)
            swapped_validation_eta = build_visual_swapped_light_eta(
                own_light_eta=fit_result["pred_validation_light"]["light_eta"],
                own_light_place=fit_result["pred_validation_light"]["light_place"],
                paired_light_place=paired_validation_light_place,
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_mask=validation_mask,
            )
            swapped_test_eta = build_visual_swapped_light_eta(
                own_light_eta=fit_result["pred_test_light"]["light_eta"],
                own_light_place=fit_result["pred_test_light"]["light_place"],
                paired_light_place=paired_test_light_place,
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_mask=test_mask,
            )
            grid = np.linspace(POS_BOUNDS[0], POS_BOUNDS[1], 200)
            speed_ref = _speed_reference_design(fit_result["speed_transform"])
            grid_pred = _predict_visual_components(
                p_eval=grid,
                speed_design=np.repeat(speed_ref, grid.size, axis=0),
                basis=fit_result["basis"],
                intercept=fit_result["intercept"],
                coef_light=fit_result["coef_light"],
                coef_place_dark=fit_result["coef_place_dark"],
                coef_place_light=fit_result["coef_place_light"],
                coef_speed_basis=fit_result["coef_speed_basis"],
            )
            swap_grid_mask = _segment_mask(grid, segment_edges, swap_segment_index)
            paired_grid_pred = _predict_visual_components(
                p_eval=grid,
                speed_design=np.repeat(speed_ref, grid.size, axis=0),
                basis=paired_result["basis"],
                intercept=paired_result["intercept"],
                coef_light=paired_result["coef_light"],
                coef_place_dark=paired_result["coef_place_dark"],
                coef_place_light=paired_result["coef_place_light"],
                coef_speed_basis=paired_result["coef_speed_basis"],
            )
            swapped_grid_eta = build_visual_swapped_light_eta(
                own_light_eta=grid_pred["light_eta"],
                own_light_place=grid_pred["light_place"],
                paired_light_place=paired_grid_pred["light_place"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_mask=swap_grid_mask,
            )
        elif model_name == "task_segment_bump":
            swapped_validation_eta = build_task_swapped_light_eta(
                own_light_eta=fit_result["pred_validation_light"]["light_eta"],
                own_gain_basis=fit_result["pred_validation_light"]["gain_basis"],
                own_coef_gain=fit_result["coef_segment_bump_gain"],
                paired_coef_gain=paired_result["coef_segment_bump_gain"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_segment_index=swap_segment_index,
                swap_mask=validation_mask,
            )
            swapped_test_eta = build_task_swapped_light_eta(
                own_light_eta=fit_result["pred_test_light"]["light_eta"],
                own_gain_basis=fit_result["pred_test_light"]["gain_basis"],
                own_coef_gain=fit_result["coef_segment_bump_gain"],
                paired_coef_gain=paired_result["coef_segment_bump_gain"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_segment_index=swap_segment_index,
                swap_mask=test_mask,
            )
            grid = np.linspace(POS_BOUNDS[0], POS_BOUNDS[1], 200)
            speed_ref = _speed_reference_design(fit_result["speed_transform"])
            grid_pred = _predict_task_components(
                p_eval=grid,
                speed_design=np.repeat(speed_ref, grid.size, axis=0),
                dark_basis=fit_result["basis"],
                segment_edges=segment_edges,
                intercept=fit_result["intercept"],
                coef_light=fit_result["coef_light"],
                coef_place_dark=fit_result["coef_place_dark"],
                coef_gain=fit_result["coef_segment_bump_gain"],
                coef_speed_basis=fit_result["coef_speed_basis"],
            )
            swap_grid_mask = _segment_mask(grid, segment_edges, swap_segment_index)
            swapped_grid_eta = build_task_swapped_light_eta(
                own_light_eta=grid_pred["light_eta"],
                own_gain_basis=grid_pred["gain_basis"],
                own_coef_gain=fit_result["coef_segment_bump_gain"],
                paired_coef_gain=paired_result["coef_segment_bump_gain"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_segment_index=swap_segment_index,
                swap_mask=swap_grid_mask,
            )
        elif model_name == "task_segment_scalar":
            swapped_validation_eta = build_task_swapped_light_eta(
                own_light_eta=fit_result["pred_validation_light"]["light_eta"],
                own_gain_basis=fit_result["pred_validation_light"]["gain_basis"],
                own_coef_gain=fit_result["coef_segment_scalar_gain"],
                paired_coef_gain=paired_result["coef_segment_scalar_gain"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_segment_index=swap_segment_index,
                swap_mask=validation_mask,
            )
            swapped_test_eta = build_task_swapped_light_eta(
                own_light_eta=fit_result["pred_test_light"]["light_eta"],
                own_gain_basis=fit_result["pred_test_light"]["gain_basis"],
                own_coef_gain=fit_result["coef_segment_scalar_gain"],
                paired_coef_gain=paired_result["coef_segment_scalar_gain"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_segment_index=swap_segment_index,
                swap_mask=test_mask,
            )
            grid = np.linspace(POS_BOUNDS[0], POS_BOUNDS[1], 200)
            speed_ref = _speed_reference_design(fit_result["speed_transform"])
            grid_pred = _predict_task_scalar_components(
                p_eval=grid,
                speed_design=np.repeat(speed_ref, grid.size, axis=0),
                dark_basis=fit_result["basis"],
                segment_edges=segment_edges,
                intercept=fit_result["intercept"],
                coef_light=fit_result["coef_light"],
                coef_place_dark=fit_result["coef_place_dark"],
                coef_segment_scalar_gain=fit_result["coef_segment_scalar_gain"],
                coef_speed_basis=fit_result["coef_speed_basis"],
            )
            swap_grid_mask = _segment_mask(grid, segment_edges, swap_segment_index)
            swapped_grid_eta = build_task_swapped_light_eta(
                own_light_eta=grid_pred["light_eta"],
                own_gain_basis=grid_pred["gain_basis"],
                own_coef_gain=fit_result["coef_segment_scalar_gain"],
                paired_coef_gain=paired_result["coef_segment_scalar_gain"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_segment_index=swap_segment_index,
                swap_mask=swap_grid_mask,
            )
        elif model_name == "task_dense_gain":
            paired_validation_gain_part = np.asarray(
                paired_result["gain_basis"].compute_features(validation_light["p"]),
                dtype=float,
            ) @ np.asarray(paired_result["coef_gain_spline"], dtype=float)
            paired_test_gain_part = np.asarray(
                paired_result["gain_basis"].compute_features(test_light["p"]),
                dtype=float,
            ) @ np.asarray(paired_result["coef_gain_spline"], dtype=float)
            swapped_validation_eta = build_visual_swapped_light_eta(
                own_light_eta=fit_result["pred_validation_light"]["light_eta"],
                own_light_place=fit_result["pred_validation_light"]["gain_part"],
                paired_light_place=paired_validation_gain_part,
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_mask=validation_mask,
            )
            swapped_test_eta = build_visual_swapped_light_eta(
                own_light_eta=fit_result["pred_test_light"]["light_eta"],
                own_light_place=fit_result["pred_test_light"]["gain_part"],
                paired_light_place=paired_test_gain_part,
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_mask=test_mask,
            )
            grid = np.linspace(POS_BOUNDS[0], POS_BOUNDS[1], 200)
            speed_ref = _speed_reference_design(fit_result["speed_transform"])
            grid_pred = _predict_task_dense_gain_components(
                p_eval=grid,
                speed_design=np.repeat(speed_ref, grid.size, axis=0),
                dark_basis=fit_result["basis"],
                gain_basis=fit_result["gain_basis"],
                intercept=fit_result["intercept"],
                coef_light=fit_result["coef_light"],
                coef_place_dark=fit_result["coef_place_dark"],
                coef_gain_spline=fit_result["coef_gain_spline"],
                coef_speed_basis=fit_result["coef_speed_basis"],
            )
            swap_grid_mask = _segment_mask(grid, segment_edges, swap_segment_index)
            paired_grid_pred = _predict_task_dense_gain_components(
                p_eval=grid,
                speed_design=np.repeat(speed_ref, grid.size, axis=0),
                dark_basis=paired_result["basis"],
                gain_basis=paired_result["gain_basis"],
                intercept=paired_result["intercept"],
                coef_light=paired_result["coef_light"],
                coef_place_dark=paired_result["coef_place_dark"],
                coef_gain_spline=paired_result["coef_gain_spline"],
                coef_speed_basis=paired_result["coef_speed_basis"],
            )
            swapped_grid_eta = build_visual_swapped_light_eta(
                own_light_eta=grid_pred["light_eta"],
                own_light_place=grid_pred["gain_part"],
                paired_light_place=paired_grid_pred["gain_part"],
                own_light_offset=fit_result["coef_light"],
                paired_light_offset=paired_result["coef_light"],
                swap_mask=swap_grid_mask,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        # The Poisson GLMs are fit to spike counts per bin, so likelihood-based
        # metrics must use expected counts per bin rather than Hz.
        train_dark_count = np.exp(fit_result["pred_train_dark"]["dark_eta"])
        train_light_count = np.exp(fit_result["pred_train_light"]["light_eta"])
        validation_unswapped_count = np.exp(
            fit_result["pred_validation_light"]["light_eta"]
        )
        validation_swapped_count = np.exp(swapped_validation_eta)
        test_unswapped_count = np.exp(fit_result["pred_test_light"]["light_eta"])
        test_swapped_count = np.exp(swapped_test_eta)

        train_dark_null_fit = np.asarray(train_dark["y"], dtype=float)
        train_light_null_fit = np.asarray(train_light["y"], dtype=float)
        train_light_mask = _segment_mask(
            train_light["p"],
            segment_edges,
            swap_segment_index,
        )
        validation_y_scored = np.asarray(validation_light["y"], dtype=float)[validation_mask]
        validation_unswapped_count_scored = validation_unswapped_count[validation_mask]
        validation_swapped_count_scored = validation_swapped_count[validation_mask]
        test_y_scored = np.asarray(test_light["y"], dtype=float)[test_mask]
        test_unswapped_count_scored = test_unswapped_count[test_mask]
        test_swapped_count_scored = test_swapped_count[test_mask]
        heldout_null_fit = np.asarray(train_light["y"], dtype=float)[train_light_mask]

        results[trajectory] = {
            **fit_result,
            "swap_source_trajectory": paired_trajectory,
            "swap_segment_index_1based": int(swap_segment_index + 1),
            "swap_segment_start": float(segment_edges[swap_segment_index]),
            "swap_segment_end": float(segment_edges[swap_segment_index + 1]),
            "tp_grid": grid,
            "dark_hz_grid": np.exp(grid_pred["dark_eta"]) / bin_size_s,
            "train_light_hz_grid": np.exp(grid_pred["light_eta"]) / bin_size_s,
            "test_light_unswapped_hz_grid": np.exp(grid_pred["light_eta"]) / bin_size_s,
            "test_light_swapped_hz_grid": np.exp(swapped_grid_eta) / bin_size_s,
            "train_dark_metrics": summarize_poisson_metrics(
                train_dark["y"],
                train_dark_count,
                train_dark_null_fit,
            ),
            "train_light_metrics": summarize_poisson_metrics(
                train_light["y"],
                train_light_count,
                train_light_null_fit,
            ),
            "validation_light_unswapped_metrics": summarize_poisson_metrics(
                validation_y_scored,
                validation_unswapped_count_scored,
                heldout_null_fit,
            ),
            "validation_light_swapped_metrics": summarize_poisson_metrics(
                validation_y_scored,
                validation_swapped_count_scored,
                heldout_null_fit,
            ),
            "test_light_unswapped_metrics": summarize_poisson_metrics(
                test_y_scored,
                test_unswapped_count_scored,
                heldout_null_fit,
            ),
            "test_light_swapped_metrics": summarize_poisson_metrics(
                test_y_scored,
                test_swapped_count_scored,
                heldout_null_fit,
            ),
            "train_dark_null_rate_hz": (
                np.mean(np.asarray(train_dark["y"], dtype=float), axis=0) / bin_size_s
            ),
            "train_light_null_rate_hz": (
                np.mean(np.asarray(train_light["y"], dtype=float), axis=0) / bin_size_s
            ),
            "validation_light_null_rate_hz": (
                np.mean(np.asarray(heldout_null_fit, dtype=float), axis=0) / bin_size_s
                if heldout_null_fit.size > 0
                else np.full((fit_result["unit_ids"].size,), np.nan, dtype=float)
            ),
            "test_light_null_rate_hz": (
                np.mean(np.asarray(heldout_null_fit, dtype=float), axis=0) / bin_size_s
                if heldout_null_fit.size > 0
                else np.full((fit_result["unit_ids"].size,), np.nan, dtype=float)
            ),
            "train_dark_summary": build_observed_summary(
                train_dark["y"],
                train_dark["p"],
                observed_bin_edges,
                bin_size_s=bin_size_s,
            ),
            "train_light_summary": build_observed_summary(
                train_light["y"],
                train_light["p"],
                observed_bin_edges,
                bin_size_s=bin_size_s,
            ),
            "validation_light_summary": build_observed_summary(
                validation_light["y"],
                validation_light["p"],
                observed_bin_edges,
                bin_size_s=bin_size_s,
            ),
            "test_light_summary": build_observed_summary(
                test_light["y"],
                test_light["p"],
                observed_bin_edges,
                bin_size_s=bin_size_s,
            ),
        }
    return results


def fit_model_family(
    *,
    model_name: str,
    spikes: Any,
    trajectory_ep_by_epoch: dict[str, dict[str, Any]],
    tp_by_epoch: dict[str, dict[str, Any]],
    speed_by_epoch: dict[str, Any] | None,
    movement_by_run: dict[str, Any],
    heldout_split_by_traj: dict[str, dict[str, Any]],
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    ridge: float,
    n_splines: int,
    spline_order: int,
    segment_edges: np.ndarray,
    bin_size_s: float,
    unit_mask: np.ndarray,
    include_light_offset_term: bool = True,
    speed_feature_mode: str,
    n_splines_speed: int,
    spline_order_speed: int,
    speed_bounds: tuple[float, float] | None,
    observed_bin_edges: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Fit one model family across the four trajectories."""
    fit_by_traj: dict[str, dict[str, Any]] = {}
    for trajectory in TRAJECTORY_TYPES:
        validation_interval = heldout_split_by_traj[trajectory]["validation_interval"]
        test_interval = heldout_split_by_traj[trajectory]["test_interval"]
        validation_restrict = movement_by_run[light_test_epoch].intersect(validation_interval)
        test_restrict = movement_by_run[light_test_epoch].intersect(test_interval)
        epoch_inputs = {
            "train_dark": _prepare_epoch_inputs(
                spikes=spikes,
                trajectory_ep_by_epoch=trajectory_ep_by_epoch,
                tp_by_epoch=tp_by_epoch,
                speed_by_epoch=speed_by_epoch,
                traj_name=trajectory,
                epoch=dark_train_epoch,
                bin_size_s=bin_size_s,
                unit_mask=unit_mask,
                restrict_interval=movement_by_run[dark_train_epoch],
            ),
            "train_light": _prepare_epoch_inputs(
                spikes=spikes,
                trajectory_ep_by_epoch=trajectory_ep_by_epoch,
                tp_by_epoch=tp_by_epoch,
                speed_by_epoch=speed_by_epoch,
                traj_name=trajectory,
                epoch=light_train_epoch,
                bin_size_s=bin_size_s,
                unit_mask=unit_mask,
                restrict_interval=movement_by_run[light_train_epoch],
            ),
            "validation_light": _prepare_epoch_inputs(
                spikes=spikes,
                trajectory_ep_by_epoch=trajectory_ep_by_epoch,
                tp_by_epoch=tp_by_epoch,
                speed_by_epoch=speed_by_epoch,
                traj_name=trajectory,
                epoch=light_test_epoch,
                bin_size_s=bin_size_s,
                unit_mask=unit_mask,
                restrict_interval=validation_restrict,
            ),
            "test_light": _prepare_epoch_inputs(
                spikes=spikes,
                trajectory_ep_by_epoch=trajectory_ep_by_epoch,
                tp_by_epoch=tp_by_epoch,
                speed_by_epoch=speed_by_epoch,
                traj_name=trajectory,
                epoch=light_test_epoch,
                bin_size_s=bin_size_s,
                unit_mask=unit_mask,
                restrict_interval=test_restrict,
            ),
        }
        for split_name in ("train_dark", "train_light", "validation_light", "test_light"):
            if int(np.asarray(epoch_inputs[split_name]["y"]).shape[0]) == 0:
                raise ValueError(
                    f"No binned samples remained for trajectory {trajectory!r}, "
                    f"split {split_name!r}."
                )
        if model_name == "visual":
            fit_by_traj[trajectory] = fit_visual_model_for_trajectory(
                epoch_inputs=epoch_inputs,
                ridge=ridge,
                n_splines=n_splines,
                spline_order=spline_order,
                include_light_offset_term=include_light_offset_term,
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
            )
        elif model_name == "task_segment_bump":
            fit_by_traj[trajectory] = fit_task_segment_bump_model_for_trajectory(
                epoch_inputs=epoch_inputs,
                ridge=ridge,
                n_splines=n_splines,
                spline_order=spline_order,
                segment_edges=segment_edges,
                include_light_offset_term=include_light_offset_term,
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
            )
        elif model_name == "task_segment_scalar":
            fit_by_traj[trajectory] = fit_task_segment_scalar_model_for_trajectory(
                epoch_inputs=epoch_inputs,
                ridge=ridge,
                n_splines=n_splines,
                spline_order=spline_order,
                segment_edges=segment_edges,
                include_light_offset_term=include_light_offset_term,
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
            )
        elif model_name == "task_dense_gain":
            fit_by_traj[trajectory] = fit_task_dense_gain_model_for_trajectory(
                epoch_inputs=epoch_inputs,
                ridge=ridge,
                n_splines=n_splines,
                spline_order=spline_order,
                include_light_offset_term=include_light_offset_term,
                speed_feature_mode=speed_feature_mode,
                n_splines_speed=n_splines_speed,
                spline_order_speed=spline_order_speed,
                speed_bounds=speed_bounds,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    return finalize_model_results_by_trajectory(
        model_name=model_name,
        fit_by_traj=fit_by_traj,
        segment_edges=segment_edges,
        ridge=ridge,
        observed_bin_edges=observed_bin_edges,
        bin_size_s=bin_size_s,
    )


def build_validation_sweep(
    results_by_ridge: dict[float, dict[str, dict[str, Any]]],
    *,
    ridge_values: Sequence[float],
) -> dict[str, Any]:
    """Summarize validation swapped metrics across candidate ridges."""
    ordered_ridges = [float(ridge) for ridge in ridge_values]
    validation_bits = np.stack(
        [
            _stack_traj_metric(
                results_by_ridge[float(ridge)],
                "validation_light_swapped_metrics",
                "ll_bits_per_spike",
            )
            for ridge in ordered_ridges
        ],
        axis=0,
    )
    validation_devexp = np.stack(
        [
            _stack_traj_metric(
                results_by_ridge[float(ridge)],
                "validation_light_swapped_metrics",
                "deviance_explained",
            )
            for ridge in ordered_ridges
        ],
        axis=0,
    )

    pooled_validation_bits_median = np.full((len(ordered_ridges),), np.nan, dtype=float)
    pooled_validation_devexp_median = np.full((len(ordered_ridges),), np.nan, dtype=float)
    pooled_validation_count = np.zeros((len(ordered_ridges),), dtype=int)
    for ridge_index, ridge in enumerate(ordered_ridges):
        bits_values = validation_bits[ridge_index].reshape(-1)
        devexp_values = validation_devexp[ridge_index].reshape(-1)
        finite_bits = bits_values[np.isfinite(bits_values)]
        finite_devexp = devexp_values[np.isfinite(devexp_values)]
        pooled_validation_count[ridge_index] = int(finite_bits.size)
        if finite_bits.size > 0:
            pooled_validation_bits_median[ridge_index] = float(np.median(finite_bits))
        if finite_devexp.size > 0:
            pooled_validation_devexp_median[ridge_index] = float(np.median(finite_devexp))

    selected_ridge = choose_best_ridge(
        ordered_ridges,
        {
            float(ridge): float(pooled_validation_bits_median[ridge_index])
            for ridge_index, ridge in enumerate(ordered_ridges)
        },
    )
    return {
        "candidate_ridge": np.asarray(ordered_ridges, dtype=float),
        "validation_swapped_ll_bits_per_spike": validation_bits,
        "validation_swapped_deviance_explained": validation_devexp,
        "pooled_validation_swapped_ll_bits_per_spike_median": pooled_validation_bits_median,
        "pooled_validation_swapped_deviance_explained_median": pooled_validation_devexp_median,
        "pooled_validation_finite_count": pooled_validation_count,
        "selected_ridge": float(selected_ridge),
    }


def _stack_traj_vector(
    results_by_traj: dict[str, dict[str, Any]],
    key: str,
) -> np.ndarray:
    """Stack one `(unit,)` result field into trajectory order."""
    return np.stack(
        [np.asarray(results_by_traj[trajectory][key], dtype=float) for trajectory in TRAJECTORY_TYPES],
        axis=0,
    )


def _stack_traj_metric(
    results_by_traj: dict[str, dict[str, Any]],
    metric_name: str,
    key: str,
) -> np.ndarray:
    """Stack one metric field into `(trajectory, unit)` order."""
    return np.stack(
        [
            np.asarray(results_by_traj[trajectory][metric_name][key], dtype=float)
            for trajectory in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def _stack_traj_summary(
    results_by_traj: dict[str, dict[str, Any]],
    summary_name: str,
    key: str,
) -> np.ndarray:
    """Stack one TP-binned summary into trajectory order."""
    return np.stack(
        [
            np.asarray(results_by_traj[trajectory][summary_name][key], dtype=float)
            for trajectory in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def _build_heldout_lap_arrays(
    heldout_split_by_traj: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Pack held-out validation/test lap assignments into padded arrays."""
    max_laps = max(
        int(np.asarray(heldout_split_by_traj[trajectory]["lap_start_s"]).size)
        for trajectory in TRAJECTORY_TYPES
    )
    lap_start_s = np.full((len(TRAJECTORY_TYPES), max_laps), np.nan, dtype=float)
    lap_end_s = np.full((len(TRAJECTORY_TYPES), max_laps), np.nan, dtype=float)
    lap_split = np.full((len(TRAJECTORY_TYPES), max_laps), "", dtype="<U10")
    lap_count = np.zeros((len(TRAJECTORY_TYPES),), dtype=int)

    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        start_values = np.asarray(
            heldout_split_by_traj[trajectory]["lap_start_s"],
            dtype=float,
        ).reshape(-1)
        end_values = np.asarray(
            heldout_split_by_traj[trajectory]["lap_end_s"],
            dtype=float,
        ).reshape(-1)
        split_values = np.asarray(
            heldout_split_by_traj[trajectory]["lap_split"],
            dtype=str,
        ).reshape(-1)
        n_laps = int(start_values.size)
        lap_start_s[trajectory_index, :n_laps] = start_values
        lap_end_s[trajectory_index, :n_laps] = end_values
        lap_split[trajectory_index, :n_laps] = split_values
        lap_count[trajectory_index] = n_laps

    return {
        "test_lap_start_s": lap_start_s,
        "test_lap_end_s": lap_end_s,
        "test_lap_split": lap_split,
        "test_lap_count": lap_count,
    }


def build_model_dataset(
    *,
    model_name: str,
    results_by_traj: dict[str, dict[str, Any]],
    validation_sweep: dict[str, Any],
    heldout_split_by_traj: dict[str, dict[str, Any]],
    animal_name: str,
    date: str,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    ridge: float,
    dark_movement_firing_rates: np.ndarray,
    light_movement_firing_rates: np.ndarray,
    segment_edges: np.ndarray,
    observed_bin_edges: np.ndarray,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build one model-specific NetCDF dataset across trajectories."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task-progression model-comparison fits."
        ) from exc

    first = results_by_traj[TRAJECTORY_TYPES[0]]
    unit_ids = np.asarray(first["unit_ids"])
    tp_grid = np.asarray(first["tp_grid"], dtype=float)
    observed_bin_centers = 0.5 * (
        np.asarray(observed_bin_edges[:-1], dtype=float)
        + np.asarray(observed_bin_edges[1:], dtype=float)
    )
    speed_outputs = first["speed_outputs"]
    lap_arrays = _build_heldout_lap_arrays(heldout_split_by_traj)
    include_light_offset_term = bool(
        fit_parameters.get("include_light_offset_term", True)
    )
    coef_light_offset = np.stack(
        [
            np.asarray(results_by_traj[trajectory]["coef_light"], dtype=float)
            for trajectory in TRAJECTORY_TYPES
        ],
        axis=0,
    )
    if not include_light_offset_term:
        coef_light_offset = np.zeros_like(coef_light_offset)
    attrs = {
        "schema_version": "3",
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "model_name": model_name,
        "dark_train_epoch": dark_train_epoch,
        "light_train_epoch": light_train_epoch,
        "light_test_epoch": light_test_epoch,
        "ridge": float(ridge),
        "bin_size_s": float(fit_parameters["bin_size_s"]),
        "n_splines": int(fit_parameters["n_splines"]),
        "spline_order": int(fit_parameters["spline_order"]),
        "has_speed": bool(fit_parameters["use_speed"]),
        "has_light_offset_term": include_light_offset_term,
        "pos_bounds_lower": float(POS_BOUNDS[0]),
        "pos_bounds_upper": float(POS_BOUNDS[1]),
        "speed_feature_mode": str(speed_outputs["speed_feature_mode"]),
        "n_speed_features": int(speed_outputs["n_speed_features"]),
        "speed_basis": str(speed_outputs["speed_basis"]),
        "speed_spline_order": float(speed_outputs["speed_spline_order"]),
        "min_dark_firing_rate_hz": float(
            fit_parameters.get(
                "dark_region_threshold_hz",
                fit_parameters["region_threshold_hz"],
            )
        ),
        "min_light_firing_rate_hz": float(
            fit_parameters["light_region_threshold_hz"]
        ),
        "test_scoring_scope": "swapped_segment_only",
        "heldout_split_scope": "lap_level_by_trajectory",
        "heldout_validation_fraction": 0.5,
        "heldout_split_seed": int(fit_parameters["seed"]),
        "ridge_selection_metric": (
            "pooled median validation_light_swapped ll_bits_per_spike"
        ),
        "selected_ridge": float(validation_sweep["selected_ridge"]),
        "swap_rule_json": json.dumps(SWAP_CONFIG, sort_keys=True),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
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
            "speed_basis_bounds": (
                ("trajectory", "speed_bound"),
                np.stack(
                    [
                        np.asarray(
                            results_by_traj[trajectory]["speed_outputs"]["speed_basis_bounds"],
                            dtype=float,
                        )
                        for trajectory in TRAJECTORY_TYPES
                    ],
                    axis=0,
                ),
            ),
            "speed_reference_value": (
                ("trajectory",),
                np.asarray(
                    [
                        results_by_traj[trajectory]["speed_outputs"]["speed_reference_value"]
                        for trajectory in TRAJECTORY_TYPES
                    ],
                    dtype=float,
                ),
            ),
            "speed_mean": (
                ("trajectory",),
                np.asarray(
                    [
                        results_by_traj[trajectory]["speed_outputs"]["speed_mean"]
                        for trajectory in TRAJECTORY_TYPES
                    ],
                    dtype=float,
                ),
            ),
            "speed_std": (
                ("trajectory",),
                np.asarray(
                    [
                        results_by_traj[trajectory]["speed_outputs"]["speed_std"]
                        for trajectory in TRAJECTORY_TYPES
                    ],
                    dtype=float,
                ),
            ),
            "swap_source_trajectory": (
                ("trajectory",),
                np.asarray(
                    [results_by_traj[trajectory]["swap_source_trajectory"] for trajectory in TRAJECTORY_TYPES],
                    dtype=str,
                ),
            ),
            "swap_segment_index_1based": (
                ("trajectory",),
                np.asarray(
                    [results_by_traj[trajectory]["swap_segment_index_1based"] for trajectory in TRAJECTORY_TYPES],
                    dtype=int,
                ),
            ),
            "swap_segment_start": (
                ("trajectory",),
                np.asarray(
                    [results_by_traj[trajectory]["swap_segment_start"] for trajectory in TRAJECTORY_TYPES],
                    dtype=float,
                ),
            ),
            "swap_segment_end": (
                ("trajectory",),
                np.asarray(
                    [results_by_traj[trajectory]["swap_segment_end"] for trajectory in TRAJECTORY_TYPES],
                    dtype=float,
                ),
            ),
            "train_dark_null_rate_hz": (
                ("trajectory", "unit"),
                _stack_traj_vector(results_by_traj, "train_dark_null_rate_hz"),
            ),
            "train_light_null_rate_hz": (
                ("trajectory", "unit"),
                _stack_traj_vector(results_by_traj, "train_light_null_rate_hz"),
            ),
            "test_light_null_rate_hz": (
                ("trajectory", "unit"),
                _stack_traj_vector(results_by_traj, "test_light_null_rate_hz"),
            ),
            "validation_light_null_rate_hz": (
                ("trajectory", "unit"),
                _stack_traj_vector(results_by_traj, "validation_light_null_rate_hz"),
            ),
            "train_dark_spike_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_dark_metrics", "spike_sum"),
            ),
            "train_dark_ll_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_dark_metrics", "ll_sum"),
            ),
            "train_dark_ll_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_dark_metrics", "ll_per_spike"),
            ),
            "train_dark_ll_bits_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_dark_metrics", "ll_bits_per_spike"),
            ),
            "train_dark_deviance_explained": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_dark_metrics", "deviance_explained"),
            ),
            "train_light_spike_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_light_metrics", "spike_sum"),
            ),
            "train_light_ll_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_light_metrics", "ll_sum"),
            ),
            "train_light_ll_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_light_metrics", "ll_per_spike"),
            ),
            "train_light_ll_bits_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_light_metrics", "ll_bits_per_spike"),
            ),
            "train_light_deviance_explained": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "train_light_metrics", "deviance_explained"),
            ),
            "validation_light_unswapped_spike_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_unswapped_metrics",
                    "spike_sum",
                ),
            ),
            "validation_light_unswapped_ll_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_unswapped_metrics",
                    "ll_sum",
                ),
            ),
            "validation_light_unswapped_ll_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_unswapped_metrics",
                    "ll_per_spike",
                ),
            ),
            "validation_light_unswapped_ll_bits_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_unswapped_metrics",
                    "ll_bits_per_spike",
                ),
            ),
            "validation_light_unswapped_deviance_explained": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_unswapped_metrics",
                    "deviance_explained",
                ),
            ),
            "validation_light_swapped_spike_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_swapped_metrics",
                    "spike_sum",
                ),
            ),
            "validation_light_swapped_ll_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_swapped_metrics",
                    "ll_sum",
                ),
            ),
            "validation_light_swapped_ll_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_swapped_metrics",
                    "ll_per_spike",
                ),
            ),
            "validation_light_swapped_ll_bits_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_swapped_metrics",
                    "ll_bits_per_spike",
                ),
            ),
            "validation_light_swapped_deviance_explained": (
                ("trajectory", "unit"),
                _stack_traj_metric(
                    results_by_traj,
                    "validation_light_swapped_metrics",
                    "deviance_explained",
                ),
            ),
            "test_light_unswapped_spike_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_unswapped_metrics", "spike_sum"),
            ),
            "test_light_unswapped_ll_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_unswapped_metrics", "ll_sum"),
            ),
            "test_light_unswapped_ll_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_unswapped_metrics", "ll_per_spike"),
            ),
            "test_light_unswapped_ll_bits_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_unswapped_metrics", "ll_bits_per_spike"),
            ),
            "test_light_unswapped_deviance_explained": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_unswapped_metrics", "deviance_explained"),
            ),
            "test_light_swapped_spike_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_swapped_metrics", "spike_sum"),
            ),
            "test_light_swapped_ll_sum": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_swapped_metrics", "ll_sum"),
            ),
            "test_light_swapped_ll_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_swapped_metrics", "ll_per_spike"),
            ),
            "test_light_swapped_ll_bits_per_spike": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_swapped_metrics", "ll_bits_per_spike"),
            ),
            "test_light_swapped_deviance_explained": (
                ("trajectory", "unit"),
                _stack_traj_metric(results_by_traj, "test_light_swapped_metrics", "deviance_explained"),
            ),
            "dark_hz_grid": (
                ("trajectory", "tp_grid", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["dark_hz_grid"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "train_light_hz_grid": (
                ("trajectory", "tp_grid", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["train_light_hz_grid"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "test_light_unswapped_hz_grid": (
                ("trajectory", "tp_grid", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["test_light_unswapped_hz_grid"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "test_light_swapped_hz_grid": (
                ("trajectory", "tp_grid", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["test_light_swapped_hz_grid"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "train_dark_occupancy_s": (
                ("trajectory", "tp_observed_bin"),
                _stack_traj_summary(results_by_traj, "train_dark_summary", "occupancy_s"),
            ),
            "train_dark_spike_count": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(results_by_traj, "train_dark_summary", "spike_count"),
            ),
            "train_dark_observed_rate_hz": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(results_by_traj, "train_dark_summary", "observed_rate_hz"),
            ),
            "train_light_occupancy_s": (
                ("trajectory", "tp_observed_bin"),
                _stack_traj_summary(results_by_traj, "train_light_summary", "occupancy_s"),
            ),
            "train_light_spike_count": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(results_by_traj, "train_light_summary", "spike_count"),
            ),
            "train_light_observed_rate_hz": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(results_by_traj, "train_light_summary", "observed_rate_hz"),
            ),
            "validation_light_occupancy_s": (
                ("trajectory", "tp_observed_bin"),
                _stack_traj_summary(
                    results_by_traj,
                    "validation_light_summary",
                    "occupancy_s",
                ),
            ),
            "validation_light_spike_count": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(
                    results_by_traj,
                    "validation_light_summary",
                    "spike_count",
                ),
            ),
            "validation_light_observed_rate_hz": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(
                    results_by_traj,
                    "validation_light_summary",
                    "observed_rate_hz",
                ),
            ),
            "test_light_occupancy_s": (
                ("trajectory", "tp_observed_bin"),
                _stack_traj_summary(results_by_traj, "test_light_summary", "occupancy_s"),
            ),
            "test_light_spike_count": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(results_by_traj, "test_light_summary", "spike_count"),
            ),
            "test_light_observed_rate_hz": (
                ("trajectory", "tp_observed_bin", "unit"),
                _stack_traj_summary(results_by_traj, "test_light_summary", "observed_rate_hz"),
            ),
            "coef_intercept": (
                ("trajectory", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["intercept"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "coef_light_offset": (
                ("trajectory", "unit"),
                coef_light_offset,
            ),
            "coef_place_dark": (
                ("trajectory", "place_basis", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["coef_place_dark"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "coef_speed": (
                ("trajectory", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["speed_outputs"]["coef_speed"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "coef_speed_basis": (
                ("trajectory", "speed_basis_feature", "unit"),
                np.stack(
                    [np.asarray(results_by_traj[trajectory]["speed_outputs"]["coef_speed_basis"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                    axis=0,
                ),
            ),
            "segment_edges": (
                ("segment_edge",),
                np.asarray(segment_edges, dtype=float),
            ),
            "validation_candidate_ridge": (
                ("candidate_ridge",),
                np.asarray(validation_sweep["candidate_ridge"], dtype=float),
            ),
            "validation_swapped_ll_bits_per_spike_by_ridge": (
                ("candidate_ridge", "trajectory", "unit"),
                np.asarray(
                    validation_sweep["validation_swapped_ll_bits_per_spike"],
                    dtype=float,
                ),
            ),
            "validation_swapped_deviance_explained_by_ridge": (
                ("candidate_ridge", "trajectory", "unit"),
                np.asarray(
                    validation_sweep["validation_swapped_deviance_explained"],
                    dtype=float,
                ),
            ),
            "pooled_validation_swapped_ll_bits_per_spike_median": (
                ("candidate_ridge",),
                np.asarray(
                    validation_sweep[
                        "pooled_validation_swapped_ll_bits_per_spike_median"
                    ],
                    dtype=float,
                ),
            ),
            "pooled_validation_swapped_deviance_explained_median": (
                ("candidate_ridge",),
                np.asarray(
                    validation_sweep[
                        "pooled_validation_swapped_deviance_explained_median"
                    ],
                    dtype=float,
                ),
            ),
            "pooled_validation_finite_count": (
                ("candidate_ridge",),
                np.asarray(validation_sweep["pooled_validation_finite_count"], dtype=int),
            ),
            "test_lap_start_s": (
                ("trajectory", "test_lap"),
                lap_arrays["test_lap_start_s"],
            ),
            "test_lap_end_s": (
                ("trajectory", "test_lap"),
                lap_arrays["test_lap_end_s"],
            ),
            "test_lap_split": (
                ("trajectory", "test_lap"),
                lap_arrays["test_lap_split"],
            ),
            "test_lap_count": (
                ("trajectory",),
                lap_arrays["test_lap_count"],
            ),
        },
        coords={
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "unit": unit_ids,
            "tp_grid": tp_grid,
            "tp_observed_bin": observed_bin_centers,
            "tp_observed_edge": np.asarray(observed_bin_edges, dtype=float),
            "candidate_ridge": np.asarray(validation_sweep["candidate_ridge"], dtype=float),
            "speed_bound": np.asarray(["lower", "upper"], dtype=str),
            "place_basis": np.arange(int(np.asarray(first["coef_place_dark"]).shape[0]), dtype=int),
            "speed_basis_feature": np.arange(int(np.asarray(first["speed_outputs"]["coef_speed_basis"]).shape[0]), dtype=int),
            "segment_edge": np.asarray(segment_edges, dtype=float),
            "test_lap": np.arange(int(np.asarray(lap_arrays["test_lap_start_s"]).shape[1]), dtype=int),
        },
        attrs=attrs,
    )
    dataset["coef_light_offset"].attrs["description"] = (
        "Explicit scalar light indicator coefficient; zeros when "
        "has_light_offset_term is false."
    )
    dataset["coef_light_offset"].attrs["fitted"] = include_light_offset_term

    if model_name == "visual":
        dataset["coef_place_light"] = (
            ("trajectory", "place_basis", "unit"),
            np.stack(
                [np.asarray(results_by_traj[trajectory]["coef_place_light"], dtype=float) for trajectory in TRAJECTORY_TYPES],
                axis=0,
            ),
        )
    elif model_name == "task_segment_bump":
        dataset["coef_segment_bump_gain"] = (
            ("trajectory", "segment_basis", "unit"),
            np.stack(
                [
                    np.asarray(
                        results_by_traj[trajectory]["coef_segment_bump_gain"],
                        dtype=float,
                    )
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            ),
        )
        dataset = dataset.assign_coords(
            segment_basis=np.arange(
                int(np.asarray(first["coef_segment_bump_gain"]).shape[0]),
                dtype=int,
            )
        )
    elif model_name == "task_segment_scalar":
        dataset["coef_segment_scalar_gain"] = (
            ("trajectory", "segment_basis", "unit"),
            np.stack(
                [
                    np.asarray(
                        results_by_traj[trajectory]["coef_segment_scalar_gain"],
                        dtype=float,
                    )
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            ),
        )
        dataset = dataset.assign_coords(
            segment_basis=np.arange(
                int(np.asarray(first["coef_segment_scalar_gain"]).shape[0]),
                dtype=int,
            )
        )
    elif model_name == "task_dense_gain":
        dataset["coef_gain_spline"] = (
            ("trajectory", "gain_basis_feature", "unit"),
            np.stack(
                [
                    np.asarray(
                        results_by_traj[trajectory]["coef_gain_spline"],
                        dtype=float,
                    )
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            ),
        )
        dataset = dataset.assign_coords(
            gain_basis_feature=np.arange(
                int(np.asarray(first["coef_gain_spline"]).shape[0]),
                dtype=int,
            )
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return dataset


def _stack_model_traj_array(
    results_by_model: dict[str, dict[str, dict[str, Any]]],
    model_names: Sequence[str],
    key: str,
) -> np.ndarray:
    """Stack one result field into `(model, trajectory, ...)` order."""
    return np.stack(
        [
            np.stack(
                [
                    np.asarray(results_by_model[model_name][trajectory][key], dtype=float)
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            )
            for model_name in model_names
        ],
        axis=0,
    )


def _stack_model_traj_metric(
    results_by_model: dict[str, dict[str, dict[str, Any]]],
    model_names: Sequence[str],
    metric_name: str,
    key: str,
) -> np.ndarray:
    """Stack one raw metric into `(model, trajectory, unit)` order."""
    return np.stack(
        [
            np.stack(
                [
                    np.asarray(
                        results_by_model[model_name][trajectory][metric_name][key],
                        dtype=float,
                    )
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            )
            for model_name in model_names
        ],
        axis=0,
    )


def _stack_traj_result_vector(
    results_by_model: dict[str, dict[str, dict[str, Any]]],
    model_name: str,
    key: str,
    *,
    dtype: type = float,
) -> np.ndarray:
    """Stack one trajectory-level result vector from a representative model."""
    return np.asarray(
        [results_by_model[model_name][trajectory][key] for trajectory in TRAJECTORY_TYPES],
        dtype=dtype,
    )


def build_selected_swap_dataset(
    *,
    model_names: Sequence[str],
    selected_datasets: dict[str, "xr.Dataset"],
    selected_paths: dict[str, Path],
    results_by_model: dict[str, dict[str, dict[str, Any]]],
    test_inputs_by_traj: dict[str, dict[str, Any]],
    observed_summaries_by_traj: dict[str, dict[str, np.ndarray]],
    animal_name: str,
    date: str,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    segment_edges: np.ndarray,
    observed_bin_edges: np.ndarray,
    shared_metadata: dict[str, Any],
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build one combined selected-source swap comparison dataset."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save selected-source swap comparison outputs."
        ) from exc

    model_names = list(model_names)
    reference_model = "visual"
    reference_dataset = selected_datasets[reference_model]
    unit_ids = np.asarray(reference_dataset.coords["unit"].values)
    tp_grid = np.asarray(reference_dataset.coords["tp_grid"].values, dtype=float)
    observed_bin_centers = 0.5 * (
        np.asarray(observed_bin_edges[:-1], dtype=float)
        + np.asarray(observed_bin_edges[1:], dtype=float)
    )
    metric_names = (
        "test_light_swapped_segment_unswapped",
        "test_light_swapped_segment_swapped",
        "test_light_full_unswapped",
        "test_light_full_swapped",
    )
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "selected_model_path": (
            ("model",),
            np.asarray([str(selected_paths[model_name]) for model_name in model_names], dtype=str),
        ),
        "selected_ridge": (
            ("model",),
            np.asarray(
                [
                    float(selected_datasets[model_name].attrs.get("selected_ridge", np.nan))
                    for model_name in model_names
                ],
                dtype=float,
            ),
        ),
        "selected_score": (
            ("model",),
            np.asarray(
                [
                    float(selected_datasets[model_name].attrs.get("selection_score", np.nan))
                    for model_name in model_names
                ],
                dtype=float,
            ),
        ),
        "swap_source_trajectory": (
            ("trajectory",),
            _stack_traj_result_vector(
                results_by_model,
                reference_model,
                "swap_source_trajectory",
                dtype=str,
            ),
        ),
        "swap_segment_index_1based": (
            ("trajectory",),
            _stack_traj_result_vector(
                results_by_model,
                reference_model,
                "swap_segment_index_1based",
                dtype=int,
            ),
        ),
        "swap_segment_start": (
            ("trajectory",),
            _stack_traj_result_vector(
                results_by_model,
                reference_model,
                "swap_segment_start",
            ),
        ),
        "swap_segment_end": (
            ("trajectory",),
            _stack_traj_result_vector(
                results_by_model,
                reference_model,
                "swap_segment_end",
            ),
        ),
        "dark_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            _stack_model_traj_array(results_by_model, model_names, "dark_hz_grid"),
        ),
        "train_light_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            _stack_model_traj_array(results_by_model, model_names, "train_light_hz_grid"),
        ),
        "test_light_unswapped_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            _stack_model_traj_array(
                results_by_model,
                model_names,
                "test_light_unswapped_hz_grid",
            ),
        ),
        "test_light_swapped_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            _stack_model_traj_array(
                results_by_model,
                model_names,
                "test_light_swapped_hz_grid",
            ),
        ),
        "test_light_swapped_segment_n_bins": (
            ("trajectory",),
            _stack_traj_result_vector(
                results_by_model,
                reference_model,
                "test_light_swapped_segment_n_bins",
                dtype=int,
            ),
        ),
        "test_light_full_n_bins": (
            ("trajectory",),
            _stack_traj_result_vector(
                results_by_model,
                reference_model,
                "test_light_full_n_bins",
                dtype=int,
            ),
        ),
        "test_light_occupancy_s": (
            ("trajectory", "tp_observed_bin"),
            np.stack(
                [
                    np.asarray(observed_summaries_by_traj[trajectory]["occupancy_s"], dtype=float)
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            ),
        ),
        "test_light_spike_count": (
            ("trajectory", "tp_observed_bin", "unit"),
            np.stack(
                [
                    np.asarray(observed_summaries_by_traj[trajectory]["spike_count"], dtype=float)
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            ),
        ),
        "test_light_observed_rate_hz": (
            ("trajectory", "tp_observed_bin", "unit"),
            np.stack(
                [
                    np.asarray(
                        observed_summaries_by_traj[trajectory]["observed_rate_hz"],
                        dtype=float,
                    )
                    for trajectory in TRAJECTORY_TYPES
                ],
                axis=0,
            ),
        ),
    }
    for metric_name in metric_names:
        source_name = f"{metric_name}_metrics"
        for key, output_suffix in (
            ("raw_ll_sum", "raw_ll_sum"),
            ("spike_sum", "spike_sum"),
            ("raw_ll_bits_per_spike", "raw_ll_bits_per_spike"),
        ):
            data_vars[f"{metric_name}_{output_suffix}"] = (
                ("model", "trajectory", "unit"),
                _stack_model_traj_metric(
                    results_by_model,
                    model_names,
                    source_name,
                    key,
                ),
            )

    swapped_bits = data_vars[
        "test_light_swapped_segment_swapped_raw_ll_bits_per_spike"
    ][1]
    visual_index = model_names.index("visual")
    delta = swapped_bits - swapped_bits[[visual_index], :, :]
    delta[visual_index, :, :] = np.nan
    data_vars[
        "test_light_swapped_segment_swapped_delta_model_minus_visual_raw_ll_bits_per_spike"
    ] = (("model", "trajectory", "unit"), delta)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "model": np.asarray(model_names, dtype=str),
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "unit": unit_ids,
            "tp_grid": tp_grid,
            "tp_observed_bin": observed_bin_centers,
            "tp_observed_edge": np.asarray(observed_bin_edges, dtype=float),
            "segment_edge": np.asarray(segment_edges, dtype=float),
        },
        attrs={
            "schema_version": "4",
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "dark_train_epoch": dark_train_epoch,
            "light_train_epoch": light_train_epoch,
            "light_test_epoch": light_test_epoch,
            "fit_source": "dark_light_glm_selected",
            "test_scoring_scope": "swapped_segment_primary_and_full_diagnostic",
            "primary_metric": (
                "test_light_swapped_segment_swapped_delta_model_minus_visual_"
                "raw_ll_bits_per_spike"
            ),
            "heldout_epoch_scope": "all_movement_laps",
            "raw_ll_bits_per_spike_definition": "raw_poisson_ll_sum / spike_sum / log(2)",
            "swap_light_offset": bool(fit_parameters.get("swap_light_offset", False)),
            "bin_size_s": float(shared_metadata["bin_size_s"]),
            "n_splines": int(shared_metadata["n_splines"]),
            "spline_order": int(shared_metadata["spline_order"]),
            "has_speed": bool(shared_metadata["has_speed"]),
            "speed_feature_mode": str(shared_metadata["speed_feature_mode"]),
            "n_speed_features": int(shared_metadata["n_speed_features"]),
            "speed_spline_order": float(shared_metadata["speed_spline_order"]),
            "swap_rule_json": json.dumps(SWAP_CONFIG, sort_keys=True),
            "sources_json": json.dumps(sources, sort_keys=True),
            "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        },
    )
    return dataset


def plot_metric_difference_histograms(
    family_dataset: "xr.Dataset",
    visual_dataset: "xr.Dataset",
    *,
    metric_name: str,
    out_path: Path,
) -> Path:
    """Save a 2x2 histogram figure of family-minus-visual swapped metrics."""
    import matplotlib.pyplot as plt

    metric_to_label = {
        "test_light_swapped_ll_bits_per_spike": "Held-out swapped bits/spike",
        "test_light_swapped_deviance_explained": "Held-out swapped deviance explained",
    }
    if metric_name not in metric_to_label:
        raise ValueError(f"Unsupported metric_name: {metric_name}")

    def _histogram_edges_including_zero(values: np.ndarray) -> np.ndarray:
        """Return histogram edges that explicitly include zero."""
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0:
            return np.asarray([-0.5, 0.0, 0.5], dtype=float)

        auto_edges = np.asarray(np.histogram_bin_edges(values, bins="auto"), dtype=float)
        if auto_edges.size < 2:
            center = float(values[0])
            return np.asarray([center - 0.5, 0.0, center + 0.5], dtype=float)

        if np.any(np.isclose(auto_edges, 0.0)):
            return auto_edges

        lower = float(auto_edges[0])
        upper = float(auto_edges[-1])
        if 0.0 < lower:
            return np.concatenate(([0.0], auto_edges))
        if 0.0 > upper:
            return np.concatenate((auto_edges, [0.0]))

        return np.unique(np.concatenate((auto_edges, [0.0])))

    family_name = str(family_dataset.attrs["model_name"])
    visual_name = str(visual_dataset.attrs["model_name"])
    trajectory_names = [str(value) for value in family_dataset.coords["trajectory"].values]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        f"{family_dataset.attrs['region'].upper()} | "
        f"{family_dataset.attrs['light_test_epoch']} test vs "
        f"{family_dataset.attrs['dark_train_epoch']} dark + "
        f"{family_dataset.attrs['light_train_epoch']} light | "
        f"{family_name} ridge {family_dataset.attrs['ridge']}, "
        f"visual ridge {visual_dataset.attrs['ridge']} | "
        f"{metric_to_label[metric_name]}"
    )

    for axis, trajectory in zip(axes.ravel(), trajectory_names, strict=True):
        family_values = np.asarray(
            family_dataset[metric_name].sel(trajectory=trajectory).values,
            dtype=float,
        )
        visual_values = np.asarray(
            visual_dataset[metric_name].sel(trajectory=trajectory).values,
            dtype=float,
        )
        diff = family_values - visual_values
        finite_mask = np.isfinite(diff)
        finite_values = diff[finite_mask]

        if finite_values.size > 0:
            axis.hist(
                finite_values,
                bins=_histogram_edges_including_zero(finite_values),
                color="0.3",
                edgecolor="white",
            )
            median_value = float(np.median(finite_values))
            frac_positive = float(np.mean(finite_values > 0.0))
        else:
            axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
            median_value = np.nan
            frac_positive = np.nan

        axis.axvline(0.0, color="crimson", linestyle="--", linewidth=1.0)
        axis.set_title(trajectory)
        axis.set_xlabel(f"{family_name} swapped - {visual_name} swapped")
        axis.set_ylabel("Cells")
        axis.text(
            0.98,
            0.95,
            f"n={int(finite_values.size)}\nmedian={median_value:.3g}\nfrac>0={frac_positive:.3f}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_selected_model_minus_visual_histogram(
    dataset: "xr.Dataset",
    *,
    model_name: str,
    out_path: Path,
) -> Path:
    """Save the primary pooled task-minus-visual swapped-segment LL histogram."""
    import matplotlib.pyplot as plt

    metric_name = (
        "test_light_swapped_segment_swapped_delta_model_minus_visual_"
        "raw_ll_bits_per_spike"
    )
    if model_name == "visual":
        raise ValueError("The visual baseline has no model-minus-visual histogram.")
    if model_name not in set(str(value) for value in dataset.coords["model"].values):
        raise ValueError(f"Model {model_name!r} is not present in the selected dataset.")
    values = np.asarray(dataset[metric_name].sel(model=model_name).values, dtype=float)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        edges = np.asarray([-0.5, 0.0, 0.5], dtype=float)
        median_value = np.nan
        frac_positive = np.nan
    else:
        edges = np.asarray(np.histogram_bin_edges(finite_values, bins="auto"), dtype=float)
        if edges.size < 2:
            center = float(finite_values[0])
            edges = np.asarray([center - 0.5, center + 0.5], dtype=float)
        if not np.any(np.isclose(edges, 0.0)):
            edges = np.unique(np.concatenate((edges, [0.0])))
        median_value = float(np.median(finite_values))
        frac_positive = float(np.mean(finite_values > 0.0))

    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.hist(finite_values, bins=edges, color="0.2", edgecolor="white")
    axis.axvline(0.0, color="0.15", linestyle="--", linewidth=1.0)
    if finite_values.size == 0:
        axis.text(
            0.5,
            0.5,
            "No finite delta LL values",
            ha="center",
            va="center",
            transform=axis.transAxes,
            fontsize=12,
        )
    axis.set_xlabel(
        f"{model_name} - visual swapped-segment swapped raw LL (bits/spike)"
    )
    axis.set_ylabel("Trajectory-cell pairs")
    axis.set_title(
        f"{dataset.attrs['region'].upper()} | {dataset.attrs['light_test_epoch']} test | "
        f"{model_name} vs visual"
    )
    axis.text(
        0.98,
        0.95,
        (
            f"n={int(finite_values.size)}\n"
            f"median={median_value:.3g}\n"
            f"frac>0={frac_positive:.3f}"
        ),
        ha="right",
        va="top",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the task-progression model-comparison workflow."""
    args = parse_arguments()
    model_names, model_messages = normalize_requested_models(args.models)
    for message in model_messages:
        print(message)

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    selected_dir = (
        Path(args.dark_light_glm_dir)
        if args.dark_light_glm_dir is not None
        else selected_dark_light_glm_dir(analysis_path)
    )
    if len({args.dark_train_epoch, args.light_train_epoch, args.light_test_epoch}) != 3:
        raise ValueError(
            "dark_train_epoch, light_train_epoch, and light_test_epoch must all be distinct."
        )

    print(
        "Loading held-out light test epoch for "
        f"{args.animal_name} {args.date}: light-test={args.light_test_epoch}."
    )
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=tuple(args.regions),
        selected_run_epochs=[args.light_test_epoch],
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    validate_model_comparison_epochs(
        session["available_run_epochs"],
        dark_train_epoch=args.dark_train_epoch,
        light_train_epoch=args.light_train_epoch,
        light_test_epoch=args.light_test_epoch,
    )

    observed_bin_edges = build_task_progression_bins(args.animal_name)
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    saved_datasets: list[Path] = []
    saved_figures: list[Path] = []
    selected_files_by_region: dict[str, dict[str, str]] = {}
    _require_spline_basis()

    for region in args.regions:
        print(
            f"Loading selected dark/light GLMs for {region.upper()} from {selected_dir}."
        )
        selected_datasets: dict[str, Any] = {}
        selected_paths: dict[str, Path] = {}
        for model_name in model_names:
            dataset, path = load_selected_dark_light_glm(
                selected_dir,
                region=region,
                light_train_epoch=args.light_train_epoch,
                dark_train_epoch=args.dark_train_epoch,
                model_name=model_name,
            )
            selected_datasets[model_name] = dataset
            selected_paths[model_name] = path
        shared_metadata = validate_selected_dark_light_glms(
            selected_datasets,
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
        )
        selected_files_by_region[region] = {
            model_name: str(path) for model_name, path in selected_paths.items()
        }
        segment_edges = np.asarray(shared_metadata["segment_edges"], dtype=float)
        print(
            f"Scoring {len(shared_metadata['unit_ids'])} selected {region.upper()} "
            f"unit(s) at bin={shared_metadata['bin_size_s']:g}s, "
            f"n_splines={shared_metadata['n_splines']}."
        )

        speed_by_run = (
            session["speed_by_run"] if bool(shared_metadata["has_speed"]) else None
        )
        test_inputs_by_traj: dict[str, dict[str, Any]] = {}
        observed_summaries_by_traj: dict[str, dict[str, np.ndarray]] = {}
        for trajectory in TRAJECTORY_TYPES:
            test_inputs = _prepare_test_epoch_inputs_for_units(
                spikes=session["spikes_by_region"][region],
                unit_ids=np.asarray(shared_metadata["unit_ids"]),
                trajectory_ep_by_epoch=session["trajectory_intervals"],
                tp_by_epoch=session["task_progression_by_trajectory"],
                speed_by_epoch=speed_by_run,
                traj_name=trajectory,
                epoch=args.light_test_epoch,
                bin_size_s=float(shared_metadata["bin_size_s"]),
                restrict_interval=session["movement_by_run"][args.light_test_epoch],
            )
            if int(test_inputs["y"].shape[0]) == 0:
                raise ValueError(
                    f"Held-out light test epoch {args.light_test_epoch!r} has no "
                    f"movement bins for {region!r} trajectory {trajectory!r}."
                )
            test_inputs_by_traj[trajectory] = test_inputs
            observed_summaries_by_traj[trajectory] = build_observed_summary(
                test_inputs["y"],
                test_inputs["p"],
                observed_bin_edges,
                bin_size_s=float(shared_metadata["bin_size_s"]),
            )

        results_by_model: dict[str, dict[str, dict[str, Any]]] = {}
        for model_name in model_names:
            print(
                f"  Scoring {model_name} on all movement bins in "
                f"{args.light_test_epoch}."
            )
            results_by_model[model_name] = evaluate_selected_model_on_test_epoch(
                model_name=model_name,
                datasets_by_model=selected_datasets,
                test_inputs_by_traj=test_inputs_by_traj,
                segment_edges=segment_edges,
                bin_size_s=float(shared_metadata["bin_size_s"]),
                n_splines=int(shared_metadata["n_splines"]),
                spline_order=int(shared_metadata["spline_order"]),
                swap_light_offset=bool(args.swap_light_offset),
            )

        sources = dict(session["sources"])
        sources["dark_light_glm_selected"] = {
            model_name: str(path) for model_name, path in selected_paths.items()
        }
        fit_parameters = {
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "models": list(model_names),
            "requested_models": list(args.models),
            "dark_light_glm_dir": str(selected_dir),
            "scoring_epoch_scope": "all_movement_laps",
            "swap_light_offset": bool(args.swap_light_offset),
            "swapped_component": (
                "local_model_component_with_scalar_light_offset"
                if args.swap_light_offset
                else "local_model_component_without_scalar_light_offset"
            ),
        }
        dataset = build_selected_swap_dataset(
            model_names=model_names,
            selected_datasets=selected_datasets,
            selected_paths=selected_paths,
            results_by_model=results_by_model,
            test_inputs_by_traj=test_inputs_by_traj,
            observed_summaries_by_traj=observed_summaries_by_traj,
            animal_name=args.animal_name,
            date=args.date,
            region=region,
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
            light_test_epoch=args.light_test_epoch,
            segment_edges=segment_edges,
            observed_bin_edges=observed_bin_edges,
            shared_metadata=shared_metadata,
            sources=sources,
            fit_parameters=fit_parameters,
        )
        light_offset_token = "_swap_light_offset" if args.swap_light_offset else ""
        result_stem = (
            f"{region}_{args.dark_train_epoch}_traindark_"
            f"{args.light_train_epoch}_trainlight_"
            f"{args.light_test_epoch}_testlight_"
            f"dark_light_selected_swap{light_offset_token}"
        )
        result_path = data_dir / f"{result_stem}.nc"
        dataset.to_netcdf(result_path)
        saved_datasets.append(result_path)
        print(f"Saved selected-source swap dataset to {result_path}.")

        for model_name in model_names:
            if model_name == "visual":
                continue
            figure_path = fig_dir / f"{result_stem}_{model_name}_minus_visual_bits_hist.png"
            saved_figures.append(
                plot_selected_model_minus_visual_histogram(
                    dataset,
                    model_name=model_name,
                    out_path=figure_path,
                )
            )
            print(f"Saved primary delta LL histogram to {figure_path}.")

        dataset.close()
        for dataset in selected_datasets.values():
            dataset.close()

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.swap_glm_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": args.regions,
            "models": model_names,
            "requested_models": args.models,
            "dark_train_epoch": args.dark_train_epoch,
            "light_train_epoch": args.light_train_epoch,
            "light_test_epoch": args.light_test_epoch,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "dark_light_glm_dir": selected_dir,
            "swap_light_offset": bool(args.swap_light_offset),
        },
        outputs={
            "sources": session["sources"],
            "saved_datasets": saved_datasets,
            "saved_figures": saved_figures,
            "selected_files_by_region": selected_files_by_region,
        },
    )

    if saved_datasets:
        print(f"Saved {len(saved_datasets)} NetCDF comparison dataset(s) to {data_dir}")
    if saved_figures:
        print(f"Saved {len(saved_figures)} histogram figure(s) to {fig_dir}")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
