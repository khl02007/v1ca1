"""Explicit activation for the project-owned Spyglass tables.

Importing this module is passive: DataJoint and Spyglass are imported only by
``activate``. Runtime computation is likewise reached only through explicitly
activated computed tables.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import date, datetime
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from v1ca1.spyglass import table_specs


SOURCE_TABLE_KEYS = (
    "epoch_intervals",
    "trajectory_intervals",
    "ripple_intervals",
    "position",
    "wtrack_graph",
    "spike_sorting_figurl",
)

_LEGACY_TUNING_POSITION_SERIES_NAME = "head_position"
_LEGACY_TUNING_POSITION_ROLE = "head"
_LEGACY_TUNING_POSITION_OFFSET_SAMPLES = 10
_DPP_TRAJECTORY_TYPES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)
_DPP_FULL_GRAPH_CONFIGURATION_NAME = "full_w"
_SWAP_TUNING_EPOCH_ROLES = ("dark", "light_train", "light_test")


def _database_bool(value: Any, *, name: str) -> bool:
    """Normalize one DataJoint-compatible bool without accepting truthy junk."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return bool(int(value))
    raise TypeError(f"{name} must be a bool or database integer 0/1.")


def _validate_parameter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one all-scalar RippleModulation parameter row."""
    expected = set(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "RippleModulation parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )

    values = dict(row)
    name = values["ripple_modulation_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "ripple_modulation_param_name must be a non-empty string of at most 64 characters."
        )

    numeric_names = (
        "bin_size_s",
        "time_before_s",
        "time_after_s",
        "response_window_start_s",
        "response_window_end_s",
        "baseline_window_start_s",
        "baseline_window_end_s",
        "expected_detector_zscore_threshold",
    )
    for field_name in numeric_names:
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value

    for field_name in ("bin_size_s", "time_before_s", "time_after_s"):
        if values[field_name] <= 0:
            raise ValueError(f"{field_name} must be positive.")
    if values["response_window_start_s"] >= values["response_window_end_s"]:
        raise ValueError("response window start must be smaller than its end.")
    if values["baseline_window_start_s"] >= values["baseline_window_end_s"]:
        raise ValueError("baseline window start must be smaller than its end.")

    lower_bound = -values["time_before_s"]
    upper_bound = values["time_after_s"]
    for prefix in ("response", "baseline"):
        start = values[f"{prefix}_window_start_s"]
        stop = values[f"{prefix}_window_end_s"]
        if start < lower_bound or stop > upper_bound:
            raise ValueError(
                f"{prefix} window {(start, stop)!r} lies outside "
                f"the peri-ripple window {(lower_bound, upper_bound)!r}."
            )

    if values["heatmap_normalize"] not in {"max", "zscore"}:
        raise ValueError("heatmap_normalize must be 'max' or 'zscore'.")
    if values["expected_detector_zscore_threshold"] <= 0:
        raise ValueError("expected_detector_zscore_threshold must be positive.")
    require_speed_gated = values["require_speed_gated"]
    if isinstance(require_speed_gated, (str, bytes, list, tuple, dict)):
        raise TypeError("require_speed_gated must be a bool scalar.")
    try:
        is_boolean_scalar = require_speed_gated in (True, False)
    except (TypeError, ValueError):
        is_boolean_scalar = False
    if not is_boolean_scalar:
        raise TypeError("require_speed_gated must be a bool scalar.")
    values["require_speed_gated"] = bool(require_speed_gated)
    return values


def _validate_tuning_curve_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one shared tuning-curve binning and smoothing parameter row."""
    expected = set(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "TuningCurve parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["tuning_curve_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "tuning_curve_param_name must be a non-empty string "
            "of at most 64 characters."
        )
    mode = values["binning_mode"]
    if mode not in {"bin_size_cm", "bin_count"}:
        raise ValueError("binning_mode must be 'bin_size_cm' or 'bin_count'.")

    sigma = values["gaussian_smoothing_sigma_bins"]
    if isinstance(sigma, bool) or not isinstance(sigma, Real):
        raise TypeError("gaussian_smoothing_sigma_bins must be one numeric scalar.")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError(
            "gaussian_smoothing_sigma_bins must be finite and non-negative."
        )
    values["gaussian_smoothing_sigma_bins"] = sigma

    bin_size = values["place_bin_size_cm"]
    bin_count = values["position_bin_count"]
    if mode == "bin_size_cm":
        if isinstance(bin_size, bool) or not isinstance(bin_size, Real):
            raise TypeError("place_bin_size_cm must be one numeric scalar.")
        bin_size = float(bin_size)
        if not math.isfinite(bin_size) or bin_size <= 0:
            raise ValueError("place_bin_size_cm must be positive and finite.")
        if bin_count is not None:
            raise ValueError(
                "position_bin_count must be NULL when binning_mode is bin_size_cm."
            )
        values["place_bin_size_cm"] = bin_size
    else:
        if bin_size is not None:
            raise ValueError(
                "place_bin_size_cm must be NULL when binning_mode is bin_count."
            )
        if isinstance(bin_count, bool) or not isinstance(bin_count, Integral):
            raise TypeError("position_bin_count must be one integer scalar.")
        if int(bin_count) <= 0 or int(bin_count) > 65_535:
            raise ValueError(
                "position_bin_count must be a positive integer no larger than 65535."
            )
        values["position_bin_count"] = int(bin_count)
    return values


def _validate_tuning_similarity_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one fixed path-specific tuning-similarity metric row."""
    expected = set(table_specs.CORRELATION_TUNING_SIMILARITY_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "TuningSimilarity parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["tuning_similarity_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "tuning_similarity_param_name must be a non-empty string "
            "of at most 64 characters."
        )
    metric = values["similarity_metric"]
    supported_metrics = {
        parameters["similarity_metric"]
        for parameters in table_specs.TUNING_SIMILARITY_PARAMETER_PRESETS
    }
    if metric not in supported_metrics:
        raise ValueError(
            f"similarity_metric must be one of {sorted(supported_metrics)!r}."
        )
    return values


def _validate_dpp_encoding_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one four-model DPP encoding-comparison parameter row."""
    expected = set(table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "DPPEncoding parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["dpp_encoding_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "dpp_encoding_param_name must be a non-empty string "
            "of at most 64 characters."
        )

    n_folds = values["n_folds"]
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral):
        raise TypeError("n_folds must be one integer scalar.")
    n_folds = int(n_folds)
    if n_folds < 2 or n_folds > 65_535:
        raise ValueError("n_folds must be between 2 and 65535.")
    values["n_folds"] = n_folds

    random_seed = values["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be one integer scalar.")
    random_seed = int(random_seed)
    if random_seed < 0 or random_seed > 2_147_483_647:
        raise ValueError("random_seed must fit a non-negative signed 32-bit integer.")
    values["random_seed"] = random_seed

    for field_name in (
        "evaluation_bin_size_s",
        "spatial_bin_size_cm",
        "gaussian_smoothing_sigma_bins",
        "minimum_movement_firing_rate_hz",
        "minimum_stability_correlation",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value
    for field_name in ("evaluation_bin_size_s", "spatial_bin_size_cm"):
        if values[field_name] <= 0.0:
            raise ValueError(f"{field_name} must be positive.")
    for field_name in (
        "gaussian_smoothing_sigma_bins",
        "minimum_movement_firing_rate_hz",
    ):
        if values[field_name] < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")
    if not -1.0 <= values["minimum_stability_correlation"] <= 1.0:
        raise ValueError(
            "minimum_stability_correlation must be between -1 and 1."
        )
    return values


def _validate_path_progression_decoding_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one path-progression Bayesian-decoding parameter row."""
    expected = set(
        table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "PathProgressionDecoding parameters must have exactly the "
            f"declared fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["path_progression_decoding_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "path_progression_decoding_param_name must be a non-empty "
            "string of at most 64 characters."
        )

    for field_name, minimum in (("sliding_window_size_bins", 1),):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field_name} must be one integer scalar.")
        value = int(value)
        if value < minimum or value > 65_535:
            raise ValueError(
                f"{field_name} must be between {minimum} and 65535."
            )
        values[field_name] = value

    for field_name in (
        "decoding_bin_size_s",
        "spatial_bin_size_cm",
        "minimum_movement_firing_rate_hz",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value
    for field_name in ("decoding_bin_size_s", "spatial_bin_size_cm"):
        if values[field_name] <= 0.0:
            raise ValueError(f"{field_name} must be positive.")
    if values["minimum_movement_firing_rate_hz"] < 0.0:
        raise ValueError(
            "minimum_movement_firing_rate_hz must be non-negative."
        )

    stability = values["minimum_stability_correlation"]
    if stability is not None:
        if isinstance(stability, bool) or not isinstance(stability, Real):
            raise TypeError(
                "minimum_stability_correlation must be NULL or one numeric scalar."
            )
        stability = float(stability)
        if not math.isfinite(stability) or not -1.0 <= stability <= 1.0:
            raise ValueError(
                "minimum_stability_correlation must be NULL or finite within [-1, 1]."
            )
        values["minimum_stability_correlation"] = stability
    return values


def _validate_path_specific_place_decoding_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one within-epoch path-specific place decoder row."""
    expected = set(
        table_specs.MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "PathSpecificPlaceDecoding parameters must have exactly the "
            f"declared fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["path_specific_place_decoding_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "path_specific_place_decoding_param_name must be a non-empty "
            "string of at most 64 characters."
        )
    for field_name, minimum in (
        ("n_folds", 2),
        ("sliding_window_size_bins", 1),
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field_name} must be one integer scalar.")
        value = int(value)
        if value < minimum or value > 65_535:
            raise ValueError(
                f"{field_name} must be between {minimum} and 65535."
            )
        values[field_name] = value
    random_seed = values["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be one integer scalar.")
    random_seed = int(random_seed)
    if random_seed < 0 or random_seed > 2_147_483_647:
        raise ValueError(
            "random_seed must fit a non-negative signed 32-bit integer."
        )
    values["random_seed"] = random_seed
    for field_name in ("decoding_bin_size_s", "spatial_bin_size_cm"):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field_name} must be positive and finite.")
        values[field_name] = value
    return values


def _validate_motor_encoding_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one nine-model nested-CV motor-encoding parameter row."""
    expected = set(
        table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "MotorEncoding parameters must have exactly the "
            f"declared fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["motor_encoding_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "motor_encoding_param_name must be a non-empty "
            "string of at most 64 characters."
        )

    for field_name, minimum in (
        ("outer_n_folds", 2),
        ("inner_n_folds", 2),
        ("motor_spline_n_basis", 1),
        ("motor_spline_order", 1),
        ("position_spline_order", 1),
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field_name} must be one integer scalar.")
        value = int(value)
        if value < minimum or value > 65_535:
            raise ValueError(
                f"{field_name} must be between {minimum} and 65535."
            )
        values[field_name] = value
    random_seed = values["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be one integer scalar.")
    random_seed = int(random_seed)
    if random_seed < 0 or random_seed > 2_147_483_647:
        raise ValueError(
            "random_seed must fit a non-negative signed 32-bit integer."
        )
    values["random_seed"] = random_seed

    for field_name in (
        "evaluation_bin_size_s",
        "minimum_movement_firing_rate_hz",
        "minimum_stability_correlation",
        "motor_zscore_eps",
        "speed_smoothing_sigma_s",
        "generalized_place_branch_gap_cm",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value
    for field_name in (
        "evaluation_bin_size_s",
        "motor_zscore_eps",
    ):
        if values[field_name] <= 0.0:
            raise ValueError(f"{field_name} must be positive.")
    for field_name in (
        "minimum_movement_firing_rate_hz",
        "speed_smoothing_sigma_s",
        "generalized_place_branch_gap_cm",
    ):
        if values[field_name] < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")
    if not -1.0 <= values["minimum_stability_correlation"] <= 1.0:
        raise ValueError(
            "minimum_stability_correlation must be between -1 and 1."
        )

    for field_name, allow_zero in (
        ("ridge_values", True),
        ("spatial_bin_sizes_cm", False),
    ):
        raw_values = values[field_name]
        if isinstance(raw_values, (str, bytes, Mapping)):
            raise TypeError(f"{field_name} must be a one-dimensional sequence.")
        array = np.asarray(raw_values)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(
                f"{field_name} must be a non-empty one-dimensional sequence."
            )
        if np.issubdtype(array.dtype, np.bool_):
            raise TypeError(f"{field_name} must contain numeric values.")
        raw_list = array.tolist()
        if any(isinstance(value, (bool, np.bool_)) for value in raw_list):
            raise TypeError(f"{field_name} must contain numeric values.")
        try:
            numeric_values = tuple(float(value) for value in raw_list)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{field_name} must contain numeric values."
            ) from exc
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError(f"{field_name} must contain only finite values.")
        if allow_zero:
            invalid = any(value < 0.0 for value in numeric_values)
        else:
            invalid = any(value <= 0.0 for value in numeric_values)
        if invalid:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(
                f"{field_name} must contain only {qualifier} values."
            )
        if len(set(numeric_values)) != len(numeric_values):
            raise ValueError(f"{field_name} must not contain duplicates.")
        values[field_name] = numeric_values

    if values["motor_feature_mode"] not in {"zscore", "spline"}:
        raise ValueError("motor_feature_mode must be 'zscore' or 'spline'.")
    return values


def _validate_dark_light_glm_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one coupled dark/light GLM parameter row exactly."""
    from v1ca1.spyglass.dark_light_glm import (
        validate_dark_light_glm_parameters,
    )

    expected = set(table_specs.CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "DarkLightGLM parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["dark_light_glm_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "dark_light_glm_param_name must be a non-empty string of at "
            "most 64 characters."
        )
    for field_name, minimum in (
        ("n_folds", 2),
        ("spline_order", 1),
        ("n_splines_speed", 1),
        ("spline_order_speed", 1),
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{field_name} must be one integer scalar.")
        value = int(value)
        if value < minimum or value > 65_535:
            raise ValueError(
                f"{field_name} must be between {minimum} and 65535."
            )
        values[field_name] = value
    random_seed = values["random_seed"]
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be one integer scalar.")
    random_seed = int(random_seed)
    if random_seed < 0 or random_seed > 2_147_483_647:
        raise ValueError(
            "random_seed must fit a non-negative signed 32-bit integer."
        )
    values["random_seed"] = random_seed
    for field_name in (
        "min_dark_firing_rate_hz",
        "min_light_firing_rate_hz",
        "speed_smoothing_sigma_s",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or (
            field_name == "speed_smoothing_sigma_s" and value == 0.0
        ):
            qualifier = (
                "positive"
                if field_name == "speed_smoothing_sigma_s"
                else "non-negative"
            )
            raise ValueError(f"{field_name} must be {qualifier} and finite.")
        values[field_name] = value
    use_speed = values["use_speed"]
    if not isinstance(use_speed, (bool, np.bool_)):
        raise TypeError("use_speed must be one bool scalar.")
    values["use_speed"] = bool(use_speed)
    validated = validate_dark_light_glm_parameters(
        basis_candidate_mode=values["basis_candidate_mode"],
        basis_candidates=values["basis_candidates"],
        bin_sizes_s=values["bin_sizes_s"],
        ridges=values["ridges"],
        n_folds=values["n_folds"],
        random_seed=values["random_seed"],
        spline_order=values["spline_order"],
        min_dark_firing_rate_hz=values["min_dark_firing_rate_hz"],
        min_light_firing_rate_hz=values["min_light_firing_rate_hz"],
        use_speed=values["use_speed"],
        speed_feature_mode=values["speed_feature_mode"],
        n_splines_speed=values["n_splines_speed"],
        spline_order_speed=values["spline_order_speed"],
        speed_bounds=values["speed_bounds"],
        speed_smoothing_sigma_s=values["speed_smoothing_sigma_s"],
    )
    return {
        "dark_light_glm_param_name": name,
        **{
            field_name: validated[field_name]
            for field_name in expected
            if field_name != "dark_light_glm_param_name"
        },
    }


def _validate_swap_glm_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one held-out swapped-light scoring parameter row exactly."""
    from v1ca1.spyglass.swap_glm import validate_swap_glm_parameters

    expected = set(table_specs.DEFAULT_SWAP_GLM_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "SwapGLM parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["swap_glm_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "swap_glm_param_name must be a non-empty string of at most "
            "64 characters."
        )
    validated = validate_swap_glm_parameters(
        swap_light_offset=values["swap_light_offset"],
        observed_spatial_bin_size_cm=values[
            "observed_spatial_bin_size_cm"
        ],
    )
    return {"swap_glm_param_name": name, **validated}


def _validate_swap_tuning_curve_comparison_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one empirical held-out swap-tuning parameter row."""
    from v1ca1.spyglass.swap_tuning import (
        validate_swap_tuning_curve_comparison_parameters,
    )

    expected = set(
        table_specs.MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "SwapTuningCurveComparison parameters must have exactly the "
            f"declared fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["swap_tuning_curve_comparison_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "swap_tuning_curve_comparison_param_name must be a non-empty "
            "string of at most 64 characters."
        )
    for field_name in (
        "evaluation_bin_size_s",
        "gaussian_smoothing_sigma_bins",
        "min_dark_firing_rate_hz",
        "min_light_firing_rate_hz",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value
    if values["evaluation_bin_size_s"] <= 0.0:
        raise ValueError("evaluation_bin_size_s must be positive.")
    for field_name in (
        "gaussian_smoothing_sigma_bins",
        "min_dark_firing_rate_hz",
        "min_light_firing_rate_hz",
    ):
        if values[field_name] < 0.0:
            raise ValueError(f"{field_name} must be non-negative.")
    validated = validate_swap_tuning_curve_comparison_parameters(
        evaluation_bin_size_s=values["evaluation_bin_size_s"],
        gaussian_smoothing_sigma_bins=values[
            "gaussian_smoothing_sigma_bins"
        ],
        min_dark_firing_rate_hz=values["min_dark_firing_rate_hz"],
        min_light_firing_rate_hz=values["min_light_firing_rate_hz"],
    )
    return {
        "swap_tuning_curve_comparison_param_name": name,
        **validated,
    }


def _ripple_glm_parameter_kwargs(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the model fields accepted by the database-free RippleGLM API."""
    excluded = {
        "ripple_glm_param_name",
        "source_target_windows_differ",
    }
    return {
        field_name: value
        for field_name, value in parameters.items()
        if field_name not in excluded
    }


def _validate_ripple_glm_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact CA1-to-V1 ripple population-GLM parameter row."""
    from v1ca1.spyglass.ripple_glm import validate_ripple_glm_parameters

    expected = set(
        table_specs.MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "RippleGLM parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["ripple_glm_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "ripple_glm_param_name must be a non-empty string of at most "
            "64 characters."
        )
    values["require_speed_gated"] = _database_bool(
        values["require_speed_gated"],
        name="require_speed_gated",
    )
    values["source_target_windows_differ"] = _database_bool(
        values["source_target_windows_differ"],
        name="source_target_windows_differ",
    )
    validated = validate_ripple_glm_parameters(
        **_ripple_glm_parameter_kwargs(values)
    )
    expected_derived = values["source_target_windows_differ"]
    if bool(validated["source_target_windows_differ"]) != expected_derived:
        raise ValueError(
            "source_target_windows_differ does not match the effective "
            "source and target windows."
        )
    return {"ripple_glm_param_name": name, **validated}


def _ripple_cross_region_xcorr_parameter_kwargs(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return model fields accepted by the database-free xcorr API."""
    return {
        field_name: value
        for field_name, value in parameters.items()
        if field_name != "ripple_cross_region_xcorr_param_name"
    }


def _validate_ripple_cross_region_xcorr_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact ripple-restricted cross-region parameter row."""
    from v1ca1.spyglass.ripple_cross_region_xcorr import (
        validate_ripple_cross_region_xcorr_parameters,
    )

    expected = set(table_specs.MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "RippleCrossRegionXCorr parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["ripple_cross_region_xcorr_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "ripple_cross_region_xcorr_param_name must be a non-empty string of at "
            "most 64 characters."
        )
    values["norm"] = _database_bool(values["norm"], name="norm")
    values["require_speed_gated"] = _database_bool(
        values["require_speed_gated"],
        name="require_speed_gated",
    )
    validated = validate_ripple_cross_region_xcorr_parameters(
        **_ripple_cross_region_xcorr_parameter_kwargs(values)
    )
    return {"ripple_cross_region_xcorr_param_name": name, **validated}


def _validate_movement_parameter_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy one shared movement parameter row."""
    expected = set(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "Movement parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["movement_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "movement_param_name must be a non-empty string of at most 64 "
            "characters."
        )
    for field_name in (
        "speed_threshold_cm_s",
        "speed_smoothing_sigma_s",
    ):
        value = values[field_name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field_name} must be one numeric scalar.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        values[field_name] = value
    if values["speed_threshold_cm_s"] < 0:
        raise ValueError("speed_threshold_cm_s must be non-negative.")
    if values["speed_smoothing_sigma_s"] <= 0:
        raise ValueError("speed_smoothing_sigma_s must be positive.")
    return values


def _validate_epoch_motor_behavior_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sole epoch motor-behavior parameter row."""
    from v1ca1.spyglass.epoch_motor_behavior import (
        validate_epoch_motor_behavior_parameters,
    )

    expected = set(table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "EpochMotorBehavior parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    name = row["epoch_motor_behavior_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "epoch_motor_behavior_param_name must be a non-empty string of "
            "at most 64 characters."
        )
    validated = validate_epoch_motor_behavior_parameters(
        progression_bin_size_cm=row["progression_bin_size_cm"]
    )
    return {
        "epoch_motor_behavior_param_name": name,
        **validated,
    }


def _validate_cv_pca_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one named cvPCA parameter row without deriving its region."""
    from v1ca1.spyglass.cv_pca import validate_cv_pca_parameters

    expected = set(table_specs.MANUSCRIPT_V1_CV_PCA_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "CVPCA parameters must have exactly the declared fields; "
            f"missing={missing!r}, extra={extra!r}."
        )
    name = row["cv_pca_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "cv_pca_param_name must be a non-empty string of at most 64 "
            "characters."
        )
    # An explicit threshold makes validation independent of the selected
    # regional group.  Region is never a parameter-table attribute.
    validated = validate_cv_pca_parameters(
        region="parameter_validation",
        **{
            field_name: row[field_name]
            for field_name in expected
            if field_name != "cv_pca_param_name"
        },
    )
    return {"cv_pca_param_name": name, **validated}


def _validate_legacy_tuning_curve_inputs(
    *,
    position_row: Mapping[str, Any],
    movement_parameters: Mapping[str, Any],
) -> None:
    """Require inputs compatible with the legacy tuning-curve workflow."""
    expected_position = {
        "position_series_name": _LEGACY_TUNING_POSITION_SERIES_NAME,
        "position_role": _LEGACY_TUNING_POSITION_ROLE,
        "analysis_start_offset_samples": (
            _LEGACY_TUNING_POSITION_OFFSET_SAMPLES
        ),
    }
    for field_name, expected_value in expected_position.items():
        if str(position_row.get(field_name)) != str(expected_value):
            raise ValueError(
                "Legacy tuning-curve registration requires cleaned DLC head "
                f"position with a 10-sample analysis offset; {field_name}="
                f"{position_row.get(field_name)!r}."
            )

    expected_movement = table_specs.DEFAULT_MOVEMENT_PARAMETERS
    for field_name in (
        "speed_threshold_cm_s",
        "speed_smoothing_sigma_s",
    ):
        actual = movement_parameters.get(field_name)
        expected = expected_movement[field_name]
        try:
            matches = math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            matches = False
        if not matches:
            raise ValueError(
                "Legacy tuning-curve registration requires the legacy "
                f"movement defaults; {field_name}={actual!r}, expected "
                f"{expected!r}."
            )


def _fetch1_dict(table: Any, key: Mapping[str, Any]) -> dict[str, Any]:
    """Fetch one relation row as a plain dictionary."""
    row = (table & dict(key)).fetch1()
    if not isinstance(row, Mapping):
        raise TypeError(f"{table!r}.fetch1() must return a mapping.")
    return dict(row)


def _load_catalog_nwb_object(
    table: Any,
    key: Mapping[str, Any],
    *,
    nwbfile_table: Any,
    loader: Callable[..., Any],
    loader_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Open one row's source NWB read-only and run its database-free loader."""
    import pynwb

    row = _fetch1_dict(table, key)
    nwb_file_name = row.get("nwb_file_name", key.get("nwb_file_name"))
    if nwb_file_name is None:
        raise ValueError("Source-table key does not identify an nwb_file_name.")
    nwb_path = Path(nwbfile_table.get_abs_path(str(nwb_file_name)))
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        return loader(io.read(), row, **dict(loader_kwargs or {}))


def _session_identity(session_table: Any, key: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve artifact identity from standard Session metadata."""
    subject_id, start_time = (session_table & dict(key)).fetch1(
        "subject_id",
        "session_start_time",
    )
    if subject_id is None or not str(subject_id).strip():
        raise ValueError(
            "Session.subject_id is required for analysis artifact paths."
        )
    if isinstance(start_time, (datetime, date)):
        session_date = start_time.strftime("%Y%m%d")
    elif hasattr(start_time, "strftime"):
        session_date = start_time.strftime("%Y%m%d")
    else:
        raise TypeError("Session.session_start_time must provide strftime().")
    return str(subject_id), str(session_date)


def _git_commit(path: Path) -> str | None:
    """Return the repository HEAD containing ``path``, if available."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if commit else None


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one existing artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_created_artifacts(paths: list[str]) -> None:
    """Remove newly created files or UUID bundle directories after failure."""
    parents: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        parents.add(path.parent)
        if path.is_file():
            path.unlink()
        elif path.is_dir() and not path.is_symlink():
            import shutil
            import uuid

            try:
                uuid.UUID(path.name)
            except ValueError:
                continue
            shutil.rmtree(path)
    for parent in parents:
        try:
            parent.rmdir()
        except OSError:
            pass


def _existing_result_row(
    table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return an existing result before any registration-side file writes."""
    try:
        relation = table & dict(key)
    except (AttributeError, TypeError):
        # Minimal injected table fakes used by dependency-free tests do not
        # implement DataJoint relation operators.
        return None
    if not relation:
        return None
    row = relation.fetch1()
    if not isinstance(row, Mapping):
        raise TypeError("Existing computed result must fetch as a mapping.")
    return dict(row)


def _transaction_context(table: Any) -> Any:
    """Return a new table transaction or a no-op for dependency-free fakes."""
    connection = getattr(table, "connection", None)
    in_transaction = getattr(connection, "in_transaction", None)
    if in_transaction is True or in_transaction is None:
        return nullcontext()
    transaction = getattr(connection, "transaction", None)
    if transaction is None:
        raise RuntimeError(
            "Analysis-NWB registration requires a DataJoint transaction."
        )
    return transaction


def _v1ca1_git_commit() -> str | None:
    """Return the local V1-CA1 HEAD without enforcing a particular commit."""
    return _git_commit(Path(__file__).resolve().parents[3])


def _spyglass_git_commit() -> str | None:
    """Return the runtime Spyglass checkout HEAD without enforcing the pin."""
    try:
        import spyglass
    except ModuleNotFoundError:
        return None
    package_path = getattr(spyglass, "__file__", None)
    if package_path is None:
        return None
    return _git_commit(Path(package_path).resolve().parent)


def _intervals_to_frame(intervals: Any, *, epoch: str) -> Any:
    """Convert a Pynapple-like IntervalSet to a detector-table dataframe."""
    import pandas as pd

    as_dataframe = getattr(intervals, "as_dataframe", None)
    if callable(as_dataframe):
        frame = as_dataframe().reset_index(drop=True)
        frame = frame.rename(
            columns={"start": "start_time", "end": "end_time", "stop": "end_time"}
        )
    else:
        starts = getattr(intervals, "start", None)
        stops = getattr(intervals, "end", None)
        if starts is None or stops is None:
            raise TypeError("Loaded ripple intervals do not expose start/end values.")
        frame = pd.DataFrame({"start_time": starts, "end_time": stops})
    missing = [name for name in ("start_time", "end_time") if name not in frame]
    if missing:
        raise ValueError(f"Loaded ripple intervals are missing columns {missing!r}.")
    frame["epoch"] = str(epoch)
    return frame


def _ripple_detector_values(
    ripple_row: Mapping[str, Any],
) -> tuple[float, bool]:
    """Return validated detector threshold and database-safe speed gate."""
    actual_threshold = ripple_row.get("detector_zscore_threshold")
    if isinstance(actual_threshold, bool) or not isinstance(
        actual_threshold, Real
    ):
        raise TypeError(
            "RippleIntervals.detector_zscore_threshold must be one numeric scalar."
        )
    threshold = float(actual_threshold)
    if not math.isfinite(threshold):
        raise ValueError("RippleIntervals.detector_zscore_threshold must be finite.")
    speed_gated = _database_bool(
        ripple_row.get("speed_gated"),
        name="RippleIntervals.speed_gated",
    )
    return threshold, speed_gated


def _validate_ripple_provenance(
    ripple_row: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> None:
    """Match selected detector metadata to explicit upstream expectations."""
    actual_threshold, speed_gated = _ripple_detector_values(ripple_row)
    if not math.isclose(
        actual_threshold,
        float(parameters["expected_detector_zscore_threshold"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "RippleIntervals.detector_zscore_threshold does not match "
            "expected_detector_zscore_threshold."
        )
    if parameters["require_speed_gated"] and not speed_gated:
        raise ValueError(
            "Selected RippleIntervals row must be explicitly speed-gated."
        )


def _parameter_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Translate stored scalar columns to the database-free analysis API."""
    return {
        "bin_size_s": parameters["bin_size_s"],
        "time_before_s": parameters["time_before_s"],
        "time_after_s": parameters["time_after_s"],
        "response_window": (
            parameters["response_window_start_s"],
            parameters["response_window_end_s"],
        ),
        "baseline_window": (
            parameters["baseline_window_start_s"],
            parameters["baseline_window_end_s"],
        ),
    }


def _analysis_region(value: Any) -> str:
    """Require the canonical lowercase project analysis region."""
    region = str(value).strip()
    if region not in {"v1", "ca1"}:
        raise ValueError("region must be the canonical lowercase 'v1' or 'ca1'.")
    return region


def _parameter_snapshot_field(
    parameters: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, str]:
    """Return one immutable parameter-value digest for a selection row."""
    from v1ca1.spyglass.selection import provenance_sha256

    return {field_name: provenance_sha256(dict(parameters))}


def _validate_frozen_parameters(
    selection: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    """Require current Manual parameters to match their selection snapshot."""
    current = _parameter_snapshot_field(parameters, field_name=field_name)[field_name]
    if str(selection.get(field_name, "")) != current:
        raise ValueError(
            "Analysis parameters changed after selection insertion: "
            f"{field_name}. Create a new selection."
        )


def _ripple_modulation_selection_row(
    *,
    key: Mapping[str, Any],
    ripples_table: Any,
    epoch_intervals_table: Any,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable RippleModulation selection."""
    from v1ca1.spyglass.selection import selection_uuid

    natural_key = {
        field_name: key[field_name]
        for field_name in (
            "nwb_file_name",
            "epoch",
            "ripple_modulation_param_name",
            "region_sorted_spikes_group_id",
        )
    }
    _fetch1_dict(ripples_table, natural_key)
    _fetch1_dict(epoch_intervals_table, natural_key)
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": natural_key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    if str(region_row.get("nwb_file_name")) != str(
        natural_key["nwb_file_name"]
    ):
        raise ValueError(
            "RippleModulation and RegionSortedSpikesGroup must belong to "
            "the same NWB file."
        )
    parameters = _validate_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    identity_payload = {**natural_key, **parameter_snapshot}
    return {
        "ripple_modulation_id": selection_uuid(
            "RippleModulation",
            identity_payload,
        ),
        **natural_key,
        **parameter_snapshot,
    }


def _ripple_glm_group_snapshot(
    row: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    """Return immutable regional sorting fields for one RippleGLM role."""
    return {
        f"{role}_sorting_group_members_sha256": str(
            row["sorting_group_members_sha256"]
        ),
        f"{role}_unit_filter_params_sha256": str(
            row["unit_filter_params_sha256"]
        ),
        f"{role}_selected_units_sha256": str(
            row["selected_units_sha256"]
        ),
        f"{role}_n_units": int(row["n_units"]),
    }


def _ripple_glm_source_intervals_sha256(
    ripple_table: Any,
    *,
    epoch: str,
) -> str:
    """Digest every raw source ripple start/end pair before GLM selection."""
    import pandas as pd

    from v1ca1.spyglass.selection import provenance_sha256

    as_dataframe = getattr(ripple_table, "as_dataframe", None)
    if callable(as_dataframe):
        table = as_dataframe().copy()
    elif isinstance(ripple_table, pd.DataFrame):
        table = ripple_table.copy()
    else:
        table = pd.DataFrame(ripple_table)
    table = table.rename(
        columns={"start": "start_time", "stop": "end_time", "end": "end_time"}
    )
    if "epoch" in table:
        table = table.loc[table["epoch"].astype(str) == str(epoch)]
    missing = [
        field_name
        for field_name in ("start_time", "end_time")
        if field_name not in table
    ]
    if missing:
        raise ValueError(
            f"RippleGLM source intervals are missing columns {missing!r}."
        )
    bounds = table[["start_time", "end_time"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError(
            "RippleGLM source intervals must contain finite positive bounds."
        )
    ordered = bounds[np.argsort(bounds[:, 0], kind="stable")]
    return provenance_sha256(
        [
            {"start_time": float(start), "end_time": float(end)}
            for start, end in ordered
        ]
    )


def _ripple_glm_provenance_sha256(
    ripple_row: Mapping[str, Any],
) -> str:
    """Digest detector settings and NWB object pointers for one ripple row."""
    from v1ca1.spyglass.selection import provenance_sha256

    fields = (
        "detector_zscore_threshold",
        "speed_gated",
        "detection_parameters",
        "provenance_path",
        "provenance_object_id",
        "source_table_path",
        "source_table_object_id",
        "source_object_path",
        "source_object_id",
    )
    return provenance_sha256(
        {field_name: ripple_row.get(field_name) for field_name in fields}
    )


def _ripple_glm_selection_row(
    *,
    key: Mapping[str, Any],
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    parameters_table: Any,
    nwbfile_table: Any | None = None,
    ripple_table: Any | None = None,
    epoch_interval: Any | None = None,
) -> dict[str, Any]:
    """Validate, freeze, and identify one epoch-level RippleGLM selection."""
    from v1ca1.spyglass.ripple_glm import (
        OUTPUT_RULE,
        OUTPUT_RULE_SHA256,
        prepare_ripple_glm_event_selection,
    )
    from v1ca1.spyglass.selection import selection_uuid

    nwb_file_name = str(key["nwb_file_name"])
    epoch = str(key["epoch"])
    ripple_row = _fetch1_dict(
        ripples_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    parameters = _validate_ripple_glm_parameter_row(
        _fetch1_dict(
            parameters_table,
            {"ripple_glm_param_name": key["ripple_glm_param_name"]},
        )
    )
    _validate_ripple_provenance(ripple_row, parameters)
    detector_zscore_threshold, speed_gated = _ripple_detector_values(
        ripple_row
    )

    group_rows = {
        role: _fetch1_dict(
            region_sorted_spikes_group_table,
            {
                "region_sorted_spikes_group_id": key[
                    f"{role}_region_sorted_spikes_group_id"
                ]
            },
        )
        for role in ("source", "target")
    }
    expected_regions = {"source": "ca1", "target": "v1"}
    for role, expected_region in expected_regions.items():
        row = group_rows[role]
        if str(row.get("region_name")) != expected_region:
            raise ValueError(
                f"RippleGLM {role} group must select region "
                f"{expected_region!r}."
            )
        if str(row.get("nwb_file_name")) != nwb_file_name:
            raise ValueError(
                "RippleGLM ripple and regional sorting inputs must belong "
                "to the same NWB file."
            )

    if ripple_table is None or epoch_interval is None:
        if nwbfile_table is None:
            raise ValueError(
                "nwbfile_table is required when ripple and epoch intervals "
                "are not supplied."
            )
        loaded_ripples, loaded_epoch = _load_ripple_glm_interval_inputs(
            nwb_file_name=nwb_file_name,
            ripple_row=ripple_row,
            epoch_row=_fetch1_dict(
                epoch_intervals_table,
                {"nwb_file_name": nwb_file_name, "epoch": epoch},
            ),
            nwbfile_table=nwbfile_table,
        )
        if ripple_table is None:
            ripple_table = loaded_ripples
        if epoch_interval is None:
            epoch_interval = loaded_epoch
    event_selection = prepare_ripple_glm_event_selection(
        epoch=epoch,
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        **_ripple_glm_parameter_kwargs(parameters),
    )
    if int(event_selection["n_ripples_before_selection"]) != int(
        ripple_row["ripple_count"]
    ):
        raise ValueError(
            "RippleIntervals.ripple_count disagrees with its NWB interval data."
        )
    if dict(OUTPUT_RULE) != dict(table_specs.RIPPLE_GLM_OUTPUT_RULE):
        raise RuntimeError(
            "RippleGLM table and database-free output rules have diverged."
        )

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="ripple_glm_parameters_sha256",
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "source_region_sorted_spikes_group_id": key[
            "source_region_sorted_spikes_group_id"
        ],
        "target_region_sorted_spikes_group_id": key[
            "target_region_sorted_spikes_group_id"
        ],
        "ripple_glm_param_name": parameters["ripple_glm_param_name"],
        "source_region": "ca1",
        "target_region": "v1",
        "source_ripple_count": int(ripple_row["ripple_count"]),
        "detector_zscore_threshold": detector_zscore_threshold,
        "speed_gated": speed_gated,
        "source_ripple_intervals_sha256": (
            _ripple_glm_source_intervals_sha256(
                ripple_table,
                epoch=epoch,
            )
        ),
        "ripple_provenance_sha256": _ripple_glm_provenance_sha256(
            ripple_row
        ),
        "n_selected_ripples": int(
            event_selection["n_ripples_after_window_bounds"]
        ),
        "selected_ripple_events_sha256": str(
            event_selection["selected_ripple_events_sha256"]
        ),
        **_ripple_glm_group_snapshot(group_rows["source"], role="source"),
        **_ripple_glm_group_snapshot(group_rows["target"], role="target"),
        **parameter_snapshot,
        "ripple_glm_output_rule_sha256": OUTPUT_RULE_SHA256,
    }
    return {
        "ripple_glm_id": selection_uuid("RippleGLM", natural_key),
        **natural_key,
    }


def _ripple_cross_region_xcorr_selection_row(
    *,
    key: Mapping[str, Any],
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    parameters_table: Any,
    nwbfile_table: Any | None = None,
    ripple_table: Any | None = None,
) -> dict[str, Any]:
    """Validate, freeze, and identify one exact-ripple xcorr selection."""
    from v1ca1.spyglass import ripple_cross_region_xcorr
    from v1ca1.spyglass.selection import selection_uuid

    nwb_file_name = str(key["nwb_file_name"])
    epoch = str(key["epoch"])
    ripple_row = _fetch1_dict(
        ripples_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    parameters = _validate_ripple_cross_region_xcorr_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "ripple_cross_region_xcorr_param_name": key[
                    "ripple_cross_region_xcorr_param_name"
                ]
            },
        )
    )
    _validate_ripple_provenance(ripple_row, parameters)
    detector_zscore_threshold, speed_gated = _ripple_detector_values(
        ripple_row
    )
    group_rows = {
        role: _fetch1_dict(
            region_sorted_spikes_group_table,
            {
                "region_sorted_spikes_group_id": key[
                    f"{role}_region_sorted_spikes_group_id"
                ]
            },
        )
        for role in ("source", "target")
    }
    for role, expected_region in (("source", "ca1"), ("target", "v1")):
        row = group_rows[role]
        if str(row.get("region_name")) != expected_region:
            raise ValueError(
                f"RippleCrossRegionXCorr {role} group must select region "
                f"{expected_region!r}."
            )
        if str(row.get("nwb_file_name")) != nwb_file_name:
            raise ValueError(
                "RippleCrossRegionXCorr ripple and regional sorting inputs must "
                "belong to the same NWB file."
            )
    if ripple_table is None:
        if nwbfile_table is None:
            raise ValueError(
                "nwbfile_table is required when ripple intervals are not "
                "supplied."
            )
        ripple_table, _ = _load_ripple_glm_interval_inputs(
            nwb_file_name=nwb_file_name,
            ripple_row=ripple_row,
            epoch_row=epoch_row,
            nwbfile_table=nwbfile_table,
        )
    event_selection = (
        ripple_cross_region_xcorr.prepare_ripple_cross_region_xcorr_event_selection(
            epoch=epoch,
            ripple_table=ripple_table,
        )
    )
    normalized_ripples = event_selection["selected_ripple_table"]
    if int(event_selection["n_ripples"]) != int(ripple_row["ripple_count"]):
        raise ValueError(
            "RippleIntervals.ripple_count disagrees with its NWB interval data."
        )
    if dict(ripple_cross_region_xcorr.OUTPUT_RULE) != dict(
        table_specs.RIPPLE_CROSS_REGION_XCORR_OUTPUT_RULE
    ):
        raise RuntimeError(
            "RippleCrossRegionXCorr table and database-free output rules have "
            "diverged."
        )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="ripple_cross_region_xcorr_parameters_sha256",
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "source_region_sorted_spikes_group_id": key[
            "source_region_sorted_spikes_group_id"
        ],
        "target_region_sorted_spikes_group_id": key[
            "target_region_sorted_spikes_group_id"
        ],
        "ripple_cross_region_xcorr_param_name": parameters[
            "ripple_cross_region_xcorr_param_name"
        ],
        "source_region": "ca1",
        "target_region": "v1",
        "source_ripple_count": int(ripple_row["ripple_count"]),
        "detector_zscore_threshold": detector_zscore_threshold,
        "speed_gated": speed_gated,
        "selected_ripple_intervals_sha256": (
            event_selection["selected_ripple_intervals_sha256"]
        ),
        "ripple_provenance_sha256": _ripple_glm_provenance_sha256(
            ripple_row
        ),
        **_ripple_glm_group_snapshot(group_rows["source"], role="source"),
        **_ripple_glm_group_snapshot(group_rows["target"], role="target"),
        **parameter_snapshot,
        "ripple_cross_region_xcorr_output_rule_sha256": (
            ripple_cross_region_xcorr.OUTPUT_RULE_SHA256
        ),
    }
    return {
        "ripple_cross_region_xcorr_id": selection_uuid(
            "RippleCrossRegionXCorr",
            natural_key,
        ),
        **natural_key,
    }


def _interval_set_sha256(intervals: Any) -> str:
    """Digest exact ordered seconds bounds for one selected interval set."""
    from v1ca1.spyglass.selection import provenance_sha256

    starts = np.asarray(getattr(intervals, "start"), dtype=float).reshape(-1)
    ends = np.asarray(getattr(intervals, "end"), dtype=float).reshape(-1)
    if starts.shape != ends.shape or not np.all(np.isfinite(starts)) or not np.all(
        np.isfinite(ends)
    ):
        raise ValueError("Selected interval bounds must be aligned and finite.")
    if np.any(ends <= starts) or (
        len(starts) > 1 and np.any(starts[1:] < ends[:-1])
    ):
        raise ValueError("Selected intervals must be positive and non-overlapping.")
    return provenance_sha256(
        {"start_time_s": starts.tolist(), "end_time_s": ends.tolist()}
    )


def _movement_rates_table_sha256(table: Any) -> str:
    """Digest stable unit identity and exact epoch movement rates."""
    from v1ca1.spyglass.selection import provenance_sha256

    required = (
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "movement_firing_rate_hz",
        "firing_rate_status",
    )
    missing = [name for name in required if name not in table]
    if missing:
        raise ValueError(
            "MovementFiringRate artifact is missing unit-rate fields "
            f"{missing!r}."
        )
    rates = np.asarray(table["movement_firing_rate_hz"], dtype=float)
    finite = np.isfinite(rates)
    if np.any(rates[finite] < 0.0):
        raise ValueError("Finite movement firing rates must be non-negative.")
    statuses = set(table["firing_rate_status"].astype(str))
    if not np.all(finite) and not statuses.issubset(
        {"no_movement", "no_valid_position"}
    ):
        raise ValueError(
            "Nonfinite movement rates are allowed only for terminal movement artifacts."
        )
    return provenance_sha256(
        [
            {
                "spikesorting_merge_id": str(row["spikesorting_merge_id"]),
                "unit_id": str(row["unit_id"]),
                "stable_unit_id": str(row["stable_unit_id"]),
                "movement_firing_rate_hz": (
                    float(row["movement_firing_rate_hz"])
                    if np.isfinite(float(row["movement_firing_rate_hz"]))
                    else None
                ),
                "firing_rate_status": str(row["firing_rate_status"]),
            }
            for row in table.loc[:, list(required)].to_dict("records")
        ]
    )


def _movement_result_semantic_sha256(row: Mapping[str, Any]) -> str:
    """Digest one movement result without random NWB storage identifiers."""
    storage_fields = {
        "analysis_file_name",
        "movement_firing_rate_object_id",
        "movement_intervals_object_id",
    }
    return _catalog_row_sha256(
        {
            name: value
            for name, value in dict(row).items()
            if name not in storage_fields
        }
    )


def _catalog_row_sha256(row: Mapping[str, Any]) -> str:
    """Digest one catalog row without transient DataJoint fields."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            str(name): value
            for name, value in dict(row).items()
            if not str(name).startswith("_")
        }
    )


def _registered_nwb_source_identity(
    *,
    nwbfile_table: Any,
    nwb_file_name: str,
) -> dict[str, Any]:
    """Return authoritative DataJoint filepath identity and current byte size."""
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    custom_loader = getattr(
        nwbfile_table,
        "get_registered_source_identity",
        None,
    )
    if callable(custom_loader):
        registered = dict(custom_loader(nwb_file_name))
    else:
        table = (
            nwbfile_table()
            if isinstance(nwbfile_table, type)
            else nwbfile_table
        )
        try:
            attribute = table.heading.attributes["nwb_file_abs_path"]
            external = table.connection.schemas[attribute.database].external[
                attribute.store
            ]
            relative_path = nwb_path.relative_to(
                Path(external.spec["stage"])
            ).as_posix()
            contents_hash, registered_size = (
                external & {"filepath": relative_path}
            ).fetch1("contents_hash", "size")
        except (AttributeError, KeyError, LookupError, ValueError) as exc:
            raise ValueError(
                "Could not resolve the Nwbfile filepath@raw registry identity."
            ) from exc
        registered = {
            "contents_hash": contents_hash,
            "size": registered_size,
        }
    contents_hash = registered.get("contents_hash")
    if contents_hash is None or not str(contents_hash).strip():
        raise ValueError(
            "Nwbfile filepath@raw must have a registered contents_hash."
        )
    try:
        import uuid

        contents_hash_text = str(uuid.UUID(str(contents_hash)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(
            "Nwbfile filepath@raw contents_hash must be a UUID."
        ) from exc
    registered_size = registered.get("size")
    if isinstance(registered_size, bool) or not isinstance(
        registered_size,
        Integral,
    ):
        raise TypeError("Nwbfile filepath@raw size must be an integer.")
    registered_size = int(registered_size)
    if registered_size < 0:
        raise ValueError("Nwbfile filepath@raw size must be non-negative.")
    try:
        actual_size = nwb_path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(
            f"Registered raw NWB file is unavailable: {nwb_path}"
        ) from exc
    if actual_size != registered_size:
        raise ValueError(
            "Raw NWB byte size differs from its DataJoint filepath registry."
        )
    return {
        "nwb_path": nwb_path,
        "registered_source_contents_hash": contents_hash_text,
        "registered_source_size_bytes": registered_size,
    }


def _epoch_motor_array_sha256(values: Any) -> str:
    """Digest one exact numeric array including shape and normalized dtype."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _epoch_motor_json_value(value: Any) -> Any:
    """Return one recursive JSON-safe graph-input snapshot."""
    if isinstance(value, Mapping):
        return {
            str(name): _epoch_motor_json_value(item)
            for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return _epoch_motor_json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_epoch_motor_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _epoch_motor_json_value(value.item())
    return value


def _epoch_motor_graph_input_sha256(graph: Mapping[str, Any]) -> str:
    """Digest the exact graph arguments passed to track linearization."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(_epoch_motor_json_value(dict(graph)))


def _epoch_motor_position_source_sha256(
    row: Mapping[str, Any],
    *,
    values: np.ndarray,
    timestamps: np.ndarray,
) -> str:
    """Digest one catalog row and its exact already-offset NWB samples."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            "catalog_row_sha256": _catalog_row_sha256(row),
            "timestamps_sha256": _epoch_motor_array_sha256(timestamps),
            "values_sha256": _epoch_motor_array_sha256(values),
        }
    )


def _epoch_motor_behavior_selection_row(
    *,
    key: Mapping[str, Any],
    epoch_intervals_table: Any,
    position_table: Any,
    movement_parameters_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
    position_inputs: Mapping[str, Any] | None = None,
    trajectory_interval_sets: Mapping[str, Any] | None = None,
    graph_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate, freeze, and UUID one exact run-epoch motor selection."""
    from v1ca1.spyglass import epoch_motor_behavior as motor_behavior
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid

    nwb_file_name = str(key["nwb_file_name"])
    epoch = str(key["epoch"])
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    if str(epoch_row.get("epoch_type")) != "run":
        raise ValueError("EpochMotorBehavior requires an explicit run epoch.")
    parameters = _validate_epoch_motor_behavior_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "epoch_motor_behavior_param_name": key[
                    "epoch_motor_behavior_param_name"
                ]
            },
        )
    )
    movement_row = _validate_movement_parameter_row(
        _fetch1_dict(
            movement_parameters_table,
            {"movement_param_name": key["movement_param_name"]},
        )
    )
    movement_snapshot = motor_behavior.validate_movement_parameter_snapshot(
        movement_row
    )

    primary_name = str(key["primary_position_series_name"])
    reference_name = str(key["orientation_reference_position_series_name"])
    primary_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": primary_name,
        },
    )
    reference_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": reference_name,
        },
    )
    supplied_positions = dict(position_inputs or {})
    primary_position = supplied_positions.get("primary_position")
    reference_position = supplied_positions.get(
        "orientation_reference_position"
    )
    if primary_position is None:
        primary_position = position_table.load_position(
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "position_series_name": primary_name,
            },
            apply_analysis_offset=True,
        )
    if reference_position is None:
        reference_position = position_table.load_position(
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "position_series_name": reference_name,
            },
            apply_analysis_offset=True,
        )
    validated_position = motor_behavior.validate_position_inputs(
        epoch=epoch,
        primary_position=primary_position,
        orientation_reference_position=reference_position,
        primary_position_row=primary_row,
        orientation_reference_position_row=reference_row,
    )

    trajectory_rows: dict[str, dict[str, Any]] = {}
    graph_rows: dict[str, dict[str, Any]] = {}
    source_fields: dict[str, Any] = {}
    supplied_intervals = dict(trajectory_interval_sets or {})
    supplied_graphs = dict(graph_inputs or {})
    selected_intervals: dict[str, Any] = {}
    selected_graphs: dict[str, Mapping[str, Any]] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type or str(
            key.get(configuration_field, trajectory_type)
        ) != trajectory_type:
            raise ValueError(
                "EpochMotorBehavior trajectory and graph aliases must match "
                "the four canonical natural-direction paths."
            )
        trajectory_key = {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "trajectory_type": trajectory_type,
        }
        graph_key = {
            "nwb_file_name": nwb_file_name,
            "configuration_name": trajectory_type,
        }
        trajectory_rows[trajectory_type] = _fetch1_dict(
            trajectory_intervals_table, trajectory_key
        )
        graph_rows[trajectory_type] = _fetch1_dict(
            wtrack_graph_table, graph_key
        )
        selected_intervals[trajectory_type] = supplied_intervals.get(
            trajectory_type
        )
        if selected_intervals[trajectory_type] is None:
            selected_intervals[trajectory_type] = (
                trajectory_intervals_table.load_intervals(trajectory_key)
            )
        selected_graphs[trajectory_type] = supplied_graphs.get(
            trajectory_type
        )
        if selected_graphs[trajectory_type] is None:
            selected_graphs[trajectory_type] = wtrack_graph_table.load_graph(
                graph_key
            )
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    motor_behavior._validate_trajectory_inputs(selected_intervals)
    motor_behavior._validate_graph_inputs(selected_graphs)
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="epoch_motor_behavior_parameters_sha256",
    )
    trajectory_hashes = {
        trajectory_type: _interval_set_sha256(
            selected_intervals[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    trajectory_row_hashes = {
        trajectory_type: _catalog_row_sha256(
            trajectory_rows[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_row_hashes = {
        trajectory_type: _catalog_row_sha256(
            graph_rows[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_input_hashes = {
        trajectory_type: _epoch_motor_graph_input_sha256(
            selected_graphs[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    timestamps = np.asarray(validated_position["timestamps"], dtype=float)
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "primary_position_series_name": primary_name,
        "orientation_reference_position_series_name": reference_name,
        "movement_param_name": movement_snapshot["movement_param_name"],
        **source_fields,
        "epoch_motor_behavior_param_name": parameters[
            "epoch_motor_behavior_param_name"
        ],
        "primary_position_role": validated_position["primary_position_role"],
        "orientation_reference_position_role": validated_position[
            "orientation_reference_position_role"
        ],
        "position_offset_samples": int(
            validated_position["position_offset_samples"]
        ),
        "epoch_interval_row_sha256": _catalog_row_sha256(
            epoch_row
        ),
        "primary_position_row_sha256": _catalog_row_sha256(
            primary_row
        ),
        "orientation_reference_position_row_sha256": (
            _catalog_row_sha256(reference_row)
        ),
        "aligned_position_timestamps_sha256": _epoch_motor_array_sha256(
            timestamps
        ),
        "primary_position_source_sha256": (
            _epoch_motor_position_source_sha256(
                primary_row,
                values=validated_position["primary_values"],
                timestamps=timestamps,
            )
        ),
        "orientation_reference_position_source_sha256": (
            _epoch_motor_position_source_sha256(
                reference_row,
                values=validated_position[
                    "orientation_reference_values"
                ],
                timestamps=timestamps,
            )
        ),
        "trajectory_rows_sha256_by_type": trajectory_row_hashes,
        "trajectory_intervals_sha256_by_type": trajectory_hashes,
        "graph_rows_sha256_by_trajectory": graph_row_hashes,
        "graph_inputs_sha256_by_trajectory": graph_input_hashes,
        "movement_parameters_sha256": movement_snapshot[
            "movement_parameters_sha256"
        ],
        **parameter_snapshot,
        "epoch_motor_behavior_output_rule_sha256": provenance_sha256(
            dict(motor_behavior.OUTPUT_RULE)
        ),
    }
    return {
        "epoch_motor_behavior_id": selection_uuid(
            "EpochMotorBehavior", natural_key
        ),
        **natural_key,
    }


def _cv_pca_epoch_bounds_sha256(row: Mapping[str, Any]) -> str:
    """Digest exact epoch bounds and condition fields used by cvPCA."""
    from v1ca1.spyglass.selection import provenance_sha256

    start = float(row["start_time"])
    stop = float(row["stop_time"])
    if not math.isfinite(start) or not math.isfinite(stop) or stop <= start:
        raise ValueError("CVPCA epoch bounds must be finite and positive.")
    return provenance_sha256(
        {
            "start_time_s": start,
            "stop_time_s": stop,
            "nwb_epoch_start_time_s": float(row["nwb_epoch_start_time"]),
            "nwb_epoch_stop_time_s": float(row["nwb_epoch_stop_time"]),
            "epoch_type": row.get("epoch_type"),
            "condition": row.get("condition"),
            "is_light": (
                None
                if row.get("is_light") is None
                else _database_bool(row["is_light"], name="is_light")
            ),
        }
    )


def _cv_pca_position_snapshot(
    *,
    condition: str,
    row: Mapping[str, Any],
    position: Any,
) -> dict[str, Any]:
    """Validate and digest one untrimmed Position source."""
    from v1ca1.spyglass import cv_pca

    if str(row.get("spatial_unit")) != "cm":
        raise ValueError("CVPCA Position rows must use centimeters.")
    offset = int(row.get("analysis_start_offset_samples", -1))
    if offset != cv_pca.DEFAULT_POSITION_OFFSET_SAMPLES:
        raise ValueError("CVPCA Position rows must declare a 10-sample offset.")
    values, timestamps = cv_pca._position_arrays(
        position, name=f"{condition}_position"
    )
    if len(values) != int(row.get("sample_count", -1)):
        raise ValueError(
            "CVPCA requires an untrimmed Position load matching sample_count."
        )
    if offset > len(values):
        raise ValueError("CVPCA position offset exceeds its untrimmed series.")
    return {
        "values": values,
        "timestamps": timestamps,
        "row_sha256": _catalog_row_sha256(row),
        "values_sha256": _epoch_motor_array_sha256(values),
        "timestamps_sha256": _epoch_motor_array_sha256(timestamps),
        "position_role": str(row.get("position_role", "")),
        "position_offset_samples": offset,
    }


def _cv_pca_selection_row(
    *,
    key: Mapping[str, Any],
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
    session_table: Any,
    position_inputs_by_condition: Mapping[str, Any] | None = None,
    movement_artifacts_by_condition: Mapping[str, Mapping[str, Any]] | None = None,
    trajectory_interval_sets_by_condition: Mapping[str, Mapping[str, Any]]
    | None = None,
    graph_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate, freeze, and UUID one exact light/dark cvPCA selection."""
    from v1ca1.spyglass import cv_pca
    from v1ca1.spyglass.selection import (
        provenance_sha256,
        selection_uuid,
        unit_identity_sha256,
    )

    nwb_file_name = str(key["nwb_file_name"])
    epochs = {
        condition: str(key[f"{condition}_epoch"])
        for condition in ("light", "dark")
    }
    if epochs["light"] == epochs["dark"]:
        raise ValueError("CVPCA light and dark epochs must differ.")
    epoch_rows = {
        condition: _fetch1_dict(
            epoch_intervals_table,
            {"nwb_file_name": nwb_file_name, "epoch": epoch},
        )
        for condition, epoch in epochs.items()
    }
    if str(epoch_rows["light"].get("epoch_type")) != "run" or not _database_bool(
        epoch_rows["light"].get("is_light"), name="light is_light"
    ):
        raise ValueError("CVPCA light_epoch must be an explicit light run epoch.")
    if str(epoch_rows["dark"].get("epoch_type")) != "run" or str(
        epoch_rows["dark"].get("condition")
    ) != "dark":
        raise ValueError("CVPCA dark_epoch must be an explicit dark run epoch.")
    if epoch_rows["dark"].get("is_light") is not None and _database_bool(
        epoch_rows["dark"]["is_light"], name="dark is_light"
    ):
        raise ValueError("CVPCA dark_epoch cannot be marked as light.")

    group_key = {
        "region_sorted_spikes_group_id": key["region_sorted_spikes_group_id"]
    }
    group_row = _fetch1_dict(region_sorted_spikes_group_table, group_key)
    if str(group_row.get("nwb_file_name")) != nwb_file_name:
        raise ValueError("CVPCA regional sorting group must share the selected NWB.")
    region = _analysis_region(group_row.get("region_name"))
    parameters = _validate_cv_pca_parameter_row(
        _fetch1_dict(
            parameters_table,
            {"cv_pca_param_name": key["cv_pca_param_name"]},
        )
    )
    effective_parameters = cv_pca.validate_cv_pca_parameters(
        region=region,
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "cv_pca_param_name"
        },
    )
    output_rule_sha256 = provenance_sha256(dict(table_specs.CV_PCA_OUTPUT_RULE))
    if output_rule_sha256 != cv_pca.OUTPUT_RULE_SHA256:
        raise RuntimeError("Passive and standalone CVPCA output rules differ.")

    animal_name, session_date = _session_identity(
        session_table, {"nwb_file_name": nwb_file_name}
    )
    supplied_positions = dict(position_inputs_by_condition or {})
    supplied_movement = dict(movement_artifacts_by_condition or {})
    movement_results: dict[str, dict[str, Any]] = {}
    movement_selections: dict[str, dict[str, Any]] = {}
    movement_parameters: dict[str, dict[str, Any]] = {}
    movement_loaded: dict[str, dict[str, Any]] = {}
    position_rows: dict[str, dict[str, Any]] = {}
    positions: dict[str, Any] = {}
    position_snapshots: dict[str, dict[str, Any]] = {}
    for condition in ("light", "dark"):
        movement_key = {
            "movement_firing_rate_id": key[
                f"{condition}_movement_firing_rate_id"
            ]
        }
        movement_results[condition] = _fetch1_dict(
            movement_firing_rate_table, movement_key
        )
        movement_selections[condition] = _fetch1_dict(
            movement_firing_rate_selection_table, movement_key
        )
        movement_selection = movement_selections[condition]
        movement_result = movement_results[condition]
        for field_name, expected in (
            ("nwb_file_name", nwb_file_name),
            ("epoch", epochs[condition]),
        ):
            if str(movement_selection.get(field_name)) != expected:
                raise ValueError(
                    f"CVPCA {condition} MovementFiringRate has wrong {field_name}."
                )
        _validate_region_movement_identity(
            analysis_name="CVPCA",
            region_row=group_row,
            movement_selection=movement_selection,
            movement_result=movement_result,
        )
        movement_parameters[condition] = _validate_movement_parameter_row(
            _fetch1_dict(movement_parameters_table, movement_selection)
        )
        _validate_frozen_parameters(
            movement_selection,
            movement_parameters[condition],
            field_name="movement_parameters_sha256",
        )
        if condition in supplied_movement:
            movement_loaded[condition] = dict(supplied_movement[condition])
        else:
            movement_loaded[condition] = _load_movement_result_artifacts(
                result_row=movement_result,
                movement_firing_rate_table=movement_firing_rate_table,
                parameters=movement_parameters[condition],
                expected_metadata={
                    "animal_name": animal_name,
                    "date": session_date,
                    "region": region,
                    "epoch": epochs[condition],
                },
            )
        if str(movement_loaded[condition]["analysis_status"]) != str(
            movement_result.get("analysis_status")
        ):
            raise ValueError("CVPCA MovementFiringRate status is inconsistent.")

        position_key = {
            "nwb_file_name": nwb_file_name,
            "epoch": epochs[condition],
            "position_series_name": movement_selection[
                "position_series_name"
            ],
        }
        position_rows[condition] = _fetch1_dict(position_table, position_key)
        positions[condition] = supplied_positions.get(condition)
        if positions[condition] is None:
            positions[condition] = position_table.load_position(
                position_key, apply_analysis_offset=False
            )
        position_snapshots[condition] = _cv_pca_position_snapshot(
            condition=condition,
            row=position_rows[condition],
            position=positions[condition],
        )

    if movement_parameters["light"] != movement_parameters["dark"]:
        raise ValueError("CVPCA MovementFiringRate rows must share one definition.")
    if position_snapshots["light"]["position_role"] != position_snapshots[
        "dark"
    ]["position_role"] or not position_snapshots["light"]["position_role"]:
        raise ValueError("CVPCA Position rows must share one non-empty role.")

    identity_columns = ("spikesorting_merge_id", "unit_id")
    identity_rows: dict[str, list[dict[str, str]]] = {}
    for condition in ("light", "dark"):
        table = movement_loaded[condition]["table"]
        identity_rows[condition] = [
            {name: str(row[name]) for name in identity_columns}
            for row in table.loc[:, list(identity_columns)].to_dict("records")
        ]
        if unit_identity_sha256(identity_rows[condition]) != str(
            group_row["selected_units_sha256"]
        ):
            raise ValueError(
                "CVPCA MovementFiringRate artifact units differ from its group."
            )
    if identity_rows["light"] != identity_rows["dark"]:
        raise ValueError("CVPCA light and dark unit order must match exactly.")

    trajectory_rows: dict[str, dict[str, dict[str, Any]]] = {
        "light": {},
        "dark": {},
    }
    selected_intervals: dict[str, dict[str, Any]] = {"light": {}, "dark": {}}
    source_fields: dict[str, Any] = {}
    supplied_intervals = dict(trajectory_interval_sets_by_condition or {})
    for condition in ("light", "dark"):
        condition_inputs = dict(supplied_intervals.get(condition, {}))
        for trajectory_type in _DPP_TRAJECTORY_TYPES:
            field_name = f"{condition}_{trajectory_type}_trajectory_type"
            if str(key.get(field_name, trajectory_type)) != trajectory_type:
                raise ValueError("CVPCA trajectory aliases must be canonical.")
            trajectory_key = {
                "nwb_file_name": nwb_file_name,
                "epoch": epochs[condition],
                "trajectory_type": trajectory_type,
            }
            trajectory_rows[condition][trajectory_type] = _fetch1_dict(
                trajectory_intervals_table, trajectory_key
            )
            selected_intervals[condition][trajectory_type] = condition_inputs.get(
                trajectory_type
            )
            if selected_intervals[condition][trajectory_type] is None:
                selected_intervals[condition][trajectory_type] = (
                    trajectory_intervals_table.load_intervals(trajectory_key)
                )
            source_fields[field_name] = trajectory_type
        cv_pca._validate_trajectory_mapping(
            selected_intervals[condition],
            name=f"{condition}_trajectory_intervals",
        )

    selected_graphs: dict[str, Mapping[str, Any]] = {}
    graph_rows: dict[str, dict[str, Any]] = {}
    supplied_graphs = dict(graph_inputs or {})
    for trajectory_type in ("center_to_left", "center_to_right"):
        field_name = f"{trajectory_type}_configuration_name"
        if str(key.get(field_name, trajectory_type)) != trajectory_type:
            raise ValueError("CVPCA graph aliases must be canonical.")
        graph_key = {
            "nwb_file_name": nwb_file_name,
            "configuration_name": trajectory_type,
        }
        graph_rows[trajectory_type] = _fetch1_dict(
            wtrack_graph_table, graph_key
        )
        selected_graphs[trajectory_type] = supplied_graphs.get(trajectory_type)
        if selected_graphs[trajectory_type] is None:
            selected_graphs[trajectory_type] = wtrack_graph_table.load_graph(
                graph_key
            )
        source_fields[field_name] = trajectory_type
    cv_pca._normalize_graph_inputs(selected_graphs)

    trajectory_row_hashes = {
        condition: {
            trajectory: _catalog_row_sha256(
                trajectory_rows[condition][trajectory]
            )
            for trajectory in _DPP_TRAJECTORY_TYPES
        }
        for condition in ("light", "dark")
    }
    trajectory_interval_hashes = {
        condition: {
            trajectory: _interval_set_sha256(
                selected_intervals[condition][trajectory]
            )
            for trajectory in _DPP_TRAJECTORY_TYPES
        }
        for condition in ("light", "dark")
    }
    graph_row_hashes = {
        trajectory: _catalog_row_sha256(graph_rows[trajectory])
        for trajectory in selected_graphs
    }
    graph_input_hashes = {
        trajectory: _epoch_motor_graph_input_sha256(selected_graphs[trajectory])
        for trajectory in selected_graphs
    }
    movement_parameter_hash = str(
        movement_selections["light"]["movement_parameters_sha256"]
    )
    if movement_parameter_hash != str(
        movement_selections["dark"]["movement_parameters_sha256"]
    ):
        raise ValueError("CVPCA movement parameter snapshots must match.")

    natural_key = {
        "nwb_file_name": nwb_file_name,
        "light_epoch": epochs["light"],
        "dark_epoch": epochs["dark"],
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "light_movement_firing_rate_id": key[
            "light_movement_firing_rate_id"
        ],
        "dark_movement_firing_rate_id": key[
            "dark_movement_firing_rate_id"
        ],
        **source_fields,
        "cv_pca_param_name": parameters["cv_pca_param_name"],
        "light_position_series_name": str(
            movement_selections["light"]["position_series_name"]
        ),
        "dark_position_series_name": str(
            movement_selections["dark"]["position_series_name"]
        ),
        "position_role": position_snapshots["light"]["position_role"],
        "position_offset_samples": cv_pca.DEFAULT_POSITION_OFFSET_SAMPLES,
        **{
            f"{condition}_epoch_row_sha256": (
                _catalog_row_sha256(epoch_rows[condition])
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_epoch_bounds_sha256": _cv_pca_epoch_bounds_sha256(
                epoch_rows[condition]
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_position_row_sha256": position_snapshots[condition][
                "row_sha256"
            ]
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_position_values_sha256": position_snapshots[
                condition
            ]["values_sha256"]
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_position_timestamps_sha256": position_snapshots[
                condition
            ]["timestamps_sha256"]
            for condition in ("light", "dark")
        },
        "region_group_row_sha256": _catalog_row_sha256(
            group_row
        ),
        "sorting_group_members_sha256": str(
            group_row["sorting_group_members_sha256"]
        ),
        "unit_filter_params_sha256": str(group_row["unit_filter_params_sha256"]),
        "selected_units_sha256": str(group_row["selected_units_sha256"]),
        "n_input_units": int(group_row["n_units"]),
        "movement_param_name": str(
            movement_parameters["light"]["movement_param_name"]
        ),
        "movement_parameters_sha256": movement_parameter_hash,
        **{
            f"{condition}_movement_selection_row_sha256": (
                _catalog_row_sha256(
                    movement_selections[condition]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_result_row_sha256": (
                _movement_result_semantic_sha256(
                    movement_results[condition]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_rates_sha256": (
                _movement_rates_table_sha256(
                    movement_loaded[condition]["table"]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_support_sha256": (
                _interval_set_sha256(
                    movement_loaded[condition]["movement_intervals"]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_analysis_status": str(
                movement_loaded[condition]["analysis_status"]
            )
            for condition in ("light", "dark")
        },
        "trajectory_rows_sha256_by_epoch_and_type": trajectory_row_hashes,
        "trajectory_intervals_sha256_by_epoch_and_type": (
            trajectory_interval_hashes
        ),
        "graph_rows_sha256_by_trajectory": graph_row_hashes,
        "graph_inputs_sha256_by_trajectory": graph_input_hashes,
        "cv_pca_parameters_sha256": provenance_sha256(parameters),
        "cv_pca_effective_parameters_sha256": provenance_sha256(
            effective_parameters
        ),
        "cv_pca_output_rule_sha256": output_rule_sha256,
    }
    return {
        "cv_pca_id": selection_uuid("CVPCA", natural_key),
        **natural_key,
    }


def _movement_firing_rate_selection_row(
    *,
    key: Mapping[str, Any],
    position_table: Any,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable movement firing-rate selection."""
    from v1ca1.spyglass.selection import selection_uuid

    natural_key = {
        field_name: key[field_name]
        for field_name in (
            "nwb_file_name",
            "epoch",
            "position_series_name",
            "movement_param_name",
            "region_sorted_spikes_group_id",
        )
    }
    position_row = _fetch1_dict(position_table, natural_key)
    if position_row.get("spatial_unit") != "cm":
        raise ValueError("MovementFiringRate position must use centimeters.")
    parameters = _validate_movement_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": natural_key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    if str(region_row.get("nwb_file_name")) != str(
        natural_key["nwb_file_name"]
    ):
        raise ValueError(
            "MovementFiringRate and RegionSortedSpikesGroup must belong to "
            "the same NWB file."
        )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="movement_parameters_sha256",
    )
    identity_payload = {**natural_key, **parameter_snapshot}
    return {
        "movement_firing_rate_id": selection_uuid(
            "MovementFiringRate",
            identity_payload,
        ),
        **natural_key,
        **parameter_snapshot,
    }


def _validate_region_movement_identity(
    *,
    analysis_name: str,
    region_row: Mapping[str, Any],
    movement_selection: Mapping[str, Any],
    movement_result: Mapping[str, Any],
) -> None:
    """Require movement artifacts to use one exact registered region group."""
    region_group_id = str(region_row["region_sorted_spikes_group_id"])
    if str(movement_selection.get("region_sorted_spikes_group_id")) != (
        region_group_id
    ):
        raise ValueError(
            f"{analysis_name} regional spikes and MovementFiringRate must "
            "use the same RegionSortedSpikesGroup row."
        )
    if str(region_row.get("nwb_file_name")) != str(
        movement_selection.get("nwb_file_name")
    ):
        raise ValueError(
            f"{analysis_name} regional spikes and MovementFiringRate must "
            "use the same nwb_file_name."
        )
    if str(region_row.get("selected_units_sha256")) != str(
        movement_result.get("selected_units_sha256")
    ):
        raise ValueError(
            f"{analysis_name} regional spikes and MovementFiringRate must "
            "contain the same persistent units."
        )
    if int(region_row.get("n_units", -1)) != int(
        movement_result.get("n_units", -2)
    ):
        raise ValueError(
            f"{analysis_name} regional spikes and MovementFiringRate unit "
            "counts disagree."
        )


def _path_specific_place_tuning_curve_selection_row(
    *,
    key: Mapping[str, Any],
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable path-specific tuning selection."""
    from v1ca1.spyglass.selection import selection_uuid

    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    _fetch1_dict(movement_firing_rate_table, movement_key)
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    for field_name in ("nwb_file_name", "epoch"):
        supplied = key.get(field_name, movement_selection[field_name])
        if str(supplied) != str(movement_selection[field_name]):
            raise ValueError(
                "PathSpecificPlaceTuningCurve and MovementFiringRate must select "
                f"the same {field_name}."
            )
    trial_subset = str(key["trial_subset"])
    if trial_subset not in {"all", "odd", "even"}:
        raise ValueError("trial_subset must be 'all', 'odd', or 'even'.")
    natural_key = {
        "nwb_file_name": movement_selection["nwb_file_name"],
        "epoch": movement_selection["epoch"],
        "trajectory_type": key["trajectory_type"],
        "configuration_name": key["configuration_name"],
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        "tuning_curve_param_name": key["tuning_curve_param_name"],
        "trial_subset": trial_subset,
    }
    epoch_row = _fetch1_dict(epoch_intervals_table, natural_key)
    _fetch1_dict(trajectory_intervals_table, natural_key)
    graph_row = _fetch1_dict(wtrack_graph_table, natural_key)
    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    if natural_key["trajectory_type"] != natural_key["configuration_name"]:
        raise ValueError(
            "PathSpecificPlaceTuningCurve requires configuration_name to equal "
            "trajectory_type."
        )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("PathSpecificPlaceTuningCurve requires a run epoch.")
    if graph_row.get("coordinate_unit") != "cm":
        raise ValueError("PathSpecificPlaceTuningCurve graph must use centimeters.")
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    identity_payload = {**natural_key, **parameter_snapshot}
    return {
        "path_specific_place_tuning_curve_id": selection_uuid(
            "PathSpecificPlaceTuningCurve",
            identity_payload,
        ),
        **natural_key,
        **parameter_snapshot,
    }


def _dpp_tuning_curve_selection_row(
    *,
    key: Mapping[str, Any],
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable same-turn DPP selection."""
    from v1ca1.spyglass.dpp import TURN_TYPES, get_dpp_trajectory_pair
    from v1ca1.spyglass.selection import selection_uuid

    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    _fetch1_dict(movement_firing_rate_table, movement_key)
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    for field_name in ("nwb_file_name", "epoch"):
        supplied = key.get(field_name, movement_selection[field_name])
        if str(supplied) != str(movement_selection[field_name]):
            raise ValueError(
                "DPPTuningCurve and MovementFiringRate must select the same "
                f"{field_name}."
            )

    turn_type = str(key["turn_type"])
    if turn_type not in TURN_TYPES:
        raise ValueError(f"turn_type must be one of {TURN_TYPES!r}.")
    trial_subset = str(key["trial_subset"])
    if trial_subset not in {"all", "odd", "even"}:
        raise ValueError("trial_subset must be 'all', 'odd', or 'even'.")

    outbound_trajectory, inbound_trajectory = get_dpp_trajectory_pair(turn_type)
    expected_sources = {
        "outbound_trajectory_type": outbound_trajectory,
        "inbound_trajectory_type": inbound_trajectory,
        "outbound_configuration_name": outbound_trajectory,
        "inbound_configuration_name": inbound_trajectory,
    }
    for field_name, expected_value in expected_sources.items():
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                f"DPPTuningCurve {field_name} is fixed by turn_type="
                f"{turn_type!r} and must equal {expected_value!r}."
            )

    nwb_file_name = movement_selection["nwb_file_name"]
    epoch = movement_selection["epoch"]
    source_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
    }
    epoch_row = _fetch1_dict(epoch_intervals_table, source_key)
    for trajectory_type in (outbound_trajectory, inbound_trajectory):
        _fetch1_dict(
            trajectory_intervals_table,
            {**source_key, "trajectory_type": trajectory_type},
        )
    graph_rows = [
        _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": configuration_name,
            },
        )
        for configuration_name in (
            outbound_trajectory,
            inbound_trajectory,
        )
    ]
    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(
            parameters_table,
            {"tuning_curve_param_name": key["tuning_curve_param_name"]},
        )
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("DPPTuningCurve requires a run epoch.")
    if any(row.get("coordinate_unit") != "cm" for row in graph_rows):
        raise ValueError("DPPTuningCurve graphs must use centimeters.")

    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        **expected_sources,
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        "tuning_curve_param_name": key["tuning_curve_param_name"],
        "turn_type": turn_type,
        "trial_subset": trial_subset,
    }
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    identity_payload = {**natural_key, **parameter_snapshot}
    return {
        "dpp_tuning_curve_id": selection_uuid(
            "DPPTuningCurve",
            identity_payload,
        ),
        **natural_key,
        **parameter_snapshot,
    }


def _stability_selection_row(
    *,
    key: Mapping[str, Any],
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
) -> dict[str, Any]:
    """Validate and identify one odd/even path-specific curve pair."""
    from v1ca1.spyglass.selection import selection_uuid

    curve_ids = {
        subset: key[f"{subset}_path_specific_place_tuning_curve_id"]
        for subset in ("odd", "even")
    }
    curve_results: dict[str, dict[str, Any]] = {}
    curve_selections: dict[str, dict[str, Any]] = {}
    for subset, curve_id in curve_ids.items():
        curve_key = {"path_specific_place_tuning_curve_id": curve_id}
        curve_results[subset] = _fetch1_dict(tuning_curve_table, curve_key)
        curve_selections[subset] = _fetch1_dict(
            tuning_curve_selection_table,
            curve_key,
        )
        if str(curve_selections[subset].get("trial_subset")) != subset:
            raise ValueError(
                "PathSpecificPlaceStability requires matching odd and even "
                "PathSpecificPlaceTuningCurve rows."
            )

    shared_fields = (
        "nwb_file_name",
        "epoch",
        "trajectory_type",
        "configuration_name",
        "movement_firing_rate_id",
        "tuning_curve_param_name",
        "tuning_curve_parameters_sha256",
    )
    for field_name in shared_fields:
        if str(curve_selections["odd"].get(field_name)) != str(
            curve_selections["even"].get(field_name)
        ):
            raise ValueError(
                "Odd and even PathSpecificPlaceTuningCurve rows must share "
                f"the same {field_name}."
            )
    if str(curve_results["odd"].get("selected_units_sha256")) != str(
        curve_results["even"].get("selected_units_sha256")
    ):
        raise ValueError(
            "Odd and even PathSpecificPlaceTuningCurve rows must contain the "
            "same selected units."
        )

    identity_payload = {
        "odd_path_specific_place_tuning_curve_id": curve_ids["odd"],
        "even_path_specific_place_tuning_curve_id": curve_ids["even"],
    }
    return {
        "path_specific_place_stability_id": selection_uuid(
            "PathSpecificPlaceStability",
            identity_payload,
        ),
        **identity_payload,
    }


def _tuning_similarity_selection_row(
    *,
    key: Mapping[str, Any],
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable four-path similarity selection."""
    from v1ca1.spyglass.selection import selection_uuid

    curve_id_fields = {
        "center_to_left": "center_to_left_tuning_curve_id",
        "center_to_right": "center_to_right_tuning_curve_id",
        "left_to_center": "left_to_center_tuning_curve_id",
        "right_to_center": "right_to_center_tuning_curve_id",
    }
    curve_results: dict[str, dict[str, Any]] = {}
    curve_selections: dict[str, dict[str, Any]] = {}
    curve_ids: dict[str, Any] = {}
    for trajectory_type, field_name in curve_id_fields.items():
        curve_id = key[field_name]
        curve_key = {"path_specific_place_tuning_curve_id": curve_id}
        curve_results[trajectory_type] = _fetch1_dict(
            tuning_curve_table,
            curve_key,
        )
        curve_selections[trajectory_type] = _fetch1_dict(
            tuning_curve_selection_table,
            curve_key,
        )
        selection = curve_selections[trajectory_type]
        if str(selection.get("trajectory_type")) != trajectory_type:
            raise ValueError(
                "PathSpecificPlaceTuningSimilarity curve aliases must match "
                f"their trajectory types; {field_name} selected "
                f"{selection.get('trajectory_type')!r}."
            )
        if str(selection.get("configuration_name")) != trajectory_type:
            raise ValueError(
                "PathSpecificPlaceTuningSimilarity requires each graph "
                "configuration to match its trajectory type."
            )
        if str(selection.get("trial_subset")) != "all":
            raise ValueError(
                "PathSpecificPlaceTuningSimilarity requires four all-trial "
                "PathSpecificPlaceTuningCurve rows."
            )
        curve_ids[field_name] = curve_id

    reference_trajectory = "center_to_left"
    reference_selection = curve_selections[reference_trajectory]
    reference_result = curve_results[reference_trajectory]
    shared_selection_fields = (
        "nwb_file_name",
        "epoch",
        "movement_firing_rate_id",
        "tuning_curve_param_name",
        "tuning_curve_parameters_sha256",
    )
    for trajectory_type, selection in curve_selections.items():
        for field_name in shared_selection_fields:
            if str(selection.get(field_name)) != str(
                reference_selection.get(field_name)
            ):
                raise ValueError(
                    "PathSpecificPlaceTuningSimilarity curves must share "
                    f"the same {field_name}; mismatch for {trajectory_type!r}."
                )
        result = curve_results[trajectory_type]
        for field_name in (
            "selected_units_sha256",
            "n_units",
            "n_position_bins",
        ):
            if str(result.get(field_name)) != str(reference_result.get(field_name)):
                raise ValueError(
                    "PathSpecificPlaceTuningSimilarity curves must share "
                    f"the same {field_name}; mismatch for {trajectory_type!r}."
                )

    parameters = _validate_tuning_similarity_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "tuning_similarity_param_name": key[
                    "tuning_similarity_param_name"
                ]
            },
        )
    )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="tuning_similarity_parameters_sha256",
    )
    natural_key = {
        **curve_ids,
        "tuning_similarity_param_name": parameters[
            "tuning_similarity_param_name"
        ],
        **parameter_snapshot,
    }
    return {
        "path_specific_place_tuning_similarity_id": selection_uuid(
            "PathSpecificPlaceTuningSimilarity",
            natural_key,
        ),
        **natural_key,
    }


def _dpp_encoding_selection_row(
    *,
    key: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable four-model encoding selection."""
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid

    region_key = {
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ]
    }
    region_row = _fetch1_dict(region_sorted_spikes_group_table, region_key)
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(movement_firing_rate_table, movement_key)
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    if str(movement_result.get("analysis_status")) != "valid":
        raise ValueError(
            "DPPEncoding requires a valid MovementFiringRate row."
        )
    _validate_region_movement_identity(
        analysis_name="DPPEncoding",
        region_row=region_row,
        movement_selection=movement_selection,
        movement_result=movement_result,
    )

    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    for field_name, expected_value in (
        ("nwb_file_name", nwb_file_name),
        ("epoch", epoch),
    ):
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                "DPPEncoding supplied source does not match its "
                f"MovementFiringRate: {field_name}."
            )
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("DPPEncoding requires a run epoch.")

    parameters = _validate_dpp_encoding_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "dpp_encoding_param_name": key[
                    "dpp_encoding_param_name"
                ]
            },
        )
    )
    source_key = {"nwb_file_name": nwb_file_name, "epoch": epoch}
    source_fields: dict[str, Any] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        supplied_trajectory = str(key.get(trajectory_field, trajectory_type))
        if supplied_trajectory != trajectory_type:
            raise ValueError(
                f"DPPEncoding {trajectory_field} must equal "
                f"{trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {**source_key, "trajectory_type": trajectory_type},
        )
        if int(trajectory_row.get("interval_count", -1)) < parameters["n_folds"]:
            raise ValueError(
                f"DPPEncoding requires at least {parameters['n_folds']} "
                f"laps for {trajectory_type!r}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        supplied_configuration = str(
            key.get(configuration_field, trajectory_type)
        )
        if supplied_configuration != trajectory_type:
            raise ValueError(
                f"DPPEncoding {configuration_field} must equal "
                f"{trajectory_type!r}."
            )
        graph_row = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if graph_row.get("coordinate_unit") != "cm":
            raise ValueError("DPPEncoding graphs must use centimeters.")
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    full_w_name = str(
        key.get(
            "full_w_configuration_name",
            _DPP_FULL_GRAPH_CONFIGURATION_NAME,
        )
    )
    if full_w_name != _DPP_FULL_GRAPH_CONFIGURATION_NAME:
        raise ValueError(
            "DPPEncoding full_w_configuration_name must equal "
            f"{_DPP_FULL_GRAPH_CONFIGURATION_NAME!r}."
        )
    full_w_row = _fetch1_dict(
        wtrack_graph_table,
        {
            "nwb_file_name": nwb_file_name,
            "configuration_name": full_w_name,
        },
    )
    if full_w_row.get("coordinate_unit") != "cm":
        raise ValueError("DPPEncoding graphs must use centimeters.")
    source_fields["full_w_configuration_name"] = full_w_name

    stability_fields: dict[str, Any] = {}
    legacy_tuning_parameters_sha256 = provenance_sha256(
        dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    )
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        field_name = f"{trajectory_type}_stability_id"
        stability_id = key[field_name]
        stability_key = {
            "path_specific_place_stability_id": stability_id
        }
        stability_result = _fetch1_dict(stability_table, stability_key)
        stability_selection = _fetch1_dict(
            stability_selection_table,
            stability_key,
        )
        if str(stability_result.get("selected_units_sha256")) != str(
            movement_result.get("selected_units_sha256")
        ):
            raise ValueError(
                "DPPEncoding stability and movement rows must "
                "contain the same persistent units."
            )
        if int(stability_result.get("n_units", -1)) != int(
            movement_result.get("n_units", -2)
        ):
            raise ValueError(
                "DPPEncoding stability and movement rows must "
                "contain the same unit count."
            )
        if str(stability_result.get("analysis_status")) not in {
            "valid",
            "no_valid_units",
        }:
            raise ValueError(
                "DPPEncoding stability inputs must be valid or "
                "no_valid_units when their MovementFiringRate is valid."
            )
        for subset in ("odd", "even"):
            curve_id = stability_selection[
                f"{subset}_path_specific_place_tuning_curve_id"
            ]
            curve_selection = _fetch1_dict(
                tuning_curve_selection_table,
                {"path_specific_place_tuning_curve_id": curve_id},
            )
            expected = {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "configuration_name": trajectory_type,
                "movement_firing_rate_id": key["movement_firing_rate_id"],
                "tuning_curve_param_name": (
                    table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
                        "tuning_curve_param_name"
                    ]
                ),
                "tuning_curve_parameters_sha256": (
                    legacy_tuning_parameters_sha256
                ),
                "trial_subset": subset,
            }
            for expected_field, expected_value in expected.items():
                if str(curve_selection.get(expected_field)) != str(
                    expected_value
                ):
                    raise ValueError(
                        "DPPEncoding stability input does not match "
                        f"its {trajectory_type!r} slot: {expected_field}."
                    )
        stability_fields[field_name] = stability_id

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="dpp_encoding_parameters_sha256",
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        **source_fields,
        **stability_fields,
        "dpp_encoding_param_name": parameters[
            "dpp_encoding_param_name"
        ],
        **parameter_snapshot,
    }
    return {
        "dpp_encoding_id": selection_uuid(
            "DPPEncoding",
            natural_key,
        ),
        **natural_key,
    }


def _path_progression_decoding_selection_row(
    *,
    key: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable shared-cohort decoder selection."""
    from v1ca1.spyglass.path_progression_decoding import TRANSFER_SPEC_SHA256
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid

    region_key = {
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ]
    }
    region_row = _fetch1_dict(region_sorted_spikes_group_table, region_key)
    parameters = _validate_path_progression_decoding_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "path_progression_decoding_param_name": key[
                    "path_progression_decoding_param_name"
                ]
            },
        )
    )

    movement_sources: dict[str, dict[str, Any]] = {}
    for source_name, id_field in (
        ("target", "movement_firing_rate_id"),
        ("cohort", "cohort_movement_firing_rate_id"),
    ):
        movement_key = {"movement_firing_rate_id": key[id_field]}
        result = _fetch1_dict(movement_firing_rate_table, movement_key)
        selection = _fetch1_dict(
            movement_firing_rate_selection_table,
            movement_key,
        )
        movement_status = str(result.get("analysis_status"))
        expected_movement_status = (
            "no_units" if int(region_row.get("n_units", -1)) == 0 else "valid"
        )
        if movement_status != expected_movement_status:
            raise ValueError(
                "PathProgressionDecoding requires target and cohort "
                "MovementFiringRate rows matching the regional unit count."
            )
        _validate_region_movement_identity(
            analysis_name="PathProgressionDecoding",
            region_row=region_row,
            movement_selection=selection,
            movement_result=result,
        )
        epoch_row = _fetch1_dict(epoch_intervals_table, selection)
        if epoch_row.get("epoch_type") not in (None, "run"):
            raise ValueError(
                "PathProgressionDecoding requires run epochs."
            )
        position_row = _fetch1_dict(position_table, selection)
        if str(position_row.get("spatial_unit")) != "cm":
            raise ValueError(
                "PathProgressionDecoding position must use centimeters."
            )
        movement_sources[source_name] = {
            "id_field": id_field,
            "result": result,
            "selection": selection,
            "epoch_row": epoch_row,
            "position_row": position_row,
        }

    target = movement_sources["target"]
    cohort = movement_sources["cohort"]
    target_selection = target["selection"]
    cohort_selection = cohort["selection"]
    if str(target_selection["nwb_file_name"]) != str(
        cohort_selection["nwb_file_name"]
    ):
        raise ValueError(
            "Target and cohort movement sources must belong to the same NWB file."
        )
    for field_name in (
        "movement_param_name",
        "movement_parameters_sha256",
        "position_series_name",
    ):
        if str(target_selection.get(field_name)) != str(
            cohort_selection.get(field_name)
        ):
            raise ValueError(
                "Target and cohort movement sources must share the same "
                f"{field_name}."
            )
    for field_name in ("position_role", "analysis_start_offset_samples"):
        if str(target["position_row"].get(field_name)) != str(
            cohort["position_row"].get(field_name)
        ):
            raise ValueError(
                "Target and cohort movement sources must use the same "
                f"{field_name}."
            )

    nwb_file_name = str(target_selection["nwb_file_name"])
    epoch = str(target_selection["epoch"])
    cohort_epoch = str(cohort_selection["epoch"])
    for field_name, expected_value in (
        ("nwb_file_name", nwb_file_name),
        ("epoch", epoch),
        ("cohort_epoch", cohort_epoch),
    ):
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                "PathProgressionDecoding supplied source does not "
                f"match its movement rows: {field_name}."
            )

    source_key = {"nwb_file_name": nwb_file_name, "epoch": epoch}
    source_fields: dict[str, Any] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                f"PathProgressionDecoding {trajectory_field} must "
                f"equal {trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {**source_key, "trajectory_type": trajectory_type},
        )
        if int(trajectory_row.get("interval_count", -1)) < 1:
            raise ValueError(
                "PathProgressionDecoding requires at least one "
                f"lap for {trajectory_type!r}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "PathProgressionDecoding graph aliases must match "
                "their trajectory types."
            )
        graph_row = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if str(graph_row.get("coordinate_unit")) != "cm":
            raise ValueError(
                "PathProgressionDecoding graphs must use centimeters."
            )
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    legacy_parameters_sha256 = provenance_sha256(
        dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    )
    stability_fields: dict[str, Any] = {}
    for source_name, prefix in (("target", ""), ("cohort", "cohort_")):
        movement_source = movement_sources[source_name]
        movement_id = key[movement_source["id_field"]]
        source_epoch = str(movement_source["selection"]["epoch"])
        for trajectory_type in _DPP_TRAJECTORY_TYPES:
            field_name = f"{prefix}{trajectory_type}_stability_id"
            stability_key = {
                "path_specific_place_stability_id": key[field_name]
            }
            stability_result = _fetch1_dict(stability_table, stability_key)
            stability_selection = _fetch1_dict(
                stability_selection_table,
                stability_key,
            )
            if str(stability_result.get("selected_units_sha256")) != str(
                movement_source["result"].get("selected_units_sha256")
            ) or int(stability_result.get("n_units", -1)) != int(
                movement_source["result"].get("n_units", -2)
            ):
                raise ValueError(
                    "PathProgressionDecoding stability and movement "
                    "sources must contain identical persistent units."
                )
            allowed_stability_statuses = (
                {"no_units"}
                if int(region_row.get("n_units", -1)) == 0
                else {"valid", "no_valid_units"}
            )
            if str(stability_result.get("analysis_status")) not in (
                allowed_stability_statuses
            ):
                raise ValueError(
                    "PathProgressionDecoding stability sources must "
                    "be valid or no_valid_units."
                )
            for subset in ("odd", "even"):
                curve_id = stability_selection[
                    f"{subset}_path_specific_place_tuning_curve_id"
                ]
                curve_selection = _fetch1_dict(
                    tuning_curve_selection_table,
                    {"path_specific_place_tuning_curve_id": curve_id},
                )
                expected = {
                    "nwb_file_name": nwb_file_name,
                    "epoch": source_epoch,
                    "trajectory_type": trajectory_type,
                    "configuration_name": trajectory_type,
                    "movement_firing_rate_id": movement_id,
                    "tuning_curve_param_name": (
                        table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
                            "tuning_curve_param_name"
                        ]
                    ),
                    "tuning_curve_parameters_sha256": legacy_parameters_sha256,
                    "trial_subset": subset,
                }
                for expected_field, expected_value in expected.items():
                    if str(curve_selection.get(expected_field)) != str(
                        expected_value
                    ):
                        raise ValueError(
                            "PathProgressionDecoding stability input "
                            f"does not match {source_name} {trajectory_type!r}: "
                            f"{expected_field}."
                        )
            stability_fields[field_name] = key[field_name]

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="path_progression_decoding_parameters_sha256",
    )
    eligibility_rule_sha256 = provenance_sha256(
        dict(table_specs.PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE)
    )
    decoding_output_rule_sha256 = provenance_sha256(
        dict(table_specs.PATH_PROGRESSION_DECODING_OUTPUT_RULE)
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        "cohort_movement_firing_rate_id": key[
            "cohort_movement_firing_rate_id"
        ],
        **source_fields,
        **stability_fields,
        "path_progression_decoding_param_name": parameters[
            "path_progression_decoding_param_name"
        ],
        "cohort_epoch": cohort_epoch,
        **parameter_snapshot,
        "eligibility_rule_sha256": eligibility_rule_sha256,
        "transfer_spec_sha256": TRANSFER_SPEC_SHA256,
        "decoding_output_rule_sha256": decoding_output_rule_sha256,
    }
    return {
        "path_progression_decoding_id": selection_uuid(
            "PathProgressionDecoding",
            natural_key,
        ),
        **natural_key,
    }


def _path_specific_place_decoding_selection_row(
    *,
    key: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable within-epoch place decoder."""
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    parameters = _validate_path_specific_place_decoding_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "path_specific_place_decoding_param_name": key[
                    "path_specific_place_decoding_param_name"
                ]
            },
        )
    )
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    _validate_region_movement_identity(
        analysis_name="PathSpecificPlaceDecoding",
        region_row=region_row,
        movement_selection=movement_selection,
        movement_result=movement_result,
    )
    allowed_statuses = (
        {"no_units"}
        if int(region_row.get("n_units", -1)) == 0
        else {"valid", "no_valid_position", "no_movement"}
    )
    if str(movement_result.get("analysis_status")) not in allowed_statuses:
        raise ValueError(
            "PathSpecificPlaceDecoding movement status is incompatible "
            "with its regional unit count."
        )

    epoch_row = _fetch1_dict(epoch_intervals_table, movement_selection)
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("PathSpecificPlaceDecoding requires one run epoch.")
    position_row = _fetch1_dict(position_table, movement_selection)
    if str(position_row.get("spatial_unit")) != "cm":
        raise ValueError(
            "PathSpecificPlaceDecoding position must use centimeters."
        )
    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    for field_name, expected_value in (
        ("nwb_file_name", nwb_file_name),
        ("epoch", epoch),
    ):
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                "PathSpecificPlaceDecoding supplied source does not match "
                f"its movement row: {field_name}."
            )

    source_fields: dict[str, Any] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                f"PathSpecificPlaceDecoding {trajectory_field} must equal "
                f"{trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
            },
        )
        if int(trajectory_row.get("interval_count", -1)) < int(
            parameters["n_folds"]
        ):
            raise ValueError(
                "PathSpecificPlaceDecoding requires at least n_folds laps "
                f"for {trajectory_type!r}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "PathSpecificPlaceDecoding graph aliases must match their "
                "trajectory types."
            )
        graph_row = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if str(graph_row.get("coordinate_unit")) != "cm":
            raise ValueError(
                "PathSpecificPlaceDecoding graphs must use centimeters."
            )
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="path_specific_place_decoding_parameters_sha256",
    )
    output_rule_sha256 = provenance_sha256(
        dict(table_specs.PATH_SPECIFIC_PLACE_DECODING_OUTPUT_RULE)
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        **source_fields,
        "path_specific_place_decoding_param_name": parameters[
            "path_specific_place_decoding_param_name"
        ],
        **parameter_snapshot,
        "path_specific_place_decoding_output_rule_sha256": (
            output_rule_sha256
        ),
    }
    return {
        "path_specific_place_decoding_id": selection_uuid(
            "PathSpecificPlaceDecoding",
            natural_key,
        ),
        **natural_key,
    }


def _validate_motor_position_rows(
    *,
    primary_position_row: Mapping[str, Any],
    orientation_reference_position_row: Mapping[str, Any],
) -> None:
    """Require two distinct centimeter position series on one sample grid."""
    primary_name = str(primary_position_row.get("position_series_name", ""))
    reference_name = str(
        orientation_reference_position_row.get("position_series_name", "")
    )
    if not primary_name or not reference_name or primary_name == reference_name:
        raise ValueError(
            "MotorEncoding requires distinct primary and "
            "orientation-reference position series."
        )
    for label, row in (
        ("primary", primary_position_row),
        ("orientation-reference", orientation_reference_position_row),
    ):
        if str(row.get("spatial_unit")) != "cm":
            raise ValueError(
                f"MotorEncoding {label} position must use centimeters."
            )
    for field_name in (
        "nwb_file_name",
        "epoch",
        "start_index",
        "stop_index_exclusive",
        "sample_count",
        "analysis_start_offset_samples",
        "start_time",
        "stop_time",
        "first_frame",
        "last_frame",
        "video_series_name",
    ):
        if str(primary_position_row.get(field_name)) != str(
            orientation_reference_position_row.get(field_name)
        ):
            raise ValueError(
                "MotorEncoding position series must share aligned "
                f"sampling metadata: {field_name}."
            )


def _motor_encoding_selection_row(
    *,
    key: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable nine-model motor comparison."""
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    parameters = _validate_motor_encoding_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "motor_encoding_param_name": key[
                    "motor_encoding_param_name"
                ]
            },
        )
    )
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    _validate_region_movement_identity(
        analysis_name="MotorEncoding",
        region_row=region_row,
        movement_selection=movement_selection,
        movement_result=movement_result,
    )
    allowed_movement_statuses = (
        {"no_units"}
        if int(region_row.get("n_units", -1)) == 0
        else {"valid", "no_valid_position", "no_movement"}
    )
    if str(movement_result.get("analysis_status")) not in (
        allowed_movement_statuses
    ):
        raise ValueError(
            "MotorEncoding movement status is incompatible with "
            "its regional unit count."
        )

    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    for field_name, expected_value in (
        ("nwb_file_name", nwb_file_name),
        ("epoch", epoch),
    ):
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                "MotorEncoding supplied source does not match its "
                f"MovementFiringRate: {field_name}."
            )
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("MotorEncoding requires one run epoch.")

    primary_position_name = str(
        key.get(
            "primary_position_series_name",
            movement_selection["position_series_name"],
        )
    )
    if primary_position_name != str(
        movement_selection["position_series_name"]
    ):
        raise ValueError(
            "MotorEncoding primary position must be the position "
            "used by MovementFiringRate."
        )
    if "orientation_reference_position_series_name" not in key:
        raise ValueError(
            "MotorEncoding requires an explicit "
            "orientation_reference_position_series_name."
        )
    orientation_reference_name = str(
        key["orientation_reference_position_series_name"]
    )
    primary_position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": primary_position_name,
        },
    )
    orientation_reference_position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": orientation_reference_name,
        },
    )
    _validate_motor_position_rows(
        primary_position_row=primary_position_row,
        orientation_reference_position_row=(
            orientation_reference_position_row
        ),
    )

    source_fields: dict[str, Any] = {
        "primary_position_series_name": primary_position_name,
        "orientation_reference_position_series_name": (
            orientation_reference_name
        ),
    }
    outer_n_folds = int(parameters["outer_n_folds"])
    inner_n_folds = int(parameters["inner_n_folds"])
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                f"MotorEncoding {trajectory_field} must equal "
                f"{trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
            },
        )
        n_laps = int(trajectory_row.get("interval_count", -1))
        largest_outer_test = int(math.ceil(n_laps / outer_n_folds))
        minimum_outer_train = n_laps - largest_outer_test
        if n_laps < outer_n_folds or minimum_outer_train < inner_n_folds:
            raise ValueError(
                "MotorEncoding requires enough laps for outer and "
                f"nested inner CV for {trajectory_type!r}; found {n_laps}, "
                f"outer_n_folds={outer_n_folds}, "
                f"inner_n_folds={inner_n_folds}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "MotorEncoding graph aliases must match their "
                "trajectory types."
            )
        graph_row = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if str(graph_row.get("coordinate_unit")) != "cm":
            raise ValueError(
                "MotorEncoding graphs must use centimeters."
            )
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    full_w_name = str(
        key.get(
            "full_w_configuration_name",
            _DPP_FULL_GRAPH_CONFIGURATION_NAME,
        )
    )
    if full_w_name != _DPP_FULL_GRAPH_CONFIGURATION_NAME:
        raise ValueError(
            "MotorEncoding full_w_configuration_name must equal "
            f"{_DPP_FULL_GRAPH_CONFIGURATION_NAME!r}."
        )
    full_w_row = _fetch1_dict(
        wtrack_graph_table,
        {
            "nwb_file_name": nwb_file_name,
            "configuration_name": full_w_name,
        },
    )
    if str(full_w_row.get("coordinate_unit")) != "cm":
        raise ValueError("MotorEncoding graphs must use centimeters.")
    source_fields["full_w_configuration_name"] = full_w_name

    stability_fields: dict[str, Any] = {}
    legacy_tuning_parameters_sha256 = provenance_sha256(
        dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    )
    movement_status = str(movement_result.get("analysis_status"))
    allowed_stability_statuses = {
        "valid": {"valid", "no_valid_units"},
        "no_valid_position": {"no_valid_position"},
        "no_movement": {"no_movement"},
        "no_units": {"no_units"},
    }[movement_status]
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        field_name = f"{trajectory_type}_stability_id"
        stability_id = key[field_name]
        stability_key = {
            "path_specific_place_stability_id": stability_id
        }
        stability_result = _fetch1_dict(stability_table, stability_key)
        stability_selection = _fetch1_dict(
            stability_selection_table,
            stability_key,
        )
        if str(stability_result.get("selected_units_sha256")) != str(
            movement_result.get("selected_units_sha256")
        ) or int(stability_result.get("n_units", -1)) != int(
            movement_result.get("n_units", -2)
        ):
            raise ValueError(
                "MotorEncoding stability and movement rows must contain "
                "the same persistent units."
            )
        if str(stability_result.get("analysis_status")) not in (
            allowed_stability_statuses
        ):
            raise ValueError(
                "MotorEncoding stability status is incompatible with its "
                "MovementFiringRate status."
            )
        for subset in ("odd", "even"):
            curve_id = stability_selection[
                f"{subset}_path_specific_place_tuning_curve_id"
            ]
            curve_selection = _fetch1_dict(
                tuning_curve_selection_table,
                {"path_specific_place_tuning_curve_id": curve_id},
            )
            expected = {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "configuration_name": trajectory_type,
                "movement_firing_rate_id": key["movement_firing_rate_id"],
                "tuning_curve_param_name": (
                    table_specs.LEGACY_TUNING_CURVE_PARAMETERS[
                        "tuning_curve_param_name"
                    ]
                ),
                "tuning_curve_parameters_sha256": (
                    legacy_tuning_parameters_sha256
                ),
                "trial_subset": subset,
            }
            for expected_field, expected_value in expected.items():
                if str(curve_selection.get(expected_field)) != str(
                    expected_value
                ):
                    raise ValueError(
                        "MotorEncoding stability input does not match its "
                        f"{trajectory_type!r} slot: {expected_field}."
                    )
        stability_fields[field_name] = stability_id

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="motor_encoding_parameters_sha256",
    )
    model_spec_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_MODEL_SPEC)
    )
    output_rule_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_OUTPUT_RULE)
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        **source_fields,
        **stability_fields,
        "motor_encoding_param_name": parameters[
            "motor_encoding_param_name"
        ],
        **parameter_snapshot,
        "motor_encoding_model_spec_sha256": model_spec_sha256,
        "motor_encoding_output_rule_sha256": (
            output_rule_sha256
        ),
    }
    return {
        "motor_encoding_id": selection_uuid(
            "MotorEncoding",
            natural_key,
        ),
        **natural_key,
    }


def _dark_light_glm_selection_row(
    *,
    key: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable coupled dark/light GLM."""
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    parameters = _validate_dark_light_glm_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "dark_light_glm_param_name": key[
                    "dark_light_glm_param_name"
                ]
            },
        )
    )
    movement_results: dict[str, dict[str, Any]] = {}
    movement_selections: dict[str, dict[str, Any]] = {}
    for condition_name in ("dark", "light"):
        movement_key = {
            "movement_firing_rate_id": key[
                f"{condition_name}_movement_firing_rate_id"
            ]
        }
        movement_results[condition_name] = _fetch1_dict(
            movement_firing_rate_table,
            movement_key,
        )
        movement_selections[condition_name] = _fetch1_dict(
            movement_firing_rate_selection_table,
            movement_key,
        )

    reference_selection = movement_selections["dark"]
    reference_result = movement_results["dark"]
    for condition_name in ("dark", "light"):
        movement_selection = movement_selections[condition_name]
        movement_result = movement_results[condition_name]
        _validate_region_movement_identity(
            analysis_name="DarkLightGLM",
            region_row=region_row,
            movement_selection=movement_selection,
            movement_result=movement_result,
        )
        allowed_movement_statuses = (
            {"no_units"}
            if int(region_row.get("n_units", -1)) == 0
            else {"valid", "no_valid_position", "no_movement"}
        )
        if str(movement_result.get("analysis_status")) not in (
            allowed_movement_statuses
        ):
            raise ValueError(
                "DarkLightGLM movement status is incompatible with its "
                "regional unit count."
            )
        position_row = _fetch1_dict(
            position_table,
            {
                "nwb_file_name": movement_selection["nwb_file_name"],
                "epoch": movement_selection["epoch"],
                "position_series_name": movement_selection[
                    "position_series_name"
                ],
            },
        )
        if str(position_row.get("spatial_unit")) != "cm":
            raise ValueError("DarkLightGLM positions must use centimeters.")

    for field_name in (
        "nwb_file_name",
        "region_sorted_spikes_group_id",
        "movement_param_name",
        "movement_parameters_sha256",
        "position_series_name",
    ):
        if str(reference_selection.get(field_name)) != str(
            movement_selections["light"].get(field_name)
        ):
            raise ValueError(
                "DarkLightGLM dark and light movement rows must share their "
                f"frozen source snapshot: {field_name}."
            )
    for field_name in ("selected_units_sha256", "n_units"):
        if str(reference_result.get(field_name)) != str(
            movement_results["light"].get(field_name)
        ):
            raise ValueError(
                "DarkLightGLM dark and light movement results must contain "
                f"the same units: {field_name}."
            )

    nwb_file_name = str(reference_selection["nwb_file_name"])
    epochs = {
        condition_name: str(movement_selections[condition_name]["epoch"])
        for condition_name in ("dark", "light")
    }
    if epochs["dark"] == epochs["light"]:
        raise ValueError("DarkLightGLM requires distinct dark and light epochs.")
    epoch_rows: dict[str, dict[str, Any]] = {}
    for condition_name in ("dark", "light"):
        epoch_field = f"{condition_name}_epoch"
        if epoch_field in key and str(key[epoch_field]) != epochs[condition_name]:
            raise ValueError(
                f"DarkLightGLM {epoch_field} must match its MovementFiringRate."
            )
        epoch_row = _fetch1_dict(
            epoch_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epochs[condition_name],
            },
        )
        if str(epoch_row.get("epoch_type")) != "run":
            raise ValueError("DarkLightGLM requires two explicit run epochs.")
        epoch_rows[condition_name] = epoch_row
    dark_is_light = epoch_rows["dark"].get("is_light")
    if str(epoch_rows["dark"].get("condition")) != "dark" or (
        dark_is_light is None or bool(dark_is_light)
    ):
        raise ValueError(
            "DarkLightGLM dark epoch must have condition='dark' and "
            "is_light=False."
        )
    light_condition = str(epoch_rows["light"].get("condition"))
    light_is_light = epoch_rows["light"].get("is_light")
    if light_condition not in {"AB", "BA", "gray", "bright"} or (
        light_is_light is None or not bool(light_is_light)
    ):
        raise ValueError(
            "DarkLightGLM light epoch must have an explicit light condition "
            "and is_light=True."
        )

    source_fields: dict[str, Any] = {
        "dark_epoch": epochs["dark"],
        "light_epoch": epochs["light"],
    }
    n_folds = int(parameters["n_folds"])
    for condition_name in ("dark", "light"):
        for trajectory_type in _DPP_TRAJECTORY_TYPES:
            trajectory_field = (
                f"{condition_name}_{trajectory_type}_trajectory_type"
            )
            if str(key.get(trajectory_field, trajectory_type)) != trajectory_type:
                raise ValueError(
                    f"DarkLightGLM {trajectory_field} must equal "
                    f"{trajectory_type!r}."
                )
            trajectory_row = _fetch1_dict(
                trajectory_intervals_table,
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": epochs[condition_name],
                    "trajectory_type": trajectory_type,
                },
            )
            if int(trajectory_row.get("interval_count", -1)) < n_folds:
                raise ValueError(
                    "DarkLightGLM requires at least n_folds laps in every "
                    f"epoch and trajectory; {condition_name} "
                    f"{trajectory_type!r} has "
                    f"{trajectory_row.get('interval_count')!r}."
                )
            source_fields[trajectory_field] = trajectory_type

    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "DarkLightGLM graph aliases must match their trajectory types."
            )
        graph_row = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if str(graph_row.get("coordinate_unit")) != "cm":
            raise ValueError("DarkLightGLM graphs must use centimeters.")
        source_fields[configuration_field] = trajectory_type

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="dark_light_glm_parameters_sha256",
    )
    output_rule_sha256 = provenance_sha256(
        dict(table_specs.DARK_LIGHT_GLM_OUTPUT_RULE)
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "dark_movement_firing_rate_id": key[
            "dark_movement_firing_rate_id"
        ],
        "light_movement_firing_rate_id": key[
            "light_movement_firing_rate_id"
        ],
        **source_fields,
        "dark_light_glm_param_name": parameters[
            "dark_light_glm_param_name"
        ],
        **parameter_snapshot,
        "dark_light_glm_output_rule_sha256": output_rule_sha256,
    }
    return {
        "dark_light_glm_id": selection_uuid("DarkLightGLM", natural_key),
        **natural_key,
    }


def _load_swap_dark_light_snapshot(
    *,
    dark_light_glm_table: Any,
    dark_light_glm_id: Any,
) -> dict[str, Any]:
    """Load one DarkLightGLM NWB and freeze its semantic model hashes."""
    from v1ca1.spyglass.dark_light_glm import (
        dark_light_glm_nwb_hashes,
        dark_light_glm_selected_model_sha256s,
    )

    key = {"dark_light_glm_id": dark_light_glm_id}
    result_row = _fetch1_dict(dark_light_glm_table, key)
    bundle = _load_dark_light_glm_result(
        result_row=result_row,
        dark_light_glm_table=dark_light_glm_table,
    )
    result_hashes = dark_light_glm_nwb_hashes(bundle)
    selected_hashes = dark_light_glm_selected_model_sha256s(bundle)
    return {
        "result_row": result_row,
        "bundle": bundle,
        "dark_light_glm_sha256": result_hashes["dark_light_glm_sha256"],
        "selected_model_sha256_by_model": selected_hashes,
        "parameter_sha256": str(bundle["parameters"]["parameter_sha256"]),
        "output_rule_sha256": str(bundle["parameters"]["output_rule_sha256"]),
        "analysis_status": str(bundle["analysis_status"]),
        "metadata": dict(bundle["metadata"]),
    }


def _swap_glm_selection_row(
    *,
    key: Mapping[str, Any],
    dark_light_glm_table: Any,
    dark_light_glm_selection_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
    dark_light_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and identify one immutable held-out swapped-light score."""
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid
    from v1ca1.spyglass.swap_glm import OUTPUT_RULE_SHA256

    dark_light_key = {"dark_light_glm_id": key["dark_light_glm_id"]}
    dark_light_selection = _fetch1_dict(
        dark_light_glm_selection_table,
        dark_light_key,
    )
    snapshot = (
        dict(dark_light_snapshot)
        if dark_light_snapshot is not None
        else _load_swap_dark_light_snapshot(
            dark_light_glm_table=dark_light_glm_table,
            dark_light_glm_id=key["dark_light_glm_id"],
        )
    )
    parameters = _validate_swap_glm_parameter_row(
        _fetch1_dict(
            parameters_table,
            {"swap_glm_param_name": key["swap_glm_param_name"]},
        )
    )

    expected_region_group_id = dark_light_selection[
        "region_sorted_spikes_group_id"
    ]
    if str(key.get("region_sorted_spikes_group_id", expected_region_group_id)) != (
        str(expected_region_group_id)
    ):
        raise ValueError(
            "SwapGLM must use the RegionSortedSpikesGroup selected by "
            "DarkLightGLM."
        )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {"region_sorted_spikes_group_id": expected_region_group_id},
    )
    snapshot_metadata = snapshot["metadata"]
    expected_snapshot_fields = {
        "dark_epoch": dark_light_selection["dark_epoch"],
        "light_epoch": dark_light_selection["light_epoch"],
        "region": region_row["region_name"],
    }
    for field_name, expected_value in expected_snapshot_fields.items():
        if str(snapshot_metadata.get(field_name)) != str(expected_value):
            raise ValueError(
                "DarkLightGLM artifact disagrees with its selection: "
                f"{field_name}."
            )
    if str(snapshot["parameter_sha256"]) != str(
        dark_light_selection["dark_light_glm_parameters_sha256"]
    ) or str(snapshot["output_rule_sha256"]) != str(
        dark_light_selection["dark_light_glm_output_rule_sha256"]
    ):
        raise ValueError(
            "DarkLightGLM artifact parameter/output hashes disagree with "
            "its selection."
        )

    light_test_movement_key = {
        "movement_firing_rate_id": key[
            "light_test_movement_firing_rate_id"
        ]
    }
    light_test_movement_result = _fetch1_dict(
        movement_firing_rate_table,
        light_test_movement_key,
    )
    light_test_movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        light_test_movement_key,
    )
    training_movement_selections = {
        condition_name: _fetch1_dict(
            movement_firing_rate_selection_table,
            {
                "movement_firing_rate_id": dark_light_selection[
                    f"{condition_name}_movement_firing_rate_id"
                ]
            },
        )
        for condition_name in ("dark", "light")
    }
    training_reference = training_movement_selections["dark"]
    for field_name in (
        "nwb_file_name",
        "region_sorted_spikes_group_id",
        "movement_param_name",
        "movement_parameters_sha256",
        "position_series_name",
    ):
        expected_value = training_reference.get(field_name)
        if any(
            str(selection.get(field_name)) != str(expected_value)
            for selection in (
                training_movement_selections["light"],
                light_test_movement_selection,
            )
        ):
            raise ValueError(
                "SwapGLM training and held-out movement rows must share the "
                f"same frozen source snapshot: {field_name}."
            )
    _validate_region_movement_identity(
        analysis_name="SwapGLM",
        region_row=region_row,
        movement_selection=light_test_movement_selection,
        movement_result=light_test_movement_result,
    )
    allowed_movement_statuses = (
        {"no_units"}
        if int(region_row.get("n_units", -1)) == 0
        else {"valid", "no_valid_position", "no_movement"}
    )
    if str(light_test_movement_result.get("analysis_status")) not in (
        allowed_movement_statuses
    ):
        raise ValueError(
            "SwapGLM held-out movement status is incompatible with its "
            "regional unit count."
        )

    nwb_file_name = str(training_reference["nwb_file_name"])
    epochs = {
        "dark": str(dark_light_selection["dark_epoch"]),
        "light_train": str(dark_light_selection["light_epoch"]),
        "light_test": str(light_test_movement_selection["epoch"]),
    }
    if len(set(epochs.values())) != 3:
        raise ValueError(
            "SwapGLM dark, light-train, and light-test epochs must be distinct."
        )
    for field_name, expected_value in (
        ("nwb_file_name", nwb_file_name),
        ("dark_epoch", epochs["dark"]),
        ("light_train_epoch", epochs["light_train"]),
        ("light_test_epoch", epochs["light_test"]),
    ):
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                f"SwapGLM {field_name} does not match its upstream row."
            )

    epoch_rows = {
        condition_name: _fetch1_dict(
            epoch_intervals_table,
            {"nwb_file_name": nwb_file_name, "epoch": epoch},
        )
        for condition_name, epoch in epochs.items()
    }
    if any(str(row.get("epoch_type")) != "run" for row in epoch_rows.values()):
        raise ValueError("SwapGLM requires three explicit run epochs.")
    dark_condition = str(epoch_rows["dark"].get("condition"))
    dark_is_light = epoch_rows["dark"].get("is_light")
    if dark_condition != "dark" or dark_is_light is None or bool(dark_is_light):
        raise ValueError(
            "SwapGLM dark epoch must have condition='dark' and is_light=False."
        )
    light_conditions = {
        condition_name: str(epoch_rows[condition_name].get("condition"))
        for condition_name in ("light_train", "light_test")
    }
    allowed_light_conditions = {"AB", "BA", "gray", "bright"}
    if any(
        condition not in allowed_light_conditions
        or epoch_rows[condition_name].get("is_light") is None
        or not bool(epoch_rows[condition_name].get("is_light"))
        for condition_name, condition in light_conditions.items()
    ):
        raise ValueError(
            "SwapGLM train and test light epochs require explicit light "
            "conditions and is_light=True."
        )
    if len(set(light_conditions.values())) != 2:
        raise ValueError(
            "SwapGLM light-train and light-test conditions must differ."
        )
    expected_conditions = {
        "dark_condition": dark_condition,
        "light_train_condition": light_conditions["light_train"],
        "light_test_condition": light_conditions["light_test"],
    }
    for field_name, expected_value in expected_conditions.items():
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                f"SwapGLM {field_name} does not match EpochIntervals."
            )

    source_fields: dict[str, Any] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"light_test_{trajectory_type}_trajectory_type"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                f"SwapGLM {trajectory_field} must equal {trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epochs["light_test"],
                "trajectory_type": trajectory_type,
            },
        )
        if int(trajectory_row.get("interval_count", -1)) < 1:
            raise ValueError(
                "SwapGLM requires at least one held-out lap for every "
                f"trajectory; {trajectory_type!r} has "
                f"{trajectory_row.get('interval_count')!r}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "SwapGLM graph aliases must match their trajectory types."
            )
        graph_row = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if str(graph_row.get("coordinate_unit")) != "cm":
            raise ValueError("SwapGLM graphs must use centimeters.")
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    output_rule_sha256 = provenance_sha256(
        dict(table_specs.SWAP_GLM_OUTPUT_RULE)
    )
    if output_rule_sha256 != OUTPUT_RULE_SHA256:
        raise ValueError(
            "SwapGLM table and artifact output rules have diverged."
        )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="swap_glm_parameters_sha256",
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "dark_light_glm_id": key["dark_light_glm_id"],
        "region_sorted_spikes_group_id": expected_region_group_id,
        "light_test_movement_firing_rate_id": key[
            "light_test_movement_firing_rate_id"
        ],
        "dark_epoch": epochs["dark"],
        "light_train_epoch": epochs["light_train"],
        "light_test_epoch": epochs["light_test"],
        **source_fields,
        "swap_glm_param_name": parameters["swap_glm_param_name"],
        **expected_conditions,
        "dark_light_glm_sha256": snapshot["dark_light_glm_sha256"],
        "dark_light_selected_model_sha256_by_model": snapshot[
            "selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": snapshot["parameter_sha256"],
        "dark_light_output_rule_sha256": snapshot["output_rule_sha256"],
        "upstream_analysis_status": snapshot["analysis_status"],
        **parameter_snapshot,
        "swap_glm_output_rule_sha256": output_rule_sha256,
    }
    return {
        "swap_glm_id": selection_uuid("SwapGLM", natural_key),
        **natural_key,
    }


def _load_swap_tuning_curve_snapshot(
    *,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    tuning_curve_id: Any,
) -> dict[str, Any]:
    """Load, validate, and hash one upstream all-trial tuning curve."""
    from v1ca1.spyglass.path_specific_place import (
        path_specific_place_tuning_curve_sha256,
    )

    curve_key = {"path_specific_place_tuning_curve_id": tuning_curve_id}
    result_row = _fetch1_dict(tuning_curve_table, curve_key)
    selection_row = _fetch1_dict(tuning_curve_selection_table, curve_key)
    curve = _load_path_specific_place_tuning_curve_result(
        result_row=result_row,
        tuning_curve_table=tuning_curve_table,
        selection_row=selection_row,
    )
    return {
        "selection": selection_row,
        "result": result_row,
        "curve": curve,
        "artifact_sha256": path_specific_place_tuning_curve_sha256(curve),
    }


def _swap_tuning_curve_comparison_selection_row(
    *,
    key: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    tuning_curve_parameters_table: Any,
    epoch_intervals_table: Any,
    parameters_table: Any,
    curve_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    movement_artifact_sha256_by_role: Mapping[
        str, Mapping[str, str]
    ] | None = None,
) -> dict[str, Any]:
    """Validate and identify one immutable empirical swap-tuning bundle."""
    from v1ca1.spyglass.selection import provenance_sha256, selection_uuid
    from v1ca1.spyglass.swap_tuning import OUTPUT_RULE_SHA256

    parameters = _validate_swap_tuning_curve_comparison_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "swap_tuning_curve_comparison_param_name": key[
                    "swap_tuning_curve_comparison_param_name"
                ]
            },
        )
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    nwb_file_name = str(region_row["nwb_file_name"])
    region = str(region_row["region_name"])
    selected_units_sha256 = str(region_row["selected_units_sha256"])
    n_units = int(region_row["n_units"])

    movement_ids = {
        epoch_role: key[f"{epoch_role}_movement_firing_rate_id"]
        for epoch_role in _SWAP_TUNING_EPOCH_ROLES
    }
    movement_results: dict[str, dict[str, Any]] = {}
    movement_selections: dict[str, dict[str, Any]] = {}
    for epoch_role, movement_id in movement_ids.items():
        movement_key = {"movement_firing_rate_id": movement_id}
        movement_results[epoch_role] = _fetch1_dict(
            movement_firing_rate_table,
            movement_key,
        )
        movement_selections[epoch_role] = _fetch1_dict(
            movement_firing_rate_selection_table,
            movement_key,
        )
    supplied_movement_hashes = dict(movement_artifact_sha256_by_role or {})
    movement_firing_rate_sha256: dict[str, str] = {}
    movement_intervals_sha256: dict[str, str] = {}
    for epoch_role, movement_result in movement_results.items():
        if epoch_role in supplied_movement_hashes:
            artifact_hashes = dict(supplied_movement_hashes[epoch_role])
            movement_firing_rate_sha256[epoch_role] = str(
                artifact_hashes["firing_rate"]
            )
            movement_intervals_sha256[epoch_role] = str(
                artifact_hashes["movement_intervals"]
            )
        else:
            movement_firing_rate_sha256[epoch_role] = str(
                movement_result["movement_firing_rate_sha256"]
            )
            movement_intervals_sha256[epoch_role] = str(
                movement_result["movement_intervals_sha256"]
            )

    movement_reference = movement_selections["dark"]
    shared_movement_fields = (
        "nwb_file_name",
        "region_sorted_spikes_group_id",
        "movement_param_name",
        "movement_parameters_sha256",
        "position_series_name",
    )
    for epoch_role, movement_selection in movement_selections.items():
        for field_name in shared_movement_fields:
            if str(movement_selection.get(field_name)) != str(
                movement_reference.get(field_name)
            ):
                raise ValueError(
                    "SwapTuningCurveComparison movement rows must share the "
                    f"same frozen source definition: {field_name}."
                )
        expected_epoch = str(movement_selection["epoch"])
        supplied_epoch = str(key.get(f"{epoch_role}_epoch", expected_epoch))
        if supplied_epoch != expected_epoch:
            raise ValueError(
                f"SwapTuningCurveComparison {epoch_role}_epoch does not "
                "match its MovementFiringRate row."
            )
    for epoch_role, movement_result in movement_results.items():
        _validate_region_movement_identity(
            analysis_name="SwapTuningCurveComparison",
            region_row=region_row,
            movement_selection=movement_selections[epoch_role],
            movement_result=movement_result,
        )
        allowed_statuses = (
            {"no_units"}
            if n_units == 0
            else {"valid", "no_valid_position", "no_movement"}
        )
        if str(movement_result.get("analysis_status")) not in allowed_statuses:
            raise ValueError(
                "SwapTuningCurveComparison movement status is incompatible "
                f"with its regional unit count: {epoch_role}."
            )

    selected_movement_parameters: dict[str, dict[str, Any]] = {}
    position_rows: dict[str, dict[str, Any]] = {}
    for epoch_role, movement_selection in movement_selections.items():
        selected_movement_parameters[epoch_role] = (
            _validate_movement_parameter_row(
                _fetch1_dict(movement_parameters_table, movement_selection)
            )
        )
        _validate_frozen_parameters(
            movement_selection,
            selected_movement_parameters[epoch_role],
            field_name="movement_parameters_sha256",
        )
        position_rows[epoch_role] = _fetch1_dict(
            position_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": movement_selection["epoch"],
                "position_series_name": movement_selection[
                    "position_series_name"
                ],
            },
        )
        if str(position_rows[epoch_role].get("spatial_unit")) != "cm":
            raise ValueError(
                "SwapTuningCurveComparison positions must use centimeters."
            )
    movement_parameter_reference = selected_movement_parameters["dark"]
    if any(
        parameter_row != movement_parameter_reference
        for parameter_row in selected_movement_parameters.values()
    ):
        raise ValueError(
            "SwapTuningCurveComparison movement rows must share exact "
            "movement parameter values."
        )
    position_offsets = {
        int(row["analysis_start_offset_samples"])
        for row in position_rows.values()
    }
    if len(position_offsets) != 1:
        raise ValueError(
            "SwapTuningCurveComparison positions must share one analysis "
            "start offset."
        )
    position_offset_samples = position_offsets.pop()

    epochs = {
        epoch_role: str(movement_selections[epoch_role]["epoch"])
        for epoch_role in _SWAP_TUNING_EPOCH_ROLES
    }
    if len(set(epochs.values())) != 3:
        raise ValueError(
            "SwapTuningCurveComparison requires three distinct epochs."
        )
    epoch_rows = {
        epoch_role: _fetch1_dict(
            epoch_intervals_table,
            {"nwb_file_name": nwb_file_name, "epoch": epoch},
        )
        for epoch_role, epoch in epochs.items()
    }
    if any(str(row.get("epoch_type")) != "run" for row in epoch_rows.values()):
        raise ValueError(
            "SwapTuningCurveComparison requires three explicit run epochs."
        )
    dark_is_light = epoch_rows["dark"].get("is_light")
    if (
        str(epoch_rows["dark"].get("condition")) != "dark"
        or dark_is_light is None
        or bool(dark_is_light)
    ):
        raise ValueError(
            "SwapTuningCurveComparison dark epoch must have condition='dark' "
            "and is_light=False."
        )
    allowed_light_conditions = {"AB", "BA", "gray", "bright"}
    light_conditions = {
        epoch_role: str(epoch_rows[epoch_role].get("condition"))
        for epoch_role in ("light_train", "light_test")
    }
    if any(
        condition not in allowed_light_conditions
        or not bool(epoch_rows[epoch_role].get("is_light"))
        for epoch_role, condition in light_conditions.items()
    ):
        raise ValueError(
            "SwapTuningCurveComparison light epochs require explicit light "
            "conditions and is_light=True."
        )
    if len(set(light_conditions.values())) != 2:
        raise ValueError(
            "SwapTuningCurveComparison light-train and light-test conditions "
            "must differ."
        )
    conditions = {
        "dark_condition": "dark",
        "light_train_condition": light_conditions["light_train"],
        "light_test_condition": light_conditions["light_test"],
    }
    for field_name, expected_value in conditions.items():
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                f"SwapTuningCurveComparison {field_name} does not match "
                "EpochIntervals."
            )

    curve_ids: dict[str, Any] = {}
    curve_sha256: dict[str, dict[str, str]] = {}
    tuning_parameter_sha256: dict[str, dict[str, str]] = {}
    common_tuning_parameter_name: str | None = None
    common_tuning_parameter_sha256: str | None = None
    supplied_snapshots = dict(curve_snapshots or {})
    for epoch_role in _SWAP_TUNING_EPOCH_ROLES:
        curve_sha256[epoch_role] = {}
        tuning_parameter_sha256[epoch_role] = {}
        for trajectory_type in _DPP_TRAJECTORY_TYPES:
            source_name = f"{epoch_role}:{trajectory_type}"
            field_name = f"{epoch_role}_{trajectory_type}_tuning_curve_id"
            curve_id = key[field_name]
            snapshot = dict(
                supplied_snapshots[source_name]
                if source_name in supplied_snapshots
                else _load_swap_tuning_curve_snapshot(
                    tuning_curve_table=tuning_curve_table,
                    tuning_curve_selection_table=(
                        tuning_curve_selection_table
                    ),
                    tuning_curve_id=curve_id,
                )
            )
            curve_selection = dict(snapshot["selection"])
            curve_result = dict(snapshot["result"])
            expected_curve_fields = {
                "nwb_file_name": nwb_file_name,
                "epoch": epochs[epoch_role],
                "trajectory_type": trajectory_type,
                "configuration_name": trajectory_type,
                "movement_firing_rate_id": movement_ids[epoch_role],
                "trial_subset": "all",
            }
            for curve_field, expected_value in expected_curve_fields.items():
                if str(curve_selection.get(curve_field)) != str(expected_value):
                    raise ValueError(
                        "SwapTuningCurveComparison requires matching all-trial "
                        f"source curves; {source_name} has stale {curve_field}."
                    )
            if str(curve_result.get("selected_units_sha256")) != (
                selected_units_sha256
            ) or int(curve_result.get("n_units", -1)) != n_units:
                raise ValueError(
                    "SwapTuningCurveComparison source curves and regional "
                    f"spikes must contain identical units: {source_name}."
                )
            tuning_parameters = _validate_tuning_curve_parameter_row(
                _fetch1_dict(tuning_curve_parameters_table, curve_selection)
            )
            _validate_frozen_parameters(
                curve_selection,
                tuning_parameters,
                field_name="tuning_curve_parameters_sha256",
            )
            if (
                tuning_parameters["binning_mode"] != "bin_size_cm"
                or not math.isclose(
                    float(tuning_parameters["place_bin_size_cm"]),
                    4.0,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                or float(
                    tuning_parameters["gaussian_smoothing_sigma_bins"]
                )
                != 0.0
            ):
                raise ValueError(
                    "SwapTuningCurveComparison requires all-trial, unsmoothed "
                    "4-cm PathSpecificPlaceTuningCurve inputs."
                )
            parameter_name = str(
                curve_selection["tuning_curve_param_name"]
            )
            parameter_sha256 = str(
                curve_selection["tuning_curve_parameters_sha256"]
            )
            if common_tuning_parameter_name is None:
                common_tuning_parameter_name = parameter_name
                common_tuning_parameter_sha256 = parameter_sha256
            elif (
                parameter_name != common_tuning_parameter_name
                or parameter_sha256 != common_tuning_parameter_sha256
            ):
                raise ValueError(
                    "SwapTuningCurveComparison source curves must share one "
                    "frozen tuning-curve parameter definition."
                )
            curve_ids[field_name] = curve_id
            curve_sha256[epoch_role][trajectory_type] = str(
                snapshot["artifact_sha256"]
            )
            tuning_parameter_sha256[epoch_role][trajectory_type] = (
                parameter_sha256
            )

    output_rule_sha256 = provenance_sha256(
        dict(table_specs.SWAP_TUNING_CURVE_COMPARISON_OUTPUT_RULE)
    )
    if output_rule_sha256 != OUTPUT_RULE_SHA256:
        raise ValueError(
            "SwapTuningCurveComparison table and artifact output rules have "
            "diverged."
        )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="swap_tuning_curve_comparison_parameters_sha256",
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        **{
            f"{epoch_role}_movement_firing_rate_id": movement_id
            for epoch_role, movement_id in movement_ids.items()
        },
        **curve_ids,
        **{f"{epoch_role}_epoch": epoch for epoch_role, epoch in epochs.items()},
        "swap_tuning_curve_comparison_param_name": parameters[
            "swap_tuning_curve_comparison_param_name"
        ],
        **conditions,
        "selected_units_sha256": selected_units_sha256,
        "position_offset_samples": position_offset_samples,
        "speed_threshold_cm_s": float(
            movement_parameter_reference["speed_threshold_cm_s"]
        ),
        "source_tuning_curve_sha256_by_role_trajectory": curve_sha256,
        "source_tuning_parameters_sha256_by_role_trajectory": (
            tuning_parameter_sha256
        ),
        "movement_firing_rate_table_sha256_by_role": (
            movement_firing_rate_sha256
        ),
        "movement_intervals_sha256_by_role": movement_intervals_sha256,
        **parameter_snapshot,
        "swap_tuning_curve_comparison_output_rule_sha256": (
            output_rule_sha256
        ),
    }
    return {
        "swap_tuning_curve_comparison_id": selection_uuid(
            "SwapTuningCurveComparison",
            natural_key,
        ),
        **natural_key,
    }


def _tuning_curve_artifact_attributes(
    selection: Mapping[str, Any],
    *,
    selected_units_sha256: str,
) -> dict[str, str]:
    """Return storage-safe identity linking one curve to its selection row."""
    field_names = (
        "path_specific_place_tuning_curve_id",
        "nwb_file_name",
        "epoch",
        "trajectory_type",
        "configuration_name",
        "movement_firing_rate_id",
        "tuning_curve_param_name",
        "trial_subset",
        "tuning_curve_parameters_sha256",
    )
    return {
        **{field_name: str(selection[field_name]) for field_name in field_names},
        "selected_units_sha256": str(selected_units_sha256),
    }


def _validate_tuning_curve_artifact_link(
    *,
    curve: Any,
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
) -> None:
    """Require one canonical curve to match its DataJoint selection and row."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    expected_attributes = _tuning_curve_artifact_attributes(
        selection_row,
        selected_units_sha256=str(result_row["selected_units_sha256"]),
    )
    for field_name, expected_value in expected_attributes.items():
        if str(curve.attrs.get(field_name, "")) != str(expected_value):
            raise ValueError(
                "PathSpecificPlaceTuningCurve artifact does not match its "
                f"DataJoint row: {field_name}."
            )
    expected_scalars = {
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[curve.dims[1]]),
        "analysis_status": str(curve.attrs["analysis_status"]),
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row[field_name]) != str(expected_value):
            raise ValueError(
                "PathSpecificPlaceTuningCurve result metadata disagrees with "
                f"its artifact: {field_name}."
            )
    if not math.isclose(
        float(result_row["support_duration_s"]),
        float(curve.attrs["support_duration_s"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "PathSpecificPlaceTuningCurve result metadata disagrees with its "
            "artifact: support_duration_s."
        )
    curve_unit_ids = [
        {
            "spikesorting_merge_id": merge_id,
            "unit_id": unit_id,
        }
        for merge_id, unit_id in zip(
            np.asarray(curve.coords["spikesorting_merge_id"].values).astype(str),
            np.asarray(curve.coords["unit_id"].values).astype(str),
            strict=True,
        )
    ]
    if unit_identity_sha256(curve_unit_ids) != str(
        result_row["selected_units_sha256"]
    ):
        raise ValueError(
            "PathSpecificPlaceTuningCurve unit identities disagree with its "
            "selected-unit digest."
        )


def _dpp_tuning_curve_artifact_attributes(
    selection: Mapping[str, Any],
    *,
    selected_units_sha256: str,
) -> dict[str, str]:
    """Return storage-safe identity linking one DPP curve to its selection."""
    field_names = (
        "dpp_tuning_curve_id",
        "nwb_file_name",
        "epoch",
        "outbound_trajectory_type",
        "inbound_trajectory_type",
        "outbound_configuration_name",
        "inbound_configuration_name",
        "movement_firing_rate_id",
        "tuning_curve_param_name",
        "turn_type",
        "trial_subset",
        "tuning_curve_parameters_sha256",
    )
    return {
        **{field_name: str(selection[field_name]) for field_name in field_names},
        "selected_units_sha256": str(selected_units_sha256),
    }


def _validate_dpp_tuning_curve_artifact_link(
    *,
    curve: Any,
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
) -> None:
    """Require one canonical DPP curve to match its selection and result."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    expected_attributes = _dpp_tuning_curve_artifact_attributes(
        selection_row,
        selected_units_sha256=str(result_row["selected_units_sha256"]),
    )
    for field_name, expected_value in expected_attributes.items():
        if str(curve.attrs.get(field_name, "")) != expected_value:
            raise ValueError(
                "DPPTuningCurve artifact does not match its DataJoint row: "
                f"{field_name}."
            )
    expected_scalars = {
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "n_outbound_trials": int(curve.attrs["n_outbound_trials"]),
        "n_inbound_trials": int(curve.attrs["n_inbound_trials"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[curve.dims[1]]),
        "analysis_status": str(curve.attrs["analysis_status"]),
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row[field_name]) != str(expected_value):
            raise ValueError(
                "DPPTuningCurve result metadata disagrees with its artifact: "
                f"{field_name}."
            )
    if not math.isclose(
        float(result_row["support_duration_s"]),
        float(curve.attrs["support_duration_s"]),
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "DPPTuningCurve result metadata disagrees with its artifact: "
            "support_duration_s."
        )
    curve_unit_ids = [
        {
            "spikesorting_merge_id": merge_id,
            "unit_id": unit_id,
        }
        for merge_id, unit_id in zip(
            np.asarray(curve.coords["spikesorting_merge_id"].values).astype(str),
            np.asarray(curve.coords["unit_id"].values).astype(str),
            strict=True,
        )
    ]
    if unit_identity_sha256(curve_unit_ids) != str(
        result_row["selected_units_sha256"]
    ):
        raise ValueError(
            "DPPTuningCurve unit identities disagree with its selected-unit "
            "digest."
        )


def _sorted_spikes_group_key(key: Mapping[str, Any]) -> dict[str, Any]:
    """Return the session-constrained standard sorting-group key."""
    required = (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
    )
    missing = [name for name in required if name not in key]
    if missing:
        raise ValueError(f"RippleModulation selection is missing group keys {missing!r}.")
    return {name: key[name] for name in required}


def _load_group_unit_data(
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    region: str,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Load one group through the shared strict sorting adapter."""
    from v1ca1.spyglass.spikes import load_sorted_spikes_group

    return load_sorted_spikes_group(
        sorted_spikes_group,
        unit_selection_params,
        spike_sorting_output,
        key,
        region=region,
        allow_empty=allow_empty,
    )


def _load_registered_region_spikes(
    *,
    region_sorted_spikes_group_table: Any,
    key: Mapping[str, Any],
    time_support: Any | tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Reload one registered regional group and verify its frozen identity."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    region_key = {
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ]
    }
    region_row = _fetch1_dict(region_sorted_spikes_group_table, region_key)
    if "nwb_file_name" in key and str(region_row.get("nwb_file_name")) != str(
        key["nwb_file_name"]
    ):
        raise ValueError(
            "RegionSortedSpikesGroup and analysis selection must belong to "
            "the same NWB file."
        )
    loaded = region_sorted_spikes_group_table.load_spikes(
        region_key,
        time_support=time_support,
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    if unit_digest != str(region_row["selected_units_sha256"]):
        raise ValueError(
            "RegionSortedSpikesGroup selected units changed after registration."
        )
    if int(loaded["n_units"]) != int(region_row["n_units"]):
        raise ValueError(
            "RegionSortedSpikesGroup unit count changed after registration."
        )
    return {**loaded, "registration_row": region_row}


def _load_ripple_glm_interval_inputs(
    *,
    nwb_file_name: str,
    ripple_row: Mapping[str, Any],
    epoch_row: Mapping[str, Any],
    nwbfile_table: Any,
) -> tuple[Any, Any]:
    """Load RippleGLM ripple and epoch intervals from one NWB read-only."""
    import pynwb

    from v1ca1.spyglass.nwb import load_interval_set

    nwb_path = Path(nwbfile_table.get_abs_path(str(nwb_file_name)))
    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        ripple_intervals = load_interval_set(nwbfile, ripple_row)
        epoch_interval = load_interval_set(nwbfile, epoch_row)
    return (
        _intervals_to_frame(
            ripple_intervals,
            epoch=str(epoch_row["epoch"]),
        ),
        epoch_interval,
    )


def _validate_ripple_glm_group_snapshot(
    selection: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    role: str,
) -> None:
    """Require one current regional row to match its frozen role snapshot."""
    expected = _ripple_glm_group_snapshot(row, role=role)
    for field_name, current_value in expected.items():
        if str(selection.get(field_name)) != str(current_value):
            raise ValueError(
                f"RippleGLM {role} regional sorting snapshot changed after "
                f"selection insertion: {field_name}."
            )


def _load_ripple_glm_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Reload and verify all lightweight inputs for one RippleGLM row."""
    from v1ca1.spyglass.ripple_glm import (
        OUTPUT_RULE_SHA256,
        prepare_ripple_glm_event_selection,
    )

    selection = dict(key)
    parameters = _validate_ripple_glm_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    _validate_frozen_parameters(
        selection,
        parameters,
        field_name="ripple_glm_parameters_sha256",
    )
    if str(selection.get("ripple_glm_output_rule_sha256")) != (
        OUTPUT_RULE_SHA256
    ):
        raise ValueError(
            "RippleGLM fixed output rule changed after selection insertion. "
            "Create a new selection."
        )
    ripple_row = _fetch1_dict(ripples_table, selection)
    epoch_row = _fetch1_dict(epoch_intervals_table, selection)
    _validate_ripple_provenance(ripple_row, parameters)
    detector_zscore_threshold, speed_gated = _ripple_detector_values(
        ripple_row
    )
    try:
        selected_detector_threshold = float(
            selection["detector_zscore_threshold"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "RippleGLM selection lacks a valid detector threshold snapshot."
        ) from exc
    selected_speed_gated = _database_bool(
        selection.get("speed_gated"),
        name="RippleGLMSelection.speed_gated",
    )
    if not math.isclose(
        selected_detector_threshold,
        detector_zscore_threshold,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ) or selected_speed_gated != speed_gated:
        raise ValueError(
            "RippleGLM ripple detector values changed after selection "
            "insertion."
        )
    selection["detector_zscore_threshold"] = selected_detector_threshold
    selection["speed_gated"] = selected_speed_gated
    if int(ripple_row["ripple_count"]) != int(
        selection["source_ripple_count"]
    ):
        raise ValueError(
            "RippleGLM selected RippleIntervals row changed after selection insertion."
        )
    region_rows = {
        role: _fetch1_dict(
            region_sorted_spikes_group_table,
            {
                "region_sorted_spikes_group_id": selection[
                    f"{role}_region_sorted_spikes_group_id"
                ]
            },
        )
        for role in ("source", "target")
    }
    for role, expected_region in (("source", "ca1"), ("target", "v1")):
        row = region_rows[role]
        if str(row.get("nwb_file_name")) != str(selection["nwb_file_name"]):
            raise ValueError(
                "RippleGLM regional sorting rows must remain in the "
                "selected NWB file."
            )
        if str(row.get("region_name")) != expected_region:
            raise ValueError(
                f"RippleGLM {role} region must remain {expected_region!r}."
            )
        _validate_ripple_glm_group_snapshot(selection, row, role=role)
    ripple_table, epoch_interval = _load_ripple_glm_interval_inputs(
        nwb_file_name=str(selection["nwb_file_name"]),
        ripple_row=ripple_row,
        epoch_row=epoch_row,
        nwbfile_table=nwbfile_table,
    )
    prepared = prepare_ripple_glm_event_selection(
        epoch=str(selection["epoch"]),
        ripple_table=ripple_table,
        epoch_interval=epoch_interval,
        **_ripple_glm_parameter_kwargs(parameters),
    )
    if int(prepared["n_ripples_before_selection"]) != int(
        selection["source_ripple_count"]
    ) or int(prepared["n_ripples_after_window_bounds"]) != int(
        selection["n_selected_ripples"]
    ):
        raise ValueError(
            "RippleGLM selected ripple counts changed after selection insertion."
        )
    if str(prepared["selected_ripple_events_sha256"]) != str(
        selection["selected_ripple_events_sha256"]
    ):
        raise ValueError(
            "RippleGLM selected ripple intervals changed after selection "
            "insertion."
        )
    if _ripple_glm_source_intervals_sha256(
        ripple_table,
        epoch=str(selection["epoch"]),
    ) != str(selection["source_ripple_intervals_sha256"]):
        raise ValueError(
            "RippleGLM raw source ripple intervals changed after selection "
            "insertion."
        )
    if _ripple_glm_provenance_sha256(ripple_row) != str(
        selection["ripple_provenance_sha256"]
    ):
        raise ValueError(
            "RippleGLM ripple detector provenance changed after selection "
            "insertion."
        )
    animal_name, session_date = _session_identity(session_table, selection)
    return {
        "selection": selection,
        "parameters": parameters,
        "ripple_row": ripple_row,
        "epoch_row": epoch_row,
        "region_rows": region_rows,
        "ripple_table": ripple_table,
        "epoch_interval": epoch_interval,
        "animal_name": animal_name,
        "date": session_date,
    }


def _load_ripple_glm_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
    analysis_name: str = "RippleGLM",
) -> dict[str, dict[str, Any]]:
    """Load selected CA1 and V1 groups over one analysis epoch support."""
    epoch_row = context["epoch_row"]
    time_support = (
        float(epoch_row["start_time"]),
        float(epoch_row["stop_time"]),
    )
    if (
        not all(math.isfinite(value) for value in time_support)
        or time_support[1] <= time_support[0]
    ):
        raise ValueError(
            f"{analysis_name} EpochIntervals must contain finite "
            "start_time < stop_time."
        )
    loaded = {
        role: region_sorted_spikes_group_table.load_spikes(
            {
                "region_sorted_spikes_group_id": context["selection"][
                    f"{role}_region_sorted_spikes_group_id"
                ]
            },
            time_support=time_support,
        )
        for role in ("source", "target")
    }
    for role, group in loaded.items():
        expected_count = int(context["selection"][f"{role}_n_units"])
        expected_sha256 = str(
            context["selection"][f"{role}_selected_units_sha256"]
        )
        if int(group["n_units"]) != expected_count:
            raise ValueError(
                f"{analysis_name} {role} unit count changed after selection "
                "insertion."
            )
        from v1ca1.spyglass.selection import unit_identity_sha256

        if unit_identity_sha256(group["unit_ids"]) != expected_sha256:
            raise ValueError(
                f"{analysis_name} {role} unit identities changed after "
                "selection "
                "insertion."
            )
    return loaded


def _ripple_glm_upstream_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact selection snapshot embedded in RippleGLM bundles."""
    fields = (
        "nwb_file_name",
        "epoch",
        "source_region",
        "target_region",
        "source_ripple_count",
        "detector_zscore_threshold",
        "speed_gated",
        "source_ripple_intervals_sha256",
        "ripple_provenance_sha256",
        "n_selected_ripples",
        "selected_ripple_events_sha256",
        "source_sorting_group_members_sha256",
        "source_unit_filter_params_sha256",
        "source_selected_units_sha256",
        "source_n_units",
        "target_sorting_group_members_sha256",
        "target_unit_filter_params_sha256",
        "target_selected_units_sha256",
        "target_n_units",
        "ripple_glm_parameters_sha256",
        "ripple_glm_output_rule_sha256",
    )
    provenance = {
        "ripple_glm_id": str(selection["ripple_glm_id"]),
        "source_region_sorted_spikes_group_id": str(
            selection["source_region_sorted_spikes_group_id"]
        ),
        "target_region_sorted_spikes_group_id": str(
            selection["target_region_sorted_spikes_group_id"]
        ),
        **{field_name: selection[field_name] for field_name in fields},
    }
    provenance["detector_zscore_threshold"] = float(
        provenance["detector_zscore_threshold"]
    )
    provenance["speed_gated"] = _database_bool(
        provenance["speed_gated"],
        name="RippleGLMSelection.speed_gated",
    )
    return provenance


def _validate_ripple_glm_upstream_link(
    upstream: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Require a RippleGLM bundle to embed its exact selection snapshot."""
    expected = _ripple_glm_upstream_provenance(selection)
    if dict(upstream) != expected:
        raise ValueError(
            "RippleGLM upstream provenance does not match its immutable "
            "selection."
        )


def _make_ripple_glm_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one complete RippleGLM analysis NWB."""
    from v1ca1.spyglass.ripple_glm import (
        compute_ripple_glm,
    )

    context = _load_ripple_glm_context(
        key=key,
        parameters_table=parameters_table,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    loaded = _load_ripple_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    selection = context["selection"]
    parameters = context["parameters"]
    result = compute_ripple_glm(
        ripple_glm_id=selection["ripple_glm_id"],
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        ripple_table=context["ripple_table"],
        epoch_interval=context["epoch_interval"],
        source_spikes=loaded["source"]["ts_group"],
        source_stable_unit_ids=loaded["source"]["unit_ids"],
        target_spikes=loaded["target"]["ts_group"],
        target_stable_unit_ids=loaded["target"]["unit_ids"],
        upstream_provenance=_ripple_glm_upstream_provenance(selection),
        expected_selected_ripple_events_sha256=selection[
            "selected_ripple_events_sha256"
        ],
        parameter_name=parameters["ripple_glm_param_name"],
        parameter_sha256=selection["ripple_glm_parameters_sha256"],
        output_rule_sha256=selection["ripple_glm_output_rule_sha256"],
        **_ripple_glm_parameter_kwargs(parameters),
    )
    _validate_ripple_glm_upstream_link(
        result["upstream_provenance"], selection
    )
    if str(result["selected_ripple_events_sha256"]) != str(
        selection["selected_ripple_events_sha256"]
    ):
        raise ValueError(
            "RippleGLM computation changed its selected ripple intervals."
        )
    del artifact_root
    return _write_ripple_glm_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


_RIPPLE_GLM_NWB_OBJECT_NAMES = (
    "selected_units",
    "summary",
    "events",
    "source_features",
    "target_results",
    "provenance",
)


def _write_ripple_glm_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one complete RippleGLM analysis NWB."""
    import pynwb

    from v1ca1.spyglass.ripple_glm import (
        BUNDLE_SCHEMA_VERSION,
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        ripple_glm_nwb_hashes,
        ripple_glm_result_from_nwb_objects,
        ripple_glm_result_to_nwb_objects,
        validate_ripple_glm_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for RippleGLM output."
        )
    canonical = validate_ripple_glm_result(result)
    if str(
        canonical["dataset"].attrs.get(
            "ripple_glm_result_schema_version", ""
        )
    ) != RESULT_SCHEMA_VERSION:
        raise ValueError(
            "RippleGLM analysis NWB requires the current result schema."
        )
    expected_hashes = ripple_glm_nwb_hashes(canonical)
    objects = ripple_glm_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in _RIPPLE_GLM_NWB_OBJECT_NAMES:
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError("RippleGLM NWB object IDs must be unique.")
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "RippleGLM analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = ripple_glm_result_from_nwb_objects(
                    **{
                        name: stored_nwb.objects[
                            object_ids[f"{name}_object_id"]
                        ]
                        for name in _RIPPLE_GLM_NWB_OBJECT_NAMES
                    }
                )
                if ripple_glm_nwb_hashes(stored) != expected_hashes:
                    raise ValueError(
                        "RippleGLM NWB objects changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("RippleGLM analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        **{
            field_name: canonical[field_name]
            for field_name in (
                "n_source_units",
                "n_target_units",
                "n_source_units_in_fit",
                "n_target_units_in_fit",
                "n_valid_target_units",
                "n_ripples",
                "selected_ripple_events_sha256",
                "selected_units_sha256",
                "analysis_status",
            )
        },
        **expected_hashes,
        "legacy_artifact_provenance": canonical[
            "legacy_artifact_provenance"
        ],
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_ripple_glm_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of RippleGLM scratch objects."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading RippleGLM."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError("RippleGLM NWB fetch must resolve exactly one result.")
    record = dict(records[0])
    missing = sorted(set(_RIPPLE_GLM_NWB_OBJECT_NAMES).difference(record))
    if missing:
        raise ValueError(
            f"RippleGLM NWB fetch is missing objects {missing!r}."
        )
    return {name: record[name] for name in _RIPPLE_GLM_NWB_OBJECT_NAMES}


def _load_ripple_glm_result(
    *,
    result_row: Mapping[str, Any],
    ripple_glm_table: Any,
) -> dict[str, Any]:
    """Load a RippleGLM NWB result and verify semantic object hashes."""
    from v1ca1.spyglass.ripple_glm import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        ripple_glm_nwb_hashes,
        ripple_glm_result_from_nwb_objects,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("RippleGLM artifact schema version is unsupported.")
    objects = _fetch_ripple_glm_nwb_objects(
        ripple_glm_table,
        {"ripple_glm_id": result_row["ripple_glm_id"]},
    )
    bundle = ripple_glm_result_from_nwb_objects(**objects)
    for field_name, observed in ripple_glm_nwb_hashes(bundle).items():
        if str(result_row.get(field_name)) != str(observed):
            raise ValueError(
                "RippleGLM result metadata disagrees with its NWB object: "
                f"{field_name}."
            )
    return bundle


def _validate_ripple_glm_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one loaded RippleGLM bundle to match its DataJoint rows."""
    from v1ca1.spyglass.ripple_glm import (
        BUNDLE_SCHEMA_VERSION,
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        ripple_glm_nwb_hashes,
        validate_ripple_glm_result,
    )

    validated = validate_ripple_glm_result(bundle)
    expected_metadata = {
        "ripple_glm_id": str(selection_row["ripple_glm_id"]),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(selection_row["epoch"]),
        "source_region": "ca1",
        "target_region": "v1",
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated.get(field_name)) != expected_value:
            raise ValueError(
                "RippleGLM artifact does not match its selection: "
                f"{field_name}."
            )
    parameters = _validate_ripple_glm_parameter_row(parameters_row)
    expected_parameters = {
        "parameter_name": parameters["ripple_glm_param_name"],
        "parameter_sha256": selection_row[
            "ripple_glm_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "ripple_glm_output_rule_sha256"
        ],
        **{
            key: value
            for key, value in parameters.items()
            if key != "ripple_glm_param_name"
        },
    }
    if validated["parameters"] != expected_parameters:
        raise ValueError(
            "RippleGLM artifact parameters do not match its selection."
        )
    _validate_ripple_glm_upstream_link(
        validated["upstream_provenance"], selection_row
    )
    expected_scalars = {
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        **{
            field_name: validated[field_name]
            for field_name in (
                "n_source_units",
                "n_target_units",
                "n_source_units_in_fit",
                "n_target_units_in_fit",
                "n_valid_target_units",
                "n_ripples",
                "selected_ripple_events_sha256",
                "selected_units_sha256",
                "analysis_status",
                "artifact_origin",
            )
        },
    }
    expected_scalars.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            **ripple_glm_nwb_hashes(validated),
        }
    )
    for field_name, expected_value in expected_scalars.items():
        if str(result_row.get(field_name)) != str(expected_value):
            raise ValueError(
                "RippleGLM result metadata disagrees with its artifact: "
                f"{field_name}."
            )
    if result_row.get("legacy_artifact_provenance") != validated.get(
        "legacy_artifact_provenance"
    ):
        raise ValueError(
            "RippleGLM result-row legacy provenance differs from its artifact."
        )


def _legacy_ripple_glm_unit_identity_resolver(
    loaded_spikes: Mapping[str, Any],
    *,
    role: str,
    analysis_name: str = "RippleGLM",
) -> dict[str, dict[str, str]]:
    """Map legacy imported-sorting IDs to persistent regional identities."""
    non_imported = [
        str(member["spikesorting_merge_id"])
        for member in loaded_spikes["member_provenance"]
        if int(member["n_selected_units"]) > 0
        and str(member["merge_parent"]) != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            f"Legacy {analysis_name} {role} registration requires matching "
            "ImportedSpikeSorting units; found non-imported outputs "
            f"{non_imported!r}."
        )
    metadata_rows = list(loaded_spikes["unit_metadata"])
    if len(metadata_rows) != len(loaded_spikes["unit_ids"]):
        raise ValueError(
            f"{analysis_name} {role} metadata must contain one row per "
            "selected unit."
        )
    resolver: dict[str, dict[str, str]] = {}
    for group_unit_id, metadata in zip(
        loaded_spikes["ts_group"].keys(),
        metadata_rows,
        strict=True,
    ):
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in resolver:
            raise ValueError(
                f"Every {analysis_name} {role} unit requires a unique "
                "sorting_unit_id for legacy registration."
            )
        resolver[resolver_key] = {
            "spikesorting_merge_id": str(metadata["spikesorting_merge_id"]),
            "unit_id": str(metadata["unit_id"]),
            "stable_unit_id": (
                f"{metadata['spikesorting_merge_id']}:{metadata['unit_id']}"
            ),
            "group_unit_id": str(group_unit_id),
        }
    return resolver


def _register_existing_ripple_glm_row(
    *,
    key: Mapping[str, Any],
    source_result_path: Path,
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Strictly reconstruct one legacy NetCDF and store it in analysis NWB."""
    from v1ca1.spyglass.ripple_glm import (
        register_existing_ripple_glm_artifact,
    )

    context = _load_ripple_glm_context(
        key=key,
        parameters_table=parameters_table,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    loaded = _load_ripple_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    selection = context["selection"]
    parameters = context["parameters"]
    legacy_resolvers = {
        role: _legacy_ripple_glm_unit_identity_resolver(
            loaded[role],
            role=role,
        )
        for role in ("source", "target")
    }
    registered = register_existing_ripple_glm_artifact(
            source_result_path=Path(source_result_path),
            destination_path=None,
            ripple_glm_id=selection["ripple_glm_id"],
            animal_name=context["animal_name"],
            date=context["date"],
            epoch=str(selection["epoch"]),
            ripple_table=context["ripple_table"],
            epoch_interval=context["epoch_interval"],
            source_spikes=loaded["source"]["ts_group"],
            source_stable_unit_ids=loaded["source"]["unit_ids"],
            target_spikes=loaded["target"]["ts_group"],
            target_stable_unit_ids=loaded["target"]["unit_ids"],
            upstream_provenance=_ripple_glm_upstream_provenance(selection),
            expected_selected_ripple_events_sha256=selection[
                "selected_ripple_events_sha256"
            ],
            parameter_name=parameters["ripple_glm_param_name"],
            parameter_sha256=selection["ripple_glm_parameters_sha256"],
            output_rule_sha256=selection["ripple_glm_output_rule_sha256"],
            source_v1ca1_git_commit=source_v1ca1_git_commit,
            source_spyglass_git_commit=source_spyglass_git_commit,
            source_sorting_type="ImportedSpikeSorting",
            target_sorting_type="ImportedSpikeSorting",
            source_legacy_unit_identity_resolver=legacy_resolvers["source"],
            target_legacy_unit_identity_resolver=legacy_resolvers["target"],
            overwrite=False,
            **_ripple_glm_parameter_kwargs(parameters),
        )
    _validate_ripple_glm_upstream_link(
        registered["upstream_provenance"], selection
    )
    if str(registered["selected_ripple_events_sha256"]) != str(
        selection["selected_ripple_events_sha256"]
    ):
        raise ValueError(
            "Registered RippleGLM selected ripple intervals disagree "
            "with its selection."
        )
    del artifact_root
    return _write_ripple_glm_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _ripple_cross_region_xcorr_upstream_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact RippleCrossRegionXCorr selection snapshot for artifacts."""
    fields = (
        "nwb_file_name",
        "epoch",
        "source_region",
        "target_region",
        "source_ripple_count",
        "detector_zscore_threshold",
        "speed_gated",
        "selected_ripple_intervals_sha256",
        "ripple_provenance_sha256",
        "source_sorting_group_members_sha256",
        "source_unit_filter_params_sha256",
        "source_selected_units_sha256",
        "source_n_units",
        "target_sorting_group_members_sha256",
        "target_unit_filter_params_sha256",
        "target_selected_units_sha256",
        "target_n_units",
        "ripple_cross_region_xcorr_parameters_sha256",
        "ripple_cross_region_xcorr_output_rule_sha256",
    )
    provenance = {
        "ripple_cross_region_xcorr_id": str(selection["ripple_cross_region_xcorr_id"]),
        "source_region_sorted_spikes_group_id": str(
            selection["source_region_sorted_spikes_group_id"]
        ),
        "target_region_sorted_spikes_group_id": str(
            selection["target_region_sorted_spikes_group_id"]
        ),
        **{field_name: selection[field_name] for field_name in fields},
    }
    provenance["source_ripple_count"] = int(
        provenance["source_ripple_count"]
    )
    provenance["source_n_units"] = int(provenance["source_n_units"])
    provenance["target_n_units"] = int(provenance["target_n_units"])
    provenance["detector_zscore_threshold"] = float(
        provenance["detector_zscore_threshold"]
    )
    provenance["speed_gated"] = _database_bool(
        provenance["speed_gated"],
        name="RippleCrossRegionXCorrSelection.speed_gated",
    )
    return provenance


def _validate_ripple_cross_region_xcorr_upstream_link(
    upstream: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Require a RippleCrossRegionXCorr bundle to embed its exact selection."""
    if dict(upstream) != _ripple_cross_region_xcorr_upstream_provenance(selection):
        raise ValueError(
            "RippleCrossRegionXCorr upstream provenance does not match its "
            "immutable selection."
        )


def _load_ripple_cross_region_xcorr_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Reload and verify all lightweight RippleCrossRegionXCorr inputs."""
    from v1ca1.spyglass import ripple_cross_region_xcorr

    selection = dict(key)
    parameters = _validate_ripple_cross_region_xcorr_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    _validate_frozen_parameters(
        selection,
        parameters,
        field_name="ripple_cross_region_xcorr_parameters_sha256",
    )
    if str(selection.get("ripple_cross_region_xcorr_output_rule_sha256")) != (
        ripple_cross_region_xcorr.OUTPUT_RULE_SHA256
    ):
        raise ValueError(
            "RippleCrossRegionXCorr fixed output rule changed after selection "
            "insertion. Create a new selection."
        )
    ripple_row = _fetch1_dict(ripples_table, selection)
    epoch_row = _fetch1_dict(epoch_intervals_table, selection)
    _validate_ripple_provenance(ripple_row, parameters)
    detector_zscore_threshold, speed_gated = _ripple_detector_values(
        ripple_row
    )
    selected_detector_threshold = float(
        selection["detector_zscore_threshold"]
    )
    selected_speed_gated = _database_bool(
        selection.get("speed_gated"),
        name="RippleCrossRegionXCorrSelection.speed_gated",
    )
    if not math.isclose(
        selected_detector_threshold,
        detector_zscore_threshold,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ) or selected_speed_gated != speed_gated:
        raise ValueError(
            "RippleCrossRegionXCorr detector values changed after selection "
            "insertion."
        )
    selection["detector_zscore_threshold"] = selected_detector_threshold
    selection["speed_gated"] = selected_speed_gated
    if int(ripple_row["ripple_count"]) != int(
        selection["source_ripple_count"]
    ):
        raise ValueError(
            "RippleCrossRegionXCorr selected RippleIntervals row changed after selection "
            "insertion."
        )
    region_rows = {
        role: _fetch1_dict(
            region_sorted_spikes_group_table,
            {
                "region_sorted_spikes_group_id": selection[
                    f"{role}_region_sorted_spikes_group_id"
                ]
            },
        )
        for role in ("source", "target")
    }
    for role, expected_region in (("source", "ca1"), ("target", "v1")):
        row = region_rows[role]
        if str(row.get("nwb_file_name")) != str(selection["nwb_file_name"]):
            raise ValueError(
                "RippleCrossRegionXCorr regional groups must remain in the "
                "selected NWB file."
            )
        if str(row.get("region_name")) != expected_region:
            raise ValueError(
                f"RippleCrossRegionXCorr {role} region must remain "
                f"{expected_region!r}."
            )
        _validate_ripple_glm_group_snapshot(selection, row, role=role)
    ripple_table, _ = _load_ripple_glm_interval_inputs(
        nwb_file_name=str(selection["nwb_file_name"]),
        ripple_row=ripple_row,
        epoch_row=epoch_row,
        nwbfile_table=nwbfile_table,
    )
    event_selection = (
        ripple_cross_region_xcorr.prepare_ripple_cross_region_xcorr_event_selection(
            epoch=str(selection["epoch"]),
            ripple_table=ripple_table,
        )
    )
    normalized_ripples = event_selection["selected_ripple_table"]
    if int(event_selection["n_ripples"]) != int(
        selection["source_ripple_count"]
    ):
        raise ValueError(
            "RippleCrossRegionXCorr exact ripple count changed after selection."
        )
    if str(event_selection["selected_ripple_intervals_sha256"]) != str(
        selection["selected_ripple_intervals_sha256"]
    ):
        raise ValueError(
            "RippleCrossRegionXCorr exact ripple boundaries changed after selection."
        )
    if _ripple_glm_provenance_sha256(ripple_row) != str(
        selection["ripple_provenance_sha256"]
    ):
        raise ValueError(
            "RippleCrossRegionXCorr ripple detector provenance changed after "
            "selection."
        )
    animal_name, session_date = _session_identity(session_table, selection)
    return {
        "selection": selection,
        "parameters": parameters,
        "ripple_row": ripple_row,
        "epoch_row": epoch_row,
        "region_rows": region_rows,
        "ripple_table": normalized_ripples.assign(
            epoch=str(selection["epoch"])
        ),
        "animal_name": animal_name,
        "date": session_date,
    }


def _make_ripple_cross_region_xcorr_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one RippleCrossRegionXCorr analysis NWB."""
    from v1ca1.spyglass.ripple_cross_region_xcorr import (
        compute_ripple_cross_region_xcorr,
    )

    context = _load_ripple_cross_region_xcorr_context(
        key=key,
        parameters_table=parameters_table,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    loaded = _load_ripple_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        analysis_name="RippleCrossRegionXCorr",
    )
    selection = context["selection"]
    parameters = context["parameters"]
    result = compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=selection["ripple_cross_region_xcorr_id"],
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        ripple_table=context["ripple_table"],
        ca1_spikes=loaded["source"]["ts_group"],
        ca1_stable_unit_ids=loaded["source"]["unit_ids"],
        v1_spikes=loaded["target"]["ts_group"],
        v1_stable_unit_ids=loaded["target"]["unit_ids"],
        upstream_provenance=_ripple_cross_region_xcorr_upstream_provenance(
            selection
        ),
        expected_selected_ripple_intervals_sha256=selection[
            "selected_ripple_intervals_sha256"
        ],
        parameter_name=parameters["ripple_cross_region_xcorr_param_name"],
        parameter_sha256=selection[
            "ripple_cross_region_xcorr_parameters_sha256"
        ],
        output_rule_sha256=selection[
            "ripple_cross_region_xcorr_output_rule_sha256"
        ],
        **_ripple_cross_region_xcorr_parameter_kwargs(parameters),
    )
    _validate_ripple_cross_region_xcorr_upstream_link(
        result["upstream_provenance"], selection
    )
    if str(result["selected_ripple_intervals_sha256"]) != str(
        selection["selected_ripple_intervals_sha256"]
    ):
        raise ValueError(
            "RippleCrossRegionXCorr computation changed its exact ripple intervals."
        )
    del artifact_root
    return _write_ripple_cross_region_xcorr_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _write_ripple_cross_region_xcorr_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one cross-region xcorr analysis NWB."""
    import pynwb

    from v1ca1.spyglass.ripple_cross_region_xcorr import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        ripple_cross_region_xcorr_nwb_hashes,
        ripple_cross_region_xcorr_result_from_nwb_objects,
        ripple_cross_region_xcorr_result_to_nwb_objects,
        validate_ripple_cross_region_xcorr_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for RippleCrossRegionXCorr output."
        )
    canonical = validate_ripple_cross_region_xcorr_result(result)
    expected_hashes = ripple_cross_region_xcorr_nwb_hashes(canonical)
    objects = ripple_cross_region_xcorr_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in (
                "ca1_units",
                "v1_units",
                "pair_xcorr",
                "lag_axis",
                "provenance",
            ):
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            support = objects["ripple_support"]
            object_ids["ripple_support_object_id"] = str(support.object_id)
            _io, analysis_nwb = builder.open_nwb
            analysis_nwb.add_time_intervals(support)
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "RippleCrossRegionXCorr analysis NWB failed PyNWB "
                    f"validation: {validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = ripple_cross_region_xcorr_result_from_nwb_objects(
                    ca1_units=stored_nwb.objects[
                        object_ids["ca1_units_object_id"]
                    ],
                    v1_units=stored_nwb.objects[
                        object_ids["v1_units_object_id"]
                    ],
                    pair_xcorr=stored_nwb.objects[
                        object_ids["pair_xcorr_object_id"]
                    ],
                    lag_axis=stored_nwb.objects[
                        object_ids["lag_axis_object_id"]
                    ],
                    ripple_support=stored_nwb.objects[
                        object_ids["ripple_support_object_id"]
                    ],
                    provenance=stored_nwb.objects[
                        object_ids["provenance_object_id"]
                    ],
                )
                if ripple_cross_region_xcorr_nwb_hashes(stored) != expected_hashes:
                    raise ValueError(
                        "RippleCrossRegionXCorr NWB objects changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("RippleCrossRegionXCorr analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        **{
            field_name: canonical[field_name]
            for field_name in (
                "n_ripples",
                "ripple_duration_s",
                "n_ca1_units",
                "n_v1_units",
                "n_ca1_units_in_xcorr",
                "n_v1_units_in_xcorr",
                "n_pairs",
                "n_valid_pairs",
                "selected_ripple_intervals_sha256",
                "analysis_status",
                "artifact_origin",
                "legacy_artifact_provenance",
            )
        },
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_ripple_cross_region_xcorr_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of cross-region xcorr NWB objects."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading RippleCrossRegionXCorr."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "RippleCrossRegionXCorr NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    expected = {
        "ca1_units",
        "v1_units",
        "pair_xcorr",
        "lag_axis",
        "ripple_support",
        "provenance",
    }
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(
            "RippleCrossRegionXCorr NWB fetch is missing objects "
            f"{missing!r}."
        )
    return {name: record[name] for name in expected}


def _validate_ripple_cross_region_xcorr_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one fetched xcorr result to match its DataJoint rows."""
    from v1ca1.spyglass.ripple_cross_region_xcorr import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        ripple_cross_region_xcorr_nwb_hashes,
        validate_ripple_cross_region_xcorr_result,
    )

    validated = validate_ripple_cross_region_xcorr_result(bundle)
    expected_metadata = {
        "ripple_cross_region_xcorr_id": str(
            selection_row["ripple_cross_region_xcorr_id"]
        ),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(selection_row["epoch"]),
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated.get(field_name)) != expected_value:
            raise ValueError(
                "RippleCrossRegionXCorr artifact does not match its selection: "
                f"{field_name}."
            )
    parameters = _validate_ripple_cross_region_xcorr_parameter_row(parameters_row)
    expected_parameters = {
        "parameter_name": parameters["ripple_cross_region_xcorr_param_name"],
        "parameter_sha256": selection_row[
            "ripple_cross_region_xcorr_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "ripple_cross_region_xcorr_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "ripple_cross_region_xcorr_param_name"
        },
    }
    if validated["parameters"] != expected_parameters:
        raise ValueError(
            "RippleCrossRegionXCorr artifact parameters do not match its selection."
        )
    _validate_ripple_cross_region_xcorr_upstream_link(
        validated["upstream_provenance"], selection_row
    )
    for field_name in (
        "n_ripples",
        "n_ca1_units",
        "n_v1_units",
        "n_ca1_units_in_xcorr",
        "n_v1_units_in_xcorr",
        "n_pairs",
        "n_valid_pairs",
        "selected_ripple_intervals_sha256",
        "ca1_units_sha256",
        "v1_units_sha256",
        "summary_sha256",
        "analysis_status",
        "artifact_origin",
    ):
        if str(result_row.get(field_name)) != str(validated[field_name]):
            raise ValueError(
                "RippleCrossRegionXCorr result metadata disagrees with its "
                f"artifact: {field_name}."
            )
    if not math.isclose(
        float(result_row["ripple_duration_s"]),
        float(validated["ripple_duration_s"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "RippleCrossRegionXCorr result duration disagrees with its artifact."
        )
    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "RippleCrossRegionXCorr NWB artifact schema version is unsupported."
        )
    for field_name, observed in ripple_cross_region_xcorr_nwb_hashes(
        validated
    ).items():
        if str(result_row.get(field_name)) != observed:
            raise ValueError(
                "RippleCrossRegionXCorr result metadata disagrees with its NWB "
                f"objects: {field_name}."
            )
    if result_row.get("legacy_artifact_provenance") != validated.get(
        "legacy_artifact_provenance"
    ):
        raise ValueError(
            "RippleCrossRegionXCorr result-row legacy provenance differs from its "
            "artifact."
        )


def _load_ripple_cross_region_xcorr_result(
    *,
    result_row: Mapping[str, Any],
    result_table: Any,
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Fetch, reconstruct, and verify one cross-region xcorr result."""
    from v1ca1.spyglass.ripple_cross_region_xcorr import (
        ripple_cross_region_xcorr_result_from_nwb_objects,
    )

    objects = _fetch_ripple_cross_region_xcorr_nwb_objects(
        result_table,
        {
            "ripple_cross_region_xcorr_id": result_row[
                "ripple_cross_region_xcorr_id"
            ]
        },
    )
    result = ripple_cross_region_xcorr_result_from_nwb_objects(**objects)
    _validate_ripple_cross_region_xcorr_artifact_link(
        bundle=result,
        result_row=result_row,
        selection_row=selection_row,
        parameters_row=parameters_row,
        animal_name=animal_name,
        date=date,
    )
    return result


def _legacy_ripple_cross_region_xcorr_identity_resolver(
    loaded_spikes: Mapping[str, Any],
    *,
    role: str,
) -> Callable[[Any], list[dict[str, str]]]:
    """Build one sequence resolver for legacy imported-sorting unit IDs."""
    identity_by_sorting_id = _legacy_ripple_glm_unit_identity_resolver(
        loaded_spikes,
        role=role,
        analysis_name="RippleCrossRegionXCorr",
    )

    def resolve(legacy_unit_ids: Any) -> list[dict[str, str]]:
        resolved = []
        for legacy_unit_id in legacy_unit_ids:
            matches = [
                identity
                for sorting_unit_id, identity in identity_by_sorting_id.items()
                if str(sorting_unit_id) == str(legacy_unit_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Legacy RippleCrossRegionXCorr {role} unit "
                    f"{legacy_unit_id!r} has {len(matches)} imported-sorting "
                    "identity matches."
                )
            resolved.append(dict(matches[0]))
        return resolved

    return resolve


def _register_existing_ripple_cross_region_xcorr_row(
    *,
    key: Mapping[str, Any],
    source_ca1_unit_filter_path: Path,
    source_v1_unit_filter_path: Path,
    source_summary_path: Path,
    source_result_path: Path,
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Normalize one exact legacy xcorr set into an analysis NWB."""
    from v1ca1.spyglass.ripple_cross_region_xcorr import (
        register_existing_ripple_cross_region_xcorr_artifact,
    )

    context = _load_ripple_cross_region_xcorr_context(
        key=key,
        parameters_table=parameters_table,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    loaded = _load_ripple_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        analysis_name="RippleCrossRegionXCorr",
    )
    resolvers = {
        role: _legacy_ripple_cross_region_xcorr_identity_resolver(
            loaded[role],
            role=role,
        )
        for role in ("source", "target")
    }
    selection = context["selection"]
    parameters = context["parameters"]
    registered = register_existing_ripple_cross_region_xcorr_artifact(
        source_ca1_unit_filter_path=Path(source_ca1_unit_filter_path),
        source_v1_unit_filter_path=Path(source_v1_unit_filter_path),
        source_summary_path=Path(source_summary_path),
        source_result_path=Path(source_result_path),
        destination_path=None,
        ripple_cross_region_xcorr_id=selection[
            "ripple_cross_region_xcorr_id"
        ],
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        ripple_table=context["ripple_table"],
        ca1_spikes=loaded["source"]["ts_group"],
        ca1_stable_unit_ids=loaded["source"]["unit_ids"],
        v1_spikes=loaded["target"]["ts_group"],
        v1_stable_unit_ids=loaded["target"]["unit_ids"],
        upstream_provenance=_ripple_cross_region_xcorr_upstream_provenance(
            selection
        ),
        expected_selected_ripple_intervals_sha256=selection[
            "selected_ripple_intervals_sha256"
        ],
        ca1_legacy_identity_resolver=resolvers["source"],
        v1_legacy_identity_resolver=resolvers["target"],
        ca1_sorting_type="ImportedSpikeSorting",
        v1_sorting_type="ImportedSpikeSorting",
        parameter_name=parameters["ripple_cross_region_xcorr_param_name"],
        parameter_sha256=selection[
            "ripple_cross_region_xcorr_parameters_sha256"
        ],
        output_rule_sha256=selection[
            "ripple_cross_region_xcorr_output_rule_sha256"
        ],
        source_v1ca1_git_commit=source_v1ca1_git_commit,
        source_spyglass_git_commit=source_spyglass_git_commit,
        overwrite=False,
        **_ripple_cross_region_xcorr_parameter_kwargs(parameters),
    )
    _validate_ripple_cross_region_xcorr_upstream_link(
        registered["upstream_provenance"], selection
    )
    if str(registered["selected_ripple_intervals_sha256"]) != str(
        selection["selected_ripple_intervals_sha256"]
    ):
        raise ValueError(
            "Registered RippleCrossRegionXCorr ripple boundaries disagree "
            "with its selection."
        )
    del artifact_root
    return _write_ripple_cross_region_xcorr_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _write_ripple_modulation_nwb(
    *,
    nwb_file_name: str,
    summary: Any,
    peri_ripple_firing_rate: Any,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, validate, and register one RippleModulation analysis NWB."""
    import pynwb

    from v1ca1.spyglass.ripple_modulation import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        peri_ripple_firing_rate_from_dynamic_table,
        peri_ripple_firing_rate_sha256,
        peri_ripple_firing_rate_to_dynamic_table,
        ripple_modulation_summary_from_dynamic_table,
        ripple_modulation_summary_sha256,
        ripple_modulation_summary_to_dynamic_table,
        validate_ripple_modulation_tables,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for RippleModulation output."
        )
    summary, peri_ripple_firing_rate = validate_ripple_modulation_tables(
        summary,
        peri_ripple_firing_rate,
    )
    summary_sha256 = ripple_modulation_summary_sha256(summary)
    peri_sha256 = peri_ripple_firing_rate_sha256(
        peri_ripple_firing_rate
    )
    summary_object = ripple_modulation_summary_to_dynamic_table(summary)
    peri_object = peri_ripple_firing_rate_to_dynamic_table(
        peri_ripple_firing_rate
    )
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            summary_object_id = str(builder.add_nwb_object(summary_object))
            peri_object_id = str(builder.add_nwb_object(peri_object))
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "RippleModulation analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_summary = (
                    ripple_modulation_summary_from_dynamic_table(
                        stored_nwb.objects[summary_object_id]
                    )
                )
                stored_peri = peri_ripple_firing_rate_from_dynamic_table(
                    stored_nwb.objects[peri_object_id]
                )
                validate_ripple_modulation_tables(
                    stored_summary,
                    stored_peri,
                )
                if ripple_modulation_summary_sha256(stored_summary) != (
                    summary_sha256
                ):
                    raise ValueError(
                        "RippleModulation summary NWB object changed during write."
                    )
                if peri_ripple_firing_rate_sha256(stored_peri) != peri_sha256:
                    raise ValueError(
                        "RippleModulation peri-ripple NWB object changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("RippleModulation analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        "ripple_modulation_summary_object_id": summary_object_id,
        "peri_ripple_firing_rate_object_id": peri_object_id,
        "ripple_modulation_summary_sha256": summary_sha256,
        "peri_ripple_firing_rate_sha256": peri_sha256,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_ripple_modulation_result_nwb_objects(
    ripple_modulation_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one pair of RippleModulation NWB objects."""
    relation = ripple_modulation_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading RippleModulation."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "RippleModulation NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    required = (
        "ripple_modulation_summary",
        "peri_ripple_firing_rate",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(
            f"RippleModulation NWB fetch is missing objects {missing!r}."
        )
    return record


def _load_ripple_modulation_result(
    *,
    result_row: Mapping[str, Any],
    ripple_modulation_table: Any,
) -> dict[str, Any]:
    """Load and verify one RippleModulation analysis-NWB object pair."""
    from v1ca1.spyglass.ripple_modulation import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        peri_ripple_firing_rate_from_dynamic_table,
        peri_ripple_firing_rate_sha256,
        ripple_modulation_summary_from_dynamic_table,
        ripple_modulation_summary_sha256,
        validate_ripple_modulation_tables,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("RippleModulation artifact schema version is unsupported.")
    ripple_modulation_id = result_row["ripple_modulation_id"]
    fetched = _fetch_ripple_modulation_result_nwb_objects(
        ripple_modulation_table,
        {"ripple_modulation_id": ripple_modulation_id},
    )
    summary = ripple_modulation_summary_from_dynamic_table(
        fetched["ripple_modulation_summary"]
    )
    peri = peri_ripple_firing_rate_from_dynamic_table(
        fetched["peri_ripple_firing_rate"]
    )
    summary, peri = validate_ripple_modulation_tables(summary, peri)
    actual_hashes = {
        "ripple_modulation_summary_sha256": (
            ripple_modulation_summary_sha256(summary)
        ),
        "peri_ripple_firing_rate_sha256": (
            peri_ripple_firing_rate_sha256(peri)
        ),
    }
    for field_name, actual in actual_hashes.items():
        if str(result_row.get(field_name)) != actual:
            raise ValueError(
                "RippleModulation result metadata disagrees with its NWB "
                f"object: {field_name}."
            )

    n_units = int(result_row["n_units"])
    n_ripples = int(result_row["n_ripples"])
    n_valid_units = int(result_row["n_valid_units"])
    if summary.empty:
        if not (n_units == 0 or n_ripples == 0):
            raise ValueError(
                "Empty RippleModulation NWB tables require no units or no ripples."
            )
        observed_valid_units = 0
    else:
        observed_n_ripples = summary["n_ripples"].astype(int).unique().tolist()
        if observed_n_ripples != [n_ripples] or len(summary) != n_units:
            raise ValueError(
                "RippleModulation result counts disagree with its NWB objects."
            )
        observed_valid_units = int(summary["invalid_reason"].isna().sum())
    if observed_valid_units != n_valid_units:
        raise ValueError(
            "RippleModulation n_valid_units disagrees with its NWB summary."
        )
    expected_status = (
        "no_units"
        if n_units == 0
        else "no_ripples"
        if n_ripples == 0
        else "valid"
        if n_valid_units > 0
        else "no_valid_units"
    )
    if str(result_row["analysis_status"]) != expected_status:
        raise ValueError(
            "RippleModulation analysis_status disagrees with its result counts."
        )
    return {
        "summary": summary,
        "peri_ripple_firing_rate": peri,
        "analysis_status": expected_status,
    }


def _make_ripple_modulation_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one keyed RippleModulation result to analysis NWB."""
    import pynwb

    from v1ca1.spyglass.nwb import load_interval_set
    from v1ca1.spyglass.ripple_modulation import (
        compute_epoch_region_ripple_modulation,
        empty_ripple_modulation_result,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    parameters = _validate_parameter_row(_fetch1_dict(parameters_table, key))
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    ripple_row = _fetch1_dict(ripples_table, key)
    _validate_ripple_provenance(ripple_row, parameters)
    epoch_row = _fetch1_dict(epoch_intervals_table, key)
    animal_name, session_date = _session_identity(session_table, key)
    nwb_path = Path(nwbfile_table.get_abs_path(str(key["nwb_file_name"])))

    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if (
        not math.isfinite(epoch_start)
        or not math.isfinite(epoch_stop)
        or epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")

    with pynwb.NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True) as io:
        ripple_intervals = load_interval_set(io.read(), ripple_row)
    ripple_table = _intervals_to_frame(ripple_intervals, epoch=str(key["epoch"]))
    loaded_spikes = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=key,
        time_support=(epoch_start, epoch_stop),
    )
    region = _analysis_region(
        loaded_spikes["registration_row"]["region_name"]
    )
    if loaded_spikes["status"] == "no_units":
        result = empty_ripple_modulation_result(
            animal_name=animal_name,
            date=session_date,
            epoch=str(key["epoch"]),
            region=region,
            n_ripples=int(ripple_row["ripple_count"]),
            parameters=_parameter_kwargs(parameters),
        )
    else:
        result = compute_epoch_region_ripple_modulation(
            animal_name=animal_name,
            date=session_date,
            epoch=str(key["epoch"]),
            region=region,
            ripple_table=ripple_table,
            epoch_timestamps=[epoch_start, epoch_stop],
            region_spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            **_parameter_kwargs(parameters),
        )
    if not result["summary"].empty:
        summary = result["summary"].copy()
        missing_reason = summary["invalid_reason"].isna()
        nonfinite_response = ~np.isfinite(
            summary["response_zscore"].to_numpy(dtype=float)
        )
        summary.loc[missing_reason & nonfinite_response, "invalid_reason"] = (
            "nonfinite_response_zscore"
        )
        result = {**result, "summary": summary}
    if int(result["n_ripples"]) != int(ripple_row["ripple_count"]):
        raise ValueError(
            "RippleModulation ripple count does not match the selected "
            "RippleIntervals catalog row."
        )
    del artifact_root
    n_units = int(loaded_spikes["n_units"])
    if n_units == 0:
        analysis_status = "no_units"
        n_valid_units = 0
    elif int(result["n_ripples"]) == 0:
        analysis_status = "no_ripples"
        n_valid_units = 0
    else:
        reasons = result["summary"]["invalid_reason"]
        n_valid_units = int(reasons.isna().sum())
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    artifact_row = _write_ripple_modulation_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        summary=result["summary"],
        peri_ripple_firing_rate=result["peri_ripple_firing_rate"],
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **artifact_row,
        "n_ripples": int(result["n_ripples"]),
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_spikes["unit_ids"]),
        "legacy_artifact_provenance": None,
        "artifact_origin": "computed",
    }


def _filter_registered_table(
    table: Any,
    *,
    artifact_name: str,
    artifact_key: Mapping[str, str],
    parameters: Mapping[str, Any],
    allow_empty: bool = False,
) -> Any:
    """Select and validate one key from a legacy single- or all-region table."""
    required_key_columns = tuple(artifact_key)
    missing = [column for column in required_key_columns if column not in table.columns]
    if missing:
        raise ValueError(f"{artifact_name} parquet is missing key columns {missing!r}.")

    include = None
    for column, expected in artifact_key.items():
        matches = table[column].astype(str) == str(expected)
        include = matches if include is None else include & matches
    selected = table.loc[include].copy().reset_index(drop=True)
    if selected.empty:
        if allow_empty and table.empty:
            return selected
        raise ValueError(
            f"{artifact_name} parquet has no rows for artifact key {dict(artifact_key)!r}."
        )
    for column, expected in artifact_key.items():
        unique_values = selected[column].astype(str).unique().tolist()
        if unique_values != [str(expected)]:
            raise ValueError(
                f"{artifact_name} parquet has ambiguous {column}: {unique_values!r}."
            )

    for column in ("bin_size_s", "time_before_s", "time_after_s"):
        if column not in selected.columns:
            raise ValueError(f"{artifact_name} parquet is missing parameter column {column!r}.")
        unique_values = selected[column].dropna().astype(float).unique().tolist()
        if len(unique_values) != 1 or not math.isclose(
            unique_values[0],
            float(parameters[column]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{artifact_name} parquet {column} does not match the selected parameters."
            )
    if "n_ripples" not in selected.columns:
        raise ValueError(f"{artifact_name} parquet is missing n_ripples.")
    ripple_counts = selected["n_ripples"].dropna().astype(int).unique().tolist()
    if len(ripple_counts) != 1 or ripple_counts[0] < 0:
        raise ValueError(f"{artifact_name} parquet has ambiguous n_ripples values.")
    return selected


def _attach_registered_unit_identity(
    table: Any,
    *,
    unit_metadata: list[Mapping[str, Any]],
    artifact_name: str,
) -> Any:
    """Key a legacy artifact to stable sorting-merge and NWB unit ids."""
    import pandas as pd

    if "unit_id" not in table.columns:
        raise ValueError(f"{artifact_name} parquet is missing unit_id.")

    catalog_by_stable_id: dict[tuple[str, str], dict[str, Any]] = {}
    catalog_by_sorting_unit_id: dict[str, list[dict[str, Any]]] = {}
    for raw_metadata in unit_metadata:
        metadata = dict(raw_metadata)
        try:
            stable_id = (
                str(metadata["spikesorting_merge_id"]),
                str(metadata["unit_id"]),
            )
        except KeyError as exc:
            raise ValueError("Loaded unit metadata lacks stable unit identity.") from exc
        if stable_id in catalog_by_stable_id:
            raise ValueError(f"Duplicate loaded stable unit identity {stable_id!r}.")
        catalog_by_stable_id[stable_id] = metadata
        if metadata.get("sorting_unit_id") is not None:
            catalog_by_sorting_unit_id.setdefault(
                str(metadata["sorting_unit_id"]), []
            ).append(metadata)

    if not catalog_by_stable_id:
        if not table.empty:
            raise ValueError("Selected SortedSpikesGroup has no unit metadata.")
        output = table.copy()
        output["group_unit_id"] = output["unit_id"].to_numpy(copy=True)
        output["spikesorting_merge_id"] = pd.Series(dtype=str)
        output["unit_id"] = pd.Series(dtype=str)
        output["stable_unit_id"] = pd.Series(dtype=str)
        return output
    row_metadata: list[dict[str, Any]] = []
    has_old_explicit = {
        "spikesorting_merge_id",
        "nwb_unit_id",
    }.issubset(table.columns)
    has_new_explicit = (
        "spikesorting_merge_id" in table.columns
        and "unit_id" in table.columns
        and not has_old_explicit
    )
    if has_old_explicit or has_new_explicit:
        source_unit_column = "nwb_unit_id" if has_old_explicit else "unit_id"
        stable_pairs = zip(
            table["spikesorting_merge_id"].astype(str),
            table[source_unit_column].astype(str),
        )
        for stable_id in stable_pairs:
            if stable_id not in catalog_by_stable_id:
                raise ValueError(
                    f"{artifact_name} contains unit {stable_id!r} outside the "
                    "selected SortedSpikesGroup and region."
                )
            row_metadata.append(catalog_by_stable_id[stable_id])
    else:
        ambiguous_sorting_ids = {
            sorting_unit_id
            for sorting_unit_id, records in catalog_by_sorting_unit_id.items()
            if len(records) != 1
        }
        legacy_ids = table["unit_id"].astype(str).to_list()
        missing_legacy_ids = sorted(
            {
                unit_id
                for unit_id in legacy_ids
                if unit_id not in catalog_by_sorting_unit_id
                or unit_id in ambiguous_sorting_ids
            }
        )
        if missing_legacy_ids:
            raise ValueError(
                f"{artifact_name} legacy unit_id values cannot be mapped uniquely "
                "through the augmented NWB sorting_unit_id column: "
                f"{missing_legacy_ids!r}. Supply artifacts with explicit "
                "spikesorting_merge_id and nwb_unit_id columns."
            )
        row_metadata = [
            catalog_by_sorting_unit_id[unit_id][0] for unit_id in legacy_ids
        ]

    output = table.copy()
    if "unit_id" in output:
        output["group_unit_id"] = output["unit_id"].to_numpy(copy=True)
    output["spikesorting_merge_id"] = [
        str(metadata["spikesorting_merge_id"]) for metadata in row_metadata
    ]
    output["unit_id"] = [str(metadata["unit_id"]) for metadata in row_metadata]
    output["stable_unit_id"] = [
        f"{merge_id}:{source_unit_id}"
        for merge_id, source_unit_id in zip(
            output["spikesorting_merge_id"],
            output["unit_id"],
        )
    ]
    output = output.drop(columns=["nwb_unit_id"], errors="ignore")
    return output


def _register_existing_ripple_modulation_row(
    *,
    key: Mapping[str, Any],
    summary_path: Path,
    peri_ripple_firing_rate_path: Path,
    overwrite: bool,
    parameters_table: Any,
    ripples_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Validate legacy Parquets and write one RippleModulation analysis NWB."""
    from v1ca1.spyglass.ripple_modulation import (
        plan_register_existing,
        read_planned_artifacts,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent

    parameters = _validate_parameter_row(_fetch1_dict(parameters_table, key))
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    ripple_row = _fetch1_dict(ripples_table, key)
    _validate_ripple_provenance(ripple_row, parameters)
    animal_name, session_date = _session_identity(session_table, key)
    loaded_units = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=key,
    )
    artifact_key = {
        "animal_name": animal_name,
        "date": session_date,
        "epoch": str(key["epoch"]),
        "region": _analysis_region(
            loaded_units["registration_row"]["region_name"]
        ),
    }
    plan_kwargs = {
        **artifact_key,
        "existing_summary_path": Path(summary_path),
        "existing_peri_ripple_firing_rate_path": Path(
            peri_ripple_firing_rate_path
        ),
        "ripple_modulation_id": key["ripple_modulation_id"],
        **_parameter_kwargs(parameters),
        "heatmap_normalize": parameters["heatmap_normalize"],
    }
    if artifact_root is not None:
        plan_kwargs["artifact_root"] = artifact_root
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(
            spike_sorting_output,
            {"merge_id": merge_id},
        )
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy RippleModulation registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )
    allow_empty_artifacts = (
        loaded_units["status"] == "no_units"
        or int(ripple_row["ripple_count"]) == 0
    )
    plan = plan_register_existing(**plan_kwargs)
    selected_tables = read_planned_artifacts(
        plan,
        allow_unkeyed_same_path=overwrite,
        allow_empty=allow_empty_artifacts,
    )
    legacy_artifact_provenance = {
        "summary": {
            "source_path": str(Path(summary_path).resolve(strict=True)),
            "sha256": _file_sha256(Path(summary_path)),
        },
        "peri_ripple_firing_rate": {
            "source_path": str(
                Path(peri_ripple_firing_rate_path).resolve(strict=True)
            ),
            "sha256": _file_sha256(Path(peri_ripple_firing_rate_path)),
        },
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "source_spyglass_git_commit": source_spyglass_git_commit,
    }
    selected_summary = _filter_registered_table(
        selected_tables["summary"],
        artifact_name="summary",
        artifact_key=artifact_key,
        parameters=parameters,
        allow_empty=allow_empty_artifacts,
    )
    selected_peri = _filter_registered_table(
        selected_tables["peri_ripple_firing_rate"],
        artifact_name="peri_ripple_firing_rate",
        artifact_key=artifact_key,
        parameters=parameters,
        allow_empty=allow_empty_artifacts,
    )

    selected_summary = _attach_registered_unit_identity(
        selected_summary,
        unit_metadata=loaded_units["unit_metadata"],
        artifact_name="summary",
    )
    selected_peri = _attach_registered_unit_identity(
        selected_peri,
        unit_metadata=loaded_units["unit_metadata"],
        artifact_name="peri_ripple_firing_rate",
    )
    summary_units = set(selected_summary["stable_unit_id"].astype(str))
    peri_units = set(selected_peri["stable_unit_id"].astype(str))
    catalog_units = {
        f"{unit['spikesorting_merge_id']}:{unit['unit_id']}"
        for unit in loaded_units["unit_ids"]
    }
    no_ripples = int(ripple_row["ripple_count"]) == 0
    if no_ripples and (not selected_summary.empty or not selected_peri.empty):
        raise ValueError(
            "Zero-ripple legacy artifacts must contain canonical empty tables."
        )
    if not no_ripples and (
        len(selected_summary) != len(summary_units)
        or summary_units != peri_units
        or summary_units != catalog_units
    ):
        raise ValueError(
            "Existing summary must contain one row per selected "
            "SortedSpikesGroup unit, and both artifacts must contain exactly "
            "the same units."
        )
    if not selected_summary.empty:
        summary_n_ripples = int(selected_summary["n_ripples"].iloc[0])
        peri_n_ripples = int(selected_peri["n_ripples"].iloc[0])
    else:
        summary_n_ripples = int(ripple_row["ripple_count"])
        peri_n_ripples = summary_n_ripples
    if summary_n_ripples != peri_n_ripples:
        raise ValueError("Existing artifacts disagree on n_ripples.")
    if (
        summary_n_ripples != int(ripple_row["ripple_count"])
    ):
        raise ValueError(
            "Existing artifact n_ripples does not match the selected "
            "RippleIntervals catalog row."
        )
    reasons = selected_summary["invalid_reason"]
    n_valid_units = int(reasons.isna().sum())
    n_units = len(catalog_units)
    if n_units == 0:
        analysis_status = "no_units"
    elif summary_n_ripples == 0:
        analysis_status = "no_ripples"
    else:
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    artifact_row = _write_ripple_modulation_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        summary=selected_summary,
        peri_ripple_firing_rate=selected_peri,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **artifact_row,
        "n_ripples": summary_n_ripples,
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_units["unit_ids"]),
        "legacy_artifact_provenance": legacy_artifact_provenance,
        "artifact_origin": "registered_existing",
    }


def _load_epoch_motor_behavior_inputs(
    *,
    selection: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
) -> dict[str, Any]:
    """Load the two positions, four lap sets, and four graphs in a selection."""
    nwb_file_name = str(selection["nwb_file_name"])
    epoch = str(selection["epoch"])
    primary_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "position_series_name": selection["primary_position_series_name"],
    }
    reference_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "position_series_name": selection[
            "orientation_reference_position_series_name"
        ],
    }
    return {
        "position_inputs": {
            "primary_position": position_table.load_position(
                primary_key, apply_analysis_offset=True
            ),
            "orientation_reference_position": position_table.load_position(
                reference_key, apply_analysis_offset=True
            ),
        },
        "primary_position_row": _fetch1_dict(position_table, primary_key),
        "orientation_reference_position_row": _fetch1_dict(
            position_table, reference_key
        ),
        "trajectory_intervals": {
            trajectory_type: trajectory_intervals_table.load_intervals(
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch,
                    "trajectory_type": trajectory_type,
                }
            )
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        },
        "graph_inputs": {
            trajectory_type: wtrack_graph_table.load_graph(
                {
                    "nwb_file_name": nwb_file_name,
                    "configuration_name": trajectory_type,
                }
            )
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        },
    }


def _load_epoch_motor_behavior_context(
    *,
    key: Mapping[str, Any],
    epoch_intervals_table: Any,
    position_table: Any,
    movement_parameters_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    parameters_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Reload and verify every frozen epoch motor-behavior input."""
    from v1ca1.spyglass import epoch_motor_behavior as motor_behavior
    from v1ca1.spyglass.selection import provenance_sha256

    loaded = _load_epoch_motor_behavior_inputs(
        selection=key,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
    )
    selection = _epoch_motor_behavior_selection_row(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        position_table=position_table,
        movement_parameters_table=movement_parameters_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
        position_inputs=loaded["position_inputs"],
        trajectory_interval_sets=loaded["trajectory_intervals"],
        graph_inputs=loaded["graph_inputs"],
    )
    for field_name, expected in selection.items():
        if field_name not in key or provenance_sha256(
            _epoch_motor_json_value(key[field_name])
        ) != provenance_sha256(_epoch_motor_json_value(expected)):
            raise ValueError(
                "EpochMotorBehavior selection changed after insertion: "
                f"{field_name}."
            )
    parameters = _validate_epoch_motor_behavior_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, selection)
    )
    movement_snapshot = motor_behavior.validate_movement_parameter_snapshot(
        movement_parameters,
        movement_parameters_sha256=selection["movement_parameters_sha256"],
    )
    animal_name, session_date = _session_identity(session_table, selection)
    return {
        "selection": selection,
        "parameters": parameters,
        "movement_parameters": movement_snapshot,
        "animal_name": animal_name,
        "date": session_date,
        **loaded,
    }


def _epoch_motor_behavior_compute_kwargs(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the standalone compute arguments for one frozen selection."""
    selection = context["selection"]
    parameters = context["parameters"]
    positions = context["position_inputs"]
    return {
        "animal_name": context["animal_name"],
        "date": context["date"],
        "epoch": str(selection["epoch"]),
        "epoch_motor_behavior_id": selection["epoch_motor_behavior_id"],
        "primary_position": positions["primary_position"],
        "orientation_reference_position": positions[
            "orientation_reference_position"
        ],
        "primary_position_row": context["primary_position_row"],
        "orientation_reference_position_row": context[
            "orientation_reference_position_row"
        ],
        "trajectory_intervals_by_type": context["trajectory_intervals"],
        "graph_inputs_by_configuration": context["graph_inputs"],
        "epoch_type": "run",
        "parameter_name": parameters["epoch_motor_behavior_param_name"],
        "parameter_sha256": selection[
            "epoch_motor_behavior_parameters_sha256"
        ],
        "output_rule_sha256": selection[
            "epoch_motor_behavior_output_rule_sha256"
        ],
        "progression_bin_size_cm": parameters[
            "progression_bin_size_cm"
        ],
        "movement_parameters": context["movement_parameters"],
        "movement_parameters_sha256": selection[
            "movement_parameters_sha256"
        ],
    }


def _epoch_motor_behavior_result_row(
    result: Mapping[str, Any],
    nwb_artifact: Mapping[str, Any],
    *,
    created_artifact_paths: Sequence[str],
) -> dict[str, Any]:
    """Return one DataJoint payload from a validated standalone result."""
    return {
        **dict(nwb_artifact),
        **{
            field_name: result[field_name]
            for field_name in (
                "n_position_samples_input",
                "n_finite_position_samples",
                "n_dropped_nonfinite_samples",
                "n_movement_samples",
                "movement_duration_s",
                "n_supported_trajectories",
                "sampling_rate_hz",
                "median_sample_interval_s",
                "maximum_sample_gap_s",
                "analysis_status",
            )
        },
        "legacy_artifact_provenance": (
            dict(result["legacy_artifact_provenance"])
            if result.get("legacy_artifact_provenance")
            else None
        ),
        "_created_artifact_paths": list(created_artifact_paths),
    }


def _write_epoch_motor_behavior_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write and validate the three motor tables in one analysis NWB."""
    import pynwb

    from v1ca1.spyglass.epoch_motor_behavior import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        distribution_summary_from_dynamic_table,
        distribution_summary_sha256,
        distribution_summary_to_dynamic_table,
        progression_summary_from_dynamic_table,
        progression_summary_sha256,
        progression_summary_to_dynamic_table,
        trajectory_qc_from_dynamic_table,
        trajectory_qc_sha256,
        trajectory_qc_to_dynamic_table,
        validate_epoch_motor_behavior_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for EpochMotorBehavior output."
        )
    validated = validate_epoch_motor_behavior_result(result)
    tables = {
        "distribution_summary": validated["distribution_summary"],
        "progression_summary": validated["progression_summary"],
        "trajectory_qc": validated["trajectory_qc"],
    }
    converters = {
        "distribution_summary": distribution_summary_to_dynamic_table,
        "progression_summary": progression_summary_to_dynamic_table,
        "trajectory_qc": trajectory_qc_to_dynamic_table,
    }
    readers = {
        "distribution_summary": distribution_summary_from_dynamic_table,
        "progression_summary": progression_summary_from_dynamic_table,
        "trajectory_qc": trajectory_qc_from_dynamic_table,
    }
    hashers = {
        "distribution_summary": distribution_summary_sha256,
        "progression_summary": progression_summary_sha256,
        "trajectory_qc": trajectory_qc_sha256,
    }
    hashes = {name: hashers[name](table) for name, table in tables.items()}
    objects = {name: converters[name](table) for name, table in tables.items()}
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            object_ids = {
                name: str(builder.add_nwb_object(nwb_object))
                for name, nwb_object in objects.items()
            }
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError(
                    "EpochMotorBehavior NWB tables must have distinct object IDs."
                )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "EpochMotorBehavior analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_tables = {
                    name: readers[name](stored_nwb.objects[object_id])
                    for name, object_id in object_ids.items()
                }
                validate_epoch_motor_behavior_result(
                    {**validated, **stored_tables}
                )
                for name, table in stored_tables.items():
                    if hashers[name](table) != hashes[name]:
                        raise ValueError(
                            "EpochMotorBehavior NWB object changed during write: "
                            f"{name}."
                        )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("EpochMotorBehavior analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **{
            f"{name}_object_id": object_id
            for name, object_id in object_ids.items()
        },
        **{f"{name}_sha256": digest for name, digest in hashes.items()},
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_epoch_motor_behavior_nwb_objects(
    epoch_motor_behavior_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one set of three EpochMotorBehavior NWB tables."""
    relation = epoch_motor_behavior_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading EpochMotorBehavior."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "EpochMotorBehavior NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    required = (
        "distribution_summary",
        "progression_summary",
        "trajectory_qc",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(
            f"EpochMotorBehavior NWB fetch is missing objects {missing!r}."
        )
    return record


def _load_epoch_motor_behavior_result(
    *,
    result_row: Mapping[str, Any],
    epoch_motor_behavior_table: Any,
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    movement_parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Load and verify one analysis-NWB motor-behavior result."""
    from v1ca1.spyglass import epoch_motor_behavior as motor_behavior

    if str(result_row.get("artifact_schema_version")) != (
        motor_behavior.NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "EpochMotorBehavior artifact schema version is unsupported."
        )
    fetched = _fetch_epoch_motor_behavior_nwb_objects(
        epoch_motor_behavior_table,
        {
            "epoch_motor_behavior_id": result_row[
                "epoch_motor_behavior_id"
            ]
        },
    )
    tables = {
        "distribution_summary": (
            motor_behavior.distribution_summary_from_dynamic_table(
                fetched["distribution_summary"]
            )
        ),
        "progression_summary": (
            motor_behavior.progression_summary_from_dynamic_table(
                fetched["progression_summary"]
            )
        ),
        "trajectory_qc": motor_behavior.trajectory_qc_from_dynamic_table(
            fetched["trajectory_qc"]
        ),
    }
    parameters = _validate_epoch_motor_behavior_parameter_row(parameters_row)
    movement = motor_behavior.validate_movement_parameter_snapshot(
        _validate_movement_parameter_row(movement_parameters_row),
        movement_parameters_sha256=selection_row["movement_parameters_sha256"],
    )
    bundle = motor_behavior.validate_epoch_motor_behavior_result(
        {
            "metadata": {
                "epoch_motor_behavior_id": str(
                    selection_row["epoch_motor_behavior_id"]
                ),
                "animal_name": str(animal_name),
                "date": str(date),
                "epoch": str(selection_row["epoch"]),
                "epoch_type": "run",
                "primary_position_source": str(
                    selection_row["primary_position_series_name"]
                ),
                "primary_position_role": str(
                    selection_row["primary_position_role"]
                ),
                "orientation_reference_position_source": str(
                    selection_row[
                        "orientation_reference_position_series_name"
                    ]
                ),
                "orientation_reference_position_role": str(
                    selection_row["orientation_reference_position_role"]
                ),
                "position_offset_samples": int(
                    selection_row["position_offset_samples"]
                ),
            },
            "parameters": {
                "parameter_name": parameters[
                    "epoch_motor_behavior_param_name"
                ],
                "parameter_sha256": selection_row[
                    "epoch_motor_behavior_parameters_sha256"
                ],
                "output_rule_sha256": selection_row[
                    "epoch_motor_behavior_output_rule_sha256"
                ],
                "progression_bin_size_cm": parameters[
                    "progression_bin_size_cm"
                ],
            },
            "movement_parameters": movement,
            **tables,
            **{
                name: result_row[name]
                for name in (
                    "n_position_samples_input",
                    "n_finite_position_samples",
                    "n_dropped_nonfinite_samples",
                    "n_movement_samples",
                    "movement_duration_s",
                    "n_supported_trajectories",
                    "sampling_rate_hz",
                    "median_sample_interval_s",
                    "maximum_sample_gap_s",
                    "analysis_status",
                    "artifact_origin",
                    "legacy_artifact_provenance",
                )
            },
        }
    )
    actual_hashes = {
        "distribution_summary_sha256": (
            motor_behavior.distribution_summary_sha256(
                bundle["distribution_summary"]
            )
        ),
        "progression_summary_sha256": (
            motor_behavior.progression_summary_sha256(
                bundle["progression_summary"]
            )
        ),
        "trajectory_qc_sha256": motor_behavior.trajectory_qc_sha256(
            bundle["trajectory_qc"]
        ),
    }
    for field_name, actual in actual_hashes.items():
        if str(result_row.get(field_name)) != actual:
            raise ValueError(
                "EpochMotorBehavior result metadata disagrees with its NWB "
                f"object: {field_name}."
            )
    _validate_epoch_motor_behavior_artifact_link(
        bundle=bundle,
        result_row=result_row,
        selection_row=selection_row,
        parameters_row=parameters_row,
        movement_parameters_row=movement_parameters_row,
        animal_name=animal_name,
        date=date,
    )
    return bundle


def _make_epoch_motor_behavior_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    position_table: Any,
    movement_parameters_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one immutable motor-behavior analysis NWB."""
    from v1ca1.spyglass.epoch_motor_behavior import (
        compute_selected_epoch_motor_behavior,
    )

    context = _load_epoch_motor_behavior_context(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        position_table=position_table,
        movement_parameters_table=movement_parameters_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
        session_table=session_table,
    )
    result = compute_selected_epoch_motor_behavior(
        **_epoch_motor_behavior_compute_kwargs(context),
        artifact_origin="computed",
        legacy_artifact_provenance=None,
    )
    del artifact_root
    nwb_artifact = _write_epoch_motor_behavior_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    created_artifact_paths = list(
        nwb_artifact.pop("_created_artifact_paths", ())
    )
    return _epoch_motor_behavior_result_row(
        result,
        nwb_artifact,
        created_artifact_paths=created_artifact_paths,
    )


def _validate_epoch_motor_behavior_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    movement_parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one loaded motor bundle to match all DataJoint rows."""
    from v1ca1.spyglass import epoch_motor_behavior as motor_behavior

    validated = motor_behavior.validate_epoch_motor_behavior_result(bundle)
    metadata = validated["metadata"]
    expected_metadata = {
        "epoch_motor_behavior_id": str(
            selection_row["epoch_motor_behavior_id"]
        ),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(selection_row["epoch"]),
        "epoch_type": "run",
        "primary_position_source": str(
            selection_row["primary_position_series_name"]
        ),
        "primary_position_role": str(selection_row["primary_position_role"]),
        "orientation_reference_position_source": str(
            selection_row["orientation_reference_position_series_name"]
        ),
        "orientation_reference_position_role": str(
            selection_row["orientation_reference_position_role"]
        ),
        "position_offset_samples": int(
            selection_row["position_offset_samples"]
        ),
    }
    if metadata != expected_metadata:
        raise ValueError(
            "EpochMotorBehavior artifact metadata differs from its selection."
        )
    parameters = _validate_epoch_motor_behavior_parameter_row(parameters_row)
    expected_parameters = {
        "parameter_name": parameters["epoch_motor_behavior_param_name"],
        "parameter_sha256": selection_row[
            "epoch_motor_behavior_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "epoch_motor_behavior_output_rule_sha256"
        ],
        "progression_bin_size_cm": parameters[
            "progression_bin_size_cm"
        ],
    }
    if validated["parameters"] != expected_parameters:
        raise ValueError(
            "EpochMotorBehavior artifact parameters differ from its selection."
        )
    expected_movement = motor_behavior.validate_movement_parameter_snapshot(
        _validate_movement_parameter_row(movement_parameters_row),
        movement_parameters_sha256=selection_row["movement_parameters_sha256"],
    )
    if validated["movement_parameters"] != expected_movement:
        raise ValueError(
            "EpochMotorBehavior movement parameters differ from its selection."
        )

    for field_name in (
        "n_position_samples_input",
        "n_finite_position_samples",
        "n_dropped_nonfinite_samples",
        "n_movement_samples",
        "movement_duration_s",
        "n_supported_trajectories",
        "sampling_rate_hz",
        "median_sample_interval_s",
        "maximum_sample_gap_s",
        "analysis_status",
        "artifact_origin",
    ):
        left = result_row.get(field_name)
        right = validated[field_name]
        if isinstance(right, Real) and not isinstance(right, Integral):
            matches = (
                np.isnan(float(left)) and np.isnan(float(right))
            ) or np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12)
        else:
            matches = str(left) == str(right)
        if not matches:
            raise ValueError(
                "EpochMotorBehavior result metadata disagrees with its "
                f"artifact: {field_name}."
            )
    if str(result_row.get("artifact_schema_version")) != (
        motor_behavior.NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("EpochMotorBehavior NWB schema version disagrees.")
    if result_row.get("legacy_artifact_provenance") != validated.get(
        "legacy_artifact_provenance"
    ):
        raise ValueError("EpochMotorBehavior legacy provenance disagrees.")


def _register_existing_epoch_motor_behavior_row(
    *,
    key: Mapping[str, Any],
    source_distribution_path: Path,
    source_progression_path: Path,
    source_run_log_path: Path | None,
    parameters_table: Any,
    epoch_intervals_table: Any,
    position_table: Any,
    movement_parameters_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Verify legacy motor tables and write one analysis NWB."""
    from v1ca1.spyglass.epoch_motor_behavior import (
        register_existing_epoch_motor_behavior_artifact,
    )

    context = _load_epoch_motor_behavior_context(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        position_table=position_table,
        movement_parameters_table=movement_parameters_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
        session_table=session_table,
    )
    del artifact_root
    registered = register_existing_epoch_motor_behavior_artifact(
        source_distribution_path=Path(source_distribution_path),
        source_progression_path=Path(source_progression_path),
        source_run_log_path=(
            None
            if source_run_log_path is None
            else Path(source_run_log_path)
        ),
        destination_path=None,
        overwrite=False,
        write_artifact=False,
        **_epoch_motor_behavior_compute_kwargs(context),
    )
    nwb_artifact = _write_epoch_motor_behavior_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    created_artifact_paths = list(
        nwb_artifact.pop("_created_artifact_paths", ())
    )
    return _epoch_motor_behavior_result_row(
        registered,
        nwb_artifact,
        created_artifact_paths=created_artifact_paths,
    )


def _make_movement_firing_rate_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    position_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one epoch-wide movement result to an analysis NWB."""
    import pynwb

    from v1ca1.spyglass.movement import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        compute_selected_movement_firing_rate,
        movement_firing_rate_table_from_dynamic_table,
        movement_firing_rate_table_sha256,
        movement_firing_rate_table_to_dynamic_table,
        movement_interval_set_from_time_intervals,
        movement_interval_set_sha256,
        movement_interval_set_to_time_intervals,
        validate_movement_artifacts,
    )
    from v1ca1.spyglass.nwb import load_position
    from v1ca1.spyglass.selection import unit_identity_sha256

    parameters = _validate_movement_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="movement_parameters_sha256",
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, key)
    position_row = _fetch1_dict(position_table, key)
    if position_row.get("spatial_unit") != "cm":
        raise ValueError("MovementFiringRate position must use centimeters.")

    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")

    loaded_spikes = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=key,
        time_support=(epoch_start, epoch_stop),
    )
    region = _analysis_region(
        loaded_spikes["registration_row"]["region_name"]
    )

    position = None
    if loaded_spikes["status"] != "no_units":
        nwb_path = Path(nwbfile_table.get_abs_path(str(key["nwb_file_name"])))
        with pynwb.NWBHDF5IO(
            str(nwb_path),
            mode="r",
            load_namespaces=True,
        ) as io:
            position = load_position(
                io.read(),
                position_row,
                apply_analysis_offset=True,
            )

    animal_name, session_date = _session_identity(session_table, key)
    result = compute_selected_movement_firing_rate(
        animal_name=animal_name,
        date=session_date,
        region=region,
        epoch=str(key["epoch"]),
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=position,
        speed_threshold_cm_s=parameters["speed_threshold_cm_s"],
        speed_smoothing_sigma_s=parameters["speed_smoothing_sigma_s"],
    )
    del artifact_root
    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for MovementFiringRate output."
        )

    firing_rate_sha256 = movement_firing_rate_table_sha256(result["table"])
    intervals_sha256 = movement_interval_set_sha256(
        result["movement_intervals"]
    )
    firing_rate_object = movement_firing_rate_table_to_dynamic_table(
        result["table"]
    )
    interval_object = movement_interval_set_to_time_intervals(
        result["movement_intervals"]
    )
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    try:
        with analysis_table.build(str(key["nwb_file_name"])) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            firing_rate_object_id = str(
                builder.add_nwb_object(firing_rate_object)
            )
            interval_object_id = str(interval_object.object_id)
            _io, analysis_nwb = builder.open_nwb
            analysis_nwb.add_time_intervals(interval_object)
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "MovementFiringRate analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_table = movement_firing_rate_table_from_dynamic_table(
                    stored_nwb.objects[firing_rate_object_id]
                )
                stored_intervals = movement_interval_set_from_time_intervals(
                    stored_nwb.objects[interval_object_id]
                )
                validate_movement_artifacts(stored_table, stored_intervals)
                if (
                    movement_firing_rate_table_sha256(stored_table)
                    != firing_rate_sha256
                ):
                    raise ValueError(
                        "Movement firing-rate NWB object changed during write."
                    )
                if (
                    movement_interval_set_sha256(stored_intervals)
                    != intervals_sha256
                ):
                    raise ValueError(
                        "Movement interval NWB object changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("MovementFiringRate analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        "movement_firing_rate_object_id": firing_rate_object_id,
        "movement_intervals_object_id": interval_object_id,
        "movement_firing_rate_sha256": firing_rate_sha256,
        "movement_intervals_sha256": intervals_sha256,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "n_units_with_spikes": int(result["n_units_with_spikes"]),
        "movement_interval_count": int(result["movement_interval_count"]),
        "movement_duration_s": float(result["movement_duration_s"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": unit_identity_sha256(
            loaded_spikes["unit_ids"]
        ),
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_movement_result_nwb_objects(
    movement_firing_rate_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one pair of MovementFiringRate NWB objects."""
    relation = movement_firing_rate_table & dict(key)
    records = relation.fetch_nwb()
    if len(records) != 1:
        raise ValueError(
            "MovementFiringRate NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    missing = [
        name
        for name in ("movement_firing_rate", "movement_intervals")
        if name not in record
    ]
    if missing:
        raise ValueError(
            f"MovementFiringRate NWB fetch is missing objects {missing!r}."
        )
    return record


def _load_movement_result_artifacts(
    *,
    result_row: Mapping[str, Any],
    movement_firing_rate_table: Any,
    parameters: Mapping[str, Any] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load movement NWB objects and verify their DataJoint summary scalars."""
    from v1ca1.spyglass.movement import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        movement_firing_rate_table_from_dynamic_table,
        movement_firing_rate_table_sha256,
        movement_interval_set_from_time_intervals,
        movement_interval_set_sha256,
        movement_interval_summary,
        validate_movement_artifacts,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("MovementFiringRate artifact schema version is unsupported.")
    objects = _fetch_movement_result_nwb_objects(
        movement_firing_rate_table,
        {"movement_firing_rate_id": result_row["movement_firing_rate_id"]},
    )
    table = movement_firing_rate_table_from_dynamic_table(
        objects["movement_firing_rate"]
    )
    movement_intervals = movement_interval_set_from_time_intervals(
        objects["movement_intervals"]
    )
    validate_movement_artifacts(table, movement_intervals)
    actual_hashes = {
        "movement_firing_rate_sha256": movement_firing_rate_table_sha256(
            table
        ),
        "movement_intervals_sha256": movement_interval_set_sha256(
            movement_intervals
        ),
    }
    for field_name, actual in actual_hashes.items():
        if str(result_row.get(field_name)) != actual:
            raise ValueError(
                "MovementFiringRate result metadata disagrees with its NWB "
                f"object: {field_name}."
            )
    if not table.empty:
        for field_name, expected_value in dict(expected_metadata or {}).items():
            actual_values = table[field_name].astype(str).unique().tolist()
            if actual_values != [str(expected_value)]:
                raise ValueError(
                    "MovementFiringRate artifact does not match its selection: "
                    f"{field_name}."
                )
        if parameters is not None:
            for field_name in (
                "speed_threshold_cm_s",
                "speed_smoothing_sigma_s",
            ):
                actual_values = table[field_name].astype(float).unique().tolist()
                if len(actual_values) != 1 or not math.isclose(
                    actual_values[0],
                    float(parameters[field_name]),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "MovementFiringRate artifact does not match its selection: "
                        f"{field_name}."
                    )
    interval_count, duration = movement_interval_summary(movement_intervals)
    status = "no_units" if table.empty else str(
        table["firing_rate_status"].iloc[0]
    )
    n_units = len(table)
    n_valid_units = n_units if status == "valid" else 0
    n_units_with_spikes = (
        0
        if table.empty
        else int((table["movement_spike_count"].astype(int) > 0).sum())
    )
    expected = {
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "n_units_with_spikes": n_units_with_spikes,
        "movement_interval_count": interval_count,
        "analysis_status": status,
    }
    for field_name, current in expected.items():
        stored = result_row.get(field_name)
        if str(stored) != str(current):
            raise ValueError(
                "MovementFiringRate result metadata disagrees with its "
                f"artifacts: {field_name}."
            )
    if not math.isclose(
        float(result_row["movement_duration_s"]),
        duration,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "MovementFiringRate result metadata disagrees with its artifacts: "
            "movement_duration_s."
        )
    return {
        "table": table,
        "movement_intervals": movement_intervals,
        "analysis_status": status,
    }


def _load_cv_pca_inputs(
    *,
    selection: Mapping[str, Any],
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load exact untrimmed positions, movement artifacts, laps, and graphs."""
    nwb_file_name = str(selection["nwb_file_name"])
    animal_name, session_date = _session_identity(session_table, selection)
    movement_results = {}
    movement_selections = {}
    movement_parameters = {}
    movement_artifacts = {}
    positions = {}
    for condition in ("light", "dark"):
        movement_key = {
            "movement_firing_rate_id": selection[
                f"{condition}_movement_firing_rate_id"
            ]
        }
        movement_results[condition] = _fetch1_dict(
            movement_firing_rate_table, movement_key
        )
        movement_selections[condition] = _fetch1_dict(
            movement_firing_rate_selection_table, movement_key
        )
        movement_parameters[condition] = _validate_movement_parameter_row(
            _fetch1_dict(
                movement_parameters_table, movement_selections[condition]
            )
        )
        movement_artifacts[condition] = _load_movement_result_artifacts(
            result_row=movement_results[condition],
            movement_firing_rate_table=movement_firing_rate_table,
            parameters=movement_parameters[condition],
            expected_metadata={
                "animal_name": animal_name,
                "date": session_date,
                "region": str(selection["_resolved_region"]),
                "epoch": selection[f"{condition}_epoch"],
            },
        )
        positions[condition] = position_table.load_position(
            {
                "nwb_file_name": nwb_file_name,
                "epoch": selection[f"{condition}_epoch"],
                "position_series_name": selection[
                    f"{condition}_position_series_name"
                ],
            },
            apply_analysis_offset=False,
        )
    trajectory_intervals = {
        condition: {
            trajectory_type: trajectory_intervals_table.load_intervals(
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": selection[f"{condition}_epoch"],
                    "trajectory_type": trajectory_type,
                }
            )
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        }
        for condition in ("light", "dark")
    }
    graphs = {
        trajectory_type: wtrack_graph_table.load_graph(
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            }
        )
        for trajectory_type in ("center_to_left", "center_to_right")
    }
    return {
        "movement_results": movement_results,
        "movement_selections": movement_selections,
        "movement_parameters": movement_parameters,
        "movement_artifacts": movement_artifacts,
        "positions": positions,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graphs,
    }


def _cv_pca_upstream_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the immutable selection digests stored inside a cvPCA bundle."""
    scalar_fields = (
        "nwb_file_name",
        "light_epoch",
        "dark_epoch",
        "region_sorted_spikes_group_id",
        "light_movement_firing_rate_id",
        "dark_movement_firing_rate_id",
        "light_epoch_row_sha256",
        "dark_epoch_row_sha256",
        "light_epoch_bounds_sha256",
        "dark_epoch_bounds_sha256",
        "light_position_row_sha256",
        "dark_position_row_sha256",
        "light_position_values_sha256",
        "dark_position_values_sha256",
        "light_position_timestamps_sha256",
        "dark_position_timestamps_sha256",
        "region_group_row_sha256",
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
        "selected_units_sha256",
        "movement_parameters_sha256",
        "light_movement_selection_row_sha256",
        "dark_movement_selection_row_sha256",
        "light_movement_result_row_sha256",
        "dark_movement_result_row_sha256",
        "light_movement_rates_sha256",
        "dark_movement_rates_sha256",
        "light_movement_support_sha256",
        "dark_movement_support_sha256",
        "cv_pca_parameters_sha256",
        "cv_pca_effective_parameters_sha256",
        "cv_pca_output_rule_sha256",
    )
    nested_fields = (
        "trajectory_rows_sha256_by_epoch_and_type",
        "trajectory_intervals_sha256_by_epoch_and_type",
        "graph_rows_sha256_by_trajectory",
        "graph_inputs_sha256_by_trajectory",
    )
    return {
        **{field: str(selection[field]) for field in scalar_fields},
        **{
            field: _epoch_motor_json_value(selection[field])
            for field in nested_fields
        },
        "position_offset_samples": int(selection["position_offset_samples"]),
        "n_input_units": int(selection["n_input_units"]),
        "light_movement_analysis_status": str(
            selection["light_movement_analysis_status"]
        ),
        "dark_movement_analysis_status": str(
            selection["dark_movement_analysis_status"]
        ),
    }


def _load_cv_pca_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Reload and verify every frozen cvPCA source before computation."""
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256

    group_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {"region_sorted_spikes_group_id": key["region_sorted_spikes_group_id"]},
    )
    region = _analysis_region(group_row["region_name"])
    load_key = {**dict(key), "_resolved_region": region}
    loaded = _load_cv_pca_inputs(
        selection=load_key,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=movement_firing_rate_selection_table,
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    selection = _cv_pca_selection_row(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=movement_firing_rate_selection_table,
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
        session_table=session_table,
        position_inputs_by_condition=loaded["positions"],
        movement_artifacts_by_condition=loaded["movement_artifacts"],
        trajectory_interval_sets_by_condition=loaded["trajectory_intervals"],
        graph_inputs=loaded["graph_inputs"],
    )
    for field_name, expected in selection.items():
        if field_name not in key or provenance_sha256(
            _epoch_motor_json_value(key[field_name])
        ) != provenance_sha256(_epoch_motor_json_value(expected)):
            raise ValueError(
                f"CVPCA selection changed after insertion: {field_name}."
            )
    parameters = _validate_cv_pca_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    _validate_frozen_parameters(
        selection,
        parameters,
        field_name="cv_pca_parameters_sha256",
    )
    epoch_start = min(
        float(
            _fetch1_dict(
                epoch_intervals_table,
                {
                    "nwb_file_name": selection["nwb_file_name"],
                    "epoch": selection[f"{condition}_epoch"],
                },
            )["start_time"]
        )
        for condition in ("light", "dark")
    )
    epoch_stop = max(
        float(
            _fetch1_dict(
                epoch_intervals_table,
                {
                    "nwb_file_name": selection["nwb_file_name"],
                    "epoch": selection[f"{condition}_epoch"],
                },
            )["stop_time"]
        )
        for condition in ("light", "dark")
    )
    loaded_spikes = region_sorted_spikes_group_table.load_spikes(
        {"region_sorted_spikes_group_id": selection["region_sorted_spikes_group_id"]},
        time_support=(epoch_start, epoch_stop),
    )
    if unit_identity_sha256(loaded_spikes["unit_ids"]) != str(
        selection["selected_units_sha256"]
    ):
        raise ValueError("CVPCA regional group units changed after selection.")
    spike_identity = [
        (str(row["spikesorting_merge_id"]), str(row["unit_id"]))
        for row in loaded_spikes["unit_ids"]
    ]
    for condition in ("light", "dark"):
        movement_identity = [
            (str(row["spikesorting_merge_id"]), str(row["unit_id"]))
            for row in loaded["movement_artifacts"][condition]["table"].to_dict(
                "records"
            )
        ]
        if movement_identity != spike_identity:
            raise ValueError(
                "CVPCA spike and MovementFiringRate unit order must match."
            )
    animal_name, session_date = _session_identity(session_table, selection)
    return {
        "selection": selection,
        "parameters": parameters,
        "group_row": group_row,
        "region": region,
        "loaded_spikes": loaded_spikes,
        "animal_name": animal_name,
        "date": session_date,
        **loaded,
    }


def _cv_pca_compute_inputs(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return standalone cvPCA arguments from one verified table context."""
    selection = context["selection"]
    parameters = context["parameters"]
    return {
        "cv_pca_id": selection["cv_pca_id"],
        "animal_name": context["animal_name"],
        "date": context["date"],
        "region": context["region"],
        "light_epoch": selection["light_epoch"],
        "dark_epoch": selection["dark_epoch"],
        "spikes": context["loaded_spikes"]["ts_group"],
        "stable_unit_ids": context["loaded_spikes"]["unit_ids"],
        "light_position": context["positions"]["light"],
        "dark_position": context["positions"]["dark"],
        "light_movement_intervals": context["movement_artifacts"]["light"][
            "movement_intervals"
        ],
        "dark_movement_intervals": context["movement_artifacts"]["dark"][
            "movement_intervals"
        ],
        "light_movement_firing_rate_hz": context["movement_artifacts"]["light"][
            "table"
        ]["movement_firing_rate_hz"].to_numpy(dtype=float),
        "dark_movement_firing_rate_hz": context["movement_artifacts"]["dark"][
            "table"
        ]["movement_firing_rate_hz"].to_numpy(dtype=float),
        "light_trajectory_intervals": context["trajectory_intervals"]["light"],
        "dark_trajectory_intervals": context["trajectory_intervals"]["dark"],
        "graph_inputs": context["graph_inputs"],
        "upstream_provenance": _cv_pca_upstream_provenance(selection),
        "parameter_name": parameters["cv_pca_param_name"],
        "parameter_sha256": selection["cv_pca_effective_parameters_sha256"],
        "position_offset_samples": int(selection["position_offset_samples"]),
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "cv_pca_param_name"
        },
    }


def _write_cv_pca_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one complete cvPCA analysis NWB."""
    import pynwb

    from v1ca1.spyglass import cv_pca
    from v1ca1.spyglass.selection import unit_identity_sha256

    if analysis_nwbfile_table is None:
        raise ValueError("analysis_nwbfile_table is required for CVPCA output.")
    canonical = cv_pca.validate_cv_pca_result(result)
    expected_hashes = cv_pca.cv_pca_nwb_hashes(canonical)
    objects = cv_pca.cv_pca_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in (
                "selected_units",
                "lap_assignments",
                "trajectory_qc",
                "summary",
                "spectrum",
                "dataset",
                "provenance",
            ):
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError("CVPCA NWB object IDs must be unique.")
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "CVPCA analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = cv_pca.cv_pca_result_from_nwb_objects(
                    **{
                        name: stored_nwb.objects[
                            object_ids[f"{name}_object_id"]
                        ]
                        for name in (
                            "selected_units",
                            "lap_assignments",
                            "trajectory_qc",
                            "summary",
                            "spectrum",
                            "dataset",
                            "provenance",
                        )
                    }
                )
                if cv_pca.cv_pca_nwb_hashes(stored) != expected_hashes:
                    raise ValueError("CVPCA NWB objects changed during write.")
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("CVPCA analysis NWB was not created.")
    identities = canonical["selected_units"].loc[
        :, ["spikesorting_merge_id", "unit_id"]
    ].to_dict("records")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": cv_pca.NWB_ARTIFACT_SCHEMA_VERSION,
        "n_input_units": int(canonical["n_input_units"]),
        "n_selected_units": int(canonical["n_selected_units"]),
        "analysis_status": str(canonical["analysis_status"]),
        "selected_units_sha256": unit_identity_sha256(identities),
        "legacy_artifact_provenance": (
            dict(canonical["legacy_artifact_provenance"])
            if canonical.get("legacy_artifact_provenance")
            else None
        ),
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_cv_pca_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of cvPCA NWB scratch objects."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading CVPCA."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError("CVPCA NWB fetch must resolve exactly one result.")
    record = dict(records[0])
    expected = {
        "selected_units",
        "lap_assignments",
        "trajectory_qc",
        "summary",
        "spectrum",
        "dataset",
        "provenance",
    }
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(f"CVPCA NWB fetch is missing objects {missing!r}.")
    return {name: record[name] for name in expected}


def _make_cv_pca_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one immutable cvPCA analysis NWB."""
    from v1ca1.spyglass import cv_pca

    context = _load_cv_pca_context(
        key=key,
        parameters_table=parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=movement_firing_rate_selection_table,
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    result = cv_pca.compute_cv_pca(**_cv_pca_compute_inputs(context))
    del artifact_root
    return _write_cv_pca_nwb(
        nwb_file_name=str(context["selection"]["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _validate_cv_pca_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one loaded cvPCA bundle to match its DataJoint rows."""
    from v1ca1.spyglass import cv_pca
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256

    validated = cv_pca.validate_cv_pca_result(bundle)
    parameters = _validate_cv_pca_parameter_row(parameters_row)
    expected_parameters = cv_pca.validate_cv_pca_parameters(
        region=str(region_row["region_name"]),
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "cv_pca_param_name"
        },
    )
    expected_metadata = {
        "cv_pca_id": str(selection_row["cv_pca_id"]),
        "animal_name": str(animal_name),
        "date": str(date),
        "region": str(region_row["region_name"]),
        "light_epoch": str(selection_row["light_epoch"]),
        "dark_epoch": str(selection_row["dark_epoch"]),
        "parameter_name": parameters["cv_pca_param_name"],
        "parameter_sha256": selection_row[
            "cv_pca_effective_parameters_sha256"
        ],
    }
    for field_name, expected in expected_metadata.items():
        if str(validated[field_name]) != str(expected):
            raise ValueError(
                f"CVPCA artifact metadata differs from selection: {field_name}."
            )
    if validated["parameters"] != expected_parameters or validated[
        "upstream_provenance"
    ] != _cv_pca_upstream_provenance(selection_row):
        raise ValueError("CVPCA artifact parameters or provenance differ.")
    if int(validated["position_offset_samples"]) != int(
        selection_row["position_offset_samples"]
    ):
        raise ValueError("CVPCA artifact position offset differs from selection.")

    for field_name in (
        "n_input_units",
        "n_selected_units",
        "analysis_status",
        "artifact_origin",
    ):
        if str(result_row.get(field_name)) != str(validated[field_name]):
            raise ValueError(
                f"CVPCA result metadata disagrees with artifact: {field_name}."
            )
    identities = validated["selected_units"].loc[
        :, ["spikesorting_merge_id", "unit_id"]
    ].to_dict("records")
    selected_digest = unit_identity_sha256(identities)
    if selected_digest != str(result_row["selected_units_sha256"]) or (
        selected_digest != str(region_row["selected_units_sha256"])
    ):
        raise ValueError("CVPCA selected-unit digest disagrees with its group.")
    if str(result_row.get("artifact_schema_version")) != str(
        cv_pca.NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("CVPCA NWB artifact schema version disagrees.")
    for field_name, observed in cv_pca.cv_pca_nwb_hashes(validated).items():
        if str(result_row.get(field_name)) != observed:
            raise ValueError(
                "CVPCA result metadata disagrees with its NWB objects: "
                f"{field_name}."
            )
    artifact_legacy = validated.get("legacy_artifact_provenance") or None
    if result_row.get("legacy_artifact_provenance") != artifact_legacy:
        raise ValueError("CVPCA legacy provenance disagrees.")
    if provenance_sha256(parameters) != str(
        selection_row["cv_pca_parameters_sha256"]
    ):
        raise ValueError("CVPCA parameter snapshot is stale.")


def _load_cv_pca_result(
    *,
    result_row: Mapping[str, Any],
    result_table: Any,
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Fetch, reconstruct, and validate one canonical cvPCA NWB result."""
    from v1ca1.spyglass import cv_pca

    objects = _fetch_cv_pca_nwb_objects(
        result_table,
        {"cv_pca_id": result_row["cv_pca_id"]},
    )
    result = cv_pca.cv_pca_result_from_nwb_objects(**objects)
    _validate_cv_pca_artifact_link(
        bundle=result,
        result_row=result_row,
        selection_row=selection_row,
        parameters_row=parameters_row,
        region_row=region_row,
        animal_name=animal_name,
        date=date,
    )
    return result


def _register_existing_cv_pca_row(
    *,
    key: Mapping[str, Any],
    legacy_result_path: Path,
    legacy_summary_path: Path,
    parameters_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Strictly recompute and copy one legacy cvPCA pair into NWB."""
    from v1ca1.spyglass import cv_pca

    context = _load_cv_pca_context(
        key=key,
        parameters_table=parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=movement_firing_rate_selection_table,
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    registered = cv_pca.register_existing_cv_pca_artifact(
        legacy_result_path=Path(legacy_result_path),
        legacy_summary_path=Path(legacy_summary_path),
        compute_inputs=_cv_pca_compute_inputs(context),
        artifact_root=None,
        overwrite=False,
    )
    del artifact_root
    return _write_cv_pca_nwb(
        nwb_file_name=str(context["selection"]["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _write_path_specific_place_tuning_curve_nwb(
    *,
    nwb_file_name: str,
    curve: Any,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write and verify the three path-specific tuning DynamicTables."""
    import pynwb

    from v1ca1.spyglass.path_specific_place import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        path_specific_place_bins_sha256,
        path_specific_place_bins_to_dynamic_table,
        path_specific_place_provenance_sha256,
        path_specific_place_provenance_to_dynamic_table,
        path_specific_place_tuning_curve_from_nwb_objects,
        path_specific_place_tuning_sha256,
        path_specific_place_tuning_to_dynamic_table,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for "
            "PathSpecificPlaceTuningCurve output."
        )
    object_specs = {
        "path_specific_place_tuning": (
            path_specific_place_tuning_to_dynamic_table(curve)
        ),
        "path_specific_place_bins": (
            path_specific_place_bins_to_dynamic_table(curve)
        ),
        "path_specific_place_provenance": (
            path_specific_place_provenance_to_dynamic_table(curve)
        ),
    }
    expected_hashes = {
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
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name, nwb_object in object_specs.items():
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(nwb_object)
                )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "PathSpecificPlaceTuningCurve analysis NWB failed PyNWB "
                    f"validation: {validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_curve = path_specific_place_tuning_curve_from_nwb_objects(
                    stored_nwb.objects[
                        object_ids["path_specific_place_tuning_object_id"]
                    ],
                    stored_nwb.objects[
                        object_ids["path_specific_place_bins_object_id"]
                    ],
                    stored_nwb.objects[
                        object_ids[
                            "path_specific_place_provenance_object_id"
                        ]
                    ],
                )
                observed_hashes = {
                    "path_specific_place_tuning_sha256": (
                        path_specific_place_tuning_sha256(stored_curve)
                    ),
                    "path_specific_place_bins_sha256": (
                        path_specific_place_bins_sha256(stored_curve)
                    ),
                    "path_specific_place_provenance_sha256": (
                        path_specific_place_provenance_sha256(stored_curve)
                    ),
                }
                if observed_hashes != expected_hashes:
                    raise ValueError(
                        "Path-specific place tuning NWB objects changed "
                        "during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError(
            "PathSpecificPlaceTuningCurve analysis NWB was not created."
        )
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_path_specific_place_tuning_curve_nwb_objects(
    tuning_curve_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one set of path-specific tuning NWB objects."""
    relation = tuning_curve_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading PathSpecificPlaceTuningCurve."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "PathSpecificPlaceTuningCurve NWB fetch must resolve exactly one "
            "result."
        )
    record = dict(records[0])
    expected = {
        "path_specific_place_tuning",
        "path_specific_place_bins",
        "path_specific_place_provenance",
    }
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(
            "PathSpecificPlaceTuningCurve NWB fetch is missing objects "
            f"{missing!r}."
        )
    return {name: record[name] for name in expected}


def _load_path_specific_place_tuning_curve_result(
    *,
    result_row: Mapping[str, Any],
    tuning_curve_table: Any,
    selection_row: Mapping[str, Any],
) -> Any:
    """Load, reconstruct, and cross-check one path-specific tuning curve."""
    from v1ca1.spyglass.path_specific_place import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        path_specific_place_bins_sha256,
        path_specific_place_provenance_sha256,
        path_specific_place_tuning_curve_from_nwb_objects,
        path_specific_place_tuning_sha256,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "PathSpecificPlaceTuningCurve artifact schema version is "
            "unsupported."
        )
    curve_id = result_row["path_specific_place_tuning_curve_id"]
    objects = _fetch_path_specific_place_tuning_curve_nwb_objects(
        tuning_curve_table,
        {"path_specific_place_tuning_curve_id": curve_id},
    )
    curve = path_specific_place_tuning_curve_from_nwb_objects(
        objects["path_specific_place_tuning"],
        objects["path_specific_place_bins"],
        objects["path_specific_place_provenance"],
    )
    observed_hashes = {
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
    for field_name, observed in observed_hashes.items():
        if observed != str(result_row.get(field_name)):
            raise ValueError(
                "PathSpecificPlaceTuningCurve result metadata disagrees with "
                f"its NWB objects: {field_name}."
            )
    _validate_tuning_curve_artifact_link(
        curve=curve,
        result_row=result_row,
        selection_row=selection_row,
    )
    return curve


def _make_path_specific_place_tuning_curve_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    position_table: Any,
    wtrack_graph_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one trial-subset tuning curve to analysis NWB."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.path_specific_place import (
        compute_selected_path_specific_place_tuning_curve,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    validated_selection = _path_specific_place_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        parameters_table=parameters_table,
    )
    if str(validated_selection["path_specific_place_tuning_curve_id"]) != str(
        key["path_specific_place_tuning_curve_id"]
    ):
        raise ValueError("PathSpecificPlaceTuningCurve selection UUID is stale.")
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    context = {**key, **movement_selection}
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, context)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, context)
    trajectory_row = _fetch1_dict(trajectory_intervals_table, context)
    position_row = _fetch1_dict(position_table, context)
    graph_row = _fetch1_dict(wtrack_graph_table, context)
    if str(key["trajectory_type"]) != str(key["configuration_name"]):
        raise ValueError(
            "PathSpecificPlaceTuningCurve graph configuration must match "
            "trajectory_type."
        )
    animal_name, session_date = _session_identity(session_table, context)
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")

    loaded_spikes = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=context,
        time_support=(epoch_start, epoch_stop),
    )
    region = _analysis_region(
        loaded_spikes["registration_row"]["region_name"]
    )
    selected_units_sha256 = unit_identity_sha256(loaded_spikes["unit_ids"])
    if selected_units_sha256 != str(
        movement_result["selected_units_sha256"]
    ):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )

    nwb_path = Path(
        nwbfile_table.get_abs_path(str(movement_selection["nwb_file_name"]))
    )
    position = None
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        trajectory_interval = load_interval_set(nwbfile, trajectory_row)
        graph_inputs = load_wtrack_graph(nwbfile, graph_row)
        if movement["analysis_status"] == "valid":
            position = load_position(
                nwbfile,
                position_row,
                apply_analysis_offset=True,
            )

    result = compute_selected_path_specific_place_tuning_curve(
        animal_name=animal_name,
        date=session_date,
        region=region,
        epoch=str(movement_selection["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=position,
        trajectory_intervals=trajectory_interval,
        graph_inputs=graph_inputs,
        movement_intervals=movement["movement_intervals"],
        movement_analysis_status=movement["analysis_status"],
        bin_size_cm=parameters["place_bin_size_cm"],
        bin_count=parameters["position_bin_count"],
        sigma_bins=parameters["gaussian_smoothing_sigma_bins"],
        trial_subset=str(key["trial_subset"]),
    )
    result["tuning_curve"].attrs.update(
        _tuning_curve_artifact_attributes(
            key,
            selected_units_sha256=selected_units_sha256,
        )
    )
    del artifact_root
    nwb_artifact = _write_path_specific_place_tuning_curve_nwb(
        nwb_file_name=str(movement_selection["nwb_file_name"]),
        curve=result["tuning_curve"],
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "n_trials": int(result["n_trials"]),
        "support_duration_s": float(result["support_duration_s"]),
        "n_feature_samples": int(result["n_feature_samples"]),
        "n_position_bins": int(result["n_position_bins"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def _write_dpp_tuning_curve_nwb(
    *,
    nwb_file_name: str,
    curve: Any,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write and verify the three directional-progression DynamicTables."""
    import pynwb

    from v1ca1.spyglass.dpp import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        dpp_bins_sha256,
        dpp_bins_to_dynamic_table,
        dpp_provenance_sha256,
        dpp_provenance_to_dynamic_table,
        dpp_tuning_curve_from_nwb_objects,
        dpp_tuning_sha256,
        dpp_tuning_to_dynamic_table,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for DPPTuningCurve output."
        )
    object_specs = {
        "dpp_tuning": dpp_tuning_to_dynamic_table(curve),
        "dpp_bins": dpp_bins_to_dynamic_table(curve),
        "dpp_provenance": dpp_provenance_to_dynamic_table(curve),
    }
    expected_hashes = {
        "dpp_tuning_sha256": dpp_tuning_sha256(curve),
        "dpp_bins_sha256": dpp_bins_sha256(curve),
        "dpp_provenance_sha256": dpp_provenance_sha256(curve),
    }
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name, nwb_object in object_specs.items():
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(nwb_object)
                )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "DPPTuningCurve analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_curve = dpp_tuning_curve_from_nwb_objects(
                    stored_nwb.objects[object_ids["dpp_tuning_object_id"]],
                    stored_nwb.objects[object_ids["dpp_bins_object_id"]],
                    stored_nwb.objects[object_ids["dpp_provenance_object_id"]],
                )
                observed_hashes = {
                    "dpp_tuning_sha256": dpp_tuning_sha256(stored_curve),
                    "dpp_bins_sha256": dpp_bins_sha256(stored_curve),
                    "dpp_provenance_sha256": (
                        dpp_provenance_sha256(stored_curve)
                    ),
                }
                if observed_hashes != expected_hashes:
                    raise ValueError(
                        "DPP tuning NWB objects changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("DPPTuningCurve analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_dpp_tuning_curve_nwb_objects(
    tuning_curve_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one set of directional-progression NWB objects."""
    relation = tuning_curve_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading DPPTuningCurve."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "DPPTuningCurve NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    expected = {"dpp_tuning", "dpp_bins", "dpp_provenance"}
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(
            f"DPPTuningCurve NWB fetch is missing objects {missing!r}."
        )
    return {name: record[name] for name in expected}


def _load_dpp_tuning_curve_result(
    *,
    result_row: Mapping[str, Any],
    tuning_curve_table: Any,
    selection_row: Mapping[str, Any],
) -> Any:
    """Load, reconstruct, and cross-check one DPP tuning curve."""
    from v1ca1.spyglass.dpp import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        dpp_bins_sha256,
        dpp_provenance_sha256,
        dpp_tuning_curve_from_nwb_objects,
        dpp_tuning_sha256,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("DPPTuningCurve artifact schema version is unsupported.")
    curve_id = result_row["dpp_tuning_curve_id"]
    objects = _fetch_dpp_tuning_curve_nwb_objects(
        tuning_curve_table,
        {"dpp_tuning_curve_id": curve_id},
    )
    curve = dpp_tuning_curve_from_nwb_objects(
        objects["dpp_tuning"],
        objects["dpp_bins"],
        objects["dpp_provenance"],
    )
    observed_hashes = {
        "dpp_tuning_sha256": dpp_tuning_sha256(curve),
        "dpp_bins_sha256": dpp_bins_sha256(curve),
        "dpp_provenance_sha256": dpp_provenance_sha256(curve),
    }
    for field_name, observed in observed_hashes.items():
        if observed != str(result_row.get(field_name)):
            raise ValueError(
                "DPPTuningCurve result metadata disagrees with its NWB "
                f"objects: {field_name}."
            )
    _validate_dpp_tuning_curve_artifact_link(
        curve=curve,
        result_row=result_row,
        selection_row=selection_row,
    )
    return curve


def _make_dpp_tuning_curve_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    position_table: Any,
    wtrack_graph_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one trial-subset DPP curve to analysis NWB."""
    import pynwb

    from v1ca1.spyglass.dpp import (
        compute_selected_dpp_tuning_curve,
        get_dpp_trajectory_pair,
    )
    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    validated_selection = _dpp_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        parameters_table=parameters_table,
    )
    if str(validated_selection["dpp_tuning_curve_id"]) != str(
        key["dpp_tuning_curve_id"]
    ):
        raise ValueError("DPPTuningCurve selection UUID is stale.")

    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    context = {**key, **movement_selection}
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, context)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, context)
    position_row = _fetch1_dict(position_table, context)
    trajectory_pair = get_dpp_trajectory_pair(str(key["turn_type"]))
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": movement_selection["nwb_file_name"],
                "epoch": movement_selection["epoch"],
                "trajectory_type": trajectory_type,
            },
        )
        for trajectory_type in trajectory_pair
    }
    graph_rows = {
        trajectory_type: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": movement_selection["nwb_file_name"],
                "configuration_name": trajectory_type,
            },
        )
        for trajectory_type in trajectory_pair
    }
    animal_name, session_date = _session_identity(session_table, context)
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")

    loaded_spikes = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=context,
        time_support=(epoch_start, epoch_stop),
    )
    region = _analysis_region(
        loaded_spikes["registration_row"]["region_name"]
    )
    selected_units_sha256 = unit_identity_sha256(loaded_spikes["unit_ids"])
    if selected_units_sha256 != str(
        movement_result["selected_units_sha256"]
    ):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )

    nwb_path = Path(
        nwbfile_table.get_abs_path(str(movement_selection["nwb_file_name"]))
    )
    position = None
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
        if movement["analysis_status"] == "valid":
            position = load_position(
                nwbfile,
                position_row,
                apply_analysis_offset=True,
            )

    result = compute_selected_dpp_tuning_curve(
        animal_name=animal_name,
        date=session_date,
        region=region,
        epoch=str(movement_selection["epoch"]),
        turn_type=str(key["turn_type"]),
        trial_subset=str(key["trial_subset"]),
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=position,
        trajectory_intervals_by_type=trajectory_intervals,
        graph_inputs_by_trajectory=graph_inputs,
        movement_intervals=movement["movement_intervals"],
        movement_analysis_status=movement["analysis_status"],
        bin_size_cm=parameters["place_bin_size_cm"],
        bin_count=parameters["position_bin_count"],
        sigma_bins=parameters["gaussian_smoothing_sigma_bins"],
    )
    result["tuning_curve"].attrs.update(
        _dpp_tuning_curve_artifact_attributes(
            key,
            selected_units_sha256=selected_units_sha256,
        )
    )
    del artifact_root
    nwb_artifact = _write_dpp_tuning_curve_nwb(
        nwb_file_name=str(movement_selection["nwb_file_name"]),
        curve=result["tuning_curve"],
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "n_trials": int(result["n_trials"]),
        "n_outbound_trials": int(result["n_outbound_trials"]),
        "n_inbound_trials": int(result["n_inbound_trials"]),
        "support_duration_s": float(result["support_duration_s"]),
        "n_feature_samples": int(result["n_feature_samples"]),
        "n_position_bins": int(result["n_position_bins"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def _write_path_specific_place_stability_nwb(
    *,
    nwb_file_name: str,
    table: Any,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, validate, and register one stability DynamicTable."""
    import pynwb

    from v1ca1.spyglass.stability import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        stability_table_from_dynamic_table,
        stability_table_sha256,
        stability_table_to_dynamic_table,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for PathSpecificPlaceStability "
            "output."
        )
    stability_sha256 = stability_table_sha256(table)
    stability_object = stability_table_to_dynamic_table(table)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            stability_object_id = str(
                builder.add_nwb_object(stability_object)
            )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "PathSpecificPlaceStability analysis NWB failed PyNWB "
                    f"validation: {validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_table = stability_table_from_dynamic_table(
                    stored_nwb.objects[stability_object_id]
                )
                if stability_table_sha256(stored_table) != stability_sha256:
                    raise ValueError(
                        "Path-specific place-stability NWB object changed "
                        "during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("PathSpecificPlaceStability analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        "stability_object_id": stability_object_id,
        "stability_sha256": stability_sha256,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _make_path_specific_place_stability_row(
    *,
    key: Mapping[str, Any],
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute stability from one persisted odd/even tuning-curve pair."""
    from v1ca1.spyglass.stability import (
        compute_selected_stability_from_tuning_curves,
    )

    validated_selection = _stability_selection_row(
        key=key,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
    )
    if str(validated_selection["path_specific_place_stability_id"]) != str(
        key["path_specific_place_stability_id"]
    ):
        raise ValueError("PathSpecificPlaceStability selection UUID is stale.")
    curve_results: dict[str, dict[str, Any]] = {}
    curve_selections: dict[str, dict[str, Any]] = {}
    curves: dict[str, Any] = {}
    for subset in ("odd", "even"):
        curve_key = {
            "path_specific_place_tuning_curve_id": key[
                f"{subset}_path_specific_place_tuning_curve_id"
            ]
        }
        curve_results[subset] = _fetch1_dict(tuning_curve_table, curve_key)
        curve_selections[subset] = _fetch1_dict(
            tuning_curve_selection_table,
            curve_key,
        )
        if str(curve_selections[subset]["trial_subset"]) != subset:
            raise ValueError(
                "PathSpecificPlaceStability requires matching odd and even curves."
            )
        curves[subset] = _load_path_specific_place_tuning_curve_result(
            result_row=curve_results[subset],
            tuning_curve_table=tuning_curve_table,
            selection_row=curve_selections[subset],
        )

    odd_selection = curve_selections["odd"]
    movement_key = {
        "movement_firing_rate_id": odd_selection["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(movement_firing_rate_table, movement_key)
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, movement_selection)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    animal_name, session_date = _session_identity(
        session_table,
        movement_selection,
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": movement_selection[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    _validate_region_movement_identity(
        analysis_name="PathSpecificPlaceStability",
        region_row=region_row,
        movement_selection=movement_selection,
        movement_result=movement_result,
    )
    region = _analysis_region(region_row["region_name"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    selected_units_sha256 = str(movement_result["selected_units_sha256"])
    for subset in ("odd", "even"):
        curve = curves[subset]
        expected_metadata = {
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
            "trajectory_type": str(odd_selection["trajectory_type"]),
            "trial_subset": subset,
        }
        for field_name, expected_value in expected_metadata.items():
            if str(curve.attrs[field_name]) != str(expected_value):
                raise ValueError(
                    "PathSpecificPlaceTuningCurve artifact does not match its "
                    f"selection: {field_name}."
                )
        if str(curve_results[subset]["selected_units_sha256"]) != (
            selected_units_sha256
        ):
            raise ValueError(
                "Odd/even tuning curves and MovementFiringRate must contain "
                "the same selected units."
            )
    result = compute_selected_stability_from_tuning_curves(
        odd_tuning_curve=curves["odd"],
        even_tuning_curve=curves["even"],
        movement_firing_rate_table=movement["table"],
    )
    del artifact_root
    nwb_artifact = _write_path_specific_place_stability_nwb(
        nwb_file_name=str(movement_selection["nwb_file_name"]),
        table=result["table"],
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def _load_tuning_similarity_inputs(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    tuning_curve_parameters_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and cross-check the four curves and shared movement artifact."""
    validated_selection = _tuning_similarity_selection_row(
        key=key,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        parameters_table=parameters_table,
    )
    if str(
        validated_selection["path_specific_place_tuning_similarity_id"]
    ) != str(key["path_specific_place_tuning_similarity_id"]):
        raise ValueError(
            "PathSpecificPlaceTuningSimilarity selection UUID is stale."
        )
    parameters = _validate_tuning_similarity_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "tuning_similarity_param_name": key[
                    "tuning_similarity_param_name"
                ]
            },
        )
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="tuning_similarity_parameters_sha256",
    )

    curve_id_fields = {
        "center_to_left": "center_to_left_tuning_curve_id",
        "center_to_right": "center_to_right_tuning_curve_id",
        "left_to_center": "left_to_center_tuning_curve_id",
        "right_to_center": "right_to_center_tuning_curve_id",
    }
    curve_results: dict[str, dict[str, Any]] = {}
    curve_selections: dict[str, dict[str, Any]] = {}
    curves: dict[str, Any] = {}
    for trajectory_type, field_name in curve_id_fields.items():
        curve_key = {
            "path_specific_place_tuning_curve_id": key[field_name]
        }
        result_row = _fetch1_dict(tuning_curve_table, curve_key)
        selection_row = _fetch1_dict(
            tuning_curve_selection_table,
            curve_key,
        )
        curve = _load_path_specific_place_tuning_curve_result(
            result_row=result_row,
            tuning_curve_table=tuning_curve_table,
            selection_row=selection_row,
        )
        curve_results[trajectory_type] = result_row
        curve_selections[trajectory_type] = selection_row
        curves[trajectory_type] = curve

    reference_selection = curve_selections["center_to_left"]
    tuning_curve_parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(tuning_curve_parameters_table, reference_selection)
    )
    _validate_frozen_parameters(
        reference_selection,
        tuning_curve_parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    movement_key = {
        "movement_firing_rate_id": reference_selection[
            "movement_firing_rate_id"
        ]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, movement_selection)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    animal_name, session_date = _session_identity(
        session_table,
        movement_selection,
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": movement_selection[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    _validate_region_movement_identity(
        analysis_name="PathSpecificPlaceTuningSimilarity",
        region_row=region_row,
        movement_selection=movement_selection,
        movement_result=movement_result,
    )
    region = _analysis_region(region_row["region_name"])
    epoch = str(movement_selection["epoch"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": epoch,
        },
    )
    selected_units_sha256 = str(movement_result["selected_units_sha256"])
    expected_curve_metadata = {
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
        "epoch": epoch,
        "trial_subset": "all",
    }
    for trajectory_type, curve in curves.items():
        if str(curve_results[trajectory_type]["selected_units_sha256"]) != (
            selected_units_sha256
        ):
            raise ValueError(
                "All four tuning curves and MovementFiringRate must contain "
                "the same selected units."
            )
        metadata = {
            **expected_curve_metadata,
            "trajectory_type": trajectory_type,
        }
        for field_name, expected_value in metadata.items():
            if str(curve.attrs.get(field_name, "")) != str(expected_value):
                raise ValueError(
                    "PathSpecificPlaceTuningCurve artifact does not match the "
                    f"shared similarity selection: {field_name}."
                )

    return {
        "parameters": parameters,
        "tuning_curve_parameters": tuning_curve_parameters,
        "curves": curves,
        "curve_results": curve_results,
        "curve_selections": curve_selections,
        "movement_result": movement_result,
        "movement_selection": movement_selection,
        "movement_parameters": movement_parameters,
        "movement_table": movement["table"],
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
        "epoch": epoch,
        "selected_units_sha256": selected_units_sha256,
    }


def _validate_tuning_similarity_artifact_link(
    *,
    table: Any,
    result_row: Mapping[str, Any],
    similarity_metric: str,
) -> None:
    """Require one canonical similarity artifact to match its result row."""
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.tuning_similarity import (
        summarize_tuning_similarity_table,
    )

    summary = summarize_tuning_similarity_table(table)
    for field_name in (
        "n_units",
        "n_valid_comparisons",
        "n_units_with_valid_comparison",
        "analysis_status",
    ):
        if str(result_row[field_name]) != str(summary[field_name]):
            raise ValueError(
                "PathSpecificPlaceTuningSimilarity result metadata disagrees "
                f"with its artifact: {field_name}."
            )
    if not table.empty:
        metrics = table["similarity_metric"].astype(str).unique().tolist()
        if metrics != [str(similarity_metric)]:
            raise ValueError(
                "PathSpecificPlaceTuningSimilarity artifact does not match "
                "its selected metric."
            )
    identities = (
        table.loc[:, ["spikesorting_merge_id", "unit_id", "stable_unit_id"]]
        .drop_duplicates("stable_unit_id")
        .to_dict("records")
    )
    if unit_identity_sha256(identities) != str(
        result_row["selected_units_sha256"]
    ):
        raise ValueError(
            "PathSpecificPlaceTuningSimilarity unit identities disagree with "
            "its selected-unit digest."
        )


def _write_path_specific_place_tuning_similarity_nwb(
    *,
    nwb_file_name: str,
    table: Any,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, validate, and register one tuning-similarity DynamicTable."""
    import pynwb

    from v1ca1.spyglass.tuning_similarity import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        tuning_similarity_table_from_dynamic_table,
        tuning_similarity_table_sha256,
        tuning_similarity_table_to_dynamic_table,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for "
            "PathSpecificPlaceTuningSimilarity output."
        )
    similarity_sha256 = tuning_similarity_table_sha256(table)
    similarity_object = tuning_similarity_table_to_dynamic_table(table)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            similarity_object_id = str(
                builder.add_nwb_object(similarity_object)
            )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "PathSpecificPlaceTuningSimilarity analysis NWB failed "
                    f"PyNWB validation: {validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_table = tuning_similarity_table_from_dynamic_table(
                    stored_nwb.objects[similarity_object_id]
                )
                if (
                    tuning_similarity_table_sha256(stored_table)
                    != similarity_sha256
                ):
                    raise ValueError(
                        "Path-specific tuning-similarity NWB object changed "
                        "during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError(
            "PathSpecificPlaceTuningSimilarity analysis NWB was not created."
        )
    return {
        "analysis_file_name": analysis_file_name,
        "similarity_object_id": similarity_object_id,
        "similarity_sha256": similarity_sha256,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_tuning_similarity_result_nwb_object(
    similarity_table: Any,
    key: Mapping[str, Any],
) -> Any:
    """Fetch exactly one PathSpecificPlaceTuningSimilarity NWB object."""
    relation = similarity_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading PathSpecificPlaceTuningSimilarity."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "PathSpecificPlaceTuningSimilarity NWB fetch must resolve exactly "
            "one result."
        )
    record = dict(records[0])
    if "similarity" not in record:
        raise ValueError(
            "PathSpecificPlaceTuningSimilarity NWB fetch is missing the "
            "similarity object."
        )
    return record["similarity"]


def _load_path_specific_place_tuning_similarity_result(
    *,
    result_row: Mapping[str, Any],
    similarity_table: Any,
    similarity_metric: str,
) -> Any:
    """Load one similarity DynamicTable and cross-check result metadata."""
    from v1ca1.spyglass.tuning_similarity import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        tuning_similarity_table_from_dynamic_table,
        tuning_similarity_table_sha256,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "PathSpecificPlaceTuningSimilarity artifact schema version is "
            "unsupported."
        )
    similarity_id = result_row[
        "path_specific_place_tuning_similarity_id"
    ]
    fetched = _fetch_tuning_similarity_result_nwb_object(
        similarity_table,
        {"path_specific_place_tuning_similarity_id": similarity_id},
    )
    table = tuning_similarity_table_from_dynamic_table(fetched)
    if tuning_similarity_table_sha256(table) != str(
        result_row.get("similarity_sha256")
    ):
        raise ValueError(
            "PathSpecificPlaceTuningSimilarity result metadata disagrees "
            "with its NWB object: similarity_sha256."
        )
    _validate_tuning_similarity_artifact_link(
        table=table,
        result_row=result_row,
        similarity_metric=similarity_metric,
    )
    return table


def _make_path_specific_place_tuning_similarity_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    tuning_curve_parameters_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write four path comparisons for every selected unit."""
    from v1ca1.spyglass.tuning_similarity import (
        compute_tuning_similarity_from_curves,
    )

    inputs = _load_tuning_similarity_inputs(
        key=key,
        parameters_table=parameters_table,
        tuning_curve_parameters_table=tuning_curve_parameters_table,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        session_table=session_table,
    )
    result = compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=inputs["curves"],
        movement_firing_rate_table=inputs["movement_table"],
        similarity_metric=inputs["parameters"]["similarity_metric"],
    )
    del artifact_root
    nwb_artifact = _write_path_specific_place_tuning_similarity_nwb(
        nwb_file_name=str(inputs["movement_selection"]["nwb_file_name"]),
        table=result["table"],
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units": int(result["n_units"]),
        "n_valid_comparisons": int(result["n_valid_comparisons"]),
        "n_units_with_valid_comparison": int(
            result["n_units_with_valid_comparison"]
        ),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": inputs["selected_units_sha256"],
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def _register_existing_path_specific_place_tuning_similarity_row(
    *,
    key: Mapping[str, Any],
    similarity_path: Path,
    overwrite: bool,
    parameters_table: Any,
    tuning_curve_parameters_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Validate and register one matching complete legacy similarity file."""
    import pandas as pd

    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent
    from v1ca1.spyglass.tuning_similarity import (
        normalize_legacy_all_units_similarity_table,
        summarize_tuning_similarity_table,
    )

    inputs = _load_tuning_similarity_inputs(
        key=key,
        parameters_table=parameters_table,
        tuning_curve_parameters_table=tuning_curve_parameters_table,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        session_table=session_table,
    )
    tuning_curve_parameters = inputs["tuning_curve_parameters"]
    if tuning_curve_parameters != dict(
        table_specs.LEGACY_TUNING_CURVE_PARAMETERS
    ):
        raise ValueError(
            "Legacy tuning-similarity registration requires four "
            "legacy_4cm_unsmoothed tuning curves."
        )
    default_movement = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    for field_name in (
        "speed_threshold_cm_s",
        "speed_smoothing_sigma_s",
    ):
        if not math.isclose(
            float(inputs["movement_parameters"][field_name]),
            float(default_movement[field_name]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Legacy tuning-similarity registration requires the default "
                f"movement parameters; {field_name} differs."
            )

    context = {**key, **inputs["movement_selection"]}
    loaded_units = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=context,
    )
    selected_units_sha256 = unit_identity_sha256(loaded_units["unit_ids"])
    if selected_units_sha256 != inputs["selected_units_sha256"]:
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(spike_sorting_output, {"merge_id": merge_id})
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy tuning-similarity registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )

    unit_identity_resolver: dict[str, dict[str, Any]] = {}
    for metadata in loaded_units["unit_metadata"]:
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in unit_identity_resolver:
            raise ValueError(
                "Every imported selected unit requires a unique "
                "sorting_unit_id for legacy tuning-similarity registration."
            )
        unit_identity_resolver[resolver_key] = {
            "spikesorting_merge_id": metadata["spikesorting_merge_id"],
            "unit_id": metadata["unit_id"],
        }
    if len(unit_identity_resolver) != len(loaded_units["unit_ids"]):
        raise ValueError(
            "Every imported selected unit requires one legacy identity."
        )

    source = Path(similarity_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Legacy tuning-similarity artifact not found: {source}"
        )
    if source.suffix != ".parquet":
        raise ValueError(
            "Legacy tuning-similarity source must be a Parquet file."
        )
    if not source.stem.endswith("_all_units"):
        raise ValueError(
            "Legacy registration requires a *_all_units.parquet source."
        )
    table = normalize_legacy_all_units_similarity_table(
        pd.read_parquet(source),
        tuning_curves_by_trajectory=inputs["curves"],
        movement_firing_rate_table=inputs["movement_table"],
        similarity_metric=inputs["parameters"]["similarity_metric"],
        unit_identity_resolver=unit_identity_resolver,
    )
    summary = summarize_tuning_similarity_table(table)
    provenance = {
        "source_path": str(source.resolve(strict=True)),
        "source_sha256": _file_sha256(source),
        "legacy_unit_column": "unit",
    }
    provenance.update(
        {
            "source_v1ca1_git_commit": source_v1ca1_git_commit,
            "source_spyglass_git_commit": source_spyglass_git_commit,
            "assumed_parameters": {
                "movement": inputs["movement_parameters"],
                "tuning_curve": tuning_curve_parameters,
                "tuning_similarity": inputs["parameters"],
            },
        }
    )
    del artifact_root, overwrite
    nwb_artifact = _write_path_specific_place_tuning_similarity_nwb(
        nwb_file_name=str(inputs["movement_selection"]["nwb_file_name"]),
        table=table,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units": int(summary["n_units"]),
        "n_valid_comparisons": int(summary["n_valid_comparisons"]),
        "n_units_with_valid_comparison": int(
            summary["n_units_with_valid_comparison"]
        ),
        "analysis_status": str(summary["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
    }


def _register_existing_path_specific_place_tuning_curve_row(
    *,
    key: Mapping[str, Any],
    tuning_curve_path: Path,
    overwrite: bool,
    parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    position_table: Any,
    wtrack_graph_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Normalize one legacy all-trial curve into three analysis-NWB tables."""
    import pynwb
    import xarray as xr

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.path_specific_place import (
        build_path_specific_linear_position,
        graph_length_from_inputs,
        normalize_legacy_all_trial_tuning_curve,
        select_trial_subset_intervals,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent

    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    validated_selection = _path_specific_place_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        parameters_table=parameters_table,
    )
    if str(validated_selection["path_specific_place_tuning_curve_id"]) != str(
        key["path_specific_place_tuning_curve_id"]
    ):
        raise ValueError("PathSpecificPlaceTuningCurve selection UUID is stale.")
    if str(key["trial_subset"]) != "all":
        raise ValueError(
            "Legacy tuning-curve registration is only available for trial_subset='all'."
        )
    legacy_parameters = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    if any(parameters[name] != legacy_parameters[name] for name in parameters):
        raise ValueError(
            "Legacy tuning-curve registration requires the "
            "legacy_4cm_unsmoothed parameter preset."
        )

    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(movement_firing_rate_table, movement_key)
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    context = {**key, **movement_selection}
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, context)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, context)
    trajectory_row = _fetch1_dict(trajectory_intervals_table, context)
    position_row = _fetch1_dict(position_table, context)
    graph_row = _fetch1_dict(wtrack_graph_table, context)
    _validate_legacy_tuning_curve_inputs(
        position_row=position_row,
        movement_parameters=movement_parameters,
    )
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")
    animal_name, session_date = _session_identity(session_table, context)
    loaded_units = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=context,
    )
    region = _analysis_region(
        loaded_units["registration_row"]["region_name"]
    )
    selected_units_sha256 = unit_identity_sha256(loaded_units["unit_ids"])
    if selected_units_sha256 != str(movement_result["selected_units_sha256"]):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(spike_sorting_output, {"merge_id": merge_id})
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy tuning-curve registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    if movement["analysis_status"] not in {"valid", "no_units"}:
        raise ValueError(
            "A legacy tuning curve cannot represent an upstream terminal "
            f"movement status {movement['analysis_status']!r}."
        )

    nwb_path = Path(
        nwbfile_table.get_abs_path(str(movement_selection["nwb_file_name"]))
    )
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        trajectory_intervals = load_interval_set(nwbfile, trajectory_row)
        graph_inputs = load_wtrack_graph(nwbfile, graph_row)
        position = (
            load_position(
                nwbfile,
                position_row,
                apply_analysis_offset=True,
            )
            if movement["analysis_status"] == "valid"
            else None
        )
    selected_trials = select_trial_subset_intervals(trajectory_intervals, "all")
    n_trials = len(np.asarray(selected_trials.start).reshape(-1))
    support = selected_trials.intersect(movement["movement_intervals"])
    support_duration_s = float(support.tot_length())
    graph_length_cm = graph_length_from_inputs(
        graph_inputs,
        trajectory_type=str(key["trajectory_type"]),
    )
    n_feature_samples = 0
    n_valid_position_samples = 0
    if movement["analysis_status"] == "valid":
        linear_position, computed_length = build_path_specific_linear_position(
            position=position,
            trajectory_intervals=trajectory_intervals,
            graph_inputs=graph_inputs,
            trajectory_type=str(key["trajectory_type"]),
        )
        if not math.isclose(
            graph_length_cm,
            computed_length,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError("Linearization and WTrackGraph lengths disagree.")
        feature_values = np.asarray(
            linear_position.restrict(support).d,
            dtype=float,
        ).reshape(-1)
        n_feature_samples = len(feature_values)
        n_valid_position_samples = int(
            np.count_nonzero(np.isfinite(feature_values))
        )

    unit_identity_resolver: dict[Any, dict[str, Any]] = {}
    for group_unit_id, metadata in enumerate(loaded_units["unit_metadata"]):
        sorting_unit_id = metadata.get("sorting_unit_id")
        if sorting_unit_id is None or sorting_unit_id in unit_identity_resolver:
            raise ValueError(
                "Every imported selected unit requires a unique sorting_unit_id "
                "for legacy tuning-curve registration."
            )
        unit_identity_resolver[sorting_unit_id] = {
            "spikesorting_merge_id": metadata["spikesorting_merge_id"],
            "unit_id": metadata["unit_id"],
            "group_unit_id": group_unit_id,
            "sorting_unit_id": sorting_unit_id,
        }

    del artifact_root, overwrite
    source_path = Path(tuning_curve_path).resolve(strict=True)
    with xr.open_dataarray(source_path) as opened:
        legacy_curve = opened.load()
    curve = normalize_legacy_all_trial_tuning_curve(
        legacy_curve,
        unit_identity_resolver=unit_identity_resolver,
        animal_name=animal_name,
        date=session_date,
        region=region,
        epoch=str(movement_selection["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        graph_length_cm=graph_length_cm,
        n_trials=n_trials,
        support_duration_s=support_duration_s,
        n_feature_samples=n_feature_samples,
        n_valid_position_samples=n_valid_position_samples,
        bin_size_cm=parameters["place_bin_size_cm"],
        bin_count=parameters["position_bin_count"],
        sigma_bins=parameters["gaussian_smoothing_sigma_bins"],
    )
    curve.attrs.update(
        _tuning_curve_artifact_attributes(
            key,
            selected_units_sha256=selected_units_sha256,
        )
    )
    nwb_artifact = _write_path_specific_place_tuning_curve_nwb(
        nwb_file_name=str(movement_selection["nwb_file_name"]),
        curve=curve,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    provenance = {
        "source_path": str(source_path),
        "source_sha256": _file_sha256(source_path),
        "legacy_unit_coordinate": "sorting_unit_id",
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "source_spyglass_git_commit": source_spyglass_git_commit,
        "assumed_parameters": {
            "position": {
                field_name: position_row[field_name]
                for field_name in (
                    "position_series_name",
                    "position_role",
                    "analysis_start_offset_samples",
                )
            },
            "movement": movement_parameters,
            "tuning_curve": parameters,
        },
    }
    return {
        **nwb_artifact,
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "support_duration_s": float(curve.attrs["support_duration_s"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[curve.dims[1]]),
        "analysis_status": str(curve.attrs["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
    }


def _register_existing_dpp_tuning_curve_row(
    *,
    key: Mapping[str, Any],
    tuning_curve_path: Path,
    overwrite: bool,
    parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    position_table: Any,
    wtrack_graph_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Normalize one legacy all-trial DPP curve into three NWB tables."""
    import pynwb
    import xarray as xr

    from v1ca1.spyglass.dpp import (
        common_graph_length_from_inputs,
        get_dpp_trajectory_pair,
        normalize_legacy_all_trial_dpp_tuning_curve,
        select_dpp_trial_intervals,
    )
    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent
    from v1ca1.spyglass.stability import build_task_progression_from_graph

    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    validated_selection = _dpp_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        parameters_table=parameters_table,
    )
    if str(validated_selection["dpp_tuning_curve_id"]) != str(
        key["dpp_tuning_curve_id"]
    ):
        raise ValueError("DPPTuningCurve selection UUID is stale.")
    if str(key["trial_subset"]) != "all":
        raise ValueError(
            "Legacy DPP tuning-curve registration is only available for "
            "trial_subset='all'."
        )
    legacy_parameters = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    if any(parameters[name] != legacy_parameters[name] for name in parameters):
        raise ValueError(
            "Legacy DPP tuning-curve registration requires the "
            "legacy_4cm_unsmoothed parameter preset."
        )

    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(movement_firing_rate_table, movement_key)
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    context = {**key, **movement_selection}
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, context)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, context)
    position_row = _fetch1_dict(position_table, context)
    _validate_legacy_tuning_curve_inputs(
        position_row=position_row,
        movement_parameters=movement_parameters,
    )
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError("EpochIntervals must contain finite start_time < stop_time.")
    animal_name, session_date = _session_identity(session_table, context)
    loaded_units = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=context,
    )
    region = _analysis_region(
        loaded_units["registration_row"]["region_name"]
    )
    selected_units_sha256 = unit_identity_sha256(loaded_units["unit_ids"])
    if selected_units_sha256 != str(movement_result["selected_units_sha256"]):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(spike_sorting_output, {"merge_id": merge_id})
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy DPP tuning-curve registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    if movement["analysis_status"] not in {"valid", "no_units"}:
        raise ValueError(
            "A legacy DPP curve cannot represent an upstream terminal "
            f"movement status {movement['analysis_status']!r}."
        )

    trajectory_pair = get_dpp_trajectory_pair(str(key["turn_type"]))
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": movement_selection["nwb_file_name"],
                "epoch": movement_selection["epoch"],
                "trajectory_type": trajectory_type,
            },
        )
        for trajectory_type in trajectory_pair
    }
    graph_rows = {
        trajectory_type: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": movement_selection["nwb_file_name"],
                "configuration_name": trajectory_type,
            },
        )
        for trajectory_type in trajectory_pair
    }
    nwb_path = Path(
        nwbfile_table.get_abs_path(str(movement_selection["nwb_file_name"]))
    )
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
        position = (
            load_position(
                nwbfile,
                position_row,
                apply_analysis_offset=True,
            )
            if movement["analysis_status"] == "valid"
            else None
        )

    selected_intervals, _pooled_intervals = select_dpp_trial_intervals(
        trajectory_intervals,
        turn_type=str(key["turn_type"]),
        trial_subset="all",
    )
    support_by_trajectory = {
        trajectory_type: intervals.intersect(movement["movement_intervals"])
        for trajectory_type, intervals in selected_intervals.items()
    }
    n_trials_by_trajectory = {
        trajectory_type: len(np.asarray(intervals.start).reshape(-1))
        for trajectory_type, intervals in selected_intervals.items()
    }
    support_duration_s_by_trajectory = {
        trajectory_type: float(intervals.tot_length())
        for trajectory_type, intervals in support_by_trajectory.items()
    }
    common_length, graph_lengths = common_graph_length_from_inputs(
        graph_inputs,
        turn_type=str(key["turn_type"]),
    )
    feature_counts = {trajectory_type: 0 for trajectory_type in trajectory_pair}
    valid_counts = {trajectory_type: 0 for trajectory_type in trajectory_pair}
    if movement["analysis_status"] == "valid":
        for trajectory_type in trajectory_pair:
            progression, computed_length = build_task_progression_from_graph(
                position=position,
                trajectory_interval=selected_intervals[trajectory_type],
                graph_inputs=graph_inputs[trajectory_type],
                trajectory_type=trajectory_type,
            )
            if not math.isclose(
                common_length,
                computed_length,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Linearization and validated DPP graph lengths disagree."
                )
            values = np.asarray(
                progression.restrict(support_by_trajectory[trajectory_type]).d,
                dtype=float,
            ).reshape(-1)
            feature_counts[trajectory_type] = len(values)
            valid_counts[trajectory_type] = int(
                np.count_nonzero(np.isfinite(values))
            )

    unit_identity_resolver: dict[Any, dict[str, Any]] = {}
    for group_unit_id, metadata in enumerate(loaded_units["unit_metadata"]):
        sorting_unit_id = metadata.get("sorting_unit_id")
        if sorting_unit_id is None or sorting_unit_id in unit_identity_resolver:
            raise ValueError(
                "Every imported selected unit requires a unique sorting_unit_id "
                "for legacy DPP tuning-curve registration."
            )
        unit_identity_resolver[sorting_unit_id] = {
            "spikesorting_merge_id": metadata["spikesorting_merge_id"],
            "unit_id": metadata["unit_id"],
            "group_unit_id": group_unit_id,
            "sorting_unit_id": sorting_unit_id,
        }

    del artifact_root, overwrite
    source_path = Path(tuning_curve_path).resolve(strict=True)
    with xr.open_dataarray(source_path) as opened:
        legacy_curve = opened.load()
    curve = normalize_legacy_all_trial_dpp_tuning_curve(
        legacy_curve,
        unit_identity_resolver=unit_identity_resolver,
        animal_name=animal_name,
        date=session_date,
        region=region,
        epoch=str(movement_selection["epoch"]),
        turn_type=str(key["turn_type"]),
        common_graph_length_cm=common_length,
        graph_length_cm_by_trajectory=graph_lengths,
        n_trials_by_trajectory=n_trials_by_trajectory,
        support_duration_s_by_trajectory=(
            support_duration_s_by_trajectory
        ),
        n_feature_samples_by_trajectory=feature_counts,
        n_valid_position_samples_by_trajectory=valid_counts,
        bin_size_cm=parameters["place_bin_size_cm"],
        bin_count=parameters["position_bin_count"],
        sigma_bins=parameters["gaussian_smoothing_sigma_bins"],
    )
    curve.attrs.update(
        _dpp_tuning_curve_artifact_attributes(
            key,
            selected_units_sha256=selected_units_sha256,
        )
    )
    nwb_artifact = _write_dpp_tuning_curve_nwb(
        nwb_file_name=str(movement_selection["nwb_file_name"]),
        curve=curve,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    provenance = {
        "source_path": str(source_path),
        "source_sha256": _file_sha256(source_path),
        "legacy_unit_coordinate": "sorting_unit_id",
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "source_spyglass_git_commit": source_spyglass_git_commit,
        "assumed_parameters": {
            "position": {
                field_name: position_row[field_name]
                for field_name in (
                    "position_series_name",
                    "position_role",
                    "analysis_start_offset_samples",
                )
            },
            "movement": movement_parameters,
            "tuning_curve": parameters,
        },
    }
    return {
        **nwb_artifact,
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "n_outbound_trials": int(curve.attrs["n_outbound_trials"]),
        "n_inbound_trials": int(curve.attrs["n_inbound_trials"]),
        "support_duration_s": float(curve.attrs["support_duration_s"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[curve.dims[1]]),
        "analysis_status": str(curve.attrs["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
    }


def _validate_legacy_stability_schema(table: Any) -> None:
    """Require every canonical QC field in a legacy stability artifact."""
    from v1ca1.spyglass.stability import empty_stability_table

    required_columns = set(empty_stability_table().columns).difference(
        {
            "spikesorting_merge_id",
            "unit_id",
            "stable_unit_id",
            "group_unit_id",
        }
    )
    required_columns.add("unit")
    missing = sorted(required_columns.difference(table.columns))
    if missing:
        raise ValueError(
            f"Existing stability artifact is missing canonical columns {missing!r}."
        )


def _register_existing_path_specific_place_stability_row(
    *,
    key: Mapping[str, Any],
    stability_path: Path,
    overwrite: bool,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    tuning_curve_parameters_table: Any,
    movement_parameters_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    session_table: Any,
    region_sorted_spikes_group_table: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Filter and register one partition of the complete legacy artifact."""
    import pandas as pd

    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent
    from v1ca1.spyglass.stability import (
        compute_selected_stability_from_tuning_curves,
        empty_stability_table,
    )

    validated_selection = _stability_selection_row(
        key=key,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
    )
    if str(validated_selection["path_specific_place_stability_id"]) != str(
        key["path_specific_place_stability_id"]
    ):
        raise ValueError("PathSpecificPlaceStability selection UUID is stale.")
    curve_results: dict[str, dict[str, Any]] = {}
    curve_selections: dict[str, dict[str, Any]] = {}
    curves: dict[str, Any] = {}
    for subset in ("odd", "even"):
        curve_key = {
            "path_specific_place_tuning_curve_id": key[
                f"{subset}_path_specific_place_tuning_curve_id"
            ]
        }
        curve_results[subset] = _fetch1_dict(tuning_curve_table, curve_key)
        curve_selections[subset] = _fetch1_dict(
            tuning_curve_selection_table,
            curve_key,
        )
        curves[subset] = _load_path_specific_place_tuning_curve_result(
            result_row=curve_results[subset],
            tuning_curve_table=tuning_curve_table,
            selection_row=curve_selections[subset],
        )
    curve_selection = curve_selections["odd"]
    parameters = _validate_tuning_curve_parameter_row(
        _fetch1_dict(tuning_curve_parameters_table, curve_selection)
    )
    _validate_frozen_parameters(
        curve_selection,
        parameters,
        field_name="tuning_curve_parameters_sha256",
    )
    legacy_parameters = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    if any(parameters[name] != legacy_parameters[name] for name in parameters):
        raise ValueError(
            "Legacy stability registration requires odd/even curves with the "
            "legacy_4cm_unsmoothed parameter preset."
        )

    movement_key = {
        "movement_firing_rate_id": curve_selection["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    context = {**curve_selection, **movement_selection}
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, context)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    default_movement_parameters = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    for field_name in (
        "speed_threshold_cm_s",
        "speed_smoothing_sigma_s",
    ):
        if not math.isclose(
            movement_parameters[field_name],
            default_movement_parameters[field_name],
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Legacy stability registration is only valid for the regenerated "
                f"default movement parameters; {field_name} differs."
            )

    animal_name, session_date = _session_identity(session_table, context)
    loaded_units = _load_registered_region_spikes(
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        key=context,
    )
    region = _analysis_region(
        loaded_units["registration_row"]["region_name"]
    )
    selected_units_sha256 = unit_identity_sha256(loaded_units["unit_ids"])
    if selected_units_sha256 != str(
        movement_result["selected_units_sha256"]
    ):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    non_imported = [
        str(merge_id)
        for merge_id in loaded_units["merge_ids"]
        if resolve_merge_parent(spike_sorting_output, {"merge_id": merge_id})
        != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy stability registration is restricted to matching "
            f"ImportedSpikeSorting outputs; found {non_imported!r}."
        )

    stability_path = Path(stability_path)
    if not stability_path.is_file():
        raise FileNotFoundError(f"Existing stability artifact not found: {stability_path}")
    table = pd.read_parquet(stability_path)
    _validate_legacy_stability_schema(table)
    expected_partition = {
        "animal_name": animal_name,
        "date": session_date,
        "epoch": str(movement_selection["epoch"]),
        "trajectory_type": str(curve_selection["trajectory_type"]),
        "region": region,
    }
    include = np.ones(len(table), dtype=bool)
    for column, expected in expected_partition.items():
        include &= table[column].astype(str).to_numpy() == str(expected)
    selected = table.loc[include].copy().reset_index(drop=True)
    if selected.empty and loaded_units["unit_ids"]:
        raise ValueError(
            "Existing stability artifact has no rows for the selected partition."
        )
    legacy_ids = {
        str(metadata["sorting_unit_id"])
        for metadata in loaded_units["unit_metadata"]
        if metadata.get("sorting_unit_id") is not None
    }
    if len(legacy_ids) != len(loaded_units["unit_ids"]):
        raise ValueError(
            "Every imported selected unit needs a unique sorting_unit_id for "
            "legacy stability registration."
        )
    selected = selected.loc[selected["unit"].astype(str).isin(legacy_ids)].copy()
    if loaded_units["unit_ids"]:
        selected = selected.rename(columns={"unit": "unit_id"})
        selected = _attach_registered_unit_identity(
            selected,
            unit_metadata=loaded_units["unit_metadata"],
            artifact_name="stability",
        )
        curve_group_ids = pd.Series(
            np.asarray(curves["odd"].coords["group_unit_id"].values).astype(
                str
            ),
            index=np.asarray(
                curves["odd"].coords["stable_unit_id"].values
            ).astype(str),
        )
        selected["group_unit_id"] = (
            selected["stable_unit_id"].astype(str).map(curve_group_ids)
        )
        if selected["group_unit_id"].isna().any():
            raise ValueError(
                "Existing stability units do not map to the upstream tuning "
                "curve group identities."
            )
    else:
        selected = empty_stability_table()
    selected_units = set(selected["stable_unit_id"].astype(str))
    expected_units = {
        f"{unit['spikesorting_merge_id']}:{unit['unit_id']}"
        for unit in loaded_units["unit_ids"]
    }
    for subset in ("odd", "even"):
        curve_units = set(
            np.asarray(curves[subset].coords["stable_unit_id"].values).astype(str)
        )
        if curve_units != expected_units or str(
            curve_results[subset]["selected_units_sha256"]
        ) != selected_units_sha256:
            raise ValueError(
                "Odd/even tuning-curve artifacts do not contain exactly the "
                "selected SortedSpikesGroup units."
            )
    if len(selected) != len(selected_units) or selected_units != expected_units:
        raise ValueError(
            "Existing stability partition must contain exactly one row per "
            "selected SortedSpikesGroup unit."
        )
    movement_units = set(
        movement["table"].get("stable_unit_id", pd.Series(dtype=str)).astype(str)
    )
    if movement_units != expected_units:
        raise ValueError(
            "MovementFiringRate artifact does not contain exactly the selected "
            "SortedSpikesGroup units."
        )
    if expected_units:
        selected_rates = (
            selected.set_index("stable_unit_id")["firing_rate_hz"]
            .astype(float)
            .loc[sorted(expected_units)]
            .to_numpy()
        )
        movement_rates = (
            movement["table"]
            .set_index("stable_unit_id")["movement_firing_rate_hz"]
            .astype(float)
            .loc[sorted(expected_units)]
            .to_numpy()
        )
        if not np.allclose(
            selected_rates,
            movement_rates,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                "Existing stability firing_rate_hz does not match the upstream "
                "MovementFiringRate artifact."
            )
    expected_result = compute_selected_stability_from_tuning_curves(
        odd_tuning_curve=curves["odd"],
        even_tuning_curve=curves["even"],
        movement_firing_rate_table=movement["table"],
    )
    canonical_columns = list(empty_stability_table().columns)
    missing_canonical = sorted(set(canonical_columns).difference(selected.columns))
    if missing_canonical:
        raise ValueError(
            "Existing stability artifact is missing normalized columns "
            f"{missing_canonical!r}."
        )
    selected_for_validation = (
        selected.loc[:, canonical_columns]
        .sort_values("stable_unit_id", kind="stable")
        .reset_index(drop=True)
    )
    expected_for_validation = (
        expected_result["table"]
        .loc[:, canonical_columns]
        .sort_values("stable_unit_id", kind="stable")
        .reset_index(drop=True)
    )
    string_columns = (
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "group_unit_id",
        "animal_name",
        "date",
        "region",
        "epoch",
        "trajectory_type",
        "stability_status",
    )
    for column in string_columns:
        selected_for_validation[column] = selected_for_validation[column].astype(
            str
        )
        expected_for_validation[column] = expected_for_validation[column].astype(
            str
        )
    try:
        pd.testing.assert_frame_equal(
            selected_for_validation,
            expected_for_validation,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-12,
        )
    except AssertionError as exc:
        raise ValueError(
            "Existing stability artifact does not match stability recomputed "
            "from the selected odd/even tuning curves."
        ) from exc
    selected = selected_for_validation
    ordered_identity = [
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "group_unit_id",
    ]
    ordered_identity.extend(
        column for column in selected if column not in ordered_identity
    )
    selected = selected.loc[:, ordered_identity]
    n_valid_units = int(
        selected["stability_status"].astype(str).eq("valid").sum()
    )
    if not expected_units:
        analysis_status = "no_units"
    elif movement["analysis_status"] in {
        "no_valid_position",
        "no_movement",
    }:
        statuses = set(selected["stability_status"].astype(str))
        if statuses != {movement["analysis_status"]}:
            raise ValueError(
                "Existing stability terminal status does not match the upstream "
                "MovementFiringRate status."
            )
        analysis_status = movement["analysis_status"]
    else:
        analysis_status = "valid" if n_valid_units else "no_valid_units"
    del artifact_root, overwrite
    nwb_artifact = _write_path_specific_place_stability_nwb(
        nwb_file_name=str(movement_selection["nwb_file_name"]),
        table=selected,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units": len(expected_units),
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": {
            "source_path": str(stability_path.resolve(strict=True)),
            "sha256": _file_sha256(stability_path),
            "source_v1ca1_git_commit": source_v1ca1_git_commit,
            "source_spyglass_git_commit": source_spyglass_git_commit,
            "assumed_parameters": {
                "movement": movement_parameters,
                "tuning_curve": parameters,
            },
        },
    }


def _fetch_stability_result_nwb_object(
    stability_table: Any,
    key: Mapping[str, Any],
) -> Any:
    """Fetch exactly one PathSpecificPlaceStability NWB object."""
    relation = stability_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading PathSpecificPlaceStability."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "PathSpecificPlaceStability NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    if "stability" not in record:
        raise ValueError(
            "PathSpecificPlaceStability NWB fetch is missing the stability object."
        )
    return record["stability"]


def _load_path_specific_place_stability_result(
    *,
    result_row: Mapping[str, Any],
    stability_table: Any,
    expected_metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Load one stability DynamicTable and cross-check its result metadata."""
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.stability import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        stability_table_from_dynamic_table,
        stability_table_sha256,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "PathSpecificPlaceStability artifact schema version is unsupported."
        )
    stability_id = result_row["path_specific_place_stability_id"]
    fetched = _fetch_stability_result_nwb_object(
        stability_table,
        {"path_specific_place_stability_id": stability_id},
    )
    table = stability_table_from_dynamic_table(fetched)
    if stability_table_sha256(table) != str(result_row.get("stability_sha256")):
        raise ValueError(
            "PathSpecificPlaceStability result metadata disagrees with its NWB "
            "object: stability_sha256."
        )
    if len(table) != int(result_row["n_units"]):
        raise ValueError(
            "PathSpecificPlaceStability result unit count disagrees with its "
            "NWB object."
        )
    n_valid = int(table["stability_status"].astype(str).eq("valid").sum())
    if n_valid != int(result_row["n_valid_units"]):
        raise ValueError(
            "PathSpecificPlaceStability valid-unit count disagrees with its "
            "NWB object."
        )
    status = str(result_row["analysis_status"])
    if status == "no_units":
        if not table.empty:
            raise ValueError("no_units stability results require an empty table.")
    elif status in {"no_valid_position", "no_movement"}:
        observed = set(table["stability_status"].astype(str))
        if table.empty or n_valid or observed != {status}:
            raise ValueError(
                "Terminal PathSpecificPlaceStability status disagrees with its "
                "NWB object."
            )
    elif status == "valid":
        if table.empty or n_valid == 0:
            raise ValueError("valid stability results require valid unit rows.")
    elif status == "no_valid_units":
        if table.empty or n_valid:
            raise ValueError(
                "no_valid_units stability results require nonempty invalid rows."
            )
    else:
        raise ValueError(
            f"Unsupported PathSpecificPlaceStability status {status!r}."
        )
    identities = table.loc[
        :, ["spikesorting_merge_id", "unit_id"]
    ].to_dict("records")
    if unit_identity_sha256(identities) != str(
        result_row["selected_units_sha256"]
    ):
        raise ValueError(
            "PathSpecificPlaceStability unit identities disagree with its "
            "result digest."
        )
    if not table.empty:
        for field_name, expected_value in dict(expected_metadata or {}).items():
            observed = table[field_name].astype(str).unique().tolist()
            if observed != [str(expected_value)]:
                raise ValueError(
                    "PathSpecificPlaceStability NWB object does not match its "
                    f"selection: {field_name}."
                )
    return table


def _load_dpp_stability_artifact(
    *,
    result_row: Mapping[str, Any],
    stability_table: Any,
    trajectory_type: str,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> Any:
    """Load and cross-check one selected stability NWB object."""
    return _load_path_specific_place_stability_result(
        result_row=result_row,
        stability_table=stability_table,
        expected_metadata={
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "trajectory_type": trajectory_type,
        },
    )


def _load_dpp_encoding_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and revalidate shared inputs for one encoding comparison."""
    validated_selection = _dpp_encoding_selection_row(
        key=key,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        parameters_table=parameters_table,
    )
    if str(validated_selection["dpp_encoding_id"]) != str(
        key["dpp_encoding_id"]
    ):
        raise ValueError("DPPEncoding selection UUID is stale.")
    parameters = _validate_dpp_encoding_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="dpp_encoding_parameters_sha256",
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, movement_selection)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    animal_name, session_date = _session_identity(
        session_table,
        movement_selection,
    )
    region = str(region_row["region_name"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    if movement["analysis_status"] != "valid":
        raise ValueError(
            "DPPEncoding requires valid movement support."
        )
    epoch_row = _fetch1_dict(epoch_intervals_table, movement_selection)
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError(
            "EpochIntervals must contain finite start_time < stop_time."
        )

    stability_tables: dict[str, Any] = {}
    stability_rows: dict[str, dict[str, Any]] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        stability_key = {
            "path_specific_place_stability_id": key[
                f"{trajectory_type}_stability_id"
            ]
        }
        stability_row = _fetch1_dict(stability_table, stability_key)
        stability_tables[trajectory_type] = _load_dpp_stability_artifact(
            result_row=stability_row,
            stability_table=stability_table,
            trajectory_type=trajectory_type,
            animal_name=animal_name,
            date=session_date,
            region=region,
            epoch=str(movement_selection["epoch"]),
        )
        stability_rows[trajectory_type] = stability_row
    return {
        "selection": validated_selection,
        "parameters": parameters,
        "region_row": region_row,
        "movement_result": movement_result,
        "movement_selection": movement_selection,
        "movement_parameters": movement_parameters,
        "movement": movement,
        "epoch_row": epoch_row,
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
        "epoch": str(movement_selection["epoch"]),
        "epoch_time_support": (epoch_start, epoch_stop),
        "stability_tables": stability_tables,
        "stability_rows": stability_rows,
    }


def _load_dpp_encoding_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Open one source NWB and load the selected position, laps, and graphs."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    movement_selection = dict(context["movement_selection"])
    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    position_row = _fetch1_dict(position_table, movement_selection)
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    configuration_names = (*_DPP_TRAJECTORY_TYPES, _DPP_FULL_GRAPH_CONFIGURATION_NAME)
    graph_rows = {
        configuration_name: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": configuration_name,
            },
        )
        for configuration_name in configuration_names
    }
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        position = load_position(
            nwbfile,
            position_row,
            apply_analysis_offset=True,
        )
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            configuration_name: load_wtrack_graph(nwbfile, row)
            for configuration_name, row in graph_rows.items()
        }
    return {
        "position": position,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "position_row": position_row,
    }


def _load_dpp_encoding_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify the regional units selected for one comparison."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["region_row"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=context["epoch_time_support"],
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    expected_digest = str(context["region_row"]["selected_units_sha256"])
    if unit_digest != expected_digest or unit_digest != str(
        context["movement_result"]["selected_units_sha256"]
    ):
        raise ValueError(
            "DPPEncoding regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "DPPEncoding regional unit count changed after selection."
        )
    return loaded


def _load_path_progression_decoding_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and revalidate one shared-cohort cross-path decoder selection."""
    validated_selection = _path_progression_decoding_selection_row(
        key=key,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        parameters_table=parameters_table,
    )
    if str(validated_selection["path_progression_decoding_id"]) != str(
        key["path_progression_decoding_id"]
    ):
        raise ValueError(
            "PathProgressionDecoding selection UUID is stale."
        )
    parameters = _validate_path_progression_decoding_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="path_progression_decoding_parameters_sha256",
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    animal_name, session_date = _session_identity(session_table, key)
    region = str(region_row["region_name"])

    movement_sources: dict[str, dict[str, Any]] = {}
    for source_name, id_field in (
        ("target", "movement_firing_rate_id"),
        ("cohort", "cohort_movement_firing_rate_id"),
    ):
        movement_key = {"movement_firing_rate_id": key[id_field]}
        result = _fetch1_dict(movement_firing_rate_table, movement_key)
        selection = _fetch1_dict(
            movement_firing_rate_selection_table,
            movement_key,
        )
        movement_parameters = _validate_movement_parameter_row(
            _fetch1_dict(movement_parameters_table, selection)
        )
        _validate_frozen_parameters(
            selection,
            movement_parameters,
            field_name="movement_parameters_sha256",
        )
        movement = _load_movement_result_artifacts(
            result_row=result,
            movement_firing_rate_table=movement_firing_rate_table,
            parameters=movement_parameters,
            expected_metadata={
                "animal_name": animal_name,
                "date": session_date,
                "region": region,
                "epoch": str(selection["epoch"]),
            },
        )
        expected_movement_status = (
            "no_units" if int(region_row["n_units"]) == 0 else "valid"
        )
        if movement["analysis_status"] != expected_movement_status:
            raise ValueError(
                "PathProgressionDecoding movement support does not "
                "match the regional unit count."
            )
        epoch_row = _fetch1_dict(epoch_intervals_table, selection)
        epoch_start = float(epoch_row["start_time"])
        epoch_stop = float(epoch_row["stop_time"])
        if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
            epoch_stop <= epoch_start
        ):
            raise ValueError(
                "EpochIntervals must contain finite start_time < stop_time."
            )
        movement_sources[source_name] = {
            "result": result,
            "selection": selection,
            "parameters": movement_parameters,
            "movement": movement,
            "epoch_row": epoch_row,
            "epoch_time_support": (epoch_start, epoch_stop),
        }

    stability_tables: dict[str, dict[str, Any]] = {
        "target": {},
        "cohort": {},
    }
    stability_rows: dict[str, dict[str, dict[str, Any]]] = {
        "target": {},
        "cohort": {},
    }
    for source_name, prefix in (("target", ""), ("cohort", "cohort_")):
        source = movement_sources[source_name]
        source_epoch = str(source["selection"]["epoch"])
        for trajectory_type in _DPP_TRAJECTORY_TYPES:
            stability_key = {
                "path_specific_place_stability_id": key[
                    f"{prefix}{trajectory_type}_stability_id"
                ]
            }
            result_row = _fetch1_dict(stability_table, stability_key)
            stability_tables[source_name][trajectory_type] = (
                _load_dpp_stability_artifact(
                    result_row=result_row,
                    stability_table=stability_table,
                    trajectory_type=trajectory_type,
                    animal_name=animal_name,
                    date=session_date,
                    region=region,
                    epoch=source_epoch,
                )
            )
            stability_rows[source_name][trajectory_type] = result_row

    return {
        "selection": validated_selection,
        "parameters": parameters,
        "region_row": region_row,
        "movement_sources": movement_sources,
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
        "epoch": str(movement_sources["target"]["selection"]["epoch"]),
        "cohort_epoch": str(
            movement_sources["cohort"]["selection"]["epoch"]
        ),
        "epoch_time_support": movement_sources["target"][
            "epoch_time_support"
        ],
        "stability_tables": stability_tables,
        "stability_rows": stability_rows,
    }


def _load_path_progression_decoding_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load target-epoch position, laps, and four path graphs from NWB."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    movement_selection = dict(
        context["movement_sources"]["target"]["selection"]
    )
    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    position_row = _fetch1_dict(position_table, movement_selection)
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_rows = {
        trajectory_type: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        position = load_position(
            nwbfile,
            position_row,
            apply_analysis_offset=True,
        )
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
    return {
        "position": position,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "position_row": position_row,
    }


def _load_path_progression_decoding_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify regional units for one target decoding epoch."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["region_row"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=context["epoch_time_support"],
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    expected_digest = str(context["region_row"]["selected_units_sha256"])
    source_results = context["movement_sources"]
    if unit_digest != expected_digest or any(
        unit_digest
        != str(source_results[source_name]["result"]["selected_units_sha256"])
        for source_name in ("target", "cohort")
    ):
        raise ValueError(
            "PathProgressionDecoding regional units changed after "
            "selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "PathProgressionDecoding regional unit count changed "
            "after selection."
        )
    return loaded


def _load_path_specific_place_decoding_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and revalidate one within-epoch place decoder selection."""
    validated_selection = _path_specific_place_decoding_selection_row(
        key=key,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
    )
    if str(validated_selection["path_specific_place_decoding_id"]) != str(
        key["path_specific_place_decoding_id"]
    ):
        raise ValueError("PathSpecificPlaceDecoding selection UUID is stale.")
    parameters = _validate_path_specific_place_decoding_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="path_specific_place_decoding_parameters_sha256",
    )
    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, movement_selection)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    animal_name, session_date = _session_identity(
        session_table,
        movement_selection,
    )
    region = str(region_row["region_name"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, movement_selection)
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError(
            "EpochIntervals must contain finite start_time < stop_time."
        )
    return {
        "selection": validated_selection,
        "parameters": parameters,
        "region_row": region_row,
        "movement_result": movement_result,
        "movement_selection": movement_selection,
        "movement_parameters": movement_parameters,
        "movement": movement,
        "epoch_row": epoch_row,
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
        "epoch": str(movement_selection["epoch"]),
        "epoch_time_support": (epoch_start, epoch_stop),
    }


def _load_path_specific_place_decoding_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load position, four lap sets, and four path graphs from NWB."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    movement_selection = dict(context["movement_selection"])
    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    position_row = _fetch1_dict(position_table, movement_selection)
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_rows = {
        trajectory_type: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        position = load_position(
            nwbfile,
            position_row,
            apply_analysis_offset=True,
        )
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
    return {
        "position": position,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "position_row": position_row,
    }


def _load_path_specific_place_decoding_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify all regional units for one place decoder."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["region_row"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=context["epoch_time_support"],
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    if unit_digest != str(
        context["region_row"]["selected_units_sha256"]
    ) or unit_digest != str(
        context["movement_result"]["selected_units_sha256"]
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "PathSpecificPlaceDecoding regional unit count changed after "
            "selection."
        )
    return loaded


def _load_motor_encoding_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and revalidate one nine-model motor-encoding selection."""
    from v1ca1.spyglass.selection import provenance_sha256

    validated_selection = _motor_encoding_selection_row(
        key=key,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        parameters_table=parameters_table,
    )
    if str(validated_selection["motor_encoding_id"]) != str(
        key["motor_encoding_id"]
    ):
        raise ValueError("MotorEncoding selection UUID is stale.")
    parameters = _validate_motor_encoding_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="motor_encoding_parameters_sha256",
    )
    expected_model_spec_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_MODEL_SPEC)
    )
    if str(key.get("motor_encoding_model_spec_sha256", "")) != (
        expected_model_spec_sha256
    ):
        raise ValueError(
            "MotorEncoding fixed model specification changed after "
            "selection insertion. Create a new selection."
        )
    expected_output_rule_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_OUTPUT_RULE)
    )
    if str(key.get("motor_encoding_output_rule_sha256", "")) != (
        expected_output_rule_sha256
    ):
        raise ValueError(
            "MotorEncoding fixed output rule changed after selection "
            "insertion. Create a new selection."
        )

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    movement_key = {
        "movement_firing_rate_id": key["movement_firing_rate_id"]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, movement_selection)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    animal_name, session_date = _session_identity(
        session_table,
        movement_selection,
    )
    region = str(region_row["region_name"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region,
            "epoch": str(movement_selection["epoch"]),
        },
    )
    epoch_row = _fetch1_dict(epoch_intervals_table, movement_selection)
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError(
            "EpochIntervals must contain finite start_time < stop_time."
        )
    stability_tables: dict[str, Any] = {}
    stability_rows: dict[str, dict[str, Any]] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        stability_key = {
            "path_specific_place_stability_id": key[
                f"{trajectory_type}_stability_id"
            ]
        }
        stability_row = _fetch1_dict(stability_table, stability_key)
        stability_tables[trajectory_type] = _load_dpp_stability_artifact(
            result_row=stability_row,
            stability_table=stability_table,
            trajectory_type=trajectory_type,
            animal_name=animal_name,
            date=session_date,
            region=region,
            epoch=str(movement_selection["epoch"]),
        )
        stability_rows[trajectory_type] = stability_row
    return {
        "selection": validated_selection,
        "parameters": parameters,
        "region_row": region_row,
        "movement_result": movement_result,
        "movement_selection": movement_selection,
        "movement_parameters": movement_parameters,
        "movement": movement,
        "epoch_row": epoch_row,
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
        "epoch": str(movement_selection["epoch"]),
        "epoch_time_support": (epoch_start, epoch_stop),
        "stability_tables": stability_tables,
        "stability_rows": stability_rows,
    }


def _load_motor_encoding_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load two aligned positions, four lap sets, and five graphs from NWB."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    selection = dict(context["selection"])
    nwb_file_name = str(selection["nwb_file_name"])
    epoch = str(selection["epoch"])
    primary_position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": selection[
                "primary_position_series_name"
            ],
        },
    )
    orientation_reference_position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": selection[
                "orientation_reference_position_series_name"
            ],
        },
    )
    _validate_motor_position_rows(
        primary_position_row=primary_position_row,
        orientation_reference_position_row=(
            orientation_reference_position_row
        ),
    )
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": selection[
                    f"{trajectory_type}_trajectory_type"
                ],
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    configuration_names = (
        *(
            selection[f"{trajectory_type}_configuration_name"]
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        ),
        selection["full_w_configuration_name"],
    )
    graph_rows = {
        str(configuration_name): _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": configuration_name,
            },
        )
        for configuration_name in configuration_names
    }
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        primary_position = load_position(
            nwbfile,
            primary_position_row,
            apply_analysis_offset=True,
        )
        orientation_reference_position = load_position(
            nwbfile,
            orientation_reference_position_row,
            apply_analysis_offset=True,
        )
        primary_timestamps = np.asarray(primary_position.t, dtype=float)
        orientation_reference_timestamps = np.asarray(
            orientation_reference_position.t,
            dtype=float,
        )
        if not np.array_equal(
            primary_timestamps,
            orientation_reference_timestamps,
        ):
            raise ValueError(
                "MotorEncoding primary and orientation-reference "
                "position timestamps must match exactly."
            )
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            configuration_name: load_wtrack_graph(nwbfile, row)
            for configuration_name, row in graph_rows.items()
        }
    return {
        "primary_position": primary_position,
        "orientation_reference_position": orientation_reference_position,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "primary_position_row": primary_position_row,
        "orientation_reference_position_row": (
            orientation_reference_position_row
        ),
    }


def _load_motor_encoding_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify all regional units for one motor comparison."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["region_row"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=context["epoch_time_support"],
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    if unit_digest != str(
        context["region_row"]["selected_units_sha256"]
    ) or unit_digest != str(
        context["movement_result"]["selected_units_sha256"]
    ):
        raise ValueError(
            "MotorEncoding regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "MotorEncoding regional unit count changed after "
            "selection."
        )
    return loaded


def _load_dark_light_glm_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and revalidate one coupled dark/light GLM selection."""
    from v1ca1.spyglass.selection import provenance_sha256

    selection = _dark_light_glm_selection_row(
        key=key,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
    )
    if str(selection["dark_light_glm_id"]) != str(key["dark_light_glm_id"]):
        raise ValueError("DarkLightGLM selection UUID is stale.")
    parameters = _validate_dark_light_glm_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="dark_light_glm_parameters_sha256",
    )
    expected_output_rule_sha256 = provenance_sha256(
        dict(table_specs.DARK_LIGHT_GLM_OUTPUT_RULE)
    )
    if str(key.get("dark_light_glm_output_rule_sha256", "")) != (
        expected_output_rule_sha256
    ):
        raise ValueError(
            "DarkLightGLM fixed output rule changed after selection insertion. "
            "Create a new selection."
        )

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": key[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    animal_name, session_date = _session_identity(session_table, selection)
    region = str(region_row["region_name"])
    movement_results: dict[str, dict[str, Any]] = {}
    movement_selections: dict[str, dict[str, Any]] = {}
    movement_parameters: dict[str, dict[str, Any]] = {}
    movements: dict[str, dict[str, Any]] = {}
    epoch_rows: dict[str, dict[str, Any]] = {}
    epoch_time_support: dict[str, tuple[float, float]] = {}
    for condition_name in ("dark", "light"):
        movement_key = {
            "movement_firing_rate_id": selection[
                f"{condition_name}_movement_firing_rate_id"
            ]
        }
        movement_results[condition_name] = _fetch1_dict(
            movement_firing_rate_table,
            movement_key,
        )
        movement_selections[condition_name] = _fetch1_dict(
            movement_firing_rate_selection_table,
            movement_key,
        )
        movement_parameters[condition_name] = _validate_movement_parameter_row(
            _fetch1_dict(
                movement_parameters_table,
                movement_selections[condition_name],
            )
        )
        _validate_frozen_parameters(
            movement_selections[condition_name],
            movement_parameters[condition_name],
            field_name="movement_parameters_sha256",
        )
        movements[condition_name] = _load_movement_result_artifacts(
            result_row=movement_results[condition_name],
            movement_firing_rate_table=movement_firing_rate_table,
            parameters=movement_parameters[condition_name],
            expected_metadata={
                "animal_name": animal_name,
                "date": session_date,
                "region": region,
                "epoch": selection[f"{condition_name}_epoch"],
            },
        )
        epoch_rows[condition_name] = _fetch1_dict(
            epoch_intervals_table,
            {
                "nwb_file_name": selection["nwb_file_name"],
                "epoch": selection[f"{condition_name}_epoch"],
            },
        )
        epoch_start = float(epoch_rows[condition_name]["start_time"])
        epoch_stop = float(epoch_rows[condition_name]["stop_time"])
        if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
            epoch_stop <= epoch_start
        ):
            raise ValueError(
                "EpochIntervals must contain finite start_time < stop_time."
            )
        epoch_time_support[condition_name] = (epoch_start, epoch_stop)
    if movement_parameters["dark"] != movement_parameters["light"]:
        raise ValueError(
            "DarkLightGLM dark and light movement parameters changed after "
            "selection."
        )
    return {
        "selection": selection,
        "parameters": parameters,
        "region_row": region_row,
        "movement_results": movement_results,
        "movement_selections": movement_selections,
        "movement_parameters": movement_parameters["dark"],
        "movements": movements,
        "epoch_rows": epoch_rows,
        "epoch_time_support": epoch_time_support,
        "animal_name": animal_name,
        "date": session_date,
        "region": region,
    }


def _load_dark_light_glm_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load two positions, eight lap sets, and four graphs from one NWB."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    selection = dict(context["selection"])
    nwb_file_name = str(selection["nwb_file_name"])
    position_rows: dict[str, dict[str, Any]] = {}
    trajectory_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for condition_name in ("dark", "light"):
        movement_selection = context["movement_selections"][condition_name]
        position_rows[condition_name] = _fetch1_dict(
            position_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": selection[f"{condition_name}_epoch"],
                "position_series_name": movement_selection[
                    "position_series_name"
                ],
            },
        )
        if str(position_rows[condition_name].get("spatial_unit")) != "cm":
            raise ValueError("DarkLightGLM positions must use centimeters.")
        trajectory_rows[condition_name] = {
            trajectory_type: _fetch1_dict(
                trajectory_intervals_table,
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": selection[f"{condition_name}_epoch"],
                    "trajectory_type": selection[
                        f"{condition_name}_{trajectory_type}_trajectory_type"
                    ],
                },
            )
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        }
    graph_rows = {
        trajectory_type: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": selection[
                    f"{trajectory_type}_configuration_name"
                ],
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        positions = {
            condition_name: load_position(
                nwbfile,
                position_rows[condition_name],
                apply_analysis_offset=True,
            )
            for condition_name in ("dark", "light")
        }
        trajectory_intervals = {
            condition_name: {
                trajectory_type: load_interval_set(nwbfile, row)
                for trajectory_type, row in trajectory_rows[
                    condition_name
                ].items()
            }
            for condition_name in ("dark", "light")
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
    return {
        "positions": positions,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "position_rows": position_rows,
    }


def _load_dark_light_glm_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify all regional units across the selected epoch pair."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    starts = [support[0] for support in context["epoch_time_support"].values()]
    stops = [support[1] for support in context["epoch_time_support"].values()]
    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["region_row"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=(min(starts), max(stops)),
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    expected_digests = {
        str(context["region_row"]["selected_units_sha256"]),
        *(
            str(result["selected_units_sha256"])
            for result in context["movement_results"].values()
        ),
    }
    if expected_digests != {unit_digest}:
        raise ValueError(
            "DarkLightGLM regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "DarkLightGLM regional unit count changed after selection."
        )
    return loaded


def _load_swap_glm_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    dark_light_glm_table: Any,
    dark_light_glm_selection_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Load and revalidate one held-out swapped-light selection."""
    from v1ca1.spyglass.selection import provenance_sha256
    from v1ca1.spyglass.swap_glm import OUTPUT_RULE_SHA256

    snapshot = _load_swap_dark_light_snapshot(
        dark_light_glm_table=dark_light_glm_table,
        dark_light_glm_id=key["dark_light_glm_id"],
    )
    selection = _swap_glm_selection_row(
        key=key,
        dark_light_glm_table=dark_light_glm_table,
        dark_light_glm_selection_table=dark_light_glm_selection_table,
        region_sorted_spikes_group_table=(
            region_sorted_spikes_group_table
        ),
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
        dark_light_snapshot=snapshot,
    )
    if str(selection["swap_glm_id"]) != str(key["swap_glm_id"]):
        raise ValueError("SwapGLM selection UUID is stale.")
    parameters = _validate_swap_glm_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="swap_glm_parameters_sha256",
    )
    expected_output_rule_sha256 = provenance_sha256(
        dict(table_specs.SWAP_GLM_OUTPUT_RULE)
    )
    if expected_output_rule_sha256 != OUTPUT_RULE_SHA256 or str(
        key.get("swap_glm_output_rule_sha256", "")
    ) != expected_output_rule_sha256:
        raise ValueError(
            "SwapGLM fixed output rule changed after selection insertion. "
            "Create a new selection."
        )
    frozen_snapshot = {
        "dark_light_glm_sha256": snapshot["dark_light_glm_sha256"],
        "dark_light_selected_model_sha256_by_model": snapshot[
            "selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": snapshot["parameter_sha256"],
        "dark_light_output_rule_sha256": snapshot["output_rule_sha256"],
        "upstream_analysis_status": snapshot["analysis_status"],
    }
    for field_name, expected_value in frozen_snapshot.items():
        if key.get(field_name) != expected_value:
            raise ValueError(
                "DarkLightGLM artifacts changed after SwapGLM selection: "
                f"{field_name}. Create a new selection."
            )

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": selection[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    movement_key = {
        "movement_firing_rate_id": selection[
            "light_test_movement_firing_rate_id"
        ]
    }
    movement_result = _fetch1_dict(
        movement_firing_rate_table,
        movement_key,
    )
    movement_selection = _fetch1_dict(
        movement_firing_rate_selection_table,
        movement_key,
    )
    movement_parameters = _validate_movement_parameter_row(
        _fetch1_dict(movement_parameters_table, movement_selection)
    )
    _validate_frozen_parameters(
        movement_selection,
        movement_parameters,
        field_name="movement_parameters_sha256",
    )
    animal_name, session_date = _session_identity(session_table, selection)
    snapshot_metadata = snapshot["metadata"]
    if str(snapshot_metadata["animal_name"]) != animal_name or str(
        snapshot_metadata["date"]
    ) != session_date:
        raise ValueError(
            "DarkLightGLM artifact session identity disagrees with Session."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        movement_firing_rate_table=movement_firing_rate_table,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": region_row["region_name"],
            "epoch": selection["light_test_epoch"],
        },
    )
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {
            "nwb_file_name": selection["nwb_file_name"],
            "epoch": selection["light_test_epoch"],
        },
    )
    epoch_start = float(epoch_row["start_time"])
    epoch_stop = float(epoch_row["stop_time"])
    if not math.isfinite(epoch_start) or not math.isfinite(epoch_stop) or (
        epoch_stop <= epoch_start
    ):
        raise ValueError(
            "EpochIntervals must contain finite start_time < stop_time."
        )
    return {
        "selection": selection,
        "parameters": parameters,
        "dark_light_snapshot": snapshot,
        "region_row": region_row,
        "movement_result": movement_result,
        "movement_selection": movement_selection,
        "movement_parameters": movement_parameters,
        "movement": movement,
        "epoch_row": epoch_row,
        "epoch_time_support": (epoch_start, epoch_stop),
        "animal_name": animal_name,
        "date": session_date,
        "region": str(region_row["region_name"]),
    }


def _load_swap_glm_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load held-out position, four lap sets, and four path graphs."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    selection = context["selection"]
    nwb_file_name = str(selection["nwb_file_name"])
    light_test_epoch = str(selection["light_test_epoch"])
    position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": light_test_epoch,
            "position_series_name": context["movement_selection"][
                "position_series_name"
            ],
        },
    )
    if str(position_row.get("spatial_unit")) != "cm":
        raise ValueError("SwapGLM held-out position must use centimeters.")
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": light_test_epoch,
                "trajectory_type": selection[
                    f"light_test_{trajectory_type}_trajectory_type"
                ],
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_rows = {
        trajectory_type: _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": selection[
                    f"{trajectory_type}_configuration_name"
                ],
            },
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        position = load_position(
            nwbfile,
            position_row,
            apply_analysis_offset=True,
        )
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
    return {
        "position": position,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "position_row": position_row,
    }


def _load_swap_glm_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify all regional units for held-out light scoring."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["region_row"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=context["epoch_time_support"],
    )
    unit_digest = unit_identity_sha256(loaded["unit_ids"])
    expected_digests = {
        str(context["region_row"]["selected_units_sha256"]),
        str(context["movement_result"]["selected_units_sha256"]),
    }
    if expected_digests != {unit_digest}:
        raise ValueError("SwapGLM regional units changed after selection.")
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "SwapGLM regional unit count changed after selection."
        )
    return loaded


def _load_swap_tuning_curve_comparison_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    tuning_curve_parameters_table: Any,
    epoch_intervals_table: Any,
    session_table: Any,
) -> dict[str, Any]:
    """Reload and revalidate one three-epoch swap-tuning selection."""
    from v1ca1.spyglass.selection import provenance_sha256
    from v1ca1.spyglass.swap_tuning import OUTPUT_RULE_SHA256

    curve_snapshots = {
        f"{epoch_role}:{trajectory_type}": (
            _load_swap_tuning_curve_snapshot(
                tuning_curve_table=tuning_curve_table,
                tuning_curve_selection_table=tuning_curve_selection_table,
                tuning_curve_id=key[
                    f"{epoch_role}_{trajectory_type}_tuning_curve_id"
                ],
            )
        )
        for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    selection = _swap_tuning_curve_comparison_selection_row(
        key=key,
        region_sorted_spikes_group_table=(
            region_sorted_spikes_group_table
        ),
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        tuning_curve_parameters_table=tuning_curve_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        parameters_table=parameters_table,
        curve_snapshots=curve_snapshots,
    )
    if str(selection["swap_tuning_curve_comparison_id"]) != str(
        key["swap_tuning_curve_comparison_id"]
    ):
        raise ValueError(
            "SwapTuningCurveComparison selection UUID is stale."
        )
    parameters = _validate_swap_tuning_curve_comparison_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    _validate_frozen_parameters(
        selection,
        parameters,
        field_name="swap_tuning_curve_comparison_parameters_sha256",
    )
    output_rule_sha256 = provenance_sha256(
        dict(table_specs.SWAP_TUNING_CURVE_COMPARISON_OUTPUT_RULE)
    )
    if output_rule_sha256 != OUTPUT_RULE_SHA256 or str(
        selection.get("swap_tuning_curve_comparison_output_rule_sha256", "")
    ) != output_rule_sha256:
        raise ValueError(
            "SwapTuningCurveComparison fixed output rule changed after "
            "selection insertion. Create a new selection."
        )

    region_row = _fetch1_dict(
        region_sorted_spikes_group_table,
        {
            "region_sorted_spikes_group_id": selection[
                "region_sorted_spikes_group_id"
            ]
        },
    )
    animal_name, session_date = _session_identity(session_table, selection)
    movement_results: dict[str, dict[str, Any]] = {}
    movement_selections: dict[str, dict[str, Any]] = {}
    movement_parameters: dict[str, dict[str, Any]] = {}
    movement: dict[str, dict[str, Any]] = {}
    epoch_rows: dict[str, dict[str, Any]] = {}
    for epoch_role in _SWAP_TUNING_EPOCH_ROLES:
        movement_key = {
            "movement_firing_rate_id": selection[
                f"{epoch_role}_movement_firing_rate_id"
            ]
        }
        movement_results[epoch_role] = _fetch1_dict(
            movement_firing_rate_table,
            movement_key,
        )
        movement_selections[epoch_role] = _fetch1_dict(
            movement_firing_rate_selection_table,
            movement_key,
        )
        movement_parameters[epoch_role] = _validate_movement_parameter_row(
            _fetch1_dict(
                movement_parameters_table,
                movement_selections[epoch_role],
            )
        )
        _validate_frozen_parameters(
            movement_selections[epoch_role],
            movement_parameters[epoch_role],
            field_name="movement_parameters_sha256",
        )
        movement[epoch_role] = _load_movement_result_artifacts(
            result_row=movement_results[epoch_role],
            movement_firing_rate_table=movement_firing_rate_table,
            parameters=movement_parameters[epoch_role],
            expected_metadata={
                "animal_name": animal_name,
                "date": session_date,
                "region": region_row["region_name"],
                "epoch": selection[f"{epoch_role}_epoch"],
            },
        )
        epoch_rows[epoch_role] = _fetch1_dict(
            epoch_intervals_table,
            {
                "nwb_file_name": selection["nwb_file_name"],
                "epoch": selection[f"{epoch_role}_epoch"],
            },
        )
        start_time = float(epoch_rows[epoch_role]["start_time"])
        stop_time = float(epoch_rows[epoch_role]["stop_time"])
        if (
            not math.isfinite(start_time)
            or not math.isfinite(stop_time)
            or stop_time <= start_time
        ):
            raise ValueError(
                "EpochIntervals must contain finite start_time < stop_time."
            )

    reference_movement_parameters = movement_parameters["dark"]
    if any(
        parameters_row != reference_movement_parameters
        for parameters_row in movement_parameters.values()
    ):
        raise ValueError(
            "SwapTuningCurveComparison movement parameter values changed "
            "after selection."
        )
    return {
        "selection": selection,
        "parameters": parameters,
        "region_row": region_row,
        "movement_results": movement_results,
        "movement_selections": movement_selections,
        "movement_parameters": reference_movement_parameters,
        "movement": movement,
        "epoch_rows": epoch_rows,
        "curve_snapshots": curve_snapshots,
        "tuning_curves_by_role_trajectory": {
            epoch_role: {
                trajectory_type: curve_snapshots[
                    f"{epoch_role}:{trajectory_type}"
                ]["curve"]
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            }
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        "movement_firing_rate_tables": {
            epoch_role: movement[epoch_role]["table"]
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        "animal_name": animal_name,
        "date": session_date,
        "region": str(region_row["region_name"]),
        "light_test_time_support": (
            float(epoch_rows["light_test"]["start_time"]),
            float(epoch_rows["light_test"]["stop_time"]),
        ),
    }


def _load_swap_tuning_curve_comparison_nwb_inputs(
    *,
    context: Mapping[str, Any],
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load exact held-out NWB position, paths, and graph definitions."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    selection = context["selection"]
    nwb_file_name = str(selection["nwb_file_name"])
    light_test_epoch = str(selection["light_test_epoch"])
    light_test_movement = context["movement_selections"]["light_test"]
    position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": light_test_epoch,
            "position_series_name": light_test_movement[
                "position_series_name"
            ],
        },
    )
    if str(position_row.get("spatial_unit")) != "cm":
        raise ValueError(
            "SwapTuningCurveComparison held-out position must use centimeters."
        )
    trajectory_rows: dict[str, dict[str, Any]] = {}
    graph_rows: dict[str, dict[str, Any]] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        source_selection = context["curve_snapshots"][
            f"light_test:{trajectory_type}"
        ]["selection"]
        trajectory_rows[trajectory_type] = _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": light_test_epoch,
                "trajectory_type": source_selection["trajectory_type"],
            },
        )
        graph_rows[trajectory_type] = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": source_selection[
                    "configuration_name"
                ],
            },
        )
        if str(graph_rows[trajectory_type].get("coordinate_unit")) != "cm":
            raise ValueError(
                "SwapTuningCurveComparison graphs must use centimeters."
            )
    nwb_path = Path(nwbfile_table.get_abs_path(nwb_file_name))
    with pynwb.NWBHDF5IO(
        str(nwb_path),
        mode="r",
        load_namespaces=True,
    ) as io:
        nwbfile = io.read()
        position = load_position(
            nwbfile,
            position_row,
            apply_analysis_offset=True,
        )
        trajectory_intervals = {
            trajectory_type: load_interval_set(nwbfile, row)
            for trajectory_type, row in trajectory_rows.items()
        }
        graph_inputs = {
            trajectory_type: load_wtrack_graph(nwbfile, row)
            for trajectory_type, row in graph_rows.items()
        }
    return {
        "position": position,
        "position_row": position_row,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
    }


def _load_swap_tuning_curve_comparison_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, Any]:
    """Reload and verify all regional units for held-out empirical scoring."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    loaded = region_sorted_spikes_group_table.load_spikes(
        {
            "region_sorted_spikes_group_id": context["selection"][
                "region_sorted_spikes_group_id"
            ]
        },
        time_support=context["light_test_time_support"],
    )
    digest = unit_identity_sha256(loaded["unit_ids"])
    expected_digests = {
        str(context["selection"]["selected_units_sha256"]),
        str(context["region_row"]["selected_units_sha256"]),
        *(
            str(row["selected_units_sha256"])
            for row in context["movement_results"].values()
        ),
    }
    if expected_digests != {digest}:
        raise ValueError(
            "SwapTuningCurveComparison regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "SwapTuningCurveComparison regional unit count changed after "
            "selection."
        )
    return loaded


def _legacy_swap_tuning_curve_comparison_unit_identity_resolver(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Map legacy imported-sorting IDs to persistent regional identities."""
    non_imported = [
        str(member["spikesorting_merge_id"])
        for member in loaded_spikes["member_provenance"]
        if int(member["n_selected_units"]) > 0
        and str(member["merge_parent"]) != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy SwapTuningCurveComparison registration requires matching "
            "ImportedSpikeSorting units; found non-imported outputs "
            f"{non_imported!r}."
        )
    resolver: dict[str, dict[str, str]] = {}
    metadata_rows = list(loaded_spikes["unit_metadata"])
    if len(metadata_rows) != len(loaded_spikes["unit_ids"]):
        raise ValueError(
            "Regional spike metadata must contain one row per selected unit."
        )
    for group_unit_id, metadata in zip(
        loaded_spikes["ts_group"].keys(),
        metadata_rows,
        strict=True,
    ):
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in resolver:
            raise ValueError(
                "Every selected unit requires a unique sorting_unit_id for "
                "legacy SwapTuningCurveComparison registration."
            )
        resolver[resolver_key] = {
            "spikesorting_merge_id": str(metadata["spikesorting_merge_id"]),
            "unit_id": str(metadata["unit_id"]),
            "stable_unit_id": (
                f"{metadata['spikesorting_merge_id']}:{metadata['unit_id']}"
            ),
            "group_unit_id": str(group_unit_id),
        }
    return resolver


def _legacy_swap_glm_unit_identity_resolver(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Map imported-sorting IDs to persistent held-out unit identities."""
    non_imported = [
        str(member["spikesorting_merge_id"])
        for member in loaded_spikes["member_provenance"]
        if int(member["n_selected_units"]) > 0
        and str(member["merge_parent"]) != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy SwapGLM registration requires matching "
            "ImportedSpikeSorting units; found non-imported outputs "
            f"{non_imported!r}."
        )
    resolver: dict[str, dict[str, str]] = {}
    metadata_rows = list(loaded_spikes["unit_metadata"])
    if len(metadata_rows) != len(loaded_spikes["unit_ids"]):
        raise ValueError(
            "Regional spike metadata must contain one row per selected unit."
        )
    for group_unit_id, metadata in zip(
        loaded_spikes["ts_group"].keys(),
        metadata_rows,
        strict=True,
    ):
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in resolver:
            raise ValueError(
                "Every selected unit requires a unique sorting_unit_id for "
                "legacy SwapGLM registration."
            )
        resolver[resolver_key] = {
            "spikesorting_merge_id": str(metadata["spikesorting_merge_id"]),
            "unit_id": str(metadata["unit_id"]),
            "stable_unit_id": (
                f"{metadata['spikesorting_merge_id']}:{metadata['unit_id']}"
            ),
            "group_unit_id": str(group_unit_id),
        }
    return resolver


def _legacy_dark_light_unit_identity_resolver(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Map imported-sorting unit IDs to exact persistent group identities."""
    non_imported = [
        str(member["spikesorting_merge_id"])
        for member in loaded_spikes["member_provenance"]
        if int(member["n_selected_units"]) > 0
        and str(member["merge_parent"]) != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy DarkLightGLM registration requires matching "
            "ImportedSpikeSorting units; found non-imported outputs "
            f"{non_imported!r}."
        )
    resolver: dict[str, dict[str, str]] = {}
    metadata_rows = list(loaded_spikes["unit_metadata"])
    if len(metadata_rows) != len(loaded_spikes["unit_ids"]):
        raise ValueError(
            "Regional spike metadata must contain one row per selected unit."
        )
    for group_unit_id, metadata in zip(
        loaded_spikes["ts_group"].keys(),
        metadata_rows,
        strict=True,
    ):
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in resolver:
            raise ValueError(
                "Every selected unit requires a unique sorting_unit_id for "
                "legacy DarkLightGLM registration."
            )
        resolver[resolver_key] = {
            "spikesorting_merge_id": str(metadata["spikesorting_merge_id"]),
            "unit_id": str(metadata["unit_id"]),
            "stable_unit_id": (
                f"{metadata['spikesorting_merge_id']}:{metadata['unit_id']}"
            ),
            "group_unit_id": str(group_unit_id),
        }
    return resolver


def _legacy_dpp_unit_identity_resolver(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Map unique augmented-NWB sorting IDs to persistent unit identities."""
    non_imported = [
        str(member["spikesorting_merge_id"])
        for member in loaded_spikes["member_provenance"]
        if int(member["n_selected_units"]) > 0
        and str(member["merge_parent"]) != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy DPP encoding-comparison registration requires matching "
            "ImportedSpikeSorting units; found non-imported outputs "
            f"{non_imported!r}."
        )
    resolver: dict[str, dict[str, str]] = {}
    metadata_rows = list(loaded_spikes["unit_metadata"])
    if len(metadata_rows) != len(loaded_spikes["unit_ids"]):
        raise ValueError(
            "Regional spike metadata must contain one row per selected unit."
        )
    for metadata in metadata_rows:
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in resolver:
            raise ValueError(
                "Every selected unit requires a unique sorting_unit_id for "
                "legacy DPP encoding-comparison registration."
            )
        resolver[resolver_key] = {
            "spikesorting_merge_id": str(metadata["spikesorting_merge_id"]),
            "unit_id": str(metadata["unit_id"]),
        }
    return resolver


def _legacy_motor_unit_identity_resolver(
    loaded_spikes: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Map legacy imported-sorting IDs to persistent regional identities."""
    non_imported = [
        str(member["spikesorting_merge_id"])
        for member in loaded_spikes["member_provenance"]
        if int(member["n_selected_units"]) > 0
        and str(member["merge_parent"]) != "ImportedSpikeSorting"
    ]
    if non_imported:
        raise ValueError(
            "Legacy MotorEncoding registration requires matching "
            "ImportedSpikeSorting units; found non-imported outputs "
            f"{non_imported!r}."
        )
    resolver: dict[str, dict[str, str]] = {}
    metadata_rows = list(loaded_spikes["unit_metadata"])
    if len(metadata_rows) != len(loaded_spikes["unit_ids"]):
        raise ValueError(
            "Regional spike metadata must contain one row per selected unit."
        )
    for group_unit_id, metadata in enumerate(metadata_rows):
        sorting_unit_id = metadata.get("sorting_unit_id")
        resolver_key = "" if sorting_unit_id is None else str(sorting_unit_id)
        if not resolver_key or resolver_key in resolver:
            raise ValueError(
                "Every selected unit requires a unique sorting_unit_id for "
                "legacy MotorEncoding registration."
            )
        resolver[resolver_key] = {
            "spikesorting_merge_id": str(metadata["spikesorting_merge_id"]),
            "unit_id": str(metadata["unit_id"]),
            "group_unit_id": str(group_unit_id),
        }
    return resolver


def _validate_legacy_dpp_encoding_source_path(
    path: Path,
    *,
    region: str,
    epoch: str,
    parameters: Mapping[str, Any],
) -> Path:
    """Require legacy filenames to attest their encoded fit parameters."""
    from v1ca1.task_progression.encoding_comparison import (
        format_encoding_binning_token,
    )

    source = Path(path)
    binning_token = format_encoding_binning_token(
        bin_size_s=parameters["evaluation_bin_size_s"],
        place_bin_size_cm=parameters["spatial_bin_size_cm"],
    )
    expected_name = (
        f"{region}_{epoch}_cv{parameters['n_folds']}_{binning_token}_"
        "encoding_summary.parquet"
    )
    if source.name != expected_name:
        raise ValueError(
            "Legacy DPP encoding-comparison filename does not encode the "
            "selected region, epoch, fold count, and evaluation/spatial bin "
            f"sizes; expected {expected_name!r}."
        )
    return source


def _validate_dpp_encoding_artifact_link(
    *,
    table: Any,
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one canonical comparison artifact to match its result row."""
    from v1ca1.spyglass.dpp_encoding import summarize_dpp_encoding_table

    parameters = _validate_dpp_encoding_parameter_row(
        parameters_row
    )
    _validate_frozen_parameters(
        selection_row,
        parameters,
        field_name="dpp_encoding_parameters_sha256",
    )
    summary = summarize_dpp_encoding_table(table)
    for field_name in (
        "n_units_eligible",
        "n_units_valid",
        "analysis_status",
        "eligible_units_sha256",
    ):
        if str(result_row[field_name]) != str(summary[field_name]):
            raise ValueError(
                "DPPEncoding result metadata disagrees with its "
                f"artifact: {field_name}."
            )
    if int(result_row["n_units_input"]) != int(region_row["n_units"]):
        raise ValueError(
            "DPPEncoding input-unit count disagrees with its "
            "RegionSortedSpikesGroup."
        )
    if int(result_row["n_units_input"]) < int(summary["n_units_eligible"]):
        raise ValueError(
            "DPPEncoding n_units_input cannot be smaller than the "
            "eligible-unit count."
        )

    comparison_id = str(selection_row["dpp_encoding_id"])
    if not table.empty:
        artifact_ids = (
            table["dpp_encoding_id"].astype(str).unique().tolist()
        )
        if artifact_ids != [comparison_id]:
            raise ValueError(
                "DPPEncoding artifact does not match its selection "
                "UUID."
            )
        expected_metadata = {
            "animal_name": animal_name,
            "date": date,
            "region": region_row["region_name"],
            "epoch": selection_row["epoch"],
        }
        for field_name, expected_value in expected_metadata.items():
            observed = table[field_name].astype(str).unique().tolist()
            if observed != [str(expected_value)]:
                raise ValueError(
                    "DPPEncoding artifact does not match its "
                    f"selection: {field_name}."
                )
        for field_name in (
            "n_folds",
            "evaluation_bin_size_s",
            "spatial_bin_size_cm",
            "gaussian_smoothing_sigma_bins",
            "random_seed",
            "minimum_movement_firing_rate_hz",
            "minimum_stability_correlation",
        ):
            observed = table[field_name].iloc[0]
            expected = parameters[field_name]
            if isinstance(expected, Integral):
                matches = int(observed) == int(expected)
            else:
                matches = math.isclose(
                    float(observed),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            if not matches:
                raise ValueError(
                    "DPPEncoding artifact does not match its "
                    f"selected parameters: {field_name}."
                )



def _write_dpp_encoding_nwb(
    *,
    nwb_file_name: str,
    table: Any,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, validate, and register one DPPEncoding DynamicTable."""
    import pynwb

    from v1ca1.spyglass.dpp_encoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        dpp_encoding_table_from_dynamic_table,
        dpp_encoding_table_sha256,
        dpp_encoding_table_to_dynamic_table,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for DPPEncoding output."
        )
    dpp_encoding_sha256 = dpp_encoding_table_sha256(table)
    dpp_encoding_object = dpp_encoding_table_to_dynamic_table(table)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            dpp_encoding_object_id = str(
                builder.add_nwb_object(dpp_encoding_object)
            )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "DPPEncoding analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_table = dpp_encoding_table_from_dynamic_table(
                    stored_nwb.objects[dpp_encoding_object_id]
                )
                if dpp_encoding_table_sha256(stored_table) != (
                    dpp_encoding_sha256
                ):
                    raise ValueError(
                        "DPPEncoding NWB object changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("DPPEncoding analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        "dpp_encoding_object_id": dpp_encoding_object_id,
        "dpp_encoding_sha256": dpp_encoding_sha256,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_dpp_encoding_result_nwb_object(
    dpp_encoding_table: Any,
    key: Mapping[str, Any],
) -> Any:
    """Fetch exactly one DPPEncoding NWB object."""
    relation = dpp_encoding_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading DPPEncoding."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "DPPEncoding NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    if "dpp_encoding" not in record:
        raise ValueError(
            "DPPEncoding NWB fetch is missing the dpp_encoding object."
        )
    return record["dpp_encoding"]


def _load_dpp_encoding_result(
    *,
    result_row: Mapping[str, Any],
    dpp_encoding_table: Any,
) -> Any:
    """Load one DPPEncoding DynamicTable and verify its semantic hash."""
    from v1ca1.spyglass.dpp_encoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        dpp_encoding_table_from_dynamic_table,
        dpp_encoding_table_sha256,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("DPPEncoding artifact schema version is unsupported.")
    dpp_encoding_id = result_row["dpp_encoding_id"]
    fetched = _fetch_dpp_encoding_result_nwb_object(
        dpp_encoding_table,
        {"dpp_encoding_id": dpp_encoding_id},
    )
    table = dpp_encoding_table_from_dynamic_table(fetched)
    if dpp_encoding_table_sha256(table) != str(
        result_row.get("dpp_encoding_sha256")
    ):
        raise ValueError(
            "DPPEncoding result metadata disagrees with its NWB object: "
            "dpp_encoding_sha256."
        )
    return table


def _validate_path_progression_decoding_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one canonical decoding bundle to match its DataJoint rows."""
    from v1ca1.spyglass.path_progression_decoding import (
        ARTIFACT_DIRNAME,
        ELIGIBILITY_FILENAME,
        MANIFEST_FILENAME,
        METRICS_FILENAME,
        summarize_decoding_artifact_bundle,
    )

    parameters = _validate_path_progression_decoding_parameter_row(
        parameters_row
    )
    _validate_frozen_parameters(
        selection_row,
        parameters,
        field_name="path_progression_decoding_parameters_sha256",
    )
    summary = summarize_decoding_artifact_bundle(bundle)
    for field_name in (
        "n_units_input",
        "n_units_eligible",
        "n_transfer_pairs_expected",
        "n_transfer_pairs_valid",
        "n_decoded_samples",
        "analysis_status",
        "eligible_units_sha256",
    ):
        if str(result_row[field_name]) != str(summary[field_name]):
            raise ValueError(
                "PathProgressionDecoding result metadata "
                f"disagrees with its artifact: {field_name}."
            )
    if int(result_row["n_units_input"]) != int(region_row["n_units"]):
        raise ValueError(
            "PathProgressionDecoding input count disagrees with "
            "RegionSortedSpikesGroup."
        )
    from v1ca1.spyglass.selection import unit_identity_sha256

    input_units_sha256 = unit_identity_sha256(
        bundle["unit_eligibility"]
        .loc[:, ["spikesorting_merge_id", "unit_id"]]
        .to_dict("records")
    )
    if input_units_sha256 != str(region_row["selected_units_sha256"]):
        raise ValueError(
            "PathProgressionDecoding input identities disagree "
            "with RegionSortedSpikesGroup."
        )

    expected_metadata = {
        "path_progression_decoding_id": selection_row[
            "path_progression_decoding_id"
        ],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "epoch": selection_row["epoch"],
        "cohort_epoch": selection_row["cohort_epoch"],
        "parameter_name": selection_row[
            "path_progression_decoding_param_name"
        ],
        "parameter_sha256": selection_row[
            "path_progression_decoding_parameters_sha256"
        ],
        "eligibility_rule_sha256": selection_row[
            "eligibility_rule_sha256"
        ],
        "transfer_spec_sha256": selection_row["transfer_spec_sha256"],
        "decoding_output_rule_sha256": selection_row[
            "decoding_output_rule_sha256"
        ],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(summary[field_name]) != str(expected_value):
            raise ValueError(
                "PathProgressionDecoding artifact does not match "
                f"its selection: {field_name}."
            )

    manifest_path = Path(result_row["artifact_manifest_path"])
    summary_path = Path(result_row["decoding_summary_path"])
    eligibility_path = Path(result_row["unit_eligibility_path"])
    result_id = str(selection_row["path_progression_decoding_id"])
    expected_tail = (
        str(animal_name),
        str(date),
        ARTIFACT_DIRNAME,
        str(selection_row["epoch"]),
        str(region_row["region_name"]),
        result_id,
        MANIFEST_FILENAME,
    )
    if tuple(manifest_path.parts[-len(expected_tail) :]) != expected_tail:
        raise ValueError(
            "PathProgressionDecoding manifest path does not match "
            "its canonical session/epoch/region/UUID layout."
        )
    if summary_path != manifest_path.parent / METRICS_FILENAME or (
        eligibility_path != manifest_path.parent / ELIGIBILITY_FILENAME
    ):
        raise ValueError(
            "PathProgressionDecoding result paths do not describe "
            "one canonical artifact bundle."
        )
    if Path(bundle["path"]) != manifest_path.parent:
        raise ValueError(
            "Loaded decoding bundle path disagrees with the result row."
        )


def _validate_path_specific_place_decoding_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one fetched place-decoding result to match immutable rows."""
    from v1ca1.spyglass.selection import provenance_sha256

    metadata = dict(bundle["metadata"])
    expected_metadata = {
        "path_specific_place_decoding_id": selection_row[
            "path_specific_place_decoding_id"
        ],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "epoch": selection_row["epoch"],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(metadata.get(field_name)) != str(expected_value):
            raise ValueError(
                "PathSpecificPlaceDecoding artifact does not match its "
                f"selection: {field_name}."
            )
    parameters = _validate_path_specific_place_decoding_parameter_row(
        parameters_row
    )
    expected_parameter_sha256 = provenance_sha256(parameters)
    if str(bundle["parameters"]["parameter_sha256"]) != str(
        selection_row["path_specific_place_decoding_parameters_sha256"]
    ) or expected_parameter_sha256 != str(
        selection_row["path_specific_place_decoding_parameters_sha256"]
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding artifact parameter digest is stale."
        )
    if str(bundle["parameters"]["output_rule_sha256"]) != str(
        selection_row["path_specific_place_decoding_output_rule_sha256"]
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding artifact output-rule digest is stale."
        )
    for field_name in (
        "n_folds",
        "decoding_bin_size_s",
        "sliding_window_size_bins",
        "spatial_bin_size_cm",
        "random_seed",
    ):
        if str(bundle["parameters"][field_name]) != str(parameters[field_name]):
            raise ValueError(
                "PathSpecificPlaceDecoding artifact parameters disagree: "
                f"{field_name}."
            )
    expected_scalars = {
        "n_units": int(bundle["n_units"]),
        "n_folds_expected": int(bundle["n_folds_expected"]),
        "n_folds_valid": int(bundle["n_folds_valid"]),
        "n_decoded_samples": int(bundle["n_decoded_samples"]),
        "analysis_status": str(bundle["analysis_status"]),
        "selected_units_sha256": str(bundle["selected_units_sha256"]),
        "artifact_origin": str(bundle["artifact_origin"]),
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row[field_name]) != str(expected_value):
            raise ValueError(
                "PathSpecificPlaceDecoding result disagrees with its "
                f"artifact: {field_name}."
            )
    if provenance_sha256(bundle["legacy_artifact_provenance"]) != (
        provenance_sha256(result_row.get("legacy_artifact_provenance"))
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding result disagrees with its legacy "
            "artifact provenance."
        )


_MOTOR_ENCODING_NWB_OBJECT_NAMES = (
    "selected_units",
    "dataset_index",
    "coordinates",
    "nested_cv_arrays",
    "full_refit_arrays",
    "provenance",
)


def _write_motor_encoding_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one complete MotorEncoding analysis NWB."""
    import pynwb

    from v1ca1.spyglass.motor_encoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        motor_encoding_nwb_hashes,
        motor_encoding_result_from_nwb_objects,
        motor_encoding_result_to_nwb_objects,
        validate_motor_encoding_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for MotorEncoding output."
        )
    canonical = validate_motor_encoding_result(result)
    expected_hashes = motor_encoding_nwb_hashes(canonical)
    objects = motor_encoding_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in _MOTOR_ENCODING_NWB_OBJECT_NAMES:
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError("MotorEncoding NWB object IDs must be unique.")
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "MotorEncoding analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = motor_encoding_result_from_nwb_objects(
                    **{
                        name: stored_nwb.objects[
                            object_ids[f"{name}_object_id"]
                        ]
                        for name in _MOTOR_ENCODING_NWB_OBJECT_NAMES
                    }
                )
                if motor_encoding_nwb_hashes(stored) != expected_hashes:
                    raise ValueError(
                        "MotorEncoding NWB objects changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("MotorEncoding analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "n_units_input": int(canonical["n_units_input"]),
        "n_units_eligible": int(canonical["n_units_eligible"]),
        "n_units_valid": int(canonical["n_units_valid"]),
        "n_outer_folds_expected": int(canonical["n_outer_folds_expected"]),
        "n_outer_folds_valid": int(canonical["n_outer_folds_valid"]),
        "analysis_status": str(canonical["analysis_status"]),
        "selected_units_sha256": str(canonical["selected_units_sha256"]),
        **expected_hashes,
        "legacy_artifact_provenance": canonical[
            "legacy_artifact_provenance"
        ],
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_motor_encoding_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of MotorEncoding scratch tables."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading MotorEncoding."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError("MotorEncoding NWB fetch must resolve exactly one result.")
    record = dict(records[0])
    missing = sorted(set(_MOTOR_ENCODING_NWB_OBJECT_NAMES).difference(record))
    if missing:
        raise ValueError(
            f"MotorEncoding NWB fetch is missing objects {missing!r}."
        )
    return {name: record[name] for name in _MOTOR_ENCODING_NWB_OBJECT_NAMES}


def _load_motor_encoding_result(
    *,
    result_row: Mapping[str, Any],
    motor_encoding_table: Any,
) -> dict[str, Any]:
    """Load one MotorEncoding NWB result and verify semantic hashes."""
    from v1ca1.spyglass.motor_encoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        motor_encoding_nwb_hashes,
        motor_encoding_result_from_nwb_objects,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("MotorEncoding artifact schema version is unsupported.")
    if str(result_row.get("schema_version")) != RESULT_SCHEMA_VERSION:
        raise ValueError("MotorEncoding result schema version is unsupported.")
    objects = _fetch_motor_encoding_nwb_objects(
        motor_encoding_table,
        {"motor_encoding_id": result_row["motor_encoding_id"]},
    )
    bundle = motor_encoding_result_from_nwb_objects(**objects)
    for field_name, observed in motor_encoding_nwb_hashes(bundle).items():
        if str(result_row.get(field_name)) != str(observed):
            raise ValueError(
                "MotorEncoding result metadata disagrees with its NWB object: "
                f"{field_name}."
            )
    return bundle


def _validate_motor_encoding_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one motor-encoding bundle to match its immutable rows."""
    from v1ca1.spyglass.motor_encoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        motor_encoding_nwb_hashes,
        validate_motor_encoding_result,
    )
    from v1ca1.spyglass.selection import (
        provenance_sha256,
        unit_identity_sha256,
    )

    validated = validate_motor_encoding_result(bundle)
    expected_metadata = {
        "motor_encoding_id": selection_row[
            "motor_encoding_id"
        ],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "epoch": selection_row["epoch"],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated["metadata"].get(field_name)) != str(expected_value):
            raise ValueError(
                "MotorEncoding artifact does not match its "
                f"selection: {field_name}."
            )

    parameters = _validate_motor_encoding_parameter_row(
        parameters_row
    )
    expected_parameters = {
        "parameter_name": parameters[
            "motor_encoding_param_name"
        ],
        "parameter_sha256": selection_row[
            "motor_encoding_parameters_sha256"
        ],
        "model_spec_sha256": selection_row[
            "motor_encoding_model_spec_sha256"
        ],
        "output_rule_sha256": selection_row[
            "motor_encoding_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "motor_encoding_param_name"
        },
    }
    if provenance_sha256(dict(validated["parameters"])) != provenance_sha256(
        expected_parameters
    ):
        raise ValueError(
            "MotorEncoding artifact parameters disagree with its "
            "selection."
        )

    expected_scalars = {
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "n_units_input": int(validated["n_units_input"]),
        "n_units_eligible": int(validated["n_units_eligible"]),
        "n_units_valid": int(validated["n_units_valid"]),
        "n_outer_folds_expected": int(
            validated["n_outer_folds_expected"]
        ),
        "n_outer_folds_valid": int(validated["n_outer_folds_valid"]),
        "analysis_status": str(validated["analysis_status"]),
        "selected_units_sha256": str(
            validated["selected_units_sha256"]
        ),
        "artifact_origin": str(validated["artifact_origin"]),
        **motor_encoding_nwb_hashes(validated),
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row[field_name]) != str(expected_value):
            raise ValueError(
                "MotorEncoding result disagrees with its artifact: "
                f"{field_name}."
            )
    if int(result_row["n_units_input"]) != int(region_row["n_units"]):
        raise ValueError(
            "MotorEncoding input count disagrees with "
            "RegionSortedSpikesGroup."
        )
    selected_unit_digest = unit_identity_sha256(
        validated["selected_units"]
        .loc[:, ["spikesorting_merge_id", "unit_id"]]
        .to_dict("records")
    )
    if selected_unit_digest != str(region_row["selected_units_sha256"]):
        raise ValueError(
            "MotorEncoding input identities disagree with "
            "RegionSortedSpikesGroup."
        )
    for dataset_name in ("nested_cv", "full_refit"):
        dataset = validated[dataset_name]
        expected_sources = {
            "primary_position_source": selection_row[
                "primary_position_series_name"
            ],
            "orientation_reference_position_source": selection_row[
                "orientation_reference_position_series_name"
            ],
        }
        for field_name, expected_value in expected_sources.items():
            if str(dataset.attrs.get(field_name, "")) != str(expected_value):
                raise ValueError(
                    "MotorEncoding artifact position provenance "
                    f"disagrees with its selection: {field_name}."
                )
    if provenance_sha256(validated["legacy_artifact_provenance"]) != (
        provenance_sha256(result_row.get("legacy_artifact_provenance"))
    ):
        raise ValueError(
            "MotorEncoding result disagrees with its legacy provenance."
        )


_DARK_LIGHT_GLM_NWB_OBJECT_NAMES = (
    "selected_units",
    "dataset_index",
    "axes",
    "candidate_results",
    "selected_results",
    "selection_summary",
    "provenance",
)


def _write_dark_light_glm_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one complete DarkLightGLM analysis NWB."""
    import pynwb

    from v1ca1.spyglass.dark_light_glm import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        dark_light_glm_nwb_hashes,
        dark_light_glm_result_from_nwb_objects,
        dark_light_glm_result_to_nwb_objects,
        dark_light_glm_selected_model_sha256s,
        validate_dark_light_glm_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for DarkLightGLM output."
        )
    canonical = validate_dark_light_glm_result(result)
    expected_hashes = dark_light_glm_nwb_hashes(canonical)
    selected_model_hashes = dark_light_glm_selected_model_sha256s(canonical)
    objects = dark_light_glm_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in _DARK_LIGHT_GLM_NWB_OBJECT_NAMES:
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError("DarkLightGLM NWB object IDs must be unique.")
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "DarkLightGLM analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = dark_light_glm_result_from_nwb_objects(
                    **{
                        name: stored_nwb.objects[
                            object_ids[f"{name}_object_id"]
                        ]
                        for name in _DARK_LIGHT_GLM_NWB_OBJECT_NAMES
                    }
                )
                if dark_light_glm_nwb_hashes(stored) != expected_hashes:
                    raise ValueError(
                        "DarkLightGLM NWB objects changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("DarkLightGLM analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": str(canonical["parameters"]["schema_version"]),
        "n_units": int(canonical["n_units"]),
        "n_candidates": int(canonical["n_candidates"]),
        "n_selected_models": int(canonical["n_selected_models"]),
        "analysis_status": str(canonical["analysis_status"]),
        "selected_units_sha256": str(canonical["selected_units_sha256"]),
        **expected_hashes,
        "selected_model_sha256_by_model": selected_model_hashes,
        "legacy_artifact_provenance": canonical[
            "legacy_artifact_provenance"
        ],
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_dark_light_glm_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of DarkLightGLM scratch objects."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading DarkLightGLM."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "DarkLightGLM NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    missing = sorted(set(_DARK_LIGHT_GLM_NWB_OBJECT_NAMES).difference(record))
    if missing:
        raise ValueError(
            f"DarkLightGLM NWB fetch is missing objects {missing!r}."
        )
    return {
        name: record[name] for name in _DARK_LIGHT_GLM_NWB_OBJECT_NAMES
    }


def _load_dark_light_glm_result(
    *,
    result_row: Mapping[str, Any],
    dark_light_glm_table: Any,
) -> dict[str, Any]:
    """Load a DarkLightGLM NWB result and verify semantic hashes."""
    from v1ca1.spyglass.dark_light_glm import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        dark_light_glm_nwb_hashes,
        dark_light_glm_result_from_nwb_objects,
        dark_light_glm_selected_model_sha256s,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("DarkLightGLM artifact schema version is unsupported.")
    objects = _fetch_dark_light_glm_nwb_objects(
        dark_light_glm_table,
        {"dark_light_glm_id": result_row["dark_light_glm_id"]},
    )
    bundle = dark_light_glm_result_from_nwb_objects(**objects)
    for field_name, observed in dark_light_glm_nwb_hashes(bundle).items():
        if str(result_row.get(field_name)) != str(observed):
            raise ValueError(
                "DarkLightGLM result metadata disagrees with its NWB object: "
                f"{field_name}."
            )
    if dict(result_row.get("selected_model_sha256_by_model") or {}) != (
        dark_light_glm_selected_model_sha256s(bundle)
    ):
        raise ValueError(
            "DarkLightGLM selected-model hashes disagree with its NWB objects."
        )
    return bundle


def _validate_dark_light_glm_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one dark/light artifact bundle to match its immutable rows."""
    from v1ca1.spyglass.dark_light_glm import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        SCHEMA_VERSION_BY_MODE,
        dark_light_glm_nwb_hashes,
        dark_light_glm_selected_model_sha256s,
        validate_dark_light_glm_result,
    )
    from v1ca1.spyglass.selection import (
        provenance_sha256,
        unit_identity_sha256,
    )

    validated = validate_dark_light_glm_result(bundle)
    expected_metadata = {
        "dark_light_glm_id": selection_row["dark_light_glm_id"],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "light_epoch": selection_row["light_epoch"],
        "dark_epoch": selection_row["dark_epoch"],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated["metadata"].get(field_name)) != str(expected_value):
            raise ValueError(
                "DarkLightGLM artifact does not match its selection: "
                f"{field_name}."
            )
    parameters = _validate_dark_light_glm_parameter_row(parameters_row)
    expected_parameters = {
        "schema_version": SCHEMA_VERSION_BY_MODE[
            parameters["basis_candidate_mode"]
        ],
        "parameter_name": parameters["dark_light_glm_param_name"],
        "parameter_sha256": selection_row[
            "dark_light_glm_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "dark_light_glm_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "dark_light_glm_param_name"
        },
    }
    if provenance_sha256(dict(validated["parameters"])) != provenance_sha256(
        expected_parameters
    ):
        raise ValueError(
            "DarkLightGLM artifact parameters disagree with its selection."
        )
    expected_scalars = {
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": str(validated["parameters"]["schema_version"]),
        "n_units": int(validated["n_units"]),
        "n_candidates": int(validated["n_candidates"]),
        "n_selected_models": int(validated["n_selected_models"]),
        "analysis_status": str(validated["analysis_status"]),
        "selected_units_sha256": str(validated["selected_units_sha256"]),
        "artifact_origin": str(validated["artifact_origin"]),
        **dark_light_glm_nwb_hashes(validated),
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row[field_name]) != str(expected_value):
            raise ValueError(
                "DarkLightGLM result disagrees with its artifact: "
                f"{field_name}."
            )
    selected_units = validated["selected_units"]
    selected_digest = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    if selected_digest != str(result_row["selected_units_sha256"]):
        raise ValueError(
            "DarkLightGLM selected-unit identities disagree with their digest."
        )
    if int(result_row["n_units"]) > int(region_row["n_units"]):
        raise ValueError(
            "DarkLightGLM selected unit count exceeds RegionSortedSpikesGroup."
        )

    if dict(result_row.get("selected_model_sha256_by_model") or {}) != (
        dark_light_glm_selected_model_sha256s(validated)
    ):
        raise ValueError(
            "DarkLightGLM result selected-model hashes are stale."
        )


def _validate_swap_glm_upstream_link(
    upstream: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Require exact frozen DarkLight provenance for one swap result."""
    expected = {
        "dark_light_glm_id": selection["dark_light_glm_id"],
        "dark_light_glm_sha256": selection[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": selection[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": selection[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": selection[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": selection["upstream_analysis_status"],
    }
    for field_name, expected_value in expected.items():
        observed = upstream.get(field_name)
        if isinstance(expected_value, Mapping):
            matches = dict(observed or {}) == dict(expected_value)
        else:
            matches = str(observed) == str(expected_value)
        if not matches:
            raise ValueError(
                "SwapGLM DarkLight provenance changed after selection: "
                f"{field_name}. Create a new selection."
            )


_SWAP_GLM_NWB_OBJECT_NAMES = (
    "selected_units",
    "model_metadata",
    "axes",
    "trajectory_metadata",
    "model_results",
    "observed_response",
    "provenance",
)


def _write_swap_glm_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one complete SwapGLM analysis NWB."""
    import pynwb

    from v1ca1.spyglass.swap_glm import (
        BUNDLE_SCHEMA_VERSION,
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        swap_glm_nwb_hashes,
        swap_glm_result_from_nwb_objects,
        swap_glm_result_to_nwb_objects,
        validate_swap_glm_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError("analysis_nwbfile_table is required for SwapGLM output.")
    canonical = validate_swap_glm_result(result)
    if str(canonical["dataset"].attrs.get("schema_version", "")) != (
        RESULT_SCHEMA_VERSION
    ):
        raise ValueError("SwapGLM analysis NWB requires the current result schema.")
    expected_hashes = swap_glm_nwb_hashes(canonical)
    objects = swap_glm_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in _SWAP_GLM_NWB_OBJECT_NAMES:
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError("SwapGLM NWB object IDs must be unique.")
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "SwapGLM analysis NWB failed PyNWB validation: "
                    f"{validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = swap_glm_result_from_nwb_objects(
                    **{
                        name: stored_nwb.objects[
                            object_ids[f"{name}_object_id"]
                        ]
                        for name in _SWAP_GLM_NWB_OBJECT_NAMES
                    }
                )
                if swap_glm_nwb_hashes(stored) != expected_hashes:
                    raise ValueError(
                        "SwapGLM NWB objects changed during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("SwapGLM analysis NWB was not created.")
    upstream = canonical["upstream_provenance"]
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_units": int(canonical["n_units"]),
        "n_valid_units": int(canonical["n_valid_units"]),
        "analysis_status": str(canonical["analysis_status"]),
        "selected_units_sha256": str(canonical["selected_units_sha256"]),
        **expected_hashes,
        "dark_light_glm_sha256": str(
            upstream["dark_light_glm_sha256"]
        ),
        "dark_light_selected_model_sha256_by_model": dict(
            upstream["dark_light_selected_model_sha256_by_model"]
        ),
        "dark_light_parameter_sha256": str(
            upstream["dark_light_parameter_sha256"]
        ),
        "dark_light_output_rule_sha256": str(
            upstream["dark_light_output_rule_sha256"]
        ),
        "upstream_analysis_status": str(upstream["upstream_analysis_status"]),
        "legacy_artifact_provenance": canonical[
            "legacy_artifact_provenance"
        ],
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_swap_glm_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of SwapGLM NWB scratch objects."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading SwapGLM."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError("SwapGLM NWB fetch must resolve exactly one result.")
    record = dict(records[0])
    missing = sorted(set(_SWAP_GLM_NWB_OBJECT_NAMES).difference(record))
    if missing:
        raise ValueError(
            f"SwapGLM NWB fetch is missing objects {missing!r}."
        )
    return {name: record[name] for name in _SWAP_GLM_NWB_OBJECT_NAMES}


def _load_swap_glm_result(
    *,
    result_row: Mapping[str, Any],
    swap_glm_table: Any,
) -> dict[str, Any]:
    """Load a SwapGLM NWB result and verify all semantic object hashes."""
    from v1ca1.spyglass.swap_glm import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        swap_glm_nwb_hashes,
        swap_glm_result_from_nwb_objects,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("SwapGLM artifact schema version is unsupported.")
    objects = _fetch_swap_glm_nwb_objects(
        swap_glm_table,
        {"swap_glm_id": result_row["swap_glm_id"]},
    )
    bundle = swap_glm_result_from_nwb_objects(**objects)
    observed_hashes = swap_glm_nwb_hashes(bundle)
    for field_name, observed in observed_hashes.items():
        if str(result_row.get(field_name)) != str(observed):
            raise ValueError(
                "SwapGLM result metadata disagrees with its NWB object: "
                f"{field_name}."
            )
    return bundle


def _validate_swap_glm_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one held-out swap bundle to match its immutable rows."""
    from v1ca1.spyglass.selection import provenance_sha256
    from v1ca1.spyglass.swap_glm import (
        BUNDLE_SCHEMA_VERSION,
        NWB_ARTIFACT_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        swap_glm_nwb_hashes,
        validate_swap_glm_result,
    )

    validated = validate_swap_glm_result(bundle)
    expected_metadata = {
        "swap_glm_id": selection_row["swap_glm_id"],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "dark_epoch": selection_row["dark_epoch"],
        "light_train_epoch": selection_row["light_train_epoch"],
        "light_test_epoch": selection_row["light_test_epoch"],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated["metadata"].get(field_name)) != str(expected_value):
            raise ValueError(
                "SwapGLM artifact does not match its selection: "
                f"{field_name}."
            )
    parameters = _validate_swap_glm_parameter_row(parameters_row)
    expected_parameters = {
        "parameter_name": parameters["swap_glm_param_name"],
        "parameter_sha256": selection_row[
            "swap_glm_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "swap_glm_output_rule_sha256"
        ],
        "swap_light_offset": parameters["swap_light_offset"],
        "observed_spatial_bin_size_cm": parameters[
            "observed_spatial_bin_size_cm"
        ],
    }
    if provenance_sha256(dict(validated["parameters"])) != provenance_sha256(
        expected_parameters
    ):
        raise ValueError("SwapGLM artifact parameters disagree with selection.")
    _validate_swap_glm_upstream_link(
        validated["upstream_provenance"],
        selection_row,
    )
    expected_upstream = {
        field_name: validated["upstream_provenance"][field_name]
        for field_name in (
            "dark_light_glm_id",
            "dark_light_glm_sha256",
            "dark_light_selected_model_sha256_by_model",
            "dark_light_parameter_sha256",
            "dark_light_output_rule_sha256",
            "upstream_analysis_status",
        )
    }
    expected_scalars = {
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_units": validated["n_units"],
        "n_valid_units": validated["n_valid_units"],
        "analysis_status": validated["analysis_status"],
        "selected_units_sha256": validated["selected_units_sha256"],
        "dark_light_glm_sha256": expected_upstream[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": expected_upstream[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": expected_upstream[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": expected_upstream[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": expected_upstream[
            "upstream_analysis_status"
        ],
        "artifact_origin": validated["artifact_origin"],
        **swap_glm_nwb_hashes(validated),
    }
    for field_name, expected_value in expected_scalars.items():
        if result_row.get(field_name) != expected_value and str(
            result_row.get(field_name)
        ) != str(expected_value):
            raise ValueError(
                "SwapGLM result row disagrees with its artifact: "
                f"{field_name}."
            )
    if int(result_row["n_units"]) > int(region_row["n_units"]):
        raise ValueError(
            "SwapGLM selected unit count exceeds RegionSortedSpikesGroup."
        )

    if result_row.get("legacy_artifact_provenance") != validated.get(
        "legacy_artifact_provenance"
    ):
        raise ValueError(
            "SwapGLM result-row legacy provenance differs from its NWB object."
        )


def _validate_swap_tuning_curve_comparison_upstream_link(
    upstream: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Require exact frozen upstream provenance for one empirical swap."""
    expected = {
        "selected_units_sha256": selection["selected_units_sha256"],
        "source_tuning_curve_sha256_by_role_trajectory": selection[
            "source_tuning_curve_sha256_by_role_trajectory"
        ],
        "source_tuning_parameters_sha256_by_role_trajectory": selection[
            "source_tuning_parameters_sha256_by_role_trajectory"
        ],
        "source_tuning_curve_ids_by_role_trajectory": {
            epoch_role: {
                trajectory_type: str(
                    selection[
                        f"{epoch_role}_{trajectory_type}_tuning_curve_id"
                    ]
                )
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            }
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        "movement_firing_rate_table_sha256_by_role": selection[
            "movement_firing_rate_table_sha256_by_role"
        ],
        "movement_firing_rate_ids_by_role": {
            epoch_role: str(
                selection[f"{epoch_role}_movement_firing_rate_id"]
            )
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        "movement_intervals_sha256_by_role": selection[
            "movement_intervals_sha256_by_role"
        ],
        "position_offset_samples": int(selection["position_offset_samples"]),
        "speed_threshold_cm_s": float(selection["speed_threshold_cm_s"]),
    }
    for field_name, expected_value in expected.items():
        observed = upstream.get(field_name)
        if isinstance(expected_value, Mapping):
            matches = dict(observed or {}) == dict(expected_value)
        elif isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(observed),
                    expected_value,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = str(observed) == str(expected_value)
        if not matches:
            raise ValueError(
                "SwapTuningCurveComparison upstream provenance changed after "
                f"selection: {field_name}. Create a new selection."
            )


def _write_swap_tuning_curve_comparison_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one empirical swap-tuning analysis NWB."""
    import pynwb

    from v1ca1.spyglass.swap_tuning import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        swap_tuning_curve_comparison_nwb_hashes,
        swap_tuning_curve_comparison_result_from_nwb_objects,
        swap_tuning_curve_comparison_result_to_nwb_objects,
        validate_swap_tuning_curve_comparison_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for "
            "SwapTuningCurveComparison output."
        )
    canonical = validate_swap_tuning_curve_comparison_result(result)
    expected_hashes = swap_tuning_curve_comparison_nwb_hashes(canonical)
    objects = swap_tuning_curve_comparison_result_to_nwb_objects(canonical)
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name in (
                "selected_units",
                "score_summary",
                "source_profiles",
                "model_profiles",
                "geometry",
                "provenance",
            ):
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(objects[name])
                )
            if len(set(object_ids.values())) != len(object_ids):
                raise ValueError(
                    "SwapTuningCurveComparison NWB object IDs must be unique."
                )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "SwapTuningCurveComparison analysis NWB failed PyNWB "
                    f"validation: {validation_errors!r}."
                )
            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored = swap_tuning_curve_comparison_result_from_nwb_objects(
                    **{
                        name: stored_nwb.objects[
                            object_ids[f"{name}_object_id"]
                        ]
                        for name in (
                            "selected_units",
                            "score_summary",
                            "source_profiles",
                            "model_profiles",
                            "geometry",
                            "provenance",
                        )
                    }
                )
                if (
                    swap_tuning_curve_comparison_nwb_hashes(stored)
                    != expected_hashes
                ):
                    raise ValueError(
                        "SwapTuningCurveComparison NWB objects changed during "
                        "write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise
    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError(
            "SwapTuningCurveComparison analysis NWB was not created."
        )
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "n_source_units": int(canonical["n_source_units"]),
        "n_units": int(canonical["n_units"]),
        "n_valid_units": int(canonical["n_valid_units"]),
        "analysis_status": str(canonical["analysis_status"]),
        "selected_units_sha256": str(canonical["selected_units_sha256"]),
        "legacy_artifact_provenance": canonical[
            "legacy_artifact_provenance"
        ],
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_swap_tuning_curve_comparison_nwb_objects(
    result_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of empirical swap-tuning NWB objects."""
    relation = result_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading SwapTuningCurveComparison."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "SwapTuningCurveComparison NWB fetch must resolve exactly one result."
        )
    record = dict(records[0])
    expected = {
        "selected_units",
        "score_summary",
        "source_profiles",
        "model_profiles",
        "geometry",
        "provenance",
    }
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(
            "SwapTuningCurveComparison NWB fetch is missing objects "
            f"{missing!r}."
        )
    return {name: record[name] for name in expected}


def _validate_swap_tuning_curve_comparison_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one empirical swap bundle to match immutable table rows."""
    from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256
    from v1ca1.spyglass.swap_tuning import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        swap_tuning_curve_comparison_nwb_hashes,
        validate_swap_tuning_curve_comparison_result,
    )

    validated = validate_swap_tuning_curve_comparison_result(bundle)
    expected_metadata = {
        "swap_tuning_curve_comparison_id": selection_row[
            "swap_tuning_curve_comparison_id"
        ],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "dark_epoch": selection_row["dark_epoch"],
        "light_train_epoch": selection_row["light_train_epoch"],
        "light_test_epoch": selection_row["light_test_epoch"],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated["metadata"].get(field_name)) != str(expected_value):
            raise ValueError(
                "SwapTuningCurveComparison artifact does not match its "
                f"selection: {field_name}."
            )
    parameters = _validate_swap_tuning_curve_comparison_parameter_row(
        parameters_row
    )
    expected_parameters = {
        "parameter_name": parameters[
            "swap_tuning_curve_comparison_param_name"
        ],
        "parameter_sha256": selection_row[
            "swap_tuning_curve_comparison_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "swap_tuning_curve_comparison_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "swap_tuning_curve_comparison_param_name"
        },
    }
    if provenance_sha256(dict(validated["parameters"])) != provenance_sha256(
        expected_parameters
    ):
        raise ValueError(
            "SwapTuningCurveComparison artifact parameters disagree with "
            "selection."
        )
    _validate_swap_tuning_curve_comparison_upstream_link(
        validated["upstream_provenance"],
        selection_row,
    )
    expected_scalars = {
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "n_source_units": int(validated["n_source_units"]),
        "n_units": int(validated["n_units"]),
        "n_valid_units": int(validated["n_valid_units"]),
        "analysis_status": str(validated["analysis_status"]),
        "selected_units_sha256": str(validated["selected_units_sha256"]),
        "artifact_origin": str(validated["artifact_origin"]),
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row.get(field_name)) != str(expected_value):
            raise ValueError(
                "SwapTuningCurveComparison result row disagrees with its "
                f"artifact: {field_name}."
            )
    if result_row.get("legacy_artifact_provenance") != validated.get(
        "legacy_artifact_provenance"
    ):
        raise ValueError(
            "SwapTuningCurveComparison result-row legacy provenance differs "
            "from its artifact."
        )
    if int(result_row["n_source_units"]) != int(region_row["n_units"]):
        raise ValueError(
            "SwapTuningCurveComparison source count disagrees with "
            "RegionSortedSpikesGroup."
        )
    if str(result_row["selected_units_sha256"]) != str(
        selection_row["selected_units_sha256"]
    ) or str(result_row["selected_units_sha256"]) != str(
        region_row["selected_units_sha256"]
    ):
        raise ValueError(
            "SwapTuningCurveComparison source unit digest disagrees with "
            "its selection."
        )
    selected_units = validated["selected_units"]
    selected_digest = unit_identity_sha256(
        selected_units.loc[:, ["spikesorting_merge_id", "unit_id"]].to_dict(
            "records"
        )
    )
    if selected_digest != str(result_row["selected_units_sha256"]):
        raise ValueError(
            "SwapTuningCurveComparison selected-unit identities disagree "
            "with their digest."
        )
    for field_name, observed in swap_tuning_curve_comparison_nwb_hashes(
        validated
    ).items():
        if str(result_row.get(field_name)) != observed:
            raise ValueError(
                "SwapTuningCurveComparison result metadata disagrees with its "
                f"NWB objects: {field_name}."
            )


def _load_swap_tuning_curve_comparison_result(
    *,
    result_row: Mapping[str, Any],
    result_table: Any,
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Fetch, reconstruct, and validate one empirical swap-tuning result."""
    from v1ca1.spyglass.swap_tuning import (
        swap_tuning_curve_comparison_result_from_nwb_objects,
    )

    objects = _fetch_swap_tuning_curve_comparison_nwb_objects(
        result_table,
        {
            "swap_tuning_curve_comparison_id": result_row[
                "swap_tuning_curve_comparison_id"
            ]
        },
    )
    result = swap_tuning_curve_comparison_result_from_nwb_objects(**objects)
    _validate_swap_tuning_curve_comparison_artifact_link(
        bundle=result,
        result_row=result_row,
        selection_row=selection_row,
        parameters_row=parameters_row,
        region_row=region_row,
        animal_name=animal_name,
        date=date,
    )
    return result


def _make_dpp_encoding_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    epoch_intervals_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and write one strict four-model encoding comparison."""
    from v1ca1.spyglass.dpp_encoding import compute_selected_dpp_encoding

    context = _load_dpp_encoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        session_table=session_table,
    )
    loaded_spikes = _load_dpp_encoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_dpp_encoding_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    parameters = context["parameters"]
    result = compute_selected_dpp_encoding(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=nwb_inputs["position"],
        trajectory_intervals_by_type=nwb_inputs["trajectory_intervals"],
        graph_inputs_by_configuration=nwb_inputs["graph_inputs"],
        movement_intervals=context["movement"]["movement_intervals"],
        movement_firing_rate_table=context["movement"]["table"],
        stability_tables_by_trajectory=context["stability_tables"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
        dpp_encoding_id=key["dpp_encoding_id"],
        n_folds=parameters["n_folds"],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        random_seed=parameters["random_seed"],
    )
    del artifact_root
    nwb_artifact = _write_dpp_encoding_nwb(
        nwb_file_name=str(context["movement_selection"]["nwb_file_name"]),
        table=result["table"],
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units_input": int(context["region_row"]["n_units"]),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_units_valid": int(result["n_units_valid"]),
        "analysis_status": str(result["analysis_status"]),
        "eligible_units_sha256": str(result["eligible_units_sha256"]),
        "legacy_artifact_provenance": None,
    }


def _register_existing_dpp_encoding_row(
    *,
    key: Mapping[str, Any],
    dpp_encoding_path: Path,
    overwrite: bool,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Normalize and register one exact-coverage legacy comparison."""
    import pandas as pd

    from v1ca1.spyglass.dpp_encoding import (
        normalize_legacy_dpp_encoding_table,
        summarize_dpp_encoding_table,
    )

    if overwrite:
        raise ValueError(
            "Registered DPPEncoding artifacts are immutable."
        )
    context = _load_dpp_encoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        session_table=session_table,
    )
    loaded_spikes = _load_dpp_encoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    resolver = _legacy_dpp_unit_identity_resolver(loaded_spikes)
    parameters = context["parameters"]
    source_path = _validate_legacy_dpp_encoding_source_path(
        Path(dpp_encoding_path),
        region=context["region"],
        epoch=context["epoch"],
        parameters=parameters,
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Legacy encoding artifact not found: {source_path}"
        )
    table = normalize_legacy_dpp_encoding_table(
        pd.read_parquet(source_path),
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        movement_firing_rate_table=context["movement"]["table"],
        stability_tables_by_trajectory=context["stability_tables"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
        unit_identity_resolver=resolver,
        dpp_encoding_id=key["dpp_encoding_id"],
        n_folds=parameters["n_folds"],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        random_seed=parameters["random_seed"],
    )
    summary = summarize_dpp_encoding_table(table)
    provenance = {
        "source_path": str(source_path.resolve(strict=True)),
        "source_sha256": _file_sha256(source_path),
        "source_v1ca1_git_commit": source_v1ca1_git_commit,
        "legacy_log_likelihood_units": "nats_per_spike",
        "canonical_log_likelihood_units": "total_nats",
        "eligible_unit_set_validated": True,
    }
    provenance["source_spyglass_git_commit"] = source_spyglass_git_commit
    provenance["assumed_parameters"] = dict(parameters)
    provenance["source_parameter_validation"] = {
        "verified_from_filename": [
            "n_folds",
            "evaluation_bin_size_s",
            "spatial_bin_size_cm",
        ],
        "recomputed_from_upstream": [
            "minimum_movement_firing_rate_hz",
            "minimum_stability_correlation",
        ],
        "caller_attested_not_encoded_in_legacy_artifact": [
            "gaussian_smoothing_sigma_bins",
            "random_seed",
        ],
    }
    provenance["source_fold_qc_validation"] = (
        "not_reconstructable_from_legacy_summary"
    )
    del artifact_root, overwrite
    nwb_artifact = _write_dpp_encoding_nwb(
        nwb_file_name=str(context["movement_selection"]["nwb_file_name"]),
        table=table,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    return {
        **nwb_artifact,
        "n_units_input": int(context["region_row"]["n_units"]),
        "n_units_eligible": int(summary["n_units_eligible"]),
        "n_units_valid": int(summary["n_units_valid"]),
        "analysis_status": str(summary["analysis_status"]),
        "eligible_units_sha256": str(summary["eligible_units_sha256"]),
        "legacy_artifact_provenance": provenance,
    }


def _make_path_progression_decoding_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one shared-cohort cross-path decoding NWB."""
    from v1ca1.spyglass.path_progression_decoding import (
        compute_path_progression_decoding,
    )

    context = _load_path_progression_decoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        session_table=session_table,
    )
    loaded_spikes = _load_path_progression_decoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_path_progression_decoding_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    parameters = context["parameters"]
    result = compute_path_progression_decoding(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        cohort_epoch=context["cohort_epoch"],
        path_progression_decoding_id=key[
            "path_progression_decoding_id"
        ],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        target_movement_firing_rate_table=context["movement_sources"][
            "target"
        ]["movement"]["table"],
        cohort_movement_firing_rate_table=context["movement_sources"][
            "cohort"
        ]["movement"]["table"],
        target_stability_tables_by_trajectory=context["stability_tables"][
            "target"
        ],
        cohort_stability_tables_by_trajectory=context["stability_tables"][
            "cohort"
        ],
        position=nwb_inputs["position"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        graph_inputs=nwb_inputs["graph_inputs"],
        movement_interval=context["movement_sources"]["target"][
            "movement"
        ]["movement_intervals"],
        decoding_bin_size_s=parameters["decoding_bin_size_s"],
        sliding_window_size_bins=parameters["sliding_window_size_bins"],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
        parameter_name=parameters["path_progression_decoding_param_name"],
        parameter_sha256=key[
            "path_progression_decoding_parameters_sha256"
        ],
        eligibility_rule_sha256=key["eligibility_rule_sha256"],
        transfer_spec_sha256=key["transfer_spec_sha256"],
        decoding_output_rule_sha256=key[
            "decoding_output_rule_sha256"
        ],
    )
    del artifact_root
    return _write_path_progression_decoding_nwb(
        nwb_file_name=str(
            context["movement_sources"]["target"]["selection"][
                "nwb_file_name"
            ]
        ),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _path_progression_decoding_hashes(
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Return storage-independent hashes for one progression decoder."""
    from v1ca1.spyglass.path_progression_decoding import (
        binned_error_sha256,
        build_transfer_index_table,
        decoding_provenance_sha256,
        decoding_summary_sha256,
        selected_units_table_sha256,
        transfer_index_sha256,
        unit_eligibility_sha256,
    )

    transfer_index = build_transfer_index_table(result)
    return {
        "unit_eligibility_sha256": unit_eligibility_sha256(
            result["unit_eligibility"]
        ),
        "selected_units_table_sha256": selected_units_table_sha256(
            result["selected_units"]
        ),
        "decoding_summary_sha256": decoding_summary_sha256(
            result["cross_path_metrics"]
        ),
        "cross_path_binned_error_sha256": binned_error_sha256(
            result["cross_path_binned_error"]
        ),
        "transfer_index_sha256": transfer_index_sha256(transfer_index),
        "decoding_provenance_sha256": decoding_provenance_sha256(result),
    }


def _path_progression_transfer_hashes(
    result: Mapping[str, Any],
) -> dict[tuple[str, str, str], dict[str, str]]:
    """Return logical hashes for every valid transfer output."""
    from v1ca1.spyglass.path_progression_decoding import (
        transfer_progression_sha256,
        transfer_support_sha256,
        validate_decoding_comparison_result,
    )

    canonical = validate_decoding_comparison_result(result)
    return {
        key: {
            "true_progression_sha256": transfer_progression_sha256(
                output["true"],
                key=key,
                role="true",
            ),
            "decoded_progression_sha256": transfer_progression_sha256(
                output["decoded"],
                key=key,
                role="decoded",
            ),
            "decoding_support_sha256": transfer_support_sha256(
                output["true"],
                output["decoded"],
                key=key,
            ),
        }
        for key, output in canonical["cross_path_outputs"].items()
    }


def _write_path_progression_decoding_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one path-progression decoding NWB."""
    import pynwb

    from v1ca1.spyglass.path_progression_decoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        binned_error_to_dynamic_table,
        build_transfer_index_table,
        decoding_provenance_to_dynamic_table,
        decoding_summary_to_dynamic_table,
        path_progression_decoding_result_from_nwb_objects,
        selected_units_to_dynamic_table,
        transfer_index_to_dynamic_table,
        transfer_progression_to_time_series,
        transfer_support_to_time_intervals,
        unit_eligibility_to_dynamic_table,
        validate_decoding_comparison_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for "
            "PathProgressionDecoding output."
        )
    canonical = validate_decoding_comparison_result(result)
    expected_hashes = _path_progression_decoding_hashes(canonical)
    expected_transfer_hashes = _path_progression_transfer_hashes(canonical)
    transfer_index = build_transfer_index_table(canonical)
    scratch_objects = {
        "unit_eligibility": unit_eligibility_to_dynamic_table(
            canonical["unit_eligibility"]
        ),
        "selected_units": selected_units_to_dynamic_table(
            canonical["selected_units"]
        ),
        "decoding_summary": decoding_summary_to_dynamic_table(
            canonical["cross_path_metrics"]
        ),
        "cross_path_binned_error": binned_error_to_dynamic_table(
            canonical["cross_path_binned_error"]
        ),
        "transfer_index": transfer_index_to_dynamic_table(transfer_index),
        "decoding_provenance": decoding_provenance_to_dynamic_table(
            canonical
        ),
    }
    transfer_objects = {}
    for key, output in canonical["cross_path_outputs"].items():
        transfer_objects[key] = {
            "true_progression": transfer_progression_to_time_series(
                output["true"],
                key=key,
                role="true",
            ),
            "decoded_progression": transfer_progression_to_time_series(
                output["decoded"],
                key=key,
                role="decoded",
            ),
            "decoding_support": transfer_support_to_time_intervals(
                output["true"],
                output["decoded"],
                key=key,
            ),
        }
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    transfer_rows: list[dict[str, Any]] = []
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name, nwb_object in scratch_objects.items():
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(nwb_object)
                )
            _io, analysis_nwb = builder.open_nwb
            for key, objects in transfer_objects.items():
                true_object_id = str(
                    builder.add_nwb_object(objects["true_progression"])
                )
                decoded_object_id = str(
                    builder.add_nwb_object(objects["decoded_progression"])
                )
                support_object = objects["decoding_support"]
                analysis_nwb.add_time_intervals(support_object)
                hashes = expected_transfer_hashes[key]
                metric = canonical["cross_path_metrics"]
                metric = metric[
                    (metric["transfer_family"] == key[0])
                    & (metric["source_trajectory"] == key[1])
                    & (metric["target_trajectory"] == key[2])
                ].iloc[0]
                transfer_rows.append(
                    {
                        "transfer_family": key[0],
                        "source_trajectory": key[1],
                        "target_trajectory": key[2],
                        "true_progression_object_id": true_object_id,
                        "decoded_progression_object_id": decoded_object_id,
                        "decoding_support_object_id": str(
                            support_object.object_id
                        ),
                        **hashes,
                        "n_samples": int(metric["n_samples"]),
                    }
                )
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "PathProgressionDecoding analysis NWB failed PyNWB "
                    f"validation: {validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_transfers = {
                    (
                        str(row["transfer_family"]),
                        str(row["source_trajectory"]),
                        str(row["target_trajectory"]),
                    ): {
                        "true_progression": stored_nwb.objects[
                            row["true_progression_object_id"]
                        ],
                        "decoded_progression": stored_nwb.objects[
                            row["decoded_progression_object_id"]
                        ],
                        "decoding_support": stored_nwb.objects[
                            row["decoding_support_object_id"]
                        ],
                    }
                    for row in transfer_rows
                }
                stored_result = (
                    path_progression_decoding_result_from_nwb_objects(
                        unit_eligibility=stored_nwb.objects[
                            object_ids["unit_eligibility_object_id"]
                        ],
                        selected_units=stored_nwb.objects[
                            object_ids["selected_units_object_id"]
                        ],
                        decoding_summary=stored_nwb.objects[
                            object_ids["decoding_summary_object_id"]
                        ],
                        binned_error=stored_nwb.objects[
                            object_ids["cross_path_binned_error_object_id"]
                        ],
                        transfer_index=stored_nwb.objects[
                            object_ids["transfer_index_object_id"]
                        ],
                        provenance=stored_nwb.objects[
                            object_ids["decoding_provenance_object_id"]
                        ],
                        transfer_objects=stored_transfers,
                    )
                )
                if _path_progression_decoding_hashes(
                    stored_result
                ) != expected_hashes or _path_progression_transfer_hashes(
                    stored_result
                ) != expected_transfer_hashes:
                    raise ValueError(
                        "Path-progression decoding NWB objects changed "
                        "during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError("PathProgressionDecoding analysis NWB was not created.")
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "n_units_input": int(canonical["n_units_input"]),
        "n_units_eligible": int(canonical["n_units_eligible"]),
        "n_transfer_pairs_expected": int(
            canonical["n_transfer_pairs_expected"]
        ),
        "n_transfer_pairs_valid": int(canonical["n_transfer_pairs_valid"]),
        "n_decoded_samples": int(canonical["n_decoded_samples"]),
        "analysis_status": str(canonical["analysis_status"]),
        "eligible_units_sha256": str(canonical["eligible_units_sha256"]),
        "_transfer_rows": transfer_rows,
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_path_progression_decoding_nwb_objects(
    decoding_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one set of fixed path-progression NWB objects."""
    relation = decoding_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading PathProgressionDecoding."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "PathProgressionDecoding NWB fetch must resolve one result."
        )
    record = dict(records[0])
    expected = {
        "unit_eligibility",
        "selected_units",
        "decoding_summary",
        "cross_path_binned_error",
        "transfer_index",
        "decoding_provenance",
    }
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(
            "PathProgressionDecoding NWB fetch is missing objects "
            f"{missing!r}."
        )
    return {name: record[name] for name in expected}


def _fetch_path_progression_transfer_nwb_objects(
    transfer_table: Any,
    key: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    """Fetch and order every valid transfer's three NWB objects."""
    from v1ca1.spyglass.path_progression_decoding import TRANSFER_PAIR_SPECS

    relation = transfer_table & dict(key)
    try:
        records = [dict(record) for record in relation.fetch_nwb()]
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading PathProgressionDecoding."
            ) from exc
        raise
    by_key = {}
    for record in records:
        transfer_key = (
            str(record["transfer_family"]),
            str(record["source_trajectory"]),
            str(record["target_trajectory"]),
        )
        if transfer_key in by_key:
            raise ValueError("Duplicate PathProgressionDecoding transfer row.")
        expected_objects = {
            "true_progression",
            "decoded_progression",
            "decoding_support",
        }
        missing = sorted(expected_objects.difference(record))
        if missing:
            raise ValueError(
                "PathProgressionDecoding transfer fetch is missing objects "
                f"{missing!r}."
            )
        by_key[transfer_key] = record
    ordered_keys = [
        (
            str(spec["transfer_family"]),
            str(spec["source_trajectory"]),
            str(spec["target_trajectory"]),
        )
        for spec in TRANSFER_PAIR_SPECS
        if (
            str(spec["transfer_family"]),
            str(spec["source_trajectory"]),
            str(spec["target_trajectory"]),
        )
        in by_key
    ]
    if len(ordered_keys) != len(by_key):
        raise ValueError("Transfer rows include a noncanonical transfer key.")
    objects = {
        transfer_key: {
            "true_progression": by_key[transfer_key]["true_progression"],
            "decoded_progression": by_key[transfer_key]["decoded_progression"],
            "decoding_support": by_key[transfer_key]["decoding_support"],
        }
        for transfer_key in ordered_keys
    }
    rows = {transfer_key: by_key[transfer_key] for transfer_key in ordered_keys}
    return objects, rows


def _load_path_progression_decoding_result(
    *,
    result_row: Mapping[str, Any],
    decoding_table: Any,
    transfer_table: Any,
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Fetch, reconstruct, and cross-check one progression decoder."""
    from v1ca1.spyglass.path_progression_decoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        path_progression_decoding_result_from_nwb_objects,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "PathProgressionDecoding artifact schema version is unsupported."
        )
    key = {
        "path_progression_decoding_id": result_row[
            "path_progression_decoding_id"
        ]
    }
    fixed_objects = _fetch_path_progression_decoding_nwb_objects(
        decoding_table,
        key,
    )
    transfer_objects, transfer_rows = (
        _fetch_path_progression_transfer_nwb_objects(transfer_table, key)
    )
    result = path_progression_decoding_result_from_nwb_objects(
        unit_eligibility=fixed_objects["unit_eligibility"],
        selected_units=fixed_objects["selected_units"],
        decoding_summary=fixed_objects["decoding_summary"],
        binned_error=fixed_objects["cross_path_binned_error"],
        transfer_index=fixed_objects["transfer_index"],
        provenance=fixed_objects["decoding_provenance"],
        transfer_objects=transfer_objects,
    )
    metadata = result["metadata"]
    expected_metadata = {
        "path_progression_decoding_id": str(
            selection_row["path_progression_decoding_id"]
        ),
        "animal_name": str(animal_name),
        "date": str(date),
        "region": str(region_row["region_name"]),
        "epoch": str(selection_row["epoch"]),
        "cohort_epoch": str(selection_row["cohort_epoch"]),
        "parameter_name": str(
            parameters_row["path_progression_decoding_param_name"]
        ),
        "parameter_sha256": str(
            selection_row["path_progression_decoding_parameters_sha256"]
        ),
        "eligibility_rule_sha256": str(
            selection_row["eligibility_rule_sha256"]
        ),
        "transfer_spec_sha256": str(selection_row["transfer_spec_sha256"]),
        "decoding_output_rule_sha256": str(
            selection_row["decoding_output_rule_sha256"]
        ),
    }
    if {name: str(metadata[name]) for name in expected_metadata} != (
        expected_metadata
    ):
        raise ValueError(
            "PathProgressionDecoding NWB metadata does not match selection."
        )
    expected_counts = {
        "n_units_input": int(result["n_units_input"]),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_transfer_pairs_expected": int(result["n_transfer_pairs_expected"]),
        "n_transfer_pairs_valid": int(result["n_transfer_pairs_valid"]),
        "n_decoded_samples": int(result["n_decoded_samples"]),
    }
    if any(int(result_row[name]) != value for name, value in expected_counts.items()):
        raise ValueError("PathProgressionDecoding NWB counts disagree with row.")
    if str(result_row["analysis_status"]) != str(result["analysis_status"]) or (
        str(result_row["eligible_units_sha256"])
        != str(result["eligible_units_sha256"])
    ):
        raise ValueError("PathProgressionDecoding NWB summary disagrees with row.")
    hashes = _path_progression_decoding_hashes(result)
    if any(str(result_row[name]) != value for name, value in hashes.items()):
        raise ValueError("PathProgressionDecoding NWB object hash mismatch.")
    transfer_hashes = _path_progression_transfer_hashes(result)
    for transfer_key, expected in transfer_hashes.items():
        row = transfer_rows.get(transfer_key)
        if row is None or any(
            str(row[name]) != value for name, value in expected.items()
        ):
            raise ValueError("PathProgressionDecoding transfer hash mismatch.")
    return result


def _path_specific_place_decoding_hashes(
    result: Mapping[str, Any],
) -> dict[str, str]:
    """Return storage-independent hashes for one place decoder."""
    from v1ca1.spyglass.path_specific_decoding import (
        binned_error_sha256,
        decoding_provenance_sha256,
        decoding_summary_sha256,
        decoding_support_sha256,
        fold_qc_sha256,
        position_time_series_sha256,
        selected_units_sha256,
    )

    return {
        "selected_units_table_sha256": selected_units_sha256(
            result["selected_units"]
        ),
        "fold_qc_sha256": fold_qc_sha256(result["fold_qc"]),
        "decoding_summary_sha256": decoding_summary_sha256(
            result["summary"]
        ),
        "decoding_error_by_position_sha256": binned_error_sha256(
            result["binned_error"]
        ),
        "true_position_sha256": position_time_series_sha256(
            result["true"],
            kind="true",
        ),
        "decoded_position_sha256": position_time_series_sha256(
            result["decoded"],
            kind="decoded",
        ),
        "decoding_support_sha256": decoding_support_sha256(
            result["true"],
            result["decoded"],
        ),
        "decoding_provenance_sha256": decoding_provenance_sha256(result),
    }


def _write_path_specific_place_decoding_nwb(
    *,
    nwb_file_name: str,
    result: Mapping[str, Any],
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Write, reopen, and verify one place-decoding analysis NWB."""
    import pynwb

    from v1ca1.spyglass.path_specific_decoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        binned_error_to_dynamic_table,
        decoded_position_to_time_series,
        decoding_provenance_to_dynamic_table,
        decoding_summary_to_dynamic_table,
        decoding_support_to_time_intervals,
        fold_qc_to_dynamic_table,
        path_specific_place_decoding_result_from_nwb_objects,
        selected_units_to_dynamic_table,
        true_position_to_time_series,
        validate_path_specific_decoding_result,
    )

    if analysis_nwbfile_table is None:
        raise ValueError(
            "analysis_nwbfile_table is required for "
            "PathSpecificPlaceDecoding output."
        )
    canonical = validate_path_specific_decoding_result(result)
    expected_hashes = _path_specific_place_decoding_hashes(canonical)
    scratch_objects = {
        "selected_units": selected_units_to_dynamic_table(
            canonical["selected_units"]
        ),
        "fold_qc": fold_qc_to_dynamic_table(canonical["fold_qc"]),
        "decoding_summary": decoding_summary_to_dynamic_table(
            canonical["summary"]
        ),
        "decoding_error_by_position": binned_error_to_dynamic_table(
            canonical["binned_error"]
        ),
        "true_position": true_position_to_time_series(canonical["true"]),
        "decoded_position": decoded_position_to_time_series(
            canonical["decoded"]
        ),
        "decoding_provenance": decoding_provenance_to_dynamic_table(
            canonical
        ),
    }
    support_object = decoding_support_to_time_intervals(
        canonical["true"],
        canonical["decoded"],
    )
    analysis_table = (
        analysis_nwbfile_table()
        if isinstance(analysis_nwbfile_table, type)
        else analysis_nwbfile_table
    )
    analysis_file_name: str | None = None
    analysis_file_path: str | None = None
    object_ids: dict[str, str] = {}
    try:
        with analysis_table.build(str(nwb_file_name)) as builder:
            analysis_file_name = str(builder.analysis_file_name)
            analysis_file_path = str(builder.get_path())
            for name, nwb_object in scratch_objects.items():
                object_ids[f"{name}_object_id"] = str(
                    builder.add_nwb_object(nwb_object)
                )
            object_ids["decoding_support_object_id"] = str(
                support_object.object_id
            )
            _io, analysis_nwb = builder.open_nwb
            analysis_nwb.add_time_intervals(support_object)
            builder.close_and_write()
            Path(analysis_file_path).chmod(0o644)
            validation_errors = pynwb.validate(path=analysis_file_path)
            if validation_errors:
                raise ValueError(
                    "PathSpecificPlaceDecoding analysis NWB failed PyNWB "
                    f"validation: {validation_errors!r}."
                )

            with pynwb.NWBHDF5IO(
                analysis_file_path,
                mode="r",
                load_namespaces=True,
            ) as io:
                stored_nwb = io.read()
                stored_result = (
                    path_specific_place_decoding_result_from_nwb_objects(
                        selected_units=stored_nwb.objects[
                            object_ids["selected_units_object_id"]
                        ],
                        fold_qc=stored_nwb.objects[
                            object_ids["fold_qc_object_id"]
                        ],
                        summary=stored_nwb.objects[
                            object_ids["decoding_summary_object_id"]
                        ],
                        binned_error=stored_nwb.objects[
                            object_ids[
                                "decoding_error_by_position_object_id"
                            ]
                        ],
                        true_position=stored_nwb.objects[
                            object_ids["true_position_object_id"]
                        ],
                        decoded_position=stored_nwb.objects[
                            object_ids["decoded_position_object_id"]
                        ],
                        decoding_support=stored_nwb.objects[
                            object_ids["decoding_support_object_id"]
                        ],
                        provenance=stored_nwb.objects[
                            object_ids["decoding_provenance_object_id"]
                        ],
                    )
                )
                if (
                    _path_specific_place_decoding_hashes(stored_result)
                    != expected_hashes
                ):
                    raise ValueError(
                        "Path-specific place decoding NWB objects changed "
                        "during write."
                    )
    except Exception:
        if analysis_file_path is not None:
            _remove_created_artifacts([analysis_file_path])
        raise

    if analysis_file_name is None or analysis_file_path is None:
        raise RuntimeError(
            "PathSpecificPlaceDecoding analysis NWB was not created."
        )
    return {
        "analysis_file_name": analysis_file_name,
        **object_ids,
        **expected_hashes,
        "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
        "n_units": int(canonical["n_units"]),
        "n_folds_expected": int(canonical["n_folds_expected"]),
        "n_folds_valid": int(canonical["n_folds_valid"]),
        "n_decoded_samples": int(canonical["n_decoded_samples"]),
        "analysis_status": str(canonical["analysis_status"]),
        "selected_units_sha256": str(canonical["selected_units_sha256"]),
        "artifact_origin": str(canonical["artifact_origin"]),
        "legacy_artifact_provenance": canonical[
            "legacy_artifact_provenance"
        ],
        "_created_artifact_paths": [analysis_file_path],
    }


def _fetch_path_specific_place_decoding_nwb_objects(
    decoding_table: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Fetch exactly one complete set of place-decoding NWB objects."""
    relation = decoding_table & dict(key)
    try:
        records = relation.fetch_nwb()
    except ValueError as exc:
        if "not found in registry" in str(exc):
            raise RuntimeError(
                "The custom AnalysisNwbfile table must be registered with "
                "Spyglass before loading PathSpecificPlaceDecoding."
            ) from exc
        raise
    if len(records) != 1:
        raise ValueError(
            "PathSpecificPlaceDecoding NWB fetch must resolve exactly one "
            "result."
        )
    record = dict(records[0])
    expected = {
        "selected_units",
        "fold_qc",
        "decoding_summary",
        "decoding_error_by_position",
        "true_position",
        "decoded_position",
        "decoding_support",
        "decoding_provenance",
    }
    missing = sorted(expected.difference(record))
    if missing:
        raise ValueError(
            "PathSpecificPlaceDecoding NWB fetch is missing objects "
            f"{missing!r}."
        )
    return {name: record[name] for name in expected}


def _load_path_specific_place_decoding_result(
    *,
    result_row: Mapping[str, Any],
    decoding_table: Any,
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    region_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Fetch, reconstruct, and cross-check one place decoder."""
    from v1ca1.spyglass.path_specific_decoding import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        path_specific_place_decoding_result_from_nwb_objects,
    )

    if str(result_row.get("artifact_schema_version")) != (
        NWB_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding artifact schema version is unsupported."
        )
    objects = _fetch_path_specific_place_decoding_nwb_objects(
        decoding_table,
        {
            "path_specific_place_decoding_id": result_row[
                "path_specific_place_decoding_id"
            ]
        },
    )
    result = path_specific_place_decoding_result_from_nwb_objects(
        selected_units=objects["selected_units"],
        fold_qc=objects["fold_qc"],
        summary=objects["decoding_summary"],
        binned_error=objects["decoding_error_by_position"],
        true_position=objects["true_position"],
        decoded_position=objects["decoded_position"],
        decoding_support=objects["decoding_support"],
        provenance=objects["decoding_provenance"],
    )
    for field_name, observed in (
        _path_specific_place_decoding_hashes(result)
    ).items():
        if observed != str(result_row.get(field_name)):
            raise ValueError(
                "PathSpecificPlaceDecoding result metadata disagrees with "
                f"its NWB objects: {field_name}."
            )
    _validate_path_specific_place_decoding_artifact_link(
        bundle=result,
        result_row=result_row,
        selection_row=selection_row,
        parameters_row=parameters_row,
        region_row=region_row,
        animal_name=animal_name,
        date=date,
    )
    return result


def _make_path_specific_place_decoding_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one within-epoch decoder analysis NWB."""
    from v1ca1.spyglass.path_specific_decoding import (
        compute_path_specific_place_decoding,
    )

    context = _load_path_specific_place_decoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    loaded_spikes = _load_path_specific_place_decoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_path_specific_place_decoding_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    parameters = context["parameters"]
    result = compute_path_specific_place_decoding(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        path_specific_place_decoding_id=key[
            "path_specific_place_decoding_id"
        ],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=nwb_inputs["position"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        graph_inputs=nwb_inputs["graph_inputs"],
        movement_interval=context["movement"]["movement_intervals"],
        parameter_name=parameters[
            "path_specific_place_decoding_param_name"
        ],
        parameter_sha256=key[
            "path_specific_place_decoding_parameters_sha256"
        ],
        output_rule_sha256=key[
            "path_specific_place_decoding_output_rule_sha256"
        ],
        n_folds=parameters["n_folds"],
        decoding_bin_size_s=parameters["decoding_bin_size_s"],
        sliding_window_size_bins=parameters[
            "sliding_window_size_bins"
        ],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        random_seed=parameters["random_seed"],
    )
    del artifact_root
    return _write_path_specific_place_decoding_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _register_existing_path_specific_place_decoding_row(
    *,
    key: Mapping[str, Any],
    source_true_path: Path,
    source_decoded_path: Path,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Validate and normalize one legacy decoder into an analysis NWB."""
    from v1ca1.spyglass.path_specific_decoding import (
        register_existing_path_specific_decoding_artifact,
    )
    from v1ca1.spyglass.path_specific_place import graph_length_from_inputs

    context = _load_path_specific_place_decoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    loaded_spikes = _load_path_specific_place_decoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_path_specific_place_decoding_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    path_lengths = [
        graph_length_from_inputs(nwb_inputs["graph_inputs"][trajectory_type])
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    ]
    if not all(
        math.isclose(
            path_lengths[0],
            path_length,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        for path_length in path_lengths[1:]
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding requires four common-length paths."
        )
    parameters = context["parameters"]
    registered = register_existing_path_specific_decoding_artifact(
        source_true_path=Path(source_true_path),
        source_decoded_path=Path(source_decoded_path),
        destination_path=None,
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        path_specific_place_decoding_id=key[
            "path_specific_place_decoding_id"
        ],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        movement_interval=context["movement"]["movement_intervals"],
        path_length_cm=path_lengths[0],
        parameter_name=parameters[
            "path_specific_place_decoding_param_name"
        ],
        parameter_sha256=key[
            "path_specific_place_decoding_parameters_sha256"
        ],
        output_rule_sha256=key[
            "path_specific_place_decoding_output_rule_sha256"
        ],
        n_folds=parameters["n_folds"],
        decoding_bin_size_s=parameters["decoding_bin_size_s"],
        sliding_window_size_bins=parameters[
            "sliding_window_size_bins"
        ],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        random_seed=parameters["random_seed"],
        source_v1ca1_git_commit=source_v1ca1_git_commit,
    )
    provenance = dict(registered["legacy_artifact_provenance"])
    provenance["source_spyglass_git_commit"] = (
        source_spyglass_git_commit
    )
    registered["legacy_artifact_provenance"] = provenance
    del artifact_root
    return _write_path_specific_place_decoding_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _make_motor_encoding_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Compute and persist one nine-model motor-encoding analysis NWB."""
    from v1ca1.spyglass.motor_encoding import compute_motor_encoding

    context = _load_motor_encoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        session_table=session_table,
    )
    loaded_spikes = _load_motor_encoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_motor_encoding_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    parameters = context["parameters"]
    selection = context["selection"]
    result = compute_motor_encoding(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        motor_encoding_id=key[
            "motor_encoding_id"
        ],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        primary_position=nwb_inputs["primary_position"],
        orientation_reference_position=nwb_inputs[
            "orientation_reference_position"
        ],
        primary_position_source=selection[
            "primary_position_series_name"
        ],
        orientation_reference_position_source=selection[
            "orientation_reference_position_series_name"
        ],
        trajectory_intervals_by_type=nwb_inputs["trajectory_intervals"],
        graph_inputs_by_configuration=nwb_inputs["graph_inputs"],
        movement_intervals=context["movement"]["movement_intervals"],
        movement_firing_rate_table=context["movement"]["table"],
        stability_tables_by_trajectory=context["stability_tables"],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
        parameter_name=parameters[
            "motor_encoding_param_name"
        ],
        parameter_sha256=selection[
            "motor_encoding_parameters_sha256"
        ],
        model_spec_sha256=selection[
            "motor_encoding_model_spec_sha256"
        ],
        output_rule_sha256=selection[
            "motor_encoding_output_rule_sha256"
        ],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        outer_n_folds=parameters["outer_n_folds"],
        inner_n_folds=parameters["inner_n_folds"],
        random_seed=parameters["random_seed"],
        ridge_values=parameters["ridge_values"],
        spatial_bin_sizes_cm=parameters["spatial_bin_sizes_cm"],
        motor_feature_mode=parameters["motor_feature_mode"],
        motor_zscore_eps=parameters["motor_zscore_eps"],
        motor_spline_n_basis=parameters["motor_spline_n_basis"],
        motor_spline_order=parameters["motor_spline_order"],
        position_spline_order=parameters["position_spline_order"],
        speed_smoothing_sigma_s=parameters[
            "speed_smoothing_sigma_s"
        ],
        generalized_place_branch_gap_cm=parameters[
            "generalized_place_branch_gap_cm"
        ],
    )
    del artifact_root
    return _write_motor_encoding_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _register_existing_motor_encoding_row(
    *,
    key: Mapping[str, Any],
    source_nested_cv_path: Path,
    source_full_refit_path: Path,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    stability_table: Any,
    stability_selection_table: Any,
    tuning_curve_selection_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any,
) -> dict[str, Any]:
    """Validate and store one paired legacy motor fit in analysis NWB."""
    from v1ca1.spyglass.motor_encoding import (
        register_existing_motor_encoding_artifact,
    )

    context = _load_motor_encoding_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        stability_table=stability_table,
        stability_selection_table=stability_selection_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        session_table=session_table,
    )
    loaded_spikes = _load_motor_encoding_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_motor_encoding_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    resolver = _legacy_motor_unit_identity_resolver(loaded_spikes)
    parameters = context["parameters"]
    selection = context["selection"]
    registered = register_existing_motor_encoding_artifact(
        source_nested_cv_path=Path(source_nested_cv_path),
        source_full_refit_path=Path(source_full_refit_path),
        destination_path=None,
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        motor_encoding_id=key[
            "motor_encoding_id"
        ],
        movement_firing_rate_table=context["movement"]["table"],
        stability_tables_by_trajectory=context["stability_tables"],
        graph_inputs_by_configuration=nwb_inputs["graph_inputs"],
        unit_identity_resolver=resolver,
        primary_position_source=selection[
            "primary_position_series_name"
        ],
        orientation_reference_position_source=selection[
            "orientation_reference_position_series_name"
        ],
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        minimum_stability_correlation=parameters[
            "minimum_stability_correlation"
        ],
        parameter_name=parameters[
            "motor_encoding_param_name"
        ],
        parameter_sha256=selection[
            "motor_encoding_parameters_sha256"
        ],
        model_spec_sha256=selection[
            "motor_encoding_model_spec_sha256"
        ],
        output_rule_sha256=selection[
            "motor_encoding_output_rule_sha256"
        ],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        outer_n_folds=parameters["outer_n_folds"],
        inner_n_folds=parameters["inner_n_folds"],
        random_seed=parameters["random_seed"],
        ridge_values=parameters["ridge_values"],
        spatial_bin_sizes_cm=parameters["spatial_bin_sizes_cm"],
        motor_feature_mode=parameters["motor_feature_mode"],
        motor_zscore_eps=parameters["motor_zscore_eps"],
        motor_spline_n_basis=parameters["motor_spline_n_basis"],
        motor_spline_order=parameters["motor_spline_order"],
        position_spline_order=parameters["position_spline_order"],
        speed_smoothing_sigma_s=parameters[
            "speed_smoothing_sigma_s"
        ],
        generalized_place_branch_gap_cm=parameters[
            "generalized_place_branch_gap_cm"
        ],
        source_v1ca1_git_commit=source_v1ca1_git_commit,
        overwrite=False,
    )
    provenance = dict(registered["legacy_artifact_provenance"])
    provenance["source_spyglass_git_commit"] = (
        source_spyglass_git_commit
    )
    registered["legacy_artifact_provenance"] = provenance
    del artifact_root
    return _write_motor_encoding_nwb(
        nwb_file_name=str(key["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _make_dark_light_glm_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one coupled dark/light four-model analysis NWB."""
    from v1ca1.spyglass.dark_light_glm import (
        compute_dark_light_glm,
    )

    context = _load_dark_light_glm_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    loaded_spikes = _load_dark_light_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_dark_light_glm_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    selection = context["selection"]
    parameters = context["parameters"]
    epoch_by_condition = {
        condition_name: str(selection[f"{condition_name}_epoch"])
        for condition_name in ("dark", "light")
    }
    position_by_epoch = {
        epoch_by_condition[condition_name]: nwb_inputs["positions"][
            condition_name
        ]
        for condition_name in ("dark", "light")
    }
    trajectory_intervals_by_epoch = {
        epoch_by_condition[condition_name]: nwb_inputs[
            "trajectory_intervals"
        ][condition_name]
        for condition_name in ("dark", "light")
    }
    movement_by_epoch = {
        epoch_by_condition[condition_name]: context["movements"][
            condition_name
        ]["movement_intervals"]
        for condition_name in ("dark", "light")
    }
    movement_tables = {
        condition_name: context["movements"][condition_name]["table"]
        for condition_name in ("dark", "light")
    }
    result = compute_dark_light_glm(
        dark_light_glm_id=selection["dark_light_glm_id"],
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        light_epoch=epoch_by_condition["light"],
        dark_epoch=epoch_by_condition["dark"],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        dark_movement_firing_rate_table=movement_tables["dark"],
        light_movement_firing_rate_table=movement_tables["light"],
        movement_by_epoch=movement_by_epoch,
        trajectory_intervals_by_epoch=trajectory_intervals_by_epoch,
        graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
        position_by_epoch=position_by_epoch,
        parameter_name=parameters["dark_light_glm_param_name"],
        parameter_sha256=selection["dark_light_glm_parameters_sha256"],
        output_rule_sha256=selection["dark_light_glm_output_rule_sha256"],
        basis_candidate_mode=parameters["basis_candidate_mode"],
        basis_candidates=parameters["basis_candidates"],
        bin_sizes_s=parameters["bin_sizes_s"],
        ridges=parameters["ridges"],
        n_folds=parameters["n_folds"],
        random_seed=parameters["random_seed"],
        spline_order=parameters["spline_order"],
        min_dark_firing_rate_hz=parameters[
            "min_dark_firing_rate_hz"
        ],
        min_light_firing_rate_hz=parameters[
            "min_light_firing_rate_hz"
        ],
        use_speed=parameters["use_speed"],
        speed_feature_mode=parameters["speed_feature_mode"],
        n_splines_speed=parameters["n_splines_speed"],
        spline_order_speed=parameters["spline_order_speed"],
        speed_bounds=parameters["speed_bounds"],
        speed_smoothing_sigma_s=parameters[
            "speed_smoothing_sigma_s"
        ],
        sources={
            "dark_position_series_name": context["movement_selections"][
                "dark"
            ]["position_series_name"],
            "light_position_series_name": context["movement_selections"][
                "light"
            ]["position_series_name"],
            **{
                f"{trajectory_type}_configuration_name": selection[
                    f"{trajectory_type}_configuration_name"
                ]
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            },
        },
    )
    del artifact_root
    return _write_dark_light_glm_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _register_existing_dark_light_glm_row(
    *,
    key: Mapping[str, Any],
    source_candidate_paths: list[Path],
    source_selected_paths_by_model: Mapping[str, Path],
    source_selection_summary_path: Path,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Validate and convert one exact imported dark/light result to NWB."""
    from v1ca1.spyglass.dark_light_glm import (
        SCHEMA_VERSION_BY_MODE,
        register_existing_dark_light_glm_artifact,
    )
    from v1ca1.spyglass.selection import provenance_sha256

    context = _load_dark_light_glm_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    loaded_spikes = _load_dark_light_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_dark_light_glm_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    resolver = _legacy_dark_light_unit_identity_resolver(loaded_spikes)
    selection = context["selection"]
    parameters = context["parameters"]
    registered = register_existing_dark_light_glm_artifact(
            source_candidate_paths=[
                Path(path) for path in source_candidate_paths
            ],
            source_selected_paths_by_model={
                model_name: Path(path)
                for model_name, path in source_selected_paths_by_model.items()
            },
            source_selection_summary_path=Path(
                source_selection_summary_path
            ),
            destination_path=None,
            dark_light_glm_id=selection["dark_light_glm_id"],
            animal_name=context["animal_name"],
            date=context["date"],
            region=context["region"],
            light_epoch=selection["light_epoch"],
            dark_epoch=selection["dark_epoch"],
            unit_identity_resolver=resolver,
            graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
            basis_candidate_mode=parameters["basis_candidate_mode"],
            basis_candidates=parameters["basis_candidates"],
            parameter_name=parameters["dark_light_glm_param_name"],
            parameter_sha256=selection[
                "dark_light_glm_parameters_sha256"
            ],
            output_rule_sha256=selection[
                "dark_light_glm_output_rule_sha256"
            ],
            speed_smoothing_sigma_s=parameters[
                "speed_smoothing_sigma_s"
            ],
            source_v1ca1_git_commit=source_v1ca1_git_commit,
            overwrite=False,
        )
    expected_parameters = {
        "schema_version": SCHEMA_VERSION_BY_MODE[
            parameters["basis_candidate_mode"]
        ],
        "parameter_name": parameters["dark_light_glm_param_name"],
        "parameter_sha256": selection[
            "dark_light_glm_parameters_sha256"
        ],
        "output_rule_sha256": selection[
            "dark_light_glm_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "dark_light_glm_param_name"
        },
    }
    if provenance_sha256(dict(registered["parameters"])) != (
        provenance_sha256(expected_parameters)
    ):
        raise ValueError(
            "Legacy DarkLightGLM parameters do not match the selected "
            "parameter row."
        )
    provenance = dict(registered["legacy_artifact_provenance"] or {})
    provenance["source_spyglass_git_commit"] = source_spyglass_git_commit
    registered = {
        **registered,
        "legacy_artifact_provenance": provenance,
    }
    del artifact_root
    return _write_dark_light_glm_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _make_swap_glm_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    dark_light_glm_table: Any,
    dark_light_glm_selection_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one held-out swapped-light analysis NWB."""
    from v1ca1.spyglass.swap_glm import (
        compute_swap_glm,
    )

    context = _load_swap_glm_context(
        key=key,
        parameters_table=parameters_table,
        dark_light_glm_table=dark_light_glm_table,
        dark_light_glm_selection_table=dark_light_glm_selection_table,
        region_sorted_spikes_group_table=(
            region_sorted_spikes_group_table
        ),
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    loaded_spikes = _load_swap_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_swap_glm_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    selection = context["selection"]
    parameters = context["parameters"]
    result = compute_swap_glm(
        swap_glm_id=selection["swap_glm_id"],
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        dark_light_glm_result=context["dark_light_snapshot"]["bundle"],
        dark_light_glm_upstream_provenance={
            "dark_light_glm_id": str(selection["dark_light_glm_id"]),
            "dark_light_glm_sha256": selection["dark_light_glm_sha256"],
            "dark_light_selected_model_sha256_by_model": selection[
                "dark_light_selected_model_sha256_by_model"
            ],
            "dark_light_parameter_sha256": selection[
                "dark_light_parameter_sha256"
            ],
            "dark_light_output_rule_sha256": selection[
                "dark_light_output_rule_sha256"
            ],
            "upstream_analysis_status": selection[
                "upstream_analysis_status"
            ],
        },
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        movement_interval=context["movement"]["movement_intervals"],
        movement_analysis_status=context["movement"]["analysis_status"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
        position=nwb_inputs["position"],
        parameter_name=parameters["swap_glm_param_name"],
        parameter_sha256=selection["swap_glm_parameters_sha256"],
        output_rule_sha256=selection["swap_glm_output_rule_sha256"],
        swap_light_offset=parameters["swap_light_offset"],
        observed_spatial_bin_size_cm=parameters[
            "observed_spatial_bin_size_cm"
        ],
        sources={
            "light_test_position_series_name": context[
                "movement_selection"
            ]["position_series_name"],
            "light_test_movement_firing_rate_id": str(
                selection["light_test_movement_firing_rate_id"]
            ),
            **{
                f"{trajectory_type}_configuration_name": selection[
                    f"{trajectory_type}_configuration_name"
                ]
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            },
        },
    )
    _validate_swap_glm_upstream_link(
        result["upstream_provenance"],
        selection,
    )
    del artifact_root
    return _write_swap_glm_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _register_existing_swap_glm_row(
    *,
    key: Mapping[str, Any],
    source_result_path: Path,
    parameters_table: Any,
    dark_light_glm_table: Any,
    dark_light_glm_selection_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Strictly validate and convert an imported swap result to NWB."""
    from v1ca1.spyglass.swap_glm import (
        register_existing_swap_glm_artifact,
    )

    context = _load_swap_glm_context(
        key=key,
        parameters_table=parameters_table,
        dark_light_glm_table=dark_light_glm_table,
        dark_light_glm_selection_table=dark_light_glm_selection_table,
        region_sorted_spikes_group_table=(
            region_sorted_spikes_group_table
        ),
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
    )
    loaded_spikes = _load_swap_glm_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    _legacy_swap_glm_unit_identity_resolver(loaded_spikes)
    nwb_inputs = _load_swap_glm_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    selection = context["selection"]
    parameters = context["parameters"]
    registered = register_existing_swap_glm_artifact(
        source_result_path=Path(source_result_path),
        destination_path=None,
        swap_glm_id=selection["swap_glm_id"],
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        dark_light_glm_result=context["dark_light_snapshot"]["bundle"],
        dark_light_glm_upstream_provenance={
            "dark_light_glm_id": str(selection["dark_light_glm_id"]),
            "dark_light_glm_sha256": selection["dark_light_glm_sha256"],
            "dark_light_selected_model_sha256_by_model": selection[
                "dark_light_selected_model_sha256_by_model"
            ],
            "dark_light_parameter_sha256": selection[
                "dark_light_parameter_sha256"
            ],
            "dark_light_output_rule_sha256": selection[
                "dark_light_output_rule_sha256"
            ],
            "upstream_analysis_status": selection[
                "upstream_analysis_status"
            ],
            "dark_light_legacy_selected_sha256_by_model": dict(
                (
                    context["dark_light_snapshot"]["bundle"].get(
                        "legacy_artifact_provenance"
                    )
                    or {}
                ).get("source_selected_sha256", {})
            ),
        },
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        movement_interval=context["movement"]["movement_intervals"],
        movement_analysis_status=context["movement"]["analysis_status"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
        position=nwb_inputs["position"],
        position_offset_samples=int(
            nwb_inputs["position_row"]["analysis_start_offset_samples"]
        ),
        speed_threshold_cm_s=float(
            context["movement_parameters"]["speed_threshold_cm_s"]
        ),
        parameter_name=parameters["swap_glm_param_name"],
        parameter_sha256=selection["swap_glm_parameters_sha256"],
        output_rule_sha256=selection["swap_glm_output_rule_sha256"],
        swap_light_offset=parameters["swap_light_offset"],
        observed_spatial_bin_size_cm=parameters[
            "observed_spatial_bin_size_cm"
        ],
        source_v1ca1_git_commit=source_v1ca1_git_commit,
        overwrite=False,
    )
    _validate_swap_glm_upstream_link(
        registered["upstream_provenance"],
        selection,
    )
    provenance = dict(registered["legacy_artifact_provenance"] or {})
    provenance["source_spyglass_git_commit"] = source_spyglass_git_commit
    registered = {
        **registered,
        "legacy_artifact_provenance": provenance,
    }
    del artifact_root
    return _write_swap_glm_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _swap_tuning_curve_comparison_sources(
    *,
    context: Mapping[str, Any],
    selection: Mapping[str, Any],
    source_spyglass_git_commit: str | None = None,
) -> dict[str, Any]:
    """Return the shared informative source payload for one swap bundle."""
    sources = {
        "position_series_name": context["movement_selections"][
            "light_test"
        ]["position_series_name"],
        "position_offset_samples": int(selection["position_offset_samples"]),
        "speed_threshold_cm_s": float(selection["speed_threshold_cm_s"]),
        "movement_firing_rate_ids_by_epoch": {
            selection[f"{epoch_role}_epoch"]: str(
                selection[f"{epoch_role}_movement_firing_rate_id"]
            )
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        "tuning_curve_ids_by_epoch_trajectory": {
            selection[f"{epoch_role}_epoch"]: {
                trajectory_type: str(
                    selection[
                        f"{epoch_role}_{trajectory_type}_tuning_curve_id"
                    ]
                )
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            }
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        "source_conditions": {
            selection[f"{epoch_role}_epoch"]: selection[
                f"{epoch_role}_condition"
            ]
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
    }
    if source_spyglass_git_commit is not None:
        sources["source_spyglass_git_commit"] = str(
            source_spyglass_git_commit
        )
    return sources


def _make_swap_tuning_curve_comparison_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    tuning_curve_parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Compute and persist one empirical held-out swap-tuning analysis NWB."""
    from v1ca1.spyglass.swap_tuning import (
        compute_swap_tuning_curve_comparison,
    )

    context = _load_swap_tuning_curve_comparison_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=(
            region_sorted_spikes_group_table
        ),
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        tuning_curve_parameters_table=tuning_curve_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        session_table=session_table,
    )
    loaded_spikes = _load_swap_tuning_curve_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_swap_tuning_curve_comparison_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    selection = context["selection"]
    parameters = context["parameters"]
    light_test_movement = context["movement"]["light_test"]
    result = compute_swap_tuning_curve_comparison(
        swap_tuning_curve_comparison_id=selection[
            "swap_tuning_curve_comparison_id"
        ],
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        tuning_curve_artifact_paths=None,
        tuning_curves_by_role_trajectory=context[
            "tuning_curves_by_role_trajectory"
        ],
        movement_firing_rate_tables_by_role=context[
            "movement_firing_rate_tables"
        ],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=nwb_inputs["position"],
        position_offset_samples=int(
            nwb_inputs["position_row"]["analysis_start_offset_samples"]
        ),
        movement_interval=light_test_movement["movement_intervals"],
        movement_analysis_status=light_test_movement["analysis_status"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
        parameter_name=parameters[
            "swap_tuning_curve_comparison_param_name"
        ],
        parameter_sha256=selection[
            "swap_tuning_curve_comparison_parameters_sha256"
        ],
        output_rule_sha256=selection[
            "swap_tuning_curve_comparison_output_rule_sha256"
        ],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        min_dark_firing_rate_hz=parameters[
            "min_dark_firing_rate_hz"
        ],
        min_light_firing_rate_hz=parameters[
            "min_light_firing_rate_hz"
        ],
        source_tuning_curve_ids_by_role_trajectory={
            epoch_role: {
                trajectory_type: selection[
                    f"{epoch_role}_{trajectory_type}_tuning_curve_id"
                ]
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            }
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        source_tuning_parameters_sha256_by_role_trajectory=selection[
            "source_tuning_parameters_sha256_by_role_trajectory"
        ],
        movement_firing_rate_ids_by_role={
            epoch_role: selection[
                f"{epoch_role}_movement_firing_rate_id"
            ]
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        movement_firing_rate_table_sha256_by_role=selection[
            "movement_firing_rate_table_sha256_by_role"
        ],
        movement_intervals_sha256_by_role=selection[
            "movement_intervals_sha256_by_role"
        ],
        sources=_swap_tuning_curve_comparison_sources(
            context=context,
            selection=selection,
        ),
    )
    _validate_swap_tuning_curve_comparison_upstream_link(
        result["upstream_provenance"],
        selection,
    )
    del artifact_root
    return _write_swap_tuning_curve_comparison_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=result,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _register_existing_swap_tuning_curve_comparison_row(
    *,
    key: Mapping[str, Any],
    source_result_path: Path,
    source_summary_path: Path,
    parameters_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    tuning_curve_parameters_table: Any,
    epoch_intervals_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
    analysis_nwbfile_table: Any | None = None,
) -> dict[str, Any]:
    """Rebuild and verify one legacy empirical swap into an analysis NWB."""
    from v1ca1.spyglass.swap_tuning import (
        register_existing_swap_tuning_curve_comparison_artifact,
    )

    context = _load_swap_tuning_curve_comparison_context(
        key=key,
        parameters_table=parameters_table,
        region_sorted_spikes_group_table=(
            region_sorted_spikes_group_table
        ),
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        tuning_curve_table=tuning_curve_table,
        tuning_curve_selection_table=tuning_curve_selection_table,
        tuning_curve_parameters_table=tuning_curve_parameters_table,
        epoch_intervals_table=epoch_intervals_table,
        session_table=session_table,
    )
    loaded_spikes = _load_swap_tuning_curve_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    unit_identity_resolver = (
        _legacy_swap_tuning_curve_comparison_unit_identity_resolver(
            loaded_spikes
        )
    )
    nwb_inputs = _load_swap_tuning_curve_comparison_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    _validate_legacy_tuning_curve_inputs(
        position_row=nwb_inputs["position_row"],
        movement_parameters=context["movement_parameters"],
    )
    selection = context["selection"]
    parameters = context["parameters"]
    light_test_movement = context["movement"]["light_test"]
    registered = register_existing_swap_tuning_curve_comparison_artifact(
        source_result_path=Path(source_result_path),
        source_summary_path=Path(source_summary_path),
        destination_path=None,
        swap_tuning_curve_comparison_id=selection[
            "swap_tuning_curve_comparison_id"
        ],
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        tuning_curve_artifact_paths=None,
        tuning_curves_by_role_trajectory=context[
            "tuning_curves_by_role_trajectory"
        ],
        movement_firing_rate_tables_by_role=context[
            "movement_firing_rate_tables"
        ],
        spikes=loaded_spikes["ts_group"],
        stable_unit_ids=loaded_spikes["unit_ids"],
        position=nwb_inputs["position"],
        position_offset_samples=int(
            nwb_inputs["position_row"]["analysis_start_offset_samples"]
        ),
        movement_interval=light_test_movement["movement_intervals"],
        movement_analysis_status=light_test_movement["analysis_status"],
        trajectory_intervals=nwb_inputs["trajectory_intervals"],
        graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
        parameter_name=parameters[
            "swap_tuning_curve_comparison_param_name"
        ],
        parameter_sha256=selection[
            "swap_tuning_curve_comparison_parameters_sha256"
        ],
        output_rule_sha256=selection[
            "swap_tuning_curve_comparison_output_rule_sha256"
        ],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        min_dark_firing_rate_hz=parameters["min_dark_firing_rate_hz"],
        min_light_firing_rate_hz=parameters["min_light_firing_rate_hz"],
        source_tuning_curve_ids_by_role_trajectory={
            epoch_role: {
                trajectory_type: selection[
                    f"{epoch_role}_{trajectory_type}_tuning_curve_id"
                ]
                for trajectory_type in _DPP_TRAJECTORY_TYPES
            }
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        source_tuning_parameters_sha256_by_role_trajectory=selection[
            "source_tuning_parameters_sha256_by_role_trajectory"
        ],
        movement_firing_rate_ids_by_role={
            epoch_role: selection[
                f"{epoch_role}_movement_firing_rate_id"
            ]
            for epoch_role in _SWAP_TUNING_EPOCH_ROLES
        },
        movement_firing_rate_table_sha256_by_role=selection[
            "movement_firing_rate_table_sha256_by_role"
        ],
        movement_intervals_sha256_by_role=selection[
            "movement_intervals_sha256_by_role"
        ],
        sources=_swap_tuning_curve_comparison_sources(
            context=context,
            selection=selection,
            source_spyglass_git_commit=source_spyglass_git_commit,
        ),
        unit_identity_resolver=unit_identity_resolver,
        source_sorting_type="ImportedSpikeSorting",
        source_v1ca1_git_commit=source_v1ca1_git_commit,
        source_spyglass_git_commit=source_spyglass_git_commit,
        overwrite=False,
    )
    _validate_swap_tuning_curve_comparison_upstream_link(
        registered["upstream_provenance"],
        selection,
    )
    del artifact_root
    return _write_swap_tuning_curve_comparison_nwb(
        nwb_file_name=str(selection["nwb_file_name"]),
        result=registered,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )


def _new_schema(schema_factory: Callable[..., Any], context: dict[str, Any]) -> Any:
    """Construct one schema while supporting minimal injectable factories."""
    try:
        return schema_factory(context=context)
    except TypeError:
        schema = schema_factory()
        if hasattr(schema, "context"):
            schema.context = context
    return schema


def _validate_analysis_schema_prefix(
    dj_module: Any,
    analysis_nwbfile_schema_name: str,
) -> None:
    """Fail before DDL when Spyglass's configured custom prefix cannot match."""
    if analysis_nwbfile_schema_name.count("_") != 1:
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    expected_prefix, suffix = analysis_nwbfile_schema_name.split("_", 1)
    if suffix != "nwbfile":
        raise ValueError(
            "AnalysisNwbfile schema must have the form '<prefix>_nwbfile'."
        )
    custom_config = dj_module.config.get("custom", {})
    configured_prefix = custom_config.get("database.prefix")
    if configured_prefix != expected_prefix:
        raise ValueError(
            "Spyglass custom AnalysisNwbfile activation requires "
            "dj.config['custom']['database.prefix'] to equal "
            f"{expected_prefix!r}; found {configured_prefix!r}."
        )


def _construct_tables(
    *,
    dj_module: Any,
    session_table: Any,
    nwbfile_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    spyglass_mixin: type,
    spyglass_analysis: type,
    schema_factory: Callable[..., Any],
    schema_name: str,
    analysis_nwbfile_schema_name: str,
    connection: Any,
    create_schema: bool,
    create_tables: bool,
    runtime_hooks: Mapping[str, Callable[..., Any]] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Build and decorate tables from injected DataJoint-like dependencies."""
    runtime_hooks = dict(runtime_hooks or {})
    ripple_compute_hook = runtime_hooks.get(
        "ripple_modulation_compute",
        runtime_hooks.get("compute", _make_ripple_modulation_row),
    )
    ripple_register_hook = runtime_hooks.get(
        "ripple_modulation_register_existing",
        runtime_hooks.get(
            "register_existing",
            _register_existing_ripple_modulation_row,
        ),
    )
    movement_compute_hook = runtime_hooks.get(
        "movement_firing_rate_compute",
        _make_movement_firing_rate_row,
    )
    epoch_motor_behavior_compute_hook = runtime_hooks.get(
        "epoch_motor_behavior_compute",
        _make_epoch_motor_behavior_row,
    )
    epoch_motor_behavior_register_hook = runtime_hooks.get(
        "epoch_motor_behavior_register_existing",
        _register_existing_epoch_motor_behavior_row,
    )
    cv_pca_compute_hook = runtime_hooks.get(
        "cv_pca_compute",
        _make_cv_pca_row,
    )
    cv_pca_register_hook = runtime_hooks.get(
        "cv_pca_register_existing",
        _register_existing_cv_pca_row,
    )
    tuning_curve_compute_hook = runtime_hooks.get(
        "path_specific_place_tuning_curve_compute",
        _make_path_specific_place_tuning_curve_row,
    )
    tuning_curve_register_hook = runtime_hooks.get(
        "path_specific_place_tuning_curve_register_existing",
        _register_existing_path_specific_place_tuning_curve_row,
    )
    tuning_similarity_compute_hook = runtime_hooks.get(
        "path_specific_place_tuning_similarity_compute",
        _make_path_specific_place_tuning_similarity_row,
    )
    tuning_similarity_register_hook = runtime_hooks.get(
        "path_specific_place_tuning_similarity_register_existing",
        _register_existing_path_specific_place_tuning_similarity_row,
    )
    dpp_tuning_curve_compute_hook = runtime_hooks.get(
        "dpp_tuning_curve_compute",
        _make_dpp_tuning_curve_row,
    )
    dpp_tuning_curve_register_hook = runtime_hooks.get(
        "dpp_tuning_curve_register_existing",
        _register_existing_dpp_tuning_curve_row,
    )
    stability_compute_hook = runtime_hooks.get(
        "path_specific_place_stability_compute",
        _make_path_specific_place_stability_row,
    )
    stability_register_hook = runtime_hooks.get(
        "path_specific_place_stability_register_existing",
        _register_existing_path_specific_place_stability_row,
    )
    dpp_encoding_compute_hook = runtime_hooks.get(
        "dpp_encoding_compute",
        _make_dpp_encoding_row,
    )
    dpp_encoding_register_hook = runtime_hooks.get(
        "dpp_encoding_register_existing",
        _register_existing_dpp_encoding_row,
    )
    path_progression_decoding_compute_hook = runtime_hooks.get(
        "path_progression_decoding_compute",
        _make_path_progression_decoding_row,
    )
    path_specific_place_decoding_compute_hook = runtime_hooks.get(
        "path_specific_place_decoding_compute",
        _make_path_specific_place_decoding_row,
    )
    path_specific_place_decoding_register_hook = runtime_hooks.get(
        "path_specific_place_decoding_register_existing",
        _register_existing_path_specific_place_decoding_row,
    )
    motor_encoding_compute_hook = runtime_hooks.get(
        "motor_encoding_compute",
        _make_motor_encoding_row,
    )
    motor_encoding_register_hook = runtime_hooks.get(
        "motor_encoding_register_existing",
        _register_existing_motor_encoding_row,
    )
    dark_light_glm_compute_hook = runtime_hooks.get(
        "dark_light_glm_compute",
        _make_dark_light_glm_row,
    )
    dark_light_glm_register_hook = runtime_hooks.get(
        "dark_light_glm_register_existing",
        _register_existing_dark_light_glm_row,
    )
    swap_glm_compute_hook = runtime_hooks.get(
        "swap_glm_compute",
        _make_swap_glm_row,
    )
    swap_glm_register_hook = runtime_hooks.get(
        "swap_glm_register_existing",
        _register_existing_swap_glm_row,
    )
    swap_tuning_curve_comparison_compute_hook = runtime_hooks.get(
        "swap_tuning_curve_comparison_compute",
        _make_swap_tuning_curve_comparison_row,
    )
    swap_tuning_curve_comparison_register_hook = runtime_hooks.get(
        "swap_tuning_curve_comparison_register_existing",
        _register_existing_swap_tuning_curve_comparison_row,
    )
    ripple_glm_compute_hook = runtime_hooks.get(
        "ripple_glm_compute",
        _make_ripple_glm_row,
    )
    ripple_glm_register_hook = runtime_hooks.get(
        "ripple_glm_register_existing",
        _register_existing_ripple_glm_row,
    )
    ripple_cross_region_xcorr_compute_hook = runtime_hooks.get(
        "ripple_cross_region_xcorr_compute",
        _make_ripple_cross_region_xcorr_row,
    )
    ripple_cross_region_xcorr_register_hook = runtime_hooks.get(
        "ripple_cross_region_xcorr_register_existing",
        _register_existing_ripple_cross_region_xcorr_row,
    )
    if not all(
        callable(hook)
        for hook in (
            ripple_compute_hook,
            ripple_register_hook,
            movement_compute_hook,
            epoch_motor_behavior_compute_hook,
            epoch_motor_behavior_register_hook,
            cv_pca_compute_hook,
            cv_pca_register_hook,
            tuning_curve_compute_hook,
            tuning_curve_register_hook,
            tuning_similarity_compute_hook,
            tuning_similarity_register_hook,
            dpp_tuning_curve_compute_hook,
            dpp_tuning_curve_register_hook,
            stability_compute_hook,
            stability_register_hook,
            dpp_encoding_compute_hook,
            dpp_encoding_register_hook,
            path_progression_decoding_compute_hook,
            path_specific_place_decoding_compute_hook,
            path_specific_place_decoding_register_hook,
            motor_encoding_compute_hook,
            motor_encoding_register_hook,
            dark_light_glm_compute_hook,
            dark_light_glm_register_hook,
            swap_glm_compute_hook,
            swap_glm_register_hook,
            swap_tuning_curve_comparison_compute_hook,
            swap_tuning_curve_comparison_register_hook,
            ripple_glm_compute_hook,
            ripple_glm_register_hook,
            ripple_cross_region_xcorr_compute_hook,
            ripple_cross_region_xcorr_register_hook,
        )
    ):
        raise TypeError("Analysis runtime hooks must be callable.")

    main_context: dict[str, Any] = {
        "Session": session_table,
        "SortedSpikesGroup": sorted_spikes_group,
        "UnitSelectionParams": unit_selection_params,
        "SpikeSortingOutput": spike_sorting_output,
    }
    main_schema = _new_schema(schema_factory, main_context)
    main_schema.activate(
        schema_name,
        connection=connection,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=main_context,
    )

    analysis_context = {"Nwbfile": nwbfile_table}
    analysis_schema = _new_schema(schema_factory, analysis_context)
    analysis_schema.activate(
        analysis_nwbfile_schema_name,
        connection=(
            connection
            if connection is not None
            else getattr(main_schema, "connection", None)
        ),
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=analysis_context,
    )

    class AnalysisNwbfile(spyglass_analysis, dj_module.Manual):
        definition = table_specs.ANALYSIS_NWBFILE_DEFINITION

        def _register_table(self) -> None:
            """Suppress Spyglass's registry insert during DDL-only activation."""
            return None

        def register_with_spyglass(self) -> None:
            """Explicitly add this table to Spyglass's AnalysisRegistry."""
            spyglass_analysis._register_table(self)

    AnalysisNwbfile = analysis_schema(AnalysisNwbfile)
    main_context["AnalysisNwbfile"] = AnalysisNwbfile

    class EpochIntervals(spyglass_mixin, dj_module.Manual):
        definition = table_specs.EPOCH_INTERVALS_DEFINITION

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load one epoch's ephys-reference interval from its NWB file."""
            from v1ca1.spyglass.nwb import load_interval_set

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_interval_set,
            )

    EpochIntervals = main_schema(EpochIntervals)
    main_context["EpochIntervals"] = EpochIntervals

    class TrajectoryIntervals(spyglass_mixin, dj_module.Manual):
        definition = table_specs.TRAJECTORY_INTERVALS_DEFINITION

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load all laps for one epoch and trajectory type."""
            from v1ca1.spyglass.nwb import load_interval_set

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_interval_set,
            )

    TrajectoryIntervals = main_schema(TrajectoryIntervals)
    main_context["TrajectoryIntervals"] = TrajectoryIntervals

    class RippleIntervals(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_INTERVALS_DEFINITION

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load detector-qualified, speed-gated ripples for one epoch."""
            from v1ca1.spyglass.nwb import load_interval_set

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_interval_set,
            )

    RippleIntervals = main_schema(RippleIntervals)
    main_context["RippleIntervals"] = RippleIntervals

    class Position(spyglass_mixin, dj_module.Manual):
        definition = table_specs.POSITION_DEFINITION

        @classmethod
        def load_position(
            cls,
            key: Mapping[str, Any],
            *,
            apply_analysis_offset: bool = True,
        ) -> Any:
            """Load one explicitly named epoch position series in centimeters."""
            from v1ca1.spyglass.nwb import load_position

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_position,
                loader_kwargs={"apply_analysis_offset": apply_analysis_offset},
            )

    Position = main_schema(Position)
    main_context["Position"] = Position

    class WTrackGraph(spyglass_mixin, dj_module.Manual):
        definition = table_specs.WTRACK_GRAPH_DEFINITION

        @classmethod
        def load_graph(cls, key: Mapping[str, Any]) -> dict[str, Any]:
            """Load track-linearization graph inputs in centimeters."""
            from v1ca1.spyglass.nwb import load_wtrack_graph

            return _load_catalog_nwb_object(
                cls,
                key,
                nwbfile_table=nwbfile_table,
                loader=load_wtrack_graph,
            )

    WTrackGraph = main_schema(WTrackGraph)
    main_context["WTrackGraph"] = WTrackGraph

    class SpikeSortingFigurl(spyglass_mixin, dj_module.Manual):
        definition = table_specs.SPIKE_SORTING_FIGURL_DEFINITION

        @classmethod
        def get_url(cls, key: Mapping[str, Any]) -> str:
            """Return the indexed spike-sorting FigURL for one probe/shank."""
            return str((cls & dict(key)).fetch1("figurl_url"))

    SpikeSortingFigurl = main_schema(SpikeSortingFigurl)
    main_context["SpikeSortingFigurl"] = SpikeSortingFigurl

    class RegionSortedSpikesGroup(spyglass_mixin, dj_module.Manual):
        definition = table_specs.REGION_SORTED_SPIKES_GROUP_DEFINITION

        @classmethod
        def register_regions(
            cls,
            key: Mapping[str, Any],
            *,
            region_names: tuple[str, ...] = ("v1", "ca1"),
            skip_duplicates: bool = False,
        ) -> list[dict[str, Any]]:
            """Register nonempty logical region views of one sorted group."""
            from v1ca1.spyglass.region_sorted_spikes import (
                build_region_sorted_spikes_group_row,
                normalize_region,
            )

            normalized_regions = tuple(
                normalize_region(region) for region in region_names
            )
            if not normalized_regions:
                raise ValueError("region_names must contain at least one region.")
            if len(set(normalized_regions)) != len(normalized_regions):
                raise ValueError("region_names must contain unique values.")
            group_key = _sorted_spikes_group_key(key)
            rows: list[dict[str, Any]] = []
            for region_name in normalized_regions:
                loaded = _load_group_unit_data(
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    key=group_key,
                    region=region_name,
                    allow_empty=True,
                )
                if int(loaded["n_units"]) == 0:
                    continue
                row = build_region_sorted_spikes_group_row(loaded)
                rows.append(row)
            if not rows:
                raise ValueError(
                    "None of the requested regions contains units in the "
                    "selected SortedSpikesGroup."
                )
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

        @classmethod
        def load_spikes(
            cls,
            key: Mapping[str, Any],
            *,
            time_support: Any | tuple[float, float] | None = None,
        ) -> dict[str, Any]:
            """Reload and verify one registered regional sorting view."""
            from v1ca1.spyglass.region_sorted_spikes import (
                reload_region_sorted_spikes_group,
            )

            row = _fetch1_dict(cls, key)
            return reload_region_sorted_spikes_group(
                row,
                sorted_spikes_group=sorted_spikes_group,
                unit_selection_params=unit_selection_params,
                spike_sorting_output=spike_sorting_output,
                time_support=time_support,
            )

    RegionSortedSpikesGroup = main_schema(RegionSortedSpikesGroup)
    main_context["RegionSortedSpikesGroup"] = RegionSortedSpikesGroup

    class MovementParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.MOVEMENT_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one shared movement definition."""
            validated = _validate_movement_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_default(cls, *, skip_duplicates: bool = True) -> dict[str, Any]:
            """Explicitly insert the canonical movement parameters."""
            return cls.insert_parameters(
                table_specs.DEFAULT_MOVEMENT_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    MovementParameters = main_schema(MovementParameters)
    main_context["MovementParameters"] = MovementParameters

    class EpochMotorBehaviorParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.EPOCH_MOTOR_BEHAVIOR_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one motor progression-bin definition."""
            validated = _validate_epoch_motor_behavior_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_default(
            cls, *, skip_duplicates: bool = True
        ) -> dict[str, Any]:
            """Explicitly insert the manuscript four-centimeter preset."""
            return cls.insert_parameters(
                table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    EpochMotorBehaviorParameters = main_schema(EpochMotorBehaviorParameters)
    main_context["EpochMotorBehaviorParameters"] = EpochMotorBehaviorParameters

    class EpochMotorBehaviorSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.EPOCH_MOTOR_BEHAVIOR_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, UUID, and insert one run-epoch selection."""
            row = _epoch_motor_behavior_selection_row(
                key=key,
                epoch_intervals_table=EpochIntervals,
                position_table=Position,
                movement_parameters_table=MovementParameters,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                parameters_table=EpochMotorBehaviorParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    EpochMotorBehaviorSelection = main_schema(EpochMotorBehaviorSelection)
    main_context["EpochMotorBehaviorSelection"] = EpochMotorBehaviorSelection

    class EpochMotorBehavior(spyglass_mixin, dj_module.Computed):
        definition = table_specs.EPOCH_MOTOR_BEHAVIOR_DEFINITION
        _compute_hook = staticmethod(epoch_motor_behavior_compute_hook)
        _register_existing_hook = staticmethod(
            epoch_motor_behavior_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one motor-behavior analysis NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "EpochMotorBehavior.make() must run through populate() so "
                    "the AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(EpochMotorBehaviorSelection, key)
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=EpochMotorBehaviorParameters,
                    epoch_intervals_table=EpochIntervals,
                    position_table=Position,
                    movement_parameters_table=MovementParameters,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "epoch_motor_behavior_id": selection[
                            "epoch_motor_behavior_id"
                        ],
                        **artifact_row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_epoch_motor_behavior_bundle(
            cls, key: Mapping[str, Any]
        ) -> dict[str, Any]:
            """Load and verify one canonical motor-behavior analysis NWB."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                EpochMotorBehaviorSelection,
                {
                    "epoch_motor_behavior_id": row[
                        "epoch_motor_behavior_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                EpochMotorBehaviorParameters, selection
            )
            movement_parameters = _fetch1_dict(
                MovementParameters, selection
            )
            animal_name, session_date = _session_identity(
                session_table, selection
            )
            return _load_epoch_motor_behavior_result(
                result_row=row,
                epoch_motor_behavior_table=cls,
                selection_row=selection,
                parameters_row=parameters,
                movement_parameters_row=movement_parameters,
                animal_name=animal_name,
                date=session_date,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_distribution_path: Path | str,
            source_progression_path: Path | str,
            source_run_log_path: Path | str | None = None,
            overwrite: bool = False,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Strictly recompute and register one legacy session artifact."""
            if overwrite:
                raise ValueError(
                    "EpochMotorBehavior results are immutable; create a new "
                    "selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(EpochMotorBehaviorSelection, key)
                result_key = {
                    "epoch_motor_behavior_id": selection[
                        "epoch_motor_behavior_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "EpochMotorBehavior already contains this immutable "
                        "selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_distribution_path=Path(
                            source_distribution_path
                        ),
                        source_progression_path=Path(
                            source_progression_path
                        ),
                        source_run_log_path=(
                            None
                            if source_run_log_path is None
                            else Path(source_run_log_path)
                        ),
                        parameters_table=EpochMotorBehaviorParameters,
                        epoch_intervals_table=EpochIntervals,
                        position_table=Position,
                        movement_parameters_table=MovementParameters,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        session_table=session_table,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    **result_key,
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    EpochMotorBehavior = main_schema(EpochMotorBehavior)
    main_context["EpochMotorBehavior"] = EpochMotorBehavior

    class MovementFiringRateSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.MOVEMENT_FIRING_RATE_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one movement selection."""
            row = _movement_firing_rate_selection_row(
                key=key,
                position_table=Position,
                parameters_table=MovementParameters,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    MovementFiringRateSelection = main_schema(MovementFiringRateSelection)
    main_context["MovementFiringRateSelection"] = MovementFiringRateSelection

    class MovementFiringRate(spyglass_mixin, dj_module.Computed):
        definition = table_specs.MOVEMENT_FIRING_RATE_DEFINITION
        _compute_hook = staticmethod(movement_compute_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one movement artifact bundle."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "MovementFiringRate.make() must run through populate() so "
                    "the AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(MovementFiringRateSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=MovementParameters,
                    epoch_intervals_table=EpochIntervals,
                    position_table=Position,
                    session_table=session_table,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "movement_firing_rate_id": selection[
                            "movement_firing_rate_id"
                        ],
                        **row,
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_firing_rates(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one all-unit movement-rate NWB table."""
            row = _fetch1_dict(cls, key)
            return _load_movement_result_artifacts(
                result_row=row,
                movement_firing_rate_table=cls,
            )["table"]

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one exact movement-support NWB object."""
            row = _fetch1_dict(cls, key)
            return _load_movement_result_artifacts(
                result_row=row,
                movement_firing_rate_table=cls,
            )["movement_intervals"]

    MovementFiringRate = main_schema(MovementFiringRate)
    main_context["MovementFiringRate"] = MovementFiringRate

    class CVPCAParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.CV_PCA_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one named cvPCA parameter row."""
            validated = _validate_cv_pca_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls, *, skip_duplicates: bool = True
        ) -> list[dict[str, Any]]:
            """Explicitly insert the V1 and CA1 manuscript seed-47 rows."""
            return [
                cls.insert_parameters(
                    parameters,
                    skip_duplicates=skip_duplicates,
                )
                for parameters in table_specs.CV_PCA_PARAMETER_PRESETS
            ]

    CVPCAParameters = main_schema(CVPCAParameters)
    main_context["CVPCAParameters"] = CVPCAParameters

    class CVPCASelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.CV_PCA_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, UUID, and insert one cvPCA selection."""
            row = _cv_pca_selection_row(
                key=key,
                epoch_intervals_table=EpochIntervals,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=MovementFiringRateSelection,
                movement_parameters_table=MovementParameters,
                position_table=Position,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                parameters_table=CVPCAParameters,
                session_table=session_table,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    CVPCASelection = main_schema(CVPCASelection)
    main_context["CVPCASelection"] = CVPCASelection

    class CVPCA(spyglass_mixin, dj_module.Computed):
        definition = table_specs.CV_PCA_DEFINITION
        _compute_hook = staticmethod(cv_pca_compute_hook)
        _register_existing_hook = staticmethod(cv_pca_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one immutable cvPCA NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "CVPCA.make() must run through populate() so "
                    "AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(CVPCASelection, key)
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=CVPCAParameters,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "cv_pca_id": selection["cv_pca_id"],
                        **artifact_row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_cv_pca_bundle(
            cls, key: Mapping[str, Any]
        ) -> dict[str, Any]:
            """Fetch and verify one canonical cvPCA NWB result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                CVPCASelection, {"cv_pca_id": row["cv_pca_id"]}
            )
            parameters = _fetch1_dict(CVPCAParameters, selection)
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table, selection
            )
            return _load_cv_pca_result(
                result_row=row,
                result_table=cls,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            legacy_result_path: Path | str,
            legacy_summary_path: Path | str,
            overwrite: bool = False,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Strictly recompute, compare, and register one legacy pair."""
            if overwrite:
                raise ValueError(
                    "CVPCA results are immutable; create a new selection "
                    "instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(CVPCASelection, key)
                result_key = {"cv_pca_id": selection["cv_pca_id"]}
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "CVPCA already contains this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        legacy_result_path=Path(legacy_result_path),
                        legacy_summary_path=Path(legacy_summary_path),
                        parameters_table=CVPCAParameters,
                        epoch_intervals_table=EpochIntervals,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        position_table=Position,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        session_table=session_table,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    **result_key,
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    CVPCA = main_schema(CVPCA)
    main_context["CVPCA"] = CVPCA

    class RippleModulationParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_MODULATION_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one scalar parameter set."""
            validated = _validate_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_default(cls, *, skip_duplicates: bool = True) -> dict[str, Any]:
            """Explicitly insert the canonical no-extra-threshold parameters."""
            return cls.insert_parameters(
                table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    RippleModulationParameters = main_schema(RippleModulationParameters)
    main_context["RippleModulationParameters"] = RippleModulationParameters

    class RippleModulationSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_MODULATION_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one selection."""
            row = _ripple_modulation_selection_row(
                key=key,
                ripples_table=RippleIntervals,
                epoch_intervals_table=EpochIntervals,
                parameters_table=RippleModulationParameters,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    RippleModulationSelection = main_schema(RippleModulationSelection)
    main_context["RippleModulationSelection"] = RippleModulationSelection

    class RippleModulation(spyglass_mixin, dj_module.Computed):
        definition = table_specs.RIPPLE_MODULATION_DEFINITION
        _compute_hook = staticmethod(ripple_compute_hook)
        _register_existing_hook = staticmethod(ripple_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and register one selected analysis NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "RippleModulation.make() must run through populate() so "
                    "the AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(RippleModulationSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleModulationParameters,
                    ripples_table=RippleIntervals,
                    epoch_intervals_table=EpochIntervals,
                    session_table=session_table,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "ripple_modulation_id": selection[
                            "ripple_modulation_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_artifacts(cls, key: Mapping[str, Any]) -> dict[str, Any]:
            """Load and validate both RippleModulation NWB tables."""
            row = _fetch1_dict(cls, key)
            return _load_ripple_modulation_result(
                result_row=row,
                ripple_modulation_table=cls,
            )

        @classmethod
        def load_summary(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one per-unit ripple-modulation summary."""
            return cls.load_artifacts(key)["summary"]

        @classmethod
        def load_peri_ripple_firing_rate(
            cls,
            key: Mapping[str, Any],
        ) -> Any:
            """Load and validate one mean peri-ripple firing-rate table."""
            return cls.load_artifacts(key)["peri_ripple_firing_rate"]

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            summary_path: Path | str,
            peri_ripple_firing_rate_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize keyed legacy Parquets into one analysis NWB."""
            if overwrite:
                raise ValueError(
                    "Registered RippleModulation results are immutable; create "
                    "a new selection instead of overwriting an artifact."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(RippleModulationSelection, key)
                result_key = {
                    "ripple_modulation_id": selection["ripple_modulation_id"]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "RippleModulation already contains this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        summary_path=Path(summary_path),
                        peri_ripple_firing_rate_path=Path(
                            peri_ripple_firing_rate_path
                        ),
                        overwrite=False,
                        parameters_table=RippleModulationParameters,
                        ripples_table=RippleIntervals,
                        session_table=session_table,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        spike_sorting_output=spike_sorting_output,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=source_spyglass_git_commit,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "ripple_modulation_id": selection["ripple_modulation_id"],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    RippleModulation = main_schema(RippleModulation)
    main_context["RippleModulation"] = RippleModulation

    class TuningCurveParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.TUNING_CURVE_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one tuning-curve parameter set."""
            validated = _validate_tuning_curve_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the documented legacy and Figure 1D presets."""
            return [
                cls.insert_parameters(
                    parameters,
                    skip_duplicates=skip_duplicates,
                )
                for parameters in table_specs.TUNING_CURVE_PARAMETER_PRESETS
            ]

    TuningCurveParameters = main_schema(TuningCurveParameters)
    main_context["TuningCurveParameters"] = TuningCurveParameters

    class TuningSimilarityParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.TUNING_SIMILARITY_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one fixed tuning-similarity metric."""
            validated = _validate_tuning_similarity_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the three documented similarity metrics."""
            return [
                cls.insert_parameters(
                    parameters,
                    skip_duplicates=skip_duplicates,
                )
                for parameters in table_specs.TUNING_SIMILARITY_PARAMETER_PRESETS
            ]

    TuningSimilarityParameters = main_schema(TuningSimilarityParameters)
    main_context["TuningSimilarityParameters"] = TuningSimilarityParameters

    class PathSpecificPlaceTuningCurveSelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.PATH_SPECIFIC_PLACE_TUNING_CURVE_SELECTION_DEFINITION
        )

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one curve selection."""
            row = _path_specific_place_tuning_curve_selection_row(
                key=key,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                parameters_table=TuningCurveParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    PathSpecificPlaceTuningCurveSelection = main_schema(
        PathSpecificPlaceTuningCurveSelection
    )
    main_context["PathSpecificPlaceTuningCurveSelection"] = (
        PathSpecificPlaceTuningCurveSelection
    )

    class PathSpecificPlaceTuningCurve(spyglass_mixin, dj_module.Computed):
        definition = table_specs.PATH_SPECIFIC_PLACE_TUNING_CURVE_DEFINITION
        _compute_hook = staticmethod(tuning_curve_compute_hook)
        _register_existing_hook = staticmethod(tuning_curve_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one selected tuning-curve NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "PathSpecificPlaceTuningCurve.make() must run through "
                    "populate() so the AnalysisNwbfile and result rows share "
                    "one transaction."
                )
            selection = _fetch1_dict(
                PathSpecificPlaceTuningCurveSelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=TuningCurveParameters,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    position_table=Position,
                    wtrack_graph_table=WTrackGraph,
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "path_specific_place_tuning_curve_id": selection[
                            "path_specific_place_tuning_curve_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_tuning_curve(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one canonical tuning-curve DataArray."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                PathSpecificPlaceTuningCurveSelection,
                {
                    "path_specific_place_tuning_curve_id": row[
                        "path_specific_place_tuning_curve_id"
                    ]
                },
            )
            return _load_path_specific_place_tuning_curve_result(
                result_row=row,
                tuning_curve_table=cls,
                selection_row=selection,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            tuning_curve_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one matching legacy all-trial NetCDF and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered PathSpecificPlaceTuningCurve results are "
                    "immutable; create a new selection instead of overwriting "
                    "an artifact."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(
                    PathSpecificPlaceTuningCurveSelection,
                    key,
                )
                result_key = {
                    "path_specific_place_tuning_curve_id": selection[
                        "path_specific_place_tuning_curve_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "PathSpecificPlaceTuningCurve already contains this "
                        "immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        tuning_curve_path=Path(tuning_curve_path),
                        overwrite=False,
                        parameters_table=TuningCurveParameters,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        position_table=Position,
                        wtrack_graph_table=WTrackGraph,
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        session_table=session_table,
                        spike_sorting_output=spike_sorting_output,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=source_spyglass_git_commit,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "path_specific_place_tuning_curve_id": selection[
                        "path_specific_place_tuning_curve_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    PathSpecificPlaceTuningCurve = main_schema(
        PathSpecificPlaceTuningCurve
    )
    main_context["PathSpecificPlaceTuningCurve"] = (
        PathSpecificPlaceTuningCurve
    )

    class PathSpecificPlaceTuningSimilaritySelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.PATH_SPECIFIC_PLACE_TUNING_SIMILARITY_SELECTION_DEFINITION
        )

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one similarity selection."""
            row = _tuning_similarity_selection_row(
                key=key,
                tuning_curve_table=PathSpecificPlaceTuningCurve,
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
                parameters_table=TuningSimilarityParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    PathSpecificPlaceTuningSimilaritySelection = main_schema(
        PathSpecificPlaceTuningSimilaritySelection
    )
    main_context["PathSpecificPlaceTuningSimilaritySelection"] = (
        PathSpecificPlaceTuningSimilaritySelection
    )

    class PathSpecificPlaceTuningSimilarity(
        spyglass_mixin,
        dj_module.Computed,
    ):
        definition = table_specs.PATH_SPECIFIC_PLACE_TUNING_SIMILARITY_DEFINITION
        _compute_hook = staticmethod(tuning_similarity_compute_hook)
        _register_existing_hook = staticmethod(tuning_similarity_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one all-unit similarity artifact."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "PathSpecificPlaceTuningSimilarity.make() must run "
                    "through populate() so the AnalysisNwbfile and result "
                    "rows share one transaction."
                )
            selection = _fetch1_dict(
                PathSpecificPlaceTuningSimilaritySelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=TuningSimilarityParameters,
                    tuning_curve_parameters_table=TuningCurveParameters,
                    tuning_curve_table=PathSpecificPlaceTuningCurve,
                    tuning_curve_selection_table=(
                        PathSpecificPlaceTuningCurveSelection
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "path_specific_place_tuning_similarity_id": selection[
                            "path_specific_place_tuning_similarity_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_similarity(cls, key: Mapping[str, Any]) -> Any:
            """Load and cross-check one all-unit similarity DynamicTable."""
            from v1ca1.spyglass.tuning_similarity import (
                validate_tuning_similarity_against_inputs,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                PathSpecificPlaceTuningSimilaritySelection,
                {
                    "path_specific_place_tuning_similarity_id": row[
                        "path_specific_place_tuning_similarity_id"
                    ]
                },
            )
            inputs = _load_tuning_similarity_inputs(
                key=selection,
                parameters_table=TuningSimilarityParameters,
                tuning_curve_parameters_table=TuningCurveParameters,
                tuning_curve_table=PathSpecificPlaceTuningCurve,
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                movement_parameters_table=MovementParameters,
                region_sorted_spikes_group_table=(
                    RegionSortedSpikesGroup
                ),
                session_table=session_table,
            )
            table = _load_path_specific_place_tuning_similarity_result(
                result_row=row,
                similarity_table=cls,
                similarity_metric=inputs["parameters"]["similarity_metric"],
            )
            validate_tuning_similarity_against_inputs(
                table,
                tuning_curves_by_trajectory=inputs["curves"],
                movement_firing_rate_table=inputs["movement_table"],
                similarity_metric=inputs["parameters"]["similarity_metric"],
            )
            return table

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            similarity_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate one complete all-unit legacy Parquet and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered PathSpecificPlaceTuningSimilarity results are "
                    "immutable; create a new selection instead of overwriting "
                    "an artifact."
                )

            with _transaction_context(cls):
                selection = _fetch1_dict(
                    PathSpecificPlaceTuningSimilaritySelection,
                    key,
                )
                result_key = {
                    "path_specific_place_tuning_similarity_id": selection[
                        "path_specific_place_tuning_similarity_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "PathSpecificPlaceTuningSimilarity already contains "
                        "this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        similarity_path=Path(similarity_path),
                        overwrite=False,
                        parameters_table=TuningSimilarityParameters,
                        tuning_curve_parameters_table=TuningCurveParameters,
                        tuning_curve_table=PathSpecificPlaceTuningCurve,
                        tuning_curve_selection_table=(
                            PathSpecificPlaceTuningCurveSelection
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        session_table=session_table,
                        spike_sorting_output=spike_sorting_output,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "path_specific_place_tuning_similarity_id": selection[
                        "path_specific_place_tuning_similarity_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    PathSpecificPlaceTuningSimilarity = main_schema(
        PathSpecificPlaceTuningSimilarity
    )
    main_context["PathSpecificPlaceTuningSimilarity"] = (
        PathSpecificPlaceTuningSimilarity
    )

    class DPPTuningCurveSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DPP_TUNING_CURVE_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one DPP selection."""
            row = _dpp_tuning_curve_selection_row(
                key=key,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                parameters_table=TuningCurveParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    DPPTuningCurveSelection = main_schema(DPPTuningCurveSelection)
    main_context["DPPTuningCurveSelection"] = DPPTuningCurveSelection

    class DPPTuningCurve(spyglass_mixin, dj_module.Computed):
        definition = table_specs.DPP_TUNING_CURVE_DEFINITION
        _compute_hook = staticmethod(dpp_tuning_curve_compute_hook)
        _register_existing_hook = staticmethod(dpp_tuning_curve_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one selected DPP tuning NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "DPPTuningCurve.make() must run through populate() so "
                    "the AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(DPPTuningCurveSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=TuningCurveParameters,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    position_table=Position,
                    wtrack_graph_table=WTrackGraph,
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "dpp_tuning_curve_id": selection[
                            "dpp_tuning_curve_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_tuning_curve(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one canonical DPP tuning DataArray."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                DPPTuningCurveSelection,
                {"dpp_tuning_curve_id": row["dpp_tuning_curve_id"]},
            )
            return _load_dpp_tuning_curve_result(
                result_row=row,
                tuning_curve_table=cls,
                selection_row=selection,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            tuning_curve_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one matching legacy all-trial DPP and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered DPPTuningCurve results are immutable; create "
                    "a new selection instead of overwriting an artifact."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(DPPTuningCurveSelection, key)
                result_key = {
                    "dpp_tuning_curve_id": selection["dpp_tuning_curve_id"]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "DPPTuningCurve already contains this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        tuning_curve_path=Path(tuning_curve_path),
                        overwrite=False,
                        parameters_table=TuningCurveParameters,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        position_table=Position,
                        wtrack_graph_table=WTrackGraph,
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        session_table=session_table,
                        spike_sorting_output=spike_sorting_output,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "dpp_tuning_curve_id": selection[
                        "dpp_tuning_curve_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    DPPTuningCurve = main_schema(DPPTuningCurve)
    main_context["DPPTuningCurve"] = DPPTuningCurve

    class PathSpecificPlaceStabilitySelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.PATH_SPECIFIC_PLACE_STABILITY_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one stability selection."""
            row = _stability_selection_row(
                key=key,
                tuning_curve_table=PathSpecificPlaceTuningCurve,
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    PathSpecificPlaceStabilitySelection = main_schema(
        PathSpecificPlaceStabilitySelection
    )
    main_context["PathSpecificPlaceStabilitySelection"] = (
        PathSpecificPlaceStabilitySelection
    )

    class PathSpecificPlaceStability(spyglass_mixin, dj_module.Computed):
        definition = table_specs.PATH_SPECIFIC_PLACE_STABILITY_DEFINITION
        _compute_hook = staticmethod(stability_compute_hook)
        _register_existing_hook = staticmethod(stability_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one selected stability artifact."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "PathSpecificPlaceStability.make() must run through "
                    "populate() so the AnalysisNwbfile and result rows share "
                    "one transaction."
                )
            selection = _fetch1_dict(PathSpecificPlaceStabilitySelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    tuning_curve_table=PathSpecificPlaceTuningCurve,
                    tuning_curve_selection_table=(
                        PathSpecificPlaceTuningCurveSelection
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "path_specific_place_stability_id": selection[
                            "path_specific_place_stability_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_stability(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one all-unit stability DynamicTable."""
            row = _fetch1_dict(cls, key)
            return _load_path_specific_place_stability_result(
                result_row=row,
                stability_table=cls,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            stability_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Filter the complete legacy Parquet and insert one result row."""
            if overwrite:
                raise ValueError(
                    "Registered PathSpecificPlaceStability results are immutable; "
                    "create a new selection instead of overwriting an artifact."
                )

            with _transaction_context(cls):
                selection = _fetch1_dict(
                    PathSpecificPlaceStabilitySelection,
                    key,
                )
                result_key = {
                    "path_specific_place_stability_id": selection[
                        "path_specific_place_stability_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "PathSpecificPlaceStability already contains this "
                        "immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        stability_path=Path(stability_path),
                        overwrite=False,
                        tuning_curve_table=PathSpecificPlaceTuningCurve,
                        tuning_curve_selection_table=(
                            PathSpecificPlaceTuningCurveSelection
                        ),
                        tuning_curve_parameters_table=TuningCurveParameters,
                        movement_parameters_table=MovementParameters,
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        session_table=session_table,
                        spike_sorting_output=spike_sorting_output,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "path_specific_place_stability_id": selection[
                        "path_specific_place_stability_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    PathSpecificPlaceStability = main_schema(PathSpecificPlaceStability)
    main_context["PathSpecificPlaceStability"] = PathSpecificPlaceStability

    class DPPEncodingParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DPP_ENCODING_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one encoding-comparison parameter row."""
            validated = _validate_dpp_encoding_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the canonical encoding presets."""
            return [
                cls.insert_parameters(
                    preset,
                    skip_duplicates=skip_duplicates,
                )
                for preset in table_specs.DPP_ENCODING_PARAMETER_PRESETS
            ]

        @classmethod
        def insert_default(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> dict[str, Any]:
            """Explicitly insert the manuscript 50-ms preset."""
            return cls.insert_parameters(
                table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    DPPEncodingParameters = main_schema(
        DPPEncodingParameters
    )
    main_context["DPPEncodingParameters"] = (
        DPPEncodingParameters
    )

    class DPPEncodingSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DPP_ENCODING_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one comparison."""
            row = _dpp_encoding_selection_row(
                key=key,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                stability_table=PathSpecificPlaceStability,
                stability_selection_table=(
                    PathSpecificPlaceStabilitySelection
                ),
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
                parameters_table=DPPEncodingParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    DPPEncodingSelection = main_schema(
        DPPEncodingSelection
    )
    main_context["DPPEncodingSelection"] = (
        DPPEncodingSelection
    )

    class DPPEncoding(spyglass_mixin, dj_module.Computed):
        definition = table_specs.DPP_ENCODING_DEFINITION
        _compute_hook = staticmethod(dpp_encoding_compute_hook)
        _register_existing_hook = staticmethod(
            dpp_encoding_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one four-model comparison."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "DPPEncoding.make() must run through populate() so the "
                    "AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(DPPEncodingSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=DPPEncodingParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    epoch_intervals_table=EpochIntervals,
                    position_table=Position,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    stability_table=PathSpecificPlaceStability,
                    stability_selection_table=(
                        PathSpecificPlaceStabilitySelection
                    ),
                    tuning_curve_selection_table=(
                        PathSpecificPlaceTuningCurveSelection
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "dpp_encoding_id": selection[
                            "dpp_encoding_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_dpp_encoding(
            cls,
            key: Mapping[str, Any],
        ) -> Any:
            """Load and validate one canonical comparison DynamicTable."""

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                DPPEncodingSelection,
                {
                    "dpp_encoding_id": row[
                        "dpp_encoding_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                DPPEncodingParameters,
                {
                    "dpp_encoding_param_name": selection[
                        "dpp_encoding_param_name"
                    ]
                },
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            table = _load_dpp_encoding_result(
                result_row=row,
                dpp_encoding_table=cls,
            )
            _validate_dpp_encoding_artifact_link(
                table=table,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )
            return table

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            dpp_encoding_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one exact-coverage legacy artifact and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered DPPEncoding results are immutable; "
                    "create a new selection instead of overwriting an artifact."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(DPPEncodingSelection, key)
                result_key = {
                    "dpp_encoding_id": selection[
                        "dpp_encoding_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "DPPEncoding already contains this immutable "
                        "selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        dpp_encoding_path=Path(
                            dpp_encoding_path
                        ),
                        overwrite=False,
                        parameters_table=DPPEncodingParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        stability_table=PathSpecificPlaceStability,
                        stability_selection_table=(
                            PathSpecificPlaceStabilitySelection
                        ),
                        tuning_curve_selection_table=(
                            PathSpecificPlaceTuningCurveSelection
                        ),
                        session_table=session_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "dpp_encoding_id": selection[
                        "dpp_encoding_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    DPPEncoding = main_schema(DPPEncoding)
    main_context["DPPEncoding"] = DPPEncoding

    class PathProgressionDecodingParameters(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.PATH_PROGRESSION_DECODING_PARAMETERS_DEFINITION
        )

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one cross-path decoding parameter row."""
            validated = _validate_path_progression_decoding_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the canonical decoding presets."""
            return [
                cls.insert_parameters(
                    preset,
                    skip_duplicates=skip_duplicates,
                )
                for preset in (
                    table_specs.PATH_PROGRESSION_DECODING_PARAMETER_PRESETS
                )
            ]

        @classmethod
        def insert_default(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> dict[str, Any]:
            """Explicitly insert the manuscript-compatible decoding preset."""
            return cls.insert_parameters(
                table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    PathProgressionDecodingParameters = main_schema(
        PathProgressionDecodingParameters
    )
    main_context["PathProgressionDecodingParameters"] = (
        PathProgressionDecodingParameters
    )

    class PathProgressionDecodingSelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = table_specs.PATH_PROGRESSION_DECODING_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one decoding selection."""
            row = _path_progression_decoding_selection_row(
                key=key,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                position_table=Position,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                stability_table=PathSpecificPlaceStability,
                stability_selection_table=(
                    PathSpecificPlaceStabilitySelection
                ),
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
                parameters_table=PathProgressionDecodingParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    PathProgressionDecodingSelection = main_schema(
        PathProgressionDecodingSelection
    )
    main_context["PathProgressionDecodingSelection"] = (
        PathProgressionDecodingSelection
    )

    class PathProgressionDecoding(
        spyglass_mixin,
        dj_module.Computed,
    ):
        definition = table_specs.PATH_PROGRESSION_DECODING_DEFINITION
        _compute_hook = staticmethod(path_progression_decoding_compute_hook)

        class Transfer(spyglass_mixin, dj_module.Part):
            definition = (
                table_specs.PATH_PROGRESSION_DECODING_TRANSFER_DEFINITION
            )
            _nwb_table = AnalysisNwbfile

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one shared-cohort decoder NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "PathProgressionDecoding.make() must run through "
                    "populate() so AnalysisNwbfile, parent, and transfer "
                    "rows share one transaction."
                )
            selection = _fetch1_dict(
                PathProgressionDecodingSelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=PathProgressionDecodingParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    stability_table=PathSpecificPlaceStability,
                    stability_selection_table=(
                        PathSpecificPlaceStabilitySelection
                    ),
                    tuning_curve_selection_table=(
                        PathSpecificPlaceTuningCurveSelection
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            transfer_rows = list(row.pop("_transfer_rows", ()))
            parent_row = {
                "path_progression_decoding_id": selection[
                    "path_progression_decoding_id"
                ],
                **row,
                "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                "runtime_spyglass_git_commit": _spyglass_git_commit(),
            }
            try:
                self.insert1(parent_row)
                if transfer_rows:
                    self.Transfer.insert(
                        [
                            {
                                "path_progression_decoding_id": selection[
                                    "path_progression_decoding_id"
                                ],
                                "analysis_file_name": row[
                                    "analysis_file_name"
                                ],
                                **transfer_row,
                            }
                            for transfer_row in transfer_rows
                        ]
                    )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_decoding_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Fetch and validate one canonical cross-path decoding result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                PathProgressionDecodingSelection,
                {
                    "path_progression_decoding_id": row[
                        "path_progression_decoding_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                PathProgressionDecodingParameters,
                {
                    "path_progression_decoding_param_name": selection[
                        "path_progression_decoding_param_name"
                    ]
                },
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            return _load_path_progression_decoding_result(
                result_row=row,
                decoding_table=cls,
                transfer_table=cls.Transfer,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )

    PathProgressionDecoding = main_schema(
        PathProgressionDecoding
    )
    main_context["PathProgressionDecoding"] = (
        PathProgressionDecoding
    )

    class PathSpecificPlaceDecodingParameters(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.PATH_SPECIFIC_PLACE_DECODING_PARAMETERS_DEFINITION
        )

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one within-epoch decoder parameter row."""
            validated = _validate_path_specific_place_decoding_parameter_row(
                row
            )
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the canonical place-decoder presets."""
            return [
                cls.insert_parameters(
                    preset,
                    skip_duplicates=skip_duplicates,
                )
                for preset in (
                    table_specs.PATH_SPECIFIC_PLACE_DECODING_PARAMETER_PRESETS
                )
            ]

        @classmethod
        def insert_default(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> dict[str, Any]:
            """Explicitly insert the manuscript place-decoder preset."""
            return cls.insert_parameters(
                table_specs.MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    PathSpecificPlaceDecodingParameters = main_schema(
        PathSpecificPlaceDecodingParameters
    )
    main_context["PathSpecificPlaceDecodingParameters"] = (
        PathSpecificPlaceDecodingParameters
    )

    class PathSpecificPlaceDecodingSelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.PATH_SPECIFIC_PLACE_DECODING_SELECTION_DEFINITION
        )

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one place decoder."""
            row = _path_specific_place_decoding_selection_row(
                key=key,
                region_sorted_spikes_group_table=(
                    RegionSortedSpikesGroup
                ),
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                position_table=Position,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                parameters_table=PathSpecificPlaceDecodingParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    PathSpecificPlaceDecodingSelection = main_schema(
        PathSpecificPlaceDecodingSelection
    )
    main_context["PathSpecificPlaceDecodingSelection"] = (
        PathSpecificPlaceDecodingSelection
    )

    class PathSpecificPlaceDecoding(
        spyglass_mixin,
        dj_module.Computed,
    ):
        definition = table_specs.PATH_SPECIFIC_PLACE_DECODING_DEFINITION
        _compute_hook = staticmethod(
            path_specific_place_decoding_compute_hook
        )
        _register_existing_hook = staticmethod(
            path_specific_place_decoding_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one within-epoch decoder NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "PathSpecificPlaceDecoding.make() must run through "
                    "populate() so AnalysisNwbfile and result rows share one "
                    "transaction."
                )
            selection = _fetch1_dict(
                PathSpecificPlaceDecodingSelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=PathSpecificPlaceDecodingParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "path_specific_place_decoding_id": selection[
                            "path_specific_place_decoding_id"
                        ],
                        **row,
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_decoding_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Load and validate one canonical place-decoding analysis NWB."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                PathSpecificPlaceDecodingSelection,
                {
                    "path_specific_place_decoding_id": row[
                        "path_specific_place_decoding_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                PathSpecificPlaceDecodingParameters,
                {
                    "path_specific_place_decoding_param_name": selection[
                        "path_specific_place_decoding_param_name"
                    ]
                },
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            return _load_path_specific_place_decoding_result(
                result_row=row,
                decoding_table=cls,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_true_path: Path | str,
            source_decoded_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize two matching legacy Tsd artifacts and insert them."""
            if overwrite:
                raise ValueError(
                    "Registered PathSpecificPlaceDecoding results are "
                    "immutable; create a new selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(
                    PathSpecificPlaceDecodingSelection,
                    key,
                )
                result_key = {
                    "path_specific_place_decoding_id": selection[
                        "path_specific_place_decoding_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "PathSpecificPlaceDecoding already contains this "
                        "immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_true_path=Path(source_true_path),
                        source_decoded_path=Path(source_decoded_path),
                        parameters_table=PathSpecificPlaceDecodingParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        position_table=Position,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    **result_key,
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    PathSpecificPlaceDecoding = main_schema(PathSpecificPlaceDecoding)
    main_context["PathSpecificPlaceDecoding"] = PathSpecificPlaceDecoding

    class MotorEncodingParameters(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.MOTOR_ENCODING_PARAMETERS_DEFINITION
        )

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one motor-comparison parameter row."""
            validated = _validate_motor_encoding_parameter_row(
                row
            )
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_presets(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the canonical V1 and CA1 presets."""
            return [
                cls.insert_parameters(
                    preset,
                    skip_duplicates=skip_duplicates,
                )
                for preset in (
                    table_specs.MOTOR_ENCODING_PARAMETER_PRESETS
                )
            ]

        @classmethod
        def insert_default(
            cls,
            *,
            region: str,
            skip_duplicates: bool = True,
        ) -> dict[str, Any]:
            """Explicitly insert the manuscript preset for one region."""
            canonical_region = _analysis_region(region)
            preset = {
                "v1": (
                    table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS
                ),
                "ca1": (
                    table_specs.MANUSCRIPT_CA1_MOTOR_ENCODING_PARAMETERS
                ),
            }[canonical_region]
            return cls.insert_parameters(
                preset,
                skip_duplicates=skip_duplicates,
            )

    MotorEncodingParameters = main_schema(
        MotorEncodingParameters
    )
    main_context["MotorEncodingParameters"] = (
        MotorEncodingParameters
    )

    class MotorEncodingSelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.MOTOR_ENCODING_SELECTION_DEFINITION
        )

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one motor comparison."""
            row = _motor_encoding_selection_row(
                key=key,
                region_sorted_spikes_group_table=(
                    RegionSortedSpikesGroup
                ),
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                position_table=Position,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                stability_table=PathSpecificPlaceStability,
                stability_selection_table=(
                    PathSpecificPlaceStabilitySelection
                ),
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
                parameters_table=MotorEncodingParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    MotorEncodingSelection = main_schema(
        MotorEncodingSelection
    )
    main_context["MotorEncodingSelection"] = (
        MotorEncodingSelection
    )

    class MotorEncoding(
        spyglass_mixin,
        dj_module.Computed,
    ):
        definition = table_specs.MOTOR_ENCODING_DEFINITION
        _compute_hook = staticmethod(motor_encoding_compute_hook)
        _register_existing_hook = staticmethod(
            motor_encoding_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one nine-model analysis NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "MotorEncoding.make() must run through populate() so "
                    "AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(
                MotorEncodingSelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=MotorEncodingParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    stability_table=PathSpecificPlaceStability,
                    stability_selection_table=(
                        PathSpecificPlaceStabilitySelection
                    ),
                    tuning_curve_selection_table=(
                        PathSpecificPlaceTuningCurveSelection
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "motor_encoding_id": selection[
                            "motor_encoding_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_motor_encoding_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Load and validate one canonical motor-encoding NWB result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                MotorEncodingSelection,
                {
                    "motor_encoding_id": row[
                        "motor_encoding_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                MotorEncodingParameters,
                {
                    "motor_encoding_param_name": selection[
                        "motor_encoding_param_name"
                    ]
                },
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            bundle = _load_motor_encoding_result(
                result_row=row,
                motor_encoding_table=cls,
            )
            _validate_motor_encoding_artifact_link(
                bundle=bundle,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )
            return bundle

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_nested_cv_path: Path | str,
            source_full_refit_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one paired legacy motor fit and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered MotorEncoding results are immutable; "
                    "create a new selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(
                    MotorEncodingSelection,
                    key,
                )
                result_key = {
                    "motor_encoding_id": selection[
                        "motor_encoding_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "MotorEncoding already contains this immutable "
                        "selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_nested_cv_path=Path(source_nested_cv_path),
                        source_full_refit_path=Path(source_full_refit_path),
                        parameters_table=MotorEncodingParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        position_table=Position,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        stability_table=PathSpecificPlaceStability,
                        stability_selection_table=(
                            PathSpecificPlaceStabilitySelection
                        ),
                        tuning_curve_selection_table=(
                            PathSpecificPlaceTuningCurveSelection
                        ),
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "motor_encoding_id": selection[
                        "motor_encoding_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    MotorEncoding = main_schema(MotorEncoding)
    main_context["MotorEncoding"] = MotorEncoding

    class DarkLightGLMParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DARK_LIGHT_GLM_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one dark/light GLM parameter row."""
            validated = _validate_dark_light_glm_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_defaults(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the four current and legacy presets."""
            rows = [
                _validate_dark_light_glm_parameter_row(parameters)
                for parameters in table_specs.DARK_LIGHT_GLM_PARAMETER_PRESETS
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    DarkLightGLMParameters = main_schema(DarkLightGLMParameters)
    main_context["DarkLightGLMParameters"] = DarkLightGLMParameters

    class DarkLightGLMSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DARK_LIGHT_GLM_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one epoch-pair fit."""
            row = _dark_light_glm_selection_row(
                key=key,
                region_sorted_spikes_group_table=(
                    RegionSortedSpikesGroup
                ),
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                position_table=Position,
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                parameters_table=DarkLightGLMParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    DarkLightGLMSelection = main_schema(DarkLightGLMSelection)
    main_context["DarkLightGLMSelection"] = DarkLightGLMSelection

    class DarkLightGLM(spyglass_mixin, dj_module.Computed):
        definition = table_specs.DARK_LIGHT_GLM_DEFINITION
        _compute_hook = staticmethod(dark_light_glm_compute_hook)
        _register_existing_hook = staticmethod(dark_light_glm_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one dark/light GLM analysis NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "DarkLightGLM.make() must run through populate() so "
                    "AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(DarkLightGLMSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=DarkLightGLMParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "dark_light_glm_id": selection["dark_light_glm_id"],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_dark_light_glm_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Fetch and validate one canonical coupled NWB result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                DarkLightGLMSelection,
                {"dark_light_glm_id": row["dark_light_glm_id"]},
            )
            parameters = _fetch1_dict(
                DarkLightGLMParameters,
                {
                    "dark_light_glm_param_name": selection[
                        "dark_light_glm_param_name"
                    ]
                },
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            bundle = _load_dark_light_glm_result(
                result_row=row,
                dark_light_glm_table=cls,
            )
            _validate_dark_light_glm_artifact_link(
                bundle=bundle,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )
            return bundle

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_candidate_paths: list[Path | str],
            source_selected_paths_by_model: Mapping[str, Path | str],
            source_selection_summary_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one exact legacy/current artifact set and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered DarkLightGLM results are immutable; create a "
                    "new selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(DarkLightGLMSelection, key)
                result_key = {
                    "dark_light_glm_id": selection["dark_light_glm_id"]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "DarkLightGLM already contains this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_candidate_paths=[
                            Path(path) for path in source_candidate_paths
                        ],
                        source_selected_paths_by_model={
                            model_name: Path(path)
                            for model_name, path in (
                                source_selected_paths_by_model.items()
                            )
                        },
                        source_selection_summary_path=Path(
                            source_selection_summary_path
                        ),
                        parameters_table=DarkLightGLMParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        position_table=Position,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "dark_light_glm_id": selection["dark_light_glm_id"],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    DarkLightGLM = main_schema(DarkLightGLM)
    main_context["DarkLightGLM"] = DarkLightGLM

    class SwapGLMParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.SWAP_GLM_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one held-out swap parameter row."""
            validated = _validate_swap_glm_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_defaults(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the current held-out scoring preset."""
            rows = [
                _validate_swap_glm_parameter_row(parameters)
                for parameters in table_specs.SWAP_GLM_PARAMETER_PRESETS
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    SwapGLMParameters = main_schema(SwapGLMParameters)
    main_context["SwapGLMParameters"] = SwapGLMParameters

    class SwapGLMSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.SWAP_GLM_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one held-out score."""
            row = _swap_glm_selection_row(
                key=key,
                dark_light_glm_table=DarkLightGLM,
                dark_light_glm_selection_table=DarkLightGLMSelection,
                region_sorted_spikes_group_table=(
                    RegionSortedSpikesGroup
                ),
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                epoch_intervals_table=EpochIntervals,
                trajectory_intervals_table=TrajectoryIntervals,
                wtrack_graph_table=WTrackGraph,
                parameters_table=SwapGLMParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    SwapGLMSelection = main_schema(SwapGLMSelection)
    main_context["SwapGLMSelection"] = SwapGLMSelection

    class SwapGLM(spyglass_mixin, dj_module.Computed):
        definition = table_specs.SWAP_GLM_DEFINITION
        _compute_hook = staticmethod(swap_glm_compute_hook)
        _register_existing_hook = staticmethod(swap_glm_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one held-out SwapGLM NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "SwapGLM.make() must run through populate() so "
                    "AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(SwapGLMSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=SwapGLMParameters,
                    dark_light_glm_table=DarkLightGLM,
                    dark_light_glm_selection_table=DarkLightGLMSelection,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "swap_glm_id": selection["swap_glm_id"],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_swap_glm_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Fetch and validate one canonical held-out SwapGLM result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                SwapGLMSelection,
                {"swap_glm_id": row["swap_glm_id"]},
            )
            parameters = _fetch1_dict(
                SwapGLMParameters,
                {
                    "swap_glm_param_name": selection[
                        "swap_glm_param_name"
                    ]
                },
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            bundle = _load_swap_glm_result(
                result_row=row,
                swap_glm_table=cls,
            )
            _validate_swap_glm_artifact_link(
                bundle=bundle,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )
            return bundle

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_result_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one exact legacy held-out result and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered SwapGLM results are immutable; create a new "
                    "selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(SwapGLMSelection, key)
                result_key = {"swap_glm_id": selection["swap_glm_id"]}
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "SwapGLM already contains this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_result_path=Path(source_result_path),
                        parameters_table=SwapGLMParameters,
                        dark_light_glm_table=DarkLightGLM,
                        dark_light_glm_selection_table=DarkLightGLMSelection,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        position_table=Position,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=source_spyglass_git_commit,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "swap_glm_id": selection["swap_glm_id"],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    SwapGLM = main_schema(SwapGLM)
    main_context["SwapGLM"] = SwapGLM

    class SwapTuningCurveComparisonParameters(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.SWAP_TUNING_CURVE_COMPARISON_PARAMETERS_DEFINITION
        )

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one empirical swap-tuning parameter row."""
            validated = (
                _validate_swap_tuning_curve_comparison_parameter_row(row)
            )
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_defaults(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the manuscript V1 and CA1 presets."""
            rows = [
                _validate_swap_tuning_curve_comparison_parameter_row(
                    parameters
                )
                for parameters in (
                    table_specs.SWAP_TUNING_CURVE_COMPARISON_PARAMETER_PRESETS
                )
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    SwapTuningCurveComparisonParameters = main_schema(
        SwapTuningCurveComparisonParameters
    )
    main_context["SwapTuningCurveComparisonParameters"] = (
        SwapTuningCurveComparisonParameters
    )

    class SwapTuningCurveComparisonSelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.SWAP_TUNING_CURVE_COMPARISON_SELECTION_DEFINITION
        )

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one empirical swap."""
            row = _swap_tuning_curve_comparison_selection_row(
                key=key,
                region_sorted_spikes_group_table=(
                    RegionSortedSpikesGroup
                ),
                movement_firing_rate_table=MovementFiringRate,
                movement_firing_rate_selection_table=(
                    MovementFiringRateSelection
                ),
                movement_parameters_table=MovementParameters,
                position_table=Position,
                tuning_curve_table=PathSpecificPlaceTuningCurve,
                tuning_curve_selection_table=(
                    PathSpecificPlaceTuningCurveSelection
                ),
                tuning_curve_parameters_table=TuningCurveParameters,
                epoch_intervals_table=EpochIntervals,
                parameters_table=SwapTuningCurveComparisonParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    SwapTuningCurveComparisonSelection = main_schema(
        SwapTuningCurveComparisonSelection
    )
    main_context["SwapTuningCurveComparisonSelection"] = (
        SwapTuningCurveComparisonSelection
    )

    class SwapTuningCurveComparison(spyglass_mixin, dj_module.Computed):
        definition = table_specs.SWAP_TUNING_CURVE_COMPARISON_DEFINITION
        _compute_hook = staticmethod(
            swap_tuning_curve_comparison_compute_hook
        )
        _register_existing_hook = staticmethod(
            swap_tuning_curve_comparison_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one empirical swap-tuning NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "SwapTuningCurveComparison.make() must run through "
                    "populate() so AnalysisNwbfile and result rows share one "
                    "transaction."
                )
            selection = _fetch1_dict(
                SwapTuningCurveComparisonSelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=SwapTuningCurveComparisonParameters,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    movement_firing_rate_table=MovementFiringRate,
                    movement_firing_rate_selection_table=(
                        MovementFiringRateSelection
                    ),
                    movement_parameters_table=MovementParameters,
                    position_table=Position,
                    tuning_curve_table=PathSpecificPlaceTuningCurve,
                    tuning_curve_selection_table=(
                        PathSpecificPlaceTuningCurveSelection
                    ),
                    tuning_curve_parameters_table=TuningCurveParameters,
                    epoch_intervals_table=EpochIntervals,
                    trajectory_intervals_table=TrajectoryIntervals,
                    wtrack_graph_table=WTrackGraph,
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "swap_tuning_curve_comparison_id": selection[
                            "swap_tuning_curve_comparison_id"
                        ],
                        **row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": (
                            _spyglass_git_commit()
                        ),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_swap_tuning_curve_comparison_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Fetch and validate one canonical empirical swap result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                SwapTuningCurveComparisonSelection,
                {
                    "swap_tuning_curve_comparison_id": row[
                        "swap_tuning_curve_comparison_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                SwapTuningCurveComparisonParameters,
                selection,
            )
            region_row = _fetch1_dict(
                RegionSortedSpikesGroup,
                {
                    "region_sorted_spikes_group_id": selection[
                        "region_sorted_spikes_group_id"
                    ]
                },
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            return _load_swap_tuning_curve_comparison_result(
                result_row=row,
                result_table=cls,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_result_path: Path | str,
            source_summary_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Rebuild, verify, copy, and insert one exact legacy bundle."""
            if overwrite:
                raise ValueError(
                    "Registered SwapTuningCurveComparison results are "
                    "immutable; create a new selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(
                    SwapTuningCurveComparisonSelection,
                    key,
                )
                result_key = {
                    "swap_tuning_curve_comparison_id": selection[
                        "swap_tuning_curve_comparison_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "SwapTuningCurveComparison already contains this "
                        "immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_result_path=Path(source_result_path),
                        source_summary_path=Path(source_summary_path),
                        parameters_table=SwapTuningCurveComparisonParameters,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        movement_firing_rate_table=MovementFiringRate,
                        movement_firing_rate_selection_table=(
                            MovementFiringRateSelection
                        ),
                        movement_parameters_table=MovementParameters,
                        position_table=Position,
                        tuning_curve_table=PathSpecificPlaceTuningCurve,
                        tuning_curve_selection_table=(
                            PathSpecificPlaceTuningCurveSelection
                        ),
                        tuning_curve_parameters_table=TuningCurveParameters,
                        epoch_intervals_table=EpochIntervals,
                        trajectory_intervals_table=TrajectoryIntervals,
                        wtrack_graph_table=WTrackGraph,
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=source_spyglass_git_commit,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "swap_tuning_curve_comparison_id": selection[
                        "swap_tuning_curve_comparison_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    SwapTuningCurveComparison = main_schema(SwapTuningCurveComparison)
    main_context["SwapTuningCurveComparison"] = SwapTuningCurveComparison

    class RippleGLMParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_GLM_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one ripple population-GLM parameter row."""
            validated = _validate_ripple_glm_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_defaults(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the two manuscript predictor presets."""
            rows = [
                _validate_ripple_glm_parameter_row(parameters)
                for parameters in table_specs.RIPPLE_GLM_PARAMETER_PRESETS
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    RippleGLMParameters = main_schema(RippleGLMParameters)
    main_context["RippleGLMParameters"] = RippleGLMParameters

    class RippleGLMSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_GLM_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one RippleGLM row."""
            row = _ripple_glm_selection_row(
                key=key,
                ripples_table=RippleIntervals,
                epoch_intervals_table=EpochIntervals,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                parameters_table=RippleGLMParameters,
                nwbfile_table=nwbfile_table,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    RippleGLMSelection = main_schema(RippleGLMSelection)
    main_context["RippleGLMSelection"] = RippleGLMSelection

    class RippleGLM(spyglass_mixin, dj_module.Computed):
        definition = table_specs.RIPPLE_GLM_DEFINITION
        _compute_hook = staticmethod(ripple_glm_compute_hook)
        _register_existing_hook = staticmethod(ripple_glm_register_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one RippleGLM analysis NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "RippleGLM.make() must run through populate() so "
                    "AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(RippleGLMSelection, key)
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleGLMParameters,
                    ripples_table=RippleIntervals,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "ripple_glm_id": selection["ripple_glm_id"],
                        **artifact_row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_ripple_glm_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Fetch and validate one canonical RippleGLM NWB result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                RippleGLMSelection,
                {"ripple_glm_id": row["ripple_glm_id"]},
            )
            parameters = _fetch1_dict(RippleGLMParameters, selection)
            animal_name, session_date = _session_identity(
                session_table, selection
            )
            bundle = _load_ripple_glm_result(
                result_row=row,
                ripple_glm_table=cls,
            )
            _validate_ripple_glm_artifact_link(
                bundle=bundle,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                animal_name=animal_name,
                date=session_date,
            )
            return bundle

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_result_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Verify, normalize, copy, and insert one legacy RippleGLM."""
            if overwrite:
                raise ValueError(
                    "Registered RippleGLM results are immutable; create a "
                    "new selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(RippleGLMSelection, key)
                result_key = {
                    "ripple_glm_id": selection["ripple_glm_id"]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "RippleGLM already contains this immutable selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_result_path=Path(source_result_path),
                        parameters_table=RippleGLMParameters,
                        ripples_table=RippleIntervals,
                        epoch_intervals_table=EpochIntervals,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=(
                            source_spyglass_git_commit
                        ),
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "ripple_glm_id": selection["ripple_glm_id"],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    RippleGLM = main_schema(RippleGLM)
    main_context["RippleGLM"] = RippleGLM

    class RippleCrossRegionXCorrParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_CROSS_REGION_XCORR_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one fixed ripple-xcorr parameter row."""
            validated = _validate_ripple_cross_region_xcorr_parameter_row(row)
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_defaults(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the fixed manuscript xcorr preset."""
            rows = [
                _validate_ripple_cross_region_xcorr_parameter_row(parameters)
                for parameters in (
                    table_specs.RIPPLE_CROSS_REGION_XCORR_PARAMETER_PRESETS
                )
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    RippleCrossRegionXCorrParameters = main_schema(RippleCrossRegionXCorrParameters)
    main_context["RippleCrossRegionXCorrParameters"] = RippleCrossRegionXCorrParameters

    class RippleCrossRegionXCorrSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLE_CROSS_REGION_XCORR_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one exact-ripple xcorr."""
            row = _ripple_cross_region_xcorr_selection_row(
                key=key,
                ripples_table=RippleIntervals,
                epoch_intervals_table=EpochIntervals,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                parameters_table=RippleCrossRegionXCorrParameters,
                nwbfile_table=nwbfile_table,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    RippleCrossRegionXCorrSelection = main_schema(RippleCrossRegionXCorrSelection)
    main_context["RippleCrossRegionXCorrSelection"] = RippleCrossRegionXCorrSelection

    class RippleCrossRegionXCorr(spyglass_mixin, dj_module.Computed):
        definition = table_specs.RIPPLE_CROSS_REGION_XCORR_DEFINITION
        _compute_hook = staticmethod(ripple_cross_region_xcorr_compute_hook)
        _register_existing_hook = staticmethod(
            ripple_cross_region_xcorr_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one cross-region xcorr NWB."""
            if getattr(self.connection, "in_transaction", None) is False:
                raise RuntimeError(
                    "RippleCrossRegionXCorr.make() must run through populate() "
                    "so AnalysisNwbfile and result rows share one transaction."
                )
            selection = _fetch1_dict(RippleCrossRegionXCorrSelection, key)
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleCrossRegionXCorrParameters,
                    ripples_table=RippleIntervals,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                    analysis_nwbfile_table=AnalysisNwbfile,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "ripple_cross_region_xcorr_id": selection[
                            "ripple_cross_region_xcorr_id"
                        ],
                        **artifact_row,
                        "artifact_origin": "computed",
                        "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                        "runtime_spyglass_git_commit": _spyglass_git_commit(),
                    }
                )
            except Exception:
                _remove_created_artifacts(created_artifact_paths)
                raise

        @classmethod
        def load_ripple_cross_region_xcorr_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Fetch and validate one canonical cross-region xcorr result."""
            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                RippleCrossRegionXCorrSelection,
                {"ripple_cross_region_xcorr_id": row["ripple_cross_region_xcorr_id"]},
            )
            parameters = _fetch1_dict(
                RippleCrossRegionXCorrParameters,
                selection,
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            return _load_ripple_cross_region_xcorr_result(
                result_row=row,
                result_table=cls,
                selection_row=selection,
                parameters_row=parameters,
                animal_name=animal_name,
                date=session_date,
            )

        @classmethod
        def register_existing(
            cls,
            key: Mapping[str, Any],
            *,
            source_ca1_unit_filter_path: Path | str,
            source_v1_unit_filter_path: Path | str,
            source_summary_path: Path | str,
            source_result_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Verify and insert one exact legacy four-artifact xcorr set."""
            if overwrite:
                raise ValueError(
                    "Registered RippleCrossRegionXCorr results are immutable; "
                    "create a new selection instead of overwriting."
                )
            with _transaction_context(cls):
                selection = _fetch1_dict(RippleCrossRegionXCorrSelection, key)
                result_key = {
                    "ripple_cross_region_xcorr_id": selection[
                        "ripple_cross_region_xcorr_id"
                    ]
                }
                existing = _existing_result_row(cls, result_key)
                if existing is not None:
                    if skip_duplicates:
                        return existing
                    raise ValueError(
                        "RippleCrossRegionXCorr already contains this immutable "
                        "selection."
                    )
                artifact_row = dict(
                    cls._register_existing_hook(
                        key=selection,
                        source_ca1_unit_filter_path=Path(
                            source_ca1_unit_filter_path
                        ),
                        source_v1_unit_filter_path=Path(
                            source_v1_unit_filter_path
                        ),
                        source_summary_path=Path(source_summary_path),
                        source_result_path=Path(source_result_path),
                        parameters_table=RippleCrossRegionXCorrParameters,
                        ripples_table=RippleIntervals,
                        epoch_intervals_table=EpochIntervals,
                        region_sorted_spikes_group_table=(
                            RegionSortedSpikesGroup
                        ),
                        session_table=session_table,
                        nwbfile_table=nwbfile_table,
                        source_v1ca1_git_commit=source_v1ca1_git_commit,
                        source_spyglass_git_commit=source_spyglass_git_commit,
                        artifact_root=artifact_root,
                        analysis_nwbfile_table=AnalysisNwbfile,
                    )
                )
                created_artifact_paths = list(
                    artifact_row.pop("_created_artifact_paths", ())
                )
                row = {
                    "ripple_cross_region_xcorr_id": selection[
                        "ripple_cross_region_xcorr_id"
                    ],
                    **artifact_row,
                    "artifact_origin": "registered_existing",
                    "runtime_v1ca1_git_commit": _v1ca1_git_commit(),
                    "runtime_spyglass_git_commit": _spyglass_git_commit(),
                }
                try:
                    cls.insert1(
                        row,
                        skip_duplicates=False,
                        allow_direct_insert=True,
                    )
                except Exception:
                    _remove_created_artifacts(created_artifact_paths)
                    raise
                return row

    RippleCrossRegionXCorr = main_schema(RippleCrossRegionXCorr)
    main_context["RippleCrossRegionXCorr"] = RippleCrossRegionXCorr

    return {
        "epoch_intervals": EpochIntervals,
        "trajectory_intervals": TrajectoryIntervals,
        "ripple_intervals": RippleIntervals,
        "position": Position,
        "wtrack_graph": WTrackGraph,
        "spike_sorting_figurl": SpikeSortingFigurl,
        "region_sorted_spikes_group": RegionSortedSpikesGroup,
        "movement_parameters": MovementParameters,
        "epoch_motor_behavior_parameters": EpochMotorBehaviorParameters,
        "epoch_motor_behavior_selection": EpochMotorBehaviorSelection,
        "epoch_motor_behavior": EpochMotorBehavior,
        "movement_firing_rate_selection": MovementFiringRateSelection,
        "movement_firing_rate": MovementFiringRate,
        "cv_pca_parameters": CVPCAParameters,
        "cv_pca_selection": CVPCASelection,
        "cv_pca": CVPCA,
        "ripple_modulation_parameters": RippleModulationParameters,
        "ripple_modulation_selection": RippleModulationSelection,
        "ripple_modulation": RippleModulation,
        "tuning_curve_parameters": TuningCurveParameters,
        "tuning_similarity_parameters": TuningSimilarityParameters,
        "path_specific_place_tuning_curve_selection": (
            PathSpecificPlaceTuningCurveSelection
        ),
        "path_specific_place_tuning_curve": PathSpecificPlaceTuningCurve,
        "path_specific_place_tuning_similarity_selection": (
            PathSpecificPlaceTuningSimilaritySelection
        ),
        "path_specific_place_tuning_similarity": (
            PathSpecificPlaceTuningSimilarity
        ),
        "dpp_tuning_curve_selection": DPPTuningCurveSelection,
        "dpp_tuning_curve": DPPTuningCurve,
        "path_specific_place_stability_selection": (
            PathSpecificPlaceStabilitySelection
        ),
        "path_specific_place_stability": PathSpecificPlaceStability,
        "dpp_encoding_parameters": (
            DPPEncodingParameters
        ),
        "dpp_encoding_selection": DPPEncodingSelection,
        "dpp_encoding": DPPEncoding,
        "path_progression_decoding_parameters": (
            PathProgressionDecodingParameters
        ),
        "path_progression_decoding_selection": (
            PathProgressionDecodingSelection
        ),
        "path_progression_decoding": (
            PathProgressionDecoding
        ),
        "path_specific_place_decoding_parameters": (
            PathSpecificPlaceDecodingParameters
        ),
        "path_specific_place_decoding_selection": (
            PathSpecificPlaceDecodingSelection
        ),
        "path_specific_place_decoding": PathSpecificPlaceDecoding,
        "motor_encoding_parameters": (
            MotorEncodingParameters
        ),
        "motor_encoding_selection": (
            MotorEncodingSelection
        ),
        "motor_encoding": MotorEncoding,
        "dark_light_glm_parameters": DarkLightGLMParameters,
        "dark_light_glm_selection": DarkLightGLMSelection,
        "dark_light_glm": DarkLightGLM,
        "swap_glm_parameters": SwapGLMParameters,
        "swap_glm_selection": SwapGLMSelection,
        "swap_glm": SwapGLM,
        "swap_tuning_curve_comparison_parameters": (
            SwapTuningCurveComparisonParameters
        ),
        "swap_tuning_curve_comparison_selection": (
            SwapTuningCurveComparisonSelection
        ),
        "swap_tuning_curve_comparison": SwapTuningCurveComparison,
        "ripple_glm_parameters": RippleGLMParameters,
        "ripple_glm_selection": RippleGLMSelection,
        "ripple_glm": RippleGLM,
        "ripple_cross_region_xcorr_parameters": RippleCrossRegionXCorrParameters,
        "ripple_cross_region_xcorr_selection": RippleCrossRegionXCorrSelection,
        "ripple_cross_region_xcorr": RippleCrossRegionXCorr,
        "analysis_nwbfile": AnalysisNwbfile,
    }


def activate(
    schema_name: str = table_specs.DEFAULT_SCHEMA_NAME,
    *,
    analysis_nwbfile_schema_name: str = table_specs.DEFAULT_ANALYSIS_NWBFILE_SCHEMA_NAME,
    connection: Any = None,
    create_schema: bool = True,
    create_tables: bool = True,
    runtime_hooks: Mapping[str, Callable[..., Any]] | None = None,
    artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    """Explicitly import dependencies, activate schemas, and return table classes.

    Activation declares tables only.  It never calls ``insert_default``,
    ``populate``, ``make``, or ``register_existing``.
    A custom ``artifact_root`` must be inside the configured ``analysis``
    external-store stage so DataJoint can accept ``filepath@analysis`` values.
    """
    import datajoint as dj

    from spyglass.common import Nwbfile, Session
    from spyglass.spikesorting.analysis.v1.group import (
        SortedSpikesGroup,
        UnitSelectionParams,
    )
    from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput
    from spyglass.utils import SpyglassAnalysis, SpyglassMixin

    _validate_analysis_schema_prefix(dj, analysis_nwbfile_schema_name)

    return _construct_tables(
        dj_module=dj,
        session_table=Session,
        nwbfile_table=Nwbfile,
        sorted_spikes_group=SortedSpikesGroup,
        unit_selection_params=UnitSelectionParams,
        spike_sorting_output=SpikeSortingOutput,
        spyglass_mixin=SpyglassMixin,
        spyglass_analysis=SpyglassAnalysis,
        schema_factory=dj.Schema,
        schema_name=schema_name,
        analysis_nwbfile_schema_name=analysis_nwbfile_schema_name,
        connection=connection,
        create_schema=create_schema,
        create_tables=create_tables,
        runtime_hooks=runtime_hooks,
        artifact_root=None if artifact_root is None else Path(artifact_root),
    )


__all__ = ["SOURCE_TABLE_KEYS", "activate"]
