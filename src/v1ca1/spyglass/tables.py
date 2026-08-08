"""Explicit activation for the project-owned Spyglass tables.

Importing this module is passive: DataJoint and Spyglass are imported only by
``activate``. Runtime computation is likewise reached only through explicitly
activated computed tables.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    "ripples",
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


def _validate_dpp_encoding_comparison_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one four-model DPP encoding-comparison parameter row."""
    expected = set(table_specs.MANUSCRIPT_DPP_ENCODING_COMPARISON_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "DPPEncodingComparison parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["dpp_encoding_comparison_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "dpp_encoding_comparison_param_name must be a non-empty string "
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


def _validate_motor_encoding_comparison_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one nine-model nested-CV motor-encoding parameter row."""
    expected = set(
        table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_COMPARISON_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "MotorEncodingComparison parameters must have exactly the "
            f"declared fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["motor_encoding_comparison_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "motor_encoding_comparison_param_name must be a non-empty "
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


def _cross_region_xcorr_parameter_kwargs(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return model fields accepted by the database-free xcorr API."""
    return {
        field_name: value
        for field_name, value in parameters.items()
        if field_name != "cross_region_xcorr_param_name"
    }


def _validate_cross_region_xcorr_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact ripple-restricted cross-region parameter row."""
    from v1ca1.spyglass.cross_region_xcorr import (
        validate_cross_region_xcorr_parameters,
    )

    expected = set(table_specs.MANUSCRIPT_CROSS_REGION_XCORR_PARAMETERS)
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "CrossRegionXCorr parameters must have exactly the declared "
            f"fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["cross_region_xcorr_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "cross_region_xcorr_param_name must be a non-empty string of at "
            "most 64 characters."
        )
    values["norm"] = _database_bool(values["norm"], name="norm")
    values["require_speed_gated"] = _database_bool(
        values["require_speed_gated"],
        name="require_speed_gated",
    )
    validated = validate_cross_region_xcorr_parameters(
        **_cross_region_xcorr_parameter_kwargs(values)
    )
    return {"cross_region_xcorr_param_name": name, **validated}


def _ripple_decoding_comparison_parameter_kwargs(
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Return fields accepted by the database-free decoding API."""
    return {
        field_name: value
        for field_name, value in parameters.items()
        if field_name != "ripple_decoding_comparison_param_name"
    }


def _validate_ripple_decoding_comparison_parameter_row(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one fixed categorical ripple-decoding parameter row."""
    from v1ca1.spyglass.ripple_decoding_comparison import (
        validate_ripple_decoding_comparison_parameters,
    )

    expected = set(
        table_specs.MANUSCRIPT_RIPPLE_DECODING_COMPARISON_PARAMETERS
    )
    missing = sorted(expected.difference(row))
    extra = sorted(set(row).difference(expected))
    if missing or extra:
        raise ValueError(
            "RippleDecodingComparison parameters must have exactly the "
            f"declared fields; missing={missing!r}, extra={extra!r}."
        )
    values = dict(row)
    name = values["ripple_decoding_comparison_param_name"]
    if not isinstance(name, str) or not name.strip() or len(name) > 64:
        raise ValueError(
            "ripple_decoding_comparison_param_name must be a non-empty "
            "string of at most 64 characters."
        )
    values["require_speed_gated"] = _database_bool(
        values["require_speed_gated"], name="require_speed_gated"
    )
    validated = validate_ripple_decoding_comparison_parameters(
        **_ripple_decoding_comparison_parameter_kwargs(values)
    )
    return {"ripple_decoding_comparison_param_name": name, **validated}


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
            "Ripples.detector_zscore_threshold must be one numeric scalar."
        )
    threshold = float(actual_threshold)
    if not math.isfinite(threshold):
        raise ValueError("Ripples.detector_zscore_threshold must be finite.")
    speed_gated = _database_bool(
        ripple_row.get("speed_gated"),
        name="Ripples.speed_gated",
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
            "Ripples.detector_zscore_threshold does not match "
            "expected_detector_zscore_threshold."
        )
    if parameters["require_speed_gated"] and not speed_gated:
        raise ValueError("Selected Ripples row must be explicitly speed-gated.")


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


def _sorting_snapshot_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return selection columns for one resolved sorting-group snapshot."""
    parameters = dict(provenance["unit_selection_params"])
    return {
        "sorting_group_members": list(provenance["sorting_group_members"]),
        "sorting_group_members_sha256": str(
            provenance["sorting_group_members_sha256"]
        ),
        "unit_filter_include_labels": list(parameters["include_labels"]),
        "unit_filter_exclude_labels": list(parameters["exclude_labels"]),
        "unit_filter_params_sha256": str(
            provenance["unit_selection_params_sha256"]
        ),
    }


def _resolve_sorting_snapshot(
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    key: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one standard group and its immutable label-filter snapshot."""
    from v1ca1.spyglass.spikes import resolve_sorted_spikes_group_provenance

    return resolve_sorted_spikes_group_provenance(
        sorted_spikes_group,
        unit_selection_params,
        key,
    )


def _validate_frozen_sorting_snapshot(
    selection: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Require current group membership/filter values to match a selection."""
    expected = _sorting_snapshot_fields(provenance)
    for field_name, current_value in expected.items():
        selected_value = selection.get(field_name)
        if field_name.endswith("labels") or field_name == "sorting_group_members":
            selected_value = list(selected_value or ())
        if selected_value != current_value:
            raise ValueError(
                "SortedSpikesGroup membership or UnitSelectionParams changed "
                f"after selection insertion: {field_name}. Create a new selection."
            )


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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
) -> dict[str, Any]:
    """Validate and identify one immutable RippleModulation selection."""
    from v1ca1.spyglass.selection import selection_uuid

    natural_key = {
        field_name: key[field_name]
        for field_name in (
            "nwb_file_name",
            "epoch",
            "ripple_modulation_param_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
        )
    }
    natural_key["region"] = _analysis_region(key["region"])
    _fetch1_dict(ripples_table, natural_key)
    _fetch1_dict(epoch_intervals_table, natural_key)
    parameters = _validate_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    provenance = _resolve_sorting_snapshot(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        key=natural_key,
    )
    snapshot = _sorting_snapshot_fields(provenance)
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="ripple_modulation_parameters_sha256",
    )
    identity_payload = {**natural_key, **snapshot, **parameter_snapshot}
    return {
        "ripple_modulation_id": selection_uuid(
            "RippleModulation",
            identity_payload,
        ),
        **natural_key,
        **snapshot,
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
            "Ripples.ripple_count disagrees with its NWB interval data."
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


def _cross_region_xcorr_selection_row(
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
    from v1ca1.spyglass import cross_region_xcorr
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
    parameters = _validate_cross_region_xcorr_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "cross_region_xcorr_param_name": key[
                    "cross_region_xcorr_param_name"
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
                f"CrossRegionXCorr {role} group must select region "
                f"{expected_region!r}."
            )
        if str(row.get("nwb_file_name")) != nwb_file_name:
            raise ValueError(
                "CrossRegionXCorr ripple and regional sorting inputs must "
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
        cross_region_xcorr.prepare_cross_region_xcorr_event_selection(
            epoch=epoch,
            ripple_table=ripple_table,
        )
    )
    normalized_ripples = event_selection["selected_ripple_table"]
    if int(event_selection["n_ripples"]) != int(ripple_row["ripple_count"]):
        raise ValueError(
            "Ripples.ripple_count disagrees with its NWB interval data."
        )
    if dict(cross_region_xcorr.OUTPUT_RULE) != dict(
        table_specs.CROSS_REGION_XCORR_OUTPUT_RULE
    ):
        raise RuntimeError(
            "CrossRegionXCorr table and database-free output rules have "
            "diverged."
        )
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="cross_region_xcorr_parameters_sha256",
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
        "cross_region_xcorr_param_name": parameters[
            "cross_region_xcorr_param_name"
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
        "cross_region_xcorr_output_rule_sha256": (
            cross_region_xcorr.OUTPUT_RULE_SHA256
        ),
    }
    return {
        "cross_region_xcorr_id": selection_uuid(
            "CrossRegionXCorr",
            natural_key,
        ),
        **natural_key,
    }


def _ripple_decoding_interval_sha256(intervals: Any) -> str:
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


def _ripple_decoding_movement_rates_sha256(table: Any) -> str:
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


def _ripple_decoding_catalog_row_sha256(row: Mapping[str, Any]) -> str:
    """Digest one catalog row without transient DataJoint fields."""
    from v1ca1.spyglass.selection import provenance_sha256

    return provenance_sha256(
        {
            str(name): value
            for name, value in dict(row).items()
            if not str(name).startswith("_")
        }
    )


def _ripple_decoding_comparison_upstream_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact JSON-safe upstream snapshot embedded in artifacts."""
    fields = (
        "nwb_file_name",
        "train_epoch",
        "decode_epoch",
        "train_position_series_name",
        "representation",
        "source_region",
        "target_region",
        "source_ripple_count",
        "detector_zscore_threshold",
        "speed_gated",
        "selected_ripple_intervals_sha256",
        "ripple_provenance_sha256",
        "ca1_region_sorted_spikes_group_id",
        "ca1_sorting_group_members_sha256",
        "ca1_unit_filter_params_sha256",
        "ca1_selected_units_sha256",
        "ca1_n_units",
        "v1_region_sorted_spikes_group_id",
        "v1_sorting_group_members_sha256",
        "v1_unit_filter_params_sha256",
        "v1_selected_units_sha256",
        "v1_n_units",
        "ca1_movement_firing_rate_id",
        "v1_movement_firing_rate_id",
        "movement_param_name",
        "movement_parameters_sha256",
        "movement_speed_threshold_cm_s",
        "movement_speed_sigma_s",
        "ca1_movement_firing_rate_sha256",
        "ca1_movement_intervals_sha256",
        "ca1_movement_rates_sha256",
        "ca1_movement_support_sha256",
        "ca1_movement_analysis_status",
        "v1_movement_firing_rate_sha256",
        "v1_movement_intervals_sha256",
        "v1_movement_rates_sha256",
        "v1_movement_support_sha256",
        "v1_movement_analysis_status",
        "position_snapshot_sha256",
        "center_to_left_trajectory_type",
        "center_to_right_trajectory_type",
        "left_to_center_trajectory_type",
        "right_to_center_trajectory_type",
        "center_to_left_configuration_name",
        "center_to_right_configuration_name",
        "left_to_center_configuration_name",
        "right_to_center_configuration_name",
        "trajectory_intervals_sha256_by_type",
        "graph_rows_sha256_by_trajectory",
        "graph_policy_sha256",
        "ripple_decoding_comparison_parameters_sha256",
        "ripple_decoding_comparison_output_rule_sha256",
    )
    output = {
        "ripple_decoding_comparison_id": str(
            selection["ripple_decoding_comparison_id"]
        ),
        **{name: selection[name] for name in fields},
    }
    output["speed_gated"] = _database_bool(
        output["speed_gated"],
        name="RippleDecodingComparisonSelection.speed_gated",
    )
    output["detector_zscore_threshold"] = float(
        output["detector_zscore_threshold"]
    )
    output["movement_speed_threshold_cm_s"] = float(
        output["movement_speed_threshold_cm_s"]
    )
    output["movement_speed_sigma_s"] = float(
        output["movement_speed_sigma_s"]
    )
    for name in (
        "ca1_region_sorted_spikes_group_id",
        "v1_region_sorted_spikes_group_id",
        "ca1_movement_firing_rate_id",
        "v1_movement_firing_rate_id",
    ):
        output[name] = str(output[name])
    for name in ("source_ripple_count", "ca1_n_units", "v1_n_units"):
        output[name] = int(output[name])
    output["trajectory_intervals_sha256_by_type"] = dict(
        output["trajectory_intervals_sha256_by_type"]
    )
    output["graph_rows_sha256_by_trajectory"] = dict(
        output["graph_rows_sha256_by_trajectory"]
    )
    return output


def _ripple_decoding_comparison_selection_row(
    *,
    key: Mapping[str, Any],
    ripples_table: Any,
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
    nwbfile_table: Any | None = None,
    ripple_table: Any | None = None,
    decode_epoch_interval: Any | None = None,
    trajectory_interval_sets: Mapping[str, Any] | None = None,
    graph_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    movement_artifacts_by_region: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate, freeze, and UUID one complete ripple-decoding selection."""
    from v1ca1.spyglass import ripple_decoding_comparison as decoding
    from v1ca1.spyglass.selection import selection_uuid

    nwb_file_name = str(key["nwb_file_name"])
    train_epoch = str(key["train_epoch"])
    decode_epoch = str(key["decode_epoch"])
    representation = str(key["representation"])
    if representation not in decoding.REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {decoding.REPRESENTATIONS!r}."
        )
    parameters = _validate_ripple_decoding_comparison_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "ripple_decoding_comparison_param_name": key[
                    "ripple_decoding_comparison_param_name"
                ]
            },
        )
    )
    if dict(decoding.OUTPUT_RULE) != dict(
        table_specs.RIPPLE_DECODING_COMPARISON_OUTPUT_RULE
    ):
        raise RuntimeError(
            "RippleDecodingComparison table and artifact output rules diverged."
        )
    train_epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": train_epoch},
    )
    decode_epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": decode_epoch},
    )
    if train_epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("RippleDecodingComparison training epoch must be a run.")
    ripple_row = _fetch1_dict(
        ripples_table,
        {"nwb_file_name": nwb_file_name, "epoch": decode_epoch},
    )
    _validate_ripple_provenance(ripple_row, parameters)
    detector_zscore_threshold, speed_gated = _ripple_detector_values(
        ripple_row
    )
    group_rows = {
        region: _fetch1_dict(
            region_sorted_spikes_group_table,
            {
                "region_sorted_spikes_group_id": key[
                    f"{region}_region_sorted_spikes_group_id"
                ]
            },
        )
        for region in ("ca1", "v1")
    }
    movement_results = {}
    movement_selections = {}
    movement_parameters = {}
    movement_loaded = {}
    supplied_movement = dict(movement_artifacts_by_region or {})
    animal_name, session_date = _session_identity(
        session_table, {"nwb_file_name": nwb_file_name}
    )
    for region in ("ca1", "v1"):
        group_row = group_rows[region]
        if str(group_row.get("region_name")) != region or str(
            group_row.get("nwb_file_name")
        ) != nwb_file_name:
            raise ValueError(
                f"RippleDecodingComparison {region} group has wrong region or NWB."
            )
        movement_key = {
            "movement_firing_rate_id": key[f"{region}_movement_firing_rate_id"]
        }
        movement_results[region] = _fetch1_dict(
            movement_firing_rate_table, movement_key
        )
        movement_selections[region] = _fetch1_dict(
            movement_firing_rate_selection_table, movement_key
        )
        movement_selection = movement_selections[region]
        movement_result = movement_results[region]
        for field_name, expected in (
            ("nwb_file_name", nwb_file_name),
            ("epoch", train_epoch),
            ("region", region),
        ):
            if str(movement_selection.get(field_name)) != expected:
                raise ValueError(
                    f"RippleDecodingComparison {region} movement must share "
                    f"the selected {field_name}."
                )
        for field_name in (
            "nwb_file_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
            "sorting_group_members_sha256",
            "unit_filter_params_sha256",
        ):
            if str(group_row.get(field_name)) != str(
                movement_selection.get(field_name)
            ):
                raise ValueError(
                    f"RippleDecodingComparison {region} regional group and "
                    f"movement row differ in {field_name}."
                )
        if str(group_row.get("selected_units_sha256")) != str(
            movement_result.get("selected_units_sha256")
        ) or int(group_row.get("n_units", -1)) != int(
            movement_result.get("n_units", -2)
        ):
            raise ValueError(
                f"RippleDecodingComparison {region} movement and regional "
                "group must contain identical units."
            )
        movement_parameters[region] = _validate_movement_parameter_row(
            _fetch1_dict(movement_parameters_table, movement_selection)
        )
        _validate_frozen_parameters(
            movement_selection,
            movement_parameters[region],
            field_name="movement_parameters_sha256",
        )
        if region in supplied_movement:
            movement_loaded[region] = dict(supplied_movement[region])
        else:
            movement_loaded[region] = _load_movement_result_artifacts(
                result_row=movement_result,
                parameters=movement_parameters[region],
                expected_metadata={
                    "animal_name": animal_name,
                    "date": session_date,
                    "region": region,
                    "epoch": train_epoch,
                },
            )
    if movement_parameters["ca1"] != movement_parameters["v1"]:
        raise ValueError(
            "CA1 and V1 MovementFiringRate rows must share one movement definition."
        )
    shared_movement = movement_parameters["ca1"]
    if not math.isclose(
        float(shared_movement["speed_threshold_cm_s"]),
        4.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(shared_movement["speed_smoothing_sigma_s"]),
        0.1,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "RippleDecodingComparison requires upstream movement parameters "
            "4.0 cm/s and 0.1 s."
        )
    position_series_names = {
        str(row["position_series_name"])
        for row in movement_selections.values()
    }
    if len(position_series_names) != 1:
        raise ValueError("CA1 and V1 movement rows must share one position series.")
    position_series_name = position_series_names.pop()
    supplied_position_name = str(
        key.get("train_position_series_name", position_series_name)
    )
    if supplied_position_name != position_series_name:
        raise ValueError(
            "train_position_series_name must equal the movement position series."
        )
    position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": train_epoch,
            "position_series_name": position_series_name,
        },
    )
    if str(position_row.get("spatial_unit")) != "cm":
        raise ValueError("RippleDecodingComparison position must use centimeters.")

    trajectory_rows = {}
    graph_rows = {}
    source_fields: dict[str, Any] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type or str(
            key.get(configuration_field, trajectory_type)
        ) != trajectory_type:
            raise ValueError(
                "RippleDecodingComparison trajectory and graph aliases must "
                "match the four canonical path names."
            )
        trajectory_rows[trajectory_type] = _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": train_epoch,
                "trajectory_type": trajectory_type,
            },
        )
        graph_rows[trajectory_type] = _fetch1_dict(
            wtrack_graph_table,
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
            },
        )
        if str(graph_rows[trajectory_type].get("coordinate_unit")) != "cm":
            raise ValueError("RippleDecodingComparison graphs must use centimeters.")
        source_fields[trajectory_field] = trajectory_type
        source_fields[configuration_field] = trajectory_type

    if ripple_table is None or decode_epoch_interval is None:
        if nwbfile_table is None:
            raise ValueError(
                "nwbfile_table is required when ripple/epoch intervals are not supplied."
            )
        loaded_ripple_table, loaded_decode_epoch = _load_ripple_glm_interval_inputs(
            nwb_file_name=nwb_file_name,
            ripple_row=ripple_row,
            epoch_row=decode_epoch_row,
            nwbfile_table=nwbfile_table,
        )
        if ripple_table is None:
            ripple_table = loaded_ripple_table
        if decode_epoch_interval is None:
            decode_epoch_interval = loaded_decode_epoch
    event_selection = decoding.prepare_ripple_decoding_comparison_event_selection(
        decode_epoch=decode_epoch, ripple_table=ripple_table
    )
    decoding._require_ripples_inside_epoch(
        event_selection["selected_ripple_table"], decode_epoch_interval
    )
    if int(event_selection["n_ripple_events_input"]) != int(
        ripple_row["ripple_count"]
    ):
        raise ValueError("Ripples.ripple_count disagrees with exact NWB intervals.")

    if trajectory_interval_sets is None:
        trajectory_interval_sets = {
            trajectory_type: trajectory_intervals_table.load_intervals(
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": train_epoch,
                    "trajectory_type": trajectory_type,
                }
            )
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        }
    if graph_inputs is None:
        graph_inputs = {
            trajectory_type: wtrack_graph_table.load_graph(
                {
                    "nwb_file_name": nwb_file_name,
                    "configuration_name": trajectory_type,
                }
            )
            for trajectory_type in _DPP_TRAJECTORY_TYPES
        }
    geometry = decoding._graph_geometry(graph_inputs)
    trajectory_hashes = {
        trajectory_type: _ripple_decoding_interval_sha256(
            trajectory_interval_sets[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_row_hashes = {
        trajectory_type: _ripple_decoding_catalog_row_sha256(
            graph_rows[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    movement_snapshots = {}
    support_hashes = {}
    for region in ("ca1", "v1"):
        movement_result = movement_results[region]
        loaded = movement_loaded[region]
        movement_status = str(loaded["analysis_status"])
        if movement_status not in {"valid", "no_movement", "no_valid_position"}:
            raise ValueError(
                "RippleDecodingComparison movement artifact has unsupported "
                f"analysis_status {movement_status!r}."
            )
        observed_statuses = set(
            loaded["table"]["firing_rate_status"].astype(str)
        )
        if observed_statuses != {movement_status}:
            raise ValueError(
                "RippleDecodingComparison movement table and result status "
                f"disagree for {region}."
            )
        support_hashes[region] = _ripple_decoding_interval_sha256(
            loaded["movement_intervals"]
        )
        movement_snapshots.update(
            {
                f"{region}_movement_firing_rate_sha256": _file_sha256(
                    Path(movement_result["movement_firing_rate_path"])
                ),
                f"{region}_movement_intervals_sha256": _file_sha256(
                    Path(movement_result["movement_intervals_path"])
                ),
                f"{region}_movement_rates_sha256": (
                    _ripple_decoding_movement_rates_sha256(loaded["table"])
                ),
                f"{region}_movement_support_sha256": support_hashes[region],
                f"{region}_movement_analysis_status": movement_status,
            }
        )
    if support_hashes["ca1"] != support_hashes["v1"]:
        raise ValueError("CA1 and V1 movement support must match exactly.")
    if (
        movement_snapshots["ca1_movement_analysis_status"]
        != movement_snapshots["v1_movement_analysis_status"]
    ):
        raise ValueError(
            "CA1 and V1 movement analysis_status must match exactly."
        )

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="ripple_decoding_comparison_parameters_sha256",
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "train_epoch": train_epoch,
        "decode_epoch": decode_epoch,
        "train_position_series_name": position_series_name,
        "ca1_region_sorted_spikes_group_id": key[
            "ca1_region_sorted_spikes_group_id"
        ],
        "v1_region_sorted_spikes_group_id": key[
            "v1_region_sorted_spikes_group_id"
        ],
        "ca1_movement_firing_rate_id": key["ca1_movement_firing_rate_id"],
        "v1_movement_firing_rate_id": key["v1_movement_firing_rate_id"],
        **source_fields,
        "ripple_decoding_comparison_param_name": parameters[
            "ripple_decoding_comparison_param_name"
        ],
        "representation": representation,
        "source_region": "ca1",
        "target_region": "v1",
        "source_ripple_count": int(ripple_row["ripple_count"]),
        "detector_zscore_threshold": detector_zscore_threshold,
        "speed_gated": speed_gated,
        "selected_ripple_intervals_sha256": event_selection[
            "selected_ripple_intervals_sha256"
        ],
        "ripple_provenance_sha256": _ripple_glm_provenance_sha256(ripple_row),
        **_ripple_glm_group_snapshot(group_rows["ca1"], role="ca1"),
        **_ripple_glm_group_snapshot(group_rows["v1"], role="v1"),
        "movement_param_name": shared_movement["movement_param_name"],
        "movement_parameters_sha256": movement_selections["ca1"][
            "movement_parameters_sha256"
        ],
        "movement_speed_threshold_cm_s": float(
            shared_movement["speed_threshold_cm_s"]
        ),
        "movement_speed_sigma_s": float(
            shared_movement["speed_smoothing_sigma_s"]
        ),
        **movement_snapshots,
        "position_snapshot_sha256": _ripple_decoding_catalog_row_sha256(
            position_row
        ),
        "trajectory_intervals_sha256_by_type": trajectory_hashes,
        "graph_rows_sha256_by_trajectory": graph_row_hashes,
        "graph_policy_sha256": geometry["graph_policy_sha256"],
        **parameter_snapshot,
        "ripple_decoding_comparison_output_rule_sha256": (
            decoding.OUTPUT_RULE_SHA256
        ),
    }
    return {
        "ripple_decoding_comparison_id": selection_uuid(
            "RippleDecodingComparison", natural_key
        ),
        **natural_key,
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
            "catalog_row_sha256": _ripple_decoding_catalog_row_sha256(row),
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
        trajectory_type: _ripple_decoding_interval_sha256(
            selected_intervals[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    trajectory_row_hashes = {
        trajectory_type: _ripple_decoding_catalog_row_sha256(
            trajectory_rows[trajectory_type]
        )
        for trajectory_type in _DPP_TRAJECTORY_TYPES
    }
    graph_row_hashes = {
        trajectory_type: _ripple_decoding_catalog_row_sha256(
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
        "epoch_interval_row_sha256": _ripple_decoding_catalog_row_sha256(
            epoch_row
        ),
        "primary_position_row_sha256": _ripple_decoding_catalog_row_sha256(
            primary_row
        ),
        "orientation_reference_position_row_sha256": (
            _ripple_decoding_catalog_row_sha256(reference_row)
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
        "row_sha256": _ripple_decoding_catalog_row_sha256(row),
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
            ("region", region),
        ):
            if str(movement_selection.get(field_name)) != expected:
                raise ValueError(
                    f"CVPCA {condition} MovementFiringRate has wrong {field_name}."
                )
        for field_name in (
            "nwb_file_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
            "sorting_group_members_sha256",
            "unit_filter_params_sha256",
        ):
            if str(group_row.get(field_name)) != str(
                movement_selection.get(field_name)
            ):
                raise ValueError(
                    "CVPCA regional group and MovementFiringRate differ in "
                    f"{field_name}."
                )
        if str(group_row.get("selected_units_sha256")) != str(
            movement_result.get("selected_units_sha256")
        ) or int(group_row.get("n_units", -1)) != int(
            movement_result.get("n_units", -2)
        ):
            raise ValueError(
                "CVPCA regional group and MovementFiringRate units differ."
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
            trajectory: _ripple_decoding_catalog_row_sha256(
                trajectory_rows[condition][trajectory]
            )
            for trajectory in _DPP_TRAJECTORY_TYPES
        }
        for condition in ("light", "dark")
    }
    trajectory_interval_hashes = {
        condition: {
            trajectory: _ripple_decoding_interval_sha256(
                selected_intervals[condition][trajectory]
            )
            for trajectory in _DPP_TRAJECTORY_TYPES
        }
        for condition in ("light", "dark")
    }
    graph_row_hashes = {
        trajectory: _ripple_decoding_catalog_row_sha256(graph_rows[trajectory])
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
                _ripple_decoding_catalog_row_sha256(epoch_rows[condition])
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
        "region_group_row_sha256": _ripple_decoding_catalog_row_sha256(
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
                _ripple_decoding_catalog_row_sha256(
                    movement_selections[condition]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_result_row_sha256": (
                _ripple_decoding_catalog_row_sha256(
                    movement_results[condition]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_firing_rate_file_sha256": _file_sha256(
                Path(movement_results[condition]["movement_firing_rate_path"])
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_intervals_file_sha256": _file_sha256(
                Path(movement_results[condition]["movement_intervals_path"])
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_rates_sha256": (
                _ripple_decoding_movement_rates_sha256(
                    movement_loaded[condition]["table"]
                )
            )
            for condition in ("light", "dark")
        },
        **{
            f"{condition}_movement_support_sha256": (
                _ripple_decoding_interval_sha256(
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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
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
            "unit_filter_params_name",
            "sorted_spikes_group_name",
        )
    }
    natural_key["region"] = _analysis_region(key["region"])
    position_row = _fetch1_dict(position_table, natural_key)
    if position_row.get("spatial_unit") != "cm":
        raise ValueError("MovementFiringRate position must use centimeters.")
    parameters = _validate_movement_parameter_row(
        _fetch1_dict(parameters_table, natural_key)
    )
    provenance = _resolve_sorting_snapshot(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        key=natural_key,
    )
    snapshot = _sorting_snapshot_fields(provenance)
    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="movement_parameters_sha256",
    )
    identity_payload = {**natural_key, **snapshot, **parameter_snapshot}
    return {
        "movement_firing_rate_id": selection_uuid(
            "MovementFiringRate",
            identity_payload,
        ),
        **natural_key,
        **snapshot,
        **parameter_snapshot,
    }


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


def _dpp_encoding_comparison_selection_row(
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
            "DPPEncodingComparison requires a valid MovementFiringRate row."
        )
    for field_name in (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
    ):
        if str(region_row.get(field_name)) != str(
            movement_selection.get(field_name)
        ):
            raise ValueError(
                "RegionSortedSpikesGroup and MovementFiringRate must share "
                f"the same {field_name}."
            )
    if str(region_row.get("region_name")) != str(
        movement_selection.get("region")
    ):
        raise ValueError(
            "RegionSortedSpikesGroup and MovementFiringRate must select the "
            "same region."
        )
    for field_name in (
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
    ):
        if str(region_row.get(field_name)) != str(
            movement_selection.get(field_name)
        ):
            raise ValueError(
                "RegionSortedSpikesGroup and MovementFiringRate must share "
                f"the same frozen {field_name}."
            )
    if str(region_row.get("selected_units_sha256")) != str(
        movement_result.get("selected_units_sha256")
    ):
        raise ValueError(
            "RegionSortedSpikesGroup and MovementFiringRate must contain the "
            "same persistent units."
        )
    if int(region_row.get("n_units", -1)) != int(
        movement_result.get("n_units", -2)
    ):
        raise ValueError(
            "RegionSortedSpikesGroup and MovementFiringRate unit counts disagree."
        )

    nwb_file_name = str(movement_selection["nwb_file_name"])
    epoch = str(movement_selection["epoch"])
    for field_name, expected_value in (
        ("nwb_file_name", nwb_file_name),
        ("epoch", epoch),
    ):
        if field_name in key and str(key[field_name]) != expected_value:
            raise ValueError(
                "DPPEncodingComparison supplied source does not match its "
                f"MovementFiringRate: {field_name}."
            )
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("DPPEncodingComparison requires a run epoch.")

    parameters = _validate_dpp_encoding_comparison_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "dpp_encoding_comparison_param_name": key[
                    "dpp_encoding_comparison_param_name"
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
                f"DPPEncodingComparison {trajectory_field} must equal "
                f"{trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {**source_key, "trajectory_type": trajectory_type},
        )
        if int(trajectory_row.get("interval_count", -1)) < parameters["n_folds"]:
            raise ValueError(
                f"DPPEncodingComparison requires at least {parameters['n_folds']} "
                f"laps for {trajectory_type!r}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        supplied_configuration = str(
            key.get(configuration_field, trajectory_type)
        )
        if supplied_configuration != trajectory_type:
            raise ValueError(
                f"DPPEncodingComparison {configuration_field} must equal "
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
            raise ValueError("DPPEncodingComparison graphs must use centimeters.")
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
            "DPPEncodingComparison full_w_configuration_name must equal "
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
        raise ValueError("DPPEncodingComparison graphs must use centimeters.")
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
                "DPPEncodingComparison stability and movement rows must "
                "contain the same persistent units."
            )
        if int(stability_result.get("n_units", -1)) != int(
            movement_result.get("n_units", -2)
        ):
            raise ValueError(
                "DPPEncodingComparison stability and movement rows must "
                "contain the same unit count."
            )
        if str(stability_result.get("analysis_status")) not in {
            "valid",
            "no_valid_units",
        }:
            raise ValueError(
                "DPPEncodingComparison stability inputs must be valid or "
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
                        "DPPEncodingComparison stability input does not match "
                        f"its {trajectory_type!r} slot: {expected_field}."
                    )
        stability_fields[field_name] = stability_id

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="dpp_encoding_comparison_parameters_sha256",
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
        "dpp_encoding_comparison_param_name": parameters[
            "dpp_encoding_comparison_param_name"
        ],
        **parameter_snapshot,
    }
    return {
        "dpp_encoding_comparison_id": selection_uuid(
            "DPPEncodingComparison",
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
    from v1ca1.spyglass.decoding_comparison import TRANSFER_SPEC_SHA256
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
                "PathProgressionDecodingComparison requires target and cohort "
                "MovementFiringRate rows matching the regional unit count."
            )
        for field_name in (
            "nwb_file_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
        ):
            if str(region_row.get(field_name)) != str(selection.get(field_name)):
                raise ValueError(
                    "RegionSortedSpikesGroup and decoding movement sources "
                    f"must share {field_name}."
                )
        if str(region_row.get("region_name")) != str(selection.get("region")):
            raise ValueError(
                "RegionSortedSpikesGroup and decoding movement sources must "
                "select the same region."
            )
        for field_name in (
            "sorting_group_members_sha256",
            "unit_filter_params_sha256",
        ):
            if str(region_row.get(field_name)) != str(selection.get(field_name)):
                raise ValueError(
                    "RegionSortedSpikesGroup and decoding movement sources "
                    f"must share the frozen {field_name}."
                )
        if str(region_row.get("selected_units_sha256")) != str(
            result.get("selected_units_sha256")
        ) or int(region_row.get("n_units", -1)) != int(
            result.get("n_units", -2)
        ):
            raise ValueError(
                "RegionSortedSpikesGroup and decoding movement sources must "
                "contain the same persistent units."
            )
        epoch_row = _fetch1_dict(epoch_intervals_table, selection)
        if epoch_row.get("epoch_type") not in (None, "run"):
            raise ValueError(
                "PathProgressionDecodingComparison requires run epochs."
            )
        position_row = _fetch1_dict(position_table, selection)
        if str(position_row.get("spatial_unit")) != "cm":
            raise ValueError(
                "PathProgressionDecodingComparison position must use centimeters."
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
                "PathProgressionDecodingComparison supplied source does not "
                f"match its movement rows: {field_name}."
            )

    source_key = {"nwb_file_name": nwb_file_name, "epoch": epoch}
    source_fields: dict[str, Any] = {}
    for trajectory_type in _DPP_TRAJECTORY_TYPES:
        trajectory_field = f"{trajectory_type}_trajectory_type"
        if str(key.get(trajectory_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                f"PathProgressionDecodingComparison {trajectory_field} must "
                f"equal {trajectory_type!r}."
            )
        trajectory_row = _fetch1_dict(
            trajectory_intervals_table,
            {**source_key, "trajectory_type": trajectory_type},
        )
        if int(trajectory_row.get("interval_count", -1)) < 1:
            raise ValueError(
                "PathProgressionDecodingComparison requires at least one "
                f"lap for {trajectory_type!r}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "PathProgressionDecodingComparison graph aliases must match "
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
                "PathProgressionDecodingComparison graphs must use centimeters."
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
                    "PathProgressionDecodingComparison stability and movement "
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
                    "PathProgressionDecodingComparison stability sources must "
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
                            "PathProgressionDecodingComparison stability input "
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
        "path_progression_decoding_comparison_id": selection_uuid(
            "PathProgressionDecodingComparison",
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
    for field_name in (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
    ):
        if str(region_row.get(field_name)) != str(
            movement_selection.get(field_name)
        ):
            raise ValueError(
                "PathSpecificPlaceDecoding regional spikes and movement "
                f"must share {field_name}."
            )
    if str(region_row.get("region_name")) != str(
        movement_selection.get("region")
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding regional spikes and movement must "
            "select the same region."
        )
    for field_name in (
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
    ):
        if str(region_row.get(field_name)) != str(
            movement_selection.get(field_name)
        ):
            raise ValueError(
                "PathSpecificPlaceDecoding regional spikes and movement "
                f"must share frozen {field_name}."
            )
    if str(region_row.get("selected_units_sha256")) != str(
        movement_result.get("selected_units_sha256")
    ) or int(region_row.get("n_units", -1)) != int(
        movement_result.get("n_units", -2)
    ):
        raise ValueError(
            "PathSpecificPlaceDecoding regional spikes and movement must "
            "contain identical persistent units."
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
            "MotorEncodingComparison requires distinct primary and "
            "orientation-reference position series."
        )
    for label, row in (
        ("primary", primary_position_row),
        ("orientation-reference", orientation_reference_position_row),
    ):
        if str(row.get("spatial_unit")) != "cm":
            raise ValueError(
                f"MotorEncodingComparison {label} position must use centimeters."
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
                "MotorEncodingComparison position series must share aligned "
                f"sampling metadata: {field_name}."
            )


def _motor_encoding_comparison_selection_row(
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
    parameters = _validate_motor_encoding_comparison_parameter_row(
        _fetch1_dict(
            parameters_table,
            {
                "motor_encoding_comparison_param_name": key[
                    "motor_encoding_comparison_param_name"
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
    for field_name in (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
    ):
        if str(region_row.get(field_name)) != str(
            movement_selection.get(field_name)
        ):
            raise ValueError(
                "MotorEncodingComparison regional spikes and movement must "
                f"share {field_name}."
            )
    if str(region_row.get("region_name")) != str(
        movement_selection.get("region")
    ):
        raise ValueError(
            "MotorEncodingComparison regional spikes and movement must "
            "select the same region."
        )
    for field_name in (
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
    ):
        if str(region_row.get(field_name)) != str(
            movement_selection.get(field_name)
        ):
            raise ValueError(
                "MotorEncodingComparison regional spikes and movement must "
                f"share frozen {field_name}."
            )
    if str(region_row.get("selected_units_sha256")) != str(
        movement_result.get("selected_units_sha256")
    ) or int(region_row.get("n_units", -1)) != int(
        movement_result.get("n_units", -2)
    ):
        raise ValueError(
            "MotorEncodingComparison regional spikes and movement must "
            "contain identical persistent units."
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
            "MotorEncodingComparison movement status is incompatible with "
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
                "MotorEncodingComparison supplied source does not match its "
                f"MovementFiringRate: {field_name}."
            )
    epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {"nwb_file_name": nwb_file_name, "epoch": epoch},
    )
    if epoch_row.get("epoch_type") not in (None, "run"):
        raise ValueError("MotorEncodingComparison requires one run epoch.")

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
            "MotorEncodingComparison primary position must be the position "
            "used by MovementFiringRate."
        )
    if "orientation_reference_position_series_name" not in key:
        raise ValueError(
            "MotorEncodingComparison requires an explicit "
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
                f"MotorEncodingComparison {trajectory_field} must equal "
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
                "MotorEncodingComparison requires enough laps for outer and "
                f"nested inner CV for {trajectory_type!r}; found {n_laps}, "
                f"outer_n_folds={outer_n_folds}, "
                f"inner_n_folds={inner_n_folds}."
            )
        configuration_field = f"{trajectory_type}_configuration_name"
        if str(key.get(configuration_field, trajectory_type)) != trajectory_type:
            raise ValueError(
                "MotorEncodingComparison graph aliases must match their "
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
                "MotorEncodingComparison graphs must use centimeters."
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
            "MotorEncodingComparison full_w_configuration_name must equal "
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
        raise ValueError("MotorEncodingComparison graphs must use centimeters.")
    source_fields["full_w_configuration_name"] = full_w_name

    parameter_snapshot = _parameter_snapshot_field(
        parameters,
        field_name="motor_encoding_comparison_parameters_sha256",
    )
    model_spec_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_COMPARISON_MODEL_SPEC)
    )
    output_rule_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_COMPARISON_OUTPUT_RULE)
    )
    natural_key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "region_sorted_spikes_group_id": key[
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": key["movement_firing_rate_id"],
        **source_fields,
        "motor_encoding_comparison_param_name": parameters[
            "motor_encoding_comparison_param_name"
        ],
        **parameter_snapshot,
        "motor_encoding_comparison_model_spec_sha256": model_spec_sha256,
        "motor_encoding_comparison_output_rule_sha256": (
            output_rule_sha256
        ),
    }
    return {
        "motor_encoding_comparison_id": selection_uuid(
            "MotorEncodingComparison",
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
        for field_name in (
            "nwb_file_name",
            "unit_filter_params_name",
            "sorted_spikes_group_name",
        ):
            if str(region_row.get(field_name)) != str(
                movement_selection.get(field_name)
            ):
                raise ValueError(
                    "DarkLightGLM regional spikes and both movement rows must "
                    f"share {field_name}."
                )
        if str(region_row.get("region_name")) != str(
            movement_selection.get("region")
        ):
            raise ValueError(
                "DarkLightGLM regional spikes and both movement rows must "
                "select the same region."
            )
        for field_name in (
            "sorting_group_members_sha256",
            "unit_filter_params_sha256",
        ):
            if str(region_row.get(field_name)) != str(
                movement_selection.get(field_name)
            ):
                raise ValueError(
                    "DarkLightGLM regional spikes and both movement rows must "
                    f"share frozen {field_name}."
                )
        if str(region_row.get("selected_units_sha256")) != str(
            movement_result.get("selected_units_sha256")
        ) or int(region_row.get("n_units", -1)) != int(
            movement_result.get("n_units", -2)
        ):
            raise ValueError(
                "DarkLightGLM regional spikes and both movement rows must "
                "contain identical persistent units."
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
        "unit_filter_params_name",
        "sorted_spikes_group_name",
        "region",
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
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
    """Load one canonical dark/light bundle and freeze its selected files."""
    from v1ca1.spyglass.dark_light_glm import (
        MANIFEST_COLUMNS,
        MANIFEST_FILENAME,
    )
    from v1ca1.spyglass.swap_glm import SOURCE_MODEL_NAMES
    import pandas as pd

    key = {"dark_light_glm_id": dark_light_glm_id}
    result_row = _fetch1_dict(dark_light_glm_table, key)
    manifest_path = Path(result_row["artifact_manifest_path"])
    if manifest_path.name != MANIFEST_FILENAME:
        raise ValueError(
            "DarkLightGLM artifact_manifest_path has a noncanonical name."
        )
    if manifest_path.parent.name != str(dark_light_glm_id):
        raise ValueError(
            "DarkLightGLM artifact directory does not match its result UUID."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"DarkLightGLM artifact manifest not found: {manifest_path}"
        )
    manifest = pd.read_parquet(manifest_path)
    if tuple(manifest.columns) != MANIFEST_COLUMNS or manifest.empty:
        raise ValueError("DarkLightGLM manifest does not have canonical schema.")
    if manifest["artifact_key"].duplicated().any():
        raise ValueError("DarkLightGLM manifest artifact keys must be unique.")
    for _, row in manifest.iterrows():
        relative_path = Path(str(row["relative_path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("DarkLightGLM manifest contains an unsafe path.")
        artifact_path = manifest_path.parent / relative_path
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"DarkLightGLM artifact not found: {artifact_path}"
            )
        if artifact_path.stat().st_size != int(row["file_size_bytes"]) or (
            _file_sha256(artifact_path) != str(row["sha256"])
        ):
            raise ValueError(
                f"DarkLightGLM artifact checksum mismatch: {artifact_path}"
            )
    first = manifest.iloc[0]
    for field_name in MANIFEST_COLUMNS[5:]:
        if not np.all(
            manifest[field_name].astype(str) == str(first[field_name])
        ):
            raise ValueError(
                "DarkLightGLM manifest has inconsistent common field "
                f"{field_name!r}."
            )
    if str(first["dark_light_glm_id"]) != str(dark_light_glm_id):
        raise ValueError("DarkLightGLM artifact has a mismatched result UUID.")
    expected_scalars = {
        "analysis_status": first["analysis_status"],
        "selected_units_sha256": first["selected_units_sha256"],
        "n_units": first["n_units"],
        "schema_version": first["schema_version"],
        "artifact_origin": first["artifact_origin"],
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row.get(field_name)) != str(expected_value):
            raise ValueError(
                "DarkLightGLM result row disagrees with its canonical "
                f"artifact: {field_name}."
            )
    selected_hashes: dict[str, str] = {}
    for model_name in SOURCE_MODEL_NAMES:
        rows = manifest.loc[
            manifest["artifact_key"].astype(str)
            == f"selected:{model_name}"
        ]
        if len(rows) != 1:
            raise ValueError(
                "DarkLightGLM manifest must contain exactly one selected "
                f"artifact for {model_name!r}."
            )
        digest = str(rows.iloc[0]["sha256"])
        if len(digest) != 64:
            raise ValueError(
                "DarkLightGLM selected-model checksum is not SHA-256."
            )
        selected_hashes[model_name] = digest
    return {
        "result_row": result_row,
        "artifact_dir": manifest_path.parent,
        "manifest_sha256": _file_sha256(manifest_path),
        "selected_sha256_by_model": selected_hashes,
        "parameter_sha256": str(first["parameter_sha256"]),
        "output_rule_sha256": str(first["output_rule_sha256"]),
        "analysis_status": str(first["analysis_status"]),
        "metadata": {
            field_name: str(first[field_name])
            for field_name in (
                "dark_light_glm_id",
                "animal_name",
                "date",
                "region",
                "light_epoch",
                "dark_epoch",
            )
        },
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
        "unit_filter_params_name",
        "sorted_spikes_group_name",
        "region",
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
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
    for field_name in (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
    ):
        if str(region_row.get(field_name)) != str(
            light_test_movement_selection.get(field_name)
        ):
            raise ValueError(
                "SwapGLM regional spikes and held-out movement must share "
                f"{field_name}."
            )
    if str(region_row.get("region_name")) != str(
        light_test_movement_selection.get("region")
    ):
        raise ValueError(
            "SwapGLM regional spikes and held-out movement must select the "
            "same region."
        )
    if str(region_row.get("selected_units_sha256")) != str(
        light_test_movement_result.get("selected_units_sha256")
    ) or int(region_row.get("n_units", -1)) != int(
        light_test_movement_result.get("n_units", -2)
    ):
        raise ValueError(
            "SwapGLM regional spikes and held-out movement must contain "
            "identical persistent units."
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
        "dark_light_manifest_sha256": snapshot["manifest_sha256"],
        "dark_light_selected_sha256_by_model": snapshot[
            "selected_sha256_by_model"
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
        load_path_specific_place_artifact,
    )

    curve_key = {"path_specific_place_tuning_curve_id": tuning_curve_id}
    result_row = _fetch1_dict(tuning_curve_table, curve_key)
    selection_row = _fetch1_dict(tuning_curve_selection_table, curve_key)
    artifact_path = Path(result_row["tuning_curve_path"]).resolve(strict=True)
    curve = load_path_specific_place_artifact(artifact_path)
    try:
        _validate_tuning_curve_artifact_link(
            curve=curve,
            result_row=result_row,
            selection_row=selection_row,
        )
    finally:
        close = getattr(curve, "close", None)
        if callable(close):
            close()
    return {
        "selection": selection_row,
        "result": result_row,
        "artifact_path": artifact_path,
        "artifact_sha256": _file_sha256(artifact_path),
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
            movement_firing_rate_sha256[epoch_role] = _file_sha256(
                Path(
                    movement_result["movement_firing_rate_path"]
                ).resolve(strict=True)
            )
            movement_intervals_sha256[epoch_role] = _file_sha256(
                Path(
                    movement_result["movement_intervals_path"]
                ).resolve(strict=True)
            )

    movement_reference = movement_selections["dark"]
    shared_movement_fields = (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
        "region",
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
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
    for field_name in (
        "nwb_file_name",
        "unit_filter_params_name",
        "sorted_spikes_group_name",
        "sorting_group_members_sha256",
        "unit_filter_params_sha256",
    ):
        if str(region_row.get(field_name)) != str(
            movement_reference.get(field_name)
        ):
            raise ValueError(
                "SwapTuningCurveComparison regional spikes and movement "
                f"rows must share {field_name}."
            )
    if region != str(movement_reference.get("region")):
        raise ValueError(
            "SwapTuningCurveComparison regional spikes and movement rows "
            "must select the same region."
        )
    for epoch_role, movement_result in movement_results.items():
        if str(movement_result.get("selected_units_sha256")) != (
            selected_units_sha256
        ) or int(movement_result.get("n_units", -1)) != n_units:
            raise ValueError(
                "SwapTuningCurveComparison movement rows must contain "
                f"identical persistent units: {epoch_role}."
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
    """Return NetCDF-safe identity linking one curve to its selection row."""
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
    """Return NetCDF-safe identity linking one DPP curve to its selection."""
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


def _load_group_spikes(
    *,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    key: Mapping[str, Any],
    region: str,
    time_support: tuple[float, float],
) -> dict[str, Any]:
    """Build one Pynapple TsGroup from all validated sorting-group members."""
    from v1ca1.spyglass.spikes import load_sorted_spikes_group

    return load_sorted_spikes_group(
        sorted_spikes_group,
        unit_selection_params,
        spike_sorting_output,
        key,
        region=region,
        time_support=time_support,
        allow_empty=True,
    )


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
            "RippleGLM selected Ripples row changed after selection insertion."
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
) -> dict[str, Any]:
    """Compute and write one immutable RippleGLM artifact bundle."""
    from v1ca1.spyglass.ripple_glm import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        compute_ripple_glm,
        get_ripple_glm_artifact_paths,
        write_ripple_glm_artifact,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_ripple_glm_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        ripple_glm_id=selection["ripple_glm_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_ripple_glm_artifact(result, artifact_dir)
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "summary_path": str(written["summary_path"]),
        "ripple_glm_path": str(written["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        **{
            field_name: result[field_name]
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
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


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
        RESULT_SCHEMA_VERSION,
        get_ripple_glm_artifact_paths,
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

    artifact_dir = Path(result_row["artifact_manifest_path"]).parent
    try:
        artifact_root = artifact_dir.parents[4]
    except IndexError as exc:
        raise ValueError(
            "RippleGLM artifact does not use the canonical bundle layout."
        ) from exc
    expected_paths = get_ripple_glm_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=str(selection_row["epoch"]),
        ripple_glm_id=selection_row["ripple_glm_id"],
        artifact_root=artifact_root,
    )
    path_fields = {
        "artifact_manifest_path": "artifact_manifest_path",
        "selected_units_path": "selected_units_path",
        "summary_path": "summary_path",
        "ripple_glm_path": "result_path",
    }
    for row_field, path_key in path_fields.items():
        if Path(result_row[row_field]) != Path(expected_paths[path_key]):
            raise ValueError(
                "RippleGLM result paths do not use the canonical bundle "
                f"layout: {row_field}."
            )
    expected_scalars = {
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
    }
    expected_scalars.update(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
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
) -> dict[str, Any]:
    """Strictly reconstruct and register one existing RippleGLM NetCDF."""
    from v1ca1.spyglass.ripple_glm import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        get_ripple_glm_artifact_paths,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_ripple_glm_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        ripple_glm_id=selection["ripple_glm_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    try:
        registered = register_existing_ripple_glm_artifact(
            source_result_path=Path(source_result_path),
            destination_path=artifact_dir,
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
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    written = registered["artifact_paths"]
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "summary_path": str(written["summary_path"]),
        "ripple_glm_path": str(written["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        **{
            field_name: registered[field_name]
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
        "legacy_artifact_provenance": dict(
            registered["legacy_artifact_provenance"]
        ),
        "_created_artifact_paths": created_artifact_paths,
    }


def _cross_region_xcorr_upstream_provenance(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact CrossRegionXCorr selection snapshot for artifacts."""
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
        "cross_region_xcorr_parameters_sha256",
        "cross_region_xcorr_output_rule_sha256",
    )
    provenance = {
        "cross_region_xcorr_id": str(selection["cross_region_xcorr_id"]),
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
        name="CrossRegionXCorrSelection.speed_gated",
    )
    return provenance


def _validate_cross_region_xcorr_upstream_link(
    upstream: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Require a CrossRegionXCorr bundle to embed its exact selection."""
    if dict(upstream) != _cross_region_xcorr_upstream_provenance(selection):
        raise ValueError(
            "CrossRegionXCorr upstream provenance does not match its "
            "immutable selection."
        )


def _load_cross_region_xcorr_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Reload and verify all lightweight CrossRegionXCorr inputs."""
    from v1ca1.spyglass import cross_region_xcorr

    selection = dict(key)
    parameters = _validate_cross_region_xcorr_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    _validate_frozen_parameters(
        selection,
        parameters,
        field_name="cross_region_xcorr_parameters_sha256",
    )
    if str(selection.get("cross_region_xcorr_output_rule_sha256")) != (
        cross_region_xcorr.OUTPUT_RULE_SHA256
    ):
        raise ValueError(
            "CrossRegionXCorr fixed output rule changed after selection "
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
        name="CrossRegionXCorrSelection.speed_gated",
    )
    if not math.isclose(
        selected_detector_threshold,
        detector_zscore_threshold,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ) or selected_speed_gated != speed_gated:
        raise ValueError(
            "CrossRegionXCorr detector values changed after selection "
            "insertion."
        )
    selection["detector_zscore_threshold"] = selected_detector_threshold
    selection["speed_gated"] = selected_speed_gated
    if int(ripple_row["ripple_count"]) != int(
        selection["source_ripple_count"]
    ):
        raise ValueError(
            "CrossRegionXCorr selected Ripples row changed after selection "
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
                "CrossRegionXCorr regional groups must remain in the "
                "selected NWB file."
            )
        if str(row.get("region_name")) != expected_region:
            raise ValueError(
                f"CrossRegionXCorr {role} region must remain "
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
        cross_region_xcorr.prepare_cross_region_xcorr_event_selection(
            epoch=str(selection["epoch"]),
            ripple_table=ripple_table,
        )
    )
    normalized_ripples = event_selection["selected_ripple_table"]
    if int(event_selection["n_ripples"]) != int(
        selection["source_ripple_count"]
    ):
        raise ValueError(
            "CrossRegionXCorr exact ripple count changed after selection."
        )
    if str(event_selection["selected_ripple_intervals_sha256"]) != str(
        selection["selected_ripple_intervals_sha256"]
    ):
        raise ValueError(
            "CrossRegionXCorr exact ripple boundaries changed after selection."
        )
    if _ripple_glm_provenance_sha256(ripple_row) != str(
        selection["ripple_provenance_sha256"]
    ):
        raise ValueError(
            "CrossRegionXCorr ripple detector provenance changed after "
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


def _make_cross_region_xcorr_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one immutable CrossRegionXCorr bundle."""
    from v1ca1.spyglass.cross_region_xcorr import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        compute_cross_region_xcorr,
        get_cross_region_xcorr_artifact_paths,
        write_cross_region_xcorr_artifact,
    )

    context = _load_cross_region_xcorr_context(
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
        analysis_name="CrossRegionXCorr",
    )
    selection = context["selection"]
    parameters = context["parameters"]
    result = compute_cross_region_xcorr(
        cross_region_xcorr_id=selection["cross_region_xcorr_id"],
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        ripple_table=context["ripple_table"],
        ca1_spikes=loaded["source"]["ts_group"],
        ca1_stable_unit_ids=loaded["source"]["unit_ids"],
        v1_spikes=loaded["target"]["ts_group"],
        v1_stable_unit_ids=loaded["target"]["unit_ids"],
        upstream_provenance=_cross_region_xcorr_upstream_provenance(
            selection
        ),
        expected_selected_ripple_intervals_sha256=selection[
            "selected_ripple_intervals_sha256"
        ],
        parameter_name=parameters["cross_region_xcorr_param_name"],
        parameter_sha256=selection[
            "cross_region_xcorr_parameters_sha256"
        ],
        output_rule_sha256=selection[
            "cross_region_xcorr_output_rule_sha256"
        ],
        **_cross_region_xcorr_parameter_kwargs(parameters),
    )
    _validate_cross_region_xcorr_upstream_link(
        result["upstream_provenance"], selection
    )
    if str(result["selected_ripple_intervals_sha256"]) != str(
        selection["selected_ripple_intervals_sha256"]
    ):
        raise ValueError(
            "CrossRegionXCorr computation changed its exact ripple intervals."
        )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_cross_region_xcorr_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        cross_region_xcorr_id=selection["cross_region_xcorr_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_cross_region_xcorr_artifact(result, artifact_dir)
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "ca1_units_path": str(written["ca1_units_path"]),
        "v1_units_path": str(written["v1_units_path"]),
        "summary_path": str(written["summary_path"]),
        "cross_region_xcorr_path": str(written["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        **{
            field_name: result[field_name]
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
                "ca1_units_sha256",
                "v1_units_sha256",
                "summary_sha256",
                "analysis_status",
            )
        },
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


def _validate_cross_region_xcorr_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one loaded xcorr bundle to match its DataJoint rows."""
    from v1ca1.spyglass.cross_region_xcorr import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        get_cross_region_xcorr_artifact_paths,
        validate_cross_region_xcorr_result,
    )

    validated = validate_cross_region_xcorr_result(bundle)
    expected_metadata = {
        "cross_region_xcorr_id": str(
            selection_row["cross_region_xcorr_id"]
        ),
        "animal_name": str(animal_name),
        "date": str(date),
        "epoch": str(selection_row["epoch"]),
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated.get(field_name)) != expected_value:
            raise ValueError(
                "CrossRegionXCorr artifact does not match its selection: "
                f"{field_name}."
            )
    parameters = _validate_cross_region_xcorr_parameter_row(parameters_row)
    expected_parameters = {
        "parameter_name": parameters["cross_region_xcorr_param_name"],
        "parameter_sha256": selection_row[
            "cross_region_xcorr_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "cross_region_xcorr_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "cross_region_xcorr_param_name"
        },
    }
    if validated["parameters"] != expected_parameters:
        raise ValueError(
            "CrossRegionXCorr artifact parameters do not match its selection."
        )
    _validate_cross_region_xcorr_upstream_link(
        validated["upstream_provenance"], selection_row
    )
    artifact_dir = Path(result_row["artifact_manifest_path"]).parent
    try:
        artifact_root = artifact_dir.parents[4]
    except IndexError as exc:
        raise ValueError(
            "CrossRegionXCorr artifact does not use the canonical layout."
        ) from exc
    expected_paths = get_cross_region_xcorr_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=str(selection_row["epoch"]),
        cross_region_xcorr_id=selection_row["cross_region_xcorr_id"],
        artifact_root=artifact_root,
    )
    path_fields = {
        "artifact_manifest_path": "artifact_manifest_path",
        "ca1_units_path": "ca1_units_path",
        "v1_units_path": "v1_units_path",
        "summary_path": "summary_path",
        "cross_region_xcorr_path": "result_path",
    }
    for row_field, path_key in path_fields.items():
        if Path(result_row[row_field]) != Path(expected_paths[path_key]):
            raise ValueError(
                "CrossRegionXCorr result paths do not use the canonical "
                f"layout: {row_field}."
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
                "CrossRegionXCorr result metadata disagrees with its "
                f"artifact: {field_name}."
            )
    if not math.isclose(
        float(result_row["ripple_duration_s"]),
        float(validated["ripple_duration_s"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "CrossRegionXCorr result duration disagrees with its artifact."
        )
    if str(result_row.get("schema_version")) != RESULT_SCHEMA_VERSION or str(
        result_row.get("bundle_schema_version")
    ) != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "CrossRegionXCorr result schema versions disagree with its artifact."
        )
    if result_row.get("legacy_artifact_provenance") != validated.get(
        "legacy_artifact_provenance"
    ):
        raise ValueError(
            "CrossRegionXCorr result-row legacy provenance differs from its "
            "artifact."
        )


def _legacy_cross_region_xcorr_identity_resolver(
    loaded_spikes: Mapping[str, Any],
    *,
    role: str,
) -> Callable[[Any], list[dict[str, str]]]:
    """Build one sequence resolver for legacy imported-sorting unit IDs."""
    identity_by_sorting_id = _legacy_ripple_glm_unit_identity_resolver(
        loaded_spikes,
        role=role,
        analysis_name="CrossRegionXCorr",
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
                    f"Legacy CrossRegionXCorr {role} unit "
                    f"{legacy_unit_id!r} has {len(matches)} imported-sorting "
                    "identity matches."
                )
            resolved.append(dict(matches[0]))
        return resolved

    return resolve


def _register_existing_cross_region_xcorr_row(
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
) -> dict[str, Any]:
    """Recompute and register one exact legacy four-artifact xcorr set."""
    from v1ca1.spyglass.cross_region_xcorr import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        get_cross_region_xcorr_artifact_paths,
        register_existing_cross_region_xcorr_artifact,
    )

    context = _load_cross_region_xcorr_context(
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
        analysis_name="CrossRegionXCorr",
    )
    resolvers = {
        role: _legacy_cross_region_xcorr_identity_resolver(
            loaded[role],
            role=role,
        )
        for role in ("source", "target")
    }
    selection = context["selection"]
    parameters = context["parameters"]
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_cross_region_xcorr_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        cross_region_xcorr_id=selection["cross_region_xcorr_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    try:
        registered = register_existing_cross_region_xcorr_artifact(
            source_ca1_unit_filter_path=Path(source_ca1_unit_filter_path),
            source_v1_unit_filter_path=Path(source_v1_unit_filter_path),
            source_summary_path=Path(source_summary_path),
            source_result_path=Path(source_result_path),
            destination_path=artifact_dir,
            cross_region_xcorr_id=selection["cross_region_xcorr_id"],
            animal_name=context["animal_name"],
            date=context["date"],
            epoch=str(selection["epoch"]),
            ripple_table=context["ripple_table"],
            ca1_spikes=loaded["source"]["ts_group"],
            ca1_stable_unit_ids=loaded["source"]["unit_ids"],
            v1_spikes=loaded["target"]["ts_group"],
            v1_stable_unit_ids=loaded["target"]["unit_ids"],
            upstream_provenance=_cross_region_xcorr_upstream_provenance(
                selection
            ),
            expected_selected_ripple_intervals_sha256=selection[
                "selected_ripple_intervals_sha256"
            ],
            ca1_legacy_identity_resolver=resolvers["source"],
            v1_legacy_identity_resolver=resolvers["target"],
            ca1_sorting_type="ImportedSpikeSorting",
            v1_sorting_type="ImportedSpikeSorting",
            parameter_name=parameters["cross_region_xcorr_param_name"],
            parameter_sha256=selection[
                "cross_region_xcorr_parameters_sha256"
            ],
            output_rule_sha256=selection[
                "cross_region_xcorr_output_rule_sha256"
            ],
            source_v1ca1_git_commit=source_v1ca1_git_commit,
            source_spyglass_git_commit=source_spyglass_git_commit,
            overwrite=False,
            **_cross_region_xcorr_parameter_kwargs(parameters),
        )
        _validate_cross_region_xcorr_upstream_link(
            registered["upstream_provenance"], selection
        )
        if str(registered["selected_ripple_intervals_sha256"]) != str(
            selection["selected_ripple_intervals_sha256"]
        ):
            raise ValueError(
                "Registered CrossRegionXCorr ripple boundaries disagree "
                "with its selection."
            )
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    return {
        "artifact_manifest_path": str(registered["artifact_manifest_path"]),
        "ca1_units_path": str(registered["ca1_units_path"]),
        "v1_units_path": str(registered["v1_units_path"]),
        "summary_path": str(registered["summary_path"]),
        "cross_region_xcorr_path": str(registered["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        **{
            field_name: registered[field_name]
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
                "ca1_units_sha256",
                "v1_units_sha256",
                "summary_sha256",
                "analysis_status",
            )
        },
        "legacy_artifact_provenance": dict(
            registered["legacy_artifact_provenance"]
        ),
        "_created_artifact_paths": created_artifact_paths,
    }


def _load_ripple_decoding_comparison_context(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Reload and verify every frozen ripple-decoding upstream input."""
    selection = _ripple_decoding_comparison_selection_row(
        key=key,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        parameters_table=parameters_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    if str(selection["ripple_decoding_comparison_id"]) != str(
        key["ripple_decoding_comparison_id"]
    ):
        raise ValueError("RippleDecodingComparison selection UUID is stale.")
    parameters = _validate_ripple_decoding_comparison_parameter_row(
        _fetch1_dict(parameters_table, selection)
    )
    _validate_frozen_parameters(
        selection,
        parameters,
        field_name="ripple_decoding_comparison_parameters_sha256",
    )
    region_rows = {
        region: _fetch1_dict(
            region_sorted_spikes_group_table,
            {
                "region_sorted_spikes_group_id": selection[
                    f"{region}_region_sorted_spikes_group_id"
                ]
            },
        )
        for region in ("ca1", "v1")
    }
    movement_results = {}
    movement_selections = {}
    movement_parameters = {}
    movement = {}
    animal_name, session_date = _session_identity(session_table, selection)
    for region in ("ca1", "v1"):
        movement_key = {
            "movement_firing_rate_id": selection[
                f"{region}_movement_firing_rate_id"
            ]
        }
        movement_results[region] = _fetch1_dict(
            movement_firing_rate_table, movement_key
        )
        movement_selections[region] = _fetch1_dict(
            movement_firing_rate_selection_table, movement_key
        )
        movement_parameters[region] = _validate_movement_parameter_row(
            _fetch1_dict(
                movement_parameters_table, movement_selections[region]
            )
        )
        _validate_frozen_parameters(
            movement_selections[region],
            movement_parameters[region],
            field_name="movement_parameters_sha256",
        )
        movement[region] = _load_movement_result_artifacts(
            result_row=movement_results[region],
            parameters=movement_parameters[region],
            expected_metadata={
                "animal_name": animal_name,
                "date": session_date,
                "region": region,
                "epoch": selection["train_epoch"],
            },
        )
        current = {
            f"{region}_movement_firing_rate_sha256": _file_sha256(
                Path(movement_results[region]["movement_firing_rate_path"])
            ),
            f"{region}_movement_intervals_sha256": _file_sha256(
                Path(movement_results[region]["movement_intervals_path"])
            ),
            f"{region}_movement_rates_sha256": (
                _ripple_decoding_movement_rates_sha256(
                    movement[region]["table"]
                )
            ),
            f"{region}_movement_support_sha256": (
                _ripple_decoding_interval_sha256(
                    movement[region]["movement_intervals"]
                )
            ),
        }
        for field_name, value in current.items():
            if str(selection[field_name]) != str(value):
                raise ValueError(
                    "RippleDecodingComparison movement artifact changed "
                    f"after selection: {field_name}."
                )
    train_epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {
            "nwb_file_name": selection["nwb_file_name"],
            "epoch": selection["train_epoch"],
        },
    )
    decode_epoch_row = _fetch1_dict(
        epoch_intervals_table,
        {
            "nwb_file_name": selection["nwb_file_name"],
            "epoch": selection["decode_epoch"],
        },
    )
    return {
        "selection": selection,
        "parameters": parameters,
        "region_rows": region_rows,
        "movement_results": movement_results,
        "movement_selections": movement_selections,
        "movement_parameters": movement_parameters,
        "movement": movement,
        "train_epoch_row": train_epoch_row,
        "decode_epoch_row": decode_epoch_row,
        "animal_name": animal_name,
        "date": session_date,
    }


def _load_ripple_decoding_comparison_nwb_inputs(
    *,
    context: Mapping[str, Any],
    ripples_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    nwbfile_table: Any,
) -> dict[str, Any]:
    """Load selected position, laps, graphs, ripples, and decode epoch once."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )

    selection = dict(context["selection"])
    nwb_file_name = str(selection["nwb_file_name"])
    train_epoch = str(selection["train_epoch"])
    decode_epoch = str(selection["decode_epoch"])
    position_row = _fetch1_dict(
        position_table,
        {
            "nwb_file_name": nwb_file_name,
            "epoch": train_epoch,
            "position_series_name": selection["train_position_series_name"],
        },
    )
    ripple_row = _fetch1_dict(
        ripples_table,
        {"nwb_file_name": nwb_file_name, "epoch": decode_epoch},
    )
    trajectory_rows = {
        trajectory_type: _fetch1_dict(
            trajectory_intervals_table,
            {
                "nwb_file_name": nwb_file_name,
                "epoch": train_epoch,
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
        str(nwb_path), mode="r", load_namespaces=True
    ) as io:
        nwbfile = io.read()
        position = load_position(
            nwbfile, position_row, apply_analysis_offset=True
        )
        ripple_intervals = load_interval_set(nwbfile, ripple_row)
        decode_epoch_interval = load_interval_set(
            nwbfile, context["decode_epoch_row"]
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
        "ripple_table": _intervals_to_frame(
            ripple_intervals, epoch=decode_epoch
        ),
        "decode_epoch_interval": decode_epoch_interval,
        "trajectory_intervals": trajectory_intervals,
        "graph_inputs": graph_inputs,
        "position_row": position_row,
    }


def _load_ripple_decoding_comparison_spikes(
    *,
    context: Mapping[str, Any],
    region_sorted_spikes_group_table: Any,
) -> dict[str, dict[str, Any]]:
    """Load both regional groups and align frozen movement rates by stable id."""
    from v1ca1.spyglass.selection import unit_identity_sha256

    epoch_bounds = (
        float(context["train_epoch_row"]["start_time"]),
        float(context["train_epoch_row"]["stop_time"]),
        float(context["decode_epoch_row"]["start_time"]),
        float(context["decode_epoch_row"]["stop_time"]),
    )
    if not all(math.isfinite(value) for value in epoch_bounds):
        raise ValueError("RippleDecodingComparison epoch bounds must be finite.")
    time_support = (min(epoch_bounds[0], epoch_bounds[2]), max(epoch_bounds[1], epoch_bounds[3]))
    if time_support[1] <= time_support[0]:
        raise ValueError("RippleDecodingComparison epoch support is empty.")
    loaded = {}
    for region in ("ca1", "v1"):
        group = region_sorted_spikes_group_table.load_spikes(
            {
                "region_sorted_spikes_group_id": context["selection"][
                    f"{region}_region_sorted_spikes_group_id"
                ]
            },
            time_support=time_support,
        )
        expected_count = int(context["selection"][f"{region}_n_units"])
        expected_digest = str(
            context["selection"][f"{region}_selected_units_sha256"]
        )
        if int(group["n_units"]) != expected_count or unit_identity_sha256(
            group["unit_ids"]
        ) != expected_digest:
            raise ValueError(
                f"RippleDecodingComparison {region} units changed after selection."
            )
        rate_table = context["movement"][region]["table"]
        rate_by_stable = {
            str(row["stable_unit_id"]): float(row["movement_firing_rate_hz"])
            for row in rate_table.to_dict("records")
        }
        stable_ids = [
            f"{row['spikesorting_merge_id']}:{row['unit_id']}"
            for row in group["unit_ids"]
        ]
        if set(stable_ids) != set(rate_by_stable) or len(rate_by_stable) != len(
            stable_ids
        ):
            raise ValueError(
                f"RippleDecodingComparison {region} movement rates and "
                "regional units differ."
            )
        loaded[region] = {
            **group,
            "movement_firing_rates_hz": np.asarray(
                [rate_by_stable[stable_id] for stable_id in stable_ids],
                dtype=float,
            ),
        }
    return loaded


def _validate_ripple_decoding_comparison_upstream_link(
    upstream: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    """Require one artifact to embed the exact immutable selection snapshot."""
    if dict(upstream) != _ripple_decoding_comparison_upstream_provenance(selection):
        raise ValueError(
            "RippleDecodingComparison artifact upstream provenance differs "
            "from its immutable selection."
        )


def _ripple_decoding_comparison_compute_kwargs(
    *,
    context: Mapping[str, Any],
    loaded_spikes: Mapping[str, Mapping[str, Any]],
    nwb_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact standalone-compute arguments from frozen inputs."""
    selection = context["selection"]
    parameters = context["parameters"]
    return {
        "ripple_decoding_comparison_id": selection[
            "ripple_decoding_comparison_id"
        ],
        "animal_name": context["animal_name"],
        "date": context["date"],
        "train_epoch": str(selection["train_epoch"]),
        "decode_epoch": str(selection["decode_epoch"]),
        "representation": str(selection["representation"]),
        "ca1_spikes": loaded_spikes["ca1"]["ts_group"],
        "ca1_stable_unit_ids": loaded_spikes["ca1"]["unit_ids"],
        "ca1_movement_firing_rates_hz": loaded_spikes["ca1"][
            "movement_firing_rates_hz"
        ],
        "v1_spikes": loaded_spikes["v1"]["ts_group"],
        "v1_stable_unit_ids": loaded_spikes["v1"]["unit_ids"],
        "v1_movement_firing_rates_hz": loaded_spikes["v1"][
            "movement_firing_rates_hz"
        ],
        "position": nwb_inputs["position"],
        "trajectory_intervals": nwb_inputs["trajectory_intervals"],
        "graph_inputs": nwb_inputs["graph_inputs"],
        "movement_interval": context["movement"]["ca1"][
            "movement_intervals"
        ],
        "ripple_table": nwb_inputs["ripple_table"],
        "decode_epoch_interval": nwb_inputs["decode_epoch_interval"],
        "upstream_provenance": (
            _ripple_decoding_comparison_upstream_provenance(selection)
        ),
        "parameter_name": parameters[
            "ripple_decoding_comparison_param_name"
        ],
        "parameter_sha256": selection[
            "ripple_decoding_comparison_parameters_sha256"
        ],
        "output_rule_sha256": selection[
            "ripple_decoding_comparison_output_rule_sha256"
        ],
        "expected_selected_ripple_intervals_sha256": selection[
            "selected_ripple_intervals_sha256"
        ],
        "expected_graph_policy_sha256": selection["graph_policy_sha256"],
        **_ripple_decoding_comparison_parameter_kwargs(parameters),
    }


def _ripple_decoding_comparison_result_row(
    result: Mapping[str, Any],
    written: Mapping[str, Any],
    *,
    created_artifact_paths: Sequence[str],
) -> dict[str, Any]:
    """Return the DataJoint result payload for one validated bundle."""
    from v1ca1.spyglass.ripple_decoding_comparison import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
    )

    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "ripple_qc_path": str(written["ripple_qc_path"]),
        "ripple_metrics_path": str(written["ripple_metrics_path"]),
        "epoch_summary_path": str(written["epoch_summary_path"]),
        "ripple_decoding_comparison_path": str(written["result_path"]),
        "ca1_decoded_path": str(written["ca1_decoded_path"]),
        "v1_decoded_path": str(written["v1_decoded_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        **{
            field_name: result[field_name]
            for field_name in (
                "n_ripple_events_input",
                "n_ripples",
                "n_ripple_bins",
                "n_ca1_units",
                "n_v1_units",
                "n_ca1_units_in_decoder",
                "n_v1_units_in_decoder",
                "selected_ripple_intervals_sha256",
                "graph_policy_sha256",
                "selected_units_sha256",
                "ripple_qc_sha256",
                "ripple_metrics_sha256",
                "epoch_summary_sha256",
                "analysis_status",
            )
        },
        "legacy_artifact_provenance": (
            dict(result["legacy_artifact_provenance"])
            if result["legacy_artifact_provenance"]
            else None
        ),
        "_created_artifact_paths": list(created_artifact_paths),
    }


def _make_ripple_decoding_comparison_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one immutable RippleDecodingComparison bundle."""
    from v1ca1.spyglass.ripple_decoding_comparison import (
        compute_ripple_decoding_comparison,
        get_ripple_decoding_comparison_artifact_paths,
        write_ripple_decoding_comparison_artifact,
    )

    context = _load_ripple_decoding_comparison_context(
        key=key,
        parameters_table=parameters_table,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    loaded_spikes = _load_ripple_decoding_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_ripple_decoding_comparison_nwb_inputs(
        context=context,
        ripples_table=ripples_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    result = compute_ripple_decoding_comparison(
        **_ripple_decoding_comparison_compute_kwargs(
            context=context,
            loaded_spikes=loaded_spikes,
            nwb_inputs=nwb_inputs,
        )
    )
    selection = context["selection"]
    _validate_ripple_decoding_comparison_upstream_link(
        result["upstream_provenance"], selection
    )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_ripple_decoding_comparison_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        train_epoch=str(selection["train_epoch"]),
        decode_epoch=str(selection["decode_epoch"]),
        representation=str(selection["representation"]),
        ripple_decoding_comparison_id=selection[
            "ripple_decoding_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_ripple_decoding_comparison_artifact(result, artifact_dir)
    return _ripple_decoding_comparison_result_row(
        result,
        written,
        created_artifact_paths=created_artifact_paths,
    )


def _validate_ripple_decoding_comparison_artifact_link(
    *,
    bundle: Mapping[str, Any],
    result_row: Mapping[str, Any],
    selection_row: Mapping[str, Any],
    parameters_row: Mapping[str, Any],
    animal_name: str,
    date: str,
) -> None:
    """Require one loaded decoding bundle to match its DataJoint rows."""
    from v1ca1.spyglass.ripple_decoding_comparison import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        get_ripple_decoding_comparison_artifact_paths,
        validate_ripple_decoding_comparison_result,
    )

    validated = validate_ripple_decoding_comparison_result(bundle)
    expected_metadata = {
        "ripple_decoding_comparison_id": str(
            selection_row["ripple_decoding_comparison_id"]
        ),
        "animal_name": str(animal_name),
        "date": str(date),
        "train_epoch": str(selection_row["train_epoch"]),
        "decode_epoch": str(selection_row["decode_epoch"]),
        "representation": str(selection_row["representation"]),
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated.get(field_name)) != expected_value:
            raise ValueError(
                "RippleDecodingComparison artifact does not match its "
                f"selection: {field_name}."
            )
    parameters = _validate_ripple_decoding_comparison_parameter_row(
        parameters_row
    )
    expected_parameters = {
        "parameter_name": parameters[
            "ripple_decoding_comparison_param_name"
        ],
        "parameter_sha256": selection_row[
            "ripple_decoding_comparison_parameters_sha256"
        ],
        "output_rule_sha256": selection_row[
            "ripple_decoding_comparison_output_rule_sha256"
        ],
        **_ripple_decoding_comparison_parameter_kwargs(parameters),
    }
    if validated["parameters"] != expected_parameters:
        raise ValueError(
            "RippleDecodingComparison artifact parameters do not match its "
            "selection."
        )
    _validate_ripple_decoding_comparison_upstream_link(
        validated["upstream_provenance"], selection_row
    )
    artifact_dir = Path(result_row["artifact_manifest_path"]).parent
    try:
        artifact_root = artifact_dir.parents[5]
    except IndexError as exc:
        raise ValueError(
            "RippleDecodingComparison artifact does not use the canonical layout."
        ) from exc
    expected_paths = get_ripple_decoding_comparison_artifact_paths(
        animal_name=animal_name,
        date=date,
        train_epoch=str(selection_row["train_epoch"]),
        decode_epoch=str(selection_row["decode_epoch"]),
        representation=str(selection_row["representation"]),
        ripple_decoding_comparison_id=selection_row[
            "ripple_decoding_comparison_id"
        ],
        artifact_root=artifact_root,
    )
    path_fields = {
        "artifact_manifest_path": "artifact_manifest_path",
        "selected_units_path": "selected_units_path",
        "ripple_qc_path": "ripple_qc_path",
        "ripple_metrics_path": "ripple_metrics_path",
        "epoch_summary_path": "epoch_summary_path",
        "ripple_decoding_comparison_path": "result_path",
        "ca1_decoded_path": "ca1_decoded_path",
        "v1_decoded_path": "v1_decoded_path",
    }
    for row_field, path_key in path_fields.items():
        if Path(result_row[row_field]) != Path(expected_paths[path_key]):
            raise ValueError(
                "RippleDecodingComparison result paths do not use the "
                f"canonical layout: {row_field}."
            )
    for field_name in (
        "n_ripple_events_input",
        "n_ripples",
        "n_ripple_bins",
        "n_ca1_units",
        "n_v1_units",
        "n_ca1_units_in_decoder",
        "n_v1_units_in_decoder",
        "selected_ripple_intervals_sha256",
        "graph_policy_sha256",
        "selected_units_sha256",
        "ripple_qc_sha256",
        "ripple_metrics_sha256",
        "epoch_summary_sha256",
        "analysis_status",
        "artifact_origin",
    ):
        if str(result_row.get(field_name)) != str(validated[field_name]):
            raise ValueError(
                "RippleDecodingComparison result metadata disagrees with "
                f"its artifact: {field_name}."
            )
    if str(result_row.get("schema_version")) != RESULT_SCHEMA_VERSION or str(
        result_row.get("bundle_schema_version")
    ) != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            "RippleDecodingComparison result schema versions disagree with "
            "its artifact."
        )
    expected_legacy = validated.get("legacy_artifact_provenance") or None
    if result_row.get("legacy_artifact_provenance") != expected_legacy:
        raise ValueError(
            "RippleDecodingComparison result-row legacy provenance differs "
            "from its artifact."
        )


def _legacy_ripple_decoding_comparison_identity_resolver(
    loaded_spikes: Mapping[str, Any],
    *,
    region: str,
) -> Callable[[Any], list[dict[str, str]]]:
    """Build one strict imported-sorting resolver for a legacy unit axis."""
    identity_by_sorting_id = _legacy_ripple_glm_unit_identity_resolver(
        loaded_spikes,
        role=region,
        analysis_name="RippleDecodingComparison",
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
                    f"Legacy RippleDecodingComparison {region} unit "
                    f"{legacy_unit_id!r} has {len(matches)} imported-sorting "
                    "identity matches."
                )
            resolved.append(dict(matches[0]))
        return resolved

    return resolve


def _register_existing_ripple_decoding_comparison_row(
    *,
    key: Mapping[str, Any],
    source_ca1_decoded_path: Path,
    source_v1_decoded_path: Path,
    source_ripple_metrics_path: Path,
    source_epoch_summary_path: Path,
    source_result_path: Path,
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    region_sorted_spikes_group_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    position_table: Any,
    trajectory_intervals_table: Any,
    wtrack_graph_table: Any,
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Redecode, verify, and register one complete five-file legacy set."""
    from v1ca1.spyglass.ripple_decoding_comparison import (
        get_ripple_decoding_comparison_artifact_paths,
        register_existing_ripple_decoding_comparison_artifact,
    )

    context = _load_ripple_decoding_comparison_context(
        key=key,
        parameters_table=parameters_table,
        ripples_table=ripples_table,
        epoch_intervals_table=epoch_intervals_table,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
        movement_firing_rate_table=movement_firing_rate_table,
        movement_firing_rate_selection_table=(
            movement_firing_rate_selection_table
        ),
        movement_parameters_table=movement_parameters_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        session_table=session_table,
        nwbfile_table=nwbfile_table,
    )
    loaded_spikes = _load_ripple_decoding_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_ripple_decoding_comparison_nwb_inputs(
        context=context,
        ripples_table=ripples_table,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    selection = context["selection"]
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_ripple_decoding_comparison_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        train_epoch=str(selection["train_epoch"]),
        decode_epoch=str(selection["decode_epoch"]),
        representation=str(selection["representation"]),
        ripple_decoding_comparison_id=selection[
            "ripple_decoding_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    try:
        registered = register_existing_ripple_decoding_comparison_artifact(
            source_ca1_decoded_path=Path(source_ca1_decoded_path),
            source_v1_decoded_path=Path(source_v1_decoded_path),
            source_ripple_metrics_path=Path(source_ripple_metrics_path),
            source_epoch_summary_path=Path(source_epoch_summary_path),
            source_result_path=Path(source_result_path),
            destination_path=artifact_dir,
            ca1_legacy_identity_resolver=(
                _legacy_ripple_decoding_comparison_identity_resolver(
                    loaded_spikes["ca1"], region="ca1"
                )
            ),
            v1_legacy_identity_resolver=(
                _legacy_ripple_decoding_comparison_identity_resolver(
                    loaded_spikes["v1"], region="v1"
                )
            ),
            ca1_sorting_type="ImportedSpikeSorting",
            v1_sorting_type="ImportedSpikeSorting",
            source_v1ca1_git_commit=source_v1ca1_git_commit,
            source_spyglass_git_commit=source_spyglass_git_commit,
            overwrite=False,
            **_ripple_decoding_comparison_compute_kwargs(
                context=context,
                loaded_spikes=loaded_spikes,
                nwb_inputs=nwb_inputs,
            ),
        )
        _validate_ripple_decoding_comparison_upstream_link(
            registered["upstream_provenance"], selection
        )
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    return _ripple_decoding_comparison_result_row(
        registered,
        registered,
        created_artifact_paths=created_artifact_paths,
    )


def _make_ripple_modulation_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    ripples_table: Any,
    epoch_intervals_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one keyed RippleModulation result."""
    import pynwb

    from v1ca1.spyglass.nwb import load_interval_set
    from v1ca1.spyglass.ripple_modulation import (
        compute_epoch_region_ripple_modulation,
        empty_ripple_modulation_result,
        get_ripple_modulation_artifact_paths,
        write_ripple_modulation_artifacts,
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
    region = _analysis_region(key["region"])

    loaded_spikes = _load_group_spikes(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=region,
        time_support=(epoch_start, epoch_stop),
    )
    _validate_frozen_sorting_snapshot(key, loaded_spikes)
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
            "Ripples catalog row."
        )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_ripple_modulation_artifact_paths(
        animal_name=animal_name,
        date=session_date,
        epoch=str(key["epoch"]),
        region=region,
        ripple_modulation_id=key["ripple_modulation_id"],
        **path_kwargs,
    )
    created_artifact_paths = [
        str(paths[name])
        for name in ("summary", "peri_ripple_firing_rate")
        if not Path(paths[name]).exists()
    ]
    written = write_ripple_modulation_artifacts(result, paths)
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
    return {
        "summary_path": str(written["summary"]),
        "peri_ripple_firing_rate_path": str(written["peri_ripple_firing_rate"]),
        "n_ripples": int(result["n_ripples"]),
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_spikes["unit_ids"]),
        "legacy_artifact_provenance": None,
        "artifact_origin": "computed",
        "_created_artifact_paths": created_artifact_paths,
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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Validate, key, and write existing RippleModulation Parquets."""
    from v1ca1.spyglass.ripple_modulation import (
        plan_register_existing,
        read_planned_artifacts,
        write_planned_artifacts,
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
    artifact_key = {
        "animal_name": animal_name,
        "date": session_date,
        "epoch": str(key["epoch"]),
        "region": _analysis_region(key["region"]),
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
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=artifact_key["region"],
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(key, loaded_units)
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
    prepared_tables = {
        "summary": selected_summary,
        "peri_ripple_firing_rate": selected_peri,
    }
    for copy in plan["copies"]:
        artifact_name = str(copy["artifact"])
        if (
            not copy.get("copy_required", True)
            and not overwrite
            and not selected_tables[artifact_name].equals(
                prepared_tables[artifact_name]
            )
        ):
            raise ValueError(
                "A same-path registration source requires stable-unit "
                f"normalization and cannot be registered in place: {copy['source']}."
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
            "Ripples catalog row."
        )
    created_artifact_paths = [
        str(copy["destination"])
        for copy in plan["copies"]
        if (copy.get("copy_required", True) or overwrite)
        and not Path(copy["destination"]).exists()
    ]
    destinations = write_planned_artifacts(
        plan,
        prepared_tables,
        overwrite=overwrite,
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
    return {
        "summary_path": str(destinations["summary"]),
        "peri_ripple_firing_rate_path": str(destinations["peri_ripple_firing_rate"]),
        "n_ripples": summary_n_ripples,
        "n_units": n_units,
        "n_valid_units": n_valid_units,
        "analysis_status": analysis_status,
        "selected_units_sha256": unit_identity_sha256(loaded_units["unit_ids"]),
        "legacy_artifact_provenance": legacy_artifact_provenance,
        "artifact_origin": "registered_existing",
        "_created_artifact_paths": created_artifact_paths,
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
    paths: Mapping[str, Any],
    *,
    created_artifact_paths: Sequence[str],
) -> dict[str, Any]:
    """Return one DataJoint payload from a validated standalone result."""
    from v1ca1.spyglass.epoch_motor_behavior import (
        BUNDLE_SCHEMA_VERSION,
        SCHEMA_VERSION,
    )

    return {
        "artifact_manifest_path": str(paths["artifact_manifest_path"]),
        "distribution_summary_path": str(paths["distribution_summary_path"]),
        "progression_summary_path": str(paths["progression_summary_path"]),
        "trajectory_qc_path": str(paths["trajectory_qc_path"]),
        "schema_version": SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
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
) -> dict[str, Any]:
    """Compute and write one immutable epoch motor-behavior bundle."""
    from v1ca1.spyglass.epoch_motor_behavior import (
        compute_selected_epoch_motor_behavior,
        get_epoch_motor_behavior_artifact_paths,
        write_epoch_motor_behavior_artifact,
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
    selection = context["selection"]
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_epoch_motor_behavior_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        epoch_motor_behavior_id=selection["epoch_motor_behavior_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_epoch_motor_behavior_artifact(result, artifact_dir)
    return _epoch_motor_behavior_result_row(
        result,
        written,
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

    artifact_dir = Path(result_row["artifact_manifest_path"]).parent
    try:
        artifact_root = artifact_dir.parents[4]
    except IndexError as exc:
        raise ValueError(
            "EpochMotorBehavior artifact does not use the canonical layout."
        ) from exc
    expected_paths = motor_behavior.get_epoch_motor_behavior_artifact_paths(
        animal_name=animal_name,
        date=date,
        epoch=str(selection_row["epoch"]),
        epoch_motor_behavior_id=selection_row["epoch_motor_behavior_id"],
        artifact_root=artifact_root,
    )
    for field_name in (
        "artifact_manifest_path",
        "distribution_summary_path",
        "progression_summary_path",
        "trajectory_qc_path",
    ):
        if Path(result_row[field_name]) != Path(expected_paths[field_name]):
            raise ValueError(
                "EpochMotorBehavior result path is not canonical: "
                f"{field_name}."
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
    if str(result_row.get("schema_version")) != motor_behavior.SCHEMA_VERSION or str(
        result_row.get("bundle_schema_version")
    ) != motor_behavior.BUNDLE_SCHEMA_VERSION:
        raise ValueError("EpochMotorBehavior schema versions disagree.")
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
) -> dict[str, Any]:
    """Recompute, verify, and register one legacy session motor artifact."""
    from v1ca1.spyglass.epoch_motor_behavior import (
        get_epoch_motor_behavior_artifact_paths,
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
    selection = context["selection"]
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_epoch_motor_behavior_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=str(selection["epoch"]),
        epoch_motor_behavior_id=selection["epoch_motor_behavior_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    try:
        registered = register_existing_epoch_motor_behavior_artifact(
            source_distribution_path=Path(source_distribution_path),
            source_progression_path=Path(source_progression_path),
            source_run_log_path=(
                None
                if source_run_log_path is None
                else Path(source_run_log_path)
            ),
            destination_path=artifact_dir,
            overwrite=False,
            **_epoch_motor_behavior_compute_kwargs(context),
        )
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    return _epoch_motor_behavior_result_row(
        registered,
        paths,
        created_artifact_paths=created_artifact_paths,
    )


def _make_movement_firing_rate_row(
    *,
    key: Mapping[str, Any],
    parameters_table: Any,
    epoch_intervals_table: Any,
    position_table: Any,
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one epoch-wide movement firing-rate bundle."""
    import pynwb

    from v1ca1.spyglass.movement import (
        compute_selected_movement_firing_rate,
        get_movement_artifact_paths,
        write_movement_artifacts,
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

    region = _analysis_region(key["region"])
    loaded_spikes = _load_group_spikes(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=key,
        region=region,
        time_support=(epoch_start, epoch_stop),
    )
    _validate_frozen_sorting_snapshot(key, loaded_spikes)

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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_movement_artifact_paths(
        animal_name=animal_name,
        date=session_date,
        epoch=str(key["epoch"]),
        region=region,
        movement_firing_rate_id=key["movement_firing_rate_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = (
        [
            str(paths["firing_rate_path"]),
            str(paths["movement_intervals_path"]),
        ]
        if not artifact_dir.exists()
        else []
    )
    written = write_movement_artifacts(
        result["table"],
        result["movement_intervals"],
        artifact_dir,
    )
    return {
        "movement_firing_rate_path": str(written["firing_rate_path"]),
        "movement_intervals_path": str(written["movement_intervals_path"]),
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "n_units_with_spikes": int(result["n_units_with_spikes"]),
        "movement_interval_count": int(result["movement_interval_count"]),
        "movement_duration_s": float(result["movement_duration_s"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": unit_identity_sha256(
            loaded_spikes["unit_ids"]
        ),
        "_created_artifact_paths": created_artifact_paths,
    }


def _load_movement_result_artifacts(
    *,
    result_row: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a movement bundle and verify its DataJoint summary scalars."""
    from v1ca1.spyglass.movement import (
        load_movement_firing_rate_artifact,
        load_movement_interval_artifact,
        movement_interval_summary,
        validate_movement_artifacts,
    )

    table = load_movement_firing_rate_artifact(
        Path(result_row["movement_firing_rate_path"])
    )
    movement_intervals = load_movement_interval_artifact(
        Path(result_row["movement_intervals_path"])
    )
    validate_movement_artifacts(table, movement_intervals)
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
            parameters=movement_parameters[condition],
            expected_metadata={
                "animal_name": animal_name,
                "date": session_date,
                "region": selection.get("_resolved_region", "") or str(
                    movement_selections[condition]["region"]
                ),
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
        "light_movement_firing_rate_file_sha256",
        "dark_movement_firing_rate_file_sha256",
        "light_movement_intervals_file_sha256",
        "dark_movement_intervals_file_sha256",
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


def _cv_pca_result_row(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one DataJoint payload from a validated standalone cvPCA bundle."""
    from v1ca1.spyglass import cv_pca
    from v1ca1.spyglass.selection import unit_identity_sha256

    paths = result["artifact_paths"]
    selected_units = result["selected_units"]
    identities = selected_units.loc[
        :, ["spikesorting_merge_id", "unit_id"]
    ].to_dict("records")
    return {
        "artifact_manifest_path": str(paths["manifest_path"]),
        "result_path": str(paths["result_path"]),
        "summary_path": str(paths["summary_path"]),
        "spectrum_path": str(paths["spectrum_path"]),
        "selected_units_path": str(paths["selected_units_path"]),
        "lap_assignments_path": str(paths["lap_assignments_path"]),
        "trajectory_qc_path": str(paths["trajectory_qc_path"]),
        "result_schema_version": cv_pca.RESULT_SCHEMA_VERSION,
        "bundle_schema_version": cv_pca.BUNDLE_SCHEMA_VERSION,
        "n_input_units": int(result["n_input_units"]),
        "n_selected_units": int(result["n_selected_units"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": unit_identity_sha256(identities),
        "legacy_artifact_provenance": (
            dict(result["legacy_artifact_provenance"])
            if result.get("legacy_artifact_provenance")
            else None
        ),
        "_created_artifact_paths": list(
            result.get("_created_artifact_paths", ())
        ),
    }


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
) -> dict[str, Any]:
    """Compute and write one immutable cvPCA bundle."""
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
    write_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        write_kwargs["artifact_root"] = artifact_root
    written = cv_pca.write_cv_pca_artifact(result, **write_kwargs)
    return _cv_pca_result_row(written)


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

    artifact_dir = Path(result_row["artifact_manifest_path"]).parent
    try:
        artifact_root = artifact_dir.parents[5]
    except IndexError as exc:
        raise ValueError("CVPCA artifact does not use the canonical layout.") from exc
    expected_paths = cv_pca.get_cv_pca_artifact_paths(
        animal_name=animal_name,
        date=date,
        light_epoch=selection_row["light_epoch"],
        dark_epoch=selection_row["dark_epoch"],
        region=region_row["region_name"],
        cv_pca_id=selection_row["cv_pca_id"],
        artifact_root=artifact_root,
    )
    for field_name, path_key in (
        ("artifact_manifest_path", "manifest_path"),
        ("result_path", "result_path"),
        ("summary_path", "summary_path"),
        ("spectrum_path", "spectrum_path"),
        ("selected_units_path", "selected_units_path"),
        ("lap_assignments_path", "lap_assignments_path"),
        ("trajectory_qc_path", "trajectory_qc_path"),
    ):
        if Path(result_row[field_name]) != Path(expected_paths[path_key]):
            raise ValueError(f"CVPCA result path is not canonical: {field_name}.")
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
    if str(result_row["result_schema_version"]) != cv_pca.RESULT_SCHEMA_VERSION or str(
        result_row["bundle_schema_version"]
    ) != cv_pca.BUNDLE_SCHEMA_VERSION:
        raise ValueError("CVPCA schema versions disagree.")
    artifact_legacy = validated.get("legacy_artifact_provenance") or None
    if result_row.get("legacy_artifact_provenance") != artifact_legacy:
        raise ValueError("CVPCA legacy provenance disagrees.")
    if provenance_sha256(parameters) != str(
        selection_row["cv_pca_parameters_sha256"]
    ):
        raise ValueError("CVPCA parameter snapshot is stale.")


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
) -> dict[str, Any]:
    """Strictly recompute, compare, and register one legacy cvPCA pair."""
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
    register_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        register_kwargs["artifact_root"] = artifact_root
    registered = cv_pca.register_existing_cv_pca_artifact(
        legacy_result_path=Path(legacy_result_path),
        legacy_summary_path=Path(legacy_summary_path),
        compute_inputs=_cv_pca_compute_inputs(context),
        overwrite=False,
        **register_kwargs,
    )
    return _cv_pca_result_row(registered)


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
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one trial-subset path-specific tuning curve."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.path_specific_place import (
        compute_selected_path_specific_place_tuning_curve,
        get_path_specific_place_artifact_path,
        write_path_specific_place_artifact,
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

    loaded_spikes = _load_group_spikes(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=context,
        region=_analysis_region(movement_selection["region"]),
        time_support=(epoch_start, epoch_stop),
    )
    _validate_frozen_sorting_snapshot(movement_selection, loaded_spikes)
    selected_units_sha256 = unit_identity_sha256(loaded_spikes["unit_ids"])
    if selected_units_sha256 != str(
        movement_result["selected_units_sha256"]
    ):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": _analysis_region(movement_selection["region"]),
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
        region=_analysis_region(movement_selection["region"]),
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    artifact_path = get_path_specific_place_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(movement_selection["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        trial_subset=str(key["trial_subset"]),
        region=_analysis_region(movement_selection["region"]),
        path_specific_place_tuning_curve_id=key[
            "path_specific_place_tuning_curve_id"
        ],
        **path_kwargs,
    )
    created_artifact_paths = [] if artifact_path.exists() else [str(artifact_path)]
    written_path = write_path_specific_place_artifact(
        result["tuning_curve"],
        artifact_path,
    )
    return {
        "tuning_curve_path": str(written_path),
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
        "_created_artifact_paths": created_artifact_paths,
    }


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
    session_table: Any,
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write one trial-subset DPP tuning curve."""
    import pynwb

    from v1ca1.spyglass.dpp import (
        compute_selected_dpp_tuning_curve,
        get_dpp_artifact_path,
        get_dpp_trajectory_pair,
        write_dpp_artifact,
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

    loaded_spikes = _load_group_spikes(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=context,
        region=_analysis_region(movement_selection["region"]),
        time_support=(epoch_start, epoch_stop),
    )
    _validate_frozen_sorting_snapshot(movement_selection, loaded_spikes)
    selected_units_sha256 = unit_identity_sha256(loaded_spikes["unit_ids"])
    if selected_units_sha256 != str(
        movement_result["selected_units_sha256"]
    ):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
        parameters=movement_parameters,
        expected_metadata={
            "animal_name": animal_name,
            "date": session_date,
            "region": _analysis_region(movement_selection["region"]),
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
        region=_analysis_region(movement_selection["region"]),
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    artifact_path = get_dpp_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(movement_selection["epoch"]),
        turn_type=str(key["turn_type"]),
        trial_subset=str(key["trial_subset"]),
        region=_analysis_region(movement_selection["region"]),
        dpp_tuning_curve_id=key["dpp_tuning_curve_id"],
        **path_kwargs,
    )
    created_artifact_paths = [] if artifact_path.exists() else [str(artifact_path)]
    written_path = write_dpp_artifact(
        result["tuning_curve"],
        artifact_path,
    )
    return {
        "tuning_curve_path": str(written_path),
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
        "_created_artifact_paths": created_artifact_paths,
    }


def _make_path_specific_place_stability_row(
    *,
    key: Mapping[str, Any],
    tuning_curve_table: Any,
    tuning_curve_selection_table: Any,
    movement_firing_rate_table: Any,
    movement_firing_rate_selection_table: Any,
    movement_parameters_table: Any,
    session_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute stability from one persisted odd/even tuning-curve pair."""
    from v1ca1.spyglass.path_specific_place import (
        load_path_specific_place_artifact,
    )
    from v1ca1.spyglass.stability import (
        compute_selected_stability_from_tuning_curves,
        get_stability_artifact_path,
        write_stability_artifact,
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
        curves[subset] = load_path_specific_place_artifact(
            Path(curve_results[subset]["tuning_curve_path"])
        )
        _validate_tuning_curve_artifact_link(
            curve=curves[subset],
            result_row=curve_results[subset],
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
    region = _analysis_region(movement_selection["region"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    artifact_path = get_stability_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(movement_selection["epoch"]),
        trajectory_type=str(odd_selection["trajectory_type"]),
        region=region,
        path_specific_place_stability_id=key[
            "path_specific_place_stability_id"
        ],
        **path_kwargs,
    )
    created_artifact_paths = [] if artifact_path.exists() else [str(artifact_path)]
    written_path = write_stability_artifact(result["table"], artifact_path)
    return {
        "stability_path": str(written_path),
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
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
    session_table: Any,
) -> dict[str, Any]:
    """Load and cross-check the four curves and shared movement artifact."""
    from v1ca1.spyglass.path_specific_place import (
        load_path_specific_place_artifact,
    )

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
        curve = load_path_specific_place_artifact(
            Path(result_row["tuning_curve_path"])
        )
        _validate_tuning_curve_artifact_link(
            curve=curve,
            result_row=result_row,
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
    region = _analysis_region(movement_selection["region"])
    epoch = str(movement_selection["epoch"])
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
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
    session_table: Any,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Compute and write four path comparisons for every selected unit."""
    from v1ca1.spyglass.tuning_similarity import (
        compute_tuning_similarity_from_curves,
        get_tuning_similarity_artifact_path,
        write_tuning_similarity_artifact,
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
        session_table=session_table,
    )
    result = compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=inputs["curves"],
        movement_firing_rate_table=inputs["movement_table"],
        similarity_metric=inputs["parameters"]["similarity_metric"],
    )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    artifact_path = get_tuning_similarity_artifact_path(
        animal_name=inputs["animal_name"],
        date=inputs["date"],
        epoch=inputs["epoch"],
        region=inputs["region"],
        similarity_metric=inputs["parameters"]["similarity_metric"],
        path_specific_place_tuning_similarity_id=key[
            "path_specific_place_tuning_similarity_id"
        ],
        **path_kwargs,
    )
    created_artifact_paths = [] if artifact_path.exists() else [str(artifact_path)]
    written_path = write_tuning_similarity_artifact(
        result["table"],
        artifact_path,
    )
    return {
        "similarity_path": str(written_path),
        "n_units": int(result["n_units"]),
        "n_valid_comparisons": int(result["n_valid_comparisons"]),
        "n_units_with_valid_comparison": int(
            result["n_units_with_valid_comparison"]
        ),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": inputs["selected_units_sha256"],
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Validate and register one matching complete legacy similarity file."""
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent
    from v1ca1.spyglass.tuning_similarity import (
        get_tuning_similarity_artifact_path,
        register_existing_tuning_similarity_artifact,
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
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=context,
        region=inputs["region"],
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(inputs["movement_selection"], loaded_units)
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

    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    destination = get_tuning_similarity_artifact_path(
        animal_name=inputs["animal_name"],
        date=inputs["date"],
        epoch=inputs["epoch"],
        region=inputs["region"],
        similarity_metric=inputs["parameters"]["similarity_metric"],
        path_specific_place_tuning_similarity_id=key[
            "path_specific_place_tuning_similarity_id"
        ],
        **path_kwargs,
    )
    registered = register_existing_tuning_similarity_artifact(
        source_path=Path(similarity_path),
        destination_path=destination,
        tuning_curves_by_trajectory=inputs["curves"],
        movement_firing_rate_table=inputs["movement_table"],
        similarity_metric=inputs["parameters"]["similarity_metric"],
        unit_identity_resolver=unit_identity_resolver,
        overwrite=overwrite,
    )
    provenance = dict(registered["legacy_artifact_provenance"])
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
    return {
        "similarity_path": str(registered["similarity_path"]),
        "n_units": int(registered["n_units"]),
        "n_valid_comparisons": int(registered["n_valid_comparisons"]),
        "n_units_with_valid_comparison": int(
            registered["n_units_with_valid_comparison"]
        ),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": list(
            registered.get("_created_artifact_paths", ())
        ),
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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Subset and normalize one legacy all-trial place-tuning artifact."""
    import pynwb

    from v1ca1.spyglass.nwb import (
        load_interval_set,
        load_position,
        load_wtrack_graph,
    )
    from v1ca1.spyglass.path_specific_place import (
        build_path_specific_linear_position,
        get_path_specific_place_artifact_path,
        graph_length_from_inputs,
        register_existing_path_specific_place_artifact,
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
    region = _analysis_region(movement_selection["region"])
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=context,
        region=region,
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(movement_selection, loaded_units)
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

    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    destination = get_path_specific_place_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(movement_selection["epoch"]),
        trajectory_type=str(key["trajectory_type"]),
        trial_subset="all",
        region=region,
        path_specific_place_tuning_curve_id=key[
            "path_specific_place_tuning_curve_id"
        ],
        **path_kwargs,
    )
    registered = register_existing_path_specific_place_artifact(
        source_path=Path(tuning_curve_path),
        destination_path=destination,
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
        artifact_attributes=_tuning_curve_artifact_attributes(
            key,
            selected_units_sha256=selected_units_sha256,
        ),
        overwrite=overwrite,
    )
    provenance = dict(registered["legacy_artifact_provenance"])
    provenance.update(
        {
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
    )
    return {
        "tuning_curve_path": str(registered["tuning_curve_path"]),
        "n_units": int(registered["n_units"]),
        "n_valid_units": int(registered["n_valid_units"]),
        "n_trials": int(registered["n_trials"]),
        "support_duration_s": float(registered["support_duration_s"]),
        "n_feature_samples": int(registered["n_feature_samples"]),
        "n_position_bins": int(registered["n_position_bins"]),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": list(
            registered.get("_created_artifact_paths", ())
        ),
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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Subset and normalize one legacy all-trial DPP tuning artifact."""
    import pynwb

    from v1ca1.spyglass.dpp import (
        common_graph_length_from_inputs,
        get_dpp_artifact_path,
        get_dpp_trajectory_pair,
        register_existing_dpp_artifact,
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
    region = _analysis_region(movement_selection["region"])
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=context,
        region=region,
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(movement_selection, loaded_units)
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

    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    destination = get_dpp_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(movement_selection["epoch"]),
        turn_type=str(key["turn_type"]),
        trial_subset="all",
        region=region,
        dpp_tuning_curve_id=key["dpp_tuning_curve_id"],
        **path_kwargs,
    )
    registered = register_existing_dpp_artifact(
        source_path=Path(tuning_curve_path),
        destination_path=destination,
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
        artifact_attributes=_dpp_tuning_curve_artifact_attributes(
            key,
            selected_units_sha256=selected_units_sha256,
        ),
        overwrite=overwrite,
    )
    provenance = dict(registered["legacy_artifact_provenance"])
    provenance.update(
        {
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
    )
    return {
        "tuning_curve_path": str(registered["tuning_curve_path"]),
        "n_units": int(registered["n_units"]),
        "n_valid_units": int(registered["n_valid_units"]),
        "n_trials": int(registered["n_trials"]),
        "n_outbound_trials": int(registered["n_outbound_trials"]),
        "n_inbound_trials": int(registered["n_inbound_trials"]),
        "support_duration_s": float(registered["support_duration_s"]),
        "n_feature_samples": int(registered["n_feature_samples"]),
        "n_position_bins": int(registered["n_position_bins"]),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": selected_units_sha256,
        "artifact_origin": "registered_existing",
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": list(
            registered.get("_created_artifact_paths", ())
        ),
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
    sorted_spikes_group: Any,
    unit_selection_params: Any,
    spike_sorting_output: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Filter and register one partition of the complete legacy artifact."""
    import pandas as pd

    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.spikes import resolve_merge_parent
    from v1ca1.spyglass.path_specific_place import (
        load_path_specific_place_artifact,
    )
    from v1ca1.spyglass.stability import (
        compute_selected_stability_from_tuning_curves,
        empty_stability_table,
        get_stability_artifact_path,
        write_stability_artifact,
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
        curves[subset] = load_path_specific_place_artifact(
            Path(curve_results[subset]["tuning_curve_path"])
        )
        _validate_tuning_curve_artifact_link(
            curve=curves[subset],
            result_row=curve_results[subset],
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
    region = _analysis_region(movement_selection["region"])
    loaded_units = _load_group_unit_data(
        sorted_spikes_group=sorted_spikes_group,
        unit_selection_params=unit_selection_params,
        spike_sorting_output=spike_sorting_output,
        key=context,
        region=region,
        allow_empty=True,
    )
    _validate_frozen_sorting_snapshot(movement_selection, loaded_units)
    selected_units_sha256 = unit_identity_sha256(loaded_units["unit_ids"])
    if selected_units_sha256 != str(
        movement_result["selected_units_sha256"]
    ):
        raise ValueError(
            "MovementFiringRate selected units changed after computation."
        )
    movement = _load_movement_result_artifacts(
        result_row=movement_result,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    destination = get_stability_artifact_path(
        animal_name=animal_name,
        date=session_date,
        epoch=str(movement_selection["epoch"]),
        trajectory_type=str(curve_selection["trajectory_type"]),
        region=region,
        path_specific_place_stability_id=key[
            "path_specific_place_stability_id"
        ],
        **path_kwargs,
    )
    created_artifact_paths = [] if destination.exists() else [str(destination)]
    written_path = write_stability_artifact(
        selected,
        destination,
        overwrite=overwrite,
    )
    return {
        "stability_path": str(written_path),
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
        "_created_artifact_paths": created_artifact_paths,
    }


def _load_dpp_stability_artifact(
    *,
    result_row: Mapping[str, Any],
    trajectory_type: str,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
) -> Any:
    """Load and cross-check one selected stability Parquet."""
    import pandas as pd

    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.stability import (
        ARTIFACT_FILENAME,
        empty_stability_table,
    )

    path = Path(result_row["stability_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"PathSpecificPlaceStability artifact not found: {path}"
        )
    stability_id = str(result_row["path_specific_place_stability_id"])
    if path.name != ARTIFACT_FILENAME or path.parent.name != stability_id:
        raise ValueError(
            "PathSpecificPlaceStability artifact path does not contain its "
            "result UUID."
        )
    table = pd.read_parquet(path)
    missing = sorted(set(empty_stability_table().columns).difference(table))
    if missing:
        raise ValueError(
            "PathSpecificPlaceStability artifact is missing canonical "
            f"columns {missing!r}."
        )
    if table["stable_unit_id"].astype(str).duplicated().any():
        raise ValueError(
            "PathSpecificPlaceStability artifact has duplicate stable units."
        )
    if len(table) != int(result_row["n_units"]):
        raise ValueError(
            "PathSpecificPlaceStability result unit count disagrees with its "
            "artifact."
        )
    n_valid = int(table["stability_status"].astype(str).eq("valid").sum())
    if n_valid != int(result_row["n_valid_units"]):
        raise ValueError(
            "PathSpecificPlaceStability valid-unit count disagrees with its "
            "artifact."
        )
    expected_status = "valid" if n_valid else (
        "no_units" if table.empty else "no_valid_units"
    )
    if str(result_row["analysis_status"]) != expected_status:
        raise ValueError(
            "PathSpecificPlaceStability status disagrees with its artifact."
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
    expected_metadata = {
        "animal_name": animal_name,
        "date": date,
        "region": region,
        "epoch": epoch,
        "trajectory_type": trajectory_type,
    }
    if not table.empty:
        for field_name, expected_value in expected_metadata.items():
            observed = table[field_name].astype(str).unique().tolist()
            if observed != [str(expected_value)]:
                raise ValueError(
                    "PathSpecificPlaceStability artifact does not match the "
                    f"DPP selection: {field_name}."
                )
    return table


def _load_dpp_encoding_comparison_context(
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
    validated_selection = _dpp_encoding_comparison_selection_row(
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
    if str(validated_selection["dpp_encoding_comparison_id"]) != str(
        key["dpp_encoding_comparison_id"]
    ):
        raise ValueError("DPPEncodingComparison selection UUID is stale.")
    parameters = _validate_dpp_encoding_comparison_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="dpp_encoding_comparison_parameters_sha256",
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
            "DPPEncodingComparison requires valid movement support."
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


def _load_dpp_encoding_comparison_nwb_inputs(
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


def _load_dpp_encoding_comparison_spikes(
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
            "DPPEncodingComparison regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "DPPEncodingComparison regional unit count changed after selection."
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
    if str(validated_selection["path_progression_decoding_comparison_id"]) != str(
        key["path_progression_decoding_comparison_id"]
    ):
        raise ValueError(
            "PathProgressionDecodingComparison selection UUID is stale."
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
                "PathProgressionDecodingComparison movement support does not "
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
            "PathProgressionDecodingComparison regional units changed after "
            "selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "PathProgressionDecodingComparison regional unit count changed "
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


def _load_motor_encoding_comparison_context(
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
    """Load and revalidate one nine-model motor-encoding selection."""
    from v1ca1.spyglass.selection import provenance_sha256

    validated_selection = _motor_encoding_comparison_selection_row(
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
    if str(validated_selection["motor_encoding_comparison_id"]) != str(
        key["motor_encoding_comparison_id"]
    ):
        raise ValueError("MotorEncodingComparison selection UUID is stale.")
    parameters = _validate_motor_encoding_comparison_parameter_row(
        _fetch1_dict(parameters_table, key)
    )
    _validate_frozen_parameters(
        key,
        parameters,
        field_name="motor_encoding_comparison_parameters_sha256",
    )
    expected_model_spec_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_COMPARISON_MODEL_SPEC)
    )
    if str(key.get("motor_encoding_comparison_model_spec_sha256", "")) != (
        expected_model_spec_sha256
    ):
        raise ValueError(
            "MotorEncodingComparison fixed model specification changed after "
            "selection insertion. Create a new selection."
        )
    expected_output_rule_sha256 = provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_COMPARISON_OUTPUT_RULE)
    )
    if str(key.get("motor_encoding_comparison_output_rule_sha256", "")) != (
        expected_output_rule_sha256
    ):
        raise ValueError(
            "MotorEncodingComparison fixed output rule changed after selection "
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


def _load_motor_encoding_comparison_nwb_inputs(
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
                "MotorEncodingComparison primary and orientation-reference "
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


def _load_motor_encoding_comparison_spikes(
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
            "MotorEncodingComparison regional units changed after selection."
        )
    expected_count = int(context["region_row"]["n_units"])
    if int(loaded["n_units"]) != expected_count or len(
        loaded["unit_ids"]
    ) != expected_count:
        raise ValueError(
            "MotorEncodingComparison regional unit count changed after "
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
        "dark_light_manifest_sha256": snapshot["manifest_sha256"],
        "dark_light_selected_sha256_by_model": snapshot[
            "selected_sha256_by_model"
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
        "tuning_curve_artifact_paths": {
            epoch_role: {
                trajectory_type: curve_snapshots[
                    f"{epoch_role}:{trajectory_type}"
                ]["artifact_path"]
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
            "Legacy MotorEncodingComparison registration requires matching "
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
                "legacy MotorEncodingComparison registration."
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


def _validate_dpp_encoding_comparison_artifact_link(
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
    from v1ca1.spyglass.encoding_comparison import (
        ARTIFACT_DIRNAME,
        ARTIFACT_FILENAME,
        summarize_encoding_comparison_table,
    )

    parameters = _validate_dpp_encoding_comparison_parameter_row(
        parameters_row
    )
    _validate_frozen_parameters(
        selection_row,
        parameters,
        field_name="dpp_encoding_comparison_parameters_sha256",
    )
    summary = summarize_encoding_comparison_table(table)
    for field_name in (
        "n_units_eligible",
        "n_units_valid",
        "analysis_status",
        "eligible_units_sha256",
    ):
        if str(result_row[field_name]) != str(summary[field_name]):
            raise ValueError(
                "DPPEncodingComparison result metadata disagrees with its "
                f"artifact: {field_name}."
            )
    if int(result_row["n_units_input"]) != int(region_row["n_units"]):
        raise ValueError(
            "DPPEncodingComparison input-unit count disagrees with its "
            "RegionSortedSpikesGroup."
        )
    if int(result_row["n_units_input"]) < int(summary["n_units_eligible"]):
        raise ValueError(
            "DPPEncodingComparison n_units_input cannot be smaller than the "
            "eligible-unit count."
        )

    comparison_id = str(selection_row["dpp_encoding_comparison_id"])
    if not table.empty:
        artifact_ids = (
            table["dpp_encoding_comparison_id"].astype(str).unique().tolist()
        )
        if artifact_ids != [comparison_id]:
            raise ValueError(
                "DPPEncodingComparison artifact does not match its selection "
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
                    "DPPEncodingComparison artifact does not match its "
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
                    "DPPEncodingComparison artifact does not match its "
                    f"selected parameters: {field_name}."
                )

    artifact_path = Path(result_row["encoding_comparison_path"])
    expected_path_tail = (
        str(animal_name),
        str(date),
        ARTIFACT_DIRNAME,
        str(selection_row["epoch"]),
        str(region_row["region_name"]),
        comparison_id,
        ARTIFACT_FILENAME,
    )
    if tuple(artifact_path.parts[-len(expected_path_tail) :]) != (
        expected_path_tail
    ):
        raise ValueError(
            "DPPEncodingComparison artifact path does not match its session, "
            "epoch, region, and selection UUID."
        )


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
    from v1ca1.spyglass.decoding_comparison import (
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
                "PathProgressionDecodingComparison result metadata "
                f"disagrees with its artifact: {field_name}."
            )
    if int(result_row["n_units_input"]) != int(region_row["n_units"]):
        raise ValueError(
            "PathProgressionDecodingComparison input count disagrees with "
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
            "PathProgressionDecodingComparison input identities disagree "
            "with RegionSortedSpikesGroup."
        )

    expected_metadata = {
        "path_progression_decoding_comparison_id": selection_row[
            "path_progression_decoding_comparison_id"
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
                "PathProgressionDecodingComparison artifact does not match "
                f"its selection: {field_name}."
            )

    manifest_path = Path(result_row["artifact_manifest_path"])
    summary_path = Path(result_row["decoding_summary_path"])
    eligibility_path = Path(result_row["unit_eligibility_path"])
    result_id = str(selection_row["path_progression_decoding_comparison_id"])
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
            "PathProgressionDecodingComparison manifest path does not match "
            "its canonical session/epoch/region/UUID layout."
        )
    if summary_path != manifest_path.parent / METRICS_FILENAME or (
        eligibility_path != manifest_path.parent / ELIGIBILITY_FILENAME
    ):
        raise ValueError(
            "PathProgressionDecodingComparison result paths do not describe "
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
    """Require one place-decoding bundle to match its immutable rows."""
    from v1ca1.spyglass.path_specific_decoding import (
        ARTIFACT_DIRNAME,
        BINNED_ERROR_FILENAME,
        FOLD_QC_FILENAME,
        MANIFEST_FILENAME,
        SELECTED_UNITS_FILENAME,
        SUMMARY_FILENAME,
    )
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

    manifest_path = Path(result_row["artifact_manifest_path"])
    result_id = str(selection_row["path_specific_place_decoding_id"])
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
            "PathSpecificPlaceDecoding manifest path does not match its "
            "canonical session/epoch/region/UUID layout."
        )
    expected_paths = {
        "selected_units_path": SELECTED_UNITS_FILENAME,
        "fold_qc_path": FOLD_QC_FILENAME,
        "decoding_summary_path": SUMMARY_FILENAME,
        "decoding_error_by_position_path": BINNED_ERROR_FILENAME,
    }
    for field_name, filename in expected_paths.items():
        if Path(result_row[field_name]) != manifest_path.parent / filename:
            raise ValueError(
                "PathSpecificPlaceDecoding result paths do not describe one "
                f"canonical bundle: {field_name}."
            )


def _validate_motor_encoding_comparison_artifact_link(
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
        ARTIFACT_DIRNAME,
        FULL_REFIT_FILENAME,
        MANIFEST_FILENAME,
        NESTED_CV_FILENAME,
        SELECTED_UNITS_FILENAME,
        validate_motor_encoding_result,
    )
    from v1ca1.spyglass.selection import (
        provenance_sha256,
        unit_identity_sha256,
    )

    validated = validate_motor_encoding_result(bundle)
    expected_metadata = {
        "motor_encoding_comparison_id": selection_row[
            "motor_encoding_comparison_id"
        ],
        "animal_name": animal_name,
        "date": date,
        "region": region_row["region_name"],
        "epoch": selection_row["epoch"],
    }
    for field_name, expected_value in expected_metadata.items():
        if str(validated["metadata"].get(field_name)) != str(expected_value):
            raise ValueError(
                "MotorEncodingComparison artifact does not match its "
                f"selection: {field_name}."
            )

    parameters = _validate_motor_encoding_comparison_parameter_row(
        parameters_row
    )
    expected_parameters = {
        "parameter_name": parameters[
            "motor_encoding_comparison_param_name"
        ],
        "parameter_sha256": selection_row[
            "motor_encoding_comparison_parameters_sha256"
        ],
        "model_spec_sha256": selection_row[
            "motor_encoding_comparison_model_spec_sha256"
        ],
        "output_rule_sha256": selection_row[
            "motor_encoding_comparison_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "motor_encoding_comparison_param_name"
        },
    }
    if provenance_sha256(dict(validated["parameters"])) != provenance_sha256(
        expected_parameters
    ):
        raise ValueError(
            "MotorEncodingComparison artifact parameters disagree with its "
            "selection."
        )

    expected_scalars = {
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
    }
    for field_name, expected_value in expected_scalars.items():
        if str(result_row[field_name]) != str(expected_value):
            raise ValueError(
                "MotorEncodingComparison result disagrees with its artifact: "
                f"{field_name}."
            )
    if int(result_row["n_units_input"]) != int(region_row["n_units"]):
        raise ValueError(
            "MotorEncodingComparison input count disagrees with "
            "RegionSortedSpikesGroup."
        )
    selected_unit_digest = unit_identity_sha256(
        validated["selected_units"]
        .loc[:, ["spikesorting_merge_id", "unit_id"]]
        .to_dict("records")
    )
    if selected_unit_digest != str(region_row["selected_units_sha256"]):
        raise ValueError(
            "MotorEncodingComparison input identities disagree with "
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
                    "MotorEncodingComparison artifact position provenance "
                    f"disagrees with its selection: {field_name}."
                )

    manifest_path = Path(result_row["artifact_manifest_path"])
    result_id = str(selection_row["motor_encoding_comparison_id"])
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
            "MotorEncodingComparison manifest path does not match its "
            "canonical session/epoch/region/UUID layout."
        )
    expected_paths = {
        "selected_units_path": SELECTED_UNITS_FILENAME,
        "nested_cv_path": NESTED_CV_FILENAME,
        "full_refit_path": FULL_REFIT_FILENAME,
    }
    for field_name, filename in expected_paths.items():
        if Path(result_row[field_name]) != manifest_path.parent / filename:
            raise ValueError(
                "MotorEncodingComparison result paths do not describe one "
                f"canonical bundle: {field_name}."
            )


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
        ARTIFACT_DIRNAME,
        MANIFEST_FILENAME,
        MODEL_NAMES,
        SCHEMA_VERSION_BY_MODE,
        SELECTED_UNITS_FILENAME,
        SELECTION_SUMMARY_FILENAME,
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
        "schema_version": str(validated["parameters"]["schema_version"]),
        "n_units": int(validated["n_units"]),
        "n_candidates": int(validated["n_candidates"]),
        "n_selected_models": int(validated["n_selected_models"]),
        "analysis_status": str(validated["analysis_status"]),
        "selected_units_sha256": str(validated["selected_units_sha256"]),
        "artifact_origin": str(validated["artifact_origin"]),
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

    manifest_path = Path(result_row["artifact_manifest_path"])
    result_id = str(selection_row["dark_light_glm_id"])
    pair_name = (
        f"{selection_row['light_epoch']}_vs_{selection_row['dark_epoch']}"
    )
    expected_tail = (
        str(animal_name),
        str(date),
        ARTIFACT_DIRNAME,
        pair_name,
        str(region_row["region_name"]),
        result_id,
        MANIFEST_FILENAME,
    )
    if tuple(manifest_path.parts[-len(expected_tail) :]) != expected_tail:
        raise ValueError(
            "DarkLightGLM manifest path does not match its canonical "
            "session/epoch-pair/region/UUID layout."
        )
    expected_paths = {
        "selected_units_path": manifest_path.parent / SELECTED_UNITS_FILENAME,
        "selection_summary_path": (
            manifest_path.parent / SELECTION_SUMMARY_FILENAME
        ),
        **{
            f"{model_name}_model_path": (
                manifest_path.parent / "selected" / f"{model_name}.nc"
            )
            for model_name in MODEL_NAMES
        },
    }
    for field_name, expected_path in expected_paths.items():
        if Path(result_row[field_name]) != expected_path:
            raise ValueError(
                "DarkLightGLM result paths do not describe one canonical "
                f"bundle: {field_name}."
            )


def _validate_swap_glm_upstream_link(
    upstream: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    """Require exact frozen DarkLight provenance for one swap result."""
    expected = {
        "dark_light_glm_id": selection["dark_light_glm_id"],
        "dark_light_manifest_sha256": selection[
            "dark_light_manifest_sha256"
        ],
        "dark_light_selected_sha256_by_model": selection[
            "dark_light_selected_sha256_by_model"
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
        ARTIFACT_DIRNAME,
        BUNDLE_SCHEMA_VERSION,
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        RESULT_SCHEMA_VERSION,
        SELECTED_UNITS_FILENAME,
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
            "dark_light_manifest_sha256",
            "dark_light_selected_sha256_by_model",
            "dark_light_parameter_sha256",
            "dark_light_output_rule_sha256",
            "upstream_analysis_status",
        )
    }
    expected_scalars = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_units": validated["n_units"],
        "n_valid_units": validated["n_valid_units"],
        "analysis_status": validated["analysis_status"],
        "selected_units_sha256": validated["selected_units_sha256"],
        "dark_light_manifest_sha256": expected_upstream[
            "dark_light_manifest_sha256"
        ],
        "dark_light_selected_sha256_by_model": expected_upstream[
            "dark_light_selected_sha256_by_model"
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

    manifest_path = Path(result_row["artifact_manifest_path"])
    result_id = str(selection_row["swap_glm_id"])
    light_pair = (
        f"{selection_row['light_train_epoch']}_train_to_"
        f"{selection_row['light_test_epoch']}_test"
    )
    expected_tail = (
        str(animal_name),
        str(date),
        ARTIFACT_DIRNAME,
        light_pair,
        f"dark_{selection_row['dark_epoch']}",
        str(region_row["region_name"]),
        result_id,
        MANIFEST_FILENAME,
    )
    if tuple(manifest_path.parts[-len(expected_tail) :]) != expected_tail:
        raise ValueError(
            "SwapGLM manifest path does not match its canonical "
            "session/epoch-pair/region/UUID layout."
        )
    expected_paths = {
        "selected_units_path": manifest_path.parent / SELECTED_UNITS_FILENAME,
        "swap_glm_path": manifest_path.parent / RESULT_FILENAME,
    }
    for field_name, expected_path in expected_paths.items():
        if Path(result_row[field_name]) != expected_path:
            raise ValueError(
                "SwapGLM result paths do not describe one canonical bundle: "
                f"{field_name}."
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
        ARTIFACT_DIRNAME,
        BUNDLE_SCHEMA_VERSION,
        MANIFEST_FILENAME,
        RESULT_FILENAME,
        RESULT_SCHEMA_VERSION,
        SELECTED_UNITS_FILENAME,
        SUMMARY_FILENAME,
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
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
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

    manifest_path = Path(result_row["artifact_manifest_path"])
    result_id = str(selection_row["swap_tuning_curve_comparison_id"])
    light_pair = (
        f"{selection_row['light_train_epoch']}_train_to_"
        f"{selection_row['light_test_epoch']}_test"
    )
    expected_tail = (
        str(animal_name),
        str(date),
        ARTIFACT_DIRNAME,
        light_pair,
        f"dark_{selection_row['dark_epoch']}",
        str(region_row["region_name"]),
        result_id,
        MANIFEST_FILENAME,
    )
    if tuple(manifest_path.parts[-len(expected_tail) :]) != expected_tail:
        raise ValueError(
            "SwapTuningCurveComparison manifest path does not match its "
            "canonical session/epoch-triplet/region/UUID layout."
        )
    expected_paths = {
        "selected_units_path": manifest_path.parent / SELECTED_UNITS_FILENAME,
        "summary_path": manifest_path.parent / SUMMARY_FILENAME,
        "swap_tuning_curve_comparison_path": (
            manifest_path.parent / RESULT_FILENAME
        ),
    }
    for field_name, expected_path in expected_paths.items():
        if Path(result_row[field_name]) != expected_path:
            raise ValueError(
                "SwapTuningCurveComparison result paths do not describe one "
                f"canonical bundle: {field_name}."
            )


def _make_dpp_encoding_comparison_row(
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
) -> dict[str, Any]:
    """Compute and write one strict four-model encoding comparison."""
    from v1ca1.spyglass.encoding_comparison import (
        compute_selected_dpp_encoding_comparison,
        get_encoding_comparison_artifact_path,
        write_encoding_comparison_artifact,
    )

    context = _load_dpp_encoding_comparison_context(
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
    loaded_spikes = _load_dpp_encoding_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_dpp_encoding_comparison_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    parameters = context["parameters"]
    result = compute_selected_dpp_encoding_comparison(
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
        dpp_encoding_comparison_id=key["dpp_encoding_comparison_id"],
        n_folds=parameters["n_folds"],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        random_seed=parameters["random_seed"],
    )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    artifact_path = get_encoding_comparison_artifact_path(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        region=context["region"],
        dpp_encoding_comparison_id=key["dpp_encoding_comparison_id"],
        **path_kwargs,
    )
    created_artifact_paths = (
        [] if artifact_path.exists() else [str(artifact_path)]
    )
    written_path = write_encoding_comparison_artifact(
        result["table"],
        artifact_path,
    )
    return {
        "encoding_comparison_path": str(written_path),
        "n_units_input": int(context["region_row"]["n_units"]),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_units_valid": int(result["n_units_valid"]),
        "analysis_status": str(result["analysis_status"]),
        "eligible_units_sha256": str(result["eligible_units_sha256"]),
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


def _register_existing_dpp_encoding_comparison_row(
    *,
    key: Mapping[str, Any],
    encoding_comparison_path: Path,
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
) -> dict[str, Any]:
    """Normalize and register one exact-coverage legacy comparison."""
    from v1ca1.spyglass.encoding_comparison import (
        get_encoding_comparison_artifact_path,
        register_existing_encoding_comparison_artifact,
    )

    if overwrite:
        raise ValueError(
            "Registered DPPEncodingComparison artifacts are immutable."
        )
    context = _load_dpp_encoding_comparison_context(
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
    loaded_spikes = _load_dpp_encoding_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    resolver = _legacy_dpp_unit_identity_resolver(loaded_spikes)
    parameters = context["parameters"]
    source_path = _validate_legacy_dpp_encoding_source_path(
        Path(encoding_comparison_path),
        region=context["region"],
        epoch=context["epoch"],
        parameters=parameters,
    )
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    destination = get_encoding_comparison_artifact_path(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        region=context["region"],
        dpp_encoding_comparison_id=key["dpp_encoding_comparison_id"],
        **path_kwargs,
    )
    registered = register_existing_encoding_comparison_artifact(
        source_path=source_path,
        destination_path=destination,
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
        dpp_encoding_comparison_id=key["dpp_encoding_comparison_id"],
        n_folds=parameters["n_folds"],
        evaluation_bin_size_s=parameters["evaluation_bin_size_s"],
        spatial_bin_size_cm=parameters["spatial_bin_size_cm"],
        gaussian_smoothing_sigma_bins=parameters[
            "gaussian_smoothing_sigma_bins"
        ],
        random_seed=parameters["random_seed"],
        source_v1ca1_git_commit=source_v1ca1_git_commit,
    )
    provenance = dict(registered["legacy_artifact_provenance"])
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
    return {
        "encoding_comparison_path": str(registered["path"]),
        "n_units_input": int(context["region_row"]["n_units"]),
        "n_units_eligible": int(registered["n_units_eligible"]),
        "n_units_valid": int(registered["n_units_valid"]),
        "analysis_status": str(registered["analysis_status"]),
        "eligible_units_sha256": str(
            registered["eligible_units_sha256"]
        ),
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": list(
            registered.get("_created_artifact_paths", ())
        ),
    }


def _make_path_progression_decoding_comparison_row(
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
) -> dict[str, Any]:
    """Compute and persist one shared-cohort cross-path decoding bundle."""
    from v1ca1.spyglass.decoding_comparison import (
        compute_path_progression_decoding_comparison,
        get_decoding_artifact_paths,
        write_decoding_artifact_bundle,
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
    result = compute_path_progression_decoding_comparison(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        cohort_epoch=context["cohort_epoch"],
        path_progression_decoding_comparison_id=key[
            "path_progression_decoding_comparison_id"
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_decoding_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        cohort_epoch=context["cohort_epoch"],
        region=context["region"],
        path_progression_decoding_comparison_id=key[
            "path_progression_decoding_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_decoding_artifact_bundle(
        result,
        paths,
        overwrite=False,
    )
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "decoding_summary_path": str(written["decoding_summary_path"]),
        "unit_eligibility_path": str(written["unit_eligibility_path"]),
        "n_units_input": int(result["n_units_input"]),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_transfer_pairs_expected": int(
            result["n_transfer_pairs_expected"]
        ),
        "n_transfer_pairs_valid": int(result["n_transfer_pairs_valid"]),
        "n_decoded_samples": int(result["n_decoded_samples"]),
        "analysis_status": str(result["analysis_status"]),
        "eligible_units_sha256": str(result["eligible_units_sha256"]),
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Compute and persist one within-epoch physical-place decoder."""
    from v1ca1.spyglass.path_specific_decoding import (
        compute_path_specific_place_decoding,
        get_path_specific_decoding_artifact_paths,
        write_path_specific_decoding_artifact,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_path_specific_decoding_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        region=context["region"],
        path_specific_place_decoding_id=key[
            "path_specific_place_decoding_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_path_specific_decoding_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "fold_qc_path": str(written["fold_qc_path"]),
        "decoding_summary_path": str(written["decoding_summary_path"]),
        "decoding_error_by_position_path": str(
            written["binned_error_path"]
        ),
        "n_units": int(result["n_units"]),
        "n_folds_expected": int(result["n_folds_expected"]),
        "n_folds_valid": int(result["n_folds_valid"]),
        "n_decoded_samples": int(result["n_decoded_samples"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Validate, normalize, and copy one legacy place decoder bundle."""
    from v1ca1.spyglass.path_specific_decoding import (
        get_path_specific_decoding_artifact_paths,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_path_specific_decoding_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        region=context["region"],
        path_specific_place_decoding_id=key[
            "path_specific_place_decoding_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    registered = register_existing_path_specific_decoding_artifact(
        source_true_path=Path(source_true_path),
        source_decoded_path=Path(source_decoded_path),
        destination_path=artifact_dir,
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
    written = paths
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "fold_qc_path": str(written["fold_qc_path"]),
        "decoding_summary_path": str(written["decoding_summary_path"]),
        "decoding_error_by_position_path": str(
            written["binned_error_path"]
        ),
        "n_units": int(registered["n_units"]),
        "n_folds_expected": int(registered["n_folds_expected"]),
        "n_folds_valid": int(registered["n_folds_valid"]),
        "n_decoded_samples": int(registered["n_decoded_samples"]),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": str(
            registered["selected_units_sha256"]
        ),
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": created_artifact_paths,
    }


def _make_motor_encoding_comparison_row(
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
) -> dict[str, Any]:
    """Compute and persist one nine-model motor-encoding bundle."""
    from v1ca1.spyglass.motor_encoding import (
        compute_motor_encoding_comparison,
        get_motor_encoding_artifact_paths,
        write_motor_encoding_artifact,
    )

    context = _load_motor_encoding_comparison_context(
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
    loaded_spikes = _load_motor_encoding_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_motor_encoding_comparison_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    parameters = context["parameters"]
    selection = context["selection"]
    result = compute_motor_encoding_comparison(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        motor_encoding_comparison_id=key[
            "motor_encoding_comparison_id"
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
        minimum_movement_firing_rate_hz=parameters[
            "minimum_movement_firing_rate_hz"
        ],
        parameter_name=parameters[
            "motor_encoding_comparison_param_name"
        ],
        parameter_sha256=selection[
            "motor_encoding_comparison_parameters_sha256"
        ],
        model_spec_sha256=selection[
            "motor_encoding_comparison_model_spec_sha256"
        ],
        output_rule_sha256=selection[
            "motor_encoding_comparison_output_rule_sha256"
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_motor_encoding_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        region=context["region"],
        motor_encoding_comparison_id=key[
            "motor_encoding_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_motor_encoding_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "nested_cv_path": str(written["nested_cv_path"]),
        "full_refit_path": str(written["full_refit_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "n_units_input": int(result["n_units_input"]),
        "n_units_eligible": int(result["n_units_eligible"]),
        "n_units_valid": int(result["n_units_valid"]),
        "n_outer_folds_expected": int(result["n_outer_folds_expected"]),
        "n_outer_folds_valid": int(result["n_outer_folds_valid"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


def _register_existing_motor_encoding_comparison_row(
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
    session_table: Any,
    nwbfile_table: Any,
    source_v1ca1_git_commit: str | None,
    source_spyglass_git_commit: str | None,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Validate, normalize, and copy one paired legacy motor fit."""
    from v1ca1.spyglass.motor_encoding import (
        get_motor_encoding_artifact_paths,
        register_existing_motor_encoding_artifact,
    )

    context = _load_motor_encoding_comparison_context(
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
    loaded_spikes = _load_motor_encoding_comparison_spikes(
        context=context,
        region_sorted_spikes_group_table=region_sorted_spikes_group_table,
    )
    nwb_inputs = _load_motor_encoding_comparison_nwb_inputs(
        context=context,
        position_table=position_table,
        trajectory_intervals_table=trajectory_intervals_table,
        wtrack_graph_table=wtrack_graph_table,
        nwbfile_table=nwbfile_table,
    )
    resolver = _legacy_motor_unit_identity_resolver(loaded_spikes)
    parameters = context["parameters"]
    selection = context["selection"]
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_motor_encoding_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        epoch=context["epoch"],
        region=context["region"],
        motor_encoding_comparison_id=key[
            "motor_encoding_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    registered = register_existing_motor_encoding_artifact(
        source_nested_cv_path=Path(source_nested_cv_path),
        source_full_refit_path=Path(source_full_refit_path),
        destination_path=artifact_dir,
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        epoch=context["epoch"],
        motor_encoding_comparison_id=key[
            "motor_encoding_comparison_id"
        ],
        movement_firing_rate_table=context["movement"]["table"],
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
        parameter_name=parameters[
            "motor_encoding_comparison_param_name"
        ],
        parameter_sha256=selection[
            "motor_encoding_comparison_parameters_sha256"
        ],
        model_spec_sha256=selection[
            "motor_encoding_comparison_model_spec_sha256"
        ],
        output_rule_sha256=selection[
            "motor_encoding_comparison_output_rule_sha256"
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
    written = paths
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "nested_cv_path": str(written["nested_cv_path"]),
        "full_refit_path": str(written["full_refit_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "n_units_input": int(registered["n_units_input"]),
        "n_units_eligible": int(registered["n_units_eligible"]),
        "n_units_valid": int(registered["n_units_valid"]),
        "n_outer_folds_expected": int(
            registered["n_outer_folds_expected"]
        ),
        "n_outer_folds_valid": int(
            registered["n_outer_folds_valid"]
        ),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": str(
            registered["selected_units_sha256"]
        ),
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Compute and persist one coupled dark/light four-model GLM bundle."""
    from v1ca1.spyglass.dark_light_glm import (
        compute_dark_light_glm,
        get_dark_light_glm_artifact_paths,
        write_dark_light_glm_artifact,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_dark_light_glm_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        light_epoch=epoch_by_condition["light"],
        dark_epoch=epoch_by_condition["dark"],
        dark_light_glm_id=selection["dark_light_glm_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_dark_light_glm_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "selection_summary_path": str(written["selection_summary_path"]),
        **{
            f"{model_name}_model_path": str(
                written["selected_model_paths"][model_name]
            )
            for model_name in (
                "visual",
                "task_segment_bump",
                "task_segment_scalar",
                "task_dense_gain",
            )
        },
        "schema_version": str(result["parameters"]["schema_version"]),
        "n_units": int(result["n_units"]),
        "n_candidates": int(result["n_candidates"]),
        "n_selected_models": int(result["n_selected_models"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Validate and copy one exact imported-sorting dark/light artifact set."""
    from v1ca1.spyglass.dark_light_glm import (
        MODEL_NAMES,
        SCHEMA_VERSION_BY_MODE,
        get_dark_light_glm_artifact_paths,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_dark_light_glm_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        light_epoch=selection["light_epoch"],
        dark_epoch=selection["dark_epoch"],
        dark_light_glm_id=selection["dark_light_glm_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    try:
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
            destination_path=artifact_dir,
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
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    provenance = dict(registered["legacy_artifact_provenance"] or {})
    provenance["source_spyglass_git_commit"] = source_spyglass_git_commit
    written = paths
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "selection_summary_path": str(written["selection_summary_path"]),
        **{
            f"{model_name}_model_path": str(
                written["selected_model_paths"][model_name]
            )
            for model_name in MODEL_NAMES
        },
        "schema_version": str(registered["parameters"]["schema_version"]),
        "n_units": int(registered["n_units"]),
        "n_candidates": int(registered["n_candidates"]),
        "n_selected_models": int(registered["n_selected_models"]),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": str(registered["selected_units_sha256"]),
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Compute and persist one held-out swapped-light score bundle."""
    from v1ca1.spyglass.swap_glm import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        compute_swap_glm,
        get_swap_glm_artifact_paths,
        write_swap_glm_artifact,
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
        dark_light_glm_artifact_path=context["dark_light_snapshot"][
            "artifact_dir"
        ],
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_swap_glm_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        swap_glm_id=selection["swap_glm_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_swap_glm_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    upstream = result["upstream_provenance"]
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "swap_glm_path": str(written["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "dark_light_manifest_sha256": str(
            upstream["dark_light_manifest_sha256"]
        ),
        "dark_light_selected_sha256_by_model": dict(
            upstream["dark_light_selected_sha256_by_model"]
        ),
        "dark_light_parameter_sha256": str(
            upstream["dark_light_parameter_sha256"]
        ),
        "dark_light_output_rule_sha256": str(
            upstream["dark_light_output_rule_sha256"]
        ),
        "upstream_analysis_status": str(
            upstream["upstream_analysis_status"]
        ),
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Strictly validate and copy one imported-sorting swap artifact."""
    from v1ca1.spyglass.swap_glm import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        get_swap_glm_artifact_paths,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_swap_glm_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        swap_glm_id=selection["swap_glm_id"],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    try:
        registered = register_existing_swap_glm_artifact(
            source_result_path=Path(source_result_path),
            destination_path=artifact_dir,
            swap_glm_id=selection["swap_glm_id"],
            animal_name=context["animal_name"],
            date=context["date"],
            region=context["region"],
            dark_epoch=selection["dark_epoch"],
            light_train_epoch=selection["light_train_epoch"],
            light_test_epoch=selection["light_test_epoch"],
            dark_light_glm_artifact_path=context["dark_light_snapshot"][
                "artifact_dir"
            ],
            spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            movement_interval=context["movement"]["movement_intervals"],
            movement_analysis_status=context["movement"]["analysis_status"],
            trajectory_intervals=nwb_inputs["trajectory_intervals"],
            graph_inputs_by_trajectory=nwb_inputs["graph_inputs"],
            position=nwb_inputs["position"],
            position_offset_samples=int(
                nwb_inputs["position_row"][
                    "analysis_start_offset_samples"
                ]
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
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    provenance = dict(registered["legacy_artifact_provenance"] or {})
    provenance["source_spyglass_git_commit"] = source_spyglass_git_commit
    upstream = registered["upstream_provenance"]
    return {
        "artifact_manifest_path": str(paths["artifact_manifest_path"]),
        "selected_units_path": str(paths["selected_units_path"]),
        "swap_glm_path": str(paths["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_units": int(registered["n_units"]),
        "n_valid_units": int(registered["n_valid_units"]),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": str(
            registered["selected_units_sha256"]
        ),
        "dark_light_manifest_sha256": str(
            upstream["dark_light_manifest_sha256"]
        ),
        "dark_light_selected_sha256_by_model": dict(
            upstream["dark_light_selected_sha256_by_model"]
        ),
        "dark_light_parameter_sha256": str(
            upstream["dark_light_parameter_sha256"]
        ),
        "dark_light_output_rule_sha256": str(
            upstream["dark_light_output_rule_sha256"]
        ),
        "upstream_analysis_status": str(
            upstream["upstream_analysis_status"]
        ),
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Compute and persist one empirical held-out swap-tuning bundle."""
    from v1ca1.spyglass.swap_tuning import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        compute_swap_tuning_curve_comparison,
        get_swap_tuning_curve_comparison_artifact_paths,
        write_swap_tuning_curve_comparison_artifact,
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
        tuning_curve_artifact_paths=context[
            "tuning_curve_artifact_paths"
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_swap_tuning_curve_comparison_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        swap_tuning_curve_comparison_id=selection[
            "swap_tuning_curve_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    written = write_swap_tuning_curve_comparison_artifact(
        result,
        artifact_dir,
        overwrite=False,
    )
    return {
        "artifact_manifest_path": str(written["artifact_manifest_path"]),
        "selected_units_path": str(written["selected_units_path"]),
        "summary_path": str(written["summary_path"]),
        "swap_tuning_curve_comparison_path": str(written["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_source_units": int(result["n_source_units"]),
        "n_units": int(result["n_units"]),
        "n_valid_units": int(result["n_valid_units"]),
        "analysis_status": str(result["analysis_status"]),
        "selected_units_sha256": str(result["selected_units_sha256"]),
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": created_artifact_paths,
    }


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
) -> dict[str, Any]:
    """Rebuild and verify one legacy imported-sorting empirical swap."""
    from v1ca1.spyglass.swap_tuning import (
        BUNDLE_SCHEMA_VERSION,
        RESULT_SCHEMA_VERSION,
        get_swap_tuning_curve_comparison_artifact_paths,
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
    path_kwargs: dict[str, Any] = {}
    if artifact_root is not None:
        path_kwargs["artifact_root"] = artifact_root
    paths = get_swap_tuning_curve_comparison_artifact_paths(
        animal_name=context["animal_name"],
        date=context["date"],
        region=context["region"],
        dark_epoch=selection["dark_epoch"],
        light_train_epoch=selection["light_train_epoch"],
        light_test_epoch=selection["light_test_epoch"],
        swap_tuning_curve_comparison_id=selection[
            "swap_tuning_curve_comparison_id"
        ],
        **path_kwargs,
    )
    artifact_dir = Path(paths["artifact_dir"])
    created_artifact_paths = [] if artifact_dir.exists() else [str(artifact_dir)]
    light_test_movement = context["movement"]["light_test"]
    try:
        registered = register_existing_swap_tuning_curve_comparison_artifact(
            source_result_path=Path(source_result_path),
            source_summary_path=Path(source_summary_path),
            destination_path=artifact_dir,
            swap_tuning_curve_comparison_id=selection[
                "swap_tuning_curve_comparison_id"
            ],
            animal_name=context["animal_name"],
            date=context["date"],
            region=context["region"],
            dark_epoch=selection["dark_epoch"],
            light_train_epoch=selection["light_train_epoch"],
            light_test_epoch=selection["light_test_epoch"],
            tuning_curve_artifact_paths=context[
                "tuning_curve_artifact_paths"
            ],
            movement_firing_rate_tables_by_role=context[
                "movement_firing_rate_tables"
            ],
            spikes=loaded_spikes["ts_group"],
            stable_unit_ids=loaded_spikes["unit_ids"],
            position=nwb_inputs["position"],
            position_offset_samples=int(
                nwb_inputs["position_row"][
                    "analysis_start_offset_samples"
                ]
            ),
            movement_interval=light_test_movement["movement_intervals"],
            movement_analysis_status=light_test_movement[
                "analysis_status"
            ],
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
                source_spyglass_git_commit=(
                    source_spyglass_git_commit
                ),
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
    except Exception:
        _remove_created_artifacts(created_artifact_paths)
        raise
    provenance = dict(registered["legacy_artifact_provenance"] or {})
    return {
        "artifact_manifest_path": str(paths["artifact_manifest_path"]),
        "selected_units_path": str(paths["selected_units_path"]),
        "summary_path": str(paths["summary_path"]),
        "swap_tuning_curve_comparison_path": str(paths["result_path"]),
        "schema_version": RESULT_SCHEMA_VERSION,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "n_source_units": int(registered["n_source_units"]),
        "n_units": int(registered["n_units"]),
        "n_valid_units": int(registered["n_valid_units"]),
        "analysis_status": str(registered["analysis_status"]),
        "selected_units_sha256": str(
            registered["selected_units_sha256"]
        ),
        "legacy_artifact_provenance": provenance,
        "_created_artifact_paths": created_artifact_paths,
    }


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
    dpp_encoding_comparison_compute_hook = runtime_hooks.get(
        "dpp_encoding_comparison_compute",
        _make_dpp_encoding_comparison_row,
    )
    dpp_encoding_comparison_register_hook = runtime_hooks.get(
        "dpp_encoding_comparison_register_existing",
        _register_existing_dpp_encoding_comparison_row,
    )
    path_progression_decoding_compute_hook = runtime_hooks.get(
        "path_progression_decoding_comparison_compute",
        _make_path_progression_decoding_comparison_row,
    )
    path_specific_place_decoding_compute_hook = runtime_hooks.get(
        "path_specific_place_decoding_compute",
        _make_path_specific_place_decoding_row,
    )
    path_specific_place_decoding_register_hook = runtime_hooks.get(
        "path_specific_place_decoding_register_existing",
        _register_existing_path_specific_place_decoding_row,
    )
    motor_encoding_comparison_compute_hook = runtime_hooks.get(
        "motor_encoding_comparison_compute",
        _make_motor_encoding_comparison_row,
    )
    motor_encoding_comparison_register_hook = runtime_hooks.get(
        "motor_encoding_comparison_register_existing",
        _register_existing_motor_encoding_comparison_row,
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
    cross_region_xcorr_compute_hook = runtime_hooks.get(
        "cross_region_xcorr_compute",
        _make_cross_region_xcorr_row,
    )
    cross_region_xcorr_register_hook = runtime_hooks.get(
        "cross_region_xcorr_register_existing",
        _register_existing_cross_region_xcorr_row,
    )
    ripple_decoding_comparison_compute_hook = runtime_hooks.get(
        "ripple_decoding_comparison_compute",
        _make_ripple_decoding_comparison_row,
    )
    ripple_decoding_comparison_register_hook = runtime_hooks.get(
        "ripple_decoding_comparison_register_existing",
        _register_existing_ripple_decoding_comparison_row,
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
            dpp_encoding_comparison_compute_hook,
            dpp_encoding_comparison_register_hook,
            path_progression_decoding_compute_hook,
            path_specific_place_decoding_compute_hook,
            path_specific_place_decoding_register_hook,
            motor_encoding_comparison_compute_hook,
            motor_encoding_comparison_register_hook,
            dark_light_glm_compute_hook,
            dark_light_glm_register_hook,
            swap_glm_compute_hook,
            swap_glm_register_hook,
            swap_tuning_curve_comparison_compute_hook,
            swap_tuning_curve_comparison_register_hook,
            ripple_glm_compute_hook,
            ripple_glm_register_hook,
            cross_region_xcorr_compute_hook,
            cross_region_xcorr_register_hook,
            ripple_decoding_comparison_compute_hook,
            ripple_decoding_comparison_register_hook,
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

    class Ripples(spyglass_mixin, dj_module.Manual):
        definition = table_specs.RIPPLES_DEFINITION

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

    Ripples = main_schema(Ripples)
    main_context["Ripples"] = Ripples

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
            """Compute, write, and insert one epoch motor-behavior bundle."""
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
            """Load and verify one canonical epoch motor-behavior bundle."""
            from v1ca1.spyglass.epoch_motor_behavior import (
                load_epoch_motor_behavior_artifact,
            )

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
            bundle = load_epoch_motor_behavior_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_epoch_motor_behavior_artifact_link(
                bundle=bundle,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                movement_parameters_row=movement_parameters,
                animal_name=animal_name,
                date=session_date,
            )
            return bundle

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
                    source_progression_path=Path(source_progression_path),
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
                sorted_spikes_group=sorted_spikes_group,
                unit_selection_params=unit_selection_params,
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
            selection = _fetch1_dict(MovementFiringRateSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=MovementParameters,
                    epoch_intervals_table=EpochIntervals,
                    position_table=Position,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
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
            """Load and validate one all-unit movement-rate Parquet."""
            from v1ca1.spyglass.movement import (
                load_movement_firing_rate_artifact,
            )

            row = _fetch1_dict(cls, key)
            return load_movement_firing_rate_artifact(
                Path(row["movement_firing_rate_path"])
            )

        @classmethod
        def load_intervals(cls, key: Mapping[str, Any]) -> Any:
            """Load and validate one exact Pynapple movement IntervalSet."""
            from v1ca1.spyglass.movement import load_movement_interval_artifact

            row = _fetch1_dict(cls, key)
            return load_movement_interval_artifact(
                Path(row["movement_intervals_path"])
            )

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
            """Compute, write, and insert one immutable cvPCA bundle."""
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
            """Load and verify one canonical cvPCA artifact bundle."""
            from v1ca1.spyglass.cv_pca import load_cv_pca_artifact

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
            bundle = load_cv_pca_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_cv_pca_artifact_link(
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
                ripples_table=Ripples,
                epoch_intervals_table=EpochIntervals,
                parameters_table=RippleModulationParameters,
                sorted_spikes_group=sorted_spikes_group,
                unit_selection_params=unit_selection_params,
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
            """Compute, write, and register one selected artifact pair."""
            selection = _fetch1_dict(RippleModulationSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleModulationParameters,
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
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
            """Filter keyed legacy Parquets, write them, then insert one row."""
            if overwrite:
                raise ValueError(
                    "Registered RippleModulation results are immutable; create "
                    "a new selection instead of overwriting an artifact."
                )
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
                    ripples_table=Ripples,
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
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
            """Compute, write, and insert one selected tuning curve."""
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
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
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
            from v1ca1.spyglass.path_specific_place import (
                load_path_specific_place_artifact,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                PathSpecificPlaceTuningCurveSelection,
                {
                    "path_specific_place_tuning_curve_id": row[
                        "path_specific_place_tuning_curve_id"
                    ]
                },
            )
            curve = load_path_specific_place_artifact(
                Path(row["tuning_curve_path"])
            )
            _validate_tuning_curve_artifact_link(
                curve=curve,
                result_row=row,
                selection_row=selection,
            )
            return curve

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
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
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
                    session_table=session_table,
                    artifact_root=artifact_root,
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
            """Load and cross-check one canonical all-unit similarity Parquet."""
            from v1ca1.spyglass.tuning_similarity import (
                load_tuning_similarity_artifact,
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
                session_table=session_table,
            )
            table = load_tuning_similarity_artifact(
                Path(row["similarity_path"])
            )
            validate_tuning_similarity_against_inputs(
                table,
                tuning_curves_by_trajectory=inputs["curves"],
                movement_firing_rate_table=inputs["movement_table"],
                similarity_metric=inputs["parameters"]["similarity_metric"],
            )
            _validate_tuning_similarity_artifact_link(
                table=table,
                result_row=row,
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
                    "PathSpecificPlaceTuningSimilarity already contains this "
                    "immutable selection."
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
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
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
            """Compute, write, and insert one selected DPP tuning curve."""
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
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
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
            from v1ca1.spyglass.dpp import load_dpp_artifact

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                DPPTuningCurveSelection,
                {"dpp_tuning_curve_id": row["dpp_tuning_curve_id"]},
            )
            curve = load_dpp_artifact(Path(row["tuning_curve_path"]))
            _validate_dpp_tuning_curve_artifact_link(
                curve=curve,
                result_row=row,
                selection_row=selection,
            )
            return curve

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
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    nwbfile_table=nwbfile_table,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "dpp_tuning_curve_id": selection["dpp_tuning_curve_id"],
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
                    session_table=session_table,
                    artifact_root=artifact_root,
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
            selection = _fetch1_dict(PathSpecificPlaceStabilitySelection, key)
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
                    session_table=session_table,
                    sorted_spikes_group=sorted_spikes_group,
                    unit_selection_params=unit_selection_params,
                    spike_sorting_output=spike_sorting_output,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
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

    class DPPEncodingComparisonParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DPP_ENCODING_COMPARISON_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one encoding-comparison parameter row."""
            validated = _validate_dpp_encoding_comparison_parameter_row(row)
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
                for preset in table_specs.DPP_ENCODING_COMPARISON_PARAMETER_PRESETS
            ]

        @classmethod
        def insert_default(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> dict[str, Any]:
            """Explicitly insert the manuscript 50-ms preset."""
            return cls.insert_parameters(
                table_specs.MANUSCRIPT_DPP_ENCODING_COMPARISON_PARAMETERS,
                skip_duplicates=skip_duplicates,
            )

    DPPEncodingComparisonParameters = main_schema(
        DPPEncodingComparisonParameters
    )
    main_context["DPPEncodingComparisonParameters"] = (
        DPPEncodingComparisonParameters
    )

    class DPPEncodingComparisonSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.DPP_ENCODING_COMPARISON_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one comparison."""
            row = _dpp_encoding_comparison_selection_row(
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
                parameters_table=DPPEncodingComparisonParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    DPPEncodingComparisonSelection = main_schema(
        DPPEncodingComparisonSelection
    )
    main_context["DPPEncodingComparisonSelection"] = (
        DPPEncodingComparisonSelection
    )

    class DPPEncodingComparison(spyglass_mixin, dj_module.Computed):
        definition = table_specs.DPP_ENCODING_COMPARISON_DEFINITION
        _compute_hook = staticmethod(dpp_encoding_comparison_compute_hook)
        _register_existing_hook = staticmethod(
            dpp_encoding_comparison_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one four-model comparison."""
            selection = _fetch1_dict(DPPEncodingComparisonSelection, key)
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=DPPEncodingComparisonParameters,
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
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "dpp_encoding_comparison_id": selection[
                            "dpp_encoding_comparison_id"
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
        def load_encoding_comparison(
            cls,
            key: Mapping[str, Any],
        ) -> Any:
            """Load and validate one canonical comparison Parquet."""
            from v1ca1.spyglass.encoding_comparison import (
                load_encoding_comparison_artifact,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                DPPEncodingComparisonSelection,
                {
                    "dpp_encoding_comparison_id": row[
                        "dpp_encoding_comparison_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                DPPEncodingComparisonParameters,
                {
                    "dpp_encoding_comparison_param_name": selection[
                        "dpp_encoding_comparison_param_name"
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
            table = load_encoding_comparison_artifact(
                Path(row["encoding_comparison_path"])
            )
            _validate_dpp_encoding_comparison_artifact_link(
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
            encoding_comparison_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Normalize one exact-coverage legacy artifact and insert it."""
            if overwrite:
                raise ValueError(
                    "Registered DPPEncodingComparison results are immutable; "
                    "create a new selection instead of overwriting an artifact."
                )
            selection = _fetch1_dict(DPPEncodingComparisonSelection, key)
            result_key = {
                "dpp_encoding_comparison_id": selection[
                    "dpp_encoding_comparison_id"
                ]
            }
            existing = _existing_result_row(cls, result_key)
            if existing is not None:
                if skip_duplicates:
                    return existing
                raise ValueError(
                    "DPPEncodingComparison already contains this immutable "
                    "selection."
                )
            artifact_row = dict(
                cls._register_existing_hook(
                    key=selection,
                    encoding_comparison_path=Path(
                        encoding_comparison_path
                    ),
                    overwrite=False,
                    parameters_table=DPPEncodingComparisonParameters,
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
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "dpp_encoding_comparison_id": selection[
                    "dpp_encoding_comparison_id"
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

    DPPEncodingComparison = main_schema(DPPEncodingComparison)
    main_context["DPPEncodingComparison"] = DPPEncodingComparison

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

    class PathProgressionDecodingComparisonSelection(
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

    PathProgressionDecodingComparisonSelection = main_schema(
        PathProgressionDecodingComparisonSelection
    )
    main_context["PathProgressionDecodingComparisonSelection"] = (
        PathProgressionDecodingComparisonSelection
    )

    class PathProgressionDecodingComparison(
        spyglass_mixin,
        dj_module.Computed,
    ):
        definition = table_specs.PATH_PROGRESSION_DECODING_DEFINITION
        _compute_hook = staticmethod(path_progression_decoding_compute_hook)

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one shared-cohort decoder bundle."""
            selection = _fetch1_dict(
                PathProgressionDecodingComparisonSelection,
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
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "path_progression_decoding_comparison_id": selection[
                            "path_progression_decoding_comparison_id"
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
            """Load and validate one canonical cross-path decoding bundle."""
            from v1ca1.spyglass.decoding_comparison import (
                load_decoding_artifact_bundle,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                PathProgressionDecodingComparisonSelection,
                {
                    "path_progression_decoding_comparison_id": row[
                        "path_progression_decoding_comparison_id"
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
            bundle = load_decoding_artifact_bundle(
                Path(row["artifact_manifest_path"])
            )
            _validate_path_progression_decoding_artifact_link(
                bundle=bundle,
                result_row=row,
                selection_row=selection,
                parameters_row=parameters,
                region_row=region_row,
                animal_name=animal_name,
                date=session_date,
            )
            return bundle

    PathProgressionDecodingComparison = main_schema(
        PathProgressionDecodingComparison
    )
    main_context["PathProgressionDecodingComparison"] = (
        PathProgressionDecodingComparison
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
            """Compute, write, and insert one within-epoch decoder bundle."""
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
            """Load and validate one canonical place-decoding bundle."""
            from v1ca1.spyglass.path_specific_decoding import (
                load_path_specific_decoding_artifact,
            )

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
            bundle = load_path_specific_decoding_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_path_specific_place_decoding_artifact_link(
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
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "path_specific_place_decoding_id": selection[
                    "path_specific_place_decoding_id"
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

    PathSpecificPlaceDecoding = main_schema(PathSpecificPlaceDecoding)
    main_context["PathSpecificPlaceDecoding"] = PathSpecificPlaceDecoding

    class MotorEncodingComparisonParameters(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.MOTOR_ENCODING_COMPARISON_PARAMETERS_DEFINITION
        )

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one motor-comparison parameter row."""
            validated = _validate_motor_encoding_comparison_parameter_row(
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
                    table_specs.MOTOR_ENCODING_COMPARISON_PARAMETER_PRESETS
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
                    table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_COMPARISON_PARAMETERS
                ),
                "ca1": (
                    table_specs.MANUSCRIPT_CA1_MOTOR_ENCODING_COMPARISON_PARAMETERS
                ),
            }[canonical_region]
            return cls.insert_parameters(
                preset,
                skip_duplicates=skip_duplicates,
            )

    MotorEncodingComparisonParameters = main_schema(
        MotorEncodingComparisonParameters
    )
    main_context["MotorEncodingComparisonParameters"] = (
        MotorEncodingComparisonParameters
    )

    class MotorEncodingComparisonSelection(
        spyglass_mixin,
        dj_module.Manual,
    ):
        definition = (
            table_specs.MOTOR_ENCODING_COMPARISON_SELECTION_DEFINITION
        )

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one motor comparison."""
            row = _motor_encoding_comparison_selection_row(
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
                parameters_table=MotorEncodingComparisonParameters,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    MotorEncodingComparisonSelection = main_schema(
        MotorEncodingComparisonSelection
    )
    main_context["MotorEncodingComparisonSelection"] = (
        MotorEncodingComparisonSelection
    )

    class MotorEncodingComparison(
        spyglass_mixin,
        dj_module.Computed,
    ):
        definition = table_specs.MOTOR_ENCODING_COMPARISON_DEFINITION
        _compute_hook = staticmethod(motor_encoding_comparison_compute_hook)
        _register_existing_hook = staticmethod(
            motor_encoding_comparison_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one nine-model comparison bundle."""
            selection = _fetch1_dict(
                MotorEncodingComparisonSelection,
                key,
            )
            row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=MotorEncodingComparisonParameters,
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
                )
            )
            created_artifact_paths = list(
                row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "motor_encoding_comparison_id": selection[
                            "motor_encoding_comparison_id"
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
            """Load and validate one canonical motor-encoding bundle."""
            from v1ca1.spyglass.motor_encoding import (
                load_motor_encoding_artifact,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                MotorEncodingComparisonSelection,
                {
                    "motor_encoding_comparison_id": row[
                        "motor_encoding_comparison_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                MotorEncodingComparisonParameters,
                {
                    "motor_encoding_comparison_param_name": selection[
                        "motor_encoding_comparison_param_name"
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
            bundle = load_motor_encoding_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_motor_encoding_comparison_artifact_link(
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
                    "Registered MotorEncodingComparison results are immutable; "
                    "create a new selection instead of overwriting."
                )
            selection = _fetch1_dict(
                MotorEncodingComparisonSelection,
                key,
            )
            result_key = {
                "motor_encoding_comparison_id": selection[
                    "motor_encoding_comparison_id"
                ]
            }
            existing = _existing_result_row(cls, result_key)
            if existing is not None:
                if skip_duplicates:
                    return existing
                raise ValueError(
                    "MotorEncodingComparison already contains this immutable "
                    "selection."
                )
            artifact_row = dict(
                cls._register_existing_hook(
                    key=selection,
                    source_nested_cv_path=Path(source_nested_cv_path),
                    source_full_refit_path=Path(source_full_refit_path),
                    parameters_table=MotorEncodingComparisonParameters,
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
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "motor_encoding_comparison_id": selection[
                    "motor_encoding_comparison_id"
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

    MotorEncodingComparison = main_schema(MotorEncodingComparison)
    main_context["MotorEncodingComparison"] = MotorEncodingComparison

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
            """Compute, write, and insert one dark/light GLM bundle."""
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
            """Load and validate one canonical coupled artifact bundle."""
            from v1ca1.spyglass.dark_light_glm import (
                load_dark_light_glm_artifact,
            )

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
            bundle = load_dark_light_glm_artifact(
                Path(row["artifact_manifest_path"]).parent
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
            selection = _fetch1_dict(DarkLightGLMSelection, key)
            result_key = {"dark_light_glm_id": selection["dark_light_glm_id"]}
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
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
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
            """Compute, write, and insert one held-out swap bundle."""
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
            """Load and validate one canonical held-out swap bundle."""
            from v1ca1.spyglass.swap_glm import load_swap_glm_artifact

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
            bundle = load_swap_glm_artifact(
                Path(row["artifact_manifest_path"]).parent
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
            """Compute, write, and insert one empirical swap-tuning bundle."""
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
            """Load and validate one canonical empirical swap bundle."""
            from v1ca1.spyglass.swap_tuning import (
                load_swap_tuning_curve_comparison_artifact,
            )

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
            bundle = load_swap_tuning_curve_comparison_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_swap_tuning_curve_comparison_artifact_link(
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
                ripples_table=Ripples,
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
            """Compute, write, and insert one RippleGLM bundle."""
            selection = _fetch1_dict(RippleGLMSelection, key)
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleGLMParameters,
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
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
            """Load and validate one canonical RippleGLM artifact bundle."""
            from v1ca1.spyglass.ripple_glm import load_ripple_glm_artifact

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                RippleGLMSelection,
                {"ripple_glm_id": row["ripple_glm_id"]},
            )
            parameters = _fetch1_dict(RippleGLMParameters, selection)
            animal_name, session_date = _session_identity(
                session_table, selection
            )
            bundle = load_ripple_glm_artifact(
                Path(row["artifact_manifest_path"]).parent
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
            selection = _fetch1_dict(RippleGLMSelection, key)
            result_key = {"ripple_glm_id": selection["ripple_glm_id"]}
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
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
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

    class CrossRegionXCorrParameters(spyglass_mixin, dj_module.Manual):
        definition = table_specs.CROSS_REGION_XCORR_PARAMETERS_DEFINITION

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one fixed ripple-xcorr parameter row."""
            validated = _validate_cross_region_xcorr_parameter_row(row)
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
                _validate_cross_region_xcorr_parameter_row(parameters)
                for parameters in (
                    table_specs.CROSS_REGION_XCORR_PARAMETER_PRESETS
                )
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    CrossRegionXCorrParameters = main_schema(CrossRegionXCorrParameters)
    main_context["CrossRegionXCorrParameters"] = CrossRegionXCorrParameters

    class CrossRegionXCorrSelection(spyglass_mixin, dj_module.Manual):
        definition = table_specs.CROSS_REGION_XCORR_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one exact-ripple xcorr."""
            row = _cross_region_xcorr_selection_row(
                key=key,
                ripples_table=Ripples,
                epoch_intervals_table=EpochIntervals,
                region_sorted_spikes_group_table=RegionSortedSpikesGroup,
                parameters_table=CrossRegionXCorrParameters,
                nwbfile_table=nwbfile_table,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    CrossRegionXCorrSelection = main_schema(CrossRegionXCorrSelection)
    main_context["CrossRegionXCorrSelection"] = CrossRegionXCorrSelection

    class CrossRegionXCorr(spyglass_mixin, dj_module.Computed):
        definition = table_specs.CROSS_REGION_XCORR_DEFINITION
        _compute_hook = staticmethod(cross_region_xcorr_compute_hook)
        _register_existing_hook = staticmethod(
            cross_region_xcorr_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one CrossRegionXCorr bundle."""
            selection = _fetch1_dict(CrossRegionXCorrSelection, key)
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=CrossRegionXCorrParameters,
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "cross_region_xcorr_id": selection[
                            "cross_region_xcorr_id"
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
        def load_cross_region_xcorr_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Load and validate one canonical CrossRegionXCorr bundle."""
            from v1ca1.spyglass.cross_region_xcorr import (
                load_cross_region_xcorr_artifact,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                CrossRegionXCorrSelection,
                {"cross_region_xcorr_id": row["cross_region_xcorr_id"]},
            )
            parameters = _fetch1_dict(
                CrossRegionXCorrParameters,
                selection,
            )
            animal_name, session_date = _session_identity(
                session_table,
                selection,
            )
            bundle = load_cross_region_xcorr_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_cross_region_xcorr_artifact_link(
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
                    "Registered CrossRegionXCorr results are immutable; "
                    "create a new selection instead of overwriting."
                )
            selection = _fetch1_dict(CrossRegionXCorrSelection, key)
            result_key = {
                "cross_region_xcorr_id": selection["cross_region_xcorr_id"]
            }
            existing = _existing_result_row(cls, result_key)
            if existing is not None:
                if skip_duplicates:
                    return existing
                raise ValueError(
                    "CrossRegionXCorr already contains this immutable selection."
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
                    parameters_table=CrossRegionXCorrParameters,
                    ripples_table=Ripples,
                    epoch_intervals_table=EpochIntervals,
                    region_sorted_spikes_group_table=(
                        RegionSortedSpikesGroup
                    ),
                    session_table=session_table,
                    nwbfile_table=nwbfile_table,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "cross_region_xcorr_id": selection[
                    "cross_region_xcorr_id"
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

    CrossRegionXCorr = main_schema(CrossRegionXCorr)
    main_context["CrossRegionXCorr"] = CrossRegionXCorr

    class RippleDecodingComparisonParameters(
        spyglass_mixin, dj_module.Manual
    ):
        definition = (
            table_specs.RIPPLE_DECODING_COMPARISON_PARAMETERS_DEFINITION
        )

        @classmethod
        def insert_parameters(
            cls,
            row: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate and insert one fixed ripple-decoding parameter row."""
            validated = _validate_ripple_decoding_comparison_parameter_row(
                row
            )
            cls.insert1(validated, skip_duplicates=skip_duplicates)
            return validated

        @classmethod
        def insert_defaults(
            cls,
            *,
            skip_duplicates: bool = True,
        ) -> list[dict[str, Any]]:
            """Explicitly insert the fixed manuscript decoding preset."""
            rows = [
                _validate_ripple_decoding_comparison_parameter_row(parameters)
                for parameters in (
                    table_specs.RIPPLE_DECODING_COMPARISON_PARAMETER_PRESETS
                )
            ]
            cls.insert(rows, skip_duplicates=skip_duplicates)
            return rows

    RippleDecodingComparisonParameters = main_schema(
        RippleDecodingComparisonParameters
    )
    main_context["RippleDecodingComparisonParameters"] = (
        RippleDecodingComparisonParameters
    )

    class RippleDecodingComparisonSelection(
        spyglass_mixin, dj_module.Manual
    ):
        definition = table_specs.RIPPLE_DECODING_COMPARISON_SELECTION_DEFINITION

        @classmethod
        def insert_selection(
            cls,
            key: Mapping[str, Any],
            *,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Validate, freeze, identify, and insert one decoding selection."""
            row = _ripple_decoding_comparison_selection_row(
                key=key,
                ripples_table=Ripples,
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
                parameters_table=RippleDecodingComparisonParameters,
                session_table=session_table,
                nwbfile_table=nwbfile_table,
            )
            cls.insert1(row, skip_duplicates=skip_duplicates)
            return row

    RippleDecodingComparisonSelection = main_schema(
        RippleDecodingComparisonSelection
    )
    main_context["RippleDecodingComparisonSelection"] = (
        RippleDecodingComparisonSelection
    )

    class RippleDecodingComparison(spyglass_mixin, dj_module.Computed):
        definition = table_specs.RIPPLE_DECODING_COMPARISON_DEFINITION
        _compute_hook = staticmethod(ripple_decoding_comparison_compute_hook)
        _register_existing_hook = staticmethod(
            ripple_decoding_comparison_register_hook
        )

        def make(self, key: Mapping[str, Any]) -> None:
            """Compute, write, and insert one ripple-decoding bundle."""
            selection = _fetch1_dict(
                RippleDecodingComparisonSelection, key
            )
            artifact_row = dict(
                self._compute_hook(
                    key=selection,
                    parameters_table=RippleDecodingComparisonParameters,
                    ripples_table=Ripples,
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
                    nwbfile_table=nwbfile_table,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            try:
                self.insert1(
                    {
                        "ripple_decoding_comparison_id": selection[
                            "ripple_decoding_comparison_id"
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
        def load_ripple_decoding_comparison_bundle(
            cls,
            key: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Load and validate one canonical ripple-decoding bundle."""
            from v1ca1.spyglass.ripple_decoding_comparison import (
                load_ripple_decoding_comparison_artifact,
            )

            row = _fetch1_dict(cls, key)
            selection = _fetch1_dict(
                RippleDecodingComparisonSelection,
                {
                    "ripple_decoding_comparison_id": row[
                        "ripple_decoding_comparison_id"
                    ]
                },
            )
            parameters = _fetch1_dict(
                RippleDecodingComparisonParameters, selection
            )
            animal_name, session_date = _session_identity(
                session_table, selection
            )
            bundle = load_ripple_decoding_comparison_artifact(
                Path(row["artifact_manifest_path"]).parent
            )
            _validate_ripple_decoding_comparison_artifact_link(
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
            source_ca1_decoded_path: Path | str,
            source_v1_decoded_path: Path | str,
            source_ripple_metrics_path: Path | str,
            source_epoch_summary_path: Path | str,
            source_result_path: Path | str,
            overwrite: bool = False,
            source_v1ca1_git_commit: str | None = None,
            source_spyglass_git_commit: str | None = None,
            skip_duplicates: bool = False,
        ) -> dict[str, Any]:
            """Verify and insert one exact five-file legacy decoding set."""
            if overwrite:
                raise ValueError(
                    "Registered RippleDecodingComparison results are "
                    "immutable; create a new selection instead of overwriting."
                )
            selection = _fetch1_dict(
                RippleDecodingComparisonSelection, key
            )
            result_key = {
                "ripple_decoding_comparison_id": selection[
                    "ripple_decoding_comparison_id"
                ]
            }
            existing = _existing_result_row(cls, result_key)
            if existing is not None:
                if skip_duplicates:
                    return existing
                raise ValueError(
                    "RippleDecodingComparison already contains this "
                    "immutable selection."
                )
            artifact_row = dict(
                cls._register_existing_hook(
                    key=selection,
                    source_ca1_decoded_path=Path(source_ca1_decoded_path),
                    source_v1_decoded_path=Path(source_v1_decoded_path),
                    source_ripple_metrics_path=Path(
                        source_ripple_metrics_path
                    ),
                    source_epoch_summary_path=Path(
                        source_epoch_summary_path
                    ),
                    source_result_path=Path(source_result_path),
                    parameters_table=RippleDecodingComparisonParameters,
                    ripples_table=Ripples,
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
                    nwbfile_table=nwbfile_table,
                    source_v1ca1_git_commit=source_v1ca1_git_commit,
                    source_spyglass_git_commit=source_spyglass_git_commit,
                    artifact_root=artifact_root,
                )
            )
            created_artifact_paths = list(
                artifact_row.pop("_created_artifact_paths", ())
            )
            row = {
                "ripple_decoding_comparison_id": selection[
                    "ripple_decoding_comparison_id"
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

    RippleDecodingComparison = main_schema(RippleDecodingComparison)
    main_context["RippleDecodingComparison"] = RippleDecodingComparison

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

    return {
        "epoch_intervals": EpochIntervals,
        "trajectory_intervals": TrajectoryIntervals,
        "ripples": Ripples,
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
        "dpp_encoding_comparison_parameters": (
            DPPEncodingComparisonParameters
        ),
        "dpp_encoding_comparison_selection": DPPEncodingComparisonSelection,
        "dpp_encoding_comparison": DPPEncodingComparison,
        "path_progression_decoding_parameters": (
            PathProgressionDecodingParameters
        ),
        "path_progression_decoding_comparison_selection": (
            PathProgressionDecodingComparisonSelection
        ),
        "path_progression_decoding_comparison": (
            PathProgressionDecodingComparison
        ),
        "path_specific_place_decoding_parameters": (
            PathSpecificPlaceDecodingParameters
        ),
        "path_specific_place_decoding_selection": (
            PathSpecificPlaceDecodingSelection
        ),
        "path_specific_place_decoding": PathSpecificPlaceDecoding,
        "motor_encoding_comparison_parameters": (
            MotorEncodingComparisonParameters
        ),
        "motor_encoding_comparison_selection": (
            MotorEncodingComparisonSelection
        ),
        "motor_encoding_comparison": MotorEncodingComparison,
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
        "cross_region_xcorr_parameters": CrossRegionXCorrParameters,
        "cross_region_xcorr_selection": CrossRegionXCorrSelection,
        "cross_region_xcorr": CrossRegionXCorr,
        "ripple_decoding_comparison_parameters": (
            RippleDecodingComparisonParameters
        ),
        "ripple_decoding_comparison_selection": (
            RippleDecodingComparisonSelection
        ),
        "ripple_decoding_comparison": RippleDecodingComparison,
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
