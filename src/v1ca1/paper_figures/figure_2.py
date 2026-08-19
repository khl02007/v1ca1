"""Generate Figure 2 with dark-light path-tuning preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS, TRAJECTORY_TYPES
from v1ca1.helper.wtrack import (
    get_wtrack_segment_edges,
    get_wtrack_total_length,
)
from v1ca1.paper_figures import figure_2_old as _figure_2
from v1ca1.paper_figures.datasets import (
    DatasetId,
    get_processed_datasets,
    normalize_dataset_id,
)
from v1ca1.paper_figures.figure_1 import get_stability_table_path
from v1ca1.paper_figures.style import (
    EMPHASIS_HISTOGRAM_KWARGS,
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)


DEFAULT_OUTPUT_DIR = _figure_2.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_NAME = "figure_2"
LEGACY_CACHE_FIGURE_NAME = "figure_2_new"
DEFAULT_OUTPUT_FORMAT = _figure_2.DEFAULT_OUTPUT_FORMAT
FIGURE_FORMATS = _figure_2.FIGURE_FORMATS
DEFAULT_REGIONS = _figure_2.DEFAULT_REGIONS
DEFAULT_FIGURE_WIDTH_MM = _figure_2.DEFAULT_FIGURE_WIDTH_MM
FIGURE_2_PANEL_A_EXAMPLES = (
    (
        "L15",
        "20241121",
        "v1",
        409,
        ("center_to_left", "right_to_center"),
    ),
    _figure_2.FIGURE_2_PANEL_A_EXAMPLES[0],
    _figure_2.FIGURE_2_PANEL_A_EXAMPLES[3],
    (
        "L15",
        "20241121",
        "v1",
        418,
        ("center_to_left", "right_to_center"),
    ),
    (
        "L14",
        "20240611",
        "v1",
        172,
        ("center_to_right", "left_to_center"),
    ),
    _figure_2.FIGURE_2_PANEL_A_EXAMPLES[1],
    (
        "L15",
        "20241121",
        "v1",
        70,
        ("center_to_right", "left_to_center"),
    ),
    _figure_2.FIGURE_2_PANEL_A_EXAMPLES[2],
)
FIGURE_2_PANEL_A_Y_MAX_OVERRIDES = {3: 85.0}
FIGURE_2_PANEL_A_WTRACK_SCALE = 1.5
FIGURE_2_PANEL_A_YLABEL_X = -0.44
PANEL_A_HEIGHT_MM = 2.0 * _figure_2.PANEL_A_SINGLE_ROW_HEIGHT_MM
DEFAULT_FIGURE_HEIGHT_MM = (
    PANEL_A_HEIGHT_MM
    + _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM
)
PANEL_BC_VERTICAL_SHIFT_MM = 1.9
PANEL_I_HORIZONTAL_AXIS_BOUNDS = (0.07, 0.86)
PANEL_BC_ROW_WIDTH_RATIOS = (1.0, 1.0)
PANEL_B_ROW_WSPACE = _figure_2.PANEL_B2C2_ROW_WSPACE
PANEL_B_SCHEMATIC_RELATIVE_WIDTH = 0.2624
PANEL_B_PROFILE_RELATIVE_WIDTH = 0.280
PANEL_B_PROFILE_WIDTH_SCALE_FROM_PANEL_C = 0.8
PANEL_B_COMPONENT_RELATIVE_GAP = 0.170
PANEL_B_SCHEMATIC_RELATIVE_Y = 0.0375
PANEL_B_SCHEMATIC_RELATIVE_HEIGHT = 0.825
PANEL_B_PROFILE_RELATIVE_Y = 0.220
PANEL_B_PROFILE_RELATIVE_HEIGHT = 0.620
PANEL_B_PATH_ORDER = TRAJECTORY_TYPES
PANEL_B_PATH_LABELS = {
    "center_to_left": "C to L",
    "left_to_center": "L to C",
    "center_to_right": "C to R",
    "right_to_center": "R to C",
}
PANEL_B_TUNING_SIMILARITY_CACHE_VERSION = 6
PANEL_B_TUNING_SIMILARITY_CACHE_PREFIX = (
    "figure_2_new_panel_b_tuning_similarity"
)
PANEL_B_TUNING_SIMILARITY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "path",
    "trajectory_type",
    "dark_epoch",
    "light_epoch",
    "tuning_similarity_index",
    "dark_area",
    "light_area",
    "light_dark_gain_ratio",
    "segment_edges_normalized",
    "segment_tuning_similarity_indices",
    "segment_similarity_statuses",
    "dark_segment_mean_firing_rates_hz",
    "light_segment_mean_firing_rates_hz",
    "dark_segment_tuning_curve_areas",
    "light_segment_tuning_curve_areas",
    "eligible_segments",
    "n_eligible_segments",
    "whole_path_tuning_correlation",
    "whole_path_correlation_status",
    "segment_tuning_correlations",
    "segment_correlation_statuses",
    "eligible_correlation_segments",
    "n_eligible_correlation_segments",
    "dark_epoch_movement_firing_rate_hz",
    "light_epoch_movement_firing_rate_hz",
    "dark_path_movement_firing_rate_hz",
    "light_path_movement_firing_rate_hz",
    "dark_path_movement_firing_rate_status",
    "light_path_movement_firing_rate_status",
    "dark_n_odd_spikes",
    "dark_n_even_spikes",
    "dark_odd_duration_s",
    "dark_even_duration_s",
    "light_n_odd_spikes",
    "light_n_even_spikes",
    "light_odd_duration_s",
    "light_even_duration_s",
    "dark_stability_correlation",
    "light_stability_correlation",
    "dark_stability_shape_overlap",
    "light_stability_shape_overlap",
    "dark_shape_overlap_status",
    "light_shape_overlap_status",
    "dark_odd_tuning_curve_area",
    "dark_even_tuning_curve_area",
    "light_odd_tuning_curve_area",
    "light_even_tuning_curve_area",
    "dark_segment_stability_shape_overlaps",
    "light_segment_stability_shape_overlaps",
    "dark_segment_shape_overlap_statuses",
    "light_segment_shape_overlap_statuses",
    "passes_dark_path_rate_qc",
    "passes_light_path_rate_qc",
    "passes_dark_stability_qc",
    "passes_light_stability_qc",
    "passes_dark_epoch_rate_qc",
    "passes_light_epoch_rate_qc",
    "passes_unit_qc",
    "passes_qc",
    "qc_status",
    "dark_n_finite_bins",
    "light_n_finite_bins",
    "n_paired_finite_bins",
    "similarity_status",
    "dark_tuning_curve_path",
    "light_tuning_curve_path",
    "stability_table_path",
    "cache_version",
)
PANEL_B_TUNING_AVERAGE_CACHE_PREFIX = (
    "figure_2_new_panel_b_tuning_average"
)
PANEL_B_TUNING_AVERAGE_CACHE_VERSION = 2
PANEL_B_TUNING_AVERAGE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "dark_epoch_movement_firing_rate_hz",
    "light_epoch_movement_firing_rate_hz",
    "passes_dark_epoch_rate_qc",
    "passes_light_epoch_rate_qc",
    "passes_unit_qc",
    "unit_qc_status",
    "tuning_average_index",
    "n_eligible_paths",
    "eligible_paths",
    "n_eligible_segments",
    "eligible_path_segments",
    "average_status",
    "cache_version",
)
PANEL_B_DARK_SPLIT_HALF_CACHE_PREFIX = (
    "figure_2_new_panel_b_dark_split_half_average"
)
PANEL_B_DARK_SPLIT_HALF_CACHE_VERSION = 1
PANEL_B_DARK_SPLIT_HALF_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "dark_epoch_movement_firing_rate_hz",
    "passes_dark_epoch_rate_qc",
    "dark_split_half_tuning_stability_index",
    "n_eligible_paths",
    "eligible_paths",
    "average_status",
    "cache_version",
)
PANEL_B_SPLIT_HALF_CACHE_PREFIX = (
    "figure_2_new_panel_b_dark_light_split_half_average"
)
PANEL_B_SPLIT_HALF_CACHE_VERSION = 4
PANEL_B_SPLIT_HALF_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "dark_epoch_movement_firing_rate_hz",
    "light_epoch_movement_firing_rate_hz",
    "passes_dark_epoch_rate_qc",
    "passes_light_epoch_rate_qc",
    "passes_unit_qc",
    "split_half_tuning_stability_index",
    "n_eligible_paths",
    "eligible_paths",
    "average_status",
    "cache_version",
)
PANEL_B_CIRCULAR_NULL_CACHE_PREFIX = (
    "figure_2_new_panel_b_circular_shift_null"
)
PANEL_B_CIRCULAR_NULL_CACHE_VERSION = 6
PANEL_B_CIRCULAR_NULL_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "passes_dark_epoch_rate_qc",
    "passes_light_epoch_rate_qc",
    "passes_unit_qc",
    "minimum_null_tuning_stability_index",
    "n_eligible_paths",
    "eligible_paths",
    "n_circular_shifts",
    "circular_shifts_per_path",
    "null_status",
    "cache_version",
)
PANEL_D_ACHIEVABLE_STABILITY_CACHE_PREFIX = (
    "figure_2_new_panel_d_achievable_stability"
)
PANEL_D_ACHIEVABLE_STABILITY_CACHE_VERSION = 6
PANEL_D_ACHIEVABLE_STABILITY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "observed_tuning_stability_index",
    "matched_split_half_tuning_stability_index",
    "minimum_null_tuning_stability_index",
    "achievable_stability",
    "denominator",
    "n_eligible_paths",
    "eligible_paths",
    "n_eligible_segments",
    "eligible_path_segments",
    "achievable_status",
    "n_null_shifts",
    "cache_version",
)
FULL_PATH_ACHIEVABLE_STABILITY_CACHE_PREFIX = (
    "figure_2_new_full_path_achievable_stability"
)
FULL_PATH_ACHIEVABLE_STABILITY_CACHE_VERSION = 3
FULL_PATH_ACHIEVABLE_STABILITY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "dark_epoch_movement_firing_rate_hz",
    "light_epoch_movement_firing_rate_hz",
    "passes_unit_qc",
    "apply_stability_filter",
    "passes_stability_filter",
    "observed_tuning_stability_index",
    "matched_split_half_tuning_stability_index",
    "minimum_null_tuning_stability_index",
    "achievable_stability",
    "denominator",
    "n_eligible_paths",
    "eligible_paths",
    "achievable_status",
    "n_null_shifts",
    "circular_shifts_per_path",
    "cache_version",
)
PANEL_H_SHIFT_PROFILE_CACHE_PREFIX = (
    "figure_2_new_panel_h_circular_shift_overlap_profile"
)
PANEL_H_SHIFT_PROFILE_CACHE_VERSION = 4
PANEL_H_SHIFT_PROFILE_GRID = np.linspace(-0.5, 0.5, 101)
PANEL_H_SHIFT_PROFILE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "path",
    "dark_epoch",
    "light_epoch",
    "normalized_shift",
    "overlap",
    "minimum_overlap",
    "dark_split_half_overlap",
    "light_split_half_overlap",
    "split_half_overlap",
    "rescaling_denominator",
    "rescaled_overlap",
    "rescaling_status",
    "n_progression_bins",
    "profile_status",
    "cache_version",
)
SEGMENT_MATCHED_ACHIEVABLE_STABILITY_CACHE_PREFIX = (
    "figure_2_new_segment_matched_achievable_stability"
)
SEGMENT_MATCHED_ACHIEVABLE_STABILITY_CACHE_VERSION = 3
SEGMENT_MATCHED_ACHIEVABLE_STABILITY_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "observed_tuning_stability_index",
    "matched_split_half_tuning_stability_index",
    "minimum_null_tuning_stability_index",
    "achievable_stability",
    "denominator",
    "n_source_segments",
    "n_eligible_paths",
    "eligible_paths",
    "n_eligible_segments",
    "eligible_path_segments",
    "achievable_status",
    "n_null_shifts",
    "cache_version",
)
DEFAULT_PANEL_B_NULL_N_PERMUTATIONS = 1000
DEFAULT_PANEL_B_NULL_RANDOM_SEED = 47
PANEL_B_TUNING_SIMILARITY_BINS = np.linspace(0.0, 1.0, 21)
PANEL_D_ACHIEVABLE_STABILITY_BINS = np.linspace(-1.0, 1.5, 26)
SEGMENT_OVERLAP_RESPONSE_CACHE_PREFIX = (
    "figure_2_new_segment_overlap_response"
)
SEGMENT_OVERLAP_RESPONSE_CACHE_VERSION = 1
SEGMENT_OVERLAP_RESPONSE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "path",
    "dark_epoch",
    "light_epoch",
    "segment_index",
    "segment_start_normalized",
    "segment_end_normalized",
    "segment_tuning_similarity_index",
    "segment_similarity_status",
    "dark_segment_mean_firing_rate_hz",
    "light_segment_mean_firing_rate_hz",
    "dark_segment_tuning_curve_area",
    "light_segment_tuning_curve_area",
    "segment_response_ratio",
    "log2_segment_response_ratio",
    "response_ratio_status",
    "passes_unit_qc",
    "passes_qc",
    "passes_segment_rate_qc",
    "included",
    "inclusion_status",
    "cache_version",
)
SEGMENT_STABILITY_REFERENCE_CACHE_PREFIX = (
    "figure_2_new_segment_stability_reference"
)
SEGMENT_STABILITY_REFERENCE_CACHE_VERSION = 3
SEGMENT_STABILITY_REFERENCE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "path",
    "dark_epoch",
    "light_epoch",
    "segment_index",
    "observed_segment_stability",
    "dark_split_half_segment_stability",
    "light_split_half_segment_stability",
    "split_half_segment_stability",
    "split_half_status",
    "minimum_null_segment_stability",
    "null_status",
    "achievable_segment_stability",
    "denominator",
    "achievable_status",
    "n_circular_shifts",
    "cache_version",
)
TUNING_CORRELATION_VARIANTS = ("whole_path", "physical_segments")
# Use 21 equal bins so zero lies at the center of a bin rather than on an
# edge; the per-neuron circular-null expectations are tightly centered there.
TUNING_CORRELATION_BINS = np.linspace(-1.0, 1.0, 22)
TUNING_CORRELATION_CACHE_VERSION = 1
TUNING_CORRELATION_AVERAGE_CACHE_PREFIX = (
    "figure_2_new_tuning_correlation_average"
)
TUNING_CORRELATION_SPLIT_HALF_CACHE_PREFIX = (
    "figure_2_new_tuning_correlation_split_half"
)
TUNING_CORRELATION_NULL_CACHE_PREFIX = (
    "figure_2_new_tuning_correlation_circular_null"
)
ACHIEVABLE_TUNING_CORRELATION_CACHE_PREFIX = (
    "figure_2_new_achievable_tuning_correlation"
)
TUNING_CORRELATION_AVERAGE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "dark_epoch_movement_firing_rate_hz",
    "light_epoch_movement_firing_rate_hz",
    "passes_dark_epoch_rate_qc",
    "passes_light_epoch_rate_qc",
    "passes_unit_qc",
    "mean_tuning_correlation",
    "n_eligible_paths",
    "eligible_paths",
    "n_eligible_path_segments",
    "eligible_path_segments",
    "average_status",
    "variant",
    "cache_version",
)
TUNING_CORRELATION_SPLIT_HALF_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "passes_dark_epoch_rate_qc",
    "passes_light_epoch_rate_qc",
    "passes_unit_qc",
    "split_half_tuning_correlation",
    "n_eligible_paths",
    "eligible_paths",
    "reference_status",
    "variant",
    "cache_version",
)
TUNING_CORRELATION_NULL_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "passes_unit_qc",
    "permutation",
    "null_tuning_correlation",
    "n_eligible_paths",
    "eligible_paths",
    "null_status",
    "n_permutations",
    "random_seed",
    "variant",
    "cache_version",
)
ACHIEVABLE_TUNING_CORRELATION_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "dark_epoch",
    "light_epoch",
    "observed_tuning_correlation",
    "matched_split_half_tuning_correlation",
    "mean_null_tuning_correlation",
    "achievable_tuning_correlation",
    "denominator",
    "n_eligible_paths",
    "eligible_paths",
    "achievable_status",
    "n_null_permutations",
    "variant",
    "cache_version",
)
DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ = 0.5
DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ = 0.5
_TUNING_SIMILARITY_EPS = 1e-12
PANEL_C_TURN_PATHS = {
    "left": ("center_to_left", "right_to_center"),
    "right": ("center_to_right", "left_to_center"),
}
PANEL_C_TURN_LABELS = {"left": "Left turns", "right": "Right turns"}
PANEL_C_PATH_INVARIANCE_DELTA_BINS = np.linspace(-1.0, 1.0, 25)
PANEL_C_PATH_INVARIANCE_CACHE_VERSION = 1
PANEL_C_PATH_INVARIANCE_CACHE_PREFIX = (
    "figure_2_new_panel_c_path_invariance"
)
PANEL_C_PATH_INVARIANCE_COLUMNS = (
    "animal_name",
    "date",
    "region",
    "unit",
    "turn_direction",
    "x_path",
    "y_path",
    "dark_epoch",
    "light_epoch",
    "dark_path_invariance_index",
    "light_path_invariance_index",
    "dark_x_area",
    "dark_y_area",
    "dark_overlap_area",
    "light_x_area",
    "light_y_area",
    "light_overlap_area",
    "dark_x_n_finite_bins",
    "dark_y_n_finite_bins",
    "dark_n_paired_finite_bins",
    "light_x_n_finite_bins",
    "light_y_n_finite_bins",
    "light_n_paired_finite_bins",
    "dark_path_invariance_status",
    "light_path_invariance_status",
    "dark_x_path_movement_firing_rate_hz",
    "dark_y_path_movement_firing_rate_hz",
    "light_x_path_movement_firing_rate_hz",
    "light_y_path_movement_firing_rate_hz",
    "dark_x_path_movement_firing_rate_status",
    "dark_y_path_movement_firing_rate_status",
    "light_x_path_movement_firing_rate_status",
    "light_y_path_movement_firing_rate_status",
    "dark_x_stability_correlation",
    "dark_y_stability_correlation",
    "light_x_stability_correlation",
    "light_y_stability_correlation",
    "passes_dark_x_path_rate_qc",
    "passes_dark_y_path_rate_qc",
    "passes_light_x_path_rate_qc",
    "passes_light_y_path_rate_qc",
    "passes_dark_x_stability_qc",
    "passes_dark_y_stability_qc",
    "passes_light_x_stability_qc",
    "passes_light_y_stability_qc",
    "passes_qc",
    "qc_status",
    "dark_x_tuning_curve_path",
    "dark_y_tuning_curve_path",
    "light_x_tuning_curve_path",
    "light_y_tuning_curve_path",
    "stability_table_path",
    "cache_version",
)
_PATH_INVARIANCE_EPS = 1e-12


def compute_pooled_path_movement_firing_rate(
    n_odd_spikes: float,
    n_even_spikes: float,
    odd_duration_s: float,
    even_duration_s: float,
) -> dict[str, Any]:
    """Return the pooled odd/even movement rate for one cell and path."""
    spike_counts = np.asarray((n_odd_spikes, n_even_spikes), dtype=float)
    durations = np.asarray((odd_duration_s, even_duration_s), dtype=float)
    result: dict[str, Any] = {
        "path_movement_firing_rate_hz": float("nan"),
        "path_movement_firing_rate_status": "valid",
    }
    if (
        not np.isfinite(spike_counts).all()
        or np.any(spike_counts < 0.0)
        or not np.equal(spike_counts, np.floor(spike_counts)).all()
    ):
        result["path_movement_firing_rate_status"] = "invalid_spike_count"
        return result
    if not np.isfinite(durations).all() or np.any(durations < 0.0):
        result["path_movement_firing_rate_status"] = "invalid_duration"
        return result
    pooled_duration = float(durations.sum())
    if pooled_duration <= 0.0:
        result["path_movement_firing_rate_status"] = "invalid_duration"
        return result
    result["path_movement_firing_rate_hz"] = float(
        spike_counts.sum() / pooled_duration
    )
    return result


def compute_path_tuning_similarity(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
    *,
    eps: float = _TUNING_SIMILARITY_EPS,
) -> dict[str, Any]:
    """Return unit-area dark-light overlap and curve-level QC for one path."""
    dark = np.asarray(dark_curve, dtype=float)
    light = np.asarray(light_curve, dtype=float)
    shapes_match = dark.ndim == 1 and light.ndim == 1 and dark.shape == light.shape
    result: dict[str, Any] = {
        "tuning_similarity_index": float("nan"),
        "dark_area": float("nan"),
        "light_area": float("nan"),
        "light_dark_gain_ratio": float("nan"),
        "dark_n_finite_bins": int(np.count_nonzero(np.isfinite(dark))),
        "light_n_finite_bins": int(np.count_nonzero(np.isfinite(light))),
        "n_paired_finite_bins": (
            int(np.count_nonzero(np.isfinite(dark) & np.isfinite(light)))
            if shapes_match
            else 0
        ),
        "similarity_status": "valid",
    }
    if not shapes_match:
        result["similarity_status"] = "shape_mismatch"
        return result
    if result["dark_n_finite_bins"] == 0:
        result["similarity_status"] = "no_finite_dark_bins"
        return result
    if result["light_n_finite_bins"] == 0:
        result["similarity_status"] = "no_finite_light_bins"
        return result
    if np.isinf(dark).any():
        result["similarity_status"] = "nonfinite_dark_curve"
        return result
    if np.isinf(light).any():
        result["similarity_status"] = "nonfinite_light_curve"
        return result
    if np.any(dark[np.isfinite(dark)] < -float(eps)) or np.any(
        light[np.isfinite(light)] < -float(eps)
    ):
        result["similarity_status"] = "negative_firing_rate"
        return result

    from v1ca1.task_progression.similarity import interpolate_nans

    dark = np.maximum(interpolate_nans(dark), 0.0)
    light = np.maximum(interpolate_nans(light), 0.0)
    dark_area = float(dark.sum())
    light_area = float(light.sum())
    result["dark_area"] = dark_area
    result["light_area"] = light_area
    if dark_area <= float(eps):
        result["similarity_status"] = "nonpositive_dark_area"
        return result
    if light_area <= float(eps):
        result["similarity_status"] = "nonpositive_light_area"
        return result

    result["light_dark_gain_ratio"] = light_area / dark_area
    similarity = float(np.minimum(dark / dark_area, light / light_area).sum())
    result["tuning_similarity_index"] = float(np.clip(similarity, 0.0, 1.0))
    return result


def compute_path_tuning_correlation(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
    *,
    eps: float = _TUNING_SIMILARITY_EPS,
) -> dict[str, Any]:
    """Return Pearson correlation and explicit curve QC for one path."""
    from v1ca1.task_progression.similarity import interpolate_nans

    dark = np.asarray(dark_curve, dtype=float)
    light = np.asarray(light_curve, dtype=float)
    result: dict[str, Any] = {
        "tuning_correlation": float("nan"),
        "correlation_status": "valid",
    }
    if dark.ndim != 1 or light.ndim != 1 or dark.shape != light.shape:
        result["correlation_status"] = "shape_mismatch"
        return result
    if not np.isfinite(dark).any():
        result["correlation_status"] = "no_finite_dark_bins"
        return result
    if not np.isfinite(light).any():
        result["correlation_status"] = "no_finite_light_bins"
        return result
    if np.isinf(dark).any():
        result["correlation_status"] = "nonfinite_dark_curve"
        return result
    if np.isinf(light).any():
        result["correlation_status"] = "nonfinite_light_curve"
        return result
    dark = np.asarray(interpolate_nans(dark), dtype=float)
    light = np.asarray(interpolate_nans(light), dtype=float)
    if np.std(dark) <= float(eps):
        result["correlation_status"] = "constant_dark_curve"
        return result
    if np.std(light) <= float(eps):
        result["correlation_status"] = "constant_light_curve"
        return result
    correlation = float(np.corrcoef(dark, light)[0, 1])
    if not np.isfinite(correlation):
        result["correlation_status"] = "nonfinite_correlation"
        return result
    result["tuning_correlation"] = float(np.clip(correlation, -1.0, 1.0))
    return result


def compute_segmented_path_tuning_correlation(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
    progression: np.ndarray,
    segment_edges: np.ndarray,
    *,
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
) -> dict[str, Any]:
    """Return the flat mean Pearson r over active, nonconstant segments."""
    from v1ca1.task_progression.similarity import (
        interpolate_nans,
        make_segment_masks,
    )

    if min_segment_mean_firing_rate_hz < 0.0:
        raise ValueError(
            "min_segment_mean_firing_rate_hz must be non-negative."
        )
    dark = np.asarray(dark_curve, dtype=float)
    light = np.asarray(light_curve, dtype=float)
    progression = np.asarray(progression, dtype=float).reshape(-1)
    if (
        dark.ndim != 1
        or light.ndim != 1
        or dark.shape != light.shape
        or dark.size != progression.size
    ):
        raise ValueError(
            "Segmented correlation requires aligned one-dimensional curves."
        )
    masks = make_segment_masks(progression, segment_edges)
    correlations: list[float] = []
    statuses: list[str] = []
    dark_means: list[float] = []
    light_means: list[float] = []
    active_segments: list[int] = []
    eligible_segments: list[int] = []
    for index, mask in enumerate(masks):
        segment_dark = np.asarray(dark[mask], dtype=float)
        segment_light = np.asarray(light[mask], dtype=float)
        if np.isfinite(segment_dark).any() and not np.isinf(segment_dark).any():
            dark_mean = float(np.mean(interpolate_nans(segment_dark)))
        else:
            dark_mean = float("nan")
        if np.isfinite(segment_light).any() and not np.isinf(segment_light).any():
            light_mean = float(np.mean(interpolate_nans(segment_light)))
        else:
            light_mean = float("nan")
        dark_means.append(dark_mean)
        light_means.append(light_mean)
        active = bool(
            np.isfinite(max(dark_mean, light_mean))
            and max(dark_mean, light_mean)
            > float(min_segment_mean_firing_rate_hz)
        )
        if active:
            active_segments.append(index)
        result = compute_path_tuning_correlation(segment_dark, segment_light)
        correlation = float(result["tuning_correlation"])
        status = str(result["correlation_status"])
        correlations.append(correlation)
        statuses.append(status)
        if active and status == "valid" and np.isfinite(correlation):
            eligible_segments.append(index)
    values = np.asarray(
        [correlations[index] for index in eligible_segments],
        dtype=float,
    )
    if values.size:
        correlation = float(np.mean(values))
        status = "valid"
    elif active_segments:
        correlation = float("nan")
        status = "no_valid_correlation_segments"
    else:
        correlation = float("nan")
        status = "no_active_segments"
    return {
        "tuning_correlation": correlation,
        "correlation_status": status,
        "segment_tuning_correlations": json.dumps(correlations),
        "segment_correlation_statuses": json.dumps(statuses),
        "dark_segment_mean_firing_rates_hz": json.dumps(dark_means),
        "light_segment_mean_firing_rates_hz": json.dumps(light_means),
        "active_segments": json.dumps(active_segments),
        "eligible_segments": json.dumps(eligible_segments),
        "n_eligible_segments": len(eligible_segments),
    }


def compute_segmented_path_tuning_similarity(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
    progression: np.ndarray,
    segment_edges: np.ndarray,
    *,
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
) -> dict[str, Any]:
    """Return equal-mean dark/light overlap across active path segments."""
    from v1ca1.task_progression.similarity import (
        compute_segmented_shape_overlap,
    )

    if min_segment_mean_firing_rate_hz < 0.0:
        raise ValueError(
            "min_segment_mean_firing_rate_hz must be non-negative."
        )
    dark = np.asarray(dark_curve, dtype=float)
    light = np.asarray(light_curve, dtype=float)
    overlap = compute_segmented_shape_overlap(
        dark,
        light,
        progression,
        segment_edges,
    )
    scores = np.asarray(overlap["scores"], dtype=float)
    dark_means = np.asarray(overlap["mean_rates_a_hz"], dtype=float)
    light_means = np.asarray(overlap["mean_rates_b_hz"], dtype=float)
    statuses = [str(value) for value in overlap["statuses"]]
    eligible = np.isfinite(np.maximum(dark_means, light_means)) & (
        np.maximum(dark_means, light_means)
        > float(min_segment_mean_firing_rate_hz)
    )
    eligible_segments = np.flatnonzero(eligible).astype(int).tolist()
    invalid_eligible = [
        index
        for index in eligible_segments
        if statuses[index] != "valid" or not np.isfinite(scores[index])
    ]
    if invalid_eligible:
        similarity = float("nan")
        status = "invalid_eligible_segment"
    elif not eligible_segments:
        similarity = float("nan")
        status = "no_eligible_segments"
    else:
        similarity = float(np.mean(scores[eligible]))
        status = "valid"

    dark_areas = np.asarray(overlap["areas_a"], dtype=float)
    light_areas = np.asarray(overlap["areas_b"], dtype=float)
    dark_area = float(np.nansum(dark_areas))
    light_area = float(np.nansum(light_areas))
    finite_dark = np.isfinite(dark)
    finite_light = np.isfinite(light)
    return {
        "tuning_similarity_index": similarity,
        "dark_area": dark_area,
        "light_area": light_area,
        "light_dark_gain_ratio": (
            light_area / dark_area
            if dark_area > _TUNING_SIMILARITY_EPS
            else float("nan")
        ),
        "segment_edges_normalized": json.dumps(
            np.asarray(segment_edges, dtype=float).tolist()
        ),
        "segment_tuning_similarity_indices": json.dumps(
            scores.tolist()
        ),
        "segment_similarity_statuses": json.dumps(statuses),
        "dark_segment_mean_firing_rates_hz": json.dumps(
            dark_means.tolist()
        ),
        "light_segment_mean_firing_rates_hz": json.dumps(
            light_means.tolist()
        ),
        "dark_segment_tuning_curve_areas": json.dumps(dark_areas.tolist()),
        "light_segment_tuning_curve_areas": json.dumps(light_areas.tolist()),
        "eligible_segments": json.dumps(eligible_segments),
        "n_eligible_segments": len(eligible_segments),
        "dark_n_finite_bins": int(np.count_nonzero(finite_dark)),
        "light_n_finite_bins": int(np.count_nonzero(finite_light)),
        "n_paired_finite_bins": int(
            np.count_nonzero(finite_dark & finite_light)
        ),
        "similarity_status": status,
    }


def compute_path_invariance(
    x_curve: np.ndarray,
    y_curve: np.ndarray,
    *,
    eps: float = _PATH_INVARIANCE_EPS,
) -> dict[str, Any]:
    """Return raw-curve agreement for two aligned same-turn paths."""
    x = np.asarray(x_curve, dtype=float)
    y = np.asarray(y_curve, dtype=float)
    shapes_match = x.ndim == 1 and y.ndim == 1 and x.shape == y.shape
    result: dict[str, Any] = {
        "path_invariance_index": float("nan"),
        "x_area": float("nan"),
        "y_area": float("nan"),
        "overlap_area": float("nan"),
        "x_n_finite_bins": int(np.count_nonzero(np.isfinite(x))),
        "y_n_finite_bins": int(np.count_nonzero(np.isfinite(y))),
        "n_paired_finite_bins": (
            int(np.count_nonzero(np.isfinite(x) & np.isfinite(y)))
            if shapes_match
            else 0
        ),
        "path_invariance_status": "valid",
    }
    if not shapes_match:
        result["path_invariance_status"] = "shape_mismatch"
        return result
    if result["x_n_finite_bins"] == 0:
        result["path_invariance_status"] = "no_finite_x_bins"
        return result
    if result["y_n_finite_bins"] == 0:
        result["path_invariance_status"] = "no_finite_y_bins"
        return result
    if np.isinf(x).any():
        result["path_invariance_status"] = "nonfinite_x_curve"
        return result
    if np.isinf(y).any():
        result["path_invariance_status"] = "nonfinite_y_curve"
        return result
    if np.any(x[np.isfinite(x)] < -float(eps)) or np.any(
        y[np.isfinite(y)] < -float(eps)
    ):
        result["path_invariance_status"] = "negative_firing_rate"
        return result

    from v1ca1.task_progression.similarity import interpolate_nans

    x = np.maximum(interpolate_nans(x), 0.0)
    y = np.maximum(interpolate_nans(y), 0.0)
    x_area = float(x.sum())
    y_area = float(y.sum())
    overlap_area = float(np.minimum(x, y).sum())
    result["x_area"] = x_area
    result["y_area"] = y_area
    result["overlap_area"] = overlap_area
    total_area = x_area + y_area
    if total_area <= float(eps):
        result["path_invariance_status"] = "nonpositive_total_area"
        return result

    score = 2.0 * overlap_area / total_area
    result["path_invariance_index"] = float(np.clip(score, 0.0, 1.0))
    return result


def _require_aligned_tuning_curves(
    dark_curve: Any,
    light_curve: Any,
    *,
    dark_path: Path,
    light_path: Path,
) -> None:
    """Raise when paired tuning artifacts do not share exact dims and coordinates."""
    if tuple(dark_curve.dims) != tuple(light_curve.dims):
        raise ValueError(
            "Dark and light tuning curves have different dimensions: "
            f"{dark_path} has {dark_curve.dims!r}; "
            f"{light_path} has {light_curve.dims!r}."
        )
    for dimension in dark_curve.dims:
        if dimension not in dark_curve.coords or dimension not in light_curve.coords:
            raise ValueError(
                f"Paired tuning curves are missing coordinate {dimension!r}."
            )
        dark_values = np.asarray(dark_curve.coords[dimension].values)
        light_values = np.asarray(light_curve.coords[dimension].values)
        if dark_values.shape != light_values.shape or not np.array_equal(
            dark_values,
            light_values,
            equal_nan=True,
        ):
            raise ValueError(
                "Dark and light tuning curves have different "
                f"{dimension!r} coordinates: {dark_path} versus {light_path}."
            )
    if "unit" not in dark_curve.dims or dark_curve.ndim != 2:
        raise ValueError(
            "Tuning-curve artifacts must have exactly unit and path-position "
            f"dimensions; got {dark_curve.dims!r} in {dark_path}."
        )


def _load_session_stability_rows(
    data_root: Path,
    *,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_epoch: str,
    require_epoch_firing_rate: bool = False,
) -> tuple[Any, Path]:
    """Return uniquely indexed dark/light stability rows for one session."""
    import pandas as pd

    path = get_stability_table_path(data_root, animal_name, date)
    if not path.exists():
        raise FileNotFoundError(
            "Missing task-progression stability table. Expected "
            f"{path}. Run `python -m v1ca1.task_progression.stability` first."
        )
    table = pd.read_parquet(path)
    required_columns = [
        "unit",
        "region",
        "epoch",
        "trajectory_type",
        "n_odd_spikes",
        "n_even_spikes",
        "odd_duration_s",
        "even_duration_s",
        "stability_correlation",
        "stability_shape_overlap",
        "shape_overlap_status",
        "odd_tuning_curve_area",
        "even_tuning_curve_area",
        "segment_stability_shape_overlaps",
        "segment_shape_overlap_statuses",
    ]
    if require_epoch_firing_rate:
        required_columns.append("firing_rate_hz")
    missing = [column for column in required_columns if column not in table]
    if missing:
        raise ValueError(f"Stability table {path} is missing columns {missing!r}.")
    rows = table[
        (table["region"].astype(str) == str(region))
        & table["epoch"].astype(str).isin((str(dark_epoch), str(light_epoch)))
        & table["trajectory_type"].astype(str).isin(PANEL_B_PATH_ORDER)
    ].copy()
    rows["unit"] = pd.to_numeric(rows["unit"], errors="coerce")
    rows = rows[np.isfinite(rows["unit"].to_numpy(dtype=float))].copy()
    rows["unit"] = rows["unit"].astype(int)
    if rows.duplicated(["unit", "epoch", "trajectory_type"]).any():
        raise ValueError(
            f"Stability table {path} has duplicate unit/epoch/path rows."
        )
    return rows.set_index(["unit", "epoch", "trajectory_type"]), path


def _epoch_movement_firing_rate_lookup(stability: Any) -> dict[tuple[int, str], float]:
    """Return the whole-epoch movement rate repeated across path rows."""
    table = stability.reset_index()
    lookup: dict[tuple[int, str], float] = {}
    for (unit, epoch), rows in table.groupby(["unit", "epoch"], sort=False):
        values = np.asarray(rows["firing_rate_hz"], dtype=float)
        finite_values = values[np.isfinite(values)]
        unique_values = np.unique(finite_values)
        if unique_values.size > 1:
            raise ValueError(
                "Whole-epoch movement firing rate differs across path rows for "
                f"unit {int(unit)} in epoch {str(epoch)!r}."
            )
        lookup[(int(unit), str(epoch))] = (
            float(unique_values[0]) if unique_values.size else float("nan")
        )
    return lookup


def _qc_status(row: dict[str, Any]) -> str:
    """Return a compact explanation of failed dark/light eligibility checks."""
    failures = [
        name
        for name in (
            "dark_stability",
            "light_stability",
        )
        if not bool(row[f"passes_{name}_qc"])
    ]
    return "valid" if not failures else "fails_" + "_and_".join(failures)


def _build_session_tuning_similarity_table(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_epoch: str,
    min_epoch_movement_firing_rate_hz: float,
    min_path_movement_firing_rate_hz: float,
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float,
) -> Any:
    """Return eligible per-unit, per-path dark-light tuning similarities."""
    import pandas as pd
    import xarray as xr

    stability, stability_path = _load_session_stability_rows(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        dark_epoch=dark_epoch,
        light_epoch=light_epoch,
        require_epoch_firing_rate=True,
    )
    epoch_movement_rates = _epoch_movement_firing_rate_lookup(stability)
    records: list[dict[str, Any]] = []
    for path_name in PANEL_B_PATH_ORDER:
        dark_path = _figure_2.get_compute_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=dark_epoch,
            trajectory=path_name,
        )
        light_path = _figure_2.get_compute_tuning_curve_path(
            data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            epoch=light_epoch,
            trajectory=path_name,
        )
        for artifact_path in (dark_path, light_path):
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Missing tuning-curve artifact: {artifact_path}"
                )
        with xr.open_dataarray(dark_path) as dark_data, xr.open_dataarray(
            light_path
        ) as light_data:
            _require_aligned_tuning_curves(
                dark_data,
                light_data,
                dark_path=dark_path,
                light_path=light_path,
            )
            units = np.asarray(dark_data.coords["unit"].values, dtype=int)
            dark_values = np.asarray(dark_data.transpose("unit", ...).values)
            light_values = np.asarray(light_data.transpose("unit", ...).values)
            position_dimension = next(
                dimension for dimension in dark_data.dims if dimension != "unit"
            )
            progression = np.asarray(
                dark_data.coords[position_dimension].values,
                dtype=float,
            ) / get_wtrack_total_length(animal_name)
            segment_edges = get_wtrack_segment_edges(animal_name)

        for index, unit in enumerate(units):
            record: dict[str, Any] = {
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "unit": int(unit),
                "path": str(path_name),
                "trajectory_type": str(path_name),
                "dark_epoch": str(dark_epoch),
                "light_epoch": str(light_epoch),
                "dark_tuning_curve_path": str(dark_path),
                "light_tuning_curve_path": str(light_path),
                "stability_table_path": str(stability_path),
                "cache_version": PANEL_B_TUNING_SIMILARITY_CACHE_VERSION,
                **compute_segmented_path_tuning_similarity(
                    dark_values[index],
                    light_values[index],
                    progression,
                    segment_edges,
                    min_segment_mean_firing_rate_hz=(
                        min_segment_mean_firing_rate_hz
                    ),
                ),
            }
            whole_correlation = compute_path_tuning_correlation(
                dark_values[index],
                light_values[index],
            )
            segmented_correlation = compute_segmented_path_tuning_correlation(
                dark_values[index],
                light_values[index],
                progression,
                segment_edges,
                min_segment_mean_firing_rate_hz=(
                    min_segment_mean_firing_rate_hz
                ),
            )
            record["whole_path_tuning_correlation"] = whole_correlation[
                "tuning_correlation"
            ]
            record["whole_path_correlation_status"] = whole_correlation[
                "correlation_status"
            ]
            record["segment_tuning_correlations"] = segmented_correlation[
                "segment_tuning_correlations"
            ]
            record["segment_correlation_statuses"] = segmented_correlation[
                "segment_correlation_statuses"
            ]
            record["eligible_correlation_segments"] = (
                segmented_correlation["eligible_segments"]
            )
            record["n_eligible_correlation_segments"] = int(
                segmented_correlation["n_eligible_segments"]
            )
            for epoch_type, epoch in (
                ("dark", dark_epoch),
                ("light", light_epoch),
            ):
                epoch_rate = epoch_movement_rates.get(
                    (int(unit), str(epoch)),
                    float("nan"),
                )
                record[f"{epoch_type}_epoch_movement_firing_rate_hz"] = (
                    epoch_rate
                )
                record[f"passes_{epoch_type}_epoch_rate_qc"] = bool(
                    np.isfinite(epoch_rate)
                    and epoch_rate > float(min_epoch_movement_firing_rate_hz)
                )
                key = (int(unit), str(epoch), str(path_name))
                if key in stability.index:
                    qc_row = stability.loc[key]
                    correlation = float(qc_row["stability_correlation"])
                    shape_overlap = float(qc_row["stability_shape_overlap"])
                    shape_overlap_status = str(qc_row["shape_overlap_status"])
                    odd_curve_area = float(qc_row["odd_tuning_curve_area"])
                    even_curve_area = float(qc_row["even_tuning_curve_area"])
                    segment_shape_overlaps = str(
                        qc_row["segment_stability_shape_overlaps"]
                    )
                    segment_shape_overlap_statuses = str(
                        qc_row["segment_shape_overlap_statuses"]
                    )
                    n_odd_spikes = float(qc_row["n_odd_spikes"])
                    n_even_spikes = float(qc_row["n_even_spikes"])
                    odd_duration_s = float(qc_row["odd_duration_s"])
                    even_duration_s = float(qc_row["even_duration_s"])
                else:
                    correlation = float("nan")
                    shape_overlap = float("nan")
                    shape_overlap_status = "missing_stability_row"
                    odd_curve_area = float("nan")
                    even_curve_area = float("nan")
                    segment_shape_overlaps = json.dumps(
                        [float("nan")] * 3
                    )
                    segment_shape_overlap_statuses = json.dumps(
                        ["missing_stability_row"] * 3
                    )
                    n_odd_spikes = float("nan")
                    n_even_spikes = float("nan")
                    odd_duration_s = float("nan")
                    even_duration_s = float("nan")
                rate_result = compute_pooled_path_movement_firing_rate(
                    n_odd_spikes,
                    n_even_spikes,
                    odd_duration_s,
                    even_duration_s,
                )
                rate = float(rate_result["path_movement_firing_rate_hz"])
                record[f"{epoch_type}_n_odd_spikes"] = n_odd_spikes
                record[f"{epoch_type}_n_even_spikes"] = n_even_spikes
                record[f"{epoch_type}_odd_duration_s"] = odd_duration_s
                record[f"{epoch_type}_even_duration_s"] = even_duration_s
                record[f"{epoch_type}_path_movement_firing_rate_hz"] = rate
                record[f"{epoch_type}_path_movement_firing_rate_status"] = (
                    rate_result["path_movement_firing_rate_status"]
                )
                record[f"{epoch_type}_stability_correlation"] = correlation
                record[f"{epoch_type}_stability_shape_overlap"] = shape_overlap
                record[f"{epoch_type}_shape_overlap_status"] = (
                    shape_overlap_status
                )
                record[f"{epoch_type}_odd_tuning_curve_area"] = odd_curve_area
                record[f"{epoch_type}_even_tuning_curve_area"] = even_curve_area
                record[
                    f"{epoch_type}_segment_stability_shape_overlaps"
                ] = segment_shape_overlaps
                record[
                    f"{epoch_type}_segment_shape_overlap_statuses"
                ] = segment_shape_overlap_statuses
                record[f"passes_{epoch_type}_path_rate_qc"] = bool(
                    rate_result["path_movement_firing_rate_status"] == "valid"
                    and np.isfinite(rate)
                    and rate > float(min_path_movement_firing_rate_hz)
                )
                record[f"passes_{epoch_type}_stability_qc"] = bool(
                    np.isfinite(correlation)
                    and correlation > float(min_stability_correlation)
                )
            record["passes_unit_qc"] = bool(
                record["passes_dark_epoch_rate_qc"]
                and record["passes_light_epoch_rate_qc"]
            )
            record["passes_qc"] = all(
                record[column]
                for column in (
                    "passes_dark_stability_qc",
                    "passes_light_stability_qc",
                )
            )
            record["qc_status"] = _qc_status(record)
            records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=PANEL_B_TUNING_SIMILARITY_COLUMNS,
    )


def build_panel_b_tuning_similarity_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Return all path-specific similarities with explicit eligibility flags."""
    import pandas as pd

    if min_epoch_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_epoch_movement_firing_rate_hz must be non-negative."
        )
    if min_path_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_path_movement_firing_rate_hz must be non-negative."
        )
    if min_segment_mean_firing_rate_hz < 0.0:
        raise ValueError(
            "min_segment_mean_firing_rate_hz must be non-negative."
        )
    if min_stability_correlation < -1.0:
        raise ValueError("min_stability_correlation must be at least -1.")
    tables = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        resolved_dark_epoch = (
            str(dataset_dark_epoch)
            if dark_epoch is None
            else _figure_2.get_dark_epoch(animal_name, date, dark_epoch)
        )
        resolved_light_epoch = _figure_2.get_light_epoch(
            animal_name,
            date,
            light_epoch,
        )
        tables.append(
            _build_session_tuning_similarity_table(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                dark_epoch=resolved_dark_epoch,
                light_epoch=resolved_light_epoch,
                min_epoch_movement_firing_rate_hz=(
                    min_epoch_movement_firing_rate_hz
                ),
                min_path_movement_firing_rate_hz=(
                    min_path_movement_firing_rate_hz
                ),
                min_segment_mean_firing_rate_hz=(
                    min_segment_mean_firing_rate_hz
                ),
                min_stability_correlation=min_stability_correlation,
            )
        )
    if not tables:
        return pd.DataFrame(columns=PANEL_B_TUNING_SIMILARITY_COLUMNS)
    return pd.concat(tables, ignore_index=True).loc[
        :, list(PANEL_B_TUNING_SIMILARITY_COLUMNS)
    ]


def derive_panel_b_tuning_average_table(
    path_table: Any,
    *,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return one audit-ready available-path tuning average per neuron."""
    import pandas as pd

    if min_epoch_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_epoch_movement_firing_rate_hz must be non-negative."
        )
    if path_table.empty:
        return pd.DataFrame(columns=PANEL_B_TUNING_AVERAGE_COLUMNS)
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    if path_table.duplicated([*key_columns, "path"]).any():
        raise ValueError("Panel B path table has duplicate neuron/path rows.")

    records: list[dict[str, Any]] = []
    for key, rows in path_table.groupby(list(key_columns), sort=False):
        epoch_rates: dict[str, float] = {}
        for condition in ("dark", "light"):
            column = f"{condition}_epoch_movement_firing_rate_hz"
            values = np.asarray(rows[column], dtype=float)
            finite_values = np.unique(values[np.isfinite(values)])
            if finite_values.size > 1:
                raise ValueError(
                    "Whole-epoch movement firing rate differs across Panel B "
                    f"path rows for neuron key {key!r}."
                )
            epoch_rates[condition] = (
                float(finite_values[0])
                if finite_values.size
                else float("nan")
            )
        passes_dark = bool(
            np.isfinite(epoch_rates["dark"])
            and epoch_rates["dark"]
            > float(min_epoch_movement_firing_rate_hz)
        )
        passes_light = bool(
            np.isfinite(epoch_rates["light"])
            and epoch_rates["light"]
            > float(min_epoch_movement_firing_rate_hz)
        )
        passes_unit = passes_dark and passes_light

        eligible_rows = rows[
            rows["passes_qc"].astype(bool)
            & (rows["similarity_status"].astype(str) == "valid")
        ].copy()
        eligible_rows["path"] = eligible_rows["path"].astype(str)
        eligible_rows = eligible_rows.set_index("path", drop=False)
        eligible_paths: list[str] = []
        eligible_path_segments: list[list[Any]] = []
        score_values: list[float] = []
        for path in PANEL_B_PATH_ORDER:
            if path not in eligible_rows.index:
                continue
            row = eligible_rows.loc[path]
            if "eligible_segments" not in row.index:
                score = float(row["tuning_similarity_index"])
                if np.isfinite(score):
                    eligible_paths.append(path)
                    eligible_path_segments.append([path, 0])
                    score_values.append(score)
                continue
            segments = [int(value) for value in json.loads(row["eligible_segments"])]
            scores = np.asarray(
                json.loads(row["segment_tuning_similarity_indices"]),
                dtype=float,
            )
            path_has_value = False
            for segment in segments:
                if segment < 0 or segment >= scores.size:
                    raise ValueError("Panel B eligible segment index is invalid.")
                score = float(scores[segment])
                if not np.isfinite(score):
                    raise ValueError("Panel B eligible segment score is nonfinite.")
                eligible_path_segments.append([path, segment])
                score_values.append(score)
                path_has_value = True
            if path_has_value:
                eligible_paths.append(path)
        values = np.asarray(score_values, dtype=float)
        if not passes_unit:
            average = float("nan")
            failed_conditions = [
                condition
                for condition, passes in (
                    ("dark", passes_dark),
                    ("light", passes_light),
                )
                if not passes
            ]
            average_status = "fails_" + "_and_".join(
                f"{condition}_epoch_rate" for condition in failed_conditions
            )
        elif values.size == 0:
            average = float("nan")
            average_status = "no_eligible_segments"
        else:
            average = float(np.mean(values))
            average_status = "valid"

        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "dark_epoch_movement_firing_rate_hz": epoch_rates["dark"],
                "light_epoch_movement_firing_rate_hz": epoch_rates["light"],
                "passes_dark_epoch_rate_qc": passes_dark,
                "passes_light_epoch_rate_qc": passes_light,
                "passes_unit_qc": passes_unit,
                "unit_qc_status": (
                    "valid"
                    if passes_unit
                    else average_status.removeprefix("fails_")
                ),
                "tuning_average_index": average,
                "n_eligible_paths": len(eligible_paths),
                "eligible_paths": json.dumps(eligible_paths),
                "n_eligible_segments": int(values.size),
                "eligible_path_segments": json.dumps(
                    eligible_path_segments
                ),
                "average_status": average_status,
                "cache_version": PANEL_B_TUNING_AVERAGE_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=PANEL_B_TUNING_AVERAGE_COLUMNS,
    )


def derive_segment_overlap_response_table(
    path_table: Any,
    *,
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
) -> Any:
    """Expand path rows into auditable segment overlap/response rows."""
    import pandas as pd

    if min_segment_mean_firing_rate_hz < 0.0:
        raise ValueError(
            "min_segment_mean_firing_rate_hz must be non-negative."
        )
    if path_table.empty:
        return pd.DataFrame(columns=SEGMENT_OVERLAP_RESPONSE_COLUMNS)

    records: list[dict[str, Any]] = []
    array_columns = (
        "segment_tuning_similarity_indices",
        "segment_similarity_statuses",
        "dark_segment_mean_firing_rates_hz",
        "light_segment_mean_firing_rates_hz",
        "dark_segment_tuning_curve_areas",
        "light_segment_tuning_curve_areas",
    )
    for _, row in path_table.iterrows():
        arrays = {
            column: json.loads(str(row[column])) for column in array_columns
        }
        edges = np.asarray(
            json.loads(str(row["segment_edges_normalized"])),
            dtype=float,
        )
        lengths = {len(values) for values in arrays.values()}
        if len(lengths) != 1:
            raise ValueError("Segment JSON arrays have inconsistent lengths.")
        n_segments = lengths.pop()
        if edges.shape != (n_segments + 1,):
            raise ValueError("Segment edges do not match segment JSON arrays.")
        if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("Segment edges must be finite and increasing.")

        for segment in range(n_segments):
            score = float(
                arrays["segment_tuning_similarity_indices"][segment]
            )
            score_status = str(
                arrays["segment_similarity_statuses"][segment]
            )
            dark_mean = float(
                arrays["dark_segment_mean_firing_rates_hz"][segment]
            )
            light_mean = float(
                arrays["light_segment_mean_firing_rates_hz"][segment]
            )
            dark_area = float(
                arrays["dark_segment_tuning_curve_areas"][segment]
            )
            light_area = float(
                arrays["light_segment_tuning_curve_areas"][segment]
            )
            rate_valid = np.isfinite(dark_mean) and np.isfinite(light_mean)
            passes_segment_rate = bool(
                rate_valid
                and max(dark_mean, light_mean)
                >= float(min_segment_mean_firing_rate_hz)
            )

            if (
                not np.isfinite(dark_area)
                or not np.isfinite(light_area)
                or dark_area < 0.0
                or light_area < 0.0
            ):
                ratio = float("nan")
                log2_ratio = float("nan")
                ratio_status = "invalid_area"
            elif dark_area <= _TUNING_SIMILARITY_EPS:
                if light_area <= _TUNING_SIMILARITY_EPS:
                    ratio = float("nan")
                    log2_ratio = float("nan")
                    ratio_status = "both_silent"
                else:
                    ratio = float("inf")
                    log2_ratio = float("inf")
                    ratio_status = "light_only"
            elif light_area <= _TUNING_SIMILARITY_EPS:
                ratio = 0.0
                log2_ratio = float("-inf")
                ratio_status = "dark_only"
            else:
                ratio = light_area / dark_area
                log2_ratio = float(np.log2(ratio))
                ratio_status = "finite"

            passes_unit = bool(row["passes_unit_qc"])
            passes_path = bool(row["passes_qc"])
            valid_overlap = bool(
                score_status == "valid" and np.isfinite(score)
            )
            valid_ratio = ratio_status in {"finite", "dark_only", "light_only"}
            included = bool(
                passes_unit
                and passes_path
                and passes_segment_rate
                and valid_overlap
                and valid_ratio
            )
            if included:
                inclusion_status = "valid"
            elif not passes_unit:
                inclusion_status = "fails_unit_qc"
            elif not passes_path:
                inclusion_status = "fails_path_stability_qc"
            elif not passes_segment_rate:
                inclusion_status = "below_segment_rate_threshold"
            elif not valid_overlap:
                inclusion_status = "invalid_segment_overlap"
            else:
                inclusion_status = "invalid_response_ratio"

            records.append(
                {
                    "animal_name": str(row["animal_name"]),
                    "date": str(row["date"]),
                    "region": str(row["region"]),
                    "unit": int(row["unit"]),
                    "path": str(row["path"]),
                    "dark_epoch": str(row["dark_epoch"]),
                    "light_epoch": str(row["light_epoch"]),
                    "segment_index": int(segment),
                    "segment_start_normalized": float(edges[segment]),
                    "segment_end_normalized": float(edges[segment + 1]),
                    "segment_tuning_similarity_index": score,
                    "segment_similarity_status": score_status,
                    "dark_segment_mean_firing_rate_hz": dark_mean,
                    "light_segment_mean_firing_rate_hz": light_mean,
                    "dark_segment_tuning_curve_area": dark_area,
                    "light_segment_tuning_curve_area": light_area,
                    "segment_response_ratio": ratio,
                    "log2_segment_response_ratio": log2_ratio,
                    "response_ratio_status": ratio_status,
                    "passes_unit_qc": passes_unit,
                    "passes_qc": passes_path,
                    "passes_segment_rate_qc": passes_segment_rate,
                    "included": included,
                    "inclusion_status": inclusion_status,
                    "cache_version": SEGMENT_OVERLAP_RESPONSE_CACHE_VERSION,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=SEGMENT_OVERLAP_RESPONSE_COLUMNS,
    )


def _load_segment_null_curve_lookup(
    path_table: Any,
    segment_rows: Any,
) -> dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]:
    """Load normalized dark/light curves for selected physical segments."""
    import xarray as xr
    from v1ca1.task_progression.similarity import make_segment_masks

    path_key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "path",
        "dark_epoch",
        "light_epoch",
    )
    source_rows = path_table.set_index(list(path_key_columns), drop=False)
    lookup: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    requested = segment_rows.loc[:, [*path_key_columns, "segment_index"]]
    for path_key, requested_rows in requested.groupby(
        list(path_key_columns),
        sort=True,
    ):
        if path_key not in source_rows.index:
            raise ValueError(
                "Panel I segment key is missing from the source path table."
            )
        source = source_rows.loc[path_key]
        if getattr(source, "ndim", 1) != 1:
            raise ValueError("Source path table contains duplicate path keys.")
        dark_path = Path(str(source["dark_tuning_curve_path"]))
        light_path = Path(str(source["light_tuning_curve_path"]))
        with xr.open_dataarray(dark_path) as dark_data, xr.open_dataarray(
            light_path
        ) as light_data:
            _require_aligned_tuning_curves(
                dark_data,
                light_data,
                dark_path=dark_path,
                light_path=light_path,
            )
            units = np.asarray(dark_data.coords["unit"].values, dtype=int)
            unit = int(source["unit"])
            matches = np.flatnonzero(units == unit)
            if matches.size != 1:
                raise ValueError(f"Unit {unit} is not unique in {dark_path}.")
            index = int(matches[0])
            dark_curve = np.asarray(
                dark_data.transpose("unit", ...).values[index],
                dtype=float,
            )
            light_curve = np.asarray(
                light_data.transpose("unit", ...).values[index],
                dtype=float,
            )
            position_dimension = next(
                dimension for dimension in dark_data.dims
                if dimension != "unit"
            )
            progression = np.asarray(
                dark_data.coords[position_dimension].values,
                dtype=float,
            ) / get_wtrack_total_length(str(source["animal_name"]))
        edges = np.asarray(
            json.loads(str(source["segment_edges_normalized"])),
            dtype=float,
        )
        masks = make_segment_masks(progression, edges)
        for segment in sorted(
            set(np.asarray(requested_rows["segment_index"], dtype=int))
        ):
            if segment < 0 or segment >= len(masks):
                raise ValueError("Panel I segment index is invalid.")
            mask = masks[segment]
            key = (*path_key, int(segment))
            lookup[key] = (
                _normalized_segment_curve_for_circular_null(dark_curve[mask]),
                _normalized_segment_curve_for_circular_null(light_curve[mask]),
            )
    return lookup


def derive_segment_stability_reference_table(
    path_table: Any,
    segment_table: Any,
    *,
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Return matched split, null, and achievable values for Panel-I segments."""
    import pandas as pd

    del n_permutations, random_seed
    included = segment_table[segment_table["included"].astype(bool)].copy()
    if included.empty:
        return pd.DataFrame(columns=SEGMENT_STABILITY_REFERENCE_COLUMNS)
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "path",
        "dark_epoch",
        "light_epoch",
        "segment_index",
    )
    if included.duplicated(list(key_columns)).any():
        raise ValueError("Panel I contains duplicate included segment keys.")
    path_key_columns = key_columns[:-1]
    path_lookup = path_table.set_index(list(path_key_columns), drop=False)
    path_rank = {path: index for index, path in enumerate(PANEL_B_PATH_ORDER)}
    included["_path_rank"] = included["path"].astype(str).map(path_rank)
    included = included.sort_values(
        ["animal_name", "date", "region", "unit", "_path_rank", "segment_index"],
        kind="stable",
    ).drop(columns="_path_rank")
    null_curve_rows = included[
        included["response_ratio_status"].astype(str) == "finite"
    ]
    curve_lookup = _load_segment_null_curve_lookup(
        path_table,
        null_curve_rows,
    )
    records: list[dict[str, Any]] = []
    for row in included.itertuples(index=False):
        path_key = tuple(getattr(row, column) for column in path_key_columns)
        if path_key not in path_lookup.index:
            raise ValueError("Panel I segment has no source path row.")
        source = path_lookup.loc[path_key]
        if getattr(source, "ndim", 1) != 1:
            raise ValueError("Source path table contains duplicate path keys.")
        segment = int(row.segment_index)
        dark_values = np.asarray(
            json.loads(str(source["dark_segment_stability_shape_overlaps"])),
            dtype=float,
        )
        light_values = np.asarray(
            json.loads(str(source["light_segment_stability_shape_overlaps"])),
            dtype=float,
        )
        dark_statuses = json.loads(
            str(source["dark_segment_shape_overlap_statuses"])
        )
        light_statuses = json.loads(
            str(source["light_segment_shape_overlap_statuses"])
        )
        if any(
            segment >= len(values)
            for values in (
                dark_values,
                light_values,
                dark_statuses,
                light_statuses,
            )
        ):
            raise ValueError("Segment split-half arrays do not align.")
        dark_split = float(dark_values[segment])
        light_split = float(light_values[segment])
        dark_valid = bool(
            str(dark_statuses[segment]) == "valid"
            and np.isfinite(dark_split)
        )
        light_valid = bool(
            str(light_statuses[segment]) == "valid"
            and np.isfinite(light_split)
        )
        ratio_status = str(row.response_ratio_status)
        if ratio_status == "dark_only" and dark_valid:
            split_half = dark_split
            split_status = "valid_active_condition_only"
        elif ratio_status == "light_only" and light_valid:
            split_half = light_split
            split_status = "valid_active_condition_only"
        elif ratio_status == "finite" and dark_valid and light_valid:
            split_half = float(np.mean((dark_split, light_split)))
            split_status = "valid_both_conditions"
        else:
            split_half = float("nan")
            split_status = "no_valid_condition"

        if ratio_status in {"dark_only", "light_only"}:
            minimum_null = 0.0
            n_circular_shifts = 0
            null_status = "valid_one_condition_silent"
        elif ratio_status != "finite":
            minimum_null = float("nan")
            n_circular_shifts = 0
            null_status = "invalid_response_ratio"
        else:
            curve_key = (*path_key, segment)
            dark_curve, light_curve = curve_lookup[curve_key]
            minimum_null, n_circular_shifts = (
                compute_exact_circular_shift_overlap_minimum(
                    dark_curve,
                    light_curve,
                )
            )
            null_status = "valid"

        observed = float(row.segment_tuning_similarity_index)
        denominator = split_half - minimum_null
        if not np.isfinite(observed):
            achievable = float("nan")
            achievable_status = "invalid_observed"
        elif not np.isfinite(split_half):
            achievable = float("nan")
            achievable_status = "invalid_split_half"
        elif not np.isfinite(minimum_null):
            achievable = float("nan")
            achievable_status = "invalid_null"
        elif denominator <= _TUNING_SIMILARITY_EPS:
            achievable = float("nan")
            achievable_status = "nonpositive_denominator"
        else:
            achievable = (observed - minimum_null) / denominator
            achievable_status = "valid"
        records.append(
            {
                **{column: getattr(row, column) for column in key_columns},
                "observed_segment_stability": observed,
                "dark_split_half_segment_stability": (
                    dark_split if dark_valid else float("nan")
                ),
                "light_split_half_segment_stability": (
                    light_split if light_valid else float("nan")
                ),
                "split_half_segment_stability": split_half,
                "split_half_status": split_status,
                "minimum_null_segment_stability": minimum_null,
                "null_status": null_status,
                "achievable_segment_stability": float(achievable),
                "denominator": float(denominator),
                "achievable_status": achievable_status,
                "n_circular_shifts": n_circular_shifts,
                "cache_version": SEGMENT_STABILITY_REFERENCE_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SEGMENT_STABILITY_REFERENCE_COLUMNS,
    )


def derive_segment_matched_achievable_stability_table(
    segment_reference_table: Any,
) -> Any:
    """Average coherent segment-level observed, split, and null references."""
    import pandas as pd

    if segment_reference_table.empty:
        return pd.DataFrame(
            columns=SEGMENT_MATCHED_ACHIEVABLE_STABILITY_COLUMNS
        )
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    segment_key_columns = (*key_columns, "path", "segment_index")
    if segment_reference_table.duplicated(list(segment_key_columns)).any():
        raise ValueError(
            "Segment reference table contains duplicate neuron/path/segment "
            "rows."
        )
    path_rank = {path: index for index, path in enumerate(PANEL_B_PATH_ORDER)}
    rows = segment_reference_table.copy()
    rows["_path_rank"] = rows["path"].astype(str).map(path_rank)
    if rows["_path_rank"].isna().any():
        raise ValueError("Segment reference table contains an unknown path.")
    rows = rows.sort_values(
        [*key_columns, "_path_rank", "segment_index"],
        kind="stable",
    ).drop(columns="_path_rank")

    records: list[dict[str, Any]] = []
    valid_split_statuses = {
        "valid_both_conditions",
        "valid_active_condition_only",
    }
    valid_null_statuses = {
        "valid",
        "valid_one_condition_silent",
    }
    for key, cell_rows in rows.groupby(list(key_columns), sort=True):
        observed_values = np.asarray(
            cell_rows["observed_segment_stability"], dtype=float
        )
        split_values = np.asarray(
            cell_rows["split_half_segment_stability"], dtype=float
        )
        null_values = np.asarray(
            cell_rows["minimum_null_segment_stability"], dtype=float
        )
        coherent = (
            np.isfinite(observed_values)
            & np.isfinite(split_values)
            & np.isfinite(null_values)
            & cell_rows["split_half_status"]
            .astype(str)
            .isin(valid_split_statuses)
            .to_numpy()
            & cell_rows["null_status"]
            .astype(str)
            .isin(valid_null_statuses)
            .to_numpy()
        )
        eligible = cell_rows.loc[coherent]
        eligible_observed = observed_values[coherent]
        eligible_split = split_values[coherent]
        eligible_null = null_values[coherent]
        eligible_path_segments = [
            [str(row.path), int(row.segment_index)]
            for row in eligible.itertuples(index=False)
        ]
        eligible_paths = [
            path
            for path in PANEL_B_PATH_ORDER
            if path in set(eligible["path"].astype(str))
        ]
        if eligible.empty:
            observed = float("nan")
            split_half = float("nan")
            null_minimum = float("nan")
            denominator = float("nan")
            achievable = float("nan")
            status = "no_coherent_segment_references"
        else:
            observed = float(np.mean(eligible_observed))
            split_half = float(np.mean(eligible_split))
            null_minimum = float(np.mean(eligible_null))
            denominator = split_half - null_minimum
            if denominator <= _TUNING_SIMILARITY_EPS:
                achievable = float("nan")
                status = "nonpositive_denominator"
            else:
                achievable = float(
                    (observed - null_minimum) / denominator
                )
                status = "valid"
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "observed_tuning_stability_index": observed,
                "matched_split_half_tuning_stability_index": split_half,
                "minimum_null_tuning_stability_index": null_minimum,
                "achievable_stability": achievable,
                "denominator": denominator,
                "n_source_segments": int(len(cell_rows)),
                "n_eligible_paths": len(eligible_paths),
                "eligible_paths": json.dumps(eligible_paths),
                "n_eligible_segments": int(len(eligible)),
                "eligible_path_segments": json.dumps(
                    eligible_path_segments
                ),
                "achievable_status": status,
                "n_null_shifts": int(
                    np.sum(
                        np.asarray(eligible["n_circular_shifts"], dtype=int)
                    )
                ),
                "cache_version": (
                    SEGMENT_MATCHED_ACHIEVABLE_STABILITY_CACHE_VERSION
                ),
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=SEGMENT_MATCHED_ACHIEVABLE_STABILITY_COLUMNS,
    )


def derive_panel_b_dark_split_half_table(
    path_table: Any,
    *,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return one dark odd/even overlap average per neuron."""
    import pandas as pd

    if min_epoch_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_epoch_movement_firing_rate_hz must be non-negative."
        )
    if min_path_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_path_movement_firing_rate_hz must be non-negative."
        )
    if path_table.empty:
        return pd.DataFrame(columns=PANEL_B_DARK_SPLIT_HALF_COLUMNS)
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
    )
    if path_table.duplicated([*key_columns, "path"]).any():
        raise ValueError("Panel B path table has duplicate neuron/path rows.")

    records: list[dict[str, Any]] = []
    for key, rows in path_table.groupby(list(key_columns), sort=False):
        epoch_rates = np.asarray(
            rows["dark_epoch_movement_firing_rate_hz"],
            dtype=float,
        )
        finite_epoch_rates = np.unique(epoch_rates[np.isfinite(epoch_rates)])
        if finite_epoch_rates.size > 1:
            raise ValueError(
                "Dark whole-epoch movement firing rate differs across Panel B "
                f"path rows for neuron key {key!r}."
            )
        epoch_rate = (
            float(finite_epoch_rates[0])
            if finite_epoch_rates.size
            else float("nan")
        )
        passes_epoch_rate = bool(
            np.isfinite(epoch_rate)
            and epoch_rate > float(min_epoch_movement_firing_rate_hz)
        )

        eligible_paths: list[str] = []
        overlap_values: list[float] = []
        rows_by_path = rows.set_index(rows["path"].astype(str), drop=False)
        for path in PANEL_B_PATH_ORDER:
            if path not in rows_by_path.index:
                continue
            row = rows_by_path.loc[path]
            path_rate = float(row["dark_path_movement_firing_rate_hz"])
            overlap = float(row["dark_stability_shape_overlap"])
            path_is_eligible = bool(
                str(row["dark_path_movement_firing_rate_status"]) == "valid"
                and np.isfinite(path_rate)
                and path_rate > float(min_path_movement_firing_rate_hz)
                and str(row["dark_shape_overlap_status"]) == "valid"
                and np.isfinite(overlap)
            )
            if path_is_eligible:
                eligible_paths.append(path)
                overlap_values.append(overlap)

        if not passes_epoch_rate:
            average = float("nan")
            average_status = "fails_dark_epoch_rate"
        elif not overlap_values:
            average = float("nan")
            average_status = "no_eligible_paths"
        else:
            average = float(np.mean(np.asarray(overlap_values, dtype=float)))
            average_status = "valid"
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "dark_epoch_movement_firing_rate_hz": epoch_rate,
                "passes_dark_epoch_rate_qc": passes_epoch_rate,
                "dark_split_half_tuning_stability_index": average,
                "n_eligible_paths": len(eligible_paths),
                "eligible_paths": json.dumps(eligible_paths),
                "average_status": average_status,
                "cache_version": PANEL_B_DARK_SPLIT_HALF_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=PANEL_B_DARK_SPLIT_HALF_COLUMNS,
    )


def derive_panel_b_split_half_table(
    path_table: Any,
    *,
    observed_table: Any | None = None,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return the all-four-path split-half reference for observed neurons."""
    import pandas as pd

    if min_epoch_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_epoch_movement_firing_rate_hz must be non-negative."
        )
    if min_path_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_path_movement_firing_rate_hz must be non-negative."
        )
    if path_table.empty:
        return pd.DataFrame(columns=PANEL_B_SPLIT_HALF_COLUMNS)
    if observed_table is None:
        observed_table = derive_panel_b_tuning_average_table(
            path_table,
            min_epoch_movement_firing_rate_hz=(
                min_epoch_movement_firing_rate_hz
            ),
        )
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    if path_table.duplicated([*key_columns, "path"]).any():
        raise ValueError("Panel B path table has duplicate neuron/path rows.")

    valid_observed = observed_table[
        observed_table["passes_unit_qc"].astype(bool)
        & (observed_table["average_status"].astype(str) == "valid")
        & np.isfinite(
            np.asarray(observed_table["tuning_average_index"], dtype=float)
        )
    ]
    observed_keys = {
        tuple(getattr(row, column) for column in key_columns)
        for row in valid_observed.itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    for key, rows in path_table.groupby(list(key_columns), sort=False):
        if tuple(key) not in observed_keys:
            continue
        epoch_rates: dict[str, float] = {}
        for condition in ("dark", "light"):
            column = f"{condition}_epoch_movement_firing_rate_hz"
            values = np.asarray(rows[column], dtype=float)
            finite_values = np.unique(values[np.isfinite(values)])
            if finite_values.size > 1:
                raise ValueError(
                    "Whole-epoch movement firing rate differs across Panel B "
                    f"path rows for neuron key {key!r}."
                )
            epoch_rates[condition] = (
                float(finite_values[0])
                if finite_values.size
                else float("nan")
            )
        passes_dark = bool(
            np.isfinite(epoch_rates["dark"])
            and epoch_rates["dark"]
            > float(min_epoch_movement_firing_rate_hz)
        )
        passes_light = bool(
            np.isfinite(epoch_rates["light"])
            and epoch_rates["light"]
            > float(min_epoch_movement_firing_rate_hz)
        )
        passes_unit = passes_dark and passes_light

        rows_by_path = rows.set_index(rows["path"].astype(str), drop=False)
        eligible_paths: list[str] = []
        overlap_values: list[float] = []
        for path in PANEL_B_PATH_ORDER:
            if path not in rows_by_path.index:
                continue
            row = rows_by_path.loc[path]
            dark_overlap = float(row["dark_stability_shape_overlap"])
            light_overlap = float(row["light_stability_shape_overlap"])
            path_is_eligible = bool(
                str(row["dark_shape_overlap_status"]) == "valid"
                and str(row["light_shape_overlap_status"]) == "valid"
                and np.isfinite(dark_overlap)
                and np.isfinite(light_overlap)
            )
            if path_is_eligible:
                eligible_paths.append(path)
                overlap_values.append(
                    float(np.mean((dark_overlap, light_overlap)))
                )

        if eligible_paths == list(PANEL_B_PATH_ORDER):
            average = float(np.mean(np.asarray(overlap_values, dtype=float)))
            average_status = "valid"
        else:
            average = float("nan")
            average_status = "incomplete_all_four_paths"
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "dark_epoch_movement_firing_rate_hz": epoch_rates["dark"],
                "light_epoch_movement_firing_rate_hz": epoch_rates["light"],
                "passes_dark_epoch_rate_qc": passes_dark,
                "passes_light_epoch_rate_qc": passes_light,
                "passes_unit_qc": passes_unit,
                "split_half_tuning_stability_index": average,
                "n_eligible_paths": len(eligible_paths),
                "eligible_paths": json.dumps(eligible_paths),
                "average_status": average_status,
                "cache_version": PANEL_B_SPLIT_HALF_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(records, columns=PANEL_B_SPLIT_HALF_COLUMNS)


def _normalized_curve_for_circular_null(curve: np.ndarray) -> np.ndarray:
    """Return one unit-area whole-path curve for the circular null."""
    from v1ca1.task_progression.similarity import interpolate_nans

    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).any() or np.isinf(values).any():
        raise ValueError("Circular-null tuning curves must be finite 1-D data.")
    if np.any(values[np.isfinite(values)] < -_TUNING_SIMILARITY_EPS):
        raise ValueError("Circular-null tuning curves cannot contain negatives.")
    values = np.maximum(interpolate_nans(values), 0.0)
    area = float(np.sum(values))
    if not np.isfinite(area) or area <= _TUNING_SIMILARITY_EPS:
        raise ValueError("Circular-null tuning curves must have positive area.")
    return values / area


def _normalized_segment_curve_for_circular_null(
    curve: np.ndarray,
) -> np.ndarray:
    """Return a unit-area segment curve, retaining a silent curve as zeros."""
    from v1ca1.task_progression.similarity import interpolate_nans

    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).any() or np.isinf(values).any():
        raise ValueError("Segment circular-null curves must be finite 1-D data.")
    if np.any(values[np.isfinite(values)] < -_TUNING_SIMILARITY_EPS):
        raise ValueError("Segment circular-null curves cannot contain negatives.")
    values = np.maximum(interpolate_nans(values), 0.0)
    area = float(np.sum(values))
    if not np.isfinite(area):
        raise ValueError("Segment circular-null curve area must be finite.")
    return values / area if area > _TUNING_SIMILARITY_EPS else np.zeros_like(values)


def compute_exact_circular_shift_overlap_minimum(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
) -> tuple[float, int]:
    """Return the minimum overlap across every unique integer-bin shift."""
    dark_values = np.asarray(dark_curve, dtype=float)
    light_values = np.asarray(light_curve, dtype=float)
    if (
        dark_values.ndim != 1
        or light_values.ndim != 1
        or dark_values.shape != light_values.shape
        or dark_values.size == 0
    ):
        raise ValueError("Circular-null tuning curves must be aligned 1-D data.")
    if not np.isfinite(dark_values).all() or not np.isfinite(light_values).all():
        raise ValueError("Circular-null tuning curves must be finite.")
    shift_scores = np.asarray(
        [
            np.minimum(dark_values, np.roll(light_values, shift)).sum()
            for shift in range(dark_values.size)
        ],
        dtype=float,
    )
    if not np.isfinite(shift_scores).all():
        raise ValueError("Circular-null overlap scores must be finite.")
    return float(np.min(shift_scores)), int(shift_scores.size)


def _normalized_curve_for_shift_profile(
    curve: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Return a unit-area curve and whether its whole path is silent."""
    from v1ca1.task_progression.similarity import interpolate_nans

    values = np.asarray(curve, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).any():
        raise ValueError("Shift-profile tuning curves must be finite 1-D data.")
    if np.isinf(values).any():
        raise ValueError("Shift-profile tuning curves must be finite.")
    if np.any(values[np.isfinite(values)] < -_TUNING_SIMILARITY_EPS):
        raise ValueError("Shift-profile tuning curves cannot contain negatives.")
    values = np.maximum(interpolate_nans(values), 0.0)
    area = float(values.sum())
    if not np.isfinite(area):
        raise ValueError("Shift-profile tuning-curve area must be finite.")
    silent = area <= _TUNING_SIMILARITY_EPS
    return (np.zeros_like(values) if silent else values / area), silent


def compute_circular_shift_overlap_profile(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
) -> dict[str, Any]:
    """Return exact whole-path overlap at every signed circular shift."""
    dark = np.asarray(dark_curve, dtype=float)
    light = np.asarray(light_curve, dtype=float)
    if (
        dark.ndim != 1
        or light.ndim != 1
        or dark.shape != light.shape
        or dark.size == 0
    ):
        raise ValueError("Shift-profile tuning curves must be aligned 1-D data.")
    dark, dark_silent = _normalized_curve_for_shift_profile(dark)
    light, light_silent = _normalized_curve_for_shift_profile(light)
    n_bins = int(dark.size)
    shifts = np.arange(n_bins, dtype=int)
    signed_shifts = shifts.astype(float) / float(n_bins)
    signed_shifts[signed_shifts > 0.5] -= 1.0
    order = np.argsort(signed_shifts, kind="stable")
    signed_shifts = signed_shifts[order]
    if dark_silent and light_silent:
        scores = np.full(n_bins, np.nan, dtype=float)
        status = "both_conditions_silent"
    else:
        scores = np.asarray(
            [
                np.minimum(dark, np.roll(light, int(shift))).sum()
                for shift in shifts
            ],
            dtype=float,
        )[order]
        if not np.isfinite(scores).all():
            raise ValueError("Shift-profile overlap scores must be finite.")
        scores = np.clip(scores, 0.0, 1.0)
        status = (
            "one_condition_silent"
            if dark_silent or light_silent
            else "valid"
        )
    return {
        "signed_normalized_shifts": signed_shifts,
        "overlap_scores": scores,
        "n_progression_bins": n_bins,
        "profile_status": status,
    }


def _interpolate_periodic_shift_profile(
    signed_shifts: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray = PANEL_H_SHIFT_PROFILE_GRID,
) -> np.ndarray:
    """Linearly interpolate one exact periodic profile onto a common grid."""
    shifts = np.asarray(signed_shifts, dtype=float)
    values = np.asarray(scores, dtype=float)
    target = np.asarray(grid, dtype=float)
    if (
        shifts.ndim != 1
        or values.ndim != 1
        or shifts.shape != values.shape
        or shifts.size == 0
        or not np.isfinite(shifts).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("Periodic profile interpolation requires finite 1-D data.")
    order = np.argsort(shifts, kind="stable")
    shifts = shifts[order]
    values = values[order]
    extended_shifts = np.concatenate(
        ([shifts[-1] - 1.0], shifts, [shifts[0] + 1.0])
    )
    extended_values = np.concatenate(([values[-1]], values, [values[0]]))
    return np.interp(target, extended_shifts, extended_values)


def _load_panel_h_shift_profile_curve_lookup(
    path_rows: Any,
) -> dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]:
    """Load aligned raw dark/light curves for Panel H shift profiles."""
    import xarray as xr

    lookup: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    group_columns = ["dark_tuning_curve_path", "light_tuning_curve_path"]
    for (dark_name, light_name), rows in path_rows.groupby(
        group_columns,
        sort=True,
    ):
        dark_path = Path(str(dark_name))
        light_path = Path(str(light_name))
        with xr.open_dataarray(dark_path) as dark_data, xr.open_dataarray(
            light_path
        ) as light_data:
            _require_aligned_tuning_curves(
                dark_data,
                light_data,
                dark_path=dark_path,
                light_path=light_path,
            )
            units = np.asarray(dark_data.coords["unit"].values, dtype=int)
            dark_values = np.asarray(dark_data.transpose("unit", ...).values)
            light_values = np.asarray(light_data.transpose("unit", ...).values)
        unit_to_index = {int(unit): index for index, unit in enumerate(units)}
        for row in rows.itertuples(index=False):
            unit = int(row.unit)
            if unit not in unit_to_index:
                raise ValueError(f"Unit {unit} is missing from {dark_path}.")
            key = (
                str(row.animal_name),
                str(row.date),
                str(row.region),
                unit,
                str(row.dark_epoch),
                str(row.light_epoch),
                str(row.path),
            )
            index = unit_to_index[unit]
            lookup[key] = (
                np.asarray(dark_values[index], dtype=float),
                np.asarray(light_values[index], dtype=float),
            )
    return lookup


def derive_panel_h_shift_profile_table(path_table: Any) -> Any:
    """Return dark-split-half-rescaled profiles for filtered nonsilent pairs."""
    import pandas as pd

    if path_table.empty:
        return pd.DataFrame(columns=PANEL_H_SHIFT_PROFILE_COLUMNS)
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
        "path",
    )
    path_qc_columns = (
        "passes_dark_path_rate_qc",
        "passes_light_path_rate_qc",
        "passes_dark_stability_qc",
        "passes_light_stability_qc",
    )
    required = (
        *key_columns,
        "dark_tuning_curve_path",
        "light_tuning_curve_path",
        "dark_stability_shape_overlap",
        "light_stability_shape_overlap",
        "dark_shape_overlap_status",
        "light_shape_overlap_status",
        *path_qc_columns,
    )
    missing = [column for column in required if column not in path_table]
    if missing:
        raise ValueError(f"Panel H source table is missing columns {missing!r}.")
    if path_table.duplicated(list(key_columns)).any():
        raise ValueError("Panel H source table has duplicate neuron/path rows.")
    rows = path_table[
        path_table["path"].astype(str).isin(PANEL_B_PATH_ORDER)
    ].copy()
    rows = rows[
        rows.loc[:, list(path_qc_columns)]
        .fillna(False)
        .astype(bool)
        .all(axis=1)
    ]
    path_rank = {path: index for index, path in enumerate(PANEL_B_PATH_ORDER)}
    rows["_path_rank"] = rows["path"].astype(str).map(path_rank)
    rows = rows.sort_values(
        [*key_columns[:-1], "_path_rank"],
        kind="stable",
    ).drop(columns="_path_rank")
    curve_lookup = _load_panel_h_shift_profile_curve_lookup(rows)
    records: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        key = tuple(getattr(row, column) for column in key_columns)
        lookup_key = (
            str(row.animal_name),
            str(row.date),
            str(row.region),
            int(row.unit),
            str(row.dark_epoch),
            str(row.light_epoch),
            str(row.path),
        )
        if lookup_key not in curve_lookup:
            raise ValueError(f"Panel H curve lookup is missing key {lookup_key!r}.")
        dark_curve, light_curve = curve_lookup[lookup_key]
        profile = compute_circular_shift_overlap_profile(
            dark_curve,
            light_curve,
        )
        status = str(profile["profile_status"])
        if status == "both_conditions_silent":
            continue
        exact_scores = np.asarray(profile["overlap_scores"], dtype=float)
        minimum_overlap = float(np.min(exact_scores))
        dark_split_half = float(row.dark_stability_shape_overlap)
        light_split_half = float(row.light_stability_shape_overlap)
        split_half_valid = bool(
            str(row.dark_shape_overlap_status) == "valid"
            and np.isfinite(dark_split_half)
        )
        split_half = (
            dark_split_half if split_half_valid else float("nan")
        )
        denominator = split_half - minimum_overlap
        if not split_half_valid:
            rescaling_status = "invalid_split_half"
        elif denominator <= _TUNING_SIMILARITY_EPS:
            rescaling_status = "nonpositive_denominator"
        else:
            rescaling_status = "valid"
        common_scores = _interpolate_periodic_shift_profile(
            np.asarray(profile["signed_normalized_shifts"], dtype=float),
            exact_scores,
        )
        rescaled_scores = (
            (common_scores - minimum_overlap) / denominator
            if rescaling_status == "valid"
            else np.full(common_scores.shape, np.nan, dtype=float)
        )
        identity = {
            "animal_name": str(key[0]),
            "date": str(key[1]),
            "region": str(key[2]),
            "unit": int(key[3]),
            "dark_epoch": str(key[4]),
            "light_epoch": str(key[5]),
            "path": str(key[6]),
        }
        for normalized_shift, overlap, rescaled_overlap in zip(
            PANEL_H_SHIFT_PROFILE_GRID,
            common_scores,
            rescaled_scores,
            strict=True,
        ):
            records.append(
                {
                    **identity,
                    "normalized_shift": float(normalized_shift),
                    "overlap": float(overlap),
                    "minimum_overlap": minimum_overlap,
                    "dark_split_half_overlap": dark_split_half,
                    "light_split_half_overlap": light_split_half,
                    "split_half_overlap": split_half,
                    "rescaling_denominator": denominator,
                    "rescaled_overlap": float(rescaled_overlap),
                    "rescaling_status": rescaling_status,
                    "n_progression_bins": int(
                        profile["n_progression_bins"]
                    ),
                    "profile_status": status,
                    "cache_version": PANEL_H_SHIFT_PROFILE_CACHE_VERSION,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=PANEL_H_SHIFT_PROFILE_COLUMNS,
    )


def _load_panel_b_null_curve_lookup(
    path_rows: Any,
) -> dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]:
    """Load aligned unit-area dark/light whole-path curves."""
    import xarray as xr

    lookup: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    group_columns = ["dark_tuning_curve_path", "light_tuning_curve_path"]
    for (dark_name, light_name), rows in path_rows.groupby(
        group_columns,
        sort=True,
    ):
        dark_path = Path(str(dark_name))
        light_path = Path(str(light_name))
        with xr.open_dataarray(dark_path) as dark_data, xr.open_dataarray(
            light_path
        ) as light_data:
            _require_aligned_tuning_curves(
                dark_data,
                light_data,
                dark_path=dark_path,
                light_path=light_path,
            )
            dark_units = np.asarray(dark_data.coords["unit"].values, dtype=int)
            dark_values = np.asarray(dark_data.transpose("unit", ...).values)
            light_values = np.asarray(light_data.transpose("unit", ...).values)
        unit_to_index = {int(unit): index for index, unit in enumerate(dark_units)}
        for row in rows.itertuples(index=False):
            unit = int(row.unit)
            if unit not in unit_to_index:
                raise ValueError(f"Unit {unit} is missing from {dark_path}.")
            index = unit_to_index[unit]
            key = (
                str(row.animal_name),
                str(row.date),
                str(row.region),
                unit,
                str(row.dark_epoch),
                str(row.light_epoch),
                str(row.path),
            )
            lookup[key] = (
                _normalized_curve_for_circular_null(dark_values[index]),
                _normalized_curve_for_circular_null(light_values[index]),
            )
    return lookup


def derive_panel_b_circular_null_table(
    path_table: Any,
    *,
    observed_table: Any | None = None,
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Return exact all-shift null minima for observed Panel B neurons."""
    import pandas as pd

    del n_permutations, random_seed
    if path_table.empty:
        return pd.DataFrame(columns=PANEL_B_CIRCULAR_NULL_COLUMNS)
    if observed_table is None:
        observed_table = derive_panel_b_tuning_average_table(path_table)
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    if path_table.duplicated([*key_columns, "path"]).any():
        raise ValueError("Panel B path table has duplicate neuron/path rows.")

    valid_observed = observed_table[
        observed_table["passes_unit_qc"].astype(bool)
        & (observed_table["average_status"].astype(str) == "valid")
        & np.isfinite(
            np.asarray(observed_table["tuning_average_index"], dtype=float)
        )
    ].copy()
    observed_keys = {
        tuple(getattr(row, column) for column in key_columns)
        for row in valid_observed.itertuples(index=False)
    }
    eligible = path_table[
        [
            tuple(getattr(row, column) for column in key_columns)
            in observed_keys
            and str(row.path) in PANEL_B_PATH_ORDER
            for row in path_table.itertuples(index=False)
        ]
    ].copy()
    path_rank = {path: index for index, path in enumerate(PANEL_B_PATH_ORDER)}
    eligible["_path_rank"] = eligible["path"].astype(str).map(path_rank)
    eligible = eligible.sort_values(
        [*key_columns, "_path_rank", "path"],
        kind="stable",
    ).drop(columns="_path_rank")
    curve_lookup = _load_panel_b_null_curve_lookup(eligible)
    minima_by_key: dict[tuple[Any, ...], list[float]] = {}
    shifts_by_key: dict[tuple[Any, ...], list[int]] = {}
    paths_by_key: dict[tuple[Any, ...], list[str]] = {}
    for row in eligible.itertuples(index=False):
        key = tuple(getattr(row, column) for column in key_columns)
        curve_key = (*key, str(row.path))
        dark_curve, light_curve = curve_lookup[curve_key]
        if dark_curve.shape != light_curve.shape:
            raise ValueError("Circular-null tuning curves must be aligned.")
        minimum, n_shifts = compute_exact_circular_shift_overlap_minimum(
            dark_curve,
            light_curve,
        )
        minima_by_key.setdefault(key, []).append(minimum)
        shifts_by_key.setdefault(key, []).append(n_shifts)
        paths_by_key.setdefault(key, []).append(str(row.path))

    records: list[dict[str, Any]] = []
    grouped = eligible.sort_values(list(key_columns), kind="stable").groupby(
        list(key_columns),
        sort=True,
    )
    for key, rows in grouped:
        passes_dark = bool(rows["passes_dark_epoch_rate_qc"].astype(bool).all())
        passes_light = bool(rows["passes_light_epoch_rate_qc"].astype(bool).all())
        passes_unit = passes_dark and passes_light
        path_minima = minima_by_key.get(tuple(key), [])
        shift_counts = shifts_by_key.get(tuple(key), [])
        eligible_paths = paths_by_key.get(tuple(key), [])
        if eligible_paths == list(PANEL_B_PATH_ORDER):
            null_minimum = float(np.mean(path_minima))
            null_status = "valid"
        else:
            null_minimum = float("nan")
            null_status = "incomplete_all_four_paths"
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "passes_dark_epoch_rate_qc": passes_dark,
                "passes_light_epoch_rate_qc": passes_light,
                "passes_unit_qc": passes_unit,
                "minimum_null_tuning_stability_index": null_minimum,
                "n_eligible_paths": len(eligible_paths),
                "eligible_paths": json.dumps(eligible_paths),
                "n_circular_shifts": int(np.sum(shift_counts)),
                "circular_shifts_per_path": json.dumps(shift_counts),
                "null_status": null_status,
                "cache_version": PANEL_B_CIRCULAR_NULL_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=PANEL_B_CIRCULAR_NULL_COLUMNS,
    )


def derive_panel_d_achievable_stability_table(
    path_table: Any,
    observed_table: Any,
    split_half_table: Any,
    null_table: Any,
) -> Any:
    """Return observed-neuron stability relative to all-four-path limits."""
    import pandas as pd

    del path_table
    if observed_table.empty:
        return pd.DataFrame(columns=PANEL_D_ACHIEVABLE_STABILITY_COLUMNS)
    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    split_lookup = {
        tuple(key): rows
        for key, rows in split_half_table.groupby(list(key_columns), sort=False)
    }
    null_lookup = {
        key: rows
        for key, rows in null_table.groupby(list(key_columns), sort=False)
    }
    records: list[dict[str, Any]] = []
    for row in observed_table.itertuples(index=False):
        key = tuple(getattr(row, column) for column in key_columns)
        observed = float(row.tuning_average_index)
        try:
            eligible_paths = list(json.loads(str(row.eligible_paths)))
        except (TypeError, ValueError, json.JSONDecodeError):
            eligible_paths = []
        try:
            eligible_path_segments = list(
                json.loads(str(row.eligible_path_segments))
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            eligible_path_segments = []
        if (
            not bool(row.passes_unit_qc)
            or str(row.average_status) != "valid"
            or not np.isfinite(observed)
        ):
            continue
        reference_paths = list(PANEL_B_PATH_ORDER)
        cell_split = split_lookup.get(key)
        split_paths_match = False
        split_half = float("nan")
        if cell_split is not None and len(cell_split) == 1:
            split_row = cell_split.iloc[0]
            try:
                split_paths = list(json.loads(str(split_row.eligible_paths)))
            except (TypeError, ValueError, json.JSONDecodeError):
                split_paths = []
            split_paths_match = bool(
                split_paths == reference_paths
                and int(split_row.n_eligible_paths) == len(reference_paths)
            )
            if str(split_row.average_status) == "valid":
                split_half = float(
                    split_row.split_half_tuning_stability_index
                )
        cell_null = null_lookup.get(key)
        null_paths_match = True
        if cell_null is None:
            null_values = np.asarray([], dtype=float)
            n_null_shifts = 0
        else:
            valid_null = cell_null[
                cell_null["null_status"].astype(str) == "valid"
            ]
            for null_row in valid_null.itertuples(index=False):
                try:
                    null_paths = list(json.loads(str(null_row.eligible_paths)))
                except (TypeError, ValueError, json.JSONDecodeError):
                    null_paths = []
                null_paths_match = null_paths_match and bool(
                    null_paths == reference_paths
                    and int(null_row.n_eligible_paths) == len(reference_paths)
                )
            null_values = np.asarray(
                valid_null["minimum_null_tuning_stability_index"],
                dtype=float,
            )
            null_values = null_values[np.isfinite(null_values)]
            n_null_shifts = int(
                np.sum(np.asarray(valid_null["n_circular_shifts"], dtype=int))
            )
        null_minimum = (
            float(null_values[0]) if null_values.size == 1 else float("nan")
        )
        denominator = split_half - null_minimum
        if not split_paths_match or not np.isfinite(split_half):
            status = "invalid_all_four_split_half"
        elif not null_paths_match:
            status = "eligible_path_mismatch"
        elif not np.isfinite(null_minimum):
            status = "invalid_null"
        elif denominator <= _TUNING_SIMILARITY_EPS:
            status = "nonpositive_denominator"
        else:
            status = "valid"
        achievable = (
            float((observed - null_minimum) / denominator)
            if status == "valid"
            else float("nan")
        )
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "observed_tuning_stability_index": observed,
                "matched_split_half_tuning_stability_index": split_half,
                "minimum_null_tuning_stability_index": null_minimum,
                "achievable_stability": achievable,
                "denominator": denominator,
                "n_eligible_paths": len(reference_paths),
                "eligible_paths": json.dumps(reference_paths),
                "n_eligible_segments": len(eligible_path_segments),
                "eligible_path_segments": json.dumps(
                    eligible_path_segments
                ),
                "achievable_status": status,
                "n_null_shifts": n_null_shifts,
                "cache_version": PANEL_D_ACHIEVABLE_STABILITY_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=PANEL_D_ACHIEVABLE_STABILITY_COLUMNS,
    )


def derive_full_path_achievable_stability_table(
    path_table: Any,
    *,
    apply_stability_filter: bool,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Return all-four-path achievable overlap using one coherent metric."""
    import pandas as pd

    if min_epoch_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_epoch_movement_firing_rate_hz must be non-negative."
        )
    if min_stability_correlation < -1.0:
        raise ValueError("min_stability_correlation must be at least -1.")
    del n_permutations, random_seed
    apply_stability_filter = bool(apply_stability_filter)
    if path_table.empty:
        return pd.DataFrame(columns=FULL_PATH_ACHIEVABLE_STABILITY_COLUMNS)

    key_columns = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    if path_table.duplicated([*key_columns, "path"]).any():
        raise ValueError("Full-path table has duplicate neuron/path rows.")
    path_rank = {path: index for index, path in enumerate(PANEL_B_PATH_ORDER)}
    rows = path_table[path_table["path"].astype(str).isin(PANEL_B_PATH_ORDER)].copy()
    rows["_path_rank"] = rows["path"].astype(str).map(path_rank)
    rows = rows.sort_values(
        [*key_columns, "_path_rank"],
        kind="stable",
    ).drop(columns="_path_rank")
    curve_source_parts: list[Any] = []
    for _key, cell_rows in rows.groupby(list(key_columns), sort=True):
        passes_rate_gate = True
        for condition in ("dark", "light"):
            values = np.asarray(
                cell_rows[f"{condition}_epoch_movement_firing_rate_hz"],
                dtype=float,
            )
            finite_values = np.unique(values[np.isfinite(values)])
            passes_rate_gate = passes_rate_gate and bool(
                finite_values.size == 1
                and float(finite_values[0])
                > float(min_epoch_movement_firing_rate_hz)
            )
        if passes_rate_gate:
            curve_source_parts.append(cell_rows)
    curve_source = (
        pd.concat(curve_source_parts, ignore_index=True)
        if curve_source_parts
        else rows.iloc[0:0].copy()
    )
    if {"dark_area", "light_area"}.issubset(curve_source.columns):
        curve_source = curve_source[
            np.isfinite(np.asarray(curve_source["dark_area"], dtype=float))
            & np.isfinite(np.asarray(curve_source["light_area"], dtype=float))
            & (
                np.asarray(curve_source["dark_area"], dtype=float)
                > _TUNING_SIMILARITY_EPS
            )
            & (
                np.asarray(curve_source["light_area"], dtype=float)
                > _TUNING_SIMILARITY_EPS
            )
        ].copy()
    curve_lookup = _load_panel_b_null_curve_lookup(curve_source)
    records: list[dict[str, Any]] = []
    for key, cell_rows in rows.groupby(list(key_columns), sort=True):
        rows_by_path = cell_rows.set_index(cell_rows["path"].astype(str), drop=False)
        epoch_rates: dict[str, float] = {}
        for condition in ("dark", "light"):
            values = np.asarray(
                cell_rows[f"{condition}_epoch_movement_firing_rate_hz"],
                dtype=float,
            )
            finite_values = np.unique(values[np.isfinite(values)])
            if finite_values.size > 1:
                raise ValueError(
                    "Whole-epoch movement rate differs across full-path rows."
                )
            epoch_rates[condition] = (
                float(finite_values[0])
                if finite_values.size
                else float("nan")
            )
        passes_unit = all(
            np.isfinite(epoch_rates[condition])
            and epoch_rates[condition]
            > float(min_epoch_movement_firing_rate_hz)
            for condition in ("dark", "light")
        )
        paths_complete = all(path in rows_by_path.index for path in PANEL_B_PATH_ORDER)
        stability_passes: list[bool] = []
        observed_by_path: list[float] = []
        split_by_path: list[float] = []
        null_by_path: list[float] = []
        shifts_by_path: list[int] = []
        reference_valid = paths_complete
        for path in PANEL_B_PATH_ORDER:
            if path not in rows_by_path.index:
                continue
            row = rows_by_path.loc[path]
            stability_passes.append(
                bool(
                    np.isfinite(float(row["dark_stability_correlation"]))
                    and float(row["dark_stability_correlation"])
                    > float(min_stability_correlation)
                    and np.isfinite(float(row["light_stability_correlation"]))
                    and float(row["light_stability_correlation"])
                    > float(min_stability_correlation)
                )
            )
            dark_split = float(row["dark_stability_shape_overlap"])
            light_split = float(row["light_stability_shape_overlap"])
            split_valid = bool(
                str(row["dark_shape_overlap_status"]) == "valid"
                and str(row["light_shape_overlap_status"]) == "valid"
                and np.isfinite(dark_split)
                and np.isfinite(light_split)
            )
            curve_key = (*tuple(key), str(path))
            if curve_key not in curve_lookup:
                reference_valid = False
                continue
            dark_curve, light_curve = curve_lookup[curve_key]
            if dark_curve.shape != light_curve.shape:
                raise ValueError("Full-path dark/light curves must be aligned.")
            observed_by_path.append(float(np.minimum(dark_curve, light_curve).sum()))
            split_by_path.append(
                float(np.mean((dark_split, light_split)))
                if split_valid
                else float("nan")
            )
            path_null, n_path_shifts = (
                compute_exact_circular_shift_overlap_minimum(
                    dark_curve,
                    light_curve,
                )
            )
            null_by_path.append(path_null)
            shifts_by_path.append(n_path_shifts)

        passes_stability = bool(
            paths_complete
            and len(stability_passes) == len(PANEL_B_PATH_ORDER)
            and all(stability_passes)
        )
        coherent_reference = bool(
            reference_valid
            and len(observed_by_path) == len(PANEL_B_PATH_ORDER)
            and len(split_by_path) == len(PANEL_B_PATH_ORDER)
            and np.isfinite(split_by_path).all()
            and len(null_by_path) == len(PANEL_B_PATH_ORDER)
        )
        observed = (
            float(np.mean(observed_by_path))
            if coherent_reference
            else float("nan")
        )
        split_half = (
            float(np.mean(split_by_path))
            if coherent_reference
            else float("nan")
        )
        null_minimum = (
            float(np.mean(null_by_path))
            if coherent_reference and np.isfinite(null_by_path).all()
            else float("nan")
        )
        denominator = split_half - null_minimum
        if not passes_unit:
            status = "fails_epoch_rate"
        elif apply_stability_filter and not passes_stability:
            status = "fails_all_four_stability_filter"
        elif not coherent_reference:
            status = "incomplete_all_four_reference"
        elif not np.isfinite(null_minimum):
            status = "invalid_null"
        elif denominator <= _TUNING_SIMILARITY_EPS:
            status = "nonpositive_denominator"
        else:
            status = "valid"
        achievable = (
            float((observed - null_minimum) / denominator)
            if status == "valid"
            else float("nan")
        )
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "dark_epoch_movement_firing_rate_hz": epoch_rates["dark"],
                "light_epoch_movement_firing_rate_hz": epoch_rates["light"],
                "passes_unit_qc": passes_unit,
                "apply_stability_filter": apply_stability_filter,
                "passes_stability_filter": passes_stability,
                "observed_tuning_stability_index": observed,
                "matched_split_half_tuning_stability_index": split_half,
                "minimum_null_tuning_stability_index": null_minimum,
                "achievable_stability": achievable,
                "denominator": denominator,
                "n_eligible_paths": (
                    len(PANEL_B_PATH_ORDER) if coherent_reference else 0
                ),
                "eligible_paths": json.dumps(
                    list(PANEL_B_PATH_ORDER) if coherent_reference else []
                ),
                "achievable_status": status,
                "n_null_shifts": (
                    int(np.sum(shifts_by_path)) if coherent_reference else 0
                ),
                "circular_shifts_per_path": json.dumps(
                    shifts_by_path if coherent_reference else []
                ),
                "cache_version": FULL_PATH_ACHIEVABLE_STABILITY_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(
        records,
        columns=FULL_PATH_ACHIEVABLE_STABILITY_COLUMNS,
    )


def _validate_tuning_correlation_variant(variant: str) -> str:
    """Return a supported dark-light Pearson analysis variant."""
    variant = str(variant)
    if variant not in TUNING_CORRELATION_VARIANTS:
        raise ValueError(
            f"variant must be one of {TUNING_CORRELATION_VARIANTS!r}."
        )
    return variant


def _correlation_key_columns() -> tuple[str, ...]:
    """Return the neuron identity shared by correlation summary tables."""
    return (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )


def _correlation_observed_paths(row: Any) -> list[str]:
    """Decode an observed summary row's ordered path provenance."""
    try:
        return [str(value) for value in json.loads(str(row.eligible_paths))]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def derive_tuning_correlation_average_table(
    path_table: Any,
    *,
    variant: str,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return one raw-r dark-light tuning average per neuron."""
    import pandas as pd

    variant = _validate_tuning_correlation_variant(variant)
    columns = TUNING_CORRELATION_AVERAGE_COLUMNS
    if path_table.empty:
        return pd.DataFrame(columns=columns)
    key_columns = _correlation_key_columns()
    if path_table.duplicated([*key_columns, "path"]).any():
        raise ValueError("Correlation path table has duplicate neuron/path rows.")
    records: list[dict[str, Any]] = []
    for key, rows in path_table.groupby(list(key_columns), sort=False):
        rates: dict[str, float] = {}
        for condition in ("dark", "light"):
            values = np.asarray(
                rows[f"{condition}_epoch_movement_firing_rate_hz"],
                dtype=float,
            )
            unique = np.unique(values[np.isfinite(values)])
            if unique.size > 1:
                raise ValueError(
                    "Epoch firing rate differs across correlation path rows."
                )
            rates[condition] = float(unique[0]) if unique.size else float("nan")
        passes_dark = bool(
            np.isfinite(rates["dark"])
            and rates["dark"] > float(min_epoch_movement_firing_rate_hz)
        )
        passes_light = bool(
            np.isfinite(rates["light"])
            and rates["light"] > float(min_epoch_movement_firing_rate_hz)
        )
        passes_unit = passes_dark and passes_light
        eligible_paths: list[str] = []
        eligible_path_segments: list[list[Any]] = []
        scores: list[float] = []
        by_path = rows.set_index(rows["path"].astype(str), drop=False)
        for path in PANEL_B_PATH_ORDER:
            if path not in by_path.index:
                continue
            row = by_path.loc[path]
            in_observed_cohort = bool(
                row["passes_qc"]
                and str(row["similarity_status"]) == "valid"
                and int(row.get("n_eligible_segments", 0)) > 0
            )
            if not in_observed_cohort:
                continue
            if variant == "whole_path":
                score = float(row["whole_path_tuning_correlation"])
                valid = bool(
                    str(row["whole_path_correlation_status"]) == "valid"
                    and np.isfinite(score)
                )
                if valid:
                    eligible_paths.append(path)
                    scores.append(score)
                continue
            segment_scores = np.asarray(
                json.loads(str(row["segment_tuning_correlations"])),
                dtype=float,
            )
            segment_indices = [
                int(value)
                for value in json.loads(
                    str(row["eligible_correlation_segments"])
                )
            ]
            path_has_score = False
            for segment in segment_indices:
                if segment < 0 or segment >= segment_scores.size:
                    raise ValueError("Eligible correlation segment is invalid.")
                score = float(segment_scores[segment])
                if not np.isfinite(score):
                    raise ValueError("Eligible correlation score is nonfinite.")
                scores.append(score)
                eligible_path_segments.append([path, segment])
                path_has_score = True
            if path_has_score:
                eligible_paths.append(path)
        if not passes_unit:
            failed = [
                condition
                for condition, passed in (
                    ("dark", passes_dark),
                    ("light", passes_light),
                )
                if not passed
            ]
            mean_score = float("nan")
            status = "fails_" + "_and_".join(
                f"{condition}_epoch_rate" for condition in failed
            )
        elif not scores:
            mean_score = float("nan")
            status = (
                "no_valid_correlation_segments"
                if variant == "physical_segments"
                else "no_valid_correlation_paths"
            )
        else:
            mean_score = float(np.mean(np.asarray(scores, dtype=float)))
            status = "valid"
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "dark_epoch_movement_firing_rate_hz": rates["dark"],
                "light_epoch_movement_firing_rate_hz": rates["light"],
                "passes_dark_epoch_rate_qc": passes_dark,
                "passes_light_epoch_rate_qc": passes_light,
                "passes_unit_qc": passes_unit,
                "mean_tuning_correlation": mean_score,
                "n_eligible_paths": len(eligible_paths),
                "eligible_paths": json.dumps(eligible_paths),
                "n_eligible_path_segments": len(eligible_path_segments),
                "eligible_path_segments": json.dumps(
                    eligible_path_segments
                ),
                "average_status": status,
                "variant": variant,
                "cache_version": TUNING_CORRELATION_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def derive_tuning_correlation_split_half_table(
    path_table: Any,
    observed_table: Any,
    *,
    variant: str,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
) -> Any:
    """Return the broad whole-path odd/even Pearson reference cohort."""
    import pandas as pd

    del observed_table
    variant = _validate_tuning_correlation_variant(variant)
    key_columns = _correlation_key_columns()
    columns = TUNING_CORRELATION_SPLIT_HALF_COLUMNS
    if path_table.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for key, rows in path_table.groupby(list(key_columns), sort=False):
        passes_dark = bool(rows["passes_dark_epoch_rate_qc"].astype(bool).all())
        passes_light = bool(
            rows["passes_light_epoch_rate_qc"].astype(bool).all()
        )
        passes_unit = passes_dark and passes_light
        by_path = rows.set_index(rows["path"].astype(str), drop=False)
        paths: list[str] = []
        values: list[float] = []
        for path in PANEL_B_PATH_ORDER:
            if path not in by_path.index:
                continue
            row = by_path.loc[path]
            dark_rate = float(row["dark_path_movement_firing_rate_hz"])
            light_rate = float(row["light_path_movement_firing_rate_hz"])
            dark_r = float(row["dark_stability_correlation"])
            light_r = float(row["light_stability_correlation"])
            valid = bool(
                str(row["dark_path_movement_firing_rate_status"]) == "valid"
                and str(row["light_path_movement_firing_rate_status"])
                == "valid"
                and np.isfinite(dark_rate)
                and np.isfinite(light_rate)
                and dark_rate > float(min_path_movement_firing_rate_hz)
                and light_rate > float(min_path_movement_firing_rate_hz)
                and np.isfinite(dark_r)
                and np.isfinite(light_r)
            )
            if valid:
                paths.append(path)
                values.append(float(np.mean((dark_r, light_r))))
        if not passes_unit:
            score = float("nan")
            status = "fails_unit_qc"
        elif not values:
            score = float("nan")
            status = "no_eligible_paths"
        else:
            score = float(np.mean(np.asarray(values, dtype=float)))
            status = "valid"
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "passes_dark_epoch_rate_qc": passes_dark,
                "passes_light_epoch_rate_qc": passes_light,
                "passes_unit_qc": passes_unit,
                "split_half_tuning_correlation": score,
                "n_eligible_paths": len(paths),
                "eligible_paths": json.dumps(paths),
                "reference_status": status,
                "variant": variant,
                "cache_version": TUNING_CORRELATION_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _correlation_for_rolls(
    dark_curve: np.ndarray,
    light_curve: np.ndarray,
) -> np.ndarray:
    """Return Pearson r for every circular shift of one aligned curve."""
    from v1ca1.task_progression.similarity import interpolate_nans

    dark = np.asarray(interpolate_nans(dark_curve), dtype=float)
    light = np.asarray(interpolate_nans(light_curve), dtype=float)
    if dark.ndim != 1 or dark.shape != light.shape:
        raise ValueError("Circular-null correlation curves must be aligned.")
    if np.std(dark) <= _TUNING_SIMILARITY_EPS:
        raise ValueError("Circular-null dark curve must not be constant.")
    if np.std(light) <= _TUNING_SIMILARITY_EPS:
        raise ValueError("Circular-null light curve must not be constant.")
    values = np.asarray(
        [
            np.corrcoef(dark, np.roll(light, shift))[0, 1]
            for shift in range(dark.size)
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ValueError("Circular-null Pearson correlation is nonfinite.")
    return np.clip(values, -1.0, 1.0)


def derive_tuning_correlation_circular_null_table(
    path_table: Any,
    observed_table: Any,
    *,
    variant: str,
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Return whole-path circular-shift Pearson nulls on observed paths."""
    import pandas as pd

    variant = _validate_tuning_correlation_variant(variant)
    if (
        isinstance(n_permutations, (bool, np.bool_))
        or not isinstance(n_permutations, (int, np.integer))
        or int(n_permutations) <= 0
    ):
        raise ValueError("n_permutations must be a positive integer.")
    n_permutations = int(n_permutations)
    key_columns = _correlation_key_columns()
    columns = TUNING_CORRELATION_NULL_COLUMNS
    if observed_table.empty:
        return pd.DataFrame(columns=columns)
    valid_observed = observed_table[
        observed_table["passes_unit_qc"].astype(bool)
        & (observed_table["average_status"].astype(str) == "valid")
    ]
    path_lookup = {
        tuple(key): rows.set_index(rows["path"].astype(str), drop=False)
        for key, rows in path_table.groupby(list(key_columns), sort=False)
    }
    selected_rows = []
    for observed in valid_observed.itertuples(index=False):
        key = tuple(getattr(observed, column) for column in key_columns)
        by_path = path_lookup.get(key)
        if by_path is None:
            continue
        for path in _correlation_observed_paths(observed):
            if path not in by_path.index:
                raise ValueError("Observed correlation path is missing.")
            selected_rows.append(by_path.loc[path])
    if selected_rows:
        selected = pd.DataFrame(selected_rows).reset_index(drop=True)
        curve_lookup = _load_panel_b_null_curve_lookup(selected)
    else:
        curve_lookup = {}
    rng = np.random.default_rng(int(random_seed))
    records: list[dict[str, Any]] = []
    for observed in observed_table.itertuples(index=False):
        key = tuple(getattr(observed, column) for column in key_columns)
        paths = _correlation_observed_paths(observed)
        path_nulls: list[np.ndarray] = []
        status = "valid"
        if str(observed.average_status) != "valid" or not paths:
            status = "invalid_observed"
        else:
            for path in paths:
                curve_key = (*key, path)
                if curve_key not in curve_lookup:
                    status = "missing_curve"
                    break
                dark_curve, light_curve = curve_lookup[curve_key]
                scores = _correlation_for_rolls(dark_curve, light_curve)
                shifts = rng.integers(
                    0,
                    scores.size,
                    size=n_permutations,
                )
                path_nulls.append(scores[shifts])
        values = (
            np.mean(np.vstack(path_nulls), axis=0)
            if status == "valid" and path_nulls
            else np.full(n_permutations, np.nan)
        )
        for permutation, value in enumerate(values):
            records.append(
                {
                    **dict(zip(key_columns, key, strict=True)),
                    "passes_unit_qc": bool(observed.passes_unit_qc),
                    "permutation": permutation,
                    "null_tuning_correlation": float(value),
                    "n_eligible_paths": len(paths),
                    "eligible_paths": json.dumps(paths),
                    "null_status": status,
                    "n_permutations": n_permutations,
                    "random_seed": int(random_seed),
                    "variant": variant,
                    "cache_version": TUNING_CORRELATION_CACHE_VERSION,
                }
            )
    return pd.DataFrame.from_records(records, columns=columns)


def derive_achievable_tuning_correlation_table(
    path_table: Any,
    observed_table: Any,
    split_half_table: Any,
    null_table: Any,
    *,
    variant: str,
) -> Any:
    """Return Pearson stability relative to matched null and split-half."""
    import pandas as pd

    del split_half_table
    variant = _validate_tuning_correlation_variant(variant)
    key_columns = _correlation_key_columns()
    columns = ACHIEVABLE_TUNING_CORRELATION_COLUMNS
    if observed_table.empty:
        return pd.DataFrame(columns=columns)
    path_lookup = {
        tuple(key): rows.set_index(rows["path"].astype(str), drop=False)
        for key, rows in path_table.groupby(list(key_columns), sort=False)
    }
    null_lookup = {
        tuple(key): rows
        for key, rows in null_table.groupby(list(key_columns), sort=False)
    }
    records: list[dict[str, Any]] = []
    for observed in observed_table.itertuples(index=False):
        key = tuple(getattr(observed, column) for column in key_columns)
        paths = _correlation_observed_paths(observed)
        matched_values: list[float] = []
        by_path = path_lookup.get(key)
        if by_path is not None:
            for path in paths:
                if path not in by_path.index:
                    continue
                row = by_path.loc[path]
                dark_r = float(row["dark_stability_correlation"])
                light_r = float(row["light_stability_correlation"])
                if np.isfinite(dark_r) and np.isfinite(light_r):
                    matched_values.append(float(np.mean((dark_r, light_r))))
        split_half = (
            float(np.mean(matched_values))
            if paths and len(matched_values) == len(paths)
            else float("nan")
        )
        cell_null = null_lookup.get(key)
        null_paths_match = True
        if cell_null is None:
            null_values = np.asarray([], dtype=float)
        else:
            valid_null = cell_null[
                cell_null["null_status"].astype(str) == "valid"
            ]
            for null_row in valid_null.itertuples(index=False):
                null_paths_match = null_paths_match and bool(
                    _correlation_observed_paths(null_row) == paths
                )
            null_values = np.asarray(
                valid_null["null_tuning_correlation"],
                dtype=float,
            )
            null_values = null_values[np.isfinite(null_values)]
        null_mean = (
            float(np.mean(null_values)) if null_values.size else float("nan")
        )
        observed_value = float(observed.mean_tuning_correlation)
        denominator = split_half - null_mean
        if (
            str(observed.average_status) != "valid"
            or not np.isfinite(observed_value)
        ):
            status = "invalid_observed"
        elif not np.isfinite(split_half):
            status = "invalid_matched_split_half"
        elif not null_paths_match:
            status = "eligible_path_mismatch"
        elif not np.isfinite(null_mean):
            status = "invalid_null"
        elif denominator <= _TUNING_SIMILARITY_EPS:
            status = "nonpositive_denominator"
        else:
            status = "valid"
        achievable = (
            float((observed_value - null_mean) / denominator)
            if status == "valid"
            else float("nan")
        )
        records.append(
            {
                **dict(zip(key_columns, key, strict=True)),
                "observed_tuning_correlation": observed_value,
                "matched_split_half_tuning_correlation": split_half,
                "mean_null_tuning_correlation": null_mean,
                "achievable_tuning_correlation": achievable,
                "denominator": denominator,
                "n_eligible_paths": len(paths),
                "eligible_paths": json.dumps(paths),
                "achievable_status": status,
                "n_null_permutations": int(null_values.size),
                "variant": variant,
                "cache_version": TUNING_CORRELATION_CACHE_VERSION,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _artifact_fingerprint(path: Path) -> dict[str, Any]:
    """Return a lightweight source fingerprint for cache validation."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def build_panel_b_tuning_similarity_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_epoch_movement_firing_rate_hz: float,
    min_path_movement_firing_rate_hz: float,
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float,
) -> dict[str, Any]:
    """Return parameters and source fingerprints identifying the TSI cache."""
    dataset_metadata = []
    source_paths: list[Path] = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        resolved_dark_epoch = (
            str(dataset_dark_epoch)
            if dark_epoch is None
            else _figure_2.get_dark_epoch(animal_name, date, dark_epoch)
        )
        resolved_light_epoch = _figure_2.get_light_epoch(
            animal_name,
            date,
            light_epoch,
        )
        dataset_metadata.append(
            {
                "animal_name": str(animal_name),
                "date": str(date),
                "dark_epoch": str(resolved_dark_epoch),
                "light_epoch": str(resolved_light_epoch),
            }
        )
        source_paths.append(get_stability_table_path(data_root, animal_name, date))
        for path_name in PANEL_B_PATH_ORDER:
            for epoch in (resolved_dark_epoch, resolved_light_epoch):
                source_paths.append(
                    _figure_2.get_compute_tuning_curve_path(
                        data_root,
                        animal_name=animal_name,
                        date=date,
                        region=region,
                        epoch=epoch,
                        trajectory=path_name,
                    )
                )
    return {
        "cache_version": PANEL_B_TUNING_SIMILARITY_CACHE_VERSION,
        "figure": LEGACY_CACHE_FIGURE_NAME,
        "panel": "B",
        "artifact": "path_tuning_similarity",
        "metric": "segmentwise_unit_area_minimum_overlap",
        "segment_count": 3,
        "segment_definition": "geometry_derived_physical_wtrack_segments",
        "segment_bin_assignment": "left_closed_right_open_final_right_closed",
        "segment_rate_gate": (
            "dark_or_light_segment_bin_mean_strictly_above_"
            "threshold"
        ),
        "min_segment_mean_firing_rate_hz": float(
            min_segment_mean_firing_rate_hz
        ),
        "averaging": "equal_mean_across_eligible_path_segments",
        "nan_handling": "linear_interpolation_with_endpoint_extension",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "datasets": dataset_metadata,
        "path_order": list(PANEL_B_PATH_ORDER),
        "min_epoch_movement_firing_rate_hz": float(
            min_epoch_movement_firing_rate_hz
        ),
        "epoch_movement_firing_rate_source": (
            "odd_even_task_progression_stability.firing_rate_hz"
        ),
        "threshold_comparison": "strict_greater_than",
        "min_path_movement_firing_rate_hz": float(
            min_path_movement_firing_rate_hz
        ),
        "min_stability_correlation": float(min_stability_correlation),
        "columns": list(PANEL_B_TUNING_SIMILARITY_COLUMNS),
        "sources": [
            _artifact_fingerprint(path)
            for path in sorted(set(source_paths), key=lambda value: str(value))
        ],
    }


def build_panel_b_tuning_similarity_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return a versioned, metadata-addressed Panel B Parquet path."""
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    region = "".join(
        character if character.isalnum() else "_"
        for character in str(metadata["region"])
    ).strip("_")
    filename = (
        f"{PANEL_B_TUNING_SIMILARITY_CACHE_PREFIX}_{region or 'none'}_"
        f"cachev{int(metadata['cache_version'])}_{digest}.parquet"
    )
    return Path(cache_dir) / filename


def build_panel_b_tuning_average_cache_metadata(
    path_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return metadata for the per-neuron available-path average table."""
    return {
        **path_metadata,
        "cache_version": PANEL_B_TUNING_AVERAGE_CACHE_VERSION,
        "source_path_cache_version": path_metadata["cache_version"],
        "artifact": "per_neuron_available_segment_tuning_average",
        "metric": "equal_mean_of_eligible_path_segment_stability_indices",
        "unit_inclusion": (
            "whole_epoch_movement_rate_strictly_above_threshold_in_both_epochs"
        ),
        "path_inclusion": (
            "odd_even_correlation_strictly_above_threshold_in_both_epochs"
        ),
        "segment_inclusion": (
            "dark_or_light_segment_mean_rate_strictly_above_threshold"
        ),
        "averaging": "flat_equal_mean_across_eligible_path_segments",
        "zero_eligible_segment_policy": (
            "retain_row_with_nan_average_and_status"
        ),
        "columns": list(PANEL_B_TUNING_AVERAGE_COLUMNS),
    }


def build_panel_b_dark_split_half_cache_metadata(
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return metadata for the dark odd/even overlap-average table."""
    return {
        **source_metadata,
        "cache_version": PANEL_B_DARK_SPLIT_HALF_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "artifact": "per_neuron_dark_split_half_tuning_stability_average",
        "metric": "unit_area_minimum_overlap",
        "condition": "dark",
        "unit_inclusion": (
            "dark_whole_epoch_movement_rate_strictly_above_threshold"
        ),
        "path_inclusion": (
            "dark_path_rate_strictly_above_threshold_and_valid_shape_overlap"
        ),
        "odd_even_correlation_filter_applied": False,
        "light_filter_applied": False,
        "averaging": "equal_mean_across_eligible_paths",
        "zero_eligible_path_policy": "retain_row_with_nan_average_and_status",
        "columns": list(PANEL_B_DARK_SPLIT_HALF_COLUMNS),
    }


def build_panel_b_split_half_cache_metadata(
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return metadata for the whole-path odd/even overlap reference."""
    return {
        **source_metadata,
        "cache_version": PANEL_B_SPLIT_HALF_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "observed_average_cache_version": PANEL_B_TUNING_AVERAGE_CACHE_VERSION,
        "artifact": "per_neuron_dark_light_split_half_average",
        "metric": "whole_path_unit_area_minimum_overlap",
        "conditions": ["dark", "light"],
        "unit_inclusion": (
            "exact_neurons_with_valid_observed_dark_light_segment_average"
        ),
        "path_inclusion": (
            "require_valid_dark_and_light_whole_path_shape_overlap_for_each_"
            "of_the_four_task_paths_without_a_path_rate_gate"
        ),
        "odd_even_correlation_filter_applied": False,
        "averaging": (
            "equal_dark_light_mean_within_path_then_equal_mean_across_all_"
            "four_task_paths"
        ),
        "incomplete_reference_policy": (
            "retain_observed_neuron_row_with_nan_average_and_status"
        ),
        "columns": list(PANEL_B_SPLIT_HALF_COLUMNS),
    }


def build_panel_b_circular_null_cache_metadata(
    source_metadata: dict[str, Any],
    *,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Return metadata for the exact circular-shift minimum reference."""
    del n_permutations, random_seed
    return {
        **source_metadata,
        "cache_version": PANEL_B_CIRCULAR_NULL_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "observed_average_cache_version": PANEL_B_TUNING_AVERAGE_CACHE_VERSION,
        "artifact": "per_neuron_circular_shift_null",
        "metric": "whole_path_unit_area_minimum_overlap",
        "null_operation": (
            "hold_dark_fixed_and_circularly_roll_whole_light_path_then_"
            "compute_whole_path_unit_area_overlap"
        ),
        "shift_support": (
            "every_integer_bin_shift_from_zero_through_n_bins_minus_one_"
            "exactly_once_per_path"
        ),
        "unit_inclusion": (
            "exact_neurons_with_valid_observed_dark_light_segment_average"
        ),
        "path_inclusion": (
            "all_four_task_paths_with_positive_aligned_whole_path_curves_"
            "without_a_path_rate_or_observed_path_gate"
        ),
        "null_summary": (
            "minimum_across_shifts_per_path_then_equal_mean_across_all_four_"
            "paths"
        ),
        "averaging": (
            "minimum_across_all_shifts_within_path_then_equal_mean_across_"
            "all_four_task_paths"
        ),
        "columns": list(PANEL_B_CIRCULAR_NULL_COLUMNS),
    }


def build_panel_d_achievable_stability_cache_metadata(
    source_metadata: dict[str, Any],
    *,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Return metadata for the per-neuron achievable-stability table."""
    del n_permutations, random_seed
    return {
        **source_metadata,
        "cache_version": PANEL_D_ACHIEVABLE_STABILITY_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "observed_average_cache_version": PANEL_B_TUNING_AVERAGE_CACHE_VERSION,
        "split_half_cache_version": PANEL_B_SPLIT_HALF_CACHE_VERSION,
        "circular_null_cache_version": PANEL_B_CIRCULAR_NULL_CACHE_VERSION,
        "artifact": "per_neuron_achievable_tuning_stability",
        "formula": (
            "(observed_minus_null_minimum)/(split_half_minus_null_minimum)"
        ),
        "observed_metric": "segmentwise_unit_area_minimum_overlap",
        "split_half_metric": "whole_path_unit_area_minimum_overlap",
        "null_metric": "whole_path_unit_area_minimum_overlap",
        "interpretation": "cross_metric_normalized_stability_index",
        "split_half_path_policy": (
            "consume_panel_b_split_half_mean_over_all_four_task_paths"
        ),
        "null_path_policy": (
            "consume_panel_b_exact_circular_shift_minimum_over_all_four_task_"
            "paths"
        ),
        "unit_inclusion": (
            "exact_neurons_with_valid_observed_dark_light_segment_average"
        ),
        "null_summary": (
            "one_exact_per_cell_mean_of_four_pathwise_shift_minima"
        ),
        "path_provenance_columns": {
            "eligible_paths": "the_four_whole_paths_used_by_both_references",
            "eligible_path_segments": (
                "the_observed_path_segment_elements_used_by_the_dark_light_"
                "numerator"
            ),
        },
        "denominator_policy": "valid_only_when_strictly_greater_than_1e-12",
        "clipping": "none",
        "columns": list(PANEL_D_ACHIEVABLE_STABILITY_COLUMNS),
    }


def build_full_path_achievable_stability_cache_metadata(
    source_metadata: dict[str, Any],
    *,
    apply_stability_filter: bool,
    min_epoch_movement_firing_rate_hz: float,
    min_stability_correlation: float,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Return metadata for the coherent whole-path achievable analysis."""
    del n_permutations, random_seed
    return {
        **source_metadata,
        "cache_version": FULL_PATH_ACHIEVABLE_STABILITY_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "artifact": "per_neuron_full_path_achievable_tuning_stability",
        "metric": "whole_path_unit_area_minimum_overlap",
        "formula": (
            "(observed_minus_null_minimum)/(split_half_minus_null_minimum)"
        ),
        "unit_inclusion": (
            "whole_epoch_movement_rate_strictly_above_threshold_in_both_epochs"
        ),
        "min_epoch_movement_firing_rate_hz": float(
            min_epoch_movement_firing_rate_hz
        ),
        "apply_stability_filter": bool(apply_stability_filter),
        "stability_filter": (
            "odd_even_pearson_r_strictly_above_threshold_in_both_epochs_"
            "on_each_of_all_four_paths"
            if apply_stability_filter
            else "none"
        ),
        "min_stability_correlation": float(min_stability_correlation),
        "path_inclusion": (
            "all_four_task_paths_without_a_path_firing_rate_gate"
        ),
        "observed_metric": "whole_path_unit_area_minimum_overlap",
        "split_half_metric": "whole_path_unit_area_minimum_overlap",
        "null_metric": "whole_path_unit_area_minimum_overlap",
        "averaging": "equal_mean_across_all_four_task_paths",
        "null_operation": (
            "enumerate_every_unique_integer_bin_roll_of_each_whole_light_path"
        ),
        "null_summary": (
            "minimum_across_all_integer_bin_shifts_per_path_then_equal_mean_"
            "across_all_four_paths"
        ),
        "denominator_policy": "valid_only_when_strictly_greater_than_1e-12",
        "clipping": "none",
        "columns": list(FULL_PATH_ACHIEVABLE_STABILITY_COLUMNS),
    }


def build_panel_h_shift_profile_cache_metadata(
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return metadata for common-grid whole-path shift profiles."""
    return {
        **source_metadata,
        "cache_version": PANEL_H_SHIFT_PROFILE_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "panel": "B",
        "artifact": (
            "whole_path_dark_split_half_rescaled_circular_shift_overlap_"
            "profile"
        ),
        "metric": "whole_path_dark_split_half_rescaled_unit_area_overlap",
        "shift_operation": "hold_dark_fixed_and_circularly_roll_light",
        "exact_shift_support": (
            "every_integer_bin_shift_from_zero_through_n_bins_minus_one"
        ),
        "exact_shift_coordinate": (
            "signed_circular_lag_in_path_fractions_on_minus_half_to_half"
        ),
        "common_grid": PANEL_H_SHIFT_PROFILE_GRID.tolist(),
        "grid_alignment": (
            "periodic_linear_interpolation_of_each_exact_profile_without_"
            "smoothing"
        ),
        "minimum_reference": (
            "minimum_overlap_across_all_exact_native_integer_bin_shifts"
        ),
        "minimum_reference_timing": (
            "compute_from_exact_native_shift_support_before_common_grid_"
            "interpolation"
        ),
        "split_half_reference": (
            "dark_whole_path_odd_even_unit_area_overlap"
        ),
        "rescaling": (
            "(overlap_minus_minimum_overlap)/(split_half_overlap_minus_"
            "minimum_overlap)"
        ),
        "denominator_policy": "valid_only_when_strictly_greater_than_1e-12",
        "clipping": "none",
        "invalid_rescaling_policy": (
            "retain_rows_with_status_and_nan_rescaled_overlap_but_exclude_"
            "from_display"
        ),
        "unit_path_inclusion": (
            "both_dark_and_light_path_movement_firing_rate_strictly_above_"
            "threshold_and_both_dark_and_light_odd_even_correlation_"
            "strictly_above_threshold"
        ),
        "path_firing_rate_filter": (
            "passes_dark_path_rate_qc_and_passes_light_path_rate_qc"
        ),
        "min_path_movement_firing_rate_hz": float(
            source_metadata.get(
                "min_path_movement_firing_rate_hz",
                _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ,
            )
        ),
        "path_stability_filter": (
            "passes_dark_stability_qc_and_passes_light_stability_qc"
        ),
        "min_stability_correlation": float(
            source_metadata.get(
                "min_stability_correlation",
                _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
            )
        ),
        "whole_epoch_rate_filter": "none",
        "post_filter_exclusion": "exclude_pairs_silent_in_both_conditions",
        "one_condition_silent_policy": "retain_as_zero_overlap_profile",
        "population_weighting": (
            "equal_mean_across_valid_paths_within_neuron_then_equal_weight_"
            "per_neuron"
        ),
        "display_summary": (
            "mean_and_interquartile_range_across_neuron_level_profiles_at_"
            "each_shift_with_zero_shift_median_and_interquartile_range"
        ),
        "columns": list(PANEL_H_SHIFT_PROFILE_COLUMNS),
    }


def build_segment_overlap_response_cache_metadata(
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return metadata for the cell/path/segment response table."""
    return {
        **source_metadata,
        "cache_version": SEGMENT_OVERLAP_RESPONSE_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "artifact": "segment_overlap_response",
        "metric": "segmentwise_unit_area_minimum_overlap",
        "row_granularity": "all_cell_path_segment_rows_with_audit_status",
        "unit_inclusion": "same_as_panel_b_both_epoch_movement_rate_gate",
        "path_inclusion": "same_as_panel_b_both_epoch_odd_even_stability_gate",
        "segment_rate_exclusion": (
            "exclude_only_when_both_dark_and_light_segment_mean_rates_are_"
            "strictly_below_threshold"
        ),
        "min_segment_mean_firing_rate_hz": float(
            source_metadata["min_segment_mean_firing_rate_hz"]
        ),
        "response_ratio": "light_segment_area/dark_segment_area",
        "response_ratio_statuses": [
            "finite",
            "dark_only",
            "light_only",
            "both_silent",
        ],
        "x_transform": "log2_finite_response_ratio",
        "one_sided_display": "explicit_capped_edge_markers",
        "averaging": "none",
        "columns": list(SEGMENT_OVERLAP_RESPONSE_COLUMNS),
    }


def build_segment_stability_reference_cache_metadata(
    source_metadata: dict[str, Any],
    *,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Return metadata for exact Panel-I segment reference values."""
    del n_permutations, random_seed
    return {
        **source_metadata,
        "cache_version": SEGMENT_STABILITY_REFERENCE_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "artifact": "segment_stability_reference_and_achievable",
        "row_cohort": "exact_panel_i_included_cell_path_segments",
        "segment_rate_gate": (
            "include_when_max_dark_or_light_segment_mean_rate_is_greater_"
            "than_or_equal_to_threshold"
        ),
        "threshold_comparison": {
            "epoch_and_path": "strict_greater_than",
            "segment": "greater_than_or_equal",
        },
        "averaging_across_segments_or_paths": "none",
        "observed_metric": "segmentwise_unit_area_minimum_overlap",
        "split_half_metric": "same_segment_odd_even_unit_area_overlap",
        "split_half_conditions": (
            "mean_dark_and_light_when_both_valid_otherwise_active_"
            "condition_only"
        ),
        "null_operation": (
            "circularly_roll_light_within_same_physical_segment"
        ),
        "shift_support": "all_unique_integer_bin_shifts_within_segment",
        "null_summary": (
            "minimum_across_every_integer_bin_shift_within_each_segment"
        ),
        "one_condition_silent_null": 0.0,
        "achievable_formula": (
            "(observed_minus_null_minimum)/(split_half_minus_null_minimum)"
        ),
        "denominator_policy": "strictly_greater_than_1e-12",
        "clipping": "none",
        "columns": list(SEGMENT_STABILITY_REFERENCE_COLUMNS),
    }


def build_segment_matched_achievable_stability_cache_metadata(
    source_metadata: dict[str, Any],
    *,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    """Return metadata for the coherent per-neuron segment analysis."""
    del n_permutations, random_seed
    return {
        **source_metadata,
        "cache_version": SEGMENT_MATCHED_ACHIEVABLE_STABILITY_CACHE_VERSION,
        "source_segment_reference_cache_version": source_metadata[
            "cache_version"
        ],
        "artifact": "per_neuron_segment_matched_achievable_stability",
        "formula": (
            "(observed_minus_null_minimum)/(split_half_minus_null_minimum)"
        ),
        "cohort": (
            "segments_passing_existing_unit_path_and_segment_activity_gates_"
            "with_coherent_observed_split_half_and_null_references"
        ),
        "segment_exclusion": (
            "exclude_both_silent_segments_that_are_silent_in_both_conditions"
        ),
        "observed_metric": "same_segment_unit_area_minimum_overlap",
        "split_half_metric": "same_segment_odd_even_unit_area_overlap",
        "split_half_conditions": (
            "mean_dark_and_light_when_both_valid_otherwise_active_"
            "condition_only"
        ),
        "null_metric": "same_segment_unit_area_minimum_overlap",
        "null_operation": (
            "enumerate_every_unique_integer_bin_roll_within_each_same_"
            "physical_segment"
        ),
        "averaging": (
            "equal_mean_across_the_exact_same_coherent_path_segments_for_"
            "observed_split_half_and_null"
        ),
        "null_summary": (
            "equal_mean_of_exact_per_segment_circular_shift_minima"
        ),
        "one_condition_silent_null": 0.0,
        "denominator_policy": "valid_only_when_strictly_greater_than_1e-12",
        "clipping": "none",
        "columns": list(SEGMENT_MATCHED_ACHIEVABLE_STABILITY_COLUMNS),
    }


def build_tuning_correlation_cache_metadata(
    source_metadata: dict[str, Any],
    *,
    artifact: str,
    variant: str,
    n_permutations: int | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Return cache metadata for one dark-light Pearson data product."""
    variant = _validate_tuning_correlation_variant(variant)
    schemas = {
        "average": TUNING_CORRELATION_AVERAGE_COLUMNS,
        "split_half": TUNING_CORRELATION_SPLIT_HALF_COLUMNS,
        "circular_null": TUNING_CORRELATION_NULL_COLUMNS,
        "achievable": ACHIEVABLE_TUNING_CORRELATION_COLUMNS,
    }
    if artifact not in schemas:
        raise ValueError(f"Unsupported correlation cache artifact {artifact!r}.")
    metadata = {
        **source_metadata,
        "cache_version": TUNING_CORRELATION_CACHE_VERSION,
        "source_path_cache_version": source_metadata["cache_version"],
        "artifact": f"tuning_correlation_{artifact}",
        "variant": variant,
        "metric": "pearson_correlation",
        "observed_resolution": variant,
        "observed_candidate_path_cohort": (
            "identical_to_segmentwise_overlap_observed_path_cohort"
        ),
        "observed_final_path_policy": (
            "whole_path_requires_valid_whole_path_r; physical_segments_"
            "requires_at_least_one_valid_active_segment_r"
        ),
        "correlation_averaging": "arithmetic_mean_of_raw_r_values",
        "fisher_z_transform": False,
        "segment_undefined_policy": (
            "exclude_constant_or_silent_active_segments"
        ),
        "split_half_metric": "whole_path_odd_even_pearson_correlation",
        "split_half_plot_cohort": (
            "broad_both_epoch_and_path_rate_gates_without_observed_"
            "reliability_gate"
        ),
        "null_metric": "whole_path_circular_shift_pearson_correlation",
        "null_path_policy": "identical_to_variant_final_observed_paths",
        "columns": list(schemas[artifact]),
    }
    if artifact == "achievable":
        metadata.update(
            {
                "formula": (
                    "(observed_minus_null_mean)/(matched_split_half_"
                    "minus_null_mean)"
                ),
                "split_half_matching": "recompute_over_exact_observed_paths",
                "interpretation": (
                    "same_resolution_normalized_correlation"
                    if variant == "whole_path"
                    else "cross_resolution_normalized_correlation_using_"
                    "whole_path_references"
                ),
                "clipping": "none",
            }
        )
    if artifact in {"circular_null", "achievable"}:
        if n_permutations is None or random_seed is None:
            raise ValueError(
                "Null-dependent correlation caches require permutation settings."
            )
        metadata.update(
            {
                "n_permutations": int(n_permutations),
                "random_seed": int(random_seed),
                "shift_support": (
                    "integer_bins_from_zero_through_n_bins_minus_one"
                ),
                "rng": "numpy.random.default_rng",
            }
        )
    return metadata


def build_tuning_correlation_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return a metadata-addressed path for a correlation artifact."""
    prefixes = {
        "tuning_correlation_average": TUNING_CORRELATION_AVERAGE_CACHE_PREFIX,
        "tuning_correlation_split_half": (
            TUNING_CORRELATION_SPLIT_HALF_CACHE_PREFIX
        ),
        "tuning_correlation_circular_null": (
            TUNING_CORRELATION_NULL_CACHE_PREFIX
        ),
        "tuning_correlation_achievable": (
            ACHIEVABLE_TUNING_CORRELATION_CACHE_PREFIX
        ),
    }
    artifact = str(metadata.get("artifact", ""))
    if artifact not in prefixes:
        raise ValueError(f"Unsupported correlation cache artifact {artifact!r}.")
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    region = "".join(
        character if character.isalnum() else "_"
        for character in str(metadata["region"])
    ).strip("_")
    variant = str(metadata["variant"])
    return Path(cache_dir) / (
        f"{prefixes[artifact]}_{variant}_{region or 'none'}_"
        f"cachev{int(metadata['cache_version'])}_{digest}.parquet"
    )


def build_panel_b_tuning_average_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return a versioned, metadata-addressed neuron-average Parquet path."""
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    region = "".join(
        character if character.isalnum() else "_"
        for character in str(metadata["region"])
    ).strip("_")
    filename = (
        f"{PANEL_B_TUNING_AVERAGE_CACHE_PREFIX}_{region or 'none'}_"
        f"cachev{int(metadata['cache_version'])}_{digest}.parquet"
    )
    return Path(cache_dir) / filename


def build_panel_b_dark_split_half_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return a versioned dark split-half summary Parquet path."""
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    region = "".join(
        character if character.isalnum() else "_"
        for character in str(metadata["region"])
    ).strip("_")
    filename = (
        f"{PANEL_B_DARK_SPLIT_HALF_CACHE_PREFIX}_{region or 'none'}_"
        f"cachev{int(metadata['cache_version'])}_{digest}.parquet"
    )
    return Path(cache_dir) / filename


def _build_panel_b_reference_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
    prefix: str,
) -> Path:
    """Return a metadata-addressed Panel B reference-cache path."""
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    region = "".join(
        character if character.isalnum() else "_"
        for character in str(metadata["region"])
    ).strip("_")
    return Path(cache_dir) / (
        f"{prefix}_{region or 'none'}_"
        f"cachev{int(metadata['cache_version'])}_{digest}.parquet"
    )


def build_panel_b_split_half_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned dark/light split-half cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        PANEL_B_SPLIT_HALF_CACHE_PREFIX,
    )


def build_panel_b_circular_null_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned circular-null cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        PANEL_B_CIRCULAR_NULL_CACHE_PREFIX,
    )


def build_panel_d_achievable_stability_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned achievable-stability cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        PANEL_D_ACHIEVABLE_STABILITY_CACHE_PREFIX,
    )


def build_full_path_achievable_stability_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned coherent whole-path analysis cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        FULL_PATH_ACHIEVABLE_STABILITY_CACHE_PREFIX,
    )


def build_panel_h_shift_profile_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned Panel H shift-profile cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        PANEL_H_SHIFT_PROFILE_CACHE_PREFIX,
    )


def build_segment_matched_achievable_stability_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned coherent segment analysis cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        SEGMENT_MATCHED_ACHIEVABLE_STABILITY_CACHE_PREFIX,
    )


def build_segment_overlap_response_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned segment overlap/response cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        SEGMENT_OVERLAP_RESPONSE_CACHE_PREFIX,
    )


def build_segment_stability_reference_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return the versioned segment reference/achievable cache path."""
    return _build_panel_b_reference_cache_path(
        cache_dir,
        metadata,
        SEGMENT_STABILITY_REFERENCE_CACHE_PREFIX,
    )


def _tuning_similarity_metadata_path(cache_path: Path) -> Path:
    """Return the JSON metadata sidecar path for a TSI table."""
    return Path(cache_path).with_suffix(".json")


def save_panel_b_tuning_similarity_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the reproducible long-form TSI table and metadata sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, list(PANEL_B_TUNING_SIMILARITY_COLUMNS)].to_parquet(
        cache_path,
        index=False,
    )
    _tuning_similarity_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_panel_b_tuning_average_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the derived per-neuron tuning average and metadata sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, list(PANEL_B_TUNING_AVERAGE_COLUMNS)].to_parquet(
        cache_path,
        index=False,
    )
    _tuning_similarity_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_panel_b_dark_split_half_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the dark split-half neuron summary and metadata sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, list(PANEL_B_DARK_SPLIT_HALF_COLUMNS)].to_parquet(
        cache_path,
        index=False,
    )
    _tuning_similarity_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_panel_b_reference_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
    columns: Sequence[str],
) -> None:
    """Save a Panel B reference table and its exact metadata sidecar."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, list(columns)].to_parquet(cache_path, index=False)
    _tuning_similarity_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_panel_b_split_half_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the dark/light split-half summary and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        PANEL_B_SPLIT_HALF_COLUMNS,
    )


def save_panel_b_circular_null_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the circular-null summary and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        PANEL_B_CIRCULAR_NULL_COLUMNS,
    )


def save_panel_d_achievable_stability_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the achievable-stability table and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        PANEL_D_ACHIEVABLE_STABILITY_COLUMNS,
    )


def save_full_path_achievable_stability_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the coherent whole-path achievable table and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        FULL_PATH_ACHIEVABLE_STABILITY_COLUMNS,
    )


def save_panel_h_shift_profile_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save common-grid whole-path shift profiles and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        PANEL_H_SHIFT_PROFILE_COLUMNS,
    )


def save_segment_matched_achievable_stability_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the coherent per-neuron segment analysis and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        SEGMENT_MATCHED_ACHIEVABLE_STABILITY_COLUMNS,
    )


def save_segment_overlap_response_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the segment overlap/response table and metadata."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        SEGMENT_OVERLAP_RESPONSE_COLUMNS,
    )


def save_segment_stability_reference_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save exact segment reference and achievable values."""
    _save_panel_b_reference_cache(
        cache_path,
        table,
        metadata,
        SEGMENT_STABILITY_REFERENCE_COLUMNS,
    )


def save_tuning_correlation_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save one versioned correlation table and exact metadata sidecar."""
    columns = tuple(str(value) for value in metadata["columns"])
    _save_panel_b_reference_cache(cache_path, table, metadata, columns)


def load_tuning_correlation_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a correlation cache only when metadata and schema match."""
    columns = tuple(str(value) for value in expected_metadata["columns"])
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        columns,
    )


def load_panel_b_tuning_average_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return the cached per-neuron average when metadata and schema match."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _tuning_similarity_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected_metadata:
            return None
        table = pd.read_parquet(cache_path)
        missing = [
            column for column in PANEL_B_TUNING_AVERAGE_COLUMNS
            if column not in table
        ]
        if missing:
            return None
        return table.loc[:, list(PANEL_B_TUNING_AVERAGE_COLUMNS)].copy()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def load_panel_b_dark_split_half_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid cached dark split-half neuron summary."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _tuning_similarity_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected_metadata:
            return None
        table = pd.read_parquet(cache_path)
        missing = [
            column
            for column in PANEL_B_DARK_SPLIT_HALF_COLUMNS
            if column not in table
        ]
        if missing:
            return None
        return table.loc[:, list(PANEL_B_DARK_SPLIT_HALF_COLUMNS)].copy()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_panel_b_reference_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
    columns: Sequence[str],
) -> Any | None:
    """Return a Panel B reference cache when metadata and schema match."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _tuning_similarity_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected_metadata:
            return None
        table = pd.read_parquet(cache_path)
        if any(column not in table for column in columns):
            return None
        return table.loc[:, list(columns)].copy()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def load_panel_b_split_half_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid cached dark/light split-half summary."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        PANEL_B_SPLIT_HALF_COLUMNS,
    )


def load_panel_b_circular_null_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid cached circular-null summary."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        PANEL_B_CIRCULAR_NULL_COLUMNS,
    )


def load_panel_d_achievable_stability_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid cached achievable-stability table."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        PANEL_D_ACHIEVABLE_STABILITY_COLUMNS,
    )


def load_full_path_achievable_stability_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid coherent whole-path achievable cache."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        FULL_PATH_ACHIEVABLE_STABILITY_COLUMNS,
    )


def load_panel_h_shift_profile_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid common-grid whole-path shift-profile cache."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        PANEL_H_SHIFT_PROFILE_COLUMNS,
    )


def load_segment_matched_achievable_stability_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid coherent per-neuron segment analysis cache."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        SEGMENT_MATCHED_ACHIEVABLE_STABILITY_COLUMNS,
    )


def load_segment_overlap_response_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid segment overlap/response cache."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        SEGMENT_OVERLAP_RESPONSE_COLUMNS,
    )


def load_segment_stability_reference_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a valid exact-segment reference/achievable cache."""
    return _load_panel_b_reference_cache(
        cache_path,
        expected_metadata,
        SEGMENT_STABILITY_REFERENCE_COLUMNS,
    )


def load_panel_b_tuning_similarity_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return a cached TSI table only when metadata and schema still match."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _tuning_similarity_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected_metadata:
            print(f"Ignoring stale Panel B tuning-similarity cache at {cache_path}.")
            return None
        table = pd.read_parquet(cache_path)
        missing = [
            column
            for column in PANEL_B_TUNING_SIMILARITY_COLUMNS
            if column not in table
        ]
        if missing:
            print(
                "Ignoring invalid Panel B tuning-similarity cache at "
                f"{cache_path}: missing columns {missing!r}."
            )
            return None
        return table.loc[:, list(PANEL_B_TUNING_SIMILARITY_COLUMNS)].copy()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"Ignoring unreadable Panel B tuning-similarity cache at "
            f"{cache_path}: {exc}"
        )
        return None


def load_or_compute_panel_b_tuning_similarity_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute path-specific TSI values and eligibility flags."""
    metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    cache_path = (
        build_panel_b_tuning_similarity_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_b_tuning_similarity_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel B tuning-similarity cache from {cache_path}.")
            return cached
    table = build_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    if cache_path is not None:
        save_panel_b_tuning_similarity_cache(cache_path, table, metadata)
        print(f"Saved Panel B tuning-similarity cache to {cache_path}.")
    return table


def load_or_compute_panel_b_tuning_average_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute the persisted per-neuron Panel B average table."""
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    average_metadata = build_panel_b_tuning_average_cache_metadata(metadata)
    average_cache_path = (
        build_panel_b_tuning_average_cache_path(cache_dir, average_metadata)
        if cache_dir is not None
        else None
    )
    if average_cache_path is not None and not refresh_cache:
        cached = load_panel_b_tuning_average_cache(
            average_cache_path,
            average_metadata,
        )
        if cached is not None:
            print(f"Loaded Panel B tuning-average cache from {average_cache_path}.")
            return cached
    table = derive_panel_b_tuning_average_table(
        path_table,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
    )
    if average_cache_path is not None:
        save_panel_b_tuning_average_cache(
            average_cache_path,
            table,
            average_metadata,
        )
        print(f"Saved Panel B tuning-average cache to {average_cache_path}.")
    return table


def load_or_compute_segment_overlap_response_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or derive the persisted segment overlap/response table."""
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_segment_overlap_response_cache_metadata(source_metadata)
    cache_path = (
        build_segment_overlap_response_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_segment_overlap_response_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded segment overlap/response cache from {cache_path}.")
            return cached
    table = derive_segment_overlap_response_table(
        path_table,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
    )
    if cache_path is not None:
        save_segment_overlap_response_cache(cache_path, table, metadata)
        print(f"Saved segment overlap/response cache to {cache_path}.")
    return table


def load_or_compute_segment_stability_reference_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or derive reference values for the exact Panel-I segments."""
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "cache_dir": cache_dir,
        "refresh_cache": refresh_cache,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table = load_or_compute_panel_b_tuning_similarity_table(**common)
    segment_table = load_or_compute_segment_overlap_response_table(**common)
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_segment_stability_reference_cache_metadata(
        source_metadata,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_segment_stability_reference_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_segment_stability_reference_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded segment stability reference cache from {cache_path}.")
            return cached
    table = derive_segment_stability_reference_table(
        path_table,
        segment_table,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    if cache_path is not None:
        save_segment_stability_reference_cache(cache_path, table, metadata)
        print(f"Saved segment stability reference cache to {cache_path}.")
    return table


def load_or_compute_segment_matched_achievable_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or derive coherent per-neuron segment achievable stability."""
    segment_reference_table = (
        load_or_compute_segment_stability_reference_table(
            data_root=data_root,
            datasets=datasets,
            region=region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            min_epoch_movement_firing_rate_hz=(
                min_epoch_movement_firing_rate_hz
            ),
            min_path_movement_firing_rate_hz=(
                min_path_movement_firing_rate_hz
            ),
            min_segment_mean_firing_rate_hz=(
                min_segment_mean_firing_rate_hz
            ),
            min_stability_correlation=min_stability_correlation,
            n_permutations=n_permutations,
            random_seed=random_seed,
        )
    )
    path_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    reference_metadata = build_segment_stability_reference_cache_metadata(
        path_metadata,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    metadata = build_segment_matched_achievable_stability_cache_metadata(
        reference_metadata,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_segment_matched_achievable_stability_cache_path(
            cache_dir,
            metadata,
        )
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_segment_matched_achievable_stability_cache(
            cache_path,
            metadata,
        )
        if cached is not None:
            print(
                "Loaded segment-matched achievable cache from "
                f"{cache_path}."
            )
            return cached
    table = derive_segment_matched_achievable_stability_table(
        segment_reference_table
    )
    if cache_path is not None:
        save_segment_matched_achievable_stability_cache(
            cache_path,
            table,
            metadata,
        )
        print(f"Saved segment-matched achievable cache to {cache_path}.")
    return table


def load_or_compute_panel_b_dark_split_half_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute the dark odd/even overlap reference table."""
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_panel_b_dark_split_half_cache_metadata(source_metadata)
    cache_path = (
        build_panel_b_dark_split_half_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_b_dark_split_half_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel B dark split-half cache from {cache_path}.")
            return cached
    table = derive_panel_b_dark_split_half_table(
        path_table,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
    )
    if cache_path is not None:
        save_panel_b_dark_split_half_cache(cache_path, table, metadata)
        print(f"Saved Panel B dark split-half cache to {cache_path}.")
    return table


def load_or_compute_panel_b_split_half_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute the dark/light split-half reference table."""
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_panel_b_split_half_cache_metadata(source_metadata)
    cache_path = (
        build_panel_b_split_half_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_b_split_half_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel B dark/light split-half cache from {cache_path}.")
            return cached
    observed_table = derive_panel_b_tuning_average_table(
        path_table,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
    )
    table = derive_panel_b_split_half_table(
        path_table,
        observed_table=observed_table,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
    )
    if cache_path is not None:
        save_panel_b_split_half_cache(cache_path, table, metadata)
        print(f"Saved Panel B dark/light split-half cache to {cache_path}.")
    return table


def load_or_compute_panel_b_circular_null_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or compute the Panel B circular-shift minimum table."""
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_panel_b_circular_null_cache_metadata(
        source_metadata,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_panel_b_circular_null_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_b_circular_null_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel B circular-null cache from {cache_path}.")
            return cached
    observed_table = derive_panel_b_tuning_average_table(
        path_table,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
    )
    table = derive_panel_b_circular_null_table(
        path_table,
        observed_table=observed_table,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    if cache_path is not None:
        save_panel_b_circular_null_cache(cache_path, table, metadata)
        print(f"Saved Panel B circular-null cache to {cache_path}.")
    return table


def load_or_compute_panel_d_achievable_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or compute achievable stability for the Panel B cohort."""
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "cache_dir": cache_dir,
        "refresh_cache": refresh_cache,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table = load_or_compute_panel_b_tuning_similarity_table(**common)
    observed_table = load_or_compute_panel_b_tuning_average_table(**common)
    split_half_table = load_or_compute_panel_b_split_half_table(**common)
    null_table = load_or_compute_panel_b_circular_null_table(
        **common,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_panel_d_achievable_stability_cache_metadata(
        source_metadata,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_panel_d_achievable_stability_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_d_achievable_stability_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel D achievable-stability cache from {cache_path}.")
            return cached
    table = derive_panel_d_achievable_stability_table(
        path_table,
        observed_table,
        split_half_table,
        null_table,
    )
    if cache_path is not None:
        save_panel_d_achievable_stability_cache(cache_path, table, metadata)
        print(f"Saved Panel D achievable-stability cache to {cache_path}.")
    return table


def load_or_compute_full_path_achievable_stability_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    apply_stability_filter: bool,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or compute coherent whole-path achievable stability."""
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_segment_mean_firing_rate_hz=min_segment_mean_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_full_path_achievable_stability_cache_metadata(
        source_metadata,
        apply_stability_filter=apply_stability_filter,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_full_path_achievable_stability_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_full_path_achievable_stability_cache(
            cache_path,
            metadata,
        )
        if cached is not None:
            print(f"Loaded full-path achievable cache from {cache_path}.")
            return cached
    table = derive_full_path_achievable_stability_table(
        path_table,
        apply_stability_filter=apply_stability_filter,
        min_epoch_movement_firing_rate_hz=min_epoch_movement_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    if cache_path is not None:
        save_full_path_achievable_stability_cache(cache_path, table, metadata)
        print(f"Saved full-path achievable cache to {cache_path}.")
    return table


def load_or_compute_panel_h_shift_profile_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute all nonsilent neuron/path shift profiles."""
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        **common,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )
    source_metadata = build_panel_b_tuning_similarity_cache_metadata(**common)
    metadata = build_panel_h_shift_profile_cache_metadata(source_metadata)
    cache_path = (
        build_panel_h_shift_profile_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_h_shift_profile_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel H shift profiles from {cache_path}.")
            return cached
    table = derive_panel_h_shift_profile_table(path_table)
    if cache_path is not None:
        save_panel_h_shift_profile_cache(cache_path, table, metadata)
        print(f"Saved Panel H shift profiles to {cache_path}.")
    return table


def _load_correlation_path_source(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool,
    min_epoch_movement_firing_rate_hz: float,
    min_path_movement_firing_rate_hz: float,
    min_segment_mean_firing_rate_hz: float,
    min_stability_correlation: float,
) -> tuple[Any, dict[str, Any]]:
    """Return the shared path table and metadata for Pearson products."""
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table = load_or_compute_panel_b_tuning_similarity_table(
        **common,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )
    metadata = build_panel_b_tuning_similarity_cache_metadata(**common)
    return path_table, metadata


def load_or_compute_tuning_correlation_average_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    variant: str,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute one per-neuron dark-light Pearson summary."""
    variant = _validate_tuning_correlation_variant(variant)
    path_table, source_metadata = _load_correlation_path_source(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=(
            min_path_movement_firing_rate_hz
        ),
        min_segment_mean_firing_rate_hz=(
            min_segment_mean_firing_rate_hz
        ),
        min_stability_correlation=min_stability_correlation,
    )
    metadata = build_tuning_correlation_cache_metadata(
        source_metadata,
        artifact="average",
        variant=variant,
    )
    cache_path = (
        build_tuning_correlation_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_tuning_correlation_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded tuning-correlation average from {cache_path}.")
            return cached
    table = derive_tuning_correlation_average_table(
        path_table,
        variant=variant,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
    )
    if cache_path is not None:
        save_tuning_correlation_cache(cache_path, table, metadata)
        print(f"Saved tuning-correlation average to {cache_path}.")
    return table


def load_or_compute_tuning_correlation_split_half_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    variant: str,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute the broad whole-path Pearson split reference."""
    variant = _validate_tuning_correlation_variant(variant)
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "cache_dir": cache_dir,
        "refresh_cache": refresh_cache,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table, source_metadata = _load_correlation_path_source(**common)
    observed = load_or_compute_tuning_correlation_average_table(
        **common,
        variant=variant,
    )
    metadata = build_tuning_correlation_cache_metadata(
        source_metadata,
        artifact="split_half",
        variant=variant,
    )
    cache_path = (
        build_tuning_correlation_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_tuning_correlation_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded tuning-correlation split-half from {cache_path}.")
            return cached
    table = derive_tuning_correlation_split_half_table(
        path_table,
        observed,
        variant=variant,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=(
            min_path_movement_firing_rate_hz
        ),
    )
    if cache_path is not None:
        save_tuning_correlation_cache(cache_path, table, metadata)
        print(f"Saved tuning-correlation split-half to {cache_path}.")
    return table


def load_or_compute_tuning_correlation_circular_null_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    variant: str,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or compute whole-path circular-shift Pearson null scores."""
    variant = _validate_tuning_correlation_variant(variant)
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "cache_dir": cache_dir,
        "refresh_cache": refresh_cache,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table, source_metadata = _load_correlation_path_source(**common)
    observed = load_or_compute_tuning_correlation_average_table(
        **common,
        variant=variant,
    )
    metadata = build_tuning_correlation_cache_metadata(
        source_metadata,
        artifact="circular_null",
        variant=variant,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_tuning_correlation_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_tuning_correlation_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded tuning-correlation null from {cache_path}.")
            return cached
    table = derive_tuning_correlation_circular_null_table(
        path_table,
        observed,
        variant=variant,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    if cache_path is not None:
        save_tuning_correlation_cache(cache_path, table, metadata)
        print(f"Saved tuning-correlation null to {cache_path}.")
    return table


def load_or_compute_achievable_tuning_correlation_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    variant: str,
    refresh_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    n_permutations: int = DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
    random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Any:
    """Load or compute normalized achievable dark-light correlation."""
    variant = _validate_tuning_correlation_variant(variant)
    common = {
        "data_root": data_root,
        "datasets": datasets,
        "region": region,
        "light_epoch": light_epoch,
        "dark_epoch": dark_epoch,
        "cache_dir": cache_dir,
        "refresh_cache": refresh_cache,
        "min_epoch_movement_firing_rate_hz": (
            min_epoch_movement_firing_rate_hz
        ),
        "min_path_movement_firing_rate_hz": (
            min_path_movement_firing_rate_hz
        ),
        "min_segment_mean_firing_rate_hz": (
            min_segment_mean_firing_rate_hz
        ),
        "min_stability_correlation": min_stability_correlation,
    }
    path_table, source_metadata = _load_correlation_path_source(**common)
    observed = load_or_compute_tuning_correlation_average_table(
        **common,
        variant=variant,
    )
    split_half = load_or_compute_tuning_correlation_split_half_table(
        **common,
        variant=variant,
    )
    null = load_or_compute_tuning_correlation_circular_null_table(
        **common,
        variant=variant,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    metadata = build_tuning_correlation_cache_metadata(
        source_metadata,
        artifact="achievable",
        variant=variant,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    cache_path = (
        build_tuning_correlation_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_tuning_correlation_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded achievable tuning correlation from {cache_path}.")
            return cached
    table = derive_achievable_tuning_correlation_table(
        path_table,
        observed,
        split_half,
        null,
        variant=variant,
    )
    if cache_path is not None:
        save_tuning_correlation_cache(cache_path, table, metadata)
        print(f"Saved achievable tuning correlation to {cache_path}.")
    return table


def _path_invariance_qc_status(record: dict[str, Any]) -> str:
    """Return a compact explanation of failed path-pair eligibility checks."""
    checks = (
        f"{condition}_{member}_{kind}"
        for condition in ("dark", "light")
        for member in ("x", "y")
        for kind in ("path_rate", "stability")
    )
    failures = [name for name in checks if not bool(record[f"passes_{name}_qc"])]
    return "valid" if not failures else "fails_" + "_and_".join(failures)


def _build_session_path_invariance_table(
    *,
    data_root: Path,
    animal_name: str,
    date: str,
    region: str,
    dark_epoch: str,
    light_epoch: str,
    min_path_movement_firing_rate_hz: float,
    min_stability_correlation: float,
) -> Any:
    """Return per-unit same-turn path invariance in dark and light."""
    import pandas as pd
    import xarray as xr

    stability, stability_path = _load_session_stability_rows(
        data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        dark_epoch=dark_epoch,
        light_epoch=light_epoch,
    )
    epoch_by_condition = {"dark": str(dark_epoch), "light": str(light_epoch)}
    curve_paths: dict[tuple[str, str], Path] = {}
    curves: dict[tuple[str, str], Any] = {}
    for condition, epoch in epoch_by_condition.items():
        for path_name in PANEL_B_PATH_ORDER:
            curve_path = _figure_2.get_compute_tuning_curve_path(
                data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                trajectory=path_name,
            )
            if not curve_path.exists():
                raise FileNotFoundError(
                    f"Missing tuning-curve artifact: {curve_path}"
                )
            curve_paths[(condition, path_name)] = curve_path
            curves[(condition, path_name)] = xr.load_dataarray(curve_path)

    reference_key = ("dark", PANEL_B_PATH_ORDER[0])
    reference_curve = curves[reference_key]
    reference_path = curve_paths[reference_key]
    for key, curve in curves.items():
        _require_aligned_tuning_curves(
            reference_curve,
            curve,
            dark_path=reference_path,
            light_path=curve_paths[key],
        )
    units = np.asarray(reference_curve.coords["unit"].values, dtype=int)
    curve_values = {
        key: np.asarray(curve.transpose("unit", ...).values)
        for key, curve in curves.items()
    }

    records: list[dict[str, Any]] = []
    for turn_direction, (x_path, y_path) in PANEL_C_TURN_PATHS.items():
        for unit_index, unit in enumerate(units):
            record: dict[str, Any] = {
                "animal_name": str(animal_name),
                "date": str(date),
                "region": str(region),
                "unit": int(unit),
                "turn_direction": str(turn_direction),
                "x_path": str(x_path),
                "y_path": str(y_path),
                "dark_epoch": str(dark_epoch),
                "light_epoch": str(light_epoch),
                "dark_x_tuning_curve_path": str(
                    curve_paths[("dark", x_path)]
                ),
                "dark_y_tuning_curve_path": str(
                    curve_paths[("dark", y_path)]
                ),
                "light_x_tuning_curve_path": str(
                    curve_paths[("light", x_path)]
                ),
                "light_y_tuning_curve_path": str(
                    curve_paths[("light", y_path)]
                ),
                "stability_table_path": str(stability_path),
                "cache_version": PANEL_C_PATH_INVARIANCE_CACHE_VERSION,
            }
            for condition in ("dark", "light"):
                metric = compute_path_invariance(
                    curve_values[(condition, x_path)][unit_index],
                    curve_values[(condition, y_path)][unit_index],
                )
                record.update(
                    {f"{condition}_{key}": value for key, value in metric.items()}
                )
                epoch = epoch_by_condition[condition]
                for member, path_name in (("x", x_path), ("y", y_path)):
                    stability_key = (int(unit), str(epoch), str(path_name))
                    if stability_key in stability.index:
                        qc_row = stability.loc[stability_key]
                        correlation = float(qc_row["stability_correlation"])
                        rate_result = compute_pooled_path_movement_firing_rate(
                            float(qc_row["n_odd_spikes"]),
                            float(qc_row["n_even_spikes"]),
                            float(qc_row["odd_duration_s"]),
                            float(qc_row["even_duration_s"]),
                        )
                    else:
                        correlation = float("nan")
                        rate_result = compute_pooled_path_movement_firing_rate(
                            float("nan"),
                            float("nan"),
                            float("nan"),
                            float("nan"),
                        )
                    rate = float(rate_result["path_movement_firing_rate_hz"])
                    prefix = f"{condition}_{member}"
                    record[f"{prefix}_path_movement_firing_rate_hz"] = rate
                    record[f"{prefix}_path_movement_firing_rate_status"] = (
                        rate_result["path_movement_firing_rate_status"]
                    )
                    record[f"{prefix}_stability_correlation"] = correlation
                    record[f"passes_{prefix}_path_rate_qc"] = bool(
                        rate_result["path_movement_firing_rate_status"] == "valid"
                        and np.isfinite(rate)
                        and rate >= float(min_path_movement_firing_rate_hz)
                    )
                    record[f"passes_{prefix}_stability_qc"] = bool(
                        np.isfinite(correlation)
                        and correlation >= float(min_stability_correlation)
                    )
            record["passes_qc"] = all(
                bool(record[column])
                for column in PANEL_C_PATH_INVARIANCE_COLUMNS
                if column.startswith("passes_") and column != "passes_qc"
            )
            record["qc_status"] = _path_invariance_qc_status(record)
            records.append(record)

    return pd.DataFrame.from_records(
        records,
        columns=PANEL_C_PATH_INVARIANCE_COLUMNS,
    )


def build_panel_c_path_invariance_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Return pooled same-turn path-invariance scores and eligibility flags."""
    import pandas as pd

    if min_path_movement_firing_rate_hz < 0.0:
        raise ValueError(
            "min_path_movement_firing_rate_hz must be non-negative."
        )
    if min_stability_correlation < -1.0:
        raise ValueError("min_stability_correlation must be at least -1.")
    tables = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        resolved_dark_epoch = (
            str(dataset_dark_epoch)
            if dark_epoch is None
            else _figure_2.get_dark_epoch(animal_name, date, dark_epoch)
        )
        resolved_light_epoch = _figure_2.get_light_epoch(
            animal_name,
            date,
            light_epoch,
        )
        tables.append(
            _build_session_path_invariance_table(
                data_root=data_root,
                animal_name=animal_name,
                date=date,
                region=region,
                dark_epoch=resolved_dark_epoch,
                light_epoch=resolved_light_epoch,
                min_path_movement_firing_rate_hz=(
                    min_path_movement_firing_rate_hz
                ),
                min_stability_correlation=min_stability_correlation,
            )
        )
    if not tables:
        return pd.DataFrame(columns=PANEL_C_PATH_INVARIANCE_COLUMNS)
    return pd.concat(tables, ignore_index=True).loc[
        :, list(PANEL_C_PATH_INVARIANCE_COLUMNS)
    ]


def build_panel_c_path_invariance_cache_metadata(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    min_path_movement_firing_rate_hz: float,
    min_stability_correlation: float,
) -> dict[str, Any]:
    """Return parameters and source fingerprints identifying the PII cache."""
    dataset_metadata = []
    source_paths: list[Path] = []
    for dataset in datasets:
        animal_name, date, dataset_dark_epoch = normalize_dataset_id(dataset)
        resolved_dark_epoch = (
            str(dataset_dark_epoch)
            if dark_epoch is None
            else _figure_2.get_dark_epoch(animal_name, date, dark_epoch)
        )
        resolved_light_epoch = _figure_2.get_light_epoch(
            animal_name,
            date,
            light_epoch,
        )
        dataset_metadata.append(
            {
                "animal_name": str(animal_name),
                "date": str(date),
                "dark_epoch": str(resolved_dark_epoch),
                "light_epoch": str(resolved_light_epoch),
            }
        )
        source_paths.append(get_stability_table_path(data_root, animal_name, date))
        for path_name in PANEL_B_PATH_ORDER:
            for epoch in (resolved_dark_epoch, resolved_light_epoch):
                source_paths.append(
                    _figure_2.get_compute_tuning_curve_path(
                        data_root,
                        animal_name=animal_name,
                        date=date,
                        region=region,
                        epoch=epoch,
                        trajectory=path_name,
                    )
                )
    return {
        "cache_version": PANEL_C_PATH_INVARIANCE_CACHE_VERSION,
        "figure": LEGACY_CACHE_FIGURE_NAME,
        "panel": "C",
        "artifact": "same_turn_path_invariance",
        "metric": "twice_raw_minimum_overlap_over_total_area",
        "formula": "2*sum(min(x,y))/(sum(x)+sum(y))",
        "path_orientation": "saved_progression_no_reversal",
        "nan_handling": "path_local_linear_interpolation_with_endpoint_extension",
        "data_root": str(Path(data_root)),
        "region": str(region),
        "datasets": dataset_metadata,
        "turn_paths": {
            key: list(value) for key, value in PANEL_C_TURN_PATHS.items()
        },
        "min_path_movement_firing_rate_hz": float(
            min_path_movement_firing_rate_hz
        ),
        "min_stability_correlation": float(min_stability_correlation),
        "columns": list(PANEL_C_PATH_INVARIANCE_COLUMNS),
        "sources": [
            _artifact_fingerprint(path)
            for path in sorted(set(source_paths), key=lambda value: str(value))
        ],
    }


def build_panel_c_path_invariance_cache_path(
    cache_dir: Path,
    metadata: dict[str, Any],
) -> Path:
    """Return a versioned, metadata-addressed Panel C Parquet path."""
    digest = hashlib.sha1(
        json.dumps(metadata, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    region = "".join(
        character if character.isalnum() else "_"
        for character in str(metadata["region"])
    ).strip("_")
    filename = (
        f"{PANEL_C_PATH_INVARIANCE_CACHE_PREFIX}_{region or 'none'}_"
        f"cachev{int(metadata['cache_version'])}_{digest}.parquet"
    )
    return Path(cache_dir) / filename


def _path_invariance_metadata_path(cache_path: Path) -> Path:
    """Return the JSON metadata sidecar path for a path-invariance table."""
    return Path(cache_path).with_suffix(".json")


def save_panel_c_path_invariance_cache(
    cache_path: Path,
    table: Any,
    metadata: dict[str, Any],
) -> None:
    """Save the reproducible long-form path-invariance table and metadata."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, list(PANEL_C_PATH_INVARIANCE_COLUMNS)].to_parquet(
        cache_path,
        index=False,
    )
    _path_invariance_metadata_path(cache_path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_panel_c_path_invariance_cache(
    cache_path: Path,
    expected_metadata: dict[str, Any],
) -> Any | None:
    """Return cached path invariance only when metadata and schema match."""
    import pandas as pd

    cache_path = Path(cache_path)
    metadata_path = _path_invariance_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        return None
    try:
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != expected_metadata:
            print(f"Ignoring stale Panel C path-invariance cache at {cache_path}.")
            return None
        table = pd.read_parquet(cache_path)
        missing = [
            column for column in PANEL_C_PATH_INVARIANCE_COLUMNS if column not in table
        ]
        if missing:
            print(
                "Ignoring invalid Panel C path-invariance cache at "
                f"{cache_path}: missing columns {missing!r}."
            )
            return None
        return table.loc[:, list(PANEL_C_PATH_INVARIANCE_COLUMNS)].copy()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"Ignoring unreadable Panel C path-invariance cache at "
            f"{cache_path}: {exc}"
        )
        return None


def load_or_compute_panel_c_path_invariance_table(
    *,
    data_root: Path,
    datasets: Sequence[DatasetId],
    region: str,
    light_epoch: str | None,
    dark_epoch: str | None,
    cache_dir: Path | None,
    refresh_cache: bool = False,
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
) -> Any:
    """Load or compute same-turn path invariance and eligibility flags."""
    metadata = build_panel_c_path_invariance_cache_metadata(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    cache_path = (
        build_panel_c_path_invariance_cache_path(cache_dir, metadata)
        if cache_dir is not None
        else None
    )
    if cache_path is not None and not refresh_cache:
        cached = load_panel_c_path_invariance_cache(cache_path, metadata)
        if cached is not None:
            print(f"Loaded Panel C path-invariance cache from {cache_path}.")
            return cached
    table = build_panel_c_path_invariance_table(
        data_root=data_root,
        datasets=datasets,
        region=region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_path_movement_firing_rate_hz=min_path_movement_firing_rate_hz,
        min_stability_correlation=min_stability_correlation,
    )
    if cache_path is not None:
        save_panel_c_path_invariance_cache(cache_path, table, metadata)
        print(f"Saved Panel C path-invariance cache to {cache_path}.")
    return table


def _normalized_schematic_curves(
    example: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one dark/light unit-area curve pair for the TSI schematic."""
    if example is not None and "epoch_rates" in example:
        trajectory = str(example.get("trajectories", ("",))[0])
        dark_rates = example["epoch_rates"]["dark"]["firing_rates"].get(
            trajectory
        )
        light_rates = example["epoch_rates"]["light"]["firing_rates"].get(
            trajectory
        )
        if dark_rates is not None and light_rates is not None:
            dark_position, dark = dark_rates
            light_position, light = light_rates
            dark_position = np.asarray(dark_position, dtype=float)
            light_position = np.asarray(light_position, dtype=float)
            if dark_position.shape == light_position.shape and np.array_equal(
                dark_position,
                light_position,
                equal_nan=True,
            ):
                score = compute_path_tuning_similarity(dark, light)
                if score["similarity_status"] == "valid":
                    dark = np.asarray(dark, dtype=float)
                    light = np.asarray(light, dtype=float)
                    from v1ca1.task_progression.similarity import interpolate_nans

                    return (
                        np.linspace(0.0, 1.0, dark.size),
                        interpolate_nans(dark) / float(score["dark_area"]),
                        interpolate_nans(light) / float(score["light_area"]),
                    )
    position = np.linspace(0.0, 1.0, 81)
    dark = np.exp(-0.5 * ((position - 0.42) / 0.13) ** 2)
    light = np.exp(-0.5 * ((position - 0.50) / 0.15) ** 2)
    return position, dark / dark.sum(), light / light.sum()


def _path_invariance_schematic_curves(
    example: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Return two same-turn raw curves for the path-invariance schematic."""
    if example is not None and "epoch_rates" in example:
        trajectories = tuple(str(path) for path in example.get("trajectories", ()))
        is_same_turn_pair = any(
            trajectories[:2] == tuple(pair)
            for pair in PANEL_C_TURN_PATHS.values()
        )
        dark_rates = example["epoch_rates"].get("dark", {}).get(
            "firing_rates",
            {},
        )
        if is_same_turn_pair and len(trajectories) >= 2:
            x_rates = dark_rates.get(trajectories[0])
            y_rates = dark_rates.get(trajectories[1])
            if x_rates is not None and y_rates is not None:
                x_position, x_curve = x_rates
                y_position, y_curve = y_rates
                x_position = np.asarray(x_position, dtype=float)
                y_position = np.asarray(y_position, dtype=float)
                score = compute_path_invariance(x_curve, y_curve)
                if (
                    x_position.shape == y_position.shape
                    and np.array_equal(x_position, y_position, equal_nan=True)
                    and score["path_invariance_status"] == "valid"
                ):
                    from v1ca1.task_progression.similarity import interpolate_nans

                    x_curve = np.maximum(
                        interpolate_nans(np.asarray(x_curve, dtype=float)),
                        0.0,
                    )
                    y_curve = np.maximum(
                        interpolate_nans(np.asarray(y_curve, dtype=float)),
                        0.0,
                    )
                    common_scale = max(
                        float(np.max(x_curve)),
                        float(np.max(y_curve)),
                    )
                    if common_scale > 0.0:
                        x_curve = x_curve / common_scale
                        y_curve = y_curve / common_scale
                    return (
                        np.linspace(0.0, 1.0, x_curve.size),
                        x_curve,
                        y_curve,
                        trajectories[0],
                        trajectories[1],
                    )

    position = np.linspace(0.0, 1.0, 81)
    x_curve = np.exp(-0.5 * ((position - 0.43) / 0.12) ** 2)
    y_curve = 0.68 * np.exp(-0.5 * ((position - 0.49) / 0.15) ** 2)
    x_path, y_path = PANEL_C_TURN_PATHS["left"]
    return position, x_curve, y_curve, x_path, y_path


def plot_panel_b_tuning_similarity_with_schematic(
    ax: Any,
    table: Any,
    *,
    example: dict[str, Any] | None,
    split_half_table: Any | None = None,
    null_table: Any | None = None,
) -> None:
    """Plot dark-light stability with split-half and shuffled references."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes((0.015, 0.34, 0.260, 0.45))
    schematic_ax.set_in_layout(False)
    position, dark, light = _normalized_schematic_curves(example)
    overlap = np.minimum(dark, light)
    schematic_ax.fill_between(
        position,
        0.0,
        overlap,
        color="#BDBDBD",
        alpha=0.55,
        linewidth=0.0,
        label="Overlap",
    )
    schematic_ax.plot(position, dark, color="#252525", linewidth=0.9, label="Dark")
    schematic_ax.plot(position, light, color="#E6AB02", linewidth=0.9, label="Light")
    if example is not None and "animal_name" in example:
        schematic_segment_edges = get_wtrack_segment_edges(
            str(example["animal_name"])
        )
    else:
        schematic_segment_edges = np.asarray([0.0, 0.4, 0.6, 1.0])
    for boundary in schematic_segment_edges[1:-1]:
        schematic_ax.axvline(
            float(boundary),
            color="0.55",
            linestyle=(0.0, (1.5, 1.2)),
            linewidth=0.55,
            zorder=1,
        )
    schematic_ax.set_xlim(0.0, 1.0)
    schematic_ax.set_ylim(bottom=0.0)
    schematic_ax.set_xticks(())
    schematic_ax.set_yticks(())
    schematic_ax.set_title("Segment-wise overlap", fontsize=6.0, pad=1.0)
    schematic_ax.text(
        0.98,
        0.04,
        "z",
        transform=schematic_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
    )
    schematic_ax.spines[["top", "right"]].set_visible(False)
    schematic_ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=5.0,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.0,
    )
    ax.text(
        0.148,
        0.185,
        r"$S=\langle\sum_{z\in s}\min[\hat d_{p,s},\hat l_{p,s}]\rangle_{p,s}$",
        ha="center",
        va="center",
        fontsize=5.7,
    )

    if "tuning_average_index" not in table:
        table = derive_panel_b_tuning_average_table(table)
    rows = table[
        table["passes_unit_qc"].astype(bool)
        & (table["average_status"].astype(str) == "valid")
    ]
    values = np.asarray(rows["tuning_average_index"], dtype=float)
    values = values[np.isfinite(values)]
    weights = (
        np.full(values.shape, 1.0 / values.size, dtype=float)
        if values.size
        else values
    )
    hist_ax = ax.inset_axes((0.390, 0.340, 0.585, 0.500))
    hist_ax.set_in_layout(False)
    counts, _edges, _patches = hist_ax.hist(
        values,
        bins=PANEL_B_TUNING_SIMILARITY_BINS,
        weights=weights,
        color="0.35",
        label="Dark–light",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    split_half_values = np.asarray([], dtype=float)
    split_half_counts = np.asarray([], dtype=float)
    if split_half_table is not None:
        split_half_rows = split_half_table[
            split_half_table["passes_unit_qc"].astype(bool)
            & (split_half_table["average_status"].astype(str) == "valid")
        ]
        split_half_values = np.asarray(
            split_half_rows["split_half_tuning_stability_index"],
            dtype=float,
        )
        split_half_values = split_half_values[np.isfinite(split_half_values)]
        split_half_weights = (
            np.full(
                split_half_values.shape,
                1.0 / split_half_values.size,
                dtype=float,
            )
            if split_half_values.size
            else split_half_values
        )
        split_half_counts, _edges, _patches = hist_ax.hist(
            split_half_values,
            bins=PANEL_B_TUNING_SIMILARITY_BINS,
            weights=split_half_weights,
            histtype="step",
            color="#0072B2",
            linewidth=1.1,
            label="Split-half",
        )
    null_values = np.asarray([], dtype=float)
    null_counts = np.asarray([], dtype=float)
    if null_table is not None:
        null_rows = null_table[
            null_table["passes_unit_qc"].astype(bool)
            & (null_table["null_status"].astype(str) == "valid")
        ].copy()
        null_rows["minimum_null_tuning_stability_index"] = np.asarray(
            null_rows["minimum_null_tuning_stability_index"], dtype=float
        )
        null_rows = null_rows[
            np.isfinite(null_rows["minimum_null_tuning_stability_index"])
        ]
        null_values = np.asarray(
            null_rows["minimum_null_tuning_stability_index"],
            dtype=float,
        )
        null_shifts = int(
            np.sum(np.asarray(null_rows["n_circular_shifts"], dtype=int))
        )
        null_values = null_values[np.isfinite(null_values)]
        null_weights = (
            np.full(null_values.shape, 1.0 / null_values.size, dtype=float)
            if null_values.size
            else null_values
        )
        null_counts, _edges, _patches = hist_ax.hist(
            null_values,
            bins=PANEL_B_TUNING_SIMILARITY_BINS,
            weights=null_weights,
            histtype="step",
            color="#D55E00",
            linestyle=(0.0, (2.4, 1.5)),
            linewidth=1.1,
            label="Circular-shift minimum",
        )
    maximum_fraction = max(
        float(np.nanmax(counts)) if counts.size else 0.0,
        (
            float(np.nanmax(split_half_counts))
            if split_half_counts.size
            else 0.0
        ),
        float(np.nanmax(null_counts)) if null_counts.size else 0.0,
    )
    y_max = max(0.10, 1.40 * maximum_fraction)
    median = float(np.median(values)) if values.size else float("nan")
    split_half_median = (
        float(np.median(split_half_values))
        if split_half_values.size
        else float("nan")
    )
    null_distribution_median = (
        float(np.median(null_values)) if null_values.size else float("nan")
    )
    summary_lines = [
        (
            f"Dark–light: n={values.size}, med={median:.2f}"
            if np.isfinite(median)
            else "Dark–light: n=0"
        )
    ]
    if split_half_table is not None:
        split_half_summary = (
            f"med={split_half_median:.2f}"
            if np.isfinite(split_half_median)
            else "med=–"
        )
        summary_lines.append(
            f"Split-half: n={split_half_values.size}, {split_half_summary}"
        )
    if null_table is not None:
        null_summary = (
            f"med={null_distribution_median:.2f}"
            if np.isfinite(null_distribution_median)
            else "med=–"
        )
        summary_lines.append(
            f"Floor: n={null_values.size} neurons, {null_shifts:,} exact shifts, "
            f"{null_summary}"
        )
    hist_ax.text(
        0.04,
        0.96,
        "\n".join(summary_lines),
        transform=hist_ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.6,
        linespacing=1.02,
    )
    hist_ax.set_xlim(0.0, 1.0)
    hist_ax.set_ylim(0.0, y_max)
    hist_ax.set_xticks((0.0, 0.5, 1.0))
    hist_ax.set_yticks((0.0, y_max))
    hist_ax.set_yticklabels(("0", f"{y_max:.2f}"))
    hist_ax.set_xlabel("Mean tuning stability index", fontsize=6.0, labelpad=1.0)
    hist_ax.set_ylabel("Fraction of neurons", fontsize=6.0, labelpad=1.0)
    hist_ax.spines[["top", "right"]].set_visible(False)
    hist_ax.tick_params(labelsize=5.5, pad=0.7)
    if split_half_table is not None or null_table is not None:
        legend_handles, legend_labels = hist_ax.get_legend_handles_labels()
        if len(legend_handles) == 3:
            legend_order = (0, 2, 1)
            legend_handles = [legend_handles[index] for index in legend_order]
            legend_labels = [legend_labels[index] for index in legend_order]
        hist_ax.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncols=2,
            frameon=False,
            fontsize=4.3,
            handlelength=0.8,
            handletextpad=0.2,
            columnspacing=0.35,
            borderpad=0.0,
        )


def plot_panel_d_achievable_stability(ax: Any, table: Any) -> None:
    """Plot the per-neuron fraction of achievable tuning stability."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.165,
        0.61,
        r"$A=\dfrac{S_{\rm dark-light}-S_{\rm floor}}"
        r"{S_{\rm split-half}-S_{\rm floor}}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
    )
    ax.text(
        0.165,
        0.34,
        "0 = circular-shift floor\n1 = split-half level",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.5,
        linespacing=1.15,
    )
    valid = table[table["achievable_status"].astype(str) == "valid"]
    values = np.asarray(valid["achievable_stability"], dtype=float)
    values = values[np.isfinite(values)]
    default_lower = float(PANEL_D_ACHIEVABLE_STABILITY_BINS[0])
    default_upper = float(PANEL_D_ACHIEVABLE_STABILITY_BINS[-1])
    lower = min(default_lower, float(values.min())) if values.size else default_lower
    upper = max(default_upper, float(values.max())) if values.size else default_upper
    padding = max(0.05, 0.02 * (upper - lower))
    bins = np.linspace(lower - padding, upper + padding, 26)
    weights = (
        np.full(values.shape, 1.0 / values.size, dtype=float)
        if values.size
        else values
    )
    hist_ax = ax.inset_axes((0.385, 0.20, 0.590, 0.64))
    hist_ax.set_in_layout(False)
    counts, _edges, _patches = hist_ax.hist(
        values,
        bins=bins,
        weights=weights,
        color="0.35",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    hist_ax.axvline(0.0, color="#D55E00", linewidth=0.8, linestyle=(0, (2, 2)))
    hist_ax.axvline(1.0, color="#0072B2", linewidth=0.8, linestyle=(0, (2, 2)))
    maximum_fraction = float(np.max(counts)) if counts.size else 0.0
    hist_ax.set_ylim(0.0, max(0.10, 1.25 * maximum_fraction))
    hist_ax.set_xlim(float(bins[0]), float(bins[-1]))
    hist_ax.set_xlabel("Achievable tuning stability", fontsize=6.0, labelpad=1.0)
    hist_ax.set_ylabel("Fraction of neurons", fontsize=6.0, labelpad=1.0)
    hist_ax.spines[["top", "right"]].set_visible(False)
    hist_ax.tick_params(labelsize=5.5, pad=0.7)
    median = float(np.median(values)) if values.size else float("nan")
    below_zero = int(np.sum(values < 0.0))
    above_one = int(np.sum(values > 1.0))
    summary = (
        f"n={values.size}, med={median:.2f}\n"
        f"<0: {below_zero}; >1: {above_one}"
        if np.isfinite(median)
        else "n=0"
    )
    hist_ax.text(
        0.98,
        0.95,
        summary,
        transform=hist_ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
    )
    # Mirror the quantitative bounds and labels on the container for callers
    # that inspect the public panel axis rather than its inset histogram.
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_xlabel("Achievable tuning stability")
    ax.set_ylabel("Fraction of neurons")


def plot_full_path_achievable_stability(ax: Any, table: Any) -> None:
    """Plot whole-path achievable stability on a shared fixed scale."""
    ax.text(
        0.5,
        0.98,
        r"$A=(S_{\rm dark-light}-S_{\rm floor})/"
        r"(S_{\rm split-half}-S_{\rm floor})$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.5,
    )
    valid = table[table["achievable_status"].astype(str) == "valid"]
    raw_values = np.asarray(valid["achievable_stability"], dtype=float)
    raw_values = raw_values[np.isfinite(raw_values)]
    lower = float(PANEL_D_ACHIEVABLE_STABILITY_BINS[0])
    upper = float(PANEL_D_ACHIEVABLE_STABILITY_BINS[-1])
    display_values = raw_values[
        (raw_values >= lower) & (raw_values <= upper)
    ]
    weights = (
        np.full(display_values.shape, 1.0 / raw_values.size, dtype=float)
        if raw_values.size
        else display_values
    )
    counts, _edges, _patches = ax.hist(
        display_values,
        bins=PANEL_D_ACHIEVABLE_STABILITY_BINS,
        weights=weights,
        color="0.35",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    ax.axvline(0.0, color="#D55E00", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.axvline(1.0, color="#0072B2", linewidth=0.8, linestyle=(0, (2, 2)))
    maximum_fraction = float(np.max(counts)) if counts.size else 0.0
    ax.set_xlim(lower, upper)
    ax.set_ylim(0.0, max(0.10, 1.25 * maximum_fraction))
    ax.set_xlabel("Achievable whole-path stability", fontsize=6.0, labelpad=1.0)
    ax.set_ylabel("Fraction of neurons", fontsize=6.0, labelpad=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.5, pad=0.7)
    median = float(np.median(raw_values)) if raw_values.size else float("nan")
    below = int(np.sum(raw_values < lower))
    above = int(np.sum(raw_values > upper))
    nonpositive = int(
        np.sum(table["achievable_status"].astype(str) == "nonpositive_denominator")
    )
    summary = (
        f"n={raw_values.size}, med={median:.2f}\n"
        f"<{lower:g}: {below}; >{upper:g}: {above}\n"
        f"nonpositive denom.: {nonpositive}"
        if np.isfinite(median)
        else f"n=0\nnonpositive denom.: {nonpositive}"
    )
    ax.text(
        0.98,
        0.88,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
    )


def plot_segment_matched_achievable_stability(ax: Any, table: Any) -> None:
    """Plot coherent segment-matched achievable stability per neuron."""
    plot_full_path_achievable_stability(ax, table)
    ax.set_xlabel(
        "Achievable segment-matched stability",
        fontsize=6.0,
        labelpad=1.0,
    )


def plot_tuning_correlation_references(
    ax: Any,
    observed_table: Any,
    split_table: Any,
    null_table: Any,
    *,
    variant: str,
) -> None:
    """Plot dark-light Pearson r with whole-path ceiling and null."""
    variant = _validate_tuning_correlation_variant(variant)
    observed_rows = observed_table[
        observed_table["passes_unit_qc"].astype(bool)
        & (observed_table["average_status"].astype(str) == "valid")
    ]
    observed = np.asarray(
        observed_rows["mean_tuning_correlation"],
        dtype=float,
    )
    observed = observed[np.isfinite(observed)]
    split_rows = split_table[
        split_table["passes_unit_qc"].astype(bool)
        & (split_table["reference_status"].astype(str) == "valid")
    ]
    split = np.asarray(
        split_rows["split_half_tuning_correlation"],
        dtype=float,
    )
    split = split[np.isfinite(split)]
    null_rows = null_table[
        null_table["passes_unit_qc"].astype(bool)
        & (null_table["null_status"].astype(str) == "valid")
    ].copy()
    null_rows["null_tuning_correlation"] = np.asarray(
        null_rows["null_tuning_correlation"],
        dtype=float,
    )
    null_rows = null_rows[np.isfinite(null_rows["null_tuning_correlation"])]
    key_columns = [
        column
        for column in _correlation_key_columns()
        if column in null_rows
    ]
    null = (
        np.asarray(
            null_rows.groupby(key_columns, sort=False)[
                "null_tuning_correlation"
            ].mean(),
            dtype=float,
        )
        if key_columns
        else np.asarray(null_rows["null_tuning_correlation"], dtype=float)
    )
    n_shifts = (
        int(null_rows["permutation"].nunique())
        if "permutation" in null_rows
        else 0
    )
    counts: list[np.ndarray] = []
    for values, color, label, histtype, linestyle in (
        (observed, "0.35", "Dark–light", "bar", "solid"),
        (split, "#0072B2", "Split-half", "step", "solid"),
        (null, "#D55E00", "Circular null", "step", (0.0, (2.4, 1.5))),
    ):
        weights = (
            np.full(values.shape, 1.0 / values.size, dtype=float)
            if values.size
            else values
        )
        kwargs: dict[str, Any] = {
            "bins": TUNING_CORRELATION_BINS,
            "weights": weights,
            "color": color,
            "label": label,
            "histtype": histtype,
        }
        if histtype == "bar":
            kwargs.update(EMPHASIS_HISTOGRAM_KWARGS)
        else:
            kwargs.update({"linewidth": 1.1, "linestyle": linestyle})
        histogram, _edges, _patches = ax.hist(values, **kwargs)
        counts.append(histogram)
    maximum = max(
        (float(np.max(count)) if count.size else 0.0 for count in counts),
        default=0.0,
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, max(0.10, 1.35 * maximum))
    ax.set_xticks((-1.0, 0.0, 1.0))
    ax.set_xlabel("Mean dark–light tuning correlation")
    ax.set_ylabel("Fraction of neurons")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0.0, color="0.65", linewidth=0.5, zorder=0)
    ax.legend(
        frameon=False,
        fontsize=5.0,
        ncols=1,
        loc="upper left",
        handlelength=1.2,
        handletextpad=0.3,
        borderpad=0.0,
    )
    resolution = "Whole path" if variant == "whole_path" else "Segments"
    medians = [
        float(np.median(values)) if values.size else float("nan")
        for values in (observed, split, null)
    ]
    ax.text(
        0.98,
        0.96,
        f"{resolution}\n"
        f"Dark–light: n={observed.size}, med={medians[0]:.2f}\n"
        f"Split-half: n={split.size}, med={medians[1]:.2f}\n"
        f"Null: n={null.size} neurons, {n_shifts:,} shifts, "
        f"med={medians[2]:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )


def plot_achievable_tuning_correlation(
    ax: Any,
    table: Any,
    *,
    variant: str,
) -> None:
    """Plot correlation relative to its matched whole-path reference range."""
    variant = _validate_tuning_correlation_variant(variant)
    rows = table[table["achievable_status"].astype(str) == "valid"]
    values = np.asarray(rows["achievable_tuning_correlation"], dtype=float)
    values = values[np.isfinite(values)]
    lower = min(-1.0, float(values.min())) if values.size else -1.0
    upper = max(1.5, float(values.max())) if values.size else 1.5
    padding = max(0.05, 0.02 * (upper - lower))
    bins = np.linspace(lower - padding, upper + padding, 26)
    weights = (
        np.full(values.shape, 1.0 / values.size, dtype=float)
        if values.size
        else values
    )
    counts, _edges, _patches = ax.hist(
        values,
        bins=bins,
        weights=weights,
        color="0.35",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    ax.axvline(0.0, color="#D55E00", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.axvline(1.0, color="#0072B2", linewidth=0.8, linestyle=(0, (2, 2)))
    maximum = float(np.max(counts)) if counts.size else 0.0
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_ylim(0.0, max(0.10, 1.25 * maximum))
    ax.set_xlabel("Achievable tuning correlation")
    ax.set_ylabel("Fraction of neurons")
    ax.spines[["top", "right"]].set_visible(False)
    median = float(np.median(values)) if values.size else float("nan")
    resolution = "Whole path" if variant == "whole_path" else "Segments"
    ax.text(
        0.02,
        0.95,
        f"{resolution}\nn={values.size}, med={median:.2f}\n"
        f"<0: {int(np.sum(values < 0.0))}; >1: {int(np.sum(values > 1.0))}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )


def plot_circular_shift_schematic(ax: Any) -> None:
    """Illustrate normalized overlap across exact circular shifts."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    position = np.linspace(0.0, 1.0, 80, endpoint=False)
    dark = np.exp(-0.5 * ((position - 0.30) / 0.09) ** 2)
    dark /= float(np.sum(dark))
    light = np.exp(-0.5 * ((position - 0.30) / 0.15) ** 2)
    light /= float(np.sum(light))
    display_scale = 0.10 / float(np.max(dark))
    line_colors = {"dark": "#252525", "light": "#E6AB02"}
    for row_y, shift_fraction, shift_label, label_y in (
        (0.617, 0.0, r"$\Delta=0$", 0.680),
        (0.384, 0.25, r"$\Delta=+1/4$", 0.490),
        (0.180, 0.50, r"$\Delta=+1/2$", 0.280),
    ):
        shifted_light = np.roll(
            light,
            int(round(shift_fraction * light.size)),
        )
        dark_y = row_y + display_scale * dark
        light_y = row_y + display_scale * shifted_light
        ax.fill_between(
            position,
            row_y,
            np.minimum(dark_y, light_y),
            color="#BDBDBD",
            alpha=0.55,
            linewidth=0.0,
        )
        ax.plot(position, dark_y, color=line_colors["dark"], linewidth=0.8)
        ax.plot(position, light_y, color=line_colors["light"], linewidth=0.8)
        ax.plot((0.0, 1.0), (row_y, row_y), color="0.72", linewidth=0.45)
        ax.text(
            0.97,
            label_y,
            shift_label,
            ha="right",
            va="center",
            fontsize=5.4,
        )

    ax.plot((0.02, 0.10), (0.97, 0.97), color=line_colors["dark"], linewidth=0.9)
    ax.text(
        0.13,
        0.97,
        "Norm. dark tuning",
        ha="left",
        va="center",
        fontsize=4.8,
    )
    ax.plot((0.02, 0.10), (0.89, 0.89), color=line_colors["light"], linewidth=0.9)
    ax.text(
        0.13,
        0.89,
        "Norm. light tuning",
        ha="left",
        va="center",
        fontsize=4.8,
    )
    ax.fill(
        (0.67, 0.73, 0.73, 0.67),
        (0.795, 0.795, 0.825, 0.825),
        color="#BDBDBD",
        alpha=0.55,
        linewidth=0.0,
    )
    ax.text(
        0.76,
        0.81,
        "Overlap",
        ha="left",
        va="center",
        fontsize=4.8,
    )
    for arrow_start, arrow_end in ((0.00, 0.23), (0.77, 1.00)):
        ax.annotate(
            "",
            xy=(arrow_end, 0.035),
            xytext=(arrow_start, 0.035),
            arrowprops={
                "arrowstyle": "->",
                "color": line_colors["light"],
                "linewidth": 0.75,
            },
        )
    ax.text(
        0.50,
        0.035,
        "Circular shift",
        ha="center",
        va="center",
        fontsize=4.3,
    )


def plot_panel_h_shift_profiles(axes: Sequence[Any], table: Any) -> None:
    """Plot median and IQR split-half-rescaled overlap profiles by path."""
    if len(axes) != len(PANEL_B_PATH_ORDER):
        raise ValueError("Panel H requires one axis for each task path.")
    profile_keys = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
        "path",
    )
    path_summaries: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    path_rows: list[Any] = []
    for path in PANEL_B_PATH_ORDER:
        rows = table[
            (table["path"].astype(str) == str(path))
            & (table["rescaling_status"].astype(str) == "valid")
        ].copy()
        rows = rows[
            np.isfinite(np.asarray(rows["normalized_shift"], dtype=float))
            & np.isfinite(np.asarray(rows["rescaled_overlap"], dtype=float))
        ]
        grouped = rows.groupby("normalized_shift", sort=True)[
            "rescaled_overlap"
        ]
        median_summary = grouped.median()
        shifts = np.asarray(median_summary.index, dtype=float)
        median = np.asarray(median_summary, dtype=float)
        lower = np.asarray(grouped.quantile(0.25), dtype=float)
        upper = np.asarray(grouped.quantile(0.75), dtype=float)
        path_rows.append(rows)
        path_summaries.append((shifts, median, lower, upper))

    finite_summary_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for _shifts, median, lower, upper in path_summaries
            for values in (median, lower, upper)
            if values.size
        ]
        + [np.asarray([0.0, 1.0])]
    )
    y_max = float(np.max(finite_summary_values))
    y_limits = (0.0, max(1.05, y_max + 0.05))

    for index, (axis, path, rows, summary) in enumerate(
        zip(
            axes,
            PANEL_B_PATH_ORDER,
            path_rows,
            path_summaries,
            strict=True,
        )
    ):
        shifts, median, lower, upper = summary
        color = _figure_2.PANEL_TRAJECTORY_COLORS[path]
        if shifts.size:
            axis.fill_between(
                shifts,
                lower,
                upper,
                color=color,
                alpha=0.20,
                linewidth=0.0,
                label="IQR",
            )
            axis.plot(
                shifts,
                median,
                color=color,
                linewidth=1.15,
                label="Median",
            )
        axis.axvline(
            0.0,
            color="0.55",
            linewidth=0.6,
            linestyle=(0, (2, 2)),
            zorder=0,
        )
        axis.axhline(0.0, color="0.70", linewidth=0.55, zorder=0)
        axis.axhline(
            1.0,
            color="0.55",
            linewidth=0.6,
            linestyle=(0, (2, 2)),
            zorder=0,
        )
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(*y_limits)
        axis.set_xticks((-0.5, 0.0, 0.5))
        axis.set_xlabel("Normalized circular shift", fontsize=5.8, labelpad=1.0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=5.5, pad=0.7)
        n_profiles = (
            int(rows.loc[:, list(profile_keys)].drop_duplicates().shape[0])
            if not rows.empty
            else 0
        )
        axis.set_title(
            f"{PANEL_B_PATH_LABELS[path]}  (n={n_profiles:,})",
            fontsize=6.2,
            pad=2.0,
        )
        if index == 0:
            axis.set_ylabel("Rescaled overlap", fontsize=5.8, labelpad=1.0)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(
                    handles[::-1],
                    labels[::-1],
                    loc="lower left",
                    ncols=2,
                    frameon=False,
                    fontsize=5.0,
                    handlelength=1.1,
                    columnspacing=0.7,
                    borderpad=0.0,
                )
        else:
            axis.tick_params(axis="y", labelleft=False)


def plot_population_shift_profile(ax: Any, table: Any) -> None:
    """Plot an equal-neuron mean after averaging each neuron's valid paths."""
    neuron_keys = (
        "animal_name",
        "date",
        "region",
        "unit",
        "dark_epoch",
        "light_epoch",
    )
    required = (
        *neuron_keys,
        "path",
        "normalized_shift",
        "rescaled_overlap",
        "rescaling_status",
    )
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(
            f"Population shift-profile table is missing columns {missing!r}."
        )
    rows = table[
        table["path"].astype(str).isin(PANEL_B_PATH_ORDER)
        & (table["rescaling_status"].astype(str) == "valid")
    ].copy()
    rows = rows[
        np.isfinite(np.asarray(rows["normalized_shift"], dtype=float))
        & np.isfinite(np.asarray(rows["rescaled_overlap"], dtype=float))
    ]

    neuron_profiles = (
        rows.groupby(
            [*neuron_keys, "normalized_shift"],
            sort=True,
            as_index=False,
        )["rescaled_overlap"]
        .mean()
    )
    grouped = neuron_profiles.groupby("normalized_shift", sort=True)[
        "rescaled_overlap"
    ]
    mean_summary = grouped.mean()
    shifts = np.asarray(mean_summary.index, dtype=float)
    mean = np.asarray(mean_summary, dtype=float)
    lower = np.asarray(grouped.quantile(0.25), dtype=float)
    upper = np.asarray(grouped.quantile(0.75), dtype=float)
    zero_values = np.asarray(
        neuron_profiles.loc[
            np.isclose(
                np.asarray(
                    neuron_profiles["normalized_shift"],
                    dtype=float,
                ),
                0.0,
                rtol=0.0,
                atol=1e-12,
            ),
            "rescaled_overlap",
        ],
        dtype=float,
    )
    if zero_values.size:
        zero_median = float(np.median(zero_values))
        zero_q1, zero_q3 = np.quantile(zero_values, (0.25, 0.75))
        zero_summary = (
            f"Δ=0: med. {zero_median:.2f}\n"
            f"IQR {float(zero_q1):.2f}–{float(zero_q3):.2f}"
        )
    else:
        zero_summary = "Δ=0: med. n/a\nIQR n/a"

    if shifts.size:
        ax.fill_between(
            shifts,
            lower,
            upper,
            color="#7FA9CF",
            alpha=0.24,
            linewidth=0.0,
            label="IQR across neurons",
        )
        ax.plot(
            shifts,
            mean,
            color="0.22",
            linewidth=1.2,
            label="Mean across neurons",
        )
    ax.axvline(
        0.0,
        color="0.55",
        linewidth=0.6,
        linestyle=(0, (2, 2)),
        zorder=0,
    )
    ax.axhline(0.0, color="0.70", linewidth=0.55, zorder=0)
    ax.axhline(
        1.0,
        color="0.55",
        linewidth=0.6,
        linestyle=(0, (2, 2)),
        zorder=0,
    )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks((-0.5, -0.25, 0.0, 0.25, 0.5))
    ax.set_yticks((0.0, 0.5, 1.0))
    ax.set_xlabel(
        "Circular shift",
        fontsize=5.8,
        labelpad=1.0,
    )
    ax.set_ylabel("Norm. overlap", fontsize=5.8, labelpad=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=5.5, pad=0.7)

    ax.text(
        0.05,
        0.94,
        zero_summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
        linespacing=1.05,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
            "pad": 0.4,
        },
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        order = [
            labels.index("Mean across neurons"),
            labels.index("IQR across neurons"),
        ]
        ax.legend(
            [handles[index] for index in order],
            [labels[index] for index in order],
            loc="upper right",
            frameon=False,
            fontsize=5.2,
            handlelength=1.2,
            borderpad=0.0,
        )


def plot_raw_circular_shift_profiles(axes: Sequence[Any], table: Any) -> None:
    """Plot pre-rescaling unit-area overlap for the Panel B cohort."""
    raw_table = table.copy()
    raw_table["rescaled_overlap"] = np.asarray(
        raw_table["overlap"],
        dtype=float,
    )
    plot_panel_h_shift_profiles(axes, raw_table)
    for axis in axes:
        axis.set_ylim(0.0, 1.05)
        axis.set_yticks((0.0, 0.5, 1.0))
    axes[0].set_ylabel("Unit-area overlap", fontsize=5.8, labelpad=1.0)


def _add_relative_panel_axis(
    parent_ax: Any,
    bounds: tuple[float, float, float, float],
    *,
    label: str,
) -> Any:
    """Add a figure axis whose bounds remain relative to a parent axis."""
    from matplotlib.axes._base import _TransformedBoundsLocator

    figure = parent_ax.get_figure(root=False)
    locator = _TransformedBoundsLocator(bounds, parent_ax.transAxes)
    child_ax = figure.add_axes(locator(parent_ax, None).bounds, label=label)
    child_ax.set_axes_locator(locator)
    child_ax.set_in_layout(False)
    return child_ax


def plot_panel_b_location_gain_schematic(ax: Any) -> None:
    """Illustrate exact gain normalization and two schematic manipulations."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.50,
        0.90,
        r"For fixed bins, $r(z)\geq0$, and $G>0$:  $r(z)=G\,p(z)$,   "
        r"$G=\sum_z r(z)$,   $\sum_z p(z)=1$",
        ha="center",
        va="center",
        fontsize=6.0,
    )
    ax.text(
        0.25,
        0.75,
        r"Preferred location, $\mu$",
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
    )
    ax.text(
        0.75,
        0.75,
        r"Response scale / gain, $G$",
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
    )
    ax.plot(
        (0.50, 0.50),
        (0.19, 0.78),
        color="0.86",
        linewidth=0.55,
        transform=ax.transAxes,
        clip_on=False,
    )

    location_ax = _add_relative_panel_axis(
        ax,
        (0.055, 0.265, 0.390, 0.385),
        label="panel_b_location_shift",
    )
    gain_ax = _add_relative_panel_axis(
        ax,
        (0.555, 0.265, 0.390, 0.385),
        label="panel_b_gain_scaling",
    )

    position = np.linspace(0.0, 1.0, 201)
    bin_offsets = np.arange(position.size) - position.size // 2
    shape = np.exp(-0.5 * (bin_offsets / 16.0) ** 2)
    shape[np.abs(bin_offsets) > 48] = 0.0
    shape /= float(shape.sum())

    shift_bins = 35
    first_shape = np.roll(shape, -shift_bins)
    second_shape = np.roll(shape, shift_bins)
    fixed_gain = 1.0
    first_rate = fixed_gain * first_shape
    second_rate = fixed_gain * second_shape
    first_mu = float(position[int(np.argmax(first_rate))])
    second_mu = float(position[int(np.argmax(second_rate))])
    location_peak = float(first_rate.max())
    location_ax.plot(
        position,
        first_rate,
        color="#0072B2",
        linewidth=1.05,
        label=r"$r(z),\ \mu_1$",
    )
    location_ax.plot(
        position,
        second_rate,
        color="#D55E00",
        linewidth=1.05,
        label=r"$r(z),\ \mu_2$",
    )
    for center, color in (
        (first_mu, "#0072B2"),
        (second_mu, "#D55E00"),
    ):
        location_ax.vlines(
            center,
            0.0,
            location_peak,
            colors=color,
            linestyles=(0, (2, 2)),
            linewidth=0.55,
        )
    arrow_y = 1.11 * location_peak
    location_ax.annotate(
        "",
        xy=(second_mu, arrow_y),
        xytext=(first_mu, arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "0.30", "lw": 0.65},
    )
    location_ax.text(
        0.5 * (first_mu + second_mu),
        1.14 * location_peak,
        r"$\Delta\mu$",
        ha="center",
        va="bottom",
        fontsize=5.2,
    )
    location_ax.text(
        0.02,
        0.95,
        r"$\mu=\arg\max_z p(z)$",
        transform=location_ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.0,
    )

    low_gain = 0.65
    high_gain = 1.25
    low_rate = low_gain * shape
    high_rate = high_gain * shape
    common_mu = float(position[int(np.argmax(shape))])
    gain_ax.plot(
        position,
        low_rate,
        color="#7FA9CF",
        linewidth=1.05,
        label=r"$G_1p(z)$",
    )
    gain_ax.plot(
        position,
        high_rate,
        color="#1F5A85",
        linewidth=1.05,
        label=r"$G_2p(z)$",
    )
    gain_ax.vlines(
        common_mu,
        0.0,
        float(high_rate.max()),
        colors="0.35",
        linestyles=(0, (2, 2)),
        linewidth=0.55,
    )
    arrow_x = common_mu + 0.10
    gain_ax.annotate(
        "",
        xy=(arrow_x, float(high_rate.max())),
        xytext=(arrow_x, float(low_rate.max())),
        arrowprops={"arrowstyle": "<->", "color": "0.30", "lw": 0.65},
    )
    gain_ax.text(
        arrow_x + 0.025,
        0.5 * float(low_rate.max() + high_rate.max()),
        r"amplitude $\propto G$",
        ha="left",
        va="center",
        fontsize=5.0,
    )

    shared_y_max = 1.38 * float(high_rate.max())
    for child_ax in (location_ax, gain_ax):
        child_ax.set_xlim(0.0, 1.0)
        child_ax.set_ylim(0.0, shared_y_max)
        child_ax.set_xticks(())
        child_ax.set_yticks(())
        child_ax.text(
            0.985,
            0.035,
            r"$z$",
            transform=child_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.2,
        )
        child_ax.set_ylabel("Response, r(z)", fontsize=5.2, labelpad=0.8)
        child_ax.spines[["top", "right", "left"]].set_visible(False)
        child_ax.spines["bottom"].set_linewidth(0.55)
        child_ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=5.0,
            ncols=2,
            handlelength=1.2,
            columnspacing=0.8,
            handletextpad=0.3,
            borderpad=0.0,
        )

    ax.text(
        0.25,
        0.205,
        r"Same shape and $G$; horizontal displacement",
        ha="center",
        va="center",
        fontsize=5.2,
    )
    ax.text(
        0.75,
        0.205,
        r"Same $p(z)$ and $\mu$; amplitude $\propto G$",
        ha="center",
        va="center",
        fontsize=5.2,
    )
    ax.text(
        0.50,
        0.105,
        "Components are held fixed only schematically. Circular-shift profiles (C) "
        r"probe alignment of $p(z)$;",
        ha="center",
        va="center",
        fontsize=5.0,
    )
    ax.text(
        0.50,
        0.050,
        "raw-rate DPPI (D) jointly reflects location, shape, and relative gain.",
        ha="center",
        va="center",
        fontsize=5.0,
    )


def plot_panel_b_dark_same_turn_schematic(ax: Any) -> None:
    """Show path invariance and light-dependent gain in a 2-by-2 schematic."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    center_to_left = "center_to_left"
    right_to_center = "right_to_center"
    first_color = _figure_2.PANEL_TRAJECTORY_COLORS[center_to_left]
    second_color = _figure_2.PANEL_TRAJECTORY_COLORS[right_to_center]
    dark_color = "#252525"
    light_color = "#E6AB02"
    progression = np.linspace(0.0, 1.0, 201)
    tuning_shape = np.exp(-0.5 * ((progression - 0.56) / 0.13) ** 2)
    tuning_shape[tuning_shape < 1e-4] = 0.0
    preferred_progression = 0.56

    panel_specs = (
        (
            "dark",
            (0.18, 0.52, 0.35, 0.25),
            "panel_b_tuning_dark_center_to_left",
            dark_color,
            "Dark C→L",
            0.80,
        ),
        (
            "dark",
            (0.61, 0.52, 0.35, 0.25),
            "panel_b_tuning_dark_right_to_center",
            dark_color,
            "Dark R→C",
            0.80,
        ),
        (
            "light",
            (0.18, 0.20, 0.35, 0.25),
            "panel_b_tuning_light_center_to_left",
            light_color,
            "Light C→L",
            0.35,
        ),
        (
            "light",
            (0.61, 0.20, 0.35, 0.25),
            "panel_b_tuning_light_right_to_center",
            light_color,
            "Light R→C",
            1.15,
        ),
    )
    for condition, bounds, label, color, curve_label, gain in panel_specs:
        tuning_ax = _add_relative_panel_axis(ax, bounds, label=label)
        tuning_ax.plot(
            progression,
            gain * tuning_shape,
            color=color,
            linewidth=1.0,
            label=curve_label,
        )
        tuning_ax.axvline(
            preferred_progression,
            color="0.30",
            linestyle=(0, (2, 2)),
            linewidth=0.55,
        )
        if label == "panel_b_tuning_light_center_to_left":
            tuning_ax.annotate(
                "",
                xy=(0.79, 0.58),
                xytext=(0.33, 0.58),
                arrowprops={
                    "arrowstyle": "<->",
                    "color": "#E6AB02",
                    "linewidth": 0.75,
                    "mutation_scale": 6,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=5,
            )
        tuning_ax.set_xlim(0.0, 1.0)
        tuning_ax.set_ylim(0.0, 1.35)
        tuning_ax.set_yticks(())
        if condition == "light":
            tuning_ax.set_xticks((0.0, 1.0), labels=("", ""))
            tuning_ax.tick_params(axis="x", labelsize=4.7, pad=0.4)
        else:
            tuning_ax.set_xticks(())
        tuning_ax.spines[["top", "right", "left"]].set_visible(False)
        tuning_ax.spines["bottom"].set_linewidth(0.5)

    for column_x, label, color in (
        (0.355, "C→L", first_color),
        (0.785, "R→C", second_color),
    ):
        ax.text(
            column_x,
            0.79,
            label,
            color=color,
            ha="center",
            va="bottom",
            fontsize=5.2,
            fontweight="bold",
        )
    ax.plot(
        (0.355, 0.355, 0.785, 0.785),
        (0.825, 0.870, 0.870, 0.825),
        color="0.35",
        linewidth=0.6,
        solid_capstyle="butt",
        clip_on=False,
    )
    for row_y, label in ((0.645, "Dark"), (0.325, "Light")):
        ax.text(
            0.145,
            row_y,
            label,
            ha="right",
            va="center",
            fontsize=5.2,
            fontweight="bold",
        )
    ax.text(
        0.57,
        0.125,
        "Normalized path progression",
        ha="center",
        va="center",
        fontsize=4.8,
    )


def _add_shift_profile_row_axes(
    parent_ax: Any,
    *,
    label_prefix: str,
) -> list[Any]:
    """Add four aligned path-profile axes to a full-width panel row."""
    return [
        _add_relative_panel_axis(
            parent_ax,
            (x, 0.250, 0.140, 0.520),
            label=f"{label_prefix}_{index}",
        )
        for index, x in enumerate((0.300, 0.475, 0.650, 0.825))
    ]


def _format_shift_profile_row_axes(profile_axes: Sequence[Any]) -> None:
    """Apply shared-label formatting without adjacent endpoint collisions."""
    tick_label_sets = (
        ("−0.5", "0", ""),
        ("", "0", ""),
        ("", "0", ""),
        ("", "0", "+0.5"),
    )
    for index, (profile_ax, tick_labels) in enumerate(
        zip(profile_axes, tick_label_sets, strict=True)
    ):
        profile_ax.set_xlabel("")
        profile_ax.set_ylabel("")
        profile_ax.set_xticks((-0.5, 0.0, 0.5), labels=tick_labels)
        profile_ax.tick_params(labelsize=4.8, pad=0.4)
        profile_ax.title.set_fontsize(5.2)
        profile_ax.tick_params(axis="y", labelleft=index == 0)
    legend = profile_axes[0].get_legend()
    if legend is not None:
        legend.remove()


def _label_shift_profile_row(ax: Any, *, y_label: str) -> None:
    """Add common summary and axis labels to a shift-profile row."""
    ax.text(
        0.975,
        0.955,
        "Median; shading = IQR",
        ha="right",
        va="top",
        fontsize=4.8,
    )
    ax.text(
        0.6325,
        0.035,
        "Circular shift (fraction of path)",
        ha="center",
        va="bottom",
        fontsize=5.2,
    )
    ax.text(
        0.255,
        0.510,
        y_label,
        rotation=90,
        ha="center",
        va="center",
        fontsize=5.2,
    )


def plot_panel_b_circular_shift_analysis(ax: Any, table: Any) -> None:
    """Plot a shift schematic beside the neuron-weighted population profile."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    group_width = (
        PANEL_B_SCHEMATIC_RELATIVE_WIDTH
        + PANEL_B_COMPONENT_RELATIVE_GAP
        + PANEL_B_PROFILE_RELATIVE_WIDTH
    )
    group_left = 0.5 * (1.0 - group_width)
    schematic_ax = _add_relative_panel_axis(
        ax,
        (
            group_left,
            PANEL_B_SCHEMATIC_RELATIVE_Y,
            PANEL_B_SCHEMATIC_RELATIVE_WIDTH,
            PANEL_B_SCHEMATIC_RELATIVE_HEIGHT,
        ),
        label="panel_b_circular_shift_schematic",
    )
    plot_circular_shift_schematic(schematic_ax)

    profile_ax = _add_relative_panel_axis(
        ax,
        (
            group_left
            + PANEL_B_SCHEMATIC_RELATIVE_WIDTH
            + PANEL_B_COMPONENT_RELATIVE_GAP,
            PANEL_B_PROFILE_RELATIVE_Y,
            PANEL_B_PROFILE_RELATIVE_WIDTH,
            PANEL_B_PROFILE_RELATIVE_HEIGHT,
        ),
        label="panel_b_population_shift_profile",
    )
    plot_population_shift_profile(profile_ax, table)
    profile_ax.set_xticks((-0.5, 0.0, 0.5))
    legend = profile_ax.get_legend()
    if legend is not None:
        legend.remove()


def _align_panel_b_profile_with_panel_c_scatter(
    fig: Any,
    panel_b_axis: Any,
    panel_c_axis: Any,
) -> None:
    """Match B/C plot widths and center Panel B's visual components."""
    figure_axes = tuple(fig.axes)
    panel_b_candidates = (
        *figure_axes,
        *_figure_2._figure_2._iter_nested_axes(panel_b_axis),
    )
    panel_c_descendants = tuple(
        _figure_2._figure_2._iter_nested_axes(panel_c_axis)
    )
    profile_axis = next(
        (
            child_axis
            for child_axis in panel_b_candidates
            if (
                child_axis.get_label() == "panel_b_population_shift_profile"
                or child_axis.get_xlabel() == "Circular shift"
            )
        ),
        None,
    )
    schematic_axis = next(
        (
            child_axis
            for child_axis in panel_b_candidates
            if child_axis.get_label() == "panel_b_circular_shift_schematic"
        ),
        None,
    )
    scatter_axis = next(
        (
            child_axis
            for child_axis in panel_c_descendants
            if child_axis.get_xlabel() in {"Dark DPPI", "Dark PII"}
        ),
        None,
    )
    if profile_axis is None or scatter_axis is None:
        panel_b_labels = [
            (child_axis.get_label(), child_axis.get_xlabel())
            for child_axis in panel_b_candidates
        ]
        panel_c_labels = [
            (child_axis.get_label(), child_axis.get_xlabel())
            for child_axis in panel_c_descendants
        ]
        raise RuntimeError(
            "Could not locate the Panel B profile and Panel C scatter axes; "
            f"B descendants={panel_b_labels!r}, C descendants={panel_c_labels!r}."
        )

    fig.canvas.draw()
    profile_bounds = profile_axis.get_position()
    scatter_bounds = scatter_axis.get_position()
    quantitative_axes = [
        child_axis
        for child_axis in panel_c_descendants
        if (
            child_axis is scatter_axis
            or child_axis.get_xlabel() == "Frac."
            or child_axis.get_ylabel() == "Frac."
        )
    ]
    quantitative_left = min(
        child_axis.get_position().x0 for child_axis in quantitative_axes
    )
    quantitative_right = max(
        child_axis.get_position().x1 for child_axis in quantitative_axes
    )
    panel_b_bounds = panel_b_axis.get_position()
    panel_c_bounds = panel_c_axis.get_position()
    relative_left = (
        quantitative_left - panel_c_bounds.x0
    ) / panel_c_bounds.width
    relative_width = (
        quantitative_right - quantitative_left
    ) / panel_c_bounds.width * PANEL_B_PROFILE_WIDTH_SCALE_FROM_PANEL_C
    profile_relative_left = relative_left
    if schematic_axis is not None:
        schematic_bounds = schematic_axis.get_position()
        schematic_relative_width = (
            schematic_bounds.width / panel_b_bounds.width
        )
        group_width = (
            schematic_relative_width
            + PANEL_B_COMPONENT_RELATIVE_GAP
            + relative_width
        )
        group_left = 0.5 * (1.0 - group_width)
        profile_relative_left = (
            group_left
            + schematic_relative_width
            + PANEL_B_COMPONENT_RELATIVE_GAP
        )
        schematic_axis.set_axes_locator(None)
        schematic_axis.set_position(
            (
                panel_b_bounds.x0 + group_left * panel_b_bounds.width,
                panel_b_bounds.y0
                + PANEL_B_SCHEMATIC_RELATIVE_Y * panel_b_bounds.height,
                schematic_relative_width * panel_b_bounds.width,
                PANEL_B_SCHEMATIC_RELATIVE_HEIGHT * panel_b_bounds.height,
            )
        )
    profile_axis.set_axes_locator(None)
    profile_axis.set_position(
        (
            panel_b_bounds.x0
            + profile_relative_left * panel_b_bounds.width,
            scatter_bounds.y0,
            relative_width * panel_b_bounds.width,
            profile_bounds.height,
        )
    )
    fig.canvas.draw()
    scatter_label_display = scatter_axis.xaxis.label.get_transform().transform(
        scatter_axis.xaxis.label.get_position()
    )
    profile_label_axes_y = profile_axis.transAxes.inverted().transform(
        (scatter_label_display[0], scatter_label_display[1])
    )[1]
    profile_axis.xaxis.set_label_coords(0.50, float(profile_label_axes_y))


def plot_panel_c_raw_circular_shift_analysis(ax: Any, table: Any) -> None:
    """Plot unrescaled unit-area overlap profiles for the Panel B cohort."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    profile_axes = _add_shift_profile_row_axes(
        ax,
        label_prefix="panel_c_raw_shift_profile",
    )
    plot_raw_circular_shift_profiles(profile_axes, table)
    _format_shift_profile_row_axes(profile_axes)
    _label_shift_profile_row(ax, y_label="Unit-area overlap")

    ax.text(
        0.115,
        0.580,
        r"$S(\Delta)=\sum_z\min[\hat d(z),\hat l_\Delta(z)]$",
        ha="center",
        va="center",
        fontsize=5.2,
    )
    ax.text(
        0.115,
        0.380,
        r"$\sum_z\hat d(z)=\sum_z\hat l(z)=1$",
        ha="center",
        va="center",
        fontsize=5.0,
    )
    ax.text(
        0.115,
        0.270,
        "Before split-half rescaling",
        ha="center",
        va="center",
        fontsize=5.0,
    )


def plot_segment_overlap_response(ax: Any, table: Any) -> None:
    """Plot segment overlap versus response ratio with marginal histograms."""
    rows = table[table["included"].astype(bool)].copy()
    rows = rows[
        np.isfinite(
            np.asarray(rows["segment_tuning_similarity_index"], dtype=float)
        )
    ]
    finite = rows[
        rows["response_ratio_status"].astype(str) == "finite"
    ]
    finite_response = np.asarray(
        finite["log2_segment_response_ratio"],
        dtype=float,
    )
    finite_response = finite_response[np.isfinite(finite_response)]
    finite_extent = (
        float(np.max(np.abs(finite_response)))
        if finite_response.size
        else 0.0
    )
    finite_limit = max(4.0, float(np.ceil(finite_extent)))
    edge_position = finite_limit + 0.55
    colors = ("#0072B2", "#E69F00", "#009E73")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-edge_position - 0.8, edge_position + 0.8)
    ax.set_xlabel("Segment normalized overlap")
    ax.set_ylabel(
        r"Segment response ratio, $\log_2(\Sigma\,light/\Sigma\,dark)$"
    )
    ax.axis("off")
    scatter_ax = ax.inset_axes((0.08, 0.18, 0.72, 0.55))
    top_ax = ax.inset_axes((0.08, 0.77, 0.72, 0.18), sharex=scatter_ax)
    right_ax = ax.inset_axes((0.82, 0.18, 0.16, 0.55), sharey=scatter_ax)
    for child_axis in (scatter_ax, top_ax, right_ax):
        child_axis.set_in_layout(False)

    for segment, color in enumerate(colors):
        segment_rows = rows[rows["segment_index"].astype(int) == segment]
        finite_rows = segment_rows[
            segment_rows["response_ratio_status"].astype(str) == "finite"
        ]
        scatter_ax.scatter(
            np.asarray(
                finite_rows["segment_tuning_similarity_index"],
                dtype=float,
            ),
            np.asarray(
                finite_rows["log2_segment_response_ratio"],
                dtype=float,
            ),
            s=4.0,
            color=color,
            alpha=0.17,
            linewidths=0.0,
            rasterized=True,
            label=f"Segment {segment + 1}",
        )
        for status, y_value, marker in (
            ("dark_only", -edge_position, "v"),
            ("light_only", edge_position, "^"),
        ):
            one_sided = segment_rows[
                segment_rows["response_ratio_status"].astype(str) == status
            ]
            if one_sided.empty:
                continue
            scatter_ax.scatter(
                np.asarray(
                    one_sided["segment_tuning_similarity_index"],
                    dtype=float,
                ),
                np.full(len(one_sided), y_value, dtype=float),
                s=10.0,
                color=color,
                alpha=0.75,
                marker=marker,
                linewidths=0.0,
                rasterized=True,
            )

    scatter_ax.axhline(
        0.0,
        color="0.55",
        linewidth=0.6,
        linestyle=(0, (2, 2)),
    )
    scatter_ax.set_xlim(-0.025, 1.025)
    scatter_ax.set_ylim(-edge_position - 0.8, edge_position + 0.8)
    scatter_ax.set_xlabel("Normalized overlap")
    scatter_ax.set_ylabel(
        r"Response ratio, $\log_2(light/dark)$"
    )
    scatter_ax.spines[["top", "right"]].set_visible(False)
    scatter_ax.text(
        0.01,
        -edge_position,
        "Dark only",
        ha="left",
        va="top",
        fontsize=5.2,
    )
    scatter_ax.text(
        0.01,
        edge_position,
        "Light only",
        ha="left",
        va="bottom",
        fontsize=5.2,
    )

    overlap_values = np.asarray(
        rows["segment_tuning_similarity_index"],
        dtype=float,
    )
    response_values = np.asarray(
        rows["log2_segment_response_ratio"],
        dtype=float,
    )
    statuses = rows["response_ratio_status"].astype(str).to_numpy()
    response_values[statuses == "dark_only"] = -edge_position
    response_values[statuses == "light_only"] = edge_position
    overlap_weights = np.full(
        overlap_values.shape,
        1.0 / overlap_values.size,
        dtype=float,
    )
    response_weights = np.full(
        response_values.shape,
        1.0 / response_values.size,
        dtype=float,
    )
    top_ax.hist(
        overlap_values,
        bins=np.linspace(0.0, 1.0, 21),
        weights=overlap_weights,
        color="0.45",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    response_bins = np.linspace(-edge_position, edge_position, 31)
    right_ax.hist(
        response_values,
        bins=response_bins,
        weights=response_weights,
        orientation="horizontal",
        color="0.45",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    top_ax.set_xlim(scatter_ax.get_xlim())
    right_ax.set_ylim(scatter_ax.get_ylim())
    top_ax.text(
        0.01,
        0.94,
        "Fraction",
        transform=top_ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
    )
    right_ax.set_xlabel("Fraction", fontsize=5.5, labelpad=1.0)
    top_ax.tick_params(axis="x", labelbottom=False)
    right_ax.tick_params(axis="y", labelleft=False)
    top_ax.spines[["top", "right"]].set_visible(False)
    right_ax.spines[["top", "right"]].set_visible(False)
    for child_axis in (scatter_ax, top_ax, right_ax):
        child_axis.tick_params(labelsize=5.5, pad=0.7)

    neuron_columns = ["animal_name", "date", "region", "unit"]
    n_neurons = (
        int(rows.loc[:, neuron_columns].drop_duplicates().shape[0])
        if not rows.empty
        else 0
    )
    scatter_ax.text(
        0.98,
        0.96,
        f"n={len(rows):,} segments, {n_neurons:,} neurons",
        transform=scatter_ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )
    scatter_ax.legend(
        frameon=False,
        fontsize=4.8,
        ncols=1,
        loc="upper left",
        bbox_to_anchor=(1.03, 1.39),
        handletextpad=0.2,
        borderpad=0.0,
        markerscale=1.8,
    )


def plot_segment_stability_references(ax: Any, table: Any) -> None:
    """Plot unaveraged observed, split-half, and null segment overlaps."""
    observed = np.asarray(table["observed_segment_stability"], dtype=float)
    observed = observed[np.isfinite(observed)]
    split = np.asarray(table["split_half_segment_stability"], dtype=float)
    split = split[np.isfinite(split)]
    null = np.asarray(table["minimum_null_segment_stability"], dtype=float)
    null = null[np.isfinite(null)]
    counts: list[np.ndarray] = []
    for values, color, label, histtype, linestyle in (
        (observed, "0.35", "Dark–light", "bar", "solid"),
        (split, "#0072B2", "Split-half", "step", "solid"),
        (
            null,
            "#D55E00",
            "Circular-shift minimum",
            "step",
            (0.0, (2.4, 1.5)),
        ),
    ):
        weights = (
            np.full(values.shape, 1.0 / values.size, dtype=float)
            if values.size
            else values
        )
        kwargs: dict[str, Any] = {
            "bins": PANEL_B_TUNING_SIMILARITY_BINS,
            "weights": weights,
            "color": color,
            "label": label,
            "histtype": histtype,
        }
        if histtype == "bar":
            kwargs.update(EMPHASIS_HISTOGRAM_KWARGS)
        else:
            kwargs.update({"linewidth": 1.1, "linestyle": linestyle})
        histogram, _edges, _patches = ax.hist(values, **kwargs)
        counts.append(histogram)
    maximum = max(
        (float(np.max(values)) if values.size else 0.0 for values in counts),
        default=0.0,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(0.10, 1.35 * maximum))
    ax.set_xlabel("Segment normalized overlap")
    ax.set_ylabel("Fraction of segments")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=5.0,
        ncols=1,
        loc="upper left",
        handlelength=1.2,
        handletextpad=0.3,
        borderpad=0.0,
    )
    medians = [
        float(np.median(values)) if values.size else float("nan")
        for values in (observed, split, null)
    ]
    ax.text(
        0.98,
        0.96,
        f"Dark–light: n={observed.size:,}, med={medians[0]:.2f}\n"
        f"Split-half: n={split.size:,}, med={medians[1]:.2f}\n"
        f"Floor: n={null.size:,}, med={medians[2]:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )


def plot_achievable_segment_stability(ax: Any, table: Any) -> None:
    """Plot raw per-segment achievable stability with explicit overflow."""
    valid = table[table["achievable_status"].astype(str) == "valid"]
    values = np.asarray(valid["achievable_segment_stability"], dtype=float)
    values = values[np.isfinite(values)]
    display_lower = -1.0
    display_upper = 2.0
    displayed = values[(values >= display_lower) & (values <= display_upper)]
    weights = (
        np.full(displayed.shape, 1.0 / values.size, dtype=float)
        if values.size
        else displayed
    )
    counts, _edges, _patches = ax.hist(
        displayed,
        bins=np.linspace(display_lower, display_upper, 31),
        weights=weights,
        color="0.35",
        **EMPHASIS_HISTOGRAM_KWARGS,
    )
    ax.axvline(0.0, color="#D55E00", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.axvline(1.0, color="#0072B2", linewidth=0.8, linestyle=(0, (2, 2)))
    maximum = float(np.max(counts)) if counts.size else 0.0
    ax.set_xlim(display_lower, display_upper)
    ax.set_ylim(0.0, max(0.10, 1.28 * maximum))
    ax.set_xlabel("Achievable segment tuning stability")
    ax.set_ylabel("Fraction of segments")
    ax.spines[["top", "right"]].set_visible(False)
    median = float(np.median(values)) if values.size else float("nan")
    q1, q3 = (
        np.quantile(values, (0.25, 0.75))
        if values.size
        else (float("nan"), float("nan"))
    )
    nonpositive = int(
        np.sum(
            table["achievable_status"].astype(str)
            == "nonpositive_denominator"
        )
    )
    ax.text(
        0.02,
        0.95,
        f"n={values.size:,}, med={median:.2f} [{q1:.2f}, {q3:.2f}]\n"
        f"outside [{display_lower:g}, {display_upper:g}]: "
        f"{int(np.sum(values < display_lower)) + int(np.sum(values > display_upper)):,}\n"
        f"nonpositive denominator: {nonpositive:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
    )


def _select_panel_c_path_invariance_values(
    table: Any,
    turn_direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired valid dark/light scores used throughout Panel C."""
    rows = table[
        (table["turn_direction"].astype(str) == str(turn_direction))
        & table["passes_qc"].astype(bool)
        & (table["dark_path_invariance_status"].astype(str) == "valid")
        & (table["light_path_invariance_status"].astype(str) == "valid")
    ]
    dark_values = np.asarray(rows["dark_path_invariance_index"], dtype=float)
    light_values = np.asarray(rows["light_path_invariance_index"], dtype=float)
    valid = np.isfinite(dark_values) & np.isfinite(light_values)
    return dark_values[valid], light_values[valid]


def plot_panel_c_path_invariance(
    ax: Any,
    table: Any,
    *,
    example: dict[str, Any] | None = None,
) -> None:
    """Plot the path-invariance definition and dark-light comparisons."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    schematic_ax = ax.inset_axes((0.015, 0.290, 0.225, 0.550))
    schematic_ax.set_in_layout(False)
    position, x_curve, y_curve, x_path, y_path = (
        _path_invariance_schematic_curves(example)
    )
    overlap = np.minimum(x_curve, y_curve)
    schematic_ax.fill_between(
        position,
        0.0,
        overlap,
        color="#BDBDBD",
        alpha=0.55,
        linewidth=0.0,
        label="Overlap",
    )
    schematic_ax.plot(
        position,
        x_curve,
        color=_figure_2.PANEL_TRAJECTORY_COLORS[x_path],
        linewidth=0.9,
        label="Path x",
    )
    schematic_ax.plot(
        position,
        y_curve,
        color=_figure_2.PANEL_TRAJECTORY_COLORS[y_path],
        linewidth=0.9,
        label="Path y",
    )
    schematic_ax.set_xlim(0.0, 1.0)
    schematic_ax.set_ylim(bottom=0.0)
    schematic_ax.set_xticks(())
    schematic_ax.set_yticks(())
    schematic_ax.set_title("Same-turn overlap", fontsize=6.0, pad=1.0)
    schematic_ax.text(
        0.98,
        0.04,
        "z",
        transform=schematic_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
    )
    schematic_ax.spines[["top", "right"]].set_visible(False)
    schematic_ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=5.0,
        handlelength=1.0,
        handletextpad=0.3,
        borderpad=0.0,
    )
    ax.text(
        0.1275,
        0.110,
        r"$I(x,y)=\frac{2\sum_z\min[x(z),y(z)]}"
        r"{\sum_zx(z)+\sum_zy(z)}$",
        ha="center",
        va="center",
        fontsize=5.4,
    )

    bounds = ((0.310, 0.290, 0.135, 0.550), (0.495, 0.290, 0.135, 0.550))
    scatter_axes = []
    values_by_turn: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, ((turn_direction, (x_path, _y_path)), child_bounds) in enumerate(
        zip(PANEL_C_TURN_PATHS.items(), bounds, strict=True)
    ):
        plot_ax = ax.inset_axes(
            child_bounds,
            sharex=scatter_axes[0] if scatter_axes else None,
            sharey=scatter_axes[0] if scatter_axes else None,
        )
        plot_ax.set_in_layout(False)
        dark_values, light_values = _select_panel_c_path_invariance_values(
            table,
            turn_direction,
        )
        values_by_turn[turn_direction] = (dark_values, light_values)
        scatter_dark_values = np.clip(dark_values, 0.0, 1.0)
        scatter_light_values = np.clip(light_values, 0.0, 1.0)
        plot_ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="0.50",
            linestyle="--",
            linewidth=0.55,
            zorder=1,
        )
        plot_ax.scatter(
            scatter_dark_values,
            scatter_light_values,
            s=4.0,
            color=_figure_2.PANEL_TRAJECTORY_COLORS[x_path],
            alpha=0.30,
            edgecolors="none",
            zorder=2,
        )
        plot_ax.text(
            0.05,
            0.95,
            f"n={dark_values.size}",
            transform=plot_ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
        )
        plot_ax.set_xlim(0.0, 1.0)
        plot_ax.set_ylim(0.0, 1.0)
        plot_ax.set_xticks((0.0, 0.5, 1.0))
        plot_ax.set_yticks((0.0, 0.5, 1.0))
        plot_ax.set_title(
            PANEL_C_TURN_LABELS[turn_direction],
            fontsize=6.0,
            pad=1.0,
        )
        plot_ax.spines[["top", "right"]].set_visible(False)
        plot_ax.tick_params(labelsize=5.5, pad=0.7)
        plot_ax.set_box_aspect(1.0)
        if index:
            plot_ax.tick_params(axis="y", labelleft=False)
        scatter_axes.append(plot_ax)
    ax.text(
        0.470,
        0.070,
        "Dark path invariance index",
        ha="center",
        va="center",
        fontsize=6.0,
    )
    ax.text(
        0.275,
        0.565,
        "Light path\ninvariance index",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        linespacing=0.9,
    )

    delta_ax = ax.inset_axes((0.700, 0.290, 0.285, 0.550))
    delta_ax.set_in_layout(False)
    maximum_fraction = 0.0
    for turn_direction, (x_path, _y_path) in PANEL_C_TURN_PATHS.items():
        dark_values, light_values = values_by_turn[turn_direction]
        delta_values = light_values - dark_values
        weights = (
            np.full(delta_values.shape, 1.0 / delta_values.size, dtype=float)
            if delta_values.size
            else delta_values
        )
        median_delta = (
            float(np.median(delta_values)) if delta_values.size else float("nan")
        )
        label = (
            f"{PANEL_C_TURN_LABELS[turn_direction].removesuffix(' turns')}: "
            f"med Δ={median_delta:.2f}"
            if np.isfinite(median_delta)
            else PANEL_C_TURN_LABELS[turn_direction].removesuffix(" turns")
        )
        counts, _edges, _patches = delta_ax.hist(
            delta_values,
            bins=PANEL_C_PATH_INVARIANCE_DELTA_BINS,
            weights=weights,
            histtype="step",
            color=_figure_2.PANEL_TRAJECTORY_COLORS[x_path],
            alpha=0.90,
            linewidth=0.80,
            label=label,
        )
        if counts.size:
            maximum_fraction = max(maximum_fraction, float(np.nanmax(counts)))
    delta_ax.axvline(
        0.0,
        color="0.45",
        linestyle="--",
        linewidth=0.55,
        zorder=3,
    )
    delta_ax.set_xlim(-1.0, 1.0)
    delta_y_max = max(0.10, 1.15 * maximum_fraction)
    delta_ax.set_ylim(0.0, delta_y_max)
    delta_ax.set_xticks((-1.0, 0.0, 1.0))
    delta_ax.set_yticks((0.0, delta_y_max))
    delta_ax.set_yticklabels(("0", f"{delta_y_max:.2f}"))
    delta_ax.set_xlabel(
        "Change in path invariance (light − dark)",
        fontsize=5.5,
        labelpad=0.3,
    )
    delta_ax.set_ylabel("Fraction", fontsize=5.5, labelpad=0.4)
    delta_ax.tick_params(labelsize=5.0, pad=0.7)
    delta_ax.spines[["top", "right"]].set_visible(False)
    delta_ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=5.0,
        ncols=1,
        handlelength=1.2,
        columnspacing=0.8,
        handletextpad=0.3,
        borderpad=0.0,
    )


def build_output_path(
    output_dir: Path,
    output_name: str,
    output_format: str,
) -> Path:
    """Return the Figure 2 output path for a supported format."""
    return _figure_2.build_output_path(output_dir, output_name, output_format)


def make_figure_2(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = _figure_2.DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = _figure_2.DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = _figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = _figure_2.DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    panel_tuning_similarity_cache_dir: Path | None = None,
    refresh_panel_tuning_similarity_cache: bool = False,
    panel_path_invariance_cache_dir: Path | None = None,
    refresh_panel_path_invariance_cache: bool = False,
    min_epoch_movement_firing_rate_hz: float = (
        DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ
    ),
    min_path_movement_firing_rate_hz: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
    ),
    min_segment_mean_firing_rate_hz: float = (
        DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ
    ),
    min_stability_correlation: float = (
        _figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
    ),
    panel_b_null_n_permutations: int = (
        DEFAULT_PANEL_B_NULL_N_PERMUTATIONS
    ),
    panel_b_null_random_seed: int = DEFAULT_PANEL_B_NULL_RANDOM_SEED,
) -> Path:
    """Build and save Figure 2 with path-specific tuning preservation."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    panel_tuning_similarity_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_tuning_similarity_cache_dir is None
        else Path(panel_tuning_similarity_cache_dir)
    )
    quant_region = str(regions[0]) if regions else _figure_2.DEFAULT_REGIONS[0]
    panel_a_examples = [
        _figure_2.load_panel_a_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            unit_id=unit_id,
            trajectories=trajectories,
            dark_epoch=dark_epoch,
            light_epoch=light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
        for animal_name, date, region, unit_id, trajectories in (
            FIGURE_2_PANEL_A_EXAMPLES
        )
    ]
    legacy_panel_b_example_spec = _figure_2.FIGURE_2_PANEL_A_EXAMPLES[0]
    legacy_panel_b_example = panel_a_examples[
        FIGURE_2_PANEL_A_EXAMPLES.index(legacy_panel_b_example_spec)
    ]
    legacy_panel_b_overlap_table = (
        _figure_2._figure_2.load_panel_b_tuning_overlap_table(
            data_root=data_root,
            datasets=datasets,
            region=quant_region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
        )
    )
    legacy_panel_b_overlap_table = (
        _figure_2._figure_2.filter_panel_b_overlap_by_even_odd_stability(
            legacy_panel_b_overlap_table,
            data_root=data_root,
            datasets=datasets,
            region=quant_region,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            min_movement_firing_rate_hz=(
                _figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
            ),
            min_stability_correlation=(
                _figure_2._figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
            ),
        )
    )
    panel_h_shift_profile_table = load_or_compute_panel_h_shift_profile_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        cache_dir=panel_tuning_similarity_cache_dir,
        refresh_cache=refresh_panel_tuning_similarity_cache,
        min_epoch_movement_firing_rate_hz=(
            min_epoch_movement_firing_rate_hz
        ),
        min_path_movement_firing_rate_hz=(
            min_path_movement_firing_rate_hz
        ),
        min_segment_mean_firing_rate_hz=(
            min_segment_mean_firing_rate_hz
        ),
        min_stability_correlation=min_stability_correlation,
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(DEFAULT_FIGURE_WIDTH_MM, DEFAULT_FIGURE_HEIGHT_MM),
        constrained_layout=True,
    )
    layout_pads = {
        **_figure_2.CANONICAL_FIGURE_2_CONSTRAINED_LAYOUT_PADS,
        "h_pad": 0.03,
        "hspace": 0.10,
    }
    fig.get_layout_engine().set(**layout_pads)
    outer_grid = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[
            PANEL_A_HEIGHT_MM,
            _figure_2.PANEL_BC_QUANT_ROW_HEIGHT_MM,
        ],
    )
    panel_a_axis = fig.add_subplot(outer_grid[0, 0])
    bc_grid = outer_grid[1, 0].subgridspec(
        nrows=1,
        ncols=2,
        width_ratios=PANEL_BC_ROW_WIDTH_RATIOS,
        wspace=PANEL_B_ROW_WSPACE,
    )
    panel_b_axis = fig.add_subplot(bc_grid[0, 0])
    panel_c_axis = fig.add_subplot(bc_grid[0, 1])

    _figure_2.plot_panel_a2_examples_single_row(
        panel_a_axis,
        panel_a_examples,
        y_max_overrides=FIGURE_2_PANEL_A_Y_MAX_OVERRIDES,
        schematic_scale=FIGURE_2_PANEL_A_WTRACK_SCALE,
        ylabel_x=FIGURE_2_PANEL_A_YLABEL_X,
    )
    plot_panel_b_circular_shift_analysis(
        panel_b_axis,
        panel_h_shift_profile_table,
    )
    _figure_2._figure_2.plot_panel_b_dpp_overlap_with_schematic(
        panel_c_axis,
        legacy_panel_b_overlap_table,
        example=legacy_panel_b_example,
        low_threshold=(
            _figure_2._figure_2.PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
        ),
        high_threshold=(
            _figure_2._figure_2.PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
        ),
        show_grouped=False,
        show_scatter_linear_fit=True,
        show_scatter_r2=True,
        scatter_equal_aspect=True,
        schematic_style="path_colored_gray_overlap",
    )
    for old_text, new_text in (
        ("DPP index", "Path-invariance\nindex (PII)"),
        (
            "DPPI = max(left overlap,\nright overlap)",
            "PII = max(left overlap,\nright overlap)",
        ),
        ("Dark DPPI", "Dark PII"),
        ("Light DPPI", "Light PII"),
    ):
        _figure_2._figure_2._replace_nested_text(
            panel_c_axis,
            old_text,
            new_text,
            fontsize=_figure_2._figure_2.MIN_PUBLICATION_FONTSIZE_PT,
        )

    label_axis(panel_a_axis, "A", x=-0.02, y=_figure_2.PANEL_A_LABEL_Y)
    panel_a_label = panel_a_axis.texts[-1]
    panel_a_title = panel_a_axis.set_title(
        "Path-invariant progression tuning across dark and light",
        fontsize=8,
        pad=_figure_2.PANEL_A_TITLE_PAD,
    )
    panel_a_title.set_verticalalignment("bottom")
    label_axis(
        panel_b_axis,
        "B",
        x=-0.02,
        y=_figure_2.PANEL_B_LABEL_Y,
        va="baseline",
    )
    panel_b_label = panel_b_axis.texts[-1]
    panel_b_title = panel_b_axis.set_title(
        "Tuning shift across dark and light",
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    label_axis(
        panel_c_axis,
        "C",
        x=-0.035,
        y=_figure_2.PANEL_B_LABEL_Y,
        va="baseline",
    )
    panel_c_label = panel_c_axis.texts[-1]
    panel_c_title = panel_c_axis.set_title(
        "Path-invariance across dark and light",
        fontsize=8,
        pad=_figure_2.PANEL_B_TITLE_PAD,
    )
    _figure_2._raise_text_to_minimum_fontsize(
        fig,
        _figure_2.MIN_PUBLICATION_FONTSIZE_PT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _figure_2._set_axis_horizontal_bounds(
        panel_a_axis,
        left=_figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[0],
        width=_figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS[1],
    )
    bc_positions = [
        panel_axis.get_position()
        for panel_axis in (panel_b_axis, panel_c_axis)
    ]
    bc_left = bc_positions[0].x0
    bc_width = bc_positions[-1].x1 - bc_left
    target_left, target_width = _figure_2.PANEL_A_HORIZONTAL_AXIS_BOUNDS
    horizontal_scale = target_width / bc_width
    for panel_axis, position in zip(
        (panel_b_axis, panel_c_axis),
        bc_positions,
        strict=True,
    ):
        panel_axis.set_position(
            (
                target_left + (position.x0 - bc_left) * horizontal_scale,
                position.y0,
                position.width * horizontal_scale,
                position.height,
            )
        )
    panel_quant_axis_height = panel_b_axis.get_position().height
    _figure_2._set_axis_height_preserving_top(
        panel_b_axis,
        panel_quant_axis_height,
    )
    _figure_2._set_axis_height_preserving_top(
        panel_c_axis,
        panel_quant_axis_height,
    )
    fig.canvas.draw()
    _figure_2._align_text_to_reference_display_x(panel_b_label, panel_a_label)
    _figure_2._align_texts_to_reference_display_y(
        (panel_a_label, panel_a_title)
    )
    _figure_2._align_texts_to_reference_display_y(
        (
            panel_b_title,
            panel_b_label,
            panel_c_title,
            panel_c_label,
        )
    )
    _figure_2._align_panel_b_top_histogram_label_to_scatter(
        fig,
        panel_c_axis,
    )
    _align_panel_b_profile_with_panel_c_scatter(
        fig,
        panel_b_axis,
        panel_c_axis,
    )
    save_figure(fig, output_path, dpi=dpi, bbox_inches=None)
    plt.close(fig)
    print(f"Saved Figure 2 to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for Figure 2."""
    parser = argparse.ArgumentParser(description="Generate Figure 2.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Base directory containing analysis outputs. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--panel-example-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached example-cell rasters and rate curves. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help="Recompute example-cell data and overwrite matching caches.",
    )
    parser.add_argument(
        "--panel-tuning-similarity-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the per-cell, per-path TSI Parquet table. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-tuning-similarity-cache",
        action="store_true",
        help="Recompute Panel B values and reference tables.",
    )
    parser.add_argument(
        "--panel-b-null-n-permutations",
        type=int,
        default=DEFAULT_PANEL_B_NULL_N_PERMUTATIONS,
        help=(
            "Deprecated compatibility option; normalized-overlap nulls now "
            "enumerate all integer-bin circular shifts."
        ),
    )
    parser.add_argument(
        "--panel-b-null-random-seed",
        type=int,
        default=DEFAULT_PANEL_B_NULL_RANDOM_SEED,
        help=(
            "Deprecated compatibility option; exact normalized-overlap "
            "nulls do not use random sampling."
        ),
    )
    parser.add_argument(
        "--panel-path-invariance-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the per-cell, same-turn path-invariance table. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-path-invariance-cache",
        action="store_true",
        help="Recompute same-turn path-invariance values in dark and light.",
    )
    parser.add_argument(
        "--min-epoch-movement-firing-rate-hz",
        type=float,
        default=DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ,
        help=(
            "Minimum whole-epoch movement firing rate required in both dark "
            "and light; the comparison is strict (>). Default: "
            f"{DEFAULT_MIN_EPOCH_MOVEMENT_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-path-movement-firing-rate-hz",
        type=float,
        default=_figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ,
        help=(
            "Path movement firing-rate threshold for the whole-path Panel B "
            "split-half reference and Panel C. The dark-light Panel B "
            "distribution uses the segment mean-rate criterion instead. "
            "Default: "
            f"{_figure_2.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-segment-mean-firing-rate-hz",
        type=float,
        default=DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ,
        help=(
            "For Panel B, include a physical path segment when its dark or "
            "light mean tuning rate is strictly above this threshold. "
            f"Default: {DEFAULT_MIN_SEGMENT_MEAN_FIRING_RATE_HZ}"
        ),
    )
    parser.add_argument(
        "--min-stability-correlation",
        type=float,
        default=_figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
        help=(
            "For Panel B, odd/even tuning correlation required on the same "
            "path in dark and light using a strict comparison (>). Panel C "
            "retains its existing inclusive comparison. Default: "
            f"{_figure_2.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION}"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_figure_2.parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date. May be repeated. "
            "Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help=(
            "Region to include. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=_figure_2.DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {_figure_2.DEFAULT_POSITION_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=_figure_2.DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples to ignore. "
            f"Default: {_figure_2.DEFAULT_POSITION_OFFSET}"
        ),
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=_figure_2.DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {_figure_2.DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=_figure_2.DEFAULT_SIGMA_BINS,
        help=(
            "Gaussian smoothing width in bins. "
            f"Default: {_figure_2.DEFAULT_SIGMA_BINS}"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rasterization dpi for saved output. Default: 300",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run Figure 2 generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    panel_tuning_similarity_cache_dir = (
        args.panel_tuning_similarity_cache_dir
        if args.panel_tuning_similarity_cache_dir is not None
        else args.output_dir / "cache"
    )
    panel_path_invariance_cache_dir = (
        args.panel_path_invariance_cache_dir
        if args.panel_path_invariance_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_2(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
        panel_tuning_similarity_cache_dir=panel_tuning_similarity_cache_dir,
        refresh_panel_tuning_similarity_cache=(
            args.refresh_panel_tuning_similarity_cache
        ),
        panel_path_invariance_cache_dir=panel_path_invariance_cache_dir,
        refresh_panel_path_invariance_cache=(
            args.refresh_panel_path_invariance_cache
        ),
        min_path_movement_firing_rate_hz=(
            args.min_path_movement_firing_rate_hz
        ),
        min_segment_mean_firing_rate_hz=(
            args.min_segment_mean_firing_rate_hz
        ),
        min_epoch_movement_firing_rate_hz=(
            args.min_epoch_movement_firing_rate_hz
        ),
        min_stability_correlation=args.min_stability_correlation,
        panel_b_null_n_permutations=(
            args.panel_b_null_n_permutations
        ),
        panel_b_null_random_seed=args.panel_b_null_random_seed,
    )


if __name__ == "__main__":
    main()
