from __future__ import annotations

"""Estimate odd/even task-progression tuning stability for one session.

This script loads a task-progression session through the shared helpers, splits
each trajectory's laps into one-indexed odd and even trials, computes separate
normalized task-progression tuning curves for each split, and saves per-unit
odd/even tuning correlations and unit-area shape overlaps. Every source unit is
retained, including units whose stability is undefined; explicit QC columns
describe those cases. Summary histograms are saved as one 2x2 figure per epoch,
with selected regions overlaid in each trajectory panel.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.wtrack import get_wtrack_segment_edges
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    build_task_progression_bins,
    compute_movement_firing_rates,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)
from v1ca1.task_progression.tuning_analysis import (
    compute_similarity_score,
    deduplicate_requested_epochs,
    get_similarity_axis_limits,
    interpolate_nans,
)
from v1ca1.task_progression.similarity import compute_segmented_shape_overlap


REGION_COLORS = {
    "v1": "tab:blue",
    "ca1": "tab:orange",
}
MIN_FINITE_BINS_PER_CURVE = 2
MIN_FEATURE_SAMPLES_PER_SPLIT = 2
CONSTANT_CURVE_EPS = 1e-12
CORRELATION_BOUND_TOLERANCE = 1e-9
SHAPE_OVERLAP_EPS = 1e-12
SHAPE_OVERLAP_BOUND_TOLERANCE = 1e-9
STABILITY_NAN_POLICY = "linear_interpolate"


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start and end arrays from one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "IntervalSet start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts, ends


def _intervalset_is_empty(intervals: Any) -> bool:
    """Return whether one IntervalSet contains no intervals."""
    starts, _ends = _extract_interval_bounds(intervals)
    return starts.size == 0


def subset_intervalset(intervals: Any, indices: np.ndarray) -> Any:
    """Return one IntervalSet containing the requested interval indices."""
    starts, ends = _extract_interval_bounds(intervals)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    intervalset_class = intervals.__class__
    return intervalset_class(
        start=np.asarray(starts[indices], dtype=float),
        end=np.asarray(ends[indices], dtype=float),
        time_units="s",
    )


def split_laps_by_odd_even(intervals: Any) -> dict[str, Any]:
    """Split one trajectory's laps into one-indexed odd and even IntervalSets."""
    starts, _ends = _extract_interval_bounds(intervals)
    lap_indices = np.arange(starts.size, dtype=int)
    odd_indices = lap_indices[lap_indices % 2 == 0]
    even_indices = lap_indices[lap_indices % 2 == 1]
    return {
        "odd_indices": odd_indices,
        "even_indices": even_indices,
        "odd_interval": subset_intervalset(intervals, odd_indices),
        "even_interval": subset_intervalset(intervals, even_indices),
    }


def make_fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return histogram weights that normalize one finite sample vector to fraction."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _empty_stability_table() -> pd.DataFrame:
    """Return an empty odd/even stability table with the expected schema."""
    return pd.DataFrame(
        {
            "animal_name": pd.Series(dtype=str),
            "date": pd.Series(dtype=str),
            "unit": pd.Series(dtype=int),
            "region": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "trajectory_type": pd.Series(dtype=str),
            "firing_rate_hz": pd.Series(dtype=float),
            "stability_correlation": pd.Series(dtype=float),
            "n_odd_trials": pd.Series(dtype=int),
            "n_even_trials": pd.Series(dtype=int),
            "odd_duration_s": pd.Series(dtype=float),
            "even_duration_s": pd.Series(dtype=float),
            "n_odd_feature_samples": pd.Series(dtype=int),
            "n_even_feature_samples": pd.Series(dtype=int),
            "n_odd_spikes": pd.Series(dtype=int),
            "n_even_spikes": pd.Series(dtype=int),
            "n_odd_finite_bins": pd.Series(dtype=int),
            "n_even_finite_bins": pd.Series(dtype=int),
            "n_paired_finite_bins": pd.Series(dtype=int),
            "stability_status": pd.Series(dtype=str),
            "stability_shape_overlap": pd.Series(dtype=float),
            "odd_tuning_curve_area": pd.Series(dtype=float),
            "even_tuning_curve_area": pd.Series(dtype=float),
            "shape_overlap_status": pd.Series(dtype=str),
            "stability_segmented_shape_overlap": pd.Series(dtype=float),
            "segment_stability_shape_overlaps": pd.Series(dtype=str),
            "segment_shape_overlap_statuses": pd.Series(dtype=str),
            "odd_segment_mean_firing_rates_hz": pd.Series(dtype=str),
            "even_segment_mean_firing_rates_hz": pd.Series(dtype=str),
            "odd_segment_tuning_curve_areas": pd.Series(dtype=str),
            "even_segment_tuning_curve_areas": pd.Series(dtype=str),
            "segment_edges_normalized": pd.Series(dtype=str),
            "segmented_shape_overlap_status": pd.Series(dtype=str),
        }
    )


def _normalize_unit_ids(unit_ids: Sequence[Any]) -> tuple[int, ...]:
    """Return sorted, unique integer unit identifiers."""
    normalized = tuple(sorted(int(unit) for unit in unit_ids))
    if len(normalized) != len(set(normalized)):
        raise ValueError("Expected unique unit identifiers.")
    return normalized


def _curve_unit_ids(tuning_curve: Any) -> tuple[int, ...]:
    """Return the unique integer unit coordinates from one tuning curve."""
    if "unit" not in tuning_curve.coords:
        raise ValueError("Tuning curve is missing its unit coordinate.")
    return _normalize_unit_ids(tuning_curve.coords["unit"].values)


def _validate_tuning_curve_units(
    tuning_curve: Any,
    *,
    expected_unit_ids: tuple[int, ...],
    split_name: str,
) -> None:
    """Require one tuning curve to contain exactly the source sorting units."""
    curve_unit_ids = _curve_unit_ids(tuning_curve)
    if curve_unit_ids != expected_unit_ids:
        raise ValueError(
            f"{split_name.capitalize()} tuning-curve units do not match the "
            f"source sorting units. Expected {expected_unit_ids!r}, got "
            f"{curve_unit_ids!r}."
        )


def _count_finite_feature_samples(feature: Any, intervals: Any) -> int:
    """Return the number of finite feature samples inside an IntervalSet."""
    if _intervalset_is_empty(intervals) or float(intervals.tot_length()) <= 0.0:
        return 0
    values = np.asarray(feature.restrict(intervals).d, dtype=float).reshape(-1)
    return int(np.count_nonzero(np.isfinite(values)))


def _count_spikes_by_unit(
    spikes: Any,
    intervals: Any,
    *,
    unit_ids: tuple[int, ...],
) -> pd.Series:
    """Return per-unit spike counts inside an IntervalSet."""
    if _intervalset_is_empty(intervals) or float(intervals.tot_length()) <= 0.0:
        return pd.Series(0, index=unit_ids, dtype=int)
    return pd.Series(
        {
            unit: int(
                np.asarray(spikes[unit].restrict(intervals).t, dtype=float).size
            )
            for unit in unit_ids
        },
        dtype=int,
    )


def _trajectory_stability_status(metadata: dict[str, Any]) -> str | None:
    """Return a shared invalid status when a split cannot be computed."""
    if int(metadata["n_odd_trials"]) <= 0:
        return "no_odd_trials"
    if int(metadata["n_even_trials"]) <= 0:
        return "no_even_trials"
    if float(metadata["odd_duration_s"]) <= 0.0:
        return "no_odd_movement_support"
    if float(metadata["even_duration_s"]) <= 0.0:
        return "no_even_movement_support"
    if int(metadata["n_odd_feature_samples"]) < MIN_FEATURE_SAMPLES_PER_SPLIT:
        return "insufficient_odd_feature_samples"
    if int(metadata["n_even_feature_samples"]) < MIN_FEATURE_SAMPLES_PER_SPLIT:
        return "insufficient_even_feature_samples"
    return None


def _evaluate_stability_correlation(
    odd_curve: np.ndarray,
    even_curve: np.ndarray,
    *,
    n_odd_spikes: int,
    n_even_spikes: int,
) -> dict[str, Any]:
    """Return one correlation and machine-readable tuning-curve QC."""
    odd_curve = np.asarray(odd_curve, dtype=float).reshape(-1)
    even_curve = np.asarray(even_curve, dtype=float).reshape(-1)
    if odd_curve.shape != even_curve.shape:
        raise ValueError(
            "Odd and even tuning curves must have matching shapes. "
            f"Got {odd_curve.shape} and {even_curve.shape}."
        )

    odd_finite = np.isfinite(odd_curve)
    even_finite = np.isfinite(even_curve)
    diagnostics = {
        "n_odd_finite_bins": int(np.count_nonzero(odd_finite)),
        "n_even_finite_bins": int(np.count_nonzero(even_finite)),
        "n_paired_finite_bins": int(np.count_nonzero(odd_finite & even_finite)),
    }

    invalid_status: str | None = None
    if n_odd_spikes <= 0:
        invalid_status = "no_odd_spikes"
    elif n_even_spikes <= 0:
        invalid_status = "no_even_spikes"
    elif np.any(np.isinf(odd_curve)):
        invalid_status = "nonfinite_odd_curve"
    elif np.any(np.isinf(even_curve)):
        invalid_status = "nonfinite_even_curve"
    elif diagnostics["n_odd_finite_bins"] < MIN_FINITE_BINS_PER_CURVE:
        invalid_status = "insufficient_odd_finite_bins"
    elif diagnostics["n_even_finite_bins"] < MIN_FINITE_BINS_PER_CURVE:
        invalid_status = "insufficient_even_finite_bins"
    else:
        interpolated_odd = np.asarray(interpolate_nans(odd_curve), dtype=float)
        interpolated_even = np.asarray(interpolate_nans(even_curve), dtype=float)
        if np.std(interpolated_odd) <= CONSTANT_CURVE_EPS:
            invalid_status = "constant_odd_curve"
        elif np.std(interpolated_even) <= CONSTANT_CURVE_EPS:
            invalid_status = "constant_even_curve"

    correlation = np.nan
    if invalid_status is None:
        correlation = float(
            compute_similarity_score(
                odd_curve,
                even_curve,
                similarity_metric="correlation",
            )
        )
        if not np.isfinite(correlation):
            invalid_status = "nonfinite_correlation"
            correlation = np.nan
        elif not (
            -1.0 - CORRELATION_BOUND_TOLERANCE
            <= correlation
            <= 1.0 + CORRELATION_BOUND_TOLERANCE
        ):
            invalid_status = "out_of_bounds_correlation"
            correlation = np.nan
        else:
            correlation = float(np.clip(correlation, -1.0, 1.0))

    return {
        "stability_correlation": correlation,
        "stability_status": invalid_status or "valid",
        **diagnostics,
    }


def _evaluate_stability_shape_overlap(
    odd_curve: np.ndarray,
    even_curve: np.ndarray,
    *,
    n_odd_spikes: int,
    n_even_spikes: int,
) -> dict[str, Any]:
    """Return unit-area odd/even shape overlap and independent curve QC."""
    odd_curve = np.asarray(odd_curve, dtype=float).reshape(-1)
    even_curve = np.asarray(even_curve, dtype=float).reshape(-1)
    if odd_curve.shape != even_curve.shape:
        raise ValueError(
            "Odd and even tuning curves must have matching shapes. "
            f"Got {odd_curve.shape} and {even_curve.shape}."
        )

    result: dict[str, Any] = {
        "stability_shape_overlap": np.nan,
        "odd_tuning_curve_area": np.nan,
        "even_tuning_curve_area": np.nan,
        "shape_overlap_status": "valid",
    }
    odd_finite = np.isfinite(odd_curve)
    even_finite = np.isfinite(even_curve)

    invalid_status: str | None = None
    if n_odd_spikes <= 0:
        invalid_status = "no_odd_spikes"
    elif n_even_spikes <= 0:
        invalid_status = "no_even_spikes"
    elif np.any(np.isinf(odd_curve)):
        invalid_status = "nonfinite_odd_curve"
    elif np.any(np.isinf(even_curve)):
        invalid_status = "nonfinite_even_curve"
    elif np.count_nonzero(odd_finite) < MIN_FINITE_BINS_PER_CURVE:
        invalid_status = "insufficient_odd_finite_bins"
    elif np.count_nonzero(even_finite) < MIN_FINITE_BINS_PER_CURVE:
        invalid_status = "insufficient_even_finite_bins"
    elif np.any(odd_curve[odd_finite] < -SHAPE_OVERLAP_EPS):
        invalid_status = "negative_odd_curve"
    elif np.any(even_curve[even_finite] < -SHAPE_OVERLAP_EPS):
        invalid_status = "negative_even_curve"

    if invalid_status is None:
        odd = np.maximum(interpolate_nans(odd_curve), 0.0)
        even = np.maximum(interpolate_nans(even_curve), 0.0)
        odd_area = float(odd.sum())
        even_area = float(even.sum())
        result["odd_tuning_curve_area"] = odd_area
        result["even_tuning_curve_area"] = even_area
        if not np.isfinite(odd_area):
            invalid_status = "nonfinite_odd_area"
        elif not np.isfinite(even_area):
            invalid_status = "nonfinite_even_area"
        elif odd_area <= SHAPE_OVERLAP_EPS:
            invalid_status = "nonpositive_odd_area"
        elif even_area <= SHAPE_OVERLAP_EPS:
            invalid_status = "nonpositive_even_area"
        else:
            overlap = float(np.minimum(odd / odd_area, even / even_area).sum())
            if not np.isfinite(overlap):
                invalid_status = "nonfinite_shape_overlap"
            elif not (
                -SHAPE_OVERLAP_BOUND_TOLERANCE
                <= overlap
                <= 1.0 + SHAPE_OVERLAP_BOUND_TOLERANCE
            ):
                invalid_status = "out_of_bounds_shape_overlap"
            else:
                result["stability_shape_overlap"] = float(
                    np.clip(overlap, 0.0, 1.0)
                )

    result["shape_overlap_status"] = invalid_status or "valid"
    return result


def _evaluate_segmented_stability_shape_overlap(
    odd_curve: np.ndarray,
    even_curve: np.ndarray,
    progression: np.ndarray,
    segment_edges: np.ndarray,
    *,
    n_odd_spikes: int,
    n_even_spikes: int,
) -> dict[str, Any]:
    """Return odd/even unit-area overlap within physical path segments."""
    split_status = None
    if n_odd_spikes <= 0:
        split_status = "no_odd_spikes"
    elif n_even_spikes <= 0:
        split_status = "no_even_spikes"
    if split_status is not None:
        unavailable = [np.nan] * (len(segment_edges) - 1)
        return {
            "stability_segmented_shape_overlap": np.nan,
            "segment_stability_shape_overlaps": json.dumps(unavailable),
            "segment_shape_overlap_statuses": json.dumps(
                [split_status] * len(unavailable)
            ),
            "odd_segment_mean_firing_rates_hz": json.dumps(unavailable),
            "even_segment_mean_firing_rates_hz": json.dumps(unavailable),
            "odd_segment_tuning_curve_areas": json.dumps(unavailable),
            "even_segment_tuning_curve_areas": json.dumps(unavailable),
            "segment_edges_normalized": json.dumps(
                np.asarray(segment_edges, dtype=float).tolist()
            ),
            "segmented_shape_overlap_status": split_status,
        }
    try:
        result = compute_segmented_shape_overlap(
            odd_curve,
            even_curve,
            progression,
            segment_edges,
        )
    except ValueError as exc:
        if "contains no tuning-curve bin centers" not in str(exc):
            raise
        unavailable = [np.nan] * (len(segment_edges) - 1)
        return {
            "stability_segmented_shape_overlap": np.nan,
            "segment_stability_shape_overlaps": json.dumps(unavailable),
            "segment_shape_overlap_statuses": json.dumps(
                ["insufficient_segment_bins"] * len(unavailable)
            ),
            "odd_segment_mean_firing_rates_hz": json.dumps(unavailable),
            "even_segment_mean_firing_rates_hz": json.dumps(unavailable),
            "odd_segment_tuning_curve_areas": json.dumps(unavailable),
            "even_segment_tuning_curve_areas": json.dumps(unavailable),
            "segment_edges_normalized": json.dumps(
                np.asarray(segment_edges, dtype=float).tolist()
            ),
            "segmented_shape_overlap_status": "insufficient_segment_bins",
        }
    scores = np.asarray(result["scores"], dtype=float)
    statuses = [str(value) for value in result["statuses"]]
    finite = np.isfinite(scores)
    if finite.all():
        aggregate = float(np.mean(scores))
        status = "valid"
    elif finite.any():
        aggregate = float(np.mean(scores[finite]))
        status = "partial_invalid_segments"
    else:
        aggregate = float("nan")
        status = "no_valid_segments"
    return {
        "stability_segmented_shape_overlap": aggregate,
        "segment_stability_shape_overlaps": json.dumps(result["scores"]),
        "segment_shape_overlap_statuses": json.dumps(statuses),
        "odd_segment_mean_firing_rates_hz": json.dumps(
            result["mean_rates_a_hz"]
        ),
        "even_segment_mean_firing_rates_hz": json.dumps(
            result["mean_rates_b_hz"]
        ),
        "odd_segment_tuning_curve_areas": json.dumps(result["areas_a"]),
        "even_segment_tuning_curve_areas": json.dumps(result["areas_b"]),
        "segment_edges_normalized": json.dumps(
            np.asarray(segment_edges, dtype=float).tolist()
        ),
        "segmented_shape_overlap_status": status,
    }


def _tuning_curve_progression(tuning_curve: Any) -> np.ndarray:
    """Return the sole non-unit coordinate from one tuning-curve array."""
    dimensions = [str(value) for value in tuning_curve.dims if str(value) != "unit"]
    if len(dimensions) != 1 or dimensions[0] not in tuning_curve.coords:
        raise ValueError(
            "Odd/even tuning curves must have one non-unit progression coordinate."
        )
    return np.asarray(tuning_curve.coords[dimensions[0]].values, dtype=float)


def build_stability_table_for_tuning_curves(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    expected_unit_ids: Sequence[Any],
    odd_tuning_curve: Any | None,
    even_tuning_curve: Any | None,
    epoch_firing_rates: pd.Series,
    odd_spike_counts: pd.Series,
    even_spike_counts: pd.Series,
    n_odd_trials: int,
    n_even_trials: int,
    odd_duration_s: float,
    even_duration_s: float,
    n_odd_feature_samples: int,
    n_even_feature_samples: int,
    trajectory_status: str | None = None,
) -> pd.DataFrame:
    """Return one stability or explicit invalid row per source sorting unit."""
    unit_ids = _normalize_unit_ids(expected_unit_ids)
    if not unit_ids:
        return _empty_stability_table()
    if epoch_firing_rates.index.has_duplicates:
        raise ValueError("epoch_firing_rates must have unique unit identifiers.")
    if odd_spike_counts.index.has_duplicates or even_spike_counts.index.has_duplicates:
        raise ValueError("Odd and even spike counts must have unique unit identifiers.")
    if _normalize_unit_ids(odd_spike_counts.index) != unit_ids:
        raise ValueError("Odd spike-count units do not match the source sorting units.")
    if _normalize_unit_ids(even_spike_counts.index) != unit_ids:
        raise ValueError("Even spike-count units do not match the source sorting units.")

    curves_available = odd_tuning_curve is not None and even_tuning_curve is not None
    segment_edges = get_wtrack_segment_edges(animal_name)
    if curves_available:
        _validate_tuning_curve_units(
            odd_tuning_curve,
            expected_unit_ids=unit_ids,
            split_name="odd",
        )
        _validate_tuning_curve_units(
            even_tuning_curve,
            expected_unit_ids=unit_ids,
            split_name="even",
        )
        odd_progression = _tuning_curve_progression(odd_tuning_curve)
        even_progression = _tuning_curve_progression(even_tuning_curve)
        if odd_progression.shape != even_progression.shape or not np.array_equal(
            odd_progression,
            even_progression,
            equal_nan=True,
        ):
            raise ValueError("Odd/even tuning-curve progression grids differ.")
    elif trajectory_status is None:
        raise ValueError(
            "A trajectory_status is required when odd/even tuning curves are unavailable."
        )

    rows: list[dict[str, Any]] = []
    for unit in unit_ids:
        firing_rate_hz = float(epoch_firing_rates.get(int(unit), np.nan))
        n_odd_spikes = int(odd_spike_counts.get(int(unit), 0))
        n_even_spikes = int(even_spike_counts.get(int(unit), 0))
        if curves_available:
            correlation_qc = _evaluate_stability_correlation(
                odd_tuning_curve.sel(unit=unit).values,
                even_tuning_curve.sel(unit=unit).values,
                n_odd_spikes=n_odd_spikes,
                n_even_spikes=n_even_spikes,
            )
            shape_overlap_qc = _evaluate_stability_shape_overlap(
                odd_tuning_curve.sel(unit=unit).values,
                even_tuning_curve.sel(unit=unit).values,
                n_odd_spikes=n_odd_spikes,
                n_even_spikes=n_even_spikes,
            )
            segmented_shape_overlap_qc = (
                _evaluate_segmented_stability_shape_overlap(
                    odd_tuning_curve.sel(unit=unit).values,
                    even_tuning_curve.sel(unit=unit).values,
                    odd_progression,
                    segment_edges,
                    n_odd_spikes=n_odd_spikes,
                    n_even_spikes=n_even_spikes,
                )
            )
        else:
            correlation_qc = {
                "stability_correlation": np.nan,
                "stability_status": str(trajectory_status),
                "n_odd_finite_bins": 0,
                "n_even_finite_bins": 0,
                "n_paired_finite_bins": 0,
            }
            shape_overlap_qc = {
                "stability_shape_overlap": np.nan,
                "odd_tuning_curve_area": np.nan,
                "even_tuning_curve_area": np.nan,
                "shape_overlap_status": str(trajectory_status),
            }
            segmented_shape_overlap_qc = {
                "stability_segmented_shape_overlap": np.nan,
                "segment_stability_shape_overlaps": json.dumps(
                    [np.nan, np.nan, np.nan]
                ),
                "segment_shape_overlap_statuses": json.dumps(
                    [str(trajectory_status)] * 3
                ),
                "odd_segment_mean_firing_rates_hz": json.dumps(
                    [np.nan, np.nan, np.nan]
                ),
                "even_segment_mean_firing_rates_hz": json.dumps(
                    [np.nan, np.nan, np.nan]
                ),
                "odd_segment_tuning_curve_areas": json.dumps(
                    [np.nan, np.nan, np.nan]
                ),
                "even_segment_tuning_curve_areas": json.dumps(
                    [np.nan, np.nan, np.nan]
                ),
                "segment_edges_normalized": json.dumps(segment_edges.tolist()),
                "segmented_shape_overlap_status": str(trajectory_status),
            }
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "unit": int(unit),
                "region": region,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "firing_rate_hz": firing_rate_hz,
                "n_odd_trials": int(n_odd_trials),
                "n_even_trials": int(n_even_trials),
                "odd_duration_s": float(odd_duration_s),
                "even_duration_s": float(even_duration_s),
                "n_odd_feature_samples": int(n_odd_feature_samples),
                "n_even_feature_samples": int(n_even_feature_samples),
                "n_odd_spikes": n_odd_spikes,
                "n_even_spikes": n_even_spikes,
                **correlation_qc,
                **shape_overlap_qc,
                **segmented_shape_overlap_qc,
            }
        )

    if not rows:
        return _empty_stability_table()
    return pd.DataFrame(rows).sort_values("unit", kind="stable").reset_index(drop=True)


def compute_odd_even_tuning_curves_for_trajectory(
    *,
    spikes: Any,
    task_progression: Any,
    trajectory_interval: Any,
    movement_interval: Any,
    bins: np.ndarray,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    """Compute odd and even task-progression tuning curves for one trajectory."""
    import pynapple as nap

    unit_ids = _normalize_unit_ids(spikes.keys())
    split = split_laps_by_odd_even(trajectory_interval)
    odd_epoch = split["odd_interval"].intersect(movement_interval)
    even_epoch = split["even_interval"].intersect(movement_interval)
    metadata = {
        "n_odd_trials": int(np.asarray(split["odd_indices"]).size),
        "n_even_trials": int(np.asarray(split["even_indices"]).size),
        "odd_duration_s": float(odd_epoch.tot_length()),
        "even_duration_s": float(even_epoch.tot_length()),
        "n_odd_feature_samples": _count_finite_feature_samples(
            task_progression,
            odd_epoch,
        ),
        "n_even_feature_samples": _count_finite_feature_samples(
            task_progression,
            even_epoch,
        ),
        "odd_spike_counts": _count_spikes_by_unit(
            spikes,
            odd_epoch,
            unit_ids=unit_ids,
        ),
        "even_spike_counts": _count_spikes_by_unit(
            spikes,
            even_epoch,
            unit_ids=unit_ids,
        ),
    }

    if _trajectory_stability_status(metadata) is not None:
        return None, None, metadata

    odd_tuning_curve = nap.compute_tuning_curves(
        data=spikes,
        features=task_progression,
        bins=[bins],
        epochs=odd_epoch,
        feature_names=["tp"],
    )
    even_tuning_curve = nap.compute_tuning_curves(
        data=spikes,
        features=task_progression,
        bins=[bins],
        epochs=even_epoch,
        feature_names=["tp"],
    )
    return odd_tuning_curve, even_tuning_curve, metadata


def compute_trajectory_stability_table(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    spikes: Any,
    task_progression: Any,
    trajectory_interval: Any,
    movement_interval: Any,
    bins: np.ndarray,
    epoch_firing_rates: pd.Series,
) -> pd.DataFrame:
    """Compute all-unit odd/even stability for one trajectory and epoch.

    Unit identifiers are taken directly from ``spikes.keys()``. This supports
    ephemeral integer keys used by database adapters while leaving attachment
    of persistent unit identities to the caller.
    """
    if trajectory_type not in TRAJECTORY_TYPES:
        raise ValueError(
            f"Unknown trajectory_type {trajectory_type!r}; expected one of "
            f"{TRAJECTORY_TYPES!r}."
        )

    unit_ids = _normalize_unit_ids(spikes.keys())
    odd_tuning_curve, even_tuning_curve, metadata = (
        compute_odd_even_tuning_curves_for_trajectory(
            spikes=spikes,
            task_progression=task_progression,
            trajectory_interval=trajectory_interval,
            movement_interval=movement_interval,
            bins=bins,
        )
    )
    return build_stability_table_for_tuning_curves(
        animal_name=animal_name,
        date=date,
        region=region,
        epoch=epoch,
        trajectory_type=trajectory_type,
        expected_unit_ids=unit_ids,
        odd_tuning_curve=odd_tuning_curve,
        even_tuning_curve=even_tuning_curve,
        epoch_firing_rates=epoch_firing_rates,
        odd_spike_counts=metadata["odd_spike_counts"],
        even_spike_counts=metadata["even_spike_counts"],
        n_odd_trials=int(metadata["n_odd_trials"]),
        n_even_trials=int(metadata["n_even_trials"]),
        odd_duration_s=float(metadata["odd_duration_s"]),
        even_duration_s=float(metadata["even_duration_s"]),
        n_odd_feature_samples=int(metadata["n_odd_feature_samples"]),
        n_even_feature_samples=int(metadata["n_even_feature_samples"]),
        trajectory_status=_trajectory_stability_status(metadata),
    )


def compute_epoch_stability_table(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    bins: np.ndarray,
    epoch_firing_rates: pd.Series,
) -> pd.DataFrame:
    """Compute odd/even stability rows for all trajectories in one epoch."""
    tables: list[pd.DataFrame] = []
    for trajectory_type in TRAJECTORY_TYPES:
        tables.append(
            compute_trajectory_stability_table(
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                trajectory_type=trajectory_type,
                spikes=spikes,
                task_progression=task_progression_by_trajectory[trajectory_type],
                trajectory_interval=trajectory_intervals[trajectory_type],
                movement_interval=movement_interval,
                bins=bins,
                epoch_firing_rates=epoch_firing_rates,
            )
        )

    if not tables:
        return _empty_stability_table()
    return pd.concat(tables, axis=0, ignore_index=True)


def plot_epoch_stability_histograms(
    stability_table: pd.DataFrame,
    *,
    epoch: str,
    regions: tuple[str, ...],
    fig_path: Path,
) -> None:
    """Save one 2x2 odd/even stability histogram figure for an epoch."""
    import matplotlib.pyplot as plt

    axis_min, axis_max = get_similarity_axis_limits("correlation")
    bins = np.linspace(axis_min, axis_max, 25)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    for ax, trajectory_type in zip(axes.ravel(), TRAJECTORY_TYPES, strict=True):
        trajectory_rows = stability_table[stability_table["trajectory_type"] == trajectory_type]
        has_data = False
        for region in regions:
            values = pd.to_numeric(
                trajectory_rows.loc[
                    trajectory_rows["region"] == region,
                    "stability_correlation",
                ],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            has_data = True
            ax.hist(
                values,
                bins=bins,
                weights=make_fraction_histogram_weights(values),
                alpha=0.55,
                color=REGION_COLORS.get(region, None),
                edgecolor="none",
                label=region.upper(),
            )

        ax.set_title(trajectory_type.replace("_", " "))
        ax.set_xlim(axis_min, axis_max)
        if has_data:
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid units", ha="center", va="center")

    for ax in axes[-1, :]:
        ax.set_xlabel("Odd/even correlation")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of units")

    fig.suptitle(f"{epoch} task-progression stability", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stability_table(stability_table: pd.DataFrame, *, data_dir: Path) -> Path:
    """Write the selected-session odd/even stability table as parquet."""
    output_path = data_dir / "odd_even_task_progression_stability.parquet"
    stability_table.to_parquet(output_path)
    return output_path


def save_stability_figures(
    stability_table: pd.DataFrame,
    *,
    fig_dir: Path,
    regions: tuple[str, ...],
    epochs: list[str],
) -> list[Path]:
    """Write one odd/even stability histogram figure per epoch."""
    saved_paths: list[Path] = []
    for epoch in epochs:
        fig_path = fig_dir / f"{epoch}_odd_even_task_progression_stability.png"
        plot_epoch_stability_histograms(
            stability_table[stability_table["epoch"] == epoch],
            epoch=epoch,
            regions=regions,
            fig_path=fig_path,
        )
        saved_paths.append(fig_path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for odd/even tuning stability."""
    parser = argparse.ArgumentParser(
        description="Compute odd/even task-progression tuning stability"
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
        help=f"Regions to analyze. Default: {' '.join(REGIONS)}",
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        nargs="+",
        help=(
            "Run epoch labels to analyze. Defaults to all run epochs with usable "
            "cleaned-DLC head position."
        ),
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
        "--speed-smoothing-sigma-s",
        type=float,
        default=DEFAULT_SPEED_SIGMA_S,
        help=(
            "Gaussian smoothing sigma in seconds used to estimate speed. "
            f"Default: {DEFAULT_SPEED_SIGMA_S}"
        ),
    )
    parser.add_argument(
        "--place-bin-size-cm",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=(
            "Physical bin size used to construct normalized task-progression "
            f"bins. Default: {DEFAULT_PLACE_BIN_SIZE_CM}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run odd/even task-progression stability analysis for one session."""
    args = parse_arguments()
    if not np.isfinite(args.place_bin_size_cm) or args.place_bin_size_cm <= 0:
        raise ValueError("--place-bin-size-cm must be positive and finite.")
    selected_regions = tuple(dict.fromkeys(args.regions))
    selected_epochs = deduplicate_requested_epochs(args.epochs)

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        selected_run_epochs=selected_epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        speed_smoothing_sigma_s=args.speed_smoothing_sigma_s,
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    task_progression_bins = build_task_progression_bins(
        args.animal_name,
        place_bin_size_cm=args.place_bin_size_cm,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    tables: list[pd.DataFrame] = []
    for region in selected_regions:
        for epoch in session["run_epochs"]:
            epoch_firing_rates = pd.Series(
                movement_firing_rates[region][epoch],
                index=list(session["spikes_by_region"][region].keys()),
                dtype=float,
            )
            tables.append(
                compute_epoch_stability_table(
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    spikes=session["spikes_by_region"][region],
                    task_progression_by_trajectory=session[
                        "task_progression_by_trajectory"
                    ][epoch],
                    trajectory_intervals=session["trajectory_intervals"][epoch],
                    movement_interval=session["movement_by_run"][epoch],
                    bins=task_progression_bins,
                    epoch_firing_rates=epoch_firing_rates,
                )
            )

    stability_table = (
        pd.concat(tables, axis=0, ignore_index=True) if tables else _empty_stability_table()
    )
    saved_table = save_stability_table(stability_table, data_dir=data_dir)
    saved_figures = save_stability_figures(
        stability_table,
        fig_dir=fig_dir,
        regions=selected_regions,
        epochs=session["run_epochs"],
    )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.stability",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": list(selected_regions),
            "epochs": session["run_epochs"],
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "speed_smoothing_sigma_s": args.speed_smoothing_sigma_s,
            "place_bin_size_cm": args.place_bin_size_cm,
            "minimum_finite_bins_per_curve": MIN_FINITE_BINS_PER_CURVE,
            "minimum_feature_samples_per_split": MIN_FEATURE_SAMPLES_PER_SPLIT,
            "constant_curve_epsilon": CONSTANT_CURVE_EPS,
            "shape_overlap_metric": "unit_area_minimum_overlap",
            "shape_overlap_epsilon": SHAPE_OVERLAP_EPS,
            "nan_policy": STABILITY_NAN_POLICY,
            "firing_rate_filter_applied": False,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "task_progression_bin_count": int(len(task_progression_bins) - 1),
            "saved_table": saved_table,
            "saved_figures": saved_figures,
        },
    )
    print(f"Saved stability table to {saved_table}")
    print(f"Saved {len(saved_figures)} figure(s)")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
