"""Shared, database-free helpers for path-specific tuning similarity."""

from __future__ import annotations

from typing import Any

import numpy as np


SIMILARITY_METRICS = (
    "correlation",
    "absolute_overlap",
    "shape_overlap",
)
SIMILARITY_QC_COLUMNS = (
    "n_trajectory_a_finite_bins",
    "n_trajectory_b_finite_bins",
    "n_paired_finite_bins",
    "similarity_status",
)
DIRECT_COMPARISON_SPECS = (
    {
        "comparison_family": "same_turn",
        "comparison_label": "left_turn",
        "side": "left",
        "trajectory_a": "center_to_left",
        "trajectory_b": "right_to_center",
        "flip_trajectory_b": False,
    },
    {
        "comparison_family": "same_turn",
        "comparison_label": "right_turn",
        "side": "right",
        "trajectory_a": "center_to_right",
        "trajectory_b": "left_to_center",
        "flip_trajectory_b": False,
    },
    {
        "comparison_family": "same_arm",
        "comparison_label": "left_arm",
        "side": "left",
        "trajectory_a": "center_to_left",
        "trajectory_b": "left_to_center",
        "flip_trajectory_b": True,
    },
    {
        "comparison_family": "same_arm",
        "comparison_label": "right_arm",
        "side": "right",
        "trajectory_a": "center_to_right",
        "trajectory_b": "right_to_center",
        "flip_trajectory_b": True,
    },
)
DIRECT_COMPARISON_LABELS = tuple(
    spec["comparison_label"] for spec in DIRECT_COMPARISON_SPECS
)
SIDE_TO_DIRECT_LABELS = {
    "left": ("left_turn", "left_arm"),
    "right": ("right_turn", "right_arm"),
}
LABEL_TO_SPEC = {
    spec["comparison_label"]: spec for spec in DIRECT_COMPARISON_SPECS
}


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in one one-dimensional tuning curve."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D tuning curve, got shape {values.shape}.")
    if np.all(np.isnan(values)):
        return np.nan_to_num(values)

    nans = np.isnan(values)
    if not np.any(nans):
        return values

    output = values.copy()
    output[nans] = np.interp(
        np.flatnonzero(nans),
        np.flatnonzero(~nans),
        values[~nans],
    )
    return output


def make_segment_masks(
    progression: np.ndarray,
    segment_edges: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return nonoverlapping bin-center masks for normalized path segments."""
    values = np.asarray(progression, dtype=float).reshape(-1)
    edges = np.asarray(segment_edges, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Path progression must contain finite bin centers.")
    if edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError("Segment edges must contain at least two finite values.")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("Segment edges must be strictly increasing.")
    if edges[0] < 0.0 or edges[-1] > 1.0:
        raise ValueError("Normalized segment edges must lie within [0, 1].")

    values = np.clip(values, edges[0], edges[-1])
    masks = []
    for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.size - 2:
            mask = (values >= start) & (values <= stop)
        else:
            mask = (values >= start) & (values < stop)
        if not np.any(mask):
            raise ValueError(
                f"Segment {index} contains no tuning-curve bin centers."
            )
        masks.append(mask)
    if not np.all(np.sum(np.vstack(masks), axis=0) == 1):
        raise ValueError("Segment masks must assign every bin exactly once.")
    return tuple(masks)


def compute_segmented_shape_overlap(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    progression: np.ndarray,
    segment_edges: np.ndarray,
    *,
    eps: float = 1e-12,
    both_silent_score: float | None = None,
) -> dict[str, Any]:
    """Return unit-area overlap and mean rate separately for each segment."""
    values_a = np.asarray(curve_a, dtype=float)
    values_b = np.asarray(curve_b, dtype=float)
    progression = np.asarray(progression, dtype=float).reshape(-1)
    if (
        values_a.ndim != 1
        or values_b.ndim != 1
        or values_a.shape != values_b.shape
        or values_a.size != progression.size
    ):
        raise ValueError(
            "Segmented overlap requires aligned one-dimensional curves and "
            "progression bin centers."
        )
    masks = make_segment_masks(progression, segment_edges)
    scores: list[float] = []
    means_a: list[float] = []
    means_b: list[float] = []
    areas_a: list[float] = []
    areas_b: list[float] = []
    statuses: list[str] = []
    for mask in masks:
        segment_a = values_a[mask]
        segment_b = values_b[mask]
        status = "valid"
        score = np.nan
        mean_a = np.nan
        mean_b = np.nan
        area_a = np.nan
        area_b = np.nan
        if not np.isfinite(segment_a).any():
            status = "no_finite_a_bins"
        elif not np.isfinite(segment_b).any():
            status = "no_finite_b_bins"
        elif np.isinf(segment_a).any():
            status = "nonfinite_a_curve"
        elif np.isinf(segment_b).any():
            status = "nonfinite_b_curve"
        elif np.any(segment_a[np.isfinite(segment_a)] < -float(eps)) or np.any(
            segment_b[np.isfinite(segment_b)] < -float(eps)
        ):
            status = "negative_firing_rate"
        else:
            segment_a = np.maximum(interpolate_nans(segment_a), 0.0)
            segment_b = np.maximum(interpolate_nans(segment_b), 0.0)
            mean_a = float(np.mean(segment_a))
            mean_b = float(np.mean(segment_b))
            area_a = float(np.sum(segment_a))
            area_b = float(np.sum(segment_b))
            if area_a <= float(eps) and area_b <= float(eps):
                status = "both_silent"
                if both_silent_score is not None:
                    score = float(both_silent_score)
                    status = "valid"
            elif area_a <= float(eps) or area_b <= float(eps):
                score = 0.0
            else:
                score = float(
                    np.minimum(segment_a / area_a, segment_b / area_b).sum()
                )
                score = float(np.clip(score, 0.0, 1.0))
        scores.append(float(score))
        means_a.append(float(mean_a))
        means_b.append(float(mean_b))
        areas_a.append(float(area_a))
        areas_b.append(float(area_b))
        statuses.append(status)
    return {
        "scores": scores,
        "mean_rates_a_hz": means_a,
        "mean_rates_b_hz": means_b,
        "areas_a": areas_a,
        "areas_b": areas_b,
        "statuses": statuses,
    }


def flip_curve_if_requested(
    curve: np.ndarray,
    *,
    should_flip: bool,
) -> np.ndarray:
    """Return one tuning curve, optionally reversed along the path axis."""
    array = np.asarray(curve, dtype=float)
    if should_flip:
        return np.asarray(array[::-1], dtype=float)
    return array


def compute_similarity_score(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    similarity_metric: str,
    eps: float = 1e-12,
) -> float:
    """Return one similarity score for a pair of tuning curves."""
    curve_a = np.asarray(interpolate_nans(curve_a), dtype=float)
    curve_b = np.asarray(interpolate_nans(curve_b), dtype=float)

    valid = np.isfinite(curve_a) & np.isfinite(curve_b)
    if not np.any(valid):
        return np.nan

    curve_a = curve_a[valid]
    curve_b = curve_b[valid]

    if similarity_metric == "correlation":
        if np.std(curve_a) <= eps or np.std(curve_b) <= eps:
            return np.nan
        return float(np.corrcoef(curve_a, curve_b)[0, 1])

    if similarity_metric == "absolute_overlap":
        union = float(np.maximum(curve_a, curve_b).sum())
        if union <= eps:
            return np.nan
        intersection = float(np.minimum(curve_a, curve_b).sum())
        return intersection / union

    if similarity_metric == "shape_overlap":
        sum_a = float(curve_a.sum())
        sum_b = float(curve_b.sum())
        if sum_a <= eps or sum_b <= eps:
            return np.nan
        prob_a = curve_a / sum_a
        prob_b = curve_b / sum_b
        return float(np.minimum(prob_a, prob_b).sum())

    raise ValueError(f"Unsupported similarity metric: {similarity_metric!r}")


def compute_similarity_score_with_qc(
    curve_a: np.ndarray,
    curve_b: np.ndarray,
    similarity_metric: str,
) -> dict[str, Any]:
    """Return a similarity score plus pre-interpolation finite-bin QC."""
    values_a = np.asarray(curve_a, dtype=float)
    values_b = np.asarray(curve_b, dtype=float)
    if values_a.ndim != 1 or values_b.ndim != 1:
        raise ValueError("Similarity QC requires two one-dimensional tuning curves.")
    if values_a.shape != values_b.shape:
        raise ValueError(
            "Similarity QC requires tuning curves with matching shapes; "
            f"got {values_a.shape} and {values_b.shape}."
        )

    finite_a = np.isfinite(values_a)
    finite_b = np.isfinite(values_b)
    similarity = compute_similarity_score(
        values_a,
        values_b,
        similarity_metric=similarity_metric,
    )
    n_finite_a = int(np.count_nonzero(finite_a))
    n_finite_b = int(np.count_nonzero(finite_b))
    if np.isfinite(similarity):
        status = "valid"
    elif n_finite_a == 0 and n_finite_b == 0:
        status = "no_finite_bins_both_trajectories"
    elif n_finite_a == 0:
        status = "no_finite_bins_trajectory_a"
    elif n_finite_b == 0:
        status = "no_finite_bins_trajectory_b"
    else:
        status = "nonfinite_similarity"

    return {
        "similarity": float(similarity),
        "n_trajectory_a_finite_bins": n_finite_a,
        "n_trajectory_b_finite_bins": n_finite_b,
        "n_paired_finite_bins": int(np.count_nonzero(finite_a & finite_b)),
        "similarity_status": status,
    }


def get_similarity_axis_limits(similarity_metric: str) -> tuple[float, float]:
    """Return axis limits appropriate for the requested similarity metric."""
    if similarity_metric == "correlation":
        return -1.0, 1.0
    return 0.0, 1.0


def get_similarity_axis_label(similarity_metric: str) -> str:
    """Return a human-readable similarity label for figures."""
    if similarity_metric == "correlation":
        return "Correlation"
    if similarity_metric == "absolute_overlap":
        return "Absolute overlap"
    if similarity_metric == "shape_overlap":
        return "Shape overlap"
    return "Similarity"
