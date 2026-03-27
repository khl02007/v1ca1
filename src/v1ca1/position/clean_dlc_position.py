"""Detect and interpolate DLC position errors for one session epoch."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    get_analysis_path,
    load_position_timestamps,
)
from v1ca1.position._meters_per_pixel import get_session_meters_per_pixel
from v1ca1.position.fuse_head_position_imu import load_dlc_bodypart

BODY_PARTS = ("head", "body")
DEFAULT_OUTPUT_DIRNAME = "dlc_position_cleaned"
DEFAULT_OUTPUT_NAME_TEMPLATE = "{epoch}_dlc_position_cleaned.parquet"
DEFAULT_FIGURE_DIRNAME = "dlc_position_cleaned"
DEFAULT_FIGURE_NAME_TEMPLATE = "{epoch}_dlc_position_cleaned_2d.png"
DEFAULT_THRESHOLD_Z = 8.0
DEFAULT_MIN_JUMP_THRESHOLD_PX = 25.0
DEFAULT_LIKELIHOOD_LOSS_Z = 8.0
DEFAULT_MIN_PAIR_DISTANCE_THRESHOLD_PX = 50.0
DEFAULT_PAIR_DISTANCE_Z = 10.0
CM_PER_M = 100.0
CM_COLUMN_RENAMES = {
    "head_x_raw": "head_x_raw_cm",
    "head_y_raw": "head_y_raw_cm",
    "head_x_cleaned": "head_x_cleaned_cm",
    "head_y_cleaned": "head_y_cleaned_cm",
    "head_step_prev_px": "head_step_prev_cm",
    "head_step_next_px": "head_step_next_cm",
    "body_x_raw": "body_x_raw_cm",
    "body_y_raw": "body_y_raw_cm",
    "body_x_cleaned": "body_x_cleaned_cm",
    "body_y_cleaned": "body_y_cleaned_cm",
    "body_step_prev_px": "body_step_prev_cm",
    "body_step_next_px": "body_step_next_cm",
    "head_body_distance_px": "head_body_distance_cm",
}


def _validate_meters_per_pixel(meters_per_pixel: float) -> float:
    """Return one validated meters-per-pixel scale."""
    scale = float(meters_per_pixel)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("--meters-per-pixel must be a positive finite value.")
    return scale


def _resolve_position_scale(
    animal_name: str,
    date: str,
    meters_per_pixel: float | None,
) -> dict[str, float | str | None | Path]:
    """Resolve one per-session position scale from the CLI or session registry."""
    if meters_per_pixel is not None:
        scale = _validate_meters_per_pixel(meters_per_pixel)
        position_scale_source = "cli"
    else:
        try:
            configured_scale = get_session_meters_per_pixel(
                animal_name=animal_name,
                date=date,
            )
        except KeyError as exc:
            raise ValueError(
                "No meters-per-pixel was provided and no session mapping is configured. "
                "Pass --meters-per-pixel or add the session to "
                "v1ca1.position._meters_per_pixel.METERS_PER_PIXEL_BY_SESSION."
            ) from exc
        scale = _validate_meters_per_pixel(configured_scale)
        position_scale_source = "registry"

    return {
        "meters_per_pixel": scale,
        "cm_per_pixel": scale * CM_PER_M,
        "position_scale_source": position_scale_source,
        "position_scale_epoch_mapping": None,
        "position_scale_series_name": None,
        "nwb_path": None,
    }


def _convert_output_table_to_cm(
    output_table: pd.DataFrame,
    cm_per_pixel: float,
) -> pd.DataFrame:
    """Return one saved output table converted from pixels to centimeters."""
    converted = output_table.rename(columns=CM_COLUMN_RENAMES).copy()
    for column in CM_COLUMN_RENAMES.values():
        converted[column] = converted[column].to_numpy(dtype=float) * float(cm_per_pixel)
    return converted


def _step_prev_next(values_x: np.ndarray, values_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return step sizes aligned to the previous and next frame."""
    x = np.asarray(values_x, dtype=float).reshape(-1)
    y = np.asarray(values_y, dtype=float).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("values_x and values_y must have matching shapes.")

    step_size = np.hypot(np.diff(x), np.diff(y))
    step_prev_px = np.full(x.shape, np.nan, dtype=float)
    step_next_px = np.full(x.shape, np.nan, dtype=float)
    if step_size.size:
        step_prev_px[1:] = step_size
        step_next_px[:-1] = step_size
    return step_prev_px, step_next_px


def compute_low_likelihood_threshold(likelihood: np.ndarray) -> tuple[float, dict[str, float | int | str | None]]:
    """Return one low-likelihood threshold derived from the tail of -log10 likelihood."""
    likelihood_array = np.asarray(likelihood, dtype=float).reshape(-1)
    finite_likelihood = likelihood_array[np.isfinite(likelihood_array)]
    if finite_likelihood.size == 0:
        raise ValueError("likelihood does not contain any finite values.")

    clipped = np.clip(finite_likelihood, 1e-12, 1.0)
    loss = -np.log10(clipped)
    positive_loss = loss[loss > 0]
    if positive_loss.size == 0:
        return float(np.inf), {
            "threshold_source": "all_one",
            "likelihood_loss_threshold": None,
            "likelihood_loss_median": 0.0,
            "likelihood_loss_mad": 0.0,
            "likelihood_loss_robust_scale": 0.0,
            "reference_sample_count": 0,
        }

    loss_median = float(np.median(positive_loss))
    loss_mad = float(np.median(np.abs(positive_loss - loss_median)))
    robust_scale = 1.4826 * loss_mad if loss_mad > 0 else 0.0
    if robust_scale > 0:
        threshold = loss_median + (DEFAULT_LIKELIHOOD_LOSS_Z * robust_scale)
        source = "robust"
    else:
        threshold = float(np.nanpercentile(positive_loss, 99))
        source = "p99_fallback"

    return float(threshold), {
        "threshold_source": source,
        "likelihood_loss_threshold": float(threshold),
        "likelihood_loss_median": loss_median,
        "likelihood_loss_mad": loss_mad,
        "likelihood_loss_robust_scale": robust_scale,
        "reference_sample_count": int(positive_loss.size),
    }


def identify_low_likelihood_frames(
    likelihood: np.ndarray,
    loss_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one low-likelihood frame mask and the derived likelihood-loss values."""
    likelihood_array = np.asarray(likelihood, dtype=float).reshape(-1)
    loss = -np.log10(np.clip(likelihood_array, 1e-12, 1.0))
    if np.isinf(loss_threshold):
        return np.zeros(likelihood_array.shape, dtype=bool), loss
    return np.isfinite(loss) & (loss > float(loss_threshold)), loss


def compute_jump_threshold(
    step_size_px: np.ndarray,
    eligible_edge_mask: np.ndarray,
    threshold_z: float = DEFAULT_THRESHOLD_Z,
    min_jump_threshold_px: float = DEFAULT_MIN_JUMP_THRESHOLD_PX,
) -> tuple[float, dict[str, float | int | str]]:
    """Return one robust jump threshold and summary statistics."""
    steps = np.asarray(step_size_px, dtype=float).reshape(-1)
    eligible = np.asarray(eligible_edge_mask, dtype=bool).reshape(-1)
    if steps.shape != eligible.shape:
        raise ValueError("step_size_px and eligible_edge_mask must have matching shapes.")
    if threshold_z < 0:
        raise ValueError("--threshold-z must be non-negative.")
    if min_jump_threshold_px <= 0:
        raise ValueError("--min-jump-threshold-px must be positive.")

    finite_steps = np.isfinite(steps)
    reference_mask = finite_steps & eligible
    threshold_source = "eligible_edges"
    if not np.any(reference_mask):
        reference_mask = finite_steps
        threshold_source = "all_edges_fallback"
    if not np.any(reference_mask):
        return float(min_jump_threshold_px), {
            "threshold_source": "min_only",
            "step_median_px": 0.0,
            "step_mad_px": 0.0,
            "step_robust_scale_px": 0.0,
            "threshold_z": float(threshold_z),
            "min_jump_threshold_px": float(min_jump_threshold_px),
            "jump_threshold_px": float(min_jump_threshold_px),
            "reference_edge_count": 0,
        }

    reference_steps = steps[reference_mask]
    median_step = float(np.median(reference_steps))
    mad = float(np.median(np.abs(reference_steps - median_step)))
    robust_scale = 1.4826 * mad if mad > 0 else 0.0
    threshold = max(
        median_step + (float(threshold_z) * robust_scale),
        float(min_jump_threshold_px),
    )
    return float(threshold), {
        "threshold_source": threshold_source,
        "step_median_px": median_step,
        "step_mad_px": mad,
        "step_robust_scale_px": robust_scale,
        "threshold_z": float(threshold_z),
        "min_jump_threshold_px": float(min_jump_threshold_px),
        "jump_threshold_px": float(threshold),
        "reference_edge_count": int(reference_steps.size),
    }


def identify_jump_invalid_frames(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    jump_threshold_px: float,
    eligible_frame_mask: np.ndarray,
) -> np.ndarray:
    """Return one frame mask that marks jump-corrupted samples as invalid."""
    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    likelihood_array = np.asarray(likelihood, dtype=float).reshape(-1)
    eligible_frames = np.asarray(eligible_frame_mask, dtype=bool).reshape(-1)
    if not (
        x_array.shape
        == y_array.shape
        == likelihood_array.shape
        == eligible_frames.shape
    ):
        raise ValueError("x, y, likelihood, and eligible_frame_mask must have matching shapes.")
    if jump_threshold_px <= 0:
        raise ValueError("jump_threshold_px must be positive.")

    n_frames = x_array.size
    if n_frames == 0:
        raise ValueError("Position track is empty.")
    if n_frames == 1:
        return np.zeros(1, dtype=bool)

    step_size_px = np.hypot(np.diff(x_array), np.diff(y_array))
    eligible_edges = eligible_frames[:-1] & eligible_frames[1:]
    jump_edges = eligible_edges & np.isfinite(step_size_px) & (step_size_px > jump_threshold_px)
    invalid = np.zeros(n_frames, dtype=bool)

    for frame_index in range(1, n_frames - 1):
        prev_jump = bool(jump_edges[frame_index - 1])
        next_jump = bool(jump_edges[frame_index])
        if not (prev_jump and next_jump):
            continue

        bridge_step = float(
            np.hypot(
                x_array[frame_index + 1] - x_array[frame_index - 1],
                y_array[frame_index + 1] - y_array[frame_index - 1],
            )
        )
        if np.isfinite(bridge_step) and bridge_step <= jump_threshold_px:
            invalid[frame_index] = True

    for edge_index, is_jump in enumerate(jump_edges):
        if not is_jump:
            continue
        left_index = edge_index
        right_index = edge_index + 1
        if invalid[left_index] or invalid[right_index]:
            continue

        left_like = float(likelihood_array[left_index])
        right_like = float(likelihood_array[right_index])
        if left_like < right_like:
            invalid[left_index] = True
        elif right_like < left_like:
            invalid[right_index] = True
        else:
            invalid[right_index] = True

    return invalid


def compute_pair_distance_threshold(
    pair_distance_px: np.ndarray,
    eligible_frame_mask: np.ndarray,
) -> tuple[float, dict[str, float | int | str]]:
    """Return one robust head-body distance threshold."""
    distance = np.asarray(pair_distance_px, dtype=float).reshape(-1)
    eligible_frames = np.asarray(eligible_frame_mask, dtype=bool).reshape(-1)
    if distance.shape != eligible_frames.shape:
        raise ValueError("pair_distance_px and eligible_frame_mask must have matching shapes.")

    finite_distance = np.isfinite(distance)
    reference_mask = finite_distance & eligible_frames
    threshold_source = "provisional_valid"
    if not np.any(reference_mask):
        reference_mask = finite_distance
        threshold_source = "all_frames_fallback"
    if not np.any(reference_mask):
        return float(DEFAULT_MIN_PAIR_DISTANCE_THRESHOLD_PX), {
            "threshold_source": "min_only",
            "pair_distance_median_px": 0.0,
            "pair_distance_mad_px": 0.0,
            "pair_distance_robust_scale_px": 0.0,
            "pair_distance_threshold_px": float(DEFAULT_MIN_PAIR_DISTANCE_THRESHOLD_PX),
            "reference_frame_count": 0,
        }

    reference_distance = distance[reference_mask]
    median_distance = float(np.median(reference_distance))
    mad = float(np.median(np.abs(reference_distance - median_distance)))
    robust_scale = 1.4826 * mad if mad > 0 else 0.0
    threshold = max(
        median_distance + (DEFAULT_PAIR_DISTANCE_Z * robust_scale),
        float(DEFAULT_MIN_PAIR_DISTANCE_THRESHOLD_PX),
    )
    return float(threshold), {
        "threshold_source": threshold_source,
        "pair_distance_median_px": median_distance,
        "pair_distance_mad_px": mad,
        "pair_distance_robust_scale_px": robust_scale,
        "pair_distance_threshold_px": float(threshold),
        "reference_frame_count": int(reference_distance.size),
    }


def identify_pair_distance_invalid_frames(
    head_likelihood: np.ndarray,
    body_likelihood: np.ndarray,
    pair_distance_px: np.ndarray,
    pair_threshold_px: float,
    provisional_head_valid: np.ndarray,
    provisional_body_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invalidate the lower-likelihood label when head-body distance is implausible."""
    head_like = np.asarray(head_likelihood, dtype=float).reshape(-1)
    body_like = np.asarray(body_likelihood, dtype=float).reshape(-1)
    pair_distance = np.asarray(pair_distance_px, dtype=float).reshape(-1)
    head_valid = np.asarray(provisional_head_valid, dtype=bool).reshape(-1)
    body_valid = np.asarray(provisional_body_valid, dtype=bool).reshape(-1)
    if not (
        head_like.shape
        == body_like.shape
        == pair_distance.shape
        == head_valid.shape
        == body_valid.shape
    ):
        raise ValueError("Pair-distance inputs must have matching shapes.")

    pair_distance_invalid = (
        head_valid
        & body_valid
        & np.isfinite(pair_distance)
        & (pair_distance > float(pair_threshold_px))
    )
    head_pair_invalid = pair_distance_invalid & (head_like <= body_like)
    body_pair_invalid = pair_distance_invalid & ~head_pair_invalid
    return head_pair_invalid, body_pair_invalid, pair_distance_invalid


def _interpolate_1d_gaps(
    values: np.ndarray,
    frame_times: np.ndarray,
    frame_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fill only interior NaN gaps and record the interpolation anchors."""
    value_array = np.asarray(values, dtype=float).reshape(-1)
    time_array = np.asarray(frame_times, dtype=float).reshape(-1)
    frame_array = np.asarray(frame_numbers, dtype=float).reshape(-1)
    if not (value_array.shape == time_array.shape == frame_array.shape):
        raise ValueError("values, frame_times, and frame_numbers must have matching shapes.")

    valid = np.isfinite(value_array)
    interpolated = value_array.copy()
    left_sources = np.full(value_array.shape, np.nan, dtype=float)
    right_sources = np.full(value_array.shape, np.nan, dtype=float)
    interpolated_mask = np.zeros(value_array.shape, dtype=bool)

    if int(np.sum(valid)) < 2:
        return interpolated, interpolated_mask, left_sources, right_sources

    valid_indices = np.flatnonzero(valid)
    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])
    if last_valid <= first_valid:
        return interpolated, interpolated_mask, left_sources, right_sources

    filled = np.interp(time_array, time_array[valid], value_array[valid])
    index = first_valid + 1
    while index < last_valid:
        if valid[index]:
            index += 1
            continue
        stop = index
        while stop <= last_valid and not valid[stop]:
            stop += 1
        left_index = index - 1
        right_index = stop
        if left_index >= first_valid and right_index <= last_valid:
            interpolated[index:stop] = filled[index:stop]
            interpolated_mask[index:stop] = True
            left_sources[index:stop] = frame_array[left_index]
            right_sources[index:stop] = frame_array[right_index]
        index = stop

    return interpolated, interpolated_mask, left_sources, right_sources


def _clean_one_bodypart(
    raw_table: pd.DataFrame,
    frame_times: np.ndarray,
    threshold_z: float,
    min_jump_threshold_px: float,
) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    """Clean one bodypart using low-likelihood and jump rejection."""
    x_raw = raw_table["position_x_raw"].to_numpy(dtype=float)
    y_raw = raw_table["position_y_raw"].to_numpy(dtype=float)
    likelihood = raw_table["likelihood"].to_numpy(dtype=float)
    frame_numbers = raw_table["frame"].to_numpy(dtype=int)

    loss_threshold, loss_metadata = compute_low_likelihood_threshold(likelihood)
    low_likelihood, likelihood_loss = identify_low_likelihood_frames(
        likelihood=likelihood,
        loss_threshold=loss_threshold,
    )
    step_size_px = np.hypot(np.diff(x_raw), np.diff(y_raw))
    eligible_edge_mask = ~low_likelihood[:-1] & ~low_likelihood[1:]
    jump_threshold_px, jump_metadata = compute_jump_threshold(
        step_size_px=step_size_px,
        eligible_edge_mask=eligible_edge_mask,
        threshold_z=threshold_z,
        min_jump_threshold_px=min_jump_threshold_px,
    )
    jump_invalid = identify_jump_invalid_frames(
        x=x_raw,
        y=y_raw,
        likelihood=likelihood,
        jump_threshold_px=jump_threshold_px,
        eligible_frame_mask=~low_likelihood,
    )

    provisional_invalid = low_likelihood | jump_invalid
    provisional_x = x_raw.copy()
    provisional_y = y_raw.copy()
    provisional_x[provisional_invalid] = np.nan
    provisional_y[provisional_invalid] = np.nan
    step_prev_px, step_next_px = _step_prev_next(x_raw, y_raw)

    return {
        "frame": frame_numbers,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "likelihood": likelihood,
        "likelihood_loss": likelihood_loss,
        "low_likelihood": low_likelihood,
        "jump_invalid": jump_invalid,
        "provisional_invalid": provisional_invalid,
        "provisional_x": provisional_x,
        "provisional_y": provisional_y,
        "step_prev_px": step_prev_px,
        "step_next_px": step_next_px,
        "frame_times": np.asarray(frame_times, dtype=float),
    }, {
        "low_likelihood": loss_metadata,
        "jump": jump_metadata,
    }


def _finalize_one_bodypart(
    cleaned: dict[str, np.ndarray | float],
    pair_invalid: np.ndarray,
) -> tuple[dict[str, np.ndarray | float], dict[str, int]]:
    """Apply pair-distance invalidation and interpolate one bodypart."""
    x_raw = np.asarray(cleaned["x_raw"], dtype=float)
    y_raw = np.asarray(cleaned["y_raw"], dtype=float)
    frame_times = np.asarray(cleaned["frame_times"], dtype=float)
    frame_numbers = np.asarray(cleaned["frame"], dtype=float)
    low_likelihood = np.asarray(cleaned["low_likelihood"], dtype=bool)
    jump_invalid = np.asarray(cleaned["jump_invalid"], dtype=bool)
    pair_invalid_array = np.asarray(pair_invalid, dtype=bool)
    final_invalid = low_likelihood | jump_invalid | pair_invalid_array

    x_with_nans = x_raw.copy()
    y_with_nans = y_raw.copy()
    x_with_nans[final_invalid] = np.nan
    y_with_nans[final_invalid] = np.nan

    x_cleaned, interpolated_x, left_source, right_source = _interpolate_1d_gaps(
        x_with_nans,
        frame_times=frame_times,
        frame_numbers=frame_numbers,
    )
    y_cleaned, interpolated_y, _left_source_y, _right_source_y = _interpolate_1d_gaps(
        y_with_nans,
        frame_times=frame_times,
        frame_numbers=frame_numbers,
    )
    interpolated = interpolated_x | interpolated_y
    remaining_edge_nan = np.isnan(x_cleaned) | np.isnan(y_cleaned)

    return {
        **cleaned,
        "pair_invalid": pair_invalid_array,
        "final_invalid": final_invalid,
        "x_cleaned": x_cleaned,
        "y_cleaned": y_cleaned,
        "interpolated": interpolated,
        "interpolation_source_left": left_source,
        "interpolation_source_right": right_source,
        "remaining_edge_nan": remaining_edge_nan,
    }, {
        "low_likelihood_frame_count": int(np.sum(low_likelihood)),
        "jump_invalid_frame_count": int(np.sum(jump_invalid)),
        "pair_invalid_frame_count": int(np.sum(pair_invalid_array)),
        "final_invalid_frame_count": int(np.sum(final_invalid)),
        "interpolated_frame_count": int(np.sum(interpolated)),
        "remaining_edge_nan_frame_count": int(np.sum(remaining_edge_nan)),
    }


def run_joint_position_cleaning(
    head_table: pd.DataFrame,
    body_table: pd.DataFrame,
    frame_times: np.ndarray,
    threshold_z: float = DEFAULT_THRESHOLD_Z,
    min_jump_threshold_px: float = DEFAULT_MIN_JUMP_THRESHOLD_PX,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect low-likelihood, jump, and pair-distance errors jointly for head and body."""
    if head_table.shape[0] != len(frame_times):
        raise ValueError(
            "Head DLC row count does not match the frame timestamp count: "
            f"{head_table.shape[0]} vs {len(frame_times)}."
        )
    if body_table.shape[0] != len(frame_times):
        raise ValueError(
            "Body DLC row count does not match the frame timestamp count: "
            f"{body_table.shape[0]} vs {len(frame_times)}."
        )

    head_cleaned, head_metadata = _clean_one_bodypart(
        raw_table=head_table,
        frame_times=frame_times,
        threshold_z=threshold_z,
        min_jump_threshold_px=min_jump_threshold_px,
    )
    body_cleaned, body_metadata = _clean_one_bodypart(
        raw_table=body_table,
        frame_times=frame_times,
        threshold_z=threshold_z,
        min_jump_threshold_px=min_jump_threshold_px,
    )

    provisional_head_valid = ~np.asarray(head_cleaned["provisional_invalid"], dtype=bool)
    provisional_body_valid = ~np.asarray(body_cleaned["provisional_invalid"], dtype=bool)
    head_body_distance_px = np.hypot(
        np.asarray(head_cleaned["x_raw"], dtype=float) - np.asarray(body_cleaned["x_raw"], dtype=float),
        np.asarray(head_cleaned["y_raw"], dtype=float) - np.asarray(body_cleaned["y_raw"], dtype=float),
    )
    pair_threshold_px, pair_metadata = compute_pair_distance_threshold(
        pair_distance_px=head_body_distance_px,
        eligible_frame_mask=provisional_head_valid & provisional_body_valid,
    )
    head_pair_invalid, body_pair_invalid, pair_distance_invalid = identify_pair_distance_invalid_frames(
        head_likelihood=np.asarray(head_cleaned["likelihood"], dtype=float),
        body_likelihood=np.asarray(body_cleaned["likelihood"], dtype=float),
        pair_distance_px=head_body_distance_px,
        pair_threshold_px=pair_threshold_px,
        provisional_head_valid=provisional_head_valid,
        provisional_body_valid=provisional_body_valid,
    )

    head_final, head_counts = _finalize_one_bodypart(head_cleaned, head_pair_invalid)
    body_final, body_counts = _finalize_one_bodypart(body_cleaned, body_pair_invalid)
    frame_numbers = np.asarray(head_final["frame"], dtype=int)

    diagnostics = pd.DataFrame(
        {
            "frame": frame_numbers,
            "frame_time_s": np.asarray(frame_times, dtype=float),
            "head_x_raw": np.asarray(head_final["x_raw"], dtype=float),
            "head_y_raw": np.asarray(head_final["y_raw"], dtype=float),
            "head_likelihood": np.asarray(head_final["likelihood"], dtype=float),
            "head_likelihood_loss": np.asarray(head_final["likelihood_loss"], dtype=float),
            "head_x_cleaned": np.asarray(head_final["x_cleaned"], dtype=float),
            "head_y_cleaned": np.asarray(head_final["y_cleaned"], dtype=float),
            "head_step_prev_px": np.asarray(head_final["step_prev_px"], dtype=float),
            "head_step_next_px": np.asarray(head_final["step_next_px"], dtype=float),
            "head_low_likelihood": np.asarray(head_final["low_likelihood"], dtype=bool),
            "head_jump_invalid": np.asarray(head_final["jump_invalid"], dtype=bool),
            "head_pair_invalid": np.asarray(head_final["pair_invalid"], dtype=bool),
            "head_invalid": np.asarray(head_final["final_invalid"], dtype=bool),
            "head_interpolated": np.asarray(head_final["interpolated"], dtype=bool),
            "head_interpolation_source_left": np.asarray(
                head_final["interpolation_source_left"], dtype=float
            ),
            "head_interpolation_source_right": np.asarray(
                head_final["interpolation_source_right"], dtype=float
            ),
            "body_x_raw": np.asarray(body_final["x_raw"], dtype=float),
            "body_y_raw": np.asarray(body_final["y_raw"], dtype=float),
            "body_likelihood": np.asarray(body_final["likelihood"], dtype=float),
            "body_likelihood_loss": np.asarray(body_final["likelihood_loss"], dtype=float),
            "body_x_cleaned": np.asarray(body_final["x_cleaned"], dtype=float),
            "body_y_cleaned": np.asarray(body_final["y_cleaned"], dtype=float),
            "body_step_prev_px": np.asarray(body_final["step_prev_px"], dtype=float),
            "body_step_next_px": np.asarray(body_final["step_next_px"], dtype=float),
            "body_low_likelihood": np.asarray(body_final["low_likelihood"], dtype=bool),
            "body_jump_invalid": np.asarray(body_final["jump_invalid"], dtype=bool),
            "body_pair_invalid": np.asarray(body_final["pair_invalid"], dtype=bool),
            "body_invalid": np.asarray(body_final["final_invalid"], dtype=bool),
            "body_interpolated": np.asarray(body_final["interpolated"], dtype=bool),
            "body_interpolation_source_left": np.asarray(
                body_final["interpolation_source_left"], dtype=float
            ),
            "body_interpolation_source_right": np.asarray(
                body_final["interpolation_source_right"], dtype=float
            ),
            "head_body_distance_px": head_body_distance_px,
            "pair_distance_invalid": pair_distance_invalid.astype(bool),
        }
    )
    metadata = {
        "frame_count": int(len(frame_times)),
        "head": {
            **head_metadata,
            "counts": head_counts,
        },
        "body": {
            **body_metadata,
            "counts": body_counts,
        },
        "pair_distance": {
            **pair_metadata,
            "pair_distance_invalid_frame_count": int(np.sum(pair_distance_invalid)),
        },
    }
    return diagnostics, metadata


def resolve_output_dir(analysis_path: Path) -> Path:
    """Return the dedicated output directory for cleaned position artifacts."""
    return analysis_path / DEFAULT_OUTPUT_DIRNAME


def resolve_figure_dir(analysis_path: Path) -> Path:
    """Return the session figure directory for cleaned position plots."""
    return analysis_path / "figs" / DEFAULT_FIGURE_DIRNAME


def save_before_after_position_figure(
    output_table: pd.DataFrame,
    output_path: Path,
    epoch: str,
) -> Path:
    """Save a 2x2 before/after position figure for head and body."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = (
        ("Head Before", "head_x_raw_cm", "head_y_raw_cm"),
        ("Head After", "head_x_cleaned_cm", "head_y_cleaned_cm"),
        ("Body Before", "body_x_raw_cm", "body_y_raw_cm"),
        ("Body After", "body_x_cleaned_cm", "body_y_cleaned_cm"),
    )
    for axis, (title, x_column, y_column) in zip(axes.reshape(-1), panels):
        axis.plot(
            output_table[x_column].to_numpy(dtype=float),
            output_table[y_column].to_numpy(dtype=float),
            linewidth=0.6,
            alpha=0.9,
        )
        axis.set_title(title)
        axis.set_xlabel("x (cm)")
        axis.set_ylabel("y (cm)")
        axis.set_aspect("equal", adjustable="box")

    figure.suptitle(f"{epoch} head/body position before/after cleaning")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def clean_dlc_position(
    animal_name: str,
    date: str,
    epoch: str,
    dlc_h5_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    output_dirname: str = DEFAULT_OUTPUT_DIRNAME,
    output_name_template: str = DEFAULT_OUTPUT_NAME_TEMPLATE,
    threshold_z: float = DEFAULT_THRESHOLD_Z,
    min_jump_threshold_px: float = DEFAULT_MIN_JUMP_THRESHOLD_PX,
    meters_per_pixel: float | None = None,
) -> Path:
    """Clean one epoch's head/body DLC tracks and save one per-epoch parquet."""
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not output_dirname:
        raise ValueError("--output-dirname must be a non-empty string.")
    scale_metadata = _resolve_position_scale(
        animal_name=animal_name,
        date=date,
        meters_per_pixel=meters_per_pixel,
    )

    print(f"Processing {animal_name} {date} epoch {epoch}.")
    _epoch_tags, position_timestamps, timestamp_source = load_position_timestamps(analysis_path)
    if epoch not in position_timestamps:
        raise ValueError(
            f"Epoch {epoch!r} not found in saved position timestamps. "
            f"Available epochs: {sorted(position_timestamps)!r}"
        )

    frame_times = np.asarray(position_timestamps[epoch], dtype=float)
    raw_tables: dict[str, pd.DataFrame] = {}
    for bodypart in BODY_PARTS:
        raw_table = load_dlc_bodypart(
            dlc_h5_path=dlc_h5_path,
            bodypart=bodypart,
        ).rename(columns={"x": "position_x_raw", "y": "position_y_raw"})
        if raw_table.shape[0] != frame_times.size:
            raise ValueError(
                f"{bodypart.capitalize()} DLC row count does not match saved position timestamps "
                f"for the requested epoch: {raw_table.shape[0]} vs {frame_times.size}."
            )
        raw_tables[bodypart] = raw_table

    diagnostics, cleaning_metadata = run_joint_position_cleaning(
        head_table=raw_tables["head"],
        body_table=raw_tables["body"],
        frame_times=frame_times,
        threshold_z=threshold_z,
        min_jump_threshold_px=min_jump_threshold_px,
    )

    output_table = diagnostics.copy()
    output_table.insert(0, "epoch", epoch)
    output_table = _convert_output_table_to_cm(
        output_table=output_table,
        cm_per_pixel=float(scale_metadata["cm_per_pixel"]),
    )

    output_dir = analysis_path / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name_template.format(epoch=epoch)
    output_table.to_parquet(output_path, index=False)
    figure_dir = resolve_figure_dir(analysis_path)
    figure_path = save_before_after_position_figure(
        output_table=output_table,
        output_path=figure_dir / DEFAULT_FIGURE_NAME_TEMPLATE.format(epoch=epoch),
        epoch=epoch,
    )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.position.clean_dlc_position",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "dlc_h5_path": dlc_h5_path,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "output_dirname": output_dirname,
            "output_name_template": output_name_template,
            "threshold_z": threshold_z,
            "min_jump_threshold_px": min_jump_threshold_px,
            "meters_per_pixel": meters_per_pixel,
        },
        outputs={
            "position_timestamp_source": timestamp_source,
            "frame_count": int(frame_times.size),
            "cleaned_output_path": output_path,
            "figure_path": figure_path,
            "cleaning_metadata": cleaning_metadata,
            **scale_metadata,
        },
    )
    print(f"Saved cleaned head/body position parquet to {output_path}")
    print(f"Saved before/after figure to {figure_path}")
    print(f"Saved run metadata to {log_path}")
    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the DLC position cleaner CLI."""
    parser = argparse.ArgumentParser(
        description="Detect and interpolate head/body DLC position errors in one epoch."
    )
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name.",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--epoch",
        required=True,
        help="Epoch label to clean.",
    )
    parser.add_argument(
        "--dlc-h5-path",
        type=Path,
        required=True,
        help="Path to the DeepLabCut H5 file containing head and body tracks.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=(
            "Dedicated output directory created under the session analysis directory. "
            f"Default: {DEFAULT_OUTPUT_DIRNAME}"
        ),
    )
    parser.add_argument(
        "--output-name-template",
        default=DEFAULT_OUTPUT_NAME_TEMPLATE,
        help=(
            "Output parquet filename template written under the dedicated output directory. "
            f"Default: {DEFAULT_OUTPUT_NAME_TEMPLATE}"
        ),
    )
    parser.add_argument(
        "--threshold-z",
        type=float,
        default=DEFAULT_THRESHOLD_Z,
        help=f"Robust z-score multiplier used to define jump size. Default: {DEFAULT_THRESHOLD_Z}",
    )
    parser.add_argument(
        "--min-jump-threshold-px",
        type=float,
        default=DEFAULT_MIN_JUMP_THRESHOLD_PX,
        help=f"Hard lower bound on the jump threshold in pixels. Default: {DEFAULT_MIN_JUMP_THRESHOLD_PX}",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=None,
        help=(
            "Optional pixel scale override in meters per pixel. When omitted, the script "
            "uses the session registry in v1ca1.position._meters_per_pixel."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the joint head/body position cleaner CLI."""
    args = parse_arguments()
    clean_dlc_position(
        animal_name=args.animal_name,
        date=args.date,
        epoch=args.epoch,
        dlc_h5_path=args.dlc_h5_path,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        output_dirname=args.output_dirname,
        output_name_template=args.output_name_template,
        threshold_z=args.threshold_z,
        min_jump_threshold_px=args.min_jump_threshold_px,
        meters_per_pixel=args.meters_per_pixel,
    )


clean_head_position_jumps = clean_dlc_position


if __name__ == "__main__":
    main()
