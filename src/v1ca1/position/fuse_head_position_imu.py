"""Fuse DLC head tracking with extracted IMU events for one session epoch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.helper.extract_imu_from_rec import IMU_TABLE_COLUMNS
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    coerce_position_array,
    get_analysis_path,
    load_ephys_timestamps_by_epoch,
    load_position_data,
    load_position_timestamps,
    save_pickle_output,
)

DEFAULT_OUTPUT_NAME = "head_position_imu.pkl"
DEFAULT_DIAGNOSTICS_NAME_TEMPLATE = "{epoch}_head_position_imu_diagnostics.parquet"
DEFAULT_LIKELIHOOD_THRESHOLD = 0.90
DEFAULT_MAX_GAP_S = 1.0
DEFAULT_BASE_MEASUREMENT_SIGMA_PX = 4.0
DEFAULT_BASE_PROCESS_ACCEL_PX_S2 = 60.0
DEFAULT_MOTION_NOISE_GAIN = 2.5
DEFAULT_FUSION_MODEL = "legacy"
ROBUST_FUSION_MODEL = "robust_v2"
FUSION_MODEL_CHOICES = (DEFAULT_FUSION_MODEL, ROBUST_FUSION_MODEL)
DEFAULT_ROBUST_WINDOW = 5
DEFAULT_ROBUST_ITERATIONS = 3
DEFAULT_ROBUST_WEIGHT_FLOOR = 0.02
DEFAULT_ROBUST_HARD_LIKELIHOOD_FLOOR = 0.05
DEFAULT_ROBUST_SPIKE_Z = 3.5
DEFAULT_ROBUST_INNOVATION_TUNING = 2.5
DEFAULT_MIN_SPEED_FOR_TURN_PX_S = 5.0
DEFAULT_MIN_MOTION_FRAMES_FOR_AXIS_FIT = 50
GYRO_AXIS_NAMES = ("gyro_x_mean", "gyro_y_mean", "gyro_z_mean")
FRAME_IMU_COLUMNS = [
    "accel_x_mean",
    "accel_y_mean",
    "accel_z_mean",
    "gyro_x_mean",
    "gyro_y_mean",
    "gyro_z_mean",
    "accel_mag_mean",
    "accel_mag_std",
    "gyro_mag_mean",
    "accel_update_count",
    "gyro_update_count",
]


def _get_positive_dt_fallback(dt: np.ndarray, default: float = 1.0) -> float:
    """Return a robust positive time-step fallback."""
    dt_array = np.asarray(dt, dtype=float)
    positive_dt = dt_array[dt_array > 0]
    if positive_dt.size == 0:
        return float(default)
    return float(np.nanmedian(positive_dt))


def _normalize_interval_times(
    interval_starts: np.ndarray,
    interval_stops: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned frame interval bounds with non-decreasing stops."""
    starts = np.asarray(interval_starts, dtype=float).reshape(-1)
    stops = np.asarray(interval_stops, dtype=float).reshape(-1)
    if starts.shape != stops.shape:
        raise ValueError(
            "Frame interval starts and stops must have the same shape. "
            f"Got {starts.shape} and {stops.shape}."
        )
    if starts.size == 0:
        raise ValueError("Frame intervals are empty.")
    if np.any(~np.isfinite(starts)) or np.any(~np.isfinite(stops)):
        raise ValueError("Frame interval bounds must be finite.")
    if np.any(stops < starts):
        raise ValueError("Frame interval stop times must be >= start times.")
    return starts, stops


def load_head_dlc(dlc_h5_path: Path) -> pd.DataFrame:
    """Load one head-point DLC H5 into a flat dataframe."""
    if not dlc_h5_path.exists():
        raise FileNotFoundError(f"DLC H5 file not found: {dlc_h5_path}")

    table = pd.read_hdf(dlc_h5_path)
    if not isinstance(table.columns, pd.MultiIndex) or table.columns.nlevels < 3:
        raise ValueError(
            "DLC H5 must use MultiIndex columns with scorer/bodypart/coordinate levels."
        )

    scorer = table.columns.get_level_values(0)[0]
    scorer_table = table[scorer]
    bodyparts = scorer_table.columns.get_level_values(0).astype(str)
    if "head" not in set(bodyparts):
        raise ValueError(f"DLC H5 does not contain the required 'head' bodypart: {dlc_h5_path}")

    head_table = scorer_table["head"]
    missing_columns = [
        column for column in ("x", "y", "likelihood") if column not in head_table.columns
    ]
    if missing_columns:
        raise ValueError(
            "DLC H5 head track is missing required columns: "
            f"{missing_columns!r} in {dlc_h5_path}"
        )

    head = head_table[["x", "y", "likelihood"]].copy()
    head = head.rename(columns={"x": "head_x_raw", "y": "head_y_raw"})
    head.index.name = "frame"
    return head.reset_index()


def load_imu_table(imu_path: Path) -> pd.DataFrame:
    """Load one extracted IMU parquet and validate required columns."""
    if not imu_path.exists():
        raise FileNotFoundError(f"IMU parquet not found: {imu_path}")

    table = pd.read_parquet(imu_path)
    missing_columns = [column for column in IMU_TABLE_COLUMNS if column not in table.columns]
    if missing_columns:
        raise ValueError(
            "IMU parquet is missing required columns: "
            f"{missing_columns!r} in {imu_path}"
        )
    if table.empty:
        raise ValueError(f"IMU parquet contains no rows: {imu_path}")

    cleaned = table.loc[:, IMU_TABLE_COLUMNS].copy()
    if cleaned.isnull().any().any():
        raise ValueError(f"IMU parquet contains NaN values: {imu_path}")

    accel_time = cleaned["accel_time"].to_numpy(dtype=float)
    gyro_time = cleaned["gyro_time"].to_numpy(dtype=float)
    if np.any(np.diff(accel_time) < 0):
        raise ValueError("IMU accel_time values must be sorted ascending.")
    if np.any(np.diff(gyro_time) < 0):
        raise ValueError("IMU gyro_time values must be sorted ascending.")
    return cleaned


def _weighted_mean_and_std(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted mean and standard deviation for one value array."""
    normalized_weights = np.asarray(weights, dtype=float).reshape(-1)
    value_array = np.asarray(values, dtype=float)
    if value_array.shape[0] != normalized_weights.size:
        raise ValueError(
            "Weighted value count does not match weight count. "
            f"Got {value_array.shape[0]} and {normalized_weights.size}."
        )

    total_weight = float(np.sum(normalized_weights))
    if total_weight <= 0:
        normalized_weights = np.ones_like(normalized_weights, dtype=float)

    mean = np.average(value_array, axis=0, weights=normalized_weights)
    variance = np.average(
        np.square(value_array - mean),
        axis=0,
        weights=normalized_weights,
    )
    return np.asarray(mean, dtype=float), np.sqrt(np.asarray(variance, dtype=float))


def _summarize_sensor_intervals(
    event_times: np.ndarray,
    event_values: np.ndarray,
    interval_starts: np.ndarray,
    interval_stops: np.ndarray,
    sensor_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize one sample-and-hold sensor stream over frame intervals."""
    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    event_values = np.asarray(event_values, dtype=float)
    starts, stops = _normalize_interval_times(interval_starts, interval_stops)

    if event_times.size == 0:
        raise ValueError(f"{sensor_name} event times are empty.")
    if event_values.ndim != 2 or event_values.shape[1] != 3:
        raise ValueError(
            f"{sensor_name} values must have shape (n_events, 3). "
            f"Found {event_values.shape}."
        )
    if event_values.shape[0] != event_times.size:
        raise ValueError(
            f"{sensor_name} event time count does not match value rows: "
            f"{event_times.size} vs {event_values.shape[0]}."
        )
    if np.any(~np.isfinite(event_times)) or np.any(~np.isfinite(event_values)):
        raise ValueError(f"{sensor_name} events must be finite.")
    if np.any(np.diff(event_times) < 0):
        raise ValueError(f"{sensor_name} event times must be sorted ascending.")

    means = np.empty((starts.size, 3), dtype=float)
    magnitude_mean = np.empty(starts.size, dtype=float)
    magnitude_std = np.empty(starts.size, dtype=float)
    update_count = np.empty(starts.size, dtype=np.int64)

    initial_value = np.asarray(event_values[0], dtype=float)
    for index, (start, stop) in enumerate(zip(starts, stops)):
        first_update = int(np.searchsorted(event_times, start, side="left"))
        after_stop = int(np.searchsorted(event_times, stop, side="left"))
        last_before = first_update - 1

        current_value = initial_value if last_before < 0 else np.asarray(event_values[last_before], dtype=float)
        cursor_time = float(start)
        segment_values: list[np.ndarray] = []
        segment_weights: list[float] = []

        for event_index in range(first_update, after_stop):
            event_time = float(event_times[event_index])
            if event_time > cursor_time:
                segment_values.append(np.asarray(current_value, dtype=float))
                segment_weights.append(event_time - cursor_time)
            current_value = np.asarray(event_values[event_index], dtype=float)
            cursor_time = event_time

        if stop > cursor_time:
            segment_values.append(np.asarray(current_value, dtype=float))
            segment_weights.append(float(stop - cursor_time))

        if not segment_values:
            segment_values = [np.asarray(current_value, dtype=float)]
            segment_weights = [1.0]

        values = np.vstack(segment_values)
        weights = np.asarray(segment_weights, dtype=float)
        mean, _ = _weighted_mean_and_std(values, weights)
        magnitudes = np.linalg.norm(values, axis=1)
        magnitude_mean_i, magnitude_std_i = _weighted_mean_and_std(magnitudes, weights)

        means[index] = mean
        magnitude_mean[index] = float(np.asarray(magnitude_mean_i).reshape(-1)[0])
        magnitude_std[index] = float(np.asarray(magnitude_std_i).reshape(-1)[0])
        update_count[index] = after_stop - first_update

    return means, magnitude_mean, magnitude_std, update_count


def build_frame_imu_features(
    imu_table: pd.DataFrame,
    frame_times: np.ndarray,
    epoch_start_time: float,
) -> pd.DataFrame:
    """Aggregate true IMU update events into one row per video frame."""
    frame_times = np.asarray(frame_times, dtype=float).reshape(-1)
    if frame_times.size == 0:
        raise ValueError("Frame timestamps are empty.")
    if np.any(~np.isfinite(frame_times)):
        raise ValueError("Frame timestamps must be finite.")
    if np.any(np.diff(frame_times) < 0):
        raise ValueError("Frame timestamps must be sorted ascending.")

    interval_starts = np.empty_like(frame_times)
    interval_starts[0] = min(float(epoch_start_time), float(frame_times[0]))
    interval_starts[1:] = frame_times[:-1]
    interval_stops = frame_times.copy()

    accel_values = imu_table[["accel_x_g", "accel_y_g", "accel_z_g"]].to_numpy(dtype=float)
    gyro_values = imu_table[["gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s"]].to_numpy(dtype=float)
    accel_means, accel_mag_mean, accel_mag_std, accel_update_count = _summarize_sensor_intervals(
        event_times=imu_table["accel_time"].to_numpy(dtype=float),
        event_values=accel_values,
        interval_starts=interval_starts,
        interval_stops=interval_stops,
        sensor_name="Accel",
    )
    gyro_means, gyro_mag_mean, _gyro_mag_std, gyro_update_count = _summarize_sensor_intervals(
        event_times=imu_table["gyro_time"].to_numpy(dtype=float),
        event_values=gyro_values,
        interval_starts=interval_starts,
        interval_stops=interval_stops,
        sensor_name="Gyro",
    )

    return pd.DataFrame(
        {
            "accel_x_mean": accel_means[:, 0],
            "accel_y_mean": accel_means[:, 1],
            "accel_z_mean": accel_means[:, 2],
            "gyro_x_mean": gyro_means[:, 0],
            "gyro_y_mean": gyro_means[:, 1],
            "gyro_z_mean": gyro_means[:, 2],
            "accel_mag_mean": accel_mag_mean,
            "accel_mag_std": accel_mag_std,
            "gyro_mag_mean": gyro_mag_mean,
            "accel_update_count": accel_update_count,
            "gyro_update_count": gyro_update_count,
        },
        columns=FRAME_IMU_COLUMNS,
    )


def compute_visual_outlier_mask(
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    frame_times: np.ndarray,
    likelihood_threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Flag implausibly large frame-to-frame jumps in raw DLC position."""
    if x.size < 2:
        return np.zeros(x.size, dtype=bool), {
            "speed_threshold_px_s": 15.0,
            "speed_median_px_s": 0.0,
            "speed_sigma_px_s": 1.0,
        }

    low_confidence = likelihood < likelihood_threshold
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(frame_times)
    fallback_dt = _get_positive_dt_fallback(dt, default=1.0)
    dt = np.where(dt <= 0, fallback_dt, dt)
    speed = np.hypot(dx, dy) / dt

    valid_speed = ~low_confidence[:-1] & ~low_confidence[1:] & np.isfinite(speed)
    reference = speed[valid_speed]
    if reference.size < 20:
        median_speed = float(np.nanmedian(speed)) if np.any(np.isfinite(speed)) else 0.0
        mad = float(np.nanmedian(np.abs(speed - median_speed))) if np.any(np.isfinite(speed)) else 0.0
    else:
        median_speed = float(np.median(reference))
        mad = float(np.median(np.abs(reference - median_speed)))
    reference_std = float(np.nanstd(reference)) if reference.size else 0.0
    if not np.isfinite(median_speed):
        median_speed = 0.0
    if not np.isfinite(mad):
        mad = 0.0
    if not np.isfinite(reference_std):
        reference_std = 0.0
    sigma = 1.4826 * mad if mad > 0 else max(reference_std, 1.0)
    threshold = max(median_speed + (8.0 * sigma), median_speed * 2.5, 15.0)

    outlier = np.zeros(x.size, dtype=bool)
    outlier[1:] = np.isfinite(speed) & (speed > threshold)
    return outlier, {
        "speed_threshold_px_s": float(threshold),
        "speed_median_px_s": float(median_speed),
        "speed_sigma_px_s": float(sigma),
    }


def choose_gyro_axis(
    x: np.ndarray,
    y: np.ndarray,
    frame_times: np.ndarray,
    obs_valid: np.ndarray,
    imu_features: pd.DataFrame,
    min_speed_for_turn_px_s: float = DEFAULT_MIN_SPEED_FOR_TURN_PX_S,
    min_motion_frames_for_axis_fit: int = DEFAULT_MIN_MOTION_FRAMES_FOR_AXIS_FIT,
) -> tuple[str, float, float]:
    """Select the gyro axis that best matches visual turning."""
    if x.size < 3:
        return GYRO_AXIS_NAMES[0], 1.0, 0.0

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(frame_times)
    fallback_dt = _get_positive_dt_fallback(dt, default=1.0)
    dt = np.where(dt <= 0, fallback_dt, dt)
    speed = np.hypot(dx, dy) / dt
    motion_angle = np.unwrap(np.arctan2(dy, dx))
    turn_rate = np.diff(motion_angle) / dt[1:]

    move_mask = speed[:-1] > min_speed_for_turn_px_s
    valid_mask = (
        obs_valid[:-2]
        & obs_valid[1:-1]
        & obs_valid[2:]
        & np.isfinite(turn_rate)
        & move_mask
    )

    best_axis = GYRO_AXIS_NAMES[0]
    best_sign = 1.0
    best_corr = 0.0

    if int(np.sum(valid_mask)) >= min_motion_frames_for_axis_fit:
        for axis in GYRO_AXIS_NAMES:
            values = imu_features[axis].to_numpy(dtype=float)[2:]
            if np.nanstd(values[valid_mask]) == 0:
                continue
            corr = np.corrcoef(values[valid_mask], turn_rate[valid_mask])[0, 1]
            if np.isnan(corr):
                continue
            if abs(float(corr)) > abs(best_corr):
                best_axis = axis
                best_sign = 1.0 if corr >= 0 else -1.0
                best_corr = float(corr)
    else:
        std_by_axis = {
            axis: float(np.nan_to_num(np.nanstd(imu_features[axis].to_numpy(dtype=float)), nan=0.0))
            for axis in GYRO_AXIS_NAMES
        }
        best_axis = max(std_by_axis, key=std_by_axis.get)

    return best_axis, float(best_sign), float(best_corr)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    """Map one signal to the unit interval using robust percentiles."""
    value_array = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(value_array)):
        return np.zeros_like(value_array, dtype=float)
    lo = float(np.nanpercentile(value_array, 10))
    hi = float(np.nanpercentile(value_array, 90))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(value_array, dtype=float)
    normalized = (value_array - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)


def white_accel_q(dt: float, accel_scale: float) -> np.ndarray:
    """Return a white-acceleration process covariance matrix."""
    q = accel_scale**2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    return q * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=float,
    )


def kalman_smooth_head_track(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    likelihood: np.ndarray,
    frame_times: np.ndarray,
    obs_valid: np.ndarray,
    motion_energy: np.ndarray,
    base_measurement_sigma_px: float = DEFAULT_BASE_MEASUREMENT_SIGMA_PX,
    base_process_accel_px_s2: float = DEFAULT_BASE_PROCESS_ACCEL_PX_S2,
    motion_noise_gain: float = DEFAULT_MOTION_NOISE_GAIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a constant-velocity Kalman filter plus RTS smoother."""
    n_frames = x_obs.size
    if n_frames == 0:
        raise ValueError("Head observations are empty.")

    dt = np.diff(frame_times, prepend=frame_times[0])
    nominal_dt = _get_positive_dt_fallback(dt[1:], default=1.0 / 15.0)
    dt[0] = nominal_dt
    dt[dt <= 0] = nominal_dt

    x_filt = np.zeros((n_frames, 4), dtype=float)
    p_filt = np.zeros((n_frames, 4, 4), dtype=float)
    x_pred = np.zeros((n_frames, 4), dtype=float)
    p_pred = np.zeros((n_frames, 4, 4), dtype=float)

    first_valid = int(np.argmax(obs_valid)) if obs_valid.any() else 0
    init_x = float(x_obs[first_valid]) if np.isfinite(x_obs[first_valid]) else float(np.nanmedian(x_obs))
    init_y = float(y_obs[first_valid]) if np.isfinite(y_obs[first_valid]) else float(np.nanmedian(y_obs))
    if not np.isfinite(init_x) or not np.isfinite(init_y):
        raise ValueError("Cannot initialize head fusion because all raw head positions are non-finite.")

    x_state = np.array([init_x, init_y, 0.0, 0.0], dtype=float)
    p_state = np.diag([25.0, 25.0, 2500.0, 2500.0])
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
    identity = np.eye(4)

    for index in range(n_frames):
        if index == 0:
            x_prior = x_state.copy()
            p_prior = p_state.copy()
        else:
            f = np.array(
                [
                    [1.0, 0.0, dt[index], 0.0],
                    [0.0, 1.0, 0.0, dt[index]],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            accel_scale = base_process_accel_px_s2 * (
                0.35 + (motion_noise_gain * float(motion_energy[index]))
            )
            q = white_accel_q(float(dt[index]), float(accel_scale))
            x_prior = f @ x_state
            p_prior = f @ p_state @ f.T + q

        x_pred[index] = x_prior
        p_pred[index] = p_prior

        if obs_valid[index]:
            like = float(np.clip(likelihood[index], 1e-3, 1.0))
            meas_sigma = base_measurement_sigma_px / like
            r = np.diag([meas_sigma**2, meas_sigma**2])
            z = np.array([x_obs[index], y_obs[index]], dtype=float)
            residual = z - (h @ x_prior)
            s = h @ p_prior @ h.T + r
            k = p_prior @ h.T @ np.linalg.inv(s)
            x_state = x_prior + (k @ residual)
            p_state = (identity - (k @ h)) @ p_prior
        else:
            x_state = x_prior
            p_state = p_prior

        x_filt[index] = x_state
        p_filt[index] = p_state

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for index in range(n_frames - 2, -1, -1):
        f = np.array(
            [
                [1.0, 0.0, dt[index + 1], 0.0],
                [0.0, 1.0, 0.0, dt[index + 1]],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        c = p_filt[index] @ f.T @ np.linalg.inv(p_pred[index + 1])
        x_smooth[index] = x_filt[index] + c @ (x_smooth[index + 1] - x_pred[index + 1])
        p_smooth[index] = p_filt[index] + c @ (p_smooth[index + 1] - p_pred[index + 1]) @ c.T

    return x_smooth[:, 0], x_smooth[:, 1], x_smooth[:, 2], x_smooth[:, 3]


def build_long_gap_mask(
    obs_valid: np.ndarray,
    frame_times: np.ndarray,
    max_gap_s: float,
) -> np.ndarray:
    """Mark contiguous invalid runs that span longer than the allowed gap."""
    long_gap = np.zeros(obs_valid.size, dtype=bool)
    invalid = ~obs_valid
    index = 0
    while index < obs_valid.size:
        if not invalid[index]:
            index += 1
            continue
        stop = index
        while stop + 1 < obs_valid.size and invalid[stop + 1]:
            stop += 1
        left_time = frame_times[index - 1] if index > 0 else frame_times[index]
        right_time = frame_times[stop + 1] if stop + 1 < obs_valid.size else frame_times[stop]
        if (right_time - left_time) > max_gap_s:
            long_gap[index : stop + 1] = True
        index = stop + 1
    return long_gap


def run_head_position_fusion(
    dlc_table: pd.DataFrame,
    frame_times: np.ndarray,
    imu_features: pd.DataFrame,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THRESHOLD,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fuse one epoch's raw head DLC track using derived IMU features."""
    if dlc_table.shape[0] != len(frame_times):
        raise ValueError(
            "DLC row count does not match the frame timestamp count: "
            f"{dlc_table.shape[0]} vs {len(frame_times)}."
        )
    missing_feature_columns = [
        column for column in FRAME_IMU_COLUMNS if column not in imu_features.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            "Frame IMU features are missing required columns: "
            f"{missing_feature_columns!r}."
        )
    if imu_features.shape[0] != dlc_table.shape[0]:
        raise ValueError(
            "Frame IMU feature row count does not match DLC row count: "
            f"{imu_features.shape[0]} vs {dlc_table.shape[0]}."
        )

    x = dlc_table["head_x_raw"].to_numpy(dtype=float)
    y = dlc_table["head_y_raw"].to_numpy(dtype=float)
    likelihood = dlc_table["likelihood"].to_numpy(dtype=float)
    frame_numbers = dlc_table["frame"].to_numpy(dtype=int)

    outlier_rejected, outlier_metadata = compute_visual_outlier_mask(
        x=x,
        y=y,
        likelihood=likelihood,
        frame_times=np.asarray(frame_times, dtype=float),
        likelihood_threshold=likelihood_threshold,
    )
    low_confidence = likelihood < likelihood_threshold
    observation_valid = ~(low_confidence | outlier_rejected)

    best_axis, best_sign, best_corr = choose_gyro_axis(
        x=x,
        y=y,
        frame_times=np.asarray(frame_times, dtype=float),
        obs_valid=observation_valid,
        imu_features=imu_features,
    )
    selected_gyro = best_sign * imu_features[best_axis].to_numpy(dtype=float)
    motion_energy = robust_normalize(
        np.abs(selected_gyro) + (0.25 * imu_features["accel_mag_std"].to_numpy(dtype=float))
    )

    head_x_fused, head_y_fused, vel_x_fused, vel_y_fused = kalman_smooth_head_track(
        x_obs=x,
        y_obs=y,
        likelihood=likelihood,
        frame_times=np.asarray(frame_times, dtype=float),
        obs_valid=observation_valid,
        motion_energy=motion_energy,
    )

    long_gap = build_long_gap_mask(
        obs_valid=observation_valid,
        frame_times=np.asarray(frame_times, dtype=float),
        max_gap_s=max_gap_s,
    )
    gap_filled = ~observation_valid & ~long_gap
    head_x_fused = head_x_fused.copy()
    head_y_fused = head_y_fused.copy()
    head_x_fused[long_gap] = np.nan
    head_y_fused[long_gap] = np.nan

    results = pd.DataFrame(
        {
            "frame": frame_numbers,
            "frame_time_s": np.asarray(frame_times, dtype=float),
            "head_x_raw": x,
            "head_y_raw": y,
            "likelihood": likelihood,
            "head_x_fused": head_x_fused,
            "head_y_fused": head_y_fused,
            "head_vx_fused": vel_x_fused,
            "head_vy_fused": vel_y_fused,
            "low_confidence": low_confidence.astype(bool),
            "outlier_rejected": outlier_rejected.astype(bool),
            "observation_valid": observation_valid.astype(bool),
            "long_gap": long_gap.astype(bool),
            "gap_filled": gap_filled.astype(bool),
            "motion_energy": motion_energy,
            "selected_gyro": selected_gyro,
        }
    )
    results = pd.concat([results, imu_features.reset_index(drop=True)], axis=1)
    metadata = {
        "best_gyro_axis": best_axis,
        "best_gyro_sign": float(best_sign),
        "best_gyro_visual_corr": float(best_corr),
        "likelihood_threshold": float(likelihood_threshold),
        "max_gap_s": float(max_gap_s),
        "outlier_metrics": outlier_metadata,
        "low_confidence_frame_count": int(np.sum(low_confidence)),
        "outlier_frame_count": int(np.sum(outlier_rejected)),
        "long_gap_frame_count": int(np.sum(long_gap)),
        "gap_filled_frame_count": int(np.sum(gap_filled)),
    }
    return results, metadata


def get_default_imu_path(analysis_path: Path, epoch: str) -> Path:
    """Return the default extracted IMU parquet path for one epoch."""
    return analysis_path / f"{epoch}_imu.parquet"


def load_base_position_or_default(
    analysis_path: Path,
    epoch_tags: list[str],
) -> tuple[dict[str, np.ndarray], str]:
    """Load session position when available, otherwise start from an empty mapping."""
    position_path = analysis_path / "position.pkl"
    if not position_path.exists():
        return {}, "missing"

    base_position = load_position_data(analysis_path, epoch_tags)
    return {epoch: coerce_position_array(position_xy).copy() for epoch, position_xy in base_position.items()}, "pickle"


def fuse_head_position_imu(
    animal_name: str,
    date: str,
    epoch: str,
    dlc_h5_path: Path,
    imu_path: Path | None = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_name: str = DEFAULT_OUTPUT_NAME,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    likelihood_threshold: float = DEFAULT_LIKELIHOOD_THRESHOLD,
    max_gap_s: float = DEFAULT_MAX_GAP_S,
) -> Path:
    """Fuse one epoch's head track and save a sidecar position pickle.

    When `position.pkl` is present, untouched epochs are copied through. When it is
    absent, the output contains only the requested fused epoch.
    """
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    print(f"Processing {animal_name} {date} epoch {epoch}.")
    epoch_tags, position_timestamps, timestamp_source = load_position_timestamps(analysis_path)
    if epoch not in position_timestamps:
        raise ValueError(
            f"Epoch {epoch!r} not found in saved position timestamps. "
            f"Available epochs: {sorted(position_timestamps)!r}"
        )
    _, ephys_timestamps, ephys_source = load_ephys_timestamps_by_epoch(analysis_path)
    if epoch not in ephys_timestamps:
        raise ValueError(
            f"Epoch {epoch!r} not found in saved ephys timestamps. "
            f"Available epochs: {sorted(ephys_timestamps)!r}"
        )

    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")

    base_position, base_position_source = load_base_position_or_default(analysis_path, epoch_tags)
    frame_times = np.asarray(position_timestamps[epoch], dtype=float)
    epoch_ephys_timestamps = np.asarray(ephys_timestamps[epoch], dtype=float)
    if epoch_ephys_timestamps.size == 0:
        raise ValueError(f"Epoch {epoch!r} has no saved ephys timestamps.")
    if frame_times.size <= position_offset:
        raise ValueError(
            "Position offset removes all timestamp samples for one epoch. "
            f"timestamp count: {frame_times.size}, position_offset: {position_offset}"
        )

    dlc_table = load_head_dlc(dlc_h5_path)
    if dlc_table.shape[0] != frame_times.size:
        raise ValueError(
            "DLC row count does not match saved position timestamps for the requested epoch: "
            f"{dlc_table.shape[0]} vs {frame_times.size}."
        )

    trimmed_frame_times = np.asarray(frame_times[position_offset:], dtype=float)
    trimmed_dlc_table = dlc_table.iloc[position_offset:].reset_index(drop=True)

    imu_path = get_default_imu_path(analysis_path, epoch) if imu_path is None else imu_path
    imu_table = load_imu_table(imu_path)
    imu_features = build_frame_imu_features(
        imu_table=imu_table,
        frame_times=trimmed_frame_times,
        epoch_start_time=float(trimmed_frame_times[0]),
    )
    fused_results, fusion_metadata = run_head_position_fusion(
        dlc_table=trimmed_dlc_table,
        frame_times=trimmed_frame_times,
        imu_features=imu_features,
        likelihood_threshold=likelihood_threshold,
        max_gap_s=max_gap_s,
    )

    if base_position_source == "pickle":
        target_epoch_position = np.asarray(base_position[epoch], dtype=float).copy()
    else:
        target_epoch_position = dlc_table[["head_x_raw", "head_y_raw"]].to_numpy(dtype=float)
    if target_epoch_position.shape[0] != frame_times.size:
        raise ValueError(
            "Target epoch position does not match saved timestamp count: "
            f"{target_epoch_position.shape[0]} vs {frame_times.size}."
        )

    target_epoch_position[position_offset:] = fused_results[["head_x_fused", "head_y_fused"]].to_numpy(dtype=float)

    head_position_imu = {
        epoch_name: np.asarray(position_xy, dtype=float).copy()
        for epoch_name, position_xy in base_position.items()
    }
    head_position_imu[epoch] = target_epoch_position

    output_path = analysis_path / output_name
    saved_path = save_pickle_output(output_path, head_position_imu)
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.position.fuse_head_position_imu",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "epoch": epoch,
            "dlc_h5_path": dlc_h5_path,
            "imu_path": imu_path,
            "data_root": data_root,
            "output_name": output_name,
            "position_offset": position_offset,
            "likelihood_threshold": likelihood_threshold,
            "max_gap_s": max_gap_s,
        },
        outputs={
            "position_timestamp_source": timestamp_source,
            "ephys_timestamp_source": ephys_source,
            "base_position_source": base_position_source,
            "base_position_epoch_count": int(len(base_position)),
            "frame_count": int(frame_times.size),
            "trimmed_frame_count": int(trimmed_frame_times.size),
            "imu_row_count": int(imu_table.shape[0]),
            "fused_output_path": saved_path,
            "fusion_metadata": fusion_metadata,
        },
    )
    print(f"Saved fused head position pickle to {saved_path}")
    print(f"Saved run metadata to {log_path}")
    return saved_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the head fusion CLI."""
    parser = argparse.ArgumentParser(
        description="Fuse one epoch of DLC head tracking with extracted IMU events."
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
        help="Epoch label to replace in the fused sidecar output.",
    )
    parser.add_argument(
        "--dlc-h5-path",
        type=Path,
        required=True,
        help="Path to the DeepLabCut H5 file containing the head track.",
    )
    parser.add_argument(
        "--imu-path",
        type=Path,
        default=None,
        help="Optional path to the extracted IMU parquet. Default: analysis_path / '{epoch}_imu.parquet'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output pickle filename written under the session analysis directory. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore per epoch during fusion. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--likelihood-threshold",
        type=float,
        default=DEFAULT_LIKELIHOOD_THRESHOLD,
        help=f"Minimum DLC likelihood kept as a valid observation. Default: {DEFAULT_LIKELIHOOD_THRESHOLD}",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=DEFAULT_MAX_GAP_S,
        help=f"Maximum invalid gap duration to interpolate through. Default: {DEFAULT_MAX_GAP_S}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the DLC + IMU head fusion CLI."""
    args = parse_arguments()
    fuse_head_position_imu(
        animal_name=args.animal_name,
        date=args.date,
        epoch=args.epoch,
        dlc_h5_path=args.dlc_h5_path,
        imu_path=args.imu_path,
        data_root=args.data_root,
        output_name=args.output_name,
        position_offset=args.position_offset,
        likelihood_threshold=args.likelihood_threshold,
        max_gap_s=args.max_gap_s,
    )


if __name__ == "__main__":
    main()
