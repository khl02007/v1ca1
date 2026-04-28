from __future__ import annotations

"""Fit motor, task-progression, and place GLMs for one task-progression session.

This script uses the shared session loaders under
`v1ca1.task_progression._session`, exposes a CLI for session and fit settings,
and writes separate nested-CV evidence, full-data refit, and primary-delta
summary figure outputs under the analysis directory. Position inputs default to
the cleaned DLC `dlc_position_cleaned/position.parquet` export when present,
using head coordinates for task progression and body coordinates for
head-direction motor covariates. Trajectory intervals are loaded from
`trajectory_times.parquet`.

For each selected region and run epoch, the script compares five Poisson GLMs:

- strict motor only
- motor + TP group offset + trajectory-grouped task progression
- TP group offset + trajectory-grouped task progression
- motor + trajectory offset + trajectory-specific place
- trajectory offset + trajectory-specific place

Motor covariates can be represented either as instantaneous z-scored values or
as spline-expanded features, including head angular acceleration on the binned
time axis. Task progression uses one tuning curve per same-turn trajectory pair,
whereas place uses one tuning curve per trajectory type. The primary scientific
scores are pooled unit-level held-out lap-CV deltas for motor+TP vs motor and
motor+place vs motor. Ridge is selected separately per model by inner lap-CV on
the outer-train laps, and the full-data refit is saved for visualization rather
than final inference.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from v1ca1.helper.cuda import (
    configure_cuda_visible_devices,
    pop_cuda_visible_devices_argument,
)

_CUDA_VISIBLE_DEVICES_CLI = pop_cuda_visible_devices_argument()
configure_cuda_visible_devices(_CUDA_VISIBLE_DEVICES_CLI)

import jax.numpy as jnp
import numpy as np
import position_tools as pt
from nemos.basis import BSplineEval
from nemos.glm import PopulationGLM
from scipy.special import gammaln

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    get_run_epochs,
    load_clean_dlc_position_data,
    load_epoch_tags,
)
from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    compute_movement_firing_rates,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    import xarray as xr


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_RIDGES = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
DEFAULT_INNER_N_FOLDS = 3
HYPERPARAMETER_TIE_ATOL = 1e-6
TP_GROUPS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("LC_CR", ("left_to_center", "center_to_right")),
    ("RC_CL", ("right_to_center", "center_to_left")),
)
MODEL_NAMES = (
    "motor",
    "motor_tp",
    "tp_only",
    "motor_place",
    "place_only",
)
MODEL_DEFINITIONS = {
    "motor": "strict motor covariates only",
    "motor_tp": "motor covariates plus TP group offset and TP group-specific spline fields",
    "tp_only": "TP group offset plus TP group-specific spline fields",
    "motor_place": "motor covariates plus trajectory offset and trajectory-specific place fields",
    "place_only": "trajectory offset plus trajectory-specific place fields",
}
PRIMARY_DELTA_METRIC_NAMES = (
    "dll_motor_tp_vs_motor_bits_per_spike",
    "dll_motor_place_vs_motor_bits_per_spike",
)
DELTA_SPECS = (
    ("dll_motor_tp_vs_motor_bits_per_spike", "motor_tp", "motor"),
    ("dll_motor_place_vs_motor_bits_per_spike", "motor_place", "motor"),
    ("dll_tp_only_vs_motor_bits_per_spike", "tp_only", "motor"),
    ("dll_place_only_vs_motor_bits_per_spike", "place_only", "motor"),
    ("dll_motor_tp_vs_tp_only_bits_per_spike", "motor_tp", "tp_only"),
    ("dll_motor_place_vs_place_only_bits_per_spike", "motor_place", "place_only"),
    ("dll_motor_place_vs_motor_tp_bits_per_spike", "motor_place", "motor_tp"),
    ("dll_place_only_vs_tp_only_bits_per_spike", "place_only", "tp_only"),
)
CV_METRIC_NAMES = (
    "spike_sum",
    "ll_motor_bits_per_spike",
    "ll_motor_tp_bits_per_spike",
    "ll_tp_only_bits_per_spike",
    "ll_motor_place_bits_per_spike",
    "ll_place_only_bits_per_spike",
    "dll_motor_tp_vs_motor_bits_per_spike",
    "dll_tp_only_vs_motor_bits_per_spike",
    "dll_motor_tp_vs_tp_only_bits_per_spike",
    "dll_motor_place_vs_motor_bits_per_spike",
    "dll_place_only_vs_motor_bits_per_spike",
    "dll_motor_place_vs_place_only_bits_per_spike",
    "dll_motor_place_vs_motor_tp_bits_per_spike",
    "dll_place_only_vs_tp_only_bits_per_spike",
)
MOTOR_CONTINUOUS_FEATURE_NAMES = (
    "speed",
    "accel",
    "hd_vel",
    "hd_acc",
    "abs_hd_vel",
)
MOTOR_RAW_FEATURE_NAMES = (*MOTOR_CONTINUOUS_FEATURE_NAMES, "sin_hd", "cos_hd")


def select_run_epochs(
    run_epochs: list[str], requested_epochs: list[str] | None
) -> list[str]:
    """Return the requested run epochs, defaulting to all available run epochs."""
    if not requested_epochs:
        return list(run_epochs)

    selected_epochs = list(dict.fromkeys(requested_epochs))

    missing_epochs = [epoch for epoch in selected_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in available run epochs {run_epochs!r}: {missing_epochs!r}"
        )
    return selected_epochs


def has_any_finite_position(position_xy: np.ndarray | None) -> bool:
    """Return whether one XY position array contains at least one finite sample."""
    if position_xy is None:
        return False
    position_array = np.asarray(position_xy, dtype=float)
    return position_array.size > 0 and np.isfinite(position_array).any()


def select_epochs_with_usable_position_data(
    analysis_path: Path,
    requested_epochs: list[str] | None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return run epochs with usable cleaned-DLC head/body position plus skip reasons."""
    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    requested_run_epochs = select_run_epochs(get_run_epochs(epoch_tags), requested_epochs)
    clean_dlc_path = (
        analysis_path
        / DEFAULT_CLEAN_DLC_POSITION_DIRNAME
        / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    epoch_order, position_by_epoch, body_position_by_epoch = load_clean_dlc_position_data(
        analysis_path,
        input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    position_source = str(clean_dlc_path)
    body_position_source = str(clean_dlc_path)
    available_clean_dlc_epochs = {str(epoch) for epoch in epoch_order}

    usable_epochs: list[str] = []
    skipped_epochs: list[dict[str, str]] = []
    for epoch in requested_run_epochs:
        reasons: list[str] = []
        head_position = position_by_epoch.get(epoch) if epoch in available_clean_dlc_epochs else None
        body_position = (
            body_position_by_epoch.get(epoch) if epoch in available_clean_dlc_epochs else None
        )
        if head_position is None:
            reasons.append(f"head position missing from {position_source}")
        elif not has_any_finite_position(head_position):
            reasons.append(f"head position is all NaN in {position_source}")
        if body_position is None:
            reasons.append(f"body position missing from {body_position_source}")
        elif not has_any_finite_position(body_position):
            reasons.append(f"body position is all NaN in {body_position_source}")

        if reasons:
            skipped_epochs.append({"epoch": epoch, "reason": "; ".join(reasons)})
        else:
            usable_epochs.append(epoch)

    return usable_epochs, skipped_epochs


def build_position_tsdframe(
    position_xy: np.ndarray,
    timestamps_position: np.ndarray,
    position_offset: int,
) -> Any:
    """Build one trimmed XY `nap.TsdFrame` for a run epoch."""
    import pynapple as nap

    if position_xy.shape[0] != timestamps_position.size:
        raise ValueError(
            "Position samples and position timestamps must have the same length. "
            f"Got {position_xy.shape[0]} and {timestamps_position.size}."
        )
    if position_xy.shape[0] <= position_offset:
        raise ValueError(
            "Position offset removes all samples for one epoch. "
            f"position count: {position_xy.shape[0]}, position_offset: {position_offset}"
        )
    return nap.TsdFrame(
        t=np.asarray(timestamps_position[position_offset:], dtype=float),
        d=np.asarray(position_xy[position_offset:], dtype=float),
        columns=["x", "y"],
        time_units="s",
    )


def bspline_features(
    values: np.ndarray,
    *,
    n_basis: int,
    order: int = 4,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return finite B-spline features for one continuous covariate."""
    values = np.asarray(values, dtype=float).reshape(-1)

    if bounds is None:
        lower = float(np.nanpercentile(values, 1.0))
        upper = float(np.nanpercentile(values, 99.0))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower = float(np.nanmin(values))
            upper = float(np.nanmax(values))
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            lower, upper = 0.0, 1.0
    else:
        lower, upper = map(float, bounds)

    clipped_values = np.clip(values, lower, upper)
    basis = BSplineEval(
        n_basis_funcs=int(n_basis),
        order=int(order),
        bounds=(lower, upper),
    )
    features = np.asarray(basis.compute_features(clipped_values), dtype=float)
    if not np.all(np.isfinite(features)):
        bad_row, bad_col = np.argwhere(~np.isfinite(features))[0]
        raise ValueError(
            "Encountered non-finite spline features. "
            f"Example at row {int(bad_row)}, column {int(bad_col)}."
        )
    return features


def stratified_contiguous_folds(
    labels: np.ndarray,
    n_folds: int,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each label class into contiguous chunks and combine one chunk per fold."""
    labels = np.asarray(labels).reshape(-1)
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    all_indices = np.arange(labels.size, dtype=int)
    rng = np.random.default_rng(seed)
    fold_test_chunks: list[list[np.ndarray]] = [[] for _ in range(n_folds)]

    for class_label in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_label)
        if class_indices.size == 0:
            continue
        class_chunks = np.array(np.array_split(class_indices, n_folds), dtype=object)
        class_chunks = class_chunks[rng.permutation(n_folds)]
        for fold_index in range(n_folds):
            if len(class_chunks[fold_index]) > 0:
                fold_test_chunks[fold_index].append(
                    np.asarray(class_chunks[fold_index], dtype=int)
                )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(n_folds):
        if fold_test_chunks[fold_index]:
            test_indices = np.sort(np.concatenate(fold_test_chunks[fold_index]))
        else:
            test_indices = np.array([], dtype=int)
        is_test = np.zeros(labels.size, dtype=bool)
        is_test[test_indices] = True
        train_indices = all_indices[~is_test]
        folds.append((train_indices, test_indices))
    return folds


def compute_motor_covariates(
    position_xy: np.ndarray,
    body_xy: np.ndarray,
    position_timestamps: np.ndarray,
    spike_counts: Any,
) -> dict[str, np.ndarray]:
    """Compute binned motor covariates, including head angular acceleration."""
    import pynapple as nap

    sampling_rate = (len(position_timestamps) - 1) / (
        position_timestamps[-1] - position_timestamps[0]
    )
    speed = np.asarray(
        pt.get_speed(
            position=position_xy,
            time=position_timestamps,
            sampling_frequency=float(sampling_rate),
            sigma=DEFAULT_SPEED_SIGMA_S,
        ),
        dtype=float,
    )
    speed_bin = nap.Tsd(t=position_timestamps, d=speed, time_units="s").interpolate(
        spike_counts
    )
    speed_bin = np.asarray(speed_bin.to_numpy(), dtype=float).reshape(-1)

    dt = float(np.median(np.diff(np.asarray(spike_counts.t, dtype=float))))
    acceleration = np.gradient(speed_bin, dt)

    head_vector = np.asarray(position_xy, dtype=float) - np.asarray(
        body_xy, dtype=float
    )
    head_direction = np.arctan2(head_vector[:, 1], head_vector[:, 0])
    head_direction_bin = nap.Tsd(
        t=position_timestamps,
        d=head_direction,
        time_units="s",
    ).interpolate(spike_counts)
    head_direction_bin = np.asarray(head_direction_bin.to_numpy(), dtype=float).reshape(
        -1
    )
    head_direction_unwrapped = np.unwrap(head_direction)
    head_direction_unwrapped_bin = nap.Tsd(
        t=position_timestamps,
        d=head_direction_unwrapped,
        time_units="s",
    ).interpolate(spike_counts)
    head_direction_unwrapped_bin = np.asarray(
        head_direction_unwrapped_bin.to_numpy(),
        dtype=float,
    ).reshape(-1)
    head_direction_velocity = np.gradient(head_direction_unwrapped_bin, dt)
    head_direction_acceleration = np.gradient(head_direction_velocity, dt)
    sin_head_direction = np.sin(head_direction_bin)
    cos_head_direction = np.cos(head_direction_bin)

    return {
        "speed": speed_bin,
        "accel": acceleration,
        "hd_vel": head_direction_velocity,
        "hd_acc": head_direction_acceleration,
        "abs_hd_vel": np.abs(head_direction_velocity),
        "sin_hd": sin_head_direction,
        "cos_hd": cos_head_direction,
    }


def get_unit_mask(
    movement_firing_rates: np.ndarray,
    threshold_hz: float,
) -> np.ndarray:
    """Return the boolean mask of units above the requested movement-rate threshold."""
    movement_firing_rates = np.asarray(movement_firing_rates, dtype=float)
    return np.isfinite(movement_firing_rates) & (
        movement_firing_rates > float(threshold_hz)
    )


def build_cv_metric_dict(
    ll_per_model: dict[str, np.ndarray],
    spike_sum: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build one CV metric dict from per-model log-likelihoods per spike."""
    inverse_log2 = 1.0 / np.log(2.0)
    ll_bits_per_model = {
        model_name: np.asarray(values, dtype=float) * inverse_log2
        for model_name, values in ll_per_model.items()
    }
    return {
        "spike_sum": np.asarray(spike_sum, dtype=float),
        "ll_motor_bits_per_spike": ll_bits_per_model["motor"],
        "ll_motor_tp_bits_per_spike": ll_bits_per_model["motor_tp"],
        "ll_tp_only_bits_per_spike": ll_bits_per_model["tp_only"],
        "ll_motor_place_bits_per_spike": ll_bits_per_model["motor_place"],
        "ll_place_only_bits_per_spike": ll_bits_per_model["place_only"],
        "dll_motor_tp_vs_motor_bits_per_spike": (
            ll_bits_per_model["motor_tp"] - ll_bits_per_model["motor"]
        ),
        "dll_tp_only_vs_motor_bits_per_spike": (
            ll_bits_per_model["tp_only"] - ll_bits_per_model["motor"]
        ),
        "dll_motor_tp_vs_tp_only_bits_per_spike": (
            ll_bits_per_model["motor_tp"] - ll_bits_per_model["tp_only"]
        ),
        "dll_motor_place_vs_motor_bits_per_spike": (
            ll_bits_per_model["motor_place"] - ll_bits_per_model["motor"]
        ),
        "dll_place_only_vs_motor_bits_per_spike": (
            ll_bits_per_model["place_only"] - ll_bits_per_model["motor"]
        ),
        "dll_motor_place_vs_place_only_bits_per_spike": (
            ll_bits_per_model["motor_place"] - ll_bits_per_model["place_only"]
        ),
        "dll_motor_place_vs_motor_tp_bits_per_spike": (
            ll_bits_per_model["motor_place"] - ll_bits_per_model["motor_tp"]
        ),
        "dll_place_only_vs_tp_only_bits_per_spike": (
            ll_bits_per_model["place_only"] - ll_bits_per_model["tp_only"]
        ),
    }


def stack_cv_metrics(cv_metrics: dict[str, np.ndarray]) -> np.ndarray:
    """Stack one CV metric dict into a dense `(unit, cv_metric)` array."""
    return np.column_stack(
        [
            np.asarray(cv_metrics[metric_name], dtype=float)
            for metric_name in CV_METRIC_NAMES
        ]
    )


def stack_cv_metrics_by_trajectory(
    cv_by_trajectory: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Stack trajectory-specific CV metrics into `(trajectory, unit, cv_metric)`."""
    return np.stack(
        [
            stack_cv_metrics(cv_by_trajectory[trajectory_type])
            for trajectory_type in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def stack_tp_coefficients_by_group(
    coefficient_dict: dict[str, np.ndarray],
) -> np.ndarray:
    """Stack TP coefficients into `(tp_group, tp_basis, unit)`."""
    group_names = [group_name for group_name, _ in TP_GROUPS]
    return np.stack(
        [
            np.asarray(coefficient_dict[group_name], dtype=float)
            for group_name in group_names
        ],
        axis=0,
    )


def stack_tp_rate_curves(
    tp_rate_curves_hz: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Stack model-implied TP rate curves into `(trajectory, tp_grid, unit)`."""
    return np.stack(
        [
            np.asarray(tp_rate_curves_hz[trajectory_type]["rate_hz"], dtype=float)
            for trajectory_type in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def stack_place_coefficients_by_trajectory(
    coefficient_dict: dict[str, np.ndarray],
) -> np.ndarray:
    """Stack place coefficients into `(trajectory, place_basis, unit)`."""
    return np.stack(
        [
            np.asarray(coefficient_dict[trajectory_type], dtype=float)
            for trajectory_type in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def stack_place_rate_curves(
    place_rate_curves_hz: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Stack model-implied place rate curves into `(trajectory, place_grid, unit)`."""
    return np.stack(
        [
            np.asarray(place_rate_curves_hz[trajectory_type]["rate_hz"], dtype=float)
            for trajectory_type in TRAJECTORY_TYPES
        ],
        axis=0,
    )


def build_tp_group_membership() -> np.ndarray:
    """Return the `(tp_group, trajectory)` membership matrix."""
    membership = np.zeros((len(TP_GROUPS), len(TRAJECTORY_TYPES)), dtype=np.int8)
    trajectory_to_index = {
        trajectory_type: index for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }
    for group_index, (_, grouped_trajectories) in enumerate(TP_GROUPS):
        for trajectory_type in grouped_trajectories:
            membership[group_index, trajectory_to_index[trajectory_type]] = 1
    return membership


def build_histogram_bin_edges(values: np.ndarray) -> np.ndarray:
    """Return histogram bin edges with `0.0` included as an edge."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Histogram bin edges require at least one value.")

    if np.allclose(values, values[0]):
        half_width = max(0.1, abs(float(values[0])) * 0.1 + 0.1)
        bin_edges = np.linspace(
            float(values[0]) - half_width,
            float(values[0]) + half_width,
            16,
        )
    else:
        bin_edges = np.histogram_bin_edges(values, bins="auto")

    if not np.any(np.isclose(bin_edges, 0.0)):
        bin_edges = np.sort(np.unique(np.concatenate([bin_edges, np.array([0.0])])))
    return np.asarray(bin_edges, dtype=float)


def build_fit_dataset(
    fit_result: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    movement_firing_rates: np.ndarray,
    min_firing_rate_hz: float,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Convert one fit result into a labeled `xarray.Dataset`.

    The saved dataset includes two model families over the same normalized
    within-trajectory 1D coordinate:
    - task progression: 2 grouped trajectory-specific curves
    - place: 4 trajectory-specific curves
    """
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task progression motor fits as NetCDF. "
            "Install xarray in this environment and rerun the script."
        ) from exc

    unit_ids = np.asarray(fit_result["unit_ids"])
    trajectory_names = np.asarray(TRAJECTORY_TYPES, dtype=str)
    tp_group_names = np.asarray([group_name for group_name, _ in TP_GROUPS], dtype=str)
    cv_metric_names = np.asarray(CV_METRIC_NAMES, dtype=str)
    fold_test_counts = np.asarray(fit_result["cv"]["fold_test_counts"], dtype=int)
    tp_grid = np.asarray(
        fit_result["tp_rate_curves_hz"][TRAJECTORY_TYPES[0]]["tp_grid"],
        dtype=float,
    )
    place_grid = np.asarray(
        fit_result["place_rate_curves_hz"][TRAJECTORY_TYPES[0]]["place_grid"],
        dtype=float,
    )
    tp_group_membership = build_tp_group_membership()

    dataset = xr.Dataset(
        data_vars={
            "movement_firing_rate_hz": (
                "unit",
                np.asarray(movement_firing_rates, dtype=float),
            ),
            "cv_pooled": (
                ("unit", "cv_metric"),
                stack_cv_metrics(fit_result["cv"]["pooled"]),
            ),
            "cv_by_trajectory": (
                ("trajectory", "unit", "cv_metric"),
                stack_cv_metrics_by_trajectory(fit_result["cv"]["by_traj"]),
            ),
            "fold_test_counts": (
                ("fold", "trajectory"),
                fold_test_counts,
            ),
            "tp_group_membership": (
                ("tp_group", "trajectory"),
                tp_group_membership,
            ),
            "motor_intercept": (
                "unit",
                np.asarray(fit_result["coef"]["motor"]["intercept"], dtype=float),
            ),
            "motor_coef_motor": (
                ("motor_feature", "unit"),
                np.asarray(fit_result["coef"]["motor"]["coef_motor"], dtype=float),
            ),
            "motor_coef_traj": (
                ("traj_feature", "unit"),
                np.asarray(fit_result["coef"]["motor"]["coef_traj"], dtype=float),
            ),
            "motor_tp_intercept": (
                "unit",
                np.asarray(fit_result["coef"]["motor_tp"]["intercept"], dtype=float),
            ),
            "motor_tp_coef_motor": (
                ("motor_feature", "unit"),
                np.asarray(fit_result["coef"]["motor_tp"]["coef_motor"], dtype=float),
            ),
            "motor_tp_coef_traj": (
                ("traj_feature", "unit"),
                np.asarray(fit_result["coef"]["motor_tp"]["coef_traj"], dtype=float),
            ),
            "motor_tp_coef_tp_by_group": (
                ("tp_group", "tp_basis", "unit"),
                stack_tp_coefficients_by_group(
                    fit_result["coef"]["motor_tp"]["coef_tp_by_group"]
                ),
            ),
            "tp_only_intercept": (
                "unit",
                np.asarray(fit_result["coef"]["tp_only"]["intercept"], dtype=float),
            ),
            "tp_only_coef_traj": (
                ("traj_feature", "unit"),
                np.asarray(fit_result["coef"]["tp_only"]["coef_traj"], dtype=float),
            ),
            "tp_only_coef_tp_by_group": (
                ("tp_group", "tp_basis", "unit"),
                stack_tp_coefficients_by_group(
                    fit_result["coef"]["tp_only"]["coef_tp_by_group"]
                ),
            ),
            "motor_place_intercept": (
                "unit",
                np.asarray(fit_result["coef"]["motor_place"]["intercept"], dtype=float),
            ),
            "motor_place_coef_motor": (
                ("motor_feature", "unit"),
                np.asarray(fit_result["coef"]["motor_place"]["coef_motor"], dtype=float),
            ),
            "motor_place_coef_place_by_trajectory": (
                ("trajectory", "place_basis", "unit"),
                stack_place_coefficients_by_trajectory(
                    fit_result["coef"]["motor_place"]["coef_place_by_trajectory"]
                ),
            ),
            "place_only_intercept": (
                "unit",
                np.asarray(fit_result["coef"]["place_only"]["intercept"], dtype=float),
            ),
            "place_only_coef_place_by_trajectory": (
                ("trajectory", "place_basis", "unit"),
                stack_place_coefficients_by_trajectory(
                    fit_result["coef"]["place_only"]["coef_place_by_trajectory"]
                ),
            ),
            "tp_rate_curves_hz": (
                ("trajectory", "tp_grid", "unit"),
                stack_tp_rate_curves(fit_result["tp_rate_curves_hz"]),
            ),
            "place_rate_curves_hz": (
                ("trajectory", "place_grid", "unit"),
                stack_place_rate_curves(fit_result["place_rate_curves_hz"]),
            ),
        },
        coords={
            "unit": unit_ids,
            "trajectory": trajectory_names,
            "tp_group": tp_group_names,
            "cv_metric": cv_metric_names,
            "fold": np.arange(fold_test_counts.shape[0], dtype=int),
            "motor_feature": np.asarray(fit_result["feature_names_motor"], dtype=str),
            "traj_feature": np.asarray(fit_result["feature_names_traj"], dtype=str),
            "tp_basis": np.arange(
                int(fit_result["tp_basis"]["n_splines"]),
                dtype=int,
            ),
            "tp_grid": tp_grid,
            "place_basis": np.arange(
                int(fit_result["place_basis"]["n_splines"]),
                dtype=int,
            ),
            "place_grid": place_grid,
        },
        attrs={
            "schema_version": "1",
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "bin_size_s": float(fit_result["bin_size_s"]),
            "motor_feature_mode": str(fit_result["motor_feature_mode"]),
            "min_firing_rate_hz": float(min_firing_rate_hz),
            "tp_basis_n_splines": int(fit_result["tp_basis"]["n_splines"]),
            "tp_basis_order": int(fit_result["tp_basis"]["order"]),
            "tp_bounds_lower": float(fit_result["tp_basis"]["bounds"][0]),
            "tp_bounds_upper": float(fit_result["tp_basis"]["bounds"][1]),
            "place_basis_n_splines": int(fit_result["place_basis"]["n_splines"]),
            "place_basis_order": int(fit_result["place_basis"]["order"]),
            "place_bounds_lower": float(fit_result["place_basis"]["bounds"][0]),
            "place_bounds_upper": float(fit_result["place_basis"]["bounds"][1]),
            "coordinate_note": (
                "Task progression uses 2 grouped trajectory-specific curves and "
                "place uses 4 trajectory-specific curves, both built on the same "
                "normalized within-trajectory 1D coordinate."
            ),
            "tp_groups_json": json.dumps(
                {
                    group_name: list(grouped_trajectories)
                    for group_name, grouped_trajectories in TP_GROUPS
                }
            ),
            "feature_names_tp_json": json.dumps(list(fit_result["feature_names_tp"])),
            "feature_names_place_json": json.dumps(
                list(fit_result["feature_names_place"])
            ),
            "sources_json": json.dumps(sources, sort_keys=True),
            "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        },
    )

    motor_standardization = fit_result["motor_standardization"]
    if motor_standardization is not None:
        raw_feature_names = np.asarray(
            motor_standardization["raw_feature_names"],
            dtype=str,
        )
        dataset = dataset.assign_coords({"motor_raw_feature": raw_feature_names})
        dataset["motor_standardization_mean"] = (
            "motor_raw_feature",
            np.asarray(motor_standardization["mean"], dtype=float),
        )
        dataset["motor_standardization_std"] = (
            "motor_raw_feature",
            np.asarray(motor_standardization["std"], dtype=float),
        )
        dataset["motor_standardization_std_raw"] = (
            "motor_raw_feature",
            np.asarray(motor_standardization["std_raw"], dtype=float),
        )
        dataset["motor_standardization_constant_mask"] = (
            "motor_raw_feature",
            np.asarray(motor_standardization["constant_mask"], dtype=np.int8),
        )
        dataset.attrs["motor_zscore_eps"] = float(motor_standardization["eps"])

    return dataset


def plot_log_likelihood_difference_histograms(
    fit_dataset: "xr.Dataset",
    *,
    out_path: Path,
) -> Path:
    """Save one 2x2 histogram summary of pooled CV log-likelihood differences."""
    import matplotlib.pyplot as plt

    panel_specs = (
        (
            "Minus Motor",
            (
                (
                    "Task Progression",
                    "dll_motor_tp_vs_motor_bits_per_spike",
                    "Motor + TP minus Motor",
                    "#4C72B0",
                ),
                (
                    "Place",
                    "dll_motor_place_vs_motor_bits_per_spike",
                    "Motor + Place minus Motor",
                    "#55A868",
                ),
            ),
        ),
        (
            "Minus TP / Place-only",
            (
                (
                    "Task Progression",
                    "dll_motor_tp_vs_tp_only_bits_per_spike",
                    "Motor + TP minus TP-only",
                    "#DD8452",
                ),
                (
                    "Place",
                    "dll_motor_place_vs_place_only_bits_per_spike",
                    "Motor + Place minus Place-only",
                    "#C44E52",
                ),
            ),
        ),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8.4),
        constrained_layout=True,
        sharey=True,
    )

    for row_index, (row_label, distribution_specs) in enumerate(panel_specs):
        for axis, (panel_title, metric_name, label, color) in zip(
            axes[row_index], distribution_specs
        ):
            metric_values = np.asarray(
                fit_dataset["cv_pooled"].sel(cv_metric=metric_name).values,
                dtype=float,
            ).reshape(-1)
            finite_values = metric_values[np.isfinite(metric_values)]

            axis.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)
            axis.set_title(panel_title)
            axis.set_xlabel("Delta log-likelihood (bits/spike)")
            axis.set_ylabel(f"{row_label}\nFraction of units")

            if finite_values.size == 0:
                axis.text(
                    0.5,
                    0.5,
                    "No finite values",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                continue

            bin_edges = build_histogram_bin_edges(finite_values)
            axis.hist(
                finite_values,
                bins=bin_edges,
                weights=np.full(finite_values.shape, 1.0 / finite_values.size),
                color=color,
                alpha=0.55,
                edgecolor="none",
                linewidth=0.0,
                label=label,
            )
            axis.legend(loc="upper left", frameon=False)
            axis.text(
                0.98,
                0.98,
                (
                    f"{label}: n={finite_values.size}, "
                    f"frac>0={np.mean(finite_values > 0.0):.3f}, "
                    f"mean={np.mean(finite_values):.3f}, "
                    f"median={np.median(finite_values):.3f}"
                ),
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
            )

    fig.suptitle(
        (
            f"{fit_dataset.attrs['animal_name']} {fit_dataset.attrs['date']} "
            f"{fit_dataset.attrs['region']} {fit_dataset.attrs['epoch']}"
        ),
        fontsize=12,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _format_float_token(value: float) -> str:
    """Return a path-safe compact token for one float value."""
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _format_ridge_grid_token(ridge_values: Sequence[float]) -> str:
    """Return a compact path token for one ridge grid."""
    values = [float(value) for value in ridge_values]
    if not values:
        raise ValueError("At least one ridge value is required.")
    if len(values) == 1:
        return f"ridge{_format_float_token(values[0])}"
    return (
        f"ridge{_format_float_token(values[0])}-"
        f"{_format_float_token(values[-1])}n{len(values)}"
    )


def build_config_token(
    *,
    bin_size_s: float,
    tp_spline_k: int,
    motor_feature_mode: str,
    n_folds: int,
    inner_n_folds: int,
    ridge_values: Sequence[float],
) -> str:
    """Return the filename token for one nested-lap-CV configuration."""
    return (
        f"bin{_format_float_token(bin_size_s)}s_"
        f"tp{int(tp_spline_k)}_"
        f"{motor_feature_mode}_"
        f"outer{int(n_folds)}_"
        f"inner{int(inner_n_folds)}_"
        f"{_format_ridge_grid_token(ridge_values)}"
    )


def ensure_can_write(path: Path, *, overwrite: bool) -> None:
    """Raise if `path` exists and overwriting was not requested."""
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output {path}. "
            "Pass --overwrite to replace it."
        )


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start/end arrays from an IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).reshape(-1)
    ends = np.asarray(intervals.end, dtype=float).reshape(-1)
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


def summarize_lap_cv_feasibility(
    trajectory_intervals: dict[str, Any],
    *,
    n_folds: int,
    inner_n_folds: int,
) -> tuple[bool, list[str], dict[str, dict[str, int]]]:
    """Return whether an epoch has enough laps for nested lap-level CV."""
    summaries: dict[str, dict[str, int]] = {}
    reasons: list[str] = []
    for trajectory_type in TRAJECTORY_TYPES:
        starts, _ends = _extract_interval_bounds(trajectory_intervals[trajectory_type])
        n_laps = int(starts.size)
        largest_outer_test = int(np.ceil(n_laps / int(n_folds))) if n_folds > 0 else n_laps
        min_outer_train = int(n_laps - largest_outer_test)
        summaries[trajectory_type] = {
            "n_laps": n_laps,
            "min_outer_train_laps": min_outer_train,
        }
        if n_laps < int(n_folds):
            reasons.append(
                f"{trajectory_type} has {n_laps} lap(s), fewer than outer n_folds={n_folds}"
            )
        if min_outer_train < int(inner_n_folds):
            reasons.append(
                f"{trajectory_type} has minimum outer-train lap count {min_outer_train}, "
                f"fewer than inner_n_folds={inner_n_folds}"
            )
    return len(reasons) == 0, reasons, summaries


def build_lap_cv_folds_for_epoch(
    trajectory_intervals: dict[str, Any],
    *,
    n_folds: int,
    seed: int,
    candidate_indices_by_trajectory: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Split each trajectory's laps into aligned train/test folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    rng = np.random.default_rng(int(seed))
    chunks_by_trajectory: dict[str, list[np.ndarray]] = {}
    candidates_by_trajectory: dict[str, np.ndarray] = {}
    bounds_by_trajectory: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        starts, ends = _extract_interval_bounds(trajectory_intervals[trajectory_type])
        bounds_by_trajectory[trajectory_type] = (starts, ends)
        if candidate_indices_by_trajectory is None:
            candidate_indices = np.arange(starts.size, dtype=int)
        else:
            candidate_indices = np.asarray(
                candidate_indices_by_trajectory[trajectory_type],
                dtype=int,
            ).reshape(-1)
        candidate_indices = np.sort(candidate_indices)
        if candidate_indices.size < n_folds:
            raise ValueError(
                f"Trajectory {trajectory_type!r} has {candidate_indices.size} candidate "
                f"lap(s), fewer than n_folds={n_folds}."
            )
        candidates_by_trajectory[trajectory_type] = candidate_indices
        chunks_by_trajectory[trajectory_type] = [
            np.sort(chunk.astype(int, copy=False))
            for chunk in np.array_split(rng.permutation(candidate_indices), n_folds)
        ]

    folds: list[dict[str, Any]] = []
    for fold_index in range(n_folds):
        train_indices_by_trajectory: dict[str, np.ndarray] = {}
        test_indices_by_trajectory: dict[str, np.ndarray] = {}
        metadata: dict[str, Any] = {"fold_index": int(fold_index)}
        for trajectory_type in TRAJECTORY_TYPES:
            candidate_indices = candidates_by_trajectory[trajectory_type]
            test_indices = chunks_by_trajectory[trajectory_type][fold_index]
            train_indices = np.setdiff1d(
                candidate_indices,
                test_indices,
                assume_unique=True,
            )
            starts, ends = bounds_by_trajectory[trajectory_type]
            train_indices_by_trajectory[trajectory_type] = train_indices
            test_indices_by_trajectory[trajectory_type] = test_indices
            metadata[f"{trajectory_type}_train_indices"] = train_indices.tolist()
            metadata[f"{trajectory_type}_test_indices"] = test_indices.tolist()
            metadata[f"{trajectory_type}_test_start_s"] = starts[test_indices].tolist()
            metadata[f"{trajectory_type}_test_end_s"] = ends[test_indices].tolist()
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_indices_by_trajectory": train_indices_by_trajectory,
                "test_indices_by_trajectory": test_indices_by_trajectory,
                "metadata": metadata,
            }
        )
    return folds


def build_lap_row_mask(
    *,
    spike_counts: Any,
    base_keep_mask: np.ndarray,
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    lap_indices_by_trajectory: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the binned row mask for selected movement-restricted laps."""
    mask = np.zeros_like(np.asarray(base_keep_mask, dtype=bool))
    for trajectory_type in TRAJECTORY_TYPES:
        lap_indices = np.asarray(
            lap_indices_by_trajectory[trajectory_type],
            dtype=int,
        ).reshape(-1)
        if lap_indices.size == 0:
            continue
        lap_interval = _subset_intervalset(
            trajectory_intervals[trajectory_type],
            lap_indices,
        ).intersect(movement_interval)
        mask |= np.asarray(spike_counts.in_interval(lap_interval), dtype=bool).reshape(-1)
    return mask & np.asarray(base_keep_mask, dtype=bool)


def build_fold_row_masks(
    data: dict[str, Any],
    fold: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test binned row masks for one lap-level fold."""
    train_rows = build_lap_row_mask(
        spike_counts=data["spike_counts"],
        base_keep_mask=data["base_keep_mask"],
        trajectory_intervals=data["trajectory_intervals"],
        movement_interval=data["movement_interval"],
        lap_indices_by_trajectory=fold["train_indices_by_trajectory"],
    )
    test_rows = build_lap_row_mask(
        spike_counts=data["spike_counts"],
        base_keep_mask=data["base_keep_mask"],
        trajectory_intervals=data["trajectory_intervals"],
        movement_interval=data["movement_interval"],
        lap_indices_by_trajectory=fold["test_indices_by_trajectory"],
    )
    if not np.any(train_rows):
        raise ValueError(f"Fold {fold['fold_index']} has no train bins after movement restriction.")
    if not np.any(test_rows):
        raise ValueError(f"Fold {fold['fold_index']} has no test bins after movement restriction.")
    return train_rows, test_rows


def prepare_motor_epoch_data(
    *,
    spikes: Any,
    position_tsd: Any,
    body_position_tsd: Any,
    trajectory_intervals: dict[str, Any],
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    bin_size_s: float,
    tp_bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    """Build binned response, covariates, labels, and fit-eligible row masks."""
    import pynapple as nap

    for trajectory_type in TRAJECTORY_TYPES:
        if trajectory_type not in trajectory_intervals:
            raise ValueError(
                f"trajectory_intervals is missing required key: {trajectory_type}"
            )
        if trajectory_type not in task_progression_by_trajectory:
            raise ValueError(
                f"task_progression_by_trajectory is missing required key: {trajectory_type}"
            )

    position_times = np.asarray(position_tsd.t, dtype=float)
    all_interval = nap.IntervalSet(
        start=float(position_times[0]),
        end=float(position_times[-1]),
        time_units="s",
    )
    spike_counts = spikes.count(float(bin_size_s), ep=all_interval)
    unit_ids = np.asarray(spike_counts.columns)
    response = np.asarray(spike_counts.d, dtype=float)
    n_time_bins = response.shape[0]

    in_movement = np.asarray(spike_counts.in_interval(movement_interval), dtype=bool).reshape(-1)
    in_trajectory = {
        trajectory_type: np.asarray(
            spike_counts.in_interval(trajectory_intervals[trajectory_type]),
            dtype=bool,
        ).reshape(-1)
        for trajectory_type in TRAJECTORY_TYPES
    }
    in_any_trajectory = np.zeros(n_time_bins, dtype=bool)
    overlap_count = np.zeros(n_time_bins, dtype=int)
    for trajectory_type in TRAJECTORY_TYPES:
        in_any_trajectory |= in_trajectory[trajectory_type]
        overlap_count += in_trajectory[trajectory_type].astype(int)

    base_keep_mask = in_movement & in_any_trajectory
    if not np.any(base_keep_mask):
        raise ValueError("No bins remain after restricting to movement and trajectories.")
    if np.any(base_keep_mask & (overlap_count != 1)):
        raise ValueError(
            "Trajectory intervals appear to overlap within kept bins. "
            "Each retained time bin must belong to exactly one trajectory."
        )

    trajectory_labels = np.full(n_time_bins, -1, dtype=int)
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        trajectory_labels[in_trajectory[trajectory_type]] = trajectory_index

    motor_covariates = compute_motor_covariates(
        position_xy=np.asarray(position_tsd.d, dtype=float),
        body_xy=np.asarray(body_position_tsd.d, dtype=float),
        position_timestamps=np.asarray(position_tsd.t, dtype=float),
        spike_counts=spike_counts,
    )

    tp_lower, tp_upper = map(float, tp_bounds)
    if not np.isfinite(tp_lower) or not np.isfinite(tp_upper) or tp_upper <= tp_lower:
        raise ValueError(f"Invalid task-progression bounds: {tp_bounds!r}")

    task_progression = np.full(n_time_bins, np.nan, dtype=float)
    for trajectory_type in TRAJECTORY_TYPES:
        interpolated_tp = task_progression_by_trajectory[trajectory_type].interpolate(
            spike_counts
        )
        interpolated_tp = np.asarray(interpolated_tp.to_numpy(), dtype=float).reshape(-1)
        trajectory_mask = in_trajectory[trajectory_type]
        if interpolated_tp.size != int(trajectory_mask.sum()):
            raise RuntimeError(
                f"{trajectory_type}: interpolated task progression size "
                f"{interpolated_tp.size} did not match the number of bins in the "
                f"trajectory mask {int(trajectory_mask.sum())}."
            )
        task_progression[trajectory_mask] = interpolated_tp
    task_progression = np.clip(task_progression, tp_lower, tp_upper)

    finite_covariates = np.ones(n_time_bins, dtype=bool)
    for values in motor_covariates.values():
        finite_covariates &= np.isfinite(values)
    base_keep_mask &= finite_covariates & np.isfinite(task_progression)
    if not np.any(base_keep_mask):
        raise ValueError("No bins remain after dropping non-finite motor or TP samples.")

    return {
        "spike_counts": spike_counts,
        "unit_ids": unit_ids,
        "response": response,
        "trajectory_intervals": trajectory_intervals,
        "movement_interval": movement_interval,
        "base_keep_mask": base_keep_mask,
        "trajectory_labels": trajectory_labels,
        "task_progression": task_progression,
        "motor_covariates": motor_covariates,
        "bin_size_s": float(bin_size_s),
        "tp_bounds": (tp_lower, tp_upper),
    }


def _estimate_spline_bounds(values: np.ndarray) -> tuple[float, float]:
    """Return robust finite bounds for spline-expanded motor features."""
    values = np.asarray(values, dtype=float).reshape(-1)
    lower = float(np.nanpercentile(values, 1.0))
    upper = float(np.nanpercentile(values, 99.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = 0.0, 1.0
    return lower, upper


def fit_motor_feature_transform(
    data: dict[str, Any],
    train_rows: np.ndarray,
    *,
    motor_feature_mode: Literal["zscore", "spline"],
    motor_zscore_eps: float,
    motor_spline_k: int,
    motor_spline_order: int,
) -> dict[str, Any]:
    """Fit the train-only motor preprocessing transform for one split."""
    train_rows = np.asarray(train_rows, dtype=bool)
    if motor_feature_mode == "zscore":
        raw_feature_names = list(MOTOR_RAW_FEATURE_NAMES)
        train_raw = np.column_stack(
            [
                np.asarray(data["motor_covariates"][name][train_rows], dtype=float)
                for name in raw_feature_names
            ]
        )
        mean = train_raw.mean(axis=0)
        std_raw = train_raw.std(axis=0)
        std = std_raw.copy()
        constant_mask = (~np.isfinite(std)) | (std < float(motor_zscore_eps))
        std[constant_mask] = 1.0
        return {
            "mode": "zscore",
            "raw_feature_names": raw_feature_names,
            "feature_names": [f"{name}_z" for name in raw_feature_names],
            "mean": mean,
            "std": std,
            "std_raw": std_raw,
            "constant_mask": constant_mask,
            "eps": float(motor_zscore_eps),
        }
    if motor_feature_mode == "spline":
        bounds_by_feature = {
            covariate_name: _estimate_spline_bounds(
                np.asarray(data["motor_covariates"][covariate_name][train_rows], dtype=float)
            )
            for covariate_name in MOTOR_CONTINUOUS_FEATURE_NAMES
        }
        feature_names: list[str] = []
        for covariate_name in MOTOR_CONTINUOUS_FEATURE_NAMES:
            feature_names.extend(
                f"{covariate_name}_bs{basis_index}"
                for basis_index in range(int(motor_spline_k))
            )
        feature_names.extend(["sin_hd", "cos_hd"])
        return {
            "mode": "spline",
            "continuous_feature_names": list(MOTOR_CONTINUOUS_FEATURE_NAMES),
            "feature_names": feature_names,
            "n_basis": int(motor_spline_k),
            "order": int(motor_spline_order),
            "bounds_by_feature": bounds_by_feature,
        }
    raise ValueError(
        f"motor_feature_mode must be 'zscore' or 'spline', got {motor_feature_mode!r}"
    )


def apply_motor_feature_transform(
    data: dict[str, Any],
    rows: np.ndarray,
    transform: dict[str, Any],
) -> np.ndarray:
    """Apply one train-fitted motor preprocessing transform."""
    rows = np.asarray(rows, dtype=bool)
    if transform["mode"] == "zscore":
        raw_feature_names = list(transform["raw_feature_names"])
        raw = np.column_stack(
            [
                np.asarray(data["motor_covariates"][name][rows], dtype=float)
                for name in raw_feature_names
            ]
        )
        design = (raw - np.asarray(transform["mean"], dtype=float)) / np.asarray(
            transform["std"],
            dtype=float,
        )
        constant_mask = np.asarray(transform["constant_mask"], dtype=bool)
        if np.any(constant_mask):
            design[:, constant_mask] = 0.0
        return design

    if transform["mode"] == "spline":
        blocks: list[np.ndarray] = []
        for covariate_name in transform["continuous_feature_names"]:
            features = bspline_features(
                np.asarray(data["motor_covariates"][covariate_name][rows], dtype=float),
                n_basis=int(transform["n_basis"]),
                order=int(transform["order"]),
                bounds=tuple(transform["bounds_by_feature"][covariate_name]),
            )
            blocks.append(features)
        blocks.append(np.asarray(data["motor_covariates"]["sin_hd"][rows], dtype=float)[:, None])
        blocks.append(np.asarray(data["motor_covariates"]["cos_hd"][rows], dtype=float)[:, None])
        return np.concatenate(blocks, axis=1)

    raise ValueError(f"Unknown motor transform mode: {transform['mode']!r}")


def _trajectory_group_labels(trajectory_labels: np.ndarray) -> np.ndarray:
    """Return TP group labels for trajectory labels."""
    trajectory_to_index = {
        trajectory_type: index for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }
    trajectory_index_to_group = np.full(len(TRAJECTORY_TYPES), -1, dtype=int)
    for group_index, (_group_name, grouped_trajectories) in enumerate(TP_GROUPS):
        for trajectory_type in grouped_trajectories:
            trajectory_index_to_group[trajectory_to_index[trajectory_type]] = group_index
    if np.any(trajectory_index_to_group < 0):
        missing = [
            TRAJECTORY_TYPES[index]
            for index in np.flatnonzero(trajectory_index_to_group < 0)
        ]
        raise ValueError(f"Some trajectories were not assigned to a TP group: {missing!r}")
    return trajectory_index_to_group[np.asarray(trajectory_labels, dtype=int)]


def build_model_designs(
    data: dict[str, Any],
    rows: np.ndarray,
    motor_transform: dict[str, Any],
    *,
    tp_spline_k: int,
    tp_spline_order: int,
) -> dict[str, Any]:
    """Build all model design matrices for selected rows."""
    rows = np.asarray(rows, dtype=bool)
    motor_design = apply_motor_feature_transform(data, rows, motor_transform)
    trajectory_labels = np.asarray(data["trajectory_labels"][rows], dtype=int)
    task_progression = np.asarray(data["task_progression"][rows], dtype=float)
    group_labels = _trajectory_group_labels(trajectory_labels)

    tp_group_design = (group_labels == 1).astype(float)[:, None]
    tp_group_feature_names = [f"is_{TP_GROUPS[1][0]}"]

    place_trajectory_blocks: list[np.ndarray] = []
    place_trajectory_feature_names: list[str] = []
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES[1:], start=1):
        place_trajectory_blocks.append((trajectory_labels == trajectory_index).astype(float)[:, None])
        place_trajectory_feature_names.append(f"is_{trajectory_type}")
    place_trajectory_design = np.concatenate(place_trajectory_blocks, axis=1)

    tp_lower, tp_upper = data["tp_bounds"]
    tp_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(float(tp_lower), float(tp_upper)),
    )
    tp_basis_features = np.asarray(tp_basis.compute_features(task_progression), dtype=float)
    if not np.all(np.isfinite(tp_basis_features)):
        raise ValueError("Encountered non-finite task-progression spline features.")

    tp_feature_blocks: list[np.ndarray] = []
    tp_feature_names: list[str] = []
    n_tp_basis = tp_basis_features.shape[1]
    trajectory_to_index = {
        trajectory_type: index for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }
    for group_index, (group_name, grouped_trajectories) in enumerate(TP_GROUPS):
        grouped_indices = [
            trajectory_to_index[trajectory_type]
            for trajectory_type in grouped_trajectories
        ]
        gate = np.isin(trajectory_labels, grouped_indices).astype(float)[:, None]
        tp_feature_blocks.append(tp_basis_features * gate)
        tp_feature_names.extend(
            f"tp_{group_name}_bs{basis_index}" for basis_index in range(n_tp_basis)
        )
    tp_design = np.concatenate(tp_feature_blocks, axis=1)

    place_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(float(tp_lower), float(tp_upper)),
    )
    place_basis_features = np.asarray(
        place_basis.compute_features(task_progression),
        dtype=float,
    )
    if not np.all(np.isfinite(place_basis_features)):
        raise ValueError("Encountered non-finite place spline features.")

    place_feature_blocks: list[np.ndarray] = []
    place_feature_names: list[str] = []
    n_place_basis = place_basis_features.shape[1]
    for trajectory_type in TRAJECTORY_TYPES:
        trajectory_index = trajectory_to_index[trajectory_type]
        gate = (trajectory_labels == trajectory_index).astype(float)[:, None]
        place_feature_blocks.append(place_basis_features * gate)
        place_feature_names.extend(
            f"place_{trajectory_type}_bs{basis_index}"
            for basis_index in range(n_place_basis)
        )
    place_design = np.concatenate(place_feature_blocks, axis=1)

    designs: dict[str, np.ndarray] = {}
    blocks: dict[str, dict[str, slice]] = {}
    designs["motor"] = motor_design
    blocks["motor"] = {"motor": slice(0, motor_design.shape[1])}

    start = 0
    motor_tp_blocks = [motor_design, tp_group_design, tp_design]
    designs["motor_tp"] = np.concatenate(motor_tp_blocks, axis=1)
    blocks["motor_tp"] = {
        "motor": slice(start, start + motor_design.shape[1]),
    }
    start += motor_design.shape[1]
    blocks["motor_tp"]["tp_group"] = slice(start, start + tp_group_design.shape[1])
    start += tp_group_design.shape[1]
    blocks["motor_tp"]["tp"] = slice(start, start + tp_design.shape[1])

    start = 0
    designs["tp_only"] = np.concatenate([tp_group_design, tp_design], axis=1)
    blocks["tp_only"] = {
        "tp_group": slice(start, start + tp_group_design.shape[1]),
    }
    start += tp_group_design.shape[1]
    blocks["tp_only"]["tp"] = slice(start, start + tp_design.shape[1])

    start = 0
    designs["motor_place"] = np.concatenate(
        [motor_design, place_trajectory_design, place_design],
        axis=1,
    )
    blocks["motor_place"] = {
        "motor": slice(start, start + motor_design.shape[1]),
    }
    start += motor_design.shape[1]
    blocks["motor_place"]["place_traj"] = slice(
        start,
        start + place_trajectory_design.shape[1],
    )
    start += place_trajectory_design.shape[1]
    blocks["motor_place"]["place"] = slice(start, start + place_design.shape[1])

    start = 0
    designs["place_only"] = np.concatenate(
        [place_trajectory_design, place_design],
        axis=1,
    )
    blocks["place_only"] = {
        "place_traj": slice(start, start + place_trajectory_design.shape[1]),
    }
    start += place_trajectory_design.shape[1]
    blocks["place_only"]["place"] = slice(start, start + place_design.shape[1])

    for matrix_name, matrix in designs.items():
        if not np.all(np.isfinite(matrix)):
            bad_row, bad_col = np.argwhere(~np.isfinite(matrix))[0]
            raise ValueError(
                f"{matrix_name} contains non-finite values at "
                f"row {int(bad_row)}, column {int(bad_col)}."
            )

    return {
        "designs": designs,
        "blocks": blocks,
        "motor_design": motor_design,
        "feature_names": {
            "motor": list(motor_transform["feature_names"]),
            "tp_group": tp_group_feature_names,
            "place_traj": place_trajectory_feature_names,
            "tp": tp_feature_names,
            "place": place_feature_names,
        },
        "basis": {
            "tp_n_splines": int(n_tp_basis),
            "place_n_splines": int(n_place_basis),
            "order": int(tp_spline_order),
            "bounds": (float(tp_lower), float(tp_upper)),
        },
    }


def _poisson_ll_sum(y_true: np.ndarray, lam_pred: np.ndarray) -> np.ndarray:
    """Return Poisson log-likelihood sums per unit."""
    y_true = np.asarray(y_true, dtype=float)
    lam_pred = np.clip(np.asarray(lam_pred, dtype=float), 1e-12, None)
    return np.sum(
        y_true * np.log(lam_pred) - lam_pred - gammaln(y_true + 1.0),
        axis=0,
    )


def _bits_per_spike_from_ll(
    ll_sum: np.ndarray,
    null_ll_sum: np.ndarray,
    spike_sum: np.ndarray,
) -> np.ndarray:
    """Return null-corrected bits/spike with NaN for zero-spike units."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.asarray(spike_sum, dtype=float) > 0.0,
            (
                np.asarray(ll_sum, dtype=float)
                - np.asarray(null_ll_sum, dtype=float)
            )
            / np.asarray(spike_sum, dtype=float)
            / np.log(2.0),
            np.nan,
        )


def _delta_bits_per_spike(
    ll_sum_a: np.ndarray,
    ll_sum_b: np.ndarray,
    spike_sum: np.ndarray,
) -> np.ndarray:
    """Return model-vs-model LL delta in bits/spike."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            np.asarray(spike_sum, dtype=float) > 0.0,
            (
                np.asarray(ll_sum_a, dtype=float)
                - np.asarray(ll_sum_b, dtype=float)
            )
            / np.asarray(spike_sum, dtype=float)
            / np.log(2.0),
            np.nan,
        )


def extract_model_coefficients(model: PopulationGLM, n_features: int) -> np.ndarray:
    """Return model coefficients in `(feature, unit)` order."""
    coefficients = np.asarray(model.coef_)
    if coefficients.shape[0] != n_features:
        coefficients = coefficients.T
    return coefficients


def _predict_poisson_mean(model: PopulationGLM, design: np.ndarray) -> np.ndarray:
    """Return predicted Poisson mean counts for one fitted model."""
    coefficients = extract_model_coefficients(model, design.shape[1])
    eta = np.asarray(model.intercept_).reshape(1, -1) + np.asarray(design) @ coefficients
    return np.exp(np.clip(eta, -50.0, 50.0))


def fit_population_glm(
    design: np.ndarray,
    response: np.ndarray,
    *,
    ridge: float,
) -> PopulationGLM:
    """Fit one ridge-regularized Poisson population GLM."""
    model = PopulationGLM(
        "Poisson",
        regularizer="Ridge",
        regularizer_strength=float(ridge),
    )
    model.fit(np.asarray(design, dtype=float), np.asarray(response, dtype=float))
    return model


def score_model_on_split(
    model: PopulationGLM,
    design_test: np.ndarray,
    response_test: np.ndarray,
) -> np.ndarray:
    """Return held-out Poisson log-likelihood sums for one fitted model."""
    lam = _predict_poisson_mean(model, design_test)
    return _poisson_ll_sum(response_test, lam)


def compute_null_ll_sum(
    response_train: np.ndarray,
    response_test: np.ndarray,
) -> np.ndarray:
    """Return train-rate constant null log-likelihood sums on test data."""
    null_rate = np.clip(np.mean(response_train, axis=0), 1e-12, None)
    null_lam = np.repeat(null_rate[None, :], response_test.shape[0], axis=0)
    return _poisson_ll_sum(response_test, null_lam)


def compute_ridge_cv_scores(
    data: dict[str, Any],
    folds: Sequence[dict[str, Any]],
    *,
    unit_mask: np.ndarray,
    ridge_values: Sequence[float],
    tp_spline_k: int,
    tp_spline_order: int,
    motor_feature_mode: Literal["zscore", "spline"],
    motor_zscore_eps: float,
    motor_spline_k: int,
    motor_spline_order: int,
) -> dict[str, Any]:
    """Return pooled lap-CV ridge scores for each model."""
    ridge_values = [float(value) for value in ridge_values]
    unit_mask = np.asarray(unit_mask, dtype=bool)
    if unit_mask.size != int(data["response"].shape[1]):
        raise ValueError(
            "Unit mask length does not match binned response columns. "
            f"Got {unit_mask.size} and {data['response'].shape[1]}."
        )
    unit_indices = np.flatnonzero(unit_mask)
    if unit_indices.size == 0:
        raise ValueError("No units were selected for ridge CV.")

    n_models = len(MODEL_NAMES)
    n_ridges = len(ridge_values)
    n_units = int(unit_indices.size)
    ll_sum = np.zeros((n_models, n_ridges, n_units), dtype=float)
    null_ll_sum = np.zeros(n_units, dtype=float)
    spike_sum = np.zeros(n_units, dtype=float)

    for fold in folds:
        train_rows, test_rows = build_fold_row_masks(data, fold)
        motor_transform = fit_motor_feature_transform(
            data,
            train_rows,
            motor_feature_mode=motor_feature_mode,
            motor_zscore_eps=motor_zscore_eps,
            motor_spline_k=motor_spline_k,
            motor_spline_order=motor_spline_order,
        )
        train_design_info = build_model_designs(
            data,
            train_rows,
            motor_transform,
            tp_spline_k=tp_spline_k,
            tp_spline_order=tp_spline_order,
        )
        test_design_info = build_model_designs(
            data,
            test_rows,
            motor_transform,
            tp_spline_k=tp_spline_k,
            tp_spline_order=tp_spline_order,
        )
        response_train = data["response"][train_rows][:, unit_indices]
        response_test = data["response"][test_rows][:, unit_indices]
        null_ll_sum += compute_null_ll_sum(response_train, response_test)
        spike_sum += np.asarray(response_test.sum(axis=0), dtype=float)

        for model_index, model_name in enumerate(MODEL_NAMES):
            x_train = train_design_info["designs"][model_name]
            x_test = test_design_info["designs"][model_name]
            for ridge_index, ridge in enumerate(ridge_values):
                model = fit_population_glm(x_train, response_train, ridge=ridge)
                ll_sum[model_index, ridge_index] += score_model_on_split(
                    model,
                    x_test,
                    response_test,
                )

    info_bits = _bits_per_spike_from_ll(
        ll_sum,
        null_ll_sum[None, None, :],
        spike_sum[None, None, :],
    )
    score_median = np.full((n_models, n_ridges), np.nan, dtype=float)
    score_mean = np.full((n_models, n_ridges), np.nan, dtype=float)
    score_n_finite = np.zeros((n_models, n_ridges), dtype=int)
    for model_index in range(n_models):
        for ridge_index in range(n_ridges):
            finite = info_bits[model_index, ridge_index][
                np.isfinite(info_bits[model_index, ridge_index])
            ]
            score_n_finite[model_index, ridge_index] = int(finite.size)
            if finite.size > 0:
                score_median[model_index, ridge_index] = float(np.median(finite))
                score_mean[model_index, ridge_index] = float(np.mean(finite))

    selected_ridge = np.full(n_models, np.nan, dtype=float)
    selected_score = np.full(n_models, np.nan, dtype=float)
    for model_index, model_name in enumerate(MODEL_NAMES):
        best_index: int | None = None
        for ridge_index, ridge in enumerate(ridge_values):
            score = float(score_median[model_index, ridge_index])
            if not np.isfinite(score):
                continue
            if best_index is None:
                best_index = ridge_index
                continue
            best_score = float(score_median[model_index, best_index])
            best_ridge = float(ridge_values[best_index])
            if score > best_score + HYPERPARAMETER_TIE_ATOL:
                best_index = ridge_index
            elif abs(score - best_score) <= HYPERPARAMETER_TIE_ATOL and ridge > best_ridge:
                best_index = ridge_index
        if best_index is None:
            raise ValueError(f"No finite ridge-CV score for model {model_name!r}.")
        selected_ridge[model_index] = float(ridge_values[best_index])
        selected_score[model_index] = float(score_median[model_index, best_index])

    return {
        "unit_indices": unit_indices,
        "ridge_values": np.asarray(ridge_values, dtype=float),
        "ll_sum": ll_sum,
        "null_ll_sum": null_ll_sum,
        "spike_sum": spike_sum,
        "info_bits_per_spike": info_bits,
        "score_median": score_median,
        "score_mean": score_mean,
        "score_n_finite": score_n_finite,
        "selected_ridge": selected_ridge,
        "selected_score": selected_score,
    }


def score_models_on_split(
    data: dict[str, Any],
    *,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    unit_mask: np.ndarray,
    ridge_by_model: dict[str, float],
    tp_spline_k: int,
    tp_spline_order: int,
    motor_feature_mode: Literal["zscore", "spline"],
    motor_zscore_eps: float,
    motor_spline_k: int,
    motor_spline_order: int,
) -> dict[str, Any]:
    """Fit selected-ridge models on train rows and score held-out rows."""
    unit_mask = np.asarray(unit_mask, dtype=bool)
    if unit_mask.size != int(data["response"].shape[1]):
        raise ValueError(
            "Unit mask length does not match binned response columns. "
            f"Got {unit_mask.size} and {data['response'].shape[1]}."
        )
    unit_indices = np.flatnonzero(unit_mask)
    if unit_indices.size == 0:
        raise ValueError("No units were selected for split scoring.")
    motor_transform = fit_motor_feature_transform(
        data,
        train_rows,
        motor_feature_mode=motor_feature_mode,
        motor_zscore_eps=motor_zscore_eps,
        motor_spline_k=motor_spline_k,
        motor_spline_order=motor_spline_order,
    )
    train_design_info = build_model_designs(
        data,
        train_rows,
        motor_transform,
        tp_spline_k=tp_spline_k,
        tp_spline_order=tp_spline_order,
    )
    test_design_info = build_model_designs(
        data,
        test_rows,
        motor_transform,
        tp_spline_k=tp_spline_k,
        tp_spline_order=tp_spline_order,
    )
    response_train = data["response"][train_rows][:, unit_indices]
    response_test = data["response"][test_rows][:, unit_indices]
    null_ll_sum = compute_null_ll_sum(response_train, response_test)
    spike_sum = np.asarray(response_test.sum(axis=0), dtype=float)

    ll_sum = np.zeros((len(MODEL_NAMES), unit_indices.size), dtype=float)
    fitted_models: dict[str, PopulationGLM] = {}
    for model_index, model_name in enumerate(MODEL_NAMES):
        model = fit_population_glm(
            train_design_info["designs"][model_name],
            response_train,
            ridge=float(ridge_by_model[model_name]),
        )
        fitted_models[model_name] = model
        ll_sum[model_index] = score_model_on_split(
            model,
            test_design_info["designs"][model_name],
            response_test,
        )
    info_bits = _bits_per_spike_from_ll(
        ll_sum,
        null_ll_sum[None, :],
        spike_sum[None, :],
    )
    return {
        "unit_indices": unit_indices,
        "ll_sum": ll_sum,
        "null_ll_sum": null_ll_sum,
        "spike_sum": spike_sum,
        "info_bits_per_spike": info_bits,
        "fitted_models": fitted_models,
    }


def compute_train_unit_mask(
    data: dict[str, Any],
    train_rows: np.ndarray,
    *,
    threshold_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train-only firing-rate mask and rates for all units."""
    train_rows = np.asarray(train_rows, dtype=bool)
    train_duration_s = float(np.sum(train_rows) * data["bin_size_s"])
    if train_duration_s <= 0.0:
        raise ValueError("Cannot compute train firing rates from an empty train split.")
    train_rates = np.asarray(data["response"][train_rows].sum(axis=0), dtype=float) / train_duration_s
    return get_unit_mask(train_rates, threshold_hz), train_rates


def run_nested_lap_cv(
    data: dict[str, Any],
    outer_folds: Sequence[dict[str, Any]],
    *,
    ridge_values: Sequence[float],
    inner_n_folds: int,
    seed: int,
    min_firing_rate_hz: float,
    tp_spline_k: int,
    tp_spline_order: int,
    motor_feature_mode: Literal["zscore", "spline"],
    motor_zscore_eps: float,
    motor_spline_k: int,
    motor_spline_order: int,
    print_prefix: str = "",
) -> dict[str, Any]:
    """Run nested lap-level CV with per-model population ridge selection."""
    ridge_values = [float(value) for value in ridge_values]
    n_outer = len(outer_folds)
    n_models = len(MODEL_NAMES)
    n_ridges = len(ridge_values)
    n_units = int(data["response"].shape[1])

    outer_ll_sum = np.full((n_outer, n_models, n_units), np.nan, dtype=float)
    outer_null_ll_sum = np.full((n_outer, n_units), np.nan, dtype=float)
    outer_spike_sum = np.full((n_outer, n_units), np.nan, dtype=float)
    outer_info = np.full((n_outer, n_models, n_units), np.nan, dtype=float)
    outer_selected_ridge = np.full((n_outer, n_models), np.nan, dtype=float)
    outer_selected_score = np.full((n_outer, n_models), np.nan, dtype=float)
    outer_unit_selected = np.zeros((n_outer, n_units), dtype=np.int8)
    inner_info = np.full((n_outer, n_models, n_ridges, n_units), np.nan, dtype=float)
    inner_score_median = np.full((n_outer, n_models, n_ridges), np.nan, dtype=float)
    inner_score_mean = np.full((n_outer, n_models, n_ridges), np.nan, dtype=float)
    inner_score_n_finite = np.zeros((n_outer, n_models, n_ridges), dtype=int)
    outer_train_bin_count = np.zeros(n_outer, dtype=int)
    outer_test_bin_count = np.zeros(n_outer, dtype=int)

    for outer_index, outer_fold in enumerate(outer_folds):
        train_rows, test_rows = build_fold_row_masks(data, outer_fold)
        outer_train_bin_count[outer_index] = int(np.sum(train_rows))
        outer_test_bin_count[outer_index] = int(np.sum(test_rows))
        unit_mask, _train_rates = compute_train_unit_mask(
            data,
            train_rows,
            threshold_hz=min_firing_rate_hz,
        )
        n_selected = int(np.sum(unit_mask))
        print(
            f"{print_prefix}Outer fold {outer_index + 1}/{n_outer}: "
            f"{n_selected}/{n_units} units pass train-only FR>{min_firing_rate_hz:.3f} Hz; "
            f"train bins={outer_train_bin_count[outer_index]}, "
            f"test bins={outer_test_bin_count[outer_index]}."
        )
        if n_selected == 0:
            raise ValueError(
                f"Outer fold {outer_index} selected no units at "
                f"FR>{min_firing_rate_hz:.3f} Hz."
            )
        outer_unit_selected[outer_index, unit_mask] = 1

        inner_folds = build_lap_cv_folds_for_epoch(
            data["trajectory_intervals"],
            n_folds=int(inner_n_folds),
            seed=int(seed) + 1000 + outer_index,
            candidate_indices_by_trajectory=outer_fold["train_indices_by_trajectory"],
        )
        ridge_scores = compute_ridge_cv_scores(
            data,
            inner_folds,
            unit_mask=unit_mask,
            ridge_values=ridge_values,
            tp_spline_k=tp_spline_k,
            tp_spline_order=tp_spline_order,
            motor_feature_mode=motor_feature_mode,
            motor_zscore_eps=motor_zscore_eps,
            motor_spline_k=motor_spline_k,
            motor_spline_order=motor_spline_order,
        )
        unit_indices = ridge_scores["unit_indices"]
        inner_info[outer_index][:, :, unit_indices] = ridge_scores[
            "info_bits_per_spike"
        ]
        inner_score_median[outer_index] = ridge_scores["score_median"]
        inner_score_mean[outer_index] = ridge_scores["score_mean"]
        inner_score_n_finite[outer_index] = ridge_scores["score_n_finite"]
        outer_selected_ridge[outer_index] = ridge_scores["selected_ridge"]
        outer_selected_score[outer_index] = ridge_scores["selected_score"]
        ridge_by_model = {
            model_name: float(ridge_scores["selected_ridge"][model_index])
            for model_index, model_name in enumerate(MODEL_NAMES)
        }
        print(
            f"{print_prefix}  Selected ridges: "
            + ", ".join(
                f"{model_name}={ridge_by_model[model_name]:.3g}"
                for model_name in MODEL_NAMES
            )
        )

        split_scores = score_models_on_split(
            data,
            train_rows=train_rows,
            test_rows=test_rows,
            unit_mask=unit_mask,
            ridge_by_model=ridge_by_model,
            tp_spline_k=tp_spline_k,
            tp_spline_order=tp_spline_order,
            motor_feature_mode=motor_feature_mode,
            motor_zscore_eps=motor_zscore_eps,
            motor_spline_k=motor_spline_k,
            motor_spline_order=motor_spline_order,
        )
        outer_ll_sum[outer_index][:, split_scores["unit_indices"]] = split_scores[
            "ll_sum"
        ]
        outer_null_ll_sum[outer_index, split_scores["unit_indices"]] = split_scores[
            "null_ll_sum"
        ]
        outer_spike_sum[outer_index, split_scores["unit_indices"]] = split_scores[
            "spike_sum"
        ]
        outer_info[outer_index][:, split_scores["unit_indices"]] = split_scores[
            "info_bits_per_spike"
        ]

        motor_index = MODEL_NAMES.index("motor")
        motor_tp_index = MODEL_NAMES.index("motor_tp")
        motor_place_index = MODEL_NAMES.index("motor_place")
        fold_spike_sum = split_scores["spike_sum"]
        fold_tp_delta = _delta_bits_per_spike(
            split_scores["ll_sum"][motor_tp_index],
            split_scores["ll_sum"][motor_index],
            fold_spike_sum,
        )
        fold_place_delta = _delta_bits_per_spike(
            split_scores["ll_sum"][motor_place_index],
            split_scores["ll_sum"][motor_index],
            fold_spike_sum,
        )
        print(
            f"{print_prefix}  Held-out primary medians: "
            f"motor+TP - motor={np.nanmedian(fold_tp_delta):.5g}, "
            f"motor+place - motor={np.nanmedian(fold_place_delta):.5g} bits/spike."
        )

    valid = np.isfinite(outer_spike_sum) & (outer_unit_selected > 0)
    pooled_spike_sum = np.sum(np.where(valid, outer_spike_sum, 0.0), axis=0)
    pooled_null_ll_sum = np.sum(np.where(valid, outer_null_ll_sum, 0.0), axis=0)
    pooled_ll_sum = np.zeros((n_models, n_units), dtype=float)
    for model_index in range(n_models):
        pooled_ll_sum[model_index] = np.sum(
            np.where(valid, outer_ll_sum[:, model_index, :], 0.0),
            axis=0,
        )
    pooled_info = _bits_per_spike_from_ll(
        pooled_ll_sum,
        pooled_null_ll_sum[None, :],
        pooled_spike_sum[None, :],
    )

    pooled_delta = np.full((len(DELTA_SPECS), n_units), np.nan, dtype=float)
    for delta_index, (_metric_name, model_a, model_b) in enumerate(DELTA_SPECS):
        pooled_delta[delta_index] = _delta_bits_per_spike(
            pooled_ll_sum[MODEL_NAMES.index(model_a)],
            pooled_ll_sum[MODEL_NAMES.index(model_b)],
            pooled_spike_sum,
        )

    return {
        "ridge_values": np.asarray(ridge_values, dtype=float),
        "outer_ll_sum": outer_ll_sum,
        "outer_null_ll_sum": outer_null_ll_sum,
        "outer_spike_sum": outer_spike_sum,
        "outer_info_bits_per_spike": outer_info,
        "outer_selected_ridge": outer_selected_ridge,
        "outer_selected_score": outer_selected_score,
        "outer_unit_selected": outer_unit_selected,
        "inner_cv_info_bits_per_spike": inner_info,
        "inner_cv_score_median": inner_score_median,
        "inner_cv_score_mean": inner_score_mean,
        "inner_cv_score_n_finite": inner_score_n_finite,
        "outer_train_bin_count": outer_train_bin_count,
        "outer_test_bin_count": outer_test_bin_count,
        "pooled_ll_sum": pooled_ll_sum,
        "pooled_null_ll_sum": pooled_null_ll_sum,
        "pooled_spike_sum": pooled_spike_sum,
        "pooled_info_bits_per_spike": pooled_info,
        "pooled_delta_bits_per_spike": pooled_delta,
    }


def fit_full_refit_models(
    data: dict[str, Any],
    *,
    unit_mask: np.ndarray,
    ridge_by_model: dict[str, float],
    tp_spline_k: int,
    tp_spline_order: int,
    motor_feature_mode: Literal["zscore", "spline"],
    motor_zscore_eps: float,
    motor_spline_k: int,
    motor_spline_order: int,
) -> dict[str, Any]:
    """Fit selected-ridge models on all eligible bins for coefficient outputs."""
    all_rows = np.asarray(data["base_keep_mask"], dtype=bool)
    unit_mask = np.asarray(unit_mask, dtype=bool)
    if unit_mask.size != int(data["response"].shape[1]):
        raise ValueError(
            "Unit mask length does not match binned response columns. "
            f"Got {unit_mask.size} and {data['response'].shape[1]}."
        )
    unit_indices = np.flatnonzero(unit_mask)
    if unit_indices.size == 0:
        raise ValueError("No units were selected for full-data refit.")
    motor_transform = fit_motor_feature_transform(
        data,
        all_rows,
        motor_feature_mode=motor_feature_mode,
        motor_zscore_eps=motor_zscore_eps,
        motor_spline_k=motor_spline_k,
        motor_spline_order=motor_spline_order,
    )
    design_info = build_model_designs(
        data,
        all_rows,
        motor_transform,
        tp_spline_k=tp_spline_k,
        tp_spline_order=tp_spline_order,
    )
    response = data["response"][all_rows][:, unit_indices]
    fitted_models: dict[str, PopulationGLM] = {}
    coefficients: dict[str, np.ndarray] = {}
    intercepts: dict[str, np.ndarray] = {}
    for model_name in MODEL_NAMES:
        design = design_info["designs"][model_name]
        model = fit_population_glm(
            design,
            response,
            ridge=float(ridge_by_model[model_name]),
        )
        fitted_models[model_name] = model
        coefficients[model_name] = extract_model_coefficients(model, design.shape[1])
        intercepts[model_name] = np.asarray(model.intercept_).reshape(-1)

    tp_lower, tp_upper = data["tp_bounds"]
    tp_grid = np.linspace(float(tp_lower), float(tp_upper), 200)
    place_grid = np.linspace(float(tp_lower), float(tp_upper), 200)
    motor_reference = (
        np.zeros(design_info["motor_design"].shape[1], dtype=float)
        if motor_feature_mode == "zscore"
        else np.asarray(design_info["motor_design"], dtype=float).mean(axis=0)
    )
    trajectory_to_index = {
        trajectory_type: index for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }
    trajectory_index_to_group = _trajectory_group_labels(
        np.arange(len(TRAJECTORY_TYPES), dtype=int)
    )
    tp_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(float(tp_lower), float(tp_upper)),
    )
    place_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(float(tp_lower), float(tp_upper)),
    )

    tp_rate_curves_hz: dict[str, dict[str, np.ndarray]] = {}
    place_rate_curves_hz: dict[str, dict[str, np.ndarray]] = {}
    n_tp_basis = int(design_info["basis"]["tp_n_splines"])
    n_place_basis = int(design_info["basis"]["place_n_splines"])
    for trajectory_type in TRAJECTORY_TYPES:
        trajectory_index = trajectory_to_index[trajectory_type]
        group_index = int(trajectory_index_to_group[trajectory_index])

        tp_grid_features = np.asarray(tp_basis.compute_features(tp_grid), dtype=float)
        tp_group_dummy = np.zeros((tp_grid.size, 1), dtype=float)
        if group_index == 1:
            tp_group_dummy[:, 0] = 1.0
        tp_block = np.zeros((tp_grid.size, n_tp_basis * len(TP_GROUPS)), dtype=float)
        start = group_index * n_tp_basis
        tp_block[:, start : start + n_tp_basis] = tp_grid_features
        motor_tp_grid_design = np.concatenate(
            [
                np.repeat(motor_reference[None, :], tp_grid.size, axis=0),
                tp_group_dummy,
                tp_block,
            ],
            axis=1,
        )
        eta = (
            intercepts["motor_tp"][None, :]
            + motor_tp_grid_design @ coefficients["motor_tp"]
        )
        tp_rate_curves_hz[trajectory_type] = {
            "tp_grid": tp_grid,
            "rate_hz": np.exp(np.clip(eta, -50.0, 50.0)) / float(data["bin_size_s"]),
        }

        place_grid_features = np.asarray(place_basis.compute_features(place_grid), dtype=float)
        place_trajectory_dummy = np.zeros((place_grid.size, len(TRAJECTORY_TYPES) - 1), dtype=float)
        if trajectory_index > 0:
            place_trajectory_dummy[:, trajectory_index - 1] = 1.0
        place_block = np.zeros((place_grid.size, n_place_basis * len(TRAJECTORY_TYPES)), dtype=float)
        place_start = trajectory_index * n_place_basis
        place_block[:, place_start : place_start + n_place_basis] = place_grid_features
        motor_place_grid_design = np.concatenate(
            [
                np.repeat(motor_reference[None, :], place_grid.size, axis=0),
                place_trajectory_dummy,
                place_block,
            ],
            axis=1,
        )
        place_eta = (
            intercepts["motor_place"][None, :]
            + motor_place_grid_design @ coefficients["motor_place"]
        )
        place_rate_curves_hz[trajectory_type] = {
            "place_grid": place_grid,
            "rate_hz": np.exp(np.clip(place_eta, -50.0, 50.0)) / float(data["bin_size_s"]),
        }

    return {
        "unit_indices": unit_indices,
        "unit_ids": np.asarray(data["unit_ids"])[unit_indices],
        "motor_transform": motor_transform,
        "design_info": design_info,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "tp_rate_curves_hz": tp_rate_curves_hz,
        "place_rate_curves_hz": place_rate_curves_hz,
        "ridge_by_model": ridge_by_model,
    }


def build_outer_lap_arrays(
    outer_folds: Sequence[dict[str, Any]],
    trajectory_intervals: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Pack outer-fold test-lap metadata into padded arrays."""
    max_test_laps = 0
    bounds_by_trajectory = {
        trajectory_type: _extract_interval_bounds(trajectory_intervals[trajectory_type])
        for trajectory_type in TRAJECTORY_TYPES
    }
    for fold in outer_folds:
        for trajectory_type in TRAJECTORY_TYPES:
            max_test_laps = max(
                max_test_laps,
                int(fold["test_indices_by_trajectory"][trajectory_type].size),
            )
    max_test_laps = max(1, max_test_laps)

    shape = (len(outer_folds), len(TRAJECTORY_TYPES), max_test_laps)
    test_lap_index = np.full(shape, -1, dtype=int)
    test_lap_start_s = np.full(shape, np.nan, dtype=float)
    test_lap_end_s = np.full(shape, np.nan, dtype=float)
    train_lap_count = np.zeros((len(outer_folds), len(TRAJECTORY_TYPES)), dtype=int)
    test_lap_count = np.zeros((len(outer_folds), len(TRAJECTORY_TYPES)), dtype=int)

    for fold_index, fold in enumerate(outer_folds):
        for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
            starts, ends = bounds_by_trajectory[trajectory_type]
            train_indices = np.asarray(
                fold["train_indices_by_trajectory"][trajectory_type],
                dtype=int,
            )
            test_indices = np.asarray(
                fold["test_indices_by_trajectory"][trajectory_type],
                dtype=int,
            )
            n_test = int(test_indices.size)
            train_lap_count[fold_index, trajectory_index] = int(train_indices.size)
            test_lap_count[fold_index, trajectory_index] = n_test
            test_lap_index[fold_index, trajectory_index, :n_test] = test_indices
            test_lap_start_s[fold_index, trajectory_index, :n_test] = starts[test_indices]
            test_lap_end_s[fold_index, trajectory_index, :n_test] = ends[test_indices]

    return {
        "test_lap_index": test_lap_index,
        "test_lap_start_s": test_lap_start_s,
        "test_lap_end_s": test_lap_end_s,
        "train_lap_count": train_lap_count,
        "test_lap_count": test_lap_count,
    }


def build_nested_cv_dataset(
    nested_result: dict[str, Any],
    outer_folds: Sequence[dict[str, Any]],
    data: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    min_firing_rate_hz: float,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build the nested lap-CV evidence dataset."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task-progression motor nested-CV outputs."
        ) from exc

    lap_arrays = build_outer_lap_arrays(outer_folds, data["trajectory_intervals"])
    outer_fold_count = int(nested_result["outer_ll_sum"].shape[0])
    dataset = xr.Dataset(
        data_vars={
            "outer_ll_sum": (
                ("outer_fold", "model", "unit"),
                np.asarray(nested_result["outer_ll_sum"], dtype=float),
            ),
            "outer_null_ll_sum": (
                ("outer_fold", "unit"),
                np.asarray(nested_result["outer_null_ll_sum"], dtype=float),
            ),
            "outer_spike_sum": (
                ("outer_fold", "unit"),
                np.asarray(nested_result["outer_spike_sum"], dtype=float),
            ),
            "outer_info_bits_per_spike": (
                ("outer_fold", "model", "unit"),
                np.asarray(nested_result["outer_info_bits_per_spike"], dtype=float),
            ),
            "outer_selected_ridge": (
                ("outer_fold", "model"),
                np.asarray(nested_result["outer_selected_ridge"], dtype=float),
            ),
            "outer_selected_score_median": (
                ("outer_fold", "model"),
                np.asarray(nested_result["outer_selected_score"], dtype=float),
            ),
            "outer_unit_selected": (
                ("outer_fold", "unit"),
                np.asarray(nested_result["outer_unit_selected"], dtype=np.int8),
            ),
            "outer_train_bin_count": (
                "outer_fold",
                np.asarray(nested_result["outer_train_bin_count"], dtype=int),
            ),
            "outer_test_bin_count": (
                "outer_fold",
                np.asarray(nested_result["outer_test_bin_count"], dtype=int),
            ),
            "inner_cv_info_bits_per_spike": (
                ("outer_fold", "model", "ridge", "unit"),
                np.asarray(nested_result["inner_cv_info_bits_per_spike"], dtype=float),
            ),
            "inner_cv_score_median": (
                ("outer_fold", "model", "ridge"),
                np.asarray(nested_result["inner_cv_score_median"], dtype=float),
            ),
            "inner_cv_score_mean": (
                ("outer_fold", "model", "ridge"),
                np.asarray(nested_result["inner_cv_score_mean"], dtype=float),
            ),
            "inner_cv_score_n_finite": (
                ("outer_fold", "model", "ridge"),
                np.asarray(nested_result["inner_cv_score_n_finite"], dtype=int),
            ),
            "pooled_ll_sum": (
                ("model", "unit"),
                np.asarray(nested_result["pooled_ll_sum"], dtype=float),
            ),
            "pooled_null_ll_sum": (
                "unit",
                np.asarray(nested_result["pooled_null_ll_sum"], dtype=float),
            ),
            "pooled_spike_sum": (
                "unit",
                np.asarray(nested_result["pooled_spike_sum"], dtype=float),
            ),
            "pooled_info_bits_per_spike": (
                ("model", "unit"),
                np.asarray(nested_result["pooled_info_bits_per_spike"], dtype=float),
            ),
            "pooled_delta_bits_per_spike": (
                ("delta_metric", "unit"),
                np.asarray(nested_result["pooled_delta_bits_per_spike"], dtype=float),
            ),
            "outer_test_lap_index": (
                ("outer_fold", "trajectory", "test_lap"),
                lap_arrays["test_lap_index"],
            ),
            "outer_test_lap_start_s": (
                ("outer_fold", "trajectory", "test_lap"),
                lap_arrays["test_lap_start_s"],
            ),
            "outer_test_lap_end_s": (
                ("outer_fold", "trajectory", "test_lap"),
                lap_arrays["test_lap_end_s"],
            ),
            "outer_train_lap_count": (
                ("outer_fold", "trajectory"),
                lap_arrays["train_lap_count"],
            ),
            "outer_test_lap_count": (
                ("outer_fold", "trajectory"),
                lap_arrays["test_lap_count"],
            ),
        },
        coords={
            "outer_fold": np.arange(outer_fold_count, dtype=int),
            "model": np.asarray(MODEL_NAMES, dtype=str),
            "ridge": np.asarray(nested_result["ridge_values"], dtype=float),
            "unit": np.asarray(data["unit_ids"]),
            "delta_metric": np.asarray(
                [metric_name for metric_name, _model_a, _model_b in DELTA_SPECS],
                dtype=str,
            ),
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "test_lap": np.arange(lap_arrays["test_lap_index"].shape[2], dtype=int),
        },
        attrs={
            "schema_version": "2",
            "fit_stage": "nested_lap_cv_evidence",
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "bin_size_s": float(data["bin_size_s"]),
            "min_firing_rate_hz": float(min_firing_rate_hz),
            "model_definitions_json": json.dumps(MODEL_DEFINITIONS, sort_keys=True),
            "primary_delta_metric_names_json": json.dumps(list(PRIMARY_DELTA_METRIC_NAMES)),
            "delta_specs_json": json.dumps(
                [
                    {
                        "metric_name": metric_name,
                        "model_a": model_a,
                        "model_b": model_b,
                    }
                    for metric_name, model_a, model_b in DELTA_SPECS
                ],
                sort_keys=True,
            ),
            "cv_fold_scope": "nested_lap_level_by_trajectory_movement_only",
            "unit_selection": "outer_train_movement_lap_firing_rate_threshold",
            "ridge_selection": (
                "per-model population ridge selected by median unit-level "
                "inner-CV null-corrected information"
            ),
            "motor_baseline_definition": (
                "strict motor covariates only; no trajectory group or trajectory identity terms"
            ),
            "motor_tp_includes_trajectory_group_offset": "true",
            "tp_only_includes_trajectory_group_offset": "true",
            "place_models_use_trajectory_specific_basis": "true",
            "place_models_include_trajectory_offset": "true",
            "sources_json": json.dumps(sources, sort_keys=True),
            "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        },
    )
    return dataset


def _stack_group_coefficients(
    coefficients: np.ndarray,
    block_slice: slice,
    *,
    n_basis: int,
) -> np.ndarray:
    """Stack grouped TP coefficients into `(tp_group, basis, unit)` order."""
    return np.stack(
        [
            coefficients[
                block_slice.start + group_index * n_basis : block_slice.start
                + (group_index + 1) * n_basis,
                :,
            ]
            for group_index in range(len(TP_GROUPS))
        ],
        axis=0,
    )


def _stack_trajectory_coefficients(
    coefficients: np.ndarray,
    block_slice: slice,
    *,
    n_basis: int,
) -> np.ndarray:
    """Stack trajectory-specific coefficients into `(trajectory, basis, unit)`."""
    return np.stack(
        [
            coefficients[
                block_slice.start + trajectory_index * n_basis : block_slice.start
                + (trajectory_index + 1) * n_basis,
                :,
            ]
            for trajectory_index in range(len(TRAJECTORY_TYPES))
        ],
        axis=0,
    )


def build_full_refit_dataset(
    full_fit: dict[str, Any],
    ridge_cv_result: dict[str, Any],
    data: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    movement_firing_rates: np.ndarray,
    min_firing_rate_hz: float,
    sources: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> "xr.Dataset":
    """Build the full-data refit dataset for coefficients and rate curves."""
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "xarray is required to save task-progression motor full-refit outputs."
        ) from exc

    design_info = full_fit["design_info"]
    blocks = design_info["blocks"]
    coefficients = full_fit["coefficients"]
    intercepts = full_fit["intercepts"]
    n_tp_basis = int(design_info["basis"]["tp_n_splines"])
    n_place_basis = int(design_info["basis"]["place_n_splines"])
    unit_indices = np.asarray(full_fit["unit_indices"], dtype=int)
    selected_unit_rates = np.asarray(movement_firing_rates, dtype=float)[unit_indices]
    tp_grid = np.asarray(
        full_fit["tp_rate_curves_hz"][TRAJECTORY_TYPES[0]]["tp_grid"],
        dtype=float,
    )
    place_grid = np.asarray(
        full_fit["place_rate_curves_hz"][TRAJECTORY_TYPES[0]]["place_grid"],
        dtype=float,
    )

    selected_ridge = np.asarray(
        [float(full_fit["ridge_by_model"][model_name]) for model_name in MODEL_NAMES],
        dtype=float,
    )
    dataset = xr.Dataset(
        data_vars={
            "movement_firing_rate_hz": ("unit", selected_unit_rates),
            "selected_ridge": ("model", selected_ridge),
            "full_cv_info_bits_per_spike": (
                ("model", "ridge", "unit"),
                np.asarray(ridge_cv_result["info_bits_per_spike"], dtype=float),
            ),
            "full_cv_score_median": (
                ("model", "ridge"),
                np.asarray(ridge_cv_result["score_median"], dtype=float),
            ),
            "full_cv_score_mean": (
                ("model", "ridge"),
                np.asarray(ridge_cv_result["score_mean"], dtype=float),
            ),
            "full_cv_score_n_finite": (
                ("model", "ridge"),
                np.asarray(ridge_cv_result["score_n_finite"], dtype=int),
            ),
            "motor_intercept": ("unit", intercepts["motor"]),
            "motor_coef_motor": (
                ("motor_feature", "unit"),
                coefficients["motor"][blocks["motor"]["motor"], :],
            ),
            "motor_tp_intercept": ("unit", intercepts["motor_tp"]),
            "motor_tp_coef_motor": (
                ("motor_feature", "unit"),
                coefficients["motor_tp"][blocks["motor_tp"]["motor"], :],
            ),
            "motor_tp_coef_tp_group_offset": (
                ("tp_group_feature", "unit"),
                coefficients["motor_tp"][blocks["motor_tp"]["tp_group"], :],
            ),
            "motor_tp_coef_tp_by_group": (
                ("tp_group", "tp_basis", "unit"),
                _stack_group_coefficients(
                    coefficients["motor_tp"],
                    blocks["motor_tp"]["tp"],
                    n_basis=n_tp_basis,
                ),
            ),
            "tp_only_intercept": ("unit", intercepts["tp_only"]),
            "tp_only_coef_tp_group_offset": (
                ("tp_group_feature", "unit"),
                coefficients["tp_only"][blocks["tp_only"]["tp_group"], :],
            ),
            "tp_only_coef_tp_by_group": (
                ("tp_group", "tp_basis", "unit"),
                _stack_group_coefficients(
                    coefficients["tp_only"],
                    blocks["tp_only"]["tp"],
                    n_basis=n_tp_basis,
                ),
            ),
            "motor_place_intercept": ("unit", intercepts["motor_place"]),
            "motor_place_coef_motor": (
                ("motor_feature", "unit"),
                coefficients["motor_place"][blocks["motor_place"]["motor"], :],
            ),
            "motor_place_coef_trajectory_offset": (
                ("place_traj_feature", "unit"),
                coefficients["motor_place"][blocks["motor_place"]["place_traj"], :],
            ),
            "motor_place_coef_place_by_trajectory": (
                ("trajectory", "place_basis", "unit"),
                _stack_trajectory_coefficients(
                    coefficients["motor_place"],
                    blocks["motor_place"]["place"],
                    n_basis=n_place_basis,
                ),
            ),
            "place_only_intercept": ("unit", intercepts["place_only"]),
            "place_only_coef_trajectory_offset": (
                ("place_traj_feature", "unit"),
                coefficients["place_only"][blocks["place_only"]["place_traj"], :],
            ),
            "place_only_coef_place_by_trajectory": (
                ("trajectory", "place_basis", "unit"),
                _stack_trajectory_coefficients(
                    coefficients["place_only"],
                    blocks["place_only"]["place"],
                    n_basis=n_place_basis,
                ),
            ),
            "tp_rate_curves_hz": (
                ("trajectory", "tp_grid", "unit"),
                stack_tp_rate_curves(full_fit["tp_rate_curves_hz"]),
            ),
            "place_rate_curves_hz": (
                ("trajectory", "place_grid", "unit"),
                stack_place_rate_curves(full_fit["place_rate_curves_hz"]),
            ),
        },
        coords={
            "model": np.asarray(MODEL_NAMES, dtype=str),
            "ridge": np.asarray(ridge_cv_result["ridge_values"], dtype=float),
            "unit": np.asarray(full_fit["unit_ids"]),
            "motor_feature": np.asarray(design_info["feature_names"]["motor"], dtype=str),
            "tp_group_feature": np.asarray(
                design_info["feature_names"]["tp_group"],
                dtype=str,
            ),
            "place_traj_feature": np.asarray(
                design_info["feature_names"]["place_traj"],
                dtype=str,
            ),
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "tp_group": np.asarray([group_name for group_name, _ in TP_GROUPS], dtype=str),
            "tp_basis": np.arange(n_tp_basis, dtype=int),
            "place_basis": np.arange(n_place_basis, dtype=int),
            "tp_grid": tp_grid,
            "place_grid": place_grid,
        },
        attrs={
            "schema_version": "2",
            "fit_stage": "full_data_refit_for_visualization",
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "bin_size_s": float(data["bin_size_s"]),
            "motor_feature_mode": str(full_fit["motor_transform"]["mode"]),
            "min_firing_rate_hz": float(min_firing_rate_hz),
            "tp_basis_n_splines": n_tp_basis,
            "tp_basis_order": int(design_info["basis"]["order"]),
            "place_basis_n_splines": n_place_basis,
            "place_basis_order": int(design_info["basis"]["order"]),
            "basis_bounds_lower": float(design_info["basis"]["bounds"][0]),
            "basis_bounds_upper": float(design_info["basis"]["bounds"][1]),
            "model_definitions_json": json.dumps(MODEL_DEFINITIONS, sort_keys=True),
            "primary_delta_metric_names_json": json.dumps(list(PRIMARY_DELTA_METRIC_NAMES)),
            "full_refit_note": (
                "Coefficients and rate curves are fit on all eligible bins after "
                "lap-CV ridge selection; use the nested-CV dataset for held-out evidence."
            ),
            "sources_json": json.dumps(sources, sort_keys=True),
            "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        },
    )

    motor_transform = full_fit["motor_transform"]
    if motor_transform["mode"] == "zscore":
        dataset = dataset.assign_coords(
            {
                "motor_raw_feature": np.asarray(
                    motor_transform["raw_feature_names"],
                    dtype=str,
                )
            }
        )
        dataset["motor_standardization_mean"] = (
            "motor_raw_feature",
            np.asarray(motor_transform["mean"], dtype=float),
        )
        dataset["motor_standardization_std"] = (
            "motor_raw_feature",
            np.asarray(motor_transform["std"], dtype=float),
        )
        dataset["motor_standardization_std_raw"] = (
            "motor_raw_feature",
            np.asarray(motor_transform["std_raw"], dtype=float),
        )
        dataset["motor_standardization_constant_mask"] = (
            "motor_raw_feature",
            np.asarray(motor_transform["constant_mask"], dtype=np.int8),
        )
        dataset.attrs["motor_zscore_eps"] = float(motor_transform["eps"])
    else:
        continuous_names = list(motor_transform["continuous_feature_names"])
        dataset = dataset.assign_coords(
            {
                "motor_continuous_feature": np.asarray(continuous_names, dtype=str),
                "motor_spline_bound": np.asarray(["lower", "upper"], dtype=str),
            }
        )
        dataset["motor_spline_bounds"] = (
            ("motor_continuous_feature", "motor_spline_bound"),
            np.asarray(
                [
                    motor_transform["bounds_by_feature"][feature_name]
                    for feature_name in continuous_names
                ],
                dtype=float,
            ),
        )
        dataset.attrs["motor_spline_k"] = int(motor_transform["n_basis"])
        dataset.attrs["motor_spline_order"] = int(motor_transform["order"])

    return dataset


def plot_primary_delta_histograms(
    nested_dataset: "xr.Dataset",
    *,
    out_path: Path,
) -> Path:
    """Save the two-panel histogram for the primary held-out deltas."""
    import matplotlib.pyplot as plt

    panel_specs = (
        (
            "Motor + TP minus Motor",
            "dll_motor_tp_vs_motor_bits_per_spike",
            "#4C72B0",
        ),
        (
            "Motor + Place minus Motor",
            "dll_motor_place_vs_motor_bits_per_spike",
            "#55A868",
        ),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        constrained_layout=True,
        sharey=True,
    )
    for axis, (title, metric_name, color) in zip(axes, panel_specs):
        metric_values = np.asarray(
            nested_dataset["pooled_delta_bits_per_spike"]
            .sel(delta_metric=metric_name)
            .values,
            dtype=float,
        ).reshape(-1)
        finite_values = metric_values[np.isfinite(metric_values)]
        axis.axvline(0.0, color="0.2", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Held-out delta log-likelihood (bits/spike)")
        axis.set_ylabel("Fraction of units")
        if finite_values.size == 0:
            axis.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        bin_edges = build_histogram_bin_edges(finite_values)
        axis.hist(
            finite_values,
            bins=bin_edges,
            weights=np.full(finite_values.shape, 1.0 / finite_values.size),
            color=color,
            alpha=0.58,
            edgecolor="none",
            linewidth=0.0,
        )
        axis.text(
            0.98,
            0.98,
            (
                f"n={finite_values.size}\n"
                f"frac>0={np.mean(finite_values > 0.0):.3f}\n"
                f"mean={np.mean(finite_values):.4g}\n"
                f"median={np.median(finite_values):.4g}"
            ),
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
        )

    fig.suptitle(
        (
            f"{nested_dataset.attrs['animal_name']} {nested_dataset.attrs['date']} "
            f"{nested_dataset.attrs['region']} {nested_dataset.attrs['epoch']}"
        ),
        fontsize=12,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fit_motor_task_progression_place_epoch(
    *,
    spikes: Any,
    position_tsd: Any,
    body_position_tsd: Any,
    trajectory_intervals: dict[str, Any],
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    bin_size_s: float = 0.02,
    tp_spline_k: int = 25,
    tp_spline_order: int = 4,
    tp_bounds: tuple[float, float] = (0.0, 1.0),
    motor_spline_k: int = 5,
    motor_spline_order: int = 4,
    motor_feature_mode: Literal["zscore", "spline"] = "zscore",
    motor_zscore_eps: float = 1e-12,
    ridge: float = 1e-3,
    n_folds: int = 5,
    seed: int = 0,
    unit_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit motor, task-progression, and trajectory-specific place GLMs."""
    import pynapple as nap

    for trajectory_type in TRAJECTORY_TYPES:
        if trajectory_type not in trajectory_intervals:
            raise ValueError(
                f"trajectory_intervals is missing required key: {trajectory_type}"
            )
        if trajectory_type not in task_progression_by_trajectory:
            raise ValueError(
                f"task_progression_by_trajectory is missing required key: {trajectory_type}"
            )

    selected_spikes = spikes if unit_mask is None else spikes[unit_mask]
    if len(selected_spikes.keys()) == 0:
        raise ValueError("No units remain after applying the firing-rate threshold.")

    position_times = np.asarray(position_tsd.t, dtype=float)
    all_interval = nap.IntervalSet(
        start=float(position_times[0]),
        end=float(position_times[-1]),
        time_units="s",
    )
    spike_counts = selected_spikes.count(bin_size_s, ep=all_interval)
    unit_ids = np.asarray(spike_counts.columns)
    spike_count_array = np.asarray(spike_counts.d, dtype=float)
    n_time_bins, n_units = spike_count_array.shape

    in_movement = np.asarray(
        spike_counts.in_interval(movement_interval),
        dtype=bool,
    ).reshape(-1)
    in_trajectory = {
        trajectory_type: np.asarray(
            spike_counts.in_interval(trajectory_intervals[trajectory_type]),
            dtype=bool,
        ).reshape(-1)
        for trajectory_type in TRAJECTORY_TYPES
    }
    in_any_trajectory = np.zeros(n_time_bins, dtype=bool)
    overlap_count = np.zeros(n_time_bins, dtype=int)
    for trajectory_type in TRAJECTORY_TYPES:
        in_any_trajectory |= in_trajectory[trajectory_type]
        overlap_count += in_trajectory[trajectory_type].astype(int)

    keep_mask = in_movement & in_any_trajectory
    if not np.any(keep_mask):
        raise ValueError(
            "No bins remain after restricting to movement and trajectories."
        )
    if np.any(keep_mask & (overlap_count != 1)):
        raise ValueError(
            "Trajectory intervals appear to overlap within the kept bins. "
            "Each retained time bin must belong to exactly one trajectory."
        )

    trajectory_labels_full = np.full(n_time_bins, -1, dtype=int)
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        trajectory_labels_full[in_trajectory[trajectory_type]] = trajectory_index

    motor_covariates = compute_motor_covariates(
        position_xy=np.asarray(position_tsd.d, dtype=float),
        body_xy=np.asarray(body_position_tsd.d, dtype=float),
        position_timestamps=np.asarray(position_tsd.t, dtype=float),
        spike_counts=spike_counts,
    )

    tp_lower, tp_upper = map(float, tp_bounds)
    if not np.isfinite(tp_lower) or not np.isfinite(tp_upper) or tp_upper <= tp_lower:
        raise ValueError(f"Invalid task-progression bounds: {tp_bounds!r}")

    task_progression_full = np.full(n_time_bins, np.nan, dtype=float)
    for trajectory_type in TRAJECTORY_TYPES:
        interpolated_tp = task_progression_by_trajectory[trajectory_type].interpolate(
            spike_counts
        )
        interpolated_tp = np.asarray(interpolated_tp.to_numpy(), dtype=float).reshape(
            -1
        )
        trajectory_mask = in_trajectory[trajectory_type]
        if interpolated_tp.size != int(trajectory_mask.sum()):
            raise RuntimeError(
                f"{trajectory_type}: interpolated task progression size {interpolated_tp.size} "
                f"did not match the number of bins in the trajectory mask {int(trajectory_mask.sum())}."
            )
        task_progression_full[trajectory_mask] = interpolated_tp
    task_progression_full = np.clip(task_progression_full, tp_lower, tp_upper)

    finite_covariates = np.ones(n_time_bins, dtype=bool)
    for values in motor_covariates.values():
        finite_covariates &= np.isfinite(values)
    keep_mask &= finite_covariates & np.isfinite(task_progression_full)
    if not np.any(keep_mask):
        raise ValueError(
            "No bins remain after dropping non-finite motor or TP samples."
        )

    response = spike_count_array[keep_mask]
    trajectory_labels = trajectory_labels_full[keep_mask]
    task_progression = task_progression_full[keep_mask]
    masked_covariates = {
        name: np.asarray(values[keep_mask], dtype=float)
        for name, values in motor_covariates.items()
    }

    motor_feature_names: list[str] = []
    motor_standardization: dict[str, Any] | None = None
    if motor_feature_mode == "spline":
        motor_blocks: list[np.ndarray] = []
        for covariate_name in MOTOR_CONTINUOUS_FEATURE_NAMES:
            features = bspline_features(
                masked_covariates[covariate_name],
                n_basis=motor_spline_k,
                order=motor_spline_order,
            )
            motor_blocks.append(features)
            motor_feature_names.extend(
                f"{covariate_name}_bs{basis_index}"
                for basis_index in range(features.shape[1])
            )
        motor_blocks.append(masked_covariates["sin_hd"][:, None])
        motor_blocks.append(masked_covariates["cos_hd"][:, None])
        motor_feature_names.extend(["sin_hd", "cos_hd"])
        motor_design = np.concatenate(motor_blocks, axis=1)
    elif motor_feature_mode == "zscore":
        motor_raw_names = list(MOTOR_RAW_FEATURE_NAMES)
        motor_raw = np.column_stack(
            [masked_covariates[name] for name in motor_raw_names]
        )
        mean = motor_raw.mean(axis=0)
        std_raw = motor_raw.std(axis=0)
        std = std_raw.copy()
        constant_mask = (~np.isfinite(std)) | (std < float(motor_zscore_eps))
        std[constant_mask] = 1.0
        motor_design = (motor_raw - mean) / std
        if np.any(constant_mask):
            motor_design[:, constant_mask] = 0.0
        motor_feature_names = [f"{name}_z" for name in motor_raw_names]
        motor_standardization = {
            "raw_feature_names": motor_raw_names,
            "mean": mean,
            "std": std,
            "std_raw": std_raw,
            "eps": float(motor_zscore_eps),
            "constant_mask": constant_mask,
        }
    else:
        raise ValueError(
            f"motor_feature_mode must be 'zscore' or 'spline', got {motor_feature_mode!r}"
        )

    trajectory_to_index = {
        trajectory_type: index for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }
    trajectory_index_to_group = np.full(len(TRAJECTORY_TYPES), -1, dtype=int)
    for group_index, (_, grouped_trajectories) in enumerate(TP_GROUPS):
        for trajectory_type in grouped_trajectories:
            trajectory_index_to_group[trajectory_to_index[trajectory_type]] = (
                group_index
            )
    if np.any(trajectory_index_to_group < 0):
        missing = [
            TRAJECTORY_TYPES[index]
            for index in np.flatnonzero(trajectory_index_to_group < 0)
        ]
        raise ValueError(
            f"Some trajectories were not assigned to a TP group: {missing!r}"
        )

    group_labels = trajectory_index_to_group[trajectory_labels]
    trajectory_group_design = (group_labels == 1).astype(float)[:, None]
    trajectory_feature_names = [f"is_{TP_GROUPS[1][0]}"]

    tp_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(tp_lower, tp_upper),
    )
    tp_basis_features = np.asarray(
        tp_basis.compute_features(task_progression), dtype=float
    )
    if not np.all(np.isfinite(tp_basis_features)):
        raise ValueError("Encountered non-finite task-progression spline features.")

    tp_feature_blocks: list[np.ndarray] = []
    tp_feature_names: list[str] = []
    n_tp_basis = tp_basis_features.shape[1]
    for group_index, (group_name, grouped_trajectories) in enumerate(TP_GROUPS):
        grouped_indices = [
            trajectory_to_index[trajectory_type]
            for trajectory_type in grouped_trajectories
        ]
        gate = np.isin(trajectory_labels, grouped_indices).astype(float)[:, None]
        tp_feature_blocks.append(tp_basis_features * gate)
        tp_feature_names.extend(
            f"tp_{group_name}_bs{basis_index}" for basis_index in range(n_tp_basis)
        )
    tp_design = np.concatenate(tp_feature_blocks, axis=1)
    if np.allclose(tp_design, 0.0):
        raise RuntimeError("Task-progression design matrix is all zeros.")

    place_basis = BSplineEval(
        n_basis_funcs=int(tp_spline_k),
        order=int(tp_spline_order),
        bounds=(tp_lower, tp_upper),
    )
    place_basis_features = np.asarray(
        place_basis.compute_features(task_progression), dtype=float
    )
    if not np.all(np.isfinite(place_basis_features)):
        raise ValueError("Encountered non-finite place spline features.")

    place_feature_blocks: list[np.ndarray] = []
    place_feature_names: list[str] = []
    n_place_basis = place_basis_features.shape[1]
    for trajectory_type in TRAJECTORY_TYPES:
        gate = (trajectory_labels == trajectory_to_index[trajectory_type]).astype(float)[
            :, None
        ]
        place_feature_blocks.append(place_basis_features * gate)
        place_feature_names.extend(
            f"place_{trajectory_type}_bs{basis_index}"
            for basis_index in range(n_place_basis)
        )
    place_design = np.concatenate(place_feature_blocks, axis=1)
    if np.allclose(place_design, 0.0):
        raise RuntimeError("Place design matrix is all zeros.")

    model_specs = {
        "motor": {
            "design": np.concatenate([motor_design, trajectory_group_design], axis=1),
            "blocks": {
                "motor": slice(0, motor_design.shape[1]),
                "traj": slice(
                    motor_design.shape[1],
                    motor_design.shape[1] + trajectory_group_design.shape[1],
                ),
            },
        },
        "motor_tp": {
            "design": np.concatenate(
                [motor_design, trajectory_group_design, tp_design],
                axis=1,
            ),
            "blocks": {
                "motor": slice(0, motor_design.shape[1]),
                "traj": slice(
                    motor_design.shape[1],
                    motor_design.shape[1] + trajectory_group_design.shape[1],
                ),
                "tp": slice(
                    motor_design.shape[1] + trajectory_group_design.shape[1],
                    motor_design.shape[1]
                    + trajectory_group_design.shape[1]
                    + tp_design.shape[1],
                ),
            },
        },
        "tp_only": {
            "design": np.concatenate([trajectory_group_design, tp_design], axis=1),
            "blocks": {
                "traj": slice(0, trajectory_group_design.shape[1]),
                "tp": slice(
                    trajectory_group_design.shape[1],
                    trajectory_group_design.shape[1] + tp_design.shape[1],
                ),
            },
        },
        "motor_place": {
            "design": np.concatenate([motor_design, place_design], axis=1),
            "blocks": {
                "motor": slice(0, motor_design.shape[1]),
                "place": slice(
                    motor_design.shape[1],
                    motor_design.shape[1] + place_design.shape[1],
                ),
            },
        },
        "place_only": {
            "design": place_design,
            "blocks": {
                "place": slice(0, place_design.shape[1]),
            },
        },
    }

    for matrix_name, matrix in (
        ("motor_design", motor_design),
        ("trajectory_group_design", trajectory_group_design),
        ("tp_design", tp_design),
        ("place_design", place_design),
    ):
        if not np.all(np.isfinite(matrix)):
            bad_row, bad_col = np.argwhere(~np.isfinite(matrix))[0]
            raise ValueError(
                f"{matrix_name} contains non-finite values at "
                f"row {int(bad_row)}, column {int(bad_col)}."
            )

    folds = stratified_contiguous_folds(trajectory_labels, n_folds=n_folds, seed=seed)
    fold_test_counts = np.zeros((n_folds, len(TRAJECTORY_TYPES)), dtype=int)
    for fold_index, (_, test_indices) in enumerate(folds):
        for trajectory_index in range(len(TRAJECTORY_TYPES)):
            fold_test_counts[fold_index, trajectory_index] = int(
                np.sum(trajectory_labels[test_indices] == trajectory_index)
            )
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        if np.min(fold_test_counts[:, trajectory_index]) == 0:
            warnings.warn(
                f"Trajectory {trajectory_type!r} has no test bins in at least one CV fold.",
                stacklevel=2,
            )

    aggregate_sample_scores = lambda array: jnp.sum(array, axis=0)
    ll_sum_by_model = {
        model_name: np.zeros(n_units, dtype=float) for model_name in model_specs
    }
    spike_sum = np.zeros(n_units, dtype=float)
    ll_sum_by_model_by_trajectory = {
        model_name: {
            trajectory_type: np.zeros(n_units, dtype=float)
            for trajectory_type in TRAJECTORY_TYPES
        }
        for model_name in model_specs
    }
    spike_sum_by_trajectory = {
        trajectory_type: np.zeros(n_units, dtype=float)
        for trajectory_type in TRAJECTORY_TYPES
    }

    for train_indices, test_indices in folds:
        fold_models = {}
        for model_name, model_spec in model_specs.items():
            model = PopulationGLM(
                "Poisson",
                regularizer="Ridge",
                regularizer_strength=ridge,
            )
            model.fit(model_spec["design"][train_indices], response[train_indices])
            fold_models[model_name] = model

        fold_spike_sum = np.asarray(response[test_indices].sum(axis=0), dtype=float)

        for model_name, model_spec in model_specs.items():
            ll_sum_by_model[model_name] += np.asarray(
                fold_models[model_name].score(
                    model_spec["design"][test_indices],
                    response[test_indices],
                    score_type="log-likelihood",
                    aggregate_sample_scores=aggregate_sample_scores,
                )
            )
        spike_sum += fold_spike_sum

        for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
            trajectory_test_indices = test_indices[
                trajectory_labels[test_indices] == trajectory_index
            ]
            if trajectory_test_indices.size == 0:
                continue
            for model_name, model_spec in model_specs.items():
                ll_sum_by_model_by_trajectory[model_name][trajectory_type] += np.asarray(
                    fold_models[model_name].score(
                        model_spec["design"][trajectory_test_indices],
                        response[trajectory_test_indices],
                        score_type="log-likelihood",
                        aggregate_sample_scores=aggregate_sample_scores,
                    )
                )
            spike_sum_by_trajectory[trajectory_type] += np.asarray(
                response[trajectory_test_indices].sum(axis=0),
                dtype=float,
            )

    def to_per_spike(ll_values: np.ndarray, spike_values: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(spike_values > 0, ll_values / spike_values, np.nan)

    ll_per_spike_by_model = {
        model_name: to_per_spike(ll_values, spike_sum)
        for model_name, ll_values in ll_sum_by_model.items()
    }
    cv_pooled = build_cv_metric_dict(ll_per_spike_by_model, spike_sum)

    cv_by_trajectory: dict[str, dict[str, np.ndarray]] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        trajectory_spike_sum = spike_sum_by_trajectory[trajectory_type]
        ll_per_spike_trajectory = {
            model_name: to_per_spike(
                ll_sum_by_model_by_trajectory[model_name][trajectory_type],
                trajectory_spike_sum,
            )
            for model_name in model_specs
        }
        cv_by_trajectory[trajectory_type] = build_cv_metric_dict(
            ll_per_spike_trajectory,
            trajectory_spike_sum,
        )

    fitted_models_all = {}
    for model_name, model_spec in model_specs.items():
        model = PopulationGLM(
            "Poisson",
            regularizer="Ridge",
            regularizer_strength=ridge,
        )
        model.fit(model_spec["design"], response)
        fitted_models_all[model_name] = model

    def extract_coefficients(model: PopulationGLM, n_features: int) -> np.ndarray:
        coefficients = np.asarray(model.coef_)
        if coefficients.shape[0] != n_features:
            coefficients = coefficients.T
        return coefficients

    model_coefficients_all = {
        model_name: extract_coefficients(
            fitted_models_all[model_name],
            model_spec["design"].shape[1],
        )
        for model_name, model_spec in model_specs.items()
    }

    motor_tp_coefficients_by_group = {}
    tp_only_coefficients_by_group = {}
    for group_index, (group_name, _) in enumerate(TP_GROUPS):
        start = group_index * n_tp_basis
        stop = (group_index + 1) * n_tp_basis
        motor_tp_coefficients_by_group[group_name] = model_coefficients_all["motor_tp"][
            model_specs["motor_tp"]["blocks"]["tp"].start
            + start : model_specs["motor_tp"]["blocks"]["tp"].start
            + stop,
            :,
        ]
        tp_only_coefficients_by_group[group_name] = model_coefficients_all["tp_only"][
            model_specs["tp_only"]["blocks"]["tp"].start
            + start : model_specs["tp_only"]["blocks"]["tp"].start
            + stop,
            :,
        ]

    motor_place_coefficients_by_trajectory = {}
    place_only_coefficients_by_trajectory = {}
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        start = trajectory_index * n_place_basis
        stop = (trajectory_index + 1) * n_place_basis
        motor_place_coefficients_by_trajectory[trajectory_type] = (
            model_coefficients_all["motor_place"][
                model_specs["motor_place"]["blocks"]["place"].start
                + start : model_specs["motor_place"]["blocks"]["place"].start
                + stop,
                :,
            ]
        )
        place_only_coefficients_by_trajectory[trajectory_type] = (
            model_coefficients_all["place_only"][
                model_specs["place_only"]["blocks"]["place"].start
                + start : model_specs["place_only"]["blocks"]["place"].start
                + stop,
                :,
            ]
        )

    motor_mean = (
        np.zeros(motor_design.shape[1], dtype=float)
        if motor_feature_mode == "zscore"
        else motor_design.mean(axis=0)
    )
    tp_rate_curves_hz = {}
    place_rate_curves_hz = {}
    tp_grid = np.linspace(tp_lower, tp_upper, 200)
    place_grid = np.linspace(tp_lower, tp_upper, 200)
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        tp_grid_features = np.asarray(tp_basis.compute_features(tp_grid), dtype=float)
        group_index = trajectory_index_to_group[trajectory_index]
        trajectory_dummy = np.zeros(
            (tp_grid.size, trajectory_group_design.shape[1]), dtype=float
        )
        if group_index == 1:
            trajectory_dummy[:, 0] = 1.0
        tp_block = np.zeros((tp_grid.size, tp_design.shape[1]), dtype=float)
        start = group_index * n_tp_basis
        stop = (group_index + 1) * n_tp_basis
        tp_block[:, start:stop] = tp_grid_features
        grid_design = np.concatenate(
            [
                np.repeat(motor_mean[None, :], tp_grid.size, axis=0),
                trajectory_dummy,
                tp_block,
            ],
            axis=1,
        )
        eta = (
            np.asarray(fitted_models_all["motor_tp"].intercept_).reshape(1, -1)
            + grid_design @ model_coefficients_all["motor_tp"]
        )
        tp_rate_curves_hz[trajectory_type] = {
            "tp_grid": tp_grid,
            "rate_hz": np.exp(eta) / float(bin_size_s),
        }

        place_grid_features = np.asarray(
            place_basis.compute_features(place_grid), dtype=float
        )
        place_block = np.zeros((place_grid.size, place_design.shape[1]), dtype=float)
        place_start = trajectory_index * n_place_basis
        place_stop = (trajectory_index + 1) * n_place_basis
        place_block[:, place_start:place_stop] = place_grid_features
        place_grid_design = np.concatenate(
            [
                np.repeat(motor_mean[None, :], place_grid.size, axis=0),
                place_block,
            ],
            axis=1,
        )
        place_eta = (
            np.asarray(fitted_models_all["motor_place"].intercept_).reshape(1, -1)
            + place_grid_design @ model_coefficients_all["motor_place"]
        )
        place_rate_curves_hz[trajectory_type] = {
            "place_grid": place_grid,
            "rate_hz": np.exp(place_eta) / float(bin_size_s),
        }

    return {
        "unit_ids": unit_ids,
        "bin_size_s": float(bin_size_s),
        "motor_feature_mode": motor_feature_mode,
        "motor_standardization": motor_standardization,
        "feature_names_motor": motor_feature_names,
        "feature_names_traj": trajectory_feature_names,
        "feature_names_tp": tp_feature_names,
        "feature_names_place": place_feature_names,
        "tp_basis": {
            "n_splines": int(tp_spline_k),
            "order": int(tp_spline_order),
            "bounds": (tp_lower, tp_upper),
        },
        "place_basis": {
            "n_splines": int(tp_spline_k),
            "order": int(tp_spline_order),
            "bounds": (tp_lower, tp_upper),
        },
        "cv": {
            "pooled": cv_pooled,
            "by_traj": cv_by_trajectory,
            "fold_test_counts": fold_test_counts,
        },
        "coef": {
            "motor": {
                "intercept": np.asarray(fitted_models_all["motor"].intercept_).reshape(
                    -1
                ),
                "coef_motor": model_coefficients_all["motor"][
                    model_specs["motor"]["blocks"]["motor"],
                    :,
                ],
                "coef_traj": model_coefficients_all["motor"][
                    model_specs["motor"]["blocks"]["traj"],
                    :,
                ],
            },
            "motor_tp": {
                "intercept": np.asarray(
                    fitted_models_all["motor_tp"].intercept_
                ).reshape(-1),
                "coef_motor": model_coefficients_all["motor_tp"][
                    model_specs["motor_tp"]["blocks"]["motor"],
                    :,
                ],
                "coef_traj": model_coefficients_all["motor_tp"][
                    model_specs["motor_tp"]["blocks"]["traj"],
                    :,
                ],
                "coef_tp_by_group": motor_tp_coefficients_by_group,
            },
            "tp_only": {
                "intercept": np.asarray(
                    fitted_models_all["tp_only"].intercept_
                ).reshape(-1),
                "coef_traj": model_coefficients_all["tp_only"][
                    model_specs["tp_only"]["blocks"]["traj"],
                    :,
                ],
                "coef_tp_by_group": tp_only_coefficients_by_group,
            },
            "motor_place": {
                "intercept": np.asarray(
                    fitted_models_all["motor_place"].intercept_
                ).reshape(-1),
                "coef_motor": model_coefficients_all["motor_place"][
                    model_specs["motor_place"]["blocks"]["motor"],
                    :,
                ],
                "coef_place_by_trajectory": motor_place_coefficients_by_trajectory,
            },
            "place_only": {
                "intercept": np.asarray(
                    fitted_models_all["place_only"].intercept_
                ).reshape(-1),
                "coef_place_by_trajectory": place_only_coefficients_by_trajectory,
            },
        },
        "tp_rate_curves_hz": tp_rate_curves_hz,
        "place_rate_curves_hz": place_rate_curves_hz,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the task-progression TP/place GLM script."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit nested lap-CV motor, task-progression, and trajectory-specific "
            "place GLMs for one task-progression session"
        )
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
        "--region",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Regions to fit. Default: {' '.join(REGIONS)}",
    )
    parser.add_argument(
        "--epochs",
        "--epoch",
        nargs="+",
        help="Specific run epoch labels to fit. Defaults to all run epochs.",
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
        "--v1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["v1"],
        help=f"Minimum movement firing rate for V1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['v1']}",
    )
    parser.add_argument(
        "--ca1-min-fr-hz",
        type=float,
        default=DEFAULT_REGION_FR_THRESHOLDS["ca1"],
        help=f"Minimum movement firing rate for CA1 units. Default: {DEFAULT_REGION_FR_THRESHOLDS['ca1']}",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=0.05,
        help="Spike-count bin size in seconds. Default: 0.05",
    )
    parser.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=list(DEFAULT_RIDGES),
        help=(
            "Ridge penalty candidates for per-model nested-CV selection. "
            "Default: " + " ".join(f"{value:g}" for value in DEFAULT_RIDGES)
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of outer lap-level CV folds. Default: 5",
    )
    parser.add_argument(
        "--inner-n-folds",
        type=int,
        default=DEFAULT_INNER_N_FOLDS,
        help=f"Number of inner lap-level CV folds for ridge selection. Default: {DEFAULT_INNER_N_FOLDS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when assigning lap chunks to folds. Default: 0",
    )
    parser.add_argument(
        "--motor-feature-mode",
        choices=("zscore", "spline"),
        default="zscore",
        help="Whether to use z-scored instantaneous motor variables or spline-expanded covariates.",
    )
    parser.add_argument(
        "--motor-zscore-eps",
        type=float,
        default=1e-12,
        help="Minimum standard deviation when z-scoring motor covariates. Default: 1e-12",
    )
    parser.add_argument(
        "--motor-spline-k",
        type=int,
        default=5,
        help="Number of spline basis functions per continuous motor covariate when using spline mode. Default: 5",
    )
    parser.add_argument(
        "--motor-spline-order",
        type=int,
        default=4,
        help="Spline order for motor covariates in spline mode. Default: 4",
    )
    parser.add_argument(
        "--tp-spline-k",
        type=int,
        default=25,
        help="Number of spline basis functions for both task progression and place. Default: 25",
    )
    parser.add_argument(
        "--tp-spline-order",
        type=int,
        default=4,
        help="Spline order for both task progression and place. Default: 4",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing config-token output files.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the nested lap-CV task-progression TP/place GLM workflow."""
    args = parse_arguments()
    ridge_values = [float(value) for value in dict.fromkeys(args.ridges)]
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")
    if args.inner_n_folds < 2:
        raise ValueError("--inner-n-folds must be at least 2.")
    if args.tp_spline_k <= 0:
        raise ValueError("--tp-spline-k must be positive.")
    if args.motor_spline_k <= 0:
        raise ValueError("--motor-spline-k must be positive.")
    if any(value < 0 for value in ridge_values):
        raise ValueError("--ridges must be non-negative.")

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    config_token = build_config_token(
        bin_size_s=args.bin_size_s,
        tp_spline_k=args.tp_spline_k,
        motor_feature_mode=args.motor_feature_mode,
        n_folds=args.n_folds,
        inner_n_folds=args.inner_n_folds,
        ridge_values=ridge_values,
    )
    print(
        "Starting nested lap-CV motor/task-progression/place GLM fits "
        f"for {args.animal_name} {args.date}"
    )
    print(f"Analysis path: {analysis_path}")
    print(
        "Fit settings: "
        f"regions={list(args.regions)}, "
        f"bin_size_s={args.bin_size_s:.3f}, "
        f"motor_feature_mode={args.motor_feature_mode}, "
        f"n_folds={args.n_folds}, "
        f"inner_n_folds={args.inner_n_folds}, "
        "ridges="
        + ",".join(f"{value:g}" for value in ridge_values)
    )
    print(f"Output config token: {config_token}")
    print("Primary deltas: " + ", ".join(PRIMARY_DELTA_METRIC_NAMES))

    print("Checking which run epochs have usable head/body position data...")
    selected_epochs, skipped_position_epochs = select_epochs_with_usable_position_data(
        analysis_path,
        args.epochs,
    )
    if skipped_position_epochs:
        print("Skipping epochs with unusable position data:")
        for skipped_epoch in skipped_position_epochs:
            print(f"  {skipped_epoch['epoch']}: {skipped_epoch['reason']}")
    if not selected_epochs:
        raise ValueError("No requested run epochs have usable head/body position data.")
    print(
        f"Selected {len(selected_epochs)} run epoch(s) with usable position data: "
        f"{selected_epochs}"
    )

    print("Preparing shared session inputs...")
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        selected_run_epochs=selected_epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    print("Computing movement firing rates by region and epoch...")
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    data_dir = (
        get_task_progression_output_dir(analysis_path, Path(__file__).stem)
        / "nested_lap_cv"
    )
    fig_dir = (
        get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
        / "nested_lap_cv"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"NetCDF output directory: {data_dir}")
    print(f"Figure output directory: {fig_dir}")

    region_thresholds = {
        "v1": float(args.v1_min_fr_hz),
        "ca1": float(args.ca1_min_fr_hz),
    }
    saved_datasets: list[Path] = []
    saved_figures: list[Path] = []
    skipped_fits: list[dict[str, Any]] = []
    total_fits = len(args.regions) * len(selected_epochs)
    print(
        f"Beginning {total_fits} fit(s) across "
        f"{len(args.regions)} region(s) and {len(selected_epochs)} epoch(s)."
    )
    fit_index = 0

    for region in args.regions:
        for epoch in selected_epochs:
            fit_index += 1
            epoch_unit_rates = np.asarray(
                movement_firing_rates[region][epoch], dtype=float
            )
            full_refit_unit_mask = get_unit_mask(
                epoch_unit_rates,
                threshold_hz=region_thresholds[region],
            )
            n_units_kept = int(np.sum(full_refit_unit_mask))
            n_units_total = int(full_refit_unit_mask.size)
            print(
                f"[{fit_index}/{total_fits}] "
                f"Region={region}, epoch={epoch}: "
                f"{n_units_kept}/{n_units_total} units passed "
                f"the movement firing-rate threshold "
                f"({region_thresholds[region]:.3f} Hz)."
            )
            nested_path = data_dir / (
                f"{region}_{epoch}_nested_lapcv_{config_token}.nc"
            )
            full_refit_path = data_dir / (
                f"{region}_{epoch}_full_refit_{config_token}.nc"
            )
            figure_path = fig_dir / (
                f"{region}_{epoch}_primary_delta_hist_{config_token}.png"
            )
            for output_path in (nested_path, full_refit_path, figure_path):
                ensure_can_write(output_path, overwrite=args.overwrite)

            feasible, feasibility_reasons, lap_summary = summarize_lap_cv_feasibility(
                session["trajectory_intervals"][epoch],
                n_folds=args.n_folds,
                inner_n_folds=args.inner_n_folds,
            )
            print("  Lap CV feasibility:")
            for trajectory_type in TRAJECTORY_TYPES:
                summary = lap_summary[trajectory_type]
                print(
                    f"    {trajectory_type}: {summary['n_laps']} laps; "
                    f"minimum outer-train laps={summary['min_outer_train_laps']}"
                )
            if not feasible:
                print("  Skipping because nested lap-CV is not feasible:")
                for reason in feasibility_reasons:
                    print(f"    {reason}")
                skipped_fits.append(
                    {
                        "region": region,
                        "epoch": epoch,
                        "reason": "nested lap-CV infeasible",
                        "details": feasibility_reasons,
                    }
                )
                continue

            if not np.any(full_refit_unit_mask):
                print(
                    f"  Skipping {region} {epoch}: "
                    "no units passed the firing-rate threshold."
                )
                skipped_fits.append(
                    {
                        "region": region,
                        "epoch": epoch,
                        "reason": "no units passed the firing-rate threshold",
                        "threshold_hz": region_thresholds[region],
                    }
                )
                continue

            position_tsd = build_position_tsdframe(
                session["position_by_epoch"][epoch],
                session["timestamps_position"][epoch],
                args.position_offset,
            )
            body_position_tsd = build_position_tsdframe(
                session["body_position_by_epoch"][epoch],
                session["timestamps_position"][epoch],
                args.position_offset,
            )
            fit_parameters_common = {
                "cuda_visible_devices": args.cuda_visible_devices,
                "position_offset": args.position_offset,
                "speed_threshold_cm_s": args.speed_threshold_cm_s,
                "speed_sigma_s": DEFAULT_SPEED_SIGMA_S,
                "bin_size_s": args.bin_size_s,
                "ridges": ridge_values,
                "n_folds": args.n_folds,
                "inner_n_folds": args.inner_n_folds,
                "seed": args.seed,
                "motor_feature_mode": args.motor_feature_mode,
                "motor_zscore_eps": args.motor_zscore_eps,
                "motor_spline_k": args.motor_spline_k,
                "motor_spline_order": args.motor_spline_order,
                "tp_spline_k": args.tp_spline_k,
                "tp_spline_order": args.tp_spline_order,
                "model_definitions": MODEL_DEFINITIONS,
                "primary_delta_metric_names": PRIMARY_DELTA_METRIC_NAMES,
                "config_token": config_token,
            }

            print(f"  Preparing binned epoch data for {region} {epoch}...")
            try:
                epoch_data = prepare_motor_epoch_data(
                    spikes=session["spikes_by_region"][region],
                    position_tsd=position_tsd,
                    body_position_tsd=body_position_tsd,
                    trajectory_intervals=session["trajectory_intervals"][epoch],
                    task_progression_by_trajectory=session[
                        "task_progression_by_trajectory"
                    ][epoch],
                    movement_interval=session["movement_by_run"][epoch],
                    bin_size_s=args.bin_size_s,
                )
                outer_folds = build_lap_cv_folds_for_epoch(
                    session["trajectory_intervals"][epoch],
                    n_folds=args.n_folds,
                    seed=args.seed,
                )
                print(f"  Running nested lap-CV for {region} {epoch}...")
                nested_result = run_nested_lap_cv(
                    epoch_data,
                    outer_folds,
                    ridge_values=ridge_values,
                    inner_n_folds=args.inner_n_folds,
                    seed=args.seed,
                    min_firing_rate_hz=region_thresholds[region],
                    tp_spline_k=args.tp_spline_k,
                    tp_spline_order=args.tp_spline_order,
                    motor_feature_mode=args.motor_feature_mode,
                    motor_zscore_eps=args.motor_zscore_eps,
                    motor_spline_k=args.motor_spline_k,
                    motor_spline_order=args.motor_spline_order,
                    print_prefix="    ",
                )
                nested_dataset = build_nested_cv_dataset(
                    nested_result,
                    outer_folds,
                    epoch_data,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    min_firing_rate_hz=region_thresholds[region],
                    sources={**session["sources"]},
                    fit_parameters=fit_parameters_common,
                )

                print("  Selecting full-data refit ridges by lap-level CV...")
                full_refit_folds = build_lap_cv_folds_for_epoch(
                    session["trajectory_intervals"][epoch],
                    n_folds=args.n_folds,
                    seed=args.seed,
                )
                full_ridge_cv = compute_ridge_cv_scores(
                    epoch_data,
                    full_refit_folds,
                    unit_mask=full_refit_unit_mask,
                    ridge_values=ridge_values,
                    tp_spline_k=args.tp_spline_k,
                    tp_spline_order=args.tp_spline_order,
                    motor_feature_mode=args.motor_feature_mode,
                    motor_zscore_eps=args.motor_zscore_eps,
                    motor_spline_k=args.motor_spline_k,
                    motor_spline_order=args.motor_spline_order,
                )
                full_ridge_by_model = {
                    model_name: float(
                        full_ridge_cv["selected_ridge"][model_index]
                    )
                    for model_index, model_name in enumerate(MODEL_NAMES)
                }
                print(
                    "  Full-refit selected ridges: "
                    + ", ".join(
                        f"{model_name}={full_ridge_by_model[model_name]:.3g}"
                        for model_name in MODEL_NAMES
                    )
                )
                full_fit = fit_full_refit_models(
                    epoch_data,
                    unit_mask=full_refit_unit_mask,
                    ridge_by_model=full_ridge_by_model,
                    tp_spline_k=args.tp_spline_k,
                    tp_spline_order=args.tp_spline_order,
                    motor_feature_mode=args.motor_feature_mode,
                    motor_zscore_eps=args.motor_zscore_eps,
                    motor_spline_k=args.motor_spline_k,
                    motor_spline_order=args.motor_spline_order,
                )
                full_refit_dataset = build_full_refit_dataset(
                    full_fit,
                    full_ridge_cv,
                    epoch_data,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    movement_firing_rates=epoch_unit_rates,
                    min_firing_rate_hz=region_thresholds[region],
                    sources={**session["sources"]},
                    fit_parameters=fit_parameters_common,
                )
            except Exception as exc:
                print(f"  Fit failed for {region} {epoch}: {exc}")
                skipped_fits.append(
                    {
                        "region": region,
                        "epoch": epoch,
                        "reason": "fit failed",
                        "error": str(exc),
                    }
                )
                continue

            nested_dataset.to_netcdf(nested_path)
            saved_datasets.append(nested_path)
            full_refit_dataset.to_netcdf(full_refit_path)
            saved_datasets.append(full_refit_path)
            plot_primary_delta_histograms(nested_dataset, out_path=figure_path)
            saved_figures.append(figure_path)
            print(
                f"  Saved nested CV to {nested_path.name}, full refit to "
                f"{full_refit_path.name}, and figure to {figure_path.name}."
            )

    print("Writing run metadata log...")
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.motor",
        parameters={
            "cuda_visible_devices": args.cuda_visible_devices,
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": args.regions,
            "epochs": selected_epochs,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "speed_sigma_s": DEFAULT_SPEED_SIGMA_S,
            "region_thresholds_hz": region_thresholds,
            "bin_size_s": args.bin_size_s,
            "ridges": ridge_values,
            "n_folds": args.n_folds,
            "inner_n_folds": args.inner_n_folds,
            "seed": args.seed,
            "motor_feature_mode": args.motor_feature_mode,
            "motor_zscore_eps": args.motor_zscore_eps,
            "motor_spline_k": args.motor_spline_k,
            "motor_spline_order": args.motor_spline_order,
            "tp_spline_k": args.tp_spline_k,
            "tp_spline_order": args.tp_spline_order,
            "config_token": config_token,
            "workflow": "nested_lap_cv_strict_motor_baseline",
        },
        outputs={
            "sources": {
                **session["sources"],
            },
            "saved_datasets": saved_datasets,
            "saved_figures": saved_figures,
            "skipped_position_epochs": skipped_position_epochs,
            "skipped_fits": skipped_fits,
        },
    )

    if skipped_fits:
        print("Skipped fit summary:")
        for skipped_fit in skipped_fits:
            details = (
                f" ({skipped_fit['error']})"
                if "error" in skipped_fit
                else ""
            )
            print(
                f"  {skipped_fit['region']} {skipped_fit['epoch']}: "
                f"{skipped_fit['reason']}{details}"
            )
    if saved_datasets:
        print(f"Saved {len(saved_datasets)} NetCDF fit dataset(s) to {data_dir}")
    if saved_figures:
        print(f"Saved {len(saved_figures)} histogram figure(s) to {fig_dir}")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
