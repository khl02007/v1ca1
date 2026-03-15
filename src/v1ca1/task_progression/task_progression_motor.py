from __future__ import annotations

"""Fit motor, task-progression, and place GLMs for one task-progression session.

This script uses the shared session loaders under
`v1ca1.task_progression._session`, exposes a CLI for session and fit settings,
and writes one labeled NetCDF-backed `xarray.Dataset` per fit under the
analysis directory.

For each selected region and run epoch, the script compares five Poisson GLMs:

- motor only
- motor + trajectory-gated task progression
- task progression only
- motor + trajectory-specific place
- trajectory-specific place only

Motor covariates can be represented either as instantaneous z-scored values or
as spline-expanded features. Task progression uses one tuning curve per same-
turn trajectory pair, whereas place uses one tuning curve per trajectory type.
Cross-validated full Poisson log-likelihood is reported in bits/spike, both
pooled and per trajectory, and the saved dataset bundles those scores with fit
coefficients, TP/place rate curves, and fit metadata.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
import numpy as np
import position_tools as pt
from nemos.basis import BSplineEval
from nemos.glm import PopulationGLM

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    coerce_position_array,
    compute_movement_firing_rates,
    get_analysis_path,
    prepare_task_progression_session,
)

if TYPE_CHECKING:
    import xarray as xr


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
TP_GROUPS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("LC_CR", ("left_to_center", "center_to_right")),
    ("RC_CL", ("right_to_center", "center_to_left")),
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


def load_body_position_data(
    analysis_path: Path,
    epoch_tags: list[str],
) -> dict[str, np.ndarray]:
    """Load per-epoch body position arrays from `body_position.pkl`."""
    body_position_path = analysis_path / "body_position.pkl"
    if not body_position_path.exists():
        raise FileNotFoundError(f"Body position file not found: {body_position_path}")

    with open(body_position_path, "rb") as file:
        body_position_dict = pickle.load(file)

    normalized_body_position = {
        str(epoch): coerce_position_array(value)
        for epoch, value in body_position_dict.items()
    }
    missing_epochs = [
        epoch for epoch in epoch_tags if epoch not in normalized_body_position
    ]
    extra_epochs = sorted(set(normalized_body_position) - set(epoch_tags))
    if missing_epochs or extra_epochs:
        raise ValueError(
            "Body position epochs do not match saved session epochs. "
            f"Missing body position epochs: {missing_epochs!r}; extra body position epochs: {extra_epochs!r}"
        )
    return {epoch: normalized_body_position[epoch] for epoch in epoch_tags}


def select_run_epochs(
    run_epochs: list[str], requested_epochs: list[str] | None
) -> list[str]:
    """Return the requested run epochs, defaulting to all available run epochs."""
    if not requested_epochs:
        return run_epochs

    missing_epochs = [epoch for epoch in requested_epochs if epoch not in run_epochs]
    if missing_epochs:
        raise ValueError(
            f"Requested epochs were not found in available run epochs {run_epochs!r}: {missing_epochs!r}"
        )
    return requested_epochs


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
    """Compute motor covariates and interpolate them onto spike-count bins."""
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

    return {
        "speed": speed_bin,
        "accel": acceleration,
        "sin_hd": np.sin(head_direction_bin),
        "cos_hd": np.cos(head_direction_bin),
        "hd_vel": head_direction_velocity,
        "abs_hd_vel": np.abs(head_direction_velocity),
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
        for covariate_name in ("speed", "accel", "hd_vel", "abs_hd_vel"):
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
        motor_raw_names = ["speed", "accel", "hd_vel", "abs_hd_vel", "sin_hd", "cos_hd"]
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
            "Fit motor, task-progression, and trajectory-specific place GLMs "
            "for one task-progression session"
        )
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
        default=0.02,
        help="Spike-count bin size in seconds. Default: 0.02",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-3,
        help="Ridge penalty strength for PopulationGLM. Default: 1e-3",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of stratified contiguous CV folds. Default: 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when assigning contiguous chunks to folds. Default: 0",
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
    return parser.parse_args()


def main() -> None:
    """Run the modernized task-progression TP/place GLM workflow."""
    args = parse_arguments()
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    body_position_by_epoch = load_body_position_data(
        analysis_path,
        session["epoch_tags"],
    )
    selected_epochs = select_run_epochs(session["run_epochs"], args.epochs)
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    data_dir = analysis_path / "task_progression_motor"
    data_dir.mkdir(parents=True, exist_ok=True)

    region_thresholds = {
        "v1": float(args.v1_min_fr_hz),
        "ca1": float(args.ca1_min_fr_hz),
    }
    saved_datasets: list[Path] = []
    skipped_fits: list[dict[str, Any]] = []

    for region in args.regions:
        for epoch in selected_epochs:
            epoch_unit_rates = np.asarray(
                movement_firing_rates[region][epoch], dtype=float
            )
            unit_mask = get_unit_mask(
                epoch_unit_rates,
                threshold_hz=region_thresholds[region],
            )
            if not np.any(unit_mask):
                skipped_fits.append(
                    {
                        "region": region,
                        "epoch": epoch,
                        "reason": "no units passed the firing-rate threshold",
                        "threshold_hz": region_thresholds[region],
                    }
                )
                continue

            selected_unit_rates = epoch_unit_rates[unit_mask]
            position_tsd = build_position_tsdframe(
                session["position_by_epoch"][epoch],
                session["timestamps_position"][epoch],
                args.position_offset,
            )
            body_position_tsd = build_position_tsdframe(
                body_position_by_epoch[epoch],
                session["timestamps_position"][epoch],
                args.position_offset,
            )
            try:
                fit_result = fit_motor_task_progression_place_epoch(
                    spikes=session["spikes_by_region"][region],
                    position_tsd=position_tsd,
                    body_position_tsd=body_position_tsd,
                    trajectory_intervals=session["trajectory_intervals"][epoch],
                    task_progression_by_trajectory=session[
                        "task_progression_by_trajectory"
                    ][epoch],
                    movement_interval=session["movement_by_run"][epoch],
                    bin_size_s=args.bin_size_s,
                    tp_spline_k=args.tp_spline_k,
                    tp_spline_order=args.tp_spline_order,
                    motor_spline_k=args.motor_spline_k,
                    motor_spline_order=args.motor_spline_order,
                    motor_feature_mode=args.motor_feature_mode,
                    motor_zscore_eps=args.motor_zscore_eps,
                    ridge=args.ridge,
                    n_folds=args.n_folds,
                    seed=args.seed,
                    unit_mask=unit_mask,
                )
            except Exception as exc:
                skipped_fits.append(
                    {
                        "region": region,
                        "epoch": epoch,
                        "reason": "fit failed",
                        "error": str(exc),
                    }
                )
                continue

            result_stem = (
                f"{region}_{epoch}_motor_tp_place_{args.motor_feature_mode}"
            )
            fit_dataset = build_fit_dataset(
                fit_result=fit_result,
                animal_name=args.animal_name,
                date=args.date,
                region=region,
                epoch=epoch,
                movement_firing_rates=selected_unit_rates,
                min_firing_rate_hz=region_thresholds[region],
                sources={
                    **session["sources"],
                    "body_position": "pickle",
                },
                fit_parameters={
                    "position_offset": args.position_offset,
                    "speed_threshold_cm_s": args.speed_threshold_cm_s,
                    "speed_sigma_s": DEFAULT_SPEED_SIGMA_S,
                    "bin_size_s": args.bin_size_s,
                    "ridge": args.ridge,
                    "n_folds": args.n_folds,
                    "seed": args.seed,
                    "motor_feature_mode": args.motor_feature_mode,
                    "motor_zscore_eps": args.motor_zscore_eps,
                    "motor_spline_k": args.motor_spline_k,
                    "motor_spline_order": args.motor_spline_order,
                    "tp_spline_k": args.tp_spline_k,
                    "tp_spline_order": args.tp_spline_order,
                },
            )
            result_path = data_dir / f"{result_stem}.nc"
            fit_dataset.to_netcdf(result_path)
            saved_datasets.append(result_path)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.task_progression_motor",
        parameters={
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
            "ridge": args.ridge,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "motor_feature_mode": args.motor_feature_mode,
            "motor_zscore_eps": args.motor_zscore_eps,
            "motor_spline_k": args.motor_spline_k,
            "motor_spline_order": args.motor_spline_order,
            "tp_spline_k": args.tp_spline_k,
            "tp_spline_order": args.tp_spline_order,
        },
        outputs={
            "sources": {
                **session["sources"],
                "body_position": "pickle",
            },
            "saved_datasets": saved_datasets,
            "skipped_fits": skipped_fits,
        },
    )

    if saved_datasets:
        print(f"Saved {len(saved_datasets)} NetCDF fit dataset(s) to {data_dir}")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
