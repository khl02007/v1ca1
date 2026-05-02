from __future__ import annotations

"""Compare place, generalized place, task progression, and generalized TP encoding.

This script loads one dataset through shared task-progression utilities in
`v1ca1.task_progression._session`, where the relevant preprocessing is built,
including movement intervals, linearized position, and task progression. It
then rebuilds cross-validated tuning curves on movement-restricted data and
compares four encoding models for each unit in each run epoch:

- concatenated four-trajectory linear position
- generalized full-W linear position with shared center stem, both side arms
  present once, and an explicit branch gap between left and right branches
- combined two-branch task progression
- generalized one-branch task progression shared across all four trajectories

Held-out log-likelihood is reported for two cases:

- model: the feature-dependent encoding model defined by the training-fold
  tuning curve, using trajectory-concatenated linear position, generalized
  full-W linear position, directional task progression, or generalized task
  progression as the feature
- null: a baseline model with one constant firing rate for the unit across the
  evaluation epoch, set from the train-fold mean firing rate

The Poisson log-likelihood includes the full count term
`k * log(lambda * dt) - lambda * dt - log(k!)` for both the encoding model and
the null model.

The script requires NPZ-backed timestamps and the combined cleaned DLC
`position.parquet` export. The dark epoch must be specified explicitly. Light
epochs may be provided explicitly; otherwise the script uses all run epochs
with head position present in `position.parquet` except the selected dark
epoch.

Primary outputs are per-unit parquet tables, because they are summary tables
rather than time-domain artifacts. A run log is written under
`analysis_path / "v1ca1_log"`.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
from sklearn.model_selection import KFold

from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    get_run_epochs,
    load_clean_dlc_position_data,
    load_epoch_tags,
    load_ephys_timestamps_all,
    load_position_timestamps,
)
from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TP_TRANSFER_FAMILY_ORDER,
    TP_TRANSFER_PAIR_SPECS,
    TRAJECTORY_TYPES,
    build_task_progression_bins,
    build_combined_task_progression_bins,
    build_generalized_place_bins,
    build_linear_position_bins,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)


DEFAULT_N_FOLDS = 5
DEFAULT_BIN_SIZE_S = 0.02
DEFAULT_SIGMA_BINS = 1.0
DEFAULT_RANDOM_STATE = 47
DEFAULT_MIN_PLOT_SPIKES = 50


def _format_float_token(value: float) -> str:
    """Return a path-safe compact token for one numeric value."""
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def format_place_bin_size_token(place_bin_size_cm: float) -> str:
    """Return the filename token for one place-bin-size setting."""
    return f"placebin{_format_float_token(place_bin_size_cm)}cm"


def smooth_pf_along_position_nan_aware(
    pf: Any,
    pos_dim: str,
    sigma_bins: float,
    *,
    eps: float = 1e-12,
    mode: str = "nearest",
) -> Any:
    """Smooth a tuning curve without turning unsupported bins into zeros."""
    axis = pf.get_axis_num(pos_dim)
    values = np.asarray(pf.values, dtype=np.float64)

    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)

    numerator = gaussian_filter1d(filled, sigma=sigma_bins, axis=axis, mode=mode)
    denominator = gaussian_filter1d(
        mask.astype(np.float64),
        sigma=sigma_bins,
        axis=axis,
        mode=mode,
    )
    smoothed = numerator / np.maximum(denominator, eps)
    smoothed = np.where(denominator > eps, smoothed, np.nan)
    return pf.copy(data=smoothed)


def _intervalset_to_arrays(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted start and end arrays for one IntervalSet."""
    starts = np.asarray(intervals.start, dtype=np.float64).ravel()
    ends = np.asarray(intervals.end, dtype=np.float64).ravel()
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts)
    return starts[order], ends[order]


def _make_intervalset(starts: list[np.ndarray], ends: list[np.ndarray]) -> Any:
    """Create one IntervalSet from a list of interval arrays."""
    import pynapple as nap

    start_chunks = [chunk for chunk in starts if chunk.size]
    end_chunks = [chunk for chunk in ends if chunk.size]
    if not start_chunks:
        return nap.IntervalSet(start=np.array([], dtype=float), end=np.array([], dtype=float))

    start_array = np.concatenate(start_chunks).astype(float, copy=False)
    end_array = np.concatenate(end_chunks).astype(float, copy=False)
    order = np.argsort(start_array)
    return nap.IntervalSet(start=start_array[order], end=end_array[order], time_units="s")


def require_npz_session_sources(analysis_path: Path) -> list[str]:
    """Return run epochs after validating that all timestamp inputs are NPZ-backed."""
    epoch_tags, epoch_source = load_epoch_tags(analysis_path)
    if epoch_source != "pynapple":
        raise ValueError(
            "This script requires timestamps_ephys.npz. Pickle timestamp inputs are not "
            f"supported for {analysis_path}."
        )

    _position_epoch_tags, _timestamps_position, position_source = load_position_timestamps(
        analysis_path
    )
    if position_source != "pynapple":
        raise ValueError(
            "This script requires timestamps_position.npz. Pickle timestamp inputs are not "
            f"supported for {analysis_path}."
        )

    _timestamps_ephys_all, ephys_all_source = load_ephys_timestamps_all(analysis_path)
    if ephys_all_source != "pynapple":
        raise ValueError(
            "This script requires timestamps_ephys_all.npz. Pickle timestamp inputs are not "
            f"supported for {analysis_path}."
        )

    return get_run_epochs(epoch_tags)


def get_head_position_epochs_from_parquet(analysis_path: Path) -> set[str]:
    """Return epochs present in the combined cleaned DLC `position.parquet` export."""
    epoch_order, _head_position, _body_position = load_clean_dlc_position_data(
        analysis_path,
        input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    return {str(epoch) for epoch in epoch_order}


def resolve_light_dark_epochs(
    run_epochs: list[str],
    head_position_epochs: set[str],
    *,
    dark_epoch: str,
    light_epochs: list[str] | None,
) -> tuple[list[str], str, list[str]]:
    """Return validated light/dark epochs and the ordered run epochs to load."""
    if not run_epochs:
        raise ValueError("No run epochs were found for this session.")

    if dark_epoch not in run_epochs:
        raise ValueError(
            f"Requested dark epoch {dark_epoch!r} was not found in run epochs {run_epochs!r}."
        )
    if dark_epoch not in head_position_epochs:
        raise ValueError(
            f"Dark epoch {dark_epoch!r} does not have head position in "
            f"{DEFAULT_CLEAN_DLC_POSITION_NAME!r}."
        )

    if light_epochs is None:
        selected_light_epochs = [
            epoch
            for epoch in run_epochs
            if epoch in head_position_epochs and epoch != dark_epoch
        ]
    else:
        selected_light_epochs = list(dict.fromkeys(light_epochs))
        missing = [epoch for epoch in selected_light_epochs if epoch not in run_epochs]
        if missing:
            raise ValueError(
                "Requested light epochs were not found in run epochs "
                f"{run_epochs!r}: {missing!r}"
            )
        missing_head_position = [
            epoch for epoch in selected_light_epochs if epoch not in head_position_epochs
        ]
        if missing_head_position:
            raise ValueError(
                "Requested light epochs do not have head position in "
                f"{DEFAULT_CLEAN_DLC_POSITION_NAME!r}: {missing_head_position!r}"
            )
        if dark_epoch in selected_light_epochs:
            raise ValueError("--light-epoch must not include the selected --dark-epoch.")

    if not selected_light_epochs:
        raise ValueError(
            "No light epochs were selected. Provide --light-epoch explicitly or ensure "
            "position.parquet contains head position for at least one run epoch other than "
            f"{dark_epoch!r}."
        )

    selected_run_epochs = [
        epoch for epoch in run_epochs if epoch == dark_epoch or epoch in selected_light_epochs
    ]
    return selected_light_epochs, dark_epoch, selected_run_epochs


def get_min_lap_count(trajectory_intervals: dict[str, Any]) -> int:
    """Return the minimum number of laps across the four trajectory types."""
    return min(
        _intervalset_to_arrays(trajectory_intervals[trajectory_type])[0].size
        for trajectory_type in TRAJECTORY_TYPES
    )


def validate_fold_count(
    trajectory_intervals_by_epoch: dict[str, dict[str, Any]],
    n_folds: int,
) -> dict[str, int]:
    """Validate that each epoch has enough laps for the requested fold count."""
    if n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    min_lap_counts: dict[str, int] = {}
    for epoch, trajectory_intervals in trajectory_intervals_by_epoch.items():
        min_laps = get_min_lap_count(trajectory_intervals)
        min_lap_counts[epoch] = min_laps
        if min_laps < 2:
            raise ValueError(
                f"Epoch {epoch!r} has fewer than 2 laps in at least one trajectory "
                f"(minimum lap count: {min_laps})."
            )
        if n_folds > min_laps:
            raise ValueError(
                f"Epoch {epoch!r} does not support --n-folds={n_folds}. "
                f"Minimum lap count across trajectories is {min_laps}."
            )
    return min_lap_counts


def build_train_test_folds(
    trajectory_intervals: dict[str, Any],
    n_folds: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Build lap-wise train/test IntervalSets pooled across trajectory types."""
    train_starts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    train_ends: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    test_starts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    test_ends: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}

    for trajectory_type in TRAJECTORY_TYPES:
        starts, ends = _intervalset_to_arrays(trajectory_intervals[trajectory_type])
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(starts.size))):
            train_starts[fold].append(starts[train_idx])
            train_ends[fold].append(ends[train_idx])
            test_starts[fold].append(starts[test_idx])
            test_ends[fold].append(ends[test_idx])

    train_folds = {
        fold: _make_intervalset(train_starts[fold], train_ends[fold]) for fold in range(n_folds)
    }
    test_folds = {
        fold: _make_intervalset(test_starts[fold], test_ends[fold]) for fold in range(n_folds)
    }
    return train_folds, test_folds


def build_single_trajectory_train_test_folds(
    intervals: Any,
    n_folds: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Build lap-wise train/test IntervalSets for one trajectory only."""
    starts, ends = _intervalset_to_arrays(intervals)
    n_laps = starts.size
    if n_laps < 2:
        raise ValueError(
            "A trajectory-specific CV split requires at least 2 laps. "
            f"Got {n_laps}."
        )
    if n_folds > n_laps:
        raise ValueError(
            f"Requested n_folds={n_folds} exceeds the available trajectory laps ({n_laps})."
        )

    train_starts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    train_ends: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    test_starts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    test_ends: dict[int, list[np.ndarray]] = {fold: [] for fold in range(n_folds)}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_laps))):
        train_starts[fold].append(starts[train_idx])
        train_ends[fold].append(ends[train_idx])
        test_starts[fold].append(starts[test_idx])
        test_ends[fold].append(ends[test_idx])

    train_folds = {
        fold: _make_intervalset(train_starts[fold], train_ends[fold]) for fold in range(n_folds)
    }
    test_folds = {
        fold: _make_intervalset(test_starts[fold], test_ends[fold]) for fold in range(n_folds)
    }
    return train_folds, test_folds


def poisson_ll_on_epoch(
    *,
    unit_spikes: Any,
    position: Any,
    tuning_curve: npt.NDArray[np.float64],
    bin_edges: npt.NDArray[np.float64],
    epoch: Any,
    bin_size_s: float,
    epsilon: float = 1e-10,
    null_rate: float | None = None,
    fill_rate: float | None = None,
) -> tuple[float, float, int]:
    """Evaluate held-out Poisson log-likelihood for one unit and one feature.

    Returns
    -------
    ll_model, ll_null, n_spikes
        `ll_model` is the held-out log-likelihood under the feature-dependent
        encoding model defined by the training-fold tuning curve. In this
        script, that is either the linear-position tuning curve or the
        task-progression tuning curve.

        `ll_null` is the held-out log-likelihood under a null model with one
        constant firing rate for the whole evaluation epoch. When `null_rate`
        is provided, that constant rate is the train-fold mean firing rate for
        the same unit.

        Both likelihoods include the full Poisson count term `-log(k!)`.

        `n_spikes` is the total number of held-out spikes contributing to the
        likelihood.
    """
    pos_ep = position.restrict(epoch)
    spikes_ep = unit_spikes.restrict(epoch)

    binned_spikes = spikes_ep.count(bin_size_s, epoch)
    pos_at_bins = pos_ep.interpolate(binned_spikes, ep=epoch)

    spike_counts = np.asarray(binned_spikes.d, dtype=np.int64).ravel()
    positions = np.asarray(pos_at_bins.d, dtype=np.float64).ravel()
    valid = np.isfinite(positions)
    if not np.all(valid):
        spike_counts = spike_counts[valid]
        positions = positions[valid]

    n_spikes = int(spike_counts.sum())
    total_time = float(spike_counts.size * bin_size_s)
    if total_time <= 0.0 or spike_counts.size == 0:
        return 0.0, 0.0, 0

    null_rate_eval = float(null_rate) if null_rate is not None else n_spikes / total_time
    null_rate_eval = max(null_rate_eval, epsilon)
    fill_rate_eval = float(fill_rate) if fill_rate is not None else null_rate_eval
    fill_rate_eval = max(fill_rate_eval, epsilon)

    tuning_curve = np.asarray(tuning_curve, dtype=np.float64).ravel()
    spatial_idx = np.digitize(positions, bin_edges) - 1
    spatial_idx = np.clip(spatial_idx, 0, tuning_curve.size - 1)

    model_rates = tuning_curve[spatial_idx]
    model_rates = np.where(np.isfinite(model_rates), model_rates, fill_rate_eval)
    model_rates = np.maximum(model_rates, epsilon)

    log_factorial = gammaln(spike_counts + 1.0)
    ll_model = float(
        np.sum(
            spike_counts * np.log(model_rates * bin_size_s)
            - model_rates * bin_size_s
            - log_factorial
        )
    )
    ll_null = float(
        np.sum(
            spike_counts * np.log(null_rate_eval * bin_size_s)
            - null_rate_eval * bin_size_s
            - log_factorial
        )
    )
    return ll_model, ll_null, n_spikes


def initialize_cv_store(unit_ids: list[Any], n_folds: int) -> dict[Any, dict[str, Any]]:
    """Return the legacy nested CV metric structure for one epoch."""
    return {
        unit_id: {
            "fold_ll_model": np.zeros(n_folds, dtype=np.float64),
            "fold_ll_null": np.zeros(n_folds, dtype=np.float64),
            "fold_n_spikes": np.zeros(n_folds, dtype=np.int64),
            "ll_model_per_spike_cv": np.nan,
            "ll_null_per_spike_cv": np.nan,
            "info_bits_per_spike_cv": np.nan,
        }
        for unit_id in unit_ids
    }


def aggregate_cv_store(cv_store: dict[Any, dict[str, Any]]) -> None:
    """Aggregate fold-wise likelihoods into per-spike cross-validated metrics."""
    for metrics in cv_store.values():
        ll_model = float(np.sum(metrics["fold_ll_model"]))
        ll_null = float(np.sum(metrics["fold_ll_null"]))
        n_spikes = int(np.sum(metrics["fold_n_spikes"]))
        if n_spikes <= 0:
            continue
        metrics["ll_model_per_spike_cv"] = ll_model / n_spikes
        metrics["ll_null_per_spike_cv"] = ll_null / n_spikes
        metrics["info_bits_per_spike_cv"] = (ll_model - ll_null) / (np.log(2.0) * n_spikes)


def initialize_tp_cross_trajectory_store(
    unit_ids: list[Any],
    n_folds: int,
) -> dict[Any, dict[str, Any]]:
    """Return fold-wise storage for directed TP cross-trajectory transfer."""
    return {
        unit_id: {
            "fold_ll_tp_cv_target": np.zeros(n_folds, dtype=np.float64),
            "fold_ll_tp_cross": np.zeros(n_folds, dtype=np.float64),
            "fold_ll_null": np.zeros(n_folds, dtype=np.float64),
            "fold_n_spikes": np.zeros(n_folds, dtype=np.int64),
            "ll_tp_cv_target": np.nan,
            "ll_tp_cross": np.nan,
            "ll_null": np.nan,
            "delta_bits_cross_vs_cv": np.nan,
            "delta_bits_cross_vs_null": np.nan,
            "delta_bits_cv_vs_null": np.nan,
        }
        for unit_id in unit_ids
    }


def aggregate_tp_cross_trajectory_store(store: dict[Any, dict[str, Any]]) -> None:
    """Aggregate fold-wise directed TP transfer metrics into per-spike values."""
    for metrics in store.values():
        ll_tp_cv_target = float(np.sum(metrics["fold_ll_tp_cv_target"]))
        ll_tp_cross = float(np.sum(metrics["fold_ll_tp_cross"]))
        ll_null = float(np.sum(metrics["fold_ll_null"]))
        n_spikes = int(np.sum(metrics["fold_n_spikes"]))
        if n_spikes <= 0:
            continue

        ll_tp_cv_target_per_spike = ll_tp_cv_target / n_spikes
        ll_tp_cross_per_spike = ll_tp_cross / n_spikes
        ll_null_per_spike = ll_null / n_spikes
        metrics["ll_tp_cv_target"] = ll_tp_cv_target_per_spike
        metrics["ll_tp_cross"] = ll_tp_cross_per_spike
        metrics["ll_null"] = ll_null_per_spike
        metrics["delta_bits_cross_vs_cv"] = (
            ll_tp_cross_per_spike - ll_tp_cv_target_per_spike
        ) / np.log(2.0)
        metrics["delta_bits_cross_vs_null"] = (
            ll_tp_cross_per_spike - ll_null_per_spike
        ) / np.log(2.0)
        metrics["delta_bits_cv_vs_null"] = (
            ll_tp_cv_target_per_spike - ll_null_per_spike
        ) / np.log(2.0)


def run_cross_validated_encoding(
    spikes: Any,
    linear_position: Any,
    generalized_place_position: Any,
    task_progression: Any,
    generalized_task_progression: Any,
    movement_interval: Any,
    trajectory_intervals: dict[str, Any],
    position_bins: np.ndarray,
    generalized_place_bins: np.ndarray,
    task_progression_bins: np.ndarray,
    generalized_task_progression_bins: np.ndarray,
    n_folds: int,
    bin_size_s: float,
    sigma_bins: float,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[Any, dict[str, Any]]]:
    """Fit and score place, generalized place, TP, and generalized TP models."""
    import pynapple as nap

    unit_ids = list(spikes.keys())
    cv_by_model = {
        "place": initialize_cv_store(unit_ids, n_folds=n_folds),
        "generalized_place": initialize_cv_store(unit_ids, n_folds=n_folds),
        "tp": initialize_cv_store(unit_ids, n_folds=n_folds),
        "gtp": initialize_cv_store(unit_ids, n_folds=n_folds),
    }
    train_folds, test_folds = build_train_test_folds(
        trajectory_intervals,
        n_folds=n_folds,
        random_state=random_state,
    )

    for fold in range(n_folds):
        train_fold = train_folds[fold].intersect(movement_interval)
        test_fold = test_folds[fold].intersect(movement_interval)

        train_duration = float(train_fold.tot_length())
        test_duration = float(test_fold.tot_length())
        if train_duration <= 0.0 or test_duration <= 0.0:
            continue

        tuning_by_model = {
            "place": smooth_pf_along_position_nan_aware(
                nap.compute_tuning_curves(
                    data=spikes,
                    features=linear_position,
                    bins=[position_bins],
                    epochs=train_fold,
                    feature_names=["linpos"],
                ),
                pos_dim="linpos",
                sigma_bins=sigma_bins,
            ),
            "generalized_place": smooth_pf_along_position_nan_aware(
                nap.compute_tuning_curves(
                    data=spikes,
                    features=generalized_place_position,
                    bins=[generalized_place_bins],
                    epochs=train_fold,
                    feature_names=["generalized_place"],
                ),
                pos_dim="generalized_place",
                sigma_bins=sigma_bins,
            ),
            "tp": smooth_pf_along_position_nan_aware(
                nap.compute_tuning_curves(
                    data=spikes,
                    features=task_progression,
                    bins=[task_progression_bins],
                    epochs=train_fold,
                    feature_names=["tp"],
                ),
                pos_dim="tp",
                sigma_bins=sigma_bins,
            ),
            "gtp": smooth_pf_along_position_nan_aware(
                nap.compute_tuning_curves(
                    data=spikes,
                    features=generalized_task_progression,
                    bins=[generalized_task_progression_bins],
                    epochs=train_fold,
                    feature_names=["gtp"],
                ),
                pos_dim="gtp",
                sigma_bins=sigma_bins,
            ),
        }
        feature_by_model = {
            "place": linear_position,
            "generalized_place": generalized_place_position,
            "tp": task_progression,
            "gtp": generalized_task_progression,
        }
        bins_by_model = {
            "place": np.asarray(position_bins, dtype=float),
            "generalized_place": np.asarray(generalized_place_bins, dtype=float),
            "tp": np.asarray(task_progression_bins, dtype=float),
            "gtp": np.asarray(generalized_task_progression_bins, dtype=float),
        }

        for unit_id in unit_ids:
            unit_spikes = spikes[unit_id]
            n_train_spikes = int(len(unit_spikes.restrict(train_fold).t))
            train_rate = max(n_train_spikes / train_duration, 1e-10)

            for model_name, model_metrics in cv_by_model.items():
                tuning_curve = tuning_by_model[model_name].sel(unit=unit_id)
                ll_model, ll_null, n_spikes = poisson_ll_on_epoch(
                    unit_spikes=unit_spikes,
                    position=feature_by_model[model_name],
                    tuning_curve=np.asarray(tuning_curve.values, dtype=float),
                    bin_edges=bins_by_model[model_name],
                    epoch=test_fold,
                    bin_size_s=bin_size_s,
                    epsilon=1e-10,
                    fill_rate=train_rate,
                    null_rate=train_rate,
                )
                model_metrics[unit_id]["fold_ll_model"][fold] = ll_model
                model_metrics[unit_id]["fold_ll_null"][fold] = ll_null
                model_metrics[unit_id]["fold_n_spikes"][fold] = n_spikes

    for cv_store in cv_by_model.values():
        aggregate_cv_store(cv_store)
    return cv_by_model


def run_tp_cross_trajectory_encoding(
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    trajectory_intervals: dict[str, Any],
    task_progression_bins: np.ndarray,
    n_folds: int,
    bin_size_s: float,
    sigma_bins: float,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Score directed TP transfer families against trajectory-specific target CV.

    Both the source-trajectory transfer model and the target-trajectory baseline
    are trained fold-by-fold. For each fold, the source model is fit on the
    held-in source laps, the target baseline is fit on the held-in target laps,
    and both are evaluated on the same held-out target laps.
    """
    import pynapple as nap

    unit_ids = list(spikes.keys())
    rows: list[dict[str, Any]] = []
    skipped_pairs: list[dict[str, Any]] = []
    task_progression_bins = np.asarray(task_progression_bins, dtype=float)

    for pair_spec in TP_TRANSFER_PAIR_SPECS:
        transfer_family = pair_spec["transfer_family"]
        source_trajectory = pair_spec["source_trajectory"]
        target_trajectory = pair_spec["target_trajectory"]
        source_turn_type = pair_spec["source_turn_type"]
        turn_type = pair_spec["turn_type"]
        flip_tuning_curve = bool(pair_spec.get("flip_tuning_curve", False))
        source_lap_count = int(_intervalset_to_arrays(trajectory_intervals[source_trajectory])[0].size)
        target_lap_count = int(_intervalset_to_arrays(trajectory_intervals[target_trajectory])[0].size)
        if source_lap_count < 2 or n_folds > source_lap_count:
            skipped_pairs.append(
                {
                    "transfer_family": transfer_family,
                    "turn_type": turn_type,
                    "source_turn_type": source_turn_type,
                    "source_trajectory": source_trajectory,
                    "target_trajectory": target_trajectory,
                    "reason": (
                        f"source trajectory supports {source_lap_count} laps, "
                        f"which is insufficient for n_folds={n_folds}"
                    ),
                }
            )
            continue
        if target_lap_count < 2 or n_folds > target_lap_count:
            skipped_pairs.append(
                {
                    "transfer_family": transfer_family,
                    "turn_type": turn_type,
                    "source_turn_type": source_turn_type,
                    "source_trajectory": source_trajectory,
                    "target_trajectory": target_trajectory,
                    "reason": (
                        f"target trajectory supports {target_lap_count} laps, "
                        f"which is insufficient for n_folds={n_folds}"
                    ),
                }
            )
            continue

        source_train_folds, _source_test_folds = build_single_trajectory_train_test_folds(
            trajectory_intervals[source_trajectory],
            n_folds=n_folds,
            random_state=random_state,
        )
        target_train_folds, target_test_folds = build_single_trajectory_train_test_folds(
            trajectory_intervals[target_trajectory],
            n_folds=n_folds,
            random_state=random_state,
        )

        store = initialize_tp_cross_trajectory_store(unit_ids, n_folds=n_folds)
        used_fold_count = 0
        for fold in range(n_folds):
            source_train_fold = source_train_folds[fold].intersect(movement_interval)
            target_train_fold = target_train_folds[fold].intersect(movement_interval)
            target_test_fold = target_test_folds[fold].intersect(movement_interval)
            source_train_duration = float(source_train_fold.tot_length())
            target_train_duration = float(target_train_fold.tot_length())
            target_test_duration = float(target_test_fold.tot_length())
            if (
                source_train_duration <= 0.0
                or target_train_duration <= 0.0
                or target_test_duration <= 0.0
            ):
                continue

            used_fold_count += 1
            source_tuning = smooth_pf_along_position_nan_aware(
                nap.compute_tuning_curves(
                    data=spikes,
                    features=task_progression_by_trajectory[source_trajectory],
                    bins=[task_progression_bins],
                    epochs=source_train_fold,
                    feature_names=["tp"],
                ),
                pos_dim="tp",
                sigma_bins=sigma_bins,
            )
            target_tuning = smooth_pf_along_position_nan_aware(
                nap.compute_tuning_curves(
                    data=spikes,
                    features=task_progression_by_trajectory[target_trajectory],
                    bins=[task_progression_bins],
                    epochs=target_train_fold,
                    feature_names=["tp"],
                ),
                pos_dim="tp",
                sigma_bins=sigma_bins,
            )
            for unit_id in unit_ids:
                unit_spikes = spikes[unit_id]
                n_train_spikes = int(len(unit_spikes.restrict(target_train_fold).t))
                train_rate = max(n_train_spikes / target_train_duration, 1e-10)
                source_curve = np.asarray(
                    source_tuning.sel(unit=unit_id).values,
                    dtype=float,
                )
                if flip_tuning_curve:
                    source_curve = source_curve[::-1]

                ll_tp_cv_target, ll_null, n_spikes = poisson_ll_on_epoch(
                    unit_spikes=unit_spikes,
                    position=task_progression_by_trajectory[target_trajectory],
                    tuning_curve=np.asarray(
                        target_tuning.sel(unit=unit_id).values,
                        dtype=float,
                    ),
                    bin_edges=task_progression_bins,
                    epoch=target_test_fold,
                    bin_size_s=bin_size_s,
                    epsilon=1e-10,
                    fill_rate=train_rate,
                    null_rate=train_rate,
                )
                ll_tp_cross, _ll_null_cross, n_spikes_cross = poisson_ll_on_epoch(
                    unit_spikes=unit_spikes,
                    position=task_progression_by_trajectory[target_trajectory],
                    tuning_curve=source_curve,
                    bin_edges=task_progression_bins,
                    epoch=target_test_fold,
                    bin_size_s=bin_size_s,
                    epsilon=1e-10,
                    fill_rate=train_rate,
                    null_rate=train_rate,
                )
                if n_spikes_cross != n_spikes:
                    raise ValueError(
                        "Cross-trajectory and target-CV spike counts diverged for "
                        f"{source_trajectory!r} -> {target_trajectory!r}, unit {unit_id!r}, "
                        f"fold {fold}: {n_spikes_cross} vs {n_spikes}."
                    )

                store[unit_id]["fold_ll_tp_cv_target"][fold] = ll_tp_cv_target
                store[unit_id]["fold_ll_tp_cross"][fold] = ll_tp_cross
                store[unit_id]["fold_ll_null"][fold] = ll_null
                store[unit_id]["fold_n_spikes"][fold] = n_spikes

        if used_fold_count == 0:
            skipped_pairs.append(
                {
                    "transfer_family": transfer_family,
                    "turn_type": turn_type,
                    "source_turn_type": source_turn_type,
                    "source_trajectory": source_trajectory,
                    "target_trajectory": target_trajectory,
                    "reason": (
                        "all fold-matched source-train, target-train, or target-test "
                        "intervals were empty after movement restriction"
                    ),
                }
            )
            continue

        aggregate_tp_cross_trajectory_store(store)
        for unit_id in unit_ids:
            metrics = store[unit_id]
            if not (
                np.isfinite(metrics["ll_tp_cv_target"])
                or np.isfinite(metrics["ll_tp_cross"])
                or np.isfinite(metrics["ll_null"])
            ):
                continue
            rows.append(
                {
                    "unit": unit_id,
                    "transfer_family": transfer_family,
                    "source_turn_type": source_turn_type,
                    "turn_type": turn_type,
                    "source_trajectory": source_trajectory,
                    "target_trajectory": target_trajectory,
                    "n_spikes": int(np.sum(np.asarray(metrics["fold_n_spikes"], dtype=np.int64))),
                    "ll_tp_cv_target": float(metrics["ll_tp_cv_target"]),
                    "ll_tp_cross": float(metrics["ll_tp_cross"]),
                    "ll_null": float(metrics["ll_null"]),
                    "delta_bits_cross_vs_cv": float(metrics["delta_bits_cross_vs_cv"]),
                    "delta_bits_cross_vs_null": float(metrics["delta_bits_cross_vs_null"]),
                    "delta_bits_cv_vs_null": float(metrics["delta_bits_cv_vs_null"]),
                }
            )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "unit",
                "transfer_family",
                "source_turn_type",
                "turn_type",
                "source_trajectory",
                "target_trajectory",
                "n_spikes",
                "ll_tp_cv_target",
                "ll_tp_cross",
                "ll_null",
                "delta_bits_cross_vs_cv",
                "delta_bits_cross_vs_null",
                "delta_bits_cv_vs_null",
            ]
        )
        return empty, skipped_pairs

    table = pd.DataFrame(rows).sort_values(
        ["target_trajectory", "transfer_family", "source_trajectory", "unit"],
        ignore_index=True,
    )
    return table, skipped_pairs


def cv_epoch_to_df(
    cv_by_model: dict[str, dict[Any, dict[str, Any]]],
) -> pd.DataFrame:
    """Convert legacy nested CV results into one unit-indexed summary table."""
    required_models = ("place", "generalized_place", "tp", "gtp")
    missing_models = [model_name for model_name in required_models if model_name not in cv_by_model]
    if missing_models:
        raise ValueError(f"Missing CV results for models: {missing_models!r}")

    rows: list[dict[str, Any]] = []
    for unit_id in cv_by_model["place"]:
        metrics_by_model = {
            model_name: cv_by_model[model_name][unit_id] for model_name in required_models
        }
        n_spikes_values = np.asarray(
            [
                np.sum(np.asarray(metrics.get("fold_n_spikes", 0), dtype=np.int64))
                for metrics in metrics_by_model.values()
            ],
            dtype=np.int64,
        )
        if np.any(n_spikes_values != n_spikes_values[0]):
            raise ValueError(
                f"Cross-validated spike counts diverged across models for unit {unit_id!r}: "
                f"{n_spikes_values.tolist()!r}"
            )

        ll_null_values = np.asarray(
            [
                float(metrics.get("ll_null_per_spike_cv", np.nan))
                for metrics in metrics_by_model.values()
            ],
            dtype=float,
        )
        finite_null_values = ll_null_values[np.isfinite(ll_null_values)]
        if finite_null_values.size > 1 and not np.allclose(
            finite_null_values,
            finite_null_values[0],
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(
                f"Cross-validated null LLs diverged across models for unit {unit_id!r}: "
                f"{ll_null_values.tolist()!r}"
            )

        ll_place = float(metrics_by_model["place"].get("ll_model_per_spike_cv", np.nan))
        ll_generalized_place = float(
            metrics_by_model["generalized_place"].get("ll_model_per_spike_cv", np.nan)
        )
        ll_tp = float(metrics_by_model["tp"].get("ll_model_per_spike_cv", np.nan))
        ll_gtp = float(metrics_by_model["gtp"].get("ll_model_per_spike_cv", np.nan))
        ll_null = float(finite_null_values[0]) if finite_null_values.size else np.nan
        info_place = float(metrics_by_model["place"].get("info_bits_per_spike_cv", np.nan))
        info_generalized_place = float(
            metrics_by_model["generalized_place"].get("info_bits_per_spike_cv", np.nan)
        )
        info_tp = float(metrics_by_model["tp"].get("info_bits_per_spike_cv", np.nan))
        info_gtp = float(metrics_by_model["gtp"].get("info_bits_per_spike_cv", np.nan))
        rows.append(
            {
                "unit": unit_id,
                "n_spikes": int(n_spikes_values[0]),
                "ll_null": ll_null,
                "ll_place": ll_place,
                "ll_generalized_place": ll_generalized_place,
                "ll_tp": ll_tp,
                "ll_gtp": ll_gtp,
                "info_bits_place": info_place,
                "info_bits_generalized_place": info_generalized_place,
                "info_bits_tp": info_tp,
                "info_bits_gtp": info_gtp,
                "delta_bits_place_vs_tp": (ll_place - ll_tp) / np.log(2.0)
                if np.isfinite(ll_place) and np.isfinite(ll_tp)
                else np.nan,
                "delta_bits_generalized_place_vs_tp": (
                    ll_generalized_place - ll_tp
                ) / np.log(2.0)
                if np.isfinite(ll_generalized_place) and np.isfinite(ll_tp)
                else np.nan,
                "delta_bits_gtp_vs_tp": (ll_gtp - ll_tp) / np.log(2.0)
                if np.isfinite(ll_gtp) and np.isfinite(ll_tp)
                else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "n_spikes",
                "ll_null",
                "ll_place",
                "ll_generalized_place",
                "ll_tp",
                "ll_gtp",
                "info_bits_place",
                "info_bits_generalized_place",
                "info_bits_tp",
                "info_bits_gtp",
                "delta_bits_place_vs_tp",
                "delta_bits_generalized_place_vs_tp",
                "delta_bits_gtp_vs_tp",
            ],
            index=pd.Index([], name="unit"),
        )
    return pd.DataFrame(rows).set_index("unit").sort_index()


def filter_epoch_df(
    df: pd.DataFrame,
    *,
    ll_columns: tuple[str, str],
    min_spikes: int = DEFAULT_MIN_PLOT_SPIKES,
) -> pd.DataFrame:
    """Filter units with valid model LL estimates and enough held-out spikes."""
    filtered = df.copy()
    filtered = filtered[np.isfinite(filtered[ll_columns[0]]) & np.isfinite(filtered[ll_columns[1]])]
    filtered = filtered[filtered["n_spikes"] >= int(min_spikes)]
    return filtered


def build_comparison_table(
    light_df: pd.DataFrame,
    dark_df: pd.DataFrame,
    light_epoch: str,
    dark_epoch: str,
) -> pd.DataFrame:
    """Join one light and one dark epoch summary table on common units."""
    comparison = light_df.add_prefix("light_").join(
        dark_df.add_prefix("dark_"),
        how="inner",
    )
    comparison.insert(0, "dark_epoch", dark_epoch)
    comparison.insert(0, "light_epoch", light_epoch)
    comparison["delta_bits_change_place_vs_tp"] = (
        comparison["dark_delta_bits_place_vs_tp"] - comparison["light_delta_bits_place_vs_tp"]
    )
    comparison["delta_bits_change_generalized_place_vs_tp"] = (
        comparison["dark_delta_bits_generalized_place_vs_tp"]
        - comparison["light_delta_bits_generalized_place_vs_tp"]
    )
    comparison["delta_bits_change_gtp_vs_tp"] = (
        comparison["dark_delta_bits_gtp_vs_tp"] - comparison["light_delta_bits_gtp_vs_tp"]
    )
    return comparison


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ECDF coordinates for a 1D array after dropping NaNs."""
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return valid, valid
    x = np.sort(valid)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _bootstrap_ci_fraction_positive(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return the bootstrap estimate and CI for the fraction of positive values."""
    rng = np.random.default_rng(seed)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan, np.nan, np.nan

    estimate = float(np.mean(valid > 0))
    bootstrap = np.empty(int(n_boot), dtype=np.float64)
    for index in range(int(n_boot)):
        sample = rng.choice(valid, size=valid.size, replace=True)
        bootstrap[index] = np.mean(sample > 0)
    lower = float(np.quantile(bootstrap, alpha / 2))
    upper = float(np.quantile(bootstrap, 1.0 - alpha / 2))
    return estimate, lower, upper


def _build_zero_including_histogram_edges(
    values: np.ndarray,
    n_bins: int = 20,
    eps: float = 1e-9,
) -> np.ndarray:
    """Return histogram edges that span the data and include zero exactly."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return np.asarray([-0.5, 0.0, 0.5], dtype=float)

    min_value = float(valid.min())
    max_value = float(valid.max())
    span = max(abs(min_value), abs(max_value))
    if span <= eps:
        return np.asarray([-0.5, 0.0, 0.5], dtype=float)

    bin_width = max((2.0 * span) / max(int(n_bins), 1), eps)
    n_left = int(np.ceil(abs(min_value) / bin_width))
    n_right = int(np.ceil(abs(max_value) / bin_width))
    edges = bin_width * np.arange(-n_left, n_right + 1, dtype=float)
    if edges.size < 2:
        return np.asarray([-bin_width, 0.0, bin_width], dtype=float)
    return edges


def _format_delta_histogram_stats(values: np.ndarray) -> str:
    """Return summary text for one model-difference histogram."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return "No valid units"

    fraction_below_zero = float(np.mean(valid < 0.0))
    mean_value = float(np.mean(valid))
    median_value = float(np.median(valid))
    if np.isclose(mean_value, 0.0, atol=5e-13, rtol=0.0):
        mean_value = 0.0
    if np.isclose(median_value, 0.0, atol=5e-13, rtol=0.0):
        median_value = 0.0
    return (
        f"Frac < 0: {fraction_below_zero:.2f}\n"
        f"Mean: {mean_value:.3f}\n"
        f"Median: {median_value:.3f}"
    )


def plot_epoch_delta_histogram(
    df: pd.DataFrame,
    epoch: str,
    *,
    region: str,
    save_path: Path,
    delta_column: str,
    figure_title: str,
    x_label: str,
    ll_columns: tuple[str, str],
    min_spikes: int = DEFAULT_MIN_PLOT_SPIKES,
    n_bins: int = 20,
) -> None:
    """Save one histogram figure for a single epoch and model contrast."""
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = filter_epoch_df(
        df,
        ll_columns=ll_columns,
        min_spikes=min_spikes,
    )
    values = filtered[delta_column].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    bin_edges = _build_zero_including_histogram_edges(values, n_bins=n_bins)

    figure, axis = plt.subplots(figsize=(6, 4.5))
    if values.size == 0:
        axis.text(0.5, 0.5, "No valid units", ha="center", va="center")
    else:
        weights = np.full(values.shape, 1.0 / values.size, dtype=float)
        axis.hist(values, bins=bin_edges, weights=weights, color="0.45", edgecolor="white")
        axis.text(
            0.98,
            0.95,
            _format_delta_histogram_stats(values),
            ha="right",
            va="top",
            transform=axis.transAxes,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.6",
                "alpha": 0.9,
            },
        )
    axis.axvline(0.0, color="k", linestyle="--", linewidth=1)
    axis.set_title(f"{epoch} (n={values.size})")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Fraction of units")

    figure.suptitle(f"{region.upper()} {figure_title}")
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_tp_transfer_family_comparison_histogram(
    df: pd.DataFrame,
    epoch: str,
    *,
    region: str,
    target_trajectory: str,
    save_path: Path,
    ll_columns: tuple[str, str],
    delta_column: str,
    x_label: str,
    min_spikes: int = DEFAULT_MIN_PLOT_SPIKES,
    n_bins: int = 20,
) -> bool:
    """Save one target-centric TP transfer-family overlay histogram."""
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    target_df = df[df["target_trajectory"] == target_trajectory]

    family_payloads: list[dict[str, Any]] = []
    combined_values: list[np.ndarray] = []
    family_colors = {
        "same_turn_cross_arm": "tab:blue",
        "opposite_turn_same_arm": "tab:orange",
        "opposite_turn_same_arm_flipped": "tab:green",
    }

    for transfer_family in TP_TRANSFER_FAMILY_ORDER:
        family_df = target_df[target_df["transfer_family"] == transfer_family]
        family_df = filter_epoch_df(
            family_df,
            ll_columns=ll_columns,
            min_spikes=min_spikes,
        )
        values = family_df[delta_column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        source_trajectories = sorted({str(value) for value in family_df["source_trajectory"].tolist()})
        family_payloads.append(
            {
                "transfer_family": transfer_family,
                "source_label": ", ".join(source_trajectories),
                "values": values,
                "color": family_colors.get(transfer_family, "0.45"),
            }
        )
        combined_values.append(values)

    if len(family_payloads) < len(TP_TRANSFER_FAMILY_ORDER):
        return False

    bin_edges = _build_zero_including_histogram_edges(
        np.concatenate(combined_values),
        n_bins=n_bins,
    )

    figure, axis = plt.subplots(figsize=(6, 4.5))
    for payload in family_payloads:
        values = np.asarray(payload["values"], dtype=float)
        weights = np.full(values.shape, 1.0 / values.size, dtype=float)
        label = (
            f"{payload['transfer_family']}: {payload['source_label']} "
            f"(n={values.size})"
        )
        axis.hist(
            values,
            bins=bin_edges,
            weights=weights,
            alpha=0.45,
            label=label,
            color=payload["color"],
            edgecolor="white",
        )

    axis.axvline(0.0, color="k", linestyle="--", linewidth=1)
    axis.set_title(f"{epoch} target={target_trajectory}")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Fraction of units")
    axis.legend(frameon=False)

    figure.suptitle(f"{region.upper()} TP transfer family comparison")
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_cv_light_dark_comparison(
    df_light: pd.DataFrame,
    df_dark: pd.DataFrame,
    *,
    light_epoch: str,
    dark_epoch: str,
    region: str,
    save_path: Path,
    x_ll_column: str,
    y_ll_column: str,
    x_label: str,
    y_label: str,
    delta_column: str,
    comparison_label: str,
    fraction_title: str,
) -> None:
    """Save a summary figure comparing one light epoch against the dark epoch."""
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    common_units = df_light.index.intersection(df_dark.index)

    mosaic = [
        ["scatter_a", "scatter_b"],
        ["paired", "fraction"],
        ["ecdf", "ecdf"],
    ]
    figure, axes = plt.subplot_mosaic(mosaic, figsize=(11, 10))

    def _scatter(df: pd.DataFrame, ax: plt.Axes, title: str) -> None:
        ax.scatter(df[x_ll_column], df[y_ll_column], s=12, alpha=0.6)
        if not df.empty:
            lower = float(np.nanmin([df[x_ll_column].min(), df[y_ll_column].min()]))
            upper = float(np.nanmax([df[x_ll_column].max(), df[y_ll_column].max()]))
            ax.plot([lower, upper], [lower, upper], "k-", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    _scatter(df_light, axes["scatter_a"], f"{light_epoch}: {comparison_label}")
    _scatter(df_dark, axes["scatter_b"], f"{dark_epoch}: {comparison_label}")

    paired_ax = axes["paired"]
    if common_units.empty:
        paired_ax.text(0.5, 0.5, "No paired units", ha="center", va="center")
        paired_ax.set_axis_off()
    else:
        delta_light = df_light.loc[common_units, delta_column].to_numpy()
        delta_dark = df_dark.loc[common_units, delta_column].to_numpy()
        order = np.argsort(delta_light)
        delta_light = delta_light[order]
        delta_dark = delta_dark[order]
        for start, end in zip(delta_light, delta_dark):
            paired_ax.plot([0, 1], [start, end], alpha=0.25)
        paired_ax.axhline(0.0, color="k", linewidth=1)
        paired_ax.scatter(
            [0, 1],
            [float(np.median(delta_light)), float(np.median(delta_dark))],
            s=90,
        )
        paired_ax.set_xticks([0, 1])
        paired_ax.set_xticklabels([light_epoch, dark_epoch])
        paired_ax.set_ylabel(delta_column)
        paired_ax.set_title("Paired change per unit")

    fraction_ax = axes["fraction"]
    p_light, lo_light, hi_light = _bootstrap_ci_fraction_positive(df_light[delta_column].to_numpy())
    p_dark, lo_dark, hi_dark = _bootstrap_ci_fraction_positive(df_dark[delta_column].to_numpy())
    yerr = [[p_light - lo_light, p_dark - lo_dark], [hi_light - p_light, hi_dark - p_dark]]
    fraction_ax.bar([light_epoch, dark_epoch], [p_light, p_dark], yerr=yerr, capsize=6)
    fraction_ax.set_ylim(0, 1)
    fraction_ax.set_ylabel(f"P({delta_column} > 0)")
    fraction_ax.set_title(fraction_title)

    ecdf_ax = axes["ecdf"]
    x_light, y_light = _ecdf(df_light[delta_column].to_numpy())
    x_dark, y_dark = _ecdf(df_dark[delta_column].to_numpy())
    ecdf_ax.plot(x_light, y_light, label=light_epoch)
    ecdf_ax.plot(x_dark, y_dark, label=dark_epoch)
    ecdf_ax.axvline(0.0, color="k", linewidth=1)
    ecdf_ax.set_xlabel(delta_column)
    ecdf_ax.set_ylabel("ECDF")
    ecdf_ax.set_title("Distribution shift (ECDF)")
    ecdf_ax.legend()

    figure.suptitle(
        f"{region.upper()} light vs dark CV comparison: {light_epoch} vs {dark_epoch} "
        f"({comparison_label})"
    )
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_epoch_tables(
    summary_tables: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
    n_folds: int,
    place_bin_size_cm: float,
) -> list[Path]:
    """Write one parquet summary table per region and run epoch."""
    saved_paths: list[Path] = []
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    for region, tables_by_epoch in summary_tables.items():
        for epoch, table in tables_by_epoch.items():
            path = (
                data_dir
                / f"{region}_{epoch}_cv{n_folds}_{place_bin_token}_encoding_summary.parquet"
            )
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def save_comparison_tables(
    comparison_tables: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
    dark_epoch: str,
    n_folds: int,
    place_bin_size_cm: float,
) -> list[Path]:
    """Write one parquet comparison table per region and light epoch."""
    saved_paths: list[Path] = []
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    for region, tables_by_light_epoch in comparison_tables.items():
        for light_epoch, table in tables_by_light_epoch.items():
            path = (
                data_dir
                / (
                    f"{region}_{light_epoch}_{dark_epoch}_cv{n_folds}_"
                    f"{place_bin_token}_encoding_comparison.parquet"
                )
            )
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def save_tp_cross_trajectory_tables(
    cross_trajectory_tables: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
    n_folds: int,
    place_bin_size_cm: float,
) -> list[Path]:
    """Write one TP cross-trajectory transfer table per region and run epoch."""
    saved_paths: list[Path] = []
    place_bin_token = format_place_bin_size_token(place_bin_size_cm)
    for region, tables_by_epoch in cross_trajectory_tables.items():
        for epoch, table in tables_by_epoch.items():
            path = (
                data_dir
                / (
                    f"{region}_{epoch}_cv{n_folds}_{place_bin_token}_"
                    "tp_cross_trajectory_encoding.parquet"
                )
            )
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the encoding comparison workflow."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare place, generalized place, task progression, and generalized "
            "task progression encoding with cross-validated Poisson likelihood"
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
        "--region",
        nargs="+",
        choices=REGIONS,
        default=list(REGIONS),
        help=f"Regions to fit. Default: {' '.join(REGIONS)}",
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
        "--generalized-place-branch-gap-cm",
        type=float,
        default=DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM,
        help=(
            "Gap inserted between left and right branches in the generalized full-W "
            "linear-position coordinate. The center stem is shared and both side arms "
            f"are present once. Default: {DEFAULT_GENERALIZED_PLACE_BRANCH_GAP_CM}"
        ),
    )
    parser.add_argument(
        "--dark-epoch",
        required=True,
        help="Run epoch to treat as the dark comparison epoch.",
    )
    parser.add_argument(
        "--light-epoch",
        action="append",
        default=None,
        help=(
            "Run epoch to treat as a light comparison epoch. Repeat to compare multiple light "
            "epochs. If omitted, all run epochs present in position.parquet except --dark-epoch "
            "are used."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Number of lap-wise cross-validation folds. Default: {DEFAULT_N_FOLDS}",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Time bin size in seconds for held-out Poisson likelihood. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--place-bin-size-cm",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
        help=(
            "Spatial bin size in cm for place and task-progression tuning curves. "
            f"Default: {DEFAULT_PLACE_BIN_SIZE_CM}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in tuning-curve bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the cross-validated encoding comparison workflow for one session."""
    args = parse_arguments()
    if args.generalized_place_branch_gap_cm < 0:
        raise ValueError("--generalized-place-branch-gap-cm must be non-negative.")
    if not np.isfinite(args.place_bin_size_cm) or args.place_bin_size_cm <= 0:
        raise ValueError("--place-bin-size-cm must be positive.")

    selected_regions = tuple(args.regions)
    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    run_epochs = require_npz_session_sources(analysis_path)
    head_position_epochs = get_head_position_epochs_from_parquet(analysis_path)
    light_epochs, dark_epoch, selected_run_epochs = resolve_light_dark_epochs(
        run_epochs,
        head_position_epochs,
        dark_epoch=args.dark_epoch,
        light_epochs=args.light_epoch,
    )

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        selected_run_epochs=selected_run_epochs,
        position_source="clean_dlc_head",
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        include_generalized_place=True,
        generalized_place_branch_gap_cm=args.generalized_place_branch_gap_cm,
    )
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    min_lap_counts = validate_fold_count(
        session["trajectory_intervals"],
        n_folds=args.n_folds,
    )
    place_bin_token = format_place_bin_size_token(args.place_bin_size_cm)
    position_bins = build_linear_position_bins(
        args.animal_name,
        args.place_bin_size_cm,
    )
    generalized_place_bins = build_generalized_place_bins(
        args.animal_name,
        branch_gap_cm=args.generalized_place_branch_gap_cm,
        place_bin_size_cm=args.place_bin_size_cm,
    )
    task_progression_bins = build_combined_task_progression_bins(
        args.animal_name,
        args.place_bin_size_cm,
    )
    single_trajectory_task_progression_bins = build_task_progression_bins(
        args.animal_name,
        args.place_bin_size_cm,
    )

    summary_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in selected_regions
    }
    tp_cross_trajectory_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in selected_regions
    }
    tp_cross_trajectory_skips: dict[str, dict[str, list[dict[str, Any]]]] = {
        region: {} for region in selected_regions
    }

    for region in selected_regions:
        for epoch in session["run_epochs"]:
            cv_by_model = run_cross_validated_encoding(
                spikes=session["spikes_by_region"][region],
                linear_position=session["linear_position_by_run"][epoch],
                generalized_place_position=session["generalized_place_position_by_run"][epoch],
                task_progression=session["task_progression_by_run"][epoch],
                generalized_task_progression=session["generalized_task_progression_by_run"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                position_bins=position_bins,
                generalized_place_bins=generalized_place_bins,
                task_progression_bins=task_progression_bins,
                generalized_task_progression_bins=single_trajectory_task_progression_bins,
                n_folds=args.n_folds,
                bin_size_s=args.bin_size_s,
                sigma_bins=args.sigma_bins,
                random_state=DEFAULT_RANDOM_STATE,
            )
            summary_tables[region][epoch] = cv_epoch_to_df(cv_by_model)
            tp_cross_trajectory_table, skipped_pairs = run_tp_cross_trajectory_encoding(
                spikes=session["spikes_by_region"][region],
                task_progression_by_trajectory=session["task_progression_by_trajectory"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                task_progression_bins=single_trajectory_task_progression_bins,
                n_folds=args.n_folds,
                bin_size_s=args.bin_size_s,
                sigma_bins=args.sigma_bins,
                random_state=DEFAULT_RANDOM_STATE,
            )
            tp_cross_trajectory_tables[region][epoch] = tp_cross_trajectory_table
            tp_cross_trajectory_skips[region][epoch] = skipped_pairs

    comparison_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in selected_regions
    }
    for region in selected_regions:
        for light_epoch in light_epochs:
            comparison_tables[region][light_epoch] = build_comparison_table(
                summary_tables[region][light_epoch],
                summary_tables[region][dark_epoch],
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
            )

    saved_epoch_tables = save_epoch_tables(
        summary_tables,
        data_dir=data_dir,
        n_folds=args.n_folds,
        place_bin_size_cm=args.place_bin_size_cm,
    )
    saved_comparison_tables = save_comparison_tables(
        comparison_tables,
        data_dir=data_dir,
        dark_epoch=dark_epoch,
        n_folds=args.n_folds,
        place_bin_size_cm=args.place_bin_size_cm,
    )
    saved_tp_cross_trajectory_tables = save_tp_cross_trajectory_tables(
        tp_cross_trajectory_tables,
        data_dir=data_dir,
        n_folds=args.n_folds,
        place_bin_size_cm=args.place_bin_size_cm,
    )

    saved_figures: list[Path] = []
    for region in selected_regions:
        for epoch in session["run_epochs"]:
            histogram_path = (
                fig_dir
                / f"{region}_{epoch}_cv_{place_bin_token}_place_vs_tp_histogram.png"
            )
            plot_epoch_delta_histogram(
                summary_tables[region][epoch],
                epoch,
                region=region,
                save_path=histogram_path,
                ll_columns=("ll_place", "ll_tp"),
                delta_column="delta_bits_place_vs_tp",
                figure_title="place - TP histogram",
                x_label="place - TP (bits/spike)",
            )
            saved_figures.append(histogram_path)

            histogram_path = (
                fig_dir
                / (
                    f"{region}_{epoch}_cv_{place_bin_token}_"
                    "generalized_place_vs_tp_histogram.png"
                )
            )
            plot_epoch_delta_histogram(
                summary_tables[region][epoch],
                epoch,
                region=region,
                save_path=histogram_path,
                ll_columns=("ll_generalized_place", "ll_tp"),
                delta_column="delta_bits_generalized_place_vs_tp",
                figure_title="generalized place - TP histogram",
                x_label="generalized place - TP (bits/spike)",
            )
            saved_figures.append(histogram_path)

            histogram_path = (
                fig_dir / f"{region}_{epoch}_cv_{place_bin_token}_gtp_vs_tp_histogram.png"
            )
            plot_epoch_delta_histogram(
                summary_tables[region][epoch],
                epoch,
                region=region,
                save_path=histogram_path,
                ll_columns=("ll_gtp", "ll_tp"),
                delta_column="delta_bits_gtp_vs_tp",
                figure_title="generalized TP - TP histogram",
                x_label="generalized TP - TP (bits/spike)",
            )
            saved_figures.append(histogram_path)

            for target_trajectory in TRAJECTORY_TYPES:
                histogram_path = (
                    fig_dir
                    / (
                        f"{region}_{epoch}_{target_trajectory}_{place_bin_token}_"
                        "tp_transfer_family_comparison_histogram.png"
                    )
                )
                saved = plot_tp_transfer_family_comparison_histogram(
                    tp_cross_trajectory_tables[region][epoch],
                    epoch,
                    region=region,
                    target_trajectory=target_trajectory,
                    save_path=histogram_path,
                    ll_columns=("ll_tp_cross", "ll_tp_cv_target"),
                    delta_column="delta_bits_cross_vs_cv",
                    x_label="cross - target CV (bits/spike)",
                )
                if saved:
                    saved_figures.append(histogram_path)

        filtered_dark_place_tp = filter_epoch_df(
            summary_tables[region][dark_epoch],
            ll_columns=("ll_place", "ll_tp"),
        )
        filtered_dark_generalized_place_tp = filter_epoch_df(
            summary_tables[region][dark_epoch],
            ll_columns=("ll_generalized_place", "ll_tp"),
        )
        filtered_dark_gtp_tp = filter_epoch_df(
            summary_tables[region][dark_epoch],
            ll_columns=("ll_gtp", "ll_tp"),
        )
        for light_epoch in light_epochs:
            filtered_light_place_tp = filter_epoch_df(
                summary_tables[region][light_epoch],
                ll_columns=("ll_place", "ll_tp"),
            )
            figure_path = (
                fig_dir
                / f"{region}_{light_epoch}_vs_{dark_epoch}_cv_{place_bin_token}_place_vs_tp.png"
            )
            plot_cv_light_dark_comparison(
                filtered_light_place_tp,
                filtered_dark_place_tp,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                region=region,
                save_path=figure_path,
                x_ll_column="ll_tp",
                y_ll_column="ll_place",
                x_label="LL_TP (per spike)",
                y_label="LL_place (per spike)",
                delta_column="delta_bits_place_vs_tp",
                comparison_label="LL_place vs LL_TP",
                fraction_title="Fraction of units favoring place",
            )
            saved_figures.append(figure_path)

            filtered_light_generalized_place_tp = filter_epoch_df(
                summary_tables[region][light_epoch],
                ll_columns=("ll_generalized_place", "ll_tp"),
            )
            figure_path = (
                fig_dir
                / (
                    f"{region}_{light_epoch}_vs_{dark_epoch}_cv_{place_bin_token}_"
                    "generalized_place_vs_tp.png"
                )
            )
            plot_cv_light_dark_comparison(
                filtered_light_generalized_place_tp,
                filtered_dark_generalized_place_tp,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                region=region,
                save_path=figure_path,
                x_ll_column="ll_tp",
                y_ll_column="ll_generalized_place",
                x_label="LL_TP (per spike)",
                y_label="LL_generalized place (per spike)",
                delta_column="delta_bits_generalized_place_vs_tp",
                comparison_label="LL_generalized place vs LL_TP",
                fraction_title="Fraction of units favoring generalized place",
            )
            saved_figures.append(figure_path)

            filtered_light_gtp_tp = filter_epoch_df(
                summary_tables[region][light_epoch],
                ll_columns=("ll_gtp", "ll_tp"),
            )
            figure_path = (
                fig_dir
                / f"{region}_{light_epoch}_vs_{dark_epoch}_cv_{place_bin_token}_gtp_vs_tp.png"
            )
            plot_cv_light_dark_comparison(
                filtered_light_gtp_tp,
                filtered_dark_gtp_tp,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                region=region,
                save_path=figure_path,
                x_ll_column="ll_tp",
                y_ll_column="ll_gtp",
                x_label="LL_TP (per spike)",
                y_label="LL_generalized TP (per spike)",
                delta_column="delta_bits_gtp_vs_tp",
                comparison_label="LL_generalized TP vs LL_TP",
                fraction_title="Fraction of units favoring generalized TP",
            )
            saved_figures.append(figure_path)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.encoding_comparison",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": list(selected_regions),
            "light_epochs": list(light_epochs),
            "dark_epoch": dark_epoch,
            "selected_run_epochs": selected_run_epochs,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "generalized_place_branch_gap_cm": args.generalized_place_branch_gap_cm,
            "n_folds": args.n_folds,
            "bin_size_s": args.bin_size_s,
            "place_bin_size_cm": args.place_bin_size_cm,
            "sigma_bins": args.sigma_bins,
            "random_state": DEFAULT_RANDOM_STATE,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "min_lap_counts": min_lap_counts,
            "saved_epoch_tables": saved_epoch_tables,
            "saved_comparison_tables": saved_comparison_tables,
            "saved_tp_cross_trajectory_tables": saved_tp_cross_trajectory_tables,
            "saved_figures": saved_figures,
            "tp_cross_trajectory_skips": tp_cross_trajectory_skips,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
