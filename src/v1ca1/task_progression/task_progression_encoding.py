from __future__ import annotations

"""Compare place and task-progression encoding with cross-validated Poisson likelihood.

This script loads one dataset through shared task-progression utilities in
`v1ca1.task_progression._session`, where the relevant preprocessing is built,
including movement intervals, linearized position, and task progression. It
then rebuilds cross-validated tuning curves on movement-restricted data and
compares two encoding models for each unit in each run epoch:

- concatenated four-trajectory linear position
- combined two-branch task progression

Held-out log-likelihood is reported for two cases:

- model: the feature-dependent encoding model defined by the training-fold
  tuning curve, using either linear position or task progression as the
  feature
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
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    build_combined_task_progression_bins,
    build_linear_position_bins,
    get_analysis_path,
    prepare_task_progression_session,
)


DEFAULT_N_FOLDS = 5
DEFAULT_BIN_SIZE_S = 0.02
DEFAULT_SIGMA_BINS = 1.0
DEFAULT_RANDOM_STATE = 47
DEFAULT_MIN_PLOT_SPIKES = 50


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


def run_cross_validated_encoding(
    spikes: Any,
    linear_position: Any,
    task_progression: Any,
    movement_interval: Any,
    trajectory_intervals: dict[str, Any],
    position_bins: np.ndarray,
    task_progression_bins: np.ndarray,
    n_folds: int,
    bin_size_s: float,
    sigma_bins: float,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]]]:
    """Fit and score place and task-progression models across CV folds."""
    import pynapple as nap

    unit_ids = list(spikes.keys())
    cv_place = initialize_cv_store(unit_ids, n_folds=n_folds)
    cv_tp = initialize_cv_store(unit_ids, n_folds=n_folds)
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

        place_tuning = nap.compute_tuning_curves(
            data=spikes,
            features=linear_position,
            bins=[position_bins],
            epochs=train_fold,
            feature_names=["linpos"],
        )
        place_tuning = smooth_pf_along_position_nan_aware(
            place_tuning,
            pos_dim="linpos",
            sigma_bins=sigma_bins,
        )
        task_progression_tuning = nap.compute_tuning_curves(
            data=spikes,
            features=task_progression,
            bins=[task_progression_bins],
            epochs=train_fold,
            feature_names=["tp"],
        )
        task_progression_tuning = smooth_pf_along_position_nan_aware(
            task_progression_tuning,
            pos_dim="tp",
            sigma_bins=sigma_bins,
        )

        for unit_id in unit_ids:
            unit_spikes = spikes[unit_id]
            n_train_spikes = int(len(unit_spikes.restrict(train_fold).t))
            train_rate = max(n_train_spikes / train_duration, 1e-10)

            place_curve = place_tuning.sel(unit=unit_id)
            ll_model_place, ll_null_place, n_spikes_place = poisson_ll_on_epoch(
                unit_spikes=unit_spikes,
                position=linear_position,
                tuning_curve=np.asarray(place_curve.values, dtype=float),
                bin_edges=np.asarray(position_bins, dtype=float),
                epoch=test_fold,
                bin_size_s=bin_size_s,
                epsilon=1e-10,
                fill_rate=train_rate,
                null_rate=train_rate,
            )
            cv_place[unit_id]["fold_ll_model"][fold] = ll_model_place
            cv_place[unit_id]["fold_ll_null"][fold] = ll_null_place
            cv_place[unit_id]["fold_n_spikes"][fold] = n_spikes_place

            tp_curve = task_progression_tuning.sel(unit=unit_id)
            ll_model_tp, ll_null_tp, n_spikes_tp = poisson_ll_on_epoch(
                unit_spikes=unit_spikes,
                position=task_progression,
                tuning_curve=np.asarray(tp_curve.values, dtype=float),
                bin_edges=np.asarray(task_progression_bins, dtype=float),
                epoch=test_fold,
                bin_size_s=bin_size_s,
                epsilon=1e-10,
                fill_rate=train_rate,
                null_rate=train_rate,
            )
            cv_tp[unit_id]["fold_ll_model"][fold] = ll_model_tp
            cv_tp[unit_id]["fold_ll_null"][fold] = ll_null_tp
            cv_tp[unit_id]["fold_n_spikes"][fold] = n_spikes_tp

    aggregate_cv_store(cv_place)
    aggregate_cv_store(cv_tp)
    return cv_place, cv_tp


def cv_epoch_to_df(
    cv_place: dict[Any, dict[str, Any]],
    cv_tp: dict[Any, dict[str, Any]],
) -> pd.DataFrame:
    """Convert legacy nested CV results into one unit-indexed summary table."""
    rows: list[dict[str, Any]] = []
    for unit_id, place_metrics in cv_place.items():
        tp_metrics = cv_tp[unit_id]
        ll_place = float(place_metrics.get("ll_model_per_spike_cv", np.nan))
        ll_tp = float(tp_metrics.get("ll_model_per_spike_cv", np.nan))
        ll_null_place = float(place_metrics.get("ll_null_per_spike_cv", np.nan))
        ll_null_tp = float(tp_metrics.get("ll_null_per_spike_cv", np.nan))
        info_place = float(place_metrics.get("info_bits_per_spike_cv", np.nan))
        info_tp = float(tp_metrics.get("info_bits_per_spike_cv", np.nan))
        delta_info_bits = (
            info_place - info_tp
            if np.isfinite(info_place) and np.isfinite(info_tp)
            else np.nan
        )
        rows.append(
            {
                "unit": unit_id,
                "n_spikes": int(np.sum(np.asarray(place_metrics.get("fold_n_spikes", 0)))),
                "ll_place": ll_place,
                "ll_tp": ll_tp,
                "ll_null_place": ll_null_place,
                "ll_null_tp": ll_null_tp,
                "info_bits_place": info_place,
                "info_bits_tp": info_tp,
                "delta_bits": (ll_place - ll_tp) / np.log(2.0)
                if np.isfinite(ll_place) and np.isfinite(ll_tp)
                else np.nan,
                "delta_info_bits": delta_info_bits,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "n_spikes",
                "ll_place",
                "ll_tp",
                "ll_null_place",
                "ll_null_tp",
                "info_bits_place",
                "info_bits_tp",
                "delta_bits",
                "delta_info_bits",
            ],
            index=pd.Index([], name="unit"),
        )
    return pd.DataFrame(rows).set_index("unit").sort_index()


def filter_epoch_df(df: pd.DataFrame, min_spikes: int = DEFAULT_MIN_PLOT_SPIKES) -> pd.DataFrame:
    """Filter units with invalid LL estimates or too few held-out spikes for plots."""
    filtered = df.copy()
    filtered = filtered[np.isfinite(filtered["ll_place"]) & np.isfinite(filtered["ll_tp"])]
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
    comparison["delta_bits_change"] = comparison["dark_delta_bits"] - comparison["light_delta_bits"]
    comparison["delta_info_bits_change"] = (
        comparison["dark_delta_info_bits"] - comparison["light_delta_info_bits"]
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


def plot_cv_light_dark_comparison(
    df_light: pd.DataFrame,
    df_dark: pd.DataFrame,
    *,
    light_epoch: str,
    dark_epoch: str,
    region: str,
    save_path: Path,
    delta_column: str = "delta_bits",
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
        ax.scatter(df["ll_tp"], df["ll_place"], s=12, alpha=0.6)
        if not df.empty:
            lower = float(np.nanmin([df["ll_tp"].min(), df["ll_place"].min()]))
            upper = float(np.nanmax([df["ll_tp"].max(), df["ll_place"].max()]))
            ax.plot([lower, upper], [lower, upper], "k-", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("LL_tp (per spike)")
        ax.set_ylabel("LL_place (per spike)")

    _scatter(df_light, axes["scatter_a"], f"{light_epoch}: LL_place vs LL_tp")
    _scatter(df_dark, axes["scatter_b"], f"{dark_epoch}: LL_place vs LL_tp")

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
    fraction_ax.set_title("Fraction of units favoring place")

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

    figure.suptitle(f"{region.upper()} light vs dark CV comparison: {light_epoch} vs {dark_epoch}")
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_epoch_tables(
    summary_tables: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
    n_folds: int,
) -> list[Path]:
    """Write one parquet summary table per region and run epoch."""
    saved_paths: list[Path] = []
    for region, tables_by_epoch in summary_tables.items():
        for epoch, table in tables_by_epoch.items():
            path = data_dir / f"{region}_{epoch}_cv{n_folds}_encoding_summary.parquet"
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def save_comparison_tables(
    comparison_tables: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
    dark_epoch: str,
    n_folds: int,
) -> list[Path]:
    """Write one parquet comparison table per region and light epoch."""
    saved_paths: list[Path] = []
    for region, tables_by_light_epoch in comparison_tables.items():
        for light_epoch, table in tables_by_light_epoch.items():
            path = (
                data_dir
                / f"{region}_{light_epoch}_{dark_epoch}_cv{n_folds}_encoding_comparison.parquet"
            )
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the encoding comparison workflow."""
    parser = argparse.ArgumentParser(
        description="Compare place and task-progression encoding with cross-validated Poisson likelihood"
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
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in tuning-curve bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the cross-validated encoding comparison workflow for one session."""
    args = parse_arguments()
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
    )
    data_dir = analysis_path / "task_progression_encoding"
    fig_dir = analysis_path / "figs" / "task_progression_encoding"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    min_lap_counts = validate_fold_count(
        session["trajectory_intervals"],
        n_folds=args.n_folds,
    )
    position_bins = build_linear_position_bins(args.animal_name)
    task_progression_bins = build_combined_task_progression_bins(args.animal_name)

    summary_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in selected_regions
    }

    for region in selected_regions:
        for epoch in session["run_epochs"]:
            cv_place, cv_tp = run_cross_validated_encoding(
                spikes=session["spikes_by_region"][region],
                linear_position=session["linear_position_by_run"][epoch],
                task_progression=session["task_progression_by_run"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                position_bins=position_bins,
                task_progression_bins=task_progression_bins,
                n_folds=args.n_folds,
                bin_size_s=args.bin_size_s,
                sigma_bins=args.sigma_bins,
                random_state=DEFAULT_RANDOM_STATE,
            )
            summary_tables[region][epoch] = cv_epoch_to_df(cv_place, cv_tp)

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

    saved_epoch_tables = save_epoch_tables(summary_tables, data_dir=data_dir, n_folds=args.n_folds)
    saved_comparison_tables = save_comparison_tables(
        comparison_tables,
        data_dir=data_dir,
        dark_epoch=dark_epoch,
        n_folds=args.n_folds,
    )

    saved_figures: list[Path] = []
    for region in selected_regions:
        filtered_dark = filter_epoch_df(summary_tables[region][dark_epoch])
        for light_epoch in light_epochs:
            filtered_light = filter_epoch_df(summary_tables[region][light_epoch])
            figure_path = fig_dir / f"{region}_{light_epoch}_vs_{dark_epoch}_cv.png"
            plot_cv_light_dark_comparison(
                filtered_light,
                filtered_dark,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                region=region,
                save_path=figure_path,
                delta_column="delta_bits",
            )
            saved_figures.append(figure_path)

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.task_progression_encoding",
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
            "n_folds": args.n_folds,
            "bin_size_s": args.bin_size_s,
            "sigma_bins": args.sigma_bins,
            "random_state": DEFAULT_RANDOM_STATE,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "min_lap_counts": min_lap_counts,
            "saved_epoch_tables": saved_epoch_tables,
            "saved_comparison_tables": saved_comparison_tables,
            "saved_figures": saved_figures,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
