from __future__ import annotations

"""Compare place and task-progression decoding with lap-wise cross-validation.

This script loads one dataset through shared task-progression utilities in
`v1ca1.task_progression._session`, where movement intervals, linearized
position, task progression, and trajectory-specific task progression are built.
It then runs two decoding workflows on movement-restricted data:

- within-epoch cross-validated decoding of concatenated linear position
- within-epoch cross-validated decoding of combined task progression

It also keeps the cross-trajectory task-progression decoder, where tuning
curves fit on one trajectory are used to decode the paired same-turn
trajectory.

Primary time-series outputs are saved as pynapple-backed `.npz` files, because
decoded and true trajectories are time-domain artifacts. Per-epoch decoding
metrics, light-vs-dark comparisons, and binned error summaries are saved as
parquet tables. The binned error plots can summarize either signed or absolute
error, with either mean +/- std or median + IQR error bars. Light-vs-dark
figures use fixed epochs `02_r1`, `04_r2`, and `06_r3` versus `08_r4`. A run
log is written under
`analysis_path / "v1ca1_log"`.
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    build_combined_task_progression_bins,
    build_linear_position_bins,
    build_task_progression_bins,
    compute_movement_firing_rates,
    get_analysis_path,
    prepare_task_progression_session,
)


DEFAULT_N_FOLDS = 5
DEFAULT_BIN_SIZE_S = 0.02
DEFAULT_SLIDING_WINDOW_SIZE_BINS = 4
DEFAULT_RANDOM_STATE = 47
DEFAULT_CROSS_TRAJ_FR_THRESHOLD_HZ = 0.5
DEFAULT_MIN_BIN_COUNT = 5
DEFAULT_DARK_EPOCH = "08_r4"
DEFAULT_LIGHT_EPOCHS = ("02_r1", "04_r2", "06_r3")
SUMMARY_MODES = ("mean_std", "median_iqr")
ERROR_MODES = ("signed", "absolute")
CROSS_TRAJECTORY_DECODING_MAP = {
    "center_to_left": "right_to_center",
    "right_to_center": "center_to_left",
    "center_to_right": "left_to_center",
    "left_to_center": "center_to_right",
}


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
        return nap.IntervalSet(
            start=np.array([], dtype=float),
            end=np.array([], dtype=float),
            time_units="s",
        )

    start_array = np.concatenate(start_chunks).astype(float, copy=False)
    end_array = np.concatenate(end_chunks).astype(float, copy=False)
    order = np.argsort(start_array)
    return nap.IntervalSet(
        start=start_array[order],
        end=end_array[order],
        time_units="s",
    )


def _make_empty_tsd(time_support: Any | None = None) -> Any:
    """Return an empty Tsd with second-based timestamps."""
    import pynapple as nap

    kwargs: dict[str, Any] = {"time_units": "s"}
    if time_support is not None:
        kwargs["time_support"] = time_support
    return nap.Tsd(
        t=np.array([], dtype=float),
        d=np.array([], dtype=float),
        **kwargs,
    )


def _concatenate_tsds(tsds: list[Any], time_support: Any) -> Any:
    """Concatenate Tsds and return one sorted Tsd."""
    import pynapple as nap

    if not tsds:
        return _make_empty_tsd(time_support=time_support)

    times = [np.asarray(tsd.t, dtype=float) for tsd in tsds if len(np.asarray(tsd.t)) > 0]
    values = [np.asarray(tsd.d, dtype=float) for tsd in tsds if len(np.asarray(tsd.t)) > 0]
    if not times:
        return _make_empty_tsd(time_support=time_support)

    all_times = np.concatenate(times)
    all_values = np.concatenate(values)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=time_support,
        time_units="s",
    )


def get_light_dark_epochs(
    run_epochs: list[str],
    dark_epoch: str = DEFAULT_DARK_EPOCH,
    light_epochs: tuple[str, ...] = DEFAULT_LIGHT_EPOCHS,
) -> tuple[tuple[str, ...], str]:
    """Return validated fixed light and dark run epochs."""
    if not run_epochs:
        raise ValueError("No run epochs were found for this session.")

    requested_epochs = [*light_epochs, dark_epoch]
    missing = [epoch for epoch in requested_epochs if epoch not in run_epochs]
    if missing:
        raise ValueError(
            "Requested light/dark epochs were not found in run epochs "
            f"{run_epochs!r}: {missing!r}"
        )
    return light_epochs, dark_epoch


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
    """Validate that each epoch supports the requested lap-wise fold count."""
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
        fold: _make_intervalset(train_starts[fold], train_ends[fold])
        for fold in range(n_folds)
    }
    test_folds = {
        fold: _make_intervalset(test_starts[fold], test_ends[fold])
        for fold in range(n_folds)
    }
    return train_folds, test_folds


def _get_active_unit_ids(
    spikes: Any,
    firing_rates: np.ndarray,
    threshold_hz: float,
) -> list[Any]:
    """Return unit ids above the requested firing-rate threshold."""
    unit_ids = list(spikes.keys())
    return [
        unit_id
        for unit_id, rate in zip(unit_ids, np.asarray(firing_rates, dtype=float), strict=False)
        if np.isfinite(rate) and float(rate) > threshold_hz
    ]


def _subset_spikes(spikes: Any, unit_ids: list[Any]) -> Any:
    """Return a TsGroup restricted to the requested unit ids."""
    import pynapple as nap

    return nap.TsGroup({unit_id: spikes[unit_id] for unit_id in unit_ids}, time_units="s")


def decode_cv(
    spikes: Any,
    feature_by_epoch: dict[str, Any],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    epoch: str,
    bins: np.ndarray,
    n_folds: int,
    bin_size_s: float,
    sliding_window_size_bins: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[Any, Any]:
    """Decode one feature with lap-wise train/test folds for one epoch."""
    import pynapple as nap

    train_folds, test_folds = build_train_test_folds(
        trajectory_intervals,
        n_folds=n_folds,
        random_state=random_state,
    )
    decoded_chunks: list[Any] = []
    true_chunks: list[Any] = []

    for fold in range(n_folds):
        train_fold = train_folds[fold].intersect(movement_interval)
        test_fold = test_folds[fold].intersect(movement_interval)
        if float(train_fold.tot_length()) <= 0.0 or float(test_fold.tot_length()) <= 0.0:
            continue

        tuning_curves = nap.compute_tuning_curves(
            data=spikes,
            features=feature_by_epoch[epoch],
            bins=[bins],
            epochs=train_fold,
        )
        decoded, _ = nap.decode_bayes(
            tuning_curves=tuning_curves,
            data=spikes,
            epochs=test_fold,
            sliding_window_size=sliding_window_size_bins,
            bin_size=bin_size_s,
        )
        decoded_chunks.append(decoded)
        true_chunks.append(feature_by_epoch[epoch].restrict(test_fold))

    return (
        _concatenate_tsds(true_chunks, movement_interval),
        _concatenate_tsds(decoded_chunks, movement_interval),
    )


def decode_task_cross_trajectory(
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    epoch: str,
    bins: np.ndarray,
    active_unit_ids: list[Any],
    bin_size_s: float,
    sliding_window_size_bins: int,
) -> tuple[dict[tuple[str, str], Any], dict[tuple[str, str], Any]]:
    """Decode one trajectory using task-progression tuning from the paired turn."""
    import pynapple as nap

    true_by_pair: dict[tuple[str, str], Any] = {}
    decoded_by_pair: dict[tuple[str, str], Any] = {}
    if not active_unit_ids:
        for encoding_trajectory, decoding_trajectory in CROSS_TRAJECTORY_DECODING_MAP.items():
            key = (encoding_trajectory, decoding_trajectory)
            empty = _make_empty_tsd()
            true_by_pair[key] = empty
            decoded_by_pair[key] = empty
        return true_by_pair, decoded_by_pair

    selected_spikes = _subset_spikes(spikes, active_unit_ids)
    for encoding_trajectory, decoding_trajectory in CROSS_TRAJECTORY_DECODING_MAP.items():
        key = (encoding_trajectory, decoding_trajectory)
        train_epoch = trajectory_intervals[encoding_trajectory].intersect(movement_interval)
        test_epoch = trajectory_intervals[decoding_trajectory].intersect(movement_interval)
        if float(train_epoch.tot_length()) <= 0.0 or float(test_epoch.tot_length()) <= 0.0:
            true_by_pair[key] = _make_empty_tsd(time_support=test_epoch)
            decoded_by_pair[key] = _make_empty_tsd(time_support=test_epoch)
            continue

        tuning_curves = nap.compute_tuning_curves(
            data=selected_spikes,
            features=task_progression_by_trajectory[epoch][encoding_trajectory],
            bins=[bins],
            epochs=train_epoch,
        )
        decoded, _ = nap.decode_bayes(
            tuning_curves=tuning_curves,
            data=selected_spikes,
            epochs=test_epoch,
            sliding_window_size=sliding_window_size_bins,
            bin_size=bin_size_s,
        )
        true_by_pair[key] = task_progression_by_trajectory[epoch][decoding_trajectory].restrict(test_epoch)
        decoded_by_pair[key] = decoded

    return true_by_pair, decoded_by_pair


def align_true_to_decoded(true_tsd: Any, decoded_tsd: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned true and decoded values on decoded timestamps."""
    if len(np.asarray(decoded_tsd.t)) == 0 or len(np.asarray(true_tsd.t)) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    ep = true_tsd.time_support.intersect(decoded_tsd.time_support)
    true_restricted = true_tsd.restrict(ep)
    decoded_restricted = decoded_tsd.restrict(ep)
    true_at_decoded = true_restricted.interpolate(
        decoded_restricted,
        ep=ep,
        left=np.nan,
        right=np.nan,
    )
    true_values = np.asarray(true_at_decoded.d, dtype=float)
    decoded_values = np.asarray(decoded_restricted.d, dtype=float)
    valid = np.isfinite(true_values) & np.isfinite(decoded_values)
    return true_values[valid], decoded_values[valid]


def get_error_ylabel(error_mode: str, summary: str) -> str:
    """Return the y-axis label for one decoding-error summary plot."""
    if error_mode == "signed":
        ylabel = "Decoded - true"
    else:
        ylabel = "|Decoded - true|"

    if summary == "mean_std":
        return f"{ylabel} (mean +/- std)"
    return f"{ylabel} (median, IQR)"


def summarize_decoding_metrics(true_tsd: Any, decoded_tsd: Any) -> dict[str, float | int]:
    """Return global decoding error metrics for one decoded trajectory."""
    true_values, decoded_values = align_true_to_decoded(true_tsd, decoded_tsd)
    if true_values.size == 0:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "mean_signed_error": np.nan,
            "median_abs_error": np.nan,
            "n_samples": 0,
        }

    signed_error = decoded_values - true_values
    abs_error = np.abs(signed_error)
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(signed_error**2))),
        "mean_signed_error": float(np.mean(signed_error)),
        "median_abs_error": float(np.median(abs_error)),
        "n_samples": int(signed_error.size),
    }


def summarize_decoding_error_by_position(
    true_tsd: Any,
    decoded_tsd: Any,
    *,
    bin_edges: np.ndarray,
    error_mode: str = "signed",
    summary: str = "median_iqr",
    min_count: int = DEFAULT_MIN_BIN_COUNT,
) -> pd.DataFrame:
    """Return a binned decoding-error summary along true position."""
    if error_mode not in ERROR_MODES:
        raise ValueError(f"Unsupported error mode {error_mode!r}. Expected one of {ERROR_MODES!r}.")
    if summary not in SUMMARY_MODES:
        raise ValueError(f"Unsupported summary mode {summary!r}. Expected one of {SUMMARY_MODES!r}.")

    true_values, decoded_values = align_true_to_decoded(true_tsd, decoded_tsd)
    bin_edges = np.asarray(bin_edges, dtype=float)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rows: list[dict[str, Any]] = []

    if true_values.size == 0:
        for bin_left, bin_right, center in zip(bin_edges[:-1], bin_edges[1:], centers, strict=False):
            rows.append(
                {
                    "bin_left": float(bin_left),
                    "bin_right": float(bin_right),
                    "bin_center": float(center),
                    "n": 0,
                    "center": np.nan,
                    "yerr_low": np.nan,
                    "yerr_high": np.nan,
                }
            )
        return pd.DataFrame(rows)

    error = decoded_values - true_values
    if error_mode == "absolute":
        error = np.abs(error)

    bin_indices = np.digitize(true_values, bin_edges) - 1
    in_range = (bin_indices >= 0) & (bin_indices < centers.size)
    true_values = true_values[in_range]
    error = error[in_range]
    bin_indices = bin_indices[in_range]

    for index, (bin_left, bin_right, center) in enumerate(
        zip(bin_edges[:-1], bin_edges[1:], centers, strict=False)
    ):
        bin_error = error[bin_indices == index]
        count = int(bin_error.size)
        if count < min_count:
            rows.append(
                {
                    "bin_left": float(bin_left),
                    "bin_right": float(bin_right),
                    "bin_center": float(center),
                    "n": count,
                    "center": np.nan,
                    "yerr_low": np.nan,
                    "yerr_high": np.nan,
                }
            )
            continue

        if summary == "mean_std":
            center_value = float(np.mean(bin_error))
            spread = float(np.std(bin_error, ddof=1)) if count > 1 else np.nan
            yerr_low = spread
            yerr_high = spread
        else:
            center_value = float(np.median(bin_error))
            q1, q3 = np.quantile(bin_error, [0.25, 0.75]).astype(float)
            yerr_low = center_value - q1
            yerr_high = q3 - center_value

        rows.append(
            {
                "bin_left": float(bin_left),
                "bin_right": float(bin_right),
                "bin_center": float(center),
                "n": count,
                "center": center_value,
                "yerr_low": yerr_low,
                "yerr_high": yerr_high,
            }
        )
    return pd.DataFrame(rows)


def build_metric_comparison_table(
    light_metrics: pd.DataFrame,
    dark_metrics: pd.DataFrame,
    *,
    light_epoch: str,
    dark_epoch: str,
    join_columns: list[str],
) -> pd.DataFrame:
    """Join light and dark metric tables and add metric deltas."""
    comparison = light_metrics.merge(
        dark_metrics,
        on=join_columns,
        suffixes=("_light", "_dark"),
        how="inner",
    )
    comparison.insert(0, "dark_epoch", dark_epoch)
    comparison.insert(0, "light_epoch", light_epoch)
    for metric in ("mae", "rmse", "mean_signed_error", "median_abs_error", "n_samples"):
        comparison[f"{metric}_change"] = (
            comparison[f"{metric}_dark"] - comparison[f"{metric}_light"]
        )
    return comparison


def build_binned_error_comparison_table(
    light_summary: pd.DataFrame,
    dark_summary: pd.DataFrame,
    *,
    light_epoch: str,
    dark_epoch: str,
) -> pd.DataFrame:
    """Join light and dark binned decoding-error summaries."""
    comparison = light_summary.merge(
        dark_summary,
        on=["bin_left", "bin_right", "bin_center"],
        suffixes=("_light", "_dark"),
        how="outer",
    ).sort_values("bin_center")
    comparison.insert(0, "dark_epoch", dark_epoch)
    comparison.insert(0, "light_epoch", light_epoch)
    return comparison


def plot_binned_error_comparison(
    comparison_table: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    error_mode: str,
    summary: str,
    light_epoch: str,
    dark_epoch: str,
    save_path: Path,
) -> None:
    """Plot light and dark binned decoding error on the same axes."""
    figure, axis = plt.subplots(figsize=(12, 3))
    for prefix, color, label in (
        ("light", "tab:orange", light_epoch),
        ("dark", "tab:gray", dark_epoch),
    ):
        valid = (
            np.isfinite(comparison_table[f"center_{prefix}"])
            & np.isfinite(comparison_table[f"yerr_low_{prefix}"])
            & np.isfinite(comparison_table[f"yerr_high_{prefix}"])
        )
        axis.errorbar(
            comparison_table.loc[valid, "bin_center"],
            comparison_table.loc[valid, f"center_{prefix}"],
            yerr=np.vstack(
                [
                    comparison_table.loc[valid, f"yerr_low_{prefix}"],
                    comparison_table.loc[valid, f"yerr_high_{prefix}"],
                ]
            ),
            fmt="o-",
            lw=1.0,
            ms=4,
            capsize=2,
            color=color,
            label=label,
        )

    if error_mode == "signed":
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(get_error_ylabel(error_mode, summary))
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_epoch_error_profiles(
    summaries_by_epoch: dict[str, pd.DataFrame],
    *,
    epoch_order: list[str],
    title: str,
    xlabel: str,
    error_mode: str,
    summary: str,
    save_path: Path,
) -> None:
    """Plot one binned decoding-error profile for each epoch on shared axes."""
    figure, axis = plt.subplots(figsize=(12, 3))
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(epoch_order), 1)))

    for color, epoch in zip(colors, epoch_order, strict=False):
        table = summaries_by_epoch[epoch]
        valid = (
            np.isfinite(table["center"])
            & np.isfinite(table["yerr_low"])
            & np.isfinite(table["yerr_high"])
        )
        axis.errorbar(
            table.loc[valid, "bin_center"],
            table.loc[valid, "center"],
            yerr=np.vstack(
                [
                    table.loc[valid, "yerr_low"],
                    table.loc[valid, "yerr_high"],
                ]
            ),
            fmt="o-",
            lw=1.0,
            ms=4,
            capsize=2,
            color=color,
            label=epoch,
        )

    if error_mode == "signed":
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(get_error_ylabel(error_mode, summary))
    axis.set_title(title)
    axis.legend(frameon=False, ncols=min(4, max(len(epoch_order), 1)))
    axis.grid(True, alpha=0.2)

    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_cross_trajectory_epoch_error_profiles(
    summaries_by_epoch: dict[str, dict[tuple[str, str], pd.DataFrame]],
    *,
    epoch_order: list[str],
    encoding_trajectory: str,
    decoding_trajectory: str,
    title: str,
    xlabel: str,
    error_mode: str,
    summary: str,
    save_path: Path,
) -> None:
    """Plot epoch-wise cross-trajectory decoding-error profiles for one trajectory pair."""
    key = (encoding_trajectory, decoding_trajectory)
    plot_epoch_error_profiles(
        {epoch: summaries_by_epoch[epoch][key] for epoch in epoch_order},
        epoch_order=epoch_order,
        title=title,
        xlabel=xlabel,
        error_mode=error_mode,
        summary=summary,
        save_path=save_path,
    )


def save_within_epoch_tsds(
    true_by_region: dict[str, dict[str, Any]],
    decoded_by_region: dict[str, dict[str, Any]],
    *,
    model_name: str,
    save_dir: Path,
) -> list[Path]:
    """Save within-epoch decoded and true Tsds as `.npz` files."""
    saved_paths: list[Path] = []
    for region, values_by_epoch in true_by_region.items():
        for epoch, true_tsd in values_by_epoch.items():
            true_path = save_dir / f"{region}_{epoch}_true_{model_name}.npz"
            true_tsd.save(true_path)
            saved_paths.append(true_path)

            decoded_path = save_dir / f"{region}_{epoch}_decoded_{model_name}.npz"
            decoded_by_region[region][epoch].save(decoded_path)
            saved_paths.append(decoded_path)
    return saved_paths


def save_cross_trajectory_tsds(
    true_by_region: dict[str, dict[str, dict[tuple[str, str], Any]]],
    decoded_by_region: dict[str, dict[str, dict[tuple[str, str], Any]]],
    save_dir: Path,
) -> list[Path]:
    """Save cross-trajectory decoded and true task-progression Tsds as `.npz` files."""
    saved_paths: list[Path] = []
    for region, values_by_epoch in true_by_region.items():
        for epoch, values_by_pair in values_by_epoch.items():
            for (encoding_trajectory, decoding_trajectory), true_tsd in values_by_pair.items():
                suffix = f"{encoding_trajectory}_to_{decoding_trajectory}"
                true_path = save_dir / f"{region}_{epoch}_{suffix}_true_tp_cross_traj.npz"
                true_tsd.save(true_path)
                saved_paths.append(true_path)

                decoded_path = (
                    save_dir / f"{region}_{epoch}_{suffix}_decoded_tp_cross_traj.npz"
                )
                decoded_by_region[region][epoch][
                    (encoding_trajectory, decoding_trajectory)
                ].save(decoded_path)
                saved_paths.append(decoded_path)
    return saved_paths


def save_epoch_metric_tables(
    metric_tables: dict[str, dict[str, pd.DataFrame]],
    save_dir: Path,
    suffix: str,
) -> list[Path]:
    """Save one parquet metric table per region and epoch."""
    saved_paths: list[Path] = []
    for region, tables_by_epoch in metric_tables.items():
        for epoch, table in tables_by_epoch.items():
            path = save_dir / f"{region}_{epoch}_{suffix}.parquet"
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def save_comparison_tables(
    comparison_tables: dict[str, dict[str, pd.DataFrame]],
    save_dir: Path,
    suffix: str,
) -> list[Path]:
    """Save one parquet light-vs-dark comparison table per region and light epoch."""
    saved_paths: list[Path] = []
    for region, tables_by_light_epoch in comparison_tables.items():
        for light_epoch, table in tables_by_light_epoch.items():
            dark_epoch = str(table["dark_epoch"].iloc[0]) if not table.empty else DEFAULT_DARK_EPOCH
            path = save_dir / f"{region}_{light_epoch}_{dark_epoch}_{suffix}.parquet"
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def save_legacy_pickles(
    true_place_by_region: dict[str, dict[str, Any]],
    decoded_place_by_region: dict[str, dict[str, Any]],
    true_tp_by_region: dict[str, dict[str, Any]],
    decoded_tp_by_region: dict[str, dict[str, Any]],
    true_tp_cross_by_region: dict[str, dict[str, dict[tuple[str, str], Any]]],
    decoded_tp_cross_by_region: dict[str, dict[str, dict[tuple[str, str], Any]]],
    save_dir: Path,
) -> list[Path]:
    """Write legacy nested pickle outputs for compatibility."""
    saved_paths: list[Path] = []
    outputs = {
        "true_place.pkl": true_place_by_region,
        "decoded_place.pkl": decoded_place_by_region,
        "true_tp.pkl": true_tp_by_region,
        "decoded_tp.pkl": decoded_tp_by_region,
        "true_tp_cross_traj.pkl": true_tp_cross_by_region,
        "decoded_tp_cross_traj.pkl": decoded_tp_cross_by_region,
    }
    for filename, payload in outputs.items():
        path = save_dir / filename
        with open(path, "wb") as file:
            pickle.dump(payload, file)
        saved_paths.append(path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the decoding workflow."""
    parser = argparse.ArgumentParser(
        description="Compare place and task-progression decoding with lap-wise cross-validation"
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
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Number of lap-wise cross-validation folds. Default: {DEFAULT_N_FOLDS}",
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=DEFAULT_BIN_SIZE_S,
        help=f"Time bin size in seconds for Bayesian decoding. Default: {DEFAULT_BIN_SIZE_S}",
    )
    parser.add_argument(
        "--sliding-window-size-bins",
        type=int,
        default=DEFAULT_SLIDING_WINDOW_SIZE_BINS,
        help=(
            "Sliding window size, in decode bins, passed to `nap.decode_bayes`. "
            f"Default: {DEFAULT_SLIDING_WINDOW_SIZE_BINS}"
        ),
    )
    parser.add_argument(
        "--cross-traj-fr-threshold-hz",
        type=float,
        default=DEFAULT_CROSS_TRAJ_FR_THRESHOLD_HZ,
        help=(
            "Movement firing-rate threshold used to select units for cross-trajectory decoding. "
            f"Default: {DEFAULT_CROSS_TRAJ_FR_THRESHOLD_HZ}"
        ),
    )
    parser.add_argument(
        "--save-legacy-pickle",
        action="store_true",
        help="Also write the legacy nested pickle outputs for compatibility.",
    )
    parser.add_argument(
        "--error-mode",
        choices=ERROR_MODES,
        default="signed",
        help="How to summarize decoding error in the binned error plots. Default: signed",
    )
    parser.add_argument(
        "--error-summary",
        choices=SUMMARY_MODES,
        default="median_iqr",
        help="How to summarize per-bin decoding error bars in the plots. Default: median_iqr",
    )
    return parser.parse_args()


def main() -> None:
    """Run the task-progression decoding workflow for one dataset."""
    args = parse_arguments()
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    light_epochs, dark_epoch = get_light_dark_epochs(session["run_epochs"])

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    save_dir = analysis_path / "task_progression_decoding"
    fig_dir = analysis_path / "figs" / "task_progression_decoding"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    min_lap_counts = validate_fold_count(
        session["trajectory_intervals"],
        n_folds=args.n_folds,
    )
    linear_position_bins = build_linear_position_bins(args.animal_name)
    task_progression_bins = build_combined_task_progression_bins(args.animal_name)
    task_progression_by_trajectory_bins = build_task_progression_bins(args.animal_name)
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    true_place_by_region: dict[str, dict[str, Any]] = {region: {} for region in REGIONS}
    decoded_place_by_region: dict[str, dict[str, Any]] = {region: {} for region in REGIONS}
    true_tp_by_region: dict[str, dict[str, Any]] = {region: {} for region in REGIONS}
    decoded_tp_by_region: dict[str, dict[str, Any]] = {region: {} for region in REGIONS}
    true_tp_cross_by_region: dict[str, dict[str, dict[tuple[str, str], Any]]] = {
        region: {} for region in REGIONS
    }
    decoded_tp_cross_by_region: dict[str, dict[str, dict[tuple[str, str], Any]]] = {
        region: {} for region in REGIONS
    }
    epoch_metric_tables: dict[str, dict[str, pd.DataFrame]] = {region: {} for region in REGIONS}
    epoch_cross_metric_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in REGIONS
    }
    epoch_binned_tables: dict[str, dict[str, dict[str, pd.DataFrame]]] = {
        region: {} for region in REGIONS
    }
    epoch_cross_binned_tables: dict[str, dict[str, dict[tuple[str, str], pd.DataFrame]]] = {
        region: {} for region in REGIONS
    }

    for region in REGIONS:
        active_unit_ids = _get_active_unit_ids(
            session["spikes_by_region"][region],
            movement_firing_rates[region][dark_epoch],
            threshold_hz=args.cross_traj_fr_threshold_hz,
        )
        for epoch in session["run_epochs"]:
            true_place, decoded_place = decode_cv(
                spikes=session["spikes_by_region"][region],
                feature_by_epoch=session["linear_position_by_run"],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                epoch=epoch,
                bins=linear_position_bins,
                n_folds=args.n_folds,
                bin_size_s=args.bin_size_s,
                sliding_window_size_bins=args.sliding_window_size_bins,
                random_state=DEFAULT_RANDOM_STATE,
            )
            true_tp, decoded_tp = decode_cv(
                spikes=session["spikes_by_region"][region],
                feature_by_epoch=session["task_progression_by_run"],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                epoch=epoch,
                bins=task_progression_bins,
                n_folds=args.n_folds,
                bin_size_s=args.bin_size_s,
                sliding_window_size_bins=args.sliding_window_size_bins,
                random_state=DEFAULT_RANDOM_STATE,
            )
            true_place_by_region[region][epoch] = true_place
            decoded_place_by_region[region][epoch] = decoded_place
            true_tp_by_region[region][epoch] = true_tp
            decoded_tp_by_region[region][epoch] = decoded_tp

            true_tp_cross, decoded_tp_cross = decode_task_cross_trajectory(
                spikes=session["spikes_by_region"][region],
                task_progression_by_trajectory=session["task_progression_by_trajectory"],
                trajectory_intervals=session["trajectory_intervals"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                epoch=epoch,
                bins=task_progression_by_trajectory_bins,
                active_unit_ids=active_unit_ids,
                bin_size_s=args.bin_size_s,
                sliding_window_size_bins=args.sliding_window_size_bins,
            )
            true_tp_cross_by_region[region][epoch] = true_tp_cross
            decoded_tp_cross_by_region[region][epoch] = decoded_tp_cross
            epoch_binned_tables[region][epoch] = {
                "place": summarize_decoding_error_by_position(
                    true_place,
                    decoded_place,
                    bin_edges=linear_position_bins,
                    error_mode=args.error_mode,
                    summary=args.error_summary,
                    min_count=DEFAULT_MIN_BIN_COUNT,
                ),
                "task_progression": summarize_decoding_error_by_position(
                    true_tp,
                    decoded_tp,
                    bin_edges=task_progression_bins,
                    error_mode=args.error_mode,
                    summary=args.error_summary,
                    min_count=DEFAULT_MIN_BIN_COUNT,
                ),
            }
            epoch_cross_binned_tables[region][epoch] = {
                key: summarize_decoding_error_by_position(
                    true_tp_cross[key],
                    decoded_tp_cross[key],
                    bin_edges=task_progression_by_trajectory_bins,
                    error_mode=args.error_mode,
                    summary=args.error_summary,
                    min_count=DEFAULT_MIN_BIN_COUNT,
                )
                for key in true_tp_cross
            }

            epoch_metric_tables[region][epoch] = pd.DataFrame(
                [
                    {
                        "model": "place",
                        "n_units": len(list(session["spikes_by_region"][region].keys())),
                        **summarize_decoding_metrics(true_place, decoded_place),
                    },
                    {
                        "model": "task_progression",
                        "n_units": len(list(session["spikes_by_region"][region].keys())),
                        **summarize_decoding_metrics(true_tp, decoded_tp),
                    },
                ]
            )
            cross_rows: list[dict[str, Any]] = []
            for encoding_trajectory, decoding_trajectory in CROSS_TRAJECTORY_DECODING_MAP.items():
                key = (encoding_trajectory, decoding_trajectory)
                cross_rows.append(
                    {
                        "encoding_trajectory": encoding_trajectory,
                        "decoding_trajectory": decoding_trajectory,
                        "n_units": len(active_unit_ids),
                        **summarize_decoding_metrics(
                            true_tp_cross[key],
                            decoded_tp_cross[key],
                        ),
                    }
                )
            epoch_cross_metric_tables[region][epoch] = pd.DataFrame(cross_rows)

    light_dark_metric_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in REGIONS
    }
    light_dark_cross_metric_tables: dict[str, dict[str, pd.DataFrame]] = {
        region: {} for region in REGIONS
    }
    binned_comparison_tables: dict[str, dict[str, dict[str, pd.DataFrame]]] = {
        region: {} for region in REGIONS
    }
    cross_binned_comparison_tables: dict[str, dict[str, dict[tuple[str, str], pd.DataFrame]]] = {
        region: {} for region in REGIONS
    }
    saved_epoch_error_figures: list[Path] = []
    saved_light_dark_figures: list[Path] = []

    for region in REGIONS:
        for model_name, xlabel in (
            ("place", "Linear position"),
            ("task_progression", "Task progression"),
        ):
            figure_path = fig_dir / f"{region}_{model_name}_decoding_error_by_epoch.png"
            plot_epoch_error_profiles(
                {
                    epoch: epoch_binned_tables[region][epoch][model_name]
                    for epoch in session["run_epochs"]
                },
                epoch_order=session["run_epochs"],
                title=f"{region.upper()} {model_name} decoding error by epoch",
                xlabel=xlabel,
                error_mode=args.error_mode,
                summary=args.error_summary,
                save_path=figure_path,
            )
            saved_epoch_error_figures.append(figure_path)

        for encoding_trajectory, decoding_trajectory in CROSS_TRAJECTORY_DECODING_MAP.items():
            figure_path = (
                fig_dir
                / f"{region}_{encoding_trajectory}_to_{decoding_trajectory}_tp_cross_traj_error_by_epoch.png"
            )
            plot_cross_trajectory_epoch_error_profiles(
                epoch_cross_binned_tables[region],
                epoch_order=session["run_epochs"],
                encoding_trajectory=encoding_trajectory,
                decoding_trajectory=decoding_trajectory,
                title=(
                    f"{region.upper()} tp cross-traj error by epoch: "
                    f"{encoding_trajectory} -> {decoding_trajectory}"
                ),
                xlabel="Task progression",
                error_mode=args.error_mode,
                summary=args.error_summary,
                save_path=figure_path,
            )
            saved_epoch_error_figures.append(figure_path)

        dark_metrics = epoch_metric_tables[region][dark_epoch]
        dark_cross_metrics = epoch_cross_metric_tables[region][dark_epoch]
        binned_comparison_tables[region] = {}
        cross_binned_comparison_tables[region] = {}
        for light_epoch in light_epochs:
            light_dark_metric_tables[region][light_epoch] = build_metric_comparison_table(
                epoch_metric_tables[region][light_epoch],
                dark_metrics,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                join_columns=["model"],
            )
            light_dark_cross_metric_tables[region][light_epoch] = build_metric_comparison_table(
                epoch_cross_metric_tables[region][light_epoch],
                dark_cross_metrics,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                join_columns=["encoding_trajectory", "decoding_trajectory"],
            )

            binned_comparison_tables[region][light_epoch] = {}
            for model_name, xlabel in (
                (
                    "place",
                    "Linear position",
                ),
                (
                    "task_progression",
                    "Task progression",
                ),
            ):
                light_summary = epoch_binned_tables[region][light_epoch][model_name]
                dark_summary = epoch_binned_tables[region][dark_epoch][model_name]
                comparison = build_binned_error_comparison_table(
                    light_summary,
                    dark_summary,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                )
                binned_comparison_tables[region][light_epoch][model_name] = comparison
                figure_path = (
                    fig_dir / f"{region}_{light_epoch}_{dark_epoch}_{model_name}_de_by_position.png"
                )
                plot_binned_error_comparison(
                    comparison,
                    title=f"{region.upper()} {model_name} decoding error",
                    xlabel=xlabel,
                    error_mode=args.error_mode,
                    summary=args.error_summary,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    save_path=figure_path,
                )
                saved_light_dark_figures.append(figure_path)

            cross_binned_comparison_tables[region][light_epoch] = {}
            for encoding_trajectory, decoding_trajectory in CROSS_TRAJECTORY_DECODING_MAP.items():
                key = (encoding_trajectory, decoding_trajectory)
                light_summary = epoch_cross_binned_tables[region][light_epoch][key]
                dark_summary = epoch_cross_binned_tables[region][dark_epoch][key]
                comparison = build_binned_error_comparison_table(
                    light_summary,
                    dark_summary,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                )
                cross_binned_comparison_tables[region][light_epoch][key] = comparison
                figure_path = (
                    fig_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{encoding_trajectory}_to_{decoding_trajectory}_tp_cross_traj_de.png"
                )
                plot_binned_error_comparison(
                    comparison,
                    title=(
                        f"{region.upper()} tp cross-traj error: "
                        f"{encoding_trajectory} -> {decoding_trajectory}"
                    ),
                    xlabel="Task progression",
                    error_mode=args.error_mode,
                    summary=args.error_summary,
                    light_epoch=light_epoch,
                    dark_epoch=dark_epoch,
                    save_path=figure_path,
                )
                saved_light_dark_figures.append(figure_path)

    saved_npz_paths: list[Path] = []
    saved_npz_paths.extend(
        save_within_epoch_tsds(
            true_place_by_region,
            decoded_place_by_region,
            model_name="place",
            save_dir=save_dir,
        )
    )
    saved_npz_paths.extend(
        save_within_epoch_tsds(
            true_tp_by_region,
            decoded_tp_by_region,
            model_name="tp",
            save_dir=save_dir,
        )
    )
    saved_npz_paths.extend(
        save_cross_trajectory_tsds(
            true_tp_cross_by_region,
            decoded_tp_cross_by_region,
            save_dir=save_dir,
        )
    )

    saved_epoch_metric_tables = save_epoch_metric_tables(
        epoch_metric_tables,
        save_dir=save_dir,
        suffix="decoding_summary",
    )
    saved_epoch_cross_metric_tables = save_epoch_metric_tables(
        epoch_cross_metric_tables,
        save_dir=save_dir,
        suffix="cross_trajectory_decoding_summary",
    )
    saved_epoch_binned_tables: list[Path] = []
    for region, tables_by_epoch in epoch_binned_tables.items():
        for epoch, tables_by_model in tables_by_epoch.items():
            for model_name, table in tables_by_model.items():
                path = save_dir / f"{region}_{epoch}_{model_name}_decoding_error_by_position.parquet"
                table.to_parquet(path)
                saved_epoch_binned_tables.append(path)
    for region, tables_by_epoch in epoch_cross_binned_tables.items():
        for epoch, tables_by_pair in tables_by_epoch.items():
            for (encoding_trajectory, decoding_trajectory), table in tables_by_pair.items():
                path = (
                    save_dir
                    / f"{region}_{epoch}_{encoding_trajectory}_to_{decoding_trajectory}_tp_cross_traj_error_by_position.parquet"
                )
                table.to_parquet(path)
                saved_epoch_binned_tables.append(path)
    saved_light_dark_metric_tables = save_comparison_tables(
        light_dark_metric_tables,
        save_dir=save_dir,
        suffix="decoding_comparison",
    )
    saved_light_dark_cross_metric_tables = save_comparison_tables(
        light_dark_cross_metric_tables,
        save_dir=save_dir,
        suffix="cross_trajectory_decoding_comparison",
    )

    saved_binned_tables: list[Path] = []
    for region, tables_by_light_epoch in binned_comparison_tables.items():
        for light_epoch, tables_by_model in tables_by_light_epoch.items():
            for model_name, table in tables_by_model.items():
                path = (
                    save_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{model_name}_decoding_error_by_position.parquet"
                )
                table.to_parquet(path)
                saved_binned_tables.append(path)
    for region, tables_by_light_epoch in cross_binned_comparison_tables.items():
        for light_epoch, tables_by_pair in tables_by_light_epoch.items():
            for (encoding_trajectory, decoding_trajectory), table in tables_by_pair.items():
                path = (
                    save_dir
                    / f"{region}_{light_epoch}_{dark_epoch}_{encoding_trajectory}_to_{decoding_trajectory}_tp_cross_traj_error_by_position.parquet"
                )
                table.to_parquet(path)
                saved_binned_tables.append(path)

    saved_legacy_pickles: list[Path] = []
    if args.save_legacy_pickle:
        saved_legacy_pickles = save_legacy_pickles(
            true_place_by_region,
            decoded_place_by_region,
            true_tp_by_region,
            decoded_tp_by_region,
            true_tp_cross_by_region,
            decoded_tp_cross_by_region,
            save_dir=save_dir,
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.task_progression_decoding",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "n_folds": args.n_folds,
            "bin_size_s": args.bin_size_s,
            "sliding_window_size_bins": args.sliding_window_size_bins,
            "cross_traj_fr_threshold_hz": args.cross_traj_fr_threshold_hz,
            "light_epochs": list(light_epochs),
            "dark_epoch": dark_epoch,
            "error_mode": args.error_mode,
            "error_summary": args.error_summary,
            "save_legacy_pickle": args.save_legacy_pickle,
            "random_state": DEFAULT_RANDOM_STATE,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "min_lap_counts": min_lap_counts,
            "saved_npz_paths": saved_npz_paths,
            "saved_epoch_metric_tables": saved_epoch_metric_tables,
            "saved_epoch_cross_metric_tables": saved_epoch_cross_metric_tables,
            "saved_epoch_binned_tables": saved_epoch_binned_tables,
            "saved_light_dark_metric_tables": saved_light_dark_metric_tables,
            "saved_light_dark_cross_metric_tables": saved_light_dark_cross_metric_tables,
            "saved_binned_tables": saved_binned_tables,
            "saved_epoch_error_figures": saved_epoch_error_figures,
            "saved_light_dark_figures": saved_light_dark_figures,
            "saved_legacy_pickles": saved_legacy_pickles,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
