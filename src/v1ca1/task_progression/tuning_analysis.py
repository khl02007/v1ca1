from __future__ import annotations

"""Compare same-turn task-progression tuning similarity across trajectories.

This script loads one session's spike trains, per-epoch position timestamps,
XY position samples, and trajectory intervals; rebuilds movement intervals and
trajectory-specific task-progression coordinates; computes within-epoch tuning
curve similarity for the same-turn trajectory pairs; and then compares those
similarity scores across one light epoch and one dark epoch for left, right,
and pooled similarity.

The similarity metric is configurable at the command line and can be one of:

- `correlation`
- `absolute_overlap`
- `shape_overlap`

Outputs are written under the analysis directory in metric-specific parquet
tables and scatter plots. The saved tables are per-unit summary outputs rather
than time series, so parquet is the preferred format here: rows are units,
columns hold summary values such as within-epoch similarity or light-vs-dark
comparison scores. In this repository, pynapple-backed `.npz` outputs are a
better fit for time-domain artifacts such as timestamps, intervals, spikes, or
continuous time series. Run metadata is also recorded under
`analysis_path / "v1ca1_log"`.

Optionally, the script can also estimate per-unit significance for within-epoch
left- and right-turn similarity using circular spike-time shifts on the
concatenated movement axis. When enabled, raw empirical p-values, BH-FDR
q-values, and null summary statistics are appended to the within-epoch tables,
and one significance scatter figure is saved per region/turn/epoch table.
"""

import argparse
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TURN_TRAJECTORY_PAIRS,
    build_task_progression_bins,
    compute_movement_firing_rates,
    compute_trajectory_task_progression_tuning_curves,
    get_analysis_path,
    get_run_epochs,
    load_epoch_tags,
    prepare_task_progression_session,
)


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SHUFFLE_SEED = 47
DEFAULT_MIN_SHIFT_FRACTION = 0.1
P_VALUE_THRESHOLD = 0.05


def interpolate_nans(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in one 1D tuning curve."""
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


def _require_statsmodels() -> None:
    """Ensure statsmodels is installed when significance is requested."""
    try:
        import statsmodels  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Per-unit significance requires `statsmodels`. Install the project "
            "dependencies or update the environment before using "
            "`--compute-significance`."
        ) from exc


def _intervalset_to_arrays(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted start and end arrays for one IntervalSet."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts)
    return starts[order], ends[order]


def _movement_interval_axis(intervals: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return concatenated-axis metadata for one movement IntervalSet."""
    starts, ends = _intervalset_to_arrays(intervals)
    durations = ends - starts
    total_length = float(durations.sum())
    axis_starts = np.concatenate(([0.0], np.cumsum(durations[:-1], dtype=float)))
    return starts, ends, axis_starts, total_length


def _times_to_movement_positions(
    times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    axis_starts: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Map times inside the movement IntervalSet onto the concatenated movement axis."""
    if times.size == 0:
        return np.array([], dtype=float)

    interval_index = np.searchsorted(starts, times, side="right") - 1
    if np.any(interval_index < 0):
        raise ValueError("Encountered spike times before the first movement interval.")

    if np.any(times > ends[interval_index] + eps):
        raise ValueError("Encountered spike times outside the movement IntervalSet.")

    return axis_starts[interval_index] + (times - starts[interval_index])


def _movement_positions_to_times(
    positions: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    axis_starts: np.ndarray,
) -> np.ndarray:
    """Map concatenated movement-axis positions back to timestamps."""
    if positions.size == 0:
        return np.array([], dtype=float)

    durations = ends - starts
    axis_ends = axis_starts + durations
    interval_index = np.searchsorted(axis_ends, positions, side="right")
    interval_index = np.clip(interval_index, 0, starts.size - 1)
    return starts[interval_index] + (positions - axis_starts[interval_index])


def _stable_seed_component(value: str | int) -> int:
    """Return a deterministic uint32 seed component for one value."""
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFFFFFFFF
    return zlib.crc32(str(value).encode("utf-8")) & 0xFFFFFFFF


def _make_shuffle_rng(
    base_seed: int,
    *,
    region: str,
    epoch: str,
    turn_type: str,
    unit: int,
) -> np.random.Generator:
    """Return a deterministic RNG for one region/epoch/turn/unit combination."""
    seed_sequence = np.random.SeedSequence(
        [
            int(base_seed) & 0xFFFFFFFF,
            _stable_seed_component(region),
            _stable_seed_component(epoch),
            _stable_seed_component(turn_type),
            _stable_seed_component(unit),
        ]
    )
    return np.random.default_rng(seed_sequence)


def circular_shift_unit_spikes_on_movement_axis(
    unit_spikes: Any,
    movement_interval: Any,
    *,
    rng: np.random.Generator,
    min_shift_fraction: float = DEFAULT_MIN_SHIFT_FRACTION,
) -> Any:
    """Circularly shift one unit's movement-restricted spikes on the movement axis."""
    import pynapple as nap

    restricted_spikes = unit_spikes.restrict(movement_interval)
    spike_times = np.asarray(restricted_spikes.t, dtype=float)
    if spike_times.size == 0:
        return nap.Ts(t=np.array([], dtype=float), time_units="s")

    starts, ends, axis_starts, total_length = _movement_interval_axis(movement_interval)
    if total_length <= 0.0:
        return nap.Ts(t=np.array([], dtype=float), time_units="s")

    if not 0.0 <= float(min_shift_fraction) < 0.5:
        raise ValueError("min_shift_fraction must lie in [0, 0.5).")

    min_shift = float(min_shift_fraction) * total_length
    max_shift = (1.0 - float(min_shift_fraction)) * total_length
    shift_amount = rng.uniform(min_shift, max_shift)

    movement_positions = _times_to_movement_positions(
        spike_times,
        starts=starts,
        ends=ends,
        axis_starts=axis_starts,
    )
    shifted_positions = np.mod(movement_positions + shift_amount, total_length)
    shifted_times = _movement_positions_to_times(
        shifted_positions,
        starts=starts,
        ends=ends,
        axis_starts=axis_starts,
    )
    shifted_times.sort()
    return nap.Ts(t=shifted_times, time_units="s")


def compute_unit_turn_similarity(
    unit: int,
    unit_spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    turn_type: str,
    bins: np.ndarray,
    similarity_metric: str,
) -> float:
    """Compute one unit's same-turn similarity score for one epoch."""
    import pynapple as nap

    tuning_curves = compute_trajectory_task_progression_tuning_curves(
        nap.TsGroup({unit: unit_spikes}, time_units="s"),
        task_progression_by_trajectory,
        movement_interval,
        bins=bins,
    )
    trajectory_a, trajectory_b = TURN_TRAJECTORY_PAIRS[turn_type]
    return compute_similarity_score(
        tuning_curves[trajectory_a].sel(unit=unit).values,
        tuning_curves[trajectory_b].sel(unit=unit).values,
        similarity_metric=similarity_metric,
    )


def compute_empirical_p_value(
    observed_score: float,
    null_scores: np.ndarray,
) -> tuple[float, int]:
    """Return the one-sided empirical p-value and null exceedance count."""
    valid_null_scores = np.asarray(null_scores, dtype=float)
    valid_null_scores = valid_null_scores[np.isfinite(valid_null_scores)]
    if not np.isfinite(observed_score) or valid_null_scores.size == 0:
        return np.nan, 0

    ge_count = int(np.sum(valid_null_scores >= float(observed_score)))
    p_value = (1.0 + ge_count) / (1.0 + valid_null_scores.size)
    return float(p_value), ge_count


def compute_q_values(p_values: pd.Series) -> pd.Series:
    """Return BH-FDR q-values for one p-value series using statsmodels."""
    from statsmodels.stats.multitest import multipletests

    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = np.isfinite(p_values.to_numpy(dtype=float))
    if not np.any(valid):
        return q_values

    adjusted = multipletests(
        p_values.to_numpy(dtype=float)[valid],
        method="fdr_bh",
    )[1]
    q_values.loc[p_values.index[valid]] = adjusted
    return q_values


def annotate_turn_similarity_significance(
    similarity_table: pd.DataFrame,
    *,
    region: str,
    epoch: str,
    turn_type: str,
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    bins: np.ndarray,
    similarity_metric: str,
    n_shuffles: int,
    shuffle_seed: int,
) -> pd.DataFrame:
    """Append per-unit circular-shift significance columns to one within-epoch table."""
    annotated = similarity_table.copy()
    if annotated.empty:
        for column in (
            "p_value",
            "q_value",
            "null_mean",
            "null_std",
            "null_median",
            "null_ge_count",
            "n_shuffles",
            "n_null_valid",
        ):
            annotated[column] = pd.Series(dtype=float)
        return annotated

    rows: list[dict[str, float | int]] = []
    for unit in annotated.index.to_numpy():
        rng = _make_shuffle_rng(
            shuffle_seed,
            region=region,
            epoch=epoch,
            turn_type=turn_type,
            unit=int(unit),
        )
        observed_score = float(annotated.at[unit, "similarity"])
        unit_spikes = spikes[int(unit)]
        null_scores = np.full(int(n_shuffles), np.nan, dtype=float)

        for shuffle_index in range(int(n_shuffles)):
            shifted_unit_spikes = circular_shift_unit_spikes_on_movement_axis(
                unit_spikes,
                movement_interval,
                rng=rng,
            )
            null_scores[shuffle_index] = compute_unit_turn_similarity(
                int(unit),
                shifted_unit_spikes,
                task_progression_by_trajectory,
                movement_interval,
                turn_type=turn_type,
                bins=bins,
                similarity_metric=similarity_metric,
            )

        p_value, ge_count = compute_empirical_p_value(observed_score, null_scores)
        valid_null_scores = null_scores[np.isfinite(null_scores)]
        rows.append(
            {
                "unit": int(unit),
                "p_value": p_value,
                "null_mean": float(np.mean(valid_null_scores))
                if valid_null_scores.size
                else np.nan,
                "null_std": float(np.std(valid_null_scores))
                if valid_null_scores.size
                else np.nan,
                "null_median": float(np.median(valid_null_scores))
                if valid_null_scores.size
                else np.nan,
                "null_ge_count": ge_count,
                "n_shuffles": int(n_shuffles),
                "n_null_valid": int(valid_null_scores.size),
            }
        )

    significance_df = pd.DataFrame(rows).set_index("unit").sort_index()
    significance_df["q_value"] = compute_q_values(significance_df["p_value"])
    return annotated.join(significance_df, how="left")


def get_light_and_dark_epochs(
    run_epochs: list[str],
    light_epoch: str | None,
    dark_epoch: str | None,
) -> tuple[str, str]:
    """Return validated light and dark epoch labels for one session."""
    if not run_epochs:
        raise ValueError("No run epochs were found for this session.")

    selected_light_epoch = light_epoch or run_epochs[0]
    selected_dark_epoch = dark_epoch or run_epochs[-1]
    missing = [
        epoch
        for epoch in (selected_light_epoch, selected_dark_epoch)
        if epoch not in run_epochs
    ]
    if missing:
        raise ValueError(
            f"Requested epochs were not found in run epochs {run_epochs!r}: {missing!r}"
        )
    return selected_light_epoch, selected_dark_epoch


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


def compute_turn_similarity(
    tuning_curves_by_trajectory: dict[str, Any],
    turn_type: str,
    dark_epoch_firing_rates: pd.Series,
    firing_rate_threshold_hz: float,
    similarity_metric: str,
) -> pd.DataFrame:
    """Compute within-epoch same-turn similarity for all active units."""
    trajectory_a, trajectory_b = TURN_TRAJECTORY_PAIRS[turn_type]
    tuning_curve_a = tuning_curves_by_trajectory[trajectory_a]
    tuning_curve_b = tuning_curves_by_trajectory[trajectory_b]

    units_a = np.asarray(tuning_curve_a.coords["unit"].values)
    units_b = np.asarray(tuning_curve_b.coords["unit"].values)
    common_units = np.intersect1d(units_a, units_b)

    rows: list[dict[str, float | int | str]] = []
    for unit in common_units:
        rate = dark_epoch_firing_rates.get(unit, np.nan)
        if not np.isfinite(rate) or float(rate) <= firing_rate_threshold_hz:
            continue

        score = compute_similarity_score(
            tuning_curve_a.sel(unit=unit).values,
            tuning_curve_b.sel(unit=unit).values,
            similarity_metric=similarity_metric,
        )
        if np.isfinite(score):
            rows.append({"unit": unit, "similarity": float(score), "turn_type": turn_type})

    if not rows:
        return pd.DataFrame(columns=["similarity", "turn_type"], index=pd.Index([], name="unit"))
    return pd.DataFrame(rows).set_index("unit")


def compute_pooled_similarity(
    left_similarity: pd.DataFrame,
    right_similarity: pd.DataFrame,
) -> pd.DataFrame:
    """Pool left and right similarity by taking the per-unit maximum."""
    joined = pd.concat(
        [
            left_similarity["similarity"].rename("left_similarity"),
            right_similarity["similarity"].rename("right_similarity"),
        ],
        axis=1,
    )
    joined["similarity"] = joined.max(axis=1, skipna=True)
    joined = joined.dropna(subset=["similarity"])
    joined["turn_type"] = "pooled"
    return joined


def join_epoch_similarity(
    light_similarity: pd.DataFrame,
    dark_similarity: pd.DataFrame,
) -> pd.DataFrame:
    """Join light-epoch and dark-epoch similarity tables on shared units."""
    joined = pd.concat(
        [
            light_similarity["similarity"].rename("light_similarity"),
            dark_similarity["similarity"].rename("dark_similarity"),
        ],
        axis=1,
        join="inner",
    )
    return joined.dropna()


def get_metric_axis_limits(similarity_metric: str) -> tuple[float, float]:
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


def plot_similarity_scatter(
    similarity_comparison: pd.DataFrame,
    region: str,
    turn_label: str,
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
    fig_path: Path,
) -> None:
    """Save one light-vs-dark similarity scatter plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        similarity_comparison["light_similarity"],
        similarity_comparison["dark_similarity"],
        alpha=0.5,
        s=24,
        color="tab:purple",
    )

    axis_min, axis_max = get_metric_axis_limits(similarity_metric)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1)
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel(f"{light_epoch} similarity")
    ax.set_ylabel(f"{dark_epoch} similarity")
    ax.set_title(f"{region.upper()} {turn_label} ({similarity_metric})")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_similarity_significance_scatter(
    similarity_table: pd.DataFrame,
    *,
    region: str,
    epoch: str,
    turn_type: str,
    similarity_metric: str,
    fig_path: Path,
    p_value_threshold: float = P_VALUE_THRESHOLD,
) -> None:
    """Save one observed-similarity vs significance scatter plot."""
    import matplotlib.pyplot as plt

    valid = similarity_table.copy()
    valid = valid[
        np.isfinite(valid["similarity"]) & np.isfinite(valid["p_value"]) & (valid["p_value"] > 0.0)
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    if valid.empty:
        ax.text(0.5, 0.5, "No valid units", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.scatter(
            valid["similarity"],
            -np.log10(valid["p_value"]),
            alpha=0.6,
            s=24,
            color="tab:blue",
        )
        axis_min, axis_max = get_metric_axis_limits(similarity_metric)
        ax.set_xlim(axis_min, axis_max)
        ax.axhline(-np.log10(p_value_threshold), color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(get_similarity_axis_label(similarity_metric))
        ax.set_ylabel("-log10(p_value)")
    ax.set_title(f"{region.upper()} {epoch} {turn_type} significance")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)


def compute_similarity_outputs(
    tuning_curves_by_region: dict[str, dict[str, dict[str, Any]]],
    movement_firing_rates: dict[str, dict[str, np.ndarray]],
    spikes_by_region: dict[str, Any],
    selected_regions: tuple[str, ...],
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute left, right, and pooled similarity tables for each region."""
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for region in selected_regions:
        dark_epoch_firing_rates = pd.Series(
            movement_firing_rates[region][dark_epoch],
            index=list(spikes_by_region[region].keys()),
            dtype=float,
        )
        region_outputs: dict[str, pd.DataFrame] = {}

        for turn_type in ("left", "right"):
            light_similarity = compute_turn_similarity(
                tuning_curves_by_region[region][light_epoch],
                turn_type=turn_type,
                dark_epoch_firing_rates=dark_epoch_firing_rates,
                firing_rate_threshold_hz=DEFAULT_REGION_FR_THRESHOLDS[region],
                similarity_metric=similarity_metric,
            )
            dark_similarity = compute_turn_similarity(
                tuning_curves_by_region[region][dark_epoch],
                turn_type=turn_type,
                dark_epoch_firing_rates=dark_epoch_firing_rates,
                firing_rate_threshold_hz=DEFAULT_REGION_FR_THRESHOLDS[region],
                similarity_metric=similarity_metric,
            )
            region_outputs[f"{turn_type}_light"] = light_similarity
            region_outputs[f"{turn_type}_dark"] = dark_similarity
            region_outputs[turn_type] = join_epoch_similarity(light_similarity, dark_similarity)

        pooled_light = compute_pooled_similarity(
            region_outputs["left_light"],
            region_outputs["right_light"],
        )
        pooled_dark = compute_pooled_similarity(
            region_outputs["left_dark"],
            region_outputs["right_dark"],
        )
        region_outputs["pooled_light"] = pooled_light
        region_outputs["pooled_dark"] = pooled_dark
        region_outputs["pooled"] = join_epoch_similarity(pooled_light, pooled_dark)
        outputs[region] = region_outputs

    return outputs


def compute_similarity_significance_outputs(
    similarity_outputs: dict[str, dict[str, pd.DataFrame]],
    *,
    session: dict[str, Any],
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
    bins: np.ndarray,
    n_shuffles: int,
    shuffle_seed: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Append per-unit significance to within-epoch left/right similarity tables."""
    outputs_with_significance: dict[str, dict[str, pd.DataFrame]] = {}
    for region, region_outputs in similarity_outputs.items():
        updated_region_outputs = dict(region_outputs)
        for turn_type in ("left", "right"):
            for epoch_label, epoch in (("light", light_epoch), ("dark", dark_epoch)):
                key = f"{turn_type}_{epoch_label}"
                updated_region_outputs[key] = annotate_turn_similarity_significance(
                    region_outputs[key],
                    region=region,
                    epoch=epoch,
                    turn_type=turn_type,
                    spikes=session["spikes_by_region"][region],
                    task_progression_by_trajectory=session["task_progression_by_trajectory"][epoch],
                    movement_interval=session["movement_by_run"][epoch],
                    bins=bins,
                    similarity_metric=similarity_metric,
                    n_shuffles=n_shuffles,
                    shuffle_seed=shuffle_seed,
                )
        outputs_with_significance[region] = updated_region_outputs
    return outputs_with_significance


def save_similarity_tables(
    similarity_outputs: dict[str, dict[str, pd.DataFrame]],
    data_dir: Path,
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
) -> list[Path]:
    """Write metric-specific per-unit similarity tables as parquet files.

    These outputs are indexed by unit and store summary columns, not time-based
    signals, so parquet is more appropriate than pynapple for this artifact
    type.
    """
    saved_paths: list[Path] = []
    for region, region_outputs in similarity_outputs.items():
        for key, table in region_outputs.items():
            if key.endswith("_light"):
                epoch = light_epoch
                suffix = key[: -len("_light")]
                filename = (
                    f"{region}_{epoch}_{similarity_metric}_{suffix}_within_epoch_similarity.parquet"
                )
            elif key.endswith("_dark"):
                epoch = dark_epoch
                suffix = key[: -len("_dark")]
                filename = (
                    f"{region}_{epoch}_{similarity_metric}_{suffix}_within_epoch_similarity.parquet"
                )
            else:
                filename = (
                    f"{region}_{light_epoch}_{dark_epoch}_{similarity_metric}_{key}_comparison.parquet"
                )
            path = data_dir / filename
            table.to_parquet(path)
            saved_paths.append(path)
    return saved_paths


def save_similarity_figures(
    similarity_outputs: dict[str, dict[str, pd.DataFrame]],
    fig_dir: Path,
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
) -> list[Path]:
    """Write scatter plots for left, right, and pooled similarity comparisons."""
    saved_paths: list[Path] = []
    for region, region_outputs in similarity_outputs.items():
        for turn_label in ("left", "right", "pooled"):
            comparison = region_outputs[turn_label]
            fig_path = (
                fig_dir
                / f"{region}_{light_epoch}_{dark_epoch}_{similarity_metric}_{turn_label}_similarity.png"
            )
            plot_similarity_scatter(
                comparison,
                region=region,
                turn_label=turn_label,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                similarity_metric=similarity_metric,
                fig_path=fig_path,
            )
            saved_paths.append(fig_path)
    return saved_paths


def save_similarity_significance_figures(
    similarity_outputs: dict[str, dict[str, pd.DataFrame]],
    fig_dir: Path,
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
) -> list[Path]:
    """Write observed-similarity vs significance scatter plots for within-epoch tables."""
    saved_paths: list[Path] = []
    for region, region_outputs in similarity_outputs.items():
        for turn_type in ("left", "right"):
            for epoch_label, epoch in (("light", light_epoch), ("dark", dark_epoch)):
                key = f"{turn_type}_{epoch_label}"
                table = region_outputs[key]
                if "p_value" not in table.columns:
                    continue
                fig_path = (
                    fig_dir
                    / f"{region}_{epoch}_{similarity_metric}_{turn_type}_significance.png"
                )
                plot_similarity_significance_scatter(
                    table,
                    region=region,
                    epoch=epoch,
                    turn_type=turn_type,
                    similarity_metric=similarity_metric,
                    fig_path=fig_path,
                )
                saved_paths.append(fig_path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the tuning similarity script."""
    parser = argparse.ArgumentParser(
        description="Compare task-progression tuning similarity across light and dark epochs"
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
        "--light-epoch",
        help="Run epoch label to use as the light epoch. Defaults to the first run epoch.",
    )
    parser.add_argument(
        "--dark-epoch",
        help="Run epoch label to use as the dark epoch. Defaults to the last run epoch.",
    )
    parser.add_argument(
        "--similarity-metric",
        choices=("correlation", "absolute_overlap", "shape_overlap"),
        default="correlation",
        help="Similarity metric used to compare same-turn tuning curves.",
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
        "--compute-significance",
        action="store_true",
        help=(
            "Estimate per-unit within-epoch significance using circular spike-time shifts "
            "on the movement axis."
        ),
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help=f"Number of circular-shift surrogates when significance is enabled. Default: {DEFAULT_N_SHUFFLES}",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=f"Random seed used for circular-shift significance. Default: {DEFAULT_SHUFFLE_SEED}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the task-progression tuning similarity workflow for one session."""
    args = parse_arguments()
    selected_regions = tuple(args.regions)
    if args.compute_significance:
        _require_statsmodels()
        if args.n_shuffles < 1:
            raise ValueError("--n-shuffles must be at least 1 when significance is enabled.")

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    epoch_tags, _epoch_source = load_epoch_tags(analysis_path)
    available_run_epochs = get_run_epochs(epoch_tags)
    light_epoch, dark_epoch = get_light_and_dark_epochs(
        available_run_epochs,
        args.light_epoch,
        args.dark_epoch,
    )
    selected_epochs = [light_epoch]
    if dark_epoch != light_epoch:
        selected_epochs.append(dark_epoch)

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        selected_run_epochs=selected_epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    data_dir = analysis_path / "task_progression_tuning"
    fig_dir = analysis_path / "figs" / "task_progression_tuning"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    task_progression_bins = build_task_progression_bins(args.animal_name)
    tuning_curves_by_region: dict[str, dict[str, dict[str, Any]]] = {}
    for region in selected_regions:
        tuning_curves_by_region[region] = {}
        for epoch in (light_epoch, dark_epoch):
            tuning_curves_by_region[region][epoch] = compute_trajectory_task_progression_tuning_curves(
                session["spikes_by_region"][region],
                session["task_progression_by_trajectory"][epoch],
                session["movement_by_run"][epoch],
                bins=task_progression_bins,
            )

    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        selected_epochs,
    )
    similarity_outputs = compute_similarity_outputs(
        tuning_curves_by_region,
        movement_firing_rates,
        session["spikes_by_region"],
        selected_regions=selected_regions,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        similarity_metric=args.similarity_metric,
    )
    if args.compute_significance:
        similarity_outputs = compute_similarity_significance_outputs(
            similarity_outputs,
            session=session,
            light_epoch=light_epoch,
            dark_epoch=dark_epoch,
            similarity_metric=args.similarity_metric,
            bins=task_progression_bins,
            n_shuffles=args.n_shuffles,
            shuffle_seed=args.shuffle_seed,
        )

    saved_tables = save_similarity_tables(
        similarity_outputs,
        data_dir=data_dir,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        similarity_metric=args.similarity_metric,
    )
    saved_figures = save_similarity_figures(
        similarity_outputs,
        fig_dir=fig_dir,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        similarity_metric=args.similarity_metric,
    )
    if args.compute_significance:
        saved_figures.extend(
            save_similarity_significance_figures(
                similarity_outputs,
                fig_dir=fig_dir,
                light_epoch=light_epoch,
                dark_epoch=dark_epoch,
                similarity_metric=args.similarity_metric,
            )
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.tuning_analysis",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": list(selected_regions),
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
            "similarity_metric": args.similarity_metric,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "compute_significance": args.compute_significance,
            "n_shuffles": args.n_shuffles,
            "shuffle_seed": args.shuffle_seed,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "saved_tables": saved_tables,
            "saved_figures": saved_figures,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
