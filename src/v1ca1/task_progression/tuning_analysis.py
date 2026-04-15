from __future__ import annotations

"""Quantify per-epoch task-progression tuning similarity across trajectories.

This workflow treats within-epoch trajectory-pair similarity as the primary
analysis product. For each selected run epoch and region, it computes direct
same-turn and same-arm similarity scores for each sufficiently active unit,
adds pooled within-family summaries, and writes one long-form parquet table
plus summary figures. Optionally, it also estimates per-unit significance for
the direct pairwise rows using circular spike-time shifts on the concatenated
movement axis.

An optional secondary comparison mode can then compare the same similarity
labels across two explicit epochs. Those comparison outputs are saved
separately from the per-epoch products.
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
    build_task_progression_bins,
    compute_movement_firing_rates,
    compute_trajectory_task_progression_tuning_curves,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)


DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.0}
DEFAULT_N_SHUFFLES = 1000
DEFAULT_SHUFFLE_SEED = 47
DEFAULT_MIN_SHIFT_FRACTION = 0.1
P_VALUE_THRESHOLD = 0.05
SIGNIFICANCE_COLUMNS = (
    "p_value",
    "q_value",
    "null_mean",
    "null_std",
    "null_median",
    "null_ge_count",
    "n_shuffles",
    "n_null_valid",
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
POOLED_COMPARISON_SPECS = (
    {
        "comparison_family": "same_turn",
        "comparison_label": "pooled_same_turn",
    },
    {
        "comparison_family": "same_arm",
        "comparison_label": "pooled_same_arm",
    },
)
DIRECT_COMPARISON_LABELS = tuple(
    spec["comparison_label"] for spec in DIRECT_COMPARISON_SPECS
)
POOLED_COMPARISON_LABELS = tuple(
    spec["comparison_label"] for spec in POOLED_COMPARISON_SPECS
)
COMPARISON_LABEL_ORDER = (
    "left_turn",
    "right_turn",
    "left_arm",
    "right_arm",
    "pooled_same_turn",
    "pooled_same_arm",
)
SIDE_TO_DIRECT_LABELS = {
    "left": ("left_turn", "left_arm"),
    "right": ("right_turn", "right_arm"),
}
LABEL_TO_SPEC = {
    spec["comparison_label"]: spec for spec in DIRECT_COMPARISON_SPECS
}


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


def _movement_interval_axis(
    intervals: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    comparison_label: str,
    unit: int,
) -> np.random.Generator:
    """Return a deterministic RNG for one region/epoch/comparison/unit combination."""
    seed_sequence = np.random.SeedSequence(
        [
            int(base_seed) & 0xFFFFFFFF,
            _stable_seed_component(region),
            _stable_seed_component(epoch),
            _stable_seed_component(comparison_label),
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


def _flip_curve_if_requested(curve: np.ndarray, *, should_flip: bool) -> np.ndarray:
    """Return one tuning curve, optionally reversed along the task axis."""
    array = np.asarray(curve, dtype=float)
    if should_flip:
        return np.asarray(array[::-1], dtype=float)
    return array


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


def _empty_similarity_table() -> pd.DataFrame:
    """Return an empty long-form similarity table with the expected schema."""
    return pd.DataFrame(
        {
            "unit": pd.Series(dtype=int),
            "region": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "comparison_family": pd.Series(dtype=str),
            "comparison_label": pd.Series(dtype=str),
            "side": pd.Series(dtype=object),
            "trajectory_a": pd.Series(dtype=object),
            "trajectory_b": pd.Series(dtype=object),
            "flip_trajectory_b": pd.Series(dtype=object),
            "firing_rate_hz": pd.Series(dtype=float),
            "similarity": pd.Series(dtype=float),
        }
    )


def _empty_comparison_table() -> pd.DataFrame:
    """Return an empty long-form comparison table with the expected schema."""
    return pd.DataFrame(
        {
            "unit": pd.Series(dtype=int),
            "region": pd.Series(dtype=str),
            "epoch_a": pd.Series(dtype=str),
            "epoch_b": pd.Series(dtype=str),
            "comparison_family": pd.Series(dtype=str),
            "comparison_label": pd.Series(dtype=str),
            "side": pd.Series(dtype=object),
            "trajectory_a": pd.Series(dtype=object),
            "trajectory_b": pd.Series(dtype=object),
            "flip_trajectory_b": pd.Series(dtype=object),
            "similarity_epoch_a": pd.Series(dtype=float),
            "similarity_epoch_b": pd.Series(dtype=float),
            "delta_similarity": pd.Series(dtype=float),
        }
    )


def deduplicate_requested_epochs(selected_epochs: list[str] | None) -> list[str] | None:
    """Return a requested epoch list with duplicates removed while preserving order."""
    if selected_epochs is None:
        return None
    return list(dict.fromkeys(selected_epochs))


def resolve_compare_epochs(
    compare_epochs: list[str] | None,
    analyzed_epochs: list[str],
) -> tuple[str, str] | None:
    """Validate and return the optional epoch pair used for cross-epoch comparison."""
    if compare_epochs is None:
        return None
    if len(compare_epochs) != 2:
        raise ValueError("--compare-epochs must provide exactly two epoch labels.")

    epoch_a, epoch_b = (str(compare_epochs[0]), str(compare_epochs[1]))
    if epoch_a == epoch_b:
        raise ValueError("--compare-epochs must name two distinct epochs.")

    missing = [epoch for epoch in (epoch_a, epoch_b) if epoch not in analyzed_epochs]
    if missing:
        raise ValueError(
            "Comparison epochs must already be part of the analyzed epoch set. "
            f"Analyzed epochs: {analyzed_epochs!r}; missing: {missing!r}"
        )
    return epoch_a, epoch_b


def compute_unit_pair_similarity(
    unit: int,
    unit_spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    bins: np.ndarray,
    similarity_metric: str,
    *,
    trajectory_a: str,
    trajectory_b: str,
    flip_trajectory_b: bool,
) -> float:
    """Compute one unit's direct pairwise similarity score for one epoch."""
    import pynapple as nap

    tuning_curves = compute_trajectory_task_progression_tuning_curves(
        nap.TsGroup({int(unit): unit_spikes}, time_units="s"),
        task_progression_by_trajectory,
        movement_interval,
        bins=bins,
    )
    curve_a = np.asarray(tuning_curves[trajectory_a].sel(unit=unit).values, dtype=float)
    curve_b = _flip_curve_if_requested(
        tuning_curves[trajectory_b].sel(unit=unit).values,
        should_flip=bool(flip_trajectory_b),
    )
    return compute_similarity_score(
        curve_a,
        curve_b,
        similarity_metric=similarity_metric,
    )


def compute_epoch_similarity_table(
    *,
    region: str,
    epoch: str,
    tuning_curves_by_trajectory: dict[str, Any],
    epoch_firing_rates: pd.Series,
    firing_rate_threshold_hz: float,
    similarity_metric: str,
) -> pd.DataFrame:
    """Compute one long-form per-epoch similarity table for one region."""
    rows: list[dict[str, Any]] = []
    for spec in DIRECT_COMPARISON_SPECS:
        trajectory_a = str(spec["trajectory_a"])
        trajectory_b = str(spec["trajectory_b"])
        tuning_curve_a = tuning_curves_by_trajectory[trajectory_a]
        tuning_curve_b = tuning_curves_by_trajectory[trajectory_b]

        units_a = np.asarray(tuning_curve_a.coords["unit"].values, dtype=int)
        units_b = np.asarray(tuning_curve_b.coords["unit"].values, dtype=int)
        common_units = np.intersect1d(units_a, units_b)

        for unit in common_units:
            firing_rate_hz = float(epoch_firing_rates.get(int(unit), np.nan))
            if not np.isfinite(firing_rate_hz) or firing_rate_hz <= firing_rate_threshold_hz:
                continue

            curve_a = np.asarray(tuning_curve_a.sel(unit=unit).values, dtype=float)
            curve_b = _flip_curve_if_requested(
                tuning_curve_b.sel(unit=unit).values,
                should_flip=bool(spec["flip_trajectory_b"]),
            )
            similarity = compute_similarity_score(
                curve_a,
                curve_b,
                similarity_metric=similarity_metric,
            )
            if not np.isfinite(similarity):
                continue

            rows.append(
                {
                    "unit": int(unit),
                    "region": region,
                    "epoch": epoch,
                    "comparison_family": str(spec["comparison_family"]),
                    "comparison_label": str(spec["comparison_label"]),
                    "side": str(spec["side"]),
                    "trajectory_a": trajectory_a,
                    "trajectory_b": trajectory_b,
                    "flip_trajectory_b": bool(spec["flip_trajectory_b"]),
                    "firing_rate_hz": firing_rate_hz,
                    "similarity": float(similarity),
                }
            )

    if not rows:
        return _empty_similarity_table()

    table = pd.DataFrame(rows)
    table["comparison_label"] = pd.Categorical(
        table["comparison_label"],
        categories=COMPARISON_LABEL_ORDER,
        ordered=True,
    )
    return table.sort_values(
        ["unit", "comparison_label"],
        kind="stable",
    ).reset_index(drop=True)


def append_pooled_similarity_rows(similarity_table: pd.DataFrame) -> pd.DataFrame:
    """Append pooled same-turn and same-arm rows using the per-unit max rule."""
    if similarity_table.empty:
        return similarity_table.copy()

    pooled_rows: list[dict[str, Any]] = []
    direct = similarity_table[
        similarity_table["comparison_label"].isin(DIRECT_COMPARISON_LABELS)
    ].copy()
    if direct.empty:
        return similarity_table.copy()

    for pooled_spec in POOLED_COMPARISON_SPECS:
        family = str(pooled_spec["comparison_family"])
        family_rows = direct[direct["comparison_family"] == family]
        if family_rows.empty:
            continue

        grouped = family_rows.groupby("unit", sort=True, observed=False)
        for unit, unit_rows in grouped:
            valid_similarity = pd.to_numeric(unit_rows["similarity"], errors="coerce")
            if not np.isfinite(valid_similarity.to_numpy(dtype=float)).any():
                continue
            pooled_rows.append(
                {
                    "unit": int(unit),
                    "region": str(unit_rows["region"].iloc[0]),
                    "epoch": str(unit_rows["epoch"].iloc[0]),
                    "comparison_family": family,
                    "comparison_label": str(pooled_spec["comparison_label"]),
                    "side": None,
                    "trajectory_a": None,
                    "trajectory_b": None,
                    "flip_trajectory_b": None,
                    "firing_rate_hz": float(
                        pd.to_numeric(unit_rows["firing_rate_hz"], errors="coerce").max()
                    ),
                    "similarity": float(valid_similarity.max()),
                }
            )

    if not pooled_rows:
        return similarity_table.copy()

    combined = pd.concat(
        [similarity_table, pd.DataFrame(pooled_rows)],
        axis=0,
        ignore_index=True,
    )
    combined["comparison_label"] = pd.Categorical(
        combined["comparison_label"],
        categories=COMPARISON_LABEL_ORDER,
        ordered=True,
    )
    return combined.sort_values(
        ["unit", "comparison_label"],
        kind="stable",
    ).reset_index(drop=True)


def annotate_pairwise_similarity_significance(
    similarity_table: pd.DataFrame,
    *,
    region: str,
    epoch: str,
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    bins: np.ndarray,
    similarity_metric: str,
    n_shuffles: int,
    shuffle_seed: int,
) -> pd.DataFrame:
    """Append per-unit significance to the direct pairwise rows in one epoch table."""
    annotated = similarity_table.copy()
    for column in SIGNIFICANCE_COLUMNS:
        annotated[column] = np.nan

    if annotated.empty:
        return annotated

    direct_mask = annotated["comparison_label"].isin(DIRECT_COMPARISON_LABELS)
    direct_rows = annotated.loc[direct_mask].copy()
    if direct_rows.empty:
        return annotated

    result_rows: list[dict[str, Any]] = []
    for row_index, row in direct_rows.iterrows():
        unit = int(row["unit"])
        comparison_label = str(row["comparison_label"])
        rng = _make_shuffle_rng(
            shuffle_seed,
            region=region,
            epoch=epoch,
            comparison_label=comparison_label,
            unit=unit,
        )
        observed_score = float(row["similarity"])
        unit_spikes = spikes[unit]
        null_scores = np.full(int(n_shuffles), np.nan, dtype=float)

        for shuffle_index in range(int(n_shuffles)):
            shifted_unit_spikes = circular_shift_unit_spikes_on_movement_axis(
                unit_spikes,
                movement_interval,
                rng=rng,
            )
            null_scores[shuffle_index] = compute_unit_pair_similarity(
                unit,
                shifted_unit_spikes,
                task_progression_by_trajectory,
                movement_interval,
                bins,
                similarity_metric,
                trajectory_a=str(row["trajectory_a"]),
                trajectory_b=str(row["trajectory_b"]),
                flip_trajectory_b=bool(row["flip_trajectory_b"]),
            )

        p_value, ge_count = compute_empirical_p_value(observed_score, null_scores)
        valid_null_scores = null_scores[np.isfinite(null_scores)]
        result_rows.append(
            {
                "row_index": int(row_index),
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

    if not result_rows:
        return annotated

    significance_df = pd.DataFrame(result_rows).set_index("row_index").sort_index()
    significance_df["q_value"] = compute_q_values(significance_df["p_value"])
    for column in SIGNIFICANCE_COLUMNS:
        annotated.loc[significance_df.index, column] = significance_df[column]
    return annotated


def build_epoch_comparison_table(
    epoch_table_a: pd.DataFrame,
    epoch_table_b: pd.DataFrame,
    *,
    region: str,
    epoch_a: str,
    epoch_b: str,
) -> pd.DataFrame:
    """Join two per-epoch tables on shared units and matching comparison labels."""
    if epoch_table_a.empty or epoch_table_b.empty:
        return _empty_comparison_table()

    join_columns = ["unit", "comparison_family", "comparison_label"]
    metadata_columns = [
        "side",
        "trajectory_a",
        "trajectory_b",
        "flip_trajectory_b",
    ]
    base_columns = join_columns + metadata_columns + ["similarity"]
    if "firing_rate_hz" in epoch_table_a.columns:
        base_columns.append("firing_rate_hz")

    left = epoch_table_a.loc[:, base_columns].rename(
        columns={
            "similarity": "similarity_epoch_a",
            "firing_rate_hz": "firing_rate_hz_epoch_a",
        }
    )
    right = epoch_table_b.loc[:, base_columns].rename(
        columns={
            "similarity": "similarity_epoch_b",
            "firing_rate_hz": "firing_rate_hz_epoch_b",
        }
    )

    for column in SIGNIFICANCE_COLUMNS:
        if column in epoch_table_a.columns:
            left[f"{column}_epoch_a"] = epoch_table_a[column].to_numpy()
        if column in epoch_table_b.columns:
            right[f"{column}_epoch_b"] = epoch_table_b[column].to_numpy()

    comparison = left.merge(
        right.drop(columns=metadata_columns, errors="ignore"),
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        return _empty_comparison_table()

    comparison.insert(1, "region", region)
    comparison.insert(2, "epoch_a", epoch_a)
    comparison.insert(3, "epoch_b", epoch_b)
    comparison["delta_similarity"] = (
        comparison["similarity_epoch_b"] - comparison["similarity_epoch_a"]
    )
    comparison["comparison_label"] = pd.Categorical(
        comparison["comparison_label"],
        categories=COMPARISON_LABEL_ORDER,
        ordered=True,
    )
    return comparison.sort_values(
        ["unit", "comparison_label"],
        kind="stable",
    ).reset_index(drop=True)


def plot_epoch_similarity_distributions(
    similarity_table: pd.DataFrame,
    *,
    region: str,
    epoch: str,
    similarity_metric: str,
    fig_path: Path,
) -> None:
    """Save one per-epoch histogram figure with left and right panels."""
    import matplotlib.pyplot as plt

    axis_min, axis_max = get_similarity_axis_limits(similarity_metric)
    bins = np.linspace(axis_min, axis_max, 25)
    colors = {
        "left_turn": "tab:blue",
        "left_arm": "tab:orange",
        "right_turn": "tab:blue",
        "right_arm": "tab:orange",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, side in zip(axes, ("left", "right"), strict=True):
        side_rows = similarity_table[similarity_table["side"] == side]
        has_data = False
        for label in SIDE_TO_DIRECT_LABELS[side]:
            values = pd.to_numeric(
                side_rows.loc[side_rows["comparison_label"] == label, "similarity"],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            has_data = True
            ax.hist(
                values,
                bins=bins,
                alpha=0.5,
                label=label,
                color=colors[label],
                edgecolor="none",
            )
        if has_data:
            ax.set_xlim(axis_min, axis_max)
            ax.set_xlabel(get_similarity_axis_label(similarity_metric))
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid units", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(f"{side} comparisons")

    axes[0].set_ylabel("Units")
    fig.suptitle(f"{region.upper()} {epoch} similarity distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_similarity_significance(
    similarity_table: pd.DataFrame,
    *,
    region: str,
    epoch: str,
    similarity_metric: str,
    fig_path: Path,
    p_value_threshold: float = P_VALUE_THRESHOLD,
) -> None:
    """Save one per-epoch similarity-vs-significance figure."""
    import matplotlib.pyplot as plt

    axis_min, axis_max = get_similarity_axis_limits(similarity_metric)
    colors = {
        "left_turn": "tab:blue",
        "left_arm": "tab:orange",
        "right_turn": "tab:blue",
        "right_arm": "tab:orange",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, side in zip(axes, ("left", "right"), strict=True):
        side_rows = similarity_table[similarity_table["side"] == side]
        has_data = False
        for label in SIDE_TO_DIRECT_LABELS[side]:
            label_rows = side_rows[side_rows["comparison_label"] == label]
            similarity = pd.to_numeric(label_rows["similarity"], errors="coerce").to_numpy(
                dtype=float
            )
            p_values = pd.to_numeric(label_rows["p_value"], errors="coerce").to_numpy(
                dtype=float
            )
            valid = np.isfinite(similarity) & np.isfinite(p_values) & (p_values > 0.0)
            if not np.any(valid):
                continue
            has_data = True
            ax.scatter(
                similarity[valid],
                -np.log10(p_values[valid]),
                alpha=0.6,
                s=24,
                color=colors[label],
                label=label,
            )
        if has_data:
            ax.set_xlim(axis_min, axis_max)
            ax.axhline(-np.log10(p_value_threshold), color="k", linestyle="--", linewidth=1)
            ax.set_xlabel(get_similarity_axis_label(similarity_metric))
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid units", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(f"{side} comparisons")

    axes[0].set_ylabel("-log10(p_value)")
    fig.suptitle(f"{region.upper()} {epoch} similarity significance", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_comparison(
    comparison_table: pd.DataFrame,
    *,
    region: str,
    epoch_a: str,
    epoch_b: str,
    similarity_metric: str,
    fig_path: Path,
) -> None:
    """Save one cross-epoch comparison figure for the direct labels."""
    import matplotlib.pyplot as plt

    axis_min, axis_max = get_similarity_axis_limits(similarity_metric)
    colors = {
        "left_turn": "tab:blue",
        "left_arm": "tab:orange",
        "right_turn": "tab:blue",
        "right_arm": "tab:orange",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, side in zip(axes, ("left", "right"), strict=True):
        side_rows = comparison_table[comparison_table["side"] == side]
        has_data = False
        for label in SIDE_TO_DIRECT_LABELS[side]:
            label_rows = side_rows[side_rows["comparison_label"] == label]
            x = pd.to_numeric(label_rows["similarity_epoch_a"], errors="coerce").to_numpy(
                dtype=float
            )
            y = pd.to_numeric(label_rows["similarity_epoch_b"], errors="coerce").to_numpy(
                dtype=float
            )
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                continue
            has_data = True
            ax.scatter(
                x[valid],
                y[valid],
                alpha=0.6,
                s=24,
                color=colors[label],
                label=label,
            )
        if has_data:
            ax.plot([axis_min, axis_max], [axis_min, axis_max], "k--", linewidth=1)
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.set_xlabel(f"{epoch_a} {get_similarity_axis_label(similarity_metric).lower()}")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No shared units", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(f"{side} comparisons")

    axes[0].set_ylabel(f"{epoch_b} {get_similarity_axis_label(similarity_metric).lower()}")
    fig.suptitle(f"{region.upper()} {epoch_a} vs {epoch_b} similarity", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_epoch_similarity_table(
    similarity_table: pd.DataFrame,
    *,
    data_dir: Path,
    region: str,
    epoch: str,
    similarity_metric: str,
) -> Path:
    """Write one per-epoch long-form similarity table as parquet."""
    output_path = data_dir / f"{region}_{epoch}_{similarity_metric}_within_epoch_similarity.parquet"
    similarity_table.to_parquet(output_path)
    return output_path


def save_epoch_similarity_figures(
    similarity_table: pd.DataFrame,
    *,
    fig_dir: Path,
    region: str,
    epoch: str,
    similarity_metric: str,
    compute_significance: bool,
) -> list[Path]:
    """Write the per-epoch summary figures for one region and epoch."""
    saved_paths: list[Path] = []
    distribution_path = (
        fig_dir / f"{region}_{epoch}_{similarity_metric}_similarity_distributions.png"
    )
    plot_epoch_similarity_distributions(
        similarity_table,
        region=region,
        epoch=epoch,
        similarity_metric=similarity_metric,
        fig_path=distribution_path,
    )
    saved_paths.append(distribution_path)

    if compute_significance:
        significance_path = (
            fig_dir / f"{region}_{epoch}_{similarity_metric}_similarity_significance.png"
        )
        plot_epoch_similarity_significance(
            similarity_table,
            region=region,
            epoch=epoch,
            similarity_metric=similarity_metric,
            fig_path=significance_path,
        )
        saved_paths.append(significance_path)

    return saved_paths


def save_epoch_comparison_table(
    comparison_table: pd.DataFrame,
    *,
    data_dir: Path,
    region: str,
    epoch_a: str,
    epoch_b: str,
    similarity_metric: str,
) -> Path:
    """Write one cross-epoch similarity comparison table as parquet."""
    output_path = (
        data_dir
        / f"{region}_{epoch_a}_{epoch_b}_{similarity_metric}_similarity_comparison.parquet"
    )
    comparison_table.to_parquet(output_path)
    return output_path


def save_epoch_comparison_figure(
    comparison_table: pd.DataFrame,
    *,
    fig_dir: Path,
    region: str,
    epoch_a: str,
    epoch_b: str,
    similarity_metric: str,
) -> Path:
    """Write one cross-epoch comparison figure."""
    fig_path = fig_dir / f"{region}_{epoch_a}_{epoch_b}_{similarity_metric}_similarity_comparison.png"
    plot_epoch_comparison(
        comparison_table,
        region=region,
        epoch_a=epoch_a,
        epoch_b=epoch_b,
        similarity_metric=similarity_metric,
        fig_path=fig_path,
    )
    return fig_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the tuning similarity script."""
    parser = argparse.ArgumentParser(
        description="Quantify per-epoch task-progression tuning similarity across trajectories"
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
        nargs="+",
        help=(
            "Run epoch labels to analyze. Defaults to all run epochs with usable "
            "cleaned-DLC head position."
        ),
    )
    parser.add_argument(
        "--compare-epochs",
        nargs=2,
        metavar=("EPOCH_A", "EPOCH_B"),
        help=(
            "Optional explicit epoch pair for cross-epoch comparison. Both epochs "
            "must already be part of the analyzed epoch set."
        ),
    )
    parser.add_argument(
        "--similarity-metric",
        choices=("correlation", "absolute_overlap", "shape_overlap"),
        default="correlation",
        help="Similarity metric used to compare trajectory-pair tuning curves.",
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
        "--compute-significance",
        action="store_true",
        help=(
            "Estimate per-unit significance for the direct pairwise rows using "
            "circular spike-time shifts on the movement axis."
        ),
    )
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=DEFAULT_N_SHUFFLES,
        help=(
            "Number of circular-shift surrogates when significance is enabled. "
            f"Default: {DEFAULT_N_SHUFFLES}"
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help=(
            "Random seed used for circular-shift significance. "
            f"Default: {DEFAULT_SHUFFLE_SEED}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the task-progression tuning similarity workflow for one session."""
    args = parse_arguments()
    selected_regions = tuple(args.regions)
    selected_epochs = deduplicate_requested_epochs(args.epochs)
    if args.compute_significance:
        _require_statsmodels()
        if args.n_shuffles < 1:
            raise ValueError("--n-shuffles must be at least 1 when significance is enabled.")

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        selected_run_epochs=selected_epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    compare_epochs = resolve_compare_epochs(args.compare_epochs, session["run_epochs"])

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    task_progression_bins = build_task_progression_bins(args.animal_name)
    tuning_curves_by_region: dict[str, dict[str, dict[str, Any]]] = {}
    for region in selected_regions:
        tuning_curves_by_region[region] = {}
        for epoch in session["run_epochs"]:
            tuning_curves_by_region[region][epoch] = compute_trajectory_task_progression_tuning_curves(
                session["spikes_by_region"][region],
                session["task_progression_by_trajectory"][epoch],
                session["movement_by_run"][epoch],
                bins=task_progression_bins,
            )

    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    per_epoch_tables: dict[tuple[str, str], pd.DataFrame] = {}
    saved_tables: list[Path] = []
    saved_figures: list[Path] = []
    for region in selected_regions:
        for epoch in session["run_epochs"]:
            epoch_firing_rates = pd.Series(
                movement_firing_rates[region][epoch],
                index=list(session["spikes_by_region"][region].keys()),
                dtype=float,
            )
            similarity_table = compute_epoch_similarity_table(
                region=region,
                epoch=epoch,
                tuning_curves_by_trajectory=tuning_curves_by_region[region][epoch],
                epoch_firing_rates=epoch_firing_rates,
                firing_rate_threshold_hz=DEFAULT_REGION_FR_THRESHOLDS[region],
                similarity_metric=args.similarity_metric,
            )
            similarity_table = append_pooled_similarity_rows(similarity_table)
            if args.compute_significance:
                similarity_table = annotate_pairwise_similarity_significance(
                    similarity_table,
                    region=region,
                    epoch=epoch,
                    spikes=session["spikes_by_region"][region],
                    task_progression_by_trajectory=session["task_progression_by_trajectory"][epoch],
                    movement_interval=session["movement_by_run"][epoch],
                    bins=task_progression_bins,
                    similarity_metric=args.similarity_metric,
                    n_shuffles=args.n_shuffles,
                    shuffle_seed=args.shuffle_seed,
                )

            per_epoch_tables[(region, epoch)] = similarity_table
            saved_tables.append(
                save_epoch_similarity_table(
                    similarity_table,
                    data_dir=data_dir,
                    region=region,
                    epoch=epoch,
                    similarity_metric=args.similarity_metric,
                )
            )
            saved_figures.extend(
                save_epoch_similarity_figures(
                    similarity_table,
                    fig_dir=fig_dir,
                    region=region,
                    epoch=epoch,
                    similarity_metric=args.similarity_metric,
                    compute_significance=args.compute_significance,
                )
            )

    comparison_outputs: dict[str, Any] = {}
    if compare_epochs is not None:
        epoch_a, epoch_b = compare_epochs
        saved_comparison_tables: list[Path] = []
        saved_comparison_figures: list[Path] = []
        for region in selected_regions:
            comparison_table = build_epoch_comparison_table(
                per_epoch_tables[(region, epoch_a)],
                per_epoch_tables[(region, epoch_b)],
                region=region,
                epoch_a=epoch_a,
                epoch_b=epoch_b,
            )
            saved_comparison_tables.append(
                save_epoch_comparison_table(
                    comparison_table,
                    data_dir=data_dir,
                    region=region,
                    epoch_a=epoch_a,
                    epoch_b=epoch_b,
                    similarity_metric=args.similarity_metric,
                )
            )
            saved_comparison_figures.append(
                save_epoch_comparison_figure(
                    comparison_table[
                        comparison_table["comparison_label"].isin(DIRECT_COMPARISON_LABELS)
                    ],
                    fig_dir=fig_dir,
                    region=region,
                    epoch_a=epoch_a,
                    epoch_b=epoch_b,
                    similarity_metric=args.similarity_metric,
                )
            )
        saved_tables.extend(saved_comparison_tables)
        saved_figures.extend(saved_comparison_figures)
        comparison_outputs = {
            "compare_epochs": list(compare_epochs),
            "saved_tables": saved_comparison_tables,
            "saved_figures": saved_comparison_figures,
        }

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.tuning_analysis",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": list(selected_regions),
            "epochs": session["run_epochs"],
            "compare_epochs": list(compare_epochs) if compare_epochs is not None else None,
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
            "comparison_outputs": comparison_outputs,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
