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
"""

import argparse
from pathlib import Path
from typing import Any

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


def compute_similarity_outputs(
    tuning_curves_by_region: dict[str, dict[str, dict[str, Any]]],
    movement_firing_rates: dict[str, dict[str, np.ndarray]],
    spikes_by_region: dict[str, Any],
    light_epoch: str,
    dark_epoch: str,
    similarity_metric: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Compute left, right, and pooled similarity tables for each region."""
    outputs: dict[str, dict[str, pd.DataFrame]] = {}
    for region in REGIONS:
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
    return parser.parse_args()


def main() -> None:
    """Run the task-progression tuning similarity workflow for one session."""
    args = parse_arguments()
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
    for region in REGIONS:
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
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        similarity_metric=args.similarity_metric,
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

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.tuning_analysis",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "light_epoch": light_epoch,
            "dark_epoch": dark_epoch,
            "similarity_metric": args.similarity_metric,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
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
