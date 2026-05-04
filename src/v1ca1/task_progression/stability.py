from __future__ import annotations

"""Estimate odd/even task-progression tuning stability for one session.

This script loads a task-progression session through the shared helpers, splits
each trajectory's laps into one-indexed odd and even trials, computes separate
normalized task-progression tuning curves for each split, and saves per-unit
odd/even tuning correlations. Summary histograms are saved as one 2x2 figure
per epoch, with selected regions overlaid in each trajectory panel.
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
    TRAJECTORY_TYPES,
    build_task_progression_bins,
    compute_movement_firing_rates,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)
from v1ca1.task_progression.tuning_analysis import (
    compute_similarity_score,
    deduplicate_requested_epochs,
    get_similarity_axis_limits,
)


REGION_COLORS = {
    "v1": "tab:blue",
    "ca1": "tab:orange",
}
DEFAULT_REGION_FR_THRESHOLDS = {"v1": 0.5, "ca1": 0.1}


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned start and end arrays from one IntervalSet-like object."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "IntervalSet start and end arrays must have matching shapes. "
            f"Got {starts.shape} and {ends.shape}."
        )
    return starts, ends


def _intervalset_is_empty(intervals: Any) -> bool:
    """Return whether one IntervalSet contains no intervals."""
    starts, _ends = _extract_interval_bounds(intervals)
    return starts.size == 0


def subset_intervalset(intervals: Any, indices: np.ndarray) -> Any:
    """Return one IntervalSet containing the requested interval indices."""
    starts, ends = _extract_interval_bounds(intervals)
    indices = np.asarray(indices, dtype=int).reshape(-1)
    intervalset_class = intervals.__class__
    return intervalset_class(
        start=np.asarray(starts[indices], dtype=float),
        end=np.asarray(ends[indices], dtype=float),
        time_units="s",
    )


def split_laps_by_odd_even(intervals: Any) -> dict[str, Any]:
    """Split one trajectory's laps into one-indexed odd and even IntervalSets."""
    starts, _ends = _extract_interval_bounds(intervals)
    lap_indices = np.arange(starts.size, dtype=int)
    odd_indices = lap_indices[lap_indices % 2 == 0]
    even_indices = lap_indices[lap_indices % 2 == 1]
    return {
        "odd_indices": odd_indices,
        "even_indices": even_indices,
        "odd_interval": subset_intervalset(intervals, odd_indices),
        "even_interval": subset_intervalset(intervals, even_indices),
    }


def make_fraction_histogram_weights(values: np.ndarray) -> np.ndarray:
    """Return histogram weights that normalize one finite sample vector to fraction."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.full(values.shape, 1.0 / float(values.size), dtype=float)


def _empty_stability_table() -> pd.DataFrame:
    """Return an empty odd/even stability table with the expected schema."""
    return pd.DataFrame(
        {
            "animal_name": pd.Series(dtype=str),
            "date": pd.Series(dtype=str),
            "unit": pd.Series(dtype=int),
            "region": pd.Series(dtype=str),
            "epoch": pd.Series(dtype=str),
            "trajectory_type": pd.Series(dtype=str),
            "firing_rate_hz": pd.Series(dtype=float),
            "stability_correlation": pd.Series(dtype=float),
            "n_odd_trials": pd.Series(dtype=int),
            "n_even_trials": pd.Series(dtype=int),
            "odd_duration_s": pd.Series(dtype=float),
            "even_duration_s": pd.Series(dtype=float),
        }
    )


def build_stability_table_for_tuning_curves(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    trajectory_type: str,
    odd_tuning_curve: Any,
    even_tuning_curve: Any,
    epoch_firing_rates: pd.Series,
    n_odd_trials: int,
    n_even_trials: int,
    odd_duration_s: float,
    even_duration_s: float,
    firing_rate_threshold_hz: float,
) -> pd.DataFrame:
    """Return per-unit odd/even tuning correlations for one trajectory."""
    odd_units = np.asarray(odd_tuning_curve.coords["unit"].values, dtype=int)
    even_units = np.asarray(even_tuning_curve.coords["unit"].values, dtype=int)
    common_units = np.intersect1d(odd_units, even_units)
    if common_units.size == 0:
        return _empty_stability_table()

    rows: list[dict[str, Any]] = []
    for unit in common_units:
        firing_rate_hz = float(epoch_firing_rates.get(int(unit), np.nan))
        if not np.isfinite(firing_rate_hz) or firing_rate_hz <= float(firing_rate_threshold_hz):
            continue

        odd_curve = np.asarray(odd_tuning_curve.sel(unit=unit).values, dtype=float).reshape(-1)
        even_curve = np.asarray(even_tuning_curve.sel(unit=unit).values, dtype=float).reshape(-1)
        rows.append(
            {
                "animal_name": animal_name,
                "date": date,
                "unit": int(unit),
                "region": region,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "firing_rate_hz": firing_rate_hz,
                "stability_correlation": compute_similarity_score(
                    odd_curve,
                    even_curve,
                    similarity_metric="correlation",
                ),
                "n_odd_trials": int(n_odd_trials),
                "n_even_trials": int(n_even_trials),
                "odd_duration_s": float(odd_duration_s),
                "even_duration_s": float(even_duration_s),
            }
        )

    if not rows:
        return _empty_stability_table()
    return pd.DataFrame(rows).sort_values("unit", kind="stable").reset_index(drop=True)


def compute_odd_even_tuning_curves_for_trajectory(
    *,
    spikes: Any,
    task_progression: Any,
    trajectory_interval: Any,
    movement_interval: Any,
    bins: np.ndarray,
) -> tuple[Any | None, Any | None, dict[str, float | int]]:
    """Compute odd and even task-progression tuning curves for one trajectory."""
    import pynapple as nap

    split = split_laps_by_odd_even(trajectory_interval)
    odd_epoch = split["odd_interval"].intersect(movement_interval)
    even_epoch = split["even_interval"].intersect(movement_interval)
    metadata = {
        "n_odd_trials": int(np.asarray(split["odd_indices"]).size),
        "n_even_trials": int(np.asarray(split["even_indices"]).size),
        "odd_duration_s": float(odd_epoch.tot_length()),
        "even_duration_s": float(even_epoch.tot_length()),
    }

    if _intervalset_is_empty(odd_epoch) or _intervalset_is_empty(even_epoch):
        return None, None, metadata

    odd_tuning_curve = nap.compute_tuning_curves(
        data=spikes,
        features=task_progression,
        bins=[bins],
        epochs=odd_epoch,
        feature_names=["tp"],
    )
    even_tuning_curve = nap.compute_tuning_curves(
        data=spikes,
        features=task_progression,
        bins=[bins],
        epochs=even_epoch,
        feature_names=["tp"],
    )
    return odd_tuning_curve, even_tuning_curve, metadata


def compute_epoch_stability_table(
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    bins: np.ndarray,
    epoch_firing_rates: pd.Series,
    firing_rate_threshold_hz: float,
) -> pd.DataFrame:
    """Compute odd/even stability rows for all trajectories in one epoch."""
    tables: list[pd.DataFrame] = []
    for trajectory_type in TRAJECTORY_TYPES:
        odd_tuning_curve, even_tuning_curve, metadata = (
            compute_odd_even_tuning_curves_for_trajectory(
                spikes=spikes,
                task_progression=task_progression_by_trajectory[trajectory_type],
                trajectory_interval=trajectory_intervals[trajectory_type],
                movement_interval=movement_interval,
                bins=bins,
            )
        )
        if odd_tuning_curve is None or even_tuning_curve is None:
            continue

        tables.append(
            build_stability_table_for_tuning_curves(
                animal_name=animal_name,
                date=date,
                region=region,
                epoch=epoch,
                trajectory_type=trajectory_type,
                odd_tuning_curve=odd_tuning_curve,
                even_tuning_curve=even_tuning_curve,
                epoch_firing_rates=epoch_firing_rates,
                n_odd_trials=int(metadata["n_odd_trials"]),
                n_even_trials=int(metadata["n_even_trials"]),
                odd_duration_s=float(metadata["odd_duration_s"]),
                even_duration_s=float(metadata["even_duration_s"]),
                firing_rate_threshold_hz=firing_rate_threshold_hz,
            )
        )

    if not tables:
        return _empty_stability_table()
    return pd.concat(tables, axis=0, ignore_index=True)


def plot_epoch_stability_histograms(
    stability_table: pd.DataFrame,
    *,
    epoch: str,
    regions: tuple[str, ...],
    fig_path: Path,
) -> None:
    """Save one 2x2 odd/even stability histogram figure for an epoch."""
    import matplotlib.pyplot as plt

    axis_min, axis_max = get_similarity_axis_limits("correlation")
    bins = np.linspace(axis_min, axis_max, 25)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    for ax, trajectory_type in zip(axes.ravel(), TRAJECTORY_TYPES, strict=True):
        trajectory_rows = stability_table[stability_table["trajectory_type"] == trajectory_type]
        has_data = False
        for region in regions:
            values = pd.to_numeric(
                trajectory_rows.loc[
                    trajectory_rows["region"] == region,
                    "stability_correlation",
                ],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            has_data = True
            ax.hist(
                values,
                bins=bins,
                weights=make_fraction_histogram_weights(values),
                alpha=0.55,
                color=REGION_COLORS.get(region, None),
                edgecolor="none",
                label=region.upper(),
            )

        ax.set_title(trajectory_type.replace("_", " "))
        ax.set_xlim(axis_min, axis_max)
        if has_data:
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No valid units", ha="center", va="center")

    for ax in axes[-1, :]:
        ax.set_xlabel("Odd/even correlation")
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of units")

    fig.suptitle(f"{epoch} task-progression stability", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_stability_table(stability_table: pd.DataFrame, *, data_dir: Path) -> Path:
    """Write the selected-session odd/even stability table as parquet."""
    output_path = data_dir / "odd_even_task_progression_stability.parquet"
    stability_table.to_parquet(output_path)
    return output_path


def save_stability_figures(
    stability_table: pd.DataFrame,
    *,
    fig_dir: Path,
    regions: tuple[str, ...],
    epochs: list[str],
) -> list[Path]:
    """Write one odd/even stability histogram figure per epoch."""
    saved_paths: list[Path] = []
    for epoch in epochs:
        fig_path = fig_dir / f"{epoch}_odd_even_task_progression_stability.png"
        plot_epoch_stability_histograms(
            stability_table[stability_table["epoch"] == epoch],
            epoch=epoch,
            regions=regions,
            fig_path=fig_path,
        )
        saved_paths.append(fig_path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for odd/even tuning stability."""
    parser = argparse.ArgumentParser(
        description="Compute odd/even task-progression tuning stability"
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
        "--epoch",
        nargs="+",
        help=(
            "Run epoch labels to analyze. Defaults to all run epochs with usable "
            "cleaned-DLC head position."
        ),
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
    return parser.parse_args()


def main() -> None:
    """Run odd/even task-progression stability analysis for one session."""
    args = parse_arguments()
    selected_regions = tuple(dict.fromkeys(args.regions))
    selected_epochs = deduplicate_requested_epochs(args.epochs)

    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        regions=selected_regions,
        selected_run_epochs=selected_epochs,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    fig_dir = get_task_progression_figure_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    task_progression_bins = build_task_progression_bins(args.animal_name)
    firing_rate_thresholds = {
        region: DEFAULT_REGION_FR_THRESHOLDS[region] for region in selected_regions
    }
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )

    tables: list[pd.DataFrame] = []
    for region in selected_regions:
        for epoch in session["run_epochs"]:
            epoch_firing_rates = pd.Series(
                movement_firing_rates[region][epoch],
                index=list(session["spikes_by_region"][region].keys()),
                dtype=float,
            )
            tables.append(
                compute_epoch_stability_table(
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    spikes=session["spikes_by_region"][region],
                    task_progression_by_trajectory=session[
                        "task_progression_by_trajectory"
                    ][epoch],
                    trajectory_intervals=session["trajectory_intervals"][epoch],
                    movement_interval=session["movement_by_run"][epoch],
                    bins=task_progression_bins,
                    epoch_firing_rates=epoch_firing_rates,
                    firing_rate_threshold_hz=firing_rate_thresholds[region],
                )
            )

    stability_table = (
        pd.concat(tables, axis=0, ignore_index=True) if tables else _empty_stability_table()
    )
    saved_table = save_stability_table(stability_table, data_dir=data_dir)
    saved_figures = save_stability_figures(
        stability_table,
        fig_dir=fig_dir,
        regions=selected_regions,
        epochs=session["run_epochs"],
    )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.stability",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "regions": list(selected_regions),
            "epochs": session["run_epochs"],
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "firing_rate_thresholds_hz": firing_rate_thresholds,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "task_progression_bin_count": int(len(task_progression_bins) - 1),
            "saved_table": saved_table,
            "saved_figures": saved_figures,
        },
    )
    print(f"Saved stability table to {saved_table}")
    print(f"Saved {len(saved_figures)} figure(s)")
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
