from __future__ import annotations

"""Plot one unit's swapped-segment GLM predictions with rasters and observed rate."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.helper.wtrack import get_wtrack_total_length
from v1ca1.task_progression._session import (
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    TRAJECTORY_TYPES,
    get_analysis_path,
    get_task_progression_figure_dir,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)


DEFAULT_TASK_MODEL = "task_segment_bump"
DEFAULT_SELECTION_METRIC = "test_light_swapped_ll_per_spike"
TASK_MODELS = ("task_segment_bump", "task_segment_scalar", "task_dense_gain")
SELECT_BY_CHOICES = ("mean", "task", "visual", "task_minus_visual")


def _configure_runtime_caches() -> None:
    """Point runtime caches at writable directories before heavy imports."""
    temp_root = Path(tempfile.gettempdir())
    numba_cache_dir = temp_root / "v1ca1-numba-cache"
    matplotlib_cache_dir = temp_root / "v1ca1-matplotlib-cache"
    xdg_cache_dir = temp_root / "v1ca1-xdg-cache"
    for cache_dir in (numba_cache_dir, matplotlib_cache_dir, xdg_cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def _find_model_dataset_path(
    *,
    analysis_path: Path,
    region: str,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    model_name: str,
) -> Path:
    """Return the unique selected-ridge dataset path for one model."""
    data_dir = get_task_progression_output_dir(analysis_path, "swap_glm_comparison")
    pattern = (
        f"{region}_{dark_train_epoch}_traindark_"
        f"{light_train_epoch}_trainlight_"
        f"{light_test_epoch}_testlight_"
        f"{model_name}_ridge*.nc"
    )
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one dataset matching {pattern!r} in {data_dir}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _load_fit_parameters(dataset: Any) -> dict[str, Any]:
    """Return fit parameters stored in a swap-GLM NetCDF dataset."""
    raw = dataset.attrs.get("fit_parameters_json", "{}")
    if not raw:
        return {}
    return json.loads(str(raw))


def _dataset_parameter(dataset: Any, key: str, default: Any) -> Any:
    """Return one saved fit parameter with a fallback default."""
    fit_parameters = _load_fit_parameters(dataset)
    return fit_parameters.get(key, default)


def _validate_matching_datasets(task_dataset: Any, visual_dataset: Any) -> None:
    """Require task and visual datasets to describe the same session and units."""
    attrs_to_match = (
        "animal_name",
        "date",
        "region",
        "dark_train_epoch",
        "light_train_epoch",
        "light_test_epoch",
    )
    for attr_name in attrs_to_match:
        if task_dataset.attrs.get(attr_name) != visual_dataset.attrs.get(attr_name):
            raise ValueError(
                f"Task and visual datasets disagree on {attr_name}: "
                f"{task_dataset.attrs.get(attr_name)!r} vs "
                f"{visual_dataset.attrs.get(attr_name)!r}."
            )

    task_units = np.asarray(task_dataset.coords["unit"].values)
    visual_units = np.asarray(visual_dataset.coords["unit"].values)
    if task_units.shape != visual_units.shape or not np.all(task_units == visual_units):
        raise ValueError("Task and visual datasets have different unit coordinates.")


def choose_unit_id(
    *,
    task_dataset: Any,
    visual_dataset: Any,
    metric_name: str,
    select_by: str,
) -> tuple[int, float]:
    """Choose the unit with the largest requested score."""
    if metric_name not in task_dataset or metric_name not in visual_dataset:
        raise ValueError(
            f"Metric {metric_name!r} must exist in both task and visual datasets."
        )

    task_score = task_dataset[metric_name].mean("trajectory", skipna=True)
    visual_score = visual_dataset[metric_name].mean("trajectory", skipna=True)
    if select_by == "task":
        score = task_score
    elif select_by == "visual":
        score = visual_score
    elif select_by == "mean":
        score = (task_score + visual_score) / 2.0
    elif select_by == "task_minus_visual":
        score = task_score - visual_score
    else:
        raise ValueError(f"Unknown select_by value: {select_by!r}")

    values = np.asarray(score.values, dtype=float)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError(
            f"No finite unit scores for metric={metric_name!r}, select_by={select_by!r}."
        )
    masked_values = np.where(finite_mask, values, -np.inf)
    unit_index = int(np.argmax(masked_values))
    unit_id = int(np.asarray(score.coords["unit"].values)[unit_index])
    return unit_id, float(values[unit_index])


def _get_test_lap_bounds(
    dataset: Any,
    trajectory: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-out test-lap start and end times for one trajectory."""
    starts = np.asarray(
        dataset["test_lap_start_s"].sel(trajectory=trajectory).values,
        dtype=float,
    )
    ends = np.asarray(
        dataset["test_lap_end_s"].sel(trajectory=trajectory).values,
        dtype=float,
    )
    splits = np.asarray(
        dataset["test_lap_split"].sel(trajectory=trajectory).values,
        dtype=str,
    )
    valid_mask = np.isfinite(starts) & np.isfinite(ends) & (splits == "test")
    return starts[valid_mask], ends[valid_mask]


def _interpolate_tsd_values(tsd: Any, query_times_s: np.ndarray) -> np.ndarray:
    """Linearly interpolate one pynapple Tsd at query times."""
    query_times_s = np.asarray(query_times_s, dtype=float).reshape(-1)
    if query_times_s.size == 0:
        return np.asarray([], dtype=float)

    times = np.asarray(tsd.t, dtype=float).reshape(-1)
    values = np.asarray(tsd.d, dtype=float).reshape(-1)
    valid_mask = np.isfinite(times) & np.isfinite(values)
    times = times[valid_mask]
    values = values[valid_mask]
    if times.size < 2:
        return np.full(query_times_s.shape, np.nan, dtype=float)

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    return np.interp(query_times_s, times, values, left=np.nan, right=np.nan)


def _collect_trial_spike_positions(
    *,
    spike_times_s: np.ndarray,
    task_progression: Any,
    lap_start_s: np.ndarray,
    lap_end_s: np.ndarray,
    segment_start: float,
    segment_end: float,
) -> list[np.ndarray]:
    """Return one normalized TP spike-position vector per held-out test lap."""
    trial_positions: list[np.ndarray] = []
    for start_s, end_s in zip(lap_start_s, lap_end_s, strict=True):
        trial_spike_times = spike_times_s[
            (spike_times_s > float(start_s)) & (spike_times_s <= float(end_s))
        ]
        positions = _interpolate_tsd_values(task_progression, trial_spike_times)
        valid_mask = (
            np.isfinite(positions)
            & (positions >= float(segment_start))
            & (positions <= float(segment_end))
        )
        trial_positions.append(np.asarray(positions[valid_mask], dtype=float))
    return trial_positions


def _plot_one_trajectory(
    *,
    fig: Any,
    subplot_spec: Any,
    trajectory: str,
    unit_id: int,
    task_dataset: Any,
    visual_dataset: Any,
    trial_positions: list[np.ndarray],
    total_length_cm: float,
    show_xlabel: bool,
) -> tuple[Any, Any]:
    """Plot raster and rate axes for one trajectory."""
    inner = subplot_spec.subgridspec(2, 1, height_ratios=(1.0, 2.2), hspace=0.06)
    raster_axis = fig.add_subplot(inner[0])
    rate_axis = fig.add_subplot(inner[1], sharex=raster_axis)

    segment_start = float(task_dataset["swap_segment_start"].sel(trajectory=trajectory))
    segment_end = float(task_dataset["swap_segment_end"].sel(trajectory=trajectory))
    x_min_cm = segment_start * total_length_cm
    x_max_cm = segment_end * total_length_cm

    for trial_index, positions in enumerate(trial_positions, start=1):
        if positions.size == 0:
            continue
        raster_axis.plot(
            positions * total_length_cm,
            np.full(positions.shape, trial_index, dtype=float),
            "|",
            color="black",
            markersize=4.0,
        )
    raster_axis.set_ylim(0.0, max(1, len(trial_positions)) + 1.0)
    raster_axis.set_ylabel("Test laps")
    raster_axis.tick_params(axis="x", labelbottom=False)
    raster_axis.set_title(
        f"{trajectory} | swapped segment {segment_start:.2f}-{segment_end:.2f}"
    )

    grid_cm = np.asarray(task_dataset.coords["tp_grid"].values, dtype=float) * total_length_cm
    observed_bin_cm = (
        np.asarray(task_dataset.coords["tp_observed_bin"].values, dtype=float)
        * total_length_cm
    )
    segment_grid_mask = (grid_cm >= x_min_cm) & (grid_cm <= x_max_cm)
    segment_bin_mask = (observed_bin_cm >= x_min_cm) & (observed_bin_cm <= x_max_cm)

    empirical_rate = np.asarray(
        task_dataset["test_light_observed_rate_hz"]
        .sel(trajectory=trajectory, unit=unit_id)
        .values,
        dtype=float,
    )
    visual_rate = np.asarray(
        visual_dataset["test_light_swapped_hz_grid"]
        .sel(trajectory=trajectory, unit=unit_id)
        .values,
        dtype=float,
    )
    task_rate = np.asarray(
        task_dataset["test_light_swapped_hz_grid"]
        .sel(trajectory=trajectory, unit=unit_id)
        .values,
        dtype=float,
    )

    rate_axis.plot(
        grid_cm[segment_grid_mask],
        visual_rate[segment_grid_mask],
        color="#0072B2",
        linewidth=2.0,
        label="Visual swapped prediction",
    )
    rate_axis.plot(
        grid_cm[segment_grid_mask],
        task_rate[segment_grid_mask],
        color="#D55E00",
        linewidth=2.0,
        label="Task swapped prediction",
    )
    finite_empirical = segment_bin_mask & np.isfinite(empirical_rate)
    if np.any(finite_empirical):
        rate_axis.plot(
            observed_bin_cm[finite_empirical],
            empirical_rate[finite_empirical],
            color="0.25",
            linewidth=1.5,
            marker="o",
            markersize=3.0,
            label="Empirical test rate",
        )

    rate_axis.set_xlim(x_min_cm, x_max_cm)
    rate_axis.set_ylim(bottom=0.0)
    rate_axis.set_ylabel("Rate (Hz)")
    if show_xlabel:
        rate_axis.set_xlabel("Task progression position (cm)")
    else:
        rate_axis.tick_params(axis="x", labelbottom=False)
    return raster_axis, rate_axis


def plot_unit_swap_figure(
    *,
    task_dataset: Any,
    visual_dataset: Any,
    spike_times_s: np.ndarray,
    task_progression_by_trajectory: dict[str, Any],
    unit_id: int,
    task_model: str,
    out_path: Path,
    show: bool,
) -> Path:
    """Save one unit-level swapped-segment prediction and raster figure."""
    import matplotlib.pyplot as plt

    total_length_cm = get_wtrack_total_length(str(task_dataset.attrs["animal_name"]))
    fig = plt.figure(figsize=(13.5, 8.0), constrained_layout=True)
    outer = fig.add_gridspec(2, 2)
    rate_axes: list[Any] = []

    for trajectory_index, trajectory in enumerate(TRAJECTORY_TYPES):
        lap_start_s, lap_end_s = _get_test_lap_bounds(task_dataset, trajectory)
        trial_positions = _collect_trial_spike_positions(
            spike_times_s=spike_times_s,
            task_progression=task_progression_by_trajectory[trajectory],
            lap_start_s=lap_start_s,
            lap_end_s=lap_end_s,
            segment_start=float(task_dataset["swap_segment_start"].sel(trajectory=trajectory)),
            segment_end=float(task_dataset["swap_segment_end"].sel(trajectory=trajectory)),
        )
        _, rate_axis = _plot_one_trajectory(
            fig=fig,
            subplot_spec=outer[trajectory_index // 2, trajectory_index % 2],
            trajectory=trajectory,
            unit_id=unit_id,
            task_dataset=task_dataset,
            visual_dataset=visual_dataset,
            trial_positions=trial_positions,
            total_length_cm=total_length_cm,
            show_xlabel=trajectory_index >= 2,
        )
        rate_axes.append(rate_axis)

    handles, labels = rate_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3)
    fig.suptitle(
        f"{task_dataset.attrs['animal_name']} {task_dataset.attrs['date']} "
        f"{task_dataset.attrs['region'].upper()} unit {unit_id} | "
        f"{task_model} vs visual | held-out {task_dataset.attrs['light_test_epoch']}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def build_output_path(
    *,
    output_dir: Path,
    region: str,
    unit_id: int,
    dark_train_epoch: str,
    light_train_epoch: str,
    light_test_epoch: str,
    task_model: str,
) -> Path:
    """Return the output figure path for one unit/model setup."""
    stem = (
        f"{region}_{unit_id}_"
        f"{dark_train_epoch}_traindark_"
        f"{light_train_epoch}_trainlight_"
        f"{light_test_epoch}_testlight_"
        f"{task_model}_vs_visual_swapped_segment"
    )
    return output_dir / f"{stem}.png"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the unit-level swapped GLM figure."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot visual/task swapped-segment GLM predictions, held-out rasters, "
            "and empirical held-out rate for one unit."
        )
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument("--date", required=True, help="Session date in YYYYMMDD format")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base analysis directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default="v1",
        help="Region to plot. Default: v1",
    )
    parser.add_argument("--dark-train-epoch", required=True, help="Dark train epoch")
    parser.add_argument("--light-train-epoch", required=True, help="Light train epoch")
    parser.add_argument("--light-test-epoch", required=True, help="Held-out light epoch")
    parser.add_argument(
        "--task-model",
        choices=TASK_MODELS,
        default=DEFAULT_TASK_MODEL,
        help=f"Task model to compare against visual. Default: {DEFAULT_TASK_MODEL}",
    )
    parser.add_argument(
        "--unit-id",
        type=int,
        help="Unit ID to plot. Default: auto-select from --selection-metric.",
    )
    parser.add_argument(
        "--selection-metric",
        default=DEFAULT_SELECTION_METRIC,
        help=f"Metric used when --unit-id is omitted. Default: {DEFAULT_SELECTION_METRIC}",
    )
    parser.add_argument(
        "--select-by",
        choices=SELECT_BY_CHOICES,
        default="mean",
        help=(
            "How to auto-select the unit when --unit-id is omitted. "
            "Default: mean of task and visual scores."
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        help=(
            "Number of leading position samples to ignore. Default: read from "
            "the GLM dataset, falling back to the task-progression default."
        ),
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        help=(
            "Movement speed threshold used by the session loader. Default: read "
            "from the GLM dataset, falling back to the task-progression default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Default: task-progression swap GLM figure directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure in addition to saving it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the unit-level swapped GLM plotting workflow."""
    import xarray as xr

    _configure_runtime_caches()
    args = parse_arguments(argv)
    analysis_path = get_analysis_path(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    visual_path = _find_model_dataset_path(
        analysis_path=analysis_path,
        region=args.region,
        dark_train_epoch=args.dark_train_epoch,
        light_train_epoch=args.light_train_epoch,
        light_test_epoch=args.light_test_epoch,
        model_name="visual",
    )
    task_path = _find_model_dataset_path(
        analysis_path=analysis_path,
        region=args.region,
        dark_train_epoch=args.dark_train_epoch,
        light_train_epoch=args.light_train_epoch,
        light_test_epoch=args.light_test_epoch,
        model_name=args.task_model,
    )

    with xr.open_dataset(visual_path) as visual_dataset, xr.open_dataset(task_path) as task_dataset:
        _validate_matching_datasets(task_dataset, visual_dataset)
        if args.unit_id is None:
            unit_id, selection_score = choose_unit_id(
                task_dataset=task_dataset,
                visual_dataset=visual_dataset,
                metric_name=args.selection_metric,
                select_by=args.select_by,
            )
            print(
                f"Selected unit {unit_id} by {args.select_by} "
                f"{args.selection_metric} score={selection_score:.6g}."
            )
        else:
            unit_id = int(args.unit_id)
            units = set(np.asarray(task_dataset.coords["unit"].values, dtype=int).tolist())
            if unit_id not in units:
                raise ValueError(
                    f"Requested unit {unit_id} is not in the saved {args.region} dataset."
                )

        position_offset = int(
            args.position_offset
            if args.position_offset is not None
            else _dataset_parameter(task_dataset, "position_offset", DEFAULT_POSITION_OFFSET)
        )
        speed_threshold_cm_s = float(
            args.speed_threshold_cm_s
            if args.speed_threshold_cm_s is not None
            else _dataset_parameter(
                task_dataset,
                "speed_threshold_cm_s",
                DEFAULT_SPEED_THRESHOLD_CM_S,
            )
        )

        session = prepare_task_progression_session(
            animal_name=args.animal_name,
            date=args.date,
            data_root=args.data_root,
            regions=(args.region,),
            selected_run_epochs=[args.light_test_epoch],
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            load_body_position=False,
        )
        spike_times_s = np.asarray(
            session["spikes_by_region"][args.region][unit_id].t,
            dtype=float,
        )
        output_dir = (
            args.output_dir
            if args.output_dir is not None
            else get_task_progression_figure_dir(analysis_path, "swap_glm_comparison")
            / "unit_swapped_segment"
        )
        out_path = build_output_path(
            output_dir=output_dir,
            region=args.region,
            unit_id=unit_id,
            dark_train_epoch=args.dark_train_epoch,
            light_train_epoch=args.light_train_epoch,
            light_test_epoch=args.light_test_epoch,
            task_model=args.task_model,
        )
        saved_path = plot_unit_swap_figure(
            task_dataset=task_dataset,
            visual_dataset=visual_dataset,
            spike_times_s=spike_times_s,
            task_progression_by_trajectory=session["task_progression_by_trajectory"][
                args.light_test_epoch
            ],
            unit_id=unit_id,
            task_model=args.task_model,
            out_path=out_path,
            show=args.show,
        )

    print(f"Saved swapped-segment unit figure to {saved_path}")


if __name__ == "__main__":
    main()
