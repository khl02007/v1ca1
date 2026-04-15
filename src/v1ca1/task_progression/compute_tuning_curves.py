from __future__ import annotations

"""Compute and save trajectory-resolved tuning curves for one session.

This script loads one dataset through the shared task-progression session
helpers, rebuilds trajectory-specific task-progression coordinates for each
run epoch, computes one place tuning curve per trajectory and one
task-progression tuning curve per same-turn trajectory pair, and saves each
curve as its own NetCDF-backed xarray output.

Each saved file preserves one tuning curve as an xarray `DataArray`, written
under `analysis_path / "task_progression_tuning_curves"`. Run metadata is
recorded under `analysis_path / "v1ca1_log"`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.task_progression._session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLACE_BIN_SIZE_CM,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    REGIONS,
    TRAJECTORY_TYPES,
    TURN_TRAJECTORY_PAIRS,
    build_task_progression_bins,
    get_analysis_path,
    get_task_progression_output_dir,
    prepare_task_progression_session,
)
from v1ca1.helper.wtrack import get_wtrack_total_length


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


def build_place_bins(
    animal_name: str,
    place_bin_size_cm: float = DEFAULT_PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    """Return place-tuning bin edges for one single trajectory."""
    if place_bin_size_cm <= 0:
        raise ValueError("--place-bin-size-cm must be positive.")
    total_length = get_wtrack_total_length(animal_name)
    return np.arange(0.0, total_length + place_bin_size_cm, place_bin_size_cm)


def build_place_coordinate(
    task_progression_tsd: Any,
    *,
    total_length_cm: float,
) -> Any:
    """Convert one normalized task-progression Tsd into single-trajectory position."""
    import pynapple as nap

    return nap.Tsd(
        t=np.asarray(task_progression_tsd.t, dtype=float),
        d=np.asarray(task_progression_tsd.d, dtype=float) * total_length_cm,
        time_support=task_progression_tsd.time_support,
        time_units="s",
    )


def build_same_turn_task_progression(
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    turn_type: str,
) -> tuple[Any, Any]:
    """Return one pooled same-turn Tsd and its paired-trajectory interval union."""
    import pynapple as nap

    trajectory_a, trajectory_b = TURN_TRAJECTORY_PAIRS[turn_type]
    start_chunks: list[np.ndarray] = []
    end_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    for trajectory_type in (trajectory_a, trajectory_b):
        trajectory_epoch = task_progression_by_trajectory[trajectory_type].time_support.intersect(
            movement_interval
        )
        if _intervalset_is_empty(trajectory_epoch):
            continue
        starts, ends = _extract_interval_bounds(trajectory_epoch)
        start_chunks.append(starts)
        end_chunks.append(ends)
        restricted = task_progression_by_trajectory[trajectory_type].restrict(trajectory_epoch)
        restricted_times = np.asarray(restricted.t, dtype=float)
        if restricted_times.size == 0:
            continue
        time_chunks.append(restricted_times)
        value_chunks.append(np.asarray(restricted.d, dtype=float))

    if not start_chunks:
        same_turn_interval = nap.IntervalSet(
            start=np.array([], dtype=float),
            end=np.array([], dtype=float),
            time_units="s",
        )
    else:
        starts = np.concatenate(start_chunks)
        ends = np.concatenate(end_chunks)
        order = np.argsort(starts)
        same_turn_interval = nap.IntervalSet(
            start=starts[order],
            end=ends[order],
            time_units="s",
        )

    if not time_chunks:
        return nap.Tsd(
            t=np.array([], dtype=float),
            d=np.array([], dtype=float),
            time_support=same_turn_interval,
            time_units="s",
        ), same_turn_interval

    all_times = np.concatenate(time_chunks)
    all_values = np.concatenate(value_chunks)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=same_turn_interval,
        time_units="s",
    ), same_turn_interval


def compute_place_tuning_curves_for_epoch(
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    position_bins: np.ndarray,
    *,
    animal_name: str,
) -> dict[str, Any]:
    """Compute one place tuning curve per trajectory for one epoch."""
    import pynapple as nap

    total_length_cm = get_wtrack_total_length(animal_name)
    tuning_curves: dict[str, Any] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        trajectory_epoch = task_progression_by_trajectory[trajectory_type].time_support.intersect(
            movement_interval
        )
        if _intervalset_is_empty(trajectory_epoch):
            continue
        place_coordinate = build_place_coordinate(
            task_progression_by_trajectory[trajectory_type],
            total_length_cm=total_length_cm,
        )
        tuning_curves[trajectory_type] = nap.compute_tuning_curves(
            data=spikes,
            features=place_coordinate,
            bins=[position_bins],
            epochs=trajectory_epoch,
            feature_names=["linpos"],
        )
    return tuning_curves


def compute_task_progression_tuning_curves_for_epoch(
    spikes: Any,
    task_progression_by_trajectory: dict[str, Any],
    movement_interval: Any,
    task_progression_bins: np.ndarray,
) -> dict[str, Any]:
    """Compute one same-turn task-progression tuning curve per turn type."""
    import pynapple as nap

    tuning_curves: dict[str, Any] = {}
    for turn_type in TURN_TRAJECTORY_PAIRS:
        task_progression, same_turn_interval = build_same_turn_task_progression(
            task_progression_by_trajectory,
            movement_interval,
            turn_type,
        )
        if _intervalset_is_empty(same_turn_interval):
            continue
        if len(np.asarray(task_progression.t, dtype=float)) == 0:
            continue
        tuning_curves[turn_type] = nap.compute_tuning_curves(
            data=spikes,
            features=task_progression,
            bins=[task_progression_bins],
            epochs=same_turn_interval,
            feature_names=["tp"],
        )
    return tuning_curves


def _make_netcdf_safe_attr_value(value: Any) -> Any:
    """Convert one attribute value into a NetCDF-safe representation."""
    def _to_json_safe(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return [_to_json_safe(item) for item in obj.tolist()]
        if isinstance(obj, dict):
            return {str(key): _to_json_safe(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_safe(item) for item in obj]
        return obj

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return json.dumps(_to_json_safe(value))
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_to_json_safe(value))
    return str(value)


def prepare_tuning_curve_for_save(
    tuning_curve: Any,
    *,
    animal_name: str,
    date: str,
    region: str,
    epoch: str,
    model_name: str,
    trajectory_type: str | None = None,
    turn_type: str | None = None,
) -> Any:
    """Return one tuning curve with NetCDF-safe attrs for xarray export."""
    output = tuning_curve.rename("firing_rate_hz").copy()
    output.attrs = {
        key: _make_netcdf_safe_attr_value(value)
        for key, value in getattr(tuning_curve, "attrs", {}).items()
    }
    output.attrs.update(
        {
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
            "model_name": model_name,
        }
    )
    if trajectory_type is not None:
        output.attrs["trajectory_type"] = trajectory_type
    if turn_type is not None:
        output.attrs["turn_type"] = turn_type
    return output


def save_tuning_curves(
    tuning_curves: dict[str, dict[str, dict[str, Any]]],
    data_dir: Path,
) -> list[Path]:
    """Write one NetCDF tuning curve per region, epoch, and saved curve name."""
    saved_paths: list[Path] = []
    for region, region_curves in tuning_curves.items():
        for epoch, named_curves in region_curves.items():
            for curve_name, tuning_curve in named_curves.items():
                path = data_dir / f"{region}_{epoch}_{curve_name}_tuning_curves.nc"
                tuning_curve.to_netcdf(path)
                print(f"Saved {curve_name} tuning curve for {region} {epoch} to {path}")
                saved_paths.append(path)
    return saved_paths


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the tuning-curve export workflow."""
    parser = argparse.ArgumentParser(
        description="Compute and save place and task-progression tuning curves"
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
    return parser.parse_args()


def main() -> None:
    """Compute and save place and task-progression tuning curves for one session."""
    args = parse_arguments()
    print(
        f"Loading task-progression session for animal={args.animal_name} date={args.date} "
        f"from {args.data_root}"
    )
    session = prepare_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
    )
    print(
        f"Loaded session with {len(session['run_epochs'])} run epoch(s): "
        f"{', '.join(session['run_epochs'])}"
    )

    analysis_path = get_analysis_path(args.animal_name, args.date, args.data_root)
    data_dir = get_task_progression_output_dir(analysis_path, Path(__file__).stem)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving tuning curves under {data_dir}")

    position_bins = build_place_bins(args.animal_name)
    task_progression_bins = build_task_progression_bins(args.animal_name)
    print(
        "Built tuning-curve bins: "
        f"{len(position_bins) - 1} place bins, "
        f"{len(task_progression_bins) - 1} task-progression bins"
    )

    tuning_curves_by_region: dict[str, dict[str, dict[str, Any]]] = {region: {} for region in REGIONS}
    for region in REGIONS:
        print(f"Computing tuning curves for region {region}")
        for epoch in session["run_epochs"]:
            print(f"  Processing epoch {epoch}")
            place_tuning_by_trajectory = compute_place_tuning_curves_for_epoch(
                spikes=session["spikes_by_region"][region],
                task_progression_by_trajectory=session["task_progression_by_trajectory"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                position_bins=position_bins,
                animal_name=args.animal_name,
            )
            task_progression_tuning_by_turn = compute_task_progression_tuning_curves_for_epoch(
                spikes=session["spikes_by_region"][region],
                task_progression_by_trajectory=session["task_progression_by_trajectory"][epoch],
                movement_interval=session["movement_by_run"][epoch],
                task_progression_bins=task_progression_bins,
            )
            tuning_curves_by_region[region][epoch] = {}
            for trajectory_type, place_tuning in place_tuning_by_trajectory.items():
                tuning_curves_by_region[region][epoch][
                    f"place_{trajectory_type}"
                ] = prepare_tuning_curve_for_save(
                    place_tuning,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    model_name="place",
                    trajectory_type=trajectory_type,
                )
            for turn_type, task_progression_tuning in task_progression_tuning_by_turn.items():
                tuning_curves_by_region[region][epoch][
                    f"task_progression_{turn_type}"
                ] = prepare_tuning_curve_for_save(
                    task_progression_tuning,
                    animal_name=args.animal_name,
                    date=args.date,
                    region=region,
                    epoch=epoch,
                    model_name="task_progression",
                    turn_type=turn_type,
                )

    print("Writing NetCDF outputs")
    saved_netcdf_paths = save_tuning_curves(tuning_curves_by_region, data_dir=data_dir)
    print(f"Saved {len(saved_netcdf_paths)} NetCDF file(s)")

    print("Writing run metadata log")
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.task_progression.compute_tuning_curves",
        parameters={
            "animal_name": args.animal_name,
            "date": args.date,
            "data_root": args.data_root,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
        },
        outputs={
            "sources": session["sources"],
            "run_epochs": session["run_epochs"],
            "n_position_bins": int(len(position_bins) - 1),
            "n_task_progression_bins": int(len(task_progression_bins) - 1),
            "saved_netcdf_paths": saved_netcdf_paths,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
