from __future__ import annotations

"""Compute trajectory tuning curves for a legacy combined multiday dataset.

The combined multiday recordings store position and timestamps as nested
``date -> epoch`` pickle mappings. This workflow adapts those inputs to the
shared task-progression preprocessing helpers and uses Pynapple to compute
movement-restricted firing rates for every sorted unit. One NetCDF DataArray
is saved per date, epoch, region, and trajectory.
"""

import argparse
import gc
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    TRAJECTORY_TYPES,
    build_movement_interval,
    build_speed_tsd,
)
from v1ca1.task_progression._session import (
    build_task_progression_bins,
    build_task_progression_by_trajectory,
)
from v1ca1.task_progression.compute_tuning_curves import (
    prepare_tuning_curve_for_save,
)


DEFAULT_ANALYSIS_PATH = Path(
    "/stelmo/kyu/analysis/L14/"
    "20240605_20240606_20240607_20240609_20240611"
)
DEFAULT_ANIMAL_NAME = "L14"
DEFAULT_REGION = "v1"
DEFAULT_EPOCHS = ("02_r1", "08_r4")
DEFAULT_PLACE_BIN_SIZE_CM = 4.0
DEFAULT_OUTPUT_DIRNAME = "multiday_tuning_curves"
DEFAULT_SPIKE_READ_CHUNK_SIZE = 5_000_000


def _load_pickle(path: Path) -> Any:
    """Load one required legacy multiday pickle."""
    if not path.exists():
        raise FileNotFoundError(f"Missing multiday input: {path}")
    with path.open("rb") as file:
        return pickle.load(file)


def get_tuning_curve_path(
    output_dir: Path,
    *,
    region: str,
    date: str,
    epoch: str,
    trajectory: str,
) -> Path:
    """Return the explicit NetCDF path for one saved multiday tuning curve."""
    return Path(output_dir) / (
        f"{region}_{date}_{epoch}_{trajectory}_tuning_curves.nc"
    )


def _trajectory_interval(values: Any) -> Any:
    """Convert legacy trial bounds into a Pynapple IntervalSet."""
    import pynapple as nap

    bounds = np.asarray(values, dtype=float)
    if bounds.size == 0:
        return nap.IntervalSet(
            start=np.array([], dtype=float),
            end=np.array([], dtype=float),
            time_units="s",
        )
    if bounds.ndim == 1:
        if bounds.size < 2:
            raise ValueError("A trajectory interval must contain start and end.")
        starts = np.asarray([bounds[0]], dtype=float)
        ends = np.asarray([bounds[-1]], dtype=float)
    elif bounds.ndim == 2 and bounds.shape[1] >= 2:
        starts = bounds[:, 0]
        ends = bounds[:, -1]
    else:
        raise ValueError(
            "Trajectory bounds must have shape (2,) or (n_trials, >=2); "
            f"got {bounds.shape}."
        )
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Trajectory intervals contain non-finite bounds.")
    if np.any(ends <= starts):
        raise ValueError("Every trajectory interval must have end > start.")
    return nap.IntervalSet(start=starts, end=ends, time_units="s")


def _validate_selection(
    *,
    timestamps_position: Mapping[str, Any],
    position_by_date: Mapping[str, Any],
    trajectory_times: Mapping[str, Any],
    dates: Sequence[str],
    epochs: Sequence[str],
    trajectories: Sequence[str],
) -> None:
    """Validate requested day, epoch, and trajectory identifiers."""
    if not dates or len(set(dates)) != len(dates):
        raise ValueError("dates must contain unique identifiers.")
    if not epochs or len(set(epochs)) != len(epochs):
        raise ValueError("epochs must contain unique identifiers.")
    if not trajectories or len(set(trajectories)) != len(trajectories):
        raise ValueError("trajectories must contain unique identifiers.")
    unknown_trajectories = set(trajectories).difference(TRAJECTORY_TYPES)
    if unknown_trajectories:
        raise ValueError(
            f"Unknown trajectories: {sorted(unknown_trajectories)!r}."
        )

    for date in dates:
        for name, mapping in (
            ("timestamps_position.pkl", timestamps_position),
            ("position.pkl", position_by_date),
            ("trajectory_times.pkl", trajectory_times),
        ):
            if date not in mapping:
                raise ValueError(f"Date {date!r} is absent from {name}.")
        for epoch in epochs:
            for name, mapping in (
                ("timestamps_position.pkl", timestamps_position),
                ("position.pkl", position_by_date),
                ("trajectory_times.pkl", trajectory_times),
            ):
                if epoch not in mapping[date]:
                    raise ValueError(
                        f"Epoch {epoch!r} for {date} is absent from {name}."
                    )
            missing_paths = set(trajectories).difference(
                trajectory_times[date][epoch]
            )
            if missing_paths:
                raise ValueError(
                    f"Trajectory data for {date} {epoch} are missing "
                    f"{sorted(missing_paths)!r}."
                )


def _selected_sample_ranges(
    timestamps_ephys_by_date: Mapping[str, Any],
    timestamps_position: Mapping[str, Any],
    *,
    dates: Sequence[str],
    epochs: Sequence[str],
    position_offset: int,
) -> tuple[list[tuple[int, int, int, np.ndarray]], int]:
    """Return global sorting-frame ranges for the requested run epochs."""
    selected_dates = set(dates)
    ranges: list[tuple[int, int, int, np.ndarray]] = []
    frame_offset = 0
    for date, raw_timestamps in timestamps_ephys_by_date.items():
        date = str(date)
        day_timestamps = np.asarray(raw_timestamps, dtype=float)
        if day_timestamps.ndim != 1 or day_timestamps.size == 0:
            raise ValueError(
                f"Ephys timestamps for {date} must be a nonempty 1D array."
            )
        if date in selected_dates:
            for epoch in epochs:
                position_timestamps = np.asarray(
                    timestamps_position[date][epoch],
                    dtype=float,
                )
                if position_timestamps.size <= position_offset:
                    raise ValueError(
                        f"Position offset removes all samples for {date} {epoch}."
                    )
                local_start = int(
                    np.searchsorted(
                        day_timestamps,
                        position_timestamps[position_offset],
                        side="left",
                    )
                )
                local_end = int(
                    np.searchsorted(
                        day_timestamps,
                        position_timestamps[-1],
                        side="right",
                    )
                )
                ranges.append(
                    (
                        frame_offset + local_start,
                        frame_offset + local_end,
                        frame_offset,
                        day_timestamps,
                    )
                )
        frame_offset += day_timestamps.size
    missing_dates = selected_dates.difference(
        str(date) for date in timestamps_ephys_by_date
    )
    if missing_dates:
        raise ValueError(
            "Requested dates are absent from timestamps_ephys_all.pkl: "
            f"{sorted(missing_dates)!r}."
        )
    return ranges, frame_offset


def load_multiday_spikes(
    analysis_path: Path,
    *,
    region: str,
    dates: Sequence[str],
    epochs: Sequence[str],
    timestamps_position: Mapping[str, Any],
    position_offset: int = DEFAULT_POSITION_OFFSET,
    chunk_size: int = DEFAULT_SPIKE_READ_CHUNK_SIZE,
) -> Any:
    """Load requested-epoch spikes with one pass over the combined sorting."""
    import pynapple as nap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    analysis_path = Path(analysis_path)
    sorting_path = analysis_path / f"sorting_{region}"
    info_path = sorting_path / "numpysorting_info.json"
    spikes_path = sorting_path / "spikes.npy"
    if not info_path.exists() or not spikes_path.exists():
        raise FileNotFoundError(
            f"Missing NumPySorting inputs under {sorting_path}."
        )
    with info_path.open("r", encoding="utf-8") as file:
        sorting_info = json.load(file)
    unit_ids = tuple(int(value) for value in sorting_info["unit_ids"])
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("The combined sorting has duplicate unit identifiers.")

    timestamps_ephys_by_date = _load_pickle(
        analysis_path / "timestamps_ephys_all.pkl"
    )
    if not isinstance(timestamps_ephys_by_date, Mapping):
        raise ValueError(
            "timestamps_ephys_all.pkl must contain a date mapping."
        )
    selected_ranges, total_frame_count = _selected_sample_ranges(
        timestamps_ephys_by_date,
        timestamps_position,
        dates=dates,
        epochs=epochs,
        position_offset=position_offset,
    )

    spikes = np.load(spikes_path, mmap_mode="r")
    required_fields = {"sample_index", "unit_index"}
    if not required_fields.issubset(set(spikes.dtype.names or ())):
        raise ValueError(
            f"{spikes_path} lacks NumPySorting fields "
            f"{sorted(required_fields)!r}."
        )
    time_chunks: list[list[np.ndarray]] = [[] for _ in unit_ids]
    for start in range(0, len(spikes), chunk_size):
        block = spikes[start : start + chunk_size]
        sample_indices = np.asarray(block["sample_index"], dtype=np.int64)
        unit_indices = np.asarray(block["unit_index"], dtype=np.int64)
        if unit_indices.size and (
            unit_indices.min() < 0 or unit_indices.max() >= len(unit_ids)
        ):
            raise ValueError("spikes.npy contains an invalid unit_index.")
        for global_start, global_end, day_offset, day_timestamps in selected_ranges:
            selected = (sample_indices >= global_start) & (
                sample_indices < global_end
            )
            if not np.any(selected):
                continue
            selected_units = unit_indices[selected]
            selected_times = day_timestamps[
                sample_indices[selected] - day_offset
            ]
            for unit_index in np.unique(selected_units):
                time_chunks[int(unit_index)].append(
                    np.asarray(
                        selected_times[selected_units == unit_index],
                        dtype=float,
                    )
                )
    if len(spikes) and int(spikes["sample_index"][-1]) >= total_frame_count:
        raise ValueError(
            "The combined sorting extends beyond timestamps_ephys_all.pkl."
        )

    spike_group = nap.TsGroup(
        {
            unit_id: nap.Ts(
                t=(
                    np.concatenate(time_chunks[unit_index])
                    if time_chunks[unit_index]
                    else np.array([], dtype=float)
                ),
                time_units="s",
            )
            for unit_index, unit_id in enumerate(unit_ids)
        },
        time_units="s",
    )
    del timestamps_ephys_by_date, spikes, time_chunks
    gc.collect()
    return spike_group


def _empty_tuning_curve(unit_ids: Sequence[int], bins: np.ndarray) -> Any:
    """Return an all-NaN curve for a path with no trials."""
    import xarray as xr

    centers = 0.5 * (bins[:-1] + bins[1:])
    return xr.DataArray(
        np.full((len(unit_ids), centers.size), np.nan, dtype=float),
        dims=("unit", "tp"),
        coords={"unit": np.asarray(unit_ids, dtype=int), "tp": centers},
        name="firing_rate_hz",
    )


def compute_tuning_curve(
    spikes: Any,
    task_progression: Any,
    movement_interval: Any,
    bins: np.ndarray,
) -> Any:
    """Compute one movement-restricted path curve with Pynapple."""
    import pynapple as nap

    path_interval = task_progression.time_support.intersect(movement_interval)
    starts = np.asarray(path_interval.start, dtype=float).ravel()
    if starts.size == 0 or len(np.asarray(task_progression.t)) == 0:
        return _empty_tuning_curve(tuple(spikes.keys()), bins)
    return nap.compute_tuning_curves(
        data=spikes,
        features=task_progression,
        bins=[bins],
        epochs=path_interval,
        feature_names=["tp"],
    )


def compute_and_save_tuning_curves(
    *,
    analysis_path: Path,
    animal_name: str,
    region: str,
    dates: Sequence[str],
    epochs: Sequence[str],
    trajectories: Sequence[str],
    output_dir: Path,
    place_bin_size_cm: float = DEFAULT_PLACE_BIN_SIZE_CM,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    speed_smoothing_sigma_s: float = DEFAULT_SPEED_SIGMA_S,
) -> tuple[Path, ...]:
    """Compute and save all requested multiday path tuning curves."""
    analysis_path = Path(analysis_path)
    dates = tuple(str(value) for value in dates)
    epochs = tuple(str(value) for value in epochs)
    trajectories = tuple(str(value) for value in trajectories)
    timestamps_position = _load_pickle(
        analysis_path / "timestamps_position.pkl"
    )
    position_by_date = _load_pickle(analysis_path / "position.pkl")
    trajectory_times = _load_pickle(analysis_path / "trajectory_times.pkl")
    if not all(
        isinstance(value, Mapping)
        for value in (timestamps_position, position_by_date, trajectory_times)
    ):
        raise ValueError("Multiday position inputs must contain date mappings.")
    _validate_selection(
        timestamps_position=timestamps_position,
        position_by_date=position_by_date,
        trajectory_times=trajectory_times,
        dates=dates,
        epochs=epochs,
        trajectories=trajectories,
    )
    bins = build_task_progression_bins(
        animal_name,
        place_bin_size_cm=place_bin_size_cm,
    )
    spikes = load_multiday_spikes(
        analysis_path,
        region=region,
        dates=dates,
        epochs=epochs,
        timestamps_position=timestamps_position,
        position_offset=position_offset,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for date in dates:
        for epoch in epochs:
            epoch_timestamps = np.asarray(
                timestamps_position[date][epoch],
                dtype=float,
            )
            epoch_position = np.asarray(
                position_by_date[date][epoch],
                dtype=float,
            )
            intervals = {
                trajectory_type: _trajectory_interval(
                    trajectory_times[date][epoch][trajectory_type]
                )
                for trajectory_type in TRAJECTORY_TYPES
            }
            speed = build_speed_tsd(
                epoch_position,
                epoch_timestamps,
                position_offset=position_offset,
                speed_smoothing_sigma_s=speed_smoothing_sigma_s,
            )
            movement = build_movement_interval(
                speed,
                speed_threshold_cm_s=speed_threshold_cm_s,
            )
            progression_by_trajectory = build_task_progression_by_trajectory(
                animal_name,
                epoch_position,
                epoch_timestamps,
                intervals,
                position_offset=position_offset,
            )
            for trajectory in trajectories:
                tuning_curve = compute_tuning_curve(
                    spikes,
                    progression_by_trajectory[trajectory],
                    movement,
                    bins,
                )
                output = prepare_tuning_curve_for_save(
                    tuning_curve,
                    animal_name=animal_name,
                    date=date,
                    region=region,
                    epoch=epoch,
                    model_name="pynapple_task_progression",
                    trajectory_type=trajectory,
                )
                output.attrs.update(
                    {
                        "n_trials": int(
                            len(
                                np.asarray(
                                    intervals[trajectory].start,
                                    dtype=float,
                                ).ravel()
                            )
                        ),
                        "place_bin_size_cm": float(place_bin_size_cm),
                        "position_offset": int(position_offset),
                        "speed_threshold_cm_s": float(speed_threshold_cm_s),
                        "speed_smoothing_sigma_s": float(
                            speed_smoothing_sigma_s
                        ),
                    }
                )
                output_path = get_tuning_curve_path(
                    output_dir,
                    region=region,
                    date=date,
                    epoch=epoch,
                    trajectory=trajectory,
                )
                output.to_netcdf(output_path)
                print(output_path)
                saved_paths.append(output_path)
    return tuple(saved_paths)


def parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for multiday tuning-curve export."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute Pynapple task-progression tuning curves for a combined "
            "multiday sorting."
        )
    )
    parser.add_argument(
        "--analysis-path",
        type=Path,
        default=DEFAULT_ANALYSIS_PATH,
    )
    parser.add_argument("--animal-name", default=DEFAULT_ANIMAL_NAME)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        help="Date to compute, repeatable. Default: every stored date.",
    )
    parser.add_argument(
        "--epoch",
        action="append",
        dest="epochs",
        help="Run epoch to compute, repeatable. Default: 02_r1 and 08_r4.",
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        dest="trajectories",
        choices=TRAJECTORY_TYPES,
        help="Trajectory to compute, repeatable. Default: all four paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Default: "
            "<analysis-path>/multiday_tuning_curves."
        ),
    )
    parser.add_argument(
        "--place-bin-size-cm",
        type=float,
        default=DEFAULT_PLACE_BIN_SIZE_CM,
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
    )
    parser.add_argument(
        "--speed-smoothing-sigma-s",
        type=float,
        default=DEFAULT_SPEED_SIGMA_S,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> tuple[Path, ...]:
    """Compute the configured multiday tuning curves and record a run log."""
    args = parse_arguments(argv)
    timestamps_position = _load_pickle(
        args.analysis_path / "timestamps_position.pkl"
    )
    dates = (
        tuple(str(date) for date in timestamps_position)
        if args.dates is None
        else tuple(args.dates)
    )
    epochs = DEFAULT_EPOCHS if args.epochs is None else tuple(args.epochs)
    trajectories = (
        TRAJECTORY_TYPES
        if args.trajectories is None
        else tuple(args.trajectories)
    )
    output_dir = (
        args.analysis_path / DEFAULT_OUTPUT_DIRNAME
        if args.output_dir is None
        else args.output_dir
    )
    saved_paths = compute_and_save_tuning_curves(
        analysis_path=args.analysis_path,
        animal_name=args.animal_name,
        region=args.region,
        dates=dates,
        epochs=epochs,
        trajectories=trajectories,
        output_dir=output_dir,
        place_bin_size_cm=args.place_bin_size_cm,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        speed_smoothing_sigma_s=args.speed_smoothing_sigma_s,
    )
    write_run_log(
        analysis_path=args.analysis_path,
        script_name="v1ca1.multiday.compute_tuning_curves",
        parameters={
            "analysis_path": args.analysis_path,
            "animal_name": args.animal_name,
            "region": args.region,
            "dates": dates,
            "epochs": epochs,
            "trajectories": trajectories,
            "place_bin_size_cm": args.place_bin_size_cm,
            "position_offset": args.position_offset,
            "speed_threshold_cm_s": args.speed_threshold_cm_s,
            "speed_smoothing_sigma_s": args.speed_smoothing_sigma_s,
        },
        outputs={"saved_tuning_curves": saved_paths},
    )
    return saved_paths


if __name__ == "__main__":
    main()
