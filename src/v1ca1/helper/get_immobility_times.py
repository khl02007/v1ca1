from __future__ import annotations

"""Compute movement and immobility intervals for one session.

This script loads per-epoch position arrays and position timestamps from the
analysis folder, computes speed for each epoch, uses pynapple thresholding to
define movement periods, and defines immobility as the complement of movement
within the full epoch interval.

By default it writes root-level pynapple `IntervalSet` files
(`run_times.npz` and `immobility_times.npz`). The pynapple files store
the full epoch label as metadata on each interval row. The script prefers
`timestamps_position.npz` when available and readable, and otherwise falls back
to `timestamps_position.pkl`.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    build_movement_interval,
    build_speed_tsd,
    coerce_position_array,
    get_analysis_path,
    get_position_sampling_rate,
    load_position_data,
    load_position_timestamps,
    save_pickle_output,
)

if TYPE_CHECKING:
    import pynapple as nap


def get_epoch_interval(speed_tsd: "nap.Tsd") -> "nap.IntervalSet":
    """Return the full epoch interval covered by one speed Tsd."""
    import pynapple as nap

    return nap.IntervalSet(
        start=float(speed_tsd.t[0]),
        end=float(speed_tsd.t[-1]),
        time_units="s",
    )


def compute_movement_and_immobility_intervals(
    speed_tsd: "nap.Tsd",
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
) -> tuple["nap.IntervalSet", "nap.IntervalSet"]:
    """Return movement and immobility IntervalSets for one epoch."""
    epoch_interval = get_epoch_interval(speed_tsd)
    movement_ep = build_movement_interval(
        speed_tsd=speed_tsd,
        speed_threshold_cm_s=speed_threshold_cm_s,
    )
    immobility_ep = epoch_interval.set_diff(movement_ep)
    return movement_ep, immobility_ep


def intervalset_to_dataframe(intervals: "nap.IntervalSet") -> pd.DataFrame:
    """Convert a pynapple IntervalSet to the legacy dataframe layout."""
    start = np.asarray(intervals.start, dtype=float)
    end = np.asarray(intervals.end, dtype=float)
    return pd.DataFrame(
        {
            "start_time": start,
            "end_time": end,
            "duration": end - start,
        }
    )


def save_legacy_interval_pickle_output(
    analysis_path: Path,
    state_name: str,
    intervals_by_epoch: dict[str, "nap.IntervalSet"],
) -> Path:
    """Write one root-level pickle mapping each epoch to its legacy dataframe."""
    output_path = analysis_path / f"{state_name}.pkl"
    serializable = {
        epoch: intervalset_to_dataframe(intervals)
        for epoch, intervals in intervals_by_epoch.items()
    }
    return save_pickle_output(output_path, serializable)


def save_pynapple_interval_output(
    analysis_path: Path,
    state_name: str,
    intervals_by_epoch: dict[str, "nap.IntervalSet"],
) -> Path:
    """Write one root-level IntervalSet with epoch metadata for one state."""
    import pynapple as nap

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    epochs: list[str] = []

    for epoch, intervals in intervals_by_epoch.items():
        start_array = np.asarray(intervals.start, dtype=float)
        end_array = np.asarray(intervals.end, dtype=float)
        if start_array.size == 0:
            continue
        if start_array.shape != end_array.shape:
            raise ValueError(
                f"Mismatched start/end interval arrays for {state_name!r} in epoch {epoch!r}."
            )

        starts.append(start_array)
        ends.append(end_array)
        epochs.extend([str(epoch)] * start_array.shape[0])

    if starts:
        all_starts = np.concatenate(starts).astype(float, copy=False)
        all_ends = np.concatenate(ends).astype(float, copy=False)
    else:
        all_starts = np.array([], dtype=float)
        all_ends = np.array([], dtype=float)

    interval_set = nap.IntervalSet(start=all_starts, end=all_ends, time_units="s")
    interval_set.set_info(epoch=epochs)

    output_path = analysis_path / f"{state_name}.npz"
    interval_set.save(output_path)
    return output_path


def get_immobility_times(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    save_pkl: bool = False,
) -> None:
    """Compute and save movement and immobility intervals for one session."""
    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_tags, timestamps_position, timestamp_source = load_position_timestamps(analysis_path)
    position_dict = load_position_data(analysis_path, epoch_tags)

    print(f"Processing {animal_name} {date}.")

    run_intervals: dict[str, "nap.IntervalSet"] = {}
    immobility_intervals: dict[str, "nap.IntervalSet"] = {}
    epoch_summaries: dict[str, dict[str, float]] = {}

    for epoch in epoch_tags:
        speed_tsd = build_speed_tsd(
            position=position_dict[epoch],
            timestamps_position=timestamps_position[epoch],
            position_offset=position_offset,
        )
        movement_ep, immobility_ep = compute_movement_and_immobility_intervals(
            speed_tsd=speed_tsd,
            speed_threshold_cm_s=speed_threshold_cm_s,
        )

        run_intervals[epoch] = movement_ep
        immobility_intervals[epoch] = immobility_ep
        epoch_summaries[epoch] = {
            "speed_sample_count": float(len(speed_tsd.t)),
            "movement_interval_count": float(len(movement_ep)),
            "movement_total_duration_s": float(movement_ep.tot_length()),
            "immobility_interval_count": float(len(immobility_ep)),
            "immobility_total_duration_s": float(immobility_ep.tot_length()),
        }

    outputs: dict[str, Any] = {
        "timestamps_position_source": timestamp_source,
        "epochs": epoch_tags,
        "epoch_summaries": epoch_summaries,
        "run_times_pynapple_path": save_pynapple_interval_output(
            analysis_path=analysis_path,
            state_name="run_times",
            intervals_by_epoch=run_intervals,
        ),
        "immobility_times_pynapple_path": save_pynapple_interval_output(
            analysis_path=analysis_path,
            state_name="immobility_times",
            intervals_by_epoch=immobility_intervals,
        ),
    }
    if save_pkl:
        outputs["run_times_pickle_path"] = save_legacy_interval_pickle_output(
            analysis_path=analysis_path,
            state_name="run_times",
            intervals_by_epoch=run_intervals,
        )
        outputs["immobility_times_pickle_path"] = save_legacy_interval_pickle_output(
            analysis_path=analysis_path,
            state_name="immobility_times",
            intervals_by_epoch=immobility_intervals,
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.helper.get_immobility_times",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "position_offset": position_offset,
            "speed_threshold_cm_s": speed_threshold_cm_s,
            "speed_sigma_s": DEFAULT_SPEED_SIGMA_S,
            "save_pkl": save_pkl,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the movement-state export CLI."""
    parser = argparse.ArgumentParser(description="Save run and immobility intervals")
    parser.add_argument(
        "--animal-name",
        required=True,
        help="Animal name",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Recording date in YYYYMMDD format",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--save-pkl",
        action="store_true",
        help="Also write compatibility pickle exports alongside the default .npz outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the movement-state export CLI."""
    args = parse_arguments()
    get_immobility_times(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        save_pkl=args.save_pkl,
    )


if __name__ == "__main__":
    main()
