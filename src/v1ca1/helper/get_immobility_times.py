from __future__ import annotations

"""Compute movement and immobility intervals for one session.

This script loads per-epoch position timestamps plus cleaned DLC head position
from the analysis folder, selects only epochs with usable head position,
computes speed for each selected epoch, uses pynapple thresholding to define
movement periods, and defines immobility as the complement of movement within
the full epoch interval.

The canonical outputs are root-level parquet tables (`run_times.parquet` and
`immobility_times.parquet`) with one row per interval and columns `start`,
`end`, and `epoch`. The script prefers `timestamps_position.npz` when
available and readable, and otherwise falls back to `timestamps_position.pkl`.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
    DEFAULT_DATA_ROOT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SPEED_SIGMA_S,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    build_movement_interval,
    build_speed_tsd,
    coerce_position_array,
    get_analysis_path,
    get_position_sampling_rate,
    load_position_data_with_precedence,
    load_position_timestamps,
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


def has_any_finite_position(position_xy: np.ndarray | None) -> bool:
    """Return whether one XY position array contains at least one finite sample."""
    if position_xy is None:
        return False
    position_array = np.asarray(position_xy, dtype=float)
    return position_array.size > 0 and np.isfinite(position_array).any()


def select_epochs_with_usable_head_position(
    epoch_tags: list[str],
    *,
    position_by_epoch: dict[str, np.ndarray],
    position_source: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return epochs with usable head position plus skipped-epoch reasons."""
    usable_epochs: list[str] = []
    skipped_epochs: list[dict[str, str]] = []
    for epoch in epoch_tags:
        head_position = position_by_epoch.get(epoch)
        if head_position is None:
            skipped_epochs.append(
                {"epoch": epoch, "reason": f"head position missing from {position_source}"}
            )
            continue
        if not has_any_finite_position(head_position):
            skipped_epochs.append(
                {"epoch": epoch, "reason": f"head position is all NaN in {position_source}"}
            )
            continue
        usable_epochs.append(epoch)
    return usable_epochs, skipped_epochs


def save_interval_table_output(
    analysis_path: Path,
    *,
    output_name: str,
    intervals_by_epoch: dict[str, "nap.IntervalSet"],
) -> Path:
    """Write one canonical parquet table of interval rows."""
    rows: list[dict[str, float | str]] = []

    for epoch, intervals in intervals_by_epoch.items():
        start_array = np.asarray(intervals.start, dtype=float)
        end_array = np.asarray(intervals.end, dtype=float)
        if start_array.size == 0:
            continue
        if start_array.shape != end_array.shape:
            raise ValueError(
                f"Mismatched start/end interval arrays for {output_name!r} in epoch {epoch!r}."
            )
        for start, end in zip(start_array.tolist(), end_array.tolist(), strict=True):
            rows.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "epoch": str(epoch),
                }
            )

    interval_table = pd.DataFrame.from_records(rows, columns=["start", "end", "epoch"])
    if not interval_table.empty:
        interval_table = interval_table.sort_values(
            by=["start", "end", "epoch"],
            kind="stable",
        ).reset_index(drop=True)

    output_path = analysis_path / f"{output_name}.parquet"
    interval_table.to_parquet(output_path, index=False)
    return output_path


def get_immobility_times(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
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
    position_dict, position_source = load_position_data_with_precedence(
        analysis_path,
        position_source="clean_dlc_head",
        clean_dlc_input_dirname=DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
        clean_dlc_input_name=DEFAULT_CLEAN_DLC_POSITION_NAME,
        validate_timestamps=True,
    )
    selected_epochs, skipped_epochs = select_epochs_with_usable_head_position(
        epoch_tags,
        position_by_epoch=position_dict,
        position_source=position_source,
    )
    if skipped_epochs:
        print(f"Skipping epochs without usable head position: {skipped_epochs!r}")
    if not selected_epochs:
        raise ValueError(
            "No epochs have usable head position in the combined cleaned DLC position parquet. "
            f"Checked {position_source}."
        )

    print(f"Processing {animal_name} {date}.")

    run_intervals: dict[str, "nap.IntervalSet"] = {}
    immobility_intervals: dict[str, "nap.IntervalSet"] = {}
    epoch_summaries: dict[str, dict[str, float]] = {}

    for epoch in selected_epochs:
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
        "position_source": position_source,
        "selected_epochs": selected_epochs,
        "skipped_epochs_unusable_head_position": skipped_epochs,
        "epoch_summaries": epoch_summaries,
        "run_times_path": save_interval_table_output(
            analysis_path=analysis_path,
            output_name="run_times",
            intervals_by_epoch=run_intervals,
        ),
        "immobility_times_path": save_interval_table_output(
            analysis_path=analysis_path,
            output_name="immobility_times",
            intervals_by_epoch=immobility_intervals,
        ),
    }

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
    return parser.parse_args()


def main() -> None:
    """Run the movement-state export CLI."""
    args = parse_arguments()
    get_immobility_times(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
