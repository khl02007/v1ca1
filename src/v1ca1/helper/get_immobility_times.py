from __future__ import annotations

"""Compute movement and immobility intervals for one session.

This script loads per-epoch position arrays and position timestamps from the
analysis folder, computes speed for each epoch, uses pynapple thresholding to
define movement periods, and defines immobility as the complement of movement
within the full epoch interval.

By default it writes both legacy root-level pickle artifacts (`run_times.pkl`
and `immobility_times.pkl`) and matching root-level pynapple `IntervalSet`
files (`run_times.npz` and `immobility_times.npz`). The pynapple files store
the full epoch label as metadata on each interval row. The script prefers
`timestamps_position.npz` when available and readable, and otherwise falls back
to `timestamps_position.pkl`.
"""

import argparse
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import position_tools as pt

from v1ca1.helper.run_logging import write_run_log

if TYPE_CHECKING:
    import pynapple as nap


DEFAULT_DATA_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_POSITION_OFFSET = 10
DEFAULT_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_SPEED_SIGMA_S = 0.1


def get_analysis_path(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the analysis directory for one animal/date session."""
    return data_root / animal_name / date


def _extract_epoch_tags_from_position_group(position_group: "nap.TsGroup") -> list[str]:
    """Extract saved epoch labels from a pynapple TsGroup."""
    try:
        epoch_info = position_group["epoch"]
    except Exception:
        epoch_info = None

    if epoch_info is None:
        raise ValueError(
            "timestamps_position.npz does not contain the saved epoch labels needed "
            "to align timestamps with position.pkl."
        )

    epoch_array = np.asarray(epoch_info)
    if epoch_array.size == 0:
        raise ValueError("timestamps_position.npz does not contain any saved epoch labels.")

    return [str(epoch) for epoch in epoch_array.tolist()]


def load_position_timestamps(
    analysis_path: Path,
) -> tuple[list[str], dict[str, np.ndarray], str]:
    """Load per-epoch position timestamps, preferring pynapple outputs."""
    position_npz_path = analysis_path / "timestamps_position.npz"
    npz_error: Exception | None = None
    if position_npz_path.exists():
        try:
            import pynapple as nap
        except ModuleNotFoundError:
            pass
        else:
            try:
                position_group = nap.load_file(position_npz_path)
                epoch_tags = _extract_epoch_tags_from_position_group(position_group)
                if len(epoch_tags) != len(position_group):
                    raise ValueError(
                        "Mismatch between epoch labels and time series count in "
                        f"{position_npz_path}."
                    )
                timestamps_position = {
                    epoch: np.asarray(position_group[index].t, dtype=float)
                    for index, epoch in enumerate(epoch_tags)
                }
                return epoch_tags, timestamps_position, "pynapple"
            except Exception as exc:
                npz_error = exc

    position_pickle_path = analysis_path / "timestamps_position.pkl"
    if not position_pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {position_npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            f"Could not find timestamps_position.npz or timestamps_position.pkl under {analysis_path}."
        )

    if npz_error is not None:
        print(
            "Falling back to timestamps_position.pkl because timestamps_position.npz "
            f"could not be loaded: {npz_error}"
        )

    with open(position_pickle_path, "rb") as f:
        timestamps_position = pickle.load(f)

    epoch_tags = [str(epoch) for epoch in timestamps_position.keys()]
    return (
        epoch_tags,
        {
            str(epoch): np.asarray(timestamps, dtype=float)
            for epoch, timestamps in timestamps_position.items()
        },
        "pickle",
    )


def load_position_data(
    analysis_path: Path,
    epoch_tags: list[str],
) -> dict[str, np.ndarray]:
    """Load per-epoch position samples aligned to the saved epoch labels."""
    position_path = analysis_path / "position.pkl"
    if not position_path.exists():
        raise FileNotFoundError(f"Position file not found: {position_path}")

    with open(position_path, "rb") as f:
        position_dict = pickle.load(f)

    normalized_position_dict = {
        str(epoch): value for epoch, value in position_dict.items()
    }
    position_keys = set(normalized_position_dict.keys())
    missing_epochs = [epoch for epoch in epoch_tags if epoch not in position_keys]
    extra_epochs = sorted(position_keys - set(epoch_tags))
    if missing_epochs or extra_epochs:
        raise ValueError(
            "Position epochs do not match saved position timestamp epochs. "
            f"Missing position epochs: {missing_epochs!r}; extra position epochs: {extra_epochs!r}"
        )

    return {
        epoch: coerce_position_array(normalized_position_dict[epoch])
        for epoch in epoch_tags
    }


def coerce_position_array(position: Any) -> np.ndarray:
    """Coerce a position-like object to a `(n_time, 2)` NumPy array."""
    position_array = np.asarray(position)
    if position_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D position array, got shape {position_array.shape}."
        )
    if position_array.shape[1] >= 2:
        return position_array[:, :2]
    if position_array.shape[0] >= 2:
        return position_array[:2, :].T
    raise ValueError(
        f"Could not interpret position array of shape {position_array.shape} as XY samples."
    )


def get_position_sampling_rate(timestamps_position: np.ndarray) -> float:
    """Return the position sampling rate inferred from epoch timestamps."""
    if timestamps_position.ndim != 1 or timestamps_position.size < 2:
        raise ValueError("Position timestamps must be a 1D array with at least two samples.")
    duration = float(timestamps_position[-1] - timestamps_position[0])
    if duration <= 0:
        raise ValueError("Position timestamps must span a positive duration.")
    return (len(timestamps_position) - 1) / duration


def build_speed_tsd(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_sigma_s: float = DEFAULT_SPEED_SIGMA_S,
):
    """Compute a pynapple Tsd of speed for one epoch."""
    import pynapple as nap

    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if position.shape[0] <= position_offset or timestamps_position.size <= position_offset:
        raise ValueError(
            "Position offset removes all samples for one epoch. "
            f"position samples: {position.shape[0]}, timestamp samples: {timestamps_position.size}, "
            f"position_offset: {position_offset}"
        )
    if position.shape[0] != timestamps_position.size:
        raise ValueError(
            "Position samples and position timestamps must have the same length. "
            f"Got {position.shape[0]} samples and {timestamps_position.size} timestamps."
        )

    epoch_position = position[position_offset:]
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    sampling_rate = get_position_sampling_rate(epoch_timestamps)
    speed = np.asarray(
        pt.get_speed(
            position=epoch_position,
            time=epoch_timestamps,
            sampling_frequency=sampling_rate,
            sigma=speed_sigma_s,
        ),
        dtype=float,
    )

    if speed.shape[0] != epoch_timestamps.shape[0]:
        raise ValueError(
            "Speed computation returned a different number of samples than the "
            "trimmed position timestamps."
        )

    return nap.Tsd(t=epoch_timestamps, d=speed, time_units="s")


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
    if speed_threshold_cm_s < 0:
        raise ValueError("--speed-threshold-cm-s must be non-negative.")

    epoch_interval = get_epoch_interval(speed_tsd)
    movement_ep = speed_tsd.threshold(speed_threshold_cm_s, method="above").time_support
    movement_ep = movement_ep.intersect(epoch_interval)
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
    with open(output_path, "wb") as f:
        pickle.dump(serializable, f)
    return output_path


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
    speed_sigma_s: float = DEFAULT_SPEED_SIGMA_S,
    output_format: str = "both",
) -> None:
    """Compute and save movement and immobility intervals for one session."""
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
            speed_sigma_s=speed_sigma_s,
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
    }
    if output_format in {"pickle", "both"}:
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
    if output_format in {"pynapple", "both"}:
        outputs["run_times_pynapple_path"] = save_pynapple_interval_output(
            analysis_path=analysis_path,
            state_name="run_times",
            intervals_by_epoch=run_intervals,
        )
        outputs["immobility_times_pynapple_path"] = save_pynapple_interval_output(
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
            "speed_sigma_s": speed_sigma_s,
            "output_format": output_format,
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
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of initial position samples to discard per epoch. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=f"Speed threshold in cm/s used to define movement. Default: {DEFAULT_SPEED_THRESHOLD_CM_S}",
    )
    parser.add_argument(
        "--speed-sigma-s",
        type=float,
        default=DEFAULT_SPEED_SIGMA_S,
        help=f"Smoothing sigma passed to position_tools.get_speed. Default: {DEFAULT_SPEED_SIGMA_S}",
    )
    parser.add_argument(
        "--output-format",
        choices=["both", "pickle", "pynapple"],
        default="both",
        help="Output format for saved intervals. Default: both",
    )
    return parser.parse_args()


def main() -> None:
    """Run the movement-state export CLI."""
    args = parse_arguments()
    get_immobility_times(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        speed_sigma_s=args.speed_sigma_s,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
