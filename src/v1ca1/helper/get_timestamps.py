from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pynwb

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_NWB_ROOT


DEFAULT_DATA_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_GAP_THRESHOLD_S = 10.0


def get_analysis_path(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the analysis path for one session."""
    return data_root / animal_name / date


def extract_epoch_metadata(nwbfile: "pynwb.NWBFile") -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return ordered epoch labels and NWB epoch start/stop times."""
    if nwbfile.epochs is None:
        raise ValueError("NWB file does not contain an epochs table.")

    epochs = nwbfile.epochs[:]
    epoch_tags: list[str] = []
    for row_tags in epochs["tags"]:
        if len(row_tags) != 1:
            raise ValueError(
                "Expected exactly one tag per NWB epoch row, "
                f"found {len(row_tags)} tags: {row_tags!r}"
            )
        epoch_tags.append(str(row_tags[0]))

    start_times = np.asarray(epochs["start_time"], dtype=float)
    stop_times = np.asarray(epochs["stop_time"], dtype=float)

    if len(epoch_tags) == 0:
        raise ValueError("NWB epochs table is empty.")

    return epoch_tags, start_times, stop_times


def split_timestamps_by_gap(
    timestamps: np.ndarray,
    num_epochs: int,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Split a concatenated timestamp vector into per-epoch segments."""
    timestamps = np.asarray(timestamps, dtype=float)

    if timestamps.ndim != 1:
        raise ValueError("timestamps_ephys_all must be one-dimensional.")
    if timestamps.size == 0:
        raise ValueError("timestamps_ephys_all is empty.")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}.")
    if timestamps.size < num_epochs:
        raise ValueError(
            "timestamps_ephys_all has fewer samples than the requested number of epochs."
        )
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps_ephys_all contains non-finite values.")

    diffs = np.diff(timestamps)
    if diffs.size and not np.all(diffs > 0):
        raise ValueError("timestamps_ephys_all must be strictly increasing.")

    if gap_threshold_s <= 0:
        raise ValueError("--gap-threshold-s must be positive.")

    selected_gap_indices = np.flatnonzero(diffs > gap_threshold_s)
    selected_gaps = diffs[selected_gap_indices]
    split_indices = selected_gap_indices + 1
    threshold_mode = (
        "default" if np.isclose(gap_threshold_s, DEFAULT_GAP_THRESHOLD_S) else "manual"
    )

    segments = [np.asarray(segment, dtype=float) for segment in np.split(timestamps, split_indices)]
    if len(segments) != num_epochs:
        raise ValueError(
            "Gap segmentation produced the wrong number of epochs: "
            f"expected {num_epochs}, found {len(segments)}. "
            f"Using gap threshold {gap_threshold_s} s."
        )
    if any(segment.size == 0 for segment in segments):
        raise ValueError("Gap segmentation produced an empty epoch.")

    within_epoch_mask = np.ones(diffs.shape, dtype=bool)
    if selected_gap_indices.size:
        within_epoch_mask[selected_gap_indices] = False
    within_epoch_diffs = diffs[within_epoch_mask]
    positive_within_epoch_diffs = within_epoch_diffs[within_epoch_diffs > 0]
    if positive_within_epoch_diffs.size == 0:
        positive_within_epoch_diffs = diffs[diffs > 0]
    median_positive_dt = (
        float(np.median(positive_within_epoch_diffs))
        if positive_within_epoch_diffs.size
        else 0.0
    )

    if selected_gaps.size and median_positive_dt > 0:
        min_selected_gap = float(np.min(selected_gaps))
        if min_selected_gap < 10 * median_positive_dt:
            raise ValueError(
                "Selected boundary gaps are not sufficiently separated from the "
                "within-epoch sampling interval. Pass --gap-threshold-s to "
                "override the automatic split."
            )

    metadata: dict[str, Any] = {
        "threshold_mode": threshold_mode,
        "gap_threshold_s": float(gap_threshold_s),
        "num_epochs": int(num_epochs),
        "split_indices": split_indices.tolist(),
        "selected_gap_indices": selected_gap_indices.tolist(),
        "selected_gaps_s": selected_gaps.tolist(),
        "median_positive_dt_s": median_positive_dt,
        "validation": {
            "strictly_increasing": True,
            "segment_count_matches": True,
            "all_segments_nonempty": True,
            "min_selected_gap_ratio": (
                float(np.min(selected_gaps) / median_positive_dt)
                if selected_gaps.size and median_positive_dt > 0
                else None
            ),
        },
    }
    return segments, metadata


def validate_epoch_alignment(
    epoch_tags: list[str],
    epoch_segments: list[np.ndarray],
    epoch_start_times: np.ndarray,
    epoch_stop_times: np.ndarray,
    median_positive_dt_s: float,
) -> None:
    """Validate that split segments overlap the NWB epoch intervals."""
    if not (
        len(epoch_tags)
        == len(epoch_segments)
        == len(epoch_start_times)
        == len(epoch_stop_times)
    ):
        raise ValueError("Epoch labels, segments, and NWB epoch bounds must have matching lengths.")

    tolerance = max(10 * median_positive_dt_s, 1e-6)
    for epoch, segment, start_time, stop_time in zip(
        epoch_tags, epoch_segments, epoch_start_times, epoch_stop_times
    ):
        if segment[0] > stop_time + tolerance:
            raise ValueError(
                f"Gap-derived segment for epoch {epoch} starts after the NWB epoch stop time."
            )
        if segment[-1] < start_time - tolerance:
            raise ValueError(
                f"Gap-derived segment for epoch {epoch} ends before the NWB epoch start time."
            )


def get_timestamps_position(
    nwbfile: "pynwb.NWBFile",
    epoch_tags: list[str],
) -> dict[str, np.ndarray]:
    """Extract per-epoch position timestamps from the NWB video processing module."""
    timestamps_position: dict[str, np.ndarray] = {}
    video = nwbfile.processing["video_files"].data_interfaces["video"]
    video_files = list(video.time_series.keys())

    if len(video_files) != len(epoch_tags):
        raise ValueError(
            "Video epoch count does not match the NWB epoch count: "
            f"{len(video_files)} video files vs {len(epoch_tags)} epochs."
        )

    for epoch, video_file in zip(epoch_tags, video_files):
        print(f"processing epoch {epoch}")
        timestamps_position[epoch] = np.asarray(
            video.time_series[video_file].timestamps[:], dtype=float
        )

    return timestamps_position


def save_pynapple_outputs(
    analysis_path: Path,
    timestamps_ephys_all: np.ndarray,
    timestamps_position: dict[str, np.ndarray],
    epoch_tags: list[str],
    epoch_segments: list[np.ndarray],
) -> None:
    """Write pynapple-backed timestamp artifacts for the migration path."""
    import pynapple as nap

    timestamps_all = nap.Ts(t=timestamps_ephys_all, time_units="s")
    timestamps_all.save(analysis_path / "timestamps_ephys_all.npz")

    epoch_intervals = nap.IntervalSet(
        start=np.asarray([segment[0] for segment in epoch_segments], dtype=float),
        end=np.asarray([segment[-1] for segment in epoch_segments], dtype=float),
    )
    epoch_intervals.set_info(epoch=epoch_tags)
    epoch_intervals.save(analysis_path / "timestamps_ephys.npz")

    position_group = nap.TsGroup(
        {
            index: nap.Ts(t=timestamps_position[epoch], time_units="s")
            for index, epoch in enumerate(epoch_tags)
        },
        time_units="s",
    )
    position_group.set_info(epoch=epoch_tags)
    position_group.save(analysis_path / "timestamps_position.npz")


def ensure_analysis_path(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the session analysis path, creating it when missing."""
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    analysis_path.mkdir(parents=True, exist_ok=True)
    return analysis_path


def get_timestamps(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    gap_threshold_s: float = DEFAULT_GAP_THRESHOLD_S,
) -> None:
    """Save timestamps for one session."""
    import pynwb

    analysis_path = ensure_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"

    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    print(f"Processing {animal_name} {date}.")
    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        epoch_tags, epoch_start_times, epoch_stop_times = extract_epoch_metadata(nwbfile)
        timestamps_ephys_all = np.asarray(
            nwbfile.acquisition["e-series"].timestamps[:], dtype=float
        )
        timestamps_position = get_timestamps_position(nwbfile, epoch_tags)

    epoch_segments, split_metadata = split_timestamps_by_gap(
        timestamps=timestamps_ephys_all,
        num_epochs=len(epoch_tags),
        gap_threshold_s=gap_threshold_s,
    )
    validate_epoch_alignment(
        epoch_tags=epoch_tags,
        epoch_segments=epoch_segments,
        epoch_start_times=epoch_start_times,
        epoch_stop_times=epoch_stop_times,
        median_positive_dt_s=split_metadata["median_positive_dt_s"],
    )

    save_pynapple_outputs(
        analysis_path=analysis_path,
        timestamps_ephys_all=timestamps_ephys_all,
        timestamps_position=timestamps_position,
        epoch_tags=epoch_tags,
        epoch_segments=epoch_segments,
    )

    outputs: dict[str, Any] = {
        "timestamps_ephys_all_pynapple_path": analysis_path / "timestamps_ephys_all.npz",
        "timestamps_ephys_pynapple_path": analysis_path / "timestamps_ephys.npz",
        "timestamps_position_pynapple_path": analysis_path / "timestamps_position.npz",
        "ephys_segmentation": {
            "epoch_tags": epoch_tags,
            "epoch_start_times_s": np.asarray(epoch_start_times, dtype=float).tolist(),
            "epoch_stop_times_s": np.asarray(epoch_stop_times, dtype=float).tolist(),
            "epoch_segment_start_times_s": [
                float(segment[0]) for segment in epoch_segments
            ],
            "epoch_segment_stop_times_s": [
                float(segment[-1]) for segment in epoch_segments
            ],
            **split_metadata,
        },
    }

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.helper.get_timestamps",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "gap_threshold_s": gap_threshold_s,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Save timestamps")
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
        "--nwb-root",
        type=Path,
        default=DEFAULT_NWB_ROOT,
        help=f"Base directory containing NWB files. Default: {DEFAULT_NWB_ROOT}",
    )
    parser.add_argument(
        "--gap-threshold-s",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_S,
        help=(
            "Threshold in seconds for detecting inter-epoch gaps. "
            f"Default: {DEFAULT_GAP_THRESHOLD_S}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the timestamp export CLI."""
    args = parse_arguments()
    get_timestamps(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        gap_threshold_s=args.gap_threshold_s,
    )


if __name__ == "__main__":
    main()
