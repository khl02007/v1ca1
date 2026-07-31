from __future__ import annotations

"""Export an augmented copy of one NWB file from saved analysis outputs.

This workflow preserves the source epochs table, adds saved ephys recording
intervals, and can also add poke-defined trajectory intervals, speed-gated
ripples, canonical head/body position, and curated spike sorting results.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    DEFAULT_POSITION_OFFSET,
    TRAJECTORY_TYPES,
    get_analysis_path,
    load_ephys_timestamps_all,
)
from v1ca1.helper.wtrack import (
    DEFAULT_WTRACK_BRANCH_GAP_CM,
    get_wtrack_branch_graph_inputs,
    get_wtrack_branch_side,
    get_wtrack_direction,
    get_wtrack_full_graph_inputs,
)
from v1ca1.spikesorting.consolidate_sorting import DEFAULT_CURATION_ROOT

if TYPE_CHECKING:
    import pandas as pd
    import pynapple as nap
    import pynwb


DEFAULT_OUTPUT_SUFFIX = "_augmented.nwb"
EPHYS_COMPRESSION = "gzip"
EPHYS_COMPRESSION_LEVEL = 4
EPHYS_COMPRESSION_SHUFFLE = True
EPHYS_INTERVALS_TABLE_NAME = "ephys_recording_intervals"
TRAJECTORY_INTERVALS_TABLE_NAME = "trajectory_times"
TRAJECTORY_TIMES_FILENAME = "trajectory_times.parquet"
RIPPLES_INTERVALS_TABLE_NAME = "ripples"
RIPPLE_TIMES_RELATIVE_PATH = Path("ripple") / "ripple_times.parquet"
RIPPLE_PROVENANCE_SCRATCH_NAME = "ripple_detection_provenance"
RIPPLE_RUN_LOG_GLOB = "v1ca1_ripple_detect_ripples_*.json"
RIPPLE_PROVENANCE_SCHEMA_VERSION = 1
POSITION_RELATIVE_PATH = Path("dlc_position_cleaned") / "position.parquet"
POSITION_INTERFACE_NAME = "position"
HEAD_POSITION_SERIES_NAME = "head_position"
BODY_POSITION_SERIES_NAME = "body_position"
POSITION_SAMPLE_METADATA_NAME = "position_sample_metadata"
POSITION_EPOCHS_TABLE_NAME = "position_epochs"
POSITION_PROVENANCE_SCRATCH_NAME = "position_provenance"
POSITION_PROVENANCE_SCHEMA_VERSION = 1
WTRACK_LINEARIZATION_TABLE_NAME = "wtrack_linearization"
WTRACK_FULL_GRAPH_CONFIGURATION_NAME = "full_w"
POSITION_SPATIAL_UNIT = "centimeters"
POSITION_PRODUCER_SCRIPTS = (
    "v1ca1.position.combine_clean_dlc_position",
    "v1ca1.position.convert_legacy_position_pickles",
)
POSITION_PRODUCER_LOG_GLOBS = (
    "v1ca1_position_combine_clean_dlc_position_*.json",
    "v1ca1_position_convert_legacy_position_pickles_*.json",
)
POSITION_COLUMNS = (
    "epoch",
    "frame",
    "frame_time_s",
    "head_x_cm",
    "head_y_cm",
    "body_x_cm",
    "body_y_cm",
)
REGIONS = ("v1", "ca1")
REQUIRED_SORTING_PROPERTIES = (
    "region",
    "probe_idx",
    "shank_idx",
    "curation_json_relpath",
)
SORTING_PROVENANCE_SCRATCH_NAME = "spike_sorting_curation_provenance"
UNITS_TABLE_DESCRIPTION = (
    "Curated spike sorting units exported from consolidated SpikeInterface sortings."
)
EPHYS_INTERVALS_DESCRIPTION = (
    "Contiguous spans of recorded ephys timestamps imported from "
    "timestamps_ephys.npz, produced by v1ca1.helper.get_timestamps. Each row "
    "corresponds to one source NWB epoch tag, with start_time and stop_time "
    "equal to the first and last saved sample timestamps. The source epochs "
    "table is preserved unchanged."
)
TRAJECTORY_INTERVALS_DESCRIPTION = (
    "Poke-to-poke run trajectory intervals imported from "
    "trajectory_times.parquet, produced by v1ca1.helper.get_trajectory_times. "
    "Each stop_time is the destination poke time."
)
RIPPLES_INTERVALS_DESCRIPTION = (
    "Speed-gated ripples imported from ripple/ripple_times.parquet, produced "
    "by v1ca1.ripple.detect_ripples. Detection metadata is stored in scratch "
    f"{RIPPLE_PROVENANCE_SCRATCH_NAME!r}."
)
RIPPLE_PROVENANCE_DESCRIPTION = (
    "JSON provenance for the speed-gated ripples stored in /intervals/ripples."
)
POSITION_REFERENCE_FRAME = (
    "Two-dimensional camera coordinate frame represented by the canonical "
    "position artifact, expressed in centimeters. NWB export preserves the "
    "source origin and axis directions and applies no translation, rotation, "
    "axis inversion, or track linearization."
)
POSITION_SERIES_DESCRIPTION = (
    "Canonical two-dimensional {bodypart} position imported from "
    "dlc_position_cleaned/position.parquet. Sample i corresponds to row i of "
    "/processing/behavior/position_sample_metadata."
)
POSITION_SAMPLE_METADATA_DESCRIPTION = (
    "Row-aligned metadata for /processing/behavior/position/head_position and "
    "/processing/behavior/position/body_position. Table row id i identifies "
    "sample i in both SpatialSeries."
)
POSITION_EPOCHS_DESCRIPTION = (
    "Explicit epoch membership and half-open sample-index ranges for the "
    "exported head and body position SpatialSeries. For each row, samples in "
    "[start_index, stop_index_exclusive) belong to the named epoch in both "
    "SpatialSeries. start_time and stop_time are the first and last video "
    "timestamps in that range. analysis_start_offset_samples records how many "
    "leading samples analyses should ignore without removing stored position "
    "samples or changing their timestamps."
)
POSITION_PROVENANCE_DESCRIPTION = (
    "JSON provenance for the canonical head/body position stored under "
    "/processing/behavior."
)
WTRACK_LINEARIZATION_DESCRIPTION = (
    "Explicit effective graph inputs used for W-track position linearization. "
    "Each row is one independently reconstructable graph configuration. Node "
    "coordinates are in centimeters and use the same camera coordinate frame "
    "as /processing/behavior/position/head_position and body_position. This "
    "table contains configuration data, not time-varying observations."
)
_EPHYS_INTERVAL_COLUMN_DESCRIPTIONS = {
    "epoch": "Single tag identifying the corresponding source NWB epoch row.",
}
_TRAJECTORY_INTERVAL_COLUMN_DESCRIPTIONS = {
    "epoch": "Run epoch containing this trajectory.",
    "trajectory_type": (
        "Poke-defined trajectory direction: left_to_center, center_to_left, "
        "right_to_center, or center_to_right."
    ),
}
_RIPPLE_INTERVAL_COLUMN_DESCRIPTIONS = {
    "epoch": "Ephys epoch containing this ripple.",
    "duration": "Detector-reported ripple duration in seconds.",
    "max_thresh": "Detector-reported maximum sustained normalized threshold.",
    "mean_zscore": "Mean normalized detector signal during the ripple.",
    "median_zscore": "Median normalized detector signal during the ripple.",
    "max_zscore": "Maximum normalized detector signal during the ripple.",
    "min_zscore": "Minimum normalized detector signal during the ripple.",
    "area": "Detector-reported integral of the normalized signal.",
    "total_energy": "Detector-reported integral of the squared normalized signal.",
    "speed_at_start": "Animal speed at ripple start, in centimeters per second.",
    "speed_at_end": "Animal speed at ripple stop, in centimeters per second.",
    "max_speed": "Maximum animal speed during the ripple, in centimeters per second.",
    "min_speed": "Minimum animal speed during the ripple, in centimeters per second.",
    "median_speed": "Median animal speed during the ripple, in centimeters per second.",
    "mean_speed": "Mean animal speed during the ripple, in centimeters per second.",
}
_RIPPLE_NUMERIC_COLUMNS = tuple(
    column_name
    for column_name in _RIPPLE_INTERVAL_COLUMN_DESCRIPTIONS
    if column_name != "epoch"
)
_POSITION_EPOCH_COLUMN_DESCRIPTIONS = {
    "epoch": "NWB epoch tag for this contiguous position sample range.",
    "start_index": (
        "Zero-based index of the first sample for this epoch in both position "
        "SpatialSeries."
    ),
    "stop_index_exclusive": (
        "Exclusive zero-based stop index for this epoch in both position "
        "SpatialSeries."
    ),
    "sample_count": "Number of position samples in this epoch.",
    "analysis_start_offset_samples": (
        "Number of leading position samples to ignore within this epoch. This "
        "is a zero-based offset: 10 means the first included sample has local "
        "index 10 and local indices 0 through 9 remain stored but are excluded "
        "by analyses. Position data and timestamps are not truncated."
    ),
    "first_frame": "Original frame number for the first exported sample.",
    "last_frame": "Original frame number for the last exported sample.",
    "video_series_name": (
        "Name of the source NWB video ImageSeries whose timestamps were used."
    ),
}
_WTRACK_LINEARIZATION_COLUMN_DESCRIPTIONS = {
    "configuration_name": (
        "Unique configuration name: one of the four trajectory directions or "
        f"{WTRACK_FULL_GRAPH_CONFIGURATION_NAME!r}."
    ),
    "node_positions_cm": (
        "Node coordinates with shape (n_nodes, 2), in centimeters and in the "
        "same coordinate frame as the exported head/body position. Node id is "
        "the zero-based row index within this array."
    ),
    "edges": (
        "Graph edges with shape (n_edges, 2); values are zero-based node ids."
    ),
    "edge_order": (
        "Directed edge traversal order passed to get_linearized_position, with "
        "shape (n_edges, 2)."
    ),
    "edge_spacing_cm": (
        "Explicit spacing after each ordered edge except the last, in "
        "centimeters. Branch configurations contain zeros; the full-W "
        "configuration includes the configured inter-branch gap."
    ),
    "use_hmm": (
        "Value of the track_linearization get_linearized_position use_HMM "
        "argument used by current analyses."
    ),
}
_UNITS_COLUMN_DESCRIPTIONS = {
    "region": "Brain region for the curated unit.",
    "probe_idx": "Probe index that produced the curated unit.",
    "shank_idx": "Shank index that produced the curated unit.",
    "sorting_unit_id": "Unit id from the consolidated SpikeInterface sorting.",
    "curation_json_relpath": "Path to the raw sortingview curation JSON relative to the curation root.",
    "is_merged": "Whether the curated unit resulted from a manual merge.",
}
_SPIKEINTERFACE = None


def get_spikeinterface():
    """Import SpikeInterface lazily."""
    global _SPIKEINTERFACE
    if _SPIKEINTERFACE is None:
        import spikeinterface.full as si

        _SPIKEINTERFACE = si
    return _SPIKEINTERFACE


def _extract_interval_dataframe(intervals: "nap.IntervalSet") -> "pd.DataFrame":
    """Return a dataframe-like view of one pynapple IntervalSet."""
    if hasattr(intervals, "as_dataframe"):
        return intervals.as_dataframe()
    if hasattr(intervals, "_metadata"):
        return intervals._metadata.copy()  # type: ignore[attr-defined]
    raise ValueError("Could not read metadata from timestamps_ephys.npz.")


def _extract_epoch_tags(intervals: "nap.IntervalSet") -> list[str]:
    """Extract saved epoch labels from timestamps_ephys.npz."""
    try:
        epoch_info = intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is not None:
        epoch_array = np.asarray(epoch_info)
        if epoch_array.size:
            return [str(epoch) for epoch in epoch_array.tolist()]

    interval_df = _extract_interval_dataframe(intervals)
    if "epoch" in interval_df.columns:
        return [str(epoch) for epoch in interval_df["epoch"].tolist()]

    raise ValueError("timestamps_ephys.npz does not contain saved epoch labels.")


def _extract_interval_bounds(intervals: "nap.IntervalSet") -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned start and end arrays from timestamps_ephys.npz."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "timestamps_ephys.npz has mismatched start/end arrays: "
            f"{starts.shape} vs {ends.shape}."
        )
    return starts, ends


def load_epoch_bounds_npz(analysis_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load saved epoch labels and bounds from timestamps_ephys.npz."""
    import pynapple as nap

    npz_path = analysis_path / "timestamps_ephys.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"timestamps_ephys.npz not found: {npz_path}")

    try:
        intervals = nap.load_file(npz_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {npz_path}.") from exc

    epoch_tags = _extract_epoch_tags(intervals)
    start_times, stop_times = _extract_interval_bounds(intervals)
    if len(epoch_tags) != start_times.size:
        raise ValueError(
            "Mismatch between saved epoch labels and interval bounds in "
            f"{npz_path}."
        )
    if start_times.size == 0:
        raise ValueError(f"timestamps_ephys.npz does not contain any epochs: {npz_path}")
    if not (np.all(np.isfinite(start_times)) and np.all(np.isfinite(stop_times))):
        raise ValueError(f"timestamps_ephys.npz contains non-finite interval bounds: {npz_path}")
    if np.any(stop_times < start_times):
        raise ValueError(f"timestamps_ephys.npz contains an interval with stop < start: {npz_path}")

    return epoch_tags, start_times, stop_times


def resolve_output_path(
    nwb_path: Path,
    animal_name: str,
    date: str,
    output_path: Path | None,
) -> Path:
    """Return the requested output path, defaulting to a sibling NWB copy."""
    if output_path is not None:
        return output_path
    return nwb_path.with_name(f"{animal_name}{date}{DEFAULT_OUTPUT_SUFFIX}")


def validate_output_path(source_path: Path, output_path: Path, overwrite: bool) -> None:
    """Validate the destination path for the augmented NWB file."""
    if output_path.resolve() == source_path.resolve():
        raise ValueError("Output path must differ from the source NWB path.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_path}. Pass --overwrite to replace it."
        )


def validate_epoch_tags(
    nwb_epoch_tags: list[str],
    saved_epoch_tags: list[str],
) -> None:
    """Require exact epoch count and order agreement between NWB and npz."""
    if nwb_epoch_tags != saved_epoch_tags:
        raise ValueError(
            "Epoch labels from timestamps_ephys.npz do not match the NWB epochs table. "
            f"NWB: {nwb_epoch_tags!r}; timestamps_ephys.npz: {saved_epoch_tags!r}"
        )


def load_trajectory_times_parquet(analysis_path: Path) -> tuple["pd.DataFrame", Path]:
    """Load and validate the canonical poke-defined trajectory table."""
    import pandas as pd

    parquet_path = analysis_path / TRAJECTORY_TIMES_FILENAME
    if not parquet_path.exists():
        raise FileNotFoundError(f"{TRAJECTORY_TIMES_FILENAME} not found: {parquet_path}")

    try:
        trajectory_table = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {parquet_path}.") from exc

    required_columns = {"start", "end", "epoch", "trajectory_type"}
    missing_columns = required_columns.difference(trajectory_table.columns)
    if missing_columns:
        raise ValueError(
            f"{parquet_path} is missing required columns: {sorted(missing_columns)!r}."
        )

    trajectory_table = trajectory_table.loc[
        :,
        ["start", "end", "epoch", "trajectory_type"],
    ].copy()
    if trajectory_table.loc[:, ["epoch", "trajectory_type"]].isna().any().any():
        raise ValueError(f"{parquet_path} contains a missing epoch or trajectory type.")

    try:
        trajectory_table["start"] = trajectory_table["start"].astype(float)
        trajectory_table["end"] = trajectory_table["end"].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parquet_path} contains non-numeric interval bounds.") from exc
    trajectory_table["epoch"] = trajectory_table["epoch"].astype(str)
    trajectory_table["trajectory_type"] = trajectory_table["trajectory_type"].astype(str)

    starts = trajectory_table["start"].to_numpy(dtype=float)
    stops = trajectory_table["end"].to_numpy(dtype=float)
    if not (np.all(np.isfinite(starts)) and np.all(np.isfinite(stops))):
        raise ValueError(f"{parquet_path} contains non-finite interval bounds.")
    if np.any(stops < starts):
        raise ValueError(f"{parquet_path} contains an interval with end < start.")

    invalid_trajectory_types = sorted(
        set(trajectory_table["trajectory_type"]).difference(TRAJECTORY_TYPES)
    )
    if invalid_trajectory_types:
        raise ValueError(
            f"{parquet_path} contains unsupported trajectory types: "
            f"{invalid_trajectory_types!r}."
        )
    return trajectory_table, parquet_path


def load_position_parquet(analysis_path: Path) -> tuple["pd.DataFrame", Path]:
    """Load and validate the canonical combined head/body position table."""
    import pandas as pd

    parquet_path = analysis_path / POSITION_RELATIVE_PATH
    if not parquet_path.exists():
        raise FileNotFoundError(f"Combined position parquet not found: {parquet_path}")

    try:
        position_table = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {parquet_path}.") from exc

    if not position_table.columns.is_unique:
        raise ValueError(f"{parquet_path} contains duplicate column names.")
    missing_columns = [
        column_name
        for column_name in POSITION_COLUMNS
        if column_name not in position_table.columns
    ]
    unexpected_columns = [
        str(column_name)
        for column_name in position_table.columns
        if column_name not in POSITION_COLUMNS
    ]
    if missing_columns:
        raise ValueError(
            f"{parquet_path} is missing required columns: {missing_columns!r}."
        )
    if unexpected_columns:
        raise ValueError(
            f"{parquet_path} contains unexpected columns: {unexpected_columns!r}."
        )
    if position_table.empty:
        raise ValueError(f"Combined position parquet is empty: {parquet_path}")

    position_table = position_table.loc[:, list(POSITION_COLUMNS)].copy().reset_index(
        drop=True
    )
    if position_table["epoch"].isna().any():
        raise ValueError(f"{parquet_path} contains a missing epoch label.")
    position_table["epoch"] = position_table["epoch"].astype(str)
    if position_table["epoch"].str.strip().eq("").any():
        raise ValueError(f"{parquet_path} contains an empty epoch label.")

    try:
        frame_values_float = pd.to_numeric(
            position_table["frame"],
            errors="raise",
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parquet_path} contains non-numeric frame values.") from exc
    if not np.all(np.isfinite(frame_values_float)):
        raise ValueError(f"{parquet_path} contains non-finite frame values.")
    if not np.all(frame_values_float == np.floor(frame_values_float)):
        raise ValueError(f"{parquet_path} contains non-integer frame values.")
    int64_info = np.iinfo(np.int64)
    if np.any(frame_values_float < int64_info.min) or np.any(
        frame_values_float > int64_info.max
    ):
        raise ValueError(f"{parquet_path} contains frame values outside int64 range.")
    position_table["frame"] = frame_values_float.astype(np.int64)

    numeric_columns = [
        "frame_time_s",
        "head_x_cm",
        "head_y_cm",
        "body_x_cm",
        "body_y_cm",
    ]
    for column_name in numeric_columns:
        try:
            position_table[column_name] = pd.to_numeric(
                position_table[column_name],
                errors="raise",
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{parquet_path} contains non-numeric values in {column_name!r}."
            ) from exc

    frame_times = position_table["frame_time_s"].to_numpy(dtype=float)
    if not np.all(np.isfinite(frame_times)):
        raise ValueError(f"{parquet_path} contains non-finite frame timestamps.")
    coordinate_values = position_table.loc[
        :,
        ["head_x_cm", "head_y_cm", "body_x_cm", "body_y_cm"],
    ].to_numpy(dtype=float)
    if np.any(np.isinf(coordinate_values)):
        raise ValueError(f"{parquet_path} contains infinite position coordinates.")

    epoch_values = position_table["epoch"].tolist()
    epoch_blocks = [
        epoch
        for index, epoch in enumerate(epoch_values)
        if index == 0 or epoch != epoch_values[index - 1]
    ]
    if len(epoch_blocks) != len(set(epoch_blocks)):
        raise ValueError(
            f"{parquet_path} contains a repeated non-contiguous epoch block."
        )

    for epoch, epoch_table in position_table.groupby("epoch", sort=False):
        frame_values = epoch_table["frame"].to_numpy(dtype=np.int64)
        if np.unique(frame_values).size != frame_values.size:
            raise ValueError(
                f"{parquet_path} contains duplicate frames for epoch {epoch!r}."
            )
        if frame_values.size > 1 and np.any(np.diff(frame_values) <= 0):
            raise ValueError(
                f"{parquet_path} frames are not strictly increasing for epoch {epoch!r}."
            )
        epoch_times = epoch_table["frame_time_s"].to_numpy(dtype=float)
        if epoch_times.size > 1 and np.any(np.diff(epoch_times) <= 0):
            raise ValueError(
                f"{parquet_path} frame timestamps are not strictly increasing "
                f"for epoch {epoch!r}."
            )

    if frame_times.size > 1 and np.any(np.diff(frame_times) <= 0):
        raise ValueError(
            f"{parquet_path} frame timestamps are not strictly increasing "
            "in source row order."
        )
    return position_table, parquet_path


def get_nwb_video_timestamps(
    nwbfile: "pynwb.NWBFile",
    epoch_tags: list[str],
) -> dict[str, tuple[str, np.ndarray]]:
    """Return source video-series names and timestamps keyed by NWB epoch tag."""
    if "video_files" not in nwbfile.processing:
        raise ValueError(
            "NWB file does not contain the video_files processing module "
            "required for position export."
        )
    video_module = nwbfile.processing["video_files"]
    if "video" not in video_module.data_interfaces:
        raise ValueError(
            "NWB video_files module does not contain the 'video' data interface "
            "required for position export."
        )
    video_interface = video_module.data_interfaces["video"]
    time_series = getattr(video_interface, "time_series", None)
    if time_series is None:
        raise ValueError(
            "NWB video data interface does not provide per-epoch time series."
        )

    video_series_names = list(time_series.keys())
    if len(video_series_names) != len(epoch_tags):
        raise ValueError(
            "Video epoch count does not match the NWB epoch count: "
            f"{len(video_series_names)} video series vs {len(epoch_tags)} epochs."
        )

    series_names_by_epoch: dict[str, str] = {}
    for epoch in epoch_tags:
        epoch_pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(epoch)}(?![A-Za-z0-9])"
        )
        matching_series_names = [
            str(series_name)
            for series_name in video_series_names
            if epoch_pattern.search(str(series_name))
        ]
        if len(matching_series_names) == 1:
            series_names_by_epoch[str(epoch)] = matching_series_names[0]

    use_named_mapping = (
        len(series_names_by_epoch) == len(epoch_tags)
        and len(set(series_names_by_epoch.values())) == len(epoch_tags)
    )
    if use_named_mapping:
        ordered_series_names = [
            series_names_by_epoch[str(epoch)]
            for epoch in epoch_tags
        ]
    else:
        ordered_series_names = [
            str(series_name)
            for series_name in video_series_names
        ]

    video_timestamps_by_epoch: dict[str, tuple[str, np.ndarray]] = {}
    for epoch, series_name in zip(
        epoch_tags,
        ordered_series_names,
        strict=True,
    ):
        series = time_series[series_name]
        if series.timestamps is None:
            raise ValueError(
                f"NWB video series {series_name!r} for epoch {epoch!r} "
                "does not contain explicit timestamps."
            )
        timestamps = np.asarray(series.timestamps[:], dtype=float).reshape(-1)
        if timestamps.size == 0:
            raise ValueError(
                f"NWB video series {series_name!r} for epoch {epoch!r} is empty."
            )
        if not np.all(np.isfinite(timestamps)):
            raise ValueError(
                f"NWB video series {series_name!r} for epoch {epoch!r} "
                "contains non-finite timestamps."
            )
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(
                f"NWB video series {series_name!r} for epoch {epoch!r} "
                "timestamps are not strictly increasing."
            )
        video_timestamps_by_epoch[str(epoch)] = (
            str(series_name),
            timestamps,
        )
    return video_timestamps_by_epoch


def validate_position_against_nwb_videos(
    position_table: "pd.DataFrame",
    epoch_tags: list[str],
    video_timestamps_by_epoch: dict[str, tuple[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Validate position rows against source video timestamps and return epoch ranges."""
    position_epoch_order = list(dict.fromkeys(position_table["epoch"].tolist()))
    unknown_epochs = sorted(set(position_epoch_order).difference(epoch_tags))
    if unknown_epochs:
        raise ValueError(
            f"{POSITION_RELATIVE_PATH} contains epochs absent from the NWB "
            f"epochs table: {unknown_epochs!r}."
        )
    expected_epoch_order = [
        epoch
        for epoch in epoch_tags
        if epoch in set(position_epoch_order)
    ]
    if position_epoch_order != expected_epoch_order:
        raise ValueError(
            f"{POSITION_RELATIVE_PATH} epoch order does not match NWB epoch order. "
            f"Position: {position_epoch_order!r}; expected: {expected_epoch_order!r}."
        )

    epoch_ranges: list[dict[str, Any]] = []
    start_index = 0
    for epoch in position_epoch_order:
        epoch_table = position_table.loc[
            position_table["epoch"] == epoch,
            :,
        ]
        series_name, video_timestamps = video_timestamps_by_epoch[epoch]
        frame_times = epoch_table["frame_time_s"].to_numpy(dtype=float)
        if frame_times.size != video_timestamps.size:
            raise ValueError(
                f"{POSITION_RELATIVE_PATH} row count does not match NWB video "
                f"series {series_name!r} for epoch {epoch!r}: "
                f"{frame_times.size} vs {video_timestamps.size}."
            )
        timestamp_errors = np.abs(frame_times - video_timestamps)
        if not np.allclose(
            frame_times,
            video_timestamps,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                f"{POSITION_RELATIVE_PATH} timestamps do not match NWB video "
                f"series {series_name!r} for epoch {epoch!r} within 1 ns."
            )

        sample_count = int(len(epoch_table))
        stop_index_exclusive = start_index + sample_count
        frame_values = epoch_table["frame"].to_numpy(dtype=np.int64)
        epoch_ranges.append(
            {
                "epoch": str(epoch),
                "start_time": float(frame_times[0]),
                "stop_time": float(frame_times[-1]),
                "start_index": int(start_index),
                "stop_index_exclusive": int(stop_index_exclusive),
                "sample_count": sample_count,
                "first_frame": int(frame_values[0]),
                "last_frame": int(frame_values[-1]),
                "video_series_name": str(series_name),
                "max_abs_timestamp_error_s": float(
                    np.max(timestamp_errors, initial=0.0)
                ),
            }
        )
        start_index = stop_index_exclusive

    if start_index != len(position_table):
        raise RuntimeError("Position epoch ranges do not cover every source row.")
    return epoch_ranges


def validate_position_offset(position_offset: int) -> int:
    """Return one validated per-epoch position analysis offset."""
    if isinstance(position_offset, bool) or not isinstance(
        position_offset,
        (int, np.integer),
    ):
        raise TypeError("position_offset must be an integer.")
    if position_offset < 0:
        raise ValueError("position_offset must be non-negative.")
    return int(position_offset)


def load_ripple_times_parquet(analysis_path: Path) -> tuple["pd.DataFrame", Path]:
    """Load and validate the canonical speed-gated ripple table."""
    import pandas as pd

    parquet_path = analysis_path / RIPPLE_TIMES_RELATIVE_PATH
    if not parquet_path.exists():
        raise FileNotFoundError(f"ripple_times.parquet not found: {parquet_path}")

    try:
        ripple_table = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(f"Failed to load {parquet_path}.") from exc

    if not ripple_table.columns.is_unique:
        raise ValueError(f"{parquet_path} contains duplicate column names.")

    required_columns = {"start", "end", "epoch"}
    missing_columns = required_columns.difference(ripple_table.columns)
    if missing_columns:
        raise ValueError(
            f"{parquet_path} is missing required columns: {sorted(missing_columns)!r}."
        )

    reserved_columns = {
        "id",
        "start_time",
        "stop_time",
        "tags",
        "timeseries",
    }.intersection(ripple_table.columns)
    if reserved_columns:
        raise ValueError(
            f"{parquet_path} contains columns reserved by NWB TimeIntervals: "
            f"{sorted(reserved_columns)!r}."
        )

    missing_metric_columns = [
        column_name
        for column_name in _RIPPLE_NUMERIC_COLUMNS
        if column_name not in ripple_table.columns
    ]
    if missing_metric_columns and not ripple_table.empty:
        raise ValueError(
            f"{parquet_path} is missing detector metric columns: "
            f"{missing_metric_columns!r}."
        )

    ripple_table = ripple_table.copy().reset_index(drop=True)
    for column_name in missing_metric_columns:
        ripple_table[column_name] = pd.Series(dtype=float)

    if ripple_table["epoch"].isna().any():
        raise ValueError(f"{parquet_path} contains a missing epoch.")
    ripple_table["epoch"] = ripple_table["epoch"].astype(str)
    if ripple_table["epoch"].str.strip().eq("").any():
        raise ValueError(f"{parquet_path} contains an empty epoch label.")

    numeric_columns = [
        "start",
        "end",
        *[
            column_name
            for column_name in _RIPPLE_NUMERIC_COLUMNS
            if column_name in ripple_table.columns
        ],
    ]
    for column_name in numeric_columns:
        try:
            ripple_table[column_name] = pd.to_numeric(
                ripple_table[column_name],
                errors="raise",
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{parquet_path} contains non-numeric values in {column_name!r}."
            ) from exc
        values = ripple_table[column_name].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{parquet_path} contains non-finite values in {column_name!r}."
            )

    extra_columns = [
        column_name
        for column_name in ripple_table.columns
        if column_name not in {"start", "end", "epoch", *_RIPPLE_NUMERIC_COLUMNS}
    ]
    for column_name in extra_columns:
        values = ripple_table[column_name]
        if values.isna().any():
            raise ValueError(
                f"{parquet_path} contains missing values in extra column "
                f"{column_name!r}."
            )
        if values.empty:
            continue
        if pd.api.types.is_bool_dtype(values.dtype):
            continue
        if pd.api.types.is_numeric_dtype(values.dtype):
            if pd.api.types.is_complex_dtype(values.dtype):
                raise ValueError(
                    f"{parquet_path} contains unsupported complex values in "
                    f"extra column {column_name!r}."
                )
            if not np.all(np.isfinite(values.to_numpy(dtype=float))):
                raise ValueError(
                    f"{parquet_path} contains non-finite values in extra column "
                    f"{column_name!r}."
                )
            continue
        if not all(isinstance(value, str) for value in values.tolist()):
            raise ValueError(
                f"{parquet_path} contains unsupported values in extra column "
                f"{column_name!r}; only scalar numeric, boolean, and string "
                "values can be exported."
            )

    starts = ripple_table["start"].to_numpy(dtype=float)
    stops = ripple_table["end"].to_numpy(dtype=float)
    if np.any(stops <= starts):
        raise ValueError(f"{parquet_path} contains a ripple with end <= start.")

    if "duration" in ripple_table.columns:
        durations = ripple_table["duration"].to_numpy(dtype=float)
        if not np.allclose(
            durations,
            stops - starts,
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(
                f"{parquet_path} contains duration values inconsistent with end - start."
            )

    expected_order = (
        ripple_table.sort_values(
            by=["start", "end", "epoch"],
            kind="mergesort",
        )
        .index.to_numpy()
    )
    if not np.array_equal(expected_order, np.arange(len(ripple_table))):
        raise ValueError(
            f"{parquet_path} is not in canonical start/end/epoch order."
        )

    ordered_columns = [
        "start",
        "end",
        "epoch",
        *_RIPPLE_NUMERIC_COLUMNS,
    ]
    ordered_columns.extend(
        column_name
        for column_name in ripple_table.columns
        if column_name not in ordered_columns
    )
    return ripple_table.loc[:, ordered_columns], parquet_path


def _get_ripple_event_counts(ripple_table: "pd.DataFrame") -> dict[str, int]:
    """Return ripple counts keyed by epoch, preserving source epoch order."""
    return {
        str(epoch): int(count)
        for epoch, count in ripple_table.groupby("epoch", sort=False).size().items()
    }


def _get_logged_selected_epochs(record: dict[str, Any]) -> list[str] | None:
    """Return selected detector epochs from one run log when available."""
    outputs = record.get("outputs")
    parameters = record.get("parameters")
    if not isinstance(outputs, dict) or not isinstance(parameters, dict):
        return None

    output_epochs = outputs.get("selected_epochs")
    parameter_epochs = parameters.get("epochs")
    selected_epochs = output_epochs if isinstance(output_epochs, list) else parameter_epochs
    if not isinstance(selected_epochs, list):
        return None

    normalized_epochs = [str(epoch) for epoch in selected_epochs]
    if isinstance(output_epochs, list) and isinstance(parameter_epochs, list):
        if normalized_epochs != [str(epoch) for epoch in parameter_epochs]:
            return None
    return normalized_epochs


def _ripple_run_log_matches_table(
    record: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    ripple_table: "pd.DataFrame",
) -> bool:
    """Return whether one detector run log describes the complete ripple table."""
    if record.get("script") != "v1ca1.ripple.detect_ripples":
        return False

    parameters = record.get("parameters")
    outputs = record.get("outputs")
    if not isinstance(parameters, dict) or not isinstance(outputs, dict):
        return False
    if str(parameters.get("animal_name")) != str(animal_name):
        return False
    if str(parameters.get("date")) != str(date):
        return False
    if parameters.get("use_speed_gating") is not True:
        return False

    saved_interval_path = outputs.get("saved_interval_parquet")
    if not isinstance(saved_interval_path, str):
        return False
    saved_interval_path = Path(saved_interval_path)
    if (
        saved_interval_path.name != RIPPLE_TIMES_RELATIVE_PATH.name
        or saved_interval_path.parent.name != RIPPLE_TIMES_RELATIVE_PATH.parent.name
    ):
        return False

    selected_epochs = _get_logged_selected_epochs(record)
    epoch_summaries = outputs.get("epoch_summaries")
    if selected_epochs is None or not isinstance(epoch_summaries, dict):
        return False
    if set(epoch_summaries) != set(selected_epochs):
        return False

    summary_counts: dict[str, int] = {}
    for epoch in selected_epochs:
        summary = epoch_summaries.get(epoch)
        if not isinstance(summary, dict):
            return False
        ripple_count = summary.get("ripple_count")
        if isinstance(ripple_count, bool) or not isinstance(ripple_count, int):
            return False
        if ripple_count < 0:
            return False
        summary_counts[epoch] = int(ripple_count)

    positive_summary_counts = {
        epoch: count
        for epoch, count in summary_counts.items()
        if count > 0
    }
    if positive_summary_counts != _get_ripple_event_counts(ripple_table):
        return False
    if sum(summary_counts.values()) != len(ripple_table):
        return False

    run_epochs = outputs.get("run_epochs")
    if run_epochs is not None:
        if not isinstance(run_epochs, list):
            return False
        if [str(epoch) for epoch in run_epochs] != selected_epochs:
            return False

    skipped_epochs = outputs.get("skipped_existing_output_epochs")
    if skipped_epochs is not None and skipped_epochs != []:
        return False
    return True


def load_matching_ripple_run_log(
    analysis_path: Path,
    *,
    animal_name: str,
    date: str,
    ripple_table: "pd.DataFrame",
) -> tuple[dict[str, Any], Path]:
    """Load the unique full speed-gated detector log matching the ripple table."""
    log_dir = analysis_path / "v1ca1_log"
    matching_logs: list[tuple[dict[str, Any], Path]] = []
    if log_dir.exists():
        for log_path in sorted(log_dir.glob(RIPPLE_RUN_LOG_GLOB)):
            try:
                record = json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            if _ripple_run_log_matches_table(
                record,
                animal_name=animal_name,
                date=date,
                ripple_table=ripple_table,
            ):
                matching_logs.append((record, log_path))

    if not matching_logs:
        raise ValueError(
            "No complete speed-gated ripple detector run log matches "
            f"{analysis_path / RIPPLE_TIMES_RELATIVE_PATH}. Rerun "
            "`python -m v1ca1.ripple.detect_ripples --animal-name ... "
            "--date ... --overwrite` before exporting ripples."
        )
    if len(matching_logs) > 1:
        operational_parameters = {
            "animal_name",
            "date",
            "data_root",
            "nwb_root",
            "epochs",
            "overwrite",
        }
        configuration_records = []
        for record, _log_path in matching_logs:
            parameters = record["parameters"]
            configuration_records.append(
                {
                    "package_version": record.get("package_version"),
                    "git_commit": record.get("git_commit"),
                    "git_dirty": record.get("git_dirty"),
                    "parameters": {
                        key: value
                        for key, value in parameters.items()
                        if key not in operational_parameters
                    },
                }
            )
        configurations = {
            json.dumps(configuration, sort_keys=True)
            for configuration in configuration_records
        }
        all_clean = all(
            configuration["git_dirty"] is False
            for configuration in configuration_records
        )
        if len(configurations) != 1 or not all_clean:
            matching_paths = [str(log_path) for _record, log_path in matching_logs]
            raise ValueError(
                "Multiple speed-gated ripple detector run logs match the ripple "
                "table with conflicting or uncommitted configurations, so "
                f"provenance is ambiguous: {matching_paths!r}."
            )
        matching_logs.sort(
            key=lambda item: (
                str(item[0].get("timestamp_utc", "")),
                item[1].name,
            )
        )
        return matching_logs[-1]
    return matching_logs[0]


def _sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_relative_or_absolute(path: Path, analysis_path: Path) -> str:
    """Return an analysis-relative path when possible, otherwise an absolute path."""
    try:
        return path.relative_to(analysis_path).as_posix()
    except ValueError:
        return str(path)


def _path_ends_with(path_value: Any, expected_suffix: Path) -> bool:
    """Return whether one logged path ends with the expected path components."""
    if not isinstance(path_value, str) or not path_value:
        return False
    path_parts = Path(path_value).parts
    suffix_parts = expected_suffix.parts
    if len(path_parts) < len(suffix_parts):
        return False
    return path_parts[-len(suffix_parts) :] == suffix_parts


def _read_json_run_logs(
    analysis_path: Path,
    glob_patterns: tuple[str, ...],
) -> list[tuple[dict[str, Any], Path]]:
    """Return readable JSON run-log records matching any requested glob."""
    log_dir = analysis_path / "v1ca1_log"
    if not log_dir.exists():
        return []

    records: list[tuple[dict[str, Any], Path]] = []
    seen_paths: set[Path] = set()
    for glob_pattern in glob_patterns:
        for log_path in sorted(log_dir.glob(glob_pattern)):
            if log_path in seen_paths:
                continue
            seen_paths.add(log_path)
            try:
                record = json.loads(log_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                records.append((record, log_path))
    return records


def _parse_run_log_timestamp(record: dict[str, Any]) -> float | None:
    """Return one run-log UTC timestamp as Unix seconds when parseable."""
    from datetime import datetime

    timestamp_value = record.get("timestamp_utc")
    if not isinstance(timestamp_value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        return None
    return float(timestamp.timestamp())


def _resolve_log_match(
    *,
    candidates: list[tuple[dict[str, Any], Path]],
    artifact_path: Path,
    analysis_path: Path,
) -> dict[str, Any]:
    """Resolve structural run-log candidates without blocking on ambiguity."""
    candidate_paths = [
        _analysis_relative_or_absolute(log_path, analysis_path)
        for _record, log_path in candidates
    ]
    artifact_mtime_s = float(artifact_path.stat().st_mtime)
    resolution: dict[str, Any] = {
        "artifact_mtime_utc_s": artifact_mtime_s,
        "candidate_count": int(len(candidates)),
        "candidate_paths": candidate_paths,
    }
    if not candidates:
        return {
            **resolution,
            "status": "missing",
            "selection_basis": "no structural match",
        }

    selected: tuple[dict[str, Any], Path] | None = None
    selection_basis = "unique structural match"
    if len(candidates) == 1:
        selected = candidates[0]
    else:
        temporal_candidates: list[
            tuple[float, dict[str, Any], Path]
        ] = []
        for record, log_path in candidates:
            log_timestamp_s = _parse_run_log_timestamp(record)
            if log_timestamp_s is None:
                continue
            delta_s = log_timestamp_s - artifact_mtime_s
            if 0.0 <= delta_s <= 300.0:
                temporal_candidates.append((delta_s, record, log_path))
        temporal_candidates.sort(key=lambda item: (item[0], item[2].name))
        if temporal_candidates:
            best_delta_s = temporal_candidates[0][0]
            tied_best = [
                item
                for item in temporal_candidates
                if np.isclose(item[0], best_delta_s, rtol=0.0, atol=1e-6)
            ]
            if len(tied_best) == 1:
                _delta_s, record, log_path = tied_best[0]
                selected = (record, log_path)
                selection_basis = (
                    "structural match plus nearest run timestamp within "
                    "five minutes after artifact mtime"
                )
                resolution["selected_log_delay_after_artifact_s"] = float(
                    best_delta_s
                )

    if selected is None:
        return {
            **resolution,
            "status": "ambiguous",
            "selection_basis": (
                "multiple structural matches without a unique near-after "
                "artifact timestamp"
            ),
        }

    record, log_path = selected
    return {
        **resolution,
        "status": "matched",
        "selection_basis": selection_basis,
        "analysis_relative_path": _analysis_relative_or_absolute(
            log_path,
            analysis_path,
        ),
        "record": record,
    }


def _position_producer_log_matches(
    record: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    epoch_order: list[str],
    row_count: int,
) -> bool:
    """Return whether one producer log structurally matches the position table."""
    script_name = record.get("script")
    if script_name not in POSITION_PRODUCER_SCRIPTS:
        return False
    parameters = record.get("parameters")
    outputs = record.get("outputs")
    if not isinstance(parameters, dict) or not isinstance(outputs, dict):
        return False
    if str(parameters.get("animal_name")) != str(animal_name):
        return False
    if str(parameters.get("date")) != str(date):
        return False
    if not _path_ends_with(
        outputs.get("combined_output_path"),
        POSITION_RELATIVE_PATH,
    ):
        return False
    combined_frame_count = outputs.get("combined_frame_count")
    if (
        isinstance(combined_frame_count, bool)
        or not isinstance(combined_frame_count, int)
        or int(combined_frame_count) != row_count
    ):
        return False

    epoch_field = (
        "processed_epochs"
        if script_name == "v1ca1.position.combine_clean_dlc_position"
        else "written_epochs"
    )
    logged_epochs = outputs.get(epoch_field)
    if not isinstance(logged_epochs, list):
        return False
    if [str(epoch) for epoch in logged_epochs] != epoch_order:
        return False

    if script_name == "v1ca1.position.combine_clean_dlc_position":
        source_paths_by_epoch = outputs.get("source_paths_by_epoch")
        if not isinstance(source_paths_by_epoch, dict):
            return False
        if [str(epoch) for epoch in source_paths_by_epoch] != epoch_order:
            return False
        if not all(
            isinstance(path_value, str) and bool(path_value)
            for path_value in source_paths_by_epoch.values()
        ):
            return False
    return True


def _clean_position_log_matches(
    record: dict[str, Any],
    *,
    animal_name: str,
    date: str,
    epoch: str,
    source_path: Path,
    frame_count: int,
    producer_timestamp_s: float | None,
) -> bool:
    """Return whether one per-epoch cleaning log matches a combiner source."""
    if record.get("script") != "v1ca1.position.clean_dlc_position":
        return False
    parameters = record.get("parameters")
    outputs = record.get("outputs")
    if not isinstance(parameters, dict) or not isinstance(outputs, dict):
        return False
    if str(parameters.get("animal_name")) != str(animal_name):
        return False
    if str(parameters.get("date")) != str(date):
        return False
    if str(parameters.get("epoch")) != str(epoch):
        return False
    cleaned_output_path = outputs.get("cleaned_output_path")
    if not _path_ends_with(
        cleaned_output_path,
        Path(source_path.parent.name) / source_path.name,
    ):
        return False
    logged_frame_count = outputs.get("frame_count")
    if (
        isinstance(logged_frame_count, bool)
        or not isinstance(logged_frame_count, int)
        or int(logged_frame_count) != frame_count
    ):
        return False
    if producer_timestamp_s is not None:
        clean_timestamp_s = _parse_run_log_timestamp(record)
        if clean_timestamp_s is not None and clean_timestamp_s > producer_timestamp_s:
            return False
    return True


def load_position_producer_provenance(
    *,
    analysis_path: Path,
    animal_name: str,
    date: str,
    position_path: Path,
    position_table: "pd.DataFrame",
) -> dict[str, Any]:
    """Return producer-aware run-log provenance for one position artifact."""
    epoch_order = list(dict.fromkeys(position_table["epoch"].tolist()))
    producer_candidates = [
        (record, log_path)
        for record, log_path in _read_json_run_logs(
            analysis_path,
            POSITION_PRODUCER_LOG_GLOBS,
        )
        if _position_producer_log_matches(
            record,
            animal_name=animal_name,
            date=date,
            epoch_order=epoch_order,
            row_count=len(position_table),
        )
    ]
    producer = _resolve_log_match(
        candidates=producer_candidates,
        artifact_path=position_path,
        analysis_path=analysis_path,
    )
    if producer.get("status") != "matched":
        return producer

    producer_record = producer["record"]
    script_name = str(producer_record.get("script"))
    producer["producer_script"] = script_name
    if script_name != "v1ca1.position.combine_clean_dlc_position":
        return producer

    outputs = producer_record["outputs"]
    source_paths_by_epoch = outputs["source_paths_by_epoch"]
    producer_timestamp_s = _parse_run_log_timestamp(producer_record)
    cleaning_logs = _read_json_run_logs(
        analysis_path,
        ("v1ca1_position_clean_dlc_position_*.json",),
    )
    epoch_frame_counts = {
        str(epoch): int(count)
        for epoch, count in position_table.groupby("epoch", sort=False).size().items()
    }
    cleaning_provenance: dict[str, Any] = {}
    for epoch in epoch_order:
        logged_source_path = Path(str(source_paths_by_epoch[epoch]))
        current_source_path = (
            analysis_path
            / POSITION_RELATIVE_PATH.parent
            / logged_source_path.name
        )
        source_record: dict[str, Any] = {
            "logged_path": str(logged_source_path),
            "analysis_relative_candidate_path": (
                current_source_path.relative_to(analysis_path).as_posix()
            ),
            "exists_at_export": current_source_path.exists(),
        }
        if current_source_path.exists():
            source_record["sha256_at_export"] = _sha256_file(current_source_path)

        matching_clean_logs = [
            (record, log_path)
            for record, log_path in cleaning_logs
            if _clean_position_log_matches(
                record,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
                source_path=logged_source_path,
                frame_count=epoch_frame_counts[epoch],
                producer_timestamp_s=producer_timestamp_s,
            )
        ]
        log_match = _resolve_log_match(
            candidates=matching_clean_logs,
            artifact_path=(
                current_source_path
                if current_source_path.exists()
                else position_path
            ),
            analysis_path=analysis_path,
        )
        cleaning_provenance[epoch] = {
            "source_artifact": source_record,
            "cleaning_run_log": log_match,
        }

    producer["per_epoch_cleaning"] = cleaning_provenance
    return producer


def build_position_provenance(
    *,
    analysis_path: Path,
    position_path: Path,
    position_table: "pd.DataFrame",
    epoch_ranges: list[dict[str, Any]],
    timestamps_reference_time: Any,
    producer_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build structured provenance for exported head/body position."""
    epoch_order = [str(epoch_range["epoch"]) for epoch_range in epoch_ranges]
    frame_counts_by_epoch = {
        str(epoch_range["epoch"]): int(epoch_range["sample_count"])
        for epoch_range in epoch_ranges
    }
    coordinate_columns = [
        "head_x_cm",
        "head_y_cm",
        "body_x_cm",
        "body_y_cm",
    ]
    nan_counts_by_epoch = {
        epoch: {
            column_name: int(
                position_table.loc[
                    position_table["epoch"] == epoch,
                    column_name,
                ].isna().sum()
            )
            for column_name in coordinate_columns
        }
        for epoch in epoch_order
    }
    timestamp_validation = {
        str(epoch_range["epoch"]): {
            "video_series_name": str(epoch_range["video_series_name"]),
            "sample_count": int(epoch_range["sample_count"]),
            "start_time_s": float(epoch_range["start_time"]),
            "stop_time_s": float(epoch_range["stop_time"]),
            "max_abs_error_s": float(
                epoch_range["max_abs_timestamp_error_s"]
            ),
        }
        for epoch_range in epoch_ranges
    }
    reference_time_value = (
        timestamps_reference_time.isoformat()
        if hasattr(timestamps_reference_time, "isoformat")
        else str(timestamps_reference_time)
    )
    limitations = [
        (
            "Producer run logs do not contain artifact checksums; matched "
            "logs are selected from structural fields and, when needed, "
            "artifact/run timestamp chronology."
        ),
        (
            "Cleaning settings absent from an embedded historical run log "
            "are not inferred from current code defaults."
        ),
        (
            "The combined parquet omits raw DLC coordinates, likelihoods, "
            "and per-frame cleaning/interpolation diagnostics."
        ),
    ]
    if producer_provenance.get("producer_script") == (
        "v1ca1.position.convert_legacy_position_pickles"
    ):
        limitations.append(
            "The legacy converter log does not establish upstream DLC "
            "cleaning parameters or independently verify physical scaling."
        )
    return {
        "schema_version": POSITION_PROVENANCE_SCHEMA_VERSION,
        "spatial_series": {
            "interface_path": (
                f"/processing/behavior/{POSITION_INTERFACE_NAME}"
            ),
            "head_series_name": HEAD_POSITION_SERIES_NAME,
            "body_series_name": BODY_POSITION_SERIES_NAME,
            "sample_metadata_path": (
                f"/processing/behavior/{POSITION_SAMPLE_METADATA_NAME}"
            ),
            "epoch_intervals_path": (
                f"/intervals/{POSITION_EPOCHS_TABLE_NAME}"
            ),
        },
        "source_artifact": {
            "analysis_relative_path": position_path.relative_to(
                analysis_path
            ).as_posix(),
            "sha256": _sha256_file(position_path),
            "row_count": int(len(position_table)),
            "imported_columns": [
                str(column_name)
                for column_name in position_table.columns
            ],
            "imported_dtypes": {
                str(column_name): str(dtype)
                for column_name, dtype in position_table.dtypes.items()
            },
            "epoch_order": epoch_order,
            "frame_counts_by_epoch": frame_counts_by_epoch,
            "nan_counts_by_epoch": nan_counts_by_epoch,
            "source_row_order_preserved": True,
        },
        "timestamps": {
            "unit": "seconds",
            "source": "source NWB video ImageSeries.timestamps",
            "timestamps_reference_time": reference_time_value,
            "validation_tolerance_s": 1e-9,
            "validation_by_epoch": timestamp_validation,
        },
        "coordinates": {
            "unit": POSITION_SPATIAL_UNIT,
            "conversion": 1.0,
            "reference_frame": POSITION_REFERENCE_FRAME,
            "nan_values_preserved": True,
        },
        "producer": producer_provenance,
        "limitations": limitations,
    }


def build_ripple_detection_provenance(
    *,
    analysis_path: Path,
    ripple_path: Path,
    ripple_table: "pd.DataFrame",
    run_log_path: Path,
    run_log_record: dict[str, Any],
) -> dict[str, Any]:
    """Build structured provenance for the exported speed-gated ripples."""
    return {
        "schema_version": RIPPLE_PROVENANCE_SCHEMA_VERSION,
        "intervals_table_name": RIPPLES_INTERVALS_TABLE_NAME,
        "source_artifact": {
            "analysis_relative_path": ripple_path.relative_to(analysis_path).as_posix(),
            "sha256": _sha256_file(ripple_path),
            "row_count": int(len(ripple_table)),
            "imported_columns": [
                str(column_name)
                for column_name in ripple_table.columns
            ],
            "imported_dtypes": {
                str(column_name): str(dtype)
                for column_name, dtype in ripple_table.dtypes.items()
            },
            "event_counts_by_epoch": _get_ripple_event_counts(ripple_table),
            "source_row_order_preserved": True,
        },
        "run_log": {
            "analysis_relative_path": run_log_path.relative_to(analysis_path).as_posix(),
            "record": run_log_record,
        },
        "limitations": [
            (
                "The source parquet does not contain per-event detector run IDs; "
                "the embedded run log is matched by full epoch coverage and exact "
                "per-epoch ripple counts."
            ),
            (
                "Detector settings absent from the embedded run log are not inferred "
                "from the export environment."
            ),
        ],
    }


def _require_available_interval_table_name(
    nwbfile: "pynwb.NWBFile",
    table_name: str,
) -> None:
    """Reject a source NWB that already contains one destination interval table."""
    if table_name in nwbfile.intervals:
        raise ValueError(
            f"NWB file already contains an intervals table named {table_name!r}."
        )


def _require_available_scratch_name(
    nwbfile: "pynwb.NWBFile",
    scratch_name: str,
) -> None:
    """Reject a source NWB that already contains one destination scratch item."""
    if scratch_name in nwbfile.scratch:
        raise ValueError(
            f"NWB file already contains scratch data named {scratch_name!r}."
        )


def add_ephys_recording_intervals_to_nwb(
    nwbfile: "pynwb.NWBFile",
    epoch_tags: list[str],
    start_times: np.ndarray,
    stop_times: np.ndarray,
) -> int:
    """Add saved ephys recording intervals while preserving the source epochs."""
    _require_available_interval_table_name(nwbfile, EPHYS_INTERVALS_TABLE_NAME)

    interval_table = nwbfile.create_time_intervals(
        name=EPHYS_INTERVALS_TABLE_NAME,
        description=EPHYS_INTERVALS_DESCRIPTION,
    )
    for column_name, description in _EPHYS_INTERVAL_COLUMN_DESCRIPTIONS.items():
        interval_table.add_column(column_name, description)

    for epoch, start_time, stop_time in zip(
        epoch_tags,
        start_times,
        stop_times,
        strict=True,
    ):
        interval_table.add_interval(
            start_time=float(start_time),
            stop_time=float(stop_time),
            epoch=str(epoch),
        )
    nwbfile.set_modified()
    return len(epoch_tags)


def add_trajectory_intervals_to_nwb(
    nwbfile: "pynwb.NWBFile",
    trajectory_table: "pd.DataFrame",
    ephys_bounds_by_epoch: dict[str, tuple[float, float]],
) -> int:
    """Add saved poke-defined trajectory intervals to one NWB object."""
    _require_available_interval_table_name(nwbfile, TRAJECTORY_INTERVALS_TABLE_NAME)

    unknown_epochs = sorted(set(trajectory_table["epoch"]).difference(ephys_bounds_by_epoch))
    if unknown_epochs:
        raise ValueError(
            f"{TRAJECTORY_TIMES_FILENAME} contains epochs absent from "
            f"{EPHYS_INTERVALS_TABLE_NAME}: {unknown_epochs!r}."
        )

    for row in trajectory_table.itertuples(index=False):
        epoch_start, epoch_stop = ephys_bounds_by_epoch[str(row.epoch)]
        if float(row.start) < epoch_start or float(row.end) > epoch_stop:
            raise ValueError(
                f"{TRAJECTORY_TIMES_FILENAME} interval "
                f"({float(row.start)}, {float(row.end)}) falls outside the "
                f"ephys bounds for epoch {str(row.epoch)!r}: "
                f"({epoch_start}, {epoch_stop})."
            )

    interval_table = nwbfile.create_time_intervals(
        name=TRAJECTORY_INTERVALS_TABLE_NAME,
        description=TRAJECTORY_INTERVALS_DESCRIPTION,
    )
    for column_name, description in _TRAJECTORY_INTERVAL_COLUMN_DESCRIPTIONS.items():
        if trajectory_table.empty:
            interval_table.add_column(
                column_name,
                description,
                data=np.asarray([], dtype=object),
            )
        else:
            interval_table.add_column(column_name, description)

    for row in trajectory_table.itertuples(index=False):
        interval_table.add_interval(
            start_time=float(row.start),
            stop_time=float(row.end),
            epoch=str(row.epoch),
            trajectory_type=str(row.trajectory_type),
        )
    nwbfile.set_modified()
    return int(len(trajectory_table))


def add_ripples_to_nwb(
    nwbfile: "pynwb.NWBFile",
    ripple_table: "pd.DataFrame",
    ephys_bounds_by_epoch: dict[str, tuple[float, float]],
    provenance: dict[str, Any],
) -> int:
    """Add speed-gated ripple intervals and their provenance to one NWB object."""
    _require_available_interval_table_name(nwbfile, RIPPLES_INTERVALS_TABLE_NAME)
    _require_available_scratch_name(nwbfile, RIPPLE_PROVENANCE_SCRATCH_NAME)

    unknown_epochs = sorted(set(ripple_table["epoch"]).difference(ephys_bounds_by_epoch))
    if unknown_epochs:
        raise ValueError(
            f"{RIPPLE_TIMES_RELATIVE_PATH} contains epochs absent from "
            f"{EPHYS_INTERVALS_TABLE_NAME}: {unknown_epochs!r}."
        )

    for row in ripple_table.itertuples(index=False):
        epoch = str(row.epoch)
        epoch_start, epoch_stop = ephys_bounds_by_epoch[epoch]
        if float(row.start) < epoch_start or float(row.end) > epoch_stop:
            raise ValueError(
                f"{RIPPLE_TIMES_RELATIVE_PATH} ripple "
                f"({float(row.start)}, {float(row.end)}) falls outside the "
                f"ephys bounds for epoch {epoch!r}: "
                f"({epoch_start}, {epoch_stop})."
            )

    interval_table = nwbfile.create_time_intervals(
        name=RIPPLES_INTERVALS_TABLE_NAME,
        description=RIPPLES_INTERVALS_DESCRIPTION,
    )
    custom_columns = [
        column_name
        for column_name in ripple_table.columns
        if column_name not in {"start", "end"}
    ]
    for column_name in custom_columns:
        description = _RIPPLE_INTERVAL_COLUMN_DESCRIPTIONS.get(
            column_name,
            f"Value imported from {RIPPLE_TIMES_RELATIVE_PATH}.",
        )
        if ripple_table.empty:
            interval_table.add_column(
                column_name,
                description,
                data=ripple_table[column_name].to_numpy(copy=True),
            )
        else:
            interval_table.add_column(column_name, description)

    for record in ripple_table.to_dict("records"):
        interval_values = {}
        for column_name in custom_columns:
            value = record[column_name]
            if isinstance(value, np.generic):
                value = value.item()
            interval_values[column_name] = value
        interval_table.add_interval(
            start_time=float(record["start"]),
            stop_time=float(record["end"]),
            **interval_values,
        )

    provenance_json = json.dumps(
        provenance,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    nwbfile.add_scratch(
        provenance_json,
        name=RIPPLE_PROVENANCE_SCRATCH_NAME,
        description=RIPPLE_PROVENANCE_DESCRIPTION,
    )
    nwbfile.set_modified()
    return int(len(ripple_table))


def _build_wtrack_linearization_configuration(
    *,
    configuration_name: str,
    node_positions: np.ndarray,
    edges: np.ndarray,
    edge_order: list[tuple[int, int]],
    edge_spacing_cm: np.ndarray | list[float],
) -> dict[str, Any]:
    """Validate and normalize one saved W-track graph configuration."""
    node_positions_array = np.asarray(node_positions, dtype=float)
    edges_array = np.asarray(edges, dtype=np.int64)
    edge_order_array = np.asarray(edge_order, dtype=np.int64)
    edge_spacing_array = np.asarray(edge_spacing_cm, dtype=float).reshape(-1)

    if node_positions_array.ndim != 2 or node_positions_array.shape[1] != 2:
        raise ValueError(
            f"W-track configuration {configuration_name!r} node positions "
            "must have shape (n_nodes, 2)."
        )
    if node_positions_array.shape[0] == 0 or not np.all(
        np.isfinite(node_positions_array)
    ):
        raise ValueError(
            f"W-track configuration {configuration_name!r} contains no nodes "
            "or non-finite node positions."
        )
    if edges_array.ndim != 2 or edges_array.shape[1] != 2:
        raise ValueError(
            f"W-track configuration {configuration_name!r} edges must have "
            "shape (n_edges, 2)."
        )
    if edge_order_array.shape != edges_array.shape:
        raise ValueError(
            f"W-track configuration {configuration_name!r} edge_order must "
            "have the same shape as edges."
        )
    node_ids = np.concatenate((edges_array.reshape(-1), edge_order_array.reshape(-1)))
    if np.any(node_ids < 0) or np.any(node_ids >= node_positions_array.shape[0]):
        raise ValueError(
            f"W-track configuration {configuration_name!r} contains an edge "
            "node id outside node_positions."
        )
    expected_spacing_count = max(0, edge_order_array.shape[0] - 1)
    if edge_spacing_array.size != expected_spacing_count:
        raise ValueError(
            f"W-track configuration {configuration_name!r} must contain "
            f"{expected_spacing_count} edge spacing values."
        )
    if not np.all(np.isfinite(edge_spacing_array)) or np.any(edge_spacing_array < 0):
        raise ValueError(
            f"W-track configuration {configuration_name!r} contains invalid "
            "edge spacing values."
        )

    return {
        "configuration_name": str(configuration_name),
        "node_positions_cm": node_positions_array,
        "edges": edges_array,
        "edge_order": edge_order_array,
        "edge_spacing_cm": edge_spacing_array,
        "use_hmm": False,
    }


def build_wtrack_linearization_configurations(
    animal_name: str,
) -> list[dict[str, Any]]:
    """Build the graph configurations used by current W-track analyses."""
    configurations: list[dict[str, Any]] = []
    for trajectory_type in TRAJECTORY_TYPES:
        node_positions, edges, edge_order = get_wtrack_branch_graph_inputs(
            animal_name=animal_name,
            branch_side=get_wtrack_branch_side(trajectory_type),
            direction=get_wtrack_direction(trajectory_type),
        )
        configurations.append(
            _build_wtrack_linearization_configuration(
                configuration_name=trajectory_type,
                node_positions=node_positions,
                edges=edges,
                edge_order=edge_order,
                edge_spacing_cm=np.zeros(max(0, len(edge_order) - 1)),
            )
        )

    node_positions, edges, edge_order, edge_spacing = (
        get_wtrack_full_graph_inputs(
            animal_name=animal_name,
            branch_gap_cm=DEFAULT_WTRACK_BRANCH_GAP_CM,
        )
    )
    configurations.append(
        _build_wtrack_linearization_configuration(
            configuration_name=WTRACK_FULL_GRAPH_CONFIGURATION_NAME,
            node_positions=node_positions,
            edges=edges,
            edge_order=edge_order,
            edge_spacing_cm=edge_spacing,
        )
    )
    return configurations


def _validate_wtrack_position_units(behavior_module: Any) -> None:
    """Require any canonical position series to use the W-track coordinate unit."""
    if behavior_module is None:
        return
    position_interface = behavior_module.data_interfaces.get(
        POSITION_INTERFACE_NAME
    )
    spatial_series = getattr(position_interface, "spatial_series", {})
    for series_name in (HEAD_POSITION_SERIES_NAME, BODY_POSITION_SERIES_NAME):
        if series_name not in spatial_series:
            continue
        unit = str(spatial_series[series_name].unit)
        if unit != POSITION_SPATIAL_UNIT:
            raise ValueError(
                f"NWB position series {series_name!r} uses unit {unit!r}; "
                "W-track node positions require 'centimeters'."
            )


def add_wtrack_linearization_to_nwb(
    nwbfile: "pynwb.NWBFile",
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add effective W-track graph inputs as one behavior DynamicTable."""
    from hdmf.common import DynamicTable

    behavior_module = nwbfile.processing.get("behavior")
    if (
        behavior_module is not None
        and WTRACK_LINEARIZATION_TABLE_NAME in behavior_module.data_interfaces
    ):
        raise ValueError(
            "NWB behavior module already contains a data interface named "
            f"{WTRACK_LINEARIZATION_TABLE_NAME!r}."
        )
    _validate_wtrack_position_units(behavior_module)

    if behavior_module is None:
        behavior_module = nwbfile.create_processing_module(
            name="behavior",
            description="Behavioral data.",
        )

    table = DynamicTable(
        name=WTRACK_LINEARIZATION_TABLE_NAME,
        description=WTRACK_LINEARIZATION_DESCRIPTION,
        id=np.arange(len(configurations), dtype=np.int64),
    )
    table.add_column(
        name="configuration_name",
        description=_WTRACK_LINEARIZATION_COLUMN_DESCRIPTIONS[
            "configuration_name"
        ],
        data=[
            configuration["configuration_name"]
            for configuration in configurations
        ],
    )
    for column_name in ("node_positions_cm", "edges", "edge_order"):
        table.add_column(
            name=column_name,
            description=_WTRACK_LINEARIZATION_COLUMN_DESCRIPTIONS[column_name],
            data=[
                np.asarray(configuration[column_name]).tolist()
                for configuration in configurations
            ],
            index=2,
        )
    table.add_column(
        name="edge_spacing_cm",
        description=_WTRACK_LINEARIZATION_COLUMN_DESCRIPTIONS["edge_spacing_cm"],
        data=[
            np.asarray(configuration["edge_spacing_cm"], dtype=float).tolist()
            for configuration in configurations
        ],
        index=True,
    )
    table.add_column(
        name="use_hmm",
        description=_WTRACK_LINEARIZATION_COLUMN_DESCRIPTIONS["use_hmm"],
        data=np.asarray(
            [configuration["use_hmm"] for configuration in configurations],
            dtype=bool,
        ),
    )
    behavior_module.add(table)
    nwbfile.set_modified()
    return {
        "wtrack_linearization_table_path": (
            f"/processing/behavior/{WTRACK_LINEARIZATION_TABLE_NAME}"
        ),
        "wtrack_linearization_configuration_names": [
            str(configuration["configuration_name"])
            for configuration in configurations
        ],
        "wtrack_linearization_configuration_count": int(len(configurations)),
        "wtrack_linearization_position_unit": POSITION_SPATIAL_UNIT,
    }


def add_position_to_nwb(
    nwbfile: "pynwb.NWBFile",
    position_table: "pd.DataFrame",
    epoch_ranges: list[dict[str, Any]],
    provenance: dict[str, Any],
    position_offset: int,
) -> dict[str, Any]:
    """Add canonical head/body position and explicit epoch membership to one NWB."""
    import pynwb
    from hdmf.common import DynamicTable

    position_offset = validate_position_offset(position_offset)
    _require_available_interval_table_name(
        nwbfile,
        POSITION_EPOCHS_TABLE_NAME,
    )
    _require_available_scratch_name(
        nwbfile,
        POSITION_PROVENANCE_SCRATCH_NAME,
    )

    behavior_module = nwbfile.processing.get("behavior")
    position_interface = None
    if behavior_module is not None:
        if POSITION_SAMPLE_METADATA_NAME in behavior_module.data_interfaces:
            raise ValueError(
                "NWB behavior module already contains a data interface named "
                f"{POSITION_SAMPLE_METADATA_NAME!r}."
            )
        position_interface = behavior_module.data_interfaces.get(
            POSITION_INTERFACE_NAME
        )
        if position_interface is not None and not isinstance(
            position_interface,
            pynwb.behavior.Position,
        ):
            raise ValueError(
                "NWB behavior data interface named "
                f"{POSITION_INTERFACE_NAME!r} is not a Position object."
            )
        if position_interface is not None:
            existing_series_names = set(position_interface.spatial_series)
            colliding_series_names = sorted(
                existing_series_names.intersection(
                    {
                        HEAD_POSITION_SERIES_NAME,
                        BODY_POSITION_SERIES_NAME,
                    }
                )
            )
            if colliding_series_names:
                raise ValueError(
                    "NWB Position interface already contains destination "
                    f"SpatialSeries names: {colliding_series_names!r}."
                )

    if behavior_module is None:
        behavior_module = nwbfile.create_processing_module(
            name="behavior",
            description="Behavioral data.",
        )
    if position_interface is None:
        position_interface = pynwb.behavior.Position(
            name=POSITION_INTERFACE_NAME
        )
        behavior_module.add(position_interface)

    frame_times = position_table["frame_time_s"].to_numpy(dtype=float)
    head_position = position_table.loc[
        :,
        ["head_x_cm", "head_y_cm"],
    ].to_numpy(dtype=float)
    body_position = position_table.loc[
        :,
        ["body_x_cm", "body_y_cm"],
    ].to_numpy(dtype=float)

    head_series = position_interface.create_spatial_series(
        name=HEAD_POSITION_SERIES_NAME,
        data=head_position,
        timestamps=frame_times,
        unit=POSITION_SPATIAL_UNIT,
        conversion=1.0,
        reference_frame=POSITION_REFERENCE_FRAME,
        description=POSITION_SERIES_DESCRIPTION.format(bodypart="head"),
    )
    position_interface.create_spatial_series(
        name=BODY_POSITION_SERIES_NAME,
        data=body_position,
        timestamps=head_series,
        unit=POSITION_SPATIAL_UNIT,
        conversion=1.0,
        reference_frame=POSITION_REFERENCE_FRAME,
        description=POSITION_SERIES_DESCRIPTION.format(bodypart="body"),
    )

    sample_metadata = DynamicTable(
        name=POSITION_SAMPLE_METADATA_NAME,
        description=POSITION_SAMPLE_METADATA_DESCRIPTION,
        id=np.arange(len(position_table), dtype=np.int64),
    )
    sample_metadata.add_column(
        name="epoch",
        description="Source NWB epoch tag for this position sample.",
        data=position_table["epoch"].tolist(),
    )
    sample_metadata.add_column(
        name="frame",
        description=(
            "Original per-epoch video frame identifier retained from the "
            "combined position artifact."
        ),
        data=position_table["frame"].to_numpy(dtype=np.int64),
    )
    behavior_module.add(sample_metadata)

    epoch_intervals = nwbfile.create_time_intervals(
        name=POSITION_EPOCHS_TABLE_NAME,
        description=POSITION_EPOCHS_DESCRIPTION,
    )
    for column_name, description in _POSITION_EPOCH_COLUMN_DESCRIPTIONS.items():
        epoch_intervals.add_column(column_name, description)
    for epoch_range in epoch_ranges:
        epoch_intervals.add_interval(
            start_time=float(epoch_range["start_time"]),
            stop_time=float(epoch_range["stop_time"]),
            epoch=str(epoch_range["epoch"]),
            start_index=int(epoch_range["start_index"]),
            stop_index_exclusive=int(
                epoch_range["stop_index_exclusive"]
            ),
            sample_count=int(epoch_range["sample_count"]),
            analysis_start_offset_samples=position_offset,
            first_frame=int(epoch_range["first_frame"]),
            last_frame=int(epoch_range["last_frame"]),
            video_series_name=str(epoch_range["video_series_name"]),
        )

    provenance_json = json.dumps(
        provenance,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    nwbfile.add_scratch(
        provenance_json,
        name=POSITION_PROVENANCE_SCRATCH_NAME,
        description=POSITION_PROVENANCE_DESCRIPTION,
    )
    nwbfile.set_modified()
    return {
        "position_interface_path": (
            f"/processing/behavior/{POSITION_INTERFACE_NAME}"
        ),
        "position_spatial_series_names": [
            HEAD_POSITION_SERIES_NAME,
            BODY_POSITION_SERIES_NAME,
        ],
        "position_sample_metadata_path": (
            f"/processing/behavior/{POSITION_SAMPLE_METADATA_NAME}"
        ),
        "position_epochs_table_name": POSITION_EPOCHS_TABLE_NAME,
        "position_provenance_scratch_name": (
            POSITION_PROVENANCE_SCRATCH_NAME
        ),
        "position_sample_count": int(len(position_table)),
        "position_epoch_count": int(len(epoch_ranges)),
        "position_analysis_start_offset_samples": position_offset,
        "position_epoch_order": [
            str(epoch_range["epoch"])
            for epoch_range in epoch_ranges
        ],
    }


def configure_ephys_compression(nwbfile: "pynwb.NWBFile") -> list[str]:
    """Configure streamed compression for every source ElectricalSeries."""
    import h5py
    from hdmf.backends.hdf5 import H5DataIO
    from hdmf.data_utils import DataChunkIterator
    from pynwb.ecephys import ElectricalSeries

    dataset_paths: list[str] = []
    for series in nwbfile.objects.values():
        if not isinstance(series, ElectricalSeries):
            continue

        source_data = series.data
        if not isinstance(source_data, h5py.Dataset):
            raise TypeError(
                "ElectricalSeries compression requires HDF5-backed source data. "
                f"Series {series.name!r} contains {type(source_data).__name__}."
            )

        source_chunks = source_data.chunks
        default_buffer_rows = source_chunks[0] if source_chunks is not None else 4096
        buffer_rows = max(1, min(int(default_buffer_rows), int(source_data.shape[0])))
        dataset_paths.append(str(source_data.name))
        series.set_data_io(
            dataset_name="data",
            data_io_class=H5DataIO,
            data_io_kwargs={
                "chunks": source_chunks if source_chunks is not None else True,
                "maxshape": source_data.maxshape,
                "compression": EPHYS_COMPRESSION,
                "compression_opts": EPHYS_COMPRESSION_LEVEL,
                "shuffle": EPHYS_COMPRESSION_SHUFFLE,
                "fillvalue": source_data.fillvalue,
            },
            data_chunk_iterator_class=DataChunkIterator,
            data_chunk_iterator_kwargs={
                "maxshape": source_data.shape,
                "dtype": source_data.dtype,
                "buffer_size": buffer_rows,
                "iter_axis": 0,
            },
        )
        # The source build manager otherwise reuses the original uncompressed builder.
        series.set_modified()

    return sorted(dataset_paths)


def validate_ephys_compression(
    output_path: Path,
    dataset_paths: list[str],
) -> None:
    """Require each configured ElectricalSeries dataset to use the requested filters."""
    import h5py

    with h5py.File(output_path, "r") as h5_file:
        for dataset_path in dataset_paths:
            if dataset_path not in h5_file:
                raise ValueError(
                    f"Exported NWB is missing ElectricalSeries data at {dataset_path}."
                )
            dataset = h5_file[dataset_path]
            actual_settings = (
                dataset.compression,
                dataset.compression_opts,
                bool(dataset.shuffle),
            )
            expected_settings = (
                EPHYS_COMPRESSION,
                EPHYS_COMPRESSION_LEVEL,
                EPHYS_COMPRESSION_SHUFFLE,
            )
            if actual_settings != expected_settings:
                raise ValueError(
                    "Exported ElectricalSeries does not use the requested compression "
                    f"at {dataset_path}: expected {expected_settings!r}, "
                    f"found {actual_settings!r}."
                )


def _create_temporary_output_path(output_path: Path) -> Path:
    """Return a temporary sibling path for staging the rewritten NWB file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=output_path.suffix,
        dir=output_path.parent,
        delete=False,
    ) as temp_file:
        return Path(temp_file.name)


def get_region_sorting_path(analysis_path: Path, region: str) -> Path:
    """Return the consolidated sorting folder for one region."""
    return analysis_path / f"sorting_{region}"


def load_consolidated_region_sortings(analysis_path: Path) -> dict[str, Any]:
    """Load any consolidated region sortings available for NWB export."""
    si = get_spikeinterface()
    region_sortings: dict[str, Any] = {}
    for region in REGIONS:
        sorting_path = get_region_sorting_path(analysis_path, region)
        if sorting_path.exists():
            region_sortings[region] = si.load(sorting_path)
    if not region_sortings:
        raise FileNotFoundError(
            "No consolidated region sorting folders were found under "
            f"{analysis_path}. Run v1ca1.spikesorting.consolidate_sorting first."
        )
    return region_sortings


def validate_sorting_provenance(region_sortings: dict[str, Any]) -> None:
    """Require the consolidated sortings to retain stamped provenance properties."""
    for region, sorting in region_sortings.items():
        property_keys = set(sorting.get_property_keys())
        missing_properties = [
            property_name
            for property_name in REQUIRED_SORTING_PROPERTIES
            if property_name not in property_keys
        ]
        if missing_properties:
            raise ValueError(
                "Consolidated sorting for region "
                f"{region!r} is missing required properties {missing_properties!r}. "
                "Rerun v1ca1.spikesorting.consolidate_sorting after updating it to stamp "
                "sorting provenance properties."
            )

        region_values = np.asarray(sorting.get_property("region")).astype(str)
        if np.any(region_values != region):
            raise ValueError(
                f"Consolidated sorting for region {region!r} has inconsistent 'region' property values."
            )


def _get_electrode_row_indices_by_id(nwbfile: "pynwb.NWBFile") -> dict[int, int]:
    """Return NWB electrode table row indices keyed by the electrode id column."""
    if nwbfile.electrodes is None:
        raise ValueError(
            "NWB file does not contain an electrodes table required for spike sorting export."
        )

    electrode_df = nwbfile.electrodes.to_dataframe()
    electrode_ids = [int(electrode_id) for electrode_id in electrode_df.index.tolist()]
    if len(electrode_ids) != len(set(electrode_ids)):
        raise ValueError("NWB electrode table contains duplicate electrode ids.")
    return {
        electrode_id: row_index
        for row_index, electrode_id in enumerate(electrode_ids)
    }


def _get_shank_electrode_rows(
    electrode_rows_by_id: dict[int, int],
    probe_idx: int,
    shank_idx: int,
) -> np.ndarray:
    """Return NWB electrode row indices for one probe/shank using lab channel ids."""
    electrode_ids = list(range(128 * probe_idx + 32 * shank_idx, 128 * probe_idx + 32 * (shank_idx + 1)))
    missing_ids = [electrode_id for electrode_id in electrode_ids if electrode_id not in electrode_rows_by_id]
    if missing_ids:
        raise ValueError(
            "NWB electrode table is missing the expected channel ids for "
            f"probe {probe_idx} shank {shank_idx}: missing {missing_ids!r}."
        )
    return np.asarray([electrode_rows_by_id[electrode_id] for electrode_id in electrode_ids], dtype=int)


def _create_units_table(nwbfile: "pynwb.NWBFile") -> None:
    """Create a fresh NWB units table for curated export."""
    from pynwb.misc import Units

    if nwbfile.electrodes is None:
        raise ValueError(
            "NWB file does not contain an electrodes table required for spike sorting export."
        )
    if nwbfile.units is not None:
        raise ValueError(
            "NWB file already contains a units table. "
            "Spike sorting export currently requires a source NWB without existing units."
        )

    nwbfile.units = Units(
        name="units",
        description=UNITS_TABLE_DESCRIPTION,
        electrode_table=nwbfile.electrodes,
    )
    for column_name, description in _UNITS_COLUMN_DESCRIPTIONS.items():
        nwbfile.add_unit_column(column_name, description)


def _load_curation_provenance_record(
    curation_root: Path,
    region: str,
    probe_idx: int,
    shank_idx: int,
    curation_json_relpath: str,
) -> dict[str, Any]:
    """Load one raw curation JSON payload for storage in NWB scratch."""
    curation_json_path = curation_root / curation_json_relpath
    if not curation_json_path.exists():
        raise FileNotFoundError(
            f"Curation JSON not found for exported units: {curation_json_path}"
        )

    curation_payload = json.loads(curation_json_path.read_text(encoding="utf-8"))
    return {
        "region": region,
        "probe_idx": int(probe_idx),
        "shank_idx": int(shank_idx),
        "curation_json_relpath": curation_json_relpath,
        "labels_by_unit_json": json.dumps(curation_payload.get("labelsByUnit", {}), sort_keys=True),
        "merge_groups_json": json.dumps(curation_payload.get("mergeGroups", [])),
    }


def export_curated_sorting_to_nwb(
    nwbfile: "pynwb.NWBFile",
    analysis_path: Path,
    curation_root: Path,
    ephys_start_times: np.ndarray,
    ephys_stop_times: np.ndarray,
) -> dict[str, Any]:
    """Add curated sorting results and provenance to the NWB units table."""
    import pandas as pd

    region_sortings = load_consolidated_region_sortings(analysis_path)
    validate_sorting_provenance(region_sortings)
    timestamps_ephys_all, timestamps_source = load_ephys_timestamps_all(analysis_path)
    obs_intervals = np.column_stack(
        (
            np.asarray(ephys_start_times, dtype=float),
            np.asarray(ephys_stop_times, dtype=float),
        )
    )
    electrode_rows_by_id = _get_electrode_row_indices_by_id(nwbfile)

    if SORTING_PROVENANCE_SCRATCH_NAME in nwbfile.scratch:
        del nwbfile.scratch[SORTING_PROVENANCE_SCRATCH_NAME]

    _create_units_table(nwbfile)

    provenance_records: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    sorting_source_paths: dict[str, Path] = {}
    sorting_unit_counts: dict[str, int] = {}
    next_unit_id = 0

    for region in REGIONS:
        sorting = region_sortings.get(region)
        if sorting is None:
            continue

        sorting_source_paths[region] = get_region_sorting_path(analysis_path, region)
        property_keys = set(sorting.get_property_keys())
        sorting_unit_ids = list(sorting.get_unit_ids())
        unit_index_by_id = {int(unit_id): index for index, unit_id in enumerate(sorting_unit_ids)}
        region_values = np.asarray(sorting.get_property("region")).astype(str)
        probe_values = np.asarray(sorting.get_property("probe_idx"))
        shank_values = np.asarray(sorting.get_property("shank_idx"))
        curation_relpaths = np.asarray(sorting.get_property("curation_json_relpath")).astype(str)
        is_merged_values = (
            np.asarray(sorting.get_property("is_merged"))
            if "is_merged" in property_keys
            else None
        )

        exported_region_units = 0
        for sorting_unit_id in sorted((int(unit_id) for unit_id in sorting_unit_ids)):
            unit_index = unit_index_by_id[sorting_unit_id]
            region_value = str(region_values[unit_index])
            probe_idx = int(probe_values[unit_index])
            shank_idx = int(shank_values[unit_index])
            curation_json_relpath = str(curation_relpaths[unit_index])
            is_merged = bool(is_merged_values[unit_index]) if is_merged_values is not None else False

            if region_value != region:
                raise ValueError(
                    f"Consolidated sorting for region {region!r} has unit rows tagged as {region_value!r}."
                )
            if not curation_json_relpath:
                raise ValueError(
                    f"Consolidated sorting for region {region!r} has an empty curation_json_relpath value."
                )

            spike_sample_indices = np.asarray(
                sorting.get_unit_spike_train(sorting_unit_id),
                dtype=int,
            ).ravel()
            if spike_sample_indices.size:
                if np.any(spike_sample_indices < 0):
                    raise ValueError(
                        f"Sorting unit {sorting_unit_id} contains negative spike sample indices."
                    )
                if np.any(spike_sample_indices >= timestamps_ephys_all.size):
                    raise ValueError(
                        f"Sorting unit {sorting_unit_id} contains spike samples beyond timestamps_ephys_all."
                    )
            spike_times_s = np.asarray(timestamps_ephys_all[spike_sample_indices], dtype=float)
            electrodes = _get_shank_electrode_rows(
                electrode_rows_by_id=electrode_rows_by_id,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
            )

            nwbfile.add_unit(
                id=next_unit_id,
                spike_times=spike_times_s,
                obs_intervals=obs_intervals,
                electrodes=electrodes,
                region=region_value,
                probe_idx=probe_idx,
                shank_idx=shank_idx,
                sorting_unit_id=int(sorting_unit_id),
                curation_json_relpath=curation_json_relpath,
                is_merged=is_merged,
            )
            next_unit_id += 1
            exported_region_units += 1

            provenance_key = (region_value, probe_idx, shank_idx, curation_json_relpath)
            if provenance_key not in provenance_records:
                provenance_records[provenance_key] = _load_curation_provenance_record(
                    curation_root=curation_root,
                    region=region_value,
                    probe_idx=probe_idx,
                    shank_idx=shank_idx,
                    curation_json_relpath=curation_json_relpath,
                )

        sorting_unit_counts[region] = exported_region_units

    provenance_dataframe = pd.DataFrame(
        [
            provenance_records[key]
            for key in sorted(provenance_records)
        ],
        columns=[
            "region",
            "probe_idx",
            "shank_idx",
            "curation_json_relpath",
            "labels_by_unit_json",
            "merge_groups_json",
        ],
    )
    nwbfile.add_scratch(
        provenance_dataframe,
        name=SORTING_PROVENANCE_SCRATCH_NAME,
        description="Raw sortingview curation payloads used to produce the curated NWB units table.",
    )
    nwbfile.set_modified()
    return {
        "sorting_source_paths": sorting_source_paths,
        "sorting_unit_counts": sorting_unit_counts,
        "timestamps_ephys_all_source": timestamps_source,
        "sorting_curation_provenance_scratch_name": SORTING_PROVENANCE_SCRATCH_NAME,
        "units_table_row_count": int(next_unit_id),
    }


def write_nwb_copy(
    read_io: Any,
    nwbfile: "pynwb.NWBFile",
    output_path: Path,
) -> None:
    """Write the augmented NWB object to a new file."""
    import pynwb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pynwb.NWBHDF5IO(output_path, "w") as write_io:
        export = getattr(write_io, "export", None)
        if callable(export):
            export(src_io=read_io, nwbfile=nwbfile)
        else:
            write_io.write(nwbfile)


def export_augmented_nwb(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    output_path: Path | None = None,
    overwrite: bool = False,
    add_sorting: bool = False,
    curation_root: Path = DEFAULT_CURATION_ROOT,
    add_trajectory_times: bool = False,
    add_ripples: bool = False,
    add_position: bool = False,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    add_wtrack_linearization: bool = False,
) -> Path:
    """Write a new NWB file augmented from saved analysis outputs."""
    import pynwb

    position_offset = validate_position_offset(position_offset)
    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    resolved_output_path = resolve_output_path(
        nwb_path=nwb_path,
        animal_name=animal_name,
        date=date,
        output_path=output_path,
    )

    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    ephys_epoch_tags, ephys_start_times, ephys_stop_times = (
        load_epoch_bounds_npz(analysis_path)
    )
    validate_output_path(source_path=nwb_path, output_path=resolved_output_path, overwrite=overwrite)

    trajectory_table = None
    trajectory_source_path = None
    trajectory_outputs: dict[str, Any] = {
        "trajectory_export_enabled": bool(add_trajectory_times),
    }
    if add_trajectory_times:
        trajectory_table, trajectory_source_path = load_trajectory_times_parquet(
            analysis_path
        )

    ripple_table = None
    ripple_source_path = None
    ripple_provenance = None
    ripple_provenance_log_path = None
    ripple_outputs: dict[str, Any] = {
        "ripple_export_enabled": bool(add_ripples),
    }
    if add_ripples:
        ripple_table, ripple_source_path = load_ripple_times_parquet(analysis_path)
        ripple_run_log, ripple_provenance_log_path = load_matching_ripple_run_log(
            analysis_path,
            animal_name=animal_name,
            date=date,
            ripple_table=ripple_table,
        )
        ripple_provenance = build_ripple_detection_provenance(
            analysis_path=analysis_path,
            ripple_path=ripple_source_path,
            ripple_table=ripple_table,
            run_log_path=ripple_provenance_log_path,
            run_log_record=ripple_run_log,
        )

    position_table = None
    position_source_path = None
    position_producer_provenance = None
    position_outputs: dict[str, Any] = {
        "position_export_enabled": bool(add_position),
    }
    if add_position:
        position_table, position_source_path = load_position_parquet(
            analysis_path
        )
        position_producer_provenance = load_position_producer_provenance(
            analysis_path=analysis_path,
            animal_name=animal_name,
            date=date,
            position_path=position_source_path,
            position_table=position_table,
        )

    wtrack_configurations = None
    wtrack_outputs: dict[str, Any] = {
        "wtrack_linearization_export_enabled": bool(add_wtrack_linearization),
    }
    if add_wtrack_linearization:
        wtrack_configurations = build_wtrack_linearization_configurations(
            animal_name
        )

    print(f"Processing {animal_name} {date}.")
    compressed_dataset_paths: list[str] = []
    sorting_outputs: dict[str, Any] = {
        "sorting_export_enabled": bool(add_sorting),
    }
    with pynwb.NWBHDF5IO(nwb_path, "r") as read_io:
        nwbfile = read_io.read()
        nwb_epoch_tags, _nwb_start_times, _nwb_stop_times = extract_epoch_metadata(nwbfile)
        validate_epoch_tags(
            nwb_epoch_tags=nwb_epoch_tags,
            saved_epoch_tags=ephys_epoch_tags,
        )
        if not (
            len(nwb_epoch_tags)
            == ephys_start_times.size
            == ephys_stop_times.size
        ):
            raise ValueError("Ephys recording intervals do not match the NWB epoch count.")

        ephys_bounds_by_epoch = {
            epoch: (float(start_time), float(stop_time))
            for epoch, start_time, stop_time in zip(
                ephys_epoch_tags,
                ephys_start_times,
                ephys_stop_times,
                strict=True,
            )
        }
        ephys_interval_count = add_ephys_recording_intervals_to_nwb(
            nwbfile=nwbfile,
            epoch_tags=ephys_epoch_tags,
            start_times=ephys_start_times,
            stop_times=ephys_stop_times,
        )
        if add_trajectory_times:
            if trajectory_table is None or trajectory_source_path is None:
                raise RuntimeError("Trajectory export inputs were not loaded.")
            trajectory_interval_count = add_trajectory_intervals_to_nwb(
                nwbfile=nwbfile,
                trajectory_table=trajectory_table,
                ephys_bounds_by_epoch=ephys_bounds_by_epoch,
            )
            trajectory_outputs.update(
                {
                    "trajectory_intervals_table_name": TRAJECTORY_INTERVALS_TABLE_NAME,
                    "trajectory_intervals_source_path": trajectory_source_path,
                    "trajectory_intervals_row_count": trajectory_interval_count,
                }
            )
        if add_ripples:
            if (
                ripple_table is None
                or ripple_source_path is None
                or ripple_provenance is None
                or ripple_provenance_log_path is None
            ):
                raise RuntimeError("Ripple export inputs were not loaded.")
            ripple_interval_count = add_ripples_to_nwb(
                nwbfile=nwbfile,
                ripple_table=ripple_table,
                ephys_bounds_by_epoch=ephys_bounds_by_epoch,
                provenance=ripple_provenance,
            )
            ripple_outputs.update(
                {
                    "ripple_intervals_table_name": RIPPLES_INTERVALS_TABLE_NAME,
                    "ripple_intervals_source_path": ripple_source_path,
                    "ripple_intervals_row_count": ripple_interval_count,
                    "ripple_detection_provenance_scratch_name": (
                        RIPPLE_PROVENANCE_SCRATCH_NAME
                    ),
                    "ripple_detection_run_log_path": ripple_provenance_log_path,
                }
            )
        if add_position:
            if (
                position_table is None
                or position_source_path is None
                or position_producer_provenance is None
            ):
                raise RuntimeError("Position export inputs were not loaded.")
            video_timestamps_by_epoch = get_nwb_video_timestamps(
                nwbfile=nwbfile,
                epoch_tags=nwb_epoch_tags,
            )
            position_epoch_ranges = validate_position_against_nwb_videos(
                position_table=position_table,
                epoch_tags=nwb_epoch_tags,
                video_timestamps_by_epoch=video_timestamps_by_epoch,
            )
            position_provenance = build_position_provenance(
                analysis_path=analysis_path,
                position_path=position_source_path,
                position_table=position_table,
                epoch_ranges=position_epoch_ranges,
                timestamps_reference_time=nwbfile.timestamps_reference_time,
                producer_provenance=position_producer_provenance,
            )
            position_outputs.update(
                {
                    "position_source_path": position_source_path,
                    "position_producer_provenance_status": (
                        position_producer_provenance["status"]
                    ),
                    **add_position_to_nwb(
                        nwbfile=nwbfile,
                        position_table=position_table,
                        epoch_ranges=position_epoch_ranges,
                        provenance=position_provenance,
                        position_offset=position_offset,
                    ),
                }
            )
        if add_wtrack_linearization:
            if wtrack_configurations is None:
                raise RuntimeError("W-track configurations were not built.")
            wtrack_outputs.update(
                add_wtrack_linearization_to_nwb(
                    nwbfile=nwbfile,
                    configurations=wtrack_configurations,
                )
            )
        if add_sorting:
            sorting_outputs.update(
                export_curated_sorting_to_nwb(
                    nwbfile=nwbfile,
                    analysis_path=analysis_path,
                    curation_root=curation_root,
                    ephys_start_times=ephys_start_times,
                    ephys_stop_times=ephys_stop_times,
                )
            )

        compressed_dataset_paths = configure_ephys_compression(nwbfile)
        temp_output_path = _create_temporary_output_path(resolved_output_path)
        try:
            write_nwb_copy(
                read_io=read_io,
                nwbfile=nwbfile,
                output_path=temp_output_path,
            )
            validate_ephys_compression(
                output_path=temp_output_path,
                dataset_paths=compressed_dataset_paths,
            )
            temp_output_path.replace(resolved_output_path)
        finally:
            if temp_output_path.exists():
                temp_output_path.unlink()

    outputs = {
        "source_nwb_path": nwb_path,
        "output_nwb_path": resolved_output_path,
        "epoch_column_names": list(getattr(nwbfile.epochs, "colnames", [])),
        "source_epochs_preserved": True,
        "ephys_recording_intervals_source_path": analysis_path / "timestamps_ephys.npz",
        "ephys_recording_intervals_table_name": EPHYS_INTERVALS_TABLE_NAME,
        "ephys_recording_intervals_row_count": ephys_interval_count,
        "compressed_ephys_data_paths": compressed_dataset_paths,
        "ephys_data_compression": {
            "compression": EPHYS_COMPRESSION,
            "compression_opts": EPHYS_COMPRESSION_LEVEL,
            "shuffle": EPHYS_COMPRESSION_SHUFFLE,
        },
        **trajectory_outputs,
        **ripple_outputs,
        **position_outputs,
        **wtrack_outputs,
        **sorting_outputs,
    }
    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.nwb.export_augmented_nwb",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "output_path": resolved_output_path,
            "overwrite": overwrite,
            "add_trajectory_times": add_trajectory_times,
            "add_ripples": add_ripples,
            "add_position": add_position,
            "position_offset": position_offset,
            "add_wtrack_linearization": add_wtrack_linearization,
            "add_sorting": add_sorting,
            "curation_root": curation_root,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")
    return resolved_output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the augmented NWB export CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Export an augmented NWB copy with saved ephys intervals and optional "
            "trajectory, ripple, position, W-track linearization, and curated "
            "sorting outputs"
        )
    )
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
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path for the augmented NWB copy. Default: a sibling "
            f"*{DEFAULT_OUTPUT_SUFFIX} file."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--add-trajectory-times",
        action="store_true",
        help=(
            "Add poke-defined trajectory intervals from "
            f"{TRAJECTORY_TIMES_FILENAME}."
        ),
    )
    parser.add_argument(
        "--add-ripples",
        action="store_true",
        help=(
            "Add speed-gated ripples from "
            f"{RIPPLE_TIMES_RELATIVE_PATH}."
        ),
    )
    parser.add_argument(
        "--add-position",
        action="store_true",
        help=(
            "Add canonical combined head/body position from "
            f"{POSITION_RELATIVE_PATH}."
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=(
            "Number of leading position samples analyses should ignore within "
            f"each epoch. Samples remain stored. Default: {DEFAULT_POSITION_OFFSET}"
        ),
    )
    parser.add_argument(
        "--add-wtrack-linearization",
        action="store_true",
        help=(
            "Add the animal-specific W-track graph inputs used for position "
            "linearization under /processing/behavior."
        ),
    )
    parser.add_argument(
        "--add-sorting",
        action="store_true",
        help="Add curated consolidated spike sorting output to the NWB units table.",
    )
    parser.add_argument(
        "--curation-root",
        type=Path,
        default=DEFAULT_CURATION_ROOT,
        help=f"Base directory for the local sorting-curations checkout. Default: {DEFAULT_CURATION_ROOT}",
    )
    return parser.parse_args()


def main() -> None:
    """Run the augmented NWB export CLI."""
    args = parse_arguments()
    export_augmented_nwb(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        output_path=args.output_path,
        overwrite=args.overwrite,
        add_trajectory_times=args.add_trajectory_times,
        add_ripples=args.add_ripples,
        add_position=args.add_position,
        position_offset=args.position_offset,
        add_wtrack_linearization=args.add_wtrack_linearization,
        add_sorting=args.add_sorting,
        curation_root=args.curation_root,
    )


if __name__ == "__main__":
    main()
