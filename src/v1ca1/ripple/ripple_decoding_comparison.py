from __future__ import annotations

"""Compare CA1 and V1 ripple decoding content for one session."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_DATA_ROOT, get_analysis_path
from v1ca1.ripple._decoding import (
    build_region_unit_mask_table as build_region_unit_mask_table_from_session,
    get_representation_inputs as get_representation_inputs_from_session,
)
from v1ca1.ripple.ripple_glm import (
    build_epoch_intervals as build_epoch_intervals_from_timestamps,
    load_ripple_tables as load_ripple_tables_for_session,
)
from v1ca1.task_progression._session import (
    prepare_task_progression_session as load_task_progression_session,
)

REPRESENTATIONS = ("place", "task_progression")
REGIONS = ("v1", "ca1")

BIN_SIZE_S = 0.002
CA1_MIN_MOVEMENT_FR_HZ = 0
V1_MIN_MOVEMENT_FR_HZ = 0.5
N_SHUFFLES = 100
SHUFFLE_SEED = 45

NUM_SLEEP_EPOCHS = 5
NUM_RUN_EPOCHS = 4
POSITION_OFFSET = 10
SPEED_THRESHOLD_CM_S = 4.0
SPEED_SIGMA_S = 0.1
PLACE_BIN_SIZE_CM = 4.0

FEATURE_NAME_BY_REPRESENTATION = {
    "place": "linpos",
    "task_progression": "tp",
}
TRAJECTORY_TYPES = (
    "center_to_left",
    "left_to_center",
    "center_to_right",
    "right_to_center",
)
SCORING_SCHEMES = ("continuous", "trajectory", "turn_group", "arm_identity")
CATEGORICAL_SCORING_SCHEMES = ("trajectory", "turn_group", "arm_identity")
TURN_GROUPS = ("left", "right")
ARM_IDENTITY_LABELS = ("other", "left", "right")
TURN_GROUP_BY_TRAJECTORY = {
    "center_to_left": "left",
    "right_to_center": "left",
    "center_to_right": "right",
    "left_to_center": "right",
}
METRIC_LABELS = {
    "pearson_r": "Pearson r",
    "mean_abs_difference": "Mean Absolute Difference",
    "mean_abs_difference_normalized": "Normalized Mean Absolute Difference",
    "mean_signed_difference": "Mean Signed Difference",
    "start_difference": "Start Difference",
    "end_difference": "End Difference",
}
METRIC_DIRECTION = {
    "pearson_r": "higher",
    "mean_abs_difference": "lower",
    "mean_abs_difference_normalized": "lower",
    "mean_signed_difference": "zero",
    "start_difference": "zero",
    "end_difference": "zero",
}

DX = 9.5
DY = 9.0
DIAGONAL_SEGMENT_LENGTH = float(np.sqrt(DX**2 + DY**2))
LONG_SEGMENT_LENGTH = 81.0 - 17.0 - 2.0
SHORT_SEGMENT_LENGTH = 13.5
TOTAL_LENGTH_PER_TRAJECTORY = (
    LONG_SEGMENT_LENGTH * 2.0 + SHORT_SEGMENT_LENGTH + 2.0 * DIAGONAL_SEGMENT_LENGTH
)
# Treat the terminal arm as the final long segment plus half of the preceding diagonal.
ARM_START_WITHIN_TRAJECTORY = (
    TOTAL_LENGTH_PER_TRAJECTORY
    - LONG_SEGMENT_LENGTH
    - 0.5 * DIAGONAL_SEGMENT_LENGTH
)

NODE_POSITIONS_RIGHT = np.array(
    [
        (55.5, 81.0),
        (55.5, 81.0 - LONG_SEGMENT_LENGTH),
        (55.5 - DX, 81.0 - LONG_SEGMENT_LENGTH - DY),
        (55.5 - DX - SHORT_SEGMENT_LENGTH, 81.0 - LONG_SEGMENT_LENGTH - DY),
        (55.5 - 2.0 * DX - SHORT_SEGMENT_LENGTH, 81.0 - LONG_SEGMENT_LENGTH),
        (55.5 - 2.0 * DX - SHORT_SEGMENT_LENGTH, 81.0),
    ]
)
NODE_POSITIONS_LEFT = np.array(
    [
        (55.5, 81.0),
        (55.5, 81.0 - LONG_SEGMENT_LENGTH),
        (55.5 + DX, 81.0 - LONG_SEGMENT_LENGTH - DY),
        (55.5 + DX + SHORT_SEGMENT_LENGTH, 81.0 - LONG_SEGMENT_LENGTH - DY),
        (55.5 + 2.0 * DX + SHORT_SEGMENT_LENGTH, 81.0 - LONG_SEGMENT_LENGTH),
        (55.5 + 2.0 * DX + SHORT_SEGMENT_LENGTH, 81.0),
    ]
)
EDGES_FROM_CENTER = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
EDGES_TO_CENTER = np.array([(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)])
LINEAR_EDGE_ORDER_FROM_CENTER = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
LINEAR_EDGE_ORDER_TO_CENTER = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0)]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CA1 and V1 ripple decoding content for one session."
    )
    parser.add_argument("--animal-name", required=True, help="Animal name")
    parser.add_argument(
        "--date",
        required=True,
        help="Session date in YYYYMMDD format",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--decode-epoch",
        default=None,
        help=(
            "Optional single run epoch to decode. If omitted, decode all run epochs unless "
            "--train-epoch is provided alone."
        ),
    )
    parser.add_argument(
        "--train-epoch",
        default=None,
        help=(
            "Optional run epoch used to build tuning curves. If omitted, the decode epoch "
            "is also used as the training epoch."
        ),
    )
    parser.add_argument(
        "--bin-size-s",
        type=float,
        default=BIN_SIZE_S,
        help=f"Time bin size in seconds for ripple decoding. Default: {BIN_SIZE_S}",
    )
    parser.add_argument(
        "--scoring-schemes",
        nargs="+",
        choices=SCORING_SCHEMES,
        default=list(SCORING_SCHEMES),
        help=(
            "Scoring schemes to compute. "
            f"Default: {' '.join(SCORING_SCHEMES)}"
        ),
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.bin_size_s <= 0:
        raise ValueError("--bin-size-s must be positive.")


def resolve_scoring_schemes(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(SCORING_SCHEMES)
    resolved: list[str] = []
    seen: set[str] = set()
    for scheme in raw_values:
        if scheme not in seen:
            seen.add(scheme)
            resolved.append(scheme)
    return resolved


def get_epoch_list(
    num_sleep_epochs: int, num_run_epochs: int
) -> tuple[list[str], list[str]]:
    """Return alternating sleep/run epoch names matching the local session convention."""
    if abs(num_sleep_epochs - num_run_epochs) != 1:
        raise ValueError("The run and sleep epochs must alternate.")

    sleep_epoch_tags = [f"s{i + 1}" for i in range(num_sleep_epochs)]
    run_epoch_tags = [f"r{i + 1}" for i in range(num_run_epochs)]
    epoch_list: list[str] = []
    if num_sleep_epochs > num_run_epochs:
        for i in range(num_sleep_epochs + num_run_epochs):
            if i % 2 == 0:
                epoch_tag = sleep_epoch_tags[i // 2]
            else:
                epoch_tag = run_epoch_tags[i // 2]
            epoch_list.append(f"{i + 1:02d}_{epoch_tag}")
        run_epoch_list = epoch_list[1::2]
    else:
        for i in range(num_sleep_epochs + num_run_epochs):
            if i % 2 == 0:
                epoch_tag = run_epoch_tags[i // 2]
            else:
                epoch_tag = sleep_epoch_tags[i // 2]
            epoch_list.append(f"{i + 1:02d}_{epoch_tag}")
        run_epoch_list = epoch_list[::2]
    return epoch_list, run_epoch_list


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def _sampling_rate(timestamps: np.ndarray) -> float:
    values = np.asarray(timestamps, dtype=float)
    return (len(values) - 1) / (values[-1] - values[0])


def _empty_intervalset() -> Any:
    import pynapple as nap

    return nap.IntervalSet(
        start=np.array([], dtype=float), end=np.array([], dtype=float)
    )


def _trajectory_interval(values: Any) -> Any:
    import pynapple as nap

    array = np.asarray(values)
    if array.size == 0:
        return _empty_intervalset()
    if array.ndim == 1:
        if array.size < 2:
            return _empty_intervalset()
        return nap.IntervalSet(
            start=[float(array[0])],
            end=[float(array[-1])],
            time_units="s",
        )
    if array.shape[1] < 2:
        return _empty_intervalset()
    return nap.IntervalSet(
        start=np.asarray(array[:, 0], dtype=float),
        end=np.asarray(array[:, -1], dtype=float),
        time_units="s",
    )


def _build_track_graphs() -> tuple[dict[str, Any], dict[str, Any]]:
    import track_linearization as tl

    track_graph_from_center: dict[str, Any] = {}
    track_graph_to_center: dict[str, Any] = {}
    for trajectory_type in TRAJECTORY_TYPES:
        if trajectory_type in {"center_to_right", "right_to_center"}:
            node_positions = NODE_POSITIONS_RIGHT
        else:
            node_positions = NODE_POSITIONS_LEFT
        track_graph_from_center[trajectory_type] = tl.make_track_graph(
            node_positions,
            EDGES_FROM_CENTER,
        )
        track_graph_to_center[trajectory_type] = tl.make_track_graph(
            node_positions,
            EDGES_TO_CENTER,
        )
    return track_graph_from_center, track_graph_to_center


def get_tsgroup(sorting: Any, timestamps_ephys_all: np.ndarray) -> Any:
    import pynapple as nap

    data = {}
    for unit_id in sorting.get_unit_ids():
        data[unit_id] = nap.Ts(
            t=np.asarray(
                timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)], dtype=float
            ),
            time_units="s",
        )
    return nap.TsGroup(data, time_units="s")


def build_epoch_interval(timestamps_position: np.ndarray, position_offset: int) -> Any:
    import pynapple as nap

    values = np.asarray(timestamps_position, dtype=float)
    if values.ndim != 1 or values.size <= position_offset:
        raise ValueError(
            "Position timestamps are empty or shorter than position_offset."
        )
    return nap.IntervalSet(
        start=float(values[position_offset]),
        end=float(values[-1]),
        time_units="s",
    )


def build_speed_tsd(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    position_offset: int,
) -> Any:
    import position_tools as pt
    import pynapple as nap

    epoch_position = np.asarray(position[position_offset:], dtype=float)
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    speed = pt.get_speed(
        position=epoch_position,
        time=epoch_timestamps,
        sampling_frequency=_sampling_rate(epoch_timestamps),
        sigma=SPEED_SIGMA_S,
    )
    return nap.Tsd(t=epoch_timestamps, d=np.asarray(speed, dtype=float), time_units="s")


def build_movement_interval(speed_tsd: Any, speed_threshold_cm_s: float) -> Any:
    return speed_tsd.threshold(float(speed_threshold_cm_s), method="above").time_support


def build_trajectory_intervals(
    trajectory_times: dict[str, Any],
    run_epochs: list[str],
) -> dict[str, dict[str, Any]]:
    trajectory_intervals: dict[str, dict[str, Any]] = {}
    for epoch in run_epochs:
        trajectory_intervals[epoch] = {}
        for trajectory_type in TRAJECTORY_TYPES:
            trajectory_intervals[epoch][trajectory_type] = _trajectory_interval(
                trajectory_times[epoch].get(trajectory_type, [])
            )
    return trajectory_intervals


def build_linear_position(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    position_offset: int,
) -> Any:
    import pynapple as nap
    import track_linearization as tl

    track_graph_from_center, _track_graph_to_center = _build_track_graphs()
    epoch_position = np.asarray(position[position_offset:], dtype=float)
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    for trajectory_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        position_df = tl.get_linearized_position(
            position=epoch_position,
            track_graph=track_graph_from_center[trajectory_type],
            edge_order=LINEAR_EDGE_ORDER_FROM_CENTER,
            edge_spacing=0,
        )
        linear_position = nap.Tsd(
            t=epoch_timestamps,
            d=np.asarray(position_df["linear_position"], dtype=float)
            + TOTAL_LENGTH_PER_TRAJECTORY * trajectory_index,
            time_support=trajectory_intervals[trajectory_type],
            time_units="s",
        )
        time_chunks.append(np.asarray(linear_position.t, dtype=float))
        value_chunks.append(np.asarray(linear_position.d, dtype=float))

    all_times = np.concatenate(time_chunks)
    all_values = np.concatenate(value_chunks)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=movement_interval,
        time_units="s",
    )


def build_task_progression(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    trajectory_intervals: dict[str, Any],
    movement_interval: Any,
    position_offset: int,
) -> Any:
    import pynapple as nap
    import track_linearization as tl

    track_graph_from_center, track_graph_to_center = _build_track_graphs()
    epoch_position = np.asarray(position[position_offset:], dtype=float)
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    time_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []

    for trajectory_type in TRAJECTORY_TYPES:
        if trajectory_type in {"center_to_left", "center_to_right"}:
            track_graph = track_graph_from_center[trajectory_type]
            edge_order = LINEAR_EDGE_ORDER_FROM_CENTER
        else:
            track_graph = track_graph_to_center[trajectory_type]
            edge_order = LINEAR_EDGE_ORDER_TO_CENTER

        position_df = tl.get_linearized_position(
            position=epoch_position,
            track_graph=track_graph,
            edge_order=edge_order,
            edge_spacing=0,
        )
        offset = (
            0.0 if trajectory_type in {"center_to_left", "right_to_center"} else 1.0
        )
        task_progression = nap.Tsd(
            t=epoch_timestamps,
            d=np.asarray(position_df["linear_position"], dtype=float)
            / TOTAL_LENGTH_PER_TRAJECTORY
            + offset,
            time_support=trajectory_intervals[trajectory_type],
            time_units="s",
        )
        time_chunks.append(np.asarray(task_progression.t, dtype=float))
        value_chunks.append(np.asarray(task_progression.d, dtype=float))

    all_times = np.concatenate(time_chunks)
    all_values = np.concatenate(value_chunks)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=movement_interval,
        time_units="s",
    )


def build_linear_position_bins(
    place_bin_size_cm: float = PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    return np.arange(
        0.0,
        TOTAL_LENGTH_PER_TRAJECTORY * len(TRAJECTORY_TYPES) + place_bin_size_cm,
        place_bin_size_cm,
    )


def build_combined_task_progression_bins(
    place_bin_size_cm: float = PLACE_BIN_SIZE_CM,
) -> np.ndarray:
    task_progression_bin_size = place_bin_size_cm / TOTAL_LENGTH_PER_TRAJECTORY
    return np.arange(0.0, 2.0 + task_progression_bin_size, task_progression_bin_size)


def compute_movement_firing_rates(
    spikes_by_region: dict[str, Any],
    movement_by_run: dict[str, Any],
    run_epochs: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    movement_rates: dict[str, dict[str, np.ndarray]] = {}
    for region, spikes in spikes_by_region.items():
        movement_rates[region] = {}
        for epoch in run_epochs:
            movement_interval = movement_by_run[epoch]
            duration = float(movement_interval.tot_length())
            counts = np.asarray(
                spikes.count(ep=movement_interval).to_numpy(), dtype=float
            )
            if counts.ndim == 1:
                counts = counts[np.newaxis, :]
            total_counts = np.sum(counts, axis=0).ravel()
            if duration <= 0:
                movement_rates[region][epoch] = np.zeros(
                    total_counts.shape, dtype=float
                )
            else:
                movement_rates[region][epoch] = total_counts / duration
    return movement_rates


def prepare_task_progression_session(analysis_path: Path) -> dict[str, Any]:
    import spikeinterface.full as si

    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")

    epoch_list, run_epoch_list = get_epoch_list(
        num_sleep_epochs=NUM_SLEEP_EPOCHS,
        num_run_epochs=NUM_RUN_EPOCHS,
    )

    timestamps_ephys_by_epoch = _load_pickle(analysis_path / "timestamps_ephys.pkl")
    timestamps_position = _load_pickle(analysis_path / "timestamps_position.pkl")
    timestamps_ephys_all = _load_pickle(analysis_path / "timestamps_ephys_all.pkl")
    position_by_epoch = _load_pickle(analysis_path / "position.pkl")
    trajectory_times = _load_pickle(analysis_path / "trajectory_times.pkl")

    sorting_by_region = {
        region: si.load(analysis_path / f"sorting_{region}") for region in REGIONS
    }
    spikes_by_region = {
        region: get_tsgroup(sorting, np.asarray(timestamps_ephys_all, dtype=float))
        for region, sorting in sorting_by_region.items()
    }

    trajectory_intervals = build_trajectory_intervals(trajectory_times, run_epoch_list)
    all_epoch_by_run: dict[str, Any] = {}
    speed_by_run: dict[str, Any] = {}
    movement_by_run: dict[str, Any] = {}
    linear_position_by_run: dict[str, Any] = {}
    task_progression_by_run: dict[str, Any] = {}

    for epoch in run_epoch_list:
        all_epoch_by_run[epoch] = build_epoch_interval(
            timestamps_position[epoch],
            position_offset=POSITION_OFFSET,
        )
        speed_by_run[epoch] = build_speed_tsd(
            position=position_by_epoch[epoch],
            timestamps_position=timestamps_position[epoch],
            position_offset=POSITION_OFFSET,
        )
        movement_by_run[epoch] = build_movement_interval(
            speed_tsd=speed_by_run[epoch],
            speed_threshold_cm_s=SPEED_THRESHOLD_CM_S,
        )
        linear_position_by_run[epoch] = build_linear_position(
            position=position_by_epoch[epoch],
            timestamps_position=timestamps_position[epoch],
            trajectory_intervals=trajectory_intervals[epoch],
            movement_interval=movement_by_run[epoch],
            position_offset=POSITION_OFFSET,
        )
        task_progression_by_run[epoch] = build_task_progression(
            position=position_by_epoch[epoch],
            timestamps_position=timestamps_position[epoch],
            trajectory_intervals=trajectory_intervals[epoch],
            movement_interval=movement_by_run[epoch],
            position_offset=POSITION_OFFSET,
        )

    return {
        "analysis_path": analysis_path,
        "epoch_tags": list(epoch_list),
        "run_epochs": list(run_epoch_list),
        "timestamps_ephys_by_epoch": timestamps_ephys_by_epoch,
        "timestamps_position": timestamps_position,
        "timestamps_ephys_all": np.asarray(timestamps_ephys_all, dtype=float),
        "position_by_epoch": position_by_epoch,
        "spikes_by_region": spikes_by_region,
        "trajectory_intervals": trajectory_intervals,
        "all_epoch_by_run": all_epoch_by_run,
        "speed_by_run": speed_by_run,
        "movement_by_run": movement_by_run,
        "linear_position_by_run": linear_position_by_run,
        "task_progression_by_run": task_progression_by_run,
        "sources": {
            "timestamps_ephys": "pickle",
            "timestamps_position": "pickle",
            "timestamps_ephys_all": "pickle",
            "position": "pickle",
            "trajectory_intervals": "trajectory_times.pkl",
            "sorting": "spikeinterface",
            "ripple_events": "pickle",
        },
    }


def get_representation_inputs(
    session: dict[str, Any],
    representation: str,
) -> tuple[dict[str, Any], np.ndarray, str]:
    if representation == "place":
        return (
            session["linear_position_by_run"],
            build_linear_position_bins(),
            FEATURE_NAME_BY_REPRESENTATION[representation],
        )
    if representation == "task_progression":
        return (
            session["task_progression_by_run"],
            build_combined_task_progression_bins(),
            FEATURE_NAME_BY_REPRESENTATION[representation],
        )
    raise ValueError(f"Unsupported representation {representation!r}.")


def validate_run_epoch(
    run_epochs: list[str],
    epoch: str | None,
    flag_name: str,
) -> str | None:
    if epoch is None:
        return None
    if epoch not in run_epochs:
        raise ValueError(f"{flag_name} must be one of {run_epochs!r}. Got {epoch!r}.")
    return str(epoch)


def resolve_epoch_pairs(
    run_epochs: list[str],
    *,
    requested_decode_epoch: str | None,
    requested_train_epoch: str | None,
) -> list[tuple[str, str]]:
    decode_epoch = validate_run_epoch(run_epochs, requested_decode_epoch, "--decode-epoch")
    train_epoch = validate_run_epoch(run_epochs, requested_train_epoch, "--train-epoch")

    if decode_epoch is not None and train_epoch is not None:
        return [(decode_epoch, train_epoch)]
    if decode_epoch is not None:
        return [(decode_epoch, decode_epoch)]
    if train_epoch is not None:
        return [(train_epoch, train_epoch)]
    return [(epoch, epoch) for epoch in run_epochs]


def normalize_ripple_table(result: Any, epoch: str) -> pd.DataFrame:
    if result is None or (isinstance(result, dict) and not result):
        dataframe = pd.DataFrame(columns=["start_time", "end_time"])
    elif isinstance(result, pd.DataFrame):
        dataframe = result.copy()
    elif hasattr(result, "to_dataframe"):
        dataframe = result.to_dataframe()
    else:
        dataframe = pd.DataFrame(result)

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            f"Ripple event payload for epoch {epoch!r} could not be normalized to a DataFrame."
        )

    dataframe = dataframe.reset_index(drop=True)
    if dataframe.empty:
        for column_name in ("start_time", "end_time"):
            if column_name not in dataframe.columns:
                dataframe[column_name] = pd.Series(dtype=float)

    missing_columns = [
        column_name
        for column_name in ("start_time", "end_time")
        if column_name not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Ripple events for epoch {epoch!r} are missing required columns: {missing_columns!r}."
        )

    dataframe["start_time"] = np.asarray(dataframe["start_time"], dtype=float)
    dataframe["end_time"] = np.asarray(dataframe["end_time"], dtype=float)
    return dataframe


def load_ripple_tables(analysis_path: Path) -> tuple[dict[str, pd.DataFrame], str]:
    ripple_path = analysis_path / "ripple" / "Kay_ripple_detector.pkl"
    if not ripple_path.exists():
        raise FileNotFoundError(
            f"Legacy ripple detector pickle not found: {ripple_path}"
        )

    loaded = _load_pickle(ripple_path)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Legacy ripple detector pickle is not a dictionary: {ripple_path}"
        )

    return (
        {
            str(epoch): normalize_ripple_table(table, epoch=str(epoch))
            for epoch, table in loaded.items()
        },
        "pickle",
    )


def empty_ripple_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": pd.Series(dtype=float),
            "end_time": pd.Series(dtype=float),
        }
    )


def compute_tuning_curves_for_epoch(
    *,
    spikes: Any,
    feature: Any,
    movement_interval: Any,
    bin_edges: np.ndarray,
    feature_name: str,
) -> Any:
    import pynapple as nap

    return nap.compute_tuning_curves(
        data=spikes,
        features=feature,
        bins=[np.asarray(bin_edges, dtype=float)],
        epochs=movement_interval,
        feature_names=[feature_name],
    )


def build_epoch_intervals(timestamps_by_epoch: dict[str, np.ndarray]) -> dict[str, Any]:
    import pynapple as nap

    epoch_intervals: dict[str, Any] = {}
    for epoch, timestamps in timestamps_by_epoch.items():
        epoch_timestamps = np.asarray(timestamps, dtype=float)
        if epoch_timestamps.ndim != 1 or epoch_timestamps.size == 0:
            raise ValueError(
                f"Ephys timestamps for epoch {epoch!r} are empty or malformed."
            )
        epoch_intervals[epoch] = nap.IntervalSet(
            start=float(epoch_timestamps[0]),
            end=float(epoch_timestamps[-1]),
            time_units="s",
        )
    return epoch_intervals


def make_intervalset_from_bounds(starts: np.ndarray, ends: np.ndarray) -> Any:
    import pynapple as nap

    return nap.IntervalSet(
        start=np.asarray(starts, dtype=float),
        end=np.asarray(ends, dtype=float),
        time_units="s",
    )


def make_empty_tsd(time_support: Any | None = None) -> Any:
    import pynapple as nap

    kwargs: dict[str, Any] = {"time_units": "s"}
    if time_support is not None:
        kwargs["time_support"] = time_support
    return nap.Tsd(
        t=np.array([], dtype=float),
        d=np.array([], dtype=float),
        **kwargs,
    )


def require_xarray():
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "This script requires `xarray` to save ripple comparison outputs as NetCDF."
        ) from exc
    return xr


def extract_time_values(tsd_like: Any) -> np.ndarray:
    if hasattr(tsd_like, "t"):
        return np.asarray(tsd_like.t, dtype=float)
    if hasattr(tsd_like, "index"):
        index = getattr(tsd_like, "index")
        if hasattr(index, "values"):
            return np.asarray(index.values, dtype=float)
    raise ValueError("Could not extract time values from the provided pynapple object.")


def concatenate_tsds(tsds: list[Any], time_support: Any) -> Any:
    import pynapple as nap

    if not tsds:
        return make_empty_tsd(time_support=time_support)

    times = [
        np.asarray(tsd.t, dtype=float) for tsd in tsds if len(np.asarray(tsd.t)) > 0
    ]
    values = [
        np.asarray(tsd.d, dtype=float) for tsd in tsds if len(np.asarray(tsd.t)) > 0
    ]
    if not times:
        return make_empty_tsd(time_support=time_support)

    all_times = np.concatenate(times)
    all_values = np.concatenate(values)
    order = np.argsort(all_times)
    return nap.Tsd(
        t=all_times[order],
        d=all_values[order],
        time_support=time_support,
        time_units="s",
    )


def interpolate_nans_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return values.copy()

    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, np.nan, dtype=float)
    if np.all(finite):
        return values.copy()

    indices = np.arange(values.size, dtype=float)
    output = values.copy()
    output[~finite] = np.interp(indices[~finite], indices[finite], values[finite])
    return output


def deranged_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 1:
        return np.arange(size, dtype=int)
    for _ in range(128):
        permutation = rng.permutation(size)
        if not np.any(permutation == np.arange(size, dtype=int)):
            return permutation
    return np.roll(np.arange(size, dtype=int), 1)


def assemble_decoded_ripple_epoch_data(
    *,
    spikes: Any,
    tuning_curves: Any,
    ripple_table: pd.DataFrame,
    epoch_interval: Any,
    bin_size_s: float,
) -> dict[str, Any]:
    import pynapple as nap

    decoded_chunks: list[Any] = []
    ripple_starts: list[float] = []
    ripple_ends: list[float] = []
    ripple_source_indices: list[int] = []
    decoded_state_chunks: list[np.ndarray] = []
    bin_time_chunks: list[np.ndarray] = []
    ripple_id_chunks: list[np.ndarray] = []
    skipped_ripples: list[dict[str, Any]] = []
    kept_ripple_count = 0

    for ripple_row_index, ripple_row in ripple_table.reset_index(drop=True).iterrows():
        ripple_ep = make_intervalset_from_bounds(
            np.array([float(ripple_row["start_time"])], dtype=float),
            np.array([float(ripple_row["end_time"])], dtype=float),
        ).intersect(epoch_interval)
        if float(ripple_ep.tot_length()) <= 0.0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "No overlap with decode epoch interval.",
                }
            )
            continue

        decoded, _ = nap.decode_bayes(
            tuning_curves=tuning_curves,
            data=spikes,
            epochs=ripple_ep,
            sliding_window_size=None,
            bin_size=float(bin_size_s),
        )
        decoded_times = extract_time_values(decoded).reshape(-1)
        if decoded_times.size == 0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "Decoding returned no time bins.",
                }
            )
            continue

        decoded_state = interpolate_nans_1d(
            np.asarray(decoded.d, dtype=float).reshape(-1)
        )
        if decoded_state.shape[0] != decoded_times.size:
            raise ValueError("Decoded state values do not match decoded timestamps.")
        if not np.any(np.isfinite(decoded_state)):
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_row_index),
                    "reason": "Decoded state was non-finite for all ripple bins.",
                }
            )
            continue

        decoded_chunks.append(decoded)
        decoded_state_chunks.append(decoded_state)
        bin_time_chunks.append(decoded_times)
        ripple_id_chunks.append(
            np.full(decoded_state.shape[0], kept_ripple_count, dtype=int)
        )
        ripple_starts.append(float(np.asarray(ripple_ep.start, dtype=float)[0]))
        ripple_ends.append(float(np.asarray(ripple_ep.end, dtype=float)[0]))
        ripple_source_indices.append(int(ripple_row_index))
        kept_ripple_count += 1

    ripple_support = make_intervalset_from_bounds(
        np.asarray(ripple_starts, dtype=float),
        np.asarray(ripple_ends, dtype=float),
    )
    decoded_tsd = concatenate_tsds(decoded_chunks, ripple_support)
    if not decoded_state_chunks:
        return {
            "decoded_tsd": decoded_tsd,
            "decoded_state": np.array([], dtype=float),
            "bin_times_s": np.array([], dtype=float),
            "ripple_ids": np.array([], dtype=int),
            "n_ripples_kept": 0,
            "n_bins": 0,
            "ripple_start_times_s": np.array([], dtype=float),
            "ripple_end_times_s": np.array([], dtype=float),
            "ripple_source_indices": np.array([], dtype=int),
            "skipped_ripples": skipped_ripples,
        }

    return {
        "decoded_tsd": decoded_tsd,
        "decoded_state": np.concatenate(decoded_state_chunks).astype(float, copy=False),
        "bin_times_s": np.concatenate(bin_time_chunks).astype(float, copy=False),
        "ripple_ids": np.concatenate(ripple_id_chunks).astype(int, copy=False),
        "n_ripples_kept": int(kept_ripple_count),
        "n_bins": int(sum(chunk.shape[0] for chunk in decoded_state_chunks)),
        "ripple_start_times_s": np.asarray(ripple_starts, dtype=float),
        "ripple_end_times_s": np.asarray(ripple_ends, dtype=float),
        "ripple_source_indices": np.asarray(ripple_source_indices, dtype=int),
        "skipped_ripples": skipped_ripples,
    }


def _subset_spikes(spikes: Any, unit_ids: list[Any]) -> Any:
    import pynapple as nap

    return nap.TsGroup(
        {unit_id: spikes[unit_id] for unit_id in unit_ids}, time_units="s"
    )


def build_region_unit_mask_table(
    *,
    unit_ids: np.ndarray,
    movement_firing_rates_hz: np.ndarray,
    min_movement_fr_hz: float,
    region: str,
) -> pd.DataFrame:
    """Backward-compatible wrapper around the shared ripple mask helper."""
    return build_region_unit_mask_table_from_session(
        unit_ids=unit_ids,
        movement_firing_rates_hz=movement_firing_rates_hz,
        min_movement_fr_hz=min_movement_fr_hz,
        region=region,
    )


def _build_ripple_lookup(decoded_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    ripple_ids = np.asarray(decoded_data["ripple_ids"], dtype=int)
    for local_ripple_id, ripple_source_index in enumerate(
        np.asarray(decoded_data["ripple_source_indices"], dtype=int)
    ):
        mask = ripple_ids == int(local_ripple_id)
        lookup[int(ripple_source_index)] = {
            "ripple_id": int(local_ripple_id),
            "state": np.asarray(decoded_data["decoded_state"], dtype=float)[mask],
            "times": np.asarray(decoded_data["bin_times_s"], dtype=float)[mask],
            "start_time_s": float(
                decoded_data["ripple_start_times_s"][local_ripple_id]
            ),
            "end_time_s": float(decoded_data["ripple_end_times_s"][local_ripple_id]),
        }
    return lookup


def align_decoded_ripple_data(
    ca1_decoded: dict[str, Any],
    v1_decoded: dict[str, Any],
) -> dict[str, Any]:
    ca1_lookup = _build_ripple_lookup(ca1_decoded)
    v1_lookup = _build_ripple_lookup(v1_decoded)
    common_ripple_source_indices = sorted(set(ca1_lookup) & set(v1_lookup))

    ca1_state_chunks: list[np.ndarray] = []
    v1_state_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    ripple_id_chunks: list[np.ndarray] = []
    ripple_source_indices: list[int] = []
    ripple_start_times_s: list[float] = []
    ripple_end_times_s: list[float] = []
    skipped_ripples: list[dict[str, Any]] = []
    kept_ripple_count = 0

    for ripple_source_index in common_ripple_source_indices:
        ca1_block = ca1_lookup[ripple_source_index]
        v1_block = v1_lookup[ripple_source_index]
        ca1_times = np.asarray(ca1_block["times"], dtype=float)
        v1_times = np.asarray(v1_block["times"], dtype=float)
        ca1_times_rounded = np.round(ca1_times, decimals=9)
        v1_times_rounded = np.round(v1_times, decimals=9)
        common_times, ca1_indices, v1_indices = np.intersect1d(
            ca1_times_rounded,
            v1_times_rounded,
            assume_unique=False,
            return_indices=True,
        )
        if common_times.size == 0:
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": "CA1 and V1 decoded bins did not overlap.",
                }
            )
            continue

        ca1_state = np.asarray(ca1_block["state"], dtype=float)[ca1_indices]
        v1_state = np.asarray(v1_block["state"], dtype=float)[v1_indices]
        valid = np.isfinite(ca1_state) & np.isfinite(v1_state)
        if not np.any(valid):
            skipped_ripples.append(
                {
                    "ripple_index": int(ripple_source_index),
                    "reason": "Aligned CA1 and V1 bins were non-finite.",
                }
            )
            continue

        ca1_state = ca1_state[valid]
        v1_state = v1_state[valid]
        aligned_times = np.asarray(ca1_times[ca1_indices], dtype=float)[valid]
        ca1_state_chunks.append(ca1_state)
        v1_state_chunks.append(v1_state)
        time_chunks.append(aligned_times)
        ripple_id_chunks.append(
            np.full(ca1_state.shape[0], kept_ripple_count, dtype=int)
        )
        ripple_source_indices.append(int(ripple_source_index))
        ripple_start_times_s.append(
            float(max(ca1_block["start_time_s"], v1_block["start_time_s"]))
        )
        ripple_end_times_s.append(
            float(min(ca1_block["end_time_s"], v1_block["end_time_s"]))
        )
        kept_ripple_count += 1

    if not ca1_state_chunks:
        return {
            "ca1_decoded_state": np.array([], dtype=float),
            "v1_decoded_state": np.array([], dtype=float),
            "bin_times_s": np.array([], dtype=float),
            "ripple_ids": np.array([], dtype=int),
            "n_ripples": 0,
            "n_bins": 0,
            "ripple_source_indices": np.array([], dtype=int),
            "ripple_start_times_s": np.array([], dtype=float),
            "ripple_end_times_s": np.array([], dtype=float),
            "skipped_ripples": skipped_ripples,
        }

    return {
        "ca1_decoded_state": np.concatenate(ca1_state_chunks).astype(float, copy=False),
        "v1_decoded_state": np.concatenate(v1_state_chunks).astype(float, copy=False),
        "bin_times_s": np.concatenate(time_chunks).astype(float, copy=False),
        "ripple_ids": np.concatenate(ripple_id_chunks).astype(int, copy=False),
        "n_ripples": int(kept_ripple_count),
        "n_bins": int(sum(chunk.shape[0] for chunk in ca1_state_chunks)),
        "ripple_source_indices": np.asarray(ripple_source_indices, dtype=int),
        "ripple_start_times_s": np.asarray(ripple_start_times_s, dtype=float),
        "ripple_end_times_s": np.asarray(ripple_end_times_s, dtype=float),
        "skipped_ripples": skipped_ripples,
    }


def get_scoring_scheme_availability(representation: str) -> dict[str, dict[str, Any]]:
    availability: dict[str, dict[str, Any]] = {
        "continuous": {"applicable": True, "reason": "ok"},
        "turn_group": {"applicable": True, "reason": "ok"},
    }
    if representation == "place":
        availability["trajectory"] = {"applicable": True, "reason": "ok"}
        availability["arm_identity"] = {"applicable": True, "reason": "ok"}
    else:
        availability["trajectory"] = {
            "applicable": False,
            "reason": (
                "The current task_progression representation collapses the four trajectories "
                "into two turn groups, so exact trajectory identity is not recoverable."
            ),
        }
        availability["arm_identity"] = {
            "applicable": False,
            "reason": (
                "The current task_progression representation does not preserve within-trajectory "
                "position, so arm occupancy is not recoverable."
            ),
        }
    return availability


def _empty_label_array(size: int) -> np.ndarray:
    return np.full(int(size), -1, dtype=int)


def map_place_state_to_trajectory_label(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    labels = _empty_label_array(state.size)
    if state.size == 0:
        return labels

    lower = 0.0
    upper = TOTAL_LENGTH_PER_TRAJECTORY * len(TRAJECTORY_TYPES)
    valid = np.isfinite(state) & (state >= lower) & (state <= upper)
    if not np.any(valid):
        return labels

    clipped = np.clip(
        state[valid],
        lower,
        np.nextafter(upper, lower),
    )
    labels[valid] = np.floor(clipped / TOTAL_LENGTH_PER_TRAJECTORY).astype(int)
    labels[valid] = np.clip(labels[valid], 0, len(TRAJECTORY_TYPES) - 1)
    return labels


def map_trajectory_label_to_turn_group(trajectory_labels: np.ndarray) -> np.ndarray:
    trajectory_labels = np.asarray(trajectory_labels, dtype=int).reshape(-1)
    turn_group_labels = _empty_label_array(trajectory_labels.size)
    for trajectory_index, trajectory_name in enumerate(TRAJECTORY_TYPES):
        turn_group_name = TURN_GROUP_BY_TRAJECTORY[trajectory_name]
        turn_group_index = TURN_GROUPS.index(turn_group_name)
        turn_group_labels[trajectory_labels == trajectory_index] = turn_group_index
    return turn_group_labels


def map_task_progression_state_to_turn_group_label(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    labels = _empty_label_array(state.size)
    if state.size == 0:
        return labels

    valid = np.isfinite(state) & (state >= 0.0) & (state <= 2.0)
    if not np.any(valid):
        return labels

    clipped = np.clip(state[valid], 0.0, np.nextafter(2.0, 0.0))
    labels[valid] = (clipped >= 1.0).astype(int)
    return labels


def map_place_state_to_arm_identity_label(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=float).reshape(-1)
    labels = _empty_label_array(state.size)
    if state.size == 0:
        return labels

    lower = 0.0
    upper = TOTAL_LENGTH_PER_TRAJECTORY * len(TRAJECTORY_TYPES)
    valid = np.isfinite(state) & (state >= lower) & (state <= upper)
    if not np.any(valid):
        return labels

    clipped = np.clip(
        state[valid],
        lower,
        np.nextafter(upper, lower),
    )
    trajectory_labels = np.floor(clipped / TOTAL_LENGTH_PER_TRAJECTORY).astype(int)
    trajectory_labels = np.clip(trajectory_labels, 0, len(TRAJECTORY_TYPES) - 1)
    local_position = clipped - trajectory_labels * TOTAL_LENGTH_PER_TRAJECTORY

    labels_valid = np.zeros(clipped.shape, dtype=int)
    in_arm = local_position >= ARM_START_WITHIN_TRAJECTORY
    if np.any(in_arm):
        labels_valid[in_arm] = (
            map_trajectory_label_to_turn_group(trajectory_labels[in_arm]) + 1
        )
    labels[valid] = labels_valid
    return labels


def get_label_names_for_scheme(scheme: str) -> tuple[str, ...]:
    if scheme == "trajectory":
        return tuple(TRAJECTORY_TYPES)
    if scheme == "turn_group":
        return tuple(TURN_GROUPS)
    if scheme == "arm_identity":
        return tuple(ARM_IDENTITY_LABELS)
    raise ValueError(f"Unsupported categorical scoring scheme {scheme!r}.")


def get_state_labels_for_scheme(
    *,
    state: np.ndarray,
    representation: str,
    scheme: str,
) -> np.ndarray:
    if scheme == "trajectory":
        if representation != "place":
            raise ValueError(
                "Trajectory labels are not defined for the current task_progression representation."
            )
        return map_place_state_to_trajectory_label(state)
    if scheme == "turn_group":
        if representation == "place":
            return map_trajectory_label_to_turn_group(map_place_state_to_trajectory_label(state))
        if representation == "task_progression":
            return map_task_progression_state_to_turn_group_label(state)
        raise ValueError(f"Unsupported representation {representation!r}.")
    if scheme == "arm_identity":
        if representation != "place":
            raise ValueError(
                "Arm identity labels are not defined for the current task_progression representation."
            )
        return map_place_state_to_arm_identity_label(state)
    raise ValueError(f"Unsupported categorical scoring scheme {scheme!r}.")


def build_categorical_scheme_bin_data(
    *,
    aligned_data: dict[str, Any],
    representation: str,
    scheme: str,
    v1_state_override: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ca1_state = np.asarray(aligned_data["ca1_decoded_state"], dtype=float)
    if v1_state_override is None:
        v1_state = np.asarray(aligned_data["v1_decoded_state"], dtype=float)
    else:
        v1_state = np.asarray(v1_state_override, dtype=float)

    ca1_labels = get_state_labels_for_scheme(
        state=ca1_state,
        representation=representation,
        scheme=scheme,
    )
    v1_labels = get_state_labels_for_scheme(
        state=v1_state,
        representation=representation,
        scheme=scheme,
    )
    valid = (ca1_labels >= 0) & (v1_labels >= 0)
    match = np.zeros(valid.shape, dtype=bool)
    match[valid] = ca1_labels[valid] == v1_labels[valid]
    return {
        "ca1_labels": ca1_labels,
        "v1_labels": v1_labels,
        "valid": valid,
        "match": match,
    }


def compute_per_ripple_categorical_metrics(
    *,
    ca1_labels: np.ndarray,
    v1_labels: np.ndarray,
) -> dict[str, float | int]:
    ca1_labels = np.asarray(ca1_labels, dtype=int).reshape(-1)
    v1_labels = np.asarray(v1_labels, dtype=int).reshape(-1)
    if ca1_labels.shape != v1_labels.shape:
        raise ValueError("CA1 and V1 categorical labels must share the same shape.")

    valid = (ca1_labels >= 0) & (v1_labels >= 0)
    n_valid_labeled_bins = int(np.sum(valid))
    if n_valid_labeled_bins == 0:
        return {
            "match_rate": np.nan,
            "n_matching_bins": 0,
            "n_valid_labeled_bins": 0,
        }

    n_matching_bins = int(np.sum(ca1_labels[valid] == v1_labels[valid]))
    return {
        "match_rate": float(n_matching_bins / n_valid_labeled_bins),
        "n_matching_bins": n_matching_bins,
        "n_valid_labeled_bins": n_valid_labeled_bins,
    }


def compute_per_ripple_metrics(
    *,
    ca1_state: np.ndarray,
    v1_state: np.ndarray,
    state_span: float,
) -> dict[str, float | int]:
    ca1_state = np.asarray(ca1_state, dtype=float).reshape(-1)
    v1_state = np.asarray(v1_state, dtype=float).reshape(-1)
    if ca1_state.shape != v1_state.shape:
        raise ValueError("CA1 and V1 ripple states must share the same shape.")

    diff = v1_state - ca1_state
    mean_abs_difference = float(np.mean(np.abs(diff))) if diff.size else np.nan
    if diff.size >= 2 and np.nanstd(ca1_state) > 0.0 and np.nanstd(v1_state) > 0.0:
        pearson_r = float(np.corrcoef(ca1_state, v1_state)[0, 1])
    else:
        pearson_r = np.nan

    if state_span > 0:
        mean_abs_difference_normalized = mean_abs_difference / float(state_span)
    else:
        mean_abs_difference_normalized = np.nan

    return {
        "pearson_r": pearson_r,
        "mean_abs_difference": mean_abs_difference,
        "mean_abs_difference_normalized": float(mean_abs_difference_normalized),
        "mean_signed_difference": float(np.mean(diff)) if diff.size else np.nan,
        "start_difference": float(diff[0]) if diff.size else np.nan,
        "end_difference": float(diff[-1]) if diff.size else np.nan,
        "n_bins": int(diff.size),
    }


def build_per_ripple_metric_table(
    *,
    aligned_data: dict[str, Any],
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    state_span: float,
    scoring_schemes: list[str],
    scheme_availability: dict[str, dict[str, Any]],
    v1_state_override: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    ca1_state_all = np.asarray(aligned_data["ca1_decoded_state"], dtype=float)
    if v1_state_override is None:
        v1_state_all = np.asarray(aligned_data["v1_decoded_state"], dtype=float)
    else:
        v1_state_all = np.asarray(v1_state_override, dtype=float)
    ripple_ids = np.asarray(aligned_data["ripple_ids"], dtype=int)
    categorical_bin_data: dict[str, dict[str, np.ndarray]] = {}
    for scheme in scoring_schemes:
        if scheme in CATEGORICAL_SCORING_SCHEMES and scheme_availability[scheme]["applicable"]:
            categorical_bin_data[scheme] = build_categorical_scheme_bin_data(
                aligned_data=aligned_data,
                representation=representation,
                scheme=scheme,
                v1_state_override=v1_state_override,
            )

    rows: list[dict[str, Any]] = []
    for ripple_id, ripple_source_index in enumerate(
        np.asarray(aligned_data["ripple_source_indices"], dtype=int)
    ):
        mask = ripple_ids == int(ripple_id)
        row = {
            "representation": str(representation),
            "train_epoch": str(train_epoch),
            "decode_epoch": str(decode_epoch),
            "ripple_id": int(ripple_id),
            "ripple_source_index": int(ripple_source_index),
            "ripple_start_time_s": float(aligned_data["ripple_start_times_s"][ripple_id]),
            "ripple_end_time_s": float(aligned_data["ripple_end_times_s"][ripple_id]),
        }
        if "continuous" in scoring_schemes:
            row.update(
                compute_per_ripple_metrics(
                    ca1_state=ca1_state_all[mask],
                    v1_state=v1_state_all[mask],
                    state_span=state_span,
                )
            )

        for scheme in CATEGORICAL_SCORING_SCHEMES:
            if scheme not in scoring_schemes:
                continue
            row[f"{scheme}_scheme_requested"] = True
            row[f"{scheme}_scheme_applicable"] = bool(scheme_availability[scheme]["applicable"])
            if scheme_availability[scheme]["applicable"]:
                metrics = compute_per_ripple_categorical_metrics(
                    ca1_labels=categorical_bin_data[scheme]["ca1_labels"][mask],
                    v1_labels=categorical_bin_data[scheme]["v1_labels"][mask],
                )
                row[f"{scheme}_match_rate"] = metrics["match_rate"]
                row[f"{scheme}_n_matching_bins"] = metrics["n_matching_bins"]
                row[f"{scheme}_n_valid_labeled_bins"] = metrics["n_valid_labeled_bins"]
            else:
                row[f"{scheme}_match_rate"] = np.nan
                row[f"{scheme}_n_matching_bins"] = 0
                row[f"{scheme}_n_valid_labeled_bins"] = 0
        rows.append(row)
    return pd.DataFrame(rows), categorical_bin_data


def shuffle_ripple_state_blocks_by_length(
    state: np.ndarray,
    ripple_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    state = np.asarray(state, dtype=float).reshape(-1)
    ripple_ids = np.asarray(ripple_ids, dtype=int).reshape(-1)
    if state.shape[0] != ripple_ids.size:
        raise ValueError("State rows must match ripple_ids length.")
    if state.size == 0:
        return state.copy(), False

    unique_ripple_ids = np.unique(ripple_ids)
    blocks = [state[ripple_ids == ripple_id] for ripple_id in unique_ripple_ids]
    block_indices_by_length: dict[int, list[int]] = {}
    for block_index, block in enumerate(blocks):
        block_indices_by_length.setdefault(int(block.size), []).append(block_index)

    shuffled_blocks = list(blocks)
    any_changed = False
    for block_indices in block_indices_by_length.values():
        if len(block_indices) < 2:
            continue
        permutation = deranged_permutation(len(block_indices), rng)
        if not np.array_equal(permutation, np.arange(len(block_indices), dtype=int)):
            any_changed = True
        for target_index, source_offset in zip(
            block_indices, permutation, strict=False
        ):
            shuffled_blocks[target_index] = blocks[block_indices[source_offset]]

    return np.concatenate(shuffled_blocks).astype(float, copy=False), any_changed


def summarize_metric_against_shuffle(
    observed: float,
    null_samples: np.ndarray,
    *,
    direction: str,
) -> dict[str, float]:
    null_samples = np.asarray(null_samples, dtype=float).reshape(-1)
    finite_null = null_samples[np.isfinite(null_samples)]
    if not np.isfinite(observed):
        return {
            "shuffle_mean": float(np.mean(finite_null)) if finite_null.size else np.nan,
            "shuffle_sd": (
                float(np.std(finite_null, ddof=0)) if finite_null.size else np.nan
            ),
            "p_value": np.nan,
        }
    if finite_null.size == 0:
        return {
            "shuffle_mean": np.nan,
            "shuffle_sd": np.nan,
            "p_value": np.nan,
        }

    if direction == "higher":
        p_value = (1.0 + float(np.sum(finite_null >= observed))) / float(
            finite_null.size + 1
        )
    elif direction == "lower":
        p_value = (1.0 + float(np.sum(finite_null <= observed))) / float(
            finite_null.size + 1
        )
    elif direction == "zero":
        p_value = (1.0 + float(np.sum(np.abs(finite_null) <= abs(observed)))) / float(
            finite_null.size + 1
        )
    else:
        raise ValueError(f"Unsupported metric direction {direction!r}.")

    return {
        "shuffle_mean": float(np.mean(finite_null)),
        "shuffle_sd": float(np.std(finite_null, ddof=0)),
        "p_value": float(p_value),
    }


def build_epoch_summary_table(
    *,
    aligned_data: dict[str, Any],
    ripple_table: pd.DataFrame,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    state_span: float,
    scoring_schemes: list[str],
    scheme_availability: dict[str, dict[str, Any]],
    n_shuffles: int,
    shuffle_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], int]:
    observed_table, categorical_bin_data = build_per_ripple_metric_table(
        aligned_data=aligned_data,
        representation=representation,
        train_epoch=train_epoch,
        decode_epoch=decode_epoch,
        state_span=state_span,
        scoring_schemes=scoring_schemes,
        scheme_availability=scheme_availability,
    )
    null_samples: dict[str, np.ndarray] = {}
    metric_direction: dict[str, str] = {}

    if "continuous" in scoring_schemes:
        for metric_name in METRIC_LABELS:
            null_samples[metric_name] = np.full(int(n_shuffles), np.nan, dtype=float)
            metric_direction[metric_name] = METRIC_DIRECTION[metric_name]
    for scheme in CATEGORICAL_SCORING_SCHEMES:
        if scheme in scoring_schemes and scheme_availability[scheme]["applicable"]:
            metric_name = f"{scheme}_match_rate"
            null_samples[metric_name] = np.full(int(n_shuffles), np.nan, dtype=float)
            metric_direction[metric_name] = "higher"

    effective_shuffles = 0
    if aligned_data["n_ripples"] >= 2 and n_shuffles > 0 and null_samples:
        rng = np.random.default_rng(shuffle_seed)
        for shuffle_index in range(n_shuffles):
            shuffled_v1_state, changed = shuffle_ripple_state_blocks_by_length(
                aligned_data["v1_decoded_state"],
                aligned_data["ripple_ids"],
                rng,
            )
            if not changed:
                continue
            shuffled_table, _ = build_per_ripple_metric_table(
                aligned_data=aligned_data,
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
                state_span=state_span,
                scoring_schemes=scoring_schemes,
                scheme_availability=scheme_availability,
                v1_state_override=shuffled_v1_state,
            )
            for metric_name in null_samples:
                null_samples[metric_name][shuffle_index] = float(
                    np.nanmean(shuffled_table[metric_name].to_numpy(dtype=float))
                )
            effective_shuffles += 1

    summary_row: dict[str, Any] = {
        "representation": str(representation),
        "train_epoch": str(train_epoch),
        "decode_epoch": str(decode_epoch),
        "n_ripples": int(aligned_data["n_ripples"]),
        "n_ripple_bins": int(aligned_data["n_bins"]),
        "n_ripple_events_input": int(len(ripple_table)),
        "n_effective_shuffles": int(effective_shuffles),
        "scoring_schemes_json": json.dumps(scoring_schemes),
    }

    if "continuous" in scoring_schemes:
        for metric_name in METRIC_LABELS:
            observed = float(np.nanmean(observed_table[metric_name].to_numpy(dtype=float)))
            summary_row[metric_name] = observed
            shuffle_summary = summarize_metric_against_shuffle(
                observed,
                null_samples[metric_name],
                direction=metric_direction[metric_name],
            )
            summary_row[f"{metric_name}_shuffle_mean"] = shuffle_summary["shuffle_mean"]
            summary_row[f"{metric_name}_shuffle_sd"] = shuffle_summary["shuffle_sd"]
            summary_row[f"{metric_name}_p_value"] = shuffle_summary["p_value"]

    for scheme in CATEGORICAL_SCORING_SCHEMES:
        requested = scheme in scoring_schemes
        applicable = bool(scheme_availability[scheme]["applicable"])
        summary_row[f"{scheme}_scheme_requested"] = requested
        summary_row[f"{scheme}_scheme_applicable"] = applicable
        summary_row[f"{scheme}_scheme_reason"] = scheme_availability[scheme]["reason"]
        if not requested:
            continue
        metric_name = f"{scheme}_match_rate"
        if applicable:
            valid_match_rates = observed_table[metric_name].to_numpy(dtype=float)
            summary_row[f"{scheme}_n_valid_ripples"] = int(np.sum(np.isfinite(valid_match_rates)))
            observed = float(np.nanmean(valid_match_rates))
            summary_row[metric_name] = observed
            shuffle_summary = summarize_metric_against_shuffle(
                observed,
                null_samples[metric_name],
                direction=metric_direction[metric_name],
            )
            summary_row[f"{metric_name}_shuffle_mean"] = shuffle_summary["shuffle_mean"]
            summary_row[f"{metric_name}_shuffle_sd"] = shuffle_summary["shuffle_sd"]
            summary_row[f"{metric_name}_p_value"] = shuffle_summary["p_value"]
        else:
            summary_row[f"{scheme}_n_valid_ripples"] = 0
            summary_row[metric_name] = np.nan
            summary_row[f"{metric_name}_shuffle_mean"] = np.nan
            summary_row[f"{metric_name}_shuffle_sd"] = np.nan
            summary_row[f"{metric_name}_p_value"] = np.nan

    return (
        observed_table,
        pd.DataFrame([summary_row]),
        null_samples,
        categorical_bin_data,
        effective_shuffles,
    )


def build_output_stem(
    *,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
) -> str:
    return f"{representation}_train-{train_epoch}_decode-{decode_epoch}"


def build_decoded_output_name(
    *,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    region: str,
) -> str:
    return (
        f"{build_output_stem(representation=representation, train_epoch=train_epoch, decode_epoch=decode_epoch)}"
        f"_{region}_decoded"
    )


def build_epoch_dataset_name(
    *,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
) -> str:
    return (
        f"{build_output_stem(representation=representation, train_epoch=train_epoch, decode_epoch=decode_epoch)}"
        "_comparison_dataset.nc"
    )


def build_epoch_dataset(
    *,
    aligned_data: dict[str, Any],
    ca1_mask_table: pd.DataFrame,
    v1_mask_table: pd.DataFrame,
    per_ripple_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    shuffle_samples: dict[str, np.ndarray],
    categorical_bin_data: dict[str, dict[str, np.ndarray]],
    animal_name: str,
    date: str,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    bin_size_s: float,
    scoring_schemes: list[str],
    scheme_availability: dict[str, dict[str, Any]],
    sources: dict[str, Any],
    skipped_ripples: dict[str, Any],
    fit_parameters: dict[str, Any],
) -> Any:
    xr = require_xarray()

    summary_row = summary_table.iloc[0]
    attrs = {
        "schema_version": "1",
        "animal_name": animal_name,
        "date": date,
        "representation": representation,
        "train_epoch": train_epoch,
        "decode_epoch": decode_epoch,
        "comparison_direction": "ca1_vs_v1_decoded_state",
        "bin_size_s": float(bin_size_s),
        "n_ripples": int(aligned_data["n_ripples"]),
        "n_ripple_bins": int(aligned_data["n_bins"]),
        "n_effective_shuffles": int(summary_row["n_effective_shuffles"]),
        "scoring_schemes_json": json.dumps(scoring_schemes),
        "scheme_availability_json": json.dumps(scheme_availability, sort_keys=True),
        "sources_json": json.dumps(sources, sort_keys=True),
        "fit_parameters_json": json.dumps(fit_parameters, sort_keys=True),
        "skipped_ripples_json": json.dumps(skipped_ripples, sort_keys=True),
    }

    data_vars: dict[str, Any] = {
        "bin_time_s": (("bin",), np.asarray(aligned_data["bin_times_s"], dtype=float)),
        "ripple_id": (("bin",), np.asarray(aligned_data["ripple_ids"], dtype=int)),
        "ca1_decoded_state": (
            ("bin",),
            np.asarray(aligned_data["ca1_decoded_state"], dtype=float),
        ),
        "v1_decoded_state": (
            ("bin",),
            np.asarray(aligned_data["v1_decoded_state"], dtype=float),
        ),
        "ripple_source_index": (
            ("ripple",),
            np.asarray(aligned_data["ripple_source_indices"], dtype=int),
        ),
        "ripple_start_time_s": (
            ("ripple",),
            np.asarray(aligned_data["ripple_start_times_s"], dtype=float),
        ),
        "ripple_end_time_s": (
            ("ripple",),
            np.asarray(aligned_data["ripple_end_times_s"], dtype=float),
        ),
        "ca1_movement_firing_rate_hz": (
            ("ca1_unit",),
            ca1_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "ca1_keep_unit": (
            ("ca1_unit",),
            ca1_mask_table["keep_unit"].to_numpy(dtype=bool),
        ),
        "v1_movement_firing_rate_hz": (
            ("v1_unit",),
            v1_mask_table["movement_firing_rate_hz"].to_numpy(dtype=float),
        ),
        "v1_keep_unit": (("v1_unit",), v1_mask_table["keep_unit"].to_numpy(dtype=bool)),
    }
    if "continuous" in scoring_schemes:
        for metric_name in METRIC_LABELS:
            data_vars[metric_name] = (
                ("ripple",),
                per_ripple_table[metric_name].to_numpy(dtype=float),
            )
            data_vars[f"{metric_name}_observed"] = float(summary_row[metric_name])
            data_vars[f"{metric_name}_shuffle"] = (
                ("shuffle",),
                np.asarray(shuffle_samples[metric_name], dtype=float),
            )
            data_vars[f"{metric_name}_shuffle_mean"] = float(
                summary_row[f"{metric_name}_shuffle_mean"]
            )
            data_vars[f"{metric_name}_shuffle_sd"] = float(
                summary_row[f"{metric_name}_shuffle_sd"]
            )
            data_vars[f"{metric_name}_p_value"] = float(
                summary_row[f"{metric_name}_p_value"]
            )

    for scheme in CATEGORICAL_SCORING_SCHEMES:
        if scheme not in scoring_schemes or not scheme_availability[scheme]["applicable"]:
            continue
        metric_name = f"{scheme}_match_rate"
        data_vars[f"{scheme}_ca1_label"] = (
            ("bin",),
            categorical_bin_data[scheme]["ca1_labels"].astype(int, copy=False),
        )
        data_vars[f"{scheme}_v1_label"] = (
            ("bin",),
            categorical_bin_data[scheme]["v1_labels"].astype(int, copy=False),
        )
        data_vars[f"{scheme}_bin_match"] = (
            ("bin",),
            categorical_bin_data[scheme]["match"].astype(bool, copy=False),
        )
        data_vars[metric_name] = (
            ("ripple",),
            per_ripple_table[metric_name].to_numpy(dtype=float),
        )
        data_vars[f"{scheme}_n_matching_bins"] = (
            ("ripple",),
            per_ripple_table[f"{scheme}_n_matching_bins"].to_numpy(dtype=int),
        )
        data_vars[f"{scheme}_n_valid_labeled_bins"] = (
            ("ripple",),
            per_ripple_table[f"{scheme}_n_valid_labeled_bins"].to_numpy(dtype=int),
        )
        data_vars[f"{metric_name}_observed"] = float(summary_row[metric_name])
        data_vars[f"{metric_name}_shuffle"] = (
            ("shuffle",),
            np.asarray(shuffle_samples[metric_name], dtype=float),
        )
        data_vars[f"{metric_name}_shuffle_mean"] = float(
            summary_row[f"{metric_name}_shuffle_mean"]
        )
        data_vars[f"{metric_name}_shuffle_sd"] = float(
            summary_row[f"{metric_name}_shuffle_sd"]
        )
        data_vars[f"{metric_name}_p_value"] = float(
            summary_row[f"{metric_name}_p_value"]
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "bin": np.arange(int(aligned_data["n_bins"]), dtype=int),
            "ripple": np.arange(int(aligned_data["n_ripples"]), dtype=int),
            "shuffle": np.arange(
                int(len(next(iter(shuffle_samples.values()), []))), dtype=int
            ),
            "ca1_unit": ca1_mask_table["unit_id"].to_numpy(),
            "v1_unit": v1_mask_table["unit_id"].to_numpy(),
        },
        attrs=attrs,
    )


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_epoch_summary(
    *,
    aligned_data: dict[str, Any],
    summary_table: pd.DataFrame,
    shuffle_samples: dict[str, np.ndarray],
    animal_name: str,
    date: str,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    out_path: Path,
) -> Path:
    plt = get_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    ca1_state = np.asarray(aligned_data["ca1_decoded_state"], dtype=float)
    v1_state = np.asarray(aligned_data["v1_decoded_state"], dtype=float)
    valid = np.isfinite(ca1_state) & np.isfinite(v1_state)
    if np.any(valid):
        heatmap = axes[0].hist2d(
            ca1_state[valid],
            v1_state[valid],
            bins=50,
            cmap="viridis",
            cmin=1,
        )
        finite_values = np.concatenate([ca1_state[valid], v1_state[valid]])
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
        axes[0].plot(
            [value_min, value_max], [value_min, value_max], color="black", alpha=0.6
        )
        colorbar = figure.colorbar(heatmap[3], ax=axes[0], fraction=0.046, pad=0.04)
        colorbar.set_label("Bin count")
    else:
        axes[0].text(0.5, 0.5, "No finite aligned bins", ha="center", va="center")
    axes[0].set_xlabel("CA1 decoded state")
    axes[0].set_ylabel("V1 decoded state")
    axes[0].set_title("Binwise decoded-state agreement heatmap")
    axes[0].grid(True, alpha=0.2)

    summary_row = summary_table.iloc[0]
    for axis, metric_name in zip(
        axes[1:],
        ("pearson_r", "mean_abs_difference"),
        strict=False,
    ):
        shuffle_values = np.asarray(shuffle_samples[metric_name], dtype=float)
        finite_shuffle = shuffle_values[np.isfinite(shuffle_values)]
        if finite_shuffle.size:
            axis.hist(
                finite_shuffle, bins=min(20, max(finite_shuffle.size, 5)), color="0.75"
            )
            axis.axvline(float(summary_row[metric_name]), color="tab:red", linewidth=2)
        else:
            axis.text(0.5, 0.5, "No effective shuffles", ha="center", va="center")
        axis.set_xlabel(METRIC_LABELS[metric_name])
        axis.set_ylabel("Shuffle count")
        axis.set_title(
            f"Observed vs shuffle\np={summary_row[f'{metric_name}_p_value']:.3g}"
            if np.isfinite(summary_row[f"{metric_name}_p_value"])
            else "Observed vs shuffle\np=NaN"
        )
        axis.grid(True, alpha=0.2)

    figure.suptitle(
        f"{animal_name} {date} {representation} train {train_epoch} decode {decode_epoch}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def plot_categorical_summary(
    *,
    summary_table: pd.DataFrame,
    shuffle_samples: dict[str, np.ndarray],
    animal_name: str,
    date: str,
    representation: str,
    train_epoch: str,
    decode_epoch: str,
    scoring_schemes: list[str],
    scheme_availability: dict[str, dict[str, Any]],
    out_path: Path,
) -> Path | None:
    schemes_to_plot = [
        scheme
        for scheme in CATEGORICAL_SCORING_SCHEMES
        if scheme in scoring_schemes and scheme_availability[scheme]["applicable"]
    ]
    if not schemes_to_plot:
        return None

    plt = get_pyplot()
    figure, axes = plt.subplots(
        1,
        len(schemes_to_plot),
        figsize=(5.5 * len(schemes_to_plot), 4.5),
        constrained_layout=True,
    )
    if len(schemes_to_plot) == 1:
        axes = [axes]

    summary_row = summary_table.iloc[0]
    for axis, scheme in zip(axes, schemes_to_plot, strict=False):
        metric_name = f"{scheme}_match_rate"
        shuffle_values = np.asarray(shuffle_samples[metric_name], dtype=float)
        finite_shuffle = shuffle_values[np.isfinite(shuffle_values)]
        if finite_shuffle.size:
            axis.hist(
                finite_shuffle,
                bins=min(20, max(finite_shuffle.size, 5)),
                color="0.75",
            )
            axis.axvline(float(summary_row[metric_name]), color="tab:red", linewidth=2)
        else:
            axis.text(0.5, 0.5, "No effective shuffles", ha="center", va="center")
        axis.set_xlabel(f"{scheme.replace('_', ' ').title()} Match Rate")
        axis.set_ylabel("Shuffle count")
        axis.set_title(
            f"Observed vs shuffle\np={summary_row[f'{metric_name}_p_value']:.3g}"
            if np.isfinite(summary_row[f"{metric_name}_p_value"])
            else "Observed vs shuffle\np=NaN"
        )
        axis.grid(True, alpha=0.2)

    figure.suptitle(
        f"{animal_name} {date} {representation} train {train_epoch} decode {decode_epoch}",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=200)
    plt.close(figure)
    return out_path


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    scoring_schemes = resolve_scoring_schemes(args.scoring_schemes)

    analysis_path = get_analysis_path(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    session = load_task_progression_session(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
    )
    movement_firing_rates = compute_movement_firing_rates(
        session["spikes_by_region"],
        session["movement_by_run"],
        session["run_epochs"],
    )
    epoch_pairs = resolve_epoch_pairs(
        session["run_epochs"],
        requested_decode_epoch=args.decode_epoch,
        requested_train_epoch=args.train_epoch,
    )

    data_dir = analysis_path / "ripple_decoding_comparison"
    fig_dir = analysis_path / "figs" / "ripple_decoding_comparison"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    epoch_intervals = build_epoch_intervals_from_timestamps(
        session["timestamps_ephys_by_epoch"]
    )
    ripple_tables, ripple_source = load_ripple_tables_for_session(analysis_path)

    sources = dict(session["sources"])
    sources["ripple_events"] = ripple_source
    fit_parameters = {
        "animal_name": args.animal_name,
        "date": args.date,
        "data_root": args.data_root,
        "representations": list(REPRESENTATIONS),
        "epoch_pairs": [
            {"decode_epoch": decode_epoch, "train_epoch": train_epoch}
            for decode_epoch, train_epoch in epoch_pairs
        ],
        "decode_epoch": args.decode_epoch,
        "train_epoch": args.train_epoch,
        "num_sleep_epochs": NUM_SLEEP_EPOCHS,
        "num_run_epochs": NUM_RUN_EPOCHS,
        "position_offset": POSITION_OFFSET,
        "speed_threshold_cm_s": SPEED_THRESHOLD_CM_S,
        "bin_size_s": args.bin_size_s,
        "scoring_schemes": scoring_schemes,
        "ca1_min_movement_fr_hz": CA1_MIN_MOVEMENT_FR_HZ,
        "v1_min_movement_fr_hz": V1_MIN_MOVEMENT_FR_HZ,
        "n_shuffles": N_SHUFFLES,
        "shuffle_seed": SHUFFLE_SEED,
        "comparison_direction": "ca1_vs_v1_decoded_state",
    }

    ca1_spikes_all = session["spikes_by_region"]["ca1"]
    v1_spikes_all = session["spikes_by_region"]["v1"]
    ca1_unit_ids_all = np.asarray(list(ca1_spikes_all.keys()))
    v1_unit_ids_all = np.asarray(list(v1_spikes_all.keys()))

    saved_decoded_paths: list[Path] = []
    saved_ripple_table_paths: list[Path] = []
    saved_summary_table_paths: list[Path] = []
    saved_dataset_paths: list[Path] = []
    saved_figure_paths: list[Path] = []
    successful_jobs: list[dict[str, Any]] = []
    skipped_epochs: list[dict[str, Any]] = []

    for representation in REPRESENTATIONS:
        feature_by_epoch, bin_edges, feature_name = get_representation_inputs_from_session(
            session,
            animal_name=args.animal_name,
            representation=representation,
        )
        scheme_availability = get_scoring_scheme_availability(representation)
        state_span = (
            float(bin_edges[-1] - bin_edges[0]) if len(bin_edges) >= 2 else np.nan
        )

        for decode_epoch, train_epoch in epoch_pairs:
            ripple_table = ripple_tables.get(decode_epoch, empty_ripple_table())
            if ripple_table.empty:
                skipped_epochs.append(
                    {
                        "representation": representation,
                        "epoch": decode_epoch,
                        "train_epoch": train_epoch,
                        "reason": "No ripple events were found for this epoch.",
                    }
                )
                print(f"Skipping {representation} {decode_epoch}: no ripple events")
                continue

            ca1_mask_table = build_region_unit_mask_table_from_session(
                unit_ids=ca1_unit_ids_all,
                movement_firing_rates_hz=np.asarray(
                    movement_firing_rates["ca1"][train_epoch],
                    dtype=float,
                ),
                min_movement_fr_hz=CA1_MIN_MOVEMENT_FR_HZ,
                region="ca1",
            )
            v1_mask_table = build_region_unit_mask_table_from_session(
                unit_ids=v1_unit_ids_all,
                movement_firing_rates_hz=np.asarray(
                    movement_firing_rates["v1"][train_epoch],
                    dtype=float,
                ),
                min_movement_fr_hz=V1_MIN_MOVEMENT_FR_HZ,
                region="v1",
            )
            ca1_keep_unit_ids = ca1_mask_table.loc[
                ca1_mask_table["keep_unit"], "unit_id"
            ].tolist()
            v1_keep_unit_ids = v1_mask_table.loc[
                v1_mask_table["keep_unit"], "unit_id"
            ].tolist()
            if not ca1_keep_unit_ids or not v1_keep_unit_ids:
                skipped_epochs.append(
                    {
                        "representation": representation,
                        "epoch": decode_epoch,
                        "train_epoch": train_epoch,
                        "reason": "No CA1 or V1 units passed the movement firing-rate threshold.",
                    }
                )
                print(
                    f"Skipping {representation} {decode_epoch}: "
                    "no CA1 or V1 units passed movement-rate filtering"
                )
                continue

            ca1_spikes = _subset_spikes(ca1_spikes_all, ca1_keep_unit_ids)
            v1_spikes = _subset_spikes(v1_spikes_all, v1_keep_unit_ids)
            try:
                ca1_tuning_curves = compute_tuning_curves_for_epoch(
                    spikes=ca1_spikes,
                    feature=feature_by_epoch[train_epoch],
                    movement_interval=session["movement_by_run"][train_epoch],
                    bin_edges=bin_edges,
                    feature_name=feature_name,
                )
                v1_tuning_curves = compute_tuning_curves_for_epoch(
                    spikes=v1_spikes,
                    feature=feature_by_epoch[train_epoch],
                    movement_interval=session["movement_by_run"][train_epoch],
                    bin_edges=bin_edges,
                    feature_name=feature_name,
                )
                ca1_decoded = assemble_decoded_ripple_epoch_data(
                    spikes=ca1_spikes,
                    tuning_curves=ca1_tuning_curves,
                    ripple_table=ripple_table,
                    epoch_interval=epoch_intervals[decode_epoch],
                    bin_size_s=args.bin_size_s,
                )
                v1_decoded = assemble_decoded_ripple_epoch_data(
                    spikes=v1_spikes,
                    tuning_curves=v1_tuning_curves,
                    ripple_table=ripple_table,
                    epoch_interval=epoch_intervals[decode_epoch],
                    bin_size_s=args.bin_size_s,
                )
            except Exception as exc:
                skipped_epochs.append(
                    {
                        "representation": representation,
                        "epoch": decode_epoch,
                        "train_epoch": train_epoch,
                        "reason": "Failed to assemble ripple decoding inputs.",
                        "error": str(exc),
                    }
                )
                print(
                    f"Skipping {representation} {decode_epoch}: "
                    f"failed to decode ripple inputs: {exc}"
                )
                continue

            aligned_data = align_decoded_ripple_data(ca1_decoded, v1_decoded)
            if aligned_data["n_ripples"] == 0 or aligned_data["n_bins"] == 0:
                skipped_epochs.append(
                    {
                        "representation": representation,
                        "epoch": decode_epoch,
                        "train_epoch": train_epoch,
                        "reason": "No common aligned CA1/V1 ripple bins were available.",
                        "ca1_skipped_ripples": ca1_decoded["skipped_ripples"],
                        "v1_skipped_ripples": v1_decoded["skipped_ripples"],
                        "alignment_skipped_ripples": aligned_data["skipped_ripples"],
                    }
                )
                print(
                    f"Skipping {representation} {decode_epoch}: "
                    "no common aligned CA1/V1 ripple bins"
                )
                continue

            (
                per_ripple_table,
                summary_table,
                shuffle_samples,
                categorical_bin_data,
                _effective_shuffles,
            ) = build_epoch_summary_table(
                aligned_data=aligned_data,
                ripple_table=ripple_table,
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
                state_span=state_span,
                scoring_schemes=scoring_schemes,
                scheme_availability=scheme_availability,
                n_shuffles=N_SHUFFLES,
                shuffle_seed=SHUFFLE_SEED,
            )
            skipped_ripples = {
                "ca1": ca1_decoded["skipped_ripples"],
                "v1": v1_decoded["skipped_ripples"],
                "alignment": aligned_data["skipped_ripples"],
            }
            epoch_dataset = build_epoch_dataset(
                aligned_data=aligned_data,
                ca1_mask_table=ca1_mask_table,
                v1_mask_table=v1_mask_table,
                per_ripple_table=per_ripple_table,
                summary_table=summary_table,
                shuffle_samples=shuffle_samples,
                categorical_bin_data=categorical_bin_data,
                animal_name=args.animal_name,
                date=args.date,
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
                bin_size_s=args.bin_size_s,
                scoring_schemes=scoring_schemes,
                scheme_availability=scheme_availability,
                sources=sources,
                skipped_ripples=skipped_ripples,
                fit_parameters=fit_parameters,
            )

            ca1_decoded_path = (
                data_dir
                / f"{build_decoded_output_name(representation=representation, train_epoch=train_epoch, decode_epoch=decode_epoch, region='ca1')}.npz"
            )
            v1_decoded_path = (
                data_dir
                / f"{build_decoded_output_name(representation=representation, train_epoch=train_epoch, decode_epoch=decode_epoch, region='v1')}.npz"
            )
            ca1_decoded["decoded_tsd"].save(ca1_decoded_path)
            v1_decoded["decoded_tsd"].save(v1_decoded_path)
            saved_decoded_paths.extend([ca1_decoded_path, v1_decoded_path])

            output_stem = build_output_stem(
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
            )
            ripple_metrics_path = data_dir / f"{output_stem}_ripple_metrics.parquet"
            summary_table_path = data_dir / f"{output_stem}_epoch_summary.parquet"
            dataset_path = data_dir / build_epoch_dataset_name(
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
            )
            figure_path = fig_dir / f"{output_stem}_coherence_summary.png"
            categorical_figure_path = fig_dir / f"{output_stem}_categorical_summary.png"

            per_ripple_table.to_parquet(ripple_metrics_path, index=False)
            summary_table.to_parquet(summary_table_path, index=False)
            epoch_dataset.to_netcdf(dataset_path)
            if "continuous" in scoring_schemes:
                plot_epoch_summary(
                    aligned_data=aligned_data,
                    summary_table=summary_table,
                    shuffle_samples=shuffle_samples,
                    animal_name=args.animal_name,
                    date=args.date,
                    representation=representation,
                    train_epoch=train_epoch,
                    decode_epoch=decode_epoch,
                    out_path=figure_path,
                )
            else:
                figure_path = None
            saved_categorical_figure_path = plot_categorical_summary(
                summary_table=summary_table,
                shuffle_samples=shuffle_samples,
                animal_name=args.animal_name,
                date=args.date,
                representation=representation,
                train_epoch=train_epoch,
                decode_epoch=decode_epoch,
                scoring_schemes=scoring_schemes,
                scheme_availability=scheme_availability,
                out_path=categorical_figure_path,
            )

            saved_ripple_table_paths.append(ripple_metrics_path)
            saved_summary_table_paths.append(summary_table_path)
            saved_dataset_paths.append(dataset_path)
            if figure_path is not None:
                saved_figure_paths.append(figure_path)
            if saved_categorical_figure_path is not None:
                saved_figure_paths.append(saved_categorical_figure_path)
            successful_jobs.append(
                {
                    "representation": representation,
                    "decode_epoch": decode_epoch,
                    "train_epoch": train_epoch,
                    "n_ripples": int(aligned_data["n_ripples"]),
                    "n_ripple_bins": int(aligned_data["n_bins"]),
                    "n_ca1_units": int(len(ca1_keep_unit_ids)),
                    "n_v1_units": int(len(v1_keep_unit_ids)),
                    "ca1_decoded_path": ca1_decoded_path,
                    "v1_decoded_path": v1_decoded_path,
                    "ripple_metrics_path": ripple_metrics_path,
                    "summary_table_path": summary_table_path,
                    "dataset_path": dataset_path,
                    "figure_path": figure_path,
                    "categorical_figure_path": saved_categorical_figure_path,
                    "scheme_availability": scheme_availability,
                }
            )

    if not saved_summary_table_paths:
        raise RuntimeError(
            "All requested decode epochs were skipped. "
            f"Epoch reasons: {skipped_epochs!r}"
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name=Path(__file__).stem,
        parameters=fit_parameters,
        outputs={
            "sources": sources,
            "saved_decoded_paths": saved_decoded_paths,
            "saved_ripple_table_paths": saved_ripple_table_paths,
            "saved_summary_table_paths": saved_summary_table_paths,
            "saved_dataset_paths": saved_dataset_paths,
            "saved_figure_paths": saved_figure_paths,
            "successful_jobs": successful_jobs,
            "skipped_epochs": skipped_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")


if __name__ == "__main__":
    main()
