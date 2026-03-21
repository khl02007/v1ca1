from __future__ import annotations

"""Shared session-loading and preprocessing helpers.

These utilities centralize the common analysis-path, timestamp, position,
trajectory, spike-loading, and movement-preprocessing logic used across
multiple subpackages. They are intentionally generic to the recording session
and W-track setup, leaving analysis-specific representations to downstream
modules such as `task_progression` or `signal_dim`.
"""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from v1ca1.helper.wtrack import (
    get_wtrack_branch_graph,
    get_wtrack_branch_side,
    get_wtrack_direction,
    get_wtrack_total_length,
)

if TYPE_CHECKING:
    import pandas as pd
    import pynapple as nap


DEFAULT_DATA_ROOT = Path("/stelmo/kyu/analysis")
DEFAULT_NWB_ROOT = Path("/stelmo/nwb/raw")
DEFAULT_POSITION_OFFSET = 10
DEFAULT_SPEED_THRESHOLD_CM_S = 4.0
DEFAULT_SPEED_SIGMA_S = 0.1
DEFAULT_PLACE_BIN_SIZE_CM = 4.0

REGIONS = ("v1", "ca1")
TRAJECTORY_TYPES = (
    "center_to_left",
    "left_to_center",
    "center_to_right",
    "right_to_center",
)
TURN_TRAJECTORY_PAIRS = {
    "left": ("center_to_left", "right_to_center"),
    "right": ("center_to_right", "left_to_center"),
}


def get_analysis_path(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the analysis directory for one animal/date session."""
    return data_root / animal_name / date


def save_pickle_output(output_path: Path, value: Any) -> Path:
    """Write one compatibility pickle artifact and return its path."""
    with open(output_path, "wb") as f:
        pickle.dump(value, f)
    return output_path


def _extract_interval_dataframe(intervals: "nap.IntervalSet") -> "pd.DataFrame":
    """Return a dataframe-like view of a pynapple IntervalSet."""
    if hasattr(intervals, "as_dataframe"):
        return intervals.as_dataframe()
    if hasattr(intervals, "_metadata"):
        return intervals._metadata.copy()  # type: ignore[attr-defined]
    raise ValueError("Could not read metadata from the pynapple IntervalSet.")


def _extract_epoch_tags_from_intervalset(epoch_intervals: "nap.IntervalSet") -> list[str]:
    """Extract saved epoch labels from a pynapple IntervalSet."""
    try:
        epoch_info = epoch_intervals.get_info("epoch")
    except Exception:
        epoch_info = None

    if epoch_info is not None:
        epoch_array = np.asarray(epoch_info)
        if epoch_array.size:
            return [str(epoch) for epoch in epoch_array.tolist()]

    interval_df = _extract_interval_dataframe(epoch_intervals)
    if "epoch" in interval_df.columns:
        return [str(epoch) for epoch in interval_df["epoch"].tolist()]

    raise ValueError("The pynapple IntervalSet does not contain saved epoch labels.")


def _extract_interval_bounds_from_intervalset(
    intervals: "nap.IntervalSet",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned start/end arrays from a pynapple IntervalSet."""
    starts = np.asarray(intervals.start, dtype=float).ravel()
    ends = np.asarray(intervals.end, dtype=float).ravel()
    if starts.shape != ends.shape:
        raise ValueError(
            "The pynapple IntervalSet has mismatched start/end arrays: "
            f"{starts.shape} vs {ends.shape}."
        )
    return starts, ends


def _extract_epoch_tags_from_tsgroup(position_group: "nap.TsGroup") -> list[str]:
    """Extract saved epoch labels from a pynapple TsGroup."""
    try:
        epoch_info = position_group["epoch"]
    except Exception:
        epoch_info = None

    if epoch_info is None:
        raise ValueError("timestamps_position.npz does not contain the saved epoch labels.")

    epoch_array = np.asarray(epoch_info)
    if epoch_array.size == 0:
        raise ValueError("timestamps_position.npz does not contain any saved epoch labels.")
    return [str(epoch) for epoch in epoch_array.tolist()]


def load_epoch_tags(analysis_path: Path) -> tuple[list[str], str]:
    """Load saved epoch labels, preferring `timestamps_ephys.npz`."""
    npz_path = analysis_path / "timestamps_ephys.npz"
    npz_error: Exception | None = None
    if npz_path.exists():
        try:
            import pynapple as nap
        except ModuleNotFoundError:
            pass
        else:
            try:
                epoch_intervals = nap.load_file(npz_path)
                return _extract_epoch_tags_from_intervalset(epoch_intervals), "pynapple"
            except Exception as exc:
                npz_error = exc

    pickle_path = analysis_path / "timestamps_ephys.pkl"
    if not pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            f"Could not find timestamps_ephys.npz or timestamps_ephys.pkl under {analysis_path}."
        )

    with open(pickle_path, "rb") as f:
        timestamps_ephys = pickle.load(f)
    return [str(epoch) for epoch in timestamps_ephys.keys()], "pickle"


def get_run_epochs(epoch_tags: list[str]) -> list[str]:
    """Infer run epochs from saved epoch labels using the lab `r*` naming convention."""
    run_epochs = [epoch for epoch in epoch_tags if "r" in epoch.lower()]
    if not run_epochs:
        raise ValueError(
            "Could not infer run epochs from saved epoch labels. "
            f"Available epochs: {epoch_tags!r}"
        )
    return run_epochs


def load_ephys_timestamps_all(analysis_path: Path) -> tuple[np.ndarray, str]:
    """Load the concatenated ephys timestamps, preferring `timestamps_ephys_all.npz`."""
    npz_path = analysis_path / "timestamps_ephys_all.npz"
    npz_error: Exception | None = None
    if npz_path.exists():
        try:
            import pynapple as nap
        except ModuleNotFoundError:
            pass
        else:
            try:
                timestamps_all = nap.load_file(npz_path)
                return np.asarray(timestamps_all.t, dtype=float), "pynapple"
            except Exception as exc:
                npz_error = exc

    pickle_path = analysis_path / "timestamps_ephys_all.pkl"
    if not pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            f"Could not find timestamps_ephys_all.npz or timestamps_ephys_all.pkl under {analysis_path}."
        )

    with open(pickle_path, "rb") as f:
        timestamps_all = pickle.load(f)
    return np.asarray(timestamps_all, dtype=float), "pickle"


def load_ephys_timestamps_by_epoch(
    analysis_path: Path,
) -> tuple[list[str], dict[str, np.ndarray], str]:
    """Load per-epoch ephys timestamps, preferring `timestamps_ephys.npz`."""
    npz_path = analysis_path / "timestamps_ephys.npz"
    npz_error: Exception | None = None
    if npz_path.exists():
        try:
            import pynapple as nap
        except ModuleNotFoundError:
            pass
        else:
            try:
                epoch_intervals = nap.load_file(npz_path)
                epoch_tags = _extract_epoch_tags_from_intervalset(epoch_intervals)
                starts, ends = _extract_interval_bounds_from_intervalset(epoch_intervals)
                if len(epoch_tags) != starts.size:
                    raise ValueError(
                        "Mismatch between epoch labels and saved epoch intervals in "
                        f"{npz_path}."
                    )

                timestamps_all, _ = load_ephys_timestamps_all(analysis_path)
                timestamps_by_epoch: dict[str, np.ndarray] = {}
                for epoch, start, end in zip(epoch_tags, starts, ends):
                    start_index = int(np.searchsorted(timestamps_all, float(start), side="left"))
                    end_index = int(np.searchsorted(timestamps_all, float(end), side="right"))
                    epoch_timestamps = np.asarray(
                        timestamps_all[start_index:end_index],
                        dtype=float,
                    )
                    if epoch_timestamps.size == 0:
                        raise ValueError(
                            "Could not reconstruct any ephys timestamps for epoch "
                            f"{epoch!r} from {npz_path}."
                        )
                    timestamps_by_epoch[epoch] = epoch_timestamps
                return epoch_tags, timestamps_by_epoch, "pynapple"
            except Exception as exc:
                npz_error = exc

    pickle_path = analysis_path / "timestamps_ephys.pkl"
    if not pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            f"Could not find timestamps_ephys.npz or timestamps_ephys.pkl under {analysis_path}."
        )

    with open(pickle_path, "rb") as f:
        timestamps_ephys = pickle.load(f)
    return (
        [str(epoch) for epoch in timestamps_ephys.keys()],
        {
            str(epoch): np.asarray(timestamps, dtype=float)
            for epoch, timestamps in timestamps_ephys.items()
        },
        "pickle",
    )


def load_position_timestamps(analysis_path: Path) -> tuple[list[str], dict[str, np.ndarray], str]:
    """Load per-epoch position timestamps, preferring `timestamps_position.npz`."""
    npz_path = analysis_path / "timestamps_position.npz"
    npz_error: Exception | None = None
    if npz_path.exists():
        try:
            import pynapple as nap
        except ModuleNotFoundError:
            pass
        else:
            try:
                position_group = nap.load_file(npz_path)
                epoch_tags = _extract_epoch_tags_from_tsgroup(position_group)
                if len(epoch_tags) != len(position_group):
                    raise ValueError(
                        "Mismatch between epoch labels and time series count in "
                        f"{npz_path}."
                    )
                timestamps_position = {
                    epoch: np.asarray(position_group[index].t, dtype=float)
                    for index, epoch in enumerate(epoch_tags)
                }
                return epoch_tags, timestamps_position, "pynapple"
            except Exception as exc:
                npz_error = exc

    pickle_path = analysis_path / "timestamps_position.pkl"
    if not pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            f"Could not find timestamps_position.npz or timestamps_position.pkl under {analysis_path}."
        )

    with open(pickle_path, "rb") as f:
        timestamps_position = pickle.load(f)
    return (
        [str(epoch) for epoch in timestamps_position.keys()],
        {
            str(epoch): np.asarray(timestamps, dtype=float)
            for epoch, timestamps in timestamps_position.items()
        },
        "pickle",
    )


def coerce_position_array(position: Any) -> np.ndarray:
    """Coerce a position-like object to a two-column XY NumPy array."""
    position_array = np.asarray(position)
    if position_array.ndim != 2:
        raise ValueError(
            f"Expected a 2D position array, got shape {position_array.shape}."
        )
    if position_array.shape[1] == 2:
        return position_array
    if position_array.shape[0] == 2:
        return position_array[:2, :].T
    if position_array.shape[1] > 2:
        return position_array[:, :2]
    raise ValueError(
        f"Could not interpret position array of shape {position_array.shape} as XY samples."
    )


def load_position_data(
    analysis_path: Path,
    epoch_tags: list[str],
) -> dict[str, np.ndarray]:
    """Load per-epoch XY position arrays from `position.pkl`."""
    position_path = analysis_path / "position.pkl"
    if not position_path.exists():
        raise FileNotFoundError(f"Position file not found: {position_path}")

    with open(position_path, "rb") as f:
        position_dict = pickle.load(f)

    normalized_position_dict = {
        str(epoch): coerce_position_array(value)
        for epoch, value in position_dict.items()
    }
    missing_epochs = [epoch for epoch in epoch_tags if epoch not in normalized_position_dict]
    extra_epochs = sorted(set(normalized_position_dict) - set(epoch_tags))
    if missing_epochs or extra_epochs:
        raise ValueError(
            "Position epochs do not match saved timestamp epochs. "
            f"Missing position epochs: {missing_epochs!r}; extra position epochs: {extra_epochs!r}"
        )
    return {epoch: normalized_position_dict[epoch] for epoch in epoch_tags}


def _get_interval_metadata_values(
    intervals: "nap.IntervalSet",
    key: str,
) -> np.ndarray:
    """Return one metadata column from a pynapple IntervalSet."""
    try:
        values = intervals.get_info(key)
    except Exception:
        values = None

    if values is not None:
        value_array = np.asarray(values)
        if value_array.size:
            return value_array

    interval_df = _extract_interval_dataframe(intervals)
    if key in interval_df.columns:
        return np.asarray(interval_df[key].tolist())
    raise ValueError(f"The pynapple IntervalSet does not contain metadata {key!r}.")


def load_trajectory_intervals(
    analysis_path: Path,
    run_epochs: list[str],
) -> tuple[dict[str, dict[str, "nap.IntervalSet"]], str]:
    """Load trajectory intervals per epoch and trajectory, preferring `trajectory_times.npz`."""
    import pynapple as nap

    npz_path = analysis_path / "trajectory_times.npz"
    npz_error: Exception | None = None
    if npz_path.exists():
        try:
            trajectory_intervals = nap.load_file(npz_path)
            epochs = _get_interval_metadata_values(trajectory_intervals, "epoch").astype(str)
            trajectory_types = _get_interval_metadata_values(
                trajectory_intervals,
                "trajectory_type",
            ).astype(str)
            starts = np.asarray(trajectory_intervals.start, dtype=float)
            ends = np.asarray(trajectory_intervals.end, dtype=float)

            intervals_by_epoch: dict[str, dict[str, nap.IntervalSet]] = {}
            for epoch in run_epochs:
                intervals_by_epoch[epoch] = {}
                for trajectory_type in TRAJECTORY_TYPES:
                    mask = (epochs == epoch) & (trajectory_types == trajectory_type)
                    intervals_by_epoch[epoch][trajectory_type] = nap.IntervalSet(
                        start=starts[mask],
                        end=ends[mask],
                        time_units="s",
                    )
            return intervals_by_epoch, "pynapple"
        except Exception as exc:
            npz_error = exc

    pickle_path = analysis_path / "trajectory_times.pkl"
    if not pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            f"Could not find trajectory_times.npz or trajectory_times.pkl under {analysis_path}."
        )

    with open(pickle_path, "rb") as f:
        trajectory_times = pickle.load(f)

    intervals_by_epoch = {}
    for epoch in run_epochs:
        intervals_by_epoch[epoch] = {}
        for trajectory_type in TRAJECTORY_TYPES:
            interval_array = np.asarray(trajectory_times[epoch][trajectory_type], dtype=float)
            if interval_array.size == 0:
                start = np.array([], dtype=float)
                end = np.array([], dtype=float)
            else:
                start = interval_array[:, 0]
                end = interval_array[:, 1]
            intervals_by_epoch[epoch][trajectory_type] = nap.IntervalSet(
                start=start,
                end=end,
                time_units="s",
            )
    return intervals_by_epoch, "pickle"


def get_position_sampling_rate(timestamps_position: np.ndarray) -> float:
    """Return the position sampling rate inferred from epoch timestamps."""
    if timestamps_position.ndim != 1 or timestamps_position.size < 2:
        raise ValueError("Position timestamps must have at least two samples.")
    duration = float(timestamps_position[-1] - timestamps_position[0])
    if duration <= 0:
        raise ValueError("Position timestamps must span a positive duration.")
    return (len(timestamps_position) - 1) / duration


def get_spike_tsgroup(sorting: Any, timestamps_ephys_all: np.ndarray) -> "nap.TsGroup":
    """Convert one SpikeInterface sorting object to a pynapple TsGroup."""
    import pynapple as nap

    spikes = {
        unit_id: nap.Ts(
            t=timestamps_ephys_all[sorting.get_unit_spike_train(unit_id)],
            time_units="s",
        )
        for unit_id in sorting.get_unit_ids()
    }
    return nap.TsGroup(spikes, time_units="s")


def load_spikes_by_region(
    analysis_path: Path,
    timestamps_ephys_all: np.ndarray,
    regions: tuple[str, ...] = REGIONS,
) -> dict[str, "nap.TsGroup"]:
    """Load spike trains for the requested regions as pynapple TsGroups."""
    import spikeinterface.full as si

    return {
        region: get_spike_tsgroup(
            si.load(analysis_path / f"sorting_{region}"),
            timestamps_ephys_all,
        )
        for region in regions
    }


def get_track_graph_for_trajectory(
    animal_name: str,
    trajectory_type: str,
) -> tuple[Any, list[tuple[int, int]]]:
    """Return the track graph and edge order for one trajectory type."""
    return get_wtrack_branch_graph(
        animal_name=animal_name,
        branch_side=get_wtrack_branch_side(trajectory_type),
        direction=get_wtrack_direction(trajectory_type),
    )


def _get_track_graph_for_trajectory(
    animal_name: str,
    trajectory_type: str,
) -> tuple[Any, list[tuple[int, int]]]:
    """Backward-compatible alias for the track-graph helper."""
    return get_track_graph_for_trajectory(animal_name, trajectory_type)


def build_epoch_interval(
    timestamps_position: np.ndarray,
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> "nap.IntervalSet":
    """Return the full trimmed epoch interval after the requested position offset."""
    import pynapple as nap

    if position_offset < 0:
        raise ValueError("--position-offset must be non-negative.")
    if timestamps_position.size <= position_offset:
        raise ValueError(
            "Position offset removes all timestamp samples for one epoch. "
            f"timestamp count: {timestamps_position.size}, position_offset: {position_offset}"
        )
    return nap.IntervalSet(
        start=float(timestamps_position[position_offset]),
        end=float(timestamps_position[-1]),
        time_units="s",
    )


def build_speed_tsd(
    position: np.ndarray,
    timestamps_position: np.ndarray,
    position_offset: int = DEFAULT_POSITION_OFFSET,
) -> "nap.Tsd":
    """Compute a speed Tsd for one epoch after trimming the leading position offset."""
    import position_tools as pt
    import pynapple as nap

    if position.shape[0] != timestamps_position.size:
        raise ValueError(
            "Position samples and position timestamps must have the same length. "
            f"Got {position.shape[0]} and {timestamps_position.size}."
        )
    if position.shape[0] <= position_offset:
        raise ValueError(
            "Position offset removes all position samples for one epoch. "
            f"position count: {position.shape[0]}, position_offset: {position_offset}"
        )

    epoch_position = position[position_offset:]
    epoch_timestamps = np.asarray(timestamps_position[position_offset:], dtype=float)
    speed = np.asarray(
        pt.get_speed(
            position=epoch_position,
            time=epoch_timestamps,
            sampling_frequency=get_position_sampling_rate(epoch_timestamps),
            sigma=DEFAULT_SPEED_SIGMA_S,
        ),
        dtype=float,
    )
    if speed.shape[0] != epoch_timestamps.shape[0]:
        raise ValueError(
            "Speed computation returned a different number of samples than the trimmed timestamps."
        )
    return nap.Tsd(t=epoch_timestamps, d=speed, time_units="s")


def build_movement_interval(
    speed_tsd: "nap.Tsd",
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
) -> "nap.IntervalSet":
    """Return the movement IntervalSet for one epoch using a speed threshold."""
    if speed_threshold_cm_s < 0:
        raise ValueError("--speed-threshold-cm-s must be non-negative.")
    epoch_interval = speed_tsd.time_support
    movement = speed_tsd.threshold(speed_threshold_cm_s, method="above").time_support
    return movement.intersect(epoch_interval)


def compute_movement_firing_rates(
    spikes_by_region: dict[str, "nap.TsGroup"],
    movement_by_run: dict[str, "nap.IntervalSet"],
    run_epochs: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute mean firing rates during movement for each region and run epoch."""
    movement_firing_rates: dict[str, dict[str, np.ndarray]] = {}
    for region, spikes in spikes_by_region.items():
        movement_firing_rates[region] = {}
        for epoch in run_epochs:
            movement_interval = movement_by_run[epoch]
            duration = float(movement_interval.tot_length())
            if duration <= 0:
                movement_firing_rates[region][epoch] = np.zeros(len(spikes.keys()), dtype=float)
                continue
            rates = spikes.count(ep=movement_interval).to_numpy() / duration
            movement_firing_rates[region][epoch] = np.sum(rates, axis=0).ravel()
    return movement_firing_rates
