from __future__ import annotations

"""Build poke-defined trajectory intervals for one session.

This script reads per-epoch ephys timestamp bounds from the analysis folder,
loads left/center/right poke events from the NWB file, infers which epochs are
run epochs from the saved epoch labels, and writes poke-to-poke intervals for
the four supported trajectory types:

- `left_to_center`
- `center_to_left`
- `right_to_center`
- `center_to_right`

By default it writes a single `trajectory_times.npz` pynapple `IntervalSet`
whose metadata stores the full
epoch label (for example `02_r1`) and trajectory type for each interval row.

The script prefers the pynapple-backed `timestamps_ephys.npz` export when it is
available and readable, and otherwise falls back to `timestamps_ephys.pkl`.

It assumes the NWB file has a `processing["behavior"]` module with a
`behavioral_events` interface containing the event series `Poke Left`,
`Poke Center`, and `Poke Right`. If any of those fields are missing, the script
raises a descriptive error listing the missing event and the available event
series found in the NWB file.
"""

import argparse
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import DEFAULT_NWB_ROOT, save_pickle_output

if TYPE_CHECKING:
    import pandas as pd
    import pynapple as nap
    import pynwb


DEFAULT_DATA_ROOT = Path("/stelmo/kyu/analysis")

POKE_EVENT_NAMES = {
    "left": "Poke Left",
    "center": "Poke Center",
    "right": "Poke Right",
}
TRAJECTORY_TYPES = {
    ("left", "center"): "left_to_center",
    ("center", "left"): "center_to_left",
    ("right", "center"): "right_to_center",
    ("center", "right"): "center_to_right",
}


def get_analysis_path(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> Path:
    """Return the analysis directory for one animal/date session."""
    return data_root / animal_name / date


def _extract_interval_dataframe(epoch_intervals: "nap.IntervalSet") -> "pd.DataFrame":
    """Return a dataframe-like view of a pynapple IntervalSet."""
    if hasattr(epoch_intervals, "as_dataframe"):
        interval_df = epoch_intervals.as_dataframe()
    elif hasattr(epoch_intervals, "_metadata"):
        interval_df = epoch_intervals._metadata.copy()  # type: ignore[attr-defined]
    else:
        raise ValueError(
            "Could not read timestamps_ephys.npz metadata from the pynapple IntervalSet."
        )
    return interval_df


def _extract_interval_bounds(epoch_intervals: "nap.IntervalSet") -> np.ndarray:
    """Extract epoch start/stop bounds from a pynapple IntervalSet."""
    interval_df = _extract_interval_dataframe(epoch_intervals)
    if {"start", "end"}.issubset(interval_df.columns):
        return interval_df.loc[:, ["start", "end"]].to_numpy(dtype=float)

    starts = getattr(epoch_intervals, "start", None)
    ends = getattr(epoch_intervals, "end", None)
    if starts is not None and ends is not None:
        return np.column_stack((np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)))

    values = np.asarray(epoch_intervals, dtype=float)
    if values.ndim == 2 and values.shape[1] >= 2:
        return values[:, :2]

    raise ValueError("Could not extract interval bounds from timestamps_ephys.npz.")


def _extract_epoch_tags(epoch_intervals: "nap.IntervalSet") -> list[str]:
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

    raise ValueError(
        "timestamps_ephys.npz does not contain the saved epoch labels needed "
        "to identify run epochs."
    )


def load_epoch_time_bounds(
    analysis_path: Path,
) -> tuple[list[str], dict[str, tuple[float, float]], str]:
    """Load per-epoch ephys start/stop bounds, preferring pynapple outputs.

    Returns the saved epoch labels, a mapping from epoch label to `(start, stop)`
    time bounds, and the source used (`"pynapple"` or `"pickle"`).
    """
    ephys_npz_path = analysis_path / "timestamps_ephys.npz"
    npz_error: Exception | None = None
    if ephys_npz_path.exists():
        try:
            import pynapple as nap
        except ModuleNotFoundError:
            pass
        else:
            try:
                epoch_intervals = nap.load_file(ephys_npz_path)
                epoch_tags = _extract_epoch_tags(epoch_intervals)
                epoch_bounds = _extract_interval_bounds(epoch_intervals)
                if epoch_bounds.shape[0] != len(epoch_tags):
                    raise ValueError(
                        "Mismatch between epoch labels and interval bounds in "
                        f"{ephys_npz_path}."
                    )
                return (
                    epoch_tags,
                    {
                        epoch: (float(bounds[0]), float(bounds[1]))
                        for epoch, bounds in zip(epoch_tags, epoch_bounds, strict=True)
                    },
                    "pynapple",
                )
            except Exception as exc:
                npz_error = exc

    ephys_pickle_path = analysis_path / "timestamps_ephys.pkl"
    if not ephys_pickle_path.exists():
        if npz_error is not None:
            raise ValueError(
                f"Failed to load {ephys_npz_path} and no pickle fallback was found."
            ) from npz_error
        raise FileNotFoundError(
            "Could not find timestamps_ephys.npz or timestamps_ephys.pkl under "
            f"{analysis_path}."
        )

    if npz_error is not None:
        print(
            "Falling back to timestamps_ephys.pkl because timestamps_ephys.npz "
            f"could not be loaded: {npz_error}"
        )

    with open(ephys_pickle_path, "rb") as f:
        timestamps_ephys = pickle.load(f)

    epoch_tags = [str(epoch) for epoch in timestamps_ephys.keys()]
    epoch_bounds: dict[str, tuple[float, float]] = {}
    for epoch, timestamps in timestamps_ephys.items():
        timestamps_array = np.asarray(timestamps, dtype=float)
        if timestamps_array.ndim != 1 or timestamps_array.size == 0:
            raise ValueError(f"Expected non-empty 1D timestamps for epoch {epoch!r}.")
        epoch_bounds[str(epoch)] = (float(timestamps_array[0]), float(timestamps_array[-1]))

    return epoch_tags, epoch_bounds, "pickle"


def get_run_epochs(epoch_tags: list[str]) -> list[str]:
    """Infer run epochs from saved epoch labels using the lab's `r*` convention."""
    run_epochs = [epoch for epoch in epoch_tags if "r" in epoch.lower()]
    if not run_epochs:
        raise ValueError(
            "Could not infer run epochs from timestamp labels. "
            f"Available epochs: {epoch_tags!r}"
        )
    return run_epochs


def get_behavioral_events_interface(nwbfile: "pynwb.NWBFile"):
    """Return the NWB behavioral_events interface with a descriptive error if missing."""
    processing_module = nwbfile.processing.get("behavior")
    if processing_module is None:
        available_modules = sorted(nwbfile.processing.keys())
        raise ValueError(
            "NWB file is missing processing['behavior'], which is required to load "
            "trajectory poke events. "
            f"Available processing modules: {available_modules!r}"
        )

    behavioral_events = processing_module.data_interfaces.get("behavioral_events")
    if behavioral_events is None:
        available_interfaces = sorted(processing_module.data_interfaces.keys())
        raise ValueError(
            "NWB file is missing processing['behavior'].data_interfaces['behavioral_events'], "
            "which is required to load trajectory poke events. "
            f"Available behavior interfaces: {available_interfaces!r}"
        )
    return behavioral_events


def load_poke_times(nwbfile: "pynwb.NWBFile", event_name: str) -> np.ndarray:
    """Return poke-on timestamps for one behavioral event series."""
    behavioral_events = get_behavioral_events_interface(nwbfile)
    if event_name not in behavioral_events.time_series:
        available_event_names = sorted(behavioral_events.time_series.keys())
        raise ValueError(
            f"NWB file is missing behavioral event series {event_name!r}, which is "
            "required by get_trajectory_times.py. "
            f"Available behavioral event series: {available_event_names!r}"
        )

    time_series = behavioral_events.time_series[event_name]
    data = np.asarray(time_series.data[:])
    timestamps = np.asarray(time_series.timestamps[:], dtype=float)
    return timestamps[data == 1]


def to_interval_array(intervals: list[list[float]]) -> np.ndarray:
    """Convert `[start, stop]` pairs to a NumPy array with shape `(n, 2)`."""
    return np.asarray(intervals, dtype=float).reshape((-1, 2))


def save_legacy_trajectory_pickle_output(
    analysis_path: Path,
    trajectory_times: dict[str, dict[str, np.ndarray]],
) -> Path:
    """Write the legacy nested trajectory pickle used by downstream scripts."""
    output_path = analysis_path / "trajectory_times.pkl"
    return save_pickle_output(output_path, trajectory_times)


def save_pynapple_trajectory_output(
    analysis_path: Path,
    trajectory_times: dict[str, dict[str, np.ndarray]],
) -> Path:
    """Write one pynapple IntervalSet with epoch and trajectory metadata."""
    import pynapple as nap

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    epochs: list[str] = []
    trajectory_types: list[str] = []

    for epoch, trajectory_times_by_type in trajectory_times.items():
        for trajectory_type, intervals in trajectory_times_by_type.items():
            interval_array = np.asarray(intervals, dtype=float)
            if interval_array.size == 0:
                continue
            if interval_array.ndim != 2 or interval_array.shape[1] != 2:
                raise ValueError(
                    "Expected trajectory intervals with shape (n, 2) for "
                    f"{epoch!r} / {trajectory_type!r}, got {interval_array.shape}."
                )

            starts.append(interval_array[:, 0])
            ends.append(interval_array[:, 1])
            epochs.extend([str(epoch)] * interval_array.shape[0])
            trajectory_types.extend([str(trajectory_type)] * interval_array.shape[0])

    if starts:
        start_array = np.concatenate(starts).astype(float, copy=False)
        end_array = np.concatenate(ends).astype(float, copy=False)
    else:
        start_array = np.array([], dtype=float)
        end_array = np.array([], dtype=float)

    interval_set = nap.IntervalSet(start=start_array, end=end_array, time_units="s")
    interval_set.set_info(epoch=epochs, trajectory_type=trajectory_types)

    output_path = analysis_path / "trajectory_times.npz"
    interval_set.save(output_path)
    return output_path


def get_trajectory_times_for_epoch(
    epoch_start: float,
    epoch_stop: float,
    poke_times_by_well: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build poke-to-poke trajectory intervals for one run epoch.

    Pokes are filtered to the epoch bounds, merged into one time-ordered event
    list, and adjacent poke pairs are converted into the supported trajectory
    types when they match one of the allowed transitions.
    """
    poke_times: list[np.ndarray] = []
    poke_labels: list[np.ndarray] = []
    for well_name, well_poke_times in poke_times_by_well.items():
        epoch_poke_times = well_poke_times[
            (well_poke_times >= epoch_start) & (well_poke_times < epoch_stop)
        ]
        poke_times.append(epoch_poke_times)
        poke_labels.append(np.full(epoch_poke_times.shape, well_name, dtype=object))

    all_poke_times = np.concatenate(poke_times)
    all_poke_labels = np.concatenate(poke_labels)
    poke_order = np.argsort(all_poke_times)
    all_poke_times = all_poke_times[poke_order]
    all_poke_labels = all_poke_labels[poke_order]

    trajectory_times = {trajectory_type: [] for trajectory_type in TRAJECTORY_TYPES.values()}
    for i in range(len(all_poke_times) - 1):
        transition = (str(all_poke_labels[i]), str(all_poke_labels[i + 1]))
        trajectory_type = TRAJECTORY_TYPES.get(transition)
        if trajectory_type is None:
            continue
        trajectory_times[trajectory_type].append(
            [float(all_poke_times[i]), float(all_poke_times[i + 1])]
        )

    return {
        trajectory_type: to_interval_array(intervals)
        for trajectory_type, intervals in trajectory_times.items()
    }


def get_trajectory_times(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
    save_pkl: bool = False,
) -> None:
    """Compute and save poke-defined trajectory intervals for one session."""
    import pynwb

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    nwb_path = nwb_root / f"{animal_name}{date}.nwb"

    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis path not found: {analysis_path}")
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    epoch_tags, epoch_bounds, timestamp_source = load_epoch_time_bounds(analysis_path)
    run_epoch_list = get_run_epochs(epoch_tags)

    print(f"Processing {animal_name} {date}.")
    with pynwb.NWBHDF5IO(path=nwb_path, mode="r") as io:
        nwbfile = io.read()
        poke_times_by_well = {
            well_name: load_poke_times(nwbfile, event_name)
            for well_name, event_name in POKE_EVENT_NAMES.items()
        }

    trajectory_times: dict[str, dict[str, np.ndarray]] = {}
    for run_epoch in run_epoch_list:
        epoch_start, epoch_stop = epoch_bounds[run_epoch]
        trajectory_times[run_epoch] = get_trajectory_times_for_epoch(
            epoch_start=epoch_start,
            epoch_stop=epoch_stop,
            poke_times_by_well=poke_times_by_well,
        )

    outputs = {
        "timestamp_source": timestamp_source,
        "epoch_tags": epoch_tags,
        "run_epochs": run_epoch_list,
        "trajectory_types": list(TRAJECTORY_TYPES.values()),
        "trajectory_times_pynapple_path": save_pynapple_trajectory_output(
            analysis_path=analysis_path,
            trajectory_times=trajectory_times,
        ),
    }
    if save_pkl:
        outputs["trajectory_times_pickle_path"] = save_legacy_trajectory_pickle_output(
            analysis_path=analysis_path,
            trajectory_times=trajectory_times,
        )

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.helper.get_trajectory_times",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
            "save_pkl": save_pkl,
        },
        outputs=outputs,
    )
    print(f"Saved run metadata to {log_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the trajectory time export CLI."""
    parser = argparse.ArgumentParser(description="Save trajectory times")
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
        "--save-pkl",
        action="store_true",
        help="Also write the legacy trajectory_times.pkl export alongside trajectory_times.npz.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the trajectory time export CLI."""
    args = parse_arguments()
    get_trajectory_times(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
        save_pkl=args.save_pkl,
    )


if __name__ == "__main__":
    main()
