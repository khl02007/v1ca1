from __future__ import annotations

"""Plot reward timing from NWB pump events for one session.

This script loads one session NWB file, extracts left/center/right reward pump
intervals from ``processing["behavior"].data_interfaces["behavioral_events"]``
and aligns reward onsets to NWB run epochs. It saves a run-by-run figure plus
parquet summary tables under the session analysis directory.

Rewards are defined by pump onset (the rising edge from 0 to 1), since the
pumps are only active while reward is dispensed. The saved event table includes
per-reward onset/offset timestamps and pump durations for quality control, and
`reward_counts.parquet` stores both all-session totals and per-run counts by
well.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from v1ca1.helper.get_timestamps import extract_epoch_metadata
from v1ca1.helper.get_trajectory_times import get_behavioral_events_interface
from v1ca1.helper.run_logging import write_run_log
from v1ca1.helper.session import (
    DEFAULT_DATA_ROOT,
    DEFAULT_NWB_ROOT,
    get_analysis_path,
    get_run_epochs,
)

if TYPE_CHECKING:
    import pynwb


PUMP_EVENT_NAMES = {
    "left": "Pump Left",
    "center": "Pump Center",
    "right": "Pump Right",
}
WELL_ORDER = ("left", "center", "right")
WELL_COLORS = {
    "left": "tab:blue",
    "center": "tab:green",
    "right": "tab:orange",
}


def get_run_epoch_bounds(nwbfile: "pynwb.NWBFile") -> list[tuple[str, float, float]]:
    """Return ordered `(epoch, start, stop)` tuples for all run epochs."""
    epoch_tags, epoch_start_times, epoch_stop_times = extract_epoch_metadata(nwbfile)
    run_epochs = get_run_epochs(epoch_tags)
    run_epoch_set = set(run_epochs)
    return [
        (str(epoch), float(start), float(stop))
        for epoch, start, stop in zip(
            epoch_tags,
            epoch_start_times,
            epoch_stop_times,
            strict=True,
        )
        if epoch in run_epoch_set
    ]


def load_binary_event_intervals(
    behavioral_events: Any,
    event_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return onset and offset timestamps for one binary NWB event series."""
    if event_name not in behavioral_events.time_series:
        available_event_names = sorted(behavioral_events.time_series.keys())
        raise ValueError(
            f"NWB file is missing behavioral event series {event_name!r}. "
            f"Available behavioral event series: {available_event_names!r}"
        )

    time_series = behavioral_events.time_series[event_name]
    data = np.asarray(time_series.data[:]).reshape(-1)
    timestamps = np.asarray(time_series.timestamps[:], dtype=float).reshape(-1)

    if data.shape != timestamps.shape:
        raise ValueError(
            f"Behavioral event series {event_name!r} has mismatched data and timestamp "
            f"shapes: {data.shape} vs {timestamps.shape}."
        )
    if data.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if not np.all(np.isin(data, [0, 1])):
        unique_values = np.unique(data).tolist()
        raise ValueError(
            f"Behavioral event series {event_name!r} must be binary, got values "
            f"{unique_values!r}."
        )

    binary_data = data.astype(int, copy=False)
    onset_indices = np.flatnonzero(
        (binary_data == 1)
        & np.concatenate(([True], binary_data[:-1] == 0))
    )
    offset_indices = np.flatnonzero(
        (binary_data == 0)
        & np.concatenate(([False], binary_data[:-1] == 1))
    )

    if binary_data[-1] == 1:
        raise ValueError(
            f"Behavioral event series {event_name!r} ends in the on state and cannot "
            "be paired into complete pump intervals."
        )
    if onset_indices.size != offset_indices.size:
        raise ValueError(
            f"Behavioral event series {event_name!r} has mismatched onset/offset counts: "
            f"{onset_indices.size} vs {offset_indices.size}."
        )

    onset_times = timestamps[onset_indices].astype(float, copy=False)
    offset_times = timestamps[offset_indices].astype(float, copy=False)
    if np.any(offset_times <= onset_times):
        raise ValueError(
            f"Behavioral event series {event_name!r} has non-positive pump durations."
        )
    return onset_times, offset_times


def load_reward_intervals_by_well(
    nwbfile: "pynwb.NWBFile",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return reward onset/offset timestamps for each well."""
    behavioral_events = get_behavioral_events_interface(nwbfile)
    return {
        well: load_binary_event_intervals(behavioral_events, event_name)
        for well, event_name in PUMP_EVENT_NAMES.items()
    }


def build_reward_events_table(
    reward_intervals_by_well: dict[str, tuple[np.ndarray, np.ndarray]],
    run_epoch_bounds: list[tuple[str, float, float]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return one row per reward and counts of rewards outside run epochs."""
    rows: list[dict[str, float | str]] = []
    skipped_outside_run_epochs: dict[str, int] = {}

    for well in WELL_ORDER:
        onset_times, offset_times = reward_intervals_by_well[well]
        assigned_mask = np.zeros(onset_times.shape, dtype=bool)

        for epoch, start_time, stop_time in run_epoch_bounds:
            in_epoch = (onset_times >= start_time) & (onset_times < stop_time)
            if not np.any(in_epoch):
                continue

            assigned_mask |= in_epoch
            for onset_time, offset_time in zip(
                onset_times[in_epoch],
                offset_times[in_epoch],
                strict=True,
            ):
                rows.append(
                    {
                        "epoch": epoch,
                        "well": well,
                        "reward_onset_s": float(onset_time),
                        "reward_offset_s": float(offset_time),
                        "pump_duration_s": float(offset_time - onset_time),
                        "run_start_s": float(start_time),
                        "run_stop_s": float(stop_time),
                        "time_from_run_start_s": float(onset_time - start_time),
                    }
                )

        skipped_outside_run_epochs[well] = int((~assigned_mask).sum())

    reward_events = pd.DataFrame.from_records(
        rows,
        columns=[
            "epoch",
            "well",
            "reward_onset_s",
            "reward_offset_s",
            "pump_duration_s",
            "run_start_s",
            "run_stop_s",
            "time_from_run_start_s",
        ],
    )
    if reward_events.empty:
        return reward_events, skipped_outside_run_epochs

    reward_events["well"] = pd.Categorical(
        reward_events["well"],
        categories=list(WELL_ORDER),
        ordered=True,
    )
    reward_events["epoch"] = pd.Categorical(
        reward_events["epoch"],
        categories=[epoch for epoch, _, _ in run_epoch_bounds],
        ordered=True,
    )
    reward_events = reward_events.sort_values(
        ["epoch", "well", "reward_onset_s"],
        kind="stable",
    ).reset_index(drop=True)
    return reward_events, skipped_outside_run_epochs


def summarize_reward_counts(
    reward_events: pd.DataFrame,
    run_epochs: list[str],
) -> pd.DataFrame:
    """Return total and per-run reward counts by well."""
    all_index = pd.Index(WELL_ORDER, name="well")
    totals = (
        reward_events.groupby("well", observed=False)
        .size()
        .reindex(all_index, fill_value=0)
        .rename("n_rewards")
        .reset_index()
    )
    totals.insert(0, "epoch", "all")

    per_epoch_index = pd.MultiIndex.from_product(
        [run_epochs, WELL_ORDER],
        names=["epoch", "well"],
    )
    per_epoch = (
        reward_events.groupby(["epoch", "well"], observed=False)
        .size()
        .reindex(per_epoch_index, fill_value=0)
        .rename("n_rewards")
        .reset_index()
    )

    return pd.concat([totals, per_epoch], ignore_index=True)


def save_parquet_table(table: pd.DataFrame, output_path: Path) -> Path:
    """Write one DataFrame to parquet with a descriptive engine error."""
    try:
        table.to_parquet(output_path, index=False)
    except ImportError as exc:
        raise ImportError(
            "Saving parquet outputs requires `pyarrow` or `fastparquet` to be installed."
        ) from exc
    return output_path


def plot_reward_timing_by_run(
    reward_events: pd.DataFrame,
    run_epoch_bounds: list[tuple[str, float, float]],
    output_path: Path,
) -> Path:
    """Plot reward timing in one panel per run epoch."""
    import matplotlib.pyplot as plt

    figure_height = max(2.5, 1.8 * len(run_epoch_bounds) + 0.8)
    figure, axes = plt.subplots(
        len(run_epoch_bounds),
        1,
        figsize=(12, figure_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]

    for axis, (epoch, start_time, stop_time) in zip(axes, run_epoch_bounds, strict=True):
        epoch_events = reward_events.loc[reward_events["epoch"] == epoch]
        event_positions = [
            epoch_events.loc[epoch_events["well"] == well, "time_from_run_start_s"]
            .to_numpy(dtype=float)
            / 60.0
            for well in WELL_ORDER
        ]
        axis.eventplot(
            event_positions,
            lineoffsets=np.arange(len(WELL_ORDER), dtype=float),
            linelengths=0.75,
            linewidths=1.5,
            colors=[WELL_COLORS[well] for well in WELL_ORDER],
        )
        axis.set_yticks(np.arange(len(WELL_ORDER), dtype=float))
        axis.set_yticklabels([well.title() for well in WELL_ORDER])
        axis.set_ylim(-0.75, len(WELL_ORDER) - 0.25)

        run_duration_min = max((stop_time - start_time) / 60.0, 1e-6)
        axis.set_xlim(0.0, run_duration_min)
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(f"{epoch} reward timing ({run_duration_min:.1f} min)")

    axes[-1].set_xlabel("Time from run start (min)")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_reward_timing(
    animal_name: str,
    date: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    nwb_root: Path = DEFAULT_NWB_ROOT,
) -> None:
    """Load one NWB session and save reward timing outputs under analysis."""
    import pynwb

    analysis_path = get_analysis_path(
        animal_name=animal_name,
        date=date,
        data_root=data_root,
    )
    output_dir = analysis_path / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)

    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    print(f"Processing {animal_name} {date}.")
    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        run_epoch_bounds = get_run_epoch_bounds(nwbfile)
        reward_intervals_by_well = load_reward_intervals_by_well(nwbfile)

    reward_events, skipped_outside_run_epochs = build_reward_events_table(
        reward_intervals_by_well=reward_intervals_by_well,
        run_epoch_bounds=run_epoch_bounds,
    )
    run_epochs = [epoch for epoch, _, _ in run_epoch_bounds]
    reward_counts = summarize_reward_counts(reward_events, run_epochs=run_epochs)

    events_path = save_parquet_table(reward_events, output_dir / "reward_events.parquet")
    counts_path = save_parquet_table(reward_counts, output_dir / "reward_counts.parquet")
    figure_path = plot_reward_timing_by_run(
        reward_events=reward_events,
        run_epoch_bounds=run_epoch_bounds,
        output_path=output_dir / "reward_timing_by_run.png",
    )

    totals = reward_counts.loc[reward_counts["epoch"] == "all", ["well", "n_rewards"]]
    total_counts_by_well = {
        str(row.well): int(row.n_rewards)
        for row in totals.itertuples(index=False)
    }

    print("Total rewards by well:")
    for well in WELL_ORDER:
        print(f"  {well}: {total_counts_by_well.get(well, 0)}")
    print(f"Saved reward events to {events_path}")
    print(f"Saved reward counts to {counts_path}")
    print(f"Saved reward timing figure to {figure_path}")

    log_path = write_run_log(
        analysis_path=analysis_path,
        script_name="v1ca1.behavior.plot_reward_timing",
        parameters={
            "animal_name": animal_name,
            "date": date,
            "data_root": data_root,
            "nwb_root": nwb_root,
        },
        outputs={
            "run_epochs": run_epochs,
            "reward_events_path": events_path,
            "reward_counts_path": counts_path,
            "reward_timing_figure_path": figure_path,
            "reward_counts_by_well": total_counts_by_well,
            "skipped_rewards_outside_run_epochs": skipped_outside_run_epochs,
        },
    )
    print(f"Saved run metadata to {log_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for reward timing plotting."""
    parser = argparse.ArgumentParser(description="Plot reward timing from NWB pump events")
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
    return parser.parse_args()


def main() -> None:
    """Run the reward timing plotting CLI."""
    args = parse_arguments()
    plot_reward_timing(
        animal_name=args.animal_name,
        date=args.date,
        data_root=args.data_root,
        nwb_root=args.nwb_root,
    )


if __name__ == "__main__":
    main()
