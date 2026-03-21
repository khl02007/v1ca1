from __future__ import annotations

"""Plot reward timing from NWB pump events for one session.

This script loads one session NWB file, extracts left/center/right reward pump
intervals from ``processing["behavior"].data_interfaces["behavioral_events"]``
and aligns reward onsets to run epochs reconstructed from the
``timestamps_ephys`` outputs written by ``v1ca1.helper.get_timestamps``. It
saves run-by-run reward timing and trial-performance figures plus parquet
summary tables under the session analysis directory, with figures written under
`analysis_path / "figs" / "behavior"`.

Rewards are defined by pump onset (the rising edge from 0 to 1), since the
pumps are only active while reward is dispensed. The saved event table includes
per-reward onset/offset timestamps and pump durations for quality control, and
`reward_counts.parquet` stores both all-session totals and per-run counts by
well. Trial performance is reconstructed from poke-defined inbound and outbound
transitions, then marked rewarded when a pump onset occurs at the destination
well after arrival and before the next poke in that run epoch. The reward-timing
figure also overlays unrewarded first pokes at each well, defined as same-well
attempts that follow a pause threshold and do not receive reward before the next
poke in that epoch.
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from v1ca1.helper.get_trajectory_times import (
    get_behavioral_events_interface,
    load_epoch_time_bounds,
    load_poke_times,
)
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
POKE_EVENT_NAMES = {
    "left": "Poke Left",
    "center": "Poke Center",
    "right": "Poke Right",
}
WELL_ORDER = ("left", "center", "right")
WELL_COLORS = {
    "left": "tab:blue",
    "center": "tab:green",
    "right": "tab:orange",
}
PERFORMANCE_DIRECTIONS = ("outbound", "inbound")
PERFORMANCE_WINDOW_TRIALS = 8
PERFORMANCE_COLORS = {
    "outbound": "tab:purple",
    "inbound": "tab:red",
}
UNREWARDED_ATTEMPT_THRESHOLD_S = 5.0


def get_run_epoch_bounds(
    analysis_path: Path,
) -> tuple[list[tuple[str, float, float]], str]:
    """Return ordered `(epoch, start, stop)` tuples for all run epochs."""
    epoch_tags, epoch_bounds, timestamp_source = load_epoch_time_bounds(analysis_path)
    run_epochs = get_run_epochs(epoch_tags)
    run_epoch_set = set(run_epochs)
    run_epoch_bounds = [
        (epoch, float(epoch_bounds[epoch][0]), float(epoch_bounds[epoch][1]))
        for epoch in epoch_tags
        if epoch in run_epoch_set
    ]
    return run_epoch_bounds, timestamp_source


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


def build_unrewarded_attempts_table(
    poke_times_by_well: dict[str, np.ndarray],
    reward_events: pd.DataFrame,
    run_epoch_bounds: list[tuple[str, float, float]],
    same_well_interpoke_threshold_s: float = UNREWARDED_ATTEMPT_THRESHOLD_S,
) -> pd.DataFrame:
    """Return unrewarded first pokes to reward wells within each run epoch."""
    rows: list[dict[str, float | str]] = []

    reward_lookup: dict[tuple[str, str], np.ndarray] = {}
    if not reward_events.empty:
        for (epoch, well), reward_group in reward_events.groupby(["epoch", "well"], observed=False):
            reward_lookup[(str(epoch), str(well))] = reward_group["reward_onset_s"].to_numpy(
                dtype=float
            )

    for epoch, epoch_start, epoch_stop in run_epoch_bounds:
        epoch_poke_times_by_well = {
            well: poke_times_by_well[well][
                (poke_times_by_well[well] >= epoch_start) & (poke_times_by_well[well] < epoch_stop)
            ]
            for well in WELL_ORDER
        }
        all_epoch_pokes = np.concatenate(list(epoch_poke_times_by_well.values()))
        if all_epoch_pokes.size == 0:
            continue
        all_epoch_pokes = np.sort(all_epoch_pokes.astype(float, copy=False))

        for well in WELL_ORDER:
            well_poke_times = np.asarray(epoch_poke_times_by_well[well], dtype=float)
            if well_poke_times.size < 2:
                continue

            same_well_interpoke = np.diff(well_poke_times)
            candidate_indices = np.flatnonzero(
                same_well_interpoke >= same_well_interpoke_threshold_s
            ) + 1
            if candidate_indices.size == 0:
                continue

            reward_onsets = reward_lookup.get((epoch, well), np.array([], dtype=float))
            for candidate_index in candidate_indices:
                attempt_poke_s = float(well_poke_times[candidate_index])
                same_well_interpoke_s = float(same_well_interpoke[candidate_index - 1])
                next_poke_index = np.searchsorted(all_epoch_pokes, attempt_poke_s, side="right")
                attempt_window_stop_s = (
                    float(all_epoch_pokes[next_poke_index])
                    if next_poke_index < all_epoch_pokes.size
                    else float(epoch_stop)
                )
                rewarded_mask = (reward_onsets >= attempt_poke_s) & (reward_onsets < attempt_window_stop_s)
                if np.any(rewarded_mask):
                    continue

                rows.append(
                    {
                        "epoch": epoch,
                        "well": well,
                        "attempt_poke_s": attempt_poke_s,
                        "run_start_s": float(epoch_start),
                        "run_stop_s": float(epoch_stop),
                        "time_from_run_start_s": float(attempt_poke_s - epoch_start),
                        "same_well_interpoke_s": same_well_interpoke_s,
                        "attempt_window_stop_s": attempt_window_stop_s,
                    }
                )

    unrewarded_attempts = pd.DataFrame.from_records(
        rows,
        columns=[
            "epoch",
            "well",
            "attempt_poke_s",
            "run_start_s",
            "run_stop_s",
            "time_from_run_start_s",
            "same_well_interpoke_s",
            "attempt_window_stop_s",
        ],
    )
    if unrewarded_attempts.empty:
        return unrewarded_attempts

    unrewarded_attempts["well"] = pd.Categorical(
        unrewarded_attempts["well"],
        categories=list(WELL_ORDER),
        ordered=True,
    )
    unrewarded_attempts["epoch"] = pd.Categorical(
        unrewarded_attempts["epoch"],
        categories=[epoch for epoch, _, _ in run_epoch_bounds],
        ordered=True,
    )
    unrewarded_attempts = unrewarded_attempts.sort_values(
        ["epoch", "well", "attempt_poke_s"],
        kind="stable",
    ).reset_index(drop=True)
    return unrewarded_attempts


def build_trial_performance_table(
    poke_times_by_well: dict[str, np.ndarray],
    reward_events: pd.DataFrame,
    run_epoch_bounds: list[tuple[str, float, float]],
    performance_window_trials: int = PERFORMANCE_WINDOW_TRIALS,
) -> pd.DataFrame:
    """Return per-epoch inbound/outbound trial outcomes and rolling performance."""
    rows: list[dict[str, float | int | str | bool]] = []

    reward_lookup: dict[tuple[str, str], np.ndarray] = {}
    if not reward_events.empty:
        for (epoch, well), reward_group in reward_events.groupby(["epoch", "well"], observed=False):
            reward_lookup[(str(epoch), str(well))] = reward_group["reward_onset_s"].to_numpy(
                dtype=float
            )

    for epoch, epoch_start, epoch_stop in run_epoch_bounds:
        epoch_poke_times: list[np.ndarray] = []
        epoch_poke_labels: list[np.ndarray] = []
        for well in WELL_ORDER:
            well_poke_times = poke_times_by_well[well]
            in_epoch = (well_poke_times >= epoch_start) & (well_poke_times < epoch_stop)
            epoch_poke_times.append(well_poke_times[in_epoch])
            epoch_poke_labels.append(np.full(in_epoch.sum(), well, dtype=object))

        all_poke_times = np.concatenate(epoch_poke_times)
        all_poke_labels = np.concatenate(epoch_poke_labels)
        if all_poke_times.size < 2:
            continue

        poke_order = np.argsort(all_poke_times)
        all_poke_times = all_poke_times[poke_order]
        all_poke_labels = all_poke_labels[poke_order]

        for trial_index in range(all_poke_times.size - 1):
            start_well = str(all_poke_labels[trial_index])
            stop_well = str(all_poke_labels[trial_index + 1])
            if start_well == "center" and stop_well in {"left", "right"}:
                direction = "outbound"
            elif start_well in {"left", "right"} and stop_well == "center":
                direction = "inbound"
            else:
                continue

            trial_start_s = float(all_poke_times[trial_index])
            trial_stop_s = float(all_poke_times[trial_index + 1])
            reward_window_stop_s = (
                float(all_poke_times[trial_index + 2])
                if trial_index + 2 < all_poke_times.size
                else float(epoch_stop)
            )
            reward_onsets = reward_lookup.get((epoch, stop_well), np.array([], dtype=float))
            matched_reward_onsets = reward_onsets[
                (reward_onsets >= trial_stop_s) & (reward_onsets < reward_window_stop_s)
            ]

            rows.append(
                {
                    "epoch": epoch,
                    "direction": direction,
                    "start_well": start_well,
                    "stop_well": stop_well,
                    "trial_start_s": trial_start_s,
                    "trial_stop_s": trial_stop_s,
                    "reward_window_stop_s": reward_window_stop_s,
                    "rewarded": bool(matched_reward_onsets.size),
                    "n_rewards_in_window": int(matched_reward_onsets.size),
                    "matched_reward_onset_s": (
                        float(matched_reward_onsets[0]) if matched_reward_onsets.size else np.nan
                    ),
                }
            )

    trial_performance = pd.DataFrame.from_records(
        rows,
        columns=[
            "epoch",
            "direction",
            "start_well",
            "stop_well",
            "trial_start_s",
            "trial_stop_s",
            "reward_window_stop_s",
            "rewarded",
            "n_rewards_in_window",
            "matched_reward_onset_s",
        ],
    )
    if trial_performance.empty:
        return trial_performance

    trial_performance["epoch"] = pd.Categorical(
        trial_performance["epoch"],
        categories=[epoch for epoch, _, _ in run_epoch_bounds],
        ordered=True,
    )
    trial_performance["direction"] = pd.Categorical(
        trial_performance["direction"],
        categories=list(PERFORMANCE_DIRECTIONS),
        ordered=True,
    )
    trial_performance = trial_performance.sort_values(
        ["epoch", "trial_start_s", "trial_stop_s"],
        kind="stable",
    ).reset_index(drop=True)
    trial_performance["trial_number"] = (
        trial_performance.groupby("epoch", observed=False).cumcount() + 1
    )
    trial_performance["direction_trial_number"] = (
        trial_performance.groupby(["epoch", "direction"], observed=False).cumcount() + 1
    )
    trial_performance["performance_sliding_avg"] = trial_performance.groupby(
        ["epoch", "direction"],
        observed=False,
    )["rewarded"].transform(
        lambda outcomes: outcomes.astype(float)
        .rolling(window=performance_window_trials, min_periods=1)
        .mean()
    )
    trial_performance["performance_window_trials"] = int(performance_window_trials)
    return trial_performance


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
    unrewarded_attempts: pd.DataFrame,
    run_epoch_bounds: list[tuple[str, float, float]],
    output_path: Path,
    same_well_interpoke_threshold_s: float = UNREWARDED_ATTEMPT_THRESHOLD_S,
) -> Path:
    """Plot rewarded pump onsets and unrewarded first pokes in one panel per run epoch."""
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
        epoch_attempts = unrewarded_attempts.loc[unrewarded_attempts["epoch"] == epoch]
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
        for well_index, well in enumerate(WELL_ORDER):
            attempt_positions = (
                epoch_attempts.loc[epoch_attempts["well"] == well, "time_from_run_start_s"]
                .to_numpy(dtype=float)
                / 60.0
            )
            if attempt_positions.size == 0:
                continue
            axis.scatter(
                attempt_positions,
                np.full(attempt_positions.shape, float(well_index)),
                marker="x",
                s=42,
                linewidths=1.5,
                color=WELL_COLORS[well],
                zorder=3,
            )

        axis.set_yticks(np.arange(len(WELL_ORDER), dtype=float))
        axis.set_yticklabels([well.title() for well in WELL_ORDER])
        axis.set_ylim(-0.75, len(WELL_ORDER) - 0.25)

        run_duration_min = max((stop_time - start_time) / 60.0, 1e-6)
        axis.set_xlim(0.0, run_duration_min)
        axis.grid(axis="x", alpha=0.25)
        axis.set_title(
            f"{epoch} reward timing ({run_duration_min:.1f} min, unrewarded threshold {same_well_interpoke_threshold_s:.0f} s)"
        )

    axes[-1].set_xlabel("Time from run start (min)")
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_trial_performance(
    trial_performance: pd.DataFrame,
    run_epoch_bounds: list[tuple[str, float, float]],
    output_path: Path,
    performance_window_trials: int = PERFORMANCE_WINDOW_TRIALS,
) -> Path:
    """Plot inbound and outbound rolling reward performance in one panel per epoch."""
    import matplotlib.pyplot as plt

    figure_height = max(2.5, 1.8 * len(run_epoch_bounds) + 0.8)
    figure, axes = plt.subplots(
        len(run_epoch_bounds),
        1,
        figsize=(10, figure_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[:, 0]

    for axis, (epoch, _, _) in zip(axes, run_epoch_bounds, strict=True):
        epoch_trials = trial_performance.loc[trial_performance["epoch"] == epoch]
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("Performance")
        axis.grid(alpha=0.25)
        axis.set_title(
            f"{epoch} performance ({performance_window_trials}-trial sliding average)"
        )

        if epoch_trials.empty:
            axis.text(
                0.5,
                0.5,
                "No trials",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
            continue

        x_max = 1.0
        for direction in PERFORMANCE_DIRECTIONS:
            direction_trials = epoch_trials.loc[epoch_trials["direction"] == direction]
            if direction_trials.empty:
                continue
            x_values = direction_trials["direction_trial_number"].to_numpy(dtype=float)
            rewarded = direction_trials["rewarded"].to_numpy(dtype=float)
            performance = direction_trials["performance_sliding_avg"].to_numpy(dtype=float)
            color = PERFORMANCE_COLORS[direction]
            axis.scatter(
                x_values,
                rewarded,
                color=color,
                s=18,
                alpha=0.25,
                zorder=2,
            )
            axis.plot(
                x_values,
                performance,
                color=color,
                linewidth=2.0,
                label=direction.title(),
                zorder=3,
            )
            x_max = max(x_max, float(x_values[-1]))

        axis.set_xlim(0.5, x_max + 0.5)
        axis.legend(loc="lower right")

    axes[-1].set_xlabel("Trial number")
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
    fig_dir = analysis_path / "figs" / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    nwb_path = nwb_root / f"{animal_name}{date}.nwb"
    if not nwb_path.exists():
        raise FileNotFoundError(f"NWB file not found: {nwb_path}")

    try:
        run_epoch_bounds, timestamp_source = get_run_epoch_bounds(analysis_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not load epoch bounds from helper timestamp outputs under "
            f"{analysis_path}. Run `python -m v1ca1.helper.get_timestamps` for this "
            "session before plotting reward timing."
        ) from exc

    print(f"Processing {animal_name} {date}.")
    with pynwb.NWBHDF5IO(nwb_path, "r") as io:
        nwbfile = io.read()
        reward_intervals_by_well = load_reward_intervals_by_well(nwbfile)
        poke_times_by_well = {
            well: load_poke_times(nwbfile, event_name)
            for well, event_name in POKE_EVENT_NAMES.items()
        }

    reward_events, skipped_outside_run_epochs = build_reward_events_table(
        reward_intervals_by_well=reward_intervals_by_well,
        run_epoch_bounds=run_epoch_bounds,
    )
    run_epochs = [epoch for epoch, _, _ in run_epoch_bounds]
    reward_counts = summarize_reward_counts(reward_events, run_epochs=run_epochs)
    unrewarded_attempts = build_unrewarded_attempts_table(
        poke_times_by_well=poke_times_by_well,
        reward_events=reward_events,
        run_epoch_bounds=run_epoch_bounds,
    )
    trial_performance = build_trial_performance_table(
        poke_times_by_well=poke_times_by_well,
        reward_events=reward_events,
        run_epoch_bounds=run_epoch_bounds,
    )

    events_path = save_parquet_table(reward_events, output_dir / "reward_events.parquet")
    counts_path = save_parquet_table(reward_counts, output_dir / "reward_counts.parquet")
    unrewarded_attempts_path = save_parquet_table(
        unrewarded_attempts,
        output_dir / "unrewarded_attempts.parquet",
    )
    trial_performance_path = save_parquet_table(
        trial_performance,
        output_dir / "trial_performance.parquet",
    )
    figure_path = plot_reward_timing_by_run(
        reward_events=reward_events,
        unrewarded_attempts=unrewarded_attempts,
        run_epoch_bounds=run_epoch_bounds,
        output_path=fig_dir / "reward_timing_by_run.png",
    )
    performance_figure_path = plot_trial_performance(
        trial_performance=trial_performance,
        run_epoch_bounds=run_epoch_bounds,
        output_path=fig_dir / "trial_performance.png",
    )

    totals = reward_counts.loc[reward_counts["epoch"] == "all", ["well", "n_rewards"]]
    total_counts_by_well = {
        str(row.well): int(row.n_rewards)
        for row in totals.itertuples(index=False)
    }
    attempt_counts_by_well = (
        unrewarded_attempts.groupby("well", observed=False)
        .size()
        .reindex(pd.Index(WELL_ORDER, name="well"), fill_value=0)
        .astype(int)
        .to_dict()
        if not unrewarded_attempts.empty
        else {well: 0 for well in WELL_ORDER}
    )

    print("Total rewards by well:")
    for well in WELL_ORDER:
        print(f"  {well}: {total_counts_by_well.get(well, 0)}")
    print("Total unrewarded first pokes by well:")
    for well in WELL_ORDER:
        print(f"  {well}: {attempt_counts_by_well.get(well, 0)}")
    print(f"Saved reward events to {events_path}")
    print(f"Saved reward counts to {counts_path}")
    print(f"Saved unrewarded attempts to {unrewarded_attempts_path}")
    print(f"Saved trial performance to {trial_performance_path}")
    print(f"Saved reward timing figure to {figure_path}")
    print(f"Saved trial performance figure to {performance_figure_path}")

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
            "epoch_bounds_source": timestamp_source,
            "reward_events_path": events_path,
            "reward_counts_path": counts_path,
            "unrewarded_attempts_path": unrewarded_attempts_path,
            "trial_performance_path": trial_performance_path,
            "reward_timing_figure_path": figure_path,
            "trial_performance_figure_path": performance_figure_path,
            "reward_counts_by_well": total_counts_by_well,
            "unrewarded_attempt_counts_by_well": attempt_counts_by_well,
            "skipped_rewards_outside_run_epochs": skipped_outside_run_epochs,
            "unrewarded_attempts_threshold_s": UNREWARDED_ATTEMPT_THRESHOLD_S,
            "performance_window_trials": PERFORMANCE_WINDOW_TRIALS,
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
