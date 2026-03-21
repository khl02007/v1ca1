from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from v1ca1.behavior.plot_reward_timing import (
    build_reward_events_table,
    build_trial_performance_table,
    build_unrewarded_attempts_table,
    get_run_epoch_bounds,
    load_binary_event_intervals,
    load_reward_intervals_by_well,
    plot_reward_timing_by_run,
    plot_trial_performance,
    summarize_reward_counts,
)


@dataclass
class FakeTimeSeries:
    data: np.ndarray
    timestamps: np.ndarray


@dataclass
class FakeBehavioralEvents:
    time_series: dict[str, FakeTimeSeries]


@dataclass
class FakeProcessingModule:
    data_interfaces: dict[str, FakeBehavioralEvents]


@dataclass
class FakeNWBFile:
    processing: dict[str, FakeProcessingModule]


def test_get_run_epoch_bounds_uses_helper_timestamp_outputs(tmp_path) -> None:
    timestamps_ephys = {
        "01_s1": np.array([0.0, 0.1, 0.2]),
        "02_r1": np.array([1.0, 1.1, 1.2]),
        "03_s2": np.array([2.0, 2.1, 2.2]),
        "04_r2": np.array([3.0, 3.2, 3.4]),
    }
    with open(tmp_path / "timestamps_ephys.pkl", "wb") as file:
        pickle.dump(timestamps_ephys, file)

    run_epoch_bounds, timestamp_source = get_run_epoch_bounds(tmp_path)

    assert timestamp_source == "pickle"
    assert run_epoch_bounds == [
        ("02_r1", 1.0, 1.2),
        ("04_r2", 3.0, 3.4),
    ]


def test_load_binary_event_intervals_uses_rising_edges() -> None:
    behavioral_events = FakeBehavioralEvents(
        time_series={
            "Pump Left": FakeTimeSeries(
                data=np.array([0, 1, 1, 0, 0, 1, 0]),
                timestamps=np.array([0.0, 1.0, 1.1, 1.3, 1.5, 2.0, 2.2]),
            )
        }
    )

    onset_times, offset_times = load_binary_event_intervals(behavioral_events, "Pump Left")

    assert np.allclose(onset_times, [1.0, 2.0])
    assert np.allclose(offset_times, [1.3, 2.2])
    assert np.allclose(offset_times - onset_times, [0.3, 0.2])


def test_load_reward_intervals_by_well_requires_all_pump_series() -> None:
    nwbfile = FakeNWBFile(
        processing={
            "behavior": FakeProcessingModule(
                data_interfaces={
                    "behavioral_events": FakeBehavioralEvents(
                        time_series={
                            "Pump Left": FakeTimeSeries(
                                data=np.array([0, 1, 0]),
                                timestamps=np.array([0.0, 0.1, 0.2]),
                            ),
                            "Pump Center": FakeTimeSeries(
                                data=np.array([0, 1, 0]),
                                timestamps=np.array([0.0, 0.1, 0.2]),
                            ),
                        }
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="Pump Right") as excinfo:
        load_reward_intervals_by_well(nwbfile)

    assert "Pump Left" in str(excinfo.value)
    assert "Pump Center" in str(excinfo.value)


def test_build_reward_events_table_assigns_only_run_epochs() -> None:
    reward_intervals_by_well = {
        "left": (
            np.array([0.5, 1.5, 5.5]),
            np.array([0.7, 1.7, 5.7]),
        ),
        "center": (
            np.array([2.0]),
            np.array([2.2]),
        ),
        "right": (
            np.array([4.5]),
            np.array([4.7]),
        ),
    }
    run_epoch_bounds = [
        ("01_r1", 1.0, 3.0),
        ("03_r2", 4.0, 5.0),
    ]

    reward_events, skipped = build_reward_events_table(
        reward_intervals_by_well=reward_intervals_by_well,
        run_epoch_bounds=run_epoch_bounds,
    )

    assert reward_events[["epoch", "well"]].to_dict("records") == [
        {"epoch": "01_r1", "well": "left"},
        {"epoch": "01_r1", "well": "center"},
        {"epoch": "03_r2", "well": "right"},
    ]
    assert np.allclose(reward_events["time_from_run_start_s"], [0.5, 1.0, 0.5])
    assert skipped == {"left": 2, "center": 0, "right": 0}


def test_summarize_reward_counts_includes_totals_and_per_epoch() -> None:
    reward_events = pd.DataFrame(
        {
            "epoch": pd.Categorical(
                ["01_r1", "01_r1", "03_r2"],
                categories=["01_r1", "03_r2"],
                ordered=True,
            ),
            "well": pd.Categorical(
                ["left", "center", "right"],
                categories=["left", "center", "right"],
                ordered=True,
            ),
            "reward_onset_s": [1.5, 2.0, 4.5],
            "reward_offset_s": [1.7, 2.2, 4.7],
            "pump_duration_s": [0.2, 0.2, 0.2],
            "run_start_s": [1.0, 1.0, 4.0],
            "run_stop_s": [3.0, 3.0, 5.0],
            "time_from_run_start_s": [0.5, 1.0, 0.5],
        }
    )

    counts = summarize_reward_counts(reward_events, run_epochs=["01_r1", "03_r2"])

    assert counts.loc[counts["epoch"] == "all", "n_rewards"].tolist() == [1, 1, 1]
    assert counts.loc[
        (counts["epoch"] == "01_r1") & (counts["well"] == "right"),
        "n_rewards",
    ].item() == 0
    assert counts.loc[
        (counts["epoch"] == "03_r2") & (counts["well"] == "right"),
        "n_rewards",
    ].item() == 1


def test_build_unrewarded_attempts_table_detects_first_pokes_after_pause() -> None:
    poke_times_by_well = {
        "left": np.array([1.0, 2.0, 6.5, 7.0, 10.0]),
        "center": np.array([], dtype=float),
        "right": np.array([3.0, 8.0]),
    }
    reward_events = pd.DataFrame(
        {
            "epoch": pd.Categorical(
                ["01_r1", "01_r1"],
                categories=["01_r1"],
                ordered=True,
            ),
            "well": pd.Categorical(
                ["left", "right"],
                categories=["left", "center", "right"],
                ordered=True,
            ),
            "reward_onset_s": [6.7, 10.5],
            "reward_offset_s": [6.8, 10.6],
            "pump_duration_s": [0.1, 0.1],
            "run_start_s": [0.0, 0.0],
            "run_stop_s": [12.0, 12.0],
            "time_from_run_start_s": [6.7, 10.5],
        }
    )

    unrewarded_attempts = build_unrewarded_attempts_table(
        poke_times_by_well=poke_times_by_well,
        reward_events=reward_events,
        run_epoch_bounds=[("01_r1", 0.0, 12.0)],
    )

    assert unrewarded_attempts[["epoch", "well"]].to_dict("records") == [
        {"epoch": "01_r1", "well": "left"},
        {"epoch": "01_r1", "well": "right"},
    ]
    assert np.allclose(unrewarded_attempts["attempt_poke_s"], [10.0, 8.0])
    assert np.allclose(unrewarded_attempts["same_well_interpoke_s"], [3.0, 5.0])
    assert np.allclose(unrewarded_attempts["attempt_window_stop_s"], [12.0, 10.0])
    assert np.allclose(unrewarded_attempts["time_from_run_start_s"], [10.0, 8.0])


def test_build_trial_performance_table_matches_rewards_and_ignores_invalid_transitions() -> None:
    poke_times_by_well = {
        "left": np.array([2.0, 5.0]),
        "center": np.array([1.0, 3.0, 6.0]),
        "right": np.array([4.0]),
    }
    reward_events = pd.DataFrame(
        {
            "epoch": pd.Categorical(
                ["01_r1", "01_r1", "01_r1"],
                categories=["01_r1"],
                ordered=True,
            ),
            "well": pd.Categorical(
                ["left", "center", "right"],
                categories=["left", "center", "right"],
                ordered=True,
            ),
            "reward_onset_s": [2.2, 3.4, 4.4],
            "reward_offset_s": [2.4, 3.6, 4.6],
            "pump_duration_s": [0.2, 0.2, 0.2],
            "run_start_s": [0.0, 0.0, 0.0],
            "run_stop_s": [7.0, 7.0, 7.0],
            "time_from_run_start_s": [2.2, 3.4, 4.4],
        }
    )

    trial_performance = build_trial_performance_table(
        poke_times_by_well=poke_times_by_well,
        reward_events=reward_events,
        run_epoch_bounds=[("01_r1", 0.0, 7.0)],
    )

    assert trial_performance[["direction", "start_well", "stop_well"]].to_dict("records") == [
        {"direction": "outbound", "start_well": "center", "stop_well": "left"},
        {"direction": "inbound", "start_well": "left", "stop_well": "center"},
        {"direction": "outbound", "start_well": "center", "stop_well": "right"},
        {"direction": "inbound", "start_well": "left", "stop_well": "center"},
    ]
    assert trial_performance["rewarded"].tolist() == [True, True, True, False]
    assert trial_performance["n_rewards_in_window"].tolist() == [1, 1, 1, 0]
    assert trial_performance["trial_number"].tolist() == [1, 2, 3, 4]
    assert trial_performance["direction_trial_number"].tolist() == [1, 1, 2, 2]
    assert np.allclose(
        trial_performance["matched_reward_onset_s"].to_numpy(dtype=float)[:3],
        [2.2, 3.4, 4.4],
    )
    assert np.isnan(trial_performance.loc[3, "matched_reward_onset_s"])
    assert np.allclose(trial_performance["performance_sliding_avg"], [1.0, 1.0, 1.0, 0.5])


def test_build_trial_performance_table_uses_epoch_specific_rolling_average() -> None:
    poke_times_by_well = {
        "left": np.array([1.0, 3.0, 5.0, 7.0, 21.0, 23.0]),
        "center": np.array([0.0, 2.0, 4.0, 6.0, 20.0, 22.0, 24.0]),
        "right": np.array([], dtype=float),
    }
    reward_events = pd.DataFrame(
        {
            "epoch": pd.Categorical(
                ["01_r1", "01_r1", "01_r1", "02_r2", "02_r2"],
                categories=["01_r1", "02_r2"],
                ordered=True,
            ),
            "well": pd.Categorical(
                ["left", "center", "center", "left", "center"],
                categories=["left", "center", "right"],
                ordered=True,
            ),
            "reward_onset_s": [1.2, 2.2, 6.2, 21.2, 22.2],
            "reward_offset_s": [1.3, 2.3, 6.3, 21.3, 22.3],
            "pump_duration_s": [0.1, 0.1, 0.1, 0.1, 0.1],
            "run_start_s": [0.0, 0.0, 0.0, 20.0, 20.0],
            "run_stop_s": [8.0, 8.0, 8.0, 25.0, 25.0],
            "time_from_run_start_s": [1.2, 2.2, 6.2, 1.2, 2.2],
        }
    )

    trial_performance = build_trial_performance_table(
        poke_times_by_well=poke_times_by_well,
        reward_events=reward_events,
        run_epoch_bounds=[("01_r1", 0.0, 8.0), ("02_r2", 20.0, 25.0)],
    )

    outbound = trial_performance.loc[trial_performance["direction"] == "outbound"]
    inbound = trial_performance.loc[trial_performance["direction"] == "inbound"]

    assert outbound["trial_number"].tolist() == [1, 2, 3, 1, 2]
    assert inbound["trial_number"].tolist() == [2, 4, 1, 3]
    assert outbound["direction_trial_number"].tolist() == [1, 2, 3, 1, 2]
    assert inbound["direction_trial_number"].tolist() == [1, 2, 1, 2]
    assert np.allclose(outbound["performance_sliding_avg"], [1.0, 0.5, 2 / 3, 1.0, 0.5])
    assert np.allclose(inbound["performance_sliding_avg"], [1.0, 1.0, 1.0, 0.5])


def test_plot_reward_timing_by_run_handles_empty_inputs(tmp_path) -> None:
    reward_events = pd.DataFrame(
        columns=[
            "epoch",
            "well",
            "reward_onset_s",
            "reward_offset_s",
            "pump_duration_s",
            "run_start_s",
            "run_stop_s",
            "time_from_run_start_s",
        ]
    )
    unrewarded_attempts = pd.DataFrame(
        columns=[
            "epoch",
            "well",
            "attempt_poke_s",
            "run_start_s",
            "run_stop_s",
            "time_from_run_start_s",
            "same_well_interpoke_s",
            "attempt_window_stop_s",
        ]
    )

    output_path = plot_reward_timing_by_run(
        reward_events=reward_events,
        unrewarded_attempts=unrewarded_attempts,
        run_epoch_bounds=[("01_r1", 0.0, 1.0)],
        output_path=tmp_path / "reward_timing_by_run.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_trial_performance_handles_empty_table(tmp_path) -> None:
    trial_performance = pd.DataFrame(
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
            "trial_number",
            "direction_trial_number",
            "performance_sliding_avg",
            "performance_window_trials",
        ]
    )

    output_path = plot_trial_performance(
        trial_performance,
        [("01_r1", 0.0, 1.0), ("02_r2", 1.0, 2.0)],
        tmp_path / "trial_performance.png",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
