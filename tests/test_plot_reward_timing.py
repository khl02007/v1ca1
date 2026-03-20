from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from v1ca1.behavior.plot_reward_timing import (
    build_reward_events_table,
    load_binary_event_intervals,
    load_reward_intervals_by_well,
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
