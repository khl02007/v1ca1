"""Tests for the database-free Spyglass stability adapter."""

from __future__ import annotations

import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass import stability


class _Position:
    """Minimal Pynapple-like position input."""

    def __init__(self) -> None:
        self.t = np.asarray([10.0, 10.1, 10.2], dtype=float)
        self.d = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=float,
        )


def test_stability_artifact_path_is_uuid_keyed_and_session_first(
    tmp_path: Path,
) -> None:
    stability_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    path = stability.get_stability_artifact_path(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        trajectory_type="center_to_left",
        region="v1",
        task_progression_stability_id=stability_id,
        artifact_root=tmp_path,
    )

    assert path == (
        tmp_path
        / "L14"
        / "20240611"
        / "task_progression_stability"
        / "02_r1"
        / "center_to_left"
        / "v1"
        / str(stability_id)
        / "stability.parquet"
    )

    with pytest.raises(ValueError, match="UUID"):
        stability.get_stability_artifact_path(
            animal_name="L14",
            date="20240611",
            epoch="02_r1",
            trajectory_type="center_to_left",
            region="v1",
            task_progression_stability_id="not-a-uuid",
            artifact_root=tmp_path,
        )


def test_attach_unit_identity_uses_ephemeral_group_key_mapping() -> None:
    table = pd.DataFrame(
        {
            "unit": [1, 0],
            "stability_correlation": [0.25, 0.75],
            "stability_status": ["valid", "valid"],
        }
    )
    spikes = {0: object(), 1: object()}
    stable_unit_ids = [
        {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        {"spikesorting_merge_id": "merge-b", "unit_id": 22},
    ]

    result = stability._attach_unit_identity(
        table,
        spikes=spikes,
        stable_unit_ids=stable_unit_ids,
    )

    assert result.columns[:4].tolist() == list(stability.IDENTITY_COLUMNS)
    assert result["group_unit_id"].tolist() == [1, 0]
    assert result["spikesorting_merge_id"].tolist() == ["merge-b", "merge-a"]
    assert result["unit_id"].tolist() == ["22", "11"]
    assert result["stable_unit_id"].tolist() == ["merge-b:22", "merge-a:11"]


def test_empty_stability_table_has_persistent_identity_schema() -> None:
    table = stability.empty_stability_table()

    assert table.empty
    assert list(table.columns[:4]) == list(stability.IDENTITY_COLUMNS)
    assert "unit" not in table.columns
    assert {
        "animal_name",
        "date",
        "region",
        "epoch",
        "trajectory_type",
        "stability_correlation",
        "stability_status",
    }.issubset(table.columns)


def test_build_task_progression_rejects_nonselected_graph_configuration() -> None:
    with pytest.raises(ValueError, match="configuration_name"):
        stability.build_task_progression_from_graph(
            position=_Position(),
            trajectory_interval=object(),
            graph_inputs={
                "configuration_name": "center_to_right",
                "coordinate_unit": "cm",
            },
            trajectory_type="center_to_left",
        )


def test_compute_selected_stability_attaches_identity_and_counts_valid_units(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    movement_interval = object()
    task_progression = object()
    trajectory_interval = object()
    spikes = {0: object(), 1: object()}

    def fake_build_speed_tsd(position, timestamps, **kwargs):
        calls["speed_position"] = np.asarray(position)
        calls["speed_timestamps"] = np.asarray(timestamps)
        calls["speed_kwargs"] = kwargs
        return "speed"

    def fake_build_movement_interval(speed, **kwargs):
        calls["movement_speed"] = speed
        calls["movement_kwargs"] = kwargs
        return movement_interval

    def fake_build_task_progression_from_graph(**kwargs):
        calls["graph_kwargs"] = kwargs
        return task_progression, 20.0

    def fake_movement_firing_rates(received_spikes, received_interval):
        assert received_spikes is spikes
        assert received_interval is movement_interval
        return pd.Series({0: 1.0, 1: 0.0}, dtype=float)

    def fake_compute_trajectory_stability_table(**kwargs):
        calls["compute_kwargs"] = kwargs
        return pd.DataFrame(
            {
                "animal_name": ["L14", "L14"],
                "date": ["20240611", "20240611"],
                "unit": [0, 1],
                "region": ["v1", "v1"],
                "epoch": ["02_r1", "02_r1"],
                "trajectory_type": ["center_to_left", "center_to_left"],
                "firing_rate_hz": [1.0, 0.0],
                "stability_correlation": [0.8, np.nan],
                "stability_status": ["valid", "no_even_spikes"],
            }
        )

    monkeypatch.setattr(stability, "build_speed_tsd", fake_build_speed_tsd)
    monkeypatch.setattr(
        stability,
        "build_movement_interval",
        fake_build_movement_interval,
    )
    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        fake_build_task_progression_from_graph,
    )
    monkeypatch.setattr(
        stability,
        "_movement_firing_rates",
        fake_movement_firing_rates,
    )
    monkeypatch.setattr(
        stability,
        "compute_trajectory_stability_table",
        fake_compute_trajectory_stability_table,
    )

    result = stability.compute_selected_stability(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        spikes=spikes,
        stable_unit_ids=[
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
            {"spikesorting_merge_id": "merge-b", "unit_id": 22},
        ],
        position=_Position(),
        trajectory_interval=trajectory_interval,
        graph_inputs={"configuration_name": "center_to_left"},
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.25,
        place_bin_size_cm=4.0,
    )

    assert result["analysis_status"] == "valid"
    assert result["n_units"] == 2
    assert result["n_valid_units"] == 1
    assert result["table"]["stable_unit_id"].tolist() == [
        "merge-a:11",
        "merge-b:22",
    ]
    assert calls["speed_kwargs"] == {
        "position_offset": 0,
        "speed_smoothing_sigma_s": 0.25,
    }
    assert calls["movement_kwargs"] == {"speed_threshold_cm_s": 4.0}
    compute_kwargs = calls["compute_kwargs"]
    assert compute_kwargs["trajectory_interval"] is trajectory_interval
    assert compute_kwargs["movement_interval"] is movement_interval
    assert compute_kwargs["task_progression"] is task_progression
    assert np.allclose(compute_kwargs["bins"], np.arange(0.0, 1.2, 0.2))


@pytest.mark.parametrize("spikes", [None, {}])
def test_compute_selected_stability_returns_terminal_no_units(spikes) -> None:
    result = stability.compute_selected_stability(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        spikes=spikes,
        stable_unit_ids=[],
        position=None,
        trajectory_interval=None,
        graph_inputs={},
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
        place_bin_size_cm=4.0,
    )

    assert result["analysis_status"] == "no_units"
    assert result["n_units"] == 0
    assert result["n_valid_units"] == 0
    assert result["table"].empty
    assert list(result["table"].columns[:4]) == list(stability.IDENTITY_COLUMNS)


def test_write_stability_artifact_is_atomic_and_refuses_implicit_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "nested" / "stability.parquet"
    original = pd.DataFrame({"unit_id": ["11"], "value": [1.0]})
    replacement = pd.DataFrame({"unit_id": ["22"], "value": [2.0]})

    assert stability.write_stability_artifact(original, path) == path
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert not list(path.parent.glob(f".{path.name}.*"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        stability.write_stability_artifact(replacement, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)

    real_replace = os.replace
    replace_count = 0

    def fail_new_artifact_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(stability.os, "replace", fail_new_artifact_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        stability.write_stability_artifact(replacement, path, overwrite=True)

    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert not list(path.parent.glob(f".{path.name}.*"))
