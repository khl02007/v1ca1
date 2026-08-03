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


class _Interval:
    """Minimal Pynapple-like interval input with a known duration."""

    def __init__(self, duration_s: float) -> None:
        self.duration_s = float(duration_s)

    def tot_length(self) -> float:
        return self.duration_s


STABLE_UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    {"spikesorting_merge_id": "merge-b", "unit_id": 22},
)


def _movement_firing_rate_table(
    *,
    status: str = "valid",
    duration_s: float = 2.0,
) -> pd.DataFrame:
    """Return one canonical two-unit upstream movement-rate artifact."""
    if status == "valid":
        spike_counts = [2, 0]
        firing_rates = [1.0, 0.0]
    else:
        spike_counts = [0, 0]
        firing_rates = [np.nan, np.nan]
    return pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge-a", "merge-b"],
            "unit_id": ["11", "22"],
            "stable_unit_id": ["merge-a:11", "merge-b:22"],
            "group_unit_id": [0, 1],
            "movement_spike_count": spike_counts,
            "movement_duration_s": [duration_s, duration_s],
            "movement_firing_rate_hz": firing_rates,
            "firing_rate_status": [status, status],
        }
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


def test_compute_selected_stability_rekeys_saved_rates_by_persistent_identity(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    movement_interval = _Interval(2.0)
    task_progression = object()
    trajectory_interval = object()
    spikes = {7: object(), 3: object()}

    def fake_build_task_progression_from_graph(**kwargs):
        calls["graph_kwargs"] = kwargs
        return task_progression, 20.0

    def fake_compute_trajectory_stability_table(**kwargs):
        calls["compute_kwargs"] = kwargs
        return pd.DataFrame(
            {
                "animal_name": ["L14", "L14"],
                "date": ["20240611", "20240611"],
                "unit": [7, 3],
                "region": ["v1", "v1"],
                "epoch": ["02_r1", "02_r1"],
                "trajectory_type": ["center_to_left", "center_to_left"],
                "firing_rate_hz": [1.0, 0.0],
                "stability_correlation": [0.8, np.nan],
                "stability_status": ["valid", "no_even_spikes"],
            }
        )

    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        fake_build_task_progression_from_graph,
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
        stable_unit_ids=STABLE_UNIT_IDS,
        position=_Position(),
        trajectory_interval=trajectory_interval,
        graph_inputs={"configuration_name": "center_to_left"},
        movement_interval=movement_interval,
        movement_firing_rate_table=_movement_firing_rate_table(),
        place_bin_size_cm=4.0,
    )

    assert result["analysis_status"] == "valid"
    assert result["n_units"] == 2
    assert result["n_valid_units"] == 1
    assert result["table"]["stable_unit_id"].tolist() == [
        "merge-a:11",
        "merge-b:22",
    ]
    assert result["table"]["group_unit_id"].tolist() == [7, 3]
    assert result["table"]["firing_rate_hz"].tolist() == [1.0, 0.0]
    compute_kwargs = calls["compute_kwargs"]
    assert compute_kwargs["trajectory_interval"] is trajectory_interval
    assert compute_kwargs["movement_interval"] is movement_interval
    assert compute_kwargs["task_progression"] is task_progression
    pd.testing.assert_series_equal(
        compute_kwargs["epoch_firing_rates"],
        pd.Series({7: 1.0, 3: 0.0}, dtype=float),
    )
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
        movement_interval=_Interval(0.0),
        movement_firing_rate_table=pd.DataFrame(
            columns=stability.MOVEMENT_RATE_COLUMNS
        ),
        place_bin_size_cm=4.0,
    )

    assert result["analysis_status"] == "no_units"
    assert result["n_units"] == 0
    assert result["n_valid_units"] == 0
    assert result["table"].empty
    assert list(result["table"].columns[:4]) == list(stability.IDENTITY_COLUMNS)


def test_no_unit_movement_artifact_requires_empty_interval() -> None:
    with pytest.raises(ValueError, match="empty movement interval"):
        stability.compute_selected_stability(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            trajectory_type="center_to_left",
            spikes={},
            stable_unit_ids=[],
            position=None,
            trajectory_interval=None,
            graph_inputs={},
            movement_interval=_Interval(3.0),
            movement_firing_rate_table=pd.DataFrame(
                columns=stability.MOVEMENT_RATE_COLUMNS
            ),
            place_bin_size_cm=4.0,
        )


@pytest.mark.parametrize("status", ["no_valid_position", "no_movement"])
def test_compute_selected_stability_propagates_upstream_terminal_status(
    status: str,
    monkeypatch,
) -> None:
    def fail_graph_build(**kwargs):
        raise AssertionError("terminal upstream status must skip linearization")

    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        fail_graph_build,
    )
    result = stability.compute_selected_stability(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        spikes={0: object(), 1: object()},
        stable_unit_ids=STABLE_UNIT_IDS,
        position=None,
        trajectory_interval=None,
        graph_inputs={},
        movement_interval=_Interval(0.0),
        movement_firing_rate_table=_movement_firing_rate_table(
            status=status,
            duration_s=0.0,
        ),
        place_bin_size_cm=4.0,
    )

    assert result["analysis_status"] == status
    assert result["n_units"] == 2
    assert result["n_valid_units"] == 0
    assert result["table"]["stable_unit_id"].tolist() == [
        "merge-a:11",
        "merge-b:22",
    ]
    assert result["table"]["stability_status"].tolist() == [status, status]
    assert result["table"]["firing_rate_hz"].isna().all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda table: table.assign(
                stable_unit_id=["merge-a:11", "merge-other:22"]
            ),
            "identities do not exactly match",
        ),
        (
            lambda table: table.assign(movement_firing_rate_hz=[2.0, 0.0]),
            "must equal spike count divided by duration",
        ),
        (
            lambda table: table.assign(movement_duration_s=[3.0, 3.0]),
            "duration does not match movement_interval",
        ),
    ],
)
def test_compute_selected_stability_rejects_misaligned_movement_artifact(
    mutation,
    message: str,
) -> None:
    movement_table = mutation(_movement_firing_rate_table())

    with pytest.raises(ValueError, match=message):
        stability.compute_selected_stability(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            trajectory_type="center_to_left",
            spikes={0: object(), 1: object()},
            stable_unit_ids=STABLE_UNIT_IDS,
            position=_Position(),
            trajectory_interval=object(),
            graph_inputs={},
            movement_interval=_Interval(2.0),
            movement_firing_rate_table=movement_table,
            place_bin_size_cm=4.0,
        )


def test_compute_selected_stability_rejects_changed_core_firing_rates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        lambda **kwargs: (object(), 20.0),
    )
    monkeypatch.setattr(
        stability,
        "compute_trajectory_stability_table",
        lambda **kwargs: pd.DataFrame(
            {
                "unit": [0, 1],
                "firing_rate_hz": [1.5, 0.0],
                "stability_status": ["valid", "valid"],
            }
        ),
    )

    with pytest.raises(ValueError, match="upstream movement artifact"):
        stability.compute_selected_stability(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            trajectory_type="center_to_left",
            spikes={0: object(), 1: object()},
            stable_unit_ids=STABLE_UNIT_IDS,
            position=_Position(),
            trajectory_interval=object(),
            graph_inputs={},
            movement_interval=_Interval(2.0),
            movement_firing_rate_table=_movement_firing_rate_table(),
            place_bin_size_cm=4.0,
        )


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
