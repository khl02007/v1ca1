"""Tests for database-free Spyglass movement-rate artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass import movement


class _Position:
    """Minimal already-offset Pynapple-like position."""

    def __init__(self, values: np.ndarray | None = None) -> None:
        self.d = np.asarray(
            values
            if values is not None
            else [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            dtype=float,
        )
        self.t = np.arange(self.d.shape[0], dtype=float) * 0.1 + 10.0


class _Intervals:
    """Small IntervalSet-like test double."""

    def __init__(self, starts: list[float], ends: list[float]) -> None:
        self.start = np.asarray(starts, dtype=float)
        self.end = np.asarray(ends, dtype=float)

    def tot_length(self) -> float:
        return float(np.sum(self.end - self.start))


class _Spikes:
    """Small TsGroup-like object with deterministic interval counts."""

    def __init__(self, keys: list[int], counts: np.ndarray) -> None:
        self._keys = list(keys)
        self._counts = np.asarray(counts, dtype=float)

    def keys(self):
        return iter(self._keys)

    def count(self, *, ep):
        del ep
        return SimpleNamespace(to_numpy=lambda: self._counts.copy())


def _stable_ids() -> list[dict[str, object]]:
    return [
        {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        {"spikesorting_merge_id": "merge-b", "unit_id": 22},
    ]


def _valid_table(*, first_count: int = 3) -> pd.DataFrame:
    counts = np.asarray([first_count, 0], dtype=np.int64)
    return movement._build_movement_table(
        identity_rows=[
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "stable_unit_id": "merge-a:11",
                "group_unit_id": 0,
            },
            {
                "spikesorting_merge_id": "merge-b",
                "unit_id": "22",
                "stable_unit_id": "merge-b:22",
                "group_unit_id": 1,
            },
        ],
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        movement_spike_counts=counts,
        movement_duration_s=2.0,
        movement_firing_rates_hz=counts.astype(float) / 2.0,
        firing_rate_status="valid",
        position_sample_count=100,
        finite_position_sample_count=100,
        finite_speed_sample_count=100,
        movement_interval_count=2,
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )


def test_movement_artifact_paths_are_uuid_keyed_and_session_first(
    tmp_path: Path,
) -> None:
    movement_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    paths = movement.get_movement_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        movement_firing_rate_id=movement_id,
        artifact_root=tmp_path,
    )

    expected_dir = (
        tmp_path
        / "L14"
        / "20240611"
        / "movement_firing_rate"
        / "02_r1"
        / "v1"
        / str(movement_id)
    )
    assert paths == {
        "artifact_dir": expected_dir,
        "firing_rate_path": expected_dir / "movement_firing_rate.parquet",
        "movement_intervals_path": expected_dir / "movement_intervals.npz",
    }
    with pytest.raises(ValueError, match="UUID"):
        movement.get_movement_artifact_dir(
            animal_name="L14",
            date="20240611",
            epoch="02_r1",
            region="v1",
            movement_firing_rate_id="not-a-uuid",
            artifact_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("threshold", "sigma", "message"),
    [
        (-0.1, 0.1, "speed_threshold_cm_s"),
        (np.nan, 0.1, "speed_threshold_cm_s"),
        (4.0, 0.0, "speed_smoothing_sigma_s"),
        (4.0, np.inf, "speed_smoothing_sigma_s"),
    ],
)
def test_compute_rejects_invalid_movement_parameters(
    threshold: float,
    sigma: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        movement.compute_selected_movement_firing_rate(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            spikes={},
            stable_unit_ids=[],
            position=None,
            speed_threshold_cm_s=threshold,
            speed_smoothing_sigma_s=sigma,
        )


def test_compute_no_units_precedes_position_validation() -> None:
    result = movement.compute_selected_movement_firing_rate(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        spikes={},
        stable_unit_ids=[],
        position=None,
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )

    assert result["analysis_status"] == "no_units"
    assert result["table"].empty
    assert result["table"].columns.tolist() == list(movement.MOVEMENT_TABLE_COLUMNS)
    assert movement.movement_interval_summary(result["movement_intervals"]) == (0, 0.0)


def test_compute_rejects_misaligned_stable_unit_identity() -> None:
    with pytest.raises(ValueError, match="lengths must match"):
        movement.compute_selected_movement_firing_rate(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            spikes=_Spikes([0], np.asarray([[1.0]])),
            stable_unit_ids=[],
            position=_Position(),
            speed_threshold_cm_s=4.0,
            speed_smoothing_sigma_s=0.1,
        )


def test_compute_no_valid_position_retains_all_units(monkeypatch) -> None:
    monkeypatch.setattr(
        movement,
        "build_speed_tsd",
        lambda *args, **kwargs: pytest.fail("speed must not be computed"),
    )
    result = movement.compute_selected_movement_firing_rate(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        spikes=_Spikes([0, 1], np.asarray([[0.0, 0.0]])),
        stable_unit_ids=_stable_ids(),
        position=_Position([[np.nan, np.nan], [1.0, 2.0]]),
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )

    table = result["table"]
    assert result["analysis_status"] == "no_valid_position"
    assert table["stable_unit_id"].tolist() == ["merge-a:11", "merge-b:22"]
    assert table["movement_spike_count"].tolist() == [0, 0]
    assert table["movement_firing_rate_hz"].isna().all()
    assert table["finite_position_sample_count"].unique().tolist() == [1]
    assert movement.movement_interval_summary(result["movement_intervals"]) == (0, 0.0)


def test_compute_no_valid_speed_retains_all_units(monkeypatch) -> None:
    monkeypatch.setattr(
        movement,
        "build_speed_tsd",
        lambda *args, **kwargs: SimpleNamespace(d=np.full(4, np.nan)),
    )
    result = movement.compute_selected_movement_firing_rate(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        spikes=_Spikes([0, 1], np.asarray([[0.0, 0.0]])),
        stable_unit_ids=_stable_ids(),
        position=_Position(),
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )

    assert result["analysis_status"] == "no_valid_position"
    assert result["finite_position_sample_count"] == 4
    assert result["finite_speed_sample_count"] == 0
    assert result["table"]["movement_firing_rate_hz"].isna().all()


def test_compute_no_movement_uses_strict_threshold_and_undefined_rates(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_speed(position, timestamps, **kwargs):
        calls["position"] = np.asarray(position)
        calls["timestamps"] = np.asarray(timestamps)
        calls["speed_kwargs"] = kwargs
        return SimpleNamespace(d=np.asarray([4.0, 4.0, 3.0, 4.0]))

    def fake_movement(speed, **kwargs):
        calls["speed"] = speed
        calls["movement_kwargs"] = kwargs
        return _Intervals([], [])

    monkeypatch.setattr(movement, "build_speed_tsd", fake_speed)
    monkeypatch.setattr(movement, "build_movement_interval", fake_movement)
    result = movement.compute_selected_movement_firing_rate(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        spikes=_Spikes([0, 1], np.asarray([[9.0, 9.0]])),
        stable_unit_ids=_stable_ids(),
        position=_Position(),
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.25,
    )

    assert result["analysis_status"] == "no_movement"
    assert result["table"]["movement_firing_rate_hz"].isna().all()
    assert calls["speed_kwargs"] == {
        "position_offset": 0,
        "speed_smoothing_sigma_s": 0.25,
    }
    assert calls["movement_kwargs"] == {"speed_threshold_cm_s": 4.0}


def test_movement_threshold_is_strictly_above() -> None:
    import pynapple as nap

    speed = nap.Tsd(
        t=np.asarray([0.0, 1.0, 2.0]),
        d=np.asarray([4.0, 5.0, 4.0]),
        time_units="s",
    )
    intervals = movement.build_movement_interval(
        speed,
        speed_threshold_cm_s=4.0,
    )

    assert np.asarray(intervals.start, dtype=float).tolist() == pytest.approx([0.5])
    assert np.asarray(intervals.end, dtype=float).tolist() == pytest.approx([1.5])


def test_compute_valid_rates_preserves_identity_and_counts(monkeypatch) -> None:
    intervals = _Intervals([10.0, 12.0], [11.0, 13.0])
    monkeypatch.setattr(
        movement,
        "build_speed_tsd",
        lambda *args, **kwargs: SimpleNamespace(d=np.asarray([1.0, 5.0, 6.0, 2.0])),
    )
    monkeypatch.setattr(
        movement,
        "build_movement_interval",
        lambda *args, **kwargs: intervals,
    )
    result = movement.compute_selected_movement_firing_rate(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        spikes=_Spikes([7, 3], np.asarray([[1.0, 0.0], [2.0, 3.0]])),
        stable_unit_ids=_stable_ids(),
        position=_Position(),
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )

    table = result["table"]
    assert result["analysis_status"] == "valid"
    assert result["n_units"] == 2
    assert result["n_units_with_spikes"] == 2
    assert result["movement_duration_s"] == pytest.approx(2.0)
    assert table["group_unit_id"].tolist() == [7, 3]
    assert table["movement_spike_count"].tolist() == [3, 3]
    assert table["movement_firing_rate_hz"].tolist() == pytest.approx([1.5, 1.5])
    assert table["firing_rate_status"].unique().tolist() == ["valid"]


def test_table_validation_rejects_inconsistent_rate() -> None:
    table = _valid_table()
    table.loc[0, "movement_firing_rate_hz"] = 99.0

    with pytest.raises(ValueError, match="spike count divided by duration"):
        movement.validate_movement_firing_rate_table(table)


def test_align_movement_rates_restores_ephemeral_group_order() -> None:
    table = _valid_table().iloc[::-1].reset_index(drop=True)
    rates = movement.align_movement_firing_rates(
        table,
        spikes=_Spikes([9, 4], np.asarray([[0.0, 0.0]])),
        stable_unit_ids=_stable_ids(),
    )

    assert rates.index.tolist() == [9, 4]
    assert rates.tolist() == pytest.approx([1.5, 0.0])


def test_write_and_load_artifacts_roundtrip_including_empty_intervalset(
    tmp_path: Path,
) -> None:
    import pynapple as nap

    artifact_dir = tmp_path / "valid"
    intervals = nap.IntervalSet(
        start=np.asarray([10.0, 12.0]),
        end=np.asarray([11.0, 13.0]),
        time_units="s",
    )
    paths = movement.write_movement_artifacts(
        _valid_table(),
        intervals,
        artifact_dir,
    )
    loaded = movement.load_movement_artifacts(artifact_dir)

    assert paths == {
        "firing_rate_path": artifact_dir / movement.FIRING_RATE_FILENAME,
        "movement_intervals_path": artifact_dir / movement.INTERVALS_FILENAME,
    }
    pd.testing.assert_frame_equal(loaded["table"], _valid_table())
    assert loaded["analysis_status"] == "valid"
    assert movement.movement_interval_summary(loaded["movement_intervals"]) == (2, 2.0)

    empty_dir = tmp_path / "empty"
    empty_intervals = nap.IntervalSet(
        start=np.array([], dtype=float),
        end=np.array([], dtype=float),
        time_units="s",
    )
    movement.write_movement_artifacts(
        movement.empty_movement_firing_rate_table(),
        empty_intervals,
        empty_dir,
    )
    loaded_empty = movement.load_movement_artifacts(empty_dir)
    assert loaded_empty["analysis_status"] == "no_units"
    assert loaded_empty["table"].empty
    assert movement.movement_interval_summary(loaded_empty["movement_intervals"]) == (
        0,
        0.0,
    )


def test_write_artifact_bundle_is_atomic_and_refuses_implicit_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pynapple as nap

    artifact_dir = tmp_path / "movement"
    intervals = nap.IntervalSet(
        start=np.asarray([10.0, 12.0]),
        end=np.asarray([11.0, 13.0]),
        time_units="s",
    )
    original = _valid_table(first_count=3)
    replacement = _valid_table(first_count=5)
    movement.write_movement_artifacts(original, intervals, artifact_dir)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        movement.write_movement_artifacts(replacement, intervals, artifact_dir)
    pd.testing.assert_frame_equal(
        movement.load_movement_artifacts(artifact_dir)["table"],
        original,
    )

    real_replace = os.replace
    replace_count = 0

    def fail_bundle_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated bundle replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(movement.os, "replace", fail_bundle_replace)
    with pytest.raises(OSError, match="simulated bundle replacement failure"):
        movement.write_movement_artifacts(
            replacement,
            intervals,
            artifact_dir,
            overwrite=True,
        )

    pd.testing.assert_frame_equal(
        movement.load_movement_artifacts(artifact_dir)["table"],
        original,
    )
    assert not list(tmp_path.glob(".movement.*"))
