from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import v1ca1.helper.get_timestamps as get_timestamps_module
from v1ca1.helper.get_timestamps import (
    get_timestamps,
    parse_arguments,
    save_pynapple_outputs,
    split_timestamps_by_gap,
    validate_epoch_alignment,
)


def test_split_timestamps_by_gap_uses_default_10_second_threshold() -> None:
    timestamps = np.array(
        [
            0.0,
            0.001,
            0.002,
            15.0,
            15.001,
            15.002,
            30.0,
            30.001,
            30.002,
        ]
    )

    segments, metadata = split_timestamps_by_gap(timestamps=timestamps, num_epochs=3)

    assert [segment.tolist() for segment in segments] == [
        [0.0, 0.001, 0.002],
        [15.0, 15.001, 15.002],
        [30.0, 30.001, 30.002],
    ]
    assert metadata["threshold_mode"] == "default"
    assert metadata["split_indices"] == [3, 6]
    assert metadata["gap_threshold_s"] == 10.0


def test_split_timestamps_by_gap_manual_override() -> None:
    timestamps = np.array([0.0, 0.001, 0.002, 0.5, 0.501, 0.502])

    segments, metadata = split_timestamps_by_gap(
        timestamps=timestamps,
        num_epochs=2,
        gap_threshold_s=0.1,
    )

    assert [segment.tolist() for segment in segments] == [
        [0.0, 0.001, 0.002],
        [0.5, 0.501, 0.502],
    ]
    assert metadata["threshold_mode"] == "manual"
    assert metadata["gap_threshold_s"] == 0.1


def test_split_timestamps_by_gap_keeps_subthreshold_dropout_in_same_segment() -> None:
    timestamps = np.array([0.0, 0.001, 0.002, 5.0, 5.001, 5.002])

    with pytest.raises(ValueError, match="wrong number of epochs"):
        split_timestamps_by_gap(timestamps=timestamps, num_epochs=2)


def test_split_timestamps_by_gap_rejects_non_monotonic_timestamps() -> None:
    timestamps = np.array([0.0, 0.001, 0.0005, 1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        split_timestamps_by_gap(timestamps=timestamps, num_epochs=2)


def test_split_timestamps_by_gap_rejects_wrong_manual_segment_count() -> None:
    timestamps = np.array([0.0, 0.001, 0.002])

    with pytest.raises(ValueError, match="wrong number of epochs"):
        split_timestamps_by_gap(
            timestamps=timestamps,
            num_epochs=2,
            gap_threshold_s=10.0,
        )


def test_validate_epoch_alignment_accepts_overlapping_bounds() -> None:
    epoch_tags = ["sleep1", "run1"]
    epoch_segments = [
        np.array([0.0, 0.001, 0.002]),
        np.array([1.0, 1.001, 1.002]),
    ]
    epoch_start_times = np.array([0.0, 1.0])
    epoch_stop_times = np.array([0.0025, 1.0025])

    validate_epoch_alignment(
        epoch_tags=epoch_tags,
        epoch_segments=epoch_segments,
        epoch_start_times=epoch_start_times,
        epoch_stop_times=epoch_stop_times,
        median_positive_dt_s=0.001,
    )


def test_save_pynapple_outputs_round_trip(tmp_path) -> None:
    nap = pytest.importorskip("pynapple")

    timestamps_ephys_all = np.array([0.0, 0.001, 0.002, 1.0, 1.001, 1.002])
    timestamps_position = {
        "sleep1": np.array([0.0, 0.1, 0.2]),
        "run1": np.array([1.0, 1.1, 1.2]),
    }
    epoch_tags = ["sleep1", "run1"]
    epoch_segments = [
        np.array([0.0, 0.001, 0.002]),
        np.array([1.0, 1.001, 1.002]),
    ]

    save_pynapple_outputs(
        analysis_path=tmp_path,
        timestamps_ephys_all=timestamps_ephys_all,
        timestamps_position=timestamps_position,
        epoch_tags=epoch_tags,
        epoch_segments=epoch_segments,
    )

    timestamps_all = nap.load_file(tmp_path / "timestamps_ephys_all.npz")
    epoch_intervals = nap.load_file(tmp_path / "timestamps_ephys.npz")
    position_group = nap.load_file(tmp_path / "timestamps_position.npz")

    assert np.allclose(timestamps_all.t, timestamps_ephys_all)
    assert np.allclose(epoch_intervals.start, [0.0, 1.0])
    assert np.allclose(epoch_intervals.end, [0.002, 1.002])
    assert list(epoch_intervals["epoch"]) == epoch_tags
    assert np.allclose(position_group[0].t, timestamps_position["sleep1"])
    assert np.allclose(position_group[1].t, timestamps_position["run1"])
    assert list(position_group["epoch"]) == epoch_tags


def test_get_timestamps_creates_missing_analysis_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "analysis"
    nwb_root = tmp_path / "nwb"
    nwb_root.mkdir()
    nwb_path = nwb_root / "animal20240101.nwb"
    nwb_path.touch()

    fake_nwbfile = types.SimpleNamespace(
        acquisition={
            "e-series": types.SimpleNamespace(
                timestamps=np.array([0.0, 0.001, 0.002]),
            )
        }
    )

    class FakeNWBHDF5IO:
        def __init__(self, path, mode) -> None:
            assert path == nwb_path
            assert mode == "r"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self):
            return fake_nwbfile

    fake_pynwb = types.ModuleType("pynwb")
    fake_pynwb.NWBHDF5IO = FakeNWBHDF5IO
    monkeypatch.setitem(sys.modules, "pynwb", fake_pynwb)
    monkeypatch.setattr(
        get_timestamps_module,
        "extract_epoch_metadata",
        lambda nwbfile: (
            ["run1"],
            np.array([0.0]),
            np.array([0.002]),
        ),
    )
    monkeypatch.setattr(
        get_timestamps_module,
        "get_timestamps_position",
        lambda nwbfile, epoch_tags: {"run1": np.array([0.0, 0.1])},
    )

    saved_paths: list[object] = []
    monkeypatch.setattr(
        get_timestamps_module,
        "save_pynapple_outputs",
        lambda analysis_path, **kwargs: saved_paths.append(analysis_path),
    )
    monkeypatch.setattr(
        get_timestamps_module,
        "write_run_log",
        lambda analysis_path, **kwargs: analysis_path / "run_log.json",
    )

    get_timestamps(
        animal_name="animal",
        date="20240101",
        data_root=data_root,
        nwb_root=nwb_root,
    )

    expected_analysis_path = data_root / "animal" / "20240101"
    assert expected_analysis_path.is_dir()
    assert saved_paths == [expected_analysis_path]


def test_parse_arguments_does_not_expose_save_pkl_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_timestamps.py",
            "--animal-name",
            "animal",
            "--date",
            "20240101",
        ],
    )

    args = parse_arguments()

    assert not hasattr(args, "save_pkl")


def test_parse_arguments_rejects_save_pkl_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_timestamps.py",
            "--animal-name",
            "animal",
            "--date",
            "20240101",
            "--save-pkl",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()
