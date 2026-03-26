from __future__ import annotations

import sys
import pickle

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.get_trajectory_times import (
    parse_arguments,
    save_trajectory_table_output,
)
from v1ca1.helper.session import load_trajectory_intervals, load_trajectory_time_bounds


def test_save_trajectory_table_output_writes_expected_parquet(tmp_path) -> None:
    trajectory_times = {
        "04_r2": {
            "left_to_center": np.array([[9.0, 10.0]]),
            "center_to_left": np.empty((0, 2), dtype=float),
            "right_to_center": np.empty((0, 2), dtype=float),
            "center_to_right": np.array([[11.0, 12.0], [13.0, 14.0]]),
        },
        "02_r1": {
            "left_to_center": np.array([[3.0, 4.0], [1.0, 2.0]]),
            "center_to_left": np.array([[5.0, 6.0]]),
            "right_to_center": np.empty((0, 2), dtype=float),
            "center_to_right": np.array([[7.0, 8.0]]),
        },
    }

    output_path = save_trajectory_table_output(
        analysis_path=tmp_path,
        trajectory_times=trajectory_times,
    )

    table = pd.read_parquet(output_path)

    assert list(table.columns) == ["start", "end", "epoch", "trajectory_type"]
    assert table.to_dict("records") == [
        {
            "start": 1.0,
            "end": 2.0,
            "epoch": "02_r1",
            "trajectory_type": "left_to_center",
        },
        {
            "start": 3.0,
            "end": 4.0,
            "epoch": "02_r1",
            "trajectory_type": "left_to_center",
        },
        {
            "start": 5.0,
            "end": 6.0,
            "epoch": "02_r1",
            "trajectory_type": "center_to_left",
        },
        {
            "start": 7.0,
            "end": 8.0,
            "epoch": "02_r1",
            "trajectory_type": "center_to_right",
        },
        {
            "start": 9.0,
            "end": 10.0,
            "epoch": "04_r2",
            "trajectory_type": "left_to_center",
        },
        {
            "start": 11.0,
            "end": 12.0,
            "epoch": "04_r2",
            "trajectory_type": "center_to_right",
        },
        {
            "start": 13.0,
            "end": 14.0,
            "epoch": "04_r2",
            "trajectory_type": "center_to_right",
        },
    ]


def test_load_trajectory_intervals_prefers_parquet_and_reconstructs_groups(tmp_path) -> None:
    nap = pytest.importorskip("pynapple")

    trajectory_times = {
        "02_r1": {
            "left_to_center": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "center_to_left": np.array([[5.0, 6.0]]),
            "right_to_center": np.empty((0, 2), dtype=float),
            "center_to_right": np.array([[7.0, 8.0]]),
        },
        "04_r2": {
            "left_to_center": np.empty((0, 2), dtype=float),
            "center_to_left": np.empty((0, 2), dtype=float),
            "right_to_center": np.array([[9.0, 10.0]]),
            "center_to_right": np.array([[11.0, 12.0], [13.0, 14.0]]),
        },
    }

    output_path = save_trajectory_table_output(
        analysis_path=tmp_path,
        trajectory_times=trajectory_times,
    )

    assert output_path.name == "trajectory_times.parquet"

    intervals_by_epoch, source = load_trajectory_intervals(
        analysis_path=tmp_path,
        run_epochs=["02_r1", "04_r2"],
    )

    assert source == "parquet"
    assert np.allclose(intervals_by_epoch["02_r1"]["left_to_center"].start, [1.0, 3.0])
    assert np.allclose(intervals_by_epoch["02_r1"]["left_to_center"].end, [2.0, 4.0])
    assert np.allclose(intervals_by_epoch["02_r1"]["center_to_left"].start, [5.0])
    assert np.allclose(intervals_by_epoch["02_r1"]["center_to_left"].end, [6.0])
    assert len(intervals_by_epoch["02_r1"]["right_to_center"]) == 0
    assert np.allclose(intervals_by_epoch["02_r1"]["center_to_right"].start, [7.0])
    assert np.allclose(intervals_by_epoch["02_r1"]["center_to_right"].end, [8.0])
    assert len(intervals_by_epoch["04_r2"]["left_to_center"]) == 0
    assert len(intervals_by_epoch["04_r2"]["center_to_left"]) == 0
    assert np.allclose(intervals_by_epoch["04_r2"]["right_to_center"].start, [9.0])
    assert np.allclose(intervals_by_epoch["04_r2"]["right_to_center"].end, [10.0])
    assert np.allclose(intervals_by_epoch["04_r2"]["center_to_right"].start, [11.0, 13.0])
    assert np.allclose(intervals_by_epoch["04_r2"]["center_to_right"].end, [12.0, 14.0])


def test_load_trajectory_time_bounds_prefers_parquet_and_preserves_empty_shapes(
    tmp_path,
) -> None:
    pytest.importorskip("pynapple")

    trajectory_times = {
        "02_r1": {
            "left_to_center": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "center_to_left": np.array([[5.0, 6.0]]),
            "right_to_center": np.empty((0, 2), dtype=float),
            "center_to_right": np.array([[7.0, 8.0]]),
        },
        "04_r2": {
            "left_to_center": np.empty((0, 2), dtype=float),
            "center_to_left": np.empty((0, 2), dtype=float),
            "right_to_center": np.array([[9.0, 10.0]]),
            "center_to_right": np.array([[11.0, 12.0], [13.0, 14.0]]),
        },
    }
    save_trajectory_table_output(
        analysis_path=tmp_path,
        trajectory_times=trajectory_times,
    )

    bounds_by_epoch, source = load_trajectory_time_bounds(
        analysis_path=tmp_path,
        run_epochs=["02_r1", "04_r2"],
    )

    assert source == "parquet"
    assert np.allclose(bounds_by_epoch["02_r1"]["left_to_center"], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(bounds_by_epoch["02_r1"]["center_to_left"], [[5.0, 6.0]])
    assert bounds_by_epoch["02_r1"]["right_to_center"].shape == (0, 2)
    assert bounds_by_epoch["04_r2"]["left_to_center"].shape == (0, 2)
    assert np.allclose(bounds_by_epoch["04_r2"]["right_to_center"], [[9.0, 10.0]])
    assert np.allclose(
        bounds_by_epoch["04_r2"]["center_to_right"],
        [[11.0, 12.0], [13.0, 14.0]],
    )


def test_load_trajectory_intervals_falls_back_to_pickle(tmp_path) -> None:
    nap = pytest.importorskip("pynapple")

    trajectory_times = {
        "02_r1": {
            "left_to_center": np.array([[1.0, 2.0]]),
            "center_to_left": np.empty((0, 2), dtype=float),
            "right_to_center": np.array([[3.0, 4.0]]),
            "center_to_right": np.empty((0, 2), dtype=float),
        },
        "04_r2": {
            "left_to_center": np.empty((0, 2), dtype=float),
            "center_to_left": np.array([[5.0, 6.0]]),
            "right_to_center": np.empty((0, 2), dtype=float),
            "center_to_right": np.array([[7.0, 8.0]]),
        },
    }
    with open(tmp_path / "trajectory_times.pkl", "wb") as f:
        pickle.dump(trajectory_times, f)

    intervals_by_epoch, source = load_trajectory_intervals(
        analysis_path=tmp_path,
        run_epochs=["02_r1", "04_r2"],
    )

    assert source == "pickle"
    assert isinstance(intervals_by_epoch["02_r1"]["left_to_center"], nap.IntervalSet)
    assert np.allclose(intervals_by_epoch["02_r1"]["left_to_center"].start, [1.0])
    assert np.allclose(intervals_by_epoch["02_r1"]["left_to_center"].end, [2.0])
    assert np.allclose(intervals_by_epoch["02_r1"]["right_to_center"].start, [3.0])
    assert np.allclose(intervals_by_epoch["04_r2"]["center_to_left"].start, [5.0])
    assert np.allclose(intervals_by_epoch["04_r2"]["center_to_right"].end, [8.0])


@pytest.mark.parametrize(
    ("argv", "expected_save_pkl"),
    [
        (
            [
                "get_trajectory_times.py",
                "--animal-name",
                "animal",
                "--date",
                "20240101",
            ],
            False,
        ),
        (
            [
                "get_trajectory_times.py",
                "--animal-name",
                "animal",
                "--date",
                "20240101",
                "--save-pkl",
            ],
            True,
        ),
    ],
)
def test_parse_arguments_save_pkl_flag(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_save_pkl: bool,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        argv,
    )

    args = parse_arguments()

    assert args.save_pkl is expected_save_pkl
