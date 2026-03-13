from __future__ import annotations

import numpy as np
import pytest

from v1ca1.helper.get_immobility_times import (
    coerce_position_array,
    get_position_sampling_rate,
    intervalset_to_dataframe,
    save_pynapple_interval_output,
)


class DummyIntervalSet:
    def __init__(self, start, end) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)


def test_coerce_position_array_accepts_time_by_xy() -> None:
    position = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    result = coerce_position_array(position)

    assert np.array_equal(result, position)


def test_coerce_position_array_transposes_xy_by_time() -> None:
    position = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])

    result = coerce_position_array(position)

    assert np.array_equal(
        result,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


def test_get_position_sampling_rate_uses_time_span() -> None:
    timestamps_position = np.array([10.0, 10.5, 11.0, 11.5, 12.0])

    sampling_rate = get_position_sampling_rate(timestamps_position)

    assert sampling_rate == 2.0


def test_intervalset_to_dataframe_matches_legacy_columns() -> None:
    intervals = DummyIntervalSet(start=[1.0, 3.0], end=[2.5, 5.0])

    df = intervalset_to_dataframe(intervals)

    assert list(df.columns) == ["start_time", "end_time", "duration"]
    assert np.allclose(df["start_time"], [1.0, 3.0])
    assert np.allclose(df["end_time"], [2.5, 5.0])
    assert np.allclose(df["duration"], [1.5, 2.0])


def test_save_pynapple_interval_output_round_trip(tmp_path) -> None:
    nap = pytest.importorskip("pynapple")

    intervals_by_epoch = {
        "02_r1": nap.IntervalSet(start=[1.0, 3.0], end=[2.0, 4.0]),
        "04_r2": nap.IntervalSet(start=[5.0], end=[6.0]),
    }

    output_path = save_pynapple_interval_output(
        analysis_path=tmp_path,
        state_name="run_times",
        intervals_by_epoch=intervals_by_epoch,
    )

    interval_set = nap.load_file(output_path)

    assert np.allclose(interval_set.start, [1.0, 3.0, 5.0])
    assert np.allclose(interval_set.end, [2.0, 4.0, 6.0])
    assert list(interval_set["epoch"]) == ["02_r1", "02_r1", "04_r2"]
