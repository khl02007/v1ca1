from __future__ import annotations

import numpy as np
import pytest

from v1ca1.helper.get_trajectory_times import save_pynapple_trajectory_output


def test_save_pynapple_trajectory_output_round_trip(tmp_path) -> None:
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

    output_path = save_pynapple_trajectory_output(
        analysis_path=tmp_path,
        trajectory_times=trajectory_times,
    )

    interval_set = nap.load_file(output_path)

    assert np.allclose(interval_set.start, [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0])
    assert np.allclose(interval_set.end, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    assert list(interval_set["epoch"]) == [
        "02_r1",
        "02_r1",
        "02_r1",
        "02_r1",
        "04_r2",
        "04_r2",
        "04_r2",
    ]
    assert list(interval_set["trajectory_type"]) == [
        "left_to_center",
        "left_to_center",
        "center_to_left",
        "center_to_right",
        "right_to_center",
        "center_to_right",
        "center_to_right",
    ]
