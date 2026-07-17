from __future__ import annotations

"""Tests for shared 1D decoder cross-validation helpers."""

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from v1ca1.decoding._1d import (
    BINNING_SCHEME,
    CV_SCHEME,
    TRAJECTORY_TYPES,
    build_classifier_output_paths,
    build_contiguous_time_folds,
    build_prediction_output_paths,
    build_time_bins,
    build_trajectory_time_mask,
    classifier_output_path,
    figurl_output_path,
    get_spike_indicator,
    make_classifier,
    prediction_output_path,
)
from v1ca1.decoding.fit_1d import build_fold_training_mask


class FakeSorting:
    """Provide the small sorting-extractor interface needed by spike-bin tests."""

    def __init__(self, spike_indices_by_unit: dict[Any, np.ndarray]) -> None:
        self.spike_indices_by_unit = spike_indices_by_unit

    def get_unit_ids(self) -> list[Any]:
        """Return unit identifiers in insertion order."""
        return list(self.spike_indices_by_unit)

    def get_unit_spike_train(self, unit_id: Any) -> np.ndarray:
        """Return ephys timestamp indices for one unit."""
        return self.spike_indices_by_unit[unit_id]


def test_build_time_bins_returns_exact_width_edges_and_centers() -> None:
    """Decoder coordinates should be centers of exact-width complete bins."""
    timestamps_position = np.array([1.0, 1.001, 1.0065])

    time_bin_edges, time_grid = build_time_bins(
        timestamps_position,
        position_offset=1,
        time_bin_size_s=0.002,
    )

    np.testing.assert_allclose(time_bin_edges, np.array([1.001, 1.003, 1.005]))
    np.testing.assert_allclose(time_grid, np.array([1.002, 1.004]))
    np.testing.assert_allclose(np.diff(time_bin_edges), 0.002)
    np.testing.assert_allclose(time_grid, time_bin_edges[:-1] + 0.001)
    assert timestamps_position[-1] - time_bin_edges[-1] == pytest.approx(0.0015)
    assert 0.0 <= timestamps_position[-1] - time_bin_edges[-1] < 0.002


@pytest.mark.parametrize(
    ("start_time", "end_time", "expected_n_bins"),
    [
        (10.0, 10.004, 2),
        (100.0, 100.002, 1),
    ],
)
def test_build_time_bins_keeps_complete_bins_after_float_subtraction(
    start_time: float,
    end_time: float,
    expected_n_bins: int,
) -> None:
    """Floating-point cancellation must not remove a complete final bin."""
    time_bin_edges, time_grid = build_time_bins(
        np.array([start_time, end_time]),
        position_offset=0,
        time_bin_size_s=0.002,
    )

    assert time_grid.size == expected_n_bins
    assert time_bin_edges.size == expected_n_bins + 1
    np.testing.assert_allclose(np.diff(time_bin_edges), 0.002)
    assert time_bin_edges[-1] == pytest.approx(end_time)


@pytest.mark.parametrize(
    ("timestamps_position", "position_offset", "time_bin_size_s"),
    [
        (np.array([0.0, 0.004]), 0, 0.0),
        (np.array([0.0, 0.004]), 0, -0.002),
        (np.array([0.0, 0.004]), 0, np.inf),
        (np.array([0.0, 0.004]), -1, 0.002),
        (np.array([0.0, 0.004]), 2, 0.002),
        (np.array([[0.0, 0.002], [0.004, 0.006]]), 0, 0.002),
        (np.array([0.0, np.nan, 0.004]), 0, 0.002),
        (np.array([0.0, 0.004, 0.002]), 0, 0.002),
        (np.array([0.0, 0.001]), 0, 0.002),
    ],
)
def test_build_time_bins_rejects_invalid_inputs(
    timestamps_position: np.ndarray,
    position_offset: int,
    time_bin_size_s: float,
) -> None:
    """Malformed inputs should fail instead of creating misleading bins."""
    with pytest.raises(ValueError):
        build_time_bins(
            timestamps_position,
            position_offset=position_offset,
            time_bin_size_s=time_bin_size_s,
        )


def test_get_spike_indicator_uses_all_bins_and_preserves_counts() -> None:
    """First, interior, and final edges should follow NumPy histogram semantics."""
    timestamps_ephys_all = np.array(
        [-0.001, 0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    )
    sorting = FakeSorting(
        {
            11: np.arange(timestamps_ephys_all.size),
            12: np.array([1]),
            13: np.array([3, 5]),
            14: np.array([7]),
            15: np.array([0, 8]),
        }
    )

    spike_indicator = get_spike_indicator(
        sorting,
        timestamps_ephys_all=timestamps_ephys_all,
        time_bin_edges=np.array([0.0, 0.002, 0.004, 0.006]),
    )

    assert np.array_equal(
        spike_indicator,
        np.array(
            [
                [2, 1, 0, 0, 0],
                [2, 0, 1, 0, 0],
                [3, 0, 1, 1, 0],
            ]
        ),
    )
    assert np.array_equal(spike_indicator.sum(axis=0), np.array([7, 1, 2, 1, 0]))
    assert np.array_equal(spike_indicator[-1], np.array([3, 0, 1, 1, 0]))


def test_get_spike_indicator_preserves_time_axis_without_units() -> None:
    """An empty unit selection should retain one row per time bin."""
    spike_indicator = get_spike_indicator(
        FakeSorting({}),
        timestamps_ephys_all=np.array([0.0, 0.001, 0.002]),
        time_bin_edges=np.array([0.0, 0.002, 0.004, 0.006]),
    )

    assert spike_indicator.shape == (3, 0)
    assert spike_indicator.dtype == float


@pytest.mark.parametrize(
    "time_bin_edges",
    [
        np.array([0.0]),
        np.array([[0.0, 0.002], [0.004, 0.006]]),
        np.array([0.0, np.nan, 0.004]),
        np.array([0.0, 0.004, 0.002]),
        np.array([0.0, 0.002, 0.002]),
    ],
)
def test_get_spike_indicator_rejects_invalid_edges(
    time_bin_edges: np.ndarray,
) -> None:
    """Spike counts require a finite, strictly increasing one-dimensional grid."""
    with pytest.raises(ValueError):
        get_spike_indicator(
            FakeSorting({1: np.array([], dtype=int)}),
            timestamps_ephys_all=np.array([0.0, 0.001]),
            time_bin_edges=time_bin_edges,
        )


def test_build_contiguous_time_folds_is_an_exhaustive_partition() -> None:
    """Every time bin should belong to one nonempty contiguous fold."""
    time_grid = np.arange(11, dtype=float) * 0.002

    fold_by_time, fold_records = build_contiguous_time_folds(
        time_grid,
        n_folds=5,
    )

    assert CV_SCHEME == "contiguous_time"
    assert np.array_equal(
        fold_by_time,
        np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
    )
    assert fold_records == [
        {
            "fold": 0,
            "start_index": 0,
            "stop_index_exclusive": 3,
            "start_time_s": time_grid[0],
            "end_time_s": time_grid[2],
            "n_time_bins": 3,
        },
        {
            "fold": 1,
            "start_index": 3,
            "stop_index_exclusive": 5,
            "start_time_s": time_grid[3],
            "end_time_s": time_grid[4],
            "n_time_bins": 2,
        },
        {
            "fold": 2,
            "start_index": 5,
            "stop_index_exclusive": 7,
            "start_time_s": time_grid[5],
            "end_time_s": time_grid[6],
            "n_time_bins": 2,
        },
        {
            "fold": 3,
            "start_index": 7,
            "stop_index_exclusive": 9,
            "start_time_s": time_grid[7],
            "end_time_s": time_grid[8],
            "n_time_bins": 2,
        },
        {
            "fold": 4,
            "start_index": 9,
            "stop_index_exclusive": 11,
            "start_time_s": time_grid[9],
            "end_time_s": time_grid[10],
            "n_time_bins": 2,
        },
    ]
    assert np.array_equal(np.unique(fold_by_time), np.arange(5))
    assert np.array_equal(np.bincount(fold_by_time), np.array([3, 2, 2, 2, 2]))

    held_out_count = np.zeros(time_grid.size, dtype=int)
    for fold in range(5):
        fold_indices = np.flatnonzero(fold_by_time == fold)
        assert fold_indices.size > 0
        assert np.all(np.diff(fold_indices) == 1)
        held_out_count[fold_indices] += 1
    assert np.array_equal(held_out_count, np.ones(time_grid.size, dtype=int))


@pytest.mark.parametrize(
    ("time_grid", "n_folds", "match"),
    [
        (np.array([], dtype=float), 2, "non-empty"),
        (np.zeros((2, 2), dtype=float), 2, "one-dimensional"),
        (np.array([0.0, 0.0, 1.0]), 2, "strictly increasing"),
        (np.arange(3, dtype=float), 1, "at least 2"),
        (np.arange(3, dtype=float), 4, "time bins"),
    ],
)
def test_build_contiguous_time_folds_rejects_invalid_inputs(
    time_grid: np.ndarray,
    n_folds: int,
    match: str,
) -> None:
    """Invalid grids and fold counts should fail before creating empty folds."""
    with pytest.raises(ValueError, match=match):
        build_contiguous_time_folds(time_grid, n_folds=n_folds)


def test_build_trajectory_time_mask_unions_lap_intervals() -> None:
    """Training eligibility should include all bins in any trajectory interval."""
    trajectory_intervals = {
        trajectory_type: SimpleNamespace(
            start=np.array([], dtype=float),
            end=np.array([], dtype=float),
        )
        for trajectory_type in TRAJECTORY_TYPES
    }
    trajectory_intervals[TRAJECTORY_TYPES[0]] = SimpleNamespace(
        start=np.array([0.0, 4.0]),
        end=np.array([2.0, 5.0]),
    )
    time_grid = np.arange(7, dtype=float)

    trajectory_mask = build_trajectory_time_mask(time_grid, trajectory_intervals)

    assert np.array_equal(
        trajectory_mask,
        np.array([False, True, True, False, False, True, False]),
    )


def test_build_fold_training_mask_uses_movement_outside_held_out_fold() -> None:
    """Training should exclude the held-out chunk, stationary bins, and gaps."""
    fold_by_time = np.array([0, 0, 0, 1, 1, 1])
    trajectory_mask = np.array([True, True, False, True, True, True])
    linear_position = np.array([0.0, 1.0, 2.0, 3.0, np.nan, 5.0])
    speed = np.array([5.0, 0.0, 10.0, 5.0, 10.0, 0.0])

    fold_zero_training = build_fold_training_mask(
        fold_by_time=fold_by_time,
        fold=0,
        trajectory_mask=trajectory_mask,
        linear_position=linear_position,
        speed=speed,
        movement=True,
        speed_threshold_cm_s=4.0,
    )
    fold_one_training = build_fold_training_mask(
        fold_by_time=fold_by_time,
        fold=1,
        trajectory_mask=trajectory_mask,
        linear_position=linear_position,
        speed=speed,
        movement=True,
        speed_threshold_cm_s=4.0,
    )

    assert np.array_equal(
        fold_zero_training,
        np.array([False, False, False, True, False, False]),
    )
    assert np.array_equal(
        fold_one_training,
        np.array([True, False, False, False, False, False]),
    )


def test_make_classifier_configures_uniform_initial_conditions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every independently predicted fold should start from a uniform prior."""
    classifier_kwargs: dict[str, Any] = {}

    class FakeUniformInitialConditions:
        pass

    fake_rtc = ModuleType("replay_trajectory_classification")
    fake_rtc.Environment = lambda **kwargs: ("environment", kwargs)
    fake_rtc.UniformInitialConditions = FakeUniformInitialConditions

    def fake_classifier(**kwargs: Any) -> object:
        classifier_kwargs.update(kwargs)
        return object()

    fake_rtc.SortedSpikesClassifier = fake_classifier
    monkeypatch.setitem(sys.modules, "replay_trajectory_classification", fake_rtc)

    make_classifier(
        place_bin_size=2.0,
        track_graph=object(),
        edge_order=[(0, 1)],
        edge_spacing=[0.0],
        continuous_transition_types=[[object()]],
        observation_models=None,
        position_std=4.0,
    )

    assert isinstance(
        classifier_kwargs["initial_conditions_type"],
        FakeUniformInitialConditions,
    )


def _path_kwargs() -> dict[str, Any]:
    """Return common model settings for 1D output-path tests."""
    return {
        "epoch": "02_r1",
        "n_folds": 5,
        "time_bin_size_s": 0.002,
        "position_offset": 10,
        "direction": True,
        "movement": True,
        "speed_threshold_cm_s": 4.0,
        "position_std": 4.0,
        "discrete_var": "switching",
        "place_bin_size": 2.0,
        "movement_var": 6.0,
        "branch_gap_cm": 15.0,
    }


def test_1d_path_builders_identify_contiguous_time_cv(tmp_path: Path) -> None:
    """New artifacts should not collide with historical shuffled-lap outputs."""
    common_kwargs = _path_kwargs()
    paths = [
        classifier_output_path(
            tmp_path,
            region="ca1",
            fold=0,
            **common_kwargs,
        ),
        prediction_output_path(
            tmp_path,
            region="ca1",
            **common_kwargs,
        ),
        figurl_output_path(
            tmp_path,
            regions=("ca1",),
            **common_kwargs,
        ),
    ]

    assert all("_cv_contiguous_time" in path.name for path in paths)
    assert BINNING_SCHEME == "edges_centers_v1"
    assert all("_binning_edges_centers_v1" in path.name for path in paths)
    assert all("_tb_0.002" in path.name for path in paths)
    assert all("_offset_10" in path.name for path in paths)

    different_time_grid_path = prediction_output_path(
        tmp_path,
        region="ca1",
        **{
            **common_kwargs,
            "time_bin_size_s": 0.004,
            "position_offset": 20,
        },
    )
    assert different_time_grid_path != paths[1]


@pytest.mark.parametrize(
    "path_builder",
    [
        classifier_output_path,
        build_classifier_output_paths,
        prediction_output_path,
        build_prediction_output_paths,
        figurl_output_path,
    ],
)
def test_1d_path_builders_do_not_accept_random_state(path_builder: Any) -> None:
    """Contiguous deterministic folds should expose no random-seed setting."""
    assert "random_state" not in inspect.signature(path_builder).parameters
