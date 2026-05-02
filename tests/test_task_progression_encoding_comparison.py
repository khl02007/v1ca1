from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import numpy as np
import pytest


SESSION_MODULE_NAME = "v1ca1.task_progression._session"
ENCODING_MODULE_NAME = "v1ca1.task_progression.encoding_comparison"


class _FakeGraph:
    def __init__(self, nodes: dict[int, tuple[float, float]]) -> None:
        self.nodes = {
            node_id: {"pos": np.asarray(position, dtype=float)}
            for node_id, position in nodes.items()
        }


class _FakeTsd:
    def __init__(self, *, t: Any, d: Any, time_support: Any, time_units: str) -> None:
        self.t = np.asarray(t, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.time_support = time_support
        self.time_units = str(time_units)


def _reload_session_module():
    sys.modules.pop(SESSION_MODULE_NAME, None)
    return importlib.import_module(SESSION_MODULE_NAME)


def _reload_encoding_module():
    sys.modules.pop(ENCODING_MODULE_NAME, None)
    return importlib.import_module(ENCODING_MODULE_NAME)


def _make_cv_metrics(ll: float, info_bits: float, *, n_spikes: int = 7) -> dict[str, Any]:
    return {
        "fold_n_spikes": np.asarray([n_spikes], dtype=np.int64),
        "ll_model_per_spike_cv": float(ll),
        "ll_null_per_spike_cv": -2.0,
        "info_bits_per_spike_cv": float(info_bits),
    }


class _FakeParquetTable:
    """Minimal table stand-in that records parquet save paths."""

    def __init__(self) -> None:
        self.saved_paths: list[Any] = []

    def to_parquet(self, path: Any) -> None:
        self.saved_paths.append(path)


def test_build_generalized_place_bins_uses_full_w_length_and_branch_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_session_module()

    calls: list[dict[str, Any]] = []

    def _fake_get_wtrack_full_graph(animal_name: str, branch_gap_cm: float):
        calls.append(
            {
                "animal_name": animal_name,
                "branch_gap_cm": float(branch_gap_cm),
            }
        )
        graph = _FakeGraph(
            {
                0: (0.0, 0.0),
                1: (3.0, 4.0),
                2: (6.0, 4.0),
            }
        )
        return graph, [(0, 1), (1, 2)], [float(branch_gap_cm)]

    monkeypatch.setattr(module, "get_wtrack_full_graph", _fake_get_wtrack_full_graph)

    bins = module.build_generalized_place_bins(
        "L14",
        branch_gap_cm=15.0,
        place_bin_size_cm=4.0,
    )

    assert calls == [{"animal_name": "L14", "branch_gap_cm": 15.0}]
    assert np.allclose(bins, np.asarray([0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]))


def test_build_generalized_place_position_reuses_full_w_linearizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_session_module()

    fake_pynapple = types.ModuleType("pynapple")
    fake_pynapple.Tsd = _FakeTsd
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)

    linearizer_calls: list[dict[str, Any]] = []
    fake_decoding_1d = types.ModuleType("v1ca1.decoding._1d")

    def _fake_linearize_full_w_position(
        *,
        animal_name: str,
        position_interp: np.ndarray,
        branch_gap_cm: float,
    ):
        linearizer_calls.append(
            {
                "animal_name": animal_name,
                "position_interp": np.asarray(position_interp, dtype=float),
                "branch_gap_cm": float(branch_gap_cm),
            }
        )
        return np.asarray([10.0, 14.0], dtype=float), "graph", "order", "spacing"

    fake_decoding_1d.linearize_full_w_position = _fake_linearize_full_w_position
    monkeypatch.setitem(sys.modules, "v1ca1.decoding._1d", fake_decoding_1d)

    position = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=float,
    )
    timestamps = np.asarray([0.0, 0.5, 1.0], dtype=float)
    movement_interval = object()

    generalized_place = module.build_generalized_place_position(
        "L14",
        position,
        timestamps,
        movement_interval,
        position_offset=1,
        branch_gap_cm=15.0,
    )

    assert isinstance(generalized_place, _FakeTsd)
    assert np.allclose(generalized_place.t, [0.5, 1.0])
    assert np.allclose(generalized_place.d, [10.0, 14.0])
    assert generalized_place.time_support is movement_interval
    assert generalized_place.time_units == "s"
    assert linearizer_calls[0]["animal_name"] == "L14"
    assert linearizer_calls[0]["branch_gap_cm"] == 15.0
    assert np.allclose(linearizer_calls[0]["position_interp"], position[1:])


def test_cv_epoch_to_df_adds_generalized_place_metrics() -> None:
    module = _reload_encoding_module()

    cv_by_model = {
        "place": {11: _make_cv_metrics(-1.0, 0.2)},
        "generalized_place": {11: _make_cv_metrics(-0.5, 0.7)},
        "tp": {11: _make_cv_metrics(-1.5, 0.1)},
        "gtp": {11: _make_cv_metrics(-1.2, 0.4)},
    }

    table = module.cv_epoch_to_df(cv_by_model)

    assert table.index.tolist() == [11]
    assert table.loc[11, "ll_generalized_place"] == -0.5
    assert table.loc[11, "info_bits_generalized_place"] == 0.7
    assert np.isclose(
        table.loc[11, "delta_bits_generalized_place_vs_tp"],
        (-0.5 - -1.5) / np.log(2.0),
    )


def test_cv_epoch_to_df_requires_generalized_place_model() -> None:
    module = _reload_encoding_module()

    cv_by_model = {
        "place": {11: _make_cv_metrics(-1.0, 0.2)},
        "tp": {11: _make_cv_metrics(-1.5, 0.1)},
        "gtp": {11: _make_cv_metrics(-1.2, 0.4)},
    }

    with pytest.raises(ValueError, match="generalized_place"):
        module.cv_epoch_to_df(cv_by_model)


def test_format_delta_histogram_stats_reports_fraction_mean_and_median() -> None:
    module = _reload_encoding_module()

    text = module._format_delta_histogram_stats(
        np.asarray([-0.2, -0.1, 0.0, 0.3], dtype=float)
    )

    assert "Frac < 0: 0.50" in text
    assert "Mean: 0.000" in text
    assert "Median: -0.050" in text


def test_parse_arguments_accepts_place_bin_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_encoding_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "encoding_comparison.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-epoch",
            "run2",
            "--place-bin-size-cm",
            "2.5",
        ],
    )

    args = module.parse_arguments()

    assert args.place_bin_size_cm == 2.5


def test_save_paths_include_place_bin_size_token(tmp_path) -> None:
    module = _reload_encoding_module()
    table = _FakeParquetTable()

    epoch_paths = module.save_epoch_tables(
        {"v1": {"run1": table}},
        data_dir=tmp_path,
        n_folds=5,
        place_bin_size_cm=2.5,
    )
    comparison_paths = module.save_comparison_tables(
        {"v1": {"run1": table}},
        data_dir=tmp_path,
        dark_epoch="run2",
        n_folds=5,
        place_bin_size_cm=2.5,
    )
    cross_paths = module.save_tp_cross_trajectory_tables(
        {"v1": {"run1": table}},
        data_dir=tmp_path,
        n_folds=5,
        place_bin_size_cm=2.5,
    )

    assert epoch_paths == [
        tmp_path / "v1_run1_cv5_placebin2p5cm_encoding_summary.parquet"
    ]
    assert comparison_paths == [
        tmp_path / "v1_run1_run2_cv5_placebin2p5cm_encoding_comparison.parquet"
    ]
    assert cross_paths == [
        tmp_path / "v1_run1_cv5_placebin2p5cm_tp_cross_trajectory_encoding.parquet"
    ]
    assert table.saved_paths == epoch_paths + comparison_paths + cross_paths
