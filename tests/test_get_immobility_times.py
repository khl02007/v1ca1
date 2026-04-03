from __future__ import annotations

import sys

import numpy as np
import pytest

from v1ca1.helper import session
import v1ca1.helper.get_immobility_times as immobility_module
from v1ca1.helper.get_immobility_times import (
    DEFAULT_DATA_ROOT,
    coerce_position_array,
    get_position_sampling_rate,
    has_any_finite_position,
    parse_arguments,
    save_interval_table_output,
    select_epochs_with_usable_head_position,
)


class DummyIntervalSet:
    def __init__(self, start, end) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    def __len__(self) -> int:
        return int(self.start.shape[0])

    def tot_length(self) -> float:
        return float((self.end - self.start).sum()) if self.start.size else 0.0


class DummySpeedTsd:
    def __init__(self, t) -> None:
        self.t = np.asarray(t, dtype=float)


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


def test_coerce_position_array_preserves_first_two_columns_for_extra_features() -> None:
    position = np.array(
        [
            [1.0, 2.0, 9.0],
            [3.0, 4.0, 8.0],
            [5.0, 6.0, 7.0],
        ]
    )

    result = coerce_position_array(position)

    assert np.array_equal(
        result,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )


def test_get_position_sampling_rate_uses_time_span() -> None:
    timestamps_position = np.array([10.0, 10.5, 11.0, 11.5, 12.0])

    sampling_rate = get_position_sampling_rate(timestamps_position)

    assert sampling_rate == 2.0


def test_has_any_finite_position_requires_one_finite_sample() -> None:
    assert has_any_finite_position(np.array([[np.nan, np.nan], [1.0, np.nan]]))
    assert not has_any_finite_position(np.array([[np.nan, np.nan], [np.nan, np.nan]]))
    assert not has_any_finite_position(None)


def test_select_epochs_with_usable_head_position_skips_missing_and_all_nan() -> None:
    selected_epochs, skipped_epochs = select_epochs_with_usable_head_position(
        ["01_s1", "02_r1", "03_r2"],
        position_by_epoch={
            "02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "03_r2": np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float),
        },
        position_source="/tmp/position.parquet",
    )

    assert selected_epochs == ["02_r1"]
    assert skipped_epochs == [
        {"epoch": "01_s1", "reason": "head position missing from /tmp/position.parquet"},
        {"epoch": "03_r2", "reason": "head position is all NaN in /tmp/position.parquet"},
    ]


def test_save_interval_table_output_writes_sorted_parquet_rows(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    intervals_by_epoch = {
        "04_r2": DummyIntervalSet(start=[5.0], end=[6.0]),
        "02_r1": DummyIntervalSet(start=[3.0, 1.0], end=[4.0, 2.0]),
    }

    output_path = save_interval_table_output(
        analysis_path=tmp_path,
        output_name="run_times",
        intervals_by_epoch=intervals_by_epoch,
    )

    table = pd.read_parquet(output_path)

    assert list(table.columns) == ["start", "end", "epoch"]
    assert table.to_dict("records") == [
        {"start": 1.0, "end": 2.0, "epoch": "02_r1"},
        {"start": 3.0, "end": 4.0, "epoch": "02_r1"},
        {"start": 5.0, "end": 6.0, "epoch": "04_r2"},
    ]
    assert output_path == tmp_path / "run_times.parquet"


def test_save_interval_table_output_writes_empty_table(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    output_path = save_interval_table_output(
        analysis_path=tmp_path,
        output_name="immobility_times",
        intervals_by_epoch={"02_r1": DummyIntervalSet(start=[], end=[])},
    )

    table = pd.read_parquet(output_path)

    assert list(table.columns) == ["start", "end", "epoch"]
    assert table.empty


def test_get_immobility_times_uses_only_epochs_with_usable_head_position(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    monkeypatch.setattr(
        immobility_module,
        "load_position_timestamps",
        lambda path: (
            ["01_s1", "02_r1", "03_r2"],
            {
                "01_s1": np.array([0.0, 1.0], dtype=float),
                "02_r1": np.array([10.0, 11.0], dtype=float),
                "03_r2": np.array([20.0, 21.0], dtype=float),
            },
            "pynapple",
        ),
    )
    monkeypatch.setattr(
        immobility_module,
        "load_position_data_with_precedence",
        lambda *args, **kwargs: (
            {
                "02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
                "03_r2": np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float),
            },
            str(analysis_path / "dlc_position_cleaned" / "position.parquet"),
        ),
    )

    observed_positions: list[np.ndarray] = []
    monkeypatch.setattr(
        immobility_module,
        "build_speed_tsd",
        lambda position, timestamps_position, position_offset: (
            observed_positions.append(np.asarray(position, dtype=float)) or DummySpeedTsd([0.0, 1.0])
        ),
    )
    monkeypatch.setattr(
        immobility_module,
        "compute_movement_and_immobility_intervals",
        lambda speed_tsd, speed_threshold_cm_s: (
            DummyIntervalSet([0.0], [0.5]),
            DummyIntervalSet([0.5], [1.0]),
        ),
    )

    saved_outputs: dict[str, dict[str, DummyIntervalSet]] = {}

    def _save_interval_table_output(analysis_path, *, output_name, intervals_by_epoch):
        saved_outputs[output_name] = intervals_by_epoch
        return analysis_path / f"{output_name}.parquet"

    monkeypatch.setattr(immobility_module, "save_interval_table_output", _save_interval_table_output)

    logged: dict[str, object] = {}

    def _write_run_log(**kwargs):
        logged.update(kwargs)
        return analysis_path / "v1ca1_log" / "immobility_log.json"

    monkeypatch.setattr(immobility_module, "write_run_log", _write_run_log)

    immobility_module.get_immobility_times(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
    )

    captured = capsys.readouterr()
    assert "Skipping epochs without usable head position" in captured.out
    assert len(observed_positions) == 1
    assert np.array_equal(
        observed_positions[0],
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
    )
    assert list(saved_outputs["run_times"]) == ["02_r1"]
    assert list(saved_outputs["immobility_times"]) == ["02_r1"]
    assert logged["outputs"]["selected_epochs"] == ["02_r1"]
    assert logged["outputs"]["skipped_epochs_unusable_head_position"] == [
        {
            "epoch": "01_s1",
            "reason": f"head position missing from {analysis_path / 'dlc_position_cleaned' / 'position.parquet'}",
        },
        {
            "epoch": "03_r2",
            "reason": f"head position is all NaN in {analysis_path / 'dlc_position_cleaned' / 'position.parquet'}",
        },
    ]


def test_parse_arguments_uses_shared_default_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_immobility_times.py",
            "--animal-name",
            "animal",
            "--date",
            "20240101",
        ],
    )

    args = parse_arguments()

    assert args.data_root == session.DEFAULT_DATA_ROOT
    assert DEFAULT_DATA_ROOT == session.DEFAULT_DATA_ROOT
    assert not hasattr(args, "save_pkl")
    assert not hasattr(args, "position_offset")
    assert not hasattr(args, "speed_threshold_cm_s")


def test_parse_arguments_rejects_removed_threshold_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_immobility_times.py",
            "--animal-name",
            "animal",
            "--date",
            "20240101",
            "--position-offset",
            "5",
            "--speed-threshold-cm-s",
            "3.0",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        parse_arguments()


def test_parse_arguments_rejects_removed_save_pkl_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_immobility_times.py",
            "--animal-name",
            "animal",
            "--date",
            "20240101",
            "--save-pkl",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()
