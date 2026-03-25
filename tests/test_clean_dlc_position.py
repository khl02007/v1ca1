from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from v1ca1.position.clean_dlc_position import (
    DEFAULT_FIGURE_NAME_TEMPLATE,
    DEFAULT_OUTPUT_DIRNAME,
    DEFAULT_OUTPUT_NAME_TEMPLATE,
    clean_dlc_position,
    parse_arguments,
    run_joint_position_cleaning,
)


def _write_pickle(path: Path, value: object) -> None:
    with open(path, "wb") as file:
        pickle.dump(value, file)


def _write_session_files(
    analysis_path: Path,
    timestamps_position: dict[str, np.ndarray],
) -> None:
    analysis_path.mkdir(parents=True)
    _write_pickle(analysis_path / "timestamps_position.pkl", timestamps_position)


def _write_dlc_h5(
    path: Path,
    tracks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")

    columns: list[tuple[str, str, str]] = []
    data: list[np.ndarray] = []
    for bodypart, (x, y, likelihood) in tracks.items():
        columns.extend(
            [
                ("scorer", bodypart, "x"),
                ("scorer", bodypart, "y"),
                ("scorer", bodypart, "likelihood"),
            ]
        )
        data.extend(
            [
                np.asarray(x, dtype=float),
                np.asarray(y, dtype=float),
                np.asarray(likelihood, dtype=float),
            ]
        )

    table = pd.DataFrame(
        np.column_stack(data),
        columns=pd.MultiIndex.from_tuples(columns),
    )
    table.to_hdf(path, key="df", mode="w")


def test_run_joint_position_cleaning_interpolates_low_likelihood_and_jump_frames() -> None:
    pd = pytest.importorskip("pandas")

    frame_times = np.arange(5, dtype=float)
    diagnostics, metadata = run_joint_position_cleaning(
        head_table=pd.DataFrame(
            {
                "frame": np.arange(5, dtype=int),
                "position_x_raw": np.array([0.0, 1.0, 100.0, 3.0, 4.0]),
                "position_y_raw": np.zeros(5),
                "likelihood": np.array([0.999, 0.999, 0.01, 0.999, 0.999]),
            }
        ),
        body_table=pd.DataFrame(
            {
                "frame": np.arange(5, dtype=int),
                "position_x_raw": np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
                "position_y_raw": np.zeros(5),
                "likelihood": np.full(5, 0.999),
            }
        ),
        frame_times=frame_times,
        threshold_z=0.0,
        min_jump_threshold_px=10.0,
    )

    assert diagnostics["head_low_likelihood"].tolist() == [False, False, True, False, False]
    assert diagnostics["head_invalid"].tolist() == [False, False, True, False, False]
    assert diagnostics["head_interpolated"].tolist() == [False, False, True, False, False]
    assert diagnostics.loc[2, "head_x_cleaned"] == pytest.approx(2.0)
    assert diagnostics.loc[2, "head_interpolation_source_left"] == 1
    assert diagnostics.loc[2, "head_interpolation_source_right"] == 3
    assert metadata["head"]["counts"]["low_likelihood_frame_count"] == 1


def test_run_joint_position_cleaning_invalidates_lower_likelihood_label_on_pair_distance() -> None:
    pd = pytest.importorskip("pandas")

    diagnostics, metadata = run_joint_position_cleaning(
        head_table=pd.DataFrame(
            {
                "frame": np.arange(6, dtype=int),
                "position_x_raw": np.array([0.0, 1.0, 2.0, 120.0, 4.0, 5.0]),
                "position_y_raw": np.zeros(6),
                "likelihood": np.array([0.98, 0.99, 0.995, 0.97, 0.99, 0.98]),
            }
        ),
        body_table=pd.DataFrame(
            {
                "frame": np.arange(6, dtype=int),
                "position_x_raw": np.array([35.0, 36.0, 37.0, 38.0, 39.0, 40.0]),
                "position_y_raw": np.zeros(6),
                "likelihood": np.array([0.99, 0.99, 0.99, 0.99, 0.99, 0.99]),
            }
        ),
        frame_times=np.arange(6, dtype=float),
        threshold_z=10.0,
        min_jump_threshold_px=200.0,
    )

    assert diagnostics["pair_distance_invalid"].tolist() == [False, False, False, True, False, False]
    assert diagnostics["head_pair_invalid"].tolist() == [False, False, False, True, False, False]
    assert diagnostics["body_pair_invalid"].tolist() == [False, False, False, False, False, False]
    assert diagnostics.loc[3, "head_x_cleaned"] == pytest.approx(3.0)
    assert metadata["pair_distance"]["pair_distance_invalid_frame_count"] == 1


def test_run_joint_position_cleaning_prefers_body_on_equal_likelihood_pair_failures() -> None:
    pd = pytest.importorskip("pandas")

    diagnostics, _metadata = run_joint_position_cleaning(
        head_table=pd.DataFrame(
            {
                "frame": np.arange(4, dtype=int),
                "position_x_raw": np.array([0.0, 1.0, 120.0, 3.0]),
                "position_y_raw": np.zeros(4),
                "likelihood": np.full(4, 0.999),
            }
        ),
        body_table=pd.DataFrame(
            {
                "frame": np.arange(4, dtype=int),
                "position_x_raw": np.array([35.0, 36.0, 37.0, 38.0]),
                "position_y_raw": np.zeros(4),
                "likelihood": np.full(4, 0.999),
            }
        ),
        frame_times=np.arange(4, dtype=float),
        threshold_z=10.0,
        min_jump_threshold_px=200.0,
    )

    assert diagnostics["pair_distance_invalid"].tolist() == [False, False, True, False]
    assert diagnostics["head_pair_invalid"].tolist() == [False, False, True, False]
    assert diagnostics["body_pair_invalid"].tolist() == [False, False, False, False]


def test_clean_dlc_position_writes_joint_epoch_parquet_and_figure(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    frame_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": frame_times},
    )

    dlc_h5_path = tmp_path / "joint.h5"
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (
                np.array([0.0, 1.0, 100.0, 3.0, 4.0]),
                np.zeros(5),
                np.array([0.999, 0.999, 0.01, 0.999, 0.999]),
            ),
            "body": (
                np.array([35.0, 36.0, 37.0, 38.0, 39.0]),
                np.zeros(5),
                np.full(5, 0.999),
            ),
        },
    )

    output_path = clean_dlc_position(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        dlc_h5_path=dlc_h5_path,
        data_root=tmp_path,
        threshold_z=0.0,
        min_jump_threshold_px=10.0,
    )

    expected_path = analysis_path / DEFAULT_OUTPUT_DIRNAME / DEFAULT_OUTPUT_NAME_TEMPLATE.format(
        epoch="02_r1"
    )
    assert output_path == expected_path
    diagnostics = pd.read_parquet(output_path)
    assert diagnostics.columns.tolist() == [
        "epoch",
        "frame",
        "frame_time_s",
        "head_x_raw",
        "head_y_raw",
        "head_likelihood",
        "head_likelihood_loss",
        "head_x_cleaned",
        "head_y_cleaned",
        "head_step_prev_px",
        "head_step_next_px",
        "head_low_likelihood",
        "head_jump_invalid",
        "head_pair_invalid",
        "head_invalid",
        "head_interpolated",
        "head_interpolation_source_left",
        "head_interpolation_source_right",
        "body_x_raw",
        "body_y_raw",
        "body_likelihood",
        "body_likelihood_loss",
        "body_x_cleaned",
        "body_y_cleaned",
        "body_step_prev_px",
        "body_step_next_px",
        "body_low_likelihood",
        "body_jump_invalid",
        "body_pair_invalid",
        "body_invalid",
        "body_interpolated",
        "body_interpolation_source_left",
        "body_interpolation_source_right",
        "head_body_distance_px",
        "pair_distance_invalid",
    ]
    assert diagnostics["epoch"].unique().tolist() == ["02_r1"]
    assert diagnostics.loc[2, "head_x_cleaned"] == pytest.approx(2.0)
    figure_path = analysis_path / "figs" / DEFAULT_OUTPUT_DIRNAME / DEFAULT_FIGURE_NAME_TEMPLATE.format(
        epoch="02_r1"
    )
    assert figure_path.exists()

    log_files = sorted(
        (analysis_path / "v1ca1_log").glob("v1ca1_position_clean_dlc_position_*.json")
    )
    assert len(log_files) == 1
    log_text = log_files[0].read_text(encoding="utf-8")
    assert "clean_dlc_position" in log_text
    assert str(output_path) in log_text
    assert str(figure_path) in log_text
    assert '"pair_distance"' in log_text


def test_clean_dlc_position_cleans_full_epoch_without_internal_offset(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    frame_times = np.linspace(0.0, 14.0, 15, dtype=float)
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": frame_times},
    )

    dlc_h5_path = tmp_path / "joint_full.h5"
    head_x = np.linspace(0.0, 14.0, frame_times.size)
    head_x[2] = 100.0
    body_x = np.linspace(30.0, 44.0, frame_times.size)
    body_x[2] = 130.0
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (head_x, np.zeros(frame_times.size), np.full(frame_times.size, 0.999)),
            "body": (body_x, np.zeros(frame_times.size), np.full(frame_times.size, 0.999)),
        },
    )

    output_path = clean_dlc_position(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        dlc_h5_path=dlc_h5_path,
        data_root=tmp_path,
        threshold_z=0.0,
        min_jump_threshold_px=10.0,
    )

    cleaned_position = pd.read_parquet(output_path)
    assert cleaned_position.loc[2, "head_x_cleaned"] == pytest.approx(2.0)
    assert cleaned_position.loc[2, "body_x_cleaned"] == pytest.approx(32.0)


def test_clean_dlc_position_rejects_missing_bodypart(tmp_path: Path) -> None:
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    dlc_h5_path = tmp_path / "head_only.h5"
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (
                np.array([0.0, 1.0]),
                np.zeros(2),
                np.full(2, 0.999),
            ),
        },
    )

    with pytest.raises(ValueError, match="required 'body' bodypart"):
        clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
        )


def test_clean_dlc_position_rejects_dlc_timestamp_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0, 2.0], dtype=float)},
    )
    dlc_h5_path = tmp_path / "joint_bad.h5"
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (np.array([0.0, 1.0]), np.zeros(2), np.full(2, 0.999)),
            "body": (np.array([30.0, 31.0]), np.zeros(2), np.full(2, 0.999)),
        },
    )

    with pytest.raises(ValueError, match="Head DLC row count does not match saved position timestamps"):
        clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
        )


def test_clean_dlc_position_rejects_missing_epoch(tmp_path: Path) -> None:
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"01_s1": np.array([0.0, 1.0], dtype=float)},
    )
    dlc_h5_path = tmp_path / "joint.h5"
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (np.array([0.0, 1.0]), np.zeros(2), np.full(2, 0.999)),
            "body": (np.array([30.0, 31.0]), np.zeros(2), np.full(2, 0.999)),
        },
    )

    with pytest.raises(ValueError, match="Epoch '02_r1' not found"):
        clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
        )


def test_parse_arguments_requires_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_dlc_position.py",
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
            "--dlc-h5-path",
            "/tmp/head.h5",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_parse_arguments_does_not_expose_position_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "clean_dlc_position.py",
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
            "--epoch",
            "02_r1",
            "--dlc-h5-path",
            "/tmp/head.h5",
        ],
    )

    args = parse_arguments()
    assert not hasattr(args, "position_offset")
