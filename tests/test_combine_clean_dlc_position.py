from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from v1ca1.helper.session import load_clean_dlc_position_data
from v1ca1.position.clean_dlc_position import DEFAULT_OUTPUT_DIRNAME
from v1ca1.position.combine_clean_dlc_position import (
    DEFAULT_OUTPUT_NAME,
    combine_clean_dlc_position,
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


def _write_cleaned_epoch_parquet(
    path: Path,
    epoch: str,
    frame_times: np.ndarray,
    head_xy: np.ndarray,
    body_xy: np.ndarray,
    head_invalid: np.ndarray | None = None,
    body_invalid: np.ndarray | None = None,
    head_interpolated: np.ndarray | None = None,
    body_interpolated: np.ndarray | None = None,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    frame_times_array = np.asarray(frame_times, dtype=float)
    head_array = np.asarray(head_xy, dtype=float)
    body_array = np.asarray(body_xy, dtype=float)
    n_frames = frame_times_array.size
    if head_invalid is None:
        head_invalid = np.zeros(n_frames, dtype=bool)
    if body_invalid is None:
        body_invalid = np.zeros(n_frames, dtype=bool)
    if head_interpolated is None:
        head_interpolated = np.zeros(n_frames, dtype=bool)
    if body_interpolated is None:
        body_interpolated = np.zeros(n_frames, dtype=bool)

    table = pd.DataFrame(
        {
            "epoch": epoch,
            "frame": np.arange(n_frames, dtype=int),
            "frame_time_s": frame_times_array,
            "head_x_cleaned": head_array[:, 0],
            "head_y_cleaned": head_array[:, 1],
            "body_x_cleaned": body_array[:, 0],
            "body_y_cleaned": body_array[:, 1],
            "head_invalid": np.asarray(head_invalid, dtype=bool),
            "body_invalid": np.asarray(body_invalid, dtype=bool),
            "head_interpolated": np.asarray(head_interpolated, dtype=bool),
            "body_interpolated": np.asarray(body_interpolated, dtype=bool),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def _write_combined_cleaned_parquet(
    path: Path,
    epoch: str,
    frame_times: np.ndarray,
    head_xy: np.ndarray,
    body_xy: np.ndarray,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    frame_times_array = np.asarray(frame_times, dtype=float)
    head_array = np.asarray(head_xy, dtype=float)
    body_array = np.asarray(body_xy, dtype=float)
    n_frames = frame_times_array.size
    table = pd.DataFrame(
        {
            "epoch": epoch,
            "frame": np.arange(n_frames, dtype=int),
            "frame_time_s": frame_times_array,
            "head_x": head_array[:, 0],
            "head_y": head_array[:, 1],
            "body_x": body_array[:, 0],
            "body_y": body_array[:, 1],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def test_combine_clean_dlc_position_writes_one_combined_parquet_and_loader_round_trips(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_position = {
        "01_s1": np.array([0.0, 1.0], dtype=float),
        "02_r1": np.array([10.0, 11.0, 12.0], dtype=float),
        "03_s2": np.array([20.0, 21.0], dtype=float),
    }
    _write_session_files(analysis_path=analysis_path, timestamps_position=timestamps_position)

    output_dir = analysis_path / DEFAULT_OUTPUT_DIRNAME
    _write_cleaned_epoch_parquet(
        output_dir / "02_r1_dlc_position_cleaned.parquet",
        epoch="02_r1",
        frame_times=timestamps_position["02_r1"],
        head_xy=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
        body_xy=np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=float),
        head_invalid=np.array([False, True, False]),
        head_interpolated=np.array([False, True, False]),
    )
    _write_cleaned_epoch_parquet(
        output_dir / "03_s2_dlc_position_cleaned.parquet",
        epoch="03_s2",
        frame_times=timestamps_position["03_s2"],
        head_xy=np.array([[13.0, 14.0], [15.0, 16.0]], dtype=float),
        body_xy=np.array([[17.0, 18.0], [19.0, 20.0]], dtype=float),
    )

    output_path = combine_clean_dlc_position(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
    )

    assert output_path == output_dir / DEFAULT_OUTPUT_NAME
    combined = pd.read_parquet(output_path)
    assert combined.columns.tolist() == [
        "epoch",
        "frame",
        "frame_time_s",
        "head_x",
        "head_y",
        "body_x",
        "body_y",
    ]
    assert combined["epoch"].tolist() == ["02_r1", "02_r1", "02_r1", "03_s2", "03_s2"]
    assert combined.shape[0] == 5
    assert np.allclose(combined.loc[0:2, "frame_time_s"].to_numpy(dtype=float), timestamps_position["02_r1"])

    epoch_order, head_position, body_position = load_clean_dlc_position_data(analysis_path)
    assert epoch_order == ["02_r1", "03_s2"]
    assert np.allclose(head_position["02_r1"], np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert np.allclose(body_position["03_s2"], np.array([[17.0, 18.0], [19.0, 20.0]]))


def test_combine_clean_dlc_position_rejects_missing_timestamp_epoch(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    output_dir = analysis_path / DEFAULT_OUTPUT_DIRNAME
    _write_cleaned_epoch_parquet(
        output_dir / "03_s2_dlc_position_cleaned.parquet",
        epoch="03_s2",
        frame_times=np.array([2.0, 3.0], dtype=float),
        head_xy=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        body_xy=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="was not found in saved position timestamps"):
        combine_clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
        )


def test_combine_clean_dlc_position_rejects_duplicate_epoch_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_position = {"02_r1": np.array([0.0, 1.0], dtype=float)}
    _write_session_files(analysis_path=analysis_path, timestamps_position=timestamps_position)
    output_dir = analysis_path / DEFAULT_OUTPUT_DIRNAME
    _write_cleaned_epoch_parquet(
        output_dir / "02_r1_dlc_position_cleaned.parquet",
        epoch="02_r1",
        frame_times=timestamps_position["02_r1"],
        head_xy=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        body_xy=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
    )
    _write_cleaned_epoch_parquet(
        output_dir / "02_r1_copy_dlc_position_cleaned.parquet",
        epoch="02_r1",
        frame_times=timestamps_position["02_r1"],
        head_xy=np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
        body_xy=np.array([[13.0, 14.0], [15.0, 16.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="duplicate cleaned DLC outputs"):
        combine_clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
        )


def test_combine_clean_dlc_position_rejects_empty_input_dir(tmp_path: Path) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    (analysis_path / DEFAULT_OUTPUT_DIRNAME).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="No cleaned DLC parquet files"):
        combine_clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
        )


def test_load_clean_dlc_position_data_rejects_timestamp_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    output_dir = analysis_path / DEFAULT_OUTPUT_DIRNAME
    _write_combined_cleaned_parquet(
        output_dir / DEFAULT_OUTPUT_NAME,
        epoch="02_r1",
        frame_times=np.array([0.0, 2.0], dtype=float),
        head_xy=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        body_xy=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
    )

    with pytest.raises(ValueError, match="timestamps do not match saved position timestamps"):
        load_clean_dlc_position_data(analysis_path)
