from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from v1ca1.helper.session import DEFAULT_POSITION_OFFSET
from v1ca1.position.fuse_head_position_imu import (
    DEFAULT_OUTPUT_NAME,
    build_frame_imu_features,
    fuse_head_position_imu,
    load_head_dlc,
    load_imu_table,
    parse_arguments,
    run_head_position_fusion,
)


def _write_pickle(path: Path, value: object) -> None:
    with open(path, "wb") as file:
        pickle.dump(value, file)


def _write_session_files(
    analysis_path: Path,
    timestamps_position: dict[str, np.ndarray],
    timestamps_ephys: dict[str, np.ndarray],
    position_by_epoch: dict[str, np.ndarray],
) -> None:
    analysis_path.mkdir(parents=True)
    _write_pickle(analysis_path / "timestamps_position.pkl", timestamps_position)
    _write_pickle(analysis_path / "timestamps_ephys.pkl", timestamps_ephys)
    _write_pickle(analysis_path / "position.pkl", position_by_epoch)


def _write_dlc_h5(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    likelihood: np.ndarray,
    *,
    bodypart: str = "head",
    include_likelihood: bool = True,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")

    columns = [
        ("scorer", bodypart, "x"),
        ("scorer", bodypart, "y"),
    ]
    data = [np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
    if include_likelihood:
        columns.append(("scorer", bodypart, "likelihood"))
        data.append(np.asarray(likelihood, dtype=float))

    table = pd.DataFrame(
        np.column_stack(data),
        columns=pd.MultiIndex.from_tuples(columns),
    )
    table.to_hdf(path, key="df", mode="w")


def _write_imu_parquet(
    path: Path,
    *,
    accel_time: np.ndarray,
    gyro_time: np.ndarray,
    accel_x: np.ndarray,
    accel_y: np.ndarray,
    accel_z: np.ndarray,
    gyro_x: np.ndarray,
    gyro_y: np.ndarray,
    gyro_z: np.ndarray,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    table = pd.DataFrame(
        {
            "accel_time": np.asarray(accel_time, dtype=float),
            "gyro_time": np.asarray(gyro_time, dtype=float),
            "accel_x_g": np.asarray(accel_x, dtype=float),
            "accel_y_g": np.asarray(accel_y, dtype=float),
            "accel_z_g": np.asarray(accel_z, dtype=float),
            "gyro_x_deg_s": np.asarray(gyro_x, dtype=float),
            "gyro_y_deg_s": np.asarray(gyro_y, dtype=float),
            "gyro_z_deg_s": np.asarray(gyro_z, dtype=float),
        }
    )
    table.to_parquet(path, index=False)


def test_build_frame_imu_features_carries_forward_values_when_no_updates() -> None:
    pd = pytest.importorskip("pandas")

    imu_table = pd.DataFrame(
        {
            "accel_time": [0.05, 0.15, 0.30],
            "gyro_time": [0.08, 0.28, 0.40],
            "accel_x_g": [1.0, 3.0, 5.0],
            "accel_y_g": [0.0, 0.0, 0.0],
            "accel_z_g": [0.0, 0.0, 0.0],
            "gyro_x_deg_s": [10.0, 30.0, 30.0],
            "gyro_y_deg_s": [0.0, 0.0, 0.0],
            "gyro_z_deg_s": [0.0, 0.0, 0.0],
        }
    )

    frame_features = build_frame_imu_features(
        imu_table=imu_table,
        frame_times=np.array([0.10, 0.20, 0.35]),
        epoch_start_time=0.0,
    )

    assert np.allclose(frame_features["accel_x_mean"].to_numpy(), [1.0, 2.0, 11.0 / 3.0])
    assert np.allclose(frame_features["gyro_x_mean"].to_numpy(), [10.0, 10.0, 58.0 / 3.0])
    assert frame_features["accel_update_count"].tolist() == [1, 1, 1]
    assert frame_features["gyro_update_count"].tolist() == [1, 0, 1]
    assert np.allclose(frame_features["accel_mag_std"].to_numpy(), [0.0, 1.0, np.sqrt(8.0 / 9.0)])


def test_build_frame_imu_features_allows_first_frame_before_epoch_start() -> None:
    pd = pytest.importorskip("pandas")

    imu_table = pd.DataFrame(
        {
            "accel_time": [0.12, 0.22],
            "gyro_time": [0.12, 0.22],
            "accel_x_g": [1.0, 2.0],
            "accel_y_g": [0.0, 0.0],
            "accel_z_g": [0.0, 0.0],
            "gyro_x_deg_s": [10.0, 20.0],
            "gyro_y_deg_s": [0.0, 0.0],
            "gyro_z_deg_s": [0.0, 0.0],
        }
    )

    frame_features = build_frame_imu_features(
        imu_table=imu_table,
        frame_times=np.array([0.10, 0.20, 0.30]),
        epoch_start_time=0.11,
    )

    assert frame_features.shape[0] == 3
    assert np.isfinite(frame_features.to_numpy(dtype=float)).all()
    assert frame_features.loc[0, "accel_update_count"] == 0


def test_run_head_position_fusion_marks_short_and_long_gaps() -> None:
    pd = pytest.importorskip("pandas")

    frame_times = np.arange(0.0, 1.0, 0.1)
    dlc_table = pd.DataFrame(
        {
            "frame": np.arange(frame_times.size, dtype=int),
            "head_x_raw": np.linspace(0.0, 9.0, frame_times.size),
            "head_y_raw": np.linspace(0.0, 4.5, frame_times.size),
            "likelihood": np.array([0.99, 0.99, 0.20, 0.99, 0.99, 0.20, 0.20, 0.20, 0.99, 0.99]),
        }
    )
    imu_features = pd.DataFrame(
        {
            "accel_x_mean": np.zeros(frame_times.size),
            "accel_y_mean": np.zeros(frame_times.size),
            "accel_z_mean": np.ones(frame_times.size),
            "gyro_x_mean": np.linspace(0.0, 1.0, frame_times.size),
            "gyro_y_mean": np.zeros(frame_times.size),
            "gyro_z_mean": np.zeros(frame_times.size),
            "accel_mag_mean": np.ones(frame_times.size),
            "accel_mag_std": np.full(frame_times.size, 0.1),
            "gyro_mag_mean": np.linspace(0.0, 1.0, frame_times.size),
            "accel_update_count": np.ones(frame_times.size, dtype=int),
            "gyro_update_count": np.ones(frame_times.size, dtype=int),
        }
    )

    results, metadata = run_head_position_fusion(
        dlc_table=dlc_table,
        frame_times=frame_times,
        imu_features=imu_features,
        max_gap_s=0.3,
    )

    assert bool(results.loc[2, "gap_filled"])
    assert not bool(results.loc[2, "long_gap"])
    assert np.isfinite(results.loc[2, "head_x_fused"])
    assert results.loc[5:7, "long_gap"].tolist() == [True, True, True]
    assert results.loc[5:7, "gap_filled"].tolist() == [False, False, False]
    assert np.isnan(results.loc[5:7, "head_x_fused"]).all()
    assert metadata["gap_filled_frame_count"] == 1
    assert metadata["long_gap_frame_count"] == 3


def test_fuse_head_position_imu_writes_sidecar_pickle_and_preserves_other_epochs(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    frame_times = np.array([0.10, 0.20, 0.30, 0.40, 0.50], dtype=float)
    timestamps_position = {
        "01_s1": np.array([1.0, 1.5], dtype=float),
        "02_r1": frame_times,
    }
    timestamps_ephys = {
        "01_s1": np.array([1.0, 1.1], dtype=float),
        "02_r1": np.linspace(0.0, 0.50, 51),
    }
    original_run_position = np.column_stack(
        [np.linspace(100.0, 104.0, frame_times.size), np.linspace(50.0, 54.0, frame_times.size)]
    )
    other_epoch_position = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position=timestamps_position,
        timestamps_ephys=timestamps_ephys,
        position_by_epoch={"01_s1": other_epoch_position, "02_r1": original_run_position},
    )

    dlc_h5_path = tmp_path / "head.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        y=np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
        likelihood=np.array([0.99, 0.98, 0.99, 0.97, 0.99]),
    )
    imu_path = analysis_path / "02_r1_imu.parquet"
    _write_imu_parquet(
        imu_path,
        accel_time=np.array([0.02, 0.12, 0.22, 0.32, 0.42]),
        gyro_time=np.array([0.04, 0.14, 0.24, 0.34, 0.44]),
        accel_x=np.array([0.0, 0.1, 0.2, 0.3, 0.4]),
        accel_y=np.zeros(5),
        accel_z=np.ones(5),
        gyro_x=np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
        gyro_y=np.zeros(5),
        gyro_z=np.zeros(5),
    )

    output_path = fuse_head_position_imu(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        dlc_h5_path=dlc_h5_path,
        data_root=tmp_path,
        position_offset=0,
    )

    assert output_path == analysis_path / DEFAULT_OUTPUT_NAME
    with open(output_path, "rb") as file:
        fused_position = pickle.load(file)

    assert sorted(fused_position) == ["01_s1", "02_r1"]
    assert np.allclose(fused_position["01_s1"], other_epoch_position)
    assert fused_position["02_r1"].shape == (frame_times.size, 2)
    assert np.isfinite(fused_position["02_r1"]).all()
    assert not np.allclose(fused_position["02_r1"], original_run_position)
    log_files = sorted((analysis_path / "v1ca1_log").glob("v1ca1_position_fuse_head_position_imu_*.json"))
    assert len(log_files) == 1
    log_text = log_files[0].read_text(encoding="utf-8")
    assert "fuse_head_position_imu" in log_text
    assert str(output_path) in log_text
    assert str(imu_path) in log_text


def test_fuse_head_position_imu_uses_default_position_offset_for_fusion(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    frame_times = np.linspace(0.10, 1.50, 15, dtype=float)
    timestamps_position = {"02_r1": frame_times}
    timestamps_ephys = {"02_r1": np.linspace(0.0, 1.50, 151)}
    original_run_position = np.column_stack(
        [np.linspace(100.0, 114.0, frame_times.size), np.linspace(200.0, 214.0, frame_times.size)]
    )
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position=timestamps_position,
        timestamps_ephys=timestamps_ephys,
        position_by_epoch={"02_r1": original_run_position},
    )

    dlc_h5_path = tmp_path / "head_default_offset.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.linspace(0.0, 14.0, frame_times.size),
        y=np.linspace(50.0, 64.0, frame_times.size),
        likelihood=np.full(frame_times.size, 0.99),
    )
    imu_path = analysis_path / "02_r1_imu.parquet"
    _write_imu_parquet(
        imu_path,
        accel_time=np.linspace(0.02, 1.42, frame_times.size),
        gyro_time=np.linspace(0.04, 1.44, frame_times.size),
        accel_x=np.linspace(0.0, 0.7, frame_times.size),
        accel_y=np.zeros(frame_times.size),
        accel_z=np.ones(frame_times.size),
        gyro_x=np.linspace(0.0, 1.4, frame_times.size),
        gyro_y=np.zeros(frame_times.size),
        gyro_z=np.zeros(frame_times.size),
    )

    output_path = fuse_head_position_imu(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        dlc_h5_path=dlc_h5_path,
        data_root=tmp_path,
    )

    with open(output_path, "rb") as file:
        fused_position = pickle.load(file)

    assert fused_position["02_r1"].shape == (frame_times.size, 2)
    assert np.allclose(
        fused_position["02_r1"][:DEFAULT_POSITION_OFFSET],
        original_run_position[:DEFAULT_POSITION_OFFSET],
    )
    assert np.isfinite(fused_position["02_r1"][DEFAULT_POSITION_OFFSET:]).all()
    assert not np.allclose(
        fused_position["02_r1"][DEFAULT_POSITION_OFFSET:],
        original_run_position[DEFAULT_POSITION_OFFSET:],
    )

    log_files = sorted((analysis_path / "v1ca1_log").glob("v1ca1_position_fuse_head_position_imu_*.json"))
    assert len(log_files) == 1
    log_text = log_files[0].read_text(encoding="utf-8")
    assert f'"position_offset": {DEFAULT_POSITION_OFFSET}' in log_text
    assert '"trimmed_frame_count": 5' in log_text


def test_fuse_head_position_imu_writes_one_epoch_output_when_position_pickle_is_missing(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)
    frame_times = np.array([0.10, 0.20, 0.30, 0.40], dtype=float)
    _write_pickle(analysis_path / "timestamps_position.pkl", {"02_r1": frame_times})
    _write_pickle(analysis_path / "timestamps_ephys.pkl", {"02_r1": np.linspace(0.0, 0.40, 41)})

    dlc_h5_path = tmp_path / "head_only.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.array([5.0, 6.0, 7.0, 8.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
        likelihood=np.array([0.99, 0.95, 0.97, 0.99]),
    )
    imu_path = analysis_path / "02_r1_imu.parquet"
    _write_imu_parquet(
        imu_path,
        accel_time=np.array([0.02, 0.12, 0.22, 0.32]),
        gyro_time=np.array([0.04, 0.14, 0.24, 0.34]),
        accel_x=np.array([0.0, 0.1, 0.2, 0.3]),
        accel_y=np.zeros(4),
        accel_z=np.ones(4),
        gyro_x=np.array([0.0, 0.1, 0.2, 0.3]),
        gyro_y=np.zeros(4),
        gyro_z=np.zeros(4),
    )

    output_path = fuse_head_position_imu(
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        dlc_h5_path=dlc_h5_path,
        data_root=tmp_path,
        position_offset=0,
    )

    assert output_path == analysis_path / DEFAULT_OUTPUT_NAME
    with open(output_path, "rb") as file:
        fused_position = pickle.load(file)

    assert sorted(fused_position) == ["02_r1"]
    assert fused_position["02_r1"].shape == (frame_times.size, 2)
    assert np.isfinite(fused_position["02_r1"]).all()

    log_files = sorted((analysis_path / "v1ca1_log").glob("v1ca1_position_fuse_head_position_imu_*.json"))
    assert len(log_files) == 1
    log_text = log_files[0].read_text(encoding="utf-8")
    assert '"base_position_source": "missing"' in log_text


def test_fuse_head_position_imu_rejects_missing_epoch(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"01_s1": np.array([0.1, 0.2])},
        timestamps_ephys={"01_s1": np.array([0.0, 0.1, 0.2])},
        position_by_epoch={"01_s1": np.array([[0.0, 0.0], [1.0, 1.0]])},
    )
    dlc_h5_path = tmp_path / "head.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        likelihood=np.array([0.99, 0.99]),
    )
    _write_imu_parquet(
        analysis_path / "01_s1_imu.parquet",
        accel_time=np.array([0.05]),
        gyro_time=np.array([0.05]),
        accel_x=np.array([0.0]),
        accel_y=np.array([0.0]),
        accel_z=np.array([1.0]),
        gyro_x=np.array([0.0]),
        gyro_y=np.array([0.0]),
        gyro_z=np.array([0.0]),
    )

    with pytest.raises(ValueError, match="Epoch '02_r1' not found"):
        fuse_head_position_imu(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
            position_offset=0,
        )


def test_fuse_head_position_imu_rejects_dlc_timestamp_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.1, 0.2, 0.3])},
        timestamps_ephys={"02_r1": np.array([0.0, 0.1, 0.2, 0.3])},
        position_by_epoch={"02_r1": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])},
    )
    dlc_h5_path = tmp_path / "head.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        likelihood=np.array([0.99, 0.99]),
    )
    _write_imu_parquet(
        analysis_path / "02_r1_imu.parquet",
        accel_time=np.array([0.05, 0.15]),
        gyro_time=np.array([0.05, 0.15]),
        accel_x=np.array([0.0, 0.1]),
        accel_y=np.array([0.0, 0.0]),
        accel_z=np.array([1.0, 1.0]),
        gyro_x=np.array([0.0, 0.1]),
        gyro_y=np.array([0.0, 0.0]),
        gyro_z=np.array([0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="DLC row count does not match saved position timestamps"):
        fuse_head_position_imu(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
            position_offset=0,
        )


def test_load_head_dlc_rejects_missing_head_bodypart(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("tables")

    dlc_h5_path = tmp_path / "nose.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        likelihood=np.array([0.99, 0.99]),
        bodypart="nose",
    )

    with pytest.raises(ValueError, match="required 'head' bodypart"):
        load_head_dlc(dlc_h5_path)


def test_load_head_dlc_rejects_missing_required_columns(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("tables")

    dlc_h5_path = tmp_path / "head_missing_like.h5"
    _write_dlc_h5(
        dlc_h5_path,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        likelihood=np.array([0.99, 0.99]),
        include_likelihood=False,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_head_dlc(dlc_h5_path)


def test_load_imu_table_rejects_missing_required_columns(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    imu_path = tmp_path / "bad_imu.parquet"
    pd.DataFrame(
        {
            "accel_time": [0.1],
            "gyro_time": [0.1],
            "accel_x_g": [0.0],
        }
    ).to_parquet(imu_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_imu_table(imu_path)


def test_parse_arguments_requires_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fuse_head_position_imu.py",
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
