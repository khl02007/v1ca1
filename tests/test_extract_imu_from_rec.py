from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from v1ca1.helper.extract_imu_from_rec import (
    ACCEL_SCALE_G_PER_LSB,
    GYRO_SCALE_DEG_S_PER_LSB,
    IMU_TABLE_COLUMNS,
    build_imu_dataframe,
    extract_imu_from_rec,
    get_default_output_name,
    parse_arguments,
)


def _write_timestamps_ephys_pickle(
    analysis_path: Path,
    timestamps_by_epoch: dict[str, np.ndarray],
) -> None:
    with open(analysis_path / "timestamps_ephys.pkl", "wb") as file:
        pickle.dump(timestamps_by_epoch, file)



def test_build_imu_dataframe_rejects_mismatched_true_sample_counts() -> None:
    accel_timestamps = np.array([0.0, 0.1, 0.2])
    accel_samples = np.arange(9, dtype=np.int16).reshape(3, 3)
    gyro_timestamps = np.array([0.05, 0.15])
    gyro_samples = np.arange(6, dtype=np.int16).reshape(2, 3)

    with pytest.raises(ValueError, match="Accel and gyro true sample counts do not match"):
        build_imu_dataframe(
            accel_timestamps=accel_timestamps,
            accel_samples=accel_samples,
            gyro_timestamps=gyro_timestamps,
            gyro_samples=gyro_samples,
        )



def test_extract_imu_from_rec_writes_true_imu_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)
    _write_timestamps_ephys_pickle(
        analysis_path=analysis_path,
        timestamps_by_epoch={"02_r1": np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])},
    )

    rec_path = tmp_path / "epoch.rec"
    rec_path.write_bytes(b"fake")
    monkeypatch.setattr(
        "v1ca1.helper.extract_imu_from_rec.extract_true_imu_events",
        lambda path: {
            "num_packets": 6,
            "packet_sampling_rate_hz": 30000.0,
            "accel_packet_indices": np.array([0, 2, 4]),
            "accel_samples": np.array(
                [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
                dtype=np.int16,
            ),
            "gyro_packet_indices": np.array([1, 3, 5]),
            "gyro_samples": np.array(
                [[13, 14, 15], [23, 24, 25], [33, 34, 35]],
                dtype=np.int16,
            ),
        },
    )

    output_path = extract_imu_from_rec(
        rec_path=rec_path,
        animal_name="RatA",
        date="20240101",
        epoch="02_r1",
        data_root=tmp_path,
    )

    assert output_path == analysis_path / get_default_output_name("02_r1")

    table = pd.read_parquet(output_path)
    assert list(table.columns) == IMU_TABLE_COLUMNS
    assert table["accel_time"].tolist() == [1.0, 1.2, 1.4]
    assert table["gyro_time"].tolist() == [1.1, 1.3, 1.5]
    assert np.allclose(table["accel_x_g"].to_numpy(), np.array([10, 20, 30]) * ACCEL_SCALE_G_PER_LSB)
    assert np.allclose(table["gyro_z_deg_s"].to_numpy(), np.array([15, 25, 35]) * GYRO_SCALE_DEG_S_PER_LSB)



def test_extract_imu_from_rec_rejects_rec_packet_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)
    _write_timestamps_ephys_pickle(
        analysis_path=analysis_path,
        timestamps_by_epoch={"02_r1": np.array([0.0, 0.1, 0.2])},
    )

    rec_path = tmp_path / "epoch.rec"
    rec_path.write_bytes(b"fake")
    monkeypatch.setattr(
        "v1ca1.helper.extract_imu_from_rec.extract_true_imu_events",
        lambda path: {
            "num_packets": 4,
            "packet_sampling_rate_hz": 30000.0,
            "accel_packet_indices": np.array([0]),
            "accel_samples": np.array([[1, 2, 3]], dtype=np.int16),
            "gyro_packet_indices": np.array([1]),
            "gyro_samples": np.array([[4, 5, 6]], dtype=np.int16),
        },
    )

    with pytest.raises(ValueError, match="REC packet count"):
        extract_imu_from_rec(
            rec_path=rec_path,
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            data_root=tmp_path,
        )



def test_extract_imu_from_rec_rejects_missing_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)
    _write_timestamps_ephys_pickle(
        analysis_path=analysis_path,
        timestamps_by_epoch={"01_s1": np.array([0.0, 0.1])},
    )

    rec_path = tmp_path / "epoch.rec"
    rec_path.write_bytes(b"fake")
    monkeypatch.setattr(
        "v1ca1.helper.extract_imu_from_rec.extract_true_imu_events",
        lambda path: {
            "num_packets": 2,
            "packet_sampling_rate_hz": 30000.0,
            "accel_packet_indices": np.array([0]),
            "accel_samples": np.array([[1, 2, 3]], dtype=np.int16),
            "gyro_packet_indices": np.array([1]),
            "gyro_samples": np.array([[4, 5, 6]], dtype=np.int16),
        },
    )

    with pytest.raises(ValueError, match="Epoch '02_r1' not found"):
        extract_imu_from_rec(
            rec_path=rec_path,
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            data_root=tmp_path,
        )



def test_parse_arguments_requires_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_imu_from_rec.py",
            "--rec-path",
            "/tmp/example.rec",
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()
