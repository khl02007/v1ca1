from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from v1ca1.helper.session import load_position_data_with_precedence
from v1ca1.position.fit_wtrack_geometry import (
    compute_total_wtrack_length,
    fit_wtrack_geometry,
    fit_wtrack_geometry_from_positions,
    make_wtrack_geometry_dict,
    sample_wtrack_skeleton,
)


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(value, file)


def _write_timestamps(analysis_path: Path, epochs: list[str]) -> None:
    timestamps = {
        epoch: np.linspace(index * 10.0, index * 10.0 + 1.0, 5, dtype=float)
        for index, epoch in enumerate(epochs)
    }
    _write_pickle(analysis_path / "timestamps_position.pkl", timestamps)
    _write_pickle(analysis_path / "timestamps_ephys.pkl", timestamps)


def _write_position_pickle(path: Path, values_by_epoch: dict[str, np.ndarray]) -> None:
    _write_pickle(path, {epoch: np.asarray(values, dtype=float) for epoch, values in values_by_epoch.items()})


def _write_cleaned_position_parquet(
    path: Path,
    epoch: str,
    frame_times: np.ndarray,
    head_xy: np.ndarray,
    body_xy: np.ndarray,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    table = pd.DataFrame(
        {
            "epoch": epoch,
            "frame": np.arange(len(frame_times), dtype=int),
            "frame_time_s": np.asarray(frame_times, dtype=float),
            "head_x_cm": np.asarray(head_xy, dtype=float)[:, 0],
            "head_y_cm": np.asarray(head_xy, dtype=float)[:, 1],
            "body_x_cm": np.asarray(body_xy, dtype=float)[:, 0],
            "body_y_cm": np.asarray(body_xy, dtype=float)[:, 1],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def _make_synthetic_position(theta: float = -0.85) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(7)
    parameters = {
        "center_x": 41.0,
        "center_y": 34.0,
        "theta": theta,
        "long_segment_length": 62.0,
        "short_segment_length": 16.0,
        "dx": 10.0,
        "dy": 8.0,
    }
    geometry = make_wtrack_geometry_dict(
        center_well_xy=np.array([parameters["center_x"], parameters["center_y"]], dtype=float),
        theta=parameters["theta"],
        long_segment_length=parameters["long_segment_length"],
        short_segment_length=parameters["short_segment_length"],
        dx=parameters["dx"],
        dy=parameters["dy"],
    )
    node_positions = {
        "left": np.asarray(geometry["node_positions_left"], dtype=float),
        "right": np.asarray(geometry["node_positions_right"], dtype=float),
    }
    skeleton_points = sample_wtrack_skeleton(node_positions, step_cm=1.5)
    repeated = np.repeat(skeleton_points, 3, axis=0)
    noisy = repeated + rng.normal(scale=0.6, size=repeated.shape)
    return noisy, parameters


def _node_rmse(geometry_a: dict[str, object], geometry_b: dict[str, object]) -> float:
    left_a = np.asarray(geometry_a["node_positions_left"], dtype=float)
    right_a = np.asarray(geometry_a["node_positions_right"], dtype=float)
    left_b = np.asarray(geometry_b["node_positions_left"], dtype=float)
    right_b = np.asarray(geometry_b["node_positions_right"], dtype=float)
    direct = np.vstack([left_a, right_a])
    target = np.vstack([left_b, right_b])
    swapped = np.vstack([right_b, left_b])
    direct_rmse = float(np.sqrt(np.mean(np.sum((direct - target) ** 2, axis=1))))
    swapped_rmse = float(np.sqrt(np.mean(np.sum((direct - swapped) ** 2, axis=1))))
    return min(direct_rmse, swapped_rmse)


def test_load_position_data_with_precedence_prefers_cleaned_dlc(tmp_path: Path) -> None:
    analysis_path = tmp_path / "L14" / "20240611"
    epochs = ["02_r1"]
    _write_timestamps(analysis_path, epochs)

    frame_times = np.linspace(0.0, 1.0, 5, dtype=float)
    _write_cleaned_position_parquet(
        analysis_path / "dlc_position_cleaned" / "position.parquet",
        epoch="02_r1",
        frame_times=frame_times,
        head_xy=np.column_stack([np.arange(5, dtype=float), np.arange(10, 15, dtype=float)]),
        body_xy=np.column_stack([np.arange(20, 25, dtype=float), np.arange(30, 35, dtype=float)]),
    )
    _write_position_pickle(
        analysis_path / "position.pkl",
        {"02_r1": np.full((5, 2), fill_value=999.0, dtype=float)},
    )
    _write_position_pickle(
        analysis_path / "body_position.pkl",
        {"02_r1": np.full((5, 2), fill_value=555.0, dtype=float)},
    )

    position_by_epoch, source = load_position_data_with_precedence(analysis_path)

    assert source.endswith("dlc_position_cleaned/position.parquet")
    assert np.allclose(position_by_epoch["02_r1"], [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]])


def test_load_position_data_with_precedence_falls_back_to_position_pickle(tmp_path: Path) -> None:
    analysis_path = tmp_path / "L14" / "20240611"
    _write_position_pickle(
        analysis_path / "position.pkl",
        {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
    )
    _write_position_pickle(
        analysis_path / "body_position.pkl",
        {"02_r1": np.array([[50.0, 60.0], [70.0, 80.0]], dtype=float)},
    )

    position_by_epoch, source = load_position_data_with_precedence(analysis_path)

    assert source.endswith("position.pkl")
    assert np.allclose(position_by_epoch["02_r1"], [[1.0, 2.0], [3.0, 4.0]])


def test_load_position_data_with_precedence_falls_back_to_body_position_pickle(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "L14" / "20240611"
    _write_position_pickle(
        analysis_path / "body_position.pkl",
        {"02_r1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)},
    )

    position_by_epoch, source = load_position_data_with_precedence(analysis_path)

    assert source.endswith("body_position.pkl")
    assert np.allclose(position_by_epoch["02_r1"], [[5.0, 6.0], [7.0, 8.0]])


def test_load_position_data_with_precedence_fails_when_no_source_exists(tmp_path: Path) -> None:
    analysis_path = tmp_path / "L14" / "20240611"
    analysis_path.mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match="Could not find cleaned DLC position, position.pkl, or body_position.pkl",
    ):
        load_position_data_with_precedence(analysis_path)


def test_fit_wtrack_geometry_recovers_synthetic_geometry() -> None:
    synthetic_position, parameters = _make_synthetic_position()

    fit_result = fit_wtrack_geometry_from_positions(
        synthetic_position,
        occupancy_bin_size_cm=1.5,
        skeleton_step_cm=1.5,
    )

    true_geometry = make_wtrack_geometry_dict(
        center_well_xy=np.array([parameters["center_x"], parameters["center_y"]], dtype=float),
        theta=parameters["theta"],
        long_segment_length=parameters["long_segment_length"],
        short_segment_length=parameters["short_segment_length"],
        dx=parameters["dx"],
        dy=parameters["dy"],
    )
    fitted_geometry = fit_result["geometry"]
    assert fit_result["fit_ok"] is True
    assert abs(
        compute_total_wtrack_length(fitted_geometry) - compute_total_wtrack_length(true_geometry)
    ) < 8.0
    assert _node_rmse(true_geometry, fitted_geometry) < 4.0


def test_fit_wtrack_geometry_marks_incomplete_occupancy_as_not_ok() -> None:
    center_stem = np.column_stack(
        [
            np.zeros(80, dtype=float) + 20.0,
            np.linspace(40.0, 100.0, 80, dtype=float),
        ]
    )

    fit_result = fit_wtrack_geometry_from_positions(
        center_stem,
        occupancy_bin_size_cm=2.0,
        skeleton_step_cm=2.0,
    )

    assert fit_result["fit_ok"] is False
    assert fit_result["fit_metrics"]["skeleton_within_threshold_fraction"] < 0.60


def test_fit_wtrack_geometry_writes_outputs_for_one_session(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")

    analysis_path = tmp_path / "L14" / "20240611"
    _write_timestamps(analysis_path, ["02_r1"])
    synthetic_position, _parameters = _make_synthetic_position()
    _write_position_pickle(
        analysis_path / "position.pkl",
        {"02_r1": synthetic_position},
    )

    outputs = fit_wtrack_geometry(
        animal_name="L14",
        date="20240611",
        data_root=tmp_path,
        output_dirname="wtrack_geometry_fit_test",
    )

    draft_path = outputs["draft_path"]
    snippet_path = outputs["snippet_path"]
    qc_path = outputs["qc_path"]
    assert Path(draft_path).exists()
    assert Path(snippet_path).exists()
    assert Path(qc_path).exists()

    with open(draft_path, "r", encoding="utf-8") as file:
        draft = json.load(file)
    assert draft["animal_name"] == "L14"
    assert draft["selected_epochs"] == ["02_r1"]
    assert "geometry" in draft
