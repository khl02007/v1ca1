from __future__ import annotations

import pickle
import sys
from datetime import datetime, timezone
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
    save_before_after_position_figure,
)


def _write_pickle(path: Path, value: object) -> None:
    with open(path, "wb") as file:
        pickle.dump(value, file)


def _write_session_files(
    analysis_path: Path,
    timestamps_position: dict[str, np.ndarray],
) -> None:
    analysis_path.mkdir(parents=True, exist_ok=True)
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


def _write_test_position_nwb(
    nwb_path: Path,
    epoch_tags: list[str],
    series_specs: list[dict[str, object]] | None,
    *,
    include_behavior: bool = True,
    include_position: bool = True,
) -> None:
    pynwb = pytest.importorskip("pynwb")

    nwbfile = pynwb.NWBFile(
        session_description="test session",
        identifier="test-id",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    for index, epoch_tag in enumerate(epoch_tags):
        start_time = float(index * 10.0)
        nwbfile.add_epoch(
            start_time=start_time,
            stop_time=start_time + 5.0,
            tags=[str(epoch_tag)],
        )

    if include_behavior:
        behavior = nwbfile.create_processing_module("behavior", "test behavior module")
        if include_position:
            position = pynwb.behavior.Position(name="position")
            for index, spec in enumerate(series_specs or []):
                data = np.asarray(
                    spec.get("data", np.zeros((5, 2), dtype=float)),
                    dtype=float,
                )
                timestamps = np.asarray(
                    spec.get("timestamps", np.arange(data.shape[0], dtype=float)),
                    dtype=float,
                )
                position.add_spatial_series(
                    pynwb.behavior.SpatialSeries(
                        name=str(spec.get("name", f"series_{index}")),
                        data=data,
                        timestamps=timestamps,
                        reference_frame="origin",
                        unit=str(spec.get("unit", "meters")),
                        conversion=float(spec.get("conversion", 1.0)),
                    )
                )
            behavior.add(position)

    with pynwb.NWBHDF5IO(nwb_path, "w") as io:
        io.write(nwbfile)


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


def test_clean_dlc_position_writes_joint_epoch_parquet_and_figure_in_cm(
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
        meters_per_pixel=0.02,
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
        "head_x_raw_cm",
        "head_y_raw_cm",
        "head_likelihood",
        "head_likelihood_loss",
        "head_x_cleaned_cm",
        "head_y_cleaned_cm",
        "head_step_prev_cm",
        "head_step_next_cm",
        "head_low_likelihood",
        "head_jump_invalid",
        "head_pair_invalid",
        "head_invalid",
        "head_interpolated",
        "head_interpolation_source_left",
        "head_interpolation_source_right",
        "body_x_raw_cm",
        "body_y_raw_cm",
        "body_likelihood",
        "body_likelihood_loss",
        "body_x_cleaned_cm",
        "body_y_cleaned_cm",
        "body_step_prev_cm",
        "body_step_next_cm",
        "body_low_likelihood",
        "body_jump_invalid",
        "body_pair_invalid",
        "body_invalid",
        "body_interpolated",
        "body_interpolation_source_left",
        "body_interpolation_source_right",
        "head_body_distance_cm",
        "pair_distance_invalid",
    ]
    assert diagnostics["epoch"].unique().tolist() == ["02_r1"]
    assert diagnostics.loc[2, "head_x_cleaned_cm"] == pytest.approx(4.0)
    assert diagnostics.loc[2, "body_x_cleaned_cm"] == pytest.approx(74.0)
    assert diagnostics.loc[2, "head_body_distance_cm"] == pytest.approx(126.0)

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
    assert '"position_scale_source": "cli"' in log_text
    assert '"meters_per_pixel": 0.02' in log_text


def test_clean_dlc_position_cleans_full_epoch_without_internal_offset_and_scales_output(
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
        meters_per_pixel=0.02,
    )

    cleaned_position = pd.read_parquet(output_path)
    assert cleaned_position.loc[2, "head_x_cleaned_cm"] == pytest.approx(4.0)
    assert cleaned_position.loc[2, "body_x_cleaned_cm"] == pytest.approx(64.0)


def test_clean_dlc_position_resolves_scale_from_nwb_all_epoch_mapping(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    frame_times = np.arange(5, dtype=float)
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": frame_times},
    )
    _write_test_position_nwb(
        tmp_path / "RatA20240101.nwb",
        epoch_tags=["01_s1", "02_r1"],
        series_specs=[
            {"conversion": 0.01, "unit": "meters"},
            {"conversion": 0.02, "unit": "meters"},
        ],
    )

    dlc_h5_path = tmp_path / "all_epoch_mapping.h5"
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
        nwb_root=tmp_path,
        threshold_z=0.0,
        min_jump_threshold_px=10.0,
    )

    diagnostics = pd.read_parquet(output_path)
    assert diagnostics.loc[2, "head_x_cleaned_cm"] == pytest.approx(4.0)

    log_files = sorted(
        (analysis_path / "v1ca1_log").glob("v1ca1_position_clean_dlc_position_*.json")
    )
    log_text = log_files[0].read_text(encoding="utf-8")
    assert '"position_scale_source": "nwb"' in log_text
    assert '"position_scale_epoch_mapping": "all_epochs"' in log_text
    assert '"position_scale_series_name": "series_1"' in log_text


def test_clean_dlc_position_resolves_scale_from_nwb_run_epoch_mapping(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    frame_times = np.arange(5, dtype=float)
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"04_r2": frame_times},
    )
    _write_test_position_nwb(
        tmp_path / "RatA20240101.nwb",
        epoch_tags=["01_s1", "02_r1", "03_s2", "04_r2"],
        series_specs=[
            {"conversion": 0.02, "unit": "meters"},
            {"conversion": 0.03, "unit": "meters"},
        ],
    )

    dlc_h5_path = tmp_path / "run_epoch_mapping.h5"
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
        epoch="04_r2",
        dlc_h5_path=dlc_h5_path,
        data_root=tmp_path,
        nwb_root=tmp_path,
        threshold_z=0.0,
        min_jump_threshold_px=10.0,
    )

    diagnostics = pd.read_parquet(output_path)
    assert diagnostics.loc[2, "head_x_cleaned_cm"] == pytest.approx(6.0)


def test_clean_dlc_position_rejects_nonpositive_meters_per_pixel(tmp_path: Path) -> None:
    pytest.importorskip("tables")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    dlc_h5_path = tmp_path / "joint.h5"
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (np.array([0.0, 1.0]), np.zeros(2), np.full(2, 0.999)),
            "body": (np.array([30.0, 31.0]), np.zeros(2), np.full(2, 0.999)),
        },
    )

    with pytest.raises(ValueError, match="meters-per-pixel"):
        clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
            meters_per_pixel=0.0,
        )


@pytest.mark.parametrize(
    ("epoch_tags", "series_specs", "include_behavior", "include_position", "message"),
    [
        (
            ["01_s1", "02_r1"],
            None,
            False,
            False,
            "behavior processing module",
        ),
        (
            ["01_s1", "02_r1"],
            None,
            True,
            False,
            "position interface",
        ),
        (
            ["01_s1", "02_r1"],
            [],
            True,
            True,
            "spatial series",
        ),
        (
            ["01_s1", "02_r1"],
            [{"conversion": 0.02, "unit": "centimeters"}],
            True,
            True,
            "meter units",
        ),
        (
            ["01_s1", "02_r1"],
            [{"conversion": 0.0, "unit": "meters"}],
            True,
            True,
            "meters-per-pixel",
        ),
        (
            ["01_s1", "02_r1", "03_s2"],
            [
                {"conversion": 0.02, "unit": "meters"},
                {"conversion": 0.03, "unit": "meters"},
            ],
            True,
            True,
            "Could not map NWB position spatial series",
        ),
    ],
)
def test_clean_dlc_position_rejects_invalid_nwb_scale_sources(
    tmp_path: Path,
    epoch_tags: list[str],
    series_specs: list[dict[str, object]] | None,
    include_behavior: bool,
    include_position: bool,
    message: str,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("pynwb")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path=analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    _write_test_position_nwb(
        tmp_path / "RatA20240101.nwb",
        epoch_tags=epoch_tags,
        series_specs=series_specs,
        include_behavior=include_behavior,
        include_position=include_position,
    )

    dlc_h5_path = tmp_path / "joint_bad_scale.h5"
    _write_dlc_h5(
        dlc_h5_path,
        tracks={
            "head": (np.array([0.0, 1.0]), np.zeros(2), np.full(2, 0.999)),
            "body": (np.array([30.0, 31.0]), np.zeros(2), np.full(2, 0.999)),
        },
    )

    with pytest.raises(ValueError, match=message):
        clean_dlc_position(
            animal_name="RatA",
            date="20240101",
            epoch="02_r1",
            dlc_h5_path=dlc_h5_path,
            data_root=tmp_path,
            nwb_root=tmp_path,
        )


def test_save_before_after_position_figure_labels_axes_in_cm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    plt = pytest.importorskip("matplotlib.pyplot")

    labels: list[tuple[str | None, str | None]] = []

    class FakeAxis:
        def __init__(self) -> None:
            self.xlabel: str | None = None
            self.ylabel: str | None = None

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def set_title(self, _title: str) -> None:
            pass

        def set_xlabel(self, value: str) -> None:
            self.xlabel = value

        def set_ylabel(self, value: str) -> None:
            self.ylabel = value

        def set_aspect(self, *_args: object, **_kwargs: object) -> None:
            labels.append((self.xlabel, self.ylabel))

    class FakeFigure:
        def suptitle(self, _title: str) -> None:
            pass

        def savefig(self, path: Path, **_kwargs: object) -> None:
            path.write_bytes(b"fake")

    axes = np.array([[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]], dtype=object)
    monkeypatch.setattr(plt, "subplots", lambda *args, **kwargs: (FakeFigure(), axes))
    monkeypatch.setattr(plt, "close", lambda _figure: None)

    output_table = pd.DataFrame(
        {
            "head_x_raw_cm": [0.0, 1.0],
            "head_y_raw_cm": [0.0, 1.0],
            "head_x_cleaned_cm": [0.0, 1.0],
            "head_y_cleaned_cm": [0.0, 1.0],
            "body_x_raw_cm": [2.0, 3.0],
            "body_y_raw_cm": [2.0, 3.0],
            "body_x_cleaned_cm": [2.0, 3.0],
            "body_y_cleaned_cm": [2.0, 3.0],
        }
    )

    save_before_after_position_figure(
        output_table=output_table,
        output_path=tmp_path / "figure.png",
        epoch="02_r1",
    )

    assert labels == [("x (cm)", "y (cm)")] * 4


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
            meters_per_pixel=0.01,
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
            meters_per_pixel=0.01,
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
            meters_per_pixel=0.01,
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
            "--meters-per-pixel",
            "0.02",
        ],
    )

    args = parse_arguments()
    assert not hasattr(args, "position_offset")
    assert args.meters_per_pixel == pytest.approx(0.02)
