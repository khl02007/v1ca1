from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from v1ca1.helper.session import load_clean_dlc_position_data
from v1ca1.position.convert_legacy_position_pickles import (
    DEFAULT_BODY_POSITION_NAME,
    DEFAULT_POSITION_NAME,
    OUTPUT_COLUMNS,
    convert_legacy_position_pickles,
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


def _write_epoch_pickle_dir(
    directory_path: Path,
    values_by_epoch: dict[str, np.ndarray],
    *,
    prefix: str = "20240101_RatA",
) -> None:
    directory_path.mkdir(parents=True, exist_ok=True)
    for epoch, values in values_by_epoch.items():
        _write_pickle(directory_path / f"{prefix}_{epoch}.pkl", values)


def _convert_session_pickles(
    analysis_path: Path,
    tmp_path: Path,
    *,
    overwrite: bool = False,
):
    return convert_legacy_position_pickles(
        animal_name="RatA",
        date="20240101",
        head_position_path=analysis_path / DEFAULT_POSITION_NAME,
        body_position_path=analysis_path / DEFAULT_BODY_POSITION_NAME,
        data_root=tmp_path,
        overwrite=overwrite,
    )


def test_convert_legacy_position_pickles_writes_canonical_parquet_and_round_trips(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_position = {
        "01_s1": np.array([0.0, 1.0], dtype=float),
        "02_r1": np.array([10.0, 11.0, 12.0], dtype=float),
        "03_s2": np.array([20.0], dtype=float),
    }
    _write_session_files(analysis_path, timestamps_position=timestamps_position)
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {
            "01_s1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "02_r1": np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=float),
            "03_s2": np.array([[11.0, 12.0]], dtype=float),
        },
    )
    _write_pickle(
        analysis_path / DEFAULT_BODY_POSITION_NAME,
        {
            "02_r1": np.array([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]], dtype=float),
        },
    )

    output_path = _convert_session_pickles(analysis_path, tmp_path)

    combined = pd.read_parquet(output_path)
    assert combined.columns.tolist() == list(OUTPUT_COLUMNS)
    assert combined["epoch"].tolist() == ["01_s1", "01_s1", "02_r1", "02_r1", "02_r1", "03_s2"]
    assert combined["frame"].tolist() == [0, 1, 0, 1, 2, 0]
    assert np.allclose(
        combined["frame_time_s"].to_numpy(dtype=float),
        np.array([0.0, 1.0, 10.0, 11.0, 12.0, 20.0], dtype=float),
    )
    assert np.isnan(
        combined.loc[combined["epoch"] == "01_s1", ["body_x_cm", "body_y_cm"]].to_numpy(dtype=float)
    ).all()
    assert np.isnan(
        combined.loc[combined["epoch"] == "03_s2", ["body_x_cm", "body_y_cm"]].to_numpy(dtype=float)
    ).all()
    assert np.allclose(
        combined.loc[combined["epoch"] == "02_r1", ["body_x_cm", "body_y_cm"]].to_numpy(dtype=float),
        np.array([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]], dtype=float),
    )

    epoch_order, head_position, body_position = load_clean_dlc_position_data(analysis_path)
    assert epoch_order == ["01_s1", "02_r1", "03_s2"]
    assert np.allclose(head_position["02_r1"], np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=float))
    assert np.isnan(body_position["01_s1"]).all()
    assert np.allclose(body_position["02_r1"], np.array([[15.0, 16.0], [17.0, 18.0], [19.0, 20.0]], dtype=float))


def test_convert_legacy_position_pickles_writes_only_epochs_present_in_position_pickle(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_position = {
        "01_s1": np.array([0.0], dtype=float),
        "02_r1": np.array([1.0, 2.0], dtype=float),
        "03_s2": np.array([3.0], dtype=float),
    }
    _write_session_files(analysis_path, timestamps_position=timestamps_position)
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {
            "02_r1": np.array([[1.0, 10.0], [2.0, 20.0]], dtype=float),
        },
    )
    _write_pickle(
        analysis_path / DEFAULT_BODY_POSITION_NAME,
        {
            "02_r1": np.array([[3.0, 30.0], [4.0, 40.0]], dtype=float),
        },
    )

    output_path = _convert_session_pickles(analysis_path, tmp_path)

    combined = pd.read_parquet(output_path)
    assert combined["epoch"].tolist() == ["02_r1", "02_r1"]
    assert combined["frame"].tolist() == [0, 1]
    assert np.allclose(combined["frame_time_s"].to_numpy(dtype=float), timestamps_position["02_r1"])


def test_convert_legacy_position_pickles_accepts_per_epoch_directories_and_mixed_inputs(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_position = {
        "01_s1": np.array([0.0, 1.0], dtype=float),
        "02_r1": np.array([10.0], dtype=float),
        "03_s2": np.array([20.0, 21.0], dtype=float),
    }
    _write_session_files(analysis_path, timestamps_position=timestamps_position)
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {
            "01_s1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "03_s2": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
        },
    )
    _write_epoch_pickle_dir(
        analysis_path / "body_position",
        {
            "01_s1": np.array([[11.0, 12.0], [13.0, 14.0]], dtype=float),
            "03_s2": np.array([[15.0, 16.0], [17.0, 18.0]], dtype=float),
        },
    )

    output_path = convert_legacy_position_pickles(
        animal_name="RatA",
        date="20240101",
        head_position_path=analysis_path / DEFAULT_POSITION_NAME,
        body_position_path=analysis_path / "body_position",
        data_root=tmp_path,
    )

    combined = pd.read_parquet(output_path)
    assert combined["epoch"].tolist() == ["01_s1", "01_s1", "03_s2", "03_s2"]
    assert np.allclose(
        combined.loc[:, ["body_x_cm", "body_y_cm"]].to_numpy(dtype=float),
        np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]], dtype=float),
    )


def test_convert_legacy_position_pickles_accepts_directory_inputs_with_timestamp_epoch_subset(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_position = {
        "01_s1": np.array([0.0], dtype=float),
        "02_r1": np.array([1.0, 2.0], dtype=float),
        "03_s2": np.array([3.0], dtype=float),
    }
    _write_session_files(analysis_path, timestamps_position=timestamps_position)
    _write_epoch_pickle_dir(
        analysis_path / "position",
        {
            "02_r1": np.array([[1.0, 10.0], [2.0, 20.0]], dtype=float),
        },
    )
    _write_epoch_pickle_dir(
        analysis_path / "body_position",
        {
            "02_r1": np.array([[3.0, 30.0], [4.0, 40.0]], dtype=float),
        },
    )

    output_path = convert_legacy_position_pickles(
        animal_name="RatA",
        date="20240101",
        head_position_path=analysis_path / "position",
        body_position_path=analysis_path / "body_position",
        data_root=tmp_path,
    )

    combined = pd.read_parquet(output_path)
    assert combined["epoch"].tolist() == ["02_r1", "02_r1"]
    assert np.allclose(combined["frame_time_s"].to_numpy(dtype=float), timestamps_position["02_r1"])


def test_convert_legacy_position_pickles_rejects_ambiguous_directory_epoch_mapping(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandas")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"01_s1": np.array([0.0], dtype=float), "1": np.array([1.0], dtype=float)},
    )
    _write_epoch_pickle_dir(
        analysis_path / "position",
        {
            "01_s1": np.array([[1.0, 2.0]], dtype=float),
        },
    )
    _write_epoch_pickle_dir(
        analysis_path / "body_position",
        {
            "01_s1": np.array([[3.0, 4.0]], dtype=float),
        },
    )

    with pytest.raises(
        ValueError,
        match="Could not map per-epoch position pickle to exactly one saved position epoch",
    ):
        convert_legacy_position_pickles(
            animal_name="RatA",
            date="20240101",
            head_position_path=analysis_path / "position",
            body_position_path=analysis_path / "body_position",
            data_root=tmp_path,
        )


@pytest.mark.parametrize("input_dirname", ["position", "body_position"])
def test_convert_legacy_position_pickles_rejects_duplicate_directory_epoch_mappings(
    tmp_path: Path,
    input_dirname: str,
) -> None:
    pytest.importorskip("pandas")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"02_r1": np.array([0.0], dtype=float)},
    )
    _write_epoch_pickle_dir(
        analysis_path / "position",
        {
            "02_r1": np.array([[1.0, 2.0]], dtype=float),
        },
    )
    _write_epoch_pickle_dir(
        analysis_path / "body_position",
        {
            "02_r1": np.array([[3.0, 4.0]], dtype=float),
        },
    )
    duplicate_dir = analysis_path / input_dirname
    _write_pickle(duplicate_dir / "duplicate_02_r1.pkl", np.array([[5.0, 6.0]], dtype=float))

    with pytest.raises(
        ValueError,
        match="Found duplicate per-epoch position pickle mappings for one epoch",
    ):
        convert_legacy_position_pickles(
            animal_name="RatA",
            date="20240101",
            head_position_path=analysis_path / "position",
            body_position_path=analysis_path / "body_position",
            data_root=tmp_path,
        )


@pytest.mark.parametrize("input_name", [DEFAULT_POSITION_NAME, DEFAULT_BODY_POSITION_NAME])
def test_convert_legacy_position_pickles_rejects_unknown_epochs(
    tmp_path: Path,
    input_name: str,
) -> None:
    pytest.importorskip("pandas")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_BODY_POSITION_NAME,
        {"02_r1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)},
    )
    _write_pickle(
        analysis_path / input_name,
        {
            "02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            "99_x": np.array([[9.0, 10.0]], dtype=float),
        },
    )

    with pytest.raises(
        ValueError,
        match="contains epochs not found in saved position timestamps",
    ):
        _convert_session_pickles(analysis_path, tmp_path)


@pytest.mark.parametrize("input_name", [DEFAULT_POSITION_NAME, DEFAULT_BODY_POSITION_NAME])
def test_convert_legacy_position_pickles_rejects_length_mismatch(
    tmp_path: Path,
    input_name: str,
) -> None:
    pytest.importorskip("pandas")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0, 2.0], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_BODY_POSITION_NAME,
        {"02_r1": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=float)},
    )
    _write_pickle(
        analysis_path / input_name,
        {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
    )

    with pytest.raises(
        ValueError,
        match="row count does not match saved position timestamps",
    ):
        _convert_session_pickles(analysis_path, tmp_path)


@pytest.mark.parametrize("missing_name", [DEFAULT_POSITION_NAME, DEFAULT_BODY_POSITION_NAME])
def test_convert_legacy_position_pickles_requires_explicit_pickle_paths(
    tmp_path: Path,
    missing_name: str,
) -> None:
    pytest.importorskip("pandas")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    if missing_name != DEFAULT_POSITION_NAME:
        _write_pickle(
            analysis_path / DEFAULT_POSITION_NAME,
            {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
        )
    if missing_name != DEFAULT_BODY_POSITION_NAME:
        _write_pickle(
            analysis_path / DEFAULT_BODY_POSITION_NAME,
            {"02_r1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)},
        )

    with pytest.raises(FileNotFoundError, match="Position file not found"):
        _convert_session_pickles(analysis_path, tmp_path)


def test_convert_legacy_position_pickles_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pandas")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
    )
    output_path = analysis_path / "dlc_position_cleaned" / "position.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("already here", encoding="utf-8")
    _write_pickle(
        analysis_path / DEFAULT_BODY_POSITION_NAME,
        {"02_r1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)},
    )

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        _convert_session_pickles(analysis_path, tmp_path)


def test_convert_legacy_position_pickles_overwrites_existing_output_when_requested(
    tmp_path: Path,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    _write_session_files(
        analysis_path,
        timestamps_position={"02_r1": np.array([0.0, 1.0], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_POSITION_NAME,
        {"02_r1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)},
    )
    _write_pickle(
        analysis_path / DEFAULT_BODY_POSITION_NAME,
        {"02_r1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)},
    )
    output_path = analysis_path / "dlc_position_cleaned" / "position.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": ["junk"],
            "frame": [0],
            "frame_time_s": [0.0],
            "head_x_cm": [0.0],
            "head_y_cm": [0.0],
            "body_x_cm": [0.0],
            "body_y_cm": [0.0],
        }
    ).to_parquet(output_path, index=False)

    saved_path = _convert_session_pickles(analysis_path, tmp_path, overwrite=True)

    combined = pd.read_parquet(saved_path)
    assert combined["epoch"].tolist() == ["02_r1", "02_r1"]
    assert np.allclose(
        combined.loc[:, ["head_x_cm", "head_y_cm"]].to_numpy(dtype=float),
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
    )
    assert np.allclose(
        combined.loc[:, ["body_x_cm", "body_y_cm"]].to_numpy(dtype=float),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
    )
