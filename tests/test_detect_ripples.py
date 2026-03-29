from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

import v1ca1.ripple.detect_ripples as detect_ripples_module
from v1ca1.helper.session import (
    DEFAULT_CLEAN_DLC_POSITION_DIRNAME,
    DEFAULT_CLEAN_DLC_POSITION_NAME,
)


def test_parse_arguments_disables_notch_filter_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["detect_ripples.py", "--animal-name", "RatA", "--date", "20240101"],
    )

    args = detect_ripples_module.parse_arguments()

    assert args.enable_notch_filter is False


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        ("--enable-notch-filter", True),
        ("--disable-notch-filter", False),
    ],
)
def test_parse_arguments_accepts_explicit_notch_filter_flags(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["detect_ripples.py", "--animal-name", "RatA", "--date", "20240101", flag],
    )

    args = detect_ripples_module.parse_arguments()

    assert args.enable_notch_filter is expected


def _write_ephys_npz(
    analysis_path: Path,
    *,
    epoch_tags: list[str],
    epoch_segments: list[np.ndarray],
) -> np.ndarray:
    nap = pytest.importorskip("pynapple")

    analysis_path.mkdir(parents=True, exist_ok=True)
    timestamps_ephys_all = np.concatenate(epoch_segments)
    nap.Ts(t=timestamps_ephys_all, time_units="s").save(analysis_path / "timestamps_ephys_all.npz")

    epoch_intervals = nap.IntervalSet(
        start=np.asarray([segment[0] for segment in epoch_segments], dtype=float),
        end=np.asarray([segment[-1] for segment in epoch_segments], dtype=float),
        time_units="s",
    )
    epoch_intervals.set_info(epoch=epoch_tags)
    epoch_intervals.save(analysis_path / "timestamps_ephys.npz")
    return timestamps_ephys_all


def _write_position_npz(
    analysis_path: Path,
    *,
    epoch_tags: list[str],
    timestamps_position: dict[str, np.ndarray],
) -> None:
    nap = pytest.importorskip("pynapple")

    position_group = nap.TsGroup(
        {
            index: nap.Ts(t=np.asarray(timestamps_position[epoch], dtype=float), time_units="s")
            for index, epoch in enumerate(epoch_tags)
        },
        time_units="s",
    )
    position_group.set_info(epoch=epoch_tags)
    position_group.save(analysis_path / "timestamps_position.npz")


def _write_combined_cleaned_position(
    analysis_path: Path,
    *,
    rows: list[dict[str, object]],
) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    output_path = (
        analysis_path
        / DEFAULT_CLEAN_DLC_POSITION_DIRNAME
        / DEFAULT_CLEAN_DLC_POSITION_NAME
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def _stub_compute_or_load_ripple_lfp_cache(
    *,
    selected_epochs: list[str],
    channel_ids: list[int],
    **_: object,
) -> tuple[dict[str, object], str]:
    return (
        {
            "time": {
                epoch: np.array([0.0, 0.01, 0.02], dtype=float)
                for epoch in selected_epochs
            },
            "data": {
                epoch: np.zeros((3, len(channel_ids)), dtype=float)
                for epoch in selected_epochs
            },
            "fs": {epoch: 1000.0 for epoch in selected_epochs},
            "epoch_cache_actions": {
                epoch: {"action": "compute", "reason": "stubbed cache for unit test"}
                for epoch in selected_epochs
            },
        },
        "computed",
    )


def _load_epoch_lfp_dataset(path: Path):
    xr = pytest.importorskip("xarray")
    return xr.load_dataset(path, engine="scipy")


def _write_epoch_lfp_cache_file(
    path: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    timestamps: np.ndarray,
    filtered_lfp: np.ndarray,
    sampling_frequency: float,
    channel_ids: list[int],
    enable_notch_filter: bool,
) -> Path:
    dataset = detect_ripples_module.build_epoch_lfp_dataset(
        animal_name=animal_name,
        date=date,
        epoch=epoch,
        timestamps=timestamps,
        filtered_lfp=filtered_lfp,
        sampling_frequency=sampling_frequency,
        channel_ids=channel_ids,
        enable_notch_filter=enable_notch_filter,
    )
    return detect_ripples_module.save_epoch_lfp_dataset(path, dataset)


def test_compute_or_load_ripple_lfp_cache_writes_epoch_netcdf_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("xarray")
    cache_dir = tmp_path / "ripple" / detect_ripples_module.LFP_CACHE_DIRNAME
    call_epochs: list[str] = []

    def _fake_filter(
        recording: object,
        *,
        epoch: str,
        epoch_timestamps: np.ndarray,
        timestamps_ephys_all: np.ndarray,
        ripple_channels: list[int],
        enable_notch_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        call_epochs.append(epoch)
        if epoch == "01_r1":
            return (
                np.array([0.0, 0.01, 0.02], dtype=float),
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
                1000.0,
                enable_notch_filter,
            )
        return (
            np.array([10.0, 10.01], dtype=float),
            np.array([[7.0, 8.0], [9.0, 10.0]], dtype=float),
            1000.0,
            enable_notch_filter,
        )

    monkeypatch.setattr(detect_ripples_module, "filter_ripple_band_for_epoch", _fake_filter)

    cache, source = detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1", "02_r2"],
        timestamps_by_epoch={
            "01_r1": np.array([0.0, 0.001, 0.002], dtype=float),
            "02_r2": np.array([10.0, 10.001], dtype=float),
        },
        timestamps_ephys_all=np.array([0.0, 0.001, 0.002, 10.0, 10.001], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=False,
    )

    assert source == "netcdf"
    assert call_epochs == ["01_r1", "02_r2"]
    assert cache["epoch_cache_actions"]["01_r1"]["action"] == "compute"
    assert cache["epoch_cache_actions"]["01_r1"]["reason"] == "no existing cache was found"
    assert np.allclose(cache["time"]["01_r1"], np.array([0.0, 0.01, 0.02], dtype=float))
    assert np.allclose(
        cache["data"]["02_r2"],
        np.array([[7.0, 8.0], [9.0, 10.0]], dtype=float),
    )
    assert float(cache["fs"]["01_r1"]) == 1000.0
    captured = capsys.readouterr()
    assert "Ripple LFP cache compute for 01_r1" in captured.out

    epoch_path = detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1")
    assert epoch_path.exists()
    dataset = _load_epoch_lfp_dataset(epoch_path)
    assert "filtered_lfp" in dataset.data_vars
    assert "sampling_frequency_hz" in dataset.data_vars
    assert "time" in dataset.coords
    assert "channel" in dataset.coords
    assert np.allclose(dataset["time"].values, np.array([0.0, 0.01, 0.02], dtype=float))
    assert np.allclose(dataset["channel"].values, np.array([1, 2], dtype=int))
    assert dataset.attrs["epoch"] == "01_r1"
    assert dataset.attrs["animal_name"] == "RatA"
    assert dataset.attrs["date"] == "20240101"
    assert dataset.attrs["cache_format"] == detect_ripples_module.LFP_CACHE_FORMAT


def test_compute_or_load_ripple_lfp_cache_reuses_valid_epoch_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("xarray")
    cache_dir = tmp_path / "ripple" / detect_ripples_module.LFP_CACHE_DIRNAME
    epoch_path = detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1")
    _write_epoch_lfp_cache_file(
        epoch_path,
        animal_name="RatA",
        date="20240101",
        epoch="01_r1",
        timestamps=np.array([0.0, 0.01, 0.02], dtype=float),
        filtered_lfp=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
        sampling_frequency=1000.0,
        channel_ids=[1, 2],
        enable_notch_filter=True,
    )
    monkeypatch.setattr(
        detect_ripples_module,
        "filter_ripple_band_for_epoch",
        lambda *args, **kwargs: pytest.fail("valid NetCDF cache should be reused"),
    )

    cache, source = detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1"],
        timestamps_by_epoch={"01_r1": np.array([0.0, 0.001, 0.002], dtype=float)},
        timestamps_ephys_all=np.array([0.0, 0.001, 0.002], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=False,
    )

    assert source == "netcdf"
    assert np.allclose(cache["time"]["01_r1"], np.array([0.0, 0.01, 0.02], dtype=float))
    assert cache["epoch_cache_actions"]["01_r1"]["action"] == "reuse"
    captured = capsys.readouterr()
    assert "Ripple LFP cache reuse for 01_r1" in captured.out


def test_compute_or_load_ripple_lfp_cache_recomputes_only_missing_epoch_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    cache_dir = tmp_path / "ripple" / detect_ripples_module.LFP_CACHE_DIRNAME
    _write_epoch_lfp_cache_file(
        detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1"),
        animal_name="RatA",
        date="20240101",
        epoch="01_r1",
        timestamps=np.array([0.0, 0.01], dtype=float),
        filtered_lfp=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        sampling_frequency=1000.0,
        channel_ids=[1, 2],
        enable_notch_filter=True,
    )
    call_epochs: list[str] = []

    def _fake_filter(
        recording: object,
        *,
        epoch: str,
        epoch_timestamps: np.ndarray,
        timestamps_ephys_all: np.ndarray,
        ripple_channels: list[int],
        enable_notch_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        call_epochs.append(epoch)
        return (
            np.array([10.0, 10.01], dtype=float),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            1000.0,
            enable_notch_filter,
        )

    monkeypatch.setattr(detect_ripples_module, "filter_ripple_band_for_epoch", _fake_filter)

    cache, _source = detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1", "02_r2"],
        timestamps_by_epoch={
            "01_r1": np.array([0.0, 0.001], dtype=float),
            "02_r2": np.array([10.0, 10.001], dtype=float),
        },
        timestamps_ephys_all=np.array([0.0, 0.001, 10.0, 10.001], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=False,
    )

    assert call_epochs == ["02_r2"]
    assert set(cache["time"]) == {"01_r1", "02_r2"}
    assert detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "02_r2").exists()


def test_compute_or_load_ripple_lfp_cache_recomputes_on_channel_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("xarray")
    cache_dir = tmp_path / "ripple" / detect_ripples_module.LFP_CACHE_DIRNAME
    epoch_path = detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1")
    _write_epoch_lfp_cache_file(
        epoch_path,
        animal_name="RatA",
        date="20240101",
        epoch="01_r1",
        timestamps=np.array([0.0, 0.01], dtype=float),
        filtered_lfp=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        sampling_frequency=1000.0,
        channel_ids=[3, 4],
        enable_notch_filter=True,
    )
    call_epochs: list[str] = []

    def _fake_filter(
        recording: object,
        *,
        epoch: str,
        epoch_timestamps: np.ndarray,
        timestamps_ephys_all: np.ndarray,
        ripple_channels: list[int],
        enable_notch_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        call_epochs.append(epoch)
        return (
            np.array([0.0, 0.01], dtype=float),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            1000.0,
            enable_notch_filter,
        )

    monkeypatch.setattr(detect_ripples_module, "filter_ripple_band_for_epoch", _fake_filter)

    cache, _source = detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1"],
        timestamps_by_epoch={"01_r1": np.array([0.0, 0.001], dtype=float)},
        timestamps_ephys_all=np.array([0.0, 0.001], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=False,
    )

    assert call_epochs == ["01_r1"]
    assert np.allclose(cache["data"]["01_r1"], np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float))
    assert cache["epoch_cache_actions"]["01_r1"]["action"] == "recompute"
    assert "does not match the current request" in cache["epoch_cache_actions"]["01_r1"]["reason"]
    captured = capsys.readouterr()
    assert "Ripple LFP cache recompute for 01_r1" in captured.out


def test_compute_or_load_ripple_lfp_cache_recomputes_on_bad_netcdf_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xr = pytest.importorskip("xarray")
    cache_dir = tmp_path / "ripple" / detect_ripples_module.LFP_CACHE_DIRNAME
    epoch_path = detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1")
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"wrong_var": (("sample",), np.array([1.0, 2.0], dtype=float))}).to_netcdf(
        epoch_path,
        engine="scipy",
    )
    call_epochs: list[str] = []

    def _fake_filter(
        recording: object,
        *,
        epoch: str,
        epoch_timestamps: np.ndarray,
        timestamps_ephys_all: np.ndarray,
        ripple_channels: list[int],
        enable_notch_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        call_epochs.append(epoch)
        return (
            np.array([0.0, 0.01], dtype=float),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float),
            1000.0,
            enable_notch_filter,
        )

    monkeypatch.setattr(detect_ripples_module, "filter_ripple_band_for_epoch", _fake_filter)

    detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1"],
        timestamps_by_epoch={"01_r1": np.array([0.0, 0.001], dtype=float)},
        timestamps_ephys_all=np.array([0.0, 0.001], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=False,
    )

    assert call_epochs == ["01_r1"]


def test_compute_or_load_ripple_lfp_cache_overwrite_recomputes_valid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("xarray")
    cache_dir = tmp_path / "ripple" / detect_ripples_module.LFP_CACHE_DIRNAME
    _write_epoch_lfp_cache_file(
        detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1"),
        animal_name="RatA",
        date="20240101",
        epoch="01_r1",
        timestamps=np.array([0.0, 0.01], dtype=float),
        filtered_lfp=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        sampling_frequency=1000.0,
        channel_ids=[1, 2],
        enable_notch_filter=True,
    )
    call_epochs: list[str] = []

    def _fake_filter(
        recording: object,
        *,
        epoch: str,
        epoch_timestamps: np.ndarray,
        timestamps_ephys_all: np.ndarray,
        ripple_channels: list[int],
        enable_notch_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        call_epochs.append(epoch)
        return (
            np.array([0.0, 0.01], dtype=float),
            np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float),
            1000.0,
            enable_notch_filter,
        )

    monkeypatch.setattr(detect_ripples_module, "filter_ripple_band_for_epoch", _fake_filter)

    cache, _source = detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1"],
        timestamps_by_epoch={"01_r1": np.array([0.0, 0.001], dtype=float)},
        timestamps_ephys_all=np.array([0.0, 0.001], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=True,
    )

    assert call_epochs == ["01_r1"]
    assert np.allclose(cache["data"]["01_r1"], np.array([[9.0, 10.0], [11.0, 12.0]], dtype=float))
    assert cache["epoch_cache_actions"]["01_r1"]["action"] == "recompute"
    assert "overwrite requested" in cache["epoch_cache_actions"]["01_r1"]["reason"]
    captured = capsys.readouterr()
    assert "overwrite requested" in captured.out


def test_compute_or_load_ripple_lfp_cache_ignores_legacy_pickle_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    ripple_dir = tmp_path / "ripple"
    cache_dir = ripple_dir / detect_ripples_module.LFP_CACHE_DIRNAME
    ripple_dir.mkdir(parents=True, exist_ok=True)
    (ripple_dir / "ripple_channels_lfp.pkl").write_bytes(b"not-a-pickle")
    call_epochs: list[str] = []

    def _fake_filter(
        recording: object,
        *,
        epoch: str,
        epoch_timestamps: np.ndarray,
        timestamps_ephys_all: np.ndarray,
        ripple_channels: list[int],
        enable_notch_filter: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        call_epochs.append(epoch)
        return (
            np.array([0.0, 0.01], dtype=float),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
            1000.0,
            enable_notch_filter,
        )

    monkeypatch.setattr(detect_ripples_module, "filter_ripple_band_for_epoch", _fake_filter)

    detect_ripples_module.compute_or_load_ripple_lfp_cache(
        cache_dir=cache_dir,
        animal_name="RatA",
        date="20240101",
        recording=object(),
        selected_epochs=["01_r1"],
        timestamps_by_epoch={"01_r1": np.array([0.0, 0.001], dtype=float)},
        timestamps_ephys_all=np.array([0.0, 0.001], dtype=float),
        channel_ids=[1, 2],
        enable_notch_filter=True,
        overwrite=False,
    )

    assert call_epochs == ["01_r1"]
    assert detect_ripples_module.get_epoch_lfp_cache_path(cache_dir, "01_r1").exists()


def test_get_ripple_times_uses_clean_dlc_head_position_for_speed_gating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    analysis_path = tmp_path / "RatA" / "20240101"

    _write_ephys_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        epoch_segments=[np.array([0.0, 0.001, 0.002], dtype=float)],
    )
    position_timestamps = {"01_r1": np.array([0.0, 0.1, 0.2], dtype=float)}
    _write_position_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        timestamps_position=position_timestamps,
    )
    position_path = _write_combined_cleaned_position(
        analysis_path,
        rows=[
            {
                "epoch": "01_r1",
                "frame": 0,
                "frame_time_s": 0.0,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": 100.0,
                "body_y_cm": 200.0,
            },
            {
                "epoch": "01_r1",
                "frame": 1,
                "frame_time_s": 0.1,
                "head_x_cm": 3.0,
                "head_y_cm": 4.0,
                "body_x_cm": 300.0,
                "body_y_cm": 400.0,
            },
            {
                "epoch": "01_r1",
                "frame": 2,
                "frame_time_s": 0.2,
                "head_x_cm": 5.0,
                "head_y_cm": 6.0,
                "body_x_cm": 500.0,
                "body_y_cm": 600.0,
            },
        ],
    )

    observed: dict[str, np.ndarray] = {}

    def _fake_compute_speed(
        position: np.ndarray,
        timestamps_position: np.ndarray,
        position_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed["position"] = np.asarray(position, dtype=float)
        observed["timestamps_position"] = np.asarray(timestamps_position, dtype=float)
        observed["position_offset"] = np.asarray([position_offset], dtype=int)
        return np.asarray(timestamps_position, dtype=float), np.zeros_like(
            np.asarray(timestamps_position, dtype=float)
        )

    monkeypatch.setattr(
        detect_ripples_module,
        "get_ripple_channels_for_session",
        lambda animal_name, date: [1, 2],
    )
    monkeypatch.setattr(detect_ripples_module, "get_recording", lambda **_: object())
    monkeypatch.setattr(
        detect_ripples_module,
        "compute_or_load_ripple_lfp_cache",
        _stub_compute_or_load_ripple_lfp_cache,
    )
    monkeypatch.setattr(detect_ripples_module, "compute_speed", _fake_compute_speed)
    monkeypatch.setattr(
        detect_ripples_module,
        "run_kay_ripple_detector",
        lambda **_: pd.DataFrame({"start_time": [0.0], "end_time": [0.01]}),
    )

    result = detect_ripples_module.get_ripple_times(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        nwb_root=tmp_path,
    )

    assert np.allclose(
        observed["position"],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
    )
    assert np.allclose(observed["timestamps_position"], position_timestamps["01_r1"])
    assert int(observed["position_offset"][0]) == detect_ripples_module.DEFAULT_POSITION_OFFSET
    assert result["sources"]["timestamps_ephys"] == "npz"
    assert result["sources"]["timestamps_ephys_all"] == "npz"
    assert result["sources"]["timestamps_position"] == "npz"
    assert result["sources"]["position"] == str(position_path)
    assert result["selected_epochs"] == ["01_r1"]
    assert result["use_speed_gating"] is True
    assert result["enable_notch_filter"] is False
    assert result["epoch_summaries"]["01_r1"]["ripple_count"] == 1
    interval_table = pd.read_parquet(result["interval_parquet_path"])
    assert list(interval_table.columns) == ["start", "end", "epoch"]
    assert np.allclose(interval_table["start"], [0.0])
    assert np.allclose(interval_table["end"], [0.01])
    assert interval_table["epoch"].tolist() == ["01_r1"]
    assert Path(result["interval_parquet_path"]).exists()
    assert not (analysis_path / "ripple" / "Kay_ripple_detector.pkl").exists()


def test_get_ripple_times_rejects_pickle_only_session(tmp_path: Path) -> None:
    pytest.importorskip("pynapple")
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True, exist_ok=True)
    with open(analysis_path / "timestamps_ephys.pkl", "wb") as file:
        pickle.dump({"01_r1": np.array([0.0, 0.001], dtype=float)}, file)
    with open(analysis_path / "timestamps_ephys_all.pkl", "wb") as file:
        pickle.dump(np.array([0.0, 0.001], dtype=float), file)

    with pytest.raises(FileNotFoundError, match="Required timestamps_ephys.npz not found"):
        detect_ripples_module.get_ripple_times(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
            nwb_root=tmp_path,
            disable_speed_gating=True,
        )


def test_get_ripple_times_requires_timestamps_ephys_all_npz(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_ephys_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        epoch_segments=[np.array([0.0, 0.001, 0.002], dtype=float)],
    )
    (analysis_path / "timestamps_ephys_all.npz").unlink()

    with pytest.raises(FileNotFoundError, match="Required timestamps_ephys_all.npz not found"):
        detect_ripples_module.get_ripple_times(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
            nwb_root=tmp_path,
            disable_speed_gating=True,
        )


def test_get_ripple_times_requires_combined_clean_dlc_parquet_for_speed_gating(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_ephys_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        epoch_segments=[np.array([0.0, 0.001, 0.002], dtype=float)],
    )
    _write_position_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        timestamps_position={"01_r1": np.array([0.0, 0.1, 0.2], dtype=float)},
    )

    with pytest.raises(FileNotFoundError, match="combine_clean_dlc_position"):
        detect_ripples_module.get_ripple_times(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
            nwb_root=tmp_path,
        )


def test_get_ripple_times_skips_position_inputs_when_speed_gating_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_ephys_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        epoch_segments=[np.array([0.0, 0.001, 0.002], dtype=float)],
    )

    monkeypatch.setattr(
        detect_ripples_module,
        "get_ripple_channels_for_session",
        lambda animal_name, date: [1, 2],
    )
    monkeypatch.setattr(detect_ripples_module, "get_recording", lambda **_: object())
    monkeypatch.setattr(
        detect_ripples_module,
        "compute_or_load_ripple_lfp_cache",
        _stub_compute_or_load_ripple_lfp_cache,
    )
    monkeypatch.setattr(
        detect_ripples_module,
        "compute_speed",
        lambda *args, **kwargs: pytest.fail("compute_speed should not run when speed gating is disabled"),
    )
    monkeypatch.setattr(
        detect_ripples_module,
        "run_kay_ripple_detector",
        lambda **_: pd.DataFrame({"start_time": [0.0], "end_time": [0.01]}),
    )

    result = detect_ripples_module.get_ripple_times(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        nwb_root=tmp_path,
        disable_speed_gating=True,
    )

    assert result["use_speed_gating"] is False
    assert result["sources"]["timestamps_position"] == "not_required_disable_speed_gating"
    assert result["sources"]["position"] == "not_required_disable_speed_gating"
    interval_table = pd.read_parquet(result["interval_parquet_path"])
    assert list(interval_table.columns) == ["start", "end", "epoch"]
    assert np.allclose(interval_table["start"], [0.0])
    assert np.allclose(interval_table["end"], [0.01])
    assert interval_table["epoch"].tolist() == ["01_r1"]
    assert Path(result["interval_parquet_path"]).exists()
    assert not (analysis_path / "ripple" / "Kay_ripple_detector_no_speed.pkl").exists()


def test_get_ripple_times_filters_to_complete_speed_gated_epochs_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_ephys_npz(
        analysis_path,
        epoch_tags=["01_r1", "02_r2", "03_r3"],
        epoch_segments=[
            np.array([0.0, 0.001, 0.002], dtype=float),
            np.array([10.0, 10.001, 10.002], dtype=float),
            np.array([20.0, 20.001, 20.002], dtype=float),
        ],
    )
    _write_position_npz(
        analysis_path,
        epoch_tags=["01_r1", "02_r2"],
        timestamps_position={
            "01_r1": np.array([0.0, 0.1, 0.2], dtype=float),
            "02_r2": np.array([10.0, 10.1, 10.2], dtype=float),
        },
    )
    _write_combined_cleaned_position(
        analysis_path,
        rows=[
            {
                "epoch": "01_r1",
                "frame": 0,
                "frame_time_s": 0.0,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": 10.0,
                "body_y_cm": 20.0,
            },
            {
                "epoch": "01_r1",
                "frame": 1,
                "frame_time_s": 0.1,
                "head_x_cm": 3.0,
                "head_y_cm": 4.0,
                "body_x_cm": 30.0,
                "body_y_cm": 40.0,
            },
            {
                "epoch": "01_r1",
                "frame": 2,
                "frame_time_s": 0.2,
                "head_x_cm": 5.0,
                "head_y_cm": 6.0,
                "body_x_cm": 50.0,
                "body_y_cm": 60.0,
            },
            {
                "epoch": "03_r3",
                "frame": 0,
                "frame_time_s": 20.0,
                "head_x_cm": 7.0,
                "head_y_cm": 8.0,
                "body_x_cm": 70.0,
                "body_y_cm": 80.0,
            },
            {
                "epoch": "03_r3",
                "frame": 1,
                "frame_time_s": 20.1,
                "head_x_cm": 9.0,
                "head_y_cm": 10.0,
                "body_x_cm": 90.0,
                "body_y_cm": 100.0,
            },
            {
                "epoch": "03_r3",
                "frame": 2,
                "frame_time_s": 20.2,
                "head_x_cm": 11.0,
                "head_y_cm": 12.0,
                "body_x_cm": 110.0,
                "body_y_cm": 120.0,
            },
        ],
    )

    monkeypatch.setattr(
        detect_ripples_module,
        "get_ripple_channels_for_session",
        lambda animal_name, date: [1, 2],
    )
    monkeypatch.setattr(detect_ripples_module, "get_recording", lambda **_: object())
    monkeypatch.setattr(
        detect_ripples_module,
        "compute_or_load_ripple_lfp_cache",
        _stub_compute_or_load_ripple_lfp_cache,
    )
    monkeypatch.setattr(
        detect_ripples_module,
        "compute_speed",
        lambda position, timestamps_position, position_offset: (
            np.asarray(timestamps_position, dtype=float),
            np.zeros(3, dtype=float),
        ),
    )
    monkeypatch.setattr(
        detect_ripples_module,
        "run_kay_ripple_detector",
        lambda **_: pd.DataFrame({"start_time": [0.0], "end_time": [0.01]}),
    )

    result = detect_ripples_module.get_ripple_times(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        nwb_root=tmp_path,
    )

    assert result["available_epochs"] == ["01_r1", "02_r2", "03_r3"]
    assert result["selected_epochs"] == ["01_r1"]


def test_get_ripple_times_rejects_selected_epochs_missing_from_position_npz(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_ephys_npz(
        analysis_path,
        epoch_tags=["01_r1", "02_r2"],
        epoch_segments=[
            np.array([0.0, 0.001, 0.002], dtype=float),
            np.array([10.0, 10.001, 10.002], dtype=float),
        ],
    )
    _write_position_npz(
        analysis_path,
        epoch_tags=["01_r1"],
        timestamps_position={"01_r1": np.array([0.0, 0.1, 0.2], dtype=float)},
    )
    _write_combined_cleaned_position(
        analysis_path,
        rows=[
            {
                "epoch": "01_r1",
                "frame": 0,
                "frame_time_s": 0.0,
                "head_x_cm": 1.0,
                "head_y_cm": 2.0,
                "body_x_cm": 10.0,
                "body_y_cm": 20.0,
            },
            {
                "epoch": "01_r1",
                "frame": 1,
                "frame_time_s": 0.1,
                "head_x_cm": 3.0,
                "head_y_cm": 4.0,
                "body_x_cm": 30.0,
                "body_y_cm": 40.0,
            },
            {
                "epoch": "01_r1",
                "frame": 2,
                "frame_time_s": 0.2,
                "head_x_cm": 5.0,
                "head_y_cm": 6.0,
                "body_x_cm": 50.0,
                "body_y_cm": 60.0,
            },
            {
                "epoch": "02_r2",
                "frame": 0,
                "frame_time_s": 10.0,
                "head_x_cm": 7.0,
                "head_y_cm": 8.0,
                "body_x_cm": 70.0,
                "body_y_cm": 80.0,
            },
            {
                "epoch": "02_r2",
                "frame": 1,
                "frame_time_s": 10.1,
                "head_x_cm": 9.0,
                "head_y_cm": 10.0,
                "body_x_cm": 90.0,
                "body_y_cm": 100.0,
            },
            {
                "epoch": "02_r2",
                "frame": 2,
                "frame_time_s": 10.2,
                "head_x_cm": 11.0,
                "head_y_cm": 12.0,
                "body_x_cm": 110.0,
                "body_y_cm": 120.0,
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"Missing epochs by source: position_timestamps: \['02_r2'\]",
    ):
        detect_ripples_module.get_ripple_times(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
            nwb_root=tmp_path,
            epochs=["01_r1", "02_r2"],
        )
