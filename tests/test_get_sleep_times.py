from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import v1ca1.sleep.get_sleep_times as sleep_times_module
from v1ca1.helper import session


class _FakeRecording:
    def __init__(self, channel_ids: list[int], sampling_frequency: float = 300.0) -> None:
        self._channel_ids = list(channel_ids)
        self._sampling_frequency = float(sampling_frequency)

    def get_channel_ids(self) -> list[int]:
        return list(self._channel_ids)

    def get_sampling_frequency(self) -> float:
        return float(self._sampling_frequency)


def test_parse_arguments_uses_shared_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "get_sleep_times.py",
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
        ],
    )

    args = sleep_times_module.parse_arguments()

    assert args.data_root == session.DEFAULT_DATA_ROOT
    assert args.nwb_root == session.DEFAULT_NWB_ROOT
    assert args.position_offset == session.DEFAULT_POSITION_OFFSET
    assert args.epochs is None


def test_get_sleep_times_for_session_rejects_missing_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    monkeypatch.setattr(
        sleep_times_module,
        "load_sleep_session_inputs",
        lambda path: {
            "epoch_tags": ["01_s1"],
            "timestamps_ephys": {"01_s1": np.array([0.0, 0.5], dtype=float)},
            "timestamps_ephys_all": np.array([0.0, 0.5], dtype=float),
            "timestamps_position": {"01_s1": np.array([0.0, 0.5], dtype=float)},
            "position_by_epoch": {"01_s1": np.zeros((2, 2), dtype=float)},
            "sources": {},
        },
    )

    with pytest.raises(ValueError, match="Requested epochs were not found"):
        sleep_times_module.get_sleep_times_for_session(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
            epochs=["02_r1"],
        )


def test_load_sleep_session_inputs_prefers_combined_clean_dlc_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v1ca1.sleep import _session as sleep_session_module

    monkeypatch.setattr(
        sleep_session_module,
        "load_ephys_timestamps_by_epoch",
        lambda path: (["01_s1"], {"01_s1": np.array([0.0, 1.0], dtype=float)}, "pickle"),
    )
    monkeypatch.setattr(
        sleep_session_module,
        "load_ephys_timestamps_all",
        lambda path: (np.array([0.0, 1.0], dtype=float), "pickle"),
    )
    monkeypatch.setattr(
        sleep_session_module,
        "load_position_timestamps",
        lambda path: (["01_s1"], {"01_s1": np.array([0.0, 1.0], dtype=float)}, "pynapple"),
    )
    monkeypatch.setattr(
        sleep_session_module,
        "load_position_data_with_precedence",
        lambda analysis_path, **kwargs: (
            {"01_s1": np.zeros((2, 2), dtype=float)},
            str(analysis_path / "dlc_position_cleaned" / "position.parquet"),
        ),
    )

    session_inputs = sleep_session_module.load_sleep_session_inputs(tmp_path)

    assert session_inputs["sources"]["position"].endswith("dlc_position_cleaned/position.parquet")
    assert "01_s1" in session_inputs["position_by_epoch"]


def test_get_sleep_times_for_session_writes_intervalset_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    monkeypatch.setattr(
        sleep_times_module,
        "load_sleep_session_inputs",
        lambda path: {
            "epoch_tags": ["01_s1"],
            "timestamps_ephys": {"01_s1": np.array([0.0, 0.4], dtype=float)},
            "timestamps_ephys_all": np.linspace(0.0, 0.4, 5, dtype=float),
            "timestamps_position": {"01_s1": np.linspace(0.0, 0.4, 5, dtype=float)},
            "position_by_epoch": {"01_s1": np.zeros((5, 2), dtype=float)},
            "sources": {"timestamps_ephys": "pynapple"},
        },
    )
    monkeypatch.setattr(
        sleep_times_module,
        "load_recording",
        lambda path: _FakeRecording([12], sampling_frequency=300.0),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "get_epoch_trace",
        lambda recording, **kwargs: (
            np.array([0.0, 0.2, 0.4], dtype=float),
            np.array([1.0, 2.0, 3.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "get_speed_trace",
        lambda position, timestamps_position, position_offset: (
            np.zeros((5, 2), dtype=float),
            np.linspace(0.0, 0.4, 5, dtype=float),
            np.array([10.0, 1.0, 1.0, 1.0, 10.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "butter_filter_and_decimate",
        lambda timestamps, data, **kwargs: (
            np.array([0.0, 0.2, 0.4], dtype=float),
            np.array([0.0, 1.0, 1.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "compute_spectrogram_principal_component",
        lambda signal, sampling_frequency, **kwargs: (
            np.array([0.0, 0.2, 0.4], dtype=float),
            np.array([0.0, 1.0, 1.0], dtype=float),
        ),
    )
    log_path = analysis_path / "v1ca1_log" / "sleep_log.json"
    monkeypatch.setattr(sleep_times_module, "write_run_log", lambda **kwargs: log_path)

    outputs = sleep_times_module.get_sleep_times_for_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        nwb_root=tmp_path,
        position_offset=0,
    )

    output_path = outputs["sleep_times_path"]
    assert output_path.exists()
    table = pd.read_parquet(output_path)
    assert list(table.columns) == ["start", "end", "epoch"]
    assert table.shape == (1, 3)
    assert table["epoch"].tolist() == ["01_s1"]
    assert output_path == analysis_path / "sleep_times.parquet"
    assert outputs["selected_epochs"] == ["01_s1"]
    assert outputs["log_path"] == log_path


def test_get_sleep_times_for_session_skips_epochs_missing_position_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    monkeypatch.setattr(
        sleep_times_module,
        "load_sleep_session_inputs",
        lambda path: {
            "epoch_tags": ["01_s1", "02_r1"],
            "timestamps_ephys": {
                "01_s1": np.array([0.0, 0.4], dtype=float),
                "02_r1": np.array([1.0, 1.4], dtype=float),
            },
            "timestamps_ephys_all": np.linspace(0.0, 1.4, 10, dtype=float),
            "timestamps_position": {"02_r1": np.linspace(1.0, 1.4, 5, dtype=float)},
            "position_by_epoch": {"02_r1": np.zeros((5, 2), dtype=float)},
            "sources": {"timestamps_ephys": "pynapple"},
        },
    )
    monkeypatch.setattr(
        sleep_times_module,
        "load_recording",
        lambda path: _FakeRecording([12], sampling_frequency=300.0),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "get_epoch_trace",
        lambda recording, **kwargs: (
            np.array([1.0, 1.2, 1.4], dtype=float),
            np.array([1.0, 2.0, 3.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "get_speed_trace",
        lambda position, timestamps_position, position_offset: (
            np.zeros((5, 2), dtype=float),
            np.linspace(1.0, 1.4, 5, dtype=float),
            np.array([10.0, 1.0, 1.0, 1.0, 10.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "butter_filter_and_decimate",
        lambda timestamps, data, **kwargs: (
            np.array([1.0, 1.2, 1.4], dtype=float),
            np.array([0.0, 1.0, 1.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_times_module,
        "compute_spectrogram_principal_component",
        lambda signal, sampling_frequency, **kwargs: (
            np.array([0.0, 0.2, 0.4], dtype=float),
            np.array([0.0, 1.0, 1.0], dtype=float),
        ),
    )
    log_path = analysis_path / "v1ca1_log" / "sleep_log.json"
    monkeypatch.setattr(sleep_times_module, "write_run_log", lambda **kwargs: log_path)

    outputs = sleep_times_module.get_sleep_times_for_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        nwb_root=tmp_path,
        position_offset=0,
    )

    captured = capsys.readouterr()
    assert "Skipping epochs missing position timestamps or position samples: ['01_s1']" in captured.out
    assert outputs["selected_epochs"] == ["02_r1"]
    assert outputs["skipped_epochs_missing_position"] == ["01_s1"]

    table = pd.read_parquet(outputs["sleep_times_path"])
    assert table["epoch"].tolist() == ["02_r1"]
