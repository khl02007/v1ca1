from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import v1ca1.oscillation.get_theta_phase as theta_module


class _FakeRecording:
    def __init__(
        self,
        traces: np.ndarray,
        channel_ids: list[int],
        sampling_frequency: float = 30000.0,
    ) -> None:
        self._traces = np.asarray(traces, dtype=float).reshape(-1, 1)
        self._channel_ids = list(channel_ids)
        self._sampling_frequency = float(sampling_frequency)
        self.calls: list[dict[str, object]] = []

    def get_channel_ids(self) -> list[int]:
        return list(self._channel_ids)

    def get_sampling_frequency(self) -> float:
        return float(self._sampling_frequency)

    def get_traces(
        self,
        *,
        start_frame: int,
        end_frame: int,
        channel_ids: list[int],
        return_in_uV: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "channel_ids": list(channel_ids),
                "return_in_uV": bool(return_in_uV),
            }
        )
        return np.asarray(self._traces[start_frame:end_frame], dtype=float)


def _install_fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeTsd:
        def __init__(self, *, t: np.ndarray, d: np.ndarray, time_units: str) -> None:
            self.t = np.asarray(t, dtype=float)
            self.d = np.asarray(d, dtype=float)
            self.time_units = time_units

        def save(self, path: Path) -> None:
            np.savez(path, t=self.t, d=self.d, time_units=self.time_units)

    fake_pynapple = types.SimpleNamespace(Tsd=_FakeTsd)
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)


def test_extract_theta_for_epoch_preserves_last_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = _FakeRecording([10.0, 11.0, 12.0, 13.0], [162], sampling_frequency=2000.0)
    monkeypatch.setattr(
        theta_module,
        "downsample_theta_trace",
        lambda timestamps, values, **kwargs: (
            np.asarray(timestamps, dtype=float)[::2],
            np.asarray(values, dtype=float)[::2],
            1000.0,
        ),
    )
    monkeypatch.setattr(
        theta_module,
        "compute_instantaneous_phase",
        lambda values: np.asarray(values, dtype=float) + 0.5,
    )

    timestamps_epoch, theta_lfp, theta_phase, output_sampling_frequency = theta_module.extract_theta_for_epoch(
        recording,
        epoch="01_r1",
        epoch_timestamps=np.array([1.0, 2.0], dtype=float),
        timestamps_ephys_all=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        theta_channel=162,
    )

    assert np.allclose(timestamps_epoch, np.array([1.0], dtype=float))
    assert np.allclose(theta_lfp, np.array([11.0], dtype=float))
    assert np.allclose(theta_phase, np.array([11.5], dtype=float))
    assert output_sampling_frequency == 1000.0
    assert recording.calls == [
        {
            "start_frame": 1,
            "end_frame": 3,
            "channel_ids": [162],
            "return_in_uV": True,
        }
    ]


def test_validate_epochs_rejects_missing_epochs() -> None:
    with pytest.raises(ValueError, match="Requested epochs were not found"):
        theta_module.validate_epochs(["01_r1", "02_r2"], ["01_r1", "03_r3"])


def test_resolve_theta_channel_prefers_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        theta_module,
        "get_session_theta_channel",
        lambda animal_name, date: pytest.fail("argument should bypass registry"),
    )

    theta_channel, source = theta_module.resolve_theta_channel("L15", "20241121", 162)

    assert theta_channel == 162
    assert source == "argument"


def test_resolve_theta_channel_uses_registry_when_missing() -> None:
    theta_channel, source = theta_module.resolve_theta_channel("L15", "20241121", None)

    assert theta_channel == 191
    assert source == "registry"


def test_load_theta_inputs_returns_timestamp_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        theta_module,
        "load_ephys_timestamps_by_epoch",
        lambda path: (
            ["01_r1"],
            {"01_r1": np.array([0.0, 0.1], dtype=float)},
            "pynapple",
        ),
    )
    monkeypatch.setattr(
        theta_module,
        "load_ephys_timestamps_all",
        lambda path: (np.array([0.0, 0.1], dtype=float), "pickle"),
    )

    epoch_tags, timestamps_by_epoch, timestamps_ephys_all, sources = theta_module.load_theta_inputs(
        tmp_path
    )

    assert epoch_tags == ["01_r1"]
    assert np.allclose(timestamps_by_epoch["01_r1"], np.array([0.0, 0.1], dtype=float))
    assert np.allclose(timestamps_ephys_all, np.array([0.0, 0.1], dtype=float))
    assert sources == {
        "timestamps_ephys": "pynapple",
        "timestamps_ephys_all": "pickle",
    }


def test_get_theta_phase_for_session_writes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_pynapple(monkeypatch)
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        theta_module,
        "load_theta_inputs",
        lambda path: (
            ["01_r1", "02_r2"],
            {
                "01_r1": np.array([0.0, 0.1, 0.2], dtype=float),
                "02_r2": np.array([1.0, 1.1], dtype=float),
            },
            np.array([0.0, 0.1, 0.2, 1.0, 1.1], dtype=float),
            {
                "timestamps_ephys": "pynapple",
                "timestamps_ephys_all": "pynapple",
            },
        ),
    )

    recording = _FakeRecording([10.0, 11.0, 12.0, 20.0, 21.0], [162, 200])
    monkeypatch.setattr(theta_module, "load_theta_filtered_recording", lambda *args, **kwargs: recording)
    monkeypatch.setattr(
        theta_module,
        "downsample_theta_trace",
        lambda timestamps, values, **kwargs: (
            np.asarray(timestamps, dtype=float)[::2],
            np.asarray(values, dtype=float)[::2],
            1000.0,
        ),
    )
    monkeypatch.setattr(
        theta_module,
        "compute_instantaneous_phase",
        lambda values: np.asarray(values, dtype=float) * -1.0,
    )
    log_path = analysis_path / "v1ca1_log" / "theta_log.json"
    monkeypatch.setattr(theta_module, "write_run_log", lambda **kwargs: log_path)

    outputs = theta_module.get_theta_phase_for_session(
        animal_name="RatA",
        date="20240101",
        theta_channel=162,
        data_root=tmp_path,
        nwb_root=tmp_path,
    )

    assert outputs["sources"] == {
        "timestamps_ephys": "pynapple",
        "timestamps_ephys_all": "pynapple",
    }
    assert outputs["theta_channel"] == 162
    assert outputs["theta_channel_source"] == "argument"
    assert outputs["output_sampling_frequency_hz"] == 1000.0
    assert outputs["selected_epochs"] == ["01_r1", "02_r2"]

    lfp_epoch_path = analysis_path / "oscillation" / "theta_lfp" / "01_r1.npz"
    phase_epoch_path = analysis_path / "oscillation" / "theta_phase" / "02_r2.npz"
    assert lfp_epoch_path.exists()
    assert phase_epoch_path.exists()
    assert not (analysis_path / "theta_lfp.pkl").exists()
    assert not (analysis_path / "theta_phase.pkl").exists()

    lfp_epoch_data = np.load(lfp_epoch_path)
    phase_epoch_data = np.load(phase_epoch_path)
    assert np.allclose(lfp_epoch_data["t"], np.array([0.0, 0.2], dtype=float))
    assert np.allclose(lfp_epoch_data["d"], np.array([10.0, 12.0], dtype=float))
    assert np.allclose(phase_epoch_data["t"], np.array([1.0], dtype=float))
    assert np.allclose(phase_epoch_data["d"], np.array([-20.0], dtype=float))

    metadata_path = analysis_path / "oscillation" / "theta_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["theta_channel"] == 162
    assert metadata["theta_channel_source"] == "argument"
    assert metadata["theta_band_hz"] == [4.0, 12.0]
    assert metadata["output_sampling_frequency_hz"] == 1000.0
    assert metadata["target_output_sampling_frequency_hz"] == 1000.0
    assert metadata["sources"]["timestamps_ephys"] == "pynapple"
    assert metadata["epochs"] == ["01_r1", "02_r2"]
    assert outputs["metadata_path"] == metadata_path
    assert outputs["log_path"] == log_path


def test_get_theta_phase_for_session_uses_registry_theta_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_pynapple(monkeypatch)
    analysis_path = tmp_path / "L15" / "20241121"
    analysis_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        theta_module,
        "load_theta_inputs",
        lambda path: (
            ["01_r1"],
            {"01_r1": np.array([0.0, 0.1], dtype=float)},
            np.array([0.0, 0.1], dtype=float),
            {
                "timestamps_ephys": "pynapple",
                "timestamps_ephys_all": "pynapple",
            },
        ),
    )
    recording = _FakeRecording([10.0, 11.0], [191, 200])
    monkeypatch.setattr(theta_module, "load_theta_filtered_recording", lambda *args, **kwargs: recording)
    monkeypatch.setattr(
        theta_module,
        "downsample_theta_trace",
        lambda timestamps, values, **kwargs: (
            np.asarray(timestamps, dtype=float),
            np.asarray(values, dtype=float),
            1000.0,
        ),
    )
    monkeypatch.setattr(
        theta_module,
        "compute_instantaneous_phase",
        lambda values: np.asarray(values, dtype=float),
    )
    monkeypatch.setattr(theta_module, "write_run_log", lambda **kwargs: analysis_path / "v1ca1_log" / "theta_log.json")

    outputs = theta_module.get_theta_phase_for_session(
        animal_name="L15",
        date="20241121",
        data_root=tmp_path,
        nwb_root=tmp_path,
    )

    assert outputs["theta_channel"] == 191
    assert outputs["theta_channel_source"] == "registry"
