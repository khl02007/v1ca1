from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import v1ca1.sleep.plot_sleep_phases as sleep_plot_module
from v1ca1.helper import session


class _FakeRecording:
    def __init__(self, channel_ids: list[int], sampling_frequency: float = 300.0) -> None:
        self._channel_ids = list(channel_ids)
        self._sampling_frequency = float(sampling_frequency)

    def get_channel_ids(self) -> list[int]:
        return list(self._channel_ids)

    def get_sampling_frequency(self) -> float:
        return float(self._sampling_frequency)


class _FakeSorting:
    def __init__(self, spike_trains: dict[int, np.ndarray]) -> None:
        self._spike_trains = {
            int(unit_id): np.asarray(spike_train, dtype=int)
            for unit_id, spike_train in spike_trains.items()
        }

    def get_unit_ids(self) -> list[int]:
        return list(self._spike_trains)

    def get_unit_spike_train(self, unit_id: int) -> np.ndarray:
        return np.asarray(self._spike_trains[int(unit_id)], dtype=int)

    def time_slice(self, start_time: float, end_time: float) -> "_FakeSorting":
        return self


def test_parse_arguments_accepts_epoch_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_sleep_phases.py",
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
            "--epochs",
            "01_s1",
            "02_r1",
            "--region",
            "v1",
            "--show",
        ],
    )

    args = sleep_plot_module.parse_arguments()

    assert args.data_root == session.DEFAULT_DATA_ROOT
    assert args.nwb_root == session.DEFAULT_NWB_ROOT
    assert args.epochs == ["01_s1", "02_r1"]
    assert args.region == "v1"
    assert args.show is True


def test_select_plot_epochs_skips_epochs_without_position_data() -> None:
    selected_epochs, skipped_epochs = sleep_plot_module.select_plot_epochs(
        ["01_s1", "02_r1", "03_s2"],
        timestamps_position={
            "01_s1": np.array([0.0, 1.0], dtype=float),
            "02_r1": np.array([0.0, 1.0], dtype=float),
        },
        position_by_epoch={
            "02_r1": np.zeros((2, 2), dtype=float),
            "03_s2": np.zeros((2, 2), dtype=float),
        },
    )

    assert selected_epochs == ["02_r1"]
    assert skipped_epochs == ["01_s1", "03_s2"]


def test_plot_sleep_phases_for_session_rejects_missing_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    monkeypatch.setattr(
        sleep_plot_module,
        "load_sleep_session_inputs",
        lambda path: {
            "epoch_tags": ["01_s1"],
            "timestamps_ephys": {"01_s1": np.array([0.0, 1.0], dtype=float)},
            "timestamps_ephys_all": np.array([0.0, 1.0], dtype=float),
            "timestamps_position": {"01_s1": np.array([0.0, 1.0], dtype=float)},
            "position_by_epoch": {"01_s1": np.zeros((2, 2), dtype=float)},
            "sources": {},
        },
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "load_recording",
        lambda path: _FakeRecording([336], sampling_frequency=300.0),
    )

    with pytest.raises(ValueError, match="V1 LFP channel 12 was not found"):
        sleep_plot_module.plot_sleep_phases_for_session(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
            nwb_root=tmp_path,
        )


def test_plot_sleep_phases_for_session_saves_epoch_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    monkeypatch.setattr(
        sleep_plot_module,
        "load_sleep_session_inputs",
        lambda path: {
            "epoch_tags": ["02_r1"],
            "timestamps_ephys": {"02_r1": np.array([0.0, 1.0], dtype=float)},
            "timestamps_ephys_all": np.linspace(0.0, 1.0, 11, dtype=float),
            "timestamps_position": {"02_r1": np.linspace(0.0, 1.0, 11, dtype=float)},
            "position_by_epoch": {"02_r1": np.zeros((11, 2), dtype=float)},
            "sources": {"timestamps_ephys": "pynapple"},
        },
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "load_recording",
        lambda path: _FakeRecording([12, 336], sampling_frequency=300.0),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "load_sleep_sortings",
        lambda path: {
            "ca1": _FakeSorting({11: np.array([1, 2, 3]), 12: np.array([4, 5])}),
            "v1": _FakeSorting({21: np.array([1, 2, 3]), 22: np.array([5, 6])}),
        },
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "get_epoch_trace",
        lambda recording, **kwargs: (
            np.linspace(0.0, 1.0, 11, dtype=float),
            np.linspace(1.0, 2.0, 11, dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "butter_filter_and_decimate",
        lambda timestamps, data, **kwargs: (
            np.linspace(0.0, 1.0, 11, dtype=float),
            np.linspace(0.0, 1.0, 11, dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "lowpass_filter",
        lambda signal, cutoff_hz, sampling_frequency: np.asarray(signal, dtype=float),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "decimate_signal",
        lambda signal, factor: np.asarray(signal, dtype=float),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "compute_spectrogram_principal_component",
        lambda signal, sampling_frequency, **kwargs: (
            np.linspace(0.0, 1.0, 11, dtype=float),
            np.linspace(0.0, 2.0, 11, dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "compute_theta_delta_ratio",
        lambda signal, sampling_frequency, **kwargs: (
            np.linspace(0.0, 1.0, 11, dtype=float),
            np.linspace(1.0, 3.0, 11, dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "get_time_spike_indicator",
        lambda timestamps_position, **kwargs: (
            np.linspace(0.0, 1.0, 11, dtype=float),
            np.ones((11, 2), dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "compute_multiunit_population_rate",
        lambda spike_indicator, timestamps: np.linspace(0.0, 10.0, 11, dtype=float),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "get_speed_trace",
        lambda position, timestamps_position, position_offset: (
            np.zeros((11, 2), dtype=float),
            np.linspace(0.0, 1.0, 11, dtype=float),
            np.linspace(0.0, 20.0, 11, dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "compute_firing_rate_matrix",
        lambda sorting, **kwargs: (
            np.array([[0.0, 1.0, 0.5], [1.0, 0.0, -0.5]], dtype=float),
            np.array([0.2, 0.5, 0.8], dtype=float),
        ),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "order_units_by_epoch_spike_count",
        lambda sorting, **kwargs: np.array([1, 0], dtype=int),
    )
    monkeypatch.setattr(
        sleep_plot_module,
        "get_linearized_run_position",
        lambda position: np.linspace(0.0, 100.0, 11, dtype=float),
    )
    log_path = analysis_path / "v1ca1_log" / "sleep_plot_log.json"
    monkeypatch.setattr(sleep_plot_module, "write_run_log", lambda **kwargs: log_path)

    outputs = sleep_plot_module.plot_sleep_phases_for_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        nwb_root=tmp_path,
        epochs=["02_r1"],
        position_offset=0,
        show=False,
    )

    output_path = outputs["figure_paths"]["02_r1"]
    assert output_path.exists()
    assert output_path == analysis_path / "figs" / "sleep" / "02_r1_ca1_v1.png"
    assert outputs["selected_epochs"] == ["02_r1"]
    assert outputs["heatmap_regions"] == ["ca1", "v1"]
    assert outputs["log_path"] == log_path
