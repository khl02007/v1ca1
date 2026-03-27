from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import v1ca1.oscillation.plot_theta_phase_preference as phase_plot


class _FakeTs:
    def __init__(self, times: np.ndarray) -> None:
        self.t = np.asarray(times, dtype=float)


class _FakeTsGroup(dict):
    pass


class _FakeThetaPhaseTsd:
    def __init__(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        self.t = np.asarray(timestamps, dtype=float)
        self.d = np.asarray(values, dtype=float)


class _FakeIntervalSet:
    def __init__(self, start: np.ndarray, end: np.ndarray) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)


def _install_fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> None:
    def _load_file(path: Path) -> _FakeThetaPhaseTsd:
        data = np.load(path)
        return _FakeThetaPhaseTsd(data["t"], data["d"])

    fake_pynapple = types.SimpleNamespace(load_file=_load_file)
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)


def test_get_theta_phase_epoch_paths_fails_when_outputs_missing(tmp_path: Path) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Run `python -m v1ca1.oscillation.get_theta_phase"):
        phase_plot.get_theta_phase_epoch_paths(analysis_path)


def test_select_theta_phase_epochs_uses_run_epoch_intersection() -> None:
    selected = phase_plot.select_theta_phase_epochs(
        ["02_r1", "03_r2"],
        {
            "01_s1": Path("01_s1.npz"),
            "03_r2": Path("03_r2.npz"),
            "05_r3": Path("05_r3.npz"),
        },
    )

    assert selected == ["03_r2"]


def test_select_theta_phase_epochs_rejects_requested_epoch_without_output() -> None:
    with pytest.raises(FileNotFoundError, match="does not have a saved theta-phase output"):
        phase_plot.select_theta_phase_epochs(
            ["02_r1", "04_r2"],
            {"02_r1": Path("02_r1.npz")},
            requested_epoch="04_r2",
        )


def test_sample_phase_at_spike_times_interpolates_across_wrap() -> None:
    sampled = phase_plot.sample_phase_at_spike_times(
        np.array([0.5], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        np.array([np.pi - 0.1, -np.pi + 0.1], dtype=float),
    )

    assert sampled.shape == (1,)
    assert np.isclose(np.abs(sampled[0]), np.pi, atol=0.15)
    assert not np.isclose(sampled[0], 0.0, atol=0.5)


def test_build_unit_phase_histograms_filters_and_counts_units() -> None:
    spikes = _FakeTsGroup(
        {
            11: _FakeTs(np.array([0.10, 0.20, 0.30], dtype=float)),
            12: _FakeTs(np.array([0.15], dtype=float)),
            13: _FakeTs(np.array([1.50, 1.60], dtype=float)),
        }
    )

    unit_ids, histograms, movement_firing_rates_hz = phase_plot.build_unit_phase_histograms(
        spikes,
        np.array([0.0, 0.2, 0.4, 0.6], dtype=float),
        np.array([-np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0], dtype=float),
        phase_bin_edges=phase_plot.build_phase_bin_edges(4),
        movement_interval=_FakeIntervalSet(np.array([0.0]), np.array([0.5])),
        min_spikes=2,
    )

    assert np.array_equal(unit_ids, [11])
    assert histograms.shape == (1, 4)
    assert np.isclose(histograms.sum(), 3.0)
    assert np.allclose(movement_firing_rates_hz, [6.0])


def test_build_unit_phase_histograms_uses_only_movement_spikes() -> None:
    spikes = _FakeTsGroup(
        {
            11: _FakeTs(np.array([0.10, 0.20, 0.30, 0.80, 0.90], dtype=float)),
        }
    )

    unit_ids, histograms, movement_firing_rates_hz = phase_plot.build_unit_phase_histograms(
        spikes,
        np.linspace(0.0, 1.0, 11, dtype=float),
        np.linspace(-np.pi, np.pi, 11, dtype=float),
        phase_bin_edges=phase_plot.build_phase_bin_edges(5),
        movement_interval=_FakeIntervalSet(np.array([0.0]), np.array([0.35])),
        min_spikes=4,
    )

    assert unit_ids.size == 0
    assert histograms.shape == (0, 5)
    assert movement_firing_rates_hz.size == 0


def test_compute_unit_order_sorts_by_peak_bin_and_pushes_invalid_rows_last() -> None:
    values = np.array(
        [
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    order = phase_plot.compute_unit_order(values)

    assert np.array_equal(order, [2, 0, 1, 3])


def test_compute_unit_order_by_movement_rate_sorts_descending() -> None:
    order = phase_plot.compute_unit_order_by_movement_rate(
        np.array([2.5, np.nan, 7.0, 2.5], dtype=float)
    )

    assert np.array_equal(order, [2, 0, 3, 1])


def test_plot_theta_phase_preference_for_session_saves_region_epoch_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    _install_fake_pynapple(monkeypatch)

    analysis_path = tmp_path / "RatA" / "20240101"
    theta_phase_dir = analysis_path / "oscillation" / "theta_phase"
    theta_phase_dir.mkdir(parents=True)
    np.savez(
        theta_phase_dir / "02_r1.npz",
        t=np.linspace(0.0, 1.0, 101, dtype=float),
        d=np.linspace(-np.pi, np.pi, 101, dtype=float),
        time_units="s",
    )
    with open(analysis_path / "oscillation" / "theta_metadata.json", "w", encoding="utf-8") as file:
        json.dump({"epochs": ["02_r1"]}, file)

    monkeypatch.setattr(
        phase_plot,
        "load_epoch_tags",
        lambda path: (["01_s1", "02_r1", "03_s2"], "pynapple"),
    )
    monkeypatch.setattr(
        phase_plot,
        "load_ephys_timestamps_all",
        lambda path: (np.linspace(0.0, 1.0, 1001, dtype=float), "pynapple"),
    )
    monkeypatch.setattr(
        phase_plot,
        "load_spikes_by_region",
        lambda analysis_path, timestamps_ephys_all, regions: {
            region: _FakeTsGroup(
                {
                    11: _FakeTs(np.linspace(0.1, 0.9, 30, dtype=float)),
                    12: _FakeTs(np.linspace(0.15, 0.85, 35, dtype=float)),
                }
            )
            for region in regions
        },
    )
    monkeypatch.setattr(
        phase_plot,
        "load_position_timestamps",
        lambda path: (["02_r1"], {"02_r1": np.linspace(0.0, 1.0, 101, dtype=float)}, "pynapple"),
    )
    monkeypatch.setattr(
        phase_plot,
        "load_position_data_with_precedence",
        lambda analysis_path, **kwargs: (
            {"02_r1": np.column_stack((np.linspace(0.0, 1.0, 101), np.linspace(1.0, 2.0, 101)))},
            "position.parquet",
        ),
    )
    monkeypatch.setattr(
        phase_plot,
        "build_speed_tsd",
        lambda position, timestamps_position, position_offset: object(),
    )
    monkeypatch.setattr(
        phase_plot,
        "build_movement_interval",
        lambda speed_tsd, speed_threshold_cm_s: _FakeIntervalSet(
            np.array([0.0], dtype=float),
            np.array([1.0], dtype=float),
        ),
    )

    output_paths = phase_plot.plot_theta_phase_preference_for_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        region="v1",
        phase_bin_count=12,
        min_spikes=5,
        show=False,
    )

    output_path = output_paths["v1"]["02_r1"]
    assert output_path is not None
    assert output_path.exists()
    assert output_path == analysis_path / "figs" / "theta_phase_preference" / "v1_02_r1.png"
