from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import v1ca1.ripple.legacy.plot_ripple_modulation as legacy_ripple_plot
import v1ca1.ripple.plot_ripple_modulation as ripple_plot


class _FakeTs:
    def __init__(self, t: np.ndarray, **_kwargs: object) -> None:
        self.t = np.asarray(t, dtype=float)


class _FakeIntervalSet:
    def __init__(self, start: float, end: float, **_kwargs: object) -> None:
        self.start = np.asarray([start], dtype=float)
        self.end = np.asarray([end], dtype=float)


class _FakePerievent:
    def __init__(self, counts: pd.DataFrame) -> None:
        self._counts = counts

    def count(self, _bin_size_s: float) -> pd.DataFrame:
        return self._counts


def _install_fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> None:
    def _compute_perievent(
        *,
        timestamps: types.SimpleNamespace,
        tref: _FakeTs,
        minmax: tuple[float, float],
        time_unit: str,
    ) -> _FakePerievent:
        assert isinstance(tref, _FakeTs)
        assert minmax == (-0.2, 0.2)
        assert time_unit == "s"
        if timestamps.unit_id == 11:
            return _FakePerievent(
                pd.DataFrame(
                    [[0.0, 1.0], [2.0, 1.0], [4.0, 5.0], [3.0, 2.0]],
                    index=np.array([-0.15, -0.05, 0.05, 0.15], dtype=float),
                )
            )
        return _FakePerievent(
            pd.DataFrame(
                [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]],
                index=np.array([-0.15, -0.05, 0.05, 0.15], dtype=float),
            )
        )

    fake_pynapple = types.SimpleNamespace(
        Ts=_FakeTs,
        IntervalSet=_FakeIntervalSet,
        compute_perievent=_compute_perievent,
    )
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)


def _write_detect_ripple_parquet(analysis_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    ripple_path = analysis_path / "ripple" / ripple_plot.RIPPLE_EVENT_FILENAME
    ripple_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "epoch": ["02_r1", "02_r1", "03_r2", "03_r2"],
            "start": [0.25, 0.75, 1.25, 1.75],
            "end": [0.30, 0.80, 1.30, 1.80],
            "mean_zscore": [4.5, 5.0, 4.2, 4.6],
        }
    ).to_parquet(ripple_path, index=False)
    return ripple_path


def test_parse_arguments_uses_modern_defaults() -> None:
    args = ripple_plot.parse_arguments(["--animal-name", "RatA", "--date", "20240101"])

    assert args.data_root == ripple_plot.DEFAULT_DATA_ROOT
    assert args.region is None
    assert args.epochs is None
    assert args.ripple_threshold_zscore == ripple_plot.DEFAULT_RIPPLE_THRESHOLD_ZSCORE
    assert args.bin_size_s == ripple_plot.DEFAULT_BIN_SIZE_S
    assert args.heatmap_normalize == ripple_plot.DEFAULT_HEATMAP_NORMALIZE
    assert args.overwrite is False
    assert args.show is False


def test_legacy_module_reexports_new_entrypoint() -> None:
    assert (
        legacy_ripple_plot.plot_ripple_modulation_for_session
        is ripple_plot.plot_ripple_modulation_for_session
    )


def test_load_detect_ripple_tables_uses_detect_ripples_parquet(tmp_path: Path) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    ripple_path = _write_detect_ripple_parquet(analysis_path)

    loaded_tables, source = ripple_plot.load_detect_ripple_tables(analysis_path)

    assert ripple_path.exists()
    assert source == "detect_ripples_parquet"
    assert sorted(loaded_tables) == ["02_r1", "03_r2"]
    assert np.allclose(loaded_tables["02_r1"]["start_time"], [0.25, 0.75])
    assert np.allclose(loaded_tables["03_r2"]["end_time"], [1.30, 1.80])


def test_main_fails_cleanly_when_detect_ripple_output_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    (analysis_path / "sorting_v1").mkdir(parents=True)

    with pytest.raises(SystemExit) as exc_info:
        ripple_plot.main(
            [
                "--animal-name",
                "RatA",
                "--date",
                "20240101",
                "--data-root",
                str(tmp_path),
                "--region",
                "v1",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Could not find saved ripple events at" in captured.err
    assert "Run `python -m v1ca1.ripple.detect_ripples" in captured.err


def test_select_epochs_uses_ephys_ripple_intersection_by_default() -> None:
    selected, skipped = ripple_plot.select_epochs(
        ["01_s1", "02_r1", "03_r2"],
        {
            "02_r1": pd.DataFrame({"start_time": [1.0], "end_time": [1.1]}),
            "03_r2": pd.DataFrame({"start_time": [2.0], "end_time": [2.1]}),
        },
    )

    assert selected == ["02_r1", "03_r2"]
    assert skipped == ["01_s1"]


def test_select_epochs_skips_requested_epoch_without_ripple_output() -> None:
    selected, skipped = ripple_plot.select_epochs(
        ["01_s1", "02_r1"],
        {"02_r1": pd.DataFrame({"start_time": [1.0], "end_time": [1.1]})},
        requested_epochs=["01_s1", "02_r1"],
    )

    assert selected == ["02_r1"]
    assert skipped == ["01_s1"]


def test_filter_ripple_table_by_threshold_filters_rows() -> None:
    table = pd.DataFrame(
        {
            "start_time": [1.0, 2.0, 3.0],
            "end_time": [1.1, 2.1, 3.1],
            "mean_zscore": [3.9, 4.0, 4.1],
        }
    )

    filtered = ripple_plot.filter_ripple_table_by_threshold(
        table,
        epoch="02_r1",
        ripple_threshold_zscore=4.0,
    )

    assert np.allclose(filtered["start_time"], [3.0])


def test_compute_modulation_stats_returns_expected_zscore() -> None:
    stats = ripple_plot.compute_modulation_stats(
        np.array([-0.15, -0.05, 0.05, 0.15], dtype=float),
        np.array([5.0, 15.0, 45.0, 25.0], dtype=float),
        response_window=(0.0, 0.2),
        baseline_window=(-0.2, 0.0),
    )

    assert stats["baseline_mean_hz"] == pytest.approx(10.0)
    assert stats["baseline_std_hz"] == pytest.approx(5.0)
    assert stats["response_mean_hz"] == pytest.approx(35.0)
    assert stats["response_zscore"] == pytest.approx(5.0)
    assert stats["invalid_reason"] is None


def test_compute_modulation_stats_marks_zero_baseline_std_invalid() -> None:
    stats = ripple_plot.compute_modulation_stats(
        np.array([-0.15, -0.05, 0.05, 0.15], dtype=float),
        np.array([10.0, 10.0, 20.0, 20.0], dtype=float),
        response_window=(0.0, 0.2),
        baseline_window=(-0.2, 0.0),
    )

    assert stats["baseline_std_hz"] == pytest.approx(0.0)
    assert np.isnan(stats["response_zscore"])
    assert stats["invalid_reason"] == "zero_baseline_std"


def test_normalize_heatmap_rows_peak_normalizes_by_default() -> None:
    normalized = ripple_plot.normalize_heatmap_rows(
        np.array([[1.0, 2.0, 4.0], [2.0, 2.0, 1.0]], dtype=float),
        mode="max",
    )

    assert np.allclose(normalized[0], [0.25, 0.5, 1.0])
    assert np.allclose(normalized[1], [1.0, 1.0, 0.5])


def test_normalize_heatmap_rows_supports_zscore() -> None:
    normalized = ripple_plot.normalize_heatmap_rows(
        np.array([[1.0, 2.0, 3.0]], dtype=float),
        mode="zscore",
    )

    assert np.allclose(normalized[0], [-1.22474487, 0.0, 1.22474487])


def test_compute_heatmap_unit_order_sorts_by_peak_firing() -> None:
    order = ripple_plot.compute_heatmap_unit_order(
        np.array(
            [
                [1.0, 3.0, 2.0],
                [0.0, 5.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=float,
        )
    )

    assert np.array_equal(order, [1, 0, 2])


def test_plot_ripple_modulation_for_session_saves_epoch_specific_outputs_and_reuses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("pyarrow")
    _install_fake_pynapple(monkeypatch)

    analysis_path = tmp_path / "RatA" / "20240101"
    (analysis_path / "sorting_v1").mkdir(parents=True)
    _write_detect_ripple_parquet(analysis_path)

    monkeypatch.setattr(
        ripple_plot,
        "load_ephys_timestamps_by_epoch",
        lambda path: (
            ["01_s1", "02_r1", "03_r2"],
            {
                "01_s1": np.array([0.0, 1.0], dtype=float),
                "02_r1": np.array([0.0, 1.0], dtype=float),
                "03_r2": np.array([1.0, 2.0], dtype=float),
            },
            "pynapple",
        ),
    )
    monkeypatch.setattr(
        ripple_plot,
        "load_ephys_timestamps_all",
        lambda path: (np.linspace(0.0, 2.0, 2001, dtype=float), "pynapple"),
    )
    monkeypatch.setattr(
        ripple_plot,
        "load_spikes_by_region",
        lambda analysis_path, timestamps_ephys_all, regions: {
            "v1": {
                11: types.SimpleNamespace(unit_id=11),
                12: types.SimpleNamespace(unit_id=12),
            }
        },
    )

    result = ripple_plot.plot_ripple_modulation_for_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        region="v1",
        ripple_threshold_zscore=4.0,
        bin_size_s=0.1,
        time_before_s=0.2,
        time_after_s=0.2,
        response_window_start_s=0.0,
        response_window_end_s=0.2,
        baseline_window_start_s=-0.2,
        baseline_window_end_s=0.0,
        heatmap_normalize="max",
        show=False,
    )

    assert result["selected_epochs"] == ["02_r1", "03_r2"]
    assert result["skipped_epochs_without_ripple_output"] == ["01_s1"]
    assert sorted(result["epoch_results"]) == ["02_r1", "03_r2"]

    epoch_result = result["epoch_results"]["02_r1"]
    assert epoch_result["peri_ripple_firing_rate_path"].exists()
    assert epoch_result["summary_path"].exists()
    assert epoch_result["figure_path"].exists()
    assert "02_r1" in epoch_result["peri_ripple_firing_rate_path"].name
    assert "02_r1" in epoch_result["summary_path"].name
    assert "02_r1" in epoch_result["figure_path"].name

    firing_rate_table = pd.read_parquet(epoch_result["peri_ripple_firing_rate_path"])
    assert set(ripple_plot.PERI_RIPPLE_FIRING_RATE_COLUMNS).issubset(firing_rate_table.columns)
    assert set(firing_rate_table["epoch"]) == {"02_r1"}

    summary_table = pd.read_parquet(epoch_result["summary_path"])
    assert set(ripple_plot.SUMMARY_COLUMNS).issubset(summary_table.columns)
    assert summary_table.shape[0] == 2

    monkeypatch.setattr(
        ripple_plot,
        "load_spikes_by_region",
        lambda *args, **kwargs: pytest.fail("load_spikes_by_region should not run when epoch caches exist"),
    )

    cached_result = ripple_plot.plot_ripple_modulation_for_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
        region="v1",
        ripple_threshold_zscore=4.0,
        bin_size_s=0.1,
        time_before_s=0.2,
        time_after_s=0.2,
        response_window_start_s=0.0,
        response_window_end_s=0.2,
        baseline_window_start_s=-0.2,
        baseline_window_end_s=0.0,
        heatmap_normalize="max",
        show=False,
    )

    captured = capsys.readouterr()
    assert "Loading detect_ripples.py output" in captured.out
    assert "Skipping epochs not present in ripple_times.parquet: ['01_s1']" in captured.out
    assert "Using saved peri-ripple firing-rate table for epoch 02_r1" in captured.out
    assert "Using saved peri-ripple firing-rate table for epoch 03_r2" in captured.out
    assert cached_result["epoch_results"]["02_r1"]["peri_ripple_firing_rate_path"] == epoch_result["peri_ripple_firing_rate_path"]
