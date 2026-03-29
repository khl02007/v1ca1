from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

import v1ca1.ripple.ripple_glm3 as ripple_glm3_module


def _write_ephys_pickles(
    analysis_path: Path,
    *,
    timestamps_by_epoch: dict[str, np.ndarray],
) -> np.ndarray:
    analysis_path.mkdir(parents=True, exist_ok=True)
    timestamps_ephys_all = np.concatenate(list(timestamps_by_epoch.values()))
    with open(analysis_path / "timestamps_ephys.pkl", "wb") as file:
        pickle.dump(timestamps_by_epoch, file)
    with open(analysis_path / "timestamps_ephys_all.pkl", "wb") as file:
        pickle.dump(timestamps_ephys_all, file)
    return timestamps_ephys_all


def _write_modern_ephys_npz(
    analysis_path: Path,
    *,
    epoch_tags: list[str],
    epoch_segments: list[np.ndarray],
) -> None:
    nap = pytest.importorskip("pynapple")

    timestamps_ephys_all = np.concatenate(epoch_segments)
    nap.Ts(t=timestamps_ephys_all, time_units="s").save(analysis_path / "timestamps_ephys_all.npz")
    epoch_intervals = nap.IntervalSet(
        start=np.asarray([segment[0] for segment in epoch_segments], dtype=float),
        end=np.asarray([segment[-1] for segment in epoch_segments], dtype=float),
        time_units="s",
    )
    epoch_intervals.set_info(epoch=epoch_tags)
    epoch_intervals.save(analysis_path / "timestamps_ephys.npz")


def _write_legacy_ripple_pickle(
    analysis_path: Path,
    *,
    tables: dict[str, dict[str, list[float]]],
) -> Path:
    pd = pytest.importorskip("pandas")

    output_path = analysis_path / "ripple" / "Kay_ripple_detector.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(
            {
                epoch: pd.DataFrame(table)
                for epoch, table in tables.items()
            },
            file,
        )
    return output_path


def _write_modern_ripple_parquet(
    analysis_path: Path,
    *,
    rows: list[dict[str, object]],
) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    output_path = analysis_path / "ripple" / "ripple_times.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def _fake_fit_results() -> dict[str, object]:
    return {
        "epoch": "01_s1",
        "random_seed": 45,
        "min_spikes_per_ripple": 0.1,
        "n_splits": 2,
        "n_shuffles_ripple": 2,
        "ripple_window": 0.2,
        "pre_window_s": 0.2,
        "pre_buffer_s": 0.02,
        "pre_exclude_guard_s": 0.05,
        "exclude_ripples": False,
        "n_ripples": 3,
        "n_pre": 2,
        "n_cells": 2,
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102]),
        "fold_info": [{"fold": 0}, {"fold": 1}],
        "pseudo_r2_ripple_folds": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "mae_ripple_folds": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
        "ll_ripple_folds": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "devexp_ripple_folds": np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32),
        "bits_per_spike_ripple_folds": np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
        "pseudo_r2_ripple_shuff_folds": np.zeros((2, 2, 2), dtype=np.float32),
        "mae_ripple_shuff_folds": np.ones((2, 2, 2), dtype=np.float32),
        "ll_ripple_shuff_folds": np.ones((2, 2, 2), dtype=np.float32),
        "devexp_ripple_shuff_folds": np.zeros((2, 2, 2), dtype=np.float32),
        "bits_per_spike_ripple_shuff_folds": np.zeros((2, 2, 2), dtype=np.float32),
        "y_ripple_test": np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "yhat_ripple": np.array([[0.1, 0.9], [0.9, 0.1], [0.8, 0.7]], dtype=np.float32),
        "yhat_ripple_shuff": np.zeros((2, 3, 2), dtype=np.float32),
        "y_pre_test": np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        "yhat_pre": np.array([[0.2, 0.8], [0.7, 0.6]], dtype=np.float32),
        "pseudo_r2_pre": np.array([0.05, 0.06], dtype=np.float32),
        "mae_pre": np.array([0.3, 0.4], dtype=np.float32),
        "ll_pre": np.array([0.7, 0.8], dtype=np.float32),
        "devexp_pre": np.array([0.04, 0.05], dtype=np.float32),
        "bits_per_spike_pre": np.array([0.01, 0.02], dtype=np.float32),
        "keep_x_all": np.array([True, True]),
        "x_mean_all": np.array([0.5, 0.5]),
        "x_std_all": np.array([1.0, 1.0]),
    }


def test_load_legacy_ripple_tables_groups_rows_by_epoch(tmp_path: Path) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_legacy_ripple_pickle(
        analysis_path,
        tables={
            "01_s1": {"start_time": [1.0, 2.5], "end_time": [1.2, 2.7], "mean_zscore": [2.0, 3.0]},
            "02_r1": {"start_time": [5.0], "end_time": [5.2], "mean_zscore": [4.0]},
        },
    )

    ripple_tables = ripple_glm3_module.load_legacy_ripple_tables(analysis_path)

    assert sorted(ripple_tables) == ["01_s1", "02_r1"]
    assert np.allclose(ripple_tables["01_s1"]["start_time"], [1.0, 2.5])
    assert np.allclose(ripple_tables["02_r1"]["end_time"], [5.2])
    assert np.allclose(ripple_tables["01_s1"]["mean_zscore"], [2.0, 3.0])


def test_prepare_ripple_glm3_session_rejects_missing_pickle_inputs_even_if_modern_exist(
    tmp_path: Path,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    analysis_path.mkdir(parents=True)
    _write_modern_ephys_npz(
        analysis_path,
        epoch_tags=["01_s1"],
        epoch_segments=[np.linspace(0.0, 0.9, 10, dtype=float)],
    )
    _write_modern_ripple_parquet(
        analysis_path,
        rows=[{"epoch": "01_s1", "start": 0.2, "end": 0.3}],
    )

    with pytest.raises(FileNotFoundError, match="timestamps_ephys.pkl"):
        ripple_glm3_module.prepare_ripple_glm3_session(
            animal_name="RatA",
            date="20240101",
            data_root=tmp_path,
        )


def test_prepare_ripple_glm3_session_uses_legacy_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    timestamps_ephys_all = _write_ephys_pickles(
        analysis_path,
        timestamps_by_epoch={
            "01_s1": np.linspace(0.0, 0.9, 10, dtype=float),
            "02_r1": np.linspace(2.0, 2.9, 10, dtype=float),
        },
    )
    _write_legacy_ripple_pickle(
        analysis_path,
        tables={
            "01_s1": {"start_time": [0.2], "end_time": [0.3]},
            "02_r1": {"start_time": [2.2], "end_time": [2.3]},
        },
    )
    (analysis_path / "sorting_v1").mkdir()
    (analysis_path / "sorting_ca1").mkdir()

    captured: dict[str, object] = {}

    def _fake_load_spikes_by_region(
        called_analysis_path: Path,
        called_timestamps: np.ndarray,
        regions: tuple[str, ...],
    ) -> dict[str, object]:
        captured["analysis_path"] = called_analysis_path
        captured["timestamps"] = np.asarray(called_timestamps, dtype=float)
        captured["regions"] = regions
        return {"v1": object(), "ca1": object()}

    monkeypatch.setattr(ripple_glm3_module, "load_spikes_by_region", _fake_load_spikes_by_region)

    session = ripple_glm3_module.prepare_ripple_glm3_session(
        animal_name="RatA",
        date="20240101",
        data_root=tmp_path,
    )

    assert session["sources"]["timestamps_ephys"] == "pickle"
    assert session["sources"]["timestamps_ephys_all"] == "pickle"
    assert session["sources"]["ripple_events"] == "pickle"
    assert session["ripple_event_epochs"] == ["01_s1", "02_r1"]
    assert np.allclose(captured["timestamps"], timestamps_ephys_all)
    assert captured["analysis_path"] == analysis_path
    assert captured["regions"] == ("v1", "ca1")
    assert set(session["epoch_intervals"]) == {"01_s1", "02_r1"}


def test_main_smoke_writes_legacy_npz_and_figures_from_pickle_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis_path = tmp_path / "RatA" / "20240101"
    _write_ephys_pickles(
        analysis_path,
        timestamps_by_epoch={"01_s1": np.linspace(0.0, 0.9, 10, dtype=float)},
    )
    _write_legacy_ripple_pickle(
        analysis_path,
        tables={"01_s1": {"start_time": [0.2], "end_time": [0.3]}},
    )
    (analysis_path / "sorting_v1").mkdir()
    (analysis_path / "sorting_ca1").mkdir()

    monkeypatch.setattr(
        ripple_glm3_module,
        "load_spikes_by_region",
        lambda analysis_path, timestamps_ephys_all, regions: {"v1": object(), "ca1": object()},
    )
    monkeypatch.setattr(
        ripple_glm3_module,
        "fit_ripple_glm_train_on_ripple_predict_pre",
        lambda **kwargs: _fake_fit_results(),
    )

    def _touch_plot(*args, out_path: Path, **kwargs) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        return out_path

    monkeypatch.setattr(ripple_glm3_module, "plot_ripple_glm_pseudo_r2", _touch_plot)
    monkeypatch.setattr(ripple_glm3_module, "plot_ripple_glm_mae", _touch_plot)
    monkeypatch.setattr(ripple_glm3_module, "plot_ripple_glm_delta_ll", _touch_plot)
    monkeypatch.setattr(ripple_glm3_module, "plot_pre_pseudo_r2_summary", _touch_plot)
    monkeypatch.setattr(ripple_glm3_module, "plot_pre_mae_summary", _touch_plot)
    monkeypatch.setattr(ripple_glm3_module, "plot_pre_ll_summary", _touch_plot)

    ripple_glm3_module.main(
        [
            "--animal-name",
            "RatA",
            "--date",
            "20240101",
            "--data-root",
            str(tmp_path),
            "--epochs",
            "01_s1",
        ]
    )

    result_path = (
        analysis_path
        / "ripple"
        / "glm_results_train_on_ripple"
        / "01_s1_train_ripple_predict_ripple_and_pre_exclude_ripples_False_ripple_window_0.2_min_spikes_per_ripple_0.1_ridge_strength_0.1.npz"
    )
    assert result_path.exists()
    loaded = ripple_glm3_module.load_results_npz(result_path)
    assert {"yhat_ripple", "ll_ripple_folds", "y_pre_test", "yhat_pre", "ll_pre"} <= set(loaded)

    figure_dir = analysis_path / "figs" / "ripple" / "train_on_ripple"
    assert len(list(figure_dir.glob("*.png"))) == 6
