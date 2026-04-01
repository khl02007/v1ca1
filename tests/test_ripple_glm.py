from __future__ import annotations

import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import v1ca1.ripple.ripple_glm as ripple_glm_module
from v1ca1.ripple.ripple_glm import (
    _build_ripple_sample_windows,
    _count_spikes_in_windows,
    _format_nemos_solver_selection_message,
    _resolve_nemos_population_glm_solver,
    build_epoch_fit_dataset,
    build_metric_figure_data,
    empirical_p_values,
    fit_ripple_glm_train_on_ripple,
    get_epoch_skip_reason,
    keep_single_ripple_windows,
    load_ripple_tables,
    load_ripple_tables_from_parquet_output,
    load_ripple_tables_from_interval_output,
    load_ripple_tables_from_legacy_pickle,
    parse_arguments,
    remove_duplicate_ripples,
    save_epoch_figures,
    validate_arguments,
)


def test_load_ripple_tables_normalizes_modern_and_legacy_sources(tmp_path) -> None:
    nap = pytest.importorskip("pynapple")

    modern_path = tmp_path / "ripple_times.npz"
    legacy_path = tmp_path / "Kay_ripple_detector.pkl"

    interval_set = nap.IntervalSet(
        start=np.array([1.0, 2.5, 5.0], dtype=float),
        end=np.array([1.2, 2.7, 5.2], dtype=float),
        time_units="s",
    )
    interval_set.set_info(epoch=["01_s1", "01_s1", "02_r1"])
    interval_set.save(modern_path)

    legacy_payload = {
        "01_s1": pd.DataFrame(
            {
                "start_time": [1.0, 2.5],
                "end_time": [1.2, 2.7],
            }
        ),
        "02_r1": pd.DataFrame({"start_time": [5.0], "end_time": [5.2]}),
    }
    with open(legacy_path, "wb") as file:
        pickle.dump(legacy_payload, file)

    modern_tables = load_ripple_tables_from_interval_output(modern_path)
    legacy_tables = load_ripple_tables_from_legacy_pickle(legacy_path)

    assert sorted(modern_tables) == ["01_s1", "02_r1"]
    assert sorted(legacy_tables) == ["01_s1", "02_r1"]
    assert np.allclose(modern_tables["01_s1"]["start_time"], [1.0, 2.5])
    assert np.allclose(modern_tables["01_s1"]["end_time"], [1.2, 2.7])
    assert np.allclose(legacy_tables["02_r1"]["start_time"], [5.0])
    assert np.allclose(legacy_tables["02_r1"]["end_time"], [5.2])


def test_resolve_nemos_population_glm_solver_keeps_lbfgs_for_nemos_0_2_5() -> None:
    version_text, solver_name = _resolve_nemos_population_glm_solver(
        SimpleNamespace(__version__="0.2.5")
    )

    message = _format_nemos_solver_selection_message(version_text, solver_name)

    assert version_text == "0.2.5"
    assert solver_name == "LBFGS"
    assert "0.2.5" in message
    assert "solver_name='LBFGS'" in message


def test_resolve_nemos_population_glm_solver_uses_jaxopt_after_0_2_5() -> None:
    version_text, solver_name = _resolve_nemos_population_glm_solver(
        SimpleNamespace(__version__="0.2.7")
    )

    message = _format_nemos_solver_selection_message(version_text, solver_name)

    assert version_text == "0.2.7"
    assert solver_name == "LBFGS[jaxopt]"
    assert "0.2.7" in message
    assert "solver_name='LBFGS[jaxopt]'" in message


def test_resolve_nemos_population_glm_solver_falls_back_to_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ripple_glm_module, "package_version", lambda name: "0.2.6")

    version_text, solver_name = _resolve_nemos_population_glm_solver(SimpleNamespace())

    message = _format_nemos_solver_selection_message(version_text, solver_name)

    assert version_text == "0.2.6"
    assert solver_name == "LBFGS[jaxopt]"
    assert "0.2.6" in message
    assert "solver_name='LBFGS[jaxopt]'" in message


def test_load_ripple_tables_prefers_parquet_and_preserves_extra_columns(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    analysis_path = tmp_path / "RatA" / "20240101"
    ripple_dir = analysis_path / "ripple"
    ripple_dir.mkdir(parents=True)
    parquet_path = ripple_dir / "ripple_times.parquet"
    legacy_path = ripple_dir / "Kay_ripple_detector.pkl"

    pd.DataFrame(
        {
            "epoch": ["01_s1", "01_s1", "02_r1"],
            "start": [1.0, 2.5, 5.0],
            "end": [1.2, 2.7, 5.2],
            "mean_zscore": [2.0, 3.0, 4.0],
        }
    ).to_parquet(parquet_path, index=False)

    with open(legacy_path, "wb") as file:
        pickle.dump(
            {"02_r1": pd.DataFrame({"start_time": [9.0], "end_time": [9.2]})},
            file,
        )

    parquet_tables = load_ripple_tables_from_parquet_output(parquet_path)
    loaded_tables, source = load_ripple_tables(analysis_path)

    assert source == "parquet"
    assert sorted(parquet_tables) == ["01_s1", "02_r1"]
    assert np.allclose(parquet_tables["01_s1"]["start_time"], [1.0, 2.5])
    assert np.allclose(parquet_tables["02_r1"]["end_time"], [5.2])
    assert np.allclose(parquet_tables["01_s1"]["mean_zscore"], [2.0, 3.0])
    assert np.allclose(loaded_tables["02_r1"]["mean_zscore"], [4.0])


def test_remove_duplicate_ripples_uses_previous_kept_ripple() -> None:
    ripple_table = pd.DataFrame(
        {
            "start_time": [0.0, 0.05, 0.19, 0.26],
            "end_time": [0.02, 0.07, 0.21, 0.28],
        }
    )

    filtered_table, keep_mask = remove_duplicate_ripples(
        ripple_table,
        ripple_window_s=0.2,
    )

    assert np.array_equal(keep_mask, [True, False, False, True])
    assert np.allclose(filtered_table["start_time"], [0.0, 0.26])
    assert np.allclose(filtered_table["end_time"], [0.02, 0.28])


def test_keep_single_ripple_windows_requires_bidirectional_isolation() -> None:
    ripple_table = pd.DataFrame(
        {
            "start_time": [0.0, 0.2, 0.39, 1.0],
            "end_time": [0.02, 0.22, 0.41, 1.02],
        }
    )

    filtered_table, keep_mask = keep_single_ripple_windows(
        ripple_table,
        ripple_window_s=0.2,
    )

    assert np.array_equal(keep_mask, [True, False, False, True])
    assert np.allclose(filtered_table["start_time"], [0.0, 1.0])


def test_validate_arguments_rejects_conflicting_ripple_selection_modes() -> None:
    args = SimpleNamespace(
        ripple_window_s=0.2,
        remove_duplicate_ripples=True,
        keep_single_ripple_windows=True,
        min_spikes_per_ripple=0.1,
        min_ca1_spikes_per_ripple=0.0,
        n_splits=5,
        n_shuffles_ripple=0,
        ridge_strengths=[1e-1],
        maxiter=20,
        tol=1e-4,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_arguments(args)


def test_build_ripple_sample_windows_preserves_overlapping_fixed_windows() -> None:
    ripple_table = pd.DataFrame(
        {
            "start_time": [1.05, 1.0],
            "end_time": [1.2, 1.1],
        }
    )

    starts, ends = _build_ripple_sample_windows(ripple_table, ripple_window_s=0.1)

    assert np.allclose(starts, [1.0, 1.05])
    assert np.allclose(ends, [1.1, 1.15])


def test_count_spikes_in_windows_preserves_overlapping_rows() -> None:
    class FakeTs:
        def __init__(self, timestamps: list[float]) -> None:
            self.t = np.asarray(timestamps, dtype=float)

    class FakeTsGroup:
        def __init__(self, spikes_by_unit: dict[int, list[float]]) -> None:
            self._spikes_by_unit = {
                unit_id: FakeTs(timestamps) for unit_id, timestamps in spikes_by_unit.items()
            }

        def keys(self):
            return list(self._spikes_by_unit)

        def __getitem__(self, unit_id: int) -> FakeTs:
            return self._spikes_by_unit[unit_id]

    counts, unit_ids = _count_spikes_in_windows(
        FakeTsGroup(
            {
                101: [1.01, 1.06, 1.20],
                102: [1.03, 1.11],
            }
        ),
        window_starts=np.array([1.0, 1.05], dtype=float),
        window_ends=np.array([1.1, 1.15], dtype=float),
    )

    assert np.array_equal(unit_ids, [101, 102])
    assert counts.shape == (2, 2)
    assert np.allclose(counts, [[2.0, 1.0], [1.0, 1.0]])


def test_build_epoch_fit_dataset_contains_raw_and_summary_vars() -> None:
    pytest.importorskip("xarray")

    results = {
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102, 103]),
        "coef_ca1_unit_ids": np.array([101, 103]),
        "n_ripples": 8,
        "pseudo_r2_ripple_folds": np.array([[0.2, 0.4], [0.4, 0.6]], dtype=float),
        "mae_ripple_folds": np.array([[0.3, 0.7], [0.5, 0.9]], dtype=float),
        "devexp_ripple_folds": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "bits_per_spike_ripple_folds": np.array([[0.5, 0.7], [0.7, 0.9]], dtype=float),
        "pseudo_r2_ripple_shuff_folds": np.array(
            [
                [[0.0, 0.6], [0.2, 0.7]],
                [[0.2, 0.6], [0.2, 0.7]],
            ],
            dtype=float,
        ),
        "mae_ripple_shuff_folds": np.array(
            [
                [[0.6, 0.4], [0.5, 0.5]],
                [[0.6, 0.4], [0.5, 0.5]],
            ],
            dtype=float,
        ),
        "devexp_ripple_shuff_folds": np.array(
            [
                [[0.0, 0.3], [0.2, 0.4]],
                [[0.0, 0.3], [0.2, 0.4]],
            ],
            dtype=float,
        ),
        "bits_per_spike_ripple_shuff_folds": np.array(
            [
                [[0.1, 0.8], [0.2, 0.9]],
                [[0.1, 0.8], [0.2, 0.9]],
            ],
            dtype=float,
        ),
        "coef_ca1_full_all": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "coef_intercept_full_all": np.array([0.5, 0.6], dtype=float),
    }

    dataset = build_epoch_fit_dataset(
        results,
        animal_name="L14",
        date="20240611",
        epoch="01_s1",
        sources={"ripple_events": "pynapple"},
        fit_parameters={
            "n_splits": 2,
            "ridge_strength": 0.1,
            "ripple_selection_mode": "single",
            "n_ripples_before_selection": 10,
            "n_ripples_removed_by_selection": 2,
            "n_ripples_after_selection": 8,
        },
    )

    assert dataset.sizes["fold"] == 2
    assert dataset.sizes["shuffle"] == 2
    assert dataset.sizes["unit"] == 2
    assert dataset.sizes["coef_source_unit"] == 2
    assert np.array_equal(dataset.coords["unit"].values, [11, 12])
    assert np.array_equal(dataset["ca1_unit_id"].values, [101, 102, 103])
    assert np.array_equal(dataset["coef_ca1_unit_id"].values, [101, 103])
    assert dataset["pseudo_r2_ripple_folds"].dims == ("fold", "unit")
    assert dataset["pseudo_r2_ripple_shuff_folds"].dims == ("fold", "shuffle", "unit")
    assert dataset["coef_ca1_full_all"].dims == ("coef_source_unit", "unit")
    assert dataset["coef_intercept_full_all"].dims == ("unit",)
    assert np.allclose(dataset["coef_ca1_full_all"].values, [[0.1, 0.2], [0.3, 0.4]])
    assert np.allclose(dataset["coef_intercept_full_all"].values, [0.5, 0.6])
    assert np.allclose(dataset["ripple_pseudo_r2_mean"].values, [0.3, 0.5])
    assert np.allclose(dataset["ripple_mae_mean"].values, [0.4, 0.8])
    assert np.allclose(dataset["ripple_pseudo_r2_p_value"].values, [1.0 / 3.0, 1.0])
    assert np.allclose(dataset["ripple_mae_p_value"].values, [1.0 / 3.0, 1.0])
    assert np.allclose(dataset["ripple_devexp_p_value"].values, [2.0 / 3.0, 1.0])
    assert np.allclose(dataset["ripple_bits_per_spike_p_value"].values, [1.0 / 3.0, 1.0])
    assert "ll_ripple_folds" not in dataset.data_vars
    assert "pseudo_r2_pre" not in dataset.data_vars
    assert dataset.attrs["animal_name"] == "L14"
    assert dataset.attrs["epoch"] == "01_s1"
    assert dataset.attrs["model_direction"] == "ca1_to_v1"
    assert dataset.attrs["schema_version"] == "4"
    assert dataset.attrs["ripple_selection_mode"] == "single"
    assert dataset.attrs["n_ripples_before_selection"] == 10
    assert dataset.attrs["n_ripples_removed_by_selection"] == 2
    assert dataset.attrs["n_ripples_after_selection"] == 8
    assert dataset.attrs["coef_ca1_full_all_space"] == "preprocessed_predictor"
    assert json.loads(dataset.attrs["sources_json"]) == {"ripple_events": "pynapple"}
    assert json.loads(dataset.attrs["fit_parameters_json"]) == {
        "n_splits": 2,
        "ridge_strength": 0.1,
        "ripple_selection_mode": "single",
        "n_ripples_before_selection": 10,
        "n_ripples_removed_by_selection": 2,
        "n_ripples_after_selection": 8,
    }
    assert json.loads(dataset.attrs["coef_ca1_full_all_preprocess_json"]) == {
        "center": True,
        "clip_abs": 10.0,
        "divide_by_sqrt_n_features": True,
        "scale": True,
    }


def test_fit_ripple_glm_returns_full_fit_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePopulationGLM:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.coef_ = np.empty((0, 0), dtype=float)
            self.intercept_ = np.empty((0,), dtype=float)

        def fit(self, X, y) -> None:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            n_features = X.shape[1]
            n_units = y.shape[1]
            self.coef_ = np.arange(1, n_features * n_units + 1, dtype=float).reshape(
                n_features, n_units
            )
            self.intercept_ = np.linspace(0.25, 0.25 * n_units, n_units, dtype=float)

        def predict(self, X) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            return np.exp(X @ self.coef_ + self.intercept_[None, :])

    class FakeTs:
        def __init__(self, timestamps: list[float]) -> None:
            self.t = np.asarray(timestamps, dtype=float)

    class FakeTsGroup:
        def __init__(self, spikes_by_unit: dict[int, list[float]]) -> None:
            self._spikes_by_unit = {
                unit_id: FakeTs(timestamps) for unit_id, timestamps in spikes_by_unit.items()
            }

        def keys(self):
            return list(self._spikes_by_unit)

        def __getitem__(self, unit_id: int) -> FakeTs:
            return self._spikes_by_unit[unit_id]

    monkeypatch.setitem(
        sys.modules,
        "nemos",
        SimpleNamespace(
            __version__="0.2.6",
            glm=SimpleNamespace(PopulationGLM=FakePopulationGLM),
        ),
    )
    monkeypatch.setattr(ripple_glm_module, "_clear_jax_caches", lambda: None)

    spikes = {
        "ca1": FakeTsGroup(
            {
                101: [1.01, 3.01, 3.03, 4.01, 4.05],
                102: [2.01, 3.02],
                103: [],
            }
        ),
        "v1": FakeTsGroup(
            {
                11: [1.01, 3.01, 3.03, 4.01, 4.05],
                12: [2.01, 3.02],
            }
        ),
    }

    results = fit_ripple_glm_train_on_ripple(
        epoch="01_s1",
        spikes=spikes,
        epoch_interval=SimpleNamespace(),
        ripple_table=pd.DataFrame(
            {
                "start_time": [1.0, 2.0, 3.0, 4.0],
                "end_time": [1.2, 2.2, 3.2, 4.2],
            }
        ),
        n_splits=2,
        n_shuffles_ripple=0,
        min_spikes_per_ripple=0.1,
        min_ca1_spikes_per_ripple=0.0,
        ripple_window_s=0.2,
        ridge_strength=0.1,
        maxiter=20,
        tol=1e-4,
    )

    assert results["n_ripples"] == 4
    assert np.array_equal(results["ca1_unit_ids"], [101, 102, 103])
    assert np.array_equal(results["coef_ca1_unit_ids"], [101, 102])
    assert results["coef_ca1_full_all"].shape == (2, 2)
    assert results["coef_intercept_full_all"].shape == (2,)
    assert np.allclose(results["coef_ca1_full_all"], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(results["coef_intercept_full_all"], [0.25, 0.5])


def test_fit_ripple_glm_preserves_overlapping_fixed_windows_in_fit_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePopulationGLM:
        def __init__(self, **kwargs) -> None:
            self.coef_ = np.empty((0, 0), dtype=float)
            self.intercept_ = np.empty((0,), dtype=float)

        def fit(self, X, y) -> None:
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float)
            self.coef_ = np.ones((X.shape[1], y.shape[1]), dtype=float)
            self.intercept_ = np.zeros(y.shape[1], dtype=float)

        def predict(self, X) -> np.ndarray:
            X = np.asarray(X, dtype=float)
            return np.ones((X.shape[0], self.intercept_.size), dtype=float)

    class FakeTs:
        def __init__(self, timestamps: list[float]) -> None:
            self.t = np.asarray(timestamps, dtype=float)

    class FakeTsGroup:
        def __init__(self, spikes_by_unit: dict[int, list[float]]) -> None:
            self._spikes_by_unit = {
                unit_id: FakeTs(timestamps) for unit_id, timestamps in spikes_by_unit.items()
            }

        def keys(self):
            return list(self._spikes_by_unit)

        def __getitem__(self, unit_id: int) -> FakeTs:
            return self._spikes_by_unit[unit_id]

    monkeypatch.setitem(
        sys.modules,
        "nemos",
        SimpleNamespace(
            __version__="0.2.6",
            glm=SimpleNamespace(PopulationGLM=FakePopulationGLM),
        ),
    )
    monkeypatch.setattr(ripple_glm_module, "_clear_jax_caches", lambda: None)

    spikes = {
        "ca1": FakeTsGroup({101: [1.01, 1.06, 2.01, 2.07], 102: [1.03, 1.11, 2.03, 2.11]}),
        "v1": FakeTsGroup({11: [1.02, 1.07, 2.02, 2.07], 12: [1.08, 2.08]}),
    }

    results = fit_ripple_glm_train_on_ripple(
        epoch="02_r1",
        spikes=spikes,
        epoch_interval=SimpleNamespace(),
        ripple_table=pd.DataFrame(
            {
                "start_time": [1.0, 1.05, 2.0, 2.05],
                "end_time": [1.2, 1.25, 2.2, 2.25],
            }
        ),
        n_splits=2,
        n_shuffles_ripple=0,
        min_spikes_per_ripple=0.0,
        min_ca1_spikes_per_ripple=0.0,
        ripple_window_s=0.1,
        ridge_strength=0.1,
        maxiter=20,
        tol=1e-4,
    )

    assert results["n_ripples"] == 4
    assert results["pseudo_r2_ripple_folds"].shape == (2, 2)


def test_build_metric_figure_data_uses_ripple_shuffle_p_values() -> None:
    results = {
        "devexp_ripple_folds": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "devexp_ripple_shuff_folds": np.array(
            [
                [[0.0, 0.3], [0.2, 0.4]],
                [[0.0, 0.3], [0.2, 0.4]],
            ],
            dtype=float,
        ),
    }

    figure_data = build_metric_figure_data(results, metric_name="devexp")

    assert np.allclose(figure_data["ripple_values"], [0.2, 0.3])
    assert np.allclose(figure_data["ripple_p_value"], [2.0 / 3.0, 1.0])


def test_empirical_p_values_treat_near_equal_shuffle_means_as_ties() -> None:
    observed = np.array([0.30000000000000004, 0.4], dtype=float)
    null_samples = np.array(
        [
            [0.3, 0.2],
            [0.1, 0.4],
        ],
        dtype=float,
    )

    p_values = empirical_p_values(
        observed=observed,
        null_samples=null_samples,
        higher_is_better=True,
    )

    assert np.allclose(p_values, [2.0 / 3.0, 2.0 / 3.0])


def test_parse_arguments_rejects_removed_save_legacy_npz_flag(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ripple_glm.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--save-legacy-npz",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_parse_arguments_rejects_removed_pre_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ripple_glm.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--pre-window-s",
            "0.2",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


def test_save_epoch_figures_returns_one_combined_metric_path(monkeypatch, tmp_path) -> None:
    metric_calls: list[dict[str, object]] = []

    def fake_plot_epoch_metric_summary(**kwargs):
        out_path = kwargs["out_path"]
        metric_calls.append(kwargs)
        return out_path

    monkeypatch.setattr(
        "v1ca1.ripple.ripple_glm.plot_epoch_metric_summary",
        fake_plot_epoch_metric_summary,
    )

    results = {
        "pseudo_r2_ripple_folds": np.ones((2, 2), dtype=float),
        "pseudo_r2_ripple_shuff_folds": np.ones((2, 3, 2), dtype=float),
        "mae_ripple_folds": np.ones((2, 2), dtype=float),
        "mae_ripple_shuff_folds": np.ones((2, 3, 2), dtype=float),
        "devexp_ripple_folds": np.ones((2, 2), dtype=float),
        "devexp_ripple_shuff_folds": np.ones((2, 3, 2), dtype=float),
        "bits_per_spike_ripple_folds": np.ones((2, 2), dtype=float),
        "bits_per_spike_ripple_shuff_folds": np.ones((2, 3, 2), dtype=float),
    }

    figure_paths = save_epoch_figures(
        results=results,
        fig_dir=tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="01_s1",
        ripple_window_s=0.2,
        ripple_selection_suffix="single",
        ridge_strength=1e-1,
    )

    assert len(figure_paths) == 1
    assert len(metric_calls) == 1
    assert all("ll" not in path.name for path in figure_paths)
    assert figure_paths[0].name.endswith("samplewise_metrics_summary.png")
    assert "_samplewise_" in figure_paths[0].name
    metric_panels = metric_calls[0]["metric_panels"]
    assert len(metric_panels) == 4
    pseudo_r2_panel = next(panel for panel in metric_panels if panel["metric_label"] == "Pseudo R^2")
    assert np.allclose(pseudo_r2_panel["ripple_values"], [1.0, 1.0])
    assert np.allclose(
        pseudo_r2_panel["ripple_p_value"],
        np.array([1.0, 1.0], dtype=float),
    )


def test_get_epoch_skip_reason_handles_common_weak_epoch_cases() -> None:
    assert (
        get_epoch_skip_reason(n_ripples=3, n_splits=5, n_kept_v1_units=4)
        == "Not enough ripples for CV: n_ripples=3, n_splits=5"
    )
    assert (
        get_epoch_skip_reason(n_ripples=10, n_splits=5, n_kept_v1_units=0)
        == "No V1 units passed the minimum spikes-per-ripple threshold."
    )
    assert get_epoch_skip_reason(n_ripples=10, n_splits=5, n_kept_v1_units=2) is None
