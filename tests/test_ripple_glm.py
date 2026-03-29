from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import v1ca1.ripple.ripple_glm as ripple_glm_module
from v1ca1.ripple.ripple_glm import (
    _format_nemos_solver_selection_message,
    _resolve_nemos_population_glm_solver,
    build_epoch_fit_dataset,
    build_metric_figure_data,
    empirical_p_values,
    get_epoch_skip_reason,
    load_ripple_tables,
    load_ripple_tables_from_parquet_output,
    load_ripple_tables_from_interval_output,
    load_ripple_tables_from_legacy_pickle,
    make_preripple_ep,
    parse_arguments,
    save_epoch_figures,
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


def test_make_preripple_ep_clips_to_epoch_bounds() -> None:
    nap = pytest.importorskip("pynapple")

    ripple_ep = nap.IntervalSet(start=[0.25], end=[0.35], time_units="s")
    epoch_ep = nap.IntervalSet(start=[0.0], end=[2.0], time_units="s")

    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=epoch_ep,
        window_s=0.5,
        buffer_s=0.1,
        exclude_ripples=False,
    )

    assert np.allclose(pre_ep.start, [0.0])
    assert np.allclose(pre_ep.end, [0.15])


def test_make_preripple_ep_excludes_nearby_ripples() -> None:
    nap = pytest.importorskip("pynapple")

    ripple_ep = nap.IntervalSet(
        start=[1.0, 1.15],
        end=[1.1, 1.2],
        time_units="s",
    )
    epoch_ep = nap.IntervalSet(start=[0.0], end=[3.0], time_units="s")

    pre_ep = make_preripple_ep(
        ripple_ep=ripple_ep,
        epoch_ep=epoch_ep,
        window_s=0.2,
        buffer_s=0.0,
        exclude_ripples=True,
        exclude_ripple_guard_s=0.1,
    )

    assert np.allclose(pre_ep.start, [0.8])
    assert np.allclose(pre_ep.end, [0.9])


def test_build_epoch_fit_dataset_contains_raw_and_summary_vars() -> None:
    pytest.importorskip("xarray")

    results = {
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102, 103]),
        "n_ripples": 8,
        "n_pre": 6,
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
        "pseudo_r2_pre": np.array([0.05, 0.06], dtype=float),
        "mae_pre": np.array([0.4, 0.5], dtype=float),
        "devexp_pre": np.array([0.02, 0.03], dtype=float),
        "bits_per_spike_pre": np.array([0.15, 0.25], dtype=float),
    }

    dataset = build_epoch_fit_dataset(
        results,
        animal_name="L14",
        date="20240611",
        epoch="01_s1",
        sources={"ripple_events": "pynapple"},
        fit_parameters={"n_splits": 2, "ridge_strength": 0.1},
    )

    assert dataset.sizes["fold"] == 2
    assert dataset.sizes["shuffle"] == 2
    assert dataset.sizes["unit"] == 2
    assert np.array_equal(dataset.coords["unit"].values, [11, 12])
    assert np.array_equal(dataset["ca1_unit_id"].values, [101, 102, 103])
    assert dataset["pseudo_r2_ripple_folds"].dims == ("fold", "unit")
    assert dataset["pseudo_r2_ripple_shuff_folds"].dims == ("fold", "shuffle", "unit")
    assert dataset["pseudo_r2_pre"].dims == ("unit",)
    assert np.allclose(dataset["ripple_pseudo_r2_mean"].values, [0.3, 0.5])
    assert np.allclose(dataset["ripple_mae_mean"].values, [0.4, 0.8])
    assert np.allclose(dataset["bits_per_spike_pre"].values, [0.15, 0.25])
    assert np.allclose(dataset["ripple_pseudo_r2_p_value"].values, [1.0 / 3.0, 1.0])
    assert np.allclose(dataset["ripple_mae_p_value"].values, [1.0 / 3.0, 1.0])
    assert np.allclose(dataset["ripple_devexp_p_value"].values, [2.0 / 3.0, 1.0])
    assert np.allclose(dataset["ripple_bits_per_spike_p_value"].values, [1.0 / 3.0, 1.0])
    assert "ll_ripple_folds" not in dataset.data_vars
    assert "ll_pre" not in dataset.data_vars
    assert dataset.attrs["animal_name"] == "L14"
    assert dataset.attrs["epoch"] == "01_s1"
    assert dataset.attrs["model_direction"] == "ca1_to_v1"
    assert json.loads(dataset.attrs["sources_json"]) == {"ripple_events": "pynapple"}
    assert json.loads(dataset.attrs["fit_parameters_json"]) == {
        "n_splits": 2,
        "ridge_strength": 0.1,
    }


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
        "devexp_pre": np.array([0.05, 0.06], dtype=float),
    }

    figure_data = build_metric_figure_data(results, metric_name="devexp")

    assert np.allclose(figure_data["ripple_values"], [0.2, 0.3])
    assert np.allclose(figure_data["pre_values"], [0.05, 0.06])
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


def test_save_epoch_figures_returns_four_metric_paths(monkeypatch, tmp_path) -> None:
    metric_calls: list[dict[str, object]] = []

    def fake_plot_metric_summary(**kwargs):
        out_path = kwargs["out_path"]
        metric_calls.append(kwargs)
        return out_path

    monkeypatch.setattr(
        "v1ca1.ripple.ripple_glm.plot_metric_summary",
        fake_plot_metric_summary,
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
        "pseudo_r2_pre": np.ones(2, dtype=float),
        "mae_pre": np.ones(2, dtype=float),
        "devexp_pre": np.ones(2, dtype=float),
        "bits_per_spike_pre": np.ones(2, dtype=float),
    }

    figure_paths = save_epoch_figures(
        results=results,
        fig_dir=tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="01_s1",
        ripple_window_s=0.2,
        ridge_strength=1e-1,
    )

    assert len(figure_paths) == 4
    assert len(metric_calls) == 4
    assert all("ll" not in path.name for path in figure_paths)
    assert any(path.name.endswith("devexp_summary.png") for path in figure_paths)
    assert any(path.name.endswith("bits_per_spike_summary.png") for path in figure_paths)
    pseudo_r2_call = next(call for call in metric_calls if call["metric_label"] == "Pseudo R^2")
    assert np.allclose(pseudo_r2_call["ripple_values"], [1.0, 1.0])
    assert np.allclose(pseudo_r2_call["pre_values"], [1.0, 1.0])
    assert np.allclose(
        pseudo_r2_call["ripple_p_value"],
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
