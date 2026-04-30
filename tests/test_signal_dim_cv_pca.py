from __future__ import annotations

import numpy as np
import pytest

from v1ca1.signal_dim import cv_pca


def test_classify_and_select_units_supports_requested_filter_modes() -> None:
    unit_ids = np.asarray([11, 12, 13, 14, 15])
    dark_fr = np.asarray([1.0, 1.0, 0.0, 1.0, 1.0])
    light_fr = np.asarray([1.0, 0.0, 1.0, 1.0, 1.0])
    dark_sd = np.asarray([1.0, 1.0, 0.0, 1.0, 1e-8])
    light_sd = np.asarray([1.0, 0.0, 1.0, 1e-8, 1.0])

    shared = cv_pca.classify_and_select_units(
        unit_ids=unit_ids,
        dark_firing_rate_hz=dark_fr,
        light_firing_rate_hz=light_fr,
        dark_condition_sd_hz=dark_sd,
        light_condition_sd_hz=light_sd,
        min_firing_rate_hz=0.5,
        min_condition_sd_hz=0.1,
        unit_filter_mode="shared-active",
    )
    dark = cv_pca.classify_and_select_units(
        unit_ids=unit_ids,
        dark_firing_rate_hz=dark_fr,
        light_firing_rate_hz=light_fr,
        dark_condition_sd_hz=dark_sd,
        light_condition_sd_hz=light_sd,
        min_firing_rate_hz=0.5,
        min_condition_sd_hz=0.1,
        unit_filter_mode="dark-active",
    )
    union = cv_pca.classify_and_select_units(
        unit_ids=unit_ids,
        dark_firing_rate_hz=dark_fr,
        light_firing_rate_hz=light_fr,
        dark_condition_sd_hz=dark_sd,
        light_condition_sd_hz=light_sd,
        min_firing_rate_hz=0.5,
        min_condition_sd_hz=0.1,
        unit_filter_mode="union-active",
    )

    assert shared.keep_mask.tolist() == [True, False, False, False, False]
    assert dark.keep_mask.tolist() == [True, True, False, True, False]
    assert union.keep_mask.tolist() == [True, True, True, True, True]
    assert shared.unit_classes.tolist() == [
        "shared_active",
        "dark_only",
        "light_only",
        "dark_only",
        "light_only",
    ]


def _make_grouped_tensor(scale: float = 1.0) -> np.ndarray:
    conditions = np.linspace(-1.0, 1.0, 6)
    base = np.column_stack(
        [
            conditions,
            0.5 * conditions,
            conditions**2 - np.mean(conditions**2),
        ]
    )
    groups = []
    for group_index in range(4):
        noise = 0.02 * (group_index - 1.5)
        groups.append(scale * base + noise)
    return np.stack(groups, axis=0)


def test_compute_cv_pca_metrics_uses_diagonal_within_and_all_cross_projections() -> None:
    dark = _make_grouped_tensor(scale=1.0)
    light = _make_grouped_tensor(scale=1.2)

    metrics = cv_pca.compute_cv_pca_metrics(
        {"dark": dark, "light": light},
        unit_classes=np.asarray(["shared_active", "shared_active", "shared_active"]),
        normalization="zscore",
        min_scale=1e-9,
    )

    dark_index = cv_pca.CONDITIONS.index("dark")
    light_index = cv_pca.CONDITIONS.index("light")
    valid = metrics["valid_projection"]

    assert valid[dark_index, dark_index].sum() == 4
    assert np.all(np.diag(valid[dark_index, dark_index]))
    assert valid[dark_index, light_index].sum() == 16
    assert valid[light_index, dark_index].sum() == 16
    assert metrics["within_cv_spectrum_signed"].shape == (2, 3)
    assert metrics["residualized_light_cv_spectrum_signed"].shape == (3, 3)
    assert metrics["residualized_light_cv_spectrum_signed_by_fold"].shape == (3, 4, 3)
    assert np.isfinite(metrics["within_cv_participation_ratio"]).all()
    assert np.isfinite(metrics["residualized_light_cv_participation_ratio"]).any()
    assert (
        metrics["residual_fraction"][dark_index, light_index, 0, 0, -1]
        < metrics["residual_fraction"][dark_index, light_index, 0, 0, 0]
    )


def test_build_result_dataset_and_summary_table_keep_scalar_outputs(tmp_path) -> None:
    pytest.importorskip("xarray")
    dark = _make_grouped_tensor(scale=1.0)
    light = _make_grouped_tensor(scale=1.2)
    unit_selection = cv_pca.UnitSelection(
        keep_mask=np.asarray([True, True, True]),
        unit_classes=np.asarray(["shared_active", "shared_active", "shared_active"]),
        dark_active=np.asarray([True, True, True]),
        light_active=np.asarray([True, True, True]),
        dark_modulated=np.asarray([True, True, True]),
        light_modulated=np.asarray([True, True, True]),
    )
    pair_tensors = cv_pca.PairTuningTensors(
        dark=dark,
        light=light,
        unit_ids=np.asarray([11, 12, 13]),
        unit_classes=np.asarray(["shared_active", "shared_active", "shared_active"]),
        dark_firing_rate_hz=np.asarray([1.0, 1.1, 1.2]),
        light_firing_rate_hz=np.asarray([1.2, 1.3, 1.4]),
        dark_condition_sd_hz=np.asarray([0.4, 0.5, 0.6]),
        light_condition_sd_hz=np.asarray([0.5, 0.6, 0.7]),
        condition_trajectory=np.asarray(["center_to_left"] * dark.shape[1]),
        condition_bin_center=np.arange(dark.shape[1], dtype=float),
        condition_bin_index=np.arange(dark.shape[1], dtype=int),
        n_valid_bins_by_trajectory={
            "center_to_left": dark.shape[1],
            "center_to_right": 0,
            "left_to_center": 0,
            "right_to_center": 0,
        },
        unit_selection=unit_selection,
    )
    settings = {
        "normalization": "zscore",
        "unit_filter_mode": "shared-active",
        "min_firing_rate_hz": 0.5,
        "min_condition_sd_hz": 1e-6,
        "bin_size_cm": 4.0,
        "min_occupancy_s": 0.01,
        "n_groups": 4,
    }
    metrics = cv_pca.compute_cv_pca_metrics(
        {"dark": dark, "light": light},
        unit_classes=pair_tensors.unit_classes,
        normalization="zscore",
        min_scale=1e-9,
    )

    dataset = cv_pca.build_result_dataset(
        pair_tensors=pair_tensors,
        metrics=metrics,
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        settings=settings,
    )
    table = cv_pca.build_summary_table(dataset, settings=settings)
    residualized_table = cv_pca.build_residualized_light_summary_table(
        dataset,
        settings=settings,
    )
    normalized_capture_table = cv_pca.build_normalized_capture_summary_table(
        dataset,
        settings=settings,
    )
    spectrum_table = cv_pca.build_within_cv_spectrum_table(
        dataset,
        repeat_index=0,
        random_seed=47,
    )
    dataset.to_netcdf(tmp_path / "cv_pca.nc")

    assert dataset.attrs["unit_class_counts_kept_json"]
    assert "residualized_light_cv_participation_ratio" in dataset
    assert dataset["residualized_light_cv_participation_ratio"].sizes[
        "residualized_component"
    ] == 3
    assert table.shape[0] == 4
    assert set(table["projection_direction"]) == {
        "dark_to_dark",
        "dark_to_light",
        "light_to_dark",
        "light_to_light",
    }
    assert np.isfinite(table["source_cv_participation_ratio"]).all()
    assert "residualized_light_cv_participation_ratio_at_dark_pr" not in table.columns
    assert residualized_table.shape[0] == 3
    assert "residualized_light_cv_participation_ratio" in residualized_table.columns
    assert normalized_capture_table.shape[0] == 6
    assert "normalized_capture_fraction" in normalized_capture_table.columns
    assert spectrum_table.shape[0] == 2 * dark.shape[2]
    assert set(spectrum_table["condition"]) == {"dark", "light"}
    assert "within_cv_spectrum_positive" in spectrum_table.columns
    assert spectrum_table["repeat_index"].unique().tolist() == [0]


def test_residual_power_trajectory_figure_uses_saved_residual_matrices(tmp_path) -> None:
    pytest.importorskip("xarray")
    pytest.importorskip("matplotlib")
    dark = _make_grouped_tensor(scale=1.0)
    light = _make_grouped_tensor(scale=1.2)
    unit_selection = cv_pca.UnitSelection(
        keep_mask=np.asarray([True, True, True]),
        unit_classes=np.asarray(["shared_active", "shared_active", "shared_active"]),
        dark_active=np.asarray([True, True, True]),
        light_active=np.asarray([True, True, True]),
        dark_modulated=np.asarray([True, True, True]),
        light_modulated=np.asarray([True, True, True]),
    )
    pair_tensors = cv_pca.PairTuningTensors(
        dark=dark,
        light=light,
        unit_ids=np.asarray([11, 12, 13]),
        unit_classes=np.asarray(["shared_active", "shared_active", "shared_active"]),
        dark_firing_rate_hz=np.asarray([1.0, 1.1, 1.2]),
        light_firing_rate_hz=np.asarray([1.2, 1.3, 1.4]),
        dark_condition_sd_hz=np.asarray([0.4, 0.5, 0.6]),
        light_condition_sd_hz=np.asarray([0.5, 0.6, 0.7]),
        condition_trajectory=np.asarray(
            [
                "center_to_left",
                "center_to_left",
                "center_to_right",
                "center_to_right",
                "left_to_center",
                "right_to_center",
            ]
        ),
        condition_bin_center=np.asarray([2.0, 6.0, 2.0, 6.0, 2.0, 2.0]),
        condition_bin_index=np.asarray([0, 1, 0, 1, 0, 0]),
        n_valid_bins_by_trajectory={
            "center_to_left": 2,
            "center_to_right": 2,
            "left_to_center": 1,
            "right_to_center": 1,
        },
        unit_selection=unit_selection,
    )
    settings = {
        "normalization": "zscore",
        "unit_filter_mode": "shared-active",
        "min_firing_rate_hz": 0.5,
        "min_condition_sd_hz": 1e-6,
        "bin_size_cm": 4.0,
        "min_occupancy_s": 0.01,
        "n_groups": 4,
    }
    metrics = cv_pca.compute_cv_pca_metrics(
        {"dark": dark, "light": light},
        unit_classes=pair_tensors.unit_classes,
        normalization="zscore",
        min_scale=1e-9,
        save_residual_matrices=True,
    )

    dataset = cv_pca.build_result_dataset(
        pair_tensors=pair_tensors,
        metrics=metrics,
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        settings=settings,
    )
    component_count = cv_pca._component_count_for_condition(
        dataset,
        condition="dark",
        component_mode="pr",
    )
    power = cv_pca._mean_residual_power_by_condition_unit(
        dataset,
        source="dark",
        target="light",
        component_count=component_count,
    )
    saved = cv_pca.plot_summary_figures(
        dataset,
        fig_dir=tmp_path,
        stem="test_cv_pca",
        residual_power_component="pr",
    )

    assert component_count >= 1
    assert power.shape == (dark.shape[1], dark.shape[2])
    assert np.isfinite(power).all()
    assert any(path.name.endswith("_residual_power_by_trajectory.png") for path in saved)
    assert any(
        path.name.endswith("_residualized_light_dimensionality.png") for path in saved
    )
    assert any(path.name.endswith("_normalized_capture_summary.png") for path in saved)
    assert (tmp_path / "test_cv_pca_residual_power_by_trajectory.png").exists()
    assert (tmp_path / "test_cv_pca_residualized_light_dimensionality.png").exists()
    assert (tmp_path / "test_cv_pca_normalized_capture_summary.png").exists()


def test_plot_repeat_dimensionality_summary_writes_errorbar_figure(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    repeat_table = pd.DataFrame(
        {
            "repeat_index": [0, 0, 1, 1],
            "random_seed": [47, 47, 48, 48],
            "source_condition": ["dark", "light", "dark", "light"],
            "target_condition": ["dark", "light", "dark", "light"],
            "source_cv_participation_ratio": [12.0, 18.0, 13.0, 20.0],
            "source_n_components_80": [17, 24, 18, 25],
            "source_n_components_90": [29, 36, 30, 38],
        }
    )
    residualized_repeat_table = pd.DataFrame(
        {
            "repeat_index": [0, 1],
            "random_seed": [47, 48],
            "dark_epoch": ["08_r4", "08_r4"],
            "light_epoch": ["02_r1", "02_r1"],
            "residualized_component": ["pr", "pr"],
            "residualized_light_cv_participation_ratio": [8.0, 9.0],
            "residualized_light_cv_n_components_80": [11, 12],
            "residualized_light_cv_n_components_90": [17, 18],
        }
    )
    repeat_spectrum_table = pd.DataFrame(
        {
            "repeat_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "random_seed": [47, 47, 47, 47, 48, 48, 48, 48],
            "dark_epoch": ["08_r4"] * 8,
            "light_epoch": ["02_r1"] * 8,
            "condition": ["dark", "dark", "light", "light"] * 2,
            "component": [1, 2, 1, 2, 1, 2, 1, 2],
            "within_cv_spectrum_positive": [
                1.0,
                0.5,
                1.4,
                0.7,
                1.1,
                0.4,
                1.5,
                0.8,
            ],
            "within_cv_cumulative_shared_variance": [
                0.67,
                1.0,
                0.67,
                1.0,
                0.73,
                1.0,
                0.65,
                1.0,
            ],
        }
    )

    figure_path = cv_pca.plot_repeat_dimensionality_summary(
        repeat_table,
        fig_dir=tmp_path,
        stem="test_cv_pca",
        residualized_repeat_table=residualized_repeat_table,
    )
    spectrum_figure_path = cv_pca.plot_repeat_spectrum_summary(
        repeat_spectrum_table,
        fig_dir=tmp_path,
        stem="test_cv_pca",
    )

    assert figure_path == tmp_path / "test_cv_pca_within_cv_pr_repeats.png"
    assert figure_path.exists()
    assert spectrum_figure_path == (
        tmp_path / "test_cv_pca_within_cv_spectrum_repeats.png"
    )
    assert spectrum_figure_path.exists()
