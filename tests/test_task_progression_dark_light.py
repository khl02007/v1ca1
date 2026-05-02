from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


MODULE_NAME = "v1ca1.task_progression.dark_light_glm"


def _reload_dark_light_module(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
):
    monkeypatch.setattr(sys, "argv", list(argv))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)



def _fake_selected_result(*, model_name: str = "visual") -> dict[str, object]:
    n_units = 2
    result = {
        "model_name": model_name,
        "unit_ids": np.asarray([11, 12], dtype=int),
        "has_speed": False,
        "bin_size_s": 0.02,
        "spatial_bin_size_cm": 4.0,
        "trajectory_length_cm": 12.0,
        "n_splines": 3,
        "spline_order": 4,
        "ridge": 0.1,
        "pos_bounds": np.asarray([0.0, 1.0], dtype=float),
        "grid_tp": np.linspace(0.0, 1.0, 5),
        "dark_hz_grid": np.full((5, n_units), 1.0, dtype=float),
        "light_hz_grid": np.full((5, n_units), 2.0, dtype=float),
        "train_light_hz_grid": np.full((5, n_units), 2.0, dtype=float),
        "coef_intercept": np.asarray([0.1, 0.2], dtype=float),
        "coef_light_offset": np.asarray([0.3, 0.4], dtype=float),
        "coef_place_dark": np.full((3, n_units), 0.5, dtype=float),
        "speed_feature_mode": "none",
        "n_speed_features": 0,
        "speed_basis": "none",
        "speed_spline_order": np.nan,
        "speed_basis_bounds": np.asarray([np.nan, np.nan], dtype=float),
        "speed_reference_value": np.nan,
        "speed_mean": np.nan,
        "speed_std": np.nan,
        "coef_speed": np.full((n_units,), np.nan, dtype=float),
        "coef_speed_basis": np.full((0, n_units), np.nan, dtype=float),
        "segment_edges": np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
    }
    for suffix in ("combined", "dark", "light"):
        result[f"ll_sum_cv_{suffix}"] = np.asarray([1.0, 2.0], dtype=float)
        result[f"null_ll_sum_cv_{suffix}"] = np.asarray([0.5, 1.0], dtype=float)
        result[f"spike_sum_cv_{suffix}"] = np.asarray([10.0, 20.0], dtype=float)
        result[f"ll_bits_per_spike_cv_{suffix}"] = np.asarray([0.1, 0.2], dtype=float)
    if model_name == "visual":
        result["coef_place_light"] = np.full((3, n_units), 0.6, dtype=float)
    else:
        raise ValueError(model_name)
    return result


def test_cuda_visible_devices_is_preparsed_before_normal_argparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    module = _reload_dark_light_module(
        monkeypatch,
        [
            "dark_light_glm.py",
            "--cuda-visible-devices",
            "0,1",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-epoch",
            "08_r4",
        ],
    )

    assert module._CUDA_VISIBLE_DEVICES_CLI == "0,1"
    assert "--cuda-visible-devices" not in sys.argv
    assert "0,1" not in sys.argv
    assert module.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"

    args = module.parse_arguments()

    assert args.cuda_visible_devices == "0,1"
    assert args.animal_name == "L14"
    assert args.date == "20240611"
    assert args.dark_epoch == "08_r4"
    assert args.v1_min_light_fr_hz == pytest.approx(
        module.DEFAULT_REGION_FR_THRESHOLDS["v1"]
    )
    assert args.ca1_min_light_fr_hz == pytest.approx(
        module.DEFAULT_REGION_FR_THRESHOLDS["ca1"]
    )
    assert args.spatial_bin_sizes_cm == list(module.DEFAULT_SPATIAL_BIN_SIZES_CM)


def test_parse_arguments_requires_dark_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        [
            "dark_light_glm.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_arguments()


def test_build_position_basis_configs_derive_n_splines_from_spatial_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    monkeypatch.setattr(module, "get_wtrack_total_length", lambda animal_name: 11.0)

    configs = module.build_position_basis_configs(
        animal_name="L14",
        spatial_bin_sizes_cm=[2.0, 4.0],
        spline_order=3,
    )

    assert [config["spatial_bin_size_cm"] for config in configs] == [2.0, 4.0]
    assert [config["n_splines"] for config in configs] == [6, 3]
    assert configs[0]["trajectory_length_cm"] == pytest.approx(11.0)


def test_build_train_epoch_fr_mask_requires_dark_and_light_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    masks = module.build_train_epoch_fr_mask(
        dark_epoch_rates=np.asarray([0.6, 0.6, 0.4, np.nan], dtype=float),
        light_epoch_rates=np.asarray([0.7, 0.2, 0.8, 0.8], dtype=float),
        min_dark_fr_hz=0.5,
        min_light_fr_hz=0.5,
    )

    assert np.array_equal(masks["dark"], np.asarray([True, True, False, False]))
    assert np.array_equal(masks["light"], np.asarray([True, False, True, True]))
    assert np.array_equal(masks["combined"], np.asarray([True, False, False, False]))


def test_format_speed_outputs_linear_preserves_scalar_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    transform = module._fit_speed_feature_transform(
        np.asarray([1.0, 2.0, 4.0]),
        speed_feature_mode="linear",
    )

    outputs = module._format_speed_outputs(
        transform=transform,
        coef_speed_basis_base=np.asarray([[0.5, -0.25]]),
        coef_speed_basis_full=np.asarray([[0.75, -0.5]]),
        n_units=2,
    )

    assert outputs["speed_feature_mode"] == "linear"
    assert outputs["n_speed_features"] == 1
    assert np.allclose(outputs["coef_speed_base_all"], [0.5, -0.25])
    assert np.allclose(outputs["coef_speed_full_all"], [0.75, -0.5])
    assert np.isfinite(outputs["speed_mean"])
    assert np.isfinite(outputs["speed_std"])
    assert outputs["coef_speed_basis_base_all"].shape == (1, 2)


def test_format_speed_outputs_linear_preserves_full_refit_transform_without_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    transform = module._fit_speed_feature_transform(
        np.asarray([1.0, 2.0, 4.0]),
        speed_feature_mode="linear",
    )

    outputs = module._format_speed_outputs(
        transform=transform,
        coef_speed_basis_base=None,
        coef_speed_basis_full=np.asarray([[0.75, -0.5]]),
        n_units=2,
    )

    assert outputs["speed_feature_mode"] == "linear"
    assert np.isnan(outputs["coef_speed_base_all"]).all()
    assert np.allclose(outputs["coef_speed_full_all"], [0.75, -0.5])
    assert np.isfinite(outputs["speed_mean"])
    assert np.isfinite(outputs["speed_std"])
    assert outputs["speed_std"] > 0.0


def test_select_light_dark_pairs_defaults_to_all_nondark_valid_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    pairs = module.select_light_dark_pairs(
        ["02_r1", "04_r2", "06_r3", "08_r4", "10_r5"],
        dark_epoch="08_r4",
        light_epochs=None,
    )

    assert pairs == [
        ("02_r1", "08_r4"),
        ("04_r2", "08_r4"),
        ("06_r3", "08_r4"),
        ("10_r5", "08_r4"),
    ]


def test_select_light_dark_pairs_deduplicates_explicit_light_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    pairs = module.select_light_dark_pairs(
        ["02_r1", "04_r2", "06_r3", "08_r4"],
        dark_epoch="08_r4",
        light_epochs=["02_r1", "04_r2", "02_r1"],
    )

    assert pairs == [
        ("02_r1", "08_r4"),
        ("04_r2", "08_r4"),
    ]


def test_select_light_dark_pairs_rejects_light_epochs_matching_dark_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    with pytest.raises(ValueError, match="must differ"):
        module.select_light_dark_pairs(
            ["02_r1", "04_r2", "08_r4"],
            dark_epoch="08_r4",
            light_epochs=["02_r1", "08_r4"],
        )


def test_normalize_model_names_adds_visual_and_maps_swap_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    names, messages = module.normalize_model_names(
        ["segment_bump_gain", "task_dense_gain"]
    )

    assert names == ["visual", "task_segment_bump", "task_dense_gain"]
    assert any("segment_bump_gain" in message for message in messages)
    assert any("visual" in message for message in messages)


def test_normalize_model_names_rejects_unknown_legacy_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    with pytest.raises(ValueError, match="Unknown model"):
        module.normalize_model_names(["additive_light"])


def test_visual_selection_tiebreaks_bin_spatial_bin_and_ridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    records = [
        {
            "model_name": "visual",
            "bin_size_s": 0.02,
            "spatial_bin_size_cm": 4.0,
            "trajectory_length_cm": 100.0,
            "n_splines": 25,
            "ridge": 0.1,
            "score_median": 0.2,
            "score_mean": 0.2,
            "n_finite": 8,
        },
        {
            "model_name": "visual",
            "bin_size_s": 0.05,
            "spatial_bin_size_cm": 2.0,
            "trajectory_length_cm": 100.0,
            "n_splines": 50,
            "ridge": 0.001,
            "score_median": 0.2 + 0.5 * module.HYPERPARAMETER_TIE_ATOL,
            "score_mean": 0.2,
            "n_finite": 8,
        },
        {
            "model_name": "visual",
            "bin_size_s": 0.05,
            "spatial_bin_size_cm": 8.0,
            "trajectory_length_cm": 100.0,
            "n_splines": 13,
            "ridge": 0.001,
            "score_median": 0.2 + 0.5 * module.HYPERPARAMETER_TIE_ATOL,
            "score_mean": 0.2,
            "n_finite": 8,
        },
        {
            "model_name": "visual",
            "bin_size_s": 0.05,
            "spatial_bin_size_cm": 8.0,
            "trajectory_length_cm": 100.0,
            "n_splines": 13,
            "ridge": 0.01,
            "score_median": 0.2 + 0.5 * module.HYPERPARAMETER_TIE_ATOL,
            "score_mean": 0.2,
            "n_finite": 8,
        },
    ]

    selected = module.choose_visual_shared_hyperparameters(records)

    assert selected["bin_size_s"] == pytest.approx(0.05)
    assert selected["spatial_bin_size_cm"] == pytest.approx(8.0)
    assert selected["n_splines"] == 13
    assert selected["visual_ridge"] == pytest.approx(0.01)


def test_build_selected_candidate_dataset_uses_swap_names_and_writes_netcdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    results_by_traj = {
        trajectory: {0.1: _fake_selected_result(model_name="visual")}
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_selected_candidate_dataset(
        model_name="visual",
        results_by_traj=results_by_traj,
        ridge_values=[0.1],
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        dark_movement_firing_rates=np.asarray([1.0, 2.0]),
        light_movement_firing_rates=np.asarray([1.5, 2.5]),
        min_dark_firing_rate_hz=0.5,
        min_light_firing_rate_hz=0.5,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={"selection_metric": module.SELECTION_METRIC},
    )

    assert dataset.attrs["fit_stage"] == "candidate"
    assert dataset.attrs["schema_version"] == "5"
    assert dataset.attrs["spatial_bin_size_cm"] == pytest.approx(4.0)
    assert dataset.attrs["trajectory_length_cm"] == pytest.approx(12.0)
    assert dataset.attrs["n_splines"] == 3
    assert "coef_light_offset" in dataset
    assert "coef_place_dark" in dataset
    assert "coef_place_light" in dataset
    assert "train_light_hz_grid" in dataset
    assert module.SELECTION_METRIC in dataset
    assert "ll_base_sum_cv" not in dataset

    output_path = tmp_path / "candidate.nc"
    dataset.to_netcdf(output_path)
    assert output_path.exists()

    selected = module.build_selected_model_dataset(
        dataset,
        selected_ridge=0.1,
        selection_score=0.2,
        shared_selection={
            "bin_size_s": 0.02,
            "spatial_bin_size_cm": 4.0,
            "n_splines": 3,
            "visual_ridge": 0.1,
            "score_median": 0.2,
        },
    )

    assert selected.attrs["fit_stage"] == "selected"
    assert selected.attrs["selected_spatial_bin_size_cm"] == pytest.approx(4.0)
    assert selected.attrs["selected_n_splines"] == 3
    assert "ridge" not in selected.dims


def test_build_selection_summary_dataset_uses_spatial_bin_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    records = [
        {
            "model_name": "visual",
            "bin_size_s": 0.02,
            "spatial_bin_size_cm": 4.0,
            "trajectory_length_cm": 12.0,
            "n_splines": 3,
            "ridge": 0.1,
            "score_median": 0.2,
            "score_mean": 0.2,
            "n_finite": 8,
        }
    ]
    position_basis_configs = [
        {
            "spatial_bin_size_cm": 4.0,
            "trajectory_length_cm": 12.0,
            "n_splines": 3,
        },
        {
            "spatial_bin_size_cm": 8.0,
            "trajectory_length_cm": 12.0,
            "n_splines": 2,
        },
    ]

    dataset = module.build_selection_summary_dataset(
        selection_records=records,
        model_names=["visual"],
        bin_sizes_s=[0.02],
        position_basis_configs=position_basis_configs,
        ridge_values=[0.1],
        selected_by_model={"visual": records[0]},
        shared_selection={
            "bin_size_s": 0.02,
            "spatial_bin_size_cm": 4.0,
            "n_splines": 3,
            "visual_ridge": 0.1,
            "score_median": 0.2,
        },
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        fit_parameters={},
    )

    assert dataset.attrs["schema_version"] == "5"
    assert dataset.attrs["selected_spatial_bin_size_cm"] == pytest.approx(4.0)
    assert "spatial_bin_size_cm" in dataset.dims
    assert "n_splines_by_spatial_bin_size" in dataset
    assert np.array_equal(
        dataset["n_splines_by_spatial_bin_size"].values,
        np.asarray([3, 2], dtype=int),
    )


def test_as_interval_set_accepts_intervalset_and_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    class _FakeIntervalSet:
        pass

    class _FakeWrapper:
        def __init__(self, time_support):
            self.time_support = time_support

    interval = _FakeIntervalSet()

    assert module._as_interval_set(interval) is interval
    assert module._as_interval_set(_FakeWrapper(interval)) is interval


def test_make_speed_design_train_test_uses_training_bounds_for_bspline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    if module.BSplineEval is None:
        pytest.skip("nemos is required for bspline speed-feature tests")

    values = np.asarray([0.0, 1.0, 2.0, 100.0], dtype=float)
    train_idx = np.asarray([0, 1, 2], dtype=int)
    test_idx = np.asarray([3], dtype=int)

    train_design, test_design, transform = module._make_speed_design_train_test(
        values,
        train_idx,
        test_idx,
        speed_feature_mode="bspline",
        n_splines_speed=4,
        spline_order_speed=3,
    )

    assert np.allclose(transform["bounds"], [0.0, 2.0])
    assert train_design.shape == (3, 4)
    assert test_design.shape == (1, 4)
    assert np.all(np.isfinite(test_design))


def test_format_speed_outputs_bspline_stores_basis_coefficients_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    transform = {
        "mode": "bspline",
        "basis": "bspline",
        "n_features": 4,
        "spline_order": 3,
        "bounds": np.asarray([0.0, 2.0], dtype=float),
        "reference_value": 1.0,
        "mean": np.nan,
        "std": np.nan,
    }

    outputs = module._format_speed_outputs(
        transform=transform,
        coef_speed_basis_base=np.arange(8, dtype=float).reshape(4, 2),
        coef_speed_basis_full=np.arange(8, 16, dtype=float).reshape(4, 2),
        n_units=2,
    )

    assert outputs["speed_feature_mode"] == "bspline"
    assert outputs["speed_basis"] == "bspline"
    assert outputs["n_speed_features"] == 4
    assert np.isnan(outputs["coef_speed_base_all"]).all()
    assert np.isnan(outputs["coef_speed_full_all"]).all()
    assert outputs["coef_speed_basis_base_all"].shape == (4, 2)
    assert outputs["coef_speed_basis_full_all"].shape == (4, 2)
