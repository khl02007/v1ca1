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


def _fake_result(
    module,
    *,
    speed_outputs: dict[str, object],
    family_name: str = "dense_gain",
) -> dict[str, object]:
    n_units = 2
    result = {
        "unit_ids": np.asarray([11, 12], dtype=int),
        "has_speed": True,
        **speed_outputs,
        "bin_size_s": 0.02,
        "n_splines": 3,
        "spline_order": 4,
        "pos_bounds": np.asarray([0.0, 1.0], dtype=float),
        "spike_sum_cv": np.asarray([10.0, 11.0]),
        "ll_base_sum_cv": np.asarray([1.0, 2.0]),
        "ll_full_sum_cv": np.asarray([1.5, 2.5]),
        "dLL_sum_cv": np.asarray([0.5, 0.5]),
        "ll_base_per_spike_cv": np.asarray([0.1, 0.2]),
        "ll_full_per_spike_cv": np.asarray([0.15, 0.25]),
        "dll_per_spike_cv": np.asarray([0.05, 0.05]),
        "ll_base_bits_per_spike_cv": np.asarray([0.2, 0.3]),
        "ll_full_bits_per_spike_cv": np.asarray([0.25, 0.35]),
        "dll_bits_per_spike_cv": np.asarray([0.05, 0.05]),
        "coef_intercept_base_all": np.asarray([0.1, 0.2]),
        "coef_place_base_all": np.ones((3, n_units), dtype=float),
        "coef_light_base_all": np.asarray([0.3, 0.4]),
        "coef_place_x_light_base_all": np.full((2, n_units), np.nan),
        "coef_intercept_full_all": np.asarray([0.2, 0.3]),
        "coef_place_dark_full_all": np.full((3, n_units), 2.0, dtype=float),
        "coef_light_full_all": np.asarray([0.4, 0.5]),
        "coef_add_intercept_full_all": np.full((n_units,), np.nan),
        "coef_add_place_full_all": np.full((0, n_units), np.nan),
        "grid_tp": np.linspace(0.0, 1.0, 5),
        "dark_hz_grid": np.full((5, n_units), 1.0, dtype=float),
        "light_hz_grid": np.full((5, n_units), 2.0, dtype=float),
    }
    if family_name == "dense_gain":
        result["coef_light_gain_spline_full_all"] = np.full((2, n_units), 0.25)
    elif family_name == "independent_light_field":
        result["coef_place_light_full_all"] = np.full((3, n_units), 0.75)
    elif family_name == "segment_scalar_gain":
        result["coef_segment_scalar_gain_full_all"] = np.full((3, n_units), 0.25)
        result["segment_edges"] = np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float)
        result["gain_basis"] = "segment_scalar"
        result["n_splines_gain"] = 3
    else:
        raise ValueError(f"Unsupported fake family_name={family_name!r}")
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


def test_select_light_dark_pairs_defaults_to_all_nondark_selected_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    pairs = module.select_light_dark_pairs(
        ["02_r1", "04_r2", "06_r3", "08_r4", "10_r5"],
        selected_epochs=["04_r2", "06_r3", "08_r4", "10_r5"],
        light_epochs=None,
        dark_epochs=["08_r4"],
    )

    assert pairs == [
        ("04_r2", "08_r4"),
        ("06_r3", "08_r4"),
        ("10_r5", "08_r4"),
    ]


def test_select_run_epochs_deduplicates_requested_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    selected_epochs = module.select_run_epochs(
        ["02_r1", "04_r2", "06_r3"],
        ["04_r2", "02_r1", "04_r2"],
    )

    assert selected_epochs == ["04_r2", "02_r1"]


def test_select_light_dark_pairs_deduplicates_light_and_dark_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )

    pairs = module.select_light_dark_pairs(
        ["02_r1", "04_r2", "06_r3", "08_r4"],
        selected_epochs=["02_r1", "04_r2", "06_r3", "08_r4"],
        light_epochs=["02_r1", "04_r2", "02_r1"],
        dark_epochs=["08_r4", "08_r4", "06_r3"],
    )

    assert pairs == [
        ("02_r1", "08_r4"),
        ("02_r1", "06_r3"),
        ("04_r2", "08_r4"),
        ("04_r2", "06_r3"),
    ]


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


def test_build_family_dataset_includes_speed_basis_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    speed_outputs = module._format_speed_outputs(
        transform={
            "mode": "bspline",
            "basis": "bspline",
            "n_features": 4,
            "spline_order": 3,
            "bounds": np.asarray([0.0, 2.0], dtype=float),
            "reference_value": 1.0,
            "mean": np.nan,
            "std": np.nan,
        },
        coef_speed_basis_base=np.arange(8, dtype=float).reshape(4, 2),
        coef_speed_basis_full=np.arange(8, 16, dtype=float).reshape(4, 2),
        n_units=2,
    )
    results_by_traj = {
        trajectory: {
            0.1: _fake_result(
                module,
                speed_outputs=speed_outputs,
                family_name="dense_gain",
            )
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_family_dataset(
        family_name="dense_gain",
        results_by_traj=results_by_traj,
        ridge_values=[0.1],
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        dark_movement_firing_rates=np.asarray([1.0, 2.0]),
        min_dark_firing_rate_hz=0.5,
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={"speed_feature_mode": "bspline"},
    )

    assert "speed_feature_mode" in dataset
    assert "coef_speed_basis_base_all" in dataset
    assert "coef_speed_basis_full_all" in dataset
    assert dataset["speed_feature_mode"].sel(
        trajectory=module.TRAJECTORY_TYPES[0],
        ridge=0.1,
    ).item() == "bspline"
    assert dataset["coef_speed_basis_full_all"].dims == (
        "trajectory",
        "ridge",
        "speed_basis_feature",
        "unit",
    )
    assert "coef_light_gain_spline_full_all" in dataset
    assert dataset["coef_light_gain_spline_full_all"].dims == (
        "trajectory",
        "ridge",
        "gain_basis_feature",
        "unit",
    )
    assert dataset.sizes["speed_basis_feature"] == 4
    assert list(dataset.coords["speed_bound"].values) == ["lower", "upper"]


def test_build_family_dataset_sep_uses_explicit_dark_and_light_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    speed_outputs = module._format_speed_outputs(
        transform=module._empty_speed_feature_transform(),
        coef_speed_basis_base=np.full((0, 2), np.nan),
        coef_speed_basis_full=np.full((0, 2), np.nan),
        n_units=2,
    )
    results_by_traj = {
        trajectory: {
            0.1: _fake_result(
                module,
                speed_outputs=speed_outputs,
                family_name="independent_light_field",
            )
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_family_dataset(
        family_name="independent_light_field",
        results_by_traj=results_by_traj,
        ridge_values=[0.1],
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        dark_movement_firing_rates=np.asarray([1.0, 2.0]),
        min_dark_firing_rate_hz=0.5,
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={},
    )

    assert "coef_place_dark_full_all" in dataset
    assert "coef_place_light_full_all" in dataset
    assert "coef_place_full_all" not in dataset
    assert "coef_place_x_light_full_all" not in dataset
    assert dataset["coef_place_light_full_all"].dims == (
        "trajectory",
        "ridge",
        "place_basis",
        "unit",
    )


def test_build_family_dataset_scalar_gain_uses_segment_basis_coord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_dark_light_module(
        monkeypatch,
        ["dark_light_glm.py"],
    )
    speed_outputs = module._format_speed_outputs(
        transform=module._empty_speed_feature_transform(),
        coef_speed_basis_base=np.full((0, 2), np.nan),
        coef_speed_basis_full=np.full((0, 2), np.nan),
        n_units=2,
    )
    results_by_traj = {
        trajectory: {
            0.1: _fake_result(
                module,
                speed_outputs=speed_outputs,
                family_name="segment_scalar_gain",
            )
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_family_dataset(
        family_name="segment_scalar_gain",
        results_by_traj=results_by_traj,
        ridge_values=[0.1],
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        dark_movement_firing_rates=np.asarray([1.0, 2.0]),
        min_dark_firing_rate_hz=0.5,
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={},
    )

    assert "coef_segment_scalar_gain_full_all" in dataset
    assert dataset["coef_segment_scalar_gain_full_all"].dims == (
        "trajectory",
        "ridge",
        "segment_basis",
        "unit",
    )
    assert dataset.sizes["segment_basis"] == 3
    assert "segment_edges" in dataset
