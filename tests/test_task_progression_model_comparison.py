from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_NAME = "v1ca1.task_progression.swap_glm_comparison"


def _reload_model_comparison_module(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
):
    monkeypatch.setattr(sys, "argv", list(argv))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _fake_speed_outputs() -> dict[str, object]:
    return {
        "speed_feature_mode": "linear",
        "n_speed_features": 1,
        "speed_basis": "linear",
        "speed_spline_order": np.nan,
        "speed_basis_bounds": np.asarray([np.nan, np.nan], dtype=float),
        "speed_reference_value": 3.0,
        "speed_mean": 3.0,
        "speed_std": 1.0,
        "coef_speed": np.asarray([0.1, 0.2], dtype=float),
        "coef_speed_basis": np.asarray([[0.1, 0.2]], dtype=float),
    }


class _FakeBasis:
    def __init__(self, n_basis_funcs=2, order=4, bounds=(0.0, 1.0)):
        self.n_basis_funcs = int(n_basis_funcs)

    def compute_features(self, values):
        values = np.asarray(values, dtype=float).reshape(-1)
        return np.column_stack(
            [values ** power for power in range(self.n_basis_funcs)]
        )


def _patch_fake_glm_backend(module, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, np.ndarray] = {}

    class FakePopulationGLM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, x_train, y_train):
            x_train = np.asarray(x_train, dtype=float)
            y_train = np.asarray(y_train, dtype=float)
            captured["x_train"] = x_train
            self.intercept_ = np.zeros((y_train.shape[1],), dtype=float)
            self.coef_ = np.arange(
                x_train.shape[1] * y_train.shape[1],
                dtype=float,
            ).reshape(x_train.shape[1], y_train.shape[1])
            return self

    monkeypatch.setattr(module, "BSplineEval", _FakeBasis)
    monkeypatch.setattr(module, "PopulationGLM", FakePopulationGLM)
    return captured


def _fake_epoch_inputs(n_units: int = 1) -> dict[str, dict[str, np.ndarray]]:
    unit_ids = np.arange(11, 11 + n_units)

    def make_split(p_values):
        p_values = np.asarray(p_values, dtype=float)
        return {
            "unit_ids": unit_ids,
            "y": np.ones((p_values.size, n_units), dtype=float),
            "p": p_values,
            "v": None,
        }

    return {
        "train_dark": make_split([0.1, 0.4]),
        "train_light": make_split([0.2, 0.6]),
        "validation_light": make_split([0.2, 0.8]),
        "test_light": make_split([0.2, 0.8]),
    }


def _fake_result(module, *, model_name: str) -> dict[str, object]:
    n_units = 2
    result = {
        "unit_ids": np.asarray([11, 12], dtype=int),
        "intercept": np.asarray([0.5, 0.6], dtype=float),
        "coef_light": np.asarray([0.2, 0.3], dtype=float),
        "coef_place_dark": np.full((3, n_units), 0.4, dtype=float),
        "speed_outputs": _fake_speed_outputs(),
        "swap_source_trajectory": "center_to_right",
        "swap_segment_index_1based": 3,
        "swap_segment_start": 0.6,
        "swap_segment_end": 1.0,
        "tp_grid": np.linspace(0.0, 1.0, 5),
        "dark_hz_grid": np.full((5, n_units), 1.0, dtype=float),
        "train_light_hz_grid": np.full((5, n_units), 2.0, dtype=float),
        "test_light_unswapped_hz_grid": np.full((5, n_units), 2.0, dtype=float),
        "test_light_swapped_hz_grid": np.full((5, n_units), 2.5, dtype=float),
        "train_dark_metrics": {
            "spike_sum": np.asarray([10.0, 11.0]),
            "ll_sum": np.asarray([1.0, 2.0]),
            "ll_per_spike": np.asarray([0.1, 0.2]),
            "ll_bits_per_spike": np.asarray([0.2, 0.3]),
            "deviance_explained": np.asarray([0.4, 0.5]),
        },
        "train_light_metrics": {
            "spike_sum": np.asarray([12.0, 13.0]),
            "ll_sum": np.asarray([1.5, 2.5]),
            "ll_per_spike": np.asarray([0.15, 0.25]),
            "ll_bits_per_spike": np.asarray([0.25, 0.35]),
            "deviance_explained": np.asarray([0.45, 0.55]),
        },
        "validation_light_unswapped_metrics": {
            "spike_sum": np.asarray([13.0, 14.0]),
            "ll_sum": np.asarray([1.6, 2.6]),
            "ll_per_spike": np.asarray([0.16, 0.26]),
            "ll_bits_per_spike": np.asarray([0.26, 0.36]),
            "deviance_explained": np.asarray([0.46, 0.56]),
        },
        "validation_light_swapped_metrics": {
            "spike_sum": np.asarray([13.0, 14.0]),
            "ll_sum": np.asarray([1.8, 2.8]),
            "ll_per_spike": np.asarray([0.18, 0.28]),
            "ll_bits_per_spike": np.asarray([0.28, 0.38]),
            "deviance_explained": np.asarray([0.48, 0.58]),
        },
        "test_light_unswapped_metrics": {
            "spike_sum": np.asarray([14.0, 15.0]),
            "ll_sum": np.asarray([1.7, 2.7]),
            "ll_per_spike": np.asarray([0.17, 0.27]),
            "ll_bits_per_spike": np.asarray([0.27, 0.37]),
            "deviance_explained": np.asarray([0.47, 0.57]),
        },
        "test_light_swapped_metrics": {
            "spike_sum": np.asarray([14.0, 15.0]),
            "ll_sum": np.asarray([2.0, 3.0]),
            "ll_per_spike": np.asarray([0.2, 0.3]),
            "ll_bits_per_spike": np.asarray([0.3, 0.4]),
            "deviance_explained": np.asarray([0.5, 0.6]),
        },
        "train_dark_null_rate_hz": np.asarray([1.0, 1.1], dtype=float),
        "train_light_null_rate_hz": np.asarray([1.2, 1.3], dtype=float),
        "validation_light_null_rate_hz": np.asarray([1.2, 1.3], dtype=float),
        "test_light_null_rate_hz": np.asarray([1.2, 1.3], dtype=float),
        "train_dark_summary": {
            "occupancy_s": np.asarray([0.1, 0.2], dtype=float),
            "spike_count": np.ones((2, n_units), dtype=float),
            "observed_rate_hz": np.full((2, n_units), 10.0, dtype=float),
        },
        "train_light_summary": {
            "occupancy_s": np.asarray([0.2, 0.3], dtype=float),
            "spike_count": np.full((2, n_units), 2.0, dtype=float),
            "observed_rate_hz": np.full((2, n_units), 12.0, dtype=float),
        },
        "validation_light_summary": {
            "occupancy_s": np.asarray([0.25, 0.35], dtype=float),
            "spike_count": np.full((2, n_units), 2.5, dtype=float),
            "observed_rate_hz": np.full((2, n_units), 13.0, dtype=float),
        },
        "test_light_summary": {
            "occupancy_s": np.asarray([0.3, 0.4], dtype=float),
            "spike_count": np.full((2, n_units), 3.0, dtype=float),
            "observed_rate_hz": np.full((2, n_units), 14.0, dtype=float),
        },
    }
    if model_name == "visual":
        result["coef_place_light"] = np.full((3, n_units), 0.7, dtype=float)
    elif model_name == "task_segment_bump":
        result["coef_segment_bump_gain"] = np.full((3, n_units), 0.8, dtype=float)
    elif model_name == "task_segment_scalar":
        result["coef_segment_scalar_gain"] = np.full((3, n_units), 0.9, dtype=float)
    elif model_name == "task_dense_gain":
        result["coef_gain_spline"] = np.full((3, n_units), 1.1, dtype=float)
    else:
        raise ValueError(model_name)
    return result


def _fake_selected_dark_light_dataset(module, xr, *, model_name: str):
    coords = {
        "trajectory": np.asarray(module.TRAJECTORY_TYPES, dtype=str),
        "unit": np.asarray([11, 12], dtype=int),
        "tp_grid": np.linspace(0.0, 1.0, 5),
        "segment_edge": np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
    }
    return xr.Dataset(
        data_vars={},
        coords=coords,
        attrs={
            "schema_version": "4",
            "fit_stage": "selected",
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "model_name": model_name,
            "dark_train_epoch": "08_r4",
            "light_train_epoch": "02_r1",
            "bin_size_s": 0.05,
            "selected_bin_size_s": 0.05,
            "selected_n_splines": 3,
            "spline_order": 4,
            "has_speed": False,
            "has_light_offset_term": True,
            "speed_feature_mode": "none",
            "n_speed_features": 0,
            "speed_spline_order": np.nan,
            "selected_ridge": 1e-3,
            "selection_score": 0.2,
        },
    )


def _with_linear_speed_metadata(dataset, *, speed_mean: float, speed_std: float):
    n_trajectories = int(dataset.sizes["trajectory"])
    dataset = dataset.assign(
        {
            "speed_feature_mode": (
                ("trajectory",),
                np.full((n_trajectories,), "linear", dtype=object),
            ),
            "speed_basis": (
                ("trajectory",),
                np.full((n_trajectories,), "linear", dtype=object),
            ),
            "n_speed_features": (
                ("trajectory",),
                np.ones((n_trajectories,), dtype=int),
            ),
            "speed_spline_order": (
                ("trajectory",),
                np.full((n_trajectories,), np.nan, dtype=float),
            ),
            "speed_basis_bounds": (
                ("trajectory", "speed_bound"),
                np.full((n_trajectories, 2), np.nan, dtype=float),
            ),
            "speed_reference_value": (
                ("trajectory",),
                np.full((n_trajectories,), speed_mean, dtype=float),
            ),
            "speed_mean": (
                ("trajectory",),
                np.full((n_trajectories,), speed_mean, dtype=float),
            ),
            "speed_std": (
                ("trajectory",),
                np.full((n_trajectories,), speed_std, dtype=float),
            ),
        }
    )
    dataset.attrs.update(
        {
            "has_speed": True,
            "speed_feature_mode": "linear",
            "n_speed_features": 1,
            "speed_spline_order": np.nan,
        }
    )
    return dataset


def _fake_selected_swap_results(module, *, swapped_bits: np.ndarray):
    swapped_bits = np.asarray(swapped_bits, dtype=float)
    results = {}
    for trajectory in module.TRAJECTORY_TYPES:
        results[trajectory] = {
            "unit_ids": np.asarray([11, 12], dtype=int),
            "swap_source_trajectory": module.SWAP_CONFIG[trajectory]["source_trajectory"],
            "swap_segment_index_1based": (
                int(module.SWAP_CONFIG[trajectory]["segment_index"]) + 1
            ),
            "swap_segment_start": 0.3,
            "swap_segment_end": 0.7,
            "dark_hz_grid": np.full((5, 2), 1.0, dtype=float),
            "train_light_hz_grid": np.full((5, 2), 2.0, dtype=float),
            "test_light_unswapped_hz_grid": np.full((5, 2), 2.0, dtype=float),
            "test_light_swapped_hz_grid": np.full((5, 2), 2.5, dtype=float),
            "test_light_swapped_segment_n_bins": 4,
            "test_light_full_n_bins": 10,
        }
        for metric_name in (
            "test_light_swapped_segment_unswapped_metrics",
            "test_light_swapped_segment_swapped_metrics",
            "test_light_full_unswapped_metrics",
            "test_light_full_swapped_metrics",
        ):
            results[trajectory][metric_name] = {
                "raw_ll_sum": np.asarray([1.0, 2.0], dtype=float),
                "spike_sum": np.asarray([10.0, 20.0], dtype=float),
                "raw_ll_bits_per_spike": swapped_bits.copy(),
            }
    return results


def test_validate_model_comparison_epochs_rejects_duplicate_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    with pytest.raises(ValueError, match="must all be distinct"):
        module.validate_model_comparison_epochs(
            ["02_r1", "04_r2", "08_r4"],
            dark_train_epoch="08_r4",
            light_train_epoch="02_r1",
            light_test_epoch="02_r1",
        )


def test_parse_arguments_rejects_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        [
            "swap_glm_comparison.py",
            "--cuda-visible-devices",
            "0,1",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-train-epoch",
            "08_r4",
            "--light-train-epoch",
            "02_r1",
            "--light-test-epoch",
            "04_r2",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_arguments()


def test_parse_arguments_supports_model_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        [
            "swap_glm_comparison.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-train-epoch",
            "08_r4",
            "--light-train-epoch",
            "02_r1",
            "--light-test-epoch",
            "04_r2",
            "--models",
            "visual",
            "task_segment_scalar",
        ],
    )

    args = module.parse_arguments()
    assert args.models == ["visual", "task_segment_scalar"]
    assert not hasattr(args, "bin_size_s")
    assert not hasattr(args, "include_light_offset_term")
    assert args.dark_light_glm_dir is None
    assert args.swap_light_offset is False


def test_parse_arguments_supports_selected_glm_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        [
            "swap_glm_comparison.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--dark-train-epoch",
            "08_r4",
            "--light-train-epoch",
            "02_r1",
            "--light-test-epoch",
            "04_r2",
            "--dark-light-glm-dir",
            str(tmp_path),
            "--swap-light-offset",
        ],
    )

    args = module.parse_arguments()
    assert args.dark_light_glm_dir == tmp_path
    assert args.swap_light_offset is True


def test_normalize_requested_models_adds_visual_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    model_names, messages = module.normalize_requested_models(["task_segment_bump"])

    assert model_names == ["visual", "task_segment_bump"]
    assert messages == ["Added required baseline model 'visual'."]


def test_validate_selected_dark_light_glms_accepts_matching_nan_speed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xr = pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    datasets = {
        "visual": _fake_selected_dark_light_dataset(module, xr, model_name="visual"),
        "task_segment_bump": _fake_selected_dark_light_dataset(
            module,
            xr,
            model_name="task_segment_bump",
        ),
    }

    metadata = module.validate_selected_dark_light_glms(
        datasets,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
    )

    assert metadata["bin_size_s"] == pytest.approx(0.05)
    assert metadata["n_splines"] == 3
    assert metadata["speed_feature_mode"] == "none"


def test_validate_selected_dark_light_glms_rejects_mismatched_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xr = pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    visual = _fake_selected_dark_light_dataset(module, xr, model_name="visual")
    task = _fake_selected_dark_light_dataset(
        module,
        xr,
        model_name="task_segment_bump",
    ).assign_coords(unit=np.asarray([11, 13], dtype=int))

    with pytest.raises(ValueError, match="coordinate 'unit'"):
        module.validate_selected_dark_light_glms(
            {"visual": visual, "task_segment_bump": task},
            animal_name="L14",
            date="20240611",
            region="v1",
            dark_train_epoch="08_r4",
            light_train_epoch="02_r1",
        )


def test_validate_selected_dark_light_glms_rejects_stale_linear_speed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xr = pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    visual = _with_linear_speed_metadata(
        _fake_selected_dark_light_dataset(module, xr, model_name="visual"),
        speed_mean=np.nan,
        speed_std=np.nan,
    )

    with pytest.raises(ValueError, match="invalid linear speed metadata"):
        module.validate_selected_dark_light_glms(
            {"visual": visual},
            animal_name="L14",
            date="20240611",
            region="v1",
            dark_train_epoch="08_r4",
            light_train_epoch="02_r1",
        )


def test_output_stem_marks_no_light_offset_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    default_stem = module._output_stem(
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        model_name="task_segment_bump",
        bin_size_s=0.02,
        ridge=1e-3,
    )
    no_offset_stem = module._output_stem(
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        model_name="task_segment_bump",
        bin_size_s=0.05,
        ridge=1e-3,
        include_light_offset_term=False,
    )

    assert "no_light_offset" not in default_stem
    assert "testlight_bin0p02s_task_segment_bump_ridge" in default_stem
    assert "testlight_bin0p05s_task_segment_bump_no_light_offset_ridge" in no_offset_stem
    assert "task_segment_bump_no_light_offset_ridge" in no_offset_stem
    assert no_offset_stem != default_stem


def test_build_train_epoch_fr_mask_requires_dark_and_light_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
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


def test_build_visual_swapped_light_eta_replaces_masked_bins_with_full_light_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    own_light_eta = np.asarray(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
        dtype=float,
    )
    own_light_place = np.asarray(
        [[0.5, 1.0], [0.6, 1.1], [0.7, 1.2], [0.8, 1.3]],
        dtype=float,
    )
    paired_light_place = np.asarray(
        [[5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0]],
        dtype=float,
    )
    own_light_offset = np.asarray([0.2, 0.4], dtype=float)
    paired_light_offset = np.asarray([1.2, 1.4], dtype=float)
    swap_mask = np.asarray([False, True, False, True])

    swapped = module.build_visual_swapped_light_eta(
        own_light_eta=own_light_eta,
        own_light_place=own_light_place,
        paired_light_place=paired_light_place,
        own_light_offset=own_light_offset,
        paired_light_offset=paired_light_offset,
        swap_mask=swap_mask,
    )

    expected = own_light_eta.copy()
    expected[1] += paired_light_place[1] - own_light_place[1]
    expected[3] += paired_light_place[3] - own_light_place[3]
    assert np.allclose(swapped, expected)
    assert np.allclose(swapped[0], own_light_eta[0])
    assert np.allclose(swapped[2], own_light_eta[2])


def test_build_visual_swapped_light_eta_can_swap_light_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    own_light_eta = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float)
    own_light_place = np.asarray([[0.5, 1.0], [0.6, 1.1]], dtype=float)
    paired_light_place = np.asarray([[5.0, 6.0], [6.0, 7.0]], dtype=float)
    own_light_offset = np.asarray([0.2, 0.4], dtype=float)
    paired_light_offset = np.asarray([1.2, 1.4], dtype=float)
    swap_mask = np.asarray([False, True])

    swapped = module.build_visual_swapped_light_eta(
        own_light_eta=own_light_eta,
        own_light_place=own_light_place,
        paired_light_place=paired_light_place,
        own_light_offset=own_light_offset,
        paired_light_offset=paired_light_offset,
        swap_mask=swap_mask,
        swap_light_offset=True,
    )

    expected = own_light_eta.copy()
    expected[1] += (
        paired_light_place[1]
        - own_light_place[1]
        + paired_light_offset
        - own_light_offset
    )
    assert np.allclose(swapped, expected)


def test_build_task_swapped_light_eta_replaces_masked_segment_light_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    own_light_eta = np.zeros((3, 2), dtype=float)
    own_gain_basis = np.asarray(
        [[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    own_coef_gain = np.asarray(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=float,
    )
    paired_coef_gain = np.asarray(
        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        dtype=float,
    )
    own_light_offset = np.asarray([0.5, 1.0], dtype=float)
    paired_light_offset = np.asarray([2.5, 4.0], dtype=float)
    swap_mask = np.asarray([False, True, False])

    swapped = module.build_task_swapped_light_eta(
        own_light_eta=own_light_eta,
        own_gain_basis=own_gain_basis,
        own_coef_gain=own_coef_gain,
        paired_coef_gain=paired_coef_gain,
        own_light_offset=own_light_offset,
        paired_light_offset=paired_light_offset,
        swap_segment_index=1,
        swap_mask=swap_mask,
    )

    expected = np.zeros((3, 2), dtype=float)
    expected[swap_mask] += own_gain_basis[swap_mask, [1]] * (
        paired_coef_gain[[1], :] - own_coef_gain[[1], :]
    )
    assert np.allclose(swapped, expected)
    assert np.allclose(swapped[0], [0.0, 0.0])
    assert np.allclose(swapped[2], [0.0, 0.0])


def test_build_task_swapped_light_eta_can_swap_light_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    own_light_eta = np.zeros((2, 2), dtype=float)
    own_gain_basis = np.asarray([[1.0, 0.0], [0.25, 0.75]], dtype=float)
    own_coef_gain = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    paired_coef_gain = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    own_light_offset = np.asarray([0.5, 1.0], dtype=float)
    paired_light_offset = np.asarray([2.5, 4.0], dtype=float)
    swap_mask = np.asarray([False, True])

    swapped = module.build_task_swapped_light_eta(
        own_light_eta=own_light_eta,
        own_gain_basis=own_gain_basis,
        own_coef_gain=own_coef_gain,
        paired_coef_gain=paired_coef_gain,
        own_light_offset=own_light_offset,
        paired_light_offset=paired_light_offset,
        swap_segment_index=1,
        swap_mask=swap_mask,
        swap_light_offset=True,
    )

    expected = np.zeros((2, 2), dtype=float)
    expected[swap_mask] += own_gain_basis[swap_mask, [1]] * (
        paired_coef_gain[[1], :] - own_coef_gain[[1], :]
    )
    expected[swap_mask] += paired_light_offset - own_light_offset
    assert np.allclose(swapped, expected)


def test_fit_visual_model_can_omit_light_offset_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    captured = _patch_fake_glm_backend(module, monkeypatch)

    result = module.fit_visual_model_for_trajectory(
        epoch_inputs=_fake_epoch_inputs(),
        ridge=1e-3,
        n_splines=2,
        spline_order=4,
        include_light_offset_term=False,
        speed_feature_mode="linear",
        n_splines_speed=5,
        spline_order_speed=4,
        speed_bounds=None,
    )

    assert captured["x_train"].shape == (4, 4)
    assert result["has_light_offset_term"] is False
    assert np.allclose(result["coef_light"], [0.0])


def test_fit_visual_model_keeps_light_offset_column_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    captured = _patch_fake_glm_backend(module, monkeypatch)

    result = module.fit_visual_model_for_trajectory(
        epoch_inputs=_fake_epoch_inputs(),
        ridge=1e-3,
        n_splines=2,
        spline_order=4,
        speed_feature_mode="linear",
        n_splines_speed=5,
        spline_order_speed=4,
        speed_bounds=None,
    )

    assert captured["x_train"].shape == (4, 5)
    assert result["has_light_offset_term"] is True
    assert np.allclose(result["coef_light"], [4.0])


def test_fit_task_scalar_model_can_omit_light_offset_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    captured = _patch_fake_glm_backend(module, monkeypatch)

    result = module.fit_task_segment_scalar_model_for_trajectory(
        epoch_inputs=_fake_epoch_inputs(),
        ridge=1e-3,
        n_splines=2,
        spline_order=4,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        include_light_offset_term=False,
        speed_feature_mode="linear",
        n_splines_speed=5,
        spline_order_speed=4,
        speed_bounds=None,
    )

    assert captured["x_train"].shape == (4, 5)
    assert result["has_light_offset_term"] is False
    assert np.allclose(result["coef_light"], [0.0])
    assert np.allclose(
        result["coef_segment_scalar_gain"].reshape(-1),
        [2.0, 3.0, 4.0],
    )


def test_build_model_dataset_task_scalar_includes_family_specific_gain_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    results_by_traj = {
        trajectory: _fake_result(module, model_name="task_segment_scalar")
        for trajectory in module.TRAJECTORY_TYPES
    }
    validation_sweep = {
        "candidate_ridge": np.asarray([1e-2, 1e-3], dtype=float),
        "validation_swapped_ll_bits_per_spike": np.full((2, 4, 2), 0.5, dtype=float),
        "validation_swapped_deviance_explained": np.full((2, 4, 2), 0.2, dtype=float),
        "pooled_validation_swapped_ll_bits_per_spike_median": np.asarray(
            [0.45, 0.5],
            dtype=float,
        ),
        "pooled_validation_swapped_deviance_explained_median": np.asarray(
            [0.18, 0.2],
            dtype=float,
        ),
        "pooled_validation_finite_count": np.asarray([8, 8], dtype=int),
        "selected_ridge": 1e-3,
    }
    heldout_split_by_traj = {
        trajectory: {
            "lap_start_s": np.asarray([0.0, 1.0], dtype=float),
            "lap_end_s": np.asarray([0.5, 1.5], dtype=float),
            "lap_split": np.asarray(["validation", "test"], dtype=str),
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_model_dataset(
        model_name="task_segment_scalar",
        results_by_traj=results_by_traj,
        validation_sweep=validation_sweep,
        heldout_split_by_traj=heldout_split_by_traj,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        ridge=1e-3,
        dark_movement_firing_rates=np.asarray([1.0, 2.0], dtype=float),
        light_movement_firing_rates=np.asarray([1.5, 2.5], dtype=float),
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={
            "bin_size_s": 0.02,
            "n_splines": 3,
            "spline_order": 4,
            "use_speed": True,
            "region_threshold_hz": 0.5,
            "dark_region_threshold_hz": 0.5,
            "light_region_threshold_hz": 0.5,
            "seed": 47,
        },
    )

    assert "coef_segment_scalar_gain" in dataset
    assert dataset["coef_segment_scalar_gain"].dims == ("trajectory", "segment_basis", "unit")


def test_build_model_dataset_task_dense_gain_uses_gain_basis_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    results_by_traj = {
        trajectory: _fake_result(module, model_name="task_dense_gain")
        for trajectory in module.TRAJECTORY_TYPES
    }
    validation_sweep = {
        "candidate_ridge": np.asarray([1e-2, 1e-3], dtype=float),
        "validation_swapped_ll_bits_per_spike": np.full((2, 4, 2), 0.5, dtype=float),
        "validation_swapped_deviance_explained": np.full((2, 4, 2), 0.2, dtype=float),
        "pooled_validation_swapped_ll_bits_per_spike_median": np.asarray(
            [0.45, 0.5],
            dtype=float,
        ),
        "pooled_validation_swapped_deviance_explained_median": np.asarray(
            [0.18, 0.2],
            dtype=float,
        ),
        "pooled_validation_finite_count": np.asarray([8, 8], dtype=int),
        "selected_ridge": 1e-3,
    }
    heldout_split_by_traj = {
        trajectory: {
            "lap_start_s": np.asarray([0.0, 1.0], dtype=float),
            "lap_end_s": np.asarray([0.5, 1.5], dtype=float),
            "lap_split": np.asarray(["validation", "test"], dtype=str),
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_model_dataset(
        model_name="task_dense_gain",
        results_by_traj=results_by_traj,
        validation_sweep=validation_sweep,
        heldout_split_by_traj=heldout_split_by_traj,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        ridge=1e-3,
        dark_movement_firing_rates=np.asarray([1.0, 2.0], dtype=float),
        light_movement_firing_rates=np.asarray([1.5, 2.5], dtype=float),
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={
            "bin_size_s": 0.02,
            "n_splines": 3,
            "spline_order": 4,
            "use_speed": True,
            "region_threshold_hz": 0.5,
            "dark_region_threshold_hz": 0.5,
            "light_region_threshold_hz": 0.5,
            "seed": 47,
        },
    )

    assert "coef_gain_spline" in dataset
    assert dataset["coef_gain_spline"].dims == ("trajectory", "gain_basis_feature", "unit")


def test_build_model_dataset_visual_includes_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    results_by_traj = {
        trajectory: _fake_result(module, model_name="visual")
        for trajectory in module.TRAJECTORY_TYPES
    }
    validation_sweep = {
        "candidate_ridge": np.asarray([1e-2, 1e-3], dtype=float),
        "validation_swapped_ll_bits_per_spike": np.full((2, 4, 2), 0.5, dtype=float),
        "validation_swapped_deviance_explained": np.full((2, 4, 2), 0.2, dtype=float),
        "pooled_validation_swapped_ll_bits_per_spike_median": np.asarray(
            [0.45, 0.5],
            dtype=float,
        ),
        "pooled_validation_swapped_deviance_explained_median": np.asarray(
            [0.18, 0.2],
            dtype=float,
        ),
        "pooled_validation_finite_count": np.asarray([8, 8], dtype=int),
        "selected_ridge": 1e-3,
    }
    heldout_split_by_traj = {
        trajectory: {
            "lap_start_s": np.asarray([0.0, 1.0], dtype=float),
            "lap_end_s": np.asarray([0.5, 1.5], dtype=float),
            "lap_split": np.asarray(["validation", "test"], dtype=str),
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_model_dataset(
        model_name="visual",
        results_by_traj=results_by_traj,
        validation_sweep=validation_sweep,
        heldout_split_by_traj=heldout_split_by_traj,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        ridge=1e-3,
        dark_movement_firing_rates=np.asarray([1.0, 2.0], dtype=float),
        light_movement_firing_rates=np.asarray([1.5, 2.5], dtype=float),
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={
            "bin_size_s": 0.02,
            "n_splines": 3,
            "spline_order": 4,
            "use_speed": True,
            "region_threshold_hz": 0.5,
            "dark_region_threshold_hz": 0.5,
            "light_region_threshold_hz": 0.5,
            "seed": 47,
        },
    )

    assert "coef_place_light" in dataset
    assert "light_train_movement_firing_rate_hz" in dataset
    assert "swap_source_trajectory" in dataset
    assert "test_light_swapped_ll_bits_per_spike" in dataset
    assert "validation_swapped_ll_bits_per_spike_by_ridge" in dataset
    assert "test_lap_start_s" in dataset
    assert "speed_basis_bounds" in dataset
    assert dataset["coef_intercept"].dims == ("trajectory", "unit")
    assert dataset["coef_place_light"].dims == ("trajectory", "place_basis", "unit")
    assert dataset["speed_basis_bounds"].dims == ("trajectory", "speed_bound")
    assert dataset["validation_swapped_ll_bits_per_spike_by_ridge"].dims == (
        "candidate_ridge",
        "trajectory",
        "unit",
    )
    assert dataset["test_lap_split"].dims == ("trajectory", "test_lap")
    assert np.allclose(dataset.coords["segment_edge"].values, [0.0, 0.3, 0.7, 1.0])
    assert dataset.attrs["test_scoring_scope"] == "swapped_segment_only"
    assert dataset.attrs["schema_version"] == "3"
    assert dataset.attrs["has_light_offset_term"] is True
    assert dataset.attrs["min_light_firing_rate_hz"] == pytest.approx(0.5)
    assert dataset.attrs["selected_ridge"] == pytest.approx(1e-3)


def test_build_model_dataset_no_light_offset_saves_zero_offset_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    results_by_traj = {
        trajectory: _fake_result(module, model_name="visual")
        for trajectory in module.TRAJECTORY_TYPES
    }
    validation_sweep = {
        "candidate_ridge": np.asarray([1e-2, 1e-3], dtype=float),
        "validation_swapped_ll_bits_per_spike": np.full((2, 4, 2), 0.5, dtype=float),
        "validation_swapped_deviance_explained": np.full((2, 4, 2), 0.2, dtype=float),
        "pooled_validation_swapped_ll_bits_per_spike_median": np.asarray(
            [0.45, 0.5],
            dtype=float,
        ),
        "pooled_validation_swapped_deviance_explained_median": np.asarray(
            [0.18, 0.2],
            dtype=float,
        ),
        "pooled_validation_finite_count": np.asarray([8, 8], dtype=int),
        "selected_ridge": 1e-3,
    }
    heldout_split_by_traj = {
        trajectory: {
            "lap_start_s": np.asarray([0.0, 1.0], dtype=float),
            "lap_end_s": np.asarray([0.5, 1.5], dtype=float),
            "lap_split": np.asarray(["validation", "test"], dtype=str),
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_model_dataset(
        model_name="visual",
        results_by_traj=results_by_traj,
        validation_sweep=validation_sweep,
        heldout_split_by_traj=heldout_split_by_traj,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        ridge=1e-3,
        dark_movement_firing_rates=np.asarray([1.0, 2.0], dtype=float),
        light_movement_firing_rates=np.asarray([1.5, 2.5], dtype=float),
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        sources={"analysis_path": "/tmp/example"},
        fit_parameters={
            "bin_size_s": 0.02,
            "n_splines": 3,
            "spline_order": 4,
            "use_speed": True,
            "include_light_offset_term": False,
            "region_threshold_hz": 0.5,
            "dark_region_threshold_hz": 0.5,
            "light_region_threshold_hz": 0.5,
            "seed": 47,
        },
    )

    assert dataset.attrs["has_light_offset_term"] is False
    assert dataset["coef_light_offset"].attrs["fitted"] is False
    assert np.allclose(dataset["coef_light_offset"].values, 0.0)
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    assert fit_parameters["include_light_offset_term"] is False


def test_choose_best_ridge_prefers_stronger_regularization_in_near_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    selected = module.choose_best_ridge(
        [1e-3, 1e-2],
        {1e-3: 0.5, 1e-2: 0.5 + 5e-7},
        atol=1e-6,
    )

    assert selected == pytest.approx(1e-2)


def test_split_test_light_laps_by_trajectory_splits_laps_reproducibly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    class FakeIntervalSet:
        def __init__(self, *, start, end, time_units="s"):
            self.start = np.asarray(start, dtype=float)
            self.end = np.asarray(end, dtype=float)
            self.time_units = time_units

    trajectory_intervals = {
        trajectory: FakeIntervalSet(
            start=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float),
            end=np.asarray([0.5, 1.5, 2.5, 3.5], dtype=float),
        )
        for trajectory in module.TRAJECTORY_TYPES
    }

    split_a = module.split_test_light_laps_by_trajectory(trajectory_intervals, seed=47)
    split_b = module.split_test_light_laps_by_trajectory(trajectory_intervals, seed=47)

    for trajectory in module.TRAJECTORY_TYPES:
        assert np.array_equal(
            split_a[trajectory]["validation_indices"],
            split_b[trajectory]["validation_indices"],
        )
        assert np.array_equal(
            split_a[trajectory]["test_indices"],
            split_b[trajectory]["test_indices"],
        )
        assert split_a[trajectory]["validation_indices"].size == 2
        assert split_a[trajectory]["test_indices"].size == 2


def test_finalize_model_results_scores_test_epoch_on_swapped_segment_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    def fake_predict_task_components(**kwargs):
        p_eval = np.asarray(kwargs["p_eval"], dtype=float)
        n_rows = p_eval.shape[0]
        return {
            "dark_eta": np.zeros((n_rows, 1), dtype=float),
            "light_eta": np.zeros((n_rows, 1), dtype=float),
            "gain_basis": np.ones((n_rows, 3), dtype=float),
        }

    def fake_speed_reference_design(_transform):
        return np.zeros((1, 0), dtype=float)

    def fake_build_observed_summary(y_epoch, p_epoch, bin_edges, *, bin_size_s):
        return {
            "occupancy_s": np.asarray([1.0, 1.0], dtype=float),
            "spike_count": np.zeros((2, y_epoch.shape[1]), dtype=float),
            "observed_rate_hz": np.zeros((2, y_epoch.shape[1]), dtype=float),
        }

    def fake_summarize_poisson_metrics(y_true, lam_pred, y_null_fit):
        n_units = y_true.shape[1]
        rows = float(y_true.shape[0])
        return {
            "spike_sum": np.full((n_units,), rows, dtype=float),
            "ll_sum": np.full((n_units,), rows, dtype=float),
            "ll_per_spike": np.full((n_units,), rows, dtype=float),
            "ll_bits_per_spike": np.full((n_units,), rows, dtype=float),
            "deviance_explained": np.full((n_units,), rows, dtype=float),
        }

    monkeypatch.setattr(module, "_predict_task_components", fake_predict_task_components)
    monkeypatch.setattr(module, "_speed_reference_design", fake_speed_reference_design)
    monkeypatch.setattr(module, "build_observed_summary", fake_build_observed_summary)
    monkeypatch.setattr(module, "summarize_poisson_metrics", fake_summarize_poisson_metrics)

    epoch_inputs = {
        "train_dark": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
        "train_light": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
        "validation_light": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
        "test_light": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
    }
    fit_template = {
        "speed_transform": {"mode": "none"},
        "basis": object(),
        "intercept": np.asarray([0.0], dtype=float),
        "coef_light": np.asarray([0.0], dtype=float),
        "coef_place_dark": np.asarray([[0.0]], dtype=float),
        "coef_segment_bump_gain": np.asarray([[0.0], [0.0], [0.0]], dtype=float),
        "coef_speed_basis": np.zeros((0, 1), dtype=float),
        "pred_train_dark": {"dark_eta": np.zeros((3, 1), dtype=float)},
        "pred_train_light": {"light_eta": np.zeros((3, 1), dtype=float)},
        "pred_validation_light": {
            "light_eta": np.zeros((3, 1), dtype=float),
            "gain_basis": np.ones((3, 3), dtype=float),
        },
        "pred_test_light": {
            "light_eta": np.zeros((3, 1), dtype=float),
            "gain_basis": np.ones((3, 3), dtype=float),
        },
        "epoch_inputs": epoch_inputs,
        "unit_ids": np.asarray([11], dtype=int),
        "speed_outputs": _fake_speed_outputs(),
    }
    fit_by_traj = {
        trajectory: dict(fit_template)
        for trajectory in module.TRAJECTORY_TYPES
    }

    results = module.finalize_model_results_by_trajectory(
        model_name="task_segment_bump",
        fit_by_traj=fit_by_traj,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        ridge=1e-3,
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        bin_size_s=0.05,
    )

    assert results["center_to_left"]["test_light_swapped_metrics"]["spike_sum"][0] == 1.0
    assert results["left_to_center"]["test_light_swapped_metrics"]["spike_sum"][0] == 1.0
    assert results["center_to_right"]["test_light_swapped_metrics"]["spike_sum"][0] == 1.0
    assert results["right_to_center"]["test_light_swapped_metrics"]["spike_sum"][0] == 1.0
    assert (
        results["center_to_left"]["validation_light_swapped_metrics"]["spike_sum"][0]
        == 1.0
    )


def test_finalize_model_results_dense_gain_swaps_only_scored_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    def fake_predict_task_dense_gain_components(**kwargs):
        p_eval = np.asarray(kwargs["p_eval"], dtype=float)
        n_rows = p_eval.shape[0]
        return {
            "dark_eta": np.zeros((n_rows, 1), dtype=float),
            "light_eta": np.zeros((n_rows, 1), dtype=float),
            "gain_part": np.zeros((n_rows, 1), dtype=float),
        }

    def fake_speed_reference_design(_transform):
        return np.zeros((1, 0), dtype=float)

    def fake_build_observed_summary(y_epoch, p_epoch, bin_edges, *, bin_size_s):
        return {
            "occupancy_s": np.asarray([1.0, 1.0], dtype=float),
            "spike_count": np.zeros((2, y_epoch.shape[1]), dtype=float),
            "observed_rate_hz": np.zeros((2, y_epoch.shape[1]), dtype=float),
        }

    def fake_summarize_poisson_metrics(y_true, lam_pred, y_null_fit):
        n_units = y_true.shape[1]
        rows = float(y_true.shape[0])
        return {
            "spike_sum": np.full((n_units,), rows, dtype=float),
            "ll_sum": np.full((n_units,), rows, dtype=float),
            "ll_per_spike": np.full((n_units,), rows, dtype=float),
            "ll_bits_per_spike": np.full((n_units,), rows, dtype=float),
            "deviance_explained": np.full((n_units,), rows, dtype=float),
        }

    class FakeBasis:
        def compute_features(self, p_eval):
            return np.asarray(p_eval, dtype=float).reshape(-1, 1)

    monkeypatch.setattr(
        module,
        "_predict_task_dense_gain_components",
        fake_predict_task_dense_gain_components,
    )
    monkeypatch.setattr(module, "_speed_reference_design", fake_speed_reference_design)
    monkeypatch.setattr(module, "build_observed_summary", fake_build_observed_summary)
    monkeypatch.setattr(module, "summarize_poisson_metrics", fake_summarize_poisson_metrics)

    epoch_inputs = {
        "train_dark": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
        "train_light": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
        "validation_light": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
        "test_light": {
            "y": np.asarray([[1.0], [2.0], [3.0]], dtype=float),
            "p": np.asarray([0.1, 0.4, 0.8], dtype=float),
        },
    }
    fit_template = {
        "speed_transform": {"mode": "none"},
        "basis": FakeBasis(),
        "gain_basis": FakeBasis(),
        "intercept": np.asarray([0.0], dtype=float),
        "coef_light": np.asarray([0.0], dtype=float),
        "coef_place_dark": np.asarray([[0.0]], dtype=float),
        "coef_gain_spline": np.asarray([[0.0]], dtype=float),
        "coef_speed_basis": np.zeros((0, 1), dtype=float),
        "pred_train_dark": {"dark_eta": np.zeros((3, 1), dtype=float)},
        "pred_train_light": {"light_eta": np.zeros((3, 1), dtype=float)},
        "pred_validation_light": {
            "light_eta": np.zeros((3, 1), dtype=float),
            "gain_part": np.zeros((3, 1), dtype=float),
        },
        "pred_test_light": {
            "light_eta": np.zeros((3, 1), dtype=float),
            "gain_part": np.zeros((3, 1), dtype=float),
        },
        "epoch_inputs": epoch_inputs,
        "unit_ids": np.asarray([11], dtype=int),
        "speed_outputs": _fake_speed_outputs(),
    }
    fit_by_traj = {
        trajectory: dict(fit_template)
        for trajectory in module.TRAJECTORY_TYPES
    }

    results = module.finalize_model_results_by_trajectory(
        model_name="task_dense_gain",
        fit_by_traj=fit_by_traj,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        ridge=1e-3,
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        bin_size_s=0.05,
    )

    assert results["center_to_left"]["test_light_swapped_metrics"]["spike_sum"][0] == 1.0


def test_summarize_poisson_metrics_uses_expected_counts_per_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    y_true = np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=float)
    lam_pred = np.asarray([[0.05], [0.95], [0.05], [0.95]], dtype=float)
    y_null_fit = np.asarray([[0.5], [0.5], [0.5], [0.5]], dtype=float)

    metrics = module.summarize_poisson_metrics(y_true, lam_pred, y_null_fit)

    assert np.isfinite(metrics["ll_bits_per_spike"]).all()
    assert np.isfinite(metrics["deviance_explained"]).all()
    assert metrics["ll_bits_per_spike"][0] > 0
    assert metrics["deviance_explained"][0] > 0


def test_plot_metric_difference_histograms_writes_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xr = pytest.importorskip("xarray")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )

    coords = {
        "trajectory": np.asarray(module.TRAJECTORY_TYPES, dtype=str),
        "unit": np.asarray([11, 12, 13], dtype=int),
    }
    task_dataset = xr.Dataset(
        data_vars={
            "test_light_swapped_ll_bits_per_spike": (
                ("trajectory", "unit"),
                np.full((4, 3), 0.3, dtype=float),
            ),
            "test_light_swapped_deviance_explained": (
                ("trajectory", "unit"),
                np.full((4, 3), 0.5, dtype=float),
            ),
        },
        coords=coords,
        attrs={
            "region": "v1",
            "light_test_epoch": "04_r2",
            "dark_train_epoch": "08_r4",
            "light_train_epoch": "02_r1",
            "ridge": 1e-3,
            "model_name": "task_segment_scalar",
        },
    )
    visual_dataset = xr.Dataset(
        data_vars={
            "test_light_swapped_ll_bits_per_spike": (
                ("trajectory", "unit"),
                np.full((4, 3), 0.1, dtype=float),
            ),
            "test_light_swapped_deviance_explained": (
                ("trajectory", "unit"),
                np.full((4, 3), 0.2, dtype=float),
            ),
        },
        coords=coords,
        attrs={**task_dataset.attrs, "model_name": "visual"},
    )

    out_path = tmp_path / "metric_hist.png"
    saved_path = module.plot_metric_difference_histograms(
        task_dataset,
        visual_dataset,
        metric_name="test_light_swapped_ll_bits_per_spike",
        out_path=out_path,
    )

    assert saved_path == out_path
    assert out_path.exists()


def test_build_selected_swap_dataset_saves_primary_delta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xr = pytest.importorskip("xarray")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    selected_datasets = {
        "visual": _fake_selected_dark_light_dataset(module, xr, model_name="visual"),
        "task_segment_bump": _fake_selected_dark_light_dataset(
            module,
            xr,
            model_name="task_segment_bump",
        ),
    }
    results_by_model = {
        "visual": _fake_selected_swap_results(
            module,
            swapped_bits=np.asarray([0.1, 0.2], dtype=float),
        ),
        "task_segment_bump": _fake_selected_swap_results(
            module,
            swapped_bits=np.asarray([0.4, 0.15], dtype=float),
        ),
    }
    observed_summaries = {
        trajectory: {
            "occupancy_s": np.asarray([0.1, 0.2], dtype=float),
            "spike_count": np.ones((2, 2), dtype=float),
            "observed_rate_hz": np.full((2, 2), 5.0, dtype=float),
        }
        for trajectory in module.TRAJECTORY_TYPES
    }

    dataset = module.build_selected_swap_dataset(
        model_names=["visual", "task_segment_bump"],
        selected_datasets=selected_datasets,
        selected_paths={
            "visual": tmp_path / "visual.nc",
            "task_segment_bump": tmp_path / "task.nc",
        },
        results_by_model=results_by_model,
        test_inputs_by_traj={},
        observed_summaries_by_traj=observed_summaries,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        shared_metadata={
            "bin_size_s": 0.05,
            "n_splines": 3,
            "spline_order": 4,
            "has_speed": False,
            "speed_feature_mode": "none",
            "n_speed_features": 0,
            "speed_spline_order": np.nan,
        },
        sources={},
        fit_parameters={},
    )

    delta = dataset[
        "test_light_swapped_segment_swapped_delta_model_minus_visual_raw_ll_bits_per_spike"
    ]
    assert np.all(np.isnan(delta.sel(model="visual").values))
    assert np.allclose(
        delta.sel(model="task_segment_bump").values,
        np.tile(np.asarray([0.3, -0.05], dtype=float), (4, 1)),
    )
    assert "test_light_swapped_segment_swapped_raw_ll_bits_per_spike" in dataset
    assert not any("deviance" in name for name in dataset.data_vars)


def test_plot_selected_model_minus_visual_histogram_writes_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xr = pytest.importorskip("xarray")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    module = _reload_model_comparison_module(
        monkeypatch,
        ["swap_glm_comparison.py"],
    )
    selected_datasets = {
        "visual": _fake_selected_dark_light_dataset(module, xr, model_name="visual"),
        "task_segment_bump": _fake_selected_dark_light_dataset(
            module,
            xr,
            model_name="task_segment_bump",
        ),
    }
    results_by_model = {
        "visual": _fake_selected_swap_results(
            module,
            swapped_bits=np.asarray([0.1, 0.2], dtype=float),
        ),
        "task_segment_bump": _fake_selected_swap_results(
            module,
            swapped_bits=np.asarray([0.4, 0.15], dtype=float),
        ),
    }
    observed_summaries = {
        trajectory: {
            "occupancy_s": np.asarray([0.1, 0.2], dtype=float),
            "spike_count": np.ones((2, 2), dtype=float),
            "observed_rate_hz": np.full((2, 2), 5.0, dtype=float),
        }
        for trajectory in module.TRAJECTORY_TYPES
    }
    dataset = module.build_selected_swap_dataset(
        model_names=["visual", "task_segment_bump"],
        selected_datasets=selected_datasets,
        selected_paths={
            "visual": tmp_path / "visual.nc",
            "task_segment_bump": tmp_path / "task.nc",
        },
        results_by_model=results_by_model,
        test_inputs_by_traj={},
        observed_summaries_by_traj=observed_summaries,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float),
        observed_bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        shared_metadata={
            "bin_size_s": 0.05,
            "n_splines": 3,
            "spline_order": 4,
            "has_speed": False,
            "speed_feature_mode": "none",
            "n_speed_features": 0,
            "speed_spline_order": np.nan,
        },
        sources={},
        fit_parameters={},
    )

    out_path = tmp_path / "selected_metric_hist.png"
    saved_path = module.plot_selected_model_minus_visual_histogram(
        dataset,
        model_name="task_segment_bump",
        out_path=out_path,
    )

    assert saved_path == out_path
    assert out_path.exists()
