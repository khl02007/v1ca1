from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_NAME = "v1ca1.task_progression.dark_light_transfer"


def _reload_module(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", list(argv))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_parse_arguments_defaults_to_no_speed_and_registry_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(
        monkeypatch,
        [
            "dark_light_transfer.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
        ],
    )

    args = module.parse_arguments()

    assert args.animal_name == "L14"
    assert args.date == "20240611"
    assert args.light_epoch is None
    assert args.dark_epoch is None
    assert args.estimator == "empirical"
    assert args.use_speed is False
    assert args.bin_size_s == pytest.approx(0.05)
    assert args.n_folds == 5
    assert args.inner_n_folds == 3
    assert args.n_shuffles == 100
    assert args.spatial_bin_size_cm == pytest.approx(4.0)
    assert args.sigma_bins == pytest.approx(1.0)
    assert args.use_tuning_stability_filter is True
    assert args.min_tuning_stability_correlation == pytest.approx(0.5)


def test_validate_arguments_rejects_speed_for_empirical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(
        monkeypatch,
        [
            "dark_light_transfer.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--use-speed",
        ],
    )
    args = module.parse_arguments()

    with pytest.raises(ValueError, match="not supported"):
        module._validate_arguments(args)


def test_validate_arguments_rejects_invalid_stability_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(
        monkeypatch,
        [
            "dark_light_transfer.py",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
            "--min-tuning-stability-correlation",
            "1.5",
        ],
    )
    args = module.parse_arguments()

    with pytest.raises(ValueError, match="between -1 and 1"):
        module._validate_arguments(args)


def test_cuda_visible_devices_is_preparsed_before_glm_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    sys.modules.pop("v1ca1.task_progression.dark_light_glm", None)

    module = _reload_module(
        monkeypatch,
        [
            "dark_light_transfer.py",
            "--cuda-visible-devices",
            "2",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
        ],
    )

    assert module._CUDA_VISIBLE_DEVICES_CLI == "2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert "--cuda-visible-devices" not in sys.argv
    assert "2" not in sys.argv

    args = module.parse_arguments()

    assert args.cuda_visible_devices == "2"


def test_load_tuning_stability_table_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    with pytest.raises(FileNotFoundError, match="task_progression.stability"):
        module.load_tuning_stability_table(
            analysis_path=tmp_path,
            animal_name="L14",
            date="20240611",
        )


def test_select_stable_units_for_epoch_uses_any_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])
    table = pd.DataFrame(
        {
            "unit": [1, 1, 2, 3, 4, 5],
            "region": ["v1", "v1", "v1", "v1", "ca1", "v1"],
            "epoch": ["02_r1", "02_r1", "02_r1", "08_r4", "02_r1", "02_r1"],
            "trajectory_type": [
                "center_to_left",
                "right_to_center",
                "center_to_left",
                "center_to_left",
                "center_to_left",
                "not_a_trajectory",
            ],
            "stability_correlation": [0.1, 0.6, np.nan, 0.7, 0.9, 1.0],
        }
    )

    stable_units = module.select_stable_units_for_epoch(
        table,
        region="v1",
        epoch="02_r1",
        min_correlation=0.5,
    )

    assert stable_units.tolist() == [1]


def test_build_unit_filter_diagnostics_combines_fr_light_and_dark_stability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    final_mask, diagnostics = module.build_unit_filter_diagnostics(
        unit_ids=np.asarray([1, 2, 3, 4]),
        fr_mask=np.asarray([True, True, False, True]),
        light_stable_units=np.asarray([1, 2, 4]),
        dark_stable_units=np.asarray([2, 4]),
        min_stability_correlation=0.5,
    )

    assert final_mask.tolist() == [False, True, False, True]
    assert diagnostics["n_units_total"] == 4
    assert diagnostics["n_units_fr_pass"] == 3
    assert diagnostics["n_units_light_stable"] == 3
    assert diagnostics["n_units_dark_stable"] == 2
    assert diagnostics["n_units_final"] == 2


def test_validate_shift_range_uses_auto_max_and_rejects_short_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    assert module.validate_shift_range(
        duration_s=10.0,
        min_shift_s=0.5,
        max_shift_s=None,
    ) == pytest.approx((0.5, 9.5))
    assert module.validate_shift_range(
        duration_s=10.0,
        min_shift_s=0.5,
        max_shift_s=4.0,
    ) == pytest.approx((0.5, 4.0))

    with pytest.raises(ValueError, match="too short"):
        module.validate_shift_range(
            duration_s=1.0,
            min_shift_s=0.5,
            max_shift_s=None,
        )


def test_interpolate_empirical_curve_fills_gaps_and_uses_mean_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    curve, used_fallback = module.interpolate_empirical_curve(
        np.asarray([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=float),
        fallback_rate_hz=0.5,
    )

    assert used_fallback is False
    assert curve == pytest.approx([1.0, 1.0, 2.0, 3.0, 3.0])

    curve, used_fallback = module.interpolate_empirical_curve(
        np.asarray([np.nan, np.nan], dtype=float),
        fallback_rate_hz=0.5,
    )

    assert used_fallback is True
    assert curve == pytest.approx([0.5, 0.5])


def test_compute_binned_empirical_tuning_curves_uses_count_occupancy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])
    train_inputs = {
        "y": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 2.0],
                [3.0, 1.0],
                [5.0, 5.0],
            ]
        ),
        "p": np.asarray([0.1, 0.2, 0.8, np.nan]),
    }

    curves = module.compute_binned_empirical_tuning_curves(
        train_inputs,
        bin_edges=np.asarray([0.0, 0.5, 1.0]),
        bin_size_s=0.5,
    )

    assert curves.shape == (2, 2)
    np.testing.assert_allclose(
        curves,
        np.asarray(
            [
                [1.0, 2.0],
                [6.0, 2.0],
            ]
        ),
    )


def test_make_score_rows_omits_turn_type_and_records_dark_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])
    rows = module.make_score_rows(
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        model="dark",
        fold=0,
        trajectory="center_to_left",
        unit_ids=np.asarray([11, 12]),
        metrics={
            "ll_sum": np.asarray([1.0, 2.0]),
            "null_ll_sum": np.asarray([0.5, 1.5]),
            "spike_sum": np.asarray([10.0, 20.0]),
            "bits_per_spike": np.asarray([0.1, 0.2]),
        },
        selected={
            "estimator": "empirical",
            "dark_model_scope": "trajectory",
        },
        bin_size_s=0.05,
        fallback_to_mean=np.asarray([True, False]),
    )

    assert "turn_type" not in rows[0]
    assert rows[0]["dark_model_scope"] == "trajectory"
    assert rows[0]["tuning_curve_fallback_to_mean"] is True
    assert rows[1]["tuning_curve_fallback_to_mean"] is False


def test_predict_poisson_counts_uses_constant_fallback_without_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])
    fit = {
        "model": None,
        "fit_unit_mask": np.asarray([False, False]),
        "fallback_mean_count": np.asarray([0.0, 0.25]),
    }

    predicted = module.predict_poisson_counts(
        fit,
        {"p": np.asarray([0.1, 0.2, 0.3])},
    )

    assert predicted.shape == (3, 2)
    assert predicted[:, 0] == pytest.approx(np.full(3, 1e-12))
    assert predicted[:, 1] == pytest.approx(np.full(3, 0.25))


def test_choose_hyperparameter_record_uses_finite_unit_minimum_and_tie_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    records = [
        {
            "score_median": 0.3,
            "n_finite_units": 4,
            "spatial_bin_size_cm": 2.0,
            "ridge": 0.1,
        },
        {
            "score_median": 0.2,
            "n_finite_units": 5,
            "spatial_bin_size_cm": 2.0,
            "ridge": 0.01,
        },
        {
            "score_median": 0.2,
            "n_finite_units": 5,
            "spatial_bin_size_cm": 4.0,
            "ridge": 0.01,
        },
        {
            "score_median": 0.2,
            "n_finite_units": 5,
            "spatial_bin_size_cm": 4.0,
            "ridge": 0.1,
        },
    ]

    selected = module.choose_hyperparameter_record(records, min_selection_units=5)

    assert selected["spatial_bin_size_cm"] == pytest.approx(4.0)
    assert selected["ridge"] == pytest.approx(0.1)


def test_compute_transfer_index_marks_unstable_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    value, reason = module.compute_transfer_index(
        light_bits=0.4,
        dark_bits=0.3,
        shuffle_bits=0.1,
    )
    assert value == pytest.approx(2.0 / 3.0)
    assert reason == "valid"

    value, reason = module.compute_transfer_index(
        light_bits=0.1,
        dark_bits=0.2,
        shuffle_bits=0.1,
    )
    assert np.isnan(value)
    assert reason == "light_not_above_shuffle"


def test_aggregate_shuffle_table_excludes_incomplete_shuffles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])
    shuffle_table = pd.DataFrame(
        [
            {
                "unit": 11,
                "shuffle_index": 0,
                "trajectory": "center_to_left",
                "ll_sum": 3.0,
                "null_ll_sum": 1.0,
                "spike_sum": 4.0,
            },
            {
                "unit": 11,
                "shuffle_index": 0,
                "trajectory": "right_to_center",
                "ll_sum": 5.0,
                "null_ll_sum": 2.0,
                "spike_sum": 6.0,
            },
            {
                "unit": 11,
                "shuffle_index": 1,
                "trajectory": "center_to_left",
                "ll_sum": 100.0,
                "null_ll_sum": 1.0,
                "spike_sum": 4.0,
            },
        ]
    )

    by_index, summary = module._aggregate_shuffle_table(
        shuffle_table,
        expected_components=2,
    )

    assert by_index["complete"].tolist() == [True, False]
    assert summary["unit"].tolist() == [11]
    assert summary["shuffle_complete_count"].tolist() == [1]
    expected_bits = (8.0 - 3.0) / (10.0 * np.log(2.0))
    assert summary["shuffle_mean_bits_per_spike"].iloc[0] == pytest.approx(
        expected_bits
    )


def test_output_stem_includes_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_module(monkeypatch, ["dark_light_transfer.py"])

    assert module._output_stem("v1", "02_r1", "08_r4", "empirical") == (
        "v1_02_r1_light_08_r4_dark_transfer_empirical"
    )
