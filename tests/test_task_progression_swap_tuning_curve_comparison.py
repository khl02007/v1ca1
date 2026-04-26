from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


MODULE_NAME = "v1ca1.task_progression.swap_tuning_curve_comparison"


def _reload_module(monkeypatch: pytest.MonkeyPatch, argv: list[str]):
    monkeypatch.setattr(sys, "argv", list(argv))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _fake_result(module):
    unit_ids = np.asarray([11, 12], dtype=int)
    n_trajectories = len(module.TRAJECTORY_TYPES)
    n_bins = 3
    n_units = unit_ids.size
    metrics = {
        "test_light_spike_sum": np.full((n_trajectories, n_units), 10.0),
        "ll_task_empirical_sum": np.full((n_trajectories, n_units), -20.0),
        "ll_visual_empirical_sum": np.full((n_trajectories, n_units), -22.0),
        "ll_task_empirical_per_spike": np.full((n_trajectories, n_units), -2.0),
        "ll_visual_empirical_per_spike": np.full((n_trajectories, n_units), -2.2),
        "delta_ll_sum_task_vs_visual": np.full(
            (n_trajectories, n_units),
            2.0,
        ),
        "delta_ll_bits_task_vs_visual": np.full(
            (n_trajectories, n_units),
            2.0 / np.log(2.0),
        ),
        "delta_ll_bits_per_spike_task_vs_visual": np.full(
            (n_trajectories, n_units),
            0.2 / np.log(2.0),
        ),
        "corr_task_empirical": np.full((n_trajectories, n_units), 0.5),
        "corr_visual_empirical": np.full((n_trajectories, n_units), 0.1),
        "delta_corr_task_vs_visual": np.full(
            (n_trajectories, n_units),
            0.4,
        ),
    }
    segment_edges = np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float)
    swap_segment_index = np.asarray(
        [module.SWAP_CONFIG[trajectory]["segment_index"] for trajectory in module.TRAJECTORY_TYPES],
        dtype=int,
    )
    return {
        "unit_ids": unit_ids,
        "bin_edges": np.asarray([0.0, 0.33, 0.66, 1.0], dtype=float),
        "bin_centers": np.asarray([0.165, 0.495, 0.83], dtype=float),
        "segment_edges": segment_edges,
        "swap_source_trajectory": np.asarray(
            [
                module.SWAP_CONFIG[trajectory]["source_trajectory"]
                for trajectory in module.TRAJECTORY_TYPES
            ],
            dtype=object,
        ),
        "swap_segment_index": swap_segment_index,
        "segment_bin_mask": np.ones((n_trajectories, n_bins), dtype=bool),
        "same_dark_tuning": np.ones((n_trajectories, n_bins, n_units), dtype=float),
        "other_dark_tuning": np.full(
            (n_trajectories, n_bins, n_units),
            0.5,
            dtype=float,
        ),
        "other_light_tuning": np.full(
            (n_trajectories, n_bins, n_units),
            2.0,
            dtype=float,
        ),
        "task_empirical_tuning": np.full(
            (n_trajectories, n_bins, n_units),
            4.0,
            dtype=float,
        ),
        "visual_empirical_tuning": np.full(
            (n_trajectories, n_bins, n_units),
            2.0,
            dtype=float,
        ),
        "test_light_tuning": np.full(
            (n_trajectories, n_bins, n_units),
            3.0,
            dtype=float,
        ),
        "train_dark_same_rate_hz": np.full((n_trajectories, n_units), 1.0),
        "train_dark_other_rate_hz": np.full((n_trajectories, n_units), 0.5),
        "train_light_other_rate_hz": np.full((n_trajectories, n_units), 2.0),
        "test_light_target_rate_hz": np.full((n_trajectories, n_units), 3.0),
        "test_light_bin_count": np.full((n_trajectories,), 5.0),
        "test_light_duration_s": np.full((n_trajectories,), 0.1),
        "metrics": metrics,
    }


def test_parse_arguments_defaults_to_fr_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_module(
        monkeypatch,
        [
            "swap_tuning_curve_comparison.py",
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

    args = module.parse_arguments()

    assert args.apply_fr_filter is True
    assert args.v1_min_dark_fr_hz == pytest.approx(0.5)
    assert args.v1_min_light_fr_hz == pytest.approx(0.5)
    assert args.ca1_min_dark_fr_hz == pytest.approx(0.0)
    assert args.ca1_min_light_fr_hz == pytest.approx(0.0)


def test_parse_arguments_supports_disabling_fr_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(
        monkeypatch,
        [
            "swap_tuning_curve_comparison.py",
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
            "--no-fr-filter",
        ],
    )

    assert module.parse_arguments().apply_fr_filter is False


def test_swap_config_uses_last_segment_for_outbound_and_first_for_inbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])

    assert module.SWAP_CONFIG["center_to_left"]["source_trajectory"] == "center_to_right"
    assert module.SWAP_CONFIG["center_to_left"]["segment_index"] == 2
    assert module.SWAP_CONFIG["center_to_right"]["source_trajectory"] == "center_to_left"
    assert module.SWAP_CONFIG["center_to_right"]["segment_index"] == 2
    assert module.SWAP_CONFIG["left_to_center"]["source_trajectory"] == "right_to_center"
    assert module.SWAP_CONFIG["left_to_center"]["segment_index"] == 0
    assert module.SWAP_CONFIG["right_to_center"]["source_trajectory"] == "left_to_center"
    assert module.SWAP_CONFIG["right_to_center"]["segment_index"] == 0


def test_segment_mask_excludes_internal_right_edge_but_includes_final_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])
    values = np.asarray([0.0, 0.29, 0.3, 0.7, 1.0], dtype=float)
    edges = np.asarray([0.0, 0.3, 0.7, 1.0], dtype=float)

    assert np.array_equal(
        module.segment_mask(values, edges, 0),
        np.asarray([True, True, False, False, False]),
    )
    assert np.array_equal(
        module.segment_mask(values, edges, 2),
        np.asarray([False, False, False, True, True]),
    )


def test_interpolate_nans_uses_linear_internal_and_nearest_edge_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])

    interpolated = module.interpolate_nans_1d(
        np.asarray([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=float),
        fallback_value=9.0,
    )

    assert np.allclose(interpolated, [1.0, 1.0, 2.0, 3.0, 3.0])
    assert np.allclose(
        module.interpolate_nans_1d(
            np.asarray([np.nan, np.nan], dtype=float),
            fallback_value=2.5,
        ),
        [2.5, 2.5],
    )


def test_smooth_interpolated_tuning_matrix_can_disable_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])

    matrix = module.smooth_interpolated_tuning_matrix(
        np.asarray(
            [
                [np.nan, 1.0],
                [np.nan, np.nan],
                [4.0, 3.0],
            ],
            dtype=float,
        ),
        fallback_rates_hz=np.asarray([2.0, 5.0], dtype=float),
        sigma_bins=0.0,
    )

    assert np.allclose(matrix[:, 0], [4.0, 4.0, 4.0])
    assert np.allclose(matrix[:, 1], [1.0, 2.0, 3.0])


def test_build_empirical_task_tuning_uses_same_dark_times_other_light_over_other_dark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])

    task_tuning = module.build_empirical_task_tuning(
        same_dark_tuning=np.asarray([[4.0, 2.0]], dtype=float),
        other_light_tuning=np.asarray([[6.0, 3.0]], dtype=float),
        other_dark_tuning=np.asarray([[2.0, 1.5]], dtype=float),
    )

    assert np.allclose(task_tuning, [[12.0, 4.0]])


def test_score_segment_binned_counts_delta_is_task_minus_visual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])

    score = module.score_segment_binned_counts(
        spike_counts=np.asarray([[20.0], [20.0], [0.0]], dtype=float),
        positions=np.asarray([0.75, 0.90, 0.25], dtype=float),
        task_empirical_tuning=np.asarray([[1.0], [20.0]], dtype=float),
        visual_empirical_tuning=np.asarray([[1.0], [2.0]], dtype=float),
        bin_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        segment_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        segment_index=1,
        bin_size_s=1.0,
    )

    assert score["test_light_spike_sum"][0] == pytest.approx(40.0)
    assert score["ll_task_empirical_sum"][0] > score["ll_visual_empirical_sum"][0]
    assert score["delta_ll_sum_task_vs_visual"][0] > 0.0
    assert score["delta_ll_bits_per_spike_task_vs_visual"][0] > 0.0


def test_compute_segment_correlations_uses_task_minus_visual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])

    correlations = module.compute_segment_correlations(
        test_light_tuning=np.asarray([[9.0], [8.0], [1.0], [2.0]], dtype=float),
        task_empirical_tuning=np.asarray([[9.0], [8.0], [2.0], [4.0]], dtype=float),
        visual_empirical_tuning=np.asarray([[9.0], [8.0], [4.0], [2.0]], dtype=float),
        bin_centers=np.asarray([0.1, 0.4, 0.7, 0.9], dtype=float),
        segment_edges=np.asarray([0.0, 0.5, 1.0], dtype=float),
        segment_index=1,
    )

    assert correlations["corr_task_empirical"][0] == pytest.approx(1.0)
    assert correlations["corr_visual_empirical"][0] == pytest.approx(-1.0)
    assert correlations["delta_corr_task_vs_visual"][0] == pytest.approx(2.0)


def test_build_results_table_and_dataset_preserve_metric_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_module(monkeypatch, ["swap_tuning_curve_comparison.py"])
    result = _fake_result(module)

    table = module.build_results_table(
        result,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_movement_firing_rates=np.asarray([0.6, 0.7], dtype=float),
        light_movement_firing_rates=np.asarray([0.8, 0.9], dtype=float),
        apply_fr_filter=True,
        min_dark_fr_hz=0.5,
        min_light_fr_hz=0.5,
    )
    dataset = module.build_region_dataset(
        result,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_train_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_movement_firing_rates=np.asarray([0.6, 0.7], dtype=float),
        light_movement_firing_rates=np.asarray([0.8, 0.9], dtype=float),
        bin_size_s=0.02,
        sigma_bins=1.0,
        place_bin_size_cm=2.0,
        apply_fr_filter=True,
        min_dark_fr_hz=0.5,
        min_light_fr_hz=0.5,
        sources={"position": "fake.parquet"},
    )

    assert table.shape[0] == len(module.TRAJECTORY_TYPES) * 2
    assert set(table["swap_segment_index_1based"]) == {1, 3}
    assert dataset.sizes["trajectory"] == len(module.TRAJECTORY_TYPES)
    assert dataset.sizes["unit"] == 2
    assert dataset.sizes["tp_bin"] == 3
    assert dataset["same_dark_train_tuning_hz"].shape == (
        len(module.TRAJECTORY_TYPES),
        3,
        2,
    )
    assert "task_empirical_tuning_hz" in dataset
    assert "delta_ll_bits_per_spike_task_vs_visual" in dataset
    assert (
        dataset.attrs["primary_ll_delta"]
        == "task_empirical_minus_visual_empirical_bits_per_spike"
    )
