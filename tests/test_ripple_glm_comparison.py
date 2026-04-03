from __future__ import annotations

import numpy as np
import pytest

from v1ca1.ripple.ripple_glm import build_epoch_fit_dataset
from v1ca1.ripple.ripple_glm_comparison import compare_runs, load_current_saved_output


def _example_results() -> dict[str, np.ndarray | int]:
    return {
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102, 103]),
        "coef_ca1_unit_ids": np.array([101, 103]),
        "n_ripples": 8,
        "ripple_start_time_s": np.array([1.0, 2.0, 3.0], dtype=float),
        "ripple_window_start_s": np.array([1.0, 2.0, 3.0], dtype=float),
        "ripple_window_end_s": np.array([1.2, 2.2, 3.2], dtype=float),
        "ripple_fold_index": np.array([0, 1, 0], dtype=int),
        "ripple_observed_count_oof": np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]], dtype=float),
        "ripple_predicted_count_oof": np.array([[0.8, 0.1], [0.2, 1.8], [2.7, 1.2]], dtype=float),
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


def test_load_current_saved_output_reads_ripple_only_dataset(tmp_path) -> None:
    pytest.importorskip("xarray")

    dataset = build_epoch_fit_dataset(
        _example_results(),
        animal_name="L14",
        date="20240611",
        epoch="01_s1",
        sources={"ripple_events": "pynapple"},
        fit_parameters={
            "n_splits": 2,
            "ridge_strength": 0.1,
            "ripple_window_s": 0.2,
            "ripple_window_offset_s": 0.0,
            "ripple_selection_mode": "allripples",
            "n_ripples_before_selection": 8,
            "n_ripples_removed_by_selection": 0,
            "n_ripples_after_selection": 8,
        },
    )
    output_path = tmp_path / "01_s1_rw_0p2s_allripples_ridge_1e-1_samplewise_ripple_glm.nc"
    dataset.to_netcdf(output_path)

    loaded = load_current_saved_output(
        output_path,
        label="current_artifacts_current_code_saved_output",
        input_summary={
            "analysis_path": str(tmp_path),
            "sources": {"ripple_events": "pynapple"},
            "raw_ripple_rows": 8,
            "fixed_window_ripple_count": 8,
            "ripple_table_columns": ["start_time", "end_time"],
        },
    )

    assert loaded["n_ripples"] == 8
    assert set(loaded["metric_arrays"]) == {"ripple"}
    assert np.allclose(
        loaded["metric_arrays"]["ripple"]["pseudo_r2"],
        np.array([[0.2, 0.4], [0.4, 0.6]], dtype=float),
    )


def test_compare_runs_compares_ripple_metrics_only() -> None:
    metric_arrays = {
        "ripple": {
            "pseudo_r2": np.array([[0.2, 0.4], [0.4, 0.6]], dtype=float),
            "mae": np.array([[0.3, 0.7], [0.5, 0.9]], dtype=float),
            "devexp": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
            "bits_per_spike": np.array([[0.5, 0.7], [0.7, 0.9]], dtype=float),
        }
    }
    left = {
        "label": "left",
        "n_ripples": 8,
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102]),
        "metric_arrays": metric_arrays,
    }
    right = {
        "label": "right",
        "n_ripples": 8,
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102]),
        "metric_arrays": metric_arrays,
    }

    comparison = compare_runs(left, right)

    assert comparison["comparable"] is True
    assert set(comparison["metric_diffs"]) == {"ripple"}
    assert comparison["metric_diffs"]["ripple"]["pseudo_r2"]["allclose"] is True
