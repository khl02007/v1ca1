from __future__ import annotations

import numpy as np
import pandas as pd

from v1ca1.ripple.ripple_decoding_glm import (
    build_expected_rate_matrix,
    build_metric_summary_table,
    build_output_stem,
    build_ripple_fold_masks,
    build_v1_unit_mask_table,
    interpolate_nans_1d,
    plot_metric_scatter_with_marginals,
    select_decode_epochs,
    shuffle_response_by_ripple_blocks,
    validate_v1_tuning_epoch,
)


class FakeSelectedTuningCurves:
    def __init__(self, array: np.ndarray) -> None:
        self._array = np.asarray(array, dtype=float)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._array, dtype=float)


class FakeTuningCurves:
    def __init__(self, unit_ids: list[int], array: np.ndarray) -> None:
        self._unit_ids = np.asarray(unit_ids)
        self._array = np.asarray(array, dtype=float)

    def sel(self, *, unit: list[int]) -> FakeSelectedTuningCurves:
        indices = [int(np.where(self._unit_ids == unit_id)[0][0]) for unit_id in unit]
        return FakeSelectedTuningCurves(self._array[indices])


def _make_fit_results(n_units: int) -> dict[str, np.ndarray]:
    base_real = np.array(
        [
            [0.2, 0.4][:n_units],
            [0.4, 0.6][:n_units],
        ],
        dtype=float,
    )
    base_shuffle = np.array(
        [
            [[0.0, 0.7][:n_units], [0.1, 0.8][:n_units]],
            [[0.2, 0.7][:n_units], [0.1, 0.8][:n_units]],
        ],
        dtype=float,
    )
    return {
        "pseudo_r2_folds": base_real,
        "pseudo_r2_shuff_folds": base_shuffle,
        "devexp_folds": base_real + 0.1,
        "devexp_shuff_folds": base_shuffle + 0.1,
        "bits_per_spike_folds": base_real + 0.2,
        "bits_per_spike_shuff_folds": base_shuffle + 0.2,
    }


def test_select_decode_epochs_defaults_to_all_run_epochs() -> None:
    run_epochs = ["02_r1", "04_r2", "08_r4"]
    assert select_decode_epochs(run_epochs, None) == run_epochs


def test_select_decode_epochs_rejects_non_run_epochs() -> None:
    run_epochs = ["02_r1", "04_r2", "08_r4"]
    try:
        select_decode_epochs(run_epochs, ["02_r1", "01_s1"])
    except ValueError as exc:
        assert "run epochs" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-run decode epoch.")


def test_validate_v1_tuning_epoch_rejects_unknown_epoch() -> None:
    try:
        validate_v1_tuning_epoch(["02_r1", "08_r4"], "04_r2")
    except ValueError as exc:
        assert "--v1-tuning-epoch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid V1 tuning epoch.")


def test_interpolate_nans_1d_fills_internal_and_edge_nans() -> None:
    values = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype=float)
    interpolated = interpolate_nans_1d(values)
    assert np.allclose(interpolated, [1.0, 1.0, 2.0, 3.0, 3.0])


def test_build_expected_rate_matrix_interpolates_curves_and_zeroes_all_nan_units() -> None:
    tuning_curves = FakeTuningCurves(
        unit_ids=[11, 12, 13],
        array=np.array(
            [
                [np.nan, 1.0, 3.0],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )
    expected_rate = build_expected_rate_matrix(
        tuning_curves=tuning_curves,
        unit_ids=np.array([11, 12, 13]),
        decoded_state=np.array([0.5, 1.5, 2.5], dtype=float),
        bin_edges=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
    )

    assert expected_rate.shape == (3, 3)
    assert np.allclose(expected_rate[:, 0], [1.0, 1.0, 3.0])
    assert np.allclose(expected_rate[:, 1], 0.0)
    assert np.allclose(expected_rate[:, 2], 0.0)


def test_build_v1_unit_mask_table_applies_all_thresholds() -> None:
    table = build_v1_unit_mask_table(
        unit_ids=np.array([11, 12, 13]),
        movement_firing_rates_hz=np.array([0.6, 0.4, 0.7], dtype=float),
        ripple_counts=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
        expected_rate_hz=np.array(
            [
                [1.0, 1.0, 0.0],
                [1.0, 1.0, np.nan],
            ],
            dtype=float,
        ),
        bin_size_s=1.0,
        min_movement_fr_hz=0.5,
        min_ripple_fr_hz=0.1,
    )

    assert table["passes_movement_firing_rate"].tolist() == [True, False, True]
    assert table["passes_ripple_firing_rate"].tolist() == [True, True, False]
    assert table["passes_expected_rate_predictor"].tolist() == [True, True, False]
    assert table["keep_unit"].tolist() == [True, False, False]


def test_build_ripple_fold_masks_keeps_each_ripple_in_one_fold() -> None:
    ripple_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
    folds = build_ripple_fold_masks(ripple_ids, n_splits=2)

    assert len(folds) == 2
    for train_mask, test_mask in folds:
        train_ripples = set(ripple_ids[train_mask].tolist())
        test_ripples = set(ripple_ids[test_mask].tolist())
        assert train_ripples.isdisjoint(test_ripples)
        for ripple_id in np.unique(ripple_ids):
            in_train = train_mask[ripple_ids == ripple_id]
            in_test = test_mask[ripple_ids == ripple_id]
            assert np.all(in_train) or np.all(in_test)


def test_shuffle_response_by_ripple_blocks_preserves_block_order_within_each_ripple() -> None:
    response = np.array([[10.0], [11.0], [20.0], [21.0], [30.0], [31.0]], dtype=float)
    ripple_ids = np.array([0, 0, 1, 1, 2, 2], dtype=int)
    shuffled = shuffle_response_by_ripple_blocks(
        response,
        ripple_ids,
        np.random.default_rng(0),
    )

    original_blocks = {
        tuple(block.ravel().tolist())
        for block in (
            response[ripple_ids == 0],
            response[ripple_ids == 1],
            response[ripple_ids == 2],
        )
    }
    shuffled_blocks = {
        tuple(shuffled[index : index + 2].ravel().tolist())
        for index in range(0, shuffled.shape[0], 2)
    }
    assert shuffled_blocks == original_blocks
    assert not np.array_equal(shuffled, response)


def test_build_metric_summary_table_includes_epoch_ids_and_nan_for_dropped_units() -> None:
    unit_mask_table = pd.DataFrame(
        {
            "unit_id": [11, 12, 13],
            "movement_firing_rate_hz": [0.6, 0.4, 0.7],
            "ripple_firing_rate_hz": [0.2, 0.2, 0.05],
            "expected_rate_mean_hz": [1.0, 1.0, 0.0],
            "expected_rate_max_hz": [2.0, 2.0, 0.0],
            "passes_movement_firing_rate": [True, False, True],
            "passes_ripple_firing_rate": [True, True, False],
            "passes_expected_rate_predictor": [True, True, False],
            "keep_unit": [True, False, False],
        }
    )

    table = build_metric_summary_table(
        unit_mask_table=unit_mask_table,
        kept_unit_ids=np.array([11]),
        fit_results=_make_fit_results(1),
        representation="place",
        v1_tuning_epoch="08_r4",
        decode_epoch="02_r1",
        n_ripples=12,
        n_ripple_bins=48,
    )

    assert table["representation"].tolist() == ["place", "place", "place"]
    assert table["v1_tuning_epoch"].tolist() == ["08_r4", "08_r4", "08_r4"]
    assert table["decode_epoch"].tolist() == ["02_r1", "02_r1", "02_r1"]
    assert np.isfinite(table.loc[0, "pseudo_r2"])
    assert np.isnan(table.loc[1, "pseudo_r2"])
    assert np.isnan(table.loc[2, "pseudo_r2"])
    assert np.isfinite(table.loc[0, "pseudo_r2_p_value"])


def test_build_output_stem_includes_both_epochs() -> None:
    stem = build_output_stem(
        representation="task_progression",
        v1_tuning_epoch="08_r4",
        decode_epoch="02_r1",
    )
    assert "v1train-08_r4" in stem
    assert "ca1train-02_r1_decode-02_r1" in stem


def test_plot_metric_scatter_with_marginals_writes_file(tmp_path) -> None:
    output_path = tmp_path / "metric.png"
    written_path = plot_metric_scatter_with_marginals(
        metric_values=np.array([0.1, 0.2, 0.3], dtype=float),
        p_values=np.array([0.05, 0.5, 0.9], dtype=float),
        animal_name="L14",
        date="20240611",
        representation="place",
        v1_tuning_epoch="08_r4",
        decode_epoch="02_r1",
        metric_label="Pseudo R^2",
        out_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()
