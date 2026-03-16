from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from v1ca1.ripple.ripple_decoding_comparison import (
    build_epoch_dataset,
    build_epoch_summary_table,
    build_per_ripple_metric_table,
    build_region_unit_mask_table,
    compute_per_ripple_metrics,
    resolve_epoch_pairs,
    resolve_train_epoch,
    shuffle_ripple_state_blocks_by_length,
    validate_train_epoch,
)


def _make_aligned_data() -> dict[str, np.ndarray | int]:
    return {
        "ca1_decoded_state": np.array([0.0, 1.0, 1.0, 0.0], dtype=float),
        "v1_decoded_state": np.array([0.0, 1.0, 1.0, 0.0], dtype=float),
        "bin_times_s": np.array([1.0, 1.002, 2.0, 2.002], dtype=float),
        "ripple_ids": np.array([0, 0, 1, 1], dtype=int),
        "n_ripples": 2,
        "n_bins": 4,
        "ripple_source_indices": np.array([5, 7], dtype=int),
        "ripple_start_times_s": np.array([0.99, 1.99], dtype=float),
        "ripple_end_times_s": np.array([1.01, 2.01], dtype=float),
        "skipped_ripples": [],
    }


def test_resolve_train_epoch_defaults_to_decode_epoch() -> None:
    assert resolve_train_epoch("04_r2", None) == "04_r2"
    assert resolve_train_epoch("04_r2", "08_r4") == "08_r4"


def test_validate_train_epoch_rejects_unknown_epoch() -> None:
    with pytest.raises(ValueError, match="--train-epoch"):
        validate_train_epoch(["02_r1", "08_r4"], "04_r2")


def test_resolve_epoch_pairs_defaults_to_matching_all_run_epochs() -> None:
    assert resolve_epoch_pairs(
        ["02_r1", "04_r2", "08_r4"],
        requested_decode_epoch=None,
        requested_decode_epochs=None,
        requested_train_epoch=None,
    ) == [
        ("02_r1", "02_r1"),
        ("04_r2", "04_r2"),
        ("08_r4", "08_r4"),
    ]


def test_resolve_epoch_pairs_pairs_single_epoch_with_itself_when_only_one_side_is_given() -> None:
    assert resolve_epoch_pairs(
        ["02_r1", "04_r2", "08_r4"],
        requested_decode_epoch="04_r2",
        requested_decode_epochs=None,
        requested_train_epoch=None,
    ) == [("04_r2", "04_r2")]
    assert resolve_epoch_pairs(
        ["02_r1", "04_r2", "08_r4"],
        requested_decode_epoch=None,
        requested_decode_epochs=None,
        requested_train_epoch="08_r4",
    ) == [("08_r4", "08_r4")]


def test_resolve_epoch_pairs_supports_fixed_train_epoch_with_explicit_decode_epochs() -> None:
    assert resolve_epoch_pairs(
        ["02_r1", "04_r2", "08_r4"],
        requested_decode_epoch=None,
        requested_decode_epochs=["02_r1", "04_r2"],
        requested_train_epoch="08_r4",
    ) == [
        ("02_r1", "08_r4"),
        ("04_r2", "08_r4"),
    ]


def test_shuffle_ripple_state_blocks_by_length_preserves_within_block_order_and_lengths() -> None:
    state = np.array([10.0, 11.0, 20.0, 21.0, 22.0, 30.0, 31.0], dtype=float)
    ripple_ids = np.array([0, 0, 1, 1, 1, 2, 2], dtype=int)

    shuffled, changed = shuffle_ripple_state_blocks_by_length(
        state,
        ripple_ids,
        np.random.default_rng(0),
    )

    assert changed
    original_blocks = {
        2: {(10.0, 11.0), (30.0, 31.0)},
        3: {(20.0, 21.0, 22.0)},
    }
    shuffled_blocks = [
        tuple(shuffled[ripple_ids == ripple_id].tolist())
        for ripple_id in np.unique(ripple_ids)
    ]
    assert shuffled_blocks[0] in original_blocks[2]
    assert shuffled_blocks[1] in original_blocks[3]
    assert shuffled_blocks[2] in original_blocks[2]
    assert shuffled_blocks[1] == (20.0, 21.0, 22.0)


def test_compute_per_ripple_metrics_handles_normal_constant_and_single_bin_inputs() -> None:
    normal_metrics = compute_per_ripple_metrics(
        ca1_state=np.array([0.0, 1.0, 2.0], dtype=float),
        v1_state=np.array([0.0, 1.0, 2.0], dtype=float),
        state_span=2.0,
    )
    assert np.isclose(normal_metrics["pearson_r"], 1.0)
    assert np.isclose(normal_metrics["mean_abs_difference"], 0.0)
    assert np.isclose(normal_metrics["mean_abs_difference_normalized"], 0.0)

    constant_metrics = compute_per_ripple_metrics(
        ca1_state=np.array([1.0, 1.0, 1.0], dtype=float),
        v1_state=np.array([1.0, 1.0, 1.0], dtype=float),
        state_span=1.0,
    )
    assert np.isnan(constant_metrics["pearson_r"])

    single_bin_metrics = compute_per_ripple_metrics(
        ca1_state=np.array([1.0], dtype=float),
        v1_state=np.array([1.5], dtype=float),
        state_span=2.0,
    )
    assert np.isnan(single_bin_metrics["pearson_r"])
    assert np.isclose(single_bin_metrics["start_difference"], 0.5)
    assert np.isclose(single_bin_metrics["end_difference"], 0.5)


def test_build_epoch_summary_table_reports_expected_higher_and_lower_p_values() -> None:
    aligned_data = _make_aligned_data()
    aligned_data["v1_decoded_state"] = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    ripple_table = pd.DataFrame(
        {
            "start_time": [0.99, 1.99],
            "end_time": [1.01, 2.01],
        }
    )

    summary_table, shuffle_samples, effective_shuffles = build_epoch_summary_table(
        aligned_data=aligned_data,
        ripple_table=ripple_table,
        representation="place",
        train_epoch="08_r4",
        decode_epoch="02_r1",
        state_span=2.0,
        n_shuffles=4,
        shuffle_seed=0,
    )

    row = summary_table.iloc[0]
    assert effective_shuffles == 4
    assert np.isclose(row["pearson_r"], 1.0)
    assert np.isclose(row["mean_abs_difference"], 0.0)
    assert np.isclose(row["pearson_r_p_value"], 0.2)
    assert np.isclose(row["mean_abs_difference_p_value"], 0.2)
    assert np.allclose(shuffle_samples["pearson_r"][np.isfinite(shuffle_samples["pearson_r"])], -1.0)
    assert np.allclose(
        shuffle_samples["mean_abs_difference"][
            np.isfinite(shuffle_samples["mean_abs_difference"])
        ],
        1.0,
    )


def test_build_epoch_dataset_contains_decoded_states_masks_and_shuffle_outputs() -> None:
    xr = pytest.importorskip("xarray")
    aligned_data = _make_aligned_data()
    per_ripple_table = build_per_ripple_metric_table(
        aligned_data=aligned_data,
        representation="place",
        train_epoch="08_r4",
        decode_epoch="02_r1",
        state_span=2.0,
    )
    summary_table = pd.DataFrame(
        [
            {
                "representation": "place",
                "train_epoch": "08_r4",
                "decode_epoch": "02_r1",
                "n_ripples": 2,
                "n_ripple_bins": 4,
                "n_ripple_events_input": 2,
                "n_effective_shuffles": 2,
                "pearson_r": 1.0,
                "pearson_r_shuffle_mean": -1.0,
                "pearson_r_shuffle_sd": 0.0,
                "pearson_r_p_value": 1.0 / 3.0,
                "mean_abs_difference": 0.0,
                "mean_abs_difference_shuffle_mean": 1.0,
                "mean_abs_difference_shuffle_sd": 0.0,
                "mean_abs_difference_p_value": 1.0 / 3.0,
                "mean_abs_difference_normalized": 0.0,
                "mean_abs_difference_normalized_shuffle_mean": 0.5,
                "mean_abs_difference_normalized_shuffle_sd": 0.0,
                "mean_abs_difference_normalized_p_value": 1.0 / 3.0,
                "mean_signed_difference": 0.0,
                "mean_signed_difference_shuffle_mean": 0.0,
                "mean_signed_difference_shuffle_sd": 0.0,
                "mean_signed_difference_p_value": 1.0,
                "start_difference": 0.0,
                "start_difference_shuffle_mean": 0.0,
                "start_difference_shuffle_sd": 0.0,
                "start_difference_p_value": 1.0,
                "end_difference": 0.0,
                "end_difference_shuffle_mean": 0.0,
                "end_difference_shuffle_sd": 0.0,
                "end_difference_p_value": 1.0,
            }
        ]
    )
    shuffle_samples = {
        "pearson_r": np.array([-1.0, -1.0], dtype=float),
        "mean_abs_difference": np.array([1.0, 1.0], dtype=float),
        "mean_abs_difference_normalized": np.array([0.5, 0.5], dtype=float),
        "mean_signed_difference": np.array([0.0, 0.0], dtype=float),
        "start_difference": np.array([0.0, 0.0], dtype=float),
        "end_difference": np.array([0.0, 0.0], dtype=float),
    }
    ca1_mask_table = build_region_unit_mask_table(
        unit_ids=np.array([11, 12]),
        movement_firing_rates_hz=np.array([0.7, 0.3], dtype=float),
        min_movement_fr_hz=0.5,
        region="ca1",
    )
    v1_mask_table = build_region_unit_mask_table(
        unit_ids=np.array([21, 22, 23]),
        movement_firing_rates_hz=np.array([0.8, 0.9, 0.2], dtype=float),
        min_movement_fr_hz=0.5,
        region="v1",
    )

    dataset = build_epoch_dataset(
        aligned_data=aligned_data,
        ca1_mask_table=ca1_mask_table,
        v1_mask_table=v1_mask_table,
        per_ripple_table=per_ripple_table,
        summary_table=summary_table,
        shuffle_samples=shuffle_samples,
        animal_name="L14",
        date="20240611",
        representation="place",
        train_epoch="08_r4",
        decode_epoch="02_r1",
        bin_size_s=0.002,
        sources={"ripple_events": "table.parquet"},
        skipped_ripples={"ca1": [], "v1": [], "alignment": []},
        fit_parameters={"bin_size_s": 0.002},
    )

    assert isinstance(dataset, xr.Dataset)
    assert np.allclose(dataset["ca1_decoded_state"].to_numpy(), [0.0, 1.0, 1.0, 0.0])
    assert np.allclose(dataset["v1_decoded_state"].to_numpy(), [0.0, 1.0, 1.0, 0.0])
    assert dataset["pearson_r_shuffle"].shape == (2,)
    assert dataset["ca1_keep_unit"].to_numpy().tolist() == [True, False]
    assert dataset["v1_keep_unit"].to_numpy().tolist() == [True, True, False]
    assert dataset.coords["ca1_unit"].to_numpy().tolist() == [11, 12]
    assert dataset.coords["v1_unit"].to_numpy().tolist() == [21, 22, 23]
    assert dataset.attrs["decode_epoch"] == "02_r1"
    assert json.loads(dataset.attrs["sources_json"]) == {"ripple_events": "table.parquet"}
