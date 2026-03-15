from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from v1ca1.ripple.ripple_glm import (
    build_unit_summary_table,
    get_epoch_skip_reason,
    load_ripple_tables_from_interval_output,
    load_ripple_tables_from_legacy_pickle,
    make_preripple_ep,
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


def test_build_unit_summary_table_aggregates_fold_metrics() -> None:
    results = {
        "v1_unit_ids": np.array([11, 12]),
        "ca1_unit_ids": np.array([101, 102, 103]),
        "n_ripples": 8,
        "n_pre": 6,
        "pseudo_r2_ripple_folds": np.array([[0.2, 0.4], [0.4, 0.6]], dtype=float),
        "mae_ripple_folds": np.array([[0.3, 0.7], [0.5, 0.9]], dtype=float),
        "ll_ripple_folds": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float),
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
        "ll_ripple_shuff_folds": np.zeros((2, 2, 2), dtype=float),
        "devexp_ripple_shuff_folds": np.zeros((2, 2, 2), dtype=float),
        "bits_per_spike_ripple_shuff_folds": np.array(
            [
                [[0.1, 0.8], [0.2, 0.9]],
                [[0.1, 0.8], [0.2, 0.9]],
            ],
            dtype=float,
        ),
        "pseudo_r2_pre": np.array([0.05, 0.06], dtype=float),
        "mae_pre": np.array([0.4, 0.5], dtype=float),
        "ll_pre": np.array([1.5, 2.5], dtype=float),
        "devexp_pre": np.array([0.02, 0.03], dtype=float),
        "bits_per_spike_pre": np.array([0.15, 0.25], dtype=float),
    }

    summary = build_unit_summary_table(
        results,
        animal_name="L14",
        date="20240611",
        epoch="01_s1",
    )

    assert list(summary["v1_unit_id"]) == [11, 12]
    assert np.allclose(summary["ripple_pseudo_r2_mean"], [0.3, 0.5])
    assert np.allclose(summary["ripple_mae_mean"], [0.4, 0.8])
    assert np.allclose(summary["ripple_bits_per_spike_mean"], [0.6, 0.8])
    assert np.allclose(summary["pre_bits_per_spike"], [0.15, 0.25])
    assert np.allclose(summary["ripple_pseudo_r2_p_value"], [1.0 / 3.0, 1.0])
    assert np.allclose(summary["ripple_mae_p_value"], [1.0 / 3.0, 1.0])
    assert np.allclose(summary["ripple_bits_per_spike_p_value"], [1.0 / 3.0, 1.0])


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
