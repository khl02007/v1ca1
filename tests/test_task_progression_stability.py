from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


MODULE_NAME = "v1ca1.task_progression.stability"


class _FakeIntervalSet:
    """Minimal IntervalSet-like object for parity split tests."""

    def __init__(self, start, end, time_units: str = "s") -> None:
        del time_units
        self.start = np.asarray(start, dtype=float).reshape(-1)
        self.end = np.asarray(end, dtype=float).reshape(-1)


def _reload_stability_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _make_curve(values_by_unit: dict[int, list[float]]) -> xr.DataArray:
    units = np.asarray(sorted(values_by_unit), dtype=int)
    values = np.asarray([values_by_unit[unit] for unit in units], dtype=float)
    return xr.DataArray(
        values,
        dims=("unit", "tp"),
        coords={
            "unit": units,
            "tp": np.arange(values.shape[1], dtype=float),
        },
    )


def test_split_laps_by_odd_even_uses_one_indexed_trial_order() -> None:
    module = _reload_stability_module()

    intervals = _FakeIntervalSet(
        start=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        end=np.asarray([1.5, 2.5, 3.5, 4.5, 5.5]),
    )

    split = module.split_laps_by_odd_even(intervals)

    assert split["odd_indices"].tolist() == [0, 2, 4]
    assert split["even_indices"].tolist() == [1, 3]
    assert np.allclose(split["odd_interval"].start, [1.0, 3.0, 5.0])
    assert np.allclose(split["odd_interval"].end, [1.5, 3.5, 5.5])
    assert np.allclose(split["even_interval"].start, [2.0, 4.0])
    assert np.allclose(split["even_interval"].end, [2.5, 4.5])


def test_make_fraction_histogram_weights_sum_to_one() -> None:
    module = _reload_stability_module()

    weights = module.make_fraction_histogram_weights(np.asarray([-0.5, 0.0, 0.5, 1.0]))

    assert np.allclose(weights, [0.25, 0.25, 0.25, 0.25])
    assert np.allclose(weights.sum(), 1.0)
    assert module.make_fraction_histogram_weights(np.asarray([])).size == 0


def test_build_stability_table_retains_all_units_without_firing_rate_filter() -> None:
    module = _reload_stability_module()

    odd_curve = _make_curve(
        {
            11: [1.0, 2.0, 3.0],
            12: [1.0, 2.0, 3.0],
            13: [4.0, 4.0, 4.0],
        }
    )
    even_curve = _make_curve(
        {
            11: [2.0, 4.0, 6.0],
            12: [3.0, 2.0, 1.0],
            13: [1.0, 2.0, 3.0],
        }
    )
    epoch_firing_rates = pd.Series({11: 0.1, 12: 0.0, 13: 2.0}, dtype=float)

    table = module.build_stability_table_for_tuning_curves(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        expected_unit_ids=[11, 12, 13],
        odd_tuning_curve=odd_curve,
        even_tuning_curve=even_curve,
        epoch_firing_rates=epoch_firing_rates,
        odd_spike_counts=pd.Series({11: 4, 12: 3, 13: 2}, dtype=int),
        even_spike_counts=pd.Series({11: 3, 12: 2, 13: 1}, dtype=int),
        n_odd_trials=3,
        n_even_trials=2,
        odd_duration_s=12.0,
        even_duration_s=8.0,
        n_odd_feature_samples=120,
        n_even_feature_samples=80,
    )

    assert table["unit"].tolist() == [11, 12, 13]
    assert np.allclose(table.loc[0, "stability_correlation"], 1.0)
    assert np.allclose(table.loc[1, "stability_correlation"], -1.0)
    assert np.isnan(table.loc[2, "stability_correlation"])
    assert table["stability_status"].tolist() == [
        "valid",
        "valid",
        "constant_odd_curve",
    ]
    assert np.allclose(table["firing_rate_hz"], [0.1, 0.0, 2.0])
    assert table["n_odd_trials"].tolist() == [3, 3, 3]
    assert table["n_even_trials"].tolist() == [2, 2, 2]
    assert table["n_odd_spikes"].tolist() == [4, 3, 2]
    assert table["n_even_spikes"].tolist() == [3, 2, 1]
    assert table["n_paired_finite_bins"].tolist() == [3, 3, 3]


def test_build_stability_table_retains_unavailable_trajectory_rows() -> None:
    module = _reload_stability_module()

    table = module.build_stability_table_for_tuning_curves(
        animal_name="L14",
        date="20240611",
        region="ca1",
        epoch="02_r1",
        trajectory_type="center_to_right",
        expected_unit_ids=[21, 22],
        odd_tuning_curve=None,
        even_tuning_curve=None,
        epoch_firing_rates=pd.Series({21: 0.0, 22: 1.5}, dtype=float),
        odd_spike_counts=pd.Series({21: 0, 22: 2}, dtype=int),
        even_spike_counts=pd.Series({21: 0, 22: 0}, dtype=int),
        n_odd_trials=1,
        n_even_trials=0,
        odd_duration_s=4.0,
        even_duration_s=0.0,
        n_odd_feature_samples=40,
        n_even_feature_samples=0,
        trajectory_status="no_even_trials",
    )

    assert table["unit"].tolist() == [21, 22]
    assert table["stability_correlation"].isna().all()
    assert table["stability_status"].tolist() == [
        "no_even_trials",
        "no_even_trials",
    ]
    assert table["n_even_trials"].tolist() == [0, 0]
    assert table["n_even_finite_bins"].tolist() == [0, 0]


def test_build_stability_table_normalizes_infinite_curve_to_invalid_nan() -> None:
    module = _reload_stability_module()

    table = module.build_stability_table_for_tuning_curves(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="left_to_center",
        expected_unit_ids=[11],
        odd_tuning_curve=_make_curve({11: [1.0, np.inf, 3.0]}),
        even_tuning_curve=_make_curve({11: [1.0, 2.0, 3.0]}),
        epoch_firing_rates=pd.Series({11: np.nan}, dtype=float),
        odd_spike_counts=pd.Series({11: 2}, dtype=int),
        even_spike_counts=pd.Series({11: 2}, dtype=int),
        n_odd_trials=2,
        n_even_trials=2,
        odd_duration_s=4.0,
        even_duration_s=4.0,
        n_odd_feature_samples=40,
        n_even_feature_samples=40,
    )

    assert np.isnan(table.loc[0, "stability_correlation"])
    assert table.loc[0, "stability_status"] == "nonfinite_odd_curve"
    assert np.isnan(table.loc[0, "firing_rate_hz"])


def test_build_stability_table_rejects_curve_unit_mismatch() -> None:
    module = _reload_stability_module()

    with pytest.raises(ValueError, match="source sorting units"):
        module.build_stability_table_for_tuning_curves(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            trajectory_type="right_to_center",
            expected_unit_ids=[11, 12],
            odd_tuning_curve=_make_curve({11: [1.0, 2.0, 3.0]}),
            even_tuning_curve=_make_curve({11: [1.0, 2.0, 3.0]}),
            epoch_firing_rates=pd.Series({11: 1.0, 12: 2.0}, dtype=float),
            odd_spike_counts=pd.Series({11: 2, 12: 2}, dtype=int),
            even_spike_counts=pd.Series({11: 2, 12: 2}, dtype=int),
            n_odd_trials=2,
            n_even_trials=2,
            odd_duration_s=4.0,
            even_duration_s=4.0,
            n_odd_feature_samples=40,
            n_even_feature_samples=40,
        )


def test_compute_epoch_stability_retains_units_when_even_laps_are_missing() -> None:
    import pynapple as nap

    module = _reload_stability_module()
    spikes = nap.TsGroup(
        {
            11: nap.Ts(t=np.asarray([0.2, 0.8]), time_units="s"),
            12: nap.Ts(t=np.asarray([]), time_units="s"),
        }
    )
    feature_times = np.linspace(0.0, 1.0, 21)
    task_progression = nap.Tsd(
        t=feature_times,
        d=feature_times,
        time_units="s",
    )
    one_lap = nap.IntervalSet(start=[0.0], end=[1.0], time_units="s")

    table = module.compute_epoch_stability_table(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        spikes=spikes,
        task_progression_by_trajectory={
            trajectory_type: task_progression
            for trajectory_type in module.TRAJECTORY_TYPES
        },
        trajectory_intervals={
            trajectory_type: one_lap
            for trajectory_type in module.TRAJECTORY_TYPES
        },
        movement_interval=one_lap,
        bins=np.linspace(0.0, 1.0, 6),
        epoch_firing_rates=pd.Series({11: 2.0, 12: 0.0}, dtype=float),
    )

    assert len(table) == 2 * len(module.TRAJECTORY_TYPES)
    assert table.groupby("trajectory_type")["unit"].nunique().eq(2).all()
    assert table["stability_correlation"].isna().all()
    assert set(table["stability_status"]) == {"no_even_trials"}
