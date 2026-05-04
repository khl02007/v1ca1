from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
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


def test_build_stability_table_applies_firing_rate_threshold() -> None:
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
        odd_tuning_curve=odd_curve,
        even_tuning_curve=even_curve,
        epoch_firing_rates=epoch_firing_rates,
        n_odd_trials=3,
        n_even_trials=2,
        odd_duration_s=12.0,
        even_duration_s=8.0,
        firing_rate_threshold_hz=0.0,
    )

    assert table["unit"].tolist() == [11, 13]
    assert np.allclose(table.loc[0, "stability_correlation"], 1.0)
    assert np.isnan(table.loc[1, "stability_correlation"])
    assert np.allclose(table["firing_rate_hz"], [0.1, 2.0])
    assert table["n_odd_trials"].tolist() == [3, 3]
    assert table["n_even_trials"].tolist() == [2, 2]
