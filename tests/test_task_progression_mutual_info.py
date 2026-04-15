from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


MODULE_NAME = "v1ca1.task_progression.mutual_info"


class _FakeTs:
    """Minimal spike train container used by mutual-info shuffle tests."""

    def __init__(self, label: str) -> None:
        self.label = str(label)


class _FakeTsGroup:
    """Minimal `nap.TsGroup` replacement storing spike trains by unit id."""

    def __init__(self, spikes_by_unit: dict[int, object], time_units: str = "s") -> None:
        del time_units
        self.spikes_by_unit = dict(spikes_by_unit)

    def keys(self):
        return list(self.spikes_by_unit.keys())

    def __getitem__(self, unit: int) -> object:
        return self.spikes_by_unit[unit]


class _FakeInterval:
    """Minimal interval object exposing total length and start/end arrays."""

    def __init__(self, total_length: float, *, start: float = 0.0, end: float = 100.0) -> None:
        self._total_length = float(total_length)
        self.start = np.asarray([start], dtype=float)
        self.end = np.asarray([end], dtype=float)

    def tot_length(self) -> float:
        return self._total_length


def _reload_mutual_info_module():
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_circular_shift_spikes_on_movement_axis_uses_movement_duration_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_mutual_info_module()

    fake_pynapple = types.ModuleType("pynapple")
    fake_pynapple.TsGroup = _FakeTsGroup
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)

    calls: list[dict[str, object]] = []

    def _fake_shift(unit_spikes, movement_epoch, *, rng, min_shift_fraction):
        del rng
        calls.append(
            {
                "unit_spikes": unit_spikes,
                "movement_epoch": movement_epoch,
                "min_shift_fraction": float(min_shift_fraction),
            }
        )
        return _FakeTs(f"shifted_{unit_spikes.label}")

    monkeypatch.setattr(
        module,
        "circular_shift_unit_spikes_on_movement_axis",
        _fake_shift,
    )

    spikes = _FakeTsGroup({11: _FakeTs("u11"), 12: _FakeTs("u12")})
    movement_epoch = _FakeInterval(total_length=100.0)

    shifted = module.circular_shift_spikes_on_movement_axis(
        spikes,
        movement_epoch,
        rng=np.random.default_rng(0),
        min_shift_s=20.0,
    )

    assert isinstance(shifted, _FakeTsGroup)
    assert shifted.keys() == [11, 12]
    assert shifted[11].label == "shifted_u11"
    assert shifted[12].label == "shifted_u12"
    assert [call["min_shift_fraction"] for call in calls] == [0.2, 0.2]
    assert all(call["movement_epoch"] is movement_epoch for call in calls)


def test_compute_shuffled_si_uses_movement_axis_shuffle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_mutual_info_module()

    fake_pynapple = types.ModuleType("pynapple")

    def _unexpected_shift_timestamps(*args, **kwargs):
        raise AssertionError("compute_shuffled_si should not call nap.shift_timestamps")

    fake_pynapple.shift_timestamps = _unexpected_shift_timestamps
    fake_pynapple.compute_tuning_curves = lambda data, features, bins, epochs: {
        "data": data,
        "features": features,
        "bins": bins,
        "epochs": epochs,
    }
    fake_pynapple.compute_mutual_information = lambda tuning_curve: pd.DataFrame(
        {
            "bits/sec": [1.0, 2.0],
            "bits/spike": [0.5, 1.5],
        },
        index=[11, 12],
    )
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)

    helper_calls: list[dict[str, object]] = []

    def _fake_group_shift(spikes, movement_epoch, *, rng, min_shift_s):
        helper_calls.append(
            {
                "spikes": spikes,
                "movement_epoch": movement_epoch,
                "rng": rng,
                "min_shift_s": float(min_shift_s),
            }
        )
        return spikes

    monkeypatch.setattr(module, "circular_shift_spikes_on_movement_axis", _fake_group_shift)

    spikes = _FakeTsGroup({11: _FakeTs("u11"), 12: _FakeTs("u12")})
    epoch = _FakeInterval(total_length=200.0)
    movement_epoch = _FakeInterval(total_length=100.0)

    shuffled_si = module.compute_shuffled_si(
        spikes=spikes,
        epoch=epoch,
        movement_epoch=movement_epoch,
        feature="feature",
        bins=np.asarray([0.0, 1.0], dtype=float),
        n_shuffles=2,
        min_shift_s=20.0,
    )

    assert list(shuffled_si.index) == [11, 12]
    assert helper_calls[0]["spikes"] is spikes
    assert helper_calls[0]["movement_epoch"] is movement_epoch
    assert helper_calls[0]["min_shift_s"] == 20.0
    assert helper_calls[0]["rng"] is helper_calls[1]["rng"]
