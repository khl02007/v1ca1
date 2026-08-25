from __future__ import annotations

import numpy as np

from v1ca1.multiday import compute_tuning_curves as tuning


def test_get_tuning_curve_path_is_explicit(tmp_path):
    path = tuning.get_tuning_curve_path(
        tmp_path,
        region="v1",
        date="20240605",
        epoch="02_r1",
        trajectory="center_to_left",
    )

    assert path == (
        tmp_path
        / "v1_20240605_02_r1_center_to_left_tuning_curves.nc"
    )


def test_compute_tuning_curve_uses_pynapple_and_preserves_units():
    import pynapple as nap

    spikes = nap.TsGroup(
        {
            21: nap.Ts(t=np.array([0.1, 0.2, 0.8])),
            22: nap.Ts(t=np.array([0.3, 0.4, 0.9])),
        }
    )
    progression = nap.Tsd(
        t=np.linspace(0.0, 1.0, 101),
        d=np.linspace(0.0, 1.0, 101),
        time_support=nap.IntervalSet(start=0.0, end=1.0),
    )

    result = tuning.compute_tuning_curve(
        spikes,
        progression,
        nap.IntervalSet(start=0.0, end=0.5),
        np.linspace(0.0, 1.0, 6),
    )

    assert result.dims == ("unit", "tp")
    np.testing.assert_array_equal(result.coords["unit"], [21, 22])
    assert result.sizes == {"unit": 2, "tp": 5}


def test_empty_path_returns_nan_curve_for_every_unit():
    import pynapple as nap

    spikes = nap.TsGroup(
        {
            21: nap.Ts(t=np.array([0.1, 0.3])),
            22: nap.Ts(t=np.array([0.2, 0.4])),
        }
    )
    empty_interval = nap.IntervalSet(
        start=np.array([], dtype=float),
        end=np.array([], dtype=float),
    )
    progression = nap.Tsd(
        t=np.array([], dtype=float),
        d=np.array([], dtype=float),
        time_support=empty_interval,
    )

    result = tuning.compute_tuning_curve(
        spikes,
        progression,
        empty_interval,
        np.linspace(0.0, 1.0, 5),
    )

    assert result.shape == (2, 4)
    assert np.isnan(result).all()


def test_cli_defaults_target_l14_multiday_data():
    args = tuning.parse_arguments([])

    assert args.analysis_path == tuning.DEFAULT_ANALYSIS_PATH
    assert args.animal_name == "L14"
    assert args.region == "v1"
    assert args.dates is None
    assert args.epochs is None
    assert args.trajectories is None
    assert args.output_dir is None
    assert args.place_bin_size_cm == 4.0
