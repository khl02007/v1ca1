from __future__ import annotations

"""Tests for task-progression motor feature construction."""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MODULE_NAME = "v1ca1.task_progression.motor"


class _FakeInterpolated:
    """Minimal object exposing `to_numpy()` like pynapple outputs."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = np.asarray(values, dtype=float)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self._values, dtype=float)


class _FakeIntervalSet:
    """Simple interval container for fake spike-count masking."""

    def __init__(self, start, end, time_units: str = "s") -> None:
        del time_units
        self.start = np.atleast_1d(np.asarray(start, dtype=float))
        self.end = np.atleast_1d(np.asarray(end, dtype=float))

    def contains(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        mask = np.zeros(times.shape, dtype=bool)
        for start, end in zip(self.start, self.end, strict=False):
            mask |= (times >= start) & (times <= end)
        return mask


class _FakeTsd:
    """Minimal `nap.Tsd` replacement with interpolation support."""

    def __init__(
        self,
        t,
        d,
        time_units: str = "s",
        time_support: _FakeIntervalSet | None = None,
    ) -> None:
        del time_units
        self.t = np.asarray(t, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.time_support = time_support

    def interpolate(self, target) -> _FakeInterpolated:
        target_t = np.asarray(target.t, dtype=float)
        interpolated = np.interp(target_t, self.t, self.d)
        if self.time_support is not None:
            interpolated = interpolated[self.time_support.contains(target_t)]
        return _FakeInterpolated(interpolated)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.d, dtype=float)


class _FakeTsdFrame:
    """Minimal `nap.TsdFrame` replacement used by the fit helper."""

    def __init__(self, t, d, columns=None, time_units: str = "s") -> None:
        del columns, time_units
        self.t = np.asarray(t, dtype=float)
        self.d = np.asarray(d, dtype=float)


class _FakeSpikeCounts:
    """Simplified binned spike-count container."""

    def __init__(self, times: np.ndarray, counts: np.ndarray, unit_ids: np.ndarray) -> None:
        self.t = np.asarray(times, dtype=float)
        self.d = np.asarray(counts, dtype=float)
        self.columns = np.asarray(unit_ids)

    def in_interval(self, interval: _FakeIntervalSet) -> np.ndarray:
        return interval.contains(self.t)


class _FakeSpikes:
    """Tiny `TsGroup`-like object supporting count and masking."""

    def __init__(self, times: np.ndarray, counts: np.ndarray, unit_ids: np.ndarray) -> None:
        self._times = np.asarray(times, dtype=float)
        self._counts = np.asarray(counts, dtype=float)
        self._unit_ids = np.asarray(unit_ids)

    def keys(self) -> list[int]:
        return self._unit_ids.tolist()

    def __getitem__(self, unit_mask: np.ndarray) -> "_FakeSpikes":
        unit_mask = np.asarray(unit_mask, dtype=bool)
        return _FakeSpikes(
            self._times,
            self._counts[:, unit_mask],
            self._unit_ids[unit_mask],
        )

    def count(self, bin_size_s: float, ep: _FakeIntervalSet) -> _FakeSpikeCounts:
        del bin_size_s, ep
        return _FakeSpikeCounts(self._times, self._counts, self._unit_ids)


class _FakeBSplineEval:
    """Simple finite polynomial basis used in place of nemos splines."""

    def __init__(self, n_basis_funcs: int, order: int, bounds) -> None:
        del order, bounds
        self.n_basis_funcs = int(n_basis_funcs)

    def compute_features(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        return np.column_stack(
            [np.asarray(values**basis_index, dtype=float) for basis_index in range(self.n_basis_funcs)]
        )


class _FakePopulationGLM:
    """Deterministic stand-in for `nemos.glm.PopulationGLM`."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, design: np.ndarray, response: np.ndarray) -> "_FakePopulationGLM":
        n_features = int(design.shape[1])
        n_units = int(response.shape[1])
        self.coef_ = np.zeros((n_features, n_units), dtype=float)
        self.intercept_ = np.zeros(n_units, dtype=float)
        return self

    def score(
        self,
        design: np.ndarray,
        response: np.ndarray,
        score_type: str,
        aggregate_sample_scores,
    ) -> np.ndarray:
        del design, score_type, aggregate_sample_scores
        return np.zeros(response.shape[1], dtype=float)


class _FakeCvSelection:
    """Minimal xarray-like selection result exposing `.values`."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float)


class _FakeCvPooled:
    """Minimal xarray-like DataArray for pooled CV metric lookup."""

    def __init__(self, metric_values: dict[str, np.ndarray]) -> None:
        self._metric_values = {
            key: np.asarray(values, dtype=float) for key, values in metric_values.items()
        }

    def sel(self, *, cv_metric: str) -> _FakeCvSelection:
        return _FakeCvSelection(self._metric_values[cv_metric])


class _FakeFitDataset:
    """Minimal dataset wrapper used by histogram plotting tests."""

    def __init__(self, metric_values: dict[str, np.ndarray]) -> None:
        self._cv_pooled = _FakeCvPooled(metric_values)
        self.attrs = {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "run1",
        }

    def __getitem__(self, key: str) -> _FakeCvPooled:
        if key != "cv_pooled":
            raise KeyError(key)
        return self._cv_pooled


class _FakeDeltaValues:
    """Minimal xarray-like delta selection result exposing `.values`."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values, dtype=float)


class _FakeDeltaArray:
    """Minimal xarray-like DataArray for primary delta lookup."""

    def __init__(self, metric_values: dict[str, np.ndarray]) -> None:
        self._metric_values = {
            key: np.asarray(values, dtype=float) for key, values in metric_values.items()
        }

    def sel(self, *, delta_metric: str) -> _FakeDeltaValues:
        return _FakeDeltaValues(self._metric_values[delta_metric])


class _FakeNestedDataset:
    """Minimal dataset wrapper used by primary histogram plotting tests."""

    def __init__(self, metric_values: dict[str, np.ndarray]) -> None:
        self._pooled_delta = _FakeDeltaArray(metric_values)
        self.attrs = {
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "run1",
        }

    def __getitem__(self, key: str) -> _FakeDeltaArray:
        if key != "pooled_delta_bits_per_spike":
            raise KeyError(key)
        return self._pooled_delta


class _FakeAxis:
    """Simple matplotlib axis stand-in that records histogram calls."""

    def __init__(self) -> None:
        self.transAxes = object()
        self.hist_calls: list[dict[str, object]] = []
        self.ylabel: str | None = None

    def axvline(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_title(self, title: str) -> None:
        del title

    def set_xlabel(self, label: str) -> None:
        del label

    def set_ylabel(self, label: str) -> None:
        self.ylabel = label

    def text(self, *args, **kwargs) -> None:
        del args, kwargs

    def legend(self, *args, **kwargs) -> None:
        del args, kwargs

    def hist(self, values, **kwargs) -> None:
        self.hist_calls.append(
            {
                "values": np.asarray(values, dtype=float),
                "kwargs": kwargs,
            }
        )


class _FakeFigure:
    """Simple matplotlib figure stand-in for save/close verification."""

    def __init__(self) -> None:
        self.savefig_calls: list[dict[str, object]] = []

    def suptitle(self, title: str, fontsize: float) -> None:
        del title, fontsize

    def savefig(self, out_path: Path, dpi: int, bbox_inches: str) -> None:
        self.savefig_calls.append(
            {"out_path": out_path, "dpi": dpi, "bbox_inches": bbox_inches}
        )


class _FakePyplot(types.ModuleType):
    """Minimal `matplotlib.pyplot` replacement for histogram plotting tests."""

    def __init__(self) -> None:
        super().__init__("matplotlib.pyplot")
        self.subplots_calls: list[dict[str, object]] = []
        self.closed_figures: list[_FakeFigure] = []
        self.axes_grid = np.asarray(
            [[_FakeAxis(), _FakeAxis()], [_FakeAxis(), _FakeAxis()]],
            dtype=object,
        )
        self.figure = _FakeFigure()

    def subplots(self, nrows: int, ncols: int, **kwargs):
        self.subplots_calls.append(
            {"nrows": nrows, "ncols": ncols, "kwargs": kwargs}
        )
        return self.figure, self.axes_grid

    def close(self, fig: _FakeFigure) -> None:
        self.closed_figures.append(fig)


class _FakePrimaryPyplot(types.ModuleType):
    """Minimal pyplot replacement for primary histogram plotting tests."""

    def __init__(self) -> None:
        super().__init__("matplotlib.pyplot")
        self.subplots_calls: list[dict[str, object]] = []
        self.closed_figures: list[_FakeFigure] = []
        self.axes = np.asarray([_FakeAxis(), _FakeAxis(), _FakeAxis(), _FakeAxis()])
        self.figure = _FakeFigure()

    def subplots(self, nrows: int, ncols: int, **kwargs):
        self.subplots_calls.append(
            {"nrows": nrows, "ncols": ncols, "kwargs": kwargs}
        )
        return self.figure, self.axes

    def close(self, fig: _FakeFigure) -> None:
        self.closed_figures.append(fig)


def _install_fake_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_jax = types.ModuleType("jax")
    fake_jax_numpy = types.ModuleType("jax.numpy")
    fake_jax_numpy.sum = np.sum
    fake_jax.numpy = fake_jax_numpy
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jax_numpy)


def _install_fake_nemos(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nemos = types.ModuleType("nemos")
    fake_nemos_basis = types.ModuleType("nemos.basis")
    fake_nemos_glm = types.ModuleType("nemos.glm")
    fake_nemos_basis.BSplineEval = _FakeBSplineEval
    fake_nemos_glm.PopulationGLM = _FakePopulationGLM
    fake_nemos.basis = fake_nemos_basis
    fake_nemos.glm = fake_nemos_glm
    monkeypatch.setitem(sys.modules, "nemos", fake_nemos)
    monkeypatch.setitem(sys.modules, "nemos.basis", fake_nemos_basis)
    monkeypatch.setitem(sys.modules, "nemos.glm", fake_nemos_glm)


def _install_fake_position_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_position_tools = types.ModuleType("position_tools")

    def _get_speed(position, time, sampling_frequency: float, sigma: float) -> np.ndarray:
        del sampling_frequency, sigma
        position = np.asarray(position, dtype=float)
        time = np.asarray(time, dtype=float)
        velocity = np.gradient(position, time, axis=0)
        return np.linalg.norm(velocity, axis=1)

    fake_position_tools.get_speed = _get_speed
    monkeypatch.setitem(sys.modules, "position_tools", fake_position_tools)


def _install_fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_pynapple = types.ModuleType("pynapple")
    fake_pynapple.IntervalSet = _FakeIntervalSet
    fake_pynapple.Tsd = _FakeTsd
    fake_pynapple.TsdFrame = _FakeTsdFrame
    monkeypatch.setitem(sys.modules, "pynapple", fake_pynapple)


def _reload_motor_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sys, "argv", ["motor.py"])
    _install_fake_jax(monkeypatch)
    _install_fake_nemos(monkeypatch)
    _install_fake_position_tools(monkeypatch)
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def _make_fake_fit_inputs(module):
    times = np.arange(8, dtype=float)
    counts = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    interval_bounds = {
        module.TRAJECTORY_TYPES[0]: (0.0, 1.0),
        module.TRAJECTORY_TYPES[1]: (2.0, 3.0),
        module.TRAJECTORY_TYPES[2]: (4.0, 5.0),
        module.TRAJECTORY_TYPES[3]: (6.0, 7.0),
    }
    trajectory_intervals = {
        trajectory_type: _FakeIntervalSet(start=start, end=end)
        for trajectory_type, (start, end) in interval_bounds.items()
    }
    task_progression_by_trajectory = {
        trajectory_type: _FakeTsd(
            t=times,
            d=np.linspace(0.0, 1.0, times.size),
            time_support=trajectory_intervals[trajectory_type],
        )
        for trajectory_type in module.TRAJECTORY_TYPES
    }
    return {
        "spikes": _FakeSpikes(times, counts, np.asarray([11, 12], dtype=int)),
        "position_tsd": _FakeTsdFrame(
            t=times,
            d=np.column_stack((times, np.zeros_like(times))),
        ),
        "body_position_tsd": _FakeTsdFrame(
            t=times,
            d=np.column_stack((times - 0.5, np.zeros_like(times))),
        ),
        "trajectory_intervals": trajectory_intervals,
        "task_progression_by_trajectory": task_progression_by_trajectory,
        "movement_interval": _FakeIntervalSet(start=times[0], end=times[-1]),
    }


def _fake_covariates() -> dict[str, np.ndarray]:
    phase = np.linspace(0.0, np.pi / 2.0, 8, dtype=float)
    hd_vel = np.linspace(-0.6, 0.8, 8, dtype=float)
    return {
        "speed": np.linspace(5.0, 12.0, 8, dtype=float),
        "accel": np.linspace(-2.0, 2.0, 8, dtype=float),
        "hd_vel": hd_vel,
        "hd_acc": np.linspace(-0.4, 0.3, 8, dtype=float),
        "abs_hd_vel": np.abs(hd_vel),
        "sin_hd": np.sin(phase),
        "cos_hd": np.cos(phase),
    }


def test_compute_motor_covariates_includes_hd_acc_and_matches_velocity_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)
    _install_fake_pynapple(monkeypatch)

    timestamps = np.linspace(0.0, 0.8, 9, dtype=float)
    angles = np.pi * np.square(np.linspace(0.0, 1.0, timestamps.size, dtype=float))
    head_position = np.column_stack((np.cos(angles), np.sin(angles)))
    body_position = np.zeros_like(head_position)
    spike_counts = _FakeSpikeCounts(
        times=timestamps,
        counts=np.ones((timestamps.size, 2), dtype=float),
        unit_ids=np.asarray([1, 2], dtype=int),
    )

    covariates = module.compute_motor_covariates(
        position_xy=head_position,
        body_xy=body_position,
        position_timestamps=timestamps,
        spike_counts=spike_counts,
    )

    dt = float(np.median(np.diff(timestamps)))
    assert "hd_acc" in covariates
    assert covariates["hd_acc"].shape == timestamps.shape
    assert np.isfinite(covariates["hd_acc"]).all()
    assert np.allclose(covariates["hd_acc"], np.gradient(covariates["hd_vel"], dt))


def test_select_run_epochs_deduplicates_requested_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    selected_epochs = module.select_run_epochs(
        ["run1", "run2", "run3"],
        ["run2", "run1", "run2"],
    )

    assert selected_epochs == ["run2", "run1"]


def test_get_task_progression_figure_dir_uses_task_progression_and_script_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    figure_dir = module.get_task_progression_figure_dir(Path("/tmp/example_session"), "motor")

    assert figure_dir == Path("/tmp/example_session/figs/task_progression/motor")


def test_get_task_progression_output_dir_uses_task_progression_and_script_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    output_dir = module.get_task_progression_output_dir(Path("/tmp/example_session"), "motor")

    assert output_dir == Path("/tmp/example_session/task_progression/motor")


def test_select_epochs_with_usable_position_data_uses_clean_dlc_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)
    analysis_path = Path("/tmp/example_session")
    clean_dlc_path = (
        analysis_path
        / module.DEFAULT_CLEAN_DLC_POSITION_DIRNAME
        / module.DEFAULT_CLEAN_DLC_POSITION_NAME
    )

    monkeypatch.setattr(module, "load_epoch_tags", lambda path: (["run1", "run2"], "pynapple"))
    monkeypatch.setattr(module, "get_run_epochs", lambda epoch_tags: list(epoch_tags))
    monkeypatch.setattr(
        module,
        "load_clean_dlc_position_data",
        lambda path, input_dirname, input_name, validate_timestamps: (
            ["run1"],
            {"run1": np.asarray([[1.0, 2.0]], dtype=float)},
            {"run1": np.asarray([[3.0, 4.0]], dtype=float)},
        ),
    )

    usable_epochs, skipped_epochs = module.select_epochs_with_usable_position_data(
        analysis_path,
        ["run1", "run2", "run2"],
    )

    assert usable_epochs == ["run1"]
    assert skipped_epochs == [
        {"epoch": "run2", "reason": f"head position missing from {clean_dlc_path}; body position missing from {clean_dlc_path}"}
    ]


def test_build_position_basis_config_uses_spatial_bin_size_and_full_w_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    class _FakeGraph:
        def __init__(self) -> None:
            self.nodes = {
                0: {"pos": np.asarray((0.0, 0.0), dtype=float)},
                1: {"pos": np.asarray((3.0, 4.0), dtype=float)},
                2: {"pos": np.asarray((6.0, 4.0), dtype=float)},
            }

    monkeypatch.setattr(module, "get_wtrack_total_length", lambda animal_name: 11.0)
    monkeypatch.setattr(
        module,
        "get_wtrack_full_graph",
        lambda animal_name, branch_gap_cm: (
            _FakeGraph(),
            [(0, 1), (1, 2)],
            [float(branch_gap_cm)],
        ),
    )

    config = module.build_position_basis_config(
        animal_name="L14",
        spatial_bin_size_cm=4.0,
        spline_order=3,
        generalized_place_branch_gap_cm=9.0,
    )

    assert config["trajectory_length_cm"] == 11.0
    assert config["generalized_place_length_cm"] == 17.0
    assert config["trajectory_n_splines"] == 3
    assert config["generalized_place_n_splines"] == 5
    assert config["trajectory_bounds"] == (0.0, 1.0)
    assert config["generalized_place_bounds"] == (0.0, 17.0)


def test_build_model_designs_adds_no_offset_generalized_place_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    n_time = 8
    rows = np.ones(n_time, dtype=bool)
    data = {
        "motor_covariates": _fake_covariates(),
        "trajectory_labels": np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=int),
        "task_progression": np.linspace(0.0, 1.0, n_time, dtype=float),
        "generalized_place": np.linspace(0.0, 40.0, n_time, dtype=float),
        "generalized_task_progression": np.linspace(1.0, 0.0, n_time, dtype=float),
    }
    motor_transform = module.fit_motor_feature_transform(
        data,
        rows,
        motor_feature_mode="zscore",
        motor_zscore_eps=1e-12,
        motor_spline_k=3,
        motor_spline_order=4,
    )
    design_info = module.build_model_designs(
        data,
        rows,
        motor_transform,
        position_basis={
            "spatial_bin_size_cm": 4.0,
            "spline_order": 3,
            "trajectory_length_cm": 12.0,
            "generalized_place_length_cm": 40.0,
            "trajectory_n_splines": 3,
            "generalized_place_n_splines": 5,
            "trajectory_bounds": (0.0, 1.0),
            "generalized_place_bounds": (0.0, 40.0),
        },
    )

    assert tuple(design_info["designs"]) == module.MODEL_NAMES
    assert design_info["designs"]["motor"].shape == (n_time, 7)
    assert design_info["designs"]["motor_tp"].shape == (n_time, 14)
    assert design_info["designs"]["tp_only"].shape == (n_time, 7)
    assert design_info["designs"]["motor_place"].shape == (n_time, 22)
    assert design_info["designs"]["place_only"].shape == (n_time, 15)
    assert design_info["designs"]["motor_generalized_place"].shape == (n_time, 12)
    assert design_info["designs"]["generalized_place_only"].shape == (n_time, 5)
    assert design_info["designs"]["motor_generalized_task_progression"].shape == (
        n_time,
        10,
    )
    assert design_info["designs"]["generalized_task_progression_only"].shape == (
        n_time,
        3,
    )
    assert list(design_info["blocks"]["generalized_place_only"]) == [
        "generalized_place"
    ]
    assert list(design_info["blocks"]["generalized_task_progression_only"]) == [
        "generalized_tp"
    ]


def test_full_refit_dataset_builds_with_spline_motor_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    n_time = 8
    data = {
        "response": np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        ),
        "unit_ids": np.asarray([11, 12], dtype=int),
        "base_keep_mask": np.ones(n_time, dtype=bool),
        "motor_covariates": _fake_covariates(),
        "trajectory_labels": np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=int),
        "task_progression": np.linspace(0.0, 1.0, n_time, dtype=float),
        "generalized_place": np.linspace(0.0, 40.0, n_time, dtype=float),
        "generalized_task_progression": np.linspace(1.0, 0.0, n_time, dtype=float),
        "tp_bounds": (0.0, 1.0),
        "bin_size_s": 0.05,
    }
    position_basis = {
        "spatial_bin_size_cm": 4.0,
        "spline_order": 3,
        "trajectory_length_cm": 12.0,
        "generalized_place_length_cm": 40.0,
        "trajectory_n_splines": 3,
        "generalized_place_n_splines": 5,
        "trajectory_bounds": (0.0, 1.0),
        "generalized_place_bounds": (0.0, 40.0),
    }
    full_fit = module.fit_full_refit_models(
        data,
        unit_mask=np.asarray([True, True]),
        ridge_by_model={model_name: 1e-3 for model_name in module.MODEL_NAMES},
        position_basis_by_model={
            model_name: position_basis for model_name in module.MODEL_NAMES
        },
        motor_feature_mode="spline",
        motor_zscore_eps=1e-12,
        motor_spline_k=3,
        motor_spline_order=4,
    )
    ridge_cv_result = {
        "info_bits_per_spike": np.zeros(
            (len(module.MODEL_NAMES), 1, 1, 2),
            dtype=float,
        ),
        "score_median": np.zeros((len(module.MODEL_NAMES), 1, 1), dtype=float),
        "score_mean": np.zeros((len(module.MODEL_NAMES), 1, 1), dtype=float),
        "score_n_finite": np.ones((len(module.MODEL_NAMES), 1, 1), dtype=int),
        "spatial_bin_sizes_cm": np.asarray([4.0], dtype=float),
        "ridge_values": np.asarray([1e-3], dtype=float),
    }

    dataset = module.build_full_refit_dataset(
        full_fit,
        ridge_cv_result,
        data,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="run1",
        movement_firing_rates=np.asarray([2.0, 3.0], dtype=float),
        min_firing_rate_hz=0.5,
        sources={},
        fit_parameters={},
    )

    assert dataset["motor_spline_bounds"].shape == (
        len(module.MOTOR_CONTINUOUS_FEATURE_NAMES),
        2,
    )
    assert dataset["generalized_task_progression_rate_curve_hz"].shape == (200, 2)


def test_build_histogram_bin_edges_always_includes_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)

    bin_edges = module.build_histogram_bin_edges(np.asarray([0.2, 0.4, 0.9], dtype=float))

    assert np.any(np.isclose(bin_edges, 0.0))
    assert np.all(np.diff(bin_edges) > 0.0)


def test_plot_log_likelihood_difference_histograms_uses_two_rows_and_no_outlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)
    fake_pyplot = _FakePyplot()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    fit_dataset = _FakeFitDataset(
        {
            "dll_motor_tp_vs_motor_bits_per_spike": np.asarray([0.1, 0.4, 0.8], dtype=float),
            "dll_motor_place_vs_motor_bits_per_spike": np.asarray(
                [0.2, 0.5, 0.7], dtype=float
            ),
            "dll_motor_tp_vs_tp_only_bits_per_spike": np.asarray(
                [-0.3, 0.2, 0.6], dtype=float
            ),
            "dll_motor_place_vs_place_only_bits_per_spike": np.asarray(
                [-0.4, -0.1, 0.3], dtype=float
            ),
        }
    )

    module.plot_log_likelihood_difference_histograms(
        fit_dataset,
        out_path=Path("/tmp/task_progression_motor_ll_hist.png"),
    )

    assert fake_pyplot.subplots_calls == [
        {
            "nrows": 2,
            "ncols": 2,
            "kwargs": {
                "figsize": (12, 8.4),
                "constrained_layout": True,
                "sharey": True,
            },
        }
    ]
    assert fake_pyplot.axes_grid[0, 0].ylabel == "Minus Motor\nFraction of units"
    assert fake_pyplot.axes_grid[1, 0].ylabel == "Minus TP / Place-only\nFraction of units"

    for axis in fake_pyplot.axes_grid.ravel():
        assert len(axis.hist_calls) == 1
        hist_kwargs = axis.hist_calls[0]["kwargs"]
        assert hist_kwargs["edgecolor"] == "none"
        assert hist_kwargs["linewidth"] == 0.0
        assert np.any(np.isclose(np.asarray(hist_kwargs["bins"], dtype=float), 0.0))


def test_plot_primary_delta_histograms_uses_four_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _reload_motor_module(monkeypatch)
    fake_pyplot = _FakePrimaryPyplot()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_pyplot
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)

    nested_dataset = _FakeNestedDataset(
        {
            "dll_motor_tp_vs_motor_bits_per_spike": np.asarray([0.1, 0.4]),
            "dll_motor_place_vs_motor_bits_per_spike": np.asarray([0.2, 0.5]),
            "dll_motor_generalized_place_vs_motor_bits_per_spike": np.asarray(
                [0.3, 0.6]
            ),
            "dll_motor_generalized_task_progression_vs_motor_bits_per_spike": (
                np.asarray([0.4, 0.7])
            ),
        }
    )

    module.plot_primary_delta_histograms(
        nested_dataset,
        out_path=Path("/tmp/task_progression_motor_primary_delta_hist.png"),
    )

    assert fake_pyplot.subplots_calls == [
        {
            "nrows": 1,
            "ncols": 4,
            "kwargs": {
                "figsize": (20.0, 4.2),
                "constrained_layout": True,
                "sharey": True,
            },
        }
    ]
    for axis in fake_pyplot.axes:
        assert len(axis.hist_calls) == 1
        hist_kwargs = axis.hist_calls[0]["kwargs"]
        assert hist_kwargs["edgecolor"] == "none"
        assert hist_kwargs["linewidth"] == 0.0
