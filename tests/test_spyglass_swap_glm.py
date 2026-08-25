"""Tests for database-free held-out swap-GLM artifact handling."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import swap_glm as module


RESULT_ID = uuid.UUID("519388fb-1ea5-5b5b-925a-405c26020550")
UPSTREAM_ID = uuid.UUID("a04e12d4-f8f5-58a2-b421-f66de4aca4b3")


class FakeIntervals:
    """Minimal interval object for terminal-state tests."""

    def __init__(self, starts=(0.0,), ends=(1.0,)):
        self.start = np.asarray(starts, dtype=float)
        self.end = np.asarray(ends, dtype=float)


def _valid_position() -> SimpleNamespace:
    """Return a minimal finite position object for registration tests."""
    return SimpleNamespace(
        t=np.asarray([0.0, 1.0]),
        d=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )


def _graph_inputs() -> dict[str, dict[str, object]]:
    nodes = np.column_stack((np.arange(6, dtype=float) * 10.0, np.zeros(6)))
    edge_order = [[index, index + 1] for index in range(5)]
    return {
        trajectory: {
            "configuration_name": trajectory,
            "coordinate_unit": "cm",
            "track_graph_kwargs": {"node_positions": nodes.tolist()},
            "linearization_kwargs": {
                "edge_order": edge_order,
                "edge_spacing": [0.0, 0.0, 0.0, 0.0],
            },
        }
        for trajectory in TRAJECTORY_TYPES
    }


def _source_dataset(model_name: str, group_unit_ids=("101", "102")):
    xr = pytest.importorskip("xarray")
    return xr.Dataset(
        coords={"unit": np.asarray(group_unit_ids)},
        attrs={"model_name": model_name},
    )


def _upstream(unit_ids=(1, 2), group_unit_ids=("101", "102")) -> dict[str, object]:
    selected_units = pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge-a"] * len(unit_ids),
            "unit_id": [str(value) for value in unit_ids],
            "stable_unit_id": [f"merge-a:{value}" for value in unit_ids],
            "group_unit_id": [str(value) for value in group_unit_ids],
            "selection_index": np.arange(len(unit_ids), dtype=int),
            "dark_movement_firing_rate_hz": np.arange(len(unit_ids)) + 1.0,
            "light_movement_firing_rate_hz": np.arange(len(unit_ids)) + 1.5,
            "n_selected_model_trajectory_fits": [16] * len(unit_ids),
            "valid_glm_fit": [True] * len(unit_ids),
        }
    )
    return {
        "metadata": {
            "dark_light_glm_id": str(UPSTREAM_ID),
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "light_epoch": "02_r1",
            "dark_epoch": "08_r4",
        },
        "parameters": {
            "parameter_sha256": "d" * 64,
            "output_rule_sha256": "e" * 64,
            "speed_smoothing_sigma_s": 0.1,
        },
        "selected_datasets": {
            name: _source_dataset(name, group_unit_ids)
            for name in module.SOURCE_MODEL_NAMES
        },
        "selected_units": selected_units,
        "segment_edges": np.asarray([0.0, 0.3, 0.7, 1.0]),
        "analysis_status": "valid",
        "upstream_provenance": {
            "dark_light_glm_id": str(UPSTREAM_ID),
            "dark_light_manifest_sha256": "a" * 64,
            "dark_light_selected_sha256_by_model": {
                name: "b" * 64 for name in module.SOURCE_MODEL_NAMES
            },
            "dark_light_parameter_sha256": "d" * 64,
            "dark_light_output_rule_sha256": "e" * 64,
            "upstream_analysis_status": "valid",
            "dark_light_artifact_path": "/tmp/upstream",
        },
    }


def _patch_identities(monkeypatch: pytest.MonkeyPatch, *, conflict=False) -> None:
    from v1ca1.spyglass import path_specific_place

    rows = []
    for unit_id, group_unit_id in zip((1, 2), (101, 102), strict=True):
        rows.append(
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": str(unit_id),
                "stable_unit_id": (
                    "wrong" if conflict and unit_id == 2 else f"merge-a:{unit_id}"
                ),
                "group_unit_id": str(group_unit_id),
                "_group_key": group_unit_id,
            }
        )
    monkeypatch.setattr(
        path_specific_place,
        "_identity_rows",
        lambda _spikes, _stable: rows,
    )


def _result_dataset(
    *,
    second_unit_valid=True,
    unit_ids=(101, 102),
    selected_source_paths: dict[str, str] | None = None,
):
    xr = pytest.importorskip("xarray")
    model_count = len(module.MODEL_NAMES)
    trajectory_count = len(TRAJECTORY_TYPES)
    unit_count = len(unit_ids)
    tp_grid = np.linspace(0.0, 1.0, 3)
    observed_edges = module._observed_bin_edges(50.0, 4.0)
    observed_bins = 0.5 * (observed_edges[:-1] + observed_edges[1:])
    occupancy = np.zeros((trajectory_count, len(observed_bins)), dtype=float)
    occupancy[:, :2] = 0.02
    spike_count = np.zeros(
        (trajectory_count, len(observed_bins), unit_count),
        dtype=float,
    )
    spike_count[:, 0] = np.asarray([1.0, 2.0])[:unit_count]
    spike_count[:, 1] = np.asarray([2.0, 1.0])[:unit_count]
    observed_rate = np.full_like(spike_count, np.nan)
    occupied = occupancy > 0.0
    observed_rate[occupied] = spike_count[occupied] / occupancy[occupied, None]
    full_spike_sum = np.sum(spike_count, axis=1)

    data_vars = {
        "selected_model_path": (
            ("model",),
            np.asarray([f"/tmp/{name}.nc" for name in module.MODEL_NAMES]),
        ),
        "selected_source_model": (
            ("model",),
            np.asarray(
                [
                    module.DERIVED_MODEL_SOURCES.get(name, name)
                    for name in module.MODEL_NAMES
                ],
                dtype=str,
            ),
        ),
        "selected_ridge": (("model",), np.full(model_count, 1e-3)),
        "selected_score": (("model",), np.linspace(0.1, 0.5, model_count)),
        "swap_source_trajectory": (
            ("trajectory",),
            np.asarray(
                [
                    module.OUTPUT_RULE["swap_configuration"][name][
                        "source_trajectory"
                    ]
                    for name in TRAJECTORY_TYPES
                ],
                dtype=str,
            ),
        ),
        "swap_segment_index_1based": (
            ("trajectory",),
            np.asarray(
                [
                    module.OUTPUT_RULE["swap_configuration"][name][
                        "segment_index"
                    ]
                    + 1
                    for name in TRAJECTORY_TYPES
                ],
                dtype=int,
            ),
        ),
        "swap_segment_start": (
            ("trajectory",),
            np.asarray([0.7, 0.0, 0.7, 0.0]),
        ),
        "swap_segment_end": (
            ("trajectory",),
            np.asarray([1.0, 0.3, 1.0, 0.3]),
        ),
        "dark_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            np.full((model_count, trajectory_count, len(tp_grid), unit_count), 2.0),
        ),
        "train_light_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            np.full((model_count, trajectory_count, len(tp_grid), unit_count), 3.0),
        ),
        "test_light_unswapped_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            np.full((model_count, trajectory_count, len(tp_grid), unit_count), 4.0),
        ),
        "test_light_swapped_hz_grid": (
            ("model", "trajectory", "tp_grid", "unit"),
            np.full((model_count, trajectory_count, len(tp_grid), unit_count), 5.0),
        ),
        "test_light_swapped_segment_n_bins": (
            ("trajectory",),
            np.ones(trajectory_count, dtype=int),
        ),
        "test_light_full_n_bins": (
            ("trajectory",),
            np.full(trajectory_count, 2, dtype=int),
        ),
        "test_light_occupancy_s": (
            ("trajectory", "tp_observed_bin"),
            occupancy,
        ),
        "test_light_spike_count": (
            ("trajectory", "tp_observed_bin", "unit"),
            spike_count,
        ),
        "test_light_observed_rate_hz": (
            ("trajectory", "tp_observed_bin", "unit"),
            observed_rate,
        ),
    }
    for prefix in module.METRIC_PREFIXES:
        spikes = np.broadcast_to(
            full_spike_sum[None, :, :],
            (model_count, trajectory_count, unit_count),
        ).copy()
        bits = np.broadcast_to(
            (-2.0 + 0.25 * np.arange(model_count))[:, None, None],
            spikes.shape,
        ).copy()
        if (
            prefix == "test_light_swapped_segment_swapped"
            and not second_unit_valid
        ):
            bits[2, 1, 1] = np.nan
        raw_ll = bits * spikes * np.log(2.0)
        data_vars[f"{prefix}_raw_ll_sum"] = (
            ("model", "trajectory", "unit"),
            raw_ll,
        )
        data_vars[f"{prefix}_spike_sum"] = (
            ("model", "trajectory", "unit"),
            spikes,
        )
        data_vars[f"{prefix}_raw_ll_bits_per_spike"] = (
            ("model", "trajectory", "unit"),
            bits,
        )
    swapped_bits = data_vars[
        "test_light_swapped_segment_swapped_raw_ll_bits_per_spike"
    ][1]
    delta = swapped_bits - swapped_bits[[0]]
    delta[0] = np.nan
    if not second_unit_valid:
        assert np.isnan(delta[2, 1, 1])
    data_vars[module.PRIMARY_METRIC] = (
        ("model", "trajectory", "unit"),
        delta,
    )
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "model": np.asarray(module.MODEL_NAMES, dtype=str),
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "unit": np.asarray(unit_ids),
            "tp_grid": tp_grid,
            "tp_observed_bin": observed_bins,
            "segment_edge": np.asarray([0.0, 0.3, 0.7, 1.0]),
            "tp_observed_edge": observed_edges,
        },
        attrs={
            "schema_version": module.RESULT_SCHEMA_VERSION,
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "dark_train_epoch": "08_r4",
            "light_train_epoch": "02_r1",
            "light_test_epoch": "04_r2",
            "fit_source": "dark_light_glm_selected",
            "test_scoring_scope": "swapped_segment_primary_and_full_diagnostic",
            "primary_metric": module.PRIMARY_METRIC,
            "heldout_epoch_scope": "all_movement_laps",
            "raw_ll_bits_per_spike_definition": (
                "raw_poisson_ll_sum / spike_sum / log(2)"
            ),
            "swap_light_offset": False,
            "bin_size_s": 0.02,
            "spatial_bin_size_cm": 4.0,
            "n_splines": 25,
            "spline_order": 4,
            "has_speed": False,
            "speed_feature_mode": "none",
            "n_speed_features": 0,
            "speed_spline_order": np.nan,
            "swap_rule_json": json.dumps(
                dict(module.OUTPUT_RULE["swap_configuration"]),
                sort_keys=True,
            ),
            "derived_model_sources_json": json.dumps(
                dict(module.DERIVED_MODEL_SOURCES),
                sort_keys=True,
            ),
            "sources_json": json.dumps({}, sort_keys=True),
            "fit_parameters_json": json.dumps(
                {
                    "models": list(module.MODEL_NAMES),
                    "derived_model_sources": dict(module.DERIVED_MODEL_SOURCES),
                    "scoring_epoch_scope": "all_movement_laps",
                    "position_offset": 10,
                    "speed_threshold_cm_s": 4.0,
                    "swap_light_offset": False,
                    "swapped_component": (
                        "local_model_component_without_scalar_light_offset"
                    ),
                },
                sort_keys=True,
            ),
            "prediction_count_clip_eps": module.PREDICTION_COUNT_CLIP_EPS,
            "visual_empirical_model_definitions_json": json.dumps(
                dict(module.VISUAL_EMPIRICAL_MODEL_DEFINITIONS),
                sort_keys=True,
            ),
        },
    )
    if selected_source_paths is not None:
        dataset.attrs["sources_json"] = json.dumps(
            {"dark_light_glm_selected": selected_source_paths},
            sort_keys=True,
        )
    return dataset


def _schema6_dataset(
    *,
    unit_ids=(11, 12),
    selected_source_paths: dict[str, str] | None = None,
):
    """Return a realistic nine-model legacy schema-6 result."""
    dataset = _result_dataset(
        unit_ids=unit_ids,
        selected_source_paths=selected_source_paths,
    )
    source_indices = np.asarray([0, 1, 0, 0, 0, 2, 3, 4, 5], dtype=int)
    dataset = dataset.isel(model=source_indices).assign_coords(
        model=np.asarray(module.LEGACY_SCHEMA6_MODEL_NAMES, dtype=str)
    )
    dataset["selected_source_model"] = (
        ("model",),
        np.asarray(
            [
                module.LEGACY_SCHEMA6_DERIVED_MODEL_SOURCES.get(name, name)
                for name in module.LEGACY_SCHEMA6_MODEL_NAMES
            ],
            dtype=str,
        ),
    )
    swapped_bits = np.asarray(
        dataset[
            "test_light_swapped_segment_swapped_raw_ll_bits_per_spike"
        ].values,
        dtype=float,
    )
    delta = swapped_bits - swapped_bits[[0]]
    delta[0] = np.nan
    dataset[module.PRIMARY_METRIC] = (
        ("model", "trajectory", "unit"),
        delta,
    )
    for name in module.LEGACY_SCHEMA6_DIAGNOSTIC_VARIABLES:
        dataset[name] = (
            ("model", "trajectory", "unit"),
            np.zeros(
                (
                    len(module.LEGACY_SCHEMA6_MODEL_NAMES),
                    len(TRAJECTORY_TYPES),
                    len(unit_ids),
                )
            ),
        )
    dataset.attrs["schema_version"] = "6"
    dataset.attrs["prediction_count_clip_eps"] = 1e-12
    dataset.attrs["visual_empirical_model_definitions_json"] = json.dumps(
        dict(module.LEGACY_SCHEMA6_VISUAL_EMPIRICAL_MODEL_DEFINITIONS),
        sort_keys=True,
    )
    dataset.attrs["derived_model_sources_json"] = json.dumps(
        dict(module.LEGACY_SCHEMA6_DERIVED_MODEL_SOURCES),
        sort_keys=True,
    )
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    fit_parameters.update(
        {
            "models": list(module.LEGACY_SCHEMA6_MODEL_NAMES),
            "requested_models": list(module.LEGACY_SCHEMA6_MODEL_NAMES),
            "derived_model_sources": dict(
                module.LEGACY_SCHEMA6_DERIVED_MODEL_SOURCES
            ),
        }
    )
    dataset.attrs["fit_parameters_json"] = json.dumps(
        fit_parameters,
        sort_keys=True,
    )
    return dataset


def _reduced_schema4_dataset(
    *,
    unit_ids=(11, 12),
    selected_source_paths: dict[str, str] | None = None,
):
    """Return the real reverse-direction four-source-model schema shape."""
    dataset = _result_dataset(
        unit_ids=unit_ids,
        selected_source_paths=selected_source_paths,
    ).sel(model=list(module.SOURCE_MODEL_NAMES))
    dataset = dataset.drop_vars("selected_source_model")
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs.pop("derived_model_sources_json", None)
    dataset.attrs.pop("spatial_bin_size_cm", None)
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    fit_parameters["models"] = list(module.SOURCE_MODEL_NAMES)
    fit_parameters["requested_models"] = list(module.SOURCE_MODEL_NAMES)
    fit_parameters.pop("derived_model_sources", None)
    dataset.attrs["fit_parameters_json"] = json.dumps(
        fit_parameters,
        sort_keys=True,
    )
    dataset.attrs["schema_version"] = "4"
    dataset.attrs.pop("prediction_count_clip_eps", None)
    dataset.attrs.pop("visual_empirical_model_definitions_json", None)
    return dataset


def _full_schema4_dataset(
    *,
    unit_ids=(11, 12),
    selected_source_paths: dict[str, str] | None = None,
):
    """Return the historical five-model canonical schema-4 result."""
    dataset = _result_dataset(
        unit_ids=unit_ids,
        selected_source_paths=selected_source_paths,
    ).sel(model=list(module.LEGACY_SCHEMA4_MODEL_NAMES))
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["schema_version"] = "4"
    dataset.attrs.pop("prediction_count_clip_eps", None)
    dataset.attrs.pop("visual_empirical_model_definitions_json", None)
    dataset.attrs["derived_model_sources_json"] = json.dumps(
        dict(module.LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES),
        sort_keys=True,
    )
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    fit_parameters["models"] = list(module.LEGACY_SCHEMA4_MODEL_NAMES)
    fit_parameters["requested_models"] = list(
        module.LEGACY_SCHEMA4_MODEL_NAMES
    )
    fit_parameters["derived_model_sources"] = dict(
        module.LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES
    )
    dataset.attrs["fit_parameters_json"] = json.dumps(
        fit_parameters,
        sort_keys=True,
    )
    return dataset


def _write_legacy_selected_sources(
    directory: Path,
    *,
    unit_ids=(11, 12),
) -> tuple[dict[str, str], dict[str, str]]:
    """Write one exact legacy selected-source set for registration tests."""
    paths = {}
    hashes = {}
    for model_name in module.SOURCE_MODEL_NAMES:
        path = directory / f"legacy_{model_name}.nc"
        _source_dataset(model_name, unit_ids).to_netcdf(path)
        paths[model_name] = str(path)
        hashes[model_name] = module._file_sha256(path)
    paths["dark"] = paths["task_segment_bump"]
    paths[module.VISUAL_ADDITIVE_MODEL_NAME] = paths["visual"]
    return paths, hashes


def _prepare_registration_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema: str,
) -> tuple[Path, FakeAnalysis]:
    """Write one source result and patch exact upstream registration inputs."""
    upstream = _upstream()
    fake_analysis = FakeAnalysis()
    selected_paths, selected_hashes = _write_legacy_selected_sources(tmp_path)
    upstream["upstream_provenance"][
        "dark_light_legacy_selected_sha256_by_model"
    ] = selected_hashes
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    monkeypatch.setattr(module, "_analysis_module", lambda: fake_analysis)
    _patch_fake_additive_evaluator(monkeypatch)
    monkeypatch.setattr(
        module,
        "_derive_task_progression",
        lambda **_kwargs: {name: object() for name in TRAJECTORY_TYPES},
    )
    _patch_identities(monkeypatch)
    if schema == "6":
        dataset = _schema6_dataset(
            unit_ids=(11, 12),
            selected_source_paths=selected_paths,
        )
    elif schema == "4-full":
        dataset = _full_schema4_dataset(
            unit_ids=(11, 12),
            selected_source_paths=selected_paths,
        )
    elif schema == "4-reduced":
        dataset = _reduced_schema4_dataset(
            unit_ids=(11, 12),
            selected_source_paths=selected_paths,
        )
    else:
        raise ValueError(f"Unsupported test schema {schema!r}.")
    source = tmp_path / f"legacy-{schema}.nc"
    dataset.to_netcdf(source)
    return source, fake_analysis


def _register_kwargs(source: Path, destination: Path) -> dict[str, object]:
    """Return common strict registration arguments."""
    return {
        "source_result_path": source,
        "destination_path": destination,
        "swap_glm_id": RESULT_ID,
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "dark_epoch": "08_r4",
        "light_train_epoch": "02_r1",
        "light_test_epoch": "04_r2",
        "dark_light_glm_artifact_path": destination.parent / "upstream",
        "spikes": object(),
        "stable_unit_ids": [],
        "movement_interval": FakeIntervals(),
        "movement_analysis_status": "valid",
        "trajectory_intervals": {
            name: FakeIntervals() for name in TRAJECTORY_TYPES
        },
        "graph_inputs_by_trajectory": _graph_inputs(),
        "position": _valid_position(),
        "position_offset_samples": 10,
        "speed_threshold_cm_s": 4.0,
    }


class FakeAnalysis:
    """Small source-compatible facade that records reuse of legacy helpers."""

    DEFAULT_MODEL_NAMES = module.LEGACY_SCHEMA4_MODEL_NAMES
    DERIVED_SELECTED_MODEL_SOURCES = dict(
        module.LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES
    )
    SWAP_CONFIG = dict(module.OUTPUT_RULE["swap_configuration"])

    def __init__(self, *, second_unit_valid=True, missing_trajectory=None):
        self.evaluated = []
        self.prepared_unit_ids = []
        self.second_unit_valid = second_unit_valid
        self.missing_trajectory = missing_trajectory

    def validate_selected_dark_light_glms(self, *_args, **_kwargs):
        return {
            "unit_ids": np.asarray([101, 102]),
            "trajectory": np.asarray(TRAJECTORY_TYPES),
            "tp_grid": np.linspace(0.0, 1.0, 3),
            "segment_edges": np.asarray([0.0, 0.3, 0.7, 1.0]),
            "bin_size_s": 0.02,
            "spatial_bin_size_cm": 4.0,
            "n_splines": 25,
            "spline_order": 4,
            "has_speed": False,
            "speed_feature_mode": "linear",
            "n_speed_features": 0,
            "speed_spline_order": np.nan,
        }

    def _prepare_test_epoch_inputs_for_units(self, **kwargs):
        self.prepared_unit_ids.append(tuple(kwargs["unit_ids"]))
        if kwargs["traj_name"] == self.missing_trajectory:
            return {
                "unit_ids": np.asarray(kwargs["unit_ids"]),
                "y": np.empty((0, len(kwargs["unit_ids"]))),
                "p": np.asarray([], dtype=float),
                "v": None,
            }
        return {
            "unit_ids": np.asarray(kwargs["unit_ids"]),
            "y": np.asarray([[1.0, 2.0], [0.0, 1.0]]),
            "p": np.asarray([0.2, 0.8]),
            "v": None,
        }

    def build_observed_summary(self, y, _p, edges, *, bin_size_s):
        n_bins = len(edges) - 1
        return {
            "occupancy_s": np.ones(n_bins) * bin_size_s,
            "spike_count": np.ones((n_bins, y.shape[1])),
            "observed_rate_hz": np.ones((n_bins, y.shape[1])),
        }

    def evaluate_selected_model_on_test_epoch(self, *, model_name, **_kwargs):
        self.evaluated.append(model_name)
        return {trajectory: {} for trajectory in TRAJECTORY_TYPES}

    def build_selected_swap_dataset(self, **_kwargs):
        unit_ids = np.asarray(
            _kwargs["selected_datasets"]["visual"].coords["unit"].values
        )
        return _result_dataset(
            second_unit_valid=self.second_unit_valid,
            unit_ids=unit_ids,
        )


def _patch_fake_additive_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replace the exact additive evaluator only for facade-based tests."""

    def evaluate(*, analysis, **_kwargs):
        analysis.evaluated.append(module.VISUAL_ADDITIVE_MODEL_NAME)
        return {trajectory: {} for trajectory in TRAJECTORY_TYPES}

    monkeypatch.setattr(
        module,
        "_evaluate_visual_additive_delta_on_test_epoch",
        evaluate,
    )


def _compute(
    monkeypatch: pytest.MonkeyPatch,
    *,
    second_unit_valid=True,
    movement=None,
    movement_status=None,
    upstream_valid=(True, True),
    upstream_status="valid",
    missing_trajectory=None,
):
    upstream = _upstream()
    upstream["selected_units"]["valid_glm_fit"] = list(upstream_valid)
    upstream["analysis_status"] = upstream_status
    upstream["upstream_provenance"]["upstream_analysis_status"] = (
        upstream_status
    )
    fake_analysis = FakeAnalysis(
        second_unit_valid=second_unit_valid,
        missing_trajectory=missing_trajectory,
    )
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    monkeypatch.setattr(module, "_analysis_module", lambda: fake_analysis)
    _patch_fake_additive_evaluator(monkeypatch)
    _patch_identities(monkeypatch)
    result = module.compute_swap_glm(
        swap_glm_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_light_glm_artifact_path=Path("/tmp/upstream"),
        spikes=object(),
        stable_unit_ids=[],
        movement_interval=FakeIntervals() if movement is None else movement,
        movement_analysis_status=movement_status,
        trajectory_intervals={name: FakeIntervals() for name in TRAJECTORY_TYPES},
        graph_inputs_by_trajectory=_graph_inputs(),
        task_progression_by_trajectory={name: object() for name in TRAJECTORY_TYPES},
    )
    return result, fake_analysis


def _historical_schema4_result(result: dict[str, object]) -> dict[str, object]:
    """Return a historical five-model view of one current test result."""
    dataset = result["dataset"].sel(
        model=list(module.LEGACY_SCHEMA4_MODEL_NAMES)
    ).copy(deep=True)
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["schema_version"] = "4"
    dataset.attrs["output_rule_sha256"] = (
        module.LEGACY_SCHEMA4_OUTPUT_RULE_SHA256
    )
    dataset.attrs.pop("prediction_count_clip_eps", None)
    dataset.attrs.pop("visual_empirical_model_definitions_json", None)
    dataset.attrs["derived_model_sources_json"] = json.dumps(
        dict(module.LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES),
        sort_keys=True,
    )
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    fit_parameters["models"] = list(module.LEGACY_SCHEMA4_MODEL_NAMES)
    fit_parameters["derived_model_sources"] = dict(
        module.LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES
    )
    dataset.attrs["fit_parameters_json"] = json.dumps(
        fit_parameters,
        sort_keys=True,
    )
    selected_units, status = module._audit_scores(
        result["selected_units"],
        dataset,
        model_names=module.LEGACY_SCHEMA4_MODEL_NAMES,
    )
    dataset.attrs["analysis_status"] = status
    return {
        **result,
        "parameters": {
            **result["parameters"],
            "output_rule_sha256": module.LEGACY_SCHEMA4_OUTPUT_RULE_SHA256,
        },
        "selected_units": selected_units,
        "dataset": dataset,
        "analysis_status": status,
    }


def test_artifact_path_is_session_first_and_epoch_explicit(tmp_path: Path) -> None:
    paths = module.get_swap_glm_artifact_paths(
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        swap_glm_id=RESULT_ID,
        artifact_root=tmp_path,
    )
    assert paths["artifact_dir"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "swap_glm"
        / "02_r1_train_to_04_r2_test"
        / "dark_08_r4"
        / "v1"
        / str(RESULT_ID)
    )


def test_parameters_preserve_legacy_defaults() -> None:
    parameters = module.validate_swap_glm_parameters()
    assert parameters == {
        "swap_light_offset": False,
        "observed_spatial_bin_size_cm": 4.0,
    }


def test_visual_additive_delta_uses_count_space_delta_and_clips() -> None:
    target_light = np.asarray([[7.0, 8.0], [9.0, 10.0]])
    predicted = module._visual_additive_delta_count(
        target_light_count=target_light,
        target_dark_count=np.asarray([[1.0, 1.0], [4.0, 1.0]]),
        source_light_reference_count=np.asarray(
            [[2.0, 2.0], [9.0, 2.0]]
        ),
        source_dark_reference_count=np.asarray(
            [[3.0, 3.0], [3.0, 8.0]]
        ),
        swap_mask=np.asarray([False, True]),
    )

    np.testing.assert_allclose(predicted[0], target_light[0])
    assert predicted[1, 0] == pytest.approx(10.0)
    assert predicted[1, 1] == pytest.approx(
        module.PREDICTION_COUNT_CLIP_EPS
    )


def test_visual_additive_evaluator_uses_heldout_target_and_reference_source() -> None:
    xr = pytest.importorskip("xarray")
    grid = np.asarray([0.1, 0.9])
    light_by_trajectory = np.asarray(
        [[7.0, 7.0], [9.0, 9.0], [7.0, 7.0], [9.0, 9.0]]
    )[:, :, None]
    dark_by_trajectory = np.asarray(
        [[4.0, 4.0], [3.0, 3.0], [4.0, 4.0], [3.0, 3.0]]
    )[:, :, None]
    dataset = xr.Dataset(
        {
            "train_light_hz_grid": (
                ("trajectory", "tp_grid", "unit"),
                light_by_trajectory,
            ),
            "dark_hz_grid": (
                ("trajectory", "tp_grid", "unit"),
                dark_by_trajectory,
            ),
        },
        coords={
            "trajectory": np.asarray(TRAJECTORY_TYPES),
            "tp_grid": grid,
            "unit": np.asarray(["101"]),
        },
    )

    class AdditiveAnalysis:
        def __init__(self):
            self.calls = []

        @staticmethod
        def _segment_mask(values, edges, segment_index):
            values = np.asarray(values)
            if segment_index == len(edges) - 2:
                return (values >= edges[segment_index]) & (
                    values <= edges[segment_index + 1]
                )
            return (values >= edges[segment_index]) & (
                values < edges[segment_index + 1]
            )

        def predict_selected_light_eta(
            self, *, trajectory, speed_values, p_eval, **_kwargs
        ):
            self.calls.append(("light", trajectory, speed_values))
            count = 7.0 if speed_values is not None else 9.0
            return {"light_eta": np.log(np.full((len(p_eval), 1), count))}

        def predict_selected_dark_eta(
            self, *, trajectory, speed_values, p_eval, **_kwargs
        ):
            self.calls.append(("dark", trajectory, speed_values))
            count = 4.0 if speed_values is not None else 3.0
            return {"dark_eta": np.log(np.full((len(p_eval), 1), count))}

        @staticmethod
        def _selected_var_by_trajectory(dataset, name, trajectory):
            return np.asarray(dataset[name].sel(trajectory=trajectory).values)

        @staticmethod
        def summarize_raw_poisson_metrics(_observed, predicted):
            return {"predicted_count": np.asarray(predicted)}

    analysis = AdditiveAnalysis()
    heldout_speed = np.asarray([2.0, 3.0])
    test_inputs = {
        trajectory: {
            "p": grid,
            "y": np.zeros((2, 1)),
            "v": heldout_speed,
        }
        for trajectory in TRAJECTORY_TYPES
    }
    result = module._evaluate_visual_additive_delta_on_test_epoch(
        analysis=analysis,
        dataset=dataset,
        test_inputs_by_traj=test_inputs,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0]),
        bin_size_s=1.0,
        n_splines=5,
        spline_order=3,
    )

    full_prediction = result["center_to_left"][
        "test_light_full_swapped_metrics"
    ]["predicted_count"][:, 0]
    np.testing.assert_allclose(full_prediction, [7.0, 10.0])
    assert any(
        kind == "light"
        and trajectory == "center_to_left"
        and speed is heldout_speed
        for kind, trajectory, speed in analysis.calls
    )
    assert any(
        kind == "light"
        and trajectory == "center_to_right"
        and speed is None
        for kind, trajectory, speed in analysis.calls
    )


def test_output_rule_matches_reused_analysis_contract() -> None:
    analysis = module._analysis_module()
    module._validate_reused_analysis_contract(analysis)
    assert analysis.BSplineEval is not None
    assert module.OUTPUT_RULE["unit_validity_policy"] == (
        "upstream_valid_glm_fit_and_all_expected_primary_scores_finite"
    )
    assert module.OUTPUT_RULE["trajectory_support_policy"] == (
        "all_or_none_terminal_if_any_path_has_no_movement_bins"
    )
    assert module.OUTPUT_RULE["movement_terminal_status_policy"] == (
        "selected_movement_firing_rate_status_precedes_interval_fallback"
    )


def test_reused_analysis_contract_rejects_drift() -> None:
    drifted = SimpleNamespace(
        DEFAULT_MODEL_NAMES=module.LEGACY_SCHEMA4_MODEL_NAMES,
        DERIVED_SELECTED_MODEL_SOURCES=dict(
            module.LEGACY_SCHEMA4_DERIVED_MODEL_SOURCES
        ),
        SWAP_CONFIG={**dict(module.OUTPUT_RULE["swap_configuration"]), "bad": {}},
    )
    with pytest.raises(ValueError, match="swap configuration"):
        module._validate_reused_analysis_contract(drifted)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_parameters_reject_invalid_observed_bin_size(value: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        module.validate_swap_glm_parameters(observed_spatial_bin_size_cm=value)


def test_compute_reuses_existing_evaluators_and_additive_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake_analysis = _compute(monkeypatch)
    assert fake_analysis.evaluated == list(module.MODEL_NAMES)
    assert result["analysis_status"] == "valid"
    assert result["selected_units"]["valid_swap_score"].tolist() == [True, True]
    assert result["selected_units"]["test_light_spike_count"].tolist() == [12.0, 12.0]
    assert fake_analysis.prepared_unit_ids == [(101, 102)] * len(TRAJECTORY_TYPES)
    assert result["dataset"].coords["unit"].values.tolist() == ["101", "102"]


def test_one_nonfinite_unit_does_not_invalidate_other_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch, second_unit_valid=False)
    assert result["analysis_status"] == "partial_valid"
    audit = result["selected_units"]
    assert audit["valid_swap_score"].tolist() == [True, False]
    assert audit["n_finite_primary_scores"].tolist() == [20, 19]


def test_upstream_invalid_fit_cannot_produce_valid_swap_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch, upstream_valid=(True, False))
    audit = result["selected_units"]
    assert result["analysis_status"] == "partial_valid"
    assert audit["n_finite_primary_scores"].tolist() == [20, 20]
    assert audit["valid_swap_score"].tolist() == [True, False]


def test_compute_records_exact_upstream_identity_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    assert result["upstream_provenance"]["dark_light_glm_id"] == str(UPSTREAM_ID)
    assert result["upstream_provenance"]["dark_light_manifest_sha256"] == "a" * 64
    assert result["selected_units"]["stable_unit_id"].tolist() == [
        "merge-a:1",
        "merge-a:2",
    ]


def test_compute_rejects_heldout_identity_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: _upstream())
    _patch_identities(monkeypatch, conflict=True)
    with pytest.raises(ValueError, match="conflicting stable_unit_id"):
        module.compute_swap_glm(
            swap_glm_id=RESULT_ID,
            animal_name="L14",
            date="20240611",
            region="v1",
            dark_epoch="08_r4",
            light_train_epoch="02_r1",
            light_test_epoch="04_r2",
            dark_light_glm_artifact_path=Path("/tmp/upstream"),
            spikes=object(),
            stable_unit_ids=[],
            movement_interval=FakeIntervals(),
            trajectory_intervals={name: FakeIntervals() for name in TRAJECTORY_TYPES},
            graph_inputs_by_trajectory=_graph_inputs(),
            task_progression_by_trajectory={name: object() for name in TRAJECTORY_TYPES},
        )


def test_empty_movement_is_persistable_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake_analysis = _compute(
        monkeypatch,
        movement=FakeIntervals(starts=(), ends=()),
    )
    assert result["analysis_status"] == "no_movement"
    assert result["dataset"].attrs["fit_stage"] == "terminal"
    assert fake_analysis.evaluated == []


def test_selected_movement_terminal_status_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake_analysis = _compute(
        monkeypatch,
        movement=FakeIntervals(starts=(), ends=()),
        movement_status="no_valid_position",
    )
    assert result["analysis_status"] == "no_valid_position"
    assert fake_analysis.evaluated == []


def test_upstream_terminal_status_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake_analysis = _compute(
        monkeypatch,
        upstream_status="no_movement",
    )
    assert result["analysis_status"] == "upstream_terminal"
    assert result["upstream_provenance"]["upstream_analysis_status"] == (
        "no_movement"
    )
    assert result["dataset"].attrs["upstream_analysis_status"] == "no_movement"
    assert fake_analysis.evaluated == []


def test_missing_one_trajectory_is_an_all_or_none_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, fake_analysis = _compute(
        monkeypatch,
        missing_trajectory="left_to_center",
    )
    assert result["analysis_status"] == "no_trajectory_samples"
    assert result["dataset"].attrs["missing_trajectory"] == "left_to_center"
    assert fake_analysis.evaluated == []


def test_empty_position_is_persistable_terminal_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _upstream()
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    _patch_identities(monkeypatch)
    result = module.compute_swap_glm(
        swap_glm_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_light_glm_artifact_path=Path("/tmp/upstream"),
        spikes=object(),
        stable_unit_ids=[],
        movement_interval=FakeIntervals(),
        trajectory_intervals={name: FakeIntervals() for name in TRAJECTORY_TYPES},
        graph_inputs_by_trajectory=_graph_inputs(),
        position=SimpleNamespace(t=np.asarray([]), d=np.empty((0, 2))),
    )
    assert result["analysis_status"] == "no_valid_position"
    assert result["dataset"].attrs["fit_stage"] == "terminal"


def test_one_finite_position_sample_is_not_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _upstream()
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    _patch_identities(monkeypatch)
    result = module.compute_swap_glm(
        swap_glm_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_light_glm_artifact_path=Path("/tmp/upstream"),
        spikes=object(),
        stable_unit_ids=[],
        movement_interval=FakeIntervals(starts=(), ends=()),
        trajectory_intervals={name: FakeIntervals() for name in TRAJECTORY_TYPES},
        graph_inputs_by_trajectory=_graph_inputs(),
        position=SimpleNamespace(
            t=np.asarray([1.0]),
            d=np.asarray([[0.0, 0.0]]),
        ),
    )
    assert result["analysis_status"] == "no_valid_position"


def test_invalid_provided_position_precedes_empty_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _upstream()
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    _patch_identities(monkeypatch)
    result = module.compute_swap_glm(
        swap_glm_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_light_glm_artifact_path=Path("/tmp/upstream"),
        spikes=object(),
        stable_unit_ids=[],
        movement_interval=FakeIntervals(starts=(), ends=()),
        trajectory_intervals={name: FakeIntervals() for name in TRAJECTORY_TYPES},
        graph_inputs_by_trajectory=_graph_inputs(),
        position=SimpleNamespace(t=np.asarray([]), d=np.empty((0, 2))),
        task_progression_by_trajectory={
            name: object() for name in TRAJECTORY_TYPES
        },
    )
    assert result["analysis_status"] == "no_valid_position"


def test_write_load_roundtrip_and_checksum_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch, second_unit_valid=False)
    destination = tmp_path / str(RESULT_ID)
    paths = module.write_swap_glm_artifact(result, destination)
    loaded = module.load_swap_glm_artifact(destination)
    assert loaded["analysis_status"] == "partial_valid"
    assert loaded["selected_units_sha256"] == result["selected_units_sha256"]
    assert paths["result_path"].is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.write_swap_glm_artifact(result, destination)
    paths["result_path"].write_bytes(paths["result_path"].read_bytes() + b"bad")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_swap_glm_artifact(destination)


@pytest.mark.parametrize("terminal", [False, True])
def test_analysis_nwb_objects_roundtrip_valid_and_terminal_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: bool,
) -> None:
    """All seven SwapGLM scratch objects survive an NWB HDF5 round trip."""
    pynwb = pytest.importorskip("pynwb")
    result, _ = _compute(monkeypatch, second_unit_valid=False)
    if terminal:
        selected_units = result["selected_units"].iloc[0:0].copy()
        terminal_dataset = module._terminal_dataset(
            metadata=result["metadata"],
            unit_ids=np.asarray([], dtype=str),
            segment_edges=np.asarray(
                result["dataset"].coords["segment_edge"].values,
                dtype=float,
            ),
            parameters=result["parameters"],
            upstream_provenance=result["upstream_provenance"],
            analysis_status="no_units",
        )
        result = module.validate_swap_glm_result(
            {
                **result,
                "selected_units": selected_units,
                "dataset": terminal_dataset,
                "analysis_status": "no_units",
            }
        )
    expected_hashes = module.swap_glm_nwb_hashes(result)
    objects = module.swap_glm_result_to_nwb_objects(result)
    assert set(objects) == {
        "selected_units",
        "model_metadata",
        "axes",
        "trajectory_metadata",
        "model_results",
        "observed_response",
        "provenance",
    }
    assert len({value.object_id for value in objects.values()}) == 7
    nwbfile = pynwb.NWBFile(
        session_description="SwapGLM NWB round-trip",
        identifier=f"swap-glm-{terminal}",
        session_start_time=datetime.now(timezone.utc),
    )
    for value in objects.values():
        nwbfile.add_scratch(value)
    path = tmp_path / f"swap_glm_{terminal}.nwb"
    with pynwb.NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)
    assert pynwb.validate(path=path) == []
    with pynwb.NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        stored = io.read()
        loaded = module.swap_glm_result_from_nwb_objects(
            **{
                name: stored.objects[value.object_id]
                for name, value in objects.items()
            }
        )
    assert loaded["analysis_status"] == result["analysis_status"]
    assert module.swap_glm_nwb_hashes(loaded) == expected_hashes


def test_load_preserves_historical_schema4_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, _ = _compute(monkeypatch)
    historical = module.validate_swap_glm_result(
        _historical_schema4_result(current)
    )
    destination = tmp_path / str(RESULT_ID)
    destination.mkdir()
    historical["selected_units"].to_parquet(
        destination / module.SELECTED_UNITS_FILENAME,
        index=False,
    )
    historical["dataset"].to_netcdf(destination / module.RESULT_FILENAME)
    common = module._manifest_common(historical)
    rows = []
    for artifact_key, filename, artifact_kind in (
        ("selected_units", module.SELECTED_UNITS_FILENAME, "parquet"),
        ("swap_glm", module.RESULT_FILENAME, "netcdf"),
    ):
        artifact_path = destination / filename
        rows.append(
            {
                "artifact_key": artifact_key,
                "relative_path": filename,
                "artifact_kind": artifact_kind,
                "file_size_bytes": artifact_path.stat().st_size,
                "sha256": module._file_sha256(artifact_path),
                **common,
            }
        )
    pd.DataFrame.from_records(
        rows,
        columns=module.MANIFEST_COLUMNS,
    ).to_parquet(destination / module.MANIFEST_FILENAME, index=False)

    loaded = module.load_swap_glm_artifact(destination)

    assert loaded["historical_schema_version"] == "4"
    assert loaded["dataset"].coords["model"].values.tolist() == list(
        module.LEGACY_SCHEMA4_MODEL_NAMES
    )


def test_validate_rejects_dataset_unit_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    tampered["dataset"] = result["dataset"].sel(unit=["102", "101"])
    with pytest.raises(ValueError, match="unit order"):
        module.validate_swap_glm_result(tampered)


def test_validate_rejects_parameter_hash_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    tampered["parameters"] = {**result["parameters"], "parameter_sha256": "0" * 64}
    with pytest.raises(ValueError, match="parameter_sha256"):
        module.validate_swap_glm_result(tampered)


def test_validate_recomputes_spike_count_audit_from_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    tampered["selected_units"] = result["selected_units"].copy()
    tampered["selected_units"].loc[0, "test_light_spike_count"] += 1.0
    with pytest.raises(ValueError, match="does not match the swap dataset"):
        module.validate_swap_glm_result(tampered)


def test_validate_recomputes_finite_score_audit_from_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    tampered["dataset"] = result["dataset"].copy(deep=True)
    metric = (
        "test_light_swapped_segment_swapped_delta_model_minus_visual_"
        "raw_ll_bits_per_spike"
    )
    tampered["dataset"][metric].loc[
        {"model": "dark", "trajectory": "center_to_left", "unit": "102"}
    ] = np.nan
    with pytest.raises(ValueError, match="score arithmetic"):
        module.validate_swap_glm_result(tampered)


def test_validate_rejects_upstream_hash_attribute_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    tampered["dataset"] = result["dataset"].copy(deep=True)
    tampered["dataset"].attrs["dark_light_parameter_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dark_light_parameter_sha256"):
        module.validate_swap_glm_result(tampered)


def test_validate_rejects_incomplete_nonterminal_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    tampered["dataset"] = result["dataset"].drop_vars("dark_hz_grid")
    with pytest.raises(ValueError, match="complete schema"):
        module.validate_swap_glm_result(tampered)


def test_validate_rejects_full_schema_dimension_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _compute(monkeypatch)
    tampered = dict(result)
    dataset = result["dataset"].drop_vars("selected_ridge")
    dataset["selected_ridge"] = (
        ("trajectory",),
        np.ones(len(TRAJECTORY_TYPES)),
    )
    tampered["dataset"] = dataset
    with pytest.raises(ValueError, match="noncanonical dimensions"):
        module.validate_swap_glm_result(tampered)


def test_schema6_normalization_selects_core_and_drops_only_diagnostics() -> None:
    source = _schema6_dataset()
    normalized, audit = module._normalize_legacy_swap_dataset(source)
    assert normalized.coords["model"].values.tolist() == list(module.MODEL_NAMES)
    assert set(normalized.data_vars) == set(module.CANONICAL_DATA_VARIABLE_DIMS)
    assert not set(module.LEGACY_SCHEMA6_DIAGNOSTIC_VARIABLES).intersection(
        normalized.data_vars
    )
    assert audit["source_schema_version"] == "6"
    assert audit["dark_score_source"] == "legacy_source_and_exact_nwb_rescore"


def test_schema6_normalization_rejects_fixed_mapping_tampering() -> None:
    source = _schema6_dataset()
    source.attrs["derived_model_sources_json"] = json.dumps(
        {"dark": "task_dense_gain"}
    )
    with pytest.raises(ValueError, match="derived-model mapping"):
        module._normalize_legacy_swap_dataset(source)


def test_legacy_selected_source_verification_checks_each_file(tmp_path: Path) -> None:
    paths = {}
    hashes = {}
    for model_name in module.SOURCE_MODEL_NAMES:
        path = tmp_path / f"{model_name}.nc"
        path.write_bytes(model_name.encode())
        paths[model_name] = str(path)
        hashes[model_name] = module._file_sha256(path)
    paths["dark"] = paths["task_segment_bump"]
    paths[module.VISUAL_ADDITIVE_MODEL_NAME] = paths["visual"]
    xr = pytest.importorskip("xarray")
    dataset = xr.Dataset(
        attrs={
            "sources_json": json.dumps({"dark_light_glm_selected": paths})
        }
    )
    provenance = {"dark_light_selected_sha256_by_model": hashes}
    verified = module._verify_legacy_selected_sources(
        dataset,
        provenance,
        model_names=module.MODEL_NAMES,
    )
    assert verified["dark"] == hashes["task_segment_bump"]
    (tmp_path / "visual.nc").write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match upstream"):
        module._verify_legacy_selected_sources(
            dataset,
            provenance,
            model_names=module.MODEL_NAMES,
        )


def test_register_existing_validates_and_copies_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _upstream()
    fake_analysis = FakeAnalysis()
    selected_paths, selected_hashes = _write_legacy_selected_sources(tmp_path)
    upstream["upstream_provenance"][
        "dark_light_legacy_selected_sha256_by_model"
    ] = selected_hashes
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    monkeypatch.setattr(module, "_analysis_module", lambda: fake_analysis)
    _patch_fake_additive_evaluator(monkeypatch)
    monkeypatch.setattr(
        module,
        "_derive_task_progression",
        lambda **_kwargs: {name: object() for name in TRAJECTORY_TYPES},
    )
    _patch_identities(monkeypatch)
    source = tmp_path / "legacy.nc"
    _result_dataset(
        unit_ids=(11, 12),
        selected_source_paths=selected_paths,
    ).to_netcdf(source)
    destination = tmp_path / "canonical" / str(RESULT_ID)
    registered = module.register_existing_swap_glm_artifact(
        source_result_path=source,
        destination_path=destination,
        swap_glm_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="04_r2",
        dark_light_glm_artifact_path=tmp_path / "upstream",
        spikes=object(),
        stable_unit_ids=[],
        movement_interval=FakeIntervals(),
        movement_analysis_status="valid",
        trajectory_intervals={name: FakeIntervals() for name in TRAJECTORY_TYPES},
        graph_inputs_by_trajectory=_graph_inputs(),
        position=_valid_position(),
        position_offset_samples=10,
        speed_threshold_cm_s=4.0,
        source_v1ca1_git_commit="abc123",
    )
    assert registered["artifact_origin"] == "registered_existing"
    assert registered["legacy_artifact_provenance"]["source_v1ca1_git_commit"] == "abc123"
    assert registered["legacy_artifact_provenance"][
        "unit_coordinate_remapped"
    ] is True
    assert registered["dataset"].coords["unit"].values.astype(str).tolist() == [
        "101",
        "102",
    ]
    assert module.load_swap_glm_artifact(destination)["analysis_status"] == "valid"


@pytest.mark.parametrize("schema", ["6", "4-full", "4-reduced"])
def test_register_existing_normalizes_legacy_schema_and_exactly_rescores(
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, fake_analysis = _prepare_registration_source(
        tmp_path,
        monkeypatch,
        schema=schema,
    )
    destination = tmp_path / "registered" / str(RESULT_ID)
    registered = module.register_existing_swap_glm_artifact(
        **_register_kwargs(source, destination)
    )
    assert fake_analysis.evaluated == list(module.MODEL_NAMES)
    assert set(registered["dataset"].data_vars) == set(
        module.CANONICAL_DATA_VARIABLE_DIMS
    )
    normalization = registered["legacy_artifact_provenance"][
        "source_normalization"
    ]
    preprocessing = registered["legacy_artifact_provenance"][
        "preprocessing_provenance"
    ]
    assert preprocessing["selected_position_offset_samples"] == 10
    assert preprocessing["selected_speed_threshold_cm_s"] == 4.0
    assert preprocessing["legacy_fields_required"] is True
    if schema == "6":
        assert normalization["compared_model_names"] == list(module.MODEL_NAMES)
        assert normalization["dark_score_source"] == (
            "legacy_source_and_exact_nwb_rescore"
        )
    elif schema == "4-full":
        assert normalization["compared_model_names"] == list(
            module.LEGACY_SCHEMA4_MODEL_NAMES
        )
        assert normalization["dark_score_source"] == (
            "legacy_source_and_exact_nwb_rescore"
        )
    else:
        assert normalization["compared_model_names"] == list(
            module.SOURCE_MODEL_NAMES
        )
        assert normalization["dark_score_source"] == (
            "exact_nwb_rescore_from_verified_task_segment_bump_not_legacy_clone"
        )
        assert registered["dataset"].coords["model"].values.tolist() == list(
            module.MODEL_NAMES
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    [
        ("position_offset", 9, "position_offset differs"),
        ("position_offset", 10.5, "position_offset must be a finite integer"),
        ("speed_threshold_cm_s", 3.5, "speed_threshold_cm_s differs"),
    ],
)
@pytest.mark.parametrize("schema", ["6", "4-full", "4-reduced"])
def test_register_existing_rejects_legacy_preprocessing_provenance_mismatch(
    schema: str,
    field: str,
    bad_value: float,
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _prepare_registration_source(
        tmp_path,
        monkeypatch,
        schema=schema,
    )
    dataset = module._load_dataset(source)
    fit_parameters = json.loads(dataset.attrs["fit_parameters_json"])
    fit_parameters[field] = bad_value
    dataset.attrs["fit_parameters_json"] = json.dumps(
        fit_parameters,
        sort_keys=True,
    )
    dataset.to_netcdf(source)
    with pytest.raises(ValueError, match=error):
        module.register_existing_swap_glm_artifact(
            **_register_kwargs(
                source,
                tmp_path / "registered" / str(RESULT_ID),
            )
        )


def test_register_existing_rejects_scientific_rescore_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _ = _prepare_registration_source(
        tmp_path,
        monkeypatch,
        schema="4-reduced",
    )
    dataset = module._load_dataset(source)
    dataset["dark_hz_grid"].loc[
        {
            "model": "visual",
            "trajectory": "center_to_left",
            "tp_grid": 0.0,
            "unit": 11,
        }
    ] += 0.5
    dataset.to_netcdf(source)
    with pytest.raises(ValueError, match="differs from exact NWB re-score"):
        module.register_existing_swap_glm_artifact(
            **_register_kwargs(
                source,
                tmp_path / "registered" / str(RESULT_ID),
            )
        )


def test_register_existing_rejects_wrong_graph_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _upstream()
    selected_paths, selected_hashes = _write_legacy_selected_sources(tmp_path)
    upstream["upstream_provenance"][
        "dark_light_legacy_selected_sha256_by_model"
    ] = selected_hashes
    monkeypatch.setattr(module, "_load_dark_light_input", lambda _path: upstream)
    monkeypatch.setattr(module, "_analysis_module", lambda: FakeAnalysis())
    monkeypatch.setattr(
        module,
        "_derive_task_progression",
        lambda **_kwargs: {name: object() for name in TRAJECTORY_TYPES},
    )
    _patch_identities(monkeypatch)
    source = tmp_path / "legacy.nc"
    _result_dataset(
        unit_ids=(11, 12),
        selected_source_paths=selected_paths,
    ).to_netcdf(source)
    graphs = _graph_inputs()
    graphs["center_to_left"]["track_graph_kwargs"]["node_positions"][5][0] = 55.0
    with pytest.raises(ValueError, match="common path length|geometry"):
        module.register_existing_swap_glm_artifact(
            source_result_path=source,
            destination_path=tmp_path / str(RESULT_ID),
            swap_glm_id=RESULT_ID,
            animal_name="L14",
            date="20240611",
            region="v1",
            dark_epoch="08_r4",
            light_train_epoch="02_r1",
            light_test_epoch="04_r2",
            dark_light_glm_artifact_path=tmp_path / "upstream",
                spikes=object(),
                stable_unit_ids=[],
                movement_interval=FakeIntervals(),
                movement_analysis_status="valid",
                trajectory_intervals={
                name: FakeIntervals() for name in TRAJECTORY_TYPES
            },
            graph_inputs_by_trajectory=graphs,
            position=_valid_position(),
            position_offset_samples=10,
            speed_threshold_cm_s=4.0,
        )


def test_epochs_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        module._metadata(
            swap_glm_id=RESULT_ID,
            animal_name="L14",
            date="20240611",
            region="v1",
            dark_epoch="08_r4",
            light_train_epoch="02_r1",
            light_test_epoch="02_r1",
        )
