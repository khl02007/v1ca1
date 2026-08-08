"""Focused tests for database-free motor-encoding artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import motor_encoding as motor
from v1ca1.spyglass.movement import empty_movement_firing_rate_table
from v1ca1.spyglass.selection import provenance_sha256
from v1ca1.spyglass.table_specs import (
    MOTOR_ENCODING_MODEL_SPEC,
    MOTOR_ENCODING_OUTPUT_RULE,
)


class _Position:
    def __init__(self, times=(0.0, 0.1), values=((0.0, 0.0), (1.0, 0.0))):
        self.t = np.asarray(times, dtype=float)
        self.d = np.asarray(values, dtype=float)


class _Intervals:
    def __init__(self, duration: float):
        self._duration = float(duration)

    def tot_length(self) -> float:
        return self._duration


def _graph(
    name: str,
    *,
    node_positions: list[list[float]],
    edge_order: list[tuple[int, int]],
    edge_spacing: list[float],
) -> dict[str, object]:
    edges = np.asarray(edge_order, dtype=int)
    positions = np.asarray(node_positions, dtype=float)
    return {
        "configuration_name": name,
        "coordinate_unit": "cm",
        "track_graph_kwargs": {
            "node_positions": positions,
            "edges": edges,
        },
        "linearization_kwargs": {
            "edge_order": list(edge_order),
            "edge_spacing": list(edge_spacing),
            "use_HMM": False,
        },
    }


def _graphs(*, full_length_scale: float = 1.0) -> dict[str, dict[str, object]]:
    path = {
        name: _graph(
            name,
            node_positions=[[0.0, 0.0], [3.0, 0.0]],
            edge_order=[(0, 1)],
            edge_spacing=[],
        )
        for name in TRAJECTORY_TYPES
    }
    path["full_w"] = _graph(
        "full_w",
        node_positions=[
            [0.0, 0.0],
            [3.0 * full_length_scale, 0.0],
            [7.0 * full_length_scale, 0.0],
        ],
        edge_order=[(0, 1), (1, 2)],
        edge_spacing=[15.0],
    )
    return path


def _parameter_bundle(name: str = "test") -> dict[str, object]:
    effective = motor.validate_motor_encoding_parameters(
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
    )
    return motor._parameter_metadata(
        parameter_name=name,
        parameter_sha256=None,
        model_spec_sha256=None,
        output_rule_sha256=None,
        parameters=effective,
    )


def _metadata() -> dict[str, str]:
    return motor._common_metadata(
        motor_encoding_id=uuid.uuid4(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
    )


def _identity() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "stable_unit_id": "merge-a:11",
                "group_unit_id": "0",
            },
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "12",
                "stable_unit_id": "merge-a:12",
                "group_unit_id": "1",
            },
        ]
    )


def _stability_tables(
    correlations: tuple[float, float] = (0.7, 0.2),
) -> dict[str, pd.DataFrame]:
    """Return four aligned path-stability tables for two test units."""
    return {
        trajectory_type: pd.DataFrame.from_records(
            [
                {
                    "spikesorting_merge_id": "merge-a",
                    "unit_id": unit_id,
                    "stable_unit_id": f"merge-a:{unit_id}",
                    "group_unit_id": group_unit_id,
                    "animal_name": "L14",
                    "date": "20240611",
                    "region": "v1",
                    "epoch": "08_r4",
                    "trajectory_type": trajectory_type,
                    "firing_rate_hz": firing_rate_hz,
                    "stability_correlation": correlation,
                    "stability_status": "valid",
                }
                for unit_id, group_unit_id, firing_rate_hz, correlation in (
                    ("11", 0, 1.0, correlations[0]),
                    ("12", 1, 0.1, correlations[1]),
                )
            ]
        )
        for trajectory_type in TRAJECTORY_TYPES
    }


def _raw_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    nested = xr.Dataset(
        data_vars={
            "outer_unit_selected": (
                ("outer_fold", "unit"),
                np.asarray([[1, 0]] * 5, dtype=np.int8),
            ),
            "outer_info_bits_per_spike": (
                ("outer_fold", "model", "unit"),
                np.concatenate(
                    (
                        np.ones((5, 9, 1), dtype=float),
                        np.full((5, 9, 1), np.nan),
                    ),
                    axis=2,
                ),
            ),
            "pooled_info_bits_per_spike": (
                ("model", "unit"),
                np.concatenate(
                    (
                        np.ones((9, 1), dtype=float),
                        np.full((9, 1), np.nan),
                    ),
                    axis=1,
                ),
            ),
            "pooled_spike_sum": ("unit", np.asarray([20.0, 0.0])),
            "outer_train_bin_count": (
                "outer_fold",
                np.full(5, 100, dtype=int),
            ),
            "outer_test_bin_count": (
                "outer_fold",
                np.full(5, 20, dtype=int),
            ),
        },
        coords={
            "outer_fold": np.arange(5),
            "model": np.asarray(motor.MODEL_NAMES),
            "unit": np.asarray(["0", "1"]),
        },
    )
    full = xr.Dataset(
        data_vars={
            "selected_ridge": ("model", np.full(9, 0.1)),
            "movement_firing_rate_hz": ("unit", np.asarray([1.0])),
        },
        coords={
            "model": np.asarray(motor.MODEL_NAMES),
            "unit": np.asarray(["0"]),
        },
    )
    return nested, full


def _valid_result() -> dict[str, object]:
    metadata = _metadata()
    parameters = _parameter_bundle()
    identity = _identity()
    nested, full = _raw_datasets()
    eligibility = motor._build_unit_eligibility_table(
        identity=identity,
        movement_firing_rates_hz=np.asarray([1.0, 0.1]),
        stability_tables_by_trajectory=_stability_tables(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
    )
    selected = motor._build_selected_units_table(
        eligibility=eligibility,
        nested_cv=nested,
        full_refit=full,
    )
    nested = motor._canonicalize_computed_dataset(
        nested,
        role="nested_cv",
        selected_units=selected,
        metadata=metadata,
        parameters=parameters,
        primary_position_source="head",
        orientation_reference_position_source="body",
        artifact_origin="computed",
    )
    full = motor._canonicalize_computed_dataset(
        full,
        role="full_refit",
        selected_units=selected,
        metadata=metadata,
        parameters=parameters,
        primary_position_source="head",
        orientation_reference_position_source="body",
        artifact_origin="computed",
    )
    from v1ca1.spyglass.selection import unit_identity_sha256

    return motor.validate_motor_encoding_result(
        {
            "metadata": metadata,
            "parameters": parameters,
            "selected_units": selected,
            "nested_cv": nested,
            "full_refit": full,
            "n_units_input": 2,
            "n_units_eligible": 1,
            "n_units_valid": 1,
            "n_outer_folds_expected": 5,
            "n_outer_folds_valid": 5,
            "selected_units_sha256": unit_identity_sha256(
                selected.loc[
                    :, ["spikesorting_merge_id", "unit_id"]
                ].to_dict("records")
            ),
            "analysis_status": "valid",
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
    )


def _movement_table() -> pd.DataFrame:
    table = empty_movement_firing_rate_table()
    return pd.DataFrame.from_records(
        [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": unit_id,
                "stable_unit_id": f"merge-a:{unit_id}",
                "group_unit_id": group_id,
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "epoch": "08_r4",
                "movement_spike_count": spike_count,
                "movement_duration_s": 10.0,
                "movement_firing_rate_hz": spike_count / 10.0,
                "firing_rate_status": "valid",
                "position_sample_count": 100,
                "finite_position_sample_count": 100,
                "finite_speed_sample_count": 100,
                "movement_interval_count": 1,
                "speed_threshold_cm_s": 4.0,
                "speed_smoothing_sigma_s": 0.1,
            }
            for unit_id, group_id, spike_count in (("11", 0, 10), ("12", 1, 1))
        ],
        columns=table.columns,
    )


def test_model_spec_and_parameter_digest_match_table_contract() -> None:
    assert dict(motor.MODEL_SPEC) == dict(MOTOR_ENCODING_MODEL_SPEC)
    assert motor.MODEL_SPEC_SHA256 == provenance_sha256(
        dict(MOTOR_ENCODING_MODEL_SPEC)
    )
    assert dict(motor.OUTPUT_RULE) == dict(MOTOR_ENCODING_OUTPUT_RULE)
    assert motor.OUTPUT_RULE_SHA256 == provenance_sha256(
        dict(MOTOR_ENCODING_OUTPUT_RULE)
    )
    parameters = _parameter_bundle("preset")
    raw = {
        "motor_encoding_param_name": "preset",
        **{
            key: value
            for key, value in parameters.items()
            if key
            not in {
                "parameter_name",
                "parameter_sha256",
                "model_spec_sha256",
                "output_rule_sha256",
            }
        },
    }
    assert parameters["parameter_sha256"] == provenance_sha256(raw)


def test_graph_basis_uses_selected_graph_lengths(monkeypatch) -> None:
    calls: list[dict[str, float]] = []

    def build(**kwargs):
        calls.append(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(
        motor,
        "_motor_module",
        lambda: SimpleNamespace(build_position_basis_config_from_lengths=build),
    )
    configs = motor.build_graph_derived_position_basis_configs(
        _graphs(),
        spatial_bin_sizes_cm=(2.0, 4.0),
        spline_order=4,
        generalized_place_branch_gap_cm=15.0,
    )
    assert len(configs) == 2
    assert calls[0]["trajectory_length_cm"] == pytest.approx(3.0)
    assert calls[0]["generalized_place_length_cm"] == pytest.approx(22.0)
    changed = _graphs()
    changed["full_w"]["linearization_kwargs"]["edge_spacing"] = [14.0]
    with pytest.raises(ValueError, match="does not match"):
        motor.build_graph_derived_position_basis_configs(
            changed,
            spatial_bin_sizes_cm=(2.0,),
            spline_order=4,
            generalized_place_branch_gap_cm=15.0,
        )


def test_position_sources_require_exact_timestamp_alignment() -> None:
    motor.validate_position_pair(_Position(), _Position())
    with pytest.raises(ValueError, match="timestamps must match exactly"):
        motor.validate_position_pair(
            _Position(),
            _Position(times=(0.0, 0.100000000001)),
        )


def test_write_load_round_trip_and_checksum(tmp_path: Path) -> None:
    result = _valid_result()
    paths = motor.get_motor_encoding_artifact_paths(
        animal_name=result["metadata"]["animal_name"],
        date=result["metadata"]["date"],
        epoch=result["metadata"]["epoch"],
        region=result["metadata"]["region"],
        motor_encoding_id=result["metadata"][
            "motor_encoding_id"
        ],
        artifact_root=tmp_path,
    )
    written = motor.write_motor_encoding_artifact(
        result,
        paths["artifact_dir"],
    )
    assert written["nested_cv_path"].is_file()
    loaded = motor.load_motor_encoding_artifact(paths["artifact_dir"])
    assert loaded["n_units_input"] == 2
    assert loaded["n_units_eligible"] == 1
    assert loaded["nested_cv"].unit.values.tolist() == [
        "merge-a:11",
        "merge-a:12",
    ]
    with pytest.raises(FileExistsError):
        motor.write_motor_encoding_artifact(result, paths["artifact_dir"])
    paths["selected_units_path"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        motor.load_motor_encoding_artifact(paths["artifact_dir"])


def test_compute_no_units_returns_terminal_bundle() -> None:
    result = motor.compute_motor_encoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        motor_encoding_id=uuid.uuid4(),
        spikes={},
        stable_unit_ids=[],
        primary_position=_Position(),
        orientation_reference_position=_Position(),
        primary_position_source="head",
        orientation_reference_position_source="body",
        trajectory_intervals_by_type={},
        graph_inputs_by_configuration={},
        movement_intervals=_Intervals(0.0),
        movement_firing_rate_table=empty_movement_firing_rate_table(),
        stability_tables_by_trajectory={
            name: pd.DataFrame() for name in TRAJECTORY_TYPES
        },
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
    )
    assert result["analysis_status"] == "no_units"
    assert result["n_units_input"] == 0
    assert result["n_outer_folds_valid"] == 0


def test_compute_orchestrates_existing_motor_helpers(monkeypatch) -> None:
    nested, full = _raw_datasets()
    calls: dict[str, object] = {}

    def basis(**kwargs):
        value = float(kwargs["spatial_bin_size_cm"])
        return {
            **kwargs,
            "trajectory_n_splines": max(4, int(np.ceil(3.0 / value))),
            "generalized_place_n_splines": max(4, int(np.ceil(22.0 / value))),
            "trajectory_bounds": (0.0, 1.0),
            "generalized_place_bounds": (0.0, 22.0),
        }

    def prepare(**kwargs):
        calls["prepare"] = kwargs
        return {"unit_ids": np.asarray([0, 1])}

    def nested_cv(*args, **kwargs):
        calls["nested"] = kwargs
        return {"nested": True}

    def hyper(*args, **kwargs):
        calls["hyper"] = kwargs
        return {
            "selected_ridge": np.full(9, 0.1),
            "selected_spatial_index": np.zeros(9, dtype=int),
        }

    def full_fit(*args, **kwargs):
        calls["full_fit"] = kwargs
        return {"full": True}

    fake = SimpleNamespace(
        build_position_basis_config_from_lengths=basis,
        get_unit_mask=lambda rates, threshold: np.asarray(rates) > threshold,
        summarize_lap_cv_feasibility=lambda *args, **kwargs: (True, [], {}),
        prepare_motor_epoch_data=prepare,
        build_lap_cv_folds_for_epoch=lambda *args, **kwargs: [object()] * 5,
        run_nested_lap_cv=nested_cv,
        build_nested_cv_dataset=lambda *args, **kwargs: nested.copy(deep=True),
        compute_hyperparameter_cv_scores=hyper,
        fit_full_refit_models=full_fit,
        build_full_refit_dataset=lambda *args, **kwargs: full.copy(deep=True),
    )
    monkeypatch.setattr(motor, "_motor_module", lambda: fake)
    monkeypatch.setattr(
        motor,
        "build_motor_model_features",
        lambda **kwargs: {
            "generalized_place_position": object(),
            "generalized_task_progression": object(),
            "task_progression_by_trajectory": {
                name: object() for name in TRAJECTORY_TYPES
            },
        },
    )
    result = motor.compute_motor_encoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        motor_encoding_id=uuid.uuid4(),
        spikes={0: object(), 1: object()},
        stable_unit_ids=[
            {"spikesorting_merge_id": "merge-a", "unit_id": "11"},
            {"spikesorting_merge_id": "merge-a", "unit_id": "12"},
        ],
        primary_position=_Position(),
        orientation_reference_position=_Position(),
        primary_position_source="head",
        orientation_reference_position_source="body",
        trajectory_intervals_by_type={name: object() for name in TRAJECTORY_TYPES},
        graph_inputs_by_configuration=_graphs(),
        movement_intervals=_Intervals(10.0),
        movement_firing_rate_table=_movement_table(),
        stability_tables_by_trajectory=_stability_tables(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
        speed_smoothing_sigma_s=0.2,
    )
    assert result["analysis_status"] == "valid"
    assert result["n_units_input"] == 2
    assert result["n_units_eligible"] == 1
    assert calls["prepare"]["speed_smoothing_sigma_s"] == pytest.approx(0.2)
    assert calls["nested"]["min_firing_rate_hz"] == pytest.approx(0.5)
    assert calls["nested"]["allowed_unit_mask"].tolist() == [True, False]
    assert calls["hyper"]["unit_mask"].tolist() == [True, False]
    assert calls["nested"]["isolate_unit_failures"] is True
    assert calls["hyper"]["isolate_unit_failures"] is True
    assert calls["full_fit"]["isolate_unit_failures"] is True


def test_legacy_graph_geometry_mismatch_is_rejected(monkeypatch) -> None:
    metadata = _metadata()
    parameters = _parameter_bundle()
    nested, full = _raw_datasets()
    fit_parameters = {
        "bin_size_s": 0.05,
        "n_folds": 5,
        "inner_n_folds": 3,
        "seed": 0,
        "ridges": list(motor.DEFAULT_RIDGE_VALUES),
        "spatial_bin_sizes_cm": list(motor.DEFAULT_SPATIAL_BIN_SIZES_CM),
        "motor_feature_mode": "zscore",
        "motor_zscore_eps": 1e-12,
        "motor_spline_k": 5,
        "motor_spline_order": 4,
        "tp_spline_order": 4,
        "speed_sigma_s": 0.1,
        "generalized_place_branch_gap_cm": 15.0,
        "position_basis_configs": [
            {
                "spatial_bin_size_cm": value,
                "spline_order": 4,
                "trajectory_length_cm": 3.0,
                "generalized_place_length_cm": 22.0,
                "trajectory_n_splines": max(4, int(np.ceil(3.0 / value))),
                "generalized_place_n_splines": max(
                    4, int(np.ceil(22.0 / value))
                ),
                "generalized_place_branch_gap_cm": 15.0,
            }
            for value in motor.DEFAULT_SPATIAL_BIN_SIZES_CM
        ],
    }
    attrs = {
        "schema_version": "2",
        **{name: metadata[name] for name in ("animal_name", "date", "region", "epoch")},
        "model_definitions_json": json.dumps(dict(motor.MODEL_SPEC)),
        "fit_parameters_json": json.dumps(fit_parameters),
        "min_firing_rate_hz": 0.5,
    }
    nested.attrs = attrs
    full.attrs = attrs

    def build(**kwargs):
        value = kwargs["spatial_bin_size_cm"]
        return {
            **kwargs,
            "trajectory_n_splines": max(4, int(np.ceil(3.0 / value))),
            "generalized_place_n_splines": max(
                4,
                int(np.ceil(kwargs["generalized_place_length_cm"] / value)),
            ),
        }

    monkeypatch.setattr(
        motor,
        "_motor_module",
        lambda: SimpleNamespace(build_position_basis_config_from_lengths=build),
    )
    motor._validate_legacy_dataset_pair(
        nested,
        full,
        metadata=metadata,
        parameters=parameters,
        graph_inputs_by_configuration=_graphs(),
    )
    with pytest.raises(ValueError, match="geometry does not match"):
        motor._validate_legacy_dataset_pair(
            nested,
            full,
            metadata=metadata,
            parameters=parameters,
            graph_inputs_by_configuration=_graphs(full_length_scale=2.0),
        )
