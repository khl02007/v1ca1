from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass import table_specs
from v1ca1.spyglass.selection import provenance_sha256, unit_identity_sha256
from v1ca1.spyglass.spikes import resolve_sorted_spikes_group_provenance
import v1ca1.spyglass.tables as tables_module
from v1ca1.spyglass.tables import (
    SOURCE_TABLE_KEYS,
    _analysis_region,
    _attach_registered_unit_identity,
    _construct_tables,
    _ripple_cross_region_xcorr_selection_row,
    _cv_pca_selection_row,
    _dark_light_glm_selection_row,
    _dpp_encoding_selection_row,
    _dpp_tuning_curve_selection_row,
    _filter_registered_table,
    _intervals_to_frame,
    _legacy_dpp_unit_identity_resolver,
    _legacy_dark_light_unit_identity_resolver,
    _legacy_ripple_glm_unit_identity_resolver,
    _legacy_ripple_cross_region_xcorr_identity_resolver,
    _legacy_swap_glm_unit_identity_resolver,
    _legacy_swap_tuning_curve_comparison_unit_identity_resolver,
    _load_cv_pca_result,
    _load_dpp_encoding_result,
    _load_dpp_tuning_curve_result,
    _load_path_progression_decoding_result,
    _load_path_specific_place_decoding_result,
    _load_path_specific_place_tuning_curve_result,
    _load_ripple_modulation_result,
    _load_ripple_cross_region_xcorr_result,
    _load_swap_tuning_curve_comparison_result,
    _load_path_specific_place_tuning_similarity_result,
    _load_tuning_similarity_inputs,
    _load_path_specific_place_stability_result,
    _make_dpp_encoding_row,
    _make_dpp_tuning_curve_row,
    _make_movement_firing_rate_row,
    _make_path_progression_decoding_row,
    _make_path_specific_place_decoding_row,
    _make_path_specific_place_tuning_curve_row,
    _make_ripple_modulation_row,
    _make_ripple_glm_row,
    _make_ripple_cross_region_xcorr_row,
    _make_cv_pca_row,
    _make_swap_glm_row,
    _make_swap_tuning_curve_comparison_row,
    _motor_encoding_selection_row,
    _movement_firing_rate_selection_row,
    _path_progression_decoding_selection_row,
    _path_specific_place_decoding_selection_row,
    _path_specific_place_tuning_curve_selection_row,
    _ripple_modulation_selection_row,
    _ripple_glm_selection_row,
    _register_existing_ripple_glm_row,
    _register_existing_ripple_cross_region_xcorr_row,
    _register_existing_cv_pca_row,
    _registered_nwb_source_identity,
    _register_existing_dpp_encoding_row,
    _register_existing_dark_light_glm_row,
    _register_existing_swap_glm_row,
    _register_existing_swap_tuning_curve_comparison_row,
    _stability_selection_row,
    _swap_glm_selection_row,
    _swap_tuning_curve_comparison_selection_row,
    _tuning_similarity_selection_row,
    _validate_analysis_schema_prefix,
    _validate_dpp_encoding_artifact_link,
    _validate_dark_light_glm_parameter_row,
    _validate_legacy_dpp_encoding_source_path,
    _validate_legacy_tuning_curve_inputs,
    _validate_legacy_stability_schema,
    _validate_path_progression_decoding_artifact_link,
    _validate_path_specific_place_decoding_artifact_link,
    _validate_ripple_provenance,
    _write_dpp_tuning_curve_nwb,
    _write_path_progression_decoding_nwb,
    _write_path_specific_place_decoding_nwb,
    _validate_ripple_glm_parameter_row,
    _validate_ripple_cross_region_xcorr_parameter_row,
    _validate_cv_pca_artifact_link,
    _validate_cv_pca_parameter_row,
    _validate_swap_glm_artifact_link,
    _validate_swap_glm_parameter_row,
    _validate_swap_tuning_curve_comparison_artifact_link,
    _validate_swap_tuning_curve_comparison_parameter_row,
    _validate_swap_tuning_curve_comparison_upstream_link,
    _write_path_specific_place_tuning_similarity_nwb,
    _write_path_specific_place_tuning_curve_nwb,
    _write_path_specific_place_stability_nwb,
    _write_dpp_encoding_nwb,
    _write_ripple_modulation_nwb,
    _write_ripple_cross_region_xcorr_nwb,
)
from v1ca1.spyglass.spikes import _sorting_output_sessions


class _FakeTable:
    @classmethod
    def insert1(cls, row, **kwargs):
        cls._insert_calls = [
            *cls.__dict__.get("_insert_calls", []),
            (dict(row), kwargs),
        ]

    @classmethod
    def insert(cls, rows, **kwargs):
        cls._insert_many_calls = [
            *cls.__dict__.get("_insert_many_calls", []),
            ([dict(row) for row in rows], kwargs),
        ]


class _FakeManual(_FakeTable):
    pass


class _FakeComputed(_FakeTable):
    pass


class _FakePart(_FakeTable):
    pass


class _FakeSpyglassMixin:
    pass


class _FakeSpyglassAnalysis:
    def _register_table(self):
        type(self)._registry_calls = type(self).__dict__.get("_registry_calls", 0) + 1


class _FakeRelation:
    def __init__(self, row=None):
        self.row = dict(row or {})

    def __and__(self, key):
        return self

    def fetch1(self, *names):
        if names:
            return tuple(self.row[name] for name in names)
        return dict(self.row)


class _RecordingRelation(_FakeRelation):
    def __init__(self, row=None):
        super().__init__(row)
        self.keys = []

    def __and__(self, key):
        self.keys.append(dict(key))
        return _FakeRelation(self.row)


class _FakeKeyedRelation:
    def __init__(self, key_name, rows):
        self.key_name = key_name
        self.rows = {key: dict(row) for key, row in rows.items()}

    def __and__(self, key):
        return _FakeRelation(self.rows[key[self.key_name]])


class _FakeRowsRelation:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def __and__(self, key):
        matches = [
            row
            for row in self.rows
            if all(row.get(name) == value for name, value in key.items())
        ]
        if len(matches) != 1:
            raise LookupError(
                f"Expected one row for {dict(key)!r}; found {len(matches)}."
            )
        return _FakeRelation(matches[0])


class _TestAnalysisFileBuilder:
    """Small real-HDF5 stand-in for Spyglass's analysis-file builder."""

    def __init__(self, path: Path):
        self.path = path
        self.analysis_file_name = path.name
        self.registered = False
        self._open_io = None
        self._open_nwb = None

    def __enter__(self):
        from pynwb import NWBHDF5IO, NWBFile

        nwbfile = NWBFile(
            session_description="MovementFiringRate table test",
            identifier="movement-table-test",
            session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        with NWBHDF5IO(str(self.path), mode="w") as io:
            io.write(nwbfile)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._open_io is not None:
            self._open_io.close()
            self._open_io = None
            self._open_nwb = None
        if exc_type is None:
            self.registered = True
        return False

    def get_path(self) -> str:
        return str(self.path)

    @property
    def open_nwb(self):
        from pynwb import NWBHDF5IO

        if self._open_io is None:
            self._open_io = NWBHDF5IO(
                str(self.path),
                mode="a",
                load_namespaces=True,
            )
            self._open_nwb = self._open_io.read()
        return self._open_io, self._open_nwb

    def add_nwb_object(self, nwb_object) -> str:
        nwbfile = self.open_nwb[1]
        nwbfile.add_scratch(nwb_object)
        return str(nwb_object.object_id)

    def close_and_write(self) -> None:
        if self._open_io is None:
            return
        self._open_io.write(self._open_nwb)
        self._open_io.close()
        self._open_io = None
        self._open_nwb = None


class _TestAnalysisNwbfile:
    """Capture one analysis-file builder without touching DataJoint."""

    def __init__(self, path: Path):
        self.builder = _TestAnalysisFileBuilder(path)

    def build(self, nwb_file_name: str) -> _TestAnalysisFileBuilder:
        assert nwb_file_name == "L1420240102_.nwb"
        return self.builder


class _FakeGroupUnits:
    def __init__(self, merge_ids):
        self.merge_ids = list(merge_ids)

    def __and__(self, key):
        return self

    def fetch(self, name):
        assert name == "spikesorting_merge_id"
        return list(self.merge_ids)


class _FakeSortedSpikesGroup:
    def __init__(self, merge_ids):
        self.Units = _FakeGroupUnits(merge_ids)


class _FakeSchema:
    def __init__(self, *, context=None):
        self.context = context
        self.connection = object()
        self.activations = []
        self.tables = []

    def activate(self, name, **kwargs):
        self.activations.append((name, kwargs))
        if kwargs.get("connection") is not None:
            self.connection = kwargs["connection"]

    def __call__(self, table):
        table.connection = self.connection
        table.database = self.activations[-1][0]
        self.tables.append(table)
        return table


def _fake_bundle(*, runtime_hooks=None):
    schemas = []

    def schema_factory(*, context=None):
        schema = _FakeSchema(context=context)
        schemas.append(schema)
        return schema

    session = _FakeRelation(
        {
            "subject_id": "L14",
            "session_start_time": datetime(2024, 1, 2, 12, 30),
        }
    )
    unit_selection_params = _FakeRelation(
        {
            "unit_filter_params_name": "all_units",
            "include_labels": [],
            "exclude_labels": [],
        }
    )
    fake_dj = SimpleNamespace(
        Manual=_FakeManual,
        Computed=_FakeComputed,
        Part=_FakePart,
    )
    bundle = _construct_tables(
        dj_module=fake_dj,
        session_table=session,
        nwbfile_table=SimpleNamespace(get_abs_path=lambda name: name),
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-a"]),
        unit_selection_params=unit_selection_params,
        spike_sorting_output=SimpleNamespace(proj=lambda **kwargs: None),
        spyglass_mixin=_FakeSpyglassMixin,
        spyglass_analysis=_FakeSpyglassAnalysis,
        schema_factory=schema_factory,
        schema_name="kyuv1ca1",
        analysis_nwbfile_schema_name="kyuv1ca1_nwbfile",
        connection=None,
        create_schema=True,
        create_tables=True,
        runtime_hooks=runtime_hooks,
        artifact_root=Path("/analysis"),
    )
    return bundle, schemas, unit_selection_params


def _sorting_provenance(
    merge_ids=("merge-a", "merge-b"),
    *,
    include_labels=("accepted",),
    exclude_labels=("noise", "mua"),
):
    key = {
        "nwb_file_name": "L1420240102_.nwb",
        "unit_filter_params_name": "curated_units",
        "sorted_spikes_group_name": "all shanks",
    }
    return resolve_sorted_spikes_group_provenance(
        _FakeSortedSpikesGroup(merge_ids),
        _FakeRelation(
            {
                "include_labels": list(include_labels),
                "exclude_labels": list(exclude_labels),
            }
        ),
        key,
    )


def _ripple_selection_key() -> dict[str, Any]:
    return {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "ripple_modulation_param_name": "default",
        "region_sorted_spikes_group_id": uuid.UUID(
            "61111111-1111-5111-8111-111111111111"
        ),
    }


def _movement_selection_key() -> dict[str, Any]:
    return {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "position_series_name": "head_position",
        "movement_param_name": "default",
        "region_sorted_spikes_group_id": uuid.UUID(
            "61111111-1111-5111-8111-111111111111"
        ),
    }


def _valid_ripple_modulation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one canonical RippleModulation table pair for NWB tests."""
    from v1ca1.spyglass import ripple_modulation

    common = {
        "animal_name": "L14",
        "date": "20240102",
        "epoch": "02_r1",
        "region": "ca1",
        "unit_id": "11",
        "n_ripples": 3,
        "bin_size_s": 0.1,
        "time_before_s": 0.2,
        "time_after_s": 0.2,
        "group_unit_id": 10,
        "spikesorting_merge_id": "merge-a",
        "stable_unit_id": "merge-a:11",
    }
    summary = pd.DataFrame.from_records(
        [
            {
                **common,
                "baseline_mean_hz": 1.0,
                "baseline_std_hz": 0.5,
                "response_mean_hz": 1.5,
                "ripple_modulation_index": 0.2,
                "response_zscore": 1.0,
                "invalid_reason": None,
            }
        ],
        columns=ripple_modulation.SUMMARY_TABLE_COLUMNS,
    )
    peri = pd.DataFrame.from_records(
        [
            {
                **common,
                "time_s": time_s,
                "mean_rate_hz": mean_rate_hz,
            }
            for time_s, mean_rate_hz in zip(
                (-0.15, -0.05, 0.05, 0.15),
                (1.0, 1.0, 1.5, 1.5),
                strict=True,
            )
        ],
        columns=ripple_modulation.PERI_RIPPLE_FIRING_RATE_TABLE_COLUMNS,
    )
    return ripple_modulation.validate_ripple_modulation_tables(summary, peri)


def _valid_epoch_motor_behavior_result() -> dict[str, Any]:
    """Return one canonical terminal motor result for NWB lifecycle tests."""
    from v1ca1.helper.session import TRAJECTORY_TYPES
    from v1ca1.spyglass import epoch_motor_behavior

    result_id = uuid.UUID("73333333-3333-5333-8333-333333333333")
    variables = tuple(epoch_motor_behavior._motor_module().MOTOR_VARIABLES)
    distribution = pd.DataFrame.from_records(
        [
            {
                "epoch": "02_r1",
                "variable": variable,
                "sample_count": 0,
                "movement_duration_s": 0.0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "circular_mean_deg": np.nan,
                "resultant_length": np.nan,
            }
            for variable in variables
        ],
        columns=epoch_motor_behavior.DISTRIBUTION_COLUMNS,
    )
    progression = pd.DataFrame(
        columns=epoch_motor_behavior.PROGRESSION_COLUMNS
    )
    trajectory_qc = pd.DataFrame.from_records(
        [
            {
                "epoch_motor_behavior_id": str(result_id),
                "animal_name": "L14",
                "date": "20240102",
                "epoch": "02_r1",
                "trajectory_type": trajectory_type,
                "trajectory_interval_count": 0,
                "trajectory_interval_duration_s": 0.0,
                "movement_supported_duration_s": 0.0,
                "movement_supported_sample_count": 0,
                "finite_progression_sample_count": 0,
                "occupied_progression_bin_count": 0,
                "graph_length_cm": 100.0,
                "trajectory_status": "no_valid_position",
            }
            for trajectory_type in TRAJECTORY_TYPES
        ],
        columns=epoch_motor_behavior.TRAJECTORY_QC_COLUMNS,
    )
    parameters = epoch_motor_behavior._parameter_snapshot(
        parameter_name="manuscript_4cm",
        parameters=epoch_motor_behavior.validate_epoch_motor_behavior_parameters(),
        parameter_sha256=None,
        output_rule_sha256=None,
    )
    movement = epoch_motor_behavior.validate_movement_parameter_snapshot()
    return epoch_motor_behavior.validate_epoch_motor_behavior_result(
        {
            "metadata": {
                "epoch_motor_behavior_id": str(result_id),
                "animal_name": "L14",
                "date": "20240102",
                "epoch": "02_r1",
                "epoch_type": "run",
                "primary_position_source": "head_position",
                "primary_position_role": "translation_anchor",
                "orientation_reference_position_source": "body_position",
                "orientation_reference_position_role": "orientation_reference",
                "position_offset_samples": 0,
            },
            "parameters": parameters,
            "movement_parameters": movement,
            "distribution_summary": distribution,
            "progression_summary": progression,
            "trajectory_qc": trajectory_qc,
            "n_position_samples_input": 20,
            "n_finite_position_samples": 0,
            "n_dropped_nonfinite_samples": 20,
            "n_movement_samples": 0,
            "movement_duration_s": 0.0,
            "n_supported_trajectories": 0,
            "sampling_rate_hz": np.nan,
            "median_sample_interval_s": np.nan,
            "maximum_sample_gap_s": np.nan,
            "analysis_status": "no_valid_position",
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
    )


def _valid_tuning_similarity_table() -> pd.DataFrame:
    """Return one canonical four-comparison similarity table."""
    from v1ca1.spyglass import tuning_similarity

    rows = []
    for spec in tuning_similarity.DIRECT_COMPARISON_SPECS:
        rows.append(
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "stable_unit_id": "merge-a:11",
                "group_unit_id": "curve-unit-a",
                "animal_name": "L14",
                "date": "20240102",
                "region": "ca1",
                "epoch": "02_r1",
                "similarity_metric": "correlation",
                "comparison_family": spec["comparison_family"],
                "comparison_label": spec["comparison_label"],
                "side": spec["side"],
                "trajectory_a": spec["trajectory_a"],
                "trajectory_b": spec["trajectory_b"],
                "flip_trajectory_b": spec["flip_trajectory_b"],
                "movement_firing_rate_hz": 1.5,
                "similarity": 0.5,
                "n_trajectory_a_finite_bins": 5,
                "n_trajectory_b_finite_bins": 5,
                "n_paired_finite_bins": 5,
                "similarity_status": "valid",
            }
        )
    table = pd.DataFrame.from_records(
        rows,
        columns=tuning_similarity.TABLE_COLUMNS,
    )
    return tuning_similarity.validate_tuning_similarity_table(table)


def _valid_path_specific_place_curve() -> tuple[Any, dict[str, Any]]:
    """Return one canonical curve with its complete live selection identity."""
    import xarray as xr

    from v1ca1.spyglass import path_specific_place

    curve_id = uuid.UUID("73333333-3333-5333-8333-333333333333")
    movement_id = uuid.UUID("74444444-4444-5444-8444-444444444444")
    legacy = xr.DataArray(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        dims=("unit", "linpos"),
        coords={"unit": [11, 22], "linpos": [2.5, 7.5]},
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240102",
            "region": "ca1",
            "epoch": "02_r1",
            "trajectory_type": "center_to_left",
            "model_name": "place",
            "bin_edges": "[[0.0,5.0,10.0]]",
        },
    )
    curve = path_specific_place.normalize_legacy_all_trial_tuning_curve(
        legacy,
        unit_identity_resolver={
            11: {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "group_unit_id": 0,
                "sorting_unit_id": 11,
            },
            22: {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "22",
                "group_unit_id": 1,
                "sorting_unit_id": 22,
            },
        },
        animal_name="L14",
        date="20240102",
        region="ca1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        graph_length_cm=10.0,
        n_trials=3,
        support_duration_s=2.4,
        n_feature_samples=24,
        n_valid_position_samples=20,
        bin_count=2,
    )
    selected_units_sha256 = unit_identity_sha256(
        [
            {"spikesorting_merge_id": "merge-a", "unit_id": "11"},
            {"spikesorting_merge_id": "merge-a", "unit_id": "22"},
        ]
    )
    selection = {
        "path_specific_place_tuning_curve_id": curve_id,
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "trajectory_type": "center_to_left",
        "configuration_name": "center_to_left",
        "movement_firing_rate_id": movement_id,
        "tuning_curve_param_name": "two_bins",
        "trial_subset": "all",
        "tuning_curve_parameters_sha256": "a" * 64,
    }
    curve.attrs.update(
        tables_module._tuning_curve_artifact_attributes(
            selection,
            selected_units_sha256=selected_units_sha256,
        )
    )
    return curve, selection


def _valid_dpp_curve() -> tuple[Any, dict[str, Any]]:
    """Return one canonical DPP curve with its live selection identity."""
    import xarray as xr

    from v1ca1.spyglass import dpp

    curve_id = uuid.UUID("75555555-5555-5555-8555-555555555555")
    movement_id = uuid.UUID("76666666-6666-5666-8666-666666666666")
    legacy = xr.DataArray(
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        dims=("unit", "tp"),
        coords={"unit": [11, 22], "tp": [0.25, 0.75]},
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240102",
            "region": "ca1",
            "epoch": "02_r1",
            "model_name": "task_progression",
            "turn_type": "left",
            "bin_edges": "[[0.0,0.5,1.0]]",
        },
    )
    curve = dpp.normalize_legacy_all_trial_dpp_tuning_curve(
        legacy,
        unit_identity_resolver={
            11: {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "group_unit_id": 0,
                "sorting_unit_id": 11,
            },
            22: {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "22",
                "group_unit_id": 1,
                "sorting_unit_id": 22,
            },
        },
        animal_name="L14",
        date="20240102",
        region="ca1",
        epoch="02_r1",
        turn_type="left",
        common_graph_length_cm=10.0,
        graph_length_cm_by_trajectory={
            "center_to_left": 10.0,
            "right_to_center": 10.0,
        },
        n_trials_by_trajectory={"center_to_left": 2, "right_to_center": 1},
        support_duration_s_by_trajectory={
            "center_to_left": 1.6,
            "right_to_center": 0.8,
        },
        n_feature_samples_by_trajectory={
            "center_to_left": 16,
            "right_to_center": 8,
        },
        n_valid_position_samples_by_trajectory={
            "center_to_left": 15,
            "right_to_center": 8,
        },
        bin_count=2,
    )
    selected_units_sha256 = unit_identity_sha256(
        [
            {"spikesorting_merge_id": "merge-a", "unit_id": "11"},
            {"spikesorting_merge_id": "merge-a", "unit_id": "22"},
        ]
    )
    selection = {
        "dpp_tuning_curve_id": curve_id,
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "outbound_trajectory_type": "center_to_left",
        "inbound_trajectory_type": "right_to_center",
        "outbound_configuration_name": "center_to_left",
        "inbound_configuration_name": "right_to_center",
        "movement_firing_rate_id": movement_id,
        "tuning_curve_param_name": "two_bins",
        "turn_type": "left",
        "trial_subset": "all",
        "tuning_curve_parameters_sha256": "a" * 64,
    }
    curve.attrs.update(
        tables_module._dpp_tuning_curve_artifact_attributes(
            selection,
            selected_units_sha256=selected_units_sha256,
        )
    )
    return curve, selection


def _valid_dpp_encoding_table() -> pd.DataFrame:
    """Return one canonical nonempty DPPEncoding table for NWB tests."""
    from v1ca1.spyglass import dpp_encoding

    comparison_id = uuid.UUID("72222222-2222-5222-8222-222222222222")
    metrics = dpp_encoding._unit_metric_row(
        store={
            "heldout_spike_count": 4,
            "null_log_likelihood_nats": -12.0,
            "zero_training_spikes": False,
            "model_log_likelihood_nats": {
                "path_specific_place": -10.0,
                "absolute_place": -11.0,
                "dpp": -8.0,
                "distance_to_reward": -9.0,
            },
            "model_failed": {
                model_name: False
                for model_name in dpp_encoding.MODEL_NAMES
            },
        }
    )
    row = {
        "spikesorting_merge_id": "merge-a",
        "unit_id": "11",
        "stable_unit_id": "merge-a:11",
        "group_unit_id": "10",
        "dpp_encoding_id": str(comparison_id),
        "animal_name": "L14",
        "date": "20240102",
        "region": "ca1",
        "epoch": "02_r1",
        "n_folds": 5,
        "evaluation_bin_size_s": 0.05,
        "spatial_bin_size_cm": 4.0,
        "gaussian_smoothing_sigma_bins": 1.0,
        "random_seed": 47,
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": 0.5,
        "movement_firing_rate_hz": 0.5,
        "center_to_left_stability_correlation": 0.5,
        "center_to_right_stability_correlation": 0.1,
        "left_to_center_stability_correlation": 0.1,
        "right_to_center_stability_correlation": 0.1,
        **metrics,
    }
    table = pd.DataFrame.from_records([row]).loc[
        :, list(dpp_encoding.TABLE_COLUMNS)
    ]
    return dpp_encoding.validate_dpp_encoding_table(table)


def _valid_stability_table() -> pd.DataFrame:
    """Return one canonical nonempty stability table for NWB-loader tests."""
    from v1ca1.spyglass.stability import (
        STABILITY_TABLE_COLUMNS,
        empty_stability_table,
        validate_stability_table,
    )

    empty = empty_stability_table()
    row = {}
    for column in STABILITY_TABLE_COLUMNS:
        if pd.api.types.is_integer_dtype(empty[column].dtype):
            row[column] = 1
        elif pd.api.types.is_float_dtype(empty[column].dtype):
            row[column] = 0.5
        else:
            row[column] = "valid"
    row.update(
        spikesorting_merge_id="merge-a",
        unit_id="11",
        stable_unit_id="merge-a:11",
        group_unit_id=0,
        animal_name="L14",
        date="20240102",
        region="ca1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        segment_stability_shape_overlaps="[0.5, 0.5, 0.5]",
        segment_shape_overlap_statuses='["valid", "valid", "valid"]',
        odd_segment_mean_firing_rates_hz="[0.5, 0.5, 0.5]",
        even_segment_mean_firing_rates_hz="[0.5, 0.5, 0.5]",
        odd_segment_tuning_curve_areas="[0.5, 0.5, 0.5]",
        even_segment_tuning_curve_areas="[0.5, 0.5, 0.5]",
        segment_edges_normalized="[0.0, 0.33, 0.67, 1.0]",
    )
    return validate_stability_table(
        pd.DataFrame.from_records([row], columns=STABILITY_TABLE_COLUMNS)
    )


def _tuning_curve_selection_key(*, trial_subset: str = "odd") -> dict[str, Any]:
    return {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "trajectory_type": "center_to_left",
        "configuration_name": "center_to_left",
        "movement_firing_rate_id": uuid.UUID(
            "33333333-3333-5333-8333-333333333333"
        ),
        "tuning_curve_param_name": "legacy_4cm_unsmoothed",
        "trial_subset": trial_subset,
    }


def _dpp_tuning_curve_selection_key(
    *,
    turn_type: str = "left",
    trial_subset: str = "all",
) -> dict[str, Any]:
    return {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "movement_firing_rate_id": uuid.UUID(
            "33333333-3333-5333-8333-333333333333"
        ),
        "tuning_curve_param_name": "legacy_4cm_unsmoothed",
        "turn_type": turn_type,
        "trial_subset": trial_subset,
    }


def _stability_selection_key() -> dict[str, Any]:
    return {
        "odd_path_specific_place_tuning_curve_id": uuid.UUID(
            "44444444-4444-5444-8444-444444444444"
        ),
        "even_path_specific_place_tuning_curve_id": uuid.UUID(
            "55555555-5555-5555-8555-555555555555"
        ),
    }


def _tuning_similarity_selection_key() -> dict[str, Any]:
    return {
        "center_to_left_tuning_curve_id": uuid.UUID(
            "81111111-1111-5111-8111-111111111111"
        ),
        "center_to_right_tuning_curve_id": uuid.UUID(
            "82222222-2222-5222-8222-222222222222"
        ),
        "left_to_center_tuning_curve_id": uuid.UUID(
            "83333333-3333-5333-8333-333333333333"
        ),
        "right_to_center_tuning_curve_id": uuid.UUID(
            "84444444-4444-5444-8444-444444444444"
        ),
        "tuning_similarity_param_name": "correlation",
    }


_DPP_ENCODING_TRAJECTORIES = (
    "center_to_left",
    "center_to_right",
    "left_to_center",
    "right_to_center",
)


def _dpp_encoding_selection_inputs() -> dict[str, Any]:
    """Return mutable, internally consistent DPP selection inputs."""
    movement_id = uuid.UUID("33333333-3333-5333-8333-333333333333")
    region_group_id = uuid.UUID("61111111-1111-5111-8111-111111111111")
    stability_ids = {
        trajectory_type: uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"v1ca1-test-stability:{trajectory_type}",
        )
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    curve_selections: dict[uuid.UUID, dict[str, Any]] = {}
    stability_selections: dict[uuid.UUID, dict[str, Any]] = {}
    for trajectory_type, stability_id in stability_ids.items():
        curve_ids = {
            subset: uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"v1ca1-test-curve:{trajectory_type}:{subset}",
            )
            for subset in ("odd", "even")
        }
        stability_selections[stability_id] = {
            "path_specific_place_stability_id": stability_id,
            "odd_path_specific_place_tuning_curve_id": curve_ids["odd"],
            "even_path_specific_place_tuning_curve_id": curve_ids["even"],
        }
        for subset, curve_id in curve_ids.items():
            curve_selections[curve_id] = {
                "path_specific_place_tuning_curve_id": curve_id,
                "nwb_file_name": "L1420240102_.nwb",
                "epoch": "02_r1",
                "trajectory_type": trajectory_type,
                "configuration_name": trajectory_type,
                "movement_firing_rate_id": movement_id,
                "tuning_curve_param_name": "legacy_4cm_unsmoothed",
                "tuning_curve_parameters_sha256": provenance_sha256(
                    dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
                ),
                "trial_subset": subset,
            }

    parameters = dict(
        table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS
    )
    return {
        "key": {
            "region_sorted_spikes_group_id": region_group_id,
            "movement_firing_rate_id": movement_id,
            **{
                f"{trajectory_type}_stability_id": stability_id
                for trajectory_type, stability_id in stability_ids.items()
            },
            "dpp_encoding_param_name": parameters[
                "dpp_encoding_param_name"
            ],
        },
        "region_row": {
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": "L1420240102_.nwb",
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region_name": "ca1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
            "n_units": 5,
            "selected_units_sha256": "a" * 64,
        },
        "movement_result": {
            "movement_firing_rate_id": movement_id,
            "n_units": 5,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        },
        "movement_selection": {
            "movement_firing_rate_id": movement_id,
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": "L1420240102_.nwb",
            "epoch": "02_r1",
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region": "ca1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
        },
        "epoch_row": {
            "nwb_file_name": "L1420240102_.nwb",
            "epoch": "02_r1",
            "epoch_type": "run",
        },
        "trajectory_rows": [
            {
                "nwb_file_name": "L1420240102_.nwb",
                "epoch": "02_r1",
                "trajectory_type": trajectory_type,
                "interval_count": 5,
            }
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        ],
        "graph_rows": [
            {
                "nwb_file_name": "L1420240102_.nwb",
                "configuration_name": configuration_name,
                "coordinate_unit": "cm",
            }
            for configuration_name in (*_DPP_ENCODING_TRAJECTORIES, "full_w")
        ],
        "stability_results": {
            stability_id: {
                "path_specific_place_stability_id": stability_id,
                "n_units": 5,
                "analysis_status": "valid",
                "selected_units_sha256": "a" * 64,
            }
            for stability_id in stability_ids.values()
        },
        "stability_selections": stability_selections,
        "curve_selections": curve_selections,
        "parameters": parameters,
    }


def _build_dpp_encoding_selection(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build one DPP selection row from mutable fake upstream tables."""
    return _dpp_encoding_selection_row(
        key=inputs["key"],
        region_sorted_spikes_group_table=_FakeRelation(inputs["region_row"]),
        movement_firing_rate_table=_FakeRelation(inputs["movement_result"]),
        movement_firing_rate_selection_table=_FakeRelation(
            inputs["movement_selection"]
        ),
        epoch_intervals_table=_FakeRelation(inputs["epoch_row"]),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        stability_table=_FakeKeyedRelation(
            "path_specific_place_stability_id",
            inputs["stability_results"],
        ),
        stability_selection_table=_FakeKeyedRelation(
            "path_specific_place_stability_id",
            inputs["stability_selections"],
        ),
        tuning_curve_selection_table=_FakeKeyedRelation(
            "path_specific_place_tuning_curve_id",
            inputs["curve_selections"],
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
    )


def _dpp_encoding_runtime_inputs(
    comparison_id: uuid.UUID,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return selected context, imported spikes, and NWB-derived sentinels."""
    parameters = dict(
        table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS
    )
    unit_ids = [
        {"spikesorting_merge_id": "merge-a", "unit_id": 10},
        {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    ]
    loaded_spikes = {
        "ts_group": object(),
        "unit_ids": unit_ids,
        "unit_metadata": [
            {**unit_id, "sorting_unit_id": sorting_unit_id}
            for unit_id, sorting_unit_id in zip(
                unit_ids,
                (101, 102),
                strict=True,
            )
        ],
        "member_provenance": [
            {
                "spikesorting_merge_id": "merge-a",
                "merge_parent": "ImportedSpikeSorting",
                "n_selected_units": 2,
            }
        ],
    }
    context = {
        "animal_name": "L14",
        "date": "20240102",
        "region": "ca1",
        "epoch": "02_r1",
        "parameters": parameters,
        "region_row": {
            "region_sorted_spikes_group_id": uuid.UUID(
                "61111111-1111-5111-8111-111111111111"
            ),
            "n_units": 2,
        },
        "movement": {
            "movement_intervals": object(),
            "table": object(),
        },
        "movement_selection": {
            "nwb_file_name": "L1420240102_.nwb",
        },
        "stability_tables": {
            trajectory_type: object()
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        "selection": {
            "dpp_encoding_id": comparison_id,
        },
    }
    nwb_inputs = {
        "position": object(),
        "trajectory_intervals": {
            trajectory_type: object()
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        "graph_inputs": {
            configuration_name: object()
            for configuration_name in (*_DPP_ENCODING_TRAJECTORIES, "full_w")
        },
    }
    return context, loaded_spikes, nwb_inputs


def _path_progression_decoding_selection_inputs() -> dict[str, Any]:
    """Return mutable target/cohort inputs for one decoding selection."""
    nwb_file_name = "L1420240102_.nwb"
    target_epoch = "02_r1"
    cohort_epoch = "08_r4"
    target_movement_id = uuid.UUID(
        "71111111-1111-5111-8111-111111111111"
    )
    cohort_movement_id = uuid.UUID(
        "72222222-2222-5222-8222-222222222222"
    )
    region_group_id = uuid.UUID(
        "73333333-3333-5333-8333-333333333333"
    )
    source_specs = (
        ("target", "", target_epoch, target_movement_id),
        ("cohort", "cohort_", cohort_epoch, cohort_movement_id),
    )
    stability_ids: dict[str, uuid.UUID] = {}
    stability_results: dict[uuid.UUID, dict[str, Any]] = {}
    stability_selections: dict[uuid.UUID, dict[str, Any]] = {}
    curve_selections: dict[uuid.UUID, dict[str, Any]] = {}
    for source_name, prefix, epoch, movement_id in source_specs:
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES:
            stability_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                (
                    "v1ca1-test-decoding-stability:"
                    f"{source_name}:{trajectory_type}"
                ),
            )
            stability_ids[
                f"{prefix}{trajectory_type}_stability_id"
            ] = stability_id
            curve_ids = {
                subset: uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        "v1ca1-test-decoding-curve:"
                        f"{source_name}:{trajectory_type}:{subset}"
                    ),
                )
                for subset in ("odd", "even")
            }
            stability_results[stability_id] = {
                "path_specific_place_stability_id": stability_id,
                "n_units": 5,
                "analysis_status": "valid",
                "selected_units_sha256": "a" * 64,
            }
            stability_selections[stability_id] = {
                "path_specific_place_stability_id": stability_id,
                "odd_path_specific_place_tuning_curve_id": curve_ids["odd"],
                "even_path_specific_place_tuning_curve_id": curve_ids["even"],
            }
            for subset, curve_id in curve_ids.items():
                curve_selections[curve_id] = {
                    "path_specific_place_tuning_curve_id": curve_id,
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch,
                    "trajectory_type": trajectory_type,
                    "configuration_name": trajectory_type,
                    "movement_firing_rate_id": movement_id,
                    "tuning_curve_param_name": "legacy_4cm_unsmoothed",
                    "tuning_curve_parameters_sha256": provenance_sha256(
                        dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
                    ),
                    "trial_subset": subset,
                }

    movement_results = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "n_units": 5,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }
        for movement_id in (target_movement_id, cohort_movement_id)
    }
    movement_selections = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region": "ca1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
            "movement_param_name": "default",
            "movement_parameters_sha256": "d" * 64,
        }
        for movement_id, epoch in (
            (target_movement_id, target_epoch),
            (cohort_movement_id, cohort_epoch),
        )
    }
    epoch_rows = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "epoch_type": "run",
        }
        for movement_id, epoch in (
            (target_movement_id, target_epoch),
            (cohort_movement_id, cohort_epoch),
        )
    }
    position_rows = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "spatial_unit": "cm",
            "position_role": "head",
            "analysis_start_offset_samples": 10,
        }
        for movement_id, epoch in (
            (target_movement_id, target_epoch),
            (cohort_movement_id, cohort_epoch),
        )
    }
    parameters = dict(
        table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS
    )
    key = {
        "nwb_file_name": nwb_file_name,
        "epoch": target_epoch,
        "cohort_epoch": cohort_epoch,
        "region_sorted_spikes_group_id": region_group_id,
        "movement_firing_rate_id": target_movement_id,
        "cohort_movement_firing_rate_id": cohort_movement_id,
        **{
            f"{trajectory_type}_trajectory_type": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        **{
            f"{trajectory_type}_configuration_name": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        **stability_ids,
        "path_progression_decoding_param_name": parameters[
            "path_progression_decoding_param_name"
        ],
    }
    return {
        "key": key,
        "region_row": {
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": nwb_file_name,
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region_name": "ca1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
            "n_units": 5,
            "selected_units_sha256": "a" * 64,
        },
        "movement_results": movement_results,
        "movement_selections": movement_selections,
        "epoch_rows": epoch_rows,
        "position_rows": position_rows,
        "trajectory_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "epoch": target_epoch,
                "trajectory_type": trajectory_type,
                "interval_count": 5,
            }
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        ],
        "graph_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
                "coordinate_unit": "cm",
            }
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        ],
        "stability_results": stability_results,
        "stability_selections": stability_selections,
        "curve_selections": curve_selections,
        "parameters": parameters,
    }


def _build_path_progression_decoding_selection(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build one decoding selection from fake target/cohort relations."""
    return _path_progression_decoding_selection_row(
        key=inputs["key"],
        region_sorted_spikes_group_table=_FakeRelation(inputs["region_row"]),
        movement_firing_rate_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_results"],
        ),
        movement_firing_rate_selection_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_selections"],
        ),
        position_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["position_rows"],
        ),
        epoch_intervals_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["epoch_rows"],
        ),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        stability_table=_FakeKeyedRelation(
            "path_specific_place_stability_id",
            inputs["stability_results"],
        ),
        stability_selection_table=_FakeKeyedRelation(
            "path_specific_place_stability_id",
            inputs["stability_selections"],
        ),
        tuning_curve_selection_table=_FakeKeyedRelation(
            "path_specific_place_tuning_curve_id",
            inputs["curve_selections"],
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
    )


def _build_path_specific_place_decoding_selection() -> dict[str, Any]:
    """Build one within-epoch place-decoder selection from fake rows."""
    inputs = _path_progression_decoding_selection_inputs()
    target_id = inputs["key"]["movement_firing_rate_id"]
    parameters = dict(
        table_specs.MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS
    )
    key = {
        "nwb_file_name": inputs["key"]["nwb_file_name"],
        "epoch": inputs["key"]["epoch"],
        "region_sorted_spikes_group_id": inputs["key"][
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": target_id,
        **{
            f"{trajectory_type}_trajectory_type": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        **{
            f"{trajectory_type}_configuration_name": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        "path_specific_place_decoding_param_name": parameters[
            "path_specific_place_decoding_param_name"
        ],
    }
    return _path_specific_place_decoding_selection_row(
        key=key,
        region_sorted_spikes_group_table=_FakeRelation(inputs["region_row"]),
        movement_firing_rate_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            {target_id: inputs["movement_results"][target_id]},
        ),
        movement_firing_rate_selection_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            {target_id: inputs["movement_selections"][target_id]},
        ),
        position_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            {target_id: inputs["position_rows"][target_id]},
        ),
        epoch_intervals_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            {target_id: inputs["epoch_rows"][target_id]},
        ),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        parameters_table=_FakeRelation(parameters),
    )


def _path_specific_place_decoding_result() -> dict[str, Any]:
    """Return one compact canonical result for NWB lifecycle tests."""
    import pynapple as nap

    from v1ca1.spyglass import path_specific_decoding as decoding

    result_id = str(uuid.UUID("12345678-1234-5678-1234-567812345678"))
    metadata = {
        "path_specific_place_decoding_id": result_id,
        "animal_name": "L14",
        "date": "20240102",
        "region": "v1",
        "epoch": "02_r1",
    }
    effective = decoding.validate_path_specific_decoding_parameters(
        n_folds=2
    )
    parameters = {
        "parameter_name": "test",
        "parameter_sha256": provenance_sha256(
            {
                "path_specific_place_decoding_param_name": "test",
                **effective,
            }
        ),
        "output_rule_sha256": provenance_sha256(dict(decoding.OUTPUT_RULE)),
        **effective,
    }
    selected_units = pd.DataFrame.from_records(
        [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "stable_unit_id": "merge-a:11",
                "group_unit_id": "0",
                "selection_index": 0,
            }
        ],
        columns=decoding.SELECTED_UNIT_COLUMNS,
    )
    selected_digest = unit_identity_sha256(
        [{"spikesorting_merge_id": "merge-a", "unit_id": "11"}]
    )
    fold_qc = pd.DataFrame.from_records(
        [
            {
                **metadata,
                "fold": fold,
                "n_train_laps": 1,
                "n_test_laps": 1,
                "train_duration_s": 1.0,
                "test_duration_s": 1.0,
                "n_decoded_samples": 2,
                "qc_status": "valid",
                "qc_message": "",
            }
            for fold in range(2)
        ],
        columns=decoding.FOLD_QC_COLUMNS,
    )
    summary = pd.DataFrame.from_records(
        [
            {
                **metadata,
                "model": "path_specific_place",
                "coordinate_unit": "cm",
                "n_units": 1,
                "n_folds_expected": 2,
                "n_folds_valid": 2,
                "mae": 0.25,
                "rmse": 0.25,
                "mean_signed_error": 0.25,
                "median_abs_error": 0.25,
                "n_samples": 4,
                "analysis_status": "valid",
            }
        ],
        columns=decoding.SUMMARY_COLUMNS,
    )
    binned_error = pd.DataFrame.from_records(
        [
            {
                **metadata,
                "coordinate_unit": "cm",
                "bin_left": 0.0,
                "bin_right": 4.0,
                "bin_center": 2.0,
                "n": 4,
                "center": 0.25,
                "yerr_low": 0.0,
                "yerr_high": 0.0,
            }
        ],
        columns=decoding.BINNED_ERROR_COLUMNS,
    )
    support = nap.IntervalSet(
        start=np.asarray([0.0, 2.0]),
        end=np.asarray([1.0, 3.0]),
        time_units="s",
    )
    true = nap.Tsd(
        t=np.asarray([0.1, 0.2, 0.3, 2.1, 2.2, 2.3]),
        d=np.asarray([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]),
        time_support=support,
        time_units="s",
    )
    decoded = nap.Tsd(
        t=np.asarray([0.2, 0.3, 2.2, 2.3]),
        d=np.asarray([2.25, 3.25, 2.25, 3.25]),
        time_support=support,
        time_units="s",
    )
    return {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": selected_units,
        "fold_qc": fold_qc,
        "summary": summary,
        "binned_error": binned_error,
        "true": true,
        "decoded": decoded,
        "path_length_cm": 10.0,
        "n_units": 1,
        "n_folds_expected": 2,
        "n_folds_valid": 2,
        "n_decoded_samples": 4,
        "selected_units_sha256": selected_digest,
        "analysis_status": "valid",
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def _motor_encoding_selection_inputs() -> dict[str, Any]:
    """Return internally consistent motor-encoding selection inputs."""
    base = _dpp_encoding_selection_inputs()
    parameters = dict(
        table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS
    )
    base["region_row"]["region_name"] = "v1"
    base["movement_selection"]["region"] = "v1"
    base["movement_selection"]["position_series_name"] = "head_position"
    base["key"] = {
        "nwb_file_name": base["movement_selection"]["nwb_file_name"],
        "epoch": base["movement_selection"]["epoch"],
        "region_sorted_spikes_group_id": base["key"][
            "region_sorted_spikes_group_id"
        ],
        "movement_firing_rate_id": base["key"]["movement_firing_rate_id"],
        **{
            f"{trajectory_type}_stability_id": base["key"][
                f"{trajectory_type}_stability_id"
            ]
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        "primary_position_series_name": "head_position",
        "orientation_reference_position_series_name": "body_position",
        "motor_encoding_param_name": parameters[
            "motor_encoding_param_name"
        ],
    }
    common_position = {
        "nwb_file_name": base["movement_selection"]["nwb_file_name"],
        "epoch": base["movement_selection"]["epoch"],
        "spatial_unit": "cm",
        "start_index": 0,
        "stop_index_exclusive": 100,
        "sample_count": 100,
        "analysis_start_offset_samples": 10,
        "start_time": 1.0,
        "stop_time": 10.0,
        "first_frame": 0,
        "last_frame": 99,
        "video_series_name": "behavior_video",
    }
    base["position_rows"] = [
        {
            **common_position,
            "position_series_name": "head_position",
            "position_role": "head",
        },
        {
            **common_position,
            "position_series_name": "body_position",
            "position_role": "body",
        },
    ]
    base["parameters"] = parameters
    return base


def _build_motor_encoding_selection(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build one motor selection row from mutable fake upstream tables."""
    return _motor_encoding_selection_row(
        key=inputs["key"],
        region_sorted_spikes_group_table=_FakeRelation(inputs["region_row"]),
        movement_firing_rate_table=_FakeRelation(inputs["movement_result"]),
        movement_firing_rate_selection_table=_FakeRelation(
            inputs["movement_selection"]
        ),
        position_table=_FakeRowsRelation(inputs["position_rows"]),
        epoch_intervals_table=_FakeRelation(inputs["epoch_row"]),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        stability_table=_FakeKeyedRelation(
            "path_specific_place_stability_id",
            inputs["stability_results"],
        ),
        stability_selection_table=_FakeKeyedRelation(
            "path_specific_place_stability_id",
            inputs["stability_selections"],
        ),
        tuning_curve_selection_table=_FakeKeyedRelation(
            "path_specific_place_tuning_curve_id",
            inputs["curve_selections"],
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
    )


def _dark_light_glm_selection_inputs() -> dict[str, Any]:
    """Return internally consistent coupled dark/light selection inputs."""
    nwb_file_name = "L1420240102_.nwb"
    epochs = {"dark": "08_r4", "light": "02_r1"}
    movement_ids = {
        condition_name: uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"v1ca1-test-dark-light-movement:{condition_name}",
        )
        for condition_name in epochs
    }
    parameters = dict(
        table_specs.CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS
    )
    region_group_id = uuid.UUID("84444444-4444-5444-8444-444444444444")
    movement_selections = {
        movement_ids[condition_name]: {
            "movement_firing_rate_id": movement_ids[condition_name],
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": "head_position",
            "movement_param_name": "default",
            "movement_parameters_sha256": "d" * 64,
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region": "v1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
        }
        for condition_name, epoch in epochs.items()
    }
    movement_results = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "n_units": 3,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }
        for movement_id in movement_ids.values()
    }
    key = {
        "nwb_file_name": nwb_file_name,
        "region_sorted_spikes_group_id": region_group_id,
        "dark_movement_firing_rate_id": movement_ids["dark"],
        "light_movement_firing_rate_id": movement_ids["light"],
        "dark_epoch": epochs["dark"],
        "light_epoch": epochs["light"],
        **{
            f"{condition_name}_{trajectory_type}_trajectory_type": (
                trajectory_type
            )
            for condition_name in ("dark", "light")
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        **{
            f"{trajectory_type}_configuration_name": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        "dark_light_glm_param_name": parameters[
            "dark_light_glm_param_name"
        ],
    }
    return {
        "key": key,
        "region_row": {
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": nwb_file_name,
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region_name": "v1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
            "n_units": 3,
            "selected_units_sha256": "a" * 64,
        },
        "movement_selections": movement_selections,
        "movement_results": movement_results,
        "position_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "position_series_name": "head_position",
                "spatial_unit": "cm",
            }
            for epoch in epochs.values()
        ],
        "epoch_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epochs["dark"],
                "epoch_type": "run",
                "condition": "dark",
                "is_light": False,
            },
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epochs["light"],
                "epoch_type": "run",
                "condition": "AB",
                "is_light": True,
            },
        ],
        "trajectory_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "interval_count": int(parameters["n_folds"]),
            }
            for epoch in epochs.values()
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        ],
        "graph_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "configuration_name": trajectory_type,
                "coordinate_unit": "cm",
            }
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        ],
        "parameters": parameters,
    }


def _build_dark_light_glm_selection(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build one dark/light selection from mutable fake upstream rows."""
    return _dark_light_glm_selection_row(
        key=inputs["key"],
        region_sorted_spikes_group_table=_FakeRelation(inputs["region_row"]),
        movement_firing_rate_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_results"],
        ),
        movement_firing_rate_selection_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_selections"],
        ),
        position_table=_FakeRowsRelation(inputs["position_rows"]),
        epoch_intervals_table=_FakeRowsRelation(inputs["epoch_rows"]),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        parameters_table=_FakeRelation(inputs["parameters"]),
    )


def _swap_glm_selection_inputs() -> dict[str, Any]:
    """Return one consistent held-out swap selection without artifact I/O."""
    from v1ca1.spyglass.swap_glm import SOURCE_MODEL_NAMES

    dark_light_inputs = _dark_light_glm_selection_inputs()
    dark_light_selection = _build_dark_light_glm_selection(
        dark_light_inputs
    )
    light_test_epoch = "06_r3"
    light_test_movement_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        "v1ca1-test-swap-light-test-movement",
    )
    light_test_movement_selection = {
        **next(iter(dark_light_inputs["movement_selections"].values())),
        "movement_firing_rate_id": light_test_movement_id,
        "epoch": light_test_epoch,
    }
    movement_selections = {
        **dark_light_inputs["movement_selections"],
        light_test_movement_id: light_test_movement_selection,
    }
    movement_results = {
        light_test_movement_id: {
            "movement_firing_rate_id": light_test_movement_id,
            "n_units": dark_light_inputs["region_row"]["n_units"],
            "analysis_status": "valid",
            "selected_units_sha256": dark_light_inputs["region_row"][
                "selected_units_sha256"
            ],
        }
    }
    parameters = dict(table_specs.DEFAULT_SWAP_GLM_PARAMETERS)
    snapshot = {
        "bundle": {"legacy_artifact_provenance": None},
        "dark_light_glm_sha256": "1" * 64,
        "selected_model_sha256_by_model": {
            model_name: str(index + 2) * 64
            for index, model_name in enumerate(SOURCE_MODEL_NAMES)
        },
        "parameter_sha256": dark_light_selection[
            "dark_light_glm_parameters_sha256"
        ],
        "output_rule_sha256": dark_light_selection[
            "dark_light_glm_output_rule_sha256"
        ],
        "analysis_status": "valid",
        "metadata": {
            "dark_light_glm_id": str(
                dark_light_selection["dark_light_glm_id"]
            ),
            "animal_name": "L14",
            "date": "20240102",
            "region": "v1",
            "light_epoch": dark_light_selection["light_epoch"],
            "dark_epoch": dark_light_selection["dark_epoch"],
        },
    }
    nwb_file_name = dark_light_selection["nwb_file_name"]
    key = {
        "nwb_file_name": nwb_file_name,
        "dark_light_glm_id": dark_light_selection["dark_light_glm_id"],
        "region_sorted_spikes_group_id": dark_light_selection[
            "region_sorted_spikes_group_id"
        ],
        "light_test_movement_firing_rate_id": light_test_movement_id,
        "dark_epoch": dark_light_selection["dark_epoch"],
        "light_train_epoch": dark_light_selection["light_epoch"],
        "light_test_epoch": light_test_epoch,
        **{
            f"light_test_{trajectory_type}_trajectory_type": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        **{
            f"{trajectory_type}_configuration_name": trajectory_type
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        },
        "swap_glm_param_name": parameters["swap_glm_param_name"],
    }
    return {
        "key": key,
        "dark_light_selection": dark_light_selection,
        "dark_light_snapshot": snapshot,
        "region_row": dark_light_inputs["region_row"],
        "movement_selections": movement_selections,
        "movement_results": movement_results,
        "epoch_rows": [
            *dark_light_inputs["epoch_rows"],
            {
                "nwb_file_name": nwb_file_name,
                "epoch": light_test_epoch,
                "epoch_type": "run",
                "condition": "BA",
                "is_light": True,
            },
        ],
        "trajectory_rows": [
            {
                "nwb_file_name": nwb_file_name,
                "epoch": light_test_epoch,
                "trajectory_type": trajectory_type,
                "interval_count": 5,
            }
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        ],
        "graph_rows": dark_light_inputs["graph_rows"],
        "parameters": parameters,
    }


def _build_swap_glm_selection(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build one held-out swap row from mutable fake upstream rows."""
    return _swap_glm_selection_row(
        key=inputs["key"],
        dark_light_glm_table=object(),
        dark_light_glm_selection_table=_FakeRelation(
            inputs["dark_light_selection"]
        ),
        region_sorted_spikes_group_table=_FakeRelation(
            inputs["region_row"]
        ),
        movement_firing_rate_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_results"],
        ),
        movement_firing_rate_selection_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_selections"],
        ),
        epoch_intervals_table=_FakeRowsRelation(inputs["epoch_rows"]),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        parameters_table=_FakeRelation(inputs["parameters"]),
        dark_light_snapshot=inputs["dark_light_snapshot"],
    )


def _swap_tuning_curve_comparison_selection_inputs() -> dict[str, Any]:
    """Return one complete three-epoch empirical swap selection."""
    region_group_id = uuid.UUID("91111111-1111-5111-8111-111111111111")
    nwb_file_name = "L1420240102_.nwb"
    selected_units_sha256 = "a" * 64
    region_row = {
        "region_sorted_spikes_group_id": region_group_id,
        "nwb_file_name": nwb_file_name,
        "unit_filter_params_name": "curated_units",
        "sorted_spikes_group_name": "all shanks",
        "region_name": "v1",
        "sorting_group_members_sha256": "b" * 64,
        "unit_filter_params_sha256": "c" * 64,
        "n_units": 5,
        "selected_units_sha256": selected_units_sha256,
    }
    epochs = {
        "dark": "08_r4",
        "light_train": "02_r1",
        "light_test": "06_r3",
    }
    conditions = {"dark": "dark", "light_train": "AB", "light_test": "BA"}
    movement_parameters = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    movement_parameters_sha256 = provenance_sha256(movement_parameters)
    movement_ids = {
        epoch_role: uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"v1ca1-test-swap-tuning-movement:{epoch_role}",
        )
        for epoch_role in epochs
    }
    movement_selections = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "region_sorted_spikes_group_id": region_group_id,
            "nwb_file_name": nwb_file_name,
            "epoch": epochs[epoch_role],
            "position_series_name": "head_position",
            "movement_param_name": "default",
            "unit_filter_params_name": "curated_units",
            "sorted_spikes_group_name": "all shanks",
            "region": "v1",
            "sorting_group_members_sha256": "b" * 64,
            "unit_filter_params_sha256": "c" * 64,
            "movement_parameters_sha256": movement_parameters_sha256,
        }
        for epoch_role, movement_id in movement_ids.items()
    }
    movement_results = {
        movement_id: {
            "movement_firing_rate_id": movement_id,
            "n_units": 5,
            "analysis_status": "valid",
            "selected_units_sha256": selected_units_sha256,
        }
        for movement_id in movement_ids.values()
    }
    movement_artifact_sha256_by_role = {
        epoch_role: {
            "firing_rate": hashlib.sha256(
                f"{epoch_role}:firing-rate".encode("utf-8")
            ).hexdigest(),
            "movement_intervals": hashlib.sha256(
                f"{epoch_role}:movement-intervals".encode("utf-8")
            ).hexdigest(),
        }
        for epoch_role in epochs
    }
    position_rows = [
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": "head_position",
            "position_role": "head",
            "analysis_start_offset_samples": 10,
            "spatial_unit": "cm",
        }
        for epoch in epochs.values()
    ]
    epoch_rows = [
        {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "epoch_type": "run",
            "condition": conditions[epoch_role],
            "is_light": epoch_role != "dark",
        }
        for epoch_role, epoch in epochs.items()
    ]
    tuning_parameters = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    tuning_parameters_sha256 = provenance_sha256(tuning_parameters)
    curve_snapshots: dict[str, dict[str, Any]] = {}
    curve_ids: dict[str, uuid.UUID] = {}
    for epoch_role, epoch in epochs.items():
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES:
            source_name = f"{epoch_role}:{trajectory_type}"
            curve_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"v1ca1-test-swap-tuning-curve:{source_name}",
            )
            curve_ids[f"{epoch_role}_{trajectory_type}_tuning_curve_id"] = (
                curve_id
            )
            curve_snapshots[source_name] = {
                "selection": {
                    "path_specific_place_tuning_curve_id": curve_id,
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch,
                    "trajectory_type": trajectory_type,
                    "configuration_name": trajectory_type,
                    "movement_firing_rate_id": movement_ids[epoch_role],
                    "tuning_curve_param_name": tuning_parameters[
                        "tuning_curve_param_name"
                    ],
                    "trial_subset": "all",
                    "tuning_curve_parameters_sha256": (
                        tuning_parameters_sha256
                    ),
                },
                "result": {
                    "n_units": 5,
                    "selected_units_sha256": selected_units_sha256,
                    "analysis_status": "valid",
                },
                "artifact_path": Path(f"/{source_name}.nc"),
                "artifact_sha256": hashlib.sha256(
                    source_name.encode("utf-8")
                ).hexdigest(),
            }
    parameters = dict(
        table_specs.MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS
    )
    key = {
        "nwb_file_name": nwb_file_name,
        "region_sorted_spikes_group_id": region_group_id,
        **{
            f"{epoch_role}_movement_firing_rate_id": movement_id
            for epoch_role, movement_id in movement_ids.items()
        },
        **curve_ids,
        **{f"{epoch_role}_epoch": epoch for epoch_role, epoch in epochs.items()},
        "swap_tuning_curve_comparison_param_name": parameters[
            "swap_tuning_curve_comparison_param_name"
        ],
    }
    return {
        "key": key,
        "region_row": region_row,
        "movement_selections": movement_selections,
        "movement_results": movement_results,
        "movement_parameters": movement_parameters,
        "movement_artifact_sha256_by_role": (
            movement_artifact_sha256_by_role
        ),
        "position_rows": position_rows,
        "epoch_rows": epoch_rows,
        "tuning_parameters": tuning_parameters,
        "curve_snapshots": curve_snapshots,
        "parameters": parameters,
        "epochs": epochs,
        "conditions": conditions,
    }


def _build_swap_tuning_curve_comparison_selection(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build one empirical swap selection from mutable fake upstream rows."""
    return _swap_tuning_curve_comparison_selection_row(
        key=inputs["key"],
        region_sorted_spikes_group_table=_FakeRelation(inputs["region_row"]),
        movement_firing_rate_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_results"],
        ),
        movement_firing_rate_selection_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            inputs["movement_selections"],
        ),
        movement_parameters_table=_FakeRelation(
            inputs["movement_parameters"]
        ),
        position_table=_FakeRowsRelation(inputs["position_rows"]),
        tuning_curve_table=object(),
        tuning_curve_selection_table=object(),
        tuning_curve_parameters_table=_FakeRelation(
            inputs["tuning_parameters"]
        ),
        epoch_intervals_table=_FakeRowsRelation(inputs["epoch_rows"]),
        parameters_table=_FakeRelation(inputs["parameters"]),
        curve_snapshots=inputs["curve_snapshots"],
        movement_artifact_sha256_by_role=inputs[
            "movement_artifact_sha256_by_role"
        ],
    )


def _swap_tuning_upstream_provenance(
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact frozen upstream block for one fake selection."""
    return {
        "selected_units_sha256": selection["selected_units_sha256"],
        "source_tuning_curve_sha256_by_role_trajectory": selection[
            "source_tuning_curve_sha256_by_role_trajectory"
        ],
        "source_tuning_parameters_sha256_by_role_trajectory": selection[
            "source_tuning_parameters_sha256_by_role_trajectory"
        ],
        "source_tuning_curve_ids_by_role_trajectory": {
            epoch_role: {
                trajectory_type: str(
                    selection[
                        f"{epoch_role}_{trajectory_type}_tuning_curve_id"
                    ]
                )
                for trajectory_type in _DPP_ENCODING_TRAJECTORIES
            }
            for epoch_role in ("dark", "light_train", "light_test")
        },
        "movement_firing_rate_table_sha256_by_role": selection[
            "movement_firing_rate_table_sha256_by_role"
        ],
        "movement_firing_rate_ids_by_role": {
            epoch_role: str(
                selection[f"{epoch_role}_movement_firing_rate_id"]
            )
            for epoch_role in ("dark", "light_train", "light_test")
        },
        "movement_intervals_sha256_by_role": selection[
            "movement_intervals_sha256_by_role"
        ],
        "position_offset_samples": selection["position_offset_samples"],
        "speed_threshold_cm_s": selection["speed_threshold_cm_s"],
    }


def test_import_is_passive_in_fresh_interpreter() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment.get("PYTHONPATH", "")]
    )
    code = """
import sys
import v1ca1.spyglass
import v1ca1.spyglass.table_specs
import v1ca1.spyglass.tables
assert 'datajoint' not in sys.modules
assert 'spyglass' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_constructed_bundle_matches_current_architecture() -> None:
    bundle, schemas, unit_selection_params = _fake_bundle()

    assert tuple(key for key in bundle if key in SOURCE_TABLE_KEYS) == SOURCE_TABLE_KEYS
    assert set(bundle) == {
        *SOURCE_TABLE_KEYS,
        "region_sorted_spikes_group",
        "movement_parameters",
        "epoch_motor_behavior_parameters",
        "epoch_motor_behavior_selection",
        "epoch_motor_behavior",
        "movement_firing_rate_selection",
        "movement_firing_rate",
        "cv_pca_parameters",
        "cv_pca_selection",
        "cv_pca",
        "ripple_modulation_parameters",
        "ripple_modulation_selection",
        "ripple_modulation",
        "tuning_curve_parameters",
        "tuning_similarity_parameters",
        "path_specific_place_tuning_curve_selection",
        "path_specific_place_tuning_curve",
        "path_specific_place_tuning_similarity_selection",
        "path_specific_place_tuning_similarity",
        "dpp_tuning_curve_selection",
        "dpp_tuning_curve",
        "path_specific_place_stability_selection",
        "path_specific_place_stability",
        "dpp_encoding_parameters",
        "dpp_encoding_selection",
        "dpp_encoding",
        "path_progression_decoding_parameters",
        "path_progression_decoding_selection",
        "path_progression_decoding",
        "path_specific_place_decoding_parameters",
        "path_specific_place_decoding_selection",
        "path_specific_place_decoding",
        "motor_encoding_parameters",
        "motor_encoding_selection",
        "motor_encoding",
        "dark_light_glm_parameters",
        "dark_light_glm_selection",
        "dark_light_glm",
        "swap_glm_parameters",
        "swap_glm_selection",
        "swap_glm",
        "swap_tuning_curve_comparison_parameters",
        "swap_tuning_curve_comparison_selection",
        "swap_tuning_curve_comparison",
        "ripple_glm_parameters",
        "ripple_glm_selection",
        "ripple_glm",
        "ripple_cross_region_xcorr_parameters",
        "ripple_cross_region_xcorr_selection",
        "ripple_cross_region_xcorr",
        "analysis_nwbfile",
    }
    assert [schema.activations[0][0] for schema in schemas] == [
        "kyuv1ca1",
        "kyuv1ca1_nwbfile",
    ]
    assert schemas[0].context["UnitSelectionParams"] is unit_selection_params
    assert "TuningCurveParameters" in schemas[0].context
    assert "PathSpecificPlaceTuningCurveSelection" in schemas[0].context
    assert "PathSpecificPlaceTuningCurve" in schemas[0].context
    assert "TuningSimilarityParameters" in schemas[0].context
    assert "PathSpecificPlaceTuningSimilaritySelection" in schemas[0].context
    assert "PathSpecificPlaceTuningSimilarity" in schemas[0].context
    assert "DPPTuningCurveSelection" in schemas[0].context
    assert "DPPTuningCurve" in schemas[0].context
    assert "PathSpecificPlaceStabilitySelection" in schemas[0].context
    assert "PathSpecificPlaceStability" in schemas[0].context
    assert "RegionSortedSpikesGroup" in schemas[0].context
    assert "CVPCAParameters" in schemas[0].context
    assert "CVPCASelection" in schemas[0].context
    assert "CVPCA" in schemas[0].context
    assert "DPPEncodingParameters" in schemas[0].context
    assert "DPPEncodingSelection" in schemas[0].context
    assert "DPPEncoding" in schemas[0].context
    assert "PathSpecificPlaceDecodingParameters" in schemas[0].context
    assert "PathSpecificPlaceDecodingSelection" in schemas[0].context
    assert "PathSpecificPlaceDecoding" in schemas[0].context
    assert "MotorEncodingParameters" in schemas[0].context
    assert "MotorEncodingSelection" in schemas[0].context
    assert "MotorEncoding" in schemas[0].context
    assert "DarkLightGLMParameters" in schemas[0].context
    assert "DarkLightGLMSelection" in schemas[0].context
    assert "DarkLightGLM" in schemas[0].context
    assert "SwapGLMParameters" in schemas[0].context
    assert "SwapGLMSelection" in schemas[0].context
    assert "SwapGLM" in schemas[0].context
    assert "SwapTuningCurveComparisonParameters" in schemas[0].context
    assert "SwapTuningCurveComparisonSelection" in schemas[0].context
    assert "SwapTuningCurveComparison" in schemas[0].context
    assert "RippleGLMParameters" in schemas[0].context
    assert "RippleGLMSelection" in schemas[0].context
    assert "RippleGLM" in schemas[0].context
    assert "RippleCrossRegionXCorrParameters" in schemas[0].context
    assert "RippleCrossRegionXCorrSelection" in schemas[0].context
    assert "RippleCrossRegionXCorr" in schemas[0].context
    assert "PathProgressionDecodingParameters" in schemas[0].context
    assert (
        "PathProgressionDecodingSelection" in schemas[0].context
    )
    assert "PathProgressionDecoding" in schemas[0].context
    assert schemas[0].context["AnalysisNwbfile"] is bundle["analysis_nwbfile"]
    legacy_stability_prefix = "".join(("TaskProgression", "Stability"))
    assert not any(
        name.startswith(legacy_stability_prefix) for name in schemas[0].context
    )
    assert all(
        activation[1]["create_schema"] and activation[1]["create_tables"]
        for schema in schemas
        for activation in schema.activations
    )
    assert not any("_insert_calls" in table.__dict__ for table in bundle.values())

    analysis_nwbfile = bundle["analysis_nwbfile"]
    analysis_nwbfile().register_with_spyglass()
    assert analysis_nwbfile._registry_calls == 1

    position_definition = bundle["position"].definition
    for source_name in ("trajectory_intervals", "ripple_intervals", "position"):
        source_definition = bundle[source_name].definition
        assert "-> EpochIntervals" in source_definition
        assert "-> Session" not in source_definition
        assert "\nepoch: varchar(64)" not in source_definition

    assert "position_series_name: varchar(255)" in position_definition
    assert "position_role: varchar(64)" in position_definition
    assert "position_type" not in position_definition
    assert "spatial_unit: enum('cm')" in position_definition

    ripple_selection = bundle["ripple_modulation_selection"].definition
    assert "ripple_modulation_id: uuid" in ripple_selection
    assert "-> RippleIntervals" in ripple_selection
    assert "-> RippleModulationParameters" in ripple_selection
    assert "-> RegionSortedSpikesGroup" in ripple_selection
    assert "-> SortedSpikesGroup" not in ripple_selection
    assert "ripple_modulation_parameters_sha256: char(64)" in ripple_selection

    ripple_result = bundle["ripple_modulation"].definition
    assert "-> RippleModulationSelection" in ripple_result
    assert "-> AnalysisNwbfile" in ripple_result
    assert "ripple_modulation_summary_object_id: varchar(40)" in ripple_result
    assert "peri_ripple_firing_rate_object_id: varchar(40)" in ripple_result
    assert "ripple_modulation_summary_sha256: char(64)" in ripple_result
    assert "peri_ripple_firing_rate_sha256: char(64)" in ripple_result
    assert "artifact_schema_version: varchar(8)" in ripple_result
    assert "summary_path:" not in ripple_result
    assert "peri_ripple_firing_rate_path:" not in ripple_result
    assert "analysis_status:" in ripple_result
    assert "selected_units_sha256: char(64)" in ripple_result
    assert "RippleModulationComputed" not in ripple_result

    movement_selection = bundle["movement_firing_rate_selection"].definition
    assert "movement_firing_rate_id: uuid" in movement_selection
    assert "-> Position" in movement_selection
    assert "-> MovementParameters" in movement_selection
    assert "-> RegionSortedSpikesGroup" in movement_selection
    assert "-> SortedSpikesGroup" not in movement_selection
    assert "movement_parameters_sha256: char(64)" in movement_selection

    movement_result = bundle["movement_firing_rate"].definition
    assert "-> MovementFiringRateSelection" in movement_result
    assert "-> AnalysisNwbfile" in movement_result
    assert "movement_firing_rate_object_id: varchar(40)" in movement_result
    assert "movement_intervals_object_id: varchar(40)" in movement_result
    assert "movement_firing_rate_sha256: char(64)" in movement_result
    assert "movement_intervals_sha256: char(64)" in movement_result
    assert "artifact_schema_version: varchar(8)" in movement_result
    assert "n_units_with_spikes: int unsigned" in movement_result
    assert "'no_valid_position'" in movement_result
    assert "'no_movement'" in movement_result
    assert "artifact_origin" not in movement_result
    assert not hasattr(bundle["movement_firing_rate"], "register_existing")

    cv_pca_parameters = bundle["cv_pca_parameters"].definition
    assert "cv_pca_param_name: varchar(64)" in cv_pca_parameters
    assert "random_seed: int unsigned" in cv_pca_parameters
    assert "min_firing_rate_hz: double" in cv_pca_parameters
    assert "\nregion:" not in cv_pca_parameters

    cv_pca_selection = bundle["cv_pca_selection"].definition
    assert "cv_pca_id: uuid" in cv_pca_selection
    assert "light_epoch='epoch'" in cv_pca_selection
    assert "dark_epoch='epoch'" in cv_pca_selection
    assert "-> RegionSortedSpikesGroup" in cv_pca_selection
    assert "light_movement_firing_rate_id='movement_firing_rate_id'" in (
        cv_pca_selection
    )
    assert "dark_movement_firing_rate_id='movement_firing_rate_id'" in (
        cv_pca_selection
    )
    assert "position_offset_samples: bigint unsigned" in cv_pca_selection
    assert "trajectory_intervals_sha256_by_epoch_and_type: longblob" in (
        cv_pca_selection
    )
    assert "graph_inputs_sha256_by_trajectory: longblob" in cv_pca_selection
    assert "cv_pca_output_rule_sha256: char(64)" in cv_pca_selection
    assert "movement_firing_rate_file_sha256" not in cv_pca_selection
    assert "movement_intervals_file_sha256" not in cv_pca_selection

    cv_pca_result = bundle["cv_pca"].definition
    assert "-> CVPCASelection" in cv_pca_result
    assert "-> AnalysisNwbfile" in cv_pca_result
    for object_name in (
        "selected_units",
        "lap_assignments",
        "trajectory_qc",
        "summary",
        "spectrum",
        "dataset",
        "provenance",
    ):
        assert f"{object_name}_object_id: varchar(40)" in cv_pca_result
    assert "artifact_manifest_path" not in cv_pca_result
    assert "filepath@analysis" not in cv_pca_result
    assert "'no_movement'" in cv_pca_result
    assert "'insufficient_laps'" in cv_pca_result
    assert hasattr(bundle["cv_pca"], "register_existing")

    tuning_selection = bundle[
        "path_specific_place_tuning_curve_selection"
    ].definition
    assert "path_specific_place_tuning_curve_id: uuid" in tuning_selection
    assert "-> TrajectoryIntervals" in tuning_selection
    assert "-> WTrackGraph" in tuning_selection
    assert "-> MovementFiringRate" in tuning_selection
    assert "-> TuningCurveParameters" in tuning_selection
    assert "trial_subset: enum('all', 'odd', 'even')" in tuning_selection
    assert "tuning_curve_parameters_sha256: char(64)" in tuning_selection
    assert "-> Position" not in tuning_selection
    assert "-> SortedSpikesGroup" not in tuning_selection

    tuning_result = bundle["path_specific_place_tuning_curve"].definition
    assert "-> PathSpecificPlaceTuningCurveSelection" in tuning_result
    assert "-> AnalysisNwbfile" in tuning_result
    for object_name in (
        "path_specific_place_tuning",
        "path_specific_place_bins",
        "path_specific_place_provenance",
    ):
        assert f"{object_name}_object_id: varchar(40)" in tuning_result
        assert f"{object_name}_sha256: char(64)" in tuning_result
    assert "artifact_schema_version: varchar(8)" in tuning_result
    assert "tuning_curve_path:" not in tuning_result
    assert "n_trials: int unsigned" in tuning_result
    assert "n_feature_samples: int unsigned" in tuning_result
    assert "n_position_bins: int unsigned" in tuning_result
    assert "'no_trials'" in tuning_result
    assert "selected_units_sha256: char(64)" in tuning_result

    similarity_parameters = bundle["tuning_similarity_parameters"].definition
    assert "tuning_similarity_param_name: varchar(64)" in similarity_parameters
    assert (
        "enum('correlation', 'absolute_overlap', 'shape_overlap')"
        in similarity_parameters
    )

    similarity_selection = bundle[
        "path_specific_place_tuning_similarity_selection"
    ].definition
    assert (
        "path_specific_place_tuning_similarity_id: uuid"
        in similarity_selection
    )
    for alias in (
        "center_to_left_tuning_curve_id",
        "center_to_right_tuning_curve_id",
        "left_to_center_tuning_curve_id",
        "right_to_center_tuning_curve_id",
    ):
        assert f"{alias}='path_specific_place_tuning_curve_id'" in (
            similarity_selection
        )
    assert "-> TuningSimilarityParameters" in similarity_selection
    assert "tuning_similarity_parameters_sha256: char(64)" in (
        similarity_selection
    )

    similarity_result = bundle[
        "path_specific_place_tuning_similarity"
    ].definition
    assert "-> PathSpecificPlaceTuningSimilaritySelection" in similarity_result
    assert "-> AnalysisNwbfile" in similarity_result
    assert "similarity_object_id: varchar(40)" in similarity_result
    assert "similarity_sha256: char(64)" in similarity_result
    assert "artifact_schema_version: varchar(8)" in similarity_result
    assert "similarity_path" not in similarity_result
    assert "n_valid_comparisons: int unsigned" in similarity_result
    assert "n_units_with_valid_comparison: int unsigned" in similarity_result
    assert "'no_valid_comparisons'" in similarity_result
    assert "selected_units_sha256: char(64)" in similarity_result

    dpp_selection = bundle["dpp_tuning_curve_selection"].definition
    assert "dpp_tuning_curve_id: uuid" in dpp_selection
    assert (
        "outbound_trajectory_type='trajectory_type'" in dpp_selection
    )
    assert "inbound_trajectory_type='trajectory_type'" in dpp_selection
    assert (
        "outbound_configuration_name='configuration_name'" in dpp_selection
    )
    assert "inbound_configuration_name='configuration_name'" in dpp_selection
    assert "-> MovementFiringRate" in dpp_selection
    assert "-> TuningCurveParameters" in dpp_selection
    assert "turn_type: enum('left', 'right')" in dpp_selection
    assert "trial_subset: enum('all', 'odd', 'even')" in dpp_selection

    dpp_result = bundle["dpp_tuning_curve"].definition
    assert "-> DPPTuningCurveSelection" in dpp_result
    assert "-> AnalysisNwbfile" in dpp_result
    for object_name in ("dpp_tuning", "dpp_bins", "dpp_provenance"):
        assert f"{object_name}_object_id: varchar(40)" in dpp_result
        assert f"{object_name}_sha256: char(64)" in dpp_result
    assert "artifact_schema_version: varchar(8)" in dpp_result
    assert "tuning_curve_path:" not in dpp_result
    assert "n_outbound_trials: int unsigned" in dpp_result
    assert "n_inbound_trials: int unsigned" in dpp_result
    assert "selected_units_sha256: char(64)" in dpp_result

    stability_selection = bundle[
        "path_specific_place_stability_selection"
    ].definition
    assert "path_specific_place_stability_id: uuid" in stability_selection
    assert "odd_path_specific_place_tuning_curve_id=" in stability_selection
    assert "even_path_specific_place_tuning_curve_id=" in stability_selection
    assert "-> TrajectoryIntervals" not in stability_selection
    assert "-> MovementFiringRate" not in stability_selection
    assert "-> TuningCurveParameters" not in stability_selection

    stability_result = bundle["path_specific_place_stability"].definition
    assert "-> PathSpecificPlaceStabilitySelection" in stability_result
    assert "-> AnalysisNwbfile" in stability_result
    assert "stability_object_id: varchar(40)" in stability_result
    assert "stability_sha256: char(64)" in stability_result
    assert "artifact_schema_version: varchar(8)" in stability_result
    assert "stability_path" not in stability_result
    assert "analysis_status:" in stability_result
    assert "'no_valid_position'" in stability_result
    assert "'no_movement'" in stability_result
    assert "selected_units_sha256: char(64)" in stability_result

    region_group = bundle["region_sorted_spikes_group"].definition
    assert "region_sorted_spikes_group_id: uuid" in region_group
    assert "-> SortedSpikesGroup" in region_group
    assert "region_name: varchar(64)" in region_group
    assert "n_units: int unsigned" in region_group
    assert "selected_units_sha256: char(64)" in region_group
    assert "unit_ids" not in region_group

    encoding_parameters = bundle[
        "dpp_encoding_parameters"
    ].definition
    assert "evaluation_bin_size_s: double" in encoding_parameters
    assert "spatial_bin_size_cm: double" in encoding_parameters
    assert "minimum_movement_firing_rate_hz: double" in encoding_parameters
    assert "minimum_stability_correlation: double" in encoding_parameters

    encoding_selection = bundle[
        "dpp_encoding_selection"
    ].definition
    assert "dpp_encoding_id: uuid" in encoding_selection
    assert "-> RegionSortedSpikesGroup" in encoding_selection
    assert "-> MovementFiringRate" in encoding_selection
    assert "full_w_configuration_name='configuration_name'" in encoding_selection
    for trajectory_type in (
        "center_to_left",
        "center_to_right",
        "left_to_center",
        "right_to_center",
    ):
        assert f"{trajectory_type}_trajectory_type='trajectory_type'" in (
            encoding_selection
        )
        assert f"{trajectory_type}_configuration_name='configuration_name'" in (
            encoding_selection
        )
        assert f"{trajectory_type}_stability_id=" in encoding_selection

    encoding_result = bundle["dpp_encoding"].definition
    assert "-> DPPEncodingSelection" in encoding_result
    assert "-> AnalysisNwbfile" in encoding_result
    assert "dpp_encoding_object_id: varchar(40)" in encoding_result
    assert "dpp_encoding_sha256: char(64)" in encoding_result
    assert "artifact_schema_version: varchar(8)" in encoding_result
    assert "dpp_encoding_path" not in encoding_result
    assert "n_units_input: int unsigned" in encoding_result
    assert "n_units_eligible: int unsigned" in encoding_result
    assert "n_units_valid: int unsigned" in encoding_result
    assert "'no_eligible_units'" in encoding_result
    assert "'no_valid_units'" in encoding_result

    decoding_parameters = bundle[
        "path_progression_decoding_parameters"
    ].definition
    assert "decoding_bin_size_s: double" in decoding_parameters
    assert "sliding_window_size_bins: smallint unsigned" in (
        decoding_parameters
    )
    assert "spatial_bin_size_cm: double" in decoding_parameters
    assert "minimum_movement_firing_rate_hz: double" in decoding_parameters
    assert "minimum_stability_correlation = NULL: double" in (
        decoding_parameters
    )
    assert "n_folds" not in decoding_parameters

    decoding_selection = bundle[
        "path_progression_decoding_selection"
    ].definition
    assert "path_progression_decoding_id: uuid" in (
        decoding_selection
    )
    assert "-> RegionSortedSpikesGroup" in decoding_selection
    assert "cohort_movement_firing_rate_id='movement_firing_rate_id'" in (
        decoding_selection
    )
    assert "cohort_epoch: varchar(64)" in decoding_selection
    assert "eligibility_rule_sha256: char(64)" in decoding_selection
    assert "transfer_spec_sha256: char(64)" in decoding_selection
    for trajectory_type in (
        "center_to_left",
        "center_to_right",
        "left_to_center",
        "right_to_center",
    ):
        assert f"{trajectory_type}_trajectory_type='trajectory_type'" in (
            decoding_selection
        )
        assert f"{trajectory_type}_configuration_name='configuration_name'" in (
            decoding_selection
        )
        assert f"{trajectory_type}_stability_id=" in decoding_selection
        assert f"cohort_{trajectory_type}_stability_id=" in decoding_selection

    decoding_result = bundle[
        "path_progression_decoding"
    ].definition
    assert "-> PathProgressionDecodingSelection" in decoding_result
    assert "-> AnalysisNwbfile" in decoding_result
    for object_id_field in (
        "unit_eligibility_object_id",
        "selected_units_object_id",
        "decoding_summary_object_id",
        "cross_path_binned_error_object_id",
        "transfer_index_object_id",
        "decoding_provenance_object_id",
    ):
        assert f"{object_id_field}: varchar(40)" in decoding_result
    assert "filepath@analysis" not in decoding_result
    assert "artifact_schema_version: varchar(16)" in decoding_result
    assert "n_transfer_pairs_expected: smallint unsigned" in decoding_result
    assert "n_transfer_pairs_valid: smallint unsigned" in decoding_result
    assert "'partial_valid'" in decoding_result
    transfer_definition = bundle[
        "path_progression_decoding"
    ].Transfer.definition
    assert transfer_definition == (
        table_specs.PATH_PROGRESSION_DECODING_TRANSFER_DEFINITION
    )
    for object_id_field in (
        "true_progression_object_id",
        "decoded_progression_object_id",
        "decoding_support_object_id",
    ):
        assert f"{object_id_field}: varchar(40)" in transfer_definition
    assert not hasattr(
        bundle["path_progression_decoding"],
        "register_existing",
    )

    place_decoding_parameters = bundle[
        "path_specific_place_decoding_parameters"
    ].definition
    assert "n_folds: smallint unsigned" in place_decoding_parameters
    assert "decoding_bin_size_s: double" in place_decoding_parameters
    assert "random_seed: int unsigned" in place_decoding_parameters

    place_decoding_selection = bundle[
        "path_specific_place_decoding_selection"
    ].definition
    assert "path_specific_place_decoding_id: uuid" in (
        place_decoding_selection
    )
    assert "-> RegionSortedSpikesGroup" in place_decoding_selection
    assert "-> MovementFiringRate" in place_decoding_selection
    assert "PathSpecificPlaceStability" not in place_decoding_selection

    place_decoding_result = bundle[
        "path_specific_place_decoding"
    ].definition
    assert "-> PathSpecificPlaceDecodingSelection" in place_decoding_result
    assert "-> AnalysisNwbfile" in place_decoding_result
    for object_id_field in (
        "selected_units_object_id",
        "fold_qc_object_id",
        "decoding_summary_object_id",
        "decoding_error_by_position_object_id",
        "true_position_object_id",
        "decoded_position_object_id",
        "decoding_support_object_id",
        "decoding_provenance_object_id",
    ):
        assert f"{object_id_field}: varchar(40)" in place_decoding_result
    assert "artifact_manifest_path" not in place_decoding_result
    assert "filepath@analysis" not in place_decoding_result
    assert "artifact_schema_version: varchar(16)" in place_decoding_result
    assert "'partial_valid'" in place_decoding_result
    assert hasattr(
        bundle["path_specific_place_decoding"],
        "register_existing",
    )

    motor_parameters = bundle[
        "motor_encoding_parameters"
    ].definition
    assert "outer_n_folds: smallint unsigned" in motor_parameters
    assert "inner_n_folds: smallint unsigned" in motor_parameters
    assert "ridge_values: longblob" in motor_parameters
    assert "spatial_bin_sizes_cm: longblob" in motor_parameters
    assert "minimum_movement_firing_rate_hz: double" in motor_parameters

    motor_selection = bundle[
        "motor_encoding_selection"
    ].definition
    assert "motor_encoding_id: uuid" in motor_selection
    assert "-> RegionSortedSpikesGroup" in motor_selection
    assert "-> MovementFiringRate" in motor_selection
    assert "primary_position_series_name='position_series_name'" in (
        motor_selection
    )
    assert (
        "orientation_reference_position_series_name='position_series_name'"
        in motor_selection
    )
    assert "full_w_configuration_name='configuration_name'" in motor_selection
    assert "motor_encoding_model_spec_sha256" in motor_selection
    assert "motor_encoding_output_rule_sha256" in motor_selection

    motor_result = bundle["motor_encoding"].definition
    assert "-> MotorEncodingSelection" in motor_result
    assert "-> AnalysisNwbfile" in motor_result
    assert "selected_units_object_id: varchar(40)" in motor_result
    assert "dataset_index_object_id: varchar(40)" in motor_result
    assert "coordinates_object_id: varchar(40)" in motor_result
    assert "nested_cv_arrays_object_id: varchar(40)" in motor_result
    assert "full_refit_arrays_object_id: varchar(40)" in motor_result
    assert "provenance_object_id: varchar(40)" in motor_result
    assert "motor_encoding_sha256: char(64)" in motor_result
    assert "filepath@analysis" not in motor_result
    assert "'partial_valid'" in motor_result
    assert hasattr(bundle["motor_encoding"], "register_existing")

    dark_light_parameters = bundle["dark_light_glm_parameters"].definition
    assert "basis_candidate_mode:" in dark_light_parameters
    assert "speed_smoothing_sigma_s = 0.1: double" in dark_light_parameters
    dark_light_selection = bundle["dark_light_glm_selection"].definition
    assert "dark_light_glm_id: uuid" in dark_light_selection
    assert "dark_movement_firing_rate_id='movement_firing_rate_id'" in (
        dark_light_selection
    )
    assert "light_movement_firing_rate_id='movement_firing_rate_id'" in (
        dark_light_selection
    )
    assert "dark_center_to_left_trajectory_type='trajectory_type'" in (
        dark_light_selection
    )
    assert "light_right_to_center_trajectory_type='trajectory_type'" in (
        dark_light_selection
    )
    dark_light_result = bundle["dark_light_glm"].definition
    assert "-> DarkLightGLMSelection" in dark_light_result
    assert "-> AnalysisNwbfile" in dark_light_result
    assert "candidate_results_object_id: varchar(40)" in dark_light_result
    assert "selection_summary_object_id: varchar(40)" in dark_light_result
    assert "artifact_manifest_path" not in dark_light_result
    assert "'partial_valid'" in dark_light_result
    assert hasattr(bundle["dark_light_glm"], "register_existing")

    swap_parameters = bundle["swap_glm_parameters"].definition
    assert "swap_light_offset: bool" in swap_parameters
    assert "observed_spatial_bin_size_cm: double" in swap_parameters
    swap_selection = bundle["swap_glm_selection"].definition
    assert "swap_glm_id: uuid" in swap_selection
    assert "-> DarkLightGLM" in swap_selection
    assert "-> RegionSortedSpikesGroup" in swap_selection
    assert "light_test_movement_firing_rate_id=" in swap_selection
    assert "light_test_center_to_left_trajectory_type=" in swap_selection
    assert "dark_light_glm_sha256: char(64)" in swap_selection
    assert (
        "dark_light_selected_model_sha256_by_model: longblob"
        in swap_selection
    )
    assert "dark_light_parameter_sha256: char(64)" in swap_selection
    assert "upstream_analysis_status:" in swap_selection
    swap_result = bundle["swap_glm"].definition
    assert "-> SwapGLMSelection" in swap_result
    assert "-> AnalysisNwbfile" in swap_result
    assert "model_results_object_id: varchar(40)" in swap_result
    assert "observed_response_object_id: varchar(40)" in swap_result
    assert "artifact_manifest_path" not in swap_result
    assert "'upstream_terminal'" in swap_result
    assert "'no_trajectory_samples'" in swap_result
    assert hasattr(bundle["swap_glm"], "register_existing")

    swap_tuning_parameters = bundle[
        "swap_tuning_curve_comparison_parameters"
    ].definition
    assert "evaluation_bin_size_s: double" in swap_tuning_parameters
    assert "gaussian_smoothing_sigma_bins: double" in (
        swap_tuning_parameters
    )
    swap_tuning_selection = bundle[
        "swap_tuning_curve_comparison_selection"
    ].definition
    assert "swap_tuning_curve_comparison_id: uuid" in swap_tuning_selection
    assert "-> RegionSortedSpikesGroup" in swap_tuning_selection
    for epoch_role in ("dark", "light_train", "light_test"):
        assert (
            f"{epoch_role}_movement_firing_rate_id='movement_firing_rate_id'"
            in swap_tuning_selection
        )
        assert f"{epoch_role}_epoch='epoch'" in swap_tuning_selection
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES:
            assert (
                f"{epoch_role}_{trajectory_type}_tuning_curve_id="
                in swap_tuning_selection
            )
    assert "source_tuning_curve_sha256_by_role_trajectory" in (
        swap_tuning_selection
    )
    assert "source_tuning_parameters_sha256_by_role_trajectory" in (
        swap_tuning_selection
    )
    assert "movement_firing_rate_table_sha256_by_role" in (
        swap_tuning_selection
    )
    assert "movement_intervals_sha256_by_role" in swap_tuning_selection
    swap_tuning_result = bundle[
        "swap_tuning_curve_comparison"
    ].definition
    assert "-> SwapTuningCurveComparisonSelection" in swap_tuning_result
    assert "-> AnalysisNwbfile" in swap_tuning_result
    for object_name in (
        "selected_units",
        "score_summary",
        "source_profiles",
        "model_profiles",
        "geometry",
        "provenance",
    ):
        assert f"{object_name}_object_id: varchar(40)" in swap_tuning_result
    for hash_name in (
        "selected_units_table",
        "score_summary",
        "source_profiles",
        "model_profiles",
        "geometry",
        "provenance",
    ):
        assert f"{hash_name}_sha256: char(64)" in swap_tuning_result
    assert "artifact_schema_version: varchar(8)" in swap_tuning_result
    assert "artifact_manifest_path:" not in swap_tuning_result
    assert "swap_tuning_curve_comparison_path:" not in swap_tuning_result
    assert "n_source_units: int unsigned" in swap_tuning_result
    assert "'upstream_terminal'" in swap_tuning_result
    assert hasattr(
        bundle["swap_tuning_curve_comparison"],
        "register_existing",
    )

    ripple_glm_parameters = bundle["ripple_glm_parameters"].definition
    assert "ripple_selection_mode:" in ripple_glm_parameters
    assert "source_predictor_mode:" in ripple_glm_parameters
    ripple_glm_selection = bundle["ripple_glm_selection"].definition
    assert "ripple_glm_id: uuid" in ripple_glm_selection
    assert "-> RippleIntervals" in ripple_glm_selection
    assert "source_region_sorted_spikes_group_id=" in ripple_glm_selection
    assert "target_region_sorted_spikes_group_id=" in ripple_glm_selection
    assert "detector_zscore_threshold: double" in ripple_glm_selection
    assert "speed_gated: bool" in ripple_glm_selection
    assert "source_ripple_intervals_sha256" in ripple_glm_selection
    assert "ripple_provenance_sha256" in ripple_glm_selection
    ripple_glm_result = bundle["ripple_glm"].definition
    assert "-> RippleGLMSelection" in ripple_glm_result
    assert "-> AnalysisNwbfile" in ripple_glm_result
    for object_name in (
        "selected_units",
        "summary",
        "events",
        "source_features",
        "target_results",
        "provenance",
    ):
        assert f"{object_name}_object_id: varchar(40)" in ripple_glm_result
    assert "artifact_schema_version: varchar(8)" in ripple_glm_result
    assert "ripple_glm_sha256: char(64)" in ripple_glm_result
    assert "artifact_manifest_path" not in ripple_glm_result
    assert "ripple_glm_path" not in ripple_glm_result
    assert "n_valid_target_units: int unsigned" in ripple_glm_result
    assert hasattr(bundle["ripple_glm"], "register_existing")

    xcorr_parameters = bundle["ripple_cross_region_xcorr_parameters"].definition
    assert "bin_size_s: double" in xcorr_parameters
    assert "min_ripple_spikes: int unsigned" in xcorr_parameters
    xcorr_selection = bundle["ripple_cross_region_xcorr_selection"].definition
    assert "ripple_cross_region_xcorr_id: uuid" in xcorr_selection
    assert "-> RippleIntervals" in xcorr_selection
    assert "source_region_sorted_spikes_group_id=" in xcorr_selection
    assert "target_region_sorted_spikes_group_id=" in xcorr_selection
    assert "selected_ripple_intervals_sha256: char(64)" in xcorr_selection
    xcorr_result = bundle["ripple_cross_region_xcorr"].definition
    assert "-> RippleCrossRegionXCorrSelection" in xcorr_result
    assert "-> AnalysisNwbfile" in xcorr_result
    assert "ca1_units_object_id: varchar(40)" in xcorr_result
    assert "v1_units_object_id: varchar(40)" in xcorr_result
    assert "pair_xcorr_object_id: varchar(40)" in xcorr_result
    assert "lag_axis_object_id: varchar(40)" in xcorr_result
    assert "ripple_support_object_id: varchar(40)" in xcorr_result
    assert "provenance_object_id: varchar(40)" in xcorr_result
    assert "artifact_manifest_path" not in xcorr_result
    assert "ripple_cross_region_xcorr_path" not in xcorr_result
    assert "n_valid_pairs: int unsigned" in xcorr_result
    assert hasattr(bundle["ripple_cross_region_xcorr"], "register_existing")

def test_region_sorted_group_registration_skips_empty_and_bulk_inserts(
    monkeypatch,
) -> None:
    from v1ca1.spyglass import region_sorted_spikes

    loaded_regions = []

    def load_group(*, region, **_kwargs):
        loaded_regions.append(region)
        return {"region": region, "n_units": 2 if region == "v1" else 0}

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_load_group_unit_data",
        load_group,
    )
    monkeypatch.setattr(
        region_sorted_spikes,
        "build_region_sorted_spikes_group_row",
        lambda loaded: {
            "region_sorted_spikes_group_id": uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"test-region:{loaded['region']}",
            ),
            "region_name": loaded["region"],
            "n_units": loaded["n_units"],
        },
    )
    bundle, _, _ = _fake_bundle()
    table = bundle["region_sorted_spikes_group"]
    key = {
        "nwb_file_name": "L1420240102_.nwb",
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": "all shanks",
    }

    rows = table.register_regions(
        key,
        region_names=("V1", "ca1"),
        skip_duplicates=True,
    )

    assert loaded_regions == ["v1", "ca1"]
    assert [row["region_name"] for row in rows] == ["v1"]
    assert table._insert_many_calls == [
        (rows, {"skip_duplicates": True})
    ]

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_load_group_unit_data",
        lambda **kwargs: {"region": kwargs["region"], "n_units": 0},
    )
    empty_bundle, _, _ = _fake_bundle()
    with pytest.raises(ValueError, match="None of the requested regions"):
        empty_bundle["region_sorted_spikes_group"].register_regions(key)
    assert "_insert_many_calls" not in empty_bundle[
        "region_sorted_spikes_group"
    ].__dict__


def test_analysis_schema_prefix_is_validated_before_activation() -> None:
    matching = SimpleNamespace(config={"custom": {"database.prefix": "kyuv1ca1"}})
    _validate_analysis_schema_prefix(matching, "kyuv1ca1_nwbfile")

    with pytest.raises(ValueError, match="database.prefix.*kyuv1ca1"):
        _validate_analysis_schema_prefix(SimpleNamespace(config={}), "kyuv1ca1_nwbfile")

    mismatch = SimpleNamespace(
        config={"custom": {"database.prefix": "another_project"}}
    )
    with pytest.raises(ValueError, match="found 'another_project'"):
        _validate_analysis_schema_prefix(mismatch, "kyuv1ca1_nwbfile")

    with pytest.raises(ValueError, match="<prefix>_nwbfile"):
        _validate_analysis_schema_prefix(matching, "kyuv1ca1_results")


def test_parameter_tables_insert_current_scalar_defaults() -> None:
    bundle, _, _ = _fake_bundle()
    movement_parameters = bundle["movement_parameters"]
    ripple_parameters = bundle["ripple_modulation_parameters"]
    tuning_parameters = bundle["tuning_curve_parameters"]
    similarity_parameters = bundle["tuning_similarity_parameters"]
    encoding_parameters = bundle["dpp_encoding_parameters"]
    decoding_parameters = bundle["path_progression_decoding_parameters"]
    motor_parameters = bundle["motor_encoding_parameters"]
    swap_tuning_parameters = bundle[
        "swap_tuning_curve_comparison_parameters"
    ]
    ripple_glm_parameters = bundle["ripple_glm_parameters"]
    xcorr_parameters = bundle["ripple_cross_region_xcorr_parameters"]
    cv_pca_parameters = bundle["cv_pca_parameters"]

    movement_row = movement_parameters.insert_default()
    ripple_row = ripple_parameters.insert_default()
    tuning_rows = tuning_parameters.insert_presets()
    similarity_rows = similarity_parameters.insert_presets()
    encoding_row = encoding_parameters.insert_default()
    decoding_row = decoding_parameters.insert_default()
    motor_rows = motor_parameters.insert_presets()
    swap_tuning_rows = swap_tuning_parameters.insert_defaults()
    ripple_glm_rows = ripple_glm_parameters.insert_defaults()
    xcorr_rows = xcorr_parameters.insert_defaults()
    cv_pca_rows = cv_pca_parameters.insert_presets()

    assert movement_row == dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    assert movement_row == {
        "movement_param_name": "default",
        "speed_threshold_cm_s": 4.0,
        "speed_smoothing_sigma_s": 0.1,
    }
    assert movement_parameters._insert_calls == [
        (movement_row, {"skip_duplicates": True})
    ]
    assert cv_pca_rows == [
        dict(row) for row in table_specs.CV_PCA_PARAMETER_PRESETS
    ]
    assert cv_pca_parameters._insert_calls == [
        (row, {"skip_duplicates": True}) for row in cv_pca_rows
    ]
    assert _validate_cv_pca_parameter_row(
        {
            **cv_pca_rows[0],
            "n_groups": np.int64(4),
            "random_seed": np.int64(47),
        }
    ) == cv_pca_rows[0]
    with pytest.raises(ValueError, match="exactly the declared fields"):
        cv_pca_parameters.insert_parameters(
            {**cv_pca_rows[0], "region": "v1"}
        )
    with pytest.raises(ValueError, match="at least 3"):
        cv_pca_parameters.insert_parameters(
            {**cv_pca_rows[0], "n_groups": 2}
        )

    assert ripple_row == dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    assert "minimum_ripple_mean_zscore" not in ripple_row
    assert ripple_row["expected_detector_zscore_threshold"] == pytest.approx(2.0)
    assert ripple_row["require_speed_gated"] is True
    assert "longblob" not in ripple_parameters.definition.lower()
    assert ripple_parameters._insert_calls == [
        (ripple_row, {"skip_duplicates": True})
    ]

    assert tuning_rows == [
        dict(row) for row in table_specs.TUNING_CURVE_PARAMETER_PRESETS
    ]
    assert tuning_rows == [
        {
            "tuning_curve_param_name": "legacy_4cm_unsmoothed",
            "binning_mode": "bin_size_cm",
            "place_bin_size_cm": 4.0,
            "position_bin_count": None,
            "gaussian_smoothing_sigma_bins": 0.0,
        },
        {
            "tuning_curve_param_name": "figure1d_50bin_sigma1p5",
            "binning_mode": "bin_count",
            "place_bin_size_cm": None,
            "position_bin_count": 50,
            "gaussian_smoothing_sigma_bins": 1.5,
        },
    ]
    assert tuning_parameters._insert_calls == [
        (row, {"skip_duplicates": True}) for row in tuning_rows
    ]

    assert similarity_rows == [
        dict(row) for row in table_specs.TUNING_SIMILARITY_PARAMETER_PRESETS
    ]
    assert similarity_rows == [
        {
            "tuning_similarity_param_name": "correlation",
            "similarity_metric": "correlation",
        },
        {
            "tuning_similarity_param_name": "absolute_overlap",
            "similarity_metric": "absolute_overlap",
        },
        {
            "tuning_similarity_param_name": "shape_overlap",
            "similarity_metric": "shape_overlap",
        },
    ]
    assert similarity_parameters._insert_calls == [
        (row, {"skip_duplicates": True}) for row in similarity_rows
    ]

    assert encoding_row == dict(
        table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS
    )
    assert encoding_row == {
        "dpp_encoding_param_name": (
            "manuscript_5fold_50ms_4cm_sigma1"
        ),
        "n_folds": 5,
        "evaluation_bin_size_s": 0.05,
        "spatial_bin_size_cm": 4.0,
        "gaussian_smoothing_sigma_bins": 1.0,
        "random_seed": 47,
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": 0.5,
    }
    assert encoding_parameters._insert_calls == [
        (encoding_row, {"skip_duplicates": True})
    ]
    assert decoding_row == dict(
        table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS
    )
    assert decoding_row == {
        "path_progression_decoding_param_name": (
            "manuscript_20ms_window4_4cm_mfr0p5"
        ),
        "decoding_bin_size_s": 0.02,
        "sliding_window_size_bins": 4,
        "spatial_bin_size_cm": 4.0,
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": None,
    }
    assert decoding_parameters._insert_calls == [
        (decoding_row, {"skip_duplicates": True})
    ]
    assert motor_rows == [
        dict(row)
        for row in table_specs.MOTOR_ENCODING_PARAMETER_PRESETS
    ]
    assert [
        row["minimum_movement_firing_rate_hz"] for row in motor_rows
    ] == [0.5, 0.0]
    assert all(row["evaluation_bin_size_s"] == 0.05 for row in motor_rows)
    assert all(row["outer_n_folds"] == 5 for row in motor_rows)
    assert all(row["inner_n_folds"] == 3 for row in motor_rows)
    assert all(row["random_seed"] == 0 for row in motor_rows)
    assert motor_parameters._insert_calls == [
        (row, {"skip_duplicates": True}) for row in motor_rows
    ]
    assert swap_tuning_rows == [
        dict(row)
        for row in table_specs.SWAP_TUNING_CURVE_COMPARISON_PARAMETER_PRESETS
    ]
    assert [
        row["min_dark_firing_rate_hz"] for row in swap_tuning_rows
    ] == [0.5, 0.0]
    assert [
        row["min_light_firing_rate_hz"] for row in swap_tuning_rows
    ] == [0.5, 0.0]
    assert all(
        row["evaluation_bin_size_s"] == 0.05
        for row in swap_tuning_rows
    )
    assert swap_tuning_parameters._insert_many_calls == [
        (swap_tuning_rows, {"skip_duplicates": True})
    ]
    assert ripple_glm_rows == [
        dict(row) for row in table_specs.RIPPLE_GLM_PARAMETER_PRESETS
    ]
    assert [row["source_predictor_mode"] for row in ripple_glm_rows] == [
        "unit_vector",
        "mean_activity",
    ]
    assert all(row["ripple_selection_mode"] == "single" for row in ripple_glm_rows)
    assert all(row["n_shuffles_ripple"] == 100 for row in ripple_glm_rows)
    assert all(
        row["expected_detector_zscore_threshold"] == pytest.approx(2.0)
        and row["require_speed_gated"] is True
        for row in ripple_glm_rows
    )
    assert ripple_glm_parameters._insert_many_calls == [
        (ripple_glm_rows, {"skip_duplicates": True})
    ]
    fetched_ripple_glm_row = {
        **ripple_glm_rows[0],
        "source_target_windows_differ": np.int64(0),
        "require_speed_gated": np.bool_(True),
    }
    assert _validate_ripple_glm_parameter_row(
        fetched_ripple_glm_row
    ) == ripple_glm_rows[0]
    assert xcorr_rows == [
        dict(table_specs.MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS)
    ]
    assert xcorr_parameters._insert_many_calls == [
        (xcorr_rows, {"skip_duplicates": True})
    ]
    fetched_xcorr_row = {
        **xcorr_rows[0],
        "norm": np.int64(1),
        "require_speed_gated": np.bool_(True),
    }
    assert _validate_ripple_cross_region_xcorr_parameter_row(
        fetched_xcorr_row
    ) == xcorr_rows[0]
    with pytest.raises(TypeError, match="numeric scalar"):
        ripple_parameters.insert_parameters({**ripple_row, "bin_size_s": [0.02]})
    with pytest.raises(ValueError, match="outside the peri-ripple window"):
        ripple_parameters.insert_parameters(
            {**ripple_row, "baseline_window_start_s": -0.6}
        )
    zero_threshold = movement_parameters.insert_parameters(
        {**movement_row, "movement_param_name": "zero", "speed_threshold_cm_s": 0}
    )
    assert zero_threshold["speed_threshold_cm_s"] == 0.0
    with pytest.raises(TypeError, match="numeric scalar"):
        movement_parameters.insert_parameters(
            {**movement_row, "speed_threshold_cm_s": [4.0]}
        )
    with pytest.raises(ValueError, match="non-negative"):
        movement_parameters.insert_parameters(
            {**movement_row, "speed_threshold_cm_s": -1.0}
        )
    with pytest.raises(ValueError, match="positive"):
        movement_parameters.insert_parameters(
            {**movement_row, "speed_smoothing_sigma_s": 0.0}
        )
    with pytest.raises(ValueError, match="between 2 and 65535"):
        encoding_parameters.insert_parameters(
            {**encoding_row, "n_folds": 1}
        )
    with pytest.raises(ValueError, match="between -1 and 1"):
        encoding_parameters.insert_parameters(
            {**encoding_row, "minimum_stability_correlation": 1.1}
        )
    with pytest.raises(ValueError, match="between 1 and 65535"):
        decoding_parameters.insert_parameters(
            {**decoding_row, "sliding_window_size_bins": 0}
        )
    with pytest.raises(ValueError, match="positive"):
        decoding_parameters.insert_parameters(
            {**decoding_row, "decoding_bin_size_s": 0.0}
        )
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        decoding_parameters.insert_parameters(
            {**decoding_row, "minimum_stability_correlation": -1.1}
        )
    with pytest.raises(ValueError, match="positive and finite"):
        tuning_parameters.insert_parameters(
            {**tuning_rows[0], "place_bin_size_cm": 0.0}
        )
    with pytest.raises(ValueError, match="must be NULL"):
        tuning_parameters.insert_parameters(
            {**tuning_rows[0], "position_bin_count": 50}
        )
    with pytest.raises(ValueError, match="positive"):
        tuning_parameters.insert_parameters(
            {**tuning_rows[1], "position_bin_count": 0}
        )
    with pytest.raises(ValueError, match="65535"):
        tuning_parameters.insert_parameters(
            {**tuning_rows[1], "position_bin_count": 65_536}
        )
    with pytest.raises(ValueError, match="non-negative"):
        tuning_parameters.insert_parameters(
            {
                **tuning_rows[1],
                "gaussian_smoothing_sigma_bins": -0.1,
            }
        )
    with pytest.raises(ValueError, match="similarity_metric"):
        similarity_parameters.insert_parameters(
            {
                "tuning_similarity_param_name": "unsupported",
                "similarity_metric": "cosine",
            }
        )
    with pytest.raises(ValueError, match="exactly the declared fields"):
        similarity_parameters.insert_parameters(
            {
                **similarity_rows[0],
                "minimum_firing_rate_hz": 1.0,
            }
        )
    with pytest.raises(ValueError, match="positive"):
        swap_tuning_parameters.insert_parameters(
            {**swap_tuning_rows[0], "evaluation_bin_size_s": 0.0}
        )
    with pytest.raises(ValueError, match="non-negative"):
        swap_tuning_parameters.insert_parameters(
            {**swap_tuning_rows[0], "min_dark_firing_rate_hz": -0.1}
        )
    with pytest.raises(ValueError, match="threshold 2.0"):
        ripple_glm_parameters.insert_parameters(
            {
                **ripple_glm_rows[0],
                "expected_detector_zscore_threshold": 3.0,
            }
        )
    with pytest.raises(ValueError, match="speed-gated"):
        ripple_glm_parameters.insert_parameters(
            {**ripple_glm_rows[0], "require_speed_gated": False}
        )
    with pytest.raises(TypeError, match="database integer 0/1"):
        _validate_ripple_glm_parameter_row(
            {**ripple_glm_rows[0], "require_speed_gated": 2}
        )
    with pytest.raises(ValueError, match="fixed value 0.005"):
        xcorr_parameters.insert_parameters(
            {**xcorr_rows[0], "bin_size_s": 0.01}
        )
    with pytest.raises(ValueError, match="speed-gated"):
        xcorr_parameters.insert_parameters(
            {**xcorr_rows[0], "require_speed_gated": False}
        )
    with pytest.raises(TypeError, match="database integer 0/1"):
        _validate_ripple_cross_region_xcorr_parameter_row(
            {**xcorr_rows[0], "norm": 2}
        )

    assert _analysis_region("ca1") == "ca1"
    with pytest.raises(ValueError, match="canonical lowercase"):
        _analysis_region("CA1")


def test_path_progression_decoding_preset_and_definitions_are_exact() -> None:
    bundle, _, _ = _fake_bundle()
    parameters = bundle["path_progression_decoding_parameters"]

    rows = parameters.insert_presets()

    expected = dict(
        table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS
    )
    assert tuple(
        dict(row)
        for row in table_specs.PATH_PROGRESSION_DECODING_PARAMETER_PRESETS
    ) == (expected,)
    assert rows == [expected]
    assert parameters._insert_calls == [
        (expected, {"skip_duplicates": True})
    ]
    assert dict(table_specs.PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE) == {
        "version": 1,
        "cohort_policy": "target_and_cohort_intersection",
        "movement_operator": "greater_than_or_equal",
        "stability_aggregation": "at_least_one_trajectory",
        "stability_operator": "greater_than_or_equal",
        "null_stability_threshold": "disabled",
    }
    assert dict(table_specs.PATH_PROGRESSION_DECODING_OUTPUT_RULE) == {
        "version": 1,
        "coordinate_unit": "normalized_path_progression",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": "signed",
        "error_summary": "median_iqr",
        "min_bin_count": 5,
    }

    selection_definition = bundle[
        "path_progression_decoding_selection"
    ].definition
    assert selection_definition.count(
        "-> PathSpecificPlaceStability.proj("
    ) == 8
    assert selection_definition.count(
        "-> MovementFiringRate"
    ) == 2
    result = bundle["path_progression_decoding"]
    assert result.definition == table_specs.PATH_PROGRESSION_DECODING_DEFINITION
    assert "-> AnalysisNwbfile" in result.definition
    assert "filepath@analysis" not in result.definition
    assert result.Transfer.definition == (
        table_specs.PATH_PROGRESSION_DECODING_TRANSFER_DEFINITION
    )
    assert result.Transfer._nwb_table is bundle["analysis_nwbfile"]
    assert "artifact_origin" not in result.definition
    assert "legacy_artifact_provenance" not in result.definition
    assert not hasattr(result, "register_existing")


def test_legacy_tuning_registration_requires_original_position_and_movement() -> None:
    position = {
        "position_series_name": "head_position",
        "position_role": "head",
        "analysis_start_offset_samples": 10,
    }
    movement = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)

    _validate_legacy_tuning_curve_inputs(
        position_row=position,
        movement_parameters=movement,
    )

    for changed_position in (
        {**position, "position_series_name": "body_position"},
        {**position, "position_role": "body"},
        {**position, "analysis_start_offset_samples": 0},
    ):
        with pytest.raises(ValueError, match="cleaned DLC head position"):
            _validate_legacy_tuning_curve_inputs(
                position_row=changed_position,
                movement_parameters=movement,
            )
    with pytest.raises(ValueError, match="movement defaults"):
        _validate_legacy_tuning_curve_inputs(
            position_row=position,
            movement_parameters={**movement, "speed_threshold_cm_s": 5.0},
        )


def test_source_table_loaders_delegate_without_copying_arrays(monkeypatch) -> None:
    calls = []

    def load_catalog(table, key, **kwargs):
        calls.append((table, dict(key), kwargs))
        return kwargs["loader"].__name__

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_load_catalog_nwb_object",
        load_catalog,
    )
    bundle, _, _ = _fake_bundle()
    key = {"nwb_file_name": "L14.nwb", "epoch": "02_r1"}

    assert bundle["epoch_intervals"].load_intervals(key) == "load_interval_set"
    assert (
        bundle["position"].load_position(
            {**key, "position_series_name": "head_position"},
            apply_analysis_offset=False,
        )
        == "load_position"
    )
    assert bundle["wtrack_graph"].load_graph(
        {"nwb_file_name": "L14.nwb", "configuration_name": "full_w"}
    ) == "load_wtrack_graph"
    assert calls[1][2]["loader_kwargs"] == {"apply_analysis_offset": False}


def test_ripple_selection_uuid_captures_region_group_and_parameters() -> None:
    key = _ripple_selection_key()
    region_group_id = key["region_sorted_spikes_group_id"]
    changed_region_group_id = uuid.UUID(
        "62222222-2222-5222-8222-222222222222"
    )
    region_groups = _FakeKeyedRelation(
        "region_sorted_spikes_group_id",
        {
            region_group_id: {
                "region_sorted_spikes_group_id": region_group_id,
                "nwb_file_name": key["nwb_file_name"],
                "region_name": "ca1",
            },
            changed_region_group_id: {
                "region_sorted_spikes_group_id": changed_region_group_id,
                "nwb_file_name": key["nwb_file_name"],
                "region_name": "ca1",
            },
        },
    )
    parameter_values = dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    parameters = _FakeRelation(parameter_values)
    source = _FakeRelation({})

    first = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=parameters,
        region_sorted_spikes_group_table=region_groups,
    )
    repeated = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=parameters,
        region_sorted_spikes_group_table=region_groups,
    )
    changed_region_group = _ripple_modulation_selection_row(
        key={
            **key,
            "region_sorted_spikes_group_id": changed_region_group_id,
        },
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=parameters,
        region_sorted_spikes_group_table=region_groups,
    )
    changed_parameter_values = {**parameter_values, "bin_size_s": 0.025}
    changed_parameters = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=_FakeRelation(changed_parameter_values),
        region_sorted_spikes_group_table=region_groups,
    )

    assert isinstance(first["ripple_modulation_id"], uuid.UUID)
    assert first["ripple_modulation_id"].version == 5
    assert first["ripple_modulation_id"] == repeated["ripple_modulation_id"]
    assert first["ripple_modulation_id"] != changed_region_group[
        "ripple_modulation_id"
    ]
    assert first["ripple_modulation_id"] != changed_parameters["ripple_modulation_id"]
    assert first["region_sorted_spikes_group_id"] == region_group_id
    assert first["ripple_modulation_parameters_sha256"] == provenance_sha256(
        parameter_values
    )
    assert changed_parameters[
        "ripple_modulation_parameters_sha256"
    ] == provenance_sha256(changed_parameter_values)


def test_movement_selection_uuid_captures_region_group_and_parameters() -> None:
    key = _movement_selection_key()
    region_group_id = key["region_sorted_spikes_group_id"]
    changed_region_group_id = uuid.UUID(
        "62222222-2222-5222-8222-222222222222"
    )
    region_groups = _FakeKeyedRelation(
        "region_sorted_spikes_group_id",
        {
            region_group_id: {
                "region_sorted_spikes_group_id": region_group_id,
                "nwb_file_name": key["nwb_file_name"],
                "region_name": "ca1",
            },
            changed_region_group_id: {
                "region_sorted_spikes_group_id": changed_region_group_id,
                "nwb_file_name": key["nwb_file_name"],
                "region_name": "ca1",
            },
        },
    )
    position = _FakeRelation({"spatial_unit": "cm"})
    parameter_values = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    parameters = _FakeRelation(parameter_values)

    first = _movement_firing_rate_selection_row(
        key=key,
        position_table=position,
        parameters_table=parameters,
        region_sorted_spikes_group_table=region_groups,
    )
    repeated = _movement_firing_rate_selection_row(
        key=key,
        position_table=position,
        parameters_table=parameters,
        region_sorted_spikes_group_table=region_groups,
    )
    changed_region_group = _movement_firing_rate_selection_row(
        key={
            **key,
            "region_sorted_spikes_group_id": changed_region_group_id,
        },
        position_table=position,
        parameters_table=parameters,
        region_sorted_spikes_group_table=region_groups,
    )
    changed_parameter_values = {
        **parameter_values,
        "speed_threshold_cm_s": 5.0,
    }
    changed_parameters = _movement_firing_rate_selection_row(
        key=key,
        position_table=position,
        parameters_table=_FakeRelation(changed_parameter_values),
        region_sorted_spikes_group_table=region_groups,
    )

    assert isinstance(first["movement_firing_rate_id"], uuid.UUID)
    assert first["movement_firing_rate_id"].version == 5
    assert first["movement_firing_rate_id"] == repeated["movement_firing_rate_id"]
    assert first["movement_firing_rate_id"] != changed_region_group[
        "movement_firing_rate_id"
    ]
    assert first["movement_firing_rate_id"] != changed_parameters[
        "movement_firing_rate_id"
    ]
    assert first["region_sorted_spikes_group_id"] == region_group_id
    assert first["movement_parameters_sha256"] == provenance_sha256(
        parameter_values
    )
    assert changed_parameters["movement_parameters_sha256"] == provenance_sha256(
        changed_parameter_values
    )

    with pytest.raises(ValueError, match="centimeters"):
        _movement_firing_rate_selection_row(
            key=key,
            position_table=_FakeRelation({"spatial_unit": "m"}),
            parameters_table=parameters,
            region_sorted_spikes_group_table=region_groups,
        )


def test_tuning_curve_selection_uuid_captures_subset_and_parameters() -> None:
    key = _tuning_curve_selection_key()
    source = _FakeRelation({})
    epoch = _FakeRelation({"epoch_type": "run"})
    graph = _FakeRelation({"coordinate_unit": "cm"})
    movement_result = _FakeRelation(
        {"movement_firing_rate_id": key["movement_firing_rate_id"]}
    )
    movement_selection = _FakeRelation(
        {
            "movement_firing_rate_id": key["movement_firing_rate_id"],
            **_movement_selection_key(),
        }
    )
    parameter_values = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    parameters = _FakeRelation(parameter_values)

    row = _path_specific_place_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch,
        trajectory_intervals_table=source,
        wtrack_graph_table=graph,
        movement_firing_rate_table=movement_result,
        movement_firing_rate_selection_table=movement_selection,
        parameters_table=parameters,
    )
    repeated = _path_specific_place_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch,
        trajectory_intervals_table=source,
        wtrack_graph_table=graph,
        movement_firing_rate_table=movement_result,
        movement_firing_rate_selection_table=movement_selection,
        parameters_table=parameters,
    )
    changed_parameter_values = {**parameter_values, "place_bin_size_cm": 5.0}
    changed_parameters = _path_specific_place_tuning_curve_selection_row(
        key=key,
        epoch_intervals_table=epoch,
        trajectory_intervals_table=source,
        wtrack_graph_table=graph,
        movement_firing_rate_table=movement_result,
        movement_firing_rate_selection_table=movement_selection,
        parameters_table=_FakeRelation(changed_parameter_values),
    )
    even = _path_specific_place_tuning_curve_selection_row(
        key={**key, "trial_subset": "even"},
        epoch_intervals_table=epoch,
        trajectory_intervals_table=source,
        wtrack_graph_table=graph,
        movement_firing_rate_table=movement_result,
        movement_firing_rate_selection_table=movement_selection,
        parameters_table=parameters,
    )

    assert isinstance(row["path_specific_place_tuning_curve_id"], uuid.UUID)
    assert row["path_specific_place_tuning_curve_id"].version == 5
    assert row["path_specific_place_tuning_curve_id"] == repeated[
        "path_specific_place_tuning_curve_id"
    ]
    assert row["trajectory_type"] == "center_to_left"
    assert row["configuration_name"] == "center_to_left"
    assert row["movement_firing_rate_id"] == key["movement_firing_rate_id"]
    assert row["trial_subset"] == "odd"
    assert "position_series_name" not in row
    assert "sorting_group_members" not in row
    assert row["path_specific_place_tuning_curve_id"] != changed_parameters[
        "path_specific_place_tuning_curve_id"
    ]
    assert row["path_specific_place_tuning_curve_id"] != even[
        "path_specific_place_tuning_curve_id"
    ]
    assert row["tuning_curve_parameters_sha256"] == provenance_sha256(
        parameter_values
    )
    assert changed_parameters["tuning_curve_parameters_sha256"] == (
        provenance_sha256(changed_parameter_values)
    )

    with pytest.raises(ValueError, match="configuration_name"):
        _path_specific_place_tuning_curve_selection_row(
            key={**key, "configuration_name": "center_to_right"},
            epoch_intervals_table=epoch,
            trajectory_intervals_table=source,
            wtrack_graph_table=graph,
            movement_firing_rate_table=movement_result,
            movement_firing_rate_selection_table=movement_selection,
            parameters_table=parameters,
        )

    with pytest.raises(ValueError, match="same epoch"):
        _path_specific_place_tuning_curve_selection_row(
            key={**key, "epoch": "04_r2"},
            epoch_intervals_table=epoch,
            trajectory_intervals_table=source,
            wtrack_graph_table=graph,
            movement_firing_rate_table=movement_result,
            movement_firing_rate_selection_table=movement_selection,
            parameters_table=parameters,
        )


def test_dpp_selection_freezes_fixed_pair_subset_and_parameters() -> None:
    key = _dpp_tuning_curve_selection_key()
    epoch = _FakeRelation({"epoch_type": "run"})
    trajectories = _RecordingRelation({})
    graphs = _RecordingRelation({"coordinate_unit": "cm"})
    movement_result = _FakeRelation(
        {"movement_firing_rate_id": key["movement_firing_rate_id"]}
    )
    movement_selection = _FakeRelation(
        {
            "movement_firing_rate_id": key["movement_firing_rate_id"],
            **_movement_selection_key(),
        }
    )
    parameter_values = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)

    def build(selection_key, parameters=parameter_values):
        return _dpp_tuning_curve_selection_row(
            key=selection_key,
            epoch_intervals_table=epoch,
            trajectory_intervals_table=trajectories,
            wtrack_graph_table=graphs,
            movement_firing_rate_table=movement_result,
            movement_firing_rate_selection_table=movement_selection,
            parameters_table=_FakeRelation(parameters),
        )

    row = build(key)
    repeated = build(key)
    even = build({**key, "trial_subset": "even"})
    right = build({**key, "turn_type": "right"})
    changed_parameters = build(
        key,
        {**parameter_values, "place_bin_size_cm": 5.0},
    )

    assert isinstance(row["dpp_tuning_curve_id"], uuid.UUID)
    assert row["dpp_tuning_curve_id"].version == 5
    assert row["dpp_tuning_curve_id"] == repeated["dpp_tuning_curve_id"]
    assert row["dpp_tuning_curve_id"] != even["dpp_tuning_curve_id"]
    assert row["dpp_tuning_curve_id"] != right["dpp_tuning_curve_id"]
    assert row["dpp_tuning_curve_id"] != changed_parameters[
        "dpp_tuning_curve_id"
    ]
    assert row["outbound_trajectory_type"] == "center_to_left"
    assert row["inbound_trajectory_type"] == "right_to_center"
    assert row["outbound_configuration_name"] == "center_to_left"
    assert row["inbound_configuration_name"] == "right_to_center"
    assert row["tuning_curve_parameters_sha256"] == provenance_sha256(
        parameter_values
    )
    assert {
        tuple(sorted(key.items()))
        for key in trajectories.keys
    } >= {
        tuple(
            sorted(
                {
                    "nwb_file_name": key["nwb_file_name"],
                    "epoch": key["epoch"],
                    "trajectory_type": trajectory_type,
                }.items()
            )
        )
        for trajectory_type in ("center_to_left", "right_to_center")
    }
    assert {
        tuple(sorted(source_key.items()))
        for source_key in graphs.keys
    } >= {
        tuple(
            sorted(
                {
                    "nwb_file_name": key["nwb_file_name"],
                    "configuration_name": configuration_name,
                }.items()
            )
        )
        for configuration_name in ("center_to_left", "right_to_center")
    }

    with pytest.raises(ValueError, match="fixed by turn_type"):
        build({**key, "inbound_trajectory_type": "left_to_center"})
    with pytest.raises(ValueError, match="same epoch"):
        build({**key, "epoch": "04_r2"})
    with pytest.raises(ValueError, match="run epoch"):
        _dpp_tuning_curve_selection_row(
            key=key,
            epoch_intervals_table=_FakeRelation({"epoch_type": "sleep"}),
            trajectory_intervals_table=_FakeRelation({}),
            wtrack_graph_table=graphs,
            movement_firing_rate_table=movement_result,
            movement_firing_rate_selection_table=movement_selection,
            parameters_table=_FakeRelation(parameter_values),
        )
    with pytest.raises(ValueError, match="centimeters"):
        _dpp_tuning_curve_selection_row(
            key=key,
            epoch_intervals_table=epoch,
            trajectory_intervals_table=_FakeRelation({}),
            wtrack_graph_table=_FakeRelation({"coordinate_unit": "m"}),
            movement_firing_rate_table=movement_result,
            movement_firing_rate_selection_table=movement_selection,
            parameters_table=_FakeRelation(parameter_values),
        )


def test_stability_selection_identifies_one_matching_odd_even_pair() -> None:
    key = _stability_selection_key()
    common = {
        key_name: value
        for key_name, value in _tuning_curve_selection_key().items()
        if key_name != "trial_subset"
    }
    parameter_hash = provenance_sha256(
        dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    )
    curve_selections = {
        key["odd_path_specific_place_tuning_curve_id"]: {
            "path_specific_place_tuning_curve_id": key[
                "odd_path_specific_place_tuning_curve_id"
            ],
            **common,
            "trial_subset": "odd",
            "tuning_curve_parameters_sha256": parameter_hash,
        },
        key["even_path_specific_place_tuning_curve_id"]: {
            "path_specific_place_tuning_curve_id": key[
                "even_path_specific_place_tuning_curve_id"
            ],
            **common,
            "trial_subset": "even",
            "tuning_curve_parameters_sha256": parameter_hash,
        },
    }
    curve_results = {
        curve_id: {
            "path_specific_place_tuning_curve_id": curve_id,
            "selected_units_sha256": "a" * 64,
        }
        for curve_id in curve_selections
    }
    result_table = _FakeKeyedRelation(
        "path_specific_place_tuning_curve_id",
        curve_results,
    )
    selection_table = _FakeKeyedRelation(
        "path_specific_place_tuning_curve_id",
        curve_selections,
    )

    row = _stability_selection_row(
        key=key,
        tuning_curve_table=result_table,
        tuning_curve_selection_table=selection_table,
    )
    repeated = _stability_selection_row(
        key=key,
        tuning_curve_table=result_table,
        tuning_curve_selection_table=selection_table,
    )

    assert isinstance(row["path_specific_place_stability_id"], uuid.UUID)
    assert row["path_specific_place_stability_id"].version == 5
    assert row == repeated
    assert row["odd_path_specific_place_tuning_curve_id"] == key[
        "odd_path_specific_place_tuning_curve_id"
    ]
    assert row["even_path_specific_place_tuning_curve_id"] == key[
        "even_path_specific_place_tuning_curve_id"
    ]
    assert set(row) == {
        "path_specific_place_stability_id",
        "odd_path_specific_place_tuning_curve_id",
        "even_path_specific_place_tuning_curve_id",
    }

    wrong_subset = {
        curve_id: dict(selection)
        for curve_id, selection in curve_selections.items()
    }
    wrong_subset[key["even_path_specific_place_tuning_curve_id"]][
        "trial_subset"
    ] = "all"
    with pytest.raises(ValueError, match="matching odd and even"):
        _stability_selection_row(
            key=key,
            tuning_curve_table=result_table,
            tuning_curve_selection_table=_FakeKeyedRelation(
                "path_specific_place_tuning_curve_id",
                wrong_subset,
            ),
        )

    mismatched_results = {
        curve_id: dict(result)
        for curve_id, result in curve_results.items()
    }
    mismatched_results[key["even_path_specific_place_tuning_curve_id"]][
        "selected_units_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="same selected units"):
        _stability_selection_row(
            key=key,
            tuning_curve_table=_FakeKeyedRelation(
                "path_specific_place_tuning_curve_id",
                mismatched_results,
            ),
            tuning_curve_selection_table=selection_table,
        )


def test_tuning_similarity_selection_identifies_matching_four_path_rows() -> None:
    key = _tuning_similarity_selection_key()
    parameter_values = dict(
        table_specs.CORRELATION_TUNING_SIMILARITY_PARAMETERS
    )
    tuning_parameter_hash = provenance_sha256(
        dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    )
    curve_id_fields = {
        "center_to_left": "center_to_left_tuning_curve_id",
        "center_to_right": "center_to_right_tuning_curve_id",
        "left_to_center": "left_to_center_tuning_curve_id",
        "right_to_center": "right_to_center_tuning_curve_id",
    }
    common = {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "movement_firing_rate_id": uuid.UUID(
            "33333333-3333-5333-8333-333333333333"
        ),
        "tuning_curve_param_name": "legacy_4cm_unsmoothed",
        "trial_subset": "all",
        "tuning_curve_parameters_sha256": tuning_parameter_hash,
    }
    curve_selections = {
        key[field_name]: {
            "path_specific_place_tuning_curve_id": key[field_name],
            **common,
            "trajectory_type": trajectory_type,
            "configuration_name": trajectory_type,
        }
        for trajectory_type, field_name in curve_id_fields.items()
    }
    curve_results = {
        curve_id: {
            "path_specific_place_tuning_curve_id": curve_id,
            "selected_units_sha256": "a" * 64,
            "n_units": 5,
            "n_position_bins": 25,
        }
        for curve_id in curve_selections
    }

    def build(
        selection_key=key,
        *,
        selections=curve_selections,
        results=curve_results,
        parameters=parameter_values,
    ):
        return _tuning_similarity_selection_row(
            key=selection_key,
            tuning_curve_table=_FakeKeyedRelation(
                "path_specific_place_tuning_curve_id",
                results,
            ),
            tuning_curve_selection_table=_FakeKeyedRelation(
                "path_specific_place_tuning_curve_id",
                selections,
            ),
            parameters_table=_FakeRelation(parameters),
        )

    row = build()
    repeated = build()
    shape_parameters = dict(
        table_specs.SHAPE_OVERLAP_TUNING_SIMILARITY_PARAMETERS
    )
    shape = build(
        {**key, "tuning_similarity_param_name": "shape_overlap"},
        parameters=shape_parameters,
    )

    assert isinstance(
        row["path_specific_place_tuning_similarity_id"],
        uuid.UUID,
    )
    assert row["path_specific_place_tuning_similarity_id"].version == 5
    assert row == repeated
    assert row["path_specific_place_tuning_similarity_id"] != shape[
        "path_specific_place_tuning_similarity_id"
    ]
    for field_name in curve_id_fields.values():
        assert row[field_name] == key[field_name]
    assert row["tuning_similarity_param_name"] == "correlation"
    assert row["tuning_similarity_parameters_sha256"] == provenance_sha256(
        parameter_values
    )
    assert set(row) == {
        "path_specific_place_tuning_similarity_id",
        *curve_id_fields.values(),
        "tuning_similarity_param_name",
        "tuning_similarity_parameters_sha256",
    }

    wrong_subset = {
        curve_id: dict(selection)
        for curve_id, selection in curve_selections.items()
    }
    wrong_subset[key["right_to_center_tuning_curve_id"]][
        "trial_subset"
    ] = "odd"
    with pytest.raises(ValueError, match="four all-trial"):
        build(selections=wrong_subset)

    wrong_alias = {
        curve_id: dict(selection)
        for curve_id, selection in curve_selections.items()
    }
    wrong_alias[key["left_to_center_tuning_curve_id"]][
        "trajectory_type"
    ] = "right_to_center"
    with pytest.raises(ValueError, match="aliases must match"):
        build(selections=wrong_alias)

    mismatched_results = {
        curve_id: dict(result)
        for curve_id, result in curve_results.items()
    }
    mismatched_results[key["center_to_right_tuning_curve_id"]][
        "selected_units_sha256"
    ] = "b" * 64
    with pytest.raises(ValueError, match="same selected_units_sha256"):
        build(results=mismatched_results)


def test_dpp_encoding_selection_uuid_freezes_exact_upstream_rows() -> None:
    inputs = _dpp_encoding_selection_inputs()

    row = _build_dpp_encoding_selection(inputs)
    repeated = _build_dpp_encoding_selection(
        _dpp_encoding_selection_inputs()
    )

    assert isinstance(row["dpp_encoding_id"], uuid.UUID)
    assert row["dpp_encoding_id"].version == 5
    assert row == repeated
    assert row["nwb_file_name"] == "L1420240102_.nwb"
    assert row["epoch"] == "02_r1"
    assert row["region_sorted_spikes_group_id"] == inputs["key"][
        "region_sorted_spikes_group_id"
    ]
    assert row["movement_firing_rate_id"] == inputs["key"][
        "movement_firing_rate_id"
    ]
    for trajectory_type in _DPP_ENCODING_TRAJECTORIES:
        assert row[f"{trajectory_type}_trajectory_type"] == trajectory_type
        assert row[f"{trajectory_type}_configuration_name"] == trajectory_type
        assert row[f"{trajectory_type}_stability_id"] == inputs["key"][
            f"{trajectory_type}_stability_id"
        ]
    assert row["full_w_configuration_name"] == "full_w"
    assert row["dpp_encoding_parameters_sha256"] == (
        provenance_sha256(inputs["parameters"])
    )

    alternate = _dpp_encoding_selection_inputs()
    alternate_name = "same_values_alternate_name"
    alternate["parameters"][
        "dpp_encoding_param_name"
    ] = alternate_name
    alternate["key"]["dpp_encoding_param_name"] = alternate_name
    alternate_row = _build_dpp_encoding_selection(alternate)
    assert row["dpp_encoding_id"] != alternate_row[
        "dpp_encoding_id"
    ]


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("session", "same nwb_file_name"),
        ("epoch", "slot: epoch"),
        ("supplied_session", "MovementFiringRate: nwb_file_name"),
        ("supplied_epoch", "MovementFiringRate: epoch"),
        ("trajectory_alias", "center_to_left_trajectory_type must equal"),
        ("graph_alias", "center_to_left_configuration_name must equal"),
        ("stability_slot", "stability input.*trajectory_type"),
        ("stability_unit_count", "same unit count"),
        ("stability_status", "stability inputs must be valid"),
        ("legacy_curve", "stability input.*tuning_curve_param_name"),
        (
            "legacy_curve_hash",
            "stability input.*tuning_curve_parameters_sha256",
        ),
        ("trial_subset", "stability input.*trial_subset"),
    ],
)
def test_dpp_encoding_selection_rejects_mismatched_upstreams(
    mismatch: str,
    message: str,
) -> None:
    inputs = _dpp_encoding_selection_inputs()
    center_left_stability_id = inputs["key"][
        "center_to_left_stability_id"
    ]
    center_left_stability = inputs["stability_selections"][
        center_left_stability_id
    ]
    center_left_odd_id = center_left_stability[
        "odd_path_specific_place_tuning_curve_id"
    ]

    if mismatch == "session":
        inputs["region_row"]["nwb_file_name"] = "L1520240102_.nwb"
    elif mismatch == "epoch":
        inputs["curve_selections"][center_left_odd_id]["epoch"] = "04_r2"
    elif mismatch == "supplied_session":
        inputs["key"]["nwb_file_name"] = "L1520240102_.nwb"
    elif mismatch == "supplied_epoch":
        inputs["key"]["epoch"] = "04_r2"
    elif mismatch == "trajectory_alias":
        inputs["key"]["center_to_left_trajectory_type"] = "center_to_right"
    elif mismatch == "graph_alias":
        inputs["key"]["center_to_left_configuration_name"] = "full_w"
    elif mismatch == "stability_slot":
        inputs["key"]["right_to_center_stability_id"] = (
            center_left_stability_id
        )
    elif mismatch == "stability_unit_count":
        inputs["stability_results"][center_left_stability_id]["n_units"] = 4
    elif mismatch == "stability_status":
        inputs["stability_results"][center_left_stability_id][
            "analysis_status"
        ] = "no_movement"
    elif mismatch == "legacy_curve":
        inputs["curve_selections"][center_left_odd_id][
            "tuning_curve_param_name"
        ] = "figure1d_50bin_sigma1p5"
    elif mismatch == "legacy_curve_hash":
        inputs["curve_selections"][center_left_odd_id][
            "tuning_curve_parameters_sha256"
        ] = "d" * 64
    elif mismatch == "trial_subset":
        inputs["curve_selections"][center_left_odd_id]["trial_subset"] = "all"
    else:  # pragma: no cover - protects future parametrization edits
        raise AssertionError(f"Unhandled mismatch {mismatch!r}.")

    with pytest.raises(ValueError, match=message):
        _build_dpp_encoding_selection(inputs)


@pytest.mark.parametrize(
    ("source", "missing_name"),
    [
        ("trajectory_rows", "right_to_center"),
        ("graph_rows", "full_w"),
    ],
)
def test_dpp_encoding_selection_requires_all_trajectory_and_graph_rows(
    source: str,
    missing_name: str,
) -> None:
    inputs = _dpp_encoding_selection_inputs()
    identity_field = (
        "trajectory_type"
        if source == "trajectory_rows"
        else "configuration_name"
    )
    inputs[source] = [
        row
        for row in inputs[source]
        if row[identity_field] != missing_name
    ]

    with pytest.raises(LookupError, match=missing_name):
        _build_dpp_encoding_selection(inputs)


def test_path_progression_decoding_selection_uuid_freezes_shared_cohort() -> None:
    from v1ca1.spyglass.path_progression_decoding import TRANSFER_SPEC_SHA256
    from v1ca1.spyglass.selection import selection_uuid

    inputs = _path_progression_decoding_selection_inputs()
    row = _build_path_progression_decoding_selection(inputs)
    repeated = _build_path_progression_decoding_selection(
        _path_progression_decoding_selection_inputs()
    )

    assert isinstance(
        row["path_progression_decoding_id"],
        uuid.UUID,
    )
    assert row["path_progression_decoding_id"].version == 5
    assert row == repeated
    assert row["nwb_file_name"] == "L1420240102_.nwb"
    assert row["epoch"] == "02_r1"
    assert row["cohort_epoch"] == "08_r4"
    assert row["movement_firing_rate_id"] == inputs["key"][
        "movement_firing_rate_id"
    ]
    assert row["cohort_movement_firing_rate_id"] == inputs["key"][
        "cohort_movement_firing_rate_id"
    ]
    for prefix in ("", "cohort_"):
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES:
            assert row[f"{prefix}{trajectory_type}_stability_id"] == (
                inputs["key"][
                    f"{prefix}{trajectory_type}_stability_id"
                ]
            )
    assert row["path_progression_decoding_parameters_sha256"] == (
        provenance_sha256(inputs["parameters"])
    )
    assert row["eligibility_rule_sha256"] == provenance_sha256(
        dict(table_specs.PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE)
    )
    assert row["transfer_spec_sha256"] == TRANSFER_SPEC_SHA256
    assert row["decoding_output_rule_sha256"] == provenance_sha256(
        dict(table_specs.PATH_PROGRESSION_DECODING_OUTPUT_RULE)
    )
    natural_key = {
        name: value
        for name, value in row.items()
        if name != "path_progression_decoding_id"
    }
    assert row["path_progression_decoding_id"] == selection_uuid(
        "PathProgressionDecoding",
        natural_key,
    )


def test_path_progression_decoding_artifact_link_checks_session_and_all_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from v1ca1.spyglass import path_progression_decoding as decoding
    from v1ca1.spyglass.selection import unit_identity_sha256

    result_id = uuid.UUID("74444444-4444-5444-8444-444444444444")
    parameters = dict(
        table_specs.MANUSCRIPT_PATH_PROGRESSION_DECODING_PARAMETERS
    )
    selection = {
        "path_progression_decoding_id": result_id,
        "epoch": "02_r1",
        "cohort_epoch": "08_r4",
        "path_progression_decoding_param_name": parameters[
            "path_progression_decoding_param_name"
        ],
        "path_progression_decoding_parameters_sha256": provenance_sha256(
            parameters
        ),
        "eligibility_rule_sha256": provenance_sha256(
            dict(table_specs.PATH_PROGRESSION_DECODING_ELIGIBILITY_RULE)
        ),
        "transfer_spec_sha256": decoding.TRANSFER_SPEC_SHA256,
        "decoding_output_rule_sha256": provenance_sha256(
            dict(table_specs.PATH_PROGRESSION_DECODING_OUTPUT_RULE)
        ),
    }
    identities = pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge-a", "merge-b"],
            "unit_id": ["1", "2"],
        }
    )
    input_digest = unit_identity_sha256(identities.to_dict("records"))
    region = {
        "region_name": "ca1",
        "n_units": 2,
        "selected_units_sha256": input_digest,
    }
    artifact_dir = (
        tmp_path
        / "L14"
        / "20240102"
        / decoding.ARTIFACT_DIRNAME
        / "02_r1"
        / "ca1"
        / str(result_id)
    )
    bundle = {"path": artifact_dir, "unit_eligibility": identities}
    summary = {
        "path_progression_decoding_id": str(result_id),
        "animal_name": "L14",
        "date": "20240102",
        "region": "ca1",
        "epoch": "02_r1",
        "cohort_epoch": "08_r4",
        "parameter_name": parameters[
            "path_progression_decoding_param_name"
        ],
        "parameter_sha256": selection[
            "path_progression_decoding_parameters_sha256"
        ],
        "eligibility_rule_sha256": selection[
            "eligibility_rule_sha256"
        ],
        "transfer_spec_sha256": selection["transfer_spec_sha256"],
        "decoding_output_rule_sha256": selection[
            "decoding_output_rule_sha256"
        ],
        "n_units_input": 2,
        "n_units_eligible": 1,
        "n_transfer_pairs_expected": 16,
        "n_transfer_pairs_valid": 16,
        "n_decoded_samples": 100,
        "analysis_status": "valid",
        "eligible_units_sha256": "e" * 64,
    }
    result = {
        "artifact_manifest_path": str(artifact_dir / "manifest.parquet"),
        "decoding_summary_path": str(
            artifact_dir / "decoding_summary.parquet"
        ),
        "unit_eligibility_path": str(
            artifact_dir / "unit_eligibility.parquet"
        ),
        **{
            name: summary[name]
            for name in (
                "n_units_input",
                "n_units_eligible",
                "n_transfer_pairs_expected",
                "n_transfer_pairs_valid",
                "n_decoded_samples",
                "analysis_status",
                "eligible_units_sha256",
            )
        },
    }
    monkeypatch.setattr(
        decoding,
        "summarize_decoding_artifact_bundle",
        lambda loaded: dict(summary),
    )

    _validate_path_progression_decoding_artifact_link(
        bundle=bundle,
        result_row=result,
        selection_row=selection,
        parameters_row=parameters,
        region_row=region,
        animal_name="L14",
        date="20240102",
    )

    wrong_session_dir = (
        tmp_path
        / "L15"
        / "20240103"
        / decoding.ARTIFACT_DIRNAME
        / "02_r1"
        / "ca1"
        / str(result_id)
    )
    with pytest.raises(
        ValueError, match="canonical session/epoch/region/UUID layout"
    ):
        _validate_path_progression_decoding_artifact_link(
            bundle={**bundle, "path": wrong_session_dir},
            result_row={
                **result,
                "artifact_manifest_path": str(
                    wrong_session_dir / "manifest.parquet"
                ),
                "decoding_summary_path": str(
                    wrong_session_dir / "decoding_summary.parquet"
                ),
                "unit_eligibility_path": str(
                    wrong_session_dir / "unit_eligibility.parquet"
                ),
            },
            selection_row=selection,
            parameters_row=parameters,
            region_row=region,
            animal_name="L14",
            date="20240102",
        )

    changed_identities = identities.copy()
    changed_identities.loc[1, "unit_id"] = "3"
    with pytest.raises(ValueError, match="input identities disagree"):
        _validate_path_progression_decoding_artifact_link(
            bundle={**bundle, "unit_eligibility": changed_identities},
            result_row=result,
            selection_row=selection,
            parameters_row=parameters,
            region_row=region,
            animal_name="L14",
            date="20240102",
        )


def test_path_progression_decoding_selection_accepts_empty_regional_source() -> None:
    from v1ca1.spyglass.selection import unit_identity_sha256

    inputs = _path_progression_decoding_selection_inputs()
    empty_digest = unit_identity_sha256([])
    inputs["region_row"].update(
        n_units=0,
        selected_units_sha256=empty_digest,
    )
    for result in inputs["movement_results"].values():
        result.update(
            n_units=0,
            analysis_status="no_units",
            selected_units_sha256=empty_digest,
        )
    for result in inputs["stability_results"].values():
        result.update(
            n_units=0,
            analysis_status="no_units",
            selected_units_sha256=empty_digest,
        )

    row = _build_path_progression_decoding_selection(inputs)

    assert row["nwb_file_name"] == "L1420240102_.nwb"
    assert row["epoch"] == "02_r1"


def test_empty_stability_artifact_can_support_empty_decoding_source(
) -> None:
    from v1ca1.spyglass.selection import unit_identity_sha256
    from v1ca1.spyglass.stability import (
        NWB_ARTIFACT_SCHEMA_VERSION,
        empty_stability_table,
        stability_table_sha256,
    )
    from v1ca1.spyglass.tables import _load_dpp_stability_artifact

    stability_id = uuid.UUID("75555555-5555-5555-8555-555555555555")
    empty = empty_stability_table()

    class StabilityFetch:
        def __and__(self, key):
            assert key == {"path_specific_place_stability_id": stability_id}
            return self

        def fetch_nwb(self):
            return [{"stability": empty.copy()}]

    table = _load_dpp_stability_artifact(
        result_row={
            "path_specific_place_stability_id": stability_id,
            "stability_sha256": stability_table_sha256(empty),
            "artifact_schema_version": NWB_ARTIFACT_SCHEMA_VERSION,
            "n_units": 0,
            "n_valid_units": 0,
            "analysis_status": "no_units",
            "selected_units_sha256": unit_identity_sha256([]),
        },
        stability_table=StabilityFetch(),
        trajectory_type="center_to_left",
        animal_name="L14",
        date="20240102",
        region="ca1",
        epoch="02_r1",
    )

    assert table.empty


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("nwb_file_name", "L1520240102_.nwb", "nwb_file_name"),
        ("epoch", "04_r2", "epoch"),
        ("cohort_epoch", "06_r3", "cohort_epoch"),
        (
            "center_to_left_trajectory_type",
            "center_to_right",
            "center_to_left_trajectory_type must equal",
        ),
        (
            "center_to_left_configuration_name",
            "right_to_center",
            "graph aliases must match",
        ),
    ],
)
def test_path_progression_decoding_selection_rejects_key_mismatch(
    field_name: str,
    bad_value: str,
    message: str,
) -> None:
    inputs = _path_progression_decoding_selection_inputs()
    inputs["key"][field_name] = bad_value

    with pytest.raises(ValueError, match=message):
        _build_path_progression_decoding_selection(inputs)


def test_dpp_encoding_artifact_link_checks_summary_and_uuid() -> None:
    from v1ca1.spyglass.dpp_encoding import (
        empty_dpp_encoding_table,
        summarize_dpp_encoding_table,
    )

    comparison_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        "v1ca1-test-dpp-artifact-link",
    )
    table = empty_dpp_encoding_table()
    result_row = {
        **summarize_dpp_encoding_table(table),
        "n_units_input": 5,
    }
    selection_row = {
        "dpp_encoding_id": comparison_id,
        "epoch": "02_r1",
        "dpp_encoding_parameters_sha256": provenance_sha256(
            dict(table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS)
        ),
    }

    _validate_dpp_encoding_artifact_link(
        table=table,
        result_row=result_row,
        selection_row=selection_row,
        parameters_row=dict(
            table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS
        ),
        region_row={"region_name": "ca1", "n_units": 5},
        animal_name="L14",
        date="20240102",
    )

    with pytest.raises(ValueError, match="n_units_eligible"):
        _validate_dpp_encoding_artifact_link(
            table=table,
            result_row={**result_row, "n_units_eligible": 1},
            selection_row=selection_row,
            parameters_row=dict(
                table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS
            ),
            region_row={"region_name": "ca1", "n_units": 5},
            animal_name="L14",
            date="20240102",
        )
def test_similarity_input_loader_rejects_stale_tuning_parameters(
    monkeypatch,
) -> None:
    from v1ca1.spyglass import path_specific_place

    curve_id_fields = {
        "center_to_left": "center_to_left_tuning_curve_id",
        "center_to_right": "center_to_right_tuning_curve_id",
        "left_to_center": "left_to_center_tuning_curve_id",
        "right_to_center": "right_to_center_tuning_curve_id",
    }
    curve_ids = {
        trajectory_type: uuid.uuid4()
        for trajectory_type in curve_id_fields
    }
    key = {
        "path_specific_place_tuning_similarity_id": uuid.uuid4(),
        **{
            field_name: curve_ids[trajectory_type]
            for trajectory_type, field_name in curve_id_fields.items()
        },
        "tuning_similarity_param_name": "correlation",
        "tuning_similarity_parameters_sha256": provenance_sha256(
            dict(table_specs.CORRELATION_TUNING_SIMILARITY_PARAMETERS)
        ),
    }
    curve_results = {
        curve_id: {
            "path_specific_place_tuning_curve_id": curve_id,
            "tuning_curve_path": f"/unused/{trajectory_type}.nc",
        }
        for trajectory_type, curve_id in curve_ids.items()
    }
    curve_selections = {
        curve_id: {
            "path_specific_place_tuning_curve_id": curve_id,
            "tuning_curve_param_name": "legacy_4cm_unsmoothed",
            "tuning_curve_parameters_sha256": "0" * 64,
        }
        for curve_id in curve_ids.values()
    }
    monkeypatch.setitem(
        _load_tuning_similarity_inputs.__globals__,
        "_tuning_similarity_selection_row",
        lambda **kwargs: dict(key),
    )
    monkeypatch.setitem(
        _load_tuning_similarity_inputs.__globals__,
        "_load_path_specific_place_tuning_curve_result",
        lambda **kwargs: SimpleNamespace(attrs={}),
    )

    with pytest.raises(ValueError, match="parameters changed after selection"):
        _load_tuning_similarity_inputs(
            key=key,
            parameters_table=_FakeRelation(
                dict(table_specs.CORRELATION_TUNING_SIMILARITY_PARAMETERS)
            ),
            tuning_curve_parameters_table=_FakeRelation(
                dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
            ),
            tuning_curve_table=_FakeKeyedRelation(
                "path_specific_place_tuning_curve_id",
                curve_results,
            ),
            tuning_curve_selection_table=_FakeKeyedRelation(
                "path_specific_place_tuning_curve_id",
                curve_selections,
            ),
            movement_firing_rate_table=object(),
            movement_firing_rate_selection_table=object(),
            movement_parameters_table=object(),
            region_sorted_spikes_group_table=object(),
            session_table=object(),
        )


def test_compute_helpers_reject_parameters_changed_after_selection() -> None:
    ripple_parameters = dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    ripple_selection = {
        **_ripple_selection_key(),
        "ripple_modulation_id": uuid.uuid4(),
        "ripple_modulation_parameters_sha256": provenance_sha256(
            ripple_parameters
        ),
    }
    with pytest.raises(ValueError, match="parameters changed after selection"):
        _make_ripple_modulation_row(
            key=ripple_selection,
            parameters_table=_FakeRelation(
                {**ripple_parameters, "bin_size_s": 0.025}
            ),
            ripples_table=object(),
            epoch_intervals_table=object(),
            session_table=object(),
            region_sorted_spikes_group_table=object(),
            nwbfile_table=object(),
            artifact_root=None,
        )

    movement_parameters = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    movement_selection = {
        **_movement_selection_key(),
        "movement_firing_rate_id": uuid.uuid4(),
        "movement_parameters_sha256": provenance_sha256(movement_parameters),
    }
    with pytest.raises(ValueError, match="parameters changed after selection"):
        _make_movement_firing_rate_row(
            key=movement_selection,
            parameters_table=_FakeRelation(
                {**movement_parameters, "speed_threshold_cm_s": 5.0}
            ),
            epoch_intervals_table=object(),
            position_table=object(),
            session_table=object(),
            region_sorted_spikes_group_table=object(),
            nwbfile_table=object(),
            artifact_root=None,
        )

    tuning_parameters = dict(table_specs.LEGACY_TUNING_CURVE_PARAMETERS)
    tuning_selection = {
        **_tuning_curve_selection_key(),
        "path_specific_place_tuning_curve_id": uuid.uuid4(),
        "tuning_curve_parameters_sha256": provenance_sha256(
            tuning_parameters
        ),
    }
    with pytest.raises(ValueError, match="parameters changed after selection"):
        _make_path_specific_place_tuning_curve_row(
            key=tuning_selection,
            parameters_table=_FakeRelation(
                {**tuning_parameters, "place_bin_size_cm": 5.0}
            ),
            epoch_intervals_table=object(),
            trajectory_intervals_table=object(),
            position_table=object(),
            wtrack_graph_table=object(),
            movement_firing_rate_table=object(),
            movement_firing_rate_selection_table=object(),
            movement_parameters_table=object(),
            session_table=object(),
            region_sorted_spikes_group_table=object(),
            nwbfile_table=object(),
            artifact_root=None,
        )

    dpp_selection = {
        **_dpp_tuning_curve_selection_key(),
        "dpp_tuning_curve_id": uuid.uuid4(),
        "tuning_curve_parameters_sha256": provenance_sha256(
            tuning_parameters
        ),
    }
    with pytest.raises(ValueError, match="parameters changed after selection"):
        _make_dpp_tuning_curve_row(
            key=dpp_selection,
            parameters_table=_FakeRelation(
                {**tuning_parameters, "place_bin_size_cm": 5.0}
            ),
            epoch_intervals_table=object(),
            trajectory_intervals_table=object(),
            position_table=object(),
            wtrack_graph_table=object(),
            movement_firing_rate_table=object(),
            movement_firing_rate_selection_table=object(),
            movement_parameters_table=object(),
            session_table=object(),
            region_sorted_spikes_group_table=object(),
            nwbfile_table=object(),
            artifact_root=None,
        )


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid RippleModulation NWB"]],
)
def test_ripple_modulation_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
    monkeypatch,
    validation_errors: list[str] | None,
) -> None:
    """Both ripple tables are validated before one NWB is registered."""
    import pynwb

    from v1ca1.spyglass import ripple_modulation

    summary, peri = _valid_ripple_modulation_tables()
    analysis_path = tmp_path / "ripple-modulation-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate
    validation_calls = []

    def validate_before_registration(*, path):
        validation_calls.append(str(path))
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    kwargs = {
        "nwb_file_name": "L1420240102_.nwb",
        "summary": summary,
        "peri_ripple_firing_rate": peri,
        "analysis_nwbfile_table": analysis_table,
    }
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_ripple_modulation_nwb(**kwargs)
        assert validation_calls == [str(analysis_path)]
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_ripple_modulation_nwb(**kwargs)

    assert validation_calls == [str(analysis_path)]
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == (
        ripple_modulation.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["ripple_modulation_summary_sha256"] == (
        ripple_modulation.ripple_modulation_summary_sha256(summary)
    )
    assert row["peri_ripple_firing_rate_sha256"] == (
        ripple_modulation.peri_ripple_firing_rate_sha256(peri)
    )
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_ripple_modulation_result_loader_uses_fetch_nwb_and_checks_hash() -> None:
    """Live ripple readers resolve and verify both tables through fetch_nwb()."""
    from v1ca1.spyglass import ripple_modulation

    summary, peri = _valid_ripple_modulation_tables()
    ripple_modulation_id = uuid.uuid4()

    class RippleModulationRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [
                {
                    "ripple_modulation_summary": summary.copy(),
                    "peri_ripple_firing_rate": peri.copy(),
                }
            ]

    relation = RippleModulationRelation()
    result = {
        "ripple_modulation_id": ripple_modulation_id,
        "ripple_modulation_summary_sha256": (
            ripple_modulation.ripple_modulation_summary_sha256(summary)
        ),
        "peri_ripple_firing_rate_sha256": (
            ripple_modulation.peri_ripple_firing_rate_sha256(peri)
        ),
        "artifact_schema_version": ripple_modulation.NWB_ARTIFACT_SCHEMA_VERSION,
        "n_ripples": 3,
        "n_units": 1,
        "n_valid_units": 1,
        "analysis_status": "valid",
    }

    loaded = _load_ripple_modulation_result(
        result_row=result,
        ripple_modulation_table=relation,
    )
    pd.testing.assert_frame_equal(loaded["summary"], summary)
    pd.testing.assert_frame_equal(loaded["peri_ripple_firing_rate"], peri)
    assert relation.keys == [{"ripple_modulation_id": ripple_modulation_id}]

    with pytest.raises(ValueError, match="summary_sha256"):
        _load_ripple_modulation_result(
            result_row={
                **result,
                "ripple_modulation_summary_sha256": "0" * 64,
            },
            ripple_modulation_table=relation,
        )


@pytest.mark.parametrize(
    ("n_units", "n_ripples", "analysis_status"),
    [(0, 3, "no_units"), (2, 0, "no_ripples")],
)
def test_ripple_modulation_result_loader_accepts_terminal_empty_tables(
    n_units: int,
    n_ripples: int,
    analysis_status: str,
) -> None:
    """Terminal selections retain explicit result counts with empty NWB tables."""
    from v1ca1.spyglass import ripple_modulation

    summary, peri = _valid_ripple_modulation_tables()
    summary = summary.iloc[0:0]
    peri = peri.iloc[0:0]
    ripple_modulation_id = uuid.uuid4()

    class RippleModulationRelation:
        def __and__(self, key):
            assert key == {"ripple_modulation_id": ripple_modulation_id}
            return self

        def fetch_nwb(self):
            return [
                {
                    "ripple_modulation_summary": summary.copy(),
                    "peri_ripple_firing_rate": peri.copy(),
                }
            ]

    loaded = _load_ripple_modulation_result(
        result_row={
            "ripple_modulation_id": ripple_modulation_id,
            "ripple_modulation_summary_sha256": (
                ripple_modulation.ripple_modulation_summary_sha256(summary)
            ),
            "peri_ripple_firing_rate_sha256": (
                ripple_modulation.peri_ripple_firing_rate_sha256(peri)
            ),
            "artifact_schema_version": (
                ripple_modulation.NWB_ARTIFACT_SCHEMA_VERSION
            ),
            "n_ripples": n_ripples,
            "n_units": n_units,
            "n_valid_units": 0,
            "analysis_status": analysis_status,
        },
        ripple_modulation_table=RippleModulationRelation(),
    )

    assert loaded["summary"].empty
    assert loaded["peri_ripple_firing_rate"].empty
    assert loaded["analysis_status"] == analysis_status


def test_ripple_modulation_make_requires_populate_transaction() -> None:
    """Direct make cannot register a ripple NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "ripple_modulation_compute": lambda **kwargs: pytest.fail(
                "RippleModulation computation must not start outside populate()."
            )
        }
    )
    ripple = bundle["ripple_modulation"]
    ripple.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        ripple().make({"ripple_modulation_id": uuid.uuid4()})


def test_ripple_modulation_registration_uses_one_datajoint_transaction(
    monkeypatch,
) -> None:
    """Legacy normalization and both ripple result inserts are transactional."""
    ripple_modulation_id = uuid.uuid4()
    selection = {
        "ripple_modulation_id": ripple_modulation_id,
        **_ripple_selection_key(),
    }
    events = []

    class Connection:
        in_transaction = False

        @property
        def transaction(self):
            connection = self

            class Transaction:
                def __enter__(self):
                    events.append("transaction_enter")
                    connection.in_transaction = True

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("transaction_exit")
                    connection.in_transaction = False

            return Transaction()

    connection = Connection()

    def register(**kwargs):
        assert connection.in_transaction
        events.append("register_hook")
        return {
            "analysis_file_name": "registered-ripple-modulation.nwb",
            "ripple_modulation_summary_object_id": "summary-object-id",
            "peri_ripple_firing_rate_object_id": "peri-object-id",
            "ripple_modulation_summary_sha256": "d" * 64,
            "peri_ripple_firing_rate_sha256": "e" * 64,
            "artifact_schema_version": "1",
            "n_ripples": 3,
            "n_units": 1,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "f" * 64,
            "legacy_artifact_provenance": {"source": "legacy"},
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"ripple_modulation_register_existing": register}
    )
    ripple = bundle["ripple_modulation"]
    ripple.connection = connection

    row = ripple.register_existing(
        {"ripple_modulation_id": ripple_modulation_id},
        summary_path="legacy-summary.parquet",
        peri_ripple_firing_rate_path="legacy-peri.parquet",
    )

    assert events == [
        "transaction_enter",
        "register_hook",
        "transaction_exit",
    ]
    assert row["analysis_file_name"] == "registered-ripple-modulation.nwb"
    assert ripple._insert_calls[-1][0]["ripple_modulation_id"] == (
        ripple_modulation_id
    )


@pytest.mark.parametrize("validation_errors", [None, ["invalid test NWB"]])
def test_movement_make_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
    monkeypatch,
    validation_errors: list[str] | None,
) -> None:
    """Movement output is validated before registration and cleaned exactly."""
    import pynwb

    from v1ca1.spyglass import movement

    parameters = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    key = {
        **_movement_selection_key(),
        "movement_firing_rate_id": uuid.uuid4(),
        "movement_parameters_sha256": provenance_sha256(parameters),
    }
    monkeypatch.setattr(
        tables_module,
        "_load_registered_region_spikes",
        lambda **kwargs: {
            "status": "no_units",
            "ts_group": {},
            "unit_ids": [],
            "registration_row": {"region_name": "v1"},
        },
    )
    analysis_path = tmp_path / "movement-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate
    validation_calls = []

    def validate_before_registration(*, path):
        validation_calls.append(str(path))
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    kwargs = {
        "key": key,
        "parameters_table": _FakeRelation(parameters),
        "epoch_intervals_table": _FakeRelation(
            {"start_time": 10.0, "stop_time": 20.0}
        ),
        "position_table": _FakeRelation({"spatial_unit": "cm"}),
        "session_table": _FakeRelation(
            {
                "subject_id": "L14",
                "session_start_time": datetime(2024, 1, 2),
            }
        ),
        "region_sorted_spikes_group_table": object(),
        "nwbfile_table": object(),
        "artifact_root": tmp_path,
        "analysis_nwbfile_table": analysis_table,
    }

    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _make_movement_firing_rate_row(**kwargs)
        assert validation_calls == [str(analysis_path)]
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _make_movement_firing_rate_row(**kwargs)

    assert validation_calls == [str(analysis_path)]
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert analysis_path.stat().st_mode & 0o777 == 0o644
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == movement.NWB_ARTIFACT_SCHEMA_VERSION
    assert row["movement_firing_rate_sha256"] == (
        movement.movement_firing_rate_table_sha256(
            movement.empty_movement_firing_rate_table()
        )
    )
    assert row["movement_intervals_sha256"] == (
        movement.movement_interval_set_sha256(
            movement.movement_interval_set_from_time_intervals(
                pd.DataFrame(columns=["start_time", "stop_time"])
            )
        )
    )
    assert row["analysis_status"] == "no_units"
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_movement_make_requires_populate_transaction() -> None:
    """Direct make cannot register an NWB outside the populate transaction."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "movement_firing_rate_compute": lambda **kwargs: pytest.fail(
                "Movement computation must not start outside populate()."
            )
        }
    )
    movement = bundle["movement_firing_rate"]
    movement.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        movement().make({"movement_firing_rate_id": uuid.uuid4()})


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid path-specific decoding NWB"]],
)
def test_path_specific_decoding_write_uses_eight_nwb_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_errors: list[str] | None,
) -> None:
    """Tables, two timestamp grids, support, and provenance are verified."""
    import pynwb

    from v1ca1.spyglass import path_specific_decoding as decoding

    result = _path_specific_place_decoding_result()
    analysis_path = tmp_path / "path-specific-decoding-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate

    def validate_before_registration(*, path):
        assert not analysis_table.builder.registered
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_path_specific_place_decoding_nwb(
                nwb_file_name="L1420240102_.nwb",
                result=result,
                analysis_nwbfile_table=analysis_table,
            )
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_path_specific_place_decoding_nwb(
        nwb_file_name="L1420240102_.nwb",
        result=result,
        analysis_nwbfile_table=analysis_table,
    )
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["artifact_schema_version"] == decoding.NWB_ARTIFACT_SCHEMA_VERSION
    object_ids = [
        row[field_name]
        for field_name in (
            "selected_units_object_id",
            "fold_qc_object_id",
            "decoding_summary_object_id",
            "decoding_error_by_position_object_id",
            "true_position_object_id",
            "decoded_position_object_id",
            "decoding_support_object_id",
            "decoding_provenance_object_id",
        )
    ]
    assert len(object_ids) == len(set(object_ids)) == 8
    assert row["true_position_sha256"] == (
        decoding.position_time_series_sha256(result["true"], kind="true")
    )
    assert row["decoded_position_sha256"] == (
        decoding.position_time_series_sha256(
            result["decoded"],
            kind="decoded",
        )
    )
    with pynwb.NWBHDF5IO(
        analysis_path,
        mode="r",
        load_namespaces=True,
    ) as io:
        stored = io.read()
        assert decoding.NWB_TRUE_POSITION_TIMESERIES_NAME in stored.scratch
        assert decoding.NWB_DECODED_POSITION_TIMESERIES_NAME in stored.scratch
        assert decoding.NWB_DECODING_SUPPORT_NAME in stored.intervals


def test_path_specific_decoding_make_requires_populate_transaction() -> None:
    """Direct make cannot register a decoder NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_specific_place_decoding_compute": lambda **kwargs: pytest.fail(
                "Decoding computation must not start outside populate()."
            )
        }
    )
    decoding_table = bundle["path_specific_place_decoding"]
    decoding_table.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        decoding_table().make(
            {"path_specific_place_decoding_id": uuid.uuid4()}
        )


def test_path_specific_decoding_loader_uses_fetch_nwb_and_checks_hashes() -> None:
    """Live decoder readers reconstruct all eight objects through fetch_nwb."""
    from v1ca1.spyglass import path_specific_decoding as decoding

    result = _path_specific_place_decoding_result()
    objects = {
        "selected_units": decoding.selected_units_to_dynamic_table(
            result["selected_units"]
        ).to_dataframe(),
        "fold_qc": decoding.fold_qc_to_dynamic_table(
            result["fold_qc"]
        ).to_dataframe(),
        "decoding_summary": decoding.decoding_summary_to_dynamic_table(
            result["summary"]
        ).to_dataframe(),
        "decoding_error_by_position": (
            decoding.binned_error_to_dynamic_table(
                result["binned_error"]
            ).to_dataframe()
        ),
        "true_position": decoding.true_position_to_time_series(result["true"]),
        "decoded_position": decoding.decoded_position_to_time_series(
            result["decoded"]
        ),
        "decoding_support": decoding.decoding_support_to_time_intervals(
            result["true"],
            result["decoded"],
        ).to_dataframe(),
        "decoding_provenance": decoding.decoding_provenance_to_dynamic_table(
            result
        ).to_dataframe(),
    }

    class FetchTable:
        def __and__(self, key):
            return self

        def fetch_nwb(self):
            return [dict(objects)]

    result_row = {
        "path_specific_place_decoding_id": result["metadata"][
            "path_specific_place_decoding_id"
        ],
        "artifact_schema_version": decoding.NWB_ARTIFACT_SCHEMA_VERSION,
        **tables_module._path_specific_place_decoding_hashes(result),
        **{
            name: result[name]
            for name in (
                "n_units",
                "n_folds_expected",
                "n_folds_valid",
                "n_decoded_samples",
                "analysis_status",
                "selected_units_sha256",
                "artifact_origin",
            )
        },
        "legacy_artifact_provenance": result["legacy_artifact_provenance"],
    }
    parameters_row = {
        "path_specific_place_decoding_param_name": result["parameters"][
            "parameter_name"
        ],
        **{
            name: result["parameters"][name]
            for name in decoding.MANUSCRIPT_PARAMETERS
        },
    }
    selection_row = {
        "path_specific_place_decoding_id": result_row[
            "path_specific_place_decoding_id"
        ],
        "path_specific_place_decoding_param_name": parameters_row[
            "path_specific_place_decoding_param_name"
        ],
        "path_specific_place_decoding_parameters_sha256": result[
            "parameters"
        ]["parameter_sha256"],
        "path_specific_place_decoding_output_rule_sha256": result[
            "parameters"
        ]["output_rule_sha256"],
        "epoch": result["metadata"]["epoch"],
    }
    loaded = _load_path_specific_place_decoding_result(
        result_row=result_row,
        decoding_table=FetchTable(),
        selection_row=selection_row,
        parameters_row=parameters_row,
        region_row={"region_name": result["metadata"]["region"]},
        animal_name=result["metadata"]["animal_name"],
        date=result["metadata"]["date"],
    )
    assert np.array_equal(loaded["true"].t, result["true"].t)
    assert np.array_equal(loaded["decoded"].t, result["decoded"].t)

    stale = {**result_row, "decoded_position_sha256": "0" * 64}
    with pytest.raises(ValueError, match="decoded_position_sha256"):
        _load_path_specific_place_decoding_result(
            result_row=stale,
            decoding_table=FetchTable(),
            selection_row=selection_row,
            parameters_row=parameters_row,
            region_row={"region_name": result["metadata"]["region"]},
            animal_name=result["metadata"]["animal_name"],
            date=result["metadata"]["date"],
        )


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid path-specific tuning NWB"]],
)
def test_path_specific_tuning_write_uses_three_nwb_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_errors: list[str] | None,
) -> None:
    """Unit curves, bins, and provenance are verified before registration."""
    import pynwb

    from v1ca1.spyglass import path_specific_place

    curve, _selection = _valid_path_specific_place_curve()
    analysis_path = tmp_path / "path-specific-place-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate

    def validate_before_registration(*, path):
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    kwargs = {
        "nwb_file_name": "L1420240102_.nwb",
        "curve": curve,
        "analysis_nwbfile_table": analysis_table,
    }
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_path_specific_place_tuning_curve_nwb(**kwargs)
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_path_specific_place_tuning_curve_nwb(**kwargs)
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == (
        path_specific_place.NWB_ARTIFACT_SCHEMA_VERSION
    )
    object_ids = {
        row[f"{name}_object_id"]
        for name in (
            "path_specific_place_tuning",
            "path_specific_place_bins",
            "path_specific_place_provenance",
        )
    }
    assert len(object_ids) == 3
    assert row["path_specific_place_tuning_sha256"] == (
        path_specific_place.path_specific_place_tuning_sha256(curve)
    )
    assert row["path_specific_place_bins_sha256"] == (
        path_specific_place.path_specific_place_bins_sha256(curve)
    )
    assert row["path_specific_place_provenance_sha256"] == (
        path_specific_place.path_specific_place_provenance_sha256(curve)
    )
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_path_specific_tuning_loader_uses_fetch_nwb_and_checks_hash() -> None:
    """Live tuning readers reconstruct the DataArray through fetch_nwb()."""
    import xarray as xr

    from v1ca1.spyglass import path_specific_place

    curve, selection = _valid_path_specific_place_curve()
    objects = {
        "path_specific_place_tuning": (
            path_specific_place.path_specific_place_tuning_to_dynamic_table(
                curve
            ).to_dataframe()
        ),
        "path_specific_place_bins": (
            path_specific_place.path_specific_place_bins_to_dynamic_table(
                curve
            ).to_dataframe()
        ),
        "path_specific_place_provenance": (
            path_specific_place.path_specific_place_provenance_to_dynamic_table(
                curve
            ).to_dataframe()
        ),
    }

    class TuningRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [{name: value.copy() for name, value in objects.items()}]

    relation = TuningRelation()
    result = {
        "path_specific_place_tuning_curve_id": selection[
            "path_specific_place_tuning_curve_id"
        ],
        "artifact_schema_version": (
            path_specific_place.NWB_ARTIFACT_SCHEMA_VERSION
        ),
        "path_specific_place_tuning_sha256": (
            path_specific_place.path_specific_place_tuning_sha256(curve)
        ),
        "path_specific_place_bins_sha256": (
            path_specific_place.path_specific_place_bins_sha256(curve)
        ),
        "path_specific_place_provenance_sha256": (
            path_specific_place.path_specific_place_provenance_sha256(curve)
        ),
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "support_duration_s": float(curve.attrs["support_duration_s"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[curve.dims[1]]),
        "analysis_status": str(curve.attrs["analysis_status"]),
        "selected_units_sha256": str(curve.attrs["selected_units_sha256"]),
    }
    loaded = _load_path_specific_place_tuning_curve_result(
        result_row=result,
        tuning_curve_table=relation,
        selection_row=selection,
    )
    xr.testing.assert_identical(loaded, curve)
    assert relation.keys == [
        {
            "path_specific_place_tuning_curve_id": selection[
                "path_specific_place_tuning_curve_id"
            ]
        }
    ]

    with pytest.raises(ValueError, match="path_specific_place_bins_sha256"):
        _load_path_specific_place_tuning_curve_result(
            result_row={
                **result,
                "path_specific_place_bins_sha256": "0" * 64,
            },
            tuning_curve_table=relation,
            selection_row=selection,
        )


def test_path_specific_tuning_make_requires_populate_transaction() -> None:
    """Direct make cannot register a tuning NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_specific_place_tuning_curve_compute": lambda **kwargs: (
                pytest.fail("Tuning computation must not start outside populate().")
            )
        }
    )
    tuning = bundle["path_specific_place_tuning_curve"]
    tuning.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        tuning().make(
            {"path_specific_place_tuning_curve_id": uuid.uuid4()}
        )


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid DPP tuning NWB"]],
)
def test_dpp_tuning_write_uses_three_nwb_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_errors: list[str] | None,
) -> None:
    """DPP unit curves, bins, and provenance precede registration."""
    import pynwb

    from v1ca1.spyglass import dpp

    curve, _selection = _valid_dpp_curve()
    analysis_path = tmp_path / "dpp-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate

    def validate_before_registration(*, path):
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    kwargs = {
        "nwb_file_name": "L1420240102_.nwb",
        "curve": curve,
        "analysis_nwbfile_table": analysis_table,
    }
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_dpp_tuning_curve_nwb(**kwargs)
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_dpp_tuning_curve_nwb(**kwargs)
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == dpp.NWB_ARTIFACT_SCHEMA_VERSION
    object_ids = {
        row[f"{name}_object_id"]
        for name in ("dpp_tuning", "dpp_bins", "dpp_provenance")
    }
    assert len(object_ids) == 3
    assert row["dpp_tuning_sha256"] == dpp.dpp_tuning_sha256(curve)
    assert row["dpp_bins_sha256"] == dpp.dpp_bins_sha256(curve)
    assert row["dpp_provenance_sha256"] == (
        dpp.dpp_provenance_sha256(curve)
    )
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_dpp_tuning_loader_uses_fetch_nwb_and_checks_hash() -> None:
    """Live DPP readers reconstruct the DataArray through fetch_nwb()."""
    import xarray as xr

    from v1ca1.spyglass import dpp

    curve, selection = _valid_dpp_curve()
    objects = {
        "dpp_tuning": dpp.dpp_tuning_to_dynamic_table(curve).to_dataframe(),
        "dpp_bins": dpp.dpp_bins_to_dynamic_table(curve).to_dataframe(),
        "dpp_provenance": (
            dpp.dpp_provenance_to_dynamic_table(curve).to_dataframe()
        ),
    }

    class TuningRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [{name: value.copy() for name, value in objects.items()}]

    relation = TuningRelation()
    result = {
        "dpp_tuning_curve_id": selection["dpp_tuning_curve_id"],
        "artifact_schema_version": dpp.NWB_ARTIFACT_SCHEMA_VERSION,
        "dpp_tuning_sha256": dpp.dpp_tuning_sha256(curve),
        "dpp_bins_sha256": dpp.dpp_bins_sha256(curve),
        "dpp_provenance_sha256": dpp.dpp_provenance_sha256(curve),
        "n_units": int(curve.attrs["n_units"]),
        "n_valid_units": int(curve.attrs["n_valid_units"]),
        "n_trials": int(curve.attrs["n_trials"]),
        "n_outbound_trials": int(curve.attrs["n_outbound_trials"]),
        "n_inbound_trials": int(curve.attrs["n_inbound_trials"]),
        "support_duration_s": float(curve.attrs["support_duration_s"]),
        "n_feature_samples": int(curve.attrs["n_feature_samples"]),
        "n_position_bins": int(curve.sizes[curve.dims[1]]),
        "analysis_status": str(curve.attrs["analysis_status"]),
        "selected_units_sha256": str(curve.attrs["selected_units_sha256"]),
    }
    loaded = _load_dpp_tuning_curve_result(
        result_row=result,
        tuning_curve_table=relation,
        selection_row=selection,
    )
    xr.testing.assert_identical(loaded, curve)
    assert relation.keys == [
        {"dpp_tuning_curve_id": selection["dpp_tuning_curve_id"]}
    ]

    with pytest.raises(ValueError, match="dpp_bins_sha256"):
        _load_dpp_tuning_curve_result(
            result_row={**result, "dpp_bins_sha256": "0" * 64},
            tuning_curve_table=relation,
            selection_row=selection,
        )


def test_dpp_tuning_make_requires_populate_transaction() -> None:
    """Direct make cannot register a DPP NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "dpp_tuning_curve_compute": lambda **kwargs: pytest.fail(
                "DPP computation must not start outside populate()."
            )
        }
    )
    tuning = bundle["dpp_tuning_curve"]
    tuning.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        tuning().make({"dpp_tuning_curve_id": uuid.uuid4()})


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid tuning-similarity NWB"]],
)
def test_tuning_similarity_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
    monkeypatch,
    validation_errors: list[str] | None,
) -> None:
    """Similarity output is validated before registration and cleaned exactly."""
    import pynwb

    from v1ca1.spyglass import tuning_similarity

    analysis_path = tmp_path / "tuning-similarity-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate
    validation_calls = []

    def validate_before_registration(*, path):
        validation_calls.append(str(path))
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    source = _valid_tuning_similarity_table()
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_path_specific_place_tuning_similarity_nwb(
                nwb_file_name="L1420240102_.nwb",
                table=source,
                analysis_nwbfile_table=analysis_table,
            )
        assert validation_calls == [str(analysis_path)]
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_path_specific_place_tuning_similarity_nwb(
        nwb_file_name="L1420240102_.nwb",
        table=source,
        analysis_nwbfile_table=analysis_table,
    )

    assert validation_calls == [str(analysis_path)]
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == (
        tuning_similarity.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["similarity_sha256"] == (
        tuning_similarity.tuning_similarity_table_sha256(source)
    )
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_tuning_similarity_result_loader_uses_fetch_nwb_and_checks_hash() -> None:
    """Live similarity readers resolve the DynamicTable through fetch_nwb()."""
    from v1ca1.spyglass import tuning_similarity

    similarity_id = uuid.UUID("74444444-4444-5444-8444-444444444444")
    source = _valid_tuning_similarity_table()

    class SimilarityRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [{"similarity": source.copy()}]

    relation = SimilarityRelation()
    result = {
        "path_specific_place_tuning_similarity_id": similarity_id,
        "similarity_sha256": (
            tuning_similarity.tuning_similarity_table_sha256(source)
        ),
        "artifact_schema_version": (
            tuning_similarity.NWB_ARTIFACT_SCHEMA_VERSION
        ),
        "n_units": 1,
        "n_valid_comparisons": 4,
        "n_units_with_valid_comparison": 1,
        "analysis_status": "valid",
        "selected_units_sha256": unit_identity_sha256(
            [{"spikesorting_merge_id": "merge-a", "unit_id": "11"}]
        ),
    }

    loaded = _load_path_specific_place_tuning_similarity_result(
        result_row=result,
        similarity_table=relation,
        similarity_metric="correlation",
    )
    pd.testing.assert_frame_equal(loaded, source)
    assert relation.keys == [
        {"path_specific_place_tuning_similarity_id": similarity_id}
    ]

    with pytest.raises(ValueError, match="similarity_sha256"):
        _load_path_specific_place_tuning_similarity_result(
            result_row={**result, "similarity_sha256": "0" * 64},
            similarity_table=relation,
            similarity_metric="correlation",
        )


def test_tuning_similarity_make_requires_populate_transaction() -> None:
    """Direct make cannot register a similarity NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_specific_place_tuning_similarity_compute": (
                lambda **kwargs: pytest.fail(
                    "Similarity computation must not start outside populate()."
                )
            )
        }
    )
    similarity_table = bundle["path_specific_place_tuning_similarity"]
    similarity_table.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        similarity_table().make(
            {"path_specific_place_tuning_similarity_id": uuid.uuid4()}
        )


def test_tuning_similarity_registration_uses_one_datajoint_transaction(
    monkeypatch,
) -> None:
    """Legacy normalization and both result inserts share one transaction."""
    similarity_id = uuid.UUID("73333333-3333-5333-8333-333333333333")
    selection = {
        "path_specific_place_tuning_similarity_id": similarity_id,
        **_tuning_similarity_selection_key(),
        "tuning_similarity_parameters_sha256": provenance_sha256(
            dict(table_specs.CORRELATION_TUNING_SIMILARITY_PARAMETERS)
        ),
    }
    events = []

    class Connection:
        in_transaction = False

        @property
        def transaction(self):
            connection = self

            class Transaction:
                def __enter__(self):
                    events.append("transaction_enter")
                    connection.in_transaction = True

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("transaction_exit")
                    connection.in_transaction = False

            return Transaction()

    connection = Connection()

    def register(**kwargs):
        assert connection.in_transaction
        events.append("register_hook")
        return {
            "analysis_file_name": "registered-tuning-similarity.nwb",
            "similarity_object_id": "similarity-object-id",
            "similarity_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "n_units": 1,
            "n_valid_comparisons": 4,
            "n_units_with_valid_comparison": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "e" * 64,
            "legacy_artifact_provenance": {"source": "all_units"},
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_specific_place_tuning_similarity_register_existing": (
                register
            ),
        }
    )
    similarity_table = bundle["path_specific_place_tuning_similarity"]
    similarity_table.connection = connection

    row = similarity_table.register_existing(
        {"path_specific_place_tuning_similarity_id": similarity_id},
        similarity_path="legacy-similarity-all_units.parquet",
    )

    assert events == [
        "transaction_enter",
        "register_hook",
        "transaction_exit",
    ]
    assert row["analysis_file_name"] == "registered-tuning-similarity.nwb"
    assert similarity_table._insert_calls[-1][0][
        "path_specific_place_tuning_similarity_id"
    ] == similarity_id


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid DPPEncoding NWB"]],
)
def test_dpp_encoding_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
    monkeypatch,
    validation_errors: list[str] | None,
) -> None:
    """Encoding output is validated before registration and cleaned exactly."""
    import pynwb

    from v1ca1.spyglass import dpp_encoding

    analysis_path = tmp_path / "dpp-encoding-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate
    validation_calls = []

    def validate_before_registration(*, path):
        validation_calls.append(str(path))
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    source = _valid_dpp_encoding_table()
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_dpp_encoding_nwb(
                nwb_file_name="L1420240102_.nwb",
                table=source,
                analysis_nwbfile_table=analysis_table,
            )
        assert validation_calls == [str(analysis_path)]
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_dpp_encoding_nwb(
        nwb_file_name="L1420240102_.nwb",
        table=source,
        analysis_nwbfile_table=analysis_table,
    )

    assert validation_calls == [str(analysis_path)]
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == (
        dpp_encoding.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["dpp_encoding_sha256"] == (
        dpp_encoding.dpp_encoding_table_sha256(source)
    )
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_dpp_encoding_result_loader_uses_fetch_nwb_and_checks_hash() -> None:
    """Live encoding readers resolve the DynamicTable through fetch_nwb()."""
    from v1ca1.spyglass import dpp_encoding

    source = _valid_dpp_encoding_table()
    dpp_encoding_id = uuid.UUID(source["dpp_encoding_id"].iloc[0])

    class DPPEncodingRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [{"dpp_encoding": source.copy()}]

    relation = DPPEncodingRelation()
    result = {
        "dpp_encoding_id": dpp_encoding_id,
        "dpp_encoding_sha256": (
            dpp_encoding.dpp_encoding_table_sha256(source)
        ),
        "artifact_schema_version": dpp_encoding.NWB_ARTIFACT_SCHEMA_VERSION,
    }

    loaded = _load_dpp_encoding_result(
        result_row=result,
        dpp_encoding_table=relation,
    )
    pd.testing.assert_frame_equal(loaded, source)
    assert relation.keys == [{"dpp_encoding_id": dpp_encoding_id}]

    with pytest.raises(ValueError, match="dpp_encoding_sha256"):
        _load_dpp_encoding_result(
            result_row={**result, "dpp_encoding_sha256": "0" * 64},
            dpp_encoding_table=relation,
        )


def test_dpp_encoding_make_requires_populate_transaction() -> None:
    """Direct make cannot register a DPPEncoding NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "dpp_encoding_compute": lambda **kwargs: pytest.fail(
                "DPPEncoding computation must not start outside populate()."
            )
        }
    )
    dpp_encoding_table = bundle["dpp_encoding"]
    dpp_encoding_table.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        dpp_encoding_table().make({"dpp_encoding_id": uuid.uuid4()})


def test_dpp_encoding_registration_uses_one_datajoint_transaction(
    monkeypatch,
) -> None:
    """Legacy normalization and both result inserts share one transaction."""
    dpp_encoding_id = uuid.UUID("71111111-1111-5111-8111-111111111111")
    selection = {"dpp_encoding_id": dpp_encoding_id}
    events = []

    class Connection:
        in_transaction = False

        @property
        def transaction(self):
            connection = self

            class Transaction:
                def __enter__(self):
                    events.append("transaction_enter")
                    connection.in_transaction = True

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("transaction_exit")
                    connection.in_transaction = False

            return Transaction()

    connection = Connection()

    def register(**kwargs):
        assert connection.in_transaction
        events.append("register_hook")
        return {
            "analysis_file_name": "registered-dpp-encoding.nwb",
            "dpp_encoding_object_id": "dpp-encoding-object-id",
            "dpp_encoding_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "n_units_input": 2,
            "n_units_eligible": 1,
            "n_units_valid": 1,
            "analysis_status": "valid",
            "eligible_units_sha256": "e" * 64,
            "legacy_artifact_provenance": {"source": "legacy"},
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"dpp_encoding_register_existing": register}
    )
    dpp_encoding_table = bundle["dpp_encoding"]
    dpp_encoding_table.connection = connection

    row = dpp_encoding_table.register_existing(
        {"dpp_encoding_id": dpp_encoding_id},
        dpp_encoding_path="legacy-encoding.parquet",
    )

    assert events == [
        "transaction_enter",
        "register_hook",
        "transaction_exit",
    ]
    assert row["analysis_file_name"] == "registered-dpp-encoding.nwb"
    assert dpp_encoding_table._insert_calls[-1][0]["dpp_encoding_id"] == (
        dpp_encoding_id
    )


@pytest.mark.parametrize("validation_errors", [None, ["invalid stability NWB"]])
def test_stability_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
    monkeypatch,
    validation_errors: list[str] | None,
) -> None:
    """Stability output is validated before registration and cleaned exactly."""
    import pynwb

    from v1ca1.spyglass import stability

    analysis_path = tmp_path / "stability-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate
    validation_calls = []

    def validate_before_registration(*, path):
        validation_calls.append(str(path))
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    source = _valid_stability_table()
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            _write_path_specific_place_stability_nwb(
                nwb_file_name="L1420240102_.nwb",
                table=source,
                analysis_nwbfile_table=analysis_table,
            )
        assert validation_calls == [str(analysis_path)]
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = _write_path_specific_place_stability_nwb(
        nwb_file_name="L1420240102_.nwb",
        table=source,
        analysis_nwbfile_table=analysis_table,
    )

    assert validation_calls == [str(analysis_path)]
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == stability.NWB_ARTIFACT_SCHEMA_VERSION
    assert row["stability_sha256"] == stability.stability_table_sha256(source)
    assert row["_created_artifact_paths"] == [str(analysis_path)]


def test_stability_result_loader_uses_fetch_nwb_and_checks_hash() -> None:
    """Live stability readers resolve object IDs rather than filesystem paths."""
    from v1ca1.spyglass import stability

    stability_id = uuid.UUID("75555555-5555-5555-8555-555555555555")
    source = _valid_stability_table()

    class StabilityRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [{"stability": source.copy()}]

    relation = StabilityRelation()
    result = {
        "path_specific_place_stability_id": stability_id,
        "stability_sha256": stability.stability_table_sha256(source),
        "artifact_schema_version": stability.NWB_ARTIFACT_SCHEMA_VERSION,
        "n_units": 1,
        "n_valid_units": 1,
        "analysis_status": "valid",
        "selected_units_sha256": unit_identity_sha256(
            [{"spikesorting_merge_id": "merge-a", "unit_id": "11"}]
        ),
    }

    loaded = _load_path_specific_place_stability_result(
        result_row=result,
        stability_table=relation,
        expected_metadata={"trajectory_type": "center_to_left"},
    )
    pd.testing.assert_frame_equal(loaded, source)
    assert relation.keys == [
        {"path_specific_place_stability_id": stability_id}
    ]

    with pytest.raises(ValueError, match="stability_sha256"):
        _load_path_specific_place_stability_result(
            result_row={**result, "stability_sha256": "0" * 64},
            stability_table=relation,
        )


def test_stability_make_requires_populate_transaction() -> None:
    """Direct make cannot register a stability NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_specific_place_stability_compute": lambda **kwargs: pytest.fail(
                "Stability computation must not start outside populate()."
            )
        }
    )
    stability_table = bundle["path_specific_place_stability"]
    stability_table.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        stability_table().make(
            {"path_specific_place_stability_id": uuid.uuid4()}
        )


def test_stability_registration_uses_one_datajoint_transaction(monkeypatch) -> None:
    """Legacy normalization and both result inserts share one transaction."""
    stability_id = uuid.UUID("76666666-6666-5666-8666-666666666666")
    selection = {
        "path_specific_place_stability_id": stability_id,
        **_stability_selection_key(),
    }
    events = []

    class Connection:
        in_transaction = False

        @property
        def transaction(self):
            connection = self

            class Transaction:
                def __enter__(self):
                    events.append("transaction_enter")
                    connection.in_transaction = True

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("transaction_exit")
                    connection.in_transaction = False

            return Transaction()

    connection = Connection()

    def register(**kwargs):
        assert connection.in_transaction
        events.append("register_hook")
        return {
            "analysis_file_name": "registered-stability.nwb",
            "stability_object_id": "stability-object-id",
            "stability_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "n_units": 1,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "e" * 64,
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_specific_place_stability_register_existing": register,
        }
    )
    stability_table = bundle["path_specific_place_stability"]
    stability_table.connection = connection

    row = stability_table.register_existing(
        {"path_specific_place_stability_id": stability_id},
        stability_path="legacy-stability.parquet",
    )

    assert events == [
        "transaction_enter",
        "register_hook",
        "transaction_exit",
    ]
    assert row["analysis_file_name"] == "registered-stability.nwb"
    assert stability_table._insert_calls[-1][0][
        "path_specific_place_stability_id"
    ] == stability_id


def test_result_make_and_register_hooks_receive_fetched_selection(monkeypatch) -> None:
    ripple_id = uuid.UUID("11111111-1111-5111-8111-111111111111")
    stability_id = uuid.UUID("22222222-2222-5222-8222-222222222222")
    movement_id = uuid.UUID("33333333-3333-5333-8333-333333333333")
    tuning_curve_id = uuid.UUID("66666666-6666-5666-8666-666666666666")
    ripple_selection = {"ripple_modulation_id": ripple_id, **_ripple_selection_key()}
    movement_selection = {
        "movement_firing_rate_id": movement_id,
        **_movement_selection_key(),
    }
    tuning_curve_selection = {
        "path_specific_place_tuning_curve_id": tuning_curve_id,
        **_tuning_curve_selection_key(trial_subset="all"),
    }
    stability_selection = {
        "path_specific_place_stability_id": stability_id,
        **_stability_selection_key(),
    }
    calls = []

    def ripple_compute(**kwargs):
        calls.append(("ripple_compute", kwargs))
        return {
            "analysis_file_name": "ripple-modulation-analysis.nwb",
            "ripple_modulation_summary_object_id": "summary-object-id",
            "peri_ripple_firing_rate_object_id": "peri-object-id",
            "ripple_modulation_summary_sha256": "d" * 64,
            "peri_ripple_firing_rate_sha256": "e" * 64,
            "artifact_schema_version": "1",
            "n_ripples": 4,
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }

    def movement_compute(**kwargs):
        calls.append(("movement_compute", kwargs))
        return {
            "analysis_file_name": "movement-analysis.nwb",
            "movement_firing_rate_object_id": "rate-object-id",
            "movement_intervals_object_id": "interval-object-id",
            "movement_firing_rate_sha256": "d" * 64,
            "movement_intervals_sha256": "e" * 64,
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 2,
            "n_units_with_spikes": 1,
            "movement_interval_count": 3,
            "movement_duration_s": 12.5,
            "analysis_status": "valid",
            "selected_units_sha256": "c" * 64,
        }

    def ripple_register(**kwargs):
        calls.append(("ripple_register", kwargs))
        return {
            "analysis_file_name": "registered-ripple-modulation.nwb",
            "ripple_modulation_summary_object_id": "summary-object-id",
            "peri_ripple_firing_rate_object_id": "peri-object-id",
            "ripple_modulation_summary_sha256": "d" * 64,
            "peri_ripple_firing_rate_sha256": "e" * 64,
            "artifact_schema_version": "1",
            "n_ripples": 4,
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }

    def tuning_curve_compute(**kwargs):
        calls.append(("tuning_curve_compute", kwargs))
        return {
            "analysis_file_name": "tuning-curve-analysis.nwb",
            "path_specific_place_tuning_object_id": "tuning-object-id",
            "path_specific_place_bins_object_id": "bins-object-id",
            "path_specific_place_provenance_object_id": "provenance-object-id",
            "path_specific_place_tuning_sha256": "1" * 64,
            "path_specific_place_bins_sha256": "2" * 64,
            "path_specific_place_provenance_sha256": "3" * 64,
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 2,
            "n_trials": 5,
            "support_duration_s": 12.5,
            "n_feature_samples": 125,
            "n_position_bins": 25,
            "analysis_status": "valid",
            "selected_units_sha256": "b" * 64,
        }

    def tuning_curve_register(**kwargs):
        calls.append(("tuning_curve_register", kwargs))
        return {
            "analysis_file_name": "registered-tuning-curve-analysis.nwb",
            "path_specific_place_tuning_object_id": "tuning-object-id",
            "path_specific_place_bins_object_id": "bins-object-id",
            "path_specific_place_provenance_object_id": "provenance-object-id",
            "path_specific_place_tuning_sha256": "1" * 64,
            "path_specific_place_bins_sha256": "2" * 64,
            "path_specific_place_provenance_sha256": "3" * 64,
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 2,
            "n_trials": 5,
            "support_duration_s": 12.5,
            "n_feature_samples": 125,
            "n_position_bins": 25,
            "analysis_status": "valid",
            "selected_units_sha256": "b" * 64,
        }

    def stability_compute(**kwargs):
        calls.append(("stability_compute", kwargs))
        return {
            "analysis_file_name": "stability-analysis.nwb",
            "stability_object_id": "stability-object-id",
            "stability_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "b" * 64,
        }

    def stability_register(**kwargs):
        calls.append(("stability_register", kwargs))
        return {
            "analysis_file_name": "keyed-stability-analysis.nwb",
            "stability_object_id": "keyed-stability-object-id",
            "stability_sha256": "1" * 64,
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "b" * 64,
        }

    def fetch_selection(table, key):
        if table.__name__ == "RippleModulationSelection":
            assert key == {"ripple_modulation_id": ripple_id}
            return dict(ripple_selection)
        if table.__name__ == "MovementFiringRateSelection":
            assert key == {"movement_firing_rate_id": movement_id}
            return dict(movement_selection)
        if table.__name__ == "PathSpecificPlaceTuningCurveSelection":
            assert key == {
                "path_specific_place_tuning_curve_id": tuning_curve_id
            }
            return dict(tuning_curve_selection)
        if table.__name__ == "PathSpecificPlaceStabilitySelection":
            assert key == {"path_specific_place_stability_id": stability_id}
            return dict(stability_selection)
        raise AssertionError(f"Unexpected fetch from {table.__name__}")

    monkeypatch.setitem(_construct_tables.__globals__, "_fetch1_dict", fetch_selection)
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_spyglass_git_commit",
        lambda: "runtime-spyglass-commit",
    )
    bundle, _, unit_selection_params = _fake_bundle(
        runtime_hooks={
            "ripple_modulation_compute": ripple_compute,
            "ripple_modulation_register_existing": ripple_register,
            "movement_firing_rate_compute": movement_compute,
            "path_specific_place_tuning_curve_compute": tuning_curve_compute,
            "path_specific_place_tuning_curve_register_existing": (
                tuning_curve_register
            ),
            "path_specific_place_stability_compute": stability_compute,
            "path_specific_place_stability_register_existing": stability_register,
        }
    )

    ripple = bundle["ripple_modulation"]
    movement = bundle["movement_firing_rate"]
    tuning_curve = bundle["path_specific_place_tuning_curve"]
    stability = bundle["path_specific_place_stability"]
    ripple().make({"ripple_modulation_id": ripple_id})
    ripple.register_existing(
        {"ripple_modulation_id": ripple_id},
        summary_path="old-summary.parquet",
        peri_ripple_firing_rate_path="old-peri.parquet",
        source_v1ca1_git_commit="source-v1-commit",
    )
    movement().make({"movement_firing_rate_id": movement_id})
    tuning_curve().make({"path_specific_place_tuning_curve_id": tuning_curve_id})
    tuning_curve.register_existing(
        {"path_specific_place_tuning_curve_id": tuning_curve_id},
        tuning_curve_path="old-tuning-curve.nc",
        source_v1ca1_git_commit="source-v1-commit",
    )
    stability().make({"path_specific_place_stability_id": stability_id})
    stability.register_existing(
        {"path_specific_place_stability_id": stability_id},
        stability_path="old-stability.parquet",
        source_v1ca1_git_commit="source-v1-commit",
    )

    assert [name for name, _ in calls] == [
        "ripple_compute",
        "ripple_register",
        "movement_compute",
        "tuning_curve_compute",
        "tuning_curve_register",
        "stability_compute",
        "stability_register",
    ]
    assert calls[0][1]["key"] == ripple_selection
    assert calls[1][1]["key"] == ripple_selection
    assert calls[2][1]["key"] == movement_selection
    assert calls[3][1]["key"] == tuning_curve_selection
    assert calls[4][1]["key"] == tuning_curve_selection
    assert calls[5][1]["key"] == stability_selection
    assert calls[6][1]["key"] == stability_selection
    assert calls[1][1]["overwrite"] is False
    assert calls[4][1]["overwrite"] is False
    assert calls[6][1]["overwrite"] is False
    for call_index in (0, 1, 2, 3, 4):
        assert calls[call_index][1][
            "region_sorted_spikes_group_table"
        ] is bundle["region_sorted_spikes_group"]
    for call_index in (0, 1):
        assert calls[call_index][1]["analysis_nwbfile_table"] is bundle[
            "analysis_nwbfile"
        ]
    assert ripple._insert_calls[0][0]["ripple_modulation_id"] == ripple_id
    assert ripple._insert_calls[1][0]["artifact_origin"] == "registered_existing"
    assert ripple._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }
    assert movement._insert_calls[0][0]["movement_firing_rate_id"] == movement_id
    assert "artifact_origin" not in movement._insert_calls[0][0]
    assert calls[2][1]["parameters_table"] is bundle["movement_parameters"]
    assert calls[2][1]["analysis_nwbfile_table"] is bundle[
        "analysis_nwbfile"
    ]
    for call_index in (3, 4):
        assert calls[call_index][1]["movement_firing_rate_table"] is movement
        assert calls[call_index][1][
            "movement_firing_rate_selection_table"
        ] is bundle["movement_firing_rate_selection"]
        assert calls[call_index][1]["movement_parameters_table"] is bundle[
            "movement_parameters"
        ]
        assert calls[call_index][1]["parameters_table"] is bundle[
            "tuning_curve_parameters"
        ]
        assert calls[call_index][1]["analysis_nwbfile_table"] is bundle[
            "analysis_nwbfile"
        ]
    assert tuning_curve._insert_calls[0][0][
        "path_specific_place_tuning_curve_id"
    ] == tuning_curve_id
    assert (
        tuning_curve._insert_calls[1][0]["artifact_origin"]
        == "registered_existing"
    )
    assert tuning_curve._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }
    for call_index in (5, 6):
        assert calls[call_index][1]["tuning_curve_table"] is tuning_curve
        assert calls[call_index][1][
            "tuning_curve_selection_table"
        ] is bundle["path_specific_place_tuning_curve_selection"]
        assert calls[call_index][1]["analysis_nwbfile_table"] is bundle[
            "analysis_nwbfile"
        ]
    assert stability._insert_calls[0][0][
        "path_specific_place_stability_id"
    ] == stability_id
    assert stability._insert_calls[1][0]["artifact_origin"] == "registered_existing"
    assert stability._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }
    assert all(
        row["runtime_spyglass_git_commit"] == "runtime-spyglass-commit"
        for result in (ripple, movement, tuning_curve, stability)
        for row, _kwargs in result._insert_calls
    )


def test_dpp_result_hooks_receive_fetched_selection(monkeypatch) -> None:
    selection_id = uuid.UUID("77777777-7777-5777-8777-777777777777")
    selection = {
        "dpp_tuning_curve_id": selection_id,
        **_dpp_tuning_curve_selection_key(),
        "outbound_trajectory_type": "center_to_left",
        "inbound_trajectory_type": "right_to_center",
        "outbound_configuration_name": "center_to_left",
        "inbound_configuration_name": "right_to_center",
    }
    calls = []

    def result_row(analysis_file_name: str) -> dict[str, Any]:
        return {
            "analysis_file_name": analysis_file_name,
            "dpp_tuning_object_id": "dpp-tuning-object-id",
            "dpp_bins_object_id": "dpp-bins-object-id",
            "dpp_provenance_object_id": "dpp-provenance-object-id",
            "dpp_tuning_sha256": "1" * 64,
            "dpp_bins_sha256": "2" * 64,
            "dpp_provenance_sha256": "3" * 64,
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 2,
            "n_trials": 7,
            "n_outbound_trials": 4,
            "n_inbound_trials": 3,
            "support_duration_s": 12.5,
            "n_feature_samples": 125,
            "n_position_bins": 41,
            "analysis_status": "valid",
            "selected_units_sha256": "d" * 64,
        }

    def compute(**kwargs):
        calls.append(("compute", kwargs))
        return result_row("dpp-tuning-analysis.nwb")

    def register(**kwargs):
        calls.append(("register", kwargs))
        return result_row("registered-dpp-tuning-analysis.nwb")

    def fetch_selection(table, key):
        assert table.__name__ == "DPPTuningCurveSelection"
        assert key == {"dpp_tuning_curve_id": selection_id}
        return dict(selection)

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        fetch_selection,
    )
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_spyglass_git_commit",
        lambda: "runtime-spyglass-commit",
    )
    bundle, _, unit_selection_params = _fake_bundle(
        runtime_hooks={
            "dpp_tuning_curve_compute": compute,
            "dpp_tuning_curve_register_existing": register,
        }
    )
    result = bundle["dpp_tuning_curve"]

    result().make({"dpp_tuning_curve_id": selection_id})
    result.register_existing(
        {"dpp_tuning_curve_id": selection_id},
        tuning_curve_path="old-dpp-tuning-curve.nc",
    )

    assert [name for name, _kwargs in calls] == ["compute", "register"]
    assert all(call["key"] == selection for _name, call in calls)
    assert calls[1][1]["overwrite"] is False
    for _name, call in calls:
        assert call["parameters_table"] is bundle["tuning_curve_parameters"]
        assert call["movement_firing_rate_table"] is bundle[
            "movement_firing_rate"
        ]
        assert call["region_sorted_spikes_group_table"] is bundle[
            "region_sorted_spikes_group"
        ]
        assert call["analysis_nwbfile_table"] is bundle["analysis_nwbfile"]
    assert result._insert_calls[0][0]["dpp_tuning_curve_id"] == selection_id
    assert result._insert_calls[0][0]["artifact_origin"] == "computed"
    assert result._insert_calls[1][0]["artifact_origin"] == (
        "registered_existing"
    )
    assert result._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }


def test_path_progression_decoding_make_uses_default_and_injected_hook(
    monkeypatch,
) -> None:
    comparison_id = uuid.UUID("86666666-6666-5666-8666-666666666666")
    selection = {
        "path_progression_decoding_id": comparison_id,
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "cohort_epoch": "08_r4",
    }
    default_bundle, _, _ = _fake_bundle()
    assert default_bundle[
        "path_progression_decoding"
    ]._compute_hook is _make_path_progression_decoding_row

    calls = []
    transfer_row = {
        "transfer_family": "same_turn_cross_arm",
        "source_trajectory": "center_to_left",
        "target_trajectory": "center_to_right",
        "true_progression_object_id": "true-object-id",
        "decoded_progression_object_id": "decoded-object-id",
        "decoding_support_object_id": "support-object-id",
        "true_progression_sha256": "1" * 64,
        "decoded_progression_sha256": "2" * 64,
        "decoding_support_sha256": "3" * 64,
        "n_samples": 1000,
    }

    def compute(**kwargs):
        calls.append(kwargs)
        return {
            "analysis_file_name": "path-progression-analysis.nwb",
            "unit_eligibility_object_id": "eligibility-object-id",
            "selected_units_object_id": "selected-object-id",
            "decoding_summary_object_id": "summary-object-id",
            "cross_path_binned_error_object_id": "binned-object-id",
            "transfer_index_object_id": "index-object-id",
            "decoding_provenance_object_id": "provenance-object-id",
            "unit_eligibility_sha256": "a" * 64,
            "selected_units_table_sha256": "b" * 64,
            "decoding_summary_sha256": "c" * 64,
            "cross_path_binned_error_sha256": "d" * 64,
            "transfer_index_sha256": "f" * 64,
            "decoding_provenance_sha256": "0" * 64,
            "artifact_schema_version": "1",
            "n_units_input": 5,
            "n_units_eligible": 3,
            "n_transfer_pairs_expected": 16,
            "n_transfer_pairs_valid": 1,
            "n_decoded_samples": 1000,
            "analysis_status": "partial_valid",
            "eligible_units_sha256": "e" * 64,
            "_transfer_rows": [transfer_row],
            "_created_artifact_paths": [
                "/analysis/path-progression-analysis.nwb"
            ],
        }

    def fetch_selection(table, key):
        assert table.__name__ == (
            "PathProgressionDecodingSelection"
        )
        assert key == {
            "path_progression_decoding_id": comparison_id
        }
        return dict(selection)

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        fetch_selection,
    )
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_v1ca1_git_commit",
        lambda: "runtime-v1ca1-commit",
    )
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_spyglass_git_commit",
        lambda: "runtime-spyglass-commit",
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_progression_decoding_compute": compute,
        }
    )
    result = bundle["path_progression_decoding"]

    result().make(
        {"path_progression_decoding_id": comparison_id}
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["key"] == selection
    assert call["parameters_table"] is bundle[
        "path_progression_decoding_parameters"
    ]
    assert call["region_sorted_spikes_group_table"] is bundle[
        "region_sorted_spikes_group"
    ]
    assert call["movement_firing_rate_table"] is bundle[
        "movement_firing_rate"
    ]
    assert call["stability_table"] is bundle[
        "path_specific_place_stability"
    ]
    assert call["artifact_root"] == Path("/analysis")
    assert call["analysis_nwbfile_table"] is bundle["analysis_nwbfile"]
    inserted, insert_kwargs = result._insert_calls[0]
    assert insert_kwargs == {}
    assert inserted == {
        "path_progression_decoding_id": comparison_id,
        "analysis_file_name": "path-progression-analysis.nwb",
        "unit_eligibility_object_id": "eligibility-object-id",
        "selected_units_object_id": "selected-object-id",
        "decoding_summary_object_id": "summary-object-id",
        "cross_path_binned_error_object_id": "binned-object-id",
        "transfer_index_object_id": "index-object-id",
        "decoding_provenance_object_id": "provenance-object-id",
        "unit_eligibility_sha256": "a" * 64,
        "selected_units_table_sha256": "b" * 64,
        "decoding_summary_sha256": "c" * 64,
        "cross_path_binned_error_sha256": "d" * 64,
        "transfer_index_sha256": "f" * 64,
        "decoding_provenance_sha256": "0" * 64,
        "artifact_schema_version": "1",
        "n_units_input": 5,
        "n_units_eligible": 3,
        "n_transfer_pairs_expected": 16,
        "n_transfer_pairs_valid": 1,
        "n_decoded_samples": 1000,
        "analysis_status": "partial_valid",
        "eligible_units_sha256": "e" * 64,
        "runtime_v1ca1_git_commit": "runtime-v1ca1-commit",
        "runtime_spyglass_git_commit": "runtime-spyglass-commit",
    }
    assert result.Transfer._insert_many_calls == [
        (
            [
                {
                    "path_progression_decoding_id": comparison_id,
                    "analysis_file_name": "path-progression-analysis.nwb",
                    **transfer_row,
                }
            ],
            {},
        )
    ]
    assert "artifact_origin" not in inserted
    assert not hasattr(result, "register_existing")


def test_path_progression_decoding_failed_insert_removes_new_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    comparison_id = uuid.UUID("87777777-7777-5777-8777-777777777777")
    analysis_path = tmp_path / "path-progression-analysis.nwb"
    retained_path = tmp_path / "preexisting.parquet"
    retained_path.write_bytes(b"keep")

    def compute(**_kwargs):
        analysis_path.write_bytes(b"new")
        return {
            "analysis_file_name": analysis_path.name,
            "unit_eligibility_object_id": "eligibility-object-id",
            "selected_units_object_id": "selected-object-id",
            "decoding_summary_object_id": "summary-object-id",
            "cross_path_binned_error_object_id": "binned-object-id",
            "transfer_index_object_id": "index-object-id",
            "decoding_provenance_object_id": "provenance-object-id",
            "unit_eligibility_sha256": "a" * 64,
            "selected_units_table_sha256": "b" * 64,
            "decoding_summary_sha256": "c" * 64,
            "cross_path_binned_error_sha256": "d" * 64,
            "transfer_index_sha256": "f" * 64,
            "decoding_provenance_sha256": "0" * 64,
            "artifact_schema_version": "1",
            "n_units_input": 5,
            "n_units_eligible": 3,
            "n_transfer_pairs_expected": 16,
            "n_transfer_pairs_valid": 0,
            "n_decoded_samples": 1000,
            "analysis_status": "no_valid_decodes",
            "eligible_units_sha256": "e" * 64,
            "_transfer_rows": [],
            "_created_artifact_paths": [str(analysis_path)],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {
            "path_progression_decoding_id": comparison_id
        },
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_progression_decoding_compute": compute,
        }
    )
    result = bundle["path_progression_decoding"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make(
            {"path_progression_decoding_id": comparison_id}
        )

    assert not analysis_path.exists()
    assert retained_path.read_bytes() == b"keep"


def test_path_progression_decoding_make_requires_populate_transaction() -> None:
    """Direct make cannot register parent and transfer rows independently."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "path_progression_decoding_compute": lambda **kwargs: pytest.fail(
                "Path-progression computation must not start outside populate()."
            )
        }
    )
    result = bundle["path_progression_decoding"]
    result.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"path_progression_decoding_id": uuid.uuid4()})


def test_path_specific_place_decoding_selection_is_deterministic() -> None:
    first = _build_path_specific_place_decoding_selection()
    second = _build_path_specific_place_decoding_selection()

    assert first == second
    assert first["path_specific_place_decoding_id"].version == 5
    assert first[
        "path_specific_place_decoding_parameters_sha256"
    ] == provenance_sha256(
        dict(table_specs.MANUSCRIPT_PATH_SPECIFIC_PLACE_DECODING_PARAMETERS)
    )
    assert first[
        "path_specific_place_decoding_output_rule_sha256"
    ] == provenance_sha256(
        dict(table_specs.PATH_SPECIFIC_PLACE_DECODING_OUTPUT_RULE)
    )


def test_motor_encoding_selection_is_deterministic_and_freezes_rules() -> None:
    first = _build_motor_encoding_selection(
        _motor_encoding_selection_inputs()
    )
    second = _build_motor_encoding_selection(
        _motor_encoding_selection_inputs()
    )

    assert first == second
    assert first["motor_encoding_id"].version == 5
    assert first["primary_position_series_name"] == "head_position"
    assert first[
        "orientation_reference_position_series_name"
    ] == "body_position"
    assert first[
        "motor_encoding_parameters_sha256"
    ] == provenance_sha256(
        dict(table_specs.MANUSCRIPT_V1_MOTOR_ENCODING_PARAMETERS)
    )
    assert first[
        "motor_encoding_model_spec_sha256"
    ] == provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_MODEL_SPEC)
    )
    assert first[
        "motor_encoding_output_rule_sha256"
    ] == provenance_sha256(
        dict(table_specs.MOTOR_ENCODING_OUTPUT_RULE)
    )


def test_motor_encoding_selection_rejects_misaligned_position_sources() -> None:
    inputs = _motor_encoding_selection_inputs()
    inputs["position_rows"][1]["sample_count"] = 99

    with pytest.raises(ValueError, match="aligned sampling metadata"):
        _build_motor_encoding_selection(inputs)


def test_motor_encoding_make_requires_populate_transaction() -> None:
    """AnalysisNwbfile and MotorEncoding rows must share populate's transaction."""
    bundle, _, _ = _fake_bundle()
    result = bundle["motor_encoding"]
    result.connection = SimpleNamespace(in_transaction=False)
    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"motor_encoding_id": uuid.uuid4()})


def test_motor_encoding_registration_uses_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy validation, NWB registration, and result insert are transactional."""
    result_id = uuid.uuid4()
    selection = {
        "motor_encoding_id": result_id,
        "nwb_file_name": "L1420240611_.nwb",
    }
    events = []

    class Connection:
        in_transaction = False

        @property
        def transaction(self):
            connection = self

            class Transaction:
                def __enter__(self):
                    events.append("transaction_enter")
                    connection.in_transaction = True

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("transaction_exit")
                    connection.in_transaction = False

            return Transaction()

    connection = Connection()

    def register(**kwargs):
        assert connection.in_transaction
        assert kwargs["analysis_nwbfile_table"] is bundle["analysis_nwbfile"]
        events.append("register_hook")
        return {
            "analysis_file_name": "registered-motor-encoding.nwb",
            "selected_units_object_id": "selected-units-object-id",
            "dataset_index_object_id": "dataset-index-object-id",
            "coordinates_object_id": "coordinates-object-id",
            "nested_cv_arrays_object_id": "nested-cv-arrays-object-id",
            "full_refit_arrays_object_id": "full-refit-arrays-object-id",
            "provenance_object_id": "provenance-object-id",
            "artifact_schema_version": "1",
            "schema_version": "2",
            "n_units_input": 2,
            "n_units_eligible": 1,
            "n_units_valid": 1,
            "n_outer_folds_expected": 5,
            "n_outer_folds_valid": 5,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
            "selected_units_table_sha256": "b" * 64,
            "dataset_index_sha256": "c" * 64,
            "coordinates_sha256": "d" * 64,
            "nested_cv_arrays_sha256": "e" * 64,
            "full_refit_arrays_sha256": "f" * 64,
            "provenance_sha256": "1" * 64,
            "motor_encoding_sha256": "2" * 64,
            "legacy_artifact_provenance": {"source": "legacy"},
            "_created_artifact_paths": ["registered-motor-encoding.nwb"],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"motor_encoding_register_existing": register}
    )
    result = bundle["motor_encoding"]
    result.connection = connection
    row = result.register_existing(
        {"motor_encoding_id": result_id},
        source_nested_cv_path="old-nested.nc",
        source_full_refit_path="old-full-refit.nc",
    )
    assert events == [
        "transaction_enter",
        "register_hook",
        "transaction_exit",
    ]
    assert row["analysis_file_name"] == "registered-motor-encoding.nwb"


def test_dark_light_glm_parameters_and_selection_are_frozen() -> None:
    parameters = dict(
        table_specs.CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS
    )
    assert _validate_dark_light_glm_parameter_row(parameters) == parameters
    first = _build_dark_light_glm_selection(
        _dark_light_glm_selection_inputs()
    )
    second = _build_dark_light_glm_selection(
        _dark_light_glm_selection_inputs()
    )

    assert first == second
    assert first["dark_light_glm_id"].version == 5
    assert first["dark_epoch"] == "08_r4"
    assert first["light_epoch"] == "02_r1"
    assert first["dark_light_glm_parameters_sha256"] == provenance_sha256(
        parameters
    )
    assert first["dark_light_glm_output_rule_sha256"] == provenance_sha256(
        dict(table_specs.DARK_LIGHT_GLM_OUTPUT_RULE)
    )


def test_dark_light_glm_selection_checks_conditions_and_snapshots() -> None:
    inputs = _dark_light_glm_selection_inputs()
    inputs["epoch_rows"][1]["is_light"] = False
    with pytest.raises(ValueError, match="explicit light condition"):
        _build_dark_light_glm_selection(inputs)

    inputs = _dark_light_glm_selection_inputs()
    light_id = inputs["key"]["light_movement_firing_rate_id"]
    inputs["movement_selections"][light_id][
        "movement_parameters_sha256"
    ] = "e" * 64
    with pytest.raises(ValueError, match="frozen source snapshot"):
        _build_dark_light_glm_selection(inputs)

    inputs = _dark_light_glm_selection_inputs()
    light_id = inputs["key"]["light_movement_firing_rate_id"]
    inputs["movement_selections"][light_id][
        "position_series_name"
    ] = "body_position"
    inputs["position_rows"].append(
        {
            **inputs["position_rows"][1],
            "position_series_name": "body_position",
        }
    )
    with pytest.raises(ValueError, match="position_series_name"):
        _build_dark_light_glm_selection(inputs)


def test_dark_light_glm_selection_allows_terminal_movement_statuses() -> None:
    inputs = _dark_light_glm_selection_inputs()
    dark_id = inputs["key"]["dark_movement_firing_rate_id"]
    inputs["movement_results"][dark_id]["analysis_status"] = "no_movement"

    row = _build_dark_light_glm_selection(inputs)

    assert row["dark_movement_firing_rate_id"] == dark_id


def test_dark_light_legacy_resolver_requires_imported_unique_units() -> None:
    loaded = {
        "ts_group": {0: object(), 1: object()},
        "unit_ids": [
            {"spikesorting_merge_id": "merge-a", "unit_id": 10},
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        ],
        "unit_metadata": [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 10,
                "sorting_unit_id": 101,
            },
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 11,
                "sorting_unit_id": 102,
            },
        ],
        "member_provenance": [
            {
                "spikesorting_merge_id": "merge-a",
                "merge_parent": "ImportedSpikeSorting",
                "n_selected_units": 2,
            }
        ],
    }
    resolver = _legacy_dark_light_unit_identity_resolver(loaded)
    assert resolver["101"]["stable_unit_id"] == "merge-a:10"
    assert resolver["102"]["group_unit_id"] == "1"

    loaded["member_provenance"][0]["merge_parent"] = "CurationV1"
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        _legacy_dark_light_unit_identity_resolver(loaded)


def test_dark_light_registration_converts_legacy_bundle_to_analysis_nwb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parameters = dict(
        table_specs.CURRENT_V5_V1_DARK_LIGHT_GLM_PARAMETERS
    )
    selection = {
        "nwb_file_name": "L1420240611_.nwb",
        "dark_light_glm_id": uuid.uuid4(),
        "dark_light_glm_param_name": parameters[
            "dark_light_glm_param_name"
        ],
        "dark_light_glm_parameters_sha256": provenance_sha256(parameters),
        "dark_light_glm_output_rule_sha256": provenance_sha256(
            dict(table_specs.DARK_LIGHT_GLM_OUTPUT_RULE)
        ),
        "light_epoch": "02_r1",
        "dark_epoch": "08_r4",
    }
    context = {
        "selection": selection,
        "parameters": parameters,
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
    }
    monkeypatch.setattr(
        tables_module,
        "_load_dark_light_glm_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_dark_light_glm_spikes",
        lambda **kwargs: {"unit_ids": [], "member_provenance": []},
    )
    monkeypatch.setattr(
        tables_module,
        "_load_dark_light_glm_nwb_inputs",
        lambda **kwargs: {"graph_inputs": {}},
    )
    monkeypatch.setattr(
        tables_module,
        "_legacy_dark_light_unit_identity_resolver",
        lambda loaded: {},
    )

    from v1ca1.spyglass import dark_light_glm

    expected_parameters = {
        "schema_version": "5",
        "parameter_name": parameters["dark_light_glm_param_name"],
        "parameter_sha256": selection[
            "dark_light_glm_parameters_sha256"
        ],
        "output_rule_sha256": selection[
            "dark_light_glm_output_rule_sha256"
        ],
        **{
            field_name: value
            for field_name, value in parameters.items()
            if field_name != "dark_light_glm_param_name"
        },
    }
    register_calls = []
    monkeypatch.setattr(
        dark_light_glm,
        "register_existing_dark_light_glm_artifact",
        lambda **kwargs: (
            register_calls.append(kwargs)
            or {
                "parameters": expected_parameters,
                "legacy_artifact_provenance": {"source": "legacy"},
            }
        ),
    )
    writer_calls = []
    monkeypatch.setattr(
        tables_module,
        "_write_dark_light_glm_nwb",
        lambda **kwargs: writer_calls.append(kwargs)
        or {"analysis_file_name": "dark-light-analysis.nwb"},
    )
    analysis_nwbfile_table = object()

    row = _register_existing_dark_light_glm_row(
        key=selection,
        source_candidate_paths=[],
        source_selected_paths_by_model={},
        source_selection_summary_path=tmp_path / "summary.nc",
        parameters_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        position_table=object(),
        epoch_intervals_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        source_v1ca1_git_commit=None,
        source_spyglass_git_commit=None,
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    assert register_calls[0]["destination_path"] is None
    assert writer_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert writer_calls[0]["analysis_nwbfile_table"] is analysis_nwbfile_table
    assert row["analysis_file_name"] == "dark-light-analysis.nwb"


def _terminal_dark_light_glm_result():
    """Return one canonical no-unit DarkLightGLM result for table tests."""
    from v1ca1.spyglass import dark_light_glm

    result_id = uuid.uuid4()
    parameters = {
        **dark_light_glm.validate_dark_light_glm_parameters(
            basis_candidate_mode="spatial_bin_size_cm",
            basis_candidates=(2.0, 4.0),
            bin_sizes_s=(0.02,),
            ridges=(0.1,),
            n_folds=2,
            min_dark_firing_rate_hz=0.5,
            min_light_firing_rate_hz=0.5,
        ),
        "parameter_name": "test",
        "parameter_sha256": "a" * 64,
        "output_rule_sha256": "b" * 64,
    }
    result = dark_light_glm._terminal_result(
        metadata={
            "dark_light_glm_id": str(result_id),
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "light_epoch": "02_r1",
            "dark_epoch": "08_r4",
        },
        parameters=parameters,
        trajectory_length_cm=36.0,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0]),
        analysis_status="no_eligible_units",
    )
    return result, result_id


def test_dark_light_glm_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
) -> None:
    """The live writer registers seven selectively fetchable NWB objects."""
    from v1ca1.spyglass import dark_light_glm

    result, _result_id = _terminal_dark_light_glm_result()
    analysis_table = _TestAnalysisNwbfile(
        tmp_path / "dark-light-glm-analysis.nwb"
    )
    with pytest.raises(ValueError, match="analysis_nwbfile_table is required"):
        tables_module._write_dark_light_glm_nwb(
            nwb_file_name="L1420240102_.nwb",
            result=result,
            analysis_nwbfile_table=None,
        )
    row = tables_module._write_dark_light_glm_nwb(
        nwb_file_name="L1420240102_.nwb",
        result=result,
        analysis_nwbfile_table=analysis_table,
    )
    assert analysis_table.builder.registered is True
    assert row["analysis_file_name"] == "dark-light-glm-analysis.nwb"
    assert row["artifact_schema_version"] == (
        dark_light_glm.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["analysis_status"] == "no_eligible_units"
    assert {
        f"{name}_object_id"
        for name in tables_module._DARK_LIGHT_GLM_NWB_OBJECT_NAMES
    }.issubset(row)
    assert dark_light_glm.dark_light_glm_nwb_hashes(result).items() <= row.items()


def test_dark_light_glm_loader_uses_fetch_nwb_and_checks_hashes() -> None:
    """The live loader reconstructs the full search through fetch_nwb()."""
    from v1ca1.spyglass import dark_light_glm

    result, result_id = _terminal_dark_light_glm_result()
    objects = dark_light_glm.dark_light_glm_result_to_nwb_objects(result)

    class FetchNwbRelation:
        def __init__(self) -> None:
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [
                {
                    name: nwb_object.to_dataframe()
                    for name, nwb_object in objects.items()
                }
            ]

    relation = FetchNwbRelation()
    result_row = {
        "dark_light_glm_id": result_id,
        "artifact_schema_version": dark_light_glm.NWB_ARTIFACT_SCHEMA_VERSION,
        "selected_model_sha256_by_model": (
            dark_light_glm.dark_light_glm_selected_model_sha256s(result)
        ),
        **dark_light_glm.dark_light_glm_nwb_hashes(result),
    }
    loaded = tables_module._load_dark_light_glm_result(
        result_row=result_row,
        dark_light_glm_table=relation,
    )
    assert loaded["analysis_status"] == "no_eligible_units"
    assert relation.keys == [{"dark_light_glm_id": result_id}]
    with pytest.raises(ValueError, match="candidate_results_sha256"):
        tables_module._load_dark_light_glm_result(
            result_row={
                **result_row,
                "candidate_results_sha256": "0" * 64,
            },
            dark_light_glm_table=relation,
        )


def test_dark_light_glm_make_requires_populate_transaction() -> None:
    """Direct make cannot register a DarkLightGLM NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "dark_light_glm_compute": lambda **kwargs: pytest.fail(
                "DarkLightGLM computation must not start outside populate()."
            )
        }
    )
    result = bundle["dark_light_glm"]
    result.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"dark_light_glm_id": uuid.uuid4()})


def test_dark_light_glm_failed_insert_removes_new_analysis_nwb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed result insert removes only the newly written NWB file."""
    result_id = uuid.uuid4()
    artifact_path = tmp_path / f"{result_id}.nwb"
    retained_path = tmp_path / "preexisting.nc"
    retained_path.write_bytes(b"keep")

    def compute(**_kwargs):
        artifact_path.write_bytes(b"new")
        return {
            "analysis_file_name": artifact_path.name,
            "artifact_schema_version": "1",
            "schema_version": "5",
            "n_units": 2,
            "n_candidates": 4,
            "n_selected_models": 4,
            "analysis_status": "partial_valid",
            "selected_units_sha256": "a" * 64,
            "legacy_artifact_provenance": None,
            "_created_artifact_paths": [str(artifact_path)],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {"dark_light_glm_id": result_id},
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"dark_light_glm_compute": compute}
    )
    result = bundle["dark_light_glm"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make({"dark_light_glm_id": result_id})

    assert not artifact_path.exists()
    assert retained_path.read_bytes() == b"keep"


def _ripple_glm_selection_inputs() -> dict[str, Any]:
    """Return one complete RippleGLM selection with database-free intervals."""
    source_id = uuid.UUID("92222222-2222-5222-8222-222222222222")
    target_id = uuid.UUID("93333333-3333-5333-8333-333333333333")
    nwb_file_name = "L1420240102_.nwb"
    epoch = "04_s2"
    parameters = dict(
        table_specs.MANUSCRIPT_UNIT_VECTOR_RIPPLE_GLM_PARAMETERS
    )
    group_rows = {
        source_id: {
            "region_sorted_spikes_group_id": source_id,
            "nwb_file_name": nwb_file_name,
            "region_name": "ca1",
            "sorting_group_members_sha256": "a" * 64,
            "unit_filter_params_sha256": "b" * 64,
            "selected_units_sha256": "c" * 64,
            "n_units": 3,
        },
        target_id: {
            "region_sorted_spikes_group_id": target_id,
            "nwb_file_name": nwb_file_name,
            "region_name": "v1",
            "sorting_group_members_sha256": "d" * 64,
            "unit_filter_params_sha256": "e" * 64,
            "selected_units_sha256": "f" * 64,
            "n_units": 4,
        },
    }
    ripple_row = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "ripple_count": 5,
        "detector_zscore_threshold": 2.0,
        "speed_gated": True,
        "detection_parameters": {"speed_threshold": 4.0},
        "provenance_path": "scratch/ripple_detection_provenance",
        "provenance_object_id": "prov-id",
        "source_table_path": "intervals/ripples",
        "source_table_object_id": "table-id",
        "source_object_path": "intervals/ripples",
        "source_object_id": "object-id",
    }
    epoch_row = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "start_time": 0.0,
        "stop_time": 10.0,
    }
    ripple_table = pd.DataFrame(
        {
            "start_time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "end_time": [1.05, 2.05, 3.05, 4.05, 5.05],
            "epoch": [epoch] * 5,
        }
    )
    key = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "source_region_sorted_spikes_group_id": source_id,
        "target_region_sorted_spikes_group_id": target_id,
        "ripple_glm_param_name": parameters["ripple_glm_param_name"],
    }
    return {
        "key": key,
        "parameters": parameters,
        "group_rows": group_rows,
        "ripple_row": ripple_row,
        "epoch_row": epoch_row,
        "ripple_table": ripple_table,
        "epoch_interval": SimpleNamespace(start=[0.0], end=[10.0]),
    }


def _build_ripple_glm_selection(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build one RippleGLM selection from mutable fake inputs."""
    return _ripple_glm_selection_row(
        key=inputs["key"],
        ripples_table=_FakeRelation(inputs["ripple_row"]),
        epoch_intervals_table=_FakeRelation(inputs["epoch_row"]),
        region_sorted_spikes_group_table=_FakeKeyedRelation(
            "region_sorted_spikes_group_id",
            inputs["group_rows"],
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
        ripple_table=inputs["ripple_table"],
        epoch_interval=inputs["epoch_interval"],
    )


def test_ripple_glm_selection_freezes_events_groups_and_parameters() -> None:
    inputs = _ripple_glm_selection_inputs()
    first = _build_ripple_glm_selection(inputs)
    second = _build_ripple_glm_selection(_ripple_glm_selection_inputs())

    assert first == second
    assert first["ripple_glm_id"].version == 5
    assert first["source_region"] == "ca1"
    assert first["target_region"] == "v1"
    assert first["source_ripple_count"] == 5
    assert first["detector_zscore_threshold"] == pytest.approx(2.0)
    assert first["speed_gated"] is True
    assert first["n_selected_ripples"] == 5
    assert len(first["source_ripple_intervals_sha256"]) == 64
    assert len(first["ripple_provenance_sha256"]) == 64
    assert len(first["selected_ripple_events_sha256"]) == 64
    assert first["ripple_glm_parameters_sha256"] == provenance_sha256(
        inputs["parameters"]
    )
    from v1ca1.spyglass.ripple_glm import OUTPUT_RULE_SHA256

    assert first["ripple_glm_output_rule_sha256"] == OUTPUT_RULE_SHA256
    upstream = tables_module._ripple_glm_upstream_provenance(first)
    assert upstream["detector_zscore_threshold"] == pytest.approx(2.0)
    assert upstream["speed_gated"] is True

    changed_end = _ripple_glm_selection_inputs()
    changed_end["ripple_table"].loc[0, "end_time"] = 1.075
    assert _build_ripple_glm_selection(changed_end)["ripple_glm_id"] != first[
        "ripple_glm_id"
    ]

    changed_provenance = _ripple_glm_selection_inputs()
    changed_provenance["ripple_row"]["detection_parameters"] = {
        "speed_threshold": 3.0
    }
    assert _build_ripple_glm_selection(changed_provenance)[
        "ripple_glm_id"
    ] != first["ripple_glm_id"]


def test_ripple_glm_selection_rejects_wrong_region_or_nwb() -> None:
    inputs = _ripple_glm_selection_inputs()
    source_id = inputs["key"]["source_region_sorted_spikes_group_id"]
    inputs["group_rows"][source_id]["region_name"] = "v1"
    with pytest.raises(ValueError, match="source group.*ca1"):
        _build_ripple_glm_selection(inputs)

    inputs = _ripple_glm_selection_inputs()
    target_id = inputs["key"]["target_region_sorted_spikes_group_id"]
    inputs["group_rows"][target_id]["nwb_file_name"] = "other.nwb"
    with pytest.raises(ValueError, match="same NWB"):
        _build_ripple_glm_selection(inputs)


def test_ripple_glm_legacy_resolver_requires_imported_unique_units() -> None:
    loaded = {
        "ts_group": {0: object(), 1: object()},
        "unit_ids": [
            {"spikesorting_merge_id": "merge-a", "unit_id": 10},
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        ],
        "unit_metadata": [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 10,
                "sorting_unit_id": 101,
            },
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 11,
                "sorting_unit_id": 102,
            },
        ],
        "member_provenance": [
            {
                "spikesorting_merge_id": "merge-a",
                "merge_parent": "ImportedSpikeSorting",
                "n_selected_units": 2,
            }
        ],
    }
    resolver = _legacy_ripple_glm_unit_identity_resolver(
        loaded,
        role="source",
    )
    assert resolver["101"]["group_unit_id"] == "0"
    assert resolver["102"]["stable_unit_id"] == "merge-a:11"

    loaded["member_provenance"][0]["merge_parent"] = "CurationV1"
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        _legacy_ripple_glm_unit_identity_resolver(loaded, role="target")


def _terminal_ripple_glm_result() -> tuple[dict[str, Any], uuid.UUID]:
    """Return one canonical no-source RippleGLM result for table tests."""
    from v1ca1.spyglass import ripple_glm

    result_id = uuid.UUID("94444444-4444-5444-8444-444444444444")
    result = ripple_glm.compute_ripple_glm(
        ripple_glm_id=result_id,
        animal_name="L14",
        date="20240102",
        epoch="04_s2",
        ripple_table=pd.DataFrame(
            {
                "epoch": ["04_s2"] * 5,
                "start_time": [1.0, 2.0, 3.0, 4.0, 5.0],
                "end_time": [1.05, 2.05, 3.05, 4.05, 5.05],
            }
        ),
        epoch_interval=SimpleNamespace(start=[0.0], end=[10.0]),
        source_spikes={},
        source_stable_unit_ids=[],
        target_spikes={},
        target_stable_unit_ids=[],
        upstream_provenance={
            "detector_zscore_threshold": 2.0,
            "speed_gated": True,
        },
        parameter_name="test",
    )
    return result, result_id


def test_ripple_glm_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
) -> None:
    """The live writer registers six selectively fetchable NWB objects."""
    from v1ca1.spyglass import ripple_glm

    result, _result_id = _terminal_ripple_glm_result()
    analysis_table = _TestAnalysisNwbfile(
        tmp_path / "ripple-glm-analysis.nwb"
    )
    with pytest.raises(ValueError, match="analysis_nwbfile_table is required"):
        tables_module._write_ripple_glm_nwb(
            nwb_file_name="L1420240102_.nwb",
            result=result,
            analysis_nwbfile_table=None,
        )
    row = tables_module._write_ripple_glm_nwb(
        nwb_file_name="L1420240102_.nwb",
        result=result,
        analysis_nwbfile_table=analysis_table,
    )
    assert analysis_table.builder.registered is True
    assert row["analysis_file_name"] == "ripple-glm-analysis.nwb"
    assert row["artifact_schema_version"] == (
        ripple_glm.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["analysis_status"] == "no_source_units"
    assert {
        f"{name}_object_id"
        for name in tables_module._RIPPLE_GLM_NWB_OBJECT_NAMES
    }.issubset(row)
    assert ripple_glm.ripple_glm_nwb_hashes(result).items() <= row.items()


def test_ripple_glm_loader_uses_fetch_nwb_and_checks_hashes() -> None:
    """The live loader reconstructs complete RippleGLM data via fetch_nwb."""
    from v1ca1.spyglass import ripple_glm

    result, result_id = _terminal_ripple_glm_result()
    objects = ripple_glm.ripple_glm_result_to_nwb_objects(result)

    class FetchNwbRelation:
        def __init__(self) -> None:
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [
                {
                    name: nwb_object.to_dataframe()
                    for name, nwb_object in objects.items()
                }
            ]

    relation = FetchNwbRelation()
    result_row = {
        "ripple_glm_id": result_id,
        "artifact_schema_version": ripple_glm.NWB_ARTIFACT_SCHEMA_VERSION,
        **ripple_glm.ripple_glm_nwb_hashes(result),
    }
    loaded = tables_module._load_ripple_glm_result(
        result_row=result_row,
        ripple_glm_table=relation,
    )
    assert loaded["analysis_status"] == "no_source_units"
    assert relation.keys == [{"ripple_glm_id": result_id}]
    with pytest.raises(ValueError, match="target_results_sha256"):
        tables_module._load_ripple_glm_result(
            result_row={
                **result_row,
                "target_results_sha256": "0" * 64,
            },
            ripple_glm_table=relation,
        )


def test_ripple_glm_make_requires_populate_transaction() -> None:
    """Direct make cannot register a RippleGLM NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "ripple_glm_compute": lambda **kwargs: pytest.fail(
                "RippleGLM computation must not start outside populate()."
            )
        }
    )
    result = bundle["ripple_glm"]
    result.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"ripple_glm_id": uuid.uuid4()})


def test_ripple_glm_failed_insert_removes_new_analysis_nwb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed result insert removes only the newly written NWB file."""
    result_id = uuid.uuid4()
    artifact_path = tmp_path / f"{result_id}.nwb"
    retained_path = tmp_path / "preexisting.nc"
    retained_path.write_bytes(b"keep")

    def compute(**kwargs):
        assert kwargs["analysis_nwbfile_table"] is not None
        artifact_path.write_bytes(b"new")
        return {
            "analysis_file_name": artifact_path.name,
            "artifact_schema_version": "1",
            "schema_version": "1",
            "bundle_schema_version": "1",
            "n_source_units": 0,
            "n_target_units": 0,
            "n_source_units_in_fit": 0,
            "n_target_units_in_fit": 0,
            "n_valid_target_units": 0,
            "n_ripples": 0,
            "selected_ripple_events_sha256": "a" * 64,
            "selected_units_sha256": "b" * 64,
            "analysis_status": "no_source_units",
            "legacy_artifact_provenance": None,
            "_created_artifact_paths": [str(artifact_path)],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {"ripple_glm_id": result_id},
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"ripple_glm_compute": compute}
    )
    result = bundle["ripple_glm"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make({"ripple_glm_id": result_id})

    assert not artifact_path.exists()
    assert retained_path.read_bytes() == b"keep"


def test_ripple_glm_compute_writes_analysis_nwb_instead_of_legacy_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The live compute bridge sends canonical science to the NWB writer."""
    inputs = _ripple_glm_selection_inputs()
    selection = _build_ripple_glm_selection(inputs)
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "ripple_table": inputs["ripple_table"],
        "epoch_interval": inputs["epoch_interval"],
        "animal_name": "L14",
        "date": "20240102",
    }
    loaded = {
        "source": {"ts_group": object(), "unit_ids": object()},
        "target": {"ts_group": object(), "unit_ids": object()},
    }
    monkeypatch.setattr(
        tables_module,
        "_load_ripple_glm_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_ripple_glm_spikes",
        lambda **kwargs: loaded,
    )
    from v1ca1.spyglass import ripple_glm

    compute_calls = []
    writer_calls = []
    computed = {
        "upstream_provenance": tables_module._ripple_glm_upstream_provenance(
            selection
        ),
        "selected_ripple_events_sha256": selection[
            "selected_ripple_events_sha256"
        ],
    }

    def compute(**kwargs):
        compute_calls.append(kwargs)
        return computed

    def write_nwb(**kwargs):
        writer_calls.append(kwargs)
        return {"analysis_file_name": "ripple-glm-analysis.nwb"}

    monkeypatch.setattr(ripple_glm, "compute_ripple_glm", compute)
    monkeypatch.setattr(tables_module, "_write_ripple_glm_nwb", write_nwb)
    analysis_nwbfile_table = object()
    row = _make_ripple_glm_row(
        key=selection,
        parameters_table=object(),
        ripples_table=object(),
        epoch_intervals_table=object(),
        region_sorted_spikes_group_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    assert compute_calls[0]["source_spikes"] is loaded["source"]["ts_group"]
    assert compute_calls[0]["target_spikes"] is loaded["target"]["ts_group"]
    assert writer_calls == [
        {
            "nwb_file_name": selection["nwb_file_name"],
            "result": computed,
            "analysis_nwbfile_table": analysis_nwbfile_table,
        }
    ]
    assert row["analysis_file_name"] == "ripple-glm-analysis.nwb"


def test_ripple_glm_registration_passes_frozen_events_and_role_resolvers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _ripple_glm_selection_inputs()
    selection = _build_ripple_glm_selection(inputs)
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "ripple_table": inputs["ripple_table"],
        "epoch_interval": inputs["epoch_interval"],
        "animal_name": "L14",
        "date": "20240102",
    }
    loaded = {
        "source": {
            "ts_group": {0: object()},
            "unit_ids": [
                {"spikesorting_merge_id": "ca1-merge", "unit_id": 10}
            ],
            "unit_metadata": [
                {
                    "spikesorting_merge_id": "ca1-merge",
                    "unit_id": 10,
                    "sorting_unit_id": 101,
                }
            ],
            "member_provenance": [
                {
                    "spikesorting_merge_id": "ca1-merge",
                    "merge_parent": "ImportedSpikeSorting",
                    "n_selected_units": 1,
                }
            ],
        },
        "target": {
            "ts_group": {0: object()},
            "unit_ids": [
                {"spikesorting_merge_id": "v1-merge", "unit_id": 20}
            ],
            "unit_metadata": [
                {
                    "spikesorting_merge_id": "v1-merge",
                    "unit_id": 20,
                    "sorting_unit_id": 201,
                }
            ],
            "member_provenance": [
                {
                    "spikesorting_merge_id": "v1-merge",
                    "merge_parent": "ImportedSpikeSorting",
                    "n_selected_units": 1,
                }
            ],
        },
    }
    monkeypatch.setattr(
        tables_module,
        "_load_ripple_glm_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_ripple_glm_spikes",
        lambda **kwargs: loaded,
    )
    from v1ca1.spyglass import ripple_glm

    calls = []
    writer_calls = []

    def register_existing(**kwargs):
        calls.append(kwargs)
        return {
            "upstream_provenance": tables_module._ripple_glm_upstream_provenance(
                selection
            ),
            "selected_ripple_events_sha256": selection[
                "selected_ripple_events_sha256"
            ],
            "n_source_units": 1,
            "n_target_units": 1,
            "n_source_units_in_fit": 1,
            "n_target_units_in_fit": 1,
            "n_valid_target_units": 1,
            "n_ripples": selection["n_selected_ripples"],
            "selected_units_sha256": "a" * 64,
            "analysis_status": "valid",
            "artifact_origin": "registered_existing",
            "legacy_artifact_provenance": {"source": "legacy"},
        }

    def write_nwb(**kwargs):
        writer_calls.append(kwargs)
        return {
            "analysis_file_name": "ripple-glm-analysis.nwb",
            "legacy_artifact_provenance": {"source": "legacy"},
        }

    monkeypatch.setattr(
        ripple_glm,
        "register_existing_ripple_glm_artifact",
        register_existing,
    )
    monkeypatch.setattr(tables_module, "_write_ripple_glm_nwb", write_nwb)
    analysis_nwbfile_table = object()
    row = _register_existing_ripple_glm_row(
        key=selection,
        source_result_path=tmp_path / "legacy.nc",
        parameters_table=object(),
        ripples_table=object(),
        epoch_intervals_table=object(),
        region_sorted_spikes_group_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        source_v1ca1_git_commit="v1",
        source_spyglass_git_commit="sg",
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["expected_selected_ripple_events_sha256"] == selection[
        "selected_ripple_events_sha256"
    ]
    assert call["source_sorting_type"] == "ImportedSpikeSorting"
    assert call["target_sorting_type"] == "ImportedSpikeSorting"
    assert call["source_legacy_unit_identity_resolver"]["101"][
        "group_unit_id"
    ] == "0"
    assert call["target_legacy_unit_identity_resolver"]["201"][
        "stable_unit_id"
    ] == "v1-merge:20"
    assert call["destination_path"] is None
    assert writer_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert writer_calls[0]["analysis_nwbfile_table"] is (
        analysis_nwbfile_table
    )
    assert row["legacy_artifact_provenance"] == {"source": "legacy"}


def _ripple_cross_region_xcorr_selection_inputs() -> dict[str, Any]:
    """Return one complete exact-ripple xcorr selection input set."""
    inputs = _ripple_glm_selection_inputs()
    parameters = dict(table_specs.MANUSCRIPT_RIPPLE_CROSS_REGION_XCORR_PARAMETERS)
    key = {
        name: value
        for name, value in inputs["key"].items()
        if name != "ripple_glm_param_name"
    }
    key["ripple_cross_region_xcorr_param_name"] = parameters[
        "ripple_cross_region_xcorr_param_name"
    ]
    inputs.update({"key": key, "parameters": parameters})
    return inputs


def _build_ripple_cross_region_xcorr_selection(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build one xcorr selection from mutable database-free inputs."""
    return _ripple_cross_region_xcorr_selection_row(
        key=inputs["key"],
        ripples_table=_FakeRelation(inputs["ripple_row"]),
        epoch_intervals_table=_FakeRelation(inputs["epoch_row"]),
        region_sorted_spikes_group_table=_FakeKeyedRelation(
            "region_sorted_spikes_group_id",
            inputs["group_rows"],
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
        ripple_table=inputs["ripple_table"],
    )


def _terminal_ripple_cross_region_xcorr_result() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Return one exact-selection terminal result without xcorr computation."""
    from v1ca1.spyglass import ripple_cross_region_xcorr

    inputs = _ripple_cross_region_xcorr_selection_inputs()
    selection = _build_ripple_cross_region_xcorr_selection(inputs)
    parameters = inputs["parameters"]
    result = ripple_cross_region_xcorr.compute_ripple_cross_region_xcorr(
        ripple_cross_region_xcorr_id=selection[
            "ripple_cross_region_xcorr_id"
        ],
        animal_name="L14",
        date="20240102",
        epoch=str(selection["epoch"]),
        ripple_table=inputs["ripple_table"],
        ca1_spikes={},
        ca1_stable_unit_ids=[],
        v1_spikes={},
        v1_stable_unit_ids=[],
        upstream_provenance=(
            tables_module._ripple_cross_region_xcorr_upstream_provenance(
                selection
            )
        ),
        expected_selected_ripple_intervals_sha256=selection[
            "selected_ripple_intervals_sha256"
        ],
        parameter_name=parameters[
            "ripple_cross_region_xcorr_param_name"
        ],
        parameter_sha256=selection[
            "ripple_cross_region_xcorr_parameters_sha256"
        ],
        output_rule_sha256=selection[
            "ripple_cross_region_xcorr_output_rule_sha256"
        ],
        **tables_module._ripple_cross_region_xcorr_parameter_kwargs(
            parameters
        ),
    )
    return result, selection, parameters


def test_ripple_cross_region_xcorr_selection_freezes_exact_inputs() -> None:
    inputs = _ripple_cross_region_xcorr_selection_inputs()
    first = _build_ripple_cross_region_xcorr_selection(inputs)
    second = _build_ripple_cross_region_xcorr_selection(
        _ripple_cross_region_xcorr_selection_inputs()
    )

    assert first == second
    assert first["ripple_cross_region_xcorr_id"].version == 5
    assert first["source_region"] == "ca1"
    assert first["target_region"] == "v1"
    assert first["source_ripple_count"] == 5
    assert first["detector_zscore_threshold"] == pytest.approx(2.0)
    assert first["speed_gated"] is True
    assert len(first["selected_ripple_intervals_sha256"]) == 64
    assert first["ripple_cross_region_xcorr_parameters_sha256"] == (
        provenance_sha256(inputs["parameters"])
    )
    from v1ca1.spyglass.ripple_cross_region_xcorr import OUTPUT_RULE_SHA256

    assert first["ripple_cross_region_xcorr_output_rule_sha256"] == (
        OUTPUT_RULE_SHA256
    )
    upstream = tables_module._ripple_cross_region_xcorr_upstream_provenance(first)
    assert upstream["detector_zscore_threshold"] == pytest.approx(2.0)
    assert upstream["speed_gated"] is True

    changed_end = _ripple_cross_region_xcorr_selection_inputs()
    changed_end["ripple_table"].loc[0, "end_time"] = 1.075
    assert _build_ripple_cross_region_xcorr_selection(changed_end)[
        "ripple_cross_region_xcorr_id"
    ] != first["ripple_cross_region_xcorr_id"]

    changed_provenance = _ripple_cross_region_xcorr_selection_inputs()
    changed_provenance["ripple_row"]["detection_parameters"] = {
        "speed_threshold": 3.0
    }
    assert _build_ripple_cross_region_xcorr_selection(changed_provenance)[
        "ripple_cross_region_xcorr_id"
    ] != first["ripple_cross_region_xcorr_id"]


def test_ripple_cross_region_xcorr_selection_rejects_region_or_nwb_mismatch() -> None:
    inputs = _ripple_cross_region_xcorr_selection_inputs()
    source_id = inputs["key"]["source_region_sorted_spikes_group_id"]
    inputs["group_rows"][source_id]["region_name"] = "v1"
    with pytest.raises(ValueError, match="source group.*ca1"):
        _build_ripple_cross_region_xcorr_selection(inputs)

    inputs = _ripple_cross_region_xcorr_selection_inputs()
    target_id = inputs["key"]["target_region_sorted_spikes_group_id"]
    inputs["group_rows"][target_id]["nwb_file_name"] = "other.nwb"
    with pytest.raises(ValueError, match="same NWB"):
        _build_ripple_cross_region_xcorr_selection(inputs)


def test_ripple_cross_region_xcorr_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
) -> None:
    """The live writer registers six selectively fetchable NWB objects."""
    from v1ca1.spyglass import ripple_cross_region_xcorr

    result, _selection, _parameters = (
        _terminal_ripple_cross_region_xcorr_result()
    )
    analysis_table = _TestAnalysisNwbfile(tmp_path / "xcorr-analysis.nwb")
    with pytest.raises(ValueError, match="analysis_nwbfile_table is required"):
        _write_ripple_cross_region_xcorr_nwb(
            nwb_file_name="L1420240102_.nwb",
            result=result,
            analysis_nwbfile_table=None,
        )
    row = _write_ripple_cross_region_xcorr_nwb(
        nwb_file_name="L1420240102_.nwb",
        result=result,
        analysis_nwbfile_table=analysis_table,
    )
    assert analysis_table.builder.registered is True
    assert Path(analysis_table.builder.path).is_file()
    assert row["analysis_file_name"] == "xcorr-analysis.nwb"
    assert row["artifact_schema_version"] == (
        ripple_cross_region_xcorr.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["analysis_status"] == "no_ca1_units"
    assert {
        "ca1_units_object_id",
        "v1_units_object_id",
        "pair_xcorr_object_id",
        "lag_axis_object_id",
        "ripple_support_object_id",
        "provenance_object_id",
    }.issubset(row)
    assert ripple_cross_region_xcorr.ripple_cross_region_xcorr_nwb_hashes(
        result
    ).items() <= row.items()


def test_ripple_cross_region_xcorr_loader_uses_fetch_nwb_and_checks_hashes(
) -> None:
    """The live loader reconstructs all science through fetch_nwb()."""
    from v1ca1.spyglass import ripple_cross_region_xcorr

    result, selection, parameters = _terminal_ripple_cross_region_xcorr_result()
    objects = ripple_cross_region_xcorr.ripple_cross_region_xcorr_result_to_nwb_objects(
        result
    )

    class FetchNwbRelation:
        def __init__(self) -> None:
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [
                {
                    name: nwb_object.to_dataframe()
                    for name, nwb_object in objects.items()
                }
            ]

    relation = FetchNwbRelation()
    hashes = ripple_cross_region_xcorr.ripple_cross_region_xcorr_nwb_hashes(
        result
    )
    result_row = {
        "ripple_cross_region_xcorr_id": selection[
            "ripple_cross_region_xcorr_id"
        ],
        "analysis_file_name": "xcorr-analysis.nwb",
        "artifact_schema_version": (
            ripple_cross_region_xcorr.NWB_ARTIFACT_SCHEMA_VERSION
        ),
        **hashes,
        **{
            field_name: result[field_name]
            for field_name in (
                "n_ripples",
                "ripple_duration_s",
                "n_ca1_units",
                "n_v1_units",
                "n_ca1_units_in_xcorr",
                "n_v1_units_in_xcorr",
                "n_pairs",
                "n_valid_pairs",
                "selected_ripple_intervals_sha256",
                "analysis_status",
                "artifact_origin",
                "legacy_artifact_provenance",
            )
        },
    }
    loaded = _load_ripple_cross_region_xcorr_result(
        result_row=result_row,
        result_table=relation,
        selection_row=selection,
        parameters_row=parameters,
        animal_name="L14",
        date="20240102",
    )
    assert loaded["analysis_status"] == "no_ca1_units"
    assert relation.keys == [
        {
            "ripple_cross_region_xcorr_id": selection[
                "ripple_cross_region_xcorr_id"
            ]
        }
    ]
    with pytest.raises(ValueError, match="lag_axis_sha256"):
        _load_ripple_cross_region_xcorr_result(
            result_row={**result_row, "lag_axis_sha256": "0" * 64},
            result_table=relation,
            selection_row=selection,
            parameters_row=parameters,
            animal_name="L14",
            date="20240102",
        )


class _TestIntervals:
    """Minimal Pynapple-like interval bounds for table-layer tests."""

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)


def test_movement_result_loader_uses_fetch_nwb_and_checks_object_hashes() -> None:
    """The shared downstream loader resolves both objects through fetch_nwb."""
    import pynapple as nap

    from v1ca1.spyglass import movement

    table = movement._build_movement_table(
        identity_rows=[
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "1",
                "stable_unit_id": "merge-a:1",
                "group_unit_id": 0,
            }
        ],
        animal_name="L14",
        date="20240102",
        region="v1",
        epoch="02_r1",
        movement_spike_counts=np.asarray([2], dtype=np.int64),
        movement_duration_s=1.0,
        movement_firing_rates_hz=np.asarray([2.0]),
        firing_rate_status="valid",
        position_sample_count=20,
        finite_position_sample_count=20,
        finite_speed_sample_count=20,
        movement_interval_count=1,
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )
    intervals = nap.IntervalSet(
        start=np.asarray([10.0]),
        end=np.asarray([11.0]),
        time_units="s",
    )

    class FetchNwbRelation:
        def __init__(self) -> None:
            self.keys = []
            self.fetch_count = 0

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            self.fetch_count += 1
            return [
                {
                    "movement_firing_rate": table.copy(),
                    "movement_intervals": (
                        movement.movement_interval_set_to_time_intervals(
                            intervals
                        ).to_dataframe()
                    ),
                }
            ]

    relation = FetchNwbRelation()
    result_row = {
        "movement_firing_rate_id": uuid.uuid4(),
        "analysis_file_name": "movement-analysis.nwb",
        "movement_firing_rate_object_id": "rate-object-id",
        "movement_intervals_object_id": "interval-object-id",
        "movement_firing_rate_sha256": (
            movement.movement_firing_rate_table_sha256(table)
        ),
        "movement_intervals_sha256": (
            movement.movement_interval_set_sha256(intervals)
        ),
        "artifact_schema_version": movement.NWB_ARTIFACT_SCHEMA_VERSION,
        "n_units": 1,
        "n_valid_units": 1,
        "n_units_with_spikes": 1,
        "movement_interval_count": 1,
        "movement_duration_s": 1.0,
        "analysis_status": "valid",
    }

    loaded = tables_module._load_movement_result_artifacts(
        result_row=result_row,
        movement_firing_rate_table=relation,
    )

    pd.testing.assert_frame_equal(loaded["table"], table)
    assert movement.movement_interval_summary(loaded["movement_intervals"]) == (
        1,
        1.0,
    )
    assert relation.keys == [
        {"movement_firing_rate_id": result_row["movement_firing_rate_id"]}
    ]
    assert relation.fetch_count == 1

    tampered = {**result_row, "movement_firing_rate_sha256": "0" * 64}
    with pytest.raises(ValueError, match="movement_firing_rate_sha256"):
        tables_module._load_movement_result_artifacts(
            result_row=tampered,
            movement_firing_rate_table=relation,
        )


def _ripple_cross_region_xcorr_loaded_spikes() -> dict[str, dict[str, Any]]:
    """Return minimal separate imported CA1 and V1 sorting groups."""
    return {
        role: {
            "ts_group": {0: object()},
            "unit_ids": [
                {"spikesorting_merge_id": merge_id, "unit_id": unit_id}
            ],
            "unit_metadata": [
                {
                    "spikesorting_merge_id": merge_id,
                    "unit_id": unit_id,
                    "sorting_unit_id": sorting_unit_id,
                }
            ],
            "member_provenance": [
                {
                    "spikesorting_merge_id": merge_id,
                    "merge_parent": "ImportedSpikeSorting",
                    "n_selected_units": 1,
                }
            ],
        }
        for role, merge_id, unit_id, sorting_unit_id in (
            ("source", "ca1-merge", 10, 101),
            ("target", "v1-merge", 20, 201),
        )
    }


def test_ripple_cross_region_xcorr_registration_uses_separate_imported_resolvers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _ripple_cross_region_xcorr_selection_inputs()
    selection = _build_ripple_cross_region_xcorr_selection(inputs)
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "ripple_table": inputs["ripple_table"],
        "animal_name": "L14",
        "date": "20240102",
    }
    loaded = _ripple_cross_region_xcorr_loaded_spikes()
    monkeypatch.setattr(
        tables_module,
        "_load_ripple_cross_region_xcorr_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_ripple_glm_spikes",
        lambda **kwargs: loaded,
    )
    from v1ca1.spyglass import ripple_cross_region_xcorr

    calls = []
    writer_calls = []

    def register_existing(**kwargs):
        calls.append(kwargs)
        return {
            "upstream_provenance": (
                tables_module._ripple_cross_region_xcorr_upstream_provenance(
                    selection
                )
            ),
            "selected_ripple_intervals_sha256": selection[
                "selected_ripple_intervals_sha256"
            ],
            "n_ripples": 5,
            "ripple_duration_s": 0.25,
            "n_ca1_units": 1,
            "n_v1_units": 1,
            "n_ca1_units_in_xcorr": 1,
            "n_v1_units_in_xcorr": 1,
            "n_pairs": 1,
            "n_valid_pairs": 1,
            "ca1_units_sha256": "a" * 64,
            "v1_units_sha256": "b" * 64,
            "summary_sha256": "c" * 64,
            "analysis_status": "valid",
            "artifact_origin": "registered_existing",
            "legacy_artifact_provenance": {"source": "legacy"},
        }

    monkeypatch.setattr(
        ripple_cross_region_xcorr,
        "register_existing_ripple_cross_region_xcorr_artifact",
        register_existing,
    )
    monkeypatch.setattr(
        tables_module,
        "_write_ripple_cross_region_xcorr_nwb",
        lambda **kwargs: (
            writer_calls.append(kwargs)
            or {"legacy_artifact_provenance": {"source": "legacy"}}
        ),
    )
    analysis_nwbfile_table = object()
    row = _register_existing_ripple_cross_region_xcorr_row(
        key=selection,
        source_ca1_unit_filter_path=tmp_path / "ca1.parquet",
        source_v1_unit_filter_path=tmp_path / "v1.parquet",
        source_summary_path=tmp_path / "summary.parquet",
        source_result_path=tmp_path / "xcorr.nc",
        parameters_table=object(),
        ripples_table=object(),
        epoch_intervals_table=object(),
        region_sorted_spikes_group_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        source_v1ca1_git_commit="v1",
        source_spyglass_git_commit="sg",
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    call = calls[0]
    assert call["expected_selected_ripple_intervals_sha256"] == selection[
        "selected_ripple_intervals_sha256"
    ]
    assert call["ca1_sorting_type"] == "ImportedSpikeSorting"
    assert call["v1_sorting_type"] == "ImportedSpikeSorting"
    assert call["destination_path"] is None
    assert call["ca1_legacy_identity_resolver"]([101])[0][
        "stable_unit_id"
    ] == "ca1-merge:10"
    assert call["v1_legacy_identity_resolver"]([201])[0][
        "stable_unit_id"
    ] == "v1-merge:20"
    assert row["legacy_artifact_provenance"] == {"source": "legacy"}
    assert writer_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert writer_calls[0]["analysis_nwbfile_table"] is analysis_nwbfile_table


def test_ripple_cross_region_xcorr_resolver_rejects_nonimported_groups() -> None:
    loaded = _ripple_cross_region_xcorr_loaded_spikes()["source"]
    loaded["member_provenance"][0]["merge_parent"] = "CurationV1"
    with pytest.raises(ValueError, match="RippleCrossRegionXCorr.*Imported"):
        _legacy_ripple_cross_region_xcorr_identity_resolver(loaded, role="source")


def test_swap_glm_parameters_and_selection_freeze_upstream_artifacts() -> None:
    parameters = dict(table_specs.DEFAULT_SWAP_GLM_PARAMETERS)
    assert _validate_swap_glm_parameter_row(parameters) == parameters
    inputs = _swap_glm_selection_inputs()

    first = _build_swap_glm_selection(inputs)
    second = _build_swap_glm_selection(_swap_glm_selection_inputs())

    assert first == second
    assert first["swap_glm_id"].version == 5
    assert first["light_test_epoch"] == "06_r3"
    assert first["dark_condition"] == "dark"
    assert first["light_train_condition"] == "AB"
    assert first["light_test_condition"] == "BA"
    assert first["dark_light_glm_sha256"] == "1" * 64
    assert first["dark_light_selected_model_sha256_by_model"] == inputs[
        "dark_light_snapshot"
    ]["selected_model_sha256_by_model"]
    assert first["dark_light_parameter_sha256"] == inputs[
        "dark_light_snapshot"
    ]["parameter_sha256"]
    assert first["upstream_analysis_status"] == "valid"
    assert first["swap_glm_parameters_sha256"] == provenance_sha256(
        parameters
    )
    assert first["swap_glm_output_rule_sha256"] == provenance_sha256(
        dict(table_specs.SWAP_GLM_OUTPUT_RULE)
    )


def test_swap_glm_selection_requires_shared_movement_definition() -> None:
    inputs = _swap_glm_selection_inputs()
    movement_id = inputs["key"]["light_test_movement_firing_rate_id"]
    inputs["movement_selections"][movement_id][
        "movement_parameters_sha256"
    ] = "f" * 64

    with pytest.raises(ValueError, match="frozen source snapshot"):
        _build_swap_glm_selection(inputs)

    inputs = _swap_glm_selection_inputs()
    movement_id = inputs["key"]["light_test_movement_firing_rate_id"]
    inputs["movement_selections"][movement_id][
        "position_series_name"
    ] = "body_position"
    with pytest.raises(ValueError, match="position_series_name"):
        _build_swap_glm_selection(inputs)


def test_swap_glm_selection_checks_heldout_condition_and_laps() -> None:
    inputs = _swap_glm_selection_inputs()
    inputs["epoch_rows"][-1]["condition"] = "AB"
    with pytest.raises(ValueError, match="conditions must differ"):
        _build_swap_glm_selection(inputs)

    inputs = _swap_glm_selection_inputs()
    inputs["trajectory_rows"][0]["interval_count"] = 0
    with pytest.raises(ValueError, match="at least one held-out lap"):
        _build_swap_glm_selection(inputs)


def test_swap_glm_selection_allows_no_valid_position_terminal() -> None:
    inputs = _swap_glm_selection_inputs()
    movement_id = inputs["key"]["light_test_movement_firing_rate_id"]
    inputs["movement_results"][movement_id][
        "analysis_status"
    ] = "no_valid_position"

    row = _build_swap_glm_selection(inputs)

    assert row["light_test_movement_firing_rate_id"] == movement_id


def test_swap_glm_context_loads_no_valid_position_movement_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The terminal movement status is recovered from the saved artifact."""
    import numpy as np
    import pynapple as nap

    from v1ca1.spyglass import movement

    inputs = _swap_glm_selection_inputs()
    selection = _build_swap_glm_selection(inputs)
    movement_id = selection["light_test_movement_firing_rate_id"]
    n_units = int(inputs["region_row"]["n_units"])
    identity_rows = [
        {
            "spikesorting_merge_id": "merge-a",
            "unit_id": str(unit_id),
            "stable_unit_id": f"merge-a:{unit_id}",
            "group_unit_id": unit_id,
        }
        for unit_id in range(n_units)
    ]
    movement_table = movement._build_movement_table(
        identity_rows=identity_rows,
        animal_name="L14",
        date="20240102",
        region="v1",
        epoch=selection["light_test_epoch"],
        movement_spike_counts=np.zeros(n_units, dtype=np.int64),
        movement_duration_s=0.0,
        movement_firing_rates_hz=np.full(n_units, np.nan),
        firing_rate_status="no_valid_position",
        position_sample_count=1,
        finite_position_sample_count=0,
        finite_speed_sample_count=0,
        movement_interval_count=0,
        speed_threshold_cm_s=4.0,
        speed_smoothing_sigma_s=0.1,
    )
    empty_intervals = nap.IntervalSet(
        start=np.array([], dtype=float),
        end=np.array([], dtype=float),
        time_units="s",
    )
    movement_result = {
        "movement_firing_rate_id": movement_id,
        "analysis_file_name": "movement-analysis.nwb",
        "movement_firing_rate_object_id": "rate-object-id",
        "movement_intervals_object_id": "interval-object-id",
        "movement_firing_rate_sha256": (
            movement.movement_firing_rate_table_sha256(movement_table)
        ),
        "movement_intervals_sha256": (
            movement.movement_interval_set_sha256(empty_intervals)
        ),
        "artifact_schema_version": movement.NWB_ARTIFACT_SCHEMA_VERSION,
        "n_units": n_units,
        "n_valid_units": 0,
        "n_units_with_spikes": 0,
        "movement_interval_count": 0,
        "movement_duration_s": 0.0,
        "analysis_status": "no_valid_position",
        "selected_units_sha256": inputs["region_row"][
            "selected_units_sha256"
        ],
    }
    movement_selections = dict(inputs["movement_selections"])
    movement_parameters = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    movement_parameters_sha256 = provenance_sha256(movement_parameters)
    for movement_selection in movement_selections.values():
        movement_selection["movement_parameters_sha256"] = (
            movement_parameters_sha256
        )
    test_epoch_row = next(
        row
        for row in inputs["epoch_rows"]
        if row["epoch"] == selection["light_test_epoch"]
    )
    test_epoch_row.update({"start_time": 10.0, "stop_time": 20.0})
    monkeypatch.setattr(
        tables_module,
        "_load_swap_dark_light_snapshot",
        lambda **kwargs: inputs["dark_light_snapshot"],
    )
    monkeypatch.setattr(
        tables_module,
        "_fetch_movement_result_nwb_objects",
        lambda table, key: {
            "movement_firing_rate": movement_table,
            "movement_intervals": (
                movement.movement_interval_set_to_time_intervals(
                    empty_intervals
                )
            ),
        },
    )

    context = tables_module._load_swap_glm_context(
        key=selection,
        parameters_table=_FakeRelation(inputs["parameters"]),
        dark_light_glm_table=object(),
        dark_light_glm_selection_table=_FakeRelation(
            inputs["dark_light_selection"]
        ),
        region_sorted_spikes_group_table=_FakeRelation(
            inputs["region_row"]
        ),
        movement_firing_rate_table=_FakeRelation(movement_result),
        movement_firing_rate_selection_table=_FakeKeyedRelation(
            "movement_firing_rate_id",
            movement_selections,
        ),
        movement_parameters_table=_FakeRelation(movement_parameters),
        epoch_intervals_table=_FakeRowsRelation(inputs["epoch_rows"]),
        trajectory_intervals_table=_FakeRowsRelation(
            inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeRowsRelation(inputs["graph_rows"]),
        session_table=_FakeRelation(
            {
                "subject_id": "L14",
                "session_start_time": datetime(2024, 1, 2, 12, 30),
            }
        ),
    )

    assert context["movement"]["analysis_status"] == "no_valid_position"
    assert movement.movement_interval_summary(
        context["movement"]["movement_intervals"]
    ) == (0, 0.0)


def test_swap_glm_legacy_resolver_requires_imported_unique_units() -> None:
    loaded = {
        "ts_group": {0: object(), 1: object()},
        "unit_ids": [
            {"spikesorting_merge_id": "merge-a", "unit_id": 10},
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        ],
        "unit_metadata": [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 10,
                "sorting_unit_id": 101,
            },
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 11,
                "sorting_unit_id": 102,
            },
        ],
        "member_provenance": [
            {
                "spikesorting_merge_id": "merge-a",
                "merge_parent": "ImportedSpikeSorting",
                "n_selected_units": 2,
            }
        ],
    }
    resolver = _legacy_swap_glm_unit_identity_resolver(loaded)
    assert resolver["101"]["stable_unit_id"] == "merge-a:10"
    assert resolver["102"]["group_unit_id"] == "1"

    loaded["member_provenance"][0]["merge_parent"] = "CurationV1"
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        _legacy_swap_glm_unit_identity_resolver(loaded)


def _terminal_swap_glm_result():
    """Return one canonical no-unit SwapGLM result for NWB table tests."""
    from v1ca1.spyglass import swap_glm

    inputs = _swap_glm_selection_inputs()
    selection = _build_swap_glm_selection(inputs)
    metadata = {
        "swap_glm_id": str(selection["swap_glm_id"]),
        "animal_name": "L14",
        "date": "20240102",
        "region": "v1",
        "dark_epoch": selection["dark_epoch"],
        "light_train_epoch": selection["light_train_epoch"],
        "light_test_epoch": selection["light_test_epoch"],
    }
    parameters = {
        "parameter_name": inputs["parameters"]["swap_glm_param_name"],
        "parameter_sha256": selection["swap_glm_parameters_sha256"],
        "output_rule_sha256": selection["swap_glm_output_rule_sha256"],
        "swap_light_offset": inputs["parameters"]["swap_light_offset"],
        "observed_spatial_bin_size_cm": inputs["parameters"][
            "observed_spatial_bin_size_cm"
        ],
    }
    upstream = {
        "dark_light_glm_id": str(selection["dark_light_glm_id"]),
        "dark_light_glm_sha256": selection[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": selection[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": selection[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": selection[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": selection["upstream_analysis_status"],
    }
    selected_units = pd.DataFrame(columns=swap_glm.SELECTED_UNIT_COLUMNS)
    dataset = swap_glm._terminal_dataset(
        metadata=metadata,
        unit_ids=np.asarray([], dtype=str),
        segment_edges=np.linspace(0.0, 1.0, 4),
        parameters=parameters,
        upstream_provenance=upstream,
        analysis_status="no_units",
    )
    result = swap_glm.validate_swap_glm_result(
        {
            "metadata": metadata,
            "parameters": parameters,
            "upstream_provenance": upstream,
            "selected_units": selected_units,
            "dataset": dataset,
            "analysis_status": "no_units",
            "artifact_origin": "computed",
            "legacy_artifact_provenance": None,
        }
    )
    return result, selection, inputs


def test_swap_glm_write_uses_analysis_nwb_lifecycle(tmp_path: Path) -> None:
    """The live writer registers seven selectively fetchable NWB objects."""
    from v1ca1.spyglass import swap_glm

    result, _selection, _inputs = _terminal_swap_glm_result()
    analysis_table = _TestAnalysisNwbfile(tmp_path / "swap-glm-analysis.nwb")
    with pytest.raises(ValueError, match="analysis_nwbfile_table is required"):
        tables_module._write_swap_glm_nwb(
            nwb_file_name="L1420240102_.nwb",
            result=result,
            analysis_nwbfile_table=None,
        )
    row = tables_module._write_swap_glm_nwb(
        nwb_file_name="L1420240102_.nwb",
        result=result,
        analysis_nwbfile_table=analysis_table,
    )
    assert analysis_table.builder.registered is True
    assert row["analysis_file_name"] == "swap-glm-analysis.nwb"
    assert row["artifact_schema_version"] == (
        swap_glm.NWB_ARTIFACT_SCHEMA_VERSION
    )
    assert row["analysis_status"] == "no_units"
    assert {
        f"{name}_object_id"
        for name in tables_module._SWAP_GLM_NWB_OBJECT_NAMES
    }.issubset(row)
    assert swap_glm.swap_glm_nwb_hashes(result).items() <= row.items()


def test_swap_glm_loader_uses_fetch_nwb_and_checks_hashes() -> None:
    """The live loader reconstructs all SwapGLM science through fetch_nwb."""
    from v1ca1.spyglass import swap_glm

    result, selection, _inputs = _terminal_swap_glm_result()
    objects = swap_glm.swap_glm_result_to_nwb_objects(result)

    class FetchNwbRelation:
        def __init__(self) -> None:
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [
                {
                    name: nwb_object.to_dataframe()
                    for name, nwb_object in objects.items()
                }
            ]

    relation = FetchNwbRelation()
    result_row = {
        "swap_glm_id": selection["swap_glm_id"],
        "artifact_schema_version": swap_glm.NWB_ARTIFACT_SCHEMA_VERSION,
        **swap_glm.swap_glm_nwb_hashes(result),
    }
    loaded = tables_module._load_swap_glm_result(
        result_row=result_row,
        swap_glm_table=relation,
    )
    assert loaded["analysis_status"] == "no_units"
    assert relation.keys == [{"swap_glm_id": selection["swap_glm_id"]}]
    with pytest.raises(ValueError, match="model_results_sha256"):
        tables_module._load_swap_glm_result(
            result_row={**result_row, "model_results_sha256": "0" * 64},
            swap_glm_table=relation,
        )


def test_swap_glm_make_passes_heldout_nwb_inputs_and_terminal_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The compute hook receives position even for a terminal movement row."""
    from v1ca1.spyglass import swap_glm

    inputs = _swap_glm_selection_inputs()
    selection = _build_swap_glm_selection(inputs)
    movement_interval = object()
    position = object()
    trajectory_intervals = {
        trajectory_type: object()
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    graph_inputs = {
        trajectory_type: object()
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    spikes = object()
    stable_unit_ids = [
        {"spikesorting_merge_id": "merge-a", "unit_id": 10}
    ]
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "dark_light_snapshot": inputs["dark_light_snapshot"],
        "movement": {
            "movement_intervals": movement_interval,
            "analysis_status": "no_valid_position",
        },
        "movement_selection": {"position_series_name": "head_position"},
        "animal_name": "L14",
        "date": "20240102",
        "region": "v1",
    }
    monkeypatch.setattr(
        tables_module,
        "_load_swap_glm_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_glm_spikes",
        lambda **kwargs: {
            "ts_group": spikes,
            "unit_ids": stable_unit_ids,
        },
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_glm_nwb_inputs",
        lambda **kwargs: {
            "position": position,
            "trajectory_intervals": trajectory_intervals,
            "graph_inputs": graph_inputs,
        },
    )
    upstream = {
        "dark_light_glm_id": str(selection["dark_light_glm_id"]),
        "dark_light_glm_sha256": selection[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": selection[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": selection[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": selection[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": selection[
            "upstream_analysis_status"
        ],
    }
    compute_calls = []
    result = {
        "upstream_provenance": upstream,
        "n_units": 1,
        "n_valid_units": 0,
        "analysis_status": "no_valid_position",
        "selected_units_sha256": inputs["region_row"][
            "selected_units_sha256"
        ],
    }

    def compute(**kwargs):
        compute_calls.append(kwargs)
        return result

    monkeypatch.setattr(swap_glm, "compute_swap_glm", compute)
    write_calls = []
    monkeypatch.setattr(
        tables_module,
        "_write_swap_glm_nwb",
        lambda **kwargs: (
            write_calls.append(kwargs)
            or {"analysis_status": result["analysis_status"]}
        ),
    )
    analysis_nwbfile_table = object()

    row = _make_swap_glm_row(
        key=selection,
        parameters_table=object(),
        dark_light_glm_table=object(),
        dark_light_glm_selection_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        position_table=object(),
        epoch_intervals_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    assert len(compute_calls) == 1
    call = compute_calls[0]
    assert call["spikes"] is spikes
    assert call["stable_unit_ids"] is stable_unit_ids
    assert call["movement_interval"] is movement_interval
    assert call["trajectory_intervals"] is trajectory_intervals
    assert call["graph_inputs_by_trajectory"] is graph_inputs
    assert call["position"] is position
    assert call["dark_light_glm_result"] is inputs[
        "dark_light_snapshot"
    ]["bundle"]
    assert "dark_light_glm_artifact_path" not in call
    assert row["analysis_status"] == "no_valid_position"
    assert write_calls[0]["result"] is result
    assert write_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert (
        write_calls[0]["analysis_nwbfile_table"]
        is analysis_nwbfile_table
    )


def test_swap_glm_artifact_link_checks_bundle_identity_and_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A valid bundle must retain its exact selection and canonical layout."""
    from v1ca1.spyglass import swap_glm

    inputs = _swap_glm_selection_inputs()
    selection = _build_swap_glm_selection(inputs)
    upstream = {
        "dark_light_glm_id": str(selection["dark_light_glm_id"]),
        "dark_light_glm_sha256": selection[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": selection[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": selection[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": selection[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": selection[
            "upstream_analysis_status"
        ],
    }
    bundle = {
        "metadata": {
            "swap_glm_id": str(selection["swap_glm_id"]),
            "animal_name": "L14",
            "date": "20240102",
            "region": "v1",
            "dark_epoch": selection["dark_epoch"],
            "light_train_epoch": selection["light_train_epoch"],
            "light_test_epoch": selection["light_test_epoch"],
        },
        "parameters": {
            "parameter_name": inputs["parameters"]["swap_glm_param_name"],
            "parameter_sha256": selection[
                "swap_glm_parameters_sha256"
            ],
            "output_rule_sha256": selection[
                "swap_glm_output_rule_sha256"
            ],
            "swap_light_offset": inputs["parameters"][
                "swap_light_offset"
            ],
            "observed_spatial_bin_size_cm": inputs["parameters"][
                "observed_spatial_bin_size_cm"
            ],
        },
        "upstream_provenance": upstream,
        "n_units": 2,
        "n_valid_units": 1,
        "analysis_status": "partial_valid",
        "selected_units_sha256": inputs["region_row"][
            "selected_units_sha256"
        ],
        "artifact_origin": "computed",
    }
    monkeypatch.setattr(
        swap_glm,
        "validate_swap_glm_result",
        lambda candidate: candidate,
    )
    hashes = {
        "selected_units_table_sha256": "a" * 64,
        "model_metadata_sha256": "b" * 64,
        "axes_sha256": "c" * 64,
        "trajectory_metadata_sha256": "d" * 64,
        "model_results_sha256": "e" * 64,
        "observed_response_sha256": "f" * 64,
        "provenance_sha256": "0" * 64,
    }
    monkeypatch.setattr(swap_glm, "swap_glm_nwb_hashes", lambda result: hashes)
    row = {
        "artifact_schema_version": swap_glm.NWB_ARTIFACT_SCHEMA_VERSION,
        "schema_version": swap_glm.RESULT_SCHEMA_VERSION,
        "bundle_schema_version": swap_glm.BUNDLE_SCHEMA_VERSION,
        "n_units": bundle["n_units"],
        "n_valid_units": bundle["n_valid_units"],
        "analysis_status": bundle["analysis_status"],
        "selected_units_sha256": bundle["selected_units_sha256"],
        "dark_light_glm_sha256": upstream[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": upstream[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": upstream[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": upstream[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": upstream[
            "upstream_analysis_status"
        ],
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
        **hashes,
    }

    _validate_swap_glm_artifact_link(
        bundle=bundle,
        result_row=row,
        selection_row=selection,
        parameters_row=inputs["parameters"],
        region_row=inputs["region_row"],
        animal_name="L14",
        date="20240102",
    )

    bad_hash_row = {**row, "model_results_sha256": "9" * 64}
    with pytest.raises(ValueError, match="model_results_sha256"):
        _validate_swap_glm_artifact_link(
            bundle=bundle,
            result_row=bad_hash_row,
            selection_row=selection,
            parameters_row=inputs["parameters"],
            region_row=inputs["region_row"],
            animal_name="L14",
            date="20240102",
        )

    bad_identity_bundle = {
        **bundle,
        "metadata": {**bundle["metadata"], "light_test_epoch": "10_r5"},
    }
    with pytest.raises(ValueError, match="light_test_epoch"):
        _validate_swap_glm_artifact_link(
            bundle=bad_identity_bundle,
            result_row=row,
            selection_row=selection,
            parameters_row=inputs["parameters"],
            region_row=inputs["region_row"],
            animal_name="L14",
            date="20240102",
        )


def test_swap_glm_registration_uses_canonical_bundle_and_strict_resolver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _swap_glm_selection_inputs()
    selection = _build_swap_glm_selection(inputs)
    movement_interval = object()
    position = object()
    trajectory_intervals = {
        trajectory_type: object()
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    graph_inputs = {
        trajectory_type: object()
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "dark_light_snapshot": inputs["dark_light_snapshot"],
        "movement": {
            "movement_intervals": movement_interval,
            "analysis_status": "valid",
        },
        "movement_parameters": {
            "speed_threshold_cm_s": 4.0,
        },
        "animal_name": "L14",
        "date": "20240102",
        "region": "v1",
    }
    monkeypatch.setattr(
        tables_module,
        "_load_swap_glm_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_glm_spikes",
        lambda **kwargs: {"ts_group": {}, "unit_ids": []},
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_glm_nwb_inputs",
        lambda **kwargs: {
            "position": position,
            "position_row": {"analysis_start_offset_samples": 10},
            "trajectory_intervals": trajectory_intervals,
            "graph_inputs": graph_inputs,
        },
    )
    resolver_calls = []
    monkeypatch.setattr(
        tables_module,
        "_legacy_swap_glm_unit_identity_resolver",
        lambda loaded: resolver_calls.append(loaded) or {},
    )
    from v1ca1.spyglass import swap_glm

    upstream = {
        "dark_light_glm_id": str(selection["dark_light_glm_id"]),
        "dark_light_glm_sha256": selection[
            "dark_light_glm_sha256"
        ],
        "dark_light_selected_model_sha256_by_model": selection[
            "dark_light_selected_model_sha256_by_model"
        ],
        "dark_light_parameter_sha256": selection[
            "dark_light_parameter_sha256"
        ],
        "dark_light_output_rule_sha256": selection[
            "dark_light_output_rule_sha256"
        ],
        "upstream_analysis_status": selection[
            "upstream_analysis_status"
        ],
    }
    register_calls = []

    def register_existing(**kwargs):
        register_calls.append(kwargs)
        return {
            "upstream_provenance": upstream,
            "legacy_artifact_provenance": {"source": "legacy"},
            "n_units": 1,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }

    monkeypatch.setattr(
        swap_glm,
        "register_existing_swap_glm_artifact",
        register_existing,
    )
    writer_calls = []
    monkeypatch.setattr(
        tables_module,
        "_write_swap_glm_nwb",
        lambda **kwargs: writer_calls.append(kwargs) or {
            "legacy_artifact_provenance": kwargs["result"][
                "legacy_artifact_provenance"
            ]
        },
    )
    analysis_nwbfile_table = object()
    row = _register_existing_swap_glm_row(
        key=selection,
        source_result_path=tmp_path / "legacy.nc",
        parameters_table=object(),
        dark_light_glm_table=object(),
        dark_light_glm_selection_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        position_table=object(),
        epoch_intervals_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        source_v1ca1_git_commit="v1",
        source_spyglass_git_commit="sg",
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )
    assert resolver_calls
    assert len(register_calls) == 1
    registration = register_calls[0]
    assert registration["spikes"] == {}
    assert registration["stable_unit_ids"] == []
    assert registration["movement_interval"] is movement_interval
    assert registration["movement_analysis_status"] == "valid"
    assert registration["trajectory_intervals"] is trajectory_intervals
    assert registration["graph_inputs_by_trajectory"] is graph_inputs
    assert registration["position"] is position
    assert registration["position_offset_samples"] == 10
    assert isinstance(registration["position_offset_samples"], int)
    assert registration["speed_threshold_cm_s"] == 4.0
    assert isinstance(registration["speed_threshold_cm_s"], float)
    assert registration["destination_path"] is None
    assert registration["dark_light_glm_result"] is inputs[
        "dark_light_snapshot"
    ]["bundle"]
    assert "dark_light_glm_artifact_path" not in registration
    assert writer_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert (
        writer_calls[0]["analysis_nwbfile_table"]
        is analysis_nwbfile_table
    )
    assert row["legacy_artifact_provenance"][
        "source_spyglass_git_commit"
    ] == "sg"


def test_swap_glm_failed_insert_removes_new_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed DataJoint insert removes only the new analysis NWB file."""
    result_id = uuid.uuid4()
    artifact_path = tmp_path / f"{result_id}.nwb"
    retained_path = tmp_path / "preexisting.nc"
    retained_path.write_bytes(b"keep")

    def compute(**_kwargs):
        artifact_path.write_bytes(b"new")
        return {
            "analysis_file_name": artifact_path.name,
            "schema_version": "5",
            "bundle_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "partial_valid",
            "selected_units_sha256": "a" * 64,
            "dark_light_glm_sha256": "b" * 64,
            "dark_light_selected_model_sha256_by_model": {"visual": "c" * 64},
            "dark_light_parameter_sha256": "d" * 64,
            "dark_light_output_rule_sha256": "e" * 64,
            "upstream_analysis_status": "valid",
            "legacy_artifact_provenance": None,
            "_created_artifact_paths": [str(artifact_path)],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {"swap_glm_id": result_id},
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"swap_glm_compute": compute}
    )
    result = bundle["swap_glm"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make({"swap_glm_id": result_id})

    assert not artifact_path.exists()
    assert retained_path.read_bytes() == b"keep"


def test_swap_tuning_parameters_and_selection_freeze_every_source() -> None:
    parameters = dict(
        table_specs.MANUSCRIPT_V1_SWAP_TUNING_CURVE_COMPARISON_PARAMETERS
    )
    assert _validate_swap_tuning_curve_comparison_parameter_row(
        parameters
    ) == parameters
    inputs = _swap_tuning_curve_comparison_selection_inputs()

    first = _build_swap_tuning_curve_comparison_selection(inputs)
    second = _build_swap_tuning_curve_comparison_selection(
        _swap_tuning_curve_comparison_selection_inputs()
    )

    assert first == second
    assert first["swap_tuning_curve_comparison_id"].version == 5
    assert first["dark_condition"] == "dark"
    assert first["light_train_condition"] == "AB"
    assert first["light_test_condition"] == "BA"
    assert first["position_offset_samples"] == 10
    assert first["speed_threshold_cm_s"] == 4.0
    assert set(first["source_tuning_curve_sha256_by_role_trajectory"]) == {
        "dark",
        "light_train",
        "light_test",
    }
    assert first["movement_firing_rate_table_sha256_by_role"] == {
        epoch_role: digests["firing_rate"]
        for epoch_role, digests in inputs[
            "movement_artifact_sha256_by_role"
        ].items()
    }
    assert first["movement_intervals_sha256_by_role"] == {
        epoch_role: digests["movement_intervals"]
        for epoch_role, digests in inputs[
            "movement_artifact_sha256_by_role"
        ].items()
    }
    assert first[
        "swap_tuning_curve_comparison_parameters_sha256"
    ] == provenance_sha256(parameters)
    assert first[
        "swap_tuning_curve_comparison_output_rule_sha256"
    ] == provenance_sha256(
        dict(table_specs.SWAP_TUNING_CURVE_COMPARISON_OUTPUT_RULE)
    )

    changed = _swap_tuning_curve_comparison_selection_inputs()
    changed["curve_snapshots"]["light_test:center_to_left"][
        "artifact_sha256"
    ] = "f" * 64
    changed_row = _build_swap_tuning_curve_comparison_selection(changed)
    assert changed_row["swap_tuning_curve_comparison_id"] != first[
        "swap_tuning_curve_comparison_id"
    ]

    changed = _swap_tuning_curve_comparison_selection_inputs()
    changed["movement_artifact_sha256_by_role"]["light_test"][
        "movement_intervals"
    ] = "d" * 64
    changed_row = _build_swap_tuning_curve_comparison_selection(changed)
    assert changed_row["swap_tuning_curve_comparison_id"] != first[
        "swap_tuning_curve_comparison_id"
    ]

    stale_upstream = _swap_tuning_upstream_provenance(first)
    stale_upstream["movement_intervals_sha256_by_role"] = {
        **stale_upstream["movement_intervals_sha256_by_role"],
        "dark": "0" * 64,
    }
    with pytest.raises(ValueError, match="movement_intervals_sha256_by_role"):
        _validate_swap_tuning_curve_comparison_upstream_link(
            stale_upstream,
            first,
        )

    changed = _swap_tuning_curve_comparison_selection_inputs()
    changed["movement_artifact_sha256_by_role"]["dark"][
        "firing_rate"
    ] = "e" * 64
    changed_row = _build_swap_tuning_curve_comparison_selection(changed)
    assert changed_row["swap_tuning_curve_comparison_id"] != first[
        "swap_tuning_curve_comparison_id"
    ]


def test_swap_tuning_selection_rejects_mixed_sources_and_conditions() -> None:
    inputs = _swap_tuning_curve_comparison_selection_inputs()
    movement_id = inputs["key"]["light_test_movement_firing_rate_id"]
    inputs["movement_selections"][movement_id][
        "position_series_name"
    ] = "body_position"
    with pytest.raises(ValueError, match="source definition"):
        _build_swap_tuning_curve_comparison_selection(inputs)

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    inputs["epoch_rows"][-1]["condition"] = "AB"
    with pytest.raises(ValueError, match="conditions must differ"):
        _build_swap_tuning_curve_comparison_selection(inputs)

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    inputs["curve_snapshots"]["dark:center_to_left"]["selection"][
        "trial_subset"
    ] = "odd"
    with pytest.raises(ValueError, match="all-trial"):
        _build_swap_tuning_curve_comparison_selection(inputs)

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    inputs["tuning_parameters"]["place_bin_size_cm"] = 2.0
    inputs["tuning_parameters"]["tuning_curve_param_name"] = "2cm"
    new_hash = provenance_sha256(inputs["tuning_parameters"])
    for snapshot in inputs["curve_snapshots"].values():
        snapshot["selection"]["tuning_curve_param_name"] = "2cm"
        snapshot["selection"]["tuning_curve_parameters_sha256"] = new_hash
    with pytest.raises(ValueError, match="4-cm"):
        _build_swap_tuning_curve_comparison_selection(inputs)

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    inputs["position_rows"][-1]["analysis_start_offset_samples"] = 0
    with pytest.raises(ValueError, match="start offset"):
        _build_swap_tuning_curve_comparison_selection(inputs)

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    movement_id = inputs["key"]["dark_movement_firing_rate_id"]
    inputs["movement_results"][movement_id]["selected_units_sha256"] = (
        "f" * 64
    )
    with pytest.raises(ValueError, match="same persistent units"):
        _build_swap_tuning_curve_comparison_selection(inputs)


def test_swap_tuning_legacy_resolver_requires_imported_unique_units() -> None:
    loaded = {
        "ts_group": {0: object(), 1: object()},
        "unit_ids": [
            {"spikesorting_merge_id": "merge-a", "unit_id": 10},
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        ],
        "unit_metadata": [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 10,
                "sorting_unit_id": 101,
            },
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 11,
                "sorting_unit_id": 102,
            },
        ],
        "member_provenance": [
            {
                "spikesorting_merge_id": "merge-a",
                "merge_parent": "ImportedSpikeSorting",
                "n_selected_units": 2,
            }
        ],
    }
    resolver = _legacy_swap_tuning_curve_comparison_unit_identity_resolver(
        loaded
    )
    assert resolver["101"]["stable_unit_id"] == "merge-a:10"
    assert resolver["102"]["group_unit_id"] == "1"

    loaded["member_provenance"][0]["merge_parent"] = "CurationV1"
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        _legacy_swap_tuning_curve_comparison_unit_identity_resolver(loaded)


def test_swap_tuning_make_passes_exact_selected_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from v1ca1.spyglass import swap_tuning

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    selection = _build_swap_tuning_curve_comparison_selection(inputs)
    movement_interval = object()
    position = object()
    spikes = object()
    stable_unit_ids = [
        {"spikesorting_merge_id": "merge-a", "unit_id": 10}
    ]
    trajectory_intervals = {
        trajectory_type: object()
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    graph_inputs = dict(trajectory_intervals)
    tuning_paths = {
        epoch_role: {
            trajectory_type: tmp_path / f"{epoch_role}-{trajectory_type}.nc"
            for trajectory_type in _DPP_ENCODING_TRAJECTORIES
        }
        for epoch_role in ("dark", "light_train", "light_test")
    }
    movement_tables = {
        epoch_role: object()
        for epoch_role in ("dark", "light_train", "light_test")
    }
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "region_row": inputs["region_row"],
        "movement": {
            "light_test": {
                "movement_intervals": movement_interval,
                "analysis_status": "valid",
            }
        },
        "movement_selections": {
            "light_test": {"position_series_name": "head_position"}
        },
        "tuning_curves_by_role_trajectory": tuning_paths,
        "movement_firing_rate_tables": movement_tables,
        "animal_name": "L14",
        "date": "20240102",
        "region": "v1",
    }
    monkeypatch.setattr(
        tables_module,
        "_load_swap_tuning_curve_comparison_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_tuning_curve_comparison_spikes",
        lambda **kwargs: {
            "ts_group": spikes,
            "unit_ids": stable_unit_ids,
        },
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_tuning_curve_comparison_nwb_inputs",
        lambda **kwargs: {
            "position": position,
            "position_row": {"analysis_start_offset_samples": 10},
            "trajectory_intervals": trajectory_intervals,
            "graph_inputs": graph_inputs,
        },
    )
    compute_calls = []
    result = {
        "upstream_provenance": _swap_tuning_upstream_provenance(selection),
        "n_source_units": 5,
        "n_units": 3,
        "n_valid_units": 2,
        "analysis_status": "partial_valid",
        "selected_units_sha256": selection["selected_units_sha256"],
    }

    def compute(**kwargs):
        compute_calls.append(kwargs)
        return result

    monkeypatch.setattr(
        swap_tuning,
        "compute_swap_tuning_curve_comparison",
        compute,
    )
    write_calls = []

    def write_nwb(**kwargs):
        write_calls.append(kwargs)
        return {
            "analysis_file_name": "swap-tuning.nwb",
            **{
                f"{name}_object_id": f"{name}-object-id"
                for name in (
                    "selected_units",
                    "score_summary",
                    "source_profiles",
                    "model_profiles",
                    "geometry",
                    "provenance",
                )
            },
            "artifact_schema_version": "1",
            "n_source_units": 5,
            "n_units": 3,
            "n_valid_units": 2,
            "analysis_status": "partial_valid",
            "selected_units_sha256": selection["selected_units_sha256"],
            "legacy_artifact_provenance": None,
        }

    monkeypatch.setattr(
        tables_module,
        "_write_swap_tuning_curve_comparison_nwb",
        write_nwb,
    )
    analysis_nwbfile_table = object()

    row = _make_swap_tuning_curve_comparison_row(
        key=selection,
        parameters_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        position_table=object(),
        tuning_curve_table=object(),
        tuning_curve_selection_table=object(),
        tuning_curve_parameters_table=object(),
        epoch_intervals_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    assert len(compute_calls) == 1
    call = compute_calls[0]
    assert call["tuning_curve_artifact_paths"] is None
    assert call["tuning_curves_by_role_trajectory"] is tuning_paths
    assert call["movement_firing_rate_tables_by_role"] is movement_tables
    assert call["spikes"] is spikes
    assert call["stable_unit_ids"] is stable_unit_ids
    assert call["position"] is position
    assert call["position_offset_samples"] == 10
    assert call["movement_interval"] is movement_interval
    assert call["trajectory_intervals"] is trajectory_intervals
    assert call["graph_inputs_by_trajectory"] is graph_inputs
    assert set(call["source_tuning_curve_ids_by_role_trajectory"]) == {
        "dark",
        "light_train",
        "light_test",
    }
    assert call[
        "source_tuning_parameters_sha256_by_role_trajectory"
    ] == selection["source_tuning_parameters_sha256_by_role_trajectory"]
    assert call["movement_intervals_sha256_by_role"] == selection[
        "movement_intervals_sha256_by_role"
    ]
    assert call["sources"]["position_series_name"] == "head_position"
    assert "source_spyglass_git_commit" not in call["sources"]
    assert len(write_calls) == 1
    assert write_calls[0] == {
        "nwb_file_name": selection["nwb_file_name"],
        "result": result,
        "analysis_nwbfile_table": analysis_nwbfile_table,
    }
    assert row["n_source_units"] == 5
    assert row["n_units"] == 3
    assert row["n_valid_units"] == 2


def test_swap_tuning_registration_rebuilds_exact_nwb_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from v1ca1.spyglass import swap_tuning

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    selection = _build_swap_tuning_curve_comparison_selection(inputs)
    movement_interval = object()
    position = object()
    loaded_spikes = {
        "ts_group": {},
        "unit_ids": [],
        "unit_metadata": [],
        "member_provenance": [],
    }
    trajectory_intervals = {
        trajectory_type: object()
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    graph_inputs = dict(trajectory_intervals)
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "region_row": inputs["region_row"],
        "movement": {
            "light_test": {
                "movement_intervals": movement_interval,
                "analysis_status": "valid",
            }
        },
        "movement_parameters": inputs["movement_parameters"],
        "movement_selections": {
            "light_test": {"position_series_name": "head_position"}
        },
        "tuning_curves_by_role_trajectory": {
            epoch_role: {
                trajectory_type: tmp_path / f"{epoch_role}-{trajectory_type}.nc"
                for trajectory_type in _DPP_ENCODING_TRAJECTORIES
            }
            for epoch_role in ("dark", "light_train", "light_test")
        },
        "movement_firing_rate_tables": {
            epoch_role: object()
            for epoch_role in ("dark", "light_train", "light_test")
        },
        "animal_name": "L14",
        "date": "20240102",
        "region": "v1",
    }
    monkeypatch.setattr(
        tables_module,
        "_load_swap_tuning_curve_comparison_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_tuning_curve_comparison_spikes",
        lambda **kwargs: loaded_spikes,
    )
    monkeypatch.setattr(
        tables_module,
        "_load_swap_tuning_curve_comparison_nwb_inputs",
        lambda **kwargs: {
            "position": position,
            "position_row": {
                "position_series_name": "head_position",
                "position_role": "head",
                "analysis_start_offset_samples": 10,
            },
            "trajectory_intervals": trajectory_intervals,
            "graph_inputs": graph_inputs,
        },
    )
    resolver = {"101": {"stable_unit_id": "merge-a:10"}}
    monkeypatch.setattr(
        tables_module,
        "_legacy_swap_tuning_curve_comparison_unit_identity_resolver",
        lambda loaded: resolver,
    )
    register_calls = []

    def register_existing(**kwargs):
        register_calls.append(kwargs)
        return {
            "upstream_provenance": _swap_tuning_upstream_provenance(selection),
            "legacy_artifact_provenance": {
                "source": "legacy",
                "source_spyglass_git_commit": "sg",
            },
            "n_source_units": 5,
            "n_units": 3,
            "n_valid_units": 3,
            "analysis_status": "valid",
            "selected_units_sha256": selection["selected_units_sha256"],
        }

    monkeypatch.setattr(
        swap_tuning,
        "register_existing_swap_tuning_curve_comparison_artifact",
        register_existing,
    )
    write_calls = []

    def write_nwb(**kwargs):
        write_calls.append(kwargs)
        return {
            "analysis_file_name": "registered-swap-tuning.nwb",
            "legacy_artifact_provenance": kwargs["result"][
                "legacy_artifact_provenance"
            ],
        }

    monkeypatch.setattr(
        tables_module,
        "_write_swap_tuning_curve_comparison_nwb",
        write_nwb,
    )
    analysis_nwbfile_table = object()
    row = _register_existing_swap_tuning_curve_comparison_row(
        key=selection,
        source_result_path=tmp_path / "legacy.nc",
        source_summary_path=tmp_path / "legacy.parquet",
        parameters_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        position_table=object(),
        tuning_curve_table=object(),
        tuning_curve_selection_table=object(),
        tuning_curve_parameters_table=object(),
        epoch_intervals_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        source_v1ca1_git_commit="v1",
        source_spyglass_git_commit="sg",
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    assert len(register_calls) == 1
    call = register_calls[0]
    assert call["unit_identity_resolver"] is resolver
    assert call["source_sorting_type"] == "ImportedSpikeSorting"
    assert call["position"] is position
    assert call["position_offset_samples"] == 10
    assert call["movement_interval"] is movement_interval
    assert call["trajectory_intervals"] is trajectory_intervals
    assert call["graph_inputs_by_trajectory"] is graph_inputs
    assert call["source_v1ca1_git_commit"] == "v1"
    assert call["movement_intervals_sha256_by_role"] == selection[
        "movement_intervals_sha256_by_role"
    ]
    assert call["tuning_curve_artifact_paths"] is None
    assert call["tuning_curves_by_role_trajectory"] is context[
        "tuning_curves_by_role_trajectory"
    ]
    assert call["sources"]["position_series_name"] == "head_position"
    assert call["sources"]["source_spyglass_git_commit"] == "sg"
    assert call["source_spyglass_git_commit"] == "sg"
    assert call["destination_path"] is None
    assert len(write_calls) == 1
    assert write_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert write_calls[0]["analysis_nwbfile_table"] is analysis_nwbfile_table
    assert row["legacy_artifact_provenance"][
        "source_spyglass_git_commit"
    ] == "sg"


def test_swap_tuning_artifact_link_checks_identity_and_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from v1ca1.spyglass import swap_tuning
    from v1ca1.spyglass.selection import unit_identity_sha256

    inputs = _swap_tuning_curve_comparison_selection_inputs()
    selection = _build_swap_tuning_curve_comparison_selection(inputs)
    identities = [
        {"spikesorting_merge_id": "merge-a", "unit_id": "10"},
        {"spikesorting_merge_id": "merge-a", "unit_id": "11"},
    ]
    selected_digest = unit_identity_sha256(identities)
    selection = {**selection, "selected_units_sha256": selected_digest}
    region_row = {
        **inputs["region_row"],
        "n_units": 2,
        "selected_units_sha256": selected_digest,
    }
    upstream = _swap_tuning_upstream_provenance(selection)
    selected_units = pd.DataFrame(
        {
            **{
                field_name: [row[field_name] for row in identities]
                for field_name in ("spikesorting_merge_id", "unit_id")
            },
            "stable_unit_id": ["merge-a:10", "merge-a:11"],
        }
    )
    bundle = {
        "metadata": {
            "swap_tuning_curve_comparison_id": str(
                selection["swap_tuning_curve_comparison_id"]
            ),
            "animal_name": "L14",
            "date": "20240102",
            "region": "v1",
            "dark_epoch": selection["dark_epoch"],
            "light_train_epoch": selection["light_train_epoch"],
            "light_test_epoch": selection["light_test_epoch"],
        },
        "parameters": {
            "parameter_name": inputs["parameters"][
                "swap_tuning_curve_comparison_param_name"
            ],
            "parameter_sha256": selection[
                "swap_tuning_curve_comparison_parameters_sha256"
            ],
            "output_rule_sha256": selection[
                "swap_tuning_curve_comparison_output_rule_sha256"
            ],
            **{
                field_name: value
                for field_name, value in inputs["parameters"].items()
                if field_name
                != "swap_tuning_curve_comparison_param_name"
            },
        },
        "upstream_provenance": upstream,
        "selected_units": selected_units,
        "n_source_units": 2,
        "n_units": 1,
        "n_valid_units": 1,
        "analysis_status": "valid",
        "selected_units_sha256": selected_digest,
        "artifact_origin": "computed",
    }
    monkeypatch.setattr(
        swap_tuning,
        "validate_swap_tuning_curve_comparison_result",
        lambda candidate: candidate,
    )
    nwb_hashes = {
        "selected_units_table_sha256": "1" * 64,
        "score_summary_sha256": "2" * 64,
        "source_profiles_sha256": "3" * 64,
        "model_profiles_sha256": "4" * 64,
        "geometry_sha256": "5" * 64,
        "provenance_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        swap_tuning,
        "swap_tuning_curve_comparison_nwb_hashes",
        lambda candidate: nwb_hashes,
    )
    row = {
        "analysis_file_name": "swap-tuning.nwb",
        "artifact_schema_version": swap_tuning.NWB_ARTIFACT_SCHEMA_VERSION,
        "n_source_units": 2,
        "n_units": 1,
        "n_valid_units": 1,
        "analysis_status": "valid",
        "selected_units_sha256": selected_digest,
        "artifact_origin": "computed",
        **nwb_hashes,
    }

    _validate_swap_tuning_curve_comparison_artifact_link(
        bundle=bundle,
        result_row=row,
        selection_row=selection,
        parameters_row=inputs["parameters"],
        region_row=region_row,
        animal_name="L14",
        date="20240102",
    )

    with pytest.raises(ValueError, match="legacy provenance differs"):
        _validate_swap_tuning_curve_comparison_artifact_link(
            bundle=bundle,
            result_row={
                **row,
                "legacy_artifact_provenance": {"source": "row-only"},
            },
            selection_row=selection,
            parameters_row=inputs["parameters"],
            region_row=region_row,
            animal_name="L14",
            date="20240102",
        )

    with pytest.raises(ValueError, match="NWB objects"):
        _validate_swap_tuning_curve_comparison_artifact_link(
            bundle=bundle,
            result_row={
                **row,
                "score_summary_sha256": "f" * 64,
            },
            selection_row=selection,
            parameters_row=inputs["parameters"],
            region_row=region_row,
            animal_name="L14",
            date="20240102",
        )


def test_swap_tuning_loader_uses_fetch_nwb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public result loader reconstructs only the six selected NWB objects."""
    from v1ca1.spyglass import swap_tuning

    result_id = uuid.uuid4()
    object_names = (
        "selected_units",
        "score_summary",
        "source_profiles",
        "model_profiles",
        "geometry",
        "provenance",
    )
    fetched = {name: object() for name in object_names}

    class Relation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [fetched]

    reconstructed = {"reconstructed": True}
    monkeypatch.setattr(
        swap_tuning,
        "swap_tuning_curve_comparison_result_from_nwb_objects",
        lambda **objects: reconstructed
        if objects == fetched
        else pytest.fail("Unexpected NWB object set."),
    )
    validation_calls = []
    monkeypatch.setitem(
        _load_swap_tuning_curve_comparison_result.__globals__,
        "_validate_swap_tuning_curve_comparison_artifact_link",
        lambda **kwargs: validation_calls.append(kwargs),
    )
    relation = Relation()
    result_row = {"swap_tuning_curve_comparison_id": result_id}
    loaded = _load_swap_tuning_curve_comparison_result(
        result_row=result_row,
        result_table=relation,
        selection_row={"selection": True},
        parameters_row={"parameters": True},
        region_row={"region": True},
        animal_name="L14",
        date="20240102",
    )
    assert loaded is reconstructed
    assert relation.keys == [
        {"swap_tuning_curve_comparison_id": result_id}
    ]
    assert validation_calls[0]["bundle"] is reconstructed
    assert validation_calls[0]["result_row"] == result_row


def test_swap_tuning_make_requires_populate_transaction() -> None:
    """Direct make cannot register a swap-tuning NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "swap_tuning_curve_comparison_compute": lambda **kwargs: pytest.fail(
                "SwapTuningCurveComparison computation must not start outside "
                "populate()."
            )
        }
    )
    result = bundle["swap_tuning_curve_comparison"]
    result.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"swap_tuning_curve_comparison_id": uuid.uuid4()})

def test_tuning_similarity_result_hooks_receive_fetched_selection(
    monkeypatch,
) -> None:
    selection_id = uuid.UUID("85555555-5555-5555-8555-555555555555")
    selection = {
        "path_specific_place_tuning_similarity_id": selection_id,
        **_tuning_similarity_selection_key(),
        "tuning_similarity_parameters_sha256": provenance_sha256(
            dict(table_specs.CORRELATION_TUNING_SIMILARITY_PARAMETERS)
        ),
    }
    calls = []

    def result_row(file_name: str) -> dict[str, Any]:
        return {
            "analysis_file_name": file_name,
            "similarity_object_id": "similarity-object-id",
            "similarity_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "n_units": 3,
            "n_valid_comparisons": 10,
            "n_units_with_valid_comparison": 3,
            "analysis_status": "valid",
            "selected_units_sha256": "e" * 64,
        }

    def compute(**kwargs):
        calls.append(("compute", kwargs))
        return result_row("tuning-similarity.nwb")

    def register(**kwargs):
        calls.append(("register", kwargs))
        return {
            **result_row("registered-tuning-similarity.nwb"),
            "legacy_artifact_provenance": {"source": "all_units"},
        }

    def fetch_selection(table, key):
        assert table.__name__ == "PathSpecificPlaceTuningSimilaritySelection"
        assert key == {
            "path_specific_place_tuning_similarity_id": selection_id
        }
        return dict(selection)

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        fetch_selection,
    )
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_spyglass_git_commit",
        lambda: "runtime-spyglass-commit",
    )
    bundle, _, unit_selection_params = _fake_bundle(
        runtime_hooks={
            "path_specific_place_tuning_similarity_compute": compute,
            "path_specific_place_tuning_similarity_register_existing": register,
        }
    )
    result = bundle["path_specific_place_tuning_similarity"]

    result().make(
        {"path_specific_place_tuning_similarity_id": selection_id}
    )
    result.register_existing(
        {"path_specific_place_tuning_similarity_id": selection_id},
        similarity_path="old-tuning-similarity-all_units.parquet",
        source_v1ca1_git_commit="source-v1-commit",
    )

    assert [name for name, _kwargs in calls] == ["compute", "register"]
    assert all(call["key"] == selection for _name, call in calls)
    assert calls[1][1]["overwrite"] is False
    for _name, call in calls:
        assert call["parameters_table"] is bundle[
            "tuning_similarity_parameters"
        ]
        assert call["tuning_curve_table"] is bundle[
            "path_specific_place_tuning_curve"
        ]
        assert call["movement_firing_rate_table"] is bundle[
            "movement_firing_rate"
        ]
        assert call["analysis_nwbfile_table"] is bundle[
            "analysis_nwbfile"
        ]
    assert calls[1][1]["tuning_curve_parameters_table"] is bundle[
        "tuning_curve_parameters"
    ]
    assert calls[1][1]["region_sorted_spikes_group_table"] is bundle[
        "region_sorted_spikes_group"
    ]
    assert result._insert_calls[0][0][
        "path_specific_place_tuning_similarity_id"
    ] == selection_id
    assert result._insert_calls[0][0]["artifact_origin"] == "computed"
    assert result._insert_calls[1][0]["artifact_origin"] == (
        "registered_existing"
    )
    assert result._insert_calls[1][0]["legacy_artifact_provenance"] == {
        "source": "all_units"
    }
    assert result._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }
    assert all(
        row["runtime_spyglass_git_commit"] == "runtime-spyglass-commit"
        for row, _kwargs in result._insert_calls
        )


def test_legacy_dpp_encoding_filename_attests_encoded_parameters(
    tmp_path: Path,
) -> None:
    parameters = dict(
        table_specs.MANUSCRIPT_DPP_ENCODING_PARAMETERS
    )
    expected = (
        tmp_path
        / "v1_08_r4_cv5_bin0p05s_placebin4cm_encoding_summary.parquet"
    )

    assert _validate_legacy_dpp_encoding_source_path(
        expected,
        region="v1",
        epoch="08_r4",
        parameters=parameters,
    ) == expected
    with pytest.raises(ValueError, match="expected"):
        _validate_legacy_dpp_encoding_source_path(
            tmp_path
            / "v1_08_r4_cv5_bin0p02s_placebin4cm_encoding_summary.parquet",
            region="v1",
            epoch="08_r4",
            parameters=parameters,
        )


def test_legacy_dpp_unit_resolver_requires_imported_unique_sorting_ids() -> None:
    loaded = {
        "unit_ids": [
            {"spikesorting_merge_id": "merge-a", "unit_id": 10},
            {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        ],
        "unit_metadata": [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 10,
                "sorting_unit_id": 101,
            },
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": 11,
                "sorting_unit_id": 102,
            },
        ],
        "member_provenance": [
            {
                "spikesorting_merge_id": "merge-a",
                "merge_parent": "ImportedSpikeSorting",
                "n_selected_units": 2,
            }
        ],
    }

    assert _legacy_dpp_unit_identity_resolver(loaded) == {
        "101": {"spikesorting_merge_id": "merge-a", "unit_id": "10"},
        "102": {"spikesorting_merge_id": "merge-a", "unit_id": "11"},
    }
    duplicate = {
        **loaded,
        "unit_metadata": [
            loaded["unit_metadata"][0],
            {**loaded["unit_metadata"][1], "sorting_unit_id": 101},
        ],
    }
    with pytest.raises(ValueError, match="unique sorting_unit_id"):
        _legacy_dpp_unit_identity_resolver(duplicate)
    curated = {
        **loaded,
        "member_provenance": [
            {
                **loaded["member_provenance"][0],
                "merge_parent": "CurationV1",
            }
        ],
    }
    with pytest.raises(ValueError, match="non-imported"):
        _legacy_dpp_unit_identity_resolver(curated)


def test_make_dpp_encoding_row_forwards_selected_inputs_and_parameters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from v1ca1.spyglass import dpp_encoding

    comparison_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        "v1ca1-test-dpp-compute-wiring",
    )
    context, loaded_spikes, nwb_inputs = _dpp_encoding_runtime_inputs(
        comparison_id
    )
    calls: dict[str, Any] = {}
    result_table = object()
    analysis_nwbfile_table = object()

    monkeypatch.setitem(
        _make_dpp_encoding_row.__globals__,
        "_load_dpp_encoding_context",
        lambda **kwargs: context,
    )
    monkeypatch.setitem(
        _make_dpp_encoding_row.__globals__,
        "_load_dpp_encoding_spikes",
        lambda **kwargs: loaded_spikes,
    )
    monkeypatch.setitem(
        _make_dpp_encoding_row.__globals__,
        "_load_dpp_encoding_nwb_inputs",
        lambda **kwargs: nwb_inputs,
    )

    def compute(**kwargs):
        calls["compute"] = kwargs
        return {
            "table": result_table,
            "n_units_eligible": 2,
            "n_units_valid": 1,
            "analysis_status": "valid",
            "eligible_units_sha256": "d" * 64,
        }

    def write(**kwargs):
        calls["write"] = kwargs
        return {
            "analysis_file_name": "dpp-encoding-analysis.nwb",
            "dpp_encoding_object_id": "dpp-encoding-object-id",
            "dpp_encoding_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "_created_artifact_paths": ["/analysis/dpp-encoding-analysis.nwb"],
        }

    monkeypatch.setattr(
        dpp_encoding,
        "compute_selected_dpp_encoding",
        compute,
    )
    monkeypatch.setitem(
        _make_dpp_encoding_row.__globals__,
        "_write_dpp_encoding_nwb",
        write,
    )

    row = _make_dpp_encoding_row(
        key={"dpp_encoding_id": comparison_id},
        parameters_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        epoch_intervals_table=object(),
        position_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        stability_table=object(),
        stability_selection_table=object(),
        tuning_curve_selection_table=object(),
        session_table=object(),
        nwbfile_table=object(),
        artifact_root=tmp_path,
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    compute_call = calls["compute"]
    assert {
        name: compute_call[name]
        for name in (
            "n_folds",
            "evaluation_bin_size_s",
            "spatial_bin_size_cm",
            "gaussian_smoothing_sigma_bins",
            "random_seed",
            "minimum_movement_firing_rate_hz",
            "minimum_stability_correlation",
        )
    } == {
        name: context["parameters"][name]
        for name in (
            "n_folds",
            "evaluation_bin_size_s",
            "spatial_bin_size_cm",
            "gaussian_smoothing_sigma_bins",
            "random_seed",
            "minimum_movement_firing_rate_hz",
            "minimum_stability_correlation",
        )
    }
    assert compute_call["spikes"] is loaded_spikes["ts_group"]
    assert compute_call["stable_unit_ids"] is loaded_spikes["unit_ids"]
    assert compute_call["position"] is nwb_inputs["position"]
    assert compute_call["trajectory_intervals_by_type"] is nwb_inputs[
        "trajectory_intervals"
    ]
    assert compute_call["graph_inputs_by_configuration"] is nwb_inputs[
        "graph_inputs"
    ]
    assert compute_call["movement_intervals"] is context["movement"][
        "movement_intervals"
    ]
    assert compute_call["movement_firing_rate_table"] is context[
        "movement"
    ]["table"]
    assert compute_call["stability_tables_by_trajectory"] is context[
        "stability_tables"
    ]
    assert compute_call["dpp_encoding_id"] == comparison_id
    assert calls["write"] == {
        "nwb_file_name": "L1420240102_.nwb",
        "table": result_table,
        "analysis_nwbfile_table": analysis_nwbfile_table,
    }
    assert row == {
        "analysis_file_name": "dpp-encoding-analysis.nwb",
        "dpp_encoding_object_id": "dpp-encoding-object-id",
        "dpp_encoding_sha256": "f" * 64,
        "artifact_schema_version": "1",
        "n_units_input": 2,
        "n_units_eligible": 2,
        "n_units_valid": 1,
        "analysis_status": "valid",
        "eligible_units_sha256": "d" * 64,
        "legacy_artifact_provenance": None,
        "_created_artifact_paths": ["/analysis/dpp-encoding-analysis.nwb"],
    }


def test_register_dpp_encoding_row_uses_exact_legacy_source_and_resolver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from v1ca1.spyglass import dpp_encoding

    comparison_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        "v1ca1-test-dpp-register-wiring",
    )
    context, loaded_spikes, _nwb_inputs = _dpp_encoding_runtime_inputs(
        comparison_id
    )
    source_path = (
        tmp_path
        / "ca1_02_r1_cv5_bin0p05s_placebin4cm_encoding_summary.parquet"
    )
    source_path.write_bytes(b"legacy encoding")
    calls: dict[str, Any] = {}
    normalized_table = _valid_dpp_encoding_table()
    analysis_nwbfile_table = object()
    legacy_table = object()

    monkeypatch.setitem(
        _register_existing_dpp_encoding_row.__globals__,
        "_load_dpp_encoding_context",
        lambda **kwargs: context,
    )
    monkeypatch.setitem(
        _register_existing_dpp_encoding_row.__globals__,
        "_load_dpp_encoding_spikes",
        lambda **kwargs: loaded_spikes,
    )

    def read_parquet(path):
        calls["source_path"] = Path(path)
        return legacy_table

    monkeypatch.setattr(pd, "read_parquet", read_parquet)

    def normalize(legacy_table, **kwargs):
        calls["normalize"] = {
            "legacy_table": legacy_table,
            **kwargs,
        }
        return normalized_table

    monkeypatch.setattr(
        dpp_encoding,
        "normalize_legacy_dpp_encoding_table",
        normalize,
    )

    def write(**kwargs):
        calls["write"] = kwargs
        return {
            "analysis_file_name": "registered-dpp-encoding.nwb",
            "dpp_encoding_object_id": "dpp-encoding-object-id",
            "dpp_encoding_sha256": "f" * 64,
            "artifact_schema_version": "1",
            "_created_artifact_paths": ["/analysis/registered-dpp-encoding.nwb"],
        }

    monkeypatch.setitem(
        _register_existing_dpp_encoding_row.__globals__,
        "_write_dpp_encoding_nwb",
        write,
    )

    row = _register_existing_dpp_encoding_row(
        key={"dpp_encoding_id": comparison_id},
        dpp_encoding_path=source_path,
        overwrite=False,
        parameters_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        epoch_intervals_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        stability_table=object(),
        stability_selection_table=object(),
        tuning_curve_selection_table=object(),
        session_table=object(),
        source_v1ca1_git_commit="source-v1-commit",
        source_spyglass_git_commit="source-spyglass-commit",
        artifact_root=tmp_path / "analysis",
        analysis_nwbfile_table=analysis_nwbfile_table,
    )

    normalize_call = calls["normalize"]
    assert calls["source_path"] == source_path
    assert normalize_call["legacy_table"] is legacy_table
    assert normalize_call["unit_identity_resolver"] == {
        "101": {"spikesorting_merge_id": "merge-a", "unit_id": "10"},
        "102": {"spikesorting_merge_id": "merge-a", "unit_id": "11"},
    }
    assert normalize_call["spikes"] is loaded_spikes["ts_group"]
    assert normalize_call["stable_unit_ids"] is loaded_spikes["unit_ids"]
    assert normalize_call["movement_firing_rate_table"] is context[
        "movement"
    ]["table"]
    assert normalize_call["stability_tables_by_trajectory"] is context[
        "stability_tables"
    ]
    parameter_names = (
        "n_folds",
        "evaluation_bin_size_s",
        "spatial_bin_size_cm",
        "gaussian_smoothing_sigma_bins",
        "random_seed",
        "minimum_movement_firing_rate_hz",
        "minimum_stability_correlation",
    )
    assert {
        name: normalize_call[name] for name in parameter_names
    } == {
        name: context["parameters"][name] for name in parameter_names
    }
    assert calls["write"] == {
        "nwb_file_name": "L1420240102_.nwb",
        "table": normalized_table,
        "analysis_nwbfile_table": analysis_nwbfile_table,
    }

    expected_provenance = {
        "source_path": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(b"legacy encoding").hexdigest(),
        "source_v1ca1_git_commit": "source-v1-commit",
        "legacy_log_likelihood_units": "nats_per_spike",
        "canonical_log_likelihood_units": "total_nats",
        "eligible_unit_set_validated": True,
        "source_spyglass_git_commit": "source-spyglass-commit",
        "assumed_parameters": context["parameters"],
        "source_parameter_validation": {
            "verified_from_filename": [
                "n_folds",
                "evaluation_bin_size_s",
                "spatial_bin_size_cm",
            ],
            "recomputed_from_upstream": [
                "minimum_movement_firing_rate_hz",
                "minimum_stability_correlation",
            ],
            "caller_attested_not_encoded_in_legacy_artifact": [
                "gaussian_smoothing_sigma_bins",
                "random_seed",
            ],
        },
        "source_fold_qc_validation": (
            "not_reconstructable_from_legacy_summary"
        ),
    }
    summary = dpp_encoding.summarize_dpp_encoding_table(normalized_table)
    assert row == {
        "analysis_file_name": "registered-dpp-encoding.nwb",
        "dpp_encoding_object_id": "dpp-encoding-object-id",
        "dpp_encoding_sha256": "f" * 64,
        "artifact_schema_version": "1",
        "n_units_input": 2,
        "n_units_eligible": summary["n_units_eligible"],
        "n_units_valid": summary["n_units_valid"],
        "analysis_status": summary["analysis_status"],
        "eligible_units_sha256": summary["eligible_units_sha256"],
        "legacy_artifact_provenance": expected_provenance,
        "_created_artifact_paths": ["/analysis/registered-dpp-encoding.nwb"],
    }


@pytest.mark.parametrize(
    ("table_key", "selection_id_name", "hook_name", "register_kwargs"),
    [
        (
            "ripple_modulation",
            "ripple_modulation_id",
            "ripple_modulation_register_existing",
            {
                "summary_path": "old-summary.parquet",
                "peri_ripple_firing_rate_path": "old-peri.parquet",
            },
        ),
        (
            "path_specific_place_tuning_curve",
            "path_specific_place_tuning_curve_id",
            "path_specific_place_tuning_curve_register_existing",
            {"tuning_curve_path": "old-tuning-curve.nc"},
        ),
        (
            "dpp_tuning_curve",
            "dpp_tuning_curve_id",
            "dpp_tuning_curve_register_existing",
            {"tuning_curve_path": "old-dpp-tuning-curve.nc"},
        ),
        (
            "path_specific_place_tuning_similarity",
            "path_specific_place_tuning_similarity_id",
            "path_specific_place_tuning_similarity_register_existing",
            {"similarity_path": "old-similarity-all_units.parquet"},
        ),
        (
            "path_specific_place_stability",
            "path_specific_place_stability_id",
            "path_specific_place_stability_register_existing",
            {"stability_path": "old-stability.parquet"},
        ),
        (
            "dpp_encoding",
            "dpp_encoding_id",
            "dpp_encoding_register_existing",
            {"dpp_encoding_path": "old-encoding.parquet"},
        ),
        (
            "cv_pca",
            "cv_pca_id",
            "cv_pca_register_existing",
            {
                "legacy_result_path": "old-cv-pca.nc",
                "legacy_summary_path": "old-cv-pca-summary.parquet",
            },
        ),
        (
            "motor_encoding",
            "motor_encoding_id",
            "motor_encoding_register_existing",
            {
                "source_nested_cv_path": "old-nested.nc",
                "source_full_refit_path": "old-full-refit.nc",
            },
        ),
        (
            "swap_tuning_curve_comparison",
            "swap_tuning_curve_comparison_id",
            "swap_tuning_curve_comparison_register_existing",
            {
                "source_result_path": "old-swap-tuning.nc",
                "source_summary_path": "old-swap-tuning.parquet",
            },
        ),
    ],
)
def test_register_existing_rejects_overwrite_before_hook(
    monkeypatch,
    table_key: str,
    selection_id_name: str,
    hook_name: str,
    register_kwargs: dict[str, str],
) -> None:
    selection_id = uuid.uuid4()
    hook_calls = []

    def register_hook(**kwargs):
        hook_calls.append(kwargs)
        raise AssertionError("registration hook must not run")

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {selection_id_name: selection_id},
    )
    bundle, _, _ = _fake_bundle(runtime_hooks={hook_name: register_hook})
    result = bundle[table_key]

    with pytest.raises(ValueError, match="immutable"):
        result.register_existing(
            {selection_id_name: selection_id},
            overwrite=True,
            **register_kwargs,
        )

    assert hook_calls == []
    assert "_insert_calls" not in result.__dict__


@pytest.mark.parametrize(
    ("table_key", "selection_id_name", "hook_name", "register_kwargs"),
    [
        (
            "ripple_modulation",
            "ripple_modulation_id",
            "ripple_modulation_register_existing",
            {
                "summary_path": "old-summary.parquet",
                "peri_ripple_firing_rate_path": "old-peri.parquet",
            },
        ),
        (
            "path_specific_place_tuning_curve",
            "path_specific_place_tuning_curve_id",
            "path_specific_place_tuning_curve_register_existing",
            {"tuning_curve_path": "old-tuning-curve.nc"},
        ),
        (
            "dpp_tuning_curve",
            "dpp_tuning_curve_id",
            "dpp_tuning_curve_register_existing",
            {"tuning_curve_path": "old-dpp-tuning-curve.nc"},
        ),
        (
            "path_specific_place_tuning_similarity",
            "path_specific_place_tuning_similarity_id",
            "path_specific_place_tuning_similarity_register_existing",
            {"similarity_path": "old-similarity-all_units.parquet"},
        ),
        (
            "path_specific_place_stability",
            "path_specific_place_stability_id",
            "path_specific_place_stability_register_existing",
            {"stability_path": "old-stability.parquet"},
        ),
        (
            "dpp_encoding",
            "dpp_encoding_id",
            "dpp_encoding_register_existing",
            {"dpp_encoding_path": "old-encoding.parquet"},
        ),
        (
            "cv_pca",
            "cv_pca_id",
            "cv_pca_register_existing",
            {
                "legacy_result_path": "old-cv-pca.nc",
                "legacy_summary_path": "old-cv-pca-summary.parquet",
            },
        ),
        (
            "motor_encoding",
            "motor_encoding_id",
            "motor_encoding_register_existing",
            {
                "source_nested_cv_path": "old-nested.nc",
                "source_full_refit_path": "old-full-refit.nc",
            },
        ),
        (
            "swap_tuning_curve_comparison",
            "swap_tuning_curve_comparison_id",
            "swap_tuning_curve_comparison_register_existing",
            {
                "source_result_path": "old-swap-tuning.nc",
                "source_summary_path": "old-swap-tuning.parquet",
            },
        ),
    ],
)
def test_register_existing_preflights_duplicate_before_hook(
    monkeypatch,
    table_key: str,
    selection_id_name: str,
    hook_name: str,
    register_kwargs: dict[str, str],
) -> None:
    selection_id = uuid.uuid4()
    selection = {selection_id_name: selection_id}
    existing = {
        **selection,
        "artifact_origin": "registered_existing",
        "analysis_status": "valid",
    }
    hook_calls = []

    def register_hook(**kwargs):
        hook_calls.append(kwargs)
        raise AssertionError("registration hook must not run")

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(runtime_hooks={hook_name: register_hook})
    result = bundle[table_key]

    def existing_result(table, key):
        assert table is result
        assert dict(key) == selection
        return dict(existing)

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_existing_result_row",
        existing_result,
    )

    with pytest.raises(ValueError, match="already contains"):
        result.register_existing(selection, **register_kwargs)
    returned = result.register_existing(
        selection,
        skip_duplicates=True,
        **register_kwargs,
    )

    assert returned == existing
    assert hook_calls == []
    assert "_insert_calls" not in result.__dict__


@pytest.mark.parametrize(
    ("table_key", "selection_id_name", "hook_name", "artifact_specs"),
    [
        (
            "ripple_modulation",
            "ripple_modulation_id",
            "ripple_modulation_compute",
            (("analysis_file_name", "ripple-modulation-analysis.nwb"),),
        ),
        (
            "movement_firing_rate",
            "movement_firing_rate_id",
            "movement_firing_rate_compute",
            (("analysis_file_name", "movement-analysis.nwb"),),
        ),
        (
            "path_specific_place_tuning_curve",
            "path_specific_place_tuning_curve_id",
            "path_specific_place_tuning_curve_compute",
            (("analysis_file_name", "tuning-curve-analysis.nwb"),),
        ),
        (
            "dpp_tuning_curve",
            "dpp_tuning_curve_id",
            "dpp_tuning_curve_compute",
            (("analysis_file_name", "dpp-tuning-analysis.nwb"),),
        ),
        (
            "path_specific_place_tuning_similarity",
            "path_specific_place_tuning_similarity_id",
            "path_specific_place_tuning_similarity_compute",
            (("analysis_file_name", "tuning-similarity.nwb"),),
        ),
        (
            "path_specific_place_stability",
            "path_specific_place_stability_id",
            "path_specific_place_stability_compute",
            (("analysis_file_name", "stability-analysis.nwb"),),
        ),
        (
            "dpp_encoding",
            "dpp_encoding_id",
            "dpp_encoding_compute",
            (("analysis_file_name", "dpp-encoding-analysis.nwb"),),
        ),
        (
            "motor_encoding",
            "motor_encoding_id",
            "motor_encoding_compute",
            (("analysis_file_name", "motor-encoding-analysis.nwb"),),
        ),
        (
            "swap_tuning_curve_comparison",
            "swap_tuning_curve_comparison_id",
            "swap_tuning_curve_comparison_compute",
            (
                ("artifact_manifest_path", "manifest.parquet"),
                ("selected_units_path", "selected_units.parquet"),
                ("summary_path", "summary.parquet"),
                (
                    "swap_tuning_curve_comparison_path",
                    "swap_tuning.nc",
                ),
            ),
        ),
    ],
)
def test_failed_result_insert_removes_only_hook_reported_artifacts(
    tmp_path: Path,
    monkeypatch,
    table_key: str,
    selection_id_name: str,
    hook_name: str,
    artifact_specs: tuple[tuple[str, str], ...],
) -> None:
    selection_id = uuid.uuid4()
    artifact_dir = tmp_path / table_key
    artifact_dir.mkdir()
    created_paths = [
        artifact_dir / filename for _field_name, filename in artifact_specs
    ]
    retained_path = tmp_path / "preexisting.parquet"
    retained_path.write_bytes(b"keep")

    def compute(**kwargs):
        for path in created_paths:
            path.write_bytes(b"new")
        row = {
            field_name: str(path)
            for (field_name, _filename), path in zip(
                artifact_specs,
                created_paths,
                strict=True,
            )
        }
        row.update(
            {
                "n_units": 1,
                "n_valid_units": 1,
                "analysis_status": "valid",
                "selected_units_sha256": "c" * 64,
                "_created_artifact_paths": [str(path) for path in created_paths],
            }
        )
        if table_key == "ripple_modulation":
            row.update(
                {
                    "analysis_file_name": created_paths[0].name,
                    "ripple_modulation_summary_object_id": "summary-object-id",
                    "peri_ripple_firing_rate_object_id": "peri-object-id",
                    "ripple_modulation_summary_sha256": "d" * 64,
                    "peri_ripple_firing_rate_sha256": "e" * 64,
                    "artifact_schema_version": "1",
                    "n_ripples": 3,
                }
            )
        if table_key == "movement_firing_rate":
            row.update(
                {
                    "analysis_file_name": created_paths[0].name,
                    "movement_firing_rate_object_id": "rate-object-id",
                    "movement_intervals_object_id": "interval-object-id",
                    "movement_firing_rate_sha256": "d" * 64,
                    "movement_intervals_sha256": "e" * 64,
                    "artifact_schema_version": "1",
                    "n_units_with_spikes": 1,
                    "movement_interval_count": 2,
                    "movement_duration_s": 5.0,
                }
            )
        if table_key in {
            "path_specific_place_tuning_curve",
            "dpp_tuning_curve",
        }:
            row.update(
                {
                    "n_trials": 3,
                    "support_duration_s": 5.0,
                    "n_feature_samples": 50,
                    "n_position_bins": 25,
                }
            )
        if table_key == "dpp_tuning_curve":
            row.update(
                {
                    "analysis_file_name": created_paths[0].name,
                    "dpp_tuning_object_id": "dpp-tuning-object-id",
                    "dpp_bins_object_id": "dpp-bins-object-id",
                    "dpp_provenance_object_id": "dpp-provenance-object-id",
                    "dpp_tuning_sha256": "d" * 64,
                    "dpp_bins_sha256": "e" * 64,
                    "dpp_provenance_sha256": "f" * 64,
                    "artifact_schema_version": "1",
                    "n_outbound_trials": 2,
                    "n_inbound_trials": 1,
                }
            )
        if table_key == "path_specific_place_tuning_similarity":
            row.update(
                {
                    "analysis_file_name": created_paths[0].name,
                    "similarity_object_id": "similarity-object-id",
                    "similarity_sha256": "f" * 64,
                    "artifact_schema_version": "1",
                    "n_valid_comparisons": 4,
                    "n_units_with_valid_comparison": 1,
                }
            )
        if table_key == "dpp_encoding":
            row.pop("n_units")
            row.pop("n_valid_units")
            row.pop("selected_units_sha256")
            row.update(
                {
                    "analysis_file_name": created_paths[0].name,
                    "dpp_encoding_object_id": "dpp-encoding-object-id",
                    "dpp_encoding_sha256": "f" * 64,
                    "artifact_schema_version": "1",
                    "n_units_input": 2,
                    "n_units_eligible": 1,
                    "n_units_valid": 1,
                    "eligible_units_sha256": "c" * 64,
                }
            )
        if table_key == "motor_encoding":
            row.pop("n_units")
            row.pop("n_valid_units")
            row.update(
                {
                    "analysis_file_name": created_paths[0].name,
                    "selected_units_object_id": "selected-units-object-id",
                    "dataset_index_object_id": "dataset-index-object-id",
                    "coordinates_object_id": "coordinates-object-id",
                    "nested_cv_arrays_object_id": "nested-cv-arrays-object-id",
                    "full_refit_arrays_object_id": "full-refit-arrays-object-id",
                    "provenance_object_id": "provenance-object-id",
                    "artifact_schema_version": "1",
                    "schema_version": "2",
                    "n_units_input": 2,
                    "n_units_eligible": 1,
                    "n_units_valid": 1,
                    "n_outer_folds_expected": 5,
                    "n_outer_folds_valid": 5,
                }
            )
        if table_key == "swap_tuning_curve_comparison":
            row.update(
                {
                    "schema_version": "3",
                    "bundle_schema_version": "1",
                    "n_source_units": 1,
                    "legacy_artifact_provenance": None,
                }
            )
        return row

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {selection_id_name: selection_id},
    )
    bundle, _, _ = _fake_bundle(runtime_hooks={hook_name: compute})
    result = bundle[table_key]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make({selection_id_name: selection_id})

    assert all(not path.exists() for path in created_paths)
    assert not artifact_dir.exists()
    assert retained_path.read_bytes() == b"keep"


def test_register_existing_filters_partition_and_checks_parameters() -> None:
    parameters = dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    rows = pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240102", "20240102"],
            "epoch": ["02_r1", "02_r1"],
            "region": ["ca1", "v1"],
            "unit_id": [1, 2],
            "n_ripples": [5, 5],
            "bin_size_s": [0.02, 0.02],
            "time_before_s": [0.5, 0.5],
            "time_after_s": [0.5, 0.5],
        }
    )
    selected = _filter_registered_table(
        rows,
        artifact_name="summary",
        artifact_key={
            "animal_name": "L14",
            "date": "20240102",
            "epoch": "02_r1",
            "region": "ca1",
        },
        parameters=parameters,
    )
    assert selected["region"].tolist() == ["ca1"]

    with pytest.raises(ValueError, match="does not match"):
        _filter_registered_table(
            rows.assign(bin_size_s=0.04),
            artifact_name="summary",
            artifact_key={
                "animal_name": "L14",
                "date": "20240102",
                "epoch": "02_r1",
                "region": "ca1",
            },
            parameters=parameters,
        )


def test_legacy_unit_ids_map_to_canonical_composite_identity() -> None:
    table = pd.DataFrame({"unit_id": [101, 102], "n_ripples": [5, 5]})
    metadata = [
        {
            "spikesorting_merge_id": "merge-a",
            "unit_id": 12,
            "sorting_unit_id": 101,
        },
        {
            "spikesorting_merge_id": "merge-a",
            "unit_id": 13,
            "sorting_unit_id": 102,
        },
    ]

    keyed = _attach_registered_unit_identity(
        table,
        unit_metadata=metadata,
        artifact_name="summary",
    )

    assert keyed["group_unit_id"].tolist() == [101, 102]
    assert keyed["spikesorting_merge_id"].tolist() == ["merge-a", "merge-a"]
    assert keyed["unit_id"].tolist() == ["12", "13"]
    assert keyed["stable_unit_id"].tolist() == ["merge-a:12", "merge-a:13"]
    assert "nwb_unit_id" not in keyed
    with pytest.raises(ValueError, match="cannot be mapped uniquely"):
        _attach_registered_unit_identity(
            pd.DataFrame({"unit_id": [999]}),
            unit_metadata=metadata,
            artifact_name="summary",
        )


def test_empty_registered_units_keep_canonical_identity_schema() -> None:
    table = pd.DataFrame(columns=["unit_id", "invalid_reason"])

    keyed = _attach_registered_unit_identity(
        table,
        unit_metadata=[],
        artifact_name="summary",
    )

    assert keyed.empty
    assert {
        "spikesorting_merge_id",
        "unit_id",
        "stable_unit_id",
        "group_unit_id",
    }.issubset(keyed.columns)


def test_legacy_stability_requires_every_canonical_qc_column() -> None:
    from v1ca1.spyglass.stability import empty_stability_table

    legacy = empty_stability_table().rename(columns={"group_unit_id": "unit"})
    legacy = legacy.drop(
        columns=["spikesorting_merge_id", "unit_id", "stable_unit_id"]
    )
    _validate_legacy_stability_schema(legacy)

    with pytest.raises(ValueError, match="canonical columns.*n_odd_trials"):
        _validate_legacy_stability_schema(legacy.drop(columns="n_odd_trials"))


def test_ripple_metadata_is_preserved_and_detector_provenance_is_checked() -> None:
    class Intervals:
        def as_dataframe(self):
            return pd.DataFrame(
                {"start": [1.0], "end": [1.1], "mean_zscore": [3.2]}
            )

    frame = _intervals_to_frame(Intervals(), epoch="02_r1")
    assert frame.to_dict("records") == [
        {
            "start_time": 1.0,
            "end_time": 1.1,
            "mean_zscore": 3.2,
            "epoch": "02_r1",
        }
    ]

    parameters = dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    _validate_ripple_provenance(
        {"detector_zscore_threshold": 2.0, "speed_gated": True},
        parameters,
    )
    with pytest.raises(ValueError, match="zscore_threshold"):
        _validate_ripple_provenance(
            {"detector_zscore_threshold": 3.0, "speed_gated": True},
            parameters,
        )
    with pytest.raises(ValueError, match="speed-gated"):
        _validate_ripple_provenance(
            {"detector_zscore_threshold": 2.0, "speed_gated": False},
            parameters,
        )


def test_sorting_session_resolution_uses_curated_lineage_fallback() -> None:
    class Parent:
        heading = SimpleNamespace(names=("sorting_id", "curation_id"))

    class SortGroupInfo:
        heading = SimpleNamespace(names=("nwb_file_name", "region_name"))

        def fetch(self, name):
            assert name == "nwb_file_name"
            return ["L1420240102_.nwb", "L1420240102_.nwb"]

    class SortingOutput:
        def merge_get_parent(self, key):
            assert key == {"merge_id": "merge-1"}
            return Parent()

        def get_sort_group_info(self, key):
            assert key == {"merge_id": "merge-1"}
            return SortGroupInfo()

    assert _sorting_output_sessions(
        SortingOutput(),
        merge_id="merge-1",
    ) == {"L1420240102_.nwb"}


def test_snapshot_digest_matches_membership_content() -> None:
    provenance = _sorting_provenance()
    expected = hashlib.sha256(b"merge-a\0merge-b").hexdigest()

    assert provenance["sorting_group_members_sha256"] == expected
    assert len(provenance["unit_selection_params_sha256"]) == 64


def _epoch_motor_behavior_selection_inputs() -> dict[str, Any]:
    """Return complete database-free sources for one motor selection."""
    nwb_file_name = "L1420240102_augmented.nwb"
    epoch = "02_r1"
    timestamps = 10.0 + np.arange(20, dtype=float) * 0.1
    primary_values = np.column_stack(
        (np.arange(20, dtype=float), np.zeros(20, dtype=float))
    )
    reference_values = primary_values - np.asarray([1.0, 0.0])
    common_position = {
        "nwb_file_name": nwb_file_name,
        "epoch": epoch,
        "spatial_unit": "cm",
        "start_index": 100,
        "stop_index_exclusive": 130,
        "sample_count": 30,
        "analysis_start_offset_samples": 10,
        "start_time": 9.0,
        "stop_time": 11.9,
        "first_frame": 500,
        "last_frame": 529,
        "video_series_name": "camera",
        "source_table_path": "processing/behavior/position_index",
    }
    position_rows = {
        "future_head": {
            **common_position,
            "position_series_name": "future_head",
            "position_role": "translation_anchor",
            "source_object_id": "primary-id",
        },
        "future_reference": {
            **common_position,
            "position_series_name": "future_reference",
            "position_role": "orientation_anchor",
            "source_object_id": "reference-id",
        },
    }
    trajectory_bounds = {
        "center_to_left": (10.0, 10.4),
        "center_to_right": (10.5, 10.9),
        "left_to_center": (11.0, 11.4),
        "right_to_center": (11.5, 11.9),
    }
    trajectory_rows = {
        trajectory_type: {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "trajectory_type": trajectory_type,
            "interval_count": 1,
            "source_table_path": f"intervals/{trajectory_type}",
            "source_object_id": f"trajectory-{trajectory_type}",
        }
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    trajectory_intervals = {
        trajectory_type: _TestIntervals([start], [stop])
        for trajectory_type, (start, stop) in trajectory_bounds.items()
    }
    graph_inputs = {
        trajectory_type: {
            "configuration_name": trajectory_type,
            "coordinate_unit": "cm",
            "use_hmm": False,
            "track_graph_kwargs": {
                "node_positions": [[0.0, 0.0], [100.0, 0.0]],
                "edges": [[0, 1]],
            },
            "linearization_kwargs": {
                "edge_order": [[0, 1]],
                "edge_spacing": [],
                "use_HMM": False,
            },
        }
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    graph_rows = {
        trajectory_type: {
            "nwb_file_name": nwb_file_name,
            "configuration_name": trajectory_type,
            "coordinate_unit": "cm",
            "source_object_path": f"processing/behavior/{trajectory_type}",
            "source_object_id": f"graph-{trajectory_type}",
        }
        for trajectory_type in _DPP_ENCODING_TRAJECTORIES
    }
    return {
        "key": {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "primary_position_series_name": "future_head",
            "orientation_reference_position_series_name": "future_reference",
            "movement_param_name": "default",
            "epoch_motor_behavior_param_name": "manuscript_4cm",
        },
        "epoch_row": {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "epoch_type": "run",
            "start_time": 10.0,
            "stop_time": 12.0,
            "condition": "AB",
            "source_object_id": "epoch-id",
        },
        "position_rows": position_rows,
        "position_inputs": {
            "primary_position": SimpleNamespace(
                t=timestamps, d=primary_values
            ),
            "orientation_reference_position": SimpleNamespace(
                t=timestamps, d=reference_values
            ),
        },
        "trajectory_rows": trajectory_rows,
        "trajectory_intervals": trajectory_intervals,
        "graph_rows": graph_rows,
        "graph_inputs": graph_inputs,
        "movement_parameters": dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS),
        "parameters": dict(
            table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS
        ),
    }


def _build_epoch_motor_behavior_selection(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build one motor selection from injectable fake relations."""
    return tables_module._epoch_motor_behavior_selection_row(
        key=inputs["key"],
        epoch_intervals_table=_FakeRelation(inputs["epoch_row"]),
        position_table=_FakeKeyedRelation(
            "position_series_name", inputs["position_rows"]
        ),
        movement_parameters_table=_FakeRelation(
            inputs["movement_parameters"]
        ),
        trajectory_intervals_table=_FakeKeyedRelation(
            "trajectory_type", inputs["trajectory_rows"]
        ),
        wtrack_graph_table=_FakeKeyedRelation(
            "configuration_name", inputs["graph_rows"]
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
        position_inputs=inputs["position_inputs"],
        trajectory_interval_sets=inputs["trajectory_intervals"],
        graph_inputs=inputs["graph_inputs"],
    )


def test_epoch_motor_behavior_selection_freezes_every_nwb_source() -> None:
    """Epoch, position, lap, graph, movement, and output inputs enter UUIDv5."""
    inputs = _epoch_motor_behavior_selection_inputs()
    first = _build_epoch_motor_behavior_selection(inputs)
    assert first == _build_epoch_motor_behavior_selection(inputs)
    assert first["epoch_motor_behavior_id"].version == 5
    assert first["primary_position_role"] == "translation_anchor"
    assert first["orientation_reference_position_role"] == "orientation_anchor"
    assert first["position_offset_samples"] == 10
    for field_name in (
        "epoch_interval_row_sha256",
        "primary_position_source_sha256",
        "orientation_reference_position_source_sha256",
        "movement_parameters_sha256",
        "epoch_motor_behavior_parameters_sha256",
        "epoch_motor_behavior_output_rule_sha256",
    ):
        assert len(first[field_name]) == 64
    for field_name in (
        "trajectory_rows_sha256_by_type",
        "trajectory_intervals_sha256_by_type",
        "graph_rows_sha256_by_trajectory",
        "graph_inputs_sha256_by_trajectory",
    ):
        assert set(first[field_name]) == set(_DPP_ENCODING_TRAJECTORIES)

    changed_epoch = copy.deepcopy(inputs)
    changed_epoch["epoch_row"]["condition"] = "BA"
    assert _build_epoch_motor_behavior_selection(changed_epoch)[
        "epoch_motor_behavior_id"
    ] != first["epoch_motor_behavior_id"]

    changed_position = copy.deepcopy(inputs)
    changed_position["position_inputs"]["primary_position"].d[0, 0] += 0.5
    assert _build_epoch_motor_behavior_selection(changed_position)[
        "epoch_motor_behavior_id"
    ] != first["epoch_motor_behavior_id"]

    changed_trajectory_row = copy.deepcopy(inputs)
    changed_trajectory_row["trajectory_rows"]["center_to_left"][
        "source_object_id"
    ] = "changed-trajectory"
    assert _build_epoch_motor_behavior_selection(changed_trajectory_row)[
        "epoch_motor_behavior_id"
    ] != first["epoch_motor_behavior_id"]

    changed_interval = copy.deepcopy(inputs)
    changed_interval["trajectory_intervals"]["center_to_left"].end[0] += 0.01
    assert _build_epoch_motor_behavior_selection(changed_interval)[
        "epoch_motor_behavior_id"
    ] != first["epoch_motor_behavior_id"]

    changed_graph = copy.deepcopy(inputs)
    for graph in changed_graph["graph_inputs"].values():
        graph["track_graph_kwargs"]["node_positions"][1][0] = 101.0
    assert _build_epoch_motor_behavior_selection(changed_graph)[
        "epoch_motor_behavior_id"
    ] != first["epoch_motor_behavior_id"]


def test_epoch_motor_behavior_selection_requires_fixed_movement_and_run() -> None:
    """The upstream movement definition and run classification are fixed."""
    inputs = _epoch_motor_behavior_selection_inputs()
    inputs["movement_parameters"]["speed_threshold_cm_s"] = 5.0
    with pytest.raises(ValueError, match="manuscript MovementParameters"):
        _build_epoch_motor_behavior_selection(inputs)

    inputs = _epoch_motor_behavior_selection_inputs()
    inputs["epoch_row"]["epoch_type"] = "sleep"
    with pytest.raises(ValueError, match="explicit run epoch"):
        _build_epoch_motor_behavior_selection(inputs)


def test_epoch_motor_behavior_activation_definitions_and_defaults() -> None:
    """Activation exposes the three passive tables and inserts nothing."""
    bundle, _schemas, _unit_selection_params = _fake_bundle()
    assert {
        "epoch_motor_behavior_parameters",
        "epoch_motor_behavior_selection",
        "epoch_motor_behavior",
    }.issubset(bundle)
    selection_definition = bundle["epoch_motor_behavior_selection"].definition
    assert "epoch_motor_behavior_id: uuid" in selection_definition
    assert "primary_position_series_name='position_series_name'" in (
        selection_definition
    )
    assert "trajectory_rows_sha256_by_type: longblob" in selection_definition
    assert "graph_inputs_sha256_by_trajectory: longblob" in (
        selection_definition
    )
    result = bundle["epoch_motor_behavior"]
    assert "-> AnalysisNwbfile" in result.definition
    for field_name in (
        "distribution_summary",
        "progression_summary",
        "trajectory_qc",
    ):
        assert f"{field_name}_object_id: varchar(40)" in result.definition
        assert f"{field_name}_sha256: char(64)" in result.definition
    assert "artifact_schema_version: varchar(8)" in result.definition
    assert "artifact_manifest_path:" not in result.definition
    assert "distribution_summary_path:" not in result.definition
    assert hasattr(result, "register_existing")
    assert "_insert_calls" not in result.__dict__

    parameters = bundle["epoch_motor_behavior_parameters"]
    inserted = parameters.insert_default()
    assert inserted == dict(
        table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS
    )


def test_epoch_motor_behavior_load_validates_artifact_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public loader delegates all rows to the NWB result validator."""
    result_id = uuid.uuid4()
    result_row = {
        "epoch_motor_behavior_id": result_id,
        "analysis_file_name": "epoch-motor-analysis.nwb",
        "artifact_schema_version": "1",
    }
    selection = {
        "epoch_motor_behavior_id": result_id,
        "movement_param_name": "default",
        "epoch_motor_behavior_param_name": "manuscript_4cm",
    }
    parameters = dict(
        table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS
    )
    movement = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    bundle_payload = {"loaded": True}
    loader_calls = []

    bundle, _schemas, _unit_selection_params = _fake_bundle()
    result = bundle["epoch_motor_behavior"]
    selection_table = bundle["epoch_motor_behavior_selection"]
    parameters_table = bundle["epoch_motor_behavior_parameters"]
    movement_table = bundle["movement_parameters"]

    def fetch(table, key):
        if table is result:
            return dict(result_row)
        if table is selection_table:
            return dict(selection)
        if table is parameters_table:
            return dict(parameters)
        if table is movement_table:
            return dict(movement)
        raise AssertionError(f"Unexpected table {table!r}")

    monkeypatch.setitem(_construct_tables.__globals__, "_fetch1_dict", fetch)
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_load_epoch_motor_behavior_result",
        lambda **kwargs: loader_calls.append(kwargs) or bundle_payload,
    )

    loaded = result.load_epoch_motor_behavior_bundle(
        {"epoch_motor_behavior_id": result_id}
    )
    assert loaded is bundle_payload
    assert loader_calls[0]["result_row"] == result_row
    assert loader_calls[0]["selection_row"] == selection
    assert loader_calls[0]["epoch_motor_behavior_table"] is result


@pytest.mark.parametrize(
    "validation_errors",
    [None, ["invalid EpochMotorBehavior NWB"]],
)
def test_epoch_motor_behavior_write_uses_analysis_nwb_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_errors: list[str] | None,
) -> None:
    """Three motor tables are validated before one NWB is registered."""
    import pynwb

    from v1ca1.spyglass import epoch_motor_behavior

    result = _valid_epoch_motor_behavior_result()
    analysis_path = tmp_path / "epoch-motor-behavior-analysis.nwb"
    analysis_table = _TestAnalysisNwbfile(analysis_path)
    real_validate = pynwb.validate

    def validate_before_registration(*, path):
        assert not analysis_table.builder.registered
        assert analysis_path.stat().st_mode & 0o777 == 0o644
        if validation_errors is not None:
            return validation_errors
        return real_validate(path=path)

    monkeypatch.setattr(pynwb, "validate", validate_before_registration)
    kwargs = {
        "nwb_file_name": "L1420240102_.nwb",
        "result": result,
        "analysis_nwbfile_table": analysis_table,
    }
    if validation_errors is not None:
        with pytest.raises(ValueError, match="failed PyNWB validation"):
            tables_module._write_epoch_motor_behavior_nwb(**kwargs)
        assert not analysis_table.builder.registered
        assert not analysis_path.exists()
        return

    row = tables_module._write_epoch_motor_behavior_nwb(**kwargs)
    assert analysis_table.builder.registered
    assert analysis_path.exists()
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == (
        epoch_motor_behavior.NWB_ARTIFACT_SCHEMA_VERSION
    )
    object_ids = {
        row[f"{name}_object_id"]
        for name in (
            "distribution_summary",
            "progression_summary",
            "trajectory_qc",
        )
    }
    assert len(object_ids) == 3
    assert row["distribution_summary_sha256"] == (
        epoch_motor_behavior.distribution_summary_sha256(
            result["distribution_summary"]
        )
    )
    assert row["progression_summary_sha256"] == (
        epoch_motor_behavior.progression_summary_sha256(
            result["progression_summary"]
        )
    )
    assert row["trajectory_qc_sha256"] == (
        epoch_motor_behavior.trajectory_qc_sha256(result["trajectory_qc"])
    )


def test_epoch_motor_behavior_result_loader_uses_fetch_nwb() -> None:
    """Live motor readers resolve and cross-check all three NWB tables."""
    from v1ca1.spyglass import epoch_motor_behavior

    bundle = _valid_epoch_motor_behavior_result()
    result_id = uuid.UUID(bundle["metadata"]["epoch_motor_behavior_id"])

    class EpochMotorBehaviorRelation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [
                {
                    "distribution_summary": bundle[
                        "distribution_summary"
                    ].copy(),
                    "progression_summary": bundle[
                        "progression_summary"
                    ].copy(),
                    "trajectory_qc": bundle["trajectory_qc"].copy(),
                }
            ]

    relation = EpochMotorBehaviorRelation()
    result_row = {
        "epoch_motor_behavior_id": result_id,
        "artifact_schema_version": (
            epoch_motor_behavior.NWB_ARTIFACT_SCHEMA_VERSION
        ),
        "distribution_summary_sha256": (
            epoch_motor_behavior.distribution_summary_sha256(
                bundle["distribution_summary"]
            )
        ),
        "progression_summary_sha256": (
            epoch_motor_behavior.progression_summary_sha256(
                bundle["progression_summary"]
            )
        ),
        "trajectory_qc_sha256": epoch_motor_behavior.trajectory_qc_sha256(
            bundle["trajectory_qc"]
        ),
        **{
            name: bundle[name]
            for name in (
                "n_position_samples_input",
                "n_finite_position_samples",
                "n_dropped_nonfinite_samples",
                "n_movement_samples",
                "movement_duration_s",
                "n_supported_trajectories",
                "sampling_rate_hz",
                "median_sample_interval_s",
                "maximum_sample_gap_s",
                "analysis_status",
                "artifact_origin",
                "legacy_artifact_provenance",
            )
        },
    }
    selection = {
        "epoch_motor_behavior_id": result_id,
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "primary_position_series_name": "head_position",
        "primary_position_role": "translation_anchor",
        "orientation_reference_position_series_name": "body_position",
        "orientation_reference_position_role": "orientation_reference",
        "position_offset_samples": 0,
        "epoch_motor_behavior_parameters_sha256": bundle["parameters"][
            "parameter_sha256"
        ],
        "epoch_motor_behavior_output_rule_sha256": bundle["parameters"][
            "output_rule_sha256"
        ],
        "movement_parameters_sha256": bundle["movement_parameters"][
            "movement_parameters_sha256"
        ],
    }
    loaded = tables_module._load_epoch_motor_behavior_result(
        result_row=result_row,
        epoch_motor_behavior_table=relation,
        selection_row=selection,
        parameters_row=dict(
            table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS
        ),
        movement_parameters_row=dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS),
        animal_name="L14",
        date="20240102",
    )
    pd.testing.assert_frame_equal(
        loaded["distribution_summary"],
        bundle["distribution_summary"],
        check_dtype=False,
        check_categorical=False,
    )
    assert relation.keys == [{"epoch_motor_behavior_id": result_id}]

    with pytest.raises(ValueError, match="distribution_summary_sha256"):
        tables_module._load_epoch_motor_behavior_result(
            result_row={
                **result_row,
                "distribution_summary_sha256": "0" * 64,
            },
            epoch_motor_behavior_table=relation,
            selection_row=selection,
            parameters_row=dict(
                table_specs.MANUSCRIPT_EPOCH_MOTOR_BEHAVIOR_PARAMETERS
            ),
            movement_parameters_row=dict(
                table_specs.DEFAULT_MOVEMENT_PARAMETERS
            ),
            animal_name="L14",
            date="20240102",
        )


def test_epoch_motor_behavior_failed_insert_removes_new_analysis_nwb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed result insert removes only the hook-reported analysis NWB."""
    result_id = uuid.uuid4()
    artifact_path = tmp_path / "epoch-motor-analysis.nwb"
    retained = tmp_path / "retained.parquet"
    retained.write_bytes(b"keep")

    def compute(**_kwargs):
        artifact_path.write_bytes(b"new")
        return {
            "analysis_file_name": artifact_path.name,
            "distribution_summary_object_id": "distribution-object-id",
            "progression_summary_object_id": "progression-object-id",
            "trajectory_qc_object_id": "qc-object-id",
            "distribution_summary_sha256": "a" * 64,
            "progression_summary_sha256": "b" * 64,
            "trajectory_qc_sha256": "c" * 64,
            "artifact_schema_version": "1",
            "n_position_samples_input": 20,
            "n_finite_position_samples": 20,
            "n_dropped_nonfinite_samples": 0,
            "n_movement_samples": 20,
            "movement_duration_s": 2.0,
            "n_supported_trajectories": 4,
            "sampling_rate_hz": 10.0,
            "median_sample_interval_s": 0.1,
            "maximum_sample_gap_s": 0.1,
            "analysis_status": "valid",
            "legacy_artifact_provenance": None,
            "_created_artifact_paths": [str(artifact_path)],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {"epoch_motor_behavior_id": result_id},
    )
    bundle, _schemas, _unit_selection_params = _fake_bundle(
        runtime_hooks={"epoch_motor_behavior_compute": compute}
    )
    result = bundle["epoch_motor_behavior"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make({"epoch_motor_behavior_id": result_id})
    assert not artifact_path.exists()
    assert retained.read_bytes() == b"keep"


def test_epoch_motor_behavior_make_requires_populate_transaction() -> None:
    """Direct make cannot register a motor NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "epoch_motor_behavior_compute": lambda **kwargs: pytest.fail(
                "EpochMotorBehavior computation must not start outside populate()."
            )
        }
    )
    result = bundle["epoch_motor_behavior"]
    result.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"epoch_motor_behavior_id": uuid.uuid4()})


def test_epoch_motor_behavior_registration_hook_delegates_strict_recompute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table hook calls the standalone NWB-recomputation registrar only."""
    from v1ca1.spyglass import epoch_motor_behavior

    result_id = uuid.uuid4()
    selection = {
        "epoch_motor_behavior_id": result_id,
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
    }
    context = {
        "selection": selection,
        "parameters": {
            "epoch_motor_behavior_param_name": "manuscript_4cm",
            "progression_bin_size_cm": 4.0,
        },
        "movement_parameters": {
            **dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS),
            "movement_parameters_sha256": "m" * 64,
        },
        "animal_name": "L14",
        "date": "20240102",
        "position_inputs": {
            "primary_position": object(),
            "orientation_reference_position": object(),
        },
        "primary_position_row": {"position_series_name": "future_head"},
        "orientation_reference_position_row": {
            "position_series_name": "future_reference"
        },
        "trajectory_intervals": {"four": "intervals"},
        "graph_inputs": {"four": "graphs"},
    }
    selection.update(
        {
            "epoch_motor_behavior_parameters_sha256": "p" * 64,
            "epoch_motor_behavior_output_rule_sha256": "o" * 64,
            "movement_parameters_sha256": "m" * 64,
        }
    )
    captured = []
    writer_calls = []

    def register(**kwargs):
        captured.append(kwargs)
        return {
            "distribution_summary": pd.DataFrame(),
            "progression_summary": pd.DataFrame(),
            "trajectory_qc": pd.DataFrame(),
            "n_position_samples_input": 20,
            "n_finite_position_samples": 20,
            "n_dropped_nonfinite_samples": 0,
            "n_movement_samples": 20,
            "movement_duration_s": 2.0,
            "n_supported_trajectories": 4,
            "sampling_rate_hz": 10.0,
            "median_sample_interval_s": 0.1,
            "maximum_sample_gap_s": 0.1,
            "analysis_status": "valid",
            "legacy_artifact_provenance": {"verification": "recomputed"},
        }

    def write_nwb(**kwargs):
        writer_calls.append(kwargs)
        return {
            "analysis_file_name": "registered-motor.nwb",
            "distribution_summary_object_id": "distribution-object-id",
            "progression_summary_object_id": "progression-object-id",
            "trajectory_qc_object_id": "qc-object-id",
            "distribution_summary_sha256": "a" * 64,
            "progression_summary_sha256": "b" * 64,
            "trajectory_qc_sha256": "c" * 64,
            "artifact_schema_version": "1",
            "_created_artifact_paths": ["registered-motor.nwb"],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_load_epoch_motor_behavior_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        epoch_motor_behavior,
        "register_existing_epoch_motor_behavior_artifact",
        register,
    )
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_write_epoch_motor_behavior_nwb",
        write_nwb,
    )
    source_distribution = tmp_path / "legacy_distribution.parquet"
    source_progression = tmp_path / "legacy_progression.parquet"
    row = tables_module._register_existing_epoch_motor_behavior_row(
        key=selection,
        source_distribution_path=source_distribution,
        source_progression_path=source_progression,
        source_run_log_path=None,
        parameters_table=object(),
        epoch_intervals_table=object(),
        position_table=object(),
        movement_parameters_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        artifact_root=tmp_path,
        analysis_nwbfile_table="analysis-table",
    )
    assert captured[0]["source_distribution_path"] == source_distribution
    assert captured[0]["source_progression_path"] == source_progression
    assert captured[0]["primary_position"] is (
        context["position_inputs"]["primary_position"]
    )
    assert captured[0]["parameter_sha256"] == "p" * 64
    assert captured[0]["destination_path"] is None
    assert captured[0]["write_artifact"] is False
    assert writer_calls[0]["nwb_file_name"] == "L1420240102_.nwb"
    assert writer_calls[0]["analysis_nwbfile_table"] == "analysis-table"
    assert row["legacy_artifact_provenance"] == {
        "verification": "recomputed"
    }
    assert row["analysis_file_name"] == "registered-motor.nwb"
    assert row["_created_artifact_paths"] == ["registered-motor.nwb"]


def test_epoch_motor_behavior_registration_uses_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy validation and both motor result inserts are transactional."""
    result_id = uuid.uuid4()
    selection = {
        "epoch_motor_behavior_id": result_id,
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
    }
    events = []

    class Connection:
        in_transaction = False

        @property
        def transaction(self):
            connection = self

            class Transaction:
                def __enter__(self):
                    events.append("transaction_enter")
                    connection.in_transaction = True

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("transaction_exit")
                    connection.in_transaction = False

            return Transaction()

    connection = Connection()

    def register(**kwargs):
        assert connection.in_transaction
        events.append("register_hook")
        return {
            "analysis_file_name": "registered-motor.nwb",
            "distribution_summary_object_id": "distribution-object-id",
            "progression_summary_object_id": "progression-object-id",
            "trajectory_qc_object_id": "qc-object-id",
            "distribution_summary_sha256": "a" * 64,
            "progression_summary_sha256": "b" * 64,
            "trajectory_qc_sha256": "c" * 64,
            "artifact_schema_version": "1",
            "n_position_samples_input": 20,
            "n_finite_position_samples": 0,
            "n_dropped_nonfinite_samples": 20,
            "n_movement_samples": 0,
            "movement_duration_s": 0.0,
            "n_supported_trajectories": 0,
            "sampling_rate_hz": np.nan,
            "median_sample_interval_s": np.nan,
            "maximum_sample_gap_s": np.nan,
            "analysis_status": "no_valid_position",
            "legacy_artifact_provenance": {"source": "legacy"},
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: dict(selection),
    )
    bundle, _, _ = _fake_bundle(
        runtime_hooks={"epoch_motor_behavior_register_existing": register}
    )
    result = bundle["epoch_motor_behavior"]
    result.connection = connection

    row = result.register_existing(
        {"epoch_motor_behavior_id": result_id},
        source_distribution_path="legacy-distribution.parquet",
        source_progression_path="legacy-progression.parquet",
    )

    assert events == [
        "transaction_enter",
        "register_hook",
        "transaction_exit",
    ]
    assert row["analysis_file_name"] == "registered-motor.nwb"


def _cv_pca_selection_inputs(tmp_path: Path) -> dict[str, Any]:
    """Return complete database-free sources for one cvPCA selection."""
    nwb_file_name = "L1420240102_augmented.nwb"
    epochs = {"light": "02_r1", "dark": "08_r4"}
    group_id = uuid.uuid4()
    movement_ids = {condition: uuid.uuid4() for condition in epochs}
    unit_ids = [
        {"spikesorting_merge_id": "merge-a", "unit_id": "1"},
        {"spikesorting_merge_id": "merge-a", "unit_id": "2"},
    ]
    selected_units_sha256 = unit_identity_sha256(unit_ids)
    group_row = {
        "region_sorted_spikes_group_id": group_id,
        "nwb_file_name": nwb_file_name,
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": "all shanks",
        "region_name": "v1",
        "sorting_group_members": ["merge-a"],
        "sorting_group_members_sha256": "a" * 64,
        "unit_filter_include_labels": [],
        "unit_filter_exclude_labels": [],
        "unit_filter_params_sha256": "b" * 64,
        "n_units": 2,
        "selected_units_sha256": selected_units_sha256,
    }
    epoch_rows = {
        "light": {
            "nwb_file_name": nwb_file_name,
            "epoch": epochs["light"],
            "start_time": 10.0,
            "stop_time": 20.0,
            "nwb_epoch_start_time": 10.0,
            "nwb_epoch_stop_time": 20.0,
            "epoch_type": "run",
            "condition": "AB",
            "is_light": True,
            "source_object_id": "light-epoch",
        },
        "dark": {
            "nwb_file_name": nwb_file_name,
            "epoch": epochs["dark"],
            "start_time": 30.0,
            "stop_time": 40.0,
            "nwb_epoch_start_time": 30.0,
            "nwb_epoch_stop_time": 40.0,
            "epoch_type": "run",
            "condition": "dark",
            "is_light": False,
            "source_object_id": "dark-epoch",
        },
    }
    movement_parameters = dict(table_specs.DEFAULT_MOVEMENT_PARAMETERS)
    movement_parameters_sha256 = provenance_sha256(movement_parameters)
    movement_selections = {}
    movement_results = {}
    movement_artifacts = {}
    position_rows = {}
    position_inputs = {}
    for condition, epoch in epochs.items():
        position_name = f"head_{epoch}"
        movement_selections[condition] = {
            "movement_firing_rate_id": movement_ids[condition],
            "region_sorted_spikes_group_id": group_id,
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": position_name,
            "movement_param_name": "default",
            "unit_filter_params_name": "all_units",
            "sorted_spikes_group_name": "all shanks",
            "region": "v1",
            "sorting_group_members": ["merge-a"],
            "sorting_group_members_sha256": "a" * 64,
            "unit_filter_include_labels": [],
            "unit_filter_exclude_labels": [],
            "unit_filter_params_sha256": "b" * 64,
            "movement_parameters_sha256": movement_parameters_sha256,
        }
        movement_results[condition] = {
            "movement_firing_rate_id": movement_ids[condition],
            "analysis_file_name": f"{condition}-movement.nwb",
            "movement_firing_rate_object_id": f"{condition}-rate-object",
            "movement_intervals_object_id": f"{condition}-interval-object",
            "movement_firing_rate_sha256": hashlib.sha256(
                f"{condition}-rates".encode()
            ).hexdigest(),
            "movement_intervals_sha256": hashlib.sha256(
                f"{condition}-intervals".encode()
            ).hexdigest(),
            "artifact_schema_version": "1",
            "n_units": 2,
            "n_valid_units": 2,
            "n_units_with_spikes": 2,
            "movement_interval_count": 1,
            "movement_duration_s": 8.0,
            "analysis_status": "valid",
            "selected_units_sha256": selected_units_sha256,
        }
        table = pd.DataFrame(
            {
                "spikesorting_merge_id": ["merge-a", "merge-a"],
                "unit_id": ["1", "2"],
                "stable_unit_id": ["merge-a:1", "merge-a:2"],
                "movement_firing_rate_hz": [1.0, 1.5],
                "firing_rate_status": ["valid", "valid"],
            }
        )
        start = 10.0 if condition == "light" else 30.0
        movement_artifacts[condition] = {
            "table": table,
            "movement_intervals": _TestIntervals(
                [start + 1.0], [start + 9.0]
            ),
            "analysis_status": "valid",
        }
        timestamps = start + np.arange(100, dtype=float) * 0.1
        values = np.column_stack(
            (np.arange(100, dtype=float) * 0.2, np.zeros(100))
        )
        position_rows[condition] = {
            "nwb_file_name": nwb_file_name,
            "epoch": epoch,
            "position_series_name": position_name,
            "position_role": "head",
            "start_index": 0,
            "stop_index_exclusive": 100,
            "sample_count": 100,
            "analysis_start_offset_samples": 10,
            "start_time": float(timestamps[0]),
            "stop_time": float(timestamps[-1]),
            "spatial_unit": "cm",
            "source_object_id": f"{condition}-position",
        }
        position_inputs[condition] = SimpleNamespace(t=timestamps, d=values)

    trajectory_rows = []
    trajectory_intervals = {"light": {}, "dark": {}}
    for condition, epoch in epochs.items():
        base = 10.5 if condition == "light" else 30.5
        for trajectory_index, trajectory in enumerate(
            _DPP_ENCODING_TRAJECTORIES
        ):
            trajectory_rows.append(
                {
                    "nwb_file_name": nwb_file_name,
                    "epoch": epoch,
                    "trajectory_type": trajectory,
                    "interval_count": 4,
                    "source_object_id": f"{condition}-{trajectory}",
                }
            )
            starts = base + trajectory_index * 0.1 + np.arange(4) * 2.0
            trajectory_intervals[condition][trajectory] = (
                _TestIntervals(starts, starts + 0.5)
            )
    graph_inputs = {
        trajectory: {
            "configuration_name": trajectory,
            "coordinate_unit": "cm",
            "track_graph_kwargs": {
                "node_positions": [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
                "edges": [[0, 1], [1, 2]],
            },
            "linearization_kwargs": {
                "edge_order": [[0, 1], [1, 2]],
                "edge_spacing": [0.0],
                "use_HMM": False,
            },
        }
        for trajectory in ("center_to_left", "center_to_right")
    }
    graph_rows = {
        trajectory: {
            "nwb_file_name": nwb_file_name,
            "configuration_name": trajectory,
            "coordinate_unit": "cm",
            "use_hmm": False,
            "source_object_id": f"graph-{trajectory}",
        }
        for trajectory in graph_inputs
    }
    key = {
        "nwb_file_name": nwb_file_name,
        "light_epoch": epochs["light"],
        "dark_epoch": epochs["dark"],
        "region_sorted_spikes_group_id": group_id,
        "light_movement_firing_rate_id": movement_ids["light"],
        "dark_movement_firing_rate_id": movement_ids["dark"],
        "cv_pca_param_name": "manuscript_v1_seed47",
    }
    return {
        "key": key,
        "epoch_rows": epoch_rows,
        "group_row": group_row,
        "movement_parameters": movement_parameters,
        "movement_selections": movement_selections,
        "movement_results": movement_results,
        "movement_artifacts": movement_artifacts,
        "position_rows": position_rows,
        "position_inputs": position_inputs,
        "trajectory_rows": trajectory_rows,
        "trajectory_intervals": trajectory_intervals,
        "graph_rows": graph_rows,
        "graph_inputs": graph_inputs,
        "parameters": dict(table_specs.MANUSCRIPT_V1_CV_PCA_PARAMETERS),
    }


def _build_cv_pca_selection(inputs: dict[str, Any]) -> dict[str, Any]:
    """Build one cvPCA selection from injectable fake relations."""
    return _cv_pca_selection_row(
        key=inputs["key"],
        epoch_intervals_table=_FakeRowsRelation(
            list(inputs["epoch_rows"].values())
        ),
        region_sorted_spikes_group_table=_FakeRelation(inputs["group_row"]),
        movement_firing_rate_table=_FakeRowsRelation(
            list(inputs["movement_results"].values())
        ),
        movement_firing_rate_selection_table=_FakeRowsRelation(
            list(inputs["movement_selections"].values())
        ),
        movement_parameters_table=_FakeRelation(inputs["movement_parameters"]),
        position_table=_FakeRowsRelation(list(inputs["position_rows"].values())),
        trajectory_intervals_table=_FakeRowsRelation(inputs["trajectory_rows"]),
        wtrack_graph_table=_FakeKeyedRelation(
            "configuration_name", inputs["graph_rows"]
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
        session_table=_FakeRelation(
            {
                "subject_id": "L14",
                "session_start_time": datetime(2024, 1, 2),
            }
        ),
        position_inputs_by_condition=inputs["position_inputs"],
        movement_artifacts_by_condition=inputs["movement_artifacts"],
        trajectory_interval_sets_by_condition=inputs["trajectory_intervals"],
        graph_inputs=inputs["graph_inputs"],
    )


def test_cv_pca_selection_freezes_every_source_and_uuid_mutation(
    tmp_path: Path,
) -> None:
    """Every scientific source and random seed participates in UUIDv5."""
    from v1ca1.spyglass import cv_pca

    inputs = _cv_pca_selection_inputs(tmp_path)
    first = _build_cv_pca_selection(inputs)
    assert first == _build_cv_pca_selection(inputs)
    assert first["cv_pca_id"].version == 5
    assert first["position_offset_samples"] == 10
    assert first["cv_pca_output_rule_sha256"] == cv_pca.OUTPUT_RULE_SHA256
    assert set(first["graph_inputs_sha256_by_trajectory"]) == {
        "center_to_left",
        "center_to_right",
    }

    storage_changed = copy.deepcopy(inputs)
    storage_changed["movement_results"]["dark"].update(
        {
            "analysis_file_name": "repacked-dark-movement.nwb",
            "movement_firing_rate_object_id": "repacked-rate-object",
            "movement_intervals_object_id": "repacked-interval-object",
        }
    )
    assert _build_cv_pca_selection(storage_changed)["cv_pca_id"] == first[
        "cv_pca_id"
    ]

    mutations = []
    changed = copy.deepcopy(inputs)
    changed["epoch_rows"]["light"]["condition"] = "BA"
    mutations.append(changed)
    changed = copy.deepcopy(inputs)
    changed["position_inputs"]["light"].d[0, 0] += 0.5
    mutations.append(changed)
    changed = copy.deepcopy(inputs)
    changed["movement_artifacts"]["dark"]["table"].loc[
        0, "movement_firing_rate_hz"
    ] += 0.25
    mutations.append(changed)
    changed = copy.deepcopy(inputs)
    changed["trajectory_intervals"]["light"]["center_to_left"].end[0] += 0.01
    mutations.append(changed)
    changed = copy.deepcopy(inputs)
    changed["graph_inputs"]["center_to_left"]["track_graph_kwargs"][
        "node_positions"
    ][1][0] += 0.25
    changed["graph_inputs"]["center_to_right"]["track_graph_kwargs"][
        "node_positions"
    ][1][0] += 0.25
    mutations.append(changed)
    changed = copy.deepcopy(inputs)
    changed["parameters"]["random_seed"] = 48
    mutations.append(changed)
    assert all(
        _build_cv_pca_selection(changed)["cv_pca_id"] != first["cv_pca_id"]
        for changed in mutations
    )


def test_cv_pca_selection_loads_untrimmed_position_once(
    tmp_path: Path,
) -> None:
    """The table adapter never applies Position's analysis offset itself."""
    inputs = _cv_pca_selection_inputs(tmp_path)

    class PositionTable(_FakeRowsRelation):
        def __init__(self, rows, positions):
            super().__init__(rows)
            self.positions = positions
            self.calls = []

        def load_position(self, key, *, apply_analysis_offset=True):
            self.calls.append((dict(key), apply_analysis_offset))
            condition = "light" if key["epoch"] == "02_r1" else "dark"
            return self.positions[condition]

    position_table = PositionTable(
        list(inputs["position_rows"].values()), inputs["position_inputs"]
    )
    _cv_pca_selection_row(
        key=inputs["key"],
        epoch_intervals_table=_FakeRowsRelation(
            list(inputs["epoch_rows"].values())
        ),
        region_sorted_spikes_group_table=_FakeRelation(inputs["group_row"]),
        movement_firing_rate_table=_FakeRowsRelation(
            list(inputs["movement_results"].values())
        ),
        movement_firing_rate_selection_table=_FakeRowsRelation(
            list(inputs["movement_selections"].values())
        ),
        movement_parameters_table=_FakeRelation(inputs["movement_parameters"]),
        position_table=position_table,
        trajectory_intervals_table=_FakeRowsRelation(inputs["trajectory_rows"]),
        wtrack_graph_table=_FakeKeyedRelation(
            "configuration_name", inputs["graph_rows"]
        ),
        parameters_table=_FakeRelation(inputs["parameters"]),
        session_table=_FakeRelation(
            {
                "subject_id": "L14",
                "session_start_time": datetime(2024, 1, 2),
            }
        ),
        movement_artifacts_by_condition=inputs["movement_artifacts"],
        trajectory_interval_sets_by_condition=inputs["trajectory_intervals"],
        graph_inputs=inputs["graph_inputs"],
    )
    assert len(position_table.calls) == 2
    assert all(apply_offset is False for _key, apply_offset in position_table.calls)


def test_cv_pca_passive_and_standalone_rules_are_identical() -> None:
    """The passive schema and standalone computation share one exact rule."""
    from v1ca1.spyglass import cv_pca

    assert dict(table_specs.CV_PCA_OUTPUT_RULE) == cv_pca.OUTPUT_RULE
    assert provenance_sha256(dict(table_specs.CV_PCA_OUTPUT_RULE)) == (
        cv_pca.OUTPUT_RULE_SHA256
    )
    assert "cv_pca_parameters" in table_specs.TABLE_DEFINITIONS
    assert "cv_pca_selection" in table_specs.TABLE_DEFINITIONS
    assert "cv_pca" in table_specs.TABLE_DEFINITIONS


def test_cv_pca_terminal_status_is_inserted_as_a_result_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected empty-input outcomes remain successful computed rows."""
    result_id = uuid.uuid4()

    def compute(**_kwargs):
        return {
            "analysis_file_name": "cv_pca.nwb",
            **{
                f"{name}_object_id": f"{name}-object"
                for name in (
                    "selected_units",
                    "lap_assignments",
                    "trajectory_qc",
                    "summary",
                    "spectrum",
                    "dataset",
                    "provenance",
                )
            },
            "artifact_schema_version": "1",
            "n_input_units": 2,
            "n_selected_units": 0,
            "analysis_status": "no_movement",
            "selected_units_sha256": "c" * 64,
            "selected_units_table_sha256": "1" * 64,
            "lap_assignments_sha256": "2" * 64,
            "trajectory_qc_sha256": "3" * 64,
            "summary_sha256": "4" * 64,
            "spectrum_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "provenance_sha256": "7" * 64,
            "legacy_artifact_provenance": None,
            "_created_artifact_paths": [],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {
            "cv_pca_id": result_id,
            "nwb_file_name": "session.nwb",
        },
    )
    bundle, _schemas, _unit_selection_params = _fake_bundle(
        runtime_hooks={"cv_pca_compute": compute}
    )
    result = bundle["cv_pca"]
    result().make({"cv_pca_id": result_id})

    inserted, kwargs = result._insert_calls[0]
    assert kwargs == {}
    assert inserted["cv_pca_id"] == result_id
    assert inserted["analysis_status"] == "no_movement"
    assert inserted["n_selected_units"] == 0
    assert inserted["artifact_origin"] == "computed"


def test_cv_pca_load_validates_artifact_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public loader delegates fetch_nwb reconstruction and validation."""
    result_id = uuid.uuid4()
    group_id = uuid.uuid4()
    result_row = {"cv_pca_id": result_id}
    selection = {
        "cv_pca_id": result_id,
        "cv_pca_param_name": "manuscript_v1_seed47",
        "region_sorted_spikes_group_id": group_id,
        "light_epoch": "02_r1",
        "dark_epoch": "08_r4",
    }
    parameters = dict(table_specs.MANUSCRIPT_V1_CV_PCA_PARAMETERS)
    region_row = {
        "region_sorted_spikes_group_id": group_id,
        "region_name": "v1",
    }
    bundle_payload = {"loaded": True}
    load_calls = []

    bundle, _schemas, _unit_selection_params = _fake_bundle()
    result = bundle["cv_pca"]
    selection_table = bundle["cv_pca_selection"]
    parameters_table = bundle["cv_pca_parameters"]
    group_table = bundle["region_sorted_spikes_group"]

    def fetch(table, key):
        if table is result:
            return dict(result_row)
        if table is selection_table:
            return dict(selection)
        if table is parameters_table:
            return dict(parameters)
        if table is group_table:
            return dict(region_row)
        raise AssertionError(f"Unexpected table {table!r}")

    monkeypatch.setitem(_construct_tables.__globals__, "_fetch1_dict", fetch)
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_load_cv_pca_result",
        lambda **kwargs: load_calls.append(kwargs) or bundle_payload,
    )

    loaded = result.load_cv_pca_bundle({"cv_pca_id": result_id})
    assert loaded is bundle_payload
    assert load_calls[0]["result_row"] == result_row
    assert load_calls[0]["result_table"] is result
    assert load_calls[0]["selection_row"] == selection
    assert load_calls[0]["parameters_row"] == parameters
    assert load_calls[0]["region_row"] == region_row


def test_cv_pca_loader_uses_fetch_nwb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cvPCA loader reconstructs exactly the seven selected NWB objects."""
    from v1ca1.spyglass import cv_pca

    result_id = uuid.uuid4()
    object_names = (
        "selected_units",
        "lap_assignments",
        "trajectory_qc",
        "summary",
        "spectrum",
        "dataset",
        "provenance",
    )
    fetched = {name: object() for name in object_names}

    class Relation:
        def __init__(self):
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return [fetched]

    reconstructed = {"reconstructed": True}
    monkeypatch.setattr(
        cv_pca,
        "cv_pca_result_from_nwb_objects",
        lambda **objects: reconstructed
        if objects == fetched
        else pytest.fail("Unexpected CVPCA NWB object set."),
    )
    validation_calls = []
    monkeypatch.setitem(
        _load_cv_pca_result.__globals__,
        "_validate_cv_pca_artifact_link",
        lambda **kwargs: validation_calls.append(kwargs),
    )
    relation = Relation()
    result_row = {"cv_pca_id": result_id}
    loaded = _load_cv_pca_result(
        result_row=result_row,
        result_table=relation,
        selection_row={"selection": True},
        parameters_row={"parameters": True},
        region_row={"region": True},
        animal_name="L14",
        date="20240102",
    )
    assert loaded is reconstructed
    assert relation.keys == [{"cv_pca_id": result_id}]
    assert validation_calls[0]["bundle"] is reconstructed
    assert validation_calls[0]["result_row"] == result_row


def test_cv_pca_make_requires_populate_transaction() -> None:
    """Direct make cannot register a cvPCA NWB outside populate()."""
    bundle, _, _ = _fake_bundle(
        runtime_hooks={
            "cv_pca_compute": lambda **kwargs: pytest.fail(
                "CVPCA computation must not start outside populate()."
            )
        }
    )
    result = bundle["cv_pca"]
    result.connection = SimpleNamespace(in_transaction=False)

    with pytest.raises(RuntimeError, match="must run through populate"):
        result().make({"cv_pca_id": uuid.uuid4()})


def test_cv_pca_registration_hook_uses_exact_recomputed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy registration receives the exact frozen NWB-derived context."""
    from v1ca1.spyglass import cv_pca

    inputs = _cv_pca_selection_inputs(tmp_path)
    selection = _build_cv_pca_selection(inputs)
    loaded_spikes = {
        "ts_group": object(),
        "unit_ids": [
            {"spikesorting_merge_id": "merge-a", "unit_id": "1"},
            {"spikesorting_merge_id": "merge-a", "unit_id": "2"},
        ],
    }
    context = {
        "selection": selection,
        "parameters": inputs["parameters"],
        "group_row": inputs["group_row"],
        "region": "v1",
        "loaded_spikes": loaded_spikes,
        "animal_name": "L14",
        "date": "20240102",
        "movement_results": inputs["movement_results"],
        "movement_selections": inputs["movement_selections"],
        "movement_parameters": {
            condition: inputs["movement_parameters"]
            for condition in ("light", "dark")
        },
        "movement_artifacts": inputs["movement_artifacts"],
        "positions": inputs["position_inputs"],
        "trajectory_intervals": inputs["trajectory_intervals"],
        "graph_inputs": inputs["graph_inputs"],
    }
    monkeypatch.setitem(
        _register_existing_cv_pca_row.__globals__,
        "_load_cv_pca_context",
        lambda **kwargs: context,
    )
    captured = []
    selected_units = pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge-a", "merge-a"],
            "unit_id": ["1", "2"],
        }
    )

    def register(**kwargs):
        captured.append(kwargs)
        return {
            "selected_units": selected_units,
            "n_input_units": 2,
            "n_selected_units": 2,
            "analysis_status": "valid",
            "legacy_artifact_provenance": {
                "verification": "exact_nwb_recomputation"
            },
        }

    monkeypatch.setattr(
        cv_pca, "register_existing_cv_pca_artifact", register
    )
    write_calls = []
    expected_row = {
        "analysis_file_name": "cv_pca.nwb",
        "analysis_status": "valid",
        "selected_units_sha256": selection["selected_units_sha256"],
        "legacy_artifact_provenance": {
            "verification": "exact_nwb_recomputation"
        },
        "_created_artifact_paths": [str(tmp_path / "cv_pca.nwb")],
    }
    monkeypatch.setitem(
        _register_existing_cv_pca_row.__globals__,
        "_write_cv_pca_nwb",
        lambda **kwargs: write_calls.append(kwargs) or dict(expected_row),
    )
    legacy_result = tmp_path / "legacy.nc"
    legacy_summary = tmp_path / "legacy_summary.parquet"
    row = _register_existing_cv_pca_row(
        key=selection,
        legacy_result_path=legacy_result,
        legacy_summary_path=legacy_summary,
        parameters_table=object(),
        epoch_intervals_table=object(),
        region_sorted_spikes_group_table=object(),
        movement_firing_rate_table=object(),
        movement_firing_rate_selection_table=object(),
        movement_parameters_table=object(),
        position_table=object(),
        trajectory_intervals_table=object(),
        wtrack_graph_table=object(),
        session_table=object(),
        artifact_root=tmp_path / "analysis",
        analysis_nwbfile_table="analysis-table",
    )

    call = captured[0]
    assert call["legacy_result_path"] == legacy_result
    assert call["legacy_summary_path"] == legacy_summary
    assert call["overwrite"] is False
    assert call["artifact_root"] is None
    compute_inputs = call["compute_inputs"]
    assert compute_inputs["spikes"] is loaded_spikes["ts_group"]
    assert compute_inputs["light_position"] is inputs["position_inputs"]["light"]
    assert compute_inputs["dark_position"] is inputs["position_inputs"]["dark"]
    assert compute_inputs["position_offset_samples"] == 10
    assert compute_inputs["random_seed"] == 47
    assert compute_inputs["upstream_provenance"] == (
        tables_module._cv_pca_upstream_provenance(selection)
    )
    assert write_calls[0]["nwb_file_name"] == selection["nwb_file_name"]
    assert write_calls[0]["result"]["analysis_status"] == "valid"
    assert write_calls[0]["analysis_nwbfile_table"] == "analysis-table"
    assert row == expected_row


def test_cv_pca_failed_insert_removes_only_new_uuid_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed database insert removes the hook-reported bundle only."""
    result_id = uuid.uuid4()
    artifact_path = tmp_path / "cv_pca.nwb"
    retained = tmp_path / "retained.parquet"
    retained.write_bytes(b"keep")

    def compute(**_kwargs):
        artifact_path.write_bytes(b"new")
        return {
            "analysis_file_name": artifact_path.name,
            **{
                f"{name}_object_id": f"{name}-object"
                for name in (
                    "selected_units",
                    "lap_assignments",
                    "trajectory_qc",
                    "summary",
                    "spectrum",
                    "dataset",
                    "provenance",
                )
            },
            "artifact_schema_version": "1",
            "n_input_units": 2,
            "n_selected_units": 2,
            "analysis_status": "valid",
            "selected_units_sha256": "c" * 64,
            "selected_units_table_sha256": "1" * 64,
            "lap_assignments_sha256": "2" * 64,
            "trajectory_qc_sha256": "3" * 64,
            "summary_sha256": "4" * 64,
            "spectrum_sha256": "5" * 64,
            "dataset_sha256": "6" * 64,
            "provenance_sha256": "7" * 64,
            "legacy_artifact_provenance": None,
            "_created_artifact_paths": [str(artifact_path)],
        }

    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_fetch1_dict",
        lambda table, key: {
            "cv_pca_id": result_id,
            "nwb_file_name": "session.nwb",
        },
    )
    bundle, _schemas, _unit_selection_params = _fake_bundle(
        runtime_hooks={"cv_pca_compute": compute}
    )
    result = bundle["cv_pca"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    result.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        result().make({"cv_pca_id": result_id})
    assert not artifact_path.exists()
    assert retained.read_bytes() == b"keep"


class _FakeNwbfileRegistry:
    """Inject one filepath@raw identity without importing DataJoint."""

    def __init__(self, path: Path, *, contents_hash: uuid.UUID | None = None):
        self.path = Path(path)
        self.contents_hash = contents_hash or uuid.UUID(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        )
        self.registered_size = self.path.stat().st_size

    def get_abs_path(self, _name: str) -> str:
        return str(self.path)

    def get_registered_source_identity(self, _name: str) -> dict[str, Any]:
        return {
            "contents_hash": self.contents_hash,
            "size": self.registered_size,
        }


class _FakeRawExternalStore:
    """Minimal DataJoint external-store relation for fallback-path tests."""

    def __init__(
        self,
        *,
        stage: Path,
        contents_hash: uuid.UUID | None,
        size: int,
        fetch_error: Exception | None = None,
    ):
        self.spec = {"stage": str(stage)}
        self.contents_hash = contents_hash
        self.size = size
        self.fetch_error = fetch_error
        self.restrictions = []

    def __and__(self, restriction):
        self.restrictions.append(dict(restriction))
        return self

    def fetch1(self, *fields):
        assert fields == ("contents_hash", "size")
        if self.fetch_error is not None:
            raise self.fetch_error
        return self.contents_hash, self.size


class _FakeFallbackNwbfileTable:
    """Expose the real heading/connection path used by Nwbfile filepath@raw."""

    def __init__(self, path: Path, external: _FakeRawExternalStore):
        self.path = Path(path)
        attribute = SimpleNamespace(database="common_nwbfile", store="raw")
        self.heading = SimpleNamespace(
            attributes={"nwb_file_abs_path": attribute}
        )
        schema = SimpleNamespace(external={"raw": external})
        self.connection = SimpleNamespace(
            schemas={"common_nwbfile": schema}
        )

    def get_abs_path(self, _name: str) -> str:
        return str(self.path)


def test_registered_nwb_identity_real_fallback_and_failures(
    tmp_path: Path,
) -> None:
    """The dependency-free fake exercises the production external-store path."""
    stage = tmp_path / "raw"
    nested = stage / "nested"
    nested.mkdir(parents=True)
    path = nested / "session.nwb"
    path.write_bytes(b"registered-raw-bytes")
    contents_hash = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    external = _FakeRawExternalStore(
        stage=stage,
        contents_hash=contents_hash,
        size=path.stat().st_size,
    )
    table = _FakeFallbackNwbfileTable(path, external)

    identity = _registered_nwb_source_identity(
        nwbfile_table=table,
        nwb_file_name=path.name,
    )

    assert identity == {
        "nwb_path": path,
        "registered_source_contents_hash": str(contents_hash),
        "registered_source_size_bytes": path.stat().st_size,
    }
    assert external.restrictions == [{"filepath": "nested/session.nwb"}]

    null_hash = _FakeRawExternalStore(
        stage=stage,
        contents_hash=None,
        size=path.stat().st_size,
    )
    with pytest.raises(ValueError, match="registered contents_hash"):
        _registered_nwb_source_identity(
            nwbfile_table=_FakeFallbackNwbfileTable(path, null_hash),
            nwb_file_name=path.name,
        )

    missing_row = _FakeRawExternalStore(
        stage=stage,
        contents_hash=contents_hash,
        size=path.stat().st_size,
        fetch_error=LookupError("missing external row"),
    )
    with pytest.raises(ValueError, match="Could not resolve"):
        _registered_nwb_source_identity(
            nwbfile_table=_FakeFallbackNwbfileTable(path, missing_row),
            nwb_file_name=path.name,
        )

    wrong_stage = _FakeRawExternalStore(
        stage=tmp_path / "different-stage",
        contents_hash=contents_hash,
        size=path.stat().st_size,
    )
    with pytest.raises(ValueError, match="Could not resolve"):
        _registered_nwb_source_identity(
            nwbfile_table=_FakeFallbackNwbfileTable(path, wrong_stage),
            nwb_file_name=path.name,
        )

    wrong_size = _FakeRawExternalStore(
        stage=stage,
        contents_hash=contents_hash,
        size=path.stat().st_size + 1,
    )
    with pytest.raises(ValueError, match="byte size differs"):
        _registered_nwb_source_identity(
            nwbfile_table=_FakeFallbackNwbfileTable(path, wrong_size),
            nwb_file_name=path.name,
        )
