from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
import uuid

import pandas as pd
import pytest

from v1ca1.spyglass import table_specs
from v1ca1.spyglass.selection import provenance_sha256
from v1ca1.spyglass.spikes import resolve_sorted_spikes_group_provenance
from v1ca1.spyglass.tables import (
    SOURCE_TABLE_KEYS,
    _analysis_region,
    _attach_registered_unit_identity,
    _construct_tables,
    _filter_registered_table,
    _intervals_to_frame,
    _make_ripple_modulation_row,
    _make_task_progression_stability_row,
    _ripple_modulation_selection_row,
    _stability_selection_row,
    _validate_analysis_schema_prefix,
    _validate_frozen_sorting_snapshot,
    _validate_legacy_stability_schema,
    _validate_ripple_provenance,
)
from v1ca1.spyglass.spikes import _sorting_output_sessions


class _FakeTable:
    @classmethod
    def insert1(cls, row, **kwargs):
        cls._insert_calls = [
            *cls.__dict__.get("_insert_calls", []),
            (dict(row), kwargs),
        ]


class _FakeManual(_FakeTable):
    pass


class _FakeComputed(_FakeTable):
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
    fake_dj = SimpleNamespace(Manual=_FakeManual, Computed=_FakeComputed)
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
        "unit_filter_params_name": "curated_units",
        "sorted_spikes_group_name": "all shanks",
        "region": "ca1",
    }


def _stability_selection_key() -> dict[str, Any]:
    return {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "trajectory_type": "center_to_left",
        "position_series_name": "head_position",
        "configuration_name": "center_to_left",
        "task_progression_stability_param_name": "default",
        "unit_filter_params_name": "curated_units",
        "sorted_spikes_group_name": "all shanks",
        "region": "ca1",
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
        "ripple_modulation_parameters",
        "ripple_modulation_selection",
        "ripple_modulation",
        "task_progression_stability_parameters",
        "task_progression_stability_selection",
        "task_progression_stability",
        "analysis_nwbfile",
    }
    assert [schema.activations[0][0] for schema in schemas] == [
        "kyuv1ca1",
        "kyuv1ca1_nwbfile",
    ]
    assert schemas[0].context["UnitSelectionParams"] is unit_selection_params
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
    for source_name in ("trajectory_intervals", "ripples", "position"):
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
    assert "-> Ripples" in ripple_selection
    assert "-> RippleModulationParameters" in ripple_selection
    assert "-> SortedSpikesGroup" in ripple_selection
    assert "sorting_group_members_sha256: char(64)" in ripple_selection
    assert "unit_filter_params_sha256: char(64)" in ripple_selection
    assert "ripple_modulation_parameters_sha256: char(64)" in ripple_selection

    ripple_result = bundle["ripple_modulation"].definition
    assert "-> RippleModulationSelection" in ripple_result
    assert "analysis_status:" in ripple_result
    assert "selected_units_sha256: char(64)" in ripple_result
    assert "RippleModulationComputed" not in ripple_result

    stability_selection = bundle["task_progression_stability_selection"].definition
    assert "task_progression_stability_id: uuid" in stability_selection
    assert "-> TrajectoryIntervals" in stability_selection
    assert "-> Position" in stability_selection
    assert "-> WTrackGraph" in stability_selection
    assert "-> TaskProgressionStabilityParameters" in stability_selection
    assert "-> SortedSpikesGroup" in stability_selection
    assert (
        "task_progression_stability_parameters_sha256: char(64)"
        in stability_selection
    )

    stability_result = bundle["task_progression_stability"].definition
    assert "-> TaskProgressionStabilitySelection" in stability_result
    assert "stability_path: filepath@analysis" in stability_result
    assert "analysis_status:" in stability_result
    assert "selected_units_sha256: char(64)" in stability_result


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
    ripple_parameters = bundle["ripple_modulation_parameters"]
    stability_parameters = bundle["task_progression_stability_parameters"]

    ripple_row = ripple_parameters.insert_default()
    stability_row = stability_parameters.insert_default()

    assert ripple_row == dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    assert "minimum_ripple_mean_zscore" not in ripple_row
    assert ripple_row["expected_detector_zscore_threshold"] == pytest.approx(2.0)
    assert ripple_row["require_speed_gated"] is True
    assert "longblob" not in ripple_parameters.definition.lower()
    assert ripple_parameters._insert_calls == [
        (ripple_row, {"skip_duplicates": True})
    ]

    assert stability_row == dict(
        table_specs.DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS
    )
    assert stability_row == {
        "task_progression_stability_param_name": "default",
        "speed_threshold_cm_s": 4.0,
        "speed_smoothing_sigma_s": 0.1,
        "place_bin_size_cm": 4.0,
    }
    assert stability_parameters._insert_calls == [
        (stability_row, {"skip_duplicates": True})
    ]

    with pytest.raises(TypeError, match="numeric scalar"):
        ripple_parameters.insert_parameters({**ripple_row, "bin_size_s": [0.02]})
    with pytest.raises(ValueError, match="outside the peri-ripple window"):
        ripple_parameters.insert_parameters(
            {**ripple_row, "baseline_window_start_s": -0.6}
        )
    with pytest.raises(ValueError, match="positive and finite"):
        stability_parameters.insert_parameters(
            {**stability_row, "speed_threshold_cm_s": 0.0}
        )

    assert _analysis_region("ca1") == "ca1"
    with pytest.raises(ValueError, match="canonical lowercase"):
        _analysis_region("CA1")


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


def test_ripple_selection_uuid_is_deterministic_and_freezes_sorting_snapshot() -> None:
    key = _ripple_selection_key()
    parameter_values = dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    parameters = _FakeRelation(parameter_values)
    source = _FakeRelation({})
    unit_parameters = _FakeRelation(
        {"include_labels": ["accepted"], "exclude_labels": ["noise", "mua"]}
    )

    first = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=parameters,
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-b", "merge-a"]),
        unit_selection_params=unit_parameters,
    )
    reordered = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=parameters,
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-a", "merge-b"]),
        unit_selection_params=unit_parameters,
    )
    changed = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=parameters,
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-a", "merge-c"]),
        unit_selection_params=unit_parameters,
    )
    changed_parameter_values = {**parameter_values, "bin_size_s": 0.025}
    changed_parameters = _ripple_modulation_selection_row(
        key=key,
        ripples_table=source,
        epoch_intervals_table=source,
        parameters_table=_FakeRelation(changed_parameter_values),
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-a", "merge-b"]),
        unit_selection_params=unit_parameters,
    )

    assert isinstance(first["ripple_modulation_id"], uuid.UUID)
    assert first["ripple_modulation_id"].version == 5
    assert first["ripple_modulation_id"] == reordered["ripple_modulation_id"]
    assert first["ripple_modulation_id"] != changed["ripple_modulation_id"]
    assert first["ripple_modulation_id"] != changed_parameters["ripple_modulation_id"]
    assert first["sorting_group_members"] == ["merge-a", "merge-b"]
    assert first["unit_filter_include_labels"] == ["accepted"]
    assert first["unit_filter_exclude_labels"] == ["mua", "noise"]
    assert first["ripple_modulation_parameters_sha256"] == provenance_sha256(
        parameter_values
    )
    assert changed_parameters[
        "ripple_modulation_parameters_sha256"
    ] == provenance_sha256(changed_parameter_values)

    current = _sorting_provenance()
    _validate_frozen_sorting_snapshot(first, current)
    with pytest.raises(ValueError, match="changed after selection insertion"):
        _validate_frozen_sorting_snapshot(first, _sorting_provenance(("merge-a",)))


def test_stability_selection_uuid_captures_sources_and_parameter_values() -> None:
    key = _stability_selection_key()
    source = _FakeRelation({})
    epoch = _FakeRelation({"epoch_type": "run"})
    position = _FakeRelation({"spatial_unit": "cm"})
    graph = _FakeRelation({"coordinate_unit": "cm"})
    parameter_values = dict(table_specs.DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS)
    parameters = _FakeRelation(parameter_values)
    unit_parameters = _FakeRelation(
        {"include_labels": ["accepted"], "exclude_labels": ["noise", "mua"]}
    )

    row = _stability_selection_row(
        key=key,
        epoch_intervals_table=epoch,
        trajectory_intervals_table=source,
        position_table=position,
        wtrack_graph_table=graph,
        parameters_table=parameters,
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-b", "merge-a"]),
        unit_selection_params=unit_parameters,
    )
    changed_parameter_values = {**parameter_values, "speed_threshold_cm_s": 5.0}
    changed_parameters = _stability_selection_row(
        key=key,
        epoch_intervals_table=epoch,
        trajectory_intervals_table=source,
        position_table=position,
        wtrack_graph_table=graph,
        parameters_table=_FakeRelation(changed_parameter_values),
        sorted_spikes_group=_FakeSortedSpikesGroup(["merge-a", "merge-b"]),
        unit_selection_params=unit_parameters,
    )

    assert isinstance(row["task_progression_stability_id"], uuid.UUID)
    assert row["task_progression_stability_id"].version == 5
    assert row["trajectory_type"] == "center_to_left"
    assert row["configuration_name"] == "center_to_left"
    assert row["position_series_name"] == "head_position"
    assert row["sorting_group_members"] == ["merge-a", "merge-b"]
    assert row["task_progression_stability_id"] != changed_parameters[
        "task_progression_stability_id"
    ]
    assert row[
        "task_progression_stability_parameters_sha256"
    ] == provenance_sha256(parameter_values)
    assert changed_parameters[
        "task_progression_stability_parameters_sha256"
    ] == provenance_sha256(changed_parameter_values)

    with pytest.raises(ValueError, match="configuration_name"):
        _stability_selection_row(
            key={**key, "configuration_name": "center_to_right"},
            epoch_intervals_table=epoch,
            trajectory_intervals_table=source,
            position_table=position,
            wtrack_graph_table=graph,
            parameters_table=parameters,
            sorted_spikes_group=_FakeSortedSpikesGroup(["merge-a"]),
            unit_selection_params=unit_parameters,
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
            sorted_spikes_group=object(),
            unit_selection_params=object(),
            spike_sorting_output=object(),
            nwbfile_table=object(),
            artifact_root=None,
        )

    stability_parameters = dict(
        table_specs.DEFAULT_TASK_PROGRESSION_STABILITY_PARAMETERS
    )
    stability_selection = {
        **_stability_selection_key(),
        "task_progression_stability_id": uuid.uuid4(),
        "task_progression_stability_parameters_sha256": provenance_sha256(
            stability_parameters
        ),
    }
    with pytest.raises(ValueError, match="parameters changed after selection"):
        _make_task_progression_stability_row(
            key=stability_selection,
            parameters_table=_FakeRelation(
                {**stability_parameters, "speed_threshold_cm_s": 5.0}
            ),
            epoch_intervals_table=object(),
            trajectory_intervals_table=object(),
            position_table=object(),
            wtrack_graph_table=object(),
            session_table=object(),
            sorted_spikes_group=object(),
            unit_selection_params=object(),
            spike_sorting_output=object(),
            nwbfile_table=object(),
            artifact_root=None,
        )


def test_result_make_and_register_hooks_receive_fetched_selection(monkeypatch) -> None:
    ripple_id = uuid.UUID("11111111-1111-5111-8111-111111111111")
    stability_id = uuid.UUID("22222222-2222-5222-8222-222222222222")
    ripple_selection = {"ripple_modulation_id": ripple_id, **_ripple_selection_key()}
    stability_selection = {
        "task_progression_stability_id": stability_id,
        **_stability_selection_key(),
    }
    calls = []

    def ripple_compute(**kwargs):
        calls.append(("ripple_compute", kwargs))
        return {
            "summary_path": "/analysis/summary.parquet",
            "peri_ripple_firing_rate_path": "/analysis/peri.parquet",
            "n_ripples": 4,
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }

    def ripple_register(**kwargs):
        calls.append(("ripple_register", kwargs))
        return {
            "summary_path": "/analysis/keyed-summary.parquet",
            "peri_ripple_firing_rate_path": "/analysis/keyed-peri.parquet",
            "n_ripples": 4,
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "a" * 64,
        }

    def stability_compute(**kwargs):
        calls.append(("stability_compute", kwargs))
        return {
            "stability_path": "/analysis/stability.parquet",
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "b" * 64,
        }

    def stability_register(**kwargs):
        calls.append(("stability_register", kwargs))
        return {
            "stability_path": "/analysis/keyed-stability.parquet",
            "n_units": 2,
            "n_valid_units": 1,
            "analysis_status": "valid",
            "selected_units_sha256": "b" * 64,
        }

    def fetch_selection(table, key):
        if table.__name__ == "RippleModulationSelection":
            assert key == {"ripple_modulation_id": ripple_id}
            return dict(ripple_selection)
        if table.__name__ == "TaskProgressionStabilitySelection":
            assert key == {"task_progression_stability_id": stability_id}
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
            "task_progression_stability_compute": stability_compute,
            "task_progression_stability_register_existing": stability_register,
        }
    )

    ripple = bundle["ripple_modulation"]
    stability = bundle["task_progression_stability"]
    ripple().make({"ripple_modulation_id": ripple_id})
    ripple.register_existing(
        {"ripple_modulation_id": ripple_id},
        summary_path="old-summary.parquet",
        peri_ripple_firing_rate_path="old-peri.parquet",
        source_v1ca1_git_commit="source-v1-commit",
    )
    stability().make({"task_progression_stability_id": stability_id})
    stability.register_existing(
        {"task_progression_stability_id": stability_id},
        stability_path="old-stability.parquet",
        source_v1ca1_git_commit="source-v1-commit",
    )

    assert [name for name, _ in calls] == [
        "ripple_compute",
        "ripple_register",
        "stability_compute",
        "stability_register",
    ]
    assert calls[0][1]["key"] == ripple_selection
    assert calls[1][1]["key"] == ripple_selection
    assert calls[2][1]["key"] == stability_selection
    assert calls[3][1]["key"] == stability_selection
    assert calls[1][1]["overwrite"] is False
    assert calls[3][1]["overwrite"] is False
    assert all(
        kwargs["unit_selection_params"] is unit_selection_params
        for _, kwargs in calls
    )
    assert ripple._insert_calls[0][0]["ripple_modulation_id"] == ripple_id
    assert ripple._insert_calls[1][0]["artifact_origin"] == "registered_existing"
    assert ripple._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }
    assert stability._insert_calls[0][0][
        "task_progression_stability_id"
    ] == stability_id
    assert stability._insert_calls[1][0]["artifact_origin"] == "registered_existing"
    assert stability._insert_calls[1][1] == {
        "skip_duplicates": False,
        "allow_direct_insert": True,
    }
    assert all(
        row["runtime_spyglass_git_commit"] == "runtime-spyglass-commit"
        for result in (ripple, stability)
        for row, _kwargs in result._insert_calls
    )


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
            "task_progression_stability",
            "task_progression_stability_id",
            "task_progression_stability_register_existing",
            {"stability_path": "old-stability.parquet"},
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
            "task_progression_stability",
            "task_progression_stability_id",
            "task_progression_stability_register_existing",
            {"stability_path": "old-stability.parquet"},
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
    ("table_key", "selection_id_name", "hook_name", "artifact_fields"),
    [
        (
            "ripple_modulation",
            "ripple_modulation_id",
            "ripple_modulation_compute",
            ("summary_path", "peri_ripple_firing_rate_path"),
        ),
        (
            "task_progression_stability",
            "task_progression_stability_id",
            "task_progression_stability_compute",
            ("stability_path",),
        ),
    ],
)
def test_failed_result_insert_removes_only_hook_reported_artifacts(
    tmp_path: Path,
    monkeypatch,
    table_key: str,
    selection_id_name: str,
    hook_name: str,
    artifact_fields: tuple[str, ...],
) -> None:
    selection_id = uuid.uuid4()
    created_paths = [
        tmp_path / f"created-{index}.parquet"
        for index in range(len(artifact_fields))
    ]
    retained_path = tmp_path / "preexisting.parquet"
    retained_path.write_bytes(b"keep")

    def compute(**kwargs):
        for path in created_paths:
            path.write_bytes(b"new")
        row = {
            field: str(path) for field, path in zip(artifact_fields, created_paths)
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
            row["n_ripples"] = 3
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
