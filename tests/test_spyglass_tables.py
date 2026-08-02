from __future__ import annotations

from datetime import datetime
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from v1ca1.spyglass import table_specs
from v1ca1.spyglass.tables import (
    SOURCE_TABLE_KEYS,
    _analysis_region,
    _attach_registered_unit_identity,
    _construct_tables,
    _filter_registered_table,
    _group_artifact_paths,
    _intervals_to_frame,
    _load_group_spikes,
    _sorting_output_sessions,
    _validate_analysis_schema_prefix,
    _validate_ripple_provenance,
)


class _FakeTable:
    @classmethod
    def insert1(cls, row, **kwargs):
        cls._insert_calls = [*cls.__dict__.get("_insert_calls", []), (dict(row), kwargs)]


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
    fake_dj = SimpleNamespace(Manual=_FakeManual, Computed=_FakeComputed)
    bundle = _construct_tables(
        dj_module=fake_dj,
        session_table=session,
        nwbfile_table=SimpleNamespace(get_abs_path=lambda name: name),
        sorted_spikes_group=SimpleNamespace(Units=object()),
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
    return bundle, schemas


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


def test_constructed_bundle_has_expected_passive_tables() -> None:
    bundle, schemas = _fake_bundle()

    assert tuple(key for key in bundle if key in SOURCE_TABLE_KEYS) == SOURCE_TABLE_KEYS
    assert set(bundle) == {
        *SOURCE_TABLE_KEYS,
        "ripple_modulation_parameters",
        "ripple_modulation_selection",
        "ripple_modulation_computed",
        "analysis_nwbfile",
    }
    assert [schema.activations[0][0] for schema in schemas] == [
        "kyuv1ca1",
        "kyuv1ca1_nwbfile",
    ]
    assert all(
        activation[1]["create_schema"] and activation[1]["create_tables"]
        for schema in schemas
        for activation in schema.activations
    )
    assert not any("_insert_calls" in table.__dict__ for table in bundle.values())
    analysis_nwbfile = bundle["analysis_nwbfile"]
    analysis_nwbfile().register_with_spyglass()
    assert analysis_nwbfile._registry_calls == 1

    assert "epoch: varchar(64)" in bundle["epoch_intervals"].definition
    assert "trajectory_type: varchar(64)" in bundle["trajectory_intervals"].definition
    assert "detector_zscore_threshold = NULL: double" in bundle["ripples"].definition
    assert "position_type: enum('head', 'body')" in bundle["position"].definition
    assert "spatial_unit: enum('cm')" in bundle["position"].definition
    assert "coordinate_unit: enum('cm')" in bundle["wtrack_graph"].definition
    assert "probe_idx: int unsigned" in bundle["spike_sorting_figurl"].definition
    assert "region: enum('v1', 'ca1')" in bundle[
        "ripple_modulation_selection"
    ].definition
    assert "-> SortedSpikesGroup" in bundle[
        "ripple_modulation_selection"
    ].definition
    assert "-> EpochIntervals" in bundle["ripple_modulation_selection"].definition
    assert "summary_path: filepath@analysis" in bundle[
        "ripple_modulation_computed"
    ].definition
    assert "sorting_group_members_sha256: char(64)" in bundle[
        "ripple_modulation_computed"
    ].definition
    assert "legacy_artifact_provenance = NULL: longblob" in bundle[
        "ripple_modulation_computed"
    ].definition
    assert "-> RippleModulationSelection" in bundle[
        "ripple_modulation_computed"
    ].definition


def test_analysis_schema_prefix_is_validated_before_activation() -> None:
    matching = SimpleNamespace(
        config={"custom": {"database.prefix": "kyuv1ca1"}}
    )
    _validate_analysis_schema_prefix(matching, "kyuv1ca1_nwbfile")

    missing = SimpleNamespace(config={})
    with pytest.raises(ValueError, match="database.prefix.*kyuv1ca1"):
        _validate_analysis_schema_prefix(missing, "kyuv1ca1_nwbfile")

    mismatch = SimpleNamespace(
        config={"custom": {"database.prefix": "another_project"}}
    )
    with pytest.raises(ValueError, match="found 'another_project'"):
        _validate_analysis_schema_prefix(mismatch, "kyuv1ca1_nwbfile")

    with pytest.raises(ValueError, match="<prefix>_nwbfile"):
        _validate_analysis_schema_prefix(matching, "kyuv1ca1_results")


def test_parameters_are_manual_scalar_rows_with_explicit_default_insert() -> None:
    bundle, _ = _fake_bundle()
    parameters = bundle["ripple_modulation_parameters"]

    inserted = parameters.insert_default()

    assert inserted == dict(table_specs.DEFAULT_RIPPLE_MODULATION_PARAMETERS)
    assert parameters._insert_calls == [(inserted, {"skip_duplicates": True})]
    assert inserted["minimum_ripple_mean_zscore"] is None
    assert inserted["expected_detector_zscore_threshold"] == pytest.approx(2.0)
    assert inserted["require_speed_gated"] is True
    assert "longblob" not in parameters.definition.lower()

    invalid = {**inserted, "bin_size_s": [0.02]}
    with pytest.raises(TypeError, match="numeric scalar"):
        parameters.insert_parameters(invalid)
    with pytest.raises(ValueError, match="outside the peri-ripple window"):
        parameters.insert_parameters(
            {**inserted, "baseline_window_start_s": -0.6}
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
    bundle, _ = _fake_bundle()
    key = {"nwb_file_name": "L14.nwb", "epoch": "02_r1"}

    assert bundle["epoch_intervals"].load_intervals(key) == "load_interval_set"
    assert (
        bundle["position"].load_position(
            {**key, "position_type": "head"},
            apply_analysis_offset=False,
        )
        == "load_position"
    )
    assert bundle["wtrack_graph"].load_graph(
        {"nwb_file_name": "L14.nwb", "configuration_name": "full_wtrack"}
    ) == "load_wtrack_graph"
    assert calls[1][2]["loader_kwargs"] == {"apply_analysis_offset": False}


def test_failed_computed_insert_removes_only_new_artifacts(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.parquet"
    peri_path = tmp_path / "peri.parquet"

    def compute(**_kwargs):
        summary_path.write_bytes(b"summary")
        peri_path.write_bytes(b"peri")
        return {
            "summary_path": str(summary_path),
            "peri_ripple_firing_rate_path": str(peri_path),
            "n_ripples": 4,
            "n_units": 2,
            "artifact_origin": "computed",
            "_created_artifact_paths": [str(summary_path), str(peri_path)],
        }

    bundle, _ = _fake_bundle(runtime_hooks={"compute": compute})
    computed = bundle["ripple_modulation_computed"]

    def fail_insert(cls, row, **kwargs):
        raise RuntimeError("database insert failed")

    computed.insert1 = classmethod(fail_insert)
    with pytest.raises(RuntimeError, match="database insert failed"):
        computed().make(
            {
                "nwb_file_name": "L14.nwb",
                "epoch": "02_r1",
                "ripple_modulation_param_name": "default",
                "unit_filter_params_name": "all_units",
                "sorted_spikes_group_name": "CA1 shanks",
                "region": "ca1",
            }
        )

    assert not summary_path.exists()
    assert not peri_path.exists()


def test_computed_methods_use_injected_real_call_paths(monkeypatch) -> None:
    monkeypatch.setitem(
        _construct_tables.__globals__,
        "_spyglass_git_commit",
        lambda: "runtime-spyglass-commit",
    )
    calls = []

    def compute(**kwargs):
        calls.append(("compute", kwargs))
        return {
            "summary_path": "/analysis/summary.parquet",
            "peri_ripple_firing_rate_path": "/analysis/peri.parquet",
            "n_ripples": 4,
            "n_units": 2,
            "artifact_origin": "computed",
        }

    def register_existing(**kwargs):
        calls.append(("register", kwargs))
        return {
            "summary_path": "/analysis/keyed_summary.parquet",
            "peri_ripple_firing_rate_path": "/analysis/keyed_peri.parquet",
            "n_ripples": 3,
            "n_units": 1,
            "artifact_origin": "registered_existing",
        }

    bundle, _ = _fake_bundle(
        runtime_hooks={"compute": compute, "register_existing": register_existing}
    )
    computed = bundle["ripple_modulation_computed"]
    key = {
        "nwb_file_name": "L1420240102_.nwb",
        "epoch": "02_r1",
        "ripple_modulation_param_name": "default",
        "unit_filter_params_name": "all_units",
        "sorted_spikes_group_name": "CA1 shanks",
        "region": "ca1",
    }

    computed().make(key)
    registered = computed.register_existing(
        key,
        summary_path="old_summary.parquet",
        peri_ripple_firing_rate_path="old_peri.parquet",
        source_v1ca1_git_commit="source-v1-commit",
    )

    assert calls[0][0] == "compute"
    assert calls[0][1]["key"]["region"] == "ca1"
    assert calls[1][0] == "register"
    assert calls[1][1]["key"]["region"] == "ca1"
    assert computed._insert_calls[0][0]["artifact_origin"] == "computed"
    assert (
        computed._insert_calls[0][0]["runtime_spyglass_git_commit"]
        == "runtime-spyglass-commit"
    )
    assert registered["artifact_origin"] == "registered_existing"
    assert registered["runtime_spyglass_git_commit"] == "runtime-spyglass-commit"
    assert calls[1][1]["source_v1ca1_git_commit"] == "source-v1-commit"

    paths = _group_artifact_paths(
        {
            "directory": Path("/analysis/L14/20240102/ripple_modulation/02_r1/ca1"),
            "summary": Path("/analysis/summary.parquet"),
            "peri_ripple_firing_rate": Path("/analysis/peri.parquet"),
        },
        key,
        merge_ids=["merge-2", "merge-1"],
    )
    assert "group-CA1-shanks-all_units-" in str(paths["summary"])
    assert "-members-" in str(paths["summary"])
    with pytest.raises(ValueError, match="all_units"):
        _group_artifact_paths(
            paths,
            {**key, "unit_filter_params_name": "exclude_noise"},
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


def test_register_existing_filters_all_regions_and_checks_parameters() -> None:
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


def test_legacy_unit_ids_map_to_stable_sorting_and_nwb_identity() -> None:
    table = pd.DataFrame(
        {
            "unit_id": [101, 102],
            "n_ripples": [5, 5],
        }
    )
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

    assert keyed["unit_id"].tolist() == ["merge-a:12", "merge-a:13"]
    assert keyed["spikesorting_merge_id"].tolist() == ["merge-a", "merge-a"]
    assert keyed["nwb_unit_id"].tolist() == ["12", "13"]
    with pytest.raises(ValueError, match="cannot be mapped uniquely"):
        _attach_registered_unit_identity(
            pd.DataFrame({"unit_id": [999]}),
            unit_metadata=metadata,
            artifact_name="summary",
        )


def test_ripple_metadata_is_preserved_and_detector_provenance_is_checked() -> None:
    class Intervals:
        def as_dataframe(self):
            return pd.DataFrame(
                {
                    "start": [1.0],
                    "end": [1.1],
                    "mean_zscore": [3.2],
                }
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


def test_group_loader_region_validates_and_combines_every_member(monkeypatch) -> None:
    class Units:
        def __and__(self, key):
            assert key == {
                "nwb_file_name": "L1420240102_.nwb",
                "unit_filter_params_name": "all_units",
                "sorted_spikes_group_name": "CA1 shanks",
            }
            return self

        def fetch(self, name):
            assert name == "spikesorting_merge_id"
            return ["merge-1", "merge-2"]

    calls = []

    class Parent:
        def __init__(self, nwb_file_name):
            self.nwb_file_name = nwb_file_name

        def fetch(self, name):
            assert name == "nwb_file_name"
            return [self.nwb_file_name]

    class SortingOutput:
        def merge_get_parent(self, key):
            return Parent("L1420240102_.nwb")

    sorting_output = SortingOutput()

    def fetch_spikes(source, key, *, region):
        calls.append((source, key, region))
        merge_id = key["merge_id"]
        unit_ids = [{"spikesorting_merge_id": merge_id, "unit_id": 0}]
        return [np.array([1.0])], unit_ids, [
            {**unit_ids[0], "sorting_unit_id": len(calls)}
        ]

    def build_group(spike_times, unit_ids, *, time_support):
        return {
            "spike_times": spike_times,
            "unit_ids": unit_ids,
            "time_support": time_support,
        }

    spikes_module = importlib.import_module("v1ca1.spyglass.spikes")
    monkeypatch.setattr(
        spikes_module,
        "fetch_spike_times_seconds_with_metadata",
        fetch_spikes,
    )
    monkeypatch.setattr(spikes_module, "build_spike_tsgroup", build_group)
    result = _load_group_spikes(
        sorted_spikes_group=SimpleNamespace(Units=Units()),
        spike_sorting_output=sorting_output,
        key={
            "nwb_file_name": "L1420240102_.nwb",
            "unit_filter_params_name": "all_units",
            "sorted_spikes_group_name": "CA1 shanks",
        },
        region="ca1",
        time_support=(0.0, 2.0),
    )

    assert [call[1] for call in calls] == [
        {"merge_id": "merge-1"},
        {"merge_id": "merge-2"},
    ]
    assert [call[2] for call in calls] == ["ca1", "ca1"]
    assert all(call[0] is sorting_output for call in calls)
    assert result["merge_ids"] == ["merge-1", "merge-2"]
    assert len(result["unit_ids"]) == 2
    assert [unit["sorting_unit_id"] for unit in result["unit_metadata"]] == [1, 2]

    class WrongSessionSortingOutput(SortingOutput):
        def merge_get_parent(self, key):
            return Parent("L1520240102_.nwb")

    with pytest.raises(ValueError, match="not sorting-group session"):
        _load_group_spikes(
            sorted_spikes_group=SimpleNamespace(Units=Units()),
            spike_sorting_output=WrongSessionSortingOutput(),
            key={
                "nwb_file_name": "L1420240102_.nwb",
                "unit_filter_params_name": "all_units",
                "sorted_spikes_group_name": "CA1 shanks",
            },
            region="ca1",
            time_support=(0.0, 2.0),
        )
