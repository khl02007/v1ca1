from __future__ import annotations

"""Tests for database-free augmented-NWB catalog readers."""

import json
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from v1ca1.spyglass.nwb import (
    POSITION_EPOCHS_TABLE_PATH,
    POSITION_INTERFACE_PATH,
    catalog_augmented_nwb,
    load_interval_set,
    load_position,
    load_wtrack_graph,
    read_epoch_intervals,
    read_position_index,
    read_ripples,
    read_spike_sorting_figurls,
    read_trajectory_intervals,
    read_wtrack_graphs,
)


class _Column:
    """Minimal VectorData-like column."""

    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def __getitem__(self, index: int | slice) -> Any:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)


class _UnreadableColumn(_Column):
    """Column that detects accidental catalog-time array reads."""

    def __getitem__(self, index: int | slice) -> Any:
        raise AssertionError("large source array was read while cataloging")


class _Table:
    """Minimal DynamicTable-like object."""

    def __init__(
        self,
        name: str,
        columns: dict[str, list[Any] | _Column],
        *,
        object_id: str,
    ) -> None:
        self.name = name
        self.object_id = object_id
        self.colnames = tuple(columns)
        self._columns = {
            column_name: (
                values if isinstance(values, _Column) else _Column(values)
            )
            for column_name, values in columns.items()
        }
        row_count = len(next(iter(self._columns.values()))) if self._columns else 0
        self.id = _Column(list(range(row_count)))

    def __getitem__(self, column_name: str) -> _Column:
        return self._columns[column_name]

    def __len__(self) -> int:
        return len(self.id)


class _SpatialSeries:
    """Minimal SpatialSeries-like object."""

    def __init__(
        self,
        name: str,
        *,
        data: Any,
        timestamps: Any,
        object_id: str,
    ) -> None:
        self.name = name
        self.data = data
        self.timestamps = timestamps
        self.object_id = object_id
        self.unit = "centimeters"


def _module(**interfaces: Any) -> SimpleNamespace:
    """Return one processing-module fake."""
    return SimpleNamespace(data_interfaces=interfaces)


def _base_nwb() -> SimpleNamespace:
    """Return an augmented NWB fake with all supported source components."""
    epochs = _Table(
        "epochs",
        {
            "start_time": [0.0, 10.0],
            "stop_time": [9.0, 19.0],
            "tags": [["r1"], ["s1"]],
            "epoch_type": ["run", "sleep"],
            "condition": ["stim1", None],
        },
        object_id="epochs-id",
    )
    ephys = _Table(
        "ephys_recording_intervals",
        {
            "start_time": [0.2, 10.2],
            "stop_time": [8.8, 18.8],
            "epoch": ["r1", "s1"],
        },
        object_id="ephys-id",
    )
    trajectories = _Table(
        "trajectory_times",
        {
            "start_time": [1.0, 3.0, 5.0],
            "stop_time": [2.0, 4.0, 6.0],
            "epoch": ["r1", "r1", "r1"],
            "trajectory_type": [
                "center_to_left",
                "center_to_left",
                "left_to_center",
            ],
        },
        object_id="trajectory-id",
    )
    ripples = _Table(
        "ripples",
        {
            "start_time": [1.1, 3.1, 11.1],
            "stop_time": [1.2, 3.2, 11.2],
            "epoch": ["r1", "r1", "s1"],
            "mean_zscore": [2.0, 3.0, 4.0],
        },
        object_id="ripple-id",
    )
    position_epochs = _Table(
        "position_epochs",
        {
            "start_time": [0.5],
            "stop_time": [3.5],
            "epoch": ["r1"],
            "start_index": [0],
            "stop_index_exclusive": [4],
            "sample_count": [4],
            "analysis_start_offset_samples": [1],
            "first_frame": [100],
            "last_frame": [103],
            "video_series_name": ["camera_r1"],
        },
        object_id="position-epochs-id",
    )
    head_series = _SpatialSeries(
        "head_position",
        data=np.arange(8, dtype=float).reshape(4, 2),
        timestamps=np.asarray([0.5, 1.5, 2.5, 3.5]),
        object_id="head-id",
    )
    body_series = _SpatialSeries(
        "body_position",
        data=np.arange(8, 16, dtype=float).reshape(4, 2),
        timestamps=SimpleNamespace(timestamps=head_series.timestamps),
        object_id="body-id",
    )
    position = SimpleNamespace(
        spatial_series={
            "head_position": head_series,
            "body_position": body_series,
        }
    )
    wtrack = _Table(
        "wtrack_linearization",
        {
            "configuration_name": ["center_to_left", "full_w"],
            "node_positions_cm": [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
            ],
            "edges": [[[0, 1]], [[0, 1], [1, 2]]],
            "edge_order": [[[0, 1]], [[0, 1], [1, 2]]],
            "edge_spacing_cm": [[], [5.0]],
            "use_hmm": [False, False],
        },
        object_id="wtrack-id",
    )
    figurls = _Table(
        "spike_sorting_figurls",
        {
            "probe_idx": [0, 0],
            "shank_idx": [0, 1],
            "sorter": ["mountainsort4", "mountainsort4"],
            "figurl_url": ["https://figurl.org/f?a", "https://figurl.org/f?b"],
            "data_uri": ["sha1://a", "sha1://b"],
            "curation_uri": ["gh://a", "gh://b"],
            "source_file": ["figurl/a.txt", "figurl/b.txt"],
        },
        object_id="figurl-id",
    )
    return SimpleNamespace(
        epochs=epochs,
        intervals={
            "ephys_recording_intervals": ephys,
            "trajectory_times": trajectories,
            "ripples": ripples,
            "position_epochs": position_epochs,
        },
        processing={
            "behavior": _module(
                position=position,
                wtrack_linearization=wtrack,
            ),
            "ecephys": _module(spike_sorting_figurls=figurls),
        },
        scratch={
            "ripple_detection_provenance": SimpleNamespace(
                data=json.dumps(
                    {
                        "run_log": {
                            "record": {
                                "parameters": {
                                    "zscore_threshold": 2.0,
                                    "use_speed_gating": True,
                                }
                            }
                        }
                    }
                ),
                object_id="ripple-provenance-id",
            )
        },
    )


class _FakeIntervalSet:
    """Small pynapple IntervalSet test double."""

    def __init__(self, *, start: Any, end: Any, time_units: str) -> None:
        self.start = np.asarray(start)
        self.end = np.asarray(end)
        self.time_units = time_units
        self.metadata: dict[str, Any] = {}

    def set_info(self, **metadata: Any) -> None:
        self.metadata.update(metadata)


class _FakeTsdFrame:
    """Small pynapple TsdFrame test double."""

    def __init__(
        self,
        *,
        t: Any,
        d: Any,
        columns: list[str],
        time_units: str,
    ) -> None:
        self.t = np.asarray(t)
        self.d = np.asarray(d)
        self.columns = columns
        self.time_units = time_units


@pytest.fixture
def fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install small pynapple constructor doubles for explicit loader tests."""
    monkeypatch.setitem(
        sys.modules,
        "pynapple",
        SimpleNamespace(IntervalSet=_FakeIntervalSet, TsdFrame=_FakeTsdFrame),
    )


def test_catalog_augmented_nwb_groups_source_tables() -> None:
    nwbfile = _base_nwb()

    catalog = catalog_augmented_nwb(nwbfile, nwb_file_name="L1420240101_.nwb")

    assert set(catalog) == {
        "epoch_intervals",
        "trajectory_intervals",
        "ripples",
        "position",
        "wtrack_graph",
        "spike_sorting_figurl",
    }
    assert [row["epoch"] for row in catalog["epoch_intervals"]] == ["r1", "s1"]
    assert catalog["epoch_intervals"][0]["condition"] == "AB"
    assert catalog["epoch_intervals"][0]["is_light"] is True
    assert catalog["epoch_intervals"][1]["is_light"] is None
    assert catalog["epoch_intervals"][0]["tags"] == ["r1"]
    assert catalog["epoch_intervals"][0]["start_time"] == pytest.approx(0.2)
    assert catalog["epoch_intervals"][0]["nwb_epoch_start_time"] == pytest.approx(0.0)
    assert catalog["epoch_intervals"][0]["nwb_file_name"] == "L1420240101_.nwb"

    assert [
        (row["trajectory_type"], row["interval_count"])
        for row in catalog["trajectory_intervals"]
    ] == [("center_to_left", 2), ("left_to_center", 1)]
    assert [(row["epoch"], row["ripple_count"]) for row in catalog["ripples"]] == [
        ("r1", 2),
        ("s1", 1),
    ]
    assert catalog["ripples"][0]["detector_zscore_threshold"] == pytest.approx(2.0)
    assert catalog["ripples"][0]["speed_gated"] is True
    assert catalog["ripples"][0]["provenance_object_id"] == "ripple-provenance-id"
    assert [
        (
            row["epoch"],
            row["position_series_name"],
            row["position_role"],
        )
        for row in catalog["position"]
    ] == [
        ("r1", "head_position", "head"),
        ("r1", "body_position", "body"),
    ]
    assert {row["spatial_unit"] for row in catalog["position"]} == {"cm"}
    assert catalog["position"][0]["source_table_path"] == POSITION_EPOCHS_TABLE_PATH
    assert catalog["position"][0]["source_object_id"] == "head-id"
    assert [row["configuration_name"] for row in catalog["wtrack_graph"]] == [
        "center_to_left",
        "full_w",
    ]
    assert {row["coordinate_unit"] for row in catalog["wtrack_graph"]} == {"cm"}
    assert [(row["probe_idx"], row["shank_idx"]) for row in catalog["spike_sorting_figurl"]] == [
        (0, 0),
        (0, 1),
    ]


def test_catalog_readers_do_not_copy_position_or_graph_arrays() -> None:
    nwbfile = _base_nwb()
    position = nwbfile.processing["behavior"].data_interfaces["position"]
    position.spatial_series["head_position"].data = _UnreadableColumn([])
    position.spatial_series["body_position"].data = _UnreadableColumn([])
    wtrack = nwbfile.processing["behavior"].data_interfaces["wtrack_linearization"]
    for column_name in (
        "node_positions_cm",
        "edges",
        "edge_order",
        "edge_spacing_cm",
    ):
        wtrack._columns[column_name] = _UnreadableColumn([None, None])

    assert len(read_position_index(nwbfile)) == 2
    assert len(read_wtrack_graphs(nwbfile)) == 2


def test_interval_loader_filters_group_and_preserves_event_metadata(
    fake_pynapple: None,
) -> None:
    nwbfile = _base_nwb()
    trajectory_row = read_trajectory_intervals(nwbfile)[0]
    ripple_row = read_ripples(nwbfile)[0]

    trajectories = load_interval_set(nwbfile, trajectory_row)
    ripples = load_interval_set(nwbfile, ripple_row)

    np.testing.assert_allclose(trajectories.start, [1.0, 3.0])
    np.testing.assert_allclose(trajectories.end, [2.0, 4.0])
    assert trajectories.metadata["trajectory_type"] == [
        "center_to_left",
        "center_to_left",
    ]
    np.testing.assert_allclose(ripples.start, [1.1, 3.1])
    assert ripples.metadata["mean_zscore"] == [2.0, 3.0]


def test_zero_count_selected_ripple_epoch_is_cataloged_and_loads_empty(
    fake_pynapple: None,
) -> None:
    nwbfile = _base_nwb()
    nwbfile.intervals["ripples"] = _Table(
        "ripples",
        {
            "start_time": [1.1, 3.1],
            "stop_time": [1.2, 3.2],
            "epoch": ["r1", "r1"],
            "mean_zscore": [2.0, 3.0],
        },
        object_id="ripple-id",
    )
    provenance = json.loads(
        nwbfile.scratch["ripple_detection_provenance"].data
    )
    provenance["run_log"]["record"]["outputs"] = {
        "selected_epochs": ["r1", "s1"],
        "epoch_summaries": {
            "r1": {"ripple_count": 2},
            "s1": {"ripple_count": 0},
        },
    }
    nwbfile.scratch["ripple_detection_provenance"].data = json.dumps(
        provenance
    )

    ripple_rows = read_ripples(nwbfile)
    zero_row = next(row for row in ripple_rows if row["epoch"] == "s1")
    empty_intervals = load_interval_set(nwbfile, zero_row)

    assert zero_row["ripple_count"] == 0
    assert empty_intervals.start.size == 0
    assert empty_intervals.end.size == 0
    assert empty_intervals.metadata["epoch"] == []


def test_loaders_reject_catalog_object_ids_from_another_nwb(
    fake_pynapple: None,
) -> None:
    nwbfile = _base_nwb()
    ripple_row = read_ripples(nwbfile)[0]
    position_row = read_position_index(nwbfile)[0]
    graph_row = read_wtrack_graphs(nwbfile)[0]

    ripple_row["source_table_object_id"] = "other-ripples"
    position_row["source_object_id"] = "other-position"
    graph_row["source_object_id"] = "other-graph"

    with pytest.raises(ValueError, match="does not match"):
        load_interval_set(nwbfile, ripple_row)
    with pytest.raises(ValueError, match="does not match"):
        load_position(nwbfile, position_row)
    with pytest.raises(ValueError, match="does not match"):
        load_wtrack_graph(nwbfile, graph_row)


def test_position_loader_applies_offset_by_default(fake_pynapple: None) -> None:
    nwbfile = _base_nwb()
    head_row = read_position_index(nwbfile)[0]

    trimmed = load_position(nwbfile, head_row)
    stored = load_position(nwbfile, head_row, apply_analysis_offset=False)

    np.testing.assert_allclose(trimmed.t, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(
        trimmed.d,
        np.asarray([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]),
    )
    np.testing.assert_allclose(stored.t, [0.5, 1.5, 2.5, 3.5])
    assert stored.d.shape == (4, 2)


def test_position_loader_selects_explicit_future_series_name(
    fake_pynapple: None,
) -> None:
    nwbfile = _base_nwb()
    position = nwbfile.processing["behavior"].data_interfaces["position"]
    position.spatial_series["nose_tip_xy"] = _SpatialSeries(
        "nose_tip_xy",
        data=np.arange(20, 28, dtype=float).reshape(4, 2),
        timestamps=np.asarray([0.5, 1.5, 2.5, 3.5]),
        object_id="nose-tip-id",
    )
    row = read_position_index(nwbfile)[0]
    row.update(
        {
            "position_series_name": "nose_tip_xy",
            "position_role": "nose_tip",
            "source_object_path": f"{POSITION_INTERFACE_PATH}/nose_tip_xy",
            "source_object_id": "nose-tip-id",
        }
    )

    loaded = load_position(nwbfile, row)

    np.testing.assert_allclose(loaded.t, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(
        loaded.d,
        np.asarray([[22.0, 23.0], [24.0, 25.0], [26.0, 27.0]]),
    )


def test_position_catalog_rejects_units_that_do_not_match_graph_cm() -> None:
    nwbfile = _base_nwb()
    position = nwbfile.processing["behavior"].data_interfaces["position"]
    position.spatial_series["body_position"].unit = "meters"

    with pytest.raises(ValueError, match="must use centimeters"):
        read_position_index(nwbfile)


def test_position_loader_rejects_changed_series_path_and_units(
    fake_pynapple: None,
) -> None:
    nwbfile = _base_nwb()
    row = read_position_index(nwbfile)[0]

    with pytest.raises(ValueError, match="source_object_path is not canonical"):
        load_position(
            nwbfile,
            {
                **row,
                "source_object_path": f"{POSITION_INTERFACE_PATH}/body_position",
            },
        )

    position = nwbfile.processing["behavior"].data_interfaces["position"]
    position.spatial_series["head_position"].unit = "meters"
    with pytest.raises(ValueError, match="must use centimeters"):
        load_position(nwbfile, row)


def test_position_loader_rejects_truncated_source_slice(fake_pynapple: None) -> None:
    nwbfile = _base_nwb()
    row = read_position_index(nwbfile)[0]
    row["stop_index_exclusive"] = 6

    with pytest.raises(ValueError, match="bounds do not match"):
        load_position(nwbfile, row, apply_analysis_offset=False)


def test_position_loader_rejects_changed_epoch_times_and_offset(
    fake_pynapple: None,
) -> None:
    nwbfile = _base_nwb()
    row = read_position_index(nwbfile)[0]

    with pytest.raises(ValueError, match="epoch time bounds do not match"):
        load_position(nwbfile, {**row, "start_time": 0.6})
    with pytest.raises(ValueError, match="bounds do not match"):
        load_position(
            nwbfile,
            {**row, "analysis_start_offset_samples": 0},
        )


def test_wtrack_loader_reads_only_selected_configuration() -> None:
    nwbfile = _base_nwb()
    graph_row = read_wtrack_graphs(nwbfile)[1]

    graph = load_wtrack_graph(nwbfile, graph_row)

    assert graph["configuration_name"] == "full_w"
    np.testing.assert_array_equal(graph["edges"], [[0, 1], [1, 2]])
    np.testing.assert_allclose(graph["edge_spacing_cm"], [5.0])
    assert graph["linearization_kwargs"] == {
        "edge_order": [(0, 1), (1, 2)],
        "edge_spacing": [5.0],
        "use_HMM": False,
    }
    np.testing.assert_array_equal(
        graph["track_graph_kwargs"]["node_positions"],
        graph["node_positions_cm"],
    )


def test_missing_optional_components_return_empty_catalogs() -> None:
    nwbfile = _base_nwb()
    nwbfile.intervals = {
        "ephys_recording_intervals": nwbfile.intervals["ephys_recording_intervals"]
    }
    nwbfile.processing = {}

    assert len(read_epoch_intervals(nwbfile)) == 2
    assert read_trajectory_intervals(nwbfile) == []
    assert read_ripples(nwbfile) == []
    assert read_position_index(nwbfile) == []
    assert read_wtrack_graphs(nwbfile) == []
    assert read_spike_sorting_figurls(nwbfile) == []


def test_epoch_reader_does_not_guess_missing_condition() -> None:
    nwbfile = _base_nwb()
    nwbfile.epochs = _Table(
        "epochs",
        {
            "start_time": [0.0, 10.0],
            "stop_time": [9.0, 19.0],
            "tags": [["r1"], ["s1"]],
        },
        object_id="epochs-id",
    )

    rows = read_epoch_intervals(nwbfile)

    assert rows[0]["epoch_type"] is None
    assert rows[0]["condition"] is None


def test_epoch_reader_normalizes_reverse_stimulus_as_light() -> None:
    nwbfile = _base_nwb()
    nwbfile.epochs._columns["condition"] = _Column(["stim3", None])

    run_row, sleep_row = read_epoch_intervals(nwbfile)

    assert run_row["condition"] == "BA"
    assert run_row["is_light"] is True
    assert "project_normalization:stim3->BA" in run_row["condition_source"]
    assert sleep_row["is_light"] is None


def test_epoch_reader_rejects_conditions_outside_declared_vocabulary() -> None:
    nwbfile = _base_nwb()
    nwbfile.epochs._columns["condition"] = _Column(["unexpected", None])

    with pytest.raises(ValueError, match="Unsupported condition"):
        read_epoch_intervals(nwbfile)


def test_epoch_reader_uses_tasks_then_associated_file_condition() -> None:
    nwbfile = _base_nwb()
    nwbfile.epochs = _Table(
        "epochs",
        {
            "start_time": [0.0, 10.0],
            "stop_time": [9.0, 19.0],
            "tags": [["r1"], ["s1"]],
        },
        object_id="epochs-id",
    )
    run_task = _Table(
        "task_0",
        {
            "task_name": ["Run"],
            "task_description": ["Run"],
            "task_epochs": [np.asarray([1])],
            "task_environment": ["W-track"],
        },
        object_id="run-task-id",
    )
    sleep_task = _Table(
        "task_1",
        {
            "task_name": ["Sleep"],
            "task_description": ["Sleep"],
            "task_epochs": [np.asarray([2])],
            "task_environment": ["Sleep box"],
        },
        object_id="sleep-task-id",
    )
    statescript = SimpleNamespace(
        name="Statescript Run 1",
        description="state script",
        task_epochs="1, ",
        object_id="statescript-id",
    )
    stimulus = SimpleNamespace(
        name="stim1",
        description="outside-arm stimulus layout",
        task_epochs="1, ",
        object_id="stimulus-id",
    )
    nwbfile.processing["tasks"] = SimpleNamespace(
        data_interfaces={"task_0": run_task, "task_1": sleep_task}
    )
    nwbfile.processing["associated_files"] = SimpleNamespace(
        data_interfaces={"statescript_run_1": statescript, "stim1": stimulus}
    )

    run_row, sleep_row = read_epoch_intervals(nwbfile)

    assert run_row["epoch_type"] == "run"
    assert run_row["task_name"] == "Run"
    assert run_row["associated_file_names"] == ["Statescript Run 1", "stim1"]
    assert run_row["condition"] == "AB"
    assert run_row["is_light"] is True
    assert run_row["condition_source"].startswith("associated_file:")
    assert "project_normalization:stim1->AB" in run_row["condition_source"]
    assert sleep_row["epoch_type"] == "sleep"
    assert sleep_row["condition"] == "sleep"
    assert sleep_row["is_light"] is None
    assert sleep_row["condition_source"] == "task:/processing/tasks/task_1"
