"""Tests for database-free Figure 1 example-cell payloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from v1ca1.spyglass.offline import figure_1_examples as examples


def _catalog(epoch: str = "02_r1") -> dict[str, list[dict[str, Any]]]:
    """Return a minimal light-run example catalog."""
    trajectories = ("center_to_left", "right_to_center")
    return {
        "epoch_intervals": [
            {
                "epoch": epoch,
                "epoch_type": "run",
                "condition": "AB",
                "is_light": True,
                "start_time": 0.0,
                "stop_time": 5.0,
                "source_table_path": "/intervals/epochs",
            }
        ],
        "position": [
            {
                "epoch": epoch,
                "position_role": "head",
                "position_series_name": "head_position",
                "spatial_unit": "cm",
                "analysis_start_offset_samples": 10,
                "source_table_path": "/intervals/position_epochs",
                "source_object_path": "/processing/behavior/position/head_position",
            }
        ],
        "trajectory_intervals": [
            {
                "epoch": epoch,
                "trajectory_type": trajectory,
                "source_table_path": "/intervals/trajectory_times",
            }
            for trajectory in trajectories
        ],
        "wtrack_graph": [
            {
                "configuration_name": trajectory,
                "coordinate_unit": "cm",
                "source_table_path": "/processing/behavior/wtrack_linearization",
            }
            for trajectory in trajectories
        ],
    }


def _loaded_spikes() -> dict[str, Any]:
    """Return aligned augmented-NWB unit arrays for two V1 cells."""
    merge_id = "a0000000-0000-0000-0000-000000000001"
    identities = [
        {"spikesorting_merge_id": merge_id, "unit_id": 4},
        {"spikesorting_merge_id": merge_id, "unit_id": 12},
    ]
    return {
        "source": "ImportedSpikeSorting",
        "region": "v1",
        "unit_ids": identities,
        "unit_metadata": [
            {**identities[0], "sorting_unit_id": 229, "region": "V1"},
            {**identities[1], "sorting_unit_id": 303, "region": "v1"},
        ],
        "spike_times_s": [
            np.asarray([0.5, 1.0, 2.5, 3.5, 4.5]),
            np.asarray([1.5]),
        ],
    }


def _selection(epoch: str = "02_r1") -> dict[str, Any]:
    """Return selected catalog rows in computation form."""
    catalog = _catalog(epoch)
    return {
        "epoch_row": catalog["epoch_intervals"][0],
        "position_row": catalog["position"][0],
        "trajectory_rows": {
            row["trajectory_type"]: row
            for row in catalog["trajectory_intervals"]
        },
        "graph_rows": {
            row["configuration_name"]: row for row in catalog["wtrack_graph"]
        },
    }


def test_light_epoch_catalog_is_accepted_and_position_contract_is_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        examples,
        "catalog_augmented_nwb",
        lambda *args, **kwargs: _catalog(),
    )

    selected = examples.select_example_epoch_catalog(
        object(),
        nwb_file_name="L1420240611_augmented.nwb",
        epoch="02_r1",
        trajectory_types=("center_to_left", "right_to_center"),
    )

    assert selected["epoch_row"]["is_light"] is True
    assert selected["position_row"]["analysis_start_offset_samples"] == 10

    dark_catalog = _catalog("08_r4")
    dark_catalog["epoch_intervals"][0].update(
        {"condition": "dark", "is_light": False}
    )
    monkeypatch.setattr(
        examples,
        "catalog_augmented_nwb",
        lambda *args, **kwargs: dark_catalog,
    )
    dark = examples.select_example_epoch_catalog(
        object(),
        nwb_file_name="L1420240611_augmented.nwb",
        epoch="08_r4",
        trajectory_types=("center_to_left", "right_to_center"),
    )
    assert dark["epoch_row"]["is_light"] is False

    invalid = _catalog()
    invalid["position"][0]["analysis_start_offset_samples"] = 9
    monkeypatch.setattr(
        examples,
        "catalog_augmented_nwb",
        lambda *args, **kwargs: invalid,
    )
    with pytest.raises(ValueError, match="offset_samples=10"):
        examples.select_example_epoch_catalog(
            object(),
            nwb_file_name="L1420240611_augmented.nwb",
            epoch="02_r1",
            trajectory_types=("center_to_left", "right_to_center"),
        )


def test_resolve_example_unit_uses_region_and_sorting_id() -> None:
    unit = examples.resolve_example_unit(
        _loaded_spikes(),
        region="V1",
        sorting_unit_id=229,
    )

    assert unit["sorting_unit_id"] == 229
    assert unit["unit_id"] == 4
    assert unit["spike_time_unit"] == "s"
    assert unit["spike_time_reference"] == "NWB/ephys timestamps"
    np.testing.assert_allclose(unit["spike_times_s"], [0.5, 1.0, 2.5, 3.5, 4.5])

    with pytest.raises(ValueError, match="found 0"):
        examples.resolve_example_unit(
            _loaded_spikes(),
            region="v1",
            sorting_unit_id=999,
        )


class _Intervals:
    """Small interval object used by the raster helper."""

    def __init__(self, start: tuple[float, ...], end: tuple[float, ...]) -> None:
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)


def test_compute_payload_keeps_all_trial_spikes_but_passes_movement_to_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectories = ("center_to_left", "right_to_center")
    intervals = {
        trajectory: _Intervals((0.0, 3.0), (2.0, 5.0))
        for trajectory in trajectories
    }
    graph_inputs = {trajectory: {"name": trajectory} for trajectory in trajectories}
    movement_intervals = object()
    example_unit = examples.resolve_example_unit(
        _loaded_spikes(),
        region="v1",
        sorting_unit_id=229,
    )
    progression = SimpleNamespace(
        t=np.linspace(0.0, 5.0, 11),
        d=np.linspace(0.0, 1.0, 11),
    )
    monkeypatch.setattr(
        examples,
        "build_task_progression_from_graph",
        lambda **kwargs: (progression, 100.0),
    )
    calls = []

    def _compute_tuning(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        position = np.linspace(0.01, 0.99, 50)
        curve = SimpleNamespace(
            values=np.full((1, 50), 2.0),
            coords={"path_fraction": SimpleNamespace(values=position)},
        )
        return {"tuning_curve": curve, "analysis_status": "valid"}

    monkeypatch.setattr(
        examples.path_specific_place,
        "compute_selected_path_specific_place_tuning_curve",
        _compute_tuning,
    )

    payload = examples.compute_example_payload_from_sources(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        nwb_file_name="L1420240611_augmented.nwb",
        example_unit=example_unit,
        spikes="one-unit-tsgroup",
        position="head-position-cm",
        trajectory_intervals=intervals,
        graph_inputs=graph_inputs,
        movement_intervals=movement_intervals,
        movement_analysis_status="valid",
        selection=_selection(),
    )

    for trajectory in trajectories:
        assert [
            len(values) for values in payload["raster_positions"][trajectory]
        ] == [2, 2]
        assert payload["firing_rates"][trajectory][1].shape == (50,)
    assert all(call["movement_intervals"] is movement_intervals for call in calls)
    assert all(call["trial_subset"] == "all" for call in calls)
    assert all(call["bin_count"] == 50 for call in calls)
    assert all(call["sigma_bins"] == 1.5 for call in calls)
    assert payload["metadata"]["parameters"]["raster_spike_support"] == (
        "complete_trajectory_intervals"
    )


def test_npz_round_trip_is_hashable_run_local_and_never_overwrites(
    tmp_path: Path,
) -> None:
    trajectories = ("center_to_left", "right_to_center")
    metadata = {
        "schema_version": 1,
        "trajectory_types": list(trajectories),
        "parameters": {"position_bin_count": 50},
    }
    payload = {
        "metadata": metadata,
        "raster_positions": {
            trajectory: [np.asarray([0.1, 0.8]), np.asarray([])]
            for trajectory in trajectories
        },
        "firing_rates": {
            trajectory: (
                np.linspace(0.01, 0.99, 50),
                np.linspace(0.0, 3.0, 50),
            )
            for trajectory in trajectories
        },
    }
    path = examples.get_example_payload_path(
        tmp_path,
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        sorting_unit_id=229,
    )

    record = examples.write_example_payload(payload, path, run_dir=tmp_path)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["artifact_sha256"] == expected_hash
    loaded = examples.load_example_payload(path, expected_sha256=expected_hash)
    np.testing.assert_allclose(
        loaded["raster_positions"]["center_to_left"][0],
        [0.1, 0.8],
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        examples.write_example_payload(payload, path, run_dir=tmp_path)
    with pytest.raises(ValueError, match="inside run_dir"):
        examples.write_example_payload(
            payload,
            tmp_path.parent / "outside.npz",
            run_dir=tmp_path,
        )
