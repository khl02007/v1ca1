"""Tests for database-free within-epoch path-specific place decoding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import path_specific_decoding as decoding


DECODING_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
STABLE_UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    {"spikesorting_merge_id": "merge-b", "unit_id": 22},
)


def _intervals_by_trajectory():
    """Return two non-overlapping trials for each trajectory."""
    import pynapple as nap

    return {
        trajectory_type: nap.IntervalSet(
            start=np.asarray([4.0 * index, 4.0 * index + 2.0]),
            end=np.asarray([4.0 * index + 1.0, 4.0 * index + 3.0]),
            time_units="s",
        )
        for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }


def _all_support(intervals):
    """Return one support spanning the synthetic trajectory intervals."""
    import pynapple as nap

    starts = np.concatenate(
        [np.asarray(intervals[name].start, dtype=float) for name in TRAJECTORY_TYPES]
    )
    ends = np.concatenate(
        [np.asarray(intervals[name].end, dtype=float) for name in TRAJECTORY_TYPES]
    )
    order = np.argsort(starts)
    return nap.IntervalSet(start=starts[order], end=ends[order], time_units="s")


def _spikes():
    """Return two units covering the synthetic session."""
    import pynapple as nap

    support = nap.IntervalSet(start=0.0, end=16.0, time_units="s")
    return nap.TsGroup(
        {
            10: nap.Ts(np.arange(0.1, 15.9, 0.4), time_units="s"),
            20: nap.Ts(np.arange(0.2, 15.9, 0.5), time_units="s"),
        },
        time_support=support,
        time_units="s",
    )


def _computed_result(monkeypatch, *, partial: bool = False):
    """Return one deterministic result with the numerical decoder patched."""
    import pynapple as nap

    intervals = _intervals_by_trajectory()
    movement = _all_support(intervals)
    feature = nap.Tsd(
        t=np.arange(0.0, 16.0, 0.1),
        d=np.arange(0.0, 16.0, 0.1),
        time_support=movement,
        time_units="s",
    )
    monkeypatch.setattr(
        decoding,
        "build_concatenated_path_specific_position",
        lambda **kwargs: (feature, np.arange(0.0, 17.0), 4.0),
    )
    calls = []

    def fake_decode_fold(**kwargs):
        calls.append(tuple(kwargs["spikes"].keys()))
        fold = len(calls) - 1
        if partial and fold == 1:
            return (
                None,
                None,
                "no_test_movement",
                "No test movement support.",
                4.0,
                0.0,
            )
        times = np.asarray([0.2 + 2.0 * fold, 0.4 + 2.0 * fold])
        support = nap.IntervalSet(
            start=float(times[0] - 0.1),
            end=float(times[-1] + 0.1),
            time_units="s",
        )
        true = nap.Tsd(
            t=times,
            d=np.asarray([1.0, 2.0]),
            time_support=support,
            time_units="s",
        )
        decoded = nap.Tsd(
            t=times,
            d=np.asarray([1.25, 2.25]),
            time_support=support,
            time_units="s",
        )
        return true, decoded, "valid", "", 4.0, 2.0

    monkeypatch.setattr(decoding, "_decode_fold", fake_decode_fold)
    result = decoding.compute_path_specific_place_decoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        path_specific_place_decoding_id=DECODING_ID,
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        position=object(),
        trajectory_intervals=intervals,
        graph_inputs={name: {} for name in TRAJECTORY_TYPES},
        movement_interval=movement,
        parameter_name="test",
        n_folds=2,
    )
    return result, calls


def test_paths_are_session_first_and_uuid_keyed(tmp_path: Path) -> None:
    paths = decoding.get_path_specific_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        path_specific_place_decoding_id=DECODING_ID,
        artifact_root=tmp_path,
    )
    assert paths["artifact_dir"] == (
        tmp_path
        / "L14"
        / "20240611"
        / decoding.ARTIFACT_DIRNAME
        / "02_r1"
        / "v1"
        / str(DECODING_ID)
    )
    with pytest.raises(ValueError, match="animal_name"):
        decoding.get_path_specific_decoding_artifact_paths(
            animal_name="../L14",
            date="20240611",
            epoch="02_r1",
            region="v1",
            path_specific_place_decoding_id=DECODING_ID,
            artifact_root=tmp_path,
        )


def test_parameter_validation_matches_legacy_defaults() -> None:
    assert decoding.validate_path_specific_decoding_parameters() == {
        "n_folds": 5,
        "sliding_window_size_bins": 4,
        "random_seed": 47,
        "decoding_bin_size_s": 0.02,
        "spatial_bin_size_cm": 4.0,
    }
    assert tuple(decoding.OUTPUT_RULE["trajectory_order"]) == TRAJECTORY_TYPES
    assert decoding.OUTPUT_RULE["unit_policy"] == (
        "all_region_sorted_spikes_group_units"
    )
    with pytest.raises(ValueError, match="at least 2"):
        decoding.validate_path_specific_decoding_parameters(n_folds=1)
    with pytest.raises(TypeError, match="random_seed"):
        decoding.validate_path_specific_decoding_parameters(random_seed=True)


def test_concatenated_coordinate_uses_legacy_order_and_orientation(
    monkeypatch,
) -> None:
    import pynapple as nap
    from v1ca1.spyglass import stability

    intervals = {
        name: nap.IntervalSet(
            start=10.0 * index,
            end=10.0 * index + 2.0,
            time_units="s",
        )
        for index, name in enumerate(TRAJECTORY_TYPES)
    }
    support = _all_support(intervals)

    def fake_progression(*, trajectory_interval, **kwargs):
        start = float(np.asarray(trajectory_interval.start)[0])
        return (
            nap.Tsd(
                t=np.asarray([start, start + 1.0, start + 2.0]),
                d=np.asarray([0.0, 0.5, 1.0]),
                time_support=trajectory_interval,
                time_units="s",
            ),
            100.0,
        )

    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        fake_progression,
    )
    feature, bins, path_length = (
        decoding.build_concatenated_path_specific_position(
            position=object(),
            trajectory_intervals=intervals,
            graph_inputs={
                name: {"configuration_name": name} for name in TRAJECTORY_TYPES
            },
            movement_interval=support,
            spatial_bin_size_cm=25.0,
        )
    )
    assert path_length == 100.0
    assert np.array_equal(bins, np.arange(0.0, 425.0, 25.0))
    expected = {
        "center_to_left": [0.0, 50.0, 100.0],
        "left_to_center": [200.0, 150.0, 100.0],
        "center_to_right": [200.0, 250.0, 300.0],
        "right_to_center": [400.0, 350.0, 300.0],
    }
    for name, values in expected.items():
        assert np.allclose(feature.restrict(intervals[name]).d, values)


def test_compute_uses_all_regional_units_and_tracks_fold_qc(monkeypatch) -> None:
    result, calls = _computed_result(monkeypatch)
    assert calls == [(10, 20), (10, 20)]
    assert result["n_units"] == 2
    assert result["n_folds_valid"] == 2
    assert result["analysis_status"] == "valid"
    assert result["n_decoded_samples"] == 4
    assert result["fold_qc"]["qc_status"].tolist() == ["valid", "valid"]
    assert np.allclose(result["summary"]["mae"], 0.25)
    assert result["selected_units"]["group_unit_id"].tolist() == ["10", "20"]


def test_expected_fold_support_failure_is_isolated(monkeypatch) -> None:
    result, _calls = _computed_result(monkeypatch, partial=True)
    assert result["n_folds_valid"] == 1
    assert result["analysis_status"] == "partial_valid"
    assert result["fold_qc"]["qc_status"].tolist() == [
        "valid",
        "no_test_movement",
    ]
    assert result["n_decoded_samples"] == 2


def test_artifact_round_trip_and_checksum_detection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result, _calls = _computed_result(monkeypatch)
    paths = decoding.get_path_specific_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        path_specific_place_decoding_id=DECODING_ID,
        artifact_root=tmp_path,
    )
    decoding.write_path_specific_decoding_artifact(
        result,
        paths["artifact_dir"],
    )
    loaded = decoding.load_path_specific_decoding_artifact(paths["artifact_dir"])
    assert loaded["analysis_status"] == "valid"
    assert loaded["n_decoded_samples"] == 4
    assert np.allclose(loaded["decoded"].d, result["decoded"].d)
    with pytest.raises(FileExistsError):
        decoding.write_path_specific_decoding_artifact(
            result,
            paths["artifact_dir"],
        )

    summary = pd.read_parquet(paths["decoding_summary_path"])
    summary.loc[0, "mae"] = 999.0
    summary.to_parquet(paths["decoding_summary_path"], index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        decoding.load_path_specific_decoding_artifact(paths["artifact_dir"])


def test_nwb_objects_round_trip_independent_position_timestamps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import pynapple as nap
    import pynwb

    result, _calls = _computed_result(monkeypatch)
    decoded_times = np.asarray(result["decoded"].t, dtype=float)
    decoded_values = np.asarray(result["true"].d, dtype=float)
    true_times = np.sort(
        np.concatenate((decoded_times - 0.05, decoded_times, decoded_times + 0.05))
    )
    true_values = np.interp(true_times, decoded_times, decoded_values)
    result["true"] = nap.Tsd(
        t=true_times,
        d=true_values,
        time_support=result["decoded"].time_support,
        time_units="s",
    )
    decoding.validate_path_specific_decoding_result(result)

    objects = {
        "selected_units": decoding.selected_units_to_dynamic_table(
            result["selected_units"]
        ),
        "fold_qc": decoding.fold_qc_to_dynamic_table(result["fold_qc"]),
        "summary": decoding.decoding_summary_to_dynamic_table(
            result["summary"]
        ),
        "binned_error": decoding.binned_error_to_dynamic_table(
            result["binned_error"]
        ),
        "true_position": decoding.true_position_to_time_series(result["true"]),
        "decoded_position": decoding.decoded_position_to_time_series(
            result["decoded"]
        ),
        "provenance": decoding.decoding_provenance_to_dynamic_table(result),
    }
    support = decoding.decoding_support_to_time_intervals(
        result["true"],
        result["decoded"],
    )
    object_ids = {name: value.object_id for name, value in objects.items()}
    object_ids["support"] = support.object_id
    nwbfile = pynwb.NWBFile(
        session_description="path-specific decoding test",
        identifier="path-specific-decoding-test",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    for nwb_object in objects.values():
        nwbfile.add_scratch(nwb_object)
    nwbfile.add_time_intervals(support)
    nwb_path = tmp_path / "path-specific-decoding.nwb"
    with pynwb.NWBHDF5IO(nwb_path, mode="w") as io:
        io.write(nwbfile)
    assert pynwb.validate(path=nwb_path) == []

    with pynwb.NWBHDF5IO(nwb_path, mode="r", load_namespaces=True) as io:
        stored = io.read()
        reconstructed = decoding.path_specific_place_decoding_result_from_nwb_objects(
            selected_units=stored.objects[object_ids["selected_units"]],
            fold_qc=stored.objects[object_ids["fold_qc"]],
            summary=stored.objects[object_ids["summary"]],
            binned_error=stored.objects[object_ids["binned_error"]],
            true_position=stored.objects[object_ids["true_position"]],
            decoded_position=stored.objects[object_ids["decoded_position"]],
            decoding_support=stored.objects[object_ids["support"]],
            provenance=stored.objects[object_ids["provenance"]],
        )
        assert np.array_equal(reconstructed["true"].t, result["true"].t)
        assert np.array_equal(reconstructed["true"].d, result["true"].d)
        assert np.array_equal(reconstructed["decoded"].t, decoded_times)
        assert np.array_equal(
            reconstructed["decoded"].d,
            np.asarray(result["decoded"].d, dtype=float),
        )
        assert decoding.decoding_support_sha256(
            reconstructed["true"], reconstructed["decoded"]
        ) == decoding.decoding_support_sha256(result["true"], result["decoded"])
        assert decoding.decoding_provenance_sha256(
            reconstructed
        ) == decoding.decoding_provenance_sha256(result)


def test_nwb_dynamic_tables_support_terminal_empty_rows(tmp_path: Path) -> None:
    import pynapple as nap
    import pynwb

    specifications = (
        (
            decoding.selected_units_to_dynamic_table,
            decoding.selected_units_from_dynamic_table,
            decoding.SELECTED_UNIT_COLUMNS,
        ),
        (
            decoding.binned_error_to_dynamic_table,
            decoding.binned_error_from_dynamic_table,
            decoding.BINNED_ERROR_COLUMNS,
        ),
    )
    empty_objects = []
    for to_dynamic_table, from_dynamic_table, columns in specifications:
        nwb_table = to_dynamic_table(pd.DataFrame(columns=columns))
        restored = from_dynamic_table(nwb_table)
        assert restored.empty
        assert tuple(restored.columns) == tuple(columns)
        empty_objects.append(nwb_table)

    support = nap.IntervalSet(start=0.0, end=1.0, time_units="s")
    empty_true = nap.Tsd(
        t=np.asarray([], dtype=float),
        d=np.asarray([], dtype=float),
        time_support=support,
        time_units="s",
    )
    empty_decoded = nap.Tsd(
        t=np.asarray([], dtype=float),
        d=np.asarray([], dtype=float),
        time_support=support,
        time_units="s",
    )
    empty_objects.extend(
        (
            decoding.true_position_to_time_series(empty_true),
            decoding.decoded_position_to_time_series(empty_decoded),
        )
    )
    support_object = decoding.decoding_support_to_time_intervals(
        empty_true,
        empty_decoded,
    )
    nwbfile = pynwb.NWBFile(
        session_description="terminal decoding test",
        identifier="terminal-decoding-test",
        session_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    for nwb_object in empty_objects:
        nwbfile.add_scratch(nwb_object)
    nwbfile.add_time_intervals(support_object)
    path = tmp_path / "terminal-decoding.nwb"
    with pynwb.NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)
    assert pynwb.validate(path=path) == []


def test_legacy_registration_reconstructs_fold_qc_and_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result, _calls = _computed_result(monkeypatch)
    source_paths = decoding.get_path_specific_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        path_specific_place_decoding_id=DECODING_ID,
        artifact_root=tmp_path / "source",
    )
    decoding.write_path_specific_decoding_artifact(
        result,
        source_paths["artifact_dir"],
    )
    registered_id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    destination_paths = decoding.get_path_specific_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        path_specific_place_decoding_id=registered_id,
        artifact_root=tmp_path / "registered",
    )
    intervals = _intervals_by_trajectory()
    registered = decoding.register_existing_path_specific_decoding_artifact(
        source_true_path=source_paths["true_path"],
        source_decoded_path=source_paths["decoded_path"],
        destination_path=destination_paths["artifact_dir"],
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        path_specific_place_decoding_id=registered_id,
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        trajectory_intervals=intervals,
        movement_interval=_all_support(intervals),
        path_length_cm=4.0,
        parameter_name="test",
        n_folds=2,
        source_v1ca1_git_commit="abc123",
    )
    assert registered["artifact_origin"] == "registered_existing"
    assert registered["analysis_status"] == "valid"
    assert registered["n_folds_valid"] == 2
    assert registered["legacy_artifact_provenance"][
        "source_v1ca1_git_commit"
    ] == "abc123"
    loaded = decoding.load_path_specific_decoding_artifact(
        destination_paths["artifact_dir"]
    )
    assert loaded["artifact_origin"] == "registered_existing"
    assert np.allclose(loaded["true"].d, result["true"].d)
