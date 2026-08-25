"""Tests for database-free epoch motor-behavior artifact bundles."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import epoch_motor_behavior as motor_behavior


RESULT_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
EPOCH = "02_r1"


class _Position:
    """Minimal already-offset position object."""

    def __init__(self, timestamps: np.ndarray, values: np.ndarray) -> None:
        self.t = np.asarray(timestamps, dtype=float)
        self.d = np.asarray(values, dtype=float)


def _position_rows() -> tuple[dict[str, object], dict[str, object]]:
    """Return aligned catalog rows with deliberately nonstandard roles."""
    common = {
        "nwb_file_name": "L14_20240611_augmented.nwb",
        "epoch": EPOCH,
        "spatial_unit": "cm",
        "start_index": 100,
        "stop_index_exclusive": 130,
        "sample_count": 30,
        "analysis_start_offset_samples": 10,
        "start_time": 9.0,
        "stop_time": 11.9,
        "first_frame": 500,
        "last_frame": 529,
        "video_series_name": "camera_future",
    }
    return (
        {
            **common,
            "position_series_name": "snout_future",
            "position_role": "translation_anchor",
        },
        {
            **common,
            "position_series_name": "torso_future",
            "position_role": "orientation_anchor",
        },
    )


def _positions(*, all_nonfinite: bool = False) -> tuple[_Position, _Position]:
    """Return twenty loaded samples from thirty stored samples at offset ten."""
    timestamps = 10.0 + np.arange(20, dtype=float) * 0.1
    primary = np.column_stack(
        (np.arange(20, dtype=float) * 10.0, np.zeros(20, dtype=float))
    )
    reference = primary - np.asarray([1.0, 0.0])
    if all_nonfinite:
        primary[:] = np.nan
        reference[:] = np.nan
    else:
        primary[5] = np.nan
    return _Position(timestamps, primary), _Position(timestamps, reference)


def _intervals(*, empty_trajectory: str | None = None) -> dict[str, object]:
    """Return four disjoint natural-direction trial interval sets."""
    import pynapple as nap

    bounds = {
        "center_to_left": (10.0, 10.4),
        "left_to_center": (10.6, 10.9),
        "center_to_right": (11.0, 11.4),
        "right_to_center": (11.5, 11.9),
    }
    result = {}
    for trajectory_type in TRAJECTORY_TYPES:
        if trajectory_type == empty_trajectory:
            result[trajectory_type] = nap.IntervalSet(
                start=np.asarray([], dtype=float),
                end=np.asarray([], dtype=float),
                time_units="s",
            )
        else:
            start, stop = bounds[trajectory_type]
            result[trajectory_type] = nap.IntervalSet(
                start=start,
                end=stop,
                time_units="s",
            )
    return result


def _graphs() -> dict[str, dict[str, object]]:
    """Return four equal-length natural-direction graph configurations."""
    return {
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
        for trajectory_type in TRAJECTORY_TYPES
    }


@pytest.fixture
def fake_linearization(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Return deterministic progression while recording selected graph rows."""
    import pynapple as nap
    from v1ca1.spyglass import stability

    calls: list[str] = []

    def fake_build_task_progression_from_graph(**kwargs):
        trajectory_type = str(kwargs["trajectory_type"])
        calls.append(trajectory_type)
        timestamps = np.asarray(kwargs["position"].t, dtype=float)
        interval = kwargs["trajectory_interval"]
        mask = np.zeros(timestamps.shape, dtype=bool)
        for start, stop in zip(interval.start, interval.end, strict=True):
            mask |= (timestamps >= float(start)) & (timestamps <= float(stop))
        selected = timestamps[mask]
        progression = (
            np.linspace(0.0, 1.0, selected.size)
            if selected.size
            else np.asarray([], dtype=float)
        )
        return (
            nap.Tsd(t=selected, d=progression, time_units="s"),
            100.0,
        )

    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        fake_build_task_progression_from_graph,
    )
    return calls


def _compute_kwargs(
    *,
    empty_trajectory: str | None = None,
    all_nonfinite: bool = False,
) -> dict[str, object]:
    """Return canonical selected-NWB computation arguments."""
    primary, reference = _positions(all_nonfinite=all_nonfinite)
    primary_row, reference_row = _position_rows()
    return {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": EPOCH,
        "epoch_motor_behavior_id": RESULT_ID,
        "primary_position": primary,
        "orientation_reference_position": reference,
        "primary_position_row": primary_row,
        "orientation_reference_position_row": reference_row,
        "trajectory_intervals_by_type": _intervals(
            empty_trajectory=empty_trajectory
        ),
        "graph_inputs_by_configuration": _graphs(),
    }


def test_paths_and_manuscript_parameters_are_canonical(tmp_path: Path) -> None:
    """The bundle is session first, epoch scoped, and UUID keyed."""
    paths = motor_behavior.get_epoch_motor_behavior_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch=EPOCH,
        epoch_motor_behavior_id=RESULT_ID,
        artifact_root=tmp_path,
    )

    artifact_dir = (
        tmp_path
        / "L14"
        / "20240611"
        / "epoch_motor_behavior"
        / EPOCH
        / str(RESULT_ID)
    )
    assert paths["artifact_dir"] == artifact_dir
    assert dict(motor_behavior.MANUSCRIPT_PARAMETERS) == {
        "progression_bin_size_cm": 4.0,
    }
    assert dict(motor_behavior.MANUSCRIPT_MOVEMENT_PARAMETERS) == {
        "movement_param_name": "default",
        "speed_threshold_cm_s": 4.0,
        "speed_smoothing_sigma_s": 0.1,
    }
    assert motor_behavior.OUTPUT_RULE["nonfinite_policy"] == (
        "drop_joint_rows_then_differentiate_across_remaining_gaps"
    )
    assert motor_behavior.OUTPUT_RULE["head_direction_progression_policy"] == (
        "legacy_linear_median_and_quartiles"
    )
    with pytest.raises(ValueError, match="requires the manuscript MovementParameters"):
        motor_behavior.validate_movement_parameter_snapshot(
            {
                "movement_param_name": "changed",
                "speed_threshold_cm_s": 4.0,
                "speed_smoothing_sigma_s": 0.2,
            }
        )


def test_compute_valid_epoch_uses_roles_offset_and_four_graphs(
    fake_linearization: list[str],
) -> None:
    """One nonfinite row is dropped once and all natural paths are summarized."""
    result = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs()
    )

    assert result["analysis_status"] == "valid"
    assert result["n_position_samples_input"] == 20
    assert result["n_finite_position_samples"] == 19
    assert result["n_dropped_nonfinite_samples"] == 1
    assert result["maximum_sample_gap_s"] == pytest.approx(0.2)
    assert result["n_supported_trajectories"] == 4
    assert fake_linearization == list(TRAJECTORY_TYPES)
    assert result["metadata"]["position_offset_samples"] == 10
    assert result["metadata"]["primary_position_source"] == "snout_future"
    assert result["metadata"]["primary_position_role"] == "translation_anchor"
    assert result["metadata"]["orientation_reference_position_source"] == (
        "torso_future"
    )
    assert result["parameters"] == {
        "parameter_name": "manuscript_4cm",
        "parameter_sha256": result["parameters"]["parameter_sha256"],
        "output_rule_sha256": result["parameters"]["output_rule_sha256"],
        "progression_bin_size_cm": 4.0,
    }
    assert result["movement_parameters"]["movement_param_name"] == "default"
    assert result["movement_parameters"]["speed_threshold_cm_s"] == 4.0
    assert result["movement_parameters"]["speed_smoothing_sigma_s"] == 0.1
    assert len(result["movement_parameters"]["movement_parameters_sha256"]) == 64
    distribution = result["distribution_summary"]
    assert distribution.columns.tolist() == list(
        motor_behavior.DISTRIBUTION_COLUMNS
    )
    assert len(distribution) == 6
    assert distribution["sample_count"].nunique() == 1
    assert int(distribution["sample_count"].iloc[0]) == 19
    assert result["progression_summary"].columns.tolist() == list(
        motor_behavior.PROGRESSION_COLUMNS
    )
    qc = result["trajectory_qc"]
    assert qc["trajectory_type"].tolist() == list(TRAJECTORY_TYPES)
    assert qc["trajectory_status"].tolist() == ["valid"] * 4


def test_position_validation_requires_exact_alignment_cm_and_single_offset() -> None:
    """The two selected series must be the exact already-offset NWB pair."""
    kwargs = _compute_kwargs()
    reference = kwargs["orientation_reference_position"]
    kwargs["orientation_reference_position"] = _Position(
        np.asarray(reference.t) + 0.001,
        np.asarray(reference.d),
    )
    with pytest.raises(ValueError, match="timestamps must match exactly"):
        motor_behavior.compute_selected_epoch_motor_behavior(**kwargs)

    kwargs = _compute_kwargs()
    kwargs["primary_position_row"] = {
        **kwargs["primary_position_row"],
        "spatial_unit": "m",
    }
    with pytest.raises(ValueError, match="centimeters"):
        motor_behavior.compute_selected_epoch_motor_behavior(**kwargs)

    kwargs = _compute_kwargs()
    kwargs["primary_position_row"] = {
        **kwargs["primary_position_row"],
        "analysis_start_offset_samples": 11,
    }
    with pytest.raises(ValueError, match="exact sampling metadata"):
        motor_behavior.compute_selected_epoch_motor_behavior(**kwargs)

    kwargs = _compute_kwargs()
    primary = kwargs["primary_position"]
    kwargs["primary_position"] = _Position(primary.t[:-1], primary.d[:-1])
    with pytest.raises(ValueError, match="do not truncate a second time"):
        motor_behavior.compute_selected_epoch_motor_behavior(**kwargs)


def test_compute_terminal_and_partial_statuses(
    fake_linearization: list[str],
) -> None:
    """Terminal rows remain canonical and a missing path is explicitly partial."""
    terminal = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs(all_nonfinite=True)
    )
    assert terminal["analysis_status"] == "no_valid_position"
    assert terminal["n_movement_samples"] == 0
    assert terminal["progression_summary"].empty
    assert terminal["distribution_summary"]["sample_count"].tolist() == [0] * 6
    assert terminal["trajectory_qc"]["trajectory_status"].tolist() == [
        "no_valid_position"
    ] * 4
    assert fake_linearization == []

    partial = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs(empty_trajectory="left_to_center")
    )
    assert partial["analysis_status"] == "partial_valid"
    assert partial["n_supported_trajectories"] == 3
    by_path = partial["trajectory_qc"].set_index("trajectory_type")
    assert by_path.loc["left_to_center", "trajectory_status"] == "no_trials"
    assert set(fake_linearization) == {
        "center_to_left",
        "center_to_right",
        "right_to_center",
    }


def test_write_load_is_atomic_immutable_and_checksum_validated(
    tmp_path: Path,
    fake_linearization: list[str],
) -> None:
    """Canonical Parquets round trip and byte changes invalidate the manifest."""
    del fake_linearization
    result = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs()
    )
    destination = tmp_path / str(RESULT_ID)

    paths = motor_behavior.write_epoch_motor_behavior_artifact(
        result,
        destination,
    )
    loaded = motor_behavior.load_epoch_motor_behavior_artifact(destination)
    assert loaded["analysis_status"] == "valid"
    assert loaded["manifest"]["artifact_key"].tolist() == [
        "distribution_summary",
        "progression_summary",
        "trajectory_qc",
    ]
    assert set(loaded["manifest"]["schema_version"].astype(str)) == {
        motor_behavior.SCHEMA_VERSION
    }
    assert set(loaded["manifest"]["bundle_schema_version"].astype(str)) == {
        motor_behavior.BUNDLE_SCHEMA_VERSION
    }
    pd.testing.assert_frame_equal(
        loaded["distribution_summary"],
        result["distribution_summary"],
        check_dtype=False,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        motor_behavior.write_epoch_motor_behavior_artifact(result, destination)

    distribution = pd.read_parquet(paths["distribution_summary_path"])
    distribution.loc[0, "median"] = 999.0
    distribution.to_parquet(paths["distribution_summary_path"], index=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        motor_behavior.load_epoch_motor_behavior_artifact(destination)


def test_epoch_motor_behavior_nwb_tables_roundtrip_real_hdf5(
    tmp_path: Path,
    fake_linearization: list[str],
) -> None:
    """All three DynamicTables retain their schema, values, and logical hash."""
    del fake_linearization
    from pynwb import NWBHDF5IO, NWBFile

    result = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs()
    )
    specifications = (
        (
            "distribution_summary",
            motor_behavior.distribution_summary_to_dynamic_table,
            motor_behavior.distribution_summary_from_dynamic_table,
            motor_behavior.distribution_summary_sha256,
        ),
        (
            "progression_summary",
            motor_behavior.progression_summary_to_dynamic_table,
            motor_behavior.progression_summary_from_dynamic_table,
            motor_behavior.progression_summary_sha256,
        ),
        (
            "trajectory_qc",
            motor_behavior.trajectory_qc_to_dynamic_table,
            motor_behavior.trajectory_qc_from_dynamic_table,
            motor_behavior.trajectory_qc_sha256,
        ),
    )
    objects = {
        name: to_dynamic_table(result[name])
        for name, to_dynamic_table, _from_dynamic_table, _hasher in specifications
    }
    object_ids = {name: str(obj.object_id) for name, obj in objects.items()}
    assert len(set(object_ids.values())) == 3

    path = tmp_path / "epoch-motor-behavior.nwb"
    nwbfile = NWBFile(
        session_description="EpochMotorBehavior NWB roundtrip test",
        identifier="epoch-motor-behavior-test",
        session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    for obj in objects.values():
        nwbfile.add_scratch(obj)
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)

    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        stored = io.read()
        for name, _to_dynamic_table, from_dynamic_table, hasher in specifications:
            observed = from_dynamic_table(stored.objects[object_ids[name]])
            pd.testing.assert_frame_equal(
                observed,
                result[name],
                check_dtype=False,
                check_categorical=False,
            )
            assert hasher(observed) == hasher(result[name])

    empty = result["progression_summary"].iloc[0:0].copy()
    empty_object = motor_behavior.progression_summary_to_dynamic_table(empty)
    assert motor_behavior.progression_summary_from_dynamic_table(
        empty_object
    ).empty


def test_strict_registration_selects_epoch_and_recomputes_from_nwb(
    tmp_path: Path,
    fake_linearization: list[str],
) -> None:
    """Legacy session Parquets are accepted only after exact epoch recomputation."""
    del fake_linearization
    computed = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs()
    )
    extra_distribution = computed["distribution_summary"].copy()
    extra_distribution["epoch"] = "04_r2"
    extra_progression = computed["progression_summary"].copy()
    extra_progression["epoch"] = "04_r2"
    distribution_path = tmp_path / "motor_distribution_summary.parquet"
    progression_path = tmp_path / "motor_progression_summary.parquet"
    pd.concat(
        [computed["distribution_summary"], extra_distribution],
        ignore_index=True,
    ).to_parquet(distribution_path, index=False)
    pd.concat(
        [computed["progression_summary"], extra_progression],
        ignore_index=True,
    ).to_parquet(progression_path, index=False)
    run_log_path = tmp_path / "run_log.json"
    run_log_path.write_text(
        json.dumps(
            {
                "script": "v1ca1.motor.compare_epoch_motor_behavior",
                "parameters": {
                    "animal_name": "L14",
                    "date": "20240611",
                    "epochs": [EPOCH, "04_r2"],
                    "position_offset": 10,
                    "speed_threshold_cm_s": 4.0,
                    "progression_bin_size_cm": 4.0,
                },
                "git_commit": "abc123",
                "git_dirty": False,
                "timestamp_utc": "2026-08-07T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "registered" / str(RESULT_ID)

    registered = motor_behavior.register_existing_epoch_motor_behavior_artifact(
        source_distribution_path=distribution_path,
        source_progression_path=progression_path,
        source_run_log_path=run_log_path,
        destination_path=destination,
        **_compute_kwargs(),
    )

    assert registered["artifact_origin"] == "registered_existing"
    assert set(registered["distribution_summary"]["epoch"].astype(str)) == {EPOCH}
    assert set(registered["progression_summary"]["epoch"].astype(str)) == {EPOCH}
    provenance = registered["legacy_artifact_provenance"]
    assert provenance["source_epoch"] == EPOCH
    assert provenance["source_v1ca1_git_commit"] == "abc123"
    assert len(registered["_created_artifact_paths"]) == 4
    assert not list(destination.rglob("*.js"))


def test_registration_rejects_legacy_values_that_do_not_match_nwb(
    tmp_path: Path,
    fake_linearization: list[str],
) -> None:
    """Registration cannot bless an artifact whose selected epoch has drifted."""
    del fake_linearization
    computed = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs()
    )
    distribution = computed["distribution_summary"].copy()
    distribution.loc[distribution.index[0], "median"] += 1.0
    distribution_path = tmp_path / "distribution.parquet"
    progression_path = tmp_path / "progression.parquet"
    distribution.to_parquet(distribution_path, index=False)
    computed["progression_summary"].to_parquet(progression_path, index=False)

    with pytest.raises(ValueError, match="do not match exact NWB recomputation"):
        motor_behavior.register_existing_epoch_motor_behavior_artifact(
            source_distribution_path=distribution_path,
            source_progression_path=progression_path,
            destination_path=tmp_path / str(RESULT_ID),
            **_compute_kwargs(),
        )


def test_result_validation_rejects_semantic_tampering(
    fake_linearization: list[str],
) -> None:
    """Checksums cannot substitute for distribution/progression/QC semantics."""
    del fake_linearization
    result = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs()
    )

    distribution_tamper = {
        **result,
        "distribution_summary": result["distribution_summary"].copy(),
    }
    distribution_tamper["distribution_summary"].loc[0, "p10"] = (
        float(distribution_tamper["distribution_summary"].loc[0, "median"])
        + 1.0
    )
    with pytest.raises(ValueError, match="Distribution quantiles"):
        motor_behavior.validate_epoch_motor_behavior_result(
            distribution_tamper
        )

    progression_tamper = {
        **result,
        "progression_summary": result["progression_summary"].copy(),
    }
    progression_tamper["progression_summary"].loc[0, "q25"] = (
        float(progression_tamper["progression_summary"].loc[0, "median"])
        + 1.0
    )
    with pytest.raises(ValueError, match="Progression quantiles"):
        motor_behavior.validate_epoch_motor_behavior_result(
            progression_tamper
        )

    qc_tamper = {**result, "trajectory_qc": result["trajectory_qc"].copy()}
    qc_tamper["trajectory_qc"].loc[0, "occupied_progression_bin_count"] += 1
    with pytest.raises(ValueError, match="progression counts|occupied bins"):
        motor_behavior.validate_epoch_motor_behavior_result(qc_tamper)

    provenance_tamper = {
        **result,
        "legacy_artifact_provenance": {"source": "not-computed"},
    }
    with pytest.raises(ValueError, match="Computed artifacts"):
        motor_behavior.validate_epoch_motor_behavior_result(provenance_tamper)


def test_empty_distribution_and_interval_semantics_are_strict(
    fake_linearization: list[str],
) -> None:
    """Empty statistics stay NaN and trajectory intervals have duration."""
    del fake_linearization
    terminal = motor_behavior.compute_selected_epoch_motor_behavior(
        **_compute_kwargs(all_nonfinite=True)
    )
    terminal_tamper = {
        **terminal,
        "distribution_summary": terminal["distribution_summary"].copy(),
    }
    terminal_tamper["distribution_summary"].loc[0, "mean"] = 0.0
    with pytest.raises(ValueError, match="Empty distributions"):
        motor_behavior.validate_epoch_motor_behavior_result(terminal_tamper)

    kwargs = _compute_kwargs()
    kwargs["trajectory_intervals_by_type"] = {
        **kwargs["trajectory_intervals_by_type"],
        "center_to_left": SimpleNamespace(
            start=np.asarray([10.0]), end=np.asarray([10.0])
        ),
    }
    with pytest.raises(ValueError, match="strictly after"):
        motor_behavior.compute_selected_epoch_motor_behavior(**kwargs)
