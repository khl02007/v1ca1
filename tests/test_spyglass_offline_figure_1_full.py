"""Tests for database-free full-Figure-1 campaign orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass.offline import figure_1_full as full
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import provenance_sha256


def _write(path: Path, value: bytes = b"artifact") -> Path:
    """Write one small artifact below a temporary test run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _model_record(
    run_dir: Path,
    *,
    analysis_name: str,
    fields: tuple[str, ...],
) -> dict[str, object]:
    """Return one checksum-bearing offline model record."""
    artifacts = {}
    for field in fields:
        path = _write(run_dir / "model" / analysis_name / field)
        artifacts[field] = {
            "relative_path": path.relative_to(run_dir).as_posix(),
            "file_size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    record = {
        "analysis_name": analysis_name,
        "selection": {"selection_id": f"{analysis_name}-selection"},
        "effective_parameters": {"test": 1},
        "artifact_origin": "computed",
        "analysis_status": "valid",
        "n_units_input": 2,
        "n_units_eligible": 1,
        "n_units_valid": 1,
        "artifacts": artifacts,
    }
    record["record_sha256"] = provenance_sha256(record)
    return record


def _complete_session(run_dir: Path) -> dict[str, object]:
    """Create a minimal complete full-Figure-1 session and its files."""
    example = _write(run_dir / "L14" / "20240611" / "example.npz")
    decoding_paths = {
        field: _write(run_dir / "L14" / "20240611" / field)
        for field in full._DECODING_ARTIFACT_FIELDS
    }
    dpp = _model_record(
        run_dir,
        analysis_name="DPPEncoding",
        fields=("dpp_encoding_path",),
    )
    motor = _model_record(
        run_dir,
        analysis_name="MotorEncoding",
        fields=(
            "artifact_manifest_path",
            "selected_units_path",
            "nested_cv_path",
            "full_refit_path",
        ),
    )
    return {
        "schema_version": full.MANIFEST_SCHEMA_VERSION,
        "run_id": "figure1-full",
        "status": "complete",
        "animal_name": "L14",
        "date": "20240611",
        "artifacts": {
            "figure_examples": [
                {
                    "payload_path": example.relative_to(run_dir).as_posix(),
                    "artifact_sha256": file_sha256(example),
                    "artifact_origin": "computed",
                }
            ],
            "dpp_encoding": [dpp],
            "motor_encoding": [motor],
            "path_progression_decoding": [
                {
                    "artifact_origin": "computed",
                    **{
                        field: path.relative_to(run_dir).as_posix()
                        for field, path in decoding_paths.items()
                    },
                    "artifact_sha256": {
                        field: file_sha256(path)
                        for field, path in decoding_paths.items()
                    },
                }
            ],
        },
    }


def test_parent_snapshot_hashes_and_sorts_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_dir = full.get_run_dir("parent", scratch_root=tmp_path)
    manifest_path = _write(parent_dir / "manifest.json", b"parent campaign")
    first = _write(parent_dir / "L15" / "20241121" / "session_manifest.json")
    second = _write(parent_dir / "L14" / "20240611" / "session_manifest.json")
    campaign = {
        "analysis_parameters": {"pipeline": "figure_1_initial_slice"},
        "sessions": [
            {
                "animal_name": "L15",
                "date": "20241121",
                "session_manifest_path": first.relative_to(parent_dir).as_posix(),
            },
            {
                "animal_name": "L14",
                "date": "20240611",
                "session_manifest_path": second.relative_to(parent_dir).as_posix(),
            },
        ],
    }
    monkeypatch.setattr(
        full,
        "load_campaign_manifest",
        lambda *args, **kwargs: campaign,
    )

    snapshot = full.build_parent_snapshot("parent", scratch_root=tmp_path)

    assert snapshot["manifest_sha256"] == file_sha256(manifest_path)
    assert [row["animal_name"] for row in snapshot["sessions"]] == [
        "L14",
        "L15",
    ]
    assert snapshot["sessions"][0]["session_manifest_sha256"] == (
        file_sha256(second)
    )


def test_full_configuration_freezes_computed_approved_parameters() -> None:
    parent = {"run_id": "parent", "manifest_sha256": "a" * 64, "sessions": []}

    configuration = full.build_full_figure_configuration(parent)

    assert configuration["pipeline"] == "figure_1_full"
    assert configuration["artifact_origin"] == "computed"
    assert configuration["diagnostic_figures"] is False
    assert configuration["parent_figure_1d"] == parent
    assert configuration["motor_encoding_parameters"][
        "minimum_movement_firing_rate_hz"
    ] == 0.5
    assert configuration["dpp_encoding_parameters"] == {
        **dict(full.FIGURE_1_DPP_ENCODING_PARAMETERS)
    }
    assert configuration["path_progression_decoding_parameters"][
        "minimum_movement_firing_rate_hz"
    ] == 0.5


def test_full_session_loader_requires_structure_and_checksums(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "figure1-full"
    session = _complete_session(run_dir)
    path = run_dir / "L14" / "20240611" / "session_manifest.json"
    path.write_text(json.dumps(session), encoding="utf-8")

    loaded = full.load_full_figure_session_manifest(path, run_dir=run_dir)
    assert loaded["animal_name"] == "L14"

    changed = dict(session)
    changed["artifacts"] = dict(session["artifacts"])
    changed["artifacts"].pop("motor_encoding")
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="motor_encoding"):
        full.load_full_figure_session_manifest(path, run_dir=run_dir)

    changed = copy.deepcopy(session)
    changed["artifacts"]["dpp_encoding"][0]["analysis_status"] = "changed"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="record checksum"):
        full.load_full_figure_session_manifest(path, run_dir=run_dir)

    path.write_text(json.dumps(session), encoding="utf-8")
    example_path = run_dir / session["artifacts"]["figure_examples"][0][
        "payload_path"
    ]
    example_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        full.load_full_figure_session_manifest(path, run_dir=run_dir)


def test_full_campaign_validates_identity_parent_and_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = full.get_run_dir("figure1-full", scratch_root=tmp_path)
    session = _complete_session(run_dir)
    session_path = run_dir / "L14" / "20240611" / "session_manifest.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    parent = {"run_id": "parent", "manifest_sha256": "a" * 64, "sessions": []}
    campaign = {
        "schema_version": full.MANIFEST_SCHEMA_VERSION,
        "run_id": "figure1-full",
        "analysis_parameters": {
            "pipeline": full.FULL_FIGURE_PIPELINE,
            "parent_figure_1d": parent,
        },
        "sessions": [
            {
                "animal_name": "L14",
                "date": "20240611",
                "session_manifest_path": session_path.relative_to(
                    run_dir
                ).as_posix(),
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    monkeypatch.setattr(
        full,
        "build_parent_snapshot",
        lambda *args, **kwargs: parent,
    )

    _, _, sessions = full.load_full_figure_campaign(
        "figure1-full",
        scratch_root=tmp_path,
    )
    assert len(sessions) == 1

    campaign["sessions"].append(dict(campaign["sessions"][0]))
    (run_dir / "manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        full.load_full_figure_campaign(
            "figure1-full",
            scratch_root=tmp_path,
        )


def test_prepare_full_campaign_reloads_its_own_session_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = full.get_run_dir("figure1-full", scratch_root=tmp_path)
    session = _complete_session(run_dir)
    session_path = run_dir / "L14" / "20240611" / "session_manifest.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    parent = {"run_id": "parent", "manifest_sha256": "a" * 64, "sessions": []}
    campaign = {
        "schema_version": full.MANIFEST_SCHEMA_VERSION,
        "run_id": "figure1-full",
        "analysis_parameters": full.build_full_figure_configuration(parent),
        "source_identity_policy": dict(full.SOURCE_IDENTITY_POLICY),
        "sessions": [
            {
                "animal_name": "L14",
                "date": "20240611",
                "session_manifest_path": session_path.relative_to(
                    run_dir
                ).as_posix(),
            }
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(campaign), encoding="utf-8")
    monkeypatch.setattr(
        full,
        "build_parent_snapshot",
        lambda *args, **kwargs: parent,
    )

    loaded_run_dir, loaded, loaded_parent = full.prepare_full_figure_campaign(
        run_id="figure1-full",
        parent_run_id="parent",
        scratch_root=tmp_path,
    )

    assert loaded_run_dir == run_dir
    assert full.canonical_json(loaded) == full.canonical_json(campaign)
    assert loaded_parent == parent


def test_master_session_delegates_loaded_inputs_without_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pynwb

    run_dir = tmp_path / "runs" / "figure1-full"
    run_dir.mkdir(parents=True)
    parent_dir = tmp_path / "runs" / "parent"
    parent_dir.mkdir(parents=True)
    nwb_path = _write(tmp_path / "L1420240611_augmented.nwb", b"nwb")
    parent_snapshot = {
        "run_id": "parent",
        "sessions": [
            {
                "animal_name": "L14",
                "date": "20240611",
                "session_manifest_path": "L14/20240611/session_manifest.json",
                "session_manifest_sha256": "a" * 64,
            }
        ],
    }
    campaign = {"analysis_parameters": {"pipeline": full.FULL_FIGURE_PIPELINE}}
    parent_session = {
        "nwb_path": str(nwb_path),
        "epochs": ["08_r4"],
        "nwb_fingerprint": {},
    }
    stability_records = {
        name: {"path_specific_place_stability_id": f"stability-{name}"}
        for name in TRAJECTORY_TYPES
    }
    parent_inputs = {
        "movement_record": {"movement_firing_rate_id": "movement-id"},
        "movement": {
            "movement_intervals": object(),
            "table": pd.DataFrame(),
        },
        "stability_records": stability_records,
        "stability_tables": {
            name: pd.DataFrame() for name in TRAJECTORY_TYPES
        },
        "source_identity": {
            "source": "ImportedSpikeSorting",
            "spikesorting_merge_id": "merge-id",
            "selected_units_sha256": "units-sha",
            "offline_region_sorted_spikes_view_id": "regional-id",
        },
    }
    position_rows = {
        role: {
            "position_series_name": f"{role}_position",
            "position_role": role,
            "spatial_unit": "cm",
            "analysis_start_offset_samples": 10,
            "source_table_path": "/intervals/position_epochs",
            "source_object_path": f"/processing/behavior/{role}_position",
        }
        for role in ("head", "body")
    }
    graph_rows = {
        name: {
            "configuration_name": name,
            "coordinate_unit": "cm",
            "source_table_path": "/processing/behavior/wtrack_graph",
            "source_object_path": f"/processing/behavior/{name}",
        }
        for name in (*TRAJECTORY_TYPES, "full_w")
    }
    selection = {
        "epoch_row": {"start_time": 0.0, "stop_time": 10.0},
        "position_rows": position_rows,
        "graph_rows": graph_rows,
    }
    sources = {
        "positions": {"head": object(), "body": object()},
        "trajectory_intervals": {
            name: object() for name in TRAJECTORY_TYPES
        },
        "graph_inputs": {
            name: {} for name in (*TRAJECTORY_TYPES, "full_w")
        },
    }
    loaded_spikes = {
        "spikesorting_merge_id": "merge-id",
        "selected_units_sha256": "units-sha",
        "ts_group": object(),
        "unit_ids": (
            {"spikesorting_merge_id": "merge-id", "unit_id": 1},
        ),
    }
    nwbfile = object()

    class FakeIO:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return nwbfile

    calls: dict[str, dict[str, object]] = {}

    def fake_analysis(name):
        def run(**kwargs):
            calls[name] = kwargs
            if name == "decoding":
                return {"artifact_record": {"analysis_status": "valid"}}
            return {"analysis_name": name, "analysis_status": "valid"}

        return run

    example_spec = {
        "panel": "B",
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "02_r1",
        "region": "v1",
        "sorting_unit_id": 229,
    }
    monkeypatch.setattr(
        full,
        "prepare_full_figure_campaign",
        lambda **kwargs: (run_dir, campaign, parent_snapshot),
    )
    monkeypatch.setattr(
        full,
        "_load_parent_session",
        lambda *args, **kwargs: (parent_dir, parent_session),
    )
    monkeypatch.setattr(
        full,
        "load_parent_analysis_inputs",
        lambda *args, **kwargs: parent_inputs,
    )
    monkeypatch.setattr(pynwb, "NWBHDF5IO", FakeIO)
    monkeypatch.setattr(
        full,
        "validate_nwb_session_identity",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(full, "_require_parent_fingerprint", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        full,
        "select_run_epoch_catalog",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        full,
        "load_run_epoch_catalog_objects",
        lambda *args, **kwargs: sources,
    )
    monkeypatch.setattr(
        full,
        "load_nwb_region_spikes",
        lambda *args, **kwargs: loaded_spikes,
    )
    monkeypatch.setattr(full, "run_offline_dpp_encoding", fake_analysis("dpp"))
    monkeypatch.setattr(full, "run_offline_motor_encoding", fake_analysis("motor"))
    monkeypatch.setattr(full, "run_figure_1_decoding", fake_analysis("decoding"))
    monkeypatch.setattr(full, "FULL_FIGURE_EXAMPLES", (example_spec,))
    monkeypatch.setattr(
        full,
        "compute_nwb_example_payload",
        lambda *args, **kwargs: {
            "metadata": {"persistent_unit_identity": "merge-id:1"}
        },
    )

    def fake_example_write(payload, path, *, run_dir):
        _write(path, b"example")
        return {"payload_path": path, "artifact_sha256": file_sha256(path)}

    monkeypatch.setattr(full, "write_example_payload", fake_example_write)
    appended = []
    monkeypatch.setattr(
        full,
        "append_session_manifest",
        lambda campaign, session, *, run_dir: appended.append(session),
    )

    manifest = full._run_full_figure_session(
        run_id="figure1-full",
        parent_run_id="parent",
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        scratch_root=tmp_path,
    )

    assert calls["dpp"]["position"] is sources["positions"]["head"]
    assert calls["motor"]["primary_position"] is sources["positions"]["head"]
    assert calls["motor"]["orientation_reference_position"] is sources[
        "positions"
    ]["body"]
    assert calls["decoding"]["position"] is sources["positions"]["head"]
    assert calls["motor"]["movement_firing_rate_table"] is parent_inputs[
        "movement"
    ]["table"]
    assert manifest["artifacts"]["figure_examples"][0]["artifact_origin"] == (
        "computed"
    )
    assert manifest["parent_figure_1d"]["run_id"] == "parent"
    assert appended == [manifest]
    assert (
        run_dir / "L14" / "20240611" / full.SESSION_MANIFEST_FILENAME
    ).is_file()


def test_public_runner_removes_only_new_session_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = full.get_run_dir("figure1-full", scratch_root=tmp_path)
    session_dir = full.get_session_dir(
        run_dir,
        animal_name="L14",
        date="20240611",
    )

    def fail(**kwargs):
        session_dir.mkdir(parents=True, exist_ok=True)
        _write(session_dir / "partial.nc")
        raise RuntimeError("fit failed")

    monkeypatch.setattr(full, "_run_full_figure_session", fail)
    with pytest.raises(RuntimeError, match="fit failed"):
        full.run_full_figure_session(
            run_id="figure1-full",
            animal_name="L14",
            date="20240611",
            epoch="08_r4",
            scratch_root=tmp_path,
        )
    assert not session_dir.exists()

    session_dir.mkdir(parents=True)
    marker = _write(session_dir / "existing.txt")
    with pytest.raises(RuntimeError, match="fit failed"):
        full.run_full_figure_session(
            run_id="figure1-full",
            animal_name="L14",
            date="20240611",
            epoch="08_r4",
            scratch_root=tmp_path,
        )
    assert marker.is_file()
