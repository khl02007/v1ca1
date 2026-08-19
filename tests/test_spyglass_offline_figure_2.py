"""Tests for database-free Figure 2 campaign orchestration."""

from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path

import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.paper_figures import figure_2 as paper_figure_2
from v1ca1.spyglass.offline import figure_2
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import provenance_sha256


_HISTORICAL_PANEL_A_EXAMPLES = (
    {
        "animal_name": "L14",
        "date": "20240611",
        "sorting_unit_id": 34,
        "trajectory_types": ("center_to_left", "right_to_center"),
    },
    {
        "animal_name": "L15",
        "date": "20241121",
        "sorting_unit_id": 473,
        "trajectory_types": ("center_to_right", "left_to_center"),
    },
    {
        "animal_name": "L12",
        "date": "20240421",
        "sorting_unit_id": 37,
        "trajectory_types": ("center_to_right", "left_to_center"),
    },
    {
        "animal_name": "L14",
        "date": "20240611",
        "sorting_unit_id": 30,
        "trajectory_types": ("center_to_left", "right_to_center"),
    },
)


def _write(path: Path, value: bytes = b"artifact") -> Path:
    """Write one small synthetic artifact inside a test campaign."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _seal(record: dict[str, object]) -> dict[str, object]:
    """Add the immutable record checksum required by Figure 2 manifests."""
    record["record_sha256"] = provenance_sha256(record)
    return record


def _flat_record(
    run_dir: Path,
    relative_path: str,
    **metadata: object,
) -> dict[str, object]:
    """Return one checksummed record using the initial-run layout."""
    path = _write(run_dir / relative_path)
    return _seal(
        {
            "artifact_origin": "computed",
            **metadata,
            "artifact_path": path.relative_to(run_dir).as_posix(),
            "artifact_sha256": {"artifact_path": file_sha256(path)},
        }
    )


def _nested_record(
    run_dir: Path,
    relative_path: str,
    **metadata: object,
) -> dict[str, object]:
    """Return one checksummed record using the focused-wrapper layout."""
    path = _write(run_dir / relative_path)
    return _seal(
        {
            "artifact_origin": "computed",
            **metadata,
            "artifacts": {
                "result_path": {
                    "relative_path": path.relative_to(run_dir).as_posix(),
                    "file_size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            },
        }
    )


def _example_record(
    run_dir: Path,
    *,
    sorting_unit_id: int,
    epoch: str,
    trajectory_types: tuple[str, ...],
) -> dict[str, object]:
    """Return one canonical checksummed Panel A example record."""
    path = _write(run_dir / f"examples/{sorting_unit_id}_{epoch}.npz")
    return _seal(
        {
            "artifact_origin": "computed",
            "sorting_unit_id": sorting_unit_id,
            "epoch": epoch,
            "trajectory_types": list(trajectory_types),
            "payload_path": path.relative_to(run_dir).as_posix(),
            "artifact_sha256": file_sha256(path),
        }
    )


def _parent_inputs() -> dict[str, object]:
    """Return the parent identity expected by a synthetic session."""
    fingerprint = {
        "resolved_path": "/stelmo/nwb/raw/L1420240611_augmented.nwb",
        "size_bytes": 10,
        "mtime_ns": 20,
        "nwb_identifier": "nwb-id",
        "units_object_id": "units-id",
        "full_file_sha256": None,
    }
    return {
        "dark_epoch": "08_r4",
        "parent_artifacts": {"parent": "snapshot"},
        "source_identity": {
            "region": "v1",
            "spikesorting_merge_id": "merge-id",
            "selected_units_sha256": "a" * 64,
        },
        "full_session": {
            "nwb_file_name": "L1420240611_augmented.nwb",
            "nwb_path": fingerprint["resolved_path"],
            "nwb_fingerprint": fingerprint,
        },
    }


def _nwb_source_snapshot(
    *,
    epoch: str,
    condition: str,
    is_light: bool,
) -> dict[str, object]:
    """Return one complete reconstructable NWB catalog snapshot."""
    pointer = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "source_table_path": "/source/table",
        "source_table_object_id": "table-id",
        "source_object_path": "/source/object",
        "source_object_id": "object-id",
    }
    selection = {
        "epoch_row": {
            "epoch": epoch,
            "condition": condition,
            "is_light": is_light,
            **pointer,
        },
        "position_rows": {
            "head": {
                "epoch": epoch,
                "position_series_name": "head_position",
                "position_role": "head",
                "source_row_index": 2,
                "start_index": 10,
                "stop_index_exclusive": 20,
                "sample_count": 10,
                "analysis_start_offset_samples": 10,
                "start_time": 1.0,
                "stop_time": 2.0,
                **pointer,
            }
        },
        "trajectory_rows": {
            trajectory_type: {
                "epoch": epoch,
                "trajectory_type": trajectory_type,
                "interval_count": 3,
                **pointer,
            }
            for trajectory_type in TRAJECTORY_TYPES
        },
        "graph_rows": {
            configuration_name: {
                "configuration_name": configuration_name,
                "coordinate_unit": "cm",
                "use_hmm": False,
                "source_row_index": index,
                **pointer,
            }
            for index, configuration_name in enumerate(
                (*TRAJECTORY_TYPES, figure_2.FULL_W_CONFIGURATION_NAME)
            )
        },
    }
    return figure_2._catalog_snapshot(selection)


def _complete_session(
    run_dir: Path,
    *,
    panel_a_examples: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    """Create the exact artifact multiplicities required by Figure 2."""
    if panel_a_examples is None:
        panel_a_examples = figure_2.FIGURE_2_PANEL_A_EXAMPLES
    movement = [
        _flat_record(
            run_dir,
            f"movement/{epoch}.parquet",
            epoch=epoch,
            region="v1",
        )
        for epoch in ("02_r1", "06_r3")
    ]
    tuning = [
        _flat_record(
            run_dir,
            f"tuning/{trajectory_type}_{trial_subset}.nc",
            epoch="02_r1",
            region="v1",
            trajectory_type=trajectory_type,
            trial_subset=trial_subset,
        )
        for trajectory_type in TRAJECTORY_TYPES
        for trial_subset in figure_2.TRIAL_SUBSETS
    ]
    stability = [
        _flat_record(
            run_dir,
            f"stability/{trajectory_type}.parquet",
            epoch="02_r1",
            region="v1",
            trajectory_type=trajectory_type,
        )
        for trajectory_type in TRAJECTORY_TYPES
    ]
    artifacts = {
        "movement_firing_rate": movement,
        "path_specific_place_tuning_curve": tuning,
        "path_specific_place_stability": stability,
        "path_specific_place_tuning_similarity": [
            _nested_record(run_dir, f"similarity/{epoch}.parquet", epoch=epoch)
            for epoch in ("08_r4", "02_r1")
        ],
        "path_progression_decoding": [
            _nested_record(run_dir, "path_progression/manifest.parquet")
        ],
        "path_specific_place_decoding": [
            _nested_record(run_dir, f"place/{epoch}/manifest.parquet", epoch=epoch)
            for epoch in ("08_r4", "02_r1")
        ],
        "dark_light_glm": [_nested_record(run_dir, "dark_light/manifest.parquet")],
        "swap_glm": [_nested_record(run_dir, "swap/manifest.parquet")],
        "figure_examples": [
            _example_record(
                run_dir,
                sorting_unit_id=int(spec["sorting_unit_id"]),
                epoch=epoch,
                trajectory_types=tuple(spec["trajectory_types"]),
            )
            for spec in panel_a_examples
            if spec["animal_name"] == "L14" and spec["date"] == "20240611"
            for epoch in ("08_r4", "02_r1")
        ],
    }
    parent = _parent_inputs()
    return {
        "schema_version": figure_2.MANIFEST_SCHEMA_VERSION,
        "run_id": "figure2-test",
        "status": "complete",
        "animal_name": "L14",
        "date": "20240611",
        "nwb_file_name": parent["full_session"]["nwb_file_name"],
        "nwb_path": parent["full_session"]["nwb_path"],
        "nwb_fingerprint": parent["full_session"]["nwb_fingerprint"],
        "epochs": {"dark": "08_r4", "AB": "02_r1", "BA": "06_r3"},
        "parameters": {
            "panel_a_examples": [
                {
                    **dict(spec),
                    "trajectory_types": list(spec["trajectory_types"]),
                }
                for spec in panel_a_examples
            ]
        },
        "parent_figure_1_full": {"run_id": "figure1-parent"},
        "parent_artifacts": parent["parent_artifacts"],
        "source_identity": [parent["source_identity"]],
        "artifacts": artifacts,
    }


def test_configuration_freezes_approved_figure_2_contract() -> None:
    parent = {"run_id": "figure1-parent", "manifest_sha256": "a" * 64}

    configuration = figure_2.build_figure_2_configuration(parent)

    assert configuration["pipeline"] == "figure_2"
    assert configuration["position_roles"] == ["head"]
    assert configuration["epochs"] == {
        "AB": "02_r1",
        "BA": "06_r3",
        "dark": "from_parent_figure_1_session",
    }
    assert (
        configuration["tuning_similarity_parameters"]["similarity_metric"]
        == "absolute_overlap"
    )
    assert (
        configuration["dark_light_glm_parameters"]["dark_light_glm_param_name"]
        == "legacy_v4_v1"
    )
    assert configuration["dark_light_glm_parameters"]["basis_candidates"] == (
        25,
        40,
        60,
    )
    assert configuration["swap_glm_parameters"]["swap_light_offset"] is False


def test_panel_a_examples_match_the_current_paper_figure() -> None:
    paper_examples = tuple(
        {
            "animal_name": animal_name,
            "date": date,
            "sorting_unit_id": sorting_unit_id,
            "trajectory_types": trajectory_types,
        }
        for animal_name, date, region, sorting_unit_id, trajectory_types in (
            paper_figure_2.FIGURE_2_PANEL_A_EXAMPLES
        )
        if region == figure_2.FIGURE_2_REGION
    )

    assert len(paper_examples) == 8
    assert figure_2.FIGURE_2_PANEL_A_EXAMPLES == paper_examples


def test_catalog_snapshot_retains_reconstructable_nwb_selectors() -> None:
    pointer = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "source_table_path": "/source/table",
        "source_table_object_id": "table-id",
        "source_object_path": "/source/object",
        "source_object_id": "object-id",
    }
    selection = {
        "epoch_row": {
            "epoch": "02_r1",
            "condition": "AB",
            "is_light": True,
            "start_time": 1.0,
            "stop_time": 2.0,
            **pointer,
        },
        "position_rows": {
            "head": {
                "epoch": "02_r1",
                "position_series_name": "head_position",
                "position_role": "head",
                "spatial_unit": "cm",
                "source_row_index": 2,
                "start_index": 10,
                "stop_index_exclusive": 20,
                "sample_count": 10,
                "analysis_start_offset_samples": 10,
                "start_time": 1.0,
                "stop_time": 2.0,
                **pointer,
            }
        },
        "trajectory_rows": {
            trajectory_type: {
                "epoch": "02_r1",
                "trajectory_type": trajectory_type,
                "interval_count": 3,
                **pointer,
            }
            for trajectory_type in TRAJECTORY_TYPES
        },
        "graph_rows": {
            trajectory_type: {
                "configuration_name": trajectory_type,
                "coordinate_unit": "cm",
                "source_row_index": index,
                **pointer,
            }
            for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
        },
    }

    snapshot = figure_2._catalog_snapshot(selection)

    assert snapshot["positions"]["head"]["source_row_index"] == 2
    assert snapshot["positions"]["head"]["start_index"] == 10
    assert snapshot["trajectory_intervals"]["center_to_left"]["interval_count"] == 3
    assert snapshot["wtrack_graphs"]["center_to_left"]["source_row_index"] == 0
    assert (
        snapshot["wtrack_graphs"]["center_to_left"]["source_object_id"] == "object-id"
    )


def test_nwb_source_validator_enforces_epoch_conditions() -> None:
    epochs = {"dark": "08_r4", "AB": "02_r1", "BA": "06_r3"}
    session = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "nwb_sources": {
            "08_r4": _nwb_source_snapshot(
                epoch="08_r4",
                condition="dark",
                is_light=False,
            ),
            "02_r1": _nwb_source_snapshot(
                epoch="02_r1",
                condition="AB",
                is_light=True,
            ),
            "06_r3": _nwb_source_snapshot(
                epoch="06_r3",
                condition="BA",
                is_light=True,
            ),
        },
    }

    figure_2._validate_nwb_source_snapshots(session, epochs)

    changed = copy.deepcopy(session)
    changed["nwb_sources"]["06_r3"]["epoch_interval"]["condition"] = "AB"
    with pytest.raises(ValueError, match="condition provenance"):
        figure_2._validate_nwb_source_snapshots(changed, epochs)


def test_analysis_role_validator_enforces_forward_swap_linkage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure2-test"
    progression_path = _write(run_dir / "progression/manifest.parquet")
    dark_place_dir = run_dir / "place" / "08_r4"
    light_place_dir = run_dir / "place" / "02_r1"
    swap_dir = run_dir / "swap"
    for directory in (dark_place_dir, light_place_dir, swap_dir):
        directory.mkdir(parents=True)
    session = {
        "animal_name": "L14",
        "date": "20240611",
        "nwb_file_name": "L1420240611_augmented.nwb",
        "epochs": {"dark": "08_r4", "AB": "02_r1", "BA": "06_r3"},
    }
    artifacts = {
        "path_specific_place_tuning_similarity": [
            {
                "analysis_name": "PathSpecificPlaceTuningSimilarity",
                "epoch": epoch,
            }
            for epoch in ("08_r4", "02_r1")
        ],
        "path_progression_decoding": [
            {
                "epoch": "02_r1",
                "artifact_manifest_path": progression_path.relative_to(
                    run_dir
                ).as_posix(),
            }
        ],
        "path_specific_place_decoding": [
            {
                "analysis_name": "PathSpecificPlaceDecoding",
                "epoch": epoch,
                "artifact_dir": directory.relative_to(run_dir).as_posix(),
            }
            for epoch, directory in (
                ("08_r4", dark_place_dir),
                ("02_r1", light_place_dir),
            )
        ],
        "dark_light_glm": [
            {
                "analysis_name": "DarkLightGLM",
                "dark_light_glm_id": "dark-light-id",
                "dark_epoch": "08_r4",
                "light_epoch": "02_r1",
                "region": "v1",
                "nwb_file_name": "L1420240611_augmented.nwb",
            }
        ],
        "swap_glm": [
            {
                "analysis_name": "SwapGLM",
                "dark_light_glm_id": "dark-light-id",
                "dark_epoch": "08_r4",
                "light_train_epoch": "02_r1",
                "light_test_epoch": "06_r3",
                "region": "v1",
                "nwb_file_name": "L1420240611_augmented.nwb",
                "artifact_dir": swap_dir.relative_to(run_dir).as_posix(),
            }
        ],
    }
    monkeypatch.setattr(
        figure_2.path_progression_decoding,
        "load_decoding_artifact_bundle",
        lambda _path: {
            "metadata": {
                "animal_name": "L14",
                "date": "20240611",
                "epoch": "02_r1",
                "cohort_epoch": "02_r1",
                "region": "v1",
            }
        },
    )
    monkeypatch.setattr(
        figure_2.path_specific_decoding,
        "load_path_specific_decoding_artifact",
        lambda path: {
            "metadata": {
                "animal_name": "L14",
                "date": "20240611",
                "epoch": path.name,
                "region": "v1",
            }
        },
    )
    monkeypatch.setattr(
        figure_2.swap_glm,
        "load_swap_glm_artifact",
        lambda _path: {
            "metadata": {
                "animal_name": "L14",
                "date": "20240611",
                "dark_epoch": "08_r4",
                "light_train_epoch": "02_r1",
                "light_test_epoch": "06_r3",
                "region": "v1",
            }
        },
    )

    figure_2._validate_analysis_roles(
        session,
        artifacts,
        run_dir=run_dir,
    )

    changed = copy.deepcopy(artifacts)
    changed["swap_glm"][0]["dark_light_glm_id"] = "other-id"
    with pytest.raises(ValueError, match="does not reference"):
        figure_2._validate_analysis_roles(
            session,
            changed,
            run_dir=run_dir,
        )


def test_session_loader_checks_artifacts_and_parent_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure2-test"
    session = _complete_session(run_dir)
    path = run_dir / "L14" / "20240611" / "session_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(session), encoding="utf-8")
    parent_snapshot = session["parent_figure_1_full"]
    parent_inputs = _parent_inputs()
    monkeypatch.setattr(
        figure_2,
        "build_full_figure_1_parent_snapshot",
        lambda *args, **kwargs: parent_snapshot,
    )
    monkeypatch.setattr(
        figure_2,
        "_load_parent_inputs",
        lambda *args, **kwargs: parent_inputs,
    )
    monkeypatch.setattr(
        figure_2,
        "_validate_nwb_source_snapshots",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        figure_2,
        "_validate_analysis_roles",
        lambda *args, **kwargs: None,
    )

    loaded = figure_2.load_figure_2_session_manifest(
        path,
        run_dir=run_dir,
        scratch_root=tmp_path,
    )
    assert loaded["animal_name"] == "L14"

    changed = copy.deepcopy(session)
    changed["artifacts"]["path_specific_place_tuning_similarity"][0].pop(
        "record_sha256"
    )
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="missing its checksum"):
        figure_2.load_figure_2_session_manifest(
            path,
            run_dir=run_dir,
            scratch_root=tmp_path,
        )

    changed = copy.deepcopy(session)
    changed["nwb_fingerprint"]["mtime_ns"] += 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="nwb_fingerprint"):
        figure_2.load_figure_2_session_manifest(
            path,
            run_dir=run_dir,
            scratch_root=tmp_path,
        )

    path.write_text(json.dumps(session), encoding="utf-8")
    artifact_path = (
        run_dir
        / session["artifacts"]["swap_glm"][0]["artifacts"]["result_path"][
            "relative_path"
        ]
    )
    artifact_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        figure_2.load_figure_2_session_manifest(
            path,
            run_dir=run_dir,
            scratch_root=tmp_path,
        )


def test_session_loader_accepts_historical_frozen_panel_a_examples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure2-historical"
    session = _complete_session(
        run_dir,
        panel_a_examples=_HISTORICAL_PANEL_A_EXAMPLES,
    )
    path = run_dir / "L14" / "20240611" / "session_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(session), encoding="utf-8")
    parent_snapshot = session["parent_figure_1_full"]
    monkeypatch.setattr(
        figure_2,
        "build_full_figure_1_parent_snapshot",
        lambda *args, **kwargs: parent_snapshot,
    )
    monkeypatch.setattr(
        figure_2,
        "_load_parent_inputs",
        lambda *args, **kwargs: _parent_inputs(),
    )
    monkeypatch.setattr(
        figure_2,
        "_validate_nwb_source_snapshots",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        figure_2,
        "_validate_analysis_roles",
        lambda *args, **kwargs: None,
    )

    loaded = figure_2.load_figure_2_session_manifest(
        path,
        run_dir=run_dir,
        scratch_root=tmp_path,
    )

    assert len(loaded["parameters"]["panel_a_examples"]) == 4
    assert len(loaded["artifacts"]["figure_examples"]) == 4


def test_runner_removes_only_a_new_failed_session_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_root = tmp_path / "scratch"
    run_dir = figure_2.get_run_dir("figure2-test", scratch_root=scratch_root)
    session_dir = figure_2.get_session_dir(
        run_dir,
        animal_name="L14",
        date="20240611",
    )

    def fail(**kwargs: object) -> None:
        del kwargs
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "partial.nc").write_bytes(b"partial")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(
        figure_2,
        "prepare_figure_2_campaign",
        lambda **_kwargs: (run_dir, {}, {}),
    )
    monkeypatch.setattr(figure_2, "_run_figure_2_session", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        figure_2.run_figure_2_session(
            run_id="figure2-test",
            animal_name="L14",
            date="20240611",
            scratch_root=scratch_root,
        )
    assert not session_dir.exists()

    session_dir.mkdir(parents=True)
    sentinel = session_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        figure_2.run_figure_2_session(
            run_id="figure2-test",
            animal_name="L14",
            date="20240611",
            scratch_root=scratch_root,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    lock_path = session_dir.parent / f".{session_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            figure_2.run_figure_2_session(
                run_id="figure2-test",
                animal_name="L14",
                date="20240611",
                scratch_root=scratch_root,
            )
