"""Tests for the shared database-free supplementary campaign."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from v1ca1.spyglass.offline import supplementary_figures as module
from v1ca1.spyglass.offline.manifests import (
    CAMPAIGN_MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    SESSION_MANIFEST_FILENAME,
    file_sha256,
    get_run_dir,
    prepare_campaign,
    write_json_once,
)


def _write(path: Path, content: bytes = b"artifact") -> Path:
    """Write one small synthetic artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _parent_snapshot(
    parent_run_dir: Path,
    identities: list[tuple[str, str]],
) -> dict[str, object]:
    """Write parent session manifests and return their frozen snapshot."""
    sessions = []
    for animal_name, date in identities:
        relative = f"{animal_name}/{date}/{SESSION_MANIFEST_FILENAME}"
        path = parent_run_dir / relative
        write_json_once(
            {
                "animal_name": animal_name,
                "date": date,
                "status": "complete",
            },
            path,
        )
        sessions.append(
            {
                "animal_name": animal_name,
                "date": date,
                "session_manifest_path": relative,
                "session_manifest_sha256": file_sha256(path),
            }
        )
    campaign_path = _write(
        parent_run_dir / CAMPAIGN_MANIFEST_FILENAME,
        b"parent campaign",
    )
    return {
        "run_id": parent_run_dir.name,
        "campaign_manifest_sha256": file_sha256(campaign_path),
        "sessions": sessions,
    }


def _bundle_record(
    run_dir: Path,
    name: str,
    *,
    extra_path_fields: tuple[str, ...] = (),
    **metadata: object,
) -> dict[str, object]:
    """Create one sealed run-local record for contract validation."""
    artifact_dir = run_dir / "bundles" / name
    result_path = _write(artifact_dir / "result.bin", name.encode())
    paths = {
        "artifact_dir": artifact_dir,
        "result_path": result_path,
        **{
            field: _write(artifact_dir / f"{field}.bin", field.encode())
            for field in extra_path_fields
        },
    }
    return module._relative_artifact_record(
        {
            "artifact_origin": "computed",
            **metadata,
        },
        paths,
        run_dir=run_dir,
        path_fields=("result_path", *extra_path_fields),
    )


def test_configuration_defines_fixed_artifact_contract(tmp_path: Path) -> None:
    """Configuration pins its parent and the three requested families."""
    parent = {
        "run_id": "figure2",
        "campaign_manifest_sha256": "a" * 64,
        "sessions": [],
    }
    configuration = module.build_supplementary_figures_configuration(parent)

    assert configuration["pipeline"] == "supplementary_figures"
    assert configuration["parent_figure_2"] == parent
    assert configuration["epoch_motor_behavior_epochs"] == ["dark", "AB"]
    assert configuration["light_test_tuning_curve_parameters"] == dict(
        module.LEGACY_TUNING_CURVE_PARAMETERS
    )
    assert configuration["artifact_origin"] == "computed"
    assert module.ARTIFACT_FAMILIES == (
        "cv_pca",
        "epoch_motor_behavior",
        "swap_tuning_curve_comparison",
    )


def test_build_parent_snapshot_freezes_complete_figure_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent selection freezes every manuscript session and file hash."""
    parent_run_dir = tmp_path / "runs" / "parent"
    identities = sorted(module._EXPECTED_SESSIONS)
    snapshot = _parent_snapshot(parent_run_dir, identities)
    summaries = [
        {
            "animal_name": row["animal_name"],
            "date": row["date"],
            "session_manifest_path": row["session_manifest_path"],
            "status": "complete",
        }
        for row in snapshot["sessions"]
    ]
    campaign = {"sessions": summaries}
    sessions = [{"identity": identity} for identity in identities]
    monkeypatch.setattr(
        module,
        "load_figure_2_campaign",
        lambda *_args, **_kwargs: (parent_run_dir, campaign, sessions),
    )

    observed = module.build_figure_2_parent_snapshot(
        "parent",
        scratch_root=tmp_path,
    )

    assert observed == snapshot
    campaign["sessions"][0]["status"] = "in_progress"
    with pytest.raises(ValueError, match="must be complete"):
        module.build_figure_2_parent_snapshot("parent", scratch_root=tmp_path)


def test_load_parent_sessions_checks_snapshot_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public parent loader rejects a mutated Figure 2 session."""
    parent_run_dir = get_run_dir("parent", scratch_root=tmp_path)
    identities = [("A", "20200101"), ("B", "20200102")]
    snapshot = _parent_snapshot(parent_run_dir, identities)
    monkeypatch.setattr(
        module,
        "build_figure_2_parent_snapshot",
        lambda *_args, **_kwargs: copy.deepcopy(snapshot),
    )
    monkeypatch.setattr(
        module,
        "load_figure_2_session_manifest",
        lambda path, **_kwargs: module.load_json(path),
    )
    campaign = {"analysis_parameters": {"parent_figure_2": snapshot}}

    observed_run_dir, sessions = module.load_parent_figure_2_sessions(
        campaign,
        scratch_root=tmp_path,
    )

    assert observed_run_dir == parent_run_dir
    assert [(row["animal_name"], row["date"]) for row in sessions] == identities

    changed_path = parent_run_dir / snapshot["sessions"][0][
        "session_manifest_path"
    ]
    changed_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        module.load_parent_figure_2_sessions(campaign, scratch_root=tmp_path)


def test_sealed_record_rejects_tampered_or_external_artifacts(
    tmp_path: Path,
) -> None:
    """Run-local record checksums cover metadata, files, and containment."""
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "bundle"
    result_path = _write(artifact_dir / "result.bin")
    record = module._relative_artifact_record(
        {"artifact_origin": "computed", "analysis_status": "valid"},
        {"artifact_dir": artifact_dir, "result_path": result_path},
        run_dir=run_dir,
        path_fields=("result_path",),
    )
    module._verify_record(record, run_dir=run_dir)

    result_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        module._verify_record(record, run_dir=run_dir)

    outside = _write(tmp_path / "outside.bin")
    with pytest.raises(ValueError, match="escapes the run"):
        module._relative_artifact_record(
            {"artifact_origin": "computed"},
            {"artifact_dir": artifact_dir, "result_path": outside},
            run_dir=run_dir,
            path_fields=("result_path",),
        )


def test_compute_session_artifacts_uses_three_runner_seams() -> None:
    """The campaign dispatches one cvPCA, two motor, and one swap result."""
    calls: list[str] = []

    def cv_runner(context: object) -> dict[str, object]:
        calls.append("cv")
        return {"context": context}

    def motor_runner(context: object) -> list[dict[str, object]]:
        calls.append("motor")
        return [{"role": "dark"}, {"role": "AB"}]

    def swap_runner(context: object) -> dict[str, object]:
        calls.append("swap")
        return {"context": context}

    artifacts = module._compute_session_artifacts(
        {"key": "value"},
        cv_pca_runner=cv_runner,
        epoch_motor_runner=motor_runner,
        swap_tuning_runner=swap_runner,
    )

    assert calls == ["cv", "motor", "swap"]
    assert set(artifacts) == set(module.ARTIFACT_FAMILIES)
    assert {name: len(rows) for name, rows in artifacts.items()} == {
        "cv_pca": 1,
        "epoch_motor_behavior": 2,
        "swap_tuning_curve_comparison": 1,
    }


def test_cv_pca_positions_are_loaded_without_analysis_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cvPCA alone receives raw positions so it cannot double-apply offsets."""
    calls: list[tuple[object, object, bool]] = []

    def fake_load_position(
        nwbfile: object,
        row: object,
        *,
        apply_analysis_offset: bool,
    ) -> str:
        calls.append((nwbfile, row, apply_analysis_offset))
        return f"raw-{row}"

    monkeypatch.setattr(module, "load_position", fake_load_position)
    selections = {
        "dark": {"position_rows": {"head": "dark-row"}},
        "light_train": {"position_rows": {"head": "light-row"}},
        "light_test": {"position_rows": {"head": "test-row"}},
    }

    positions = module._load_cv_pca_positions("nwb", selections)

    assert positions == {
        "dark": "raw-dark-row",
        "light_train": "raw-light-row",
    }
    assert calls == [
        ("nwb", "dark-row", False),
        ("nwb", "light-row", False),
    ]


def test_run_cv_pca_uses_raw_position_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cvPCA runner never substitutes already-trimmed shared sources."""
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "_cv_pca_parameters",
        lambda _configuration: {"parameter_name": "p", "random_seed": 7},
    )
    monkeypatch.setattr(
        module,
        "_aligned_rates",
        lambda *_args, **_kwargs: np.zeros(1),
    )

    def fake_compute(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"analysis_status": "valid"}

    monkeypatch.setattr(module.cv_pca, "compute_cv_pca", fake_compute)
    monkeypatch.setattr(
        module.cv_pca,
        "write_cv_pca_artifact",
        lambda *_args, **_kwargs: {
            "analysis_status": "valid",
            "artifact_paths": {},
        },
    )
    monkeypatch.setattr(
        module,
        "_relative_artifact_record",
        lambda record, *_args, **_kwargs: dict(record),
    )
    source = {
        "positions": {"head": "trimmed"},
        "trajectory_intervals": {},
        "graph_inputs": {name: object() for name in module.TRAJECTORY_TYPES},
    }
    context = {
        "configuration": {},
        "animal_name": "A",
        "date": "20200101",
        "epochs": {"dark": "d", "AB": "ab"},
        "source_identity": {"selected_units_sha256": "a" * 64},
        "parent_pointer": {"session_manifest_sha256": "b" * 64},
        "nwb_sources": {},
        "movement": {
            "dark": {
                "table": object(),
                "movement_intervals": object(),
                "firing_rate_path": _write(tmp_path / "dark.bin"),
            },
            "light_train": {
                "table": object(),
                "movement_intervals": object(),
                "firing_rate_path": _write(tmp_path / "light.bin"),
            },
        },
        "spikes": {"ts_group": object(), "unit_ids": []},
        "sources": {"dark": source, "light_train": source},
        "cv_pca_positions": {"dark": "raw-dark", "light_train": "raw-light"},
        "selections": {
            "dark": {
                "position_rows": {
                    "head": {"analysis_start_offset_samples": 11}
                }
            },
            "light_train": {
                "position_rows": {
                    "head": {"analysis_start_offset_samples": 11}
                }
            },
        },
        "run_dir": tmp_path,
    }

    module._run_cv_pca(context)

    assert captured["dark_position"] == "raw-dark"
    assert captured["light_position"] == "raw-light"
    assert captured["position_offset_samples"] == 11


def test_light_test_tuning_uses_canonical_selection_and_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BA tuning is computed de novo with canonical IDs, attrs, and paths."""
    calls: list[dict[str, object]] = []

    class Curve:
        """Minimal tuning curve exposing mutable attributes."""

        def __init__(self) -> None:
            self.attrs: dict[str, object] = {}

    def fake_compute(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"tuning_curve": Curve(), "analysis_status": "valid"}

    def fake_path(**kwargs: object) -> Path:
        return tmp_path / f"{kwargs['trajectory_type']}.nc"

    def fake_write(_curve: object, path: Path) -> Path:
        return _write(path)

    monkeypatch.setattr(
        module.path_specific_place,
        "compute_selected_path_specific_place_tuning_curve",
        fake_compute,
    )
    monkeypatch.setattr(
        module.path_specific_place,
        "get_path_specific_place_artifact_path",
        fake_path,
    )
    monkeypatch.setattr(
        module.path_specific_place,
        "write_path_specific_place_artifact",
        fake_write,
    )
    frozen_parameters = dict(module.LEGACY_TUNING_CURVE_PARAMETERS)
    frozen_parameters["tuning_curve_param_name"] = "frozen-ba"
    context = {
        "configuration": {
            "light_test_tuning_curve_parameters": frozen_parameters
        },
        "animal_name": "A",
        "date": "20200101",
        "nwb_file_name": "A.nwb",
        "epochs": {"BA": "06_r3"},
        "run_dir": tmp_path,
        "source_identity": {"selected_units_sha256": "a" * 64},
        "spikes": {"ts_group": object(), "unit_ids": []},
        "movement": {
            "light_test": {
                "record": {"movement_firing_rate_id": "movement-id"},
                "movement_intervals": object(),
                "analysis_status": "valid",
            }
        },
        "sources": {
            "light_test": {
                "positions": {"head": object()},
                "trajectory_intervals": {
                    name: object() for name in module.TRAJECTORY_TYPES
                },
                "graph_inputs": {
                    name: object() for name in module.TRAJECTORY_TYPES
                },
            }
        },
    }

    records = module._compute_light_test_tuning(context)

    assert tuple(records) == tuple(module.TRAJECTORY_TYPES)
    assert len(calls) == 4
    for trajectory_type, values in records.items():
        record = values["record"]
        assert record["nwb_file_name"] == "A.nwb"
        assert record["epoch"] == "06_r3"
        assert record["trajectory_type"] == trajectory_type
        assert record["trial_subset"] == "all"
        assert record["tuning_curve_param_name"] == "frozen-ba"
        assert record["movement_firing_rate_id"] == "movement-id"
        assert record["selected_units_sha256"] == "a" * 64
        assert values["path"].is_file()


def test_artifact_contract_validates_roles_models_and_bundle_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session validation enforces exact counts, roles, models, and IDs."""
    run_dir = tmp_path / "run"
    cv_record = _bundle_record(
        run_dir,
        "cv",
        animal_name="A",
        date="20200101",
        region="v1",
        dark_epoch="dark",
        light_epoch="02_r1",
        cv_pca_id="cv-id",
        parameter_name="cv-preset",
        analysis_status="valid",
    )
    dark_motor = _bundle_record(
        run_dir,
        "motor-dark",
        animal_name="A",
        date="20200101",
        epoch="dark",
        condition="dark",
        epoch_motor_behavior_id="motor-dark-id",
        parameter_name="motor-preset",
        analysis_status="valid",
    )
    light_motor = _bundle_record(
        run_dir,
        "motor-light",
        animal_name="A",
        date="20200101",
        epoch="02_r1",
        condition="AB",
        epoch_motor_behavior_id="motor-light-id",
        parameter_name="motor-preset",
        analysis_status="valid",
    )
    ba_path_fields = tuple(
        f"light_test_{name}_tuning_curve_path"
        for name in module.TRAJECTORY_TYPES
    )
    swap_record = _bundle_record(
        run_dir,
        "swap",
        extra_path_fields=ba_path_fields,
        animal_name="A",
        date="20200101",
        region="v1",
        dark_epoch="dark",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
        model_family="empirical_swap_tuning",
        models=list(module.swap_tuning.MODEL_NAMES),
        swap_tuning_curve_comparison_id="swap-id",
        parameter_name="swap-preset",
        analysis_status="valid",
    )
    session = {
        "animal_name": "A",
        "date": "20200101",
        "epochs": {"dark": "dark", "AB": "02_r1", "BA": "06_r3"},
        "parameters": {
            "cv_pca_parameters": {"cv_pca_param_name": "cv-preset"},
            "epoch_motor_behavior_parameters": {
                "epoch_motor_behavior_param_name": "motor-preset"
            },
            "swap_tuning_curve_comparison_parameters": {
                "swap_tuning_curve_comparison_param_name": "swap-preset"
            },
        },
        "artifacts": {
            "cv_pca": [cv_record],
            "epoch_motor_behavior": [dark_motor, light_motor],
            "swap_tuning_curve_comparison": [swap_record],
        },
    }
    monkeypatch.setattr(
        module.cv_pca,
        "load_cv_pca_artifact",
        lambda _path: {"cv_pca_id": "cv-id", "analysis_status": "valid"},
    )
    motor_ids = {
        "motor-dark": "motor-dark-id",
        "motor-light": "motor-light-id",
    }
    monkeypatch.setattr(
        module.epoch_motor_behavior,
        "load_epoch_motor_behavior_artifact",
        lambda path: {
            "metadata": {"epoch_motor_behavior_id": motor_ids[path.name]},
            "analysis_status": "valid",
        },
    )
    monkeypatch.setattr(
        module.swap_tuning,
        "load_swap_tuning_curve_comparison_artifact",
        lambda _path: {
            "metadata": {"swap_tuning_curve_comparison_id": "swap-id"},
            "analysis_status": "valid",
            "upstream_provenance": {
                "source_tuning_curve_sha256_by_role_trajectory": {
                    "light_test": {
                        name: swap_record["artifact_sha256"][
                            f"light_test_{name}_tuning_curve_path"
                        ]
                        for name in module.TRAJECTORY_TYPES
                    }
                }
            },
        },
    )

    module._validate_session_artifact_contract(session, run_dir=run_dir)

    missing_motor = copy.deepcopy(session)
    missing_motor["artifacts"]["epoch_motor_behavior"].pop()
    with pytest.raises(ValueError, match="must contain 2"):
        module._validate_session_artifact_contract(
            missing_motor,
            run_dir=run_dir,
        )

    stale_models = copy.deepcopy(session)
    stale_models["artifacts"]["swap_tuning_curve_comparison"][0][
        "models"
    ] = ["legacy"]
    stale_record = stale_models["artifacts"][
        "swap_tuning_curve_comparison"
    ][0]
    stale_record.pop("record_sha256")
    stale_models["artifacts"]["swap_tuning_curve_comparison"][0] = (
        module._seal_record(stale_record)
    )
    with pytest.raises(ValueError, match="metadata is stale"):
        module._validate_session_artifact_contract(
            stale_models,
            run_dir=run_dir,
        )


def test_load_empty_campaign_validates_parent_and_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public loader returns its documented tuple after parent validation."""
    parent = {
        "run_id": "parent",
        "campaign_manifest_sha256": "a" * 64,
        "sessions": [],
    }
    monkeypatch.setattr(
        module,
        "build_figure_2_parent_snapshot",
        lambda *_args, **_kwargs: copy.deepcopy(parent),
    )
    configuration = module.build_supplementary_figures_configuration(parent)
    run_dir, expected = prepare_campaign(
        run_id="supp",
        analysis_parameters=configuration,
        source_identity_policy=module.SOURCE_IDENTITY_POLICY,
        scratch_root=tmp_path,
    )

    loaded_run_dir, campaign, sessions = (
        module.load_supplementary_figures_campaign(
            "supp",
            scratch_root=tmp_path,
        )
    )

    assert loaded_run_dir == run_dir
    assert campaign == expected
    assert sessions == []


def test_campaign_receipt_detects_child_manifest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Campaign indexes checksum each child manifest and reject mutation."""
    parent = {
        "run_id": "parent",
        "campaign_manifest_sha256": "a" * 64,
        "sessions": [],
    }
    monkeypatch.setattr(
        module,
        "build_figure_2_parent_snapshot",
        lambda *_args, **_kwargs: copy.deepcopy(parent),
    )
    configuration = module.build_supplementary_figures_configuration(parent)
    run_dir, campaign = prepare_campaign(
        run_id="supp-receipt",
        analysis_parameters=configuration,
        source_identity_policy=module.SOURCE_IDENTITY_POLICY,
        scratch_root=tmp_path,
    )
    session = {
        "run_id": "supp-receipt",
        "animal_name": "A",
        "date": "20200101",
        "nwb_file_name": "A.nwb",
        "nwb_path": "/data/A.nwb",
        "status": "complete",
        "parameters": configuration,
    }
    session_path = run_dir / "A/20200101" / SESSION_MANIFEST_FILENAME
    write_json_once(session, session_path)

    updated = module._append_supplementary_session_manifest(
        campaign,
        session,
        run_dir=run_dir,
    )

    assert updated["sessions"][0]["session_manifest_sha256"] == file_sha256(
        session_path
    )
    monkeypatch.setattr(
        module,
        "load_supplementary_figures_session_manifest",
        lambda *_args, **_kwargs: copy.deepcopy(session),
    )
    _run_dir, _campaign, sessions = module.load_supplementary_figures_campaign(
        "supp-receipt",
        scratch_root=tmp_path,
    )
    assert sessions == [session]

    session_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="session manifest checksum mismatch"):
        module.load_supplementary_figures_campaign(
            "supp-receipt",
            scratch_root=tmp_path,
        )


def test_session_loader_checks_parent_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session must point at its own frozen Figure 2 parent manifest."""
    parent_run_dir = get_run_dir("parent", scratch_root=tmp_path)
    parent_path = parent_run_dir / "A/20200101/session_manifest.json"
    fingerprint = {"resolved_path": "/data/A.nwb", "size_bytes": 1}
    parent_session = {
        "animal_name": "A",
        "date": "20200101",
        "nwb_file_name": "A.nwb",
        "nwb_path": "/data/A.nwb",
        "nwb_fingerprint": fingerprint,
        "epochs": {"dark": "dark", "AB": "02_r1", "BA": "06_r3"},
    }
    write_json_once(parent_session, parent_path)
    parent = {
        "run_id": "parent",
        "campaign_manifest_sha256": "a" * 64,
        "sessions": [
            {
                "animal_name": "A",
                "date": "20200101",
                "session_manifest_path": "A/20200101/session_manifest.json",
                "session_manifest_sha256": file_sha256(parent_path),
            }
        ],
    }
    pointer = module._parent_session_pointer(
        parent,
        animal_name="A",
        date="20200101",
    )
    run_dir = get_run_dir("supp", scratch_root=tmp_path)
    session = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "supp",
        "status": "complete",
        "animal_name": "A",
        "date": "20200101",
        "nwb_file_name": "A.nwb",
        "nwb_path": "/data/A.nwb",
        "nwb_fingerprint": fingerprint,
        "epochs": {"dark": "dark", "AB": "02_r1", "BA": "06_r3"},
        "regions": ["v1"],
        "parent_figure_2": parent,
        "parent_figure_2_session": pointer,
        "artifacts": {},
    }
    session_path = run_dir / "A/20200101/session_manifest.json"
    write_json_once(session, session_path)
    monkeypatch.setattr(
        module,
        "_validate_session_artifact_contract",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        module,
        "build_figure_2_parent_snapshot",
        lambda *_args, **_kwargs: copy.deepcopy(parent),
    )
    monkeypatch.setattr(
        module,
        "load_figure_2_session_manifest",
        lambda *_args, **_kwargs: copy.deepcopy(parent_session),
    )

    observed = module.load_supplementary_figures_session_manifest(
        session_path,
        run_dir=run_dir,
        scratch_root=tmp_path,
    )

    assert observed["parent_figure_2_session"] == pointer


def test_parser_exposes_explicit_session_and_parent() -> None:
    """CLI selection remains explicit and reproducible."""
    args = module._parser().parse_args(
        [
            "--run-id",
            "supp",
            "--parent-run-id",
            "parent",
            "--animal-name",
            "L14",
            "--date",
            "20240611",
        ]
    )

    assert args.run_id == "supp"
    assert args.parent_run_id == "parent"
    assert args.animal_name == "L14"
    assert args.date == "20240611"
