"""Tests for database-free Figure 3 campaign orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from v1ca1.paper_figures.datasets import PROCESSED_DATASETS
from v1ca1.spyglass.offline import figure_3
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import provenance_sha256


def _expected_session_keys() -> list[tuple[str, str]]:
    """Return the four manuscript session identities in canonical order."""
    return [
        (str(animal_name), str(date))
        for animal_name, date, _light, _dark, _sleep in PROCESSED_DATASETS
    ]


def _parent_snapshot() -> dict[str, Any]:
    """Return one compact synthetic Figure 2 parent snapshot."""
    return {
        "run_id": "figure2-parent",
        "campaign_manifest_sha256": "a" * 64,
        "sessions": [
            {
                "animal_name": animal_name,
                "date": date,
                "session_manifest_path": (
                    f"{animal_name}/{date}/session_manifest.json"
                ),
                "session_manifest_sha256": str(index) * 64,
            }
            for index, (animal_name, date) in enumerate(
                _expected_session_keys(), start=1
            )
        ],
    }


def _minimal_session(
    *,
    animal_name: str,
    date: str,
) -> dict[str, Any]:
    """Return one loader-ready session with the canonical artifact roles."""
    special = (animal_name, date) == figure_3.FIGURE_3_XCORR_SESSION
    parent = _parent_snapshot()
    parent_artifacts = {"marker": f"{animal_name}-{date}"}
    return {
        "schema_version": figure_3.MANIFEST_SCHEMA_VERSION,
        "run_id": "figure3-test",
        "status": "complete",
        "animal_name": animal_name,
        "date": date,
        "nwb_file_name": f"{animal_name}{date}_augmented.nwb",
        "nwb_path": f"/stelmo/nwb/raw/{animal_name}{date}_augmented.nwb",
        "nwb_fingerprint": {"identifier": f"{animal_name}-{date}"},
        "epochs": {"light": figure_3.FIGURE_3_LIGHT_EPOCH},
        "regions": list(figure_3.FIGURE_3_REGIONS),
        "runtime_provenance": {
            "ripple_glm_jax": {
                "jax_version": "test",
                "default_backend": "gpu",
                "device_count": 1,
                "devices": [
                    {
                        "id": 0,
                        "platform": "gpu",
                        "device_kind": "synthetic GPU",
                        "process_index": 0,
                    }
                ],
            }
        },
        "parent_figure_2": parent,
        "parent_artifacts": parent_artifacts,
        "artifacts": {
            "ripple_modulation": [
                {"region": region} for region in figure_3.FIGURE_3_REGIONS
            ],
            "ripple_glm": [
                {
                    "source_predictor_mode": mode,
                    "ripple_glm_id": f"glm-{mode}",
                    "analysis_status": "valid",
                    "artifact_dir": f"glm/{mode}",
                }
                for mode in ("unit_vector", "mean_activity")
            ],
            "ripple_cross_region_xcorr": (
                [
                    {
                        "ripple_cross_region_xcorr_id": "xcorr-id",
                        "artifact_dir": "xcorr",
                    }
                ]
                if special
                else []
            ),
            "panel_b_schematic": (
                [
                    {
                        "payload_path": "figure_payloads/panel_b_schematic.npz",
                        "artifact_sha256": {"payload_path": "b" * 64},
                    }
                ]
                if special
                else []
            ),
        },
    }


def _patch_session_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    session: dict[str, Any],
) -> None:
    """Replace artifact and parent I/O with deterministic synthetic loaders."""
    monkeypatch.setattr(figure_3, "_verify_record", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        figure_3.ripple_glm,
        "load_ripple_glm_artifact",
        lambda path: {
            "ripple_glm_id": f"glm-{Path(path).name}",
            "analysis_status": "valid",
        },
    )
    monkeypatch.setattr(
        figure_3.ripple_cross_region_xcorr,
        "load_ripple_cross_region_xcorr_artifact",
        lambda _path: {"ripple_cross_region_xcorr_id": "xcorr-id"},
    )
    monkeypatch.setattr(
        figure_3,
        "load_panel_b_schematic_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        figure_3,
        "build_figure_2_parent_snapshot",
        lambda *_args, **_kwargs: session["parent_figure_2"],
    )
    expected_parent = {
        field: session[field]
        for field in ("nwb_file_name", "nwb_path", "nwb_fingerprint")
    }
    monkeypatch.setattr(
        figure_3,
        "_load_parent_session",
        lambda *_args, **_kwargs: (
            Path("/tmp/parent"),
            expected_parent,
            session["parent_artifacts"],
        ),
    )


def test_configuration_freezes_the_approved_figure_3_contract() -> None:
    parent = _parent_snapshot()

    configuration = figure_3.build_figure_3_configuration(parent)

    assert configuration["pipeline"] == "figure_3"
    assert configuration["parent_figure_2"] == parent
    assert configuration["epoch"] == "02_r1"
    assert configuration["condition"] == "AB"
    assert configuration["regions"] == ["ca1", "v1"]
    assert configuration["ripple_event_policy"] == {
        "detector_zscore_threshold": 2.0,
        "speed_gated": True,
        "additional_mean_zscore_filter": None,
    }
    assert configuration["diagnostic_figures"] is False
    assert configuration["artifact_origin"] == "computed"
    assert configuration["ripple_glm_runtime_policy"] == {
        "required_jax_platform": "gpu",
        "minimum_visible_gpu_devices": 1,
        "fail_before_session_artifacts": True,
    }

    modulation = configuration["ripple_modulation_parameters"]
    assert modulation["expected_detector_zscore_threshold"] == 2.0
    assert modulation["require_speed_gated"] is True
    glm_presets = configuration["ripple_glm_parameter_presets"]
    assert {row["source_predictor_mode"] for row in glm_presets} == {
        "unit_vector",
        "mean_activity",
    }
    assert all(row["ripple_selection_mode"] == "single" for row in glm_presets)
    assert all(row["n_shuffles_ripple"] == 100 for row in glm_presets)
    assert all(row["expected_detector_zscore_threshold"] == 2.0 for row in glm_presets)
    assert all(row["require_speed_gated"] is True for row in glm_presets)
    xcorr = configuration["ripple_cross_region_xcorr_parameters"]
    assert xcorr["expected_detector_zscore_threshold"] == 2.0
    assert xcorr["require_speed_gated"] is True
    assert configuration["ripple_cross_region_xcorr_session"] == [
        "L15",
        "20241121",
    ]


def test_gpu_preflight_records_devices_and_rejects_cpu() -> None:
    class Device:
        id = 3
        platform = "gpu"
        device_kind = "Synthetic A100"
        process_index = 0

    class GPUJax:
        __version__ = "test"

        @staticmethod
        def devices(platform: str) -> list[Device]:
            assert platform == "gpu"
            return [Device()]

        @staticmethod
        def default_backend() -> str:
            return "gpu"

    provenance = figure_3._require_jax_gpu(GPUJax)

    assert provenance["default_backend"] == "gpu"
    assert provenance["device_count"] == 1
    assert provenance["devices"] == [
        {
            "id": 3,
            "platform": "gpu",
            "device_kind": "Synthetic A100",
            "process_index": 0,
        }
    ]

    class CPUJax(GPUJax):
        @staticmethod
        def devices(_platform: str) -> list[Device]:
            return []

        @staticmethod
        def default_backend() -> str:
            return "cpu"

    with pytest.raises(RuntimeError, match="refuses to fit RippleGLM on CPU"):
        figure_3._require_jax_gpu(CPUJax)


def test_parent_snapshot_hashes_four_completed_sessions_even_if_in_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_run_dir = tmp_path / "runs" / "figure2-parent"
    parent_run_dir.mkdir(parents=True)
    campaign_path = parent_run_dir / figure_3.CAMPAIGN_MANIFEST_FILENAME
    campaign_path.write_text('{"status":"in_progress"}\n', encoding="utf-8")
    summaries = []
    sessions = []
    for animal_name, date in _expected_session_keys():
        relative_path = Path(animal_name) / date / "session_manifest.json"
        path = parent_run_dir / relative_path
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"animal_name": animal_name, "date": date}),
            encoding="utf-8",
        )
        summaries.append(
            {
                "animal_name": animal_name,
                "date": date,
                "session_manifest_path": relative_path.as_posix(),
                "status": "complete",
            }
        )
        sessions.append({"animal_name": animal_name, "date": date})
    campaign = {"status": "in_progress", "sessions": summaries}
    monkeypatch.setattr(
        figure_3,
        "load_figure_2_campaign",
        lambda *_args, **_kwargs: (parent_run_dir, campaign, sessions),
    )

    snapshot = figure_3.build_figure_2_parent_snapshot(
        "figure2-parent",
        scratch_root=tmp_path,
    )

    assert snapshot["campaign_manifest_sha256"] == file_sha256(campaign_path)
    assert [
        (row["animal_name"], row["date"]) for row in snapshot["sessions"]
    ] == sorted(_expected_session_keys())
    assert all(
        row["session_manifest_sha256"]
        == file_sha256(parent_run_dir / row["session_manifest_path"])
        for row in snapshot["sessions"]
    )

    changed = copy.deepcopy(campaign)
    changed["sessions"][0]["status"] = "in_progress"
    monkeypatch.setattr(
        figure_3,
        "load_figure_2_campaign",
        lambda *_args, **_kwargs: (parent_run_dir, changed, sessions),
    )
    with pytest.raises(ValueError, match="session must be complete"):
        figure_3.build_figure_2_parent_snapshot(
            "figure2-parent",
            scratch_root=tmp_path,
        )


def test_ripple_provenance_requires_detector_two_and_speed_gate() -> None:
    row = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "epoch": "02_r1",
        "ripple_count": 3,
        "detector_zscore_threshold": 2.0,
        "speed_gated": True,
        "source_object_id": "ripples-id",
    }

    provenance = figure_3._ripple_provenance(row)

    assert provenance["detector_zscore_threshold"] == 2.0
    assert provenance["speed_gated"] is True
    changed = {**row, "detector_zscore_threshold": 2.1}
    with pytest.raises(ValueError, match="threshold 2.0"):
        figure_3._ripple_provenance(changed)
    changed = {**row, "speed_gated": False}
    with pytest.raises(ValueError, match="speed-gated"):
        figure_3._ripple_provenance(changed)


@pytest.mark.parametrize(
    ("animal_name", "date", "special_count"),
    (("L14", "20240611", 0), ("L15", "20241121", 1)),
)
def test_session_loader_enforces_light_only_artifact_counts_and_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    animal_name: str,
    date: str,
    special_count: int,
) -> None:
    run_dir = tmp_path / "runs" / "figure3-test"
    run_dir.mkdir(parents=True)
    session = _minimal_session(animal_name=animal_name, date=date)
    path = run_dir / animal_name / date / "session_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(session), encoding="utf-8")
    _patch_session_dependencies(monkeypatch, session)

    loaded = figure_3.load_figure_3_session_manifest(
        path,
        run_dir=run_dir,
        scratch_root=tmp_path,
    )

    assert loaded["epochs"] == {"light": "02_r1"}
    assert {row["region"] for row in loaded["artifacts"]["ripple_modulation"]} == {
        "ca1",
        "v1",
    }
    assert {
        row["source_predictor_mode"] for row in loaded["artifacts"]["ripple_glm"]
    } == {"unit_vector", "mean_activity"}
    assert len(loaded["artifacts"]["ripple_cross_region_xcorr"]) == special_count
    assert len(loaded["artifacts"]["panel_b_schematic"]) == special_count

    changed = copy.deepcopy(session)
    changed["epochs"] = {"light": "02_r1", "dark": "08_r4"}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="only the fixed light epoch"):
        figure_3.load_figure_3_session_manifest(
            path,
            run_dir=run_dir,
            scratch_root=tmp_path,
        )

    changed = copy.deepcopy(session)
    changed["artifacts"]["ripple_glm"][1]["source_predictor_mode"] = "unit_vector"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="model roles"):
        figure_3.load_figure_3_session_manifest(
            path,
            run_dir=run_dir,
            scratch_root=tmp_path,
        )


def test_record_verification_rejects_registered_or_changed_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    path = run_dir / "modulation" / "summary.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"computed")
    record: dict[str, Any] = {
        "artifact_origin": "computed",
        "summary_path": path.relative_to(run_dir).as_posix(),
        "artifact_sha256": {"summary_path": file_sha256(path)},
    }
    record["record_sha256"] = provenance_sha256(record)

    figure_3._verify_record(record, run_dir=run_dir)

    registered = {**record, "artifact_origin": "registered_existing"}
    with pytest.raises(ValueError, match="computed de novo"):
        figure_3._verify_record(registered, run_dir=run_dir)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        figure_3._verify_record(record, run_dir=run_dir)


def test_schematic_roundtrip_preserves_stable_and_sorting_unit_ids(
    tmp_path: Path,
) -> None:
    payload = {
        "metadata": {
            "schema_version": figure_3.SCHEMATIC_SCHEMA_VERSION,
            "artifact_origin": "computed_from_augmented_nwb",
        },
        "time_s": np.asarray([-0.08, 0.0, 0.22]),
        "filtered_lfp": np.asarray([0.1, -0.2, 0.3]),
        "ripple_start_s": 10.0,
        "ripple_end_s": 10.06,
        "mean_zscore": 4.5,
        "n_ripples": 7,
        "channel": 12,
        "sampling_frequency_hz": 500.0,
        "selection_score": np.asarray([1.0, 2.0]),
        "ca1_unit_ids": ["101"],
        "v1_unit_ids": ["201"],
        "ca1_unit_identity": [
            {
                "spikesorting_merge_id": "merge",
                "unit_id": 101,
                "sorting_unit_id": 24,
                "region": "ca1",
            }
        ],
        "v1_unit_identity": [
            {
                "spikesorting_merge_id": "merge",
                "unit_id": 201,
                "sorting_unit_id": 32,
                "region": "v1",
            }
        ],
        "ca1_spike_times_s": [np.asarray([-0.01, 0.03])],
        "v1_spike_times_s": [np.asarray([0.02])],
    }
    path = tmp_path / "panel_b_schematic.npz"

    figure_3._write_schematic_payload(payload, path)
    loaded = figure_3.load_panel_b_schematic_payload(
        path,
        expected_sha256=file_sha256(path),
    )

    assert loaded["ca1_unit_ids"].tolist() == ["101"]
    assert loaded["v1_unit_ids"].tolist() == ["201"]
    assert loaded["ca1_unit_identity"][0]["sorting_unit_id"] == 24
    assert loaded["v1_unit_identity"][0]["sorting_unit_id"] == 32
    assert loaded["ca1_spike_times_s"][0].tolist() == pytest.approx(
        [-0.01, 0.03]
    )
