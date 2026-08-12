"""Tests for the immutable offline Figure 3 schematic supplement."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass.offline import figure_3
from v1ca1.spyglass.offline import figure_3_schematic_supplement as supplement
from v1ca1.spyglass.offline.manifests import file_sha256
from v1ca1.spyglass.selection import provenance_sha256


def _sealed_record(run_dir: Path, relative_path: str, **fields: Any) -> dict[str, Any]:
    """Return one sealed computed record for a real synthetic file."""
    path = run_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    record = {
        **fields,
        "payload_path": relative_path,
        "artifact_origin": "computed",
        "artifact_sha256": {"payload_path": file_sha256(path)},
    }
    record["record_sha256"] = provenance_sha256(record)
    return record


def _loader_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Create a complete synthetic supplement and its pinned base sources."""
    base_run_dir = tmp_path / "runs" / "base"
    base_run_dir.mkdir(parents=True)
    modulation = _sealed_record(
        base_run_dir,
        "L15/20241121/modulation.parquet",
        animal_name="L15",
        date="20241121",
        epoch="02_r1",
        region="ca1",
        selected_units_sha256="units",
    )
    modulation["summary_path"] = modulation.pop("payload_path")
    modulation["artifact_sha256"] = {
        "summary_path": modulation["artifact_sha256"].pop("payload_path")
    }
    modulation.pop("record_sha256")
    modulation["record_sha256"] = provenance_sha256(modulation)
    superseded = _sealed_record(
        base_run_dir,
        "L15/20241121/old-schematic.npz",
        animal_name="L15",
        date="20241121",
        epoch="02_r1",
    )
    fingerprint = {"nwb_identifier": "L15", "full_file_sha256": None}
    sessions = [
        {
            "animal_name": animal_name,
            "date": date,
            "nwb_file_name": f"{animal_name}{date}_augmented.nwb",
            "nwb_path": f"/stelmo/nwb/raw/{animal_name}{date}_augmented.nwb",
            "nwb_fingerprint": (
                fingerprint
                if animal_name == "L15"
                else {"nwb_identifier": animal_name, "full_file_sha256": None}
            ),
            "artifacts": {
                "ripple_modulation": [modulation] if animal_name == "L15" else [],
                "panel_b_schematic": [superseded] if animal_name == "L15" else [],
            },
        }
        for animal_name, date in sorted(supplement._EXPECTED_SESSIONS)
    ]
    base_manifest_path = base_run_dir / "manifest.json"
    base_manifest_path.write_text('{"status":"in_progress"}\n', encoding="utf-8")
    snapshot = {
        "run_id": "base",
        "campaign_manifest_sha256": file_sha256(base_manifest_path),
        "sessions": [],
    }
    for session in sessions:
        path = (
            base_run_dir
            / session["animal_name"]
            / session["date"]
            / "session_manifest.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(session), encoding="utf-8")
        snapshot["sessions"].append(
            {
                "animal_name": session["animal_name"],
                "date": session["date"],
                "session_manifest_path": path.relative_to(base_run_dir).as_posix(),
                "session_manifest_sha256": file_sha256(path),
            }
        )

    run_dir = tmp_path / "runs" / "supplement"
    replacement = _sealed_record(
        run_dir,
        "L15/20241121/figure_payloads/panel_b_schematic.npz",
        animal_name="L15",
        date="20241121",
        epoch="02_r1",
        schema_version=figure_3.SCHEMATIC_SCHEMA_VERSION,
        n_ripples=7,
        ripple_start_s=10.0,
        ripple_end_s=10.15,
        channel=12,
        base_figure_3_run_id="base",
        ca1_modulation_record_sha256=modulation["record_sha256"],
        supersedes_record_sha256=superseded["record_sha256"],
        selector_policy_sha256=supplement.SCHEMATIC_SELECTOR_POLICY_SHA256,
    )
    l15 = next(row for row in sessions if row["animal_name"] == "L15")
    source = supplement._source_snapshot(l15, modulation, superseded)
    manifest = {
        "schema_version": supplement.MANIFEST_SCHEMA_VERSION,
        "run_id": "supplement",
        "status": "complete",
        "pipeline": supplement.SUPPLEMENT_PIPELINE,
        "base_figure_3": snapshot,
        "source": source,
        "selector_policy": supplement.SCHEMATIC_SELECTOR_POLICY,
        "selector_policy_sha256": supplement.SCHEMATIC_SELECTOR_POLICY_SHA256,
        "l15_nwb_fingerprint": fingerprint,
        "artifacts": {"panel_b_schematic": [replacement]},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return base_run_dir, manifest, sessions


def _payload_metadata(
    manifest: dict[str, Any],
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return strict loader-compatible synthetic replacement payload metadata."""
    l15 = next(row for row in sessions if row["animal_name"] == "L15")
    modulation = l15["artifacts"]["ripple_modulation"][0]
    superseded = l15["artifacts"]["panel_b_schematic"][0]
    metadata = {
        "schema_version": figure_3.SCHEMATIC_SCHEMA_VERSION,
        "animal_name": "L15",
        "date": "20241121",
        "epoch": "02_r1",
        "nwb_file_name": l15["nwb_file_name"],
        "artifact_origin": "computed_from_augmented_nwb",
        "supplement_run_id": "supplement",
        "base_figure_3_run_id": "base",
        "base_campaign_manifest_sha256": manifest["base_figure_3"][
            "campaign_manifest_sha256"
        ],
        "ca1_modulation_record_sha256": modulation["record_sha256"],
        "superseded_schematic_record_sha256": superseded["record_sha256"],
        "selector_policy": supplement.SCHEMATIC_SELECTOR_POLICY,
        "selector_policy_sha256": supplement.SCHEMATIC_SELECTOR_POLICY_SHA256,
        "n_units_per_region": figure_3.SCHEMATIC_N_UNITS_PER_REGION,
    }
    identities = [
        {"unit_id": index, "sorting_unit_id": index}
        for index in range(figure_3.SCHEMATIC_N_UNITS_PER_REGION)
    ]
    return {
        "metadata": metadata,
        "n_ripples": 7,
        "ripple_start_s": 10.0,
        "ripple_end_s": 10.15,
        "channel": 12,
        "ca1_unit_ids": np.asarray([str(index) for index in range(5)]),
        "v1_unit_ids": np.asarray([str(index) for index in range(5)]),
        "ca1_unit_identity": identities,
        "v1_unit_identity": identities,
        "ca1_spike_times_s": [np.asarray([]) for _index in range(5)],
        "v1_spike_times_s": [np.asarray([]) for _index in range(5)],
    }


def test_canonical_ca1_ranking_uses_response_then_modulation_then_numeric_id() -> None:
    summary = pd.DataFrame(
        {
            "unit_id": ["nwb-a", "nwb-b", "nwb-c", "nwb-d", "drop"],
            "response_zscore": [2.0, 1.0, np.nan, np.nan, np.nan],
            "ripple_modulation_index": [0.1, 99.0, 0.8, -0.8, np.nan],
        }
    )
    spikes = {unit_id: np.asarray([]) for unit_id in summary["unit_id"]}

    ranked = supplement.rank_ca1_schematic_units(
        summary,
        ca1_spikes=spikes,
        sorting_unit_ids={
            "nwb-a": 40,
            "nwb-b": 2,
            "nwb-c": 10,
            "nwb-d": 3,
            "drop": 1,
        },
    )

    assert ranked == ["nwb-a", "nwb-b", "nwb-d", "nwb-c"]


def test_base_selector_default_is_unchanged_and_opt_in_ranking_preserves_nwb_ids(
) -> None:
    unit_ids = [str(index) for index in range(6)]
    summary = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "response_zscore": [100.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "ripple_modulation_index": [0.0, 6.0, 5.0, 4.0, 3.0, 2.0],
        }
    )
    spikes = {unit_id: np.asarray([0.01]) for unit_id in unit_ids}
    ripple_table = pd.DataFrame({"start_time": [0.0], "end_time": [0.15]})

    _row, base_ca1, _v1, _score = figure_3._select_schematic_event(
        ripple_table=ripple_table,
        ca1_modulation_summary=summary,
        ca1_spikes=spikes,
        v1_spikes=spikes,
    )
    _row, opted_ca1, _v1, _score = figure_3._select_schematic_event(
        ripple_table=ripple_table,
        ca1_modulation_summary=summary,
        ca1_spikes=spikes,
        v1_spikes=spikes,
        ranked_ca1_unit_ids=unit_ids,
    )

    assert base_ca1 == ["1", "2", "3", "4", "5"]
    assert opted_ca1 == ["0", "1", "2", "3", "4"]


def test_selector_policy_and_kwargs_must_be_paired() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        figure_3._build_schematic_payload(
            object(),
            animal_name="L15",
            date="20241121",
            epoch="02_r1",
            nwb_file_name="test.nwb",
            ripple_table=pd.DataFrame(),
            ca1={},
            v1={},
            ca1_modulation={},
            selector_kwargs={},
        )


def test_base_snapshot_accepts_in_progress_campaign_with_four_complete_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "base"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        '{"status":"in_progress"}\n', encoding="utf-8"
    )
    sessions = []
    summaries = []
    for animal_name, date in sorted(supplement._EXPECTED_SESSIONS):
        session = {"animal_name": animal_name, "date": date}
        path = run_dir / animal_name / date / "session_manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(session), encoding="utf-8")
        sessions.append(session)
        summaries.append(
            {
                **session,
                "status": "complete",
                "session_manifest_path": path.relative_to(run_dir).as_posix(),
            }
        )
    campaign = {"status": "in_progress", "sessions": summaries}
    monkeypatch.setattr(
        figure_3,
        "load_figure_3_campaign",
        lambda *_args, **_kwargs: (run_dir, campaign, sessions),
    )

    _run_dir, _campaign, _sessions, snapshot = (
        supplement._load_complete_base_campaign("base", scratch_root=tmp_path)
    )

    assert len(snapshot["sessions"]) == 4
    changed = copy.deepcopy(campaign)
    changed["sessions"][0]["status"] = "in_progress"
    monkeypatch.setattr(
        figure_3,
        "load_figure_3_campaign",
        lambda *_args, **_kwargs: (run_dir, changed, sessions),
    )
    with pytest.raises(ValueError, match="session must be complete"):
        supplement._load_complete_base_campaign("base", scratch_root=tmp_path)

    monkeypatch.setattr(
        figure_3,
        "load_figure_3_campaign",
        lambda *_args, **_kwargs: (run_dir, campaign, sessions[:-1]),
    )
    with pytest.raises(ValueError, match="exactly the four"):
        supplement._load_complete_base_campaign("base", scratch_root=tmp_path)

    duplicate_campaign = {
        **campaign,
        "sessions": [*summaries, copy.deepcopy(summaries[0])],
    }
    monkeypatch.setattr(
        figure_3,
        "load_figure_3_campaign",
        lambda *_args, **_kwargs: (run_dir, duplicate_campaign, sessions),
    )
    with pytest.raises(ValueError, match="exactly the four"):
        supplement._load_complete_base_campaign("base", scratch_root=tmp_path)


def test_base_snapshot_rejects_campaign_or_session_hash_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "run_id": "base",
        "campaign_manifest_sha256": "a" * 64,
        "sessions": [{"session_manifest_sha256": "b" * 64}],
    }
    changed = copy.deepcopy(snapshot)
    changed["sessions"][0]["session_manifest_sha256"] = "c" * 64
    monkeypatch.setattr(
        supplement,
        "_load_complete_base_campaign",
        lambda *_args, **_kwargs: (tmp_path / "runs" / "base", {}, [], changed),
    )

    with pytest.raises(ValueError, match="base Figure 3 campaign changed"):
        supplement._verify_base_snapshot(snapshot, scratch_root=tmp_path)


def test_supplement_loader_rejects_symlinked_run_escape(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    outside = tmp_path / "outside"
    runs_root.mkdir()
    outside.mkdir()
    (outside / "manifest.json").write_text("{}\n", encoding="utf-8")
    (runs_root / "supplement").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes its scratch root"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="base",
            scratch_root=tmp_path,
        )


def test_supplement_loader_pins_base_records_and_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run_dir, manifest, sessions = _loader_fixture(tmp_path)
    payload = _payload_metadata(manifest, sessions)
    snapshot = manifest["base_figure_3"]
    monkeypatch.setattr(
        supplement,
        "_load_complete_base_campaign",
        lambda *_args, **_kwargs: (base_run_dir, {}, sessions, snapshot),
    )
    monkeypatch.setattr(
        supplement,
        "_validate_current_nwb",
        lambda _session: (Path(_session["nwb_path"]), _session["nwb_fingerprint"]),
    )
    monkeypatch.setattr(
        figure_3,
        "load_panel_b_schematic_payload",
        lambda *_args, **_kwargs: payload,
    )

    run_dir, loaded, loaded_payload = supplement.load_figure_3_schematic_supplement(
        "supplement",
        expected_base_run_id="base",
        scratch_root=tmp_path,
    )

    assert run_dir == (tmp_path / "runs" / "supplement").resolve()
    assert loaded == manifest
    assert loaded_payload is payload
    with pytest.raises(ValueError, match="different base campaign"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="other",
            scratch_root=tmp_path,
        )

    manifest_path = run_dir / "manifest.json"
    changed = copy.deepcopy(manifest)
    changed["source"]["ca1_modulation_record_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="source records changed"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="base",
            scratch_root=tmp_path,
        )

    changed = copy.deepcopy(manifest)
    changed_record = changed["artifacts"]["panel_b_schematic"][0]
    changed_record["n_ripples"] = 8
    changed_record.pop("record_sha256")
    changed_record["record_sha256"] = provenance_sha256(changed_record)
    manifest_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with its payload"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="base",
            scratch_root=tmp_path,
        )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    record = manifest["artifacts"]["panel_b_schematic"][0]
    (run_dir / record["payload_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="base",
            scratch_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("animal_name", "L14"),
        ("date", "20240611"),
        ("nwb_file_name", "wrong.nwb"),
        ("selector_policy_sha256", "0" * 64),
    ),
)
def test_supplement_loader_rejects_stale_payload_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    base_run_dir, manifest, sessions = _loader_fixture(tmp_path)
    snapshot = manifest["base_figure_3"]
    payload = _payload_metadata(manifest, sessions)
    payload["metadata"] = {**payload["metadata"], field: value}
    monkeypatch.setattr(
        supplement,
        "_load_complete_base_campaign",
        lambda *_args, **_kwargs: (base_run_dir, {}, sessions, snapshot),
    )
    monkeypatch.setattr(
        supplement,
        "_validate_current_nwb",
        lambda session: (Path(session["nwb_path"]), session["nwb_fingerprint"]),
    )
    monkeypatch.setattr(
        figure_3,
        "load_panel_b_schematic_payload",
        lambda *_args, **_kwargs: payload,
    )

    with pytest.raises(ValueError, match="payload has stale provenance"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="base",
            scratch_root=tmp_path,
        )


@pytest.mark.parametrize("path_value", ("../escape.npz", "/tmp/escape.npz"))
def test_supplement_loader_rejects_escaping_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_value: str,
) -> None:
    base_run_dir, manifest, sessions = _loader_fixture(tmp_path)
    snapshot = manifest["base_figure_3"]
    changed = copy.deepcopy(manifest)
    record = changed["artifacts"]["panel_b_schematic"][0]
    record["payload_path"] = path_value
    record.pop("record_sha256")
    record["record_sha256"] = provenance_sha256(record)
    run_dir = tmp_path / "runs" / "supplement"
    (run_dir / "manifest.json").write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(
        supplement,
        "_load_complete_base_campaign",
        lambda *_args, **_kwargs: (base_run_dir, {}, sessions, snapshot),
    )
    monkeypatch.setattr(
        supplement,
        "_validate_current_nwb",
        lambda session: (Path(session["nwb_path"]), session["nwb_fingerprint"]),
    )

    with pytest.raises(ValueError, match="run-relative|escapes"):
        supplement.load_figure_3_schematic_supplement(
            "supplement",
            expected_base_run_id="base",
            scratch_root=tmp_path,
        )


def test_builder_is_no_overwrite_and_never_preflights_or_fits_jax_glm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run_dir, _manifest, sessions = _loader_fixture(tmp_path)
    snapshot = {"run_id": "base", "campaign_manifest_sha256": "a" * 64, "sessions": []}
    l15 = next(row for row in sessions if row["animal_name"] == "L15")
    modulation = l15["artifacts"]["ripple_modulation"][0]
    superseded = l15["artifacts"]["panel_b_schematic"][0]
    monkeypatch.setattr(
        supplement,
        "_load_complete_base_campaign",
        lambda *_args, **_kwargs: (base_run_dir, {}, sessions, snapshot),
    )
    monkeypatch.setattr(
        supplement,
        "_l15_sources",
        lambda _sessions: (l15, modulation, superseded),
    )
    monkeypatch.setattr(figure_3, "_verify_record", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        figure_3,
        "_require_jax_gpu",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("JAX reached")),
    )
    monkeypatch.setattr(
        figure_3.ripple_glm,
        "compute_ripple_glm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("GLM reached")),
    )
    payload = {
        "metadata": {},
        "n_ripples": 1,
        "ripple_start_s": 1.0,
        "ripple_end_s": 1.1,
        "channel": 1,
    }
    monkeypatch.setattr(
        supplement,
        "_build_replacement_payload",
        lambda **_kwargs: (payload, l15["nwb_fingerprint"]),
    )

    def write_payload(_payload: dict[str, Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"schematic")
        return path

    monkeypatch.setattr(figure_3, "_write_schematic_payload", write_payload)

    created = supplement.build_figure_3_schematic_supplement(
        run_id="new-supplement",
        base_run_id="base",
        scratch_root=tmp_path,
    )

    assert created["status"] == "complete"
    assert created["artifacts"].keys() == {"panel_b_schematic"}
    with pytest.raises(FileExistsError, match="overwrite"):
        supplement.build_figure_3_schematic_supplement(
            run_id="new-supplement",
            base_run_id="base",
            scratch_root=tmp_path,
        )
