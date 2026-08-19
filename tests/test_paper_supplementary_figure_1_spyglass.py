"""Tests for the artifact-backed Supplementary Figure 1 adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import v1ca1.paper_figures.supplementary_figure_1_spyglass as figure


def _movement_table(*, region: str, digest: str) -> pd.DataFrame:
    """Return two synthetic all-unit movement-rate rows."""
    return pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "epoch": ["08_r4", "08_r4"],
            "region": [region, region],
            "unit_id": ["10", "20"],
            "stable_unit_id": [f"{digest}:10", f"{digest}:20"],
            "movement_firing_rate_hz": [0.25, 1.25],
        }
    )


def _stability_table(
    *,
    region: str,
    trajectory_type: str,
    digest: str,
) -> pd.DataFrame:
    """Return two synthetic odd/even stability rows."""
    return pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "epoch": ["08_r4", "08_r4"],
            "region": [region, region],
            "trajectory_type": [trajectory_type, trajectory_type],
            "unit_id": ["10", "20"],
            "stable_unit_id": [f"{digest}:10", f"{digest}:20"],
            "stability_correlation": [0.2, 0.8],
            "stability_status": ["valid", "valid"],
        }
    )


def _session_with_artifacts(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[Path, pd.DataFrame], dict[Path, pd.DataFrame]]:
    """Write checksum targets and return one synthetic session manifest."""
    movement_records = []
    stability_records = []
    source_identity = []
    movement_tables = {}
    stability_tables = {}
    for region in figure.STABILITY_REGIONS:
        digest = f"merge-{region}"
        source_identity.append(
            {
                "region": region,
                "source": "ImportedSpikeSorting",
                "n_units": 2,
                "selected_units_sha256": digest,
            }
        )
        movement_path = run_dir / f"movement-{region}.parquet"
        movement_path.write_bytes(region.encode("utf-8"))
        movement_tables[movement_path] = _movement_table(
            region=region,
            digest=digest,
        )
        movement_records.append(
            {
                "epoch": "08_r4",
                "region": region,
                "analysis_status": "valid",
                "n_units": 2,
                "selected_units_sha256": digest,
                "firing_rate_path": movement_path.name,
                "artifact_sha256": {
                    "firing_rate_path": figure.figure_1_adapter.file_sha256(
                        movement_path
                    )
                },
            }
        )
        for trajectory_type in figure.canonical.PANEL_D_TRAJECTORY_TYPES:
            stability_path = (
                run_dir / f"stability-{region}-{trajectory_type}.parquet"
            )
            stability_path.write_bytes(
                f"{region}-{trajectory_type}".encode("utf-8")
            )
            stability_tables[stability_path] = _stability_table(
                region=region,
                trajectory_type=trajectory_type,
                digest=digest,
            )
            stability_records.append(
                {
                    "epoch": "08_r4",
                    "region": region,
                    "trajectory_type": trajectory_type,
                    "tuning_curve_param_name": (
                        figure.figure_1_adapter.STABILITY_TUNING_PRESET
                    ),
                    "analysis_status": "valid",
                    "n_units": 2,
                    "selected_units_sha256": digest,
                    "stability_path": stability_path.name,
                    "artifact_sha256": {
                        "stability_path": (
                            figure.figure_1_adapter.file_sha256(stability_path)
                        )
                    },
                }
            )
    return (
        {
            "animal_name": "L14",
            "date": "20240611",
            "source_identity": source_identity,
            "artifacts": {
                "movement_firing_rate": movement_records,
                "path_specific_place_stability": stability_records,
            },
        },
        movement_tables,
        stability_tables,
    )


def _panel_payload(run_dir: Path) -> dict[str, Any]:
    """Return a minimal complete panel payload."""
    datasets = (("L14", "20240611", "08_r4"),)
    return {
        "run_dir": run_dir,
        "campaign": {"run_id": "test-run"},
        "datasets": datasets,
        "regions": figure.STABILITY_REGIONS,
        "dark_movement_firing_rate_table": pd.DataFrame(
            {"dark_firing_rate_hz": [0.25, 1.25]}
        ),
        "dark_stability_table": pd.DataFrame(
            {
                "region": ["v1", "ca1"],
                "trajectory_type": ["center_to_left", "center_to_left"],
                "stability_correlation": [0.2, 0.8],
            }
        ),
    }


def test_session_tables_load_checksum_validated_unfiltered_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, movement_tables, stability_tables = _session_with_artifacts(tmp_path)
    monkeypatch.setattr(
        figure.figure_1_adapter,
        "load_movement_firing_rate_artifact",
        lambda path: movement_tables[Path(path)].copy(),
    )
    monkeypatch.setattr(
        figure.figure_1_adapter,
        "_load_stability_artifact",
        lambda path: stability_tables[Path(path)].copy(),
    )

    movement, stability = figure._load_session_panel_tables(
        run_dir=tmp_path,
        session_manifest=session,
        epoch="08_r4",
    )

    assert movement["dark_firing_rate_hz"].tolist() == [0.25, 1.25]
    assert movement["unit"].tolist() == ["10", "20"]
    assert len(stability) == (
        len(figure.STABILITY_REGIONS)
        * len(figure.canonical.PANEL_D_TRAJECTORY_TYPES)
        * 2
    )
    assert set(stability["region"]) == set(figure.STABILITY_REGIONS)


def test_session_tables_reject_changed_or_misaligned_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, movement_tables, stability_tables = _session_with_artifacts(tmp_path)
    first_stability = next(iter(stability_tables))
    stability_tables[first_stability] = stability_tables[first_stability].iloc[
        :1
    ].copy()
    monkeypatch.setattr(
        figure.figure_1_adapter,
        "load_movement_firing_rate_artifact",
        lambda path: movement_tables[Path(path)].copy(),
    )
    monkeypatch.setattr(
        figure.figure_1_adapter,
        "_load_stability_artifact",
        lambda path: stability_tables[Path(path)].copy(),
    )

    with pytest.raises(ValueError, match="manifest unit counts disagree"):
        figure._load_session_panel_tables(
            run_dir=tmp_path,
            session_manifest=session,
            epoch="08_r4",
        )

    changed_path = tmp_path / "movement-v1.parquet"
    changed_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 digest does not match"):
        figure._load_session_panel_tables(
            run_dir=tmp_path,
            session_manifest=session,
            epoch="08_r4",
        )


def test_payload_requires_full_figure_1d_campaign_and_pools_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = (
        ("L1", "20240101", "08_r4"),
        ("L2", "20240102", "10_r5"),
    )
    sessions = [
        {"animal_name": animal_name, "date": date}
        for animal_name, date, _epoch in datasets
    ]
    calls = {}

    def fake_load_manifests(**kwargs: Any):
        calls["manifest_kwargs"] = kwargs
        return tmp_path, {"run_id": "run"}, sessions

    def fake_load_tables(**kwargs: Any):
        session = kwargs["session_manifest"]
        animal_name = str(session["animal_name"])
        return (
            pd.DataFrame(
                {
                    "animal_name": [animal_name],
                    "dark_firing_rate_hz": [1.0],
                }
            ),
            pd.DataFrame(
                {
                    "animal_name": [animal_name],
                    "stability_correlation": [0.5],
                }
            ),
        )

    monkeypatch.setattr(figure, "EXPECTED_DATASETS", datasets)
    monkeypatch.setattr(
        figure.figure_1_adapter,
        "load_figure_1_session_manifests",
        fake_load_manifests,
    )
    monkeypatch.setattr(figure, "_load_session_panel_tables", fake_load_tables)

    payload = figure.load_supplementary_figure_1_payload(
        run_id="run",
        scratch_root=tmp_path,
    )

    assert calls["manifest_kwargs"]["mode"] == "full"
    assert payload["datasets"] == datasets
    assert payload["dark_movement_firing_rate_table"][
        "animal_name"
    ].tolist() == ["L1", "L2"]
    assert payload["dark_stability_table"]["animal_name"].tolist() == [
        "L1",
        "L2",
    ]


def test_offline_sources_validate_requests_and_restore_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    payload = _panel_payload(tmp_path)

    def legacy_movement(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy movement loader was called")

    def legacy_stability(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy stability loader was called")

    monkeypatch.setattr(
        figure.canonical,
        "load_pooled_dark_movement_firing_rate_table",
        legacy_movement,
    )
    monkeypatch.setattr(
        figure.canonical,
        "load_dark_epoch_stability_table",
        legacy_stability,
    )
    with figure._offline_sources(payload):
        movement = figure.canonical.load_pooled_dark_movement_firing_rate_table(
            tmp_path,
            payload["datasets"],
            region="v1",
            cache_dir=tmp_path / "cache",
            refresh_cache=False,
        )
        stability = figure.canonical.load_dark_epoch_stability_table(
            data_root=tmp_path,
            datasets=payload["datasets"],
            regions=payload["regions"],
        )
        assert movement.equals(payload["dark_movement_firing_rate_table"])
        assert stability.equals(payload["dark_stability_table"])
        with pytest.raises(ValueError, match="foreign sessions"):
            figure.canonical.load_dark_epoch_stability_table(
                data_root=tmp_path,
                datasets=(("foreign", "20240101", "08_r4"),),
                regions=payload["regions"],
            )
    assert (
        figure.canonical.load_pooled_dark_movement_firing_rate_table
        is legacy_movement
    )
    assert figure.canonical.load_dark_epoch_stability_table is legacy_stability


def test_renderer_is_atomic_run_local_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    payload = _panel_payload(run_dir)
    (run_dir / figure.figure_1_adapter.CAMPAIGN_MANIFEST_FILENAME).write_text(
        json.dumps(payload["campaign"]),
        encoding="utf-8",
    )

    def fail_legacy(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("legacy analysis loader was called")

    monkeypatch.setattr(
        figure.canonical,
        "load_pooled_dark_movement_firing_rate_table",
        fail_legacy,
    )
    monkeypatch.setattr(
        figure.canonical,
        "load_dark_epoch_stability_table",
        fail_legacy,
    )

    def fake_make(**kwargs: Any) -> Path:
        movement = figure.canonical.load_pooled_dark_movement_firing_rate_table(
            kwargs["data_root"],
            kwargs["datasets"],
            region="v1",
            cache_dir=kwargs["dark_movement_fr_cache_dir"],
            refresh_cache=kwargs["refresh_dark_movement_fr_cache"],
        )
        stability = figure.canonical.load_dark_epoch_stability_table(
            data_root=kwargs["data_root"],
            datasets=kwargs["datasets"],
            regions=payload["regions"],
        )
        assert movement.equals(payload["dark_movement_firing_rate_table"])
        assert stability.equals(payload["dark_stability_table"])
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("<svg/>", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        figure.canonical,
        "make_supplementary_figure_1",
        fake_make,
    )
    output_path = figure.get_output_path(run_dir=run_dir)

    assert figure.render_supplementary_figure_1(
        payload,
        output_path=output_path,
        asset_dir=tmp_path,
        dpi=40,
    ) == output_path
    receipt_path = figure.get_figure_provenance_path(output_path)
    assert output_path.read_text(encoding="utf-8") == "<svg/>"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["artifact_kind"] == figure.FIGURE_ARTIFACT_KIND
    assert receipt["run_id"] == "test-run"
    assert (
        figure.canonical.load_pooled_dark_movement_firing_rate_table
        is fail_legacy
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        figure.render_supplementary_figure_1(
            payload,
            output_path=output_path,
            asset_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="inside its campaign run"):
        figure.render_supplementary_figure_1(
            payload,
            output_path=tmp_path / "legacy.svg",
            asset_dir=tmp_path,
        )


def test_cli_requires_run_and_explicit_promotion_replacement() -> None:
    with pytest.raises(SystemExit):
        figure.parse_arguments([])
    with pytest.raises(SystemExit):
        figure.parse_arguments(
            ["--run-id", "run", "--replace-promoted-output"]
        )

    args = figure.parse_arguments(["--run-id", "run"])
    assert args.run_id == "run"
    assert args.output_format == "svg"
    assert args.scratch_root == figure.DEFAULT_SCRATCH_ROOT
