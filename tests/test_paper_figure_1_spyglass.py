"""Tests for the offline Spyglass Figure 1D artifact adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import v1ca1.paper_figures.figure_1_spyglass as figure


SESSION_PARAMETERS = {
    "pipeline": "figure_1_initial_slice",
    "movement_parameters": dict(figure.DEFAULT_MOVEMENT_PARAMETERS),
    "tuning_curve_parameter_presets": [
        dict(figure.LEGACY_TUNING_CURVE_PARAMETERS),
        dict(figure.FIGURE_1D_TUNING_CURVE_PARAMETERS),
    ],
    "stability_tuning_curve_param_name": figure.STABILITY_TUNING_PRESET,
    "trial_subsets": ["all", "odd", "even"],
    "position_role": "head",
    "regions": ["v1", "ca1"],
    "trajectory_types": list(figure.PANEL_D_TRAJECTORY_TYPES),
    "diagnostic_figures": False,
}


def _artifact_records(
    run_dir: Path,
    *,
    animal_name: str,
    date: str,
    epoch: str,
    region: str = "v1",
) -> dict[str, list[dict[str, Any]]]:
    """Create empty artifact files and their run-relative manifest records."""
    session_dir = run_dir / animal_name / date
    movement_id = f"movement-{animal_name}"
    selected_units_sha256 = f"units-{animal_name}"
    movement_dir = session_dir / "movement_firing_rate" / epoch / region / movement_id
    firing_rate_path = movement_dir / "movement_firing_rate.parquet"
    interval_path = movement_dir / "movement_intervals.npz"
    firing_rate_path.parent.mkdir(parents=True, exist_ok=True)
    firing_rate_path.touch()
    interval_path.touch()
    artifacts: dict[str, list[dict[str, Any]]] = {
        "movement_firing_rate": [
            {
                "epoch": epoch,
                "region": region,
                "movement_firing_rate_id": movement_id,
                "artifact_dir": str(movement_dir.relative_to(run_dir)),
                "firing_rate_path": str(firing_rate_path.relative_to(run_dir)),
                "movement_intervals_path": str(interval_path.relative_to(run_dir)),
                "analysis_status": "valid",
                "n_units": 2,
                "n_valid_units": 2,
                "selected_units_sha256": selected_units_sha256,
                "artifact_sha256": {
                    "firing_rate_path": figure.file_sha256(firing_rate_path),
                    "movement_intervals_path": figure.file_sha256(interval_path),
                },
            }
        ],
        "path_specific_place_tuning_curve": [],
        "path_specific_place_stability": [],
    }
    for trajectory_type in figure.PANEL_D_TRAJECTORY_TYPES:
        for trial_subset in ("all", "odd", "even"):
            for preset in (
                figure.FIGURE_1_TUNING_PRESET,
                figure.STABILITY_TUNING_PRESET,
            ):
                curve_id = (
                    f"curve-{animal_name}-{trajectory_type}-{trial_subset}-{preset}"
                )
                curve_path = (
                    session_dir
                    / "path_specific_place_tuning_curve"
                    / epoch
                    / trajectory_type
                    / trial_subset
                    / region
                    / curve_id
                    / "tuning_curve.nc"
                )
                curve_path.parent.mkdir(parents=True, exist_ok=True)
                curve_path.touch()
                artifacts["path_specific_place_tuning_curve"].append(
                    {
                        "epoch": epoch,
                        "region": region,
                        "trajectory_type": trajectory_type,
                        "trial_subset": trial_subset,
                        "tuning_curve_param_name": preset,
                        "path_specific_place_tuning_curve_id": curve_id,
                        "movement_firing_rate_id": movement_id,
                        "tuning_curve_path": str(curve_path.relative_to(run_dir)),
                        "analysis_status": "valid",
                        "n_units": 2,
                        "n_valid_units": 2,
                        "n_trials": 4,
                        "selected_units_sha256": selected_units_sha256,
                        "artifact_sha256": {
                            "tuning_curve_path": figure.file_sha256(curve_path),
                        },
                    }
                )
        stability_id = f"stability-{animal_name}-{trajectory_type}"
        stability_path = (
            session_dir
            / "path_specific_place_stability"
            / epoch
            / trajectory_type
            / region
            / stability_id
            / "stability.parquet"
        )
        stability_path.parent.mkdir(parents=True, exist_ok=True)
        stability_path.touch()
        artifacts["path_specific_place_stability"].append(
            {
                "epoch": epoch,
                "region": region,
                "trajectory_type": trajectory_type,
                "tuning_curve_param_name": figure.STABILITY_TUNING_PRESET,
                "path_specific_place_stability_id": stability_id,
                "odd_path_specific_place_tuning_curve_id": "odd-id",
                "even_path_specific_place_tuning_curve_id": "even-id",
                "stability_path": str(stability_path.relative_to(run_dir)),
                "analysis_status": "valid",
                "n_units": 2,
                "n_valid_units": 2,
                "selected_units_sha256": selected_units_sha256,
                "artifact_sha256": {
                    "stability_path": figure.file_sha256(stability_path),
                },
            }
        )
    return artifacts


def _write_run(
    scratch_root: Path,
    *,
    run_id: str = "test-run",
    datasets: tuple[tuple[str, str, str], ...] = (figure.L14_DATASET,),
) -> Path:
    """Write one coherent synthetic campaign and completed session manifests."""
    run_dir = scratch_root / figure.RUNS_DIRNAME / run_id
    run_dir.mkdir(parents=True)
    session_summaries = []
    for animal_name, date, epoch in datasets:
        nwb_path = scratch_root / "nwb" / f"{animal_name}{date}_augmented.nwb"
        nwb_path.parent.mkdir(parents=True, exist_ok=True)
        nwb_path.touch()
        nwb_stat = nwb_path.stat()
        session_manifest_path = (
            run_dir / animal_name / date / figure.SESSION_MANIFEST_FILENAME
        )
        session_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        session = {
            "schema_version": figure.MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "animal_name": animal_name,
            "date": date,
            "nwb_file_name": nwb_path.name,
            "nwb_path": str(nwb_path),
            "nwb_fingerprint": {
                "resolved_path": str(nwb_path.resolve()),
                "size_bytes": nwb_stat.st_size,
                "mtime_ns": nwb_stat.st_mtime_ns,
            },
            "code_provenance": {
                "v1ca1_git_commit": "abc123",
                "v1ca1_git_dirty": False,
            },
            "status": "complete",
            "position_selection": {
                "position_role": "head",
                "spatial_unit": "cm",
                "analysis_start_offset_samples": 10,
            },
            "epochs": [epoch],
            "regions": ["v1"],
            "trajectories": list(figure.PANEL_D_TRAJECTORY_TYPES),
            "parameters": SESSION_PARAMETERS,
            "selection_identity_scope": "offline_surrogate",
            "source_identity": [
                {
                    "region": "v1",
                    "source": "ImportedSpikeSorting",
                    "spikesorting_merge_id": f"merge-{animal_name}",
                    "offline_region_sorted_spikes_view_id": (
                        f"view-{animal_name}"
                    ),
                    "n_units": 2,
                    "selected_units_sha256": f"units-{animal_name}",
                }
            ],
            "artifacts": _artifact_records(
                run_dir,
                animal_name=animal_name,
                date=date,
                epoch=epoch,
            ),
        }
        session_manifest_path.write_text(json.dumps(session), encoding="utf-8")
        session_summaries.append(
            {
                "animal_name": animal_name,
                "date": date,
                "nwb_file_name": nwb_path.name,
                "nwb_path": str(nwb_path),
                "session_manifest_path": str(
                    session_manifest_path.relative_to(run_dir)
                ),
                "status": "complete",
            }
        )
    campaign = {
        "schema_version": figure.MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": "2026-08-09T00:00:00Z",
        "status": "in_progress",
        "code_provenance": {"v1ca1_git_commit": "abc123"},
        "analysis_parameters": SESSION_PARAMETERS,
        "source_identity_policy": dict(figure.SOURCE_IDENTITY_POLICY),
        "sessions": session_summaries,
    }
    (run_dir / figure.CAMPAIGN_MANIFEST_FILENAME).write_text(
        json.dumps(campaign),
        encoding="utf-8",
    )
    return run_dir


def _curve_from_artifact_path(path: Path) -> xr.DataArray:
    """Return a two-unit 50-bin curve identified by its canonical path."""
    parts = Path(path).parts
    artifact_index = parts.index("path_specific_place_tuning_curve")
    epoch, trajectory_type, trial_subset = parts[
        artifact_index + 1 : artifact_index + 4
    ]
    offset = 0.0 if trial_subset == "odd" else 1.0
    values = np.vstack(
        [
            np.linspace(offset, 1.0 + offset, 50),
            np.linspace(1.0 + offset, offset, 50),
        ]
    )
    return xr.DataArray(
        values,
        dims=("unit", "linear_position_cm"),
        coords={
            "unit": ["merge:10", "merge:20"],
            "stable_unit_id": ("unit", ["merge:10", "merge:20"]),
            "unit_id": ("unit", ["10", "20"]),
            "linear_position_cm": np.arange(50, dtype=float) + 0.5,
        },
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": epoch,
            "trajectory_type": trajectory_type,
            "trial_subset": trial_subset,
            "binning_mode": "bin_count",
            "bin_count": 50,
            "sigma_bins": 1.5,
        },
    )


def _movement_table() -> pd.DataFrame:
    """Return two units with only the first passing Figure 1's rate threshold."""
    return pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "epoch": ["08_r4", "08_r4"],
            "region": ["v1", "v1"],
            "stable_unit_id": ["merge:10", "merge:20"],
            "movement_firing_rate_hz": [0.7, 0.4],
        }
    )


def _stability_table(path: Path) -> pd.DataFrame:
    """Return valid stability for the first unit in one trajectory only."""
    parts = Path(path).parts
    artifact_index = parts.index("path_specific_place_stability")
    epoch, trajectory_type = parts[artifact_index + 1 : artifact_index + 3]
    correlation = 0.7 if trajectory_type == "center_to_left" else 0.2
    return pd.DataFrame(
        {
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "region": ["v1", "v1"],
            "epoch": [epoch, epoch],
            "trajectory_type": [trajectory_type, trajectory_type],
            "stable_unit_id": ["merge:10", "merge:20"],
            "unit_id": ["10", "20"],
            "stability_correlation": [correlation, 0.8],
            "stability_status": ["valid", "valid"],
        }
    )


def test_run_directory_requires_one_safe_component(tmp_path: Path) -> None:
    assert figure.get_run_dir(scratch_root=tmp_path, run_id="run-1") == (
        tmp_path / "runs" / "run-1"
    )
    with pytest.raises(ValueError, match="one non-empty path component"):
        figure.get_run_dir(scratch_root=tmp_path, run_id="../legacy")


def test_partial_mode_loads_l14_from_in_progress_campaign(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)

    loaded_run_dir, campaign, sessions = figure.load_figure_1_session_manifests(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="l14-validation",
    )

    assert loaded_run_dir == run_dir
    assert campaign["status"] == "in_progress"
    assert [(session["animal_name"], session["date"]) for session in sessions] == [
        ("L14", "20240611")
    ]


def test_campaign_rejects_unknown_source_identity_policy(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    manifest_path = run_dir / figure.CAMPAIGN_MANIFEST_FILENAME
    campaign = json.loads(manifest_path.read_text(encoding="utf-8"))
    campaign["source_identity_policy"] = {"source": "unknown"}
    manifest_path.write_text(json.dumps(campaign), encoding="utf-8")

    with pytest.raises(ValueError, match="source identity policy"):
        figure.load_figure_1_session_manifests(
            scratch_root=tmp_path,
            run_id="test-run",
            mode="l14-validation",
        )


def test_full_mode_requires_exactly_four_manuscript_sessions(tmp_path: Path) -> None:
    _write_run(tmp_path)

    with pytest.raises(ValueError, match="exactly the four manuscript sessions"):
        figure.load_figure_1_session_manifests(
            scratch_root=tmp_path,
            run_id="test-run",
            mode="full",
        )


def test_full_mode_accepts_four_completed_compatible_sessions(tmp_path: Path) -> None:
    _write_run(tmp_path, datasets=figure.FULL_DATASETS)

    _run_dir, _campaign, sessions = figure.load_figure_1_session_manifests(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="full",
    )

    assert [(session["animal_name"], session["date"]) for session in sessions] == [
        (animal_name, date) for animal_name, date, _epoch in figure.FULL_DATASETS
    ]


def test_full_mode_rejects_mixed_code_commits(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, datasets=figure.FULL_DATASETS)
    animal_name, date, _epoch = figure.FULL_DATASETS[-1]
    session_path = run_dir / animal_name / date / figure.SESSION_MANIFEST_FILENAME
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["code_provenance"]["v1ca1_git_commit"] = "different-commit"
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="different v1ca1 commits"):
        figure.load_figure_1_session_manifests(
            scratch_root=tmp_path,
            run_id="test-run",
            mode="full",
        )


def test_session_rejects_noncanonical_position_selection(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    session_path = run_dir / "L14" / "20240611" / figure.SESSION_MANIFEST_FILENAME
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["position_selection"]["analysis_start_offset_samples"] = 0
    session_path.write_text(json.dumps(session), encoding="utf-8")

    with pytest.raises(ValueError, match="10-sample analysis offset"):
        figure.load_figure_1_session_manifests(
            scratch_root=tmp_path,
            run_id="test-run",
            mode="l14-validation",
        )


def test_manifest_rejects_artifact_path_escape(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    session_path = run_dir / "L14" / "20240611" / figure.SESSION_MANIFEST_FILENAME
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["artifacts"]["movement_firing_rate"][0]["firing_rate_path"] = (
        "../../../../legacy.parquet"
    )
    session_path.write_text(json.dumps(session), encoding="utf-8")
    outside = tmp_path / "legacy.parquet"
    outside.touch()
    with pytest.raises(ValueError, match="escapes the run directory"):
        figure.load_figure_1_session_manifests(
            scratch_root=tmp_path,
            run_id="test-run",
            mode="l14-validation",
        )


def test_session_validation_rejects_changed_nwb_source(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    campaign = json.loads(
        (run_dir / figure.CAMPAIGN_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    Path(campaign["sessions"][0]["nwb_path"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed since offline computation"):
        figure.load_figure_1_session_manifests(
            scratch_root=tmp_path,
            run_id="test-run",
            mode="l14-validation",
        )


def test_artifact_loading_rejects_changed_file(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    _run_dir, _campaign, sessions = figure.load_figure_1_session_manifests(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="l14-validation",
    )
    movement_record = sessions[0]["artifacts"]["movement_firing_rate"][0]
    movement_path = run_dir / movement_record["firing_rate_path"]
    movement_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 digest does not match"):
        figure.load_session_curve_set(
            run_dir=run_dir,
            session_manifest=sessions[0],
            epoch="08_r4",
            region="v1",
        )


def test_session_curve_set_applies_rate_and_at_least_one_path_stability_filters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _write_run(tmp_path)
    _run_dir, _campaign, sessions = figure.load_figure_1_session_manifests(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="l14-validation",
    )
    monkeypatch.setattr(
        figure,
        "load_movement_firing_rate_artifact",
        lambda _path: _movement_table(),
    )
    monkeypatch.setattr(figure, "_load_stability_artifact", _stability_table)
    monkeypatch.setattr(
        figure,
        "load_path_specific_place_artifact",
        _curve_from_artifact_path,
    )

    curve_set = figure.load_session_curve_set(
        run_dir=run_dir,
        session_manifest=sessions[0],
        epoch="08_r4",
        region="v1",
    )

    assert curve_set["included_units"].tolist() == ["merge:10"]
    for subset_name in ("odd_curves", "even_curves"):
        for curve in curve_set[subset_name].values():
            assert curve.coords["unit"].values.tolist() == ["merge:10"]


def test_session_curve_set_rejects_stability_unit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = _write_run(tmp_path)
    _run_dir, _campaign, sessions = figure.load_figure_1_session_manifests(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="l14-validation",
    )
    monkeypatch.setattr(
        figure,
        "load_movement_firing_rate_artifact",
        lambda _path: _movement_table(),
    )
    monkeypatch.setattr(
        figure,
        "_load_stability_artifact",
        lambda path: _stability_table(path).iloc[:1].copy(),
    )

    with pytest.raises(ValueError, match="Stability units do not match"):
        figure.load_session_curve_set(
            run_dir=run_dir,
            session_manifest=sessions[0],
            epoch="08_r4",
            region="v1",
        )


def test_default_output_is_inside_retained_run(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "runs"
        / "test-run"
        / "figures"
        / "figure_1d_l14_spyglass_validation.svg"
    )
    assert figure.get_default_output_path(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="l14-validation",
        output_format="svg",
    ) == expected


def test_payload_builds_one_heatmap_block_per_requested_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "test-run"
    session = {"animal_name": "L14", "date": "20240611"}
    monkeypatch.setattr(
        figure,
        "load_figure_1_session_manifests",
        lambda **_kwargs: (run_dir, {"run_id": "test-run"}, [session]),
    )
    requested_regions = []

    def fake_load_session_curve_set(**kwargs):
        requested_regions.append(kwargs["region"])
        return {
            "animal_name": "L14",
            "date": "20240611",
            "region": kwargs["region"],
            "epoch": "08_r4",
        }

    def fake_build(curve_sets, **_kwargs):
        region = curve_sets[0]["region"]
        return {("region", "region"): region}, {"units": region}, {"peaks": region}

    monkeypatch.setattr(figure, "load_session_curve_set", fake_load_session_curve_set)
    monkeypatch.setattr(
        figure,
        "_build_pooled_panel_values_order_and_peaks",
        fake_build,
    )

    payload = figure.load_figure_1d_payload(
        scratch_root=tmp_path,
        run_id="test-run",
        mode="l14-validation",
        regions=("v1", "ca1"),
    )

    assert requested_regions == ["v1", "ca1"]
    assert payload["regions"] == ("v1", "ca1")
    assert set(payload["panels_by_region"]) == {"v1", "ca1"}


def test_cli_requires_explicit_run_and_mode() -> None:
    with pytest.raises(SystemExit):
        figure.parse_arguments([])

    args = figure.parse_arguments(
        ["--run-id", "test-run", "--mode", "l14-validation"]
    )
    assert args.run_id == "test-run"
    assert args.mode == "l14-validation"
    assert args.scratch_root == figure.DEFAULT_SCRATCH_ROOT
    assert args.region is None

    args = figure.parse_arguments(
        [
            "--run-id",
            "test-run",
            "--mode",
            "l14-validation",
            "--region",
            "v1",
            "--region",
            "ca1",
        ]
    )
    assert args.region == ["v1", "ca1"]


def test_renderer_stays_in_run_and_refuses_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    run_dir = tmp_path / "runs" / "test-run"
    run_dir.mkdir(parents=True)
    panels = {
        (order_trajectory, plot_trajectory): np.asarray([[0.0, 1.0]])
        for order_trajectory in figure.PANEL_D_TRAJECTORY_TYPES
        for plot_trajectory in figure.PANEL_D_TRAJECTORY_TYPES
    }
    payload = {
        "run_dir": run_dir,
        "mode": "l14-validation",
        "regions": ("v1", "ca1"),
        "panels_by_region": {"v1": panels, "ca1": panels},
    }
    output_path = run_dir / "figures" / "validation.png"

    assert figure.render_figure_1d_validation(
        payload,
        output_path=output_path,
        dpi=40,
    ) == output_path
    assert output_path.is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        figure.render_figure_1d_validation(
            payload,
            output_path=output_path,
            dpi=40,
        )
    with pytest.raises(ValueError, match="inside the selected run directory"):
        figure.render_figure_1d_validation(
            payload,
            output_path=tmp_path / "legacy.png",
            dpi=40,
        )
