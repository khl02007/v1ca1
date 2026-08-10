"""Tests for database-free Spyglass campaign execution helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass.offline import manifests
from v1ca1.spyglass.offline import sources
from v1ca1.spyglass.offline.figure_1 import (
    DEFAULT_STABILITY_TUNING_PARAM_NAME,
    DEFAULT_TUNING_PARAMETER_PRESETS,
    _parameter_configuration,
)
from v1ca1.spyglass.offline import figure_1 as offline_figure_1
from v1ca1.spyglass.table_specs import DEFAULT_MOVEMENT_PARAMETERS


class _FakeTs:
    """Minimal Pynapple Ts test double."""

    def __init__(self, t, **kwargs) -> None:
        del kwargs
        self.t = np.asarray(t, dtype=float)


class _FakeTsGroup(dict):
    """Minimal Pynapple TsGroup test double."""

    def __init__(self, data, **kwargs) -> None:
        del kwargs
        super().__init__(data)


class _FakeNap:
    """Pynapple surface required by build_spike_tsgroup."""

    Ts = _FakeTs
    TsGroup = _FakeTsGroup

    @staticmethod
    def IntervalSet(*, start, end, **kwargs):
        del kwargs
        return SimpleNamespace(start=np.asarray(start), end=np.asarray(end))


class _FakeUnits:
    """NWB Units object exposing one dataframe and object id."""

    object_id = "units-object-id"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "spike_times": [
                    np.asarray([1.0, 2.0]),
                    np.asarray([1.5]),
                    np.asarray([2.5]),
                ],
                "region": ["V1", "ca1", "v1"],
                "sorting_unit_id": [101, 202, 303],
            },
            index=[4, 8, 12],
        )


def test_imported_merge_id_matches_datajoint_key_hash() -> None:
    file_name = "L1420240611_augmented.nwb"
    expected_hex = hashlib.md5(
        f"{file_name}ImportedSpikeSorting".encode(),
        usedforsecurity=False,
    ).hexdigest()

    merge_id = sources.imported_spike_sorting_merge_id(file_name)

    assert merge_id.replace("-", "") == expected_hex
    with pytest.raises(ValueError, match="basename"):
        sources.imported_spike_sorting_merge_id(f"raw/{file_name}")


def test_load_nwb_region_spikes_uses_units_index_and_seconds() -> None:
    loaded = sources.load_nwb_region_spikes(
        SimpleNamespace(units=_FakeUnits()),
        nwb_file_name="L1420240611_augmented.nwb",
        region="v1",
        time_support=(0.5, 3.0),
        pynapple_module=_FakeNap,
    )

    assert loaded["n_units"] == 2
    assert [row["unit_id"] for row in loaded["unit_ids"]] == [4, 12]
    assert all(
        row["spikesorting_merge_id"] == loaded["spikesorting_merge_id"]
        for row in loaded["unit_ids"]
    )
    np.testing.assert_allclose(loaded["spike_times_s"][0], [1.0, 2.0])
    assert list(loaded["ts_group"]) == [0, 1]


def test_nwb_session_identity_must_match_cli_labels() -> None:
    nwbfile = SimpleNamespace(
        subject=SimpleNamespace(subject_id="L14"),
        session_start_time=datetime(2024, 6, 11, tzinfo=timezone.utc),
    )
    sources.validate_nwb_session_identity(
        nwbfile,
        animal_name="L14",
        date="20240611",
    )
    with pytest.raises(ValueError, match="subject_id"):
        sources.validate_nwb_session_identity(
            nwbfile,
            animal_name="L15",
            date="20240611",
        )
    with pytest.raises(ValueError, match="session_start_time date"):
        sources.validate_nwb_session_identity(
            nwbfile,
            animal_name="L14",
            date="20240612",
        )


def test_figure_1_catalog_requires_an_explicit_dark_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = {
        "epoch_intervals": [
            {
                "epoch": "02_r1",
                "epoch_type": "run",
                "condition": "AB",
                "is_light": True,
            }
        ],
        "position": [
            {
                "epoch": "02_r1",
                "position_role": "head",
                "spatial_unit": "cm",
            }
        ],
        "trajectory_intervals": [
            {"epoch": "02_r1", "trajectory_type": "center_to_left"}
        ],
        "wtrack_graph": [
            {"configuration_name": "center_to_left", "coordinate_unit": "cm"}
        ],
    }
    monkeypatch.setattr(
        sources,
        "catalog_augmented_nwb",
        lambda *args, **kwargs: catalog,
    )

    with pytest.raises(ValueError, match="dark run epoch"):
        sources.select_figure_1_catalog(
            object(),
            nwb_file_name="L1420240611_augmented.nwb",
            epoch="02_r1",
            trajectory_types=("center_to_left",),
        )

    catalog["epoch_intervals"][0].update(
        {"epoch": "08_r4", "condition": "dark", "is_light": False}
    )
    catalog["position"][0]["epoch"] = "08_r4"
    catalog["trajectory_intervals"][0]["epoch"] = "08_r4"
    selected = sources.select_figure_1_catalog(
        object(),
        nwb_file_name="L1420240611_augmented.nwb",
        epoch="08_r4",
        trajectory_types=("center_to_left",),
    )
    assert selected["epoch_row"]["condition"] == "dark"


def _complete_session(run_dir: Path) -> dict[str, object]:
    """Create the smallest complete manifest artifact set."""
    session_dir = run_dir / "L14" / "20240611"
    movement_dir = session_dir / "movement"
    movement_dir.mkdir(parents=True)
    firing_rate_path = movement_dir / "movement.parquet"
    interval_path = movement_dir / "intervals.npz"
    tuning_path = session_dir / "curve.nc"
    stability_path = session_dir / "stability.parquet"
    for path in (firing_rate_path, interval_path, tuning_path, stability_path):
        path.write_bytes(b"artifact")
    return {
        "schema_version": manifests.MANIFEST_SCHEMA_VERSION,
        "run_id": "campaign-a",
        "status": "complete",
        "animal_name": "L14",
        "date": "20240611",
        "nwb_file_name": "L1420240611_augmented.nwb",
        "nwb_path": "/stelmo/nwb/raw/L1420240611_augmented.nwb",
        "artifacts": {
            "movement_firing_rate": [
                {
                    "artifact_dir": manifests.relative_run_path(
                        movement_dir,
                        run_dir=run_dir,
                    ),
                    "firing_rate_path": manifests.relative_run_path(
                        firing_rate_path,
                        run_dir=run_dir,
                    ),
                    "movement_intervals_path": manifests.relative_run_path(
                        interval_path,
                        run_dir=run_dir,
                    ),
                }
            ],
            "path_specific_place_tuning_curve": [
                {
                    "tuning_curve_path": manifests.relative_run_path(
                        tuning_path,
                        run_dir=run_dir,
                    )
                }
            ],
            "path_specific_place_stability": [
                {
                    "stability_path": manifests.relative_run_path(
                        stability_path,
                        run_dir=run_dir,
                    )
                }
            ],
        },
    }


def test_campaign_appends_complete_session_without_overwriting(tmp_path: Path) -> None:
    run_dir, campaign = manifests.prepare_campaign(
        run_id="campaign-a",
        analysis_parameters={"value": 1},
        source_identity_policy={"policy": "test"},
        scratch_root=tmp_path,
    )
    assert campaign["status"] == "in_progress"
    session = _complete_session(run_dir)
    session_path = run_dir / "L14" / "20240611" / "session_manifest.json"
    manifests.write_json_once(session, session_path)
    manifests.append_session_manifest(campaign, session, run_dir=run_dir)

    loaded = manifests.load_campaign_manifest(
        "campaign-a",
        scratch_root=tmp_path,
    )

    assert loaded["sessions"][0]["session_manifest_path"] == (
        "L14/20240611/session_manifest.json"
    )
    with pytest.raises(FileExistsError, match="overwrite"):
        manifests.write_json_once(session, session_path)
    with pytest.raises(FileExistsError, match="already contains"):
        manifests.append_session_manifest(loaded, session, run_dir=run_dir)


def test_campaign_append_reloads_a_stale_session_index(tmp_path: Path) -> None:
    run_dir, initial = manifests.prepare_campaign(
        run_id="campaign-a",
        analysis_parameters={"value": 1},
        source_identity_policy={"policy": "test"},
        scratch_root=tmp_path,
    )
    first = {
        "animal_name": "L14",
        "date": "20240611",
        "nwb_file_name": "L1420240611_augmented.nwb",
        "nwb_path": "/stelmo/nwb/raw/L1420240611_augmented.nwb",
        "status": "complete",
    }
    second = {
        "animal_name": "L15",
        "date": "20241121",
        "nwb_file_name": "L1520241121_augmented.nwb",
        "nwb_path": "/stelmo/nwb/raw/L1520241121_augmented.nwb",
        "status": "complete",
    }

    manifests.append_session_manifest(initial, first, run_dir=run_dir)
    manifests.append_session_manifest(initial, second, run_dir=run_dir)

    current = manifests.load_json(run_dir / "manifest.json")
    assert [
        (row["animal_name"], row["date"]) for row in current["sessions"]
    ] == [("L14", "20240611"), ("L15", "20241121")]


def test_manifest_paths_cannot_escape_run(tmp_path: Path) -> None:
    run_dir = manifests.get_run_dir("campaign-a", scratch_root=tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        manifests.relative_run_path(tmp_path / "outside.nc", run_dir=run_dir)
    with pytest.raises(ValueError, match="escapes"):
        manifests.resolve_run_path("../outside.nc", run_dir=run_dir)
    with pytest.raises(ValueError, match="component"):
        manifests.get_run_dir("../campaign-a", scratch_root=tmp_path)


def test_default_figure_1_configuration_has_both_curve_presets() -> None:
    configuration = _parameter_configuration(
        movement_parameters=DEFAULT_MOVEMENT_PARAMETERS,
        tuning_parameter_presets=DEFAULT_TUNING_PARAMETER_PRESETS,
        stability_tuning_param_name=DEFAULT_STABILITY_TUNING_PARAM_NAME,
        position_role="head",
        regions=("v1", "ca1"),
        trajectory_types=(
            "center_to_left",
            "center_to_right",
            "left_to_center",
            "right_to_center",
        ),
    )

    presets = {
        row["tuning_curve_param_name"]: row
        for row in configuration["tuning_curve_parameter_presets"]
    }
    assert set(presets) == {
        "legacy_4cm_unsmoothed",
        "figure1d_50bin_sigma1p5",
    }
    assert presets["legacy_4cm_unsmoothed"]["place_bin_size_cm"] == 4.0
    assert presets["figure1d_50bin_sigma1p5"]["position_bin_count"] == 50
    assert configuration["stability_tuning_curve_param_name"] == (
        "legacy_4cm_unsmoothed"
    )
    assert configuration["diagnostic_figures"] is False

    with pytest.raises(ValueError, match="regions"):
        _parameter_configuration(
            movement_parameters=DEFAULT_MOVEMENT_PARAMETERS,
            tuning_parameter_presets=DEFAULT_TUNING_PARAMETER_PRESETS,
            stability_tuning_param_name=DEFAULT_STABILITY_TUNING_PARAM_NAME,
            position_role="head",
            regions=("",),
            trajectory_types=("center_to_left",),
        )

    invalid_movement = dict(DEFAULT_MOVEMENT_PARAMETERS)
    invalid_movement["speed_threshold_cm_s"] = True
    with pytest.raises(TypeError, match="speed_threshold_cm_s"):
        _parameter_configuration(
            movement_parameters=invalid_movement,
            tuning_parameter_presets=DEFAULT_TUNING_PARAMETER_PRESETS,
            stability_tuning_param_name=DEFAULT_STABILITY_TUNING_PARAM_NAME,
            position_role="head",
            regions=("v1",),
            trajectory_types=("center_to_left",),
        )


def test_campaign_manifest_is_plain_json(tmp_path: Path) -> None:
    run_dir, _ = manifests.prepare_campaign(
        run_id="campaign-a",
        analysis_parameters={"value": np.int64(1)},
        source_identity_policy={"policy": "test"},
        scratch_root=tmp_path,
    )

    payload = json.loads((run_dir / "manifest.json").read_text())

    assert payload["analysis_parameters"] == {"value": 1}
    assert payload["source_identity_policy"] == {"policy": "test"}


def test_runner_wires_one_region_path_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise runner control flow while replacing all scientific work."""
    nwb_path = tmp_path / "L1420240611_augmented.nwb"
    nwb_path.write_bytes(b"nwb-test-double")
    fake_nwbfile = SimpleNamespace(
        subject=SimpleNamespace(subject_id="L14"),
        session_start_time=datetime(2024, 6, 11, tzinfo=timezone.utc),
    )

    class _FakeIO:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def read(self):
            return fake_nwbfile

    monkeypatch.setitem(
        sys.modules,
        "pynwb",
        SimpleNamespace(NWBHDF5IO=_FakeIO),
    )
    catalog = {
        "epoch_row": {"start_time": 1.0, "stop_time": 9.0},
        "position_row": {
            "epoch": "08_r4",
            "position_series_name": "head_position",
            "position_role": "head",
            "analysis_start_offset_samples": 10,
            "spatial_unit": "cm",
            "source_table_path": "/intervals/position_epochs",
            "source_object_path": "/processing/behavior/position/head_position",
        },
    }
    source_objects = {
        "position": object(),
        "trajectory_intervals": {"center_to_left": object()},
        "graph_inputs": {"center_to_left": object()},
    }
    unit_ids = [{"spikesorting_merge_id": "merge-a", "unit_id": 4}]
    selected_units_sha256 = offline_figure_1.unit_identity_sha256(unit_ids)
    loaded_spikes = {
        "source": "ImportedSpikeSorting",
        "spikesorting_merge_id": "merge-a",
        "n_units": 1,
        "selected_units_sha256": selected_units_sha256,
        "unit_ids": unit_ids,
        "ts_group": object(),
    }
    monkeypatch.setattr(
        offline_figure_1,
        "select_figure_1_catalog",
        lambda *args, **kwargs: catalog,
    )
    monkeypatch.setattr(
        offline_figure_1,
        "load_figure_1_catalog_objects",
        lambda *args, **kwargs: source_objects,
    )
    monkeypatch.setattr(
        offline_figure_1,
        "load_nwb_region_spikes",
        lambda *args, **kwargs: loaded_spikes,
    )
    monkeypatch.setattr(
        offline_figure_1,
        "nwb_fingerprint",
        lambda *args, **kwargs: {
            "resolved_path": str(nwb_path.resolve()),
            "size_bytes": nwb_path.stat().st_size,
            "mtime_ns": nwb_path.stat().st_mtime_ns,
            "nwb_identifier": "test",
            "units_object_id": "units",
            "full_file_sha256": None,
        },
    )

    movement_table = pd.DataFrame({"stable_unit_id": ["merge-a:4"]})
    movement_intervals = object()
    monkeypatch.setattr(
        offline_figure_1.movement,
        "compute_selected_movement_firing_rate",
        lambda **kwargs: {
            "table": movement_table,
            "movement_intervals": movement_intervals,
            "analysis_status": "valid",
            "n_units": 1,
            "n_valid_units": 1,
        },
    )

    def _write_movement(table, intervals, artifact_dir, **kwargs):
        del table, intervals, kwargs
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True)
        firing_rate_path = artifact_dir / "movement_firing_rate.parquet"
        movement_intervals_path = artifact_dir / "movement_intervals.npz"
        firing_rate_path.write_bytes(b"movement")
        movement_intervals_path.write_bytes(b"intervals")
        return {
            "firing_rate_path": firing_rate_path,
            "movement_intervals_path": movement_intervals_path,
        }

    monkeypatch.setattr(
        offline_figure_1.movement,
        "write_movement_artifacts",
        _write_movement,
    )
    monkeypatch.setattr(
        offline_figure_1.movement,
        "load_movement_artifacts",
        lambda path: {
            "table": movement_table,
            "movement_intervals": movement_intervals,
            "analysis_status": "valid",
        },
    )

    saved_curves: dict[str, object] = {}

    class _FakeCurve:
        def __init__(self) -> None:
            self.attrs: dict[str, object] = {}

    monkeypatch.setattr(
        offline_figure_1.path_specific_place,
        "compute_selected_path_specific_place_tuning_curve",
        lambda **kwargs: {
            "tuning_curve": _FakeCurve(),
            "analysis_status": "valid",
            "n_units": 1,
            "n_valid_units": 1,
            "n_trials": 2,
            "n_position_bins": 5,
        },
    )

    def _write_curve(curve, path, **kwargs):
        del kwargs
        path = Path(path)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"curve")
        saved_curves[str(path)] = curve
        return path

    monkeypatch.setattr(
        offline_figure_1.path_specific_place,
        "write_path_specific_place_artifact",
        _write_curve,
    )
    monkeypatch.setattr(
        offline_figure_1.path_specific_place,
        "load_path_specific_place_artifact",
        lambda path: saved_curves[str(path)],
    )
    monkeypatch.setattr(
        offline_figure_1.stability,
        "compute_selected_stability_from_tuning_curves",
        lambda **kwargs: {
            "table": pd.DataFrame(),
            "analysis_status": "valid",
            "n_units": 1,
            "n_valid_units": 1,
        },
    )

    def _write_stability(table, path, **kwargs):
        del table, kwargs
        path = Path(path)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"stability")
        return path

    monkeypatch.setattr(
        offline_figure_1.stability,
        "write_stability_artifact",
        _write_stability,
    )

    session = offline_figure_1.run_figure_1_session(
        run_id="campaign-a",
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        nwb_path=nwb_path,
        scratch_root=tmp_path / "scratch",
        regions=("v1",),
        trajectory_types=("center_to_left",),
        tuning_parameter_presets=(DEFAULT_TUNING_PARAMETER_PRESETS[0],),
    )

    assert session["status"] == "complete"
    assert len(session["artifacts"]["movement_firing_rate"]) == 1
    assert len(session["artifacts"]["path_specific_place_tuning_curve"]) == 3
    assert len(session["artifacts"]["path_specific_place_stability"]) == 1
    campaign = manifests.load_campaign_manifest(
        "campaign-a",
        scratch_root=tmp_path / "scratch",
    )
    assert campaign["sessions"][0]["status"] == "complete"


def test_runner_removes_only_new_session_directory_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_root = tmp_path / "scratch"
    run_dir = manifests.get_run_dir("campaign-a", scratch_root=scratch_root)
    session_dir = manifests.get_session_dir(
        run_dir,
        animal_name="L14",
        date="20240611",
    )

    def _fail(**kwargs):
        del kwargs
        if not session_dir.exists():
            session_dir.mkdir(parents=True)
            (session_dir / "partial.nc").write_bytes(b"partial")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(offline_figure_1, "_run_figure_1_session", _fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        offline_figure_1.run_figure_1_session(
            run_id="campaign-a",
            animal_name="L14",
            date="20240611",
            epoch="08_r4",
            scratch_root=scratch_root,
        )
    assert not session_dir.exists()

    session_dir.mkdir(parents=True)
    sentinel = session_dir / "preexisting.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="synthetic"):
        offline_figure_1.run_figure_1_session(
            run_id="campaign-a",
            animal_name="L14",
            date="20240611",
            epoch="08_r4",
            scratch_root=scratch_root,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
