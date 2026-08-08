"""Tests for database-free Spyglass cvPCA artifact handling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.spyglass import cv_pca as module


RESULT_ID = uuid.UUID("11312c29-f6eb-5239-94ca-175a37008b90")


class _Position:
    """Minimal position object exposing pynapple-compatible fields."""

    def __init__(self, values: np.ndarray, timestamps: np.ndarray):
        self.d = np.asarray(values, dtype=float)
        self.t = np.asarray(timestamps, dtype=float)


class _Intervals:
    """Minimal interval object exposing start/end arrays."""

    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)


def _graph(configuration_name: str) -> dict[str, object]:
    """Return one two-edge from-center graph in centimeters."""
    return {
        "configuration_name": configuration_name,
        "coordinate_unit": "cm",
        "track_graph_kwargs": {
            "node_positions": [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
            "edges": [[0, 1], [1, 2]],
        },
        "linearization_kwargs": {
            "edge_order": [[0, 1], [1, 2]],
            "edge_spacing": [0.0],
            "use_HMM": False,
        },
    }


def _trajectory_intervals(offset: float = 0.0) -> dict[str, _Intervals]:
    """Return four trajectories with exactly four laps each."""
    starts = np.asarray([0.0, 1.0, 2.0, 3.0]) + offset
    ends = starts + 0.8
    return {
        trajectory: _Intervals(starts, ends)
        for trajectory in module.TRAJECTORY_TYPES
    }


def _inputs() -> dict[str, object]:
    """Return one complete explicit NWB-derived input selection."""
    timestamps = np.linspace(0.0, 9.9, 100)
    positions = np.column_stack((timestamps, np.sin(timestamps)))
    return {
        "cv_pca_id": RESULT_ID,
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "light_epoch": "02_r1",
        "dark_epoch": "08_r4",
        "spikes": {"runtime-1": object(), "runtime-2": object()},
        "stable_unit_ids": [
            {"spikesorting_merge_id": "merge-v1", "unit_id": "1"},
            {"spikesorting_merge_id": "merge-v1", "unit_id": "2"},
        ],
        "light_position": _Position(positions, timestamps),
        "dark_position": _Position(positions + [0.1, 0.0], timestamps),
        "light_movement_intervals": _Intervals([0.0], [9.9]),
        "dark_movement_intervals": _Intervals([0.0], [9.9]),
        "light_movement_firing_rate_hz": [1.0, 1.5],
        "dark_movement_firing_rate_hz": [1.2, 1.7],
        "light_trajectory_intervals": _trajectory_intervals(),
        "dark_trajectory_intervals": _trajectory_intervals(),
        "graph_inputs": {
            "center_to_left": _graph("center_to_left"),
            "center_to_right": _graph("center_to_right"),
        },
        "upstream_provenance": {
            "nwb_file_name": "L14_20240611_augmented.nwb",
            "sorted_spikes_group": "both-probes-v1",
            "light_position_id": "head-02_r1",
            "dark_position_id": "head-08_r4",
        },
    }


def _pair_tensors() -> object:
    """Return a deterministic scientific PairTuningTensors value."""
    science = module._science_module()
    rng = np.random.default_rng(4)
    dark = rng.normal(size=(4, 8, 2))
    light = rng.normal(size=(4, 8, 2)) + 0.2 * dark
    selection = science.UnitSelection(
        keep_mask=np.asarray([True, True]),
        unit_classes=np.asarray(["shared_active", "shared_active"]),
        dark_active=np.asarray([True, True]),
        light_active=np.asarray([True, True]),
        dark_modulated=np.asarray([True, True]),
        light_modulated=np.asarray([True, True]),
    )
    return science.PairTuningTensors(
        dark=dark,
        light=light,
        unit_ids=np.asarray(["runtime-1", "runtime-2"]),
        unit_classes=np.asarray(["shared_active", "shared_active"]),
        dark_firing_rate_hz=np.asarray([1.2, 1.7]),
        light_firing_rate_hz=np.asarray([1.0, 1.5]),
        dark_condition_sd_hz=np.asarray([0.4, 0.5]),
        light_condition_sd_hz=np.asarray([0.6, 0.7]),
        condition_trajectory=np.asarray(
            [trajectory for trajectory in module.TRAJECTORY_TYPES for _ in range(2)]
        ),
        condition_bin_center=np.tile([2.0, 6.0], 4),
        condition_bin_index=np.tile([0, 1], 4),
        n_valid_bins_by_trajectory={trajectory: 2 for trajectory in module.TRAJECTORY_TYPES},
        unit_selection=selection,
    )


def _patch_science(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Patch only the explicit-input session adapter and grouped tuning stage."""
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(module, "_build_science_session", lambda **kwargs: kwargs)

    def build_pairwise(session, **kwargs):
        calls.append({"session": session, **kwargs})
        return _pair_tensors()

    monkeypatch.setattr(
        module._science_module(), "build_pairwise_tuning_tensors", build_pairwise
    )
    return calls


def _compute(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Compute one valid canonical result with scientific grouping patched."""
    _patch_science(monkeypatch)
    return module.compute_cv_pca(**_inputs())


def _copy_result(result: dict[str, object]) -> dict[str, object]:
    """Return one independently mutable in-memory result bundle."""
    copied = dict(result)
    for name in (
        "selected_units",
        "lap_assignments",
        "trajectory_qc",
        "summary",
        "spectrum",
    ):
        copied[name] = result[name].copy(deep=True)
    copied["dataset"] = result["dataset"].copy(deep=True)
    copied["upstream_provenance"] = dict(result["upstream_provenance"])
    copied["legacy_artifact_provenance"] = dict(
        result["legacy_artifact_provenance"]
    )
    return copied


def test_parameters_paths_and_legacy_paths_are_explicit(tmp_path: Path) -> None:
    parameters = module.validate_cv_pca_parameters(region="v1")
    assert parameters == {
        "bin_size_cm": 4.0,
        "n_groups": 4,
        "min_occupancy_s": 0.01,
        "unit_filter_mode": "shared-active",
        "min_firing_rate_hz": 0.5,
        "min_condition_sd_hz": 1e-6,
        "normalization": "zscore",
        "min_scale": 1e-6,
        "random_seed": 47,
    }
    assert module.validate_cv_pca_parameters(region="ca1")[
        "min_firing_rate_hz"
    ] == 0.0
    with pytest.raises(ValueError, match="at least 3"):
        module.validate_cv_pca_parameters(region="v1", n_groups=2)
    with pytest.raises(ValueError, match="required outside"):
        module.validate_cv_pca_parameters(region="other")

    paths = module.get_cv_pca_artifact_paths(
        animal_name="L14",
        date="20240611",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        region="v1",
        cv_pca_id=RESULT_ID,
        artifact_root=tmp_path,
    )
    assert paths["artifact_dir"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "cv_pca"
        / "02_r1_vs_08_r4"
        / "v1"
        / str(RESULT_ID)
    )
    legacy = module.get_legacy_cv_pca_paths(
        tmp_path / "L14" / "20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
    )
    assert legacy["result_path"].name == "v1_02_r1_vs_08_r4_cv_pca_seed47.nc"
    assert legacy["summary_path"].name == (
        "v1_02_r1_vs_08_r4_cv_pca_seed47_summary.parquet"
    )


def test_compute_reuses_science_with_four_paths_and_stable_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_science(monkeypatch)
    result = module.compute_cv_pca(**_inputs())
    assert result["analysis_status"] == "valid"
    assert result["artifact_origin"] == "computed"
    assert result["n_input_units"] == result["n_selected_units"] == 2
    assert len(calls) == 1
    assert calls[0]["group_seed"] == 47
    assert calls[0]["bin_size_cm"] == 4.0
    assert calls[0]["n_groups"] == 4
    assert calls[0]["unit_filter_mode"] == "shared-active"
    assert result["summary"]["condition"].tolist() == ["dark", "light"]
    assert result["summary"]["epoch"].tolist() == ["08_r4", "02_r1"]
    assert len(result["lap_assignments"]) == 32
    assert len(result["trajectory_qc"]) == 8
    assert result["trajectory_qc"]["n_shared_valid_bins"].eq(2).all()
    assert result["dataset"].coords["stable_unit_id"].values.tolist() == [
        "merge-v1:1",
        "merge-v1:2",
    ]
    assert result["selected_units"]["group_unit_id"].tolist() == [
        "runtime-1",
        "runtime-2",
    ]
    assert result["selected_units"]["included_in_cv_pca"].all()
    for prohibited in module.PROHIBITED_RESULT_VARIABLES:
        assert prohibited not in result["dataset"]


def test_explicit_inputs_build_the_legacy_science_session() -> None:
    inputs = _inputs()
    selected_graphs, graph_length, spacing = module._normalize_graph_inputs(
        inputs["graph_inputs"]
    )
    session = module._build_science_session(
        region="v1",
        spikes=inputs["spikes"],
        dark_epoch="08_r4",
        light_epoch="02_r1",
        dark_position=inputs["dark_position"],
        light_position=inputs["light_position"],
        dark_movement_intervals=inputs["dark_movement_intervals"],
        light_movement_intervals=inputs["light_movement_intervals"],
        dark_trajectory_intervals=inputs["dark_trajectory_intervals"],
        light_trajectory_intervals=inputs["light_trajectory_intervals"],
        dark_firing_rates=np.asarray([1.2, 1.7]),
        light_firing_rates=np.asarray([1.0, 1.5]),
        selected_graphs=selected_graphs,
        graph_length_cm=graph_length,
        edge_spacing=spacing,
        position_offset_samples=10,
    )
    assert session["position_offset"] == 10
    assert len(session["position_dict"]["02_r1"]) == 100
    assert len(session["timestamps_position_dict"]["02_r1"]) == 100
    assert session["track_total_length"] == pytest.approx(20.0)
    assert set(session["track_graphs_by_side"]) == {"left", "right"}
    np.testing.assert_allclose(
        session["trajectory_times"]["08_r4"]["center_to_left"],
        np.column_stack((np.arange(4.0), np.arange(4.0) + 0.8)),
    )
    assert session["movement_firing_rates_by_region"]["v1"]["02_r1"].tolist() == [
        1.0,
        1.5,
    ]


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("no_units", "no_units"),
        ("no_position", "no_valid_position"),
        ("no_movement", "no_movement"),
        ("no_trials", "no_trials"),
        ("insufficient_laps", "insufficient_laps"),
        ("no_eligible_units", "no_eligible_units"),
    ],
)
def test_expected_empty_inputs_create_terminal_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_status: str,
) -> None:
    inputs = _inputs()
    if mutation == "no_units":
        inputs.update(
            spikes={},
            stable_unit_ids=[],
            light_movement_firing_rate_hz=[],
            dark_movement_firing_rate_hz=[],
        )
    elif mutation == "no_position":
        position = inputs["light_position"]
        assert isinstance(position, _Position)
        position.d[10:] = np.nan
    elif mutation == "no_movement":
        inputs["dark_movement_intervals"] = _Intervals([], [])
    elif mutation == "no_trials":
        trajectories = dict(inputs["dark_trajectory_intervals"])
        trajectories["center_to_left"] = _Intervals([], [])
        inputs["dark_trajectory_intervals"] = trajectories
    elif mutation == "insufficient_laps":
        trajectories = dict(inputs["dark_trajectory_intervals"])
        trajectories["center_to_left"] = _Intervals([0.0, 1.0, 2.0], [0.8, 1.8, 2.8])
        inputs["dark_trajectory_intervals"] = trajectories
    else:
        inputs["light_movement_firing_rate_hz"] = [0.1, 0.2]
    monkeypatch.setattr(
        module,
        "_build_science_session",
        lambda **kwargs: pytest.fail("Terminal inputs must not invoke cvPCA science."),
    )
    result = module.compute_cv_pca(**inputs)
    assert result["analysis_status"] == expected_status
    assert result["n_selected_units"] == 0
    assert result["spectrum"].empty
    assert result["summary"]["within_cv_participation_ratio"].isna().all()
    assert result["dataset"].sizes["component"] == 0


def test_unexpected_scientific_errors_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_build_science_session", lambda **kwargs: kwargs)

    def fail(*args, **kwargs):
        raise ValueError("unanticipated numerical failure")

    monkeypatch.setattr(module._science_module(), "build_pairwise_tuning_tensors", fail)
    with pytest.raises(ValueError, match="unanticipated numerical failure"):
        module.compute_cv_pca(**_inputs())


@pytest.mark.parametrize(
    ("message", "science_message", "rate_values"),
    [
        (
            "no_shared_position_bins",
            "Too few shared valid bins for center_to_left: 1.",
            [1.0, 1.5],
        ),
        (
            "no_eligible_units",
            "No units remain after applying firing-rate and condition-SD filters.",
            [1.0, 1.5],
        ),
    ],
)
def test_expected_scientific_empty_results_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
    message: str,
    science_message: str,
    rate_values: list[float],
) -> None:
    inputs = _inputs()
    inputs["light_movement_firing_rate_hz"] = rate_values
    monkeypatch.setattr(module, "_build_science_session", lambda **kwargs: kwargs)

    def fail(*args, **kwargs):
        raise ValueError(science_message)

    monkeypatch.setattr(module._science_module(), "build_pairwise_tuning_tensors", fail)
    result = module.compute_cv_pca(**inputs)
    assert result["analysis_status"] == message
    assert result["n_selected_units"] == 0


def test_nan_rates_are_terminal_but_infinite_rates_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    inputs["light_movement_firing_rate_hz"] = [np.nan, np.nan]
    monkeypatch.setattr(
        module,
        "_build_science_session",
        lambda **kwargs: pytest.fail("All-NaN firing rates must terminate early."),
    )
    result = module.compute_cv_pca(**inputs)
    assert result["analysis_status"] == "no_eligible_units"

    inputs = _inputs()
    inputs["light_movement_firing_rate_hz"] = [np.inf, 1.0]
    with pytest.raises(ValueError, match="non-negative finite values or NaN"):
        module.compute_cv_pca(**inputs)


def test_validation_rejects_origin_metadata_and_scientific_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _compute(monkeypatch)

    changed = _copy_result(result)
    changed["artifact_origin"] = "unknown"
    changed["dataset"].attrs["artifact_origin"] = "unknown"
    with pytest.raises(ValueError, match="artifact_origin must"):
        module.validate_cv_pca_result(changed)

    changed = _copy_result(result)
    changed["artifact_origin"] = "registered_existing"
    changed["dataset"].attrs["artifact_origin"] = "registered_existing"
    with pytest.raises(ValueError, match="require legacy provenance"):
        module.validate_cv_pca_result(changed)

    changed = _copy_result(result)
    changed["position_offset_samples"] = 11
    with pytest.raises(ValueError, match="position offset"):
        module.validate_cv_pca_result(changed)

    changed = _copy_result(result)
    changed["dataset"].attrs["result_schema_version"] = "stale"
    with pytest.raises(ValueError, match="schema version"):
        module.validate_cv_pca_result(changed)

    changed = _copy_result(result)
    changed["selected_units"].loc[0, "passes_dark_firing_rate"] = False
    with pytest.raises(ValueError, match="firing-rate flags"):
        module.validate_cv_pca_result(changed)

    changed = _copy_result(result)
    changed["dataset"]["within_cv_spectrum_positive"].data[0, 0] += 1.0
    changed["spectrum"] = module._spectrum_from_dataset(changed["dataset"])
    with pytest.raises(ValueError, match="within-condition metric"):
        module.validate_cv_pca_result(changed)


def test_bundle_round_trip_is_compact_immutable_and_checksummed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _compute(monkeypatch)
    written = module.write_cv_pca_artifact(result, artifact_root=tmp_path)
    directory = written["artifact_paths"]["artifact_dir"]
    assert written["manifest"]["artifact_key"].tolist() == [
        "cv_pca",
        "summary",
        "within_spectrum",
        "selected_units",
        "lap_assignments",
        "trajectory_qc",
    ]
    assert directory.is_dir()
    assert not any(name in written["dataset"] for name in module.PROHIBITED_RESULT_VARIABLES)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module.write_cv_pca_artifact(result, artifact_root=tmp_path)

    manifest_path = directory / module.MANIFEST_FILENAME
    stale_manifest = written["manifest"].copy()
    stale_manifest["bundle_schema_version"] = "stale"
    stale_manifest.to_parquet(manifest_path, index=False)
    with pytest.raises(ValueError, match="bundle schema version"):
        module.load_cv_pca_artifact(directory)
    written["manifest"].to_parquet(manifest_path, index=False)

    with (directory / module.SUMMARY_FILENAME).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_cv_pca_artifact(directory)


def test_final_reload_failure_removes_new_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed post-rename validation must not leave a new UUID bundle."""
    result = _compute(monkeypatch)
    paths = module.get_cv_pca_artifact_paths(
        animal_name=result["animal_name"],
        date=result["date"],
        light_epoch=result["light_epoch"],
        dark_epoch=result["dark_epoch"],
        region=result["region"],
        cv_pca_id=result["cv_pca_id"],
        artifact_root=tmp_path,
    )
    original_load = module.load_cv_pca_artifact

    def fail_final_load(path, *, _allow_temporary_name=False):
        if _allow_temporary_name:
            return original_load(path, _allow_temporary_name=True)
        raise OSError("post-rename validation failed")

    monkeypatch.setattr(module, "load_cv_pca_artifact", fail_final_load)
    with pytest.raises(OSError, match="post-rename validation failed"):
        module.write_cv_pca_artifact(result, artifact_root=tmp_path)
    assert not paths["artifact_dir"].exists()


def _legacy_summary(dataset: object) -> pd.DataFrame:
    """Build the four-row legacy projection summary used by registration."""
    rows = []
    attrs = dataset.attrs
    for source in module.CONDITIONS:
        for target in module.CONDITIONS:
            rows.append(
                {
                    "random_seed": int(attrs["random_seed"]),
                    "animal_name": str(attrs["animal_name"]),
                    "date": str(attrs["date"]),
                    "region": str(attrs["region"]),
                    "dark_epoch": str(attrs["dark_epoch"]),
                    "light_epoch": str(attrs["light_epoch"]),
                    "unit_filter_mode": str(attrs["unit_filter_mode"]),
                    "normalization": str(attrs["normalization"]),
                    "source_condition": source,
                    "target_condition": target,
                    "projection_direction": f"{source}_to_{target}",
                    "is_cross_condition": source != target,
                    "n_units": int(attrs["n_units"]),
                    "source_cv_participation_ratio": float(
                        dataset["within_cv_participation_ratio"]
                        .sel(within_condition=source)
                        .values
                    ),
                    "source_n_components_80": float(
                        dataset["within_cv_n_components_80"]
                        .sel(within_condition=source)
                        .values
                    ),
                    "source_n_components_90": float(
                        dataset["within_cv_n_components_90"]
                        .sel(within_condition=source)
                        .values
                    ),
                    "min_firing_rate_hz": float(attrs["min_firing_rate_hz"]),
                    "min_condition_sd_hz": float(attrs["min_condition_sd_hz"]),
                    "bin_size_cm": float(attrs["bin_size_cm"]),
                    "n_groups": int(attrs["n_groups"]),
                    "min_occupancy_s": float(attrs["min_occupancy_s"]),
                }
            )
    return pd.DataFrame(rows)


def test_register_existing_recomputes_compares_and_compacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _compute(monkeypatch)
    legacy_result = tmp_path / "legacy.nc"
    legacy_summary = tmp_path / "legacy_summary.parquet"
    result["dataset"].to_netcdf(legacy_result)
    _legacy_summary(result["dataset"]).to_parquet(legacy_summary, index=False)

    registered = module.register_existing_cv_pca_artifact(
        legacy_result_path=legacy_result,
        legacy_summary_path=legacy_summary,
        compute_inputs=_inputs(),
        artifact_root=tmp_path / "canonical",
    )
    assert registered["artifact_origin"] == "registered_existing"
    assert registered["legacy_artifact_provenance"]["verification"] == (
        "exact_nwb_recomputation"
    )
    assert registered["legacy_artifact_provenance"][
        "excluded_legacy_variables"
    ] == list(module.PROHIBITED_RESULT_VARIABLES)
    for prohibited in module.PROHIBITED_RESULT_VARIABLES:
        assert prohibited not in registered["dataset"]


def test_register_existing_rejects_incomplete_corrupt_or_different_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _compute(monkeypatch)
    legacy_result = tmp_path / "legacy.nc"
    legacy_summary = tmp_path / "legacy_summary.parquet"
    result["dataset"].to_netcdf(legacy_result)
    summary = _legacy_summary(result["dataset"])
    summary.to_parquet(legacy_summary, index=False)
    common = {
        "legacy_result_path": legacy_result,
        "legacy_summary_path": legacy_summary,
        "compute_inputs": _inputs(),
        "artifact_root": tmp_path / "canonical",
    }

    summary.iloc[:-1].to_parquet(legacy_summary, index=False)
    with pytest.raises(ValueError, match="summary is incomplete"):
        module.register_existing_cv_pca_artifact(**common)
    summary.to_parquet(legacy_summary, index=False)

    import xarray as xr

    with xr.open_dataset(legacy_result) as opened:
        changed = opened.load()
    changed["within_cv_spectrum_signed"].data[0, 0] += 0.25
    changed.to_netcdf(legacy_result)
    changed_summary = _legacy_summary(changed)
    changed_summary.to_parquet(legacy_summary, index=False)
    with pytest.raises(ValueError, match="differs from exact NWB recomputation"):
        module.register_existing_cv_pca_artifact(**common)

    legacy_result.write_bytes(b"not a netcdf")
    with pytest.raises(ValueError, match="unreadable or incomplete"):
        module.register_existing_cv_pca_artifact(**common)


def test_register_existing_rejects_stale_metadata_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = _compute(monkeypatch)
    legacy_result = tmp_path / "legacy.nc"
    legacy_summary = tmp_path / "legacy_summary.parquet"
    summary = _legacy_summary(result["dataset"])
    summary.to_parquet(legacy_summary, index=False)
    common = {
        "legacy_result_path": legacy_result,
        "legacy_summary_path": legacy_summary,
        "compute_inputs": _inputs(),
        "artifact_root": tmp_path / "canonical",
    }

    changed = result["dataset"].copy(deep=True)
    changed.attrs["random_seed"] = 999
    changed.to_netcdf(legacy_result)
    _legacy_summary(changed).to_parquet(legacy_summary, index=False)
    with pytest.raises(ValueError, match="metadata 'random_seed' differs"):
        module.register_existing_cv_pca_artifact(**common)

    changed = result["dataset"].copy(deep=True)
    variable = "score_covariance_by_component"
    changed[variable] = changed[variable].transpose(
        "target_condition",
        "source_condition",
        "source_fold",
        "target_group",
        "component",
    )
    changed.to_netcdf(legacy_result)
    summary.to_parquet(legacy_summary, index=False)
    with pytest.raises(ValueError, match="dimensions differ"):
        module.register_existing_cv_pca_artifact(**common)
