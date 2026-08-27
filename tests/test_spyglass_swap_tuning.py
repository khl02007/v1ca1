"""Tests for database-free empirical swap-tuning artifact handling."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import swap_tuning as module


RESULT_ID = uuid.UUID("905f516e-b982-57cf-8423-ea6d2c8cddf7")


def _audit() -> pd.DataFrame:
    """Return one eligible and one excluded source-unit audit."""
    rows = []
    for index, (unit_id, eligible) in enumerate((("1", True), ("2", False))):
        rows.append(
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": unit_id,
                "stable_unit_id": f"merge-a:{unit_id}",
                "group_unit_id": str(100 + index),
                "source_unit_index": index,
                "eligible_unit_index": 0 if eligible else -1,
                "dark_movement_firing_rate_hz": 1.0 if eligible else 0.1,
                "light_train_movement_firing_rate_hz": 1.5,
                "light_test_movement_firing_rate_hz": 2.0,
                "passes_dark_firing_rate_threshold": eligible,
                "passes_light_firing_rate_threshold": True,
                "eligible_for_comparison": eligible,
                "test_light_spike_count": 0.0,
                "n_finite_scores": 0,
                "n_expected_scores": len(module.MODEL_NAMES)
                * len(TRAJECTORY_TYPES),
                "valid_swap_tuning_score": False,
                "unit_qc_status": (
                    "not_computed" if eligible else "excluded_dark_firing_rate"
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=module.SELECTED_UNIT_COLUMNS)


def _arrays() -> dict[str, object]:
    """Return one-unit complete empirical-scoring payload."""
    n_models = len(module.MODEL_NAMES)
    n_trajectories = len(TRAJECTORY_TYPES)
    n_bins = 2
    n_units = 1
    spike_sum = np.full((n_trajectories, n_units), 2.0)
    duration = np.full(n_trajectories, 1.0)
    ll_sum = np.full((n_models, n_trajectories, n_units), -1.0)
    model_tuning = np.full(
        (n_models, n_trajectories, n_bins, n_units),
        2.0,
    )
    model_tuning[1] = 1.0
    return {
        "unit_ids": np.asarray([100]),
        "model_names": np.asarray(module.MODEL_NAMES, dtype=str),
        "bin_edges": np.asarray([0.0, 0.5, 1.0]),
        "bin_centers": np.asarray([0.25, 0.75]),
        "segment_edges": np.asarray([0.0, 0.3, 0.7, 1.0]),
        "swap_source_trajectory": np.asarray(
            [
                module.SWAP_CONFIGURATION[name]["source_trajectory"]
                for name in TRAJECTORY_TYPES
            ]
        ),
        "swap_segment_index": np.asarray(
            [module.SWAP_CONFIGURATION[name]["segment_index"] for name in TRAJECTORY_TYPES]
        ),
        "segment_bin_mask": np.asarray(
            [[False, True], [True, False], [False, True], [True, False]]
        ),
        "model_tuning": model_tuning,
        "same_dark_tuning": np.full((n_trajectories, n_bins, n_units), 1.0),
        "other_dark_tuning": np.full((n_trajectories, n_bins, n_units), 1.0),
        "other_light_tuning": np.full((n_trajectories, n_bins, n_units), 2.0),
        "test_light_tuning": np.full((n_trajectories, n_bins, n_units), 2.5),
        "train_dark_same_rate_hz": np.full((n_trajectories, n_units), 1.0),
        "train_dark_other_rate_hz": np.full((n_trajectories, n_units), 1.0),
        "train_light_other_rate_hz": np.full((n_trajectories, n_units), 2.0),
        "test_light_target_rate_hz": np.full((n_trajectories, n_units), 2.5),
        "test_light_spike_sum": spike_sum,
        "test_light_bin_count": np.full(n_trajectories, 20.0),
        "test_light_duration_s": duration,
        "metrics": {
            "ll_sum": ll_sum,
            "ll_bits_per_spike": ll_sum / (np.log(2.0) * spike_sum[None]),
            "ll_bits_per_s": ll_sum / (np.log(2.0) * duration[None, :, None]),
        },
    }


def _patch_core_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch data loading while retaining real dataset/QC/artifact code."""
    from v1ca1.spyglass import dark_light_glm

    monkeypatch.setattr(
        dark_light_glm,
        "derive_graph_geometry",
        lambda _graphs: (8.0, np.asarray([0.0, 0.3, 0.7, 1.0])),
    )
    identity = _audit().loc[:, list(module.IDENTITY_COLUMNS)]
    fake_curve = SimpleNamespace(
        attrs={
            "analysis_status": "valid",
            "bin_edges_cm_json": "[0.0, 4.0, 8.0]",
        }
    )
    curves = {
        role: {trajectory: fake_curve for trajectory in TRAJECTORY_TYPES}
        for role in ("dark", "light_train", "light_test")
    }
    hashes = {
        role: {trajectory: "a" * 64 for trajectory in TRAJECTORY_TYPES}
        for role in curves
    }
    monkeypatch.setattr(
        module,
        "_load_and_validate_tuning_inputs",
        lambda **_kwargs: (curves, identity, hashes),
    )
    audit = _audit()
    monkeypatch.setattr(
        module,
        "_align_unit_inputs",
        lambda **_kwargs: (
            audit.copy(),
            np.asarray([True, False]),
            np.asarray([100], dtype=object),
            "valid",
        ),
    )
    monkeypatch.setattr(
        module,
        "_validate_movement_table_context",
        lambda table, **_kwargs: table.copy(),
    )
    monkeypatch.setattr(module, "_speed_threshold_from_tables", lambda _tables: 4.0)
    monkeypatch.setattr(module, "_has_valid_position_samples", lambda _position: True)
    monkeypatch.setattr(module, "_interval_duration_s", lambda _interval: 1.0)
    monkeypatch.setattr(
        module,
        "_derive_task_progression",
        lambda **_kwargs: {name: object() for name in TRAJECTORY_TYPES},
    )
    monkeypatch.setattr(
        module,
        "_compute_empirical_arrays",
        lambda **_kwargs: (_arrays(), None),
    )


def _compute(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Return one fully validated result through the public compute entry point."""
    _patch_core_inputs(monkeypatch)
    roles = ("dark", "light_train", "light_test")
    return module.compute_swap_tuning_curve_comparison(
        swap_tuning_curve_comparison_id=RESULT_ID,
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
        tuning_curve_artifact_paths={
            role: {trajectory: Path(f"/{role}/{trajectory}.nc") for trajectory in TRAJECTORY_TYPES}
            for role in roles
        },
        movement_firing_rate_tables_by_role={
            role: pd.DataFrame({"placeholder": [1]}) for role in roles
        },
        spikes=object(),
        stable_unit_ids=(),
        position=object(),
        position_offset_samples=10,
        movement_interval=object(),
        movement_analysis_status="valid",
        trajectory_intervals={name: object() for name in TRAJECTORY_TYPES},
        graph_inputs_by_trajectory={name: {} for name in TRAJECTORY_TYPES},
        movement_firing_rate_table_sha256_by_role={role: "b" * 64 for role in roles},
        movement_intervals_sha256_by_role={role: "c" * 64 for role in roles},
    )


def test_parameters_and_paths_are_explicit(tmp_path: Path) -> None:
    parameters = module.validate_swap_tuning_curve_comparison_parameters()
    assert parameters == {
        "evaluation_bin_size_s": 0.05,
        "gaussian_smoothing_sigma_bins": 1.0,
        "min_dark_firing_rate_hz": 0.5,
        "min_light_firing_rate_hz": 0.5,
    }
    with pytest.raises(ValueError, match="positive"):
        module.validate_swap_tuning_curve_comparison_parameters(
            evaluation_bin_size_s=0.0
        )
    paths = module.get_swap_tuning_curve_comparison_artifact_paths(
        animal_name="L14",
        date="20240611",
        region="v1",
        dark_epoch="08_r4",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
        swap_tuning_curve_comparison_id=RESULT_ID,
        artifact_root=tmp_path,
    )
    assert paths["artifact_dir"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "swap_tuning_curve_comparison"
        / "02_r1_train_to_06_r3_test"
        / "dark_08_r4"
        / "v1"
        / str(RESULT_ID)
    )
    assert paths["summary_path"].name == "summary.parquet"


def test_compute_retains_all_unit_audit_and_scores_only_eligible(monkeypatch) -> None:
    result = _compute(monkeypatch)

    assert result["analysis_status"] == "valid"
    assert result["n_source_units"] == 2
    assert result["n_units"] == 1
    assert result["n_valid_units"] == 1
    assert result["selected_units"]["eligible_for_comparison"].tolist() == [
        True,
        False,
    ]
    assert result["dataset"].coords["unit"].values.tolist() == ["merge-a:1"]
    assert len(result["summary"]) == len(module.MODEL_NAMES) * len(TRAJECTORY_TYPES)
    assert set(result["summary"]["score_qc_status"]) == {"valid"}
    assert set(result["dataset"].data_vars) == set(module.SCIENTIFIC_VARIABLE_DIMS)


def test_bundle_roundtrip_and_checksum_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result = _compute(monkeypatch)
    destination = tmp_path / str(RESULT_ID)
    paths = module.write_swap_tuning_curve_comparison_artifact(
        result,
        destination,
    )
    loaded = module.load_swap_tuning_curve_comparison_artifact(destination)

    assert loaded["analysis_status"] == "valid"
    assert loaded["n_source_units"] == 2
    assert paths["result_path"].is_file()
    assert paths["summary_path"].is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        module.write_swap_tuning_curve_comparison_artifact(result, destination)

    paths["summary_path"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        module.load_swap_tuning_curve_comparison_artifact(destination)


def test_output_rule_preserves_legacy_scientific_contract() -> None:
    assert module.MODEL_NAMES == (
        "empirical_visual",
        "empirical_dark",
        "empirical_pointwise_multiplicative_ratio",
        "empirical_segment_multiplicative_ratio",
        "empirical_pointwise_additive_delta",
        "empirical_segment_additive_delta",
    )
    assert module.OUTPUT_RULE["heldout_firing_rate_filter"] is False
    assert module.OUTPUT_RULE["eligibility_policy"].startswith("strict_dark")
    assert module.REQUIRED_UPSTREAM_BIN_SIZE_CM == 4.0
    assert module.EMPIRICAL_EPSILON == 1e-10


def _legacy_outputs(result: dict[str, object], tmp_path: Path) -> tuple[Path, Path]:
    """Write schema-2 views of one canonical result for registration tests."""
    dataset = result["dataset"].drop_vars("light_test_movement_firing_rate_hz")
    dataset = dataset.assign_coords(unit=np.asarray([7]))
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs["schema_version"] = module.LEGACY_RESULT_SCHEMA_VERSION
    result_path = tmp_path / "legacy.nc"
    dataset.to_netcdf(result_path)

    canonical = result["summary"]
    legacy = pd.DataFrame(
        {
            "animal_name": canonical["animal_name"],
            "date": canonical["date"],
            "region": canonical["region"],
            "dark_train_epoch": canonical["dark_train_epoch"],
            "light_train_epoch": canonical["light_train_epoch"],
            "light_test_epoch": canonical["light_test_epoch"],
            "trajectory": canonical["trajectory"],
            "unit": 7,
            "apply_fr_filter": True,
            "min_dark_fr_hz": canonical["min_dark_firing_rate_hz"],
            "min_light_fr_hz": canonical["min_light_firing_rate_hz"],
            **{
                name: canonical[name]
                for name in module.LEGACY_SUMMARY_COLUMNS[11:24]
            },
            "model": canonical["model"],
            "ll_sum": canonical["ll_sum"],
            "ll_bits_per_spike": canonical["ll_bits_per_spike"],
            "ll_bits_per_s": canonical["ll_bits_per_s"],
        },
        columns=module.LEGACY_SUMMARY_COLUMNS,
    )
    summary_path = tmp_path / "legacy.parquet"
    legacy.to_parquet(summary_path, index=False)
    return result_path, summary_path


def _registration_kwargs(
    *,
    result_path: Path,
    summary_path: Path,
    destination: Path,
) -> dict[str, object]:
    """Return inert selected-input arguments for a patched re-score."""
    roles = ("dark", "light_train", "light_test")
    return {
        "source_result_path": result_path,
        "source_summary_path": summary_path,
        "destination_path": destination,
        "unit_identity_resolver": {
            7: {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "1",
                "group_unit_id": 100,
                "sorting_unit_id": 7,
            }
        },
        "source_sorting_type": "ImportedSpikeSorting",
        "swap_tuning_curve_comparison_id": RESULT_ID,
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "dark_epoch": "08_r4",
        "light_train_epoch": "02_r1",
        "light_test_epoch": "06_r3",
        "tuning_curve_artifact_paths": {
            role: {
                trajectory: Path(f"/{role}/{trajectory}.nc")
                for trajectory in TRAJECTORY_TYPES
            }
            for role in roles
        },
        "movement_firing_rate_tables_by_role": {
            role: pd.DataFrame() for role in roles
        },
        "spikes": object(),
        "stable_unit_ids": (),
        "position": object(),
        "position_offset_samples": 10,
        "movement_interval": object(),
        "movement_analysis_status": "valid",
        "trajectory_intervals": {
            trajectory: object() for trajectory in TRAJECTORY_TYPES
        },
        "graph_inputs_by_trajectory": {
            trajectory: {} for trajectory in TRAJECTORY_TYPES
        },
        "movement_intervals_sha256_by_role": {
            role: "c" * 64 for role in roles
        },
        "source_spyglass_git_commit": "spyglass-source",
    }


def test_registration_rebuilds_and_requires_complete_exact_legacy_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recomputed = _compute(monkeypatch)
    result_path, summary_path = _legacy_outputs(recomputed, tmp_path)
    monkeypatch.setattr(
        module,
        "compute_swap_tuning_curve_comparison",
        lambda **_kwargs: recomputed,
    )
    destination = tmp_path / "registered" / str(RESULT_ID)
    kwargs = _registration_kwargs(
        result_path=result_path,
        summary_path=summary_path,
        destination=destination,
    )

    registered = module.register_existing_swap_tuning_curve_comparison_artifact(
        **kwargs
    )

    assert registered["artifact_origin"] == "registered_existing"
    assert registered["legacy_artifact_provenance"]["source_sorting_type"] == (
        "ImportedSpikeSorting"
    )
    assert registered["legacy_artifact_provenance"][
        "source_spyglass_git_commit"
    ] == "spyglass-source"
    assert destination.is_dir()

    source = module._load_dataset(result_path)
    source["ll_sum"].values[0, 0, 0] += 0.01
    tampered_path = tmp_path / "tampered.nc"
    source.to_netcdf(tampered_path)
    tampered_kwargs = {
        **kwargs,
        "source_result_path": tampered_path,
        "destination_path": tmp_path / "tampered" / str(RESULT_ID),
    }
    with pytest.raises(ValueError, match="differs from NWB re-score"):
        module.register_existing_swap_tuning_curve_comparison_artifact(
            **tampered_kwargs
        )


def test_registration_rejects_nonimported_sorting_before_rescore(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recomputed = _compute(monkeypatch)
    result_path, summary_path = _legacy_outputs(recomputed, tmp_path)
    kwargs = _registration_kwargs(
        result_path=result_path,
        summary_path=summary_path,
        destination=tmp_path / str(RESULT_ID),
    )
    kwargs["source_sorting_type"] = "CurationV1"
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        module.register_existing_swap_tuning_curve_comparison_artifact(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda dataset: dataset["swap_source_trajectory"].values.__setitem__(
                0, "left_to_center"
            ),
            "source trajectories",
        ),
        (
            lambda dataset: dataset["segment_bin_mask"].values.__setitem__(
                (0, 0), True
            ),
            "segment-bin masks",
        ),
        (
            lambda dataset: dataset["test_light_duration_s"].values.__setitem__(
                0, 1.1
            ),
            "bin counts and durations",
        ),
        (
            lambda dataset: dataset["same_dark_train_tuning_hz"].values.__setitem__(
                (0, 0, 0), -1.0
            ),
            "finite/nonnegative",
        ),
        (
            lambda dataset: dataset["model_tuning_hz"].values.__setitem__(
                (0, 0, 0, 0), 3.0
            ),
            "model tuning formula",
        ),
    ),
)
def test_dataset_validation_rejects_scientific_tampering(
    monkeypatch,
    mutation,
    error: str,
) -> None:
    result = _compute(monkeypatch)
    tampered = {**result, "dataset": result["dataset"].copy(deep=True)}
    mutation(tampered["dataset"])

    with pytest.raises(ValueError, match=error):
        module.validate_swap_tuning_curve_comparison_result(tampered)


def test_unit_eligibility_qc_and_result_status_are_derived(monkeypatch) -> None:
    result = _compute(monkeypatch)
    stale_eligibility = {
        **result,
        "selected_units": result["selected_units"].copy(),
    }
    stale_eligibility["selected_units"].loc[0, "eligible_for_comparison"] = False
    stale_eligibility["selected_units"].loc[0, "eligible_unit_index"] = -1
    with pytest.raises(ValueError, match="eligibility must equal"):
        module.validate_swap_tuning_curve_comparison_result(stale_eligibility)

    stale_qc = {**result, "selected_units": result["selected_units"].copy()}
    stale_qc["selected_units"].loc[0, "unit_qc_status"] = "not_computed"
    with pytest.raises(ValueError, match="unit_qc_status"):
        module.validate_swap_tuning_curve_comparison_result(stale_qc)

    zero_valid = {
        **result,
        "selected_units": result["selected_units"].copy(),
        "dataset": result["dataset"].copy(deep=True),
    }
    zero_valid["dataset"]["test_light_spike_sum"].values[:] = 0.0
    zero_valid["dataset"]["ll_bits_per_spike"].values[:] = np.nan
    zero_valid["selected_units"].loc[0, "test_light_spike_count"] = 0.0
    zero_valid["selected_units"].loc[0, "n_finite_scores"] = 0
    zero_valid["selected_units"].loc[0, "valid_swap_tuning_score"] = False
    zero_valid["selected_units"].loc[0, "unit_qc_status"] = "zero_test_spikes"
    zero_valid["summary"] = module._build_summary(
        metadata=zero_valid["metadata"],
        parameters=zero_valid["parameters"],
        selected_units=zero_valid["selected_units"],
        dataset=zero_valid["dataset"],
    )
    with pytest.raises(ValueError, match="analysis_status"):
        module.validate_swap_tuning_curve_comparison_result(zero_valid)


def test_native_tsgroup_keys_remain_distinct_from_persistent_ids(monkeypatch) -> None:
    from v1ca1.spyglass import movement

    spikes = {101: object(), 202: object()}
    stable_ids = (
        {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        {"spikesorting_merge_id": "merge-b", "unit_id": 22},
    )
    curve_identity = pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge-a", "merge-b"],
            "unit_id": ["11", "22"],
            "stable_unit_id": ["merge-a:11", "merge-b:22"],
            "group_unit_id": ["101", "202"],
        }
    )
    role_rates = {"dark": [1.0, 0.1], "light_train": [1.0, 1.0], "light_test": [2.0, 3.0]}
    monkeypatch.setattr(
        movement,
        "align_movement_firing_rates",
        lambda table, **_kwargs: pd.Series(
            role_rates[str(table.attrs["role"])],
            index=[101, 202],
        ),
    )
    tables = {}
    for role in role_rates:
        table = pd.DataFrame(
            {"firing_rate_status": ["valid"]}
        )
        table.attrs["role"] = role
        tables[role] = table

    audit, eligible, native_keys, status = module._align_unit_inputs(
        spikes=spikes,
        stable_unit_ids=stable_ids,
        curve_identity=curve_identity,
        movement_tables_by_role=tables,
        parameters={
            "min_dark_firing_rate_hz": 0.5,
            "min_light_firing_rate_hz": 0.5,
        },
    )

    assert status == "valid"
    assert eligible.tolist() == [True, False]
    assert native_keys.tolist() == [101]
    assert audit.loc[0, "stable_unit_id"] == "merge-a:11"
    assert audit.loc[0, "group_unit_id"] == "101"


def test_upstream_curve_nan_interpolation_and_all_nan_fallback() -> None:
    xr = pytest.importorskip("xarray")
    curve = xr.DataArray(
        np.asarray([[np.nan, 2.0, np.nan], [np.nan, np.nan, np.nan]]),
        dims=("unit", "linear_position_cm"),
        coords={
            "unit": ["a", "b"],
            "linear_position_cm": [1.0, 2.0, 3.0],
            "spike_count": ("unit", [6, 8]),
        },
        attrs={"support_duration_s": 2.0},
    )
    curves = {
        role: {trajectory: curve for trajectory in TRAJECTORY_TYPES}
        for role in ("dark", "light_train", "light_test")
    }

    tunings, rates = module._prepare_tuning_inputs(
        curves,
        eligible=np.asarray([True, True]),
        sigma_bins=0.0,
    )

    np.testing.assert_allclose(
        tunings["dark"]["center_to_left"],
        np.asarray([[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]),
    )
    np.testing.assert_allclose(
        rates["dark"]["center_to_left"],
        [3.0, 4.0],
    )


def test_fixed_empirical_formulas_and_real_poisson_scoring() -> None:
    analysis = module._analysis_module()
    same_dark = np.asarray([[1.0], [2.0], [3.0]])
    other_dark = np.asarray([[2.0], [2.0], [2.0]])
    other_light = np.asarray([[4.0], [6.0], [8.0]])
    centers = np.asarray([0.1, 0.5, 0.9])
    edges = np.asarray([0.0, 0.3, 0.7, 1.0])
    predictions = analysis.build_empirical_swap_tunings(
        same_dark,
        other_light,
        other_dark,
        centers,
        edges,
        2,
        epsilon=module.EMPIRICAL_EPSILON,
    )
    np.testing.assert_allclose(predictions["empirical_visual"], other_light)
    np.testing.assert_allclose(predictions["empirical_dark"], same_dark)
    np.testing.assert_allclose(
        predictions["empirical_pointwise_multiplicative_ratio"],
        same_dark * other_light / other_dark,
    )
    np.testing.assert_allclose(
        predictions["empirical_segment_additive_delta"],
        same_dark + 6.0,
    )

    score = analysis.score_segment_binned_counts(
        spike_counts=np.asarray([[1.0], [0.0], [2.0]]),
        positions=centers,
        tunings_by_model=predictions,
        bin_edges=np.asarray([0.0, 0.3, 0.7, 1.0]),
        segment_edges=edges,
        segment_index=2,
        bin_size_s=0.05,
    )
    assert score["test_light_bin_count"] == 1.0
    assert score["test_light_spike_sum"].tolist() == [2.0]
    assert np.all(np.isfinite(score["ll_sum"]))
    np.testing.assert_allclose(
        score["ll_bits_per_s"],
        score["ll_sum"] / (np.log(2.0) * 0.05),
    )


def _terminal_result_from_valid(
    result: dict[str, object],
    status: str,
) -> dict[str, object]:
    """Return one internally consistent terminal view for status tests."""
    selected = result["selected_units"].copy()
    selected["test_light_spike_count"] = 0.0
    selected["n_finite_scores"] = 0
    selected["valid_swap_tuning_score"] = False
    selected.loc[selected["eligible_for_comparison"], "unit_qc_status"] = (
        "not_computed"
    )
    if status == "no_units":
        selected = pd.DataFrame(columns=module.SELECTED_UNIT_COLUMNS)
    elif status == "no_eligible_units":
        selected.loc[0, "dark_movement_firing_rate_hz"] = 0.1
        selected.loc[0, "passes_dark_firing_rate_threshold"] = False
        selected.loc[0, "eligible_for_comparison"] = False
        selected.loc[0, "eligible_unit_index"] = -1
        selected.loc[0, "unit_qc_status"] = "excluded_dark_firing_rate"
    elif status in {"no_valid_position", "no_movement"}:
        selected.loc[0, "eligible_for_comparison"] = False
        selected.loc[0, "eligible_unit_index"] = -1
        selected.loc[0, "unit_qc_status"] = "not_computed"
    upstream = dict(result["upstream_provenance"])
    upstream["selected_units_sha256"] = module._selected_units_sha256(selected)
    detail = {}
    if status == "upstream_terminal":
        detail["upstream_tuning_statuses_json"] = '["no_trials"]'
    elif status == "no_trajectory_samples":
        detail["missing_trajectory"] = "center_to_left"
    dataset = module._terminal_dataset(
        metadata=result["metadata"],
        parameters=result["parameters"],
        upstream_provenance=upstream,
        selected_units=selected,
        segment_edges=np.asarray([0.0, 0.3, 0.7, 1.0]),
        analysis_status=status,
        terminal_detail=detail,
    )
    return {
        "metadata": result["metadata"],
        "parameters": result["parameters"],
        "upstream_provenance": upstream,
        "selected_units": selected,
        "summary": module.empty_swap_tuning_summary(),
        "dataset": dataset,
        "analysis_status": status,
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


@pytest.mark.parametrize(
    "status",
    (
        "no_units",
        "no_eligible_units",
        "no_valid_position",
        "no_movement",
        "upstream_terminal",
        "no_trajectory_samples",
    ),
)
def test_all_terminal_markers_validate(monkeypatch, status: str) -> None:
    valid = _compute(monkeypatch)
    terminal = module.validate_swap_tuning_curve_comparison_result(
        _terminal_result_from_valid(valid, status)
    )

    assert terminal["analysis_status"] == status
    assert terminal["n_valid_units"] == 0


@pytest.mark.parametrize("terminal_status", (None, "no_trajectory_samples"))
def test_analysis_nwb_objects_roundtrip_valid_and_terminal_results(
    monkeypatch,
    tmp_path: Path,
    terminal_status: str | None,
) -> None:
    pynwb = pytest.importorskip("pynwb")
    result = _compute(monkeypatch)
    if terminal_status is not None:
        result = module.validate_swap_tuning_curve_comparison_result(
            _terminal_result_from_valid(result, terminal_status)
        )
    expected_hashes = module.swap_tuning_curve_comparison_nwb_hashes(result)
    objects = module.swap_tuning_curve_comparison_result_to_nwb_objects(result)
    nwb = pynwb.NWBFile(
        session_description="swap tuning NWB test",
        identifier=f"swap-{terminal_status or 'valid'}",
        session_start_time=datetime.now(timezone.utc),
    )
    object_ids = {}
    for name, obj in objects.items():
        nwb.add_scratch(obj)
        object_ids[name] = str(obj.object_id)
    path = tmp_path / f"swap-{terminal_status or 'valid'}.nwb"
    with pynwb.NWBHDF5IO(path, mode="w") as io:
        io.write(nwb)
    assert pynwb.validate(path=path) == []
    with pynwb.NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        stored = io.read()
        loaded = module.swap_tuning_curve_comparison_result_from_nwb_objects(
            **{
                name: stored.objects[object_id]
                for name, object_id in object_ids.items()
            }
        )
        assert module.swap_tuning_curve_comparison_nwb_hashes(loaded) == (
            expected_hashes
        )
    assert loaded["analysis_status"] == result["analysis_status"]


def test_ragged_frame_hash_canonicalizes_scalar_nan_payloads() -> None:
    """Equivalent scalar NaNs retain one hash across NWB/HDF5 round-trips."""
    canonical_nan = np.asarray([0x7FF8000000000000], dtype=np.uint64).view(
        np.float64
    )[0]
    alternate_nan = np.asarray([0x7FF8000000000001], dtype=np.uint64).view(
        np.float64
    )[0]
    first = pd.DataFrame(
        {"name": ["row"], "value": [canonical_nan], "vector": [[1.0]]}
    )
    second = pd.DataFrame(
        {"name": ["row"], "value": [alternate_nan], "vector": [[1.0]]}
    )

    assert module._ragged_frame_sha256(
        first,
        vector_columns=("vector",),
    ) == module._ragged_frame_sha256(second, vector_columns=("vector",))


def test_geometry_loader_preserves_float_text_bits(monkeypatch) -> None:
    """Duration text uses Python's exact finite-float round-trip parser."""
    geometry = module._geometry_frame(_compute(monkeypatch))
    expected = 3 * 0.05
    geometry["test_light_duration_s"] = geometry[
        "test_light_duration_s"
    ].map(str)
    geometry.loc[0, "test_light_duration_s"] = str(expected)

    loaded = module.swap_tuning_geometry_from_dynamic_table(geometry)

    assert np.asarray(
        [loaded.loc[0, "test_light_duration_s"]], dtype=np.float64
    ).view(np.uint64)[0] == np.asarray([expected], dtype=np.float64).view(
        np.uint64
    )[0]


def test_per_unit_score_failure_is_isolated(monkeypatch) -> None:
    result = _compute(monkeypatch)
    audit = result["selected_units"].copy()
    audit.loc[0, ["test_light_spike_count", "n_finite_scores"]] = [0.0, 0]
    audit.loc[0, "valid_swap_tuning_score"] = False
    audit.loc[0, "unit_qc_status"] = "not_computed"
    dataset = result["dataset"].copy(deep=True)
    dataset["ll_bits_per_spike"].values[0, 0, 0] = np.nan

    audited, status = module._audit_unit_scores(audit, dataset)

    assert status == "no_valid_units"
    assert audited.loc[0, "n_finite_scores"] == 23
    assert audited.loc[0, "unit_qc_status"] == "partial_nonfinite_score"


def test_movement_interval_hash_provenance_is_required_and_validated(monkeypatch) -> None:
    result = _compute(monkeypatch)
    assert result["upstream_provenance"]["movement_intervals_sha256_by_role"] == {
        "dark": "c" * 64,
        "light_train": "c" * 64,
        "light_test": "c" * 64,
    }
    tampered = {**result, "upstream_provenance": dict(result["upstream_provenance"])}
    tampered["upstream_provenance"]["movement_intervals_sha256_by_role"] = {
        "dark": "not-a-hash",
        "light_train": "c" * 64,
        "light_test": "c" * 64,
    }
    with pytest.raises(ValueError, match="SHA-256"):
        module.validate_swap_tuning_curve_comparison_result(tampered)


@pytest.mark.parametrize("tamper_kind", ("value", "row_order"))
def test_registration_rejects_legacy_summary_tampering(
    monkeypatch,
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    recomputed = _compute(monkeypatch)
    result_path, summary_path = _legacy_outputs(recomputed, tmp_path)
    summary = pd.read_parquet(summary_path)
    if tamper_kind == "value":
        summary.loc[0, "ll_sum"] += 0.01
    else:
        summary = summary.iloc[::-1].reset_index(drop=True)
    summary.to_parquet(summary_path, index=False)
    monkeypatch.setattr(
        module,
        "compute_swap_tuning_curve_comparison",
        lambda **_kwargs: recomputed,
    )
    kwargs = _registration_kwargs(
        result_path=result_path,
        summary_path=summary_path,
        destination=tmp_path / "registered" / str(RESULT_ID),
    )

    with pytest.raises(ValueError, match="summary column"):
        module.register_existing_swap_tuning_curve_comparison_artifact(**kwargs)


def test_registration_rejects_schema_attribute_and_preprocessing_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    recomputed = _compute(monkeypatch)
    result_path, summary_path = _legacy_outputs(recomputed, tmp_path)
    source = module._load_dataset(result_path)
    source.attrs["schema_version"] = "3"
    bad_schema = tmp_path / "bad_schema.nc"
    source.to_netcdf(bad_schema)
    monkeypatch.setattr(
        module,
        "compute_swap_tuning_curve_comparison",
        lambda **_kwargs: recomputed,
    )
    kwargs = _registration_kwargs(
        result_path=bad_schema,
        summary_path=summary_path,
        destination=tmp_path / "registered" / str(RESULT_ID),
    )
    with pytest.raises(ValueError, match="schema version 2"):
        module.register_existing_swap_tuning_curve_comparison_artifact(**kwargs)

    offset_kwargs = {
        **kwargs,
        "source_result_path": result_path,
        "position_offset_samples": 9,
    }
    with pytest.raises(ValueError, match="position offset 10"):
        module.register_existing_swap_tuning_curve_comparison_artifact(
            **offset_kwargs
        )

    stale_speed = {
        **recomputed,
        "upstream_provenance": dict(recomputed["upstream_provenance"]),
    }
    stale_speed["upstream_provenance"]["speed_threshold_cm_s"] = 3.0
    monkeypatch.setattr(
        module,
        "compute_swap_tuning_curve_comparison",
        lambda **_kwargs: stale_speed,
    )
    speed_kwargs = {
        **kwargs,
        "source_result_path": result_path,
        "destination_path": tmp_path / "speed" / str(RESULT_ID),
    }
    with pytest.raises(ValueError, match="speed threshold 4"):
        module.register_existing_swap_tuning_curve_comparison_artifact(
            **speed_kwargs
        )
