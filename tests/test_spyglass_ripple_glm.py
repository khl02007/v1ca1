"""Tests for database-free ripple GLM artifacts."""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

import v1ca1.spyglass.ripple_glm as artifact_module
from v1ca1.ripple import ripple_glm as scientific_module
from v1ca1.spyglass.ripple_glm import (
    DEFAULT_N_SHUFFLES_RIPPLE,
    OUTPUT_RULE_SHA256,
    compute_ripple_glm,
    get_ripple_glm_artifact_paths,
    load_ripple_glm_artifact,
    prepare_ripple_glm_event_selection,
    register_existing_ripple_glm_artifact,
    validate_ripple_glm_parameters,
    validate_ripple_glm_result,
    write_ripple_glm_artifact,
)


class _SpikeTrain:
    """Minimal spike train exposing seconds through ``t``."""

    def __init__(self, times: list[float]):
        self.t = np.asarray(times, dtype=float)


class _Interval:
    """Minimal second-based epoch interval."""

    def __init__(self, start: float = 0.0, end: float = 10.0):
        self.start = np.asarray([start], dtype=float)
        self.end = np.asarray([end], dtype=float)


def _spike_times(starts: list[float], counts: list[int]) -> list[float]:
    return [start + 0.05 for start, count in zip(starts, counts, strict=True) for _ in range(count)]


@pytest.fixture
def inputs() -> dict[str, object]:
    starts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    source_spikes = {
        10: _SpikeTrain(_spike_times(starts, [0, 1, 2, 3, 1, 2])),
        11: _SpikeTrain(_spike_times(starts, [2, 1, 0, 1, 2, 3])),
    }
    target_spikes = {
        20: _SpikeTrain(_spike_times(starts, [1, 0, 2, 1, 3, 1])),
        21: _SpikeTrain(_spike_times(starts, [0, 2, 1, 3, 1, 2])),
        22: _SpikeTrain([]),
    }
    return {
        "ripple_glm_id": uuid.uuid5(uuid.NAMESPACE_URL, "ripple-glm-test"),
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "ripple_table": pd.DataFrame(
            {
                "epoch": ["08_r4"] * len(starts),
                "start_time": starts,
                "end_time": np.asarray(starts) + 0.08,
            }
        ),
        "epoch_interval": _Interval(),
        "source_spikes": source_spikes,
        "source_stable_unit_ids": [
            {"spikesorting_merge_id": "source-sort", "unit_id": str(value)}
            for value in source_spikes
        ],
        "target_spikes": target_spikes,
        "target_stable_unit_ids": [
            {"spikesorting_merge_id": "target-sort", "unit_id": str(value)}
            for value in target_spikes
        ],
        "upstream_provenance": {
            "source_region_sorted_spikes_group_id": "source-group",
            "target_region_sorted_spikes_group_id": "target-group",
            "ripples_id": "ripples",
            "detector_zscore_threshold": 2.0,
            "speed_gated": True,
        },
        "parameter_name": "test",
        "n_splits": 2,
        "n_shuffles_ripple": 2,
    }


def _fake_fit(
    epoch: str,
    *,
    prepared_epoch: dict[str, object],
    source_predictor_mode: str,
    n_shuffles_ripple: int,
    shuffle_seed: int,
    ripple_window_s: float,
    ripple_window_offset_s: float,
    source_window_s: float,
    source_window_offset_s: float,
    target_window_s: float,
    target_window_offset_s: float,
    ridge_strength: float,
    maxiter: int,
    tol: float,
) -> dict[str, object]:
    """Return scientifically self-consistent fit arrays without importing nemos."""
    del shuffle_seed, ridge_strength, maxiter, tol
    observed = np.asarray(prepared_epoch["y_r"], dtype=float)
    predicted = 0.35 + 0.72 * observed
    folds = list(prepared_epoch["cv_splits"])
    fold_index = np.full(len(observed), -1, dtype=int)
    metrics = {
        name: np.full((len(folds), observed.shape[1]), np.nan, dtype=float)
        for name in artifact_module.METRIC_NAMES
    }
    for index, (train, test) in enumerate(folds):
        fold_index[test] = index
        metrics["pseudo_r2"][index] = scientific_module.mcfadden_pseudo_r2_per_neuron(
            observed[test], predicted[test], observed[train]
        )
        metrics["mae"][index] = scientific_module.mae_per_neuron(
            observed[test], predicted[test]
        )
        metrics["devexp"][index] = scientific_module.deviance_explained_per_neuron(
            observed[test], predicted[test], observed[train]
        )
        metrics["bits_per_spike"][index] = scientific_module.bits_per_spike_per_neuron(
            observed[test], predicted[test], observed[train]
        )
    result: dict[str, object] = {
        "epoch": epoch,
        "shuffle_seed": 45,
        "min_spikes_per_ripple": np.nan,
        "min_ca1_spikes_per_ripple": np.nan,
        "n_splits": len(folds),
        "n_shuffles_ripple": n_shuffles_ripple,
        "ripple_window_s": target_window_s,
        "ripple_window_offset_s": target_window_offset_s,
        "source_window_s": source_window_s,
        "source_window_offset_s": source_window_offset_s,
        "target_window_s": target_window_s,
        "target_window_offset_s": target_window_offset_s,
        "windows_differ": not (
            source_window_s == target_window_s
            and source_window_offset_s == target_window_offset_s
        ),
        "source_predictor_mode": source_predictor_mode,
        "source_predictor_description": (
            scientific_module.SOURCE_PREDICTOR_MODE_DESCRIPTIONS[source_predictor_mode]
        ),
        "n_source_predictor_features": (
            len(prepared_epoch["ca1_unit_ids"])
            if source_predictor_mode == "unit_vector"
            else 1
        ),
        "n_ripples_before_window_bounds": prepared_epoch[
            "n_ripples_before_window_bounds"
        ],
        "n_ripples_removed_by_window_bounds": prepared_epoch[
            "n_ripples_removed_by_window_bounds"
        ],
        "n_ripples": len(observed),
        "n_cells": observed.shape[1],
        "n_ca1_cells": len(prepared_epoch["ca1_unit_ids"]),
        "v1_unit_ids": np.asarray(prepared_epoch["v1_unit_ids"]),
        "ca1_unit_ids": np.asarray(prepared_epoch["ca1_unit_ids"]),
        "ripple_start_time_s": np.asarray(prepared_epoch["ripple_start_times"]),
        "ripple_window_start_s": np.asarray(prepared_epoch["target_window_starts"]),
        "ripple_window_end_s": np.asarray(prepared_epoch["target_window_ends"]),
        "source_window_start_s": np.asarray(prepared_epoch["source_window_starts"]),
        "source_window_end_s": np.asarray(prepared_epoch["source_window_ends"]),
        "target_window_start_s": np.asarray(prepared_epoch["target_window_starts"]),
        "target_window_end_s": np.asarray(prepared_epoch["target_window_ends"]),
        "ripple_fold_index": fold_index,
        "ripple_observed_count_oof": observed,
        "ripple_predicted_count_oof": predicted,
        "coef_intercept_full_all": np.full(observed.shape[1], -0.25),
    }
    if source_predictor_mode == "unit_vector":
        source_counts = np.asarray(prepared_epoch["X_r"], dtype=float)
        keep = source_counts.std(axis=0) > 1e-6
        coefficient_ids = np.asarray(prepared_epoch["ca1_unit_ids"])[keep]
        names = np.asarray([f"ca1_unit_{value}" for value in coefficient_ids], dtype=str)
    else:
        coefficient_ids = np.asarray([-1], dtype=int)
        names = np.asarray(["mean_ca1_activity"], dtype=str)
    result["coef_ca1_unit_ids"] = coefficient_ids
    result["source_predictor_feature_names"] = names
    result["coef_ca1_full_all"] = np.full(
        (len(coefficient_ids), observed.shape[1]), 0.125
    )
    for metric, real in metrics.items():
        result[f"{metric}_ripple_folds"] = real
        offsets = np.linspace(-0.05, 0.05, n_shuffles_ripple, dtype=float)
        result[f"{metric}_ripple_shuff_folds"] = (
            real[:, None, :] + offsets[None, :, None]
        )
    return result


@pytest.fixture
def computed(inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setattr(scientific_module, "_fit_ripple_glm_on_prepared_epoch", _fake_fit)
    return compute_ripple_glm(**inputs)


def test_parameters_freeze_manuscript_ripple_input_policy() -> None:
    parameters = validate_ripple_glm_parameters()
    assert DEFAULT_N_SHUFFLES_RIPPLE == 100
    assert parameters["n_shuffles_ripple"] == 100
    assert parameters["expected_detector_zscore_threshold"] == 2.0
    assert parameters["require_speed_gated"] is True
    with pytest.raises(ValueError, match="threshold 2.0"):
        validate_ripple_glm_parameters(expected_detector_zscore_threshold=2.5)
    with pytest.raises(ValueError, match="speed-gated"):
        validate_ripple_glm_parameters(require_speed_gated=False)


def test_parameters_preserve_asymmetric_windows_and_modes() -> None:
    parameters = validate_ripple_glm_parameters(
        source_window_s=0.1,
        source_window_offset_s=-0.1,
        target_window_s=0.3,
        target_window_offset_s=0.2,
        ripple_selection_mode="single",
        source_predictor_mode="mean_activity",
    )
    assert parameters["source_target_windows_differ"] is True
    assert parameters["source_window_offset_s"] == -0.1
    assert parameters["target_window_offset_s"] == 0.2
    with pytest.raises(ValueError, match="ripple_selection_mode"):
        validate_ripple_glm_parameters(ripple_selection_mode="bad")


def test_artifact_paths_are_session_first_and_uuid_keyed(tmp_path: Path) -> None:
    result_id = uuid.uuid4()
    paths = get_ripple_glm_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        ripple_glm_id=result_id,
        artifact_root=tmp_path,
    )
    assert paths["artifact_dir"] == (
        tmp_path / "L14" / "20240611" / "ripple_glm" / "08_r4" / str(result_id)
    )
    with pytest.raises(ValueError, match="path component"):
        get_ripple_glm_artifact_paths(
            animal_name="../L14",
            date="20240611",
            epoch="08_r4",
            ripple_glm_id=result_id,
            artifact_root=tmp_path,
        )


def test_event_selection_freezes_dedupe_and_epoch_clipping() -> None:
    table = pd.DataFrame(
        {
            "start_time": [0.0, 0.1, 0.35, 0.95],
            "end_time": [0.05, 0.15, 0.4, 1.0],
        }
    )
    selected = prepare_ripple_glm_event_selection(
        epoch="02_r1",
        ripple_table=table,
        epoch_interval=_Interval(0.0, 1.0),
        ripple_selection_mode="deduped",
        ripple_window_s=0.2,
    )
    assert selected["n_ripples_before_selection"] == 4
    assert selected["n_ripples_removed_by_selection"] == 1
    assert selected["n_ripples_removed_by_window_bounds"] == 1
    assert selected["ripple_start_time_s"].tolist() == [0.0, 0.35]
    assert len(selected["selected_ripple_events_sha256"]) == 64


def test_compute_delegates_fit_and_audits_all_input_units(
    computed: dict[str, object],
) -> None:
    assert computed["analysis_status"] == "valid"
    assert computed["n_source_units"] == 2
    assert computed["n_target_units"] == 3
    assert computed["n_source_units_in_fit"] == 2
    assert computed["n_target_units_in_fit"] == 2
    assert computed["n_valid_target_units"] == 2
    audit = computed["selected_units"]
    assert audit["stable_unit_id"].tolist() == [
        "source-sort:10",
        "source-sort:11",
        "target-sort:20",
        "target-sort:21",
        "target-sort:22",
    ]
    excluded = audit.loc[audit["stable_unit_id"] == "target-sort:22"].iloc[0]
    assert excluded["unit_qc_status"] == "excluded_spike_threshold"
    assert not excluded["included_in_fit"]
    assert computed["dataset"].coords["unit"].values.tolist() == [
        "target-sort:20",
        "target-sort:21",
    ]
    assert len(computed["summary"]) == 2


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    (
        ("no_source", "no_source_units"),
        ("no_target", "no_target_units"),
        ("no_ripples", "no_ripples"),
        ("insufficient", "insufficient_ripples"),
        ("no_eligible_source", "no_eligible_source_units"),
        ("no_eligible_target", "no_eligible_target_units"),
    ),
)
def test_compute_writes_explicit_terminal_results(
    inputs: dict[str, object], mutation: str, expected_status: str
) -> None:
    kwargs = dict(inputs)
    if mutation == "no_source":
        kwargs["source_spikes"] = {}
        kwargs["source_stable_unit_ids"] = []
    elif mutation == "no_target":
        kwargs["target_spikes"] = {}
        kwargs["target_stable_unit_ids"] = []
    elif mutation == "no_ripples":
        kwargs["ripple_table"] = pd.DataFrame(columns=["start_time", "end_time"])
    elif mutation == "insufficient":
        kwargs["ripple_table"] = kwargs["ripple_table"].iloc[:1].copy()
    elif mutation == "no_eligible_source":
        kwargs["min_ca1_spikes_per_ripple"] = 100.0
    else:
        kwargs["min_spikes_per_ripple"] = 100.0
    result = compute_ripple_glm(**kwargs)
    assert result["analysis_status"] == expected_status
    assert result["n_source_units_in_fit"] == 0
    assert result["n_target_units_in_fit"] == 0
    assert result["summary"].empty
    assert not result["selected_units"]["included_in_fit"].any()


def test_all_partial_targets_produce_no_valid_target_status(
    inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = dict(inputs)
    kwargs["target_spikes"] = {30: _SpikeTrain([])}
    kwargs["target_stable_unit_ids"] = [
        {"spikesorting_merge_id": "target-sort", "unit_id": "30"}
    ]
    kwargs["min_spikes_per_ripple"] = 0.0
    monkeypatch.setattr(scientific_module, "_fit_ripple_glm_on_prepared_epoch", _fake_fit)
    result = compute_ripple_glm(**kwargs)
    assert result["analysis_status"] == "no_valid_target_units"
    assert result["n_target_units_in_fit"] == 1
    assert result["n_valid_target_units"] == 0
    assert result["selected_units"].iloc[-1]["unit_qc_status"] == (
        "partial_nonfinite_metrics"
    )


def test_validation_rejects_tampered_metric_summary(
    computed: dict[str, object],
) -> None:
    tampered = dict(computed)
    tampered["dataset"] = computed["dataset"].copy(deep=True)
    tampered["dataset"]["ripple_mae_mean"].values[0] += 0.2
    with pytest.raises(ValueError, match="ripple_mae_mean"):
        validate_ripple_glm_result(tampered)


def test_validation_rejects_tampered_window_hash(
    computed: dict[str, object],
) -> None:
    tampered = dict(computed)
    tampered["dataset"] = computed["dataset"].copy(deep=True)
    tampered["dataset"]["ripple_start_time_s"].values[0] += 0.01
    with pytest.raises(ValueError, match="offset|hash"):
        validate_ripple_glm_result(tampered)


def test_validation_rejects_false_unit_inclusion(
    computed: dict[str, object],
) -> None:
    tampered = dict(computed)
    tampered["selected_units"] = computed["selected_units"].copy()
    tampered["selected_units"].loc[
        tampered["selected_units"]["stable_unit_id"] == "target-sort:22",
        "included_in_fit",
    ] = True
    with pytest.raises(ValueError, match="threshold-passing"):
        validate_ripple_glm_result(tampered)


def test_write_load_roundtrip_and_checksum(
    tmp_path: Path, computed: dict[str, object]
) -> None:
    destination = tmp_path / str(computed["ripple_glm_id"])
    paths = write_ripple_glm_artifact(computed, destination)
    loaded = load_ripple_glm_artifact(destination)
    assert paths["result_path"].is_file()
    assert loaded["selected_units_sha256"] == computed["selected_units_sha256"]
    assert loaded["analysis_status"] == "valid"
    with paths["result_path"].open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_ripple_glm_artifact(destination)


def _legacy_dataset_from_computed(computed: dict[str, object]) -> object:
    dataset = computed["dataset"].copy(deep=True)
    audit = computed["selected_units"]
    source = audit.loc[(audit["role"] == "source") & audit["included_in_fit"]]
    target = audit.loc[(audit["role"] == "target") & audit["included_in_fit"]]
    coefficient = source.loc[source["included_in_full_coefficient"]]
    dataset = dataset.assign_coords(
        {
            "unit": np.arange(2000, 2000 + len(target), dtype=int),
            "source_unit": np.arange(len(source), dtype=int),
            "coef_source_unit": np.arange(len(coefficient), dtype=int),
        }
    )
    dataset["ca1_unit_id"] = (
        "source_unit",
        np.arange(1000, 1000 + len(source), dtype=int),
    )
    dataset["coef_ca1_unit_id"] = (
        "coef_source_unit",
        np.arange(1000, 1000 + len(coefficient), dtype=int),
    )
    dataset.attrs["schema_version"] = "8"
    return dataset


def _registration_identity_kwargs(inputs: dict[str, object]) -> dict[str, object]:
    """Map legacy sorting IDs to current ephemeral keys and persistent IDs."""
    return {
        "source_sorting_type": "ImportedSpikeSorting",
        "target_sorting_type": "ImportedSpikeSorting",
        "source_legacy_unit_identity_resolver": {
            1000 + index: {
                "group_unit_id": group_id,
                **dict(identity),
            }
            for index, (group_id, identity) in enumerate(
                zip(
                    inputs["source_spikes"],
                    inputs["source_stable_unit_ids"],
                    strict=True,
                )
            )
        },
        "target_legacy_unit_identity_resolver": {
            2000 + index: {
                "group_unit_id": group_id,
                **dict(identity),
            }
            for index, (group_id, identity) in enumerate(
                zip(
                    inputs["target_spikes"],
                    inputs["target_stable_unit_ids"],
                    strict=True,
                )
            )
        },
    }


def test_register_existing_reconstructs_nwb_inputs_and_roundtrips(
    tmp_path: Path,
    inputs: dict[str, object],
    computed: dict[str, object],
) -> None:
    source = tmp_path / "legacy.nc"
    _legacy_dataset_from_computed(computed).to_netcdf(source)
    destination = tmp_path / str(inputs["ripple_glm_id"])
    registered = register_existing_ripple_glm_artifact(
        source_result_path=source,
        destination_path=destination,
        source_v1ca1_git_commit="v1-commit",
        source_spyglass_git_commit="sg-commit",
        **_registration_identity_kwargs(inputs),
        **inputs,
    )
    assert registered["artifact_origin"] == "registered_existing"
    assert registered["legacy_artifact_provenance"]["source_result_sha256"]
    assert registered["legacy_artifact_provenance"]["source_spyglass_git_commit"] == (
        "sg-commit"
    )
    loaded = load_ripple_glm_artifact(destination)
    assert loaded["selected_ripple_events_sha256"] == computed[
        "selected_ripple_events_sha256"
    ]


def test_register_existing_rejects_spike_count_tamper(
    tmp_path: Path,
    inputs: dict[str, object],
    computed: dict[str, object],
) -> None:
    source = tmp_path / "legacy.nc"
    legacy = _legacy_dataset_from_computed(computed)
    legacy["ripple_observed_count_oof"].values[0, 0] += 1
    legacy.to_netcdf(source)
    with pytest.raises(ValueError, match="target spike counts"):
        register_existing_ripple_glm_artifact(
            source_result_path=source,
            destination_path=tmp_path / str(inputs["ripple_glm_id"]),
            **_registration_identity_kwargs(inputs),
            **inputs,
        )


def test_register_existing_rejects_parameter_mismatch(
    tmp_path: Path,
    inputs: dict[str, object],
    computed: dict[str, object],
) -> None:
    source = tmp_path / "legacy.nc"
    _legacy_dataset_from_computed(computed).to_netcdf(source)
    kwargs = dict(inputs)
    kwargs["ridge_strength"] = 0.01
    with pytest.raises(ValueError, match="ridge_strength"):
        register_existing_ripple_glm_artifact(
            source_result_path=source,
            destination_path=tmp_path / str(inputs["ripple_glm_id"]),
            **_registration_identity_kwargs(inputs),
            **kwargs,
        )


def test_register_existing_rejects_nonimported_or_missing_identity(
    tmp_path: Path,
    inputs: dict[str, object],
    computed: dict[str, object],
) -> None:
    source = tmp_path / "legacy.nc"
    _legacy_dataset_from_computed(computed).to_netcdf(source)
    identity_kwargs = _registration_identity_kwargs(inputs)
    identity_kwargs["source_sorting_type"] = "CurationV1"
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        register_existing_ripple_glm_artifact(
            source_result_path=source,
            destination_path=tmp_path / str(inputs["ripple_glm_id"]),
            **identity_kwargs,
            **inputs,
        )
    identity_kwargs = _registration_identity_kwargs(inputs)
    identity_kwargs["source_legacy_unit_identity_resolver"] = {}
    with pytest.raises(ValueError, match="resolver matches"):
        register_existing_ripple_glm_artifact(
            source_result_path=source,
            destination_path=tmp_path / str(inputs["ripple_glm_id"]),
            **identity_kwargs,
            **inputs,
        )


def test_output_rule_hash_is_stable_sha256() -> None:
    assert len(OUTPUT_RULE_SHA256) == 64
    int(OUTPUT_RULE_SHA256, 16)
