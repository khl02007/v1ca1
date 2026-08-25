"""Tests for database-free shared-cohort cross-path decoding artifacts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import path_progression_decoding as decoding


COMPARISON_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
STABLE_UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    {"spikesorting_merge_id": "merge-b", "unit_id": 22},
    {"spikesorting_merge_id": "merge-c", "unit_id": 33},
)


def _spikes():
    """Return three units with persistent identities supplied separately."""
    import pynapple as nap

    return nap.TsGroup(
        {
            10: nap.Ts(np.asarray([0.1, 0.4, 0.8]), time_units="s"),
            20: nap.Ts(np.asarray([0.2, 0.5, 0.9]), time_units="s"),
            30: nap.Ts(np.asarray([0.3, 0.6, 0.95]), time_units="s"),
        },
        time_support=nap.IntervalSet(start=0.0, end=1.0, time_units="s"),
        time_units="s",
    )


def _movement_table(
    epoch: str,
    rates: tuple[float, ...],
    *,
    group_keys: tuple[int, ...] = (10, 20, 30),
) -> pd.DataFrame:
    """Return one minimal all-unit movement-rate artifact."""
    rows = []
    for group_key, identity, rate in zip(
        group_keys, STABLE_UNIT_IDS, rates, strict=True
    ):
        merge_id = str(identity["spikesorting_merge_id"])
        unit_id = str(identity["unit_id"])
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": f"{merge_id}:{unit_id}",
                "group_unit_id": str(group_key),
                "epoch": epoch,
                "movement_firing_rate_hz": rate,
            }
        )
    return pd.DataFrame.from_records(rows)


def _stability_tables(
    epoch: str,
    maxima: tuple[float, ...],
) -> dict[str, pd.DataFrame]:
    """Return four path tables whose first path contains each requested max."""
    tables = {}
    for path_index, trajectory_type in enumerate(TRAJECTORY_TYPES):
        rows = []
        for group_key, identity, maximum in zip(
            (10, 20, 30), STABLE_UNIT_IDS, maxima, strict=True
        ):
            merge_id = str(identity["spikesorting_merge_id"])
            unit_id = str(identity["unit_id"])
            rows.append(
                {
                    "spikesorting_merge_id": merge_id,
                    "unit_id": unit_id,
                    "stable_unit_id": f"{merge_id}:{unit_id}",
                    "group_unit_id": str(group_key),
                    "epoch": epoch,
                    "trajectory_type": trajectory_type,
                    "stability_correlation": (
                        maximum if path_index == 0 else maximum - 0.2
                    ),
                }
            )
        tables[trajectory_type] = pd.DataFrame.from_records(rows)
    return tables


def _eligibility_inputs() -> dict[str, object]:
    """Return target/cohort inputs selecting only the first persistent unit."""
    return {
        "target_epoch": "02_r1",
        "cohort_epoch": "04_r2",
        "target_movement_firing_rate_table": _movement_table(
            "02_r1", (0.5, 0.6, 0.8)
        ),
        "cohort_movement_firing_rate_table": _movement_table(
            "04_r2", (0.5, 0.4, 0.8)
        ),
        "target_stability_tables_by_trajectory": _stability_tables(
            "02_r1", (0.5, 0.6, 0.4)
        ),
        "cohort_stability_tables_by_trajectory": _stability_tables(
            "04_r2", (0.5, 0.9, 0.8)
        ),
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": 0.5,
    }


def _decoded_pair(offset: float = 0.1, *, different_time_grids: bool = False):
    """Return one finite true/decoded pair, optionally on different grids."""
    import pynapple as nap

    times = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    true_values = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
    support = nap.IntervalSet(start=0.0, end=1.0, time_units="s")
    decoded_times = (
        np.asarray([0.15, 0.35, 0.55, 0.75, 0.85])
        if different_time_grids
        else times
    )
    return (
        nap.Tsd(
            t=times,
            d=true_values,
            time_support=support,
            time_units="s",
        ),
        nap.Tsd(
            t=decoded_times,
            d=np.interp(decoded_times, times, true_values) + offset,
            time_support=support,
            time_units="s",
        ),
    )


def _compute(monkeypatch, *, failure_policy=None, different_time_grids=False):
    """Compute a small deterministic bundle with decoder internals patched."""
    seen_populations = []

    def fake_progressions(**kwargs):
        return ({name: object() for name in TRAJECTORY_TYPES}, np.asarray([0, 0.5, 1]))

    def fake_decode(**kwargs):
        seen_populations.append(tuple(kwargs["spikes"].keys()))
        if failure_policy is not None:
            failure_policy(kwargs["spec"])
        return _decoded_pair(different_time_grids=different_time_grids)

    monkeypatch.setattr(decoding, "_build_path_progressions", fake_progressions)
    monkeypatch.setattr(decoding, "_decode_cross_path_pair", fake_decode)
    eligibility = _eligibility_inputs()
    result = decoding.compute_path_progression_decoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        cohort_epoch="04_r2",
        path_progression_decoding_id=COMPARISON_ID,
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        target_movement_firing_rate_table=eligibility[
            "target_movement_firing_rate_table"
        ],
        cohort_movement_firing_rate_table=eligibility[
            "cohort_movement_firing_rate_table"
        ],
        target_stability_tables_by_trajectory=eligibility[
            "target_stability_tables_by_trajectory"
        ],
        cohort_stability_tables_by_trajectory=eligibility[
            "cohort_stability_tables_by_trajectory"
        ],
        position=object(),
        trajectory_intervals={name: object() for name in TRAJECTORY_TYPES},
        graph_inputs={name: {} for name in TRAJECTORY_TYPES},
        movement_interval=object(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
        parameter_name="test",
    )
    return result, seen_populations


def test_transfer_specs_are_fixed_complete_and_immutable() -> None:
    assert decoding.EXPECTED_TRANSFER_PAIR_COUNT == 16
    assert len(decoding.TRANSFER_PAIR_SPECS) == 16
    keys = [
        (
            spec["transfer_family"],
            spec["source_trajectory"],
            spec["target_trajectory"],
        )
        for spec in decoding.TRANSFER_PAIR_SPECS
    ]
    assert len(keys) == len(set(keys))
    assert Counter(key[0] for key in keys) == {
        "same_turn_cross_arm": 4,
        "opposite_turn_same_arm": 4,
        "opposite_turn_same_arm_flipped": 4,
        "same_inbound_outbound_cross_arm": 4,
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            [dict(spec) for spec in decoding.TRANSFER_PAIR_SPECS],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert decoding.TRANSFER_SPEC_SHA256 == expected_digest
    with pytest.raises(TypeError):
        decoding.TRANSFER_PAIR_SPECS[0]["source_trajectory"] = "changed"
    assert decoding.MANUSCRIPT_PARAMETERS == {
        "decoding_bin_size_s": 0.02,
        "sliding_window_size_bins": 4,
        "spatial_bin_size_cm": 4.0,
        "error_mode": "signed",
        "error_summary": "median_iqr",
        "min_bin_count": 5,
    }
    assert "n_folds" not in decoding.MANUSCRIPT_PARAMETERS
    assert "random_seed" not in decoding.MANUSCRIPT_PARAMETERS
    assert dict(decoding.DECODING_OUTPUT_RULE) == {
        "version": 1,
        "coordinate_unit": "normalized_path_progression",
        "time_unit": "s",
        "time_reference": "augmented_nwb_ephys_timestamps",
        "error_mode": "signed",
        "error_summary": "median_iqr",
        "min_bin_count": 5,
    }


def test_canonical_artifact_paths_are_session_first(tmp_path: Path) -> None:
    paths = decoding.get_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        cohort_epoch="04_r2",
        region="v1",
        path_progression_decoding_id=COMPARISON_ID,
        artifact_root=tmp_path,
    )
    expected = (
        tmp_path
        / "L14"
        / "20240611"
        / decoding.ARTIFACT_DIRNAME
        / "02_r1"
        / "v1"
        / str(COMPARISON_ID)
    )
    assert paths["artifact_dir"] == expected
    assert paths["artifact_manifest_path"] == expected / "manifest.parquet"
    assert paths["decoding_summary_path"] == expected / "decoding_summary.parquet"
    assert paths["unit_eligibility_path"] == expected / "unit_eligibility.parquet"


def test_symmetric_cohort_eligibility_is_inclusive_and_swappable() -> None:
    inputs = _eligibility_inputs()
    table = decoding.build_symmetric_cohort_eligibility_table(**inputs)
    assert list(table.columns) == list(decoding.ELIGIBILITY_COLUMNS)
    assert decoding.get_shared_eligible_stable_unit_ids(table) == ["merge-a:11"]
    first = table.set_index("stable_unit_id").loc["merge-a:11"]
    assert first["target_passes_movement_firing_rate"]
    assert first["cohort_passes_movement_firing_rate"]
    assert first["target_passes_stability"]
    assert first["cohort_passes_stability"]

    swapped = decoding.build_symmetric_cohort_eligibility_table(
        target_epoch=inputs["cohort_epoch"],
        cohort_epoch=inputs["target_epoch"],
        target_movement_firing_rate_table=inputs[
            "cohort_movement_firing_rate_table"
        ],
        cohort_movement_firing_rate_table=inputs[
            "target_movement_firing_rate_table"
        ],
        target_stability_tables_by_trajectory=inputs[
            "cohort_stability_tables_by_trajectory"
        ],
        cohort_stability_tables_by_trajectory=inputs[
            "target_stability_tables_by_trajectory"
        ],
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
    )
    assert table["stable_unit_id"].tolist() == swapped["stable_unit_id"].tolist()
    np.testing.assert_array_equal(
        table["shared_eligible"], swapped["shared_eligible"]
    )
    np.testing.assert_allclose(
        table["target_movement_firing_rate_hz"],
        swapped["cohort_movement_firing_rate_hz"],
    )
    for trajectory_type in TRAJECTORY_TYPES:
        np.testing.assert_allclose(
            table[f"target_{trajectory_type}_stability_correlation"],
            swapped[f"cohort_{trajectory_type}_stability_correlation"],
        )


def test_ephemeral_group_keys_may_differ_across_epochs_and_reload(
    monkeypatch,
) -> None:
    """Persistent identities, not temporary TsGroup keys, define the cohort."""
    eligibility = _eligibility_inputs()
    eligibility["target_movement_firing_rate_table"] = _movement_table(
        "02_r1", (0.5, 0.6, 0.8), group_keys=(110, 120, 130)
    )
    eligibility["cohort_movement_firing_rate_table"] = _movement_table(
        "04_r2", (0.5, 0.4, 0.8), group_keys=(210, 220, 230)
    )
    monkeypatch.setattr(
        decoding,
        "_build_path_progressions",
        lambda **kwargs: (
            {name: object() for name in TRAJECTORY_TYPES},
            np.asarray([0.0, 0.5, 1.0]),
        ),
    )
    monkeypatch.setattr(
        decoding,
        "_decode_cross_path_pair",
        lambda **kwargs: _decoded_pair(),
    )
    result = decoding.compute_path_progression_decoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        cohort_epoch="04_r2",
        path_progression_decoding_id=COMPARISON_ID,
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        target_movement_firing_rate_table=eligibility[
            "target_movement_firing_rate_table"
        ],
        cohort_movement_firing_rate_table=eligibility[
            "cohort_movement_firing_rate_table"
        ],
        target_stability_tables_by_trajectory=eligibility[
            "target_stability_tables_by_trajectory"
        ],
        cohort_stability_tables_by_trajectory=eligibility[
            "cohort_stability_tables_by_trajectory"
        ],
        position=object(),
        trajectory_intervals={name: object() for name in TRAJECTORY_TYPES},
        graph_inputs={name: {} for name in TRAJECTORY_TYPES},
        movement_interval=object(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
        parameter_name="test",
    )
    assert result["selected_units"]["group_unit_id"].tolist() == ["10"]
    assert result["unit_eligibility"]["group_unit_id"].tolist() == [
        "10",
        "20",
        "30",
    ]


def test_none_stability_threshold_disables_stability_filter() -> None:
    inputs = _eligibility_inputs()
    table = decoding.build_symmetric_cohort_eligibility_table(
        target_epoch="02_r1",
        cohort_epoch="04_r2",
        target_movement_firing_rate_table=inputs[
            "target_movement_firing_rate_table"
        ],
        cohort_movement_firing_rate_table=inputs[
            "cohort_movement_firing_rate_table"
        ],
        target_stability_tables_by_trajectory=None,
        cohort_stability_tables_by_trajectory=None,
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=None,
    )
    assert decoding.get_shared_eligible_stable_unit_ids(table) == [
        "merge-a:11",
        "merge-c:33",
    ]
    assert table["target_passes_stability"].all()
    assert table["cohort_passes_stability"].all()
    correlation_columns = [
        column
        for column in table
        if column.startswith(("target_", "cohort_"))
        and column.endswith("stability_correlation")
    ]
    assert table[correlation_columns].isna().all().all()

    audited = decoding.build_symmetric_cohort_eligibility_table(
        **{**inputs, "minimum_stability_correlation": None}
    )
    assert audited["target_passes_stability"].all()
    assert audited["cohort_passes_stability"].all()
    assert audited[correlation_columns].notna().all().all()


def test_eligibility_rejects_epoch_and_trajectory_mismatch() -> None:
    inputs = _eligibility_inputs()
    bad_movement = inputs["target_movement_firing_rate_table"].copy()
    bad_movement["epoch"] = "wrong"
    with pytest.raises(ValueError, match="does not match epoch"):
        decoding.build_symmetric_cohort_eligibility_table(
            **{**inputs, "target_movement_firing_rate_table": bad_movement}
        )

    bad_stability = {
        name: table.copy()
        for name, table in inputs[
            "target_stability_tables_by_trajectory"
        ].items()
    }
    bad_stability[TRAJECTORY_TYPES[0]]["trajectory_type"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        decoding.build_symmetric_cohort_eligibility_table(
            **{
                **inputs,
                "target_stability_tables_by_trajectory": bad_stability,
            }
        )


def test_compute_uses_one_exact_shared_population_for_all_transfers(
    monkeypatch,
) -> None:
    result, seen = _compute(monkeypatch)
    assert len(seen) == 16
    assert all(set(population) == {10} for population in seen)
    assert result["selected_units"]["stable_unit_id"].tolist() == ["merge-a:11"]
    assert result["n_units_input"] == 3
    assert result["n_units_eligible"] == 1
    assert result["n_transfer_pairs_expected"] == 16
    assert result["n_transfer_pairs_valid"] == 16
    assert result["n_decoded_samples"] == 80
    assert result["analysis_status"] == "valid"
    assert len(result["cross_path_outputs"]) == 16
    assert result["cross_path_metrics"]["qc_status"].eq("valid").all()
    np.testing.assert_allclose(result["cross_path_metrics"]["mae"], 0.1)
    assert len(result["eligible_units_sha256"]) == 64
    from v1ca1.spyglass.selection import provenance_sha256

    assert result["metadata"]["decoding_output_rule_sha256"] == (
        provenance_sha256(dict(decoding.DECODING_OUTPUT_RULE))
    )


def test_true_and_decoded_outputs_may_use_different_time_grids(monkeypatch) -> None:
    result, _ = _compute(monkeypatch, different_time_grids=True)
    assert result["analysis_status"] == "valid"
    assert result["n_decoded_samples"] == 80
    assert result["cross_path_metrics"]["mae"].notna().all()


def test_expected_transfer_support_failure_is_partial_and_audited(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_key = tuple(
        decoding.TRANSFER_PAIR_SPECS[0][name]
        for name in (
            "transfer_family",
            "source_trajectory",
            "target_trajectory",
        )
    )

    def fail_first(spec):
        key = tuple(
            spec[name]
            for name in (
                "transfer_family",
                "source_trajectory",
                "target_trajectory",
            )
        )
        if key == first_key:
            raise decoding.TransferSupportError(
                "no_target_count_bins", "expected short target support"
            )

    result, _ = _compute(monkeypatch, failure_policy=fail_first)
    assert result["analysis_status"] == "partial_valid"
    assert result["n_transfer_pairs_valid"] == 15
    assert result["n_decoded_samples"] == 75
    row = result["cross_path_metrics"].iloc[0]
    assert row["qc_status"] == "no_target_count_bins"
    assert row["qc_message"] == "expected short target support"
    assert first_key not in result["cross_path_outputs"]
    paths = decoding.get_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        cohort_epoch="04_r2",
        region="v1",
        path_progression_decoding_id=COMPARISON_ID,
        artifact_root=tmp_path,
    )
    written = decoding.write_decoding_artifact_bundle(result, paths)
    assert len(written["manifest"]) == 34
    loaded = decoding.load_decoding_artifact_bundle(
        written["artifact_manifest_path"]
    )
    assert loaded["analysis_status"] == "partial_valid"
    assert loaded["n_transfer_pairs_valid"] == 15


def test_all_expected_support_failures_produce_no_valid_decodes(monkeypatch) -> None:
    def fail_all(spec):
        raise decoding.TransferSupportError(
            "no_source_movement", "no source movement"
        )

    result, _ = _compute(monkeypatch, failure_policy=fail_all)
    assert result["analysis_status"] == "no_valid_decodes"
    assert result["n_transfer_pairs_valid"] == 0
    assert result["n_decoded_samples"] == 0
    assert result["cross_path_outputs"] == {}
    assert result["cross_path_binned_error"].empty
    assert result["cross_path_metrics"]["qc_status"].eq(
        "no_source_movement"
    ).all()


def test_unexpected_transfer_error_is_not_swallowed(monkeypatch) -> None:
    def fail_unexpected(spec):
        raise RuntimeError("decoder software failure")

    with pytest.raises(RuntimeError, match="decoder software failure"):
        _compute(monkeypatch, failure_policy=fail_unexpected)


def test_no_eligible_units_is_a_terminal_audited_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    eligibility = _eligibility_inputs()
    monkeypatch.setattr(
        decoding,
        "_build_path_progressions",
        lambda **kwargs: pytest.fail("terminal results must not build graphs"),
    )
    result = decoding.compute_path_progression_decoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        cohort_epoch="04_r2",
        path_progression_decoding_id=COMPARISON_ID,
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        target_movement_firing_rate_table=eligibility[
            "target_movement_firing_rate_table"
        ],
        cohort_movement_firing_rate_table=eligibility[
            "cohort_movement_firing_rate_table"
        ],
        target_stability_tables_by_trajectory=None,
        cohort_stability_tables_by_trajectory=None,
        position=object(),
        trajectory_intervals={name: object() for name in TRAJECTORY_TYPES},
        graph_inputs={name: {} for name in TRAJECTORY_TYPES},
        movement_interval=object(),
        minimum_movement_firing_rate_hz=2.0,
        minimum_stability_correlation=None,
        parameter_name="none",
    )
    assert result["analysis_status"] == "no_eligible_units"
    assert result["n_units_input"] == 3
    assert result["n_units_eligible"] == 0
    assert result["n_transfer_pairs_valid"] == 0
    assert result["cross_path_metrics"]["qc_status"].eq(
        "no_eligible_units"
    ).all()
    assert result["cross_path_outputs"] == {}
    assert result["cross_path_binned_error"].empty
    paths = decoding.get_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        cohort_epoch="04_r2",
        region="v1",
        path_progression_decoding_id=COMPARISON_ID,
        artifact_root=tmp_path,
    )
    written = decoding.write_decoding_artifact_bundle(result, paths)
    assert len(written["manifest"]) == 4
    loaded = decoding.load_decoding_artifact_bundle(
        written["artifact_manifest_path"]
    )
    assert loaded["analysis_status"] == "no_eligible_units"
    assert loaded["cross_path_outputs"] == {}


def test_no_units_is_a_terminal_roundtrip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import pynapple as nap

    monkeypatch.setattr(
        decoding,
        "_build_path_progressions",
        lambda **kwargs: pytest.fail("terminal results must not build graphs"),
    )
    spikes = nap.TsGroup(
        {},
        time_support=nap.IntervalSet(start=0.0, end=1.0, time_units="s"),
        time_units="s",
    )
    movement = pd.DataFrame(
        columns=[
            "spikesorting_merge_id",
            "unit_id",
            "stable_unit_id",
            "group_unit_id",
            "animal_name",
            "date",
            "region",
            "epoch",
            "movement_firing_rate_hz",
        ]
    )
    target_movement = movement.copy()
    cohort_movement = movement.copy()
    result = decoding.compute_path_progression_decoding(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        cohort_epoch="04_r2",
        path_progression_decoding_id=COMPARISON_ID,
        spikes=spikes,
        stable_unit_ids=(),
        target_movement_firing_rate_table=target_movement,
        cohort_movement_firing_rate_table=cohort_movement,
        target_stability_tables_by_trajectory=None,
        cohort_stability_tables_by_trajectory=None,
        position=object(),
        trajectory_intervals={name: object() for name in TRAJECTORY_TYPES},
        graph_inputs={name: {} for name in TRAJECTORY_TYPES},
        movement_interval=object(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=None,
        parameter_name="none",
    )
    assert result["analysis_status"] == "no_units"
    paths = decoding.get_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        cohort_epoch="04_r2",
        region="v1",
        path_progression_decoding_id=COMPARISON_ID,
        artifact_root=tmp_path,
    )
    written = decoding.write_decoding_artifact_bundle(result, paths)
    loaded = decoding.load_decoding_artifact_bundle(
        written["artifact_manifest_path"]
    )
    assert loaded["analysis_status"] == "no_units"
    assert loaded["n_units_input"] == 0


def test_validator_rejects_tampered_metric_binned_and_eligibility(monkeypatch) -> None:
    result, _ = _compute(monkeypatch)
    bad_metric = {**result, "cross_path_metrics": result["cross_path_metrics"].copy()}
    bad_metric["cross_path_metrics"].loc[0, "mae"] = 99.0
    with pytest.raises(ValueError, match="metric"):
        decoding.validate_decoding_comparison_result(bad_metric)

    bad_binned = {
        **result,
        "cross_path_binned_error": result["cross_path_binned_error"].copy(),
    }
    bad_binned["cross_path_binned_error"].loc[0, "center"] = 99.0
    with pytest.raises(ValueError, match="binned errors"):
        decoding.validate_decoding_comparison_result(bad_binned)

    bad_eligibility = {
        **result,
        "unit_eligibility": result["unit_eligibility"].copy(),
    }
    bad_eligibility["unit_eligibility"].loc[0, "shared_eligible"] = False
    with pytest.raises(ValueError, match="Shared eligibility"):
        decoding.validate_decoding_comparison_result(bad_eligibility)


def test_artifact_bundle_roundtrip_refuses_overwrite_and_detects_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result, _ = _compute(monkeypatch)
    paths = decoding.get_decoding_artifact_paths(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        cohort_epoch="04_r2",
        region="v1",
        path_progression_decoding_id=COMPARISON_ID,
        artifact_root=tmp_path,
    )
    written = decoding.write_decoding_artifact_bundle(result, paths)
    assert written["artifact_manifest_path"] == paths["artifact_manifest_path"]
    assert written["decoding_summary_path"] == paths["decoding_summary_path"]
    assert written["unit_eligibility_path"] == paths["unit_eligibility_path"]
    assert len(written["manifest"]) == 36
    loaded = decoding.load_decoding_artifact_bundle(
        paths["artifact_manifest_path"]
    )
    summary = decoding.summarize_decoding_artifact_bundle(loaded)
    assert summary["analysis_status"] == "valid"
    assert summary["n_transfer_pairs_valid"] == 16
    pd.testing.assert_frame_equal(
        loaded["unit_eligibility"], result["unit_eligibility"]
    )
    key = next(iter(result["cross_path_outputs"]))
    np.testing.assert_allclose(
        loaded["cross_path_outputs"][key]["decoded"].d,
        result["cross_path_outputs"][key]["decoded"].d,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        decoding.write_decoding_artifact_bundle(result, paths)

    decoded_path = paths["artifact_dir"] / decoding._npz_filename(key, "decoded")
    with decoded_path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        decoding.load_decoding_artifact_bundle(paths["artifact_manifest_path"])


def test_analysis_nwb_roundtrip_preserves_all_transfer_objects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """NWB tables and per-transfer streams reconstruct the exact result."""
    from pynwb import NWBHDF5IO, NWBFile

    result, _ = _compute(monkeypatch, different_time_grids=True)
    fixed_objects = {
        "unit_eligibility": decoding.unit_eligibility_to_dynamic_table(
            result["unit_eligibility"]
        ),
        "selected_units": decoding.selected_units_to_dynamic_table(
            result["selected_units"]
        ),
        "decoding_summary": decoding.decoding_summary_to_dynamic_table(
            result["cross_path_metrics"]
        ),
        "binned_error": decoding.binned_error_to_dynamic_table(
            result["cross_path_binned_error"]
        ),
        "transfer_index": decoding.transfer_index_to_dynamic_table(
            decoding.build_transfer_index_table(result)
        ),
        "provenance": decoding.decoding_provenance_to_dynamic_table(result),
    }
    nwbfile = NWBFile(
        session_description="PathProgressionDecoding NWB roundtrip",
        identifier="path-progression-decoding-test",
        session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    fixed_ids = {}
    for name, nwb_object in fixed_objects.items():
        nwbfile.add_scratch(nwb_object)
        fixed_ids[name] = str(nwb_object.object_id)
    transfer_ids = {}
    for key, output in result["cross_path_outputs"].items():
        true = decoding.transfer_progression_to_time_series(
            output["true"], key=key, role="true"
        )
        decoded = decoding.transfer_progression_to_time_series(
            output["decoded"], key=key, role="decoded"
        )
        support = decoding.transfer_support_to_time_intervals(
            output["true"], output["decoded"], key=key
        )
        nwbfile.add_scratch(true)
        nwbfile.add_scratch(decoded)
        nwbfile.add_time_intervals(support)
        transfer_ids[key] = {
            "true_progression": str(true.object_id),
            "decoded_progression": str(decoded.object_id),
            "decoding_support": str(support.object_id),
        }
    output_path = tmp_path / "path_progression_decoding.nwb"
    with NWBHDF5IO(str(output_path), mode="w") as io:
        io.write(nwbfile)
    assert not __import__("pynwb").validate(path=output_path)

    with NWBHDF5IO(str(output_path), mode="r", load_namespaces=True) as io:
        stored = io.read()
        loaded = decoding.path_progression_decoding_result_from_nwb_objects(
            unit_eligibility=stored.objects[fixed_ids["unit_eligibility"]],
            selected_units=stored.objects[fixed_ids["selected_units"]],
            decoding_summary=stored.objects[fixed_ids["decoding_summary"]],
            binned_error=stored.objects[fixed_ids["binned_error"]],
            transfer_index=stored.objects[fixed_ids["transfer_index"]],
            provenance=stored.objects[fixed_ids["provenance"]],
            transfer_objects={
                key: {
                    role: stored.objects[object_id]
                    for role, object_id in role_ids.items()
                }
                for key, role_ids in transfer_ids.items()
            },
        )
    pd.testing.assert_frame_equal(
        loaded["unit_eligibility"],
        decoding.unit_eligibility_from_dynamic_table(
            fixed_objects["unit_eligibility"]
        ),
    )
    assert tuple(loaded["cross_path_outputs"]) == tuple(
        result["cross_path_outputs"]
    )
    first_key = next(iter(result["cross_path_outputs"]))
    assert not np.array_equal(
        loaded["cross_path_outputs"][first_key]["true"].t,
        loaded["cross_path_outputs"][first_key]["decoded"].t,
    )
    assert decoding.decoding_provenance_sha256(loaded) == (
        decoding.decoding_provenance_sha256(result)
    )


def test_empty_nwb_tables_roundtrip_for_no_valid_decodes(monkeypatch) -> None:
    """Terminal results retain typed empty selected/binned/index tables."""
    result, _ = _compute(
        monkeypatch,
        failure_policy=lambda _spec: (_ for _ in ()).throw(
            decoding.TransferSupportError("no_target_movement", "missing")
        ),
    )
    assert result["analysis_status"] == "no_valid_decodes"
    for table, to_nwb, from_nwb in (
        (
            result["cross_path_binned_error"],
            decoding.binned_error_to_dynamic_table,
            decoding.binned_error_from_dynamic_table,
        ),
        (
            decoding.build_transfer_index_table(result),
            decoding.transfer_index_to_dynamic_table,
            decoding.transfer_index_from_dynamic_table,
        ),
    ):
        restored = from_nwb(to_nwb(table))
        pd.testing.assert_frame_equal(restored, table, check_dtype=False)


def test_live_loader_uses_parent_and_transfer_fetch_nwb(monkeypatch) -> None:
    """The live loader resolves fixed and variable objects via fetch_nwb."""
    from v1ca1.spyglass import tables

    result, _ = _compute(monkeypatch, different_time_grids=True)
    transfer_index = decoding.build_transfer_index_table(result)
    parent_record = {
        "unit_eligibility": decoding.unit_eligibility_to_dynamic_table(
            result["unit_eligibility"]
        ).to_dataframe(),
        "selected_units": decoding.selected_units_to_dynamic_table(
            result["selected_units"]
        ).to_dataframe(),
        "decoding_summary": decoding.decoding_summary_to_dynamic_table(
            result["cross_path_metrics"]
        ).to_dataframe(),
        "cross_path_binned_error": decoding.binned_error_to_dynamic_table(
            result["cross_path_binned_error"]
        ).to_dataframe(),
        "transfer_index": decoding.transfer_index_to_dynamic_table(
            transfer_index
        ).to_dataframe(),
        "decoding_provenance": decoding.decoding_provenance_to_dynamic_table(
            result
        ).to_dataframe(),
    }
    transfer_hashes = tables._path_progression_transfer_hashes(result)
    transfer_records = []
    for key, output in result["cross_path_outputs"].items():
        transfer_records.append(
            {
                "transfer_family": key[0],
                "source_trajectory": key[1],
                "target_trajectory": key[2],
                "true_progression": decoding.transfer_progression_to_time_series(
                    output["true"], key=key, role="true"
                ),
                "decoded_progression": (
                    decoding.transfer_progression_to_time_series(
                        output["decoded"], key=key, role="decoded"
                    )
                ),
                "decoding_support": (
                    decoding.transfer_support_to_time_intervals(
                        output["true"], output["decoded"], key=key
                    ).to_dataframe()
                ),
                **transfer_hashes[key],
                "n_samples": len(output["decoded"]),
            }
        )

    class FetchRelation:
        def __init__(self, records):
            self.records = records
            self.keys = []

        def __and__(self, key):
            self.keys.append(dict(key))
            return self

        def fetch_nwb(self):
            return self.records

    parent_relation = FetchRelation([parent_record])
    transfer_relation = FetchRelation(list(reversed(transfer_records)))
    metadata = result["metadata"]
    result_row = {
        "path_progression_decoding_id": metadata[
            "path_progression_decoding_id"
        ],
        "artifact_schema_version": decoding.NWB_ARTIFACT_SCHEMA_VERSION,
        **tables._path_progression_decoding_hashes(result),
        **{
            name: result[name]
            for name in (
                "n_units_input",
                "n_units_eligible",
                "n_transfer_pairs_expected",
                "n_transfer_pairs_valid",
                "n_decoded_samples",
                "analysis_status",
                "eligible_units_sha256",
            )
        },
    }
    selection_row = {
        "path_progression_decoding_id": metadata[
            "path_progression_decoding_id"
        ],
        "epoch": metadata["epoch"],
        "cohort_epoch": metadata["cohort_epoch"],
        "path_progression_decoding_param_name": metadata["parameter_name"],
        "path_progression_decoding_parameters_sha256": metadata[
            "parameter_sha256"
        ],
        "eligibility_rule_sha256": metadata["eligibility_rule_sha256"],
        "transfer_spec_sha256": metadata["transfer_spec_sha256"],
        "decoding_output_rule_sha256": metadata[
            "decoding_output_rule_sha256"
        ],
    }
    loaded = tables._load_path_progression_decoding_result(
        result_row=result_row,
        decoding_table=parent_relation,
        transfer_table=transfer_relation,
        selection_row=selection_row,
        parameters_row={
            "path_progression_decoding_param_name": metadata[
                "parameter_name"
            ]
        },
        region_row={"region_name": metadata["region"]},
        animal_name=metadata["animal_name"],
        date=metadata["date"],
    )
    assert tuple(loaded["cross_path_outputs"]) == tuple(
        result["cross_path_outputs"]
    )
    assert parent_relation.keys == [
        {
            "path_progression_decoding_id": metadata[
                "path_progression_decoding_id"
            ]
        }
    ]
    assert transfer_relation.keys == parent_relation.keys

    with pytest.raises(ValueError, match="object hash mismatch"):
        tables._load_path_progression_decoding_result(
            result_row={**result_row, "decoding_summary_sha256": "0" * 64},
            decoding_table=parent_relation,
            transfer_table=transfer_relation,
            selection_row=selection_row,
            parameters_row={
                "path_progression_decoding_param_name": metadata[
                    "parameter_name"
                ]
            },
            region_row={"region_name": metadata["region"]},
            animal_name=metadata["animal_name"],
            date=metadata["date"],
        )


def test_live_writer_creates_one_verified_analysis_nwb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The live writer stores fixed tables and all valid transfer triples."""
    from pynwb import NWBHDF5IO, NWBFile

    from v1ca1.spyglass import tables

    result, _ = _compute(monkeypatch, different_time_grids=True)
    analysis_path = tmp_path / "path-progression-analysis.nwb"

    class Builder:
        def __init__(self):
            self.analysis_file_name = analysis_path.name
            self.nwbfile = NWBFile(
                session_description="PathProgressionDecoding writer test",
                identifier="path-progression-writer-test",
                session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            )
            self.registered = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.registered = exc_type is None
            return False

        def get_path(self):
            return str(analysis_path)

        @property
        def open_nwb(self):
            return None, self.nwbfile

        def add_nwb_object(self, nwb_object):
            self.nwbfile.add_scratch(nwb_object)
            return str(nwb_object.object_id)

        def close_and_write(self):
            with NWBHDF5IO(str(analysis_path), mode="w") as io:
                io.write(self.nwbfile)

    class AnalysisTable:
        def __init__(self):
            self.builder = Builder()

        def build(self, nwb_file_name):
            assert nwb_file_name == "L1420240611_.nwb"
            return self.builder

    analysis_table = AnalysisTable()
    row = tables._write_path_progression_decoding_nwb(
        nwb_file_name="L1420240611_.nwb",
        result=result,
        analysis_nwbfile_table=analysis_table,
    )
    assert analysis_table.builder.registered
    assert row["analysis_file_name"] == analysis_path.name
    assert row["artifact_schema_version"] == decoding.NWB_ARTIFACT_SCHEMA_VERSION
    assert len(row["_transfer_rows"]) == 16
    assert len(
        {
            object_id
            for transfer in row["_transfer_rows"]
            for object_id in (
                transfer["true_progression_object_id"],
                transfer["decoded_progression_object_id"],
                transfer["decoding_support_object_id"],
            )
        }
    ) == 48
    with NWBHDF5IO(str(analysis_path), mode="r", load_namespaces=True) as io:
        stored = io.read()
        assert len(stored.intervals) == 16
        assert len(stored.scratch) == 38
