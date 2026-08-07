"""Tests for database-free path-specific tuning-similarity artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.spyglass import movement, tuning_similarity


UNIT_IDENTITIES = (
    ("merge-a", "11", "curve-unit-a"),
    ("merge-b", "22", "curve-unit-b"),
)
TRAJECTORY_VALUES = {
    "center_to_left": np.asarray(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=float,
    ),
    "right_to_center": np.asarray(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=float,
    ),
    "center_to_right": np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=float,
    ),
    "left_to_center": np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]],
        dtype=float,
    ),
}


def _canonical_curve(
    trajectory_type: str,
    *,
    trial_subset: str = "all",
    centers_cm: np.ndarray | None = None,
    identities: tuple[tuple[str, str, str], ...] = UNIT_IDENTITIES,
) -> xr.DataArray:
    """Return one minimal canonical path-specific tuning curve."""
    values_by_stable_id = {
        "merge-a:11": TRAJECTORY_VALUES[trajectory_type][0],
        "merge-b:22": TRAJECTORY_VALUES[trajectory_type][1],
    }
    values = np.asarray(
        [
            values_by_stable_id[f"{merge_id}:{unit_id}"]
            for merge_id, unit_id, _ in identities
        ],
        dtype=float,
    )
    if centers_cm is None:
        centers_cm = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=float)
    centers_cm = np.asarray(centers_cm, dtype=float)
    step_cm = float(centers_cm[1] - centers_cm[0])
    edges_cm = np.concatenate(
        ([centers_cm[0] - step_cm / 2.0], centers_cm + step_cm / 2.0)
    )
    graph_length_cm = float(edges_cm[-1])
    merge_ids = [merge_id for merge_id, _unit_id, _group_id in identities]
    unit_ids = [unit_id for _merge_id, unit_id, _group_id in identities]
    group_ids = [group_id for _merge_id, _unit_id, group_id in identities]
    stable_ids = [
        f"{merge_id}:{unit_id}" for merge_id, unit_id, _group_id in identities
    ]
    curve = xr.DataArray(
        values,
        dims=("unit", "linear_position_cm"),
        coords={
            "unit": stable_ids,
            "spikesorting_merge_id": ("unit", merge_ids),
            "unit_id": ("unit", unit_ids),
            "stable_unit_id": ("unit", stable_ids),
            "group_unit_id": ("unit", group_ids),
            "spike_count": ("unit", np.asarray([4, 4], dtype=int)),
            "linear_position_cm": centers_cm,
            "path_fraction": (
                "linear_position_cm",
                centers_cm / graph_length_cm,
            ),
        },
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "02_r1",
            "trajectory_type": trajectory_type,
            "trial_subset": trial_subset,
            "binning_mode": "bin_size_cm",
            "bin_size_cm": 2.0,
            "sigma_bins": 0.0,
            "graph_length_cm": graph_length_cm,
            "bin_edges_cm_json": json.dumps(edges_cm.tolist()),
            "n_trials": 2,
            "support_duration_s": 2.0,
            "n_feature_samples": 20,
            "n_valid_position_samples": 20,
            "n_units": 2,
            "n_valid_units": 2,
            "analysis_status": "valid",
            "units": "Hz",
        },
    )
    curve.coords["linear_position_cm"].attrs["units"] = "cm"
    curve.coords["path_fraction"].attrs["units"] = "1"
    return curve


def _curves() -> dict[str, xr.DataArray]:
    """Return all four matching canonical all-trial curves."""
    return {
        trajectory: _canonical_curve(trajectory)
        for trajectory in tuning_similarity.REQUIRED_TRAJECTORIES
    }


def _empty_curves() -> dict[str, xr.DataArray]:
    """Return four matching canonical curves for an empty selected group."""
    curves = _curves()
    for trajectory_type, curve in curves.items():
        empty = curve.isel(unit=slice(0, 0)).copy(deep=True)
        empty.attrs.update(
            {
                "n_units": 0,
                "n_valid_units": 0,
                "analysis_status": "no_units",
            }
        )
        curves[trajectory_type] = empty
    return curves


def _movement_table() -> pd.DataFrame:
    """Return the canonical upstream epoch-wide movement-rate artifact."""
    return pd.DataFrame(
        {
            "spikesorting_merge_id": ["merge-a", "merge-b"],
            "unit_id": ["11", "22"],
            "stable_unit_id": ["merge-a:11", "merge-b:22"],
            "group_unit_id": ["movement-unit-a", "movement-unit-b"],
            "animal_name": ["L14", "L14"],
            "date": ["20240611", "20240611"],
            "region": ["v1", "v1"],
            "epoch": ["02_r1", "02_r1"],
            "movement_spike_count": [3, 1],
            "movement_duration_s": [2.0, 2.0],
            "movement_firing_rate_hz": [1.5, 0.5],
            "firing_rate_status": ["valid", "valid"],
            "position_sample_count": [20, 20],
            "finite_position_sample_count": [20, 20],
            "finite_speed_sample_count": [20, 20],
            "movement_interval_count": [2, 2],
            "speed_threshold_cm_s": [4.0, 4.0],
            "speed_smoothing_sigma_s": [0.1, 0.1],
        }
    )


def _computed_table() -> pd.DataFrame:
    """Return the canonical two-unit correlation artifact."""
    return tuning_similarity.compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=_curves(),
        movement_firing_rate_table=_movement_table(),
        similarity_metric="correlation",
    )["table"]


def _legacy_table() -> pd.DataFrame:
    """Return a legacy direct-only all-unit table with ephemeral ids."""
    table = _computed_table().copy()
    legacy_ids = table["stable_unit_id"].map(
        {"merge-a:11": 101, "merge-b:22": 202}
    )
    table = table.assign(unit=legacy_ids, firing_rate_hz=table["movement_firing_rate_hz"])
    return table.loc[:, list(tuning_similarity.LEGACY_COLUMNS)]


def _resolver() -> dict[int, dict[str, object]]:
    """Return the explicit legacy-to-persistent unit identity map."""
    return {
        101: {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        202: {"spikesorting_merge_id": "merge-b", "unit_id": 22},
    }


def test_artifact_path_is_session_first_metric_and_uuid_keyed(
    tmp_path: Path,
) -> None:
    similarity_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    path = tuning_similarity.get_tuning_similarity_artifact_path(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        region="v1",
        similarity_metric="absolute_overlap",
        path_specific_place_tuning_similarity_id=similarity_id,
        artifact_root=tmp_path,
    )

    assert path == (
        tmp_path
        / "L14"
        / "20240611"
        / "path_specific_place_tuning_similarity"
        / "02_r1"
        / "v1"
        / "absolute_overlap"
        / str(similarity_id)
        / "similarity.parquet"
    )
    with pytest.raises(ValueError, match="UUID"):
        tuning_similarity.get_tuning_similarity_artifact_path(
            animal_name="L14",
            date="20240611",
            epoch="02_r1",
            region="v1",
            similarity_metric="correlation",
            path_specific_place_tuning_similarity_id="not-a-uuid",
            artifact_root=tmp_path,
        )


def test_compute_retains_all_units_four_direct_rows_qc_and_movement_rate() -> None:
    result = tuning_similarity.compute_tuning_similarity_from_curves(
        tuning_curves_by_trajectory=_curves(),
        movement_firing_rate_table=_movement_table(),
        similarity_metric="correlation",
    )
    table = result["table"]

    assert result == {
        "table": table,
        "analysis_status": "valid",
        "n_units": 2,
        "n_valid_comparisons": 4,
        "n_units_with_valid_comparison": 1,
    }
    assert list(table.columns) == list(tuning_similarity.TABLE_COLUMNS)
    assert table.groupby("stable_unit_id", sort=False).size().to_dict() == {
        "merge-a:11": 4,
        "merge-b:22": 4,
    }
    assert table.groupby("stable_unit_id", sort=False)[
        "comparison_label"
    ].agg(list).to_dict() == {
        "merge-a:11": list(tuning_similarity.DIRECT_COMPARISON_LABELS),
        "merge-b:22": list(tuning_similarity.DIRECT_COMPARISON_LABELS),
    }
    assert table["group_unit_id"].drop_duplicates().tolist() == [
        "curve-unit-a",
        "curve-unit-b",
    ]
    assert table.groupby("stable_unit_id")["movement_firing_rate_hz"].first().to_dict() == {
        "merge-a:11": 1.5,
        "merge-b:22": 0.5,
    }
    valid = table[table["stable_unit_id"] == "merge-a:11"]
    assert np.allclose(valid["similarity"], 1.0)
    assert valid["similarity_status"].eq("valid").all()
    constant = table[table["stable_unit_id"] == "merge-b:22"]
    assert constant["similarity"].isna().all()
    assert constant["similarity_status"].eq("nonfinite_similarity").all()
    assert constant["n_paired_finite_bins"].eq(4).all()
    assert not table["comparison_label"].astype(str).str.startswith("pooled").any()


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("subset", "all-trial"),
        ("identity", "ordered unit identities"),
        ("position", "exactly matching"),
        ("fraction", "exactly matching"),
    ],
)
def test_compute_rejects_mismatched_curve_inputs(mutation: str, match: str) -> None:
    curves = _curves()
    trajectory = "right_to_center"
    if mutation == "subset":
        curves[trajectory].attrs["trial_subset"] = "odd"
    elif mutation == "identity":
        curves[trajectory] = curves[trajectory].isel(unit=[1, 0])
    elif mutation == "position":
        curves[trajectory] = _canonical_curve(
            trajectory,
            centers_cm=np.asarray([1.1, 3.1, 5.1, 7.1]),
        )
    else:
        curves[trajectory].attrs["graph_length_cm"] = 9.0
        curves[trajectory].coords["path_fraction"] = (
            "linear_position_cm",
            np.asarray([1.0, 3.0, 5.0, 7.0]) / 9.0,
            {"units": "1"},
        )

    with pytest.raises(ValueError, match=match):
        tuning_similarity.compute_tuning_similarity_from_curves(
            tuning_curves_by_trajectory=curves,
            movement_firing_rate_table=_movement_table(),
            similarity_metric="correlation",
        )


def test_validator_requires_complete_direct_schema_and_qc_consistency() -> None:
    table = _computed_table()
    tuning_similarity.validate_tuning_similarity_table(table)

    with pytest.raises(ValueError, match="exact canonical schema"):
        tuning_similarity.validate_tuning_similarity_table(
            table.drop(columns=["n_paired_finite_bins"])
        )
    pooled = table.copy()
    pooled.loc[0, "comparison_label"] = "pooled_same_turn"
    with pytest.raises(ValueError, match="each direct comparison once"):
        tuning_similarity.validate_tuning_similarity_table(pooled)
    invalid_qc = table.copy()
    invalid_qc.loc[0, "similarity_status"] = "nonfinite_similarity"
    with pytest.raises(ValueError, match="Finite similarity values"):
        tuning_similarity.validate_tuning_similarity_table(invalid_qc)


def test_legacy_normalization_maps_units_rates_and_matches_recomputation() -> None:
    legacy = _legacy_table()
    extra = legacy[legacy["unit"] == 101].copy().assign(unit=303)
    legacy = pd.concat([legacy, extra], ignore_index=True)
    normalized = tuning_similarity.normalize_legacy_all_units_similarity_table(
        legacy,
        tuning_curves_by_trajectory=_curves(),
        movement_firing_rate_table=_movement_table(),
        similarity_metric="correlation",
        unit_identity_resolver=_resolver(),
    )

    pd.testing.assert_frame_equal(normalized, _computed_table())
    assert normalized["group_unit_id"].drop_duplicates().tolist() == [
        "curve-unit-a",
        "curve-unit-b",
    ]

    changed = _legacy_table()
    changed.loc[0, "similarity"] = 0.25
    with pytest.raises(ValueError, match="'similarity'.*selected curves"):
        tuning_similarity.normalize_legacy_all_units_similarity_table(
            changed,
            tuning_curves_by_trajectory=_curves(),
            movement_firing_rate_table=_movement_table(),
            similarity_metric="correlation",
            unit_identity_resolver=_resolver(),
        )


def test_input_cross_check_rejects_an_artifact_from_another_selection() -> None:
    table = _computed_table()
    tuning_similarity.validate_tuning_similarity_against_inputs(
        table,
        tuning_curves_by_trajectory=_curves(),
        movement_firing_rate_table=_movement_table(),
        similarity_metric="correlation",
    )

    wrong_epoch = table.assign(epoch="08_r4")
    with pytest.raises(ValueError, match="selected upstream"):
        tuning_similarity.validate_tuning_similarity_against_inputs(
            wrong_epoch,
            tuning_curves_by_trajectory=_curves(),
            movement_firing_rate_table=_movement_table(),
            similarity_metric="correlation",
        )


def test_empty_selection_filters_unselected_legacy_units() -> None:
    normalized = tuning_similarity.normalize_legacy_all_units_similarity_table(
        _legacy_table(),
        tuning_curves_by_trajectory=_empty_curves(),
        movement_firing_rate_table=movement.empty_movement_firing_rate_table(),
        similarity_metric="correlation",
        unit_identity_resolver={},
    )

    pd.testing.assert_frame_equal(
        normalized,
        tuning_similarity.empty_tuning_similarity_table(),
    )


def test_register_existing_requires_all_units_and_writes_canonical_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy_all_units.parquet"
    destination = tmp_path / "canonical" / "similarity.parquet"
    _legacy_table().to_parquet(source, index=False)

    result = tuning_similarity.register_existing_tuning_similarity_artifact(
        source_path=source,
        destination_path=destination,
        tuning_curves_by_trajectory=_curves(),
        movement_firing_rate_table=_movement_table(),
        similarity_metric="correlation",
        unit_identity_resolver=_resolver(),
    )

    assert result["similarity_path"] == destination
    assert result["n_units"] == 2
    assert result["n_valid_comparisons"] == 4
    assert result["_created_artifact_paths"] == [str(destination)]
    assert len(result["legacy_artifact_provenance"]["source_sha256"]) == 64
    pd.testing.assert_frame_equal(
        tuning_similarity.load_tuning_similarity_artifact(destination),
        _computed_table(),
    )

    wrong_name = tmp_path / "legacy.parquet"
    _legacy_table().to_parquet(wrong_name, index=False)
    with pytest.raises(ValueError, match=r"\*_all_units"):
        tuning_similarity.register_existing_tuning_similarity_artifact(
            source_path=wrong_name,
            destination_path=tmp_path / "unused.parquet",
            tuning_curves_by_trajectory=_curves(),
            movement_firing_rate_table=_movement_table(),
            similarity_metric="correlation",
            unit_identity_resolver=_resolver(),
        )


def test_registration_hashes_source_before_writing_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "legacy_all_units.parquet"
    destination = tmp_path / "canonical" / "similarity.parquet"
    _legacy_table().to_parquet(source, index=False)

    def fail_hash(_path: Path) -> str:
        raise OSError("hash failed")

    monkeypatch.setattr(
        tuning_similarity,
        "_file_sha256",
        fail_hash,
    )

    with pytest.raises(OSError, match="hash failed"):
        tuning_similarity.register_existing_tuning_similarity_artifact(
            source_path=source,
            destination_path=destination,
            tuning_curves_by_trajectory=_curves(),
            movement_firing_rate_table=_movement_table(),
            similarity_metric="correlation",
            unit_identity_resolver=_resolver(),
        )

    assert not destination.exists()


def test_parquet_write_is_atomic_and_refuses_implicit_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "nested" / "similarity.parquet"
    original = _computed_table()
    replacement = original.copy()
    replacement["movement_firing_rate_hz"] = replacement[
        "movement_firing_rate_hz"
    ] + 1.0

    assert tuning_similarity.write_tuning_similarity_artifact(original, path) == path
    pd.testing.assert_frame_equal(
        tuning_similarity.load_tuning_similarity_artifact(path),
        original,
    )
    assert not list(path.parent.glob(f".{path.name}.*"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        tuning_similarity.write_tuning_similarity_artifact(replacement, path)

    real_replace = os.replace
    replace_count = 0

    def fail_new_artifact_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(tuning_similarity.os, "replace", fail_new_artifact_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        tuning_similarity.write_tuning_similarity_artifact(
            replacement,
            path,
            overwrite=True,
        )

    pd.testing.assert_frame_equal(
        tuning_similarity.load_tuning_similarity_artifact(path),
        original,
    )
    assert not list(path.parent.glob(f".{path.name}.*"))
