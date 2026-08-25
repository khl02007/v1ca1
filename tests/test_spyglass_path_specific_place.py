"""Tests for the database-free path-specific place adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np
import pytest
import xarray as xr

from v1ca1.spyglass import path_specific_place as place


GRAPH_INPUTS = {
    "configuration_name": "center_to_left",
    "coordinate_unit": "cm",
    "track_graph_kwargs": {
        "node_positions": [[0.0, 0.0], [10.0, 0.0]],
        "edges": [[0, 1]],
    },
    "linearization_kwargs": {
        "edge_order": [[0, 1]],
        "edge_spacing": [],
    },
}
STABLE_UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    {"spikesorting_merge_id": "merge-b", "unit_id": 22},
)


def _intervals():
    """Return three one-indexed trials in seconds."""
    import pynapple as nap

    return nap.IntervalSet(
        start=np.asarray([0.0, 1.0, 2.0]),
        end=np.asarray([0.8, 1.8, 2.8]),
        time_units="s",
    )


def _spikes():
    """Return two units with known odd-trial spike counts."""
    import pynapple as nap

    return nap.TsGroup(
        {
            0: nap.Ts(np.asarray([0.2, 1.2, 2.2]), time_units="s"),
            1: nap.Ts(np.asarray([0.4, 2.4]), time_units="s"),
        },
        time_support=nap.IntervalSet(start=0.0, end=3.0, time_units="s"),
        time_units="s",
    )


def _legacy_curve() -> xr.DataArray:
    """Return one three-unit legacy all-trial place curve."""
    return xr.DataArray(
        np.asarray([[3.0, 3.5], [7.0, 7.5], [9.0, 9.5]]),
        dims=("unit", "linpos"),
        coords={"unit": [3, 7, 9], "linpos": [2.5, 7.5]},
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "02_r1",
            "trajectory_type": "center_to_left",
            "model_name": "place",
            "bin_edges": json.dumps([[0.0, 5.0, 10.0]]),
        },
    )


def _legacy_mapping() -> dict[int, dict[str, object]]:
    """Select and reorder two legacy units into sorted-spike group order."""
    return {
        7: {
            "spikesorting_merge_id": "merge-b",
            "unit_id": 22,
            "group_unit_id": 1,
            "sorting_unit_id": 7,
        },
        3: {
            "spikesorting_merge_id": "merge-a",
            "unit_id": 11,
            "group_unit_id": 0,
            "sorting_unit_id": 3,
        },
    }


def _normalize_legacy() -> xr.DataArray:
    """Return one canonical curve for I/O tests."""
    return place.normalize_legacy_all_trial_tuning_curve(
        _legacy_curve(),
        unit_identity_resolver=_legacy_mapping(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        graph_length_cm=10.0,
        n_trials=3,
        support_duration_s=2.4,
        n_feature_samples=24,
        n_valid_position_samples=20,
        bin_count=2,
    )


def test_parameters_require_exactly_one_valid_binning_mode() -> None:
    assert place.validate_binning_parameters(bin_size_cm=4.0) == {
        "binning_mode": "bin_size_cm",
        "bin_size_cm": 4.0,
        "bin_count": None,
        "sigma_bins": 0.0,
    }
    np.testing.assert_allclose(
        place.build_position_bin_edges(10.0, bin_count=4),
        np.linspace(0.0, 10.0, 5),
    )
    np.testing.assert_allclose(
        place.build_position_bin_edges(10.0, bin_size_cm=4.0),
        np.asarray([0.0, 4.0, 8.0, 12.0]),
    )

    with pytest.raises(ValueError, match="Exactly one"):
        place.validate_binning_parameters()
    with pytest.raises(ValueError, match="Exactly one"):
        place.validate_binning_parameters(bin_size_cm=2.0, bin_count=5)
    with pytest.raises(ValueError, match="non-negative"):
        place.validate_binning_parameters(bin_count=5, sigma_bins=-0.1)


def test_trial_subsets_use_one_indexed_odd_even_order() -> None:
    all_trials = place.select_trial_subset_intervals(_intervals(), "all")
    odd_trials = place.select_trial_subset_intervals(_intervals(), "odd")
    even_trials = place.select_trial_subset_intervals(_intervals(), "even")

    np.testing.assert_allclose(np.asarray(all_trials.start), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(np.asarray(odd_trials.start), [0.0, 2.0])
    np.testing.assert_allclose(np.asarray(even_trials.start), [1.0])


def test_graph_contract_and_artifact_path_are_explicit(tmp_path: Path) -> None:
    selection_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    assert place.graph_length_from_inputs(
        GRAPH_INPUTS,
        trajectory_type="center_to_left",
    ) == pytest.approx(10.0)
    assert place.get_path_specific_place_artifact_path(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        trajectory_type="center_to_left",
        trial_subset="odd",
        region="v1",
        path_specific_place_tuning_curve_id=selection_id,
        artifact_root=tmp_path,
    ) == (
        tmp_path
        / "L14"
        / "20240611"
        / "path_specific_place_tuning_curve"
        / "02_r1"
        / "center_to_left"
        / "odd"
        / "v1"
        / str(selection_id)
        / "tuning_curve.nc"
    )

    bad_graph = {**GRAPH_INPUTS, "configuration_name": "center_to_right"}
    with pytest.raises(ValueError, match="configuration_name"):
        place.graph_length_from_inputs(
            bad_graph,
            trajectory_type="center_to_left",
        )


def test_build_linear_position_reuses_graph_progression(monkeypatch) -> None:
    import pynapple as nap
    from v1ca1.spyglass import stability

    progression = nap.Tsd(
        t=np.asarray([0.1, 0.2]),
        d=np.asarray([0.25, 0.75]),
        time_support=nap.IntervalSet(start=0.0, end=0.8, time_units="s"),
        time_units="s",
    )
    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        lambda **_kwargs: (progression, 20.0),
    )

    linear_position, graph_length = place.build_path_specific_linear_position(
        position=object(),
        trajectory_intervals=object(),
        graph_inputs=GRAPH_INPUTS,
        trajectory_type="center_to_left",
    )

    assert graph_length == pytest.approx(20.0)
    np.testing.assert_allclose(np.asarray(linear_position.d), [5.0, 15.0])


def test_compute_preserves_all_units_and_subset_metadata(monkeypatch) -> None:
    import pynapple as nap

    times = np.arange(0.05, 2.8, 0.1)
    linear_position = nap.Tsd(
        t=times,
        d=np.mod(times, 1.0) * 10.0,
        time_support=_intervals(),
        time_units="s",
    )
    monkeypatch.setattr(
        place,
        "build_path_specific_linear_position",
        lambda **_kwargs: (linear_position, 10.0),
    )
    movement = nap.IntervalSet(start=0.0, end=2.8, time_units="s")

    result = place.compute_selected_path_specific_place_tuning_curve(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        trial_subset="odd",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        position=object(),
        trajectory_intervals=_intervals(),
        graph_inputs=GRAPH_INPUTS,
        movement_intervals=movement,
        movement_analysis_status="valid",
        bin_count=5,
        sigma_bins=0.0,
    )

    curve = result["tuning_curve"]
    assert result["analysis_status"] == "valid"
    assert result["n_units"] == 2
    assert result["n_trials"] == 2
    assert result["support_duration_s"] == pytest.approx(1.6)
    assert result["n_feature_samples"] == 16
    assert curve.dims == ("unit", "linear_position_cm")
    assert curve.coords["stable_unit_id"].values.tolist() == [
        "merge-a:11",
        "merge-b:22",
    ]
    assert curve.coords["spike_count"].values.tolist() == [2, 2]
    np.testing.assert_allclose(
        curve.coords["path_fraction"].values,
        curve.coords["linear_position_cm"].values / 10.0,
    )
    assert curve.attrs["n_feature_samples"] == 16
    assert curve.attrs["n_valid_position_samples"] == 16
    place.validate_path_specific_place_tuning_curve(curve)


@pytest.mark.parametrize("movement_status", ["no_valid_position", "no_movement"])
def test_compute_propagates_terminal_movement_status_without_linearizing(
    monkeypatch,
    movement_status: str,
) -> None:
    import pynapple as nap

    monkeypatch.setattr(
        place,
        "build_path_specific_linear_position",
        lambda **_kwargs: pytest.fail("terminal movement result must not linearize"),
    )
    empty_movement = nap.IntervalSet(
        start=np.asarray([], dtype=float),
        end=np.asarray([], dtype=float),
        time_units="s",
    )

    result = place.compute_selected_path_specific_place_tuning_curve(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        trial_subset="all",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        position=object(),
        trajectory_intervals=_intervals(),
        graph_inputs=GRAPH_INPUTS,
        movement_intervals=empty_movement,
        movement_analysis_status=movement_status,
        bin_count=5,
    )

    assert result["analysis_status"] == movement_status
    assert result["n_feature_samples"] == 0
    assert np.isnan(result["tuning_curve"].values).all()
    assert result["tuning_curve"].coords["spike_count"].values.tolist() == [0, 0]


def test_smoothing_uses_existing_nan_aware_helper() -> None:
    from v1ca1.raster.plot_place_field_heatmap import smooth_values_nan_aware

    values = np.asarray([[0.0, np.nan, 1.0, 0.0]])
    expected = smooth_values_nan_aware(
        values,
        sigma_bins=1.0,
        axis=1,
        mode="nearest",
    )

    np.testing.assert_allclose(
        place.smooth_tuning_values(values, sigma_bins=1.0),
        expected,
        equal_nan=True,
    )


def test_legacy_normalization_selects_and_reorders_units() -> None:
    curve = _normalize_legacy()

    assert curve.coords["stable_unit_id"].values.tolist() == [
        "merge-b:22",
        "merge-a:11",
    ]
    assert curve.coords["group_unit_id"].values.tolist() == ["1", "0"]
    np.testing.assert_allclose(curve.values, [[7.0, 7.5], [3.0, 3.5]])
    assert np.isnan(curve.coords["spike_count"].values).all()
    assert curve.attrs["trial_subset"] == "all"
    assert curve.attrs["legacy_normalized"] == "true"
    assert curve.attrs["n_feature_samples"] == 24
    assert curve.attrs["n_valid_position_samples"] == 20

    missing_mapping = {
        **_legacy_mapping(),
        99: {"spikesorting_merge_id": "merge-c", "unit_id": 33},
    }
    with pytest.raises(ValueError, match="exactly once"):
        place.normalize_legacy_all_trial_tuning_curve(
            _legacy_curve(),
            unit_identity_resolver=missing_mapping,
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            trajectory_type="center_to_left",
            graph_length_cm=10.0,
            n_trials=3,
            support_duration_s=2.4,
            n_feature_samples=24,
            bin_count=2,
        )


def test_validator_rejects_status_and_support_metadata_disagreement() -> None:
    curve = _normalize_legacy()
    curve.attrs["n_trials"] = 0

    with pytest.raises(ValueError, match="Valid tuning curves require"):
        place.validate_path_specific_place_tuning_curve(curve)


def test_register_existing_normalizes_to_a_new_canonical_artifact(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.nc"
    destination = tmp_path / "canonical" / "tuning_curve.nc"
    _legacy_curve().to_netcdf(source)

    result = place.register_existing_path_specific_place_artifact(
        source_path=source,
        destination_path=destination,
        unit_identity_resolver=_legacy_mapping(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        trajectory_type="center_to_left",
        graph_length_cm=10.0,
        n_trials=3,
        support_duration_s=2.4,
        n_feature_samples=24,
        bin_count=2,
        artifact_attributes={"path_specific_place_tuning_curve_id": "curve-1"},
    )

    assert result["tuning_curve_path"] == destination
    assert result["n_units"] == 2
    assert result["n_feature_samples"] == 24
    assert result["_created_artifact_paths"] == [str(destination)]
    assert len(result["legacy_artifact_provenance"]["source_sha256"]) == 64
    loaded = place.load_path_specific_place_artifact(destination)
    assert loaded.attrs["path_specific_place_tuning_curve_id"] == "curve-1"
    np.testing.assert_allclose(
        loaded.values,
        [[7.0, 7.5], [3.0, 3.5]],
    )


@pytest.mark.parametrize("empty_units", [False, True])
def test_three_nwb_tables_roundtrip_curve_and_axis_metadata(
    tmp_path: Path,
    empty_units: bool,
) -> None:
    """Unit rows bind identities/counts to vectors while bins define columns."""
    from datetime import datetime, timezone

    from pynwb import NWBFile, NWBHDF5IO

    curve = _normalize_legacy()
    if empty_units:
        curve = curve.isel(unit=slice(0, 0)).copy()
        curve.attrs.update(
            n_units=0,
            n_valid_units=0,
            analysis_status="no_units",
        )
        place.validate_path_specific_place_tuning_curve(curve)

    tuning = place.path_specific_place_tuning_to_dynamic_table(curve)
    bins = place.path_specific_place_bins_to_dynamic_table(curve)
    provenance = place.path_specific_place_provenance_to_dynamic_table(curve)
    assert tuning.name == place.NWB_TUNING_TABLE_NAME
    assert bins.name == place.NWB_BINS_TABLE_NAME
    assert provenance.name == place.NWB_PROVENANCE_TABLE_NAME
    if not empty_units:
        tuning_frame = tuning.to_dataframe()
        assert tuning_frame.loc[0, "stable_unit_id"] == "merge-b:22"
        assert np.isnan(tuning_frame.loc[0, "spike_count"])
        np.testing.assert_allclose(
            tuning_frame.loc[0, "firing_rate_hz"],
            [7.0, 7.5],
        )

    path = tmp_path / "path-specific-place.nwb"
    nwbfile = NWBFile(
        session_description="Path-specific place NWB test",
        identifier="path-specific-place-test",
        session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    object_ids = {}
    for name, nwb_object in {
        "tuning": tuning,
        "bins": bins,
        "provenance": provenance,
    }.items():
        nwbfile.add_scratch(nwb_object)
        object_ids[name] = nwb_object.object_id
    with NWBHDF5IO(str(path), mode="w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(str(path), mode="r", load_namespaces=True) as io:
        stored = io.read()
        roundtrip = place.path_specific_place_tuning_curve_from_nwb_objects(
            stored.objects[object_ids["tuning"]],
            stored.objects[object_ids["bins"]],
            stored.objects[object_ids["provenance"]],
        )

    xr.testing.assert_identical(roundtrip, curve)
    assert place.path_specific_place_tuning_sha256(roundtrip) == (
        place.path_specific_place_tuning_sha256(curve)
    )
    assert place.path_specific_place_bins_sha256(roundtrip) == (
        place.path_specific_place_bins_sha256(curve)
    )
    assert place.path_specific_place_provenance_sha256(roundtrip) == (
        place.path_specific_place_provenance_sha256(curve)
    )


def test_netcdf_write_is_atomic_and_refuses_implicit_overwrite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original = _normalize_legacy()
    replacement = original.copy(data=original.values + 1.0)
    path = tmp_path / "artifact" / "tuning_curve.nc"

    assert place.write_path_specific_place_artifact(original, path) == path
    loaded = place.load_path_specific_place_artifact(path)
    xr.testing.assert_identical(loaded, original)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        place.write_path_specific_place_artifact(replacement, path)

    real_replace = os.replace
    replace_count = 0

    def fail_new_artifact_replace(source, destination):
        nonlocal replace_count
        replace_count += 1
        if replace_count == 2:
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(place.os, "replace", fail_new_artifact_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        place.write_path_specific_place_artifact(
            replacement,
            path,
            overwrite=True,
        )

    xr.testing.assert_identical(place.load_path_specific_place_artifact(path), original)
    assert not list(path.parent.glob(f".{path.name}.*"))
