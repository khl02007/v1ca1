"""Tests for the database-free directional path-progression adapter."""

from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np
import pytest
import xarray as xr

from v1ca1.spyglass import dpp


def _graph(trajectory_type: str, *, length: float = 10.0) -> dict[str, object]:
    """Return one minimal trajectory graph with a known ordered length."""
    return {
        "configuration_name": trajectory_type,
        "coordinate_unit": "cm",
        "track_graph_kwargs": {
            "node_positions": [[0.0, 0.0], [length, 0.0]],
            "edges": [[0, 1]],
        },
        "linearization_kwargs": {
            "edge_order": [[0, 1]],
            "edge_spacing": [],
        },
    }


def _left_graphs(*, inbound_length: float = 10.0) -> dict[str, dict[str, object]]:
    """Return the fixed left-turn outbound and inbound graph rows."""
    return {
        "center_to_left": _graph("center_to_left"),
        "right_to_center": _graph(
            "right_to_center",
            length=inbound_length,
        ),
    }


def _intervals(starts: list[float]):
    """Return fixed 0.8-second trial intervals."""
    import pynapple as nap

    starts_array = np.asarray(starts, dtype=float)
    return nap.IntervalSet(
        start=starts_array,
        end=starts_array + 0.8,
        time_units="s",
    )


def _left_intervals() -> dict[str, object]:
    """Return interleaved outbound and inbound trials."""
    return {
        "center_to_left": _intervals([0.0, 2.0, 4.0]),
        "right_to_center": _intervals([1.0, 3.0]),
    }


def _spikes():
    """Return two units with known pooled odd-trial spike counts."""
    import pynapple as nap

    return nap.TsGroup(
        {
            10: nap.Ts(np.asarray([0.15, 1.15, 4.15]), time_units="s"),
            20: nap.Ts(np.asarray([0.25, 3.15, 4.25]), time_units="s"),
        },
        time_support=nap.IntervalSet(start=0.0, end=5.0, time_units="s"),
        time_units="s",
    )


STABLE_UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    {"spikesorting_merge_id": "merge-b", "unit_id": 22},
)


def _legacy_curve() -> xr.DataArray:
    """Return one legacy all-trial, same-turn task-progression curve."""
    return xr.DataArray(
        np.asarray([[3.0, 3.5], [7.0, 7.5], [9.0, 9.5]]),
        dims=("unit", "tp"),
        coords={"unit": [3, 7, 9], "tp": [0.25, 0.75]},
        name="firing_rate_hz",
        attrs={
            "animal_name": "L14",
            "date": "20240611",
            "region": "v1",
            "epoch": "02_r1",
            "model_name": "task_progression",
            "turn_type": "left",
            "bin_edges": json.dumps([[0.0, 0.5, 1.0]]),
        },
    )


def _legacy_mapping() -> dict[int, dict[str, object]]:
    """Select and reorder two legacy sorting units."""
    return {
        7: {
            "spikesorting_merge_id": "merge-b",
            "unit_id": 22,
            "group_unit_id": 20,
            "sorting_unit_id": 7,
        },
        3: {
            "spikesorting_merge_id": "merge-a",
            "unit_id": 11,
            "group_unit_id": 10,
            "sorting_unit_id": 3,
        },
    }


def _legacy_source_metadata() -> dict[str, object]:
    """Return source-derived metadata required for strict registration."""
    return {
        "common_graph_length_cm": 10.0,
        "graph_length_cm_by_trajectory": {
            "center_to_left": 10.0,
            "right_to_center": 10.0,
        },
        "n_trials_by_trajectory": {
            "center_to_left": 2,
            "right_to_center": 1,
        },
        "support_duration_s_by_trajectory": {
            "center_to_left": 1.6,
            "right_to_center": 0.8,
        },
        "n_feature_samples_by_trajectory": {
            "center_to_left": 16,
            "right_to_center": 8,
        },
        "n_valid_position_samples_by_trajectory": {
            "center_to_left": 15,
            "right_to_center": 8,
        },
    }


def _normalize_legacy() -> xr.DataArray:
    """Return one canonical legacy-normalized DPP curve."""
    return dpp.normalize_legacy_all_trial_dpp_tuning_curve(
        _legacy_curve(),
        unit_identity_resolver=_legacy_mapping(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        turn_type="left",
        bin_size_cm=5.0,
        **_legacy_source_metadata(),
    )


def test_turn_pairs_and_session_first_artifact_path(tmp_path: Path) -> None:
    selection_id = uuid.UUID("12345678-1234-5678-1234-567812345678")

    assert dpp.get_dpp_trajectory_pair("left") == (
        "center_to_left",
        "right_to_center",
    )
    assert dpp.get_dpp_trajectory_pair("right") == (
        "center_to_right",
        "left_to_center",
    )
    assert dpp.get_dpp_artifact_path(
        animal_name="L14",
        date="20240611",
        epoch="02_r1",
        turn_type="left",
        trial_subset="odd",
        region="v1",
        dpp_tuning_curve_id=selection_id,
        artifact_root=tmp_path,
    ) == (
        tmp_path
        / "L14"
        / "20240611"
        / "dpp_tuning_curve"
        / "02_r1"
        / "left"
        / "odd"
        / "v1"
        / str(selection_id)
        / "tuning_curve.nc"
    )

    with pytest.raises(ValueError, match="turn_type"):
        dpp.validate_turn_type("outbound")


def test_common_graph_length_controls_normalized_binning() -> None:
    common, lengths = dpp.common_graph_length_from_inputs(
        _left_graphs(),
        turn_type="left",
    )

    assert common == pytest.approx(10.0)
    assert lengths == {
        "center_to_left": pytest.approx(10.0),
        "right_to_center": pytest.approx(10.0),
    }
    np.testing.assert_allclose(
        dpp.build_dpp_bin_edges(common, bin_size_cm=4.0),
        [0.0, 0.4, 0.8, 1.2],
    )
    np.testing.assert_allclose(
        dpp.build_dpp_bin_edges(common, bin_count=4),
        np.linspace(0.0, 1.0, 5),
    )

    with pytest.raises(ValueError, match="common path length"):
        dpp.common_graph_length_from_inputs(
            _left_graphs(inbound_length=12.0),
            turn_type="left",
        )


def test_odd_even_trials_are_split_within_each_source_before_pooling() -> None:
    odd_by_source, odd = dpp.select_dpp_trial_intervals(
        _left_intervals(),
        turn_type="left",
        trial_subset="odd",
    )
    even_by_source, even = dpp.select_dpp_trial_intervals(
        _left_intervals(),
        turn_type="left",
        trial_subset="even",
    )

    np.testing.assert_allclose(
        np.asarray(odd_by_source["center_to_left"].start),
        [0.0, 4.0],
    )
    np.testing.assert_allclose(
        np.asarray(odd_by_source["right_to_center"].start),
        [1.0],
    )
    np.testing.assert_allclose(np.asarray(odd.start), [0.0, 1.0, 4.0])
    np.testing.assert_allclose(
        np.asarray(even_by_source["center_to_left"].start),
        [2.0],
    )
    np.testing.assert_allclose(
        np.asarray(even_by_source["right_to_center"].start),
        [3.0],
    )
    np.testing.assert_allclose(np.asarray(even.start), [2.0, 3.0])


def test_compute_pools_raw_source_samples_before_one_tuning_estimate(
    monkeypatch,
) -> None:
    import pynapple as nap

    progressions = {
        "center_to_left": nap.Tsd(
            t=np.asarray([0.1, 0.2, 4.1, 4.2]),
            d=np.asarray([0.1, 0.2, 0.7, 0.8]),
            time_support=_intervals([0.0, 4.0]),
            time_units="s",
        ),
        "right_to_center": nap.Tsd(
            t=np.asarray([1.1, 1.2]),
            d=np.asarray([0.3, 0.4]),
            time_support=_intervals([1.0]),
            time_units="s",
        ),
    }
    monkeypatch.setattr(
        dpp,
        "_build_trajectory_progression",
        lambda **kwargs: (progressions[kwargs["trajectory_type"]], 10.0),
    )
    raw_calls: list[dict[str, object]] = []

    def fake_raw(*, spikes, dpp: object, support, bin_edges_dpp):
        raw_calls.append(
            {
                "times": np.asarray(dpp.t, dtype=float),
                "values": np.asarray(dpp.d, dtype=float),
                "support_starts": np.asarray(support.start, dtype=float),
            }
        )
        centers = (bin_edges_dpp[:-1] + bin_edges_dpp[1:]) / 2.0
        return xr.DataArray(
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            dims=("unit", "dpp"),
            coords={"unit": [10, 20], "dpp": centers},
        )

    monkeypatch.setattr(dpp, "_raw_tuning_curve", fake_raw)
    movement = nap.IntervalSet(start=0.0, end=5.0, time_units="s")

    result = dpp.compute_selected_dpp_tuning_curve(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        turn_type="left",
        trial_subset="odd",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        position=object(),
        trajectory_intervals_by_type=_left_intervals(),
        graph_inputs_by_trajectory=_left_graphs(),
        movement_intervals=movement,
        bin_count=2,
    )

    assert len(raw_calls) == 1
    np.testing.assert_allclose(
        raw_calls[0]["times"],
        [0.1, 0.2, 1.1, 1.2, 4.1, 4.2],
    )
    np.testing.assert_allclose(
        raw_calls[0]["values"],
        [0.1, 0.2, 0.3, 0.4, 0.7, 0.8],
    )
    np.testing.assert_allclose(raw_calls[0]["support_starts"], [0.0, 1.0, 4.0])
    curve = result["tuning_curve"]
    assert result["analysis_status"] == "valid"
    assert result["n_outbound_trials"] == 2
    assert result["n_inbound_trials"] == 1
    assert result["n_trials"] == 3
    assert result["support_duration_s"] == pytest.approx(2.4)
    assert result["n_feature_samples"] == 6
    assert curve.attrs["outbound_trajectory_type"] == "center_to_left"
    assert curve.attrs["inbound_trajectory_type"] == "right_to_center"
    assert curve.coords["stable_unit_id"].values.tolist() == [
        "merge-a:11",
        "merge-b:22",
    ]
    assert curve.coords["spike_count"].values.tolist() == [3, 2]
    np.testing.assert_allclose(curve.coords["path_fraction"], [0.25, 0.75])
    np.testing.assert_allclose(curve.coords["linear_position_cm"], [2.5, 7.5])
    dpp.validate_dpp_tuning_curve(curve)


def test_terminal_movement_status_does_not_linearize(monkeypatch) -> None:
    import pynapple as nap

    monkeypatch.setattr(
        dpp,
        "_build_trajectory_progression",
        lambda **_kwargs: pytest.fail("terminal movement result must not linearize"),
    )
    empty = nap.IntervalSet(
        start=np.asarray([], dtype=float),
        end=np.asarray([], dtype=float),
        time_units="s",
    )

    result = dpp.compute_selected_dpp_tuning_curve(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        turn_type="left",
        trial_subset="all",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        position=object(),
        trajectory_intervals_by_type=_left_intervals(),
        graph_inputs_by_trajectory=_left_graphs(),
        movement_intervals=empty,
        movement_analysis_status="no_movement",
        bin_count=2,
    )

    assert result["analysis_status"] == "no_movement"
    assert result["n_feature_samples"] == 0
    assert np.isnan(result["tuning_curve"].values).all()


def test_legacy_normalization_is_strict_and_preserves_source_metadata() -> None:
    curve = _normalize_legacy()

    assert curve.coords["stable_unit_id"].values.tolist() == [
        "merge-b:22",
        "merge-a:11",
    ]
    np.testing.assert_allclose(curve.values, [[7.0, 7.5], [3.0, 3.5]])
    assert np.isnan(curve.coords["spike_count"]).all()
    assert curve.attrs["trial_subset"] == "all"
    assert curve.attrs["legacy_normalized"] == "true"
    assert curve.attrs["n_outbound_trials"] == 2
    assert curve.attrs["n_inbound_trials"] == 1
    assert curve.attrs["n_valid_position_samples"] == 23

    wrong = _legacy_curve().copy()
    wrong.attrs["turn_type"] = "right"
    with pytest.raises(ValueError, match="turn_type"):
        dpp.normalize_legacy_all_trial_dpp_tuning_curve(
            wrong,
            unit_identity_resolver=_legacy_mapping(),
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="02_r1",
            turn_type="left",
            bin_size_cm=5.0,
            **_legacy_source_metadata(),
        )


def test_register_existing_writes_valid_canonical_artifact(tmp_path: Path) -> None:
    source = tmp_path / "legacy.nc"
    destination = tmp_path / "canonical" / "tuning_curve.nc"
    _legacy_curve().to_netcdf(source)

    result = dpp.register_existing_dpp_artifact(
        source_path=source,
        destination_path=destination,
        unit_identity_resolver=_legacy_mapping(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="02_r1",
        turn_type="left",
        bin_size_cm=5.0,
        artifact_attributes={"dpp_tuning_curve_id": "curve-1"},
        **_legacy_source_metadata(),
    )

    assert result["tuning_curve_path"] == destination
    assert result["n_outbound_trials"] == 2
    assert result["n_inbound_trials"] == 1
    assert result["_created_artifact_paths"] == [str(destination)]
    assert len(result["legacy_artifact_provenance"]["source_sha256"]) == 64
    loaded = dpp.load_dpp_artifact(destination)
    assert loaded.attrs["dpp_tuning_curve_id"] == "curve-1"
    np.testing.assert_allclose(loaded.values, [[7.0, 7.5], [3.0, 3.5]])


def test_validator_rejects_source_trial_aggregate_disagreement() -> None:
    curve = _normalize_legacy()
    curve.attrs["n_outbound_trials"] = 99

    with pytest.raises(ValueError, match="n_outbound_trials"):
        dpp.validate_dpp_tuning_curve(curve)
