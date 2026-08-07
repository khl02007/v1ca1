"""Tests for database-free four-model DPP encoding comparison artifacts."""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import encoding_comparison as encoding
from v1ca1.spyglass import movement


COMPARISON_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
STABLE_UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": 11},
    {"spikesorting_merge_id": "merge-b", "unit_id": 22},
    {"spikesorting_merge_id": "merge-c", "unit_id": 33},
)


def _spikes():
    """Return three units spanning train and test intervals."""
    import pynapple as nap

    return nap.TsGroup(
        {
            10: nap.Ts(np.asarray([0.2, 1.2, 2.2]), time_units="s"),
            20: nap.Ts(np.asarray([0.4, 1.4, 2.4]), time_units="s"),
            30: nap.Ts(np.asarray([0.6, 1.6, 2.6]), time_units="s"),
        },
        time_support=nap.IntervalSet(start=0.0, end=3.0, time_units="s"),
        time_units="s",
    )


def _movement_table() -> pd.DataFrame:
    """Return a canonical movement artifact for the three selected units."""
    rows = []
    rates = (0.5, 0.6, 1.0)
    for group_id, identity, rate in zip(
        ("old-a", "old-b", "old-c"),
        STABLE_UNIT_IDS,
        rates,
        strict=True,
    ):
        merge_id = str(identity["spikesorting_merge_id"])
        unit_id = str(identity["unit_id"])
        rows.append(
            {
                "spikesorting_merge_id": merge_id,
                "unit_id": unit_id,
                "stable_unit_id": f"{merge_id}:{unit_id}",
                "group_unit_id": group_id,
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "epoch": "08_r4",
                "movement_spike_count": int(rate * 10.0),
                "movement_duration_s": 10.0,
                "movement_firing_rate_hz": rate,
                "firing_rate_status": "valid",
                "position_sample_count": 100,
                "finite_position_sample_count": 100,
                "finite_speed_sample_count": 100,
                "movement_interval_count": 4,
                "speed_threshold_cm_s": 4.0,
                "speed_smoothing_sigma_s": 0.1,
            }
        )
    return pd.DataFrame.from_records(rows).loc[:, list(movement.MOVEMENT_TABLE_COLUMNS)]


def _stability_tables() -> dict[str, pd.DataFrame]:
    """Return four stability inputs with two units meeting the OR criterion."""
    correlations = {
        "center_to_left": (0.5, 0.49, np.nan),
        "left_to_center": (0.1, 0.49, np.nan),
        "center_to_right": (0.2, 0.49, 0.7),
        "right_to_center": (0.3, 0.49, 0.1),
    }
    rates = (0.5, 0.6, 1.0)
    tables = {}
    for trajectory_type in TRAJECTORY_TYPES:
        rows = []
        for group_id, identity, rate, correlation in zip(
            ("stability-a", "stability-b", "stability-c"),
            STABLE_UNIT_IDS,
            rates,
            correlations[trajectory_type],
            strict=True,
        ):
            merge_id = str(identity["spikesorting_merge_id"])
            unit_id = str(identity["unit_id"])
            rows.append(
                {
                    "spikesorting_merge_id": merge_id,
                    "unit_id": unit_id,
                    "stable_unit_id": f"{merge_id}:{unit_id}",
                    "group_unit_id": group_id,
                    "animal_name": "L14",
                    "date": "20240611",
                    "region": "v1",
                    "epoch": "08_r4",
                    "trajectory_type": trajectory_type,
                    "firing_rate_hz": rate,
                    "stability_correlation": correlation,
                    "stability_status": (
                        "valid" if np.isfinite(correlation) else "constant_curve"
                    ),
                }
            )
        tables[trajectory_type] = pd.DataFrame.from_records(rows)
    return tables


def _intervals(start: float, *, n_laps: int = 2):
    """Return non-overlapping 0.8-second laps."""
    import pynapple as nap

    starts = start + 8.0 * np.arange(n_laps, dtype=float)
    return nap.IntervalSet(
        start=starts,
        end=starts + 0.8,
        time_units="s",
    )


def _trajectory_intervals(*, n_laps: int = 2) -> dict[str, object]:
    """Return interleaved trajectory laps without overlaps."""
    return {
        trajectory_type: _intervals(float(index * 2), n_laps=n_laps)
        for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }


def _graph(configuration_name: str) -> dict[str, object]:
    """Return one minimal graph input mapping."""
    return {
        "configuration_name": configuration_name,
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


def _graphs() -> dict[str, dict[str, object]]:
    """Return the four path graphs and one full-W graph."""
    return {
        name: _graph(name)
        for name in (*TRAJECTORY_TYPES, encoding.FULL_W_CONFIGURATION_NAME)
    }


def _valid_store(*, heldout_spike_count: int = 4) -> dict[str, object]:
    """Return one internally consistent four-model fold store."""
    return {
        "heldout_spike_count": heldout_spike_count,
        "null_log_likelihood_nats": -12.0,
        "zero_training_spikes": False,
        "model_log_likelihood_nats": {
            "path_specific_place": -10.0,
            "absolute_place": -11.0,
            "dpp": -8.0,
            "distance_to_reward": -9.0,
        },
        "model_failed": {name: False for name in encoding.MODEL_NAMES},
    }


def _canonical_row() -> dict[str, object]:
    """Return one valid canonical artifact row."""
    metrics = encoding._unit_metric_row(store=_valid_store())
    return {
        "spikesorting_merge_id": "merge-a",
        "unit_id": "11",
        "stable_unit_id": "merge-a:11",
        "group_unit_id": "10",
        "dpp_encoding_comparison_id": str(COMPARISON_ID),
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "epoch": "08_r4",
        "n_folds": 5,
        "evaluation_bin_size_s": 0.05,
        "spatial_bin_size_cm": 4.0,
        "gaussian_smoothing_sigma_bins": 1.0,
        "random_seed": 47,
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": 0.5,
        "movement_firing_rate_hz": 0.5,
        "center_to_left_stability_correlation": 0.5,
        "left_to_center_stability_correlation": 0.1,
        "center_to_right_stability_correlation": 0.2,
        "right_to_center_stability_correlation": 0.3,
        **metrics,
    }


def _legacy_table(units: tuple[int, ...] = (101, 303)) -> pd.DataFrame:
    """Return a legacy per-spike table for the two eligible units."""
    rows = []
    for unit in units:
        ll_null = -3.0
        ll_place = -2.5
        ll_absolute = -2.75
        ll_dpp = -2.0
        ll_distance = -2.25
        rows.append(
            {
                "unit": unit,
                "n_spikes": 4,
                "ll_null": ll_null,
                "ll_place": ll_place,
                "ll_generalized_place": ll_absolute,
                "ll_tp": ll_dpp,
                "ll_gtp": ll_distance,
                "info_bits_place": (ll_place - ll_null) / np.log(2.0),
                "info_bits_generalized_place": (
                    ll_absolute - ll_null
                ) / np.log(2.0),
                "info_bits_tp": (ll_dpp - ll_null) / np.log(2.0),
                "info_bits_gtp": (ll_distance - ll_null) / np.log(2.0),
                "delta_bits_place_vs_tp": (ll_place - ll_dpp) / np.log(2.0),
                "delta_bits_generalized_place_vs_tp": (
                    ll_absolute - ll_dpp
                ) / np.log(2.0),
                "delta_bits_gtp_vs_tp": (
                    ll_distance - ll_dpp
                ) / np.log(2.0),
            }
        )
    return pd.DataFrame.from_records(rows).set_index("unit")


def _resolver() -> dict[int, dict[str, object]]:
    """Return legacy identities for the eligible first and third units."""
    return {
        101: {"spikesorting_merge_id": "merge-a", "unit_id": 11},
        303: {"spikesorting_merge_id": "merge-c", "unit_id": 33},
    }


def test_artifact_path_and_empty_summary_are_uuid_keyed(tmp_path: Path) -> None:
    path = encoding.get_encoding_comparison_artifact_path(
        animal_name="L14",
        date="20240611",
        epoch="08_r4",
        region="v1",
        dpp_encoding_comparison_id=COMPARISON_ID,
        artifact_root=tmp_path,
    )

    assert path == (
        tmp_path
        / "L14"
        / "20240611"
        / "dpp_encoding_comparison"
        / "08_r4"
        / "v1"
        / str(COMPARISON_ID)
        / "encoding_comparison.parquet"
    )
    empty = encoding.empty_encoding_comparison_table()
    assert list(empty.columns) == list(encoding.TABLE_COLUMNS)
    summary = encoding.summarize_encoding_comparison_table(empty)
    assert summary["analysis_status"] == "no_eligible_units"
    assert summary["n_units_eligible"] == 0
    assert summary["n_units_valid"] == 0
    assert len(summary["eligible_units_sha256"]) == 64


def test_eligibility_is_inclusive_firing_rate_and_any_trajectory_stability() -> None:
    table = encoding.build_encoding_eligibility_table(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        movement_firing_rate_table=_movement_table(),
        stability_tables_by_trajectory=_stability_tables(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
    )

    assert table["stable_unit_id"].tolist() == [
        "merge-a:11",
        "merge-b:22",
        "merge-c:33",
    ]
    assert table["eligible"].tolist() == [True, False, True]
    assert table["group_unit_id"].tolist() == ["10", "20", "30"]


def test_eligibility_rejects_mismatched_stability_firing_rate() -> None:
    tables = _stability_tables()
    tables["center_to_left"].loc[0, "firing_rate_hz"] = 99.0

    with pytest.raises(ValueError, match="disagree"):
        encoding.build_encoding_eligibility_table(
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="08_r4",
            spikes=_spikes(),
            stable_unit_ids=STABLE_UNIT_IDS,
            movement_firing_rate_table=_movement_table(),
            stability_tables_by_trajectory=tables,
            minimum_movement_firing_rate_hz=0.5,
            minimum_stability_correlation=0.5,
        )


def test_model_inputs_reproduce_legacy_four_coordinate_conventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pynapple as nap
    from v1ca1.spyglass import stability

    intervals = {
        trajectory_type: nap.IntervalSet(
            start=float(index * 2),
            end=float(index * 2 + 1),
            time_units="s",
        )
        for index, trajectory_type in enumerate(TRAJECTORY_TYPES)
    }
    movement_intervals = nap.IntervalSet(
        start=np.asarray([0.0, 2.0, 4.0, 6.0]),
        end=np.asarray([1.0, 3.0, 5.0, 7.0]),
        time_units="s",
    )

    def _fake_progression(*, trajectory_type: str, trajectory_interval, **_kwargs):
        if trajectory_type == "full_w":
            times = np.asarray([0.2, 0.8, 2.2, 2.8, 4.2, 4.8, 6.2, 6.8])
            values = np.linspace(0.1, 0.8, times.size)
            return (
                nap.Tsd(
                    t=times,
                    d=values,
                    time_support=trajectory_interval,
                    time_units="s",
                ),
                30.0,
            )
        index = TRAJECTORY_TYPES.index(trajectory_type)
        times = np.asarray([index * 2 + 0.2, index * 2 + 0.8])
        return (
            nap.Tsd(
                t=times,
                d=np.asarray([0.2, 0.8]),
                time_support=trajectory_interval,
                time_units="s",
            ),
            10.0,
        )

    monkeypatch.setattr(
        stability,
        "build_task_progression_from_graph",
        _fake_progression,
    )
    result = encoding.build_encoding_model_inputs(
        position=object(),
        trajectory_intervals_by_type=intervals,
        graph_inputs_by_configuration=_graphs(),
        movement_intervals=movement_intervals,
        spatial_bin_size_cm=4.0,
    )

    np.testing.assert_allclose(
        result["features"]["path_specific_place"].d,
        [2.0, 8.0, 18.0, 12.0, 22.0, 28.0, 38.0, 32.0],
    )
    np.testing.assert_allclose(
        result["features"]["dpp"].d,
        [0.2, 0.8, 1.2, 1.8, 1.2, 1.8, 0.2, 0.8],
    )
    np.testing.assert_allclose(
        result["features"]["distance_to_reward"].d,
        [0.2, 0.8] * 4,
    )
    np.testing.assert_allclose(
        result["features"]["absolute_place"].d,
        np.linspace(3.0, 24.0, 8),
    )
    np.testing.assert_allclose(
        result["bins"]["path_specific_place"],
        [
            0.0,
            4.0,
            8.0,
            10.0,
            14.0,
            18.0,
            20.0,
            24.0,
            28.0,
            30.0,
            34.0,
            38.0,
            40.0,
        ],
    )
    np.testing.assert_allclose(
        result["bins"]["dpp"],
        [0.0, 0.4, 0.8, 1.0, 1.4, 1.8, 2.0],
    )
    assert result["smoothing_boundaries_by_model"] == {
        "path_specific_place": (10.0, 20.0, 30.0),
        "absolute_place": (),
        "dpp": (1.0,),
        "distance_to_reward": (),
    }
    for model_name, boundaries in {
        "path_specific_place": (10.0, 20.0, 30.0),
        "dpp": (1.0,),
    }.items():
        edges = result["bins"][model_name]
        for boundary in boundaries:
            assert np.count_nonzero(edges == boundary) == 1
            assert not np.any(
                (edges[:-1] < boundary) & (edges[1:] > boundary)
            )


def test_heldout_finite_mask_divergence_raises() -> None:
    class _Counts:
        t = np.asarray([0.1, 0.2, 0.3])

    class _Interpolated:
        def __init__(self, values):
            self.t = _Counts.t
            self.d = np.asarray(values, dtype=float)

    class _Feature:
        def __init__(self, values):
            self.values = values

        def interpolate(self, *_args, **_kwargs):
            return _Interpolated(self.values)

    features = {
        model_name: _Feature([0.0, 0.5, 1.0])
        for model_name in encoding.MODEL_NAMES
    }
    features["absolute_place"] = _Feature([0.0, np.nan, 1.0])

    with pytest.raises(ValueError, match="finite masks diverge"):
        encoding._fold_feature_values(
            reference_counts=_Counts(),
            features=features,
            test_fold=object(),
        )


@pytest.mark.parametrize(
    ("boundaries", "values"),
    [
        ((3.0,), [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]),
        (
            (3.0, 6.0, 9.0),
            [0.0] * 3 + [10.0] * 3 + [20.0] * 3 + [30.0] * 3,
        ),
    ],
)
def test_block_smoothing_does_not_bleed_between_coordinates(
    boundaries: tuple[float, ...],
    values: list[float],
) -> None:
    coordinates = np.arange(len(values), dtype=float) + 0.5
    curve = xr.DataArray(
        np.asarray(values, dtype=float)[:, np.newaxis],
        dims=("position", "unit"),
        coords={"position": coordinates, "unit": [10]},
    )

    blocked = encoding._smooth_tuning_curve_in_blocks(
        curve,
        pos_dim="position",
        sigma_bins=1.0,
        block_boundaries=boundaries,
    )
    global_curve = encoding._smooth_tuning_curve_in_blocks(
        curve,
        pos_dim="position",
        sigma_bins=1.0,
        block_boundaries=(),
    )

    np.testing.assert_allclose(blocked.values[:, 0], values)
    assert not np.allclose(global_curve.values[:, 0], values)


def test_lap_count_and_strict_fold_movement_are_required() -> None:
    import pynapple as nap

    with pytest.raises(ValueError, match="at least n_folds"):
        encoding.validate_trajectory_lap_counts(
            _trajectory_intervals(n_laps=2),
            n_folds=3,
        )

    movement_intervals = nap.IntervalSet(
        start=np.asarray([], dtype=float),
        end=np.asarray([], dtype=float),
        time_units="s",
    )
    with pytest.raises(ValueError, match="nonzero movement-supported"):
        encoding.build_strict_cross_validation_folds(
            trajectory_intervals_by_type=_trajectory_intervals(n_laps=2),
            movement_intervals=movement_intervals,
            n_folds=2,
            random_seed=47,
        )


def test_trajectory_lap_shuffles_are_independent_and_reproducible() -> None:
    import pynapple as nap

    intervals = _trajectory_intervals(n_laps=10)
    movement_intervals = nap.IntervalSet(
        start=0.0,
        end=80.0,
        time_units="s",
    )
    first = encoding.build_strict_cross_validation_folds(
        trajectory_intervals_by_type=intervals,
        movement_intervals=movement_intervals,
        n_folds=5,
        random_seed=47,
    )
    second = encoding.build_strict_cross_validation_folds(
        trajectory_intervals_by_type=intervals,
        movement_intervals=movement_intervals,
        n_folds=5,
        random_seed=47,
    )

    for first_folds, second_folds in zip(first, second, strict=True):
        for fold in range(5):
            np.testing.assert_allclose(
                np.asarray(first_folds[fold].start, dtype=float),
                np.asarray(second_folds[fold].start, dtype=float),
            )
            np.testing.assert_allclose(
                np.asarray(first_folds[fold].end, dtype=float),
                np.asarray(second_folds[fold].end, dtype=float),
            )

    test_folds = first[1]
    assignments = []
    for trajectory_index, _trajectory_type in enumerate(TRAJECTORY_TYPES):
        trajectory_assignment = np.full(10, -1, dtype=int)
        for fold, fold_intervals in test_folds.items():
            starts = np.asarray(fold_intervals.start, dtype=float)
            for lap in range(10):
                target = float(trajectory_index * 2 + lap * 8)
                if np.any(np.isclose(starts, target)):
                    trajectory_assignment[lap] = fold
        assert np.all(trajectory_assignment >= 0)
        assignments.append(tuple(trajectory_assignment.tolist()))
    assert len(set(assignments)) == len(TRAJECTORY_TYPES)


def test_unit_metric_qc_is_strict_and_dpp_positive() -> None:
    row = encoding._unit_metric_row(store=_valid_store())

    assert row["unit_valid"] is True
    assert row["qc_status"] == "valid"
    assert row["dpp_log_likelihood_nats"] == -8.0
    assert row["dpp_vs_absolute_place_bits_per_spike"] == pytest.approx(
        3.0 / (np.log(2.0) * 4.0)
    )

    zero_train = _valid_store()
    zero_train["zero_training_spikes"] = True
    row = encoding._unit_metric_row(store=zero_train)
    assert row["qc_status"] == "zero_training_spikes"
    assert not row["unit_valid"]
    assert all(
        row[f"{model_name}_qc_status"] == "zero_training_spikes"
        for model_name in encoding.MODEL_NAMES
    )
    assert np.isnan(row["dpp_log_likelihood_nats"])

    partial = _valid_store()
    partial["model_failed"]["absolute_place"] = True
    row = encoding._unit_metric_row(store=partial)
    assert row["qc_status"] == "partial_model_failure"
    assert row["dpp_qc_status"] == "valid"
    assert row["absolute_place_qc_status"] == "nonfinite_likelihood"
    assert np.isnan(row["dpp_vs_absolute_place_bits_per_spike"])


def test_strict_evaluator_invalidates_a_zero_training_fold() -> None:
    import pynapple as nap

    support = nap.IntervalSet(start=0.0, end=2.0, time_units="s")
    spikes = nap.TsGroup(
        {
            10: nap.Ts(
                np.asarray([0.2, 0.7, 1.2, 1.7]),
                time_units="s",
            ),
            20: nap.Ts(np.asarray([0.3, 0.8]), time_units="s"),
        },
        time_support=support,
        time_units="s",
    )
    times = np.arange(0.025, 2.0, 0.05)
    values = np.mod(times, 1.0)
    features = {
        model_name: nap.Tsd(
            t=times,
            d=values,
            time_support=support,
            time_units="s",
        )
        for model_name in encoding.MODEL_NAMES
    }
    bins = {
        model_name: np.asarray([0.0, 0.5, 1.0])
        for model_name in encoding.MODEL_NAMES
    }
    train_folds = {
        0: nap.IntervalSet(start=0.0, end=1.0, time_units="s"),
        1: nap.IntervalSet(start=1.0, end=2.0, time_units="s"),
    }
    test_folds = {0: train_folds[1], 1: train_folds[0]}

    stores = encoding._evaluate_encoding_models(
        spikes=spikes,
        features=features,
        bins=bins,
        smoothing_boundaries_by_model={
            model_name: () for model_name in encoding.MODEL_NAMES
        },
        train_folds=train_folds,
        test_folds=test_folds,
        n_folds=2,
        evaluation_bin_size_s=0.05,
        gaussian_smoothing_sigma_bins=0.0,
    )

    valid = encoding._unit_metric_row(store=stores[10])
    invalid = encoding._unit_metric_row(store=stores[20])
    assert valid["heldout_spike_count"] == 4
    assert valid["unit_valid"]
    assert np.isfinite(valid["dpp_log_likelihood_nats"])
    np.testing.assert_allclose(
        [
            valid[f"{model_name}_log_likelihood_nats"]
            for model_name in encoding.MODEL_NAMES
        ],
        valid["dpp_log_likelihood_nats"],
    )
    assert valid["dpp_vs_absolute_place_bits_per_spike"] == pytest.approx(0.0)
    assert invalid["heldout_spike_count"] == 2
    assert invalid["qc_status"] == "zero_training_spikes"
    assert not invalid["unit_valid"]


def test_compute_filters_before_fitting_and_writes_self_describing_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_inputs = {
        "features": {name: object() for name in encoding.MODEL_NAMES},
        "bins": {name: np.asarray([0.0, 1.0]) for name in encoding.MODEL_NAMES},
        "smoothing_boundaries_by_model": {
            name: () for name in encoding.MODEL_NAMES
        },
    }
    monkeypatch.setattr(
        encoding,
        "build_encoding_model_inputs",
        lambda **_kwargs: model_inputs,
    )
    monkeypatch.setattr(
        encoding,
        "build_strict_cross_validation_folds",
        lambda **_kwargs: ({0: object()}, {0: object()}),
    )
    seen: dict[str, object] = {}

    def _fake_evaluate(*, spikes, evaluation_bin_size_s, n_folds, **_kwargs):
        seen["keys"] = list(spikes.keys())
        seen["evaluation_bin_size_s"] = evaluation_bin_size_s
        seen["n_folds"] = n_folds
        return {key: _valid_store() for key in spikes.keys()}

    monkeypatch.setattr(encoding, "_evaluate_encoding_models", _fake_evaluate)
    result = encoding.compute_selected_dpp_encoding_comparison(
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        position=object(),
        trajectory_intervals_by_type=_trajectory_intervals(n_laps=5),
        graph_inputs_by_configuration=_graphs(),
        movement_intervals=object(),
        movement_firing_rate_table=_movement_table(),
        stability_tables_by_trajectory=_stability_tables(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
        dpp_encoding_comparison_id=COMPARISON_ID,
        n_folds=5,
        evaluation_bin_size_s=0.05,
        spatial_bin_size_cm=4.0,
        gaussian_smoothing_sigma_bins=1.0,
        random_seed=47,
    )
    table = result["table"]

    assert seen == {
        "keys": [10, 30],
        "evaluation_bin_size_s": 0.05,
        "n_folds": 5,
    }
    assert table["stable_unit_id"].tolist() == ["merge-a:11", "merge-c:33"]
    assert table["dpp_encoding_comparison_id"].unique().tolist() == [
        str(COMPARISON_ID)
    ]
    assert table["spatial_bin_size_cm"].unique().tolist() == [4.0]
    assert table["random_seed"].unique().tolist() == [47]
    assert result["n_units_eligible"] == 2
    assert result["n_units_valid"] == 2
    assert len(result["eligible_units_sha256"]) == 64


def test_validator_checks_equations_qc_and_exact_schema() -> None:
    table = pd.DataFrame.from_records([_canonical_row()]).loc[
        :, list(encoding.TABLE_COLUMNS)
    ]

    assert encoding.validate_encoding_comparison_table(table) is table
    bad = table.copy()
    bad.loc[0, "dpp_vs_absolute_place_bits_per_spike"] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        encoding.validate_encoding_comparison_table(bad)
    with pytest.raises(ValueError, match="exact canonical schema"):
        encoding.validate_encoding_comparison_table(table.assign(extra=1))


def test_legacy_normalization_uses_exact_eligible_set_and_total_nats() -> None:
    table = encoding.normalize_legacy_encoding_comparison_table(
        _legacy_table(),
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        movement_firing_rate_table=_movement_table(),
        stability_tables_by_trajectory=_stability_tables(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
        unit_identity_resolver=_resolver(),
        dpp_encoding_comparison_id=COMPARISON_ID,
    )

    assert table["stable_unit_id"].tolist() == ["merge-a:11", "merge-c:33"]
    assert table["dpp_log_likelihood_nats"].tolist() == [-8.0, -8.0]
    assert table["null_log_likelihood_nats"].tolist() == [-12.0, -12.0]
    assert table["dpp_vs_absolute_place_bits_per_spike"].iloc[0] == pytest.approx(
        0.75 / np.log(2.0)
    )

    with pytest.raises(ValueError, match="exactly match"):
        encoding.normalize_legacy_encoding_comparison_table(
            _legacy_table(units=(101,)),
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="08_r4",
            spikes=_spikes(),
            stable_unit_ids=STABLE_UNIT_IDS,
            movement_firing_rate_table=_movement_table(),
            stability_tables_by_trajectory=_stability_tables(),
            minimum_movement_firing_rate_hz=0.5,
            minimum_stability_correlation=0.5,
            unit_identity_resolver=_resolver(),
            dpp_encoding_comparison_id=COMPARISON_ID,
        )


def test_artifact_round_trip_and_no_implicit_overwrite(tmp_path: Path) -> None:
    table = pd.DataFrame.from_records([_canonical_row()]).loc[
        :, list(encoding.TABLE_COLUMNS)
    ]
    path = tmp_path / "encoding.parquet"

    assert encoding.write_encoding_comparison_artifact(table, path) == path
    loaded = encoding.load_encoding_comparison_artifact(path)
    pd.testing.assert_frame_equal(loaded, table, check_dtype=False)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        encoding.write_encoding_comparison_artifact(table, path)


def test_legacy_registration_records_digest_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.parquet"
    destination = tmp_path / "canonical.parquet"
    _legacy_table().to_parquet(source)

    result = encoding.register_existing_encoding_comparison_artifact(
        source,
        destination,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="08_r4",
        spikes=_spikes(),
        stable_unit_ids=STABLE_UNIT_IDS,
        movement_firing_rate_table=_movement_table(),
        stability_tables_by_trajectory=_stability_tables(),
        minimum_movement_firing_rate_hz=0.5,
        minimum_stability_correlation=0.5,
        unit_identity_resolver=_resolver(),
        dpp_encoding_comparison_id=COMPARISON_ID,
        source_v1ca1_git_commit="abc123",
    )

    assert result["path"] == destination
    assert result["legacy_artifact_provenance"]["source_v1ca1_git_commit"] == (
        "abc123"
    )
    assert len(result["legacy_artifact_provenance"]["source_sha256"]) == 64
    assert result["_created_artifact_paths"] == [str(destination)]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        encoding.register_existing_encoding_comparison_artifact(
            source,
            destination,
            animal_name="L14",
            date="20240611",
            region="v1",
            epoch="08_r4",
            spikes=_spikes(),
            stable_unit_ids=STABLE_UNIT_IDS,
            movement_firing_rate_table=_movement_table(),
            stability_tables_by_trajectory=_stability_tables(),
            minimum_movement_firing_rate_hz=0.5,
            minimum_stability_correlation=0.5,
            unit_identity_resolver=_resolver(),
            dpp_encoding_comparison_id=COMPARISON_ID,
        )
