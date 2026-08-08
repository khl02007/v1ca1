from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import ripple_decoding_comparison as comparison


class FakeIntervals:
    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)


class FakeSpikes(dict):
    pass


class FakeScience:
    @staticmethod
    def _subset_spikes(spikes, unit_ids):
        return FakeSpikes({unit_id: spikes[unit_id] for unit_id in unit_ids})

    @staticmethod
    def compute_tuning_curves_for_epoch(**kwargs):
        return {"unit_ids": tuple(kwargs["spikes"].keys())}

    @staticmethod
    def assemble_decoded_ripple_epoch_data(*, spikes, ripple_table, **kwargs):
        region = "ca1" if next(iter(spikes)).startswith("ca1") else "v1"
        states = {
            "ca1": np.asarray([75.0, 175.0, 275.0, 375.0]),
            "v1": np.asarray([75.0, 275.0, 175.0, 375.0]),
        }[region]
        starts = ripple_table["start_time"].to_numpy(dtype=float)
        ends = ripple_table["end_time"].to_numpy(dtype=float)
        times = np.concatenate(
            [np.linspace(start, end, 4, endpoint=False)[1:3] for start, end in zip(starts, ends)]
        )
        return {
            "decoded_state": states,
            "bin_times_s": times,
            "ripple_ids": np.asarray([0, 0, 1, 1]),
            "ripple_start_times_s": starts,
            "ripple_end_times_s": ends,
            "ripple_source_indices": np.arange(len(ripple_table)),
            "skipped_ripples": [],
        }

    @staticmethod
    def align_decoded_ripple_data(ca1, v1):
        return {
            "ca1_decoded_state": np.asarray(ca1["decoded_state"]),
            "v1_decoded_state": np.asarray(v1["decoded_state"]),
            "bin_times_s": np.asarray(ca1["bin_times_s"]),
            "ripple_ids": np.asarray(ca1["ripple_ids"]),
            "n_ripples": len(ca1["ripple_source_indices"]),
            "n_bins": len(ca1["decoded_state"]),
            "ripple_source_indices": np.asarray(ca1["ripple_source_indices"]),
            "ripple_start_times_s": np.asarray(ca1["ripple_start_times_s"]),
            "ripple_end_times_s": np.asarray(ca1["ripple_end_times_s"]),
            "skipped_ripples": [],
        }

    @staticmethod
    def compute_per_ripple_categorical_metrics(*, ca1_labels, v1_labels):
        ca1_labels = np.asarray(ca1_labels)
        v1_labels = np.asarray(v1_labels)
        valid = (ca1_labels >= 0) & (v1_labels >= 0)
        n_valid = int(valid.sum())
        n_matching = int((ca1_labels[valid] == v1_labels[valid]).sum())
        return {
            "match_rate": n_matching / n_valid if n_valid else np.nan,
            "n_matching_bins": n_matching,
            "n_valid_labeled_bins": n_valid,
        }

    @staticmethod
    def shuffle_ripple_state_blocks_by_length(state, ripple_ids, rng):
        state = np.asarray(state)
        ripple_ids = np.asarray(ripple_ids)
        blocks = [state[ripple_ids == ripple_id] for ripple_id in np.unique(ripple_ids)]
        return np.concatenate(blocks[::-1]), len(blocks) > 1

    @staticmethod
    def summarize_metric_against_shuffle(observed, null_samples, *, direction):
        finite = np.asarray(null_samples)[np.isfinite(null_samples)]
        return {
            "shuffle_mean": float(np.mean(finite)) if finite.size else np.nan,
            "shuffle_sd": float(np.std(finite)) if finite.size else np.nan,
            "p_value": (
                (1.0 + float(np.sum(finite >= observed))) / (finite.size + 1.0)
                if finite.size and np.isfinite(observed)
                else np.nan
            ),
        }


def _geometry():
    value = {
        "trajectory_order": list(TRAJECTORY_TYPES),
        "path_length_cm": 100.0,
        "path_length_cm_by_trajectory": {
            trajectory: 100.0 for trajectory in TRAJECTORY_TYPES
        },
        "arm_start_cm_by_trajectory": {
            trajectory: 70.0 for trajectory in TRAJECTORY_TYPES
        },
        "physical_arm_by_trajectory": comparison.PHYSICAL_ARM_BY_TRAJECTORY,
        "turn_group_by_trajectory": comparison.TURN_GROUP_BY_TRAJECTORY,
        "graphs": {},
    }
    value["graph_policy_sha256"] = comparison._provenance_sha256(value)
    return value


def _provenance():
    return {
        "detector_zscore_threshold": 2.0,
        "speed_gated": True,
        "movement_speed_threshold_cm_s": 4.0,
        "movement_speed_sigma_s": 0.1,
        "nwb_file_name": "L14_20240611_.nwb",
    }


def _compute_kwargs():
    return {
        "ripple_decoding_comparison_id": str(uuid.uuid4()),
        "animal_name": "L14",
        "date": "20240611",
        "train_epoch": "02_r1",
        "decode_epoch": "03_s2",
        "representation": "path_specific_place",
        "ca1_spikes": FakeSpikes({"ca1-0": object()}),
        "ca1_stable_unit_ids": [
            {"spikesorting_merge_id": "ca1-merge", "unit_id": "1"}
        ],
        "ca1_movement_firing_rates_hz": [0.0],
        "v1_spikes": FakeSpikes({"v1-0": object(), "v1-1": object()}),
        "v1_stable_unit_ids": [
            {"spikesorting_merge_id": "v1-merge", "unit_id": "2"},
            {"spikesorting_merge_id": "v1-merge", "unit_id": "3"},
        ],
        "v1_movement_firing_rates_hz": [0.6, 0.4],
        "position": object(),
        "trajectory_intervals": {
            trajectory: FakeIntervals([0.0], [1.0])
            for trajectory in TRAJECTORY_TYPES
        },
        "graph_inputs": {trajectory: {} for trajectory in TRAJECTORY_TYPES},
        "movement_interval": FakeIntervals([0.0], [1.0]),
        "ripple_table": pd.DataFrame(
            {
                "epoch": ["03_s2", "03_s2"],
                "start_time": [0.1, 0.3],
                "end_time": [0.2, 0.4],
            }
        ),
        "decode_epoch_interval": FakeIntervals([0.0], [1.0]),
        "upstream_provenance": _provenance(),
    }


@pytest.fixture
def computed(monkeypatch):
    monkeypatch.setattr(comparison, "_scientific_module", lambda: FakeScience)
    monkeypatch.setattr(comparison, "_graph_geometry", lambda inputs: _geometry())

    def fake_feature(**kwargs):
        bins = comparison._representation_bins(
            representation=kwargs["representation"],
            geometry=kwargs["geometry"],
            spatial_bin_size_cm=kwargs["spatial_bin_size_cm"],
        )
        return object(), bins, 20

    monkeypatch.setattr(comparison, "_build_representation_feature", fake_feature)
    return comparison.compute_ripple_decoding_comparison(**_compute_kwargs())


def test_parameters_keep_movement_definition_upstream_only():
    parameters = comparison.validate_ripple_decoding_comparison_parameters()
    assert "expected_movement_speed_threshold_cm_s" not in parameters
    assert "expected_speed_sigma_s" not in parameters
    comparison._canonical_provenance(_provenance(), parameters=parameters)
    wrong = _provenance()
    wrong["movement_speed_threshold_cm_s"] = 3.0
    with pytest.raises(ValueError, match="fixed value 4.0"):
        comparison._canonical_provenance(wrong, parameters=parameters)


def test_artifact_path_is_session_pair_representation_uuid():
    result_id = str(uuid.uuid4())
    paths = comparison.get_ripple_decoding_comparison_artifact_paths(
        animal_name="L14",
        date="20240611",
        train_epoch="02_r1",
        decode_epoch="03_s2",
        representation="dpp",
        ripple_decoding_comparison_id=result_id,
        artifact_root=Path("/analysis"),
    )
    assert paths["artifact_dir"] == Path(
        "/analysis/L14/20240611/ripple_decoding_comparison/"
        f"02_r1_train_to_03_s2_decode/dpp/{result_id}"
    )


def test_physical_arm_fix_rejects_legacy_inbound_turn_labels():
    states = np.asarray([75.0, 175.0, 275.0, 375.0])
    labels = comparison._state_labels(
        states,
        representation="path_specific_place",
        scheme="arm_identity",
        geometry=_geometry(),
    )
    legacy_turn_labeled_arms = np.asarray([1, 2, 2, 1])
    assert labels.tolist() == [1, 1, 2, 2]
    assert np.sum(labels != legacy_turn_labeled_arms) == 2
    assert comparison.OUTPUT_RULE["physical_arm_by_trajectory"] == {
        "center_to_left": "left",
        "left_to_center": "left",
        "center_to_right": "right",
        "right_to_center": "right",
    }
    assert "reject" in comparison.OUTPUT_RULE["legacy_arm_bug_policy"]


def test_dpp_only_has_turn_group_scoring():
    availability = comparison._scheme_availability("dpp")
    assert availability["turn_group"]["applicable"] is True
    assert availability["trajectory"]["applicable"] is False
    assert availability["arm_identity"]["applicable"] is False
    labels = comparison._state_labels(
        np.asarray([0.2, 1.2]),
        representation="dpp",
        scheme="turn_group",
        geometry=_geometry(),
    )
    assert labels.tolist() == [0, 1]


def test_graph_geometry_derives_center_to_arm_boundary():
    graph_inputs = {}
    for trajectory in TRAJECTORY_TYPES:
        outbound = not trajectory.endswith("_to_center")
        edge_order = (
            [[0, 1], [1, 2], [2, 3]]
            if outbound
            else [[3, 2], [2, 1], [1, 0]]
        )
        graph_inputs[trajectory] = {
            "configuration_name": trajectory,
            "coordinate_unit": "cm",
            "track_graph_kwargs": {
                "node_positions": [[0, 0], [10, 0], [20, 0], [30, 0]],
                "edges": [[0, 1], [1, 2], [2, 3]],
            },
            "linearization_kwargs": {
                "edge_order": edge_order,
                "edge_spacing": [0, 0],
                "use_HMM": False,
            },
        }
    geometry = comparison._graph_geometry(graph_inputs)
    assert geometry["path_length_cm"] == pytest.approx(30.0)
    assert set(geometry["arm_start_cm_by_trajectory"].values()) == {15.0}


def test_compute_filters_units_and_rescores_persisted_decoding(computed):
    assert computed["analysis_status"] == "valid"
    units = computed["selected_units"]
    v1 = units.loc[units["region"] == "v1"]
    assert v1["included_in_decoder"].tolist() == [True, False]
    assert computed["n_ripples"] == 2
    assert computed["ripple_qc"]["alignment_status"].tolist() == ["valid", "valid"]
    assert set(computed["ripple_metrics"]["representation"]) == {
        "path_specific_place"
    }
    tampered = dict(computed)
    tampered["ca1_decoded"] = deepcopy(computed["ca1_decoded"])
    tampered["ca1_decoded"]["decoded_state"] = np.asarray(
        tampered["ca1_decoded"]["decoded_state"]
    ).copy()
    tampered["ca1_decoded"]["decoded_state"][0] += 4.0
    with pytest.raises(ValueError, match="dataset variable 'ca1_decoded_state'"):
        comparison.validate_ripple_decoding_comparison_result(tampered)


def test_no_ripples_is_an_explicit_terminal(monkeypatch):
    monkeypatch.setattr(comparison, "_scientific_module", lambda: FakeScience)
    monkeypatch.setattr(comparison, "_graph_geometry", lambda inputs: _geometry())
    kwargs = _compute_kwargs()
    kwargs["ripple_table"] = pd.DataFrame(
        columns=["epoch", "start_time", "end_time"]
    )
    result = comparison.compute_ripple_decoding_comparison(**kwargs)
    assert result["analysis_status"] == "no_ripples"
    assert result["ripple_metrics"].empty
    assert result["epoch_summary"].iloc[0]["analysis_status"] == "no_ripples"


def test_no_train_movement_retains_nan_rate_unit_audit(monkeypatch):
    monkeypatch.setattr(comparison, "_scientific_module", lambda: FakeScience)
    monkeypatch.setattr(comparison, "_graph_geometry", lambda inputs: _geometry())
    kwargs = _compute_kwargs()
    kwargs["movement_interval"] = FakeIntervals([], [])
    kwargs["ca1_movement_firing_rates_hz"] = [np.nan]
    kwargs["v1_movement_firing_rates_hz"] = [np.nan, np.nan]
    result = comparison.compute_ripple_decoding_comparison(**kwargs)
    assert result["analysis_status"] == "no_train_movement"
    assert result["selected_units"]["movement_firing_rate_hz"].isna().all()
    assert not result["selected_units"]["passes_movement_firing_rate"].any()
    assert set(result["selected_units"]["unit_qc_status"]) == {"not_computed"}
    comparison.validate_ripple_decoding_comparison_result(result)


def test_no_train_movement_precedes_empty_ripple_input(monkeypatch):
    monkeypatch.setattr(comparison, "_scientific_module", lambda: FakeScience)
    monkeypatch.setattr(comparison, "_graph_geometry", lambda inputs: _geometry())
    kwargs = _compute_kwargs()
    kwargs["movement_interval"] = FakeIntervals([], [])
    kwargs["ca1_movement_firing_rates_hz"] = [np.nan]
    kwargs["v1_movement_firing_rates_hz"] = [np.nan, np.nan]
    kwargs["ripple_table"] = pd.DataFrame(
        columns=["epoch", "start_time", "end_time"]
    )
    result = comparison.compute_ripple_decoding_comparison(**kwargs)
    assert result["analysis_status"] == "no_train_movement"
    assert result["ripple_qc"].empty
    comparison.validate_ripple_decoding_comparison_result(result)


def test_atomic_bundle_detects_file_tampering(tmp_path, computed):
    destination = tmp_path / computed["ripple_decoding_comparison_id"]
    comparison.write_ripple_decoding_comparison_artifact(computed, destination)
    loaded = comparison.load_ripple_decoding_comparison_artifact(destination)
    assert loaded["analysis_status"] == "valid"
    with (destination / comparison.RIPPLE_METRICS_FILENAME).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        comparison.load_ripple_decoding_comparison_artifact(destination)


def test_legacy_metric_gate_rejects_old_arm_mapping(tmp_path, computed):
    metrics = computed["ripple_metrics"].copy()
    summary = computed["epoch_summary"].copy()
    metrics["representation"] = "place"
    summary["representation"] = "place"
    metrics.loc[0, "arm_identity_match_rate"] = 0.123456
    metrics_path = tmp_path / "ripple_metrics.parquet"
    summary_path = tmp_path / "epoch_summary.parquet"
    metrics.to_parquet(metrics_path, index=False)
    summary.to_parquet(summary_path, index=False)
    metadata = {
        name: computed[name]
        for name in (
            "ripple_decoding_comparison_id",
            "animal_name",
            "date",
            "train_epoch",
            "decode_epoch",
            "representation",
        )
    }
    with pytest.raises(ValueError, match="physical-arm graph scoring"):
        comparison._compare_legacy_tables(
            metrics_path=metrics_path,
            summary_path=summary_path,
            expected_metrics=computed["ripple_metrics"],
            expected_summary=computed["epoch_summary"],
            metadata=metadata,
        )


def test_legacy_registration_requires_imported_sortings():
    with pytest.raises(ValueError, match="ImportedSpikeSorting"):
        comparison.register_existing_ripple_decoding_comparison_artifact(
            source_ca1_decoded_path=Path("missing-ca1.npz"),
            source_v1_decoded_path=Path("missing-v1.npz"),
            source_ripple_metrics_path=Path("missing-metrics.parquet"),
            source_epoch_summary_path=Path("missing-summary.parquet"),
            source_result_path=Path("missing.nc"),
            destination_path=Path("unused"),
            ca1_legacy_identity_resolver=lambda values: [],
            v1_legacy_identity_resolver=lambda values: [],
            ca1_sorting_type="CurationV1",
            v1_sorting_type="ImportedSpikeSorting",
        )
