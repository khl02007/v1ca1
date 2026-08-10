"""Tests for database-free Figure 1 cross-path decoding orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass.offline import figure_1_decoding as offline


REGION_GROUP_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
MOVEMENT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
STABILITY_IDS = {
    trajectory_type: uuid.uuid5(uuid.NAMESPACE_URL, trajectory_type)
    for trajectory_type in TRAJECTORY_TYPES
}
UNIT_IDS = (
    {"spikesorting_merge_id": "merge-a", "unit_id": "4"},
    {"spikesorting_merge_id": "merge-a", "unit_id": "9"},
)


def _selection(**overrides):
    """Return one deterministic synthetic source selection."""
    arguments = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "epoch": "08_r4",
        "region_sorted_spikes_group_id": REGION_GROUP_ID,
        "movement_firing_rate_id": MOVEMENT_ID,
        "stability_source_ids": STABILITY_IDS,
    }
    arguments.update(overrides)
    return offline.build_figure_1_decoding_selection(**arguments)


def test_selection_is_deterministic_same_epoch_and_uses_fixed_parameters() -> None:
    first = _selection()
    second = _selection()

    assert first == second
    assert uuid.UUID(first["path_progression_decoding_id"])
    assert first["cohort_epoch"] == first["epoch"] == "08_r4"
    assert first["cohort_movement_firing_rate_id"] == str(MOVEMENT_ID)
    assert offline.FIGURE_1_DECODING_PARAMETERS == {
        "path_progression_decoding_param_name": (
            "manuscript_20ms_window4_4cm_mfr0p5"
        ),
        "decoding_bin_size_s": 0.02,
        "sliding_window_size_bins": 4,
        "spatial_bin_size_cm": 4.0,
        "minimum_movement_firing_rate_hz": 0.5,
        "minimum_stability_correlation": None,
    }
    for trajectory_type in TRAJECTORY_TYPES:
        assert first[f"{trajectory_type}_trajectory_type"] == trajectory_type
        assert first[f"{trajectory_type}_configuration_name"] == trajectory_type
        assert first[f"{trajectory_type}_stability_id"] == first[
            f"cohort_{trajectory_type}_stability_id"
        ]

    changed = _selection(
        movement_firing_rate_id=uuid.UUID(
            "33333333-3333-3333-3333-333333333333"
        )
    )
    assert changed["path_progression_decoding_id"] != first[
        "path_progression_decoding_id"
    ]


def test_runner_computes_de_novo_with_fixed_figure_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_compute(**kwargs):
        calls.append(kwargs)
        return {
            "n_units_input": 2,
            "n_units_eligible": 1,
            "n_transfer_pairs_expected": 16,
            "n_transfer_pairs_valid": 16,
            "n_decoded_samples": 100,
            "analysis_status": "valid",
            "eligible_units_sha256": "a" * 64,
        }

    def fake_write(result, paths, *, overwrite):
        assert result["analysis_status"] == "valid"
        assert overwrite is False
        artifact_dir = Path(paths["artifact_dir"])
        artifact_dir.mkdir(parents=True)
        written = {"path": artifact_dir}
        for name in offline._ARTIFACT_FILE_FIELDS:
            path = Path(paths[name])
            path.write_bytes(name.encode("utf-8"))
            written[name] = path
        return written

    monkeypatch.setattr(
        offline.decoding,
        "compute_path_progression_decoding",
        fake_compute,
    )
    monkeypatch.setattr(
        offline.decoding,
        "write_decoding_artifact_bundle",
        fake_write,
    )
    path_inputs = {trajectory_type: object() for trajectory_type in TRAJECTORY_TYPES}
    stability_tables = {
        trajectory_type: pd.DataFrame()
        for trajectory_type in TRAJECTORY_TYPES
    }
    output_dir = tmp_path / "explicit-output"
    inputs = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "08_r4",
        "nwb_file_name": "L1420240611_augmented.nwb",
        "region_sorted_spikes_group_id": REGION_GROUP_ID,
        "movement_firing_rate_id": MOVEMENT_ID,
        "stability_source_ids": STABILITY_IDS,
        "spikes": object(),
        "stable_unit_ids": UNIT_IDS,
        "movement_firing_rate_table": pd.DataFrame(),
        "stability_tables_by_trajectory": stability_tables,
        "position": object(),
        "trajectory_intervals": path_inputs,
        "graph_inputs": {
            trajectory_type: {} for trajectory_type in TRAJECTORY_TYPES
        },
        "movement_interval": object(),
        "output_dir": output_dir,
    }

    output = offline.run_figure_1_decoding(**inputs)

    assert len(calls) == 1
    call = calls[0]
    assert call["region"] == "v1"
    assert call["epoch"] == call["cohort_epoch"] == "08_r4"
    assert call["target_movement_firing_rate_table"] is call[
        "cohort_movement_firing_rate_table"
    ]
    assert call["target_stability_tables_by_trajectory"] is call[
        "cohort_stability_tables_by_trajectory"
    ]
    assert call["decoding_bin_size_s"] == 0.02
    assert call["sliding_window_size_bins"] == 4
    assert call["spatial_bin_size_cm"] == 4.0
    assert call["minimum_movement_firing_rate_hz"] == 0.5
    assert call["minimum_stability_correlation"] is None
    record = output["artifact_record"]
    assert record["artifact_origin"] == "computed"
    assert record["region"] == "v1"
    assert set(record["artifact_sha256"]) == set(
        offline._ARTIFACT_FILE_FIELDS
    )
    assert not Path(record["artifact_manifest_path"]).is_absolute()
    assert (
        output_dir / record["artifact_manifest_path"]
    ).is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        offline.run_figure_1_decoding(**inputs)
    assert len(calls) == 1


def test_adapter_builds_figure_sample_and_trial_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {}
    manifest_rows = []
    for _comparison, _label, family, pairs in (
        offline.FIGURE_1_DECODING_COMPARISONS
    ):
        for source, target in pairs:
            key = (family, source, target)
            outputs[key] = {"true": object(), "decoded": object()}
            for role in ("true", "decoded"):
                manifest_rows.append(
                    {
                        "artifact_key": (
                            f"cross:{family}:{source}:{target}:{role}"
                        ),
                        "relative_path": f"{family}_{source}_{target}_{role}.npz",
                    }
                )
    bundle = {
        "metadata": {
            "animal_name": "L14",
            "date": "20240611",
            "epoch": "08_r4",
            "cohort_epoch": "08_r4",
            "region": "v1",
            "parameter_name": "manuscript_20ms_window4_4cm_mfr0p5",
        },
        "parameters": {
            "decoding_bin_size_s": 0.02,
            "sliding_window_size_bins": 4,
            "spatial_bin_size_cm": 4.0,
            "minimum_movement_firing_rate_hz": 0.5,
            "minimum_stability_correlation": None,
        },
        "cross_path_outputs": outputs,
        "manifest": pd.DataFrame.from_records(manifest_rows),
        "path": tmp_path,
    }
    monkeypatch.setattr(
        offline.decoding,
        "load_decoding_artifact_bundle",
        lambda path: bundle,
    )
    monkeypatch.setattr(
        offline,
        "_aligned_absolute_error_with_times",
        lambda true, decoded: (
            np.asarray([0.1, 0.4, 0.7]),
            np.asarray([0.1, 0.2, 0.3]),
        ),
    )
    intervals = {
        trajectory_type: SimpleNamespace(
            start=np.asarray([0.0, 0.5]),
            end=np.asarray([0.5, 1.0]),
        )
        for trajectory_type in TRAJECTORY_TYPES
    }

    payload = offline.load_figure_1_decoding_payload(
        artifact_manifest_path=tmp_path / "manifest.parquet",
        trajectory_intervals=intervals,
    )

    sample = payload["absolute_error_table"]
    trials = payload["trial_error_table"]
    assert list(sample.columns) == list(offline.ABSOLUTE_ERROR_COLUMNS)
    assert list(trials.columns) == list(offline.TRIAL_ERROR_COLUMNS)
    assert len(sample) == 12 * 3
    assert len(trials) == 12 * 2
    assert set(sample["comparison"]) == {
        "same_turn_cross_arm",
        "opposite_turn_same_arm",
        "same_inbound_outbound_cross_arm",
    }
    np.testing.assert_allclose(
        sorted(trials["trial_median_absolute_error"].unique()),
        [0.15, 0.3],
    )
    assert set(trials["n_samples"]) == {1, 2}
