"""Tests for database-free Figure 1 encoding-model orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass.offline import figure_1_models as models


STABILITY_IDS = {
    trajectory_type: f"stability-{trajectory_type}"
    for trajectory_type in TRAJECTORY_TYPES
}


def _shared_inputs(tmp_path: Path) -> dict[str, object]:
    """Return lightweight placeholders for already-loaded NWB inputs."""
    return {
        "output_dir": tmp_path / "offline-run",
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "epoch": "08_r4",
        "nwb_file_name": "L1420240611_augmented.nwb",
        "region_sorted_spikes_group_id": "regional-view",
        "movement_firing_rate_id": "movement-result",
        "stability_ids_by_trajectory": STABILITY_IDS,
        "spikes": object(),
        "stable_unit_ids": (
            {"spikesorting_merge_id": "merge-a", "unit_id": 1},
        ),
        "trajectory_intervals_by_type": {
            name: object() for name in TRAJECTORY_TYPES
        },
        "graph_inputs_by_configuration": {
            **{name: {} for name in TRAJECTORY_TYPES},
            "full_w": {},
        },
        "movement_intervals": object(),
        "movement_firing_rate_table": pd.DataFrame(),
        "stability_tables_by_trajectory": {
            name: pd.DataFrame() for name in TRAJECTORY_TYPES
        },
    }


def test_selection_snapshots_are_deterministic_and_parameter_frozen() -> None:
    motor_arguments = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "epoch": "08_r4",
        "region_sorted_spikes_group_id": "regional-view",
        "movement_firing_rate_id": "movement-result",
        "primary_position_series_name": "head_position",
        "orientation_reference_position_series_name": "body_position",
        "stability_ids_by_trajectory": STABILITY_IDS,
        "parameters": dict(models.FIGURE_1_MOTOR_ENCODING_PARAMETERS),
    }
    first = models.build_motor_encoding_selection_snapshot(**motor_arguments)
    second = models.build_motor_encoding_selection_snapshot(**motor_arguments)

    assert first == second
    assert first["motor_encoding_parameters_sha256"]
    assert first["full_w_configuration_name"] == "full_w"
    assert first["center_to_left_configuration_name"] == "center_to_left"

    changed = dict(models.FIGURE_1_MOTOR_ENCODING_PARAMETERS)
    changed["minimum_movement_firing_rate_hz"] = 0.0
    motor_arguments["parameters"] = changed
    with pytest.raises(ValueError, match="approved Figure 1"):
        models.build_motor_encoding_selection_snapshot(**motor_arguments)

    dpp_arguments = {
        "nwb_file_name": "L1420240611_augmented.nwb",
        "epoch": "08_r4",
        "region_sorted_spikes_group_id": "regional-view",
        "movement_firing_rate_id": "movement-result",
        "stability_ids_by_trajectory": STABILITY_IDS,
        "parameters": dict(models.FIGURE_1_DPP_ENCODING_PARAMETERS),
    }
    assert (
        models.build_dpp_encoding_selection_snapshot(**dpp_arguments)
        == models.build_dpp_encoding_selection_snapshot(**dpp_arguments)
    )


def test_motor_runner_computes_de_novo_and_writes_below_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = _shared_inputs(tmp_path)
    captured: dict[str, object] = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return {
            "metadata": {"motor_encoding_id": kwargs["motor_encoding_id"]},
            "artifact_origin": "computed",
            "analysis_status": "valid",
            "n_units_input": 1,
            "n_units_eligible": 1,
            "n_units_valid": 1,
        }

    def fake_write(result, path, *, overwrite):
        assert overwrite is False
        path.mkdir(parents=True)
        for filename in (
            "manifest.parquet",
            "selected_units.parquet",
            "nested_cv.nc",
            "full_refit.nc",
        ):
            (path / filename).write_bytes(filename.encode())

    monkeypatch.setattr(
        models.motor_encoding,
        "compute_motor_encoding",
        fake_compute,
    )
    monkeypatch.setattr(
        models.motor_encoding,
        "write_motor_encoding_artifact",
        fake_write,
    )
    record = models.run_offline_motor_encoding(
        **supplied,
        primary_position_series_name="head_position",
        orientation_reference_position_series_name="body_position",
        parameters=dict(models.FIGURE_1_MOTOR_ENCODING_PARAMETERS),
        primary_position=object(),
        orientation_reference_position=object(),
        primary_position_source="/processing/behavior/head_position",
        orientation_reference_position_source=(
            "/processing/behavior/body_position"
        ),
    )

    assert captured["minimum_movement_firing_rate_hz"] == 0.5
    assert captured["minimum_stability_correlation"] == 0.5
    assert captured["evaluation_bin_size_s"] == 0.05
    assert captured["outer_n_folds"] == 5
    assert captured["inner_n_folds"] == 3
    assert record["artifact_origin"] == "computed"
    assert record["record_sha256"]
    for artifact in record["artifacts"].values():
        assert artifact["sha256"]
        path = Path(supplied["output_dir"]) / artifact["relative_path"]
        assert path.is_file()
        assert path.resolve().is_relative_to(
            Path(supplied["output_dir"]).resolve()
        )


def test_dpp_runner_uses_approved_parameters_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = _shared_inputs(tmp_path)
    captured: dict[str, object] = {}
    compute_calls = 0

    def fake_compute(**kwargs):
        nonlocal compute_calls
        compute_calls += 1
        captured.update(kwargs)
        return {
            "table": pd.DataFrame(),
            "analysis_status": "valid",
            "n_units_eligible": 1,
            "n_units_valid": 1,
            "eligible_units_sha256": "a" * 64,
        }

    def fake_write(table, path, *, overwrite):
        assert overwrite is False
        path.parent.mkdir(parents=True)
        path.write_bytes(b"dpp-result")

    monkeypatch.setattr(
        models.dpp_encoding,
        "compute_selected_dpp_encoding",
        fake_compute,
    )
    monkeypatch.setattr(
        models.dpp_encoding,
        "write_dpp_encoding_artifact",
        fake_write,
    )
    arguments = {
        **supplied,
        "parameters": dict(models.FIGURE_1_DPP_ENCODING_PARAMETERS),
        "position": object(),
    }
    record = models.run_offline_dpp_encoding(**arguments)

    assert captured["n_folds"] == 5
    assert captured["evaluation_bin_size_s"] == 0.05
    assert captured["spatial_bin_size_cm"] == 4.0
    assert captured["gaussian_smoothing_sigma_bins"] == 1.0
    assert captured["random_seed"] == 47
    assert captured["minimum_movement_firing_rate_hz"] == 0.5
    assert captured["minimum_stability_correlation"] == 0.5
    assert record["artifacts"]["dpp_encoding_path"]["sha256"]

    with pytest.raises(FileExistsError, match="overwrite"):
        models.run_offline_dpp_encoding(**arguments)
    assert compute_calls == 1
