"""Tests for database-free Figure 2 analysis wrappers."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import dark_light_glm, swap_glm
from v1ca1.spyglass.offline import figure_2_analyses as analyses
from v1ca1.spyglass.selection import selection_uuid


def _uuid() -> str:
    """Return one valid random source identifier for a test fixture."""
    return str(uuid.uuid4())


def _trajectory_mapping(value: object) -> dict[str, object]:
    """Return one complete four-path input mapping."""
    return {trajectory_type: value for trajectory_type in TRAJECTORY_TYPES}


def _identity_rows() -> tuple[dict[str, object], ...]:
    """Return one persistent unit identity."""
    return ({"spikesorting_merge_id": _uuid(), "unit_id": 3},)


def _curve_stub(
    *,
    animal_name: str = "L14",
    date: str = "20240611",
    region: str = "v1",
    epoch: str = "04_r2",
) -> SimpleNamespace:
    """Return lightweight tuning-curve provenance for wrapper tests."""
    return SimpleNamespace(
        attrs={
            "animal_name": animal_name,
            "date": date,
            "region": region,
            "epoch": epoch,
        }
    )


def _write_files(directory: Path, filenames: tuple[str, ...]) -> None:
    """Create a lightweight fake artifact bundle."""
    directory.mkdir(parents=True)
    for filename in filenames:
        path = directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode())


def test_selection_snapshots_are_deterministic_and_frozen() -> None:
    curve_ids = {trajectory_type: _uuid() for trajectory_type in TRAJECTORY_TYPES}
    first = analyses.build_tuning_similarity_selection_snapshot(
        tuning_curve_ids_by_trajectory=curve_ids
    )
    second = analyses.build_tuning_similarity_selection_snapshot(
        tuning_curve_ids_by_trajectory=curve_ids
    )

    assert first == second
    natural_key = {
        name: value
        for name, value in first.items()
        if name != "path_specific_place_tuning_similarity_id"
    }
    assert first["path_specific_place_tuning_similarity_id"] == str(
        selection_uuid("PathSpecificPlaceTuningSimilarity", natural_key)
    )
    assert first["tuning_similarity_param_name"] == "absolute_overlap"

    changed = dict(analyses.FIGURE_2_TUNING_SIMILARITY_PARAMETERS)
    changed["similarity_metric"] = "correlation"
    with pytest.raises(ValueError, match="approved Figure 2"):
        analyses.build_tuning_similarity_selection_snapshot(
            tuning_curve_ids_by_trajectory=curve_ids,
            parameters=changed,
        )

    dark_selection = analyses.build_dark_light_glm_selection_snapshot(
        nwb_file_name="L1420240611_augmented.nwb",
        region_sorted_spikes_group_id=_uuid(),
        dark_epoch="04_r2",
        light_epoch="02_r1",
        dark_movement_firing_rate_id=_uuid(),
        light_movement_firing_rate_id=_uuid(),
    )
    assert dark_selection["dark_light_glm_param_name"] == "legacy_v4_v1"
    assert analyses.FIGURE_2_DARK_LIGHT_GLM_PARAMETERS["min_dark_firing_rate_hz"] == 0.5
    assert (
        analyses.FIGURE_2_DARK_LIGHT_GLM_PARAMETERS["min_light_firing_rate_hz"] == 0.5
    )

    snapshot = {
        "dark_light_manifest_sha256": "a" * 64,
        "dark_light_selected_sha256_by_model": {
            model_name: "b" * 64 for model_name in swap_glm.SOURCE_MODEL_NAMES
        },
        "dark_light_parameter_sha256": dark_selection[
            "dark_light_glm_parameters_sha256"
        ],
        "dark_light_output_rule_sha256": dark_selection[
            "dark_light_glm_output_rule_sha256"
        ],
        "upstream_analysis_status": "valid",
        "metadata": {
            "dark_light_glm_id": dark_selection["dark_light_glm_id"],
            "dark_epoch": "04_r2",
            "light_epoch": "02_r1",
        },
    }
    with pytest.raises(ValueError, match="forward 02_r1-to-06_r3"):
        analyses.build_forward_swap_glm_selection_snapshot(
            nwb_file_name="L1420240611_augmented.nwb",
            region_sorted_spikes_group_id=(
                dark_selection["region_sorted_spikes_group_id"]
            ),
            dark_epoch="04_r2",
            light_train_epoch="06_r3",
            light_test_epoch="02_r1",
            dark_light_glm_id=dark_selection["dark_light_glm_id"],
            light_test_movement_firing_rate_id=_uuid(),
            dark_light_snapshot=snapshot,
        )


def test_tuning_similarity_runner_is_de_novo_and_no_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compute_calls = 0
    captured: dict[str, object] = {}

    def fake_compute(**kwargs):
        nonlocal compute_calls
        compute_calls += 1
        captured.update(kwargs)
        return {
            "table": pd.DataFrame({"score": [0.7]}),
            "analysis_status": "valid",
            "n_units": 1,
            "n_valid_comparisons": 4,
            "n_units_with_valid_comparison": 1,
        }

    def fake_write(table, path, *, overwrite):
        assert overwrite is False
        assert len(table) == 1
        path.parent.mkdir(parents=True)
        path.write_bytes(b"similarity")

    monkeypatch.setattr(
        analyses.tuning_similarity,
        "compute_tuning_similarity_from_curves",
        fake_compute,
    )
    monkeypatch.setattr(
        analyses.tuning_similarity,
        "write_tuning_similarity_artifact",
        fake_write,
    )
    monkeypatch.setattr(
        analyses.tuning_similarity,
        "load_tuning_similarity_artifact",
        lambda path: pd.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "region": ["v1"],
                "epoch": ["04_r2"],
            }
        ),
    )
    arguments = {
        "output_dir": tmp_path / "run",
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "epoch": "04_r2",
        "tuning_curve_ids_by_trajectory": {
            trajectory_type: _uuid() for trajectory_type in TRAJECTORY_TYPES
        },
        "tuning_curves_by_trajectory": _trajectory_mapping(_curve_stub()),
        "movement_firing_rate_table": pd.DataFrame(),
    }
    record = analyses.run_offline_tuning_similarity(**arguments)

    assert captured["similarity_metric"] == "absolute_overlap"
    assert record["artifact_origin"] == "computed"
    assert record["record_sha256"]
    artifact = record["artifacts"]["similarity_path"]
    assert (Path(arguments["output_dir"]) / artifact["relative_path"]).is_file()

    with pytest.raises(FileExistsError, match="overwrite"):
        analyses.run_offline_tuning_similarity(**arguments)
    assert compute_calls == 1

    mismatched = {
        **arguments,
        "output_dir": tmp_path / "other-run",
        "tuning_curves_by_trajectory": _trajectory_mapping(
            _curve_stub(animal_name="L15")
        ),
    }
    with pytest.raises(ValueError, match="mismatched animal_name"):
        analyses.run_offline_tuning_similarity(**mismatched)


def test_path_specific_decoding_runner_writes_validated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_compute(**kwargs):
        captured.update(kwargs)
        return {
            "metadata": {
                "path_specific_place_decoding_id": kwargs[
                    "path_specific_place_decoding_id"
                ]
            },
            "artifact_origin": "computed",
            "analysis_status": "valid",
            "n_units": 1,
            "n_folds_expected": 5,
            "n_folds_valid": 5,
            "n_decoded_samples": 20,
            "selected_units_sha256": "c" * 64,
        }

    artifact_fields = {
        "artifact_manifest_path": "manifest.parquet",
        "selected_units_path": "selected_units.parquet",
        "fold_qc_path": "fold_qc.parquet",
        "decoding_summary_path": "decoding_summary.parquet",
        "binned_error_path": "binned_error.parquet",
        "true_path": "true.npz",
        "decoded_path": "decoded.npz",
    }

    def fake_write(result, path, *, overwrite):
        assert overwrite is False
        _write_files(path, tuple(artifact_fields.values()))
        return {name: path / filename for name, filename in artifact_fields.items()}

    monkeypatch.setattr(
        analyses.path_specific_decoding,
        "compute_path_specific_place_decoding",
        fake_compute,
    )
    monkeypatch.setattr(
        analyses.path_specific_decoding,
        "write_path_specific_decoding_artifact",
        fake_write,
    )
    monkeypatch.setattr(
        analyses.path_specific_decoding,
        "load_path_specific_decoding_artifact",
        lambda path: {
            "metadata": {
                "path_specific_place_decoding_id": path.name,
            }
        },
    )
    output_dir = tmp_path / "run"
    record = analyses.run_offline_path_specific_decoding(
        output_dir=output_dir,
        animal_name="L14",
        date="20240611",
        region="v1",
        epoch="04_r2",
        nwb_file_name="L1420240611_augmented.nwb",
        region_sorted_spikes_group_id=_uuid(),
        movement_firing_rate_id=_uuid(),
        spikes=object(),
        stable_unit_ids=_identity_rows(),
        position=object(),
        trajectory_intervals_by_type=_trajectory_mapping(object()),
        graph_inputs_by_configuration=_trajectory_mapping({}),
        movement_intervals=object(),
    )

    assert captured["n_folds"] == 5
    assert captured["decoding_bin_size_s"] == 0.02
    assert captured["spatial_bin_size_cm"] == 4.0
    assert record["n_folds_valid"] == 5
    assert set(record["artifacts"]) == set(artifact_fields)
    assert all(
        (output_dir / artifact["relative_path"]).is_file()
        for artifact in record["artifacts"].values()
    )


def test_dark_light_and_forward_swap_runners_preserve_frozen_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    nwb_file_name = "L1420240611_augmented.nwb"
    region_group_id = _uuid()
    dark_captured: dict[str, object] = {}

    def fake_dark_compute(**kwargs):
        dark_captured.update(kwargs)
        return {
            "metadata": {"dark_light_glm_id": kwargs["dark_light_glm_id"]},
            "artifact_origin": "computed",
            "analysis_status": "valid",
            "n_units": 1,
            "n_candidates": 72,
            "n_selected_models": len(dark_light_glm.MODEL_NAMES),
            "selected_units_sha256": "d" * 64,
        }

    def fake_dark_write(result, path, *, overwrite):
        assert overwrite is False
        filenames = (
            "manifest.parquet",
            "selected_units.parquet",
            "selection_summary.nc",
            *(f"selected/{model_name}.nc" for model_name in dark_light_glm.MODEL_NAMES),
        )
        _write_files(path, tuple(filenames))
        return {
            "artifact_manifest_path": path / "manifest.parquet",
            "selected_units_path": path / "selected_units.parquet",
            "selection_summary_path": path / "selection_summary.nc",
            "selected_model_paths": {
                model_name: path / "selected" / f"{model_name}.nc"
                for model_name in dark_light_glm.MODEL_NAMES
            },
        }

    def fake_dark_load(path):
        return {
            "metadata": {
                "dark_light_glm_id": path.name,
                "animal_name": "L14",
                "date": "20240611",
                "region": "v1",
                "dark_epoch": "04_r2",
                "light_epoch": "02_r1",
            },
            "parameters": {
                "parameter_sha256": dark_captured["parameter_sha256"],
                "output_rule_sha256": dark_captured["output_rule_sha256"],
            },
            "manifest": pd.DataFrame.from_records(
                [
                    {
                        "artifact_key": f"selected:{model_name}",
                        "sha256": str(index) * 64,
                    }
                    for index, model_name in enumerate(
                        swap_glm.SOURCE_MODEL_NAMES,
                        start=1,
                    )
                ]
            ),
            "analysis_status": "valid",
            "artifact_origin": "computed",
        }

    monkeypatch.setattr(
        analyses.dark_light_glm,
        "compute_dark_light_glm",
        fake_dark_compute,
    )
    monkeypatch.setattr(
        analyses.dark_light_glm,
        "write_dark_light_glm_artifact",
        fake_dark_write,
    )
    monkeypatch.setattr(
        analyses.dark_light_glm,
        "load_dark_light_glm_artifact",
        fake_dark_load,
    )
    dark_record = analyses.run_offline_dark_light_glm(
        output_dir=output_dir,
        animal_name="L14",
        date="20240611",
        region="v1",
        nwb_file_name=nwb_file_name,
        region_sorted_spikes_group_id=region_group_id,
        dark_epoch="04_r2",
        light_epoch="02_r1",
        dark_movement_firing_rate_id=_uuid(),
        light_movement_firing_rate_id=_uuid(),
        spikes=object(),
        stable_unit_ids=_identity_rows(),
        dark_movement_firing_rate_table=pd.DataFrame(),
        light_movement_firing_rate_table=pd.DataFrame(),
        movement_by_epoch={"04_r2": object(), "02_r1": object()},
        trajectory_intervals_by_epoch={
            "04_r2": _trajectory_mapping(object()),
            "02_r1": _trajectory_mapping(object()),
        },
        graph_inputs_by_configuration=_trajectory_mapping({}),
        position_by_epoch={"04_r2": object(), "02_r1": object()},
    )

    assert dark_captured["parameter_name"] == "legacy_v4_v1"
    assert dark_captured["basis_candidate_mode"] == "n_splines"
    assert dark_captured["basis_candidates"] == (25, 40, 60)
    assert dark_captured["min_dark_firing_rate_hz"] == 0.5
    assert dark_captured["min_light_firing_rate_hz"] == 0.5

    swap_captured: dict[str, object] = {}

    def fake_swap_compute(**kwargs):
        swap_captured.update(kwargs)
        return {
            "metadata": {"swap_glm_id": kwargs["swap_glm_id"]},
            "artifact_origin": "computed",
            "analysis_status": "valid",
            "n_units": 1,
            "n_valid_units": 1,
            "selected_units_sha256": "e" * 64,
        }

    def fake_swap_write(result, path, *, overwrite):
        assert overwrite is False
        filenames = (
            "manifest.parquet",
            "selected_units.parquet",
            "swap_glm.nc",
        )
        _write_files(path, filenames)
        return {
            "artifact_manifest_path": path / filenames[0],
            "selected_units_path": path / filenames[1],
            "result_path": path / filenames[2],
        }

    monkeypatch.setattr(
        analyses.swap_glm,
        "compute_swap_glm",
        fake_swap_compute,
    )
    monkeypatch.setattr(
        analyses.swap_glm,
        "write_swap_glm_artifact",
        fake_swap_write,
    )
    monkeypatch.setattr(
        analyses.swap_glm,
        "load_swap_glm_artifact",
        lambda path: {"metadata": {"swap_glm_id": path.name}},
    )
    swap_record = analyses.run_offline_forward_swap_glm(
        output_dir=output_dir,
        animal_name="L14",
        date="20240611",
        region="v1",
        nwb_file_name=nwb_file_name,
        region_sorted_spikes_group_id=region_group_id,
        dark_epoch="04_r2",
        light_train_epoch="02_r1",
        light_test_epoch="06_r3",
        dark_light_record=dark_record,
        light_test_movement_firing_rate_id=_uuid(),
        spikes=object(),
        stable_unit_ids=_identity_rows(),
        movement_interval=object(),
        movement_analysis_status="valid",
        trajectory_intervals_by_type=_trajectory_mapping(object()),
        graph_inputs_by_configuration=_trajectory_mapping({}),
        position=object(),
    )

    assert swap_captured["light_train_epoch"] == "02_r1"
    assert swap_captured["light_test_epoch"] == "06_r3"
    assert swap_captured["dark_light_glm_artifact_path"].is_dir()
    assert swap_record["artifact_origin"] == "computed"
    assert swap_record["selection"]["light_train_condition"] == "AB"
    assert swap_record["selection"]["light_test_condition"] == "BA"
    assert swap_record["record_sha256"]
