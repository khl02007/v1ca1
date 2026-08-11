from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from v1ca1.helper.session import TRAJECTORY_TYPES
from v1ca1.spyglass import dark_light_glm as module


def _graph_payload(trajectory_type: str, *, long_edge: float = 10.0) -> dict[str, object]:
    node_positions = np.asarray(
        [
            [0.0, 0.0],
            [long_edge, 0.0],
            [long_edge + 3.0, 4.0],
            [long_edge + 9.0, 4.0],
            [long_edge + 12.0, 0.0],
            [2.0 * long_edge + 12.0, 0.0],
        ],
        dtype=float,
    )
    edge_order = [(index, index + 1) for index in range(5)]
    if trajectory_type in {"left_to_center", "right_to_center"}:
        edge_order = [(stop, start) for start, stop in reversed(edge_order)]
    return {
        "configuration_name": trajectory_type,
        "coordinate_unit": "cm",
        "track_graph_kwargs": {
            "node_positions": node_positions,
            "edges": np.asarray([(index, index + 1) for index in range(5)]),
        },
        "linearization_kwargs": {
            "edge_order": edge_order,
            "edge_spacing": [0.0] * 4,
        },
    }


def _graphs() -> dict[str, dict[str, object]]:
    return {
        trajectory_type: _graph_payload(trajectory_type)
        for trajectory_type in TRAJECTORY_TYPES
    }


def _metadata(result_id: uuid.UUID) -> dict[str, str]:
    return {
        "dark_light_glm_id": str(result_id),
        "animal_name": "L14",
        "date": "20240611",
        "region": "v1",
        "light_epoch": "02_r1",
        "dark_epoch": "08_r4",
    }


def _selected_units() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "stable_unit_id": "merge-a:11",
                "group_unit_id": 0,
                "selection_index": 0,
                "dark_movement_firing_rate_hz": 1.0,
                "light_movement_firing_rate_hz": 1.5,
                "n_selected_model_trajectory_fits": 16,
                "valid_glm_fit": True,
            }
        ],
        columns=module.SELECTED_UNIT_COLUMNS,
    )


def _fit_parameters(
    *,
    mode: str,
    basis_candidates: tuple[float | int, ...],
    trajectory_length_cm: float = 36.0,
    seed_name: str = "random_seed",
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "basis_candidate_mode": mode,
        "basis_candidates": list(basis_candidates),
        "bin_sizes_s": [0.02],
        "ridges": [0.1],
        "n_folds": 2,
        seed_name: 47,
        "spline_order": 4,
        "min_dark_firing_rate_hz": 0.5,
        "min_light_firing_rate_hz": 0.5,
        "use_speed": True,
        "speed_feature_mode": "linear",
        "n_splines_speed": 5,
        "spline_order_speed": 4,
        "speed_smoothing_sigma_s": 0.1,
        "speed_bounds": None,
        "trajectory_length_cm": trajectory_length_cm,
        "segment_edges": [0.0, 12.5 / 36.0, 23.5 / 36.0, 1.0],
    }
    if seed_name == "seed":
        parameters["dark_region_threshold_hz"] = 0.5
        parameters["light_region_threshold_hz"] = 0.5
    return parameters


def _dataset(
    *,
    metadata: dict[str, str],
    schema_version: str,
    mode: str,
    model_name: str,
    fit_stage: str,
    bin_size_s: float = 0.02,
    basis_value: float | int = 2.0,
):
    xr = pytest.importorskip("xarray")
    attrs: dict[str, object] = {
        **{name: metadata[name] for name in metadata if name != "dark_light_glm_id"},
        "schema_version": schema_version,
        "basis_candidate_mode": mode,
        "model_name": model_name,
        "fit_stage": fit_stage,
        "bin_size_s": bin_size_s,
        "n_splines": int(basis_value) if mode == "n_splines" else 18,
    }
    if mode == "spatial_bin_size_cm":
        attrs["spatial_bin_size_cm"] = float(basis_value)
    return xr.Dataset(
        data_vars={
            "dark_movement_firing_rate_hz": (
                "unit",
                np.asarray([1.0]),
            ),
            "light_movement_firing_rate_hz": (
                "unit",
                np.asarray([1.5]),
            ),
            "coef_intercept": (
                ("trajectory", "ridge", "unit")
                if fit_stage == "candidate"
                else ("trajectory", "unit"),
                (
                    np.ones((4, 1, 1), dtype=float)
                    if fit_stage == "candidate"
                    else np.ones((4, 1), dtype=float)
                ),
            ),
            "ll_bits_per_spike_cv_combined": (
                ("trajectory", "ridge", "unit"),
                np.full((4, 1, 1), 0.2, dtype=float),
            )
        },
        coords={
            "trajectory": np.asarray(TRAJECTORY_TYPES, dtype=str),
            "ridge": np.asarray([0.1]),
            "unit": np.asarray([0]),
        },
        attrs=attrs,
    )


def _result(
    result_id: uuid.UUID,
    *,
    mode: str = "spatial_bin_size_cm",
) -> dict[str, object]:
    xr = pytest.importorskip("xarray")
    metadata = _metadata(result_id)
    basis_candidates: tuple[float | int, ...]
    if mode == "spatial_bin_size_cm":
        basis_candidates = (2.0, 4.0)
        schema_version = "5"
    else:
        basis_candidates = (25, 40)
        schema_version = "4"
    parameters = {
        **module.validate_dark_light_glm_parameters(
            basis_candidate_mode=mode,
            basis_candidates=basis_candidates,
            bin_sizes_s=(0.02,),
            ridges=(0.1,),
            n_folds=2,
            min_dark_firing_rate_hz=0.5,
            min_light_firing_rate_hz=0.5,
        ),
        "parameter_name": "test",
        "parameter_sha256": "parameter",
        "output_rule_sha256": "output",
    }
    candidates = {}
    for basis_value in basis_candidates:
        key = module._candidate_key(
            "visual",
            bin_size_s=0.02,
            basis_candidate_mode=mode,
            basis_value=basis_value,
        )
        candidates[key] = _dataset(
            metadata=metadata,
            schema_version=schema_version,
            mode=mode,
            model_name="visual",
            fit_stage="candidate",
            basis_value=basis_value,
        )
    selected_basis = basis_candidates[0]
    for model_name in module.MODEL_NAMES[1:]:
        key = module._candidate_key(
            model_name,
            bin_size_s=0.02,
            basis_candidate_mode=mode,
            basis_value=selected_basis,
        )
        candidates[key] = _dataset(
            metadata=metadata,
            schema_version=schema_version,
            mode=mode,
            model_name=model_name,
            fit_stage="candidate",
            basis_value=selected_basis,
        )
    selected = {
        model_name: _dataset(
            metadata=metadata,
            schema_version=schema_version,
            mode=mode,
            model_name=model_name,
            fit_stage="selected",
            basis_value=selected_basis,
        )
        for model_name in module.MODEL_NAMES
    }
    summary = xr.Dataset(
        data_vars={"selected_ridge": ("model", np.full(4, 0.1))},
        coords={
            "model": np.asarray(module.MODEL_NAMES, dtype=str),
            mode: np.asarray(basis_candidates),
        },
        attrs={
            **{name: metadata[name] for name in metadata if name != "dark_light_glm_id"},
            "schema_version": schema_version,
            "basis_candidate_mode": mode,
            "selected_bin_size_s": 0.02,
            (
                "selected_spatial_bin_size_cm"
                if mode == "spatial_bin_size_cm"
                else "selected_n_splines"
            ): basis_candidates[0],
            "fit_parameters_json": json.dumps(
                _fit_parameters(mode=mode, basis_candidates=basis_candidates),
                sort_keys=True,
            ),
        },
    )
    return {
        "metadata": metadata,
        "parameters": parameters,
        "selected_units": _selected_units(),
        "candidate_datasets": candidates,
        "selected_datasets": selected,
        "selection_summary": summary,
        "trajectory_length_cm": 36.0,
        "segment_edges": np.asarray([0.0, 12.5 / 36.0, 23.5 / 36.0, 1.0]),
        "analysis_status": "valid",
        "artifact_origin": "computed",
        "legacy_artifact_provenance": None,
    }


def test_import_is_database_free_and_paths_are_session_first(tmp_path: Path) -> None:
    result_id = uuid.uuid4()
    paths = module.get_dark_light_glm_artifact_paths(
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        dark_light_glm_id=result_id,
        artifact_root=tmp_path,
    )

    assert paths["artifact_dir"] == (
        tmp_path
        / "L14"
        / "20240611"
        / "dark_light_glm"
        / "02_r1_vs_08_r4"
        / "v1"
        / str(result_id)
    )
    assert set(paths["selected_model_paths"]) == set(module.MODEL_NAMES)
    with pytest.raises(ValueError, match="path component"):
        module.get_dark_light_glm_artifact_paths(
            animal_name="../L14",
            date="20240611",
            region="v1",
            light_epoch="02_r1",
            dark_epoch="08_r4",
            dark_light_glm_id=result_id,
            artifact_root=tmp_path,
        )


def test_parameter_modes_remain_explicit() -> None:
    current = module.validate_dark_light_glm_parameters(
        basis_candidate_mode="spatial_bin_size_cm",
        basis_candidates=[2, 4, 8],
    )
    legacy = module.validate_dark_light_glm_parameters(
        basis_candidate_mode="n_splines",
        basis_candidates=[25, 40, 60],
    )

    assert current["schema_version"] == "5"
    assert current["basis_candidates"] == (2.0, 4.0, 8.0)
    assert legacy["schema_version"] == "4"
    assert legacy["basis_candidates"] == (25, 40, 60)
    with pytest.raises(ValueError, match="positive integers"):
        module.validate_dark_light_glm_parameters(
            basis_candidate_mode="n_splines",
            basis_candidates=[25.5],
        )


def test_graph_geometry_is_derived_without_animal_hardcoding() -> None:
    length, edges = module.derive_graph_geometry(_graphs())

    assert length == pytest.approx(36.0)
    assert edges == pytest.approx([0.0, 12.5 / 36.0, 23.5 / 36.0, 1.0])

    mismatched = _graphs()
    mismatched["center_to_right"] = _graph_payload(
        "center_to_right",
        long_edge=12.0,
    )
    with pytest.raises(ValueError, match="common path length"):
        module.derive_graph_geometry(mismatched)


@pytest.mark.parametrize("mode", ["spatial_bin_size_cm", "n_splines"])
def test_write_load_roundtrip_preserves_schema_contract(
    tmp_path: Path,
    mode: str,
) -> None:
    result_id = uuid.uuid4()
    result = _result(result_id, mode=mode)
    path = tmp_path / str(result_id)

    module.write_dark_light_glm_artifact(result, path)
    loaded = module.load_dark_light_glm_artifact(path)

    assert loaded["parameters"]["basis_candidate_mode"] == mode
    assert loaded["parameters"]["schema_version"] == (
        "5" if mode == "spatial_bin_size_cm" else "4"
    )
    assert loaded["n_candidates"] == 5
    assert set(loaded["selected_datasets"]) == set(module.MODEL_NAMES)
    if mode == "n_splines":
        assert "spatial_bin_size_cm" not in loaded["selection_summary"].dims
        assert "n_splines" in loaded["selection_summary"].dims


def test_load_detects_tampered_candidate(tmp_path: Path) -> None:
    result_id = uuid.uuid4()
    path = tmp_path / str(result_id)
    module.write_dark_light_glm_artifact(_result(result_id), path)
    candidate_path = next((path / module.CANDIDATE_DIRNAME).glob("*.nc"))

    candidate_path.write_bytes(candidate_path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_dark_light_glm_artifact(path)


def test_result_rejects_v4_artifact_claiming_spatial_candidates() -> None:
    result = _result(uuid.uuid4(), mode="n_splines")
    candidate = next(iter(result["candidate_datasets"].values()))
    candidate.attrs["spatial_bin_size_cm"] = 4.0

    with pytest.raises(ValueError, match="must not claim spatial-bin"):
        module.validate_dark_light_glm_result(result)


def test_register_existing_preserves_v4_mode_and_checks_selected_graphs(
    tmp_path: Path,
) -> None:
    source_result = _result(uuid.uuid4(), mode="n_splines")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    candidate_paths = []
    for index, dataset in enumerate(source_result["candidate_datasets"].values()):
        path = source_dir / f"candidate_{index}.nc"
        dataset.to_netcdf(path)
        candidate_paths.append(path)
    selected_paths = {}
    for model_name, dataset in source_result["selected_datasets"].items():
        path = source_dir / f"selected_{model_name}.nc"
        dataset.to_netcdf(path)
        selected_paths[model_name] = path
    summary = source_result["selection_summary"].copy(deep=True)
    summary.attrs["fit_parameters_json"] = json.dumps(
        _fit_parameters(
            mode="n_splines",
            basis_candidates=(25, 40),
            seed_name="seed",
        ),
        sort_keys=True,
    )
    summary_path = source_dir / "summary.nc"
    summary.to_netcdf(summary_path)

    result_id = uuid.uuid4()
    destination = tmp_path / str(result_id)
    registered = module.register_existing_dark_light_glm_artifact(
        source_candidate_paths=candidate_paths,
        source_selected_paths_by_model=selected_paths,
        source_selection_summary_path=summary_path,
        destination_path=destination,
        dark_light_glm_id=result_id,
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        unit_identity_resolver={
            "0": {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "group_unit_id": "group-0",
            }
        },
        graph_inputs_by_trajectory=_graphs(),
        basis_candidate_mode="n_splines",
        basis_candidates=(25, 40),
        parameter_name="legacy_v4",
    )

    assert registered["artifact_origin"] == "registered_existing"
    assert registered["parameters"]["schema_version"] == "4"
    assert registered["parameters"]["basis_candidate_mode"] == "n_splines"
    assert "n_splines" in registered["selection_summary"].dims
    assert registered["selected_units"]["group_unit_id"].tolist() == ["group-0"]
    assert registered["legacy_artifact_provenance"][
        "unit_identity_validation"
    ] == "caller_resolver_for_every_imported_unit"
    assert registered["legacy_artifact_provenance"][
        "source_selection_summary_sha256"
    ]

    with pytest.raises(ValueError, match="schema does not match"):
        module.register_existing_dark_light_glm_artifact(
            source_candidate_paths=candidate_paths,
            source_selected_paths_by_model=selected_paths,
            source_selection_summary_path=summary_path,
            destination_path=tmp_path / str(uuid.uuid4()),
            dark_light_glm_id=uuid.uuid4(),
            animal_name="L14",
            date="20240611",
            region="v1",
            light_epoch="02_r1",
            dark_epoch="08_r4",
            unit_identity_resolver={
                "0": {
                    "spikesorting_merge_id": "merge-a",
                    "unit_id": "11",
                }
            },
            graph_inputs_by_trajectory=_graphs(),
            basis_candidate_mode="spatial_bin_size_cm",
            basis_candidates=(25, 40),
            parameter_name="wrong_mode",
        )


def test_register_existing_accepts_current_random_seed_parameters(
    tmp_path: Path,
) -> None:
    source_result = _result(uuid.uuid4(), mode="spatial_bin_size_cm")
    source_dir = tmp_path / "current_source"
    source_dir.mkdir()
    candidate_paths = []
    for index, dataset in enumerate(source_result["candidate_datasets"].values()):
        path = source_dir / f"candidate_{index}.nc"
        dataset.to_netcdf(path)
        candidate_paths.append(path)
    selected_paths = {}
    for model_name, dataset in source_result["selected_datasets"].items():
        path = source_dir / f"selected_{model_name}.nc"
        dataset.to_netcdf(path)
        selected_paths[model_name] = path
    summary_path = source_dir / "summary.nc"
    source_result["selection_summary"].to_netcdf(summary_path)

    result_id = uuid.uuid4()
    registered = module.register_existing_dark_light_glm_artifact(
        source_candidate_paths=candidate_paths,
        source_selected_paths_by_model=selected_paths,
        source_selection_summary_path=summary_path,
        destination_path=tmp_path / str(result_id),
        dark_light_glm_id=result_id,
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        unit_identity_resolver={
            "0": {
                "spikesorting_merge_id": "merge-a",
                "unit_id": "11",
                "group_unit_id": "group-0",
            }
        },
        graph_inputs_by_trajectory=_graphs(),
        basis_candidate_mode="spatial_bin_size_cm",
        basis_candidates=(2.0, 4.0),
        parameter_name="current_v5",
    )

    assert registered["parameters"]["schema_version"] == "5"
    assert registered["parameters"]["random_seed"] == 47


def test_result_rejects_dataset_unit_order_mismatch() -> None:
    result = _result(uuid.uuid4())
    candidate = next(iter(result["candidate_datasets"].values()))
    candidate.coords["unit"] = np.asarray(["not-the-selected-group"])

    with pytest.raises(ValueError, match="unit coordinate/order"):
        module.validate_dark_light_glm_result(result)


def test_no_valid_units_remains_a_persisted_audit_status(tmp_path: Path) -> None:
    result_id = uuid.uuid4()
    result = _result(result_id)
    for dataset in result["selected_datasets"].values():
        dataset["coef_intercept"][:] = np.nan
    result["selected_units"]["n_selected_model_trajectory_fits"] = 0
    result["selected_units"]["valid_glm_fit"] = False
    result["analysis_status"] = "no_valid_units"

    destination = tmp_path / str(result_id)
    module.write_dark_light_glm_artifact(result, destination)
    loaded = module.load_dark_light_glm_artifact(destination)

    assert loaded["analysis_status"] == "no_valid_units"
    assert not loaded["selected_units"]["valid_glm_fit"].any()


def test_terminal_bundle_roundtrip_has_real_selected_files(tmp_path: Path) -> None:
    result_id = uuid.uuid4()
    source = _result(result_id)
    result = module._terminal_result(
        metadata=source["metadata"],
        parameters=source["parameters"],
        trajectory_length_cm=source["trajectory_length_cm"],
        segment_edges=source["segment_edges"],
        analysis_status="no_eligible_units",
    )
    destination = tmp_path / str(result_id)

    paths = module.write_dark_light_glm_artifact(result, destination)
    loaded = module.load_dark_light_glm_artifact(destination)

    assert loaded["analysis_status"] == "no_eligible_units"
    assert loaded["candidate_datasets"] == {}
    assert loaded["selected_units"].empty
    assert all(path.is_file() for path in paths["selected_model_paths"].values())


def test_speed_derivation_uses_frozen_smoothing_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_speed(position, timestamps, **kwargs):
        observed.append(kwargs["speed_smoothing_sigma_s"])
        return object()

    monkeypatch.setattr("v1ca1.helper.session.build_speed_tsd", fake_speed)
    positions = {
        epoch: SimpleNamespace(
            d=np.zeros((2, 2), dtype=float),
            t=np.asarray([0.0, 1.0]),
        )
        for epoch in ("08_r4", "02_r1")
    }

    module._derive_speed(
        positions,
        ("08_r4", "02_r1"),
        speed_smoothing_sigma_s=0.125,
    )

    assert observed == [0.125, 0.125]


def test_population_adapter_preserves_partial_unit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted = SimpleNamespace(
        coef_=np.asarray([[1.0, np.nan], [2.0, np.nan]]),
        intercept_=np.asarray([0.5, np.nan]),
    )
    monkeypatch.setitem(
        sys.modules,
        "v1ca1.task_progression.motor",
        SimpleNamespace(
            fit_population_glm_isolating_unit_failures=(
                lambda *_args, **_kwargs: fitted
            )
        ),
    )
    model = module._IsolatingPopulationGLM(regularizer_strength=0.1)

    returned = model.fit(np.ones((3, 2)), np.ones((3, 2)))

    assert returned is model
    assert np.isfinite(model.intercept_[0])
    assert np.isnan(model.intercept_[1])


def test_compute_orchestration_normalizes_explicit_spline_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xr = pytest.importorskip("xarray")
    result_id = uuid.uuid4()
    metadata = _metadata(result_id)

    def score_candidate(dataset):
        return [
            {
                "model_name": dataset.attrs["model_name"],
                "bin_size_s": dataset.attrs["bin_size_s"],
                "spatial_bin_size_cm": dataset.attrs["spatial_bin_size_cm"],
                "trajectory_length_cm": 36.0,
                "n_splines": dataset.attrs["n_splines"],
                "ridge": 0.1,
                "score_median": float(dataset.attrs["n_splines"]),
                "score_mean": float(dataset.attrs["n_splines"]),
                "n_finite": 4,
            }
        ]

    def choose_shared(records):
        record = max(
            (record for record in records if record["model_name"] == "visual"),
            key=lambda record: record["score_median"],
        )
        return {
            **record,
            "visual_ridge": record["ridge"],
        }

    def choose_ridge(records, *, model_name, bin_size_s, spatial_bin_size_cm):
        return next(
            record
            for record in records
            if record["model_name"] == model_name
            and np.isclose(record["bin_size_s"], bin_size_s)
            and np.isclose(record["spatial_bin_size_cm"], spatial_bin_size_cm)
        )

    def build_selected(candidate, **_kwargs):
        selected = candidate.copy(deep=True)
        selected.attrs["fit_stage"] = "selected"
        selected.attrs["selected_spatial_bin_size_cm"] = candidate.attrs[
            "spatial_bin_size_cm"
        ]
        return selected

    def build_summary(**kwargs):
        configs = kwargs["position_basis_configs"]
        return xr.Dataset(
            data_vars={
                "n_splines_by_spatial_bin_size": (
                    "spatial_bin_size_cm",
                    np.asarray([config["n_splines"] for config in configs]),
                )
            },
            coords={
                "spatial_bin_size_cm": np.asarray(
                    [config["spatial_bin_size_cm"] for config in configs]
                )
            },
            attrs={
                "schema_version": "5",
                "animal_name": kwargs["animal_name"],
                "date": kwargs["date"],
                "region": kwargs["region"],
                "light_epoch": kwargs["light_epoch"],
                "dark_epoch": kwargs["dark_epoch"],
                "selected_bin_size_s": kwargs["shared_selection"]["bin_size_s"],
                "selected_spatial_bin_size_cm": kwargs["shared_selection"][
                    "spatial_bin_size_cm"
                ],
                "selected_n_splines": kwargs["shared_selection"]["n_splines"],
                "fit_parameters_json": json.dumps(kwargs["fit_parameters"]),
            },
        )

    fake_analysis = SimpleNamespace(
        n_splines_from_spatial_bin_size=lambda length, size, spline_order: max(
            spline_order,
            int(np.ceil(length / size)),
        ),
        build_train_epoch_fr_mask=lambda *args, **kwargs: {
            "combined": np.asarray([True])
        },
        build_lap_cv_folds_for_trajectory=lambda **kwargs: [
            {"metadata": {"fold_index": index}}
            for index in range(kwargs["n_folds"])
        ],
        score_candidate_dataset=score_candidate,
        choose_visual_shared_hyperparameters=choose_shared,
        choose_model_ridge=choose_ridge,
        build_selected_model_dataset=build_selected,
        build_selection_summary_dataset=build_summary,
    )
    monkeypatch.setattr(module, "_analysis_module", lambda: fake_analysis)
    monkeypatch.setattr(
        module,
        "_identity_and_rates",
        lambda **kwargs: (
            _selected_units(),
            np.asarray([True]),
            np.asarray([1.0]),
            np.asarray([1.5]),
            "valid",
        ),
    )

    def fake_fit_candidate(**kwargs):
        basis = kwargs["position_basis"]
        dataset = _dataset(
            metadata=metadata,
            schema_version="5",
            mode="spatial_bin_size_cm",
            model_name=kwargs["model_name"],
            fit_stage="candidate",
            basis_value=basis["spatial_bin_size_cm"],
        )
        dataset.attrs["n_splines"] = basis["n_splines"]
        return dataset

    monkeypatch.setattr(module, "_fit_candidate_dataset", fake_fit_candidate)
    cache_cleanup_calls = []
    monkeypatch.setattr(
        module,
        "_clear_jax_fit_caches",
        lambda: cache_cleanup_calls.append(None),
    )
    epoch_paths = {
        epoch: {trajectory: object() for trajectory in TRAJECTORY_TYPES}
        for epoch in ("08_r4", "02_r1")
    }
    result = module.compute_dark_light_glm(
        dark_light_glm_id=result_id,
        animal_name="L14",
        date="20240611",
        region="v1",
        light_epoch="02_r1",
        dark_epoch="08_r4",
        spikes=object(),
        stable_unit_ids=[],
        dark_movement_firing_rate_table=pd.DataFrame(),
        light_movement_firing_rate_table=pd.DataFrame(),
        movement_by_epoch={"08_r4": object(), "02_r1": object()},
        trajectory_intervals_by_epoch=epoch_paths,
        graph_inputs_by_trajectory=_graphs(),
        task_progression_by_epoch=epoch_paths,
        basis_candidate_mode="n_splines",
        basis_candidates=(25, 40),
        bin_sizes_s=(0.02,),
        ridges=(0.1,),
        n_folds=2,
        use_speed=False,
    )

    assert result["parameters"]["schema_version"] == "4"
    assert result["n_candidates"] == 5
    assert len(cache_cleanup_calls) == 5
    assert "n_splines" in result["selection_summary"].dims
    assert all(
        "spatial_bin_size_cm" not in dataset.attrs
        for dataset in result["candidate_datasets"].values()
    )
