"""Tests for the database-free Figure 3 artifact adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from v1ca1.paper_figures import figure_3_spyglass as figure


def _sessions() -> list[dict[str, Any]]:
    """Return four canonical sessions with all expected artifact roles."""
    sessions = []
    for animal_name, date, _dark_epoch in figure.EXPECTED_DATASETS:
        special = (animal_name, date) == ("L15", "20241121")
        sessions.append(
            {
                "animal_name": animal_name,
                "date": date,
                "epochs": {"light": figure.LIGHT_EPOCH},
                "regions": list(figure.REGIONS),
                "artifacts": {
                    "ripple_modulation": [
                        {
                            "epoch": figure.LIGHT_EPOCH,
                            "region": region,
                            "artifact_origin": "computed",
                        }
                        for region in figure.REGIONS
                    ],
                    "ripple_glm": [
                        {
                            "epoch": figure.LIGHT_EPOCH,
                            "source_predictor_mode": mode,
                            "artifact_origin": "computed",
                        }
                        for mode in figure.GLM_SOURCE_MODES
                    ],
                    "ripple_cross_region_xcorr": (
                        [
                            {
                                "epoch": figure.LIGHT_EPOCH,
                                "artifact_origin": "computed",
                            }
                        ]
                        if special
                        else []
                    ),
                    "panel_b_schematic": (
                        [
                            {
                                "epoch": figure.LIGHT_EPOCH,
                                "artifact_origin": "computed",
                            }
                        ]
                        if special
                        else []
                    ),
                },
            }
        )
    return sessions


def _payload(run_dir: Path) -> dict[str, Any]:
    """Return a minimal already-adapted Figure 3 payload."""
    return {
        "run_dir": run_dir,
        "datasets": figure.EXPECTED_DATASETS,
        "regions": figure.REGIONS,
        "heatmap_epoch_tables": [{"payload": "heatmap"}],
        "glm_epoch_tables": [{"payload": "glm"}],
        "schematic_payload": {"payload": "schematic"},
        "prediction_examples": [{"payload": "prediction"}],
        "behavior_payload": {"payload": "behavior"},
        "source_comparison_payload": {"payload": "source-comparison"},
        "xcorr_payload": {"payload": "xcorr"},
    }


def test_session_order_and_exact_campaign_artifact_multiplicity() -> None:
    sessions = _sessions()

    ordered = figure._ordered_sessions(list(reversed(sessions)))

    assert [
        (row["animal_name"], row["date"]) for row in ordered
    ] == [
        (animal_name, date)
        for animal_name, date, _dark_epoch in figure.EXPECTED_DATASETS
    ]
    totals = {
        family: sum(len(session["artifacts"][family]) for session in ordered)
        for family in (
            "ripple_modulation",
            "ripple_glm",
            "ripple_cross_region_xcorr",
            "panel_b_schematic",
        )
    }
    assert totals == {
        "ripple_modulation": 8,
        "ripple_glm": 8,
        "ripple_cross_region_xcorr": 1,
        "panel_b_schematic": 1,
    }

    with pytest.raises(ValueError, match="exactly the four manuscript sessions"):
        figure._ordered_sessions(sessions[:-1])
    changed = [dict(row) for row in sessions]
    changed[0] = {**changed[0], "epochs": {"light": "06_r3"}}
    with pytest.raises(ValueError, match="noncanonical Figure 3 epochs"):
        figure._ordered_sessions(changed)


def test_unit_table_maps_stable_nwb_ids_to_sorting_ids() -> None:
    table = pd.DataFrame.from_records(
        [
            {
                "spikesorting_merge_id": "merge",
                "unit_id": "101",
                "stable_unit_id": "merge:101",
                "group_unit_id": "0",
                "region": "v1",
            },
            {
                "spikesorting_merge_id": "merge",
                "unit_id": "102",
                "stable_unit_id": "merge:102",
                "group_unit_id": "1",
                "region": "v1",
            },
        ]
    )

    mapped = figure._map_unit_table(
        table,
        region="v1",
        sorting_unit_by_nwb_id={"101": 24, "102": 32},
        label="synthetic table",
    )

    assert mapped["nwb_unit_id"].tolist() == ["101", "102"]
    assert mapped["unit_id"].tolist() == [24, 32]
    changed = table.copy()
    changed.loc[0, "stable_unit_id"] = "merge:other"
    with pytest.raises(ValueError, match="inconsistent stable unit identities"):
        figure._map_unit_table(
            changed,
            region="v1",
            sorting_unit_by_nwb_id={"101": 24, "102": 32},
            label="synthetic table",
        )


def test_schematic_unit_mapping_audits_embedded_sorting_ids() -> None:
    payload = {
        "ca1_unit_ids": np.asarray(["101"]),
        "v1_unit_ids": np.asarray(["201"]),
        "ca1_unit_identity": [{"unit_id": 101, "sorting_unit_id": 10}],
        "v1_unit_identity": [{"unit_id": 201, "sorting_unit_id": 20}],
    }

    mapped = figure._map_schematic_unit_ids(
        payload,
        unit_maps={"ca1": {"101": 10}, "v1": {"201": 20}},
    )

    assert mapped["ca1_nwb_unit_ids"].tolist() == ["101"]
    assert mapped["v1_nwb_unit_ids"].tolist() == ["201"]
    assert mapped["ca1_unit_ids"].tolist() == [10]
    assert mapped["v1_unit_ids"].tolist() == [20]
    changed = {**payload, "v1_unit_identity": [{"unit_id": 201, "sorting_unit_id": 21}]}
    with pytest.raises(ValueError, match="sorting identity is stale"):
        figure._map_schematic_unit_ids(
            changed,
            unit_maps={"ca1": {"101": 10}, "v1": {"201": 20}},
        )


def test_schematic_loader_validates_nested_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the schema returned by the real pickle-free payload loader."""
    from v1ca1.spyglass.offline import figure_3 as offline_figure

    run_dir = tmp_path / "runs" / "figure3"
    run_dir.mkdir(parents=True)
    sessions = _sessions()
    l15 = next(row for row in sessions if row["animal_name"] == "L15")
    record = l15["artifacts"]["panel_b_schematic"][0]
    record.update(
        {
            "payload_path": "L15/20241121/figure_payloads/panel_b_schematic.npz",
            "artifact_sha256": {"payload_path": "digest"},
        }
    )
    payload = {
        "metadata": {
            "animal_name": "L15",
            "date": "20241121",
            "epoch": figure.LIGHT_EPOCH,
        },
        "ca1_unit_ids": np.asarray(["101"]),
        "v1_unit_ids": np.asarray(["201"]),
        "ca1_unit_identity": [{"unit_id": 101, "sorting_unit_id": 10}],
        "v1_unit_identity": [{"unit_id": 201, "sorting_unit_id": 20}],
    }
    monkeypatch.setattr(
        offline_figure,
        "load_panel_b_schematic_payload",
        lambda _path, *, expected_sha256: payload,
    )

    loaded = figure._load_schematic_payload(
        run_dir,
        sessions,
        {("L15", "20241121"): {"ca1": {"101": 10}, "v1": {"201": 20}}},
    )

    assert loaded["ca1_unit_ids"].tolist() == [10]
    assert loaded["v1_unit_ids"].tolist() == [20]


def test_glm_validator_requires_fixed_light_gpu_campaign_parameters() -> None:
    result = {
        "animal_name": "L14",
        "date": "20240611",
        "epoch": "02_r1",
        "artifact_origin": "computed",
        "parameters": {
            "source_predictor_mode": "unit_vector",
            "ripple_selection_mode": "single",
            "ripple_window_s": 0.2,
            "ripple_window_offset_s": 0.0,
            "ridge_strength": 0.1,
            "n_shuffles_ripple": 100,
            "expected_detector_zscore_threshold": 2.0,
            "require_speed_gated": True,
        },
    }

    figure._validate_glm_result(
        result,
        animal_name="L14",
        date="20240611",
        mode="unit_vector",
    )

    changed = {
        **result,
        "parameters": {
            **result["parameters"],
            "expected_detector_zscore_threshold": 2.5,
        },
    }
    with pytest.raises(ValueError, match="expected_detector_zscore_threshold"):
        figure._validate_glm_result(
            changed,
            animal_name="L14",
            date="20240611",
            mode="unit_vector",
        )


def test_source_injection_never_opens_legacy_artifacts_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure3"
    run_dir.mkdir(parents=True)
    payload = _payload(run_dir)
    loader_names = (
        "load_pooled_ripple_heatmap_epoch_tables",
        "load_glm_epoch_summary_tables",
        "load_or_build_panel_b_schematic_example",
        "load_panel_b_prediction_examples",
        "load_glm_dark_activity_devexp_tables",
        "load_glm_source_predictor_comparison_tables",
        "load_top_ca1_xcorr_panel_data",
        "load_example_ripple_lfp_trace",
        "build_panel_b_schematic_example",
        "load_first_available_glm_prediction",
    )
    legacy_attempts: list[str] = []

    def legacy_file_access(*_args: Any, **_kwargs: Any) -> None:
        legacy_attempts.append("legacy")
        raise AssertionError("A legacy artifact loader was reached.")

    for name in loader_names:
        monkeypatch.setattr(figure.legacy, name, legacy_file_access)

    common = {
        "light_epoch": figure.LIGHT_EPOCH,
        "dark_epoch": None,
        "sleep_epoch": None,
    }
    glm = {
        "ripple_window_s": figure.legacy.DEFAULT_RIPPLE_WINDOW_S,
        "ripple_window_offset_s": figure.legacy.DEFAULT_RIPPLE_WINDOW_OFFSET_S,
        "ripple_selection": figure.legacy.DEFAULT_FIGURE_3_GLM_RIPPLE_SELECTION,
        "ridge_strength": figure.legacy.DEFAULT_RIDGE_STRENGTH,
    }

    with figure._offline_sources(payload):
        assert figure.legacy.load_pooled_ripple_heatmap_epoch_tables(
            run_dir,
            figure.EXPECTED_DATASETS,
            **common,
            ripple_threshold_zscore=None,
        ) is payload["heatmap_epoch_tables"]
        assert figure.legacy.load_glm_epoch_summary_tables(
            run_dir,
            figure.EXPECTED_DATASETS,
            **common,
            **glm,
            epoch_types=figure.legacy.PANEL_C_EPOCH_ORDER,
        ) is payload["glm_epoch_tables"]
        schematic_animal, schematic_date, schematic_epoch = (
            figure.legacy.DEFAULT_PANEL_B_SCHEMATIC_DATASET
        )
        assert figure.legacy.load_or_build_panel_b_schematic_example(
            run_dir,
            animal_name=schematic_animal,
            date=schematic_date,
            epoch=schematic_epoch,
            ripple_threshold_zscore=None,
        ) is payload["schematic_payload"]
        assert figure.legacy.load_panel_b_prediction_examples(
            run_dir,
            **glm,
        ) is payload["prediction_examples"]
        assert figure.legacy.load_glm_dark_activity_devexp_tables(
            run_dir,
            figure.EXPECTED_DATASETS,
            **common,
            **glm,
            region="v1",
            epoch_types=figure.legacy.PANEL_D_EPOCH_ORDER,
            tuning_similarity_metric=(
                figure.legacy.DEFAULT_PANEL_D_TUNING_SIMILARITY_METRIC
            ),
        ) is payload["behavior_payload"]
        assert figure.legacy.load_glm_source_predictor_comparison_tables(
            run_dir,
            figure.EXPECTED_DATASETS,
            **common,
            **glm,
            epoch_types=figure.legacy.PANEL_E_GLM_EPOCH_ORDER,
        ) is payload["source_comparison_payload"]
        xcorr_animal, xcorr_date, xcorr_epoch = figure.legacy.DEFAULT_XCORR_DATASET
        assert figure.legacy.load_top_ca1_xcorr_panel_data(
            run_dir,
            animal_name=xcorr_animal,
            date=xcorr_date,
            epoch=xcorr_epoch,
            state=figure.legacy.DEFAULT_XCORR_STATE,
            top_n_ca1_units=figure.legacy.DEFAULT_XCORR_TOP_CA1_UNITS,
            max_lag_s=figure.legacy.DEFAULT_XCORR_MAX_LAG_S,
            bin_size_s=figure.legacy.DEFAULT_XCORR_BIN_SIZE_S,
            display_vmax=figure.legacy.DEFAULT_XCORR_DISPLAY_VMAX,
        ) is payload["xcorr_payload"]
        with pytest.raises(figure._UnexpectedLegacyRequest, match="fallback"):
            figure.legacy.load_example_ripple_lfp_trace(run_dir)
        with pytest.raises(figure._UnexpectedLegacyRequest, match="fallback"):
            figure.legacy.load_first_available_glm_prediction(run_dir)

    assert not legacy_attempts
    assert all(
        getattr(figure.legacy, name) is legacy_file_access for name in loader_names
    )


def test_renderer_is_run_local_atomic_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure3"
    run_dir.mkdir(parents=True)
    payload = _payload(run_dir)
    calls: list[dict[str, Any]] = []

    def legacy_file_access(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Renderer attempted to open legacy artifacts.")

    monkeypatch.setattr(
        figure.legacy,
        "load_pooled_ripple_heatmap_epoch_tables",
        legacy_file_access,
    )

    def make_figure(**kwargs: Any) -> Path:
        calls.append(kwargs)
        assert figure.legacy.load_pooled_ripple_heatmap_epoch_tables(
            kwargs["data_root"],
            kwargs["datasets"],
            light_epoch=kwargs["light_epoch"],
            dark_epoch=kwargs["dark_epoch"],
            sleep_epoch=kwargs["sleep_epoch"],
            ripple_threshold_zscore=kwargs["ripple_threshold_zscore"],
        ) is payload["heatmap_epoch_tables"]
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True)
        output_path.write_text("figure", encoding="utf-8")
        return output_path

    monkeypatch.setattr(figure.legacy, "make_figure_3", make_figure)
    output = figure.get_output_path(run_dir=run_dir, output_format="svg")

    returned = figure.render_figure_3(payload, output_path=output)

    assert returned == output.resolve()
    assert output.read_text(encoding="utf-8") == "figure"
    assert calls[0]["data_root"] == run_dir.resolve()
    assert calls[0]["output_path"] != output
    assert calls[0]["light_epoch"] == "02_r1"
    assert calls[0]["dark_epoch"] is None
    assert calls[0]["sleep_epoch"] is None
    assert calls[0]["ripple_threshold_zscore"] is None
    assert (
        figure.legacy.load_pooled_ripple_heatmap_epoch_tables
        is legacy_file_access
    )
    assert not list(output.parent.glob(".*.tmp.svg"))
    with pytest.raises(FileExistsError, match="overwrite"):
        figure.render_figure_3(payload, output_path=output)
    with pytest.raises(ValueError, match="inside its campaign run"):
        figure.render_figure_3(
            payload,
            output_path=tmp_path / "outside.svg",
        )
    with pytest.raises(ValueError, match="unsupported format"):
        figure.render_figure_3(
            payload,
            output_path=run_dir / "figures" / "figure.txt",
        )


def test_failed_renderer_removes_only_its_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure3"
    run_dir.mkdir(parents=True)
    payload = _payload(run_dir)

    def fail_after_partial_write(**kwargs: Any) -> None:
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True)
        output_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(figure.legacy, "make_figure_3", fail_after_partial_write)
    output = run_dir / "figures" / "figure_3_spyglass.svg"

    with pytest.raises(RuntimeError, match="synthetic save failure"):
        figure.render_figure_3(payload, output_path=output)

    assert not output.exists()
    assert not list(output.parent.glob(".*.tmp.svg"))


def test_supplement_replaces_only_schematic_and_moves_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_run_dir = tmp_path / "runs" / "base"
    supplement_run_dir = tmp_path / "runs" / "supplement"
    base_run_dir.mkdir(parents=True)
    supplement_run_dir.mkdir(parents=True)
    sessions = _sessions()
    campaign = {"analysis_parameters": {"pipeline": "figure_3"}}
    sentinels = {
        "heatmap_epoch_tables": object(),
        "glm_epoch_tables": object(),
        "prediction_examples": object(),
        "behavior_payload": object(),
        "source_comparison_payload": object(),
        "xcorr_payload": object(),
    }
    monkeypatch.setattr(
        "v1ca1.spyglass.offline.figure_3.load_figure_3_campaign",
        lambda *_args, **_kwargs: (base_run_dir, campaign, sessions),
    )
    monkeypatch.setattr(figure, "_ordered_sessions", lambda values: list(values))
    monkeypatch.setattr(
        figure,
        "_load_nwb_sorting_unit_maps",
        lambda _session: {"ca1": {"101": 10}, "v1": {"201": 20}},
    )
    monkeypatch.setattr(figure, "_load_glm_results", lambda *_args: {})
    monkeypatch.setattr(
        figure,
        "_load_modulation_epoch_tables",
        lambda *_args: sentinels["heatmap_epoch_tables"],
    )
    monkeypatch.setattr(
        figure,
        "_build_glm_epoch_tables",
        lambda *_args: sentinels["glm_epoch_tables"],
    )
    monkeypatch.setattr(
        figure,
        "_load_schematic_payload",
        lambda *_args: {"payload": "base-schematic"},
    )
    monkeypatch.setattr(
        figure,
        "_build_prediction_examples",
        lambda *_args: sentinels["prediction_examples"],
    )
    monkeypatch.setattr(
        figure,
        "_build_behavior_payload",
        lambda *_args, **_kwargs: sentinels["behavior_payload"],
    )
    monkeypatch.setattr(
        figure,
        "_build_source_comparison_payload",
        lambda *_args: sentinels["source_comparison_payload"],
    )
    monkeypatch.setattr(
        figure,
        "_build_xcorr_payload",
        lambda *_args: sentinels["xcorr_payload"],
    )
    supplement_payload = {
        "ca1_unit_ids": np.asarray(["101"]),
        "v1_unit_ids": np.asarray(["201"]),
        "ca1_unit_identity": [{"unit_id": 101, "sorting_unit_id": 10}],
        "v1_unit_identity": [{"unit_id": 201, "sorting_unit_id": 20}],
    }
    calls: list[tuple[str, str | None]] = []

    def load_supplement(
        run_id: str,
        *,
        expected_base_run_id: str | None,
        scratch_root: Path,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        calls.append((run_id, expected_base_run_id))
        return supplement_run_dir, {"run_id": run_id}, supplement_payload

    monkeypatch.setattr(
        "v1ca1.spyglass.offline.figure_3_schematic_supplement."
        "load_figure_3_schematic_supplement",
        load_supplement,
    )

    payload = figure.load_figure_3_payload(
        run_id="base",
        supplement_run_id="supplement",
        scratch_root=tmp_path,
    )

    assert calls == [("supplement", "base")]
    assert payload["run_dir"] == supplement_run_dir
    assert payload["base_run_dir"] == base_run_dir
    assert payload["campaign"] is campaign
    assert payload["sessions"] == sessions
    assert payload["schematic_payload"]["ca1_unit_ids"].tolist() == [10]
    assert payload["schematic_payload"]["v1_unit_ids"].tolist() == [20]
    for name, sentinel in sentinels.items():
        assert payload[name] is sentinel
    assert figure.get_output_path(run_dir=payload["run_dir"]).is_relative_to(
        supplement_run_dir
    )


def test_renderer_cli_forwards_optional_supplement_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplement_run_dir = tmp_path / "runs" / "supplement"
    supplement_run_dir.mkdir(parents=True)
    calls: list[dict[str, Any]] = []

    def load_payload(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"run_dir": supplement_run_dir}

    rendered: list[Path] = []
    monkeypatch.setattr(figure, "load_figure_3_payload", load_payload)
    monkeypatch.setattr(
        figure,
        "render_figure_3",
        lambda _payload, *, output_path, dpi: rendered.append(Path(output_path)),
    )

    figure.main(
        [
            "--run-id",
            "base",
            "--supplement-run-id",
            "supplement",
            "--scratch-root",
            str(tmp_path),
        ]
    )

    assert calls[0]["run_id"] == "base"
    assert calls[0]["supplement_run_id"] == "supplement"
    assert rendered == [supplement_run_dir / "figures" / "figure_3_spyglass.svg"]
