"""Tests for the database-free Supplementary Figure 7 adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from v1ca1.paper_figures import supplementary_figure_7_spyglass as figure


def _sessions() -> list[dict[str, Any]]:
    """Return the four fixed light-only Figure 3 session identities."""
    return [
        {
            "animal_name": animal_name,
            "date": date,
            "epochs": {"light": figure.LIGHT_EPOCH},
            "regions": list(figure.REGIONS),
        }
        for animal_name, date, _dark_epoch in figure.EXPECTED_DATASETS
    ]


def _payload(run_dir: Path) -> dict[str, Any]:
    """Return one minimal validated Supplementary Figure 7 payload."""
    return {
        "run_dir": run_dir,
        "campaign": {
            "run_id": run_dir.name,
            "analysis_parameters": {"pipeline": "figure_3"},
        },
        "sessions": _sessions(),
        "datasets": figure.EXPECTED_DATASETS,
        "regions": figure.REGIONS,
        "heatmap_epoch_tables": [{"kind": "modulation"}],
        "source_comparison_payload": {"kind": "source-comparison"},
    }


def _write_campaign_manifest(payload: dict[str, Any]) -> None:
    """Write the exact campaign snapshot required by a render receipt."""
    run_dir = Path(payload["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(payload["campaign"]),
        encoding="utf-8",
    )


def test_adapter_targets_supplementary_figure_7_and_cli_defaults() -> None:
    assert figure.canonical.DEFAULT_OUTPUT_NAME == "supplementary_figure_7"
    assert figure.DEFAULT_OUTPUT_NAME == "supplementary_figure_7_spyglass"
    assert figure.REGIONS == ("ca1", "v1")

    args = figure.parse_arguments(["--run-id", "figure3-nwb-gpu-v2"])

    assert args.run_id == "figure3-nwb-gpu-v2"
    assert args.output_format == "svg"
    assert args.dpi == 300
    assert args.promote_to is None
    with pytest.raises(SystemExit):
        figure.parse_arguments(
            [
                "--run-id",
                "figure3-nwb-gpu-v2",
                "--replace-promoted-output",
            ]
        )


def test_payload_loads_only_modulation_and_source_comparison_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from v1ca1.spyglass.offline import figure_3 as offline_figure_3

    run_dir = tmp_path / "runs" / "figure3-nwb-gpu-v2"
    run_dir.mkdir(parents=True)
    sessions = _sessions()
    campaign = {
        "run_id": "figure3-nwb-gpu-v2",
        "analysis_parameters": {
            "pipeline": offline_figure_3.FIGURE_3_PIPELINE,
        },
    }
    calls: list[tuple[str, Any]] = []
    unit_maps = {
        (str(row["animal_name"]), str(row["date"])): {
            "ca1": {},
            "v1": {},
        }
        for row in sessions
    }
    glm_results = {"glm": "results"}
    heatmaps = [{"kind": "modulation"}]
    comparison = {"kind": "comparison"}

    monkeypatch.setattr(
        offline_figure_3,
        "load_figure_3_campaign",
        lambda *_args, **_kwargs: (run_dir, campaign, sessions),
    )

    def load_map(session: dict[str, Any]) -> dict[str, dict[str, int]]:
        key = (str(session["animal_name"]), str(session["date"]))
        calls.append(("unit-map", key))
        return unit_maps[key]

    monkeypatch.setattr(
        figure.figure_4_adapter,
        "_load_nwb_sorting_unit_maps",
        load_map,
    )
    monkeypatch.setattr(
        figure.figure_4_adapter,
        "_load_glm_results",
        lambda *args: calls.append(("glm", args)) or glm_results,
    )
    monkeypatch.setattr(
        figure.figure_4_adapter,
        "_load_modulation_epoch_tables",
        lambda *args: calls.append(("modulation", args)) or heatmaps,
    )
    monkeypatch.setattr(
        figure.figure_4_adapter,
        "_build_source_comparison_payload",
        lambda *args: calls.append(("comparison", args)) or comparison,
    )
    monkeypatch.setattr(
        figure.figure_4_adapter,
        "load_figure_4_payload",
        lambda **_kwargs: pytest.fail("The full Figure 4 payload was loaded."),
    )

    payload = figure.load_supplementary_figure_7_payload(
        run_id="figure3-nwb-gpu-v2",
        scratch_root=tmp_path,
    )

    assert payload["run_dir"] == run_dir
    assert payload["campaign"] is campaign
    assert payload["sessions"] == sessions
    assert payload["heatmap_epoch_tables"] is heatmaps
    assert payload["source_comparison_payload"] is comparison
    assert [name for name, _value in calls].count("unit-map") == 4
    assert [name for name, _value in calls].count("glm") == 1
    assert [name for name, _value in calls].count("modulation") == 1
    assert [name for name, _value in calls].count("comparison") == 1


def test_source_injection_forbids_legacy_fallback_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure3"
    run_dir.mkdir(parents=True)
    payload = _payload(run_dir)
    legacy_attempts: list[str] = []

    def legacy_loader(*_args: Any, **_kwargs: Any) -> None:
        legacy_attempts.append("legacy")
        raise AssertionError("A legacy loader was reached.")

    monkeypatch.setattr(
        figure.canonical,
        "load_pooled_ripple_heatmap_epoch_tables",
        legacy_loader,
    )
    monkeypatch.setattr(
        figure.canonical,
        "load_glm_source_predictor_comparison_tables",
        legacy_loader,
    )
    common = {
        "light_epoch": figure.LIGHT_EPOCH,
        "dark_epoch": None,
        "sleep_epoch": None,
    }
    glm = {
        "ripple_window_s": figure.RIPPLE_WINDOW_S,
        "ripple_window_offset_s": figure.RIPPLE_WINDOW_OFFSET_S,
        "ripple_selection": figure.RIPPLE_SELECTION,
        "ridge_strength": figure.RIDGE_STRENGTH,
    }

    with figure._offline_sources(payload):
        assert figure.canonical.load_pooled_ripple_heatmap_epoch_tables(
            run_dir,
            figure.EXPECTED_DATASETS,
            **common,
            ripple_threshold_zscore=None,
        ) is payload["heatmap_epoch_tables"]
        assert figure.canonical.load_glm_source_predictor_comparison_tables(
            run_dir,
            figure.EXPECTED_DATASETS,
            **common,
            **glm,
            epoch_types=figure.canonical.PANEL_E_GLM_EPOCH_ORDER,
        ) is payload["source_comparison_payload"]
        with pytest.raises(
            figure._UnexpectedLegacyRequest,
            match="cannot rethreshold",
        ):
            figure.canonical.load_pooled_ripple_heatmap_epoch_tables(
                run_dir,
                figure.EXPECTED_DATASETS,
                **common,
                ripple_threshold_zscore=2.0,
            )
        with pytest.raises(
            figure._UnexpectedLegacyRequest,
            match="foreign GLM epochs",
        ):
            figure.canonical.load_glm_source_predictor_comparison_tables(
                run_dir,
                figure.EXPECTED_DATASETS,
                **common,
                **glm,
                epoch_types=("dark",),
            )

    assert not legacy_attempts
    assert (
        figure.canonical.load_pooled_ripple_heatmap_epoch_tables
        is legacy_loader
    )
    assert (
        figure.canonical.load_glm_source_predictor_comparison_tables
        is legacy_loader
    )


def test_render_is_atomic_run_local_receipted_and_promotable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "runs" / "figure3"
    payload = _payload(run_dir)
    _write_campaign_manifest(payload)
    calls: list[dict[str, Any]] = []

    def make_figure(**kwargs: Any) -> Path:
        calls.append(kwargs)
        assert figure.canonical.load_pooled_ripple_heatmap_epoch_tables(
            kwargs["data_root"],
            kwargs["datasets"],
            light_epoch=kwargs["light_epoch"],
            dark_epoch=kwargs["dark_epoch"],
            sleep_epoch=kwargs["sleep_epoch"],
            ripple_threshold_zscore=kwargs["ripple_threshold_zscore"],
        ) is payload["heatmap_epoch_tables"]
        assert figure.canonical.load_glm_source_predictor_comparison_tables(
            kwargs["data_root"],
            kwargs["datasets"],
            light_epoch=kwargs["light_epoch"],
            dark_epoch=kwargs["dark_epoch"],
            sleep_epoch=kwargs["sleep_epoch"],
            ripple_window_s=kwargs["ripple_window_s"],
            ripple_window_offset_s=kwargs["ripple_window_offset_s"],
            ripple_selection=kwargs["ripple_selection"],
            ridge_strength=kwargs["ridge_strength"],
            epoch_types=figure.canonical.PANEL_E_GLM_EPOCH_ORDER,
        ) is payload["source_comparison_payload"]
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("supplementary figure 7", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        figure.canonical,
        "make_supplementary_figure_7",
        make_figure,
    )
    output_path = figure.get_output_path(run_dir=run_dir)

    returned = figure.render_supplementary_figure_7(
        payload,
        output_path=output_path,
        dpi=144,
    )

    assert returned == output_path.resolve()
    assert output_path.read_text(encoding="utf-8") == "supplementary figure 7"
    provenance_path = figure.get_figure_provenance_path(output_path)
    assert provenance_path.is_file()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["artifact_kind"] == figure.FIGURE_ARTIFACT_KIND
    assert calls[0]["light_epoch"] == figure.LIGHT_EPOCH
    assert calls[0]["dark_epoch"] is None
    assert calls[0]["sleep_epoch"] is None
    assert calls[0]["dpi"] == 144
    assert not list(output_path.parent.glob(".*.tmp.svg"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        figure.render_supplementary_figure_7(
            payload,
            output_path=output_path,
        )
    with pytest.raises(ValueError, match="inside its campaign"):
        figure.render_supplementary_figure_7(
            payload,
            output_path=tmp_path / "outside.svg",
        )

    publication_dir = tmp_path / "published"
    publication_dir.mkdir()
    published = publication_dir / "supplementary_figure_7.svg"
    assert figure.promote_supplementary_figure_7(
        payload,
        source_path=output_path,
        destination_path=published,
    ) == published
    assert published.read_bytes() == output_path.read_bytes()
    assert figure.get_figure_provenance_path(published).is_file()
