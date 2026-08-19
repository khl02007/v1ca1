from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.supplementary_figure_5 as figure


def test_defaults_and_output_path() -> None:
    args = figure.parse_arguments([])

    assert figure.DEFAULT_OUTPUT_NAME == "supplementary_figure_5"
    assert args.output_dir == Path("paper_figures/output")
    assert args.output_name == figure.DEFAULT_OUTPUT_NAME
    assert args.output_format == "pdf"
    assert args.ripple_selection == "single"
    assert args.region is None
    assert figure.PANEL_A_HISTOGRAM_BOTTOM == pytest.approx(0.20)
    assert figure.PANEL_A_HISTOGRAM_HEIGHT == pytest.approx(0.70)
    assert figure.build_output_path(
        Path("paper_figures/output"),
        "supplementary_figure_5",
        "svg",
    ) == Path("paper_figures/output/supplementary_figure_5.svg")
    with pytest.raises(ValueError, match="Unknown output format"):
        figure.build_output_path(Path("output"), "figure", "jpg")


def test_make_figure_reuses_the_three_requested_panel_plotters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    heatmap_tables = [{"epoch_type": "light", "table": "heatmap"}]
    filtered_tables = [{"epoch_type": "light", "table": "filtered"}]
    source_payload = {
        "comparison_table": object(),
        "missing_artifacts": [],
    }
    calls: dict[str, object] = {}

    def fake_load_heatmaps(*args: object, **kwargs: object) -> object:
        calls["heatmap_load"] = (args, kwargs)
        return heatmap_tables

    def fake_filter(payload: object, order: object) -> object:
        calls["filter"] = (payload, order)
        return filtered_tables

    def fake_load_source(*args: object, **kwargs: object) -> object:
        calls["source_load"] = (args, kwargs)
        return source_payload

    def fake_plot_heatmap(
        axis: object,
        tables: object,
        **kwargs: object,
    ) -> None:
        calls["heatmap_plot"] = (axis, tables, kwargs)

    def fake_plot_modulation(
        axis: object,
        tables: object,
        **kwargs: object,
    ) -> list[object]:
        calls["modulation_plot"] = (axis, tables, kwargs)
        return []

    def fake_plot_source(
        axis: object,
        payload: object,
        **kwargs: object,
    ) -> dict[str, object]:
        calls["source_plot"] = (axis, payload, kwargs)
        return {
            "p_value": 0.01,
            "n_vector_greater": 9,
            "n_tested": 10,
            "n_ties": 1,
        }

    def fake_save(
        mpl_figure: object,
        output_path: Path,
        **kwargs: object,
    ) -> Path:
        mpl_figure.canvas.draw()
        calls["output_path"] = output_path
        calls["save_kwargs"] = kwargs
        calls["figsize"] = tuple(mpl_figure.get_size_inches())
        calls["parent_bounds"] = [
            axis.get_position().bounds
            for axis in mpl_figure.axes
            if not axis.get_label().startswith("colorbar")
        ]
        calls["figure_text"] = {
            text.get_text() for text in mpl_figure.texts
        }
        return output_path

    monkeypatch.setattr(
        figure,
        "load_pooled_ripple_heatmap_epoch_tables",
        fake_load_heatmaps,
    )
    monkeypatch.setattr(figure, "filter_epoch_payloads", fake_filter)
    monkeypatch.setattr(
        figure,
        "load_glm_source_predictor_comparison_tables",
        fake_load_source,
    )
    monkeypatch.setattr(
        figure,
        "plot_epoch_ripple_heatmap_panel",
        fake_plot_heatmap,
    )
    monkeypatch.setattr(
        figure,
        "plot_epoch_modulation_histogram_panel",
        fake_plot_modulation,
    )
    monkeypatch.setattr(
        figure,
        "plot_glm_source_predictor_comparison_panel",
        fake_plot_source,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save)

    output_path = tmp_path / "supplementary_figure_5.svg"
    saved_path = figure.make_supplementary_figure_5(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1", "ca1"),
        light_epoch=None,
        dark_epoch=None,
        sleep_epoch=None,
        ripple_threshold_zscore=None,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection="single",
        ridge_strength=0.1,
        dpi=200,
    )

    assert saved_path == output_path
    assert calls["filter"] == (heatmap_tables, figure.PANEL_A_EPOCH_ORDER)
    heatmap_axis, heatmap_input, heatmap_kwargs = calls["heatmap_plot"]
    modulation_axis, modulation_input, modulation_kwargs = calls[
        "modulation_plot"
    ]
    source_axis, source_input, source_kwargs = calls["source_plot"]
    assert heatmap_input is filtered_tables
    assert modulation_input is filtered_tables
    assert source_input is source_payload
    assert heatmap_kwargs == {
        "regions": ("v1", "ca1"),
        "expand_heatmaps_vertically": True,
        "show_modulation_histogram": False,
        "heatmap_vertical_bounds": figure.PANEL_A_HEATMAP_VERTICAL_BOUNDS,
    }
    assert modulation_kwargs == {
        "regions": ("v1", "ca1"),
        "bottom": figure.PANEL_A_HISTOGRAM_BOTTOM,
        "height": figure.PANEL_A_HISTOGRAM_HEIGHT,
    }
    assert source_kwargs == {
        "include_per_animal": False,
        "include_pooled": True,
        "compact_labels": True,
        "show_color_note": False,
        "annotate_pooled_sign_test": True,
    }
    assert len({heatmap_axis, modulation_axis, source_axis}) == 3
    _source_args, source_load_kwargs = calls["source_load"]
    assert source_load_kwargs["epoch_types"] == figure.PANEL_E_GLM_EPOCH_ORDER
    assert source_load_kwargs["ripple_selection"] == "single"
    assert calls["output_path"] == output_path
    assert calls["save_kwargs"]["dpi"] == 200
    output_bbox = calls["save_kwargs"]["bbox_inches"]
    assert output_bbox.bounds == pytest.approx(
        (
            0.0,
            figure.OUTPUT_BOTTOM_CROP_MM / 25.4,
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            (
                figure.DEFAULT_FIGURE_HEIGHT_MM
                - figure.OUTPUT_BOTTOM_CROP_MM
            )
            / 25.4,
        )
    )
    assert calls["figsize"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            figure.DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    bounds = calls["parent_bounds"]
    assert len(bounds) == 3
    assert bounds[0][0] + bounds[0][2] < bounds[1][0]
    assert bounds[1][0] + bounds[1][2] < bounds[2][0]
    assert bounds[1][2] / bounds[0][2] == pytest.approx(1.35)
    assert bounds[2][2] == pytest.approx(bounds[0][2])
    assert {"A", "B", "C", *figure.PANEL_TITLES} <= calls["figure_text"]
