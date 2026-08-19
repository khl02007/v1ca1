"""Focused tests for the Figure 3 B/C/E Figure 4 composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import v1ca1.paper_figures.figure_4 as figure_4_module
from v1ca1.paper_figures.figure_4 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_OUTPUT_NAME,
    PANEL_B_COLUMN_LABELS,
    PANEL_E_DPPI_AXIS_BOUNDS,
    PANEL_E_SINGLE_EPOCH_AXIS_VERTICAL_BOUNDS,
    PANEL_E_SINGLE_EPOCH_COLUMN_BOUNDS,
    PANEL_F_TITLE,
    PANEL_F_XLABEL,
    PANEL_HEADER_LABELS,
    PANEL_LABELS,
    PANEL_TITLES,
    PANEL_WIDTH_RATIOS,
    build_output_path,
    load_figure_4_panel_data,
    make_figure_4,
    parse_arguments,
)


def test_defaults_keep_figure_3_width_and_use_figure_4_output() -> None:
    args = parse_arguments([])

    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        figure_4_module._figure_3.DEFAULT_FIGURE_WIDTH_MM
    )
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(165.0)
    assert DEFAULT_FIGURE_HEIGHT_MM > 0.0
    assert PANEL_WIDTH_RATIOS == (1.0, 2.0, 1.0)
    assert PANEL_LABELS == ("A", "B", "C", "D", "E", "F")
    assert PANEL_HEADER_LABELS == ("A", "", "E")
    assert PANEL_B_COLUMN_LABELS == ("B", "C", "D")
    assert PANEL_TITLES[2] == "Relationship to dark activity"
    assert PANEL_F_TITLE == "Relationship to path-invariance"
    assert PANEL_F_XLABEL == "Dark path-invariance index"
    assert args.output_dir == DEFAULT_OUTPUT_DIR == Path("paper_figures/output")
    assert args.output_name == DEFAULT_OUTPUT_NAME == "figure_4"
    assert args.output_format == DEFAULT_OUTPUT_FORMAT == "pdf"
    assert build_output_path(args.output_dir, args.output_name, "svg") == Path(
        "paper_figures/output/figure_4.svg"
    )
    assert parse_arguments(["--output-name", "custom"]).output_name == "custom"

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(args.output_dir, args.output_name, "jpg")


def test_panel_data_loader_reads_only_retained_figure_3_panels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = figure_4_module._figure_3
    calls: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def record_loader(name: str, result: Any):
        def load(*args: Any, **kwargs: Any) -> Any:
            calls[name] = (args, kwargs)
            return result

        return load

    def fail_removed_loader(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("A removed Figure 3 Panel A/D loader was called")

    glm_epoch_tables = [{"epoch_type": "light", "value": "glm"}]
    schematic_trace = {"trace": "schematic"}
    prediction_examples = [{"prediction": "example"}]
    behavior_payload = {"missing_artifacts": [], "value": "behavior"}
    xcorr_payload = {"value": "xcorr"}
    monkeypatch.setattr(
        legacy,
        "load_glm_epoch_summary_tables",
        record_loader("glm", glm_epoch_tables),
    )
    monkeypatch.setattr(
        legacy,
        "load_or_build_panel_b_schematic_example",
        record_loader("schematic", schematic_trace),
    )
    monkeypatch.setattr(
        legacy,
        "load_panel_b_prediction_examples",
        record_loader("predictions", prediction_examples),
    )
    monkeypatch.setattr(
        legacy,
        "load_glm_dark_activity_devexp_tables",
        record_loader("behavior", behavior_payload),
    )
    monkeypatch.setattr(
        legacy,
        "load_top_ca1_xcorr_panel_data",
        record_loader("xcorr", xcorr_payload),
    )
    monkeypatch.setattr(
        legacy,
        "load_pooled_ripple_heatmap_epoch_tables",
        fail_removed_loader,
    )
    monkeypatch.setattr(
        legacy,
        "load_glm_source_predictor_comparison_tables",
        fail_removed_loader,
    )

    payload = load_figure_4_panel_data(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        example_dataset=("L14", "20240611", "08_r4"),
        light_epoch=None,
        dark_epoch=None,
        sleep_epoch=None,
        ripple_threshold_zscore=None,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection="single",
        ridge_strength=0.1,
        dark_movement_fr_cache_dir=Path("cache"),
        refresh_dark_movement_fr_cache=False,
        refresh_panel_b_schematic_cache=False,
    )

    assert set(calls) == {"glm", "schematic", "predictions", "behavior", "xcorr"}
    assert payload == {
        "panel_b_xcorr_payload": xcorr_payload,
        "panel_c_epoch_tables": glm_epoch_tables,
        "panel_c_ripple_trace": schematic_trace,
        "panel_c_prediction_examples": prediction_examples,
        "panel_e_behavior_payload": behavior_payload,
    }
    assert calls["glm"][0][1] == (("L14", "20240611", "08_r4"),)
    assert calls["glm"][1]["epoch_types"] == legacy.PANEL_C_EPOCH_ORDER
    assert calls["behavior"][0][1] == (("L14", "20240611", "08_r4"),)


def test_make_figure_4_places_b_c_e_in_one_row_and_stacks_panel_e(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    legacy = figure_4_module._figure_3
    calls: dict[str, Any] = {}
    panel_data = {
        "panel_b_xcorr_payload": "xcorr",
        "panel_c_epoch_tables": ["glm"],
        "panel_c_ripple_trace": "trace",
        "panel_c_prediction_examples": ["prediction"],
        "panel_e_behavior_payload": {"missing_artifacts": []},
    }

    def fake_load_panel_data(**kwargs: Any) -> dict[str, Any]:
        calls["loader_kwargs"] = kwargs
        return panel_data

    def fake_prepare_xcorr(payload: Any) -> str:
        assert payload == "xcorr"
        return "prepared-xcorr"

    def fake_plot_xcorr(ax: Any, payload: Any, **kwargs: Any) -> None:
        calls["panel_b_axis"] = ax
        calls["panel_b_payload"] = payload
        calls["panel_b_kwargs"] = kwargs

    def fake_plot_glm(ax: Any, payload: Any, **kwargs: Any) -> None:
        calls["panel_c_axis"] = ax
        calls["panel_c_payload"] = payload
        calls["panel_c_kwargs"] = kwargs
        for index, label in enumerate(kwargs["column_panel_labels"]):
            ax.text(index / 3.0, 0.96, label, fontweight="bold")
        panel_d_bottom_axis = ax.inset_axes([0.72, 0.045, 0.25, 0.245])
        panel_d_bottom_axis.set_xlabel("Deviance explained")
        calls["panel_d_bottom_axis"] = panel_d_bottom_axis

    def fake_plot_properties(ax: Any, payload: Any, **kwargs: Any) -> None:
        calls["panel_e_axis"] = ax
        calls["panel_e_payload"] = payload
        calls["panel_e_kwargs"] = kwargs
        column_bounds = kwargs["single_epoch_column_bounds"]
        bottom, height = kwargs["single_epoch_axis_vertical_bounds"]
        fraction_ax = ax.inset_axes(
            [column_bounds[0][0], bottom, column_bounds[0][1], height]
        )
        devexp_ax = ax.inset_axes(
            [column_bounds[1][0], bottom, column_bounds[1][1], height]
        )
        dppi_ax = ax.inset_axes(kwargs["dppi_axis_bounds"])
        fraction_ax.set_xlabel(r"$p$<0.05 frac.")
        devexp_ax.set_xlabel("Dev. explained")
        dppi_ax.set_xlabel("Dark DPPI")
        panel_f_significance = dppi_ax.text(
            0.5,
            1.0,
            "**",
            ha="center",
            va="top",
            transform=dppi_ax.transAxes,
        )
        calls["panel_e_children"] = (fraction_ax, devexp_ax, dppi_ax)
        calls["panel_f_significance"] = panel_f_significance

    def fail_removed_plotter(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("A removed Figure 3 Panel A/D plotter was called")

    def fake_save_figure(
        figure: Any,
        output_path: Path,
        *,
        dpi: int,
        **kwargs: Any,
    ) -> Path:
        figure.canvas.draw()
        calls["figure"] = figure
        calls["figsize"] = tuple(figure.get_size_inches())
        calls["figure_texts"] = [text.get_text() for text in figure.texts]
        calls["figure_text_artists"] = tuple(figure.texts)
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        return output_path

    monkeypatch.setattr(
        figure_4_module,
        "load_figure_4_panel_data",
        fake_load_panel_data,
    )
    monkeypatch.setattr(legacy, "prepare_xcorr_payload_for_display", fake_prepare_xcorr)
    monkeypatch.setattr(legacy, "plot_top_ca1_xcorr_panel", fake_plot_xcorr)
    monkeypatch.setattr(legacy, "plot_glm_analysis_panel", fake_plot_glm)
    monkeypatch.setattr(
        legacy,
        "plot_glm_dark_epoch_properties_panel",
        fake_plot_properties,
    )
    monkeypatch.setattr(legacy, "plot_epoch_ripple_heatmap_panel", fail_removed_plotter)
    monkeypatch.setattr(
        legacy,
        "plot_glm_source_predictor_comparison_panel",
        fail_removed_plotter,
    )
    monkeypatch.setattr(legacy, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_4.svg"
    saved_path = make_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        example_dataset=("L14", "20240611", "08_r4"),
        light_epoch=None,
        dark_epoch=None,
        sleep_epoch=None,
        ripple_threshold_zscore=None,
        ripple_window_s=0.2,
        ripple_window_offset_s=0.0,
        ripple_selection="single",
        ridge_strength=0.1,
        dark_movement_fr_cache_dir=Path("cache"),
        refresh_dark_movement_fr_cache=False,
        refresh_panel_b_schematic_cache=False,
        dpi=144,
    )

    assert saved_path == output_path
    assert calls["figsize"] == pytest.approx(
        (DEFAULT_FIGURE_WIDTH_MM / 25.4, DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 144
    assert calls["save_kwargs"] == {}
    assert calls["panel_b_payload"] == "prepared-xcorr"
    assert calls["panel_b_kwargs"] == {
        "lag_label_y": -0.055,
        "compact_unit_titles": True,
    }
    assert calls["panel_c_payload"] == ["glm"]
    assert calls["panel_c_kwargs"] == {
        "ripple_trace": "trace",
        "prediction_examples": ["prediction"],
        "column_panel_labels": PANEL_B_COLUMN_LABELS,
    }
    assert calls["panel_e_payload"] is panel_data["panel_e_behavior_payload"]
    assert calls["panel_e_kwargs"] == {
        "single_epoch_column_bounds": PANEL_E_SINGLE_EPOCH_COLUMN_BOUNDS,
        "single_epoch_axis_vertical_bounds": (
            PANEL_E_SINGLE_EPOCH_AXIS_VERTICAL_BOUNDS
        ),
        "dppi_axis_bounds": PANEL_E_DPPI_AXIS_BOUNDS,
    }

    major_axes = (
        calls["panel_b_axis"],
        calls["panel_c_axis"],
        calls["panel_e_axis"],
    )
    major_positions = [axis.get_position() for axis in major_axes]
    assert [position.x0 for position in major_positions] == sorted(
        position.x0 for position in major_positions
    )
    assert [position.y0 for position in major_positions] == pytest.approx(
        [major_positions[0].y0] * 3
    )
    assert [position.y1 for position in major_positions] == pytest.approx(
        [major_positions[0].y1] * 3
    )

    fraction_ax, devexp_ax, dppi_ax = calls["panel_e_children"]
    fraction_position = fraction_ax.get_position()
    devexp_position = devexp_ax.get_position()
    dppi_position = dppi_ax.get_position()
    assert fraction_position.x0 < devexp_position.x0
    assert fraction_position.y0 == pytest.approx(devexp_position.y0)
    assert fraction_position.y1 == pytest.approx(devexp_position.y1)
    assert dppi_position.y1 < fraction_position.y0
    assert dppi_position.width > fraction_position.width
    assert dppi_position.width > devexp_position.width
    assert [axis.get_xlabel() for axis in (fraction_ax, devexp_ax, dppi_ax)] == [
        r"$p$<0.05 frac.",
        "Dev. explained",
        PANEL_F_XLABEL,
    ]
    assert dppi_ax.get_title() == ""
    column_labels = [
        text.get_text()
        for text in calls["panel_c_axis"].texts
        if text.get_text() in PANEL_B_COLUMN_LABELS
    ]
    assert column_labels == list(PANEL_B_COLUMN_LABELS)
    figure_text_artists = calls["figure_text_artists"]
    header_artists = {
        text.get_text(): text
        for text in figure_text_artists
        if text.get_text() in {"E", "F", PANEL_TITLES[2], PANEL_F_TITLE}
    }
    assert set(header_artists) == {"E", "F", PANEL_TITLES[2], PANEL_F_TITLE}
    assert header_artists["E"].get_position()[0] == pytest.approx(
        header_artists["F"].get_position()[0]
    )
    assert header_artists["E"].get_position()[1] == pytest.approx(
        header_artists[PANEL_TITLES[2]].get_position()[1]
    )
    assert header_artists["F"].get_position()[1] == pytest.approx(
        header_artists[PANEL_F_TITLE].get_position()[1]
    )
    assert header_artists["E"].get_fontweight() == "bold"
    assert header_artists["F"].get_fontweight() == "bold"
    renderer = calls["figure"].canvas.get_renderer()
    panel_d_xaxis_y = calls["panel_d_bottom_axis"].transAxes.transform(
        (0.0, 0.0)
    )[1]
    panel_f_xaxis_y = dppi_ax.transAxes.transform((0.0, 0.0))[1]
    assert panel_f_xaxis_y == pytest.approx(panel_d_xaxis_y, abs=0.25)
    panel_f_title_bounds = header_artists[PANEL_F_TITLE].get_window_extent(renderer)
    panel_f_significance_bounds = calls[
        "panel_f_significance"
    ].get_window_extent(renderer)
    assert panel_f_title_bounds.y0 > panel_f_significance_bounds.y1
    expected_headers = [
        item
        for label, title in zip(PANEL_HEADER_LABELS, PANEL_TITLES, strict=True)
        for item in (label, title)
    ]
    assert calls["figure_texts"][-8:] == [
        *expected_headers,
        PANEL_LABELS[5],
        PANEL_F_TITLE,
    ]
