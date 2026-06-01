from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_4 as figure_4_module
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_2_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_2_WIDTH_MM,
)
from v1ca1.paper_figures.figure_4 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_REGIONS,
    FIGURE_4_CONSTRAINED_LAYOUT_PADS,
    PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y,
    PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y,
    PANEL_B_COMPONENT_LABEL_FONTSIZE,
    PANEL_B_EXAMPLE_AXIS_BOUNDS,
    PANEL_B_EXAMPLE_COLUMN_GAP,
    PANEL_B_EXAMPLE_COLUMN_WIDTH,
    PANEL_B_EXAMPLE_FIELD_HEIGHT,
    PANEL_B_EXAMPLE_FIELD_GAP,
    PANEL_B_EXAMPLE_FIELD_WIDTH,
    PANEL_B_EXAMPLE_FIELD_Y,
    PANEL_B_EXAMPLE_ICON_BOUNDS,
    PANEL_B_EXAMPLE_LAYOUT,
    PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET,
    PANEL_B_EXAMPLE_ROW_GAP,
    PANEL_B_EXAMPLE_ROW_HEIGHT,
    PANEL_B_EXAMPLE_XLABEL_Y,
    PANEL_B_FIELD_LABEL_Y,
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_MODEL_LABEL_FONTSIZE,
    PANEL_B_MODEL_LABEL_X,
    PANEL_B_SCHEMATIC_HEIGHT_FRACTION,
    PANEL_B_SCHEMATIC_TRACK_SIZE,
    PANEL_B_SEGMENT_MODULATION_LABEL,
    PANEL_B_SEGMENT_MODULATION_LABEL_Y,
    PANEL_BC_LABEL_Y,
    PANEL_BC_TITLE_PAD,
    PANEL_C_DELTA_AXIS_BOUNDS,
    PANEL_C_DELTA_GRID_BOUNDS,
    PANEL_C_DELTA_XLABEL_Y,
    PANEL_C_EXAMPLE_AXIS_BOUNDS,
    PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS,
    PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS,
    PANEL_C_EXAMPLE_ICON_BOUNDS,
    PANEL_C_HORIZONTAL_SHIFT,
    PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y,
    PANEL_C_INDEPENDENT_TRACK_CENTER_Y,
    PANEL_C_PREDICTION_LABEL_FONTSIZE,
    PANEL_C_SCHEMATIC_AXIS_BOUNDS,
    PANEL_C_SCHEMATIC_TRACK_SIZE,
    PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y,
    PANEL_C_SHARED_DARK_TRACK_CENTER_Y,
    PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y,
    PANEL_C_SHARED_PREDICTION_LABEL_Y,
    PANEL_A_TO_GH_HEIGHT_RATIOS,
    build_output_path,
    make_figure_4,
    parse_arguments,
    parse_dataset_id,
)


def test_parse_dataset_id_requires_animal_and_date() -> None:
    assert parse_dataset_id("L14:20240611") == ("L14", "20240611", "08_r4")
    assert parse_dataset_id("L15:20241121:10_r5") == (
        "L15",
        "20241121",
        "10_r5",
    )

    with pytest.raises(argparse.ArgumentTypeError, match="animal:date"):
        parse_dataset_id("L14")


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "figure_4", "svg") == Path(
        "paper_figures/figure_4.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "figure_4", "jpg")


def test_default_cli_matches_figure_2_canvas() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "figure_4"
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.region is None
    assert args.panel_example_cache_dir is None
    assert args.refresh_panel_example_cache is False
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(FIGURE_2_WIDTH_MM)
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        FIGURE_2_HEIGHT_MM
        * 1.3
        * PANEL_A_TO_GH_HEIGHT_RATIOS[1]
        / sum(PANEL_A_TO_GH_HEIGHT_RATIOS)
    )
    assert DEFAULT_REGIONS == ("v1",)
    assert FIGURE_4_CONSTRAINED_LAYOUT_PADS == pytest.approx(
        {"h_pad": 0.01, "w_pad": 0.01, "hspace": 0.01, "wspace": 0.02}
    )
    assert PANEL_A_TO_GH_HEIGHT_RATIOS == pytest.approx((0.637, 1.3))
    assert PANEL_BC_LABEL_Y == pytest.approx(1.03)
    assert PANEL_BC_TITLE_PAD == pytest.approx(0.5)
    assert PANEL_B_SCHEMATIC_HEIGHT_FRACTION == pytest.approx(0.72)
    assert PANEL_B_SCHEMATIC_TRACK_SIZE == pytest.approx((0.2512, 0.316))
    assert PANEL_B_EXAMPLE_AXIS_BOUNDS == pytest.approx((0.0, 0.01, 1.0, 0.44))
    assert PANEL_B_EXAMPLE_FIELD_Y == pytest.approx(0.13)
    assert PANEL_B_EXAMPLE_FIELD_HEIGHT == pytest.approx(0.62)
    assert PANEL_B_EXAMPLE_ICON_BOUNDS == pytest.approx((0.04, 0.27, 0.09, 0.34))
    assert PANEL_B_EXAMPLE_XLABEL_Y == pytest.approx(0.02)
    assert PANEL_B_EXAMPLE_COLUMN_WIDTH == pytest.approx(0.50)
    assert PANEL_B_EXAMPLE_COLUMN_GAP == pytest.approx(0.0)
    assert PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET == pytest.approx(0.20)
    assert PANEL_B_EXAMPLE_FIELD_WIDTH == pytest.approx(0.28)
    assert PANEL_B_EXAMPLE_FIELD_GAP == pytest.approx(0.075)
    assert PANEL_B_EXAMPLE_LAYOUT == "rows"
    assert PANEL_B_EXAMPLE_ROW_HEIGHT == pytest.approx(0.46)
    assert PANEL_B_EXAMPLE_ROW_GAP == pytest.approx(0.05)
    assert PANEL_B_FIELD_LABEL_Y == pytest.approx(0.9619, abs=1e-4)
    assert PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)
    assert PANEL_B_MODEL_LABEL_X == pytest.approx(0.03)
    assert PANEL_B_MODEL_LABEL_FONTSIZE == pytest.approx(5.8)
    assert PANEL_B_COMPONENT_LABEL_FONTSIZE == pytest.approx(5.8)
    assert PANEL_B_SEGMENT_MODULATION_LABEL == "Segment-specific\nmodulation"
    assert PANEL_B_SEGMENT_MODULATION_LABEL_Y == pytest.approx(0.595)
    assert PANEL_C_SCHEMATIC_AXIS_BOUNDS == pytest.approx((-0.08, 0.25, 0.40, 0.72))
    assert PANEL_C_DELTA_AXIS_BOUNDS == pytest.approx((0.39, 0.35, 0.60, 0.59))
    expected_delta_grid_bounds = (
        (0.035, 0.42, 0.445, 0.50),
        (0.535, 0.42, 0.445, 0.50),
        (0.035, -0.22, 0.445, 0.50),
        (0.535, -0.22, 0.445, 0.50),
    )
    assert len(PANEL_C_DELTA_GRID_BOUNDS) == len(expected_delta_grid_bounds)
    for actual_bounds, expected_bounds in zip(
        PANEL_C_DELTA_GRID_BOUNDS,
        expected_delta_grid_bounds,
        strict=True,
    ):
        assert actual_bounds == pytest.approx(expected_bounds)
    assert PANEL_C_DELTA_XLABEL_Y == pytest.approx(-0.40)
    assert len(PANEL_C_EXAMPLE_AXIS_BOUNDS) == 2
    assert PANEL_C_EXAMPLE_AXIS_BOUNDS[0] == pytest.approx(
        (0.201, -0.18, 0.248, 0.19)
    )
    assert PANEL_C_EXAMPLE_AXIS_BOUNDS[1] == pytest.approx(
        (0.591, -0.18, 0.248, 0.19)
    )
    assert PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS == (
        (0.96, 0.94),
        (0.96, 0.06),
    )
    assert PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS == ("top", "bottom")
    assert PANEL_C_EXAMPLE_ICON_BOUNDS == pytest.approx((-0.46, 0.28, 0.26, 0.38))
    assert PANEL_C_PREDICTION_LABEL_FONTSIZE == pytest.approx(5.8)
    assert PANEL_C_INDEPENDENT_TRACK_CENTER_Y == pytest.approx(0.742)
    assert PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y == pytest.approx(0.60)
    assert PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y == pytest.approx(0.34)
    assert PANEL_C_SHARED_DARK_TRACK_CENTER_Y == pytest.approx(0.0)
    assert PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y == pytest.approx(0.17)
    assert PANEL_C_SHARED_PREDICTION_LABEL_Y == pytest.approx(-0.24)
    assert PANEL_C_SCHEMATIC_TRACK_SIZE == pytest.approx((0.628, 0.316))
    assert PANEL_C_HORIZONTAL_SHIFT == pytest.approx(-0.025)
    assert PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y == pytest.approx(0.7690, abs=1e-4)
    assert PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y == pytest.approx(0.3730, abs=1e-4)


def test_make_figure_4_uses_scaled_height_and_moved_panel_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_load_panel_glm_data(**kwargs: object) -> dict[str, object]:
        calls["glm_kwargs"] = kwargs
        return {
            "dark_light_examples": ["panel-b"],
            "swap_delta": "panel-c-delta",
            "swap_examples": ["panel-c-example"],
        }

    def fake_plot_panel_g_model_architecture(
        ax: object,
        examples: object,
        **kwargs: object,
    ) -> None:
        calls["panel_b_examples"] = examples
        calls["panel_b_kwargs"] = kwargs

    def fake_plot_panel_h_swap_delta(
        ax: object,
        swap_delta_table: object,
        swap_examples: object,
        **kwargs: object,
    ) -> None:
        calls["panel_c_delta"] = swap_delta_table
        calls["panel_c_examples"] = swap_examples
        calls["panel_c_kwargs"] = kwargs

    def fake_save_figure(figure: object, output_path: Path, dpi: int) -> Path:
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        return output_path

    monkeypatch.setattr(
        figure_4_module,
        "load_panel_glm_data",
        fake_load_panel_glm_data,
    )
    monkeypatch.setattr(
        figure_4_module,
        "plot_panel_g_model_architecture",
        fake_plot_panel_g_model_architecture,
    )
    monkeypatch.setattr(
        figure_4_module,
        "plot_panel_h_swap_delta",
        fake_plot_panel_h_swap_delta,
    )
    monkeypatch.setattr(figure_4_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_4.svg"
    saved_path = make_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        dark_epoch=None,
        position_bin_count=100,
        position_offset=5,
        speed_threshold_cm_s=4.0,
        sigma_bins=1.5,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(FIGURE_2_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    assert calls["panel_labels"] == ["A", "B"]
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["glm_kwargs"]["region"] == "v1"
    assert calls["glm_kwargs"][
        "swap_delta_min_tuning_stability_correlation"
    ] == pytest.approx(PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION)
    assert calls["panel_b_examples"] == ["panel-b"]
    assert calls["panel_b_kwargs"]["independent_track_center_y"] == pytest.approx(
        PANEL_B_ALIGNED_INDEPENDENT_TRACK_CENTER_Y
    )
    assert calls["panel_b_kwargs"]["shared_track_center_y"] == pytest.approx(
        PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
    )
    assert calls["panel_b_kwargs"]["schematic_height_fraction"] == pytest.approx(
        PANEL_B_SCHEMATIC_HEIGHT_FRACTION
    )
    assert calls["panel_b_kwargs"]["schematic_track_size"] == pytest.approx(
        PANEL_B_SCHEMATIC_TRACK_SIZE
    )
    assert calls["panel_b_kwargs"]["show_dark_track_labels"] is True
    assert calls["panel_b_kwargs"]["field_label_y"] == pytest.approx(
        PANEL_B_FIELD_LABEL_Y
    )
    assert calls["panel_b_kwargs"]["model_label_x"] == pytest.approx(
        PANEL_B_MODEL_LABEL_X
    )
    assert calls["panel_b_kwargs"]["model_label_fontsize"] == pytest.approx(
        PANEL_B_MODEL_LABEL_FONTSIZE
    )
    assert calls["panel_b_kwargs"]["component_label_fontsize"] == pytest.approx(
        PANEL_B_COMPONENT_LABEL_FONTSIZE
    )
    assert calls["panel_b_kwargs"]["segment_modulation_label_y"] == pytest.approx(
        PANEL_B_SEGMENT_MODULATION_LABEL_Y
    )
    assert (
        calls["panel_b_kwargs"]["segment_modulation_label"]
        == PANEL_B_SEGMENT_MODULATION_LABEL
    )
    assert calls["panel_b_kwargs"]["example_axis_bounds"] == pytest.approx(
        PANEL_B_EXAMPLE_AXIS_BOUNDS
    )
    assert calls["panel_b_kwargs"]["example_field_y"] == pytest.approx(
        PANEL_B_EXAMPLE_FIELD_Y
    )
    assert calls["panel_b_kwargs"]["example_field_height"] == pytest.approx(
        PANEL_B_EXAMPLE_FIELD_HEIGHT
    )
    assert calls["panel_b_kwargs"]["example_icon_bounds"] == pytest.approx(
        PANEL_B_EXAMPLE_ICON_BOUNDS
    )
    assert calls["panel_b_kwargs"]["example_xlabel_y"] == pytest.approx(
        PANEL_B_EXAMPLE_XLABEL_Y
    )
    assert calls["panel_b_kwargs"]["example_column_width"] == pytest.approx(
        PANEL_B_EXAMPLE_COLUMN_WIDTH
    )
    assert calls["panel_b_kwargs"]["example_column_gap"] == pytest.approx(
        PANEL_B_EXAMPLE_COLUMN_GAP
    )
    assert calls["panel_b_kwargs"]["example_plot_left_offset"] == pytest.approx(
        PANEL_B_EXAMPLE_PLOT_LEFT_OFFSET
    )
    assert calls["panel_b_kwargs"]["example_field_width"] == pytest.approx(
        PANEL_B_EXAMPLE_FIELD_WIDTH
    )
    assert calls["panel_b_kwargs"]["example_field_gap"] == pytest.approx(
        PANEL_B_EXAMPLE_FIELD_GAP
    )
    assert calls["panel_b_kwargs"]["example_layout"] == PANEL_B_EXAMPLE_LAYOUT
    assert calls["panel_b_kwargs"]["example_row_height"] == pytest.approx(
        PANEL_B_EXAMPLE_ROW_HEIGHT
    )
    assert calls["panel_b_kwargs"]["example_row_gap"] == pytest.approx(
        PANEL_B_EXAMPLE_ROW_GAP
    )
    assert calls["panel_c_delta"] == "panel-c-delta"
    assert calls["panel_c_examples"] == ["panel-c-example"]
    assert calls["panel_c_kwargs"]["schematic_axis_bounds"] == pytest.approx(
        PANEL_C_SCHEMATIC_AXIS_BOUNDS
    )
    assert calls["panel_c_kwargs"]["delta_axis_bounds"] == pytest.approx(
        PANEL_C_DELTA_AXIS_BOUNDS
    )
    assert len(calls["panel_c_kwargs"]["example_axis_bounds"]) == 2
    assert calls["panel_c_kwargs"]["example_axis_bounds"][0] == pytest.approx(
        PANEL_C_EXAMPLE_AXIS_BOUNDS[0]
    )
    assert calls["panel_c_kwargs"]["example_axis_bounds"][1] == pytest.approx(
        PANEL_C_EXAMPLE_AXIS_BOUNDS[1]
    )
    assert calls["panel_c_kwargs"]["schematic_track_size"] == pytest.approx(
        PANEL_C_SCHEMATIC_TRACK_SIZE
    )
    assert calls["panel_c_kwargs"]["show_dark_track_labels"] is True
    assert calls["panel_c_kwargs"]["show_model_labels"] is False
    assert calls["panel_c_kwargs"]["prediction_label_fontsize"] == pytest.approx(
        PANEL_C_PREDICTION_LABEL_FONTSIZE
    )
    assert calls["panel_c_kwargs"]["independent_track_center_y"] == pytest.approx(
        PANEL_C_INDEPENDENT_TRACK_CENTER_Y
    )
    assert calls["panel_c_kwargs"][
        "independent_prediction_label_y"
    ] == pytest.approx(PANEL_C_INDEPENDENT_PREDICTION_LABEL_Y)
    assert calls["panel_c_kwargs"][
        "segment_modulation_track_center_y"
    ] == pytest.approx(PANEL_C_SEGMENT_MODULATION_TRACK_CENTER_Y)
    assert calls["panel_c_kwargs"]["shared_dark_track_center_y"] == pytest.approx(
        PANEL_C_SHARED_DARK_TRACK_CENTER_Y
    )
    assert calls["panel_c_kwargs"]["shared_light_track_center_y"] == pytest.approx(
        PANEL_C_SHARED_LIGHT_TRACK_CENTER_Y
    )
    assert calls["panel_c_kwargs"]["shared_prediction_label_y"] == pytest.approx(
        PANEL_C_SHARED_PREDICTION_LABEL_Y
    )
    assert len(calls["panel_c_kwargs"]["delta_grid_bounds"]) == len(
        PANEL_C_DELTA_GRID_BOUNDS
    )
    for actual_bounds, expected_bounds in zip(
        calls["panel_c_kwargs"]["delta_grid_bounds"],
        PANEL_C_DELTA_GRID_BOUNDS,
        strict=True,
    ):
        assert actual_bounds == pytest.approx(expected_bounds)
    assert calls["panel_c_kwargs"]["delta_xlabel_y"] == pytest.approx(
        PANEL_C_DELTA_XLABEL_Y
    )
    assert (
        calls["panel_c_kwargs"]["example_delta_label_positions"]
        == PANEL_C_EXAMPLE_DELTA_LABEL_POSITIONS
    )
    assert (
        calls["panel_c_kwargs"]["example_delta_label_vertical_alignments"]
        == PANEL_C_EXAMPLE_DELTA_LABEL_VERTICAL_ALIGNMENTS
    )
    assert calls["panel_c_kwargs"]["example_icon_bounds"] == pytest.approx(
        PANEL_C_EXAMPLE_ICON_BOUNDS
    )
