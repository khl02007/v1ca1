from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_2 as figure_2_module
import v1ca1.paper_figures.figure_2_2 as figure_2_2_module
import v1ca1.paper_figures.figure_2_common as figure_2_common_module
from v1ca1.helper.plot_wtrack_schematic import get_w_track_geometry
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM as FIGURE_3_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM as FIGURE_3_WIDTH_MM,
)
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    DEFAULT_REGIONS,
    PANEL_A_EXAMPLE_ROW_HEIGHT_MM,
    PANEL_B_BOX_COLORS,
    PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
    PANEL_B_EXAMPLE_MODEL_COLORS,
    PANEL_B_EXAMPLE_MODEL_LABELS,
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
    PANEL_B_MIN_TUNING_STABILITY_CORRELATION,
    PANEL_B_TUNING_CORRELATION_TRAJECTORIES,
    PANEL_BC_ROW_HEIGHT_MM,
    PANEL_C_DARK_LIGHT_EXAMPLES,
    PANEL_C_SWAP_EXAMPLES,
    PANEL_C_SWAP_MODEL_LABELS,
    PANEL_C_SWAP_MODEL_NAME,
    PANEL_A_TO_GH_HEIGHT_RATIOS,
    build_output_path,
    load_panel_b_dark_dpp_index_table,
    load_panel_b_light_dpp_index_table,
    load_panel_b_light_tuning_stability_table,
    load_panel_b_tuning_correlation_table,
    make_figure_2,
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
    assert build_output_path(Path("paper_figures"), "figure_2", "svg") == Path(
        "paper_figures/figure_2.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "figure_2", "jpg")


def test_panel_a_and_bc_rows_have_requested_layout_ratio() -> None:
    assert figure_2_module.PANEL_BC_QUANT_ROW_HEIGHT_MM == pytest.approx(
        figure_2_module.PANEL_A_SINGLE_ROW_HEIGHT_MM
    )
    assert figure_2_module.PANEL_D_ROW_HEIGHT_MM == pytest.approx(
        figure_2_module.PANEL_A_SINGLE_ROW_HEIGHT_MM
    )
    assert figure_2_module.PANEL_E_ROW_HEIGHT_MM == pytest.approx(
        figure_2_module.PANEL_A_SINGLE_ROW_HEIGHT_MM
    )
    assert figure_2_module.PANEL_BC_ROW_WIDTH_RATIOS == pytest.approx((2.0, 1.0))
    assert figure_2_module.PANEL_BC_ROW_WSPACE < 0.05
    assert figure_2_module.PANEL_A_SINGLE_ROW_COLUMN_GAP >= 0.03
    assert figure_2_module.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT < 0.08
    assert (
        figure_2_module.PANEL_A_SINGLE_ROW_SCHEMATIC_AXIS_LEFT + 0.070
    ) < figure_2_module.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT
    dark_light_axis_gap = (
        figure_2_module.PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT
        - figure_2_module.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT
        - figure_2_module.PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH
    )
    assert dark_light_axis_gap >= 0.15
    assert figure_2_module.PANEL_A_HORIZONTAL_AXIS_BOUNDS[1] > 0.94
    assert figure_2_module.PANEL_B_HORIZONTAL_WIDTH_SCALE == pytest.approx(1.0)
    panel_b_right_edge = (
        figure_2_module.PANEL_B_SCATTER_AXIS_BOUNDS[0]
        + figure_2_module.PANEL_B_SCATTER_AXIS_BOUNDS[2]
    )
    assert panel_b_right_edge >= 0.99
    assert figure_2_module.PANEL_E_SCHEMATIC_AXIS_BOUNDS == pytest.approx(
        figure_2_module.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS
    )
    schematic_left, schematic_bottom, schematic_width, schematic_height = (
        figure_2_module.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_BOUNDS
    )
    assert schematic_left + schematic_width / 2.0 == pytest.approx(0.25)
    assert schematic_bottom + schematic_height / 2.0 == pytest.approx(0.50)
    assert figure_2_module.PANEL_E_DELTA_AXIS_BOUNDS[0] == pytest.approx(0.50)
    assert figure_2_module.PANEL_E_DELTA_AXIS_BOUNDS[2] == pytest.approx(0.50)
    assert figure_2_module.PANEL_E_DELTA_AXIS_BOUNDS[3] >= 0.85
    assert figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_BOUNDS == pytest.approx(
        (0.50, 0.06, 0.50, 0.90)
    )
    assert figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_WIDTH == pytest.approx(
        0.32
    )
    assert figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_HEIGHT == pytest.approx(
        0.64
    )
    assert figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_COLUMN_GAP == pytest.approx(
        0.02
    )
    assert figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_ROW_GAP == pytest.approx(0.12)
    assert figure_2_module.PANEL_E_EXAMPLE_SLOT_BOUNDS[0][1] >= 0.69
    assert figure_2_module.PANEL_E_EXAMPLE_SLOT_BOUNDS[2][1] <= 0.09
    assert figure_2_module.PANEL_E_EXAMPLE_SLOT_BOUNDS[0][2] >= 0.40
    assert figure_2_module.PANEL_E_MEAN_DELTA_AXIS_BOUNDS[0] >= 0.60
    assert figure_2_module.PANEL_E_MEAN_DELTA_AXIS_BOUNDS[2] >= 0.38
    assert figure_2_module.PANEL_E_MEAN_DELTA_AXIS_BOUNDS[3] >= 0.68
    panel_d_block_shift = (
        figure_2_module.PANEL_D_LEFT_SCHEMATIC_BLOCK_VERTICAL_SHIFT
    )
    panel_d_block_top = figure_2_module.PANEL_B_FIELD_LABEL_Y + panel_d_block_shift
    panel_d_block_bottom = (
        figure_2_module.PANEL_B_ALIGNED_SHARED_TRACK_CENTER_Y
        + panel_d_block_shift
        - figure_2_module.PANEL_C_SIDE_BY_SIDE_SCHEMATIC_TRACK_SIZE[1] / 2.0
    )
    assert (panel_d_block_top + panel_d_block_bottom) / 2.0 == pytest.approx(
        0.50,
        abs=0.01,
    )


def test_panel_b_selects_light_dark_correlation_from_dark_peak_trajectory() -> None:
    selected = figure_2_module._select_dark_peak_tuning_correlation(
        [
            ("center_to_left", 0.95, 1.0),
            ("right_to_center", 0.20, 3.5),
            ("center_to_right", -0.40, 5.0),
            ("left_to_center", 0.75, float("nan")),
        ]
    )

    assert selected is not None
    trajectory, correlation, dark_peak = selected
    assert trajectory == "center_to_right"
    assert correlation == pytest.approx(-0.40)
    assert dark_peak == pytest.approx(5.0)


def test_panel_b_dpp_loaders_preserve_dark_selected_same_turn_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")

    table_path = tmp_path / "light_tuning_similarity.parquet"
    table_path.touch()
    similarity_table = pandas.DataFrame(
        [
            {
                "unit": 1,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_label": "left_turn",
                "similarity": 0.42,
            },
            {
                "unit": 1,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_label": "right_turn",
                "similarity": 0.74,
            },
            {
                "unit": 1,
                "region": "v1",
                "epoch": "08_r4",
                "comparison_label": "left_turn",
                "similarity": 0.80,
            },
            {
                "unit": 1,
                "region": "v1",
                "epoch": "08_r4",
                "comparison_label": "right_turn",
                "similarity": 0.20,
            },
            {
                "unit": 2,
                "region": "v1",
                "epoch": "02_r1",
                "comparison_label": "pooled_same_turn",
                "similarity": 0.95,
            },
            {
                "unit": 3,
                "region": "ca1",
                "epoch": "02_r1",
                "comparison_label": "left_turn",
                "similarity": 0.91,
            },
            {
                "unit": 4,
                "region": "v1",
                "epoch": "08_r4",
                "comparison_label": "left_turn",
                "similarity": 0.88,
            },
        ]
    )

    monkeypatch.setattr(
        figure_2_common_module,
        "get_tuning_similarity_path",
        lambda data_root, animal_name, date, region, epoch: table_path,
    )
    monkeypatch.setattr(
        pandas,
        "read_parquet",
        lambda path: similarity_table,
    )

    result = load_panel_b_light_dpp_index_table(
        Path("/analysis"),
        animal_name="L14",
        date="20240611",
        light_epoch="02_r1",
        region="v1",
    )

    assert result["unit"].tolist() == [1, 1]
    assert result["comparison_label"].tolist() == ["left_turn", "right_turn"]
    assert result["light_dpp_index"].tolist() == pytest.approx([0.42, 0.74])

    dark_result = load_panel_b_dark_dpp_index_table(
        Path("/analysis"),
        animal_name="L14",
        date="20240611",
        dark_epoch="08_r4",
        region="v1",
    )

    assert dark_result["unit"].tolist() == [1, 4]
    assert dark_result["dpp_comparison_label"].tolist() == [
        "left_turn",
        "left_turn",
    ]
    assert dark_result["dark_dpp_index"].tolist() == pytest.approx([0.80, 0.88])


def test_panel_b_light_tuning_stability_requires_one_stable_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")

    table_path = tmp_path / "odd_even_task_progression_stability.parquet"
    table_path.touch()
    stability_table = pandas.DataFrame(
        [
            {
                "unit": 1,
                "region": "v1",
                "epoch": "02_r1",
                "trajectory_type": "center_to_left",
                "stability_correlation": 0.49,
            },
            {
                "unit": 1,
                "region": "v1",
                "epoch": "02_r1",
                "trajectory_type": "right_to_center",
                "stability_correlation": 0.62,
            },
            {
                "unit": 2,
                "region": "v1",
                "epoch": "02_r1",
                "trajectory_type": "center_to_right",
                "stability_correlation": 0.48,
            },
            {
                "unit": 3,
                "region": "ca1",
                "epoch": "02_r1",
                "trajectory_type": "center_to_left",
                "stability_correlation": 0.91,
            },
            {
                "unit": 4,
                "region": "v1",
                "epoch": "08_r4",
                "trajectory_type": "center_to_left",
                "stability_correlation": 0.88,
            },
        ]
    )

    monkeypatch.setattr(
        figure_2_common_module,
        "get_stability_table_path",
        lambda data_root, animal_name, date: table_path,
    )
    monkeypatch.setattr(
        pandas,
        "read_parquet",
        lambda path: stability_table,
    )

    result = load_panel_b_light_tuning_stability_table(
        Path("/analysis"),
        animal_name="L14",
        date="20240611",
        light_epoch="02_r1",
        region="v1",
    )

    assert result["unit"].tolist() == [1]
    assert result["max_light_tuning_stability_correlation"].tolist() == pytest.approx(
        [0.62]
    )


def test_panel_b_tuning_correlation_table_uses_figure_3_similarity_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = pytest.importorskip("pandas")

    calls: dict[str, object] = {}
    similarity_table = pandas.DataFrame(
        {
            "animal_name": ["L14"],
            "date": ["20240611"],
            "unit": [1],
            "comparison_label": ["left_turn"],
            "epoch_type": ["light"],
            "similarity": [0.25],
        }
    )
    paired_table = pandas.DataFrame(
        {
            "animal_name": ["L14"],
            "date": ["20240611"],
            "unit": [1],
            "comparison_label": ["left_turn"],
            "similarity_light": [0.25],
            "similarity_dark": [0.75],
        }
    )

    def fake_load_similarity_table(**kwargs: object) -> object:
        calls["similarity_kwargs"] = kwargs
        return similarity_table

    def fake_build_similarity_pairs(table: object) -> object:
        calls["similarity_table"] = table
        return paired_table

    monkeypatch.setattr(
        figure_2_common_module,
        "load_panel_c_similarity_table",
        fake_load_similarity_table,
    )
    monkeypatch.setattr(
        figure_2_common_module,
        "build_panel_c_similarity_pairs",
        fake_build_similarity_pairs,
    )

    result = load_panel_b_tuning_correlation_table(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
    )

    assert result is paired_table
    assert calls["similarity_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "v1",
        "light_epoch": None,
        "dark_epoch": None,
    }
    assert calls["similarity_table"] is similarity_table


def test_default_cli_matches_shared_figure_3_canvas() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "figure_2"
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.panel_example_cache_dir is None
    assert args.refresh_panel_example_cache is False
    assert args.dark_tuning_correlation_threshold == pytest.approx(
        PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
    )
    assert args.high_dark_tuning_correlation_threshold == pytest.approx(
        PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD
    )
    assert args.region is None
    assert args.position_bin_count == DEFAULT_POSITION_BIN_COUNT
    assert args.position_offset == DEFAULT_POSITION_OFFSET
    assert args.speed_threshold_cm_s == pytest.approx(DEFAULT_SPEED_THRESHOLD_CM_S)
    assert args.sigma_bins == pytest.approx(DEFAULT_SIGMA_BINS)
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(FIGURE_3_WIDTH_MM)
    assert PANEL_BC_ROW_HEIGHT_MM == pytest.approx(
        FIGURE_3_HEIGHT_MM
        * 1.3
        * PANEL_A_TO_GH_HEIGHT_RATIOS[1]
        / sum(PANEL_A_TO_GH_HEIGHT_RATIOS)
    )
    assert DEFAULT_FIGURE_HEIGHT_MM > PANEL_A_EXAMPLE_ROW_HEIGHT_MM
    assert DEFAULT_REGIONS == ("v1",)
    assert PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD == pytest.approx(0.5)
    assert PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD == pytest.approx(0.75)
    assert PANEL_B_TUNING_CORRELATION_TRAJECTORIES == (
        "center_to_left",
        "right_to_center",
        "center_to_right",
        "left_to_center",
    )
    assert PANEL_B_BOX_COLORS == {
        "low_dpp": "#72B7B2",
        "mid_dpp": "#9E9E9E",
        "high_dpp": "#E45756",
    }
    assert PANEL_B_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)
    assert PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION == pytest.approx(0.5)
    assert "visual" in PANEL_B_EXAMPLE_MODEL_COLORS
    assert len(PANEL_B_EXAMPLE_MODEL_COLORS) == 2
    assert set(PANEL_B_EXAMPLE_MODEL_LABELS) == set(PANEL_B_EXAMPLE_MODEL_COLORS)
    assert PANEL_C_DARK_LIGHT_EXAMPLES == (
        ("L14", "20240611", "v1", 34, "center_to_left"),
        ("L15", "20241121", "v1", 473, "center_to_right"),
        ("L12", "20240421", "v1", 37, "left_to_center"),
        ("L14", "20240611", "v1", 30, "center_to_left"),
    )
    assert len(PANEL_C_SWAP_EXAMPLES) == 3
    assert all(len(example) == 5 for example in PANEL_C_DARK_LIGHT_EXAMPLES)
    assert all(len(example) == 5 for example in PANEL_C_SWAP_EXAMPLES)
    assert PANEL_C_SWAP_MODEL_NAME == "task_segment_scalar"
    assert set(PANEL_C_SWAP_MODEL_LABELS) == {PANEL_C_SWAP_MODEL_NAME}


def test_plot_panel_a_examples_single_row_draws_all_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls = []

    def fake_plot_panel_a_example(ax: object, example: object, **kwargs: object) -> None:
        calls.append((example, kwargs))

    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_a_example",
        fake_plot_panel_a_example,
    )

    examples = [{"unit_id": unit_id} for unit_id in (34, 473, 37, 30)]
    fig, ax = plt.subplots()
    figure_2_module.plot_panel_a_examples_single_row(ax, examples)

    assert [example for example, _kwargs in calls] == examples
    panel_a_spacing_kwargs = {
        "dark_epoch_axis_left": figure_2_module.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT,
        "light_epoch_axis_left": figure_2_module.PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT,
        "epoch_axis_width": figure_2_module.PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH,
        "schematic_axis_left": (
            figure_2_module.PANEL_A_SINGLE_ROW_SCHEMATIC_AXIS_LEFT
        ),
    }
    expected_kwargs = [
        {
            "title": None,
            **panel_a_spacing_kwargs,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        },
        {
            "title": None,
            **panel_a_spacing_kwargs,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        },
        {
            "title": None,
            **panel_a_spacing_kwargs,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        },
        {
            "title": None,
            **panel_a_spacing_kwargs,
            "show_correlation": False,
            "similarity_annotation": "dppi",
            "y_max": 85.0,
        },
    ]
    assert [kwargs for _example, kwargs in calls] == expected_kwargs
    assert [text.get_text() for text in ax.texts] == []
    assert len(ax.child_axes) == 4
    fig.canvas.draw()
    example_widths = [
        example_ax.get_position().bounds[2]
        for example_ax in ax.child_axes
    ]
    assert example_widths == pytest.approx([example_widths[0]] * 4)
    xlabel_y_positions = [
        text.get_position()[1]
        for example_ax in ax.child_axes
        for text in example_ax.texts
        if text.get_text() == "Norm. path progression"
    ]
    assert xlabel_y_positions == pytest.approx(
        [figure_2_module.PANEL_A_SINGLE_ROW_XLABEL_Y] * 4
    )
    plt.close(fig)


def test_raise_text_to_minimum_fontsize_includes_schematic_a_b_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.text(0.1, 0.1, "A", fontsize=3.1)
    child_ax = ax.inset_axes([0.2, 0.2, 0.2, 0.2])
    child_ax.text(0.1, 0.1, "B", fontsize=3.8)

    figure_2_module._raise_text_to_minimum_fontsize(
        fig,
        figure_2_module.MIN_PUBLICATION_FONTSIZE_PT,
    )

    assert ax.texts[0].get_fontsize() == pytest.approx(
        figure_2_module.MIN_PUBLICATION_FONTSIZE_PT
    )
    assert child_ax.texts[0].get_fontsize() == pytest.approx(
        figure_2_module.MIN_PUBLICATION_FONTSIZE_PT
    )
    plt.close(fig)


def test_plot_panel_b_dppi_schematic_centers_title_and_separates_stack() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    example = {"trajectories": ("center_to_left", "right_to_center")}
    figure_2_module.plot_panel_b_dppi_schematic(ax, example)
    fig.canvas.draw()

    text_by_value = {text.get_text(): text for text in ax.texts}
    assert text_by_value["DPP index"].get_ha() == "center"
    assert text_by_value["DPP index"].get_position() == pytest.approx(
        figure_2_module.PANEL_B_DPPI_TITLE_POSITION
    )
    assert text_by_value[figure_2_module.PANEL_B_DPPI_OVERLAP_DEFINITION].get_ha() == (
        "center"
    )
    assert text_by_value[
        figure_2_module.PANEL_B_DPPI_OVERLAP_DEFINITION
    ].get_position() == pytest.approx(
        figure_2_module.PANEL_B_DPPI_OVERLAP_DEFINITION_POSITION
    )
    assert text_by_value["min(r1,r2)"].get_color() == (
        figure_2_module.PANEL_B_DPPI_MIN_OUTLINE_COLOR
    )
    assert text_by_value["max(r1,r2)"].get_color() == (
        figure_2_module.PANEL_B_DPPI_MAX_OUTLINE_COLOR
    )
    assert figure_2_module.PANEL_B_DPPI_MIN_OUTLINE_COLOR != (
        figure_2_module.PANEL_B_DPPI_MAX_OUTLINE_COLOR
    )
    assert figure_2_module.PANEL_B_DPPI_RATE_COLORS == ("#000000", "#000000")
    assert text_by_value[figure_2_module.PANEL_B_DPPI_EQUATION].get_ha() == "center"

    parent_bounds = ax.get_position().bounds
    child_bounds = []
    for child_ax in ax.child_axes:
        child_position = child_ax.get_position().bounds
        child_bounds.append(
            (
                (child_position[0] - parent_bounds[0]) / parent_bounds[2],
                (child_position[1] - parent_bounds[1]) / parent_bounds[3],
                child_position[2] / parent_bounds[2],
                child_position[3] / parent_bounds[3],
            )
        )
    assert len(child_bounds) == 1
    assert child_bounds[0] == pytest.approx(
        figure_2_module.PANEL_B_DPPI_CURVE_AXIS_BOUNDS
    )
    curve_ax = ax.child_axes[0]
    assert [text.get_text() for text in curve_ax.texts] == ["r1", "r2"]
    assert [
        coordinate
        for text in curve_ax.texts
        for coordinate in text.get_position()
    ] == pytest.approx(
        [
            coordinate
            for position in figure_2_module.PANEL_B_DPPI_RATE_LABEL_POSITIONS
            for coordinate in position
        ]
    )
    assert [text.get_fontsize() for text in curve_ax.texts] == pytest.approx(
        [figure_2_module.MIN_PUBLICATION_FONTSIZE_PT] * 2
    )
    assert [text.get_color() for text in curve_ax.texts] == list(
        figure_2_module.PANEL_B_DPPI_RATE_COLORS
    )
    assert len(curve_ax.lines) == 4
    assert [line.get_color() for line in curve_ax.lines[:2]] == list(
        figure_2_module.PANEL_B_DPPI_RATE_COLORS
    )
    assert len(curve_ax.collections) == 2
    assert [collection.get_alpha() for collection in curve_ax.collections] == (
        pytest.approx(
            [
                figure_2_module.PANEL_B_DPPI_MAX_FILL_ALPHA,
                figure_2_module.PANEL_B_DPPI_MIN_FILL_ALPHA,
            ]
        )
    )
    assert [line.get_linewidth() for line in curve_ax.lines[:2]] == pytest.approx(
        [figure_2_module.PANEL_B_DPPI_RATE_LINEWIDTH] * 2
    )
    assert [line.get_linewidth() for line in curve_ax.lines[2:]] == pytest.approx(
        [figure_2_module.PANEL_B_DPPI_OUTLINE_LINEWIDTH] * 2
    )
    plt.close(fig)


def test_plot_panel_b_dpp_overlap_with_schematic_preserves_marginal_axis_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots()
    table = pd.DataFrame(
        {
            "similarity_dark": [0.20, 0.45, 0.55, 0.78, 0.90],
            "similarity_light": [0.25, 0.32, 0.46, 0.62, 0.82],
        }
    )
    figure_2_module.plot_panel_b_dpp_overlap_with_schematic(
        ax,
        table,
        example={"trajectories": ("center_to_left", "right_to_center")},
        low_threshold=0.5,
        high_threshold=0.75,
    )
    fig.canvas.draw()

    parent_bounds = ax.get_position().bounds
    scatter_parent = next(
        child_ax for child_ax in ax.child_axes if len(child_ax.child_axes) == 3
    )
    scatter_bounds = scatter_parent.get_position().bounds
    assert (
        (scatter_bounds[0] - parent_bounds[0]) / parent_bounds[2],
        (scatter_bounds[1] - parent_bounds[1]) / parent_bounds[3],
        scatter_bounds[2] / parent_bounds[2],
        scatter_bounds[3] / parent_bounds[3],
    ) == pytest.approx(figure_2_module.PANEL_B_SCATTER_AXIS_BOUNDS)

    main_ax = next(
        child_ax
        for child_ax in scatter_parent.child_axes
        if child_ax.get_xlabel() == "Dark DPPI"
    )
    assert main_ax.get_ylabel() == "Light DPPI"
    assert [tick.get_text() for tick in main_ax.get_xticklabels()] == ["0", "0.5", "1"]
    assert [tick.get_text() for tick in main_ax.get_yticklabels()] == ["0", "0.5", "1"]

    marginal_axes = [
        child_ax for child_ax in scatter_parent.child_axes if child_ax is not main_ax
    ]
    assert len(marginal_axes) == 2
    assert sorted(child_ax.get_xlabel() for child_ax in marginal_axes) == ["", "Frac."]
    assert sorted(child_ax.get_ylabel() for child_ax in marginal_axes) == ["", "Frac."]
    top_ax = next(child_ax for child_ax in marginal_axes if child_ax.get_ylabel() == "Frac.")
    right_ax = next(child_ax for child_ax in marginal_axes if child_ax.get_xlabel() == "Frac.")
    assert [tick.get_text() for tick in top_ax.get_yticklabels()] == ["0", "0.1"]
    assert [tick.get_text() for tick in right_ax.get_xticklabels()] == ["0", "0.1"]
    assert not any(
        tick.get_visible() and tick.get_text()
        for tick in (*top_ax.get_xticklabels(), *right_ax.get_yticklabels())
    )
    plt.close(fig)


def test_plot_panel_b_dpp_overlap_with_schematic_can_omit_grouped_and_show_fit() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots()
    table = pd.DataFrame(
        {
            "similarity_dark": [0.10, 0.30, 0.50, 0.70, 0.90],
            "similarity_light": [0.18, 0.34, 0.50, 0.66, 0.82],
        }
    )
    figure_2_module.plot_panel_b_dpp_overlap_with_schematic(
        ax,
        table,
        example={"trajectories": ("center_to_left", "right_to_center")},
        low_threshold=0.5,
        high_threshold=0.75,
        show_grouped=False,
        show_scatter_linear_fit=True,
        show_scatter_r2=True,
        scatter_equal_aspect=True,
    )
    fig.canvas.draw()

    assert len(ax.child_axes) == 2
    parent_bounds = ax.get_position().bounds
    schematic_parent = next(
        child_ax for child_ax in ax.child_axes if len(child_ax.child_axes) == 1
    )
    scatter_parent = next(
        child_ax for child_ax in ax.child_axes if len(child_ax.child_axes) == 3
    )
    schematic_bounds = schematic_parent.get_position().bounds
    scatter_bounds = scatter_parent.get_position().bounds
    assert (
        schematic_bounds[2] / parent_bounds[2]
    ) == pytest.approx(
        figure_2_module.PANEL_B_DPPI_SCHEMATIC_WITHOUT_GROUP_AXIS_BOUNDS[2]
    )
    assert scatter_bounds[2] / schematic_bounds[2] == pytest.approx(1.5)
    assert (
        (scatter_bounds[0] - parent_bounds[0]) / parent_bounds[2],
        (scatter_bounds[1] - parent_bounds[1]) / parent_bounds[3],
        scatter_bounds[2] / parent_bounds[2],
        scatter_bounds[3] / parent_bounds[3],
    ) == pytest.approx(figure_2_module.PANEL_B_SCATTER_WITHOUT_GROUP_AXIS_BOUNDS)

    main_ax = next(
        child_ax
        for child_ax in scatter_parent.child_axes
        if child_ax.get_xlabel() == "Dark DPPI"
    )
    assert main_ax.get_box_aspect() == pytest.approx(1.0)
    assert [text.get_text() for text in main_ax.texts] == ["R²=1.00"]
    assert len(main_ax.lines) == 4
    assert main_ax.lines[-1].get_color() == "black"
    top_ax = next(
        child_ax
        for child_ax in scatter_parent.child_axes
        if child_ax.get_ylabel() == "Frac."
    )
    right_ax = next(
        child_ax
        for child_ax in scatter_parent.child_axes
        if child_ax.get_xlabel() == "Frac."
    )
    main_bounds = main_ax.get_position().bounds
    top_bounds = top_ax.get_position().bounds
    right_bounds = right_ax.get_position().bounds
    assert top_bounds[0] == pytest.approx(main_bounds[0])
    assert top_bounds[2] == pytest.approx(main_bounds[2])
    assert right_bounds[1] == pytest.approx(main_bounds[1])
    assert right_bounds[3] == pytest.approx(main_bounds[3])
    top_gap = (top_bounds[1] - (main_bounds[1] + main_bounds[3])) / scatter_bounds[3]
    right_gap = (right_bounds[0] - (main_bounds[0] + main_bounds[2])) / scatter_bounds[2]
    assert right_gap == pytest.approx(top_gap)
    renderer = fig.canvas.get_renderer()
    main_ylabel_center_x = main_ax.yaxis.label.get_window_extent(renderer).x0 + (
        main_ax.yaxis.label.get_window_extent(renderer).width / 2.0
    )
    top_ylabel_center_x = top_ax.yaxis.label.get_window_extent(renderer).x0 + (
        top_ax.yaxis.label.get_window_extent(renderer).width / 2.0
    )
    assert top_ylabel_center_x == pytest.approx(main_ylabel_center_x, abs=0.5)
    main_xlabel_center_y = main_ax.xaxis.label.get_window_extent(renderer).y0 + (
        main_ax.xaxis.label.get_window_extent(renderer).height / 2.0
    )
    right_xlabel_center_y = right_ax.xaxis.label.get_window_extent(renderer).y0 + (
        right_ax.xaxis.label.get_window_extent(renderer).height / 2.0
    )
    assert right_xlabel_center_y == pytest.approx(main_xlabel_center_y, abs=0.5)
    assert [tick.get_text() for tick in top_ax.get_yticklabels()] == ["0", "0.1"]
    assert [tick.get_text() for tick in right_ax.get_xticklabels()] == ["0", "0.1"]
    plt.close(fig)


def test_add_panel_d2_trace_legend_labels_all_prediction_traces() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    legend = figure_2_2_module._add_panel_d2_trace_legend(
        ax,
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        model_colors=figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=figure_2_module.PANEL_C_SWAP_MODEL_LABELS_2_3,
    )

    assert [text.get_text() for text in legend.get_texts()] == [
        "Empirical",
        "Independent",
        "Dark scaffold",
    ]
    assert [handle.get_color() for handle in legend.legend_handles] == [
        figure_2_2_module.GLM_EMPIRICAL_COLOR,
        figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3["visual"],
        figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3[
            figure_2_module.PANEL_C_SWAP_MODEL_NAME
        ],
    ]
    plt.close(fig)


def test_panel_d2_swap_results_uses_left_example_grid_and_right_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls: dict[str, object] = {}

    def fake_plot_panel_h_switched_segment_example(
        ax: object,
        example: object,
        **kwargs: object,
    ) -> None:
        calls.setdefault("examples", []).append((ax, kwargs))
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["0", "1"])
        ax.set_title(str(kwargs["example_label"]))
        ax.text(
            0.96,
            0.94,
            f"ΔLL={float(example['delta_ll_bits_per_spike']):.2f}",
        )

    def fake_plot_panel_d_mean_swap_delta_axis(
        ax: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        calls["histogram_axis"] = ax

    monkeypatch.setattr(
        figure_2_module,
        "_plot_panel_h_switched_segment_example",
        fake_plot_panel_h_switched_segment_example,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d_mean_swap_delta_axis",
        fake_plot_panel_d_mean_swap_delta_axis,
    )

    fig, ax = plt.subplots()
    figure_2_2_module._plot_panel_d2_swap_results(
        ax,
        swap_delta_table=None,
        swap_examples=[
            {"delta_ll_bits_per_spike": 0.44},
            {"delta_ll_bits_per_spike": 0.13},
            {"delta_ll_bits_per_spike": 0.91},
        ],
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        model_colors=figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=figure_2_module.PANEL_C_SWAP_MODEL_LABELS_2_3,
    )
    fig.canvas.draw()

    def _relative_bounds(parent_ax: object, child_ax: object) -> tuple[float, ...]:
        parent_bounds = parent_ax.get_position().bounds
        child_bounds = child_ax.get_position().bounds
        return (
            (child_bounds[0] - parent_bounds[0]) / parent_bounds[2],
            (child_bounds[1] - parent_bounds[1]) / parent_bounds[3],
            child_bounds[2] / parent_bounds[2],
            child_bounds[3] / parent_bounds[3],
        )

    example_calls = calls["examples"]
    example_axes = [example_axis for example_axis, _kwargs in example_calls]
    for example_axis, expected_bounds in zip(
        example_axes,
        figure_2_2_module.PANEL_D2_EXAMPLE_SLOT_BOUNDS,
        strict=True,
    ):
        assert _relative_bounds(ax, example_axis) == pytest.approx(expected_bounds)
        assert expected_bounds[0] < 0.6

    example_kwargs = [kwargs for _example_axis, kwargs in example_calls]
    assert [kwargs["show_ylabel"] for kwargs in example_kwargs] == [True, False, True]
    assert [kwargs["show_xlabel"] for kwargs in example_kwargs] == [False, False, True]
    assert [kwargs["show_xticklabels"] for kwargs in example_kwargs] == [
        True,
        True,
        True,
    ]
    assert [example_axis.get_title() for example_axis in example_axes] == [
        "Ex. 1 (ΔLL=0.44)",
        "Ex. 2 (ΔLL=0.13)",
        "Ex. 3 (ΔLL=0.91)",
    ]
    assert [example_axis.title.get_position()[0] for example_axis in example_axes] == (
        pytest.approx([figure_2_2_module.PANEL_D2_EXAMPLE_HEADER_X] * 3)
    )
    assert not [
        text
        for example_axis in example_axes
        for text in example_axis.texts
        if text.get_text().startswith("ΔLL=")
    ]
    assert all(
        label.get_visible()
        for example_axis in example_axes
        for label in example_axis.get_xticklabels()
    )
    assert all(
        label.get_visible()
        for example_axis in example_axes
        for label in example_axis.get_yticklabels()
    )

    histogram_axis = calls["histogram_axis"]
    assert _relative_bounds(ax, histogram_axis) == pytest.approx(
        figure_2_2_module.PANEL_D2_HISTOGRAM_AXIS_BOUNDS
    )
    example_left = min(
        bounds[0]
        for bounds in (
            *figure_2_2_module.PANEL_D2_EXAMPLE_SLOT_BOUNDS,
            figure_2_2_module.PANEL_D2_TRACE_LEGEND_SLOT_BOUNDS,
        )
    )
    example_right = max(
        bounds[0] + bounds[2]
        for bounds in (
            *figure_2_2_module.PANEL_D2_EXAMPLE_SLOT_BOUNDS,
            figure_2_2_module.PANEL_D2_TRACE_LEGEND_SLOT_BOUNDS,
        )
    )
    histogram_left = figure_2_2_module.PANEL_D2_HISTOGRAM_AXIS_BOUNDS[0]
    histogram_width = figure_2_2_module.PANEL_D2_HISTOGRAM_AXIS_BOUNDS[2]
    assert example_right < histogram_left
    assert histogram_left - example_right > 0.07
    assert (example_right - example_left) / histogram_width == pytest.approx(
        1.0,
        rel=0.03,
    )

    legend_axes = [
        child_ax
        for child_ax in ax.child_axes
        if child_ax not in example_axes and child_ax is not histogram_axis
    ]
    assert len(legend_axes) == 1
    legend_axis = legend_axes[0]
    assert _relative_bounds(ax, legend_axis) == pytest.approx(
        figure_2_2_module.PANEL_D2_TRACE_LEGEND_SLOT_BOUNDS
    )
    assert [text.get_text() for text in legend_axis.get_legend().get_texts()] == [
        "Empirical",
        "Independent",
        "Dark scaffold",
    ]
    plt.close(fig)


def test_plot_panel_d2_swap_results_panel_shows_third_example_xlabel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls: dict[str, object] = {}

    def fake_plot_panel_d2_swap_results(
        _ax: object,
        _swap_delta_table: object,
        _swap_examples: object,
        **kwargs: object,
    ) -> None:
        calls.update(kwargs)

    monkeypatch.setattr(
        figure_2_2_module,
        "_plot_panel_d2_swap_results",
        fake_plot_panel_d2_swap_results,
    )

    fig, ax = plt.subplots()
    figure_2_2_module.plot_panel_d2_swap_results_panel(
        ax,
        swap_delta_table=None,
        swap_examples=[],
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
    )

    assert calls["show_example_xlabel"] is True
    plt.close(fig)


def test_add_panel_c2_light_dark_brackets_lowers_right_bracket() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    parent_ax = fig.add_subplot(111, frameon=False)
    parent_ax.child_axes.extend(axes.tolist())
    for axis in axes:
        axis.set_ylim(0.0, 1.0)

    figure_2_2_module.add_panel_c2_light_dark_brackets(parent_ax)

    left_bracket_y = max(axes[0].lines[0].get_ydata())
    right_bracket_y = max(axes[1].lines[0].get_ydata())
    assert right_bracket_y < left_bracket_y
    assert right_bracket_y == pytest.approx(
        figure_2_2_module.PANEL_C2_RIGHT_SIGNIFICANCE_BRACKET_Y_FRACTION
        + figure_2_2_module.DECODING_SIGNIFICANCE_BRACKET_HEIGHT
    )
    plt.close(fig)


def test_panel_c2_dark_scaffold_segment_icon_uses_direction_arrows_and_left_field() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import math
    from matplotlib import colormaps
    from matplotlib.colors import to_hex, to_rgba
    from matplotlib.patches import Ellipse, Rectangle
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    figure_2_2_module.draw_panel_d2_architecture_schematic(ax)
    fig.canvas.draw()

    assert ax.get_zorder() > max(child_ax.get_zorder() for child_ax in ax.child_axes)
    assert figure_2_2_module.PANEL_D2_SEGMENT_MODULATION_LABEL in {
        text.get_text() for text in ax.texts
    }
    assert figure_2_module.PANEL_B_SEGMENT_MODULATION_LABEL not in {
        text.get_text() for text in ax.texts
    }
    independent_dark_ax = ax.child_axes[0]
    independent_light_ax = ax.child_axes[2]
    independent_predict_ax = ax.child_axes[3]
    shared_dark_ax = ax.child_axes[4]
    segment_oval_ax = ax.child_axes[5]
    shared_light_ax = ax.child_axes[6]
    shared_predict_ax = ax.child_axes[7]
    assert not [
        text
        for track_ax in (
            independent_dark_ax,
            independent_light_ax,
            independent_predict_ax,
            shared_dark_ax,
            segment_oval_ax,
            shared_light_ax,
            shared_predict_ax,
        )
        for text in track_ax.texts
        if text.get_text() in {"A", "B"}
    ]
    _outline, _points, dims = get_w_track_geometry()
    left_field_center_x = round(float((dims["x0"] + dims["x1"]) / 2.0), 4)
    right_field_center_x = round(float((dims["x4"] + dims["x5"]) / 2.0), 4)

    ovals = [patch for patch in segment_oval_ax.patches if type(patch) is Ellipse]
    assert len(ovals) == 2
    ovals_by_center_x = {
        round(float(oval.center[0]), 4): oval
        for oval in ovals
    }
    assert set(ovals_by_center_x) == {left_field_center_x, right_field_center_x}
    expected_oval_alphas = {
        left_field_center_x: figure_2_2_module.PANEL_D2_SEGMENT_OVAL_ALPHAS[0],
        right_field_center_x: figure_2_2_module.PANEL_D2_SEGMENT_OVAL_ALPHAS[1],
    }
    for center_x, fill_alpha in expected_oval_alphas.items():
        oval = ovals_by_center_x[center_x]
        assert float(oval.center[1]) == pytest.approx(
            (dims["y1"] + dims["y2"]) / 2.0
        )
        assert oval.width == pytest.approx(dims["corridor_w"] * 0.95)
        assert oval.height == pytest.approx(dims["y2"] - dims["y1"] + 0.25)
        assert oval.get_facecolor() == pytest.approx(
            to_rgba(figure_2_2_module.PANEL_D2_SEGMENT_OVAL_FILL_COLOR, fill_alpha)
        )
        assert oval.get_edgecolor() == pytest.approx(
            to_rgba(figure_2_2_module.PANEL_D2_SEGMENT_OVAL_EDGE_COLOR)
        )
        assert oval.get_linewidth() == pytest.approx(
            figure_2_2_module.PANEL_D2_SEGMENT_OVAL_LINEWIDTH
        )
    arrow_annotations = [
        text
        for text in segment_oval_ax.texts
        if getattr(text, "arrow_patch", None) is not None
        and text.arrow_patch.get_label().startswith("_panel_d2_segment_arrow_")
    ]
    arrow_by_label = {
        text.arrow_patch.get_label(): text
        for text in arrow_annotations
    }
    assert set(arrow_by_label) == {
        "_panel_d2_segment_arrow_up",
        "_panel_d2_segment_arrow_down",
    }
    segment_arrow_length = (
        dims["y2"]
        - dims["y1"]
        - 2.0 * figure_2_2_module.PANEL_D2_SEGMENT_ARROW_Y_MARGIN
    )
    assert (
        arrow_by_label["_panel_d2_segment_arrow_up"].xy[1]
        > arrow_by_label["_panel_d2_segment_arrow_up"].xyann[1]
    )
    assert (
        arrow_by_label["_panel_d2_segment_arrow_up"].xy[1]
        - arrow_by_label["_panel_d2_segment_arrow_up"].xyann[1]
    ) == pytest.approx(segment_arrow_length)
    assert (
        arrow_by_label["_panel_d2_segment_arrow_down"].xy[1]
        < arrow_by_label["_panel_d2_segment_arrow_down"].xyann[1]
    )
    assert (
        arrow_by_label["_panel_d2_segment_arrow_down"].xyann[1]
        - arrow_by_label["_panel_d2_segment_arrow_down"].xy[1]
    ) == pytest.approx(segment_arrow_length)
    assert all(
        text.arrow_patch.get_mutation_scale()
        == pytest.approx(figure_2_2_module.PANEL_D2_SEGMENT_ARROW_MUTATION_SCALE)
        for text in arrow_annotations
    )
    assert all(
        text.arrow_patch.get_linewidth()
        == pytest.approx(figure_2_2_module.PANEL_D2_SEGMENT_ARROW_LINEWIDTH)
        for text in arrow_annotations
    )
    side_outline_gap = figure_2_2_module.PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_GAP
    expected_side_outlines = {
        "_panel_d2_segment_arm_side_outline_left_arm_left": (
            dims["x0"] - side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["A"],
        ),
        "_panel_d2_segment_arm_side_outline_left_arm_right": (
            dims["x1"] + side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["A"],
        ),
        "_panel_d2_segment_arm_side_outline_right_arm_left": (
            dims["x4"] - side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["B"],
        ),
        "_panel_d2_segment_arm_side_outline_right_arm_right": (
            dims["x5"] + side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["B"],
        ),
    }
    expected_prediction_side_outlines = {
        "_panel_d2_segment_arm_side_outline_left_arm_left": (
            dims["x0"] - side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["B"],
        ),
        "_panel_d2_segment_arm_side_outline_left_arm_right": (
            dims["x1"] + side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["B"],
        ),
        "_panel_d2_segment_arm_side_outline_right_arm_left": (
            dims["x4"] - side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["A"],
        ),
        "_panel_d2_segment_arm_side_outline_right_arm_right": (
            dims["x5"] + side_outline_gap,
            figure_2_module.PANEL_B_VISUAL_ICON_COLORS["A"],
        ),
    }
    assert side_outline_gap > 0.0
    for track_ax, expected_track_side_outlines in (
        (segment_oval_ax, expected_side_outlines),
        (independent_light_ax, expected_side_outlines),
        (independent_predict_ax, expected_prediction_side_outlines),
        (shared_light_ax, expected_side_outlines),
        (shared_predict_ax, expected_prediction_side_outlines),
    ):
        side_outline_lines = {
            line.get_label(): line
            for line in track_ax.lines
            if line.get_label().startswith("_panel_d2_segment_arm_side_outline_")
        }
        assert set(side_outline_lines) == set(expected_track_side_outlines)
        for label, (x_position, color) in expected_track_side_outlines.items():
            line = side_outline_lines[label]
            assert list(line.get_xdata()) == pytest.approx([x_position, x_position])
            assert list(line.get_ydata()) == pytest.approx([dims["y1"], dims["y2"]])
            assert to_rgba(line.get_color()) == pytest.approx(to_rgba(color))
            assert line.get_linewidth() == pytest.approx(
                figure_2_2_module.PANEL_D2_SEGMENT_ARM_SIDE_OUTLINE_LINEWIDTH
            )
    assert not [
        patch
        for track_ax in (
            independent_light_ax,
            independent_predict_ax,
            shared_light_ax,
            shared_predict_ax,
        )
        for patch in track_ax.patches
        if type(patch) is Rectangle
    ]

    base_field_colors = figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_BASE_COLORS
    assert figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP == "inferno"
    assert base_field_colors == tuple(
        to_hex(
            colormaps[figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP](
                value
            )
        )
        for value in figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_COLOR_VALUES
    )
    assert figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_COLORS == (
        base_field_colors
    )
    assert figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_COLORS == (
        base_field_colors
    )
    assert figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_COLORS == (
        base_field_colors
    )
    assert (
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_RATE_GAIN
        < figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN
        < figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN
    )
    assert 0.0 < figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA < 1.0

    def _assert_rate_gain_colors(track_ax: object, gain: float) -> None:
        field_center_y = dims["y1"] + 1.45
        field_sigma = 0.58
        cmap = colormaps[figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_COLORMAP]
        gamma = figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_FIELD_RATE_GAMMA
        for patch in track_ax.patches:
            if type(patch) is not Ellipse:
                continue
            relative_rate = math.exp(
                -0.5
                * ((float(patch.center[1]) - field_center_y) / field_sigma) ** 2
            )
            color_value = min(max(float(gain) * relative_rate, 0.0), 1.0)
            color_value = color_value**gamma
            assert tuple(
                round(float(value), 4) for value in patch.get_facecolor()[:3]
            ) == pytest.approx(
                tuple(round(float(value), 4) for value in cmap(color_value)[:3]),
                abs=0.01,
            )

    _assert_rate_gain_colors(
        independent_dark_ax,
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN,
    )
    _assert_rate_gain_colors(
        shared_dark_ax,
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_RATE_GAIN,
    )
    _assert_rate_gain_colors(
        independent_light_ax,
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN,
    )
    _assert_rate_gain_colors(
        independent_predict_ax,
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN,
    )
    _assert_rate_gain_colors(
        shared_light_ax,
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_RATE_GAIN,
    )
    _assert_rate_gain_colors(
        shared_predict_ax,
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_RATE_GAIN,
    )

    independent_dark_fields = [
        patch for patch in independent_dark_ax.patches if type(patch) is Ellipse
    ]
    independent_light_fields = [
        patch for patch in independent_light_ax.patches if type(patch) is Ellipse
    ]
    independent_predict_fields = [
        patch for patch in independent_predict_ax.patches if type(patch) is Ellipse
    ]
    shared_dark_fields = [
        patch for patch in shared_dark_ax.patches if type(patch) is Ellipse
    ]
    shared_light_fields = [
        patch for patch in shared_light_ax.patches if type(patch) is Ellipse
    ]
    shared_predict_fields = [
        patch for patch in shared_predict_ax.patches if type(patch) is Ellipse
    ]
    assert {
        round(float(patch.center[0]), 4)
        for patch in shared_predict_fields
    } == {left_field_center_x}
    assert {
        round(float(patch.center[0]), 4)
        for patch in independent_predict_fields
    } == {right_field_center_x}
    assert [
        (
            round(float(patch.center[0]), 4),
            round(float(patch.center[1]), 4),
            tuple(round(float(value), 4) for value in patch.get_facecolor()),
            round(float(patch.get_alpha()), 4),
        )
        for patch in independent_dark_fields
    ] == [
        (
            round(float(patch.center[0]), 4),
            round(float(patch.center[1]), 4),
            tuple(round(float(value), 4) for value in patch.get_facecolor()),
            round(float(patch.get_alpha()), 4),
        )
        for patch in shared_dark_fields
    ]
    assert [
        (
            round(float(patch.center[0]), 4),
            round(float(patch.center[1]), 4),
            tuple(round(float(value), 4) for value in patch.get_facecolor()),
            round(float(patch.get_alpha()), 4),
        )
        for patch in independent_light_fields
    ] == [
        (
            round(float(patch.center[0]), 4),
            round(float(patch.center[1]), 4),
            tuple(round(float(value), 4) for value in patch.get_facecolor()),
            round(float(patch.get_alpha()), 4),
        )
        for patch in shared_light_fields
    ]
    assert [
        (
            round(float(patch.center[1]), 4),
            tuple(round(float(value), 4) for value in patch.get_facecolor()),
            round(float(patch.get_alpha()), 4),
        )
        for patch in independent_predict_fields
    ] == [
        (
            round(float(patch.center[1]), 4),
            tuple(round(float(value), 4) for value in patch.get_facecolor()),
            round(float(patch.get_alpha()), 4),
        )
        for patch in independent_light_fields
    ]
    assert [patch.get_alpha() for patch in independent_dark_fields] == pytest.approx(
        [figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_ALPHA]
        * len(independent_dark_fields)
    )
    assert [patch.get_alpha() for patch in independent_light_fields] == pytest.approx(
        [figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA]
        * len(independent_light_fields)
    )
    assert [patch.get_alpha() for patch in independent_predict_fields] == pytest.approx(
        [figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA]
        * len(independent_predict_fields)
    )
    assert [patch.get_alpha() for patch in shared_dark_fields] == pytest.approx(
        [figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_ALPHA]
        * len(shared_dark_fields)
    )
    assert [patch.get_alpha() for patch in shared_light_fields] == pytest.approx(
        [figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA]
        * len(shared_light_fields)
    )
    assert [patch.get_alpha() for patch in shared_predict_fields] == pytest.approx(
        [figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_ALPHA]
        * len(shared_predict_fields)
    )
    assert (
        figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_LIGHT_FIELD_ALPHA
        == figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_DARK_FIELD_ALPHA
        == figure_2_2_module.PANEL_D2_DARK_SCAFFOLD_PREDICTION_FIELD_ALPHA
        == pytest.approx(1.0)
    )
    plt.close(fig)


def test_plot_panel_b_light_similarity_by_dark_similarity_uses_fig3b_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    table = pd.DataFrame(
        {
            "similarity_dark": [0.8, 0.4, np.nan, 0.5, 0.7],
            "similarity_light": [0.8, 0.6, 0.1, -0.2, 0.7],
        }
    )

    fig, ax = plt.subplots()
    figure_2_module.plot_panel_b_light_similarity_by_dark_similarity(ax, table)

    assert len(ax.child_axes) == 1
    dpp_ax = ax.child_axes[0]
    assert [tick.get_text() for tick in dpp_ax.get_xticklabels()] == [
        "Dark DPP\n<0.5\nn=1",
        "Dark DPP\n0.5-0.75\nn=2",
        "Dark DPP\n>=0.75\nn=1",
    ]
    assert len(dpp_ax.patches) == 3
    assert dpp_ax.texts[-1].get_text() == (
        "low 100% > 0.5\nmid 50% > 0.5\nhigh 100% > 0.5"
    )
    assert dpp_ax.get_ylabel() == "Light DPP\ncorr."
    assert dpp_ax.get_title() == "Grouped by dark DPP"
    plt.close(fig)


def test_panel_d_example_layout_requests_per_example_axis_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls = {}

    def fake_plot_panel_g_example_columns(
        ax: object,
        examples: object,
        **kwargs: object,
    ) -> None:
        calls["examples"] = examples
        calls["kwargs"] = kwargs
        for label_index in range(4):
            ax.text(
                0.15 + label_index * 0.20,
                0.05,
                "Norm. path progression",
                fontsize=3.7,
            )

    monkeypatch.setattr(
        figure_2_module,
        "_plot_panel_g_example_columns",
        fake_plot_panel_g_example_columns,
    )

    examples = [{"example_id": example_id} for example_id in range(1, 5)]
    fig, ax = plt.subplots()
    figure_2_module.plot_panel_c_model_architecture_row(ax, examples)

    assert calls["examples"] == examples
    assert calls["kwargs"]["show_ylabels_for_all_examples"] is True
    assert calls["kwargs"]["show_epoch_titles"] is False
    assert calls["kwargs"]["show_light_yticklabels"] is False
    assert calls["kwargs"]["field_height"] == pytest.approx(
        figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_HEIGHT
    )
    assert calls["kwargs"]["field_width"] == pytest.approx(
        figure_2_module.PANEL_C_SIDE_BY_SIDE_EXAMPLE_FIELD_WIDTH
    )
    example_ax = ax.child_axes[1]
    path_labels = [
        text
        for text in example_ax.texts
        if text.get_text() == "Norm. path progression"
    ]
    assert len(path_labels) == 4
    assert [text.get_fontsize() for text in path_labels] == pytest.approx(
        [figure_2_module.MIN_PUBLICATION_FONTSIZE_PT] * 4
    )
    plt.close(fig)


def test_panel_e_mean_delta_histogram_uses_figure_1_delta_convention() -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for animal_name, date, unit, value in (
        ("L14", "20240611", 1, -0.2),
        ("L15", "20241121", 2, 0.3),
    ):
        for trajectory in figure_2_module.PANEL_H_DELTA_TRAJECTORIES:
            rows.append(
                {
                    "animal_name": animal_name,
                    "date": date,
                    "region": "v1",
                    "dark_epoch": "08_r4",
                    "unit": unit,
                    "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
                    "trajectory": trajectory,
                    "delta_ll_bits_per_spike": value,
                }
            )
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    figure_2_module.plot_panel_d_mean_swap_delta_axis(
        ax,
        table,
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        model_colors=figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3,
        model_labels=figure_2_module.PANEL_C_SWAP_MODEL_LABELS_2_3,
    )

    assert ax.get_xlim() == pytest.approx((-1.0, 1.0))
    assert ax.get_ylabel() == "Fraction"
    assert ax.get_xlabel() == figure_2_module.DELTA_LOG_LIKELIHOOD_AXIS_LABEL
    assert ax.xaxis.label.get_fontsize() == pytest.approx(7.0)
    assert ax.get_title() == ""
    text_labels = [text.get_text() for text in ax.texts]
    assert "Indep. better" in text_labels
    assert "Dark scaffold\nbetter" in text_labels
    assert "50% >0" in text_labels
    assert "n = 2 cells\n2 animals" in text_labels
    assert "Mean model advantage" not in text_labels
    assert ax.texts[2].get_bbox_patch() is None
    assert ax.patches[0].get_extents().x1 <= ax.transData.transform((0.0, 0.0))[0]
    assert ax.lines[0].get_color() == "black"
    plt.close(fig)


def test_panel_d_independent_light_icon_uses_figure_1b_arm_colors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, Ellipse, PathPatch, Rectangle
    import matplotlib.pyplot as plt
    import v1ca1.paper_figures.old_fig3 as old_fig3_module

    monkeypatch.setattr(
        figure_2_module,
        "_plot_panel_g_example_columns",
        lambda *_args, **_kwargs: None,
    )

    fig, ax = plt.subplots()
    figure_2_module.plot_panel_c_model_architecture_row(ax, [])
    fig.canvas.draw()

    schematic_ax = ax.child_axes[0]
    independent_dark_ax = schematic_ax.child_axes[0]
    independent_basis_ax = schematic_ax.child_axes[1]
    independent_light_ax = schematic_ax.child_axes[2]
    shared_dark_ax = schematic_ax.child_axes[3]
    segment_gain_ax = schematic_ax.child_axes[4]
    shared_light_ax = schematic_ax.child_axes[5]

    def _relative_bounds(parent_ax: object, child_ax: object) -> tuple[float, ...]:
        parent_bounds = parent_ax.get_position(original=True).bounds
        child_bounds = child_ax.get_position(original=True).bounds
        return (
            (child_bounds[0] - parent_bounds[0]) / parent_bounds[2],
            (child_bounds[1] - parent_bounds[1]) / parent_bounds[3],
            child_bounds[2] / parent_bounds[2],
            child_bounds[3] / parent_bounds[3],
        )

    def _center_x(bounds: tuple[float, ...]) -> float:
        return bounds[0] + bounds[2] / 2.0

    panel_d_track_bounds = {
        "independent_dark": _relative_bounds(schematic_ax, independent_dark_ax),
        "independent_light": _relative_bounds(schematic_ax, independent_light_ax),
        "shared_dark": _relative_bounds(schematic_ax, shared_dark_ax),
        "segment_gain": _relative_bounds(schematic_ax, segment_gain_ax),
        "shared_light": _relative_bounds(schematic_ax, shared_light_ax),
    }
    panel_d_model_label_xs = [
        text.get_position()[0]
        for text in schematic_ax.texts
        if text.get_fontweight() == "bold" and text.get_text().endswith("\nmodel")
    ]
    assert panel_d_model_label_xs == pytest.approx(
        [figure_2_module.PANEL_B_MODEL_LABEL_X] * 2
    )
    expected_arm_colors = [
        to_rgba(
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim1"][
                "left_arm"
            ],
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        ),
        to_rgba(
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim1"][
                "right_arm"
            ],
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        ),
    ]
    expected_label_colors = {
        "A": to_rgba(figure_2_module.PANEL_B_VISUAL_ICON_COLORS["A"]),
        "B": to_rgba(figure_2_module.PANEL_B_VISUAL_ICON_COLORS["B"]),
    }
    expected_gain_regions = (
        "left_arm",
    )
    expected_gain_edge_colors = [
        to_rgba(
            figure_2_module.PANEL_D_CENTER_TO_LEFT_SEGMENT_OUTLINE_COLORS[
                region_name
            ],
        )
        for region_name in expected_gain_regions
    ]
    assert len(set(expected_gain_edge_colors)) == len(expected_gain_regions)
    assert expected_gain_edge_colors == [to_rgba("#E69F00")]
    expected_gain_linewidths = [
        figure_2_module.PANEL_D_SEGMENT_GAIN_OUTLINE_LINEWIDTHS[region_name]
        for region_name in expected_gain_regions
    ]
    horizontal_line = independent_basis_ax.lines[0]
    vertical_line = independent_basis_ax.lines[1]
    horizontal_span = abs(horizontal_line.get_xdata()[1] - horizontal_line.get_xdata()[0])
    vertical_span = abs(vertical_line.get_ydata()[1] - vertical_line.get_ydata()[0])
    assert horizontal_span == pytest.approx(vertical_span)
    _outline, _points, dims = get_w_track_geometry()
    expected_gain_bounds = {
        "left_arm": (
            dims["x0"],
            dims["y1"],
            dims["x1"] - dims["x0"],
            dims["y2"] - dims["y1"],
        ),
    }
    independent_rectangles = [
        patch for patch in independent_light_ax.patches if isinstance(patch, Rectangle)
    ]
    shared_rectangles = [
        patch for patch in shared_light_ax.patches if isinstance(patch, Rectangle)
    ]
    shared_fill_rectangles = [
        patch for patch in shared_rectangles if patch.get_facecolor()[3] > 0.0
    ]
    shared_segment_patches = [
        patch
        for patch in shared_light_ax.patches
        if isinstance(patch, PathPatch)
    ]
    assert [patch.get_facecolor() for patch in independent_rectangles] == pytest.approx(
        expected_arm_colors
    )
    assert [patch.get_facecolor() for patch in shared_fill_rectangles] == pytest.approx(
        expected_arm_colors
    )
    segment_gain_patches = [
        patch for patch in segment_gain_ax.patches if isinstance(patch, PathPatch)
    ]
    assert [patch.get_facecolor()[3] for patch in segment_gain_patches] == pytest.approx(
        [0.0] * len(expected_gain_regions)
    )
    assert [patch.get_edgecolor() for patch in segment_gain_patches] == pytest.approx(
        expected_gain_edge_colors
    )
    assert [patch.get_linewidth() for patch in segment_gain_patches] == pytest.approx(
        expected_gain_linewidths
    )
    assert [patch.get_edgecolor() for patch in shared_segment_patches] == pytest.approx(
        expected_gain_edge_colors
    )
    assert [
        patch.get_linewidth() for patch in shared_segment_patches
    ] == pytest.approx(expected_gain_linewidths)
    for outline_patches in (segment_gain_patches, shared_segment_patches):
        for patch, region_name in zip(
            outline_patches,
            expected_gain_regions,
            strict=True,
        ):
            x, y, width, height = expected_gain_bounds[region_name]
            outset = old_fig3_module.PANEL_G_SEGMENT_GAIN_OUTLINE_OUTSET
            vertices = patch.get_path().vertices
            expected_vertices = (
                x - outset,
                y - outset,
                x - outset,
                y + height + outset,
                x + width + outset,
                y + height + outset,
                x + width + outset,
                y - outset,
            )
            assert tuple(vertices.ravel()) == pytest.approx(expected_vertices)
    assert not segment_gain_ax.lines
    assert not {
        text.get_text()
        for text in segment_gain_ax.texts
        if text.get_text() in {"xgA", "xgB"}
    }
    for icon_ax in (independent_light_ax, shared_light_ax, segment_gain_ax):
        label_texts = {
            text.get_text(): text
            for text in icon_ax.texts
            if text.get_text() in expected_label_colors
        }
        assert set(label_texts) == set(expected_label_colors)
        for label, expected_color in expected_label_colors.items():
            assert to_rgba(label_texts[label].get_color()) == pytest.approx(
                expected_color
            )
    for icon_ax in (
        independent_dark_ax,
        independent_light_ax,
        shared_dark_ax,
        shared_light_ax,
    ):
        assert not icon_ax.lines
        assert not [
            patch
            for patch in icon_ax.patches
            if patch.get_label().startswith("_place_field_path_arrow")
        ]
        assert not [patch for patch in icon_ax.patches if isinstance(patch, Circle)]
        place_field_patches = [
            patch for patch in icon_ax.patches if type(patch) is Ellipse
        ]
        assert len(place_field_patches) >= 5
        assert max(patch.width for patch in place_field_patches) == pytest.approx(
            dims["corridor_w"]
            * 0.96
            * figure_2_module.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE
        )
        assert max(patch.height for patch in place_field_patches) == pytest.approx(
            0.72 * figure_2_module.PANEL_D_PLACE_FIELD_BLOB_SIZE_SCALE
        )
        field_center_y = dims["y1"] + 1.45
        top_patch = max(place_field_patches, key=lambda patch: patch.get_zorder())
        assert abs(top_patch.center[1] - field_center_y) == pytest.approx(
            min(abs(patch.center[1] - field_center_y) for patch in place_field_patches)
        )
        for first_patch in place_field_patches:
            for second_patch in place_field_patches:
                first_distance = abs(first_patch.center[1] - field_center_y)
                second_distance = abs(second_patch.center[1] - field_center_y)
                if first_distance < second_distance:
                    assert first_patch.get_zorder() >= second_patch.get_zorder()
    def _rounded_field_colors(icon_ax: object) -> set[tuple[float, float, float]]:
        return {
            tuple(round(float(component), 4) for component in patch.get_facecolor()[:3])
            for patch in icon_ax.patches
            if type(patch) is Ellipse
        }

    def _rounded_palette(colors: tuple[str, ...]) -> set[tuple[float, float, float]]:
        return {
            tuple(round(float(component), 4) for component in to_rgba(color)[:3])
            for color in colors
        }

    def _field_center_x_values(icon_ax: object) -> set[float]:
        return {
            round(float(patch.center[0]), 4)
            for patch in icon_ax.patches
            if type(patch) is Ellipse
        }

    independent_dark_field_colors = _rounded_field_colors(independent_dark_ax)
    independent_light_field_colors = _rounded_field_colors(independent_light_ax)
    shared_dark_field_colors = _rounded_field_colors(shared_dark_ax)
    shared_light_field_colors = _rounded_field_colors(shared_light_ax)
    assert independent_dark_field_colors != independent_light_field_colors
    assert independent_dark_field_colors == shared_dark_field_colors
    assert shared_dark_field_colors == shared_light_field_colors
    assert independent_dark_field_colors == _rounded_palette(
        figure_2_module.PANEL_D_DARK_PLACE_FIELD_COLORS
    )
    assert independent_light_field_colors == _rounded_palette(
        figure_2_module.PANEL_D_LIGHT_PLACE_FIELD_COLORS
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    figure_2_module._draw_panel_d_swap_schematic(
        ax,
        track_size=figure_2_module.PANEL_D_SCHEMATIC_TRACK_SIZE,
        model_name=figure_2_module.PANEL_C_SWAP_MODEL_NAME,
    )
    fig.canvas.draw()
    (
        panel_e_independent_train_ax,
        panel_e_independent_predict_ax,
        panel_e_shared_dark_ax,
        panel_e_segment_gain_ax,
        panel_e_shared_light_ax,
    ) = ax.child_axes
    panel_e_track_bounds = {
        "independent_train": _relative_bounds(ax, panel_e_independent_train_ax),
        "independent_predict": _relative_bounds(ax, panel_e_independent_predict_ax),
        "shared_dark": _relative_bounds(ax, panel_e_shared_dark_ax),
        "segment_gain": _relative_bounds(ax, panel_e_segment_gain_ax),
        "shared_light": _relative_bounds(ax, panel_e_shared_light_ax),
    }
    panel_e_model_label_xs = [
        text.get_position()[0]
        for text in ax.texts
        if text.get_fontweight() == "bold" and text.get_text().endswith("\nmodel")
    ]
    assert panel_e_model_label_xs == pytest.approx(panel_d_model_label_xs)
    expected_track_size = panel_d_track_bounds["independent_dark"][2:]
    for bounds in (*panel_d_track_bounds.values(), *panel_e_track_bounds.values()):
        assert bounds[2:] == pytest.approx(expected_track_size)
    assert _center_x(panel_d_track_bounds["independent_dark"]) == pytest.approx(
        old_fig3_module.PANEL_G_DARK_TRACK_CENTER_X
    )
    assert _center_x(panel_d_track_bounds["shared_dark"]) == pytest.approx(
        old_fig3_module.PANEL_G_DARK_TRACK_CENTER_X
    )
    assert _center_x(panel_d_track_bounds["independent_light"]) == pytest.approx(
        old_fig3_module.PANEL_G_LIGHT_TRACK_CENTER_X
    )
    assert _center_x(panel_d_track_bounds["shared_light"]) == pytest.approx(
        old_fig3_module.PANEL_G_LIGHT_TRACK_CENTER_X
    )
    assert _center_x(panel_d_track_bounds["segment_gain"]) == pytest.approx(
        old_fig3_module.PANEL_G_SEGMENT_MODULATION_TRACK_CENTER_X
    )
    assert _center_x(panel_e_track_bounds["independent_train"]) == pytest.approx(
        _center_x(panel_d_track_bounds["independent_dark"])
    )
    assert _center_x(panel_e_track_bounds["shared_dark"]) == pytest.approx(
        _center_x(panel_d_track_bounds["shared_dark"])
    )
    assert _center_x(panel_e_track_bounds["independent_predict"]) == pytest.approx(
        _center_x(panel_d_track_bounds["independent_light"])
    )
    assert _center_x(panel_e_track_bounds["shared_light"]) == pytest.approx(
        _center_x(panel_d_track_bounds["shared_light"])
    )
    assert _center_x(panel_e_track_bounds["segment_gain"]) == pytest.approx(
        _center_x(panel_d_track_bounds["segment_gain"])
    )
    expected_stim2_arm_colors = [
        to_rgba(
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim2"][
                "left_arm"
            ],
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        ),
        to_rgba(
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_COLORS["stim2"][
                "right_arm"
            ],
            figure_2_module.PANEL_D_INDEPENDENT_LIGHT_ARM_FILL_ALPHA,
        ),
    ]
    assert [
        patch.get_facecolor()
        for patch in panel_e_independent_train_ax.patches
        if isinstance(patch, Rectangle)
    ] == pytest.approx(expected_arm_colors)
    assert [
        patch.get_facecolor()
        for patch in panel_e_independent_predict_ax.patches
        if isinstance(patch, Rectangle)
    ] == pytest.approx(expected_stim2_arm_colors)
    assert [
        patch.get_facecolor()
        for patch in panel_e_shared_light_ax.patches
        if isinstance(patch, Rectangle) and patch.get_facecolor()[3] > 0.0
    ] == pytest.approx(expected_stim2_arm_colors)
    assert _rounded_field_colors(panel_e_independent_train_ax) == _rounded_palette(
        figure_2_module.PANEL_D_LIGHT_PLACE_FIELD_COLORS
    )
    assert _rounded_field_colors(panel_e_independent_predict_ax) == _rounded_palette(
        figure_2_module.PANEL_D_LIGHT_PLACE_FIELD_COLORS
    )
    assert _rounded_field_colors(panel_e_shared_dark_ax) == _rounded_palette(
        figure_2_module.PANEL_D_DARK_PLACE_FIELD_COLORS
    )
    assert _rounded_field_colors(panel_e_shared_light_ax) == _rounded_palette(
        figure_2_module.PANEL_D_DARK_PLACE_FIELD_COLORS
    )
    left_field_center_x = round(float((dims["x0"] + dims["x1"]) / 2.0), 4)
    right_field_center_x = round(float((dims["x4"] + dims["x5"]) / 2.0), 4)
    assert _field_center_x_values(panel_e_independent_train_ax) == {
        left_field_center_x
    }
    assert _field_center_x_values(panel_e_independent_predict_ax) == {
        right_field_center_x
    }
    assert _field_center_x_values(panel_e_shared_dark_ax) == {right_field_center_x}
    assert _field_center_x_values(panel_e_shared_light_ax) == {right_field_center_x}
    assert not [
        patch
        for icon_ax in (
            panel_e_independent_train_ax,
            panel_e_independent_predict_ax,
            panel_e_shared_dark_ax,
            panel_e_shared_light_ax,
        )
        for patch in icon_ax.patches
        if isinstance(patch, Circle)
    ]
    for icon_ax in (panel_e_segment_gain_ax, panel_e_shared_light_ax):
        segment_patches = [
            patch for patch in icon_ax.patches if isinstance(patch, PathPatch)
        ]
        assert [patch.get_edgecolor() for patch in segment_patches] == pytest.approx(
            expected_gain_edge_colors
        )
        assert [patch.get_linewidth() for patch in segment_patches] == pytest.approx(
            expected_gain_linewidths
        )
    right_arm_bounds = (
        dims["x4"],
        dims["y1"],
        dims["x5"] - dims["x4"],
        dims["y2"] - dims["y1"],
    )
    expected_panel_e_segment_bounds = (
        expected_gain_bounds["left_arm"],
        right_arm_bounds,
    )
    for icon_ax, expected_bounds in zip(
        (panel_e_segment_gain_ax, panel_e_shared_light_ax),
        expected_panel_e_segment_bounds,
        strict=True,
    ):
        segment_patch = next(
            patch for patch in icon_ax.patches if isinstance(patch, PathPatch)
        )
        x, y, width, height = expected_bounds
        outset = old_fig3_module.PANEL_G_SEGMENT_GAIN_OUTLINE_OUTSET
        expected_vertices = (
            x - outset,
            y - outset,
            x - outset,
            y + height + outset,
            x + width + outset,
            y + height + outset,
            x + width + outset,
            y - outset,
        )
        assert tuple(segment_patch.get_path().vertices.ravel()) == pytest.approx(
            expected_vertices
        )
    plt.close(fig)


def test_panel_d_example_rows_layout_draws_four_requested_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import v1ca1.paper_figures.old_fig3 as old_fig3_module

    field_calls = []

    def fake_draw_w_track_schematic(ax: object, **kwargs: object) -> None:
        return None

    def fake_panel_g_examples_y_max(examples: object) -> float:
        return 1.0

    def fake_plot_panel_g_example_field_axis(
        ax: object,
        example: dict[str, object],
        **kwargs: object,
    ) -> None:
        field_calls.append(
            (
                example["example_id"],
                kwargs["epoch_key"],
                kwargs.get("show_legend", False),
            )
        )

    monkeypatch.setattr(
        old_fig3_module,
        "draw_w_track_schematic",
        fake_draw_w_track_schematic,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "_panel_g_examples_y_max",
        fake_panel_g_examples_y_max,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "_plot_panel_g_example_field_axis",
        fake_plot_panel_g_example_field_axis,
    )

    examples = [
        {"example_id": example_id, "trajectory": "center_to_left"}
        for example_id in range(1, 5)
    ]
    fig, ax = plt.subplots()
    old_fig3_module._plot_panel_g_example_columns(
        ax,
        examples,
        layout="rows",
        column_gap=0.02,
    )

    assert [
        text.get_text()
        for child_ax in ax.child_axes
        for text in child_ax.texts
        if text.get_text().startswith("Example")
    ] == ["Example 1", "Example 2", "Example 3", "Example 4"]
    assert [call[:2] for call in field_calls] == [
        (1, "dark"),
        (1, "light"),
        (2, "dark"),
        (2, "light"),
        (3, "dark"),
        (3, "light"),
        (4, "dark"),
        (4, "light"),
    ]
    assert [call for call in field_calls if call[2]] == [(2, "light", True)]
    plt.close(fig)


def test_load_panel_glm_data_requests_all_configured_dark_light_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import v1ca1.paper_figures.old_fig3 as old_fig3_module

    calls = {}

    def fake_load_panel_h_swap_examples(**kwargs: object) -> list[object]:
        return []

    def fake_load_panel_g_dark_light_glm_examples(**kwargs: object) -> list[object]:
        calls["dark_light_kwargs"] = kwargs
        return []

    def fake_load_panel_h_swap_delta_table(**kwargs: object) -> str:
        return "swap-delta"

    monkeypatch.setattr(
        old_fig3_module,
        "load_panel_h_swap_examples",
        fake_load_panel_h_swap_examples,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "load_panel_g_dark_light_glm_examples",
        fake_load_panel_g_dark_light_glm_examples,
    )
    monkeypatch.setattr(
        old_fig3_module,
        "load_panel_h_swap_delta_table",
        fake_load_panel_h_swap_delta_table,
    )

    old_fig3_module.load_panel_glm_data(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        dark_light_requested_examples=PANEL_C_DARK_LIGHT_EXAMPLES,
    )

    assert calls["dark_light_kwargs"]["requested_examples"] == (
        PANEL_C_DARK_LIGHT_EXAMPLES
    )
    assert calls["dark_light_kwargs"]["example_count"] == len(
        PANEL_C_DARK_LIGHT_EXAMPLES
    )


def test_make_figure_2_wires_current_panel_loaders_and_plotters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import pandas as pd

    calls: dict[str, object] = {}
    real_plot_panel_b_dpp_overlap_with_schematic = (
        figure_2_module.plot_panel_b_dpp_overlap_with_schematic
    )

    def fake_load_panel_glm_data(**kwargs: object) -> dict[str, object]:
        calls["glm_kwargs"] = kwargs
        return {
            "dark_light_examples": ["model-example"],
            "swap_delta": "swap-delta",
            "swap_examples": ["swap-example"],
        }

    def fake_load_panel_a_example_data(**kwargs: object) -> dict[str, object]:
        calls.setdefault("panel_a_loader_kwargs", []).append(kwargs)
        return {
            "animal_name": kwargs["animal_name"],
            "unit_id": kwargs["unit_id"],
            "trajectories": kwargs["trajectories"],
        }

    def fake_plot_panel_a_examples_single_row(
        ax: object,
        examples: object,
    ) -> None:
        calls["panel_a_examples"] = examples

    def fake_load_panel_b_tuning_overlap_table(**kwargs: object) -> object:
        calls["panel_b_loader_kwargs"] = kwargs
        table = pd.DataFrame(
            {
                "similarity_dark": [0.10, 0.30, 0.50, 0.70, 0.90],
                "similarity_light": [0.18, 0.34, 0.50, 0.66, 0.82],
            }
        )
        calls["panel_b_loaded_table"] = table
        return table

    def fake_filter_panel_b_overlap_by_even_odd_stability(
        table: object,
        **kwargs: object,
    ) -> object:
        calls["panel_b_filter_input"] = table
        calls["panel_b_filter_kwargs"] = kwargs
        return table

    def fake_plot_panel_b_dpp_overlap_with_schematic(
        ax: object,
        table: object,
        **kwargs: object,
    ) -> None:
        calls["panel_b_table"] = table
        calls["panel_b_plot_kwargs"] = kwargs
        real_plot_panel_b_dpp_overlap_with_schematic(ax, table, **kwargs)

    def fake_load_panel_e_decoding_error_table(**kwargs: object) -> str:
        calls["panel_c_loader_kwargs"] = kwargs
        return "panel-c-decoding"

    def fake_plot_panel_c_cross_and_place_decoding(
        ax: object,
        table: object,
    ) -> None:
        calls["panel_c_table"] = table

    def fake_plot_panel_c_model_architecture_row(
        ax: object,
        examples: object,
    ) -> None:
        calls["panel_d_examples"] = examples

    def fake_plot_panel_d_compact_swap_delta(
        ax: object,
        swap_delta_table: object,
        swap_examples: object,
        **kwargs: object,
    ) -> None:
        calls["panel_e_delta"] = swap_delta_table
        calls["panel_e_examples"] = swap_examples
        calls["panel_e_kwargs"] = kwargs

    def fake_save_figure(
        figure: object,
        output_path: Path,
        dpi: int,
        **kwargs: object,
    ) -> Path:
        figure.canvas.draw()
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["panel_labels"] = [
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        calls["panel_label_display_positions"] = {
            text.get_text(): text.get_transform().transform(text.get_position())
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C", "D", "E"}
            and text.get_fontweight() == "bold"
        }
        calls["panel_label_vertical_alignments"] = {
            text.get_text(): text.get_va()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C", "D", "E"}
            and text.get_fontweight() == "bold"
        }
        calls["title_display_positions"] = {
            ax.get_title(): ax.title.get_transform().transform(ax.title.get_position())
            for ax in figure.axes
            if ax.get_title()
        }
        calls["titled_axis_bounds"] = {
            ax.get_title(): ax.get_position().bounds
            for ax in figure.axes
            if ax.get_title()
        }
        return output_path

    monkeypatch.setattr(
        figure_2_module,
        "load_panel_glm_data",
        fake_load_panel_glm_data,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_a_example_data",
        fake_load_panel_a_example_data,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_a_examples_single_row",
        fake_plot_panel_a_examples_single_row,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_b_tuning_overlap_table",
        fake_load_panel_b_tuning_overlap_table,
    )
    monkeypatch.setattr(
        figure_2_module,
        "filter_panel_b_overlap_by_even_odd_stability",
        fake_filter_panel_b_overlap_by_even_odd_stability,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_b_dpp_overlap_with_schematic",
        fake_plot_panel_b_dpp_overlap_with_schematic,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_e_decoding_error_table",
        fake_load_panel_e_decoding_error_table,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_c_cross_and_place_decoding",
        fake_plot_panel_c_cross_and_place_decoding,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_c_model_architecture_row",
        fake_plot_panel_c_model_architecture_row,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d_compact_swap_delta",
        fake_plot_panel_d_compact_swap_delta,
    )
    monkeypatch.setattr(figure_2_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_2.svg"
    saved_path = make_figure_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(FIGURE_3_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == ["A", "B", "C", "D", "E"]
    label_positions = calls["panel_label_display_positions"]
    assert label_positions["B"][0] == pytest.approx(label_positions["A"][0])
    assert label_positions["D"][0] == pytest.approx(label_positions["A"][0])
    assert label_positions["E"][0] == pytest.approx(label_positions["A"][0])
    title_positions = calls["title_display_positions"]
    assert calls["panel_label_vertical_alignments"]["B"] == "baseline"
    assert calls["panel_label_vertical_alignments"]["C"] == "baseline"
    panel_bc_title_y = title_positions["Dark and light DPP coding"][1]
    assert label_positions["B"][1] == pytest.approx(panel_bc_title_y)
    assert label_positions["C"][1] == pytest.approx(panel_bc_title_y)
    assert title_positions["Dark and light decoding comparison"][1] == pytest.approx(
        panel_bc_title_y
    )
    axis_bounds = calls["titled_axis_bounds"]
    panel_a_bounds = axis_bounds["Example DPP cells in dark and light"]
    assert panel_a_bounds[0] == pytest.approx(
        figure_2_module.PANEL_A_HORIZONTAL_AXIS_BOUNDS[0]
    )
    assert panel_a_bounds[2] == pytest.approx(
        figure_2_module.PANEL_A_HORIZONTAL_AXIS_BOUNDS[1]
    )
    panel_b_bounds = axis_bounds["Dark and light DPP coding"]
    panel_c_bounds = axis_bounds["Dark and light decoding comparison"]
    assert panel_b_bounds[2] / panel_c_bounds[2] == pytest.approx(2.0)
    assert panel_b_bounds[0] + panel_b_bounds[2] < panel_c_bounds[0]
    assert panel_b_bounds[3] == pytest.approx(panel_a_bounds[3])
    assert panel_c_bounds[3] == pytest.approx(panel_a_bounds[3])
    panel_d_bounds = axis_bounds["Two models that relate dark and light activity"]
    panel_e_bounds = axis_bounds["Predicting activity in held-out light epoch"]
    assert panel_d_bounds[3] == pytest.approx(panel_a_bounds[3])
    assert panel_e_bounds[3] == pytest.approx(panel_a_bounds[3])

    assert len(calls["panel_a_loader_kwargs"]) == 4
    assert [
        (
            kwargs["animal_name"],
            kwargs["date"],
            kwargs["region"],
            kwargs["unit_id"],
            kwargs["trajectories"],
        )
        for kwargs in calls["panel_a_loader_kwargs"]
    ] == [
        ("L14", "20240611", "v1", 34, ("center_to_left", "right_to_center")),
        ("L15", "20241121", "v1", 473, ("center_to_right", "left_to_center")),
        ("L12", "20240421", "v1", 37, ("center_to_right", "left_to_center")),
        ("L14", "20240611", "v1", 30, ("center_to_left", "right_to_center")),
    ]
    assert calls["panel_a_examples"] == [
        {
            "animal_name": "L14",
            "unit_id": 34,
            "trajectories": ("center_to_left", "right_to_center"),
        },
        {
            "animal_name": "L15",
            "unit_id": 473,
            "trajectories": ("center_to_right", "left_to_center"),
        },
        {
            "animal_name": "L12",
            "unit_id": 37,
            "trajectories": ("center_to_right", "left_to_center"),
        },
        {
            "animal_name": "L14",
            "unit_id": 30,
            "trajectories": ("center_to_left", "right_to_center"),
        },
    ]

    expected_loader_kwargs = {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "v1",
        "light_epoch": None,
        "dark_epoch": None,
    }
    assert calls["panel_b_loader_kwargs"] == expected_loader_kwargs
    assert calls["panel_b_filter_input"] is calls["panel_b_loaded_table"]
    assert calls["panel_b_filter_kwargs"] == {
        **expected_loader_kwargs,
        "min_stability_correlation": pytest.approx(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    }
    assert calls["panel_b_table"] is calls["panel_b_loaded_table"]
    assert calls["panel_b_plot_kwargs"] == {
        "example": {
            "animal_name": "L14",
            "unit_id": 34,
            "trajectories": ("center_to_left", "right_to_center"),
        },
        "low_threshold": PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
        "high_threshold": PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
    }

    assert calls["panel_c_loader_kwargs"] == expected_loader_kwargs
    assert calls["panel_c_table"] == "panel-c-decoding"
    assert calls["panel_d_examples"] == ["model-example"]
    assert calls["panel_e_delta"] == "swap-delta"
    assert calls["panel_e_examples"] == ["swap-example"]
    assert calls["panel_e_kwargs"]["model_name"] == PANEL_C_SWAP_MODEL_NAME
    assert set(calls["panel_e_kwargs"]["model_colors"]) == {
        "visual",
        PANEL_C_SWAP_MODEL_NAME,
    }
    assert set(calls["panel_e_kwargs"]["model_labels"]) == {PANEL_C_SWAP_MODEL_NAME}
    assert calls["glm_kwargs"]["region"] == "v1"
    assert calls["glm_kwargs"]["dark_light_requested_examples"] == (
        PANEL_C_DARK_LIGHT_EXAMPLES
    )
    assert calls["glm_kwargs"][
        "swap_delta_min_tuning_stability_correlation"
    ] == pytest.approx(PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION)


def test_make_figure_2_2_splits_decoding_panel_and_swaps_c_d_locations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import pandas as pd

    calls: dict[str, object] = {}
    real_plot_panel_b_dpp_overlap_with_schematic = (
        figure_2_module.plot_panel_b_dpp_overlap_with_schematic
    )

    def fake_load_panel_glm_data(**kwargs: object) -> dict[str, object]:
        calls["glm_kwargs"] = kwargs
        return {
            "swap_delta": "swap-delta",
            "swap_examples": ["swap-example"],
        }

    def fake_load_panel_a_example_data(**kwargs: object) -> dict[str, object]:
        calls.setdefault("panel_a_loader_kwargs", []).append(kwargs)
        return {
            "animal_name": kwargs["animal_name"],
            "unit_id": kwargs["unit_id"],
            "trajectories": kwargs["trajectories"],
        }

    def fake_plot_panel_a2_examples_single_row(
        ax: object,
        examples: object,
    ) -> None:
        calls["panel_a_examples"] = examples

    def fake_load_panel_b_tuning_overlap_table(**kwargs: object) -> object:
        calls["panel_b_loader_kwargs"] = kwargs
        return pd.DataFrame(
            {
                "similarity_dark": [0.10, 0.30, 0.50, 0.70, 0.90],
                "similarity_light": [0.18, 0.34, 0.50, 0.66, 0.82],
            }
        )

    def fake_filter_panel_b_overlap_by_even_odd_stability(
        table: object,
        **kwargs: object,
    ) -> object:
        calls["panel_b_filter_input"] = table
        calls["panel_b_filter_kwargs"] = kwargs
        return table

    def fake_plot_panel_b_dpp_overlap_with_schematic(
        ax: object,
        table: object,
        **kwargs: object,
    ) -> None:
        calls["panel_b_table"] = table
        calls["panel_b_plot_kwargs"] = kwargs
        real_plot_panel_b_dpp_overlap_with_schematic(ax, table, **kwargs)

    def fake_load_panel_e_decoding_error_table(**kwargs: object) -> str:
        calls["panel_c_loader_kwargs"] = kwargs
        return "panel-c-decoding"

    def fake_plot_panel_d2_swap_results_panel(
        ax: object,
        swap_delta_table: object,
        swap_examples: object,
        **kwargs: object,
    ) -> None:
        calls["swap_results_axis"] = ax
        calls["swap_results_delta"] = swap_delta_table
        calls["swap_results_examples"] = swap_examples
        calls["swap_results_kwargs"] = kwargs

    def fake_plot_panel_d2_architecture_panel(
        ax: object,
    ) -> None:
        calls["architecture_axis"] = ax

    def fake_plot_panel_e2_decoding_panel(
        ax: object,
        decoding_error_table: object,
    ) -> None:
        calls["decoding_axis"] = ax
        calls["decoding_table"] = decoding_error_table

    def fake_save_figure(
        figure: object,
        output_path: Path,
        dpi: int,
        **kwargs: object,
    ) -> Path:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        calls["figsize"] = figure.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["titled_axis_bounds"] = {
            ax.get_title(): ax.get_position().bounds
            for ax in figure.axes
            if ax.get_title()
        }
        calls["title_display_tops"] = {
            ax.get_title(): ax.title.get_window_extent(renderer).y1
            for ax in figure.axes
            if ax.get_title()
        }
        calls["panel_label_display_tops"] = {
            text.get_text(): text.get_window_extent(renderer).y1
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"D", "E"}
            and text.get_fontweight() == "bold"
            and text.get_fontsize() == pytest.approx(8.0)
        }
        panel_b_axis = next(
            ax for ax in figure.axes if ax.get_title() == "Dark and light DPP coding"
        )
        scatter_parent = next(
            child_ax
            for child_ax in panel_b_axis.child_axes
            if len(child_ax.child_axes) == 3
        )
        scatter_axis = next(
            child_ax
            for child_ax in scatter_parent.child_axes
            if child_ax.get_xlabel() == "Dark DPPI"
        )
        top_histogram_axis = next(
            child_ax
            for child_ax in scatter_parent.child_axes
            if child_ax.get_ylabel() == "Frac."
        )
        scatter_ylabel_bbox = scatter_axis.yaxis.label.get_window_extent(renderer)
        top_ylabel_bbox = top_histogram_axis.yaxis.label.get_window_extent(renderer)
        calls["panel_b_ylabel_center_x"] = (
            scatter_ylabel_bbox.x0 + scatter_ylabel_bbox.width / 2.0,
            top_ylabel_bbox.x0 + top_ylabel_bbox.width / 2.0,
        )
        return output_path

    monkeypatch.setattr(
        figure_2_module,
        "load_panel_glm_data",
        fake_load_panel_glm_data,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_a_example_data",
        fake_load_panel_a_example_data,
    )
    monkeypatch.setattr(
        figure_2_2_module,
        "plot_panel_a2_examples_single_row",
        fake_plot_panel_a2_examples_single_row,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_b_tuning_overlap_table",
        fake_load_panel_b_tuning_overlap_table,
    )
    monkeypatch.setattr(
        figure_2_module,
        "filter_panel_b_overlap_by_even_odd_stability",
        fake_filter_panel_b_overlap_by_even_odd_stability,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_b_dpp_overlap_with_schematic",
        fake_plot_panel_b_dpp_overlap_with_schematic,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_e_decoding_error_table",
        fake_load_panel_e_decoding_error_table,
    )
    monkeypatch.setattr(
        figure_2_2_module,
        "plot_panel_d2_swap_results_panel",
        fake_plot_panel_d2_swap_results_panel,
    )
    monkeypatch.setattr(
        figure_2_2_module,
        "plot_panel_d2_architecture_panel",
        fake_plot_panel_d2_architecture_panel,
    )
    monkeypatch.setattr(
        figure_2_2_module,
        "plot_panel_e2_decoding_panel",
        fake_plot_panel_e2_decoding_panel,
    )
    monkeypatch.setattr(figure_2_2_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_2_2.svg"
    saved_path = figure_2_2_module.make_figure_2_2(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        light_epoch=None,
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(
        figure_2_2_module.DEFAULT_FIGURE_WIDTH_MM / 25.4
    )
    assert calls["output_path"] == output_path
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_b_plot_kwargs"] == {
        "example": calls["panel_a_examples"][0],
        "low_threshold": PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
        "high_threshold": PANEL_B_HIGH_DARK_TUNING_CORRELATION_THRESHOLD,
        "show_grouped": False,
        "show_scatter_linear_fit": True,
        "show_scatter_r2": True,
        "scatter_equal_aspect": True,
    }
    assert calls["panel_b_ylabel_center_x"][1] == pytest.approx(
        calls["panel_b_ylabel_center_x"][0],
        abs=0.5,
    )
    assert calls["swap_results_delta"] == "swap-delta"
    assert calls["swap_results_examples"] == ["swap-example"]
    assert "architecture_axis" in calls
    assert calls["decoding_table"] == "panel-c-decoding"

    axis_bounds = calls["titled_axis_bounds"]
    panel_b_bounds = axis_bounds["Dark and light DPP coding"]
    panel_c_bounds = axis_bounds["Two models that relate dark and light activity"]
    panel_d_bounds = axis_bounds["Dark and light cue-swap prediction comparison"]
    panel_e_bounds = axis_bounds["Dark and light decoding comparison"]
    assert panel_b_bounds[2] / panel_c_bounds[2] == pytest.approx(
        1.0,
        rel=0.02,
    )
    assert panel_b_bounds[0] + panel_b_bounds[2] < panel_c_bounds[0]
    assert panel_d_bounds[1] < panel_c_bounds[1]
    assert panel_d_bounds[0] + panel_d_bounds[2] < panel_e_bounds[0]
    assert panel_d_bounds[1] == pytest.approx(panel_e_bounds[1])
    assert panel_d_bounds[2] == pytest.approx(panel_e_bounds[2])
    panel_de_header_tops = [
        calls["panel_label_display_tops"]["D"],
        calls["panel_label_display_tops"]["E"],
        calls["title_display_tops"][
            "Dark and light cue-swap prediction comparison"
        ],
        calls["title_display_tops"]["Dark and light decoding comparison"],
    ]
    assert panel_de_header_tops == pytest.approx(
        [panel_de_header_tops[0]] * len(panel_de_header_tops),
        abs=0.5,
    )
