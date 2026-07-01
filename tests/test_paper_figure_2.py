from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_2 as figure_2_module
import v1ca1.paper_figures.figure_2_common as figure_2_common_module
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
    assert len(PANEL_C_DARK_LIGHT_EXAMPLES) == 2
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
    expected_kwargs = [
        {
            "title": None,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        },
        {
            "title": None,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        },
        {
            "title": None,
            "show_correlation": False,
            "similarity_annotation": "dppi",
        },
        {
            "title": None,
            "show_correlation": False,
            "similarity_annotation": "dppi",
            "y_max": 85.0,
        },
    ]
    assert [kwargs for _example, kwargs in calls] == expected_kwargs
    assert [text.get_text() for text in ax.texts] == []
    assert len(ax.child_axes) == 4
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


def test_make_figure_2_wires_current_panel_loaders_and_plotters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

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
        }

    def fake_plot_panel_a_examples_single_row(
        ax: object,
        examples: object,
    ) -> None:
        calls["panel_a_examples"] = examples

    def fake_load_panel_b_tuning_overlap_table(**kwargs: object) -> str:
        calls["panel_b_loader_kwargs"] = kwargs
        return "panel-b-overlap"

    def fake_filter_panel_b_overlap_by_even_odd_stability(
        table: object,
        **kwargs: object,
    ) -> str:
        calls["panel_b_filter_input"] = table
        calls["panel_b_filter_kwargs"] = kwargs
        return "filtered-panel-b-overlap"

    def fake_plot_panel_b_dpp_overlap_with_schematic(
        ax: object,
        table: object,
        **kwargs: object,
    ) -> None:
        calls["panel_b_table"] = table
        calls["panel_b_plot_kwargs"] = kwargs

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
    assert "bbox_inches" in calls["save_kwargs"]
    assert calls["panel_labels"] == ["A", "B", "C", "D", "E"]

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
        {"animal_name": "L14", "unit_id": 34},
        {"animal_name": "L15", "unit_id": 473},
        {"animal_name": "L12", "unit_id": 37},
        {"animal_name": "L14", "unit_id": 30},
    ]

    expected_loader_kwargs = {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "region": "v1",
        "light_epoch": None,
        "dark_epoch": None,
    }
    assert calls["panel_b_loader_kwargs"] == expected_loader_kwargs
    assert calls["panel_b_filter_input"] == "panel-b-overlap"
    assert calls["panel_b_filter_kwargs"] == {
        **expected_loader_kwargs,
        "min_stability_correlation": pytest.approx(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    }
    assert calls["panel_b_table"] == "filtered-panel-b-overlap"
    assert calls["panel_b_plot_kwargs"] == {
        "example": {"animal_name": "L14", "unit_id": 34},
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
    assert calls["glm_kwargs"][
        "swap_delta_min_tuning_stability_correlation"
    ] == pytest.approx(PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION)
