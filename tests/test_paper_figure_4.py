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
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(FIGURE_2_HEIGHT_MM)
    assert DEFAULT_REGIONS == ("v1",)
    assert PANEL_A_TO_GH_HEIGHT_RATIOS == pytest.approx((1.0, 1.0))


def test_make_figure_4_uses_figure_2_canvas_and_moved_panel_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_load_panel_a_example_data(**kwargs: object) -> dict[str, object]:
        calls["panel_a_kwargs"] = kwargs
        return {"example": "panel-a"}

    def fake_load_panel_glm_data(**kwargs: object) -> dict[str, object]:
        calls["glm_kwargs"] = kwargs
        return {
            "dark_light_examples": ["panel-b"],
            "swap_delta": "panel-c-delta",
            "swap_examples": ["panel-c-example"],
        }

    def fake_plot_panel_a_example(ax: object, example: object) -> None:
        calls["panel_a_example"] = example

    def fake_plot_panel_g_model_architecture(ax: object, examples: object) -> None:
        calls["panel_b_examples"] = examples

    def fake_plot_panel_h_swap_delta(
        ax: object,
        swap_delta_table: object,
        swap_examples: object,
    ) -> None:
        calls["panel_c_delta"] = swap_delta_table
        calls["panel_c_examples"] = swap_examples

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
        "load_panel_a_example_data",
        fake_load_panel_a_example_data,
    )
    monkeypatch.setattr(
        figure_4_module,
        "load_panel_glm_data",
        fake_load_panel_glm_data,
    )
    monkeypatch.setattr(
        figure_4_module,
        "plot_panel_a_example",
        fake_plot_panel_a_example,
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
    assert calls["figsize"][1] == pytest.approx(FIGURE_2_HEIGHT_MM / 25.4)
    assert calls["panel_labels"] == ["A", "B", "C"]
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_a_kwargs"]["panel_example_cache_dir"] == tmp_path / "cache"
    assert calls["panel_a_kwargs"]["position_bin_count"] == 100
    assert calls["glm_kwargs"]["region"] == "v1"
    assert calls["panel_a_example"] == {"example": "panel-a"}
    assert calls["panel_b_examples"] == ["panel-b"]
    assert calls["panel_c_delta"] == "panel-c-delta"
    assert calls["panel_c_examples"] == ["panel-c-example"]
