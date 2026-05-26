from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_3 as figure_3_module
import v1ca1.paper_figures.supplementary_figure_4 as supp_figure_4_module
from v1ca1.paper_figures.supplementary_figure_4 import (
    DEFAULT_ANIMAL_ROW_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_NAME,
    LETTER_HORIZONTAL_MARGIN_IN,
    LETTER_PAPER_WIDTH_IN,
    get_figure_height_mm,
    group_datasets_by_animal,
    make_supplementary_figure_4,
    parse_arguments,
)


def test_default_cli_matches_letter_width_with_one_inch_margins() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "supplementary_figure_4"
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        (LETTER_PAPER_WIDTH_IN - 2.0 * LETTER_HORIZONTAL_MARGIN_IN) * 25.4
    )
    assert args.output_dir == figure_3_module.DEFAULT_OUTPUT_DIR
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert args.output_format == figure_3_module.DEFAULT_OUTPUT_FORMAT
    assert args.region == figure_3_module.DEFAULT_REGIONS[0]
    assert args.dataset is None
    assert args.dark_epoch is None
    assert get_figure_height_mm(0) == pytest.approx(DEFAULT_ANIMAL_ROW_HEIGHT_MM)
    assert get_figure_height_mm(3) == pytest.approx(3 * DEFAULT_ANIMAL_ROW_HEIGHT_MM)


def test_group_datasets_by_animal_preserves_input_order() -> None:
    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
        ("L14", "20240612", "08_r4"),
    ]

    grouped = group_datasets_by_animal(datasets)

    assert list(grouped) == ["L14", "L15"]
    assert grouped["L14"] == [
        ("L14", "20240611", "08_r4"),
        ("L14", "20240612", "08_r4"),
    ]
    assert grouped["L15"] == [("L15", "20241121", "10_r5")]


def test_make_supplementary_figure_4_plots_figure_4c_histograms_per_animal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}
    load_calls = []
    plot_tables = []

    def fake_load_panel_h_swap_delta_table(**kwargs: object):
        load_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "trajectory": ["center_to_left"],
                "light_train_epoch": ["02_r1"],
                "light_test_epoch": ["06_r3"],
                "delta_ll_bits_per_spike": [0.2],
            }
        )

    def fake_plot_figure_4c_histogram_grid(ax, table):
        plot_tables.append(table)
        ax.text(0.5, 0.5, "4C")

    def fake_save_figure(figure, output_path: Path, dpi: int, **kwargs: object):
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
        supp_figure_4_module,
        "load_panel_h_swap_delta_table",
        fake_load_panel_h_swap_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_4_module,
        "plot_figure_4c_histogram_grid",
        fake_plot_figure_4c_histogram_grid,
    )
    monkeypatch.setattr(supp_figure_4_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_4.svg"
    datasets = [("L14", "20240611", "08_r4"), ("L15", "20241121", "10_r5")]
    saved_path = make_supplementary_figure_4(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        region="v1",
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(
        2 * DEFAULT_ANIMAL_ROW_HEIGHT_MM / 25.4
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == ["A"]
    assert [call["datasets"] for call in load_calls] == [
        [datasets[0]],
        [datasets[1]],
    ]
    assert all(call["region"] == "v1" for call in load_calls)
    assert len(plot_tables) == 2
