from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_3 as figure_3_module
import v1ca1.paper_figures.supplementary_figure_3 as supp_figure_3_module
from v1ca1.paper_figures.supplementary_figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_NAME,
    group_datasets_by_animal,
    make_supplementary_figure_3,
    parse_arguments,
    plot_panel_h_animal_histogram_rows,
)


def test_default_cli_matches_figure_3_size_and_region() -> None:
    args = parse_arguments([])

    assert DEFAULT_OUTPUT_NAME == "supplementary_figure_3"
    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        figure_3_module.DEFAULT_FIGURE_WIDTH_MM
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        figure_3_module.DEFAULT_FIGURE_HEIGHT_MM
    )
    assert args.region == figure_3_module.DEFAULT_REGIONS[0]
    assert args.dark_epoch is None


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


def test_plot_panel_h_animal_histogram_rows_uses_one_row_per_animal() -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    rows = []
    for animal_name in ("L14", "L15"):
        for trajectory in figure_3_module.PANEL_H_DELTA_TRAJECTORIES:
            for unit, value in enumerate((-0.2, 0.1, 0.3), start=1):
                rows.append(
                    {
                        "animal_name": animal_name,
                        "date": "20240611" if animal_name == "L14" else "20241121",
                        "unit": unit,
                        "trajectory": trajectory,
                        "light_train_epoch": figure_3_module.PANEL_H_TRAIN_LIGHT_EPOCH,
                        "light_test_epoch": figure_3_module.PANEL_H_HELDOUT_LIGHT_EPOCH,
                        "delta_ll_bits_per_spike": value,
                    }
                )
    swap_delta_table = pandas.DataFrame(rows)

    fig, ax = plt.subplots()
    row_axes = plot_panel_h_animal_histogram_rows(
        ax,
        swap_delta_table,
        datasets=datasets,
    )

    assert len(row_axes) == 2
    assert [text.get_text() for text in ax.texts[:2]] == [
        "L14\n20240611",
        "L15\n20241121",
    ]
    assert row_axes[0].get_position().y0 > row_axes[1].get_position().y0
    assert all(len(row_ax.child_axes) == 8 for row_ax in row_axes)
    first_row_hist_axes = row_axes[0].child_axes[1::2]
    assert len(first_row_hist_axes) == len(figure_3_module.PANEL_H_DELTA_TRAJECTORIES)
    assert all(len(hist_ax.patches) > 0 for hist_ax in first_row_hist_axes)
    assert all(hist_ax.lines[0].get_linestyle() == "--" for hist_ax in first_row_hist_axes)
    assert all(
        any("% >0\nmed." in text.get_text() for text in hist_ax.texts)
        for hist_ax in first_row_hist_axes
    )
    assert all(
        "n = 3 cells\n1 animal" in [text.get_text() for text in hist_ax.texts]
        for hist_ax in first_row_hist_axes
    )
    assert all(not hist_ax.get_xticklabels() for hist_ax in first_row_hist_axes)
    bottom_row_hist_axes = row_axes[1].child_axes[1::2]
    assert any(label.get_text() for label in bottom_row_hist_axes[0].get_xticklabels())
    plt.close(fig)


def test_make_supplementary_figure_3_uses_figure_3_canvas_and_panel_a(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_load_panel_h_swap_delta_table(**kwargs: object):
        calls["load_kwargs"] = kwargs
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "unit": [1],
                "trajectory": [figure_3_module.PANEL_H_DELTA_TRAJECTORIES[0]],
                "light_train_epoch": [figure_3_module.PANEL_H_TRAIN_LIGHT_EPOCH],
                "light_test_epoch": [figure_3_module.PANEL_H_HELDOUT_LIGHT_EPOCH],
                "delta_ll_bits_per_spike": [0.2],
            }
        )

    def fake_save_figure(figure, output_path: Path, dpi: int):
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
        supp_figure_3_module,
        "load_panel_h_swap_delta_table",
        fake_load_panel_h_swap_delta_table,
    )
    monkeypatch.setattr(supp_figure_3_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_3.svg"
    saved_path = make_supplementary_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        region="v1",
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["A"]
    assert calls["load_kwargs"]["region"] == "v1"
    assert calls["load_kwargs"]["light_epoch_pairs"] == (
        (
            figure_3_module.PANEL_H_TRAIN_LIGHT_EPOCH,
            figure_3_module.PANEL_H_HELDOUT_LIGHT_EPOCH,
        ),
    )
