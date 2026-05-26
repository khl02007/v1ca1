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
    assert args.light_epoch is None
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


def test_make_supplementary_figure_3_plots_per_animal_figure_3def(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pandas = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}
    similarity_calls = []
    encoding_calls = []
    decoding_calls = []

    def fake_load_panel_d_similarity_table(**kwargs: object):
        similarity_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "unit": [1],
                "comparison_label": ["left_turn"],
                "epoch_type": ["light"],
                "similarity": [0.5],
            }
        )

    def fake_load_panel_e_encoding_delta_table(**kwargs: object):
        encoding_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "unit": [1],
                "epoch_type": ["light"],
                "delta_bits_tp_vs_place": [0.2],
            }
        )

    def fake_load_panel_f_decoding_error_table(**kwargs: object):
        decoding_calls.append(kwargs)
        return pandas.DataFrame(
            {
                "animal_name": ["L14"],
                "date": ["20240611"],
                "epoch_type": ["light"],
                "epoch": ["02_r1"],
                "analysis": ["place"],
                "comparison": ["place"],
                "comparison_label": ["Place"],
                "q25_error": [0.01],
                "median_error": [0.02],
                "q75_error": [0.03],
                "n_samples": [20],
            }
        )

    def fake_plot_panel_d_similarity(ax, table):
        calls.setdefault("similarity_tables", []).append(table)
        ax.text(0.5, 0.5, "D")

    def fake_plot_panel_e_encoding_delta_histogram(ax, table):
        calls.setdefault("encoding_tables", []).append(table)
        ax.text(0.5, 0.5, "E")

    def fake_plot_panel_f_decoding_error(ax, table):
        calls.setdefault("decoding_tables", []).append(table)
        ax.text(0.5, 0.5, "F")

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
        "load_panel_d_similarity_table",
        fake_load_panel_d_similarity_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_e_encoding_delta_table",
        fake_load_panel_e_encoding_delta_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "load_panel_f_decoding_error_table",
        fake_load_panel_f_decoding_error_table,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_d_similarity",
        fake_plot_panel_d_similarity,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_e_encoding_delta_histogram",
        fake_plot_panel_e_encoding_delta_histogram,
    )
    monkeypatch.setattr(
        supp_figure_3_module,
        "plot_panel_f_decoding_error",
        fake_plot_panel_f_decoding_error,
    )
    monkeypatch.setattr(supp_figure_3_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_3.svg"
    datasets = [("L14", "20240611", "08_r4"), ("L15", "20241121", "10_r5")]
    saved_path = make_supplementary_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        region="v1",
        light_epoch=None,
        dark_epoch=None,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"][0] == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(DEFAULT_FIGURE_HEIGHT_MM / 25.4)
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["A", "B", "C"]
    assert [call["datasets"] for call in similarity_calls] == [[datasets[0]], [datasets[1]]]
    assert [call["datasets"] for call in encoding_calls] == [[datasets[0]], [datasets[1]]]
    assert [call["datasets"] for call in decoding_calls] == [[datasets[0]], [datasets[1]]]
    assert all(call["region"] == "v1" for call in similarity_calls)
    assert calls["similarity_tables"]
    assert calls["encoding_tables"]
    assert calls["decoding_tables"]
