from __future__ import annotations

from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_1 as figure_1_module
import v1ca1.paper_figures.supplementary_figure_1 as supp_figure_1_module
from v1ca1.paper_figures.supplementary_figure_1 import (
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    MODEL_COMPARISON_GRID_WSPACE,
    MODEL_COMPARISON_PANEL_C_SHIFT_PT,
    MODEL_COMPARISON_PANEL_D_SHIFT_PT,
    build_output_path,
    format_stability_summary,
    make_supplementary_figure_1,
    parse_arguments,
    plot_decoding_error_dataset_stack_panel,
    plot_dataset_stack_panel,
    plot_pooled_stability_panel,
    plot_stability_dataset_rows_panel,
    shift_model_comparison_columns,
)


def test_build_output_path_uses_requested_format() -> None:
    assert build_output_path(Path("paper_figures"), "supplementary_figure_1", "svg") == Path(
        "paper_figures/supplementary_figure_1.svg"
    )

    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(Path("paper_figures"), "supplementary_figure_1", "jpg")


def test_default_cli_matches_supplementary_figure_format() -> None:
    args = parse_arguments([])

    assert DEFAULT_FIGURE_WIDTH_MM == figure_1_module.DEFAULT_FIGURE_WIDTH_MM
    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.output_name == DEFAULT_OUTPUT_NAME
    assert args.asset_dir == DEFAULT_ASSET_DIR
    assert args.encoding_place_bin_size_cm == pytest.approx(
        figure_1_module.ENCODING_COMPARISON_PLACE_BIN_SIZE_CM
    )
    assert DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM == pytest.approx(40.0)
    assert MODEL_COMPARISON_GRID_WSPACE == pytest.approx(-0.10)
    assert MODEL_COMPARISON_PANEL_C_SHIFT_PT == pytest.approx(-10.0)
    assert MODEL_COMPARISON_PANEL_D_SHIFT_PT == pytest.approx(-33.0)
    assert not hasattr(args, "region")
    assert not hasattr(args, "panel_heatmap_cache_dir")
    assert not hasattr(args, "refresh_panel_heatmap_cache")


def test_plot_dataset_stack_panel_arranges_one_axis_per_dataset() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    plotted = []

    def plot_dataset(ax, dataset):
        plotted.append(dataset)
        ax.plot([0.0, 1.0], [0.0, 1.0])

    fig, ax = plt.subplots()
    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=datasets,
        plot_dataset=plot_dataset,
    )

    assert plotted == datasets
    assert len(row_axes) == 2
    assert len(ax.child_axes) == 2
    assert [text.get_text() for text in ax.texts] == [
        "L14\n20240611\n08_r4",
        "L15\n20241121\n10_r5",
    ]
    assert row_axes[0].get_position().y0 > row_axes[1].get_position().y0
    plt.close(fig)


def test_shift_model_comparison_columns_moves_later_columns_left() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=1, ncols=3)
    original_positions = [ax.get_position().x0 for ax in axes]

    shift_model_comparison_columns(fig, axes)

    shifted_positions = [ax.get_position().x0 for ax in axes]
    assert shifted_positions[0] == pytest.approx(original_positions[0])
    assert shifted_positions[1] < original_positions[1]
    assert shifted_positions[2] < original_positions[2]
    assert (shifted_positions[2] - original_positions[2]) < (
        shifted_positions[1] - original_positions[1]
    )
    plt.close(fig)


def test_plot_dataset_stack_panel_can_hide_repeated_dataset_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    row_axes = plot_dataset_stack_panel(
        ax,
        datasets=[("L14", "20240611", "08_r4")],
        plot_dataset=lambda row_ax, _dataset: row_ax.plot([0.0, 1.0], [0.0, 1.0]),
        show_dataset_labels=False,
    )

    assert len(row_axes) == 1
    assert len(ax.texts) == 0
    assert row_axes[0].get_position().x0 < ax.get_position().x0 + 0.05
    plt.close(fig)


def test_format_stability_summary_reports_median_and_fraction_above_half() -> None:
    assert format_stability_summary("v1", [0.1, 0.8]) == "V1 med 0.45, >0.5 50%"


def test_plot_stability_dataset_rows_panel_uses_one_horizontal_row_per_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    loaded = []
    rows = []
    for trajectory_type in figure_1_module.TRAJECTORY_TYPES:
        for region in figure_1_module.STABILITY_REGIONS:
            rows.extend(
                {
                    "trajectory_type": trajectory_type,
                    "region": region,
                    "stability_correlation": value,
                }
                for value in (0.1, 0.8)
            )

    def fake_load_dark_epoch_stability_table(**kwargs):
        loaded.append(kwargs["datasets"][0])
        return pd.DataFrame(rows)

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_dark_epoch_stability_table",
        fake_load_dark_epoch_stability_table,
    )

    fig, ax = plt.subplots()
    row_axes = plot_stability_dataset_rows_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=datasets,
    )

    assert loaded == datasets
    assert len(row_axes) == 2
    assert [text.get_text() for text in ax.texts] == [
        "L14\n20240611\n08_r4",
        "L15\n20241121\n10_r5",
    ]
    assert row_axes[0].get_position().y0 > row_axes[1].get_position().y0
    assert len(row_axes[0].child_axes) == 2 * len(figure_1_module.TRAJECTORY_TYPES)
    first_hist_axis = row_axes[0].child_axes[1]
    first_hist_text = [text.get_text() for text in first_hist_axis.texts]
    assert "V1 med 0.45, >0.5 50%" in first_hist_text
    assert "CA1 med 0.45, >0.5 50%" in first_hist_text
    plt.close(fig)


def test_plot_pooled_stability_panel_uses_all_datasets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("L14", "20240611", "08_r4"),
        ("L15", "20241121", "10_r5"),
    ]
    calls: dict[str, object] = {}
    table = pd.DataFrame(
        {
            "trajectory_type": ["center_to_left"],
            "region": ["v1"],
            "stability_correlation": [0.5],
        }
    )

    def fake_load_dark_epoch_stability_table(**kwargs):
        calls["load_kwargs"] = kwargs
        return table

    def fake_plot_stability_panel(ax, stability_table):
        calls["plot_table"] = stability_table
        ax.text(0.5, 0.5, "pooled")

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_dark_epoch_stability_table",
        fake_load_dark_epoch_stability_table,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_stability_panel",
        fake_plot_stability_panel,
    )

    fig, ax = plt.subplots()
    plot_pooled_stability_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=datasets,
    )

    assert calls["load_kwargs"]["datasets"] == datasets
    assert calls["load_kwargs"]["regions"] == figure_1_module.STABILITY_REGIONS
    assert calls["plot_table"] is table
    plt.close(fig)


def test_plot_decoding_error_dataset_stack_panel_removes_w_track_icons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for comparison, label, transfer_family, _pairs in (
        figure_1_module.DECODING_CROSS_TRAJECTORY_COMPARISONS
    ):
        rows.extend(
            {
                "comparison": comparison,
                "comparison_label": label,
                "transfer_family": transfer_family,
                "absolute_error": value,
            }
            for value in (0.1, 0.2, 0.4)
        )
    monkeypatch.setattr(
        supp_figure_1_module,
        "load_decoding_absolute_error_table",
        lambda **_kwargs: pd.DataFrame(rows),
    )

    fig, ax = plt.subplots()
    row_axes = plot_decoding_error_dataset_stack_panel(
        ax,
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
    )

    assert len(row_axes) == 1
    assert len(row_axes[0].child_axes) == 1
    assert row_axes[0].child_axes[0].get_ylabel() == "Abs. norm. error"
    assert "Train" not in [text.get_text() for text in row_axes[0].texts]
    plt.close(fig)


def test_make_supplementary_figure_1_uses_paper_style_and_figure_1_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_apply_paper_style() -> None:
        calls["styled"] = True

    def fake_plot_motor_delta_dataset_stack_panel(ax, **kwargs: object):
        calls["motor_kwargs"] = kwargs
        ax.text(0.5, 0.5, "motor")
        return []

    def fake_plot_encoding_delta_dataset_stack_panel(ax, **kwargs: object):
        calls["encoding_kwargs"] = kwargs
        ax.text(0.5, 0.5, "encoding")
        return []

    def fake_plot_decoding_error_dataset_stack_panel(ax, **kwargs: object):
        calls["decoding_kwargs"] = kwargs
        ax.text(0.5, 0.5, "decoding")
        return []

    def fake_draw_panel_a_anatomy_assets(ax, **kwargs: object):
        calls["anatomy_kwargs"] = kwargs
        ax.text(0.5, 0.5, "anatomy")

    def fake_plot_pooled_stability_panel(ax, **kwargs: object):
        calls["pooled_stability_kwargs"] = kwargs
        ax.text(0.5, 0.5, "pooled stability")

    def fake_plot_stability_dataset_rows_panel(ax, **kwargs: object):
        calls["stability_kwargs"] = kwargs
        ax.text(0.5, 0.5, "stability")
        return []

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

    monkeypatch.setattr(supp_figure_1_module, "apply_paper_style", fake_apply_paper_style)
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_motor_delta_dataset_stack_panel",
        fake_plot_motor_delta_dataset_stack_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_encoding_delta_dataset_stack_panel",
        fake_plot_encoding_delta_dataset_stack_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_decoding_error_dataset_stack_panel",
        fake_plot_decoding_error_dataset_stack_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "draw_panel_a_anatomy_assets",
        fake_draw_panel_a_anatomy_assets,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_pooled_stability_panel",
        fake_plot_pooled_stability_panel,
    )
    monkeypatch.setattr(
        supp_figure_1_module,
        "plot_stability_dataset_rows_panel",
        fake_plot_stability_dataset_rows_panel,
    )
    monkeypatch.setattr(supp_figure_1_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_1.svg"
    asset_dir = tmp_path / "assets"
    datasets = [("L14", "20240611", "08_r4")]
    saved_path = make_supplementary_figure_1(
        data_root=Path("/analysis"),
        asset_dir=asset_dir,
        output_path=output_path,
        datasets=datasets,
        encoding_place_bin_size_cm=4.0,
        dpi=300,
    )

    figure_width_in = calls["figsize"][0]
    assert saved_path == output_path
    assert calls["styled"] is True
    assert figure_width_in == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["A", "B", "C", "D", "E", "F"]
    assert calls["anatomy_kwargs"]["asset_dir"] == asset_dir
    assert calls["pooled_stability_kwargs"]["datasets"] == datasets
    assert calls["stability_kwargs"]["datasets"] == datasets
    assert calls["motor_kwargs"]["datasets"] == datasets
    assert calls["motor_kwargs"].get("show_dataset_labels", True) is True
    assert calls["encoding_kwargs"]["datasets"] == datasets
    assert calls["encoding_kwargs"]["show_dataset_labels"] is False
    assert calls["encoding_kwargs"]["place_bin_size_cm"] == pytest.approx(4.0)
    assert calls["decoding_kwargs"]["datasets"] == datasets
    assert calls["decoding_kwargs"]["show_dataset_labels"] is False
