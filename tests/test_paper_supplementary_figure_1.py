from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import v1ca1.paper_figures.figure_1 as figure_1_module
import v1ca1.paper_figures.supplementary_figure_1 as supp_figure_1_module
from v1ca1.paper_figures.supplementary_figure_1 import (
    DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ,
    DEFAULT_ASSET_DIR,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_NAME,
    build_output_path,
    load_pooled_dark_movement_firing_rate_table,
    make_supplementary_figure_1,
    parse_arguments,
    plot_pooled_dark_movement_firing_rate_histogram,
    plot_pooled_stability_panel,
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
    assert args.dark_movement_fr_cache_dir is None
    assert args.refresh_dark_movement_fr_cache is False
    assert DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM == pytest.approx(40.0)
    assert DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ == pytest.approx(0.5)
    assert not hasattr(args, "region")
    assert not hasattr(args, "encoding_place_bin_size_cm")
    assert not hasattr(args, "panel_d_cache_dir")
    assert not hasattr(args, "panel_heatmap_cache_dir")
    assert not hasattr(args, "refresh_panel_heatmap_cache")


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


def test_load_pooled_dark_movement_firing_rate_table_uses_dataset_dark_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pd = pytest.importorskip("pandas")

    calls = []

    def fake_load_dark_movement_firing_rate_table(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(
            {
                "unit": [len(calls)],
                "dark_firing_rate_hz": [0.25 * len(calls)],
            }
        )

    monkeypatch.setattr(
        supp_figure_1_module,
        "load_dark_movement_firing_rate_table",
        fake_load_dark_movement_firing_rate_table,
    )

    table = load_pooled_dark_movement_firing_rate_table(
        Path("/analysis"),
        [
            ("L14", "20240611", "08_r4"),
            ("L15", "20241121", "10_r5"),
        ],
        cache_dir=Path("/cache"),
        refresh_cache=True,
    )

    assert [call["animal_name"] for call in calls] == ["L14", "L15"]
    assert [call["dark_epoch"] for call in calls] == ["08_r4", "10_r5"]
    assert all(call["region"] == "v1" for call in calls)
    assert all(call["cache_dir"] == Path("/cache") for call in calls)
    assert all(call["refresh_cache"] is True for call in calls)
    assert table["animal_name"].tolist() == ["L14", "L15"]
    assert table["date"].tolist() == ["20240611", "20241121"]
    assert table["dark_epoch"].tolist() == ["08_r4", "10_r5"]
    assert table["dark_firing_rate_hz"].tolist() == [0.25, 0.5]


def test_plot_pooled_dark_movement_firing_rate_histogram_uses_log_threshold() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_pooled_dark_movement_firing_rate_histogram(
        ax,
        {
            "dark_firing_rate_hz": [
                0.0,
                0.1,
                0.5,
                0.6,
                1.2,
                np.nan,
            ]
        },
    )

    assert ax.get_xscale() == "log"
    threshold_lines = [
        line
        for line in ax.lines
        if line.get_linestyle() == "--"
        and np.allclose(line.get_xdata(), [DARK_MOVEMENT_FIRING_RATE_THRESHOLD_HZ] * 2)
    ]
    assert len(threshold_lines) == 1
    assert any("40% > 0.5 Hz" in text.get_text() for text in ax.texts)
    assert not any("at 0 Hz" in text.get_text() for text in ax.texts)
    assert ax.get_xlabel() == "Mean firing rate (Hz)"
    assert ax.get_ylabel() == "Frac."
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

    def fake_draw_panel_a_anatomy_assets(ax, **kwargs: object):
        calls["anatomy_kwargs"] = kwargs
        ax.text(0.5, 0.5, "anatomy")

    def fake_plot_pooled_stability_panel(ax, **kwargs: object):
        calls["pooled_stability_kwargs"] = kwargs
        calls["pooled_stability_col"] = ax.get_subplotspec().colspan.start
        ax.text(0.5, 0.5, "pooled stability")

    def fake_plot_pooled_dark_movement_firing_rate_panel(ax, **kwargs: object):
        calls["dark_movement_fr_kwargs"] = kwargs
        calls["dark_movement_fr_col"] = ax.get_subplotspec().colspan.start
        ax.text(0.5, 0.5, "dark fr")

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
        calls["axis_titles"] = [ax.get_title() for ax in figure.axes]
        return output_path

    monkeypatch.setattr(supp_figure_1_module, "apply_paper_style", fake_apply_paper_style)
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
        "plot_pooled_dark_movement_firing_rate_panel",
        fake_plot_pooled_dark_movement_firing_rate_panel,
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
        dpi=300,
    )

    figure_width_in = calls["figsize"][0]
    assert saved_path == output_path
    assert calls["styled"] is True
    assert figure_width_in == pytest.approx(DEFAULT_FIGURE_WIDTH_MM / 25.4)
    assert calls["figsize"][1] == pytest.approx(
        DEFAULT_MOVED_FIGURE_1_ROW_HEIGHT_MM / 25.4
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["panel_labels"] == ["A", "B", "C"]
    assert "V1 firing rate in darkness" in calls["axis_titles"]
    assert "Tuning stability" in calls["axis_titles"]
    assert calls["anatomy_kwargs"]["asset_dir"] == asset_dir
    assert calls["pooled_stability_kwargs"]["datasets"] == datasets
    assert calls["pooled_stability_col"] == 2
    assert calls["dark_movement_fr_kwargs"]["datasets"] == datasets
    assert calls["dark_movement_fr_col"] == 1
    assert calls["dark_movement_fr_kwargs"]["cache_dir"] == output_path.parent / "cache"
    assert calls["dark_movement_fr_kwargs"]["refresh_cache"] is False
