from __future__ import annotations

from pathlib import Path

import pytest

from v1ca1.paper_figures import supplementary_figure_3 as base
from v1ca1.paper_figures import supplementary_figure_3_3 as figure


def test_default_cli_keeps_only_supplementary_figure_3_panels_b_and_c() -> None:
    args = figure.parse_arguments([])

    assert args.output_dir == figure.DEFAULT_OUTPUT_DIR
    assert args.output_name == "supplementary_figure_3_3"
    assert args.output_format == figure.DEFAULT_OUTPUT_FORMAT
    assert figure.DEFAULT_FIGURE_WIDTH_MM == base.DEFAULT_FIGURE_WIDTH_MM
    assert figure.DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        base.DEFAULT_MOTOR_GRID_HEIGHT_MM
        + base.DEFAULT_BOTTOM_SECTION_SPACER_MM
        + base.DEFAULT_MOTOR_SUMMARY_HEIGHT_MM
    )
    assert figure.DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(178.1)
    assert not hasattr(args, "region")
    assert not hasattr(args, "light_epoch")
    assert not hasattr(args, "dark_epoch")
    assert not hasattr(args, "position_bin_count")
    assert not hasattr(args, "panel_a_cache_dir")
    assert not hasattr(args, "refresh_panel_a_cache")


def test_main_builds_named_output_and_forwards_relevant_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_make_supplementary_figure_3_3(**kwargs: object) -> Path:
        calls.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(
        figure,
        "make_supplementary_figure_3_3",
        fake_make_supplementary_figure_3_3,
    )
    figure.main(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            str(tmp_path),
            "--format",
            "svg",
            "--dataset",
            "L14:20240611:08_r4",
            "--dpi",
            "144",
        ]
    )

    assert calls == {
        "data_root": Path("/analysis"),
        "output_path": tmp_path / "supplementary_figure_3_3.svg",
        "datasets": [("L14", "20240611", "08_r4")],
        "dpi": 144,
    }


def test_make_supplementary_figure_3_3_relabels_original_panels_b_and_c(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fail_if_panel_a_is_used(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Supplementary Figure 3_3 used omitted Panel A")

    def fake_load_motor_table(**kwargs: object) -> str:
        calls["motor_load_kwargs"] = kwargs
        return "motor-table"

    def fake_plot_motor_grid(axes, table: object, **kwargs: object) -> None:
        calls["motor_axes"] = axes
        calls["motor_table"] = table
        calls["motor_plot_kwargs"] = kwargs
        axes[0, 0].text(0.5, 0.5, "motor")

    def fake_build_motor_summary(table: object, **kwargs: object) -> str:
        calls["summary_source_table"] = table
        calls["summary_build_kwargs"] = kwargs
        return "motor-summary"

    def fake_plot_motor_summary(axes, table: object, **kwargs: object) -> None:
        calls["summary_axes"] = axes
        calls["summary_table"] = table
        calls["summary_plot_kwargs"] = kwargs
        axes[0].text(0.5, 0.5, "summary")

    def fake_save_figure(fig, output_path: Path, dpi: int, **kwargs: object) -> Path:
        calls["figsize"] = fig.get_size_inches()
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        calls["panel_labels"] = [
            text.get_text()
            for ax in fig.axes
            for text in ax.texts
            if text.get_fontweight() == "bold"
        ]
        calls["figure_titles"] = [text.get_text() for text in fig.texts]
        calls["axis_count"] = len(fig.axes)
        calls["axis_off_count"] = sum(not ax.axison for ax in fig.axes)
        return output_path

    monkeypatch.setattr(figure, "apply_paper_style", lambda: None)
    monkeypatch.setattr(
        base,
        "load_panel_a_cv_pca_participation_ratio_table",
        fail_if_panel_a_is_used,
    )
    monkeypatch.setattr(
        base,
        "plot_panel_a_cv_pca_participation_ratios",
        fail_if_panel_a_is_used,
    )
    monkeypatch.setattr(
        base,
        "load_panel_b_motor_progression_table",
        fake_load_motor_table,
    )
    monkeypatch.setattr(
        base,
        "plot_panel_b_motor_progression_grid",
        fake_plot_motor_grid,
    )
    monkeypatch.setattr(
        base,
        "build_panel_c_motor_profile_correlation_table",
        fake_build_motor_summary,
    )
    monkeypatch.setattr(
        base,
        "plot_panel_c_motor_profile_correlations",
        fake_plot_motor_summary,
    )
    monkeypatch.setattr(figure, "save_figure", fake_save_figure)

    output_path = tmp_path / "supplementary_figure_3_3.svg"
    datasets = [("L14", "20240611", "08_r4")]
    saved_path = figure.make_supplementary_figure_3_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=datasets,
        dpi=300,
    )

    assert saved_path == output_path
    assert calls["figsize"] == pytest.approx(
        (
            figure.DEFAULT_FIGURE_WIDTH_MM / 25.4,
            figure.DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 300
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["panel_labels"] == ["A", "B"]
    assert calls["figure_titles"] == [base.PANEL_C_TITLE, base.PANEL_B_TITLE]
    assert calls["axis_count"] == (
        len(base.MOTOR_VARIABLES) * len(base.PANEL_B_TRAJECTORY_TYPES)
        + 1
        + len(base.PANEL_B_TRAJECTORY_TYPES)
    )
    assert calls["axis_off_count"] == 1

    assert calls["motor_load_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": datasets,
    }
    assert calls["motor_table"] == "motor-table"
    assert calls["motor_plot_kwargs"] == {"datasets": datasets}
    assert calls["summary_source_table"] == "motor-table"
    assert calls["summary_build_kwargs"] == {"datasets": datasets}
    assert calls["summary_table"] == "motor-summary"
    assert calls["summary_plot_kwargs"] == {"datasets": datasets}

    motor_axes = calls["motor_axes"]
    summary_axes = calls["summary_axes"]
    assert motor_axes.shape == (
        len(base.MOTOR_VARIABLES),
        len(base.PANEL_B_TRAJECTORY_TYPES),
    )
    assert len(summary_axes) == len(base.PANEL_B_TRAJECTORY_TYPES)
    assert [ax.get_xlabel() for ax in motor_axes[-1, :]] == [
        "Norm. path progression"
    ] * len(base.PANEL_B_TRAJECTORY_TYPES)
    assert all(not ax.get_xlabel() for ax in motor_axes[:-1, :].ravel())
    assert [ax.get_xlabel() for ax in summary_axes] == [
        "Dark-light correlation"
    ] * len(base.PANEL_B_TRAJECTORY_TYPES)
    assert motor_axes[0, 0].texts[-1].get_position() == pytest.approx(
        (-0.28, 1.05)
    )
    assert summary_axes[0].texts[-1].get_position() == pytest.approx(
        (-0.30, 1.05)
    )

    motor_grid = motor_axes[0, 0].get_subplotspec().get_gridspec()
    summary_grid = summary_axes[0].get_subplotspec().get_gridspec()
    motor_grid_params = motor_grid.get_subplot_params()
    summary_grid_params = summary_grid.get_subplot_params()
    assert motor_grid.get_geometry() == (
        len(base.MOTOR_VARIABLES),
        len(base.PANEL_B_TRAJECTORY_TYPES),
    )
    assert motor_grid_params.hspace == pytest.approx(base.MOTOR_GRID_HSPACE)
    assert motor_grid_params.wspace == pytest.approx(base.MOTOR_GRID_WSPACE)
    assert summary_grid.get_geometry() == (
        1,
        len(base.PANEL_B_TRAJECTORY_TYPES),
    )
    assert summary_grid_params.wspace == pytest.approx(
        base.MOTOR_SUMMARY_GRID_WSPACE
    )

    motor_outer_spec = motor_axes[0, 0].get_subplotspec().get_topmost_subplotspec()
    summary_outer_spec = summary_axes[0].get_subplotspec().get_topmost_subplotspec()
    outer_grid = motor_outer_spec.get_gridspec()
    outer_grid_params = outer_grid.get_subplot_params()
    assert motor_outer_spec.rowspan.start == 0
    assert summary_outer_spec.rowspan.start == 2
    assert outer_grid.get_height_ratios() == pytest.approx(
        [
            base.DEFAULT_MOTOR_GRID_HEIGHT_MM,
            base.DEFAULT_BOTTOM_SECTION_SPACER_MM,
            base.DEFAULT_MOTOR_SUMMARY_HEIGHT_MM,
        ]
    )
    assert outer_grid_params.hspace == pytest.approx(base.PANEL_GRID_HSPACE)
    assert outer_grid_params.left == pytest.approx(base.PANEL_A_GRID_LEFT)
    assert outer_grid_params.right == pytest.approx(base.PANEL_A_GRID_RIGHT)
    assert outer_grid_params.top == pytest.approx(base.PANEL_A_GRID_TOP)
    assert outer_grid_params.bottom == pytest.approx(base.PANEL_A_GRID_BOTTOM)
