"""Tests for the Figure 3 model and cue-swap panels."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import v1ca1.paper_figures.figure_2_old as figure_2_module
import v1ca1.paper_figures.figure_3 as figure_3_module
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_FIGURE_HEIGHT_MM,
    DEFAULT_FIGURE_WIDTH_MM,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_OUTPUT_NAME,
    PANEL_A_B_LABEL_X,
    PANEL_BC_SPLIT_LABEL_Y,
    PANEL_BC_TITLE_PAD,
    PANEL_B_LABEL_X,
    PANEL_C_LABEL_X,
    PANEL_LABEL_FONTSIZE,
    PANEL_TITLES,
    build_output_path,
    load_figure_3_panel_data,
    make_figure_3,
    parse_arguments,
)


def test_defaults_use_half_figure_2_width_and_two_row_units() -> None:
    args = parse_arguments([])

    assert DEFAULT_FIGURE_WIDTH_MM == pytest.approx(
        figure_2_module.DEFAULT_FIGURE_WIDTH_MM / 2.0
    )
    assert DEFAULT_FIGURE_HEIGHT_MM == pytest.approx(
        figure_2_module.PANEL_BC_QUANT_ROW_HEIGHT_MM
        + figure_2_module.PANEL_D_ROW_HEIGHT_MM
    )
    assert args.output_dir == DEFAULT_OUTPUT_DIR == Path("paper_figures/output")
    assert args.output_name == DEFAULT_OUTPUT_NAME == "figure_3"
    assert args.output_format == DEFAULT_OUTPUT_FORMAT
    assert DEFAULT_OUTPUT_FORMAT == figure_2_module.DEFAULT_OUTPUT_FORMAT
    assert args.dataset is None
    assert args.region is None
    assert args.dark_epoch is None
    assert not hasattr(args, "light_epoch")
    assert not hasattr(args, "decoding_n_permutations")
    assert not hasattr(args, "decoding_permutation_seed")

    assert build_output_path(args.output_dir, args.output_name, "svg") == Path(
        "paper_figures/output/figure_3.svg"
    )
    with pytest.raises(ValueError, match="Unknown output format"):
        build_output_path(args.output_dir, args.output_name, "jpg")


def test_canonical_figure_3_does_not_load_old_renderer() -> None:
    code = (
        "import sys; "
        "import v1ca1.paper_figures.figure_3; "
        "assert 'v1ca1.paper_figures.figure_3_old' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_exposes_data_dataset_dark_epoch_and_region_options() -> None:
    args = parse_arguments(
        [
            "--data-root",
            "/analysis",
            "--output-dir",
            "/figures",
            "--dataset",
            "L14:20240611",
            "--region",
            "v1",
            "--dark-epoch",
            "08_r4",
            "--dpi",
            "144",
        ]
    )

    assert args.data_root == Path("/analysis")
    assert args.output_dir == Path("/figures")
    assert args.dataset == [("L14", "20240611", "08_r4")]
    assert args.region == ["v1"]
    assert args.dark_epoch == "08_r4"
    assert args.dpi == 144


def test_panel_data_loader_reads_only_swap_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fail_unused_loader(**_kwargs: object) -> object:
        raise AssertionError("an unused Figure 2A/B or combined GLM loader was called")

    def fake_load_swap_examples(**kwargs: object) -> list[str]:
        calls["swap_examples_kwargs"] = kwargs
        return ["swap-example"]

    def fake_load_swap_delta(**kwargs: object) -> str:
        calls["swap_delta_kwargs"] = kwargs
        return "swap-delta"

    monkeypatch.setattr(
        figure_2_module,
        "load_panel_glm_data",
        fail_unused_loader,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_a_example_data",
        fail_unused_loader,
    )
    monkeypatch.setattr(
        figure_2_module,
        "load_panel_b_tuning_overlap_table",
        fail_unused_loader,
    )
    monkeypatch.setattr(
        figure_3_module,
        "load_panel_h_swap_examples",
        fake_load_swap_examples,
    )
    monkeypatch.setattr(
        figure_3_module,
        "load_panel_h_swap_delta_table",
        fake_load_swap_delta,
    )
    monkeypatch.setattr(
        figure_2_module,
        "build_panel_e_decoding_trial_error_table",
        fail_unused_loader,
    )
    monkeypatch.setattr(
        figure_2_module,
        "compute_panel_e_decoding_permutation_tests",
        fail_unused_loader,
    )
    monkeypatch.setattr(
        figure_2_module,
        "build_panel_e_decoding_significance_labels",
        fail_unused_loader,
    )

    panel_data = load_figure_3_panel_data(
        data_root=Path("/analysis"),
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        dark_epoch="08_r4",
    )

    assert panel_data == {
        "swap_delta": "swap-delta",
        "swap_examples": ["swap-example"],
    }
    common_loader_kwargs = {
        "data_root": Path("/analysis"),
        "datasets": (("L14", "20240611", "08_r4"),),
        "region": "v1",
    }
    assert calls["swap_examples_kwargs"] == {
        **common_loader_kwargs,
        "dark_epoch": "08_r4",
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "example_count": len(figure_2_module.PANEL_C_SWAP_EXAMPLES),
        "requested_examples": figure_2_module.PANEL_C_SWAP_EXAMPLES,
    }
    assert calls["swap_delta_kwargs"] == {
        **common_loader_kwargs,
        "dark_epoch": "08_r4",
        "min_movement_firing_rate_hz": (
            figure_2_module.PANEL_B_HISTOGRAM_MIN_MOVEMENT_FIRING_RATE_HZ
        ),
        "min_tuning_stability_correlation": (
            figure_2_module.PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
    }


def test_make_figure_3_stacks_model_and_swap_panels_at_half_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    calls: dict[str, object] = {}

    def fake_load_panel_data(**kwargs: object) -> dict[str, object]:
        calls["loader_kwargs"] = kwargs
        return {
            "swap_delta": "swap-delta",
            "swap_examples": ["swap-example"],
        }

    def fake_plot_architecture(ax: object) -> None:
        calls["architecture_axis"] = ax

    def fake_plot_swap(
        ax: object,
        swap_delta: object,
        swap_examples: object,
        **kwargs: object,
    ) -> None:
        calls["swap_axis"] = ax
        calls["swap_delta"] = swap_delta
        calls["swap_examples"] = swap_examples
        calls["swap_kwargs"] = kwargs

    def fail_if_decoding_is_plotted(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Removed Figure 3 Panel C was plotted")

    def fake_save_figure(
        figure: object,
        output_path: Path,
        dpi: int,
        **kwargs: object,
    ) -> Path:
        figure.canvas.draw()
        titled_axes = {ax.get_title(): ax for ax in figure.axes if ax.get_title()}
        blank_axes = [
            ax
            for ax in figure.axes
            if not ax.get_title() and not ax.axison and not ax.texts
        ]
        calls["figsize"] = tuple(figure.get_size_inches())
        calls["titles"] = tuple(titled_axes)
        calls["bounds"] = {
            title: ax.get_position().bounds for title, ax in titled_axes.items()
        }
        calls["blank_bounds"] = [ax.get_position().bounds for ax in blank_axes]
        calls["labels"] = {
            text.get_text()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["label_positions"] = {
            text.get_text(): text.get_position()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["label_font_sizes"] = {
            text.get_text(): text.get_fontsize()
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["label_display_positions"] = {
            text.get_text(): text.get_transform().transform(text.get_position())
            for ax in figure.axes
            for text in ax.texts
            if text.get_text() in {"A", "B", "C"}
        }
        calls["output_path"] = output_path
        calls["dpi"] = dpi
        calls["save_kwargs"] = kwargs
        return output_path

    monkeypatch.setattr(
        figure_3_module,
        "load_figure_3_panel_data",
        fake_load_panel_data,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d2_architecture_panel",
        fake_plot_architecture,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_d2_swap_results_panel",
        fake_plot_swap,
    )
    monkeypatch.setattr(
        figure_2_module,
        "plot_panel_e2_decoding_panel",
        fail_if_decoding_is_plotted,
    )
    monkeypatch.setattr(figure_3_module, "save_figure", fake_save_figure)

    output_path = tmp_path / "figure_3.svg"
    saved_path = make_figure_3(
        data_root=Path("/analysis"),
        output_path=output_path,
        datasets=[("L14", "20240611", "08_r4")],
        regions=("v1",),
        dark_epoch=None,
        dpi=144,
    )

    assert saved_path == output_path
    assert calls["figsize"] == pytest.approx(
        (
            DEFAULT_FIGURE_WIDTH_MM / 25.4,
            DEFAULT_FIGURE_HEIGHT_MM / 25.4,
        )
    )
    assert calls["titles"] == PANEL_TITLES
    assert calls["labels"] == {"A", "B", "C"}
    assert calls["label_positions"]["A"][0] == pytest.approx(PANEL_A_B_LABEL_X)
    assert calls["label_positions"]["B"] == pytest.approx(
        (PANEL_B_LABEL_X, PANEL_BC_SPLIT_LABEL_Y)
    )
    assert calls["label_positions"]["C"] == pytest.approx(
        (PANEL_C_LABEL_X, PANEL_BC_SPLIT_LABEL_Y)
    )
    assert set(calls["label_font_sizes"].values()) == {PANEL_LABEL_FONTSIZE}
    assert calls["label_display_positions"]["A"][0] == pytest.approx(
        calls["label_display_positions"]["B"][0]
    )
    assert calls["label_display_positions"]["B"][1] == pytest.approx(
        calls["label_display_positions"]["C"][1]
    )
    assert PANEL_BC_TITLE_PAD > figure_2_module.PANEL_BC_TITLE_PAD
    assert calls["blank_bounds"] == []
    assert calls["output_path"] == output_path
    assert calls["dpi"] == 144
    assert calls["save_kwargs"] == {"bbox_inches": None}
    assert calls["swap_delta"] == "swap-delta"
    assert calls["swap_examples"] == ["swap-example"]
    assert calls["swap_kwargs"] == {
        "model_name": figure_2_module.PANEL_C_SWAP_MODEL_NAME,
        "model_colors": figure_2_module.PANEL_C_SWAP_MODEL_COLORS_2_3,
        "model_labels": figure_2_module.PANEL_C_SWAP_MODEL_LABELS_2_3,
    }
    assert calls["loader_kwargs"] == {
        "data_root": Path("/analysis"),
        "datasets": [("L14", "20240611", "08_r4")],
        "regions": ("v1",),
        "dark_epoch": None,
    }

    bounds = calls["bounds"]
    panel_a = bounds[PANEL_TITLES[0]]
    panel_b = bounds[PANEL_TITLES[1]]
    assert panel_a[1] > panel_b[1]
    assert panel_a[0] == pytest.approx(panel_b[0])
    assert panel_a[2] == pytest.approx(panel_b[2])
