"""Generate a compact graphical summary of the DPP coding story."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.helper.session import DEFAULT_DATA_ROOT, REGIONS
from v1ca1.paper_figures.datasets import DatasetId, get_processed_datasets
from v1ca1.paper_figures.figure_1 import (
    DEFAULT_ASSET_DIR,
    CYCLE_ARROW_LINEWIDTH,
    CYCLE_ARROW_MUTATION_SCALE,
    CYCLE_ARROW_SPECS,
    CYCLE_TRAJECTORY_LAYOUT,
    PANEL_E_TRAJECTORY_COLORS,
    draw_panel_b_visual_epoch_icon,
    draw_visual_stimuli_schematic,
)
from v1ca1.paper_figures.figure_2 import (
    DEFAULT_POSITION_BIN_COUNT,
    DEFAULT_POSITION_OFFSET,
    DEFAULT_REGIONS,
    DEFAULT_SIGMA_BINS,
    DEFAULT_SPEED_THRESHOLD_CM_S,
    FIGURE_2_PANEL_A_EXAMPLES,
    MIN_PUBLICATION_FONTSIZE_EXEMPT_TEXT,
    MIN_PUBLICATION_FONTSIZE_PT,
    PANEL_A_EXAMPLE_Y_MAX_OVERRIDES,
    PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
    PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION,
    build_output_path,
    filter_panel_b_overlap_by_even_odd_stability,
    load_panel_a_example_data,
    load_panel_b_tuning_overlap_table,
    parse_dataset_id,
    plot_panel_b_dpp_overlap_scatter,
    plot_panel_b_dppi_schematic,
)
from v1ca1.paper_figures.figure_3 import (
    DEFAULT_OUTPUT_DIR,
    PANEL_A_DARK_EPOCH_BACKGROUND,
    PANEL_A_EPOCH_LABELS,
    plot_epoch_path_rate_axis,
    validate_panel_a_trajectories,
)
from v1ca1.paper_figures.style import (
    apply_paper_style,
    figure_size,
    label_axis,
    save_figure,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_NAME = "figure_summary"
FIGURE_FORMATS = ("svg", "png", "pdf")
DEFAULT_OUTPUT_FORMAT = "svg"
LETTER_WIDTH_WITH_HALF_INCH_MARGINS_MM = 7.5 * 25.4
DEFAULT_FIGURE_HEIGHT_MM = 86.0
SUMMARY_WIDTH_RATIOS = (0.21, 0.19, 0.34, 0.26)
SUMMARY_WSPACE = 0.11
SUMMARY_EXAMPLE_COUNT = 2
SUMMARY_TASK_LEFT_ARM_COLOR = "#E78AC3"
SUMMARY_TASK_RIGHT_ARM_COLOR = "#66C2A5"
SUMMARY_TASK_BLOCK_X = 0.03
SUMMARY_TASK_BLOCK_WIDTH = 0.55
SUMMARY_TRAJECTORY_ICON_BOUNDS = (SUMMARY_TASK_BLOCK_X, 0.60, SUMMARY_TASK_BLOCK_WIDTH, 0.37)
SUMMARY_CONDITION_TRACK_BOUNDS = (
    (SUMMARY_TASK_BLOCK_X, 0.37, 0.17, 0.22),
    (SUMMARY_TASK_BLOCK_X + 0.19, 0.37, 0.17, 0.22),
    (SUMMARY_TASK_BLOCK_X + 0.38, 0.37, 0.17, 0.22),
)
SUMMARY_VISUAL_STIMULUS_BOUNDS = (SUMMARY_TASK_BLOCK_X, 0.10, SUMMARY_TASK_BLOCK_WIDTH, 0.34)


def _raise_text_to_minimum_fontsize(fig: Any, min_fontsize: float) -> None:
    """Raise final figure text to a minimum size, preserving W-track A/B labels."""

    def _iter_axes_tree(ax: Any) -> Sequence[Any]:
        axes = [ax]
        for child_ax in getattr(ax, "child_axes", ()):
            axes.extend(_iter_axes_tree(child_ax))
        return axes

    def _maybe_raise(text: Any) -> None:
        if text is None:
            return
        if text.get_text().strip() in MIN_PUBLICATION_FONTSIZE_EXEMPT_TEXT:
            return
        if text.get_fontsize() < min_fontsize:
            text.set_fontsize(min_fontsize)

    seen_axes = set()
    for root_ax in fig.axes:
        for ax in _iter_axes_tree(root_ax):
            if id(ax) in seen_axes:
                continue
            seen_axes.add(id(ax))
            _maybe_raise(ax.title)
            _maybe_raise(ax.xaxis.label)
            _maybe_raise(ax.yaxis.label)
            for tick_label in ax.get_xticklabels():
                _maybe_raise(tick_label)
            for tick_label in ax.get_yticklabels():
                _maybe_raise(tick_label)
            for text in ax.texts:
                _maybe_raise(text)
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    _maybe_raise(text)


def _set_rate_axis_xlabel(
    ax: Any,
    *,
    show_xlabel: bool,
) -> None:
    """Clear nested x labels and show tick labels only on the labeled row."""
    ax.set_xlabel("")
    if not show_xlabel:
        ax.set_xticklabels([])


def _add_combined_rate_axis_xlabel(ax: Any) -> None:
    """Add one x-axis label centered under the paired dark/light rate axes."""
    ax.text(
        0.56,
        0.02,
        "Norm. path progression",
        ha="center",
        va="bottom",
        fontsize=6.0,
        transform=ax.transAxes,
    )


def _draw_summary_condition_track(
    ax: Any,
    *,
    fill_track: bool = False,
    left_arm_color: str | None = None,
    right_arm_color: str | None = None,
) -> None:
    """Draw one summary W-track condition icon."""
    region_fill_colors = {}
    if left_arm_color is not None:
        region_fill_colors["left_arm"] = left_arm_color
    if right_arm_color is not None:
        region_fill_colors["right_arm"] = right_arm_color
    draw_panel_b_visual_epoch_icon(
        ax,
        fill_track=fill_track,
        region_fill_colors=region_fill_colors,
        region_fill_alpha=0.92,
    )


def _hide_visual_stimulus_track_labels(ax: Any) -> None:
    """Hide the internal W-track and arm labels from a visual-stimulus schematic."""
    for text in ax.texts:
        if text.get_text().strip() in {"L", "C", "R"}:
            text.set_text("")
    for line in ax.lines:
        line.set_visible(False)
    for patch in ax.patches[:6]:
        patch.set_visible(False)
    for patch in ax.patches[6:]:
        get_x = getattr(patch, "get_x", None)
        get_width = getattr(patch, "get_width", None)
        get_height = getattr(patch, "get_height", None)
        if get_x is None or get_width is None or get_height is None:
            continue
        x = float(get_x())
        width = float(get_width())
        height = float(get_height())
        if x < 0.35 and width <= 0.02 and height <= 0.13:
            patch.set_visible(False)


def draw_summary_task_panel(ax: Any) -> None:
    """Draw the compact summary task schematic for Panel A."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    trajectory_ax = ax.inset_axes(SUMMARY_TRAJECTORY_ICON_BOUNDS)
    trajectory_ax.set_xlim(0.0, 1.0)
    trajectory_ax.set_ylim(0.0, 1.0)
    trajectory_ax.axis("off")
    for trajectory_type, bounds in CYCLE_TRAJECTORY_LAYOUT:
        inset = trajectory_ax.inset_axes(bounds)
        draw_w_track_schematic(
            inset,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=1.2,
            trajectory_linewidth=1.4,
            arrow_mutation_scale=13.0,
            fill_track=False,
        )
    for start, end, rad in CYCLE_ARROW_SPECS:
        trajectory_ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "linewidth": CYCLE_ARROW_LINEWIDTH,
                "mutation_scale": CYCLE_ARROW_MUTATION_SCALE,
                "shrinkA": 0,
                "shrinkB": 0,
                "connectionstyle": f"arc3,rad={rad}",
            },
            annotation_clip=False,
        )

    condition_specs = (
        {"fill_track": True},
        {
            "left_arm_color": SUMMARY_TASK_RIGHT_ARM_COLOR,
            "right_arm_color": SUMMARY_TASK_LEFT_ARM_COLOR,
        },
        {
            "left_arm_color": SUMMARY_TASK_LEFT_ARM_COLOR,
            "right_arm_color": SUMMARY_TASK_RIGHT_ARM_COLOR,
        },
    )
    for bounds, condition_spec in zip(
        SUMMARY_CONDITION_TRACK_BOUNDS,
        condition_specs,
        strict=True,
    ):
        _draw_summary_condition_track(ax.inset_axes(bounds), **condition_spec)
    visual_ax = ax.inset_axes(SUMMARY_VISUAL_STIMULUS_BOUNDS)
    draw_visual_stimuli_schematic(visual_ax)
    _hide_visual_stimulus_track_labels(visual_ax)


def _example_rate_y_max(example: dict[str, Any], fallback_y_max: float | None) -> float:
    """Return a compact y-axis limit for one example-cell rate plot."""
    if fallback_y_max is not None:
        return float(fallback_y_max)

    values = []
    for epoch_rates in example["epoch_rates"].values():
        for _position, rate in epoch_rates["firing_rates"].values():
            values.append(np.asarray(rate, dtype=float))
    if not values:
        return 1.0
    max_rate = float(np.nanmax(np.concatenate(values)))
    if not np.isfinite(max_rate) or max_rate <= 0.0:
        return 1.0
    step = 5.0 if max_rate <= 50.0 else 10.0
    return float(np.ceil(max_rate / step) * step)


def _plot_summary_example_row(
    ax: Any,
    example: dict[str, Any],
    *,
    label: str,
    y_max: float,
    show_xlabel: bool,
) -> None:
    """Plot dark and light tuning curves for one DPP example cell."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.0,
        0.98,
        label,
        ha="left",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        transform=ax.transAxes,
    )

    trajectories = validate_panel_a_trajectories(example["trajectories"])
    axis_y = 0.22 if show_xlabel else 0.14
    axis_height = 0.60 if show_xlabel else 0.68
    dark_ax = ax.inset_axes([0.14, axis_y, 0.37, axis_height])
    light_ax = ax.inset_axes([0.60, axis_y, 0.37, axis_height])
    dark_ax.set_facecolor(PANEL_A_DARK_EPOCH_BACKGROUND)
    for epoch_key, rate_ax in (("dark", dark_ax), ("light", light_ax)):
        plot_epoch_path_rate_axis(
            rate_ax,
            example,
            epoch_key,
            trajectories=trajectories,
            y_max=y_max,
            show_ylabel=epoch_key == "dark",
            show_title=True,
            show_correlation=False,
        )
        rate_ax.set_title(PANEL_A_EPOCH_LABELS[epoch_key], fontsize=6.0, pad=1.0)
        rate_ax.tick_params(labelsize=6.0, length=1.3, pad=0.8)
        _set_rate_axis_xlabel(rate_ax, show_xlabel=show_xlabel)
        if epoch_key == "light":
            rate_ax.set_ylabel("")
    if show_xlabel:
        _add_combined_rate_axis_xlabel(ax)


def plot_summary_examples(
    ax: Any,
    examples: Sequence[dict[str, Any]],
) -> None:
    """Plot two compact dark/light DPP example cells."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    if not examples:
        ax.text(0.5, 0.5, "No examples", ha="center", va="center")
        return

    row_gap = 0.08
    row_height = (0.88 - row_gap) / 2.0
    y_positions = (0.54, 0.54 - row_height - row_gap)
    for example_index, (example, y0) in enumerate(
        zip(examples[:SUMMARY_EXAMPLE_COUNT], y_positions, strict=False),
        start=1,
    ):
        row_ax = ax.inset_axes([0.0, y0, 1.0, row_height])
        y_max = _example_rate_y_max(
            example,
            PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(example_index),
        )
        _plot_summary_example_row(
            row_ax,
            example,
            label=f"Cell {example_index}",
            y_max=y_max,
            show_xlabel=example_index == min(len(examples), SUMMARY_EXAMPLE_COUNT),
        )


def _format_summary_dppi_scatter(ax: Any) -> None:
    """Use compact DPPI labels for the summary scatter and marginals."""
    for child_ax in ax.child_axes:
        if child_ax.get_xlabel() == "Dark DPP\noverlap":
            child_ax.set_xlabel("Dark DPPI", fontsize=6.0, labelpad=1.0)
            child_ax.set_ylabel("Light DPPI", fontsize=6.0, labelpad=1.0)
            child_ax.set_xticks([0.0, 0.5, 1.0])
            child_ax.set_yticks([0.0, 0.5, 1.0])
            child_ax.tick_params(labelsize=6.0, length=1.4, pad=1.0)
            for text in child_ax.texts:
                text.set_fontsize(6.0)
        elif child_ax.get_xlabel() == "Frac.":
            child_ax.set_xlabel("Frac.", fontsize=6.0, labelpad=0.8)
            child_ax.tick_params(axis="x", labelsize=6.0, length=1.2, pad=0.8)
        if child_ax.get_ylabel() == "Light DPP\noverlap":
            child_ax.set_ylabel("Light DPPI", fontsize=6.0, labelpad=1.0)
        elif child_ax.get_ylabel() == "Frac.":
            child_ax.set_ylabel("Frac.", fontsize=6.0, labelpad=0.8)
            child_ax.tick_params(axis="y", labelsize=6.0, length=1.2, pad=0.8)


def make_figure_summary(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    asset_dir: Path = DEFAULT_ASSET_DIR,
    position_bin_count: int = DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
    dark_tuning_correlation_threshold: float = (
        PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD
    ),
) -> Path:
    """Build and save the compact paper-summary figure."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    quant_region = str(regions[0]) if regions else DEFAULT_REGIONS[0]
    selected_examples = FIGURE_2_PANEL_A_EXAMPLES[:SUMMARY_EXAMPLE_COUNT]
    panel_examples = [
        load_panel_a_example_data(
            data_root=data_root,
            animal_name=animal_name,
            date=date,
            region=region,
            unit_id=unit_id,
            trajectories=trajectories,
            dark_epoch=dark_epoch,
            light_epoch=light_epoch,
            position_bin_count=position_bin_count,
            position_offset=position_offset,
            speed_threshold_cm_s=speed_threshold_cm_s,
            sigma_bins=sigma_bins,
            panel_example_cache_dir=panel_example_cache_dir,
            refresh_panel_example_cache=refresh_panel_example_cache,
        )
        for animal_name, date, region, unit_id, trajectories in selected_examples
    ]
    overlap_table = load_panel_b_tuning_overlap_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )
    overlap_table = filter_panel_b_overlap_by_even_odd_stability(
        overlap_table,
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
        min_stability_correlation=(
            PANEL_B_HISTOGRAM_MIN_TUNING_STABILITY_CORRELATION
        ),
    )

    apply_paper_style()
    fig = plt.figure(
        figsize=figure_size(
            LETTER_WIDTH_WITH_HALF_INCH_MARGINS_MM,
            DEFAULT_FIGURE_HEIGHT_MM,
        ),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=SUMMARY_WIDTH_RATIOS,
        wspace=SUMMARY_WSPACE,
    )
    task_ax = fig.add_subplot(grid[0, 0])
    dppi_ax = fig.add_subplot(grid[0, 1])
    examples_ax = fig.add_subplot(grid[0, 2])
    scatter_ax = fig.add_subplot(grid[0, 3])

    del asset_dir
    draw_summary_task_panel(task_ax)
    task_ax.set_title("W-track task", fontsize=8.0, pad=1.5)
    label_axis(task_ax, "A", x=-0.02, y=1.02)

    plot_panel_b_dppi_schematic(dppi_ax, panel_examples[0])
    dppi_ax.set_title("DPP index", fontsize=8.0, pad=1.5)
    label_axis(dppi_ax, "B", x=-0.02, y=1.02)

    plot_summary_examples(examples_ax, panel_examples)
    examples_ax.set_title("DPP example cells", fontsize=8.0, pad=1.5)
    label_axis(examples_ax, "C", x=-0.02, y=1.02)

    plot_panel_b_dpp_overlap_scatter(scatter_ax, overlap_table, title=None)
    _format_summary_dppi_scatter(scatter_ax)
    scatter_ax.set_title("Dark vs light DPP", fontsize=8.0, pad=1.5)
    label_axis(scatter_ax, "D", x=-0.02, y=1.02)

    _raise_text_to_minimum_fontsize(fig, MIN_PUBLICATION_FONTSIZE_PT)
    fig.set_layout_engine(None)
    save_figure(fig, output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved summary figure to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the paper summary figure."""
    parser = argparse.ArgumentParser(
        description="Generate a compact paper summary figure."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Base directory containing analysis outputs. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for figure output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output basename without extension. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=DEFAULT_ASSET_DIR,
        help=f"Directory containing Figure 1 assets. Default: {DEFAULT_ASSET_DIR}",
    )
    parser.add_argument(
        "--panel-example-cache-dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached example-cell rasters and rate curves. "
            "Default: <output-dir>/cache."
        ),
    )
    parser.add_argument(
        "--refresh-panel-example-cache",
        action="store_true",
        help="Recompute example-cell data and overwrite matching caches.",
    )
    parser.add_argument(
        "--dark-tuning-correlation-threshold",
        "--dpp-index-threshold",
        dest="dark_tuning_correlation_threshold",
        type=float,
        default=PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD,
        help=(
            "Dark DPPI threshold retained for CLI consistency. "
            f"Default: {PANEL_B_DARK_TUNING_CORRELATION_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=FIGURE_FORMATS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_id,
        help=(
            "Animal/date data set to include as animal:date. May be repeated. "
            "Default: use v1ca1.paper_figures.datasets."
        ),
    )
    parser.add_argument(
        "--region",
        action="append",
        choices=REGIONS,
        help=(
            "Region to include. May be repeated. "
            f"Default: {', '.join(DEFAULT_REGIONS)}."
        ),
    )
    parser.add_argument("--light-epoch", default=None, help="Light run epoch.")
    parser.add_argument("--dark-epoch", default=None, help="Dark run epoch.")
    parser.add_argument(
        "--position-bin-count",
        type=int,
        default=DEFAULT_POSITION_BIN_COUNT,
        help=(
            "Number of bins from normalized trajectory position 0 to 1. "
            f"Default: {DEFAULT_POSITION_BIN_COUNT}"
        ),
    )
    parser.add_argument(
        "--position-offset",
        type=int,
        default=DEFAULT_POSITION_OFFSET,
        help=f"Number of leading position samples to ignore. Default: {DEFAULT_POSITION_OFFSET}",
    )
    parser.add_argument(
        "--speed-threshold-cm-s",
        type=float,
        default=DEFAULT_SPEED_THRESHOLD_CM_S,
        help=(
            "Speed threshold in cm/s used to define movement intervals. "
            f"Default: {DEFAULT_SPEED_THRESHOLD_CM_S}"
        ),
    )
    parser.add_argument(
        "--sigma-bins",
        type=float,
        default=DEFAULT_SIGMA_BINS,
        help=f"Gaussian smoothing width in bins. Default: {DEFAULT_SIGMA_BINS}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help="Rasterization dpi for saved output. Default: 500",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run compact paper summary figure generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else DEFAULT_REGIONS
    output_path = build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_figure_summary(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
        asset_dir=args.asset_dir,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
        dark_tuning_correlation_threshold=args.dark_tuning_correlation_threshold,
    )


if __name__ == "__main__":
    main()
