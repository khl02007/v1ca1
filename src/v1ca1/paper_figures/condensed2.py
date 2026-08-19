"""Generate a condensed figure with three dark-light example cells."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from v1ca1.paper_figures import figure_2
from v1ca1.paper_figures import figure_2_old
from v1ca1.paper_figures import figure_summary as base
from v1ca1.paper_figures.condensed import (
    CONDENSED_FIGURE_HEIGHT_MM,
    CONDENSED_HEADER_FONTSIZE,
    CONDENSED_HEADER_LABEL_X_PAD,
    CONDENSED_HEADER_TOP_Y,
    CONDENSED_TIMELINE_FONT_EXEMPT_TEXT,
    CONDENSED_WSPACE,
    _emphasize_raster_axis,
    _set_shared_x_label,
)
from v1ca1.paper_figures.figure_1 import (
    PANEL_E_TRAJECTORY_COLORS,
    draw_behavior_task_design_panel,
)
from v1ca1.paper_figures.w_track_schematic import draw_w_track_schematic


DEFAULT_OUTPUT_NAME = "condensed2"
CONDENSED2_FIGURE_2_EXAMPLE_NUMBERS = (5, 2, 8)
CONDENSED2_WIDTH_RATIOS = (0.30, 0.70 / 3.0, 0.70 / 3.0, 0.70 / 3.0)
CONDENSED2_PANEL_B_TITLE = "Vision gain-modulates path-invariant firing in V1"
CONDENSED2_EXAMPLE_TITLE_Y = 0.92
CONDENSED2_PANEL_B_ICON_SCALE = 1.5
CONDENSED2_PANEL_B_ICON_WIDTH = 0.070 * CONDENSED2_PANEL_B_ICON_SCALE
CONDENSED2_PANEL_B_ICON_HEIGHT = 0.075 * CONDENSED2_PANEL_B_ICON_SCALE
CONDENSED2_PANEL_B_ICON_LEFT = -0.075
CONDENSED2_PANEL_B_ICON_TRACK_LINEWIDTH = 0.30
CONDENSED2_PANEL_B_ICON_TRAJECTORY_LINEWIDTH = 1.0
CONDENSED2_PANEL_B_YLABEL_X = -0.50
CONDENSED2_PANEL_A_TITLE = "W-maze with visual landmarks"
CONDENSED2_PANEL_A_TITLE_FONTSIZE = 8.0
CONDENSED2_TURN_GROUP_BOUNDS = (
    (0.02, 0.19, 0.40, 0.75),
    (0.58, 0.19, 0.40, 0.75),
)
CONDENSED2_TURN_GROUP_LABELS = ("Right turn", "Left turn")
CONDENSED2_TRAJECTORY_LAYOUT = (
    ("center_to_left", (0.08, 0.63, 0.28, 0.26)),
    ("right_to_center", (0.08, 0.27, 0.28, 0.26)),
    ("left_to_center", (0.64, 0.63, 0.28, 0.26)),
    ("center_to_right", (0.64, 0.27, 0.28, 0.26)),
)
CONDENSED2_CYCLE_ARROW_SPECS = (
    ((0.36, 0.76), (0.64, 0.76)),
    ((0.78, 0.63), (0.78, 0.53)),
    ((0.64, 0.40), (0.36, 0.40)),
    ((0.22, 0.53), (0.22, 0.63)),
)


def _argv_has_output_name(argv: Sequence[str] | None) -> bool:
    """Return whether the user explicitly supplied an output basename."""
    values = sys.argv[1:] if argv is None else list(argv)
    return any(value == "--output-name" or value.startswith("--output-name=") for value in values)


def _add_panel_headers(
    fig: Any,
    panel_specs: Sequence[tuple[Any, str | None, str, float]],
) -> None:
    """Add aligned labels and titles above the condensed2 panels."""
    for ax, panel_label, title, title_fontsize in panel_specs:
        bounds = ax.get_position()
        label_x = max(
            bounds.x0 - CONDENSED_HEADER_LABEL_X_PAD * bounds.width,
            0.012,
        )
        if panel_label is not None:
            fig.text(
                label_x,
                CONDENSED_HEADER_TOP_Y,
                panel_label,
                ha="right",
                va="top",
                fontsize=CONDENSED_HEADER_FONTSIZE,
                fontweight="bold",
            )
        fig.text(
            0.5 * (bounds.x0 + bounds.x1),
            CONDENSED_HEADER_TOP_Y,
            title,
            ha="center",
            va="top",
            fontsize=title_fontsize,
        )


def _add_panel_b_header(fig: Any, example_axes: Sequence[Any]) -> None:
    """Add the label and title spanning all three example cells."""
    first_bounds = example_axes[0].get_position()
    last_bounds = example_axes[-1].get_position()
    fig.text(
        first_bounds.x0 - CONDENSED_HEADER_LABEL_X_PAD * first_bounds.width,
        CONDENSED_HEADER_TOP_Y,
        "B",
        ha="right",
        va="top",
        fontsize=CONDENSED_HEADER_FONTSIZE,
        fontweight="bold",
    )
    fig.text(
        0.5 * (first_bounds.x0 + last_bounds.x1),
        CONDENSED_HEADER_TOP_Y,
        CONDENSED2_PANEL_B_TITLE,
        ha="center",
        va="top",
        fontsize=CONDENSED_HEADER_FONTSIZE,
    )


def _draw_panel_a_turn_groups(ax: Any) -> None:
    """Draw smaller trajectory icons grouped by left and right turns."""
    from matplotlib.patches import Rectangle

    for child_ax in tuple(ax.child_axes):
        child_ax.remove()
    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    for trajectory_type, bounds in CONDENSED2_TRAJECTORY_LAYOUT:
        trajectory_ax = ax.inset_axes(bounds)
        draw_w_track_schematic(
            trajectory_ax,
            trajectory_name=trajectory_type,
            arrow_color=PANEL_E_TRAJECTORY_COLORS[trajectory_type],
            track_linewidth=0.95,
            trajectory_linewidth=1.15,
            arrow_mutation_scale=9.0,
            fill_track=False,
        )
    for start, end in CONDENSED2_CYCLE_ARROW_SPECS:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "linewidth": 0.8,
                "mutation_scale": 8.5,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
            zorder=3,
        )
    for bounds, label in zip(
        CONDENSED2_TURN_GROUP_BOUNDS,
        CONDENSED2_TURN_GROUP_LABELS,
        strict=True,
    ):
        left, bottom, width, height = bounds
        ax.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor="none",
                edgecolor="0.45",
                linewidth=0.65,
                linestyle=(0, (3.0, 2.0)),
                transform=ax.transAxes,
                clip_on=False,
                zorder=2,
            )
        )
        ax.text(
            left + 0.5 * width,
            0.08,
            label,
            ha="center",
            va="center",
            fontsize=5.8,
            transform=ax.transAxes,
        )


def plot_dark_light_example_panel(
    ax: Any,
    example: dict[str, Any],
    *,
    example_number: int,
) -> None:
    """Plot one Figure 2 new example in a condensed single-column layout."""
    figure_2_old.plot_panel_a_example(
        ax,
        example,
        title=None,
        dark_epoch_axis_left=figure_2_old.PANEL_A_SINGLE_ROW_DARK_EPOCH_LEFT,
        light_epoch_axis_left=figure_2_old.PANEL_A_SINGLE_ROW_LIGHT_EPOCH_LEFT,
        epoch_axis_width=figure_2_old.PANEL_A_SINGLE_ROW_EPOCH_AXIS_WIDTH,
        schematic_axis_left=CONDENSED2_PANEL_B_ICON_LEFT,
        schematic_axis_width=CONDENSED2_PANEL_B_ICON_WIDTH,
        schematic_axis_height=CONDENSED2_PANEL_B_ICON_HEIGHT,
        schematic_track_linewidth=CONDENSED2_PANEL_B_ICON_TRACK_LINEWIDTH,
        schematic_trajectory_linewidth=(
            CONDENSED2_PANEL_B_ICON_TRAJECTORY_LINEWIDTH
        ),
        show_correlation=False,
        similarity_annotation="dppi",
    )
    raster_axes = ax.child_axes[2:4]
    for raster_ax in raster_axes:
        _emphasize_raster_axis(raster_ax)
    raster_axes[0].yaxis.set_label_coords(CONDENSED2_PANEL_B_YLABEL_X, 0.5)
    rate_axes = [child_ax for child_ax in ax.child_axes if child_ax.get_xlabel()]
    rate_axes[0].yaxis.set_label_coords(CONDENSED2_PANEL_B_YLABEL_X, 0.5)
    for rate_ax in rate_axes:
        rate_ax.set_xlabel("")
        rate_ax.tick_params(
            axis="x",
            labelsize=figure_2_old.MIN_PUBLICATION_FONTSIZE_PT,
            pad=0.4,
        )
    ax.text(
        0.5,
        CONDENSED2_EXAMPLE_TITLE_Y,
        f"Example {example_number}",
        ha="center",
        va="top",
        fontsize=figure_2_old.MIN_PUBLICATION_FONTSIZE_PT,
        transform=ax.transAxes,
    )
    _set_shared_x_label(ax)


def make_condensed2_figure(
    *,
    data_root: Path,
    output_path: Path,
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    asset_dir: Path = base.DEFAULT_ASSET_DIR,
    position_bin_count: int = base.DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = base.DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = base.DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = base.DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> Path:
    """Build and save condensed2 with examples 5, 2, and 8."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    selected_examples = [
        figure_2.FIGURE_2_PANEL_A_EXAMPLES[example_number - 1]
        for example_number in CONDENSED2_FIGURE_2_EXAMPLE_NUMBERS
    ]
    examples = [
        figure_2_old.load_panel_a_example_data(
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

    base.apply_paper_style()
    fig = plt.figure(
        figsize=base.figure_size(
            base.LETTER_WIDTH_WITH_HALF_INCH_MARGINS_MM,
            CONDENSED_FIGURE_HEIGHT_MM,
        ),
        constrained_layout=True,
    )
    fig.get_layout_engine().set(h_pad=0.01, w_pad=0.01, hspace=0.02, wspace=0.02)
    grid = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=CONDENSED2_WIDTH_RATIOS,
        wspace=CONDENSED_WSPACE,
    )
    task_ax = fig.add_subplot(grid[0, 0])
    example_axes = [fig.add_subplot(grid[0, column]) for column in range(1, 4)]

    draw_behavior_task_design_panel(task_ax, asset_dir=asset_dir)
    _draw_panel_a_turn_groups(task_ax.child_axes[0])
    for example_number, (example_ax, example) in enumerate(
        zip(example_axes, examples, strict=True),
        start=1,
    ):
        plot_dark_light_example_panel(
            example_ax,
            example,
            example_number=example_number,
        )

    base._raise_text_to_minimum_fontsize(
        fig,
        base.MIN_PUBLICATION_FONTSIZE_PT,
        additional_exempt_text=CONDENSED_TIMELINE_FONT_EXEMPT_TEXT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _add_panel_headers(
        fig,
        (
            (
                task_ax,
                "A",
                CONDENSED2_PANEL_A_TITLE,
                CONDENSED2_PANEL_A_TITLE_FONTSIZE,
            ),
        ),
    )
    _add_panel_b_header(fig, example_axes)
    fig.canvas.draw()
    base.save_figure(fig, output_path, dpi=dpi, pad_inches=0.03)
    plt.close(fig)
    print(f"Saved condensed2 figure to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None):
    """Parse arguments, defaulting to the condensed2 output basename."""
    args = base.parse_arguments(argv)
    if not _argv_has_output_name(argv):
        args.output_name = DEFAULT_OUTPUT_NAME
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run condensed2 figure generation."""
    args = parse_arguments(argv)
    output_path = base.build_output_path(
        args.output_dir,
        args.output_name,
        args.output_format,
    )
    panel_example_cache_dir = (
        args.panel_example_cache_dir
        if args.panel_example_cache_dir is not None
        else args.output_dir / "cache"
    )
    make_condensed2_figure(
        data_root=args.data_root,
        asset_dir=args.asset_dir,
        output_path=output_path,
        light_epoch=args.light_epoch,
        dark_epoch=args.dark_epoch,
        dpi=args.dpi,
        position_bin_count=args.position_bin_count,
        position_offset=args.position_offset,
        speed_threshold_cm_s=args.speed_threshold_cm_s,
        sigma_bins=args.sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=args.refresh_panel_example_cache,
    )


if __name__ == "__main__":
    main()
