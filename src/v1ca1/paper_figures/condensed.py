"""Generate the condensed variant of the paper summary figure."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from v1ca1.paper_figures.figure_1 import (
    get_dark_epoch,
    plot_panel_e_rate_axis,
    plot_position_aligned_raster_axis,
)
from v1ca1.paper_figures import figure_summary as base
from v1ca1.paper_figures import old_fig3


DEFAULT_OUTPUT_NAME = "condensed"
CONDENSED_FIGURE_2_PANEL_A_EXAMPLE_NUMBER = 1
CONDENSED_FIGURE_HEIGHT_MM = 45.0
CONDENSED_WIDTH_RATIOS = (0.30, 0.418, 0.17, 0.112)
CONDENSED_WSPACE = 0.08
CONDENSED_B_COLUMN_X = (0.14, 0.60)
CONDENSED_B_COLUMN_WIDTH = 0.35
CONDENSED_C_COLUMN_X = 0.09
CONDENSED_C_COLUMN_WIDTH = 0.86
CONDENSED_TOP_RASTER_Y = 0.67
CONDENSED_BOTTOM_RASTER_Y = 0.43
CONDENSED_RASTER_HEIGHT = 0.18
CONDENSED_RATE_Y = 0.15
CONDENSED_RATE_HEIGHT = 0.18
CONDENSED_DECODING_CROSS_AXIS_BOUNDS = (0.18, 0.55, 0.78, 0.30)
CONDENSED_DECODING_PLACE_AXIS_BOUNDS = (0.18, CONDENSED_RATE_Y, 0.78, 0.30)
CONDENSED_DECODING_CROSS_YLIM = old_fig3.PANEL_E_NORM_ERROR_YLIM
CONDENSED_DECODING_SHARED_YLABEL_X = -0.16
CONDENSED_DECODING_SHARED_YLABEL_Y = 0.50
CONDENSED_DECODING_CROSS_LABEL_X = 0.08
CONDENSED_DECODING_CROSS_LABEL_Y = 0.96
CONDENSED_DECODING_PLACE_LABEL_X = 0.96
CONDENSED_DECODING_PLACE_LABEL_Y = 0.96
CONDENSED_PANEL_B_COLUMN_LABEL_Y = 0.862
CONDENSED_PANEL_B_COLUMN_LABELS = ("Silent pair", "Active pair")
CONDENSED_PANEL_B_YLABEL_X = -0.26
CONDENSED_PANEL_TRANSITION_ARROW_PAD = 0.002
CONDENSED_PANEL_GROUP_RECT_PAD = 0.006
CONDENSED_PANEL_GROUP_RECT_COLOR = "0.55"
CONDENSED_PANEL_GROUP_RECT_LINEWIDTH = 0.55
CONDENSED_PANEL_GROUP_RECT_BOTTOM_EXTRA = 0.055
CONDENSED_PANEL_GROUP_RECT_TOP_EXTRA = 0.075
CONDENSED_RASTER_PATH_ICON_WIDTH = 0.061
CONDENSED_C_RASTER_PATH_ICON_WIDTH = (
    CONDENSED_RASTER_PATH_ICON_WIDTH
    * CONDENSED_WIDTH_RATIOS[1]
    / CONDENSED_WIDTH_RATIOS[2]
)
CONDENSED_RASTER_PATH_ICON_HEIGHT = 0.13
CONDENSED_RASTER_PATH_ICON_GAP = 0.010
CONDENSED_C_RASTER_PATH_ICON_GAP = (
    CONDENSED_RASTER_PATH_ICON_GAP
    * CONDENSED_WIDTH_RATIOS[1]
    / CONDENSED_WIDTH_RATIOS[2]
)
CONDENSED_SHARED_XLABEL_Y = 0.060
CONDENSED_HEADER_TOP_Y = 1.055
CONDENSED_HEADER_LABEL_X_PAD = 0.02
CONDENSED_HEADER_FONTSIZE = 8.0
CONDENSED_TIMELINE_FONT_EXEMPT_TEXT = ("Run1", "Run2", "Run3", "Sleep")
CONDENSED_PANEL_B_COLUMNS = (
    ("center_to_right", "left_to_center"),
    ("center_to_left", "right_to_center"),
)
CONDENSED_PANEL_C_TRAJECTORIES = CONDENSED_PANEL_B_COLUMNS[1]
CONDENSED_PANEL_D_EPOCH_ORDER = ("dark", "light")
CONDENSED_PANEL_D_CROSS_EPOCH_ORDER = CONDENSED_PANEL_D_EPOCH_ORDER
CONDENSED_TRAJECTORY_LEGEND_LABELS = {
    "center_to_left": "C→L",
    "right_to_center": "R→C",
    "center_to_right": "C→R",
    "left_to_center": "L→C",
}


def _scale_inset_bounds(
    bounds: tuple[float, float, float, float],
    *,
    x_scale: float,
    y_scale: float,
) -> list[float]:
    """Return inset bounds scaled around center and clipped to the parent axis."""
    x0, y0, width, height = bounds
    scaled_width = min(width * x_scale, 1.0)
    scaled_height = min(height * y_scale, 1.0)
    center_x = x0 + width / 2.0
    center_y = y0 + height / 2.0
    scaled_x0 = min(max(center_x - scaled_width / 2.0, 0.0), 1.0 - scaled_width)
    scaled_y0 = min(max(center_y - scaled_height / 2.0, 0.0), 1.0 - scaled_height)
    return [scaled_x0, scaled_y0, scaled_width, scaled_height]


def _draw_condensed_task_panel(ax: Any) -> None:
    """Draw a denser version of the summary task schematic for condensed Panel A."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    trajectory_ax = ax.inset_axes(
        _scale_inset_bounds(
            base.SUMMARY_TASK_TRAJECTORY_BLOCK_BOUNDS,
            x_scale=1.28,
            y_scale=1.22,
        )
    )
    base._draw_summary_trajectory_cycle(trajectory_ax)

    visual_ax = ax.inset_axes(
        _scale_inset_bounds(
            base.SUMMARY_TASK_VISUAL_BLOCK_BOUNDS,
            x_scale=1.10,
            y_scale=1.12,
        )
    )
    base._draw_summary_visual_stimulus_block(
        visual_ax,
        timeline_style="run_sleep_boxes",
    )


def _argv_has_output_name(argv: Sequence[str] | None) -> bool:
    """Return whether the user explicitly supplied an output basename."""
    values = sys.argv[1:] if argv is None else list(argv)
    return any(value == "--output-name" or value.startswith("--output-name=") for value in values)


def _direction_example_y_max(example: dict[str, Any]) -> float:
    """Return a shared y limit for the directional-progression example rates."""
    maxima = [
        float(np.nanmax(rate))
        for _position, rate in example["firing_rates"].values()
        if np.isfinite(rate).any()
    ]
    return 1.0 if not maxima else max(1.0, float(np.ceil(max(maxima))))


def _set_shared_x_label(ax: Any) -> None:
    """Add the shared normalized path-progression x label."""
    ax.text(
        0.5,
        CONDENSED_SHARED_XLABEL_Y,
        "Norm. path progression",
        ha="center",
        va="top",
        fontsize=6.0,
        transform=ax.transAxes,
        clip_on=False,
    )


def _add_condensed_panel_headers(
    fig: Any,
    panel_specs: Sequence[tuple[Any, str, str]],
) -> None:
    """Add top-aligned figure-level headers for condensed panels."""
    for ax, panel_label, title in panel_specs:
        bounds = ax.get_position()
        label_x_pad = (
            0.20 if panel_label == "D" else CONDENSED_HEADER_LABEL_X_PAD
        )
        label_x = max(
            bounds.x0 - label_x_pad * bounds.width,
            0.012,
        )
        title_x = 0.5 * (bounds.x0 + bounds.x1)
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
            title_x,
            CONDENSED_HEADER_TOP_Y,
            title,
            ha="center",
            va="top",
            fontsize=CONDENSED_HEADER_FONTSIZE,
            linespacing=0.95,
        )


def _draw_raster_path_icon(
    ax: Any,
    trajectory_type: str,
    *,
    x: float,
    raster_y: float,
    icon_width: float,
    fill_track: bool,
) -> None:
    """Draw one compact trajectory icon beside a raster row."""
    icon_ax = ax.inset_axes(
        [
            x,
            raster_y + 0.5 * (CONDENSED_RASTER_HEIGHT - CONDENSED_RASTER_PATH_ICON_HEIGHT),
            icon_width,
            CONDENSED_RASTER_PATH_ICON_HEIGHT,
        ]
    )
    base.draw_w_track_schematic(
        icon_ax,
        trajectory_name=trajectory_type,
        arrow_color=base.PANEL_E_TRAJECTORY_COLORS[trajectory_type],
        track_linewidth=0.45,
        trajectory_linewidth=0.65,
        arrow_mutation_scale=5.8,
        fill_track=fill_track,
    )


def _add_panel_transition_arrow(fig: Any, source_ax: Any, target_ax: Any) -> None:
    """Draw a figure-level arrow between two condensed panels."""
    from matplotlib.patches import FancyArrowPatch

    source_bounds = source_ax.get_position()
    target_bounds = target_ax.get_position()
    arrow_y = source_bounds.y0 + (
        CONDENSED_BOTTOM_RASTER_Y + 0.5 * CONDENSED_RASTER_HEIGHT
    ) * source_bounds.height
    arrow_start_x = source_bounds.x0 + (
        CONDENSED_B_COLUMN_X[1]
        + CONDENSED_B_COLUMN_WIDTH
        + CONDENSED_RASTER_PATH_ICON_GAP
    ) * source_bounds.width
    target_icon_left = target_bounds.x0 + (
        CONDENSED_C_COLUMN_X
        - CONDENSED_C_RASTER_PATH_ICON_WIDTH
        - CONDENSED_C_RASTER_PATH_ICON_GAP
    ) * target_bounds.width
    arrow = FancyArrowPatch(
        (
            arrow_start_x,
            arrow_y,
        ),
        (
            target_icon_left - CONDENSED_PANEL_TRANSITION_ARROW_PAD,
            arrow_y,
        ),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=6.5,
        linewidth=0.8,
        color="0.25",
        clip_on=False,
    )
    fig.add_artist(arrow)


def _add_panel_b_c_group_rectangle(fig: Any, source_ax: Any, target_ax: Any) -> None:
    """Outline the panel-B right column and panel-C example block."""
    from matplotlib.patches import Rectangle

    source_bounds = source_ax.get_position()
    target_bounds = target_ax.get_position()
    pad = CONDENSED_PANEL_GROUP_RECT_PAD
    x0 = source_bounds.x0 + (
        CONDENSED_B_COLUMN_X[1]
        - CONDENSED_RASTER_PATH_ICON_WIDTH
        - CONDENSED_RASTER_PATH_ICON_GAP
    ) * source_bounds.width
    x1 = target_bounds.x0 + (
        CONDENSED_C_COLUMN_X + CONDENSED_C_COLUMN_WIDTH
    ) * target_bounds.width
    y0 = source_bounds.y0 + (
        CONDENSED_RATE_Y - CONDENSED_PANEL_GROUP_RECT_BOTTOM_EXTRA
    ) * source_bounds.height
    y1 = source_bounds.y0 + (
        CONDENSED_TOP_RASTER_Y + CONDENSED_RASTER_HEIGHT
        + CONDENSED_PANEL_GROUP_RECT_TOP_EXTRA
    ) * source_bounds.height
    rect = Rectangle(
        (x0 - pad, y0 - pad),
        x1 - x0 + 2.0 * pad,
        y1 - y0 + 2.0 * pad,
        transform=fig.transFigure,
        facecolor="none",
        edgecolor=CONDENSED_PANEL_GROUP_RECT_COLOR,
        linewidth=CONDENSED_PANEL_GROUP_RECT_LINEWIDTH,
        clip_on=False,
        zorder=8,
    )
    fig.add_artist(rect)


def plot_swapped_direction_panel(
    ax: Any,
    example: dict[str, Any],
    *,
    y_max: float,
) -> None:
    """Plot the dark directional-progression example with swapped columns."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    for column_index, trajectory_pair in enumerate(CONDENSED_PANEL_B_COLUMNS):
        x0 = CONDENSED_B_COLUMN_X[column_index]
        ax.text(
            x0 + 0.5 * CONDENSED_B_COLUMN_WIDTH,
            CONDENSED_PANEL_B_COLUMN_LABEL_Y,
            CONDENSED_PANEL_B_COLUMN_LABELS[column_index],
            ha="center",
            va="bottom",
            fontsize=6.0,
            transform=ax.transAxes,
        )
        for trajectory_type, raster_y in (
            (trajectory_pair[0], CONDENSED_TOP_RASTER_Y),
            (trajectory_pair[1], CONDENSED_BOTTOM_RASTER_Y),
        ):
            _draw_raster_path_icon(
                ax,
                trajectory_type,
                x=x0 - CONDENSED_RASTER_PATH_ICON_WIDTH - CONDENSED_RASTER_PATH_ICON_GAP,
                raster_y=raster_y,
                icon_width=CONDENSED_RASTER_PATH_ICON_WIDTH,
                fill_track=True,
            )
            raster_ax = ax.inset_axes(
                [x0, raster_y, CONDENSED_B_COLUMN_WIDTH, CONDENSED_RASTER_HEIGHT]
            )
            plot_position_aligned_raster_axis(
                raster_ax,
                example["raster_positions"][trajectory_type],
                trajectory_type,
                show_ylabel=column_index == 0,
            )
            raster_ax.set_facecolor(base.PANEL_A_DARK_EPOCH_BACKGROUND)
            if column_index == 0:
                raster_ax.yaxis.set_label_coords(CONDENSED_PANEL_B_YLABEL_X, 0.5)
            if column_index != 0:
                raster_ax.set_ylabel("")
        rate_ax = ax.inset_axes(
            [x0, CONDENSED_RATE_Y, CONDENSED_B_COLUMN_WIDTH, CONDENSED_RATE_HEIGHT]
        )
        plot_panel_e_rate_axis(
            rate_ax,
            example["firing_rates"],
            trajectory_pair,
            y_max=y_max,
            show_ylabel=column_index == 0,
            show_legend=True,
        )
        rate_ax.set_facecolor(base.PANEL_A_DARK_EPOCH_BACKGROUND)
        if column_index == 0:
            rate_ax.yaxis.set_label_coords(CONDENSED_PANEL_B_YLABEL_X, 0.5)
        rate_ax.set_xlabel("")
        rate_ax.set_xticklabels(["0", "1"])
        if column_index != 0:
            rate_ax.set_ylabel("")
    _set_shared_x_label(ax)


def _plot_condensed_place_decoding_axis(
    ax: Any,
    decoding_error_table: Any,
    *,
    ylabel: str | None = "Abs. norm. error",
) -> None:
    """Plot condensed place decoding with dark on the left and light on the right."""
    table = decoding_error_table[
        decoding_error_table["analysis"].astype(str) == "place"
    ].copy()
    positions = np.arange(1, len(CONDENSED_PANEL_D_EPOCH_ORDER) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            old_fig3.PANEL_QUANT_EPOCH_LABELS[epoch_type]
            for epoch_type in CONDENSED_PANEL_D_EPOCH_ORDER
        ]
    )
    ax.set_xlim(0.5, len(CONDENSED_PANEL_D_EPOCH_ORDER) + 0.5)
    ax.set_ylim(*old_fig3.PANEL_E_PLACE_ERROR_YLIM)
    if table.empty:
        ax.text(0.5, 0.5, "No place\ndecoding", ha="center", va="center")
        old_fig3._style_panel_e_error_axis(ax, ylabel=ylabel)
        return

    for position, epoch_type in zip(
        positions,
        CONDENSED_PANEL_D_EPOCH_ORDER,
        strict=True,
    ):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if rows.empty:
            continue
        row = rows.iloc[0]
        old_fig3._plot_panel_e_interval_point(
            ax,
            x=float(position),
            q25=float(row["q25_error"]),
            median=float(row["median_error"]),
            q75=float(row["q75_error"]),
            color=old_fig3.PANEL_QUANT_EPOCH_COLORS[epoch_type],
            marker="o",
        )

    old_fig3._style_panel_e_error_axis(ax, ylabel=ylabel)


def _plot_condensed_cross_decoding_axis(
    ax: Any,
    decoding_error_table: Any,
    *,
    ylabel: str | None = "Cross-path decoding error",
) -> None:
    """Plot condensed cross-path decoding with the Panel D epoch order."""
    table = decoding_error_table[
        decoding_error_table["analysis"].astype(str) == "cross_trajectory"
    ].copy()
    comparisons = list(old_fig3.PANEL_E_CROSS_COMPARISONS)
    comparison = comparisons[0][0] if comparisons else None
    positions = np.arange(1, len(CONDENSED_PANEL_D_CROSS_EPOCH_ORDER) + 1, dtype=float)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            old_fig3.PANEL_QUANT_EPOCH_LABELS[epoch_type]
            for epoch_type in CONDENSED_PANEL_D_CROSS_EPOCH_ORDER
        ]
    )
    ax.set_xlim(0.5, len(CONDENSED_PANEL_D_CROSS_EPOCH_ORDER) + 0.5)
    ax.set_ylim(*CONDENSED_DECODING_CROSS_YLIM)
    if table.empty:
        ax.text(0.5, 0.5, "No cross-path\ndecoding", ha="center", va="center")
        old_fig3._style_panel_e_error_axis(ax, ylabel=ylabel)
        return

    for position, epoch_type in zip(
        positions,
        CONDENSED_PANEL_D_CROSS_EPOCH_ORDER,
        strict=True,
    ):
        rows = table[table["epoch_type"].astype(str) == epoch_type]
        if comparison is not None:
            rows = rows[rows["comparison"].astype(str) == comparison]
        if rows.empty:
            continue
        row = rows.iloc[0]
        old_fig3._plot_panel_e_interval_point(
            ax,
            x=float(position),
            q25=float(row["q25_error"]),
            median=float(row["median_error"]),
            q75=float(row["q75_error"]),
            color=old_fig3.PANEL_QUANT_EPOCH_COLORS[epoch_type],
            marker="o",
            size=11,
            linewidth=0.85,
            alpha=0.70,
        )

    old_fig3._set_panel_e_error_ylim(ax, table)
    old_fig3._style_panel_e_error_axis(ax, ylabel=ylabel)


def plot_light_example_column(
    ax: Any,
    example: dict[str, Any],
    *,
    y_max: float,
) -> None:
    """Plot the light epoch of one DPP example in one panel-B-sized column."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    trajectories = base.validate_panel_a_trajectories(example["trajectories"])

    for trajectory_type, raster_y in (
        (trajectories[0], CONDENSED_TOP_RASTER_Y),
        (trajectories[1], CONDENSED_BOTTOM_RASTER_Y),
    ):
        _draw_raster_path_icon(
            ax,
            trajectory_type,
            x=(
                CONDENSED_C_COLUMN_X
                - CONDENSED_C_RASTER_PATH_ICON_WIDTH
                - CONDENSED_C_RASTER_PATH_ICON_GAP
            ),
            raster_y=raster_y,
            icon_width=CONDENSED_C_RASTER_PATH_ICON_WIDTH,
            fill_track=False,
        )
        raster_ax = ax.inset_axes(
            [
                CONDENSED_C_COLUMN_X,
                raster_y,
                CONDENSED_C_COLUMN_WIDTH,
                CONDENSED_RASTER_HEIGHT,
            ]
        )
        base.plot_panel_a_raster_axis(
            raster_ax,
            example,
            "light",
            trajectories=[trajectory_type],
            show_ylabel=False,
            show_title=False,
        )

    rate_ax = ax.inset_axes(
        [
            CONDENSED_C_COLUMN_X,
            CONDENSED_RATE_Y,
            CONDENSED_C_COLUMN_WIDTH,
            CONDENSED_RATE_HEIGHT,
        ]
    )
    base.plot_epoch_path_rate_axis(
        rate_ax,
        example,
        "light",
        trajectories=trajectories,
        y_max=y_max,
        show_ylabel=False,
        show_legend=True,
        show_title=False,
        show_correlation=False,
    )
    rate_ax.set_xlabel("")
    legend = rate_ax.get_legend()
    if legend is not None:
        for trajectory_type, legend_text in zip(trajectories, legend.get_texts(), strict=False):
            legend_text.set_text(CONDENSED_TRAJECTORY_LEGEND_LABELS[trajectory_type])
    _set_shared_x_label(ax)


def plot_decoding_panel(ax: Any, decoding_error_table: Any) -> None:
    """Plot cross-path and place decoding summaries as condensed Panel D."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        CONDENSED_DECODING_SHARED_YLABEL_X,
        CONDENSED_DECODING_SHARED_YLABEL_Y,
        "Norm. decoding error (lower=better)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.0,
        transform=ax.transAxes,
    )

    cross_ax = ax.inset_axes(CONDENSED_DECODING_CROSS_AXIS_BOUNDS)
    _plot_condensed_cross_decoding_axis(
        cross_ax,
        decoding_error_table,
        ylabel=None,
    )
    cross_ax.set_title("")
    cross_ax.text(
        CONDENSED_DECODING_CROSS_LABEL_X,
        CONDENSED_DECODING_CROSS_LABEL_Y,
        "Same turn\ntransfer",
        ha="left",
        va="top",
        fontsize=6.0,
        linespacing=0.85,
        transform=cross_ax.transAxes,
    )
    cross_ax.set_xticklabels([])
    cross_ax.tick_params(labelsize=6.0, length=1.2, pad=0.8)

    decoding_ax = ax.inset_axes(CONDENSED_DECODING_PLACE_AXIS_BOUNDS)
    _plot_condensed_place_decoding_axis(
        decoding_ax,
        decoding_error_table,
        ylabel=None,
    )
    decoding_ax.set_title("")
    decoding_ax.text(
        CONDENSED_DECODING_PLACE_LABEL_X,
        CONDENSED_DECODING_PLACE_LABEL_Y,
        "Place",
        ha="right",
        va="top",
        fontsize=6.0,
        linespacing=0.85,
        transform=decoding_ax.transAxes,
    )
    for text in decoding_ax.texts:
        if text.get_text().strip().startswith(("Light med.", "Dark med.")):
            text.set_text("")
    decoding_ax.tick_params(labelsize=6.0, length=1.2, pad=0.8)


def make_condensed_figure(
    *,
    data_root: Path,
    output_path: Path,
    datasets: Sequence[base.DatasetId],
    regions: Sequence[str],
    light_epoch: str | None,
    dark_epoch: str | None,
    dpi: int,
    position_bin_count: int = base.DEFAULT_POSITION_BIN_COUNT,
    position_offset: int = base.DEFAULT_POSITION_OFFSET,
    speed_threshold_cm_s: float = base.DEFAULT_SPEED_THRESHOLD_CM_S,
    sigma_bins: float = base.DEFAULT_SIGMA_BINS,
    panel_example_cache_dir: Path | None = None,
    refresh_panel_example_cache: bool = False,
) -> Path:
    """Build and save the one-row condensed summary figure."""
    import matplotlib.pyplot as plt

    panel_example_cache_dir = (
        Path(output_path).parent / "cache"
        if panel_example_cache_dir is None
        else Path(panel_example_cache_dir)
    )
    example_number = CONDENSED_FIGURE_2_PANEL_A_EXAMPLE_NUMBER
    animal_name, date, region, unit_id, _figure_2a_trajectories = base.FIGURE_2_PANEL_A_EXAMPLES[
        example_number - 1
    ]
    example = base.load_panel_a_example_data(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        region=region,
        unit_id=unit_id,
        trajectories=CONDENSED_PANEL_C_TRAJECTORIES,
        dark_epoch=dark_epoch,
        light_epoch=light_epoch,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_example_cache_dir=panel_example_cache_dir,
        refresh_panel_example_cache=refresh_panel_example_cache,
    )
    direction_example = base.load_or_compute_panel_e_example_data(
        data_root=data_root,
        animal_name=animal_name,
        date=date,
        epoch=get_dark_epoch(animal_name, date, dark_epoch),
        region=region,
        unit_id=unit_id,
        position_bin_count=position_bin_count,
        position_offset=position_offset,
        speed_threshold_cm_s=speed_threshold_cm_s,
        sigma_bins=sigma_bins,
        panel_e_cache_dir=panel_example_cache_dir,
        refresh_panel_e_cache=refresh_panel_example_cache,
    )
    quant_region = str(regions[0]) if regions else base.DEFAULT_REGIONS[0]
    decoding_error_table = base.load_panel_e_decoding_error_table(
        data_root=data_root,
        datasets=datasets,
        region=quant_region,
        light_epoch=light_epoch,
        dark_epoch=dark_epoch,
    )

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
        width_ratios=CONDENSED_WIDTH_RATIOS,
        wspace=CONDENSED_WSPACE,
    )
    task_ax = fig.add_subplot(grid[0, 0])
    dpp_ax = fig.add_subplot(grid[0, 1])
    example_ax = fig.add_subplot(grid[0, 2])
    decoding_ax = fig.add_subplot(grid[0, 3])

    _draw_condensed_task_panel(task_ax)
    light_y_max = base._example_rate_y_max(
        example,
        base.PANEL_A_EXAMPLE_Y_MAX_OVERRIDES.get(example_number),
    )
    shared_y_max = max(_direction_example_y_max(direction_example), light_y_max)
    plot_swapped_direction_panel(
        dpp_ax,
        direction_example,
        y_max=shared_y_max,
    )
    plot_light_example_column(
        example_ax,
        example,
        y_max=shared_y_max,
    )
    plot_decoding_panel(decoding_ax, decoding_error_table)

    base._raise_text_to_minimum_fontsize(
        fig,
        base.MIN_PUBLICATION_FONTSIZE_PT,
        additional_exempt_text=CONDENSED_TIMELINE_FONT_EXEMPT_TEXT,
    )
    fig.canvas.draw()
    fig.set_layout_engine(None)
    _add_panel_transition_arrow(fig, dpp_ax, example_ax)
    _add_panel_b_c_group_rectangle(fig, dpp_ax, example_ax)
    _add_condensed_panel_headers(
        fig,
        (
            (task_ax, "A", "W-track task with visual landmarks"),
            (
                dpp_ax,
                "B",
                "V1 neurons generalize across\nsame-turn paths in dark",
            ),
            (
                example_ax,
                "C",
                "Light differentiates\nactive pair",
            ),
            (decoding_ax, "D", "Dark generalizes,\nlight separates"),
        ),
    )
    fig.canvas.draw()
    base.save_figure(fig, output_path, dpi=dpi, pad_inches=0.03)
    plt.close(fig)
    print(f"Saved condensed summary figure to {output_path}")
    return output_path


def parse_arguments(argv: Sequence[str] | None = None):
    """Parse arguments, defaulting to the condensed output basename."""
    args = base.parse_arguments(argv)
    if not _argv_has_output_name(argv):
        args.output_name = DEFAULT_OUTPUT_NAME
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run condensed paper-summary figure generation."""
    args = parse_arguments(argv)
    datasets = args.dataset if args.dataset is not None else base.get_processed_datasets()
    regions = tuple(args.region) if args.region is not None else base.DEFAULT_REGIONS
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
    make_condensed_figure(
        data_root=args.data_root,
        output_path=output_path,
        datasets=datasets,
        regions=regions,
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
